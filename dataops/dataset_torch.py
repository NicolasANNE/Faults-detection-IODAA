"""PyTorch data utilities for fault-centered spatial experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from rasterio.transform import Affine
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dataops.pretreatment import (
    create_blocks,
    create_fold_map,
    dilate_labels,
    fit_band_scaler,
)
from src.reproducibility import seed_worker


class FaultPatchSample(TypedDict):
    features: torch.Tensor
    feature_validity: torch.Tensor
    target: torch.Tensor
    supervision_mask: torch.Tensor
    group_id: int
    fold_id: int


@dataclass(frozen=True)
class SpatialFoldSplit:
    """Fold roles for one spatial train/validation/test rotation."""

    train_folds: tuple[int, ...]
    validation_fold: int
    test_fold: int


@dataclass(frozen=True)
class _FaultGroup:
    group_id: int
    fold_id: int
    center_row: int
    center_col: int


def spatial_fold_splits(
    test_fold: int,
    n_folds: int = 4,
) -> tuple[SpatialFoldSplit, ...]:
    """Return one 2-train/1-validation split for each non-test fold."""
    if n_folds < 3 or not 0 <= test_fold < n_folds:
        raise ValueError("test_fold must identify a fold in a 3+ fold split")

    remaining = tuple(fold for fold in range(n_folds) if fold != test_fold)
    return tuple(
        SpatialFoldSplit(
            train_folds=tuple(fold for fold in remaining if fold != validation_fold),
            validation_fold=validation_fold,
            test_fold=test_fold,
        )
        for validation_fold in remaining
    )


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_map_patch(
    array: np.ndarray,
    top: int,
    left: int,
    size: int,
    fill_value: float,
) -> np.ndarray:
    """Extract a square 2D patch, filling positions outside the raster."""
    height, width = array.shape
    bottom = top + size
    right = left + size
    source_top = max(0, top)
    source_left = max(0, left)
    source_bottom = min(height, bottom)
    source_right = min(width, right)

    patch_dtype = np.result_type(array.dtype, np.asarray(fill_value).dtype)
    patch = np.full((size, size), fill_value, dtype=patch_dtype)
    if source_top < source_bottom and source_left < source_right:
        target_top = source_top - top
        target_left = source_left - left
        patch[
            target_top : target_top + source_bottom - source_top,
            target_left : target_left + source_right - source_left,
        ] = array[source_top:source_bottom, source_left:source_right]
    return patch


class FaultPatchDataset(Dataset[FaultPatchSample]):
    """One fixed-size raster patch per fault group.

    `raw_labels` is `Y_RAW`; `kernel_size` creates `Y_buff`, then 8-connected
    `Y_blocks` and the notebook's K-Means consensus `fold_map_consensus`.
    `transform` is the raster affine transform used for component centroids.
    The target marks only original `Y_RAW == 1` pixels; supervision includes
    known original positives and zeros inside the current buffered group.

    Band statistics are fitted on finite pixels from `train_fold_ids` only.
    Non-finite covariates and out-of-raster padding are flagged in
    `feature_validity`; normalized missing values are filled with zero.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        raw_labels: np.ndarray | torch.Tensor,
        *,
        transform: Affine | Sequence[float],
        patch_size: int,
        train_fold_ids: Sequence[int],
        kernel_size: int = 10,
        n_folds: int = 4,
        random_state: int = 0,
    ) -> None:
        if patch_size <= 0 or kernel_size <= 0 or n_folds < 3:
            raise ValueError("patch_size/kernel_size must be positive and n_folds >= 3")

        feature_array = _as_numpy(features)
        self.raw_labels = _as_numpy(raw_labels).astype(np.float32, copy=False)
        if (
            feature_array.ndim != 3
            or self.raw_labels.shape != tuple(feature_array.shape[-2:])
        ):
            raise ValueError(
                "features must be (channels, height, width) and labels (height, width)"
            )

        known_labels = self.raw_labels[np.isfinite(self.raw_labels)]
        if np.isinf(self.raw_labels).any() or not np.isin(known_labels, (0, 1)).all():
            raise ValueError("raw_labels values must be 0, 1, or NaN")
        self.kernel_size = int(kernel_size)
        self.transform = (
            transform if isinstance(transform, Affine) else Affine(*transform)
        )
        self.random_state = int(random_state)
        self.buffered_labels = dilate_labels(self.raw_labels, self.kernel_size)
        self.group_ids, self.block_count = create_blocks(self.buffered_labels)
        self.patch_size = int(patch_size)
        self.n_folds = int(n_folds)
        if self.block_count < self.n_folds:
            raise ValueError(
                f"{self.block_count} groups remain after dilation with "
                f"kernel_size={self.kernel_size}; at least {self.n_folds} are "
                "required, so use a smaller kernel_size"
            )
        self.fold_map, self.fold_labels = create_fold_map(
            feature_array,
            self.raw_labels,
            self.group_ids,
            self.transform,
            self.n_folds,
            self.random_state,
        )
        self.train_fold_ids = tuple(train_fold_ids)
        self.feature_mean, self.feature_scale = fit_band_scaler(
            feature_array, self.fold_map, self.train_fold_ids
        )
        normalized_features = (
            feature_array - self.feature_mean[:, None, None]
        ) / self.feature_scale[:, None, None]
        normalized_features = np.nan_to_num(
            normalized_features, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        self.features = torch.from_numpy(normalized_features)
        self.feature_validity = torch.from_numpy(np.isfinite(feature_array))
        self.Y_buff = self.buffered_labels
        self.Y_blocks = self.group_ids
        self.fold_map_consensus = self.fold_map
        self._groups = self._index_groups()

    def _index_groups(self) -> list[_FaultGroup]:
        rows, cols = np.nonzero(np.isfinite(self.group_ids))
        unique_ids, inverse = np.unique(
            self.group_ids[rows, cols], return_inverse=True
        )
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=unique_ids.size)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        fold_values = self.fold_map[rows, cols]
        groups: list[_FaultGroup] = []

        for index, group_id in enumerate(unique_ids):
            positions = order[offsets[index] : offsets[index + 1]]
            group_rows = rows[positions]
            group_cols = cols[positions]
            groups.append(
                _FaultGroup(
                    int(group_id),
                    int(fold_values[positions[0]]),
                    int((group_rows.min() + group_rows.max()) // 2),
                    int((group_cols.min() + group_cols.max()) // 2),
                )
            )
        return groups

    def __len__(self) -> int:
        return len(self._groups)

    def indices_for_folds(self, folds: Sequence[int]) -> list[int]:
        """Return sample indices assigned to any of the requested folds."""
        selected = set(folds)
        return [
            index
            for index, group in enumerate(self._groups)
            if group.fold_id in selected
        ]

    def __getitem__(self, index: int) -> FaultPatchSample:
        group = self._groups[index]
        half = self.patch_size // 2
        top = group.center_row - half
        left = group.center_col - half
        bottom = top + self.patch_size
        right = left + self.patch_size

        height, width = self.features.shape[-2:]
        source_top = max(0, top)
        source_left = max(0, left)
        source_bottom = min(height, bottom)
        source_right = min(width, right)
        feature_patch = self.features[
            :, source_top:source_bottom, source_left:source_right
        ]
        feature_validity = self.feature_validity[
            :, source_top:source_bottom, source_left:source_right
        ]
        padding = (
            source_left - left,
            right - source_right,
            source_top - top,
            bottom - source_bottom,
        )
        if any(padding):
            feature_patch = F.pad(feature_patch, padding, mode="replicate")
            feature_validity = F.pad(
                feature_validity, padding, mode="constant", value=False
            )
        spatial_validity = torch.zeros(
            (self.patch_size, self.patch_size), dtype=torch.bool
        )
        spatial_top = source_top - top
        spatial_left = source_left - left
        spatial_validity[
            spatial_top : spatial_top + source_bottom - source_top,
            spatial_left : spatial_left + source_right - source_left,
        ] = True
        feature_validity &= spatial_validity.unsqueeze(0)

        group_patch = _extract_map_patch(
            self.group_ids, top, left, self.patch_size, fill_value=0
        )
        raw_patch = _extract_map_patch(
            self.raw_labels, top, left, self.patch_size, fill_value=np.nan
        )

        current_group = group_patch == group.group_id
        target = current_group & (raw_patch == 1)
        reliable_negative = current_group & (raw_patch == 0)
        supervision_mask = target | reliable_negative

        return {
            "features": feature_patch,
            "feature_validity": feature_validity,
            "target": torch.from_numpy(
                target.astype(np.float32, copy=False)
            ).unsqueeze(0),
            "supervision_mask": torch.from_numpy(supervision_mask).unsqueeze(0),
            "group_id": group.group_id,
            "fold_id": group.fold_id,
        }


def make_fault_dataloaders(
    dataset: FaultPatchDataset,
    split: SpatialFoldSplit,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """Build train, validation, and test loaders for one spatial split."""
    if set(dataset.train_fold_ids) != set(split.train_folds):
        raise ValueError("Dataset scaler and training loader must use the same folds")

    all_folds = (*split.train_folds, split.validation_fold, split.test_fold)
    if (
        len(split.train_folds) != dataset.n_folds - 2
        or set(all_folds) != set(range(dataset.n_folds))
    ):
        raise ValueError("split must assign every fold exactly once to train/val/test")

    fold_sets = {
        "train": split.train_folds,
        "validation": (split.validation_fold,),
        "test": (split.test_fold,),
    }
    loaders: dict[str, DataLoader] = {}
    for offset, (name, folds) in enumerate(fold_sets.items()):
        indices = dataset.indices_for_folds(folds)
        generator = torch.Generator().manual_seed(seed + offset)
        loaders[name] = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders
