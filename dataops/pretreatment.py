"""Dilation, connected fault blocks, and spatial fold assignment."""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine, xy
from scipy import ndimage
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def dilate_labels(raw_labels: np.ndarray, kernel_size: int) -> np.ndarray:
    faults = ndimage.binary_dilation(
        raw_labels == 1,
        structure=np.ones((kernel_size, kernel_size), dtype=bool),
        origin=0,
    )
    buffered = raw_labels.copy()
    buffered[faults & np.isfinite(raw_labels)] = 1
    return buffered


def create_blocks(buffered_labels: np.ndarray) -> tuple[np.ndarray, int]:
    block_ids = np.zeros(buffered_labels.shape, dtype=np.int32)
    ndimage.label(
        buffered_labels == 1,
        structure=np.ones((3, 3), dtype=bool),
        output=block_ids,
    )
    count = int(block_ids.max())
    block_ids = block_ids.astype(np.float32)
    block_ids[block_ids == 0] = np.nan
    return block_ids, count


def create_fold_map(
    features: np.ndarray,
    raw_labels: np.ndarray,
    block_ids: np.ndarray,
    transform: Affine,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(np.isfinite(block_ids))
    ids, inverse = np.unique(block_ids[rows, cols], return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=ids.size))]

    locations = np.empty((ids.size, 2))
    covariates = np.empty((ids.size, features.shape[0]))
    target_means = np.empty((ids.size, 1))
    for index in range(ids.size):
        positions = order[offsets[index] : offsets[index + 1]]
        block_rows, block_cols = rows[positions], cols[positions]
        locations[index] = xy(transform, block_rows.mean(), block_cols.mean())
        covariates[index] = np.nanmean(
            features[:, block_rows, block_cols], axis=1
        )
        observed = raw_labels[block_rows, block_cols]
        observed = observed[np.isfinite(observed)]
        target_means[index, 0] = observed.mean() if observed.size else np.nan

    def scale(values: np.ndarray) -> np.ndarray:
        return StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(values)
        )

    partitions = [
        KMeans(n_clusters=n_folds, n_init=20, random_state=seed).fit_predict(scale(x))
        for x in (locations, covariates, target_means)
    ]
    affinity = np.mean(
        np.stack([part[:, None] == part[None, :] for part in partitions]),
        axis=0,
    )
    np.fill_diagonal(affinity, 1.0)

    folds = SpectralClustering(
        n_clusters=n_folds,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=seed,
    ).fit_predict(affinity)
    fold_map = np.full(block_ids.shape, np.nan, dtype=np.float32)
    fold_map[rows, cols] = folds[inverse]
    return fold_map, folds


def fit_band_scaler(
    features: np.ndarray,
    fold_map: np.ndarray,
    train_fold_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    train_pixels = np.isin(fold_map, train_fold_ids)
    means = np.empty(features.shape[0], dtype=np.float64)
    scales = np.empty(features.shape[0], dtype=np.float64)

    for band_id, band in enumerate(features):
        values = band[train_pixels]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite training pixels for band {band_id}")
        scaler = StandardScaler().fit(values.reshape(-1, 1))
        means[band_id] = scaler.mean_[0]
        scales[band_id] = scaler.scale_[0]

    return means.astype(np.float32), scales.astype(np.float32)
