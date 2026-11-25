"""Hydra-driven training and evaluation for spatial fault patches."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import rasterio
import torch
from hydra import main as hydra_main
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from rasterio.transform import Affine
from torch import nn
from torch.optim import Optimizer

from dataops.dataset_torch import (
    FaultPatchDataset,
    SpatialFoldSplit,
    make_fault_dataloaders,
)
from src.data import load_data
from src.metrics import DTI
from src.reproducibility import seed_everything


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Reduce an elementwise criterion over known target pixels only."""
    if logits.shape != target.shape:
        raise ValueError(
            f"model output {tuple(logits.shape)} must match target "
            f"{tuple(target.shape)}"
        )
    pixel_losses = criterion(logits, target)
    if pixel_losses.shape != target.shape:
        raise ValueError("criterion must return one loss value per target pixel")

    valid = supervision_mask.bool()
    if not valid.any():
        raise ValueError("batch contains no supervised pixels")
    return pixel_losses.masked_select(valid).mean()


def _stitch_batch(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    patch_origins: torch.Tensor,
    prediction_map: np.ndarray,
    target_map: np.ndarray,
    region_map: np.ndarray,
) -> None:
    """Place supervised patch pixels onto their original raster coordinates."""
    probabilities_array = probabilities.detach().cpu().numpy()[:, 0]
    targets_array = targets.detach().cpu().numpy()[:, 0]
    masks_array = supervision_mask.detach().cpu().numpy()[:, 0].astype(bool)
    origins = patch_origins.detach().cpu().numpy()
    height, width = region_map.shape
    patch_height, patch_width = targets_array.shape[-2:]

    for index, (top, left) in enumerate(origins):
        top, left = int(top), int(left)
        raster_top, raster_left = max(top, 0), max(left, 0)
        raster_bottom = min(top + patch_height, height)
        raster_right = min(left + patch_width, width)
        if raster_top >= raster_bottom or raster_left >= raster_right:
            continue

        patch_top, patch_left = raster_top - top, raster_left - left
        patch_bottom = patch_top + raster_bottom - raster_top
        patch_right = patch_left + raster_right - raster_left
        patch_slice = np.s_[patch_top:patch_bottom, patch_left:patch_right]
        raster_slice = np.s_[raster_top:raster_bottom, raster_left:raster_right]
        valid = masks_array[index][patch_slice]
        region = region_map[raster_slice]
        if np.any(valid & region):
            raise ValueError("supervised patches overlap on the raster")

        probabilities = probabilities_array[index][patch_slice]
        targets = targets_array[index][patch_slice]
        prediction_map[raster_slice][valid] = probabilities[valid]
        target_map[raster_slice][valid] = targets[valid]
        region[valid] = True


def evaluate(
    model: nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    dataset: FaultPatchDataset,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate masked loss and one global DTI over the supervised raster area."""
    model.eval()
    height, width = dataset.raw_labels.shape
    prediction_map = np.zeros((height, width), dtype=np.float32)
    target_map = np.zeros((height, width), dtype=np.float32)
    region_map = np.zeros((height, width), dtype=bool)
    weighted_loss = 0.0
    supervised_pixels = 0

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device)
            mask = batch["supervision_mask"].to(device)
            logits = model(features)
            loss = masked_loss(logits, targets, mask, criterion)
            pixel_count = int(mask.sum().item())
            weighted_loss += float(loss.item()) * pixel_count
            supervised_pixels += pixel_count
            _stitch_batch(
                torch.sigmoid(logits),
                targets,
                mask,
                batch["patch_origin"],
                prediction_map,
                target_map,
                region_map,
            )

    if supervised_pixels == 0:
        raise ValueError("evaluation split contains no supervised pixels")
    return {
        "loss": weighted_loss / supervised_pixels,
        "dti": float(DTI(target_map, region_map)(prediction_map)),
    }


def train_epoch(
    model: nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return loss weighted by supervised pixel count."""
    model.train()
    weighted_loss = 0.0
    supervised_pixels = 0

    for batch in data_loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)
        mask = batch["supervision_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = masked_loss(logits, targets, mask, criterion)
        loss.backward()
        optimizer.step()

        pixel_count = int(mask.sum().item())
        weighted_loss += float(loss.item()) * pixel_count
        supervised_pixels += pixel_count

    if supervised_pixels == 0:
        raise ValueError("training split contains no supervised pixels")
    return weighted_loss / supervised_pixels


def train(
    model: nn.Module,
    loaders: dict[str, Any],
    dataset: FaultPatchDataset,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
    save_every: int = 0,
) -> dict[str, Any]:
    """Fit a model, selecting and restoring the checkpoint with best val DTI."""
    if epochs < 1:
        raise ValueError("training.epochs must be positive")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dti = float("-inf")
    best_epoch = 0
    best_path = checkpoint_dir / "best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        validation = evaluate(
            model, loaders["validation"], dataset, criterion, device
        )
        mlflow.log_metrics(
            {
                "train/loss": training_loss,
                "validation/loss": validation["loss"],
                "validation/dti": validation["dti"],
            },
            step=epoch,
        )

        if validation["dti"] > best_dti:
            best_dti = validation["dti"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_dti": best_dti,
                },
                best_path,
            )
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

        if save_every > 0 and epoch % save_every == 0:
            periodic_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                periodic_path,
            )
            mlflow.log_artifact(str(periodic_path), artifact_path="checkpoints")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "best_dti": best_dti,
        "best_epoch": best_epoch,
        "checkpoint_path": best_path,
    }


def _load_rasters(
    features_path: str,
    labels_path: str,
) -> tuple[np.ndarray, np.ndarray, Affine]:
    features_path = to_absolute_path(features_path)
    labels_path = to_absolute_path(labels_path)
    features = load_data(features_path)
    labels = load_data(labels_path)[0]

    with rasterio.open(features_path) as feature_source, rasterio.open(
        labels_path
    ) as label_source:
        if (
            feature_source.shape != label_source.shape
            or feature_source.transform != label_source.transform
            or feature_source.crs != label_source.crs
        ):
            raise ValueError("feature and label rasters must share a grid")
        transform = feature_source.transform
    return features, labels, transform


@hydra_main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run one configured experiment and evaluate its selected model on test."""
    seed = int(cfg.seed)
    seed_everything(seed, deterministic=bool(cfg.training.deterministic))
    requested_device = str(cfg.training.device)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "auto"
        else requested_device
    )

    features, labels, transform = _load_rasters(
        str(cfg.data.features_path), str(cfg.data.labels_path)
    )
    dataset = instantiate(
        cfg.dataset,
        features=features,
        raw_labels=labels,
        transform=tuple(transform),
    )
    split = SpatialFoldSplit(
        train_folds=tuple(int(fold) for fold in cfg.folds.train),
        validation_fold=int(cfg.folds.validation),
        test_fold=int(cfg.folds.test),
    )
    loaders = make_fault_dataloaders(
        dataset,
        split,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        seed=seed,
    )
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = instantiate(cfg.training.criterion).to(device)

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    with mlflow.start_run(run_name=str(cfg.mlflow.experiment_name)) as run:
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
        mlflow.log_params(
            {
                "seed": seed,
                "model.target": str(cfg.model._target_),
                "training.epochs": int(cfg.training.epochs),
                "training.batch_size": int(cfg.training.batch_size),
                "training.optimizer": str(cfg.training.optimizer._target_),
                "dataset.patch_size": int(cfg.dataset.patch_size),
                "dataset.kernel_size": int(cfg.dataset.kernel_size),
            }
        )
        checkpoint_dir = Path(
            to_absolute_path(str(cfg.training.checkpoint_dir))
        ) / run.info.run_id
        result = train(
            model,
            loaders,
            dataset,
            optimizer,
            criterion,
            epochs=int(cfg.training.epochs),
            device=device,
            checkpoint_dir=checkpoint_dir,
            save_every=int(cfg.training.save_every),
        )
        mlflow.log_metrics(
            {
                "validation/best_dti": result["best_dti"],
                "validation/best_epoch": result["best_epoch"],
            }
        )
        test_metrics = evaluate(
            model, loaders["test"], dataset, criterion, device
        )
        mlflow.log_metrics(
            {
                "test/loss": test_metrics["loss"],
                "test/dti": test_metrics["dti"],
            }
        )
        return {**result, "test": test_metrics}


if __name__ == "__main__":
    main()
