"""Hydra-driven training and evaluation for spatial fault patches."""

from __future__ import annotations

from collections.abc import Iterable
from inspect import getfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import rasterio
import torch
from hydra import main as hydra_main
from hydra.utils import instantiate, to_absolute_path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
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
from src.inference import predict_full_map, save_probability_geotiff
from src.metrics import DTI
from src.reproducibility import seed_everything


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Apply the configured segmentation loss over supervised pixels."""
    if logits.shape != target.shape or target.shape != supervision_mask.shape:
        raise ValueError("logits, target, and supervision mask must have one shape")
    if not supervision_mask.any():
        raise ValueError("batch contains no supervised pixels")
    loss = criterion(logits, target, supervision_mask)
    if loss.ndim != 0:
        raise ValueError("spatial loss must return a scalar")
    if not torch.isfinite(loss):
        raise FloatingPointError("configured loss is not finite")
    return loss


def save_prediction_heatmap(
    probabilities: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a full-raster prediction probability heatmap as a PNG."""
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite two-dimensional map")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12, 10), constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(
        probabilities,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axis, label="Fault probability")
    axis.set_title("Full-map fault probability")
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")
    canvas.print_figure(output_path, dpi=180, bbox_inches="tight")
    figure.clear()
    return output_path


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
    probability_patches = probabilities.detach().cpu().numpy()[:, 0]
    target_patches = targets.detach().cpu().numpy()[:, 0]
    mask_patches = supervision_mask.detach().cpu().numpy()[:, 0].astype(bool)
    origins = patch_origins.detach().cpu().numpy()
    height, width = region_map.shape
    patch_height, patch_width = target_patches.shape[-2:]

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
        valid = mask_patches[index][patch_slice]
        region = region_map[raster_slice]
        if np.any(valid & region):
            raise ValueError("supervised patches overlap on the raster")

        probabilities = probability_patches[index][patch_slice]
        targets = target_patches[index][patch_slice]
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
    """Evaluate the loss and global DTI over the supervised raster area."""
    model.eval()
    height, width = dataset.raw_labels.shape
    prediction_map = np.zeros((height, width), dtype=np.float32)
    target_map = np.zeros((height, width), dtype=np.float32)
    region_map = np.zeros((height, width), dtype=bool)
    weighted_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device)
            mask = batch["supervision_mask"].to(device)
            logits = model(features)
            loss = masked_loss(logits, targets, mask, criterion)
            count = int(features.shape[0])
            weighted_loss += float(loss.item()) * count
            sample_count += count
            _stitch_batch(
                torch.sigmoid(logits),
                targets,
                mask,
                batch["patch_origin"],
                prediction_map,
                target_map,
                region_map,
            )

    if sample_count == 0:
        raise ValueError("evaluation split contains no samples")
    return {
        "loss": weighted_loss / sample_count,
        "dti": float(DTI(target_map, region_map)(prediction_map)),
    }


def train_epoch(
    model: nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train one epoch, averaging the configured loss by sample count."""
    model.train()
    weighted_loss = 0.0
    sample_count = 0

    for batch in data_loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)
        mask = batch["supervision_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = masked_loss(logits, targets, mask, criterion)
        loss.backward()
        optimizer.step()

        count = int(features.shape[0])
        weighted_loss += float(loss.item()) * count
        sample_count += count

    if sample_count == 0:
        raise ValueError("training split contains no samples")
    return weighted_loss / sample_count


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
) -> tuple[np.ndarray, np.ndarray, Affine, dict[str, Any]]:
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
        profile = feature_source.profile.copy()
    return features, labels, transform, profile


def _instantiate_model(cfg: DictConfig) -> nn.Module:
    model_config = OmegaConf.to_container(cfg, resolve=True)
    model_config.pop("hyperparameters")
    return instantiate(model_config)


@hydra_main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run a configured experiment and optionally infer the full raster."""
    seed = int(cfg.seed)
    seed_everything(seed, deterministic=bool(cfg.training.deterministic))
    requested_device = str(cfg.training.device)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "auto"
        else requested_device
    )

    features, labels, transform, raster_profile = _load_rasters(
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
    model = _instantiate_model(cfg.model).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = instantiate(cfg.training.criterion).to(device)

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    with mlflow.start_run(run_name=str(cfg.mlflow.experiment_name)) as run:
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
        mlflow.log_artifact(
            str(Path(getfile(type(model))).resolve()),
            artifact_path="model_source",
        )
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
        result["test"] = test_metrics

        if cfg.inference.enabled or cfg.inference.heatmap_enabled:
            predictions = predict_full_map(
                model,
                dataset.features,
                patch_size=int(dataset.patch_size),
                device=device,
                gaussian_sigma_scale=float(cfg.inference.gaussian_sigma_scale),
            )
            if cfg.inference.heatmap_enabled:
                configured_heatmap_path = Path(
                    to_absolute_path(str(cfg.inference.heatmap_path))
                )
                heatmap_path = (
                    configured_heatmap_path.parent
                    / run.info.run_id
                    / configured_heatmap_path.name
                )
                save_prediction_heatmap(predictions, heatmap_path)
                mlflow.log_artifact(str(heatmap_path), artifact_path="figures")
                result["heatmap_path"] = heatmap_path

            if cfg.inference.enabled:
                output_path = Path(
                    to_absolute_path(str(cfg.inference.output_path))
                )
                save_probability_geotiff(predictions, raster_profile, output_path)
                mlflow.log_artifact(str(output_path), artifact_path="predictions")
                result["prediction_path"] = output_path

        return result


if __name__ == "__main__":
    main()
