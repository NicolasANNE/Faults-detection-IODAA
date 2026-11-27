"""Hydra-driven training and evaluation for spatial fault patches."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterable
from inspect import getfile
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from typing import Any

import mlflow
import numpy as np
import rasterio
import torch
from hydra import main as hydra_main
from hydra.utils import instantiate, to_absolute_path
from mlflow.data import from_numpy
from mlflow.data.sources import LocalArtifactDatasetSource
from mlflow.models import ModelSignature
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from rasterio.transform import Affine
from torch import nn
from torch.optim import Optimizer
from mlflow.types import Schema, TensorSpec

from dataops.dataset_torch import (
    FaultPatchDataset,
    SpatialFoldSplit,
    make_fault_dataloaders,
)
from dataops.pretreatment import create_spatial_holdout_mask
from src.data import load_data
from src.inference import (
    load_submission_mask,
    predict_full_map,
    save_probability_geotiff,
)
from src.metrics import DTI
from src.reproducibility import seed_everything

_RUN_ANIMALS = ("badger", "fox", "lemur", "monkey", "otter", "panda", "tiger")
_RUN_FRUITS = ("apple", "banana", "kiwi", "mango", "peach", "pear", "plum")


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


def _random_run_name() -> str:
    animal = secrets.choice(_RUN_ANIMALS)
    fruit = secrets.choice(_RUN_FRUITS)
    suffix = secrets.randbelow(10_000_000)
    return f"{animal}-{fruit}-{suffix:07d}"


def _make_mlflow_dataset(
    dataset: FaultPatchDataset,
    cfg: DictConfig,
) -> Any:
    feature_path = Path(to_absolute_path(str(cfg.data.features_path))).resolve()
    label_path = Path(to_absolute_path(str(cfg.data.labels_path))).resolve()
    source_metadata = {
        "features": (
            str(feature_path),
            feature_path.stat().st_size,
            feature_path.stat().st_mtime_ns,
        ),
        "labels": (
            str(label_path),
            label_path.stat().st_size,
            label_path.stat().st_mtime_ns,
        ),
        "kernel_size": int(cfg.dataset.kernel_size),
        "patch_size": int(cfg.dataset.patch_size),
        "train_folds": list(cfg.folds.train),
        "validation_fold": int(cfg.folds.validation),
        "test_fold": int(cfg.folds.test),
        "spatial_holdout": {
            "enabled": bool(cfg.evaluation.spatial_holdout.enabled),
            "grid_rows": int(cfg.evaluation.spatial_holdout.grid_rows),
            "grid_columns": int(cfg.evaluation.spatial_holdout.grid_columns),
            "region_index": int(cfg.evaluation.spatial_holdout.region_index),
        },
    }
    digest = hashlib.sha256(
        json.dumps(source_metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return from_numpy(
        features=dataset.features.detach().cpu().numpy(),
        targets=dataset.raw_labels,
        source=LocalArtifactDatasetSource(uri=feature_path.as_uri()),
        name=f"Fault_KS{int(cfg.dataset.kernel_size)}_Patched",
        digest=digest,
    )


def save_training_curves(
    history: dict[str, list[int | float]],
    output_path: str | Path,
) -> Path:
    """Save train/validation losses and validation DTI by epoch."""
    epochs = history["epoch"]
    if not epochs:
        raise ValueError("training history must contain at least one epoch")
    if any(len(history[key]) != len(epochs) for key in (
        "train/loss",
        "validation/loss",
        "validation/dti",
    )):
        raise ValueError("training metrics must have one value per epoch")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    loss_axis, dti_axis = figure.subplots(1, 2)
    loss_axis.plot(epochs, history["train/loss"], label="Train")
    loss_axis.plot(epochs, history["validation/loss"], label="Validation")
    loss_axis.set_title("Loss by epoch")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    dti_axis.plot(epochs, history["validation/dti"], marker="o")
    dti_axis.set_title("Validation DTI by epoch")
    dti_axis.set_xlabel("Epoch")
    dti_axis.set_ylabel("DTI")
    dti_axis.set_ylim(0.0, 1.0)
    canvas.print_figure(output_path, dpi=180, bbox_inches="tight")
    figure.clear()
    return output_path


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


def evaluate_usgs_dti(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_area: np.ndarray,
    spatial_holdout_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Score the USGS map over all valid pixels, optionally within a holdout tile."""
    if predictions.shape != labels.shape or valid_area.shape != labels.shape:
        raise ValueError("predictions, labels, and valid_area must have one shape")
    if spatial_holdout_mask is not None and spatial_holdout_mask.shape != labels.shape:
        raise ValueError("spatial_holdout_mask must match label shape")

    region = valid_area & np.isfinite(labels)
    if not region.any():
        raise ValueError("USGS DTI region contains no valid labels")
    if not np.isfinite(predictions[region]).all():
        raise FloatingPointError("predictions contain non-finite values in DTI region")
    if not np.isin(labels[region], (0, 1)).all():
        raise ValueError("USGS labels must be binary within the DTI region")

    target = np.nan_to_num(labels, nan=0.0)
    metrics: dict[str, float | int] = {
        "usgs_full_map_dti": float(DTI(target, region)(predictions)),
        "usgs_full_map_pixels": int(region.sum()),
    }
    if spatial_holdout_mask is not None:
        holdout_region = region & spatial_holdout_mask
        if not holdout_region.any():
            raise ValueError("spatial holdout contains no valid USGS labels")
        if not np.any((target == 1) & holdout_region):
            raise ValueError("spatial holdout contains no mapped USGS faults")
        metrics["spatial_holdout_dti"] = float(
            DTI(target, holdout_region)(predictions)
        )
        metrics["spatial_holdout_pixels"] = int(holdout_region.sum())
    return metrics


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
    patience: int | None = None,
) -> dict[str, Any]:
    """Fit a model, selecting and restoring the checkpoint with best val DTI."""
    if epochs < 1:
        raise ValueError("training.epochs must be positive")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive when configured")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dti = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = checkpoint_dir / "best_checkpoint.pt"
    history = {
        "epoch": [],
        "train/loss": [],
        "validation/loss": [],
        "validation/dti": [],
    }

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
        history["epoch"].append(epoch)
        history["train/loss"].append(training_loss)
        history["validation/loss"].append(validation["loss"])
        history["validation/dti"].append(validation["dti"])

        if validation["dti"] > best_dti:
            best_dti = validation["dti"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_dti": best_dti,
                },
                best_path,
            )
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        else:
            epochs_without_improvement += 1

        if save_every > 0 and epoch % save_every == 0:
            periodic_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                periodic_path,
            )
            mlflow.log_artifact(str(periodic_path), artifact_path="checkpoints")

        if patience is not None and epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "best_dti": best_dti,
        "best_epoch": best_epoch,
        "checkpoint_path": best_path,
        "epochs_run": len(history["epoch"]),
        "history": history,
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
    spatial_holdout_mask = None
    if bool(cfg.evaluation.spatial_holdout.enabled):
        spatial_holdout_mask = create_spatial_holdout_mask(
            labels.shape,
            int(cfg.evaluation.spatial_holdout.grid_rows),
            int(cfg.evaluation.spatial_holdout.grid_columns),
            int(cfg.evaluation.spatial_holdout.region_index),
        )
    dataset = instantiate(
        cfg.dataset,
        features=features,
        raw_labels=labels,
        transform=tuple(transform),
        scaler_excluded_region_mask=spatial_holdout_mask,
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
        excluded_region_mask=spatial_holdout_mask,
    )
    model = _instantiate_model(cfg.model).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = instantiate(cfg.training.criterion).to(device)

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    run_name = _random_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
        mlflow.log_artifact(
            str(Path(getfile(type(model))).resolve()),
            artifact_path="model_source",
        )
        mlflow.log_input(_make_mlflow_dataset(dataset, cfg), context="training")
        mlflow.log_params(
            {
                "seed": seed,
                "model.name": type(model).__name__,
                "model.target": str(cfg.model._target_),
                "training.epochs": int(cfg.training.epochs),
                "training.batch_size": int(cfg.training.batch_size),
                "training.optimizer": str(cfg.training.optimizer._target_),
                "training.criterion.bce_weight": float(cfg.training.criterion.bce_weight),
                "data.features_path": str(cfg.data.features_path),
                "data.labels_path": str(cfg.data.labels_path),
                "folds.train": ",".join(
                    str(int(fold)) for fold in cfg.folds.train
                ),
                "folds.validation": int(cfg.folds.validation),
                "folds.test": int(cfg.folds.test),
                "dataset.patch_size": int(cfg.dataset.patch_size),
                "dataset.kernel_size": int(cfg.dataset.kernel_size),
                "evaluation.full_map_dti": bool(cfg.evaluation.full_map_dti),
                "evaluation.spatial_holdout.enabled": bool(
                    cfg.evaluation.spatial_holdout.enabled
                ),
                "evaluation.spatial_holdout.region_index": int(
                    cfg.evaluation.spatial_holdout.region_index
                ),
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
        training_curves_path = save_training_curves(
            result["history"],
            checkpoint_dir / "training_curves.png",
        )
        mlflow.log_artifact(
            str(training_curves_path),
            artifact_path="figures",
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
        result["training_curves_path"] = training_curves_path

        if (
            cfg.inference.enabled
            or cfg.inference.heatmap_enabled
            or bool(cfg.evaluation.full_map_dti)
        ):
            predictions = predict_full_map(
                model,
                dataset.features,
                patch_size=int(dataset.patch_size),
                device=device,
                gaussian_sigma_scale=float(cfg.inference.gaussian_sigma_scale),
            )
            valid_mask = None
            if cfg.inference.enabled or bool(cfg.evaluation.full_map_dti):
                valid_mask = load_submission_mask(
                    to_absolute_path(str(cfg.inference.validity_mask_path)),
                    raster_profile,
                )
            if bool(cfg.evaluation.full_map_dti):
                if valid_mask is None:
                    raise RuntimeError("full-map DTI requires a valid-area mask")
                dti_metrics = evaluate_usgs_dti(
                    predictions,
                    labels,
                    valid_mask,
                    spatial_holdout_mask,
                )
                mlflow.log_metrics(
                    {
                        f"evaluation/{name}": float(value)
                        for name, value in dti_metrics.items()
                    }
                )
                result["evaluation"] = dti_metrics
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
                configured_output_path = Path(
                    to_absolute_path(str(cfg.inference.output_path))
                )
                output_path = Path(
                    configured_output_path.parent
                    / run.info.run_id
                    / configured_output_path.name
                )
                save_probability_geotiff(
                    predictions,
                    raster_profile,
                    output_path,
                    valid_mask,
                )
                mlflow.log_artifact(str(output_path), artifact_path="predictions")
                result["prediction_path"] = output_path

        model_name = type(model).__name__
        model = model.cpu().eval()
        model_input = torch.zeros(
            (
                2,
                dataset.features.shape[0],
                dataset.patch_size,
                dataset.patch_size,
            ),
            dtype=torch.float32,
            device="cpu",
        )
        model_signature = ModelSignature(
            inputs=Schema([
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, dataset.features.shape[0], dataset.patch_size, dataset.patch_size),
                )
            ]),
            outputs=Schema([
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, 1, dataset.patch_size, dataset.patch_size),
                )
            ]),
        )
        mlflow.pytorch.log_model(
            model,
            name=model_name,
            step=int(result["best_epoch"]),
            input_example=model_input,
            signature=model_signature,
        )

        return result


if __name__ == "__main__":
    main()
