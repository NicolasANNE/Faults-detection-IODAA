"""Optuna search with validation-only selection and one-shot final evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import sys
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import mlflow
import numpy as np
import optuna
import torch
from hydra import main as hydra_main
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from rasterio.transform import Affine

from dataops.dataset_torch import (
    FaultPatchDataset,
    SpatialFoldSplit,
    make_fault_dataloaders,
    make_final_dataloaders,
)
from dataops.pretreatment import create_spatial_holdout_mask
from src.inference import (
    load_submission_mask,
    predict_full_map,
    save_probability_geotiff,
)
from src.reproducibility import seed_everything
from src.trainer import (
    _instantiate_model,
    _load_rasters,
    evaluate,
    evaluate_usgs_dti,
    save_prediction_heatmap,
    train,
    train_epoch,
)


def _sample_trial_parameters(
    trial: optuna.Trial,
    cfg: DictConfig,
) -> dict[str, Any]:
    search = cfg.tuning.search_space
    learning_rate = tuple(float(value) for value in search.learning_rate)
    weight_decay = tuple(float(value) for value in search.weight_decay)
    dropout = tuple(float(value) for value in search.dropout)
    bce_weight = tuple(float(value) for value in search.bce_weight)
    return {
        "model.hyperparameters.base_channels": trial.suggest_categorical(
            "model.hyperparameters.base_channels",
            [int(value) for value in search.base_channels],
        ),
        "model.hyperparameters.dropout": trial.suggest_float(
            "model.hyperparameters.dropout", dropout[0], dropout[1]
        ),
        "model.hyperparameters.activation": trial.suggest_categorical(
            "model.hyperparameters.activation",
            [str(value) for value in search.activation],
        ),
        "training.optimizer.lr": trial.suggest_float(
            "training.optimizer.lr",
            learning_rate[0],
            learning_rate[1],
            log=True,
        ),
        "training.optimizer.weight_decay": trial.suggest_float(
            "training.optimizer.weight_decay",
            weight_decay[0],
            weight_decay[1],
            log=True,
        ),
        "training.batch_size": trial.suggest_categorical(
            "training.batch_size",
            [int(value) for value in search.batch_size],
        ),
        "training.criterion.bce_weight": trial.suggest_float(
            "training.criterion.bce_weight",
            bce_weight[0],
            bce_weight[1],
            step=0.1,
        ),
    }


def _apply_trial_parameters(
    cfg: DictConfig,
    parameters: dict[str, Any],
) -> DictConfig:
    tuned_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in parameters.items():
        OmegaConf.update(tuned_cfg, key, value, merge=True)
    return tuned_cfg


def _ensure_final_evaluation_available(study: optuna.Study) -> None:
    if study.user_attrs.get("final_evaluation_started", False):
        raise RuntimeError(
            "This Optuna study has already entered final test/holdout "
            "evaluation; use a new study name only for a genuinely new "
            "evaluation protocol."
        )


def _trial_objective(
    trial: optuna.Trial,
    cfg: DictConfig,
    dataset: FaultPatchDataset,
    split: SpatialFoldSplit,
    holdout_mask: np.ndarray,
    device: torch.device,
    checkpoint_root: Path,
) -> float:
    seed = int(cfg.seed)
    seed_everything(seed, deterministic=bool(cfg.training.deterministic))
    trial_cfg = _apply_trial_parameters(
        cfg, _sample_trial_parameters(trial, cfg)
    )
    loaders = make_fault_dataloaders(
        dataset,
        split,
        batch_size=int(trial_cfg.training.batch_size),
        num_workers=int(trial_cfg.training.num_workers),
        seed=seed,
        excluded_region_mask=holdout_mask,
        include_test=False,
    )
    model = _instantiate_model(trial_cfg.model).to(device)
    optimizer = instantiate(
        trial_cfg.training.optimizer,
        params=model.parameters(),
    )
    criterion = instantiate(trial_cfg.training.criterion).to(device)
    checkpoint_dir = checkpoint_root / f"trial_{trial.number:04d}"

    with mlflow.start_run(
        run_name=f"optuna-trial-{trial.number:04d}",
        nested=True,
    ):
        mlflow.log_params(
            {
                key: value
                for key, value in trial_cfg.training.optimizer.items()
                if key != "_target_"
            }
        )
        mlflow.log_params(
            {
                "model.base_channels": int(trial_cfg.model.base_channels),
                "model.dropout": float(trial_cfg.model.dropout),
                "model.activation": str(trial_cfg.model.activation),
                "training.batch_size": int(trial_cfg.training.batch_size),
                "training.criterion.bce_weight": float(
                    trial_cfg.training.criterion.bce_weight
                ),
                "training.criterion.dti_weight": float(
                    criterion.dti_weight
                ),
                "folds.train": ",".join(
                    str(fold) for fold in split.train_folds
                ),
                "folds.validation": split.validation_fold,
                "max_epochs": int(cfg.tuning.max_epochs),
                "patience": int(cfg.tuning.patience),
            }
        )
        result = train(
            model,
            loaders,
            dataset,
            optimizer,
            criterion,
            epochs=int(cfg.tuning.max_epochs),
            device=device,
            checkpoint_dir=checkpoint_dir,
            patience=int(cfg.tuning.patience),
        )
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        trial.set_user_attr("epochs_run", int(result["epochs_run"]))
        mlflow.log_metrics(
            {
                "optuna/best_validation_dti": float(result["best_dti"]),
                "optuna/best_epoch": int(result["best_epoch"]),
                "optuna/epochs_run": int(result["epochs_run"]),
            }
        )
    return float(result["best_dti"])


def _refit_and_evaluate(
    cfg: DictConfig,
    parameters: dict[str, Any],
    best_epoch: int,
    features: np.ndarray,
    labels: np.ndarray,
    transform: Affine,
    raster_profile: dict[str, Any],
    holdout_mask: np.ndarray,
    device: torch.device,
    run_id: str,
    best_validation_dti: float,
    study: optuna.Study,
) -> dict[str, Any]:
    final_cfg = _apply_trial_parameters(cfg, parameters)
    final_train_folds = (
        tuple(int(value) for value in cfg.folds.train)
        + (int(cfg.folds.validation),)
    )
    final_dataset = instantiate(
        final_cfg.dataset,
        features=features,
        raw_labels=labels,
        transform=tuple(transform),
        train_fold_ids=final_train_folds,
        scaler_excluded_region_mask=holdout_mask,
    )
    final_loaders = make_final_dataloaders(
        final_dataset,
        train_folds=final_train_folds,
        test_fold=int(cfg.folds.test),
        batch_size=int(final_cfg.training.batch_size),
        num_workers=int(final_cfg.training.num_workers),
        seed=int(cfg.seed),
        excluded_region_mask=holdout_mask,
    )
    seed_everything(
        int(cfg.seed),
        deterministic=bool(cfg.training.deterministic),
    )
    model = _instantiate_model(final_cfg.model).to(device)
    optimizer = instantiate(
        final_cfg.training.optimizer,
        params=model.parameters(),
    )
    criterion = instantiate(final_cfg.training.criterion).to(device)

    checkpoint_dir = (
        Path(to_absolute_path(str(cfg.training.checkpoint_dir))) / run_id
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_losses = []
    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(
            model,
            final_loaders["train"],
            optimizer,
            criterion,
            device,
        )
        training_losses.append(loss)
        mlflow.log_metric("final_train/loss", loss, step=epoch)

    checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": best_epoch,
            "best_validation_dti": best_validation_dti,
            "trial_parameters": parameters,
        },
        checkpoint_path,
    )
    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

    study.set_user_attr("final_evaluation_started", True)
    test_metrics = evaluate(
        model,
        final_loaders["test"],
        final_dataset,
        criterion,
        device,
    )
    mlflow.log_metrics(
        {
            "test/loss": test_metrics["loss"],
            "test/dti": test_metrics["dti"],
        }
    )

    predictions = predict_full_map(
        model,
        final_dataset.features,
        patch_size=int(final_dataset.patch_size),
        device=device,
        gaussian_sigma_scale=float(cfg.inference.gaussian_sigma_scale),
    )
    valid_mask = load_submission_mask(
        to_absolute_path(str(cfg.inference.validity_mask_path)),
        raster_profile,
    )
    evaluation_metrics = evaluate_usgs_dti(
        predictions,
        labels,
        valid_mask,
        spatial_holdout_mask=holdout_mask,
    )
    mlflow.log_metrics(
        {
            f"evaluation/{name}": float(value)
            for name, value in evaluation_metrics.items()
        }
    )
    study.set_user_attr(
        "final_evaluation",
        {
            "best_epoch": best_epoch,
            "test": test_metrics,
            "evaluation": evaluation_metrics,
        },
    )
    study.set_user_attr("final_evaluation_completed", True)

    if bool(cfg.inference.heatmap_enabled):
        heatmap_path = (
            Path(to_absolute_path(str(cfg.inference.heatmap_path))).parent
            / run_id
            / Path(str(cfg.inference.heatmap_path)).name
        )
        save_prediction_heatmap(predictions, heatmap_path)
        mlflow.log_artifact(str(heatmap_path), artifact_path="figures")
    else:
        heatmap_path = None

    if bool(cfg.inference.enabled):
        prediction_path = (
            Path(to_absolute_path(str(cfg.inference.output_path))).parent
            / run_id
            / Path(str(cfg.inference.output_path)).name
        )
        save_probability_geotiff(
            predictions,
            raster_profile,
            prediction_path,
            valid_mask,
        )
        mlflow.log_artifact(
            str(prediction_path),
            artifact_path="predictions",
        )
    else:
        prediction_path = None

    return {
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "training_losses": training_losses,
        "test": test_metrics,
        "evaluation": evaluation_metrics,
        "heatmap_path": heatmap_path,
        "prediction_path": prediction_path,
    }


@hydra_main(version_base="1.3", config_path="../configs", config_name="tune")
def main(cfg: DictConfig) -> dict[str, Any]:
    """Tune on validation DTI, then evaluate test and holdout exactly once."""
    n_trials = int(cfg.tuning.n_trials)
    max_epochs = int(cfg.tuning.max_epochs)
    patience = int(cfg.tuning.patience)
    if n_trials < 1 or max_epochs < 1 or patience < 1:
        raise ValueError(
            "tuning.n_trials, tuning.max_epochs, and tuning.patience "
            "must be positive"
        )
    if not bool(cfg.evaluation.spatial_holdout.enabled):
        raise ValueError(
            "Optuna tuning requires the spatial holdout to be enabled"
        )

    seed = int(cfg.seed)
    seed_everything(seed, deterministic=bool(cfg.training.deterministic))
    requested_device = str(cfg.training.device)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "auto"
        else requested_device
    )
    features, labels, transform, raster_profile = _load_rasters(
        str(cfg.data.features_path),
        str(cfg.data.labels_path),
    )
    holdout_cfg = cfg.evaluation.spatial_holdout
    holdout_mask = create_spatial_holdout_mask(
        labels.shape,
        int(holdout_cfg.grid_rows),
        int(holdout_cfg.grid_columns),
        int(holdout_cfg.region_index),
    )
    dataset = instantiate(
        cfg.dataset,
        features=features,
        raw_labels=labels,
        transform=tuple(transform),
        scaler_excluded_region_mask=holdout_mask,
    )
    split = SpatialFoldSplit(
        train_folds=tuple(int(value) for value in cfg.folds.train),
        validation_fold=int(cfg.folds.validation),
        test_fold=int(cfg.folds.test),
    )

    study_name = str(cfg.tuning.study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=str(cfg.tuning.storage),
        direction="maximize",
        load_if_exists=bool(cfg.tuning.load_if_exists),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    _ensure_final_evaluation_available(study)
    study_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", study_name).strip("_")
    checkpoint_root = (
        Path(to_absolute_path(str(cfg.training.checkpoint_dir)))
        / "optuna"
        / study_slug
    )

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    with mlflow.start_run(run_name=f"optuna-{study_slug}") as run:
        mlflow.log_text(OmegaConf.to_yaml(cfg), "tuning_config.yaml")
        mlflow.log_params(
            {
                "tuning.study_name": study_name,
                "tuning.n_trials": n_trials,
                "tuning.max_epochs": max_epochs,
                "tuning.patience": patience,
                "tuning.objective": "validation/dti",
                "folds.train": ",".join(
                    str(value) for value in split.train_folds
                ),
                "folds.validation": split.validation_fold,
                "folds.test": split.test_fold,
            }
        )
        remaining_trials = max(0, n_trials - len(study.trials))
        if remaining_trials:
            study.optimize(
                lambda trial: _trial_objective(
                    trial,
                    cfg,
                    dataset,
                    split,
                    holdout_mask,
                    device,
                    checkpoint_root,
                ),
                n_trials=remaining_trials,
            )
        if not any(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ):
            raise RuntimeError("Optuna study has no completed trials")

        best_trial = study.best_trial
        best_epoch = int(best_trial.user_attrs["best_epoch"])
        mlflow.log_metrics(
            {
                "optuna/best_validation_dti": float(best_trial.value),
                "optuna/best_epoch": best_epoch,
                "optuna/completed_trials": sum(
                    trial.state == optuna.trial.TrialState.COMPLETE
                    for trial in study.trials
                ),
            }
        )
        best_cfg = _apply_trial_parameters(cfg, best_trial.params)
        mlflow.log_text(
            OmegaConf.to_yaml(best_cfg, resolve=True),
            "best_config.yaml",
        )
        final_result = _refit_and_evaluate(
            best_cfg,
            best_trial.params,
            best_epoch,
            features,
            labels,
            transform,
            raster_profile,
            holdout_mask,
            device,
            run.info.run_id,
            float(best_trial.value),
            study,
        )

    return {
        "study_name": study_name,
        "best_trial": best_trial.number,
        "best_validation_dti": float(best_trial.value),
        "best_parameters": best_trial.params,
        **final_result,
    }


if __name__ == "__main__":
    main()
