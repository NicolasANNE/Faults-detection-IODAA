"""Sliding-window inference and georeferenced probability-map export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch import nn
from torch.nn import functional as F


def _window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _gaussian_window(patch_size: int, sigma_scale: float) -> np.ndarray:
    coordinates = np.arange(patch_size, dtype=np.float32)
    center = (patch_size - 1) / 2
    sigma = patch_size * sigma_scale
    axis = np.exp(-0.5 * ((coordinates - center) / sigma) ** 2)
    window = np.outer(axis, axis).astype(np.float32)
    return np.maximum(window, np.finfo(np.float32).eps)


def predict_full_map(
    model: nn.Module,
    features: np.ndarray | torch.Tensor,
    *,
    patch_size: int,
    device: torch.device,
    gaussian_sigma_scale: float = 0.125,
    batch_size: int = 16,
) -> np.ndarray:
    """Predict probabilities over a raster and blend overlapping patches."""
    feature_map = torch.as_tensor(features, dtype=torch.float32)
    if feature_map.ndim != 3:
        raise ValueError("features must have shape (channels, height, width)")
    if patch_size < 2 or batch_size < 1 or gaussian_sigma_scale <= 0:
        raise ValueError("patch_size, batch_size, and gaussian sigma must be positive")

    _, height, width = feature_map.shape
    stride = patch_size // 2
    origins = [
        (top, left)
        for top in _window_starts(height, patch_size, stride)
        for left in _window_starts(width, patch_size, stride)
    ]
    gaussian = _gaussian_window(patch_size, gaussian_sigma_scale)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros_like(probability_sum)
    was_training = model.training
    model.eval()

    with torch.inference_mode():
        for batch_start in range(0, len(origins), batch_size):
            batch_origins = origins[batch_start : batch_start + batch_size]
            patches = []
            for top, left in batch_origins:
                patch = feature_map[
                    :, top : top + patch_size, left : left + patch_size
                ]
                padding = (
                    0,
                    patch_size - patch.shape[-1],
                    0,
                    patch_size - patch.shape[-2],
                )
                if any(padding):
                    patch = F.pad(patch, padding, mode="replicate")
                patches.append(patch)

            logits = model(torch.stack(patches).to(device))
            if logits.shape != (len(patches), 1, patch_size, patch_size):
                raise ValueError("model must return logits shaped (B, 1, P, P)")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("model returned non-finite logits")
            probabilities = torch.sigmoid(logits[:, 0]).cpu().numpy()

            for probability, (top, left) in zip(probabilities, batch_origins):
                patch_height = min(patch_size, height - top)
                patch_width = min(patch_size, width - left)
                weights = gaussian[:patch_height, :patch_width]
                target_slice = np.s_[
                    top : top + patch_height, left : left + patch_width
                ]
                probability_sum[target_slice] += (
                    probability[:patch_height, :patch_width] * weights
                )
                weight_sum[target_slice] += weights

    if was_training:
        model.train()
    if not np.all(weight_sum > 0):
        raise RuntimeError("sliding windows did not cover the full raster")
    return probability_sum / weight_sum


def save_probability_geotiff(
    probabilities: np.ndarray,
    source_profile: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a one-band probability GeoTIFF using source georeferencing."""
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite two-dimensional map")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(source_profile)
    profile.update(
        driver="GTiff",
        height=probabilities.shape[0],
        width=probabilities.shape[1],
        count=1,
        dtype="float32",
        nodata=None,
        compress="deflate",
    )
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(probabilities.astype(np.float32, copy=False), 1)
    return output_path
