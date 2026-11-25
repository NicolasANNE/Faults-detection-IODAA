"""Visualization helpers for fault-centered PyTorch samples."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dataops.dataset_torch import FaultPatchDataset


def plot_fault_sample(
    dataset: FaultPatchDataset,
    index: int = 0,
    *,
    band_names: Sequence[str] | None = None,
    columns: int = 5,
    quantiles: tuple[float, float] = (2, 98),
) -> tuple[Figure, np.ndarray]:
    """Plot all covariate channels and the masks returned for one sample."""
    low, high = quantiles

    sample = dataset[index]
    features = sample["features"].detach().cpu().numpy()
    feature_validity = sample["feature_validity"].detach().cpu().numpy()
    target = sample["target"][0].detach().cpu().numpy().astype(bool)
    supervision = sample["supervision_mask"][0].detach().cpu().numpy().astype(bool)

    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = []
    for channel, values in enumerate(features):
        name = band_names[channel] if band_names is not None else f"Bande {channel + 1}"
        panels.append((name, values, "viridis", None, None))

    reliable_negatives = supervision & ~target
    panels.extend(
        [
            ("Groupe courant (cible)", target, "Greens", 0, 1),
            ("Zéros fiables supervisés", reliable_negatives, "Blues", 0, 1),
            ("Masque de supervision", supervision, "Oranges", 0, 1),
            (
                "Covariables valides (fraction)",
                feature_validity.mean(axis=0),
                "magma",
                0,
                1,
            ),
        ]
    )

    rows = ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.0 * rows),
        squeeze=False,
    )
    for panel_index, (axis, panel) in enumerate(zip(axes.flat, panels)):
        title, values, cmap, vmin, vmax = panel
        if panel_index < features.shape[0]:
            valid_values = values[feature_validity[panel_index]]
            if valid_values.size:
                vmin, vmax = np.percentile(valid_values, (low, high))
                if vmin == vmax:
                    vmin -= 0.5
                    vmax += 0.5
                masked_values = np.ma.masked_where(
                    ~feature_validity[panel_index], values
                )
                axis.imshow(
                    masked_values,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                title = f"{title}\n[{vmin:.3g}, {vmax:.3g}]"
            else:
                axis.set_facecolor("#bdbdbd")
        else:
            axis.imshow(
                values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()

    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"Groupe {sample['group_id']} — fold {sample['fold_id']}", fontsize=14
    )
    figure.tight_layout()
    return figure, axes


def plot_fault_overview(
    dataset: FaultPatchDataset,
    index: int = 0,
    *,
    stride: int = 4,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Show the selected group on the full fold map, downsampled for display."""
    sample = dataset[index]
    group_id = int(sample["group_id"])
    fold_map = np.asarray(dataset.fold_map, dtype=np.float32)
    fold_map[(~np.isfinite(fold_map)) | (fold_map < 0)] = np.nan
    fold_view = fold_map[::stride, ::stride]

    if ax is None:
        figure, axis = plt.subplots(figsize=(9, 8))
    else:
        axis = ax
        figure = axis.get_figure(root=True)
        assert figure is not None

    palette = plt.get_cmap("tab10")
    fold_colors = ListedColormap(
        [palette(index) for index in range(dataset.n_folds)]
    ).with_extremes(bad="#f2f2f2")
    fold_image = axis.imshow(
        np.ma.masked_invalid(fold_view),
        cmap=fold_colors,
        vmin=-0.5,
        vmax=dataset.n_folds - 0.5,
        interpolation="nearest",
    )
    colorbar = figure.colorbar(
        fold_image, ax=axis, ticks=np.arange(dataset.n_folds)
    )
    colorbar.set_label("Fold")

    selected_group = np.isfinite(dataset.group_ids) & (
        dataset.group_ids == group_id
    )
    selected_view = selected_group[::stride, ::stride]
    axis.imshow(
        np.ma.masked_where(~selected_view, selected_view),
        cmap=ListedColormap(["#ff2d2d"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    rows, cols = np.nonzero(selected_group)
    center_row = (rows.min() + rows.max()) / (2 * stride)
    center_col = (cols.min() + cols.max()) / (2 * stride)
    axis.scatter(center_col, center_row, marker="x", s=70, color="black", linewidth=2)
    axis.legend(
        handles=[
            Patch(facecolor="#ff2d2d", label=f"Groupe {group_id}"),
            Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                linestyle="none",
                label="Centre du groupe",
            ),
        ],
        loc="upper right",
    )
    axis.set_title(f"Emplacement du groupe {group_id} — fold {sample['fold_id']}")
    axis.set_xlabel("Colonne (pixels, décimées)")
    axis.set_ylabel("Ligne (pixels, décimées)")
    figure.tight_layout()
    return figure, axis
