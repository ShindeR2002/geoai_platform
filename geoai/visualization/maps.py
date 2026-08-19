"""
Visualization utilities for the GeoAI Platform.

Generates RGB composite images with prediction overlays, statistical plots,
and object boundary visualisations for inspection and reporting.

All visualisation functions write outputs to disk and return the output path.
No display functions (plt.show()) are called — all output is file-based for
server-side and pipeline use.

Single responsibility: generate visual outputs for change detection results.

Position in dependency hierarchy: visualization (depends on core, utils, analysis).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from geoai.utils.raster_utils import stretch_contrast

logger = logging.getLogger(__name__)

# Matplotlib is imported inside functions to avoid import-time overhead when
# visualization is not needed in a pipeline run.


# ---------------------------------------------------------------------------
# RGB + prediction overlay
# ---------------------------------------------------------------------------


def plot_change_overlay(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    prediction_map: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Change Detection Overlay",
    alpha: float = 0.4,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> Path:
    """Plot a change prediction overlaid on an RGB composite image.

    Applies percentile contrast stretch to the RGB bands (preserving Version 1
    NB01 visualisation behaviour), then overlays change pixels in red.

    Args:
        red: Red band array of shape (H, W).
        green: Green band array of shape (H, W).
        blue: Blue band array of shape (H, W).
        prediction_map: Float32 prediction array of shape (H, W). Change
            pixels have value 1.0.
        output_path: Destination file path (.png recommended).
        title: Plot title string.
        alpha: Opacity of the change overlay. Default 0.4.
        low_pct: Lower percentile for contrast stretch. Default 2.0.
        high_pct: Upper percentile for contrast stretch. Default 98.0.

    Returns:
        Resolved absolute path of the written image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = stretch_contrast(red, low_pct, high_pct)
    g = stretch_contrast(green, low_pct, high_pct)
    b = stretch_contrast(blue, low_pct, high_pct)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    change_mask = (prediction_map == 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Composite (T2)")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    overlay[change_mask] = [1.0, 0.0, 0.0, alpha]
    axes[1].imshow(overlay)
    n_change = int(change_mask.sum())
    axes[1].set_title(f"Change Overlay ({n_change:,} px)")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Change overlay saved: '%s'.", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importance_dict: dict,
    output_path: Union[str, Path],
    title: str = "RF Enhanced — Feature Importances",
    top_n: Optional[int] = None,
) -> Path:
    """Plot a horizontal bar chart of RF feature importances.

    Args:
        importance_dict: Dictionary mapping feature name to importance value,
            as returned by
            :meth:`~geoai.models.production.rf_enhanced.RFEnhancedModel.get_feature_importance_dict`.
        output_path: Destination file path.
        title: Plot title.
        top_n: If specified, show only the top N features. Default: show all.

    Returns:
        Resolved absolute path of the written image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        items = items[:top_n]

    names = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    bars = ax.barh(names[::-1], values[::-1], color="#2196F3")
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance plot saved: '%s'.", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Change score histogram
# ---------------------------------------------------------------------------


def plot_change_score_histogram(
    change_score: np.ndarray,
    threshold: float,
    output_path: Union[str, Path],
    title: str = "Composite Change Score Distribution",
) -> Path:
    """Plot the distribution of composite change scores with the threshold line.

    Args:
        change_score: 2D float32 change score array of shape (H, W).
        threshold: Scalar threshold value. Pixels above this are pseudo-labelled
            as change.
        output_path: Destination file path.
        title: Plot title.

    Returns:
        Resolved absolute path of the written image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = change_score.ravel()
    scores = scores[~np.isnan(scores)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=100, color="#4CAF50", alpha=0.7, label="Change score")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")

    n_change = int((scores > threshold).sum())
    ax.set_xlabel("Composite Change Score")
    ax.set_ylabel("Pixel Count")
    ax.set_title(f"{title} — {n_change:,} px above threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Change score histogram saved: '%s'.", output_path)
    return output_path.resolve()
