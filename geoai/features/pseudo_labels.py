"""
Pseudo-label generation for the GeoAI Platform.

Generates binary change mask labels from the composite change score without
requiring human annotation. This implements the rule-based labelling workflow
from Version 1 Notebook 01 (Cells 45–51) that is used to produce the training
labels (``y_v2_clean.npy``) for the Random Forest Enhanced model.

The composite change score is a weighted sum of normalised absolute change
magnitudes for SAR, NDVI, NDBI, and NDWI. Pixels exceeding a threshold of
mean + sigma_multiplier × std are labelled as change (value = 1).

This pseudo-label approach enables training without ground truth annotations
by exploiting the spectral and SAR signatures of anthropogenic change. The
default weights (SAR=0.35, NDVI=0.35, NDBI=0.20, NDWI=0.10) reflect the
relative discriminative power of each feature for man-made change detection.

All weight values and the threshold multiplier are configurable defaults that
must match Version 1 NB01 exactly when using their default values.

Single responsibility: compute pseudo-labels from temporal delta features.

Position in dependency hierarchy: features (depends on core, utils).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from geoai.core.config import PseudoLabelConfig
from geoai.core.exceptions import FeatureEngineeringError
from geoai.utils.constants import (
    DEFAULT_THRESHOLD_SIGMA,
    DEFAULT_WEIGHT_NDBI,
    DEFAULT_WEIGHT_NDVI,
    DEFAULT_WEIGHT_SAR,
    DEFAULT_WEIGHT_NDWI,
)

logger = logging.getLogger(__name__)

# Small constant to prevent division by zero when normalising delta features
# whose maximum absolute value is 0. This can occur in spatially uniform
# imagery patches.
_NORM_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Change score computation
# ---------------------------------------------------------------------------


def compute_normalised_delta(delta: np.ndarray, name: str) -> np.ndarray:
    """Normalise the absolute value of a delta feature to [0, 1].

    Computes: normalised = |delta| / max(|delta|)

    This is the per-feature normalisation step from Version 1 NB01 Cell 45.
    All delta features are normalised to the same scale before weighting so
    that their contributions to the composite score are comparable.

    Args:
        delta: Raw delta feature array of shape (H, W) (T2 - T1 values).
        name: Feature name for logging (e.g. 'delta_sar').

    Returns:
        Float32 array of shape (H, W) with values in [0, 1].
        Returns an array of zeros if ``max(|delta|)`` is effectively zero.
    """
    abs_delta = np.abs(delta.astype(np.float32))
    max_abs = float(abs_delta.max())

    if max_abs < _NORM_EPSILON:
        logger.warning(
            "Normalisation of '%s': max(|delta|) is effectively zero (%.2e). "
            "Returning zeros. This may indicate a spatially uniform input.",
            name,
            max_abs,
        )
        return np.zeros_like(abs_delta, dtype=np.float32)

    normalised = abs_delta / max_abs
    logger.debug(
        "Normalised '%s': max_abs=%.4g result_range=[%.4g, %.4g].",
        name, max_abs, float(normalised.min()), float(normalised.max()),
    )
    return normalised.astype(np.float32)


def compute_change_score(
    delta_sar: np.ndarray,
    delta_ndvi: np.ndarray,
    delta_ndbi: np.ndarray,
    delta_ndwi: np.ndarray,
    weight_sar: float = DEFAULT_WEIGHT_SAR,
    weight_ndvi: float = DEFAULT_WEIGHT_NDVI,
    weight_ndbi: float = DEFAULT_WEIGHT_NDBI,
    weight_ndwi: float = DEFAULT_WEIGHT_NDWI,
) -> np.ndarray:
    """Compute the composite change score for all pixels.

    Implements the weighted composite from Version 1 NB01 Cell 45:

        change_score = (weight_sar  * normalise(|delta_sar|)
                      + weight_ndvi * normalise(|delta_ndvi|)
                      + weight_ndbi * normalise(|delta_ndbi|)
                      + weight_ndwi * normalise(|delta_ndwi|))

    The change score is a proxy for the likelihood of anthropogenic change
    at each pixel. Higher scores indicate stronger multi-source change signals.

    Args:
        delta_sar: SAR temporal difference array (H, W).
        delta_ndvi: NDVI temporal difference array (H, W).
        delta_ndbi: NDBI temporal difference array (H, W).
        delta_ndwi: NDWI temporal difference array (H, W).
        weight_sar: Weight for the SAR change signal. Default 0.35.
        weight_ndvi: Weight for the NDVI change signal. Default 0.35.
        weight_ndbi: Weight for the NDBI change signal. Default 0.20.
        weight_ndwi: Weight for the NDWI change signal. Default 0.10.

    Returns:
        Float32 composite change score array of shape (H, W) with values in
        [0, 1] (approximately, depending on weights and data).
    """
    logger.info(
        "Computing composite change score (weights: SAR=%.2f NDVI=%.2f "
        "NDBI=%.2f NDWI=%.2f).",
        weight_sar, weight_ndvi, weight_ndbi, weight_ndwi,
    )

    sar_norm = compute_normalised_delta(delta_sar, "delta_sar")
    ndvi_norm = compute_normalised_delta(delta_ndvi, "delta_ndvi")
    ndbi_norm = compute_normalised_delta(delta_ndbi, "delta_ndbi")
    ndwi_norm = compute_normalised_delta(delta_ndwi, "delta_ndwi")

    change_score = (
        weight_sar  * sar_norm
        + weight_ndvi * ndvi_norm
        + weight_ndbi * ndbi_norm
        + weight_ndwi * ndwi_norm
    ).astype(np.float32)

    logger.info(
        "Change score computed — range=[%.4g, %.4g] mean=%.4g std=%.4g.",
        float(change_score.min()),
        float(change_score.max()),
        float(change_score.mean()),
        float(change_score.std()),
    )
    return change_score


# ---------------------------------------------------------------------------
# Threshold and pseudo-label generation
# ---------------------------------------------------------------------------


def compute_threshold(
    change_score: np.ndarray,
    sigma_multiplier: float = DEFAULT_THRESHOLD_SIGMA,
) -> float:
    """Compute the adaptive threshold for change score binarisation.

    Implements the threshold from Version 1 NB01 Cell 51:

        threshold = mean(change_score) + sigma_multiplier * std(change_score)

    Args:
        change_score: Composite change score array of shape (H, W) or (N,).
        sigma_multiplier: Multiplier applied to the standard deviation.
            Default 1.5 (from Version 1 NB01). Higher values produce fewer
            but higher-confidence pseudo-labels.

    Returns:
        Scalar threshold value. Pixels with change_score > threshold
        are labelled as change.
    """
    mean_score = float(change_score.mean())
    std_score = float(change_score.std())
    threshold = mean_score + sigma_multiplier * std_score

    logger.info(
        "Threshold computed: mean=%.4g std=%.4g sigma=%.1f → threshold=%.4g.",
        mean_score, std_score, sigma_multiplier, threshold,
    )
    return threshold


def generate_pseudo_labels(
    delta_sar: np.ndarray,
    delta_ndvi: np.ndarray,
    delta_ndbi: np.ndarray,
    delta_ndwi: np.ndarray,
    pseudo_label_config: Optional[PseudoLabelConfig] = None,
    weight_sar: float = DEFAULT_WEIGHT_SAR,
    weight_ndvi: float = DEFAULT_WEIGHT_NDVI,
    weight_ndbi: float = DEFAULT_WEIGHT_NDBI,
    weight_ndwi: float = DEFAULT_WEIGHT_NDWI,
    threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate pseudo-labels from temporal delta features.

    This is the primary entry point for pseudo-label generation. It combines
    :func:`compute_change_score` and :func:`compute_threshold` to produce a
    binary change mask suitable for use as training labels.

    When ``pseudo_label_config`` is provided, its values take precedence over
    the individual weight and threshold arguments.

    The output change mask has:
    - Pixel value 1 where change_score > threshold (change detected).
    - Pixel value 0 where change_score ≤ threshold (no change).

    Source: Version 1 NB01 Cells 45–51.

    Args:
        delta_sar: SAR temporal difference array (H, W).
        delta_ndvi: NDVI temporal difference array (H, W).
        delta_ndbi: NDBI temporal difference array (H, W).
        delta_ndwi: NDWI temporal difference array (H, W).
        pseudo_label_config: Optional configuration object. When provided,
            overrides individual weight and threshold arguments.
        weight_sar: SAR change weight (used if pseudo_label_config is None).
        weight_ndvi: NDVI change weight (used if pseudo_label_config is None).
        weight_ndbi: NDBI change weight (used if pseudo_label_config is None).
        weight_ndwi: NDWI change weight (used if pseudo_label_config is None).
        threshold_sigma: Sigma multiplier (used if pseudo_label_config is None).

    Returns:
        Tuple of three elements:
            - ``change_mask``: uint8 binary array of shape (H, W) with values
              0 (no change) and 1 (change).
            - ``change_score``: float32 composite score array of shape (H, W).
            - ``threshold``: float scalar — the computed binarisation threshold.

    Raises:
        FeatureEngineeringError: If all delta arrays do not share the same shape.
    """
    # Validate shape consistency.
    shapes = {
        "delta_sar": delta_sar.shape,
        "delta_ndvi": delta_ndvi.shape,
        "delta_ndbi": delta_ndbi.shape,
        "delta_ndwi": delta_ndwi.shape,
    }
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise FeatureEngineeringError(
            f"All delta arrays must have the same shape for pseudo-label "
            f"generation. Shapes found: {shapes}."
        )

    # If config is provided, use its values.
    if pseudo_label_config is not None:
        w = pseudo_label_config.weights
        weight_sar = w.sar
        weight_ndvi = w.ndvi
        weight_ndbi = w.ndbi
        weight_ndwi = w.ndwi
        threshold_sigma = pseudo_label_config.threshold_sigma

    logger.info("Generating pseudo-labels for array of shape %s.", delta_sar.shape)

    change_score = compute_change_score(
        delta_sar=delta_sar,
        delta_ndvi=delta_ndvi,
        delta_ndbi=delta_ndbi,
        delta_ndwi=delta_ndwi,
        weight_sar=weight_sar,
        weight_ndvi=weight_ndvi,
        weight_ndbi=weight_ndbi,
        weight_ndwi=weight_ndwi,
    )

    threshold = compute_threshold(change_score, sigma_multiplier=threshold_sigma)
    change_mask = (change_score > threshold).astype(np.uint8)

    n_change = int(change_mask.sum())
    n_total = change_mask.size
    pct_change = 100.0 * n_change / max(n_total, 1)

    logger.info(
        "Pseudo-labels generated — change pixels: %d / %d (%.2f%%).",
        n_change, n_total, pct_change,
    )
    return change_mask, change_score, threshold
