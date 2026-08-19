"""
18-feature cube assembly for the GeoAI Platform.

Assembles all per-epoch EO/SAR arrays and temporal delta features into the
canonical 18-band feature cube used for model training and inference.

The feature order is a hard scientific constraint frozen by the production
model ``rf_enhanced.pkl``. Any deviation from this order will cause the model
to produce incorrect predictions without raising an error. The order is
enforced by this module and validated against the configuration.

Feature cube structure (shape: H × W × 18):
  Indices  0– 6: Group A — Epoch T1 features (Red, Green, Blue, SAR, NDVI, NDBI, NDWI)
  Indices  7–13: Group B — Epoch T2 features (Red, Green, Blue, SAR, NDVI, NDBI, NDWI)
  Indices 14–17: Group C — Temporal deltas (Delta_NDVI, Delta_NDBI, Delta_NDWI, Delta_SAR)

Source: Version 1 NB05 Cell 10, NB09 Cell 10.

Single responsibility: assemble the canonical 18-feature cube from its
constituent arrays.

Position in dependency hierarchy: features (depends on core, utils, features/temporal).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from geoai.core.exceptions import FeatureCountMismatchError, FeatureEngineeringError
from geoai.features.temporal import compute_all_deltas
from geoai.utils.constants import CANONICAL_FEATURE_COUNT, CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary feature cube builder
# ---------------------------------------------------------------------------


def build_feature_cube(
    # Epoch T1 arrays
    red_t1: np.ndarray,
    green_t1: np.ndarray,
    blue_t1: np.ndarray,
    sar_t1: np.ndarray,
    ndvi_t1: np.ndarray,
    ndbi_t1: np.ndarray,
    ndwi_t1: np.ndarray,
    # Epoch T2 arrays
    red_t2: np.ndarray,
    green_t2: np.ndarray,
    blue_t2: np.ndarray,
    sar_t2: np.ndarray,
    ndvi_t2: np.ndarray,
    ndbi_t2: np.ndarray,
    ndwi_t2: np.ndarray,
) -> np.ndarray:
    """Assemble the canonical 18-feature cube from T1 and T2 epoch arrays.

    Computes the four temporal delta features internally and stacks all 18
    features in the canonical order using ``np.stack([...], axis=-1)``.

    This is the single authoritative implementation of feature cube assembly.
    All pipeline stages that require a feature cube must call this function.

    Args:
        red_t1: Red band, epoch T1. float32 array of shape (H, W).
        green_t1: Green band, epoch T1. float32 array of shape (H, W).
        blue_t1: Blue band, epoch T1. float32 array of shape (H, W).
        sar_t1: SAR VV band, epoch T1. float32 array of shape (H, W).
        ndvi_t1: NDVI, epoch T1. float32 array of shape (H, W).
        ndbi_t1: NDBI, epoch T1. float32 array of shape (H, W).
        ndwi_t1: NDWI, epoch T1. float32 array of shape (H, W).
        red_t2: Red band, epoch T2. float32 array of shape (H, W).
        green_t2: Green band, epoch T2. float32 array of shape (H, W).
        blue_t2: Blue band, epoch T2. float32 array of shape (H, W).
        sar_t2: SAR VV band, epoch T2. float32 array of shape (H, W).
        ndvi_t2: NDVI, epoch T2. float32 array of shape (H, W).
        ndbi_t2: NDBI, epoch T2. float32 array of shape (H, W).
        ndwi_t2: NDWI, epoch T2. float32 array of shape (H, W).

    Returns:
        Float32 feature cube of shape (H, W, 18). The third axis corresponds
        to the 18 canonical features in the order defined by
        ``CANONICAL_FEATURE_NAMES``.

    Raises:
        FeatureEngineeringError: If any two input arrays have different shapes.
        FeatureCountMismatchError: If the assembled cube does not have exactly
            18 features (programming error guard).
    """
    logger.info(
        "Assembling 18-feature cube — spatial shape: %s.", red_t1.shape
    )

    # Validate that all 14 input arrays share the same spatial shape.
    input_arrays = {
        "red_t1": red_t1, "green_t1": green_t1, "blue_t1": blue_t1,
        "sar_t1": sar_t1, "ndvi_t1": ndvi_t1, "ndbi_t1": ndbi_t1,
        "ndwi_t1": ndwi_t1,
        "red_t2": red_t2, "green_t2": green_t2, "blue_t2": blue_t2,
        "sar_t2": sar_t2, "ndvi_t2": ndvi_t2, "ndbi_t2": ndbi_t2,
        "ndwi_t2": ndwi_t2,
    }
    _validate_shapes(input_arrays)

    # Compute the four temporal delta features (Group C: indices 14–17).
    delta_ndvi, delta_ndbi, delta_ndwi, delta_sar = compute_all_deltas(
        ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2,
        ndbi_t1=ndbi_t1, ndbi_t2=ndbi_t2,
        ndwi_t1=ndwi_t1, ndwi_t2=ndwi_t2,
        sar_t1=sar_t1, sar_t2=sar_t2,
    )

    # Stack all 18 features in the canonical order using np.stack.
    # The axis=-1 argument places features along the last axis,
    # producing shape (H, W, 18).
    feature_cube = np.stack(
        [
            # --- Group A: Epoch T1 features (indices 0–6) ---
            red_t1.astype(np.float32),    # index 0
            green_t1.astype(np.float32),  # index 1
            blue_t1.astype(np.float32),   # index 2
            sar_t1.astype(np.float32),    # index 3
            ndvi_t1.astype(np.float32),   # index 4
            ndbi_t1.astype(np.float32),   # index 5
            ndwi_t1.astype(np.float32),   # index 6
            # --- Group B: Epoch T2 features (indices 7–13) ---
            red_t2.astype(np.float32),    # index 7
            green_t2.astype(np.float32),  # index 8
            blue_t2.astype(np.float32),   # index 9
            sar_t2.astype(np.float32),    # index 10
            ndvi_t2.astype(np.float32),   # index 11
            ndbi_t2.astype(np.float32),   # index 12
            ndwi_t2.astype(np.float32),   # index 13
            # --- Group C: Temporal delta features (indices 14–17) ---
            delta_ndvi,                   # index 14
            delta_ndbi,                   # index 15
            delta_ndwi,                   # index 16
            delta_sar,                    # index 17
        ],
        axis=-1,
    )

    # Guard against programming errors: verify the feature count.
    actual_features = feature_cube.shape[2]
    if actual_features != CANONICAL_FEATURE_COUNT:
        raise FeatureCountMismatchError(
            expected=CANONICAL_FEATURE_COUNT,
            actual=actual_features,
        )

    logger.info(
        "Feature cube assembled — shape=%s dtype=%s memory=%.1f MB.",
        feature_cube.shape,
        feature_cube.dtype,
        feature_cube.nbytes / (1024 ** 2),
    )
    return feature_cube


# ---------------------------------------------------------------------------
# Feature name accessor
# ---------------------------------------------------------------------------


def get_feature_names() -> Tuple[str, ...]:
    """Return the canonical ordered tuple of feature names.

    The position of each name in the returned tuple corresponds to its
    column index in the flattened feature matrix and its layer index in the
    feature cube's third axis.

    Returns:
        Tuple of 18 feature name strings in canonical order.
    """
    return CANONICAL_FEATURE_NAMES


def get_feature_index(feature_name: str) -> int:
    """Return the column index of a named feature in the feature matrix.

    Args:
        feature_name: Name of the feature (e.g. 'Delta_NDVI').

    Returns:
        Integer column index (0-indexed).

    Raises:
        FeatureEngineeringError: If the feature name is not in the canonical
            feature list.
    """
    try:
        return list(CANONICAL_FEATURE_NAMES).index(feature_name)
    except ValueError:
        raise FeatureEngineeringError(
            f"Feature '{feature_name}' is not in the canonical feature list. "
            f"Valid features: {CANONICAL_FEATURE_NAMES}."
        )


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------


def _validate_shapes(arrays: dict) -> None:
    """Assert that all arrays in the dictionary share the same shape.

    Args:
        arrays: Dictionary mapping names to NumPy arrays.

    Raises:
        FeatureEngineeringError: If any array has a shape different from the
            first array in the dictionary.
    """
    names = list(arrays.keys())
    reference_name = names[0]
    reference_shape = arrays[reference_name].shape

    mismatches = {
        name: arr.shape
        for name, arr in arrays.items()
        if arr.shape != reference_shape
    }
    if mismatches:
        raise FeatureEngineeringError(
            f"All input arrays must have the same shape. "
            f"Reference '{reference_name}' has shape {reference_shape}. "
            f"Mismatches: {mismatches}."
        )
