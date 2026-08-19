"""
NumPy raster array utility functions for the GeoAI Platform.

Provides helper functions for common array manipulation operations that appear
in multiple modules. Centralising these operations prevents code duplication
and ensures consistent behaviour across the feature engineering and analysis
layers.

Single responsibility: array shape manipulation and raster-specific NumPy
operations.

Position in dependency hierarchy: utils (depends on constants only).
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def fill_nan(array: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN values in an array with a fill value.

    This is the canonical NaN-filling operation used throughout the platform.
    It wraps ``np.nan_to_num`` to ensure consistent behaviour. All production
    modules must use this function rather than calling ``np.nan_to_num``
    directly.

    Preserves Version 1 behaviour where ``np.nan_to_num`` was called with no
    explicit fill value, defaulting to 0.0.

    Args:
        array: Input array, possibly containing NaN values.
        fill_value: Value to substitute for NaN. Default 0.0 (Version 1 default).

    Returns:
        A copy of ``array`` with NaN values replaced by ``fill_value``.
        Infinity values (positive and negative) are also replaced (by
        ``np.nan_to_num`` convention).
    """
    return np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)


def count_nan(array: np.ndarray) -> int:
    """Count the number of NaN values in an array.

    Args:
        array: Input array to inspect.

    Returns:
        Integer count of NaN values.
    """
    return int(np.isnan(array).sum())


def has_nan(array: np.ndarray) -> bool:
    """Return True if the array contains any NaN values.

    Args:
        array: Input array to inspect.

    Returns:
        True if at least one NaN value is present; False otherwise.
    """
    return bool(np.isnan(array).any())


# ---------------------------------------------------------------------------
# Array shape utilities
# ---------------------------------------------------------------------------


def flatten_spatial(array: np.ndarray) -> np.ndarray:
    """Flatten a (H, W, C) feature cube to a (H*W, C) feature matrix.

    This is the standard operation for preparing a feature cube for ML
    model input. The spatial structure is destroyed; use
    :func:`restore_spatial` to recover it.

    Args:
        array: NumPy array of shape ``(H, W, C)``.

    Returns:
        NumPy array of shape ``(H * W, C)``.

    Raises:
        ValueError: If ``array`` is not 3-dimensional.
    """
    if array.ndim != 3:
        raise ValueError(
            f"flatten_spatial expects a 3D array of shape (H, W, C), "
            f"got shape {array.shape}."
        )
    H, W, C = array.shape
    return array.reshape(H * W, C)


def restore_spatial(
    flat_array: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Restore a flat (H*W,) or (H*W, C) array to spatial shape (H, W) or (H, W, C).

    Args:
        flat_array: 1D array of shape ``(H*W,)`` or 2D array of shape
            ``(H*W, C)``.
        height: Original spatial height H.
        width: Original spatial width W.

    Returns:
        Array reshaped to ``(H, W)`` or ``(H, W, C)`` depending on input rank.

    Raises:
        ValueError: If the total element count does not equal ``height * width``
            (for 1D input) or the first dimension is not ``height * width``
            (for 2D input).
    """
    if flat_array.ndim == 1:
        if flat_array.size != height * width:
            raise ValueError(
                f"Cannot reshape flat array of size {flat_array.size} to "
                f"({height}, {width}): expected {height * width} elements."
            )
        return flat_array.reshape(height, width)
    if flat_array.ndim == 2:
        if flat_array.shape[0] != height * width:
            raise ValueError(
                f"Cannot reshape flat array of shape {flat_array.shape} to "
                f"({height}, {width}, {flat_array.shape[1]}): "
                f"first dimension should be {height * width}."
            )
        return flat_array.reshape(height, width, flat_array.shape[1])
    raise ValueError(
        f"restore_spatial expects a 1D or 2D array, got shape {flat_array.shape}."
    )


# ---------------------------------------------------------------------------
# Valid pixel mask
# ---------------------------------------------------------------------------


def build_valid_pixel_mask(feature_matrix: np.ndarray) -> np.ndarray:
    """Build the canonical valid pixel mask for a feature matrix.

    This implements the compound valid pixel mask from Version 1 Notebook 09
    Cell 12. It is the ONLY valid pixel mask used in the platform for both
    training and inference.

    A pixel is valid if:
    - No feature value is NaN (guards against nodata from raster loading).
    - Not all feature values are zero (guards against padded or empty pixels).

    Args:
        feature_matrix: 2D NumPy array of shape ``(N_pixels, N_features)``
            produced by :func:`flatten_spatial`.

    Returns:
        1D boolean NumPy array of shape ``(N_pixels,)`` where True indicates
        a valid pixel.
    """
    not_nan = ~np.isnan(feature_matrix).any(axis=1)
    not_all_zero = ~np.all(feature_matrix == 0, axis=1)
    valid_mask = not_nan & not_all_zero
    logger.debug(
        "Valid pixel mask: %d valid / %d total (%.1f%% valid).",
        int(valid_mask.sum()),
        feature_matrix.shape[0],
        100.0 * valid_mask.sum() / max(feature_matrix.shape[0], 1),
    )
    return valid_mask


def reconstruct_prediction_raster(
    predictions: np.ndarray,
    valid_mask: np.ndarray,
    total_pixels: int,
) -> np.ndarray:
    """Reconstruct a full prediction array from valid-pixel predictions.

    Positions corresponding to invalid pixels (where ``valid_mask`` is False)
    are filled with NaN. This is the canonical inference reconstruction pattern
    from Version 1 Notebook 09 Cells 14–15.

    Args:
        predictions: 1D array of shape ``(N_valid,)`` containing model
            predictions for valid pixels only.
        valid_mask: 1D boolean array of shape ``(N_total,)`` produced by
            :func:`build_valid_pixel_mask`.
        total_pixels: Total number of pixels (``N_total = H * W``).

    Returns:
        1D float32 array of shape ``(N_total,)`` with predictions at valid
        positions and NaN at invalid positions. Call
        :func:`restore_spatial` afterwards to recover the 2D raster.

    Raises:
        ValueError: If the number of predictions does not equal the number
            of True values in ``valid_mask``.
    """
    n_valid = int(valid_mask.sum())
    if len(predictions) != n_valid:
        raise ValueError(
            f"predictions has {len(predictions)} elements but valid_mask "
            f"has {n_valid} True values. These must match."
        )

    prediction_flat = np.full(total_pixels, np.nan, dtype=np.float32)
    prediction_flat[valid_mask] = predictions.astype(np.float32)
    return prediction_flat


# ---------------------------------------------------------------------------
# Contrast stretch
# ---------------------------------------------------------------------------


def stretch_contrast(
    array: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """Apply percentile-based contrast stretch to an array for visualisation.

    Clips the array to the specified percentile range and rescales to [0, 1].
    Used for RGB visualisation only; never applied to feature values used
    in model training or inference.

    Source: Version 1 Notebook 01 Cell 5.

    Args:
        array: Input array to stretch (typically a single raster band).
        low_pct: Lower percentile cutoff. Default 2.0 (from V1 NB01).
        high_pct: Upper percentile cutoff. Default 98.0 (from V1 NB01).

    Returns:
        Float32 array with values in [0.0, 1.0]. NaN values in the input
        are preserved as NaN in the output.
    """
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    low_val = float(np.percentile(valid, low_pct))
    high_val = float(np.percentile(valid, high_pct))

    if high_val == low_val:
        logger.warning(
            "Contrast stretch: low and high percentile values are equal (%.4g). "
            "Returning zeros.",
            low_val,
        )
        return np.zeros_like(array, dtype=np.float32)

    clipped = np.clip(array, low_val, high_val)
    stretched = (clipped - low_val) / (high_val - low_val)
    return stretched.astype(np.float32)
