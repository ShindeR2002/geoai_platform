"""
Prediction post-processing for the GeoAI Platform.

Converts the floating-point prediction raster produced by model inference
into a clean binary change mask ready for object extraction.

The canonical post-processing pipeline (from Version 1 NB09 Cell 16):

    binary_change = np.nan_to_num(prediction_map, nan=0).astype(np.uint8)

This module also provides optional morphological refinement operations that
can be activated via configuration. These operations are disabled by default
to preserve exact Version 1 behaviour.

Single responsibility: convert prediction maps to binary masks.

Position in dependency hierarchy: postprocessing (depends on core, utils).
"""

import logging
from typing import Optional

import numpy as np
from scipy import ndimage

from geoai.core.exceptions import GeoAIPlatformError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical binary conversion (Version 1 NB09 Cell 16)
# ---------------------------------------------------------------------------


def to_binary_mask(prediction_map: np.ndarray) -> np.ndarray:
    """Convert a float32 prediction raster to a uint8 binary change mask.

    Implements the exact Version 1 NB09 Cell 16 conversion:

        binary_change = np.nan_to_num(prediction_map, nan=0).astype(np.uint8)

    NaN values at invalid pixel positions become 0 (no change). This is the
    canonical post-processing step and must be called before object extraction.

    Args:
        prediction_map: Float32 array of shape (H, W) from
            :func:`~geoai.models.inference.run_inference`. Contains values
            0.0 (no change), 1.0 (change), or NaN (invalid pixel).

    Returns:
        uint8 array of shape (H, W) with values 0 (no change) and 1 (change).
    """
    binary = np.nan_to_num(prediction_map, nan=0).astype(np.uint8)

    n_change = int(binary.sum())
    n_total = binary.size
    logger.info(
        "Binary mask created — change=%d / %d (%.2f%%).",
        n_change, n_total, 100.0 * n_change / max(n_total, 1),
    )
    return binary


# ---------------------------------------------------------------------------
# Optional morphological operations
# ---------------------------------------------------------------------------


def apply_morphological_closing(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological closing to fill small gaps in detected change regions.

    Closing = dilation followed by erosion. It fills small holes within
    change objects and connects nearby disconnected fragments.

    This operation is NOT applied in the default pipeline to preserve exact
    Version 1 behaviour. Activate by setting it explicitly in Stage 1
    pipeline configuration.

    Args:
        binary_mask: uint8 binary array of shape (H, W).
        kernel_size: Side length of the square structuring element.
            Default 3 (3×3 kernel).

    Returns:
        uint8 binary array of shape (H, W) after morphological closing.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = ndimage.binary_closing(binary_mask, structure=kernel).astype(np.uint8)

    n_before = int(binary_mask.sum())
    n_after = int(closed.sum())
    logger.info(
        "Morphological closing (kernel=%dx%d) — change px: %d → %d.",
        kernel_size, kernel_size, n_before, n_after,
    )
    return closed


def apply_morphological_opening(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological opening to remove isolated noise pixels.

    Opening = erosion followed by dilation. It removes small isolated change
    pixels (salt noise) while preserving the shape of larger objects.

    This operation is NOT applied in the default pipeline.

    Args:
        binary_mask: uint8 binary array of shape (H, W).
        kernel_size: Side length of the square structuring element.

    Returns:
        uint8 binary array of shape (H, W) after morphological opening.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = ndimage.binary_opening(binary_mask, structure=kernel).astype(np.uint8)

    n_before = int(binary_mask.sum())
    n_after = int(opened.sum())
    logger.info(
        "Morphological opening (kernel=%dx%d) — change px: %d → %d.",
        kernel_size, kernel_size, n_before, n_after,
    )
    return opened


def apply_morphological_dilation(
    binary_mask: np.ndarray,
    iterations: int = 1,
) -> np.ndarray:
    """Expand change regions outward by a given number of pixel iterations.

    Args:
        binary_mask: uint8 binary array of shape (H, W).
        iterations: Number of dilation iterations.

    Returns:
        uint8 binary array of shape (H, W) after dilation.
    """
    dilated = ndimage.binary_dilation(
        binary_mask, iterations=iterations
    ).astype(np.uint8)
    logger.info(
        "Morphological dilation (iterations=%d) — change px: %d → %d.",
        iterations, int(binary_mask.sum()), int(dilated.sum()),
    )
    return dilated
