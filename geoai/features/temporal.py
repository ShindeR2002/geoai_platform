"""
Temporal change feature computation for the GeoAI Platform.

Computes the four temporal difference features that form Group C (indices
14–17) of the canonical 18-feature cube. Each delta feature is the simple
arithmetic difference between the T2 (later epoch) and T1 (earlier epoch)
values of the corresponding spectral band.

Delta computation: delta = T2_value - T1_value

No normalisation, clipping, or scaling is applied. The raw difference is
stored directly in the feature cube. This is the exact behaviour from
Version 1 Notebook 01.

Single responsibility: compute T2 − T1 difference arrays for each spectral
feature required by the delta block of the feature cube.

Position in dependency hierarchy: features (depends on core, utils).
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual delta computations
# ---------------------------------------------------------------------------


def compute_delta_ndvi(ndvi_t2: np.ndarray, ndvi_t1: np.ndarray) -> np.ndarray:
    """Compute the temporal NDVI difference: NDVI_T2 - NDVI_T1.

    Args:
        ndvi_t2: NDVI array for the later epoch (T2), float32 of shape (H, W).
        ndvi_t1: NDVI array for the earlier epoch (T1), float32 of shape (H, W).

    Returns:
        Delta_NDVI float32 array of shape (H, W). Feature index 14 in the
        canonical feature cube.
    """
    return _delta(ndvi_t2, ndvi_t1, "Delta_NDVI")


def compute_delta_ndbi(ndbi_t2: np.ndarray, ndbi_t1: np.ndarray) -> np.ndarray:
    """Compute the temporal NDBI difference: NDBI_T2 - NDBI_T1.

    Args:
        ndbi_t2: NDBI array for the later epoch (T2), float32 of shape (H, W).
        ndbi_t1: NDBI array for the earlier epoch (T1), float32 of shape (H, W).

    Returns:
        Delta_NDBI float32 array of shape (H, W). Feature index 15 in the
        canonical feature cube.
    """
    return _delta(ndbi_t2, ndbi_t1, "Delta_NDBI")


def compute_delta_ndwi(ndwi_t2: np.ndarray, ndwi_t1: np.ndarray) -> np.ndarray:
    """Compute the temporal NDWI difference: NDWI_T2 - NDWI_T1.

    Args:
        ndwi_t2: NDWI array for the later epoch (T2), float32 of shape (H, W).
        ndwi_t1: NDWI array for the earlier epoch (T1), float32 of shape (H, W).

    Returns:
        Delta_NDWI float32 array of shape (H, W). Feature index 16 in the
        canonical feature cube.
    """
    return _delta(ndwi_t2, ndwi_t1, "Delta_NDWI")


def compute_delta_sar(sar_t2: np.ndarray, sar_t1: np.ndarray) -> np.ndarray:
    """Compute the temporal SAR VV difference: SAR_T2 - SAR_T1.

    Args:
        sar_t2: SAR VV array for the later epoch (T2), float32 of shape (H, W).
        sar_t1: SAR VV array for the earlier epoch (T1), float32 of shape (H, W).

    Returns:
        Delta_SAR float32 array of shape (H, W). Feature index 17 in the
        canonical feature cube.
    """
    return _delta(sar_t2, sar_t1, "Delta_SAR")


# ---------------------------------------------------------------------------
# Batch delta computation
# ---------------------------------------------------------------------------


def compute_all_deltas(
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
    ndbi_t1: np.ndarray,
    ndbi_t2: np.ndarray,
    ndwi_t1: np.ndarray,
    ndwi_t2: np.ndarray,
    sar_t1: np.ndarray,
    sar_t2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all four temporal difference features in one call.

    Computes Delta_NDVI, Delta_NDBI, Delta_NDWI, and Delta_SAR in the order
    required for feature cube assembly (indices 14, 15, 16, 17).

    Args:
        ndvi_t1: NDVI array, epoch T1.
        ndvi_t2: NDVI array, epoch T2.
        ndbi_t1: NDBI array, epoch T1.
        ndbi_t2: NDBI array, epoch T2.
        ndwi_t1: NDWI array, epoch T1.
        ndwi_t2: NDWI array, epoch T2.
        sar_t1: SAR VV array, epoch T1.
        sar_t2: SAR VV array, epoch T2.

    Returns:
        Tuple of four float32 arrays of shape (H, W):
            (delta_ndvi, delta_ndbi, delta_ndwi, delta_sar)
        These map to feature cube indices 14, 15, 16, 17 respectively.
    """
    logger.info("Computing all four temporal delta features.")

    delta_ndvi = compute_delta_ndvi(ndvi_t2, ndvi_t1)
    delta_ndbi = compute_delta_ndbi(ndbi_t2, ndbi_t1)
    delta_ndwi = compute_delta_ndwi(ndwi_t2, ndwi_t1)
    delta_sar = compute_delta_sar(sar_t2, sar_t1)

    logger.info(
        "Delta features computed — "
        "NDVI range=[%.4g, %.4g], NDBI range=[%.4g, %.4g], "
        "NDWI range=[%.4g, %.4g], SAR range=[%.4g, %.4g].",
        float(delta_ndvi.min()), float(delta_ndvi.max()),
        float(delta_ndbi.min()), float(delta_ndbi.max()),
        float(delta_ndwi.min()), float(delta_ndwi.max()),
        float(delta_sar.min()), float(delta_sar.max()),
    )
    return delta_ndvi, delta_ndbi, delta_ndwi, delta_sar


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _delta(t2: np.ndarray, t1: np.ndarray, name: str) -> np.ndarray:
    """Compute T2 - T1 difference and return as float32.

    Args:
        t2: Later-epoch array.
        t1: Earlier-epoch array.
        name: Feature name for logging.

    Returns:
        Float32 difference array of the same shape as the inputs.
    """
    result = (t2.astype(np.float32) - t1.astype(np.float32))
    logger.debug(
        "%s: min=%.4g max=%.4g mean=%.4g std=%.4g.",
        name,
        float(result.min()),
        float(result.max()),
        float(result.mean()),
        float(result.std()),
    )
    return result
