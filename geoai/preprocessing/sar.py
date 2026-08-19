"""
Synthetic Aperture Radar (SAR) raster loading for the GeoAI Platform.

Loads SAR VV polarisation rasters from file into NaN-filled float32 arrays.
Implements the exact loading behaviour validated in Version 1 Notebook 01 and
reproduced in Notebook 09 for Sentinel-1 GRD data.

No normalisation, log-scaling, or speckle filtering is applied at this stage.
The values are loaded as-is from the GeoTIFF (expected to be preprocessed
dB-scaled GRD products) and NaN values are replaced with 0.0.

Single responsibility: load SAR VV raster files into NaN-filled float32 arrays.

Position in dependency hierarchy: preprocessing (depends on core, utils).
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from geoai.core.exceptions import RasterIOError, RasterNotFoundError
from geoai.core.io import read_raster, RasterProfile
from geoai.utils.raster_utils import fill_nan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAR VV loading
# ---------------------------------------------------------------------------


def load_sar_vv(
    path: Union[str, Path],
    epoch_label: str = "T",
) -> Tuple[np.ndarray, RasterProfile]:
    """Load a SAR VV polarisation raster into a NaN-filled float32 array.

    Reads band 1 from the specified GeoTIFF file, which is expected to contain
    Sentinel-1 VV polarisation data as a pre-processed GRD product. After
    loading, NaN and Inf values are replaced with 0.0 using the canonical
    Version 1 NaN-filling operation.

    This corresponds to the SAR loading performed in Version 1 Notebook 01
    (per-epoch feature assembly) and reproduced in Notebook 09.

    Args:
        path: Path to the single-band SAR VV GeoTIFF file.
        epoch_label: Short label for logging (e.g. 'T1' for 2021, 'T2' for 2024).

    Returns:
        Tuple of:
            - ``sar_vv``: float32 array of shape ``(H, W)`` with NaN filled.
            - ``profile``: rasterio profile dictionary from the source file.

    Raises:
        RasterNotFoundError: If the file does not exist at ``path``.
        RasterIOError: If the file cannot be opened.
    """
    path = Path(path)
    logger.info("Loading SAR VV raster (%s): '%s'.", epoch_label, path.name)

    raw, profile = read_raster(path, band=1, fill_nan=True)
    sar_vv = fill_nan(raw)

    nan_count = int(np.isnan(raw).sum())
    logger.info(
        "SAR VV %s loaded — shape=%s range=[%.4g, %.4g] nan_replaced=%d.",
        epoch_label,
        sar_vv.shape,
        float(sar_vv.min()) if sar_vv.size > 0 else float("nan"),
        float(sar_vv.max()) if sar_vv.size > 0 else float("nan"),
        nan_count,
    )
    return sar_vv, profile


# ---------------------------------------------------------------------------
# SAR epoch pair loading
# ---------------------------------------------------------------------------


def load_sar_epoch_pair(
    path_t1: Union[str, Path],
    path_t2: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, RasterProfile]:
    """Load SAR VV rasters for both epochs (T1 and T2).

    Convenience function that loads both epoch SAR files and returns them
    together with the T1 profile as the reference.

    Args:
        path_t1: Path to the SAR VV GeoTIFF for epoch T1.
        path_t2: Path to the SAR VV GeoTIFF for epoch T2.

    Returns:
        Tuple of:
            - ``sar_t1``: float32 array of shape ``(H, W)`` — SAR VV, epoch T1.
            - ``sar_t2``: float32 array of shape ``(H, W)`` — SAR VV, epoch T2.
            - ``profile``: rasterio profile from the T1 file.

    Raises:
        RasterNotFoundError: If either file does not exist.
        RasterIOError: If either file cannot be opened.
    """
    logger.info("Loading SAR VV epoch pair.")
    sar_t1, profile = load_sar_vv(path_t1, epoch_label="T1")
    sar_t2, _ = load_sar_vv(path_t2, epoch_label="T2")
    logger.info("SAR VV epoch pair loaded — shape=%s.", sar_t1.shape)
    return sar_t1, sar_t2, profile
