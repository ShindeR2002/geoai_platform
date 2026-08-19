"""
Electro-optical (EO) raster loading and preprocessing for the GeoAI Platform.

Loads RGB and single-band EO rasters from file into NumPy arrays with
consistent NaN filling. This module implements the exact loading behaviour
validated in Version 1 Notebook 01 and reproduced in Notebook 09.

All loaded arrays have dtype float32 and NaN values replaced by 0.0 using
``np.nan_to_num`` (the canonical Version 1 fill operation). No normalisation,
scaling, or band arithmetic is applied at this stage.

Single responsibility: load EO raster files into NaN-filled float32 arrays.

Position in dependency hierarchy: preprocessing (depends on core, utils).
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from geoai.core.exceptions import RasterIOError, RasterNotFoundError
from geoai.core.io import read_raster, read_raster_bands, RasterProfile
from geoai.utils.raster_utils import fill_nan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RGB loading
# ---------------------------------------------------------------------------


def load_rgb(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RasterProfile]:
    """Load a 3-band RGB raster and return each band as a separate float32 array.

    Reads bands 1, 2, 3 from the raster file as Red, Green, Blue respectively.
    This band order reflects the Version 1 PS10 and Dholera RGB GeoTIFF
    format where Band 1 = Red, Band 2 = Green, Band 3 = Blue.

    After loading, each band is passed through :func:`~geoai.utils.raster_utils.fill_nan`
    to replace NaN and Inf values with 0.0. This is the canonical Version 1
    NaN-filling operation from NB01 and NB09.

    Args:
        path: Path to the 3-band RGB GeoTIFF file.

    Returns:
        Tuple of four elements:
            - ``red``: float32 array of shape ``(H, W)`` — Red band.
            - ``green``: float32 array of shape ``(H, W)`` — Green band.
            - ``blue``: float32 array of shape ``(H, W)`` — Blue band.
            - ``profile``: rasterio profile dictionary from the source file.

    Raises:
        RasterNotFoundError: If the file does not exist at ``path``.
        RasterIOError: If the file cannot be opened or has fewer than 3 bands.
    """
    path = Path(path)
    logger.info("Loading RGB raster: '%s'.", path.name)

    all_bands, profile = read_raster_bands(path, bands=[1, 2, 3], fill_nan=True)

    # all_bands shape: (3, H, W) — split into individual band arrays.
    red = fill_nan(all_bands[0])
    green = fill_nan(all_bands[1])
    blue = fill_nan(all_bands[2])

    logger.info(
        "RGB loaded — shape=(%d, %d) red_range=[%.4g, %.4g] "
        "green_range=[%.4g, %.4g] blue_range=[%.4g, %.4g].",
        red.shape[0], red.shape[1],
        float(red.min()), float(red.max()),
        float(green.min()), float(green.max()),
        float(blue.min()), float(blue.max()),
    )
    return red, green, blue, profile


# ---------------------------------------------------------------------------
# Single-band index loading
# ---------------------------------------------------------------------------


def load_index_band(
    path: Union[str, Path],
    index_name: str = "index",
) -> Tuple[np.ndarray, RasterProfile]:
    """Load a single-band spectral index raster into a NaN-filled float32 array.

    Used for loading precomputed NDVI, NDBI, and NDWI GeoTIFF files (Mode 1
    of the two-mode index strategy). The returned array has NaN and Inf values
    replaced with 0.0.

    Args:
        path: Path to the single-band index GeoTIFF file.
        index_name: Descriptive name of the index for logging (e.g. 'NDVI').

    Returns:
        Tuple of:
            - ``array``: float32 array of shape ``(H, W)`` with NaN filled.
            - ``profile``: rasterio profile dictionary from the source file.

    Raises:
        RasterNotFoundError: If the file does not exist at ``path``.
        RasterIOError: If the file cannot be opened.
    """
    path = Path(path)
    logger.info("Loading %s raster: '%s'.", index_name, path.name)

    raw, profile = read_raster(path, band=1, fill_nan=True)
    filled = fill_nan(raw)

    logger.info(
        "%s loaded — shape=%s range=[%.4g, %.4g] nan_replaced=%d.",
        index_name,
        filled.shape,
        float(filled.min()) if filled.size > 0 else float("nan"),
        float(filled.max()) if filled.size > 0 else float("nan"),
        int(np.isnan(raw).sum()),
    )
    return filled, profile


# ---------------------------------------------------------------------------
# Epoch bundle loading
# ---------------------------------------------------------------------------


def load_eo_epoch(
    rgb_path: Union[str, Path],
    ndvi_path: Union[str, Path],
    ndbi_path: Union[str, Path],
    ndwi_path: Union[str, Path],
    epoch_label: str = "T",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, RasterProfile]:
    """Load all EO bands for a single epoch into individual float32 arrays.

    Convenience function that calls :func:`load_rgb` and :func:`load_index_band`
    for all four source files of one epoch. Returns the seven per-epoch feature
    arrays in the canonical order defined by the feature specification.

    This function does NOT assemble the feature cube; that is the responsibility
    of ``geoai.features.feature_cube``.

    Args:
        rgb_path: Path to the 3-band RGB GeoTIFF for this epoch.
        ndvi_path: Path to the single-band NDVI GeoTIFF for this epoch.
        ndbi_path: Path to the single-band NDBI GeoTIFF for this epoch.
        ndwi_path: Path to the single-band NDWI GeoTIFF for this epoch.
        epoch_label: Short label for logging (e.g. 'T1', 'T2').

    Returns:
        Tuple of eight elements (all arrays are float32 with shape ``(H, W)``):
            - ``red``: Red band.
            - ``green``: Green band.
            - ``blue``: Blue band.
            - ``ndvi``: NDVI values.
            - ``ndbi``: NDBI values.
            - ``ndwi``: NDWI values.
            - ``profile``: rasterio profile from the RGB file (used as the
              reference profile for this epoch).

    Raises:
        RasterNotFoundError: If any of the four source files does not exist.
        RasterIOError: If any file cannot be opened.
    """
    logger.info("Loading EO epoch bundle: %s.", epoch_label)

    red, green, blue, profile = load_rgb(rgb_path)
    ndvi, _ = load_index_band(ndvi_path, index_name=f"NDVI_{epoch_label}")
    ndbi, _ = load_index_band(ndbi_path, index_name=f"NDBI_{epoch_label}")
    ndwi, _ = load_index_band(ndwi_path, index_name=f"NDWI_{epoch_label}")

    logger.info(
        "EO epoch %s loaded — 7 bands ready, shape=%s.",
        epoch_label,
        red.shape,
    )
    return red, green, blue, ndvi, ndbi, ndwi, profile
