"""
Spectral index loading and computation for the GeoAI Platform.

Implements the two-mode spectral index strategy:

  Mode 1 — Load (default production path):
    Load precomputed NDVI, NDBI, and NDWI from single-band GeoTIFF files.
    This preserves the exact Version 1 behaviour where all index rasters
    are provided as pre-derived files from the Bhoonidhi and Copernicus
    portals.

  Mode 2 — Compute (optional, for new datasets):
    Derive NDVI, NDBI, and NDWI from raw multi-band EO imagery using
    standard spectral equations. Activated when precomputed rasters are
    not available.

Both modes return float32 arrays of shape (H, W) with NaN and Inf values
replaced by 0.0. Outputs are numerically compatible regardless of mode.

The active mode is controlled by the AOI configuration key ``index_mode``
(values: 'load' or 'compute'). See ``geoai/core/config.py`` and
``configs/processing.yaml``.

Single responsibility: provide NDVI, NDBI, NDWI arrays from either source.

Position in dependency hierarchy: preprocessing (depends on core, utils,
preprocessing/registry).
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from geoai.core.exceptions import ConfigurationError, RasterIOError, RasterNotFoundError
from geoai.core.io import read_raster, read_raster_bands, RasterProfile
from geoai.preprocessing.registry import get_sensor_spec
from geoai.utils.constants import IndexMode, SensorID
from geoai.utils.raster_utils import fill_nan

logger = logging.getLogger(__name__)

# Small constant added to denominators to prevent division-by-zero in index
# computation. NaN values resulting from genuine zero-division are replaced
# by fill_nan() after computation.
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Mode 1 — Load precomputed index rasters (default production path)
# ---------------------------------------------------------------------------


def load_ndvi(path: Union[str, Path]) -> Tuple[np.ndarray, RasterProfile]:
    """Load a precomputed NDVI raster from file (Mode 1).

    Args:
        path: Path to the single-band NDVI GeoTIFF.

    Returns:
        Tuple of (ndvi_array, profile) where ndvi_array is float32 (H, W)
        with NaN filled by 0.0.

    Raises:
        RasterNotFoundError: If the file does not exist.
        RasterIOError: If the file cannot be opened.
    """
    return _load_single_index(path, "NDVI")


def load_ndbi(path: Union[str, Path]) -> Tuple[np.ndarray, RasterProfile]:
    """Load a precomputed NDBI raster from file (Mode 1).

    Args:
        path: Path to the single-band NDBI GeoTIFF.

    Returns:
        Tuple of (ndbi_array, profile) where ndbi_array is float32 (H, W)
        with NaN filled by 0.0.

    Raises:
        RasterNotFoundError: If the file does not exist.
        RasterIOError: If the file cannot be opened.
    """
    return _load_single_index(path, "NDBI")


def load_ndwi(path: Union[str, Path]) -> Tuple[np.ndarray, RasterProfile]:
    """Load a precomputed NDWI raster from file (Mode 1).

    Args:
        path: Path to the single-band NDWI GeoTIFF.

    Returns:
        Tuple of (ndwi_array, profile) where ndwi_array is float32 (H, W)
        with NaN filled by 0.0.

    Raises:
        RasterNotFoundError: If the file does not exist.
        RasterIOError: If the file cannot be opened.
    """
    return _load_single_index(path, "NDWI")


def load_all_indices(
    ndvi_path: Union[str, Path],
    ndbi_path: Union[str, Path],
    ndwi_path: Union[str, Path],
    epoch_label: str = "T",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all three precomputed index rasters for one epoch (Mode 1).

    Convenience function that loads NDVI, NDBI, and NDWI from their
    respective files and returns the three arrays together.

    Args:
        ndvi_path: Path to the NDVI GeoTIFF.
        ndbi_path: Path to the NDBI GeoTIFF.
        ndwi_path: Path to the NDWI GeoTIFF.
        epoch_label: Short label used in logging (e.g. 'T1', 'T2').

    Returns:
        Tuple of (ndvi, ndbi, ndwi) as float32 arrays of shape (H, W).

    Raises:
        RasterNotFoundError: If any file does not exist.
        RasterIOError: If any file cannot be opened.
    """
    logger.info("Loading precomputed index rasters for epoch %s.", epoch_label)
    ndvi, _ = load_ndvi(ndvi_path)
    ndbi, _ = load_ndbi(ndbi_path)
    ndwi, _ = load_ndwi(ndwi_path)
    logger.info(
        "Indices loaded for epoch %s — NDVI range=[%.4g, %.4g], "
        "NDBI range=[%.4g, %.4g], NDWI range=[%.4g, %.4g].",
        epoch_label,
        float(ndvi.min()), float(ndvi.max()),
        float(ndbi.min()), float(ndbi.max()),
        float(ndwi.min()), float(ndwi.max()),
    )
    return ndvi, ndbi, ndwi


# ---------------------------------------------------------------------------
# Mode 2 — Compute indices from raw band data (optional)
# ---------------------------------------------------------------------------


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI from NIR and Red band arrays (Mode 2).

    NDVI = (NIR - Red) / (NIR + Red)

    Valid NDVI values range from -1.0 to +1.0:
      - Positive values indicate vegetation.
      - Values near 0 indicate bare soil or water.
      - Negative values may indicate water or cloud.

    Division-by-zero positions (where NIR + Red == 0) are set to NaN and
    subsequently filled with 0.0 by fill_nan().

    Args:
        nir: Near-infrared band array, float32 of shape (H, W).
        red: Red band array, float32 of shape (H, W).

    Returns:
        NDVI float32 array of shape (H, W) with values in [-1, 1],
        NaN positions filled with 0.0.
    """
    logger.debug("Computing NDVI from NIR and Red bands.")
    numerator = nir.astype(np.float32) - red.astype(np.float32)
    denominator = nir.astype(np.float32) + red.astype(np.float32)
    # Suppress divide-by-zero warning; resulting NaN is handled below.
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(np.abs(denominator) > _EPSILON, numerator / denominator, np.nan)
    return fill_nan(ndvi.astype(np.float32))


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDBI from SWIR and NIR band arrays (Mode 2).

    NDBI = (SWIR - NIR) / (SWIR + NIR)

    Positive NDBI values indicate built-up or impervious surfaces.
    Negative values typically indicate vegetation or water.

    Args:
        swir: Short-wave infrared band array, float32 of shape (H, W).
            For Sentinel-2: Band 11 (1610 nm).
            For LISS-IV: Band 4 (MIR, 1550-1700 nm — see registry notes).
        nir: Near-infrared band array, float32 of shape (H, W).

    Returns:
        NDBI float32 array of shape (H, W), NaN positions filled with 0.0.
    """
    logger.debug("Computing NDBI from SWIR and NIR bands.")
    numerator = swir.astype(np.float32) - nir.astype(np.float32)
    denominator = swir.astype(np.float32) + nir.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndbi = np.where(np.abs(denominator) > _EPSILON, numerator / denominator, np.nan)
    return fill_nan(ndbi.astype(np.float32))


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDWI from Green and NIR band arrays (Mode 2).

    NDWI = (Green - NIR) / (Green + NIR)

    Positive NDWI values indicate open water surfaces.
    Negative values indicate vegetation or built-up areas.

    Args:
        green: Green band array, float32 of shape (H, W).
        nir: Near-infrared band array, float32 of shape (H, W).

    Returns:
        NDWI float32 array of shape (H, W), NaN positions filled with 0.0.
    """
    logger.debug("Computing NDWI from Green and NIR bands.")
    numerator = green.astype(np.float32) - nir.astype(np.float32)
    denominator = green.astype(np.float32) + nir.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(np.abs(denominator) > _EPSILON, numerator / denominator, np.nan)
    return fill_nan(ndwi.astype(np.float32))


def compute_indices_from_multiband(
    multiband_path: Union[str, Path],
    sensor_id: str = SensorID.SENTINEL2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RasterProfile]:
    """Compute NDVI, NDBI, and NDWI from a raw multi-band EO raster (Mode 2).

    Reads the required bands from a multi-band GeoTIFF using the band
    indices specified in the sensor registry for the given sensor, then
    computes all three indices.

    Args:
        multiband_path: Path to the raw multi-band EO GeoTIFF.
        sensor_id: Sensor identifier string (e.g. 'sentinel2'). Used to
            look up the correct band indices in the sensor registry.

    Returns:
        Tuple of (ndvi, ndbi, ndwi, profile) where all index arrays are
        float32 of shape (H, W) and profile is the rasterio metadata.

    Raises:
        RasterNotFoundError: If the file does not exist.
        RasterIOError: If the file cannot be opened.
        ConfigurationError: If the sensor does not support all three indices.
        UnknownSensorError: If ``sensor_id`` is not in the registry.
    """
    spec = get_sensor_spec(sensor_id)

    if not spec.can_compute_all_indices():
        raise ConfigurationError(
            f"Sensor '{sensor_id}' does not have band definitions for all three "
            "spectral indices (NDVI, NDBI, NDWI). Check the sensor registry."
        )

    logger.info(
        "Computing indices from raw multi-band raster '%s' using sensor '%s'.",
        Path(multiband_path).name,
        sensor_id,
    )

    # Determine which unique bands need to be read.
    nir_idx, red_idx = spec.ndvi_bands
    swir_idx, _ = spec.ndbi_bands
    green_idx, _ = spec.ndwi_bands
    required_bands = sorted(set([nir_idx, red_idx, swir_idx, green_idx]))

    all_bands, profile = read_raster_bands(
        multiband_path, bands=required_bands, fill_nan=True
    )

    # Map band index to array position in the loaded subset.
    band_to_array_idx = {b: i for i, b in enumerate(required_bands)}

    nir = fill_nan(all_bands[band_to_array_idx[nir_idx]])
    red = fill_nan(all_bands[band_to_array_idx[red_idx]])
    swir = fill_nan(all_bands[band_to_array_idx[swir_idx]])
    green = fill_nan(all_bands[band_to_array_idx[green_idx]])

    ndvi = compute_ndvi(nir, red)
    ndbi = compute_ndbi(swir, nir)
    ndwi = compute_ndwi(green, nir)

    logger.info(
        "Indices computed — NDVI range=[%.4g, %.4g], "
        "NDBI range=[%.4g, %.4g], NDWI range=[%.4g, %.4g].",
        float(ndvi.min()), float(ndvi.max()),
        float(ndbi.min()), float(ndbi.max()),
        float(ndwi.min()), float(ndwi.max()),
    )
    return ndvi, ndbi, ndwi, profile


# ---------------------------------------------------------------------------
# Mode-dispatching loader (used by the feature pipeline)
# ---------------------------------------------------------------------------


def get_indices(
    mode: str,
    ndvi_path: Union[str, Path, None] = None,
    ndbi_path: Union[str, Path, None] = None,
    ndwi_path: Union[str, Path, None] = None,
    multiband_path: Union[str, Path, None] = None,
    sensor_id: str = SensorID.SENTINEL2,
    epoch_label: str = "T",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute NDVI, NDBI, and NDWI depending on the configured mode.

    This is the primary entry point for index acquisition used by the feature
    cube assembly module. It dispatches to Mode 1 or Mode 2 based on the
    ``mode`` argument.

    Args:
        mode: Index sourcing mode. Must be ``'load'`` (Mode 1, default) or
            ``'compute'`` (Mode 2).
        ndvi_path: Path to precomputed NDVI raster (required for mode='load').
        ndbi_path: Path to precomputed NDBI raster (required for mode='load').
        ndwi_path: Path to precomputed NDWI raster (required for mode='load').
        multiband_path: Path to raw multi-band EO raster (required for
            mode='compute').
        sensor_id: Sensor identifier for Mode 2 band lookup.
        epoch_label: Short epoch label for logging (e.g. 'T1', 'T2').

    Returns:
        Tuple of (ndvi, ndbi, ndwi) as float32 arrays of shape (H, W).

    Raises:
        ConfigurationError: If required paths for the selected mode are missing.
        RasterNotFoundError: If any required raster file does not exist.
        RasterIOError: If any file cannot be opened.
    """
    if mode == IndexMode.LOAD:
        for name, path in [("ndvi_path", ndvi_path),
                           ("ndbi_path", ndbi_path),
                           ("ndwi_path", ndwi_path)]:
            if path is None:
                raise ConfigurationError(
                    f"index_mode is 'load' but '{name}' was not provided. "
                    "Provide paths to precomputed index rasters or set "
                    "index_mode to 'compute'."
                )
        return load_all_indices(ndvi_path, ndbi_path, ndwi_path, epoch_label)

    if mode == IndexMode.COMPUTE:
        if multiband_path is None:
            raise ConfigurationError(
                "index_mode is 'compute' but 'multiband_path' was not provided. "
                "Provide a path to the raw multi-band EO raster or set "
                "index_mode to 'load'."
            )
        ndvi, ndbi, ndwi, _ = compute_indices_from_multiband(
            multiband_path, sensor_id
        )
        return ndvi, ndbi, ndwi

    raise ConfigurationError(
        f"Unknown index_mode '{mode}'. Valid values are 'load' and 'compute'."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_single_index(
    path: Union[str, Path],
    index_name: str,
) -> Tuple[np.ndarray, RasterProfile]:
    """Read a single-band index GeoTIFF and apply NaN fill.

    Args:
        path: Path to the GeoTIFF file.
        index_name: Name used in log messages.

    Returns:
        Tuple of (filled_array, profile).

    Raises:
        RasterNotFoundError: If the file does not exist.
        RasterIOError: If the file cannot be opened.
    """
    path = Path(path)
    logger.debug("Loading %s from '%s'.", index_name, path.name)
    raw, profile = read_raster(path, band=1, fill_nan=True)
    filled = fill_nan(raw)
    logger.debug(
        "%s loaded — shape=%s range=[%.4g, %.4g].",
        index_name,
        filled.shape,
        float(filled.min()) if filled.size > 0 else float("nan"),
        float(filled.max()) if filled.size > 0 else float("nan"),
    )
    return filled, profile
