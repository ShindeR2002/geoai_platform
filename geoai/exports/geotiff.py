"""
GeoTIFF export for the GeoAI Platform.

Writes binary change masks and classified change rasters to GeoTIFF files
in the PS10-compliant format: pixel value 1 = change, pixel value 0 = no
change, georeferenced with the CRS and affine transform of the source raster.

PS10 submission output filename convention:
    Change_Mask_{Lat}_{Long}.tif

Single responsibility: write change mask and classified rasters to GeoTIFF.

Position in dependency hierarchy: exports (depends on core/io, utils).
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from geoai.core.exceptions import GeoTIFFExportError
from geoai.core.io import RasterProfile, write_binary_mask, write_raster
from geoai.utils.constants import PS10_FILENAME_PREFIX
from geoai.utils.geo import format_ps10_filename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Binary change mask export (PS10 primary deliverable)
# ---------------------------------------------------------------------------


def export_change_mask_geotiff(
    binary_mask: np.ndarray,
    profile: RasterProfile,
    output_path: Union[str, Path],
    compress: str = "lzw",
) -> Path:
    """Write a binary change mask to a georeferenced GeoTIFF file.

    This is the primary PS10 Stage 1 deliverable. The output file contains:
    - Pixel value 1: change detected.
    - Pixel value 0: no change detected.
    - uint8 data type.
    - CRS and affine transform copied from ``profile``.

    Args:
        binary_mask: uint8 binary array of shape (H, W) with values 0 and 1.
        profile: rasterio profile dictionary providing CRS and transform.
            Obtained from the source imagery rasterio metadata.
        output_path: Destination file path. Should end with ``.tif``.
        compress: GeoTIFF compression algorithm. Default ``'lzw'``.

    Returns:
        Resolved absolute path of the written GeoTIFF file.

    Raises:
        GeoTIFFExportError: If writing fails.
    """
    output_path = Path(output_path)
    logger.info(
        "Exporting binary change mask to '%s' "
        "(change_px=%d compress=%s).",
        output_path.name,
        int(binary_mask.sum()),
        compress,
    )
    try:
        result = write_binary_mask(
            path=output_path,
            mask=binary_mask,
            profile=profile,
            compress=compress,
        )
    except Exception as exc:
        raise GeoTIFFExportError(str(output_path), str(exc)) from exc

    logger.info("Change mask GeoTIFF written: '%s'.", result)
    return result


def export_ps10_change_mask(
    binary_mask: np.ndarray,
    profile: RasterProfile,
    output_dir: Union[str, Path],
    reference_lat: float,
    reference_lon: float,
    prefix: str = PS10_FILENAME_PREFIX,
    compress: str = "lzw",
) -> Path:
    """Write the PS10-convention change mask GeoTIFF with the required filename.

    Constructs the output filename as ``Change_Mask_{Lat}_{Long}.tif`` per
    the PS10 evaluation methodology (Section 8.c.4 of the problem statement).

    Args:
        binary_mask: uint8 binary array of shape (H, W).
        profile: rasterio profile from the source imagery.
        output_dir: Directory where the file will be written.
        reference_lat: Reference latitude from the PS10 dataset table.
        reference_lon: Reference longitude from the PS10 dataset table.
        prefix: Filename prefix. Default ``'Change_Mask'``.
        compress: GeoTIFF compression. Default ``'lzw'``.

    Returns:
        Resolved absolute path of the written file.

    Raises:
        GeoTIFFExportError: If writing fails.
    """
    filename = format_ps10_filename(prefix, reference_lat, reference_lon, "tif")
    output_path = Path(output_dir) / filename

    logger.info(
        "Exporting PS10 change mask: '%s' (lat=%.4f lon=%.4f).",
        filename, reference_lat, reference_lon,
    )
    return export_change_mask_geotiff(binary_mask, profile, output_path, compress)


# ---------------------------------------------------------------------------
# Stage 2: Classified change raster export
# ---------------------------------------------------------------------------


def export_classified_raster(
    classified_array: np.ndarray,
    profile: RasterProfile,
    output_path: Union[str, Path],
    compress: str = "lzw",
) -> Path:
    """Write a classified change raster to GeoTIFF.

    Pixel values correspond to semantic class IDs defined in
    :data:`~geoai.utils.constants.CLASS_ID_MAP`:
        0 = Unknown / no change
        1 = Kacha_Track
        2 = Building
        3 = Road
        4 = Land_Clearing
        5 = Large_Infrastructure
        6 = New_Settlement

    Args:
        classified_array: uint8 array of shape (H, W) with class ID values.
        profile: rasterio profile from source imagery.
        output_path: Destination file path.
        compress: GeoTIFF compression. Default ``'lzw'``.

    Returns:
        Resolved absolute path of the written file.

    Raises:
        GeoTIFFExportError: If writing fails.
    """
    output_path = Path(output_path)
    write_profile = profile.copy()
    write_profile.update(dtype="uint8", nodata=None)

    logger.info(
        "Exporting classified change raster to '%s' "
        "— unique classes: %s.",
        output_path.name,
        np.unique(classified_array).tolist(),
    )
    try:
        result = write_raster(
            path=output_path,
            array=classified_array.astype(np.uint8),
            profile=write_profile,
            compress=compress,
        )
    except Exception as exc:
        raise GeoTIFFExportError(str(output_path), str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# Probability raster export
# ---------------------------------------------------------------------------


def export_probability_raster(
    probability_map: np.ndarray,
    profile: RasterProfile,
    output_path: Union[str, Path],
    compress: str = "lzw",
) -> Path:
    """Write a prediction probability raster to GeoTIFF.

    Stores P(change) as float32 values in [0, 1]. NaN at invalid pixels.
    Not a PS10 deliverable but useful for internal analysis and dashboards.

    Args:
        probability_map: float32 array of shape (H, W) with P(change) values.
        profile: rasterio profile from source imagery.
        output_path: Destination file path.
        compress: GeoTIFF compression.

    Returns:
        Resolved absolute path of the written file.

    Raises:
        GeoTIFFExportError: If writing fails.
    """
    output_path = Path(output_path)
    write_profile = profile.copy()
    write_profile.update(dtype="float32", nodata=float("nan"))

    logger.info(
        "Exporting probability raster to '%s'.", output_path.name
    )
    try:
        result = write_raster(
            path=output_path,
            array=probability_map.astype(np.float32),
            profile=write_profile,
            compress=compress,
        )
    except Exception as exc:
        raise GeoTIFFExportError(str(output_path), str(exc)) from exc

    return result
