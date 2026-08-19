"""
Raster alignment, reprojection, and resampling utilities for the GeoAI Platform.

Provides functions to verify and enforce spatial alignment between rasters from
different sensors or processing chains. In Version 1, all rasters are already
co-registered (alignment is performed externally before platform ingestion).
This module validates that assumption and provides resampling tools for future
multi-sensor workflows where alignment cannot be assumed.

Single responsibility: raster spatial alignment and reprojection.

Position in dependency hierarchy: preprocessing (depends on core, utils).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

from geoai.core.exceptions import RasterIOError, RasterNotFoundError, RasterValidationError
from geoai.core.io import RasterProfile, read_raster_metadata
from geoai.core.validation import validate_raster_compatibility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alignment verification
# ---------------------------------------------------------------------------


def assert_rasters_aligned(
    raster_paths: Dict[str, Union[str, Path]],
    reference_key: Optional[str] = None,
) -> None:
    """Assert that all provided rasters are spatially aligned.

    This is a convenience wrapper around
    :func:`~geoai.core.validation.validate_raster_compatibility` that raises
    a clear error if any raster fails the alignment check. It must be called
    at the start of every pipeline stage that loads multiple rasters.

    In Version 1, all input rasters are pre-aligned by external tools. This
    function verifies that assumption is satisfied before processing begins.

    Args:
        raster_paths: Dictionary mapping descriptive names to file paths.
        reference_key: Key of the raster to use as the spatial reference.
            If None, the first key is used.

    Raises:
        RasterNotFoundError: If any file does not exist.
        CRSMismatchError: If CRS compatibility fails.
        SpatialDimensionMismatchError: If dimensions differ.
        TransformMismatchError: If affine transforms differ.
    """
    logger.info(
        "Verifying spatial alignment of %d rasters.", len(raster_paths)
    )
    validate_raster_compatibility(raster_paths, reference_key=reference_key)
    logger.info("All rasters confirmed as spatially aligned.")


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------


def reproject_to_match(
    source_path: Union[str, Path],
    reference_path: Union[str, Path],
    output_path: Union[str, Path],
    resampling_method: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject and resample a raster to match a reference raster's CRS,
    extent, and resolution.

    Used when a raster from a different sensor or processing chain does not
    align spatially with the reference raster. The output will have the same
    CRS, affine transform, and pixel dimensions as the reference.

    Note: In Version 1, this function is not called because all rasters are
    pre-aligned. It is implemented here for future multi-sensor workflows
    (e.g., aligning LISS-IV 5.8m data with Sentinel-2 10m data).

    Args:
        source_path: Path to the raster to reproject.
        reference_path: Path to the reference raster whose spatial properties
            will be matched.
        output_path: Path where the reprojected raster will be written.
        resampling_method: Resampling algorithm to use. Default is bilinear
            interpolation, which is appropriate for continuous-valued rasters
            (spectral indices, SAR backscatter). Use ``Resampling.nearest``
            for categorical data.

    Returns:
        Resolved absolute path of the written reprojected raster.

    Raises:
        RasterNotFoundError: If source or reference files do not exist.
        RasterIOError: If any file cannot be opened or the reproject fails.
    """
    source_path = Path(source_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise RasterNotFoundError(str(source_path))
    if not reference_path.exists():
        raise RasterNotFoundError(str(reference_path))

    logger.info(
        "Reprojecting '%s' to match '%s'.",
        source_path.name,
        reference_path.name,
    )

    try:
        with rasterio.open(reference_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height

        with rasterio.open(source_path) as src:
            src_profile = src.profile.copy()
            dst_profile = src_profile.copy()
            dst_profile.update(
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                driver="GTiff",
                compress="lzw",
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **dst_profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling_method,
                    )
    except rasterio.errors.RasterioIOError as exc:
        raise RasterIOError(
            f"Reprojection failed for '{source_path}': {exc}."
        ) from exc

    logger.info("Reprojection complete: '%s'.", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_to_resolution(
    source_path: Union[str, Path],
    target_resolution_m: float,
    output_path: Union[str, Path],
    resampling_method: Resampling = Resampling.bilinear,
) -> Path:
    """Resample a raster to a target ground resolution in metres.

    Used to harmonise rasters from sensors with different native resolutions
    (e.g., upsampling LISS-IV from 5.8m to 10m to match Sentinel-2).

    Args:
        source_path: Path to the raster to resample.
        target_resolution_m: Target ground sampling distance in metres.
        output_path: Path where the resampled raster will be written.
        resampling_method: Resampling algorithm. Default is bilinear.

    Returns:
        Resolved absolute path of the written resampled raster.

    Raises:
        RasterNotFoundError: If the source file does not exist.
        RasterIOError: If the file cannot be processed.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise RasterNotFoundError(str(source_path))

    logger.info(
        "Resampling '%s' to %.1fm resolution.", source_path.name, target_resolution_m
    )

    try:
        with rasterio.open(source_path) as src:
            native_res = src.res[0]
            scale_factor = native_res / target_resolution_m
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)

            new_transform = src.transform * src.transform.scale(
                src.width / new_width,
                src.height / new_height,
            )

            dst_profile = src.profile.copy()
            dst_profile.update(
                width=new_width,
                height=new_height,
                transform=new_transform,
                driver="GTiff",
                compress="lzw",
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **dst_profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=new_transform,
                        dst_crs=src.crs,
                        resampling=resampling_method,
                    )
    except rasterio.errors.RasterioIOError as exc:
        raise RasterIOError(
            f"Resampling failed for '{source_path}': {exc}."
        ) from exc

    logger.info(
        "Resampling complete: '%s' (%dx%d → %dx%d).",
        output_path,
        src.width, src.height,
        new_width, new_height,
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def get_common_extent(
    raster_paths: Dict[str, Union[str, Path]],
) -> Tuple[float, float, float, float]:
    """Compute the spatial intersection of all provided rasters.

    Returns the bounding box of the area covered by all rasters. Useful for
    clipping rasters to a common extent before processing.

    Args:
        raster_paths: Dictionary mapping names to file paths.

    Returns:
        Tuple of ``(west, south, east, north)`` in the CRS of the first raster.

    Raises:
        RasterNotFoundError: If any file does not exist.
        RasterValidationError: If rasters do not share the same CRS.
    """
    profiles = validate_raster_compatibility(raster_paths)

    bounds_list = [p["bounds"] for p in profiles.values()]
    west = max(b.left for b in bounds_list)
    south = max(b.bottom for b in bounds_list)
    east = min(b.right for b in bounds_list)
    north = min(b.top for b in bounds_list)

    if west >= east or south >= north:
        raise RasterValidationError(
            "Rasters have no spatial overlap. "
            "Verify that all rasters cover the same geographic extent."
        )

    logger.info(
        "Common extent computed: west=%.6f south=%.6f east=%.6f north=%.6f.",
        west, south, east, north,
    )
    return west, south, east, north
