"""
Geographic coordinate utilities for the GeoAI Platform.

Provides helper functions for converting between pixel (row, column) coordinates
and geographic (latitude, longitude) coordinates, formatting coordinate strings
for output filenames, and performing basic CRS operations.

Single responsibility: geographic coordinate conversion and formatting.

Position in dependency hierarchy: utils (depends on constants only).
"""

import logging
from typing import Optional, Tuple

import numpy as np
from rasterio.transform import Affine, xy as transform_xy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel ↔ Geographic coordinate conversion
# ---------------------------------------------------------------------------


def pixel_to_geo(
    transform: Affine,
    row: float,
    col: float,
    offset: str = "center",
) -> Tuple[float, float]:
    """Convert a pixel (row, column) coordinate to geographic (lon, lat).

    Uses the rasterio ``transform_xy`` function to apply the raster affine
    transform. The returned coordinates are in the CRS of the raster from
    which the transform was derived.

    Args:
        transform: Affine transform of the raster (from ``raster.profile['transform']``).
        row: Row index (0-indexed) in the raster array.
        col: Column index (0-indexed) in the raster array.
        offset: Which part of the pixel to reference. ``'center'`` returns
            the pixel centre coordinate; ``'ul'`` returns the upper-left
            corner.

    Returns:
        Tuple of ``(longitude, latitude)`` in the raster's native CRS.
        For EPSG:4326 this is geographic longitude and latitude in decimal
        degrees; for projected CRS it is easting and northing in the
        projection's linear unit.
    """
    lon, lat = transform_xy(transform, row, col, offset=offset)
    return float(lon), float(lat)


def geo_to_pixel(
    transform: Affine,
    lon: float,
    lat: float,
) -> Tuple[int, int]:
    """Convert a geographic (lon, lat) coordinate to pixel (row, col).

    Inverts the affine transform to find the pixel that contains the given
    geographic coordinate.

    Args:
        transform: Affine transform of the raster.
        lon: Longitude or easting in the raster's native CRS.
        lat: Latitude or northing in the raster's native CRS.

    Returns:
        Tuple of ``(row, col)`` as integer pixel indices (0-indexed).
        Values may be outside the raster bounds if the coordinate lies
        outside the raster extent.
    """
    inv_transform = ~transform
    col_f, row_f = inv_transform * (lon, lat)
    return int(round(row_f)), int(round(col_f))


def centroid_to_geo(
    transform: Affine,
    centroid_row: float,
    centroid_col: float,
) -> Tuple[float, float]:
    """Convert an object centroid in pixel coordinates to geographic coordinates.

    Convenience wrapper around :func:`pixel_to_geo` for use in the object
    analysis and export modules.

    Args:
        transform: Affine transform from the prediction raster's profile.
        centroid_row: Row coordinate of the object centroid (may be fractional).
        centroid_col: Column coordinate of the object centroid (may be fractional).

    Returns:
        Tuple of ``(longitude, latitude)`` in the raster's native CRS.
    """
    return pixel_to_geo(transform, centroid_row, centroid_col, offset="center")


# ---------------------------------------------------------------------------
# Bounding box conversion
# ---------------------------------------------------------------------------


def pixel_bbox_to_geo(
    transform: Affine,
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int,
) -> Tuple[float, float, float, float]:
    """Convert pixel bounding box corners to geographic coordinates.

    Args:
        transform: Affine transform of the raster.
        min_row: Top row of the bounding box (smallest row index).
        min_col: Left column of the bounding box (smallest column index).
        max_row: Bottom row of the bounding box (largest row index).
        max_col: Right column of the bounding box (largest column index).

    Returns:
        Tuple of ``(west, south, east, north)`` geographic coordinates in
        the raster's native CRS. For geographic CRS this maps to
        ``(min_lon, min_lat, max_lon, max_lat)``.
    """
    # Upper-left corner of the bounding box
    west, north = pixel_to_geo(transform, min_row, min_col, offset="ul")
    # Lower-right corner of the bounding box (use center of the max pixel)
    east, south = pixel_to_geo(transform, max_row, max_col, offset="center")
    return float(west), float(south), float(east), float(north)


# ---------------------------------------------------------------------------
# Coordinate formatting for PS10 filename convention
# ---------------------------------------------------------------------------


def format_lat_lon_for_filename(lat: float, lon: float, decimals: int = 4) -> str:
    """Format latitude and longitude as a string suitable for PS10 filenames.

    PS10 output filenames follow the convention ``Change_Mask_{Lat}_{Long}.tif``
    where Lat and Long are from the "Reference Location" column of the dataset
    table. This function formats coordinate values consistently.

    Args:
        lat: Latitude in decimal degrees (positive = North).
        lon: Longitude in decimal degrees (positive = East).
        decimals: Number of decimal places to include. Default 4.

    Returns:
        String of the form ``'{lat}_{lon}'`` with the specified precision,
        e.g. ``'28.1740_77.6126'`` for the Urban test location.
    """
    lat_str = f"{lat:.{decimals}f}"
    lon_str = f"{lon:.{decimals}f}"
    return f"{lat_str}_{lon_str}"


def format_ps10_filename(
    prefix: str,
    lat: float,
    lon: float,
    extension: str,
    decimals: int = 4,
) -> str:
    """Construct a PS10-convention output filename.

    Args:
        prefix: Filename prefix (e.g. ``'Change_Mask'``).
        lat: Reference latitude in decimal degrees.
        lon: Reference longitude in decimal degrees.
        extension: File extension without dot (e.g. ``'tif'``, ``'shp'``).
        decimals: Number of decimal places in the coordinate strings.

    Returns:
        Filename string, e.g. ``'Change_Mask_28.1740_77.6126.tif'``.
    """
    coord_str = format_lat_lon_for_filename(lat, lon, decimals)
    return f"{prefix}_{coord_str}.{extension}"


# ---------------------------------------------------------------------------
# Raster extent utilities
# ---------------------------------------------------------------------------


def get_raster_centre(
    transform: Affine,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Compute the geographic centre of a raster.

    Args:
        transform: Affine transform of the raster.
        width: Raster width in pixels.
        height: Raster height in pixels.

    Returns:
        Tuple of ``(longitude, latitude)`` at the geographic centre of the
        raster extent.
    """
    centre_row = height / 2.0
    centre_col = width / 2.0
    return pixel_to_geo(transform, centre_row, centre_col)


def estimate_area_m2(
    pixel_count: int,
    resolution_m: float,
) -> float:
    """Estimate the geographic area represented by a pixel count.

    Assumes square pixels with side length ``resolution_m`` metres.

    Args:
        pixel_count: Number of pixels in the object or region.
        resolution_m: Ground sampling distance in metres per pixel.

    Returns:
        Estimated area in square metres.
    """
    return float(pixel_count) * (resolution_m ** 2)
