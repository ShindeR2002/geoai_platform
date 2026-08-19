"""
Geometric conversion utilities for the GeoAI Platform.

Converts pixel-space object properties to geographic coordinates and polygon
geometries suitable for shapefile and GeoJSON export. All coordinate
conversions use the raster affine transform from the prediction raster profile.

Single responsibility: convert pixel coordinates and bounding boxes to
geographic geometries.

Position in dependency hierarchy: analysis (depends on core, utils).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from geoai.utils.geo import centroid_to_geo, pixel_bbox_to_geo, pixel_to_geo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centroid conversion
# ---------------------------------------------------------------------------


def region_centroid_to_geo(
    region,
    transform: Affine,
) -> Tuple[Optional[float], Optional[float]]:
    """Convert a region's pixel centroid to geographic coordinates.

    Args:
        region: A skimage RegionProperties object.
        transform: Affine transform from the prediction raster profile.

    Returns:
        Tuple of (longitude, latitude) in the raster's CRS.
        Returns (None, None) if conversion fails.
    """
    centroid_row, centroid_col = region.centroid
    try:
        lon, lat = centroid_to_geo(transform, centroid_row, centroid_col)
        return float(lon), float(lat)
    except Exception as exc:
        logger.warning(
            "Could not convert centroid of object %d to geo: %s.",
            region.label, exc,
        )
        return None, None


# ---------------------------------------------------------------------------
# Bounding box polygon construction
# ---------------------------------------------------------------------------


def bbox_to_polygon(
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int,
    transform: Affine,
) -> Optional[Polygon]:
    """Build a geographic bounding box polygon from pixel coordinates.

    Args:
        min_row: Top row of the bounding box.
        min_col: Left column of the bounding box.
        max_row: Bottom row of the bounding box.
        max_col: Right column of the bounding box.
        transform: Affine transform from the prediction raster profile.

    Returns:
        A shapely Polygon representing the geographic bounding box.
        Returns None if coordinate conversion fails.
    """
    try:
        west, south, east, north = pixel_bbox_to_geo(
            transform, min_row, min_col, max_row, max_col
        )
        return Polygon([
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south),
        ])
    except Exception as exc:
        logger.warning("Could not build bounding box polygon: %s.", exc)
        return None


# ---------------------------------------------------------------------------
# Rasterio vectorisation (for export)
# ---------------------------------------------------------------------------


def vectorise_mask(
    binary_mask: np.ndarray,
    transform: Affine,
    crs_wkt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Vectorise a binary mask into a list of GeoJSON-style feature dictionaries.

    Converts connected regions of value 1 in the binary mask to polygon
    geometries using rasterio's ``features.shapes`` function. Each polygon
    represents one connected change region (before area filtering).

    This function is used by the shapefile and GeoJSON export modules to
    convert the significant change mask to vector format.

    Args:
        binary_mask: uint8 binary array of shape (H, W).
        transform: Affine transform from the prediction raster profile.
        crs_wkt: Optional CRS Well-Known Text string for metadata only.

    Returns:
        List of dictionaries, each with keys:
            - ``geometry``: Shapely geometry object (Polygon or MultiPolygon).
            - ``properties``: Dictionary with attribute fields (currently empty
              at this stage — attributes are added by the export modules).
    """
    import rasterio.features

    features = []
    for geom_dict, value in rasterio.features.shapes(
        binary_mask.astype(np.int32),
        mask=(binary_mask > 0).astype(np.uint8),
        transform=transform,
    ):
        if value == 1:
            geom = shape(geom_dict)
            if geom.is_valid and not geom.is_empty:
                features.append({
                    "geometry": geom,
                    "properties": {},
                })

    logger.info(
        "Vectorisation complete — %d polygon feature(s) extracted.",
        len(features),
    )
    return features


def merge_overlapping_polygons(features: List[Dict]) -> List[Dict]:
    """Merge overlapping or adjacent polygons from vectorisation.

    When the binary mask is vectorised pixel-by-pixel, adjacent pixels may
    produce separate but touching polygons. This function merges them into
    single geometries using shapely's unary_union.

    Args:
        features: Feature list from :func:`vectorise_mask`.

    Returns:
        New feature list with overlapping polygons merged.
    """
    if not features:
        return []

    geometries = [f["geometry"] for f in features]
    merged = unary_union(geometries)

    if merged.geom_type == "Polygon":
        result = [{"geometry": merged, "properties": {}}]
    elif merged.geom_type == "MultiPolygon":
        result = [{"geometry": g, "properties": {}} for g in merged.geoms]
    else:
        result = [{"geometry": merged, "properties": {}}]

    logger.info(
        "Polygon merge — %d → %d feature(s).",
        len(features), len(result),
    )
    return result
