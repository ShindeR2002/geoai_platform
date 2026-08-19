"""
Object statistics computation for the GeoAI Platform.

Computes per-object and summary statistics from the list of significant
change objects extracted by :mod:`geoai.analysis.objects`. Produces
structured dictionaries and DataFrames for export and reporting.

Single responsibility: compute statistics from RegionProperties lists.

Position in dependency hierarchy: analysis (depends on core, utils, analysis/objects).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from geoai.utils.constants import DEFAULT_MIN_OBJECT_SIZE_PX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-object record construction
# ---------------------------------------------------------------------------


def build_object_records(
    significant_regions,
    resolution_m: float = 10.0,
    transform=None,
) -> List[Dict[str, Any]]:
    """Build a list of per-object attribute dictionaries from region properties.

    Each dictionary contains the geometric and statistical attributes for one
    significant change object. Geographic coordinates are included when a
    raster affine transform is provided.

    Args:
        significant_regions: List of RegionProperties from
            :func:`~geoai.analysis.objects.extract_objects`.
        resolution_m: Ground sampling distance in metres. Used to compute
            area in square metres. Default 10.0 (Sentinel-2).
        transform: Optional rasterio affine transform for geographic
            coordinate conversion. When provided, centroid_lon and
            centroid_lat fields are populated.

    Returns:
        List of dictionaries, one per object. Each dictionary contains:
            - ``object_id``: Integer label from the label array.
            - ``area_px``: Object area in pixels.
            - ``area_m2``: Object area in square metres.
            - ``centroid_row``: Centroid row (pixel coordinate).
            - ``centroid_col``: Centroid column (pixel coordinate).
            - ``centroid_lon``: Geographic longitude (float or None).
            - ``centroid_lat``: Geographic latitude (float or None).
            - ``bbox_min_row``: Bounding box top row.
            - ``bbox_min_col``: Bounding box left column.
            - ``bbox_max_row``: Bounding box bottom row.
            - ``bbox_max_col``: Bounding box right column.
            - ``perimeter``: Object perimeter in pixels.
            - ``compactness``: Compactness = 4π × area / perimeter².
              Values near 1 indicate circular objects; low values indicate
              elongated or complex shapes.
    """
    from geoai.utils.geo import centroid_to_geo

    records = []
    pixel_area_m2 = resolution_m ** 2

    for region in significant_regions:
        centroid_row, centroid_col = region.centroid

        # Geographic coordinates — only computed when transform is available.
        centroid_lon, centroid_lat = None, None
        if transform is not None:
            try:
                centroid_lon, centroid_lat = centroid_to_geo(
                    transform, centroid_row, centroid_col
                )
            except Exception as exc:
                logger.warning(
                    "Could not convert centroid of object %d to geographic "
                    "coordinates: %s.",
                    region.label, exc,
                )

        # Bounding box: (min_row, min_col, max_row, max_col) in skimage convention.
        min_row, min_col, max_row, max_col = region.bbox

        # Perimeter and compactness.
        perimeter = float(region.perimeter)
        if perimeter > 0:
            compactness = (4.0 * np.pi * region.area) / (perimeter ** 2)
        else:
            compactness = 0.0

        records.append({
            "object_id": int(region.label),
            "area_px": int(region.area),
            "area_m2": float(region.area * pixel_area_m2),
            "centroid_row": float(centroid_row),
            "centroid_col": float(centroid_col),
            "centroid_lon": float(centroid_lon) if centroid_lon is not None else None,
            "centroid_lat": float(centroid_lat) if centroid_lat is not None else None,
            "bbox_min_row": int(min_row),
            "bbox_min_col": int(min_col),
            "bbox_max_row": int(max_row),
            "bbox_max_col": int(max_col),
            "perimeter": round(perimeter, 2),
            "compactness": round(float(compactness), 6),
        })

    logger.info(
        "Object records built — %d objects, resolution=%.1fm.",
        len(records), resolution_m,
    )
    return records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_summary_statistics(
    object_records: List[Dict[str, Any]],
    resolution_m: float = 10.0,
) -> Dict[str, Any]:
    """Compute summary statistics across all significant change objects.

    Args:
        object_records: List from :func:`build_object_records`.
        resolution_m: Ground sampling distance in metres.

    Returns:
        Dictionary containing:
            - ``object_count``: Total number of significant objects.
            - ``total_change_px``: Sum of all object areas in pixels.
            - ``total_change_m2``: Total change area in square metres.
            - ``total_change_ha``: Total change area in hectares.
            - ``mean_area_px``: Mean object area in pixels.
            - ``median_area_px``: Median object area in pixels.
            - ``std_area_px``: Standard deviation of object area in pixels.
            - ``min_area_px``: Smallest object area in pixels.
            - ``max_area_px``: Largest object area in pixels.
            - ``mean_compactness``: Mean object compactness score.
    """
    if not object_records:
        logger.warning(
            "Summary statistics requested but object_records is empty. "
            "Returning zero-value summary."
        )
        return {
            "object_count": 0,
            "total_change_px": 0,
            "total_change_m2": 0.0,
            "total_change_ha": 0.0,
            "mean_area_px": 0.0,
            "median_area_px": 0.0,
            "std_area_px": 0.0,
            "min_area_px": 0,
            "max_area_px": 0,
            "mean_compactness": 0.0,
        }

    areas = np.array([r["area_px"] for r in object_records])
    compactness = np.array([r["compactness"] for r in object_records])
    total_px = int(areas.sum())
    total_m2 = float(total_px * (resolution_m ** 2))

    summary = {
        "object_count": len(object_records),
        "total_change_px": total_px,
        "total_change_m2": round(total_m2, 2),
        "total_change_ha": round(total_m2 / 10_000.0, 4),
        "mean_area_px": round(float(areas.mean()), 2),
        "median_area_px": round(float(np.median(areas)), 2),
        "std_area_px": round(float(areas.std()), 2),
        "min_area_px": int(areas.min()),
        "max_area_px": int(areas.max()),
        "mean_compactness": round(float(compactness.mean()), 6),
    }

    logger.info(
        "Summary statistics — objects=%d total_area=%.2f ha "
        "mean_size=%.1f px max_size=%d px.",
        summary["object_count"],
        summary["total_change_ha"],
        summary["mean_area_px"],
        summary["max_area_px"],
    )
    return summary
