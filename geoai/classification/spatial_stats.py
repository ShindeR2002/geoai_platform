"""
Class-wise spatial statistics for the GeoAI Platform Stage 2 pipeline.

Aggregates classified ChangeObject instances into per-class summary statistics
used for the Stage 2 report, dashboard layers, and classified shapefile
attribute tables.

Single responsibility: aggregate per-class statistics from classified objects.

Position in dependency hierarchy: classification (depends on classification/schema,
utils/constants).
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from geoai.classification.schema import ChangeObject
from geoai.utils.constants import ChangeClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-class statistics
# ---------------------------------------------------------------------------


def compute_class_statistics(
    objects: List[ChangeObject],
    resolution_m: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-class statistics across all classified change objects.

    Groups objects by their assigned semantic class and computes aggregate
    area, count, and spectral statistics for each class.

    Args:
        objects: List of classified ChangeObject instances.
        resolution_m: Ground sampling distance in metres. Used to compute
            area in square metres and hectares. Default 10.0.

    Returns:
        Dictionary mapping class name strings to statistics dictionaries.
        Each class entry contains:
            - ``count``: Number of objects in this class.
            - ``total_area_px``: Total area in pixels.
            - ``total_area_m2``: Total area in square metres.
            - ``total_area_ha``: Total area in hectares.
            - ``mean_area_px``: Mean object area in pixels.
            - ``max_area_px``: Largest object area in pixels.
            - ``mean_confidence``: Mean classification confidence.
            - ``mean_ndvi_change``: Mean spectral NDVI change.
            - ``mean_ndbi_change``: Mean spectral NDBI change.
            - ``mean_sar_change``: Mean spectral SAR change.
    """
    pixel_area_m2 = resolution_m ** 2

    # Initialise empty stats for all known classes.
    class_buckets: Dict[str, List[ChangeObject]] = defaultdict(list)
    for obj in objects:
        class_buckets[obj.semantic_class].append(obj)

    results = {}
    for cls_name in [c.value for c in ChangeClass]:
        bucket = class_buckets.get(cls_name, [])
        if not bucket:
            results[cls_name] = {
                "count": 0,
                "total_area_px": 0,
                "total_area_m2": 0.0,
                "total_area_ha": 0.0,
                "mean_area_px": 0.0,
                "max_area_px": 0,
                "mean_confidence": 0.0,
                "mean_ndvi_change": None,
                "mean_ndbi_change": None,
                "mean_sar_change": None,
            }
            continue

        areas = np.array([o.area_px for o in bucket])
        confidences = np.array([o.confidence for o in bucket])
        ndvi_vals = [o.mean_ndvi_change for o in bucket if o.mean_ndvi_change is not None]
        ndbi_vals = [o.mean_ndbi_change for o in bucket if o.mean_ndbi_change is not None]
        sar_vals = [o.mean_sar_change for o in bucket if o.mean_sar_change is not None]

        total_px = int(areas.sum())
        total_m2 = float(total_px * pixel_area_m2)

        results[cls_name] = {
            "count": len(bucket),
            "total_area_px": total_px,
            "total_area_m2": round(total_m2, 2),
            "total_area_ha": round(total_m2 / 10_000.0, 4),
            "mean_area_px": round(float(areas.mean()), 2),
            "max_area_px": int(areas.max()),
            "mean_confidence": round(float(confidences.mean()), 4),
            "mean_ndvi_change": round(float(np.mean(ndvi_vals)), 6) if ndvi_vals else None,
            "mean_ndbi_change": round(float(np.mean(ndbi_vals)), 6) if ndbi_vals else None,
            "mean_sar_change": round(float(np.mean(sar_vals)), 6) if sar_vals else None,
        }

    # Log only classes with at least one object.
    active = {k: v for k, v in results.items() if v["count"] > 0}
    for cls_name, stats in active.items():
        logger.info(
            "Class '%s': count=%d total_area=%.2f ha mean_conf=%.4f.",
            cls_name, stats["count"], stats["total_area_ha"], stats["mean_confidence"],
        )

    if not active:
        logger.warning("No classified objects found in any class.")

    return results


def get_class_distribution(objects: List[ChangeObject]) -> Dict[str, int]:
    """Return a simple count of objects per class.

    Args:
        objects: List of classified ChangeObject instances.

    Returns:
        Dictionary mapping class name to object count. Only classes with
        at least one object are included.
    """
    distribution: Dict[str, int] = {}
    for obj in objects:
        distribution[obj.semantic_class] = distribution.get(obj.semantic_class, 0) + 1
    return distribution


def format_class_statistics_report(
    class_stats: Dict[str, Dict[str, Any]],
    aoi_name: str = "",
) -> str:
    """Format class statistics as a human-readable text report section.

    Args:
        class_stats: Dictionary from :func:`compute_class_statistics`.
        aoi_name: Optional AOI name for the report header.

    Returns:
        Multi-line string suitable for inclusion in the Stage 2 report.
    """
    header = "Stage 2 — Class-wise Statistics"
    if aoi_name:
        header += f" [{aoi_name}]"
    separator = "-" * 60

    lines = [separator, header, separator]
    total_objects = sum(s["count"] for s in class_stats.values())
    lines.append(f"Total significant objects: {total_objects}")
    lines.append("")

    for cls_name, stats in class_stats.items():
        if stats["count"] == 0:
            continue
        lines.append(f"  {cls_name}:")
        lines.append(f"    Count         : {stats['count']}")
        lines.append(f"    Total Area    : {stats['total_area_ha']:.4f} ha "
                     f"({stats['total_area_px']} px)")
        lines.append(f"    Mean Area     : {stats['mean_area_px']:.1f} px")
        lines.append(f"    Mean Conf.    : {stats['mean_confidence']:.4f}")
        if stats["mean_ndvi_change"] is not None:
            lines.append(f"    ΔNDVI (mean)  : {stats['mean_ndvi_change']:.4f}")
        if stats["mean_ndbi_change"] is not None:
            lines.append(f"    ΔNDBI (mean)  : {stats['mean_ndbi_change']:.4f}")
        if stats["mean_sar_change"] is not None:
            lines.append(f"    ΔSAR  (mean)  : {stats['mean_sar_change']:.4f}")
        lines.append("")

    lines.append(separator)
    return "\n".join(lines)
