"""
CSV export for object statistics in the GeoAI Platform.

Writes per-object attribute tables and class-wise summary statistics to CSV
files for analysis, reporting, and downstream processing.

Single responsibility: write object records and statistics dictionaries to CSV.

Position in dependency hierarchy: exports (depends on classification/schema).
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from geoai.classification.schema import ChangeObject
from geoai.core.exceptions import ExportError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-object CSV export
# ---------------------------------------------------------------------------


def export_objects_csv(
    objects: List[ChangeObject],
    output_path: Union[str, Path],
) -> Path:
    """Write per-object attributes to a CSV file.

    Exports all ChangeObject export attributes as a flat CSV table, one
    row per object. Includes spatial, spectral, and semantic attributes.

    Args:
        objects: List of ChangeObject instances (Stage 1 or Stage 2).
        output_path: Destination CSV file path.

    Returns:
        Resolved absolute path of the written CSV file.

    Raises:
        ExportError: If writing fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not objects:
        logger.warning(
            "export_objects_csv: no objects to export. Writing empty CSV."
        )

    rows = [obj.to_export_dict() for obj in objects]
    fieldnames = list(rows[0].keys()) if rows else [
        "object_id", "area_px", "area_m2", "centroid_lat", "centroid_lon",
        "class", "confidence", "ndvi_chg", "ndbi_chg", "ndwi_chg",
        "sar_chg", "compact", "perimeter", "method",
    ]

    logger.info(
        "Exporting %d objects to CSV: '%s'.", len(objects), output_path.name
    )

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        raise ExportError(
            f"CSV export to '{output_path}' failed: {exc}."
        ) from exc

    logger.info("Object CSV written: '%s' — %d rows.", output_path, len(rows))
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Class statistics CSV export
# ---------------------------------------------------------------------------


def export_class_statistics_csv(
    class_stats: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Write class-wise statistics to a CSV file.

    Exports the output of
    :func:`~geoai.classification.spatial_stats.compute_class_statistics`
    as a flat CSV table, one row per semantic class.

    Args:
        class_stats: Dictionary mapping class name to statistics dict.
        output_path: Destination CSV file path.

    Returns:
        Resolved absolute path of the written CSV file.

    Raises:
        ExportError: If writing fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build rows by prepending class name to the stats dictionary.
    rows = []
    for cls_name, stats in class_stats.items():
        row = {"class": cls_name}
        row.update(stats)
        rows.append(row)

    if not rows:
        logger.warning("export_class_statistics_csv: no class stats to export.")
        return output_path.resolve()

    fieldnames = list(rows[0].keys())

    logger.info(
        "Exporting class statistics to CSV: '%s' — %d classes.",
        output_path.name, len(rows),
    )

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        raise ExportError(
            f"Class statistics CSV export to '{output_path}' failed: {exc}."
        ) from exc

    logger.info("Class statistics CSV written: '%s'.", output_path)
    return output_path.resolve()
