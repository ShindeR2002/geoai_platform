"""
Report generation for the GeoAI Platform.

Generates structured plain-text and Markdown reports summarising the outputs
of Stage 1 (change detection) and Stage 2 (object classification) pipeline
runs. Reports are written to the run output directory and are human-readable
standalone documents.

Single responsibility: format and write pipeline run reports.

Position in dependency hierarchy: exports (depends on analysis, classification,
models/evaluation).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from geoai.classification.schema import ChangeObject
from geoai.classification.spatial_stats import format_class_statistics_report
from geoai.models.evaluation import format_evaluation_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 report
# ---------------------------------------------------------------------------


def generate_stage1_report(
    aoi_name: str,
    run_id: str,
    prediction_summary: Dict[str, Any],
    object_summary: Dict[str, Any],
    evaluation_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a Stage 1 change detection run report.

    Args:
        aoi_name: Name of the AOI (e.g. 'Dholera').
        run_id: Unique run identifier.
        prediction_summary: Dictionary from
            :func:`~geoai.models.inference.summarise_prediction`.
        object_summary: Dictionary from
            :func:`~geoai.analysis.statistics.compute_summary_statistics`.
        evaluation_results: Optional dictionary from
            :func:`~geoai.models.evaluation.evaluate_predictions` (only
            available when ground truth labels exist).
        output_path: Optional path to write the report. If None, the report
            is returned as a string but not written to disk.

    Returns:
        Report as a formatted string.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    sep = "=" * 60

    lines = [
        sep,
        "GeoAI Platform — Stage 1 Change Detection Report",
        sep,
        f"AOI            : {aoi_name}",
        f"Run ID         : {run_id}",
        f"Generated      : {now}",
        "",
        "--- Prediction Summary ---",
        f"Total pixels   : {prediction_summary.get('total_pixels', 'N/A'):,}",
        f"Valid pixels   : {prediction_summary.get('valid_pixels', 'N/A'):,}",
        f"Change pixels  : {prediction_summary.get('change_pixels', 'N/A'):,}",
        f"Change fraction: {prediction_summary.get('change_fraction', 0):.4f} "
        f"({prediction_summary.get('change_fraction', 0)*100:.2f}%)",
        f"Change area    : {prediction_summary.get('change_area_m2', 0):,.1f} m² "
        f"({prediction_summary.get('change_area_ha', 0):.2f} ha)",
        "",
        "--- Object Analysis Summary ---",
        f"Significant objects : {object_summary.get('object_count', 0)}",
        f"Total object area   : {object_summary.get('total_change_ha', 0):.4f} ha",
        f"Mean object size    : {object_summary.get('mean_area_px', 0):.1f} px",
        f"Largest object      : {object_summary.get('max_area_px', 0)} px",
        f"Smallest object     : {object_summary.get('min_area_px', 0)} px",
        f"Mean compactness    : {object_summary.get('mean_compactness', 0):.4f}",
    ]

    if evaluation_results:
        lines += [
            "",
            format_evaluation_report(evaluation_results, aoi_name=aoi_name),
        ]

    lines += ["", sep, "End of Stage 1 Report", sep]
    report = "\n".join(lines)

    if output_path is not None:
        _write_report(report, output_path)

    return report


# ---------------------------------------------------------------------------
# Stage 2 report
# ---------------------------------------------------------------------------


def generate_stage2_report(
    aoi_name: str,
    run_id: str,
    objects: List[ChangeObject],
    class_stats: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a Stage 2 object classification run report.

    Args:
        aoi_name: Name of the AOI.
        run_id: Unique run identifier.
        objects: List of classified ChangeObject instances.
        class_stats: Dictionary from
            :func:`~geoai.classification.spatial_stats.compute_class_statistics`.
        output_path: Optional path to write the report.

    Returns:
        Report as a formatted string.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    sep = "=" * 60

    lines = [
        sep,
        "GeoAI Platform — Stage 2 Object Classification Report",
        sep,
        f"AOI            : {aoi_name}",
        f"Run ID         : {run_id}",
        f"Generated      : {now}",
        f"Total objects  : {len(objects)}",
        "",
        format_class_statistics_report(class_stats, aoi_name=aoi_name),
        "",
        "--- Classifier Information ---",
    ]

    methods = set(o.classifier_method for o in objects)
    for method in methods:
        count = sum(1 for o in objects if o.classifier_method == method)
        lines.append(f"  {method}: {count} object(s)")

    conf_values = [o.confidence for o in objects]
    if conf_values:
        import numpy as np
        lines += [
            "",
            "--- Confidence Summary ---",
            f"  Mean confidence : {float(np.mean(conf_values)):.4f}",
            f"  Min confidence  : {float(np.min(conf_values)):.4f}",
            f"  Max confidence  : {float(np.max(conf_values)):.4f}",
        ]

    lines += ["", sep, "End of Stage 2 Report", sep]
    report = "\n".join(lines)

    if output_path is not None:
        _write_report(report, output_path)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_report(report: str, output_path: Union[str, Path]) -> Path:
    """Write a report string to a text file.

    Args:
        report: Report content string.
        output_path: Destination file path.

    Returns:
        Resolved absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report written to '%s'.", output_path)
    return output_path.resolve()
