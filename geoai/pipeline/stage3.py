"""
Stage 3 operational platform pipeline for the GeoAI Platform.

Orchestrates batch multi-AOI processing, project management, and aggregated
reporting. Stage 3 transforms the single-AOI Stage 1 and Stage 2 pipelines
into a repeatable operational system capable of processing all configured
AOIs in sequence and producing a project-level summary.

This module implements the production deployment layer: configuration-driven
batch execution, run history tracking, and aggregated statistics — without
requiring any source-code changes when new AOIs are added.

Single responsibility: orchestrate batch multi-AOI Stage 1 + Stage 2 runs
and produce project-level aggregated outputs.

Position in dependency hierarchy: pipeline (depends on pipeline/stage1,
pipeline/stage2, dashboard).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from geoai.core.config import PlatformConfig, load_platform_config
from geoai.core.exceptions import StageExecutionError
from geoai.dashboard.project_manager import ProjectManager
from geoai.pipeline.stage1 import Stage1Result, run_stage1
from geoai.pipeline.stage2 import Stage2Result, run_stage2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 3 result container
# ---------------------------------------------------------------------------


class Stage3Result:
    """Container for all Stage 3 batch pipeline outputs.

    Attributes:
        project_id: Project identifier.
        run_id: Unique run identifier for this batch execution.
        aoi_results: Dictionary mapping AOI name to (Stage1Result, Stage2Result).
        failed_aois: List of AOI names that failed processing.
        project_summary: Aggregated statistics across all AOIs.
        output_dir: Root output directory for this batch run.
    """

    def __init__(self, project_id: str, run_id: str) -> None:
        self.project_id = project_id
        self.run_id = run_id
        self.aoi_results: Dict[str, tuple] = {}
        self.failed_aois: List[str] = []
        self.project_summary: Dict[str, Any] = {}
        self.output_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Primary pipeline entry point
# ---------------------------------------------------------------------------


def run_stage3(
    platform_config: PlatformConfig,
    project_id: Optional[str] = None,
    aoi_names: Optional[List[str]] = None,
    run_stage2_classification: bool = True,
    compute_shap: bool = False,
    continue_on_error: bool = True,
) -> Stage3Result:
    """Execute the Stage 3 batch pipeline for all configured AOIs.

    Iterates over all AOIs in the platform configuration (or a specified
    subset), runs Stage 1 and optionally Stage 2 for each, and produces
    aggregated project-level outputs.

    Args:
        platform_config: Full platform configuration. AOIs are read from
            ``platform_config.processing.aois``.
        project_id: Optional project identifier. If None, a timestamped
            ID is generated.
        aoi_names: Optional list of AOI names to process. If None, all
            configured AOIs are processed.
        run_stage2_classification: If True, run Stage 2 after Stage 1 for
            each AOI. Default True.
        compute_shap: If True, compute SHAP attributions during Stage 2.
            Default False.
        continue_on_error: If True, log failures and continue with the next
            AOI when an error occurs. If False, re-raise on first failure.
            Default True.

    Returns:
        :class:`Stage3Result` with results for all successfully processed AOIs.

    Raises:
        StageExecutionError: Only raised when ``continue_on_error`` is False
            and a pipeline stage fails.
    """
    if project_id is None:
        project_id = f"project_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    run_id = f"{project_id}_batch_{datetime.utcnow().strftime('%H%M%S')}"
    result = Stage3Result(project_id=project_id, run_id=run_id)

    # Determine which AOIs to process.
    all_configured_aois = [a.name for a in platform_config.processing.aois]
    target_aois = aoi_names if aoi_names else all_configured_aois

    if not target_aois:
        logger.warning(
            "Stage 3: no AOIs configured for processing. "
            "Check processing.yaml."
        )
        return result

    logger.info(
        "=== Stage 3 batch pipeline START — project='%s' "
        "AOIs=%s ===",
        project_id, target_aois,
    )

    output_dir = Path(platform_config.export.output_base_dir) / project_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result.output_dir = output_dir

    # Initialise project manager.
    project_mgr = ProjectManager(
        manifest_dir=platform_config.dashboard.project_manifest_dir
    )
    project_mgr.create_project(project_id, aoi_names=target_aois)

    # Iterate over AOIs.
    for aoi_name in target_aois:
        logger.info(
            "Stage 3: processing AOI '%s' (%d/%d).",
            aoi_name,
            target_aois.index(aoi_name) + 1,
            len(target_aois),
        )
        aoi_run_id = f"{project_id}_{aoi_name}"

        try:
            # Stage 1.
            s1_result = run_stage1(
                platform_config=platform_config,
                aoi_name=aoi_name,
                run_id=aoi_run_id,
                generate_ps10_submission=True,
            )

            # Stage 2 (optional).
            s2_result = None
            if run_stage2_classification:
                s2_result = run_stage2(
                    stage1_result=s1_result,
                    platform_config=platform_config,
                    compute_shap=compute_shap,
                    run_id=f"{aoi_run_id}_s2",
                )

            result.aoi_results[aoi_name] = (s1_result, s2_result)

            # Record run in project manifest.
            project_mgr.record_run(
                project_id=project_id,
                aoi_name=aoi_name,
                run_id=aoi_run_id,
                stage1_summary=s1_result.prediction_summary,
                object_count=len(s1_result.object_records),
            )

            logger.info(
                "Stage 3: AOI '%s' completed successfully.", aoi_name
            )

        except Exception as exc:
            logger.error(
                "Stage 3: AOI '%s' FAILED — %s", aoi_name, exc
            )
            result.failed_aois.append(aoi_name)
            project_mgr.record_failure(project_id, aoi_name, str(exc))

            if not continue_on_error:
                raise StageExecutionError(
                    stage="stage3",
                    aoi_name=aoi_name,
                    reason=str(exc),
                ) from exc

    # Aggregate project-level summary.
    result.project_summary = _aggregate_project_summary(result)
    project_mgr.update_summary(project_id, result.project_summary)

    # Write project summary report.
    _write_project_report(result, output_dir)

    n_success = len(result.aoi_results)
    n_failed = len(result.failed_aois)
    logger.info(
        "=== Stage 3 batch pipeline COMPLETE — project='%s' "
        "success=%d failed=%d ===",
        project_id, n_success, n_failed,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_project_summary(result: Stage3Result) -> Dict[str, Any]:
    """Aggregate statistics across all successfully processed AOIs.

    Args:
        result: Stage3Result with populated aoi_results.

    Returns:
        Project-level summary dictionary.
    """
    total_change_ha = 0.0
    total_objects = 0
    total_change_px = 0

    for aoi_name, (s1, s2) in result.aoi_results.items():
        total_change_ha += s1.prediction_summary.get("change_area_ha", 0.0)
        total_objects += s1.object_summary.get("object_count", 0)
        total_change_px += s1.prediction_summary.get("change_pixels", 0)

    return {
        "project_id": result.project_id,
        "run_id": result.run_id,
        "total_aois_processed": len(result.aoi_results),
        "total_aois_failed": len(result.failed_aois),
        "failed_aois": result.failed_aois,
        "total_change_area_ha": round(total_change_ha, 4),
        "total_significant_objects": total_objects,
        "total_change_pixels": total_change_px,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }


def _write_project_report(result: Stage3Result, output_dir: Path) -> None:
    """Write a plain-text project-level summary report."""
    report_path = output_dir / "project_report.txt"
    sep = "=" * 60
    lines = [
        sep,
        "GeoAI Platform — Stage 3 Project Report",
        sep,
        f"Project ID     : {result.project_id}",
        f"Run ID         : {result.run_id}",
        f"Timestamp (UTC): {result.project_summary.get('timestamp_utc', 'N/A')}",
        "",
        "--- Batch Summary ---",
        f"AOIs processed : {result.project_summary.get('total_aois_processed', 0)}",
        f"AOIs failed    : {result.project_summary.get('total_aois_failed', 0)}",
    ]
    if result.failed_aois:
        lines.append(f"Failed AOIs    : {', '.join(result.failed_aois)}")
    lines += [
        "",
        "--- Aggregated Change Statistics ---",
        f"Total change area  : {result.project_summary.get('total_change_area_ha', 0):.4f} ha",
        f"Total objects      : {result.project_summary.get('total_significant_objects', 0)}",
        f"Total change px    : {result.project_summary.get('total_change_pixels', 0):,}",
        "",
        "--- Per-AOI Results ---",
    ]
    for aoi_name, (s1, s2) in result.aoi_results.items():
        lines.append(
            f"  {aoi_name}: "
            f"{s1.object_summary.get('object_count', 0)} objects, "
            f"{s1.prediction_summary.get('change_area_ha', 0):.2f} ha"
        )
    lines += ["", sep, "End of Project Report", sep]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Project report written: '%s'.", report_path)
