"""
Stage 2 object classification pipeline for the GeoAI Platform.

Orchestrates the complete Stage 2 workflow starting from a Stage 1 result
and producing a fully classified, attributed, and exported set of change
objects with semantic class assignments.

Pipeline steps:
  1. Build attributed ChangeObject list from Stage 1 significant objects.
  2. Extract per-object spectral features from the feature cube.
  3. Classify objects using the rule-based semantic classifier.
  4. Assign per-object confidence scores.
  5. Compute class-wise spatial statistics.
  6. Optionally compute SHAP feature attributions.
  7. Export classified shapefile, GeoJSON, and CSV.
  8. Generate Stage 2 report.

Single responsibility: orchestrate Stage 2 classification from a Stage 1
result to classified export deliverables.

Position in dependency hierarchy: pipeline (depends on all classification,
analysis, export layers).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from geoai.classification.attributes import build_attributed_objects
from geoai.classification.classifier import RuleBasedClassifier
from geoai.classification.confidence import assign_confidences
from geoai.classification.explainability import build_tree_explainer, compute_object_shap_values
from geoai.classification.schema import ChangeObject
from geoai.classification.spatial_stats import (
    compute_class_statistics,
    format_class_statistics_report,
)
from geoai.core.config import PlatformConfig
from geoai.core.exceptions import StageExecutionError
from geoai.exports.csv_export import export_class_statistics_csv, export_objects_csv
from geoai.exports.reports import generate_stage2_report
from geoai.exports.shapefile import export_classified_shapefile, export_geojson
from geoai.models.production.rf_enhanced import RFEnhancedModel
from geoai.pipeline.stage1 import Stage1Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2 result container
# ---------------------------------------------------------------------------


class Stage2Result:
    """Container for all Stage 2 pipeline outputs.

    Attributes:
        run_id: Unique identifier for this run.
        aoi_name: Name of the processed AOI.
        objects: List of fully classified ChangeObject instances.
        class_statistics: Per-class statistics dictionary.
        output_paths: Dictionary mapping output type to file path.
        report: Stage 2 report string.
    """

    def __init__(self, run_id: str, aoi_name: str) -> None:
        self.run_id = run_id
        self.aoi_name = aoi_name
        self.objects: List[ChangeObject] = []
        self.class_statistics: Dict[str, Dict] = {}
        self.output_paths: Dict[str, Path] = {}
        self.report: str = ""


# ---------------------------------------------------------------------------
# Primary pipeline entry point
# ---------------------------------------------------------------------------


def run_stage2(
    stage1_result: Stage1Result,
    platform_config: PlatformConfig,
    compute_shap: bool = False,
    run_id: Optional[str] = None,
) -> Stage2Result:
    """Execute the Stage 2 object classification pipeline.

    Takes the output of :func:`~geoai.pipeline.stage1.run_stage1` and
    produces a classified, attributed set of change objects with semantic
    classes from the PS10 Stage 2 taxonomy.

    Args:
        stage1_result: The result from a completed Stage 1 run. Must have
            ``significant_mask``, ``label_array``, ``feature_cube``,
            and ``profile`` populated.
        platform_config: Full platform configuration.
        compute_shap: If True, compute SHAP feature attributions. Requires
            the ``shap`` library and may be slow for large rasters.
            Default False.
        run_id: Optional run identifier. Defaults to
            ``'{aoi_name}_stage2_{timestamp}'``.

    Returns:
        :class:`Stage2Result` containing all Stage 2 outputs.

    Raises:
        StageExecutionError: If any pipeline step fails.
    """
    aoi_name = stage1_result.aoi_name

    if run_id is None:
        run_id = f"{aoi_name}_stage2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    result = Stage2Result(run_id=run_id, aoi_name=aoi_name)

    logger.info(
        "=== Stage 2 pipeline START — AOI='%s' run_id='%s' ===",
        aoi_name, run_id,
    )

    try:
        output_dir = _resolve_stage2_output_dir(platform_config, run_id)
        aoi = platform_config.get_aoi(aoi_name)

        # Step 1 & 2: Build attributed ChangeObjects (spatial + spectral).
        logger.info("Building attributed ChangeObject list.")
        objects = _build_objects_from_stage1(stage1_result, aoi.resolution_m)

        if not objects:
            logger.warning(
                "Stage 2: no significant objects from Stage 1 result. "
                "Producing empty Stage 2 output."
            )
            result.class_statistics = compute_class_statistics([])
            result.report = generate_stage2_report(
                aoi_name=aoi_name,
                run_id=run_id,
                objects=[],
                class_stats=result.class_statistics,
                output_path=output_dir / "reports" / "stage2_report.txt",
            )
            return result

        # Step 3: Semantic classification.
        logger.info("Running rule-based semantic classifier.")
        classifier = RuleBasedClassifier()
        objects = classifier.classify(objects)

        # Step 4: Assign confidence scores.
        logger.info("Assigning confidence scores.")
        objects = assign_confidences(
            objects=objects,
            prediction_map=stage1_result.prediction_map,
            probability_map=None,  # Stage 1 may not have computed probabilities.
            label_array=stage1_result.label_array,
        )

        # Step 5: Class-wise statistics.
        class_stats = compute_class_statistics(
            objects, resolution_m=aoi.resolution_m
        )
        result.class_statistics = class_stats

        # Step 6: Optional SHAP attribution.
        if compute_shap:
            objects = _compute_shap_if_available(
                objects=objects,
                stage1_result=stage1_result,
                platform_config=platform_config,
            )

        result.objects = objects

        # Step 7: Export classified deliverables.
        export_dir = output_dir / "exports"
        result.output_paths.update(
            _export_stage2_deliverables(
                objects=objects,
                significant_mask=stage1_result.significant_mask,
                class_stats=class_stats,
                profile=stage1_result.profile,
                aoi_name=aoi_name,
                export_dir=export_dir,
            )
        )

        # Step 8: Generate report.
        report = generate_stage2_report(
            aoi_name=aoi_name,
            run_id=run_id,
            objects=objects,
            class_stats=class_stats,
            output_path=output_dir / "reports" / "stage2_report.txt",
        )
        result.report = report
        result.output_paths["report"] = output_dir / "reports" / "stage2_report.txt"

        logger.info(
            "=== Stage 2 pipeline COMPLETE — AOI='%s' run_id='%s' "
            "objects=%d ===",
            aoi_name, run_id, len(objects),
        )

    except StageExecutionError:
        raise
    except Exception as exc:
        raise StageExecutionError(
            stage="stage2",
            aoi_name=aoi_name,
            reason=str(exc),
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_objects_from_stage1(
    stage1_result: Stage1Result,
    resolution_m: float,
) -> List[ChangeObject]:
    """Build attributed ChangeObject list from Stage 1 result data.

    Args:
        stage1_result: Completed Stage1Result.
        resolution_m: Ground sampling distance in metres.

    Returns:
        List of ChangeObject instances with spatial and spectral attributes.
    """
    from skimage.measure import label as sk_label, regionprops

    # Reconstruct region properties from the significant mask and label array.
    # We re-run regionprops on the existing label array filtered to significant objects.
    if (stage1_result.significant_mask is None
            or stage1_result.label_array is None
            or stage1_result.feature_cube is None):
        logger.warning(
            "Stage 2: Stage 1 result is missing significant_mask, "
            "label_array, or feature_cube. Returning empty object list."
        )
        return []

    # Get only the labels that are in the significant mask.
    sig_labels = set(
        int(lbl)
        for lbl in np.unique(stage1_result.label_array)
        if lbl > 0 and (stage1_result.label_array == lbl).any()
        and (stage1_result.significant_mask[stage1_result.label_array == lbl] > 0).all()
    )

    all_regions = regionprops(stage1_result.label_array)
    significant_regions = [r for r in all_regions if r.label in sig_labels]

    logger.info(
        "Stage 2: recovered %d significant regions from Stage 1 label array.",
        len(significant_regions),
    )

    transform = stage1_result.profile.get("transform") if stage1_result.profile else None

    objects = build_attributed_objects(
        significant_regions=significant_regions,
        label_array=stage1_result.label_array,
        feature_cube=stage1_result.feature_cube,
        transform=transform,
        resolution_m=resolution_m,
    )
    return objects


def _compute_shap_if_available(
    objects: List[ChangeObject],
    stage1_result: Stage1Result,
    platform_config: PlatformConfig,
) -> List[ChangeObject]:
    """Attempt SHAP attribution; log warning and continue on failure."""
    try:
        from geoai.models.registry import load_model_from_config
        model = load_model_from_config(platform_config)
        if isinstance(model, RFEnhancedModel):
            explainer = build_tree_explainer(model._rf)
            if explainer is not None:
                objects = compute_object_shap_values(
                    objects=objects,
                    explainer=explainer,
                    feature_cube=stage1_result.feature_cube,
                    label_array=stage1_result.label_array,
                )
    except Exception as exc:
        logger.warning("SHAP attribution failed: %s. Continuing without SHAP.", exc)
    return objects


def _export_stage2_deliverables(
    objects: List[ChangeObject],
    significant_mask: np.ndarray,
    class_stats: Dict,
    profile: Dict,
    aoi_name: str,
    export_dir: Path,
) -> Dict[str, Path]:
    """Export all Stage 2 output files."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Classified shapefile.
    shp_path = export_dir / f"Classified_Objects_{aoi_name}.shp"
    export_classified_shapefile(objects, significant_mask, profile, shp_path)
    paths["shapefile"] = shp_path

    # GeoJSON with full attributes.
    gj_path = export_dir / f"Classified_Objects_{aoi_name}.geojson"
    export_geojson(objects, significant_mask, profile, gj_path)
    paths["geojson"] = gj_path

    # Per-object CSV.
    obj_csv = export_dir / f"Objects_{aoi_name}.csv"
    export_objects_csv(objects, obj_csv)
    paths["objects_csv"] = obj_csv

    # Class statistics CSV.
    stats_csv = export_dir / f"Class_Statistics_{aoi_name}.csv"
    export_class_statistics_csv(class_stats, stats_csv)
    paths["class_stats_csv"] = stats_csv

    return paths


def _resolve_stage2_output_dir(config: PlatformConfig, run_id: str) -> Path:
    """Create and return the output directory for a Stage 2 run."""
    run_dir = Path(config.export.output_base_dir) / run_id
    for subdir in ("exports", "reports"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir
