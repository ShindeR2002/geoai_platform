"""
Stage 1 change detection pipeline for the GeoAI Platform.

Orchestrates the complete end-to-end Stage 1 workflow from configuration
and raw raster inputs to PS10-compliant GeoTIFF and shapefile deliverables.

Pipeline steps (implementing the canonical Version 1 NB09 workflow):
  1. Load and validate raster datasets.
  2. Load EO and SAR rasters for both epochs.
  3. Load or compute spectral indices.
  4. Build the canonical 18-feature cube.
  5. Run RF Enhanced inference.
  6. Post-process prediction to binary mask.
  7. Extract significant change objects.
  8. Compute object statistics.
  9. Export GeoTIFF, shapefile, GeoJSON, CSV.
 10. Generate run report.
 11. Generate visualisations.

All parameters are read from the PlatformConfig object. No values are
hardcoded in this module.

Single responsibility: orchestrate the Stage 1 workflow from config to outputs.

Position in dependency hierarchy: pipeline (depends on all platform layers).
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geoai.analysis.objects import extract_objects
from geoai.analysis.statistics import build_object_records, compute_summary_statistics
from geoai.classification.schema import change_objects_from_records
from geoai.core.config import AOIConfig, PlatformConfig
from geoai.core.exceptions import StageExecutionError
from geoai.core.validation import validate_all_rasters_exist, validate_raster_compatibility
from geoai.exports.csv_export import export_objects_csv
from geoai.exports.geotiff import export_change_mask_geotiff, export_ps10_change_mask
from geoai.exports.packaging import build_ps10_submission_zip, generate_model_hash_for_submission
from geoai.exports.reports import generate_stage1_report
from geoai.exports.shapefile import export_change_shapefile, export_geojson
from geoai.features.feature_cube import build_feature_cube
from geoai.features.pseudo_labels import generate_pseudo_labels
from geoai.models.inference import run_inference, summarise_prediction
from geoai.models.registry import load_model_from_config
from geoai.postprocessing.cleanup import to_binary_mask
from geoai.preprocessing.eo import load_eo_epoch
from geoai.preprocessing.indices import get_indices
from geoai.preprocessing.sar import load_sar_vv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 result dataclass
# ---------------------------------------------------------------------------


class Stage1Result:
    """Container for all Stage 1 pipeline outputs.

    Attributes:
        run_id: Unique identifier for this run.
        aoi_name: Name of the processed AOI.
        prediction_map: Float32 (H, W) prediction raster.
        binary_mask: uint8 (H, W) binary change mask.
        significant_mask: uint8 (H, W) mask of significant objects only.
        label_array: Integer (H, W) connected component labels.
        object_records: List of per-object attribute dictionaries.
        prediction_summary: Summary statistics dictionary.
        object_summary: Object aggregate statistics dictionary.
        feature_cube: Float32 (H, W, 18) feature cube.
        profile: rasterio profile from the source rasters.
        output_paths: Dictionary mapping output type to file path.
        report: Stage 1 report string.
    """

    def __init__(self, run_id: str, aoi_name: str) -> None:
        self.run_id = run_id
        self.aoi_name = aoi_name
        self.prediction_map: Optional[np.ndarray] = None
        self.binary_mask: Optional[np.ndarray] = None
        self.significant_mask: Optional[np.ndarray] = None
        self.label_array: Optional[np.ndarray] = None
        self.object_records: List[Dict] = []
        self.prediction_summary: Dict[str, Any] = {}
        self.object_summary: Dict[str, Any] = {}
        self.feature_cube: Optional[np.ndarray] = None
        self.profile: Optional[Dict] = None
        self.output_paths: Dict[str, Path] = {}
        self.report: str = ""


# ---------------------------------------------------------------------------
# Primary pipeline entry point
# ---------------------------------------------------------------------------


def run_stage1(
    platform_config: PlatformConfig,
    aoi_name: str,
    run_id: Optional[str] = None,
    compute_probabilities: bool = False,
    generate_visualisations: bool = True,
    generate_ps10_submission: bool = True,
) -> Stage1Result:
    """Execute the complete Stage 1 change detection pipeline for one AOI.

    Args:
        platform_config: Fully loaded PlatformConfig from
            :func:`~geoai.core.config.load_platform_config`.
        aoi_name: Name of the AOI to process. Must match an entry in the
            ``processing.aois`` configuration list.
        run_id: Optional run identifier. If None, a timestamped ID is generated.
        compute_probabilities: If True, also compute and export prediction
            probability rasters. Default False.
        generate_visualisations: If True, generate RGB overlay and plot images.
            Default True.
        generate_ps10_submission: If True, package the PS10 submission zip.
            Default True.

    Returns:
        :class:`Stage1Result` containing all pipeline outputs.

    Raises:
        StageExecutionError: If any step in the pipeline fails.
        ConfigurationError: If the AOI is not found in configuration.
    """
    if run_id is None:
        run_id = f"{aoi_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    result = Stage1Result(run_id=run_id, aoi_name=aoi_name)

    logger.info(
        "=== Stage 1 pipeline START — AOI='%s' run_id='%s' ===",
        aoi_name, run_id,
    )

    try:
        aoi = platform_config.get_aoi(aoi_name)
        output_dir = _make_run_output_dir(platform_config, run_id)

        # Step 1: Validate all raster files exist.
        _validate_aoi_rasters(aoi)

        # Step 2 & 3: Load EO, SAR, and indices for both epochs.
        (red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1,
         red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2,
         sar_t1, sar_t2, profile) = _load_all_rasters(aoi)

        result.profile = profile

        # Step 4: Build 18-feature cube.
        logger.info("Assembling 18-feature cube.")
        feature_cube = build_feature_cube(
            red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
            sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
            red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
            sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
        )
        result.feature_cube = feature_cube

        # Step 5: Load model and run inference.
        logger.info("Loading model and running inference.")
        model = load_model_from_config(platform_config)
        prediction_map, valid_mask, probability_map = run_inference(
            model=model,
            feature_cube=feature_cube,
            compute_probabilities=compute_probabilities,
        )
        result.prediction_map = prediction_map

        # Step 6: Convert to binary mask.
        binary_mask = to_binary_mask(prediction_map)
        result.binary_mask = binary_mask

        # Step 7: Extract significant objects.
        min_obj_size = platform_config.processing.min_object_size_px
        significant_regions, label_array, significant_mask = extract_objects(
            binary_mask=binary_mask,
            min_area_px=min_obj_size,
        )
        result.label_array = label_array
        result.significant_mask = significant_mask

        # Step 8: Compute statistics.
        transform = profile.get("transform")
        resolution_m = aoi.resolution_m
        object_records = build_object_records(
            significant_regions, resolution_m=resolution_m, transform=transform
        )
        result.object_records = object_records
        result.prediction_summary = summarise_prediction(prediction_map, resolution_m)
        result.object_summary = compute_summary_statistics(object_records, resolution_m)

        # Step 9: Export deliverables.
        export_dir = output_dir / "exports"
        result.output_paths.update(
            _export_stage1_deliverables(
                binary_mask=binary_mask,
                significant_mask=significant_mask,
                object_records=object_records,
                profile=profile,
                aoi=aoi,
                export_dir=export_dir,
                export_config=platform_config.export,
                probability_map=probability_map,
                compute_probabilities=compute_probabilities,
            )
        )

        # Step 10: Generate report.
        report = generate_stage1_report(
            aoi_name=aoi_name,
            run_id=run_id,
            prediction_summary=result.prediction_summary,
            object_summary=result.object_summary,
            output_path=output_dir / "reports" / "stage1_report.txt",
        )
        result.report = report
        result.output_paths["report"] = output_dir / "reports" / "stage1_report.txt"

        # Step 11: Visualisations.
        if generate_visualisations:
            _generate_visualisations(
                red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
                prediction_map=prediction_map,
                aoi_name=aoi_name,
                output_dir=output_dir / "figures",
            )

        # Step 12: PS10 submission packaging.
        if generate_ps10_submission:
            _package_ps10_submission(
                result=result,
                model_path=platform_config.model.model_path,
                output_dir=output_dir / "exports",
                team_name="GeoAI_Platform",
                aoi=aoi,
            )

        logger.info(
            "=== Stage 1 pipeline COMPLETE — AOI='%s' run_id='%s' ===",
            aoi_name, run_id,
        )

    except StageExecutionError:
        raise
    except Exception as exc:
        raise StageExecutionError(
            stage="stage1",
            aoi_name=aoi_name,
            reason=str(exc),
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_aoi_rasters(aoi: AOIConfig) -> None:
    """Check all required raster files exist for an AOI."""
    raster_paths = {}
    for raster_type in ["rgb", "sar_vv", "ndvi", "ndbi", "ndwi"]:
        for year in [aoi.t1_year, aoi.t2_year]:
            key = f"{raster_type}_{year}"
            raster_paths[key] = aoi.get_raster_path(raster_type, year)
    validate_all_rasters_exist(raster_paths)


def _load_all_rasters(aoi: AOIConfig):
    """Load all EO, SAR, and index rasters for both epochs."""
    logger.info("Loading rasters for AOI '%s'.", aoi.name)

    # T1 epoch
    # Use the EO epoch bundle loader which returns RGB bands + three indices
    # and the raster profile (7 elements). Previously this called the RGB
    # loader directly which returned only 4 elements and caused unpacking
    # errors.
    red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1, profile_t1 = (
        load_eo_epoch(
            aoi.get_raster_path("rgb", aoi.t1_year),
            aoi.get_raster_path("ndvi", aoi.t1_year),
            aoi.get_raster_path("ndbi", aoi.t1_year),
            aoi.get_raster_path("ndwi", aoi.t1_year),
            epoch_label="T1",
        )
    )
    # If indices are supplied via separate files, get_indices will still be
    # used for consistency; however when index_mode==load the above already
    # loaded the index arrays so we can skip an extra read.
    if aoi.index_mode != "load":
        ndvi_t1, ndbi_t1, ndwi_t1 = get_indices(
        mode=aoi.index_mode,
        ndvi_path=aoi.get_raster_path("ndvi", aoi.t1_year),
        ndbi_path=aoi.get_raster_path("ndbi", aoi.t1_year),
        ndwi_path=aoi.get_raster_path("ndwi", aoi.t1_year),
        epoch_label="T1",
    )
    sar_t1, _ = load_sar_vv(aoi.get_raster_path("sar_vv", aoi.t1_year), "T1")

    # T2 epoch
    red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2, profile_t2 = (
        load_eo_epoch(
            aoi.get_raster_path("rgb", aoi.t2_year),
            aoi.get_raster_path("ndvi", aoi.t2_year),
            aoi.get_raster_path("ndbi", aoi.t2_year),
            aoi.get_raster_path("ndwi", aoi.t2_year),
            epoch_label="T2",
        )
    )
    if aoi.index_mode != "load":
        ndvi_t2, ndbi_t2, ndwi_t2 = get_indices(
        mode=aoi.index_mode,
        ndvi_path=aoi.get_raster_path("ndvi", aoi.t2_year),
        ndbi_path=aoi.get_raster_path("ndbi", aoi.t2_year),
        ndwi_path=aoi.get_raster_path("ndwi", aoi.t2_year),
        epoch_label="T2",
    )
    sar_t2, _ = load_sar_vv(aoi.get_raster_path("sar_vv", aoi.t2_year), "T2")

    return (red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1,
            red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2,
            sar_t1, sar_t2, profile_t1)


def _load_eo_rgb(aoi: AOIConfig, year: int):
    """Load RGB bands for one epoch."""
    from geoai.preprocessing.eo import load_rgb
    rgb_path = aoi.get_raster_path("rgb", year)
    return load_rgb(rgb_path)


def _export_stage1_deliverables(
    binary_mask, significant_mask, object_records, profile,
    aoi, export_dir, export_config, probability_map, compute_probabilities,
) -> Dict[str, Path]:
    """Export all Stage 1 output files. Returns dict of output paths."""
    export_dir = Path(export_dir)
    paths = {}

    # GeoTIFF binary change mask.
    if aoi.reference_lat and aoi.reference_lon:
        tif_path = export_ps10_change_mask(
            binary_mask, profile, export_dir,
            aoi.reference_lat, aoi.reference_lon,
        )
    else:
        tif_path = export_change_mask_geotiff(
            binary_mask, profile,
            export_dir / f"Change_Mask_{aoi.name}.tif",
        )
    paths["geotiff"] = tif_path

    # Shapefile.
    if aoi.reference_lat and aoi.reference_lon:
        shp_path = Path(export_dir) / f"Change_Mask_{aoi.reference_lat:.4f}_{aoi.reference_lon:.4f}.shp"
    else:
        shp_path = export_dir / f"Change_Mask_{aoi.name}.shp"
    export_change_shapefile(significant_mask, profile, shp_path)
    paths["shapefile"] = shp_path

    # GeoJSON (optional).
    if export_config.include_geojson:
        objects = change_objects_from_records(object_records)
        gj_path = export_dir / f"Change_Objects_{aoi.name}.geojson"
        export_geojson(objects, significant_mask, profile, gj_path)
        paths["geojson"] = gj_path

    # CSV statistics (optional).
    if export_config.include_csv and object_records:
        objects = change_objects_from_records(object_records)
        csv_path = export_dir / f"Object_Statistics_{aoi.name}.csv"
        export_objects_csv(objects, csv_path)
        paths["csv"] = csv_path

    return paths


def _generate_visualisations(
    red_t2, green_t2, blue_t2, prediction_map, aoi_name, output_dir
) -> None:
    """Generate RGB overlay visualisation."""
    try:
        from geoai.visualization.maps import plot_change_overlay
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_change_overlay(
            red=red_t2, green=green_t2, blue=blue_t2,
            prediction_map=prediction_map,
            output_path=Path(output_dir) / f"change_overlay_{aoi_name}.png",
            title=f"Change Detection — {aoi_name}",
        )
    except Exception as exc:
        logger.warning("Visualisation generation failed: %s.", exc)


def _package_ps10_submission(
    result: Stage1Result,
    model_path: str,
    output_dir: Path,
    team_name: str,
    aoi: AOIConfig,
) -> None:
    """Package the PS10 submission zip and generate model hash."""
    try:
        files_to_zip = [
            v for k, v in result.output_paths.items()
            if k in ("geotiff", "shapefile")
        ]
        if files_to_zip:
            build_ps10_submission_zip(
                result_files=files_to_zip,
                output_dir=output_dir,
                team_name=team_name,
            )
        model_p = Path(model_path)
        if model_p.exists():
            generate_model_hash_for_submission(model_p, output_dir)
    except Exception as exc:
        logger.warning("PS10 submission packaging failed: %s.", exc)


def _make_run_output_dir(config: PlatformConfig, run_id: str) -> Path:
    """Create and return the output directory for a pipeline run."""
    run_dir = Path(config.export.output_base_dir) / run_id
    for subdir in ("exports", "reports", "figures", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("Run output directory: '%s'.", run_dir)
    return run_dir
