"""
Shapefile and GeoJSON export for the GeoAI Platform.

Exports significant change objects as vector files in shapefile and GeoJSON
formats. The shapefile is a mandatory PS10 deliverable; GeoJSON is an optional
additional output for web visualisation and API responses.

Stage 1 exports contain spatial attributes only. Stage 2 exports include
the full ChangeObject attribute table with semantic class and spectral features.

PS10 submission shapefile naming:
    Change_Mask_{Lat}_{Long}.shp

Single responsibility: vectorise change objects and write to shapefile / GeoJSON.

Position in dependency hierarchy: exports (depends on analysis/geometry,
classification/schema, core, utils).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from geoai.analysis.geometry import vectorise_mask, merge_overlapping_polygons
from geoai.classification.schema import ChangeObject
from geoai.core.exceptions import ShapefileExportError
from geoai.core.io import RasterProfile
from geoai.utils.constants import PS10_FILENAME_PREFIX
from geoai.utils.geo import format_ps10_filename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 shapefile export (spatial attributes only)
# ---------------------------------------------------------------------------


def export_change_shapefile(
    significant_mask: np.ndarray,
    profile: RasterProfile,
    output_path: Union[str, Path],
    merge_polygons: bool = True,
) -> Path:
    """Export significant change objects as a shapefile (Stage 1).

    Vectorises the significant change mask and writes the resulting polygons
    to a shapefile. This is the PS10 primary vector deliverable.

    Args:
        significant_mask: uint8 binary array of shape (H, W) containing
            only significant objects (area >= min_object_size).
        profile: rasterio profile dictionary from the prediction raster.
        output_path: Destination file path. Extension should be ``.shp``.
        merge_polygons: If True, merge adjacent/overlapping polygons from
            the rasterio vectorisation step. Default True.

    Returns:
        Resolved absolute path of the written shapefile.

    Raises:
        ShapefileExportError: If writing fails.
    """
    import geopandas as gpd
    from shapely.geometry import mapping

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = profile.get("transform")
    crs = profile.get("crs")

    logger.info(
        "Exporting Stage 1 shapefile to '%s' (crs=%s).",
        output_path.name, crs,
    )

    features = vectorise_mask(significant_mask, transform)
    if merge_polygons:
        features = merge_overlapping_polygons(features)

    if not features:
        logger.warning(
            "No features to export to shapefile '%s'. "
            "Writing empty shapefile.",
            output_path.name,
        )

    geometries = [f["geometry"] for f in features]
    attributes = [{"object_id": i + 1, "change": 1} for i in range(len(features))]

    try:
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
        gdf.to_file(str(output_path), driver="ESRI Shapefile")
    except Exception as exc:
        raise ShapefileExportError(str(output_path), str(exc)) from exc

    logger.info(
        "Shapefile written: '%s' — %d feature(s).",
        output_path, len(features),
    )
    return output_path.resolve()


def export_ps10_shapefile(
    significant_mask: np.ndarray,
    profile: RasterProfile,
    output_dir: Union[str, Path],
    reference_lat: float,
    reference_lon: float,
    prefix: str = PS10_FILENAME_PREFIX,
) -> Path:
    """Export the PS10-convention shapefile with the required filename.

    Args:
        significant_mask: uint8 binary mask of significant change objects.
        profile: rasterio profile from the prediction raster.
        output_dir: Directory where the shapefile will be written.
        reference_lat: Reference latitude for the filename.
        reference_lon: Reference longitude for the filename.
        prefix: Filename prefix. Default ``'Change_Mask'``.

    Returns:
        Resolved absolute path of the written shapefile.

    Raises:
        ShapefileExportError: If writing fails.
    """
    filename = format_ps10_filename(prefix, reference_lat, reference_lon, "shp")
    output_path = Path(output_dir) / filename
    logger.info("Exporting PS10 shapefile: '%s'.", filename)
    return export_change_shapefile(significant_mask, profile, output_path)


# ---------------------------------------------------------------------------
# Stage 2 shapefile export (full ChangeObject attribute table)
# ---------------------------------------------------------------------------


def export_classified_shapefile(
    objects: List[ChangeObject],
    significant_mask: np.ndarray,
    profile: RasterProfile,
    output_path: Union[str, Path],
) -> Path:
    """Export classified change objects as a shapefile with full attributes.

    Produces a shapefile where each feature is a change object polygon and
    the attribute table contains the complete ChangeObject export dictionary:
    object_id, area, centroid, semantic class, confidence, spectral features.

    Args:
        objects: List of classified ChangeObject instances.
        significant_mask: uint8 binary mask used for polygon geometry.
        profile: rasterio profile from the prediction raster.
        output_path: Destination file path.

    Returns:
        Resolved absolute path of the written shapefile.

    Raises:
        ShapefileExportError: If writing fails.
    """
    import geopandas as gpd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = profile.get("transform")
    crs = profile.get("crs")

    logger.info(
        "Exporting Stage 2 classified shapefile to '%s' "
        "(%d objects).",
        output_path.name, len(objects),
    )

    features = vectorise_mask(significant_mask, transform)
    features = merge_overlapping_polygons(features)

    if len(features) != len(objects):
        logger.warning(
            "Feature count mismatch: %d polygons vs %d ChangeObjects. "
            "Exporting with spatial attributes only for unmatched features.",
            len(features), len(objects),
        )

    records = []
    geometries = []

    # Match objects to polygons by iterating objects (primary source of truth).
    for obj in objects:
        attr = obj.to_export_dict()
        if features:
            geom = features.pop(0)["geometry"]
        else:
            geom = None
        attr["geometry"] = geom
        records.append(attr)
        geometries.append(geom)

    try:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf.to_file(str(output_path), driver="ESRI Shapefile")
    except Exception as exc:
        raise ShapefileExportError(str(output_path), str(exc)) from exc

    logger.info(
        "Classified shapefile written: '%s' — %d feature(s).",
        output_path, len(records),
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# GeoJSON export
# ---------------------------------------------------------------------------


def export_geojson(
    objects: List[ChangeObject],
    significant_mask: np.ndarray,
    profile: RasterProfile,
    output_path: Union[str, Path],
) -> Path:
    """Export change objects as a GeoJSON FeatureCollection.

    Produces a GeoJSON file with one Feature per change object. Each Feature
    includes the polygon geometry and the full ChangeObject export attributes.

    Args:
        objects: List of ChangeObject instances (Stage 1 or Stage 2).
        significant_mask: uint8 binary mask for polygon geometry.
        profile: rasterio profile from the prediction raster.
        output_path: Destination file path. Should end with ``.geojson``.

    Returns:
        Resolved absolute path of the written GeoJSON file.

    Raises:
        ShapefileExportError: If writing fails (reuses this exception class
            for all vector export failures).
    """
    from shapely.geometry import mapping as shapely_mapping

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = profile.get("transform")
    features_raw = vectorise_mask(significant_mask, transform)
    features_raw = merge_overlapping_polygons(features_raw)

    logger.info(
        "Exporting GeoJSON to '%s' (%d objects).",
        output_path.name, len(objects),
    )

    geojson_features = []
    for i, obj in enumerate(objects):
        if i < len(features_raw):
            geom = shapely_mapping(features_raw[i]["geometry"])
        else:
            geom = None
        geojson_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": obj.to_export_dict(),
        })

    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }

    try:
        output_path.write_text(
            json.dumps(geojson, indent=2, default=_json_default),
            encoding="utf-8",
        )
    except Exception as exc:
        raise ShapefileExportError(str(output_path), str(exc)) from exc

    logger.info(
        "GeoJSON written: '%s' — %d feature(s).",
        output_path, len(geojson_features),
    )
    return output_path.resolve()


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard types.

    Args:
        obj: Object that the default JSON encoder cannot handle.

    Returns:
        JSON-serialisable equivalent.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")
