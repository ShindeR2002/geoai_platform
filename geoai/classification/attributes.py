"""
Object attribute generation for the GeoAI Platform Stage 2 pipeline.

Orchestrates the population of ChangeObject instances with the complete
attribute set required for Stage 2 export. This module acts as the assembly
point between the statistics module (spatial attributes), the object_features
module (spectral attributes), and the classifier module (semantic class).

Single responsibility: produce fully attributed ChangeObject lists from
raw analysis outputs.

Position in dependency hierarchy: classification (depends on classification/schema,
classification/object_features, analysis/statistics).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from rasterio.transform import Affine

from geoai.analysis.statistics import build_object_records
from geoai.classification.object_features import extract_object_features
from geoai.classification.schema import (
    ChangeObject,
    change_objects_from_records,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary attribute population entry point
# ---------------------------------------------------------------------------


def build_attributed_objects(
    significant_regions,
    label_array: np.ndarray,
    feature_cube: np.ndarray,
    transform: Optional[Affine] = None,
    resolution_m: float = 10.0,
) -> List[ChangeObject]:
    """Build a fully attributed list of ChangeObject instances.

    Chains the statistics, geographic conversion, and spectral feature
    extraction steps into a single function call. The resulting objects
    are ready for semantic classification.

    Args:
        significant_regions: List of skimage RegionProperties from
            :func:`~geoai.analysis.objects.extract_objects`.
        label_array: Integer label array of shape (H, W).
        feature_cube: Float32 feature cube of shape (H, W, 18).
        transform: Optional rasterio affine transform for geographic coordinate
            conversion. When None, centroid_lat and centroid_lon remain None.
        resolution_m: Ground sampling distance in metres. Default 10.0.

    Returns:
        List of ChangeObject instances with spatial and spectral attributes
        populated. Semantic class is not assigned here — call the classifier
        after this function.
    """
    logger.info(
        "Building attributed objects for %d regions.", len(significant_regions)
    )

    # Step 1: Build spatial attribute records from region properties.
    records = build_object_records(
        significant_regions=significant_regions,
        resolution_m=resolution_m,
        transform=transform,
    )

    # Step 2: Convert records to ChangeObject instances.
    objects = change_objects_from_records(records)

    # Step 3: Extract per-object spectral features from the feature cube.
    objects = extract_object_features(
        objects=objects,
        feature_cube=feature_cube,
        label_array=label_array,
    )

    logger.info(
        "Attribute population complete — %d objects ready for classification.",
        len(objects),
    )
    return objects


# ---------------------------------------------------------------------------
# Attribute validation
# ---------------------------------------------------------------------------


def validate_object_attributes(objects: List[ChangeObject]) -> List[str]:
    """Validate that all ChangeObject instances have required attributes set.

    Checks that spatial attributes are non-zero and that spectral attributes
    have been extracted. Returns a list of warning messages for any issues
    found. An empty list means all objects are valid.

    Args:
        objects: List of ChangeObject instances to validate.

    Returns:
        List of warning message strings. Empty if all objects are valid.
    """
    warnings = []
    for obj in objects:
        if obj.area_px <= 0:
            warnings.append(
                f"Object {obj.object_id}: area_px={obj.area_px} is not positive."
            )
        if obj.mean_ndvi_change is None:
            warnings.append(
                f"Object {obj.object_id}: mean_ndvi_change is None — "
                "extract_object_features may not have run."
            )
        if obj.centroid_lat is None and obj.centroid_row == 0.0:
            warnings.append(
                f"Object {obj.object_id}: centroid_row=0.0 may indicate "
                "an object at the image edge or a population error."
            )

    if warnings:
        for msg in warnings:
            logger.warning("Attribute validation: %s", msg)
    else:
        logger.info(
            "Attribute validation passed — all %d objects are valid.", len(objects)
        )
    return warnings


# ---------------------------------------------------------------------------
# Attribute summary for reporting
# ---------------------------------------------------------------------------


def summarise_object_attributes(objects: List[ChangeObject]) -> Dict[str, Any]:
    """Compute summary statistics over the spectral attributes of all objects.

    Useful for generating the attribute summary section of the Stage 2 report.

    Args:
        objects: List of classified ChangeObject instances.

    Returns:
        Dictionary with mean, std, min, and max for each spectral attribute.
    """
    if not objects:
        return {}

    def _stats(values: list) -> dict:
        arr = np.array([v for v in values if v is not None], dtype=float)
        if arr.size == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
        }

    summary = {
        "mean_ndvi_change": _stats([o.mean_ndvi_change for o in objects]),
        "mean_ndbi_change": _stats([o.mean_ndbi_change for o in objects]),
        "mean_ndwi_change": _stats([o.mean_ndwi_change for o in objects]),
        "mean_sar_change": _stats([o.mean_sar_change for o in objects]),
        "compactness": _stats([o.compactness for o in objects]),
        "area_px": _stats([o.area_px for o in objects]),
    }

    logger.debug("Attribute summary computed for %d objects.", len(objects))
    return summary
