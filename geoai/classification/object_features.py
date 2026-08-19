"""
Per-object feature extraction for Stage 2 classification.

Extracts spectral change statistics and geometric attributes for each
significant change object by sampling the feature cube within the object's
pixel boundary. These per-object features are used by the Stage 2 classifier
to assign semantic classes.

No labelled data is required. Feature extraction operates entirely on the
prediction raster, label array, and original feature cube.

Single responsibility: extract per-object spectral and geometric features
from the feature cube by sampling within each object's boundary.

Position in dependency hierarchy: classification (depends on features/feature_cube,
classification/schema, utils/constants).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from geoai.classification.schema import ChangeObject
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Feature cube index constants (from canonical feature order).
_IDX_NDVI_T2 = 11
_IDX_NDBI_T2 = 12
_IDX_NDWI_T2 = 13
_IDX_RED_T2 = 7
_IDX_SAR_T2 = 10
_IDX_DELTA_NDVI = 14
_IDX_DELTA_NDBI = 15
_IDX_DELTA_NDWI = 16
_IDX_DELTA_SAR = 17


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------


def extract_object_features(
    objects: List[ChangeObject],
    feature_cube: np.ndarray,
    label_array: np.ndarray,
) -> List[ChangeObject]:
    """Extract per-object spectral features and populate ChangeObject attributes.

    For each ChangeObject, samples the feature cube at the pixels belonging
    to that object's label region and computes mean values for the key
    spectral and temporal features. These statistics are stored directly
    on the ChangeObject instance.

    Args:
        objects: List of ChangeObject instances. ``object_id`` must match
            the label values in ``label_array``.
        feature_cube: Float32 array of shape (H, W, 18) — the same cube
            used for prediction.
        label_array: Integer array of shape (H, W) from connected component
            labelling. Pixel value == object_id for pixels belonging to
            that object, 0 for background.

    Returns:
        The same list of ChangeObject instances with spectral attributes
        populated (mean_ndvi_change, mean_ndbi_change, mean_ndwi_change,
        mean_sar_change, mean_red_t2, mean_ndvi_t2).
    """
    if feature_cube.shape[2] != 18:
        logger.error(
            "Feature cube has %d features; expected 18. Skipping feature extraction.",
            feature_cube.shape[2],
        )
        return objects

    logger.info(
        "Extracting spectral features for %d objects from feature cube %s.",
        len(objects), feature_cube.shape,
    )

    for obj in objects:
        # Build a boolean mask for this object's pixels.
        obj_pixels = label_array == obj.object_id
        n_pixels = int(obj_pixels.sum())

        if n_pixels == 0:
            logger.warning(
                "Object %d: no pixels found in label array — skipping.",
                obj.object_id,
            )
            continue

        # Sample the feature cube at this object's pixel positions.
        # Result shape: (n_pixels, 18).
        pixel_features = feature_cube[obj_pixels]

        obj.mean_ndvi_change = float(
            np.nanmean(pixel_features[:, _IDX_DELTA_NDVI])
        )
        obj.mean_ndbi_change = float(
            np.nanmean(pixel_features[:, _IDX_DELTA_NDBI])
        )
        obj.mean_ndwi_change = float(
            np.nanmean(pixel_features[:, _IDX_DELTA_NDWI])
        )
        obj.mean_sar_change = float(
            np.nanmean(pixel_features[:, _IDX_DELTA_SAR])
        )
        obj.mean_red_t2 = float(
            np.nanmean(pixel_features[:, _IDX_RED_T2])
        )
        obj.mean_ndvi_t2 = float(
            np.nanmean(pixel_features[:, _IDX_NDVI_T2])
        )

        logger.debug(
            "Object %d (%d px): ΔNDVI=%.4f ΔNDBI=%.4f ΔSAR=%.4f.",
            obj.object_id, n_pixels,
            obj.mean_ndvi_change,
            obj.mean_ndbi_change,
            obj.mean_sar_change,
        )

    logger.info(
        "Spectral feature extraction complete — %d objects processed.",
        len(objects),
    )
    return objects
