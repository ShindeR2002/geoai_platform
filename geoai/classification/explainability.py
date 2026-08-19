"""
SHAP-based feature attribution for the GeoAI Platform Stage 2 pipeline.

Computes per-object SHAP (SHapley Additive exPlanations) values for the
RF Enhanced model to explain which features drove each pixel's change
prediction. SHAP values are aggregated within each object's boundary and
stored on the ChangeObject instance.

SHAP values indicate the contribution of each feature to the model's
prediction relative to a background expectation. Positive values push
the prediction toward change; negative values push toward no-change.

If the ``shap`` library is not installed, this module logs a warning and
returns objects with SHAP values set to None rather than raising an error.
This prevents the Stage 2 pipeline from failing on systems where SHAP is
not available.

Single responsibility: compute and attach SHAP feature attributions to
classified ChangeObject instances.

Position in dependency hierarchy: classification (depends on classification/schema,
models/production/rf_enhanced, utils/constants).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from geoai.classification.schema import ChangeObject
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Attempt to import SHAP. If not available, explainability is disabled
# gracefully rather than preventing the pipeline from running.
try:
    import shap
    _SHAP_AVAILABLE = True
    logger.debug("SHAP library is available — explainability is enabled.")
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning(
        "SHAP library is not installed. Feature attribution will be skipped. "
        "Install with: pip install shap"
    )


# ---------------------------------------------------------------------------
# SHAP explainer construction
# ---------------------------------------------------------------------------


def build_tree_explainer(rf_model) -> Optional[object]:
    """Build a SHAP TreeExplainer for a trained RandomForestClassifier.

    TreeExplainer is the recommended SHAP explainer for tree-based models.
    It computes exact SHAP values efficiently without sampling.

    Args:
        rf_model: A trained sklearn RandomForestClassifier instance (the
            ``._rf`` attribute of :class:`~geoai.models.production.rf_enhanced.RFEnhancedModel`).

    Returns:
        A ``shap.TreeExplainer`` instance, or None if SHAP is not available.
    """
    if not _SHAP_AVAILABLE:
        logger.warning(
            "Cannot build TreeExplainer — SHAP is not installed."
        )
        return None

    logger.info("Building SHAP TreeExplainer for RF model.")
    try:
        explainer = shap.TreeExplainer(rf_model)
        logger.info("TreeExplainer built successfully.")
        return explainer
    except Exception as exc:
        logger.warning(
            "Failed to build SHAP TreeExplainer: %s. "
            "Feature attribution will be skipped.",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Per-object SHAP computation
# ---------------------------------------------------------------------------


def compute_object_shap_values(
    objects: List[ChangeObject],
    explainer,
    feature_cube: np.ndarray,
    label_array: np.ndarray,
    max_pixels_per_object: int = 200,
) -> List[ChangeObject]:
    """Compute mean SHAP values for each ChangeObject and attach to the instance.

    For each object, a sample of pixels (up to ``max_pixels_per_object``) is
    selected from within the object's boundary. SHAP values are computed for
    this sample and averaged, producing one SHAP value per feature per object.

    Args:
        objects: List of classified ChangeObject instances.
        explainer: SHAP TreeExplainer from :func:`build_tree_explainer`.
            If None, this function returns the objects unchanged.
        feature_cube: Float32 array of shape (H, W, 18).
        label_array: Integer label array of shape (H, W).
        max_pixels_per_object: Maximum number of pixels to sample per object.
            Larger values give more stable SHAP estimates but are slower.
            Default 200.

    Returns:
        The same list of ChangeObject instances with ``shap_values`` populated
        as a dictionary mapping feature name to mean SHAP value. Objects where
        SHAP computation fails have ``shap_values`` set to None.
    """
    if explainer is None or not _SHAP_AVAILABLE:
        logger.info(
            "SHAP explainer is not available — skipping feature attribution."
        )
        return objects

    logger.info(
        "Computing SHAP values for %d objects "
        "(max_pixels_per_object=%d).",
        len(objects), max_pixels_per_object,
    )

    feature_names = list(CANONICAL_FEATURE_NAMES)
    n_computed = 0

    for obj in objects:
        obj_pixels_mask = label_array == obj.object_id
        pixel_indices = np.argwhere(obj_pixels_mask)

        if len(pixel_indices) == 0:
            logger.warning(
                "Object %d: no pixels in label array — skipping SHAP.",
                obj.object_id,
            )
            continue

        # Sample pixels if the object is large.
        if len(pixel_indices) > max_pixels_per_object:
            rng = np.random.default_rng(seed=obj.object_id)
            sampled_idx = rng.choice(
                len(pixel_indices),
                size=max_pixels_per_object,
                replace=False,
            )
            pixel_indices = pixel_indices[sampled_idx]

        # Extract feature vectors for the sampled pixels.
        rows = pixel_indices[:, 0]
        cols = pixel_indices[:, 1]
        X_sample = feature_cube[rows, cols, :].astype(np.float32)

        try:
            # SHAP values shape: (n_pixels, n_features, n_classes) for RF
            # or (n_pixels, n_features) for binary. Extract change class values.
            shap_vals = explainer.shap_values(X_sample)

            # For binary RF classifiers, shap_values returns a list of two
            # arrays (one per class). Index 1 is the change class.
            if isinstance(shap_vals, list):
                shap_change = np.array(shap_vals[1])  # shape: (n_pixels, 18)
            else:
                shap_change = np.array(shap_vals)

            # Average across sampled pixels.
            mean_shap = shap_change.mean(axis=0)

            obj.shap_values = {
                name: round(float(val), 6)
                for name, val in zip(feature_names, mean_shap)
            }
            n_computed += 1

        except Exception as exc:
            logger.warning(
                "SHAP computation failed for object %d: %s.",
                obj.object_id, exc,
            )
            obj.shap_values = None

    logger.info(
        "SHAP computation complete — %d / %d objects attributed.",
        n_computed, len(objects),
    )
    return objects


# ---------------------------------------------------------------------------
# Top feature extraction utility
# ---------------------------------------------------------------------------


def get_top_shap_features(
    shap_values: Dict[str, float],
    n: int = 5,
    direction: str = "both",
) -> List[tuple]:
    """Return the top N features by SHAP importance for an object.

    Args:
        shap_values: Dictionary mapping feature name to SHAP value.
        n: Number of top features to return. Default 5.
        direction: Which direction to rank:
            ``'positive'`` — features that increase P(change).
            ``'negative'`` — features that decrease P(change).
            ``'both'`` — features with largest absolute SHAP values.

    Returns:
        List of (feature_name, shap_value) tuples sorted by importance.
    """
    if not shap_values:
        return []

    items = list(shap_values.items())

    if direction == "positive":
        items = [(k, v) for k, v in items if v > 0]
        items.sort(key=lambda x: x[1], reverse=True)
    elif direction == "negative":
        items = [(k, v) for k, v in items if v < 0]
        items.sort(key=lambda x: x[1])
    else:
        items.sort(key=lambda x: abs(x[1]), reverse=True)

    return items[:n]
