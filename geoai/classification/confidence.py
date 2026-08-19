"""
Per-object confidence estimation for the GeoAI Platform Stage 2 pipeline.

Computes a confidence score in [0.0, 1.0] for each classified ChangeObject
using three sources of evidence:

1. RF prediction probability — the mean P(change) at the object's pixels
   from the model's ``predict_proba`` output. Higher probability = higher
   confidence that the region is truly a change.

2. Spatial coherence — the fraction of pixels within the object's bounding
   box that were predicted as change. A compact, solid change region scores
   higher than a sparse or fragmented one.

3. Area plausibility — a size-based modifier that reduces confidence for
   objects very close to the minimum area threshold (50 px) and for objects
   that are unusually large (potential commission errors from poor pseudo-labels).

The three components are combined as a weighted average. The final score is
clipped to [0.0, 1.0].

Single responsibility: compute per-object confidence scores.

Position in dependency hierarchy: classification (depends on classification/schema).
"""

import logging
from typing import List, Optional

import numpy as np

from geoai.classification.schema import ChangeObject
from geoai.utils.constants import DEFAULT_MIN_OBJECT_SIZE_PX

logger = logging.getLogger(__name__)

# Weights for the three confidence components.
_WEIGHT_PROBA: float = 0.50
_WEIGHT_COHERENCE: float = 0.30
_WEIGHT_AREA: float = 0.20

# Area plausibility bounds.
_AREA_SOFT_MIN_PX: int = 100      # Objects below this get a reduced area score.
_AREA_SOFT_MAX_PX: int = 50_000   # Objects above this get a reduced area score.


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------


def assign_confidences(
    objects: List[ChangeObject],
    prediction_map: Optional[np.ndarray] = None,
    probability_map: Optional[np.ndarray] = None,
    label_array: Optional[np.ndarray] = None,
) -> List[ChangeObject]:
    """Compute and assign confidence scores to a list of classified ChangeObjects.

    Confidence is computed from available evidence. When a probability map is
    provided, it is the primary signal. When not available, confidence is
    estimated from geometric and class-level evidence only.

    Args:
        objects: List of ChangeObject instances with semantic_class populated.
        prediction_map: Optional float32 array of shape (H, W) from inference.
            NaN at invalid pixel positions.
        probability_map: Optional float32 array of shape (H, W) with P(change)
            values from predict_proba. None if probabilities were not computed.
        label_array: Optional integer label array of shape (H, W) matching
            the label IDs in each object's ``object_id`` field. Required for
            spatial coherence computation.

    Returns:
        The same list with ``confidence`` updated on each object.
    """
    logger.info(
        "Assigning confidence scores to %d objects "
        "(proba_map=%s, label_array=%s).",
        len(objects),
        probability_map is not None,
        label_array is not None,
    )

    for obj in objects:
        confidence = _compute_object_confidence(
            obj=obj,
            probability_map=probability_map,
            label_array=label_array,
        )
        # If the classifier already set a confidence (from rule strength),
        # blend it with the evidence-based score.
        if obj.confidence > 0.0:
            obj.confidence = round(0.6 * obj.confidence + 0.4 * confidence, 4)
        else:
            obj.confidence = round(confidence, 4)

        logger.debug(
            "Object %d confidence=%.4f (class='%s').",
            obj.object_id, obj.confidence, obj.semantic_class,
        )

    confidences = [o.confidence for o in objects]
    if confidences:
        logger.info(
            "Confidence assignment complete — mean=%.4f std=%.4f "
            "min=%.4f max=%.4f.",
            np.mean(confidences), np.std(confidences),
            np.min(confidences), np.max(confidences),
        )
    return objects


# ---------------------------------------------------------------------------
# Per-object computation
# ---------------------------------------------------------------------------


def _compute_object_confidence(
    obj: ChangeObject,
    probability_map: Optional[np.ndarray],
    label_array: Optional[np.ndarray],
) -> float:
    """Compute confidence for a single ChangeObject.

    Args:
        obj: ChangeObject with spatial and spectral attributes populated.
        probability_map: P(change) raster or None.
        label_array: Integer label array or None.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    proba_score = _proba_component(obj, probability_map, label_array)
    coherence_score = _coherence_component(obj, label_array)
    area_score = _area_plausibility_component(obj.area_px)

    combined = (
        _WEIGHT_PROBA * proba_score
        + _WEIGHT_COHERENCE * coherence_score
        + _WEIGHT_AREA * area_score
    )
    return float(np.clip(combined, 0.0, 1.0))


def _proba_component(
    obj: ChangeObject,
    probability_map: Optional[np.ndarray],
    label_array: Optional[np.ndarray],
) -> float:
    """Compute the probability-based confidence component.

    Returns the mean P(change) at the object's pixels when a probability
    map is available. Returns 0.5 (neutral) otherwise.

    Args:
        obj: ChangeObject with object_id set.
        probability_map: Float32 P(change) raster (H, W) or None.
        label_array: Integer label array (H, W) or None.

    Returns:
        Float in [0.0, 1.0]. 0.5 when evidence is unavailable.
    """
    if probability_map is None or label_array is None:
        return 0.5

    obj_pixels = label_array == obj.object_id
    if not obj_pixels.any():
        return 0.5

    proba_values = probability_map[obj_pixels]
    valid = proba_values[~np.isnan(proba_values)]
    if valid.size == 0:
        return 0.5

    return float(np.clip(np.mean(valid), 0.0, 1.0))


def _coherence_component(
    obj: ChangeObject,
    label_array: Optional[np.ndarray],
) -> float:
    """Compute the spatial coherence confidence component.

    Measures what fraction of the object's bounding box is covered by the
    object itself. A solid, non-fragmented object scores near 1.0.

    Args:
        obj: ChangeObject with bounding box attributes set.
        label_array: Integer label array (H, W) or None.

    Returns:
        Float in [0.0, 1.0]. Falls back to compactness when label_array is None.
    """
    if label_array is None:
        # Use compactness as a proxy for coherence.
        return float(np.clip(obj.compactness, 0.0, 1.0))

    # Extract the sub-array of the bounding box.
    r0, c0 = obj.bbox_min_row, obj.bbox_min_col
    r1, c1 = obj.bbox_max_row, obj.bbox_max_col
    bbox_area = max((r1 - r0) * (c1 - c0), 1)

    obj_pixels_in_bbox = int((label_array[r0:r1, c0:c1] == obj.object_id).sum())
    coherence = obj_pixels_in_bbox / bbox_area
    return float(np.clip(coherence, 0.0, 1.0))


def _area_plausibility_component(area_px: int) -> float:
    """Compute the area plausibility confidence component.

    Objects very close to the minimum size threshold may be noise.
    Objects that are extremely large may be commission errors.
    Both cases receive a reduced area plausibility score.

    Args:
        area_px: Object area in pixels.

    Returns:
        Float in [0.0, 1.0].
    """
    if area_px < DEFAULT_MIN_OBJECT_SIZE_PX:
        return 0.0
    if area_px < _AREA_SOFT_MIN_PX:
        # Linearly ramp from 0.3 to 1.0 between min_size and soft_min.
        t = (area_px - DEFAULT_MIN_OBJECT_SIZE_PX) / (
            _AREA_SOFT_MIN_PX - DEFAULT_MIN_OBJECT_SIZE_PX
        )
        return float(0.3 + 0.7 * t)
    if area_px > _AREA_SOFT_MAX_PX:
        # Linearly ramp from 1.0 down to 0.5 for very large objects.
        t = min(1.0, (area_px - _AREA_SOFT_MAX_PX) / _AREA_SOFT_MAX_PX)
        return float(np.clip(1.0 - 0.5 * t, 0.5, 1.0))
    return 1.0
