"""
Stage 2 semantic classifier for the GeoAI Platform.

Provides a rule-based semantic classifier that assigns PS10 Stage 2 change
classes to detected objects using spectral change thresholds and geometric
shape criteria. No labelled training data is required.

The six PS10 Stage 2 classes (from the official problem statement):
  1. Kacha_Track          — narrow, elongated linear features with low NDVI
  2. Building             — compact, high NDBI change, moderate area
  3. Road                 — elongated, moderate NDBI change, no SAR response
  4. Land_Clearing        — high negative NDVI change, large area
  5. Large_Infrastructure — very large area, complex shape, strong SAR change
  6. New_Settlement       — cluster of building-like objects, mixed spectral

An ML-based classifier interface is also defined here. Its ``fit()`` method
logs that training data is not yet available (PS10 Stage 2 labels will be
released to participants who qualify for Stage 2). The ``classify()`` method
falls back to the rule-based classifier until the ML model is trained.

Single responsibility: assign semantic classes to ChangeObject instances.

Position in dependency hierarchy: classification (depends on classification/base,
classification/schema, utils/constants).
"""

import logging
from typing import List, Optional

import numpy as np

from geoai.classification.base import BaseClassifier
from geoai.classification.schema import ChangeObject
from geoai.utils.constants import ChangeClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification thresholds (rule-based classifier)
# ---------------------------------------------------------------------------

# These thresholds encode domain knowledge about the spectral signatures of
# each PS10 Stage 2 class. They are initial estimates and should be refined
# as domain expertise develops.
#
# NDVI change is negative for vegetation loss (land clearing).
# NDBI change is positive for built-up area growth (buildings, roads, infra).
# SAR change is positive for surface roughness increase (construction).
# Compactness near 1.0 indicates circular/square shapes (buildings).
# Low compactness indicates elongated shapes (roads, tracks).

_THRESHOLDS = {
    # Land clearing: strong negative NDVI change (vegetation removed).
    "land_clearing_ndvi_thresh": -0.10,
    "land_clearing_min_area_px": 100,

    # Buildings: positive NDBI change, compact shape, moderate area.
    "building_ndbi_thresh": 0.05,
    "building_compactness_thresh": 0.20,
    "building_max_area_px": 5000,

    # Large infrastructure: very large area, strong SAR change.
    "large_infra_min_area_px": 2000,
    "large_infra_sar_thresh": 0.05,

    # Roads: elongated shape (low compactness), moderate NDBI change.
    "road_max_compactness": 0.15,
    "road_ndbi_thresh": 0.02,

    # Kacha tracks: very elongated (lowest compactness), small/narrow.
    "kacha_max_compactness": 0.08,

    # New settlements: moderate area, mixed spectral (positive NDBI, mixed NDVI).
    "settlement_ndbi_thresh": 0.03,
    "settlement_min_area_px": 200,
    "settlement_max_area_px": 5000,
}


# ---------------------------------------------------------------------------
# Rule-based classifier (production Stage 2 implementation)
# ---------------------------------------------------------------------------


class RuleBasedClassifier(BaseClassifier):
    """Rule-based semantic classifier for PS10 Stage 2 change objects.

    Assigns semantic classes using spectral change thresholds and geometric
    shape criteria. The classification rules encode known spectral and
    structural properties of each PS10 change class.

    Classification priority order (applied top to bottom):
    1. Land_Clearing   — highest priority (most distinctive spectral signal)
    2. Large_Infrastructure
    3. Kacha_Track
    4. Road
    5. Building
    6. New_Settlement
    7. Unknown         — fallback
    """

    def __init__(self, thresholds: Optional[dict] = None) -> None:
        """Initialise the rule-based classifier.

        Args:
            thresholds: Optional dictionary of threshold overrides. Any key
                present in ``_THRESHOLDS`` can be overridden. Defaults to
                the module-level ``_THRESHOLDS`` dict.
        """
        super().__init__(classifier_name="rule_based")
        self.thresholds = dict(_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        logger.info("RuleBasedClassifier initialised with %d threshold rules.", len(self.thresholds))

    def classify(self, objects: List[ChangeObject]) -> List[ChangeObject]:
        """Assign semantic classes to a list of ChangeObject instances.

        Args:
            objects: List of ChangeObject instances with spectral attributes
                populated by :func:`~geoai.classification.object_features.extract_object_features`.

        Returns:
            The same list with ``semantic_class``, ``confidence``, and
            ``classifier_method`` populated.
        """
        logger.info("Classifying %d objects (rule-based).", len(objects))
        for obj in objects:
            cls, conf = self._classify_single(obj)
            obj.semantic_class = cls
            obj.confidence = conf
            obj.classifier_method = self.classifier_name
            logger.debug(
                "Object %d → class='%s' confidence=%.3f.", obj.object_id, cls, conf
            )
        logger.info(
            "Classification complete — class distribution: %s.",
            _class_distribution(objects),
        )
        return objects

    def get_class_names(self) -> List[str]:
        """Return all assignable class names.

        Returns:
            List of ChangeClass string values.
        """
        return [c.value for c in ChangeClass]

    # ---------------------------------------------------------------------------
    # Internal classification logic
    # ---------------------------------------------------------------------------

    def _classify_single(self, obj: ChangeObject):
        """Apply classification rules to a single object.

        Returns:
            Tuple of (semantic_class_string, confidence_float).
        """
        t = self.thresholds
        ndvi_chg = obj.mean_ndvi_change or 0.0
        ndbi_chg = obj.mean_ndbi_change or 0.0
        sar_chg = obj.mean_sar_change or 0.0
        compactness = obj.compactness
        area_px = obj.area_px

        # Rule 1: Land clearing — strong negative NDVI change over large area.
        if (ndvi_chg < t["land_clearing_ndvi_thresh"]
                and area_px >= t["land_clearing_min_area_px"]):
            confidence = min(1.0, abs(ndvi_chg) / 0.3)
            return ChangeClass.LAND_CLEARING, round(confidence, 4)

        # Rule 2: Large infrastructure — very large area with SAR change.
        if (area_px >= t["large_infra_min_area_px"]
                and abs(sar_chg) >= t["large_infra_sar_thresh"]):
            confidence = min(1.0, area_px / 10000.0)
            return ChangeClass.LARGE_INFRA, round(confidence, 4)

        # Rule 3: Kacha track — very elongated shape.
        if compactness <= t["kacha_max_compactness"]:
            confidence = max(0.0, 1.0 - compactness / t["kacha_max_compactness"])
            return ChangeClass.KACHA_TRACK, round(confidence, 4)

        # Rule 4: Road — elongated shape with some NDBI increase.
        if (compactness <= t["road_max_compactness"]
                and ndbi_chg >= t["road_ndbi_thresh"]):
            confidence = max(0.0, 1.0 - compactness / t["road_max_compactness"])
            return ChangeClass.ROAD, round(confidence, 4)

        # Rule 5: Building — compact shape, NDBI increase, moderate area.
        if (ndbi_chg >= t["building_ndbi_thresh"]
                and compactness >= t["building_compactness_thresh"]
                and area_px <= t["building_max_area_px"]):
            confidence = min(1.0, ndbi_chg / 0.2)
            return ChangeClass.BUILDING, round(confidence, 4)

        # Rule 6: New settlement — moderate NDBI increase, intermediate area.
        if (ndbi_chg >= t["settlement_ndbi_thresh"]
                and t["settlement_min_area_px"] <= area_px <= t["settlement_max_area_px"]):
            confidence = min(1.0, ndbi_chg / 0.15)
            return ChangeClass.NEW_SETTLEMENT, round(confidence, 4)

        # Fallback: Unknown.
        return ChangeClass.UNKNOWN, 0.1


# ---------------------------------------------------------------------------
# ML-based classifier (interface ready, training deferred)
# ---------------------------------------------------------------------------


class MLObjectClassifier(BaseClassifier):
    """Machine learning-based Stage 2 object classifier.

    # FUTURE EXTENSION:
    # This class is architecturally complete but training is deferred until
    # the PS10 Stage 2 labelled dataset is released to participants who
    # qualify for Stage 2. When that dataset is available:
    # 1. Implement fit(objects, labels) to train the underlying ML model.
    # 2. Implement predict(objects) to replace rule-based fallback.
    # 3. The interface here does not need to change — only the internals.
    #
    # Current behaviour: classify() delegates to RuleBasedClassifier.
    """

    def __init__(self) -> None:
        """Initialise the ML classifier with an unloaded model state."""
        super().__init__(classifier_name="ml_classifier")
        self._model = None
        self._rule_fallback = RuleBasedClassifier()
        logger.info(
            "MLObjectClassifier initialised — model not yet trained. "
            "classify() will use RuleBasedClassifier as fallback."
        )

    def fit(self, objects: List[ChangeObject], labels: List[str]) -> None:
        """Train the ML classifier on labelled Stage 2 objects.

        # FUTURE EXTENSION: Training is deferred until labelled data is available.
        # When PS10 Stage 2 labels are released, implement training here.

        Args:
            objects: List of ChangeObject instances with features extracted.
            labels: List of ground truth semantic class strings.
        """
        logger.info(
            "MLObjectClassifier.fit() called but Stage 2 labelled data is not "
            "yet available. Training is deferred until the PS10 Stage 2 dataset "
            "is released. Classifier remains in rule-based fallback mode."
        )

    def classify(self, objects: List[ChangeObject]) -> List[ChangeObject]:
        """Classify objects, falling back to the rule-based classifier.

        Args:
            objects: List of ChangeObject instances to classify.

        Returns:
            Classified ChangeObject list.
        """
        logger.info(
            "MLObjectClassifier: delegating to RuleBasedClassifier (not yet trained)."
        )
        return self._rule_fallback.classify(objects)

    def get_class_names(self) -> List[str]:
        """Return assignable class names.

        Returns:
            List of ChangeClass string values.
        """
        return [c.value for c in ChangeClass]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _class_distribution(objects: List[ChangeObject]) -> dict:
    """Compute the class distribution across a list of ChangeObjects.

    Args:
        objects: Classified ChangeObject list.

    Returns:
        Dictionary mapping class name to object count.
    """
    dist: dict = {}
    for obj in objects:
        dist[obj.semantic_class] = dist.get(obj.semantic_class, 0) + 1
    return dist
