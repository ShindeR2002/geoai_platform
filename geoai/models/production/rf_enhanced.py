"""
Random Forest Enhanced production model for the GeoAI Platform.

Implements the :class:`~geoai.models.base.BaseModel` interface for the
``rf_enhanced.pkl`` model trained in Version 1 Notebook 05. This is the
production change detection model used for all Stage 1 inference.

Model specification (from Version 1 NB05, frozen by contract):
  - Algorithm: sklearn RandomForestClassifier
  - n_estimators: 100
  - random_state: 42
  - n_jobs: -1
  - class_weight: None (not set)
  - Feature count: 18
  - Serialisation: joblib

Single responsibility: load and run inference with rf_enhanced.pkl.

Position in dependency hierarchy: models/production (depends on models/base,
core, utils).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np

from geoai.core.exceptions import (
    FeatureCountMismatchError,
    InferenceError,
    ModelError,
    ModelMetadataError,
    ModelNotFoundError,
)
from geoai.models.base import BaseModel, ModelCapabilities
from geoai.utils.constants import CANONICAL_FEATURE_COUNT, CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)


class RFEnhancedModel(BaseModel):
    """Random Forest Enhanced change detection model.

    Wraps the joblib-serialised ``rf_enhanced.pkl`` sklearn
    ``RandomForestClassifier`` trained in Version 1 Notebook 05 on the
    18-feature canonical feature matrix.

    Attributes:
        model_id: Set to ``'rf_enhanced_v1'`` after loading.
        feature_names: Set to the canonical 18 feature names after loading.
        is_loaded: True after :meth:`load` completes successfully.
        _rf: The underlying sklearn RandomForestClassifier instance.
    """

    def __init__(self) -> None:
        """Initialise the RF Enhanced model with unset internal state."""
        super().__init__()
        self._rf = None

    def load(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Load the RF Enhanced model from a joblib-serialised pkl file.

        Also loads the model metadata JSON if ``metadata_path`` is provided,
        and validates that the metadata's feature count and feature names
        match the canonical platform configuration.

        Args:
            model_path: Path to ``rf_enhanced.pkl``.
            metadata_path: Optional path to ``rf_enhanced_meta.json``.
                When provided, the metadata is validated for consistency.

        Raises:
            ModelNotFoundError: If the pkl file does not exist.
            ModelError: If the pkl file cannot be deserialised.
            ModelMetadataError: If the metadata JSON is invalid or inconsistent.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))

        logger.info("Loading RF Enhanced model from '%s'.", model_path)
        try:
            self._rf = joblib.load(model_path)
        except Exception as exc:
            raise ModelError(
                f"Failed to deserialise model from '{model_path}': {exc}."
            ) from exc

        # Set feature metadata from the canonical platform constants.
        # These are the ground-truth values regardless of what the metadata
        # file says — the model was trained on these features.
        self.feature_names = list(CANONICAL_FEATURE_NAMES)
        self.model_id = "rf_enhanced_v1"
        self.is_loaded = True

        logger.info(
            "RF Enhanced model loaded — n_estimators=%d n_features=%d.",
            self._rf.n_estimators,
            CANONICAL_FEATURE_COUNT,
        )

        # Validate metadata if provided.
        if metadata_path is not None:
            self._validate_metadata(metadata_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary change labels for a valid-pixel feature matrix.

        Args:
            X: Float32 feature matrix of shape (N_valid, 18).

        Returns:
            1D integer array of shape (N_valid,) with values 0 (no change)
            or 1 (change).

        Raises:
            ModelError: If the model is not loaded.
            FeatureCountMismatchError: If X does not have 18 features.
            InferenceError: If prediction fails.
        """
        self._assert_loaded()
        self.validate_features(X)

        logger.info(
            "Running RF Enhanced inference on %d pixels.", X.shape[0]
        )
        try:
            predictions = self._rf.predict(X)
        except Exception as exc:
            raise InferenceError(
                f"RF Enhanced predict() failed: {exc}."
            ) from exc

        n_change = int((predictions == 1).sum())
        logger.info(
            "Inference complete — change pixels: %d / %d (%.1f%%).",
            n_change, len(predictions), 100.0 * n_change / max(len(predictions), 1),
        )
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a valid-pixel feature matrix.

        Args:
            X: Float32 feature matrix of shape (N_valid, 18).

        Returns:
            Float32 array of shape (N_valid, 2) where column 0 = P(no change)
            and column 1 = P(change).

        Raises:
            ModelError: If the model is not loaded.
            FeatureCountMismatchError: If X does not have 18 features.
            InferenceError: If probability estimation fails.
        """
        self._assert_loaded()
        self.validate_features(X)

        logger.info(
            "Running RF Enhanced probability estimation on %d pixels.", X.shape[0]
        )
        try:
            probabilities = self._rf.predict_proba(X).astype(np.float32)
        except Exception as exc:
            raise InferenceError(
                f"RF Enhanced predict_proba() failed: {exc}."
            ) from exc

        logger.info(
            "Probability estimation complete — mean P(change)=%.4f.",
            float(probabilities[:, 1].mean()),
        )
        return probabilities

    def get_feature_names(self) -> List[str]:
        """Return the canonical 18 feature names expected by this model.

        Returns:
            List of 18 feature name strings in canonical order.

        Raises:
            ModelError: If the model has not been loaded.
        """
        self._assert_loaded()
        return list(self.feature_names)

    def get_feature_importances(self) -> np.ndarray:
        """Return the feature importances from the Random Forest.

        Feature importances are computed as the mean decrease in impurity
        (Gini importance) across all trees in the forest.

        Returns:
            Float32 array of shape (18,) with importance scores. Values sum
            to 1.0. Higher values indicate more important features.

        Raises:
            ModelError: If the model has not been loaded.
        """
        self._assert_loaded()
        importances = self._rf.feature_importances_.astype(np.float32)
        logger.debug(
            "Feature importances retrieved — top feature: '%s' (%.4f).",
            self.feature_names[int(importances.argmax())],
            float(importances.max()),
        )
        return importances

    def get_feature_importance_dict(self) -> dict:
        """Return feature importances as a dictionary keyed by feature name.

        Returns:
            Dictionary mapping feature name strings to importance float values,
            sorted by importance in descending order.

        Raises:
            ModelError: If the model has not been loaded.
        """
        importances = self.get_feature_importances()
        importance_pairs = sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(importance_pairs)

    def get_capabilities(self) -> ModelCapabilities:
        """Return the structured capabilities of the RF Enhanced model."""
        from geoai.models.base import ModelCapabilities
        from geoai.utils.constants import CANONICAL_FEATURE_NAMES
        return ModelCapabilities(
            model_type="Classical ML (Random Forest)",
            expected_input_channels=self.feature_names or list(CANONICAL_FEATURE_NAMES),
            training_status="Implemented / Production Ready",
            requirements=["scikit-learn", "joblib"]
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise ModelError if the model has not been loaded.

        Raises:
            ModelError: If ``is_loaded`` is False.
        """
        if not self.is_loaded or self._rf is None:
            raise ModelError(
                "RF Enhanced model is not loaded. Call load() before predict()."
            )

    def _validate_metadata(self, metadata_path: Union[str, Path]) -> None:
        """Validate a model metadata JSON file for consistency.

        Checks that the metadata's feature_count and feature_names match
        the canonical platform configuration. Logs warnings for
        inconsistencies rather than raising exceptions, because the canonical
        constants take precedence over the metadata file.

        Args:
            metadata_path: Path to the metadata JSON file.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            logger.warning(
                "Model metadata file not found at '%s'. Skipping validation.",
                metadata_path,
            )
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ModelMetadataError(
                path=str(metadata_path),
                reason=f"Could not parse JSON: {exc}",
            ) from exc

        meta_feature_count = meta.get("feature_count")
        if meta_feature_count != CANONICAL_FEATURE_COUNT:
            logger.warning(
                "Metadata feature_count=%s does not match canonical value %d. "
                "The canonical value takes precedence.",
                meta_feature_count,
                CANONICAL_FEATURE_COUNT,
            )

        meta_feature_names = meta.get("feature_names", [])
        if meta_feature_names and meta_feature_names != list(CANONICAL_FEATURE_NAMES):
            logger.warning(
                "Metadata feature_names differ from canonical names. "
                "The canonical names take precedence."
            )

        logger.info(
            "Model metadata validated from '%s' — model_id='%s'.",
            metadata_path.name,
            meta.get("model_id", "unknown"),
        )
