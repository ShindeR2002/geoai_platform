"""
Random Forest Baseline model for the GeoAI Platform.

Implements the :class:`~geoai.models.base.BaseModel` interface for the
14-feature RF Baseline trained in Version 1 Notebook 04. This model uses
only per-epoch features (no temporal delta block) and class_weight='balanced'.

This model is retained for comparison purposes and is not the production
inference model. All production inference uses RFEnhancedModel.

Single responsibility: load and run inference with the RF Baseline model.

Position in dependency hierarchy: models/baselines (depends on models/base,
core, utils).
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np

from geoai.core.exceptions import InferenceError, ModelError, ModelNotFoundError
from geoai.models.base import BaseModel, ModelCapabilities

logger = logging.getLogger(__name__)

# The 14-feature list used by the baseline model (no delta features).
# Source: Version 1 NB04. This is NOT the production feature set.
BASELINE_FEATURE_NAMES: tuple = (
    "Red_2021", "Green_2021", "Blue_2021", "SAR_2021",
    "NDVI_2021", "NDBI_2021", "NDWI_2021",
    "Red_2024", "Green_2024", "Blue_2024", "SAR_2024",
    "NDVI_2024", "NDBI_2024", "NDWI_2024",
)
BASELINE_FEATURE_COUNT: int = 14


class RFBaselineModel(BaseModel):
    """Random Forest Baseline change detection model (14 features).

    Uses class_weight='balanced' and 14 per-epoch features without temporal
    deltas. Trained in Version 1 Notebook 04.
    """

    def __init__(self) -> None:
        """Initialise the RF Baseline model."""
        super().__init__()
        self._rf = None

    def load(self, model_path: Union[str, Path]) -> None:
        """Load the RF Baseline model from a joblib-serialised pkl file.

        Args:
            model_path: Path to the baseline pkl file.

        Raises:
            ModelNotFoundError: If the file does not exist.
            ModelError: If the file cannot be deserialised.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))

        logger.info("Loading RF Baseline model from '%s'.", model_path)
        try:
            self._rf = joblib.load(model_path)
        except Exception as exc:
            raise ModelError(
                f"Failed to load RF Baseline from '{model_path}': {exc}."
            ) from exc

        self.feature_names = list(BASELINE_FEATURE_NAMES)
        self.model_id = "rf_baseline_v1"
        self.is_loaded = True
        logger.info("RF Baseline loaded — n_estimators=%d.", self._rf.n_estimators)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict change labels using the baseline model.

        Args:
            X: Float32 matrix of shape (N, 14).

        Returns:
            1D integer array of shape (N,).

        Raises:
            ModelError: If not loaded.
            FeatureCountMismatchError: If feature count != 14.
            InferenceError: If prediction fails.
        """
        self._assert_loaded()
        self.validate_features(X)
        try:
            return self._rf.predict(X)
        except Exception as exc:
            raise InferenceError(f"RF Baseline predict() failed: {exc}.") from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the baseline model.

        Args:
            X: Float32 matrix of shape (N, 14).

        Returns:
            Float32 array of shape (N, 2).

        Raises:
            ModelError: If not loaded.
            FeatureCountMismatchError: If feature count != 14.
            InferenceError: If probability estimation fails.
        """
        self._assert_loaded()
        self.validate_features(X)
        try:
            return self._rf.predict_proba(X).astype(np.float32)
        except Exception as exc:
            raise InferenceError(f"RF Baseline predict_proba() failed: {exc}.") from exc

    def get_feature_names(self) -> List[str]:
        """Return the 14-feature baseline feature names.

        Returns:
            List of 14 feature name strings.
        """
        self._assert_loaded()
        return list(self.feature_names)

    def get_capabilities(self) -> ModelCapabilities:
        """Return the structured capabilities of the RF Baseline model."""
        from geoai.models.base import ModelCapabilities
        from geoai.models.baselines.rf_baseline import BASELINE_FEATURE_NAMES
        return ModelCapabilities(
            model_type="Classical ML (Random Forest Baseline)",
            expected_input_channels=self.feature_names or list(BASELINE_FEATURE_NAMES),
            training_status="Implemented / Baseline",
            requirements=["scikit-learn", "joblib"]
        )

    def _assert_loaded(self) -> None:
        """Raise ModelError if model is not loaded."""
        if not self.is_loaded or self._rf is None:
            raise ModelError(
                "RF Baseline model is not loaded. Call load() first."
            )
