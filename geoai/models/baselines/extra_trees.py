import logging
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np

from geoai.core.exceptions import InferenceError, ModelError, ModelNotFoundError
from geoai.models.base import BaseModel, ModelCapabilities
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

logger = logging.getLogger(__name__)

class ExtraTreesModel(BaseModel):
    """Extra Trees classical baseline model (18 features)."""

    def __init__(self) -> None:
        super().__init__()
        self._et = None

    def load(self, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))

        logger.info("Loading Extra Trees model from '%s'.", model_path)
        try:
            self._et = joblib.load(model_path)
        except Exception as exc:
            raise ModelError(
                f"Failed to load Extra Trees from '{model_path}': {exc}."
            ) from exc

        self.feature_names = list(CANONICAL_FEATURE_NAMES)
        self.model_id = "extra_trees"
        self.is_loaded = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_loaded()
        self.validate_features(X)
        try:
            return self._et.predict(X)
        except Exception as exc:
            raise InferenceError(f"Extra Trees predict() failed: {exc}.") from exc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_loaded()
        self.validate_features(X)
        try:
            return self._et.predict_proba(X).astype(np.float32)
        except Exception as exc:
            raise InferenceError(f"Extra Trees predict_proba() failed: {exc}.") from exc

    def get_feature_names(self) -> List[str]:
        self._assert_loaded()
        return list(self.feature_names)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_type="Classical ML (Extra Trees Baseline)",
            expected_input_channels=self.feature_names or list(CANONICAL_FEATURE_NAMES),
            training_status="Implemented / Baseline",
            requirements=["scikit-learn", "joblib"]
        )

    def _assert_loaded(self) -> None:
        if not self.is_loaded or self._et is None:
            raise ModelError(
                "Extra Trees model is not loaded. Call load() first."
            )
