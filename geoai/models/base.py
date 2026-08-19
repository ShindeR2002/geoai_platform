"""
BaseModel interface for the GeoAI Platform.

Defines the abstract contract that every model in the platform must implement.
This ensures that the inference pipeline, evaluation module, and API layer
can interact with any model — current or future — through a consistent
interface without knowing the underlying implementation.

All production models (RF Enhanced, RF Baseline) and all future models
(XGBoost, CNN, ViT) must inherit from BaseModel and implement all abstract
methods.

Single responsibility: define the model interface contract.

Position in dependency hierarchy: models (depends on core, utils).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ModelCapabilities:
    """Structured metadata class describing the model capabilities."""
    def __init__(
        self,
        model_type: str,
        expected_input_channels: List[str],
        training_status: str,
        requirements: List[str],
        model_name: Optional[str] = None,
        reference_paper: Optional[str] = None,
        publication_year: Optional[int] = None,
        architecture_family: Optional[str] = None,
        official_parameter_count: Optional[int] = None,
        implemented_parameter_count: Optional[int] = None,
        implementation_classification: Optional[str] = None,
        backbone: Optional[str] = None,
        explainability_support: bool = False,
        dense_prediction_support: bool = False,
        checkpoint_support: bool = False
    ) -> None:
        self.model_type = model_type
        self.expected_input_channels = expected_input_channels
        self.training_status = training_status
        self.requirements = requirements
        self.model_name = model_name or "Unknown"
        self.reference_paper = reference_paper or "N/A"
        self.publication_year = publication_year
        self.architecture_family = architecture_family or "Unknown"
        self.official_parameter_count = official_parameter_count
        self.implemented_parameter_count = implemented_parameter_count
        self.implementation_classification = implementation_classification or "Unknown"
        self.backbone = backbone or "None"
        self.explainability_support = explainability_support
        self.dense_prediction_support = dense_prediction_support
        self.checkpoint_support = checkpoint_support

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "expected_input_channels": self.expected_input_channels,
            "training_status": self.training_status,
            "requirements": self.requirements,
            "model_name": self.model_name,
            "reference_paper": self.reference_paper,
            "publication_year": self.publication_year,
            "architecture_family": self.architecture_family,
            "official_parameter_count": self.official_parameter_count,
            "implemented_parameter_count": self.implemented_parameter_count,
            "implementation_classification": self.implementation_classification,
            "backbone": self.backbone,
            "explainability_support": self.explainability_support,
            "dense_prediction_support": self.dense_prediction_support,
            "checkpoint_support": self.checkpoint_support
        }


class BaseModel(ABC):
    """Abstract base class for all GeoAI Platform prediction models.

    Subclasses implement the abstract methods to provide a concrete model
    that can be loaded, used for prediction, and queried for its feature
    configuration.

    Attributes:
        model_id: Unique identifier for this model instance. Set during
            loading from the model metadata JSON.
        feature_names: Ordered list of feature names the model expects.
            Used for feature count validation before inference.
        is_loaded: True after :meth:`load` has been called successfully.
    """

    def __init__(self) -> None:
        """Initialise the base model with unset state."""
        self.model_id: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.is_loaded: bool = False

    def is_implemented(self) -> bool:
        """Return True if the model is implemented, False otherwise."""
        return True

    def get_capabilities(self) -> ModelCapabilities:
        """Return the structured capabilities of this model."""
        return ModelCapabilities(
            model_type="Unknown",
            expected_input_channels=[],
            training_status="Unknown",
            requirements=[]
        )

    @abstractmethod
    def load(self, model_path: Union[str, Path]) -> None:
        """Load a serialised model from disk.

        After this method returns successfully, ``is_loaded`` must be True
        and ``feature_names`` must be populated.

        Args:
            model_path: Path to the serialised model file.

        Raises:
            ModelNotFoundError: If the file does not exist.
            ModelError: If the file cannot be deserialised.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for a feature matrix.

        Args:
            X: Float32 feature matrix of shape (N_pixels, N_features).
               N_features must match the model's expected feature count.

        Returns:
            1D integer array of shape (N_pixels,) containing predicted
            class labels (0 = no change, 1 = change).

        Raises:
            ModelError: If the model is not loaded or inference fails.
            FeatureCountMismatchError: If X has the wrong number of features.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a feature matrix.

        Args:
            X: Float32 feature matrix of shape (N_pixels, N_features).

        Returns:
            2D float32 array of shape (N_pixels, N_classes) containing
            class probability estimates. For binary classification,
            column 0 is P(no change) and column 1 is P(change).

        Raises:
            ModelError: If the model is not loaded or inference fails.
            FeatureCountMismatchError: If X has the wrong number of features.
        """

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names this model expects.

        Returns:
            List of feature name strings in the order the model expects
            them as input columns.
        """

    def get_expected_feature_count(self) -> int:
        """Return the number of features this model expects.

        Returns:
            Integer count of expected input features.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self.feature_names is None:
            raise RuntimeError(
                "Model feature names are not set. Call load() before "
                "querying feature count."
            )
        return len(self.feature_names)

    def validate_features(self, X: np.ndarray) -> None:
        """Validate that a feature matrix has the correct number of columns.

        Args:
            X: Feature matrix to validate.

        Raises:
            FeatureCountMismatchError: If X.shape[1] != expected feature count.
            RuntimeError: If the model has not been loaded.
        """
        from geoai.core.exceptions import FeatureCountMismatchError
        expected = self.get_expected_feature_count()
        if X.ndim != 2 or X.shape[1] != expected:
            actual = X.shape[1] if X.ndim == 2 else X.shape
            raise FeatureCountMismatchError(expected=expected, actual=actual)

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns:
            String describing the model type, ID, and load state.
        """
        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id!r}, "
            f"is_loaded={self.is_loaded}, "
            f"n_features={self.get_expected_feature_count() if self.is_loaded else 'N/A'})"
        )
