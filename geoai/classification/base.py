"""
BaseClassifier interface for the GeoAI Platform Stage 2 classification system.

Defines the abstract contract that all Stage 2 object classifiers must
implement. This allows the pipeline to swap between the rule-based classifier
(currently the production Stage 2 classifier) and future ML-based classifiers
without changing any pipeline code.

Single responsibility: define the Stage 2 classifier interface.

Position in dependency hierarchy: classification (depends on classification/schema).
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from geoai.classification.schema import ChangeObject

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """Abstract base class for Stage 2 semantic change object classifiers.

    Subclasses assign semantic classes to ChangeObject instances using
    spectral, geometric, or learned features.

    Attributes:
        classifier_name: Human-readable name for this classifier implementation.
    """

    def __init__(self, classifier_name: str = "base") -> None:
        """Initialise with a classifier name.

        Args:
            classifier_name: Display name for logging and provenance tracking.
        """
        self.classifier_name = classifier_name

    @abstractmethod
    def classify(self, objects: List[ChangeObject]) -> List[ChangeObject]:
        """Assign semantic classes to a list of ChangeObject instances.

        Modifies each ChangeObject's ``semantic_class``, ``confidence``, and
        ``classifier_method`` fields in place and returns the modified list.

        Args:
            objects: List of ChangeObject instances to classify. Spectral
                attributes must be populated before calling this method.

        Returns:
            The same list with semantic_class and confidence populated.
        """

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return the list of semantic class names this classifier can assign.

        Returns:
            List of class name strings corresponding to PS10 Stage 2 classes.
        """

    def __repr__(self) -> str:
        """Return a string representation of the classifier.

        Returns:
            String with classifier class name and display name.
        """
        return f"{self.__class__.__name__}(name='{self.classifier_name}')"
