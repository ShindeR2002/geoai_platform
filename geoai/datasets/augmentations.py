from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class AbstractAugmentationPipeline(ABC):
    """Abstract interface defining standard data augmentation hooks for change detection."""
    
    @abstractmethod
    def apply_spatial(self, t1: np.ndarray, t2: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply spatial transforms (e.g. flips, rotations) simultaneously to t1, t2, and label."""
        pass
        
    @abstractmethod
    def apply_spectral(self, t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spectral transforms (e.g. noise, contrast variations) to inputs."""
        pass
