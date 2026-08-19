from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from geoai.datasets.dataset_metadata import DatasetMetadata

class UnifiedChangeDetectionDataset(ABC):
    """Canonical interface enforcing standard interaction across datasets."""
    
    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Return dataset specifications."""
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of image pairs/tiles in the dataset."""
        
    @abstractmethod
    def get_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a normalized sample:
        t1: (C, H, W) float32 numpy array, normalized to [0, 1]
        t2: (C, H, W) float32 numpy array, normalized to [0, 1]
        label: (H, W) int64 numpy array (0 = no-change, 1 = change)
        """
