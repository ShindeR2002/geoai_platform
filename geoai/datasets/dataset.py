import numpy as np
from geoai.datasets.dataset_metadata import DatasetMetadata

class Dataset:
    """Represents a loaded dataset with loaded feature matrices and label vectors."""
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, metadata: DatasetMetadata) -> None:
        self.name = name
        self.X = X
        self.y = y
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, X_shape={self.X.shape}, y_shape={self.y.shape}, version={self.metadata.version})"
