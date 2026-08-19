from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class DatasetVersioning:
    """Tracks raw, preprocessing, split, and annotation versions of a dataset."""
    raw_version: str = "1.0"
    preprocessing_version: str = "1.0"
    split_version: str = "1.0"
    annotation_version: str = "1.0"

    def to_dict(self) -> dict:
        return self.__dict__

@dataclass
class DatasetCapabilityMatrix:
    """Indicating support for different model families, sensor modalities, and change detection tasks."""
    capabilities: Dict[str, bool] = field(default_factory=lambda: {
        "Classical_ML": True,
        "CNN": True,
        "Transformer": True,
        "Foundation_Model": True,
        "Optical": True,
        "SAR": False,
        "Multispectral": False,
        "Binary": True,
        "Multiclass": False
    })

    def to_dict(self) -> dict:
        return self.capabilities

@dataclass
class DatasetMetadata:
    """Strongly typed container for all metadata attributes of a registered dataset."""
    dataset_id: str
    aoi_name: str
    sensor_eo: str
    sensor_sar: Optional[str] = None
    t1_year: int = 2021
    t2_year: int = 2024
    spatial_resolution_m: float = 10.0
    crs: str = "EPSG:4326"
    preprocessing_status: str = "Raw"
    feature_availability: List[str] = field(default_factory=list)
    train_test_split_metadata: dict = field(default_factory=dict)
    licensing: str = "Proprietary"
    checksum_md5: str = ""
    version: str = "1.0.0"
    
    # Milestone 9 Cross-AOI Generalization Metadata
    country: Optional[str] = None
    region: Optional[str] = None
    climate_zone: Optional[str] = None
    land_cover_composition: Optional[dict] = None
    sensor_acquisition_dates: Optional[dict] = None
    cloud_cover: Optional[float] = None
    season: Optional[str] = None
    ground_truth_source: Optional[str] = None
    reference_lat: Optional[float] = None
    reference_lon: Optional[float] = None
    verification_status: dict = field(default_factory=dict)

    # Milestone 11 Refinements
    is_multiclass: bool = False
    bibtex: str = ""
    versioning: DatasetVersioning = field(default_factory=DatasetVersioning)
    capabilities: DatasetCapabilityMatrix = field(default_factory=DatasetCapabilityMatrix)
