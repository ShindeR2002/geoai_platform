from typing import Dict, List, Optional, Any
from geoai.datasets.dataset_metadata import DatasetMetadata
from geoai.datasets.dataset_catalog import DATASET_CATALOG
from geoai.datasets.dataset_loader import load_dataset_from_catalog
from geoai.datasets.dataset import Dataset
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset

class DatasetRegistry:
    """Registry engine to manage pluggable datasets across experiment sessions."""
    _catalog: Dict[str, DatasetMetadata] = dict(DATASET_CATALOG)

    @classmethod
    def register_dataset(cls, metadata: DatasetMetadata) -> None:
        """Register a new dataset catalog entry."""
        cls._catalog[metadata.dataset_id] = metadata

    @classmethod
    def get_dataset_metadata(cls, dataset_id: str) -> Optional[DatasetMetadata]:
        """Fetch metadata for a dataset by ID."""
        # Ensure lazy loading registry maps public datasets first
        if dataset_id not in cls._catalog and dataset_id in ("levir_cd", "oscd", "s2looking"):
            cls.load_public_dataset(dataset_id)
        return cls._catalog.get(dataset_id)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List registered dataset identifiers."""
        # Make sure our public datasets are included in the catalog listing
        keys = list(cls._catalog.keys())
        for public_id in ("levir_cd", "oscd", "s2looking"):
            if public_id not in keys:
                keys.append(public_id)
        return keys

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str,
        configs_dir: str = "configs",
        preprocessing_config: Optional[dict] = None,
        features_config: Optional[dict] = None,
        campaign_type: str = "preprocessing",
        track: str = "production",
    ) -> Dataset:
        """Instantiate and load a registered classical dataset."""
        meta = cls.get_dataset_metadata(dataset_id)
        if meta is None:
            raise KeyError(f"Dataset '{dataset_id}' is not registered in the catalog.")
        return load_dataset_from_catalog(
            meta,
            configs_dir=configs_dir,
            preprocessing_config=preprocessing_config,
            features_config=features_config,
            campaign_type=campaign_type,
            track=track,
        )

    @classmethod
    def load_public_dataset(cls, dataset_id: str) -> Optional[UnifiedChangeDetectionDataset]:
        """Load a public change detection dataset under the unified schema loader."""
        ds = None
        if dataset_id == "levir_cd":
            from geoai.datasets.loaders.levir_loader import LEVIRCDDataset
            ds = LEVIRCDDataset()
        elif dataset_id == "oscd":
            from geoai.datasets.loaders.oscd_loader import OSCDDataset
            ds = OSCDDataset()
        elif dataset_id == "s2looking":
            from geoai.datasets.loaders.s2looking_loader import S2LookingDataset
            ds = S2LookingDataset()
            
        if ds is not None:
            cls.register_dataset(ds.get_metadata())
            
        return ds
