import os
import numpy as np
from pathlib import Path
from typing import Tuple
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset
from geoai.datasets.dataset_metadata import DatasetMetadata, DatasetVersioning, DatasetCapabilityMatrix

class LEVIRCDDataset(UnifiedChangeDetectionDataset):
    """Loader for the LEVIR-CD building change detection dataset."""
    
    def __init__(self, root_dir: str = "data/unified/levir_cd", mock_len: int = 50):
        self.root_dir = Path(root_dir)
        self.mock_len = mock_len
        self.is_mock = not (self.root_dir.exists() and len(list(self.root_dir.glob("**/*.tif"))) > 0)
        
        # Metadata setup
        self.metadata = DatasetMetadata(
            dataset_id="levir_cd",
            aoi_name="LEVIR-CD",
            sensor_eo="Google Earth API",
            spatial_resolution_m=0.5,
            crs="EPSG:3857",
            licensing="CC BY 4.0",
            bibtex="""@article{chen2020spatial,
  title={A spatial-temporal attention-based method and a new dataset for remote sensing image change detection},
  author={Chen, Hao and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={58},
  number={10},
  pages={7593--7607},
  year={2020},
  publisher={IEEE}
}""",
            versioning=DatasetVersioning("1.0", "1.0", "1.0", "1.0"),
            capabilities=DatasetCapabilityMatrix(capabilities={
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
        )

    def get_metadata(self) -> DatasetMetadata:
        return self.metadata

    def __len__(self) -> int:
        if self.is_mock:
            return self.mock_len
        return len(list((self.root_dir / "A").glob("*.tif")))

    def get_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns standard T1, T2 normalized float32 images and int64 binary change mask."""
        if self.is_mock:
            # Generate deterministic mock patterns based on index seed
            np.random.seed(idx)
            H, W = 256, 256
            
            # T1: random background with structural shapes
            t1 = np.random.rand(3, H, W).astype(np.float32) * 0.4 + 0.2
            t2 = np.copy(t1)
            
            # Binary label: standard shape at the center
            label = np.zeros((H, W), dtype=np.int64)
            # Add circular changes
            cy, cx = H // 2, W // 2
            r = 30
            y, x = np.ogrid[:H, :W]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            label[mask] = 1
            
            # Apply changes to T2
            t2[:, mask] = np.random.rand(3, np.sum(mask)).astype(np.float32) * 0.5 + 0.4
            
            return t1, t2, label
            
        # Real files load (assuming standard LEVIR layout: root/A, root/B, root/label)
        files_t1 = sorted(list((self.root_dir / "A").glob("*.tif")))
        files_t2 = sorted(list((self.root_dir / "B").glob("*.tif")))
        files_lbl = sorted(list((self.root_dir / "label").glob("*.tif")))
        
        # Load using standard libraries (such as PIL or rasterio)
        from PIL import Image
        t1_img = np.array(Image.open(files_t1[idx])).astype(np.float32) / 255.0
        t2_img = np.array(Image.open(files_t2[idx])).astype(np.float32) / 255.0
        lbl_img = np.array(Image.open(files_lbl[idx])).astype(np.int64)
        
        # Standardize format to Channels-First (C, H, W)
        if t1_img.ndim == 3:
            t1_img = np.transpose(t1_img, (2, 0, 1))
            t2_img = np.transpose(t2_img, (2, 0, 1))
        else:
            t1_img = np.expand_dims(t1_img, axis=0)
            t2_img = np.expand_dims(t2_img, axis=0)
            
        # Threshold label to binary index
        lbl_img = (lbl_img > 127).astype(np.int64)
        
        return t1_img, t2_img, lbl_img
