import os
import numpy as np
from pathlib import Path
from typing import Tuple
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset
from geoai.datasets.dataset_metadata import DatasetMetadata, DatasetVersioning, DatasetCapabilityMatrix

class S2LookingDataset(UnifiedChangeDetectionDataset):
    """Loader for the S2Looking side-looking building change detection dataset."""
    
    def __init__(self, root_dir: str = "data/unified/s2looking", mock_len: int = 30):
        self.root_dir = Path(root_dir)
        self.mock_len = mock_len
        self.is_mock = not (self.root_dir.exists() and len(list(self.root_dir.glob("**/*.tif"))) > 0)
        
        # Metadata setup
        self.metadata = DatasetMetadata(
            dataset_id="s2looking",
            aoi_name="S2Looking Areas",
            sensor_eo="GaoFen & Sentinel-2",
            spatial_resolution_m=0.8,
            crs="EPSG:3857",
            licensing="CC BY-NC-SA 4.0",
            bibtex="""@article{shen2021s2looking,
  title={S2Looking: A satellite image dataset for building change detection},
  author={Shen, Li and Lu, Chenshuaji and Chen, Hao and Wei, Zhengxin and Li, Dongyong and Shi, Zhenwei},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={14},
  pages={11574--11585},
  year={2021},
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
            np.random.seed(idx + 2000)
            H, W = 256, 256
            # Optical RGB
            t1 = np.random.rand(3, H, W).astype(np.float32) * 0.4 + 0.3
            t2 = np.copy(t1)
            
            label = np.zeros((H, W), dtype=np.int64)
            # Create a box change
            cy, cx = H // 2, W // 2
            label[cy-20:cy+20, cx-20:cx+20] = 1
            
            t2[:, label == 1] = np.random.rand(3, np.sum(label == 1)).astype(np.float32) * 0.5 + 0.2
            return t1, t2, label
            
        files_t1 = sorted(list((self.root_dir / "A").glob("*.tif")))
        files_t2 = sorted(list((self.root_dir / "B").glob("*.tif")))
        files_lbl = sorted(list((self.root_dir / "label").glob("*.tif")))
        
        from PIL import Image
        t1_img = np.array(Image.open(files_t1[idx])).astype(np.float32) / 255.0
        t2_img = np.array(Image.open(files_t2[idx])).astype(np.float32) / 255.0
        lbl_img = np.array(Image.open(files_lbl[idx])).astype(np.int64)
        
        if t1_img.ndim == 3:
            t1_img = np.transpose(t1_img, (2, 0, 1))
            t2_img = np.transpose(t2_img, (2, 0, 1))
        else:
            t1_img = np.expand_dims(t1_img, axis=0)
            t2_img = np.expand_dims(t2_img, axis=0)
            
        lbl_img = (lbl_img > 0).astype(np.int64)
        return t1_img, t2_img, lbl_img
