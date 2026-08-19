import os
import numpy as np
from pathlib import Path
from typing import Tuple
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset
from geoai.datasets.dataset_metadata import DatasetMetadata, DatasetVersioning, DatasetCapabilityMatrix

class OSCDDataset(UnifiedChangeDetectionDataset):
    """Loader for the OSCD urban change detection dataset."""
    
    def __init__(self, root_dir: str = "data/unified/oscd", mock_len: int = 24):
        self.root_dir = Path(root_dir)
        self.mock_len = mock_len
        self.is_mock = not (self.root_dir.exists() and len(list(self.root_dir.glob("**/*.tif"))) > 0)
        
        # Metadata setup
        self.metadata = DatasetMetadata(
            dataset_id="oscd",
            aoi_name="OSCD Cities",
            sensor_eo="Sentinel-2 Multispectral",
            spatial_resolution_m=10.0,
            crs="EPSG:32630",
            licensing="CC BY-SA 4.0",
            bibtex="""@inproceedings{daudt2018urban,
  title={Urban change detection for multispectral earth observation images},
  author={Daudt, Rodrigo Caye and Le Saux, Bertrand and Boulch, Alexandre and Gousseau, Yann},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={2115--2118},
  year={2018},
  organization={IEEE}
}""",
            versioning=DatasetVersioning("1.0", "1.0", "1.0", "1.0"),
            capabilities=DatasetCapabilityMatrix(capabilities={
                "Classical_ML": True,
                "CNN": True,
                "Transformer": True,
                "Foundation_Model": True,
                "Optical": True,
                "SAR": False,
                "Multispectral": True,
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
            np.random.seed(idx + 1000)
            H, W = 128, 128
            # Multispectral mock (5 bands: R, G, B, NIR, SWIR)
            t1 = np.random.rand(5, H, W).astype(np.float32) * 0.3 + 0.1
            t2 = np.copy(t1)
            
            label = np.zeros((H, W), dtype=np.int64)
            cy, cx = H // 3, W // 3
            r = 15
            y, x = np.ogrid[:H, :W]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            label[mask] = 1
            
            t2[:, mask] = np.random.rand(5, np.sum(mask)).astype(np.float32) * 0.4 + 0.3
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
