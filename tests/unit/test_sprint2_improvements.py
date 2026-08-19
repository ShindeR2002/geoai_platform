import os
import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from geoai.models.transformers.tinycd import TinyCDArch
from geoai.models.transformers.bit import BITArch
from geoai.models.transformers.changer import ChangerArch
from geoai.models.baselines.dl_wrapper import SpatialPatchDataset, ResNet18SiameseAdapter

class TestSprint2Improvements(unittest.TestCase):
    def test_resnet18_siamese_adapter(self):
        # Verify shape preservation and adapter execution
        adapter = ResNet18SiameseAdapter(in_channels=7)
        x = torch.randn(2, 7, 15, 15)
        out = adapter(x)
        self.assertEqual(out.shape, (2, 64, 15, 15))
        
    def test_tinycd_resnet18_backbone(self):
        arch = TinyCDArch(backbone="resnet18")
        x = torch.randn(2, 14, 15, 15)
        out = arch(x)
        self.assertEqual(out.shape, (2, 2, 15, 15))
        
    def test_bit_resnet18_backbone(self):
        arch = BITArch(backbone="resnet18")
        x = torch.randn(2, 14, 15, 15)
        out = arch(x)
        self.assertEqual(out.shape, (2, 2, 15, 15))
        
    def test_changer_resnet18_backbone(self):
        arch = ChangerArch(backbone="resnet18")
        x = torch.randn(2, 14, 15, 15)
        out = arch(x)
        self.assertEqual(out.shape, (2, 2, 15, 15))
        
    def test_spatial_patch_dataset_dense_segmentation(self):
        feat = np.random.randn(20, 20, 18).astype(np.float32)
        labels = np.random.randint(0, 2, (20, 20)).astype(np.int64)
        coords = np.array([[10, 10], [5, 5]])
        
        # Test center_pixel mode
        ds_center = SpatialPatchDataset(
            feature_cube=feat,
            labels=labels,
            coords=coords,
            patch_size=15,
            training_mode="center_pixel"
        )
        x_c, y_c = ds_center[0]
        self.assertEqual(x_c.shape, (18, 15, 15))
        self.assertIsInstance(y_c, int)
        
        # Test dense_segmentation mode
        ds_dense = SpatialPatchDataset(
            feature_cube=feat,
            labels=labels,
            coords=coords,
            patch_size=15,
            training_mode="dense_segmentation"
        )
        x_d, y_d = ds_dense[0]
        self.assertEqual(x_d.shape, (18, 15, 15))
        self.assertEqual(y_d.shape, (15, 15))
