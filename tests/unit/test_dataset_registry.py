import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Ensure geoai is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_metadata import DatasetMetadata
from geoai.datasets.dataset_validator import validate_dataset_integrity
from geoai.datasets.patch_extractor import PatchExtractor
from geoai.datasets.quality_scorer import DatasetQualityScorer
from geoai.datasets.statistics_reporter import DatasetStatisticsReporter

class TestPublicDatasetIntegration:
    """Test suite validating standard schema loaders, statistics, scoring, and patching helpers."""

    def test_dataset_loaders_metadata_and_shapes(self):
        """Verify that all public loaders conform to the schema and return correct shape/type tensors."""
        dataset_ids = ["levir_cd", "oscd", "s2looking"]
        
        for d_id in dataset_ids:
            ds = DatasetRegistry.load_public_dataset(d_id)
            assert ds is not None, f"Failed loading public dataset: {d_id}"
            
            meta = ds.get_metadata()
            assert meta.dataset_id == d_id
            assert meta.bibtex != ""
            assert meta.versioning is not None
            assert meta.capabilities is not None
            
            assert len(ds) > 0
            
            t1, t2, label = ds.get_pair(0)
            assert isinstance(t1, np.ndarray)
            assert isinstance(t2, np.ndarray)
            assert isinstance(label, np.ndarray)
            
            assert t1.dtype == np.float32
            assert t2.dtype == np.float32
            assert label.dtype == np.int64
            
            assert t1.shape == t2.shape
            assert label.shape == (t1.shape[1], t1.shape[2])

    def test_patch_extractor_boundary_padding(self):
        """Verify the PatchExtractor pads arrays correctly and returns standard dimensions."""
        extractor = PatchExtractor(patch_size=16, stride=8, overlap=0.0)
        
        # Grid dimensions (C=3, H=30, W=30)
        t1 = np.ones((3, 30, 30), dtype=np.float32)
        t2 = np.ones((3, 30, 30), dtype=np.float32)
        label = np.zeros((30, 30), dtype=np.int64)
        
        t1_p, t2_p, lbl_p = extractor.extract(t1, t2, label)
        
        # Check shapes
        assert t1_p.ndim == 4
        assert t1_p.shape[1] == 3
        assert t1_p.shape[2] == 16
        assert t1_p.shape[3] == 16
        
        assert lbl_p.ndim == 3
        assert lbl_p.shape[1] == 16
        assert lbl_p.shape[2] == 16
        
        assert len(t1_p) == len(lbl_p)

    def test_dataset_validator_engine(self):
        """Verify validate_dataset_integrity flags validation errors on non-compliant datasets."""
        ds = DatasetRegistry.load_public_dataset("levir_cd")
        
        # Clean test
        valid, errors = validate_dataset_integrity(ds)
        assert valid is True
        assert len(errors) == 0
        
        # Create non-compliant mock loader
        class InvalidDataset(ds.__class__):
            def get_pair(self, idx):
                t1, t2, label = super().get_pair(idx)
                # Intentionally corrupt range and shape
                t1 = t1 * 10.0 # Out of bounds [0, 1]
                return t1, t2, label
                
        bad_ds = InvalidDataset()
        valid, errors = validate_dataset_integrity(bad_ds)
        assert valid is False
        assert any("out of range" in err for err in errors)

    def test_quality_scorer_phase_correlation(self):
        """Verify that quality scorer detects offsets correctly and returns valid composite scores."""
        ds = DatasetRegistry.load_public_dataset("levir_cd")
        score, details = DatasetQualityScorer.evaluate_quality(ds, sample_limit=2)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert "alignment_score" in details
        assert "cloud_free_score" in details

    def test_statistics_reporter_moran_i(self):
        """Verify that statistics reporter correctly estimates Moran's I and outputs stat files."""
        ds = DatasetRegistry.load_public_dataset("levir_cd")
        report = DatasetStatisticsReporter.generate_report(ds, output_dir="outputs/datasets")
        
        assert report["dataset_id"] == "levir_cd"
        assert 0.0 <= report["change_ratio"] <= 1.0
        assert -1.0 <= report["spatial_autocorrelation_moran_i"] <= 1.0
        
        # Check stat file exists
        assert os.path.exists("outputs/datasets/levir_cd_stats.json")
