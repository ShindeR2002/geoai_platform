import sys
import os
import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

# Ensure geoai package is in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.experiments.experiment_registry import get_experiment_model
from geoai.models.baselines.dl_wrapper import split_dataset_spatial, get_pixel_split, BLOCK_SPLIT_MAP, set_deterministic_seeds
from geoai.models.baselines.deep_learning import FC_EF, FC_Siam_Conc, FC_Siam_Diff, Lightweight_Siam_CNN

class TestDeepLearningBaselines:
    """Tests for deep learning architectures, spatial block splitting, and seed determinism."""

    def test_seed_determinism(self):
        """Verify that synchronizing seeds guarantees deterministic initialization weights."""
        set_deterministic_seeds(42)
        model1 = FC_EF(in_channels=18, out_channels=2)
        w1 = [p.clone() for p in model1.parameters()]
        
        set_deterministic_seeds(42)
        model2 = FC_EF(in_channels=18, out_channels=2)
        w2 = [p.clone() for p in model2.parameters()]
        
        for p1, p2 in zip(w1, w2):
            assert torch.allclose(p1, p2), "Model parameters are not identical after seeding!"

        # Seed with a different value
        set_deterministic_seeds(43)
        model3 = FC_EF(in_channels=18, out_channels=2)
        w3 = [p.clone() for p in model3.parameters()]
        
        # Check that it's different
        any_diff = False
        for p1, p3 in zip(w1, w3):
            if not torch.allclose(p1, p3):
                any_diff = True
                break
        assert any_diff, "Model parameters initialized identically despite different seeds!"

    def test_spatial_split_leakage_prevention(self):
        """Verify that spatial block splitting correctly isolates training and testing pixels with a buffer zone."""
        # Define mock dimensions and a mock coordinate
        # Let H, W = 100, 100. Let patch size = 15, so radius R = 7.
        H, W = 100, 100
        R = 7
        
        # Test boundary pixels. A block boundary is at height bh = 25, width bw = 25.
        # Pixel at (24, 24) is in block 0 (train).
        # Pixel at (24, 25) is in block 1 (train).
        # Pixel at (24, 10) is in block 0 (train).
        # Pixel at (24, 20) is in block 0, but its patch (which extends 7 pixels to the right)
        # overlaps with block 1 (also train). Since they are both train, it should not be 'buffer'.
        assert get_pixel_split(24, 10, H, W, R) == 'train'
        
        # Pixel at (24, 24) is in block 0 (train), but within 7 pixels of block 3 (val, x=75) or block 2 (train).
        # Actually, let's test a coordinate close to the boundary between Train and Val.
        # Block 0 (train) and Block 3 (val). The vertical boundary is at c = 75 (since 100/4 = 25, 25*3 = 75).
        # c = 74 is in block 2 (train). c = 75 is in block 3 (val).
        # Center at (10, 74) is Train, but its patch extends to c = 81 (which is Val).
        # So it must return 'buffer'!
        assert get_pixel_split(10, 74, H, W, R) == 'buffer'
        
        # Center at (10, 75) is Val, but its patch extends to c = 68 (which is Train).
        # So it must return 'buffer'!
        assert get_pixel_split(10, 75, H, W, R) == 'buffer'

    def test_amp_and_cpu_fallback(self):
        """Verify that PyTorch models run properly with AMP enabled and fallback on CPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FC_EF(in_channels=18, out_channels=2).to(device)
        model.eval()
        
        x = torch.randn(2, 18, 15, 15).to(device)
        
        # Run forward pass with AMP autocasting
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            output = model(x)
            
        assert output.shape == (2, 2, 15, 15)
        assert not torch.isnan(output).any()

    def test_model_capabilities(self):
        """Verify model capabilities are properly reported by wrappers."""
        dl_models = ["fc_ef", "fc_siam_conc", "fc_siam_diff", "lightweight_siam_cnn"]
        for model_id in dl_models:
            model = get_experiment_model(model_id)
            assert model.is_implemented() is True
            caps = model.get_capabilities()
            assert "Deep Learning CNN" in caps.model_type
            assert len(caps.expected_input_channels) == 18
            assert "torch" in caps.requirements

    def test_dl_score_and_explainability(self):
        """Verify the wrapper's score method and Grad-CAM/activation map generation."""
        from geoai.evaluation.feature_analysis import generate_gradcam, generate_activation_maps
        
        model_wrapper = get_experiment_model("fc_ef")
        # Initialize architecture
        model_wrapper.model = FC_EF(in_channels=18, out_channels=2)
        model_wrapper.device = torch.device("cpu")
        model_wrapper.model.to(model_wrapper.device)
        model_wrapper.model.eval()
        
        # Test score method
        X_mock = np.zeros((10, 18), dtype=np.float32)
        y_mock = np.zeros(10, dtype=np.int64)
        
        # Mock coord mapping to prevent actual dataset load in unit test
        model_wrapper.row_bytes_to_coords = {row.tobytes(): [(0, 0)] for row in X_mock}
        model_wrapper.coords = np.zeros((10, 2), dtype=np.int64)
        model_wrapper.feature_cube = np.zeros((15, 15, 18), dtype=np.float32)
        model_wrapper.labels_2d = np.zeros((15, 15), dtype=np.int64)
        
        score_val = model_wrapper.score(X_mock, y_mock)
        assert isinstance(score_val, float)
        assert 0.0 <= score_val <= 1.0
        
        # Test Grad-CAM & Activation maps
        input_tensor = torch.randn(1, 18, 15, 15)
        gradcam_map = generate_gradcam(model_wrapper.model, input_tensor, target_class=1)
        assert gradcam_map.shape == (15, 15)
        assert gradcam_map.min() >= 0.0
        assert gradcam_map.max() <= 1.0
        
        activation_map = generate_activation_maps(model_wrapper.model, input_tensor)
        assert activation_map.shape == (15, 15)
        assert activation_map.min() >= 0.0
        assert activation_map.max() <= 1.0

