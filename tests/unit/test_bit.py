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
from geoai.models.transformers.bit import BIT, BITArch
from geoai.models.transformers.base import TransformerBaseWrapper

class TestBITWrapper:
    """Unit tests for the BIT wrapper class and registered pipeline operations."""

    def test_registry_lookup(self):
        """Verify that BIT is correctly registered and instantiable from registry."""
        model = get_experiment_model("bit")
        assert isinstance(model, BIT)
        assert isinstance(model, TransformerBaseWrapper)
        assert model.is_implemented() is True
        
        caps = model.get_capabilities()
        assert caps.model_type == "Transformer"
        assert len(caps.expected_input_channels) == 14
        assert "torch" in caps.requirements

    def test_wrapper_initialization(self):
        """Verify BIT wrapper initialization sets up custom architectures and default values."""
        model = BIT()
        assert model.model_id == "bit"
        assert model.architecture_class == BITArch
        assert model.model is None

    def test_forward_pass_and_inference(self):
        """Test the forward pass of BITArch and mock inference from the wrapper."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_arch = BITArch(in_channels=18, out_channels=2).to(device)
        model_arch.eval()

        # Batch size 2, 18 features (7 t1, 7 t2, 4 deltas), 15x15 patch size
        x_mock = torch.randn(2, 18, 15, 15).to(device)
        with torch.no_grad():
            out = model_arch(x_mock)
        assert out.shape == (2, 2, 15, 15)
        assert not torch.isnan(out).any()

        # Test wrapper predict and predict_proba
        wrapper = BIT()
        wrapper.model = BITArch(in_channels=18, out_channels=2)
        wrapper.device = torch.device("cpu")
        wrapper.model.to(wrapper.device)
        wrapper.model.eval()

        X_mock = np.zeros((4, 18), dtype=np.float32)
        
        # Setup mock coordinate mapping
        wrapper.row_bytes_to_coords = {row.tobytes(): [(0, 0)] for row in X_mock}
        wrapper.coords = np.zeros((4, 2), dtype=np.int64)
        wrapper.feature_cube = np.zeros((15, 15, 18), dtype=np.float32)
        wrapper.labels_2d = np.zeros((15, 15), dtype=np.int64)

        preds = wrapper.predict(X_mock)
        assert preds.shape == (4,)
        assert np.all(preds >= 0) and np.all(preds <= 1)

        probs = wrapper.predict_proba(X_mock)
        assert probs.shape == (4, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_explainability_hooks(self):
        """Verify get_attention_maps and get_token_embeddings return correct shapes."""
        wrapper = BIT()
        wrapper.model = BITArch(in_channels=18, out_channels=2)
        wrapper.device = torch.device("cpu")
        wrapper.model.to(wrapper.device)
        wrapper.model.eval()

        x1 = torch.randn(2, 7, 15, 15)
        x2 = torch.randn(2, 7, 15, 15)

        attn_map = wrapper.get_attention_maps(x1, x2)
        assert attn_map.shape == (2, 1, 15, 15)

        token_emb = wrapper.get_token_embeddings(x1, x2)
        # 32 channels from conv1 + 32 channels from conv2 = 64 channels
        assert token_emb.shape == (2, 64, 15, 15)

    def test_checkpoint_serialization(self):
        """Verify model weights and metadata save and load correctly."""
        wrapper = BIT()
        wrapper.model = BITArch(in_channels=18, out_channels=2)
        wrapper.device = torch.device("cpu")
        wrapper.model.to(wrapper.device)
        wrapper.dataset_id = "mock_dataset"
        wrapper.patch_size = 11

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "test_model.pt"
            wrapper.save(chk_path)
            assert chk_path.exists()

            # Load into a new wrapper
            new_wrapper = BIT()
            new_wrapper.device = torch.device("cpu")
            new_wrapper.load(chk_path)

            assert new_wrapper.is_loaded is True
            assert new_wrapper.dataset_id == "mock_dataset"
            assert new_wrapper.patch_size == 11
            assert len(new_wrapper.feature_names) == 18

    def test_training_lifecycle(self):
        """Verify the wrapper's fit method convergence and training loop on small mock split."""
        wrapper = BIT()
        wrapper.device = torch.device("cpu")
        wrapper.hyperparams = {
            "random_state": 42,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 1
        }
        wrapper.dataset_id = "test_run"
        wrapper.patch_size = 15

        X_train = np.zeros((4, 18), dtype=np.float32)
        y_train = np.array([0, 1, 0, 1], dtype=np.int64)

        # Pre-populate mapping structures to bypass disk IO loaders
        wrapper.row_bytes_to_coords = {row.tobytes(): [(0, 0)] for row in X_train}
        wrapper.coords = np.zeros((4, 2), dtype=np.int64)
        wrapper.feature_cube = np.zeros((15, 15, 18), dtype=np.float32)
        wrapper.labels_2d = np.zeros((15, 15), dtype=np.int64)

        # Run fit (it will automatically generate temp dirs and train for 1 epoch)
        wrapper.fit(X_train, y_train)
        assert wrapper.is_loaded is True
