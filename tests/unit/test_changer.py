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
from geoai.models.transformers.changer import Changer, ChangerArch
from geoai.models.transformers.base import TransformerBaseWrapper

class TestChangerWrapper:
    """Unit tests for the Changer wrapper class and registered pipeline operations."""

    def test_registry_lookup(self):
        """Verify that Changer is correctly registered and instantiable from registry."""
        model = get_experiment_model("changer")
        assert isinstance(model, Changer)
        assert isinstance(model, TransformerBaseWrapper)
        assert model.is_implemented() is True
        
        caps = model.get_capabilities()
        assert caps.model_type == "Transformer"
        assert len(caps.expected_input_channels) == 14
        assert "torch" in caps.requirements

    def test_wrapper_initialization(self):
        """Verify Changer wrapper initialization sets up custom architectures and default values."""
        model = Changer()
        assert model.model_id == "changer"
        assert model.architecture_class == ChangerArch
        assert model.model is None

    def test_forward_pass_and_inference(self):
        """Test the forward pass of ChangerArch and mock inference from the wrapper."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_arch = ChangerArch(in_channels=18, out_channels=2).to(device)
        model_arch.eval()

        # Batch size 2, 18 features (7 t1, 7 t2, 4 deltas), 15x15 patch size
        x_mock = torch.randn(2, 18, 15, 15).to(device)
        with torch.no_grad():
            out = model_arch(x_mock)
        assert out.shape == (2, 2, 15, 15)
        assert not torch.isnan(out).any()

        # Test wrapper predict and predict_proba
        wrapper = Changer()
        wrapper.model = ChangerArch(in_channels=18, out_channels=2)
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
        wrapper = Changer()
        wrapper.model = ChangerArch(in_channels=18, out_channels=2)
        wrapper.device = torch.device("cpu")
        wrapper.model.to(wrapper.device)
        wrapper.model.eval()

        x1 = torch.randn(2, 7, 15, 15)
        x2 = torch.randn(2, 7, 15, 15)

        attn_map = wrapper.get_attention_maps(x1, x2)
        assert attn_map.shape == (2, 1, 15, 15)

        token_emb = wrapper.get_token_embeddings(x1, x2)
        # 64 channels from conv5 + 64 channels from conv6 = 128 channels
        assert token_emb.shape == (2, 128, 15, 15)

    def test_checkpoint_serialization(self):
        """Verify model weights and metadata save and load correctly."""
        wrapper = Changer()
        wrapper.model = ChangerArch(in_channels=18, out_channels=2)
        wrapper.device = torch.device("cpu")
        wrapper.model.to(wrapper.device)
        wrapper.dataset_id = "mock_dataset"
        wrapper.patch_size = 11

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_model.pt"
            wrapper.save(chk_path)
            assert chk_path.exists()
            
            # Load checkpoint into new wrapper
            new_wrapper = Changer()
            new_wrapper.device = torch.device("cpu")
            new_wrapper.load(chk_path)
            
            assert new_wrapper.is_loaded is True
            assert new_wrapper.dataset_id == "mock_dataset"
            assert new_wrapper.patch_size == 11
            assert len(new_wrapper.feature_names) == 18

    def test_training_lifecycle(self):
        """Test one epoch of training of Changer wrapper on synthetic features."""
        wrapper = Changer()
        wrapper.device = torch.device("cpu")
        wrapper.hyperparams = {
            "batch_size": 2,
            "learning_rate": 0.01,
            "epochs": 1,
            "random_state": 42
        }
        wrapper.dataset_id = "test_run"
        wrapper.patch_size = 15
        
        X_train = np.zeros((4, 18), dtype=np.float32)
        y_train = np.array([0, 1, 0, 1], dtype=np.int64)
        
        # Setup mock coordinate mapping for training
        wrapper.row_bytes_to_coords = {row.tobytes(): [(0, 0)] for row in X_train}
        wrapper.coords = np.zeros((4, 2), dtype=np.int64)
        wrapper.feature_cube = np.zeros((15, 15, 18), dtype=np.float32)
        wrapper.labels_2d = np.zeros((15, 15), dtype=np.int64)
        
        wrapper.fit(X_train, y_train)
        assert wrapper.is_loaded is True
