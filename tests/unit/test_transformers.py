import sys
import os
import torch
import pytest

# Ensure geoai is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.models.registry import get_model_class, load_model
from geoai.models.transformers.base import TransformerBaseWrapper
from geoai.models.transformers.changeformer import ChangeFormer

class TestTransformerCampaign:
    """Test suite validating Transformer Campaign model wrappers and registry interfaces."""

    def test_transformer_wrapper_inheritance(self):
        """Verify that ChangeFormer wrapper inherits from TransformerBaseWrapper and defines capabilities."""
        model = ChangeFormer()
        assert isinstance(model, TransformerBaseWrapper)
        
        caps = model.get_capabilities()
        assert caps.model_type == "Transformer"
        assert "torch" in caps.requirements

    def test_model_registry_resolution(self):
        """Verify that registry registers ChangeFormer and BIT under standard lookups."""
        for model_id in ["changeformer", "bit", "tinycd", "stanet", "snunet"]:
            cls = get_model_class(model_id)
            assert issubclass(cls, TransformerBaseWrapper)

    def test_attention_maps_extraction(self):
        """Verify that get_attention_maps hooks extract spatial attention grids of shape (B, 1, H, W)."""
        model = ChangeFormer()
        
        # Instantiate model architecture
        model.model = model.architecture_class(in_channels=18, out_channels=2)
        model.model.to(model.device)
        
        # Mock input batch
        B, C, H, W = 2, 7, 16, 16
        x1 = torch.randn(B, C, H, W).to(model.device)
        x2 = torch.randn(B, C, H, W).to(model.device)
        
        # Extract maps
        attn_map = model.get_attention_maps(x1, x2)
        
        assert isinstance(attn_map, torch.Tensor)
        assert attn_map.shape == (B, 1, H, W)

    def test_token_embeddings_extraction(self):
        """Verify get_token_embeddings returns intermediate features maps."""
        model = ChangeFormer()
        model.model = model.architecture_class(in_channels=18, out_channels=2)
        model.model.to(model.device)
        
        B, C, H, W = 2, 7, 16, 16
        x1 = torch.randn(B, C, H, W).to(model.device)
        x2 = torch.randn(B, C, H, W).to(model.device)
        
        tokens = model.get_token_embeddings(x1, x2)
        assert isinstance(tokens, torch.Tensor)
        # Should concatenate features (32 from conv1, 32 from conv2 -> 64 channels, or 96 channels for hierarchical)
        assert tokens.shape in ((B, 64, H, W), (B, 96, H, W))

    def test_bit_wrapper_initialization(self):
        """Verify that BIT wrapper initializes, inherits from TransformerBaseWrapper, and has capabilities."""
        from geoai.models.transformers.bit import BIT
        model = BIT()
        assert isinstance(model, TransformerBaseWrapper)
        
        caps = model.get_capabilities()
        assert caps.model_type == "Transformer"
        assert "torch" in caps.requirements

    def test_bit_forward_pass(self):
        """Verify that BIT architecture forward pass executes correctly."""
        from geoai.models.transformers.bit import BITArch
        model = BITArch(in_channels=18, out_channels=2)
        
        # BITArch expects x of shape (B, 14, H, W) where 0:7 are t1 and 7:14 are t2
        B, C, H, W = 2, 14, 16, 16
        x = torch.randn(B, C, H, W)
        
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, 2, H, W)

    def test_bit_attention_maps_extraction(self):
        """Verify that BIT get_attention_maps extracts spatial attention grids of shape (B, 1, H, W)."""
        from geoai.models.transformers.bit import BIT
        model = BIT()
        model.model = model.architecture_class(in_channels=18, out_channels=2)
        model.model.to(model.device)
        
        B, C, H, W = 2, 7, 16, 16
        x1 = torch.randn(B, C, H, W).to(model.device)
        x2 = torch.randn(B, C, H, W).to(model.device)
        
        attn_map = model.get_attention_maps(x1, x2)
        assert isinstance(attn_map, torch.Tensor)
        assert attn_map.shape == (B, 1, H, W)

    def test_bit_token_embeddings_extraction(self):
        """Verify BIT get_token_embeddings returns intermediate features maps of shape (B, 64, H, W)."""
        from geoai.models.transformers.bit import BIT
        model = BIT()
        model.model = model.architecture_class(in_channels=18, out_channels=2)
        model.model.to(model.device)
        
        B, C, H, W = 2, 7, 16, 16
        x1 = torch.randn(B, C, H, W).to(model.device)
        x2 = torch.randn(B, C, H, W).to(model.device)
        
        tokens = model.get_token_embeddings(x1, x2)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (B, 64, H, W)

    def test_bit_checkpoint_serialization(self, tmp_path):
        """Verify that BIT saves and loads checkpoint weights and metadata correctly."""
        from geoai.models.transformers.bit import BIT
        model = BIT()
        model.model = model.architecture_class(in_channels=18, out_channels=2)
        model.model.to(model.device)
        
        checkpoint_file = tmp_path / "bit_test.pt"
        
        # Save checkpoint
        model.save(checkpoint_file)
        assert checkpoint_file.exists()
        
        # Load checkpoint into new wrapper instance
        loaded_model = BIT()
        loaded_model.load(checkpoint_file)
        
        assert loaded_model.is_loaded
        assert loaded_model.patch_size == model.patch_size
        assert loaded_model.model is not None
        
        # Verify weight equivalence
        for p1, p2 in zip(model.model.parameters(), loaded_model.model.parameters()):
            assert torch.allclose(p1, p2)

