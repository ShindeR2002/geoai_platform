import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from geoai.preprocessing.pipeline import (
    refined_lee_filter,
    local_variance_filter,
    reconstruct_nir,
    compute_savi,
    compute_msavi,
    get_campaign_feature_names,
)

def test_refined_lee_filter_shape_and_smoothing():
    """Verify that refined_lee_filter preserves image dimensions and reduces variance."""
    rng = np.random.default_rng(seed=42)
    # 20x20 test image with some random noise and a flat region
    img = rng.uniform(0.1, 0.9, (20, 20)).astype(np.float32)
    
    filtered = refined_lee_filter(img, size=7, noise_fraction=0.26)
    
    assert filtered.shape == img.shape
    assert filtered.dtype == np.float32
    # Speckle filtering should reduce variance compared to raw noisy image
    assert np.var(filtered) < np.var(img)


def test_local_variance_filter_shape():
    """Verify local_variance_filter returns standard deviation map with correct dimensions."""
    img = np.ones((10, 10), dtype=np.float32)
    # constant image has 0 local variance
    lv = local_variance_filter(img, size=3)
    assert lv.shape == (10, 10)
    assert np.allclose(lv, 0.0, atol=1e-5)

    # image with noise should have positive local variance
    img[2, 2] = 5.0
    lv_noisy = local_variance_filter(img, size=3)
    assert lv_noisy[2, 2] > 0.0


def test_reconstruct_nir_math():
    """Verify algebraic NIR reconstruction: NIR = Red * (1 + NDVI) / (1 - NDVI)."""
    # If Red = 0.2 and NDVI = 0.5, NIR = 0.2 * 1.5 / 0.5 = 0.6
    red = np.array([[0.2]], dtype=np.float32)
    ndvi = np.array([[0.5]], dtype=np.float32)
    
    nir = reconstruct_nir(red, ndvi)
    assert np.allclose(nir, 0.6, atol=1e-5)
    
    # Division guard check: if NDVI = 1.0 (denom = 0), it shouldn't crash or return infinity
    ndvi_one = np.array([[1.0]], dtype=np.float32)
    nir_guard = reconstruct_nir(red, ndvi_one)
    assert not np.isinf(nir_guard).any()
    assert not np.isnan(nir_guard).any()
    assert (nir_guard >= 0.0).all()


def test_savi_msavi_computations():
    """Verify correctness of SAVI and MSAVI calculations."""
    red = np.array([[0.2]], dtype=np.float32)
    nir = np.array([[0.6]], dtype=np.float32)
    
    # SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L), with L = 0.5
    # SAVI = (0.6 - 0.2) / (0.6 + 0.2 + 0.5) * 1.5 = 0.4 / 1.3 * 1.5 = 0.461538
    savi = compute_savi(nir, red, L=0.5)
    assert np.allclose(savi, 0.461538, atol=1e-5)
    
    # MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2
    # 2*0.6 + 1 = 2.2
    # 2.2^2 = 4.84
    # 8*(0.6 - 0.2) = 3.2
    # 4.84 - 3.2 = 1.64
    # MSAVI = (2.2 - sqrt(1.64)) / 2 = (2.2 - 1.280624) / 2 = 0.459687
    msavi = compute_msavi(nir, red)
    assert np.allclose(msavi, 0.459687, atol=1e-5)


def test_get_campaign_feature_names():
    """Verify dynamic mapping of features based on campaign and track settings."""
    # Preprocessing campaign should return exactly 18 canonical features
    names_preproc = get_campaign_feature_names("preprocessing", "production", {})
    assert len(names_preproc) == 18
    
    # Feature engineering production track with local variance enabled
    config_prod = {"features": {"local_variance": {"enabled": True}}}
    names_prod = get_campaign_feature_names("feature_engineering", "production", config_prod)
    assert len(names_prod) == 20
    assert "Local_Var_T1" in names_prod
    assert "SAVI_T1" not in names_prod
    
    # Feature engineering experimental track with SAVI and MSAVI enabled
    config_exp = {"features": {"local_variance": {"enabled": True}, "savi": {"enabled": True}, "msavi": {"enabled": True}}}
    names_exp = get_campaign_feature_names("feature_engineering", "experimental", config_exp)
    assert len(names_exp) == 24
    assert "Local_Var_T1" in names_exp
    assert "SAVI_T1" in names_exp
    assert "MSAVI_T1" in names_exp
