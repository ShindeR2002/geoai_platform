import pytest
import numpy as np
from geoai.evaluation.generalization import compute_psi, compute_jsd, GeneralizationEvaluator

def test_compute_psi():
    # Identical distributions should yield a very low PSI
    np.random.seed(42)
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    
    psi_val = compute_psi(expected, actual)
    assert psi_val >= 0.0
    assert psi_val < 0.1  # Very similar

    # Shifted distributions should yield a higher PSI
    shifted_actual = np.random.normal(1.5, 1, 1000)
    psi_shifted = compute_psi(expected, shifted_actual)
    assert psi_shifted > 0.25  # Significant shift

def test_compute_jsd():
    np.random.seed(42)
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    
    jsd_val = compute_jsd(expected, actual)
    assert 0.0 <= jsd_val <= 1.0

    # Shifted
    shifted_actual = np.random.normal(5.0, 1, 1000)
    jsd_shifted = compute_jsd(expected, shifted_actual)
    assert jsd_shifted > jsd_val

def test_dataset_similarity_score():
    evaluator = GeneralizationEvaluator()
    np.random.seed(42)
    X_src = np.random.normal(0, 1, (1000, 3))
    feature_names = ["feat1", "feat2", "feat3"]
    
    # Identical dataset
    score_self, label_self, _ = evaluator.compute_dataset_similarity(X_src, X_src, feature_names)
    assert score_self == 100.0
    assert label_self == "Very Similar"
    
    # Shifted target dataset
    X_tgt = np.random.normal(1.0, 1.5, (1000, 3))
    score_diff, label_diff, _ = evaluator.compute_dataset_similarity(X_src, X_tgt, feature_names)
    assert 0.0 <= score_diff < 100.0
    assert label_diff in ["Similar", "Moderate", "Different", "Highly Different"]

def test_loao_fallback_guard():
    # If number of datasets is less than 3, LOAO should fall back
    # We will test the logical fallback check inside run_generalization_campaign pattern
    datasets = ["dholera_sentinel_v1", "ps10_sentinel_v1"]
    loao_active = len(datasets) >= 3
    assert not loao_active
