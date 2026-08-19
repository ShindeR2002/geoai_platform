import numpy as np
import pytest
from unittest.mock import MagicMock
from geoai.evaluation.evaluation_manager import _compute_calibration_metrics
from geoai.evaluation.error_analysis import compute_rf_prediction_uncertainty, analyze_prediction_errors
from geoai.utils.constants import ChangeClass

def test_compute_calibration_metrics():
    # Perfect calibration mock
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.8, 0.2, 0.3, 0.1, 0.9])
    
    metrics = _compute_calibration_metrics(y_true, y_prob, n_bins=5)
    
    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier" in metrics
    assert metrics["brier"] >= 0.0
    assert len(metrics["bin_accuracies"]) == 5
    assert len(metrics["bin_confidences"]) == 5
    
    # Brier score calculation manually:
    # Mean of (y_prob - y_true)^2
    expected_brier = np.mean((y_prob - y_true)**2)
    assert pytest.approx(metrics["brier"], 1e-5) == expected_brier


def test_compute_rf_prediction_uncertainty():
    # Mock Random Forest classifier with estimators
    mock_tree1 = MagicMock()
    mock_tree1.predict.return_value = np.array([0, 1, 0, 1])
    
    mock_tree2 = MagicMock()
    mock_tree2.predict.return_value = np.array([0, 1, 1, 0])
    
    mock_clf = MagicMock()
    # Mock estimator property
    mock_clf.estimators_ = [mock_tree1, mock_tree2]
    mock_clf._rf = mock_clf
    
    X = np.zeros((4, 18))
    valid_mask = np.array([True, True, True, True])
    spatial_shape = (2, 2)
    
    results = compute_rf_prediction_uncertainty(mock_clf, X, valid_mask, spatial_shape)
    
    assert "vote_entropy_2d" in results
    assert "vote_variance_2d" in results
    assert results["vote_entropy_2d"].shape == (2, 2)
    assert results["vote_variance_2d"].shape == (2, 2)
    
    # Entropy calculation checks:
    # Pixel 0: votes are 0, 0 -> p = 0 -> Entropy = 0
    # Pixel 1: votes are 1, 1 -> p = 1 -> Entropy = 0
    # Pixel 2: votes are 0, 1 -> p = 0.5 -> Entropy = 1.0
    # Pixel 3: votes are 1, 0 -> p = 0.5 -> Entropy = 1.0
    entropy_flat = results["entropy_flat"]
    assert entropy_flat[0] < 1e-5
    assert entropy_flat[1] < 1e-5
    assert pytest.approx(entropy_flat[2], 1e-5) == 1.0
    assert pytest.approx(entropy_flat[3], 1e-5) == 1.0


def test_analyze_prediction_errors_taxonomy():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    valid_mask = np.array([True, True, True, True])
    spatial_shape = (2, 2)
    
    # Features mock
    X = np.zeros((4, 4))
    feature_names = ["Delta_NDVI", "Delta_SAR", "Delta_NDWI", "Delta_Local_Var"]
    
    results = analyze_prediction_errors(
        y_true=y_true,
        y_pred=y_pred,
        valid_mask=valid_mask,
        spatial_shape=spatial_shape,
        X=X,
        feature_names=feature_names
    )
    
    assert "taxonomy" in results
    assert "confusion_pixel_counts" in results
    assert "tp_map_2d" in results
    assert "fp_map_2d" in results
    assert "fn_map_2d" in results
    assert "tn_map_2d" in results
    
    # Confusion count checks
    counts = results["confusion_pixel_counts"]
    assert counts["tp"] == 1
    assert counts["tn"] == 1
    assert counts["fp"] == 1
    assert counts["fn"] == 1
