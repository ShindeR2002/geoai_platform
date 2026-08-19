import sys
import os
import numpy as np
import pytest
import joblib
import tempfile
from pathlib import Path

# Ensure geoai package is in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.experiments.experiment_registry import get_experiment_model
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

class MockEstimator:
    def __init__(self, n_features=18):
        self.n_features = n_features
    def predict(self, X):
        return np.ones(len(X), dtype=int)
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 1] = 1.0
        return probs

class TestClassicalBaselines:
    """Tests for baseline model wrappers implementation."""

    def test_capabilities_implemented(self):
        models = ["extra_trees", "xgboost", "lightgbm", "catboost"]
        for m_id in models:
            model = get_experiment_model(m_id)
            assert model.is_implemented() is True
            caps = model.get_capabilities()
            assert len(caps.expected_input_channels) == 18

    def test_extra_trees_wrapper(self):
        model = get_experiment_model("extra_trees")
        mock_estimator = MockEstimator()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_pkl_path = Path(tmp_dir) / "et_model.pkl"
            joblib.dump(mock_estimator, model_pkl_path)
            
            # Load wrapper
            model.load(model_pkl_path)
            assert model.is_loaded is True
            
            # Predict
            X = np.ones((5, 18), dtype=np.float32)
            preds = model.predict(X)
            assert len(preds) == 5
            assert np.all(preds == 1)
            
            probs = model.predict_proba(X)
            assert probs.shape == (5, 2)
            assert np.allclose(probs[:, 1], 1.0)

    def test_xgboost_wrapper(self):
        model = get_experiment_model("xgboost")
        mock_estimator = MockEstimator()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_pkl_path = Path(tmp_dir) / "xgb_model.pkl"
            joblib.dump(mock_estimator, model_pkl_path)
            
            # Load wrapper
            model.load(model_pkl_path)
            assert model.is_loaded is True
            
            # Predict
            X = np.ones((5, 18), dtype=np.float32)
            preds = model.predict(X)
            assert len(preds) == 5
            assert np.all(preds == 1)
            
            probs = model.predict_proba(X)
            assert probs.shape == (5, 2)

    def test_lightgbm_wrapper(self):
        model = get_experiment_model("lightgbm")
        mock_estimator = MockEstimator()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_pkl_path = Path(tmp_dir) / "lgb_model.pkl"
            joblib.dump(mock_estimator, model_pkl_path)
            
            # Load wrapper
            model.load(model_pkl_path)
            assert model.is_loaded is True
            
            # Predict
            X = np.ones((5, 18), dtype=np.float32)
            preds = model.predict(X)
            assert len(preds) == 5
            
            probs = model.predict_proba(X)
            assert probs.shape == (5, 2)

    def test_catboost_wrapper(self):
        model = get_experiment_model("catboost")
        mock_estimator = MockEstimator()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_pkl_path = Path(tmp_dir) / "cat_model.pkl"
            joblib.dump(mock_estimator, model_pkl_path)
            
            # Load wrapper
            model.load(model_pkl_path)
            assert model.is_loaded is True
            
            # Predict
            X = np.ones((5, 18), dtype=np.float32)
            preds = model.predict(X)
            assert len(preds) == 5
            
            probs = model.predict_proba(X)
            assert probs.shape == (5, 2)
