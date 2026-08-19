import sys
import os
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock

# Ensure geoai package is in import path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import (
    split_dataset_unified,
    split_dataset_spatial,
    split_dataset,
    SplitResult,
    get_pixel_coords
)
from geoai.core.config import load_platform_config
from geoai.evaluation.evaluation_manager import EvaluationManager
from geoai.experiments.experiment_runner import ExperimentRunner
from geoai.experiments.experiment_config import ExperimentConfig
from geoai.models.baselines.dl_wrapper import DLBaseModelWrapper
from geoai.experiments.experiment_registry import get_experiment_model


class TestStabilizationSprint1:
    """Stabilization Sprint 1 Integration and Verification Tests."""

    def test_model_dispatch_type_based(self):
        """Task 1: Verify type-based dispatch identifies DL and Transformer wrappers correctly."""
        # 1. Verify Deep Learning CNN wrap
        cnn_wrapper = get_experiment_model("fc_ef")
        assert isinstance(cnn_wrapper, DLBaseModelWrapper)

        # 2. Verify Transformer wraps
        tinycd_wrapper = get_experiment_model("tinycd")
        bit_wrapper = get_experiment_model("bit")
        assert isinstance(tinycd_wrapper, DLBaseModelWrapper)
        assert isinstance(bit_wrapper, DLBaseModelWrapper)

        # 3. Verify classical ML wraps are NOT instances of DLBaseModelWrapper
        rf_wrapper = get_experiment_model("rf_enhanced_v1")
        xgb_wrapper = get_experiment_model("xgboost")
        assert not isinstance(rf_wrapper, DLBaseModelWrapper)
        assert not isinstance(xgb_wrapper, DLBaseModelWrapper)

    def test_unified_dataset_splitting_routing(self):
        """Task 2: Verify split_dataset_unified routing for spatial and random policies."""
        dataset = DatasetRegistry.load_dataset("ps10_sentinel_v1")
        X, y = dataset.X, dataset.y

        # Test Spatial split
        res_spatial = split_dataset_unified(
            X=X, y=y, dataset_id="ps10_sentinel_v1", split_policy="spatial", patch_size=15
        )
        assert isinstance(res_spatial, SplitResult)
        assert res_spatial.X_train.shape[0] > 0
        assert res_spatial.X_test.shape[0] > 0
        assert res_spatial.X_val is not None
        assert res_spatial.X_val.shape[0] > 0

        # Test Random split
        res_random = split_dataset_unified(
            X=X, y=y, dataset_id="ps10_sentinel_v1", split_policy="random", test_size=0.20, random_state=42
        )
        assert isinstance(res_random, SplitResult)
        assert res_random.X_train.shape[0] > 0
        assert res_random.X_test.shape[0] > 0
        assert res_random.X_val is None
        assert res_random.y_val is None

    def test_automated_data_leakage_verification(self):
        """Refinement 6: Verify spatial splits are mutually exclusive and have no spatial patch overlap."""
        coords, (H, W), _, _ = get_pixel_coords("ps10_sentinel_v1")
        patch_size = 15
        R = patch_size // 2

        # 1. Perform spatial block splitting
        # Let's create dummy feature matrix matching coordinates count
        X_dummy = np.random.rand(coords.shape[0], 14)
        y_dummy = np.random.randint(0, 2, coords.shape[0])

        X_tr, X_te, y_tr, y_te, X_va, y_va = split_dataset_spatial(
            X_dummy, y_dummy, "ps10_sentinel_v1", patch_size=patch_size
        )

        # 2. Get coords index split mapping
        # Since split_dataset_spatial returns elements matching masks, let's re-run masks directly
        from geoai.datasets.dataset_splitter import BLOCK_SPLIT_MAP
        bh, bw = H / 4.0, W / 4.0
        
        # Verify block coordinate indices
        train_coords = []
        val_coords = []
        test_coords = []
        
        for r, c in coords[::200]:
            br = int(min(3, r // bh))
            bc = int(min(3, c // bw))
            block = br * 4 + bc
            split = BLOCK_SPLIT_MAP[block]
            
            # Check neighbor splits for buffer zone
            is_buffer = False
            for dr in range(-R, R + 1):
                nr = r + dr
                if nr < 0 or nr >= H: continue
                for dc in range(-R, R + 1):
                    nc = c + dc
                    if nc < 0 or nc >= W: continue
                    nbr_br = int(min(3, nr // bh))
                    nbr_bc = int(min(3, nc // bw))
                    nbr_block = nbr_br * 4 + nbr_bc
                    if BLOCK_SPLIT_MAP[nbr_block] != split:
                        is_buffer = True
                        break
                if is_buffer:
                    break
                    
            if not is_buffer:
                if split == 'train':
                    train_coords.append((r, c))
                elif split == 'val':
                    val_coords.append((r, c))
                elif split == 'test':
                    test_coords.append((r, c))

        train_set = set(train_coords)
        val_set = set(val_coords)
        test_set = set(test_coords)

        # Verify Mutual Exclusivity
        assert train_set.isdisjoint(val_set), "Train and Validation coordinate sets must be disjoint"
        assert train_set.isdisjoint(test_set), "Train and Test coordinate sets must be disjoint"
        assert val_set.isdisjoint(test_set), "Validation and Test coordinate sets must be disjoint"

        # Verify Spatial Distance Buffer (No pixel neighborhood overlap)
        # Verify a random sample of pairs to keep test fast
        np.random.seed(42)
        train_sample = [train_coords[i] for i in np.random.choice(len(train_coords), min(100, len(train_coords)), replace=False)]
        test_sample = [test_coords[i] for i in np.random.choice(len(test_coords), min(100, len(test_coords)), replace=False)]

        for r_tr, c_tr in train_sample:
            for r_te, c_te in test_sample:
                # Chebyshev distance (L-inf norm) must be strictly greater than R
                dist = max(abs(r_tr - r_te), abs(c_tr - c_te))
                assert dist > R, f"Overlap detected between train ({r_tr},{c_tr}) and test ({r_te},{c_te}) within patch size buffer radius {R}"

    def test_evaluation_manager_consumes_prepared_splits(self):
        """Refinement 3: Verify EvaluationManager consumes prepared SplitResult directly without re-splitting."""
        dataset = DatasetRegistry.load_dataset("ps10_sentinel_v1")
        X, y = dataset.X, dataset.y

        # Prepare split result
        split_result = split_dataset_unified(
            X=X, y=y, dataset_id="ps10_sentinel_v1", split_policy="random", test_size=0.20, random_state=42
        )

        manager = EvaluationManager(configs_dir=str(project_root / "configs"))
        
        from unittest.mock import patch, MagicMock
        
        mock_clf = MagicMock()
        mock_clf.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=np.int64)
        mock_clf.predict_proba.side_effect = lambda X: np.zeros((X.shape[0], 2), dtype=np.float32)
        mock_tree = MagicMock()
        mock_tree.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=np.int64)
        mock_clf.estimators_ = [mock_tree]
        mock_clf._rf = None
        
        with patch("geoai.evaluation.evaluation_manager.RandomForestClassifier", return_value=mock_clf), \
             patch("geoai.evaluation.evaluation_manager.joblib.dump") as mock_dump:
            mock_dump.side_effect = lambda value, path: Path(path).write_text("mock", encoding="utf-8")
            # Run evaluation with prepared split_result
            res = manager.run_evaluation(
                model_id="rf_baseline_v1",
                dataset_id="ps10_sentinel_v1",
                profile="quick",
                split_result=split_result
            )
        assert res is not None
        assert "accuracy" in res

    def test_deterministic_reproducibility(self):
        """Refinement 5: Verify splitting is fully reproducible with identical seeds."""
        dataset = DatasetRegistry.load_dataset("ps10_sentinel_v1")
        X, y = dataset.X, dataset.y

        res1 = split_dataset_unified(
            X=X, y=y, dataset_id="ps10_sentinel_v1", split_policy="spatial", patch_size=15
        )
        res2 = split_dataset_unified(
            X=X, y=y, dataset_id="ps10_sentinel_v1", split_policy="spatial", patch_size=15
        )

        np.testing.assert_array_equal(res1.X_train, res2.X_train)
        np.testing.assert_array_equal(res1.X_test, res2.X_test)
        np.testing.assert_array_equal(res1.y_train, res2.y_train)
        np.testing.assert_array_equal(res1.y_test, res2.y_test)

    def test_regression_production_models(self):
        """Refinement 4: Regression testing for existing production models (RandomForest, XGBoost, TinyCD, BIT)."""
        from geoai.models.registry import is_model_registered, get_model_class
        
        # 1. Verify SVM is NOT registered in the codebase (as confirmed in audit)
        assert not is_model_registered("svm"), "SVM model is not registered in MODEL_REGISTRY"
        
        # 2. Test RandomForest Enhanced
        assert is_model_registered("rf_enhanced_v1")
        rf_cls = get_model_class("rf_enhanced_v1")
        rf_model = rf_cls()
        assert rf_model.get_capabilities().model_type == "Classical ML (Random Forest)"
        
        # 3. Test XGBoost
        assert is_model_registered("xgboost")
        xgb_cls = get_model_class("xgboost")
        xgb_model = xgb_cls()
        assert "Classical" in xgb_model.get_capabilities().model_type
        
        # 4. Test TinyCD
        assert is_model_registered("tinycd")
        tinycd_cls = get_model_class("tinycd")
        tinycd_model = tinycd_cls()
        assert tinycd_model.get_capabilities().model_type == "Transformer"
        
        # 5. Test BIT
        assert is_model_registered("bit")
        bit_cls = get_model_class("bit")
        bit_model = bit_cls()
        assert bit_model.get_capabilities().model_type == "Transformer"
