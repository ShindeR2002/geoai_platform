import sys
import os
import numpy as np
import pytest

# Ensure geoai package is in import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geoai.features.boundary import (
    compute_continuous_gradient,
    compute_distance_transform_from_edges,
    compute_morphological_gradient,
    compute_morphological_boundaries,
    project_object_shapes,
    BoundaryRefinementCascade
)
from geoai.evaluation.boundary_metrics import (
    compute_boundary_iou,
    compute_chamfer_distance,
    compute_hausdorff_distance,
    compute_boundary_smoothness_index,
    compute_average_thickness,
    compute_boundary_campaign_metrics
)

class MockEstimator:
    """Mock estimator to mimic a scikit-learn classifier."""
    def __init__(self, output_val: int = 1) -> None:
        self.output_val = output_val

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=int) * self.output_val


class TestBoundaryFeatures:
    """Tests for boundary feature extractors."""
    
    def test_continuous_gradients(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[3:7, 3:7] = 1.0 # square object
        
        # Test Sobel, Scharr, Laplacian
        for method in ("sobel", "scharr", "laplacian"):
            grad = compute_continuous_gradient(img, method=method)
            assert grad.shape == (10, 10)
            assert np.any(grad > 0.0) # Should have detected boundaries
            
    def test_distance_transform_from_edges(self):
        img = np.zeros((20, 20), dtype=np.float32)
        img[5:15, 5:15] = 1.0
        
        edt = compute_distance_transform_from_edges(img, sigma=1.0)
        assert edt.shape == (20, 20)
        # EDT should be non-binary
        assert not np.array_equal(edt, edt.astype(bool))
        
    def test_morphological_gradient(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[3:7, 3:7] = 1.0
        
        grad = compute_morphological_gradient(img, size=3)
        assert grad.shape == (10, 10)
        assert np.any(grad > 0.0)
        
    def test_morphological_boundaries(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        
        internal, external, thickness = compute_morphological_boundaries(mask)
        assert internal.shape == (10, 10)
        assert external.shape == (10, 10)
        assert thickness.shape == (10, 10)
        
    def test_project_object_shapes(self):
        label_arr = np.zeros((10, 10), dtype=int)
        label_arr[2:5, 2:6] = 1 # rectangle (non-zero eccentricity)
        label_arr[6:9, 6:9] = 2 # object 2
        valid_mask = np.ones((10, 10), dtype=bool)
        
        shapes = project_object_shapes(label_arr, valid_mask)
        for metric in ("solidity", "compactness", "elongation", "eccentricity", "convexity"):
            assert metric in shapes
            assert shapes[metric].shape == (10, 10)
            assert np.any(shapes[metric] > 0.0)
            
    def test_boundary_refinement_cascade(self):
        base_model = MockEstimator(output_val=1)
        refined_model = MockEstimator(output_val=1)
        
        cascade = BoundaryRefinementCascade(base_model, refined_model)
        
        # 10x10 baseline inputs
        X_base_flat = np.ones((50, 18), dtype=np.float32)
        feature_cube_base = np.ones((10, 10, 18), dtype=np.float32)
        valid_mask_2d = np.zeros((10, 10), dtype=bool)
        valid_mask_2d[2:7, 2:7] = True # 25 valid pixels, but flat has 50 just for simulation
        
        # Flattened base inputs must match number of valid pixels in valid_mask_2d
        X_base_flat = np.ones((25, 18), dtype=np.float32)
        
        y_pred_refined, y_pred_base, label_arr = cascade.predict_refined(
            X_base_flat=X_base_flat,
            feature_cube_base=feature_cube_base,
            valid_mask_2d=valid_mask_2d
        )
        
        assert y_pred_refined.shape == (25,)
        assert y_pred_base.shape == (25,)
        assert label_arr.shape == (10, 10)


class TestBoundaryMetrics:
    """Tests for boundary quality metrics."""
    
    def test_boundary_iou(self):
        y_true = np.zeros((10, 10), dtype=bool)
        y_true[3:7, 3:7] = True
        
        # Same prediction
        y_pred = y_true.copy()
        iou = compute_boundary_iou(y_true, y_pred, buffer_dist=2)
        assert iou == 1.0
        
        # Shifted prediction
        y_pred = np.zeros((10, 10), dtype=bool)
        y_pred[4:8, 4:8] = True
        iou = compute_boundary_iou(y_true, y_pred, buffer_dist=2)
        assert 0.0 < iou < 1.0
        
    def test_chamfer_distance(self):
        y_true = np.zeros((10, 10), dtype=bool)
        y_true[3:7, 3:7] = True
        
        y_pred = y_true.copy()
        dist = compute_chamfer_distance(y_true, y_pred)
        assert dist == 0.0
        
        # Non-matching
        y_pred = np.zeros((10, 10), dtype=bool)
        y_pred[4:8, 4:8] = True
        dist = compute_chamfer_distance(y_true, y_pred)
        assert dist > 0.0
        
    def test_hausdorff_distance(self):
        y_true = np.zeros((10, 10), dtype=bool)
        y_true[3:7, 3:7] = True
        
        y_pred = y_true.copy()
        hd = compute_hausdorff_distance(y_true, y_pred)
        assert hd == 0.0
        
        # Non-matching
        y_pred = np.zeros((10, 10), dtype=bool)
        y_pred[4:8, 4:8] = True
        hd = compute_hausdorff_distance(y_true, y_pred)
        assert hd > 0.0
        
    def test_boundary_smoothness_index(self):
        # A simple square object
        y_true = np.zeros((10, 10), dtype=bool)
        y_true[3:7, 3:7] = True
        
        bsi = compute_boundary_smoothness_index(y_true)
        # Convex hull of a square is very close to the square itself
        assert 0.8 <= bsi <= 1.0
        
    def test_average_thickness(self):
        y_true = np.zeros((10, 10), dtype=bool)
        y_true[3:7, 3:7] = True
        
        thickness = compute_average_thickness(y_true)
        assert thickness > 0.0
        
    def test_boundary_campaign_metrics_dict(self):
        y_true = np.zeros((10, 10), dtype=bool)
        y_true[3:7, 3:7] = True
        y_pred = np.zeros((10, 10), dtype=bool)
        y_pred[4:8, 4:8] = True
        
        metrics = compute_boundary_campaign_metrics(y_true, y_pred)
        for key in ("boundary_iou", "chamfer_distance", "hausdorff_distance", "bsi_diff", "thickness_diff"):
            assert key in metrics
            assert isinstance(metrics[key], float)
