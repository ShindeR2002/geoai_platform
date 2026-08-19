from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

class BaseFeatureImportance(ABC):
    """Abstract base class for calculating feature importances in a model-agnostic way."""
    
    @abstractmethod
    def compute_importance(self, model: Any, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Compute importance scores for each feature.
        Returns a dictionary mapping feature name to importance score.
        """
        pass


class RandomForestMDIImportance(BaseFeatureImportance):
    """Calculates Mean Decrease in Impurity (MDI) importance from a trained Random Forest model."""
    
    def compute_importance(self, model: Any, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        if not hasattr(model, "feature_importances_"):
            # Attempt to look at self._rf if it's our wrapped model
            if hasattr(model, "_rf") and hasattr(model._rf, "feature_importances_"):
                importances = model._rf.feature_importances_
            else:
                raise ValueError("Model does not expose feature_importances_ attributes for MDI.")
        else:
            importances = model.feature_importances_
            
        scores = {}
        for i, val in enumerate(importances):
            if i < len(feature_names):
                scores[feature_names[i]] = float(val)
        # Sort descending
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


class PermutationFeatureImportance(BaseFeatureImportance):
    """Calculates Permutation Importance on a dataset using scikit-learn."""
    
    def __init__(self, n_repeats: int = 5, random_state: int = 42) -> None:
        self.n_repeats = n_repeats
        self.random_state = random_state

    def compute_importance(self, model: Any, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        # Handle wrapped vs raw estimator
        estimator = model
        if hasattr(model, "_rf") and model._rf is not None:
            estimator = model._rf
            
        result = permutation_importance(
            estimator, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        scores = {}
        for i, val in enumerate(result.importances_mean):
            if i < len(feature_names):
                scores[feature_names[i]] = float(val)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def compute_correlations(X: np.ndarray, feature_names: List[str], method: str = "pearson") -> pd.DataFrame:
    """
    Compute pairwise correlation matrix for all columns of X.
    Supports method='pearson', 'spearman', 'kendall'.
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"Unsupported correlation method: {method}. Allowed: pearson, spearman, kendall")
        
    df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    return df.corr(method=method)


def compute_feature_statistics(X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Compute basic distributions, variance, and missing value counts for each feature column in X.
    """
    stats = {}
    n_cols = X.shape[1]
    
    for i in range(n_cols):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        col = X[:, i]
        
        # Calculate statistics
        nan_count = int(np.isnan(col).sum())
        valid_col = col[~np.isnan(col)]
        
        if len(valid_col) == 0:
            stats[name] = {
                "mean": 0.0, "median": 0.0, "var": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "missing_count": nan_count
            }
            continue
            
        stats[name] = {
            "mean": round(float(np.mean(valid_col)), 6),
            "median": round(float(np.median(valid_col)), 6),
            "var": round(float(np.var(valid_col)), 6),
            "std": round(float(np.std(valid_col)), 6),
            "min": round(float(np.min(valid_col)), 6),
            "max": round(float(np.max(valid_col)), 6),
            "missing_count": nan_count
        }
        
    return stats


# ---------------------------------------------------------------------------
# Explainable AI (SHAP & Decision Path Analysis)
# ---------------------------------------------------------------------------

def compute_shap_explanations(
    clf: Any,
    X: np.ndarray,
    feature_names: List[str],
    num_samples: int = 500
) -> Dict[str, Any]:
    """
    Compute TreeSHAP explanations for a trained Random Forest model.
    Samples a subset of X to ensure responsive execution.
    """
    rf_model = clf._rf if hasattr(clf, "_rf") and clf._rf is not None else clf
    
    # Try importing shap inside
    try:
        import shap
    except ImportError:
        logger.warning("SHAP library not available for feature diagnostics.")
        return {}
        
    if not hasattr(rf_model, "estimators_"):
        logger.warning("Model is not an ensemble, skipping SHAP.")
        return {}
        
    n_samples = X.shape[0]
    sample_size = min(num_samples, n_samples)
    
    # Stratified sampling if possible, else random sampling
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
    X_sample = X[sample_indices].astype(np.float32)
    
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
        
        # For binary RF, shap_values is a list of [shap_class_0, shap_class_1]
        if isinstance(shap_values, list):
            shap_vals_class1 = shap_values[1]
        else:
            # For newer shap versions, shape can be (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                shap_vals_class1 = shap_values[:, :, 1]
            else:
                shap_vals_class1 = shap_values
                
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
        shap_summary = {
            name: float(mean_abs_shap[i])
            for i, name in enumerate(feature_names[:len(mean_abs_shap)])
        }
        
        return {
            "shap_values_sample": shap_vals_class1.tolist(),
            "X_sample": X_sample.tolist(),
            "mean_abs_shap": dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))
        }
    except Exception as e:
        logger.warning(f"Failed to calculate SHAP: {e}")
        return {}


def compute_spatial_shap_maps(
    clf: Any,
    feature_cube: np.ndarray,
    valid_mask: np.ndarray,
    spatial_shape: Tuple[int, int],
    feature_names: List[str],
    sample_step: int = 6
) -> Dict[str, np.ndarray]:
    """
    Project TreeSHAP values back to their 2D spatial coordinate grid.
    Samples every `sample_step` pixels to run TreeSHAP, then resizes/interpolates
    back to the original spatial shape to create a smooth contribution map.
    """
    H, W = spatial_shape
    total_pixels = H * W
    
    rf_model = clf._rf if hasattr(clf, "_rf") and clf._rf is not None else clf
    
    try:
        import shap
    except ImportError:
        return {}
        
    if not hasattr(rf_model, "estimators_"):
        return {}
        
    # Extract flattened features
    from geoai.utils.raster_utils import flatten_spatial
    X_flat = flatten_spatial(feature_cube)
    
    # Feature subsetting matching model capability (alignment)
    from geoai.utils.constants import CANONICAL_FEATURE_NAMES
    col_indices = [
        list(CANONICAL_FEATURE_NAMES).index(channel)
        for channel in feature_names
        if channel in CANONICAL_FEATURE_NAMES
    ]
    if col_indices:
        X_flat = X_flat[:, col_indices]
    
    # 2D grid index sampling
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    sampled_mask = (grid_y % sample_step == 0) & (grid_x % sample_step == 0)
    sampled_mask_flat = sampled_mask.ravel()
    
    # Combine with valid_mask
    active_mask = valid_mask & sampled_mask_flat
    
    if active_mask.sum() == 0:
        return {}
        
    X_active = X_flat[active_mask].astype(np.float32)
    
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X_active, check_additivity=False)
        
        if isinstance(shap_vals, list):
            shap_vals_class1 = shap_vals[1]
        else:
            if shap_vals.ndim == 3:
                shap_vals_class1 = shap_vals[:, :, 1]
            else:
                shap_vals_class1 = shap_vals
                
        # Reconstruct 2D sampled SHAP maps, then perform simple block dilation to fill gaps
        active_indices = np.where(active_mask)[0]
        flat_to_active_idx = {flat_idx: row_idx for row_idx, flat_idx in enumerate(active_indices)}
        
        spatial_shap_maps = {}
        for f_idx, name in enumerate(feature_names[:feature_cube.shape[2]]):
            shap_flat = np.zeros(total_pixels, dtype=np.float32)
            shap_flat[active_mask] = shap_vals_class1[:, f_idx]
            
            # Form 2D grid
            shap_2d = shap_flat.reshape(H, W)
            
            # Simple nearest-neighbor fill-in for unsampled points:
            # We can use scipy ndimage zoom or simple block expansion for display
            from scipy.ndimage import zoom
            # Extract only the sampled values on a smaller grid
            small_h = int(np.ceil(H / sample_step))
            small_w = int(np.ceil(W / sample_step))
            
            # Create a small grid and fill it
            small_grid = np.zeros((small_h, small_w), dtype=np.float32)
            for pos in np.argwhere(sampled_mask):
                sy, sx = pos
                sm_y = sy // sample_step
                sm_x = sx // sample_step
                if sm_y < small_h and sm_x < small_w:
                    # Look up flat index
                    flat_idx = sy * W + sx
                    row_idx = flat_to_active_idx.get(flat_idx)
                    if row_idx is not None:
                        small_grid[sm_y, sm_x] = shap_vals_class1[row_idx, f_idx]
                        
            # Zoom back up to H, W
            zoom_y = H / small_h
            zoom_x = W / small_w
            interpolated_shap_2d = zoom(small_grid, (zoom_y, zoom_x), order=1)[:H, :W]
            
            # Zero out non-valid pixels
            interpolated_shap_2d[~valid_mask.reshape(H, W)] = 0.0
            
            spatial_shap_maps[name] = interpolated_shap_2d
            
        return spatial_shap_maps
    except Exception as e:
        logger.warning(f"Failed spatial SHAP mapping: {e}")
        return {}


def extract_decision_path(
    clf: Any,
    sample_x: np.ndarray,
    feature_names: List[str],
    tree_idx: int = 0
) -> List[str]:
    """
    Extract the split decisions for a single test sample along a specific decision tree
    in the Random Forest model. Returns list of human-readable decision boundaries.
    """
    rf_model = clf._rf if hasattr(clf, "_rf") and clf._rf is not None else clf
    
    if not hasattr(rf_model, "estimators_") or len(rf_model.estimators_) <= tree_idx:
        return ["Decision path unavailable: Model not ensemble or tree index invalid."]
        
    tree = rf_model.estimators_[tree_idx]
    
    # Get decision path indices
    node_indicator = tree.decision_path(sample_x.reshape(1, -1))
    leaf_id = tree.apply(sample_x.reshape(1, -1))[0]
    
    # Get features and thresholds
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    path = []
    for node_id in node_index:
        if node_id == leaf_id:
            # Leaf node reached
            val = tree.tree_.value[node_id]
            pred_class = int(np.argmax(val[0]))
            path.append(f"Leaf Node {node_id} (votes: {val[0]}, predicted: {pred_class})")
            continue
            
        feature_idx = features[node_id]
        if feature_idx == -2: # undefined feature split at leaf
            continue
            
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
        threshold = thresholds[node_id]
        
        val_sample = sample_x[feature_idx]
        
        if val_sample <= threshold:
            path.append(f"Node {node_id}: {feature_name} (value: {val_sample:.4f}) <= {threshold:.4f}")
        else:
            path.append(f"Node {node_id}: {feature_name} (value: {val_sample:.4f}) > {threshold:.4f}")
            
    return path


def generate_gradcam(model: Any, input_tensor: Any, target_class: int = 1, layer: Any = None) -> np.ndarray:
    """
    Generate Gradient-weighted Class Activation Mapping (Grad-CAM) for a convolutional PyTorch model.
    If layer is None, automatically finds and hooks the last Conv2d layer in the model.
    """
    import torch
    
    # Find all conv2d layers
    conv_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
            
    if not conv_layers:
        logger.warning("No Conv2d layers found in the deep learning model for Grad-CAM.")
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        
    target_layer = layer if layer is not None else conv_layers[-1]
    
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
        
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
        
    h_f = target_layer.register_forward_hook(forward_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        h_b = target_layer.register_full_backward_hook(backward_hook)
    else:
        h_b = target_layer.register_backward_hook(backward_hook)
    
    # Run model forward
    model.zero_grad()
    logits = model(input_tensor)
    
    # Calculate score (average center pixel or global average score for target class)
    score = logits[:, target_class].sum()
    score.backward()
    
    # Remove hooks
    h_f.remove()
    h_b.remove()
    
    if activations is None or gradients is None:
        logger.warning("Failed capturing activations or gradients in target layer hook.")
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        
    alpha = gradients.mean(dim=[2, 3], keepdim=True)
    gradcam = (alpha * activations).sum(dim=1, keepdim=True)
    gradcam = torch.relu(gradcam)
    
    # Interpolate to input tensor size
    gradcam_resized = torch.nn.functional.interpolate(
        gradcam,
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode='bilinear',
        align_corners=False
    )
    
    gradcam_np = gradcam_resized[0, 0].cpu().numpy()
    
    # Normalize to [0, 1]
    denom = gradcam_np.max() - gradcam_np.min()
    if denom > 0.0:
        gradcam_np = (gradcam_np - gradcam_np.min()) / denom
    else:
        gradcam_np = np.zeros_like(gradcam_np)
        
    return gradcam_np


def generate_activation_maps(model: Any, input_tensor: Any, layer: Any = None) -> np.ndarray:
    """
    Generate mean activation map across all channels of the target layer.
    If layer is None, automatically hooks the first Conv2d layer in the model to visualize low-level activations.
    """
    import torch
    
    conv_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
            
    if not conv_layers:
        logger.warning("No Conv2d layers found in the deep learning model for activation mapping.")
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        
    target_layer = layer if layer is not None else conv_layers[0]
    
    activations = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
        
    h_f = target_layer.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        _ = model(input_tensor)
        
    h_f.remove()
    
    if activations is None:
        logger.warning("Failed capturing activations in target layer hook.")
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        
    mean_act = activations[0].mean(dim=0, keepdim=True).unsqueeze(0)
    mean_act = torch.relu(mean_act)
    
    mean_act_resized = torch.nn.functional.interpolate(
        mean_act,
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode='bilinear',
        align_corners=False
    )
    
    mean_act_np = mean_act_resized[0, 0].cpu().numpy()
    
    # Normalize to [0, 1]
    denom = mean_act_np.max() - mean_act_np.min()
    if denom > 0.0:
        mean_act_np = (mean_act_np - mean_act_np.min()) / denom
    else:
        mean_act_np = np.zeros_like(mean_act_np)
        
    return mean_act_np


