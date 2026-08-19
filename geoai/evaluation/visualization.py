import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def plot_feature_importance(importances: Dict[str, float], output_path: Path) -> None:
    """Save feature importance horizontal bar chart."""
    if not importances:
        return
        
    names = list(importances.keys())[::-1]
    values = list(importances.values())[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(names)))
    ax.barh(names, values, color=colors, edgecolor='grey')
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Model Feature Importance")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_correlation_heatmap(corr_df: pd.DataFrame, method: str, output_path: Path) -> None:
    """Save correlation matrix heatmap using matplotlib."""
    if corr_df.empty:
        return
        
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    
    # Grid ticks
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(f"Feature Correlation Heatmap ({method.capitalize()})")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_confusion_matrix(cm: List[List[int]], output_path: Path) -> None:
    """Save Confusion Matrix plot."""
    cm_arr = np.array(cm)
    if cm_arr.size != 4:
        return
        
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=['No-Change', 'Change'], yticklabels=['No-Change', 'Change'],
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black")
                    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_error_map(error_map_2d: np.ndarray, output_path: Path) -> None:
    """Save 2D spatial error map (0=TN, 1=TP, 2=FP, 3=FN)."""
    if error_map_2d.ndim != 2:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom color palette: TN=gray, TP=green, FP=red, FN=blue
    from matplotlib.colors import ListedColormap
    colors = ['#E0E0E0', '#4CAF50', '#F44336', '#2196F3']
    cmap = ListedColormap(colors)
    
    im = ax.imshow(error_map_2d, cmap=cmap, vmin=0, vmax=3)
    ax.axis("off")
    ax.set_title("Spatial Error Map (TN=Gray, TP=Green, FP=Red, FN=Blue)")
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E0E0E0', edgecolor='grey', label='True Negative (TN)'),
        Patch(facecolor='#4CAF50', edgecolor='grey', label='True Positive (TP)'),
        Patch(facecolor='#F44336', edgecolor='grey', label='False Positive (FP)'),
        Patch(facecolor='#2196F3', edgecolor='grey', label='False Negative (FN)'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_runtime_latency(latency_dict: Dict[str, float], output_path: Path) -> None:
    """Save execution runtime latency breakdown bar chart."""
    if not latency_dict:
        return
        
    # Exclude total execution time
    keys = [k for k in latency_dict.keys() if k != "total_execution_time"]
    values = [latency_dict[k] for k in keys]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(keys, values, color="skyblue", edgecolor="grey")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Runtime Latency Breakdown")
    plt.xticks(rotation=15)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_separated_error_maps(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    output_dir: Path
) -> None:
    """Save separated binary maps for True Positive, False Positive, False Negative, and True Negatives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    maps = {
        "true_positives": (tp, "Greens", "True Positive (TP) Map"),
        "false_positives": (fp, "Reds", "False Positive (FP) Map"),
        "false_negatives": (fn, "Blues", "False Negative (FN) Map"),
        "true_negatives": (tn, "Greys", "True Negative (TN) Map")
    }
    
    for filename, (data_2d, cmap, title) in maps.items():
        if data_2d.ndim != 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(data_2d, cmap=cmap)
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(output_dir / f"error_map_{filename}.png", dpi=150)
        plt.close()


def plot_reliability_diagram(
    bin_confidences: List[float],
    bin_accuracies: List[float],
    ece: float,
    mce: float,
    brier: float,
    output_path: Path
) -> None:
    """Save reliability diagram (calibration curve) with Perfect Calibration diagonal reference."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration reference diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    
    # Plot empirical calibration
    ax.plot(bin_confidences, bin_accuracies, marker="s", color="darkblue", label="Model Calibration")
    
    # Add histogram bins in the background if possible, or just plot bars
    ax.bar(bin_confidences, bin_accuracies, width=0.08, color="lightblue", alpha=0.3, align="center")
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Calibration Reliability Diagram")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Stats overlay box
    stats_text = f"ECE: {ece:.4f}\nMCE: {mce:.4f}\nBrier Score: {brier:.4f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_uncertainty_map(
    uncertainty_map: np.ndarray,
    output_path: Path
) -> None:
    """Save 2D prediction uncertainty (vote entropy) map as a heatmap."""
    if uncertainty_map.ndim != 2:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(uncertainty_map, cmap="inferno", vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Vote Entropy (Uncertainty)")
    ax.set_title("Pixel Prediction Uncertainty Map")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_vs_size(
    confidences: List[float],
    sizes_ha: List[float],
    output_path: Path
) -> None:
    """Save scatter plot representing confidence vs object size (hectares)."""
    if not confidences or not sizes_ha:
        return
        
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes_ha, confidences, color="darkviolet", alpha=0.6, edgecolors="white", s=40)
    ax.set_xlabel("Object Size (Hectares)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Prediction Confidence vs. Object Size")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_spatial_shap_maps(
    spatial_shap_maps: Dict[str, np.ndarray],
    feature_names: List[str],
    output_dir: Path
) -> None:
    """Save spatial SHAP contribution maps for main features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot top features if they exist in the dict
    for name, shap_2d in spatial_shap_maps.items():
        if shap_2d.ndim != 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use seismic colormap centered at 0.0 (red = positive push, blue = negative push)
        max_val = max(abs(shap_2d.min()), abs(shap_2d.max()))
        if max_val == 0.0:
            max_val = 1.0
        im = ax.imshow(shap_2d, cmap="seismic", vmin=-max_val, vmax=max_val)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="SHAP Value (Change Contribution)")
        ax.set_title(f"Spatial Feature Contribution: {name}")
        fig.tight_layout()
        plt.savefig(output_dir / f"spatial_shap_{name.lower()}.png", dpi=150)
        plt.close()


def plot_preprocessing_comparison(
    importances_base: Dict[str, float],
    importances_lee: Dict[str, float],
    confidences_base: np.ndarray,
    confidences_lee: np.ndarray,
    output_path: Path
) -> None:
    """Save side-by-side charts: Left: Feature Importances, Right: Confidence Distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Compare feature importances
    all_keys = list(set(importances_base.keys()).union(importances_lee.keys()))
    all_keys = sorted(all_keys, key=lambda x: importances_base.get(x, 0.0), reverse=True)[:8]
    
    x = np.arange(len(all_keys))
    width = 0.35
    
    ax1.bar(x - width/2, [importances_base.get(k, 0.0) for k in all_keys], width, label="Baseline RF", color="lightgrey", edgecolor="grey")
    ax1.bar(x + width/2, [importances_lee.get(k, 0.0) for k in all_keys], width, label="Refined Lee RF", color="emerald" if False else "#4CAF50", edgecolor="grey")
    
    ax1.set_ylabel("MDI Importance")
    ax1.set_title("Top Feature Importance Shift")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_keys, rotation=45, ha="right", fontsize=9)
    ax1.legend()
    
    # Right subplot: Confidence distributions
    ax2.hist(confidences_base, bins=20, alpha=0.5, label="Baseline RF", color="grey", edgecolor="black")
    ax2.hist(confidences_lee, bins=20, alpha=0.5, label="Refined Lee RF", color="#2196F3", edgecolor="black")
    ax2.set_xlabel("Prediction Probability (Class 1)")
    ax2.set_ylabel("Pixel Count")
    ax2.set_title("Probability Prediction Distribution Shift")
    ax2.legend()
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gradcam_and_activation(
    gradcam_map: np.ndarray,
    activation_map: np.ndarray,
    gc_output_path: Path,
    act_output_path: Path
) -> None:
    """Save Grad-CAM and low-level Conv activation maps as publication-quality PNGs."""
    # Grad-CAM plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(gradcam_map, cmap="jet", vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Normalized Grad-CAM Saliency")
    ax.set_title("Convolutional Grad-CAM Spatial Saliency Map")
    fig.tight_layout()
    plt.savefig(gc_output_path, dpi=150)
    plt.close()

    # Activation map plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(activation_map, cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Normalized Layer Activation")
    ax.set_title("First Convolutional Layer Mean Activation Map")
    fig.tight_layout()
    plt.savefig(act_output_path, dpi=150)
    plt.close()


