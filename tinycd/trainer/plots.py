"""Plotting module for TinyCD production framework.

Generates loss curves, metric heatmaps, and calibration reliability diagrams.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def generate_training_curves(history_csv: Path, output_dir: Path) -> None:
    """Generate training metric progression curves from history CSV.

    Args:
        history_csv (Path): Path to history CSV file.
        output_dir (Path): Output directory for plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history_csv.exists():
        return

    # Load CSV data manually or using numpy
    try:
        data = np.genfromtxt(history_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.size == 0:
            return
        # If there's only 1 row, genfromtxt returns a 0-D array. Wrap it.
        if data.ndim == 0:
            data = np.array([data])
            
        epochs = data["epoch"]
        
        # 1. Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs, data["val_loss"], label="Val Loss", marker="x")
        plt.title("Loss Progression Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Helper to plot metric
        def plot_metric(metric_name: str, title: str, filename: str) -> None:
            plt.figure(figsize=(8, 5))
            train_col = f"train_{metric_name}"
            val_col = f"val_{metric_name}"
            if train_col in data.dtype.names:
                plt.plot(epochs, data[train_col], label=f"Train {title}", marker="o")
            if val_col in data.dtype.names:
                plt.plot(epochs, data[val_col], label=f"Val {title}", marker="x")
            plt.title(f"{title} Progression Curve")
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()

        # 2. IoU Curve
        plot_metric("iou", "IoU", "iou_curve.png")
        # 3. Precision Curve
        plot_metric("precision", "Precision", "precision_curve.png")
        # 4. Recall Curve
        plot_metric("recall", "Recall", "recall_curve.png")
        # 5. F1 Curve
        plot_metric("f1", "F1", "f1_curve.png")

        # 6. Learning Rate Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data["learning_rate"], label="Learning Rate", color="purple", marker="d")
        plt.title("Learning Rate Schedule Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "learning_rate_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to generate training curves: {e}")


def generate_confusion_matrix_heatmap(tn: int, fp: int, fn: int, tp: int, output_dir: Path) -> None:
    """Generate confusion matrix heatmap visualization.

    Args:
        tn (int): True negatives count.
        fp (int): False positives count.
        fn (int): False negatives count.
        tp (int): True positives count.
        output_dir (Path): Output directory for plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    
    # Set labels
    ax.set_title("Confusion Matrix Heatmap")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted No-Change", "Predicted Change"])
    ax.set_yticklabels(["True No-Change", "True Change"])
    
    # Annotate values
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            # Use white text for dark backgrounds, black for light
            color = "white" if val > (matrix.max() / 2) else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            
    fig.colorbar(im, ax=ax)
    plt.savefig(output_dir / "confusion_matrix_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, num_bins: int = 10) -> None:
    """Generate calibration reliability diagram.

    Args:
        y_true (np.ndarray): Binary ground truth.
        y_prob (np.ndarray): Predicted probability of positive class (shape [N] or [N, 2]).
        output_dir (Path): Output directory for plots.
        num_bins (int): Number of bins.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation checks
    if y_prob.ndim == 2:
        probs = y_prob[:, 1]
    else:
        probs = y_prob
        
    if len(y_true) == 0 or len(probs) == 0:
        return
        
    try:
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            if i == num_bins - 1:
                in_bin = in_bin | (probs == bin_upper)
                
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_accuracies.append(np.mean(y_true[in_bin]))
                bin_confidences.append(np.mean(probs[in_bin]))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_lower + bin_upper) / 2.0)
                
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.bar(bin_boundaries[:-1], bin_accuracies, width=1.0/num_bins, align="edge", alpha=0.7, label="Model", edgecolor="black")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Reliability Diagram")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.savefig(output_dir / "reliability_diagram.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to generate reliability diagram: {e}")
