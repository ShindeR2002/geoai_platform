"""Evaluator module for TinyCD production framework.

Evaluates trained checkpoint weights on the test set and exports CSV summaries
of performance metrics, confusion matrices, and prediction probabilities.
"""

import csv
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from tinycd.trainer.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_iou,
    compute_mcc,
    compute_balanced_accuracy,
    compute_confusion_matrix,
    compute_brier_score,
    compute_ece
)

class Evaluator:
    """Orchestrates test split evaluation and exports summary files."""

    def __init__(self, metrics_dir: Path) -> None:
        """Initialize Evaluator.

        Args:
            metrics_dir (Path): Dedicated directory for metrics output.
        """
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Compute all classification and calibration performance metrics.

        Args:
            y_true (np.ndarray): Binary ground truth array.
            y_pred (np.ndarray): Predicted binary label array.
            y_prob (np.ndarray): Probability scores for the positive class (shape [N] or [N, 2]).

        Returns:
            Dict[str, float]: A dictionary containing calculated performance metric values.
        """
        # Ensure correct dimensions
        if y_prob.ndim == 2:
            probs = y_prob[:, 1]
        else:
            probs = y_prob

        metrics = {
            "Accuracy": compute_accuracy(y_true, y_pred),
            "Precision": compute_precision(y_true, y_pred),
            "Recall": compute_recall(y_true, y_pred),
            "F1": compute_f1(y_true, y_pred),
            "IoU": compute_iou(y_true, y_pred),
            "MCC": compute_mcc(y_true, y_pred),
            "Balanced Accuracy": compute_balanced_accuracy(y_true, y_pred),
            "Brier Score": compute_brier_score(y_true, probs),
            "ECE": compute_ece(y_true, probs)
        }
        return metrics

    def export_results(
        self,
        metrics: Dict[str, float],
        confusion_matrix_vals: Tuple[int, int, int, int],
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> None:
        """Export metrics, confusion matrix, and prediction probabilities to CSV files.

        Args:
            metrics (Dict[str, float]): Calculated metrics key-value dictionary.
            confusion_matrix_vals (Tuple[int, int, int, int]): Tuple containing (tn, fp, fn, tp).
            y_true (np.ndarray): Binary ground truth labels.
            y_prob (np.ndarray): Probability scores (shape [N] or [N, 2]).
        """
        # 1. Export evaluation_metrics.csv
        metrics_csv = self.metrics_dir / "evaluation_metrics.csv"
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, f"{v:.6f}"])

        # 2. Export confusion_matrix.csv
        tn, fp, fn, tp = confusion_matrix_vals
        cm_csv = self.metrics_dir / "confusion_matrix.csv"
        with open(cm_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Actual/Predicted", "Predicted No-Change", "Predicted Change"])
            writer.writerow(["Actual No-Change", tn, fp])
            writer.writerow(["Actual Change", fn, tp])

        # 3. Export prediction_scores.csv
        if y_prob.ndim == 2:
            probs = y_prob[:, 1]
        else:
            probs = y_prob
            
        scores_csv = self.metrics_dir / "prediction_scores.csv"
        with open(scores_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "predicted_probability", "ground_truth_label"])
            for idx, (prob, label) in enumerate(zip(probs, y_true)):
                writer.writerow([idx, f"{prob:.6f}", int(label)])
