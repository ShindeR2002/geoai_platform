"""Metrics Module for TinyCD production framework.

This module implements reusable functions for classification, calibration,
and placeholder targets for research campaigns.
"""

from typing import List, Tuple, Optional
import numpy as np

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute standard classification accuracy.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: Accuracy score in range [0, 1].
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))

def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary precision score.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: Precision score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))

def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary recall score.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: Recall score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary F1 score.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: F1 score.
    """
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))

def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Intersection-over-Union (Jaccard Index) for the positive class.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: IoU score.
    """
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    if union == 0:
        return 0.0
    return float(intersection / union)

def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Matthews Correlation Coefficient (MCC).

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: MCC score in range [-1, 1].
    """
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced classification accuracy.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        float: Balanced accuracy score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float((tpr + tnr) / 2.0)

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute TP, FP, TN, FN binary confusion values.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_pred (np.ndarray): Predicted binary label array.

    Returns:
        Tuple[int, int, int, int]: Tuple containing (tn, fp, fn, tp).
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score of predicted class probabilities.

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_prob (np.ndarray): Probability scores for the positive class (shape [N, 2] or [N]).

    Returns:
        float: Brier score.
    """
    if y_prob.ndim == 2:
        probs = y_prob[:, 1]
    else:
        probs = y_prob
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((probs - y_true) ** 2))

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true (np.ndarray): Binary ground truth array.
        y_prob (np.ndarray): Probability scores for the positive class (shape [N, 2] or [N]).
        num_bins (int): Number of bins to calculate calibration.

    Returns:
        float: ECE calibration score.
    """
    if y_prob.ndim == 2:
        probs = y_prob[:, 1]
    else:
        probs = y_prob
        
    if len(y_true) == 0:
        return 0.0
        
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == num_bins - 1:
            in_bin = in_bin | (probs == bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def compute_roc_auc_placeholder(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Placeholder method to compute ROC AUC for future extension.

    Args:
        y_true (np.ndarray): Ground truth list.
        y_prob (np.ndarray): Predicted probability list.

    Returns:
        Optional[float]: None by default.
    """
    return None

def compute_pr_auc_placeholder(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Placeholder method to compute PR AUC for future extension.

    Args:
        y_true (np.ndarray): Ground truth list.
        y_prob (np.ndarray): Predicted probability list.

    Returns:
        Optional[float]: None by default.
    """
    return None
