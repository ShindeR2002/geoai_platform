import time
from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    train_time: float = 0.0,
    inference_time: float = 0.0,
    model_size_bytes: int = 0,
    memory_usage_mb: float = 0.0
) -> Dict[str, Any]:
    """Compute performance metrics, IoU/Dice, confusion matrix, and execution profiling metrics."""
    # Handle NaNs
    valid = ~(np.isnan(y_true.astype(float)) | np.isnan(y_pred.astype(float)))
    y_t = y_true[valid].astype(int)
    y_p = y_pred[valid].astype(int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Calculate IoU (Jaccard)
    intersection = int(((y_t == 1) & (y_p == 1)).sum())
    union = int(((y_t == 1) | (y_p == 1)).sum())
    iou = intersection / union if union > 0 else 0.0

    # Calculate Dice coefficient
    total_pixels = int((y_t == 1).sum() + (y_p == 1).sum())
    dice = (2.0 * intersection) / total_pixels if total_pixels > 0 else 0.0

    cm = confusion_matrix(y_t, y_p).tolist()

    roc_auc = 0.0
    ap = 0.0
    if y_prob is not None and len(np.unique(y_t)) > 1:
        # Exclude NaNs from probability array
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            prob_col = y_prob[:, 1]
        else:
            prob_col = y_prob
        prob_col_valid = prob_col[valid]
        
        try:
            roc_auc = float(roc_auc_score(y_t, prob_col_valid))
            ap = float(average_precision_score(y_t, prob_col_valid))
        except Exception:
            pass

    throughput = len(y_t) / inference_time if inference_time > 0 else 0.0

    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "iou": round(iou, 6),
        "dice": round(dice, 6),
        "confusion_matrix": cm,
        "roc_auc": round(roc_auc, 6),
        "average_precision": round(ap, 6),
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4),
        "prediction_throughput": round(throughput, 2),
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 4),
        "memory_usage_mb": round(memory_usage_mb, 2)
    }
