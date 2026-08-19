"""
Model evaluation for the GeoAI Platform.

Computes all evaluation metrics required for PS10 challenge assessment and
internal model comparison. The primary PS10 metric is the Jaccard Index
(Intersection over Union). The standard report includes accuracy, precision,
recall, F1, Jaccard, and a confusion matrix.

All metric computations operate on flat integer label arrays. Spatial
aggregation (mask → flat) is handled by the pipeline before calling these
functions.

Single responsibility: compute and format model performance metrics.

Position in dependency hierarchy: models (depends on core only — no sklearn
imports create circular dependencies; sklearn is used here but not re-exported).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from geoai.core.exceptions import ModelError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute all standard evaluation metrics for a binary prediction.

    Computes the complete evaluation suite used for PS10 assessment and model
    comparison: accuracy, precision, recall, F1, Jaccard Index, and confusion
    matrix. Results are returned as a structured dictionary and logged at
    INFO level.

    Args:
        y_true: Ground truth label array of shape (N,) with integer values
            (0 = no change, 1 = change). NaN values are excluded automatically.
        y_pred: Predicted label array of shape (N,) matching y_true.
        class_names: Optional list of class name strings for report formatting.
            Defaults to ['no_change', 'change'].

    Returns:
        Dictionary with keys:
            - ``accuracy``: Overall classification accuracy.
            - ``precision``: Precision for the change class.
            - ``recall``: Recall for the change class.
            - ``f1``: F1 score for the change class.
            - ``jaccard``: Jaccard Index (IoU) for the change class.
            - ``confusion_matrix``: 2D list [[TN, FP], [FN, TP]].
            - ``classification_report``: Full sklearn text report string.
            - ``n_samples``: Number of samples evaluated.
            - ``n_change_true``: Number of true change pixels.
            - ``n_change_pred``: Number of predicted change pixels.

    Raises:
        ModelError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ModelError(
            f"y_true has {len(y_true)} elements but y_pred has {len(y_pred)}. "
            "Arrays must have the same length."
        )

    if class_names is None:
        class_names = ["no_change", "change"]

    # Remove NaN positions if present (can occur in spatial evaluation).
    valid = ~(np.isnan(y_true.astype(float)) | np.isnan(y_pred.astype(float)))
    if not valid.all():
        n_removed = int((~valid).sum())
        logger.warning(
            "Evaluation: removed %d NaN positions from evaluation arrays.",
            n_removed,
        )
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    acc = float(accuracy_score(y_true_int, y_pred_int))
    prec = float(precision_score(y_true_int, y_pred_int, zero_division=0))
    rec = float(recall_score(y_true_int, y_pred_int, zero_division=0))
    f1 = float(f1_score(y_true_int, y_pred_int, zero_division=0))
    jac = float(compute_jaccard(y_true_int, y_pred_int))
    cm = confusion_matrix(y_true_int, y_pred_int).tolist()
    report = classification_report(
        y_true_int, y_pred_int,
        target_names=class_names,
        zero_division=0,
    )

    results = {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "jaccard": round(jac, 6),
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(len(y_true_int)),
        "n_change_true": int((y_true_int == 1).sum()),
        "n_change_pred": int((y_pred_int == 1).sum()),
    }

    logger.info(
        "Evaluation — accuracy=%.4f precision=%.4f recall=%.4f "
        "f1=%.4f jaccard=%.4f n=%d.",
        acc, prec, rec, f1, jac, len(y_true_int),
    )
    return results


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_jaccard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    change_label: int = 1,
) -> float:
    """Compute the Jaccard Index (Intersection over Union) for the change class.

    The Jaccard Index is the primary evaluation metric for the PS10 challenge.

    Jaccard = |TP| / (|TP| + |FP| + |FN|)

    Where:
      TP = true positives (correctly predicted change pixels)
      FP = false positives (no-change pixels predicted as change)
      FN = false negatives (change pixels predicted as no-change)

    Args:
        y_true: Ground truth binary labels of shape (N,).
        y_pred: Predicted binary labels of shape (N,).
        change_label: Integer label denoting the change class. Default 1.

    Returns:
        Jaccard Index as a float in [0, 1]. Returns 0.0 if no change pixels
        exist in either y_true or y_pred.
    """
    y_true_bin = (y_true == change_label)
    y_pred_bin = (y_pred == change_label)

    intersection = int((y_true_bin & y_pred_bin).sum())
    union = int((y_true_bin | y_pred_bin).sum())

    if union == 0:
        logger.warning(
            "Jaccard Index: no change pixels in either y_true or y_pred. "
            "Returning 0.0."
        )
        return 0.0

    jaccard = intersection / union
    logger.debug("Jaccard Index = %d / %d = %.6f.", intersection, union, jaccard)
    return jaccard


def compute_spatial_jaccard(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> float:
    """Compute the Jaccard Index directly from 2D binary rasters.

    Convenience function for spatial evaluation without prior flattening.

    Args:
        true_mask: Ground truth binary raster of shape (H, W) with values
            0 and 1. NaN values are treated as no-change.
        pred_mask: Predicted binary raster of shape (H, W). NaN values are
            treated as no-change.

    Returns:
        Jaccard Index as a float in [0, 1].
    """
    true_flat = np.nan_to_num(true_mask, nan=0).astype(int).ravel()
    pred_flat = np.nan_to_num(pred_mask, nan=0).astype(int).ravel()
    return compute_jaccard(true_flat, pred_flat)


def format_evaluation_report(results: Dict, aoi_name: str = "") -> str:
    """Format an evaluation results dictionary as a human-readable text report.

    Args:
        results: Dictionary from :func:`evaluate_predictions`.
        aoi_name: Optional AOI name to include in the report header.

    Returns:
        Multi-line string suitable for writing to a text file or logging.
    """
    header = f"Model Evaluation Report"
    if aoi_name:
        header += f" — {aoi_name}"
    separator = "=" * 60

    cm = results.get("confusion_matrix", [[0, 0], [0, 0]])
    tn = cm[0][0] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    tp = cm[1][1] if len(cm) > 1 else 0

    lines = [
        separator,
        header,
        separator,
        f"Samples evaluated : {results.get('n_samples', 0):,}",
        f"True change pixels: {results.get('n_change_true', 0):,}",
        f"Pred change pixels: {results.get('n_change_pred', 0):,}",
        "",
        "Performance Metrics:",
        f"  Accuracy   : {results.get('accuracy', 0):.4f}",
        f"  Precision  : {results.get('precision', 0):.4f}",
        f"  Recall     : {results.get('recall', 0):.4f}",
        f"  F1 Score   : {results.get('f1', 0):.4f}",
        f"  Jaccard    : {results.get('jaccard', 0):.4f}  ← PS10 primary metric",
        "",
        "Confusion Matrix (rows=actual, cols=predicted):",
        f"               No-Change   Change",
        f"  No-Change  : {tn:>10,}  {fp:>6,}",
        f"  Change     : {fn:>10,}  {tp:>6,}",
        "",
        "Per-Class Report:",
        results.get("classification_report", ""),
        separator,
    ]
    return "\n".join(lines)
