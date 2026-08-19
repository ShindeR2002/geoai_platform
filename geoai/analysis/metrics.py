"""
Spatial evaluation metrics for the GeoAI Platform.

Provides metric computation functions at the analysis layer, operating on
spatial rasters and object lists. The Jaccard Index (Intersection over Union)
is the primary PS10 evaluation metric.

The pixel-level Jaccard functions in this module complement the array-level
functions in :mod:`geoai.models.evaluation`, providing spatial-first interfaces
that work directly with 2D raster arrays without requiring prior flattening.

Single responsibility: compute spatial evaluation metrics.

Position in dependency hierarchy: analysis (depends on core, utils).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary PS10 metric
# ---------------------------------------------------------------------------


def jaccard_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    change_value: int = 1,
    ignore_nan: bool = True,
) -> float:
    """Compute the Jaccard Index (IoU) for the change class.

    The Jaccard Index is the primary evaluation metric for the PS10 challenge
    (Section 8.a.2 of the official problem statement).

    Jaccard = |TP| / (|TP| + |FP| + |FN|)

    Accepts both 1D flat arrays and 2D spatial rasters. NaN values are
    excluded from the computation when ``ignore_nan`` is True.

    Args:
        y_true: Ground truth array (any shape). Values should be 0 or 1.
        y_pred: Predicted array matching y_true in shape.
        change_value: Integer value representing the change class. Default 1.
        ignore_nan: If True, positions where either array is NaN are excluded.

    Returns:
        Jaccard Index as a float in [0.0, 1.0]. Returns 0.0 if no change
        pixels exist in either array.
    """
    y_true_flat = y_true.ravel().astype(float)
    y_pred_flat = y_pred.ravel().astype(float)

    if ignore_nan:
        valid = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_flat = y_true_flat[valid]
        y_pred_flat = y_pred_flat[valid]

    true_pos = (y_true_flat == change_value) & (y_pred_flat == change_value)
    false_pos = (y_true_flat != change_value) & (y_pred_flat == change_value)
    false_neg = (y_true_flat == change_value) & (y_pred_flat != change_value)

    tp = int(true_pos.sum())
    fp = int(false_pos.sum())
    fn = int(false_neg.sum())
    union = tp + fp + fn

    if union == 0:
        logger.warning("Jaccard Index: union is zero — returning 0.0.")
        return 0.0

    jaccard = tp / union
    logger.debug(
        "Jaccard Index: TP=%d FP=%d FN=%d → %.6f.", tp, fp, fn, jaccard
    )
    return float(jaccard)


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    change_value: int = 1,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for the change class.

    Args:
        y_true: Ground truth array (any shape), values 0 or 1.
        y_pred: Predicted array matching y_true.
        change_value: Label of the positive (change) class.

    Returns:
        Tuple of (precision, recall, f1) as floats in [0, 1].
        Returns (0.0, 0.0, 0.0) if the change class is absent.
    """
    t = y_true.ravel().astype(int)
    p = y_pred.ravel().astype(int)

    tp = int(((t == change_value) & (p == change_value)).sum())
    fp = int(((t != change_value) & (p == change_value)).sum())
    fn = int(((t == change_value) & (p != change_value)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aoi_name: str = "",
) -> Dict[str, float]:
    """Compute the complete PS10 evaluation metric suite.

    Args:
        y_true: Ground truth binary array (any shape).
        y_pred: Predicted binary array matching y_true.
        aoi_name: Optional AOI name for logging.

    Returns:
        Dictionary with keys: jaccard, precision, recall, f1, accuracy.
    """
    jac = jaccard_index(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)

    t = y_true.ravel().astype(int)
    p = y_pred.ravel().astype(int)
    valid = ~(np.isnan(t.astype(float)) | np.isnan(p.astype(float)))
    acc = float((t[valid] == p[valid]).sum()) / max(valid.sum(), 1)

    metrics = {
        "jaccard": round(jac, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "accuracy": round(acc, 6),
    }

    prefix = f"[{aoi_name}] " if aoi_name else ""
    logger.info(
        "%sMetrics — Jaccard=%.4f Precision=%.4f Recall=%.4f F1=%.4f Acc=%.4f.",
        prefix, jac, prec, rec, f1, acc,
    )
    return metrics
