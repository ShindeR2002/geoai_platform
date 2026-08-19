import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import f1_score

def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform bootstrapping to estimate confidence intervals for IoU and F1.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(y_true)
    
    if n_samples == 0:
        return {
            "f1": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "iou": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        }
        
    f1_bootstraps = []
    iou_bootstraps = []
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]
        
        # F1
        f1_sample = f1_score(yt_sample, yp_sample, zero_division=0)
        f1_bootstraps.append(f1_sample)
        
        # IoU
        intersection = ((yt_sample == 1) & (yp_sample == 1)).sum()
        union = ((yt_sample == 1) | (yp_sample == 1)).sum()
        iou_sample = intersection / union if union > 0 else 0.0
        iou_bootstraps.append(iou_sample)
        
    lower_pct = (1.0 - confidence_level) / 2.0
    upper_pct = 1.0 - lower_pct
    
    f1_lower = np.percentile(f1_bootstraps, lower_pct * 100)
    f1_upper = np.percentile(f1_bootstraps, upper_pct * 100)
    
    iou_lower = np.percentile(iou_bootstraps, lower_pct * 100)
    iou_upper = np.percentile(iou_bootstraps, upper_pct * 100)
    
    return {
        "f1": {
            "mean": round(float(np.mean(f1_bootstraps)), 6),
            "ci_lower": round(float(f1_lower), 6),
            "ci_upper": round(float(f1_upper), 6)
        },
        "iou": {
            "mean": round(float(np.mean(iou_bootstraps)), 6),
            "ci_lower": round(float(iou_lower), 6),
            "ci_upper": round(float(iou_upper), 6)
        },
        "bootstraps_run": n_bootstraps,
        "confidence_level": confidence_level
    }

def run_paired_t_test(metrics_a: np.ndarray, metrics_b: np.ndarray) -> Dict[str, Any]:
    """
    Perform a paired t-test between two lists of paired metrics (e.g. scores across multiple runs).
    """
    from scipy.stats import ttest_rel
    metrics_a = np.asarray(metrics_a)
    metrics_b = np.asarray(metrics_b)
    if len(metrics_a) < 2 or len(metrics_b) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "message": "Insufficient samples for t-test", "significant": False}
    stat, pval = ttest_rel(metrics_a, metrics_b)
    return {
        "statistic": float(stat) if not np.isnan(stat) else 0.0,
        "p_value": float(pval) if not np.isnan(pval) else 1.0,
        "significant": bool(pval < 0.05) if not np.isnan(pval) else False
    }

def run_wilcoxon_test(metrics_a: np.ndarray, metrics_b: np.ndarray) -> Dict[str, Any]:
    """
    Perform a Wilcoxon signed-rank test between two lists of paired metrics.
    """
    from scipy.stats import wilcoxon
    metrics_a = np.asarray(metrics_a)
    metrics_b = np.asarray(metrics_b)
    if len(metrics_a) < 5 or len(metrics_b) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "message": "Insufficient samples for Wilcoxon test", "significant": False}
    try:
        stat, pval = wilcoxon(metrics_a, metrics_b)
        return {
            "statistic": float(stat) if not np.isnan(stat) else 0.0,
            "p_value": float(pval) if not np.isnan(pval) else 1.0,
            "significant": bool(pval < 0.05) if not np.isnan(pval) else False
        }
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "error": str(e), "significant": False}

def run_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """
    Perform McNemar's test for differences in pixel prediction agreement.
    Assesses if Model A and Model B have statistically significant differences in errors.
    """
    from scipy.stats import chi2
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    
    total_discordant = b + c
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "contingency_table": {"b": b, "c": c},
            "significant": False,
            "message": "No discordant predictions"
        }
    
    statistic = (abs(b - c) - 1.0) ** 2 / total_discordant
    p_value = chi2.sf(statistic, df=1)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "contingency_table": {"b": b, "c": c},
        "significant": bool(p_value < 0.05)
    }

def register_hypothesis_testing_hooks() -> Dict[str, str]:
    """Expose hooks indicating where statistical significance tests plug in."""
    return {
        "paired_t_test_hook": "geoai.evaluation.statistical_analysis.run_paired_t_test",
        "wilcoxon_signed_rank_hook": "geoai.evaluation.statistical_analysis.run_wilcoxon_test",
        "mcnemar_test_hook": "geoai.evaluation.statistical_analysis.run_mcnemar_test",
        "description": "Hooks for scientific significance testing between V1 and V2 models."
    }
