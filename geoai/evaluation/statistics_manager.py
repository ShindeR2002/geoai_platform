import numpy as np
from typing import Dict, Any, List
import logging
from geoai.evaluation.statistical_analysis import run_mcnemar_test, run_wilcoxon_test, run_paired_t_test

logger = logging.getLogger(__name__)

def compute_paired_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute confidence intervals for the metric difference (Model A - Model B) using paired bootstrapping.
    """
    from sklearn.metrics import f1_score
    
    rng = np.random.default_rng(random_seed)
    n_samples = len(y_true)
    
    diff_f1_list = []
    diff_iou_list = []
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        yt = y_true[indices]
        yp_a = y_pred_a[indices]
        yp_b = y_pred_b[indices]
        
        # F1
        f1_a = f1_score(yt, yp_a, zero_division=0)
        f1_b = f1_score(yt, yp_b, zero_division=0)
        diff_f1_list.append(f1_a - f1_b)
        
        # IoU
        intersection_a = ((yt == 1) & (yp_a == 1)).sum()
        union_a = ((yt == 1) | (yp_a == 1)).sum()
        iou_a = intersection_a / union_a if union_a > 0 else 0.0
        
        intersection_b = ((yt == 1) & (yp_b == 1)).sum()
        union_b = ((yt == 1) | (yp_b == 1)).sum()
        iou_b = intersection_b / union_b if union_b > 0 else 0.0
        
        diff_iou_list.append(iou_a - iou_b)
        
    lower_pct = (1.0 - confidence_level) / 2.0
    upper_pct = 1.0 - lower_pct
    
    f1_lower = np.percentile(diff_f1_list, lower_pct * 100)
    f1_upper = np.percentile(diff_f1_list, upper_pct * 100)
    
    iou_lower = np.percentile(diff_iou_list, lower_pct * 100)
    iou_upper = np.percentile(diff_iou_list, upper_pct * 100)
    
    return {
        "f1_difference": {
            "mean": round(float(np.mean(diff_f1_list)), 6),
            "ci_lower": round(float(f1_lower), 6),
            "ci_upper": round(float(f1_upper), 6)
        },
        "iou_difference": {
            "mean": round(float(np.mean(diff_iou_list)), 6),
            "ci_lower": round(float(iou_lower), 6),
            "ci_upper": round(float(iou_upper), 6)
        }
    }

def run_comparative_significance_analysis(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    model_seeds_metrics: Dict[str, List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Perform Wilcoxon, McNemar, and relative t-tests for all pairwise model combinations.
    """
    models = list(model_predictions.keys())
    comparisons = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a = models[i]
            m_b = models[j]
            
            y_a = model_predictions[m_a]
            y_b = model_predictions[m_b]
            
            # 1. McNemar test on pixel disagreement
            mcnemar_res = run_mcnemar_test(y_true, y_a, y_b)
            
            # 2. Paired bootstrap
            bootstrap_res = compute_paired_bootstrap_ci(y_true, y_a, y_b)
            
            # 3. Wilcoxon signed-rank and paired t-test on seed metrics
            wilcoxon_res = {"p_value": 1.0, "significant": False, "message": "No seed metrics provided"}
            t_test_res = {"p_value": 1.0, "significant": False, "message": "No seed metrics provided"}
            
            if model_seeds_metrics and m_a in model_seeds_metrics and m_b in model_seeds_metrics:
                metrics_a = np.array(model_seeds_metrics[m_a])
                metrics_b = np.array(model_seeds_metrics[m_b])
                if len(metrics_a) == len(metrics_b):
                    wilcoxon_res = run_wilcoxon_test(metrics_a, metrics_b)
                    t_test_res = run_paired_t_test(metrics_a, metrics_b)
            
            comparisons.append({
                "model_a": m_a,
                "model_b": m_b,
                "mcnemar": mcnemar_res,
                "bootstrap": bootstrap_res,
                "wilcoxon": wilcoxon_res,
                "paired_t_test": t_test_res
            })
            
    return comparisons

def generate_significance_report(
    comparisons: List[Dict[str, Any]],
    filepath: str,
    metadata: Dict[str, Any] = None
) -> None:
    """Generate Markdown report for statistical significance analysis."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, "w") as f:
        f.write("# GeoAI Platform — Statistical Significance Report\n\n")
        
        # Log metadata
        f.write("## Metadata\n\n")
        f.write(f"- **Execution Timestamp**: {timestamp}\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        f.write("\n")
        
        f.write("## Pairwise Statistical Significance Matrix\n\n")
        f.write("| Model A | Model B | Mean F1 Diff | McNemar p-value | Wilcoxon p-value | Significant (McNemar) | Paired Bootstrap 95% CI (F1) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for comp in comparisons:
            m_a = comp["model_a"]
            m_b = comp["model_b"]
            f1_diff = comp["bootstrap"]["f1_difference"]["mean"]
            mcn_p = comp["mcnemar"]["p_value"]
            wil_p = comp["wilcoxon"].get("p_value", 1.0)
            sig_mcn = "Significant" if comp["mcnemar"]["significant"] else "Not Significant"
            
            ci_low = comp["bootstrap"]["f1_difference"]["ci_lower"]
            ci_high = comp["bootstrap"]["f1_difference"]["ci_upper"]
            ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
            
            f.write(f"| {m_a} | {m_b} | {f1_diff:+.4f} | {mcn_p:.4e} | {wil_p:.4f} | {sig_mcn} | {ci_str} |\n")
            
        f.write("\n## Statistical Interpretation\n\n")
        f.write("> [!NOTE]\n")
        f.write("> The McNemar test evaluates whether the disagreement in pixel-wise classification errors between Model A and Model B is statistically significant. A p-value < 0.05 indicates reject-null of equal error rates.\n")
