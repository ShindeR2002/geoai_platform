from typing import Dict, Any, List

def compare_experiments(exp1_data: Dict[str, Any], exp2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two experiment runs and compute metric deltas.
    
    Typically, exp1_data is the baseline and exp2_data is the candidate model.
    """
    metrics1 = exp1_data.get("metrics", {})
    metrics2 = exp2_data.get("metrics", {})
    
    comparable_metrics = ["accuracy", "precision", "recall", "f1", "iou", "dice", "roc_auc", "average_precision"]
    
    deltas = {}
    for metric in comparable_metrics:
        val1 = metrics1.get(metric)
        val2 = metrics2.get(metric)
        if val1 is not None and val2 is not None:
            deltas[metric] = {
                "val1": val1,
                "val2": val2,
                "delta": round(float(val2) - float(val1), 6)
            }
            
    # System profiling comparison
    resources = ["train_time_sec", "inference_time_sec", "prediction_throughput", "model_size_mb", "memory_usage_mb"]
    resource_compare = {}
    for res in resources:
        val1 = metrics1.get(res)
        val2 = metrics2.get(res)
        if val1 is not None and val2 is not None:
            delta = float(val2) - float(val1)
            ratio = float(val2) / float(val1) if float(val1) > 0 else 0.0
            resource_compare[res] = {
                "val1": val1,
                "val2": val2,
                "delta": round(delta, 4),
                "ratio": round(ratio, 4)
            }
            
    return {
        "exp1_id": exp1_data.get("experiment_id", ""),
        "exp2_id": exp2_data.get("experiment_id", ""),
        "exp1_model": exp1_data.get("model_id", ""),
        "exp2_model": exp2_data.get("model_id", ""),
        "dataset_match": exp1_data.get("dataset_id") == exp2_data.get("dataset_id"),
        "seed_match": exp1_data.get("random_seed") == exp2_data.get("random_seed"),
        "metric_deltas": deltas,
        "resource_comparison": resource_compare
    }
