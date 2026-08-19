import hashlib
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def compute_array_hash(arr: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of a numpy array."""
    if arr is None:
        return "None"
    # Sort array along axes to ensure coordinate ordering is permutation-invariant
    if arr.ndim > 1:
        sorted_arr = arr[np.lexsort(arr.T)]
    else:
        sorted_arr = np.sort(arr)
    return hashlib.sha256(sorted_arr.tobytes()).hexdigest()

def run_consistency_audit(
    models_runs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Audit and compare multiple benchmark runs to verify consistency of splits and parameters.
    Expects models_runs as a list of dicts with structure:
    {
        "model_id": str,
        "train_coords": np.ndarray,
        "val_coords": np.ndarray,
        "test_coords": np.ndarray,
        "patch_size": int,
        "normalization": Dict[str, Any],
        "feature_ordering": List[str],
        "channel_ordering": List[str],
        "ignore_index": int,
        "random_state": int,
        "padding_strategy": str
    }
    """
    if not models_runs:
        return {"status": "FAIL", "message": "No runs provided for audit"}
        
    results = {}
    failures = []
    
    ref = models_runs[0]
    ref_id = ref["model_id"]
    
    # Hashes
    ref_train_hash = compute_array_hash(ref["train_coords"])
    ref_val_hash = compute_array_hash(ref["val_coords"])
    ref_test_hash = compute_array_hash(ref["test_coords"])
    
    results["reference_model"] = ref_id
    results["train_split_consistent"] = "PASS"
    results["val_split_consistent"] = "PASS"
    results["test_split_consistent"] = "PASS"
    results["patch_size_consistent"] = "PASS"
    results["normalization_consistent"] = "PASS"
    results["feature_ordering_consistent"] = "PASS"
    results["channel_ordering_consistent"] = "PASS"
    results["ignore_index_consistent"] = "PASS"
    results["seed_consistency"] = "PASS"
    results["padding_strategy_consistent"] = "PASS"
    
    for run in models_runs[1:]:
        m_id = run["model_id"]
        
        # Check splits
        tr_hash = compute_array_hash(run["train_coords"])
        if tr_hash != ref_train_hash:
            results["train_split_consistent"] = "FAIL"
            failures.append(f"Model {m_id} train split coordinate hash mismatch against {ref_id}.")
            
        va_hash = compute_array_hash(run["val_coords"])
        if va_hash != ref_val_hash:
            results["val_split_consistent"] = "FAIL"
            failures.append(f"Model {m_id} validation split coordinate hash mismatch against {ref_id}.")
            
        te_hash = compute_array_hash(run["test_coords"])
        if te_hash != ref_test_hash:
            results["test_split_consistent"] = "FAIL"
            failures.append(f"Model {m_id} test split coordinate hash mismatch against {ref_id}.")
            
        # Check patch size
        if run.get("patch_size") != ref.get("patch_size"):
            results["patch_size_consistent"] = "FAIL"
            failures.append(f"Model {m_id} patch size ({run.get('patch_size')}) differs from {ref_id} ({ref.get('patch_size')}).")
            
        # Check normalization
        if run.get("normalization") != ref.get("normalization"):
            results["normalization_consistent"] = "FAIL"
            failures.append(f"Model {m_id} normalization ({run.get('normalization')}) differs from {ref_id} ({ref.get('normalization')}).")
            
        # Check feature ordering
        if run.get("feature_ordering") != ref.get("feature_ordering"):
            results["feature_ordering_consistent"] = "FAIL"
            failures.append(f"Model {m_id} feature ordering differs from {ref_id}.")
            
        # Check channel ordering
        if run.get("channel_ordering") != ref.get("channel_ordering"):
            results["channel_ordering_consistent"] = "FAIL"
            failures.append(f"Model {m_id} channel ordering differs from {ref_id}.")
            
        # Check ignore index
        if run.get("ignore_index") != ref.get("ignore_index"):
            results["ignore_index_consistent"] = "FAIL"
            failures.append(f"Model {m_id} ignore_index differs from {ref_id}.")
            
        # Check seed
        if run.get("random_state") != ref.get("random_state"):
            results["seed_consistency"] = "FAIL"
            failures.append(f"Model {m_id} random_state differs from {ref_id}.")
            
        # Check padding strategy
        if run.get("padding_strategy") != ref.get("padding_strategy"):
            results["padding_strategy_consistent"] = "FAIL"
            failures.append(f"Model {m_id} padding_strategy differs from {ref_id}.")
            
    has_failures = len(failures) > 0
    results["audit_status"] = "FAIL" if has_failures else "PASS"
    results["failures"] = failures
    
    return results

def generate_consistency_report(audit_results: Dict[str, Any], filepath: str) -> None:
    """Generate Markdown report for the consistency audit."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, "w") as f:
        f.write("# GeoAI Platform — Benchmark Consistency Validator Report\n\n")
        f.write(f"- **Audited Timestamp**: {timestamp}\n")
        f.write(f"- **Overall Audit Status**: {audit_results.get('audit_status', 'UNKNOWN')}\n")
        f.write(f"- **Reference Architecture**: {audit_results.get('reference_model', 'UNKNOWN')}\n\n")
        
        f.write("## Parity Status matrix\n\n")
        f.write("| Verification Area | Status | Scientific Parity Asserted |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        for k, v in audit_results.items():
            if k in ("audit_status", "failures", "reference_model"):
                continue
            f.write(f"| {k.replace('_', ' ').title()} | {v} | {'Yes' if v == 'PASS' else 'No'} |\n")
            
        f.write("\n## Consistency Warnings / Failures Log\n\n")
        failures = audit_results.get("failures", [])
        if failures:
            f.write("> [!WARNING]\n")
            for fail in failures:
                f.write(f"> - {fail}\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> Scientific reproducibility check: all benchmark iterations used identical train/val/test splits, normalization matrices, and patch extraction layers.\n")
