import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class BenchmarkQASystem:
    """Evaluates campaign execution completeness, configuration consistency, and replication metadata to output QA scores."""
    
    @staticmethod
    def evaluate_campaign_qa(
        runs_list: List[Dict[str, Any]],
        expected_runs_count: int,
        feature_names: List[str],
        canonical_feature_names: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Evaluate campaign reproducibility criteria.
        Returns:
            qa_score (float): consolidated score from 0.0 to 100.0.
            errors (list): details on failed reproducibility checks.
        """
        errors = []
        
        # 1. pre-checks
        if not runs_list:
            return 0.0, ["No completed campaign runs found in SQLite database registries."]
            
        # 2. Check Preprocessing Identity Check
        # Ensure all runs use identical preprocessing configuration
        preprocessings = set(run.get("preprocessing", "") for run in runs_list)
        if len(preprocessings) > 1:
            errors.append(f"Inconsistent preprocessing detected across comparisons: {preprocessings}")
            
        # 3. Check Split Consistency Check
        # Verify seed uniformity across models
        seeds = set(run.get("seed", 0) for run in runs_list)
        if len(seeds) > 3:
            errors.append(f"Excessive random seed divergence: {seeds}. Keep seeds locked for fair testing.")
            
        # 4. Check Feature Ordering Check
        if feature_names != canonical_feature_names:
            errors.append("Feature column matrix ordering does not match canonical CANONICAL_FEATURE_NAMES specs.")
            
        # 5. Check Completeness Ratio
        completed_count = len(runs_list)
        completeness_ratio = completed_count / max(1, expected_runs_count)
        if completed_count < expected_runs_count:
            errors.append(f"Completed run count ({completed_count}) is less than planned permutations count ({expected_runs_count}).")
            
        # Compile Quality score
        has_consistent_prep = 1.0 if len(preprocessings) <= 1 else 0.0
        has_consistent_splits = 1.0 if len(seeds) <= 3 else 0.0
        has_correct_features = 1.0 if feature_names == canonical_feature_names else 0.0
        
        # Weighted Scoring formula:
        # 30% Preprocessing + 30% Split Consistency + 20% Feature Ordering + 20% Completeness Ratio
        score = float(
            30.0 * has_consistent_prep +
            30.0 * has_consistent_splits +
            20.0 * has_correct_features +
            20.0 * min(1.0, completeness_ratio)
        )
        
        return score, errors

    @staticmethod
    def calculate_research_maturity_index(
        qa_score: float,
        has_significance: bool,
        num_datasets: int,
        ece: float,
        has_explain: bool
    ) -> float:
        """
        Calculate a Research Maturity Score for an experiment (0-100).
        RMI = 20% QA + 20% Significance + 20% Dataset Diversity + 20% Calibration + 20% Explainability.
        """
        w_qa = 0.20 * qa_score
        w_sig = 20.0 if has_significance else 0.0
        w_ds = 20.0 * min(1.0, num_datasets / 5.0)
        # Low ECE yields higher RMI (ideal calibration close to 0)
        w_cal = 20.0 * max(0.0, 1.0 - ece)
        w_explain = 20.0 if has_explain else 0.0
        
        return float(w_qa + w_sig + w_ds + w_cal + w_explain)

