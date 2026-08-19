import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from geoai.experiments.comparison import compare_experiments

class BenchmarkSuite:
    """Evaluates multiple model experiment results against the locked baseline."""
    
    def __init__(self, outputs_dir: str = "outputs/experiments", baseline_id: str = "exp_rf_baseline") -> None:
        self.outputs_dir = Path(outputs_dir)
        self.baseline_id = baseline_id

    def load_run_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment execution metadata for a given experiment ID."""
        run_file = self.outputs_dir / experiment_id / "metadata.json"
        if not run_file.exists():
            return None
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def run_benchmark(self, candidate_ids: List[str]) -> Dict[str, Any]:
        """
        Compare a list of candidate experiment IDs against the baseline experiment.
        
        Returns a dict containing:
            - baseline_id: str
            - baseline_data: Dict or None
            - comparisons: Dict[str, Dict] maps candidate_id to the comparison result
        """
        baseline_data = self.load_run_data(self.baseline_id)
        results = {
            "baseline_id": self.baseline_id,
            "baseline_data": baseline_data,
            "comparisons": {}
        }
        
        if not baseline_data:
            # Baseline might not be run yet, or has a different ID.
            # We will still proceed, but comparisons will be marked as having no baseline reference.
            return results

        for cid in candidate_ids:
            cdata = self.load_run_data(cid)
            if cdata:
                results["comparisons"][cid] = compare_experiments(baseline_data, cdata)
                
        return results
