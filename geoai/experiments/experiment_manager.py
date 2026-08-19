import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class ExperimentManager:
    """Manages index of experiment runs, their files, configurations, and lifecycle states."""
    
    def __init__(self, outputs_dir: str = "outputs/experiments", configs_dir: str = "configs/experiments") -> None:
        self.outputs_dir = Path(outputs_dir)
        self.configs_dir = Path(configs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def list_experiments(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Scan the outputs directory and index all historical experiment execution runs."""
        experiments = []
        if not self.outputs_dir.exists():
            return []
            
        for path in self.outputs_dir.iterdir():
            if path.is_dir():
                metadata_file = path / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Respect active/archived state filters
                            state = data.get("state", "Draft")
                            if state == "Archived" and not include_archived:
                                continue
                            experiments.append(data)
                    except Exception:
                        pass
        # Sort by timestamp descending
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve execution metrics and metadata for a specific run."""
        metadata_file = self.outputs_dir / experiment_id / "metadata.json"
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def update_experiment_state(self, experiment_id: str, new_state: str) -> bool:
        """Update the lifecycle state of a target experiment run (e.g. Archived)."""
        metadata_file = self.outputs_dir / experiment_id / "metadata.json"
        if not metadata_file.exists():
            return False
            
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Enforce valid states
            from geoai.experiments.experiment import Experiment
            if new_state not in Experiment.VALID_STATES:
                raise ValueError(f"Invalid state '{new_state}'. Allowed: {Experiment.VALID_STATES}")
                
            data["state"] = new_state
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                
            # If the state is Completed, we update the leaderboard.
            # If the state is Archived/Failed/etc., we might want to remove or update leaderboard entries.
            # Let's clean up/sync leaderboard if state transitions to Archived or Failed.
            from geoai.experiments.leaderboard import Leaderboard
            lb = Leaderboard(leaderboard_dir=str(self.outputs_dir))
            entries = lb.load()
            
            if new_state in ("Archived", "Failed"):
                # Remove entry from leaderboard
                new_entries = [r for r in entries if r.get("experiment_id") != experiment_id]
                lb.save(new_entries)
            elif new_state == "Completed":
                # Re-add or verify entry is present
                # Find local leaderboard_entry.json
                entry_file = self.outputs_dir / experiment_id / "leaderboard_entry.json"
                if entry_file.exists():
                    with open(entry_file, "r", encoding="utf-8") as f:
                        entry_data = json.load(f)
                    lb.add_entry(entry_data)
                    
            return True
        except Exception:
            return False

    def list_configs(self) -> List[Path]:
        """Scan the configs folder and return all registered YAML files."""
        if not self.configs_dir.exists():
            return []
        return sorted(list(self.configs_dir.glob("*.yaml")))
