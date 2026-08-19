from datetime import datetime
from typing import Any, Dict, List, Optional

class Experiment:
    """Object class representing an individual research experiment run and its lifecycle state."""
    
    VALID_STATES = {"Draft", "Registered", "Running", "Completed", "Failed", "Archived"}

    def __init__(
        self,
        experiment_id: str,
        model_id: str,
        dataset_id: str,
        aoi_name: str,
        config: Dict[str, Any],
        random_seed: int = 42,
        state: str = "Draft"
    ) -> None:
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.aoi_name = aoi_name
        self.config = config
        self.random_seed = random_seed
        
        self._state = "Draft"
        self.set_state(state)
        
        self.timestamp: str = datetime.now().isoformat()
        self.metrics: Dict[str, Any] = {}
        self.deviations: List[str] = []
        self.execution_metadata: Dict[str, Any] = {}

    def get_state(self) -> str:
        return self._state

    def set_state(self, new_state: str) -> None:
        if new_state not in self.VALID_STATES:
            raise ValueError(f"Invalid experiment state: {new_state}. Valid: {self.VALID_STATES}")
        self._state = new_state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize experiment properties for JSON reporting."""
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "aoi_name": self.aoi_name,
            "state": self._state,
            "timestamp": self.timestamp,
            "random_seed": self.random_seed,
            "deviations": self.deviations,
            "metrics": self.metrics,
            "execution_metadata": self.execution_metadata,
            "config": self.config
        }
