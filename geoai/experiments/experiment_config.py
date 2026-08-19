import yaml
from pathlib import Path
from typing import Any, Dict

class ExperimentConfig:
    """Parser class to load, validate, and access experiment YAML configurations."""
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

    @property
    def experiment_id(self) -> str:
        return self._raw.get("experiment_id", "")

    @property
    def description(self) -> str:
        return self._raw.get("description", "")

    @property
    def model_id(self) -> str:
        return self._raw.get("model", {}).get("model_id", "")

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self._raw.get("model", {}).get("hyperparameters", {})

    @property
    def dataset_id(self) -> str:
        return self._raw.get("dataset", {}).get("dataset_id", "")

    @property
    def test_size(self) -> float:
        return self._raw.get("dataset", {}).get("test_size", 0.20)

    @property
    def random_state(self) -> int:
        return self._raw.get("dataset", {}).get("random_state", 42)

    @property
    def split_policy(self) -> str:
        return self._raw.get("dataset", {}).get("split_policy", "spatial")

    @property
    def baseline_lock(self) -> Dict[str, Any]:
        return self._raw.get("baseline_lock", {})

    @property
    def ablation(self) -> Dict[str, Any]:
        return self._raw.get("ablation", {})

    @property
    def campaign_type(self) -> str:
        return self._raw.get("campaign_type", "preprocessing")

    @property
    def track(self) -> str:
        return self._raw.get("track", "production")

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self._raw.get("preprocessing", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self._raw.get("features", {})

    def to_dict(self) -> Dict[str, Any]:
        return self._raw
