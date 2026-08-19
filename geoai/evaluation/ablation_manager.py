import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from geoai.experiments.ablation import AblationManager as ExpAblationManager
from geoai.utils.constants import CANONICAL_FEATURE_NAMES

class AblationStudyManager:
    """Manages executing predefined sweeps of ablation runs and comparing they against baseline."""
    
    def __init__(self, outputs_dir: str = "outputs/experiments", configs_dir: str = "configs") -> None:
        self.outputs_dir = Path(outputs_dir)
        self.configs_dir = Path(configs_dir)
        self.exp_ablation = ExpAblationManager(outputs_dir=str(outputs_dir), configs_dir=str(configs_dir))

    def get_feature_group_names(self, group_name: str) -> List[str]:
        """Resolve predefined scientific feature groups to absolute names in CANONICAL_FEATURE_NAMES."""
        group_name = group_name.lower()
        
        # 1. EO-only (removes SAR backscatter and delta_SAR)
        if group_name == "eo-only":
            return [name for name in CANONICAL_FEATURE_NAMES if "sar" in name.lower()]
            
        # 2. SAR-only (removes Red, Green, Blue, NDVI, NDBI, NDWI, delta_indices)
        if group_name == "sar-only":
            return [name for name in CANONICAL_FEATURE_NAMES if "sar" not in name.lower()]
            
        # 3. Indices-only (removes raw Red, Green, Blue, SAR bands)
        if group_name == "indices-only":
            return [name for name in CANONICAL_FEATURE_NAMES if not any(ind in name.lower() for ind in ("ndvi", "ndbi", "ndwi"))]
            
        # 4. Texture-only / Spectrals-only (spectrals-only removes indices)
        if group_name == "spectrals-only":
            return [name for name in CANONICAL_FEATURE_NAMES if any(ind in name.lower() for ind in ("ndvi", "ndbi", "ndwi", "sar"))]

        if group_name == "ndvi-only":
            return [name for name in CANONICAL_FEATURE_NAMES if "ndvi" not in name.lower()]
            
        return []

    def run_predefined_sweep(self, target_experiment_id: str, groups: List[str]) -> Dict[str, Any]:
        """
        Runs a predefined sweep of ablation runs on a target run.
        Groups can include: 'eo-only', 'sar-only', 'indices-only', 'spectrals-only'.
        
        Returns a dict of group: comparison_metrics.
        """
        results = {}
        for group in groups:
            features_to_remove = self.get_feature_group_names(group)
            if not features_to_remove:
                # If not predefined, treat as a single feature removal name
                features_to_remove = [group]
                
            suffix = f"ablate_{group.replace('-', '_')}"
            try:
                # Run the ablation run
                self.exp_ablation.run_ablation(
                    target_experiment_id=target_experiment_id,
                    features_to_remove=features_to_remove,
                    ablation_id=suffix
                )
                # Compute comparison
                compare = self.exp_ablation.compare_to_target(target_experiment_id, suffix)
                if compare:
                    results[group] = compare
            except Exception as e:
                # Log error and continue
                pass
                
        return results
