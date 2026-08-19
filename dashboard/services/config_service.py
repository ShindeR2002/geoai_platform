import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from geoai.core.config import PlatformConfig, load_platform_config


class ConfigService:
    """Loads and exposes platform configuration for the dashboard."""

    def __init__(self, configs_dir: str = "configs") -> None:
        self.configs_dir = configs_dir
        self.platform_config = load_platform_config(configs_dir)

    def get_platform_config(self) -> PlatformConfig:
        return self.platform_config

    def get_aoi_names(self) -> List[str]:
        return [aoi.name for aoi in self.platform_config.processing.aois]

    def get_aoi(self, aoi_name: str):
        return self.platform_config.get_aoi(aoi_name)

    def get_default_aoi(self) -> Optional[str]:
        default_name = self.platform_config.dashboard.default_aoi
        if default_name:
            return default_name
        aoi_names = self.get_aoi_names()
        return aoi_names[0] if aoi_names else None

    def get_export_base_dir(self) -> Path:
        return Path(self.platform_config.export.output_base_dir)

    def get_project_manifest_dir(self) -> Path:
        return Path(self.platform_config.dashboard.project_manifest_dir)

    def get_model_metadata(self) -> Dict[str, Any]:
        metadata_path = Path(self.platform_config.model.metadata_path)
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def get_feature_names(self) -> List[str]:
        return list(self.platform_config.features.feature_names)
