from typing import List, Optional

from geoai.pipeline.stage1 import Stage1Result, run_stage1
from geoai.pipeline.stage3 import Stage3Result, run_stage3

from dashboard.services.config_service import ConfigService


class PipelineService:
    """Wrapper around GeoAI pipeline entry points for dashboard use."""

    def __init__(self, configs_dir: str = "configs") -> None:
        self.config_service = ConfigService(configs_dir)
        self.platform_config = self.config_service.get_platform_config()

    def run_stage1(
        self,
        aoi_name: str,
        run_id: Optional[str] = None,
        compute_probabilities: bool = False,
        generate_visualisations: bool = True,
        generate_ps10_submission: bool = True,
    ) -> Stage1Result:
        return run_stage1(
            platform_config=self.platform_config,
            aoi_name=aoi_name,
            run_id=run_id,
            compute_probabilities=compute_probabilities,
            generate_visualisations=generate_visualisations,
            generate_ps10_submission=generate_ps10_submission,
        )

    def run_stage3(
        self,
        project_id: Optional[str] = None,
        aoi_names: Optional[List[str]] = None,
        run_stage2_classification: bool = True,
        compute_shap: bool = False,
        continue_on_error: bool = True,
    ) -> Stage3Result:
        return run_stage3(
            platform_config=self.platform_config,
            project_id=project_id,
            aoi_names=aoi_names,
            run_stage2_classification=run_stage2_classification,
            compute_shap=compute_shap,
            continue_on_error=continue_on_error,
        )
