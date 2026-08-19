"""POST /v1/batch_predict — batch inference endpoint."""
import logging
from fastapi import APIRouter, HTTPException
from geoai.api.schemas import BatchPredictRequest, BatchPredictResponse
from geoai.core.config import load_platform_config
from geoai.core.exceptions import GeoAIPlatformError

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Run Stage 1 + Stage 2 batch pipeline on multiple AOIs."""
    try:
        from geoai.pipeline.stage3 import run_stage3
        config = load_platform_config()
        result = run_stage3(
            platform_config=config,
            project_id=request.project_id,
            aoi_names=request.aoi_names,
            run_stage2_classification=request.run_stage2,
        )
        return BatchPredictResponse(
            project_id=result.project_id,
            run_id=result.run_id,
            aois_processed=len(result.aoi_results),
            aois_failed=len(result.failed_aois),
            failed_aois=result.failed_aois,
            total_change_area_ha=result.project_summary.get("total_change_area_ha", 0.0),
            total_objects=result.project_summary.get("total_significant_objects", 0),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
