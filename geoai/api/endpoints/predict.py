"""POST /v1/predict — Stage 1 inference endpoint."""
import logging
from fastapi import APIRouter, HTTPException
from geoai.api.schemas import PredictRequest, PredictResponse
from geoai.core.config import load_platform_config
from geoai.core.exceptions import GeoAIPlatformError

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run Stage 1 change detection on a configured AOI."""
    try:
        from geoai.pipeline.stage1 import run_stage1
        config = load_platform_config()
        result = run_stage1(
            platform_config=config,
            aoi_name=request.aoi_name,
            run_id=request.run_id,
            compute_probabilities=request.compute_probabilities,
        )
        return PredictResponse(
            run_id=result.run_id,
            aoi_name=result.aoi_name,
            status="success",
            change_pixels=result.prediction_summary.get("change_pixels", 0),
            change_area_ha=result.prediction_summary.get("change_area_ha", 0.0),
            object_count=result.object_summary.get("object_count", 0),
            output_paths={k: str(v) for k, v in result.output_paths.items()},
        )
    except GeoAIPlatformError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
