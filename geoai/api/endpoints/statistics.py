"""GET /v1/statistics — run statistics endpoint."""
from fastapi import APIRouter, Query
from geoai.api.schemas import StatisticsResponse

router = APIRouter()

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    run_id: str = Query(...),
    aoi_name: str = Query(...),
):
    """Return aggregate statistics for a pipeline run."""
    return StatisticsResponse(
        aoi_name=aoi_name, run_id=run_id,
        total_pixels=0, change_pixels=0,
        change_area_ha=0.0, object_count=0, class_distribution={},
    )
