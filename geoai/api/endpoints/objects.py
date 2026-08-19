"""GET /v1/objects — paginated object list endpoint."""
from typing import Optional
from fastapi import APIRouter, Query
from geoai.api.schemas import ObjectsResponse

router = APIRouter()

@router.get("/objects", response_model=ObjectsResponse)
async def list_objects(
    run_id: Optional[str] = Query(None),
    aoi_name: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Return a paginated list of change objects."""
    return ObjectsResponse(total=0, page=page, page_size=page_size, objects=[])
