"""
REST API Pydantic schemas for the GeoAI Platform.

Defines all request and response body models for the platform API endpoints.
All endpoint handlers use these schemas for input validation and response
serialisation.

Single responsibility: define API request/response schemas.

Position in dependency hierarchy: api (depends on utils/constants only).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prediction schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for POST /v1/predict."""
    aoi_name: str = Field(..., description="AOI name matching a configured AOI.")
    run_id: Optional[str] = Field(None, description="Optional run identifier.")
    compute_probabilities: bool = Field(False, description="Compute probability map.")

class PredictResponse(BaseModel):
    """Response body for POST /v1/predict."""
    run_id: str
    aoi_name: str
    status: str
    change_pixels: int
    change_area_ha: float
    object_count: int
    output_paths: Dict[str, str]


class BatchPredictRequest(BaseModel):
    """Request body for POST /v1/batch_predict."""
    aoi_names: Optional[List[str]] = Field(None, description="AOIs to process.")
    project_id: Optional[str] = Field(None)
    run_stage2: bool = Field(True)


class BatchPredictResponse(BaseModel):
    """Response body for POST /v1/batch_predict."""
    project_id: str
    run_id: str
    aois_processed: int
    aois_failed: int
    failed_aois: List[str]
    total_change_area_ha: float
    total_objects: int


# ---------------------------------------------------------------------------
# Object schemas
# ---------------------------------------------------------------------------

class ObjectRecord(BaseModel):
    """Single change object record for API responses."""
    object_id: int
    aoi_name: str
    run_id: str
    area_px: int
    area_m2: float
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]
    semantic_class: str
    confidence: float
    ndvi_change: Optional[float]
    ndbi_change: Optional[float]


class ObjectsResponse(BaseModel):
    """Response body for GET /v1/objects."""
    total: int
    page: int
    page_size: int
    objects: List[ObjectRecord]


# ---------------------------------------------------------------------------
# Statistics schemas
# ---------------------------------------------------------------------------

class StatisticsResponse(BaseModel):
    """Response body for GET /v1/statistics."""
    aoi_name: str
    run_id: str
    total_pixels: int
    change_pixels: int
    change_area_ha: float
    object_count: int
    class_distribution: Dict[str, int]


# ---------------------------------------------------------------------------
# Report schemas
# ---------------------------------------------------------------------------

class ReportResponse(BaseModel):
    """Response body for GET /v1/reports/{run_id}."""
    run_id: str
    report_type: str
    content: str


# ---------------------------------------------------------------------------
# Error schema
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str
    detail: Optional[str] = None
    run_id: Optional[str] = None
