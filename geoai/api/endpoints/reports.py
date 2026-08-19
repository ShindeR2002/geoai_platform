"""GET /v1/reports/{run_id} — report download endpoint."""
from pathlib import Path
from fastapi import APIRouter, HTTPException, Path as PathParam
from geoai.api.schemas import ReportResponse

router = APIRouter()

@router.get("/reports/{run_id}", response_model=ReportResponse)
async def get_report(run_id: str = PathParam(...)):
    """Return the Stage 1 report for a run."""
    report_path = Path("outputs") / run_id / "reports" / "stage1_report.txt"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No report for run_id={run_id!r}")
    return ReportResponse(
        run_id=run_id, report_type="stage1",
        content=report_path.read_text(encoding="utf-8"),
    )
