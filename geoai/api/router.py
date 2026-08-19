"""
REST API router and middleware for the GeoAI Platform.

Defines all API routes and connects them to endpoint handlers. Implements
CORS middleware, structured request logging, and consistent error formatting.

The API provides endpoints for:
  POST /v1/predict          — Stage 1 inference on a configured AOI
  POST /v1/batch_predict    — Batch Stage 1+2 on multiple AOIs
  POST /v1/classify         — Stage 2 classification on a run's objects
  GET  /v1/projects         — List all projects
  GET  /v1/projects/{id}/runs — Run history for a project
  GET  /v1/objects          — Paginated object list
  GET  /v1/statistics       — Aggregate statistics for a run
  GET  /v1/reports/{run_id} — Download a run report

Single responsibility: route API requests to handlers.

Position in dependency hierarchy: api (depends on api/schemas, api/endpoints,
pipeline, core/config).
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from geoai.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(cors_origins: Optional[list] = None) -> FastAPI:
    """Create and configure the GeoAI Platform FastAPI application.

    Args:
        cors_origins: List of allowed CORS origins. Defaults to ['*'].

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="GeoAI Platform API",
        description=(
            "REST API for the GeoAI Platform — "
            "PS10 Anthropogenic Change Detection."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware.
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware.
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(
            "API %s %s → %d (%.1f ms)",
            request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response

    # Global exception handler.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled API error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # Register routes.
    _register_routes(app)

    logger.info("GeoAI Platform API application created.")
    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    """Register all API endpoint routes on the application.

    Args:
        app: FastAPI application instance.
    """
    from geoai.api.endpoints import batch, objects, predict, reports, statistics

    app.include_router(predict.router, prefix="/v1", tags=["Prediction"])
    app.include_router(batch.router, prefix="/v1", tags=["Batch"])
    app.include_router(objects.router, prefix="/v1", tags=["Objects"])
    app.include_router(statistics.router, prefix="/v1", tags=["Statistics"])
    app.include_router(reports.router, prefix="/v1", tags=["Reports"])

    @app.get("/health", tags=["Health"])
    async def health_check():
        """API health check endpoint."""
        return {"status": "ok", "service": "GeoAI Platform API"}
