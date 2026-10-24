"""HTTP routes for the AI PDF Assistant API."""

import logging

from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.models.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Return API liveness information."""
    settings = get_settings()
    logger.debug("Health check requested (env=%s)", settings.app_env)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
    )
