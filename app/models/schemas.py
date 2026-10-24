"""API request and response models."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness / readiness payload for GET /health."""

    status: str = Field(description="Service health status")
    app: str = Field(description="Application name")
    version: str = Field(description="Application version")
