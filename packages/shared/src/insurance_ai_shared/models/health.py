from enum import StrEnum

from pydantic import BaseModel, Field


class HealthStatus(StrEnum):
    """Coarse process health for orchestrator probes."""

    OK = "ok"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    """Typed health payload returned by HTTP health endpoints."""

    status: HealthStatus = Field(description="Liveness or composite health state.")
    service: str = Field(description="Logical service identifier.")
    version: str = Field(description="Application version string.")
