from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from insurance_ai_api.config import Settings, get_settings
from insurance_ai_shared.models.health import HealthCheckResponse, HealthStatus


class HealthService:
    """Liveness checks (extend later with dependency readiness gates)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def build_live_response(self) -> HealthCheckResponse:
        return HealthCheckResponse(
            status=HealthStatus.OK,
            service=self._settings.service_name,
            version=self._settings.app_version,
        )


def get_health_service(settings: Annotated[Settings, Depends(get_settings)]) -> HealthService:
    return HealthService(settings=settings)
