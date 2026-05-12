from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from insurance_ai_api.services.health import HealthService, get_health_service
from insurance_ai_shared.models.health import HealthCheckResponse


async def live_health(
    service: Annotated[HealthService, Depends(get_health_service)],
) -> HealthCheckResponse:
    return await service.build_live_response()


def build_health_router(*, name_prefix: str) -> APIRouter:
    router = APIRouter(tags=["health"])
    router.add_api_route(
        "/health",
        live_health,
        methods=["GET"],
        response_model=HealthCheckResponse,
        name=f"{name_prefix}_health_live",
    )
    return router
