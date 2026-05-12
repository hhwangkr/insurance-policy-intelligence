from __future__ import annotations

from fastapi import APIRouter

from insurance_ai_api.api.health_routes import build_health_router

router = APIRouter()
router.include_router(build_health_router(name_prefix="v1"))
