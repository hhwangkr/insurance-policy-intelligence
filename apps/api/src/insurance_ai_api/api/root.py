from __future__ import annotations

from insurance_ai_api.api.health_routes import build_health_router

router = build_health_router(name_prefix="root")
