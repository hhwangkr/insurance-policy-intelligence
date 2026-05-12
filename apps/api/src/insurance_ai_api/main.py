from __future__ import annotations

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from insurance_ai_api.api.root import router as root_router
from insurance_ai_api.api.v1.router import router as v1_router
from insurance_ai_api.config import get_settings
from insurance_ai_api.lifespan import lifespan


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Insurance AI API",
        version=settings.app_version,
        lifespan=lifespan,
    )
    application.include_router(root_router)
    application.include_router(v1_router, prefix="/v1")
    FastAPIInstrumentor.instrument_app(application)
    return application


app = create_app()
