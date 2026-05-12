from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insurance_ai_api.config import get_settings
from insurance_ai_api.logging_config import configure_logging
from insurance_ai_api.telemetry.otel import configure_tracing, shutdown_tracing


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    configure_tracing(settings)
    yield
    shutdown_tracing()
