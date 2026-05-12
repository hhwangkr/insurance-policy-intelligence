from __future__ import annotations

import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from insurance_ai_api.config import Settings

_provider: TracerProvider | None = None


def configure_tracing(settings: Settings) -> None:
    """Install a global TracerProvider; add OTLP export when configured."""

    global _provider
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return
    service_name = os.getenv("OTEL_SERVICE_NAME") or settings.service_name
    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": settings.env,
        }
    )
    _provider = TracerProvider(resource=resource)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
        _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)


def shutdown_tracing() -> None:
    """Flush and shut down the active TracerProvider if we installed one."""

    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
