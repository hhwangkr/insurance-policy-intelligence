from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from insurance_ai_generation.llm_provider import LLMProvider, StaticLLMProvider
from insurance_ai_generation.providers.ollama_provider import OllamaProvider


class GenerationProviderConfig(BaseModel):
    """Serializable generation provider settings (no paid API keys in config)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    provider_name: str
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    static_response: str | None = Field(
        default=None,
        description="Body returned by static provider; default JSON is used when unset.",
    )


def _default_static_response_json(citation_ids: Sequence[str]) -> str:
    """Offline stub: prefer ``[C1]`` when allowed; else first citation id; else insufficient."""
    if not citation_ids:
        return json.dumps(
            {"answer": "", "citations_used": [], "insufficient_context": True},
            ensure_ascii=False,
        )
    preferred = "C1" if "C1" in citation_ids else citation_ids[0]
    return json.dumps(
        {
            "answer": f"Default static stub [{preferred}].",
            "citations_used": [preferred],
            "insufficient_context": False,
        },
        ensure_ascii=False,
    )


def create_llm_provider(
    config: GenerationProviderConfig,
    *,
    citation_ids_for_static_default: Sequence[str] | None = None,
) -> LLMProvider:
    """Construct a provider implementation from ``config``."""
    name = config.provider_name.strip().lower()
    if name == "static":
        if config.static_response is not None:
            body = config.static_response
        else:
            body = _default_static_response_json(list(citation_ids_for_static_default or []))

        return StaticLLMProvider(
            body,
            provider_name="static",
            model_name=config.model,
        )

    if name == "ollama":
        if not (config.model and str(config.model).strip()):
            raise ValueError(
                "provider_name='ollama' requires a non-empty --model / config.model "
                "(local tag, e.g. 'qwen2.5:7b'); refusing a silent default to avoid "
                "surprising pulls.",
            )
        return OllamaProvider(config.model.strip())

    raise ValueError(
        f"unknown provider_name {config.provider_name!r}; supported: 'static', 'ollama'",
    )
