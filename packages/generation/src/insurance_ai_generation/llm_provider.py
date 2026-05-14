from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from insurance_ai_generation.answer_prompt import ChatMessage


class LLMRequest(BaseModel):
    """Provider-agnostic chat completion request (no network here)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class LLMResponse(BaseModel):
    """Normalized completion from any provider implementation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    provider_name: str
    model_name: str | None = None
    raw_response: dict[str, object] | None = None


class LLMProviderError(RuntimeError):
    """Raised when a provider implementation fails to produce a response."""


@runtime_checkable
class LLMProvider(Protocol):
    """Contract for future remote LLM adapters; tests use ``StaticLLMProvider``."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return model text for the given messages (sync; streaming is out of scope)."""
        ...


class StaticLLMProvider:
    """Returns a fixed string; records the last request for tests (no I/O)."""

    def __init__(
        self,
        response_text: str,
        *,
        provider_name: str = "static",
        model_name: str | None = None,
        raw_response: dict[str, object] | None = None,
    ) -> None:
        self._response_text = response_text
        self._provider_name = provider_name
        self._model_name = model_name
        self._raw_response = raw_response
        self.last_request: LLMRequest | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            text=self._response_text,
            provider_name=self._provider_name,
            model_name=self._model_name,
            raw_response=self._raw_response,
        )
