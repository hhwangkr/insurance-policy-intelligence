from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from insurance_ai_generation.llm_provider import LLMProviderError, LLMRequest, LLMResponse

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL_EXAMPLE = "qwen2.5:7b"

HttpPostFn = Callable[[str, bytes, float], tuple[int, bytes]]


def _default_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _http_post_json(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    """POST JSON bytes to ``url``; return ``(status_code, response_body)``."""
    req = Request(  # noqa: S310
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status), resp.read()
    except HTTPError as exc:
        err_body = exc.read()
        raise LLMProviderError(
            f"Ollama HTTP {exc.code} from {url!r}: "
            f"{err_body.decode('utf-8', errors='replace')[:500]}",
        ) from exc
    except URLError as exc:
        raise LLMProviderError(
            f"Ollama connection failed for {url!r}: {exc.reason!r}",
        ) from exc


def _safe_raw_response(data: dict[str, Any]) -> dict[str, object]:
    """Trim Ollama JSON to a small, JSON-serializable map for ``LLMResponse.raw_response``."""
    msg = data.get("message")
    role: object = None
    if isinstance(msg, dict):
        role = msg.get("role")
    return {
        "model": data.get("model", ""),
        "done": bool(data.get("done", False)),
        "message_role": role,
    }


class OllamaProvider:
    """Local Ollama ``/api/chat`` adapter (stdlib HTTP only; no vendor SDK)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 600.0,
        http_post: HttpPostFn | None = None,
    ) -> None:
        self._model = model.strip()
        self._base_url = (base_url or _default_base_url()).rstrip("/")
        self._timeout = timeout_seconds
        self._http_post = http_post or _http_post_json

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = (request.model or self._model).strip()
        if not model:
            raise LLMProviderError(
                "Ollama is missing model name: set ``config.model`` or ``LLMRequest.model`` "
                f"(example local tag: {OLLAMA_DEFAULT_MODEL_EXAMPLE!r}).",
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {"temperature": float(request.temperature)},
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = int(request.max_tokens)

        if request.response_schema is not None:
            payload["format"] = request.response_schema
        elif request.response_format == "json":
            payload["format"] = "json"

        url = f"{self._base_url}/api/chat"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        status, raw_bytes = self._http_post(url, body, self._timeout)
        if status != 200:
            raise LLMProviderError(
                f"Ollama returned HTTP {status} for {url!r}: "
                f"{raw_bytes.decode('utf-8', errors='replace')[:500]}",
            )

        try:
            data: dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LLMProviderError("Ollama returned malformed JSON body.") from exc

        msg = data.get("message")
        if not isinstance(msg, dict):
            raise LLMProviderError("Ollama JSON missing a ``message`` object.")
        content = msg.get("content")
        if not isinstance(content, str):
            raise LLMProviderError("Ollama ``message.content`` is missing or not a string.")

        resp_model = data.get("model")
        model_name = str(resp_model) if resp_model is not None else model

        return LLMResponse(
            text=content,
            provider_name="ollama",
            model_name=model_name,
            raw_response=_safe_raw_response(data),
        )
