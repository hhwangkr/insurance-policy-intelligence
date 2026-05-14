from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from insurance_ai_generation.answer_prompt import ChatMessage
from insurance_ai_generation.llm_provider import LLMProviderError, LLMRequest
from insurance_ai_generation.provider_registry import GenerationProviderConfig, create_llm_provider
from insurance_ai_generation.providers.ollama_provider import OllamaProvider


def test_create_llm_provider_ollama_returns_ollama_provider() -> None:
    cfg = GenerationProviderConfig(provider_name="ollama", model="qwen2.5:7b")
    p = create_llm_provider(cfg)
    assert isinstance(p, OllamaProvider)


def test_create_llm_provider_ollama_missing_model_raises() -> None:
    cfg = GenerationProviderConfig(provider_name="ollama", model=None)
    with pytest.raises(ValueError, match="requires a non-empty"):
        create_llm_provider(cfg)

    cfg2 = GenerationProviderConfig(provider_name="ollama", model="   ")
    with pytest.raises(ValueError, match="requires a non-empty"):
        create_llm_provider(cfg2)


def test_ollama_complete_sends_expected_payload() -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        captured["url"] = url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(body.decode("utf-8"))
        return (
            200,
            json.dumps(
                {
                    "model": "qwen2.5:7b",
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"answer":"ok","citations_used":[],"insufficient_context":true}'
                        ),
                    },
                    "done": True,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

    p = OllamaProvider("qwen2.5:7b", base_url="http://localhost:11434", http_post=fake_post)
    req = LLMRequest(
        messages=[
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="user"),
        ],
        model="qwen2.5:7b",
        temperature=0.2,
        max_tokens=64,
    )
    r = p.complete(req)
    assert r.text.startswith("{")
    assert r.provider_name == "ollama"
    pl = captured["payload"]
    assert isinstance(pl, dict)
    assert pl["model"] == "qwen2.5:7b"
    assert pl["stream"] is False
    assert pl["options"]["temperature"] == 0.2
    assert pl["options"]["num_predict"] == 64
    assert pl["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert "format" not in pl


def test_ollama_includes_format_when_response_schema_set() -> None:
    from insurance_ai_generation.grounded_answer import grounded_answer_json_schema

    captured: dict[str, object] = {}

    def fake_post(_u: str, body: bytes, _t: float) -> tuple[int, bytes]:
        captured["payload"] = json.loads(body.decode("utf-8"))
        return (
            200,
            json.dumps(
                {
                    "model": "m",
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"answer":"x [C1]","citations_used":["C1"],'
                            '"insufficient_context":false}'
                        ),
                    },
                    "done": True,
                },
            ).encode(),
        )

    schema = grounded_answer_json_schema()
    p = OllamaProvider("m", http_post=fake_post)
    p.complete(
        LLMRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="m",
            response_schema=schema,
        ),
    )
    pl = captured["payload"]
    assert isinstance(pl, dict)
    assert pl.get("format") == schema


def test_ollama_format_citation_aware_schema_has_enum_on_citations_used() -> None:
    from insurance_ai_generation.grounded_answer import grounded_answer_json_schema_for_citations

    captured: dict[str, object] = {}

    def fake_post(_u: str, body: bytes, _t: float) -> tuple[int, bytes]:
        captured["payload"] = json.loads(body.decode("utf-8"))
        return (
            200,
            json.dumps(
                {
                    "model": "m",
                    "message": {"role": "assistant", "content": "{}"},
                    "done": True,
                },
            ).encode(),
        )

    schema = grounded_answer_json_schema_for_citations(["C1", "C2"])
    p = OllamaProvider("m", http_post=fake_post)
    p.complete(
        LLMRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="m",
            response_schema=schema,
        ),
    )
    pl = captured["payload"]
    assert isinstance(pl, dict)
    fmt = pl.get("format")
    assert isinstance(fmt, dict)
    cu = fmt["properties"]["citations_used"]
    assert cu["items"]["enum"] == ["C1", "C2"]


def test_ollama_includes_format_json_string_when_response_format_json() -> None:
    captured: dict[str, object] = {}

    def fake_post(_u: str, body: bytes, _t: float) -> tuple[int, bytes]:
        captured["payload"] = json.loads(body.decode("utf-8"))
        return (
            200,
            json.dumps(
                {
                    "model": "m",
                    "message": {"role": "assistant", "content": "{}"},
                    "done": True,
                },
            ).encode(),
        )

    p = OllamaProvider("m", http_post=fake_post)
    p.complete(
        LLMRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="m",
            response_format="json",
        ),
    )
    pl = captured["payload"]
    assert isinstance(pl, dict)
    assert pl["format"] == "json"


def test_ollama_complete_missing_model_raises() -> None:
    def fake_post(_u: str, _b: bytes, _t: float) -> tuple[int, bytes]:
        return (200, b"{}")

    p = OllamaProvider("", http_post=fake_post)
    with pytest.raises(LLMProviderError, match="missing model"):
        p.complete(
            LLMRequest(
                messages=[ChatMessage(role="user", content="hi")],
                model=None,
                temperature=0.0,
            ),
        )


def test_ollama_non_200_raises_llm_provider_error() -> None:
    def fake_post(_u: str, _b: bytes, _t: float) -> tuple[int, bytes]:
        return (500, b"internal error")

    p = OllamaProvider("m", http_post=fake_post)
    with pytest.raises(LLMProviderError, match="HTTP 500"):
        p.complete(LLMRequest(messages=[ChatMessage(role="user", content="x")], model="m"))


def test_ollama_malformed_json_raises() -> None:
    def fake_post(_u: str, _b: bytes, _t: float) -> tuple[int, bytes]:
        return (200, b"not-json")

    p = OllamaProvider("m", http_post=fake_post)
    with pytest.raises(LLMProviderError, match="malformed JSON"):
        p.complete(LLMRequest(messages=[ChatMessage(role="user", content="x")], model="m"))


def test_ollama_missing_message_raises() -> None:
    def fake_post(_u: str, _b: bytes, _t: float) -> tuple[int, bytes]:
        return (200, json.dumps({"done": True}).encode())

    p = OllamaProvider("m", http_post=fake_post)
    with pytest.raises(LLMProviderError, match="message"):
        p.complete(LLMRequest(messages=[ChatMessage(role="user", content="x")], model="m"))


def test_ollama_urlerror_maps_to_llm_provider_error() -> None:
    from urllib.error import URLError

    with patch(
        "insurance_ai_generation.providers.ollama_provider.urlopen",
        side_effect=URLError("connection refused"),
    ):
        p = OllamaProvider("m")
        with pytest.raises(LLMProviderError, match="connection failed"):
            p.complete(LLMRequest(messages=[ChatMessage(role="user", content="x")], model="m"))


def test_ollama_module_has_no_paid_sdk_strings() -> None:
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "insurance_ai_generation"
        / "providers"
        / "ollama_provider.py"
    )
    text = path.read_text(encoding="utf-8").lower()
    for token in ("openai", "anthropic", "cohere", "litellm", "google.generativeai"):
        assert token not in text
