from __future__ import annotations

import json
from pathlib import Path

from insurance_ai_generation.answer_prompt import ChatMessage, GroundedAnswerPrompt
from insurance_ai_generation.generate_grounded_answer import (
    generate_grounded_answer,
    parse_provider_text_to_grounded_answer,
)
from insurance_ai_generation.grounded_answer import grounded_answer_json_schema
from insurance_ai_generation.llm_provider import LLMRequest, LLMResponse, StaticLLMProvider


def _prompt(*, citation_ids: list[str]) -> GroundedAnswerPrompt:
    return GroundedAnswerPrompt(
        messages=[
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="user body"),
        ],
        query="test-query",
        citation_ids=citation_ids,
    )


def test_static_llm_provider_returns_response() -> None:
    p = StaticLLMProvider('{"answer": "ok [C1]", "citations_used": ["C1"]}', provider_name="test")
    req_messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="u"),
    ]
    r = p.complete(
        LLMRequest(messages=req_messages, model="m", temperature=0.1, max_tokens=100),
    )
    assert isinstance(r, LLMResponse)
    assert r.text.startswith("{")
    assert r.provider_name == "test"
    assert p.last_request is not None
    assert p.last_request.model == "m"


def test_generate_grounded_answer_json_with_c1_validates() -> None:
    payload = {
        "answer": "환급은 [C1]에 따릅니다.",
        "citations_used": ["C1"],
        "insufficient_context": False,
    }
    provider = StaticLLMProvider(json.dumps(payload, ensure_ascii=False))
    prompt = _prompt(citation_ids=["C1"])
    result = generate_grounded_answer(prompt, provider)
    assert result.answer.answer == payload["answer"]
    assert result.validation.is_valid
    assert result.provider_name == "static"
    assert result.raw_text == json.dumps(payload, ensure_ascii=False)


def test_generate_grounded_answer_invented_c9_fails_validation() -> None:
    payload = {
        "answer": "잘못된 인용 [C9]입니다.",
        "citations_used": ["C9"],
        "insufficient_context": False,
    }
    provider = StaticLLMProvider(json.dumps(payload, ensure_ascii=False))
    prompt = _prompt(citation_ids=["C1", "C2"])
    result = generate_grounded_answer(prompt, provider)
    assert not result.validation.is_valid
    assert any("not in allowed set" in e for e in result.validation.errors)


def test_parse_plain_text_fallback() -> None:
    raw = "설명입니다 [C1] 및 [C2] 참고."
    ga = parse_provider_text_to_grounded_answer(raw)
    assert ga.answer == raw
    assert set(ga.citations_used) == {"C1", "C2"}
    assert ga.insufficient_context is False


def test_generate_plain_text_provider_output() -> None:
    raw = "한글 답변 [C1]."
    provider = StaticLLMProvider(raw)
    prompt = _prompt(citation_ids=["C1"])
    result = generate_grounded_answer(prompt, provider)
    assert result.answer.answer == raw
    assert result.answer.citations_used == ["C1"]
    assert result.validation.is_valid


def test_generate_insufficient_context_json_without_citations() -> None:
    payload = {
        "answer": "",
        "citations_used": [],
        "insufficient_context": True,
    }
    provider = StaticLLMProvider(json.dumps(payload))
    prompt = _prompt(citation_ids=["C1", "C2"])
    result = generate_grounded_answer(prompt, provider)
    assert result.answer.insufficient_context is True
    assert result.validation.is_valid


def test_generate_grounded_answer_passes_grounded_answer_json_schema() -> None:
    provider = StaticLLMProvider("{}")
    prompt = _prompt(citation_ids=["C1"])
    generate_grounded_answer(prompt, provider)
    assert provider.last_request is not None
    assert provider.last_request.response_schema == grounded_answer_json_schema()


def test_provider_receives_prompt_messages() -> None:
    prompt = _prompt(citation_ids=["C1"])
    provider = StaticLLMProvider("{}")
    generate_grounded_answer(prompt, provider)
    assert provider.last_request is not None
    assert provider.last_request.messages == prompt.messages


def test_generation_llm_modules_have_no_sdk_import_strings() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "insurance_ai_generation"
    for fname in ("llm_provider.py", "generate_grounded_answer.py"):
        text = (root / fname).read_text(encoding="utf-8").lower()
        for token in ("openai", "anthropic", "cohere", "litellm", "google.generativeai"):
            assert token not in text, f"{token!r} found in {fname}"
