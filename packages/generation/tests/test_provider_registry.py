from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from insurance_ai_generation.answer_prompt import (
    ChatMessage,
    GroundedAnswerPrompt,
    build_grounded_answer_prompt,
)
from insurance_ai_generation.generate_grounded_answer import generate_grounded_answer
from insurance_ai_generation.grounded_answer import (
    GroundedAnswer,
    render_citation_summary,
    render_grounded_answer_with_citations,
)
from insurance_ai_generation.provider_registry import GenerationProviderConfig, create_llm_provider
from insurance_ai_retrieval.citation_context import CitationContextBundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_create_static_provider_from_config() -> None:
    cfg = GenerationProviderConfig(
        provider_name="static",
        model="stub-model",
        static_response='{"answer":"x [C1]","citations_used":["C1"],"insufficient_context":false}',
    )
    p = create_llm_provider(cfg, citation_ids_for_static_default=["C1"])
    prompt = GroundedAnswerPrompt(
        messages=[
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ],
        query="q",
        citation_ids=["C1"],
    )
    r = generate_grounded_answer(prompt, p, model=cfg.model)
    assert r.validation.is_valid
    assert r.model_name == "stub-model"


def test_unknown_provider_raises_clear_error() -> None:
    cfg = GenerationProviderConfig(provider_name="not-a-real-provider")
    with pytest.raises(ValueError, match="supported: 'static', 'ollama'"):
        create_llm_provider(cfg)


def test_registry_static_invented_citation_fails_validation() -> None:
    cfg = GenerationProviderConfig(
        provider_name="static",
        static_response=(
            '{"answer":"bad [C9]","citations_used":["C9"],"insufficient_context":false}'
        ),
    )
    provider = create_llm_provider(cfg, citation_ids_for_static_default=["C1", "C2"])
    prompt = GroundedAnswerPrompt(
        messages=[
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ],
        query="q",
        citation_ids=["C1", "C2"],
    )
    result = generate_grounded_answer(prompt, provider)
    assert not result.validation.is_valid


def test_default_static_response_empty_citation_context() -> None:
    cfg = GenerationProviderConfig(provider_name="static")
    provider = create_llm_provider(cfg, citation_ids_for_static_default=[])
    prompt = GroundedAnswerPrompt(
        messages=[
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ],
        query="q",
        citation_ids=[],
    )
    result = generate_grounded_answer(prompt, provider)
    assert result.answer.insufficient_context is True
    assert result.validation.is_valid


def test_default_static_prefers_c1_when_present() -> None:
    cfg = GenerationProviderConfig(provider_name="static")
    provider = create_llm_provider(cfg, citation_ids_for_static_default=["C2", "C1"])
    prompt = GroundedAnswerPrompt(
        messages=[
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ],
        query="q",
        citation_ids=["C2", "C1"],
    )
    result = generate_grounded_answer(prompt, provider)
    assert result.validation.is_valid
    assert "C1" in result.answer.citations_used


def test_generate_answer_cli_with_saved_context_fixture() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    raw = json.loads(ctx.read_text(encoding="utf-8"))
    bundle = CitationContextBundle.model_validate(raw)
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "static",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["provider_name"] == "static"
    assert data["validation"]["is_valid"] is True
    assert "answer" in data
    ga = GroundedAnswer.model_validate(data["answer"])
    assert data["rendered_answer"] == render_grounded_answer_with_citations(ga)
    assert data["citation_summary"] == render_citation_summary(ga, bundle)


def test_generate_answer_cli_rendered_answer_structured_static() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    raw = json.loads(ctx.read_text(encoding="utf-8"))
    bundle = CitationContextBundle.model_validate(raw)
    payload = {
        "answer": "Structured body only.",
        "citations_used": ["C2", "C3", "C5"],
        "insufficient_context": False,
    }
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "static",
        "--static-response",
        json.dumps(payload, ensure_ascii=False),
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    ga = GroundedAnswer.model_validate(data["answer"])
    assert data["rendered_answer"] == "Structured body only. [C2][C3][C5]"
    assert data["rendered_answer"] == render_grounded_answer_with_citations(ga)
    assert data["citation_summary"] == render_citation_summary(ga, bundle)


def test_generate_answer_cli_ollama_requires_model() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "ollama",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "ollama" in proc.stderr.lower() or "model" in proc.stderr.lower()


def test_generate_answer_cli_unknown_provider_exits_nonzero() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    bad = "definitely-not-supported-yet"
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        bad,
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "unknown provider_name" in proc.stderr


def test_generate_answer_cli_static_response_override() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    payload = {
        "answer": "CLI override [C1].",
        "citations_used": ["C1"],
        "insufficient_context": False,
    }
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "static",
        "--static-response",
        json.dumps(payload, ensure_ascii=False),
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["answer"]["answer"] == payload["answer"]
    assert data["validation"]["is_valid"] is True


def test_generate_answer_cli_invalid_static_default_exits_zero() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    bad = {
        "answer": "no citation markers here",
        "citations_used": [],
        "insufficient_context": False,
    }
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "static",
        "--static-response",
        json.dumps(bad, ensure_ascii=False),
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["validation"]["is_valid"] is False


def test_generate_answer_cli_fail_on_invalid_static_exits_nonzero() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    bad = {
        "answer": "no citation markers here",
        "citations_used": [],
        "insufficient_context": False,
    }
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "static",
        "--static-response",
        json.dumps(bad, ensure_ascii=False),
        "--fail-on-invalid",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 1
    data = json.loads(proc.stdout)
    assert data["validation"]["is_valid"] is False


def test_generate_answer_cli_no_fail_on_invalid_exits_zero() -> None:
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    bad = {
        "answer": "no citation markers here",
        "citations_used": [],
        "insufficient_context": False,
    }
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.generate_answer",
        "--context-path",
        str(ctx),
        "--provider",
        "static",
        "--static-response",
        json.dumps(bad, ensure_ascii=False),
        "--no-fail-on-invalid",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_registry_modules_have_no_sdk_import_strings() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "insurance_ai_generation"
    for fname in ("provider_registry.py", "generate_answer.py", "providers/ollama_provider.py"):
        text = (root / fname).read_text(encoding="utf-8").lower()
        for token in ("openai", "anthropic", "cohere", "litellm", "google.generativeai"):
            assert token not in text, f"{token!r} found in {fname}"


def test_create_llm_provider_uses_bundle_citation_ids_for_default() -> None:
    """Regression: default static body must align with bundle citation handles."""
    repo = _repo_root()
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    raw = json.loads(ctx.read_text(encoding="utf-8"))
    bundle = CitationContextBundle.model_validate(raw)
    prompt = build_grounded_answer_prompt(bundle)
    cfg = GenerationProviderConfig(provider_name="static")
    provider = create_llm_provider(cfg, citation_ids_for_static_default=prompt.citation_ids)
    result = generate_grounded_answer(prompt, provider)
    assert result.validation.is_valid
