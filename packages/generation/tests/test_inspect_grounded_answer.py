from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from insurance_ai_generation.generate_grounded_answer import GroundedAnswerGenerationResult
from insurance_ai_generation.grounded_answer import GroundedAnswer, GroundedAnswerValidation
from insurance_ai_generation.inspect_grounded_answer import (
    build_grounded_answer_inspection,
    generation_result_from_generate_answer_json,
    inspection_report_to_markdown,
)
from insurance_ai_retrieval.citation_context import (
    CitationContextBundle,
    CitationContextEntry,
)


def _entry(
    *,
    citation_id: str,
    section_title: str,
    rank: int,
    text: str,
) -> CitationContextEntry:
    return CitationContextEntry(
        citation_id=citation_id,
        chunk_id=f"ck::{citation_id}",
        section_id=f"sec::{citation_id}",
        section_title=section_title,
        section_type="article",
        document_id="doc",
        page_start=rank,
        page_end=rank + 1,
        char_start=0,
        char_end=100,
        score=1.0 - rank * 0.01,
        text=text,
    )


def _sample_bundle() -> CitationContextBundle:
    return CitationContextBundle(
        query="보험금 지급이 늦어지면 이자는 어떻게 계산돼?",
        filters={"insurer": "kyobolife"},
        top_k=5,
        dedupe_section=True,
        citations=[
            _entry(
                citation_id="C1",
                section_title="( 별표 3 )",
                rank=1,
                text="별표3 이자표 " + ("x" * 800),
            ),
            _entry(
                citation_id="C2",
                section_title="제22조",
                rank=2,
                text="제22조 본문",
            ),
            _entry(
                citation_id="C3",
                section_title="제7조",
                rank=3,
                text="제7조 본문",
            ),
            _entry(
                citation_id="C4",
                section_title="제34조",
                rank=4,
                text="제34조 본문",
            ),
            _entry(
                citation_id="C5",
                section_title="제2조",
                rank=5,
                text="제2조 정의",
            ),
        ],
    )


def _sample_result() -> GroundedAnswerGenerationResult:
    return GroundedAnswerGenerationResult(
        answer=GroundedAnswer(
            answer="지연 이자에 대한 설명.",
            citations_used=["C2", "C3", "C5"],
            insufficient_context=False,
        ),
        validation=GroundedAnswerValidation(
            is_valid=True,
            errors=[],
            citation_ids_in_text=[],
            allowed_citation_ids=["C1", "C2", "C3", "C4", "C5"],
        ),
        provider_name="ollama",
        model_name="exaone3.5:7.8b",
        raw_text='{"answer":"…"}',
    )


def test_inspection_includes_rendered_answer_and_citation_summary() -> None:
    bundle = _sample_bundle()
    result = _sample_result()
    r = build_grounded_answer_inspection(bundle, result)
    assert r.rendered_answer.endswith("[C2][C3][C5]")
    assert "제22조" in r.citation_summary
    assert "제7조" in r.citation_summary
    assert "제2조" in r.citation_summary
    assert r.citations_used == ["C2", "C3", "C5"]


def test_inspection_cited_and_uncited_split() -> None:
    bundle = _sample_bundle()
    result = _sample_result()
    r = build_grounded_answer_inspection(bundle, result)
    cited_ids = {c.citation_id for c in r.cited_contexts}
    uncited_ids = {c.citation_id for c in r.uncited_contexts}
    assert cited_ids == {"C2", "C3", "C5"}
    assert uncited_ids == {"C1", "C4"}
    assert len(r.all_citation_contexts) == 5
    assert r.all_citation_contexts[0].rank == 1
    assert r.all_citation_contexts[0].citation_id == "C1"


def test_c1_in_uncited_when_only_c2_c3_c5_used() -> None:
    bundle = _sample_bundle()
    result = _sample_result()
    r = build_grounded_answer_inspection(bundle, result)
    uncited = [c.citation_id for c in r.uncited_contexts]
    assert "C1" in uncited


def test_text_preview_truncates_long_chunk() -> None:
    bundle = _sample_bundle()
    result = _sample_result()
    r = build_grounded_answer_inspection(bundle, result)
    c1 = next(c for c in r.all_citation_contexts if c.citation_id == "C1")
    assert len(c1.text_preview) <= 500
    assert c1.text_preview.endswith("…")


def test_generation_result_from_generate_answer_json_ignores_extra_keys() -> None:
    payload = {
        "answer": {
            "answer": "x",
            "citations_used": ["C1"],
            "insufficient_context": False,
        },
        "validation": {
            "is_valid": True,
            "errors": [],
            "citation_ids_in_text": [],
            "allowed_citation_ids": ["C1"],
        },
        "provider_name": "static",
        "model_name": None,
        "raw_text": "{}",
        "rendered_answer": "should be ignored",
        "citation_summary": "ignored",
    }
    gr = generation_result_from_generate_answer_json(payload)
    assert gr.provider_name == "static"
    assert gr.answer.citations_used == ["C1"]


def test_generation_result_from_generate_answer_json_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        generation_result_from_generate_answer_json({"answer": {}})


def test_inspection_markdown_writes_roundtrip(tmp_path: Path) -> None:
    bundle = _sample_bundle()
    result = _sample_result()
    r = build_grounded_answer_inspection(bundle, result)
    md = inspection_report_to_markdown(r)
    out = tmp_path / "rep.md"
    out.write_text(md, encoding="utf-8")
    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    assert "## Uncited contexts" in body
    assert "**C1**" in body
    assert "## Cited contexts" in body
    assert "**C2**" in body


def test_inspect_answer_cli_stdin_and_report(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    ctx = repo / "data/processed/reports/citation_context_example.json"
    if not ctx.is_file():
        pytest.skip("citation_context_example.json not in workspace")
    gen = {
        "answer": {
            "answer": "stub",
            "citations_used": ["C2", "C3", "C5"],
            "insufficient_context": False,
        },
        "validation": {
            "is_valid": True,
            "errors": [],
            "citation_ids_in_text": [],
            "allowed_citation_ids": ["C1", "C2", "C3", "C4", "C5"],
        },
        "provider_name": "static",
        "model_name": None,
        "raw_text": "{}",
        "rendered_answer": "extra",
        "citation_summary": "extra",
    }
    report_md = tmp_path / "grounded_answer_inspection.md"
    cmd = [
        sys.executable,
        "-m",
        "insurance_ai_generation.inspect_answer",
        "--context-path",
        str(ctx),
        "--generation-result-path",
        "-",
        "--report-path",
        str(report_md),
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=json.dumps(gen, ensure_ascii=False),
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["validation_is_valid"] is True
    assert "C1" in {c["citation_id"] for c in data["uncited_contexts"]}
    assert report_md.read_text(encoding="utf-8").startswith("# Grounded answer inspection")


def test_inspect_grounded_answer_module_has_no_vendor_sdk_strings() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "insurance_ai_generation"
        / "inspect_grounded_answer.py"
    )
    src = path.read_text(encoding="utf-8").lower()
    for token in ("openai", "anthropic", "cohere", "litellm"):
        assert token not in src
