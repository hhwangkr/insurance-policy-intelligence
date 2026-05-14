from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from insurance_ai_api.main import create_app, get_retrieval_context_runner
from insurance_ai_api.schemas import RetrievalContextRequest
from insurance_ai_retrieval.citation_context import (
    CitationContextBundle,
    search_filters_to_mapping,
)
from insurance_ai_retrieval.index_engine import SearchFilters


def _sample_request_payload() -> dict[str, Any]:
    return {
        "query": "보험금 지급이 늦어지면 이자는 어떻게 계산돼?",
        "index_dir": "data/processed/index",
        "filters": {
            "document_id": None,
            "insurer": "kyobolife",
            "product_type": "annuity",
            "product_name": None,
            "policy_unit_name": None,
            "variant_name": "적립형",
            "include_section_types": None,
            "exclude_section_types": [],
            "use_default_section_type_excludes": True,
        },
        "top_k": 5,
        "dedupe_section": True,
    }


def _sample_bundle() -> CitationContextBundle:
    return CitationContextBundle(
        query="보험금 지급이 늦어지면 이자는 어떻게 계산돼?",
        filters=search_filters_to_mapping(
            SearchFilters(insurer="kyobolife", product_type="annuity", variant_name="적립형")
        ),
        top_k=5,
        dedupe_section=True,
        citations=[],
    )


def test_retrieval_context_request_schema_validates() -> None:
    with pytest.raises(ValidationError):
        RetrievalContextRequest(
            query="   ",
            index_dir="data/processed/index",
        )
    with pytest.raises(ValidationError):
        RetrievalContextRequest(query="x", index_dir="")


def test_retrieval_context_success_shape() -> None:
    def fake_runner(_: RetrievalContextRequest) -> CitationContextBundle:
        return _sample_bundle()

    application = create_app()
    application.dependency_overrides[get_retrieval_context_runner] = lambda: fake_runner
    try:
        client = TestClient(application)
        response = client.post("/retrieval/context", json=_sample_request_payload())
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == _sample_request_payload()["query"]
    assert data["top_k"] == 5
    assert data["dedupe_section"] is True
    assert data["filters"]["insurer"] == "kyobolife"
    assert data["filters"]["product_type"] == "annuity"
    assert data["filters"]["variant_name"] == "적립형"
    assert data["citations"] == []


def test_retrieval_context_validation_422_empty_query() -> None:
    client = TestClient(create_app())
    payload = _sample_request_payload()
    payload["query"] = "   "
    response = client.post("/retrieval/context", json=payload)
    assert response.status_code == 422


def test_retrieval_context_validation_422_top_k() -> None:
    client = TestClient(create_app())
    payload = _sample_request_payload()
    payload["top_k"] = 0
    response = client.post("/retrieval/context", json=payload)
    assert response.status_code == 422


def test_retrieval_context_value_error_maps_to_400() -> None:
    def bad_runner(_: RetrievalContextRequest) -> CitationContextBundle:
        raise ValueError("no chunks matched metadata and section filters")

    application = create_app()
    application.dependency_overrides[get_retrieval_context_runner] = lambda: bad_runner
    try:
        client = TestClient(application)
        response = client.post("/retrieval/context", json=_sample_request_payload())
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 400
    assert "no chunks matched" in response.json()["detail"]


def test_retrieval_context_unexpected_error_maps_to_500() -> None:
    def broken_runner(_: RetrievalContextRequest) -> CitationContextBundle:
        raise RuntimeError("simulated embedder failure")

    application = create_app()
    application.dependency_overrides[get_retrieval_context_runner] = lambda: broken_runner
    try:
        client = TestClient(application)
        response = client.post("/retrieval/context", json=_sample_request_payload())
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["detail"] == "retrieval failed"
