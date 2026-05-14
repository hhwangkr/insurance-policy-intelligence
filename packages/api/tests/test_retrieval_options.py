from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from insurance_ai_api.main import create_app
from insurance_ai_api.schemas import RetrievalOptionsResponse
from insurance_ai_retrieval.metadata import ChunkMetadataRecord
from insurance_ai_shared.models.chunk import DocumentChunk


def _chunk(**overrides: object) -> DocumentChunk:
    data: dict[str, object] = {
        "document_id": "kyobolife_annuity_x_policy_terms_20260101_ab12cd34",
        "chunk_id": "doc::chunk::0000::000",
        "section_id": "doc::sec::0000",
        "section_type": "article",
        "section_title": "제1조",
        "parent_section_id": None,
        "policy_unit_id": None,
        "policy_unit_name": None,
        "variant_name": None,
        "chunk_index": 0,
        "text": "alpha",
        "page_start": 1,
        "page_end": 1,
        "char_start": 10,
        "char_end": 15,
        "char_count": 5,
        "token_estimate": 2,
        "chunking_strategy": "section_aware_paragraph_v1",
        "source_section_char_start": 0,
        "source_section_char_end": 5,
    }
    data.update(overrides)
    return DocumentChunk.model_validate(data)


def _write_index_metadata(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        ChunkMetadataRecord.from_document_chunk(
            _chunk(
                document_id="kyobolife_annuity_x_policy_terms_20260101_ab12cd34",
                variant_name="적립형",
            ),
        ).model_dump_json(),
        ChunkMetadataRecord.from_document_chunk(
            _chunk(
                document_id="samsunglife_cancer_internet_cancer_insurance_policy_terms_20260101_39af0c18",
                chunk_id="other::chunk::0001::000",
                section_id="other::sec::0001",
                variant_name="기본형",
            ),
        ).model_dump_json(),
        ChunkMetadataRecord.from_document_chunk(
            _chunk(
                document_id="kyobolife_annuity_x_policy_terms_20260101_ab12cd34",
                chunk_id="doc::chunk::0002::000",
                section_id="doc::sec::0002",
                variant_name="적립형",
                text="dup insurer row",
            ),
        ).model_dump_json(),
    ]
    (index_dir / "chunk_metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_retrieval_options_success_sorted_unique(tmp_path: Path) -> None:
    _write_index_metadata(tmp_path)
    client = TestClient(create_app())
    response = client.get("/retrieval/options", params={"index_dir": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    RetrievalOptionsResponse.model_validate(data)
    assert data["insurers"] == ["kyobolife", "samsunglife"]
    assert set(data["product_types"]) == {"annuity", "cancer"}
    assert data["product_types"] == sorted(data["product_types"])
    assert data["variant_names"] == ["기본형", "적립형"]


def test_retrieval_options_missing_metadata_404(tmp_path: Path) -> None:
    empty = tmp_path / "empty_idx"
    empty.mkdir()
    client = TestClient(create_app())
    response = client.get("/retrieval/options", params={"index_dir": str(empty)})
    assert response.status_code == 404
    assert "missing metadata file" in response.json()["detail"]


def test_retrieval_options_empty_index_dir_422() -> None:
    client = TestClient(create_app())
    response = client.get("/retrieval/options", params={"index_dir": "   "})
    assert response.status_code == 422


def test_retrieval_options_response_model_accepts_empty_lists() -> None:
    RetrievalOptionsResponse.model_validate(
        {
            "insurers": [],
            "product_types": [],
            "product_names": [],
            "variant_names": [],
            "policy_unit_names": [],
        },
    )
