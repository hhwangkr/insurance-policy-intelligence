from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from insurance_ai_api.schemas import RetrievalContextRequest, RetrievalOptionsResponse
from insurance_ai_retrieval.citation_context import CitationContextBundle, build_citation_context
from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder
from insurance_ai_retrieval.index_engine import collect_index_filter_options, load_index_config

RetrievalContextRunner = Callable[[RetrievalContextRequest], CitationContextBundle]


def default_retrieval_context_runner(body: RetrievalContextRequest) -> CitationContextBundle:
    """Load index config, embed query, return citation bundle (same path as retrieval CLI)."""
    index_dir = body.resolved_index_dir()
    cfg = load_index_config(index_dir)
    embedder = LocalSentenceTransformerEmbedder(cfg.model_name)
    return build_citation_context(
        index_dir=index_dir,
        query=body.query,
        embedder=embedder,
        filters=body.filters.to_search_filters(),
        top_k=body.top_k,
        dedupe_section=body.dedupe_section,
    )


def get_retrieval_context_runner() -> RetrievalContextRunner:
    return default_retrieval_context_runner


def create_app() -> FastAPI:
    application = FastAPI(
        title="Insurance Policy Intelligence API",
        version="0.1.0",
        description=(
            "Retrieval-first MVP: returns citation-ready evidence context, "
            "not LLM-authored answers."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/retrieval/options")
    def retrieval_options(
        index_dir: Annotated[str, Query()] = "data/processed/index",
    ) -> RetrievalOptionsResponse:
        raw = index_dir.strip()
        if not raw:
            raise HTTPException(status_code=422, detail="index_dir must be non-empty")
        path = Path(raw).expanduser()
        try:
            opts = collect_index_filter_options(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"index read failed: {exc}") from exc
        return RetrievalOptionsResponse.model_validate(opts)

    @application.post("/retrieval/context")
    def retrieval_context(
        body: RetrievalContextRequest,
        runner: Annotated[
            RetrievalContextRunner,
            Depends(get_retrieval_context_runner),
        ],
    ) -> dict[str, object]:
        try:
            bundle = runner(body)
        except FileNotFoundError as exc:
            # Missing ``index_config.json`` or other index artifacts under ``index_dir``.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"index read failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="retrieval failed") from exc
        return bundle.model_dump()

    return application


app = create_app()
