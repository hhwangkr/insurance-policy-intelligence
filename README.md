# Insurance Policy Intelligence

Production-grade AI platform for insurance document understanding, retrieval, and citation-grounded reasoning.

> Exploring enterprise AI architecture for insurance policy intelligence.

---

## Why This Project Exists

Insurance policy documents are difficult for LLM systems because they contain:

- exclusion clauses
- long-context dependencies
- nested section hierarchies
- ambiguous legal language
- product-specific endorsements
- frequent policy revisions

Naive RAG pipelines often fail in insurance workflows because:

- chunk boundaries break semantic meaning
- exclusions are inconsistently retrieved
- citations become unreliable
- generated answers are difficult to audit

This repository explores how to build a more reliable insurance AI architecture using:

- hybrid retrieval
- semantic chunking
- citation grounding
- evaluation-first design
- agentic workflows

---

## Architecture

```text
Insurance PDFs
      ↓
Ingestion Pipeline
      ↓
Semantic Chunking
      ↓
Hybrid Retrieval
(BM25 + Vector)
      ↓
Citation-Grounded Reasoning
      ↓
Agent Workflows
```

---

## Repository Structure

```text
apps/
  api/            # FastAPI backend
  web/            # Frontend app

packages/
  agents/         # LangGraph workflows
  ingestion/      # PDF ingestion
  retrieval/      # Search/retrieval
  evaluation/     # Evaluation pipelines
  shared/         # Shared models/types

data/
  inbox/manual/   # transient PDF drop zone (see docs/data-staging.md)
  raw/            # committed normalized PDFs (manual/ and crawled/)
  processed/      # generated full JSON per document (gitignored; see docs/data-staging.md)
  manifests/      # YAML manifest drafts

examples/
  processed_documents/   # curated sample processed JSON (schema reference)

scripts/          # operational helpers (see docs/data-staging.md)
                  #   stage_manual_pdf.py   — single-file manual staging (explicit CLI)
                  #   stage_manual_inbox.py — rule-based inbox batch + manual.yaml

docs/
  data-staging.md   # Manual PDF staging and manifest conventions
infra/
```

---

## Tech Stack

### Backend

- Python
- FastAPI
- Pydantic v2

### AI

- LangGraph
- Hybrid Retrieval
- Qdrant

### Tooling

- uv
- Ruff
- mypy
- pytest
- Docker

---

## Current Status

### Completed

- Monorepo foundation
- FastAPI scaffold
- Shared package structure
- Tooling/linting setup
- OpenTelemetry foundation

### In Progress

- PDF ingestion pipeline
- Metadata extraction
- Semantic chunking

### Planned

- Hybrid retrieval
- Citation grounding
- Evaluation pipelines
- Agent workflows
- MCP integration

---

## Local Development

Install dependencies:

```bash
uv sync
```

Run backend:

```bash
uv run uvicorn insurance_ai_api.main:app --reload
```

Run tests:

```bash
pytest
```

---

## Data artifacts

Reproducibility and review-friendly diffs:

- **Tracked:** `data/raw/manual/*.pdf` (normalized PDFs) and `data/manifests/manual.yaml`
  (metadata and content-hash pointers).
- **Generated locally, not tracked:** full processed JSON files under `data/processed/documents/`
  (`*.json` is gitignored; the directory may contain a `.gitkeep`).
- **Tracked schema sample:** `examples/processed_documents/sample_document.json` — small excerpt of
  real ingestion output (first pages only) for portfolio and contract reference.

Regenerate full processed JSON locally:

```bash
uv run python -m insurance_ai_ingestion.ingest_manifest \
  --manifest data/manifests/manual.yaml \
  --output-dir data/processed/documents
```

See also `docs/data-staging.md` and `examples/processed_documents/README.md`.

---

## Design Philosophy

This repository prioritizes:

- maintainability
- observability
- evaluation
- modular AI architecture
- production readiness

over:

- demo-oriented shortcuts
- prompt-only implementations
- generic chatbot UX

---

## Disclaimer

This project uses only publicly available insurance documents and synthetic workflows for applied AI systems engineering exploration.