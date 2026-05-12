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
  raw/            # staged PDFs (manual/ and crawled/)
  processed/      # reserved for processed artifacts
  manifests/      # YAML manifest drafts

scripts/          # operational helpers (see docs/data-staging.md)

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