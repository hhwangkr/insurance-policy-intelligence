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
  inbox/manual/   # tracked original disclosure-room PDFs (filenames preserved; see docs/data-staging.md)
  raw/manual/     # tracked normalized, hash-keyed PDFs (ingestion reads manifest source_file here)
  processed/      # generated full JSON per document (*.json gitignored; see docs/data-staging.md)
  manifests/      # YAML manifest — lineage between originals and normalized paths

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

## Current scope

**In this repository today**

- Public insurance policy PDFs (manual staging and manifests)
- PDF ingestion to canonical document JSON
- Section detection
- Policy unit / variant assignment
- Section-aware chunking
- Local dense semantic retrieval over chunks ([`docs/retrieval_baseline.md`](docs/retrieval_baseline.md))

**Not yet**

- Answer generation or LLM-based RAG
- Frontend app workflows beyond scaffolding
- Cross-reference graph across articles
- Reranking (beyond raw embedding similarity)
- Production vector database (Qdrant server, hosted indexes)

---

## Run end-to-end locally

From the repository root, in order:

**A. Install / sync dependencies**

```bash
uv sync
```

**B. Stage or verify manual PDFs**

- **Tracked layout:** originals live under `data/inbox/manual/`; ingestion reads **only** normalized paths under `data/raw/manual/` referenced by `data/manifests/manual.yaml` (`source_file`). See [`docs/data-staging.md`](docs/data-staging.md).
- **After adding new files to** `data/inbox/manual/`, preview inferred rows (no copies, manifest not written):

```bash
uv run python scripts/stage_manual_inbox.py
```

- **Apply** staging (copy into `data/raw/manual/` and regenerate `data/manifests/manual.yaml`) when the preview looks correct:

```bash
uv run python scripts/stage_manual_inbox.py --apply
```

- **Or** verify an existing clone: confirm each `source_file` in the manifest exists on disk under `data/raw/manual/`.

**C. Ingest documents**

```bash
uv run python -m insurance_ai_ingestion.ingest_manifest \
  --manifest data/manifests/manual.yaml \
  --output-dir data/processed/documents
```

**D. Inspect ingestion quality**

```bash
uv run python -m insurance_ai_ingestion.inspect_documents \
  --input-dir data/processed/documents \
  --report-path data/processed/reports/ingestion_quality.md
```

**E. Detect sections**

```bash
uv run python -m insurance_ai_ingestion.detect_sections \
  --input-dir data/processed/documents \
  --output-dir data/processed/sections
```

**F. Chunk sections**

```bash
uv run python -m insurance_ai_ingestion.chunk_sections \
  --input-dir data/processed/sections \
  --output-dir data/processed/chunks
```

**G. Inspect chunks**

```bash
uv run python -m insurance_ai_ingestion.inspect_chunks \
  --input-dir data/processed/chunks \
  --report-path data/processed/reports/chunk_quality.md
```

**H. Build local retrieval index**

```bash
uv run python -m insurance_ai_retrieval.build_index \
  --chunks-dir data/processed/chunks \
  --index-dir data/processed/index
```

**I. Search (insurer/product-scoped smoke example)**

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --query "교보생명 개인연금저축 교보로연금보험 적립형에서 보험금 지급이 늦어지면 이자는 어떻게 계산돼?" \
  --top-k 5
```

More scoped smoke examples and evaluation notes: [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md).

**Generated artifacts (not committed)**

- `data/processed/chunks/*.chunks.json` — regenerated with **F**; gitignored.
- `data/processed/index/*` — regenerated with **H**; gitignored (directory kept via `.gitkeep`).
- The first **H** run may download the default embedding model from Hugging Face into the local cache; **no API key** is required for this baseline.

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
uv run pytest
```

---

## Data artifacts

This repo keeps a **reproducible paper trail** for public disclosure PDFs:

| Layer | Path | Role |
|-------|------|------|
| Original archive | `data/inbox/manual/*.pdf` | **Tracked** downloads using disclosure-room filenames (human-readable provenance). |
| Normalized staged PDFs | `data/raw/manual/*.pdf` | **Tracked** hash-based keys used as ingestion inputs (`manifest.source_file`). |
| Lineage + semantics | `data/manifests/manual.yaml` | **Tracked** mapping between `original_filename`, normalized `source_file`, `content_hash`, and labels. |
| Generated outputs | `data/processed/documents/*.json` | **Not tracked** (large, noisy); regenerate locally. |
| Chunk JSON (generated) | `data/processed/chunks/*.chunks.json` | **Not tracked**; regenerate with **Run end-to-end locally** (step F). |
| Local dense index | `data/processed/index/` | **Not tracked** except `.gitkeep`; regenerate with **Run end-to-end locally** (step H). |
| Portfolio sample | `examples/processed_documents/` | **Tracked** small schema exemplar. |

**Tradeoff:** storing both inbox originals and normalized copies **duplicates bytes** in git for the
same underlying PDF content. The benefit is **clear provenance** (original filenames and disclosure
context) plus **deterministic ingestion keys** (`data/raw/manual/...`) decoupled from arbitrary
download names.

**Ingestion inputs:** the pipeline resolves PDF paths from the manifest only — `source_file` must
point under **`data/raw/manual/`** (not the inbox). See `docs/data-staging.md`.

Regenerate processed JSON, sections, chunks, index, and run search using the ordered commands in **Run end-to-end locally** above. Omit `--report-path` on inspect commands for console-only output; markdown reports under `data/processed/reports/` are generated locally and gitignored.

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