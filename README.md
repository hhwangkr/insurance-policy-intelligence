# Insurance Policy Intelligence

Retrieval-first tooling for **public Korean insurance policy PDFs**: ingest disclosure PDFs into structured artifacts, detect sections, chunk with policy-unit awareness, build a **local dense retrieval index**, run **metadata-scoped search**, emit **citation-ready context** bundles, and evaluate retrieval against a curated YAML benchmark.

> Applied systems engineering for insurance document intelligence—not a hosted LLM product.

---

## Why This Project Exists

Insurance policy documents are difficult to use in automated pipelines because they contain exclusion clauses, long-context dependencies, nested section hierarchies, ambiguous legal language, product-specific endorsements, and frequent revisions.

Naive chunk-and-embed workflows often fail because chunk boundaries break meaning, exclusions are inconsistently retrieved, and downstream consumers lack stable citation handles.

This repository focuses on:

- section-aware chunking and metadata preservation
- local semantic retrieval with explicit filters
- deterministic citation IDs in query-time bundles
- evaluation against curated queries

LLM answer generation is intentionally out of scope for the current MVP; the retrieval layer produces citation-ready context for future use.

---

## Architecture (MVP)

```text
Insurance PDFs
      ↓
Ingestion Pipeline
      ↓
Section detection
      ↓
Policy-unit / variant-aware chunking
      ↓
Local retrieval index (dense embeddings)
      ↓
Metadata-scoped search + citation context bundle
      ↓
Retrieval evaluation (YAML-driven)
```

---

## Repository Structure

```text
apps/
  web/            # Frontend app (scaffolding / out of MVP focus)

packages/
  api/            # FastAPI: health + citation-ready retrieval context
  ingestion/      # PDF ingestion, sections, chunking
  retrieval/      # Index, search, citation context, retrieval eval
  evaluation/     # Evaluation pipelines (reserved)
  shared/         # Shared models/types

data/
  inbox/manual/   # tracked original disclosure-room PDFs
  raw/manual/     # tracked normalized, hash-keyed PDFs
  processed/      # generated artifacts (mostly gitignored)
  manifests/      # YAML manifest — lineage
  eval/           # curated retrieval benchmark YAML

examples/
  processed_documents/   # curated sample processed JSON

scripts/          # operational helpers (see docs/data-staging.md)

docs/
  data-staging.md, retrieval_baseline.md, testing_strategy.md, overfitting_audit.md, …

infra/
  docker/         # API image build
```

---

## Tech Stack

### Backend

- Python 3.12+
- FastAPI (minimal API package)
- Pydantic v2

### Retrieval

- sentence-transformers (default embedding model)
- NumPy local index (cosine similarity)

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
- Local dense semantic retrieval over chunks, with **metadata-scoped search** (insurer, product type, optional variant)
- **Citation context** (`CitationContextBundle`) built at query time via `build_citation_context`
- **Retrieval evaluation harness** driven by [`data/eval/retrieval_queries.yaml`](data/eval/retrieval_queries.yaml) (see [`docs/testing_strategy.md`](docs/testing_strategy.md) and [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md))
- **HTTP API** (`packages/api`): `GET /health`, `POST /retrieval/context` — same citation bundle as the CLI; **not** an LLM answer endpoint

**Not yet / out of scope for this MVP**

- LLM answer synthesis, provider registries, or hosted model calls
- Frontend app workflows beyond any existing scaffolding
- Cross-reference graph across articles
- Reranking beyond raw embedding similarity
- Production vector database service (hosted Qdrant, Pinecone, etc.)

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

**I. Search (natural-language query with metadata filters)**

Pass `--insurer` and `--product-type` (and optional `--variant-name`) to **scope** hits to one product slice of the index (recommended for evaluation-aligned behavior; omitting these flags searches the entire index).

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --insurer kyobolife \
  --product-type annuity \
  --variant-name 적립형 \
  --query "보험금 지급이 늦어지면 이자는 어떻게 계산돼?" \
  --top-k 5 \
  --dedupe-section
```

More scoped smoke examples: [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md).

**J. Build citation context bundle (query-time JSON)**

`build_citation_context` runs the same retrieval as **I** but emits a **structured JSON bundle** (citations `C1`, `C2`, … with full chunk metadata and text) for downstream use. The bundle is **built per request** in memory; in a normal query path you would call the library API and pass the result downstream without treating it as a pipeline artifact.

```bash
uv run python -m insurance_ai_retrieval.build_citation_context \
  --index-dir data/processed/index \
  --insurer kyobolife \
  --product-type annuity \
  --variant-name 적립형 \
  --query "보험금 지급이 늦어지면 이자는 어떻게 계산돼?" \
  --top-k 5 \
  --dedupe-section
```

By default the JSON is printed to **stdout** only. **`--output-path`** is optional and is for **debugging**, **reproducible examples**, or **manual inspection**—saved citation-context JSON files are **not** part of the tracked dataset or the normal ingestion/index outputs. See [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md).

**K. Run retrieval evaluation (after index exists)**

```bash
uv run python -m insurance_ai_retrieval.evaluate_retrieval \
  --index-dir data/processed/index \
  --queries data/eval/retrieval_queries.yaml \
  --report-path data/processed/reports/retrieval_eval.md
```

Writes a markdown report under `data/processed/reports/` (generated locally; gitignored). See **Retrieval evaluation** below.

**Generated artifacts (not committed)**

- `data/processed/chunks/*.chunks.json` — regenerated with **F**; gitignored.
- `data/processed/index/*` — regenerated with **H**; gitignored (directory kept via `.gitkeep`).
- The first **H** run may download the default embedding model from Hugging Face into the local cache; **no API key** is required for this baseline.

---

## Retrieval evaluation

Cases live in [`data/eval/retrieval_queries.yaml`](data/eval/retrieval_queries.yaml) (**16** curated queries across the staged products). The harness checks hit@* against expected section titles and that every hit respects the query’s metadata filters.

**Latest baseline:** hit@1 **13/16**, hit@3 **16/16**, hit@5 **16/16**, **0** failed cases at *k*=5, metadata filter consistency **100%**. For scenario detail, extra CLI smoke tests, and how to interpret reports, see [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md).

---

## HTTP API (Phase 3A)

The **`packages/api`** service exposes citation-ready retrieval context over HTTP (same `build_citation_context` path as the CLI). Responses are **`CitationContextBundle` JSON**—ranked passages with `C1`, `C2`, … handles and metadata—**not** LLM-authored answers.

A simple evidence-search frontend can be added later on top of this API.

Run from the repository root (the POST handler loads the embedding model and index from disk **per request** for now; ensure step **H** has produced `data/processed/index`):

```bash
uv run uvicorn insurance_ai_api.main:app --reload
```

Example request body for `POST /retrieval/context`:

```json
{
  "query": "보험금 지급이 늦어지면 이자는 어떻게 계산돼?",
  "index_dir": "data/processed/index",
  "filters": {
    "document_id": null,
    "insurer": "kyobolife",
    "product_type": "annuity",
    "product_name": null,
    "policy_unit_name": null,
    "variant_name": "적립형",
    "include_section_types": null,
    "exclude_section_types": [],
    "use_default_section_type_excludes": true
  },
  "top_k": 5,
  "dedupe_section": true
}
```

**Errors:** invalid filters or empty candidate sets typically yield **400** with a `detail` string; a missing index directory or `index_config.json` yields **404**; unexpected failures yield **500**.

---

## Testing and evaluation strategy

- **`pytest`** (under `packages/*/tests/`) should mostly guard **reusable pipeline invariants** (schemas, filters, deterministic citation handles)—not every product-specific section title.
- **`data/eval/*.yaml`** (e.g. [`data/eval/retrieval_queries.yaml`](data/eval/retrieval_queries.yaml)) holds **curated benchmark cases** for the **current demo corpus**. Extend them when a new document is meant to join that benchmark set.
- **Adding a new PDF** normally means manifest → ingest → section/chunk/index steps → **run generic tests** → **inspect markdown reports**; it should **not** by default require editing Python tests.
- **Quality and inspection reports** (`inspect_*` CLIs, `data/processed/reports/*.md`) are the first tools for new documents; **manual review** covers semantic quality until automated judges exist.

See [`docs/testing_strategy.md`](docs/testing_strategy.md) for the full checklist and principles.

---

## Local Development

Install dependencies:

```bash
uv sync
```

Run backend (same as [HTTP API](#http-api-phase-3a)):

```bash
uv run uvicorn insurance_ai_api.main:app --reload
```

Run tests:

```bash
uv run pytest
```

See [`docs/testing_strategy.md`](docs/testing_strategy.md) for how **pytest**, **curated eval YAML**, and **inspection reports** relate.

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
| Retrieval eval queries | `data/eval/retrieval_queries.yaml` | **Tracked** curated benchmark cases for `evaluate_retrieval` (demo corpus; step **K**). |
| Testing / eval strategy | `docs/testing_strategy.md` | **Tracked** how pytest, YAML benchmarks, and reports fit together. |
| Overfitting audit | `docs/overfitting_audit.md` | **Tracked** corpus-specific strings in tests vs production vs eval. |
| Retrieval eval report | `data/processed/reports/retrieval_eval.md` | **Not tracked**; written by step **K**. |
| Optional citation-context JSON (debug) | `data/processed/reports/*citation_context*.json` | **Not tracked** if you use `--output-path` on **J**; query-time bundles are normally in-memory only. |
| Portfolio sample | `examples/processed_documents/` | **Tracked** small schema exemplar. |

**Tradeoff:** storing both inbox originals and normalized copies **duplicates bytes** in git for the
same underlying PDF content. The benefit is **clear provenance** (original filenames and disclosure
context) plus **deterministic ingestion keys** (`data/raw/manual/...`) decoupled from arbitrary
download names.

**Ingestion inputs:** the pipeline resolves PDF paths from the manifest only — `source_file` must
point under **`data/raw/manual/`** (not the inbox). See `docs/data-staging.md`.

Regenerate processed JSON, sections, chunks, index, run search, optionally emit a query-time citation bundle (**J**), and run retrieval evaluation (**K**) using the ordered commands in **Run end-to-end locally** above. Omit `--report-path` on inspect commands for console-only output; markdown reports under `data/processed/reports/` are generated locally and gitignored.

See also `docs/data-staging.md` and `examples/processed_documents/README.md`.

---

## Design Philosophy

This repository prioritizes:

- maintainability
- observability
- evaluation
- modular layout
- reproducible retrieval baselines

over:

- demo-oriented shortcuts
- prompt-only “RAG” without solid retrieval
- generic chatbot UX

---

## Disclaimer

This project uses only publicly available insurance documents and synthetic workflows for applied AI systems engineering exploration.