# Insurance Policy Intelligence

**Retrieval-first** tooling for **public Korean insurance policy PDFs**: stage PDFs, **ingest** to structured JSON, **detect sections**, **chunk** with policy metadata, **build a local dense index**, run **metadata-scoped search**, emit **`CitationContextBundle`** (citation-ready evidence), and **evaluate** retrieval from curated YAML. A small **FastAPI** service and **Vite + React** UI expose the same citation context over HTTP.

This is **evidence search infrastructure**, not a hosted LLM product and **not** an answer-generating chatbot in the current tree.

---

## What this project is

- Ingestion and structuring (PDF → document JSON → sections → chunks).
- Local semantic retrieval with explicit metadata filters.
- Deterministic citation handles (`C1`, `C2`, …) in query-time bundles.
- **HTTP API:** `GET /health`, `POST /retrieval/context` → **`CitationContextBundle` JSON** (same path as `build_citation_context`).
- **Web UI (`apps/web`):** search form + citation cards + optional raw JSON (no chat, no model calls).

## What it does not do (current MVP)

- No LLM-authored **answer synthesis**, provider registry, or agent orchestration in-repo.
- No production vector DB service, reranker beyond cosine similarity, or cross-reference graph.

---

## Architecture

```text
PDFs (manifest) → ingest → sections → chunks → local index → search / citation bundle → API → web UI
                                      ↘ retrieval eval (YAML)
```

| Area | Location |
|------|----------|
| Evidence UI | `apps/web/` |
| HTTP API | `packages/api/` |
| Retrieval + eval | `packages/retrieval/` |
| Ingestion | `packages/ingestion/` |
| Shared models | `packages/shared/` |
| Eval package (reserved) | `packages/evaluation/` |
| Staged PDFs + manifest | `data/inbox/manual/`, `data/raw/manual/`, `data/manifests/` |
| Generated artifacts | `data/processed/*` (mostly gitignored) |
| Retrieval benchmark | `data/eval/retrieval_queries.yaml` |
| Docs | `docs/` |
| Docker API image | `infra/docker/Dockerfile.api`, `infra/docker-compose.yml` |

---

## Tech stack

- **Python 3.12+**, **uv**, **FastAPI**, **Pydantic v2**
- **sentence-transformers** + **NumPy** local index
- **Node.js**: **Vite 4**, **React 18**, **TypeScript** (`apps/web`)
- **Ruff**, **mypy**, **pytest**

---

## End-to-end commands

From the **repository root** unless noted. Order matters for a cold start.

| Step | Command |
|------|---------|
| **1. Dependencies** | `uv sync` |
| **2. Stage PDFs** (if you added inbox files) | `uv run python scripts/stage_manual_inbox.py` (preview) then `… --apply` — see [`docs/data-staging.md`](docs/data-staging.md) |
| **3. Ingest** | `uv run python -m insurance_ai_ingestion.ingest_manifest --manifest data/manifests/manual.yaml --output-dir data/processed/documents` |
| **4. Inspect ingestion** | `uv run python -m insurance_ai_ingestion.inspect_documents --input-dir data/processed/documents --report-path data/processed/reports/ingestion_quality.md` |
| **5. Detect sections** | `uv run python -m insurance_ai_ingestion.detect_sections --input-dir data/processed/documents --output-dir data/processed/sections` |
| **6. Chunk** | `uv run python -m insurance_ai_ingestion.chunk_sections --input-dir data/processed/sections --output-dir data/processed/chunks` |
| **7. Inspect chunks** | `uv run python -m insurance_ai_ingestion.inspect_chunks --input-dir data/processed/chunks --report-path data/processed/reports/chunk_quality.md` |
| **8. Build index** | `uv run python -m insurance_ai_retrieval.build_index --chunks-dir data/processed/chunks --index-dir data/processed/index` |
| **9. Search** | `uv run python -m insurance_ai_retrieval.search_index --index-dir data/processed/index --insurer kyobolife --product-type annuity --variant-name 적립형 --query "보험금 지급이 늦어지면 이자는 어떻게 계산돼?" --top-k 5 --dedupe-section` |
| **10. Citation context (CLI)** | `uv run python -m insurance_ai_retrieval.build_citation_context --index-dir data/processed/index --insurer kyobolife --product-type annuity --variant-name 적립형 --query "보험금 지급이 늦어지면 이자는 어떻게 계산돼?" --top-k 5 --dedupe-section` |
| **11. Retrieval eval** | `uv run python -m insurance_ai_retrieval.evaluate_retrieval --index-dir data/processed/index --queries data/eval/retrieval_queries.yaml --report-path data/processed/reports/retrieval_eval.md` |
| **12. HTTP API** | `uv run uvicorn insurance_ai_api.main:app --host 127.0.0.1 --port 8765` |
| **13. Web UI** | `cd apps/web` then `npm ci` and `npm run dev` (see below) |

More flags and behavior: [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md). Staging details: [`docs/data-staging.md`](docs/data-staging.md).

---

## Retrieval evaluation

Cases live in [`data/eval/retrieval_queries.yaml`](data/eval/retrieval_queries.yaml). Latest baseline and interpretation: [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md).

---

## HTTP API

- **Returns:** `CitationContextBundle` — ranked evidence chunks with `citation_id`, text, section metadata, scores — **not** model-written answers.
- **CORS:** browser calls from `http://localhost:5173` and `http://127.0.0.1:5173` are allowed (Vite dev).

```bash
uv run uvicorn insurance_ai_api.main:app --reload --host 127.0.0.1 --port 8765
```

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Liveness |
| `GET` | `/retrieval/options` | Unique filter dimensions from `chunk_metadata.jsonl` as `{ "value": "<slug>", "label": "<Korean or fallback>" }` pairs (query `index_dir`, default `data/processed/index`) |
| `POST` | `/retrieval/context` | Citation-ready bundle for a query |

Example JSON body for **`POST /retrieval/context`** (typical web UI scope):

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

Invalid input → **400**; missing index → **404**; unexpected failure → **500** (`detail` in body when applicable). **Filters** (`POST /retrieval/context` body) always use **canonical slugs** (e.g. `kyobolife`, `annuity`); **display labels** (e.g. 교보생명, 연금보험) come from the retrieval metadata layer and appear in citation JSON and in **`GET /retrieval/options`** pairs for the web UI.

---

## Evidence Search (Web UI)

**`apps/web`** — **Insurance Policy Evidence Search**: **`GET /retrieval/options`** fills **insurer / product_type / variant_name** dropdowns using **Korean labels** while requests still send **canonical values**; **`POST /retrieval/context`** returns citations that include both. **Static example question chips** only fill the query field—no chat history or auto-search. **Collapsed-by-default** raw JSON. The UI **defaults to `http://127.0.0.1:8765`** (no API URL in the main sidebar). Set **`VITE_API_BASE_URL`** at dev/build time if needed (copy `apps/web/.env.example` → `apps/web/.env`); a temporary override also lives under **Advanced** in the app.

```bash
cd apps/web
npm ci
npm run dev
```

On **Windows PowerShell**, if script execution policy blocks `npm` shims, use:

```powershell
cd apps/web
npm.cmd ci
npm.cmd run dev
```

Production bundle:

```bash
cd apps/web
npm ci
npm run build
```

Start **step 12** (API) in one terminal, **step 13** (UI) in another, then open the URL Vite prints (usually `http://localhost:5173`).

---

## Documentation

- [`docs/retrieval_baseline.md`](docs/retrieval_baseline.md) — index, search, filters, citation context CLI, retrieval eval.
- [`docs/testing_strategy.md`](docs/testing_strategy.md) — pytest vs curated YAML vs inspection reports.
- [`docs/overfitting_audit.md`](docs/overfitting_audit.md) — corpus coupling inventory (includes historical generation-test rows).
- [`docs/data-staging.md`](docs/data-staging.md) — inbox / raw / manifest workflow.
- [`docs/section_detection.md`](docs/section_detection.md) — section candidate pipeline.

---

## Testing

```bash
uv run pytest
```

Strategy (pytest vs YAML vs reports): [`docs/testing_strategy.md`](docs/testing_strategy.md). Corpus coupling notes: [`docs/overfitting_audit.md`](docs/overfitting_audit.md).

---

## Data artifacts (summary)

| Path | Tracked? | Role |
|------|----------|------|
| `data/inbox/manual/`, `data/raw/manual/`, `data/manifests/manual.yaml` | Yes (where applicable) | Staged PDFs + lineage |
| `data/processed/documents/*.json`, `sections/`, `chunks/`, `index/` | No (regenerate) | Pipeline outputs |
| `data/processed/reports/*.md`, `*citation_context*.json` | No | Inspection / debug |
| `data/eval/retrieval_queries.yaml` | Yes | Retrieval benchmark |
| `examples/processed_documents/` | Yes | Small schema samples |

Regenerate processed artifacts with the **End-to-end commands** table. Ingestion reads manifest paths under **`data/raw/manual/`** only (see `docs/data-staging.md`).

---

## Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

API listens on **port 8000** inside the image (`Dockerfile.api`). Map host ports as needed.

---

## Design philosophy

Prefer maintainability, observability, evaluation, and **reproducible retrieval** over demo-only shortcuts or chat-first UX without solid chunking and citations.

---

## Disclaimer

This project uses only **public** insurance documents and local tooling for applied systems exploration.
