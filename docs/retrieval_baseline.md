# Local retrieval baseline (Phase 2F)

This document describes the **minimal local semantic retrieval** layer over section-aware chunks. It is **retrieval only**: there is no answer generation, no LLM orchestration, and no cross-reference graph.

For **PDF → JSON → sections → chunks → index → search** from a fresh clone, see the **Run end-to-end locally** section in [`README.md`](../README.md).

## What this is not

- Not full RAG (no grounded answer synthesis).
- Not hybrid BM25 + dense (BM25 / Qdrant-style services are out of scope here).
- No reranker yet.
- No cross-reference expansion (e.g. resolving inline “제7조” mentions to section IDs).
- No hosted vector database (Pinecone, Weaviate, Qdrant server, etc.).

## Prerequisites

1. Chunk JSON artifacts under `data/processed/chunks/*.chunks.json` (gitignored by default).  
2. Regenerate them any time sections change:

```bash
uv run python -m insurance_ai_ingestion.chunk_sections \
  --input-dir data/processed/sections \
  --output-dir data/processed/chunks
```

Optional quality pass:

```bash
uv run python -m insurance_ai_ingestion.inspect_chunks \
  --input-dir data/processed/chunks \
  --report-path data/processed/reports/chunk_quality.md
```

## Build the local index

Uses **sentence-transformers** with a configurable model id (default **`intfloat/multilingual-e5-small`**).  
First run downloads model weights from Hugging Face (local cache, no API key).

Embeddings are stored as a **NumPy** matrix; search uses **L2-normalized cosine similarity** via a dot product. Metadata for each row lives in `chunk_metadata.jsonl`.

```bash
uv run python -m insurance_ai_retrieval.build_index \
  --chunks-dir data/processed/chunks \
  --index-dir data/processed/index
```

Optional flags:

- `--model-name <hf_model_id>`
- `--batch-size <n>`

Outputs (all under `data/processed/index/`, gitignored except `.gitkeep`):

| File | Role |
|------|------|
| `chunk_embeddings.npy` | `float32` matrix `(num_chunks, dim)` |
| `chunk_metadata.jsonl` | One JSON object per line (citation + `text` + derived insurer/product fields) |
| `index_config.json` | Model id, dimension, counts, timestamps |

## Search (retrieval-only)

Dense similarity is computed over the **full embedding matrix**, then **metadata and section-type filters** narrow candidates **before** top‑`k` ranking. Natural-language queries that only mention an insurer/product in Korean can still retrieve **boilerplate-similar** clauses from other PDFs (for example identical “청약의 철회” articles). For **repeatable smoke tests** on a multi-insurer corpus, combine a short factual query with **explicit metadata filters** derived from `document_id` / manifest slugs.

### Default section-type filter

Unless you pass `--no-default-section-type-filter`, `search_index` **drops** chunks whose `section_type` is one of:

- `toc`
- `cover`
- `guide`
- `summary`

Use `--include-section-types article,appendix` (comma list) to **whitelist** types instead; that implies the default exclude list is **not** applied. You can still add extra removals with `--exclude-section-types …`.

### Metadata filters (`search_index`)

Optional flags (all combined with **AND** semantics):

- `--document-id`
- `--insurer` (slug parsed from `document_id`, for example `kyobolife`)
- `--product-type` (slug parsed from `document_id`, for example `annuity`)
- `--product-name` (case-insensitive substring match on the derived slug-as-spaces label)
- `--policy-unit-name`, `--variant-name` (case-insensitive substring matches on chunk fields when present)

If filters remove **every** chunk, the CLI exits with a clear error (`no chunks matched metadata and section filters`).

### Section deduplication

`--dedupe-section` returns **at most one** hit per `section_id` (the highest-scoring chunk in that section). Default is **off**; turn it on for smoke runs where long sections would otherwise fill the entire top‑`k`.

### Example: Kyobo annuity (적립형)

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

### Example: Samsung cancer

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --insurer samsunglife \
  --product-type cancer \
  --query "암 진단 보험금 지급 사유는 뭐야?" \
  --top-k 5 \
  --dedupe-section
```

### Example: Samsung whole life

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --insurer samsunglife \
  --product-type whole_life \
  --query "해약환급금은 어떻게 지급돼?" \
  --top-k 5 \
  --dedupe-section
```

### Example: Mirae Asset variable annuity

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --insurer miraeassetlife \
  --product-type variable_annuity \
  --query "특별계정 운용은 어떻게 설명돼?" \
  --top-k 5 \
  --dedupe-section
```

### Unfiltered search (exploratory)

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --query "교보생명 개인연금저축 교보로연금보험 적립형에서 보험금 지급이 늦어지면 이자는 어떻게 계산돼?" \
  --top-k 5
```

The CLI prints rank, similarity score, identifiers, **insurer / product_type / product_name** (when known), section title/type, policy-unit fields, page span, character span, and a short text preview. **Inspect** titles and metadata only; there is no “correct answer” assertion in this baseline.

### Multi-insurer corpus: scope your queries

The default manual corpus can include **multiple insurers and products** at once (for example Kyobo Life annuity, Samsung Life cancer / whole life, Mirae Asset Life variable annuity). **Generic** questions (“청약 철회는 언제까지 가능해?”, “해약환급금은 어떻게 지급돼?”) often match **valid** clauses in **more than one** policy, so top‑`k` hits may jump between documents. That is fine for **exploratory** search, but it makes **repeatable smoke tests** hard to judge.

**Prefer metadata filters plus a short factual query** when you want a stable bar for retrieval quality: repeat the same command after pipeline or model changes and check whether the intended `document_id` still dominates the hit list.

- **Generic queries** are allowed for exploratory search across the whole index.
- **Scoped natural-language queries** help, but they are **not sufficient** when many policies share the same article titles or appendix tables.
- **Expected results** should be judged using **`section_title`**, **`section_type`**, **`insurer` / `product_type` / `product_name`**, **`policy_unit_name`**, **`variant_name`**, and **page range** (`page_start`–`page_end`), not by asking the CLI for a prose “answer.”

### Smoke-style queries (insurer / product scoped)

Use the same `search_index` command with different `--query` strings (and matching `--insurer` / `--product-type` filters). Examples below align with the products in `data/manifests/manual.yaml`; adjust wording if your local manifest differs.

**Kyobo Life — 개인연금저축 교보로연금보험 적립형**

- `교보생명 개인연금저축 교보로연금보험 적립형에서 보험금 지급이 늦어지면 이자는 어떻게 계산돼?`
- `교보생명 개인연금저축 교보로연금보험 적립형에서 청약 철회는 언제까지 가능해?`
- `교보생명 개인연금저축 교보로연금보험 적립형에서 계약 전 알릴 의무를 위반하면 어떻게 돼?`
- `교보생명 개인연금저축 교보로연금보험 적립형에서 연금 지급 기준표는 어디에 있어?`

**Samsung Life — 인터넷암보험**

- `삼성생명 인터넷암보험에서 보험금 청구 절차는 어떻게 돼?`
- `삼성생명 인터넷암보험에서 암 진단 보험금 지급 사유는 뭐야?`
- `삼성생명 인터넷암보험에서 보험금을 지급하지 않는 사유는 뭐야?`

**Samsung Life — 밸런스종신보험**

- `삼성생명 밸런스종신보험에서 해약환급금은 어떻게 지급돼?`
- `삼성생명 밸런스종신보험에서 보험계약대출은 어떻게 돼?`
- `삼성생명 밸런스종신보험에서 계약의 소멸은 어떤 경우야?`

**Mirae Asset Life — 변액연금보험**

- `미래에셋생명 변액연금보험에서 특별계정 운용은 어떻게 설명돼?`
- `미래에셋생명 변액연금보험에서 연금 지급 기준은 어디에 있어?`
- `미래에셋생명 변액연금보험에서 해약환급금은 어떻게 계산돼?`

### Exploratory (generic) examples

These can return hits from **any** policy in the index; use when browsing, not when you need a fixed expectation:

- `보험금 지급이 늦어지면 이자는 어떻게 계산돼?`
- `청약 철회는 언제까지 가능해?`
- `해약환급금은 어떻게 지급돼?`

## E5-style prefixes

For intfloat E5-family models, inputs are prefixed in code (not by hand):

- Queries: `query: …`
- Passages: `passage: …`

Helpers live in `insurance_ai_retrieval.e5_text` so formatting stays in one place.

## Citation-ready metadata

Each hit is backed by a `ChunkMetadataRecord` (mirroring `DocumentChunk` fields needed for future citations):

- `chunk_id`, `section_id`, `section_title`, `section_type`, `document_id`
- `insurer`, `product_type`, `product_name` (parsed from `document_id` when the canonical filename pattern matches)
- `policy_unit_id`, `policy_unit_name`, `variant_name`
- `page_start`, `page_end`, `char_start`, `char_end`
- `text` (full chunk body in JSONL for re-display / snippets)

Older `chunk_metadata.jsonl` rows without the new fields are **enriched at search time** from `document_id`, but **rebuilding the index** after upgrading packages repopulates JSONL eagerly.

## Citation context bundle (`build_citation_context`)

The CLI `insurance_ai_retrieval.build_citation_context` runs **metadata-scoped dense retrieval** (same filter semantics as `search_index`) and prints a **JSON bundle**: user query, filter snapshot, `top_k`, `dedupe_section`, and a `citations` array with stable handles `C1`, `C2`, … plus chunk metadata and **full** `text` for each hit.

**Lifecycle:** a citation context is a **query-time** object. In production or a normal interactive flow you build it **in memory** (Python: `build_citation_context` from `insurance_ai_retrieval.citation_context`, which wraps `search_local_index`) and pass the resulting `CitationContextBundle` to the next layer. The bundle is **JSON-serializable** for debugging or fixtures, but it is **not** normally written to disk; persistence is not part of the standard pipeline like chunks or the index.

**Default:** JSON goes to **stdout** only.

**`--output-path`:** optional. Use only for **debugging**, **saved examples**, or **reproducible inspection**. Those files are **not** part of the tracked dataset; a narrow gitignore pattern covers typical names under `data/processed/reports/`.

### Example (stdout only)

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

Filter flags match **`search_index`** (`--document-id`, `--product-name`, `--policy-unit-name`, section-type overrides, etc.). There is **no LLM** call in this CLI.

### Normal query-time path (application / service code)

This is the intended integration shape (all **in memory**; no required intermediate JSON file):

1. User query + metadata filters (`SearchFilters`).
2. `search_local_index` (or `build_citation_context`, which calls it with an embedder) → **`CitationContextBundle`** in RAM.
3. `build_grounded_answer_prompt(bundle)` from `insurance_ai_generation.answer_prompt` → **`GroundedAnswerPrompt`** (system + user messages).
4. **`generate_grounded_answer`** (`insurance_ai_generation.generate_grounded_answer`) with an **`LLMProvider`** implementation: send messages, parse **`GroundedAnswer`**, run **`validate_answer_citations`** against `prompt.citation_ids`. Today only **static/mock** providers exist; vendor SDKs are added later.

Application code should call these Python functions directly; do not rely on shell round-trips through the filesystem for normal requests.

### Debug / development path (saved JSON)

For **reproducible inspection** or **manual QA**, you may optionally save a bundle with `build_citation_context --output-path …`, then run the CLI below to print a prompt JSON. That file path is **not** the normal data plane—only a convenience for humans and tests.

### Grounded answer prompt (`build_answer_prompt`, debug only)

`insurance_ai_generation.build_answer_prompt` reads a saved **`CitationContextBundle`** JSON (for example from `build_citation_context --output-path`), builds **deterministic** system/user chat messages for a future grounded answer step, and prints JSON to stdout. **No LLM** and no API clients—prompt construction only.

**Do not** treat “save context JSON → `build_answer_prompt`” as the production architecture; it mirrors the same `CitationContextBundle` → `build_grounded_answer_prompt` logic you would call in process memory in step 3 above.

Example (debug, using a saved bundle under `data/processed/reports/`):

```bash
uv run python -m insurance_ai_generation.build_answer_prompt \
  --context-path data/processed/reports/citation_context_example.json
```

### Grounded answer schema and citation validation (Phase 2H-1, no LLM)

`insurance_ai_generation.grounded_answer` defines a small **`GroundedAnswer`** model (prose + `citations_used` + `insufficient_context`) and **`validate_answer_citations`** / **`extract_citation_ids_from_text`** to detect invented `[C…]` markers, mismatches between body and `citations_used`, and empty or uncited answers when the model claims sufficient context. Use this **before or after** a future LLM call to guard rails—still **no LLM** and no provider SDKs in that module.

**Phase 2H-2 (serving contract):** `insurance_ai_generation.llm_provider` exposes **`LLMRequest`** / **`LLMResponse`** and an **`LLMProvider`** protocol; **`StaticLLMProvider`** supports tests and offline demos. **`generate_grounded_answer`** wires prompt → provider → parse (JSON-first, plaintext fallback) → citation validation. **No live HTTP calls** or third-party SDK dependencies in this phase—real providers plug in behind the same interface later.

## Known limitations

- **Embedding quality** depends on the chosen model and chunk text (Korean layout quirks, OCR noise).
- **No reranking** beyond raw cosine similarity.
- **No cross-reference expansion** across articles.
- **No generated answers**; downstream LLM use is a separate phase.
- **Windows console**: if Korean output garbles, set `PYTHONIOENCODING=utf-8` for the shell session (chunking CLIs also call a best-effort UTF‑8 stdout configure helper).

## Why retrieval before full RAG

Dense retrieval is the smallest slice that still validates end-to-end chunk quality, embedding wiring, and metadata preservation. Full RAG adds prompt design, grounding, and evaluation surface area; this phase keeps the loop tight and reproducible on a laptop.
