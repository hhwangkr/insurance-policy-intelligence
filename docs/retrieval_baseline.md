# Local retrieval baseline (Phase 2F)

This document describes the **minimal local semantic retrieval** layer over section-aware chunks. It is **retrieval only**: there is no answer generation, no LLM orchestration, and no cross-reference graph.

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
| `chunk_metadata.jsonl` | One JSON object per line (citation + `text`) |
| `index_config.json` | Model id, dimension, counts, timestamps |

## Search (retrieval-only)

```bash
uv run python -m insurance_ai_retrieval.search_index \
  --index-dir data/processed/index \
  --query "보험금 지급이 늦어지면 이자는 어떻게 계산돼?" \
  --top-k 5
```

The CLI prints rank, similarity score, identifiers, section title/type, product metadata, page span, character span, and a short text preview. **Inspect** titles and metadata only; there is no “correct answer” assertion in this baseline.

### Smoke-style queries (manual)

Use the same command with different `--query` strings, for example:

- `보험금 지급이 늦어지면 이자는 어떻게 계산돼?`
- `청약 철회는 언제까지 가능해?`
- `해약환급금은 어떻게 지급돼?`
- `계약 전 알릴 의무를 위반하면 어떻게 돼?`
- `연금 지급 기준표는 어디에 있어?`

## E5-style prefixes

For intfloat E5-family models, inputs are prefixed in code (not by hand):

- Queries: `query: …`
- Passages: `passage: …`

Helpers live in `insurance_ai_retrieval.e5_text` so formatting stays in one place.

## Citation-ready metadata

Each hit is backed by a `ChunkMetadataRecord` (mirroring `DocumentChunk` fields needed for future citations):

- `chunk_id`, `section_id`, `section_title`, `section_type`, `document_id`
- `policy_unit_id`, `policy_unit_name`, `variant_name`
- `page_start`, `page_end`, `char_start`, `char_end`
- `text` (full chunk body in JSONL for re-display / snippets)

## Known limitations

- **Embedding quality** depends on the chosen model and chunk text (Korean layout quirks, OCR noise).
- **No reranking** beyond raw cosine similarity.
- **No cross-reference expansion** across articles.
- **No generated answers**; downstream LLM use is a separate phase.
- **Windows console**: if Korean output garbles, set `PYTHONIOENCODING=utf-8` for the shell session.

## Why retrieval before full RAG

Dense retrieval is the smallest slice that still validates end-to-end chunk quality, embedding wiring, and metadata preservation. Full RAG adds prompt design, grounding, and evaluation surface area; this phase keeps the loop tight and reproducible on a laptop.
