# Data staging (pre-ingestion)

This document describes how **manually collected** public insurance disclosure PDFs are archived,
normalized, and described in YAML **before** optional PDF-to-JSON ingestion.

## Goals

- **Original source archive**: downloads live under `data/inbox/manual/` using **disclosure-room
  filenames** as downloaded. These PDFs are **tracked in git** as the human-readable provenance
  layer (not a transient ignored drop zone).
- **Normalized staged dataset**: committed PDFs under `data/raw/manual/` use deterministic,
  hash-based filenames and are the **only** PDF paths referenced by ingestion (`manifest.source_file`).
- **Preserve lineage in the manifest**: `original_filename` records the inbox basename; `source_file`
  points at `data/raw/manual/...`; optional URL and notes live in the manifest.
- **Manifest as source of truth**: semantic fields (insurer, product, document type, dates,
  splits, language, etc.) live in YAML under `data/manifests/`. The normalized storage filename is a
  **derived key**, not semantic truth.

## Provenance vs storage tradeoff

Keeping **both** `data/inbox/manual/` (original names) and `data/raw/manual/` (normalized copies)
means **duplicate PDF bytes** in version control for the same underlying file. This project accepts
that cost to prioritize:

- **Auditability** — reviewers see filenames as published by the insurer or regulator.
- **Reproducibility** — ingestion always reads stable keys under `data/raw/manual/` via the manifest.

If minimizing repo size matters more than retaining originals in-tree, you could rely on manifest
metadata and normalized PDFs only; the default policy here is **track both**.

## Directory layout

| Path | Purpose |
|------|---------|
| `data/inbox/manual/` | **Tracked** original-source PDFs as downloaded from public disclosure rooms (filenames preserved). |
| `data/raw/manual/` | **Tracked** normalized, hash-keyed PDFs; **`source_file` in the manifest points here** for ingestion. |
| `data/raw/crawled/` | Reserved for future crawled assets (not implemented here). |
| `data/processed/documents/` | **Generated** full canonical JSON per document (`*.json` gitignored; folder may use `.gitkeep`). |
| `data/manifests/` | YAML manifests — lineage and semantics. `manual.yaml` may be **generated** by `stage_manual_inbox.py`; keep `manual.example.yaml` as a shape reference. |
| `examples/processed_documents/` | **Tracked** small sample processed JSON (portfolio / schema reference; not a full corpus). |
| `scripts/` | Operational scripts (e.g. manual PDF staging). |

## Manifest fields (lineage and ingestion)

Each document row in `data/manifests/manual.yaml` should preserve at least:

- **`original_filename`** — basename of the file under `data/inbox/manual/` (or equivalent original name).
- **`source_file`** — repo-relative path under **`data/raw/manual/`** to the normalized PDF bytes used for hash validation and ingestion.
- **`content_hash`** — SHA-256 (hex) of the **normalized** file at `source_file` (must match on disk).
- **`collection_method`** — how the PDF entered the repo (e.g. `manual`, `manual_inbox_rules`).
- **`dataset_split`** — intended split label for experiments (e.g. `train`, `unassigned`).

Downstream **ingestion does not read the inbox path**; it resolves `source_file` under `data/raw/manual/` only.

## Storage filename convention

Files under `data/raw/manual/` use:

```text
{insurer}_{product_type}_{product_slug}_{document_type}_{effective_date_compact}_{hash8}.pdf
```

Where:

- `hash8` is the first 8 hexadecimal characters of the file's **SHA-256** over raw bytes.
- `effective_date_compact` is **`YYYYMMDD`**, derived from the manifest's ISO effective date
  (`YYYY-MM-DD`).
- Other segments are user-supplied labels **normalized** (lowercase, non-alphanumeric runs
  collapsed to a single `_`, trimmed).

The manifest's `effective_date` field remains **ISO `YYYY-MM-DD`** for readability and downstream
use; only the **storage filename** uses the compact date segment.

`product_name` is **not** part of the storage key; it appears only in the manifest
(human-readable).

Manifest fields such as `insurer` and `product_name` should reflect **semantic** labels as you
intend to use them downstream. The `document_id` and on-disk basename are **derived keys** built
from normalized filename segments plus the content hash.

## Expected lifecycle

1. Download public disclosure PDFs from insurer or regulator disclosure rooms (browser, etc.).
2. Place the files under **`data/inbox/manual/`** (paths are **repo-relative** from the repository
   root).
3. Run `scripts/stage_manual_pdf.py` with `--source` pointing at the inbox file (preferred:
   `data/inbox/manual/<original>.pdf`) and the required semantic flags (see `--help`).
4. The script reads the PDF from the inbox path, computes SHA-256, copies bytes into
   **`data/raw/manual/`** with the normalized filename, and prints a **single manifest document**
   YAML snippet to stdout. The snippet uses **only repo-relative** `source_file`
   (`data/raw/manual/...`), `original_filename` (basename only), `content_hash`, and
   `collection_method` / `dataset_split` as provided on the CLI.
5. Paste or merge the snippet into a manifest under `data/manifests/`, then fill `source_url`,
   `tags`, `notes`, etc. as needed.
6. Commit **`data/inbox/manual/`** (originals), **`data/raw/manual/`** (normalized copies), and
   **`data/manifests/manual.yaml`** so provenance and ingestion inputs stay reproducible.

## Staging workflow (short)

1. Copy PDFs into `data/inbox/manual/`.
2. From the repo root, run the staging script with a **repo-relative** `--source` (see example
   below).
3. Verify `data/raw/manual/` and the printed manifest fields, then commit inbox originals,
   normalized PDFs, and the manifest together when satisfied.

Example (run from repository root):

```bash
uv run python scripts/stage_manual_pdf.py \
  --source data/inbox/manual/PolicyBook.pdf \
  --insurer "Example Mutual" \
  --product-name "Sample Auto Policy" \
  --product-type auto \
  --product-slug sample_auto \
  --document-type policy \
  --effective-date 2024-01-15 \
  --dataset-split train
```

Absolute `--source` paths are still accepted for edge cases, but **defaults and documentation assume
the inbox** so manifests and docs stay free of machine-specific layout.

Use `--dry-run` to compute the hash and print the manifest template **without** copying.

## Semi-automated rule-based inbox staging

For disclosure PDFs with **arbitrary Korean or numeric filenames**, use `scripts/stage_manual_inbox.py`
to avoid hand-mapping every field before staging.

What it does (still **staging only**—no ingestion, RAG, embeddings, vectors, LLM, agents, or
crawling):

1. Scans `data/inbox/manual/*.pdf`.
2. Uses **PyMuPDF** to read plain text from the **first N pages** (default **5**).
3. Applies **rule-based** heuristics (Korean keywords, filename tokens, simple date regexes) in
   `scripts/rule_extraction.py`.
4. Builds normalized `data/raw/manual/{...}_{hash8}.pdf` filenames via `scripts/staging_lib.py`
   (compact **`YYYYMMDD`** date segment; manifest keeps ISO dates).
5. Copies bytes into `data/raw/manual/` only when **`--apply`** is passed (default is preview-only).
6. Writes **`data/manifests/manual.yaml`** only when **`--apply`** is passed, with repo-relative
   `source_file`, full `content_hash`, `original_filename` basename only, and an `inference` block
   per field (`confidence`, `needs_review`, `evidence`).
7. Prints a **per-document YAML-style preview** (no fixed-width table) so you can spot gaps quickly;
   long Hangul filenames are shown using Python `repr(...)` to avoid fragile terminal column
   alignment.

Uncertainty is explicit: fields that are weakly inferred set `needs_review: true`. If a
**required** segment (insurer, document type, product type, or effective date) cannot be inferred
at all, that PDF is **skipped**, a line is printed to **stderr**, and the command exits with code
**3** if any file was skipped.

Commands (from repo root):

```bash
# Preview inferred rows (default): no files copied and manual.yaml not written
uv run python scripts/stage_manual_inbox.py

# Same as default (explicit)
uv run python scripts/stage_manual_inbox.py --dry-run

# After verifying preview output, copy PDFs and regenerate manual.yaml
uv run python scripts/stage_manual_inbox.py --apply

# Read fewer pages (faster, less signal)
uv run python scripts/stage_manual_inbox.py --max-pages 3
```

Fully manual, single-file staging (explicit CLI metadata) remains available via
`scripts/stage_manual_pdf.py`.

## Processed JSON (ingestion)

Committed **source data** for reproducibility:

- `data/inbox/manual/*.pdf` — original disclosure filenames (archive).
- `data/raw/manual/*.pdf` — normalized bytes referenced by `manifest.source_file`.
- `data/manifests/manual.yaml` — lineage (`original_filename`, `source_file`, `content_hash`,
  `collection_method`, `dataset_split`, etc.).

**Ingestion** reads PDFs only via **`source_file` under `data/raw/manual/`** (plus hash checks), not
via inbox paths.

**Processed outputs** under `data/processed/documents/` are **generated locally** (page-level text and
canonical `Document` JSON). **`*.json` files there are gitignored** so large artifacts do not churn
in pull requests. The directory remains in the tree via `data/processed/documents/.gitkeep`.

For a **small tracked reference** of the same schema (first few pages of one manifest PDF, fixed
`created_at`), see `examples/processed_documents/sample_document.json` and
`examples/processed_documents/README.md`.

Regenerate **full** outputs for every manifest row anytime:

```bash
uv run python -m insurance_ai_ingestion.ingest_manifest \
  --manifest data/manifests/manual.yaml \
  --output-dir data/processed/documents
```

## Related files

- `examples/processed_documents/sample_document.json` — curated ingestion output sample (schema reference).
- `data/manifests/manual.example.yaml` — illustrative manifest shape (hand-authored).
- `scripts/staging_lib.py` — hash, path resolution, filename helpers (unit-tested).
- `scripts/metadata_models.py` — Pydantic models for manifest rows and per-field inference.
- `scripts/rule_extraction.py` — rule-based text/filename inference (unit-tested).
- `scripts/inbox_staging.py` — orchestration between extraction, copy, and `manual.yaml`.
- `scripts/stage_manual_inbox.py` — CLI entry point for semi-automated staging.
