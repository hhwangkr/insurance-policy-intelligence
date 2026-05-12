# Data staging (pre-ingestion)

This document describes how **manually collected** public insurance disclosure PDFs move from a **repository-local inbox** into committed, normalized storage **before** any PDF parsing or ingestion pipeline runs.

## Goals

- **Repository-local drop zone**: downloads are placed under `data/inbox/manual/` (transient; only `.gitkeep` is tracked—see root `.gitignore`), not under user-specific folders such as OS Downloads directories.
- **Deterministic storage keys**: committed PDFs under `data/raw/manual/` use normalized, hash-based filenames.
- **Preserve provenance**: `original_filename` in the manifest is the basename of the file as dropped into the inbox; optional URL and notes live in the manifest, not in the on-disk key.
- **Manifest as source of truth**: semantic fields (insurer, product, document type, dates, splits, language, etc.) live in YAML under `data/manifests/`. The storage filename is a **derived key**, not semantic truth.

## Directory layout

| Path | Purpose |
|------|---------|
| `data/inbox/manual/` | **Transient** drop zone: user copies browser downloads here, then runs the staging script. Not committed (except `.gitkeep`). |
| `data/raw/manual/` | **Committed** normalized PDFs produced by the staging script (reproducible storage keys). |
| `data/raw/crawled/` | Reserved for future crawled assets (not implemented here). |
| `data/processed/` | Reserved for future processed artifacts (not implemented here). |
| `data/manifests/` | YAML manifests. `manual.yaml` may be **generated** by `stage_manual_inbox.py`; keep `manual.example.yaml` as a hand-authored shape reference. |
| `scripts/` | Operational scripts (e.g. manual PDF staging). |

## Storage filename convention

Files under `data/raw/manual/` use:

```text
{insurer}_{product_type}_{product_slug}_{document_type}_{effective_date_compact}_{hash8}.pdf
```

Where:

- `hash8` is the first 8 hexadecimal characters of the file’s **SHA-256** over raw bytes.
- `effective_date_compact` is **`YYYYMMDD`** derived from the manifest’s ISO effective date (`YYYY-MM-DD`).
- Other segments are user-supplied labels **normalized** (lowercase, non-alphanumeric runs collapsed to a single `_`, trimmed).

The manifest’s `effective_date` field remains **ISO `YYYY-MM-DD`** for readability and downstream use; only the **storage filename** uses the compact date segment.

`product_name` is **not** part of the storage key; it appears only in the manifest (human-readable).

Manifest fields such as `insurer` and `product_name` should reflect **semantic** labels as you intend to use them downstream. The `document_id` and on-disk basename are **derived keys** built from normalized filename segments plus the content hash.

## Expected lifecycle

1. Download public disclosure PDFs from insurer or regulator disclosure rooms (browser, etc.).
2. Place the files under **`data/inbox/manual/`** (paths are **repo-relative** from the repository root).
3. Run `scripts/stage_manual_pdf.py` with `--source` pointing at the inbox file (preferred: `data/inbox/manual/<original>.pdf`) and the required semantic flags (see `--help`).
4. The script reads the PDF from the inbox path, computes SHA-256, copies bytes into **`data/raw/manual/`** with the normalized filename, and prints a **single manifest document** YAML snippet to stdout. The snippet uses **only repo-relative** `source_file` (`data/raw/manual/...`), `original_filename` (basename only), `content_hash`, and `collection_method: manual`.
5. Paste or merge the snippet into a manifest under `data/manifests/`, then fill `source_url`, `tags`, `notes`, etc. as needed.
6. Commit **`data/raw/manual/`** and the manifest. Remove the inbox copy when satisfied (inbox stays uncommitted).

## Staging workflow (short)

1. Copy PDFs into `data/inbox/manual/`.
2. From the repo root, run the staging script with a **repo-relative** `--source` (see example below).
3. Verify `data/raw/manual/` and the printed manifest fields, then delete the inbox copy if desired.

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

Absolute `--source` paths are still accepted for edge cases, but **defaults and documentation assume the inbox** so manifests and docs stay free of machine-specific layout.

Use `--dry-run` to compute the hash and print the manifest template **without** copying.

## Semi-automated rule-based inbox staging

For disclosure PDFs with **arbitrary Korean or numeric filenames**, use `scripts/stage_manual_inbox.py` to avoid hand-mapping every field before staging.

What it does (still **staging only**—no ingestion, RAG, embeddings, vectors, LLM, agents, or crawling):

1. Scans `data/inbox/manual/*.pdf`.
2. Uses **PyMuPDF** to read plain text from the **first N pages** (default **5**).
3. Applies **rule-based** heuristics (Korean keywords, filename tokens, simple date regexes) in `scripts/rule_extraction.py`.
4. Builds normalized `data/raw/manual/{...}_{hash8}.pdf` filenames via `scripts/staging_lib.py` (compact **`YYYYMMDD`** date segment; manifest keeps ISO dates).
5. Copies bytes into `data/raw/manual/` (unless `--dry-run`).
6. Writes **`data/manifests/manual.yaml`** with repo-relative `source_file`, full `content_hash`, `original_filename` basename only, and an `inference` block per field (`confidence`, `needs_review`, `evidence`).
7. Prints a **summary table** before writing so you can spot gaps quickly.

Uncertainty is explicit: fields that are weakly inferred set `needs_review: true`. If a **required** segment (insurer, document type, product type, or effective date) cannot be inferred at all, that PDF is **skipped**, a line is printed to **stderr**, and the command exits with code **3** if any file was skipped.

Commands (from repo root):

```bash
# Preview inferred rows without copying or writing manual.yaml
uv run python scripts/stage_manual_inbox.py --dry-run

# Stage all inbox PDFs and regenerate manual.yaml
uv run python scripts/stage_manual_inbox.py

# Read fewer pages (faster, less signal)
uv run python scripts/stage_manual_inbox.py --max-pages 3
```

Fully manual, single-file staging (explicit CLI metadata) remains available via `scripts/stage_manual_pdf.py`.

## Related files

- `data/manifests/manual.example.yaml` — illustrative manifest shape (hand-authored).
- `scripts/staging_lib.py` — hash, path resolution, filename helpers (unit-tested).
- `scripts/metadata_models.py` — Pydantic models for manifest rows and per-field inference.
- `scripts/rule_extraction.py` — rule-based text/filename inference (unit-tested).
- `scripts/inbox_staging.py` — orchestration between extraction, copy, and `manual.yaml`.
- `scripts/stage_manual_inbox.py` — CLI entry point for semi-automated staging.
