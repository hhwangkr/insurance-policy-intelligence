# Data staging (pre-ingestion)

This document describes how **manually downloaded** public insurance policy PDFs are staged into the repository **before** any PDF parsing or ingestion pipeline runs.

## Goals

- **Deterministic storage keys**: repository filenames are normalized and content-addressed (short hash suffix).
- **Preserve provenance**: the original download filename and optional URL live in the manifest, not in the on-disk key.
- **Manifest as source of truth**: semantic fields (insurer, product, document type, dates, splits, language, etc.) are recorded in YAML manifests under `data/manifests/`. The storage filename is a **derived key**, not semantic truth.

## Directory layout

| Path | Purpose |
|------|---------|
| `data/raw/manual/` | PDFs copied from manual downloads (staged by script). |
| `data/raw/crawled/` | Reserved for future crawled assets (not implemented here). |
| `data/processed/` | Reserved for future processed artifacts (not implemented here). |
| `data/manifests/` | YAML manifest drafts and finalized manifests. |
| `scripts/` | Operational scripts (e.g. manual PDF staging). |

## Storage filename convention

Files under `data/raw/manual/` use:

```text
{insurer}_{product_type}_{product_slug}_{document_type}_{effective_date}_{hash8}.pdf
```

Where:

- `hash8` is the first 8 hexadecimal characters of the file’s **SHA-256** over raw bytes.
- `effective_date` is ISO `YYYY-MM-DD` (validated by the staging script).
- Other segments are user-supplied labels **normalized** (lowercase, non-alphanumeric runs collapsed to a single `_`, trimmed).

`product_name` is **not** part of the storage key; it appears only in the manifest (human-readable).

Manifest fields such as `insurer` and `product_name` should reflect **semantic** labels as you intend to use them downstream. The `document_id` and on-disk basename are **derived keys** built from normalized filename segments plus the content hash.

## Staging workflow

1. Download a public policy PDF outside the repo (browser, etc.).
2. Run `scripts/stage_manual_pdf.py` with the source path and required semantic flags (see script `--help`).
3. The script computes SHA-256, builds the normalized filename, copies the file into `data/raw/manual/`, and prints a **single manifest document** YAML snippet to stdout for pasting into a manifest under `data/manifests/`.
4. Edit the snippet to fill `source_url`, `tags`, `notes`, or other fields as needed.

Example:

```bash
uv run python scripts/stage_manual_pdf.py \
  --source "C:\Downloads\PolicyBook.pdf" \
  --insurer "Example Mutual" \
  --product-name "Sample Auto Policy" \
  --product-type auto \
  --product-slug sample_auto \
  --document-type policy \
  --effective-date 2024-01-15 \
  --dataset-split train
```

Use `--dry-run` to compute the hash and print the manifest template **without** copying.

## Related files

- `data/manifests/manual.example.yaml` — illustrative manifest shape.
- `scripts/staging_lib.py` — hash and filename helpers (unit-tested).
