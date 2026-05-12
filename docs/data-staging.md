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
| `data/manifests/` | YAML manifest drafts and finalized manifests (committed). |
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

## Related files

- `data/manifests/manual.example.yaml` — illustrative manifest shape.
- `scripts/staging_lib.py` — hash, path resolution, filename helpers (unit-tested).
