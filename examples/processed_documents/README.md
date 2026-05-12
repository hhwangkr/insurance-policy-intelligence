# Processed document JSON (sample)

This folder holds a **small, curated** processed document JSON file that demonstrates the canonical
ingestion output shape (`Document` from `insurance_ai_shared.models.document`).

- **`sample_document.json`** — First three pages of text from the manifest PDF with the **fewest
  total pages** among entries in `data/manifests/manual.yaml`, using the same extraction path as
  full ingestion (PyMuPDF). `created_at` is fixed for a stable portfolio artifact.

Full outputs for every manifest row are **not** committed (see root `.gitignore` under
`data/processed/documents/`). Regenerate them locally:

```bash
uv run python -m insurance_ai_ingestion.ingest_manifest \
  --manifest data/manifests/manual.yaml \
  --output-dir data/processed/documents
```

Committed inputs remain **`data/raw/manual/*.pdf`** and **`data/manifests/manual.yaml`**.
