# Testing and evaluation strategy

This note separates **reusable pipeline checks** from **curated benchmark fixtures** and **ad-hoc quality tooling**, so new PDFs do not imply rewriting pytest by default.

## Philosophy

- **pytest** — protect **product/document-agnostic invariants**: schemas, filter semantics, deterministic citation IDs in bundles, chunk/index wiring. Avoid encoding every insurer-specific section title in code.
- **`data/eval/*.yaml`** — **curated benchmark cases** tied to the **current demo corpus** (expected section titles, filters, queries). Extend this when a document is meant to join the benchmark set.
- **Quality / inspection reports** — first-line tools for **newly added** documents (`inspect_*` CLIs, markdown under `data/processed/reports/`, optional saved citation-context JSON). Gitignored artifacts; not the contract for every future policy.
- **Manual review** — semantic retrieval quality and clause relevance until stronger automated metrics exist.

Answer synthesis and its evaluation are **out of scope** for the current MVP; focus tests and YAML on ingestion and retrieval.

## Unit tests (packages/*/tests)

Prefer **structural invariants**, for example:

- non-empty extracted text where required
- stable `document_id` pattern and parsed metadata when filenames match the canonical pattern
- valid page / character ranges
- sections have IDs and coherent boundaries
- chunks do not cross section boundaries
- chunk metadata preserves `section_id`, `document_id`
- retrieval filters combine with AND semantics and reject impossible combinations clearly
- citation context assigns deterministic `C1`, `C2`, … handles

Some tests use **strings from the staged demo corpus** (e.g. a concrete `document_id`) as **synthetic fixtures** to exercise parsers and index rows. That is **harness data**, not a requirement that those products always remain in the repo. **Product-specific expected titles** belong in **`data/eval/retrieval_queries.yaml`**, not scattered across many code tests.

## Integration / pipeline tests

End-to-end steps (ingest → chunk → index → search) are validated over **whatever demo corpus** is present in CI or locally. Changing the manifest may change which documents exist, but **pipeline rules** should still hold without per-product pytest churn.

## Curated retrieval evaluation

[`data/eval/retrieval_queries.yaml`](../data/eval/retrieval_queries.yaml) drives `evaluate_retrieval`. It is **intentionally curated and document-specific** for the demo policies. It is **not** a universal contract for every future PDF. Benchmark rows cover **retrieval only**.

## Adding a new policy PDF (checklist)

1. Add the PDF to the manifest (`data/manifests/manual.yaml` or your staging flow; see `docs/data-staging.md`).
2. Run **ingestion** to processed JSON.
3. Run **inspect_documents** (or equivalent) and read the **ingestion quality** report.
4. **Detect sections** → **chunk sections** → **inspect_chunks** for chunk quality.
5. **Rebuild the local index** (`build_index`).
6. Run **`pytest`** — generic invariants should still pass.
7. Run **existing** `evaluate_retrieval` against `data/eval/retrieval_queries.yaml` — expect the **benchmark** to reflect only tracked eval rows; new docs do not require new rows by default.
8. Inspect **generated reports** (retrieval eval markdown, chunk/ingestion reports) for the new document.
9. Add **new curated eval rows** in `data/eval/*.yaml` **only if** the new policy should become part of the **benchmark set** you track over time.

**Default:** onboarding a new product should **not** require editing Python tests—use reports and optional new YAML cases when you choose to extend the benchmark.

For a **corpus-coupling inventory** (pytest vs eval vs production), see [`overfitting_audit.md`](overfitting_audit.md).
