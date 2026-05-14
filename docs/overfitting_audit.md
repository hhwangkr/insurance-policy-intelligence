# Product / document overfitting audit

**Date:** 2026-05-14  
**Scope (active MVP):** `packages/ingestion/src`, `packages/ingestion/tests`, `packages/retrieval/src`, `packages/retrieval/tests`, `scripts/`, plus pointers to `README.md`, `docs/*.md`, `data/eval/retrieval_queries.yaml`.  
**Note:** `packages/generation` existed when this audit was written; it was **removed from the workspace** afterward (LLM answer layer intentionally deferred). Rows that referenced generation tests are **historical** and listed only for traceability.

**Search terms:** `kyobolife`, `samsunglife`, `miraeassetlife`, Korean insurer tokens, `제7조` / `제22조` / `별표` / `청약` / `특별계정`, product slugs (`internet_cancer`, `balance_whole_life`, `variable_annuity`, …), effective dates `20260101`, `20260401`, and full demo `document_id` strings.

## Assumptions

1. **“Production”** here means shipped library code under `packages/*/src` (not tests, not `scripts/` unless noted).
2. **Korean legal headings** (`제N조`, `별표`) in ingestion code are **domain patterns** for Korean disclosure PDFs, not references to one insurer’s PDF file by themselves.
3. **Curated eval YAML** and **README / doc CLI examples** are *supposed* to name the staged demo corpus; they are not defects.
4. This pass is **documentation only**—no refactors, no fixture moves, no behavior changes.

## Why audit before refactor

Refactoring without an inventory risks (a) deleting coverage that still encodes real invariants, or (b) scattering product expectations into code when they belong in `data/eval/*.yaml`. This report classifies coupling so follow-up work can be **surgical** and aligned with [`testing_strategy.md`](testing_strategy.md).

## Classification (A–E)

| ID | Meaning |
|----|---------|
| **A** | Acceptable **demo / eval / docs** coupling (benchmark YAML, README, smoke examples). |
| **B** | Acceptable **regression fixture** (intentionally document- or bug-specific); should carry a short **comment** explaining why. |
| **C** | **Suspicious test overfitting**—pytest asserts a **specific** product string or title where a **structural** invariant or YAML would suffice; may churn when the corpus changes. |
| **D** | **Suspicious production overfitting**—runtime branches tied to one insurer/product instead of data-driven rules. |
| **E** | **Necessary mapping / convention**—slug parsing, heading regexes, staging inference tables; acceptable but flagged as **future config** candidates. |

---

## Executive summary (by category)

| Category | Finding | Notes |
|----------|---------|--------|
| **A** | `data/eval/retrieval_queries.yaml`, README / `docs/retrieval_baseline.md` / `docs/testing_strategy.md` CLI examples | Expected benchmark and documentation coupling. |
| **B** | No isolated “bug #…” regression blocks identified in this pass | Some tests act as **implicit** regressions; treat as **C + comment** unless ticket-linked. |
| **C** | **Concentrated** in `packages/retrieval/tests/*` (repeated demo `document_id` / `kyobolife` filters), `packages/ingestion/tests/test_section_detection.py` (Kyobo-style product names) | Mostly **synthetic fixtures**; risk is **maintenance churn** if corpus IDs change, not wrong behavior. **Partially mitigated (2026-05-14):** shared `demo_corpus_fixtures.py` + file-level comments in retrieval and ingestion tests (see follow-up below). |
| **D** | **No** insurer/product string matches under `packages/retrieval/src` or `packages/ingestion/src` | No `if kyobolife` style production coupling found in audited packages. |
| **E** | `document_id.py` product-type ordering table; `scripts/rule_extraction.py` Korean→slug keyword map; ingestion **별표 / 제N조** regex patterns | Domain and filename-convention coupling; document as mapping/config surface. |

---

## A — Acceptable demo / eval / docs coupling

| Location | Snippet / role | Why acceptable | Action |
|----------|----------------|----------------|--------|
| [`data/eval/retrieval_queries.yaml`](../data/eval/retrieval_queries.yaml) | Queries, filters, `expected_any_section_titles` for Kyobo / Samsung / Mirae cases | Curated benchmark for **current** staged PDFs | **keep** |
| [`README.md`](../README.md) | CLI examples with `kyobolife`, Korean queries | Onboarding / smoke docs | **keep** |
| [`docs/retrieval_baseline.md`](retrieval_baseline.md) | `search_index` / `build_citation_context` examples | Same | **keep** |

---

## B — Acceptable regression fixtures (explicit)

No file in this audit was found with a **ticket-linked** “regression for #…” style comment. The following are **strong candidates to label B** if/when touched:

| Location | Snippet | Action |
|----------|---------|--------|
| `packages/retrieval/tests/test_retrieval_evaluation.py` | `title_matches_any("( 별표 3 )", …)` | Normalization **unit** for title matcher—**keep**; optional **add comment**: “tests spacing variants, not Kyobo corpus”. |

---

## C — Suspicious test overfitting (pytest / synthetic data)

These use **real demo** `document_id` hashes or **Kyobo-like** marketing strings. They are **not** production bugs; the concern is **corpus churn** (IDs change when PDFs are re-staged).

| File | Example | Why flagged (C) | Recommended action |
|------|---------|-------------------|---------------------|
| `packages/retrieval/tests/test_retrieval_baseline.py` | Many lines: `kyobolife_annuity_kyobo_ro_annuity_insurance_policy_terms_20260101_080b9e62`, `samsunglife_cancer_…39af0c18`, `miraeassetlife_variable_annuity_…76283b26` | Reuses **exact** manifest `document_id`s; asserts `insurer == "kyobolife"` etc. | **Addressed (partial):** literals centralized in `packages/retrieval/tests/demo_corpus_fixtures.py`; file-top comment clarifies staged-ID intent. |
| `packages/retrieval/tests/test_retrieval_evaluation.py` | `_KYOBO_DOC`, YAML-ish dicts with `kyobolife` filters | Same | **Addressed (partial):** uses `demo_corpus_fixtures` + module comment. |
| `packages/retrieval/tests/test_citation_context.py` | `document_id="kyobolife_annuity_…"`, `SearchFilters(insurer="kyobolife")` | Same | **Addressed (partial):** uses `demo_corpus_fixtures` + comment after imports. |
| `packages/retrieval/tests/test_document_id.py` | Rows mapping full `document_id` → expected `insurer` / `product_name` / `effective_date` | **Borderline C/E**: asserts **parsing contract** for known filenames—legitimate, but **tied to current slug vocabulary** | **keep**; when new products appear, **extend rows** here or in docs—not scattered assertions elsewhere. |
| `packages/ingestion/tests/test_section_detection.py` | Strings like `개인연금저축교보로연금보험(적립형)`, `미래에셋생명 변액연금보험`, `제15조 (청약의 철회)` | **Highest C signal**: encodes **real product marketing names** from the demo set inside layout tests | **Partially addressed:** module docstring now states Korean product/heading strings are **layout fixtures**, not benchmark expectations for a specific revision; retrieval expectations belong in `data/eval/*.yaml`. **Optional later:** neutral fabricated names that still hit the same regex paths. |
| `packages/ingestion/tests/test_chunk_sections.py` | `( 별표 1 )` synthetic | Appendix chunking behavior | **keep** |
| `packages/retrieval/tests/test_retrieval_baseline.py` | Query / chunk text `청약 철회` | Korean **generic** query words—low overfitting | **keep** |
| `packages/retrieval/tests/test_retrieval_evaluation.py` | `title_matches_any` 별표 spacing | Structural | **keep** |

---

## D — Suspicious production overfitting

| Area | Result |
|------|--------|
| `packages/retrieval/src` | **No** matches for `kyobolife` / `samsunglife` / `miraeassetlife`. |
| `packages/ingestion/src` | **No** insurer slug literals found in audited paths. |

**Conclusion:** No **D**-class “`if insurer == kyobolife`” coupling was found in audited `packages/*/src` trees.

---

## E — Necessary mapping / metadata conventions (config candidates)

| File | Snippet / behavior | Why E (not D) | Recommended action |
|------|-------------------|---------------|---------------------|
| `packages/retrieval/src/insurance_ai_retrieval/document_id.py` | `_KNOWN_PRODUCT_TYPES = ("variable_annuity", "whole_life", "annuity", "cancer")` | Disambiguates **slug parsing** (longest-first). Not tied to one PDF file. | **keep**; document in README/testing_strategy as **filename convention**; future: externalize if product-type vocabulary grows. |
| `packages/ingestion/src/insurance_ai_ingestion/section_detection_patterns.py` | Regexes for `별표`, `( 별표 N )` | Korean **legal layout** convention. | **keep**; domain coupling is expected for KR PDFs. |
| `packages/ingestion/src/insurance_ai_ingestion/region_classification.py` | Heuristics mentioning `제N조`, `별표`, `약관` | Region scoring for Korean TOC/appendix | **keep** |
| `scripts/rule_extraction.py` | `infer_insurer`: `삼성생명`→`samsunglife`, `교보생명`→`kyobolife`, … | **Staging / ops helper** keyword map | **E**; **make config-driven later** if more insurers are onboarded frequently. |
| `scripts/tests/test_rule_extraction.py` | Asserts inferred `kyobolife` / `samsunglife` from Korean samples | Tests the **map** in `rule_extraction.py` | **keep**; belongs with script, not core pipeline. |

---

## Cross-reference: `scripts/` (outside packages)

| File | Pattern | Class | Action |
|------|---------|-------|--------|
| `scripts/rule_extraction.py` | Korean insurer keywords | **E** | **make config-driven later** |
| `scripts/tests/test_rule_extraction.py` | Same | **E** / test of **E** | **keep** |

---

## Follow-up completed (2026-05-14)

1. **C / ingestion:** Extended the **module docstring** in `test_section_detection.py` with an explicit **fixture policy** (layout vs benchmark vs `data/eval/*.yaml`).
2. **C / retrieval tests:** Added `packages/retrieval/tests/demo_corpus_fixtures.py` and switched `test_retrieval_baseline.py`, `test_retrieval_evaluation.py`, and `test_citation_context.py` to import shared demo `document_id` / slug constants; augmented comments at the top of baseline/evaluation/citation tests.
3. **Eval vs pytest:** Unchanged principle—if a test asserts **hit titles** that duplicate `retrieval_queries.yaml`, **move expectation to YAML** per [`testing_strategy.md`](testing_strategy.md). No eval YAML edits in this follow-up.
4. **MVP prune:** Removed `packages/generation` from the workspace; LLM answer layer is deferred (see `README.md` / `docs/testing_strategy.md`).

Remaining **C** exposure: `test_document_id.py` row table and any future new literals outside `demo_corpus_fixtures`.

---

## Verification

- **Initial audit commit:** documentation only (no code changes).  
- **Follow-up + MVP prune:** workspace and docs updated; **`packages/generation` removed**; run `uv run ruff check`, `uv run mypy`, and `uv run pytest` on current `main`.
