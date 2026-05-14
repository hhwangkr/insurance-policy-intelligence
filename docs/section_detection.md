# Section detection (Phase 2D)

This note describes the **deterministic** Korean policy PDF section pipeline in `insurance_ai_ingestion`. It is intentionally separate from **chunking**, **retrieval**, and **cross-reference resolution** (clause-to-clause graph work is out of scope here).

## Pipeline overview

1. **Candidate generation** (`collect_section_candidates`)  
   Per-page line scan with regex rules for TOC, guide, part (관), article (조), appendix (별표), legal index headings, and glossary. Each hit becomes a `SectionCandidate` with a stable `candidate_id` derived from the character offset.

2. **Region classification** (`region_classification`)  
   One primary `PageRegion` per page (cover / toc / guide / summary / policy_body / appendix / legal_reference / glossary / unknown) from shallow text signals. This is **heuristic** and may be swapped for a pluggable `RegionClassifier`.

3. **Filtering / suppression** (`_should_emit_candidate`, TOC-like backstops, appendix gates, legal slice gate, inline article-reference suppression)  
   Structural headings are dropped on front-matter regions; pointer-only TOC rows and appendix/article pins are suppressed even when the region label is weak.

4. **Deterministic assembly** (`assemble_document_sections`)  
   Accepted candidates are ordered by offset; each section’s `text` is a **substring** of the source from its start offset to the next accepted heading (with appendix-specific caps at the next product-variant marker when applicable). `parent_section_id` links articles to the most recent part.

5. **Policy unit assignment** (`policy_unit_detection`)  
   Integrated multi-variant PDFs (e.g. 적립형 / 거치형 / 즉시형) get `PolicyUnit` spans from variant markers; consecutive duplicate markers collapse; the last unit is trimmed at global tail headings; `attach_policy_unit_metadata` assigns units by span containment and clears product metadata after the document tail cutoff.

## Modules (post-refactor)

| Module | Role |
|--------|------|
| `section_detection.py` | Orchestration: candidates, assembly, public `detect_sections` / `build_sections_artifact`. |
| `section_detection_patterns.py` | Shared compiled regexes for headings, pointers, tail matter, and policy markers. |
| `section_detection_evidence.py` | Stable `assembly_evidence` / filter token strings and small formatters. |
| `section_text_layout.py` | Full-text join and offset → page mapping. |
| `policy_unit_detection.py` | Marker scan, dedupe, tail trim, policy metadata on `DocumentSection`. |
| `region_classification.py` | Page-level region scoring (uses shared patterns where aligned). |

## Cross-reference mapping

**Out of scope** for this layer. Section detection only produces citation-ready slices and metadata; resolving “제7조” mentions inside prose to canonical section IDs is a separate concern (graph / linker), not implemented here.

## Section-aware chunking (Phase 2E)

Deterministic chunking reads ``data/processed/sections/*.sections.json`` and writes ``data/processed/chunks/*.chunks.json`` via ``uv run python -m insurance_ai_ingestion.chunk_sections``. Each chunk is confined to a single ``DocumentSection`` (no cross-section merges). ``DocumentChunk.page_start`` / ``page_end`` inherit the section’s page span until per-chunk page mapping exists.
