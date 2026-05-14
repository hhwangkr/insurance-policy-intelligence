from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder
from insurance_ai_retrieval.index_engine import SearchFilters, load_index_config, search_local_index
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def _preview(text: str, *, max_chars: int = 240) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _parse_csv_types(raw: str | None) -> frozenset[str] | None:
    if raw is None or not str(raw).strip():
        return None
    parts = tuple(p.strip() for p in str(raw).split(",") if p.strip())
    return frozenset(parts) if parts else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query a local chunk embedding index (retrieval-only; no answer generation).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing ``chunk_embeddings.npy`` and friends.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query string.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hits to return (default: 5).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model id (must match the index; default: read from ``index_config.json``).",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="Restrict search to a single ``document_id`` (exact match).",
    )
    parser.add_argument(
        "--insurer",
        type=str,
        default=None,
        help="Restrict search to chunks whose ``document_id`` maps to this insurer slug.",
    )
    parser.add_argument(
        "--product-type",
        type=str,
        default=None,
        help="Restrict search to chunks whose ``document_id`` maps to this product type slug.",
    )
    parser.add_argument(
        "--product-name",
        type=str,
        default=None,
        help="Substring match (case-insensitive) on derived ``product_name`` from ``document_id``.",
    )
    parser.add_argument(
        "--policy-unit-name",
        type=str,
        default=None,
        help="Substring match (case-insensitive) on ``policy_unit_name`` when present.",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help="Substring match (case-insensitive) on ``variant_name`` when present.",
    )
    parser.add_argument(
        "--include-section-types",
        type=str,
        default=None,
        help=(
            "Comma-separated whitelist of ``section_type`` values "
            "(implies ``--no-default-section-type-filter``)."
        ),
    )
    parser.add_argument(
        "--exclude-section-types",
        type=str,
        default=None,
        help="Comma-separated extra ``section_type`` values to exclude after defaults.",
    )
    parser.add_argument(
        "--no-default-section-type-filter",
        action="store_true",
        help=(
            "Do not exclude toc/cover/guide/summary by default "
            "(still honors ``--exclude-section-types``)."
        ),
    )
    parser.add_argument(
        "--dedupe-section",
        action="store_true",
        help="Return at most one hit per ``section_id`` (highest-scoring chunk per section).",
    )
    args = parser.parse_args(argv)

    configure_stdout_utf8()

    include_types = _parse_csv_types(args.include_section_types)
    exclude_types = _parse_csv_types(args.exclude_section_types) or frozenset()
    use_default_excludes = not args.no_default_section_type_filter
    if include_types is not None:
        use_default_excludes = False
    filters = SearchFilters(
        document_id=args.document_id,
        insurer=args.insurer,
        product_type=args.product_type,
        product_name=args.product_name,
        policy_unit_name=args.policy_unit_name,
        variant_name=args.variant_name,
        include_section_types=include_types,
        exclude_section_types=exclude_types,
        use_default_section_type_excludes=use_default_excludes,
    )

    try:
        cfg = load_index_config(args.index_dir)
        model = args.model_name or cfg.model_name
        embedder = LocalSentenceTransformerEmbedder(model)
        hits = search_local_index(
            index_dir=args.index_dir,
            query=args.query,
            embedder=embedder,
            top_k=args.top_k,
            filters=filters,
            dedupe_section=args.dedupe_section,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        print(f"search_index: error: {exc}", file=sys.stderr)
        return 1

    for h in hits:
        m = h.metadata
        vn = "<null>" if m.variant_name is None else m.variant_name
        pun = "<null>" if m.policy_unit_name is None else m.policy_unit_name
        ins = "<null>" if m.insurer is None else m.insurer
        pt = "<null>" if m.product_type is None else m.product_type
        pn = "<null>" if m.product_name is None else m.product_name
        lines = [
            f"rank={h.rank} score={h.score:.4f}",
            f"  chunk_id={m.chunk_id}",
            f"  document_id={m.document_id}",
            f"  insurer={ins} product_type={pt} product_name={pn}",
            f"  section_title={m.section_title}",
            f"  section_type={m.section_type}",
            f"  policy_unit_name={pun}",
            f"  variant_name={vn}",
            f"  pages={m.page_start}-{m.page_end} chars={m.char_start}-{m.char_end}",
            f"  section_id={m.section_id} policy_unit_id={m.policy_unit_id}",
            f"  preview={_preview(m.text)}",
        ]
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
