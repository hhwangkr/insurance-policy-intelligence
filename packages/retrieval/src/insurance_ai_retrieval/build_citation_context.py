from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insurance_ai_retrieval.citation_context import (
    build_citation_context,
    format_citation_bundle_json,
)
from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder
from insurance_ai_retrieval.index_engine import SearchFilters, load_index_config
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def _parse_csv_types(raw: str | None) -> frozenset[str] | None:
    if raw is None or not str(raw).strip():
        return None
    parts = tuple(p.strip() for p in str(raw).split(",") if p.strip())
    return frozenset(parts) if parts else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run metadata-scoped retrieval and emit a citation-ready JSON bundle "
            "(no LLM; retrieval only)."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing ``chunk_embeddings.npy`` and friends.",
    )
    parser.add_argument("--query", type=str, required=True, help="Natural language query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits (default: 5).")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model id (must match the index; default: read from ``index_config.json``).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional debug output path for saving the query-time context bundle "
            "(same JSON as stdout, UTF-8). Not a normal pipeline artifact."
        ),
    )
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--insurer", type=str, default=None)
    parser.add_argument("--product-type", type=str, default=None)
    parser.add_argument("--product-name", type=str, default=None)
    parser.add_argument("--policy-unit-name", type=str, default=None)
    parser.add_argument("--variant-name", type=str, default=None)
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
        bundle = build_citation_context(
            index_dir=args.index_dir,
            query=args.query,
            embedder=embedder,
            filters=filters,
            top_k=args.top_k,
            dedupe_section=args.dedupe_section,
        )
        text = format_citation_bundle_json(bundle)
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        print(f"build_citation_context: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
