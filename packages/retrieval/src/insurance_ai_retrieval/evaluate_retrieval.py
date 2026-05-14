from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder
from insurance_ai_retrieval.index_engine import load_index_config
from insurance_ai_retrieval.retrieval_evaluation import (
    load_retrieval_queries,
    render_markdown_report,
    run_queries_on_index,
    write_report,
)
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic retrieval evaluation over a YAML query set "
            "(dense search only; no LLM / answer generation)."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing ``chunk_embeddings.npy`` and friends.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help=(
            "YAML file with a top-level ``queries`` list "
            "(see ``data/eval/retrieval_queries.yaml``)."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="Markdown report output path (parent directories are created).",
    )
    parser.add_argument(
        "--fail-on-miss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit with status 1 when any query fails hit@5 or filter checks (default: true).",
    )
    args = parser.parse_args(argv)

    configure_stdout_utf8()

    try:
        queries = load_retrieval_queries(args.queries)
        cfg = load_index_config(args.index_dir)
        embedder = LocalSentenceTransformerEmbedder(cfg.model_name)
        summary = run_queries_on_index(
            index_dir=args.index_dir,
            embedder=embedder,
            queries=queries,
        )
        report = render_markdown_report(
            summary,
            index_dir=args.index_dir,
            queries_path=args.queries,
        )
        write_report(args.report_path, report)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"evaluate_retrieval: error: {exc}", file=sys.stderr)
        return 1

    summary_dict = {
        "total_queries": summary.total_queries,
        "hit_at_1": summary.hit_at_1,
        "hit_at_3": summary.hit_at_3,
        "hit_at_5": summary.hit_at_5,
        "report_path": str(args.report_path.resolve()),
    }
    print(json.dumps(summary_dict, indent=2), file=sys.stderr)

    if args.fail_on_miss:
        for r in summary.per_query:
            if not (r.pass_at_5 and r.all_filters_matched):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
