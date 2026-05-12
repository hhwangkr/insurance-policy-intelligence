from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from insurance_ai_ingestion.document_inspection import (
    aggregate_summaries,
    format_console_report,
    format_markdown_report,
    inspect_document,
    load_documents_from_dir,
)


def _configure_stdio_utf8() -> None:
    """Best-effort UTF-8 so console output supports non-ASCII policy text (e.g. Korean)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize quality metrics for processed Document JSON files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *.json Document outputs (e.g. data/processed/documents).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional markdown report path (e.g. data/processed/reports/ingestion_quality.md).",
    )
    args = parser.parse_args(argv)
    _configure_stdio_utf8()

    try:
        documents = load_documents_from_dir(args.input_dir)
    except (FileNotFoundError, OSError, ValueError, ValidationError) as exc:
        print(f"inspect_documents: error: {exc}", file=sys.stderr)
        return 1

    summaries = [inspect_document(d) for d in documents]
    aggregate = aggregate_summaries(summaries)
    print(format_console_report(summaries, aggregate), end="")

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            format_markdown_report(summaries, aggregate),
            encoding="utf-8",
        )
        print(f"Wrote report: {args.report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
