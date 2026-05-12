from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inbox_staging import run_inbox_staging


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _try_reconfigure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles when printing Hangul metadata."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def main() -> int:
    _try_reconfigure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Rule-based inbox staging: read PDFs from data/inbox/manual/, infer lightweight "
            "metadata from the first pages, copy into data/raw/manual/, and write "
            "data/manifests/manual.yaml."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Copy PDFs to data/raw/manual/ and write data/manifests/manual.yaml. "
            "Default is preview-only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force preview-only even if --apply is passed.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum PDF pages to read for text extraction (default: 5).",
    )
    args = parser.parse_args()
    apply = bool(args.apply) and not bool(args.dry_run)
    return run_inbox_staging(
        repo_root=_repo_root(),
        apply=apply,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    raise SystemExit(main())
