from __future__ import annotations

import argparse
from pathlib import Path

from inbox_staging import run_inbox_staging


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rule-based inbox staging: read PDFs from data/inbox/manual/, infer lightweight "
            "metadata from the first pages, copy into data/raw/manual/, and write "
            "data/manifests/manual.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not copy PDFs or write manual.yaml.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum PDF pages to read for text extraction (default: 5).",
    )
    args = parser.parse_args()
    return run_inbox_staging(
        repo_root=_repo_root(),
        dry_run=args.dry_run,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    raise SystemExit(main())
