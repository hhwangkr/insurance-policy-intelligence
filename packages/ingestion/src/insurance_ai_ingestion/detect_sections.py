from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from insurance_ai_ingestion.document_inspection import load_documents_from_dir
from insurance_ai_ingestion.section_detection import build_sections_artifact


def sections_json_path(*, output_dir: Path, document_id: str) -> Path:
    safe_id = document_id.strip()
    if not safe_id:
        msg = "document_id must be non-empty"
        raise ValueError(msg)
    return output_dir / f"{safe_id}.sections.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect Korean insurance policy section headings from processed Document JSON "
            "and write sidecar section artifacts."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *.json Document outputs (e.g. data/processed/documents).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for *.sections.json files (e.g. data/processed/sections).",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_documents_from_dir(args.input_dir)
    except (FileNotFoundError, OSError, ValueError, ValidationError) as exc:
        print(f"detect_sections: error: {exc}", file=sys.stderr)
        return 1

    for document in documents:
        artifact = build_sections_artifact(document=document)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = sections_json_path(output_dir=args.output_dir, document_id=document.document_id)
        path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
