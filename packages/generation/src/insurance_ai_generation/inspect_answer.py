from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from insurance_ai_generation.inspect_grounded_answer import (
    build_grounded_answer_inspection,
    generation_result_from_generate_answer_json,
    inspection_report_to_markdown,
)
from insurance_ai_retrieval.citation_context import CitationContextBundle
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def _read_generation_json(path: Path) -> dict[str, Any]:
    if str(path) == "-":
        raw = sys.stdin.read()
    else:
        raw = path.read_text(encoding="utf-8")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "generation JSON must be a single object"
        raise ValueError(msg)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Debug-only: combine a saved CitationContextBundle JSON with a saved "
            "``generate_answer`` JSON (file or stdin ``-``) and emit inspection JSON; "
            "optionally write a markdown report."
        ),
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        required=True,
        help="Path to CitationContextBundle JSON.",
    )
    parser.add_argument(
        "--generation-result-path",
        type=Path,
        required=True,
        help="Path to ``generate_answer`` stdout JSON, or ``-`` to read stdin.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to write a markdown inspection report.",
    )
    args = parser.parse_args(argv)

    configure_stdout_utf8()

    try:
        bundle_raw = json.loads(args.context_path.read_text(encoding="utf-8"))
        bundle = CitationContextBundle.model_validate(bundle_raw)
        gen_payload = _read_generation_json(args.generation_result_path)
        result = generation_result_from_generate_answer_json(gen_payload)
        report = build_grounded_answer_inspection(bundle, result)
        out_json = json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    except ValueError as exc:
        print(f"inspect_answer: error: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"inspect_answer: error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"inspect_answer: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(out_json)
    if args.report_path is not None:
        md = inspection_report_to_markdown(report)
        args.report_path.write_text(md, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
