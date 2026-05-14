from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from insurance_ai_generation.answer_prompt import (
    build_grounded_answer_prompt,
    format_grounded_answer_prompt_json,
)
from insurance_ai_retrieval.citation_context import CitationContextBundle
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build LLM-ready grounded-answer messages from a saved CitationContextBundle JSON "
            "(debug / development only; no LLM call). Normal services call "
            "build_grounded_answer_prompt(bundle) in memory instead of this CLI."
        ),
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        required=True,
        help=(
            "Path to a ``CitationContextBundle`` JSON (e.g. from ``build_citation_context`` "
            "``--output-path``). Not a normal pipeline artifact."
        ),
    )
    args = parser.parse_args(argv)

    configure_stdout_utf8()

    try:
        raw = json.loads(args.context_path.read_text(encoding="utf-8"))
        bundle = CitationContextBundle.model_validate(raw)
        prompt = build_grounded_answer_prompt(bundle)
        out = format_grounded_answer_prompt_json(prompt)
    except (FileNotFoundError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"build_answer_prompt: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
