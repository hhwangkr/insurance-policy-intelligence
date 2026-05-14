from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from insurance_ai_generation.answer_prompt import build_grounded_answer_prompt
from insurance_ai_generation.generate_grounded_answer import generate_grounded_answer
from insurance_ai_generation.llm_provider import LLMProviderError
from insurance_ai_generation.provider_registry import GenerationProviderConfig, create_llm_provider
from insurance_ai_retrieval.citation_context import CitationContextBundle
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Debug-only: load a saved CitationContextBundle JSON, build a prompt, run a "
            "registry-backed provider (``static`` or local ``ollama``), and print generation JSON."
        ),
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        required=True,
        help="Path to CitationContextBundle JSON (e.g. build_citation_context --output-path).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="static",
        help="Registry provider: 'static' (offline) or 'ollama' (local HTTP; requires --model).",
    )
    parser.add_argument(
        "--static-response",
        type=str,
        default=None,
        help="Override static provider body (JSON or plain text). Default uses citation ids.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    configure_stdout_utf8()

    try:
        raw = json.loads(args.context_path.read_text(encoding="utf-8"))
        bundle = CitationContextBundle.model_validate(raw)
        prompt = build_grounded_answer_prompt(bundle)
        config = GenerationProviderConfig(
            provider_name=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            static_response=args.static_response,
        )
        provider = create_llm_provider(
            config,
            citation_ids_for_static_default=prompt.citation_ids,
        )
        result = generate_grounded_answer(
            prompt,
            provider,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        payload = {
            "answer": result.answer.model_dump(),
            "validation": result.validation.model_dump(),
            "provider_name": result.provider_name,
            "model_name": result.model_name,
            "raw_text": result.raw_text,
        }
        out = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    except ValueError as exc:
        print(f"generate_answer: error: {exc}", file=sys.stderr)
        return 1
    except LLMProviderError as exc:
        print(f"generate_answer: error: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"generate_answer: error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"generate_answer: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
