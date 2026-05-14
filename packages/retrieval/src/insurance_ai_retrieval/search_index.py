from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder
from insurance_ai_retrieval.index_engine import load_index_config, search_local_index


def _configure_stdout_utf8() -> None:
    enc = getattr(sys.stdout, "encoding", None) or ""
    if enc.casefold() == "utf-8".casefold():
        return
    reconf = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass


def _preview(text: str, *, max_chars: int = 240) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


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
    args = parser.parse_args(argv)

    _configure_stdout_utf8()

    try:
        cfg = load_index_config(args.index_dir)
        model = args.model_name or cfg.model_name
        embedder = LocalSentenceTransformerEmbedder(model)
        hits = search_local_index(
            index_dir=args.index_dir,
            query=args.query,
            embedder=embedder,
            top_k=args.top_k,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        print(f"search_index: error: {exc}", file=sys.stderr)
        return 1

    for h in hits:
        m = h.metadata
        vn = "<null>" if m.variant_name is None else m.variant_name
        lines = [
            f"rank={h.rank} score={h.score:.4f}",
            f"  chunk_id={m.chunk_id}",
            f"  document_id={m.document_id}",
            f"  section_title={m.section_title}",
            f"  section_type={m.section_type}",
            f"  policy_unit_name={m.policy_unit_name}",
            f"  variant_name={vn}",
            f"  pages={m.page_start}-{m.page_end} chars={m.char_start}-{m.char_end}",
            f"  section_id={m.section_id} policy_unit_id={m.policy_unit_id}",
            f"  preview={_preview(m.text)}",
        ]
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
