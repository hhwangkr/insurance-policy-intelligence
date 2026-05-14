from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insurance_ai_retrieval.embedder import LocalSentenceTransformerEmbedder
from insurance_ai_retrieval.index_engine import build_local_index
from insurance_ai_shared.stdio_utf8 import configure_stdout_utf8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local numpy embedding index from ``*.chunks.json``.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        required=True,
        help="Directory containing ``*.chunks.json``.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Output directory (writes ``chunk_embeddings.npy``, ``chunk_metadata.jsonl``, "
        "``index_config.json``).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="sentence-transformers model id (default: intfloat/multilingual-e5-small).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encode batch size (default: 32).",
    )
    args = parser.parse_args(argv)

    configure_stdout_utf8()

    try:
        embedder = LocalSentenceTransformerEmbedder(args.model_name)
        cfg = build_local_index(
            chunks_dir=args.chunks_dir,
            index_dir=args.index_dir,
            embedder=embedder,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        print(f"build_index: error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote index under {args.index_dir.resolve()}", file=sys.stderr)
    print(cfg.to_json_dict(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
