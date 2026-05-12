from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_hex_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
