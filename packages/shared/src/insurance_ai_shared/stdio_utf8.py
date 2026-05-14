from __future__ import annotations

import sys


def configure_stdout_utf8() -> None:
    """Best-effort UTF-8 stdout on Windows (avoids ``UnicodeEncodeError`` on Korean text)."""
    enc = getattr(sys.stdout, "encoding", None) or ""
    if enc.casefold() == "utf-8".casefold():
        return
    reconf = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass
