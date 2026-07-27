from __future__ import annotations

import re

_FINAL_RE = re.compile(r"\bFINAL:\s*(-?\d+)\b", re.IGNORECASE)
_EXACT_RE = re.compile(r"^\s*(-?\d+)\s*$")
_FIRST_INT_RE = re.compile(r"-?\d+")


def parse_final_or_exact(text: str) -> int | None:
    m = _FINAL_RE.search(str(text))
    if m:
        return int(m.group(1))
    m = _EXACT_RE.match(str(text))
    if m:
        return int(m.group(1))
    return None


def parse_legacy_first_int(text: str) -> int | None:
    """Historical notebook-compatible parser. Not recommended for new runs."""
    m = _FIRST_INT_RE.search(str(text))
    return int(m.group(0)) if m else None
