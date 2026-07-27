"""Structured fallback telemetry (V037-003).

Every routing fallback emits a structured JSON event so that silent
degradation becomes auditable. Events are written to a module-level list
(``FALLBACK_EVENTS``) for in-process inspection by tests, and also forwarded
to a caller-supplied sink (default: ``logging.getLogger("daph_learning")``).

Example event::

    {"event": "route_fallback", "from": "contextual", "to": "isolated",
     "reason": "multi_token_label", "task_id": "t042"}

The sink is a callable that accepts a single ``dict``. Tests can install
their own sink via :func:`set_fallback_sink` and assert on the emitted
events without touching the logger.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

FALLBACK_EVENTS: list[dict[str, Any]] = []
"""In-process record of every emitted fallback event. Cleared by
:func:`reset_fallback_events` (called automatically by the test fixture)."""

_MAX_FALLBACK_EVENTS = 10_000
"""Maximum number of events retained in ``FALLBACK_EVENTS``. Once the cap
is reached, the oldest event is discarded (ring-buffer semantics). This
prevents unbounded memory growth in long-running loops."""

_sink: Callable[[dict[str, Any]], None] | None = None


def set_fallback_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    """Install a callable that receives every fallback event dict.

    Pass ``None`` to restore the default sink (which writes a single-line
    JSON record to the ``daph_learning`` logger at WARNING level).
    """
    global _sink
    _sink = sink


def reset_fallback_events() -> None:
    """Clear the in-process event list. Test fixtures should call this."""
    FALLBACK_EVENTS.clear()


def emit_fallback(
    *,
    event: str,
    frm: str,
    to: str,
    reason: str,
    **extra: Any,
) -> None:
    """Emit a structured fallback event.

    Parameters
    ----------
    event : str
        Event name, e.g. ``"route_fallback"``.
    frm : str
        The resolver/path being fallen back from (``"contextual"``,
        ``"isolated"``, ``"logit"``, ``"steered"``).
    to : str
        The resolver/path being fallen back to (``"isolated"``,
        ``"generate"``, ``"baseline"``, ``"llm_default"``).
    reason : str
        Short machine-readable reason code (``"multi_token_label"``,
        ``"context_boundary"``, ``"steering_hook_failure"``, ...).
    **extra : Any
        Additional fields (``task_id``, ``iteration``, ``layer``, ...).
    """
    payload: dict[str, Any] = {
        "event": event,
        "from": frm,
        "to": to,
        "reason": reason,
    }
    payload.update(extra)
    if len(FALLBACK_EVENTS) >= _MAX_FALLBACK_EVENTS:
        FALLBACK_EVENTS.pop(0)
    FALLBACK_EVENTS.append(payload)
    if _sink is not None:
        _sink(payload)
    else:
        logging.getLogger("daph_learning").warning(
            "fallback: %s", json.dumps(payload, sort_keys=True)
        )


__all__ = [
    "FALLBACK_EVENTS",
    "emit_fallback",
    "reset_fallback_events",
    "set_fallback_sink",
]
