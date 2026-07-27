"""Canonical route-result normalization (AutoLearn v2, Phase 1A).

The v0.3.x routing pipeline can return several incompatible shapes for a
"route decision":

* a bare string ``"symbolic"`` / ``"llm"``
* a ``(action, raw_text)`` tuple from the generate-mode path
* a :class:`~daph_learning.routing.policy.RouteDecision`
* a :class:`~daph_learning.routing.arithmetic_router.RouteDecision`
  (capability-assessment record)
* a ``dict`` with a ``"route"`` key
* ``None`` (unparseable generation / abstention)

Downstream scoring code historically assumed a bare string, so the
generate-mode tuple form silently leaked the raw text into the route
label and corrupted accuracy. This module funnels every routing shape
through one canonical :class:`NormalizedRoute` so scoring, experience
collection, and acceptance evaluation all consume a single typed
representation.

This is a read-only normalization layer; it does not change the
behaviour of the existing routers. The old ``run_autolearn_loop`` keeps
working unchanged (backwards compatibility). The v2 engine consumes
``normalize_route_result`` exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from daph_learning.routing.errors import InvalidRouteDecisionError
from daph_learning.routing.policy import RouteDecision as PolicyRouteDecision

# Re-use the canonical valid-target set from the policy layer so the two
# never drift.
from daph_learning.routing.policy import _VALID_ROUTE_TARGETS

RouteLabel = str  # one of "symbolic" | "llm" | "abstain"

# Backends that can actually be executed. ``abstain`` is a routing decision
# but not an executable backend; the engine resolves it separately.
EXECUTABLE_BACKENDS: frozenset[str] = frozenset({"symbolic", "llm"})


@dataclass(frozen=True)
class NormalizedRoute:
    """Canonical, normalized routing decision.

    Attributes
    ----------
    route : str
        One of ``"symbolic"``, ``"llm"``, ``"abstain"``. Never ``None``;
        unparseable inputs become ``"abstain"`` (fail-closed) unless the
        caller overrides the default.
    raw_text : str | None
        The raw generated text the decision was derived from, when
        available (generate-mode path). Used for provenance/telemetry
        only; never used as the route label.
    confidence : float | None
        Optional confidence in ``[0, 1]``.
    raw_scores : dict[str, float]
        Per-backend raw scores that produced the decision, when available.
    source : str
        Provenance of the decision: ``"logit"``, ``"generate"``,
        ``"fallback"``, ``"baseline"``, ``"custom"``, ``"abstain"``.
    parse_warning : str | None
        Non-fatal note describing any normalization coercion applied
        (e.g. ``"tuple_unpacked"``, ``"unknown_label->abstain"``). ``None``
        when the input was already canonical.
    """

    route: RouteLabel
    raw_text: str | None = None
    confidence: float | None = None
    raw_scores: dict[str, float] = field(default_factory=dict)
    source: str = "baseline"
    parse_warning: str | None = None

    def __post_init__(self) -> None:
        if self.route not in _VALID_ROUTE_TARGETS:
            raise InvalidRouteDecisionError(
                f"NormalizedRoute.route={self.route!r} is not one of "
                f"{sorted(_VALID_ROUTE_TARGETS)}"
            )

    @property
    def is_executable(self) -> bool:
        return self.route in EXECUTABLE_BACKENDS


def _coerce_label(value: Any) -> str | None:
    """Lowercase-coerce a label string to the canonical set, or ``None``."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if label in _VALID_ROUTE_TARGETS:
        return label
    # Tolerate common casing/spacing variants without silently accepting
    # arbitrary strings.
    upper = value.strip().upper()
    if upper == "SYMBOLIC":
        return "symbolic"
    if upper == "LLM":
        return "llm"
    if upper in {"ABSTAIN", "NONE", "UNKNOWN"}:
        return "abstain"
    return None


def normalize_route_result(
    result: Any,
    *,
    source: str = "baseline",
    on_unknown: str = "abstain",
) -> NormalizedRoute:
    """Normalize any routing-path output into a :class:`NormalizedRoute`.

    Accepted input shapes:

    * ``NormalizedRoute`` — returned as-is (source re-tagged).
    * :class:`~daph_learning.routing.policy.RouteDecision` — fields copied.
    * ``str`` — coerced to a label.
    * ``tuple`` / ``list`` of length 2 interpreted as ``(label, raw_text)``
      (the generate-mode shape). This is the Phase 1A bug fix: the tuple
      form is unpacked instead of being treated as a string label.
    * ``dict`` / ``Mapping`` with a ``"route"`` (or ``"backend"``) key.
    * ``None`` — mapped to ``on_unknown`` (default ``"abstain"``).

    Unknown labels are mapped to ``on_unknown`` (fail-closed default
    ``"abstain"``) with a ``parse_warning`` rather than raising, because
    routing must always produce a decision the engine can act on. Pass
    ``on_unknown="raise"`` to instead raise
    :class:`InvalidRouteDecisionError`.

    Parameters
    ----------
    on_unknown : str
        One of ``"abstain"``, ``"llm"``, or ``"raise"``. Default
        ``"abstain"`` (fail-closed: an unparseable route does not silently
        become an LLM call).
    """
    if on_unknown not in {"abstain", "llm", "raise"}:
        raise ValueError(f"on_unknown must be 'abstain', 'llm', or 'raise'; got {on_unknown!r}")

    # Already canonical.
    if isinstance(result, NormalizedRoute):
        return NormalizedRoute(
            route=result.route,
            raw_text=result.raw_text,
            confidence=result.confidence,
            raw_scores=dict(result.raw_scores),
            source=source if source != "baseline" else result.source,
            parse_warning=result.parse_warning,
        )

    # Policy-level RouteDecision.
    if isinstance(result, PolicyRouteDecision):
        return NormalizedRoute(
            route=result.route,
            raw_text=None,
            confidence=result.confidence,
            raw_scores=dict(result.raw_scores),
            source=source if source != "baseline" else result.source,
        )

    # Generate-mode tuple / list: (label, raw_text).
    if isinstance(result, (tuple, list)):
        if len(result) == 2:
            label_raw, raw_text = result
            label = _coerce_label(label_raw)
            if label is None:
                if on_unknown == "raise":
                    raise InvalidRouteDecisionError(
                        f"tuple route label {label_raw!r} could not be coerced"
                    )
                return NormalizedRoute(
                    route=on_unknown,
                    raw_text=raw_text if isinstance(raw_text, str) else None,
                    source=source,
                    parse_warning=f"unknown_tuple_label->{on_unknown}",
                )
            return NormalizedRoute(
                route=label,
                raw_text=raw_text if isinstance(raw_text, str) else None,
                source=source,
                parse_warning="tuple_unpacked",
            )
        if len(result) == 1:
            return normalize_route_result(result[0], source=source, on_unknown=on_unknown)
        raise InvalidRouteDecisionError(
            f"tuple/list route result must have length 1 or 2; got {len(result)}"
        )

    # Mapping.
    if isinstance(result, Mapping):
        label_raw = result.get("route", result.get("backend"))
        if label_raw is None:
            if on_unknown == "raise":
                raise InvalidRouteDecisionError(
                    f"mapping route result has no 'route'/'backend' key; "
                    f"keys={sorted(map(str, result.keys()))}"
                )
            return NormalizedRoute(
                route=on_unknown,
                source=source,
                parse_warning=f"missing_route_key->{on_unknown}",
            )
        label = _coerce_label(label_raw)
        if label is None:
            if on_unknown == "raise":
                raise InvalidRouteDecisionError(
                    f"mapping route label {label_raw!r} could not be coerced"
                )
            return NormalizedRoute(
                route=on_unknown,
                source=source,
                parse_warning=f"unknown_mapping_label->{on_unknown}",
            )
        confidence = result.get("confidence")
        raw_scores = result.get("raw_scores", {})
        if confidence is not None and not isinstance(confidence, (int, float)):
            raise InvalidRouteDecisionError(
                f"mapping route confidence {confidence!r} is not numeric"
            )
        if not isinstance(raw_scores, Mapping):
            raise InvalidRouteDecisionError(
                f"mapping route raw_scores {raw_scores!r} is not a mapping"
            )
        return NormalizedRoute(
            route=label,
            confidence=float(confidence) if confidence is not None else None,
            raw_scores={str(k): float(v) for k, v in raw_scores.items()},
            source=source,
        )

    # Bare string.
    if isinstance(result, str):
        label = _coerce_label(result)
        if label is not None:
            return NormalizedRoute(route=label, source=source)
        if on_unknown == "raise":
            raise InvalidRouteDecisionError(
                f"route string {result!r} could not be coerced to a known label"
            )
        return NormalizedRoute(
            route=on_unknown,
            raw_text=result,
            source=source,
            parse_warning=f"unknown_label->{on_unknown}",
        )

    # None / anything else.
    if result is None:
        if on_unknown == "raise":
            raise InvalidRouteDecisionError("route result is None")
        return NormalizedRoute(
            route=on_unknown,
            source="abstain" if on_unknown == "abstain" else source,
            parse_warning=f"none->{on_unknown}",
        )

    if on_unknown == "raise":
        raise InvalidRouteDecisionError(
            f"route result of type {type(result).__name__} cannot be normalized"
        )
    return NormalizedRoute(
        route=on_unknown,
        source=source,
        parse_warning=f"unhandled_type_{type(result).__name__}->{on_unknown}",
    )


def normalize_route_sequence(
    results: Sequence[Any],
    *,
    source: str = "baseline",
    on_unknown: str = "abstain",
) -> list[NormalizedRoute]:
    """Normalize a sequence of routing outputs element-wise."""
    return [
        normalize_route_result(r, source=source, on_unknown=on_unknown) for r in results
    ]


__all__ = [
    "EXECUTABLE_BACKENDS",
    "NormalizedRoute",
    "RouteLabel",
    "normalize_route_result",
    "normalize_route_sequence",
]
