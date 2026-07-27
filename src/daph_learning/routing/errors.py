"""Typed routing/steering failure taxonomy (V037-003).

Replaces the v0.3.6 pattern of ``except Exception: continue`` in the routing
cascade with named failures so that:

- expected failures (multi-token labels, context-boundary drift) fall back to
  the next resolver with structured telemetry;
- unexpected failures (model crash, shape mismatch, NaN) terminate the run
  instead of being silently swallowed.

Taxonomy:

- :class:`RouteResolutionError` — base for any failure resolving a route.
- :class:`MultiTokenRouteError` — a route label tokenizes to >1 token and no
  single-token fallback is available. Subclasses ``ValueError`` so existing
  ``except ValueError`` callers keep working.
- :class:`ContextBoundaryError` — the rendered prompt / label boundary is
  ambiguous or unresolvable (e.g. no ``ACTION:`` anchor, mismatched
  leading-space).
- :class:`InvalidRouteDecisionError` — a custom ``route_fn`` returned a
  result that cannot be coerced to a ``RouteDecision``.
- :class:`SteeringApplicationError` — failure applying a steering hook
  (residual addition, layer index out of range, vector shape mismatch).
- :class:`ModelRoutingError` — a model-mediated routing forward/generate
  call failed for an unexpected reason.

The first three are :class:`RouteResolutionError` subclasses and are the
"expected" failures the routing cascade may fall back from. The last two are
independent and indicate infrastructure problems that should generally
propagate.
"""

from __future__ import annotations


class RouteResolutionError(Exception):
    """Base class for failures while resolving a route for a task."""


class MultiTokenRouteError(RouteResolutionError, ValueError):
    """A route label tokenizes to more than one token and no first-token
    fallback is available (or was disabled).

    Subclasses :class:`ValueError` so existing ``except ValueError`` callers
    keep working.
    """


class ContextBoundaryError(RouteResolutionError, ValueError):
    """The rendered prompt / label boundary is ambiguous or unresolvable.

    Raised when the routing prompt does not terminate with the expected
    ``ACTION:`` anchor, or when the leading-space boundary cannot be
    determined for the configured tokenizer.

    Subclasses :class:`ValueError` so existing ``except ValueError`` callers
    and ``pytest.raises(ValueError)`` tests keep working.
    """


class InvalidRouteDecisionError(RouteResolutionError):
    """A custom ``route_fn`` returned a result that cannot be coerced to a
    valid :class:`~daph_learning.routing.policy.RouteDecision`.
    """


class SteeringApplicationError(Exception):
    """Failure applying a steering hook (residual addition, layer index out
    of range, vector shape mismatch with the model's hidden size, etc.).
    """


class ModelRoutingError(Exception):
    """A model-mediated routing forward or generate call failed for an
    unexpected reason (shape mismatch, NaN logits, OOM, etc.).

    This is raised when a routing failure is not one of the expected
    :class:`RouteResolutionError` subclasses.
    """


__all__ = [
    "RouteResolutionError",
    "MultiTokenRouteError",
    "ContextBoundaryError",
    "InvalidRouteDecisionError",
    "SteeringApplicationError",
    "ModelRoutingError",
]
