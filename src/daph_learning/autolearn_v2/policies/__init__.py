"""Policy representations (AutoLearn v2, Phase 6).

A *policy* is a versioned object that maps a task representation to a
routing decision. The package is designed so the runtime can support
several policy families without rewriting the engine:

* :class:`SingleVectorPolicy` — Stage A/B/C: a single static steering
  vector applied as ``h' = h + alpha * v``, with incremental trust-region
  updates and reward-gap supervision.
* :class:`MultiVectorPolicy` — a stack of steering vectors
  ``[v1, ..., vr]`` with per-vector coefficients (Stage D interface).
* :class:`ConditionalSteeringPolicy` — Stage D / v2.5 stretch goal:
  ``V in R^(hidden x rank)`` with a router ``c(x) = g_phi(h(x))`` so
  ``h' = h + V c(x)``. The interface is implemented so the engine can
  later adopt it; it is not the default until it consistently beats the
  simpler system.

All policies are immutable :class:`dataclass(frozen=True)` records with
deterministic IDs and parent lineage so the registry can retain them.
"""

from .base import Policy, PolicyKind, PolicyMetrics, hash_vector
from .static_vector import SingleVectorPolicy
from .low_rank import MultiVectorPolicy
from .conditional import ConditionalSteeringPolicy

__all__ = [
    "ConditionalSteeringPolicy",
    "MultiVectorPolicy",
    "Policy",
    "PolicyKind",
    "PolicyMetrics",
    "SingleVectorPolicy",
    "hash_vector",
]
