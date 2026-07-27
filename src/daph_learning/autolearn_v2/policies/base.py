"""Policy base interface (AutoLearn v2, Phase 6)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

PolicyKind = Literal["single_vector", "multi_vector", "conditional"]


def hash_vector(values: np.ndarray) -> str:
    """Deterministic SHA-256 hash of a vector's bytes (for lineage)."""
    arr = np.ascontiguousarray(values.astype(np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class PolicyMetrics:
    """Validation metrics attached to a policy at acceptance time.

    Stored immutably on the policy record so lineage is self-describing.
    """

    utility: float
    routing_accuracy: float
    symbolic_rate: float
    llm_rate: float
    abstain_rate: float
    per_domain: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class Policy:
    """Abstract base for versioned routing policies.

    Concrete subclasses are immutable frozen dataclasses. The base
    defines the common provenance/lineage contract: every policy has a
    deterministic ``policy_id``, a ``parent_policy_id`` (``None`` for the
    root), and hashes of the model/tokenizer/dataset it was learned
    against.
    """

    kind: PolicyKind = "single_vector"  # overridden by subclasses

    # Common fields are declared on subclasses (frozen dataclasses cannot
    # inherit non-frozen defaults cleanly). This base exists only to
    # document the interface and provide helper methods.

    def apply(self, representation: np.ndarray) -> np.ndarray:
        """Apply the policy to a representation ``h`` -> ``h'``.

        Subclasses override this. For a single-vector policy this is
        ``h + alpha * v``.
        """
        raise NotImplementedError

    def route_score(self, representation: np.ndarray) -> dict[str, float]:
        """Return per-backend routing scores for a representation.

        Subclasses override this. The default single-vector
        implementation derives scores from the steered representation;
        the engine converts scores into a :class:`NormalizedRoute`.
        """
        raise NotImplementedError

    def vector_norm(self) -> float:
        raise NotImplementedError

    def displacement_from(self, other: "Policy") -> float:
        """L2 distance between this policy and another (steering displacement)."""
        raise NotImplementedError


__all__ = ["Policy", "PolicyKind", "PolicyMetrics", "hash_vector"]
