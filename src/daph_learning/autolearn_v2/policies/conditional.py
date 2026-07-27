"""Conditional low-rank steering policy (AutoLearn v2.5 stretch goal).

Stage D / v2.5::

    V in R^(hidden x rank)
    c(x) = g_phi(h(x))           # router network
    delta_h(x) = V c(x)
    h'(x) = h(x) + delta_h(x)

The interface is implemented so the engine can later adopt it, but it is
**not the default**. The design brief requires that this path not become
default until it consistently beats the simpler single-vector system.

The router ``g_phi`` is a small MLP whose weights are stored on the
policy. Inference is pure NumPy so the policy has no hard dependency on
a training framework; training the router weights is performed by the
updater's conditional variant (pluggable, not wired into the default
engine).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .base import Policy, PolicyKind, PolicyMetrics, hash_vector


def _cond_id(parent_id: str | None, weights_hash: str, layer: int, model_id: str) -> str:
    h = hashlib.sha256(
        "|".join([str(parent_id), weights_hash, str(layer), model_id]).encode("utf-8")
    )
    return f"pol_{h.hexdigest()[:16]}"


def _default_router(h: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """A small 2-layer ReLU router: hidden -> rank.

    Weights are stored as flat arrays and reshaped at application time.
    """
    z1 = h @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    return a1 @ W2 + b2


@dataclass(frozen=True)
class ConditionalSteeringPolicy(Policy):
    """A versioned conditional low-rank steering policy (v2.5).

    Attributes
    ----------
    basis : np.ndarray  shape ``[hidden, rank]``
        The steering basis ``V``.
    router_W1, router_b1, router_W2, router_b2 : np.ndarray
        Weights of the router network ``g_phi`` (hidden -> hidden_router
        -> rank).
    """

    policy_id: str
    parent_policy_id: str | None
    model_id: str
    tokenizer_hash: str
    layer: int
    basis: np.ndarray  # [hidden, rank]
    router_W1: np.ndarray
    router_b1: np.ndarray
    router_W2: np.ndarray
    router_b2: np.ndarray
    threshold: float = 0.0
    hidden_size: int = 0
    rank: int = 0
    weights_hash: str = ""
    training_experience_ids: tuple[str, ...] = ()
    utility_config_dict: dict[str, float] = field(default_factory=dict)
    validation_metrics: PolicyMetrics | None = None
    random_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: PolicyKind = "conditional"

    def __post_init__(self) -> None:
        V = np.asarray(self.basis, dtype=np.float32)
        if V.ndim != 2:
            raise ValueError("ConditionalSteeringPolicy.basis must be [hidden, rank]")
        object.__setattr__(self, "basis", V)
        object.__setattr__(self, "hidden_size", int(V.shape[0]))
        object.__setattr__(self, "rank", int(V.shape[1]))
        for name in ("router_W1", "router_b1", "router_W2", "router_b2"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        if not self.weights_hash:
            flat = np.concatenate([self.router_W1.reshape(-1), self.router_b1.reshape(-1),
                                   self.router_W2.reshape(-1), self.router_b2.reshape(-1)])
            object.__setattr__(self, "weights_hash", hash_vector(flat))

    def coefficients(self, representation: np.ndarray) -> np.ndarray:
        """Compute ``c(x) = g_phi(h(x))`` -> shape ``[rank]``."""
        h = np.asarray(representation, dtype=np.float32)
        return _default_router(h, self.router_W1, self.router_b1, self.router_W2, self.router_b2)

    def delta(self, representation: np.ndarray) -> np.ndarray:
        """``delta_h(x) = V c(x)``."""
        c = self.coefficients(representation)
        return self.basis @ c

    def apply(self, representation: np.ndarray) -> np.ndarray:
        h = np.asarray(representation, dtype=np.float32)
        return h + self.delta(h)

    def route_score(self, representation: np.ndarray) -> dict[str, float]:
        d = self.delta(np.asarray(representation, dtype=np.float32))
        dnorm = float(np.linalg.norm(d)) or 1.0
        proj = float(np.dot(np.asarray(representation, dtype=np.float32), d) / dnorm)
        return {"symbolic": proj, "llm": -proj}

    def vector_norm(self) -> float:
        return float(np.linalg.norm(self.basis))

    def displacement_from(self, other: "ConditionalSteeringPolicy") -> float:
        return float(np.linalg.norm(self.basis - other.basis))


__all__ = ["ConditionalSteeringPolicy"]
