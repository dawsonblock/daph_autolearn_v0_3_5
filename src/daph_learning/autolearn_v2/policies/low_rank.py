"""Multi-vector steering policy (AutoLearn v2, Phase 6 Stage D interface).

A stack of steering vectors ``[v1, ..., vr]`` with per-vector coefficients::

    h' = h + sum_i alpha_i * v_i

This is the simplest extension beyond a single vector and is the
interface precursor to the full conditional policy. It is *not* the
default algorithm; the engine uses :class:`SingleVectorPolicy` until a
multi-vector variant consistently beats it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Policy, PolicyKind, PolicyMetrics, hash_vector


def _multi_id(parent_id: str | None, matrix_hash: str, layer: int, model_id: str) -> str:
    h = hashlib.sha256(
        "|".join([str(parent_id), matrix_hash, str(layer), model_id]).encode("utf-8")
    )
    return f"pol_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class MultiVectorPolicy(Policy):
    """A versioned multi-vector steering policy.

    ``vectors`` is a ``[rank, hidden]`` matrix; ``coefficients`` is a
    ``[rank]`` vector. The applied steering is
    ``h' = h + sum_i coefficients[i] * vectors[i]``.
    """

    policy_id: str
    parent_policy_id: str | None
    model_id: str
    tokenizer_hash: str
    layer: int
    vectors: np.ndarray  # [rank, hidden]
    coefficients: np.ndarray  # [rank]
    threshold: float = 0.0
    hidden_size: int = 0
    rank: int = 0
    matrix_hash: str = ""
    training_experience_ids: tuple[str, ...] = ()
    utility_config_dict: dict[str, float] = field(default_factory=dict)
    validation_metrics: PolicyMetrics | None = None
    random_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: PolicyKind = "multi_vector"

    def __post_init__(self) -> None:
        V = np.asarray(self.vectors, dtype=np.float32)
        c = np.asarray(self.coefficients, dtype=np.float32)
        if V.ndim != 2:
            raise ValueError("MultiVectorPolicy.vectors must be rank-2 [rank, hidden]")
        if c.ndim != 1 or c.shape[0] != V.shape[0]:
            raise ValueError("coefficients must be [rank] matching vectors")
        object.__setattr__(self, "vectors", V)
        object.__setattr__(self, "coefficients", c)
        object.__setattr__(self, "rank", int(V.shape[0]))
        if not self.hidden_size:
            object.__setattr__(self, "hidden_size", int(V.shape[1]))
        if self.hidden_size and V.shape[1] != self.hidden_size:
            raise ValueError("vectors hidden dim != declared hidden_size")
        if not self.matrix_hash:
            object.__setattr__(self, "matrix_hash", hash_vector(V.reshape(-1)))

    def composite_vector(self) -> np.ndarray:
        """The single equivalent steering vector ``sum_i c_i v_i``."""
        return (self.coefficients[:, None] * self.vectors).sum(axis=0)

    def apply(self, representation: np.ndarray) -> np.ndarray:
        h = np.asarray(representation, dtype=np.float32)
        return h + self.composite_vector()

    def route_score(self, representation: np.ndarray) -> dict[str, float]:
        v = self.composite_vector()
        vnorm = float(np.linalg.norm(v)) or 1.0
        proj = float(np.dot(np.asarray(representation, dtype=np.float32), v) / vnorm)
        return {"symbolic": proj, "llm": -proj}

    def vector_norm(self) -> float:
        return float(np.linalg.norm(self.composite_vector()))

    def displacement_from(self, other: "MultiVectorPolicy") -> float:
        return float(np.linalg.norm(self.composite_vector() - other.composite_vector()))


__all__ = ["MultiVectorPolicy"]
