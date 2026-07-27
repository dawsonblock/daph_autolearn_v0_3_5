"""Single static steering-vector policy (AutoLearn v2, Phase 6 Stage A-C).

Represents the compatibility-preserving steering vector as a versioned
policy object::

    h' = h + alpha * v

Stage B (incremental updates) and Stage C (reward-gap supervision) are
implemented in :mod:`daph_learning.autolearn_v2.updater`; this module
defines the immutable policy record and its application semantics.

The policy is a frozen dataclass with deterministic IDs and parent
lineage so the :class:`~daph_learning.autolearn_v2.registry.PolicyRegistry`
can retain immutable copies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Policy, PolicyKind, PolicyMetrics, hash_vector


def _policy_id(parent_id: str | None, vector_hash: str, layer: int, model_id: str) -> str:
    """Deterministic policy ID from lineage + content."""
    h = hashlib.sha256(
        "|".join([str(parent_id), vector_hash, str(layer), model_id]).encode("utf-8")
    )
    return f"pol_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class SingleVectorPolicy(Policy):
    """A versioned single steering-vector policy.

    Attributes
    ----------
    policy_id : str
        Deterministic ID derived from parent + vector hash + layer + model.
    parent_policy_id : str | None
        ``None`` for the root/seed policy.
    model_id, tokenizer_hash : str
        Provenance: the model/tokenizer this policy was learned against.
        The engine asserts these match the runtime model before applying.
    layer : int
        Transformer layer the vector steers.
    vector : np.ndarray
        Rank-1 steering vector (``hidden_size``,).
    alpha : float
        Steering coefficient.
    threshold : float
        Routing margin threshold: route symbolic when
        ``score_symbolic - score_llm > threshold``.
    hidden_size : int
        Declared hidden dimension; asserted to equal ``vector.shape[0]``.
    vector_hash : str
        SHA-256 of the vector bytes (lineage integrity).
    training_experience_ids : tuple[str, ...]
        IDs of experiences used to produce this candidate.
    replay_sample_hash : str | None
        Hash of the replay sample used (reproducibility).
    utility_config_dict : dict[str, float]
        The exact utility config used (persisted for reproducibility).
    validation_metrics : PolicyMetrics | None
        Metrics measured on the immutable validation set at acceptance.
    random_seed : int
    source_commit : str | None
    metadata : dict[str, Any]
    """

    policy_id: str
    parent_policy_id: str | None
    model_id: str
    tokenizer_hash: str
    layer: int
    vector: np.ndarray
    alpha: float = 1.0
    threshold: float = 0.0
    hidden_size: int = 0
    vector_hash: str = ""
    training_experience_ids: tuple[str, ...] = ()
    replay_sample_hash: str | None = None
    utility_config_dict: dict[str, float] = field(default_factory=dict)
    validation_metrics: PolicyMetrics | None = None
    random_seed: int = 42
    source_commit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: PolicyKind = "single_vector"

    def __post_init__(self) -> None:
        arr = np.asarray(self.vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("SingleVectorPolicy.vector must be rank-1")
        # Replace with a canonical float32 array (frozen dataclass workaround).
        object.__setattr__(self, "vector", arr)
        if self.hidden_size and arr.shape[0] != self.hidden_size:
            raise ValueError(
                f"vector dim {arr.shape[0]} != declared hidden_size {self.hidden_size}"
            )
        if not self.vector_hash:
            object.__setattr__(self, "vector_hash", hash_vector(arr))
        if not self.hidden_size:
            object.__setattr__(self, "hidden_size", int(arr.shape[0]))

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, representation: np.ndarray) -> np.ndarray:
        """``h' = h + alpha * v``."""
        h = np.asarray(representation, dtype=np.float32)
        if h.shape[-1] != self.vector.shape[0]:
            raise ValueError(
                f"representation dim {h.shape[-1]} != vector dim {self.vector.shape[0]}"
            )
        return h + self.alpha * self.vector

    def route_score(self, representation: np.ndarray) -> dict[str, float]:
        """Derive per-backend scores from a representation.

        For the single-vector policy the steering direction itself is the
        routing signal: the projection of ``h`` onto ``v`` measures how
        "symbolic-favoring" the representation is. A higher projection ->
        symbolic. The engine converts the margin
        ``proj - threshold`` into a route via the normalizer.
        """
        h = np.asarray(representation, dtype=np.float32)
        v = self.vector
        vnorm = float(np.linalg.norm(v)) or 1.0
        proj = float(np.dot(h, v) / vnorm)
        # Symmetric scores around 0 so the margin is proj - threshold.
        return {"symbolic": proj, "llm": -proj}

    def vector_norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    def displacement_from(self, other: "SingleVectorPolicy") -> float:
        if self.vector.shape[0] != other.vector.shape[0]:
            raise ValueError("cannot compute displacement between mismatched dimensions")
        return float(np.linalg.norm(self.vector - other.vector))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def seed(
        cls,
        *,
        model_id: str,
        tokenizer_hash: str,
        layer: int,
        hidden_size: int,
        alpha: float = 1.0,
        threshold: float = 0.0,
        random_seed: int = 42,
    ) -> "SingleVectorPolicy":
        """Create a zero (identity) seed policy with no parent."""
        vec = np.zeros(hidden_size, dtype=np.float32)
        pid = _policy_id(None, hash_vector(vec), layer, model_id)
        return cls(
            policy_id=pid,
            parent_policy_id=None,
            model_id=model_id,
            tokenizer_hash=tokenizer_hash,
            layer=layer,
            vector=vec,
            alpha=alpha,
            threshold=threshold,
            hidden_size=hidden_size,
            random_seed=random_seed,
            metadata={"origin": "seed", "extraction_method": "zero_seed"},
        )

    @classmethod
    def from_candidate(
        cls,
        parent: "SingleVectorPolicy",
        vector: np.ndarray,
        *,
        training_experience_ids: tuple[str, ...] = (),
        replay_sample_hash: str | None = None,
        utility_config_dict: dict[str, float] | None = None,
        source_commit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SingleVectorPolicy":
        """Build a candidate child policy from a parent and a new vector."""
        vec = np.asarray(vector, dtype=np.float32)
        vh = hash_vector(vec)
        pid = _policy_id(parent.policy_id, vh, parent.layer, parent.model_id)
        return cls(
            policy_id=pid,
            parent_policy_id=parent.policy_id,
            model_id=parent.model_id,
            tokenizer_hash=parent.tokenizer_hash,
            layer=parent.layer,
            vector=vec,
            alpha=parent.alpha,
            threshold=parent.threshold,
            hidden_size=parent.hidden_size,
            vector_hash=vh,
            training_experience_ids=training_experience_ids,
            replay_sample_hash=replay_sample_hash,
            utility_config_dict=dict(utility_config_dict or {}),
            random_seed=parent.random_seed,
            source_commit=source_commit,
            metadata=dict(metadata or {}),
        )

    def with_metrics(self, metrics: PolicyMetrics) -> "SingleVectorPolicy":
        """Return a copy with validation metrics attached (at acceptance)."""
        return SingleVectorPolicy(
            policy_id=self.policy_id,
            parent_policy_id=self.parent_policy_id,
            model_id=self.model_id,
            tokenizer_hash=self.tokenizer_hash,
            layer=self.layer,
            vector=self.vector,
            alpha=self.alpha,
            threshold=self.threshold,
            hidden_size=self.hidden_size,
            vector_hash=self.vector_hash,
            training_experience_ids=self.training_experience_ids,
            replay_sample_hash=self.replay_sample_hash,
            utility_config_dict=dict(self.utility_config_dict),
            validation_metrics=metrics,
            random_seed=self.random_seed,
            source_commit=self.source_commit,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "policy_id": self.policy_id,
            "parent_policy_id": self.parent_policy_id,
            "model_id": self.model_id,
            "tokenizer_hash": self.tokenizer_hash,
            "layer": self.layer,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "hidden_size": self.hidden_size,
            "vector_hash": self.vector_hash,
            "vector": self.vector.tolist(),
            "training_experience_ids": list(self.training_experience_ids),
            "replay_sample_hash": self.replay_sample_hash,
            "utility_config_dict": dict(self.utility_config_dict),
            "validation_metrics": (
                self.validation_metrics.__dict__ if self.validation_metrics else None
            ),
            "random_seed": self.random_seed,
            "source_commit": self.source_commit,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SingleVectorPolicy":
        metrics = None
        if d.get("validation_metrics"):
            metrics = PolicyMetrics(**d["validation_metrics"])
        return cls(
            policy_id=d["policy_id"],
            parent_policy_id=d["parent_policy_id"],
            model_id=d["model_id"],
            tokenizer_hash=d["tokenizer_hash"],
            layer=d["layer"],
            vector=np.asarray(d["vector"], dtype=np.float32),
            alpha=d["alpha"],
            threshold=d["threshold"],
            hidden_size=d["hidden_size"],
            vector_hash=d.get("vector_hash", ""),
            training_experience_ids=tuple(d.get("training_experience_ids", [])),
            replay_sample_hash=d.get("replay_sample_hash"),
            utility_config_dict=dict(d.get("utility_config_dict", {})),
            validation_metrics=metrics,
            random_seed=d.get("random_seed", 42),
            source_commit=d.get("source_commit"),
            metadata=dict(d.get("metadata", {})),
        )


__all__ = ["SingleVectorPolicy"]
