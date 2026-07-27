"""First-class experience representation (AutoLearn v2, Phase 2).

An :class:`Experience` records enough information to reproduce *why* a
learning decision was made: which backends ran, what they produced, how
each output was verified, the measured reward for each backend, the
derived optimal action and reward gap, and the policy/model/dataset
provenance under which it was collected.

IDs and fingerprints are deterministic so that replay deduplication and
lineage checks are reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from daph_learning.verification import VerificationResult, VerificationStatus


@dataclass(frozen=True)
class BackendOutcome:
    """The measured outcome of running one backend on one task.

    Attributes
    ----------
    backend : str
        ``"symbolic"`` or ``"llm"``.
    output : Any
        The raw output produced by the backend (string, structured
        result, or ``None`` on failure).
    verification_status : VerificationStatus
    correctness_score : float | None
        ``1.0`` / ``0.0`` / ``None`` per the verification contract.
    latency_ms : float
        Wall-clock latency in milliseconds.
    estimated_cost : float
        Relative cost estimate (e.g. token cost for LLM, 0 for symbolic).
    confidence : float | None
        Optional backend-reported confidence in ``[0, 1]``.
    failure_type : str | None
        ``None`` on success; otherwise a short failure code
        (``"execution_error"``, ``"timeout"``, ``"unsupported_capability"``).
    metadata : Mapping[str, Any]
        Free-form provenance (e.g. symbolic executor reason code).
    """

    backend: str
    output: Any
    verification_status: VerificationStatus
    correctness_score: float | None
    latency_ms: float
    estimated_cost: float
    confidence: float | None = None
    failure_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_verification(
        cls,
        backend: str,
        output: Any,
        verification: VerificationResult,
        *,
        latency_ms: float = 0.0,
        estimated_cost: float = 0.0,
        confidence: float | None = None,
        failure_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BackendOutcome":
        return cls(
            backend=backend,
            output=output,
            verification_status=verification.status,
            correctness_score=verification.correctness_score,
            latency_ms=float(latency_ms),
            estimated_cost=float(estimated_cost),
            confidence=confidence,
            failure_type=failure_type,
            metadata=dict(metadata or {}),
        )


def fingerprint_task(task: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 fingerprint of a task's identity-relevant fields.

    Fields that define the task are hashed. ``expected`` is included
    because two tasks with the same specification but different answers
    are genuinely different tasks for deduplication purposes. Note that
    this means the fingerprint contains the answer; callers using it for
    leak detection should be aware of this. Routing labels and oracle
    fields (e.g. ``utility_oracle``) are excluded.
    """
    identity_keys = (
        "task_id",
        "capability_ids",
        "inputs",
        "specification",
        "expected",
    )
    payload = {k: task.get(k) for k in identity_keys}
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_experience_id(
    task_fingerprint: str,
    policy_id: str,
    model_id: str,
    dataset_id: str,
    created_at: str,
) -> str:
    """Deterministic experience ID from its scientific identity.

    The ``created_at`` timestamp is *not* included in the hash because it
    is provenance metadata (when the experience was collected), not
    scientific identity (what was collected). Excluding it guarantees
    that re-running the same experiment with the same seed produces the
    same experience IDs, which is required for reproducible replay
    sampling and policy lineage.
    """
    h = hashlib.sha256(
        "|".join([task_fingerprint, policy_id, model_id, dataset_id]).encode("utf-8")
    )
    return f"exp_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class Experience:
    """A single counterfactual experience record.

    Stores enough information to reproduce why a learning decision
    occurred: the task, the policy/model/dataset under which it was
    collected, every backend outcome, the per-backend rewards, the
    derived optimal action and reward gap, and free-form metadata.
    """

    experience_id: str
    task_id: str
    task_fingerprint: str
    task_text: str
    representation_ref: str | None

    policy_id: str
    model_id: str
    dataset_id: str
    selected_action: str
    outcomes: Mapping[str, BackendOutcome]
    backend_rewards: Mapping[str, float]
    optimal_action: str | None
    reward_gap: float | None
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (for checkpoints/telemetry)."""
        d = asdict(self)
        # VerificationStatus is a str enum -> serializes as its value.
        d["outcomes"] = {
            k: {**asdict(v), "verification_status": v.verification_status.value}
            for k, v in self.outcomes.items()
        }
        return d


__all__ = [
    "BackendOutcome",
    "Experience",
    "fingerprint_task",
    "make_experience_id",
]
