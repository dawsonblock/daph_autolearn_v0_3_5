"""Learning state checkpointing (AutoLearn v2, Phase 9).

AutoLearn must resume safely after interruption. The checkpoint captures:

* active policy
* candidate state (last candidate + decision)
* replay buffer
* learning iteration
* RNG state
* training metrics
* accepted/rejected history

Atomic writes: ``temp file -> fsync -> rename`` so partial/corrupt
checkpoints never replace a good one.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .acceptance import CandidateDecision
from .policies.static_vector import SingleVectorPolicy
from .replay import ReplayBuffer


@dataclass
class CheckpointState:
    """Serializable snapshot of the learning state."""

    iteration: int
    active_policy: SingleVectorPolicy
    last_candidate: SingleVectorPolicy | None = None
    last_decision: CandidateDecision | None = None
    replay: ReplayBuffer | None = None
    rng_state: tuple[Any, ...] | None = None
    training_metrics: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    config_dict: dict[str, Any] = field(default_factory=dict)
    saved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "active_policy": self.active_policy.to_dict(),
            "last_candidate": self.last_candidate.to_dict() if self.last_candidate is not None else None,
            "last_decision": self.last_decision.to_dict() if self.last_decision is not None else None,
            "replay": self.replay.to_dict() if self.replay is not None else None,
            "rng_state": list(self.rng_state) if self.rng_state is not None else None,
            "training_metrics": list(self.training_metrics),
            "history": list(self.history),
            "config_dict": dict(self.config_dict),
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointState":
        replay = ReplayBuffer.from_dict(d["replay"]) if d.get("replay") is not None else None
        last_candidate = (
            SingleVectorPolicy.from_dict(d["last_candidate"])
            if d.get("last_candidate") is not None else None
        )
        last_decision = None
        if d.get("last_decision"):
            dec = d["last_decision"]
            from .policies.base import PolicyMetrics
            active_m = PolicyMetrics(**dec["active_metrics"]) if dec.get("active_metrics") else None
            cand_m = PolicyMetrics(**dec["candidate_metrics"]) if dec.get("candidate_metrics") else None
            last_decision = CandidateDecision(
                accepted=dec["accepted"],
                reason_codes=list(dec.get("reason_codes", [])),
                active_metrics=active_m,
                candidate_metrics=cand_m,
                utility_gain=dec.get("utility_gain", 0.0),
                accuracy_delta=dec.get("accuracy_delta", 0.0),
                abstain_delta=dec.get("abstain_delta", 0.0),
                displacement=dec.get("displacement", 0.0),
                detail=dec.get("detail"),
            )
        rng_state = None
        if d.get("rng_state") is not None:
            # JSON converts the inner tuple to a list; reconstruct it.
            raw = d["rng_state"]
            if isinstance(raw, list) and len(raw) == 3 and isinstance(raw[1], list):
                rng_state = (int(raw[0]), tuple(int(x) for x in raw[1]), raw[2])
            else:
                rng_state = tuple(raw)
        return cls(
            iteration=d["iteration"],
            active_policy=SingleVectorPolicy.from_dict(d["active_policy"]),
            last_candidate=last_candidate,
            last_decision=last_decision,
            replay=replay,
            rng_state=rng_state,
            training_metrics=list(d.get("training_metrics", [])),
            history=list(d.get("history", [])),
            config_dict=dict(d.get("config_dict", {})),
            saved_at=d.get("saved_at", ""),
        )


def save_checkpoint(state: CheckpointState, path: str) -> None:
    """Atomically write a checkpoint to ``path``.

    Writes to ``path + ".tmp"``, fsyncs, then ``os.replace`` onto
    ``path`` so a crash never leaves a partial/corrupt checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: str) -> CheckpointState:
    """Load a checkpoint from ``path``.

    Raises :class:`FileNotFoundError` if absent. The caller may then
    resume the engine from the returned state.
    """
    with open(path, "r", encoding="utf-8") as f:
        return CheckpointState.from_dict(json.load(f))


def capture_rng_state(rng: random.Random) -> tuple[Any, ...]:
    """Capture a :class:`random.Random` RNG state for the checkpoint."""
    return rng.getstate()


def restore_rng_state(rng: random.Random, state: tuple[Any, ...]) -> None:
    """Restore a :class:`random.Random` RNG state from a checkpoint."""
    rng.setstate(state)


__all__ = [
    "CheckpointState",
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]
