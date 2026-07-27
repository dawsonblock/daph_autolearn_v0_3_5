"""Structured iteration telemetry (AutoLearn v2, Phase 15).

For each learning iteration the engine emits a machine-readable record
with:

* active policy ID + candidate policy ID
* replay size + new experience count
* symbolic / LLM / abstain route rates
* counterfactual disagreement rate
* mean reward gap
* utility before / after
* candidate acceptance
* vector norm + update norm
* trust-region clipping rate
* per-domain metrics
* regression count

Records are appended to a JSONL file so no critical scientific result
exists only in console output.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO


@dataclass
class IterationTelemetry:
    """One iteration's telemetry record."""

    iteration: int
    active_policy_id: str
    candidate_policy_id: str | None
    replay_size: int
    new_experience_count: int
    symbolic_route_rate: float
    llm_route_rate: float
    abstain_rate: float
    counterfactual_disagreement_rate: float
    mean_reward_gap: float | None
    utility_before: float
    utility_after: float
    candidate_accepted: bool
    reason_codes: list[str] = field(default_factory=list)
    vector_norm: float = 0.0
    update_norm: float = 0.0
    trust_region_clipped: bool = False
    max_vector_norm_clipped: bool = False
    per_domain: dict[str, float] = field(default_factory=dict)
    regression_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryLogger:
    """Append-only JSONL telemetry sink."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._fh: TextIO | None = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def log(self, record: IterationTelemetry) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def counterfactual_disagreement_rate(experiences: list[Any]) -> float:
    """Fraction of counterfactual experiences where the two backends disagree.

    "Disagree" = different verification status (one CORRECT, the other
    not) or different correctness scores.
    """
    from .experience import Experience

    if not experiences:
        return 0.0
    n_cf = 0
    n_disagree = 0
    for exp in experiences:
        if "symbolic" not in exp.outcomes or "llm" not in exp.outcomes:
            continue
        n_cf += 1
        s = exp.outcomes["symbolic"].verification_status
        l = exp.outcomes["llm"].verification_status
        if s != l:
            n_disagree += 1
    if n_cf == 0:
        return 0.0
    return n_disagree / n_cf


__all__ = [
    "IterationTelemetry",
    "TelemetryLogger",
    "counterfactual_disagreement_rate",
]
