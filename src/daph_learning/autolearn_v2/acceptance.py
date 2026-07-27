"""Candidate update acceptance gate (AutoLearn v2, Phase 7).

A candidate policy can **never** automatically replace the active
policy. The engine evaluates both the candidate and the active policy on
the *same untouched validation set* and the gate decides ACCEPT or
REJECT based on:

* total utility
* routing accuracy
* per-domain regression
* abstention rate
* catastrophic route collapse (100% symbolic / 100% LLM)
* steering displacement
* statistical uncertainty / minimum sample count

The gate returns a structured :class:`CandidateDecision` with
``reason_codes`` so rejections are diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .policies.base import PolicyMetrics
from .policies.static_vector import SingleVectorPolicy

# Reason codes (stable strings; persisted in lineage).
REASON_UTILITY_GAIN = "INSUFFICIENT_UTILITY_GAIN"
REASON_ACCURACY_REGRESSION = "ACCURACY_REGRESSION"
REASON_DOMAIN_REGRESSION = "DOMAIN_REGRESSION"
REASON_ROUTE_COLLAPSE = "ROUTE_COLLAPSE"
REASON_ABSTAIN_SPIKE = "ABSTAIN_SPIKE"
REASON_INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
REASON_EXCESSIVE_DISPLACEMENT = "EXCESSIVE_DISPLACEMENT"
REASON_UTILITY_FROM_IMBALANCE = "UTILITY_FROM_IMBALANCE"


@dataclass(frozen=True)
class AcceptanceConfig:
    """Configuration for the candidate acceptance gate."""

    minimum_utility_gain: float = 0.01
    maximum_accuracy_regression: float = 0.01
    maximum_domain_regression: float = 0.03
    minimum_samples: int = 100
    confidence_level: float = 0.95
    # Route-collapse thresholds: a route rate at/above this is collapse.
    route_collapse_rate: float = 0.98
    # Reject if abstention rises by more than this absolute amount.
    max_abstain_increase: float = 0.20
    # Reject if steering displacement exceeds this (trust-region echo).
    max_displacement: float = 1.0
    # Reject if utility gain is explained by class imbalance: if the
    # majority-class rate rises and the minority-class success rate falls.
    imbalance_guard: bool = True

    def __post_init__(self) -> None:
        if self.minimum_utility_gain < 0:
            raise ValueError("minimum_utility_gain must be >= 0")
        if not (0.0 <= self.maximum_accuracy_regression <= 1.0):
            raise ValueError("maximum_accuracy_regression must be in [0, 1]")
        if not (0.0 <= self.maximum_domain_regression <= 1.0):
            raise ValueError("maximum_domain_regression must be in [0, 1]")
        if self.minimum_samples < 0:
            raise ValueError("minimum_samples must be >= 0")
        if not (0.0 <= self.confidence_level <= 1.0):
            raise ValueError("confidence_level must be in [0, 1]")


@dataclass(frozen=True)
class CandidateDecision:
    """The structured result of evaluating a candidate policy."""

    accepted: bool
    reason_codes: list[str] = field(default_factory=list)
    active_metrics: PolicyMetrics | None = None
    candidate_metrics: PolicyMetrics | None = None
    utility_gain: float = 0.0
    accuracy_delta: float = 0.0
    abstain_delta: float = 0.0
    displacement: float = 0.0
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "active_metrics": self.active_metrics.__dict__ if self.active_metrics else None,
            "candidate_metrics": self.candidate_metrics.__dict__ if self.candidate_metrics else None,
            "utility_gain": self.utility_gain,
            "accuracy_delta": self.accuracy_delta,
            "abstain_delta": self.abstain_delta,
            "displacement": self.displacement,
            "detail": self.detail,
        }


def _route_collapse(metrics: PolicyMetrics, threshold: float) -> str | None:
    if metrics.symbolic_rate >= threshold:
        return "symbolic"
    if metrics.llm_rate >= threshold:
        return "llm"
    return None


def evaluate_candidate(
    active: SingleVectorPolicy,
    candidate: SingleVectorPolicy,
    active_metrics: PolicyMetrics,
    candidate_metrics: PolicyMetrics,
    config: AcceptanceConfig,
) -> CandidateDecision:
    """Decide whether ``candidate`` may replace ``active``.

    Both metrics must be measured on the *same* validation set by the
    engine. The gate is pure: it only compares metrics and policy
    content; it never touches the validation data.
    """
    reasons: list[str] = []

    # Minimum sample count.
    if candidate_metrics.n_samples < config.minimum_samples:
        reasons.append(REASON_INSUFFICIENT_SAMPLES)

    # Route collapse.
    collapse = _route_collapse(candidate_metrics, config.route_collapse_rate)
    if collapse is not None:
        reasons.append(REASON_ROUTE_COLLAPSE)

    # Abstention spike.
    abstain_delta = candidate_metrics.abstain_rate - active_metrics.abstain_rate
    if abstain_delta > config.max_abstain_increase:
        reasons.append(REASON_ABSTAIN_SPIKE)

    # Utility gain.
    utility_gain = candidate_metrics.utility - active_metrics.utility
    if utility_gain < config.minimum_utility_gain:
        reasons.append(REASON_UTILITY_GAIN)

    # Accuracy regression.
    accuracy_delta = candidate_metrics.routing_accuracy - active_metrics.routing_accuracy
    if accuracy_delta < -config.maximum_accuracy_regression:
        reasons.append(REASON_ACCURACY_REGRESSION)

    # Per-domain regression.
    domain_regressions: list[str] = []
    for domain, active_score in active_metrics.per_domain.items():
        cand_score = candidate_metrics.per_domain.get(domain)
        if cand_score is None:
            continue
        if cand_score < active_score - config.maximum_domain_regression:
            domain_regressions.append(domain)
    if domain_regressions:
        reasons.append(REASON_DOMAIN_REGRESSION)

    # Steering displacement.
    displacement = candidate.displacement_from(active)
    if displacement > config.max_displacement:
        reasons.append(REASON_EXCESSIVE_DISPLACEMENT)

    # Imbalance guard: utility gain driven by majority-class rate rising
    # while the minority-class success rate falls.
    if config.imbalance_guard and utility_gain >= config.minimum_utility_gain:
        active_dom = active_metrics.per_domain
        cand_dom = candidate_metrics.per_domain
        if active_dom and cand_dom:
            deltas = {
                d: cand_dom.get(d, 0.0) - active_dom.get(d, 0.0)
                for d in set(active_dom) | set(cand_dom)
            }
            if deltas:
                worst = min(deltas.values())
                best = max(deltas.values())
                # If one domain improves a lot while another regresses,
                # the global utility gain may be imbalance-driven.
                if best > 0 and worst < -config.maximum_domain_regression:
                    if REASON_DOMAIN_REGRESSION not in reasons:
                        reasons.append(REASON_UTILITY_FROM_IMBALANCE)

    accepted = len(reasons) == 0
    return CandidateDecision(
        accepted=accepted,
        reason_codes=reasons,
        active_metrics=active_metrics,
        candidate_metrics=candidate_metrics,
        utility_gain=utility_gain,
        accuracy_delta=accuracy_delta,
        abstain_delta=abstain_delta,
        displacement=displacement,
        detail=None if accepted else "; ".join(reasons),
    )


__all__ = [
    "AcceptanceConfig",
    "CandidateDecision",
    "evaluate_candidate",
    "REASON_ABSTAIN_SPIKE",
    "REASON_ACCURACY_REGRESSION",
    "REASON_DOMAIN_REGRESSION",
    "REASON_EXCESSIVE_DISPLACEMENT",
    "REASON_INSUFFICIENT_SAMPLES",
    "REASON_ROUTE_COLLAPSE",
    "REASON_UTILITY_FROM_IMBALANCE",
    "REASON_UTILITY_GAIN",
]
