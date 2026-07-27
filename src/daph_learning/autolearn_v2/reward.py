"""Utility / reward model (AutoLearn v2, Phase 4).

An explicit, configurable, independently-testable reward function. The
reward is *not* hidden inside training code; it is a pure function of a
:class:`BackendOutcome` and a :class:`UtilityConfig`.

Default utility (correctness dominates)::

    R(a, x) = w_correctness * correctness
              - w_failure     * failure_penalty
              - w_latency     * normalized_latency
              - w_cost        * normalized_cost
              - w_uncertainty * uncertainty

``correctness`` is ``1.0`` for CORRECT, ``0.0`` for INCORRECT /
EXECUTION_ERROR / TIMEOUT, and ``0.0`` for UNVERIFIABLE (UNVERIFIABLE
never contributes positive correctness reward). ``failure_penalty`` is
``1.0`` when the backend failed (EXECUTION_ERROR / TIMEOUT), else ``0.0``.

The exact :class:`UtilityConfig` is persisted with every experiment so
the reward function is reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from daph_learning.verification import VerificationStatus

from .experience import BackendOutcome

# Backends that can be counterfactually evaluated.
BACKENDS: tuple[str, ...] = ("symbolic", "llm")


@dataclass(frozen=True)
class UtilityConfig:
    """Configuration for the reward function.

    Defaults make correctness dominate: ``correctness_weight=1.0`` while
    latency/cost/uncertainty weights are small. ``abstain_margin`` defines
    the reward-gap band within which the optimal action is ABSTAIN.
    """

    correctness_weight: float = 1.0
    failure_penalty: float = 1.0
    latency_weight: float = 0.05
    cost_weight: float = 0.02
    uncertainty_weight: float = 0.05
    abstain_margin: float = 0.05
    # Normalization references for latency/cost. These are *reference*
    # scales, not learned parameters; they keep the reward scale stable
    # across runs. If unset, per-experience normalization is used.
    latency_ref_ms: float = 1000.0
    cost_ref: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "correctness_weight",
            "failure_penalty",
            "latency_weight",
            "cost_weight",
            "uncertainty_weight",
            "abstain_margin",
            "latency_ref_ms",
            "cost_ref",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f"UtilityConfig.{name} must be a non-negative number; got {v!r}")
        if self.cost_ref <= 0:
            raise ValueError("UtilityConfig.cost_ref must be > 0")
        if self.latency_ref_ms <= 0:
            raise ValueError("UtilityConfig.latency_ref_ms must be > 0")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _correctness(outcome: BackendOutcome) -> float:
    """Correctness signal in ``[0, 1]``. UNVERIFIABLE -> 0.0 (no credit)."""
    if outcome.correctness_score is not None:
        return float(outcome.correctness_score)
    return 0.0


def _failure_indicator(outcome: BackendOutcome) -> float:
    return 1.0 if outcome.verification_status in (
        VerificationStatus.EXECUTION_ERROR,
        VerificationStatus.TIMEOUT,
    ) else 0.0


def _uncertainty(outcome: BackendOutcome) -> float:
    """Uncertainty in ``[0, 1]``.

    UNVERIFIABLE outcomes carry full uncertainty (1.0). A backend-reported
    confidence in ``[0, 1]`` contributes ``1 - confidence``. Otherwise 0.
    """
    if outcome.verification_status is VerificationStatus.UNVERIFIABLE:
        return 1.0
    if outcome.confidence is not None:
        return max(0.0, 1.0 - float(outcome.confidence))
    return 0.0


def backend_reward(outcome: BackendOutcome, config: UtilityConfig) -> float:
    """Compute the reward ``R(a, x)`` for a single backend outcome.

    Pure function of (outcome, config). Independently testable.
    """
    correctness = _correctness(outcome)
    failure = _failure_indicator(outcome)
    latency = min(1.0, outcome.latency_ms / config.latency_ref_ms)
    cost = min(1.0, outcome.estimated_cost / config.cost_ref) if outcome.estimated_cost > 0 else 0.0
    uncertainty = _uncertainty(outcome)

    return float(
        config.correctness_weight * correctness
        - config.failure_penalty * failure
        - config.latency_weight * latency
        - config.cost_weight * cost
        - config.uncertainty_weight * uncertainty
    )


def reward_gap(
    outcomes: Mapping[str, BackendOutcome],
    config: UtilityConfig,
) -> float | None:
    """``reward_gap(x) = R_symbolic(x) - R_llm(x)``.

    Returns ``None`` when either backend is missing (the experience is not
    counterfactual). Positive gap -> SYMBOLIC is better; negative -> LLM.
    """
    if "symbolic" not in outcomes or "llm" not in outcomes:
        return None
    r_sym = backend_reward(outcomes["symbolic"], config)
    r_llm = backend_reward(outcomes["llm"], config)
    return float(r_sym - r_llm)


def optimal_action(
    outcomes: Mapping[str, BackendOutcome],
    config: UtilityConfig,
) -> str | None:
    """Derive the optimal action from measured backend rewards.

    Returns ``"symbolic"`` if ``gap > abstain_margin``, ``"llm"`` if
    ``gap < -abstain_margin``, and ``"abstain"`` inside the band. Returns
    ``None`` when the experience is not counterfactual (a backend is
    missing).
    """
    gap = reward_gap(outcomes, config)
    if gap is None:
        return None
    if gap > config.abstain_margin:
        return "symbolic"
    if gap < -config.abstain_margin:
        return "llm"
    return "abstain"


def reward_gap_from_rewards(
    rewards: Mapping[str, float],
) -> float | None:
    """``R_symbolic - R_llm`` from a precomputed reward mapping."""
    if "symbolic" not in rewards or "llm" not in rewards:
        return None
    return float(rewards["symbolic"] - rewards["llm"])


__all__ = [
    "BACKENDS",
    "UtilityConfig",
    "backend_reward",
    "optimal_action",
    "reward_gap",
    "reward_gap_from_rewards",
]
