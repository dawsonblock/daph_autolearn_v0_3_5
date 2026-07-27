"""Incremental steering updater (AutoLearn v2, Phase 6 Stage B/C).

Moves beyond the v0.3.x ``v = mean(pos) - mean(neg)`` replacement. The
updater produces a *candidate* update of the form::

    v_candidate = v_current + eta * delta_v

with a trust-region constraint::

    ||v_candidate - v_current|| <= epsilon

Stage C: the update direction ``delta_v`` is derived from *measured
reward gaps*, not arbitrary positive/negative labels. Given
representations ``h_i`` and reward gaps ``g_i = R_symbolic - R_llm``, the
objective is::

    J(v) = mean_i [ g_i * route_margin(h_i, v) ]
           - lambda_reg * ||v - v_parent||^2

where ``route_margin(h, v) = proj_v(h) - threshold``. The gradient pushes
``v`` toward representations where symbolic is better (positive gap) and
away from representations where LLM is better (negative gap).

The candidate is *not* deployed automatically; it is returned to the
engine, which passes it through the acceptance gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .experience import Experience
from .policies.static_vector import SingleVectorPolicy


@dataclass(frozen=True)
class UpdateConfig:
    """Configuration for the incremental steering update."""

    learning_rate: float = 0.05
    trust_region_radius: float = 0.10
    max_vector_norm: float = 20.0
    lambda_reg: float = 0.01
    # Coefficient bounds for the policy's alpha (applied at policy build).
    alpha_min: float = 0.0
    alpha_max: float = 10.0

    def __post_init__(self) -> None:
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be >= 0")
        if self.trust_region_radius < 0:
            raise ValueError("trust_region_radius must be >= 0")
        if self.max_vector_norm <= 0:
            raise ValueError("max_vector_norm must be > 0")
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be >= 0")


def trust_region_clip(
    candidate: np.ndarray,
    current: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, bool, float]:
    """Apply the trust-region constraint ``||candidate - current|| <= radius``.

    Returns ``(clipped_candidate, was_clipped, step_norm)``. If the raw
    step exceeds ``radius``, it is scaled to exactly ``radius``.
    """
    delta = np.asarray(candidate, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    step_norm = float(np.linalg.norm(delta))
    if radius <= 0:
        # No step allowed: stay at current.
        return np.asarray(current, dtype=np.float32), step_norm > 0, 0.0
    if step_norm > radius:
        delta = delta * (radius / step_norm)
        return (np.asarray(current, dtype=np.float32) + delta), True, radius
    return np.asarray(candidate, dtype=np.float32), False, step_norm


def _route_margin(h: np.ndarray, v: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorized ``proj_v(h) - threshold`` for a batch of representations."""
    vnorm = float(np.linalg.norm(v)) or 1.0
    proj = (h @ v) / vnorm
    return proj - threshold


def compute_candidate_update(
    parent: SingleVectorPolicy,
    experiences: Sequence[Experience],
    representations: np.ndarray,
    config: UpdateConfig,
    *,
    replay_sample_hash: str | None = None,
    utility_config_dict: dict[str, float] | None = None,
) -> tuple[SingleVectorPolicy, dict[str, float]]:
    """Produce a candidate child policy from reward-gap supervision.

    Parameters
    ----------
    parent : SingleVectorPolicy
        The currently accepted policy.
    experiences : sequence of Experience
        The replay sample (must align 1:1 with ``representations`` rows).
    representations : np.ndarray  shape ``[N, hidden]``
        Captured representations for each experience.
    config : UpdateConfig
    replay_sample_hash, utility_config_dict : provenance

    Returns
    -------
    (candidate_policy, update_metadata)
        ``update_metadata`` records the step norm, trust-region clipping,
        objective value, and constraint diagnostics for telemetry.
    """
    H = np.asarray(representations, dtype=np.float32)
    if H.ndim != 2:
        raise ValueError("representations must be [N, hidden]")
    if H.shape[0] != len(experiences):
        raise ValueError(
            f"representations rows {H.shape[0]} != experiences {len(experiences)}"
        )
    if H.shape[1] != parent.hidden_size:
        raise ValueError(
            f"representation dim {H.shape[1]} != policy hidden_size {parent.hidden_size}"
        )

    v_current = np.asarray(parent.vector, dtype=np.float32)
    threshold = parent.threshold

    # Reward-gap supervision signal. Only counterfactual experiences
    # (reward_gap is not None) contribute; others are skipped.
    gaps = np.zeros(len(experiences), dtype=np.float32)
    mask = np.zeros(len(experiences), dtype=bool)
    for i, exp in enumerate(experiences):
        if exp.reward_gap is not None:
            gaps[i] = float(exp.reward_gap)
            mask[i] = True

    n_used = int(mask.sum())
    if n_used == 0:
        # No counterfactual signal: candidate == parent (no-op update).
        candidate = SingleVectorPolicy.from_candidate(
            parent, v_current,
            training_experience_ids=tuple(e.experience_id for e in experiences),
            replay_sample_hash=replay_sample_hash,
            utility_config_dict=utility_config_dict,
            metadata={"origin": "no_counterfactual_signal", "n_used": 0},
        )
        return candidate, {
            "n_used": 0,
            "step_norm": 0.0,
            "trust_region_clipped": False,
            "objective_before": 0.0,
            "objective_after": 0.0,
            "vector_norm_before": float(np.linalg.norm(v_current)),
            "vector_norm_after": float(np.linalg.norm(v_current)),
            "max_vector_norm_clipped": False,
        }

    H_used = H[mask]
    g_used = gaps[mask]

    # Objective: J(v) = mean_i [ g_i * margin_i ] - lambda ||v - v_parent||^2
    # dJ/dv = mean_i [ g_i * d margin_i / dv ] - 2 lambda (v - v_parent)
    # margin_i = (h_i . v) / ||v|| - threshold
    # For a fixed-norm update direction we use the simpler proxy
    # margin_i ~ h_i . v_hat  where v_hat = v / ||v||, and take the
    # gradient w.r.t. v as  g_i * h_i (the direction that increases the
    # margin for positive gaps and decreases it for negative gaps).
    vnorm = float(np.linalg.norm(v_current)) or 1.0
    v_hat = v_current / vnorm

    # Gradient of the (unregularized) objective w.r.t. v_hat:
    #   d/dv_hat mean_i g_i (h_i . v_hat) = mean_i g_i h_i
    grad = (g_used[:, None] * H_used).mean(axis=0)

    # Pull-back to v-space: scale by 1/||v|| so the step is in v units.
    grad_v = grad / vnorm

    # Regularization toward parent: -2 lambda (v - v_parent). For a
    # candidate whose parent is the current vector, this term is zero at
    # the start and grows as v moves away, keeping updates conservative.
    reg = -2.0 * config.lambda_reg * (v_current - v_current)  # zero by construction
    grad_v = grad_v + reg

    # Normalize the gradient direction so eta controls the absolute step.
    grad_norm = float(np.linalg.norm(grad_v))
    if grad_norm > 0:
        direction = grad_v / grad_norm
    else:
        direction = np.zeros_like(v_current)

    raw_candidate = v_current + config.learning_rate * direction

    # Trust-region constraint.
    clipped, was_clipped, step_norm = trust_region_clip(
        raw_candidate, v_current, config.trust_region_radius
    )

    # Max vector norm constraint.
    cand_norm = float(np.linalg.norm(clipped))
    max_clipped = False
    if cand_norm > config.max_vector_norm:
        clipped = clipped * (config.max_vector_norm / cand_norm)
        max_clipped = True
        step_norm = float(np.linalg.norm(clipped - v_current))

    # Objective values for telemetry.
    def objective(v: np.ndarray) -> float:
        vh = v / (float(np.linalg.norm(v)) or 1.0)
        margins = H_used @ vh - threshold
        return float((g_used * margins).mean()) - config.lambda_reg * float(
            np.sum((v - v_current) ** 2)
        )

    obj_before = objective(v_current)
    obj_after = objective(clipped)

    candidate = SingleVectorPolicy.from_candidate(
        parent, clipped,
        training_experience_ids=tuple(e.experience_id for e in experiences),
        replay_sample_hash=replay_sample_hash,
        utility_config_dict=utility_config_dict,
        metadata={
            "origin": "reward_gap_update",
            "n_used": n_used,
            "extraction_method": "reward_gap_gradient",
        },
    )

    return candidate, {
        "n_used": n_used,
        "step_norm": step_norm,
        "trust_region_clipped": was_clipped,
        "objective_before": obj_before,
        "objective_after": obj_after,
        "vector_norm_before": float(np.linalg.norm(v_current)),
        "vector_norm_after": float(np.linalg.norm(clipped)),
        "max_vector_norm_clipped": max_clipped,
        "mean_reward_gap": float(g_used.mean()),
    }


def hash_replay_sample(experiences: Sequence[Experience]) -> str:
    """Deterministic hash of a replay sample (for lineage/reproducibility)."""
    h = hashlib.sha256()
    for exp in experiences:
        h.update(exp.experience_id.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


__all__ = [
    "UpdateConfig",
    "compute_candidate_update",
    "hash_replay_sample",
    "trust_region_clip",
]
