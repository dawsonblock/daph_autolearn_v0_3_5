"""Bounded experience replay buffer (AutoLearn v2, Phase 5).

The learner must not train only on the newest failures. The replay
buffer retains a bounded, balanced, prioritized set of experiences and
samples mini-batches with deterministic seeded sampling.

Priority (per the design brief)::

    priority = alpha * abs(reward_gap)
             + beta  * routing_error
             + gamma * boundary_score
             + delta * recency_score
             + epsilon * regression_anchor_weight

Sampling categories are blended so no single task family or action class
dominates:

1. recent examples
2. routing mistakes
3. high reward-gap examples
4. near-boundary examples
5. historical regression anchors
6. balanced backend examples

Supports capacity limits, deterministic seeded sampling, serialization,
checkpoint restore, statistics, and deduplication by task fingerprint.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .experience import BackendOutcome, Experience


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for the replay buffer."""

    capacity: int = 10000
    batch_size: int = 128
    # Priority weights.
    alpha: float = 1.0  # abs(reward_gap)
    beta: float = 1.0  # routing error
    gamma: float = 0.5  # boundary score
    delta: float = 0.25  # recency
    epsilon: float = 0.5  # regression anchor
    # Fractional composition of each sampled batch.
    hard_example_fraction: float = 0.40
    regression_anchor_fraction: float = 0.20
    # Boundary band: |reward_gap| within this margin is "near-boundary".
    boundary_margin: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= self.hard_example_fraction <= 1.0):
            raise ValueError("hard_example_fraction must be in [0, 1]")
        if not (0.0 <= self.regression_anchor_fraction <= 1.0):
            raise ValueError("regression_anchor_fraction must be in [0, 1]")
        if self.hard_example_fraction + self.regression_anchor_fraction > 1.0:
            raise ValueError("hard + regression fractions must sum to <= 1.0")


@dataclass
class ReplaySample:
    """A sampled mini-batch from the replay buffer."""

    experiences: list[Experience]
    indices: list[int]
    priorities: list[float]
    seed: int

    @property
    def size(self) -> int:
        return len(self.experiences)


def _routing_error(exp: Experience) -> float:
    """1.0 when the selected action differs from the optimal action."""
    if exp.optimal_action is None:
        return 0.0
    return 0.0 if exp.selected_action == exp.optimal_action else 1.0


def _boundary_score(exp: Experience, margin: float) -> float:
    """1.0 when |reward_gap| is within the boundary margin, decaying outside."""
    if exp.reward_gap is None:
        return 0.0
    g = abs(exp.reward_gap)
    if g <= margin:
        return 1.0
    # Linear decay to 0 at 2*margin.
    return max(0.0, 1.0 - (g - margin) / max(margin, 1e-9))


def _recency_score(index: int, total: int) -> float:
    """1.0 for the most recent entry, decaying linearly to 0 for the oldest."""
    if total <= 1:
        return 1.0
    return index / (total - 1)


class ReplayBuffer:
    """A bounded, prioritized, deduplicated replay buffer.

    Deduplication is by ``task_fingerprint``: a task already in the buffer
    is *updated* in place (its experience record is replaced with the
    newer one) rather than duplicated. This keeps the buffer focused on
    the most recent measurement per task.
    """

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self.config = config or ReplayConfig()
        self._entries: list[Experience] = []
        self._fingerprints: dict[str, int] = {}  # fingerprint -> index
        self._regression_anchors: set[str] = set()  # experience_ids
        self._rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, experience: Experience) -> None:
        """Add or update an experience (dedup by task fingerprint)."""
        fp = experience.task_fingerprint
        if fp in self._fingerprints:
            idx = self._fingerprints[fp]
            old = self._entries[idx]
            self._entries[idx] = experience
            # Preserve regression-anchor flag if the old entry was one.
            if old.experience_id in self._regression_anchors:
                self._regression_anchors.discard(old.experience_id)
                self._regression_anchors.add(experience.experience_id)
            return
        self._entries.append(experience)
        self._fingerprints[fp] = len(self._entries) - 1
        self._enforce_capacity()

    def add_many(self, experiences: Sequence[Experience]) -> None:
        for exp in experiences:
            self.add(exp)

    def mark_regression_anchor(self, experience_id: str) -> None:
        self._regression_anchors.add(experience_id)

    def _enforce_capacity(self) -> None:
        """Evict oldest entries (FIFO) until within capacity.

        Regression anchors are preserved from eviction when possible so
        historical regression signals persist.
        """
        evicted_any = False
        while len(self._entries) > self.config.capacity:
            # Find the oldest non-anchor entry.
            evict_idx = None
            for i, exp in enumerate(self._entries):
                if exp.experience_id not in self._regression_anchors:
                    evict_idx = i
                    break
            if evict_idx is None:
                # Everything is an anchor; evict the oldest anchor anyway.
                evict_idx = 0
            evicted = self._entries.pop(evict_idx)
            self._regression_anchors.discard(evicted.experience_id)
            evicted_any = True
        if evicted_any:
            # Rebuild fingerprint index once after all evictions.
            self._fingerprints = {
                e.task_fingerprint: i for i, e in enumerate(self._entries)
            }

    # ------------------------------------------------------------------
    # Priority + sampling
    # ------------------------------------------------------------------

    def priority(self, exp: Experience, index: int) -> float:
        """Compute the sampling priority for one entry."""
        cfg = self.config
        gap = abs(exp.reward_gap) if exp.reward_gap is not None else 0.0
        recency = _recency_score(index, len(self._entries))
        boundary = _boundary_score(exp, cfg.boundary_margin)
        anchor = 1.0 if exp.experience_id in self._regression_anchors else 0.0
        return float(
            cfg.alpha * gap
            + cfg.beta * _routing_error(exp)
            + cfg.gamma * boundary
            + cfg.delta * recency
            + cfg.epsilon * anchor
        )

    def _priorities(self) -> list[float]:
        return [self.priority(exp, i) for i, exp in enumerate(self._entries)]

    def sample(self, batch_size: int | None = None, *, seed: int | None = None) -> ReplaySample:
        """Sample a balanced, prioritized mini-batch.

        The batch is composed of:

        * ``hard_example_fraction`` drawn from the highest-priority entries
          (routing mistakes + high reward-gap).
        * ``regression_anchor_fraction`` drawn from regression anchors.
        * the remainder drawn proportionally to priority across the whole
          buffer, balanced so neither symbolic-favoring nor llm-favoring
          examples dominate.

        Sampling is deterministic given the buffer's seed (or the
        per-call ``seed`` override).
        """
        n = batch_size or self.config.batch_size
        rng = random.Random(seed) if seed is not None else self._rng
        total = len(self._entries)
        if total == 0:
            return ReplaySample(experiences=[], indices=[], priorities=[], seed=rng.getstate()[1][0])
        n = min(n, total)

        n_hard = int(round(n * self.config.hard_example_fraction))
        n_anchor = int(round(n * self.config.regression_anchor_fraction))
        n_rest = n - n_hard - n_anchor
        if n_rest < 0:
            n_rest = 0
            n_hard = n - n_anchor

        priorities = self._priorities()
        chosen: set[int] = set()

        # 1. Hard examples: top-priority entries not yet chosen.
        if n_hard > 0:
            ranked = sorted(range(total), key=lambda i: priorities[i], reverse=True)
            for i in ranked:
                if len(chosen) >= n_hard:
                    break
                chosen.add(i)

        # 2. Regression anchors.
        if n_anchor > 0:
            anchor_idxs = [
                i for i, e in enumerate(self._entries)
                if e.experience_id in self._regression_anchors and i not in chosen
            ]
            rng.shuffle(anchor_idxs)
            for i in anchor_idxs:
                if sum(1 for c in chosen if self._entries[c].experience_id in self._regression_anchors) >= n_anchor:
                    break
                chosen.add(i)

        # 3. Rest: priority-proportional, balanced by optimal action.
        if n_rest > 0 and len(chosen) < n:
            remaining = [i for i in range(total) if i not in chosen]
            # Balance by optimal action when available.
            sym_pool = [i for i in remaining if self._entries[i].optimal_action == "symbolic"]
            llm_pool = [i for i in remaining if self._entries[i].optimal_action == "llm"]
            other_pool = [i for i in remaining if i not in sym_pool and i not in llm_pool]
            rng.shuffle(sym_pool)
            rng.shuffle(llm_pool)
            rng.shuffle(other_pool)
            # Round-robin to keep balance.
            pools = [sym_pool, llm_pool, other_pool]
            order: list[int] = []
            while len(order) < n_rest:
                progressed = False
                for p in pools:
                    if p:
                        order.append(p.pop())
                        progressed = True
                        if len(order) >= n_rest:
                            break
                if not progressed:
                    break
            for i in order:
                if len(chosen) >= n:
                    break
                chosen.add(i)

        idxs = sorted(chosen)
        exps = [self._entries[i] for i in idxs]
        prios = [priorities[i] for i in idxs]
        return ReplaySample(
            experiences=exps,
            indices=idxs,
            priorities=prios,
            seed=rng.getstate()[1][0] if isinstance(rng.getstate(), tuple) else self.config.seed,
        )

    # ------------------------------------------------------------------
    # Introspection / serialization
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def regression_anchor_count(self) -> int:
        return len(self._regression_anchors)

    def statistics(self) -> dict[str, Any]:
        """Summary statistics over the buffer contents."""
        n = len(self._entries)
        if n == 0:
            return {
                "size": 0,
                "regression_anchors": 0,
                "mean_reward_gap": None,
                "n_symbolic_optimal": 0,
                "n_llm_optimal": 0,
                "n_abstain_optimal": 0,
                "n_counterfactual": 0,
                "routing_error_rate": 0.0,
            }
        gaps = [e.reward_gap for e in self._entries if e.reward_gap is not None]
        opt = [e.optimal_action for e in self._entries]
        errors = sum(1 for e in self._entries if _routing_error(e) > 0.0)
        return {
            "size": n,
            "regression_anchors": len(self._regression_anchors),
            "mean_reward_gap": float(sum(gaps) / len(gaps)) if gaps else None,
            "n_symbolic_optimal": sum(1 for o in opt if o == "symbolic"),
            "n_llm_optimal": sum(1 for o in opt if o == "llm"),
            "n_abstain_optimal": sum(1 for o in opt if o == "abstain"),
            "n_counterfactual": sum(1 for e in self._entries if e.reward_gap is not None),
            "routing_error_rate": errors / n,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "config": asdict(self.config),
            "entries": [e.to_dict() for e in self._entries],
            "regression_anchors": sorted(self._regression_anchors),
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayBuffer":
        cfg = ReplayConfig(**dict(data["config"]))
        buf = cls(cfg)
        # Re-add entries (dedup handles identity).
        for e_dict in data.get("entries", []):
            buf.add(_experience_from_dict(e_dict))
        for aid in data.get("regression_anchors", []):
            buf._regression_anchors.add(aid)
        # Restore RNG state for reproducible sampling on resume.
        rng_state = data.get("rng_state")
        if rng_state is not None:
            # JSON converts tuples to lists; reconstruct the proper format
            # expected by random.Random.setstate(): (version, internal_tuple, gauss_next)
            if isinstance(rng_state, list) and len(rng_state) == 3:
                version = int(rng_state[0])
                internal = tuple(int(x) for x in rng_state[1]) if isinstance(rng_state[1], list) else rng_state[1]
                gauss = rng_state[2]
                buf._rng.setstate((version, internal, gauss))
            else:
                buf._rng.setstate(rng_state)
        return buf


def _experience_from_dict(d: Mapping[str, Any]) -> Experience:
    """Reconstruct an Experience from its serialized dict."""
    from daph_learning.verification import VerificationStatus

    outcomes = {}
    for b, o in d.get("outcomes", {}).items():
        outcomes[b] = BackendOutcome(
            backend=o["backend"],
            output=o["output"],
            verification_status=VerificationStatus(o["verification_status"]),
            correctness_score=o["correctness_score"],
            latency_ms=o["latency_ms"],
            estimated_cost=o["estimated_cost"],
            confidence=o.get("confidence"),
            failure_type=o.get("failure_type"),
            metadata=o.get("metadata", {}),
        )
    return Experience(
        experience_id=d["experience_id"],
        task_id=d["task_id"],
        task_fingerprint=d["task_fingerprint"],
        task_text=d["task_text"],
        representation_ref=d.get("representation_ref"),
        policy_id=d["policy_id"],
        model_id=d["model_id"],
        dataset_id=d["dataset_id"],
        selected_action=d["selected_action"],
        outcomes=outcomes,
        backend_rewards=dict(d.get("backend_rewards", {})),
        optimal_action=d.get("optimal_action"),
        reward_gap=d.get("reward_gap"),
        created_at=d["created_at"],
        metadata=d.get("metadata", {}),
    )


__all__ = [
    "ReplayBuffer",
    "ReplayConfig",
    "ReplaySample",
]
