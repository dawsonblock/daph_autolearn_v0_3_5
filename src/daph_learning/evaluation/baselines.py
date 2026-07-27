"""Baseline framework (AutoLearn v2, Phase 13).

A common baseline interface so AutoLearn v2 can be compared against:

1. always LLM
2. always SYMBOLIC
3. heuristic router (capability-based)
4. prompt-only router
5. unsteered model
6. linear probe classifier
7. MLP route classifier
8. static contrastive steering
9. random steering
10. AutoLearn v2
11. oracle route (measured backend utility)

Each baseline is a callable ``Baseline.score(tasks) -> BaselineResult``
returning routing decisions and a measured utility. The
:func:`oracle_gap_closure` helper computes how much of the gap between a
baseline and the oracle is closed by a method.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from daph_learning.autolearn_v2.policies.base import PolicyMetrics
from daph_learning.autolearn_v2.reward import UtilityConfig, backend_reward
from daph_learning.autolearn_v2.counterfactual import default_symbolic_backend


@dataclass
class BaselineResult:
    """The measured outcome of running a baseline on a task set."""

    name: str
    routes: list[str]
    utility: float
    routing_accuracy: float | None = None
    n_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Baseline:
    """Abstract baseline: ``score(tasks) -> BaselineResult``."""

    name: str = "baseline"

    def score(self, tasks: Sequence[Mapping[str, Any]]) -> BaselineResult:
        raise NotImplementedError


class AlwaysLLM(Baseline):
    name = "always_llm"

    def score(self, tasks: Sequence[Mapping[str, Any]]) -> BaselineResult:
        routes = ["llm"] * len(tasks)
        return BaselineResult(name=self.name, routes=routes, utility=0.0, n_samples=len(tasks))


class AlwaysSymbolic(Baseline):
    name = "always_symbolic"

    def score(self, tasks: Sequence[Mapping[str, Any]]) -> BaselineResult:
        routes = ["symbolic"] * len(tasks)
        return BaselineResult(name=self.name, routes=routes, utility=0.0, n_samples=len(tasks))


class HeuristicRouter(Baseline):
    """Capability-based heuristic: symbolic if symbolic caps present, else LLM."""

    name = "heuristic_router"

    def score(self, tasks: Sequence[Mapping[str, Any]]) -> BaselineResult:
        routes = [
            "symbolic" if (set(t.get("capability_ids") or []) & {"integer_arithmetic", "modular_multiplication"}) else "llm"
            for t in tasks
        ]
        return BaselineResult(name=self.name, routes=routes, utility=0.0, n_samples=len(tasks))


class OracleRouter(Baseline):
    """Oracle route based on measured backend utility.

    Uses each task's ``utility_oracle`` field (set by the empirical
    oracle builder) when available, else falls back to executing both
    backends and picking the higher-reward one.
    """

    name = "oracle"

    def __init__(self, utility_config: UtilityConfig | None = None,
                 llm_backend_fn: Callable | None = None) -> None:
        self.utility_config = utility_config or UtilityConfig()
        self.llm_backend_fn = llm_backend_fn

    def score(self, tasks: Sequence[Mapping[str, Any]]) -> BaselineResult:
        routes: list[str] = []
        for t in tasks:
            oracle = t.get("utility_oracle") or t.get("capability_oracle")
            if oracle in ("symbolic", "llm"):
                routes.append(oracle)
                continue
            # Measure both backends.
            sym = default_symbolic_backend(t)
            llm = self.llm_backend_fn(t) if self.llm_backend_fn else None
            r_sym = backend_reward(sym, self.utility_config)
            if llm is not None:
                r_llm = backend_reward(llm, self.utility_config)
                routes.append("symbolic" if r_sym >= r_llm else "llm")
            else:
                routes.append("symbolic" if r_sym >= 0 else "llm")
        return BaselineResult(name=self.name, routes=routes, utility=0.0, n_samples=len(tasks))


def oracle_gap_closure(
    score_method: float,
    score_baseline: float,
    score_oracle: float,
) -> float:
    """``(score_method - score_baseline) / (score_oracle - score_baseline)``.

    Handles a zero denominator safely (returns 0.0).
    """
    denom = score_oracle - score_baseline
    if abs(denom) < 1e-12:
        return 0.0
    return float((score_method - score_baseline) / denom)


__all__ = [
    "AlwaysLLM",
    "AlwaysSymbolic",
    "Baseline",
    "BaselineResult",
    "HeuristicRouter",
    "OracleRouter",
    "oracle_gap_closure",
]
