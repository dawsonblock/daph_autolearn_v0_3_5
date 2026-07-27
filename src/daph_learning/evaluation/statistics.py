"""Experimental statistics (AutoLearn v2, Phase 12).

Upgrades the v0.3.x "five random samples -> z-score" approach with:

* bootstrap confidence intervals
* paired evaluation
* McNemar test (when appropriate)
* empirical random-direction null + empirical p-value
* effect size (Cohen's d)

Empirical p-value (per the design brief)::

    p = (1 + count(random_score >= learned_score)) / (N + 1)

This avoids reporting unstable z-scores from tiny random samples as
strong evidence.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def bootstrap_ci(
    samples: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns ``(mean, lower, upper)``.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, arr.size, arr.size)
        means[i] = arr[idx].mean()
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return (float(arr.mean()), lower, upper)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size between two samples."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    pooled_std = math.sqrt((a_arr.var(ddof=1) + b_arr.var(ddof=1)) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled_std)


def mcnemar_test(
    pairs: Sequence[tuple[bool, bool]],
) -> tuple[float, float]:
    """McNemar's test on paired binary outcomes.

    ``pairs`` is a sequence of ``(correct_a, correct_b)`` booleans.
    Returns ``(statistic, p_value)`` using the exact binomial form for
    small discordant counts and the chi-square approximation otherwise.
    """
    n01 = sum(1 for a, b in pairs if not a and b)
    n10 = sum(1 for a, b in pairs if a and not b)
    discordant = n01 + n10
    if discordant == 0:
        return (0.0, 1.0)
    if discordant < 25:
        # Exact two-sided binomial.
        from math import comb
        n = discordant
        k = min(n01, n10)
        # two-sided p-value
        p = 0.0
        for i in range(0, k + 1):
            p += comb(n, i) * (0.5 ** n)
        p = min(1.0, 2 * p)
        return (float(abs(n01 - n10)), p)
    stat = (abs(n01 - n10) - 1) ** 2 / discordant
    # chi-square df=1 survival via the error function complement.
    p = math.erfc(math.sqrt(stat / 2.0))
    return (float(stat), float(p))


def empirical_p_value(
    learned_score: float,
    random_scores: Sequence[float],
) -> float:
    """Empirical p-value::

        p = (1 + count(random_score >= learned_score)) / (N + 1)

    Per the design brief. Avoids unstable z-scores from tiny samples.
    """
    n = len(random_scores)
    if n == 0:
        return 1.0
    ge = sum(1 for s in random_scores if s >= learned_score)
    return (1 + ge) / (n + 1)


@dataclass
class RandomDirectionNull:
    """Result of an empirical random-direction null comparison."""

    learned_score: float
    random_scores: list[float]
    p_value: float
    n_random: int
    mean_random: float
    std_random: float

    def to_dict(self) -> dict:
        return {
            "learned_score": self.learned_score,
            "random_scores": list(self.random_scores),
            "p_value": self.p_value,
            "n_random": self.n_random,
            "mean_random": self.mean_random,
            "std_random": self.std_random,
        }


def random_direction_null(
    score_fn,
    *,
    n_random: int = 100,
    seed: int = 42,
    learned_score: float | None = None,
) -> RandomDirectionNull:
    """Run an empirical random-direction null.

    ``score_fn(random_state) -> float`` is called ``n_random`` times with
    independent seeded RNG states. When ``learned_score`` is None it is
    taken as the score of the deterministic (seed=0) call.
    """
    rng = random.Random(seed)
    scores: list[float] = []
    for i in range(n_random):
        sub_seed = rng.randint(0, 2**31 - 1)
        scores.append(float(score_fn(sub_seed)))
    if learned_score is None:
        learned_score = float(score_fn(seed))
    p = empirical_p_value(learned_score, scores)
    arr = np.asarray(scores, dtype=np.float64)
    return RandomDirectionNull(
        learned_score=float(learned_score),
        random_scores=scores,
        p_value=p,
        n_random=n_random,
        mean_random=float(arr.mean()) if arr.size else 0.0,
        std_random=float(arr.std()) if arr.size else 0.0,
    )


__all__ = [
    "RandomDirectionNull",
    "bootstrap_ci",
    "cohens_d",
    "empirical_p_value",
    "mcnemar_test",
    "random_direction_null",
]
