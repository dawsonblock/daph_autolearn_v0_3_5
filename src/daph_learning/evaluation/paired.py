from __future__ import annotations

from math import comb, erfc, inf, sqrt
from typing import TypedDict


def exact_paired_binomial(improved: int, regressed: int) -> float:
    """Two-sided exact binomial p-value for discordant paired outcomes."""
    if improved < 0 or regressed < 0:
        raise ValueError("discordant counts must be non-negative")
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    q = 1.0 - p
    return sum(
        comb(n, i) * (p ** i) * (q ** (n - i))
        for i in range(k + 1)
    )


def _solve_decreasing_cdf(k: int, n: int, target: float) -> float:
    """Solve BinomCDF(k; n, p) == target for p by monotone bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        value = _binomial_cdf(k, n, mid)
        if value > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion.

    Implemented without SciPy so the core evaluator keeps its lightweight base
    dependency set.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if trials == 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lower = _solve_decreasing_cdf(
            successes - 1,
            trials,
            1.0 - alpha / 2.0,
        )

    if successes == trials:
        upper = 1.0
    else:
        upper = _solve_decreasing_cdf(
            successes,
            trials,
            alpha / 2.0,
        )
    return lower, upper


class PairedTestResult(TypedDict):
    p_value: float
    mcnemar_statistic: float
    mcnemar_p_value: float
    odds_ratio: float
    improved: int
    regressed: int
    discordant_n: int
    improved_probability: float
    improved_probability_ci_95: tuple[float, float]
    net_discordant_effect: float
    net_discordant_effect_ci_95: tuple[float, float]


def paired_discordant_analysis(
    improved: int,
    regressed: int,
    *,
    confidence: float = 0.95,
) -> PairedTestResult:
    """Expanded telemetry for paired discordant outcomes.

    The exact paired-binomial test remains the primary small-sample p-value.
    McNemar's continuity-corrected chi-square statistic is additionally
    reported, along with its chi-square(1) survival probability, the discordant
    odds ratio, and an exact Clopper-Pearson interval for the probability that a
    discordant pair favors treatment.
    """
    if improved < 0 or regressed < 0:
        raise ValueError("discordant counts must be non-negative")

    exact_p = exact_paired_binomial(improved, regressed)
    n = improved + regressed

    if n == 0:
        statistic = 0.0
        mcnemar_p = 1.0
        odds_ratio = 1.0
        improved_probability = 0.5
        probability_ci = (0.0, 1.0)
    else:
        corrected_difference = max(abs(improved - regressed) - 1.0, 0.0)
        statistic = (corrected_difference ** 2) / n
        mcnemar_p = erfc(sqrt(statistic / 2.0))
        odds_ratio = improved / regressed if regressed > 0 else inf
        improved_probability = improved / n
        probability_ci = clopper_pearson_interval(
            improved,
            n,
            confidence=confidence,
        )

    net_effect = 2.0 * improved_probability - 1.0
    net_ci = (
        2.0 * probability_ci[0] - 1.0,
        2.0 * probability_ci[1] - 1.0,
    )
    return {
        "p_value": exact_p,
        "mcnemar_statistic": statistic,
        "mcnemar_p_value": mcnemar_p,
        "odds_ratio": odds_ratio,
        "improved": improved,
        "regressed": regressed,
        "discordant_n": n,
        "improved_probability": improved_probability,
        "improved_probability_ci_95": probability_ci,
        "net_discordant_effect": net_effect,
        "net_discordant_effect_ci_95": net_ci,
    }
