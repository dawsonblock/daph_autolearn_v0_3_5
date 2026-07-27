"""Typed output verification (AutoLearn v2, Phase 1B).

Replaces the v0.3.x ``str(int(expected)) in text`` substring check, which
falsely accepted ``expected=12`` against ``output="312"`` because ``"12"``
is a substring of ``"312"``.

Verification is now an explicit, typed operation returning a
:class:`VerificationStatus` enum. The central rule:

    UNVERIFIABLE is never mapped to CORRECT.

A verification result carries a continuous ``correctness_score`` in
``[0, 1]`` (``1.0`` for CORRECT, ``0.0`` for INCORRECT / EXECUTION_ERROR /
TIMEOUT, ``None`` for UNVERIFIABLE) so the reward engine can consume a
single signal without re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class VerificationStatus(str, Enum):
    """Canonical verification outcome taxonomy."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNVERIFIABLE = "unverifiable"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"

    @property
    def is_correct(self) -> bool:
        return self is VerificationStatus.CORRECT

    @property
    def is_verifiable(self) -> bool:
        """True when the status is a definitive correct/incorrect verdict."""
        return self in (VerificationStatus.CORRECT, VerificationStatus.INCORRECT)


@dataclass(frozen=True)
class VerificationResult:
    """The result of verifying one backend output against a task.

    Attributes
    ----------
    status : VerificationStatus
    correctness_score : float | None
        ``1.0`` for CORRECT, ``0.0`` for INCORRECT/EXECUTION_ERROR/TIMEOUT,
        ``None`` for UNVERIFIABLE. The reward engine treats ``None`` as
        "no signal" — it never contributes a positive correctness reward.
    detail : str | None
        Human-readable explanation of the verdict (which verifier matched,
        why it failed, etc.).
    verifier : str
        Name of the verifier that produced this result.
    """

    status: VerificationStatus
    correctness_score: float | None = None
    detail: str | None = None
    verifier: str = "unknown"

    def __post_init__(self) -> None:
        # Enforce the score/status contract.
        if self.status is VerificationStatus.CORRECT:
            object.__setattr__(self, "correctness_score", 1.0)
        elif self.status in (
            VerificationStatus.INCORRECT,
            VerificationStatus.EXECUTION_ERROR,
            VerificationStatus.TIMEOUT,
        ):
            object.__setattr__(self, "correctness_score", 0.0)
        elif self.status is VerificationStatus.UNVERIFIABLE:
            # UNVERIFIABLE must NOT carry a positive correctness score.
            if self.correctness_score is not None and self.correctness_score > 0.0:
                raise ValueError(
                    "UNVERIFIABLE status must not carry a positive correctness_score; "
                    f"got {self.correctness_score!r}"
                )
            object.__setattr__(self, "correctness_score", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_expected(task: Mapping[str, Any], expected: Any | None) -> Any | None:
    """Resolve the expected answer: explicit arg wins, then task['expected']."""
    if expected is not None:
        return expected
    return task.get("expected")


def _coerce_int(value: Any) -> int | None:
    """Coerce a value to int without substring matching."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Accept a leading "FINAL:" marker used by the symbolic executor.
        if s.startswith("FINAL:"):
            s = s[len("FINAL:"):].strip()
        try:
            return int(s)
        except ValueError:
            # Reject "312" when asked to parse a bare integer: only a clean
            # integer string is accepted. No substring search.
            return None
    return None


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


def verify_exact_integer(
    output: Any,
    expected: Any,
    *,
    verifier: str = "exact_integer",
) -> VerificationResult:
    """Verify that ``output`` is exactly the integer ``expected``.

    No substring matching. ``output="312"`` against ``expected=12`` is
    INCORRECT, not CORRECT. ``output="The answer is 12."`` is UNVERIFIABLE
    (we do not parse prose).
    """
    exp_int = _coerce_int(expected)
    if exp_int is None:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            detail="expected value is not an integer; cannot verify with exact_integer",
            verifier=verifier,
        )
    out_int = _coerce_int(output)
    if out_int is None:
        # Output is not a clean integer. It might be prose containing the
        # number, but we refuse to substring-match. Mark UNVERIFIABLE so the
        # reward engine never credits it.
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            detail="output is not a clean integer; refusing substring match",
            verifier=verifier,
        )
    if out_int == exp_int:
        return VerificationResult(
            status=VerificationStatus.CORRECT,
            detail=f"output {out_int} == expected {exp_int}",
            verifier=verifier,
        )
    return VerificationResult(
        status=VerificationStatus.INCORRECT,
        detail=f"output {out_int} != expected {exp_int}",
        verifier=verifier,
    )


def verify_normalized_string(
    output: Any,
    expected: Any,
    *,
    verifier: str = "normalized_string",
) -> VerificationResult:
    """Verify two strings after whitespace/case normalization.

    Used for non-numeric free-text answers where an exact expected string
    exists. If ``expected`` is ``None`` the result is UNVERIFIABLE (we
    cannot verify free text without a reference).
    """
    if expected is None:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            detail="no expected string provided",
            verifier=verifier,
        )
    norm_out = str(output).strip().lower() if output is not None else ""
    norm_exp = str(expected).strip().lower()
    if norm_out == norm_exp:
        return VerificationResult(
            status=VerificationStatus.CORRECT,
            detail="normalized strings match",
            verifier=verifier,
        )
    return VerificationResult(
        status=VerificationStatus.INCORRECT,
        detail="normalized strings differ",
        verifier=verifier,
    )


def verify_symbolic_result(
    output: Any,
    expected: Any,
    *,
    verified: bool = True,
    verifier: str = "symbolic_result",
) -> VerificationResult:
    """Verify a structured symbolic-execution result.

    The symbolic executor returns ``"FINAL: <int>"`` on success. We parse
    the integer exactly (no substring) and compare to ``expected``. When
    ``expected`` is ``None`` we trust the executor's own ``verified`` flag
    (the executor already validated the computation internally).
    """
    if output is None:
        return VerificationResult(
            status=VerificationStatus.EXECUTION_ERROR,
            detail="symbolic execution produced no output",
            verifier=verifier,
        )
    if not isinstance(output, str) or not output.startswith("FINAL:"):
        return VerificationResult(
            status=VerificationStatus.EXECUTION_ERROR,
            detail=f"malformed symbolic output: {output!r}",
            verifier=verifier,
        )
    if not verified:
        return VerificationResult(
            status=VerificationStatus.EXECUTION_ERROR,
            detail="symbolic executor reported unverified result",
            verifier=verifier,
        )
    if expected is None:
        # No external ground truth: trust the executor's internal verification.
        return VerificationResult(
            status=VerificationStatus.CORRECT,
            detail="symbolic executor self-verified, no external expected value",
            verifier=verifier,
        )
    return verify_exact_integer(output, expected, verifier=verifier)


def verify_unverifiable(
    output: Any,
    expected: Any,
    *,
    verifier: str = "unverifiable",
    detail: str | None = None,
) -> VerificationResult:
    """Explicitly mark an outcome as unverifiable.

    Used for LLM free-text answers with no ground truth. This NEVER maps
    to CORRECT — the reward engine treats UNVERIFIABLE as zero correctness
    signal.
    """
    return VerificationResult(
        status=VerificationStatus.UNVERIFIABLE,
        detail=detail or "no verifiable ground truth available",
        verifier=verifier,
    )


def verify_execution_error(
    *,
    failure_type: str = "execution_error",
    detail: str | None = None,
    verifier: str = "execution_error",
) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.EXECUTION_ERROR,
        detail=detail or failure_type,
        verifier=verifier,
    )


def verify_timeout(*, detail: str | None = None, verifier: str = "timeout") -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.TIMEOUT,
        detail=detail or "execution exceeded time budget",
        verifier=verifier,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def verify_task_output(
    task: Mapping[str, Any],
    output: Any,
    *,
    backend: str,
    expected: Any | None = None,
    symbolic_verified: bool = True,
    failure_type: str | None = None,
) -> VerificationResult:
    """Dispatch verification for a single backend output.

    The verifier is chosen from the backend and the task's expected value:

    * ``backend == "symbolic"`` -> :func:`verify_symbolic_result` (with the
      executor's ``verified`` flag). A ``failure_type`` short-circuits to
      EXECUTION_ERROR / TIMEOUT.
    * ``backend == "llm"`` and an integer ``expected`` exists ->
      :func:`verify_exact_integer` (typed; no substring match). This is
      the case the v0.3.x substring bug mishandled.
    * ``backend == "llm"`` and a non-integer string ``expected`` exists ->
      :func:`verify_normalized_string`.
    * ``backend == "llm"`` and no expected value -> UNVERIFIABLE.
    """
    if failure_type == "timeout":
        return verify_timeout(detail=f"backend={backend} timed out")
    if failure_type is not None:
        return verify_execution_error(
            failure_type=failure_type,
            detail=f"backend={backend} failed: {failure_type}",
        )

    exp = _task_expected(task, expected)

    if backend == "symbolic":
        return verify_symbolic_result(output, exp, verified=symbolic_verified)

    if backend == "llm":
        if exp is None:
            return verify_unverifiable(output, exp, detail="llm free-text, no ground truth")
        # If expected is integer-valued, use exact-integer verification.
        if _coerce_int(exp) is not None:
            return verify_exact_integer(output, exp)
        # Otherwise treat as a normalized-string comparison.
        return verify_normalized_string(output, exp)

    # Unknown backend: fail closed.
    return verify_execution_error(
        failure_type="unknown_backend",
        detail=f"unknown backend {backend!r}",
    )


__all__ = [
    "VerificationResult",
    "VerificationStatus",
    "verify_exact_integer",
    "verify_execution_error",
    "verify_normalized_string",
    "verify_symbolic_result",
    "verify_task_output",
    "verify_timeout",
    "verify_unverifiable",
]
