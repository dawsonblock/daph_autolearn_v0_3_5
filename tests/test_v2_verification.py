"""Tests for typed verification (AutoLearn v2 Phase 1B).

Critically covers the numeric substring bug regression: ``expected=12``
against ``output="312"`` must NOT pass.
"""

from __future__ import annotations

import pytest

from daph_learning.verification import (
    VerificationResult,
    VerificationStatus,
    verify_exact_integer,
    verify_normalized_string,
    verify_symbolic_result,
    verify_task_output,
    verify_timeout,
    verify_unverifiable,
)


# --- VerificationStatus contract ---

def test_correct_has_score_one():
    r = VerificationResult(status=VerificationStatus.CORRECT, verifier="t")
    assert r.correctness_score == 1.0


def test_incorrect_has_score_zero():
    r = VerificationResult(status=VerificationStatus.INCORRECT, verifier="t")
    assert r.correctness_score == 0.0


def test_execution_error_has_score_zero():
    r = VerificationResult(status=VerificationStatus.EXECUTION_ERROR, verifier="t")
    assert r.correctness_score == 0.0


def test_unverifiable_has_none_score():
    r = VerificationResult(status=VerificationStatus.UNVERIFIABLE, verifier="t")
    assert r.correctness_score is None


def test_unverifiable_rejects_positive_score():
    """UNVERIFIABLE must never carry a positive correctness score."""
    with pytest.raises(ValueError):
        VerificationResult(
            status=VerificationStatus.UNVERIFIABLE, correctness_score=0.5, verifier="t"
        )


def test_status_is_correct_property():
    assert VerificationStatus.CORRECT.is_correct
    assert not VerificationStatus.UNVERIFIABLE.is_correct


def test_status_is_verifiable_property():
    assert VerificationStatus.CORRECT.is_verifiable
    assert VerificationStatus.INCORRECT.is_verifiable
    assert not VerificationStatus.UNVERIFIABLE.is_verifiable


# --- The substring bug regression ---

def test_substring_bug_312_vs_12_is_incorrect():
    """expected=12, output='312' must be INCORRECT, not CORRECT."""
    r = verify_exact_integer("312", 12)
    assert r.status is VerificationStatus.INCORRECT
    assert r.correctness_score == 0.0


def test_substring_bug_312_vs_12_via_dispatch():
    """Same regression through the task-output dispatcher."""
    task = {"task_id": "t1", "expected": 12, "capability_ids": ["integer_arithmetic"]}
    r = verify_task_output(task, "312", backend="llm")
    assert r.status is VerificationStatus.INCORRECT


def test_exact_integer_match():
    r = verify_exact_integer("12", 12)
    assert r.status is VerificationStatus.CORRECT


def test_exact_integer_match_with_final_prefix():
    r = verify_exact_integer("FINAL: 42", 42)
    assert r.status is VerificationStatus.CORRECT


def test_exact_integer_prose_is_unverifiable():
    """'The answer is 12.' is UNVERIFIABLE (we refuse to substring-match)."""
    r = verify_exact_integer("The answer is 12.", 12)
    assert r.status is VerificationStatus.UNVERIFIABLE


def test_exact_integer_expected_not_integer_is_unverifiable():
    r = verify_exact_integer("12", "banana")
    assert r.status is VerificationStatus.UNVERIFIABLE


def test_exact_integer_wrong_value():
    r = verify_exact_integer("99", 12)
    assert r.status is VerificationStatus.INCORRECT


def test_exact_integer_negative():
    r = verify_exact_integer("-5", -5)
    assert r.status is VerificationStatus.CORRECT


def test_exact_integer_float_integer_valued():
    r = verify_exact_integer(12.0, 12)
    assert r.status is VerificationStatus.CORRECT


# --- Symbolic result verifier ---

def test_symbolic_result_correct():
    r = verify_symbolic_result("FINAL: 42", 42, verified=True)
    assert r.status is VerificationStatus.CORRECT


def test_symbolic_result_wrong():
    r = verify_symbolic_result("FINAL: 99", 42, verified=True)
    assert r.status is VerificationStatus.INCORRECT


def test_symbolic_result_no_expected_self_verified():
    r = verify_symbolic_result("FINAL: 42", None, verified=True)
    assert r.status is VerificationStatus.CORRECT


def test_symbolic_result_unverified_is_execution_error():
    r = verify_symbolic_result("FINAL: 42", 42, verified=False)
    assert r.status is VerificationStatus.EXECUTION_ERROR


def test_symbolic_result_malformed_is_execution_error():
    r = verify_symbolic_result("ERROR: bad", 42, verified=True)
    assert r.status is VerificationStatus.EXECUTION_ERROR


def test_symbolic_result_none_is_execution_error():
    r = verify_symbolic_result(None, 42, verified=True)
    assert r.status is VerificationStatus.EXECUTION_ERROR


# --- Normalized string verifier ---

def test_normalized_string_match():
    r = verify_normalized_string("  Hello  ", "hello")
    assert r.status is VerificationStatus.CORRECT


def test_normalized_string_mismatch():
    r = verify_normalized_string("world", "hello")
    assert r.status is VerificationStatus.INCORRECT


def test_normalized_string_no_expected_unverifiable():
    r = verify_normalized_string("hello", None)
    assert r.status is VerificationStatus.UNVERIFIABLE


# --- Dispatch ---

def test_dispatch_symbolic():
    task = {"task_id": "t1", "expected": 7, "capability_ids": ["integer_arithmetic"]}
    r = verify_task_output(task, "FINAL: 7", backend="symbolic")
    assert r.status is VerificationStatus.CORRECT


def test_dispatch_llm_integer_expected():
    task = {"task_id": "t1", "expected": 7, "capability_ids": ["integer_arithmetic"]}
    r = verify_task_output(task, "7", backend="llm")
    assert r.status is VerificationStatus.CORRECT


def test_dispatch_llm_no_expected_unverifiable():
    task = {"task_id": "t1", "expected": None, "capability_ids": []}
    r = verify_task_output(task, "some prose", backend="llm")
    assert r.status is VerificationStatus.UNVERIFIABLE
    assert r.correctness_score is None


def test_dispatch_llm_string_expected():
    task = {"task_id": "t1", "expected": "hello", "capability_ids": []}
    r = verify_task_output(task, "Hello", backend="llm")
    assert r.status is VerificationStatus.CORRECT


def test_dispatch_failure_type_timeout():
    task = {"task_id": "t1", "expected": 7, "capability_ids": ["integer_arithmetic"]}
    r = verify_task_output(task, None, backend="symbolic", failure_type="timeout")
    assert r.status is VerificationStatus.TIMEOUT


def test_dispatch_failure_type_execution_error():
    task = {"task_id": "t1", "expected": 7, "capability_ids": ["integer_arithmetic"]}
    r = verify_task_output(task, None, backend="symbolic", failure_type="SymbolicMathError")
    assert r.status is VerificationStatus.EXECUTION_ERROR


def test_dispatch_unknown_backend_fails_closed():
    task = {"task_id": "t1", "expected": 7, "capability_ids": []}
    r = verify_task_output(task, "x", backend="banana")
    assert r.status is VerificationStatus.EXECUTION_ERROR


def test_timeout_verifier():
    r = verify_timeout()
    assert r.status is VerificationStatus.TIMEOUT
    assert r.correctness_score == 0.0


def test_unverifiable_verifier_never_correct():
    r = verify_unverifiable("prose", None)
    assert r.status is VerificationStatus.UNVERIFIABLE
    assert r.correctness_score is None
