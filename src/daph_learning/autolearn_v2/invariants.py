"""Scientific invariants (AutoLearn v2, Phase 18).

Encodes important assumptions as assertions that fail loudly when
violated. These are checked at runtime by the engine and are also
exposed as test helpers.
"""

from __future__ import annotations

from typing import Sequence

from .policies.static_vector import SingleVectorPolicy


class InvariantViolation(AssertionError):
    """Raised when a scientific invariant is violated."""


def check_split_disjoint(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> None:
    """train ∩ validation ∩ test must be pairwise disjoint."""
    s_train = set(map(str, train_ids))
    s_val = set(map(str, val_ids))
    s_test = set(map(str, test_ids))
    if s_train & s_val:
        raise InvariantViolation(f"train ∩ validation non-empty: {s_train & s_val}")
    if s_train & s_test:
        raise InvariantViolation(f"train ∩ test non-empty: {s_train & s_test}")
    if s_val & s_test:
        raise InvariantViolation(f"validation ∩ test non-empty: {s_val & s_test}")


def check_family_disjoint(
    train_families: Sequence[str],
    val_families: Sequence[str],
    test_families: Sequence[str],
) -> None:
    """Template/generator families must be pairwise disjoint across splits."""
    s_train = set(map(str, train_families))
    s_val = set(map(str, val_families))
    s_test = set(map(str, test_families))
    if s_train & s_val:
        raise InvariantViolation(f"train ∩ validation family overlap: {s_train & s_val}")
    if s_train & s_test:
        raise InvariantViolation(f"train ∩ test family overlap: {s_train & s_test}")
    if s_val & s_test:
        raise InvariantViolation(f"validation ∩ test family overlap: {s_val & s_test}")


def check_candidate_not_active_until_accepted(
    active: SingleVectorPolicy,
    candidate: SingleVectorPolicy,
) -> None:
    if active.policy_id == candidate.policy_id:
        raise InvariantViolation(
            "candidate policy must differ from active policy until accepted"
        )


def check_vector_dimensions(policy: SingleVectorPolicy, expected_hidden: int) -> None:
    if policy.vector.shape[0] != expected_hidden:
        raise InvariantViolation(
            f"vector dim {policy.vector.shape[0]} != expected hidden {expected_hidden}"
        )


def check_model_identity(policy: SingleVectorPolicy, runtime_model_id: str) -> None:
    if policy.model_id != runtime_model_id:
        raise InvariantViolation(
            f"policy model_id {policy.model_id!r} != runtime model_id {runtime_model_id!r}"
        )


def check_tokenizer_identity(policy: SingleVectorPolicy, runtime_tokenizer_hash: str) -> None:
    if policy.tokenizer_hash != runtime_tokenizer_hash:
        raise InvariantViolation(
            f"policy tokenizer_hash {policy.tokenizer_hash!r} != runtime "
            f"tokenizer_hash {runtime_tokenizer_hash!r}"
        )


def check_test_not_in_training(
    test_ids: Sequence[str],
    train_ids: Sequence[str],
) -> None:
    """Test task ids must never appear in the training set."""
    s_test = set(map(str, test_ids))
    s_train = set(map(str, train_ids))
    if s_test & s_train:
        raise InvariantViolation(
            f"test ids present in training set: {s_test & s_train}"
        )


__all__ = [
    "InvariantViolation",
    "check_candidate_not_active_until_accepted",
    "check_family_disjoint",
    "check_model_identity",
    "check_split_disjoint",
    "check_test_not_in_training",
    "check_tokenizer_identity",
    "check_vector_dimensions",
]
