"""Tests for leakage detection, statistics, baselines, and invariants
(AutoLearn v2 Phases 10, 12, 13, 18)."""

from __future__ import annotations

import pytest
import numpy as np

from daph_learning.evaluation.leakage import (
    LeakageReport,
    detect_leakage,
    family_aware_split,
    normalize_prompt,
    prompt_hash,
)
from daph_learning.evaluation.statistics import (
    bootstrap_ci,
    cohens_d,
    empirical_p_value,
    mcnemar_test,
    random_direction_null,
)
from daph_learning.evaluation.baselines import (
    AlwaysLLM,
    AlwaysSymbolic,
    HeuristicRouter,
    OracleRouter,
    oracle_gap_closure,
)
from daph_learning.autolearn_v2.invariants import (
    InvariantViolation,
    check_candidate_not_active_until_accepted,
    check_family_disjoint,
    check_model_identity,
    check_split_disjoint,
    check_test_not_in_training,
    check_tokenizer_identity,
    check_vector_dimensions,
)
from daph_learning.autolearn_v2.policies import SingleVectorPolicy


# --- Leakage ---

def test_detect_leakage_clean():
    splits = {
        "train": [{"task_id": "t1", "specification": "a", "generator_family": "f1"}],
        "val": [{"task_id": "t2", "specification": "b", "generator_family": "f2"}],
        "test": [{"task_id": "t3", "specification": "c", "generator_family": "f3"}],
    }
    rep = detect_leakage(splits)
    assert not rep.has_leak


def test_detect_leakage_exact_duplicate():
    splits = {
        "train": [{"task_id": "t1", "specification": "compute 2+2", "expected": 4, "inputs": {}}],
        "test": [{"task_id": "t2", "specification": "compute 2+2", "expected": 4, "inputs": {}}],
    }
    rep = detect_leakage(splits)
    assert rep.has_leak
    assert len(rep.exact_duplicates) == 1


def test_detect_leakage_normalized_duplicate():
    # Both prompts normalize to "compute 22" (punctuation removed).
    splits = {
        "train": [{"task_id": "t1", "specification": "Compute 2+2!", "expected": 4, "inputs": {}}],
        "test": [{"task_id": "t2", "specification": "compute 2+2", "expected": 4, "inputs": {}}],
    }
    rep = detect_leakage(splits)
    assert rep.has_leak
    assert len(rep.normalized_duplicates) == 1


def test_detect_leakage_family_overlap():
    splits = {
        "train": [{"task_id": "t1", "specification": "a", "generator_family": "famA"}],
        "val": [{"task_id": "t2", "specification": "b", "generator_family": "famA"}],
    }
    rep = detect_leakage(splits)
    assert rep.has_leak
    assert len(rep.family_overlap) == 1


def test_detect_leakage_fingerprint_overlap():
    # Same task_id + identical content across splits -> fingerprint overlap.
    splits = {
        "train": [{"task_id": "t1", "specification": "s", "expected": 1, "inputs": {"a": 1}, "capability_ids": ["x"]}],
        "val": [{"task_id": "t1", "specification": "s", "expected": 1, "inputs": {"a": 1}, "capability_ids": ["x"]}],
    }
    rep = detect_leakage(splits)
    assert rep.has_leak
    assert len(rep.fingerprint_overlap) == 1


def test_family_aware_split_disjoint():
    tasks = []
    for i in range(20):
        tasks.append({"task_id": f"t{i}", "specification": f"s{i}", "generator_family": f"fam{i % 5}"})
    train, val, test = family_aware_split(tasks, seed=42)
    train_fams = {t["generator_family"] for t in train}
    val_fams = {t["generator_family"] for t in val}
    test_fams = {t["generator_family"] for t in test}
    assert not (train_fams & val_fams)
    assert not (train_fams & test_fams)
    assert not (val_fams & test_fams)


def test_family_aware_split_deterministic():
    tasks = [{"task_id": f"t{i}", "specification": f"s{i}", "generator_family": f"f{i}"} for i in range(10)]
    t1, v1, te1 = family_aware_split(tasks, seed=42)
    t2, v2, te2 = family_aware_split(tasks, seed=42)
    assert [t["task_id"] for t in t1] == [t["task_id"] for t in t2]


def test_normalize_prompt():
    assert normalize_prompt("  Hello,  World!!  ") == "hello world"


def test_prompt_hash_deterministic():
    assert prompt_hash("a") == prompt_hash("a")


# --- Statistics ---

def test_bootstrap_ci_basic():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, lo, hi = bootstrap_ci(samples, n_bootstrap=500, seed=1)
    assert mean == pytest.approx(3.0)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty():
    mean, lo, hi = bootstrap_ci([])
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


def test_cohens_d():
    a = [1, 2, 3, 4, 5]
    b = [6, 7, 8, 9, 10]
    d = cohens_d(a, b)
    assert d < 0  # a < b


def test_empirical_p_value():
    # learned score higher than all randoms -> small p
    p = empirical_p_value(10.0, [1.0, 2.0, 3.0])
    assert p == pytest.approx(1 / 4)
    # learned score lower than all randoms -> p = 1
    p = empirical_p_value(0.0, [1.0, 2.0, 3.0])
    assert p == pytest.approx(1.0)


def test_mcnemar_no_discordant():
    stat, p = mcnemar_test([(True, True), (False, False)])
    assert p == 1.0


def test_mcnemar_discordant():
    stat, p = mcnemar_test([(True, False)] * 10 + [(False, True)] * 2)
    assert p < 1.0


def test_random_direction_null():
    def score_fn(seed):
        rng = np.random.default_rng(seed)
        return float(rng.normal(0, 1))
    res = random_direction_null(score_fn, n_random=50, seed=42, learned_score=5.0)
    # learned_score very high -> p small
    assert res.p_value == pytest.approx(1 / 51)
    assert res.n_random == 50


# --- Baselines ---

def test_always_llm():
    b = AlwaysLLM()
    r = b.score([{"task_id": "t1"}, {"task_id": "t2"}])
    assert r.routes == ["llm", "llm"]


def test_always_symbolic():
    b = AlwaysSymbolic()
    r = b.score([{"task_id": "t1"}])
    assert r.routes == ["symbolic"]


def test_heuristic_router():
    b = HeuristicRouter()
    r = b.score([
        {"task_id": "t1", "capability_ids": ["integer_arithmetic"]},
        {"task_id": "t2", "capability_ids": []},
    ])
    assert r.routes == ["symbolic", "llm"]


def test_oracle_router_uses_field():
    b = OracleRouter()
    r = b.score([
        {"task_id": "t1", "utility_oracle": "symbolic"},
        {"task_id": "t2", "utility_oracle": "llm"},
    ])
    assert r.routes == ["symbolic", "llm"]


def test_oracle_gap_closure():
    assert oracle_gap_closure(0.8, 0.5, 1.0) == pytest.approx(0.6)
    assert oracle_gap_closure(0.5, 0.5, 0.5) == 0.0  # zero denominator


# --- Invariants ---

def test_check_split_disjoint_passes():
    check_split_disjoint(["a", "b"], ["c"], ["d"])


def test_check_split_disjoint_fails():
    with pytest.raises(InvariantViolation):
        check_split_disjoint(["a"], ["a"], ["b"])


def test_check_family_disjoint_fails():
    with pytest.raises(InvariantViolation):
        check_family_disjoint(["f1"], ["f1"], ["f2"])


def test_check_candidate_not_active_until_accepted():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    with pytest.raises(InvariantViolation):
        check_candidate_not_active_until_accepted(p, p)


def test_check_vector_dimensions():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    check_vector_dimensions(p, 4)
    with pytest.raises(InvariantViolation):
        check_vector_dimensions(p, 8)


def test_check_model_identity():
    p = SingleVectorPolicy.seed(model_id="m1", tokenizer_hash="t", layer=1, hidden_size=4)
    check_model_identity(p, "m1")
    with pytest.raises(InvariantViolation):
        check_model_identity(p, "m2")


def test_check_tokenizer_identity():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t1", layer=1, hidden_size=4)
    with pytest.raises(InvariantViolation):
        check_tokenizer_identity(p, "t2")


def test_check_test_not_in_training():
    check_test_not_in_training(["t1"], ["t2", "t3"])
    with pytest.raises(InvariantViolation):
        check_test_not_in_training(["t1"], ["t1"])
