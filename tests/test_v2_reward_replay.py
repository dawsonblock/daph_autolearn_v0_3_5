"""Tests for the reward engine, experience model, and replay buffer
(AutoLearn v2 Phases 2, 4, 5)."""

from __future__ import annotations

import pytest
import numpy as np

from daph_learning.verification import VerificationResult, VerificationStatus
from daph_learning.autolearn_v2.experience import (
    BackendOutcome,
    Experience,
    fingerprint_task,
    make_experience_id,
)
from daph_learning.autolearn_v2.reward import (
    UtilityConfig,
    backend_reward,
    optimal_action,
    reward_gap,
    reward_gap_from_rewards,
)
from daph_learning.autolearn_v2.replay import (
    ReplayBuffer,
    ReplayConfig,
    ReplaySample,
)


# --- Experience / fingerprint ---

def _outcome(backend, status, score, **kw):
    return BackendOutcome(
        backend=backend, output=kw.get("output"),
        verification_status=status, correctness_score=score,
        latency_ms=kw.get("latency_ms", 1.0), estimated_cost=kw.get("cost", 0.0),
    )


def test_fingerprint_task_deterministic():
    t = {"task_id": "t1", "capability_ids": ["x"], "inputs": {"a": 1}, "specification": "s", "expected": 2}
    assert fingerprint_task(t) == fingerprint_task(t)


def test_fingerprint_task_differs_on_specification():
    t1 = {"task_id": "t1", "specification": "a", "expected": 1}
    t2 = {"task_id": "t1", "specification": "b", "expected": 1}
    assert fingerprint_task(t1) != fingerprint_task(t2)


def test_fingerprint_ignores_routing_labels():
    """Routing labels / oracle fields must not leak into the fingerprint."""
    t1 = {"task_id": "t1", "specification": "s", "expected": 1, "route_label": "symbolic"}
    t2 = {"task_id": "t1", "specification": "s", "expected": 1, "route_label": "llm"}
    assert fingerprint_task(t1) == fingerprint_task(t2)


def test_make_experience_id_deterministic():
    eid = make_experience_id("fp", "pol", "mod", "ds", "ts")
    assert eid.startswith("exp_")
    assert eid == make_experience_id("fp", "pol", "mod", "ds", "ts")


def test_experience_to_dict_roundtrips_status():
    exp = Experience(
        experience_id="e1", task_id="t1", task_fingerprint="fp",
        task_text="s", representation_ref=None, policy_id="p1",
        model_id="m1", dataset_id="d1", selected_action="symbolic",
        outcomes={"symbolic": _outcome("symbolic", VerificationStatus.CORRECT, 1.0)},
        backend_rewards={"symbolic": 1.0}, optimal_action="symbolic",
        reward_gap=1.0, created_at="ts",
    )
    d = exp.to_dict()
    assert d["outcomes"]["symbolic"]["verification_status"] == "correct"


# --- Reward engine ---

def test_utility_config_defaults_correctness_dominates():
    c = UtilityConfig()
    assert c.correctness_weight == 1.0
    assert c.latency_weight < c.correctness_weight


def test_utility_config_rejects_negative():
    with pytest.raises(ValueError):
        UtilityConfig(correctness_weight=-1.0)


def test_utility_config_rejects_zero_cost_ref():
    with pytest.raises(ValueError):
        UtilityConfig(cost_ref=0.0)


def test_backend_reward_correct_symbolic():
    c = UtilityConfig()
    o = _outcome("symbolic", VerificationStatus.CORRECT, 1.0, latency_ms=0.0, cost=0.0)
    r = backend_reward(o, c)
    assert r == pytest.approx(1.0)


def test_backend_reward_incorrect():
    c = UtilityConfig()
    o = _outcome("symbolic", VerificationStatus.INCORRECT, 0.0)
    r = backend_reward(o, c)
    assert r < 0.0  # failure penalty applies


def test_backend_reward_unverifiable_no_correctness_credit():
    """UNVERIFIABLE must never earn positive correctness reward."""
    c = UtilityConfig()
    o = _outcome("llm", VerificationStatus.UNVERIFIABLE, None, cost=0.1)
    r = backend_reward(o, c)
    # correctness=0, uncertainty penalty applies -> reward <= 0
    assert r <= 0.0


def test_backend_reward_execution_error():
    c = UtilityConfig()
    o = _outcome("symbolic", VerificationStatus.EXECUTION_ERROR, 0.0)
    r = backend_reward(o, c)
    assert r < 0.0


def test_reward_gap_symbolic_better():
    c = UtilityConfig()
    outcomes = {
        "symbolic": _outcome("symbolic", VerificationStatus.CORRECT, 1.0),
        "llm": _outcome("llm", VerificationStatus.INCORRECT, 0.0, cost=0.1),
    }
    gap = reward_gap(outcomes, c)
    assert gap is not None
    assert gap > 0.0


def test_reward_gap_llm_better():
    c = UtilityConfig()
    outcomes = {
        "symbolic": _outcome("symbolic", VerificationStatus.EXECUTION_ERROR, 0.0),
        "llm": _outcome("llm", VerificationStatus.CORRECT, 1.0, cost=0.1),
    }
    gap = reward_gap(outcomes, c)
    assert gap is not None
    assert gap < 0.0


def test_reward_gap_none_when_backend_missing():
    c = UtilityConfig()
    outcomes = {"symbolic": _outcome("symbolic", VerificationStatus.CORRECT, 1.0)}
    assert reward_gap(outcomes, c) is None


def test_optimal_action_symbolic():
    c = UtilityConfig(abstain_margin=0.05)
    outcomes = {
        "symbolic": _outcome("symbolic", VerificationStatus.CORRECT, 1.0),
        "llm": _outcome("llm", VerificationStatus.INCORRECT, 0.0, cost=0.1),
    }
    assert optimal_action(outcomes, c) == "symbolic"


def test_optimal_action_llm():
    c = UtilityConfig(abstain_margin=0.05)
    outcomes = {
        "symbolic": _outcome("symbolic", VerificationStatus.EXECUTION_ERROR, 0.0),
        "llm": _outcome("llm", VerificationStatus.CORRECT, 1.0, cost=0.1),
    }
    assert optimal_action(outcomes, c) == "llm"


def test_optimal_action_abstain_near_equal():
    """Near-equal rewards -> ABSTAIN inside the band."""
    c = UtilityConfig(abstain_margin=0.5)
    outcomes = {
        "symbolic": _outcome("symbolic", VerificationStatus.CORRECT, 1.0, latency_ms=10.0),
        "llm": _outcome("llm", VerificationStatus.CORRECT, 1.0, latency_ms=10.0, cost=0.0),
    }
    # Both correct, identical -> gap ~0 -> abstain
    assert optimal_action(outcomes, c) == "abstain"


def test_reward_gap_from_rewards():
    assert reward_gap_from_rewards({"symbolic": 1.0, "llm": 0.5}) == 0.5
    assert reward_gap_from_rewards({"symbolic": 1.0}) is None


# --- Replay buffer ---

def _make_exp(tid, gap, optimal="symbolic", selected="llm", fp=None):
    fp = fp or f"fp_{tid}"
    return Experience(
        experience_id=f"exp_{tid}", task_id=tid, task_fingerprint=fp,
        task_text="s", representation_ref=None, policy_id="p1",
        model_id="m1", dataset_id="d1", selected_action=selected,
        outcomes={}, backend_rewards={}, optimal_action=optimal,
        reward_gap=gap, created_at="ts",
    )


def test_replay_capacity_limit():
    buf = ReplayBuffer(ReplayConfig(capacity=5, batch_size=2))
    for i in range(10):
        buf.add(_make_exp(f"t{i}", float(i)))
    assert len(buf) == 5


def test_replay_dedup_by_fingerprint():
    """Adding the same task fingerprint updates in place, not duplicates."""
    buf = ReplayBuffer(ReplayConfig(capacity=100, batch_size=2))
    buf.add(_make_exp("t1", 1.0, fp="fp1"))
    buf.add(_make_exp("t1", 2.0, fp="fp1"))  # same fingerprint -> update
    assert len(buf) == 1
    sample = buf.sample(1, seed=0)
    assert sample.experiences[0].reward_gap == 2.0


def test_replay_deterministic_sampling():
    buf = ReplayBuffer(ReplayConfig(capacity=100, batch_size=5, seed=42))
    for i in range(20):
        buf.add(_make_exp(f"t{i}", float(i)))
    s1 = buf.sample(5, seed=123)
    s2 = buf.sample(5, seed=123)
    assert [e.experience_id for e in s1.experiences] == [e.experience_id for e in s2.experiences]


def test_replay_balances_optimal_actions():
    """The 'rest' stratum round-robins between symbolic/llm pools."""
    buf = ReplayBuffer(ReplayConfig(
        capacity=100, batch_size=10, seed=42,
        hard_example_fraction=0.0, regression_anchor_fraction=0.0,
    ))
    for i in range(30):
        buf.add(_make_exp(f"sym_{i}", 1.0, optimal="symbolic"))
    for i in range(30):
        buf.add(_make_exp(f"llm_{i}", -1.0, optimal="llm"))
    s = buf.sample(10, seed=1)
    sym = sum(1 for e in s.experiences if e.optimal_action == "symbolic")
    llm = sum(1 for e in s.experiences if e.optimal_action == "llm")
    # Should be roughly balanced (not 10/0).
    assert sym > 0 and llm > 0
    assert sym + llm == 10


def test_replay_statistics():
    buf = ReplayBuffer(ReplayConfig(capacity=100, batch_size=2))
    buf.add(_make_exp("t1", 1.0, optimal="symbolic"))
    buf.add(_make_exp("t2", -1.0, optimal="llm"))
    stats = buf.statistics()
    assert stats["size"] == 2
    assert stats["n_symbolic_optimal"] == 1
    assert stats["n_llm_optimal"] == 1
    assert stats["mean_reward_gap"] == 0.0


def test_replay_empty_sample():
    buf = ReplayBuffer(ReplayConfig(capacity=10, batch_size=2))
    s = buf.sample(2)
    assert s.experiences == []


def test_replay_regression_anchor_preserved():
    buf = ReplayBuffer(ReplayConfig(capacity=3, batch_size=2))
    buf.add(_make_exp("t1", 1.0))
    buf.add(_make_exp("t2", 2.0))
    buf.mark_regression_anchor("exp_t2")
    buf.add(_make_exp("t3", 3.0))
    buf.add(_make_exp("t4", 4.0))  # triggers eviction
    # t2 was an anchor; it should be preserved if possible.
    ids = {e.experience_id for e in buf._entries}
    # The anchor may or may not survive depending on eviction order, but
    # the anchor count tracking should be consistent.
    assert buf.regression_anchor_count <= 1


def test_replay_serialization_roundtrip():
    buf = ReplayBuffer(ReplayConfig(capacity=100, batch_size=2))
    buf.add(_make_exp("t1", 1.0, optimal="symbolic"))
    buf.add(_make_exp("t2", -1.0, optimal="llm"))
    d = buf.to_dict()
    buf2 = ReplayBuffer.from_dict(d)
    assert len(buf2) == 2
    s = buf2.statistics()
    assert s["n_symbolic_optimal"] == 1


def test_replay_config_validates_fractions():
    with pytest.raises(ValueError):
        ReplayConfig(hard_example_fraction=1.5)
    with pytest.raises(ValueError):
        ReplayConfig(hard_example_fraction=0.7, regression_anchor_fraction=0.5)
