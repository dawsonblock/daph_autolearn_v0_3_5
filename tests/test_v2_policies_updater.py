"""Tests for policies, the incremental updater, and trust-region
(AutoLearn v2 Phase 6)."""

from __future__ import annotations

import pytest
import numpy as np

from daph_learning.autolearn_v2.policies import (
    ConditionalSteeringPolicy,
    MultiVectorPolicy,
    PolicyMetrics,
    SingleVectorPolicy,
    hash_vector,
)
from daph_learning.autolearn_v2.updater import (
    UpdateConfig,
    compute_candidate_update,
    hash_replay_sample,
    trust_region_clip,
)


# --- SingleVectorPolicy ---

def test_seed_policy_zero_vector():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    assert p.parent_policy_id is None
    assert np.allclose(p.vector, 0.0)
    assert p.vector_norm() == 0.0
    assert p.hidden_size == 4


def test_policy_apply():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4, alpha=2.0)
    p = SingleVectorPolicy.from_candidate(p, np.array([1, 0, 0, 0], dtype=np.float32))
    h = np.array([1, 1, 1, 1], dtype=np.float32)
    out = p.apply(h)
    assert np.allclose(out, [3, 1, 1, 1])


def test_policy_route_score():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    p = SingleVectorPolicy.from_candidate(p, np.array([1, 0, 0, 0], dtype=np.float32))
    scores = p.route_score(np.array([2, 0, 0, 0], dtype=np.float32))
    assert scores["symbolic"] > scores["llm"]


def test_policy_rejects_wrong_dim():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    with pytest.raises(ValueError):
        SingleVectorPolicy.from_candidate(p, np.zeros(8, dtype=np.float32))


def test_policy_lineage_parent_id():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    child = SingleVectorPolicy.from_candidate(parent, np.ones(4, dtype=np.float32))
    assert child.parent_policy_id == parent.policy_id
    assert child.policy_id != parent.policy_id


def test_policy_displacement():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    child = SingleVectorPolicy.from_candidate(parent, np.array([3, 0, 0, 0], dtype=np.float32))
    assert child.displacement_from(parent) == pytest.approx(3.0)


def test_policy_with_metrics():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    m = PolicyMetrics(utility=0.5, routing_accuracy=0.8, symbolic_rate=0.5,
                      llm_rate=0.5, abstain_rate=0.0, n_samples=100)
    p2 = p.with_metrics(m)
    assert p2.validation_metrics is not None
    assert p2.validation_metrics.utility == 0.5


def test_policy_to_dict_roundtrip():
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    p = SingleVectorPolicy.from_candidate(p, np.array([1, 2, 3, 4], dtype=np.float32))
    d = p.to_dict()
    p2 = SingleVectorPolicy.from_dict(d)
    assert p2.policy_id == p.policy_id
    assert np.allclose(p2.vector, p.vector)


def test_hash_vector_deterministic():
    v = np.array([1, 2, 3], dtype=np.float32)
    assert hash_vector(v) == hash_vector(v)


# --- MultiVectorPolicy ---

def test_multi_vector_composite():
    V = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    c = np.array([2.0, 3.0], dtype=np.float32)
    p = MultiVectorPolicy(
        policy_id="pol_m1", parent_policy_id=None, model_id="m", tokenizer_hash="t",
        layer=1, vectors=V, coefficients=c, hidden_size=4,
    )
    assert p.rank == 2
    assert np.allclose(p.composite_vector(), [2, 3, 0, 0])
    assert p.vector_norm() == pytest.approx(np.linalg.norm([2, 3, 0, 0]))


def test_multi_vector_rejects_bad_coefficients():
    V = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        MultiVectorPolicy(
            policy_id="p", parent_policy_id=None, model_id="m", tokenizer_hash="t",
            layer=1, vectors=V, coefficients=np.zeros(3, dtype=np.float32),
        )


# --- ConditionalSteeringPolicy ---

def test_conditional_policy_shapes():
    hidden, rank = 4, 2
    V = np.eye(hidden, rank, dtype=np.float32)
    W1 = np.zeros((hidden, hidden), dtype=np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = np.zeros((hidden, rank), dtype=np.float32)
    b2 = np.zeros(rank, dtype=np.float32)
    p = ConditionalSteeringPolicy(
        policy_id="pol_c1", parent_policy_id=None, model_id="m", tokenizer_hash="t",
        layer=1, basis=V, router_W1=W1, router_b1=b1, router_W2=W2, router_b2=b2,
    )
    assert p.rank == rank
    assert p.hidden_size == hidden
    h = np.ones(hidden, dtype=np.float32)
    c = p.coefficients(h)
    assert c.shape == (rank,)


# --- Trust region ---

def test_trust_region_no_clip_when_within_radius():
    cur = np.zeros(4, dtype=np.float32)
    cand = np.array([0.05, 0, 0, 0], dtype=np.float32)
    out, clipped, step = trust_region_clip(cand, cur, radius=0.1)
    assert not clipped
    assert step == pytest.approx(0.05)
    assert np.allclose(out, cand)


def test_trust_region_clips_when_exceeds_radius():
    cur = np.zeros(4, dtype=np.float32)
    cand = np.array([1.0, 0, 0, 0], dtype=np.float32)
    out, clipped, step = trust_region_clip(cand, cur, radius=0.1)
    assert clipped
    assert step == pytest.approx(0.1)
    assert np.linalg.norm(out - cur) == pytest.approx(0.1)


def test_trust_region_zero_radius_no_step():
    cur = np.array([1, 2, 3, 4], dtype=np.float32)
    cand = np.array([5, 5, 5, 5], dtype=np.float32)
    out, clipped, step = trust_region_clip(cand, cur, radius=0.0)
    assert step == 0.0
    assert np.allclose(out, cur)


# --- Updater ---

def _exp_with_gap(tid, gap, rep=None):
    from daph_learning.autolearn_v2.experience import Experience
    return Experience(
        experience_id=f"exp_{tid}", task_id=tid, task_fingerprint=f"fp_{tid}",
        task_text="s", representation_ref=rep, policy_id="p1",
        model_id="m1", dataset_id="d1", selected_action="llm",
        outcomes={}, backend_rewards={}, optimal_action="symbolic",
        reward_gap=gap, created_at="ts",
    )


def test_updater_no_counterfactual_signal_returns_noop():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    exps = [_exp_with_gap("t1", None)]
    reps = np.zeros((1, 4), dtype=np.float32)
    cand, meta = compute_candidate_update(parent, exps, reps, UpdateConfig())
    assert meta["n_used"] == 0
    assert np.allclose(cand.vector, parent.vector)


def test_updater_produces_candidate_within_trust_region():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    # Positive gaps -> push toward symbolic direction.
    exps = [_exp_with_gap(f"t{i}", 1.0) for i in range(8)]
    reps = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    cfg = UpdateConfig(learning_rate=0.5, trust_region_radius=0.1)
    cand, meta = compute_candidate_update(parent, exps, reps, cfg)
    assert meta["n_used"] == 8
    assert meta["step_norm"] <= cfg.trust_region_radius + 1e-6
    assert cand.policy_id != parent.policy_id
    assert cand.parent_policy_id == parent.policy_id


def test_updater_respects_max_vector_norm():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    exps = [_exp_with_gap(f"t{i}", 1.0) for i in range(8)]
    reps = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    cfg = UpdateConfig(learning_rate=10.0, trust_region_radius=10.0, max_vector_norm=0.5)
    cand, meta = compute_candidate_update(parent, exps, reps, cfg)
    assert meta["max_vector_norm_clipped"]
    assert cand.vector_norm() <= 0.5 + 1e-6


def test_updater_rejects_dim_mismatch():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    exps = [_exp_with_gap("t1", 1.0)]
    reps = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_candidate_update(parent, exps, reps, UpdateConfig())


def test_updater_rejects_count_mismatch():
    parent = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=4)
    exps = [_exp_with_gap("t1", 1.0)]
    reps = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_candidate_update(parent, exps, reps, UpdateConfig())


def test_hash_replay_sample_deterministic():
    e1 = _exp_with_gap("t1", 1.0)
    e2 = _exp_with_gap("t2", 2.0)
    assert hash_replay_sample([e1, e2]) == hash_replay_sample([e1, e2])
    assert hash_replay_sample([e1, e2]) != hash_replay_sample([e2, e1])


def test_update_config_validates():
    with pytest.raises(ValueError):
        UpdateConfig(learning_rate=-0.1)
    with pytest.raises(ValueError):
        UpdateConfig(max_vector_norm=0)
