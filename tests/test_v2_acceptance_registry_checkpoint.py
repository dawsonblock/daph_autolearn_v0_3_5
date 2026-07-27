"""Tests for the acceptance gate, policy registry, and checkpointing
(AutoLearn v2 Phases 7, 8, 9)."""

from __future__ import annotations

import os
import pytest
import numpy as np

from daph_learning.autolearn_v2.policies import PolicyMetrics, SingleVectorPolicy
from daph_learning.autolearn_v2.acceptance import (
    AcceptanceConfig,
    CandidateDecision,
    REASON_DOMAIN_REGRESSION,
    REASON_ROUTE_COLLAPSE,
    REASON_UTILITY_GAIN,
    evaluate_candidate,
)
from daph_learning.autolearn_v2.registry import (
    PolicyAlreadyAcceptedError,
    PolicyNotFoundError,
    PolicyRegistry,
)
from daph_learning.autolearn_v2.checkpoint import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from daph_learning.autolearn_v2.replay import ReplayBuffer, ReplayConfig


# --- Acceptance ---

def _pol(vec=None, hidden=4):
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=1, hidden_size=hidden)
    if vec is not None:
        return SingleVectorPolicy.from_candidate(p, np.asarray(vec, dtype=np.float32))
    return p


def _metrics(utility=0.5, acc=0.8, sym=0.5, llm=0.5, abstain=0.0, n=200, per_domain=None):
    return PolicyMetrics(
        utility=utility, routing_accuracy=acc, symbolic_rate=sym,
        llm_rate=llm, abstain_rate=abstain, n_samples=n,
        per_domain=per_domain or {},
    )


def test_acceptance_accepts_improvement():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics(utility=0.5)
    cm = _metrics(utility=0.7)
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig())
    assert dec.accepted
    assert dec.reason_codes == []


def test_acceptance_rejects_insufficient_utility_gain():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics(utility=0.5)
    cm = _metrics(utility=0.505)  # < minimum_utility_gain 0.01
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig(minimum_utility_gain=0.01))
    assert not dec.accepted
    assert REASON_UTILITY_GAIN in dec.reason_codes


def test_acceptance_rejects_route_collapse_symbolic():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics()
    cm = _metrics(sym=0.99, llm=0.01)  # 99% symbolic -> collapse
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig())
    assert not dec.accepted
    assert REASON_ROUTE_COLLAPSE in dec.reason_codes


def test_acceptance_rejects_route_collapse_llm():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics()
    cm = _metrics(sym=0.01, llm=0.99)
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig())
    assert not dec.accepted
    assert REASON_ROUTE_COLLAPSE in dec.reason_codes


def test_acceptance_rejects_domain_regression():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics(per_domain={"add": 0.8, "mul": 0.8})
    cm = _metrics(utility=0.7, per_domain={"add": 0.85, "mul": 0.5})  # mul regresses
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig(maximum_domain_regression=0.03))
    assert not dec.accepted
    assert REASON_DOMAIN_REGRESSION in dec.reason_codes


def test_acceptance_rejects_insufficient_samples():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics(n=200)
    cm = _metrics(utility=0.9, n=50)  # < minimum_samples
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig(minimum_samples=100))
    assert not dec.accepted
    assert "INSUFFICIENT_SAMPLES" in dec.reason_codes


def test_acceptance_rejects_abstain_spike():
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    am = _metrics(abstain=0.0)
    cm = _metrics(utility=0.7, abstain=0.5)  # huge abstention rise
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig(max_abstain_increase=0.2))
    assert not dec.accepted
    assert "ABSTAIN_SPIKE" in dec.reason_codes


def test_acceptance_rejects_excessive_displacement():
    active = _pol()
    cand = _pol([5.0, 0, 0, 0])  # large displacement
    am = _metrics()
    cm = _metrics(utility=0.9)
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig(max_displacement=1.0))
    assert not dec.accepted
    assert "EXCESSIVE_DISPLACEMENT" in dec.reason_codes


def test_candidate_decision_to_dict():
    dec = CandidateDecision(accepted=False, reason_codes=["X"], utility_gain=-0.1)
    d = dec.to_dict()
    assert d["accepted"] is False
    assert d["reason_codes"] == ["X"]


def test_acceptance_config_validates():
    with pytest.raises(ValueError):
        AcceptanceConfig(maximum_accuracy_regression=1.5)
    with pytest.raises(ValueError):
        AcceptanceConfig(minimum_utility_gain=-0.1)


# --- Registry ---

def test_registry_register_and_accept(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    p = _pol([0.1, 0, 0, 0])
    reg.register_candidate(p)
    rec = reg.get(p.policy_id)
    assert rec.status == "candidate"
    dec = CandidateDecision(accepted=True)
    reg.accept(p, dec)
    assert reg.get(p.policy_id).status == "accepted"
    assert reg.active_policy_id() == p.policy_id


def test_registry_reject_keeps_lineage(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    p = _pol([0.1, 0, 0, 0])
    reg.register_candidate(p)
    dec = CandidateDecision(accepted=False, reason_codes=["X"])
    reg.reject(p, dec)
    assert reg.get(p.policy_id).status == "rejected"


def test_registry_cannot_accept_rejected(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    p = _pol([0.1, 0, 0, 0])
    reg.register_candidate(p)
    reg.reject(p, CandidateDecision(accepted=False, reason_codes=["X"]))
    with pytest.raises(Exception):
        reg.accept(p, CandidateDecision(accepted=True))


def test_registry_lineage_chain(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    root = _pol()
    reg.accept(root, CandidateDecision(accepted=True))
    child = SingleVectorPolicy.from_candidate(root, np.array([0.1, 0, 0, 0], dtype=np.float32))
    reg.accept(child, CandidateDecision(accepted=True))
    chain = reg.lineage(child.policy_id)
    assert len(chain) == 2
    assert chain[0].policy.policy_id == root.policy_id
    assert chain[1].policy.policy_id == child.policy_id


def test_registry_rollback(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    root = _pol()
    reg.accept(root, CandidateDecision(accepted=True))
    child = SingleVectorPolicy.from_candidate(root, np.array([0.1, 0, 0, 0], dtype=np.float32))
    reg.accept(child, CandidateDecision(accepted=True))
    # Roll back to root.
    reg.rollback(root.policy_id)
    assert reg.active_policy_id() == root.policy_id


def test_registry_rollback_to_unknown_raises(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    with pytest.raises(PolicyNotFoundError):
        reg.rollback("nonexistent")


def test_registry_immutable_accepted_artifact(tmp_path):
    reg = PolicyRegistry(str(tmp_path))
    p = _pol([0.1, 0, 0, 0])
    reg.register_candidate(p)
    reg.accept(p, CandidateDecision(accepted=True))
    # The accepted companion file must exist and not be overwritten.
    accepted_path = os.path.join(str(tmp_path), f"{p.policy_id}.accepted.json")
    assert os.path.exists(accepted_path)
    # Re-accepting should raise (already accepted).
    with pytest.raises(PolicyAlreadyAcceptedError):
        reg.accept(p, CandidateDecision(accepted=True))


def test_registry_persistence_across_instances(tmp_path):
    reg1 = PolicyRegistry(str(tmp_path))
    p = _pol([0.1, 0, 0, 0])
    reg1.register_candidate(p)
    reg1.accept(p, CandidateDecision(accepted=True))
    # New registry instance loads existing records.
    reg2 = PolicyRegistry(str(tmp_path))
    assert reg2.get(p.policy_id).status == "accepted"
    assert reg2.active_policy_id() == p.policy_id


# --- Checkpoint ---

def test_checkpoint_atomic_save_load(tmp_path):
    p = _pol([0.1, 0, 0, 0])
    buf = ReplayBuffer(ReplayConfig(capacity=10, batch_size=2))
    state = CheckpointState(iteration=3, active_policy=p, replay=buf, saved_at="ts")
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(state, path)
    assert os.path.exists(path)
    # No tmp file left behind.
    assert not os.path.exists(path + ".tmp")
    loaded = load_checkpoint(path)
    assert loaded.iteration == 3
    assert loaded.active_policy.policy_id == p.policy_id
    assert loaded.replay is not None


def test_checkpoint_roundtrips_candidate_and_decision(tmp_path):
    active = _pol()
    cand = _pol([0.1, 0, 0, 0])
    dec = CandidateDecision(accepted=True, utility_gain=0.2)
    state = CheckpointState(iteration=1, active_policy=active, last_candidate=cand, last_decision=dec)
    path = str(tmp_path / "ckpt2.json")
    save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    assert loaded.last_candidate.policy_id == cand.policy_id
    assert loaded.last_decision.accepted is True


def test_checkpoint_replay_roundtrips(tmp_path):
    from daph_learning.autolearn_v2.experience import Experience
    p = _pol()
    buf = ReplayBuffer(ReplayConfig(capacity=10, batch_size=2))
    buf.add(Experience(
        experience_id="e1", task_id="t1", task_fingerprint="fp1", task_text="s",
        representation_ref=None, policy_id="p1", model_id="m1", dataset_id="d1",
        selected_action="symbolic", outcomes={}, backend_rewards={},
        optimal_action="symbolic", reward_gap=1.0, created_at="ts",
    ))
    state = CheckpointState(iteration=0, active_policy=p, replay=buf)
    path = str(tmp_path / "ckpt3.json")
    save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    assert len(loaded.replay) == 1
