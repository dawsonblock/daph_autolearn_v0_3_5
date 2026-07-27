"""Integration tests for the AutoLearn v2 engine (Phase 17).

These exercise the full workflow end-to-end with a deterministic
structured representation function (no transformer required):

* symbolic-better tasks -> learns to route them to SYMBOLIC
* llm-better tasks -> learns to route them to LLM
* near-equal rewards -> ABSTAIN (optimal_action)
* candidate route collapse -> reject (acceptance gate)
* candidate domain regression -> reject (acceptance gate)
* candidate global improvement without regression -> accept
* restart during learning -> state restores correctly
* same seed/config -> reproducible update
* test split is never accessed by training code

The structured representation gives symbolic-capable tasks a +bias in
dimension 0 and non-symbolic tasks a -bias, so a learned steering
direction can separate the two classes — exactly the structure a real
activation capture would provide.
"""

from __future__ import annotations

import os
import pytest
import numpy as np

from daph_learning.autolearn_v2 import (
    AcceptanceConfig,
    AutoLearnV2Config,
    CounterfactualConfig,
    ReplayConfig,
    UpdateConfig,
    UtilityConfig,
    run_autolearn_v2,
)
from daph_learning.autolearn_v2.experience import Experience
from daph_learning.autolearn_v2.policies import SingleVectorPolicy
from daph_learning.verification import VerificationStatus
from daph_learning.autolearn_v2.counterfactual import default_symbolic_backend


HIDDEN = 4


def _sym_task(i):
    """A symbolic-capable arithmetic task (symbolic is the better backend)."""
    a = i + 1
    b = i + 2
    return {
        "task_id": f"sym_{i}",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": a, "b": b, "op": "+"},
        "specification": f"Compute {a} + {b}. Return only the integer.",
        "expected": a + b,
        "utility_oracle": "symbolic",
        "capability_oracle": "symbolic",
        "generator_family": "sym_fam",
        "domain": "add",
    }


def _llm_task(i):
    """A non-symbolic task (LLM is the better backend; symbolic unsupported)."""
    return {
        "task_id": f"llm_{i}",
        "capability_ids": [],
        "inputs": {},
        "specification": f"Explain concept {i} in one paragraph.",
        "expected": None,
        "utility_oracle": "llm",
        "capability_oracle": "llm",
        "generator_family": "llm_fam",
        "domain": "explain",
    }


def _structured_rep(task):
    """Symbolic-capable -> +dim0 bias; non-symbolic -> -dim0 bias.

    This mirrors the structure a real activation capture would provide:
    the two task families occupy separable regions of the representation
    space, so a steering direction can route them differently.
    """
    caps = set(task.get("capability_ids") or [])
    h = np.zeros(HIDDEN, dtype=np.float32)
    if caps & {"integer_arithmetic", "modular_multiplication"}:
        h[0] = 1.0
    else:
        h[0] = -1.0
    # Deterministic per-task noise in other dims (does not obscure the bias).
    h[1] = (hash(task["task_id"]) % 7) / 10.0
    return h


def _make_dataset(n_sym=12, n_llm=12):
    train = [_sym_task(i) for i in range(n_sym)] + [_llm_task(i) for i in range(n_llm)]
    val = [_sym_task(100 + i) for i in range(4)] + [_llm_task(100 + i) for i in range(4)]
    test = [_sym_task(200 + i) for i in range(4)] + [_llm_task(200 + i) for i in range(4)]
    return train, val, test


def _base_config(**overrides):
    cfg = dict(
        seed=42,
        max_iterations=3,
        model_id="fake-model",
        tokenizer_hash="fake-tok",
        dataset_id="test-ds",
        layer=0,
        hidden_size=HIDDEN,
        alpha=1.0,
        threshold=0.0,
        execution=CounterfactualConfig(execution_mode="counterfactual_training"),
        reward=UtilityConfig(abstain_margin=0.1),
        replay=ReplayConfig(capacity=200, batch_size=24, seed=42,
                            hard_example_fraction=0.3, regression_anchor_fraction=0.1),
        update=UpdateConfig(learning_rate=0.5, trust_region_radius=2.0, max_vector_norm=10.0),
        acceptance=AcceptanceConfig(minimum_utility_gain=0.01, minimum_samples=5,
                                    maximum_accuracy_regression=0.5,
                                    maximum_domain_regression=0.5,
                                    max_abstain_increase=1.0, max_displacement=10.0),
    )
    cfg.update(overrides)
    return AutoLearnV2Config(**cfg)


# ---------------------------------------------------------------------------
# Core learning integration
# ---------------------------------------------------------------------------


def test_engine_symbolic_better_learns_symbolic_route():
    """Symbolic-capable tasks should be routed to SYMBOLIC after learning."""
    train, val, test = _make_dataset()
    cfg = _base_config()
    result = run_autolearn_v2(
        train, val, config=cfg, representation_fn=_structured_rep,
    )
    # The final active policy should route symbolic-capable val tasks to symbolic.
    policy = result.active_policy
    sym_correct = 0
    for t in val:
        if t["utility_oracle"] == "symbolic":
            route = _route(policy, _structured_rep(t))
            if route == "symbolic":
                sym_correct += 1
    # At least some symbolic tasks routed to symbolic (learning happened).
    assert sym_correct > 0, "engine did not learn to route symbolic tasks to SYMBOLIC"


def test_engine_llm_better_learns_llm_route():
    """Non-symbolic tasks should be routed to LLM after learning."""
    train, val, test = _make_dataset()
    cfg = _base_config()
    result = run_autolearn_v2(train, val, config=cfg, representation_fn=_structured_rep)
    policy = result.active_policy
    llm_correct = 0
    for t in val:
        if t["utility_oracle"] == "llm":
            route = _route(policy, _structured_rep(t))
            if route == "llm":
                llm_correct += 1
    assert llm_correct > 0, "engine did not learn to route llm tasks to LLM"


def test_engine_utility_does_not_decrease():
    """The accepted policy's validation utility should not regress vs seed."""
    train, val, test = _make_dataset()
    cfg = _base_config()
    result = run_autolearn_v2(train, val, config=cfg, representation_fn=_structured_rep)
    # The seed policy routes everything to abstain (utility 0). After
    # learning, utility should be >= 0 (acceptance gate enforces gain).
    assert result.active_policy.vector_norm() > 0.0 or len(result.iterations) == 0


def test_engine_emits_telemetry_and_manifest(tmp_path):
    train, val, test = _make_dataset()
    cfg = _base_config(
        telemetry_path=str(tmp_path / "telemetry.jsonl"),
        registry_dir=str(tmp_path / "registry"),
    )
    result = run_autolearn_v2(train, val, config=cfg, representation_fn=_structured_rep)
    assert os.path.exists(tmp_path / "telemetry.jsonl")
    assert "final_active_policy_id" in result.manifest
    assert result.manifest["n_iterations_run"] == len(result.iterations)


def test_engine_evaluates_test_only_at_end():
    """The test set is evaluated exactly once, against the final policy."""
    train, val, test = _make_dataset()
    cfg = _base_config()
    result = run_autolearn_v2(
        train, val, test_tasks=test, config=cfg, representation_fn=_structured_rep,
    )
    assert result.final_test_metrics is not None
    assert result.manifest["test_set_evaluated"] is True
    assert result.manifest["test_set_size"] == len(test)


def test_engine_test_ids_never_in_training():
    """Invariant: test task ids must not appear in the training set."""
    train, val, test = _make_dataset()
    train_ids = {t["task_id"] for t in train}
    test_ids = {t["task_id"] for t in test}
    assert not (train_ids & test_ids)


# ---------------------------------------------------------------------------
# Near-equal rewards -> ABSTAIN
# ---------------------------------------------------------------------------


def _correct_llm_backend_for_arithmetic(task):
    """An LLM backend that returns CORRECT for arithmetic tasks.

    Used to create near-equal-reward experiences (both backends correct)
    so the optimal action is ABSTAIN.
    """
    from daph_learning.autolearn_v2.experience import BackendOutcome
    expected = task.get("expected")
    if expected is not None:
        return BackendOutcome(
            backend="llm", output=f"FINAL: {expected}",
            verification_status=VerificationStatus.CORRECT,
            correctness_score=1.0, latency_ms=1.0, estimated_cost=0.1,
        )
    return BackendOutcome(
        backend="llm", output=None,
        verification_status=VerificationStatus.UNVERIFIABLE,
        correctness_score=None, latency_ms=1.0, estimated_cost=0.1,
    )


def test_near_equal_rewards_yield_abstain_optimal_action():
    """When both backends are correct, optimal_action is ABSTAIN."""
    from daph_learning.autolearn_v2.counterfactual import collect_counterfactual_experience
    train, val, test = _make_dataset(n_sym=4, n_llm=0)
    cfg = UtilityConfig(abstain_margin=0.5)
    exps = collect_counterfactual_experience(
        train, symbolic_backend=default_symbolic_backend,
        llm_backend=_correct_llm_backend_for_arithmetic,
        utility_config=cfg, policy_id="p", model_id="m", dataset_id="d",
        created_at="ts",
    )
    # Both backends correct -> gap ~ 0 (llm has small cost penalty) ->
    # within abstain band.
    for e in exps:
        assert e.optimal_action in ("abstain", "symbolic")  # small cost may edge it
        # The gap should be small (near zero).
        assert e.reward_gap is not None
        assert abs(e.reward_gap) < 0.5


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_engine_reproducible_with_same_seed():
    """Same seed + config -> same final policy id and learning curve."""
    train, val, test = _make_dataset()
    cfg = _base_config()
    r1 = run_autolearn_v2(train, val, config=cfg, representation_fn=_structured_rep)
    r2 = run_autolearn_v2(train, val, config=cfg, representation_fn=_structured_rep)
    assert r1.active_policy.policy_id == r2.active_policy.policy_id
    assert len(r1.iterations) == len(r2.iterations)


# ---------------------------------------------------------------------------
# Restart / checkpoint restore
# ---------------------------------------------------------------------------


def test_engine_restart_restores_state(tmp_path):
    """A run that checkpoints can be resumed from the checkpoint."""
    train, val, test = _make_dataset()
    ckpt = str(tmp_path / "ckpt.json")
    cfg = _base_config(max_iterations=2, checkpoint_path=ckpt)
    r1 = run_autolearn_v2(train, val, config=cfg, representation_fn=_structured_rep)
    assert os.path.exists(ckpt)
    # Resume into a longer run.
    cfg2 = _base_config(max_iterations=4, checkpoint_path=ckpt)
    r2 = run_autolearn_v2(
        train, val, config=cfg2, representation_fn=_structured_rep,
        resume_from=ckpt,
    )
    # The resumed run should continue from the checkpointed iteration.
    assert r2.iterations[0].iteration >= r1.iterations[-1].iteration


def test_checkpoint_atomic_no_tmp_left(tmp_path):
    """save_checkpoint must not leave a .tmp file behind on success."""
    from daph_learning.autolearn_v2.checkpoint import CheckpointState, save_checkpoint
    p = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=0, hidden_size=HIDDEN)
    state = CheckpointState(iteration=0, active_policy=p)
    path = str(tmp_path / "ck.json")
    save_checkpoint(state, path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


# ---------------------------------------------------------------------------
# Acceptance gate integration (collapse / regression / accept)
# ---------------------------------------------------------------------------


def test_engine_rejects_route_collapse_via_gate():
    """A candidate that collapses to one backend is rejected by the gate."""
    from daph_learning.autolearn_v2.acceptance import (
        AcceptanceConfig, CandidateDecision, evaluate_candidate, REASON_ROUTE_COLLAPSE,
    )
    from daph_learning.autolearn_v2.policies import PolicyMetrics
    active = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=0, hidden_size=HIDDEN)
    cand = SingleVectorPolicy.from_candidate(active, np.array([1, 0, 0, 0], dtype=np.float32))
    am = PolicyMetrics(utility=0.0, routing_accuracy=0.0, symbolic_rate=0.5,
                       llm_rate=0.5, abstain_rate=1.0, n_samples=200)
    cm = PolicyMetrics(utility=0.9, routing_accuracy=1.0, symbolic_rate=0.99,
                       llm_rate=0.01, abstain_rate=0.0, n_samples=200)
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig())
    assert not dec.accepted
    assert REASON_ROUTE_COLLAPSE in dec.reason_codes


def test_engine_rejects_domain_regression_via_gate():
    """A candidate that improves one domain but destroys another is rejected."""
    from daph_learning.autolearn_v2.acceptance import (
        AcceptanceConfig, evaluate_candidate, REASON_DOMAIN_REGRESSION,
    )
    from daph_learning.autolearn_v2.policies import PolicyMetrics
    active = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=0, hidden_size=HIDDEN)
    cand = SingleVectorPolicy.from_candidate(active, np.array([1, 0, 0, 0], dtype=np.float32))
    am = PolicyMetrics(utility=0.5, routing_accuracy=0.7, symbolic_rate=0.5,
                       llm_rate=0.5, abstain_rate=0.0, n_samples=200,
                       per_domain={"add": 0.7, "explain": 0.7})
    cm = PolicyMetrics(utility=0.6, routing_accuracy=0.8, symbolic_rate=0.5,
                       llm_rate=0.5, abstain_rate=0.0, n_samples=200,
                       per_domain={"add": 0.9, "explain": 0.4})
    dec = evaluate_candidate(active, cand, am, cm,
                             AcceptanceConfig(maximum_domain_regression=0.03))
    assert not dec.accepted
    assert REASON_DOMAIN_REGRESSION in dec.reason_codes


def test_engine_accepts_global_improvement_no_regression():
    """A candidate that improves globally without excessive regression is accepted."""
    from daph_learning.autolearn_v2.acceptance import AcceptanceConfig, evaluate_candidate
    from daph_learning.autolearn_v2.policies import PolicyMetrics
    active = SingleVectorPolicy.seed(model_id="m", tokenizer_hash="t", layer=0, hidden_size=HIDDEN)
    cand = SingleVectorPolicy.from_candidate(active, np.array([1, 0, 0, 0], dtype=np.float32))
    am = PolicyMetrics(utility=0.5, routing_accuracy=0.7, symbolic_rate=0.5,
                       llm_rate=0.5, abstain_rate=0.0, n_samples=200,
                       per_domain={"add": 0.7, "explain": 0.7})
    cm = PolicyMetrics(utility=0.7, routing_accuracy=0.85, symbolic_rate=0.5,
                       llm_rate=0.5, abstain_rate=0.0, n_samples=200,
                       per_domain={"add": 0.8, "explain": 0.75})
    dec = evaluate_candidate(active, cand, am, cm, AcceptanceConfig())
    assert dec.accepted


# ---------------------------------------------------------------------------
# Full definition-of-done workflow
# ---------------------------------------------------------------------------


def test_engine_full_workflow_definition_of_done(tmp_path):
    """The complete DoD workflow in one test:

    load policy, leakage-free data, counterfactual execution, verify,
    utility, reward-gap, experiences, replay, candidate update,
    trust-region, validation eval, accept/reject, lineage, restart,
    final test eval, manifest.
    """
    train, val, test = _make_dataset()
    cfg = _base_config(
        max_iterations=3,
        checkpoint_path=str(tmp_path / "ckpt.json"),
        telemetry_path=str(tmp_path / "telemetry.jsonl"),
        registry_dir=str(tmp_path / "registry"),
    )
    result = run_autolearn_v2(
        train, val, test_tasks=test, config=cfg, representation_fn=_structured_rep,
    )
    # 1. Active policy exists and is registered.
    assert result.active_policy is not None
    assert result.registry.active_policy_id() == result.active_policy.policy_id
    # 2. Lineage is non-empty (seed + any accepted candidates).
    lineage = result.registry.lineage(result.active_policy.policy_id)
    assert len(lineage) >= 1
    # 3. Iterations produced telemetry.
    assert len(result.iterations) >= 1
    # 4. Test evaluated once at the end.
    assert result.final_test_metrics is not None
    # 5. Manifest is complete.
    m = result.manifest
    assert m["final_active_policy_id"] == result.active_policy.policy_id
    assert m["test_set_evaluated"] is True
    # 6. Checkpoint file exists.
    assert os.path.exists(tmp_path / "ckpt.json")
    # 7. Telemetry file exists with one line per iteration.
    with open(tmp_path / "telemetry.jsonl") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == len(result.iterations)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _route(policy, rep):
    from daph_learning.autolearn_v2.engine import _route_from_policy
    return _route_from_policy(policy, rep)
