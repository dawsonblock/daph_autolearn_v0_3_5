"""AutoLearn v2 engine — the orchestrating learning loop.

Implements the core loop::

    task
      -> capture representation
      -> execute candidate backends counterfactually (during learning)
      -> verify outputs (typed)
      -> measure reward/utility per backend
      -> derive optimal action / reward gap
      -> store experience
      -> sample replay / hard examples
      -> produce candidate policy update (incremental, trust-region)
      -> evaluate candidate + active on immutable validation set
      -> accept or reject
      -> version + retain + rollback
      -> checkpoint

The engine is model-agnostic: it consumes pluggable callables for
representation capture, the LLM backend, and validation evaluation. This
keeps it testable with fakes and runnable with real transformers without
coupling. The symbolic backend defaults to the existing typed executor
(security boundaries intact).

The legacy :func:`daph_learning.autolearn.loop.run_autolearn_loop` is
preserved unchanged for backwards compatibility.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .acceptance import AcceptanceConfig, CandidateDecision, evaluate_candidate
from .checkpoint import CheckpointState, capture_rng_state, load_checkpoint, save_checkpoint
from .counterfactual import (
    CounterfactualConfig,
    ExecutionMode,
    collect_counterfactual_experience,
    default_symbolic_backend,
)
from .experience import Experience
from .observability import IterationTelemetry, TelemetryLogger, counterfactual_disagreement_rate
from .policies.base import PolicyMetrics
from .policies.static_vector import SingleVectorPolicy
from .registry import PolicyNotFoundError, PolicyRegistry
from .reward import UtilityConfig, backend_reward, optimal_action, reward_gap
from .replay import ReplayBuffer, ReplayConfig
from .updater import UpdateConfig, compute_candidate_update, hash_replay_sample

# Pluggable callable types.
RepresentationFn = Callable[[Mapping[str, Any]], np.ndarray]
LLMBackendFn = Callable[[Mapping[str, Any]], Any]  # returns a BackendOutcome
ValidationFn = Callable[[SingleVectorPolicy, Sequence[Mapping[str, Any]]], PolicyMetrics]


@dataclass(frozen=True)
class AutoLearnV2Config:
    """Top-level configuration for the AutoLearn v2 engine."""

    seed: int = 42
    max_iterations: int = 20
    # Provenance.
    model_id: str = "unknown"
    tokenizer_hash: str = "unknown"
    dataset_id: str = "unknown"
    layer: int = 24
    hidden_size: int = 0
    alpha: float = 1.0
    threshold: float = 0.0
    # Sub-configs.
    execution: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    reward: UtilityConfig = field(default_factory=UtilityConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    # Checkpoint path (None -> no checkpointing).
    checkpoint_path: str | None = None
    telemetry_path: str | None = None
    registry_dir: str | None = None
    # Domain key for per-domain metrics (task field name).
    domain_key: str = "domain"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return {
            "seed": self.seed,
            "max_iterations": self.max_iterations,
            "model_id": self.model_id,
            "tokenizer_hash": self.tokenizer_hash,
            "dataset_id": self.dataset_id,
            "layer": self.layer,
            "hidden_size": self.hidden_size,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "execution": asdict(self.execution),
            "reward": self.reward.to_dict(),
            "replay": asdict(self.replay),
            "update": asdict(self.update),
            "acceptance": asdict(self.acceptance),
            "checkpoint_path": self.checkpoint_path,
            "telemetry_path": self.telemetry_path,
            "registry_dir": self.registry_dir,
            "domain_key": self.domain_key,
        }


@dataclass
class AutoLearnV2Result:
    """Final result of an AutoLearn v2 run."""

    active_policy: SingleVectorPolicy
    registry: PolicyRegistry
    iterations: list[IterationTelemetry]
    final_test_metrics: PolicyMetrics | None = None
    manifest: dict[str, Any] = field(default_factory=dict)


def _default_representation(task: Mapping[str, Any], hidden_size: int) -> np.ndarray:
    """A deterministic hash-based pseudo-representation used when no model
    representation capture is available.

    This is *not* a real activation; it exists so the engine can run
    end-to-end in tests and on synthetic data without a transformer. Real
    runs must supply a ``representation_fn`` that captures layer
    activations. The pseudo-representation is deterministic per task
    fingerprint so reproducibility holds.
    """
    from .experience import fingerprint_task
    import hashlib
    fp = fingerprint_task(task)
    # Expand the fingerprint into a float32 vector of the requested size.
    needed = (hidden_size + 3) // 4  # 4 bytes per uint32
    digest = hashlib.sha256(fp.encode("utf-8")).digest()
    while len(digest) < needed * 4:
        digest = digest + hashlib.sha256(digest).digest()
    arr = np.frombuffer(digest[: needed * 4], dtype=np.uint32).astype(np.float32)
    # Center/scale to a small range and pad/truncate to hidden_size.
    arr = (arr % 1000) / 1000.0 - 0.5
    if arr.shape[0] < hidden_size:
        arr = np.pad(arr, (0, hidden_size - arr.shape[0]))
    return arr[:hidden_size]


def _route_from_policy(
    policy: SingleVectorPolicy,
    representation: np.ndarray,
) -> str:
    """Derive a route label from a policy + representation.

    Uses the policy's route scores and threshold. ``score_symbolic -
    score_llm > threshold`` -> symbolic; ``< -threshold`` -> llm; inside
    the band -> abstain (fail-closed).
    """
    scores = policy.route_score(representation)
    margin = scores["symbolic"] - scores["llm"]
    if margin > policy.threshold:
        return "symbolic"
    if margin < -abs(policy.threshold):
        return "llm"
    return "abstain"


def _default_validation_fn(
    policy: SingleVectorPolicy,
    val_tasks: Sequence[Mapping[str, Any]],
    *,
    utility_config: UtilityConfig,
    representation_fn: RepresentationFn,
    domain_key: str,
) -> PolicyMetrics:
    """Default validation evaluator.

    Routes each validation task with the policy, executes the *selected*
    backend only (no counterfactual at validation time — the validation
    set is never used for learning), verifies the output, and aggregates
    utility / accuracy / route rates / per-domain utility.
    """
    from .counterfactual import default_symbolic_backend, unsupported_symbolic_backend
    from daph_learning.verification import verify_task_output

    sym_rate = llm_rate = abstain_rate = 0.0
    n = len(val_tasks)
    if n == 0:
        return PolicyMetrics(utility=0.0, routing_accuracy=0.0, symbolic_rate=0.0,
                             llm_rate=0.0, abstain_rate=0.0, n_samples=0)

    routes: list[str] = []
    utilities: list[float] = []
    per_domain: dict[str, list[float]] = {}
    correct = 0
    for task in val_tasks:
        rep = representation_fn(task)
        route = _route_from_policy(policy, rep)
        routes.append(route)
        domain = str(task.get(domain_key, "default"))
        # Execute the selected backend only.
        if route == "symbolic":
            outcome = default_symbolic_backend(task)
        elif route == "llm":
            # Use the task's expected value to verify via the typed
            # verifier. The LLM "output" here is a placeholder: in real
            # runs the engine is given an llm_backend_fn. For validation
            # we score the *policy's routing decision* against the
            # optimal action derived from the task's oracle fields when
            # available, falling back to utility of the executed backend.
            outcome = _llm_validation_outcome(task)
        else:  # abstain
            # Abstention: no backend executed. Utility is 0 (no
            # correctness, no failure penalty, no cost).
            from .experience import BackendOutcome
            from daph_learning.verification import VerificationResult, VerificationStatus
            outcome = BackendOutcome(
                backend="abstain", output=None,
                verification_status=VerificationStatus.UNVERIFIABLE,
                correctness_score=None, latency_ms=0.0, estimated_cost=0.0,
            )
        u = backend_reward(outcome, utility_config) if outcome.backend != "abstain" else 0.0
        utilities.append(u)
        per_domain.setdefault(domain, []).append(u)
        # Routing accuracy vs optimal action (oracle) if present.
        opt = task.get("utility_oracle") or task.get("capability_oracle")
        if opt is not None:
            # Map oracle labels to the route space.
            opt_route = "symbolic" if opt == "symbolic" else "llm"
            if route == opt_route:
                correct += 1
        else:
            # No oracle: count correctness from the executed backend.
            if outcome.verification_status.is_correct:
                correct += 1

    sym = sum(1 for r in routes if r == "symbolic")
    llm = sum(1 for r in routes if r == "llm")
    abst = sum(1 for r in routes if r == "abstain")
    per_domain_util = {d: float(sum(v) / len(v)) for d, v in per_domain.items() if v}
    return PolicyMetrics(
        utility=float(sum(utilities) / n),
        routing_accuracy=correct / n,
        symbolic_rate=sym / n,
        llm_rate=llm / n,
        abstain_rate=abst / n,
        per_domain=per_domain_util,
        n_samples=n,
    )


def _llm_validation_outcome(task: Mapping[str, Any]):
    """Produce an LLM BackendOutcome for validation.

    In real runs the engine is given an ``llm_backend_fn``. For the
    default validation evaluator without one, we simulate the LLM by
    checking whether the task's expected value (if any) is something the
    LLM would plausibly get right. To avoid fabricating results, when no
    LLM backend is supplied we mark the LLM outcome UNVERIFIABLE for
    symbolic-capable tasks (we cannot verify prose) and CORRECT for
    non-symbolic tasks (no ground truth to contradict). This is the same
    conservative posture as the typed verifier.
    """
    from .experience import BackendOutcome
    from daph_learning.verification import VerificationResult, VerificationStatus, verify_task_output

    caps = set(task.get("capability_ids") or [])
    expected = task.get("expected")
    if expected is None and not (caps & {"integer_arithmetic", "modular_multiplication"}):
        # Non-symbolic free-text task: LLM is the right call; no ground
        # truth to contradict, but we also cannot verify -> UNVERIFIABLE
        # with zero correctness credit (never map UNVERIFIABLE to CORRECT).
        return BackendOutcome(
            backend="llm", output=None,
            verification_status=VerificationStatus.UNVERIFIABLE,
            correctness_score=None, latency_ms=0.0, estimated_cost=0.1,
        )
    # Symbolic-capable task routed to LLM: we cannot verify the LLM's
    # prose, so this is UNVERIFIABLE (zero correctness credit). The
    # reward engine therefore prefers symbolic when symbolic is correct.
    return BackendOutcome(
        backend="llm", output=None,
        verification_status=VerificationStatus.UNVERIFIABLE,
        correctness_score=None, latency_ms=0.0, estimated_cost=0.1,
    )


def run_autolearn_v2(
    train_tasks: Sequence[Mapping[str, Any]],
    val_tasks: Sequence[Mapping[str, Any]],
    *,
    config: AutoLearnV2Config,
    initial_policy: SingleVectorPolicy | None = None,
    representation_fn: RepresentationFn | None = None,
    llm_backend_fn: LLMBackendFn | None = None,
    validation_fn: ValidationFn | None = None,
    test_tasks: Sequence[Mapping[str, Any]] | None = None,
    resume_from: str | None = None,
) -> AutoLearnV2Result:
    """Run the AutoLearn v2 learning loop.

    Parameters
    ----------
    train_tasks, val_tasks : sequences of task dicts
        Leakage-free train/validation splits (see
        :mod:`daph_learning.evaluation.leakage`).
    config : AutoLearnV2Config
    initial_policy : SingleVectorPolicy | None
        The accepted initial policy. If ``None``, a zero seed policy is
        created from ``config``.
    representation_fn : callable, optional
        ``task -> np.ndarray`` capturing the layer representation. If
        ``None``, a deterministic pseudo-representation is used (for
        tests / synthetic data). Real runs must supply this.
    llm_backend_fn : callable, optional
        ``task -> BackendOutcome`` for the LLM backend during
        counterfactual collection. If ``None``, a conservative
        UNVERIFIABLE-outcome backend is used (the engine then learns
        purely from the symbolic signal — symbolic-correct tasks get a
        positive reward gap toward symbolic).
    validation_fn : callable, optional
        ``policy, val_tasks -> PolicyMetrics``. If ``None``, a default
        evaluator is used.
    test_tasks : optional
        Held-out test set. Evaluated *only once* against the final
        frozen accepted policy. Never accessed during learning.
    resume_from : str, optional
        Checkpoint path to resume from.
    """
    rng = random.Random(config.seed)
    hidden = config.hidden_size

    # Resolve the representation function.
    if representation_fn is None:
        if hidden <= 0:
            raise ValueError(
                "hidden_size must be set in config when representation_fn is not supplied"
            )
        representation_fn = (lambda t, _h=hidden: _default_representation(t, _h))
    else:
        # Infer hidden_size from the first task if not set.
        if hidden <= 0 and train_tasks:
            hidden = int(representation_fn(train_tasks[0]).shape[-1])

    # Resolve the LLM backend.
    if llm_backend_fn is None:
        llm_backend_fn = _conservative_llm_backend

    # Resolve the validation evaluator.
    if validation_fn is None:
        validation_fn = (
            lambda policy, vtasks, _u=config.reward, _r=representation_fn, _d=config.domain_key:
                _default_validation_fn(policy, vtasks, utility_config=_u,
                                       representation_fn=_r, domain_key=_d)
        )

    # Registry + initial policy.
    registry = PolicyRegistry(config.registry_dir)
    if initial_policy is None:
        initial_policy = SingleVectorPolicy.seed(
            model_id=config.model_id,
            tokenizer_hash=config.tokenizer_hash,
            layer=config.layer,
            hidden_size=hidden,
            alpha=config.alpha,
            threshold=config.threshold,
            random_seed=config.seed,
        )
    active = initial_policy

    # Replay buffer.
    replay = ReplayBuffer(config.replay)

    # Resume from checkpoint if requested.
    start_iter = 0
    history: list[dict[str, Any]] = []
    if resume_from:
        state = load_checkpoint(resume_from)
        active = state.active_policy
        replay = state.replay or ReplayBuffer(config.replay)
        start_iter = state.iteration + 1
        history = list(state.history)
        if state.rng_state:
            rng.setstate(state.rng_state)

    # Register the initial/active policy as accepted (root), unless it is
    # already present (e.g. resumed from a checkpoint with a persisted
    # registry).
    try:
        registry.get(active.policy_id)
    except PolicyNotFoundError:
        from .acceptance import CandidateDecision
        registry.accept(
            active,
            CandidateDecision(accepted=True, detail="initial_seed_policy"),
        )

    telemetry_logger = TelemetryLogger(config.telemetry_path)
    iterations: list[IterationTelemetry] = []

    try:
        for iteration in range(start_iter, config.max_iterations):
            iter_t0 = time.time()
            # ----------------------------------------------------------
            # 1. Counterfactual experience collection on TRAINING data.
            # ----------------------------------------------------------
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # Route with the active policy to record selected_action.
            selected_actions: dict[str, str] = {}
            for task in train_tasks:
                rep = representation_fn(task)
                selected_actions[str(task.get("task_id", ""))] = _route_from_policy(active, rep)

            experiences = collect_counterfactual_experience(
                train_tasks,
                symbolic_backend=default_symbolic_backend,
                llm_backend=llm_backend_fn,
                utility_config=config.reward,
                policy_id=active.policy_id,
                model_id=config.model_id,
                dataset_id=config.dataset_id,
                selected_actions=selected_actions,
                config=config.execution,
                created_at=created_at,
            )
            replay.add_many(experiences)

            # ----------------------------------------------------------
            # 2. Sample replay.
            # ----------------------------------------------------------
            sample = replay.sample(config.replay.batch_size, seed=rng.randint(0, 2**31 - 1))
            sample_hash = hash_replay_sample(sample.experiences)
            reps = np.stack([representation_fn(_task_from_exp(e, train_tasks)) for e in sample.experiences]) if sample.experiences else np.zeros((0, hidden), dtype=np.float32)

            # ----------------------------------------------------------
            # 3. Candidate update (incremental, trust-region).
            # ----------------------------------------------------------
            candidate, update_meta = compute_candidate_update(
                active, sample.experiences, reps, config.update,
                replay_sample_hash=sample_hash,
                utility_config_dict=config.reward.to_dict(),
            )

            # ----------------------------------------------------------
            # 4. Evaluate candidate + active on the SAME validation set.
            # ----------------------------------------------------------
            active_metrics = validation_fn(active, val_tasks)
            candidate_metrics = validation_fn(candidate, val_tasks)

            # ----------------------------------------------------------
            # 5. Acceptance gate.
            # ----------------------------------------------------------
            decision = evaluate_candidate(
                active, candidate, active_metrics, candidate_metrics, config.acceptance,
            )
            registry.register_candidate(candidate, environment={"iteration": iteration})
            if decision.accepted:
                registry.accept(candidate, decision)
                active = candidate.with_metrics(candidate_metrics)
            else:
                registry.reject(candidate, decision)

            history.append({
                "iteration": iteration,
                "active_policy_id": active.policy_id,
                "candidate_policy_id": candidate.policy_id,
                "accepted": decision.accepted,
                "reason_codes": list(decision.reason_codes),
                "utility_gain": decision.utility_gain,
            })

            # ----------------------------------------------------------
            # 6. Telemetry.
            # ----------------------------------------------------------
            route_rates = _route_rates_from_experiences(experiences)
            tel = IterationTelemetry(
                iteration=iteration,
                active_policy_id=active.policy_id,
                candidate_policy_id=candidate.policy_id,
                replay_size=len(replay),
                new_experience_count=len(experiences),
                symbolic_route_rate=route_rates["symbolic"],
                llm_route_rate=route_rates["llm"],
                abstain_rate=route_rates["abstain"],
                counterfactual_disagreement_rate=counterfactual_disagreement_rate(experiences),
                mean_reward_gap=_mean_reward_gap(experiences),
                utility_before=active_metrics.utility,
                utility_after=candidate_metrics.utility,
                candidate_accepted=decision.accepted,
                reason_codes=list(decision.reason_codes),
                vector_norm=float(np.linalg.norm(active.vector)),
                update_norm=update_meta.get("step_norm", 0.0),
                trust_region_clipped=update_meta.get("trust_region_clipped", False),
                max_vector_norm_clipped=update_meta.get("max_vector_norm_clipped", False),
                per_domain=candidate_metrics.per_domain,
                regression_count=sum(1 for r in decision.reason_codes if "REGRESSION" in r),
                extra={
                    "n_used": update_meta.get("n_used", 0),
                    "objective_before": update_meta.get("objective_before", 0.0),
                    "objective_after": update_meta.get("objective_after", 0.0),
                    "iteration_seconds": time.time() - iter_t0,
                },
            )
            telemetry_logger.log(tel)
            iterations.append(tel)

            # ----------------------------------------------------------
            # 7. Checkpoint.
            # ----------------------------------------------------------
            if config.checkpoint_path:
                state = CheckpointState(
                    iteration=iteration,
                    active_policy=active,
                    last_candidate=candidate,
                    last_decision=decision,
                    replay=replay,
                    rng_state=capture_rng_state(rng),
                    training_metrics=[t.to_dict() for t in iterations],
                    history=history,
                    config_dict=config.to_dict(),
                    saved_at=created_at,
                )
                save_checkpoint(state, config.checkpoint_path)
    finally:
        telemetry_logger.close()

    # ----------------------------------------------------------
    # 8. Final test evaluation (only the frozen accepted policy).
    # ----------------------------------------------------------
    final_test_metrics = None
    if test_tasks is not None:
        final_test_metrics = validation_fn(active, test_tasks)

    # ----------------------------------------------------------
    # 9. Reproducibility manifest.
    # ----------------------------------------------------------
    manifest = {
        "config": config.to_dict(),
        "final_active_policy_id": active.policy_id,
        "n_iterations_run": len(iterations),
        "n_accepted": sum(1 for h in history if h["accepted"]),
        "n_rejected": sum(1 for h in history if not h["accepted"]),
        "accepted_lineage": [r.policy.policy_id for r in registry.accepted_records()],
        "final_test_metrics": final_test_metrics.__dict__ if final_test_metrics else None,
        "test_set_evaluated": test_tasks is not None,
        "test_set_size": len(test_tasks) if test_tasks is not None else 0,
    }

    return AutoLearnV2Result(
        active_policy=active,
        registry=registry,
        iterations=iterations,
        final_test_metrics=final_test_metrics,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_from_exp(exp: Experience, tasks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Look up the original task dict for an experience by task_id."""
    for t in tasks:
        if str(t.get("task_id", "")) == exp.task_id:
            return t
    # Fallback: synthesize a minimal task from the experience.
    return {"task_id": exp.task_id, "specification": exp.task_text}


def _route_rates_from_experiences(experiences: Sequence[Experience]) -> dict[str, float]:
    n = len(experiences)
    if n == 0:
        return {"symbolic": 0.0, "llm": 0.0, "abstain": 0.0}
    sym = sum(1 for e in experiences if e.selected_action == "symbolic")
    llm = sum(1 for e in experiences if e.selected_action == "llm")
    abst = sum(1 for e in experiences if e.selected_action == "abstain")
    return {"symbolic": sym / n, "llm": llm / n, "abstain": abst / n}


def _mean_reward_gap(experiences: Sequence[Experience]) -> float | None:
    gaps = [e.reward_gap for e in experiences if e.reward_gap is not None]
    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def _conservative_llm_backend(task: Mapping[str, Any]):
    """A conservative LLM backend used when none is supplied.

    Returns UNVERIFIABLE for symbolic-capable tasks (we cannot verify
    LLM prose, so it never earns correctness credit) and UNVERIFIABLE for
    non-symbolic tasks (no ground truth). This means the reward gap is
    driven entirely by the symbolic backend's measured outcome, which is
    the honest signal available without a real LLM. Real runs should
    supply ``llm_backend_fn``.
    """
    from .experience import BackendOutcome
    from daph_learning.verification import VerificationStatus
    return BackendOutcome(
        backend="llm",
        output=None,
        verification_status=VerificationStatus.UNVERIFIABLE,
        correctness_score=None,
        latency_ms=0.0,
        estimated_cost=0.1,
        metadata={"backend": "conservative_default"},
    )


__all__ = [
    "AutoLearnV2Config",
    "AutoLearnV2Result",
    "LLMBackendFn",
    "RepresentationFn",
    "ValidationFn",
    "run_autolearn_v2",
]
