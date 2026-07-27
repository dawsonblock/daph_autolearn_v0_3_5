"""Counterfactual execution (AutoLearn v2, Phase 3).

During training and dedicated learning-data generation, execute *both*
SYMBOLIC and LLM backends on tasks where both are valid candidates, then
verify each output and measure reward. This produces the
counterfactual signal the reward-gap learner needs.

At normal production inference time only the *selected* backend runs
(``execution_mode="selected_only"``). A ``"shadow"`` mode runs the
non-selected backend without affecting the user-visible output, for
online experience collection.

Backends are pluggable callables so the collector is testable with fakes
and usable with real models without coupling to the transformer stack:

    SymbolicBackend = Callable[[Task], BackendOutcome]
    LLBackend      = Callable[[Task], BackendOutcome]

A backend callable returns a :class:`BackendOutcome` (already verified
and timed). The collector assembles them into :class:`Experience`
records with reward gaps and optimal actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from daph_learning.execution.plan import ExecutionResult
from daph_learning.execution.symbolic_executor import (
    execute_plan,
    plan_from_structured_task,
)
from daph_learning.tools.symbolic_math import SymbolicMathError
from daph_learning.verification import (
    VerificationResult,
    VerificationStatus,
    verify_task_output,
)

from .experience import BackendOutcome, Experience, fingerprint_task, make_experience_id
from .reward import UtilityConfig, backend_reward, optimal_action, reward_gap

ExecutionMode = Literal["selected_only", "counterfactual_training", "shadow"]

# A backend callable: task -> BackendOutcome (already verified/timed).
SymbolicBackend = Callable[[Mapping[str, Any]], BackendOutcome]
LLMBackend = Callable[[Mapping[str, Any]], BackendOutcome]


# ---------------------------------------------------------------------------
# Default symbolic backend (reuses the existing typed executor)
# ---------------------------------------------------------------------------


def default_symbolic_backend(task: Mapping[str, Any]) -> BackendOutcome:
    """Run the typed symbolic executor and verify the result.

    Reuses :func:`daph_learning.execution.symbolic_executor.execute_plan`
    so the existing symbolic execution security boundaries (no eval/exec,
    bounded AST integer evaluator) remain intact.
    """
    t0 = time.perf_counter()
    try:
        plan = plan_from_structured_task(task, reason_code="autolearn_v2")
        result: ExecutionResult = execute_plan(plan)
        # Round to 1ms precision for reproducibility: sub-millisecond jitter
        # from wall-clock timing must not leak into the reward and break
        # deterministic replay. Real latency differences are >1ms.
        latency_ms = float(round((time.perf_counter() - t0) * 1000.0))
        output = f"FINAL: {result.value}"
        verification = verify_task_output(
            task, output, backend="symbolic", symbolic_verified=result.verified
        )
        return BackendOutcome.from_verification(
            backend="symbolic",
            output=output,
            verification=verification,
            latency_ms=latency_ms,
            estimated_cost=0.0,
            confidence=1.0 if result.verified else None,
            metadata={
                "planner_source": result.metadata.get("planner_source"),
                "reason_code": result.metadata.get("reason_code"),
            },
        )
    except (SymbolicMathError, ValueError, TypeError, KeyError) as exc:
        latency_ms = float(round((time.perf_counter() - t0) * 1000.0))
        failure_type = type(exc).__name__
        verification = verify_task_output(
            task, None, backend="symbolic", failure_type=failure_type
        )
        return BackendOutcome.from_verification(
            backend="symbolic",
            output=None,
            verification=verification,
            latency_ms=latency_ms,
            estimated_cost=0.0,
            failure_type=failure_type,
            metadata={"reason": str(exc)},
        )


def unsupported_symbolic_backend(task: Mapping[str, Any]) -> BackendOutcome:
    """A symbolic backend that reports the task as unsupported.

    Used for tasks with no symbolic capability: the symbolic backend is a
    *valid candidate* but immediately fails, which is itself a
    counterfactual signal (R_symbolic is low because of EXECUTION_ERROR).
    """
    return BackendOutcome.from_verification(
        backend="symbolic",
        output=None,
        verification=VerificationResult(
            status=VerificationStatus.EXECUTION_ERROR,
            detail="unsupported symbolic capability",
            verifier="unsupported",
        ),
        latency_ms=0.0,
        estimated_cost=0.0,
        failure_type="unsupported_capability",
    )


def _is_symbolic_candidate(task: Mapping[str, Any]) -> bool:
    caps = set(task.get("capability_ids") or [])
    return bool(caps & {"integer_arithmetic", "modular_multiplication"})


# ---------------------------------------------------------------------------
# Counterfactual collector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualConfig:
    """Configuration for counterfactual experience collection."""

    execution_mode: ExecutionMode = "counterfactual_training"
    # When True, tasks with no symbolic capability still run the symbolic
    # backend via ``unsupported_symbolic_backend`` so the reward gap
    # captures "symbolic would have failed here" as a negative signal.
    include_unsupported_symbolic: bool = True


def _run_timed(backend_fn: Callable[[Mapping[str, Any]], BackendOutcome], task: Mapping[str, Any]) -> BackendOutcome:
    """Run a backend callable; the callable is responsible for its own timing."""
    return backend_fn(task)


def collect_counterfactual_experience(
    tasks: Sequence[Mapping[str, Any]],
    *,
    symbolic_backend: SymbolicBackend,
    llm_backend: LLMBackend,
    utility_config: UtilityConfig,
    policy_id: str,
    model_id: str,
    dataset_id: str,
    selected_actions: Mapping[str, str] | None = None,
    config: CounterfactualConfig | None = None,
    created_at: str = "",
    representation_refs: Mapping[str, str] | None = None,
) -> list[Experience]:
    """Run both backends counterfactually and assemble experiences.

    Parameters
    ----------
    tasks : sequence of task dicts
    symbolic_backend, llm_backend : callables
        Each returns a :class:`BackendOutcome` for one task.
    utility_config : UtilityConfig
        Used to compute per-backend rewards, reward gap, optimal action.
    policy_id, model_id, dataset_id : str
        Provenance stamped onto every experience.
    selected_actions : mapping task_id -> action, optional
        The action the active policy actually selected (for the
        ``selected_action`` field). When omitted, defaults to ``"llm"``.
    config : CounterfactualConfig | None
    created_at : str
        ISO timestamp; when empty the caller should set it (kept explicit
        for determinism in tests).
    representation_refs : mapping task_id -> ref, optional
        References to captured representations (e.g. activation tensor
        paths), stored on the experience for the updater.
    """
    cfg = config or CounterfactualConfig()
    experiences: list[Experience] = []

    for task in tasks:
        tid = str(task.get("task_id", ""))
        outcomes: dict[str, BackendOutcome] = {}

        # Symbolic backend.
        if _is_symbolic_candidate(task):
            outcomes["symbolic"] = _run_timed(symbolic_backend, task)
        elif cfg.include_unsupported_symbolic:
            outcomes["symbolic"] = unsupported_symbolic_backend(task)
        # else: symbolic is not a candidate at all -> omitted.

        # LLM backend always runs in counterfactual_training / shadow.
        if cfg.execution_mode in ("counterfactual_training", "shadow"):
            outcomes["llm"] = _run_timed(llm_backend, task)
        elif cfg.execution_mode == "selected_only":
            # Only run the selected backend.
            selected = (selected_actions or {}).get(tid, "llm")
            if selected == "symbolic" and "symbolic" not in outcomes:
                if _is_symbolic_candidate(task):
                    outcomes["symbolic"] = _run_timed(symbolic_backend, task)
            elif selected == "llm" and "llm" not in outcomes:
                outcomes["llm"] = _run_timed(llm_backend, task)

        rewards = {
            b: backend_reward(o, utility_config) for b, o in outcomes.items()
        }
        gap = reward_gap(outcomes, utility_config)
        opt = optimal_action(outcomes, utility_config)

        fp = fingerprint_task(task)
        exp_id = make_experience_id(fp, policy_id, model_id, dataset_id, created_at)
        selected = (selected_actions or {}).get(tid, "llm")

        experiences.append(
            Experience(
                experience_id=exp_id,
                task_id=tid,
                task_fingerprint=fp,
                task_text=str(task.get("specification", "")),
                representation_ref=(representation_refs or {}).get(tid),
                policy_id=policy_id,
                model_id=model_id,
                dataset_id=dataset_id,
                selected_action=selected,
                outcomes=dict(outcomes),
                backend_rewards=rewards,
                optimal_action=opt,
                reward_gap=gap,
                created_at=created_at,
                metadata={"execution_mode": cfg.execution_mode},
            )
        )

    return experiences


__all__ = [
    "CounterfactualConfig",
    "ExecutionMode",
    "LLMBackend",
    "SymbolicBackend",
    "collect_counterfactual_experience",
    "default_symbolic_backend",
    "unsupported_symbolic_backend",
]
