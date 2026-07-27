from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


Backend = Literal["llm", "symbolic"]
ExecutionMode = Literal[
    "baseline", "memory", "sham", "auto", "symbolic", "hybrid", "steered_auto"
]


@dataclass(frozen=True)
class CapabilityAssessment:
    capabilities: frozenset[str]
    exact_computation_required: bool
    operation: str | None
    complexity: float
    confidence: float


@dataclass(frozen=True)
class RoutingConfig:
    execution_mode: ExecutionMode = "baseline"
    symbolic_multiply: bool = True
    symbolic_modular: bool = True
    symbolic_nested: bool = True


@dataclass(frozen=True)
class RouteDecision:
    backend: Backend
    reason_code: str
    assessment: CapabilityAssessment
    use_memory: bool = False
    steering_policy: str | None = None


_SYMBOLIC_CAPS = {"integer_arithmetic", "modular_multiplication"}


def assess_task(task: Mapping[str, Any]) -> CapabilityAssessment:
    caps = frozenset(map(str, task.get("capability_ids", [])))
    inputs = task.get("inputs") or {}
    op = inputs.get("op")
    spec = str(task.get("specification", ""))
    nested = "(" in spec or ")" in spec

    exact = bool(caps & _SYMBOLIC_CAPS)
    complexity = 0.0
    if op in {"*", "%", "//"}:
        complexity += 1.0
    if "modular_multiplication" in caps:
        complexity += 1.0
    if nested:
        complexity += 0.5

    # Frozen benchmark capability IDs are the strongest available detector signal.
    confidence = 1.0 if exact else 0.5
    return CapabilityAssessment(caps, exact, op, complexity, confidence)


def decision_from_action(
    task: Mapping[str, Any],
    action: Backend,
    *,
    has_procedures: bool,
    reason_code: str,
    steering_policy: str | None = None,
) -> RouteDecision:
    return RouteDecision(
        backend=action,
        reason_code=reason_code,
        assessment=assess_task(task),
        use_memory=has_procedures if action == "llm" else False,
        steering_policy=steering_policy,
    )


def route_task(
    task: Mapping[str, Any],
    has_procedures: bool,
    config: RoutingConfig,
) -> RouteDecision:
    a = assess_task(task)
    mode = config.execution_mode

    if mode == "baseline":
        return RouteDecision("llm", "baseline_llm", a, use_memory=False)

    if mode in {"memory", "sham"}:
        return RouteDecision("llm", "llm_with_memory", a, use_memory=has_procedures)

    if mode == "symbolic":
        if a.capabilities & _SYMBOLIC_CAPS:
            return RouteDecision("symbolic", "forced_symbolic_supported_capability", a)
        return RouteDecision("llm", "forced_symbolic_unsupported_fallback", a)

    if mode == "hybrid":
        if a.capabilities & _SYMBOLIC_CAPS:
            return RouteDecision("symbolic", "hybrid_exact_supported_capability", a)
        return RouteDecision("llm", "hybrid_non_symbolic_llm", a, use_memory=has_procedures)

    if mode == "steered_auto":
        # A model-mediated routing pass resolves this placeholder in the CLI.
        return RouteDecision(
            "llm",
            "steered_router_pending",
            a,
            use_memory=has_procedures,
            steering_policy="invoke_symbolic_tool",
        )

    if mode == "auto":
        if "modular_multiplication" in a.capabilities and config.symbolic_modular:
            return RouteDecision("symbolic", "auto_modular_exact", a)
        if (
            "integer_arithmetic" in a.capabilities
            and a.operation == "*"
            and config.symbolic_multiply
        ):
            return RouteDecision("symbolic", "auto_multiplication_exact", a)
        if (
            "integer_arithmetic" in a.capabilities
            and a.complexity >= 0.5
            and config.symbolic_nested
        ):
            return RouteDecision("symbolic", "auto_nested_exact", a)
        return RouteDecision(
            "llm",
            "auto_easy_or_non_symbolic_llm",
            a,
            use_memory=has_procedures,
        )

    raise ValueError(f"unknown execution mode: {mode}")
