from __future__ import annotations

from typing import Any, Mapping
import re

from .plan import ExecutionPlan, ExecutionResult, SymbolicAction
from daph_learning.tools.symbolic_math import (
    execute_integer_arithmetic,
    execute_modular_multiplication,
    expression_from_specification,
    modular_inputs_from_specification,
    safe_eval_int_expr,
)


def _has_modulus(inputs: Mapping[str, Any]) -> bool:
    return any(key in inputs for key in ("modulus", "mod", "m", "c"))


def plan_from_structured_task(
    task: Mapping[str, Any],
    *,
    reason_code: str,
) -> ExecutionPlan:
    """Compile a benchmark task into a typed symbolic execution plan.

    Structured fields are authoritative. Natural-language expression parsing is
    only used when a supported arithmetic capability lacks the structured fields
    required by its typed action.
    """
    caps = set(task.get("capability_ids", []))
    inputs = dict(task.get("inputs") or {})
    task_id = str(task.get("task_id", ""))

    if "modular_multiplication" in caps and {"a", "b"} <= inputs.keys() and _has_modulus(inputs):
        action = SymbolicAction("modular_multiply", inputs)
        source = "structured_input"
    elif "integer_arithmetic" in caps and {"a", "b", "op"} <= inputs.keys():
        action = SymbolicAction("integer_arithmetic", inputs)
        source = "structured_input"
    elif "modular_multiplication" in caps:
        parsed = modular_inputs_from_specification(str(task.get("specification", "")))
        action = SymbolicAction("modular_multiply", parsed)
        source = "rule_router"
    elif "integer_arithmetic" in caps:
        expression = expression_from_specification(str(task.get("specification", "")))
        action = SymbolicAction("safe_integer_expression", {"expression": expression})
        source = "rule_router"
    else:
        raise ValueError(f"unsupported symbolic capabilities: {sorted(caps)}")

    return ExecutionPlan(
        task_id=task_id,
        route="symbolic",
        planner_source=source,
        actions=(action,),
        reason_code=reason_code,
    )


_VAR_REF_RE = re.compile(r"\$([a-zA-Z_]\w*)")


def _variable_references(value: Any) -> set[str]:
    """Extract all ``$var`` references from nested action arguments."""
    if isinstance(value, str):
        return set(_VAR_REF_RE.findall(value))
    if isinstance(value, Mapping):
        return {
            ref
            for nested in value.values()
            for ref in _variable_references(nested)
        }
    if isinstance(value, (list, tuple)):
        return {
            ref
            for nested in value
            for ref in _variable_references(nested)
        }
    return set()


def _resolve_value(value: Any, env: Mapping[str, int]) -> Any:
    """Resolve exact and embedded ``$var`` references recursively.

    An exact value like ``"$sum"`` remains an integer after substitution.
    Embedded references such as ``"($sum * 2) + 5"`` are rendered into a
    string expression and are subsequently validated by the bounded AST
    integer evaluator.
    """
    if isinstance(value, str):
        exact = _VAR_REF_RE.fullmatch(value)
        if exact:
            name = exact.group(1)
            if name not in env:
                raise KeyError(name)
            return env[name]

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in env:
                raise KeyError(name)
            return str(env[name])

        return _VAR_REF_RE.sub(substitute, value)

    if isinstance(value, Mapping):
        return {
            key: _resolve_value(nested, env)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_resolve_value(nested, env) for nested in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(nested, env) for nested in value)
    return value


def _execute_action(action: SymbolicAction, env: Mapping[str, int]) -> int:
    arguments = _resolve_value(action.arguments, env)
    if action.name == "integer_arithmetic":
        return execute_integer_arithmetic(arguments)
    if action.name == "modular_multiply":
        return execute_modular_multiplication(arguments)
    if action.name in {"safe_integer_expression", "safe_expression"}:
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise TypeError("safe_integer_expression requires a string expression")
        return safe_eval_int_expr(expression)
    raise ValueError(f"unknown symbolic action: {action.name!r}")


def execute_plan(plan: ExecutionPlan) -> ExecutionResult:
    """Execute an acyclic symbolic plan without consulting benchmark gold data.

    Action arguments may reference any named action output with ``$name``. The
    executor schedules actions when their dependencies are available, so tuple
    order is not required to be topological. Cycles, unknown references, and
    duplicate output variables fail closed.
    """
    if plan.route != "symbolic":
        raise ValueError(f"cannot symbolically execute plan route={plan.route!r}")
    if not plan.actions:
        raise ValueError("symbolic plan contains no actions")

    output_names = [a.output_var for a in plan.actions if a.output_var is not None]
    if len(output_names) != len(set(output_names)):
        raise ValueError("symbolic plan contains duplicate output variables")

    produced = set(output_names)
    for action in plan.actions:
        unknown = _variable_references(action.arguments) - produced
        if unknown:
            raise ValueError(f"symbolic plan references unknown variables: {sorted(unknown)}")

    pending = list(enumerate(plan.actions))
    env: dict[str, int] = {}
    execution_order: list[int] = []
    last_value: int | None = None

    while pending:
        progressed = False
        next_pending: list[tuple[int, SymbolicAction]] = []
        for original_index, action in pending:
            deps = _variable_references(action.arguments)
            if not deps <= env.keys():
                next_pending.append((original_index, action))
                continue
            value = _execute_action(action, env)
            last_value = value
            if action.output_var is not None:
                env[action.output_var] = value
            execution_order.append(original_index)
            progressed = True

        if not progressed:
            blocked = {
                idx: sorted(_variable_references(action.arguments) - env.keys())
                for idx, action in next_pending
            }
            raise ValueError(f"cyclic or unresolved symbolic plan dependencies: {blocked}")
        pending = next_pending

    if plan.result_var is not None:
        if plan.result_var not in env:
            raise ValueError(f"result_var {plan.result_var!r} was not produced")
        result_value = env[plan.result_var]
    else:
        assert last_value is not None
        result_value = last_value

    return ExecutionResult(
        value=result_value,
        verified=True,
        actions_executed=len(plan.actions),
        metadata={
            "planner_source": plan.planner_source,
            "reason_code": plan.reason_code,
            "variables": dict(env),
            "execution_order": execution_order,
        },
    )
