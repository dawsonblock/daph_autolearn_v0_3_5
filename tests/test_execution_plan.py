import pytest
from daph_learning.execution.plan import ExecutionPlan, SymbolicAction
from daph_learning.execution.symbolic_executor import execute_plan, plan_from_structured_task


def test_plan_executes_its_action_not_gold_expected():
    task = {
        "task_id": "x",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 7, "b": 8, "op": "*"},
        "specification": "Compute 7 * 8.",
        "expected": 999999,  # deliberately wrong gold; executor must not consult it
    }
    plan = plan_from_structured_task(task, reason_code="test")
    result = execute_plan(plan)
    assert result.value == 56
    assert result.verified is True


def test_manual_plan_dispatch_is_authoritative():
    plan = ExecutionPlan(
        task_id="x",
        route="symbolic",
        planner_source="rule_router",
        actions=(SymbolicAction("integer_arithmetic", {"a": 9, "b": 4, "op": "-"}),),
    )
    assert execute_plan(plan).value == 5


def test_unknown_action_rejected():
    plan = ExecutionPlan(
        task_id="x", route="symbolic", planner_source="rule_router",
        actions=(SymbolicAction("not_real", {}),),
    )
    with pytest.raises(ValueError):
        execute_plan(plan)


def test_modular_text_fallback_compiles_to_typed_action():
    task = {
        "task_id": "m",
        "capability_ids": ["modular_multiplication"],
        "inputs": {},
        "specification": "Compute (12 * 13) mod 17. Return only the integer.",
    }
    plan = plan_from_structured_task(task, reason_code="fallback")
    assert plan.actions[0].name == "modular_multiply"
    assert execute_plan(plan).value == (12 * 13) % 17
