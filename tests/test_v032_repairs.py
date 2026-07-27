from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from daph_learning.execution.plan import ExecutionPlan, SymbolicAction
from daph_learning.execution.symbolic_executor import execute_plan
from daph_learning.memory.procedural import Procedure, ProceduralMemory
from daph_learning.tools.symbolic_math import expression_from_specification


def test_procedural_memory_is_available_for_fallback_prompting():
    memory = ProceduralMemory([
        Procedure(
            procedure_id="rule_0",
            capability_id="integer_arithmetic",
            trigger_terms=["math"],
            action="Use exact modular reduction.",
        )
    ])
    retrieved = memory.retrieve(["integer_arithmetic"], "Compute 5 % 0.")
    assert len(retrieved) == 1


def test_parser_stops_before_trailing_alpha_and_numeric_instruction():
    assert expression_from_specification(
        "Compute (125 + 43) * 11 then return 1 integer."
    ) == "(125 + 43) * 11"
    assert expression_from_specification(
        "Compute 3591 - -5519. Return only the integer."
    ) == "3591 - -5519"


def test_multistep_dag_can_execute_out_of_tuple_order():
    plan = ExecutionPlan(
        task_id="dag-1",
        route="symbolic",
        planner_source="rule_router",
        actions=(
            # Deliberately listed before its dependency.
            SymbolicAction(
                "integer_arithmetic",
                {"a": "$sum", "b": 11, "op": "*"},
                output_var="product",
            ),
            SymbolicAction(
                "integer_arithmetic",
                {"a": 125, "b": 43, "op": "+"},
                output_var="sum",
            ),
        ),
        reason_code="test_dag",
        result_var="product",
    )
    result = execute_plan(plan)
    assert result.value == 1848
    assert result.metadata["variables"] == {"sum": 168, "product": 1848}
    assert result.metadata["execution_order"] == [1, 0]


def test_multistep_dag_rejects_cycle():
    plan = ExecutionPlan(
        task_id="cycle",
        route="symbolic",
        planner_source="rule_router",
        actions=(
            SymbolicAction("integer_arithmetic", {"a": "$b", "b": 1, "op": "+"}, output_var="a"),
            SymbolicAction("integer_arithmetic", {"a": "$a", "b": 1, "op": "+"}, output_var="b"),
        ),
    )
    with pytest.raises(ValueError, match="cyclic or unresolved"):
        execute_plan(plan)


def test_evaluate_routes_penalizes_malformed(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    routes = tmp_path / "routes.jsonl"
    tasks.write_text(json.dumps({
        "task_id": "t1",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 2, "b": 3, "op": "*"},
        "specification": "Compute 2 * 3.",
        "expected": 6,
    }) + "\n", encoding="utf-8")
    routes.write_text(json.dumps({"task_id": "t1", "route": "MALFORMED"}) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_routes.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--tasks", str(tasks), "--routes", str(routes)],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["malformed_routes"] == 1
    assert payload["fn"] == 1
    assert payload["route_accuracy"] == 0.0
    assert payload["malformed_rate"] == 1.0


def test_evaluate_routes_malformed_llm_oracle_still_lowers_accuracy(tmp_path: Path):
    tasks = tmp_path / "tasks_negative.jsonl"
    routes = tmp_path / "routes_negative.jsonl"
    tasks.write_text(json.dumps({
        "task_id": "t2",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 2, "b": 3, "op": "+"},
        "specification": "Compute 2 + 3.",
        "expected": 5,
        "gold_route": "llm",
    }) + "\n", encoding="utf-8")
    routes.write_text(json.dumps({"task_id": "t2", "route": "???"}) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_routes.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--tasks", str(tasks), "--routes", str(routes), "--label-field", "gold_route"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["malformed_routes"] == 1
    assert payload["route_accuracy"] == 0.0
    assert payload["decision_coverage"] == 0.0
