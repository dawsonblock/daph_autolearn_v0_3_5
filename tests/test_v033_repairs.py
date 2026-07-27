
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from daph_learning.execution.plan import ExecutionPlan, SymbolicAction
from daph_learning.execution.symbolic_executor import execute_plan
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import residual_addition_hook


def test_embedded_variable_substitution():
    plan = ExecutionPlan(
        task_id="embedded-1",
        route="symbolic",
        planner_source="rule_router",
        actions=(
            SymbolicAction(
                "integer_arithmetic",
                {"a": 10, "b": 20, "op": "+"},
                output_var="sum",
            ),
            SymbolicAction(
                "safe_integer_expression",
                {"expression": "($sum * 2) + 5"},
            ),
        ),
        reason_code="test_embedded",
    )
    result = execute_plan(plan)
    assert result.value == 65


def test_embedded_variable_dependency_discovery_allows_out_of_order_dag():
    plan = ExecutionPlan(
        task_id="embedded-dag",
        route="symbolic",
        planner_source="rule_router",
        actions=(
            SymbolicAction(
                "safe_integer_expression",
                {"expression": "$sum * 3"},
                output_var="product",
            ),
            SymbolicAction(
                "integer_arithmetic",
                {"a": 7, "b": 5, "op": "+"},
                output_var="sum",
            ),
        ),
        result_var="product",
        reason_code="test_embedded_dag",
    )
    result = execute_plan(plan)
    assert result.value == 36
    assert result.metadata["execution_order"] == [1, 0]


def test_evaluate_routes_tn_does_not_count_malformed(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    routes = tmp_path / "routes.jsonl"

    tasks.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "capability_ids": ["integer_arithmetic"],
                "inputs": {"a": 2, "b": 3, "op": "+"},
                "specification": "Compute 2 + 3.",
                "expected": 5,
                "gold_route": "llm",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    routes.write_text(
        json.dumps({"task_id": "t1", "route": "GARBAGE"}) + "\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_routes.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--routes",
            str(routes),
            "--label-field",
            "gold_route",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["tn"] == 0
    assert payload["malformed_routes"] == 1
    assert payload["route_accuracy"] == 0.0
    assert payload["decision_coverage"] == 0.0


class _Layer:
    def __init__(self):
        self._hook = None

    def register_forward_hook(self, hook):
        self._hook = hook

        class H:
            def remove(_self):
                self._hook = None

        return H()


class _Config:
    hidden_size = 3
    _name_or_path = "dummy"


class _Inner:
    def __init__(self):
        self.layers = [_Layer()]


class _Model:
    def __init__(self):
        self.model = _Inner()
        self.config = _Config()


def test_target_token_index_changes_only_requested_prompt_token():
    torch = pytest.importorskip("torch")
    model = _Model()
    layer = model.model.layers[0]
    hidden = torch.zeros((1, 4, 3), dtype=torch.float32)

    with residual_addition_hook(
        model,
        layer_index=0,
        vector=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        alpha=1.0,
        token_scope="last",
        target_token_index=1,
    ):
        output = layer._hook(None, None, hidden)

    assert torch.equal(output[0, 0], torch.zeros(3))
    assert torch.equal(output[0, 1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(output[0, 2], torch.zeros(3))
    assert torch.equal(output[0, 3], torch.zeros(3))


class _OffsetTokenizer:
    def __call__(self, text, **kwargs):
        # One token per character, plus a synthetic leading special token.
        ids = [999] + list(range(len(text)))
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
        return result


def test_anchor_alignment_ignores_appended_assistant_header():
    rendered = "TASK: 2 * 3\nACTION:<assistant-header>"
    idx = find_anchor_token_index(rendered, _OffsetTokenizer(), "ACTION:")
    # +1 because tokenizer injects one special token.
    assert idx == rendered.index("ACTION:") + len("ACTION:")
