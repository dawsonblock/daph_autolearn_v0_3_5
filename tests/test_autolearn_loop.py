"""Tests for the AutoLearn learning loop.

These tests cover:
1. Outcome classification (the core feedback mechanism).
2. The full learning loop with a fake model.
3. The learning curve structure.
4. Edge cases (no examples, early stopping).

See CLAIMS.md §18.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.autolearn import (
    AutoLearnConfig,
    AutoLearnResult,
    IterationMetrics,
    OutcomeLabel,
    classify_outcome,
    run_autolearn_loop,
)
from daph_learning.steering.types import SteeringSpec, SteeringVector


# --- Outcome classification tests ---

def test_classify_outcome_symbolic_correct():
    """Symbolic route with correct answer -> correct."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", "FINAL: 42") == "correct"


def test_classify_outcome_symbolic_wrong():
    """Symbolic route with wrong answer -> misrouted."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", "FINAL: 99") == "misrouted"


def test_classify_outcome_symbolic_failed():
    """Symbolic route with no output (execution failed) -> misrouted."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", None) == "misrouted"


def test_classify_outcome_symbolic_malformed():
    """Symbolic route with malformed output -> misrouted."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", "ERROR: something") == "misrouted"


def test_classify_outcome_symbolic_no_expected():
    """Symbolic route with no expected value -> correct (can't verify)."""
    task = {"task_id": "t1", "expected": None, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", "FINAL: 42") == "correct"


def test_classify_outcome_llm_non_symbolic_task():
    """LLM route on a non-symbolic task (no expected) -> correct."""
    task = {"task_id": "t1", "expected": None, "capability_ids": []}
    assert classify_outcome(task, "llm", "The sky is blue because...") == "correct"


def test_classify_outcome_llm_symbolic_task_correct():
    """LLM route on a symbolic task where LLM got the right answer -> correct."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "llm", "The answer is 42.") == "correct"


def test_classify_outcome_llm_symbolic_task_wrong():
    """LLM route on a symbolic task where LLM got the wrong answer -> unverifiable
    (we can't easily parse the LLM output, so we default to unverifiable
    when the expected value doesn't appear)."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "llm", "The answer is 99.") == "unverifiable"


def test_classify_outcome_llm_no_output():
    """LLM route with no output -> unverifiable."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "llm", None) == "unverifiable"


def test_classify_outcome_uses_task_expected_when_not_provided():
    """If expected is not passed explicitly, use task['expected']."""
    task = {"task_id": "t1", "expected": 5, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", "FINAL: 5") == "correct"
    assert classify_outcome(task, "symbolic", "FINAL: 6") == "misrouted"


# --- v0.3.6 Phase 1.1: route_raw for generate-mode multi-token outputs ---

def test_classify_outcome_route_raw_recovers_unparseable_symbolic():
    """When the parsed route is None (generation produced no clean ACTION),
    route_raw lets classify_outcome recover the intended route from the raw
    generated text. This is the multi-token tokenizer repair: a Qwen2.5
    generation of " SY", "MBOL", "IC" decodes to "SYMBOLIC" but the parser
    may have received a truncated/normalized form."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    # route=None, route_raw="SYMBOLIC" -> recovered to "symbolic", then
    # symbolic output FINAL: 42 -> correct.
    assert classify_outcome(
        task, None, "FINAL: 42", route_raw="SYMBOLIC"
    ) == "correct"


def test_classify_outcome_route_raw_recovers_unparseable_llm():
    """route_raw="LLM" with route=None recovers to the llm branch."""
    task = {"task_id": "t1", "expected": None, "capability_ids": []}
    assert classify_outcome(
        task, None, "The sky is blue.", route_raw="LLM"
    ) == "correct"


def test_classify_outcome_route_raw_llm_carries_expected_value():
    """When route=llm and output is None but route_raw contains the expected
    integer, the outcome is correct (the generate-mode path produced the
    answer in the route text itself)."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(
        task, "llm", None, route_raw="The answer is 42."
    ) == "correct"


def test_classify_outcome_route_raw_none_does_not_change_behavior():
    """Passing route_raw=None preserves the pre-v0.3.6 behavior exactly."""
    task = {"task_id": "t1", "expected": 42, "capability_ids": ["integer_arithmetic"]}
    assert classify_outcome(task, "symbolic", "FINAL: 42", route_raw=None) == "correct"
    assert classify_outcome(task, "llm", "no number here", route_raw=None) == "unverifiable"


# --- Learning loop tests ---

def _make_fake_model(hidden_size=4, num_layers=2, model_id="fake-model"):
    """Build a fake model with single-token SYMBOLIC/LLM labels."""
    import torch
    import torch.nn as nn

    class FakeTok:
        pad_token_id = 0
        pad_token = "<pad>"
        padding_side = "left"
        chat_template = None
        def encode(self, text, add_special_tokens=False):
            mapping = {" SYMBOLIC": [10], " LLM": [20], "SYMBOLIC": [10], "LLM": [20]}
            if text in mapping:
                return mapping[text]
            if text == "ACTION:":
                return [1, 2, 3]
            if text == "ACTION: SYMBOLIC":
                return [1, 2, 3, 10]
            if text == "ACTION: LLM":
                return [1, 2, 3, 20]
            return [1, 2, 3]
        def __call__(self, text, **kwargs):
            if isinstance(text, list):
                encoded = [self.encode(t, add_special_tokens=True) for t in text]
                max_len = max(len(e) for e in encoded)
                padded = [[0] * (max_len - len(e)) + e for e in encoded]
                return {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
                }
            ids = self.encode(text, add_special_tokens=True)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, hidden_size)
            self.config = type("C", (), {
                "hidden_size": hidden_size,
                "_name_or_path": model_id,
            })()
            inner = type("I", (nn.Module,), {})()
            class FakeLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(hidden_size, hidden_size)
                def forward(self, x):
                    return (self.linear(x),)
            inner.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])
            self.model = inner
            self.lm_head = nn.Linear(hidden_size, 100)
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    return FakeModel(), FakeTok()


def _make_tasks(n=8):
    """Create n arithmetic tasks with expected values."""
    tasks = []
    for i in range(n):
        a = i + 1
        b = i + 2
        tasks.append({
            "task_id": f"t{i}",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": "+"},
            "specification": f"Compute {a} + {b}. Return only the integer.",
            "expected": a + b,
            "route_label": "symbolic",
            "capability_oracle": "symbolic",
            "accuracy_oracle": "llm",
            "utility_oracle": "symbolic",
        })
    return tasks


def test_autolearn_loop_returns_result():
    """The loop must return an AutoLearnResult with the expected structure."""
    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    config = AutoLearnConfig(n_iterations=3, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    assert isinstance(result, AutoLearnResult)
    assert len(result.iterations) >= 1
    assert all(isinstance(m, IterationMetrics) for m in result.iterations)


def test_autolearn_loop_learning_curve_has_metrics():
    """Each iteration must have the full set of metrics."""
    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    for m in result.iterations:
        assert m.iteration >= 0
        assert m.n_train == 8
        assert m.n_correct + m.n_misrouted + m.n_unverifiable == m.n_train
        assert 0.0 <= m.train_accuracy <= 1.0
        assert m.vector_norm >= 0.0
        assert isinstance(m.updated, bool)


def test_autolearn_loop_first_iteration_has_no_vector():
    """The first iteration (no initial vector) should route heuristically
    and use the outcomes to extract the first vector."""
    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    config = AutoLearnConfig(n_iterations=1, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    assert len(result.iterations) == 1
    first = result.iterations[0]
    # All tasks are symbolic-capable, so heuristic routes all to symbolic
    # and they all execute correctly -> all correct
    assert first.n_correct > 0


def test_autolearn_loop_updates_vector():
    """When there are enough positive and negative examples, the vector
    should be updated (updated=True)."""
    model, tok = _make_fake_model()
    # Mix of symbolic and non-symbolic tasks to get both classes
    train = []
    for i in range(6):
        train.append({
            "task_id": f"sym_{i}",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": i, "b": i + 1, "op": "+"},
            "specification": f"Compute {i} + {i+1}. Return only the integer.",
            "expected": 2 * i + 1,
            "route_label": "symbolic",
            "capability_oracle": "symbolic",
            "accuracy_oracle": "symbolic",
            "utility_oracle": "symbolic",
        })
    for i in range(6):
        train.append({
            "task_id": f"llm_{i}",
            "capability_ids": [],
            "inputs": {},
            "specification": "Explain something non-mathematical.",
            "expected": None,
            "route_label": "llm",
            "capability_oracle": "llm",
            "accuracy_oracle": "llm",
            "utility_oracle": "llm",
        })
    val = train[:4]
    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    # The first iteration should have enough examples to update
    assert result.iterations[0].updated or result.iterations[0].n_positive_examples >= 2


def test_autolearn_loop_early_stops_when_no_update():
    """If the vector can't be updated (not enough examples), the loop
    should stop early."""
    model, tok = _make_fake_model()
    # Only symbolic tasks — all will be "correct" when routed to symbolic,
    # so there will be no negative examples for contrastive extraction.
    train = _make_tasks(8)
    val = _make_tasks(4)
    config = AutoLearnConfig(n_iterations=5, layer=0, min_examples_per_update=10)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    # With min_examples=10 and only 8 tasks, no update can happen.
    # The loop should stop after the second iteration (first with no update).
    assert len(result.iterations) <= 2


def test_autolearn_loop_with_initial_vector():
    """The loop should accept an initial vector and iterate from there."""
    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    spec = SteeringSpec(
        vector_id="initial", family="tool_policy", behavior="b",
        layer=0, alpha=1.0, anchor="ACTION",
        model_id="fake-model", hidden_size=4,
    )
    initial = SteeringVector(spec=spec, values=np.ones(4, dtype=np.float32))
    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(
        train, val, model, tok, config=config,
        initial_vector=initial, prompt_format="raw",
    )
    assert len(result.iterations) >= 1


def test_autolearn_loop_best_iteration_tracked():
    """The result should track which iteration had the best val F1."""
    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    config = AutoLearnConfig(n_iterations=3, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    assert result.best_iteration >= 0
    assert result.best_val_f1 >= 0.0
    # The best iteration should be within the range of iterations run
    assert result.best_iteration < len(result.iterations)


def test_autolearn_loop_deterministic_with_seed():
    """Same seed should produce the same learning curve."""
    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2, seed=42)
    r1 = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    r2 = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    assert len(r1.iterations) == len(r2.iterations)
    for m1, m2 in zip(r1.iterations, r2.iterations):
        assert m1.n_correct == m2.n_correct
        assert m1.train_accuracy == m2.train_accuracy


def test_autolearn_loop_empty_train():
    """An empty training set should not crash."""
    model, tok = _make_fake_model()
    config = AutoLearnConfig(n_iterations=1, layer=0)
    result = run_autolearn_loop([], [], model, tok, config=config, prompt_format="raw")
    assert len(result.iterations) >= 1
    assert result.iterations[0].n_train == 0
    assert result.iterations[0].train_accuracy == 0.0


def test_autolearn_config_defaults():
    """Config should have sensible defaults."""
    config = AutoLearnConfig()
    assert config.n_iterations == 5
    assert config.layer == 24
    assert config.alpha == 1.0
    assert config.anchor == "ACTION"
    assert config.min_examples_per_update == 4
    assert config.normalization == "l2"
    assert config.seed == 42


# --- Generate-mode routing tests ---

def test_route_with_steering_generate_fake_model():
    """The generate-mode steered routing should work with a fake model."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import numpy as np
    from daph_learning.autolearn.loop import _route_with_steering_generate

    model, tok = _make_fake_model()
    spec = SteeringSpec(
        vector_id="test", family="tool_policy", behavior="b",
        layer=0, alpha=1.0, anchor="ACTION",
        model_id="fake-model", hidden_size=4,
    )
    vec = SteeringVector(spec=spec, values=np.ones(4, dtype=np.float32))
    tasks = _make_tasks(4)
    routes = _route_with_steering_generate(tasks, model, tok, vec, 1.0, prompt_format="raw")
    assert len(routes) == 4
    # Routes should be (route, raw_text) tuples
    for r in routes:
        route, raw = r
        assert route is None or route in ("symbolic", "llm")
        assert raw is None or isinstance(raw, str)


def test_route_without_steering_generate_fake_model():
    """The generate-mode unsteered routing should work with a fake model."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from daph_learning.autolearn.loop import _route_without_steering_generate

    model, tok = _make_fake_model()
    tasks = _make_tasks(4)
    routes = _route_without_steering_generate(tasks, model, tok, prompt_format="raw")
    assert len(routes) == 4


def test_evaluate_routes_simple():
    """The simple evaluation function should compute correct metrics."""
    from daph_learning.autolearn.loop import _evaluate_routes_simple

    tasks = [
        {"task_id": "t1", "route_label": "symbolic"},
        {"task_id": "t2", "route_label": "symbolic"},
        {"task_id": "t3", "route_label": "llm"},
        {"task_id": "t4", "route_label": "llm"},
    ]
    routes = ["symbolic", "llm", "llm", "llm"]  # 1 tp, 0 fp, 1 fn, 2 tn
    metrics = _evaluate_routes_simple(tasks, routes, label_field="route_label")
    assert metrics["accuracy"] == 0.75  # 3/4 correct
    assert metrics["precision"] == 1.0  # 1 tp / (1 tp + 0 fp)
    assert metrics["recall"] == 0.5  # 1 tp / (1 tp + 1 fn)
    expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
    assert abs(metrics["f1"] - expected_f1) < 1e-6


def test_evaluate_routes_simple_none_routes():
    """None routes should be treated as 'llm'."""
    from daph_learning.autolearn.loop import _evaluate_routes_simple

    tasks = [{"task_id": "t1", "route_label": "llm"}]
    routes = [None]
    metrics = _evaluate_routes_simple(tasks, routes, label_field="route_label")
    assert metrics["accuracy"] == 1.0  # None -> llm, expected llm
    assert metrics["decision_coverage"] == 0.0  # no decisions made


def test_autolearn_loop_generate_mode():
    """The loop should work with routing_mode='generate' on a fake model."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import numpy as np

    model, tok = _make_fake_model()
    train = _make_tasks(8)
    val = _make_tasks(4)
    spec = SteeringSpec(
        vector_id="init", family="tool_policy", behavior="b",
        layer=0, alpha=1.0, anchor="ACTION",
        model_id="fake-model", hidden_size=4,
    )
    initial = SteeringVector(spec=spec, values=np.ones(4, dtype=np.float32))
    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(
        train, val, model, tok, config=config,
        initial_vector=initial, prompt_format="raw",
        routing_mode="generate",
    )
    assert len(result.iterations) >= 1
    # With generate mode, the loop should actually route tasks
    for m in result.iterations:
        assert m.n_train == 8
