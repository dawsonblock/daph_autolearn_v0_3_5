"""V037-001: contract tests for the ``route_fn`` dispatch path.

These tests verify the fix for the v0.3.6 dead path where ``route_fn`` was
accepted by ``run_autolearn_loop()`` but never executed. Acceptance criteria
from ROADMAP §V037-001:

- custom ``route_fn`` called exactly once per task
- custom result propagated
- invalid custom result rejected with a typed error
- ``route_fn`` exceptions propagate (not swallowed)
- built-in fallback behaviour unchanged
"""

from __future__ import annotations

import pytest

from daph_learning.autolearn import AutoLearnConfig, run_autolearn_loop
from daph_learning.routing.policy import (
    InvalidRouteDecisionError,
    RouteDecision,
    coerce_route_decision,
    route_tasks,
)


# --- fixtures ---

def _make_fake_model(hidden_size=4, num_layers=2, model_id="fake-model"):
    """Reuse the fake model/tokenizer shape from test_autolearn_loop."""
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
            return {"input_ids": torch.tensor(ids), "attention_mask": [1] * len(ids)}

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


def _make_tasks(n=6):
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


# --- coerce_route_decision unit tests ---

def test_coerce_string_route():
    task = {"task_id": "x1"}
    d = coerce_route_decision(task, "symbolic")
    assert d.task_id == "x1"
    assert d.route == "symbolic"
    assert d.source == "custom"


def test_coerce_dict_route_with_scores():
    task = {"task_id": "x2"}
    d = coerce_route_decision(
        task,
        {"route": "llm", "confidence": 0.7, "raw_scores": {"symbolic": -1.0, "llm": 2.0}},
    )
    assert d.route == "llm"
    assert d.confidence == pytest.approx(0.7)
    assert d.raw_scores == {"symbolic": -1.0, "llm": 2.0}


def test_coerce_routedecision_passthrough():
    task = {"task_id": "x3"}
    src = RouteDecision(task_id="x3", route="abstain", source="ignored")
    d = coerce_route_decision(task, src)
    assert d.route == "abstain"
    # source is re-tagged to "custom" by the coercion contract
    assert d.source == "custom"


def test_coerce_rejects_bad_string():
    with pytest.raises(InvalidRouteDecisionError):
        coerce_route_decision({"task_id": "x"}, "nonsense")


def test_coerce_rejects_missing_route_key():
    with pytest.raises(InvalidRouteDecisionError):
        coerce_route_decision({"task_id": "x"}, {"confidence": 0.5})


def test_coerce_rejects_unsupported_type():
    with pytest.raises(InvalidRouteDecisionError):
        coerce_route_decision({"task_id": "x"}, 42)


def test_coerce_rejects_task_id_mismatch():
    src = RouteDecision(task_id="other", route="llm")
    with pytest.raises(InvalidRouteDecisionError):
        coerce_route_decision({"task_id": "x"}, src)


# --- route_tasks dispatch tests ---

def test_route_fn_called_once_per_task():
    tasks = _make_tasks(6)
    call_count = {"n": 0}

    def my_router(task, model, tokenizer, steering):
        call_count["n"] += 1
        return "symbolic"

    decisions = route_tasks(tasks, model=None, tokenizer=None, route_fn=my_router)
    assert call_count["n"] == 6
    assert len(decisions) == 6
    assert all(d.route == "symbolic" for d in decisions)
    assert all(d.source == "custom" for d in decisions)
    assert [d.task_id for d in decisions] == [t["task_id"] for t in tasks]


def test_route_fn_result_propagated():
    tasks = _make_tasks(3)

    def my_router(task, model, tokenizer, steering):
        # Alternate routes and carry scores.
        idx = int(task["task_id"][1:])
        if idx % 2 == 0:
            return {"route": "symbolic", "confidence": 0.9,
                    "raw_scores": {"symbolic": 1.5, "llm": -0.5}}
        return "llm"

    decisions = route_tasks(tasks, model=None, tokenizer=None, route_fn=my_router)
    assert decisions[0].route == "symbolic"
    assert decisions[0].confidence == pytest.approx(0.9)
    assert decisions[0].raw_scores == {"symbolic": 1.5, "llm": -0.5}
    assert decisions[1].route == "llm"
    assert decisions[1].confidence is None


def test_route_fn_invalid_result_rejected():
    tasks = _make_tasks(2)

    def bad_router(task, model, tokenizer, steering):
        return {"backend": "symbolic"}  # missing 'route' key

    with pytest.raises(InvalidRouteDecisionError):
        route_tasks(tasks, model=None, tokenizer=None, route_fn=bad_router)


def test_route_fn_exception_propagates():
    tasks = _make_tasks(2)

    def exploding_router(task, model, tokenizer, steering):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        route_tasks(tasks, model=None, tokenizer=None, route_fn=exploding_router)


def test_builtin_fallback_baseline_when_no_route_fn_no_steering():
    tasks = _make_tasks(3)
    # No route_fn, no steering -> baseline capability heuristic.
    decisions = route_tasks(tasks, model=None, tokenizer=None)
    assert all(d.source == "baseline" for d in decisions)
    # _make_tasks has capability_ids -> symbolic
    assert all(d.route == "symbolic" for d in decisions)


def test_builtin_fallback_baseline_for_non_symbolic_task():
    tasks = [{"task_id": "n1", "capability_ids": [], "specification": "hello"}]
    decisions = route_tasks(tasks, model=None, tokenizer=None)
    assert decisions[0].route == "llm"
    assert decisions[0].source == "baseline"


# --- run_autolearn_loop integration: route_fn is actually invoked now ---

def test_autolearn_loop_calls_custom_route_fn():
    """The headline V037-001 acceptance: route_fn must actually be called by
    run_autolearn_loop. Under v0.3.6 it was accepted but never invoked."""
    model, tok = _make_fake_model()
    train = _make_tasks(6)
    val = _make_tasks(3)
    call_count = {"n": 0}

    def my_router(task, model, tokenizer, steering):
        call_count["n"] += 1
        return "symbolic"

    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(
        train, val, model, tok,
        config=config, prompt_format="raw", route_fn=my_router,
    )
    # The loop runs n_iterations; each iteration routes every train task.
    # With route_fn actually wired, the call count must be > 0 (and a multiple
    # of len(train) per iteration that reaches the routing stage).
    assert call_count["n"] > 0, "route_fn was never called by run_autolearn_loop"
    assert call_count["n"] % len(train) == 0
    assert isinstance(result.best_iteration, int)


def test_autolearn_loop_route_fn_exception_propagates():
    """A route_fn that raises must terminate the loop, not be swallowed."""
    model, tok = _make_fake_model()
    train = _make_tasks(4)
    val = _make_tasks(2)

    def exploding(task, model, tokenizer, steering):
        raise RuntimeError("router-exploded")

    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    with pytest.raises(RuntimeError, match="router-exploded"):
        run_autolearn_loop(
            train, val, model, tok,
            config=config, prompt_format="raw", route_fn=exploding,
        )


def test_autolearn_loop_route_fn_invalid_result_rejected():
    """A route_fn returning an invalid result must raise
    InvalidRouteDecisionError, not silently fall back."""
    model, tok = _make_fake_model()
    train = _make_tasks(4)
    val = _make_tasks(2)

    def bad(task, model, tokenizer, steering):
        return {"backend": "symbolic"}  # no 'route' key

    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    with pytest.raises(InvalidRouteDecisionError):
        run_autolearn_loop(
            train, val, model, tok,
            config=config, prompt_format="raw", route_fn=bad,
        )


def test_autolearn_loop_without_route_fn_unchanged():
    """Sanity: the built-in steered/baseline path still produces a result with
    the same shape when route_fn is not supplied."""
    model, tok = _make_fake_model()
    train = _make_tasks(6)
    val = _make_tasks(3)
    config = AutoLearnConfig(n_iterations=2, layer=0, min_examples_per_update=2)
    result = run_autolearn_loop(train, val, model, tok, config=config, prompt_format="raw")
    assert hasattr(result, "iterations")
    assert len(result.iterations) >= 1
