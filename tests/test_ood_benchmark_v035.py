from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from scripts.generate_ood_benchmark import (
    ALL_CATEGORIES,
    ADVERSARIAL_CATEGORIES,
    GENERATOR_VERSION,
    MALFORMED_CATEGORIES,
    NON_SYMBOLIC_CATEGORIES,
    SYMBOLIC_CATEGORIES,
    generate_benchmark,
    generate_task,
)


def _load_generator_module():
    """Ensure the module is loaded via the package (scripts.generate_ood_benchmark)."""
    return __import__("scripts.generate_ood_benchmark", fromlist=["main"])


def test_generator_version_bumped_to_v035() -> None:
    assert GENERATOR_VERSION == "v0.3.5"


def test_all_categories_present() -> None:
    """The expanded benchmark must include non-symbolic controls, adversarial
    controls, nested expressions, and malformed tasks — not just the
    original 5 arithmetic categories. See CLAIMS.md §2."""
    assert "nested_arithmetic" in ALL_CATEGORIES
    assert "signed_nested_arithmetic" in ALL_CATEGORIES
    assert "text_question" in ALL_CATEGORIES
    assert "coding_question" in ALL_CATEGORIES
    assert "knowledge_question" in ALL_CATEGORIES
    assert "irrelevant_numeric" in ALL_CATEGORIES
    assert "ambiguous_tool" in ALL_CATEGORIES
    assert "unsupported_arithmetic" in ALL_CATEGORIES
    # Original categories preserved
    for old in ("easy_add", "easy_sub", "multiply", "modulo", "modular_multiplication"):
        assert old in ALL_CATEGORIES


def test_category_groups_are_disjoint() -> None:
    all_set = set(ALL_CATEGORIES)
    assert all_set == set(SYMBOLIC_CATEGORIES) | set(NON_SYMBOLIC_CATEGORIES) | set(ADVERSARIAL_CATEGORIES) | set(MALFORMED_CATEGORIES)
    # No overlap between groups
    assert not (set(SYMBOLIC_CATEGORIES) & set(NON_SYMBOLIC_CATEGORIES))
    assert not (set(SYMBOLIC_CATEGORIES) & set(ADVERSARIAL_CATEGORIES))
    assert not (set(SYMBOLIC_CATEGORIES) & set(MALFORMED_CATEGORIES))
    assert not (set(NON_SYMBOLIC_CATEGORIES) & set(ADVERSARIAL_CATEGORIES))
    assert not (set(NON_SYMBOLIC_CATEGORIES) & set(MALFORMED_CATEGORIES))
    assert not (set(ADVERSARIAL_CATEGORIES) & set(MALFORMED_CATEGORIES))


def test_typed_oracle_fields_present_on_every_task() -> None:
    """Every task must carry capability_oracle, accuracy_oracle, and
    utility_oracle — not just route_label. See CLAIMS.md §3."""
    import random
    rng = random.Random(42)
    for category in ALL_CATEGORIES:
        task = generate_task("test-1", rng, category, seed=42)
        assert "capability_oracle" in task, f"{category} missing capability_oracle"
        assert "accuracy_oracle" in task, f"{category} missing accuracy_oracle"
        assert "utility_oracle" in task, f"{category} missing utility_oracle"
        assert task["capability_oracle"] in {"symbolic", "llm"}
        assert task["accuracy_oracle"] in {"symbolic", "llm"}
        assert task["utility_oracle"] in {"symbolic", "llm"}


def test_route_label_equals_utility_oracle_for_backward_compat() -> None:
    """route_label is preserved and equals utility_oracle so older
    consumers that read route_label still work."""
    import random
    rng = random.Random(42)
    for category in ALL_CATEGORIES:
        task = generate_task("test-1", rng, category, seed=42)
        assert task["route_label"] == task["utility_oracle"], (
            f"{category}: route_label {task['route_label']!r} != utility_oracle {task['utility_oracle']!r}"
        )


def test_non_symbolic_tasks_route_to_llm() -> None:
    """Non-symbolic controls must have all three oracles = llm. A
    pathological steering vector that pushes these to SYMBOLIC would
    produce false tool calls — that's the whole point of having them."""
    import random
    rng = random.Random(42)
    for category in NON_SYMBOLIC_CATEGORIES:
        task = generate_task("test-1", rng, category, seed=42)
        assert task["capability_oracle"] == "llm"
        assert task["accuracy_oracle"] == "llm"
        assert task["utility_oracle"] == "llm"
        assert task["expected"] is None  # no integer answer


def test_adversarial_tasks_route_to_llm() -> None:
    import random
    rng = random.Random(42)
    for category in ADVERSARIAL_CATEGORIES:
        task = generate_task("test-1", rng, category, seed=42)
        assert task["utility_oracle"] == "llm"
        assert task["expected"] is None


def test_malformed_tasks_route_to_llm() -> None:
    """Unsupported arithmetic (e.g. true division) must route to LLM,
    not crash the symbolic engine."""
    import random
    rng = random.Random(42)
    for category in MALFORMED_CATEGORIES:
        task = generate_task("test-1", rng, category, seed=42)
        assert task["utility_oracle"] == "llm"


def test_nested_arithmetic_uses_textual_fallback() -> None:
    """Nested expression tasks have empty inputs (no structured a/b/op),
    so they exercise the symbolic engine's textual expression fallback."""
    import random
    rng = random.Random(42)
    task = generate_task("test-1", rng, "nested_arithmetic", seed=42)
    assert task["inputs"] == {}
    assert task["expected"] is not None
    # The expression must be evaluable by the bounded integer evaluator
    # (supports +, -, *, parentheses, unary +/-).
    expr = task["specification"].replace("Compute ", "").replace(". Return only the integer.", "")
    # Quick sanity: the expected value should match the expression.
    # We use Python eval here ONLY in the test, not in the engine.
    assert eval(expr) == task["expected"]  # noqa: S307 — test only, controlled input


def test_signed_nested_arithmetic_produces_correct_value() -> None:
    import random
    rng = random.Random(42)
    task = generate_task("test-1", rng, "signed_nested_arithmetic", seed=42)
    expr = task["specification"].replace("Compute ", "").replace(". Return only the integer.", "")
    assert eval(expr) == task["expected"]  # noqa: S307 — test only


def test_unsupported_arithmetic_uses_division() -> None:
    """The malformed category uses true division, which the bounded
    integer evaluator rejects. This tests fallback behavior."""
    import random
    rng = random.Random(42)
    task = generate_task("test-1", rng, "unsupported_arithmetic", seed=42)
    assert task["inputs"]["op"] == "/"
    # The symbolic engine should reject this; the router should send it to LLM.
    assert task["utility_oracle"] == "llm"


def test_generate_benchmark_determinism() -> None:
    a = generate_benchmark(130, 42)
    b = generate_benchmark(130, 42)
    assert a == b


def test_generate_benchmark_balance() -> None:
    """Category counts must differ by at most one (cycled deterministically)."""
    tasks = generate_benchmark(130, 42)
    counts: dict[str, int] = {}
    for task in tasks:
        cat = task["metadata"]["category"]
        counts[cat] = counts.get(cat, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_generate_benchmark_with_category_subset() -> None:
    """--categories subset should produce only those categories."""
    tasks = generate_benchmark(50, 42, categories=("easy_add", "text_question"))
    cats = {task["metadata"]["category"] for task in tasks}
    assert cats == {"easy_add", "text_question"}


def test_generate_benchmark_shuffles_order() -> None:
    """Tasks should be shuffled, not in category-cycling order."""
    tasks = generate_benchmark(130, 42)
    cats = [task["metadata"]["category"] for task in tasks]
    # If not shuffled, adjacent tasks would cycle through categories.
    # Check that not all adjacent pairs differ (shuffling breaks the cycle).
    # More robust: the first 13 tasks should NOT be one of each category.
    first_13 = set(cats[:13])
    assert len(first_13) < 13 or cats[:13] != sorted(cats[:13], key=ALL_CATEGORIES.index)


def test_category_group_metadata_present() -> None:
    """Each task must carry category_group for aggregate analysis."""
    import random
    rng = random.Random(42)
    for category in ALL_CATEGORIES:
        task = generate_task("t", rng, category, seed=42)
        assert "category_group" in task["metadata"]
        assert task["metadata"]["category_group"] in {
            "symbolic", "non_symbolic", "adversarial", "malformed"
        }


def test_cli_symbolic_only_flag(tmp_path: Path, monkeypatch) -> None:
    mod = _load_generator_module()
    output = tmp_path / "sym.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "generate_ood_benchmark.py",
        "--output", str(output),
        "--num-tasks", "70",
        "--seed", "42",
        "--symbolic-only",
    ])
    mod.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {row["metadata"]["category_group"] for row in rows}
    assert groups == {"symbolic"}


def test_cli_non_symbolic_only_flag(tmp_path: Path, monkeypatch) -> None:
    mod = _load_generator_module()
    output = tmp_path / "ns.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "generate_ood_benchmark.py",
        "--output", str(output),
        "--num-tasks", "60",
        "--seed", "42",
        "--non-symbolic-only",
    ])
    mod.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {row["metadata"]["category_group"] for row in rows}
    assert groups == {"non_symbolic", "adversarial", "malformed"}


def test_cli_symbolic_and_non_symbolic_mutually_exclusive(tmp_path: Path, monkeypatch) -> None:
    mod = _load_generator_module()
    monkeypatch.setattr(sys, "argv", [
        "generate_ood_benchmark.py",
        "--output", str(tmp_path / "x.jsonl"),
        "--symbolic-only",
        "--non-symbolic-only",
    ])
    with pytest.raises(ValueError, match="mutually exclusive"):
        mod.main()


def test_cli_categories_flag(tmp_path: Path, monkeypatch) -> None:
    mod = _load_generator_module()
    output = tmp_path / "sub.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "generate_ood_benchmark.py",
        "--output", str(output),
        "--num-tasks", "20",
        "--seed", "42",
        "--categories", "easy_add", "text_question",
    ])
    mod.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    cats = {row["metadata"]["category"] for row in rows}
    assert cats == {"easy_add", "text_question"}


def test_evaluate_route_records_includes_oracle_kind() -> None:
    """evaluate_route_records must include label_oracle_kind in its output
    so downstream consumers cannot silently re-interpret a policy_heuristic
    result as a capability oracle."""
    from scripts.evaluate_routes import evaluate_route_records

    tasks = {"t1": {"task_id": "t1", "route_label": "symbolic", "specification": "x", "capability_ids": []}}
    routes = {"t1": {"task_id": "t1", "route": "symbolic"}}
    metrics = evaluate_route_records(tasks, routes, label_field="route_label", label_oracle_kind="policy_heuristic")
    assert metrics["label_oracle_kind"] == "policy_heuristic"
    assert metrics["label_field"] == "route_label"


def test_evaluate_route_records_oracle_kind_none_when_not_provided() -> None:
    from scripts.evaluate_routes import evaluate_route_records

    tasks = {"t1": {"task_id": "t1", "specification": "x", "capability_ids": []}}
    routes = {"t1": {"task_id": "t1", "route": "llm"}}
    metrics = evaluate_route_records(tasks, routes)
    assert metrics["label_oracle_kind"] is None
    assert metrics["label_field"] is None
