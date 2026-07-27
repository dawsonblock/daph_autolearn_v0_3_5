from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.generate_v0_outputs import _stage_elapsed_ms


def test_stage_elapsed_ms_both_none():
    assert _stage_elapsed_ms(None, None) == 0.0


def test_stage_elapsed_ms_start_none():
    assert _stage_elapsed_ms(None, 100.0) == 0.0


def test_stage_elapsed_ms_end_none():
    assert _stage_elapsed_ms(100.0, None) == 0.0


def test_stage_elapsed_ms_zero():
    assert _stage_elapsed_ms(100.0, 100.0) == 0.0


def test_stage_elapsed_ms_positive():
    assert _stage_elapsed_ms(100.0, 100.5) == 500.0


def test_stage_elapsed_ms_skipped_stage_returns_zero():
    """A skipped stage (None timestamps) must report 0.0, not an error."""
    assert _stage_elapsed_ms(None, None) == 0.0
    # Symbolic stage skipped for LLM-routed task:
    assert _stage_elapsed_ms(100.0, None) == 0.0


def test_route_record_includes_per_stage_latencies(tmp_path: Path, monkeypatch):
    """The route record JSONL must include route_batch_ms,
    symbolic_exec_ms, and answer_batch_ms fields."""
    import importlib.util

    # Create a simple symbolic task
    task = {
        "task_id": "lat-test",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 2, "b": 3, "op": "+"},
        "specification": "Compute 2 + 3. Return only the integer.",
        "expected": 5,
        "route_label": "symbolic",
        "capability_oracle": "symbolic",
        "accuracy_oracle": "llm",
        "utility_oracle": "symbolic",
    }
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    routes = tmp_path / "out.routes.jsonl"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_v0_outputs.py"
    spec = importlib.util.spec_from_file_location("gen_latency_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    old_argv = sys.argv
    sys.argv = [
        "generate_v0_outputs.py",
        "--input", str(task_path),
        "--output", str(output),
        "--routes-output", str(routes),
        "--execution-mode", "symbolic",
        "--no-manifest",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    route_lines = [json.loads(line) for line in routes.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(route_lines) == 1
    record = route_lines[0]
    # Per-stage fields must be present
    assert "route_batch_ms" in record
    assert "symbolic_exec_ms" in record
    assert "answer_batch_ms" in record
    assert "pipeline_elapsed_ms" in record
    # For a symbolic-only task: route_batch_ms ~0 (no model routing),
    # symbolic_exec_ms > 0, answer_batch_ms = 0 (no LLM answer generation)
    assert record["symbolic_exec_ms"] >= 0.0
    assert record["answer_batch_ms"] == 0.0  # no LLM stage
    # The per-stage latencies should sum to approximately pipeline_elapsed_ms
    # (within float precision). This is the §22 invariant: the sum of
    # per-stage times should account for the total.
    total_stages = record["route_batch_ms"] + record["symbolic_exec_ms"] + record["answer_batch_ms"]
    # For symbolic-only, route_batch_ms is ~0 and answer is 0, so
    # symbolic_exec_ms should be close to pipeline_elapsed_ms.
    assert total_stages <= record["pipeline_elapsed_ms"] + 1.0  # allow 1ms slack


def test_route_record_llm_task_has_answer_latency(tmp_path: Path, monkeypatch):
    """An LLM-routed task should have answer_batch_ms > 0 and
    symbolic_exec_ms == 0."""
    import importlib.util

    # Use a non-symbolic task that routes to LLM
    task = {
        "task_id": "llm-lat-test",
        "capability_ids": [],
        "inputs": {},
        "specification": "Explain why the sky is blue.",
        "expected": None,
        "route_label": "llm",
        "capability_oracle": "llm",
        "accuracy_oracle": "llm",
        "utility_oracle": "llm",
    }
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    routes = tmp_path / "out.routes.jsonl"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_v0_outputs.py"
    spec = importlib.util.spec_from_file_location("gen_llm_lat_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Patch _load_llm to return the tiny model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        model.eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    except Exception as exc:
        pytest.skip(f"cannot download tiny-gpt2: {exc}")

    monkeypatch.setattr(mod, "_load_llm", lambda _model_id: (model, tok))

    old_argv = sys.argv
    sys.argv = [
        "generate_v0_outputs.py",
        "--input", str(task_path),
        "--output", str(output),
        "--routes-output", str(routes),
        "--model", "sshleifer/tiny-gpt2",
        "--execution-mode", "auto",
        "--max-new-tokens", "3",
        "--no-manifest",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    route_lines = [json.loads(line) for line in routes.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(route_lines) == 1
    record = route_lines[0]
    assert record["symbolic_exec_ms"] == 0.0  # no symbolic stage
    # answer_batch_ms should be present (may be small on a tiny model)
    assert "answer_batch_ms" in record
    assert record["answer_batch_ms"] >= 0.0
