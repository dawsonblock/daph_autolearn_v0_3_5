from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_generator_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_v0_outputs.py"
    spec = importlib.util.spec_from_file_location("generate_v0_outputs_v032", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_symbolic_fallback_prompt_retains_retrieved_memory(tmp_path, monkeypatch):
    mod = _load_generator_module()
    tasks = tmp_path / "tasks.jsonl"
    memory = tmp_path / "memory.jsonl"
    output = tmp_path / "out.jsonl"

    # '/' is deliberately unsupported by the exact integer executor, forcing
    # auto-mode symbolic execution to fail over to the LLM.
    tasks.write_text(json.dumps({
        "task_id": "fallback-1",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 8, "b": 2, "op": "/"},
        "specification": "Compute 8 / 2. Return only the integer.",
        "expected": 4,
    }) + "\n", encoding="utf-8")
    memory.write_text(json.dumps({
        "procedure_id": "rule_0",
        "capability_id": "integer_arithmetic",
        "trigger_terms": ["integer_arithmetic"],
        "action": "FALLBACK_MEMORY_SENTINEL",
        "confidence": 1.0,
        "positive_evidence": 1,
        "negative_evidence": 0,
        "source_episode_ids": [],
    }) + "\n", encoding="utf-8")

    captured = {}
    monkeypatch.setattr(mod, "_load_llm", lambda _model: (object(), object()))

    def fake_generate(prompt, *_args, **_kwargs):
        captured["prompt"] = prompt
        return "FINAL: 4"

    monkeypatch.setattr(mod, "_generate", fake_generate)
    monkeypatch.setattr(sys, "argv", [
        "generate_v0_outputs.py",
        "--input", str(tasks),
        "--output", str(output),
        "--model", "fake/model",
        "--procedural-memory", str(memory),
        "--execution-mode", "auto",
    ])
    mod.main()
    assert "FALLBACK_MEMORY_SENTINEL" in captured["prompt"]
