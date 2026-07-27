from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from daph_learning.evaluation.manifest import (
    MANIFEST_VERSION,
    load_manifest,
    manifest_sha256,
    validate,
)


def _load_generator_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_v0_outputs.py"
    spec = importlib.util.spec_from_file_location("generate_v0_outputs_manifest_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tasks(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "task_id": "manifest-1",
                "capability_ids": ["integer_arithmetic"],
                "inputs": {"a": 12, "b": 34, "op": "*"},
                "specification": "Compute 12 * 34. Return only the integer.",
                "expected": 408,
                "route_label": "symbolic",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_generate_v0_outputs_emits_manifest(tmp_path, monkeypatch, capsys):
    mod = _load_generator_module()
    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "out.jsonl"
    routes = tmp_path / "out.routes.jsonl"
    _write_tasks(tasks)

    monkeypatch.setattr(sys, "argv", [
        "generate_v0_outputs.py",
        "--input", str(tasks),
        "--output", str(output),
        "--routes-output", str(routes),
        "--execution-mode", "symbolic",
        "--dataset-split", "dev",
        "--label-field", "route_label",
        "--label-oracle-kind", "policy_heuristic",
        "--run-id", "test-run-1",
    ])
    mod.main()

    manifest_path = Path(str(output) + ".manifest.json")
    assert manifest_path.exists(), "manifest sibling file should be written"

    manifest = load_manifest(manifest_path)
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["run_id"] == "test-run-1"
    assert manifest["dataset"]["split"] == "dev"
    assert manifest["dataset"]["label_field"] == "route_label"
    assert manifest["dataset"]["label_oracle_kind"] == "policy_heuristic"
    assert manifest["dataset"]["sha256"] is not None
    assert len(manifest["dataset"]["sha256"]) == 64
    assert manifest["model"]["repo"] == "none"  # symbolic-only run
    assert manifest["tokenizer"]["repo"] == "none"
    assert len(manifest["outputs"]) == 2  # answers + routes
    kinds = {o["kind"] for o in manifest["outputs"]}
    assert kinds == {"answers", "routes"}

    # The manifest should pass non-headline validation.
    validate(manifest)

    # The printed SHA should match the on-disk manifest's SHA.
    captured = capsys.readouterr()
    sha_line = [line for line in captured.out.splitlines() if line.startswith("manifest:")]
    assert len(sha_line) == 1
    assert manifest_sha256(manifest) in sha_line[0]


def test_generate_v0_outputs_no_manifest_flag_suppresses_file(tmp_path, monkeypatch):
    mod = _load_generator_module()
    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "out_nomf.jsonl"
    _write_tasks(tasks)

    monkeypatch.setattr(sys, "argv", [
        "generate_v0_outputs.py",
        "--input", str(tasks),
        "--output", str(output),
        "--execution-mode", "symbolic",
        "--no-manifest",
    ])
    mod.main()

    manifest_path = Path(str(output) + ".manifest.json")
    assert not manifest_path.exists()


def test_generate_v0_outputs_manifest_records_split_and_oracle_kind(tmp_path, monkeypatch):
    """The manifest must record split and label_oracle_kind so that
    anti-circularity and overclaim guards are enforceable downstream."""
    mod = _load_generator_module()
    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "out_split.jsonl"
    _write_tasks(tasks)

    monkeypatch.setattr(sys, "argv", [
        "generate_v0_outputs.py",
        "--input", str(tasks),
        "--output", str(output),
        "--execution-mode", "symbolic",
        "--dataset-split", "final_test",
        "--label-field", "route_label",
        "--label-oracle-kind", "policy_heuristic",
        "--run-id", "split-test",
    ])
    mod.main()

    manifest = load_manifest(Path(str(output) + ".manifest.json"))
    assert manifest["dataset"]["split"] == "final_test"
    assert manifest["dataset"]["label_oracle_kind"] == "policy_heuristic"


def test_evaluate_routes_emits_manifest(tmp_path, monkeypatch, capsys):
    """evaluate_routes.py should emit a manifest when --output is given."""
    eval_script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_routes.py"
    spec = importlib.util.spec_from_file_location("evaluate_routes_manifest_test", eval_script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    tasks = tmp_path / "tasks.jsonl"
    routes = tmp_path / "routes.jsonl"
    output = tmp_path / "eval.json"

    tasks.write_text(
        json.dumps(
            {
                "task_id": "ev-1",
                "capability_ids": ["integer_arithmetic"],
                "inputs": {"a": 12, "b": 34, "op": "*"},
                "specification": "Compute 12 * 34. Return only the integer.",
                "expected": 408,
                "route_label": "symbolic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    routes.write_text(
        json.dumps(
            {
                "task_id": "ev-1",
                "route": "symbolic",
                "route_method": "batched_logit_contrast",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", [
        "evaluate_routes.py",
        "--tasks", str(tasks),
        "--routes", str(routes),
        "--label-field", "route_label",
        "--label-oracle-kind", "policy_heuristic",
        "--dataset-split", "dev",
        "--output", str(output),
        "--run-id", "eval-test",
    ])
    mod.main()

    manifest_path = Path(str(output) + ".manifest.json")
    assert manifest_path.exists()
    manifest = load_manifest(manifest_path)
    assert manifest["run_id"] == "eval-test"
    assert manifest["dataset"]["label_oracle_kind"] == "policy_heuristic"
    assert manifest["outputs"][0]["kind"] == "route_evaluation"
    validate(manifest)


def test_evaluate_routes_no_manifest_flag(tmp_path, monkeypatch):
    eval_script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_routes.py"
    spec = importlib.util.spec_from_file_location("evaluate_routes_nomf_test", eval_script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    tasks = tmp_path / "tasks.jsonl"
    routes = tmp_path / "routes.jsonl"
    output = tmp_path / "eval_nomf.json"

    tasks.write_text(
        json.dumps(
            {
                "task_id": "ev-1",
                "capability_ids": ["integer_arithmetic"],
                "inputs": {"a": 12, "b": 34, "op": "*"},
                "specification": "Compute 12 * 34.",
                "expected": 408,
                "route_label": "symbolic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    routes.write_text(
        json.dumps({"task_id": "ev-1", "route": "symbolic"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", [
        "evaluate_routes.py",
        "--tasks", str(tasks),
        "--routes", str(routes),
        "--output", str(output),
        "--no-manifest",
    ])
    mod.main()

    assert not Path(str(output) + ".manifest.json").exists()


def test_load_vector_cli_returns_paths(tmp_path):
    """_load_vector_cli must return (vectors, source_paths) for manifest provenance."""
    import numpy as np

    from daph_learning.steering.io import save_vector
    from daph_learning.steering.types import SteeringSpec, SteeringVector

    vec_path = tmp_path / "v.npz"
    spec = SteeringSpec(
        vector_id="v1",
        family="tool_policy",
        behavior="b",
        layer=24,
        alpha=1.0,
        anchor="ACTION",
        model_id="m",
        hidden_size=4,
    )
    save_vector(
        SteeringVector(spec=spec, values=np.zeros(4, dtype=np.float32)),
        vec_path,
    )

    mod = _load_generator_module()
    vectors, paths = mod._load_vector_cli(
        single_path=str(vec_path),
        bundle_spec=None,
        deprecated_path=None,
    )
    assert len(vectors) == 1
    assert len(paths) == 1
    assert paths[0] == str(vec_path.resolve())
