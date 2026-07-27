"""V037-005: environment provenance capture and fail-closed manifests.

Acceptance criteria from ROADMAP §V037-005:

1. ``capture_environment`` uses ``importlib.metadata.version`` for
   installed-package versions (torch, transformers, accelerate, peft,
   safetensors, numpy, daph-learning).
2. CUDA runtime, CUDA driver, GPU name, and GPU compute capability are
   captured when torch + CUDA are available.
3. Attention implementation, dtype, and device map are captured from a
   loaded model when supplied.
4. ``fail_closed=True`` raises ``EnvironmentProvenanceError`` when a
   required field is missing.
5. ``emit_manifest(fail_closed_environment=True)`` propagates the
   fail-closed behavior.
6. The captured environment dict is compatible with the manifest
   validation schema (REQUIRED_FOR_HEADLINE_ENV).
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import pytest

from daph_learning.environment import (
    EnvironmentProvenance,
    EnvironmentProvenanceError,
    capture_environment,
)


# --- 1. importlib.metadata for installed packages ---

def test_capture_environment_uses_importlib_metadata():
    """The captured package versions must match importlib.metadata.version,
    not the imported __version__ (they can differ in editable installs)."""
    env = capture_environment(fail_closed=False)
    for pkg in ("torch", "transformers", "numpy"):
        installed = importlib.metadata.version(pkg)
        assert env["packages"][pkg] == installed, (
            f"packages[{pkg!r}] = {env['packages'][pkg]!r} != "
            f"importlib.metadata.version({pkg!r}) = {installed!r}"
        )
        # Also check the top-level convenience field.
        top_field = f"{pkg}_version"
        if top_field in env:
            assert env[top_field] == installed


def test_capture_environment_includes_all_tracked_packages():
    """All tracked packages must appear in the packages dict."""
    env = capture_environment(fail_closed=False)
    expected = {"torch", "transformers", "accelerate", "peft",
                "safetensors", "numpy", "daph-learning"}
    assert expected.issubset(set(env["packages"].keys()))


def test_capture_environment_uninstalled_package_is_none():
    """A package that is not installed must have version None, not raise."""
    env = capture_environment(fail_closed=False)
    # accelerate/peft/safetensors may or may not be installed; either way
    # the value must be a str or None, not an error.
    for pkg in ("accelerate", "peft", "safetensors"):
        val = env["packages"][pkg]
        assert val is None or isinstance(val, str)


# --- 2. Python + OS ---

def test_capture_environment_python_version():
    import platform
    env = capture_environment(fail_closed=False)
    assert env["python_version"] == platform.python_version()


def test_capture_environment_os_string():
    import platform
    env = capture_environment(fail_closed=False)
    assert platform.system().lower() in env["os"].lower()


# --- 3. CUDA / GPU info ---

def test_capture_environment_cuda_fields_present():
    """CUDA fields must be present (None if CUDA unavailable)."""
    env = capture_environment(fail_closed=False)
    for field in ("cuda_version", "cuda_driver_version", "gpu_model",
                  "gpu_compute_capability"):
        assert field in env


def test_capture_environment_cuda_consistency():
    """When CUDA is available, cuda_version and gpu_model must be non-None."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except (ImportError, ModuleNotFoundError):
        cuda_available = False

    env = capture_environment(fail_closed=False)
    if cuda_available:
        assert env["cuda_version"] is not None
        assert env["gpu_model"] is not None
        assert env["gpu_compute_capability"] is not None
    else:
        # CUDA unavailable -> all CUDA fields should be None
        assert env["cuda_version"] is None
        assert env["gpu_model"] is None


# --- 4. Model info ---

class _FakeModel:
    """Minimal model stub for testing _model_info extraction."""
    def __init__(self, attention="eager", dtype_str="float32", device_map=None):
        import torch
        import torch.nn as nn

        class C:
            _attn_implementation = attention
        self.config = C()
        self.embed = nn.Embedding(10, 4)
        if device_map is not None:
            self.hf_device_map = device_map

    def parameters(self):
        return self.embed.parameters()


def test_capture_environment_with_model_extracts_attention_dtype():
    env = capture_environment(model=_FakeModel(attention="sdpa"), fail_closed=False)
    assert env["attention_implementation"] == "sdpa"
    assert env["dtype"] == "float32"


def test_capture_environment_with_model_extracts_device_map():
    env = capture_environment(
        model=_FakeModel(device_map="auto"),
        fail_closed=False,
    )
    assert env["device_map"] == "auto"


def test_capture_environment_without_model_has_none_model_fields():
    env = capture_environment(model=None, fail_closed=False)
    assert env["attention_implementation"] is None
    assert env["dtype"] is None
    assert env["device_map"] is None


# --- 5. Fail-closed behavior ---

def test_fail_closed_raises_when_required_field_missing():
    """When fail_closed=True and torch_version is None (simulated by
    requiring a field that doesn't exist), the capture must raise."""
    # We can't easily uninstall torch, so we require a field that
    # capture_environment never populates.
    with pytest.raises(EnvironmentProvenanceError, match="missing required fields"):
        capture_environment(
            fail_closed=True,
            required_for_headline=("nonexistent_field",),
        )


def test_fail_closed_succeeds_when_all_required_present():
    """When fail_closed=True and all required fields are present, the
    capture must succeed."""
    # torch and transformers are installed in the test environment.
    env = capture_environment(
        fail_closed=True,
        required_for_headline=("torch_version", "transformers_version"),
    )
    assert env["torch_version"] is not None
    assert env["transformers_version"] is not None


def test_fail_closed_error_lists_missing_fields():
    """The error message must list the missing fields by name."""
    try:
        capture_environment(
            fail_closed=True,
            required_for_headline=("missing_a", "missing_b"),
        )
    except EnvironmentProvenanceError as exc:
        msg = str(exc)
        assert "missing_a" in msg
        assert "missing_b" in msg
    else:
        pytest.fail("EnvironmentProvenanceError was not raised")


# --- 6. emit_manifest integration ---

def test_emit_manifest_fail_closed_environment_propagates(tmp_path):
    """emit_manifest(fail_closed_environment=True) must raise
    EnvironmentProvenanceError when a required env field is missing."""
    from daph_learning.experiments.manifest import emit_manifest

    # Create a minimal dataset file.
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text('{"task_id": "t1", "capability_ids": ["integer_arithmetic"], '
                       '"inputs": {"a": 1, "b": 2, "op": "+"}, '
                       '"specification": "1 + 2", "expected": 3}\n',
                       encoding="utf-8")
    output = tmp_path / "routes.jsonl"
    output.write_text('{"task_id": "t1", "route": "symbolic"}\n',
                      encoding="utf-8")

    # We can't easily make torch_version None, so we monkeypatch
    # capture_environment to raise the fail-closed error.
    import daph_learning.environment as env_mod

    original = env_mod.capture_environment

    def _broken_env(**kwargs):
        # Simulate a missing torch_version by raising the fail-closed error.
        raise EnvironmentProvenanceError(
            "fail-closed environment provenance capture failed; missing "
            "required fields: ['torch_version']"
        )

    env_mod.capture_environment = _broken_env
    try:
        with pytest.raises(EnvironmentProvenanceError, match="torch_version"):
            emit_manifest(
                run_id="test-run",
                repo_root=tmp_path,
                output_path=output,
                output_kind="routes",
                model_id="test-model",
                dataset_path=dataset,
                dataset_split="test",
                label_field=None,
                label_oracle_kind=None,
                seed=42,
                prompt_format="raw",
                route_decision_mode="logit",
                route_token_resolver="isolated",
                batch_size=1,
                route_batch_size=1,
                max_new_tokens=5,
                fail_closed_environment=True,
            )
    finally:
        env_mod.capture_environment = original


def test_emit_manifest_default_does_not_fail_closed(tmp_path):
    """emit_manifest without fail_closed_environment must not raise even
    when env fields are missing (best-effort)."""
    from daph_learning.experiments.manifest import emit_manifest

    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text('{"task_id": "t1", "capability_ids": ["integer_arithmetic"], '
                       '"inputs": {"a": 1, "b": 2, "op": "+"}, '
                       '"specification": "1 + 2", "expected": 3}\n',
                       encoding="utf-8")
    output = tmp_path / "routes.jsonl"
    output.write_text('{"task_id": "t1", "route": "symbolic"}\n',
                      encoding="utf-8")

    # Default: fail_closed_environment=False -> should not raise.
    sha, manifest = emit_manifest(
        run_id="test-run",
        repo_root=tmp_path,
        output_path=output,
        output_kind="routes",
        model_id="test-model",
        dataset_path=dataset,
        dataset_split="test",
        label_field=None,
        label_oracle_kind=None,
        seed=42,
        prompt_format="raw",
        route_decision_mode="logit",
        route_token_resolver="isolated",
        batch_size=1,
        route_batch_size=1,
        max_new_tokens=5,
    )
    assert sha  # manifest was written
    # The environment dict must include the V037-005 package snapshot.
    env = manifest.environment
    assert "packages" in env
    assert "torch" in env["packages"]


# --- 7. EnvironmentProvenance dataclass ---

def test_environment_provenance_to_dict_roundtrip():
    prov = EnvironmentProvenance(
        python_version="3.12.0",
        os="darwin 25.2.0",
        torch_version="2.1.0",
        transformers_version="4.36.0",
    )
    d = prov.to_dict()
    assert d["python_version"] == "3.12.0"
    assert d["torch_version"] == "2.1.0"
    assert d["packages"] == {}  # default_factory
    # All V037-005 fields must be present in the dict.
    for field in ("python_version", "os", "packages", "torch_version",
                  "transformers_version", "accelerate_version", "peft_version",
                  "safetensors_version", "numpy_version", "daph_version",
                  "cuda_version", "cuda_driver_version", "gpu_model",
                  "gpu_compute_capability", "attention_implementation",
                  "dtype", "device_map"):
        assert field in d, f"missing field {field!r} in to_dict() output"
