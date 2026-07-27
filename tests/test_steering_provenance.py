from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from daph_learning.steering.extract import contrastive_mean_direction
from daph_learning.steering.io import load_vector, save_vector
from daph_learning.steering.types import SteeringSpec, SteeringVector


def _make_spec(**overrides) -> SteeringSpec:
    defaults = dict(
        vector_id="v1",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=24,
        alpha=1.0,
        anchor="ACTION",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        hidden_size=4,
    )
    defaults.update(overrides)
    return SteeringSpec(**defaults)


def test_steering_spec_defaults_provenance_to_none() -> None:
    """Older code that constructs SteeringSpec without provenance fields
    must still work — the new fields default to None."""
    spec = SteeringSpec(
        vector_id="v",
        family="tool_policy",
        behavior="b",
        layer=1,
    )
    assert spec.extraction_method is None
    assert spec.normalization is None
    assert spec.positive_n is None
    assert spec.negative_n is None
    assert spec.capture_anchor is None
    assert spec.capture_prompt_format is None
    assert spec.capture_dataset_path is None
    assert spec.capture_dataset_sha256 is None


def test_steering_spec_accepts_provenance_fields() -> None:
    spec = _make_spec(
        extraction_method="contrastive_mean_difference",
        normalization="l2",
        positive_n=200,
        negative_n=200,
        capture_anchor="ACTION:",
        capture_prompt_format="chat",
        capture_dataset_path="data/router_train.jsonl",
        capture_dataset_sha256="a" * 64,
    )
    assert spec.extraction_method == "contrastive_mean_difference"
    assert spec.normalization == "l2"
    assert spec.positive_n == 200
    assert spec.negative_n == 200
    assert spec.capture_anchor == "ACTION:"
    assert spec.capture_prompt_format == "chat"
    assert spec.capture_dataset_sha256 == "a" * 64


def test_provenance_roundtrips_through_save_and_load(tmp_path: Path) -> None:
    """Provenance fields must survive serialization to .npz and back."""
    spec = _make_spec(
        extraction_method="contrastive_mean_difference",
        normalization="l2",
        positive_n=50,
        negative_n=60,
        capture_anchor="ACTION:",
        capture_prompt_format="chat",
        capture_dataset_path="data/cap.jsonl",
        capture_dataset_sha256="b" * 64,
    )
    vec = SteeringVector(spec=spec, values=np.zeros(4, dtype=np.float32))
    path = tmp_path / "v.npz"
    save_vector(vec, path)
    loaded = load_vector(path)
    assert loaded.spec.extraction_method == "contrastive_mean_difference"
    assert loaded.spec.normalization == "l2"
    assert loaded.spec.positive_n == 50
    assert loaded.spec.negative_n == 60
    assert loaded.spec.capture_anchor == "ACTION:"
    assert loaded.spec.capture_prompt_format == "chat"
    assert loaded.spec.capture_dataset_path == "data/cap.jsonl"
    assert loaded.spec.capture_dataset_sha256 == "b" * 64


def test_old_npz_without_provenance_still_loads(tmp_path: Path) -> None:
    """A .npz file saved before the provenance patch (metadata dict has
    only the original fields) must still load, with provenance fields
    defaulting to None."""
    spec = _make_spec()  # no provenance fields
    vec = SteeringVector(spec=spec, values=np.zeros(4, dtype=np.float32))
    path = tmp_path / "old.npz"
    save_vector(vec, path)
    loaded = load_vector(path)
    assert loaded.spec.vector_id == "v1"
    assert loaded.spec.extraction_method is None
    assert loaded.spec.capture_dataset_sha256 is None


def test_extract_uses_spec_normalization_l2() -> None:
    """When spec.normalization == 'l2', the direction is L2-normalized."""
    spec = _make_spec(normalization="l2")
    pos = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec = contrastive_mean_direction(pos, neg, spec)
    norm = float(np.linalg.norm(vec.values))
    assert abs(norm - 1.0) < 1e-6


def test_extract_uses_spec_normalization_none() -> None:
    """When spec.normalization == 'none', the direction is NOT normalized."""
    spec = _make_spec(normalization="none")
    pos = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec = contrastive_mean_direction(pos, neg, spec)
    assert vec.values[0] == 3.0  # unnormalized mean difference


def test_extract_explicit_normalize_overrides_spec() -> None:
    """An explicit normalize argument takes priority over spec.normalization."""
    spec = _make_spec(normalization="none")
    pos = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec = contrastive_mean_direction(pos, neg, spec, normalize=True)
    norm = float(np.linalg.norm(vec.values))
    assert abs(norm - 1.0) < 1e-6


def test_extract_defaults_to_normalize_when_spec_normalization_is_none() -> None:
    """Backward compatibility: spec.normalization == None -> normalize=True."""
    spec = _make_spec(normalization=None)
    pos = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec = contrastive_mean_direction(pos, neg, spec)
    norm = float(np.linalg.norm(vec.values))
    assert abs(norm - 1.0) < 1e-6


def test_extract_steering_vector_cli_populates_provenance(tmp_path: Path) -> None:
    """End-to-end: extract_steering_vector.py CLI should embed provenance
    in the saved .npz metadata."""
    import importlib.util

    pos = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pos_path = tmp_path / "pos.npy"
    neg_path = tmp_path / "neg.npy"
    np.save(pos_path, pos)
    np.save(neg_path, neg)
    out_path = tmp_path / "v.npz"

    script = tmp_path.parent.parent / "scripts" / "extract_steering_vector.py"
    # Resolve the actual repo script path
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "extract_steering_vector.py"
    spec = importlib.util.spec_from_file_location("extract_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    import sys

    cap_sha = "c" * 64
    old_argv = sys.argv
    sys.argv = [
        "extract_steering_vector.py",
        "--positive", str(pos_path),
        "--negative", str(neg_path),
        "--output", str(out_path),
        "--vector-id", "v1",
        "--family", "tool_policy",
        "--behavior", "invoke_symbolic_tool",
        "--layer", "24",
        "--model-id", "Qwen/Qwen2.5-3B-Instruct",
        "--capture-anchor", "ACTION:",
        "--capture-prompt-format", "chat",
        "--capture-dataset-path", "data/cap.jsonl",
        "--capture-dataset-sha256", cap_sha,
        "--positive-n", "2",
        "--negative-n", "2",
        "--extraction-method", "contrastive_mean_difference",
        "--normalization", "l2",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    loaded = load_vector(out_path)
    assert loaded.spec.extraction_method == "contrastive_mean_difference"
    assert loaded.spec.normalization == "l2"
    assert loaded.spec.positive_n == 2
    assert loaded.spec.negative_n == 2
    assert loaded.spec.capture_anchor == "ACTION:"
    assert loaded.spec.capture_prompt_format == "chat"
    assert loaded.spec.capture_dataset_path == "data/cap.jsonl"
    assert loaded.spec.capture_dataset_sha256 == cap_sha


def test_extract_steering_vector_cli_auto_fills_sample_counts(tmp_path: Path) -> None:
    """When --positive-n / --negative-n are omitted, they are auto-filled
    from the array shapes."""
    import importlib.util
    import sys

    pos = np.array([[1.0, 0.0, 0.0, 0.0]] * 7, dtype=np.float32)
    neg = np.array([[0.0, 0.0, 0.0, 0.0]] * 5, dtype=np.float32)
    pos_path = tmp_path / "pos.npy"
    neg_path = tmp_path / "neg.npy"
    np.save(pos_path, pos)
    np.save(neg_path, neg)
    out_path = tmp_path / "v.npz"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "extract_steering_vector.py"
    spec = importlib.util.spec_from_file_location("extract_test2", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    old_argv = sys.argv
    sys.argv = [
        "extract_steering_vector.py",
        "--positive", str(pos_path),
        "--negative", str(neg_path),
        "--output", str(out_path),
        "--vector-id", "v1",
        "--family", "tool_policy",
        "--behavior", "b",
        "--layer", "24",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    loaded = load_vector(out_path)
    assert loaded.spec.positive_n == 7
    assert loaded.spec.negative_n == 5


def test_manifest_helper_reads_provenance_from_spec(tmp_path: Path) -> None:
    """scripts._manifest._vector_entry_from_steering_vector should read
    provenance from the SteeringSpec, not hardcode None."""
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "_manifest.py"
    spec = importlib.util.spec_from_file_location("_manifest_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register in sys.modules so dataclass annotation resolution works.
    sys.modules["_manifest_test"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("_manifest_test", None)

    cap_sha = "d" * 64
    steering_spec = _make_spec(
        extraction_method="contrastive_mean_difference",
        normalization="l2",
        positive_n=100,
        negative_n=100,
        capture_anchor="ACTION:",
        capture_prompt_format="chat",
        capture_dataset_sha256=cap_sha,
    )
    vec = SteeringVector(spec=steering_spec, values=np.zeros(4, dtype=np.float32))
    entry = mod._vector_entry_from_steering_vector(
        vec, alpha_override=None, path="v.npz", token_scope="last"
    )
    assert entry.extraction_method == "contrastive_mean_difference"
    assert entry.normalization == "l2"
    assert entry.positive_n == 100
    assert entry.negative_n == 100
    assert entry.capture_anchor == "ACTION:"
    assert entry.capture_prompt_format == "chat"
    assert entry.capture_dataset_sha256 == cap_sha
    assert entry.vector_sha256 is not None
    assert len(entry.vector_sha256) == 64
