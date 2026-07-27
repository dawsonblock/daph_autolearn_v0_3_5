from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from daph_learning.evaluation.manifest import (
    MANIFEST_VERSION,
    ManifestValidationError,
    OutputEntry,
    RunManifest,
    VectorEntry,
    build_manifest,
    hash_file,
    hash_vector_values,
    load_manifest,
    manifest_sha256,
    serialize,
    validate,
    write_manifest,
)


def _minimal_manifest_kwargs(tmp_path: Path) -> dict:
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text('{"task_id":"t1"}\n', encoding="utf-8")
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"answer":"42"}\n', encoding="utf-8")
    return dict(
        run_id="run-1",
        daph_version="0.3.5",
        model={"repo": "Qwen/Qwen2.5-3B-Instruct"},
        tokenizer={"repo": "Qwen/Qwen2.5-3B-Instruct"},
        dataset={
            "path": str(dataset_path),
            "sha256": hash_file(dataset_path),
            "split": "dev",
        },
        decoding={
            "seed": 42,
            "prompt_format": "chat",
            "route_token_resolver": "isolated",
        },
        outputs=[
            {"path": str(output_path), "kind": "answers", "sha256": hash_file(output_path)}
        ],
    )


def test_build_and_validate_minimal(tmp_path: Path) -> None:
    manifest = build_manifest(**_minimal_manifest_kwargs(tmp_path))
    validate(manifest)  # should not raise


def test_manifest_version_is_v1() -> None:
    assert MANIFEST_VERSION == "daph.run.v1"


def test_missing_required_field_aggregates(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["dataset"] = {"path": "x", "sha256": None, "split": "dev"}
    kw["decoding"] = {"seed": 42, "prompt_format": "chat"}  # missing resolver
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError) as exc:
        validate(manifest)
    msg = str(exc.value)
    assert "dataset.sha256" in msg
    assert "decoding.route_token_resolver" in msg
    # both errors should be aggregated on one message
    assert ";" in msg


def test_unknown_split_rejected(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["dataset"] = dict(kw["dataset"], split="production")
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="dataset.split"):
        validate(manifest)


def test_unknown_token_resolver_rejected(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["decoding"] = dict(
        kw["decoding"], route_token_resolver="magic"
    )
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="route_token_resolver"):
        validate(manifest)


def test_split_leakage_rejected(tmp_path: Path) -> None:
    """A vector captured on the same SHA as a test-split dataset is circular."""
    kw = _minimal_manifest_kwargs(tmp_path)
    ds_sha = kw["dataset"]["sha256"]
    kw["dataset"] = dict(kw["dataset"], split="final_test")
    kw["vectors"] = [
        {
            "path": "vectors/v.npz",
            "vector_id": "tool-v1",
            "family": "tool_policy",
            "layer": 24,
            "alpha": 1.0,
            "token_scope": "last",
            "capture_dataset_sha256": ds_sha,
            "vector_sha256": "a" * 64,
            "extraction_method": "contrastive_mean_difference",
            "capture_anchor": "ACTION",
            "capture_prompt_format": "chat",
        }
    ]
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="split-leakage"):
        validate(manifest)


def test_split_leakage_does_not_fire_when_capture_sha_differs(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["dataset"] = dict(kw["dataset"], split="final_test")
    kw["vectors"] = [
        {
            "path": "vectors/v.npz",
            "vector_id": "tool-v1",
            "family": "tool_policy",
            "layer": 24,
            "alpha": 1.0,
            "token_scope": "last",
            "capture_dataset_sha256": "b" * 64,
            "vector_sha256": "a" * 64,
            "extraction_method": "contrastive_mean_difference",
            "capture_anchor": "ACTION",
            "capture_prompt_format": "chat",
        }
    ]
    manifest = build_manifest(**kw)
    validate(manifest)  # should not raise


def test_headline_validation_requires_provenance(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    # No model.revision, no git_commit, no vectors.
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError) as exc:
        validate(manifest, headline=True)
    msg = str(exc.value)
    assert "model.revision" in msg
    assert "git_commit" in msg
    assert "vectors" in msg


def test_headline_validation_passes_with_full_provenance(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["model"] = {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "abc123",
        "config_hash": "c" * 64,
        "dtype": "bfloat16",
    }
    kw["tokenizer"] = {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "abc123",
        "chat_template_hash": "d" * 64,
    }
    kw["environment"] = {
        "torch_version": "2.4.0",
        "transformers_version": "4.45.0",
    }
    kw["git_commit"] = "e" * 40
    kw["vectors"] = [
        {
            "path": "vectors/v.npz",
            "vector_id": "tool-v1",
            "family": "tool_policy",
            "layer": 24,
            "alpha": 1.0,
            "token_scope": "last",
            "capture_dataset_sha256": "b" * 64,
            "vector_sha256": "a" * 64,
            "extraction_method": "contrastive_mean_difference",
            "capture_anchor": "ACTION",
            "capture_prompt_format": "chat",
        }
    ]
    manifest = build_manifest(**kw)
    validate(manifest, headline=True)  # should not raise


def test_headline_validation_flags_missing_vector_provenance(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["model"] = {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "abc123",
        "config_hash": "c" * 64,
        "dtype": "bfloat16",
    }
    kw["tokenizer"] = {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "abc123",
        "chat_template_hash": "d" * 64,
    }
    kw["environment"] = {
        "torch_version": "2.4.0",
        "transformers_version": "4.45.0",
    }
    kw["git_commit"] = "e" * 40
    kw["vectors"] = [
        {
            "path": "vectors/v.npz",
            "vector_id": "tool-v1",
            "family": "tool_policy",
            "layer": 24,
            "alpha": 1.0,
            "token_scope": "last",
            # missing capture_dataset_sha256, vector_sha256, extraction_method,
            # capture_anchor, capture_prompt_format
        }
    ]
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError) as exc:
        validate(manifest, headline=True)
    msg = str(exc.value)
    assert "capture_dataset_sha256" in msg
    assert "vector_sha256" in msg
    assert "extraction_method" in msg


def test_serialize_is_canonical_and_stable(tmp_path: Path) -> None:
    manifest = build_manifest(**_minimal_manifest_kwargs(tmp_path))
    s1 = serialize(manifest)
    s2 = serialize(manifest)
    assert s1 == s2
    # sorted keys: "created_at" should appear before "daph_version"
    assert s1.index('"created_at"') < s1.index('"daph_version"')


def test_manifest_sha256_is_stable(tmp_path: Path) -> None:
    manifest = build_manifest(**_minimal_manifest_kwargs(tmp_path))
    h1 = manifest_sha256(manifest)
    h2 = manifest_sha256(manifest)
    assert h1 == h2
    assert len(h1) == 64


def test_manifest_sha256_changes_when_field_changes(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    m1 = build_manifest(**kw)
    kw["run_id"] = "run-2"
    m2 = build_manifest(**kw)
    assert manifest_sha256(m1) != manifest_sha256(m2)


def test_write_and_load_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = build_manifest(**_minimal_manifest_kwargs(tmp_path))
    path = tmp_path / "out.jsonl.manifest.json"
    sha = write_manifest(manifest, path)
    assert path.exists()
    assert len(sha) == 64
    loaded = load_manifest(path)
    assert loaded["manifest_version"] == MANIFEST_VERSION
    assert loaded["run_id"] == manifest.run_id
    # The on-disk SHA should equal the in-memory canonical SHA.
    assert sha == manifest_sha256(manifest)


def test_hash_vector_values_is_dtype_invariant() -> None:
    v_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert hash_vector_values(v_f32) == hash_vector_values(v_f64)


def test_hash_vector_values_changes_with_values() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.001], dtype=np.float32)
    assert hash_vector_values(a) != hash_vector_values(b)


def test_vector_entry_with_bad_family_rejected(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["vectors"] = [
        {
            "path": "vectors/v.npz",
            "vector_id": "tool-v1",
            "family": "magic_policy",  # not in LEGAL_VECTOR_FAMILIES
            "layer": 24,
            "alpha": 1.0,
            "token_scope": "last",
        }
    ]
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="family"):
        validate(manifest)


def test_vector_entry_with_bad_token_scope_rejected(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["vectors"] = [
        {
            "path": "vectors/v.npz",
            "vector_id": "tool-v1",
            "family": "tool_policy",
            "layer": 24,
            "alpha": 1.0,
            "token_scope": "always",  # not in LEGAL_TOKEN_SCOPES
        }
    ]
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="token_scope"):
        validate(manifest)


def test_outputs_must_be_non_empty(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["outputs"] = []
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="outputs"):
        validate(manifest)


def test_output_entry_dataclass_roundtrips() -> None:
    oe = OutputEntry(path="x.jsonl", kind="answers", sha256="a" * 64)
    assert oe.to_dict() == {"path": "x.jsonl", "kind": "answers", "sha256": "a" * 64}


def test_vector_entry_dataclass_defaults_none() -> None:
    ve = VectorEntry(
        path="v.npz",
        vector_id="v1",
        family="tool_policy",
        layer=24,
        alpha=1.0,
        token_scope="last",
    )
    d = ve.to_dict()
    # Provenance fields default to None, not omitted, so headline validation
    # can flag them rather than silently passing.
    for key in (
        "vector_sha256",
        "capture_dataset_sha256",
        "extraction_method",
        "capture_anchor",
        "capture_prompt_format",
    ):
        assert d[key] is None


def test_validate_accepts_plain_dict_payload(tmp_path: Path) -> None:
    manifest = build_manifest(**_minimal_manifest_kwargs(tmp_path))
    payload = json.loads(serialize(manifest))
    validate(payload)  # should not raise


def test_label_oracle_kind_must_be_legal(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["dataset"] = dict(kw["dataset"], label_oracle_kind="ground_truth")
    manifest = build_manifest(**kw)
    with pytest.raises(ManifestValidationError, match="label_oracle_kind"):
        validate(manifest)


def test_label_oracle_kind_null_is_allowed(tmp_path: Path) -> None:
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["dataset"] = dict(kw["dataset"], label_oracle_kind=None)
    manifest = build_manifest(**kw)
    validate(manifest)


def test_label_oracle_kind_policy_heuristic_is_documented(tmp_path: Path) -> None:
    """The current `route_label` field is a policy_heuristic oracle, not
    ground truth. This test exists so the schema forces callers to declare
    which kind of oracle they are using."""
    kw = _minimal_manifest_kwargs(tmp_path)
    kw["dataset"] = dict(kw["dataset"], label_oracle_kind="policy_heuristic")
    manifest = build_manifest(**kw)
    validate(manifest)
