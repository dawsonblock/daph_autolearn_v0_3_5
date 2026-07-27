"""Tests for v0.3.6 Phase 3.1 manifest auto-fill helpers.

See scripts/_manifest.py:
- enrich_model_info_from_loaded_model
- enrich_tokenizer_info_from_loaded_tokenizer
- _config_hash, _chat_template_hash, _dtype_name

These tests verify the headline-required provenance fields are
auto-filled from loaded model/tokenizer objects, using a tiny fake
config-like object so they don't require a real HuggingFace model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts._manifest import (
    _chat_template_hash,
    _config_hash,
    _detect_model_info,
    _detect_tokenizer_info,
    enrich_model_info_from_loaded_model,
    enrich_tokenizer_info_from_loaded_tokenizer,
    emit_manifest,
)


class _FakeCfg:
    """Mimics HF PretrainedConfig enough for the enrichers."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def to_dict(self):
        return dict(self.__dict__)


class _FakeModel:
    def __init__(self, cfg, dtype_str="float32"):
        self.config = cfg
        import torch
        self._dtype = getattr(torch, dtype_str)
    def parameters(self):
        import torch
        yield torch.zeros(1, dtype=self._dtype)


class _FakeTokenizer:
    def __init__(self, template=None, pad_token_id=0, pad_side="left", commit=None):
        self.chat_template = template
        self.pad_token_id = pad_token_id
        self.padding_side = pad_side
        if commit:
            self._commit_hash = commit


# --- _config_hash ---

def test_config_hash_stable_across_key_order():
    cfg1 = _FakeCfg(hidden_size=896, num_hidden_layers=24, vocab_size=151936)
    cfg2 = _FakeCfg(vocab_size=151936, num_hidden_layers=24, hidden_size=896)
    assert _config_hash(_FakeModel(cfg1)) == _config_hash(_FakeModel(cfg2))


def test_config_hash_changes_when_config_changes():
    cfg1 = _FakeCfg(hidden_size=896)
    cfg2 = _FakeCfg(hidden_size=1536)
    assert _config_hash(_FakeModel(cfg1)) != _config_hash(_FakeModel(cfg2))


def test_config_hash_none_when_no_config():
    class NoCfg:
        config = None
    assert _config_hash(NoCfg()) is None


def test_config_hash_is_sha256_hex():
    cfg = _FakeCfg(hidden_size=896)
    h = _config_hash(_FakeModel(cfg))
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # valid hex


# --- _chat_template_hash ---

def test_chat_template_hash_stable():
    tok = _FakeTokenizer(template="{% for m in messages %}{{ m.content }}{% endfor %}")
    h = _chat_template_hash(tok)
    expected = hashlib.sha256(tok.chat_template.encode("utf-8")).hexdigest()
    assert h == expected


def test_chat_template_hash_none_when_no_template():
    tok = _FakeTokenizer(template=None)
    assert _chat_template_hash(tok) is None
    tok2 = _FakeTokenizer(template="")
    assert _chat_template_hash(tok2) is None


def test_chat_template_hash_changes_when_template_changes():
    h1 = _chat_template_hash(_FakeTokenizer(template="v1"))
    h2 = _chat_template_hash(_FakeTokenizer(template="v2"))
    assert h1 != h2


# --- enrich_model_info_from_loaded_model ---

def test_enrich_model_info_fills_headline_fields():
    base = _detect_model_info("Qwen/Qwen2.5-0.5B-Instruct")
    assert base["revision"] is None
    assert base["config_hash"] is None
    assert base["hidden_size"] is None
    assert base["num_layers"] is None
    assert base["dtype"] is None

    cfg = _FakeCfg(
        hidden_size=896, num_hidden_layers=24, vocab_size=151936,
        _commit_hash="abc123def456",
    )
    enriched = enrich_model_info_from_loaded_model(base, _FakeModel(cfg))
    assert enriched["hidden_size"] == 896
    assert enriched["num_layers"] == 24
    assert enriched["revision"] == "abc123def456"
    assert enriched["config_hash"] is not None
    assert len(enriched["config_hash"]) == 64
    assert enriched["dtype"] == "float32"
    # repo preserved
    assert enriched["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_enrich_model_info_handles_gpt2_fallback_field_names():
    """GPT-2 uses n_embd / n_layer instead of hidden_size / num_hidden_layers."""
    base = _detect_model_info("gpt2")
    cfg = _FakeCfg(n_embd=768, n_layer=12, vocab_size=50257)
    enriched = enrich_model_info_from_loaded_model(base, _FakeModel(cfg))
    assert enriched["hidden_size"] == 768
    assert enriched["num_layers"] == 12


def test_enrich_model_info_preserves_repo_when_no_commit_hash():
    base = _detect_model_info("some/model")
    cfg = _FakeCfg(hidden_size=64, num_hidden_layers=2)
    enriched = enrich_model_info_from_loaded_model(base, _FakeModel(cfg))
    assert enriched["revision"] is None  # no _commit_hash on cfg
    assert enriched["repo"] == "some/model"
    assert enriched["hidden_size"] == 64


# --- enrich_tokenizer_info_from_loaded_tokenizer ---

def test_enrich_tokenizer_info_fills_headline_fields():
    base = _detect_tokenizer_info("Qwen/Qwen2.5-0.5B-Instruct")
    assert base["chat_template_hash"] is None
    assert base["pad_token_id"] is None
    assert base["pad_side"] is None

    tok = _FakeTokenizer(
        template="<|im_start|>assistant\n",
        pad_token_id=151643,
        pad_side="left",
        commit="rev123",
    )
    enriched = enrich_tokenizer_info_from_loaded_tokenizer(base, tok)
    assert enriched["chat_template_hash"] is not None
    assert len(enriched["chat_template_hash"]) == 64
    assert enriched["pad_token_id"] == 151643
    assert enriched["pad_side"] == "left"
    assert enriched["revision"] == "rev123"
    assert enriched["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_enrich_tokenizer_info_no_template_keeps_hash_none():
    base = _detect_tokenizer_info("gpt2")
    tok = _FakeTokenizer(template=None, pad_token_id=50256, pad_side="right")
    enriched = enrich_tokenizer_info_from_loaded_tokenizer(base, tok)
    assert enriched["chat_template_hash"] is None
    assert enriched["pad_token_id"] == 50256
    assert enriched["pad_side"] == "right"


# --- emit_manifest integration: loaded_model/tokenizer flow through ---

def test_emit_manifest_enriches_from_loaded_objects(tmp_path: Path):
    """When loaded_model and loaded_tokenizer are passed to emit_manifest,
    the resulting manifest's model/tokenizer dicts must carry the
    headline-required fields (config_hash, chat_template_hash, etc.)
    instead of None."""
    # Build a tiny dataset file.
    ds = tmp_path / "tasks.jsonl"
    ds.write_text(
        json.dumps({"task_id": "t1", "capability_ids": [], "specification": "x"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    out.write_text('{"task_id":"t1","output":"x"}\n', encoding="utf-8")

    cfg = _FakeCfg(
        hidden_size=896, num_hidden_layers=24, vocab_size=151936,
        _commit_hash="abcdef1234567890",
    )
    model = _FakeModel(cfg)
    tok = _FakeTokenizer(
        template="<|im_start|>assistant\n",
        pad_token_id=151643,
        pad_side="left",
    )

    sha, manifest = emit_manifest(
        run_id="test-run",
        repo_root=tmp_path,
        output_path=out,
        output_kind="answers",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path=ds,
        dataset_split="test",
        label_field="route_label",
        label_oracle_kind="utility",
        seed=42,
        prompt_format="chat",
        route_decision_mode="logit",
        route_token_resolver="contextual",
        batch_size=8,
        route_batch_size=8,
        max_new_tokens=64,
        loaded_model=model,
        loaded_tokenizer=tok,
        validate_headline=False,
    )
    payload = manifest.to_dict()
    assert payload["model"]["hidden_size"] == 896
    assert payload["model"]["num_layers"] == 24
    assert payload["model"]["config_hash"] is not None
    assert payload["model"]["dtype"] == "float32"
    assert payload["model"]["revision"] == "abcdef1234567890"
    assert payload["tokenizer"]["chat_template_hash"] is not None
    assert payload["tokenizer"]["pad_token_id"] == 151643
    assert payload["tokenizer"]["pad_side"] == "left"
    # Manifest file written
    assert (tmp_path / "out.jsonl.manifest.json").exists()


def test_emit_manifest_without_loaded_objects_keeps_none_defaults(tmp_path: Path):
    """When loaded_model/tokenizer are NOT passed (the v0.3.5 path), the
    manifest must still work with None defaults — backward compatible."""
    ds = tmp_path / "tasks.jsonl"
    ds.write_text(
        json.dumps({"task_id": "t1", "capability_ids": [], "specification": "x"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    out.write_text('{"task_id":"t1","output":"x"}\n', encoding="utf-8")

    sha, manifest = emit_manifest(
        run_id="test-run",
        repo_root=tmp_path,
        output_path=out,
        output_kind="answers",
        model_id="some/model",
        dataset_path=ds,
        dataset_split="test",
        label_field="route_label",
        label_oracle_kind="utility",
        seed=42,
        prompt_format="raw",
        route_decision_mode="logit",
        route_token_resolver="isolated",
        batch_size=1,
        route_batch_size=1,
        max_new_tokens=32,
    )
    payload = manifest.to_dict()
    # v0.3.5 defaults preserved when no loaded objects supplied
    assert payload["model"]["config_hash"] is None
    assert payload["model"]["hidden_size"] is None
    assert payload["tokenizer"]["chat_template_hash"] is None


def test_emit_manifest_headline_validation_passes_with_loaded_objects(tmp_path: Path):
    """With loaded_model/tokenizer + a steering vector that carries
    capture provenance, headline validation must pass. This is the
    Phase 3.1 payoff: a single CLI run can now produce a
    headline-eligible manifest without manual field-filling."""
    from daph_learning.steering.types import SteeringSpec, SteeringVector
    import numpy as np

    ds = tmp_path / "tasks.jsonl"
    ds.write_text(
        json.dumps({"task_id": "t1", "capability_ids": [], "specification": "x"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    out.write_text('{"task_id":"t1","output":"x"}\n', encoding="utf-8")

    cfg = _FakeCfg(
        hidden_size=8, num_hidden_layers=2, vocab_size=100,
        _commit_hash="commit123",
    )
    model = _FakeModel(cfg)
    tok = _FakeTokenizer(
        template="assistant\n",
        pad_token_id=0,
        pad_side="left",
        commit="commit123",
    )

    spec = SteeringSpec(
        vector_id="v1", family="tool_policy", behavior="invoke_symbolic_tool",
        layer=1, alpha=1.0, anchor="ACTION:", model_id="some/model",
        hidden_size=8, extraction_method="contrastive_mean_difference",
        normalization="l2", positive_n=5, negative_n=5,
        capture_anchor="ACTION:", capture_prompt_format="raw",
        capture_dataset_sha256="a" * 64,
    )
    vec = SteeringVector(spec=spec, values=np.ones(8, dtype=np.float32))

    # Initialize a git repo so git_commit is populated.
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

    sha, manifest = emit_manifest(
        run_id="headline-run",
        repo_root=tmp_path,
        output_path=out,
        output_kind="answers",
        model_id="some/model",
        dataset_path=ds,
        dataset_split="test",
        label_field="route_label",
        label_oracle_kind="utility",
        seed=42,
        prompt_format="raw",
        route_decision_mode="logit",
        route_token_resolver="contextual",
        batch_size=1,
        route_batch_size=1,
        max_new_tokens=32,
        tool_vectors=[vec],
        tool_vector_paths=[str(tmp_path / "v.npz")],
        tool_alphas=[1.0],
        loaded_model=model,
        loaded_tokenizer=tok,
        validate_headline=True,
    )
    # If we got here, headline validation passed.
    payload = manifest.to_dict()
    assert payload["model"]["config_hash"] is not None
    assert payload["tokenizer"]["chat_template_hash"] is not None
    assert payload["git_commit"] is not None
