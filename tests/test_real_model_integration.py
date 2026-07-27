"""Real-model integration tests for the DAPH v0.3.5 stack.

These tests exercise the full integrated path:

    capture → extract → load → steer → route → symbolic execute → fallback
    → answer generation

against a real HuggingFace model. They are skipped if torch or
transformers is not installed, or if the model cannot be downloaded
(offline environments).

The tests use ``sshleifer/tiny-gpt2`` — a ~1MB model with a real BPE
tokenizer and 2 transformer layers. This is NOT a meaningful model for
steering research; it exists only to verify that the integrated stack
runs without errors and that the wiring between subsystems is correct.

Per the audit (§24), the qualification gate for v0.3.6 is:

    1 model, 2 layers, 2 vectors, 8-16 prompts,
    batch sizes 1 / 2 / 8, raw + chat, logit + generate routing,
    CPU/GPU if possible.

These tests cover the wiring; they do NOT qualify the steering method
scientifically. See CLAIMS.md §4 and §6.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

# Skip the entire module if torch or transformers is not available.
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.extract import contrastive_mean_direction
from daph_learning.steering.hooks import (
    capture_layer_output,
    multi_layer_residual_addition_hook,
    resolve_transformer_layers,
    validate_vector_for_model,
)
from daph_learning.steering.io import load_vector, save_vector
from daph_learning.steering.types import SteeringSpec, SteeringVector
from scripts.generate_v0_outputs import (
    _format_for_model,
    _generate_batch,
    _load_llm,
)
from scripts.generate_ood_benchmark import generate_benchmark


# Use a tiny model that downloads in seconds. This is NOT a meaningful
# model for steering research — it only verifies the wiring.
TINY_MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    """Load the tiny model once per module run."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID)
        model.eval()
    except Exception as exc:
        pytest.skip(f"cannot download {TINY_MODEL_ID}: {exc}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_resolve_transformer_layers_real_model(tiny_model):
    """resolve_transformer_layers must find the layer ModuleList on a real model."""
    model, _ = tiny_model
    layers = resolve_transformer_layers(model)
    assert len(layers) >= 1


def test_validate_vector_for_model_real_model(tiny_model):
    """validate_vector_for_model must accept a vector matching hidden_size."""
    model, _ = tiny_model
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    spec = SteeringSpec(
        vector_id="v", family="tool_policy", behavior="b",
        layer=0, hidden_size=hidden,
    )
    vec = SteeringVector(spec=spec, values=torch.zeros(hidden, dtype=torch.float32).numpy())
    validate_vector_for_model(model, vec)  # should not raise


def test_validate_vector_for_model_rejects_mismatched_dim(tiny_model):
    model, _ = tiny_model
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    spec = SteeringSpec(
        vector_id="v", family="tool_policy", behavior="b",
        layer=0, hidden_size=hidden + 1,
    )
    vec = SteeringVector(spec=spec, values=torch.zeros(hidden + 1, dtype=torch.float32).numpy())
    with pytest.raises(ValueError):
        validate_vector_for_model(model, vec)


def test_find_anchor_token_index_real_tokenizer(tiny_model):
    """find_anchor_token_index must locate ACTION: in a rendered prompt
    using a real tokenizer's offset mappings or fallback."""
    _, tokenizer = tiny_model
    prompt = "Choose the execution backend.\n\nACTION:"
    idx = find_anchor_token_index(prompt, tokenizer, "ACTION:")
    assert idx is not None
    assert idx >= 0
    # The index must be within the tokenized length.
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    assert idx < encoded["input_ids"].shape[1]


def test_capture_layer_output_real_model(tiny_model):
    """capture_layer_output must capture a real activation from a real forward pass."""
    model, tokenizer = tiny_model
    prompt = "Compute 12 * 34. Return only the integer."
    encoded = tokenizer(prompt, return_tensors="pt")
    sink: list = []
    with torch.inference_mode(), capture_layer_output(
        model, layer_index=0, sink=sink, target_token_index=-1,
    ):
        model(**encoded, use_cache=False)
    assert len(sink) == 1
    activation = sink[0]
    assert activation.ndim == 2
    assert activation.shape[0] == 1  # batch size 1


def test_multi_layer_residual_hook_real_model(tiny_model):
    """multi_layer_residual_addition_hook must install hooks on multiple
    layers of a real model and produce a different output than baseline."""
    model, tokenizer = tiny_model
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    prompt = "Hello world"
    encoded = tokenizer(prompt, return_tensors="pt")

    # Baseline forward
    with torch.inference_mode():
        baseline = model(**encoded, use_cache=False).logits

    # Steered forward with a large vector to ensure a visible difference
    # even on a tiny model with hidden_size=2.
    vec = torch.ones(hidden, dtype=torch.float32) * 100.0
    configs = [
        {"layer_index": 0, "vector": vec.numpy(), "alpha": 5.0, "token_scope": "all"},
        {"layer_index": 1, "vector": vec.numpy(), "alpha": 5.0, "token_scope": "all"},
    ]
    with torch.inference_mode(), multi_layer_residual_addition_hook(model, configs):
        steered = model(**encoded, use_cache=False).logits

    # The steered output should differ from baseline. tiny-gpt2 has
    # hidden_size=2, so the perturbation's effect on logits is small
    # (~1e-5), but it must be non-zero. We check that the max absolute
    # difference exceeds a threshold well above float32 noise.
    max_diff = (steered - baseline).abs().max().item()
    assert max_diff > 1e-7, (
        f"steering hooks produced no visible change (max diff {max_diff}); "
        "the hook may not be installed"
    )


def test_generate_batch_real_model(tiny_model):
    """_generate_batch must produce completions for a batch of prompts
    using a real model with left-padded batching."""
    model, tokenizer = tiny_model
    prompts = [
        "Compute 12 * 34. Return only the integer.",
        "Compute 99 + 1. Return only the integer.",
        "Explain why the sky is blue.",
    ]
    outputs = _generate_batch(
        prompts,
        model,
        tokenizer,
        max_new_tokens=5,
        seed=42,
        prompt_format="raw",
    )
    assert len(outputs) == 3
    assert all(isinstance(o, str) for o in outputs)
    # Each output should be non-empty (the model produces *something*).
    assert all(o for o in outputs)


def test_generate_batch_with_steering_real_model(tiny_model):
    """_generate_batch with steering vectors must run without errors and
    produce output (the output need not be meaningful for tiny-gpt2)."""
    model, tokenizer = tiny_model
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    spec = SteeringSpec(
        vector_id="test-v", family="reasoning_policy", behavior="test",
        layer=0, alpha=1.0, hidden_size=hidden,
        model_id=TINY_MODEL_ID,
    )
    vec = SteeringVector(spec=spec, values=torch.ones(hidden, dtype=torch.float32).numpy())
    prompts = ["Compute 12 * 34.", "Compute 99 + 1."]
    outputs = _generate_batch(
        prompts,
        model,
        tokenizer,
        max_new_tokens=5,
        seed=42,
        steering_vectors=[vec],
        steering_alphas=[1.0],
        steering_scope="all",
        prompt_format="raw",
    )
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_batch_batch_size_consistency(tiny_model):
    """The same prompt must produce the same output regardless of batch
    size (with deterministic decoding). This is the audit's §24 check:

        batch_size_1_decision == same_example_inside_batch_decision

    We check generation output rather than routing decisions because
    tiny-gpt2 doesn't produce single-token SYMBOLIC/LLM labels."""
    model, tokenizer = tiny_model
    prompt = "Compute 12 * 34. Return only the integer."

    # Batch size 1
    out_solo = _generate_batch(
        [prompt], model, tokenizer, max_new_tokens=5, seed=42, prompt_format="raw"
    )[0]

    # Batch size 3 (prompt is first)
    out_batch = _generate_batch(
        [prompt, "Compute 99 + 1.", "Explain the sky."],
        model, tokenizer, max_new_tokens=5, seed=42, prompt_format="raw"
    )[0]

    # With greedy decoding and the same seed, the output for the first
    # prompt should be identical whether it's alone or in a batch.
    # (Left padding shifts the position but greedy decoding is deterministic
    # given the same KV state. If this fails, it indicates a padding bug.)
    assert out_solo == out_batch, (
        f"batch-size-1 output {out_solo!r} != batched output {out_batch!r}; "
        "this indicates a left-padding or position-encoding bug"
    )


def test_full_stack_symbolic_execution(tmp_path: Path):
    """The full stack: generate tasks → symbolic execution → output.
    This doesn't need a model (symbolic-only) but verifies the CLI wiring."""
    import importlib.util

    tasks = generate_benchmark(5, 42, categories=("easy_add", "multiply"))
    task_path = tmp_path / "tasks.jsonl"
    with task_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task) + "\n")

    output = tmp_path / "out.jsonl"
    routes = tmp_path / "out.routes.jsonl"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_v0_outputs.py"
    spec = importlib.util.spec_from_file_location("gen_test", script)
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

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 5
    assert all("output" in row for row in rows)
    # Symbolic execution should produce FINAL: <integer> for arithmetic tasks.
    assert all(row["output"].startswith("FINAL:") for row in rows)


def test_full_stack_symbolic_fallback_to_llm(tmp_path: Path, tiny_model, monkeypatch):
    """The full stack: unsupported arithmetic → symbolic fallback → LLM.
    This exercises the fallback path with a real (tiny) model."""
    import importlib.util

    # Create a task with true division (unsupported by the symbolic engine).
    task = {
        "task_id": "fallback-test",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 10, "b": 2, "op": "/"},
        "specification": "Compute 10 / 2. Return only the integer.",
        "expected": 5,
        "route_label": "llm",
        "capability_oracle": "symbolic",
        "accuracy_oracle": "llm",
        "utility_oracle": "llm",
    }
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_v0_outputs.py"
    spec = importlib.util.spec_from_file_location("gen_fallback_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Patch _load_llm to return our tiny model.
    model, tokenizer = tiny_model
    monkeypatch.setattr(mod, "_load_llm", lambda _model_id: (model, tokenizer))

    old_argv = sys.argv
    sys.argv = [
        "generate_v0_outputs.py",
        "--input", str(task_path),
        "--output", str(output),
        "--model", TINY_MODEL_ID,
        "--execution-mode", "auto",
        "--max-new-tokens", "5",
        "--no-manifest",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    # The output should be a string (the LLM generated something).
    assert isinstance(rows[0]["output"], str)
    assert rows[0]["output"]  # non-empty


def test_extract_and_load_vector_roundtrip_real_model(tiny_model, tmp_path: Path):
    """End-to-end: capture activations → extract direction → save → load →
    validate against the same model. This is the provenance chain from
    §21 of the audit."""
    model, tokenizer = tiny_model
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd

    # Capture two dummy activation sets (we don't need meaningful directions
    # for tiny-gpt2; we just need to verify the wiring).
    import numpy as np

    pos = np.random.randn(5, hidden).astype(np.float32)
    neg = np.random.randn(5, hidden).astype(np.float32)

    spec = SteeringSpec(
        vector_id="integration-v1",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=0,
        alpha=1.0,
        anchor="ACTION",
        model_id=TINY_MODEL_ID,
        hidden_size=hidden,
        extraction_method="contrastive_mean_difference",
        normalization="l2",
        positive_n=5,
        negative_n=5,
        capture_anchor="ACTION:",
        capture_prompt_format="raw",
        capture_dataset_path="dummy",
        capture_dataset_sha256="a" * 64,
    )
    vec = contrastive_mean_direction(pos, neg, spec)
    path = tmp_path / "v.npz"
    save_vector(vec, path)

    loaded = load_vector(path)
    assert loaded.spec.vector_id == "integration-v1"
    assert loaded.spec.capture_dataset_sha256 == "a" * 64
    assert loaded.spec.extraction_method == "contrastive_mean_difference"

    # Validate against the real model.
    validate_vector_for_model(model, loaded)


def test_chat_template_detection(tiny_model):
    """tiny-gpt2 has no chat template, so _format_for_model must raise
    when chat format is requested. This verifies the chat-template guard."""
    _, tokenizer = tiny_model
    with pytest.raises(ValueError, match="chat template"):
        _format_for_model("test", tokenizer, "chat")


def test_raw_format_passthrough(tiny_model):
    """Raw format must return the prompt unchanged."""
    _, tokenizer = tiny_model
    assert _format_for_model("test prompt", tokenizer, "raw") == "test prompt"
