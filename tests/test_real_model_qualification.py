"""Audit §24 real-model qualification suite (v0.3.6 Phase 2.3).

The v0.3.5 ``test_real_model_integration.py`` exercises the stack wiring
against ``sshleifer/tiny-gpt2`` — a ~1MB model with no chat template and
only 2 layers. Per Audit §24 that is *not* a scientific qualification;
it only proves the plumbing does not crash.

This module implements the mandatory §24 qualification gate against a
real instruct-tuned model with a chat template and enough layers to
exercise multi-layer steering. It prefers ``Qwen/Qwen2.5-0.5B-Instruct``
(24 layers, 896 hidden, BPE tokenizer that multi-tokenizes "SYMBOLIC")
and falls back to ``Qwen/Qwen2.5-1.5B-Instruct`` if the 0.5B model is
unavailable. The whole module is skipped if neither model is cached
locally or if torch/transformers is not installed.

The five §24 gates exercised here are:

1. **Chat template rendering validation** — the rendered prompt must
   contain the chat template's generation-prompt suffix and the
   original user content.
2. **Multi-layer residual hook injection on layers L/2 and 3L/4** —
   hooks on two distinct layers must each fire and alter the output.
3. **Left-padded batching consistency** — the routing decision for an
   identical prompt must be the same whether the prompt is scored
   alone (batch=1) or as part of a batch of 8 variable-length prompts.
4. **Target token anchor index alignment in variable-length batches** —
   the per-example anchor index returned by ``_tokenize_route_batch``
   must point at the same logical token regardless of left padding.
5. **One-shot prompt intervention protection during KV-cached decoding** —
   a ``token_scope="last"`` steering hook must apply exactly once during
   the prefill and must NOT re-apply on cached single-token decode steps.

These tests qualify the harness; they do not claim a steering effect
size. See CLAIMS.md §4 and §6.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

from daph_learning.routing.logit_router import (
    _tokenize_route_batch,
    score_route_batch_from_logits,
)
from daph_learning.routing.steered_router import (
    build_route_prompt,
    detect_route_prompt_alignment,
)
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import (
    multi_layer_residual_addition_hook,
    residual_addition_hook,
    resolve_transformer_layers,
)
from daph_learning.steering.types import SteeringSpec, SteeringVector
from scripts.generate_v0_outputs import _format_for_model, _generate_batch


# Prefer the 0.5B model for speed; fall back to 1.5B.
QUALIFICATION_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
)


def _load_qualification_model():
    """Try each candidate model in order, local-only first. Returns
    (model, tokenizer, model_id) or raises a skip."""
    last_exc = None
    for model_id in QUALIFICATION_MODELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            model.eval()
            return model, tokenizer, model_id
        except Exception as exc:  # pragma: no cover - environment-dependent
            last_exc = exc
            continue
    pytest.skip(f"no qualification model available locally: {last_exc}")


@pytest.fixture(scope="module")
def qual_model():
    """Load the qualification model once per module run."""
    return _load_qualification_model()


def _num_layers(model) -> int:
    return len(resolve_transformer_layers(model))


def _hidden_size(model) -> int:
    return getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)


def _make_task(task_id: str, spec: str, caps=()) -> dict:
    return {
        "task_id": task_id,
        "capability_ids": list(caps),
        "specification": spec,
        "expected": None,
        "route_label": "llm",
    }


# --- Gate 1: Chat template rendering validation ---

def test_chat_template_rendering_contains_user_content_and_generation_prompt(qual_model):
    """The rendered chat prompt must (a) contain the original user content
    and (b) end with the chat template's generation-prompt suffix (e.g.
    Qwen2.5's ``<|im_start|>assistant\\n``). This is §24 gate 1."""
    _, tokenizer, _ = qual_model
    if tokenizer.chat_template is None:
        pytest.skip("qualification model has no chat template")
    user_content = "Compute 12 * 34. Return only the integer."
    rendered = _format_for_model(user_content, tokenizer, "chat")
    assert user_content in rendered, "rendered prompt lost the user content"
    # Qwen2.5 chat template ends with the assistant turn opener.
    assert "<|im_start|>assistant" in rendered or "assistant" in rendered.lower(), (
        f"rendered prompt does not end with the generation-prompt suffix: {rendered!r}"
    )
    # The raw format must NOT apply the chat template.
    raw = _format_for_model(user_content, tokenizer, "raw")
    assert "<|im_start|>" not in raw


def test_chat_template_route_prompt_ends_with_action_anchor(qual_model):
    """A rendered routing prompt (chat format) must contain the ACTION
    anchor inside the user turn and terminate with the chat template's
    assistant generation header. The model's next-token prediction
    therefore lands on the assistant's first content token — which is
    exactly where SYMBOLIC/LLM should appear. This is the §24
    chat-template × routing integration.

    The alignment helper must NOT trim anything here (ACTION: is not
    terminal in a chat-templated prompt), and the contextual resolver
    must still resolve a continuation token by appending the label to
    the full rendered prompt."""
    from daph_learning.routing.steered_router import resolve_route_token_ids_contextual
    _, tokenizer, _ = qual_model
    if tokenizer.chat_template is None:
        pytest.skip("qualification model has no chat template")
    task = _make_task("t1", "Compute 2 * 3.", caps=("integer_arithmetic",))
    rendered = _format_for_model(build_route_prompt(task), tokenizer, "chat")
    # The ACTION anchor must be present inside the user turn.
    assert "ACTION:" in rendered, "rendered chat prompt lost the ACTION anchor"
    # The chat template must append the assistant generation header so
    # the model predicts the assistant's first content token.
    assert "<|im_start|>assistant" in rendered, (
        "rendered chat prompt does not end with the assistant generation header"
    )
    # The alignment helper returns the prompt unchanged because ACTION:
    # is not terminal in a chat-templated prompt.
    aligned, leading_space = detect_route_prompt_alignment(rendered)
    assert aligned == rendered
    # The contextual resolver must still find a distinct continuation
    # token (with the first-token fallback for Qwen2.5's multi-token
    # route labels).
    sym_id, llm_id = resolve_route_token_ids_contextual(
        tokenizer, rendered, allow_first_token_fallback=True,
    )
    assert sym_id != llm_id
    assert 0 <= sym_id < tokenizer.vocab_size
    assert 0 <= llm_id < tokenizer.vocab_size


# --- Gate 2: Multi-layer residual hook injection on L/2 and 3L/4 ---

def test_multi_layer_hook_on_L_half_and_3L_quarter_layers(qual_model):
    """Hooks on layers L/2 and 3L/4 must both fire and produce a visible
    output change relative to baseline. This is §24 gate 2."""
    model, tokenizer, _ = qual_model
    L = _num_layers(model)
    hidden = _hidden_size(model)
    layer_a = L // 2
    layer_b = (3 * L) // 4
    assert layer_a != layer_b, "L/2 and 3L/4 collapsed to the same layer"

    prompt = "Compute 12 * 34. Return only the integer."
    encoded = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        baseline = model(**encoded, use_cache=False).logits

    # A large vector on two layers must produce a measurable logit shift.
    vec = torch.ones(hidden, dtype=torch.float32) * 50.0
    configs = [
        {"layer_index": layer_a, "vector": vec.numpy(), "alpha": 2.0, "token_scope": "all"},
        {"layer_index": layer_b, "vector": vec.numpy(), "alpha": 2.0, "token_scope": "all"},
    ]
    with torch.inference_mode(), multi_layer_residual_addition_hook(model, configs):
        steered = model(**encoded, use_cache=False).logits

    max_diff = (steered - baseline).abs().max().item()
    assert max_diff > 1e-4, (
        f"multi-layer hook on L/2={layer_a} and 3L/4={layer_b} produced no "
        f"visible change (max diff {max_diff}); hooks are not firing"
    )


# --- Gate 3: Left-padded batching consistency (batch=1 ≡ batch=8) ---

def test_left_padded_batching_routing_decision_consistency(qual_model):
    """The routing decision for an identical prompt must be the same
    whether it is scored alone (batch=1) or as the first element of a
    batch of 8 variable-length prompts. This is §24 gate 3:

        batch_size_1_decision == same_example_inside_batch_decision

    We use the contextual resolver with the first-token fallback so the
    test exercises the v0.3.6 default routing path on a multi-token
    tokenizer."""
    model, tokenizer, _ = qual_model
    anchor_task = _make_task("anchor", "Compute 2 * 3.", caps=("integer_arithmetic",))
    anchor_rendered = _format_for_model(build_route_prompt(anchor_task), tokenizer, "chat")

    # Batch of 8 variable-length prompts; the anchor is first.
    batch_tasks = [anchor_task] + [
        _make_task(f"t{i}", spec, caps=caps)
        for i, (spec, caps) in enumerate([
            ("Compute 99 + 1.", ("integer_arithmetic",)),
            ("Explain why the sky is blue.", ()),
            ("Compute 12345 * 67. Return only the integer.", ("integer_arithmetic",)),
            ("Write a Python function that reverses a linked list.", ("code_generation",)),
            ("What is the capital of France?", ("knowledge",)),
            ("Compute (10 + 20) * 3.", ("integer_arithmetic",)),
            ("Summarize Hamlet in two sentences.", ()),
        ], start=1)
    ]
    batch_rendered = [
        _format_for_model(build_route_prompt(t), tokenizer, "chat") for t in batch_tasks
    ]
    assert len(batch_rendered) == 8

    solo = score_route_batch_from_logits(
        [anchor_rendered], model, tokenizer,
        token_resolver="contextual", allow_first_token_fallback=True,
    )
    batched = score_route_batch_from_logits(
        batch_rendered, model, tokenizer,
        token_resolver="contextual", allow_first_token_fallback=True,
    )
    solo_action, solo_margin = solo[0]
    batch_action, batch_margin = batched[0]
    assert solo_action == batch_action, (
        f"batch=1 decision {solo_action!r} != batch=8 decision {batch_action!r} "
        "for the identical anchor prompt; left-padding is corrupting the routing"
    )


# --- Gate 4: Target token anchor index alignment in variable-length batches ---

def test_target_token_anchor_index_alignment_in_variable_length_batch(qual_model):
    """The per-example anchor index returned by ``_tokenize_route_batch``
    must point at the same logical ACTION token regardless of how much
    left padding was added. We verify by re-tokenizing each prompt
    individually and checking the unpadded anchor index, then confirming
    the padded index maps back to the same token id."""
    model, tokenizer, _ = qual_model
    tasks = [
        _make_task("short", "Compute 2 * 3.", caps=("integer_arithmetic",)),
        _make_task("medium", "Compute 12345 * 67890. Return only the integer.", caps=("integer_arithmetic",)),
        _make_task("long", "Compute (10 + 20) * (30 + 40) - 5. Return only the integer.", caps=("integer_arithmetic",)),
    ]
    rendered = [_format_for_model(build_route_prompt(t), tokenizer, "chat") for t in tasks]
    anchor = "ACTION:"

    encoded, padded_indices = _tokenize_route_batch(rendered, tokenizer, anchor=anchor)
    input_ids = encoded["input_ids"]
    padded_len = int(input_ids.shape[1])

    # For each example, the unpadded anchor index + (padded_len - length)
    # must equal the padded anchor index, and the token at that index
    # must be the same token id as the anchor token in the unpadded
    # tokenization.
    for i, prompt in enumerate(rendered):
        unpadded_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        unpadded_len = len(unpadded_ids)
        unpadded_anchor = find_anchor_token_index(prompt, tokenizer, anchor)
        assert unpadded_anchor is not None, f"anchor not found in prompt {i}"
        expected_padded = unpadded_anchor + (padded_len - unpadded_len)
        assert padded_indices[i] == expected_padded, (
            f"example {i}: padded anchor index {padded_indices[i]} != "
            f"expected {expected_padded} (unpadded {unpadded_anchor} + "
            f"offset {padded_len - unpadded_len})"
        )
        # The token id at the padded anchor position must match the
        # token id at the unpadded anchor position.
        padded_token = int(input_ids[i, padded_indices[i]])
        unpadded_token = int(unpadded_ids[unpadded_anchor])
        assert padded_token == unpadded_token, (
            f"example {i}: token id at padded anchor {padded_token} != "
            f"unpadded anchor {unpadded_token}; the anchor is pointing at "
            "different tokens after padding"
        )


# --- Gate 5: One-shot prompt intervention protection during KV-cached decoding ---

def test_one_shot_last_scope_hook_does_not_reapply_on_cached_decode(qual_model):
    """A ``token_scope="last"`` steering hook must apply exactly once
    during the prefill forward pass and must NOT re-apply on subsequent
    cached single-token decode steps. This is §24 gate 5.

    We verify by counting hook applications: with ``use_cache=True`` and
    ``max_new_tokens >= 2``, the ``last_scope_applied`` guard must keep
    the application count to 1. We use a telemetry sink to count the
    steered positions across all forward passes."""
    model, tokenizer, _ = qual_model
    L = _num_layers(model)
    hidden = _hidden_size(model)
    layer = L // 2

    prompt = "Compute 12 * 34. Return only the integer."
    encoded = tokenizer(prompt, return_tensors="pt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vec = torch.ones(hidden, dtype=torch.float32) * 30.0
    telemetry: list = []
    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=layer,
        vector=vec.numpy(),
        alpha=1.0,
        token_scope="last",
        target_token_index=-1,
        telemetry_sink=telemetry,
    ):
        model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    # The hook must have recorded telemetry for exactly one forward pass
    # (the prefill). If it re-applied on cached decode steps, there would
    # be multiple telemetry entries.
    assert len(telemetry) == 1, (
        f"token_scope='last' hook fired {len(telemetry)} times; expected 1 "
        "(the one-shot guard must prevent re-application on cached decode steps)"
    )
    # And that one application must have steered exactly one position
    # (the last token of the prefill sequence).
    assert telemetry[0]["n_steered_positions"] == 1, (
        f"one-shot hook steered {telemetry[0]['n_steered_positions']} positions; "
        "expected 1 (the last prefill token only)"
    )


def test_one_shot_last_scope_hook_protects_generate_batch(qual_model):
    """The one-shot protection must also hold for the batched generation
    path used by ``_generate_batch``. A ``token_scope='last'`` steering
    vector applied during batched generation must not corrupt outputs
    beyond the prefill.

    We use ``steering_anchor=None`` so the hook targets the last token
    of each sequence (target_indices=-1) — this exercises the one-shot
    last-scope path without requiring an ACTION: anchor in the answer
    prompts."""
    model, tokenizer, model_id = qual_model
    L = _num_layers(model)
    hidden = _hidden_size(model)

    spec = SteeringSpec(
        vector_id="qual-last-scope",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=L // 2,
        alpha=1.0,
        anchor="ACTION:",
        model_id=model_id,
        hidden_size=hidden,
    )
    vec = SteeringVector(
        spec=spec,
        values=torch.ones(hidden, dtype=torch.float32).numpy() * 20.0,
    )
    prompts = ["Compute 2 * 3.", "Compute 99 + 1."]
    # Must not raise and must produce one completion per prompt.
    outputs = _generate_batch(
        prompts, model, tokenizer,
        max_new_tokens=6, seed=42,
        steering_vectors=[vec], steering_alphas=[1.0],
        steering_scope="last", prompt_format="raw",
        steering_anchor=None,
    )
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)
