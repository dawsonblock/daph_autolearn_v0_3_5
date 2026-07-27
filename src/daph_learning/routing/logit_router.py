from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal, Sequence

from daph_learning.routing.steered_router import (
    RouteAction,
    resolve_route_token_ids,
    resolve_route_token_ids_contextual,
    route_action_from_logits,
)
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import (
    multi_layer_residual_addition_hook,
    residual_addition_hook,
)

TokenResolver = Literal["isolated", "contextual"]


def _tokenize_route_batch(
    rendered_prompts: Sequence[str],
    tokenizer: Any,
    *,
    anchor: str,
):
    """Tokenize a left-padded batch and map per-example anchor positions."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id for batching")
        tokenizer.pad_token = tokenizer.eos_token

    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        unpadded_anchor_indices = [
            find_anchor_token_index(prompt, tokenizer, anchor)
            for prompt in rendered_prompts
        ]
        individual_lengths = [
            len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            for prompt in rendered_prompts
        ]
        encoded = tokenizer(
            list(rendered_prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = previous_padding_side

    padded_len = int(encoded["input_ids"].shape[1])
    padded_anchor_indices = [
        anchor_idx + (padded_len - length)
        for anchor_idx, length in zip(unpadded_anchor_indices, individual_lengths)
    ]
    return encoded, padded_anchor_indices


def score_route_batch_from_logits(
    rendered_prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    *,
    vector: Any | None = None,
    alpha: float | None = None,
    vectors: Sequence[Any] | None = None,
    alphas: Sequence[float] | None = None,
    anchor: str = "ACTION:",
    threshold: float = 0.0,
    leading_space: bool | None = None,
    token_resolver: TokenResolver = "isolated",
    device: Any | None = None,
    telemetry_sink: list | None = None,
) -> list[tuple[RouteAction, float]]:
    """Score route decisions in one forward pass per batch.

    Supports either a single steering vector or a multi-layer composite vector
    list. Candidate route labels must each be one tokenizer token; callers can
    catch `ValueError` and fall back to autoregressive generation when this is
    not true for a model/tokenizer.

    ``token_resolver`` selects how route-label token IDs are derived:

    - ``"isolated"`` (default, v0.3.4 behavior): tokenizes labels via
      ``tokenizer.encode(label)`` in isolation. Fast, but not guaranteed
      to match the actual continuation token for all BPE/SentencePiece
      tokenizers. See CLAIMS.md §9.
    - ``"contextual"``: derives the continuation token by diffing
      ``T(rendered_prompt + label)`` against ``T(rendered_prompt)`` for
      the first prompt in the batch. Eliminates the boundary assumption.
      Slightly slower (one extra tokenization per label) and requires
      that all prompts in the batch share the same continuation token
      (which is true when they share the same routing-prompt template).
    """
    import torch

    if vector is not None and vectors is not None:
        raise ValueError("provide either vector or vectors, not both")
    if vectors is not None:
        vectors = list(vectors)
        if not vectors:
            raise ValueError("vectors must be non-empty")
        if alphas is None:
            alphas = [float(v.spec.alpha) for v in vectors]
        if len(alphas) != len(vectors):
            raise ValueError("alphas length must match vectors length")
    elif vector is not None:
        alpha = float(vector.spec.alpha if alpha is None else alpha)

    if token_resolver == "contextual":
        if not rendered_prompts:
            raise ValueError("contextual resolver requires at least one rendered prompt")
        sym_id, llm_id = resolve_route_token_ids_contextual(
            tokenizer,
            rendered_prompts[0],
            leading_space=leading_space,
        )
    elif token_resolver == "isolated":
        sym_id, llm_id = resolve_route_token_ids(
            tokenizer,
            leading_space=leading_space,
        )
    else:
        raise ValueError(
            f"token_resolver must be 'isolated' or 'contextual', got {token_resolver!r}"
        )

    encoded, target_indices = _tokenize_route_batch(
        rendered_prompts,
        tokenizer,
        anchor=anchor,
    )

    if device is None:
        device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    if vectors is not None:
        configs = [
            {
                "layer_index": vec.spec.layer,
                "vector": vec.values,
                "alpha": float(scale),
                "token_scope": "last",
                "target_token_index": target_indices,
            }
            for vec, scale in zip(vectors, alphas)
        ]
        ctx = multi_layer_residual_addition_hook(model, configs, telemetry_sink=telemetry_sink)
    elif vector is not None:
        ctx = residual_addition_hook(
            model,
            layer_index=vector.spec.layer,
            vector=vector.values,
            alpha=float(alpha),
            token_scope="last",
            target_token_index=target_indices,
            telemetry_sink=telemetry_sink,
        )
    else:
        ctx = nullcontext()

    with torch.inference_mode(), ctx:
        outputs = model(**encoded)
        # Left padding guarantees the final sequence position is the actual
        # prompt end for every example, not a padding position.
        last_logits = outputs.logits[:, -1, :]

    return [
        route_action_from_logits(
            row,
            sym_id,
            llm_id,
            threshold=threshold,
        )
        for row in last_logits
    ]
