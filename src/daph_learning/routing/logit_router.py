from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal, Sequence

from daph_learning.routing.steered_router import (
    RouteAction,
    detect_route_prompt_alignment,
    resolve_route_label_token_sequences,
    resolve_route_token_ids,
    resolve_route_token_ids_contextual,
    route_action_from_logits,
    route_action_from_sequence_scores,
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
    allow_first_token_fallback: bool = False,
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

    ``allow_first_token_fallback`` (v0.3.6): when ``True``, multi-token
    route labels (e.g. Qwen2.5's ``"SYMBOLIC"`` → ``[" SY", "MBOL", "IC"]``)
    are reduced to their first continuation token, enabling a first-token
    logit contrast instead of forcing a fall-through to autoregressive
    generation. This is the default routing path for the AutoLearn loop on
    multi-token tokenizers; see CLAIMS.md §9 and the v0.3.6 repair plan
    Phase 1.1.
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
        # v0.3.6 Phase 2.2: align the prompt terminal boundary so the
        # contextual resolver sees a canonical "ACTION:" + " LABEL" form.
        # When the caller did not force a leading_space hint, derive it
        # from the rendered prompt's terminal whitespace.
        aligned_prompt = rendered_prompts[0]
        effective_leading_space = leading_space
        if leading_space is None:
            aligned_prompt, effective_leading_space = detect_route_prompt_alignment(
                rendered_prompts[0]
            )
        sym_id, llm_id = resolve_route_token_ids_contextual(
            tokenizer,
            aligned_prompt,
            leading_space=effective_leading_space,
            allow_first_token_fallback=allow_first_token_fallback,
        )
    elif token_resolver == "isolated":
        sym_id, llm_id = resolve_route_token_ids(
            tokenizer,
            leading_space=leading_space,
            allow_first_token_fallback=allow_first_token_fallback,
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


def _score_label_sequence_batch(
    rendered_prompts: Sequence[str],
    label_tail_ids: list[int],
    model: Any,
    tokenizer: Any,
    *,
    anchor: str,
    vector: Any | None = None,
    alpha: float | None = None,
    vectors: Sequence[Any] | None = None,
    alphas: Sequence[float] | None = None,
    device: Any | None = None,
    telemetry_sink: list | None = None,
) -> torch.Tensor:
    """Run one forward pass per prompt with the label tokens appended and
    return ``Score(Label | x) = sum_j log P(t_j | x, t_<j)`` per example.

    Returns a ``[batch]`` tensor of sequence log-probabilities. The label
    tail token IDs are assumed identical across the batch (true when all
    prompts share the routing-prompt template, which the contextual
    sequence resolver guarantees).
    """
    import torch
    import torch.nn.functional as F

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id for batching")
        tokenizer.pad_token = tokenizer.eos_token

    label_len = len(label_tail_ids)
    if label_len < 1:
        raise ValueError("label_tail_ids must be non-empty")

    # Per-prompt: tokenize the prompt alone, then append the shared label tail.
    prompt_ids_list: list[list[int]] = [
        tokenizer.encode(prompt, add_special_tokens=True) for prompt in rendered_prompts
    ]
    full_ids_list: list[list[int]] = [pids + list(label_tail_ids) for pids in prompt_ids_list]
    prompt_lens: list[int] = [len(pids) for pids in prompt_ids_list]

    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        max_len = max(len(ids) for ids in full_ids_list)
        pad_id = tokenizer.pad_token_id
        padded = torch.full((len(full_ids_list), max_len), pad_id, dtype=torch.long)
        attention = torch.zeros((len(full_ids_list), max_len), dtype=torch.long)
        for i, ids in enumerate(full_ids_list):
            offset = max_len - len(ids)
            padded[i, offset:] = torch.tensor(ids, dtype=torch.long)
            attention[i, offset:] = 1
    finally:
        tokenizer.padding_side = previous_padding_side

    if device is None:
        device = model.get_input_embeddings().weight.device
    padded = padded.to(device)
    attention = attention.to(device)

    # Steering target: the anchor token inside the prompt portion. With left
    # padding the anchor token shifts by (max_len - prompt_len). We compute
    # per-example anchor positions so the one-shot "last" scope injects at
    # the right place. For "all" scope the whole sequence is steered so the
    # target index is irrelevant.
    steering_scope = "last"
    target_indices: int | list[int] = -1
    if (vector is not None or vectors is not None) and steering_scope == "last":
        # Recompute anchor positions in the padded frame.
        unpadded_anchor_indices = [
            find_anchor_token_index(prompt, tokenizer, anchor)
            for prompt in rendered_prompts
        ]
        target_indices = [
            anchor_idx + (max_len - prompt_len)
            for anchor_idx, prompt_len in zip(unpadded_anchor_indices, prompt_lens)
        ]

    if vectors is not None:
        configs = [
            {
                "layer_index": vec.spec.layer,
                "vector": vec.values,
                "alpha": float(scale),
                "token_scope": steering_scope,
                "target_token_index": target_indices,
            }
            for vec, scale in zip(vectors, alphas or [float(v.spec.alpha) for v in vectors])
        ]
        ctx = multi_layer_residual_addition_hook(model, configs, telemetry_sink=telemetry_sink)
    elif vector is not None:
        ctx = residual_addition_hook(
            model,
            layer_index=vector.spec.layer,
            vector=vector.values,
            alpha=float(alpha if alpha is not None else vector.spec.alpha),
            token_scope=steering_scope,
            target_token_index=target_indices,
            telemetry_sink=telemetry_sink,
        )
    else:
        ctx = nullcontext()

    with torch.inference_mode(), ctx:
        outputs = model(input_ids=padded, attention_mask=attention)
        logits = outputs.logits  # [batch, seq, vocab]

    # log P(t_j | x, t_<j): the logit at position i predicts token at i+1.
    # Label token j is at absolute position (max_len - label_len + j).
    # Its predicting logit is at position (max_len - label_len + j - 1).
    scores = torch.zeros(len(rendered_prompts), device=device)
    for j in range(label_len):
        token_id = int(label_tail_ids[j])
        predict_pos = max_len - label_len + j - 1
        row_logits = logits[:, predict_pos, :]  # [batch, vocab]
        log_probs = F.log_softmax(row_logits.float(), dim=-1)
        scores = scores + log_probs[:, token_id]
    return scores


def score_route_batch_sequence_from_logits(
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
    device: Any | None = None,
    telemetry_sink: list | None = None,
) -> list[tuple[RouteAction, float]]:
    """v0.3.8 DEF-01: full-sequence logit route scoring.

    Computes ``Score(Label | x) = sum_j log P(t_j | x, t_<j)`` over the
    complete label token sequence for each candidate label, then returns
    ``(action, margin)`` where ``margin = Score(SYMBOLIC|x) - Score(LLM|x)``.

    This eliminates the single-token contrast failure on multi-token
    BPE/SentencePiece tokenizers (e.g. Qwen2.5 ``"SYMBOLIC"`` →
    ``[" SY", "MBOL", "IC"]``). The label continuation token sequences are
    resolved contextually from the first rendered prompt (all prompts must
    share the routing-prompt template, which is the existing batching
    contract).

    Two forward passes are issued (one per label) regardless of label token
    count, so the cost is ~2x the single-token path — no autoregressive
    generation fallback is required.
    """
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

    if not rendered_prompts:
        raise ValueError("sequence scoring requires at least one rendered prompt")

    # v0.3.6 Phase 2.2: normalize the prompt terminal boundary so the
    # contextual sequence resolver sees a canonical "ACTION:" + " LABEL" form.
    aligned_prompt = rendered_prompts[0]
    effective_leading_space = leading_space
    if leading_space is None:
        aligned_prompt, effective_leading_space = detect_route_prompt_alignment(
            rendered_prompts[0]
        )

    sym_tail, llm_tail, _ = resolve_route_label_token_sequences(
        tokenizer,
        aligned_prompt,
        leading_space=effective_leading_space,
    )

    sym_scores = _score_label_sequence_batch(
        rendered_prompts,
        sym_tail,
        model,
        tokenizer,
        anchor=anchor,
        vector=vector,
        alpha=alpha,
        vectors=vectors,
        alphas=alphas,
        device=device,
        telemetry_sink=telemetry_sink,
    )
    llm_scores = _score_label_sequence_batch(
        rendered_prompts,
        llm_tail,
        model,
        tokenizer,
        anchor=anchor,
        vector=vector,
        alpha=alpha,
        vectors=vectors,
        alphas=alphas,
        device=device,
        telemetry_sink=telemetry_sink,
    )

    results: list[tuple[RouteAction, float]] = []
    for s_sym, s_llm in zip(sym_scores.tolist(), llm_scores.tolist()):
        results.append(
            route_action_from_sequence_scores(s_sym, s_llm, threshold=threshold)
        )
    return results
