from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal, Sequence

import numpy as np

TokenScope = Literal["all", "last"]


def resolve_transformer_layers(model):
    """Resolve common Hugging Face decoder-layer containers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise TypeError("unsupported model layout; provide a model-specific layer resolver")


def validate_vector_for_model(model, vector) -> None:
    layers = resolve_transformer_layers(model)
    if vector.spec.layer < 0 or vector.spec.layer >= len(layers):
        raise IndexError(
            f"steering layer {vector.spec.layer} outside [0, {len(layers)})"
        )
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is not None and int(hidden_size) != vector.values.size:
        raise ValueError(
            f"steering vector dim {vector.values.size} != model hidden size {hidden_size}"
        )
    if vector.spec.hidden_size is not None and vector.spec.hidden_size != vector.values.size:
        raise ValueError("steering-vector metadata hidden_size is inconsistent")
    actual_model_id = getattr(model.config, "_name_or_path", None)
    if vector.spec.model_id and actual_model_id and vector.spec.model_id != actual_model_id:
        raise ValueError(
            f"steering vector was extracted for {vector.spec.model_id!r}, "
            f"but loaded model reports {actual_model_id!r}; omit model_id only for an intentional transfer experiment"
        )


@contextmanager
def residual_addition_hook(
    model,
    *,
    layer_index: int,
    vector: np.ndarray,
    alpha: float,
    token_scope: TokenScope = "all",
    target_token_index: int | Sequence[int] = -1,
    telemetry_sink: list | None = None,
) -> Iterator[None]:
    """Add a rank-1 steering direction to one decoder-layer residual output.

    `last` is a one-shot prompt-bound intervention: it modifies the final token
    of the first multi-token forward pass only. It does not re-apply on cached
    single-token decode steps, and the one-shot guard also prevents repeated
    injection when generation runs without KV caching. `all` is retained for
    broad reasoning-policy experiments.

    If ``telemetry_sink`` is provided, the hook appends a dict of activation
    statistics for each forward pass where steering is actually applied:

    - ``layer_index``: the layer this telemetry came from
    - ``h_norm_mean``: mean L2 norm of the pre-steering hidden state across
      the steered token positions
    - ``av_norm_mean``: mean L2 norm of ``alpha * vector`` (constant per call)
    - ``relative_perturbation``: ``av_norm_mean / h_norm_mean``
    - ``cosine_shift_mean``: mean ``1 - cos(h, h + alpha*v)`` across steered
      positions (0 = no directional change, 2 = full reversal)

    These statistics let operators detect pathological steering (e.g. a
    vector that overwhelms the residual stream) without inspecting raw
    activations. See CLAIMS.md §20.
    """
    import torch

    layers = resolve_transformer_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"layer_index {layer_index} outside [0, {len(layers)})")
    if token_scope not in {"all", "last"}:
        raise ValueError(f"unknown token_scope: {token_scope}")

    vec = torch.as_tensor(vector, dtype=torch.float32)
    last_scope_applied = False

    def _record_telemetry(hidden_before, hidden_after, positions_mask):
        """Compute and append telemetry stats for the steered positions."""
        if telemetry_sink is None or positions_mask is None:
            return
        # hidden_before / hidden_after: [batch, seq, hidden]
        # positions_mask: [batch, seq] bool, True where steering was applied
        import torch as _torch
        steered_before = hidden_before[positions_mask]  # [N, hidden]
        steered_after = hidden_after[positions_mask]  # [N, hidden]
        if steered_before.numel() == 0:
            return
        h_norms = steered_before.norm(dim=-1)  # [N]
        av = steered_after - steered_before  # [N, hidden]
        av_norms = av.norm(dim=-1)  # [N]
        h_norm_mean = float(h_norms.mean())
        av_norm_mean = float(av_norms.mean())
        relative = av_norm_mean / h_norm_mean if h_norm_mean > 0 else float("inf")
        # cos(h, h+av) per position
        cos_sim = _torch.nn.functional.cosine_similarity(
            steered_before, steered_after, dim=-1
        )  # [N]
        cosine_shift = float((1.0 - cos_sim).mean())
        telemetry_sink.append({
            "layer_index": layer_index,
            "n_steered_positions": int(steered_before.shape[0]),
            "h_norm_mean": h_norm_mean,
            "av_norm_mean": av_norm_mean,
            "relative_perturbation": relative,
            "cosine_shift_mean": cosine_shift,
        })

    def hook(_module, _inputs, output):
        nonlocal last_scope_applied
        hidden = output[0] if isinstance(output, tuple) else output
        if not hasattr(hidden, "shape") or hidden.ndim < 2:
            raise TypeError("decoder layer hook did not receive a tensor-like hidden state")
        v = vec.to(device=hidden.device, dtype=hidden.dtype)
        if v.numel() != hidden.shape[-1]:
            raise ValueError(
                f"steering vector dim {v.numel()} != hidden size {hidden.shape[-1]}"
            )
        if token_scope == "all":
            steered = hidden + alpha * v
            if telemetry_sink is not None:
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                _record_telemetry(hidden.float(), steered.float(), mask)
        else:
            steered = hidden
            seq_len = hidden.shape[-2]
            positions_mask = None
            if seq_len > 1 and not last_scope_applied:
                steered = hidden.clone()
                positions_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                if isinstance(target_token_index, int):
                    if not (-seq_len <= target_token_index < seq_len):
                        raise IndexError(
                            f"target_token_index {target_token_index} outside "
                            f"[-{seq_len}, {seq_len}) for prompt sequence"
                        )
                    steered[..., target_token_index, :] = (
                        steered[..., target_token_index, :] + alpha * v
                    )
                    positions_mask[..., target_token_index] = True
                else:
                    indices = list(target_token_index)
                    batch_size = hidden.shape[0]
                    if len(indices) != batch_size:
                        raise ValueError(
                            f"target_token_index has {len(indices)} entries for batch size {batch_size}"
                        )
                    for batch_index, token_index in enumerate(indices):
                        if not (-seq_len <= token_index < seq_len):
                            raise IndexError(
                                f"target token {token_index} for batch item {batch_index} "
                                f"outside [-{seq_len}, {seq_len})"
                            )
                        steered[batch_index, token_index, :] = (
                            steered[batch_index, token_index, :] + alpha * v
                        )
                        positions_mask[batch_index, token_index] = True
                last_scope_applied = True
                if telemetry_sink is not None:
                    _record_telemetry(hidden.float(), steered.float(), positions_mask)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_layer_output(
    model,
    *,
    layer_index: int,
    sink: list,
    last_token_only: bool = False,
    target_token_index: int | None = None,
):
    layers = resolve_transformer_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"layer_index {layer_index} outside [0, {len(layers)})")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if last_token_only and target_token_index is not None:
            raise ValueError("use either last_token_only or target_token_index, not both")
        if target_token_index is not None:
            seq_len = hidden.shape[-2]
            if not (-seq_len <= target_token_index < seq_len):
                raise IndexError(
                    f"target_token_index {target_token_index} outside "
                    f"[-{seq_len}, {seq_len})"
                )
            captured = hidden[..., target_token_index, :]
        else:
            captured = hidden[..., -1, :] if last_token_only else hidden
        sink.append(captured.detach().float().cpu())

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()



@contextmanager
def multi_layer_residual_addition_hook(
    model,
    hooks_config: Sequence[dict[str, Any]],
    *,
    telemetry_sink: list | None = None,
) -> Iterator[None]:
    """Attach multiple residual steering interventions simultaneously.

    Each configuration supports:
      `layer_index`, `vector`, `alpha`, `token_scope`, `target_token_index`.

    `target_token_index` may be a scalar token index or one token index per
    batch item, which is required for anchor-aligned batched prompt steering.

    If ``telemetry_sink`` is provided, per-layer activation statistics are
    appended to it on each forward pass. See ``residual_addition_hook`` for
    the recorded fields.
    """
    from contextlib import ExitStack

    if not hooks_config:
        yield
        return

    seen_layers: set[int] = set()
    with ExitStack() as stack:
        for cfg in hooks_config:
            if "layer_index" not in cfg or "vector" not in cfg or "alpha" not in cfg:
                raise ValueError("each composite hook requires layer_index, vector, and alpha")
            layer_index = int(cfg["layer_index"])
            # Multiple additive hooks on the same layer are legal mathematically
            # but order-sensitive with one-shot mutation; reject ambiguity.
            if layer_index in seen_layers:
                raise ValueError(f"duplicate composite steering layer: {layer_index}")
            seen_layers.add(layer_index)
            stack.enter_context(
                residual_addition_hook(
                    model,
                    layer_index=layer_index,
                    vector=cfg["vector"],
                    alpha=float(cfg["alpha"]),
                    token_scope=cfg.get("token_scope", "all"),
                    target_token_index=cfg.get("target_token_index", -1),
                    telemetry_sink=telemetry_sink,
                )
            )
        yield
