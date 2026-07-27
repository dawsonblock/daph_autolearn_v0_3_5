from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Sequence

import numpy as np

TokenScope = Literal["all", "last"]
DecaySchedule = Literal["none", "linear", "exponential", "cosine"]


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


def decay_multiplier(
    step: int,
    *,
    schedule: DecaySchedule = "none",
    decay_steps: int = 0,
    min_multiplier: float = 0.0,
) -> float:
    """v0.3.6 Phase 3.2: per-step alpha multiplier for reasoning steering.

    Reasoning-policy steering uses ``token_scope="all"`` so the steering
    vector is applied at every generated token. Without decay the
    intervention persists at full strength across the whole generation,
    which empirically causes degenerate repetition on long outputs (the
    model cannot "escape" the reasoning bias). The decay schedule
    attenuates the effective alpha over generated token positions so the
    intervention is strongest at the start of generation and fades out.

    Parameters
    ----------
    step : int
        The generated-token position (0 = first generated token, i.e.
        the last position of the prefill forward pass).
    schedule : str
        One of:
        - ``"none"``: constant 1.0 (the v0.3.5 behaviour; default).
        - ``"linear"``: linear ramp from 1.0 at step 0 to
          ``min_multiplier`` at step ``decay_steps``, then constant at
          ``min_multiplier``.
        - ``"exponential"``: ``min_multiplier + (1 - min_multiplier) *
          exp(-step / decay_steps)``. Decays smoothly; never reaches
          ``min_multiplier`` exactly but approaches it asymptotically.
        - ``"cosine"``: ``min_multiplier + (1 - min_multiplier) * 0.5 *
          (1 + cos(pi * min(step, decay_steps) / decay_steps))``.
          Smoothest derivative at the endpoints; often the best
          empirical schedule for reasoning steering.
    decay_steps : int
        The number of generated-token positions over which the schedule
        ramps down. Ignored when ``schedule="none"``. Must be ≥ 1 for
        any non-none schedule.
    min_multiplier : float
        The asymptotic / final multiplier. 0.0 means the steering
        switches off entirely after ``decay_steps``; 0.1 means it
        retains 10% strength.

    Returns
    -------
    float
        A multiplier in ``[min_multiplier, 1.0]`` to multiply the
        effective alpha by at this step.

    Raises
    ------
    ValueError
        If ``schedule`` is unknown or ``decay_steps < 1`` for a
        non-none schedule.
    """
    if schedule == "none":
        return 1.0
    if schedule not in ("linear", "exponential", "cosine"):
        raise ValueError(f"unknown decay schedule: {schedule!r}")
    if decay_steps < 1:
        raise ValueError(
            f"decay_steps must be >= 1 for schedule={schedule!r}, got {decay_steps}"
        )
    if min_multiplier < 0 or min_multiplier > 1:
        raise ValueError(f"min_multiplier must be in [0, 1], got {min_multiplier}")
    if step <= 0:
        return 1.0
    span = 1.0 - min_multiplier
    if schedule == "linear":
        if step >= decay_steps:
            return min_multiplier
        return min_multiplier + span * (1.0 - step / decay_steps)
    if schedule == "exponential":
        return min_multiplier + span * math.exp(-step / decay_steps)
    # cosine
    t = min(step, decay_steps) / decay_steps
    return min_multiplier + span * 0.5 * (1.0 + math.cos(math.pi * t))


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
    decay_schedule: DecaySchedule = "none",
    decay_steps: int = 0,
    decay_min_multiplier: float = 0.0,
) -> Iterator[None]:
    """Add a rank-1 steering direction to one decoder-layer residual output.

    `last` is a one-shot prompt-bound intervention: it modifies the final token
    of the first multi-token forward pass only. It does not re-apply on cached
    single-token decode steps, and the one-shot guard also prevents repeated
    injection when generation runs without KV caching. `all` is retained for
    broad reasoning-policy experiments.

    v0.3.6 Phase 3.2: when ``token_scope="all"`` and ``decay_schedule`` is
    not ``"none"``, the effective alpha is multiplied by
    :func:`decay_multiplier` evaluated at the current generated-token
    position. The position counter advances by 1 for each forward pass
    with ``seq_len == 1`` (a cached decode step) and is set to 0 on the
    first multi-token forward pass (the prefill). This attenuates the
    reasoning intervention over generation so it does not pin the model
    into a degenerate loop. ``token_scope="last"`` ignores decay (the
    one-shot guard already limits it to a single application).

    If ``telemetry_sink`` is provided, the hook appends a dict of activation
    statistics for each forward pass where steering is actually applied:

    - ``layer_index``: the layer this telemetry came from
    - ``h_norm_mean``: mean L2 norm of the pre-steering hidden state across
      the steered token positions
    - ``av_norm_mean``: mean L2 norm of ``alpha * vector`` (constant per call)
    - ``relative_perturbation``: ``av_norm_mean / h_norm_mean``
    - ``cosine_shift_mean``: mean ``1 - cos(h, h + alpha*v)`` across steered
      positions (0 = no directional change, 2 = full reversal)
    - ``decay_step`` (v0.3.6): the generated-token position used for the
      decay multiplier (only present when ``token_scope="all"`` and
      ``decay_schedule != "none"``)
    - ``decay_multiplier`` (v0.3.6): the multiplier actually applied
      (only present when decay is active)

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
    # v0.3.6: generated-token position counter for decay scheduling.
    decay_step = -1  # set to 0 on the first multi-token forward pass

    def _record_telemetry(hidden_before, hidden_after, positions_mask, *, extra=None):
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
        record = {
            "layer_index": layer_index,
            "n_steered_positions": int(steered_before.shape[0]),
            "h_norm_mean": h_norm_mean,
            "av_norm_mean": av_norm_mean,
            "relative_perturbation": relative,
            "cosine_shift_mean": cosine_shift,
        }
        if extra:
            record.update(extra)
        telemetry_sink.append(record)

    def hook(_module, _inputs, output):
        nonlocal last_scope_applied, decay_step
        hidden = output[0] if isinstance(output, tuple) else output
        if not hasattr(hidden, "shape") or hidden.ndim < 2:
            raise TypeError("decoder layer hook did not receive a tensor-like hidden state")
        v = vec.to(device=hidden.device, dtype=hidden.dtype)
        if v.numel() != hidden.shape[-1]:
            raise ValueError(
                f"steering vector dim {v.numel()} != hidden size {hidden.shape[-1]}"
            )
        if token_scope == "all":
            # v0.3.6 Phase 3.2: advance the decay counter. The first
            # multi-token forward pass (prefill) is step 0; each
            # subsequent single-token decode step advances by 1.
            seq_len = hidden.shape[-2]
            if seq_len > 1:
                decay_step = 0
            else:
                decay_step += 1
            effective_alpha = alpha
            extra_telemetry: dict | None = None
            if decay_schedule != "none":
                mult = decay_multiplier(
                    decay_step,
                    schedule=decay_schedule,
                    decay_steps=decay_steps,
                    min_multiplier=decay_min_multiplier,
                )
                effective_alpha = alpha * mult
                extra_telemetry = {"decay_step": decay_step, "decay_multiplier": mult}
            steered = hidden + effective_alpha * v
            if telemetry_sink is not None:
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                _record_telemetry(
                    hidden.float(), steered.float(), mask, extra=extra_telemetry,
                )
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

    v0.3.6 Phase 3.2: each config may additionally specify
    ``decay_schedule``, ``decay_steps``, and ``decay_min_multiplier``,
    which are forwarded to :func:`residual_addition_hook`. This lets a
    reasoning-policy vector on layer L/2 decay over generation while a
    tool-policy vector on layer L/4 stays one-shot.

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
                    decay_schedule=cfg.get("decay_schedule", "none"),
                    decay_steps=int(cfg.get("decay_steps", 0)),
                    decay_min_multiplier=float(cfg.get("decay_min_multiplier", 0.0)),
                )
            )
        yield
