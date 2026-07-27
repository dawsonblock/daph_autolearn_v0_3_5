"""Policy-level routing dispatcher (V037-001).

This module introduces the single routing interface used by the AutoLearn
loop and external callers:

    route_tasks(tasks, *, model, tokenizer, steering=None, route_fn=None)
        -> list[RouteDecision]

It replaces the v0.3.6 dead path where ``route_fn`` was accepted by
``run_autolearn_loop()`` but never actually invoked (the loop branched on
``current_vector is not None and route_fn is None`` and otherwise fell through
to a capability heuristic that ignored any custom router).

Dispatch order:

1. ``route_fn is not None``  -> custom user router (one call per task)
2. ``steering is not None``   -> model-mediated steered router
3. else                       -> baseline capability heuristic

The ``RouteDecision`` defined here is the **policy-level** decision record
(``task_id``, ``route``, ``confidence``, ``raw_scores``, ``source``). It is
intentionally distinct from ``daph_learning.routing.arithmetic_router.RouteDecision``,
which is the capability-assessment record produced by the heuristic router and
carries an ``assessment`` payload. The two abstractions serve different layers
of the routing pipeline; consolidating them is deferred to a later release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from daph_learning.routing.errors import (
    InvalidRouteDecisionError,
    ModelRoutingError,
    MultiTokenRouteError,
    RouteResolutionError,
    SteeringApplicationError,
)
from daph_learning.telemetry import emit_fallback

RouteTarget = Literal["symbolic", "llm", "abstain"]
_VALID_ROUTE_TARGETS: frozenset[str] = frozenset({"symbolic", "llm", "abstain"})


# --- Typed failures (canonical taxonomy lives in daph_learning.routing.errors;
# re-exported here for backward compatibility with V037-001 callers) ---


@dataclass(frozen=True)
class RouteDecision:
    """Policy-level routing decision for a single task.

    Attributes
    ----------
    task_id : str
        Identifier of the task this decision applies to.
    route : RouteTarget
        One of ``"symbolic"``, ``"llm"``, ``"abstain"``.
    confidence : float | None
        Optional caller-supplied confidence in the range ``[0, 1]``.
        ``None`` means "not reported".
    raw_scores : dict[str, float]
        Per-backend raw scores that produced the decision (e.g.
        ``{"symbolic": 1.2, "llm": -0.4}``). Empty for heuristic routers
        that do not produce scores.
    source : str
        Provenance of the decision: ``"custom"``, ``"steered"``, or
        ``"baseline"``.
    """

    task_id: str
    route: RouteTarget
    confidence: float | None = None
    raw_scores: dict[str, float] = field(default_factory=dict)
    source: str = "baseline"


# --- Custom route_fn result coercion ---


def coerce_route_decision(
    task: Mapping[str, Any],
    result: Any,
    *,
    source: str = "custom",
) -> RouteDecision:
    """Coerce a ``route_fn`` return value into a :class:`RouteDecision`.

    Accepted result shapes:

    - ``RouteDecision``: returned as-is (``task_id`` is verified to match).
    - ``str`` in ``{"symbolic", "llm", "abstain"}``: wrapped with the task id.
    - ``dict`` with a ``"route"`` key whose value is one of the above. Optional
      keys: ``"confidence"`` (float), ``"raw_scores"`` (dict[str, float]).

    Anything else raises :class:`InvalidRouteDecisionError`.
    """
    task_id = str(task.get("task_id", "")) or ""

    if isinstance(result, RouteDecision):
        if result.task_id and task_id and result.task_id != task_id:
            raise InvalidRouteDecisionError(
                f"route_fn returned RouteDecision with task_id={result.task_id!r} "
                f"but the task being routed has task_id={task_id!r}"
            )
        # Re-tag the source so downstream telemetry knows it came through the
        # custom path even if the caller constructed a RouteDecision manually.
        return RouteDecision(
            task_id=result.task_id or task_id,
            route=result.route,
            confidence=result.confidence,
            raw_scores=dict(result.raw_scores),
            source=source,
        )

    if isinstance(result, str):
        if result not in _VALID_ROUTE_TARGETS:
            raise InvalidRouteDecisionError(
                f"route_fn returned string {result!r} for task {task_id!r}; "
                f"expected one of {sorted(_VALID_ROUTE_TARGETS)}"
            )
        return RouteDecision(task_id=task_id, route=result, source=source)  # type: ignore[arg-type]

    if isinstance(result, Mapping):
        if "route" not in result:
            raise InvalidRouteDecisionError(
                f"route_fn returned dict without a 'route' key for task {task_id!r}; "
                f"got keys {sorted(map(str, result.keys()))}"
            )
        route = result["route"]
        if isinstance(route, str) and route in _VALID_ROUTE_TARGETS:
            confidence = result.get("confidence")
            raw_scores = result.get("raw_scores", {})
            if confidence is not None and not isinstance(confidence, (int, float)):
                raise InvalidRouteDecisionError(
                    f"route_fn returned non-numeric confidence {confidence!r} "
                    f"for task {task_id!r}"
                )
            if not isinstance(raw_scores, Mapping):
                raise InvalidRouteDecisionError(
                    f"route_fn returned non-mapping raw_scores {raw_scores!r} "
                    f"for task {task_id!r}"
                )
            return RouteDecision(
                task_id=task_id,
                route=route,  # type: ignore[arg-type]
                confidence=float(confidence) if confidence is not None else None,
                raw_scores={str(k): float(v) for k, v in raw_scores.items()},
                source=source,
            )
        raise InvalidRouteDecisionError(
            f"route_fn returned dict with invalid 'route' value {route!r} "
            f"for task {task_id!r}; expected one of {sorted(_VALID_ROUTE_TARGETS)}"
        )

    raise InvalidRouteDecisionError(
        f"route_fn returned unsupported type {type(result).__name__} "
        f"for task {task_id!r}; expected RouteDecision, str, or dict"
    )


# --- Baseline capability heuristic ---


def baseline_router(
    tasks: Sequence[Mapping[str, Any]],
) -> list[RouteDecision]:
    """Capability heuristic router (no model required).

    A task with any capability id is routed to ``"symbolic"``; otherwise to
    ``"llm"``. This matches the v0.3.6 behaviour used by
    ``run_autolearn_loop`` when no steering vector is available.
    """
    decisions: list[RouteDecision] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        caps = task.get("capability_ids") or []
        route: RouteTarget = "symbolic" if caps else "llm"
        decisions.append(
            RouteDecision(
                task_id=task_id,
                route=route,
                raw_scores={},
                source="baseline",
            )
        )
    return decisions


# --- Steered (model-mediated) router ---


def _generate_route(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, Any],
    valid_targets: frozenset[str],
) -> RouteTarget:
    """Generate a few tokens and parse the route action from the output.

    Used as the fallback when direct-logit routing fails (multi-token
    labels, context-boundary issues). Returns the parsed route action
    or ``"llm"`` if parsing fails.
    """
    gen_out = model.generate(
        **encoded,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        use_cache=True,
    )
    generated = tokenizer.decode(
        gen_out[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    action = parse_route_action(generated)
    return action if action in valid_targets else "llm"  # type: ignore[return-value]


def steered_router(
    tasks: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    steering: Any,
    *,
    alpha: float = 1.0,
    prompt_format: str = "raw",
) -> list[RouteDecision]:
    """Model-mediated steered router using first-token logit contrast.

    This is a minimal, self-contained implementation that depends only on
    ``daph_learning.routing.steered_router`` and ``daph_learning.steering.hooks``
    (both in ``src/``). It does **not** depend on ``scripts/``. The richer
    batched evaluator (cascading contextual -> isolated -> generate fallback)
    lives in ``scripts/tune_steering.py`` today and will be moved into
    ``daph_learning.routing.batched`` by V037-002; once that lands the loop
    will delegate to it. Until then this provides a working steered path for
    ``route_tasks`` that is correct for single-token tokenizers and falls back
    to generate-mode for multi-token tokenizers.
    """
    import torch

    from daph_learning.routing.steered_router import (
        build_route_prompt,
        parse_route_action,
        resolve_route_token_ids_contextual,
        route_action_from_logits,
    )
    from daph_learning.steering.hooks import residual_addition_hook

    embed_device = model.get_input_embeddings().weight.device
    vec_values = np.asarray(steering.values, dtype=np.float32)
    layer_index = steering.spec.layer

    decisions: list[RouteDecision] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        prompt = build_route_prompt(task)
        if (
            prompt_format == "chat"
            and hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template is not None
        ):
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {
            k: (v.to(embed_device) if hasattr(v, "to") else torch.tensor(v).to(embed_device))
            for k, v in encoded.items()
        }

        route: RouteTarget = "llm"
        raw_scores: dict[str, float] = {}
        confidence: float | None = None

        try:
            with torch.inference_mode(), residual_addition_hook(
                model,
                layer_index=layer_index,
                vector=vec_values,
                alpha=alpha,
                token_scope="last",
            ):
                out = model(
                    **encoded,
                    output_hidden_states=False,
                    return_dict=True,
                )
            logits = out.logits
            try:
                sym_id, llm_id = resolve_route_token_ids_contextual(
                    tokenizer,
                    rendered,
                    allow_first_token_fallback=True,
                )
                action, margin = route_action_from_logits(logits, sym_id, llm_id)
                route = action  # type: ignore[assignment]
                raw_scores = {"symbolic": float(margin), "llm": 0.0}
                confidence = 1.0 / (1.0 + abs(float(margin)))
            except MultiTokenRouteError:
                # Expected: label tokenizes to >1 token. Fall back to generate
                # mode with structured telemetry.
                emit_fallback(
                    event="route_fallback",
                    frm="logit",
                    to="generate",
                    reason="multi_token_label",
                    task_id=task_id,
                )
                route = _generate_route(model, tokenizer, encoded, _VALID_ROUTE_TARGETS)
            except RouteResolutionError as exc:
                # Expected: context-boundary or other route-resolution issue.
                emit_fallback(
                    event="route_fallback",
                    frm="logit",
                    to="generate",
                    reason="context_boundary",
                    task_id=task_id,
                    detail=str(exc),
                )
                route = _generate_route(model, tokenizer, encoded, _VALID_ROUTE_TARGETS)
        except SteeringApplicationError as exc:
            # Expected-ish: steering hook failed (layer index, shape mismatch).
            # Fall back to a conservative LLM default with telemetry.
            emit_fallback(
                event="route_fallback",
                frm="steered",
                to="llm_default",
                reason="steering_hook_failure",
                task_id=task_id,
                detail=str(exc),
            )
            route = "llm"
        except ModelRoutingError as exc:
            # Model forward/generate failed unexpectedly. Conservative default.
            emit_fallback(
                event="route_fallback",
                frm="steered",
                to="llm_default",
                reason="model_routing_failure",
                task_id=task_id,
                detail=str(exc),
            )
            route = "llm"

        decisions.append(
            RouteDecision(
                task_id=task_id,
                route=route,
                confidence=confidence,
                raw_scores=raw_scores,
                source="steered",
            )
        )
    return decisions


# --- Public dispatcher ---


def route_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    steering: Any = None,
    route_fn: Callable[..., Any] | None = None,
    alpha: float = 1.0,
    prompt_format: str = "raw",
) -> list[RouteDecision]:
    """Resolve a routing decision for every task in ``tasks``.

    Dispatch order (first match wins):

    1. ``route_fn is not None``  -> call ``route_fn(task=task, model=model,
       tokenizer=tokenizer, steering=steering)`` once per task and coerce the
       result. Invalid results raise :class:`InvalidRouteDecisionError`.
       Exceptions from ``route_fn`` propagate (they are **not** swallowed).
    2. ``steering is not None``   -> :func:`steered_router` (model-mediated).
    3. else                       -> :func:`baseline_router` (capability heuristic).

    Parameters
    ----------
    tasks : sequence of mappings
        Tasks to route. Each must have a ``task_id`` (or be tagged with an
        empty id, which is preserved).
    model, tokenizer : Any
        HuggingFace-style model and tokenizer. Required by the steered path;
        ignored by the custom and baseline paths (passed through to
        ``route_fn``).
    steering : SteeringVector | None
        Steering vector for the steered path. Ignored when ``route_fn`` is
        provided.
    route_fn : callable | None
        Custom router. Signature: ``route_fn(task, model=model,
        tokenizer=tokenizer, steering=steering) -> RouteDecision | str | dict``.
    alpha : float
        Steering alpha for the steered path.
    prompt_format : str
        ``"raw"`` or ``"chat"`` for the steered path.

    Returns
    -------
    list[RouteDecision]
        One decision per task, in input order.

    Raises
    ------
    InvalidRouteDecisionError
        If ``route_fn`` returns a value that cannot be coerced to a
        ``RouteDecision``.
    """
    if route_fn is not None:
        decisions: list[RouteDecision] = []
        for task in tasks:
            result = route_fn(
                task=task,
                model=model,
                tokenizer=tokenizer,
                steering=steering,
            )
            decisions.append(coerce_route_decision(task, result, source="custom"))
        return decisions

    if steering is not None:
        return steered_router(
            tasks,
            model,
            tokenizer,
            steering,
            alpha=alpha,
            prompt_format=prompt_format,
        )

    return baseline_router(tasks)
