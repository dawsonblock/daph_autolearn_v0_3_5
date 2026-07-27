"""Core AutoLearn learning loop.

The loop iteratively improves a steering vector by learning from execution
outcomes. See ``daph_learning.autolearn.__init__`` for the design overview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np

from daph_learning.execution.plan import ExecutionResult
from daph_learning.execution.symbolic_executor import (
    execute_plan,
    plan_from_structured_task,
)
from daph_learning.routing.errors import (
    ContextBoundaryError,
    ModelRoutingError,
    MultiTokenRouteError,
    SteeringApplicationError,
)
from daph_learning.steering.extract import contrastive_mean_direction
from daph_learning.steering.hooks import capture_layer_output
from daph_learning.steering.types import SteeringSpec, SteeringVector
from daph_learning.tools.symbolic_math import SymbolicMathError


# --- Outcome classification ---

OutcomeLabel = Literal["correct", "misrouted", "unverifiable"]


def classify_outcome(
    task: dict[str, Any],
    route: str | None,
    output: str | None,
    *,
    expected: Any | None = None,
    route_raw: str | None = None,
) -> OutcomeLabel:
    """Classify the outcome of a routed task execution.

    Returns one of:
    - ``"correct"``: the route was right and the output matches the expected
      answer (for symbolic) or the task was LLM-routed with no expected
      answer to check (non-symbolic tasks).
    - ``"misrouted"``: the route was wrong — e.g. sent to symbolic but
      symbolic failed, or sent to LLM when symbolic would have succeeded
      and the LLM output doesn't match.
    - ``"unverifiable"``: the outcome cannot be checked (no expected answer,
      or the output format doesn't contain a checkable value).

    The classification logic:
    - If route=symbolic and execution failed (error/exception): misrouted.
    - If route=symbolic and execution succeeded but value != expected: misrouted.
    - If route=symbolic and execution succeeded and value == expected: correct.
    - If route=llm and task has a numeric expected answer: the LLM should
      have been able to produce it, but we can't verify the LLM output
      without parsing it. We treat this as "unverifiable" unless the output
      clearly contains the expected value.
    - If route=llm and task has no expected answer (non-symbolic task):
      correct (the LLM was the right choice).

    ``route_raw`` (v0.3.6) is the raw generated route text when the route
    decision came from autoregressive generation rather than direct-logit
    contrast (e.g. for multi-token tokenizers where the first-token logit
    contrast is unavailable). When the parsed ``route`` is ``None``-ish
    (unparseable generation), ``route_raw`` is inspected directly so a
    generated ``"SYMBOLIC"``/``"LLM"`` token sequence is still credited
    instead of being silently dropped. This is the Phase 1.1 repair: it
    lets the AutoLearn loop converge on multi-token tokenizers (Qwen2.5)
    where the isolated/contextual resolvers previously raised and the
    generate-mode fallback returned ``None`` for every task.
    """
    expected = expected if expected is not None else task.get("expected")

    # v0.3.6: when the caller passes a parsed route of None (generation
    # produced no parseable ACTION), inspect route_raw directly. This keeps
    # multi-token generated outputs (e.g. " SYMBOLIC" emitted as " SY",
    # "MBOL", "IC" then decoded) from being silently treated as LLM.
    if route is None and route_raw:
        from daph_learning.routing.steered_router import parse_route_action
        parsed = parse_route_action(route_raw)
        if parsed is not None:
            route = parsed

    if route == "symbolic":
        # Symbolic execution outcome is checked by the caller (who caught
        # exceptions). Here we check the output string.
        if output is None:
            return "misrouted"  # symbolic execution produced nothing
        # The output format is "FINAL: <value>"
        if output.startswith("FINAL:"):
            value_str = output[len("FINAL:"):].strip()
            if expected is not None:
                try:
                    if int(value_str) == int(expected):
                        return "correct"
                    return "misrouted"
                except (ValueError, TypeError):
                    return "misrouted"
            return "correct"  # no expected value to check
        return "misrouted"  # malformed symbolic output

    # route == "llm" (or still None — treat as LLM default)
    if expected is None:
        # Non-symbolic task routed to LLM — correct by default
        return "correct"

    # Symbolic-capable task routed to LLM — check if the LLM got it right.
    # v0.3.6: also check route_raw when output is unavailable, since on the
    # generate-mode routing path the "output" is the route text itself.
    candidate_texts: list[str] = []
    if output is not None:
        candidate_texts.append(output.strip())
    if route_raw is not None:
        candidate_texts.append(route_raw.strip())
    for text in candidate_texts:
        try:
            if str(int(expected)) in text:
                return "correct"
        except (ValueError, TypeError):
            pass

    # LLM was asked but we can't verify the answer — unverifiable
    return "unverifiable"


# --- Data structures ---

@dataclass(frozen=True)
class AutoLearnConfig:
    """Configuration for the AutoLearn learning loop.

    Parameters
    ----------
    n_iterations : int
        Maximum number of learning iterations.
    layer : int
        Which transformer layer to capture activations from and steer.
    alpha : float
        Steering alpha to use during routing.
    anchor : str
        The anchor token for activation capture and steering.
    capture_token_scope : str
        "last" to capture only the anchor token's activation, "all" for
        the full sequence.
    min_examples_per_update : int
        Minimum number of positive and negative examples required to
        update the steering vector. If fewer are available, the loop
        stops early.
    normalization : str
        Normalization for the extracted direction: "l2", "none", or
        "mean_centered".
    seed : int
        Random seed for reproducibility.
    """

    n_iterations: int = 5
    layer: int = 24
    alpha: float = 1.0
    anchor: str = "ACTION"
    capture_token_scope: str = "last"
    min_examples_per_update: int = 4
    normalization: str = "l2"
    seed: int = 42


@dataclass
class IterationMetrics:
    """Metrics for a single iteration of the learning loop."""

    iteration: int
    n_train: int
    n_correct: int
    n_misrouted: int
    n_unverifiable: int
    train_accuracy: float
    val_f1: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    vector_norm: float
    n_positive_examples: int
    n_negative_examples: int
    updated: bool  # whether the vector was updated this iteration


@dataclass
class AutoLearnResult:
    """Result of the AutoLearn learning loop.

    Attributes
    ----------
    best_vector : SteeringVector | None
        The steering vector from the iteration with the best validation F1.
    best_iteration : int
        The iteration index that produced the best vector.
    best_val_f1 : float
        The best validation F1 achieved.
    iterations : list[IterationMetrics]
        Metrics for each iteration, forming the learning curve.
    config : AutoLearnConfig
        The configuration used.
    """

    best_vector: SteeringVector | None
    best_iteration: int
    best_val_f1: float
    iterations: list[IterationMetrics]
    config: AutoLearnConfig


# --- The loop ---

def _capture_activations(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    anchor: str,
    prompt_format: str = "raw",
) -> np.ndarray | None:
    """Capture activations from the anchor token for a list of tasks.

    Returns an [N, H] array of activations, or None if capture fails.
    """
    import torch
    from daph_learning.routing.steered_router import build_route_prompt

    activations = []
    for task in tasks:
        prompt = build_route_prompt(task)
        # Apply prompt formatting
        if prompt_format == "chat" and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=True)
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        sink: list = []
        try:
            with torch.inference_mode(), capture_layer_output(
                model,
                layer_index=layer,
                sink=sink,
                target_token_index=-1,
            ):
                model(**encoded, use_cache=False)
            if sink:
                act = sink[0]
                if act.ndim == 2:
                    act = act[-1]  # last token
                activations.append(act.numpy().astype(np.float32))
        except (RuntimeError, ValueError, TypeError, IndexError, SteeringApplicationError) as exc:
            # Expected activation-capture failures: layer index out of range,
            # shape/device mismatch, hook application error, type mismatch
            # from a non-standard tokenizer. The task is skipped (no
            # activation collected). Unexpected exceptions propagate.
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="activation_capture_failure",
                frm="capture",
                to="skip",
                reason=type(exc).__name__,
                task_id=str(task.get("task_id", "")),
                layer=layer,
            )
            continue

    if not activations:
        return None
    return np.stack(activations, axis=0)


def _execute_symbolic(task: dict[str, Any]) -> tuple[str | None, bool]:
    """Execute a task symbolically. Returns (output_string, success).

    The output string is "FINAL: <value>" on success, None on failure.
    """
    try:
        plan = plan_from_structured_task(task, reason_code="autolearn")
        result = execute_plan(plan)
        return f"FINAL: {result.value}", result.verified
    except (SymbolicMathError, ValueError, TypeError, KeyError) as exc:
        # Expected symbolic-execution failures: unsafe expression, unsupported
        # operation, resource limit, unsupported capability, missing input.
        # These are normal "symbolic couldn't handle this task" signals, not
        # infrastructure failures. Unexpected exceptions propagate.
        from daph_learning.telemetry import emit_fallback
        emit_fallback(
            event="symbolic_execution_failure",
            frm="symbolic",
            to="none",
            reason=type(exc).__name__,
            task_id=str(task.get("task_id", "")),
        )
        return None, False


def _route_with_steering_generate(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    vector: SteeringVector,
    alpha: float,
    *,
    prompt_format: str = "raw",
    max_new_tokens: int = 5,
) -> list[tuple[str | None, str | None]]:
    """Route tasks using steering + generate mode.

    This is the fallback for tokenizers where direct-logit routing fails
    (e.g. Qwen2.5 tokenizes "SYMBOLIC" as multiple tokens). It applies
    the steering vector via a residual addition hook, generates a few
    tokens, and parses the route action from the generated text.

    Returns a list of ``(route, raw_text)`` tuples where ``route`` is
    "symbolic", "llm", or None if parsing fails, and ``raw_text`` is
    the decoded generation (for downstream route_raw recovery).
    """
    import torch
    from daph_learning.routing.steered_router import build_route_prompt, parse_route_action
    from daph_learning.steering.hooks import residual_addition_hook

    embed_device = model.get_input_embeddings().weight.device
    routes: list[tuple[str | None, str | None]] = []

    for task in tasks:
        prompt = build_route_prompt(task)
        if prompt_format == "chat" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {
            k: (v.to(embed_device) if hasattr(v, "to") else torch.tensor(v).to(embed_device))
            for k, v in encoded.items()
        }

        vec_values = np.asarray(vector.values, dtype=np.float32)
        try:
            with torch.inference_mode(), residual_addition_hook(
                model,
                layer_index=vector.spec.layer,
                vector=vec_values,
                alpha=alpha,
                token_scope="last",
            ):
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    use_cache=True,
                )
            generated = tokenizer.decode(
                out[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            action = parse_route_action(generated)
            routes.append((action, generated))
        except (RuntimeError, SteeringApplicationError, ValueError, AttributeError) as exc:
            # Expected generate-mode routing failures: model.generate error,
            # steering hook failure, decode/shape error, or model lacks a
            # generate method. The task gets a (None, None) route; unexpected
            # exceptions propagate.
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="route_fallback",
                frm="steered_generate",
                to="none",
                reason=type(exc).__name__,
                task_id=str(task.get("task_id", "")),
            )
            routes.append((None, None))

    return routes


def _route_without_steering_generate(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    prompt_format: str = "raw",
    max_new_tokens: int = 5,
) -> list[str | None]:
    """Route tasks without steering using generate mode.

    Returns a list of route strings ("symbolic", "llm", or None).
    """
    import torch
    from daph_learning.routing.steered_router import build_route_prompt, parse_route_action

    embed_device = model.get_input_embeddings().weight.device
    routes: list[str | None] = []

    for task in tasks:
        prompt = build_route_prompt(task)
        if prompt_format == "chat" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {
            k: (v.to(embed_device) if hasattr(v, "to") else torch.tensor(v).to(embed_device))
            for k, v in encoded.items()
        }

        try:
            with torch.inference_mode():
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    use_cache=True,
                )
            generated = tokenizer.decode(
                out[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            action = parse_route_action(generated)
            routes.append(action)
        except (RuntimeError, ValueError, AttributeError) as exc:
            # Expected generate-mode routing failures (no steering): model
            # error, decode error, or model lacks a generate method. The task
            # gets a None route; unexpected exceptions propagate.
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="route_fallback",
                frm="generate",
                to="none",
                reason=type(exc).__name__,
                task_id=str(task.get("task_id", "")),
            )
            routes.append(None)

    return routes


def _evaluate_routes_simple(
    tasks: list[dict[str, Any]],
    routes: list[str | None],
    label_field: str = "route_label",
) -> dict[str, float]:
    """Simple routing evaluation that doesn't require the full
    evaluate_route_records infrastructure. Works with generate-mode
    routing where route records aren't available.

    Returns accuracy, precision, recall, F1, and confusion matrix.
    """
    expected = [t.get(label_field, "llm") for t in tasks]
    routes_clean = [r if r is not None else "llm" for r in routes]

    tp = sum(1 for r, e in zip(routes_clean, expected) if r == "symbolic" and e == "symbolic")
    fp = sum(1 for r, e in zip(routes_clean, expected) if r == "symbolic" and e == "llm")
    fn = sum(1 for r, e in zip(routes_clean, expected) if r == "llm" and e == "symbolic")
    tn = sum(1 for r, e in zip(routes_clean, expected) if r == "llm" and e == "llm")

    accuracy = (tp + tn) / len(tasks) if tasks else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "route_accuracy": accuracy,
        "decision_coverage": sum(1 for r in routes if r is not None) / len(routes) if routes else 0.0,
    }


def run_autolearn_loop(
    train_tasks: list[dict[str, Any]],
    val_tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    config: AutoLearnConfig,
    initial_vector: SteeringVector | None = None,
    route_fn: Callable[..., dict[str, Any]] | None = None,
    eval_fn: Callable[..., dict[str, float]] | None = None,
    prompt_format: str = "raw",
    label_field: str | None = "route_label",
    label_oracle_kind: str | None = "policy_heuristic",
    routing_mode: str = "auto",
) -> AutoLearnResult:
    """Run the AutoLearn learning loop.

    The loop iteratively improves a steering vector by:
    1. Routing training tasks with the current vector.
    2. Executing the chosen backend.
    3. Classifying outcomes as correct/misrouted/unverifiable.
    4. Capturing activations from correct (positive) and misrouted (negative) tasks.
    5. Updating the steering vector via contrastive mean difference.
    6. Evaluating on the validation set.

    Parameters
    ----------
    train_tasks, val_tasks : list[dict]
        Training and validation task lists.
    model, tokenizer : Any
        The HuggingFace model and tokenizer.
    config : AutoLearnConfig
        Loop configuration.
    initial_vector : SteeringVector | None
        Starting vector. If None, the first iteration routes without
        steering (baseline) and uses the outcomes to extract the first vector.
    route_fn : callable, optional
        Custom routing function. If None, uses a simple symbolic-only
        execution path (no model routing) for the first pass, then
        switches to steered routing once a vector is available.
    eval_fn : callable, optional
        Custom evaluation function. If None, uses evaluate_route_records.
    prompt_format : str
        "raw" or "chat".
    label_field, label_oracle_kind : str | None
        Passed to the evaluation function.
    routing_mode : str
        "auto" (default): try logit-based routing, fall back to generate
        mode if the tokenizer produces multi-token labels.
        "logit": force logit-based routing (fails for multi-token tokenizers).
        "generate": force generate-mode routing (slower but works with
        any tokenizer).

    Returns
    -------
    AutoLearnResult
        The best vector, best iteration, and the full learning curve.
    """
    # V037-002: import reusable routing/evaluation helpers from the library
    # layer instead of from scripts/. The dependency direction is now
    # CLI -> package only.
    from daph_learning.evaluation.routes import evaluate_route_records
    from daph_learning.routing.batched import (
        evaluate_batch_steered_routes as _evaluate_batch_steered_routes,
        as_task_map as _as_task_map,
    )

    if eval_fn is None:
        eval_fn = evaluate_route_records

    rng = np.random.RandomState(config.seed)
    current_vector = initial_vector
    iterations: list[IterationMetrics] = []
    best_vector = initial_vector
    best_val_f1 = -1.0
    best_iteration = -1

    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("cannot determine model hidden_size")

    # Determine which routing method to use.
    # If routing_mode is "auto", try logit first and fall back to generate.
    use_generate_mode = routing_mode == "generate"

    for iteration in range(config.n_iterations):
        # --- Stage 1: Route and execute training tasks ---
        routes: dict[str, str] = {}
        route_raws: dict[str, str | None] = {}
        outputs: dict[str, str | None] = {}

        if route_fn is not None:
            # V037-001: actually invoke the user-supplied router. Previously
            # `route_fn` was accepted but never called (the v0.3.6 branch
            # `current_vector is not None and route_fn is None` sent every
            # other case to the capability heuristic). The custom router now
            # takes precedence over both the steered and baseline paths, and
            # invalid results raise InvalidRouteDecisionError instead of being
            # silently dropped. `route_fn` exceptions propagate.
            from daph_learning.routing.policy import route_tasks as _route_tasks
            decisions = _route_tasks(
                train_tasks,
                model=model,
                tokenizer=tokenizer,
                steering=current_vector,
                route_fn=route_fn,
                alpha=config.alpha,
                prompt_format=prompt_format,
            )
            for task, decision in zip(train_tasks, decisions):
                tid = task["task_id"]
                # v0.3.7: abstain is collapsed to "llm" for the execution
                # stage because the v0.3.7 backend executor only knows how to
                # run symbolic vs LLM. A real abstain backend arrives in v0.3.9
                # (V039-003). The decision source is preserved in
                # `route_raws` so downstream telemetry can distinguish
                # abstain-from-custom vs llm-from-steered.
                routes[tid] = "llm" if decision.route == "abstain" else decision.route
                route_raws[tid] = decision.source
        elif current_vector is not None:
            if use_generate_mode:
                # Use generate-mode steered routing (works with any tokenizer)
                task_list = train_tasks
                steered_route_list = _route_with_steering_generate(
                    task_list, model, tokenizer, current_vector, config.alpha,
                    prompt_format=prompt_format,
                )
                for task, (route, raw) in zip(task_list, steered_route_list):
                    routes[task["task_id"]] = route if route else "llm"
                    route_raws[task["task_id"]] = raw
            else:
                # v0.3.6 routing cascade: contextual first-token logit
                # contrast → isolated first-token contrast → generate mode.
                # The first-token fallback lets the loop keep using the
                # cheap single-forward logit path on multi-token tokenizers
                # (Qwen2.5) instead of immediately dropping to per-task
                # autoregressive generation. See Phase 1.1.
                task_map = _as_task_map(train_tasks)
                task_list = list(task_map.values())
                routed_via_logit = False
                for resolver in ("contextual", "isolated"):
                    try:
                        steered_routes = _evaluate_batch_steered_routes(
                            task_list,
                            model,
                            tokenizer,
                            vectors=[current_vector],
                            alphas=[config.alpha],
                            prompt_format=prompt_format,
                            token_resolver=resolver,
                            allow_first_token_fallback=True,
                        )
                        for task in task_list:
                            routes[task["task_id"]] = steered_routes[task["task_id"]]["route"]
                        routed_via_logit = True
                        break
                    except MultiTokenRouteError as exc:
                        # Expected: tokenizer produces multi-token route labels.
                        # Try the next resolver (contextual -> isolated) and
                        # emit structured telemetry for the fallback.
                        from daph_learning.telemetry import emit_fallback
                        emit_fallback(
                            event="route_fallback",
                            frm=resolver,
                            to="isolated" if resolver == "contextual" else "generate",
                            reason="multi_token_label",
                            iteration=iteration,
                            detail=str(exc),
                        )
                        continue
                    except ContextBoundaryError as exc:
                        # Expected: prompt/label boundary unresolvable.
                        from daph_learning.telemetry import emit_fallback
                        emit_fallback(
                            event="route_fallback",
                            frm=resolver,
                            to="isolated" if resolver == "contextual" else "generate",
                            reason="context_boundary",
                            iteration=iteration,
                            detail=str(exc),
                        )
                        continue
                if not routed_via_logit:
                    if routing_mode == "logit":
                        raise ModelRoutingError(
                            "logit routing mode requested but both contextual and "
                            "isolated resolvers failed to produce single-token ids"
                        )
                    # Fall back to generate mode
                    from daph_learning.telemetry import emit_fallback
                    emit_fallback(
                        event="route_fallback",
                        frm="logit",
                        to="generate",
                        reason="all_resolvers_failed",
                        iteration=iteration,
                    )
                    use_generate_mode = True
                    steered_route_list = _route_with_steering_generate(
                        task_list, model, tokenizer, current_vector, config.alpha,
                        prompt_format=prompt_format,
                    )
                    for task, (route, raw) in zip(task_list, steered_route_list):
                        routes[task["task_id"]] = route if route else "llm"
                        route_raws[task["task_id"]] = raw
        else:
            # No vector yet — use heuristic routing
            for task in train_tasks:
                routes[task["task_id"]] = "symbolic" if task.get("capability_ids") else "llm"

        # Execute
        for task in train_tasks:
            tid = task["task_id"]
            route = routes[tid]
            if route == "symbolic":
                output, _ = _execute_symbolic(task)
                outputs[tid] = output
            else:
                # LLM execution — for the learning loop, we mark as unverifiable
                # unless we actually generate. To keep the loop fast, we skip
                # LLM generation during training and rely on the symbolic
                # outcome + expected value for feedback.
                outputs[tid] = None

        # --- Stage 2: Classify outcomes ---
        positive_tasks: list[dict[str, Any]] = []
        negative_tasks: list[dict[str, Any]] = []
        n_correct = n_misrouted = n_unverifiable = 0

        for task in train_tasks:
            tid = task["task_id"]
            route = routes[tid]
            output = outputs[tid]
            outcome = classify_outcome(
                task, route, output, route_raw=route_raws.get(tid),
            )

            if outcome == "correct":
                n_correct += 1
                # For the steering vector, "positive" = should route to symbolic,
                # "negative" = should route to LLM. We use the oracle labels
                # to determine which class each task belongs to, and the
                # execution outcome to determine if the current route was right.
                # A correctly-routed symbolic task is a positive example.
                # A correctly-routed LLM task (non-symbolic) is a negative example.
                if route == "symbolic":
                    positive_tasks.append(task)
                else:
                    negative_tasks.append(task)
            elif outcome == "misrouted":
                n_misrouted += 1
                # A misrouted task that went to symbolic but should have gone
                # to LLM is a negative example. One that went to LLM but should
                # have gone to symbolic is a positive example.
                if route == "symbolic":
                    negative_tasks.append(task)
                else:
                    positive_tasks.append(task)
            else:
                n_unverifiable += 1

        # --- Stage 3: Capture activations and update vector ---
        updated = False
        n_pos = len(positive_tasks)
        n_neg = len(negative_tasks)
        vector_norm = float(np.linalg.norm(current_vector.values)) if current_vector else 0.0

        if n_pos >= config.min_examples_per_update and n_neg >= config.min_examples_per_update:
            pos_activations = _capture_activations(
                positive_tasks, model, tokenizer,
                layer=config.layer, anchor=config.anchor,
                prompt_format=prompt_format,
            )
            neg_activations = _capture_activations(
                negative_tasks, model, tokenizer,
                layer=config.layer, anchor=config.anchor,
                prompt_format=prompt_format,
            )

            if pos_activations is not None and neg_activations is not None:
                spec = SteeringSpec(
                    vector_id=f"autolearn_iter_{iteration}",
                    family="tool_policy",
                    behavior="invoke_symbolic_tool",
                    layer=config.layer,
                    alpha=config.alpha,
                    anchor=config.anchor,
                    model_id=getattr(model.config, "_name_or_path", "unknown"),
                    hidden_size=hidden_size,
                    extraction_method="contrastive_mean_difference",
                    normalization=config.normalization,
                    positive_n=n_pos,
                    negative_n=n_neg,
                    capture_anchor=config.anchor,
                    capture_prompt_format=prompt_format,
                )
                current_vector = contrastive_mean_direction(
                    pos_activations, neg_activations, spec,
                )
                vector_norm = float(np.linalg.norm(current_vector.values))
                updated = True

        # --- Stage 4: Evaluate on validation set ---
        val_f1 = val_accuracy = val_precision = val_recall = 0.0
        if val_tasks:
            val_list = val_tasks

            if current_vector is not None:
                if use_generate_mode:
                    # Use generate-mode routing for validation
                    val_route_list = _route_with_steering_generate(
                        val_list, model, tokenizer, current_vector, config.alpha,
                        prompt_format=prompt_format,
                    )
                    val_metrics = _evaluate_routes_simple(
                        val_list, val_route_list,
                        label_field=label_field or "route_label",
                    )
                    val_f1 = float(val_metrics["f1"])
                    val_accuracy = float(val_metrics["accuracy"])
                    val_precision = float(val_metrics["precision"])
                    val_recall = float(val_metrics["recall"])
                else:
                    val_map = _as_task_map(val_tasks)
                    val_routed_via_logit = False
                    for resolver in ("contextual", "isolated"):
                        try:
                            val_routes = _evaluate_batch_steered_routes(
                                val_list,
                                model,
                                tokenizer,
                                vectors=[current_vector],
                                alphas=[config.alpha],
                                prompt_format=prompt_format,
                                token_resolver=resolver,
                                allow_first_token_fallback=True,
                            )
                            val_metrics = eval_fn(
                                val_map,
                                val_routes,
                                label_field=label_field,
                                label_oracle_kind=label_oracle_kind,
                            )
                            val_f1 = float(val_metrics["f1"])
                            val_accuracy = float(val_metrics["route_accuracy"])
                            val_precision = float(val_metrics["precision"])
                            val_recall = float(val_metrics["recall"])
                            val_routed_via_logit = True
                            break
                        except MultiTokenRouteError as exc:
                            from daph_learning.telemetry import emit_fallback
                            emit_fallback(
                                event="route_fallback",
                                frm=resolver,
                                to="isolated" if resolver == "contextual" else "generate",
                                reason="multi_token_label",
                                iteration=iteration,
                                phase="validation",
                                detail=str(exc),
                            )
                            continue
                        except ContextBoundaryError as exc:
                            from daph_learning.telemetry import emit_fallback
                            emit_fallback(
                                event="route_fallback",
                                frm=resolver,
                                to="isolated" if resolver == "contextual" else "generate",
                                reason="context_boundary",
                                iteration=iteration,
                                phase="validation",
                                detail=str(exc),
                            )
                            continue
                    if not val_routed_via_logit:
                        # Fall back to generate mode for validation too
                        from daph_learning.telemetry import emit_fallback
                        emit_fallback(
                            event="route_fallback",
                            frm="logit",
                            to="generate",
                            reason="all_resolvers_failed",
                            iteration=iteration,
                            phase="validation",
                        )
                        use_generate_mode = True
                        val_route_list = _route_with_steering_generate(
                            val_list, model, tokenizer, current_vector, config.alpha,
                            prompt_format=prompt_format,
                        )
                        val_routes_only = [r for r, _ in val_route_list]
                        val_metrics = _evaluate_routes_simple(
                            val_list, val_routes_only,
                            label_field=label_field or "route_label",
                        )
                        val_f1 = float(val_metrics["f1"])
                        val_accuracy = float(val_metrics["accuracy"])
                        val_precision = float(val_metrics["precision"])
                        val_recall = float(val_metrics["recall"])

        # --- Record iteration metrics ---
        train_accuracy = n_correct / len(train_tasks) if train_tasks else 0.0
        metrics = IterationMetrics(
            iteration=iteration,
            n_train=len(train_tasks),
            n_correct=n_correct,
            n_misrouted=n_misrouted,
            n_unverifiable=n_unverifiable,
            train_accuracy=train_accuracy,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            val_precision=val_precision,
            val_recall=val_recall,
            vector_norm=vector_norm,
            n_positive_examples=n_pos,
            n_negative_examples=n_neg,
            updated=updated,
        )
        iterations.append(metrics)

        # Track best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_iteration = iteration
            best_vector = current_vector

        # Early stopping: if no update happened, the vector has converged
        if not updated and iteration > 0:
            break

    return AutoLearnResult(
        best_vector=best_vector,
        best_iteration=best_iteration,
        best_val_f1=best_val_f1,
        iterations=iterations,
        config=config,
    )
