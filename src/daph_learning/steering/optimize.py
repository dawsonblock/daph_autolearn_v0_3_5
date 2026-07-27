"""Steering-optimized direction extraction.

The contrastive mean difference direction is optimal for *classification*
(separating positive from negative activations), but not necessarily for
*causal intervention* (flipping the model's next-token prediction).

This module implements an iterative, gradient-free optimization that
adjusts the steering direction to maximize the logit margin between the
symbolic and LLM route tokens. It starts from the contrastive mean
direction and refines it using coordinate-wise perturbation.

The key insight: we want the direction that, when added to the residual
stream, makes the model *output* "SYMBOLIC" for symbolic tasks. This is
a different objective than finding the direction that best *classifies*
the activations.

See CLAIMS.md §12 (linear-probe baseline) for why classification ≠
causal intervention.

v0.3.6 (Phase 1.2) — balanced objective
---------------------------------------
The original optimizer maximized only the positive-class margin
``mean_{p in P+} (logit_SYM(p) - logit_LLM(p))``. With no constraint on
how the direction affects negative-control prompts (non-symbolic tasks
that must route to LLM), coordinate descent inflated the vector norm
until *every* prompt routed to SYMBOLIC — empirically producing
``FP=42, TN=0`` and collapsing test F1 to 0.2759 on Qwen2.5-1.5B.

The v0.3.6 objective adds a hinge penalty on negative-control margins and
an L2 anchor toward the initial direction:

    L(v) = mean_{p in P+} Margin(p, v)
         - lambda * mean_{n in P-} max(0, Margin(n, v) + gamma)
         - beta * ||v - v0||_2^2

where ``Margin(x, v) = logit_SYM(x, v) - logit_LLM(x, v)``, ``gamma`` is
the negative-margin target threshold (how negative the LLM margin must be
on control prompts before the penalty switches off), ``lambda`` weights
the false-positive suppression, and ``beta`` is the L2 anchor strength.

Callers without negative-control prompts get the legacy positive-only
objective (``negative_prompts=[]``), so existing behaviour and tests are
preserved.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Sequence

from daph_learning.steering.hooks import residual_addition_hook, capture_layer_output
from daph_learning.steering.types import SteeringSpec, SteeringVector


def _get_logit_margin(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    layer: int,
    direction: np.ndarray,
    alpha: float,
    sym_token: int,
    llm_token: int,
) -> float:
    """Get the logit margin (SYM - LLM) when steering with the given direction."""
    embed_device = model.get_input_embeddings().weight.device
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    # Convert BatchEncoding to a plain dict of tensors
    if hasattr(encoded, "items"):
        encoded = dict(encoded)
    else:
        encoded = {"input_ids": encoded}
    # Convert all values to tensors on the right device
    new_encoded = {}
    for k, v in encoded.items():
        if isinstance(v, torch.Tensor):
            new_encoded[k] = v.to(embed_device)
        elif isinstance(v, (list, int)):
            t = torch.tensor(v)
            if t.ndim == 0:
                t = t.unsqueeze(0)
            new_encoded[k] = t.to(embed_device)
        else:
            new_encoded[k] = v
    encoded = new_encoded
    # Ensure input_ids is 2D [batch, seq]
    input_ids = encoded["input_ids"]
    if input_ids.dim() == 1:
        encoded["input_ids"] = input_ids.unsqueeze(0)
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

    with torch.inference_mode(), residual_addition_hook(
        model,
        layer_index=layer,
        vector=direction.astype(np.float32),
        alpha=alpha,
        token_scope="last",
    ):
        out = model(**encoded, use_cache=False)

    logits = out.logits[0, -1]
    return float(logits[sym_token].item() - logits[llm_token].item())


def _balanced_objective(
    positive_margins: Sequence[float],
    negative_margins: Sequence[float],
    *,
    direction: np.ndarray,
    initial_direction: np.ndarray,
    lambda_neg: float,
    gamma: float,
    beta: float,
) -> float:
    """The v0.3.6 balanced objective. Higher is better.

    ``positive_margins`` are SYM-LLM logit margins on symbolic prompts
    (we want these large and positive). ``negative_margins`` are the same
    contrast on non-symbolic control prompts (we want these ≤ -gamma so
    the control prompts route to LLM). The hinge ``max(0, m + gamma)``
    only penalizes controls whose margin is above ``-gamma``.

    The L2 anchor ``-beta * ||v - v0||^2`` keeps the optimized direction
    close to the contrastive-mean starting point, preventing the norm
    blow-up that produced 100% false positives under the legacy objective.
    """
    pos_term = float(np.mean(positive_margins)) if positive_margins else 0.0
    if negative_margins:
        hinge = float(np.mean([max(0.0, m + gamma) for m in negative_margins]))
    else:
        hinge = 0.0
    reg = float(beta * np.sum((direction - initial_direction) ** 2))
    return pos_term - lambda_neg * hinge - reg


def optimize_steering_direction(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    initial_direction: np.ndarray,
    layer: int,
    alpha: float,
    sym_token: int,
    llm_token: int,
    n_iterations: int = 20,
    learning_rate: float = 0.1,
    perturbation_scale: float = 0.5,
    seed: int = 42,
    verbose: bool = False,
    method: str = "coordinate",
    n_coordinates: int = 50,
    negative_prompts: list[str] | None = None,
    lambda_neg: float = 1.5,
    gamma: float = 1.0,
    beta: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize a steering direction to maximize the logit margin
    between symbolic and LLM route tokens.

    Two methods are supported:

    - "random" (random search hill climbing): at each iteration, perturb the
      direction randomly, check if the average margin improves, and keep the
      perturbation if it does. Simple but inefficient in high dimensions.

    - "coordinate" (coordinate descent, default): pick the top-k coordinates
      by magnitude of the initial direction, and for each coordinate try
      +delta and -delta perturbations. Keep whichever improves the margin.
      This is much more efficient because it focuses on the coordinates
      that matter most.

    Parameters
    ----------
    model, tokenizer : Any
        The HuggingFace model and tokenizer.
    prompts : list[str]
        Rendered routing prompts for symbolic tasks (we want these to
        produce "SYMBOLIC" when steered).
    initial_direction : np.ndarray
        Starting direction (typically the contrastive mean difference).
    layer : int
        Which layer to steer.
    alpha : float
        Steering alpha.
    sym_token, llm_token : int
        Token IDs for the first token of "SYMBOLIC" and "LLM".
    n_iterations : int
        Number of optimization iterations (for "random" method) or
        passes over the top coordinates (for "coordinate" method).
    learning_rate : float
        Fraction of the perturbation to apply when the margin improves.
    perturbation_scale : float
        Scale of the random perturbation relative to the direction's norm.
    seed : int
        Random seed.
    verbose : bool
        If True, print progress.
    method : str
        "random" or "coordinate".
    n_coordinates : int
        For "coordinate" method: number of top-magnitude coordinates to try.
    negative_prompts : list[str] | None
        v0.3.6: rendered routing prompts for non-symbolic control tasks
        (we want these to keep routing to "LLM"). When non-empty, the
        optimizer minimizes the balanced objective
        ``mean(positive_margin) - lambda * mean(hinge(negative_margin + gamma))
        - beta * ||v - v0||^2`` instead of the legacy positive-only margin.
        When ``None`` or empty, the legacy positive-only objective is used
        so existing callers and tests are unchanged.
    lambda_neg : float
        Weight on the negative-control hinge penalty. Default 1.5.
    gamma : float
        Negative-margin target threshold for the hinge. The penalty
        switches off once a control prompt's margin is ≤ ``-gamma``.
        Default 1.0.
    beta : float
        L2 anchor strength toward ``initial_direction``. Default 0.01.

    Returns
    -------
    direction : np.ndarray
        The optimized steering direction.
    info : dict
        Optimization info: initial_margin, final_margin, iterations, etc.
        v0.3.6 additionally reports ``initial_objective``, ``final_objective``,
        ``initial_negative_margin``, ``final_negative_margin``,
        ``initial_false_positive_rate``, ``final_false_positive_rate``,
        ``lambda_neg``, ``gamma``, ``beta``, ``n_negative_prompts``.
    """
    rng = np.random.RandomState(seed)
    direction = initial_direction.copy().astype(np.float32)
    v0 = initial_direction.astype(np.float32)
    initial_norm = float(np.linalg.norm(direction))
    neg_prompts = list(negative_prompts) if negative_prompts else []
    use_balanced = bool(neg_prompts)

    def _margins_for(prompts_list: list[str], cand: np.ndarray) -> list[float]:
        return [
            _get_logit_margin(model, tokenizer, p, layer, cand, alpha, sym_token, llm_token)
            for p in prompts_list
        ]

    # Evaluate initial margins.
    pos_margins = _margins_for(prompts, direction)
    neg_margins = _margins_for(neg_prompts, direction) if use_balanced else []
    current_pos = float(np.mean(pos_margins)) if pos_margins else 0.0
    current_neg = float(np.mean(neg_margins)) if neg_margins else 0.0

    if use_balanced:
        current_score = _balanced_objective(
            pos_margins, neg_margins, direction=direction, initial_direction=v0,
            lambda_neg=lambda_neg, gamma=gamma, beta=beta,
        )
    else:
        current_score = current_pos
    best_score = current_score

    # For backward-compatible reporting, ``margin`` always refers to the
    # positive-class mean margin (the legacy objective).
    best_margin = current_pos

    if verbose:
        print(f"  Initial margin: {current_pos:.4f}, |v|={initial_norm:.4f}")
        if use_balanced:
            print(f"  Initial neg-margin: {current_neg:.4f}, objective: {current_score:.4f}")

    def _false_positive_rate(neg_ms: list[float]) -> float:
        # Fraction of control prompts that would route to SYMBOLIC
        # (margin > 0). This is the metric the v0.3.6 repair targets.
        if not neg_ms:
            return 0.0
        return float(sum(1 for m in neg_ms if m > 0.0) / len(neg_ms))

    if method == "coordinate":
        # Identify the top-k coordinates by magnitude
        top_coords = np.argsort(np.abs(direction))[-n_coordinates:]
        delta = initial_norm * 0.05  # 5% of norm per step

        for iteration in range(n_iterations):
            improved_this_pass = False
            for coord in top_coords:
                for sign in (+1, -1):
                    candidate = direction.copy()
                    candidate[coord] += sign * delta
                    cand_pos = _margins_for(prompts, candidate)
                    cand_pos_mean = float(np.mean(cand_pos)) if cand_pos else 0.0
                    if use_balanced:
                        cand_neg = _margins_for(neg_prompts, candidate)
                        cand_score = _balanced_objective(
                            cand_pos, cand_neg, direction=candidate, initial_direction=v0,
                            lambda_neg=lambda_neg, gamma=gamma, beta=beta,
                        )
                    else:
                        cand_score = cand_pos_mean

                    if cand_score > best_score:
                        direction = candidate
                        best_score = cand_score
                        best_margin = cand_pos_mean
                        improved_this_pass = True
                        break  # move to next coordinate after an improvement

            if verbose:
                print(f"  iter {iteration}: score={best_score:.4f} "
                      f"({'improved' if improved_this_pass else 'no change'})")

            if not improved_this_pass:
                # Reduce delta and try again
                delta *= 0.5
                if delta < initial_norm * 0.001:
                    break

    else:  # "random"
        for iteration in range(n_iterations):
            # Generate a random perturbation
            perturbation = rng.standard_normal(direction.shape).astype(np.float32)
            perturbation *= perturbation_scale * initial_norm
            perturbation /= max(np.linalg.norm(perturbation), 1e-8) * (perturbation_scale * initial_norm)

            # Try the perturbed direction
            candidate = direction + learning_rate * perturbation
            cand_pos = _margins_for(prompts, candidate)
            cand_pos_mean = float(np.mean(cand_pos)) if cand_pos else 0.0
            if use_balanced:
                cand_neg = _margins_for(neg_prompts, candidate)
                cand_score = _balanced_objective(
                    cand_pos, cand_neg, direction=candidate, initial_direction=v0,
                    lambda_neg=lambda_neg, gamma=gamma, beta=beta,
                )
            else:
                cand_score = cand_pos_mean

            if cand_score > best_score:
                direction = candidate
                best_score = cand_score
                best_margin = cand_pos_mean
                if verbose:
                    print(f"  iter {iteration}: score improved to {best_score:.4f}")

    # Final margins for reporting.
    final_pos_margins = _margins_for(prompts, direction)
    final_neg_margins = _margins_for(neg_prompts, direction) if use_balanced else []
    final_pos = float(np.mean(final_pos_margins)) if final_pos_margins else 0.0
    final_neg = float(np.mean(final_neg_margins)) if final_neg_margins else 0.0
    if use_balanced:
        final_score = _balanced_objective(
            final_pos_margins, final_neg_margins, direction=direction, initial_direction=v0,
            lambda_neg=lambda_neg, gamma=gamma, beta=beta,
        )
    else:
        final_score = final_pos

    info: dict[str, Any] = {
        "initial_margin": current_pos,
        "final_margin": final_pos,
        "margin_improvement": final_pos - current_pos,
        "n_iterations": n_iterations,
        "initial_norm": initial_norm,
        "final_norm": float(np.linalg.norm(direction)),
        "n_prompts": len(prompts),
        "method": method,
        # v0.3.6 balanced-objective telemetry
        "n_negative_prompts": len(neg_prompts),
        "lambda_neg": float(lambda_neg) if use_balanced else None,
        "gamma": float(gamma) if use_balanced else None,
        "beta": float(beta) if use_balanced else None,
        "initial_objective": float(current_score),
        "final_objective": float(final_score),
        "objective_improvement": float(final_score - current_score),
        "initial_negative_margin": current_neg if use_balanced else None,
        "final_negative_margin": final_neg if use_balanced else None,
        "initial_false_positive_rate": _false_positive_rate(neg_margins),
        "final_false_positive_rate": _false_positive_rate(final_neg_margins),
    }

    return direction, info
