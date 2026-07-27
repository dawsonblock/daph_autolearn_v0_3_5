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
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any

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

    Returns
    -------
    direction : np.ndarray
        The optimized steering direction.
    info : dict
        Optimization info: initial_margin, final_margin, iterations, etc.
    """
    rng = np.random.RandomState(seed)
    direction = initial_direction.copy().astype(np.float32)
    initial_norm = float(np.linalg.norm(direction))

    # Evaluate initial margin
    margins = [
        _get_logit_margin(model, tokenizer, p, layer, direction, alpha, sym_token, llm_token)
        for p in prompts
    ]
    current_margin = float(np.mean(margins))
    best_margin = current_margin

    if verbose:
        print(f"  Initial margin: {current_margin:.4f}, |v|={initial_norm:.4f}")

    if method == "coordinate":
        # Identify the top-k coordinates by magnitude
        top_coords = np.argsort(np.abs(direction))[-n_coordinates:]
        delta = initial_norm * 0.05  # 5% of norm per step

        for iteration in range(n_iterations):
            improved_this_pass = False
            for coord in top_coords:
                # Try +delta
                candidate = direction.copy()
                candidate[coord] += delta
                candidate_margins = [
                    _get_logit_margin(model, tokenizer, p, layer, candidate, alpha, sym_token, llm_token)
                    for p in prompts
                ]
                candidate_margin = float(np.mean(candidate_margins))

                if candidate_margin > best_margin:
                    direction = candidate
                    best_margin = candidate_margin
                    improved_this_pass = True
                    continue

                # Try -delta
                candidate = direction.copy()
                candidate[coord] -= delta
                candidate_margins = [
                    _get_logit_margin(model, tokenizer, p, layer, candidate, alpha, sym_token, llm_token)
                    for p in prompts
                ]
                candidate_margin = float(np.mean(candidate_margins))

                if candidate_margin > best_margin:
                    direction = candidate
                    best_margin = candidate_margin
                    improved_this_pass = True

            if verbose:
                print(f"  iter {iteration}: margin={best_margin:.4f} "
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
            candidate_margins = [
                _get_logit_margin(model, tokenizer, p, layer, candidate, alpha, sym_token, llm_token)
                for p in prompts
            ]
            candidate_margin = float(np.mean(candidate_margins))

            if candidate_margin > best_margin:
                direction = candidate
                best_margin = candidate_margin
                if verbose:
                    print(f"  iter {iteration}: margin improved to {best_margin:.4f}")

    info = {
        "initial_margin": current_margin,
        "final_margin": best_margin,
        "margin_improvement": best_margin - current_margin,
        "n_iterations": n_iterations,
        "initial_norm": initial_norm,
        "final_norm": float(np.linalg.norm(direction)),
        "n_prompts": len(prompts),
        "method": method,
    }

    return direction, info
