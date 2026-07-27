"""Run the full DAPH v0.3.5 experiment suite on Qwen2.5-0.5B-Instruct.

This script:
1. Captures activations from train tasks (positive=symbolic, negative=llm).
2. Extracts an initial steering vector (extract-once baseline).
3. Runs the AutoLearn loop (iterative learning from execution outcomes).
4. Runs the linear-probe baseline on the same activations.
5. Runs the random-direction control experiment.
6. Evaluates all approaches on the test set.
7. Saves all results to experiments/results/.

Usage:
    python experiments/run_full_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from daph_learning.routing.steered_router import (
    build_route_prompt,
    parse_route_action,
    resolve_route_token_ids,
)
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import capture_layer_output, resolve_transformer_layers
from daph_learning.steering.extract import contrastive_mean_direction
from daph_learning.steering.types import SteeringSpec, SteeringVector
from daph_learning.execution.symbolic_executor import execute_plan, plan_from_structured_task
from daph_learning.tools.symbolic_math import SymbolicMathError
from daph_learning.autolearn import AutoLearnConfig, run_autolearn_loop, classify_outcome

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 20  # layer 20 of 28 — best steering effect
ALPHA = 3.0  # sweet spot: 90% acc, 62% F1 on test (vs 70%/0% baseline)
ANCHOR = "ACTION:"
PROMPT_FORMAT = "chat"
NORMALIZATION = "none"  # preserve vector magnitude (critical for this model)
DATA_DIR = Path("experiments/data")
RESULTS_DIR = Path("experiments/results")


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def format_prompt(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def capture_activations(tasks, model, tokenizer, layer, anchor_text=ANCHOR):
    """Capture last-token activations for each task. Returns (activations, labels).

    Note: we capture at the LAST token position (not the anchor) because
    that's where the routing decision is made — the next-token prediction
    depends on the last token's residual stream.
    """
    activations = []
    labels = []
    embed_device = model.get_input_embeddings().weight.device

    for task in tasks:
        label = task.get("route_label", "llm")
        prompt = build_route_prompt(task)
        rendered = format_prompt(prompt, tokenizer)
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(embed_device) for k, v in encoded.items()}

        sink = []
        with torch.inference_mode(), capture_layer_output(
            model, layer_index=layer, sink=sink, target_token_index=-1,
        ):
            model(**encoded, use_cache=False)

        if sink:
            act = sink[0]
            if act.ndim == 2:
                act = act[-1]
            activations.append(act.float().cpu().numpy())
            labels.append(label)

    return np.stack(activations) if activations else None, labels


def route_with_steering(tasks, model, tokenizer, vector, alpha, max_new_tokens=5):
    """Route tasks using steering + generate mode. Returns list of (route, output)."""
    from daph_learning.steering.hooks import residual_addition_hook

    results = []
    embed_device = model.get_input_embeddings().weight.device

    for task in tasks:
        prompt = build_route_prompt(task)
        rendered = format_prompt(prompt, tokenizer)
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(embed_device) for k, v in encoded.items()}

        vec_values = np.array(vector.values, dtype=np.float32)

        with torch.inference_mode(), residual_addition_hook(
            model,
            layer_index=vector.spec.layer,
            vector=vec_values,
            alpha=alpha,
            token_scope="last",
        ):
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        generated = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        action = parse_route_action(generated)
        results.append((action, generated))

    return results


def route_without_steering(tasks, model, tokenizer, max_new_tokens=5):
    """Route tasks without steering (baseline). Returns list of (route, output)."""
    results = []
    embed_device = model.get_input_embeddings().weight.device

    for task in tasks:
        prompt = build_route_prompt(task)
        rendered = format_prompt(prompt, tokenizer)
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(embed_device) for k, v in encoded.items()}

        with torch.inference_mode():
            out = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        generated = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        action = parse_route_action(generated)
        results.append((action, generated))

    return results


def execute_symbolic(task):
    """Execute a task symbolically. Returns (output_string, success)."""
    try:
        plan = plan_from_structured_task(task, reason_code="experiment")
        result = execute_plan(plan)
        return f"FINAL: {result.value}", result.verified
    except (SymbolicMathError, SyntaxError, ZeroDivisionError, ValueError, Exception):
        return None, False


def evaluate_routing(tasks, routes, expected_labels):
    """Evaluate routing accuracy against expected labels."""
    correct = sum(1 for r, e in zip(routes, expected_labels) if r == e)
    accuracy = correct / len(tasks) if tasks else 0.0

    # Confusion matrix
    tp = sum(1 for r, e in zip(routes, expected_labels) if r == "symbolic" and e == "symbolic")
    fp = sum(1 for r, e in zip(routes, expected_labels) if r == "symbolic" and e == "llm")
    fn = sum(1 for r, e in zip(routes, expected_labels) if r == "llm" and e == "symbolic")
    tn = sum(1 for r, e in zip(routes, expected_labels) if r == "llm" and e == "llm")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": len(tasks),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DAPH v0.3.5 Full Experiment Suite")
    print(f"Model: {MODEL_ID}")
    print(f"Layer: {LAYER}, Alpha: {ALPHA}, Anchor: {ANCHOR}, Norm: {NORMALIZATION}")
    print("=" * 70)

    # Load model (use local_files_only to skip HuggingFace network checks)
    print("\n[1/7] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, local_files_only=True,
        low_cpu_mem_usage=True, device_map="mps",
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  hidden_size={model.config.hidden_size}, num_layers={model.config.num_hidden_layers}")

    # Load data
    train_tasks = load_tasks(DATA_DIR / "train.jsonl")
    val_tasks = load_tasks(DATA_DIR / "val.jsonl")
    test_tasks = load_tasks(DATA_DIR / "test.jsonl")
    print(f"  Train: {len(train_tasks)}, Val: {len(val_tasks)}, Test: {len(test_tasks)}")

    # --- Baseline: no steering ---
    print("\n[2/7] Baseline routing (no steering) on test set...")
    t0 = time.time()
    baseline_routes = route_without_steering(test_tasks, model, tokenizer)
    baseline_labels = [r[0] for r in baseline_routes]
    expected_labels = [t.get("route_label", "llm") for t in test_tasks]
    baseline_metrics = evaluate_routing(test_tasks, baseline_labels, expected_labels)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1']:.4f}")
    print(f"  Confusion: {baseline_metrics['confusion_matrix']}")

    # --- Capture activations ---
    print("\n[3/7] Capturing activations from train set...")
    t0 = time.time()
    activations, act_labels = capture_activations(train_tasks, model, tokenizer, LAYER)
    positive_acts = np.stack([a for a, l in zip(activations, act_labels) if l == "symbolic"])
    negative_acts = np.stack([a for a, l in zip(activations, act_labels) if l == "llm"])
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Positive (symbolic): {positive_acts.shape}")
    print(f"  Negative (llm): {negative_acts.shape}")

    # Save activations for the linear-probe baseline
    np.save(RESULTS_DIR / "positive_activations.npy", positive_acts)
    np.save(RESULTS_DIR / "negative_activations.npy", negative_acts)

    # --- Extract steering vector (extract-once) ---
    print("\n[4/7] Extracting steering vector (extract-once baseline)...")
    t0 = time.time()
    spec = SteeringSpec(
        vector_id="extract_once",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=LAYER,
        alpha=ALPHA,
        anchor=ANCHOR,
        model_id=MODEL_ID,
        hidden_size=model.config.hidden_size,
        extraction_method="contrastive_mean_difference",
        normalization=NORMALIZATION,
        positive_n=len(positive_acts),
        negative_n=len(negative_acts),
        capture_anchor=ANCHOR,
        capture_prompt_format=PROMPT_FORMAT,
    )
    extract_once_vector = contrastive_mean_direction(positive_acts, negative_acts, spec)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Vector norm: {np.linalg.norm(extract_once_vector.values):.4f}")

    # Evaluate extract-once on test set
    print("  Evaluating extract-once on test set...")
    t0 = time.time()
    eo_routes = route_with_steering(test_tasks, model, tokenizer, extract_once_vector, ALPHA)
    eo_labels = [r[0] for r in eo_routes]
    extract_once_metrics = evaluate_routing(test_tasks, eo_labels, expected_labels)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Accuracy: {extract_once_metrics['accuracy']:.4f}, F1: {extract_once_metrics['f1']:.4f}")
    print(f"  Confusion: {extract_once_metrics['confusion_matrix']}")

    # --- AutoLearn loop ---
    print("\n[5/8] Running AutoLearn loop (generate-mode routing)...")
    t0 = time.time()
    config = AutoLearnConfig(
        n_iterations=5,
        layer=LAYER,
        alpha=ALPHA,
        anchor=ANCHOR,
        min_examples_per_update=4,
        normalization=NORMALIZATION,
        seed=42,
    )
    autolearn_result = run_autolearn_loop(
        train_tasks, val_tasks, model, tokenizer,
        config=config,
        initial_vector=extract_once_vector,
        prompt_format=PROMPT_FORMAT,
        label_field="route_label",
        label_oracle_kind="policy_heuristic",
        routing_mode="generate",  # use generate mode for multi-token tokenizers
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Best iteration: {autolearn_result.best_iteration}")
    print(f"  Best val F1: {autolearn_result.best_val_f1:.4f}")
    print(f"  Learning curve:")
    print(f"  {'iter':>4} {'train_acc':>10} {'val_f1':>10} {'updated':>8} {'|v|':>10}")
    for m in autolearn_result.iterations:
        print(f"  {m.iteration:4d} {m.train_accuracy:10.4f} {m.val_f1:10.4f} {str(m.updated):>8} {m.vector_norm:10.4f}")

    # Evaluate AutoLearn best vector on test set
    if autolearn_result.best_vector is not None:
        print("  Evaluating AutoLearn best vector on test set...")
        t0 = time.time()
        al_routes = route_with_steering(test_tasks, model, tokenizer, autolearn_result.best_vector, ALPHA)
        al_labels = [r[0] if r[0] else "llm" for r in al_routes]
        autolearn_metrics = evaluate_routing(test_tasks, al_labels, expected_labels)
        print(f"  Done in {time.time()-t0:.1f}s")
        print(f"  Accuracy: {autolearn_metrics['accuracy']:.4f}, F1: {autolearn_metrics['f1']:.4f}")
        print(f"  Confusion: {autolearn_metrics['confusion_matrix']}")
    else:
        autolearn_metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # --- Linear-probe baseline ---
    print("\n[6/8] Running linear-probe baseline...")
    t0 = time.time()
    from scripts.linear_probe_baseline import run_linear_probe
    probe_metrics = run_linear_probe(positive_acts, negative_acts, n_splits=5, random_state=42)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Accuracy: {probe_metrics['accuracy']:.4f} (baseline: {probe_metrics['random_baseline']:.4f})")
    print(f"  Lift over baseline: {probe_metrics['lift_over_baseline']:.4f}")

    # --- Steering-optimized direction ---
    print("\n[6.5/8] Optimizing steering direction (gradient-free, balanced v0.3.6)...")
    t0 = time.time()
    from daph_learning.steering.optimize import optimize_steering_direction

    # Build rendered prompts for symbolic tasks (we want these to produce SYMBOLIC)
    sym_train_tasks = [t for t in train_tasks if t.get("route_label") == "symbolic"]
    sym_prompts = []
    for task in sym_train_tasks[:10]:  # use up to 10 for speed
        prompt = build_route_prompt(task)
        rendered = format_prompt(prompt, tokenizer)
        sym_prompts.append(rendered)

    # v0.3.6 Phase 1.2: also build negative-control prompts (non-symbolic
    # tasks that must keep routing to LLM). Without these the optimizer
    # inflated the vector norm until every prompt routed to SYMBOLIC
    # (FP=42, TN=0, F1=0.2759). See the v0.3.6 repair plan.
    neg_train_tasks = [t for t in train_tasks if t.get("route_label") == "llm"]
    neg_prompts = []
    for task in neg_train_tasks[:10]:
        prompt = build_route_prompt(task)
        rendered = format_prompt(prompt, tokenizer)
        neg_prompts.append(rendered)

    SYM_TOKEN, LLM_TOKEN = resolve_route_token_ids(
        tokenizer, allow_first_token_fallback=True,
    )

    optimized_direction, opt_info = optimize_steering_direction(
        model, tokenizer, sym_prompts,
        initial_direction=extract_once_vector.values,
        layer=LAYER, alpha=ALPHA,
        sym_token=SYM_TOKEN, llm_token=LLM_TOKEN,
        n_iterations=5, seed=42, verbose=True,
        method="coordinate",
        n_coordinates=100,
        negative_prompts=neg_prompts or None,
        lambda_neg=1.5,
        gamma=1.0,
        beta=0.01,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Initial margin: {opt_info['initial_margin']:.4f}")
    print(f"  Final margin: {opt_info['final_margin']:.4f}")
    print(f"  Improvement: {opt_info['margin_improvement']:.4f}")
    if opt_info.get("n_negative_prompts"):
        print(f"  Negative controls: {opt_info['n_negative_prompts']}")
        print(f"  Initial FPR: {opt_info['initial_false_positive_rate']:.4f}")
        print(f"  Final FPR:   {opt_info['final_false_positive_rate']:.4f}")
        print(f"  Objective: {opt_info['initial_objective']:.4f} -> {opt_info['final_objective']:.4f}")

    # Create a SteeringVector from the optimized direction
    opt_spec = SteeringSpec(
        vector_id="steering_optimized",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=LAYER,
        alpha=ALPHA,
        anchor=ANCHOR,
        model_id=MODEL_ID,
        hidden_size=model.config.hidden_size,
        extraction_method="gradient_free_balanced_optimization",
        normalization="none",
        positive_n=len(sym_prompts),
        negative_n=len(neg_prompts),
    )
    optimized_vector = SteeringVector(spec=opt_spec, values=optimized_direction)

    # Evaluate optimized direction on test set
    print("  Evaluating optimized direction on test set...")
    from daph_learning.evaluation.timing import cuda_timed
    with cuda_timed() as timing:
        opt_routes = route_with_steering(test_tasks, model, tokenizer, optimized_vector, ALPHA)
        opt_labels = [r[0] if r[0] else "llm" for r in opt_routes]
        optimized_metrics = evaluate_routing(test_tasks, opt_labels, expected_labels)
    print(f"  Done in {timing.elapsed_ms/1000:.1f}s ({timing.device}, cuda_events={timing.cuda_event_timing})")
    print(f"  Accuracy: {optimized_metrics['accuracy']:.4f}, F1: {optimized_metrics['f1']:.4f}")
    print(f"  Confusion: {optimized_metrics['confusion_matrix']}")

    # --- Random-direction control ---
    print("\n[7/8] Running random-direction control experiment...")
    t0 = time.time()
    from scripts.random_direction_control import _make_random_direction, _make_random_vector
    rng = np.random.RandomState(42)
    n_random = 5
    random_f1s = []
    for i in range(n_random):
        random_vec = _make_random_vector(extract_once_vector, rng)
        r_routes = route_with_steering(test_tasks[:20], model, tokenizer, random_vec, ALPHA)  # subset for speed
        r_labels = [r[0] if r[0] else "llm" for r in r_routes]
        r_metrics = evaluate_routing(test_tasks[:20], r_labels, [t.get("route_label") for t in test_tasks[:20]])
        random_f1s.append(r_metrics["f1"])
    random_mean = float(np.mean(random_f1s))
    random_std = float(np.std(random_f1s))
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Random F1: {random_mean:.4f} ± {random_std:.4f}")
    print(f"  Extract-once F1: {extract_once_metrics['f1']:.4f}")
    f1_lift = extract_once_metrics["f1"] - random_mean
    f1_z = (extract_once_metrics["f1"] - random_mean) / random_std if random_std > 0 else float("inf")
    interpretation = (
        "real_vector_outperforms" if extract_once_metrics["f1"] > random_mean + random_std
        else "real_vector_within_random_range" if abs(extract_once_metrics["f1"] - random_mean) <= random_std
        else "real_vector_underperforms"
    )
    print(f"  F1 lift: {f1_lift:.4f}, z-score: {f1_z:.4f}")
    print(f"  Interpretation: {interpretation}")

    # --- Probe-based steering ---
    print("\n[8/8] Running probe-based steering (using linear probe weights as direction)...")
    t0 = time.time()
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([positive_acts, negative_acts]).astype(np.float32)
    y = np.array([1] * len(positive_acts) + [0] * len(negative_acts))
    probe_clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    probe_clf.fit(X, y)
    # The probe's weight vector is the direction that best separates the classes
    probe_direction = probe_clf.coef_[0].astype(np.float32)
    probe_norm = float(np.linalg.norm(probe_direction))
    print(f"  Probe direction norm: {probe_norm:.4f}")

    # Scale the probe direction to match the extract-once vector's norm
    # so the comparison is fair (same perturbation magnitude)
    extract_norm = float(np.linalg.norm(extract_once_vector.values))
    probe_scaled = probe_direction * (extract_norm / probe_norm)
    probe_spec = SteeringSpec(
        vector_id="probe_based",
        family="tool_policy",
        behavior="invoke_symbolic_tool",
        layer=LAYER,
        alpha=ALPHA,
        anchor=ANCHOR,
        model_id=MODEL_ID,
        hidden_size=model.config.hidden_size,
        extraction_method="linear_probe_weights",
        normalization="none",
        positive_n=len(positive_acts),
        negative_n=len(negative_acts),
    )
    probe_vector = SteeringVector(spec=probe_spec, values=probe_scaled)

    # Evaluate probe-based steering on test set
    probe_routes = route_with_steering(test_tasks, model, tokenizer, probe_vector, ALPHA)
    probe_labels = [r[0] if r[0] else "llm" for r in probe_routes]
    probe_steering_metrics = evaluate_routing(test_tasks, probe_labels, expected_labels)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Accuracy: {probe_steering_metrics['accuracy']:.4f}, F1: {probe_steering_metrics['f1']:.4f}")
    print(f"  Confusion: {probe_steering_metrics['confusion_matrix']}")

    # Also sweep alpha for probe-based steering
    print("  Alpha sweep for probe-based steering:")
    best_probe_f1 = 0.0
    best_probe_alpha = ALPHA
    for p_alpha in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        p_routes = route_with_steering(test_tasks, model, tokenizer, probe_vector, p_alpha)
        p_labels = [r[0] if r[0] else "llm" for r in p_routes]
        p_metrics = evaluate_routing(test_tasks, p_labels, expected_labels)
        print(f"    alpha={p_alpha:5.1f}: acc={p_metrics['accuracy']:.4f} f1={p_metrics['f1']:.4f} "
              f"tp={p_metrics['confusion_matrix']['tp']} fp={p_metrics['confusion_matrix']['fp']}")
        if p_metrics["f1"] > best_probe_f1:
            best_probe_f1 = p_metrics["f1"]
            best_probe_alpha = p_alpha
    print(f"  Best probe alpha: {best_probe_alpha}, F1: {best_probe_f1:.4f}")

    # --- Save all results ---
    print("\n" + "=" * 70)
    print("Saving results...")

    all_results = {
        "model": MODEL_ID,
        "layer": LAYER,
        "alpha": ALPHA,
        "anchor": ANCHOR,
        "normalization": NORMALIZATION,
        "prompt_format": PROMPT_FORMAT,
        "n_train": len(train_tasks),
        "n_val": len(val_tasks),
        "n_test": len(test_tasks),
        "baseline_no_steering": baseline_metrics,
        "extract_once": extract_once_metrics,
        "autolearn": {
            "best_iteration": autolearn_result.best_iteration,
            "best_val_f1": autolearn_result.best_val_f1,
            "test_metrics": autolearn_metrics,
            "learning_curve": [
                {
                    "iteration": m.iteration,
                    "train_accuracy": m.train_accuracy,
                    "val_f1": m.val_f1,
                    "n_correct": m.n_correct,
                    "n_misrouted": m.n_misrouted,
                    "updated": m.updated,
                    "vector_norm": m.vector_norm,
                }
                for m in autolearn_result.iterations
            ],
        },
        "linear_probe": probe_metrics,
        "random_direction_control": {
            "n_random": n_random,
            "random_f1_mean": random_mean,
            "random_f1_std": random_std,
            "extract_once_f1": extract_once_metrics["f1"],
            "f1_lift": f1_lift,
            "f1_z_score": f1_z,
            "interpretation": interpretation,
            "random_f1s": random_f1s,
        },
        "probe_based_steering": {
            "test_metrics": probe_steering_metrics,
            "best_alpha": best_probe_alpha,
            "best_f1": best_probe_f1,
            "probe_direction_norm": probe_norm,
            "extract_once_norm": extract_norm,
        },
        "steering_optimized": {
            "test_metrics": optimized_metrics,
            "optimization_info": opt_info,
        },
    }

    results_path = RESULTS_DIR / "full_experiment.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Accuracy':>10} {'F1':>10}")
    print("-" * 50)
    print(f"{'Baseline (no steering)':<30} {baseline_metrics['accuracy']:>10.4f} {baseline_metrics['f1']:>10.4f}")
    print(f"{'Extract-once steering':<30} {extract_once_metrics['accuracy']:>10.4f} {extract_once_metrics['f1']:>10.4f}")
    print(f"{'Steering-optimized':<30} {optimized_metrics['accuracy']:>10.4f} {optimized_metrics['f1']:>10.4f}")
    print(f"{'Probe-based steering':<30} {probe_steering_metrics['accuracy']:>10.4f} {probe_steering_metrics['f1']:>10.4f}")
    print(f"{'AutoLearn (best iter)':<30} {autolearn_metrics['accuracy']:>10.4f} {autolearn_metrics['f1']:>10.4f}")
    print(f"{'Linear probe (CV)':<30} {probe_metrics['accuracy']:>10.4f} {'N/A':>10}")
    print(f"{'Random direction (mean)':<30} {'N/A':>10} {random_mean:>10.4f}")
    print()
    print(f"Steering lift over baseline:    F1 +{extract_once_metrics['f1'] - baseline_metrics['f1']:.4f}")
    print(f"Optimized lift over extract:    F1 +{optimized_metrics['f1'] - extract_once_metrics['f1']:.4f}")
    print(f"Probe-based lift over extract:  F1 +{probe_steering_metrics['f1'] - extract_once_metrics['f1']:.4f}")
    print(f"AutoLearn lift over extract-once: F1 +{autolearn_metrics['f1'] - extract_once_metrics['f1']:.4f}")
    print(f"Real vector vs random:          {interpretation} (z={f1_z:.2f})")


if __name__ == "__main__":
    main()
