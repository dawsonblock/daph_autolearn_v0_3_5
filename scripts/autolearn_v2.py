#!/usr/bin/env python
"""AutoLearn v2 execution script.

Runs the AutoLearn v2 learning loop on a dataset. Supports:

* leakage-free family-aware train/val/test splitting
* counterfactual experience collection
* reward-gap-driven policy updates with trust-region
* acceptance gating with domain regression checks
* immutable policy lineage + rollback
* atomic checkpointing
* JSONL iteration telemetry
* final test-set evaluation (held-out, evaluated once)
* reproducibility manifest

Usage::

    python scripts/autolearn_v2.py \\
        --dataset data/my_dataset.jsonl \\
        --model-id gpt2 \\
        --layer 6 \\
        --max-iterations 20 \\
        --output-dir runs/v2_run_001

For real-model runs, supply ``--model-name`` and ``--tokenizer-name`` to
use actual transformer activations. Without those, a deterministic
pseudo-representation is used (for testing and synthetic data).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from daph_learning.autolearn_v2 import (
    AcceptanceConfig,
    AutoLearnV2Config,
    CounterfactualConfig,
    ReplayConfig,
    UpdateConfig,
    UtilityConfig,
    run_autolearn_v2,
)
from daph_learning.evaluation.leakage import detect_leakage, family_aware_split


def load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset (one task dict per line)."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def make_representation_fn(model_name: str | None, tokenizer_name: str | None, layer: int):
    """Create a representation capture function.

    If model_name and tokenizer_name are supplied, uses real transformer
    activations. Otherwise returns None (the engine will use a
    deterministic pseudo-representation).
    """
    if model_name is None or tokenizer_name is None:
        return None

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    hidden_size = model.config.hidden_size

    # Determine the target layer module.
    layers = model.transformer.h if hasattr(model, "transformer") else model.model.layers
    target_layer = layers[layer]

    activations = {}

    def hook_fn(_module, _input, output):
        # output is (hidden_states, ...) for some models; take first.
        if isinstance(output, tuple):
            output = output[0]
        activations["h"] = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)

    def rep_fn(task):
        prompt = str(task.get("specification", ""))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            model(**inputs)
        h = activations.get("h")
        if h is None:
            return torch.zeros(hidden_size, dtype=torch.float32).numpy()
        # Mean over sequence positions, squeeze batch.
        return h.mean(dim=1).squeeze(0).to(torch.float32).numpy()

    return rep_fn, hidden_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AutoLearn v2 learning loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--output-dir", required=True, help="Output directory for results.")
    parser.add_argument("--model-id", default="unknown", help="Model identifier for provenance.")
    parser.add_argument("--tokenizer-hash", default="unknown", help="Tokenizer hash for provenance.")
    parser.add_argument("--model-name", default=None, help="HuggingFace model name (for real activations).")
    parser.add_argument("--tokenizer-name", default=None, help="HuggingFace tokenizer name.")
    parser.add_argument("--layer", type=int, default=6, help="Layer to capture activations from.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering alpha.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Routing threshold.")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum learning iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Update learning rate.")
    parser.add_argument("--trust-region-radius", type=float, default=2.0, help="Trust-region radius.")
    parser.add_argument("--max-vector-norm", type=float, default=10.0, help="Maximum vector norm.")
    parser.add_argument("--minimum-utility-gain", type=float, default=0.01, help="Minimum utility gain for acceptance.")
    parser.add_argument("--minimum-samples", type=int, default=50, help="Minimum validation samples.")
    parser.add_argument("--max-domain-regression", type=float, default=0.05, help="Maximum per-domain regression.")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and split the dataset.
    tasks = load_dataset(args.dataset)
    print(f"Loaded {len(tasks)} tasks from {args.dataset}")

    train, val, test = family_aware_split(tasks, seed=args.seed)
    print(f"Family-aware split: train={len(train)} val={len(val)} test={len(test)}")

    # Check for leakage.
    splits = {"train": train, "val": val, "test": test}
    report = detect_leakage(splits)
    if report.has_leak:
        print(f"WARNING: leakage detected: {report.summary()}")
        print("Consider fixing the dataset before proceeding.")
    else:
        print("No leakage detected.")

    # Set up representation function.
    rep_fn = None
    hidden_size = 0
    if args.model_name and args.tokenizer_name:
        print(f"Loading model {args.model_name} for activation capture at layer {args.layer}...")
        rep_fn, hidden_size = make_representation_fn(args.model_name, args.tokenizer_name, args.layer)
        print(f"Model hidden size: {hidden_size}")
    else:
        print("No model specified; using deterministic pseudo-representation.")
        hidden_size = 64  # default for pseudo-representation

    # Build config.
    config = AutoLearnV2Config(
        seed=args.seed,
        max_iterations=args.max_iterations,
        model_id=args.model_id,
        tokenizer_hash=args.tokenizer_hash,
        dataset_id=os.path.basename(args.dataset),
        layer=args.layer,
        hidden_size=hidden_size,
        alpha=args.alpha,
        threshold=args.threshold,
        execution=CounterfactualConfig(execution_mode="counterfactual_training"),
        reward=UtilityConfig(abstain_margin=0.05),
        replay=ReplayConfig(
            capacity=500,
            batch_size=args.batch_size,
            seed=args.seed,
            hard_example_fraction=0.3,
            regression_anchor_fraction=0.1,
        ),
        update=UpdateConfig(
            learning_rate=args.learning_rate,
            trust_region_radius=args.trust_region_radius,
            max_vector_norm=args.max_vector_norm,
        ),
        acceptance=AcceptanceConfig(
            minimum_utility_gain=args.minimum_utility_gain,
            minimum_samples=min(args.minimum_samples, len(val)),
            maximum_domain_regression=args.max_domain_regression,
        ),
        checkpoint_path=os.path.join(args.output_dir, "checkpoint.json"),
        telemetry_path=os.path.join(args.output_dir, "telemetry.jsonl"),
        registry_dir=os.path.join(args.output_dir, "registry"),
    )

    # Run.
    print(f"\nStarting AutoLearn v2 (max {args.max_iterations} iterations)...")
    result = run_autolearn_v2(
        train, val,
        test_tasks=test,
        config=config,
        representation_fn=rep_fn,
        resume_from=args.resume_from,
    )

    # Report.
    print(f"\n=== AutoLearn v2 Complete ===")
    print(f"Final active policy: {result.active_policy.policy_id}")
    print(f"Vector norm: {result.active_policy.vector_norm():.4f}")
    print(f"Iterations run: {len(result.iterations)}")
    accepted = sum(1 for it in result.iterations if it.candidate_accepted)
    rejected = len(result.iterations) - accepted
    print(f"Accepted: {accepted}  Rejected: {rejected}")
    if result.final_test_metrics:
        m = result.final_test_metrics
        print(f"\nFinal test metrics:")
        print(f"  Utility: {m.utility:.4f}")
        print(f"  Routing accuracy: {m.routing_accuracy:.4f}")
        print(f"  Symbolic rate: {m.symbolic_rate:.2f}")
        print(f"  LLM rate: {m.llm_rate:.2f}")
        print(f"  Abstain rate: {m.abstain_rate:.2f}")
        if m.per_domain:
            print(f"  Per-domain utility:")
            for domain, u in sorted(m.per_domain.items()):
                print(f"    {domain}: {u:.4f}")

    # Save manifest.
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result.manifest, f, indent=2, sort_keys=True)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Telemetry saved to {config.telemetry_path}")
    print(f"Registry saved to {config.registry_dir}")
    print(f"Checkpoint saved to {config.checkpoint_path}")


if __name__ == "__main__":
    main()
