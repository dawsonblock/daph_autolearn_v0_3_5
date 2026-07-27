"""AutoLearn CLI: run the iterative steering-vector learning loop.

This is the script that gives the project its name. It runs the learning
loop described in ``daph_learning.autolearn``:

1. Routes training tasks with the current steering vector.
2. Executes the chosen backend.
3. Learns from misrouted tasks.
4. Updates the steering vector.
5. Evaluates on validation.
6. Repeats.

Usage:

    python scripts/autolearn.py \\
        --train-tasks data/train.jsonl \\
        --val-tasks data/val.jsonl \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --layer 24 \\
        --n-iterations 5 \\
        --output artifacts/autolearn_result.json \\
        --vector-output artifacts/autolearn_vector.npz

See CLAIMS.md §18 for what this loop does and does not establish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from daph_learning.autolearn import AutoLearnConfig, run_autolearn_loop
from daph_learning.steering.io import save_vector
from scripts.tune_steering import _load_llm, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the AutoLearn learning loop: iteratively improve a steering "
            "vector by learning from execution outcomes. See CLAIMS.md §18."
        )
    )
    ap.add_argument("--train-tasks", required=True, help="Training tasks JSONL")
    ap.add_argument("--val-tasks", required=True, help="Validation tasks JSONL")
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument("--layer", type=int, default=24, help="Transformer layer to steer")
    ap.add_argument("--alpha", type=float, default=1.0, help="Steering alpha")
    ap.add_argument("--anchor", default="ACTION", help="Anchor token")
    ap.add_argument("--n-iterations", type=int, default=5)
    ap.add_argument("--min-examples", type=int, default=4, help="Min examples per class for update")
    ap.add_argument("--normalization", choices=["l2", "none", "mean_centered"], default="l2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="raw")
    ap.add_argument(
        "--routing-mode",
        choices=["auto", "logit", "generate"],
        default="auto",
        help=(
            "Routing method: 'auto' tries logit-based, falls back to generate; "
            "'logit' forces logit (fails for multi-token tokenizers); "
            "'generate' forces generate-mode (slower, works with any tokenizer)."
        ),
    )
    ap.add_argument("--label-field", default="route_label")
    ap.add_argument(
        "--label-oracle-kind",
        choices=["capability", "accuracy", "utility", "policy_heuristic"],
        default="policy_heuristic",
    )
    ap.add_argument("--output", required=True, help="Output JSON with learning curve")
    ap.add_argument("--vector-output", help="Save the best steering vector to .npz")
    args = ap.parse_args()

    train_tasks = load_jsonl(Path(args.train_tasks))
    val_tasks = load_jsonl(Path(args.val_tasks))
    model, tokenizer = _load_llm(args.model)

    config = AutoLearnConfig(
        n_iterations=args.n_iterations,
        layer=args.layer,
        alpha=args.alpha,
        anchor=args.anchor,
        min_examples_per_update=args.min_examples,
        normalization=args.normalization,
        seed=args.seed,
    )

    result = run_autolearn_loop(
        train_tasks,
        val_tasks,
        model,
        tokenizer,
        config=config,
        prompt_format=args.prompt_format,
        label_field=args.label_field,
        label_oracle_kind=args.label_oracle_kind,
        routing_mode=args.routing_mode,
    )

    # Serialize result
    output = {
        "config": {
            "n_iterations": config.n_iterations,
            "layer": config.layer,
            "alpha": config.alpha,
            "anchor": config.anchor,
            "normalization": config.normalization,
            "seed": config.seed,
            "min_examples_per_update": config.min_examples_per_update,
        },
        "best_iteration": result.best_iteration,
        "best_val_f1": result.best_val_f1,
        "learning_curve": [
            {
                "iteration": m.iteration,
                "n_train": m.n_train,
                "n_correct": m.n_correct,
                "n_misrouted": m.n_misrouted,
                "n_unverifiable": m.n_unverifiable,
                "train_accuracy": m.train_accuracy,
                "val_f1": m.val_f1,
                "val_accuracy": m.val_accuracy,
                "val_precision": m.val_precision,
                "val_recall": m.val_recall,
                "vector_norm": m.vector_norm,
                "n_positive_examples": m.n_positive_examples,
                "n_negative_examples": m.n_negative_examples,
                "updated": m.updated,
            }
            for m in result.iterations
        ],
        "n_train_tasks": len(train_tasks),
        "n_val_tasks": len(val_tasks),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Save best vector
    if args.vector_output and result.best_vector is not None:
        save_vector(result.best_vector, Path(args.vector_output))
        print(f"Best vector saved to {args.vector_output}")

    # Print learning curve summary
    print(f"\nAutoLearn complete. {len(result.iterations)} iterations.")
    print(f"Best iteration: {result.best_iteration} (val F1={result.best_val_f1:.4f})")
    print("\nLearning curve:")
    print(f"{'iter':>4} {'train_acc':>10} {'val_f1':>10} {'val_acc':>10} {'updated':>8} {'|v|':>10}")
    for m in result.iterations:
        print(f"{m.iteration:4d} {m.train_accuracy:10.4f} {m.val_f1:10.4f} {m.val_accuracy:10.4f} {str(m.updated):>8} {m.vector_norm:10.4f}")
    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
