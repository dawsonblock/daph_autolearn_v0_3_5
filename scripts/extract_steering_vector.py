from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from daph_learning.steering.extract import contrastive_mean_direction
from daph_learning.steering.io import save_vector
from daph_learning.steering.types import SteeringSpec


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a contrastive steering direction from [N,H] activation arrays")
    ap.add_argument("--positive", required=True, help=".npy array [N,H]")
    ap.add_argument("--negative", required=True, help=".npy array [N,H]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vector-id", required=True)
    ap.add_argument("--family", choices=["reasoning_policy", "tool_policy"], required=True)
    ap.add_argument("--behavior", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--anchor", default="ACTION")
    ap.add_argument("--model-id")
    # --- capture provenance (v0.3.5 manifest patch) ---
    # These are optional at the CLI level for backward compatibility, but
    # a vector without them will fail headline manifest validation. See
    # CLAIMS.md and docs/RUN_MANIFEST.md.
    ap.add_argument(
        "--extraction-method",
        default="contrastive_mean_difference",
        help="Method used to derive the direction from positive/negative activations",
    )
    ap.add_argument(
        "--normalization",
        choices=["l2", "none", "mean_centered"],
        default="l2",
        help="Normalization applied to the direction after extraction",
    )
    ap.add_argument("--positive-n", type=int, help="Number of positive examples used")
    ap.add_argument("--negative-n", type=int, help="Number of negative examples used")
    ap.add_argument(
        "--capture-anchor",
        default="ACTION:",
        help="Textual anchor used during capture (e.g. ACTION:)",
    )
    ap.add_argument(
        "--capture-prompt-format",
        choices=["raw", "chat"],
        help="Prompt format used during capture",
    )
    ap.add_argument(
        "--capture-dataset-path",
        help="Path to the dataset the activations were captured on",
    )
    ap.add_argument(
        "--capture-dataset-sha256",
        help="SHA-256 of the capture dataset file. Required for headline-eligible manifests.",
    )
    args = ap.parse_args()

    pos = np.load(args.positive, allow_pickle=False)
    neg = np.load(args.negative, allow_pickle=False)
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError("positive/negative arrays must be [N,H] with matching H")

    # Auto-fill positive_n / negative_n from the arrays when not given.
    positive_n = args.positive_n if args.positive_n is not None else int(pos.shape[0])
    negative_n = args.negative_n if args.negative_n is not None else int(neg.shape[0])

    spec = SteeringSpec(
        vector_id=args.vector_id,
        family=args.family,
        behavior=args.behavior,
        layer=args.layer,
        alpha=args.alpha,
        anchor=args.anchor,
        model_id=args.model_id,
        hidden_size=int(pos.shape[1]),
        # Capture provenance:
        extraction_method=args.extraction_method,
        normalization=args.normalization,
        positive_n=positive_n,
        negative_n=negative_n,
        capture_anchor=args.capture_anchor,
        capture_prompt_format=args.capture_prompt_format,
        capture_dataset_path=args.capture_dataset_path,
        capture_dataset_sha256=args.capture_dataset_sha256,
    )
    vec = contrastive_mean_direction(pos, neg, spec)
    save_vector(vec, Path(args.output))
    print(f"saved {args.output} ({vec.values.size} dimensions)")


if __name__ == "__main__":
    main()
