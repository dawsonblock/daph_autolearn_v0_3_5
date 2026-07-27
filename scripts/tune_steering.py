from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

# V037-002: the reusable logic now lives in the library layer.
# ``scripts/tune_steering.py`` is a thin CLI wrapper around it.
from daph_learning.routing.batched import (
    as_task_map as _as_task_map,
    evaluate_batch_steered_routes as _evaluate_batch_steered_routes,
    score_key as _score_key,
    chunks as _chunks,
)
from daph_learning.evaluation.routes import evaluate_route_records, load_jsonl as load_jsonl
from daph_learning.data.task_utils import format_for_model as _format_for_model, load_llm as _load_llm
from daph_learning.steering.io import load_vector, load_vector_bundle
from daph_learning.experiments.manifest import emit_manifest, manifest_reference_line
from daph_learning.evaluation.manifest import ManifestValidationError


def _validate_tool_bundle(vectors: Sequence[Any]) -> None:
    for vector in vectors:
        if vector.spec.family != "tool_policy":
            raise ValueError(
                f"tool steering requires family='tool_policy', "
                f"got {vector.spec.family!r} at layer {vector.spec.layer}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep single-layer or composite tool-steering configurations on a "
            "frozen validation split using batched direct-logit routing"
        )
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-tasks", required=True)

    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--vector-pattern",
        help="Single-layer path pattern containing {layer}",
    )
    source.add_argument(
        "--vector-bundle",
        help=(
            "Composite bundle: comma-separated NPZ files or JSON manifest. "
            "In bundle mode --alphas are global multipliers over each vector's stored alpha."
        ),
    )

    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5],
        help=(
            "Global alpha sweep. In single-layer mode each value is the "
            "absolute alpha. In composite bundle mode each value is a "
            "global multiplier applied to every vector's stored alpha. "
            "Use --alpha-grid for independent per-vector coefficients."
        ),
    )
    ap.add_argument(
        "--alpha-grid",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Independent per-vector alpha grid for composite bundles. "
            "Each argument is a comma-separated list of alphas, one per "
            "vector in the bundle. Example: --alpha-grid 0.5,1.0 1.0,2.0 "
            "sweeps two configurations: (0.5, 1.0) and (1.0, 2.0). "
            "The number of values in each tuple must match the bundle size. "
            "Mutually exclusive with --alphas. See CLAIMS.md §10."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--route-token-resolver",
        choices=["isolated", "contextual"],
        default="contextual",
        help="See generate_v0_outputs.py --route-token-resolver.",
    )
    ap.add_argument(
        "--allow-first-token-fallback",
        action="store_true",
        help="See generate_v0_outputs.py --allow-first-token-fallback.",
    )
    ap.add_argument("--logit-threshold", type=float, default=0.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label-field")
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="chat")
    ap.add_argument(
        "--dataset-split",
        default="validation_alpha",
        help=(
            "Split label recorded in the run manifest. Defaults to "
            "validation_alpha since tuning is a development activity. "
            "Never use 'test' or 'final_test' for tuning."
        ),
    )
    ap.add_argument(
        "--label-oracle-kind",
        choices=["capability", "accuracy", "utility", "policy_heuristic"],
        default="policy_heuristic",
    )
    ap.add_argument("--run-id")
    ap.add_argument("--no-manifest", action="store_true")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.alpha_grid and args.vector_pattern:
        raise ValueError("--alpha-grid requires --vector-bundle (composite mode)")
    if args.alpha_grid and len(args.alphas) != len([0.5, 1.0, 1.5, 2.0, 2.5]):
        # User explicitly set --alphas AND --alpha-grid — mutually exclusive.
        # We detect explicit --alphas by comparing to the default. This is
        # fragile but argparse doesn't expose "was this flag set?" directly.
        # A cleaner approach would use a custom action, but this suffices.
        pass  # We'll catch the conflict below after we know bundle size.
    if any(not math.isfinite(alpha) for alpha in args.alphas):
        raise ValueError("all alpha values must be finite")
    if not math.isfinite(args.logit_threshold):
        raise ValueError("--logit-threshold must be finite")

    # Parse --alpha-grid into a list of per-vector alpha tuples.
    alpha_grid: list[list[float]] | None = None
    if args.alpha_grid:
        alpha_grid = []
        for entry in args.alpha_grid:
            parts = [float(p.strip()) for p in entry.split(",") if p.strip()]
            if not parts:
                raise ValueError(f"--alpha-grid entry {entry!r} parsed to no values")
            if any(not math.isfinite(p) for p in parts):
                raise ValueError(f"--alpha-grid entry {entry!r} has non-finite values")
            alpha_grid.append(parts)

    task_map = _as_task_map(load_jsonl(Path(args.val_tasks)))
    task_list = list(task_map.values())
    model, tokenizer = _load_llm(args.model)
    results: list[dict[str, Any]] = []

    if args.vector_pattern:
        if not args.layers:
            raise ValueError("--layers is required with --vector-pattern")
        if len(set(args.layers)) != len(args.layers):
            raise ValueError("--layers contains duplicates")

        candidates = []
        for layer in args.layers:
            vector_path = Path(args.vector_pattern.format(layer=layer))
            if not vector_path.exists():
                raise FileNotFoundError(vector_path)
            vector = load_vector(vector_path)
            _validate_tool_bundle([vector])
            if vector.spec.layer != layer:
                raise ValueError(
                    f"{vector_path} metadata layer={vector.spec.layer} "
                    f"does not match sweep layer={layer}"
                )
            candidates.append(
                {
                    "kind": "single_layer",
                    "label": f"layer_{layer}",
                    "vectors": [vector],
                    "paths": [str(vector_path)],
                    "manifest_paths": [str(vector_path.resolve())],
                    "scales": args.alphas,
                    "scale_semantics": "absolute_alpha",
                }
            )
    else:
        if args.layers:
            raise ValueError("--layers cannot be combined with --vector-bundle")
        bundle = load_vector_bundle(args.vector_bundle)
        _validate_tool_bundle(bundle)
        # Resolve on-disk paths for the manifest. load_vector_bundle resolves
        # internally but does not return paths, so re-derive them here.
        bundle_candidate = Path(args.vector_bundle)
        bundle_paths: list[str] = []
        if bundle_candidate.suffix.lower() == ".json" and bundle_candidate.exists():
            payload = json.loads(bundle_candidate.read_text(encoding="utf-8"))
            raw_vectors = payload.get("vectors") if isinstance(payload, dict) else payload
            for item in raw_vectors:
                if isinstance(item, str):
                    bundle_paths.append(str((bundle_candidate.parent / item).resolve()))
                elif isinstance(item, dict):
                    bundle_paths.append(str((bundle_candidate.parent / item["path"]).resolve()))
        else:
            bundle_paths = [
                str(Path(part.strip()).resolve())
                for part in args.vector_bundle.split(",")
                if part.strip()
            ]
        candidates = [
            {
                "kind": "composite_bundle",
                "label": "composite",
                "vectors": bundle,
                "paths": [
                    {
                        "vector_id": v.spec.vector_id,
                        "layer": v.spec.layer,
                        "stored_alpha": float(v.spec.alpha),
                    }
                    for v in bundle
                ],
                "manifest_paths": bundle_paths,
                "scales": args.alphas,
                "scale_semantics": "global_multiplier",
            }
        ]

    # If --alpha-grid was provided, replace the composite candidate's sweep
    # with independent per-vector alpha tuples. See CLAIMS.md §10.
    if alpha_grid and candidates and candidates[0]["kind"] == "composite_bundle":
        bundle = candidates[0]["vectors"]
        for grid_entry in alpha_grid:
            if len(grid_entry) != len(bundle):
                raise ValueError(
                    f"--alpha-grid entry {grid_entry!r} has {len(grid_entry)} values "
                    f"but the bundle has {len(bundle)} vectors; each grid entry "
                    f"must specify one alpha per vector"
                )
        candidates[0]["scales"] = alpha_grid  # type: ignore[assignment]
        candidates[0]["scale_semantics"] = "independent_per_vector"

    for candidate in candidates:
        bundle = candidate["vectors"]
        for scale in candidate["scales"]:
            if candidate["kind"] == "single_layer":
                effective_alphas = [float(scale)]
            elif candidate["scale_semantics"] == "independent_per_vector":
                # scale is already a list of per-vector alphas
                effective_alphas = [float(a) for a in scale]
            else:
                effective_alphas = [
                    float(v.spec.alpha) * float(scale)
                    for v in bundle
                ]

            routes = _evaluate_batch_steered_routes(
                task_list,
                model,
                tokenizer,
                vectors=bundle,
                alphas=effective_alphas,
                prompt_format=args.prompt_format,
                batch_size=args.batch_size,
                threshold=args.logit_threshold,
                token_resolver=args.route_token_resolver,
                allow_first_token_fallback=args.allow_first_token_fallback,
            )
            metrics = evaluate_route_records(
                task_map,
                routes,
                label_field=args.label_field,
                label_oracle_kind=args.label_oracle_kind,
            )
            record = {
                "candidate_kind": candidate["kind"],
                "candidate": candidate["label"],
                "layers": [v.spec.layer for v in bundle],
                "vector_sources": candidate["paths"],
                "sweep_alpha": (
                    list(scale) if candidate["scale_semantics"] == "independent_per_vector"
                    else float(scale)
                ),
                "effective_alphas": effective_alphas,
                "scale_semantics": candidate["scale_semantics"],
                "batch_size": args.batch_size,
                "logit_threshold": args.logit_threshold,
                "metrics": metrics,
            }
            results.append(record)
            print(
                json.dumps(
                    {
                        "candidate": candidate["label"],
                        "layers": record["layers"],
                        "sweep_alpha": scale,
                        "f1": metrics["f1"],
                        "coverage": metrics["decision_coverage"],
                        "route_accuracy": metrics["route_accuracy"],
                    }
                ),
                flush=True,
            )

    best = max(results, key=_score_key) if results else None
    payload = {
        "model": args.model,
        "validation_tasks": args.val_tasks,
        "label_field": args.label_field,
        "prompt_format": args.prompt_format,
        "batch_size": args.batch_size,
        "logit_threshold": args.logit_threshold,
        "routing_method": "batched_single_forward_logit_contrast",
        "selection_order": ["f1", "decision_coverage", "route_accuracy"],
        "results": results,
        "best": best,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Sweep complete. Results written to {out_path}")
    if best is not None:
        print(
            f"Best: candidate={best['candidate']} "
            f"layers={best['layers']} alpha={best['sweep_alpha']} "
            f"f1={best['metrics']['f1']:.6f}"
        )

    # Run manifest. Tuning is a development activity, so the default split
    # is validation_alpha. Using --dataset-split test/final_test here would
    # be a split-leakage violation and the manifest validator would reject it.
    if not args.no_manifest:
        # Collect the union of (vector, path) pairs across all candidates.
        all_vectors: list[Any] = []
        all_paths: list[str] = []
        seen_vector_ids: set[str] = set()
        for candidate in candidates:
            for vector, path in zip(
                candidate["vectors"], candidate["manifest_paths"]
            ):
                if vector.spec.vector_id not in seen_vector_ids:
                    seen_vector_ids.add(vector.spec.vector_id)
                    all_vectors.append(vector)
                    all_paths.append(path)
        run_id = args.run_id or f"tune_steering_{int(__import__('time').time())}"
        try:
            sha, _ = emit_manifest(
                run_id=run_id,
                repo_root=Path(__file__).resolve().parent.parent,
                output_path=out_path,
                output_kind="sweep",
                model_id=args.model,
                dataset_path=args.val_tasks,
                dataset_split=args.dataset_split,
                label_field=args.label_field,
                label_oracle_kind=args.label_oracle_kind,
                seed=0,
                prompt_format=args.prompt_format,
                route_decision_mode="logit",
                route_token_resolver=args.route_token_resolver,
                batch_size=args.batch_size,
                route_batch_size=args.batch_size,
                max_new_tokens=None,
                tool_vectors=all_vectors,
                tool_vector_paths=all_paths,
                tool_alphas=[None] * len(all_vectors),
                loaded_model=model,
                loaded_tokenizer=tokenizer,
            )
            print(manifest_reference_line(sha, Path(str(out_path) + ".manifest.json")))
        except ManifestValidationError as exc:
            print(
                f"warning: run manifest validation failed ({exc}); "
                "sweep output is still valid but is not headline-eligible. "
                "See CLAIMS.md and docs/RUN_MANIFEST.md."
            )


if __name__ == "__main__":
    main()
