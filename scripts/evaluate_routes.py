
from __future__ import annotations

import argparse
import json
from pathlib import Path

# V037-002: the reusable logic now lives in the library layer.
# ``scripts/evaluate_routes.py`` is a thin CLI wrapper around it.
from daph_learning.evaluation.routes import evaluate_route_records, load_jsonl as load
from daph_learning.experiments.manifest import emit_manifest, manifest_reference_line
from daph_learning.evaluation.manifest import ManifestValidationError


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate route decisions against explicit labels or the deterministic auto-policy oracle"
    )
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument(
        "--label-field",
        help="Optional task field containing explicit symbolic/llm route gold labels",
    )
    ap.add_argument(
        "--label-oracle-kind",
        choices=["capability", "accuracy", "utility", "policy_heuristic"],
        default="policy_heuristic",
        help=(
            "What the label field represents. v0.3.5 route_label is a "
            "policy_heuristic, not an oracle. Recorded in the run manifest."
        ),
    )
    ap.add_argument(
        "--dataset-split",
        default="dev",
        help="Split label recorded in the run manifest.",
    )
    ap.add_argument("--model")
    ap.add_argument("--output", help="Optional JSON path for the metrics payload.")
    ap.add_argument("--run-id")
    ap.add_argument("--no-manifest", action="store_true")
    args = ap.parse_args()

    task_rows = load(Path(args.tasks))
    route_rows = load(Path(args.routes))
    tasks = {row["task_id"]: row for row in task_rows}
    routes = {row["task_id"]: row for row in route_rows}

    metrics = evaluate_route_records(
        tasks,
        routes,
        label_field=args.label_field,
        label_oracle_kind=args.label_oracle_kind,
    )
    print(json.dumps(metrics, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )

    if not args.no_manifest and args.output:
        import time

        run_id = args.run_id or f"evaluate_routes_{int(time.time())}"
        extra_outputs: list[tuple[str | Path, str]] = [(args.routes, "routes")]
        try:
            sha, _ = emit_manifest(
                run_id=run_id,
                repo_root=Path(__file__).resolve().parent.parent,
                output_path=args.output,
                output_kind="route_evaluation",
                model_id=args.model,
                dataset_path=args.tasks,
                dataset_split=args.dataset_split,
                label_field=args.label_field,
                label_oracle_kind=args.label_oracle_kind,
                seed=0,
                prompt_format="raw",
                route_decision_mode=None,
                route_token_resolver="isolated",
                batch_size=None,
                route_batch_size=None,
                max_new_tokens=None,
                extra_outputs=extra_outputs,
            )
            print(
                manifest_reference_line(
                    sha, Path(str(args.output) + ".manifest.json")
                )
            )
        except ManifestValidationError as exc:
            print(
                f"warning: run manifest validation failed ({exc}); "
                "evaluation output is still valid but is not headline-eligible. "
                "See CLAIMS.md and docs/RUN_MANIFEST.md."
            )


if __name__ == "__main__":
    main()
