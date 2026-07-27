
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from daph_learning.routing.arithmetic_router import RoutingConfig, route_task
from scripts._manifest import emit_manifest, manifest_reference_line
from daph_learning.evaluation.manifest import ManifestValidationError


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    ids = [str(row.get("task_id", "")) for row in rows]
    if any(not tid for tid in ids):
        raise ValueError(f"{path} contains a row without task_id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate task_id values")
    return rows


def evaluate_route_records(
    tasks: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Mapping[str, Any]],
    *,
    label_field: str | None = None,
    label_oracle_kind: str | None = None,
) -> dict[str, float | int | str | None]:
    """Evaluate route decisions against an oracle.

    ``label_field`` selects which task field holds the route label
    (``"symbolic"`` / ``"llm"``). ``label_oracle_kind`` records *what kind
    of oracle* that field represents — one of ``"capability"``,
    ``"accuracy"``, ``"utility"``, or ``"policy_heuristic"``. See
    ``CLAIMS.md`` §3 for why this distinction matters: a single
    ``route_label`` string silently conflates three different experimental
    questions.

    The returned metrics dict now includes ``label_oracle_kind`` so
    downstream consumers (manifests, reports) cannot silently re-interpret
    a policy_heuristic result as a capability oracle.
    """
    missing = set(tasks) - set(routes)
    if missing:
        raise ValueError(f"routes missing {len(missing)} tasks")
    extras = set(routes) - set(tasks)
    if extras:
        raise ValueError(f"routes contain {len(extras)} unknown tasks")

    tp = fp = tn = fn = malformed = correct_well_formed = 0

    for tid, task in tasks.items():
        if label_field:
            label = task.get(label_field)
            if label not in {"symbolic", "llm"}:
                raise ValueError(
                    f"task {tid} has invalid {label_field!r} route label: {label!r}"
                )
            oracle = label == "symbolic"
        else:
            oracle = (
                route_task(
                    task,
                    False,
                    RoutingConfig(execution_mode="auto"),
                ).backend
                == "symbolic"
            )

        observed_raw = routes[tid].get("route")
        is_malformed = observed_raw not in {"symbolic", "llm"}

        if is_malformed:
            malformed += 1
            # Fail closed: malformed output cannot authorize a symbolic tool.
            # For a symbolic oracle task this is operationally a false negative.
            # For an LLM oracle task it is neither TN nor FP: it is malformed and
            # is penalized via coverage and total route accuracy.
            if oracle:
                fn += 1
            continue

        observed = observed_raw == "symbolic"
        if oracle and observed:
            tp += 1
        elif not oracle and observed:
            fp += 1
        elif oracle and not observed:
            fn += 1
        else:
            tn += 1

        if observed == oracle:
            correct_well_formed += 1

    binary_precision = tp / (tp + fp) if tp + fp else 0.0
    binary_recall = tp / (tp + fn) if tp + fn else 0.0
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )

    n = len(tasks)
    malformed_rate = malformed / n if n else 0.0
    decision_coverage = 1.0 - malformed_rate if n else 0.0
    route_accuracy = correct_well_formed / n if n else 0.0

    # Coverage-adjusted headline metrics penalize every malformed decision,
    # including malformed outputs on LLM-oracle tasks that are deliberately
    # excluded from TN.
    precision = binary_precision * decision_coverage
    recall = binary_recall * decision_coverage
    f1 = binary_f1 * decision_coverage

    well_formed_n = n - malformed
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "malformed_routes": malformed,
        "malformed_rate": malformed_rate,
        "well_formed_decisions": well_formed_n,
        "decision_coverage": decision_coverage,
        "route_accuracy": route_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "specificity": specificity,
        # Oracle provenance: records what kind of oracle the labels
        # represent, so downstream consumers cannot silently re-interpret
        # a policy_heuristic result as a capability oracle. See CLAIMS.md §3.
        "label_field": label_field,
        "label_oracle_kind": label_oracle_kind,
    }


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
