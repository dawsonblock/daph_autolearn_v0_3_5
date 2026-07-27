"""Route-decision evaluation (V037-002).

Moved out of ``scripts/evaluate_routes.py`` so the library layer
(``daph_learning``) no longer depends on the CLI layer (``scripts``). The
``scripts/evaluate_routes.py`` CLI now imports from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from daph_learning.routing.arithmetic_router import RoutingConfig, route_task


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file of task rows, validating task_id presence and uniqueness."""
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


__all__ = ["evaluate_route_records", "load_jsonl"]
