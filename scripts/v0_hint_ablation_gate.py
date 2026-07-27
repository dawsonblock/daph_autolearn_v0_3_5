from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from daph_learning.evaluation.paired import paired_discordant_analysis
from daph_learning.evaluation.scoring import parse_final_or_exact, parse_legacy_first_int


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired V0 hint-ablation gate")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--baseline-outputs", required=True)
    ap.add_argument("--treatment-outputs", required=True)
    ap.add_argument("--parser", choices=["final_or_exact", "legacy_first_int"], default="final_or_exact")
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--expected-task-sha256")
    args = ap.parse_args()

    task_path = Path(args.tasks)
    digest = _digest(task_path)
    if args.expected_task_sha256 and digest.lower() != args.expected_task_sha256.lower():
        raise SystemExit(f"task digest mismatch: expected {args.expected_task_sha256}, got {digest}")

    parser = parse_final_or_exact if args.parser == "final_or_exact" else parse_legacy_first_int
    task_rows = _jsonl(task_path)
    tasks = {r["task_id"]: r for r in task_rows}
    if len(tasks) != len(task_rows):
        raise ValueError("duplicate task_id in tasks")

    def load_arm(path: str) -> dict[str, int | None]:
        rows = _jsonl(Path(path))
        out = {}
        for row in rows:
            tid = row["task_id"]
            if tid in out:
                raise ValueError(f"duplicate task_id in output arm: {tid}")
            if tid not in tasks:
                raise ValueError(f"unknown task_id in output arm: {tid}")
            out[tid] = parser(str(row["output"]))
        missing = set(tasks) - set(out)
        if missing:
            raise ValueError(f"output arm missing {len(missing)} tasks")
        return out

    baseline = load_arm(args.baseline_outputs)
    treatment = load_arm(args.treatment_outputs)
    base_ok = {tid: baseline[tid] == tasks[tid]["expected"] for tid in tasks}
    treat_ok = {tid: treatment[tid] == tasks[tid]["expected"] for tid in tasks}
    n = len(tasks)
    b_acc = sum(base_ok.values()) / n if n else 0.0
    t_acc = sum(treat_ok.values()) / n if n else 0.0
    improved = sum((not base_ok[t]) and treat_ok[t] for t in tasks)
    regressed = sum(base_ok[t] and (not treat_ok[t]) for t in tasks)
    paired = paired_discordant_analysis(improved, regressed)
    p = paired["p_value"]
    delta = t_acc - b_acc
    result = {
        "baseline_accuracy": b_acc,
        "treatment_accuracy": t_acc,
        "delta": delta,
        "discordant_improved": improved,
        "discordant_regressed": regressed,
        "p_value_two_sided": p,
        "mcnemar_statistic_continuity_corrected": paired["mcnemar_statistic"],
        "mcnemar_p_value": paired["mcnemar_p_value"],
        "discordant_odds_ratio": paired["odds_ratio"],
        "discordant_improved_probability": paired["improved_probability"],
        "discordant_improved_probability_ci_95": paired["improved_probability_ci_95"],
        "discordant_net_effect": paired["net_discordant_effect"],
        "discordant_net_effect_ci_95": paired["net_discordant_effect_ci_95"],
        "n_tasks": n,
        "task_sha256": digest,
        "parser": args.parser,
        "scientific_gate": True,
        "effect_size_pass": delta >= args.min_delta,
        "statistical_pass": p <= args.alpha,
        "gate_pass": delta >= args.min_delta and p <= args.alpha,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
