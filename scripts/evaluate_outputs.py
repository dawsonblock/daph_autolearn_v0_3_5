from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from daph_learning.evaluation.scoring import parse_final_or_exact, parse_legacy_first_int


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH; repeat for each arm")
    ap.add_argument(
        "--parser",
        choices=["final_or_exact", "legacy_first_int"],
        default="final_or_exact",
    )
    args = ap.parse_args()

    parser: Callable[[str], int | None] = (
        parse_final_or_exact if args.parser == "final_or_exact" else parse_legacy_first_int
    )

    task_rows = load_jsonl(Path(args.tasks))
    tasks = {x["task_id"]: x for x in task_rows}
    if len(tasks) != len(task_rows):
        raise ValueError("task file contains duplicate task_id values")
    summary = {}

    for arm in args.arm:
        name, path = arm.split("=", 1)
        outputs = load_jsonl(Path(path))
        seen = set()
        correct = 0
        parse_failures = 0
        for row in outputs:
            task_id = row["task_id"]
            if task_id in seen:
                raise ValueError(f"duplicate task_id in {name}: {task_id}")
            seen.add(task_id)
            if task_id not in tasks:
                raise ValueError(f"unknown task_id in {name}: {task_id}")
            pred = parser(str(row["output"]))
            if pred is None:
                parse_failures += 1
            elif pred == tasks[task_id]["expected"]:
                correct += 1
        missing = set(tasks) - seen
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} tasks")
        summary[name] = {
            "n_tasks": len(tasks),
            "accuracy": correct / len(tasks) if tasks else 0.0,
            "correct": correct,
            "parse_failures": parse_failures,
        }

    print(json.dumps({"parser": args.parser, "arms": summary}, indent=2))


if __name__ == "__main__":
    main()
