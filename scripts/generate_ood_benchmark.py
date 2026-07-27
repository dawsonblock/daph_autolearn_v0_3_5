from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


GENERATOR_VERSION = "v0.3.5"

# All categories. The first group is symbolic-arithmetic; the second group
# is the broadened OOD coverage added in v0.3.5 (non-symbolic controls,
# adversarial numeric controls, nested expressions, malformed tasks).
# See CLAIMS.md §2 and docs/EXPERIMENT_PLAN.md.
SYMBOLIC_CATEGORIES = (
    "easy_add",
    "easy_sub",
    "multiply",
    "modulo",
    "modular_multiplication",
    "nested_arithmetic",
    "signed_nested_arithmetic",
)
NON_SYMBOLIC_CATEGORIES = (
    "text_question",
    "coding_question",
    "knowledge_question",
)
ADVERSARIAL_CATEGORIES = (
    "irrelevant_numeric",  # numbers present, symbolic route would be wrong
    "ambiguous_tool",      # looks numeric but has no clean integer answer
)
MALFORMED_CATEGORIES = (
    "unsupported_arithmetic",  # uses ops the symbolic engine rejects (e.g. **, /)
)

ALL_CATEGORIES = (
    SYMBOLIC_CATEGORIES
    + NON_SYMBOLIC_CATEGORIES
    + ADVERSARIAL_CATEGORIES
    + MALFORMED_CATEGORIES
)


def _nonzero(rng: random.Random, low: int, high: int) -> int:
    value = 0
    while value == 0:
        value = rng.randint(low, high)
    return value


def _typed_oracles(*, capability: str, accuracy: str, utility: str) -> dict[str, str]:
    """Build the three typed oracle fields.

    Each is "symbolic" or "llm". They may differ — e.g. a task where the
    symbolic backend is capable (capability=symbolic) but the LLM is more
    accurate on small numbers (accuracy=llm), or where symbolic is capable
    and more accurate but the LLM is cheaper (utility=llm).

    See CLAIMS.md §3 for why these must be separate fields, not a single
    ``route_label`` string.
    """
    return {
        "capability_oracle": capability,
        "accuracy_oracle": accuracy,
        "utility_oracle": utility,
    }


def _arithmetic_task(
    task_id: str,
    rng: random.Random,
    category: str,
    *,
    seed: int,
) -> dict:
    """Generate a symbolic-arithmetic task.

    ``route_label`` is preserved for backward compatibility and equals
    ``utility_oracle`` (the v0.3.4 generator's hand-coded utility heuristic).
    The typed oracle fields make explicit which kind of oracle that is.
    """
    if category == "easy_add":
        a = rng.randint(-50_000, 50_000)
        b = rng.randint(-50_000, 50_000)
        op = "+"
        expected = a + b
        capability_ids = ["integer_arithmetic"]
        specification = f"Compute {a} + {b}. Return only the integer."
        inputs = {"a": a, "b": b, "op": op}
        oracles = _typed_oracles(capability="symbolic", accuracy="llm", utility="llm")

    elif category == "easy_sub":
        a = rng.randint(-50_000, 50_000)
        b = rng.randint(-50_000, 50_000)
        op = "-"
        expected = a - b
        capability_ids = ["integer_arithmetic"]
        specification = f"Compute {a} - {b}. Return only the integer."
        inputs = {"a": a, "b": b, "op": op}
        oracles = _typed_oracles(capability="symbolic", accuracy="llm", utility="llm")

    elif category == "multiply":
        a = rng.randint(-50_000, 50_000)
        b = rng.randint(-50_000, 50_000)
        expected = a * b
        capability_ids = ["integer_arithmetic"]
        specification = f"Compute {a} * {b}. Return only the integer."
        inputs = {"a": a, "b": b, "op": "*"}
        oracles = _typed_oracles(capability="symbolic", accuracy="symbolic", utility="symbolic")

    elif category == "modulo":
        a = rng.randint(-2_000_000, 2_000_000)
        b = _nonzero(rng, 2, 50_000)
        expected = a % b
        capability_ids = ["integer_arithmetic"]
        specification = f"Compute {a} % {b}. Return only the integer."
        inputs = {"a": a, "b": b, "op": "%"}
        oracles = _typed_oracles(capability="symbolic", accuracy="symbolic", utility="symbolic")

    elif category == "modular_multiplication":
        a = rng.randint(-50_000, 50_000)
        b = rng.randint(-50_000, 50_000)
        modulus = rng.randint(2, 20_000)
        expected = (a * b) % modulus
        capability_ids = ["modular_multiplication"]
        specification = f"Compute ({a} * {b}) mod {modulus}. Return only the integer."
        inputs = {"a": a, "b": b, "modulus": modulus}
        oracles = _typed_oracles(capability="symbolic", accuracy="symbolic", utility="symbolic")

    elif category == "nested_arithmetic":
        # Nested expression using the symbolic engine's textual fallback.
        # The engine supports +, -, *, %, //, parentheses, unary +/-.
        a = rng.randint(100, 999)
        b = rng.randint(10, 99)
        c = rng.randint(2, 20)
        expr = f"({a} + {b}) * {c}"
        expected = (a + b) * c
        capability_ids = ["integer_arithmetic"]
        specification = f"Compute {expr}. Return only the integer."
        inputs = {}  # nested expressions use the textual fallback, not structured inputs
        oracles = _typed_oracles(capability="symbolic", accuracy="symbolic", utility="symbolic")

    elif category == "signed_nested_arithmetic":
        a = rng.randint(-999, 999)
        b = rng.randint(-99, 99)
        c = rng.randint(2, 20)
        expr = f"({a} - {b}) * {c}"
        expected = (a - b) * c
        capability_ids = ["integer_arithmetic"]
        specification = f"Compute {expr}. Return only the integer."
        inputs = {}
        oracles = _typed_oracles(capability="symbolic", accuracy="symbolic", utility="symbolic")

    else:
        raise ValueError(f"unknown symbolic category: {category}")

    task: dict = {
        "task_id": task_id,
        "capability_ids": capability_ids,
        "inputs": inputs,
        "specification": specification,
        "expected": expected,
        "route_label": oracles["utility_oracle"],  # backward compat
        **oracles,
        "metadata": {
            "generator": GENERATOR_VERSION,
            "seed": seed,
            "category": category,
            "category_group": "symbolic",
        },
    }
    return task


def _non_symbolic_task(
    task_id: str,
    rng: random.Random,
    category: str,
    *,
    seed: int,
) -> dict:
    """Generate a non-symbolic control task.

    These tasks have no integer answer and must route to LLM. They exist
    so that a pathological steering vector that pushes everything toward
    SYMBOLIC can be detected (it would produce false tool calls here).
    See CLAIMS.md §2.
    """
    if category == "text_question":
        questions = [
            "Explain why the sky appears blue.",
            "Summarize the plot of Hamlet in two sentences.",
            "What are the trade-offs between recursion and iteration?",
            "Describe how a transformer attention mechanism works.",
            "Why is the sky red at sunset?",
        ]
        specification = rng.choice(questions)
        expected = None
        capability_ids = []

    elif category == "coding_question":
        questions = [
            "Write a Python function that reverses a linked list.",
            "Write a Python function to compute the Fibonacci sequence iteratively.",
            "Write a SQL query to find the second-highest salary in a table.",
            "Write a Python function that checks if a string is a palindrome.",
            "Write a Python function to merge two sorted lists.",
        ]
        specification = rng.choice(questions)
        expected = None
        capability_ids = ["code_generation"]

    elif category == "knowledge_question":
        questions = [
            "Which city is the capital of France?",
            "Who wrote 'One Hundred Years of Solitude'?",
            "What is the chemical formula for water?",
            "In what year did the Berlin Wall fall?",
            "What is the largest planet in the solar system?",
        ]
        specification = rng.choice(questions)
        expected = None
        capability_ids = ["knowledge"]

    else:
        raise ValueError(f"unknown non-symbolic category: {category}")

    oracles = _typed_oracles(capability="llm", accuracy="llm", utility="llm")
    return {
        "task_id": task_id,
        "capability_ids": capability_ids,
        "inputs": {},
        "specification": specification,
        "expected": expected,
        "route_label": "llm",  # backward compat
        **oracles,
        "metadata": {
            "generator": GENERATOR_VERSION,
            "seed": seed,
            "category": category,
            "category_group": "non_symbolic",
        },
    }


def _adversarial_task(
    task_id: str,
    rng: random.Random,
    category: str,
    *,
    seed: int,
) -> dict:
    """Generate an adversarial numeric control.

    These tasks contain numbers but should NOT route to SYMBOLIC. They
    test whether steering causes indiscriminate tool invocation.
    See CLAIMS.md §2.
    """
    if category == "irrelevant_numeric":
        # Numbers present, but the task is not a computation.
        a = rng.randint(100, 9999)
        b = rng.randint(100, 9999)
        questions = [
            f"In the year {a}, what major historical event happened related to {b}?",
            f"Is the number {a} prime? Explain your reasoning.",
            f"Estimate how many words are in a {a}-page book with {b} words per page.",
            f"If a train travels at {a} mph for {b} hours, describe the journey qualitatively.",
            f"Discuss the significance of the number {a} in mathematics.",
        ]
        specification = rng.choice(questions)
        expected = None
        capability_ids = []

    elif category == "ambiguous_tool":
        # Looks numeric but has no clean integer answer the symbolic engine
        # can produce (e.g. requires reasoning, not computation).
        a = rng.randint(2, 100)
        questions = [
            f"Is {a} a perfect square? Explain.",
            f"What is the smallest prime greater than {a}?",
            f"Approximate the square root of {a} to two decimal places.",
            f"How many divisors does {a} have? Show your reasoning.",
            f"Is {a} even or odd? Explain why.",
        ]
        specification = rng.choice(questions)
        expected = None
        capability_ids = []

    else:
        raise ValueError(f"unknown adversarial category: {category}")

    oracles = _typed_oracles(capability="llm", accuracy="llm", utility="llm")
    return {
        "task_id": task_id,
        "capability_ids": capability_ids,
        "inputs": {},
        "specification": specification,
        "expected": expected,
        "route_label": "llm",  # backward compat
        **oracles,
        "metadata": {
            "generator": GENERATOR_VERSION,
            "seed": seed,
            "category": category,
            "category_group": "adversarial",
        },
    }


def _malformed_task(
    task_id: str,
    rng: random.Random,
    category: str,
    *,
    seed: int,
) -> dict:
    """Generate a malformed/unsupported arithmetic task.

    These use operations the symbolic engine deliberately rejects (true
    division, exponentiation). They test fallback behavior: a correct
    router should send them to LLM, not crash trying to execute them
    symbolically.
    """
    if category == "unsupported_arithmetic":
        a = rng.randint(1, 1000)
        b = rng.randint(1, 1000)
        # True division — rejected by the bounded integer evaluator.
        specification = f"Compute {a} / {b}. Return only the integer."
        expected = None  # no clean integer answer
        capability_ids = ["integer_arithmetic"]
        inputs = {"a": a, "b": b, "op": "/"}
    else:
        raise ValueError(f"unknown malformed category: {category}")

    # Capability oracle says symbolic could in principle handle division,
    # but the v0.3.5 engine rejects it, so accuracy and utility say LLM.
    oracles = _typed_oracles(capability="symbolic", accuracy="llm", utility="llm")
    return {
        "task_id": task_id,
        "capability_ids": capability_ids,
        "inputs": inputs,
        "specification": specification,
        "expected": expected,
        "route_label": "llm",  # backward compat
        **oracles,
        "metadata": {
            "generator": GENERATOR_VERSION,
            "seed": seed,
            "category": category,
            "category_group": "malformed",
        },
    }


_CATEGORY_DISPATCH = {
    "easy_add": _arithmetic_task,
    "easy_sub": _arithmetic_task,
    "multiply": _arithmetic_task,
    "modulo": _arithmetic_task,
    "modular_multiplication": _arithmetic_task,
    "nested_arithmetic": _arithmetic_task,
    "signed_nested_arithmetic": _arithmetic_task,
    "text_question": _non_symbolic_task,
    "coding_question": _non_symbolic_task,
    "knowledge_question": _non_symbolic_task,
    "irrelevant_numeric": _adversarial_task,
    "ambiguous_tool": _adversarial_task,
    "unsupported_arithmetic": _malformed_task,
}


def generate_task(
    task_id: str,
    rng: random.Random,
    category: str,
    *,
    seed: int,
) -> dict:
    """Dispatch to the appropriate generator for a category."""
    handler = _CATEGORY_DISPATCH.get(category)
    if handler is None:
        raise ValueError(f"unknown category: {category}")
    return handler(task_id, rng, category, seed=seed)


def generate_benchmark(
    num_tasks: int,
    seed: int,
    *,
    categories: tuple[str, ...] = ALL_CATEGORIES,
) -> list[dict]:
    """Generate a deterministic, balanced benchmark.

    Categories are cycled deterministically so counts differ by at most
    one, then task order is shuffled with the same seeded RNG. This
    preserves the v0.3.4 determinism and balance guarantees.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if not categories:
        raise ValueError("categories must be non-empty")
    rng = random.Random(seed)
    tasks = [
        generate_task(
            f"ood-eval-{i + 1:04d}",
            rng,
            categories[i % len(categories)],
            seed=seed,
        )
        for i in range(num_tasks)
    ]
    rng.shuffle(tasks)
    return tasks


def write_benchmark(path: Path, tasks: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a deterministic balanced OOD routing benchmark with typed oracle labels"
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-tasks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--categories",
        nargs="+",
        choices=ALL_CATEGORIES,
        help=(
            "Subset of categories to include. Defaults to all categories. "
            "Use this to generate, e.g., a symbolic-only benchmark or a "
            "non-symbolic-only control set."
        ),
    )
    ap.add_argument(
        "--symbolic-only",
        action="store_true",
        help="Convenience flag: include only symbolic-arithmetic categories.",
    )
    ap.add_argument(
        "--non-symbolic-only",
        action="store_true",
        help=(
            "Convenience flag: include only non-symbolic, adversarial, and "
            "malformed categories. Useful for testing false-positive tool "
            "call rates."
        ),
    )
    args = ap.parse_args()

    if args.symbolic_only and args.non_symbolic_only:
        raise ValueError("--symbolic-only and --non-symbolic-only are mutually exclusive")
    if args.categories and (args.symbolic_only or args.non_symbolic_only):
        raise ValueError("--categories cannot be combined with --symbolic-only/--non-symbolic-only")

    if args.symbolic_only:
        categories = SYMBOLIC_CATEGORIES
    elif args.non_symbolic_only:
        categories = NON_SYMBOLIC_CATEGORIES + ADVERSARIAL_CATEGORIES + MALFORMED_CATEGORIES
    elif args.categories:
        categories = tuple(args.categories)
    else:
        categories = ALL_CATEGORIES

    tasks = generate_benchmark(args.num_tasks, args.seed, categories=categories)
    digest = write_benchmark(Path(args.output), tasks)
    counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for task in tasks:
        category = task["metadata"]["category"]
        group = task["metadata"]["category_group"]
        counts[category] = counts.get(category, 0) + 1
        group_counts[group] = group_counts.get(group, 0) + 1

    print(
        json.dumps(
            {
                "output": args.output,
                "num_tasks": len(tasks),
                "seed": args.seed,
                "generator_version": GENERATOR_VERSION,
                "categories": list(categories),
                "category_counts": counts,
                "category_group_counts": group_counts,
                "sha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
