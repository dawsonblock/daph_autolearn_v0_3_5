"""Build empirical oracle labels by measuring model + symbolic accuracy.

v0.3.6 Phase 2.1 — replace hand-authored oracle heuristics with measured
ground truth. The v0.3.5 ``generate_ood_benchmark.py`` writes three typed
oracle fields (``capability_oracle``, ``accuracy_oracle``,
``utility_oracle``) whose values are *hand-coded policy rules* derived
from the category the task was generated from. CLAIMS.md therefore
qualifies the routing labels as ``PARTIAL / BOOTSTRAP`` rather than
empirically grounded.

This script loads an existing benchmark, runs:

1. the **symbolic executor** on every task whose ``capability_ids`` claim
   a supported arithmetic capability, recording whether execution
   succeeded and the produced value; and
2. an **unsteered LLM baseline** on every task with a numeric
   ``expected`` answer, recording the generated text and whether the
   parsed integer matches ``expected``.

It then overwrites the three oracle fields with empirically measured
values:

- ``capability_oracle`` = ``"symbolic"`` iff the symbolic executor
  finished without raising and produced a value; otherwise ``"llm"``.
- ``accuracy_oracle`` = ``"symbolic"`` iff symbolic succeeded AND
  (the LLM failed OR the LLM was not evaluated); ``"llm"`` iff the LLM
  succeeded AND symbolic failed; ties keep the symbolic answer (it is
  exact and cheaper). Tasks with no numeric ``expected`` are
  ``"llm"`` (only the LLM can produce a free-form answer).
- ``utility_oracle`` = a latency-weighted trade-off: symbolic when
  symbolic succeeded and the symbolic latency is below
  ``--symbolic-latency-budget-ms`` *or* the LLM failed; otherwise llm.

The original ``route_label`` field is preserved for backward
compatibility and a new ``empirical_route_label`` field records the
utility oracle's decision so downstream scripts can opt into the
empirical labels without breaking the v0.3.5 contract.

A sidecar ``<output>.measurements.jsonl`` is written with the raw
per-task measurements (symbolic success/value/latency, LLM
text/parsed-int/latency) so the oracle derivation is auditable.

Usage::

    python scripts/build_empirical_oracles.py \\
        --input data/v0_splits/D3.jsonl \\
        --output data/v0_splits/D3_empirical.jsonl \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --prompt-format chat \\
        --max-new-tokens 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from daph_learning.evaluation.scoring import parse_final_or_exact
from daph_learning.execution.symbolic_executor import (
    execute_plan,
    plan_from_structured_task,
)
from daph_learning.tools.symbolic_math import SymbolicMathError


SCRIPT_VERSION = "v0.3.6"


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid task JSONL at {path}:{line_no}") from exc
            tasks.append(task)
    return tasks


def _measure_symbolic(task: dict[str, Any]) -> dict[str, Any]:
    """Run the symbolic executor on a task. Returns a measurement record.

    Records ``symbolic_ran``, ``symbolic_value``, ``symbolic_latency_ms``,
    and ``symbolic_error``. A task that does not declare a supported
    capability is recorded as ``symbolic_ran=False`` with
    ``symbolic_error="unsupported_capability"`` rather than raising.
    """
    caps = set(task.get("capability_ids") or [])
    supported = {"integer_arithmetic", "modular_multiplication"}
    if not (caps & supported):
        return {
            "symbolic_ran": False,
            "symbolic_value": None,
            "symbolic_latency_ms": 0.0,
            "symbolic_error": "unsupported_capability",
        }
    t0 = time.perf_counter()
    try:
        plan = plan_from_structured_task(task, reason_code="empirical_oracle")
        result = execute_plan(plan)
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "symbolic_ran": True,
            "symbolic_value": int(result.value),
            "symbolic_latency_ms": latency,
            "symbolic_error": None,
        }
    except (SymbolicMathError, SyntaxError, ZeroDivisionError, ValueError, KeyError, TypeError) as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "symbolic_ran": False,
            "symbolic_value": None,
            "symbolic_latency_ms": latency,
            "symbolic_error": f"{type(exc).__name__}: {exc}",
        }


def _build_answer_prompt(task: dict[str, Any]) -> str:
    """The same FINAL:-format answer prompt used by generate_v0_outputs."""
    specification = str(task.get("specification", ""))
    return (
        "Solve the following task:\n"
        f"{specification}\n"
        "Return the answer using exactly this final line format:\n"
        "FINAL: <integer>\n"
    )


def _measure_llm(
    task: dict[str, Any],
    model: Any,
    tokenizer: Any,
    *,
    prompt_format: str,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Run the unsteered LLM on a task and parse the integer answer.

    Tasks without a numeric ``expected`` are still run so we can record
    the raw generation, but ``llm_correct`` is ``None`` (unverifiable)
    rather than ``False``.

    v0.3.6 Phase 3.3: latency is measured with PyTorch CUDA events when
    a GPU is available so the reported latency reflects actual GPU work
    rather than host-side kernel-launch overhead. On CPU it falls back
    to ``time.perf_counter()``. The ``llm_latency_device`` field records
    which path was taken so downstream analysis can filter CPU vs CUDA
    measurements.
    """
    import torch
    from daph_learning.evaluation.timing import cuda_timed

    expected = task.get("expected")
    prompt = _build_answer_prompt(task)

    if prompt_format == "chat" and getattr(tokenizer, "chat_template", None) is not None:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt

    encoded = tokenizer(rendered, return_tensors="pt")
    embed_device = model.get_input_embeddings().weight.device
    encoded = {
        k: (v.to(embed_device) if hasattr(v, "to") else torch.tensor(v).to(embed_device))
        for k, v in encoded.items()
    }

    torch.manual_seed(seed)
    try:
        with cuda_timed() as timing:
            with torch.inference_mode():
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    use_cache=True,
                )
        text = tokenizer.decode(
            out[0][encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - defensive for OOM etc.
        return {
            "llm_ran": False,
            "llm_text": None,
            "llm_parsed_int": None,
            "llm_correct": None,
            "llm_latency_ms": 0.0,
            "llm_latency_device": None,
            "llm_cuda_event_timing": False,
            "llm_error": f"{type(exc).__name__}: {exc}",
        }

    parsed = parse_final_or_exact(text)
    if expected is None:
        correct: bool | None = None
    else:
        try:
            correct = parsed is not None and int(parsed) == int(expected)
        except (ValueError, TypeError):
            correct = False
    return {
        "llm_ran": True,
        "llm_text": text,
        "llm_parsed_int": parsed,
        "llm_correct": correct,
        "llm_latency_ms": timing.elapsed_ms,
        "llm_latency_device": timing.device,
        "llm_cuda_event_timing": timing.cuda_event_timing,
        "llm_error": None,
    }


def _derive_oracles(
    task: dict[str, Any],
    sym: dict[str, Any],
    llm: dict[str, Any],
    *,
    symbolic_latency_budget_ms: float,
) -> dict[str, Any]:
    """Derive the three typed oracle labels from measurements."""
    expected = task.get("expected")
    sym_ok = sym["symbolic_ran"] and sym["symbolic_value"] is not None
    sym_correct = (
        sym_ok
        and expected is not None
        and int(sym["symbolic_value"]) == int(expected)
    )
    llm_ok = llm["llm_ran"]
    llm_correct = llm["llm_correct"]

    # capability_oracle: symbolic iff the executor finished without error.
    capability = "symbolic" if sym_ok else "llm"

    # accuracy_oracle: who got the right answer?
    if expected is None:
        # No checkable integer answer — only the LLM can produce a
        # free-form response.
        accuracy = "llm"
    elif sym_correct and llm_correct is not True:
        # Symbolic got it right and the LLM did not (failed, wrong, or
        # unverifiable). Prefer symbolic.
        accuracy = "symbolic"
    elif llm_correct is True and not sym_correct:
        accuracy = "llm"
    elif sym_correct and llm_correct is True:
        # Both correct — prefer symbolic (exact, cheaper, deterministic).
        accuracy = "symbolic"
    else:
        # Neither got it right. Fall back to capability: if symbolic
        # could at least run, keep it; otherwise llm.
        accuracy = "symbolic" if sym_ok else "llm"

    # utility_oracle: accuracy + latency trade-off.
    if expected is None:
        utility = "llm"
    elif sym_correct and (
        llm_correct is not True
        or sym["symbolic_latency_ms"] <= symbolic_latency_budget_ms
    ):
        utility = "symbolic"
    elif llm_correct is True:
        utility = "llm"
    else:
        utility = "symbolic" if sym_ok else "llm"

    return {
        "capability_oracle": capability,
        "accuracy_oracle": accuracy,
        "utility_oracle": utility,
        "empirical_route_label": utility,
        # Auditable provenance for the empirical derivation.
        "empirical": {
            "script_version": SCRIPT_VERSION,
            "symbolic_ran": sym["symbolic_ran"],
            "symbolic_value": sym["symbolic_value"],
            "symbolic_correct": sym_correct,
            "symbolic_latency_ms": round(sym["symbolic_latency_ms"], 4),
            "symbolic_error": sym["symbolic_error"],
            "llm_ran": llm["llm_ran"],
            "llm_parsed_int": llm["llm_parsed_int"],
            "llm_correct": llm_correct,
            "llm_latency_ms": round(llm["llm_latency_ms"], 4),
            "llm_latency_device": llm.get("llm_latency_device"),
            "llm_cuda_event_timing": llm.get("llm_cuda_event_timing", False),
            "llm_error": llm["llm_error"],
        },
    }


def build_empirical_oracles(
    tasks: list[dict[str, Any]],
    model: Any | None,
    tokenizer: Any | None,
    *,
    prompt_format: str = "chat",
    max_new_tokens: int = 64,
    seed: int = 0,
    symbolic_latency_budget_ms: float = 50.0,
    run_llm: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add empirical oracle labels to each task.

    Returns ``(annotated_tasks, measurements)``. ``measurements`` is the
    per-task raw record (task_id + symbolic + llm measurement) for the
    sidecar file.

    When ``model``/``tokenizer`` are ``None`` or ``run_llm`` is ``False``,
    only the symbolic measurement is taken; the LLM fields are recorded
    as ``llm_ran=False`` and the accuracy/utility oracles fall back to
    the symbolic-only signal. This lets the script run in
    symbolic-only mode without a GPU/model.
    """
    annotated: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for task in tasks:
        sym = _measure_symbolic(task)
        if run_llm and model is not None and tokenizer is not None:
            llm = _measure_llm(
                task, model, tokenizer,
                prompt_format=prompt_format,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        else:
            llm = {
                "llm_ran": False, "llm_text": None, "llm_parsed_int": None,
                "llm_correct": None, "llm_latency_ms": 0.0,
                "llm_latency_device": None, "llm_cuda_event_timing": False,
                "llm_error": "skipped",
            }
        oracles = _derive_oracles(
            task, sym, llm,
            symbolic_latency_budget_ms=symbolic_latency_budget_ms,
        )
        new_task = dict(task)
        # Overwrite the heuristic oracle fields with empirical ones.
        new_task["capability_oracle"] = oracles["capability_oracle"]
        new_task["accuracy_oracle"] = oracles["accuracy_oracle"]
        new_task["utility_oracle"] = oracles["utility_oracle"]
        new_task["empirical_route_label"] = oracles["empirical_route_label"]
        new_task["oracle_provenance"] = "empirical_v0_3_6"
        new_task["empirical_measurements"] = oracles["empirical"]
        # Preserve the original route_label for backward compatibility.
        annotated.append(new_task)
        measurements.append({
            "task_id": task.get("task_id"),
            **oracles["empirical"],
        })
    return annotated, measurements


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build empirical oracle labels by measuring symbolic + LLM "
            "accuracy on an existing benchmark (v0.3.6 Phase 2.1)"
        )
    )
    ap.add_argument("--input", required=True, help="Input benchmark JSONL")
    ap.add_argument("--output", required=True, help="Output annotated JSONL")
    ap.add_argument(
        "--measurements-output",
        help="Sidecar measurements JSONL. Defaults to <output>.measurements.jsonl",
    )
    ap.add_argument("--model", help="HuggingFace model id. Omit for symbolic-only mode.")
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--symbolic-latency-budget-ms",
        type=float,
        default=50.0,
        help=(
            "Latency budget in ms. Symbolic wins the utility oracle if it "
            "is correct and either the LLM is wrong or symbolic latency is "
            "below this budget."
        ),
    )
    ap.add_argument(
        "--symbolic-only",
        action="store_true",
        help="Skip the LLM measurement; derive oracles from symbolic execution alone.",
    )
    args = ap.parse_args()

    tasks = _load_tasks(Path(args.input))
    model = None
    tokenizer = None
    run_llm = not args.symbolic_only
    if run_llm:
        if not args.model:
            ap.error("--model is required unless --symbolic-only is set")
        from daph_learning.data.task_utils import load_llm as _load_llm
        model, tokenizer = _load_llm(args.model)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    annotated, measurements = build_empirical_oracles(
        tasks, model, tokenizer,
        prompt_format=args.prompt_format,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        symbolic_latency_budget_ms=args.symbolic_latency_budget_ms,
        run_llm=run_llm,
    )

    out_path = Path(args.output)
    _write_jsonl(out_path, annotated)
    meas_path = Path(args.measurements_output) if args.measurements_output else Path(str(out_path) + ".measurements.jsonl")
    _write_jsonl(meas_path, measurements)

    # Summary
    n = len(annotated)
    cap_sym = sum(1 for t in annotated if t["capability_oracle"] == "symbolic")
    acc_sym = sum(1 for t in annotated if t["accuracy_oracle"] == "symbolic")
    util_sym = sum(1 for t in annotated if t["utility_oracle"] == "symbolic")
    print(f"wrote {n} annotated tasks to {out_path}")
    print(f"  measurements: {meas_path}")
    print(f"  capability_oracle=symbolic: {cap_sym}/{n} ({cap_sym/n:.2%})" if n else "  n=0")
    print(f"  accuracy_oracle=symbolic:   {acc_sym}/{n} ({acc_sym/n:.2%})" if n else "")
    print(f"  utility_oracle=symbolic:    {util_sym}/{n} ({util_sym/n:.2%})" if n else "")


if __name__ == "__main__":
    main()
