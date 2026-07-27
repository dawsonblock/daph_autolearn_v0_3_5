from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

from daph_learning.execution.symbolic_executor import execute_plan, plan_from_structured_task
from daph_learning.memory.procedural import Procedure, ProceduralMemory
from daph_learning.routing.arithmetic_router import (
    RoutingConfig,
    decision_from_action,
    route_task,
)
from daph_learning.routing.logit_router import score_route_batch_from_logits
from daph_learning.routing.steered_router import build_route_prompt, parse_route_action
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import (
    multi_layer_residual_addition_hook,
    validate_vector_for_model,
)
from daph_learning.steering.io import load_vector, load_vector_bundle
from daph_learning.tools.symbolic_math import SymbolicMathError
from scripts._manifest import emit_manifest, manifest_reference_line
from daph_learning.evaluation.manifest import ManifestValidationError


def _load_memory(path: Path | None) -> ProceduralMemory:
    if path is None:
        return ProceduralMemory()
    return ProceduralMemory.from_jsonl(path)


def _build_prompt(task: dict[str, Any], procedures: list[Procedure]) -> str:
    guidance = ""
    if procedures:
        guidance = "\nVerified procedural memory:\n" + "\n".join(
            f"- {p.action}" for p in procedures
        )
    return (
        "Solve the following task:\n"
        f"{task['specification']}"
        f"{guidance}\n"
        "Return the answer using exactly this final line format:\n"
        "FINAL: <integer>\n"
    )


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid task JSONL at {path}:{line_no}") from exc
            tid = str(task.get("task_id", ""))
            if not tid:
                raise ValueError(f"missing task_id at {path}:{line_no}")
            if tid in seen:
                raise ValueError(f"duplicate task_id {tid!r} at {path}:{line_no}")
            if "capability_ids" not in task or "specification" not in task:
                raise ValueError(f"task {tid!r} is missing required fields")
            seen.add(tid)
            tasks.append(task)
    return tasks


def _load_llm(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def _format_for_model(prompt: str, tokenizer, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError("--prompt-format chat requested but tokenizer has no chat template")
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _chunks(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _stage_elapsed_ms(start: float | None, end: float | None) -> float:
    """Compute the elapsed milliseconds for a pipeline stage.

    Returns 0.0 if either timestamp is None (the stage was skipped or
    not reached). This is the per-stage latency helper for §22: it lets
    us report route_batch_ms / symbolic_exec_ms / answer_batch_ms
    separately rather than only the total pipeline_elapsed_ms.
    """
    if start is None or end is None:
        return 0.0
    return (end - start) * 1000.0


def _validate_vector_family(vectors, family: str, flag: str) -> None:
    for vector in vectors:
        if vector.spec.family != family:
            raise ValueError(
                f"{flag} requires family={family!r}, "
                f"got {vector.spec.family!r} at layer {vector.spec.layer}"
            )


def _load_vector_cli(
    *,
    single_path: str | None,
    bundle_spec: str | None,
    deprecated_path: str | None = None,
):
    """Load vectors from CLI args. Returns ``(vectors, source_paths)``.

    ``source_paths`` parallels ``vectors`` and records where each vector was
    loaded from, so the run manifest can cite the on-disk file. For bundle
    JSON manifests, paths are resolved to absolute form.
    """
    provided = [x for x in (single_path, bundle_spec, deprecated_path) if x]
    if len(provided) > 1:
        raise ValueError("provide only one steering-vector source for each vector family")
    if bundle_spec:
        vectors = load_vector_bundle(bundle_spec)
        candidate = Path(bundle_spec)
        paths: list[str] = []
        if candidate.suffix.lower() == ".json" and candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            raw_vectors = payload.get("vectors") if isinstance(payload, dict) else payload
            for item in raw_vectors:
                if isinstance(item, str):
                    paths.append(str((candidate.parent / item).resolve()))
                elif isinstance(item, dict):
                    paths.append(str((candidate.parent / item["path"]).resolve()))
        else:
            paths = [
                str(Path(part.strip()).resolve())
                for part in bundle_spec.split(",")
                if part.strip()
            ]
        return vectors, paths
    path = single_path or deprecated_path
    if path:
        return [load_vector(path)], [str(Path(path).resolve())]
    return [], []


def _effective_alphas(vectors, override: float | None) -> list[float]:
    if override is None:
        return [float(vector.spec.alpha) for vector in vectors]
    return [float(override)] * len(vectors)


def _batch_anchor_indices(
    rendered: list[str],
    tokenizer,
    anchor: str,
    encoded,
) -> list[int]:
    unpadded_anchor_indices = [
        find_anchor_token_index(prompt, tokenizer, anchor)
        for prompt in rendered
    ]
    lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in rendered
    ]
    padded_len = int(encoded["input_ids"].shape[1])
    return [
        anchor_index + (padded_len - length)
        for anchor_index, length in zip(unpadded_anchor_indices, lengths)
    ]


def _generate_batch(
    prompts: list[str],
    model,
    tokenizer,
    *,
    max_new_tokens: int,
    seed: int,
    steering_vectors: Sequence[Any] | None = None,
    steering_alphas: Sequence[float] | None = None,
    steering_scope: str = "all",
    prompt_format: str = "raw",
    steering_anchor: str | None = None,
    telemetry_sink: list | None = None,
) -> list[str]:
    """Generate a batch of completions with optional composite steering.

    If ``telemetry_sink`` is provided, per-layer activation statistics
    (|h|, |αv|, |αv|/|h|, cos shift) are appended on each forward pass
    where steering is applied. See CLAIMS.md §20.
    """
    import torch

    if not prompts:
        return []

    vectors = list(steering_vectors or [])
    if steering_alphas is None:
        alphas = [float(vector.spec.alpha) for vector in vectors]
    else:
        alphas = [float(value) for value in steering_alphas]
    if len(vectors) != len(alphas):
        raise ValueError("steering_alphas length must match steering_vectors length")
    for vector in vectors:
        validate_vector_for_model(model, vector)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    rendered = [
        _format_for_model(prompt, tokenizer, prompt_format)
        for prompt in prompts
    ]

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires pad_token_id or eos_token_id for batched generation")
        tokenizer.pad_token = tokenizer.eos_token

    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = previous_padding_side

    target_indices: int | list[int] = -1
    if vectors and steering_scope == "last" and steering_anchor:
        target_indices = _batch_anchor_indices(
            rendered,
            tokenizer,
            steering_anchor,
            encoded,
        )

    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    ctx = nullcontext()
    if vectors:
        configs = [
            {
                "layer_index": vector.spec.layer,
                "vector": vector.values,
                "alpha": alpha,
                "token_scope": steering_scope,
                "target_token_index": target_indices,
            }
            for vector, alpha in zip(vectors, alphas)
        ]
        ctx = multi_layer_residual_addition_hook(model, configs, telemetry_sink=telemetry_sink)

    with torch.inference_mode(), ctx:
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = int(encoded["input_ids"].shape[1])
    return [
        tokenizer.decode(tokens[input_len:], skip_special_tokens=True).strip()
        for tokens in output
    ]


def _generate(
    prompt: str,
    model,
    tokenizer,
    *,
    max_new_tokens: int,
    seed: int,
    steering_vector=None,
    steering_scope: str = "all",
    prompt_format: str = "raw",
    steering_alpha_override: float | None = None,
    steering_anchor: str | None = None,
) -> str:
    """Backward-compatible single-item wrapper used by tests and utilities."""
    vectors = [steering_vector] if steering_vector is not None else []
    alphas = (
        [steering_alpha_override]
        if steering_vector is not None and steering_alpha_override is not None
        else None
    )
    return _generate_batch(
        [prompt],
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        seed=seed,
        steering_vectors=vectors,
        steering_alphas=alphas,
        steering_scope=steering_scope,
        prompt_format=prompt_format,
        steering_anchor=steering_anchor,
    )[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/v0_splits/D3.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--routes-output")
    ap.add_argument("--model")
    ap.add_argument("--procedural-memory")
    ap.add_argument(
        "--execution-mode",
        choices=[
            "baseline",
            "memory",
            "sham",
            "auto",
            "symbolic",
            "hybrid",
            "steered_auto",
        ],
        default="baseline",
    )

    ap.add_argument("--tool-steering-vector")
    ap.add_argument(
        "--tool-steering-vectors",
        help="Comma-separated NPZ paths or JSON vector-bundle manifest",
    )
    ap.add_argument(
        "--tool-steering-alpha",
        type=float,
        help="Common alpha override for every tool-policy vector in the bundle",
    )

    ap.add_argument("--reasoning-steering-vector")
    ap.add_argument(
        "--reasoning-steering-vectors",
        help="Comma-separated NPZ paths or JSON vector-bundle manifest",
    )
    ap.add_argument(
        "--reasoning-steering-alpha",
        type=float,
        help="Common alpha override for every reasoning-policy vector in the bundle",
    )
    ap.add_argument(
        "--steering-vector",
        help="Deprecated alias for --reasoning-steering-vector",
    )

    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="raw")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--route-max-new-tokens", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--route-batch-size", type=int)
    ap.add_argument(
        "--route-decision-mode",
        choices=["auto", "logit", "generate"],
        default="auto",
    )
    ap.add_argument(
        "--route-token-resolver",
        choices=["isolated", "contextual"],
        default="contextual",
        help=(
            "How to resolve route-label token IDs for direct-logit routing. "
            "'contextual' (v0.3.6 default) derives the continuation token "
            "from the actual rendered prompt, eliminating the BPE boundary "
            "assumption that fails for Qwen2.5 / Llama-3 / Mistral. "
            "'isolated' tokenizes labels via encode() in isolation (v0.3.4 "
            "behaviour, kept for backward comparison). See CLAIMS.md §9."
        ),
    )
    ap.add_argument(
        "--allow-first-token-fallback",
        action="store_true",
        help=(
            "v0.3.6: when a route label tokenizes to multiple sub-words "
            "(e.g. Qwen2.5 'SYMBOLIC' → [' SY','MBOL','IC']), reduce it to "
            "its first continuation token so the logit-contrast path stays "
            "available instead of falling through to autoregressive "
            "generation. See Phase 1.1."
        ),
    )
    ap.add_argument("--route-logit-threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dataset-split",
        default="dev",
        help=(
            "Split label recorded in the run manifest. One of: train, capture, "
            "dev, validation_alpha, validation_threshold, test, final_test. "
            "Used for anti-circularity validation; does not change execution."
        ),
    )
    ap.add_argument(
        "--label-field",
        help=(
            "Task field containing explicit route labels. Recorded in the run "
            "manifest as dataset.label_field. Does not change execution."
        ),
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
        "--run-id",
        help=(
            "Stable identifier for this run, recorded in the manifest. "
            "Defaults to a timestamp-derived id."
        ),
    )
    ap.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing the <output>.manifest.json sibling file.",
    )
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    route_batch_size = args.route_batch_size or args.batch_size
    if route_batch_size <= 0:
        raise ValueError("--route-batch-size must be positive")

    reasoning_vectors, reasoning_vector_paths = _load_vector_cli(
        single_path=args.reasoning_steering_vector,
        bundle_spec=args.reasoning_steering_vectors,
        deprecated_path=args.steering_vector,
    )
    tool_vectors, tool_vector_paths = _load_vector_cli(
        single_path=args.tool_steering_vector,
        bundle_spec=args.tool_steering_vectors,
    )
    _validate_vector_family(
        reasoning_vectors,
        "reasoning_policy",
        "--reasoning-steering-vector(s)",
    )
    _validate_vector_family(
        tool_vectors,
        "tool_policy",
        "--tool-steering-vector(s)",
    )
    if tool_vectors and args.execution_mode != "steered_auto":
        raise ValueError(
            "--tool-steering-vector(s) are only meaningful with --execution-mode steered_auto"
        )

    tool_alphas = _effective_alphas(tool_vectors, args.tool_steering_alpha)
    reasoning_alphas = _effective_alphas(
        reasoning_vectors,
        args.reasoning_steering_alpha,
    )

    tasks = _load_tasks(Path(args.input))
    memory = _load_memory(
        Path(args.procedural_memory) if args.procedural_memory else None
    )
    config = RoutingConfig(execution_mode=args.execution_mode)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    route_path = Path(args.routes_output) if args.routes_output else None
    if route_path:
        route_path.parent.mkdir(parents=True, exist_ok=True)

    model = tokenizer = None

    def ensure_llm():
        nonlocal model, tokenizer
        if model is None:
            if not args.model:
                raise ValueError(
                    "--model is required for any task routed to or through the LLM"
                )
            model, tokenizer = _load_llm(args.model)
        return model, tokenizer

    procedures_by_index = [
        memory.retrieve(
            task.get("capability_ids", []),
            task.get("specification", ""),
        )
        for task in tasks
    ]
    decisions = [
        route_task(task, bool(procedures), config)
        for task, procedures in zip(tasks, procedures_by_index)
    ]

    metadata = [
        {
            "route_raw": None,
            "route_fallback": False,
            "route_method": None,
            "route_margin": None,
            "symbolic_fallback": False,
            "error": None,
            "planner_source": None,
            "tool_success": False,
            "started": time.perf_counter(),
            # Per-stage latency timestamps (§22). Each records the perf_counter
            # at the moment the stage completes for this task; the elapsed
            # time for each stage is computed by diffing against the previous
            # stage's timestamp. Tasks that skip a stage leave it None.
            "route_completed_at": None,
            "symbolic_completed_at": None,
            "answer_completed_at": None,
        }
        for _ in tasks
    ]

    # Stage 1: model-mediated route decisions in batches.
    if args.execution_mode == "steered_auto":
        model, tokenizer = ensure_llm()
        route_anchor = (
            tool_vectors[0].spec.anchor
            if tool_vectors
            else "ACTION:"
        )
        if tool_vectors and any(
            (vector.spec.anchor or "ACTION:") != route_anchor
            for vector in tool_vectors
        ):
            raise ValueError("all tool-policy vectors in a composite bundle must use the same anchor")

        indexed = list(enumerate(tasks))
        for batch in _chunks(indexed, route_batch_size):
            indices = [index for index, _ in batch]
            batch_tasks = [task for _, task in batch]
            actions: list[str | None] = [None] * len(batch)
            margins: list[float | None] = [None] * len(batch)
            methods: list[str | None] = [None] * len(batch)
            raws: list[str | None] = [None] * len(batch)

            if args.route_decision_mode in {"auto", "logit"}:
                rendered = [
                    _format_for_model(
                        build_route_prompt(task),
                        tokenizer,
                        args.prompt_format,
                    )
                    for task in batch_tasks
                ]
                try:
                    scored = score_route_batch_from_logits(
                        rendered,
                        model,
                        tokenizer,
                        vectors=tool_vectors or None,
                        alphas=tool_alphas or None,
                        anchor=route_anchor,
                        threshold=args.route_logit_threshold,
                        leading_space=None,
                        token_resolver=args.route_token_resolver,
                        allow_first_token_fallback=args.allow_first_token_fallback,
                    )
                    for local_index, (action, margin) in enumerate(scored):
                        actions[local_index] = action
                        margins[local_index] = margin
                        methods[local_index] = "batched_logit_contrast"
                        raws[local_index] = action.upper()
                except ValueError as exc:
                    if args.route_decision_mode == "logit":
                        raise
                    if "could not resolve" not in str(exc):
                        raise

            unresolved = [
                local_index
                for local_index, action in enumerate(actions)
                if action is None
            ]
            if unresolved:
                prompts = [
                    build_route_prompt(batch_tasks[local_index])
                    for local_index in unresolved
                ]
                generated = _generate_batch(
                    prompts,
                    model,
                    tokenizer,
                    max_new_tokens=args.route_max_new_tokens,
                    seed=args.seed + min(indices),
                    steering_vectors=tool_vectors,
                    steering_alphas=tool_alphas,
                    steering_scope="last",
                    prompt_format=args.prompt_format,
                    steering_anchor=route_anchor,
                )
                for local_index, raw in zip(unresolved, generated):
                    raws[local_index] = raw
                    actions[local_index] = parse_route_action(raw)
                    methods[local_index] = "batched_autoregressive"

            for local_index, global_index in enumerate(indices):
                task = tasks[global_index]
                procedures = procedures_by_index[global_index]
                action = actions[local_index]
                meta = metadata[global_index]
                meta["route_raw"] = raws[local_index]
                meta["route_method"] = methods[local_index]
                meta["route_margin"] = margins[local_index]
                meta["route_completed_at"] = time.perf_counter()

                if action is None:
                    meta["route_fallback"] = True
                    decisions[global_index] = route_task(
                        task,
                        bool(procedures),
                        RoutingConfig(execution_mode="auto"),
                    )
                else:
                    decisions[global_index] = decision_from_action(
                        task,
                        action,
                        has_procedures=bool(procedures),
                        reason_code="steered_router_model_decision",
                        steering_policy=(
                            "invoke_symbolic_tool" if tool_vectors else None
                        ),
                    )

    outputs: list[str | None] = [None] * len(tasks)
    llm_jobs: list[tuple[int, str]] = []

    # Stage 2: deterministic symbolic execution; enqueue LLM work.
    for index, (task, procedures, decision) in enumerate(
        zip(tasks, procedures_by_index, decisions)
    ):
        meta = metadata[index]
        if decision.backend == "symbolic":
            try:
                plan = plan_from_structured_task(
                    task,
                    reason_code=decision.reason_code,
                )
                result = execute_plan(plan)
                outputs[index] = f"FINAL: {result.value}"
                meta["planner_source"] = plan.planner_source
                meta["tool_success"] = result.verified
                meta["symbolic_completed_at"] = time.perf_counter()
                continue
            except (
                SymbolicMathError,
                SyntaxError,
                ZeroDivisionError,
                ValueError,
            ) as exc:
                if args.execution_mode == "symbolic":
                    raise
                meta["symbolic_fallback"] = True
                meta["error"] = f"{type(exc).__name__}: {exc}"
                # Preserve procedural memory on symbolic -> LLM fallback.
                prompt = _build_prompt(task, procedures)
                llm_jobs.append((index, prompt))
                continue

        prompt = _build_prompt(
            task,
            procedures if decision.use_memory else [],
        )
        llm_jobs.append((index, prompt))

    # Stage 3: batched answer generation.
    if llm_jobs:
        model, tokenizer = ensure_llm()
        for batch_number, batch in enumerate(_chunks(llm_jobs, args.batch_size)):
            prompts = [prompt for _, prompt in batch]
            if len(batch) == 1 and len(reasoning_vectors) <= 1:
                vector = reasoning_vectors[0] if reasoning_vectors else None
                alpha = reasoning_alphas[0] if reasoning_alphas else None
                generated = [
                    _generate(
                        prompts[0],
                        model,
                        tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed + batch_number,
                        steering_vector=vector,
                        steering_scope="all",
                        prompt_format=args.prompt_format,
                        steering_alpha_override=alpha,
                    )
                ]
            else:
                generated = _generate_batch(
                    prompts,
                    model,
                    tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed + batch_number,
                    steering_vectors=reasoning_vectors,
                    steering_alphas=reasoning_alphas,
                    steering_scope="all",
                    prompt_format=args.prompt_format,
                )
            for (index, _), output in zip(batch, generated):
                outputs[index] = output
                metadata[index]["planner_source"] = (
                    "steered_llm" if reasoning_vectors else "llm"
                )
                metadata[index]["answer_completed_at"] = time.perf_counter()

    if any(output is None for output in outputs):
        raise RuntimeError("internal error: one or more tasks produced no output")

    with out_path.open("w", encoding="utf-8") as out_fh:
        for task, output in zip(tasks, outputs):
            out_fh.write(
                json.dumps(
                    {"task_id": task["task_id"], "output": output},
                    ensure_ascii=False,
                )
                + "\n"
            )

    if route_path:
        with route_path.open("w", encoding="utf-8") as route_fh:
            for task, decision, meta in zip(tasks, decisions, metadata):
                route_record = {
                    "task_id": task["task_id"],
                    "capabilities": sorted(decision.assessment.capabilities),
                    "route": decision.backend,
                    "reason_code": decision.reason_code,
                    "use_memory": decision.use_memory,
                    "planner_source": meta["planner_source"],
                    "tool_success": meta["tool_success"],
                    "symbolic_fallback": meta["symbolic_fallback"],
                    "route_decision_raw": meta["route_raw"],
                    "route_decision_method": meta["route_method"],
                    "route_logit_margin": meta["route_margin"],
                    "route_decision_fallback": meta["route_fallback"],
                    "error": meta["error"],
                    "pipeline_elapsed_ms": (
                        time.perf_counter() - meta["started"]
                    )
                    * 1000.0,
                    # Per-stage latencies (§22). Each is the wall time of
                    # that stage in milliseconds. Stages that were skipped
                    # (e.g. symbolic exec for an LLM-routed task) report 0.0.
                    "route_batch_ms": _stage_elapsed_ms(
                        meta["started"], meta["route_completed_at"]
                    ),
                    "symbolic_exec_ms": _stage_elapsed_ms(
                        meta["route_completed_at"], meta["symbolic_completed_at"]
                    ),
                    "answer_batch_ms": _stage_elapsed_ms(
                        meta["symbolic_completed_at"] or meta["route_completed_at"],
                        meta["answer_completed_at"],
                    ),
                    "tool_steering": {
                        "enabled": bool(tool_vectors),
                        "vectors": [
                            {
                                "vector_id": vector.spec.vector_id,
                                "layer": vector.spec.layer,
                                "alpha": alpha,
                                "scope": "last",
                            }
                            for vector, alpha in zip(tool_vectors, tool_alphas)
                        ],
                    },
                    "reasoning_steering": {
                        "enabled": bool(reasoning_vectors),
                        "vectors": [
                            {
                                "vector_id": vector.spec.vector_id,
                                "layer": vector.spec.layer,
                                "alpha": alpha,
                                "scope": "all",
                            }
                            for vector, alpha in zip(
                                reasoning_vectors,
                                reasoning_alphas,
                            )
                        ],
                    },
                }
                route_fh.write(
                    json.dumps(route_record, ensure_ascii=False) + "\n"
                )

    # Run manifest: a sibling <output>.manifest.json capturing provenance
    # required to re-interpret the output at activation-level resolution.
    # See CLAIMS.md and docs/RUN_MANIFEST.md. Emission is best-effort: a
    # validation failure prints a warning naming the missing fields but
    # does not invalidate the output file itself.
    if not args.no_manifest:
        run_id = args.run_id or f"generate_v0_outputs_{int(time.time())}"
        extra_outputs: list[tuple[str | Path, str]] = []
        if route_path:
            extra_outputs.append((route_path, "routes"))
        try:
            sha, _ = emit_manifest(
                run_id=run_id,
                repo_root=Path(__file__).resolve().parent.parent,
                output_path=out_path,
                output_kind="answers",
                model_id=args.model,
                dataset_path=args.input,
                dataset_split=args.dataset_split,
                label_field=args.label_field,
                label_oracle_kind=args.label_oracle_kind,
                seed=args.seed,
                prompt_format=args.prompt_format,
                route_decision_mode=args.route_decision_mode,
                route_token_resolver=args.route_token_resolver,
                batch_size=args.batch_size,
                route_batch_size=route_batch_size,
                max_new_tokens=args.max_new_tokens,
                extra_outputs=extra_outputs,
                tool_vectors=tool_vectors,
                tool_vector_paths=tool_vector_paths,
                tool_alphas=tool_alphas,
                reasoning_vectors=reasoning_vectors,
                reasoning_vector_paths=reasoning_vector_paths,
                reasoning_alphas=reasoning_alphas,
                loaded_model=model if args.model else None,
                loaded_tokenizer=tokenizer if args.model else None,
            )
            print(manifest_reference_line(sha, Path(str(out_path) + ".manifest.json")))
        except ManifestValidationError as exc:
            print(
                f"warning: run manifest validation failed ({exc}); "
                "output is still valid but is not headline-eligible. "
                "See CLAIMS.md and docs/RUN_MANIFEST.md."
            )


if __name__ == "__main__":
    main()
