from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from daph_learning.routing.arithmetic_router import RoutingConfig, route_task
from daph_learning.routing.steered_router import build_route_prompt
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import capture_layer_output, resolve_transformer_layers


def _load_tasks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    ids = [str(r.get("task_id", "")) for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("task IDs must be present and unique")
    return rows


def _load_model(model_id: str):
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


def _format(prompt: str, tokenizer, mode: str) -> str:
    if mode == "raw":
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError("chat prompt format requested but tokenizer has no chat template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Capture last-token residual activations for the symbolic-tool routing contrast set"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positive-out", required=True)
    ap.add_argument("--negative-out", required=True)
    ap.add_argument("--metadata-out", required=True)
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--label-field", help="Optional task field containing explicit symbolic/llm route labels")
    ap.add_argument("--anchor", default="ACTION:", help="Textual routing anchor to capture inside the rendered prompt")
    args = ap.parse_args()

    import torch

    tasks = _load_tasks(Path(args.tasks))
    model, tokenizer = _load_model(args.model)
    layers = resolve_transformer_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"layer {args.layer} outside [0, {len(layers)})")

    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    metadata: list[dict] = []
    embed_device = model.get_input_embeddings().weight.device

    for task in tasks:
        if args.label_field:
            label = task.get(args.label_field)
            if label not in {"symbolic", "llm"}:
                raise ValueError(
                    f"task {task['task_id']} has invalid {args.label_field!r}: {label!r}"
                )
        else:
            oracle = route_task(
                task,
                False,
                RoutingConfig(execution_mode="auto"),
            ).backend
            label = "symbolic" if oracle == "symbolic" else "llm"
        rendered = _format(build_route_prompt(task), tokenizer, args.prompt_format)
        anchor_index = find_anchor_token_index(
            rendered,
            tokenizer,
            args.anchor,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(embed_device) for k, v in encoded.items()}
        sink: list = []
        with torch.inference_mode(), capture_layer_output(
            model,
            layer_index=args.layer,
            sink=sink,
            target_token_index=anchor_index,
        ):
            model(**encoded, use_cache=False)
        if len(sink) != 1:
            raise RuntimeError(f"expected one captured activation, got {len(sink)}")
        activation = sink[0]
        if activation.ndim != 2 or activation.shape[0] != 1:
            raise RuntimeError(f"unexpected captured shape: {tuple(activation.shape)}")
        vec = activation[0].numpy().astype(np.float32, copy=False)
        (positives if label == "symbolic" else negatives).append(vec)
        metadata.append({
            "task_id": task["task_id"],
            "label": label,
            "layer": args.layer,
            "model": args.model,
            "prompt_format": args.prompt_format,
            "anchor": args.anchor,
            "anchor_token_index": anchor_index,
        })

    if not positives or not negatives:
        raise ValueError(
            "contrast capture needs at least one symbolic-positive and one LLM-negative task"
        )

    pos = np.stack(positives)
    neg = np.stack(negatives)
    for path, arr in ((Path(args.positive_out), pos), (Path(args.negative_out), neg)):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    meta_path = Path(args.metadata_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        for row in metadata:
            fh.write(json.dumps(row) + "\n")

    # Compute the capture dataset SHA-256 so it can be passed through to
    # extract_steering_vector.py and recorded in the steering vector's
    # provenance. This is what makes the run manifest's anti-circularity
    # guard enforceable: a vector captured on dataset X is flagged if it
    # is later used in a test-split run on the same dataset X.
    capture_dataset_sha256 = hashlib.sha256(
        Path(args.tasks).read_bytes()
    ).hexdigest()

    print(json.dumps({
        "positive_shape": list(pos.shape),
        "negative_shape": list(neg.shape),
        "layer": args.layer,
        "model": args.model,
        "positive_n": int(pos.shape[0]),
        "negative_n": int(neg.shape[0]),
        "capture_anchor": args.anchor,
        "capture_prompt_format": args.prompt_format,
        "capture_dataset_path": args.tasks,
        "capture_dataset_sha256": capture_dataset_sha256,
        # Echo the recommended extract_steering_vector.py invocation with
        # provenance flags filled in, so the user can copy-paste it.
        "extract_command": (
            f"python scripts/extract_steering_vector.py "
            f"--positive {args.positive_out} --negative {args.negative_out} "
            f"--output <output.npz> --vector-id <id> "
            f"--family tool_policy --behavior <behavior> "
            f"--layer {args.layer} --model-id {args.model} "
            f"--capture-anchor {args.anchor} "
            f"--capture-prompt-format {args.prompt_format} "
            f"--capture-dataset-path {args.tasks} "
            f"--capture-dataset-sha256 {capture_dataset_sha256} "
            f"--positive-n {pos.shape[0]} --negative-n {neg.shape[0]}"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
