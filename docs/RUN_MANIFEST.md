# Run manifest schema

A **run manifest** is a JSON document that captures everything required to
re-interpret an experimental output file at activation-level resolution.
Every result file emitted by `scripts/` should reference exactly one run
manifest by SHA-256. Without a manifest, an output file is anecdote, not
evidence.

The schema is implemented in `src/daph_learning/evaluation/manifest.py`.
This document is the human-readable spec; the Python module is authoritative
for field names and validation rules.

## Why this exists

Activation-level experiments are far more brittle than ordinary prompt-level
experiments. Two model revisions published under the same
`Qwen/Qwen2.5-3B-Instruct` name can differ in a way that rotates a steering
direction enough to flip a routing decision. Without recording the exact
revision, tokenizer, chat template, dtype, and capture conditions, a result
cannot be reproduced — and worse, cannot be *falsified*.

## Schema

```json
{
  "manifest_version": "daph.run.v1",
  "run_id": "string, caller-supplied, stable identifier",
  "created_at": "ISO-8601 UTC, e.g. 2026-07-26T12:00:00Z",
  "git_commit": "40-char hex SHA or null if not a git repo",
  "git_dirty": "bool, true if working tree had uncommitted changes",
  "daph_version": "package version, e.g. 0.3.5",

  "model": {
    "repo": "huggingface repo id, e.g. Qwen/Qwen2.5-3B-Instruct",
    "revision": "model revision SHA or null",
    "config_hash": "sha256 of model config.json, or null",
    "hidden_size": 2048,
    "num_layers": 36,
    "dtype": "float16|bfloat16|float32|auto",
    "device_map": "string or null, e.g. 'cuda:0' or 'cpu'",
    "attention_impl": "eager|sdpa|flash_attention_2|auto|null"
  },

  "tokenizer": {
    "repo": "huggingface repo id (may equal model.repo)",
    "revision": "tokenizer revision SHA or null",
    "chat_template_hash": "sha256 of applied chat template, or null",
    "pad_token_id": 151643,
    "pad_side": "left|right"
  },

  "environment": {
    "python_version": "3.10.12",
    "torch_version": "2.4.0",
    "transformers_version": "4.45.0",
    "cuda_version": "12.4 or null",
    "gpu_model": "NVIDIA H100 80GB or null",
    "os": "darwin 25.2.0"
  },

  "dataset": {
    "path": "data/ood_v035.jsonl",
    "sha256": "64-char hex digest",
    "num_tasks": 500,
    "split": "train|dev|test|capture|validation_alpha|validation_threshold|final_test",
    "label_field": "route_label|gold_route|null",
    "label_oracle_kind": "capability|accuracy|utility|policy_heuristic|null"
  },

  "decoding": {
    "seed": 42,
    "prompt_format": "raw|chat",
    "max_new_tokens": 64,
    "do_sample": false,
    "temperature": 0.0,
    "top_p": 1.0,
    "route_decision_mode": "auto|logit|generate",
    "route_token_resolver": "isolated|contextual",
    "batch_size": 16,
    "route_batch_size": 32
  },

  "vectors": [
    {
      "path": "vectors/tool_l24.npz",
      "vector_id": "tool-v1",
      "family": "tool_policy|reasoning_policy",
      "behavior": "invoke_symbolic_tool",
      "layer": 24,
      "alpha": 1.0,
      "anchor": "ACTION",
      "token_scope": "last|all",
      "hidden_size": 2048,
      "model_id": "Qwen/Qwen2.5-3B-Instruct",
      "extraction_method": "contrastive_mean_difference",
      "normalization": "l2|none|mean_centered",
      "positive_n": 200,
      "negative_n": 200,
      "capture_anchor": "ACTION",
      "capture_prompt_format": "chat",
      "capture_dataset_sha256": "64-char hex digest or null",
      "vector_sha256": "64-char hex digest of the values array"
    }
  ],

  "pytorch_determinism": {
    "seed": 42,
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "use_deterministic_algorithms": true
  },

  "outputs": [
    {
      "path": "data/v0_outputs/steered.jsonl",
      "kind": "answers|routes|sweep|paired_gate",
      "sha256": "64-char hex digest"
    }
  ]
}
```

## Required vs optional fields

The Python validator partitions fields into three tiers. A manifest that
fails `REQUIRED` validation must not be referenced by any output file that
appears in a headline result.

- **REQUIRED**: `manifest_version`, `run_id`, `daph_version`, `model.repo`,
  `tokenizer.repo`, `dataset.path`, `dataset.sha256`, `dataset.split`,
  `decoding.seed`, `decoding.prompt_format`, `decoding.route_token_resolver`,
  and one entry in `outputs`.
- **REQUIRED_FOR_HEADLINE** (in addition to REQUIRED): `model.revision`,
  `model.config_hash`, `model.dtype`, `tokenizer.revision`,
  `tokenizer.chat_template_hash`, `environment.torch_version`,
  `environment.transformers_version`, `git_commit`, and a non-empty
  `vectors` array with `vector_sha256` and `capture_dataset_sha256` on each
  entry whenever `vectors` is non-empty.
- **OPTIONAL**: everything else, including GPU/CUDA fields when running on
  CPU.

## Vector provenance

A steering vector without provenance is not reproducible. Each entry in
`vectors` must record at minimum:

- the model and layer it was extracted from;
- the dataset it was extracted on (by SHA-256);
- the extraction method and normalization;
- the positive/negative sample counts;
- the capture anchor and prompt format;
- a SHA-256 of the values array, so the file on disk can be checked against
  the manifest.

If any of those are unknown, the field must be `null` (not omitted), and the
validator will flag the manifest as unfit for headline results.

## Split discipline

`dataset.split` must be one of:

- `train` / `capture` — used to extract vectors;
- `dev` / `validation_alpha` / `validation_threshold` — used to tune alpha,
  layer, and routing threshold;
- `test` / `final_test` — touched exactly once, after the configuration is
  frozen.

A manifest whose `dataset.split` is `test` or `final_test` and whose
`vectors` array contains a vector whose `capture_dataset_sha256` equals the
manifest's own `dataset.sha256` is **circular** and must be rejected. This is
a programmatic guard against the most common split-leakage failure mode.

## Token resolver honesty

`decoding.route_token_resolver` must be one of:

- `isolated` — token IDs resolved via `tokenizer.encode(label)` in isolation.
  This is what v0.3.5 actually does. It is not guaranteed to match the
  continuation token following the rendered prompt for all tokenizers.
- `contextual` — token IDs resolved by diffing
  `T(rendered_prompt + label)` against `T(rendered_prompt)`. Not yet
  implemented.

Setting `contextual` without the matching implementation is a validation
error. This field exists so that an output file cannot silently inherit an
unqualified portability claim.

## Output referencing

Every output file should record the manifest it was produced under. The
recommended convention is to write a sibling file:

```
data/v0_outputs/steered.jsonl
data/v0_outputs/steered.jsonl.manifest.json
```

containing the manifest, and to include the manifest's own SHA-256 inside
the output file's first record when the format permits. Validators should
reject output files whose manifest SHA-256 does not match the manifest they
reference.

## Versioning

`manifest_version` is `"daph.run.v1"`. Future incompatible changes bump to
`v2`, `v3`, etc. Additive changes within `v1` are permitted and must be
ignored by older validators.
