# DAPH AutoLearn v0.3.6 Baseline (frozen)

This directory is the **frozen baseline** for v0.3.6, captured before any v0.3.7
repair work begins. It exists so that later gains (or regressions) are measurable
against a known reference point. **Do not overwrite after development starts.**

Captured at commit `e6d45e0bb8d121807f14c150537b480f76e31b6d` (branch `main`, dirty=False).

## Current implementation status

| Component                  | Status             |
|----------------------------|--------------------|
| symbolic executor          | implemented        |
| fixed steering             | implemented        |
| balanced optimizer         | implemented        |
| empirical oracle           | implemented        |
| AutoLearn loop             | experimental       |
| AutoLearn qualification    | not established    |


## Captured artifacts

| File                          | Contents                                                          |
|-------------------------------|-------------------------------------------------------------------|
| `git_state.json`              | Commit SHA, branch, dirty flag, remote, `git describe`            |
| `environment.json`            | Python, platform, package versions, torch/CUDA state              |
| `experiment_config.json`      | Headline experiment configuration (model, layer, alpha, ...)      |
| `metrics.json`                | All metrics from `experiments/results/full_experiment.json`       |
| `random_control_outputs.json` | Current random-direction control outputs, preserved as-is         |
| `hashes.json`                 | SHA-256 + size of every tracked baseline file                     |

## Hashed baseline files

21 files hashed. See `hashes.json` for the full mapping. Highlights:

- `experiments/results/full_experiment.json` — headline experiment bundle
- `experiments/results/positive_activations.npy` — positive-class activations
- `experiments/results/negative_activations.npy` — negative-class activations
- `experiments/data/{train,val,test}.jsonl` — experiment splits
- `data/ood_demo.jsonl` — OOD demo tasks
- `.v034_verify/`, `.v035_verify/` — prior release verification fixtures
- `pyproject.toml`, `README.md`, `CLAIMS.md`, `CHANGELOG.md`, `ROADMAP.md`

## Environment summary

- Python: 3.12.0
- torch: 2.10.0 (CUDA available: False)
- transformers: 5.13.0
- accelerate: 1.13.0
- safetensors: 0.8.0
- peft: None
- Host: macOS, no CUDA. GPU provenance fields are null.

## Headline experiment config

{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "layer": 20,
  "alpha": 3.0,
  "anchor": "ACTION:",
  "normalization": "none",
  "prompt_format": "chat",
  "n_train": 100,
  "n_val": 50,
  "n_test": 50
}

## Notes

- The headline experiment uses `Qwen/Qwen2.5-1.5B-Instruct`, layer 20, alpha 3.0,
  anchor `ACTION:`, no normalization, chat prompt format, 100/50/50 split.
- The current random-direction control methodology (5 vectors, 20-example subset,
  z-score headline) is preserved here verbatim and will be **replaced** in v0.3.8
  per `V038-005`. It is kept only for historical comparability.
- `model revision` and `tokenizer revision` are not captured at this baseline
  because the v0.3.6 manifest code does not record them; this gap is fixed in
  `V037-005`.
