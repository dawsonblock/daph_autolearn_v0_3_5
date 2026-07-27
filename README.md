# DAPH AutoLearn v0.3.7

**A falsifiable research harness for LLM tool-routing via residual activation steering.**

DAPH AutoLearn connects task routing, symbolic execution, activation steering, and
iterative learning into a single auditable pipeline:

```text
task → capability assessment → route policy → optional tool-policy steering
     → ExecutionPlan → typed symbolic executor → external evaluator
```

The system decides whether an arithmetic task should be handled by a deterministic
symbolic engine or by an LLM. Activation steering vectors — extracted from the
model's own residual stream — can bias that routing decision at inference time
without fine-tuning. An iterative learning loop refines the steering vector from
execution outcomes (correct vs. misrouted), giving the project its name.

> **Status disclaimer:** This is a research scaffold, not a production system.
> See [`CLAIMS.md`](CLAIMS.md) for the precise set of licensed claims. The term
> "AutoLearn" refers to the learning-loop infrastructure; not all claims are
> fully established yet.

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Deterministic Symbolic Execution](#deterministic-symbolic-execution)
  - [LLM Baseline Generation](#llm-baseline-generation)
  - [Steered Routing](#steered-routing)
  - [Activation Capture](#activation-capture)
  - [Steering Vector Extraction](#steering-vector-extraction)
  - [Steering Tuning](#steering-tuning)
  - [AutoLearn Loop](#autolearn-loop)
  - [Composite Multi-Layer Steering](#composite-multi-layer-steering)
  - [OOD Benchmark Generation](#ood-benchmark-generation)
  - [Evaluation](#evaluation)
  - [V0 Paired Gate](#v0-paired-gate)
  - [Causal Ablation Controls](#causal-ablation-controls)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Safety Boundary](#safety-boundary)
- [Claims & Provenance](#claims--provenance)
- [Documentation](#documentation)
- [Changelog](#changelog)

---

## Architecture

The pipeline runs in five stages, each auditable and independently testable:

```text
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐    ┌────────────┐
│  Task    │───▶│  Capability  │───▶│  Route Policy │───▶│  Execution    │───▶│ Evaluator  │
│  Input   │    │  Assessment  │    │  (± steering) │    │  Plan         │    │            │
└─────────┘    └──────────────┘    └───────────────┘    └───────────────┘    └────────────┘
                                          │                     │
                                          ▼                     ▼
                                   ┌─────────────┐      ┌──────────────┐
                                   │ Tool-Policy │      │  Symbolic    │
                                   │ Steering    │      │  Executor    │
                                   │ Vector      │      │  (bounded)   │
                                   └─────────────┘      └──────────────┘
```

### Core Modules

| Module | Path | Description |
|--------|------|-------------|
| **Routing** | `src/daph_learning/routing/` | Capability assessment, deterministic `auto` router, model-mediated `steered_auto` router, direct-logit routing |
| **Execution** | `src/daph_learning/execution/` | `ExecutionPlan` / `SymbolicAction` / `ExecutionResult` IR, typed symbolic executor with bounded AST |
| **Steering** | `src/daph_learning/steering/` | `SteeringSpec` / `SteeringVector` types, contrastive mean-difference extraction, residual-stream hooks, anchor-aware injection, steering-vector optimization |
| **AutoLearn** | `src/daph_learning/autolearn/` | Iterative learning loop: route → execute → classify outcome → re-capture → update vector → validate |
| **Evaluation** | `src/daph_learning/evaluation/` | Answer scoring, paired binomial / McNemar statistics, run manifest schema (`daph.run.v1`) |
| **Memory** | `src/daph_learning/memory/` | `ProceduralMemory` for static procedure retrieval by capability ID |
| **Tools** | `src/daph_learning/tools/` | Bounded symbolic math engine (no `eval`, no `sympy.sympify`) |

### Two Steering Vector Families

| Family | Token Scope | When Applied | Purpose |
|--------|-------------|--------------|---------|
| `tool_policy` | `last` | Route decision pass only | Bias the SYMBOLIC/LLM routing decision |
| `reasoning_policy` | `all` | LLM answer generation only | Influence reasoning style after LLM route is chosen |

These families are **not interchangeable**. A tool-policy vector is injected at the last token of the routing prompt; a reasoning-policy vector is injected during autoregressive decoding. This separation prevents the experimental confound where a single vector simultaneously changes routing and answer generation.

---

## Key Features

- **Falsifiable execution stack** — `ExecutionPlan.actions` are dispatched directly; the executor no longer re-reads the task. Gold answers are used only by evaluation scripts, never by runtime verification.
- **Model-mediated steering** — `steered_auto` performs a real LLM `SYMBOLIC`/`LLM` routing pass. A `tool_policy` vector is injected at the last token position of the route prompt. Malformed route decisions fall back to deterministic `auto` routing and are explicitly logged.
- **Direct-logit routing** — When `SYMBOLIC` and `LLM` each resolve to a single next-token ID, routing can be done in one forward pass via logit contrast. Falls back to autoregressive generation otherwise. Contextual token resolver handles BPE/SentencePiece boundary differences.
- **Iterative AutoLearn loop** — Routes training tasks, executes, classifies outcomes as correct/misrouted/unverifiable, re-captures activations, updates the steering vector via contrastive mean difference, and validates — repeating for N iterations.
- **Bounded symbolic engine** — No `eval`, no `sympy.sympify`, no unrestricted parsing. Structured integer fields preferred. AST fallback rejects names, calls, attributes, subscripts, comprehensions, floats, booleans, true division, and exponentiation.
- **Run manifest schema** — `daph.run.v1` manifest records model revision, tokenizer revision, chat-template hash, environment (torch/transformers/CUDA/GPU), dataset SHA-256 + split, decoding parameters, per-vector provenance, and PyTorch determinism settings. Three-tier validation: REQUIRED, REQUIRED_FOR_HEADLINE, OPTIONAL.
- **Anti-circularity guard** — A manifest whose `dataset.split` is `test`/`final_test` and whose vector's `capture_dataset_sha256` equals the manifest's own `dataset.sha256` is rejected as split-leakage.
- **Activation telemetry** — Per-forward-pass stats: `h_norm_mean`, `av_norm_mean`, `relative_perturbation` (`|αv|/|h|`), `cosine_shift_mean` (`1 - cos(h, h+αv)`).
- **Per-stage latency** — Route records include `route_batch_ms`, `symbolic_exec_ms`, `answer_batch_ms`, and `pipeline_elapsed_ms`.
- **Causal ablation controls** — Matched-norm random-direction control and linear-probe baseline scripts.
- **Deterministic OOD benchmark** — 13-category seed-controlled benchmark with SHA-256 digest. Symbolic, non-symbolic, adversarial, and malformed task groups.

---

## Installation

```bash
pip install -e '.[full]'
```

**Requirements:** Python ≥ 3.10, numpy ≥ 1.24, packaging ≥ 23. The `[full]` extra adds torch ≥ 2.1, transformers ≥ 4.45, accelerate ≥ 0.21, safetensors ≥ 0.4, and pytest ≥ 8.0.

**Minimal install** (symbolic-only, no GPU): `pip install -e .`

**LLM-only install:** `pip install -e '.[llm]'`

---

## Quick Start

```bash
# Install with all dependencies
pip install -e '.[full]'

# Run the test suite
pytest -q

# Build procedural memory
python scripts/build_v0_memory.py

# Verify integrity
python scripts/inspect_integrity.py
```

### Verify the historical D3 dataset

The historical notebook recorded this SHA-256 digest:

```text
db6efbd9414f32ed29d5bb0c6abbcfb4acf4e9c7a7f666fa01a7804128c8d031
```

Place the original D3 file at `data/v0_splits/D3.jsonl` and verify it:

```bash
python scripts/verify_dataset.py data/v0_splits/D3.jsonl \
  --sha256 db6efbd9414f32ed29d5bb0c6abbcfb4acf4e9c7a7f666fa01a7804128c8d031
```

---

## Usage Guide

### Deterministic Symbolic Execution

Run the symbolic arm on a dataset — no model required:

```bash
python scripts/generate_v0_outputs.py \
  --input data/v0_splits/D3.jsonl \
  --output data/v0_outputs/symbolic_d3.jsonl \
  --routes-output data/v0_outputs/symbolic_d3.routes.jsonl \
  --execution-mode symbolic
```

### LLM Baseline Generation

Use `--prompt-format raw` for compatibility with the reconstructed V0 generation path. `chat` is available for instruct-model experiments but should be treated as a separate condition.

```bash
python scripts/generate_v0_outputs.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/v0_splits/D3.jsonl \
  --output data/v0_outputs/baseline_d3.jsonl \
  --execution-mode baseline \
  --prompt-format raw
```

### Steered Routing

A tool-policy vector steers the route decision. A reasoning-policy vector steers the LLM answer path:

```bash
python scripts/generate_v0_outputs.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/ood_router.jsonl \
  --output data/v0_outputs/steered_auto.jsonl \
  --routes-output data/v0_outputs/steered_auto.routes.jsonl \
  --execution-mode steered_auto \
  --tool-steering-vector artifacts/invoke_symbolic_tool.npz \
  --reasoning-steering-vector artifacts/verify_before_commit.npz \
  --prompt-format chat
```

Malformed route decisions fall back to the deterministic `auto` policy and are explicitly logged — they are never silently treated as successful steering.

**Direct-logit routing** (`--route-decision-mode auto`) uses logit contrast when `SYMBOLIC` and `LLM` each resolve to one next-token ID, falling back to autoregressive routing otherwise. Use `--route-token-resolver contextual` for headline-eligible results with tokenizers where isolated label encoding differs from continuation encoding.

### Activation Capture

Capture the last-token residual at the route decision position. The deterministic `auto` policy labels hard exact-computation tasks as symbolic positives and easy/non-symbolic controls as negatives:

```bash
python scripts/capture_router_activations.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tasks data/ood_router.jsonl \
  --layer 24 \
  --positive-out artifacts/router_pos_l24.npy \
  --negative-out artifacts/router_neg_l24.npy \
  --metadata-out artifacts/router_l24.jsonl \
  --prompt-format chat \
  --label-field route_label
```

Prefer explicit human/benchmark route labels via `--label-field`. Omitting it uses the deterministic `auto` policy as a bootstrap oracle. **Never extract vectors from the held-out final test split.**

### Steering Vector Extraction

Given matched `[N, H]` activation arrays:

```bash
python scripts/extract_steering_vector.py \
  --positive positive.npy \
  --negative negative.npy \
  --output artifacts/invoke_symbolic_tool.npz \
  --vector-id tool-v1 \
  --family tool_policy \
  --behavior invoke_symbolic_tool \
  --layer 24 \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --capture-anchor ACTION: \
  --capture-prompt-format chat \
  --capture-dataset-path data/router_train.jsonl \
  --capture-dataset-sha256 <sha256-of-capture-dataset> \
  --positive-n 200 \
  --negative-n 200
```

Capture provenance flags (`--capture-*`, `--extraction-method`, `--normalization`, `--positive-n`, `--negative-n`) are optional for engineering use but **required for headline-eligible run manifests**. `capture_router_activations.py` emits a ready-to-paste `extract_command` with all provenance flags filled in.

### Steering Tuning

Sweep layer and alpha on a frozen validation split:

```bash
python scripts/tune_steering.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --val-tasks data/router_val.jsonl \
  --label-field gold_route \
  --vector-pattern 'vectors/tool_layer_{layer}.npz' \
  --layers 20 24 28 32 \
  --alphas 0.5 1.0 1.5 2.0 2.5 \
  --prompt-format chat \
  --output results/steering_sweep.json
```

The sweep optimizes coverage-adjusted route F1, then decision coverage, then route accuracy. **Final test data must remain untouched until the configuration is frozen.**

**Independent per-vector coefficients** via `--alpha-grid`:

```bash
python scripts/tune_steering.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --val-tasks data/router_val.jsonl \
  --label-field route_label \
  --vector-bundle configs/tool_bundle.json \
  --alpha-grid 0.5,1.0 1.0,2.0 1.5,2.5 \
  --batch-size 32 \
  --output results/grid_sweep.json
```

Each grid entry is a comma-separated list of alphas, one per vector in the bundle. Mutually exclusive with `--alphas`.

### AutoLearn Loop

The iterative learning loop that gives the project its name:

```bash
python scripts/autolearn.py \
  --train-tasks data/router_train.jsonl \
  --val-tasks data/router_val.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --layer 24 \
  --n-iterations 10 \
  --alpha 3.0 \
  --anchor ACTION \
  --min-examples 5 \
  --normalization none \
  --seed 42 \
  --prompt-format chat \
  --label-field route_label \
  --label-oracle-kind utility_oracle \
  --output results/autolearn.json \
  --vector-output vectors/autolearn_best.npz
```

The loop prints a learning curve (iteration, train accuracy, val F1, val accuracy, updated flag, vector norm) and saves the best vector by validation F1.

### Composite Multi-Layer Steering

A multi-layer steering bundle can be passed directly:

```bash
python scripts/generate_v0_outputs.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/ood_v035.jsonl \
  --output data/v0_outputs/full.jsonl \
  --routes-output data/v0_outputs/full.routes.jsonl \
  --execution-mode steered_auto \
  --tool-steering-vectors 'vectors/tool_l20.npz,vectors/tool_l24.npz' \
  --reasoning-steering-vectors configs/reasoning_bundle.json \
  --batch-size 16 \
  --route-batch-size 32 \
  --route-decision-mode auto \
  --prompt-format chat
```

Bundle JSON may contain paths only or per-vector alpha overrides:

```json
{
  "vectors": [
    {"path": "../vectors/tool_l20.npz", "alpha": 0.7},
    {"path": "../vectors/tool_l24.npz", "alpha": 1.1}
  ]
}
```

Composite tuning uses stored per-layer alpha values; `--alphas` acts as a global multiplier:

```bash
python scripts/tune_steering.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --val-tasks data/router_val.jsonl \
  --label-field route_label \
  --vector-bundle configs/tool_bundle.json \
  --alphas 0.5 0.75 1.0 1.25 1.5 \
  --batch-size 32 \
  --output results/composite_sweep.json
```

### OOD Benchmark Generation

Generate a deterministic, SHA-256-stable benchmark with 13 categories:

```bash
python scripts/generate_ood_benchmark.py \
  --output data/ood_v034_seed42.jsonl \
  --num-tasks 500 \
  --seed 42
```

The generator prints the resulting SHA-256 digest. **Freeze the file after generation and do not tune on the final test split.**

Categories include:
- **Symbolic:** `easy_add`, `easy_sub`, `multiply`, `modulo`, `modular_multiplication`, `nested_arithmetic`, `signed_nested_arithmetic`
- **Non-symbolic controls:** `text_question`, `coding_question`, `knowledge_question`
- **Adversarial numeric:** `irrelevant_numeric`, `ambiguous_tool`
- **Malformed:** `unsupported_arithmetic` (true division, tests fallback)

Each task carries typed oracle fields: `capability_oracle`, `accuracy_oracle`, `utility_oracle` (plus backward-compatible `route_label` = `utility_oracle`).

### Evaluation

**Answer accuracy:**

```bash
python scripts/evaluate_outputs.py \
  --tasks data/v0_splits/D3.jsonl \
  --arm baseline=data/v0_outputs/baseline_d3.jsonl \
  --arm symbolic=data/v0_outputs/symbolic_d3.jsonl
```

For the old notebook parser, explicitly request `--parser legacy_first_int`. Do not mix parser definitions inside one comparison.

**Routing quality:**

```bash
python scripts/evaluate_routes.py \
  --tasks data/ood_router.jsonl \
  --routes data/v0_outputs/steered_auto.routes.jsonl \
  --label-field route_label
```

With `--label-field`, this compares against explicit route gold labels. Without it, the deterministic `auto` policy is used only as a bootstrap oracle. Reports precision, recall, F1, FP/FN counts, and malformed routes.

### V0 Paired Gate

Reproduce the historical V0 hint-ablation gate:

```bash
python scripts/v0_hint_ablation_gate.py \
  --tasks data/v0_splits/D3.jsonl \
  --baseline-outputs data/v0_outputs/baseline_d3.jsonl \
  --treatment-outputs data/v0_outputs/treatment_d3.jsonl \
  --parser legacy_first_int \
  --min-delta 0.05 \
  --alpha 0.01 \
  --expected-task-sha256 db6efbd9414f32ed29d5bb0c6abbcfb4acf4e9c7a7f666fa01a7804128c8d031
```

The gate reports exact paired-binomial p-value, continuity-corrected McNemar statistic/p-value, discordant odds ratio, and exact 95% Clopper-Pearson confidence intervals. The notebook's 7-improved/2-regressed strict-scoring run corresponds to a two-sided exact p-value of `0.1796875`; that value is regression-tested.

### Causal Ablation Controls

**Matched-norm random-direction control:**

```bash
python scripts/random_direction_control.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tasks data/router_val.jsonl \
  --vector vectors/tool_layer_24.npz \
  --n-random 5 \
  --seed 42 \
  --prompt-format chat \
  --output results/random_control.json
```

Generates N random directions with the same L2 norm as the real steering vector, runs the routing pipeline with each, and compares metrics. Reports `f1_lift`, `f1_z_score`, and interpretation tag (`real_vector_outperforms` / `within_random_range` / `underperforms`).

**Linear-probe baseline:**

```bash
python scripts/linear_probe_baseline.py \
  --positive artifacts/router_pos_l24.npy \
  --negative artifacts/router_neg_l24.npy \
  --output results/linear_probe.json
```

Trains a logistic regression on captured router activations via 5-fold stratified cross-validation. Reports accuracy, precision, recall, F1, confusion matrix, majority-class baseline, and lift.

---

## Project Structure

```text
daph_autolearn_v0_3_5/
├── src/daph_learning/
│   ├── autolearn/          # Iterative learning loop (route → execute → update)
│   ├── evaluation/         # Scoring, paired statistics, run manifest schema
│   ├── execution/          # ExecutionPlan IR, bounded symbolic executor
│   ├── memory/             # ProceduralMemory (static procedure retrieval)
│   ├── routing/            # Capability assessment, auto/steered/logit routers
│   ├── steering/           # SteeringSpec/Vector, extraction, hooks, optimization
│   └── tools/              # Bounded symbolic math engine
├── scripts/
│   ├── autolearn.py              # AutoLearn loop CLI driver
│   ├── build_empirical_oracles.py  # Build empirical oracle labels from model outputs
│   ├── build_v0_memory.py        # Build procedural memory
│   ├── capture_router_activations.py  # Capture residual-stream activations
│   ├── evaluate_outputs.py       # Answer accuracy evaluation
│   ├── evaluate_routes.py        # Routing quality evaluation
│   ├── extract_steering_vector.py  # Contrastive mean-difference extraction
│   ├── generate_ood_benchmark.py  # Deterministic OOD benchmark generator
│   ├── generate_v0_outputs.py    # Full pipeline output generation
│   ├── inspect_integrity.py      # Integrity verification
│   ├── linear_probe_baseline.py  # Linear-probe classification baseline
│   ├── random_direction_control.py  # Matched-norm random-direction ablation
│   ├── tune_steering.py          # Layer/alpha sweep tuning
│   ├── v0_hint_ablation_gate.py  # V0 paired binomial gate
│   └── verify_dataset.py         # SHA-256 dataset verification
├── tests/                  # 35+ test files, 200+ tests
├── docs/
│   ├── EXPERIMENT_PLAN.md  # Core ablation matrix and steering research plan
│   └── RUN_MANIFEST.md     # daph.run.v1 manifest schema specification
├── data/                   # Datasets, splits, outputs, memory
├── CLAIMS.md               # Licensed claims with ESTABLISHED/BOOTSTRAP/NOT YET status
├── CHANGELOG.md            # Version history
└── pyproject.toml          # Package configuration
```

---

## Testing

```bash
# Run all tests
pytest -q

# Run a specific test module
pytest tests/test_steered_router.py -v

# Run with coverage
pytest --cov=src/daph_learning --cov-report=term-missing
```

The test suite covers:
- Symbolic execution and bounded AST security
- Routing (deterministic, steered, direct-logit)
- Steering hooks (single/multi-layer, anchor alignment, token scope)
- Activation telemetry and latency measurement
- AutoLearn loop (outcome classification, vector updates, early stopping)
- Run manifest schema validation and emission
- OOD benchmark determinism and category coverage
- Linear-probe baseline and random-direction control
- Contextual token resolver (BPE/SentencePiece boundary handling)
- Real-model integration (using `sshleifer/tiny-gpt2`)
- Alpha-grid composite coefficient sweeping
- Steering-vector optimization
- Steering decay over autoregressive steps
- Empirical oracle building
- Manifest autofill

---

## Safety Boundary

The symbolic engine **does not use** `eval`, `sympy.sympify`, or unrestricted SymPy parsing. Structured integer fields are preferred. The bounded AST fallback rejects:

- Names, calls, attributes, subscripts, comprehensions
- Floats, booleans, true division, exponentiation

With limits on expression length, AST size/depth, input digits, and result bits.

The original D3 dataset is not fabricated inside this package. The included five-item `data/ood_demo.jsonl` is a **smoke-test demonstration only**, not a statistical benchmark.

---

## Claims & Provenance

Every result file emitted by `scripts/` should be re-interpretable against the narrower readings in [`CLAIMS.md`](CLAIMS.md). If a claim in this README is broader than the matching entry in `CLAIMS.md`, **`CLAIMS.md` is authoritative**.

Key status summary:

| Claim | Status |
|-------|--------|
| Steering mechanism (extraction, hooks, injection) | `ESTABLISHED` (engineering) |
| Steering causally improves routing (Qwen2.5-1.5B) | `ESTABLISHED` (F1 0→0.62, z=8.6 vs random) |
| Random-direction control confirms direction matters | `ESTABLISHED` |
| Linear probe 100% CV accuracy but 0% steering F1 | `ESTABLISHED` |
| AutoLearn loop converges better than extract-once | `NOT YET` (multi-token tokenizer limitation) |
| Steering generalizes across model families | `NOT YET` |
| Reasoning steering for headline results | `OUT OF SCOPE` |
| OOD benchmark scientifically qualified | `PARTIAL` (expanded, not yet validated at scale) |

Run manifests (`daph.run.v1`) record full provenance: model revision, tokenizer revision, chat-template hash, environment, dataset SHA-256 + split, decoding parameters, per-vector capture provenance, and PyTorch determinism settings. See [`docs/RUN_MANIFEST.md`](docs/RUN_MANIFEST.md) for the schema specification.

---

## Documentation

- [`CLAIMS.md`](CLAIMS.md) — Licensed claims with status tags (authoritative)
- [`CHANGELOG.md`](CHANGELOG.md) — Full version history
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — Core ablation matrix and steering research plan
- [`docs/RUN_MANIFEST.md`](docs/RUN_MANIFEST.md) — `daph.run.v1` run manifest schema

---

## Changelog

### v0.3.7

- **V037-001**: Fixed `route_fn` dead path. Custom routers supplied to
  `run_autolearn_loop` were silently ignored due to a conditional check;
  replaced with a `RouteDecision` dataclass + `route_tasks` dispatcher and
  added contract tests.
- **V037-002**: Removed the library layer's dependency on `scripts/`.
  Reusable evaluation, routing, task-formatting, and manifest logic moved
  into `src/daph_learning/{evaluation,routing,data,experiments}/`; CLI
  scripts now import from the library, never the reverse. The dependency
  direction is now `CLI → package` only.
- **V037-003**: Replaced every `except Exception:` in `src/` with typed
  catches from a new `daph_learning.routing.errors` taxonomy
  (`MultiTokenRouteError`, `ContextBoundaryError`,
  `SteeringApplicationError`, `ModelRoutingError`, ...). Every routing
  fallback now emits structured telemetry via `daph_learning.telemetry`.
  A pre-existing iteration bug in the training routing cascade — masked
  by the old bare-except — was fixed.

### v0.3.6

- Balanced negative-control optimization for empirical oracle building
- Auto-enrich manifest provenance fields from vector metadata
- Empirical oracle builder (`scripts/build_empirical_oracles.py`)
- Real-model qualification tests
- Manifest autofill tests
- Steering decay tests

### v0.3.5

- AutoLearn iterative learning loop (`src/daph_learning/autolearn/`)
- Run manifest schema (`daph.run.v1`) with three-tier validation
- Anti-circularity guard for split-leakage detection
- Contextual token resolver for BPE/SentencePiece tokenizers
- Activation telemetry (`relative_perturbation`, `cosine_shift_mean`)
- Per-stage latency measurement
- Alpha-grid independent per-vector coefficient sweeping
- Linear-probe baseline and random-direction control scripts
- OOD benchmark expanded to 13 categories with typed oracle labels
- Real-model integration tests (tiny-gpt2)
- Capture provenance fields on `SteeringSpec`
- 200+ tests added

### v0.3.4

- Direct next-token logit-contrast routing
- Multi-layer composite residual steering
- Batched single-forward logit scoring
- Deterministic OOD benchmark generator with SHA-256

### v0.3.3

- Fixed malformed-route confusion accounting
- Anchor-aligned capture and injection (`find_anchor_token_index`)
- Runtime alpha overrides for tool/reasoning policy
- `tune_steering.py` layer/alpha validation sweep

### v0.3.2

- Multi-action dependency DAGs with topological scheduling
- `token_scope="last"` one-shot intervention
- Malformed route counting in confusion matrix

### v0.3.1

- `ExecutionPlan` connected to symbolic dispatch
- `steered_auto` model-mediated routing
- Tool-policy / reasoning-policy family separation
- V0 paired hint-ablation gate
- Integrity inspection from clean checkout

### v0.3.0

- Typed exact integer symbolic execution
- Bounded AST fallback
- `ExecutionPlan` / `SymbolicAction` / `ExecutionResult` IR
- Auditable route logging
- Activation steering vector types, extraction, and residual hooks

See [`CHANGELOG.md`](CHANGELOG.md) for the full detailed history.
