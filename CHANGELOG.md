# 0.3.8 (AutoLearn v2 core)

- **DEF-04 GPU Qualification Pre-Flight (Phase A)**:
    - **A1/A2 — production configs**: added
      `experiments/daph_latent_memory_v0_5_2/configs/qwen25_1_5b.yaml`
      (Qwen2.5-1.5B-Instruct, batch_size=32, layer=20) and
      `qwen25_3b.yaml` (Qwen2.5-3B-Instruct, batch_size=16, layer=24)
      targeting the repair plan §4 Task 4.1 specification. Both configs
      set `seed_count: 10`, `functional_negatives: 3`,
      `unfreeze_fraction: 0.3`, and the new
      `latent_relative_norm_limit: 0.65` (DEF-02 latent injection clamp).
      Legacy `alignment_weight` is absent (DEF-03). New test file
      `tests/test_qualification_configs.py` (28 tests) validates all
      required fields, the Gate 6 minimum seed count, the corruption-α
      endpoints (0.0, 0.5, 2.0), and per-model batch sizes.
    - **A3 — functional margin calibration**: new script
      `experiments/daph_latent_memory_v0_5_2/scripts/calibrate_margin.py`
      implements the repair plan §3 Task 3.2 procedure
      `m = 0.5 * (r_shuffled_mean - r_matched_mean)` on a 100-example IID
      probe set with an untrained instance encoder. Falls back to a 0.5-nat
      floor when the gap is non-positive (no causal signal at init).
      `train_instance_encoder.py` gains `--calibrate-margin` (inline probe)
      and `--margin-override <path>` (load a calibration JSON) flags; the
      calibration report is persisted next to the Phase B checkpoint for
      provenance. New test file `tests/test_calibrate_margin.py` (4 tests)
      validates the calibration logic with a fake model (no GPU required).
    - **A4 — LatentInjector safety clamp (DEF-02 extension)**:
      `experiments/daph_latent_memory_v0_5_2/src/daph_latent_memory/latent/injection.py`
      gains `set_relative_norm_limit(limit)` and per-row latent norm
      clamping in `build()`. When `limit > 0`, each row's latent is scaled
      by `min(1, limit * ||prompt_emb_i|| / ||latent_i||)` so the injected
      latent norm never exceeds `limit * ||prompt_emb||`. This extends the
      v0.3.8 DEF-02 steering safety guarantee from the residual-hook path
      to the latent-injection path. The per-row multipliers are recorded in
      `last_clamp_multipliers` for telemetry. `train_skill_bank.py`,
      `train_instance_encoder.py`, and `train_joint.py` read
      `training.latent_relative_norm_limit` from the config and enable the
      clamp when > 0. New test file `tests/test_injection_safety.py`
      (10 tests) validates default-off behavior, per-row clamping,
      geometry preservation, and label/mask invariance.
- **Engineering Repair & Remediation (DEF-01..DEF-05)**:
    - **DEF-01 — full-sequence logit route scoring** (Phase 1 Task 1.1):
      `src/daph_learning/routing/logit_router.py` gains
      `score_route_batch_sequence_from_logits` which computes
      `Score(Label|x) = sum_j log P(t_j | x, t_<j)` over the complete label
      token sequence. This eliminates the single-token contrast failure on
      multi-token BPE/SentencePiece tokenizers (e.g. Qwen2.5 `"SYMBOLIC"` →
      `[" SY", "MBOL", "IC"]`) without autoregressive generation fallback.
      New helpers `resolve_route_label_token_sequences` and
      `route_action_from_sequence_scores` in `steered_router.py`. Two
      teacher-forced forward passes (one per label) replace the single-token
      contrast; cost is ~2x the single-token path. New test file
      `tests/test_sequence_route_scoring.py` (11 tests).
    - **Phase 1 Task 1.2 — prompt terminal boundary alignment enforcement**:
      `detect_route_prompt_alignment()` is now applied across all batching
      entry points in `src/daph_learning/routing/batched.py`, trimming
      trailing prompt whitespace to guarantee a canonical `"ACTION:" + " LABEL"`
      tokenization boundary regardless of chat-template padding.
    - **DEF-02 — steering perturbation safety limits** (Phase 2 Task 2.1):
      new `SafetyLimits` dataclass in `src/daph_learning/steering/hooks.py`
      with defaults `R_pert ≤ 0.65`, `cosine_shift ≤ 0.25`, `KL_drift ≤ 0.50`.
      `residual_addition_hook` and `multi_layer_residual_addition_hook`
      auto-clamp the effective alpha so `|αv|/|h|` does not exceed
      `max_relative_perturbation`, preventing the unbounded-alpha
      residual-stream drift that causes autoregressive repetition loops.
      Telemetry records `safety_clamp_multiplier` and `safety_breaches`;
      `fail_closed=True` raises `SteeringApplicationError` on breach. New
      test file `tests/test_steering_safety_limits.py` (7 tests).
    - **Phase 2 Task 2.2 — mandatory cosine decay scheduling for reasoning
      steering**: `scripts/generate_v0_outputs.py` now wires
      `decay_schedule="cosine"` (default), `decay_steps=16`,
      `decay_min_multiplier=0.1` into all reasoning-policy generation passes
      via new CLI flags `--reasoning-decay-schedule`,
      `--reasoning-decay-steps`, `--reasoning-decay-min-multiplier`, and
      `--no-safety-clamp`. The decay attenuates `token_scope="all"` steering
      over generated tokens so the intervention fades out instead of
      pinning the model into a degenerate loop.
    - **DEF-03 — latent memory v0.5.2 loss migration**: verified complete.
      `experiments/daph_latent_memory_v0_5_2/src/daph_latent_memory/training/losses.py`
      contains `functional_mismatch_loss` (hinge with margin `m`), `disentangle_loss`,
      `invariance_loss`, `total_loss_v052`, and `compute_R`. The legacy
      `hidden_alignment_loss` / cosine alignment loss is fully purged. Hard
      negative sampling (`negatives.py`) covers 5 categories with K=3.
      `test_losses.py` (10 tests) + `test_negatives.py` (4 tests) pass.
    - **DEF-05 — anti-circularity & leakage audits**: verified complete.
      `src/daph_learning/evaluation/manifest.py` enforces
      `validate(manifest, headline=True)` and rejects any test-split run
      where `vector.capture_dataset_sha256 == dataset.sha256`.
      `src/daph_learning/evaluation/leakage.py` provides `detect_leakage()`
      (exact, normalized, family, template, fingerprint overlap) and
      `family_aware_split()`. 31 dedicated tests pass.
    - **DEF-04 — real-model GPU qualification**: pipeline validated
      end-to-end on CPU with the release-gates checker
      (`experiments/daph_latent_memory_v0_5_2/scripts/check_release_gates.py`)
      and dataset generator (`generate_dataset.py`). All 6 release gates
      (specificity, corruption, semantic mismatch, OOD benefit,
      non-catastrophic degradation, reproducibility) correctly pass when
      criteria are met and fail when not. The actual multi-seed GPU
      training on Qwen2.5-1.5B/3B-Instruct requires GPU hardware and is
      the remaining deferred execution step.
- **AutoLearn v2 — empirical counterfactual policy learning**: replaced the
  contrastive-vector loop with a reward-gap-driven incremental update system.
  New modules under `src/daph_learning/autolearn_v2/`:
    - `experience.py` — `BackendOutcome`, `Experience` dataclasses with
      deterministic fingerprints and experience IDs
    - `counterfactual.py` — counterfactual execution collector (both backends
      run on every training task)
    - `reward.py` — `UtilityConfig`, `backend_reward`, `reward_gap`,
      `optimal_action` (explicit, configurable, independently-testable)
    - `replay.py` — bounded replay buffer with priority, dedup, balancing
    - `policies/` — `SingleVectorPolicy` (trust-region incremental),
      `MultiVectorPolicy`, `ConditionalSteeringPolicy`
    - `updater.py` — reward-gap objective + trust-region candidate update
    - `acceptance.py` — acceptance gate (utility gain, domain regression,
      route collapse, displacement, abstain spike checks)
    - `registry.py` — immutable policy lineage with rollback
    - `checkpoint.py` — atomic checkpoint save/restore
    - `observability.py` — JSONL iteration telemetry
    - `engine.py` — orchestrates the full v2 loop
    - `invariants.py` — scientific invariant assertions
- **Phase 1 bug fixes**:
    - `routing/normalizer.py` — `normalize_route_result` fixes the
      generate-mode route normalization bug (`("symbolic", raw_text)` tuples
      were not unpacked)
    - `verification/` package — typed `VerificationStatus` + verifiers;
      fixes the numeric substring correctness bug (`expected=12`,
      `output="312"` was CORRECT; now INCORRECT)
- **Evaluation framework**:
    - `evaluation/leakage.py` — semantic-family-aware splitting + leak
      detection (exact, normalized, family, template, fingerprint overlap)
    - `evaluation/statistics.py` — bootstrap CIs, empirical p-values,
      McNemar test, Cohen's d, random-direction null
    - `evaluation/baselines.py` — baseline framework (always LLM,
      always symbolic, heuristic router, oracle router, gap closure)
- **CLI**: `daph-autolearn-v2` entry point + `scripts/autolearn_v2.py`
- **Example**: `configs/autolearn_v2_example.json`, `data/example_dataset.jsonl`
- **Tests**: 560+ passing tests (170 new v2 tests covering all modules +
  full engine integration with reproducibility, restart, acceptance gate,
  and definition-of-done workflow checks)

# 0.3.7 (route_fn + dependency direction + typed errors)

- **V037-001 — route_fn dead path**: `run_autolearn_loop` accepted a
  `route_fn` argument but never executed it (a conditional check silently
  bypassed the custom router). Introduced `RouteDecision` dataclass and
  `route_tasks` dispatcher in `src/daph_learning/routing/policy.py`;
  `run_autolearn_loop` now calls `route_tasks` when `route_fn` is supplied.
  Added `tests/test_route_fn_contract.py` covering the dispatch contract,
  `RouteDecision` coercion, and exception propagation (custom routers
  must not be silently swallowed).
- **V037-002 — library no longer depends on scripts/**: the v0.3.6 library
  layer imported reusable helpers from the CLI layer
  (`from scripts.evaluate_routes import ...`, `from scripts.tune_steering
  import ...`), inverting the correct dependency direction. Moved the
  reusable logic into new library modules:
    - `src/daph_learning/evaluation/routes.py` — `evaluate_route_records`,
      `load_jsonl`
    - `src/daph_learning/routing/batched.py` —
      `evaluate_batch_steered_routes`, `as_task_map`, `score_key`, `chunks`
    - `src/daph_learning/data/task_utils.py` — `format_for_model`, `load_llm`
    - `src/daph_learning/experiments/manifest.py` — `emit_manifest`,
      `manifest_reference_line`, `GitInfo`, enrich helpers
  `scripts/_manifest.py` is now a thin re-export shim for backward
  compatibility. All scripts import from `daph_learning.*` directly (no
  cross-script imports). Added `tests/test_no_scripts_imports.py` (AST +
  text scan) enforcing that no `src/daph_learning/**` module imports from
  `scripts/`. Side effect: the pre-existing
  `tests/test_v032_repairs.py` / `tests/test_v033_repairs.py` subprocess
  failures (scripts/evaluate_routes.py invoked as a subprocess could not
  import `scripts._manifest` when only `PYTHONPATH=src` was set) are now
  fixed. Full suite passes with `PYTHONPATH=src` only.
- **V037-003 — typed errors + telemetry**: replaced every
  `except Exception:` in `src/` with typed catches from a new
  `daph_learning.routing.errors` taxonomy:
    - `RouteResolutionError` (base)
    - `MultiTokenRouteError(ValueError, RouteResolutionError)`
    - `ContextBoundaryError(ValueError, RouteResolutionError)`
    - `InvalidRouteDecisionError(RouteResolutionError)`
    - `SteeringApplicationError`
    - `ModelRoutingError`
  `MultiTokenRouteError` and `ContextBoundaryError` subclass `ValueError`
  so existing `except ValueError` callers and `pytest.raises(ValueError)`
  tests keep working. Every routing fallback now emits structured
  telemetry via `daph_learning.telemetry.emit_fallback` (in-process event
  list + configurable sink). A pre-existing iteration bug in the training
  routing cascade — `zip(task_list, steered_routes)` where
  `steered_routes` is a dict, yielding keys not items — was masked by the
  old bare-except and is now fixed. Added
  `tests/test_exception_fallbacks.py` (17 tests covering the taxonomy,
  backward-compat, typed raises, telemetry, propagation).
- **V037-004 — version + claims discipline**: bumped version surfaces
  (`pyproject.toml`, `__init__.py`, `README.md`) to 0.3.7. Updated
  `CLAIMS.md` header to v0.3.7 and added §19 documenting the v0.3.7
  engineering changes and their (non-)impact on the scientific claims.
  Updated §6 test count to 340 passing. Updated §18 AutoLearn status to
  reflect that the multi-token tokenizer limitation that previously
  prevented the loop from updating is now addressed by the typed-error
  cascade with telemetry (the loop now correctly falls back to generate
  mode instead of silently swallowing).
- **V037-005 — environment provenance capture**: added
  `src/daph_learning/environment.py` with `capture_environment`, which
  uses `importlib.metadata.version` for installed-package versions
  (torch, transformers, accelerate, peft, safetensors, numpy,
  daph-learning) — the *installed* version, not the *imported* version,
  which matters for editable installs. Captures CUDA runtime + driver,
  GPU name, GPU compute capability, attention implementation, dtype, and
  device map. `fail_closed=True` raises
  `EnvironmentProvenanceError` when a REQUIRED_FOR_HEADLINE_ENV field
  cannot be captured. `emit_manifest(fail_closed_environment=True)`
  propagates the fail-closed behavior. The v0.3.6
  `_detect_environment` in `experiments/manifest.py` now delegates to
  `capture_environment`. Added `tests/test_manifest_environment.py`
  (16 tests).
- **V037-006 — proper CLI installation via entry points**: added
  `[project.scripts]` to `pyproject.toml` with 5 entry points
  (`daph-autolearn`, `daph-evaluate-routes`, `daph-build-oracles`,
  `daph-tune-steering`, `daph-random-control`). Created
  `src/daph_learning/cli/` with one thin module per entry point that
  loads the corresponding `scripts/*.py` by file path via
  `importlib.util` and calls its `main()`. After `pip install -e .`,
  the commands are on PATH and `PYTHONPATH=src:.` is no longer
  required. Added `tests/test_cli_entrypoints.py` (13 tests).

Test count: 392 passed, 1 skipped (was 320 + 4 pre-existing failures).

# 0.3.5 (manifest + claims patch)

- Added `CLAIMS.md` pinning down what each term used in the repository is
  currently licensed to assert (AutoLearn, OOD benchmark, gold route labels,
  steering improves routing, reasoning steering, test count, latency,
  provenance, tokenizer portability). Each entry carries an
  `ESTABLISHED` / `BOOTSTRAP` / `NOT YET` / `OUT OF SCOPE` status tag.
- Added `docs/RUN_MANIFEST.md` specifying the `daph.run.v1` run manifest
  schema: model revision / config hash / dtype, tokenizer revision /
  chat-template hash, environment (torch / transformers / CUDA / GPU),
  dataset SHA-256 + split + `label_oracle_kind`, decoding (seed, prompt
  format, `route_token_resolver`), per-vector provenance, and pytorch
  determinism. Three-tier validation: REQUIRED, REQUIRED_FOR_HEADLINE,
  OPTIONAL.
- Added `src/daph_learning/evaluation/manifest.py` implementing the schema
  with `build_manifest`, `validate` (with `headline=True` tier),
  `serialize`, `manifest_sha256`, `write_manifest` / `load_manifest`,
  `hash_file`, `hash_vector_values`. Exposed via
  `daph_learning.evaluation.__init__`.
- Added programmatic anti-circularity guard: a manifest whose
  `dataset.split` is `test`/`final_test` and whose vector's
  `capture_dataset_sha256` equals the manifest's own `dataset.sha256` is
  rejected as a split-leakage violation.
- Added token-resolver honesty field: `decoding.route_token_resolver`
  must be `isolated` (what v0.3.5 actually does) or `contextual` (not yet
  implemented). Setting `contextual` without the implementation is a
  validation error.
- Extended `SteeringSpec` with capture provenance fields:
  `extraction_method`, `normalization`, `positive_n`, `negative_n`,
  `capture_anchor`, `capture_prompt_format`, `capture_dataset_path`,
  `capture_dataset_sha256`. All default to `None` for backward
  compatibility; older `.npz` files still load. Headline manifest
  validation flags vectors missing these fields.
- Updated `contrastive_mean_direction` to honor `spec.normalization`
  (`"l2"` / `"none"` / `"mean_centered"`) when the explicit `normalize`
  argument is `None`. Backward-compatible default remains `True` (L2).
- Updated `extract_steering_vector.py` with `--extraction-method`,
  `--normalization`, `--positive-n`, `--negative-n`, `--capture-anchor`,
  `--capture-prompt-format`, `--capture-dataset-path`,
  `--capture-dataset-sha256` flags. `--positive-n` / `--negative-n`
  auto-fill from array shapes when omitted.
- Updated `capture_router_activations.py` to compute and emit the capture
  dataset SHA-256, positive/negative sample counts, capture anchor, and
  capture prompt format in its JSON summary, plus a ready-to-paste
  `extract_command` with all provenance flags filled in.
- Added `scripts/_manifest.py` shared helper for best-effort environment
  detection (python / torch / transformers / CUDA / GPU / git commit /
  dirty), vector-entry construction reading provenance from `SteeringSpec`
  with `vector_sha256`, and manifest emission as a
  `<output>.manifest.json` sibling file.
- Wired manifest emission into `generate_v0_outputs.py`,
  `tune_steering.py`, and `evaluate_routes.py`. Each now accepts
  `--dataset-split`, `--label-field`, `--label-oracle-kind`,
  `--run-id`, and `--no-manifest` flags. Manifest validation failures
  print a warning naming the missing fields but do not invalidate the
  output file itself.
- `_load_vector_cli` now returns `(vectors, source_paths)` so the
  manifest can cite the on-disk vector files.
- Added 25 schema tests (`tests/test_manifest.py`), 6 emission
  integration tests (`tests/test_manifest_emission.py`), 11
  provenance roundtrip tests (`tests/test_steering_provenance.py`),
  22 OOD benchmark tests (`tests/test_ood_benchmark_v035.py`),
  14 real-model integration tests
  (`tests/test_real_model_integration.py`),
  10 contextual token resolver tests
  (`tests/test_contextual_token_resolver.py`),
  8 activation telemetry tests (`tests/test_activation_telemetry.py`),
  8 latency measurement tests (`tests/test_latency_measurement.py`),
  5 alpha-grid tests (`tests/test_alpha_grid.py`),
  10 linear-probe baseline tests
  (`tests/test_linear_probe_baseline.py`), 6 random-direction
  control tests (`tests/test_random_direction_control.py`), and
  20 AutoLearn loop tests (`tests/test_autolearn_loop.py`).

# 0.3.5 (AutoLearn learning loop)

- Added `src/daph_learning/autolearn/` module implementing the iterative
  learning loop that gives the project its name. The loop:
  1. Routes training tasks with the current steering vector.
  2. Executes the chosen backend (symbolic or LLM).
  3. Classifies outcomes as correct / misrouted / unverifiable using
     `classify_outcome()`.
  4. Captures activations from correctly-routed (positive) and
     misrouted (negative) tasks.
  5. Updates the steering vector via contrastive mean difference.
  6. Evaluates on the validation set.
  7. Repeats for N iterations, tracking the best vector by validation F1.
  Early-stops when no update is possible (not enough examples).
  See CLAIMS.md §18.
- Added `scripts/autolearn.py` CLI driver with `--train-tasks`,
  `--val-tasks`, `--model`, `--layer`, `--n-iterations`, `--alpha`,
  `--anchor`, `--min-examples`, `--normalization`, `--seed`,
  `--prompt-format`, `--label-field`, `--label-oracle-kind`,
  `--output`, and `--vector-output` flags. Prints the learning curve
  (iteration, train accuracy, val F1, val accuracy, updated flag,
  vector norm) and saves the best vector to .npz.
- Added `classify_outcome()` function that determines whether a routed
  task execution was correct, misrouted, or unverifiable, using the
  expected answer and the output format. This is the core feedback
  signal for the learning loop.
- Ran the full experiment suite on Qwen2.5-1.5B-Instruct (28 layers,
  hidden_size=1536, MPS backend). Results on 50-task test set:
  - Baseline (no steering): 70% acc, 0% F1
  - Extract-once steering (layer=20, alpha=3.0, norm='none'):
    90% acc, 62% F1 — **steering causally improves routing**
  - Random-direction control: F1=0.05 ± 0.07, z=8.6
    (`real_vector_outperforms`) — the direction matters, not just
    magnitude
  - Linear probe: 100% CV accuracy — activations perfectly separate
    symbolic/LLM, but probe weights as steering direction produce
    F1=0.0 (classification ≠ causal intervention)
  - AutoLearn loop: did not update (logit routing fails for Qwen's
    multi-token tokenizer; needs generate-mode support)
  - Key finding: `normalization='none'` is critical — L2-normalized
    vectors (norm=1) have no effect because the residual stream norm
    is ~71. The raw mean difference (norm~15) at alpha=3.0 produces
    a perturbation that is ~60% of the residual stream norm.

# 0.3.5 (alpha-grid + linear-probe baseline + random-direction control)

- Added `--alpha-grid` to `tune_steering.py` for independent per-vector
  composite coefficient sweeping. Each grid entry is a comma-separated
  list of alphas, one per vector in the bundle. Example:
  `--alpha-grid 0.5,1.0 1.0,2.0` sweeps two configurations with
  independent per-vector alphas. Mutually exclusive with `--alphas`.
  Results record `scale_semantics: "independent_per_vector"`. See
  CLAIMS.md §10 and audit §10.
- Added `scripts/linear_probe_baseline.py`: trains a logistic regression
  on captured router activations to predict the symbolic/llm route label
  via 5-fold stratified cross-validation. Reports accuracy, precision,
  recall, F1, confusion matrix, majority-class baseline, and lift over
  baseline. Provides the steering-free upper bound required by audit §9
  for evaluating whether steering contributes beyond what a simple linear
  classifier on the same activations could do.
- Added `scripts/random_direction_control.py`: matched-norm
  random-direction control experiment. Generates N random directions
  with the same L2 norm as the real steering vector, runs the routing
  pipeline with each, and compares metrics. Reports `f1_lift` and
  `f1_z_score` with an interpretation tag
  (`real_vector_outperforms` / `within_random_range` / `underperforms`).
  This is the causal ablation required by audit §11: if random
  directions produce the same routing change, the steering vector's
  direction is not causally responsible.

- Added `resolve_route_token_ids_contextual` to
  `daph_learning.routing.steered_router`. Derives route-label token IDs
  by diffing `T(rendered_prompt + label)` against `T(rendered_prompt)`,
  eliminating the boundary assumption that fails for some BPE/SentencePiece
  tokenizers. See CLAIMS.md §9 and audit §7.
- Added `token_resolver` parameter to `score_route_batch_from_logits`
  (`"isolated"` default for backward compat, `"contextual"` for
  headline-eligible results). Wired through `generate_v0_outputs.py`
  and `tune_steering.py` via `--route-token-resolver` CLI flag.
- Added activation telemetry to `residual_addition_hook` and
  `multi_layer_residual_addition_hook`. When a `telemetry_sink` list is
  provided, the hook appends per-forward-pass stats: `layer_index`,
  `n_steered_positions`, `h_norm_mean`, `av_norm_mean`,
  `relative_perturbation` (`|αv|/|h|`), and `cosine_shift_mean`
  (`1 - cos(h, h+αv)`). Wired through `score_route_batch_from_logits`
  and `_generate_batch`. See CLAIMS.md §20 and audit §20.
- Added per-stage latency measurement to `generate_v0_outputs.py`.
  Route records now include `route_batch_ms`, `symbolic_exec_ms`, and
  `answer_batch_ms` in addition to the existing `pipeline_elapsed_ms`.
  Skipped stages report 0.0. See CLAIMS.md §22 and audit §22.

# 0.3.5 (OOD benchmark + typed oracle labels + real-model tests)

- Expanded `generate_ood_benchmark.py` from 5 to 13 categories. Added
  `nested_arithmetic`, `signed_nested_arithmetic` (symbolic group);
  `text_question`, `coding_question`, `knowledge_question`
  (non-symbolic controls); `irrelevant_numeric`, `ambiguous_tool`
  (adversarial numeric controls); `unsupported_arithmetic` (malformed,
  uses true division to test fallback). Each task carries a
  `category_group` field (`symbolic` / `non_symbolic` / `adversarial` /
  `malformed`) for aggregate analysis. Added `--symbolic-only`,
  `--non-symbolic-only`, and `--categories` CLI flags for generating
  targeted subsets.
- Added typed oracle labels: every task now carries
  `capability_oracle`, `accuracy_oracle`, and `utility_oracle` fields
  (each `"symbolic"` or `"llm"`), in addition to the backward-compatible
  `route_label` (which equals `utility_oracle`). This fixes the
  three-oracles-conflated-into-one-string problem from the audit §13.
  For example, `easy_add` now has `capability_oracle="symbolic"` (the
  tool can do it), `accuracy_oracle="llm"` (the LLM is more accurate on
  small numbers), `utility_oracle="llm"` (prefer LLM for this category).
- Updated `evaluate_route_records` to accept `label_oracle_kind` and
  include it in the returned metrics dict, so downstream consumers
  cannot silently re-interpret a `policy_heuristic` result as a
  capability oracle.
- Updated `tune_steering.py` to pass `label_oracle_kind` through to
  `evaluate_route_records`.
- Added real-model integration tests
  (`tests/test_real_model_integration.py`) using `sshleifer/tiny-gpt2`.
  Covers: layer resolution, vector validation, anchor mapping, activation
  capture, multi-layer hooks, batched generation, batch-size consistency
  (the §24 check: same prompt produces same output in batch size 1 vs
  batch), symbolic execution, symbolic→LLM fallback, extract/load
  roundtrip with provenance, chat-template guard. Skipped if torch or
  transformers is unavailable.

# 0.3.5

- Auto-detects leading-space vs raw single-token route labels for direct-logit routing.
- Added comma-separated and JSON multi-layer steering bundle loading with duplicate-layer validation.
- Added composite tool-policy and reasoning-policy steering CLI support in full output generation.
- Refactored full output generation into batched model-mediated routing and batched answer-generation stages.
- Added configurable `--batch-size` and `--route-batch-size`.
- Added composite bundle tuning; bundle sweep alpha acts as a global multiplier over stored per-layer alphas.
- Added McNemar continuity-corrected statistic/p-value, discordant odds ratio, and exact Clopper-Pearson confidence intervals.
- Extended V0 hint-ablation gate JSON telemetry with paired-effect confidence intervals.

# 0.3.4

- Added direct next-token logit-contrast routing with explicit single-token label validation.
- Added `route-decision-mode={auto,logit,generate}` and route margin/method telemetry.
- Added batch-aware anchor steering for variable-length left-padded prompt batches.
- Added multi-layer composite residual steering context manager.
- Refactored steering tuning to batched single-forward logit scoring with configurable batch size and margin threshold.
- Added deterministic, balanced, seed-controlled OOD benchmark generation with SHA-256 output.
- Added regression tests for logit routing, composite hooks, batch target indices, and benchmark determinism.

# 0.3.3

- Fixed malformed-route confusion accounting so malformed LLM-oracle decisions no longer inflate true negatives or specificity.
- Added reusable `evaluate_route_records()` for route evaluation and steering tuning.
- Added regex-based embedded `$var` discovery/substitution across nested symbolic DAG arguments.
- Added explicit `target_token_index` support to residual steering and activation capture.
- Added `find_anchor_token_index()` so `ACTION:` can be targeted inside fully rendered chat templates instead of steering the appended assistant-generation header.
- Connected anchor-aligned capture and injection paths.
- Added runtime alpha overrides for tool-policy and reasoning-policy steering.
- Added real `scripts/tune_steering.py` layer/alpha validation sweep that loads the model once, runs steered routing, scores route metrics, and selects the best configuration.
- Added v0.3.3 regression tests for TN accounting, embedded variables, target-token steering, anchor alignment, and tuning selection logic.

# 0.3.2

- Preserve retrieved procedural memory when symbolic execution fails over to the LLM.
- Count malformed route decisions in the binary confusion matrix instead of silently skipping them; report malformed rate and route accuracy.
- Make `token_scope="last"` a one-shot prompt-final-token intervention, protected against cached and non-cached autoregressive decode reapplication.
- Isolate arithmetic text before trailing natural-language/numeric instructions.
- Extend `ExecutionPlan` to multi-action dependency DAGs via `output_var`, `$var` references, topological scheduling, cycle detection, and optional `result_var`.
- Add regression tests for all v0.3.2 repairs.

# 0.3.1

- Connected ExecutionPlan actions to actual symbolic dispatch; removed gold-answer leakage from runtime verification.
- Added importable scripts package and fixed `inspect_integrity.py`.
- Added `steered_auto`: tool-policy steering now causally participates in the LLM call/no-call routing pass.
- Split tool-policy steering from reasoning-policy steering with family validation.
- Added token-scoped residual hooks (`last` for route action, `all` for reasoning experiments).
- Added vector/model dimensional validation and richer steering metadata.
- Added route evaluator, exact paired V0 hint-ablation gate, and contrastive steering-vector extraction CLI.
- Corrected `scientific_gate` semantics: integrity/paired checks are separate from the new `effect_size_pass`.
- Added optional explicit route-label fields for activation capture and routing evaluation to avoid circular policy-oracle evaluation.
- Centralized output parsers under `daph_learning.evaluation`.
- Added `mod` modulus alias and stronger task-schema/duplicate-ID validation.
- Added raw/chat prompt-format switch to preserve historical compatibility while supporting instruct chat templates.
- Added expanded integration/security tests and CLI smoke tests.

# 0.3.0

- Fixed procedural-memory loading path.
- Replaced raw/symbolic eval assumptions with typed exact integer execution.
- Added bounded AST fallback.
- Added explicit capability assessment and routing decision objects.
- Added ExecutionPlan / SymbolicAction / ExecutionResult IR.
- Added auditable route logging.
- Added deterministic symbolic, auto, and hybrid execution modes.
- Added stable `FINAL:` scoring contract.
- Added activation steering vector types, extraction, residual hooks, and serialization.
- Added integrity, memory, router, scorer, and symbolic-security tests.
