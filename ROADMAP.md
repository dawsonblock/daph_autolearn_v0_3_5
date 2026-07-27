# DAPH AutoLearn Roadmap — v0.3.8 → v0.4.0

> **v0.3.8 status**: AutoLearn v2 core is implemented and tested (560+ passing
> tests). The counterfactual execution, reward-gap updater, acceptance gate,
> immutable lineage, checkpointing, leakage detection, and statistics
> modules are complete. The engine orchestrates the full workflow
> end-to-end with reproducibility guarantees.
>
> **What remains for v0.4.0**: real-model validation (running v2 with actual
> transformer activations on the OOD benchmark), multi-vector policy
> evaluation, and the full baseline comparison suite. The v2 core is
> structurally complete; the remaining work is empirical validation, not
> architecture.

This file is the canonical, ticket-tracked plan for taking DAPH AutoLearn from a
steering-vector research harness to a counterfactual, utility-driven,
validation-gated learning system. It is the source of truth for release
sequencing. Individual tickets should be picked up in ID order within a release;
releases must be executed in order. No ticket in a later release should be
started before the prior release's exit gate is satisfied.

Ticket ID convention: `<RELEASE>-<NNN>` (e.g. `V037-004`).
Status legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked.

---

## Objective

Convert DAPH AutoLearn from repeated steering-vector extraction into a
counterfactual, utility-driven, validation-gated learning system.

Target pipeline:

```
Task → Representation capture → Current routing policy
     → Counterfactual backend evaluation
         (SYMBOLIC → result → verifier → utility)
         (LLM      → result → verifier → utility)
     → Optimal action target
     → Weighted experience dataset
     → Steering/subspace optimizer
     → Validation gate
         ├── improvement → promote candidate
         └── no improvement → reject candidate
     → Updated policy
```

## Release stages

| Release  | Focus                                      | Exit gate                                                |
|----------|--------------------------------------------|----------------------------------------------------------|
| v0.3.7   | Correctness + architecture repair          | Tests pass; package installs cleanly; no src→scripts deps |
| v0.3.8   | Benchmark + scientific protocol rebuild    | Fixed steering validated under clean protocol           |
| v0.3.9   | Counterfactual AutoLearn                   | Several accepted iterations with held-out utility gain  |
| v0.3.10  | Low-rank conditional steering              | Conditional steering significantly beats best fixed vec |
| v0.4.0   | Full qualification release                 | Qualification gates V0–V6 pass                           |

## Current implementation status (to be frozen in Phase 0)

- symbolic executor       = implemented
- fixed steering          = implemented
- balanced optimizer      = implemented
- empirical oracle        = implemented
- AutoLearn loop          = experimental
- AutoLearn qualification = **not established**

---

## Phase 0 — Freeze Current Baseline (prerequisite to all repair work)

### `P0-001` Freeze v0.3.6 baseline
- **Priority:** P0 (blocker for v0.3.7 work)
- **Depends on:** —
- **Action:** Create `artifacts/baselines/v0.3.6/` and record:
  - git commit SHA, branch, dirty/clean state
  - `pyproject.toml` lock state, `pip freeze` output
  - Python version, PyTorch version, Transformers version, Accelerate, PEFT,
    Safetensors, CUDA runtime, CUDA driver, GPU name + compute capability
  - model revision, tokenizer revision, attention impl, dtype, device map
  - benchmark files + SHA-256 hashes
  - steering vectors + vector hashes
  - all current metrics
  - full experiment configuration
  - random-control outputs (current methodology, preserved as-is)
- **Acceptance:**
  - `artifacts/baselines/v0.3.6/` exists and is read-only in spirit
    (do not overwrite after dev starts)
  - A `BASELINE.md` in that folder enumerates every captured artifact and its
    hash
  - Status table above is copied verbatim into `BASELINE.md`
- **Do not:** overwrite this baseline after development starts.

---

## v0.3.7 — Correctness Repair

### `V037-001` Fix `route_fn` dead path
- **Priority:** P1
- **Depends on:** `P0-001`
- **Claim to verify:** `route_fn` is accepted by `run_autolearn_loop()` but
  never executed.
- **Action:**
  - Verify the claim against the actual v0.3.6 tree (read
    `src/daph_learning/autolearn/loop.py` and routing dispatch sites). Document
    findings with file/line citations before refactoring.
  - Introduce a single routing interface:
    ```python
    @dataclass
    class RouteDecision:
        task_id: str
        route: Literal["symbolic", "llm", "abstain"]
        confidence: float | None
        raw_scores: dict[str, float]
        source: str
    ```
  - Implement:
    ```python
    def route_tasks(tasks, *, model, tokenizer, steering=None,
                    route_fn=None) -> list[RouteDecision]: ...
    ```
  - Dispatch:
    - `route_fn is not None` → call user fn
    - `steering is not None` → `model_router`
    - else → `baseline_router`
- **Acceptance (tests in `tests/test_route_fn_contract.py`):**
  - custom `route_fn` called exactly once per task
  - custom result propagated
  - invalid custom result rejected with a typed error
  - `route_fn` exceptions propagate (not swallowed)
  - built-in fallback behavior unchanged

### `V037-002` Remove library dependency on `scripts/`
- **Priority:** P1
- **Depends on:** `P0-001`
- **Claim to verify:** `src/daph_learning/*` imports from `scripts/*`.
- **Action:**
  - Audit every `import scripts...` from `src/`. Record offenders.
  - Create reusable modules:
    - `src/daph_learning/evaluation/routes.py`
    - `src/daph_learning/routing/batched.py`
    - `src/daph_learning/data/task_utils.py`
    - `src/daph_learning/experiments/manifest.py`
  - Move reusable functions out of `scripts/evaluate_routes.py` and
    `scripts/tune_steering.py` into the new modules.
  - Scripts become thin CLI wrappers that import from `daph_learning`.
- **Acceptance:**
  - `tests/test_no_scripts_imports.py` passes: no `src/daph_learning/**` module
    imports from `scripts/`
  - Dependency direction is CLI → package only

### `V037-003` Replace broad exception swallowing
- **Priority:** P1
- **Depends on:** `P0-001`
- **Action:**
  - Find every `except Exception: continue` (and equivalent patterns). Record
    locations.
  - Introduce typed failures:
    ```python
    class RouteResolutionError(Exception): ...
    class MultiTokenRouteError(RouteResolutionError): ...
    class ContextBoundaryError(RouteResolutionError): ...
    class SteeringApplicationError(Exception): ...
    class ModelRoutingError(Exception): ...
    ```
  - Fall back only on expected failures (`except MultiTokenRouteError:`).
    Unexpected exceptions must terminate the run.
  - Every fallback emits structured telemetry:
    ```json
    {"event": "route_fallback", "from": "contextual", "to": "sequence",
     "reason": "multi_token_label"}
    ```
- **Acceptance:**
  - `tests/test_exception_fallbacks.py` covers each typed path
  - No bare `except Exception:` remains in `src/`

### `V037-004` Version + claims discipline
- **Priority:** P1
- **Depends on:** `P0-001`
- **Action:**
  - Synchronize `__version__`, `README.md`, `CLAIMS.md`, `pyproject.toml`,
    package metadata, and experiment manifests to `0.3.7`.
  - Replace ambiguous claim labels with the canonical ladder:
    `IMPLEMENTED`, `UNIT_VALIDATED`, `INTEGRATION_VALIDATED`,
    `EMPIRICALLY_OBSERVED`, `SCIENTIFICALLY_QUALIFIED`,
    `PRODUCTION_QUALIFIED`.
  - Example target wording:
    ```
    AutoLearn loop:
      Implementation: IMPLEMENTED
      Engineering validation: INTEGRATION_VALIDATED
      Scientific validation: NOT QUALIFIED
    ```
- **Acceptance:**
  - Every claim in `CLAIMS.md` carries one ladder label
  - All version surfaces agree on `0.3.7`
  - No claim uses undefined adjectives

### `V037-005` Environment provenance capture
- **Priority:** P1
- **Depends on:** `V037-004`
- **Action:**
  - Use `importlib.metadata.version(...)` for `transformers`, `torch`,
    `accelerate`, `peft`, `safetensors`.
  - Capture: Python, PyTorch, Transformers, Accelerate, PEFT, Safetensors,
    CUDA runtime, CUDA driver, GPU name, GPU compute capability, attention
    implementation, dtype, device map.
  - Headline result manifests must **fail closed** if required provenance is
    unavailable.
- **Acceptance:**
  - `tests/test_manifest_environment.py` asserts fail-closed behavior when
    provenance is missing
  - A real run emits a complete `environment.json`

### `V037-006` Proper CLI installation via entry points
- **Priority:** P1
- **Depends on:** `V037-002`
- **Action:**
  - Add to `pyproject.toml`:
    ```toml
    [project.scripts]
    daph-autolearn = "daph_learning.cli.autolearn:main"
    daph-evaluate-routes = "daph_learning.cli.evaluate_routes:main"
    daph-build-oracles = "daph_learning.cli.build_oracles:main"
    daph-tune-steering = "daph_learning.cli.tune_steering:main"
    daph-random-control = "daph_learning.cli.random_control:main"
    ```
  - Create `src/daph_learning/cli/` with one thin module per entry point.
  - Canonical usage becomes `pip install -e .` then `daph-autolearn ...`.
- **Acceptance:**
  - `tests/test_cli_entrypoints.py` verifies each entry point exists and
    dispatches
  - `PYTHONPATH=src:.` is no longer required

### v0.3.7 exit gate
- All `V037-*` tests pass
- `pip install -e .` works on a clean venv
- No `src/daph_learning/**` module imports from `scripts/`
- Version surfaces agree on `0.3.7`

### v0.3.7 regression suite to add
- `tests/test_route_fn_contract.py`
- `tests/test_cli_entrypoints.py`
- `tests/test_no_scripts_imports.py`
- `tests/test_exception_fallbacks.py`
- `tests/test_manifest_environment.py`

---

## v0.3.8 — Benchmark + Scientific Protocol Rebuild

### `V038-001` Eliminate duplicate examples via task fingerprints
- **Priority:** P2
- **Depends on:** v0.3.7 exit gate
- **Action:**
  - `fingerprint = sha256(normalized_semantics + task_family + normalized_parameters)`
  - Reject duplicates across train / development / calibration / test.
  - Do not compare task IDs alone.
- **Acceptance:** `tests/test_split_no_duplicates.py` passes.

### `V038-002` Grouped semantic family splits
- **Priority:** P2
- **Depends on:** `V038-001`
- **Action:**
  - Hold out complete templates or families, not random rows.
  - Families: `arithmetic_direct`, `arithmetic_nested`, `modulo`,
    `modular_multiplication`, `signed_arithmetic`, `knowledge`, `reasoning`,
    `coding`, `irrelevant_numeric`, `ambiguous_numeric`,
    `unsupported_symbolic`, `instruction_following`, `tool_trap`.
  - Recommended initial structure: 5,000 tasks = 2,500 train / 1,000 dev /
    500 calibration / 1,000 final test.
  - No template leakage across partitions.
- **Acceptance:** `tests/test_template_family_isolation.py` passes.

### `V038-003` Untouched final test set
- **Priority:** P2
- **Depends on:** `V038-002`
- **Action:**
  - Forbid use of final-test results for: layer selection, alpha selection,
    threshold tuning, token-position tuning, vector-selection tuning, prompt
    tuning, architecture tuning.
  - Once final-test results are viewed during development, retire that split.
- **Acceptance:**
  - `tests/test_final_test_immutable.py` enforces no read of final-test
    results during tuning routines
  - `tests/test_capture_test_leakage.py` detects accidental access

### `V038-004` Expand all 13 benchmark classes
- **Priority:** P2
- **Depends on:** `V038-002`
- **Action:**
  - Require final qualification over all 13 categories.
  - Add explicit adversarial categories:
    - numbers in prose but no computation
    - questions requiring external knowledge
    - unsupported algebra
    - ambiguous arithmetic
    - long nested expressions
    - negative values
    - large integers
    - division edge cases
    - modulo edge cases
    - code containing numbers
    - quoted arithmetic expressions
    - malformed symbolic requests
    - symbolic-safe but deceptive wording
- **Acceptance:** every category has ≥ N_min examples in each non-test split
  and a held-out final-test slice.

### `V038-005` Rebuild random-direction controls
- **Priority:** P2
- **Depends:** `V038-003`, `V038-004`
- **Action:**
  - Delete the old `5 random vectors / 20-example subset / z = 8.6` headline
    evidence. Preserve it only inside the v0.3.6 baseline folder.
  - New protocol: `N_random >= 500` (recommended 999).
  - Every random vector uses exactly the same tasks, layer, alpha,
    normalization, route scorer, threshold, batching, and model revision as
    the real vector.
  - Compute:
    ```
    p = (1 + Σ_i 1[M_i^random ≥ M^real]) / (N + 1)
    ```
  - Report: empirical p-value, percentile, mean null score, median null
    score, 95% null interval.
  - Do not use z-score as primary evidence.
- **Acceptance:**
  - `tests/test_random_null_distribution.py` validates the p-value formula
    on a synthetic null
  - No z-score headline remains in `README.md` / `CLAIMS.md`

### `V038-006` Stronger causal controls
- **Priority:** P2
- **Depends:** `V038-005`
- **Action:**
  - Every experiment includes: zero vector, sign-flipped vector,
    norm-matched random vector, shuffled-label vector, random-class vector,
    probe-derived vector, contrastive vector, optimized vector.
  - Strong evidence requires `optimized / contrastive > all major controls`
    on unseen data.
- **Acceptance:** `tests/test_random_null_distribution.py` extended to cover
  every control type.

### v0.3.8 exit gate
- Fixed steering validated under the clean protocol on the new splits.
- All `V038-*` tests pass.

### v0.3.8 regression suite to add
- `tests/test_split_no_duplicates.py`
- `tests/test_template_family_isolation.py`
- `tests/test_capture_test_leakage.py`
- `tests/test_final_test_immutable.py`
- `tests/test_random_null_distribution.py`

---

## v0.3.9 — Counterfactual AutoLearn

### `V039-001` Counterfactual backend execution
- **Priority:** P3
- **Depends:** v0.3.8 exit gate
- **Action:**
  - For every train/dev example where feasible, execute both SYMBOLIC and LLM.
  - Record:
    ```python
    @dataclass
    class BackendOutcome:
        backend: str
        success: bool
        answer: Any
        correct: bool | None
        latency_ms: float
        compute_cost: float | None
        confidence: float | None
        failure_type: str | None
    ```
- **Acceptance:** `tests/test_counterfactual_execution.py` produces paired
  outcomes for a small task set.

### `V039-002` Verifier abstraction
- **Priority:** P3
- **Depends:** `V039-001`
- **Action:**
  - Define `OutcomeVerifier` Protocol with `verify(task, outcome) ->
    VerificationResult`.
  - Implement: `ExactMatchVerifier`, `NumericVerifier`, `UnitTestVerifier`,
    `StructuredSchemaVerifier`, `ReferenceAnswerVerifier`,
    `LLMJudgeVerifier`, `CompositeVerifier`.
  - The router must not be its own verifier.
- **Acceptance:** unit tests for each verifier; composite verifier precedence
  rules tested.

### `V039-003` Three-action routing (add ABSTAIN)
- **Priority:** P3
- **Depends:** `V039-002`
- **Action:**
  - Change `SYMBOLIC / LLM` → `SYMBOLIC / LLM / ABSTAIN`.
  - `RouteTarget = Literal["symbolic", "llm", "abstain"]`.
  - Rationale: `symbolic wrong ∧ llm wrong` must not produce a fabricated
    preference.
- **Acceptance:** `tests/test_abstain_target.py` covers the both-fail case.

### `V039-004` Replace categorical accuracy oracle
- **Priority:** P3
- **Depends:** `V039-003`
- **Action:**
  - Record `symbolic_correct`, `llm_correct`, `accuracy_delta` where
    `+1 = symbolic only correct`, `0 = tie`, `-1 = LLM only correct`.
  - Keep utility separate from accuracy.
- **Acceptance:**
  - `tests/test_accuracy_tie.py` covers tie and both-fail cases
  - `tests/test_both_backends_fail.py` ensures no fabricated preference

### `V039-005` Explicit utility model
- **Priority:** P3
- **Depends:** `V039-004`
- **Action:**
  - `U(a, x) = w_A · A − w_L · L̃ − w_C · C̃ − w_F · F`
    - `A` correctness reward
    - `L̃` normalized latency
    - `C̃` normalized compute/token cost
    - `F` execution failure penalty
  - Bootstrap weights: correctness dominant; latency/cost secondary; failure
    large penalty.
  - `y*(x) = argmax_a U(a, x)`.
  - Store `U_symbolic`, `U_llm`, `ΔU = U_symbolic − U_llm`, target action.
- **Acceptance:** `tests/test_utility_selection.py` verifies argmax behavior
  across tie / latency-dominant / failure cases.

### `V039-006` Utility-weighted learning examples
- **Priority:** P3
- **Depends:** `V039-005`
- **Action:**
  - `w_i = g(|ΔU_i|)`, e.g. `w_i = min(w_max, |ΔU_i|)`.
  - High-confidence backend preference → more weight. Near ties → little
    weight.
- **Acceptance:** `tests/test_weighted_examples.py` verifies weighting curve
  and that abstain/both-fail cases contribute zero weight.

### `V039-007` Trust-region iterative update (replace fresh re-extraction)
- **Priority:** P3
- **Depends:** `V039-006`
- **Action:**
  - Candidate direction: `v̂_t = μ⁺_weighted − μ⁻_weighted`.
  - Update: `v_{t+1} = (1 − η_t) v_t + η_t v̂_t`, then normalize per norm
    policy. `η_0 = 0.25`, decaying.
- **Acceptance:** `tests/test_candidate_update.py` verifies the update math
  and normalization.

### `V039-008` Validation acceptance gate
- **Priority:** P3
- **Depends:** `V039-007`
- **Action:**
  - `ΔM = M(v_candidate) − M(v_current)` on development data.
  - Promote only if `Δutility > ε` AND false-positive constraint satisfied
    AND perturbation constraint satisfied. Otherwise reject.
  - Store both vectors for auditability.
- **Acceptance:**
  - `tests/test_update_acceptance.py` exercises a positive ΔM
  - `tests/test_update_rejection.py` exercises rejection on no-improvement
    and on perturbation-constraint violation

### `V039-009` Vector rollback + lineage
- **Priority:** P3
- **Depends:** `V039-008`
- **Action:** Every vector records: vector ID, parent vector ID, iteration,
  training dataset SHA, development dataset SHA, optimizer parameters,
  utility weights, layer, token location, norm, alpha, metrics before,
  metrics after, promotion decision.
- **Acceptance:** `tests/test_vector_rollback.py` restores a prior vector
  from its lineage record.

### `V039-010` Experience replay buffer
- **Priority:** P3
- **Depends:** `V039-008`
- **Action:**
  - `ExperienceBuffer` holds: hard failures, recent examples, high-|ΔU|
    examples, rare categories, previously forgotten categories.
  - Sample 50% new / 25% hard / 25% historical replay.
- **Acceptance:** `tests/test_experience_replay.py` verifies sampling
  proportions and category coverage.

### v0.3.9 exit gate
- At least several accepted iterations with held-out utility improvement on
  untouched development episodes.
- All `V039-*` tests pass.

### v0.3.9 regression suite to add
- `tests/test_accuracy_tie.py`
- `tests/test_both_backends_fail.py`
- `tests/test_utility_selection.py`
- `tests/test_abstain_target.py`
- `tests/test_counterfactual_execution.py`
- `tests/test_candidate_update.py`
- `tests/test_update_acceptance.py`
- `tests/test_update_rejection.py`
- `tests/test_vector_rollback.py`
- `tests/test_experience_replay.py`
- `tests/test_weighted_examples.py`

---

## v0.3.10 — Low-Rank Conditional Steering

### `V0310-001` Full sequence route scoring (replaces first-token routing)
- **Priority:** P4
- **Depends:** v0.3.9 exit gate
- **Action:**
  - `S(label | x) = Σ_{j=1..k} log P(t_j | x, t_<j})`
  - `m(x) = S(SYMBOLIC | x) − S(LLM | x)`
  - Removes tokenizer-specific first-token artifacts.
- **Acceptance:** `tests/test_sequence_route_scoring.py` verifies scoring on
  multi-token labels.

### `V0310-002` Dedicated route head baselines
- **Priority:** P4
- **Depends:** `V0310-001`
- **Action:**
  - Implement non-language routing: `r = W h + b` with `symbolic / llm /
    abstain`.
  - Compare: text-token router, linear route head, MLP route head.
- **Acceptance:** separates representation quality from language-token
  decoding behavior; results recorded in `metrics.json`.

### `V0310-003` Layer × token-position × alpha intervention search
- **Priority:** P4
- **Depends:** `V0310-001`
- **Action:**
  - Search `layer × token position × alpha`.
  - Candidate token positions: ACTION anchor, last user token, assistant
    header, final prompt token, first-generation position.
  - Matrix: layers {8, 12, 16, 20, 24} × 5 positions.
  - Select only using development data.
- **Acceptance:** selection script emits a development-only ranked matrix.

### `V0310-004` Trainable multi-layer steering
- **Priority:** P4
- **Depends:** `V0310-003`
- **Action:**
  - `h'_l = h_l + α_l v_l`, optimize `α = [α_1, …, α_L]` under
    `L_total = L_route + λ ||α||_1`.
- **Acceptance:** `tests/test_multilayer_coefficients.py` verifies sparsity
  pressure and that only a small number of layers get non-trivial α.

### `V0310-005` Low-rank steering basis
- **Priority:** P4
- **Depends:** `V0310-004`
- **Action:**
  - Build activation differences `D` from paired/class-separated
    representations.
  - Factorize `D = U Σ Vᵀ`, keep `K ∈ {2, 4, 8}` principal causal directions.
  - `V_K = [v_1, …, v_K]`.
- **Acceptance:** `tests/test_low_rank_basis.py` verifies orthonormality and
  reconstruction error vs K.

### `V0310-006` Conditional coefficients `c(x)`
- **Priority:** P4
- **Depends:** `V0310-005`
- **Action:**
  - Replace `h' = h + α v` with `h' = h + V_K c(x)`.
  - `c(x) ∈ ℝ^K` produced by a small controller (linear or 2-layer MLP).
  - Do not introduce another large neural network.
- **Acceptance:** `tests/test_conditional_coefficients.py` verifies shape,
  controller size budget, and intervention magnitude.

### `V0310-007` Direct causal intervention optimization
- **Priority:** P4
- **Depends:** `V0310-006`
- **Action:**
  - Objective:
    ```
    L = L_route + λ_1 ||Δh||²₂ + λ_2 D_KL(p_base^nonroute ‖ p_steered^nonroute)
    ```
  - Goals: change target route; minimize representation displacement; preserve
    unrelated model behavior.
  - This is the critical shift from "find direction that separates
    representations" to "find minimum causal intervention that improves
    utility".
- **Acceptance:** perturbation safety metrics (`||Δh||₂`, `cos(h, h')`,
  non-route KL) recorded and within budget.

### v0.3.10 exit gate
- Conditional steering significantly outperforms the best fixed-vector
  configuration on held-out data.
- All `V0310-*` tests pass.

### v0.3.10 regression suite to add
- `tests/test_sequence_route_scoring.py`
- `tests/test_multilayer_coefficients.py`
- `tests/test_low_rank_basis.py`
- `tests/test_conditional_coefficients.py`
- `tests/test_random_null_distribution.py` (extended)

---

## v0.4.0 — Full Qualification Release

### `V040-001` Required metrics dashboard
- **Priority:** P5
- **Depends:** v0.3.10 exit gate
- **Action:** Track (do not optimize F1 alone):
  - routing accuracy, macro F1
  - symbolic precision / recall
  - LLM precision / recall
  - abstain rate
  - utility, correctness, latency, token usage, compute cost
  - execution failure rate, malformed route rate
  - false symbolic rate, false LLM rate
  - calibration: ECE, Brier score, confidence-vs-correctness
- **Acceptance:** every experiment run emits all of the above in
  `metrics.json`.

### `V040-002` Perturbation safety metrics
- **Priority:** P5
- **Depends:** v0.3.10 exit gate
- **Action:** Every steering experiment records `||Δh||₂`, `cos(h, h')`, and
  `D_KL(P_base ‖ P_steered)` outside the route token distribution. Flag
  interventions that achieve route gains by globally destabilizing the model.
- **Acceptance:** flagged interventions are excluded from headline claims.

### `V040-003` Experiment artifact bundle
- **Priority:** P5
- **Depends:** `V040-001`, `V040-002`
- **Action:** Every run writes:
  ```
  run/
  ├── config.json
  ├── manifest.json
  ├── environment.json
  ├── dataset_manifest.json
  ├── vector_manifest.json
  ├── metrics.json
  ├── route_records.jsonl
  ├── backend_outcomes.jsonl
  ├── oracle_targets.jsonl
  ├── learning_curve.json
  ├── control_results.json
  ├── statistical_tests.json
  └── checksums.json
  ```
- **Acceptance:** no headline number exists without the corresponding
  reproducibility bundle.

### `V040-004` Qualification gates V0–V6
- **Priority:** P5
- **Depends:** `V040-003`
- **Action:** Pass each gate in order. Each gate has a documented pass
  condition and a held-out evaluation:
  - **V0 Representation:** route-relevant information decodable from hidden
    state; group-held-out probe > baseline.
  - **V1 Fixed causal steering:** fixed steering improves held-out routing
    without test tuning.
  - **V2 Causal controls:** real direction beats random / zero /
    sign-flipped / shuffled-label / probe direction.
  - **V3 Semantic generalization:** steering improves unseen templates,
    task families, numeric structures.
  - **V4 Optimized > extract-once:** balanced intervention optimization
    outperforms `μ⁺ − μ⁻`.
  - **V5 Conditional subspace > fixed vector:** low-rank conditional
    steering outperforms fixed rank-1 steering.
  - **V6 Iterative AutoLearn:** across successive iterations
    `U_{t+1} > U_t` on untouched development episodes; require multiple
    accepted updates.
- **Acceptance:** a `QUALIFICATION.md` records each gate's evidence bundle
  and pass/fail status.

### `V040-005` (post-v0.4 optional) Continual learning gate V7
- **Priority:** P6
- **Depends:** `V040-004`
- **Action:**
  - **V7 Continual learning:** demonstrate new task families learned without
    severe regression on historical families.
- **Acceptance:** only after V7 should the project make strong continual
    AutoLearn claims. May ship post-v0.4.0.

### `V040-006` Documentation + claims synchronization
- **Priority:** P5
- **Depends:** `V040-004`
- **Action:**
  - Update `README.md`, `CLAIMS.md`, `CHANGELOG.md` to reflect only
    qualification-gate-backed claims.
  - Synchronize `__version__` to `0.4.0` across all surfaces.
  - Publish release artifacts.
- **Acceptance:** every public claim maps to a passed gate; no aspirational
  language remains in `CLAIMS.md`.

### v0.4.0 exit gate
- Qualification gates V0–V6 pass.
- Reproducible benchmark bundle published.
- Documentation synchronized with evidence.

---

## Final target architecture

```
                        DAPH AutoLearn
                              │
                              ▼
                     Task Representation
                              │
                ┌─────────────┴─────────────┐
                │                           │
        Routing Controller          Steering Controller
                │                           │
                │                    coefficients c(x)
                │                           │
                │                   low-rank basis V
                │                           │
                └─────────────┬─────────────┘
                              ▼
                     Modified hidden state
                              │
                              ▼
                       Action decision
                  ┌───────────┼───────────┐
                  │           │           │
              SYMBOLIC       LLM       ABSTAIN
                  │           │           │
                  ▼           ▼           ▼
                Results / failure / latency / cost
                              │
                              ▼
                       Independent verifier
                              │
                              ▼
                         Utility model
                              │
                              ▼
                        Experience store
                              │
                              ▼
                    Weighted policy update
                              │
                              ▼
                      Validation gate
                       │             │
                    promote        reject
                       │
                       ▼
                    new policy
```

## Strict priority order

Do these first, in order:

1. `P0-001` — Freeze v0.3.6 baseline.
2. `V037-001` — Fix `route_fn`.
3. `V037-002` — Remove src → scripts dependencies.
4. `V037-003` — Replace broad exception swallowing.
5. `V037-004` — Repair manifest/version/claims consistency.
6. `V038-001` … `V038-003` — Rebuild benchmark without duplicate/template
   leakage and create a new untouched final test.
7. `V038-005` — Replace old random-control methodology.
8. `V039-001` … `V039-005` — Wire empirical counterfactual oracles directly
   into AutoLearn with utility.
9. `V039-006` … `V039-008` — Add utility-weighted trust-region updates with
   validation-gated promotion and rollback.

Only after those are complete:

10. `V0310-001` — Full sequence route scoring.
11. `V0310-002` — Dedicated route-head baseline.
12. `V0310-003` — Layer × token-position optimization.
13. `V0310-004` — Multi-layer steering.
14. `V0310-005` — Low-rank steering basis.
15. `V0310-006` — Task-conditioned coefficients.
16. `V0310-007` — Minimal causal intervention optimization.
17. `V040-005` — Continual AutoLearn (V7).

## Strategic conclusion

The wrong move now is adding more unrelated capabilities. The project already
has enough machinery. The highest-value progression is:

```
clean experiment
  → counterfactual utility
  → validated iterative learning
  → conditional low-rank causal steering
```

The critical breakpoint is **v0.3.9**: once each routing decision is grounded
in independently measured counterfactual utility and candidate steering
updates must beat the incumbent on held-out data, the term "AutoLearn"
becomes technically defensible.
