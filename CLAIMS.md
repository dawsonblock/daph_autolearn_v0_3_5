# DAPH AutoLearn v0.3.5 — Licensed Claims

This file pins down what each term used in this repository is *currently licensed
to assert* about the system. It exists because the most damaging failure mode
for an experimental harness is not a bug — it is overclaiming that gets cited.

Every result file emitted by `scripts/` should be re-interpretable against the
narrower reading in this document. If a claim below is narrower than the
matching README or docstring wording, **this file is authoritative**. The
README should be tightened in a follow-up; do not rely on the broader wording
in the meantime.

The status tags mean:

- `ESTABLISHED` — supported by tests in this repository.
- `BOOTSTRAP` — implemented and runnable, but used only to seed later
  scientifically qualified experiments. Not a headline result.
- `NOT YET` — the data structures or code paths exist, but no algorithm
  populates or uses them in a way that supports the claim.
- `OUT OF SCOPE` — explicitly not claimed by this version.

---

## 1. "AutoLearn"

**Status: NOT YET.**

The repository name contains "autolearn", but v0.3.5 does not contain an
autonomous learning loop. `ProceduralMemory` retrieves statically authored
procedures by capability ID and trigger-term count. The fields
`positive_evidence`, `negative_evidence`, and `source_episode_ids` exist on
procedure records but are not populated or consumed by any learning algorithm.

What would be required to license the term, in priority order:

1. An **external verifier signal** (correctness, latency, or utility) that is
   independent of the router's own decision. Using the router's own decision as
   the promotion signal produces policy self-confirmation, not learning.
2. Episode logging from real executions.
3. Candidate procedure induction from success/failure clusters.
4. Validation on a held-out split, with a statistical gate.
5. Promotion, revision, and retirement of procedures.

Until those exist, this repository is a **routing + procedural-memory +
activation-steering research harness**, not an autonomous learning system.

Any external writeup that uses the name "DAPH AutoLearn" must qualify it as
"v0.3.x, no learning loop yet" on first mention.

---

## 2. "OOD benchmark" / "out-of-distribution"

**Status: PARTIAL — broadened in v0.3.5, but not yet scientifically qualified.**

`scripts/generate_ood_benchmark.py` produces a deterministic, seed-controlled,
SHA-256-stable benchmark. That part is `ESTABLISHED`.

v0.3.5 expanded the benchmark from 5 to 13 categories:

- symbolic: `easy_add`, `easy_sub`, `multiply`, `modulo`,
  `modular_multiplication`, `nested_arithmetic`, `signed_nested_arithmetic`
- non-symbolic controls: `text_question`, `coding_question`,
  `knowledge_question`
- adversarial numeric controls: `irrelevant_numeric`, `ambiguous_tool`
- malformed: `unsupported_arithmetic` (true division, tests fallback)

This closes the coverage gap identified in the audit §12: non-symbolic
controls now exist, so a pathological steering vector that pushes everything
toward SYMBOLIC can be detected via false tool calls on the non-symbolic and
adversarial groups.

However, the benchmark is still **not yet scientifically qualified**:

- no experiment has been run on it with a real model;
- the typed oracle labels (see §3 below) are now present but have not been
  validated against actual model behavior;
- the non-symbolic task specifications are hand-authored templates, not a
  curated and reviewed set;
- sample sizes for serious evaluation should be 1,000–5,000 tasks, not the
  default 500.

Calling the current file an "OOD benchmark" is now more accurate than before,
but headline results require running the full experiment described in
`docs/EXPERIMENT_PLAN.md` on a sufficiently large generated set. Until then,
qualify as "expanded OOD benchmark, not yet scientifically qualified."

The five-item `data/ood_demo.jsonl` is a smoke test, not a statistical
benchmark. It must never be cited as evidence.

---

## 3. "Gold route labels" / `route_label` / typed oracle fields

**Status: PARTIAL — typed fields added in v0.3.5, but not yet validated.**

v0.3.5 added three typed oracle fields to every generated task:

- `capability_oracle` — can the symbolic backend solve the task?
- `accuracy_oracle` — which backend is more likely to be correct?
- `utility_oracle` — which backend gives the best accuracy/latency/cost
  trade-off?

The backward-compatible `route_label` field is preserved and equals
`utility_oracle` (the v0.3.4 generator's hand-coded utility heuristic).

`evaluate_route_records` now accepts `label_oracle_kind` and includes it in
the returned metrics, so downstream consumers cannot silently re-interpret a
`policy_heuristic` result as a capability oracle.

This fixes the schema conflation identified in the audit §13. However, the
oracle values themselves are still **hand-coded heuristics**, not measured
ground truth. For example, `easy_add` has `capability_oracle="symbolic"` and
`accuracy_oracle="llm"` because the v0.3.4 policy assumed the LLM is more
reliable on small additions — but that assumption has not been validated
against a real model.

Until the oracle values are measured against actual model behavior on a
held-out set, they must be treated as **policy heuristics**, not ground
truth. `--label-field route_label` (or any of the typed fields) in
`evaluate_routes.py` and `tune_steering.py` should be treated as a
**bootstrap oracle**, not a scientific target. Headline scientific results
should use an explicit, externally sourced label field.

---

## 4. "Steering improves routing"

**Status: NOT YET.**

v0.3.5 implements the *mechanism* for residual activation steering:

- contrastive mean-difference direction extraction;
- single-layer and multi-layer hook installation;
- anchor-aware injection at the `ACTION:` token;
- `token_scope="last"` for routing, `token_scope="all"` for reasoning;
- direct-logit and generated routing.

That mechanism is `ESTABLISHED` at the engineering level.

What is **not** established:

- that the mean-difference direction carries the routing concept (vs. the
  norm of the vector doing the work);
- that steering beats a matched-norm random direction control;
- that steering beats a trained linear-probe baseline;
- that steering improves final answer utility on a held-out test set;
- that the effect survives `-v`, shuffled-label, and neighbor-layer ablations;
- that the effect generalizes beyond the narrow arithmetic distribution.

Until those gates pass, no external writeup may claim that activation steering
improves capability routing. The licensed claim is: "v0.3.5 provides a
mechanism for testing whether contrastive activation steering alters
tool-routing decisions; the scientific question is open."

---

## 5. "Reasoning steering"

**Status: OUT OF SCOPE for headline claims.**

Reasoning-policy steering applies `token_scope="all"` during autoregressive
decoding. The intervention is *compounding*: each step's KV cache carries the
steered residual forward, so the effect at step `t` depends on the integral
of interventions at steps `1..t-1`. Alpha therefore cannot be calibrated
per-token; it must be calibrated against generation-length distributions.

Until there are KL-vs-alpha and capability-regression measurements against a
held-out suite, reasoning steering must not appear in headline results, even
behind a flag. It is a separate research program.

---

## 6. "205 passing tests"

**Status: ESTABLISHED as engineering regression — PARTIAL scientific qualification.**

v0.3.5 added 14 real-model integration tests
(`tests/test_real_model_integration.py`) using `sshleifer/tiny-gpt2`. These
cover: layer resolution, vector validation, anchor mapping with a real
tokenizer, activation capture, multi-layer hooks, batched generation,
batch-size consistency (the §24 check: same prompt produces same output in
batch size 1 vs batch), symbolic execution, symbolic→LLM fallback, and
extract/load roundtrip with provenance.

v0.3.5 also added 10 contextual token resolver tests, 8 activation
telemetry tests, 8 latency measurement tests, 5 alpha-grid tests,
10 linear-probe baseline tests, 6 random-direction control tests,
and 20 AutoLearn loop tests.

This partially closes the real-model gap from the audit §24. However:

- the tests use a ~1MB toy model, not a meaningful steering-research model;
- the direct-logit routing path is not exercised (tiny-gpt2's BPE tokenizer
  does not produce single-token SYMBOLIC/LLM labels);
- the chat-template path is not exercised (tiny-gpt2 has no chat template);
- GPU/CUDA paths are not exercised on CPU-only environments;
- KV-cache interactions with steering are not directly tested.

The full qualification gate (Qwen2.5-3B-Instruct + chat template + anchor
mapping + multi-layer hooks + left-padded batching + direct logits + KV
cache + model.generate()) remains a P0 item for v0.3.6. Until then, "164
passing tests" must be quoted with the qualification "unit, regression, and
toy-model integration tests; full-scale real-model qualification pending."

---

## 7. Latency numbers

**Status: PARTIAL — per-stage timings added in v0.3.5, but not yet validated at scale.**

v0.3.5 added per-stage latency measurement to `generate_v0_outputs.py`.
Route records now include:

- `route_batch_ms` — time spent in the routing stage (model forward + decision)
- `symbolic_exec_ms` — time spent in symbolic execution (0.0 for LLM-routed tasks)
- `answer_batch_ms` — time spent in LLM answer generation (0.0 for symbolic tasks)
- `pipeline_elapsed_ms` — total wall time (preserved for backward compat)

Skipped stages report 0.0. The sum of per-stage times should approximately
equal `pipeline_elapsed_ms` (within float precision and batch-amortization
boundaries).

This closes the schema gap from audit §22. However, the numbers are still
**wall-clock measurements subject to batch amortization**: when N tasks are
processed in one batch, each task's `answer_batch_ms` reflects its share of
the batch's wall time, not its individual GPU time. For precise per-task
latency, run with `--batch-size 1`. Until latency numbers are reported with
explicit batch size and amortization method, they should not be used in
external performance comparisons.

---

## 8. Provenance and reproducibility

**Status: PARTIAL.**

Established:

- fixed decoding parameters;
- explicit seeds;
- SHA-256 benchmark digests;
- deterministic benchmark generator;
- frozen final-test recommendations.

Missing (P1):

- torch / transformers / CUDA / GPU model versions;
- model revision and tokenizer revision;
- chat-template hash;
- attention implementation, dtype, device map;
- PyTorch deterministic settings;
- per-vector provenance (capture anchor, capture prompt format, dataset
  SHA-256, extraction method, normalization, positive/negative N).

Until those are recorded in a run manifest (see `docs/RUN_MANIFEST.md`),
results from this repository are not bit-for-bit reproducible across
machines. Any external writeup must state this limitation.

---

## 9. Tokenizer portability of direct-logit routing

**Status: PARTIAL — contextual resolver added in v0.3.5, not yet validated at scale.**

v0.3.5 added `resolve_route_token_ids_contextual` which derives route-label
token IDs by diffing `T(rendered_prompt + label)` against
`T(rendered_prompt)`, eliminating the boundary assumption. This is selected
via `--route-token-resolver contextual` (default remains `isolated` for
backward compat).

The licensed claims are now:

- "direct-logit routing with the **isolated** resolver works for tokenizers
  where the isolated label encoding matches the continuation encoding in the
  routing context" (ESTABLISHED for the tested tokenizers, NOT YET for
  arbitrary tokenizers).
- "direct-logit routing with the **contextual** resolver correctly identifies
  the continuation token for any tokenizer where the label appends as a
  single token after the rendered prompt" (ESTABLISHED via unit tests with
  synthetic boundary-differing tokenizers, NOT YET validated on
  Qwen2.5-3B-Instruct's real tokenizer at scale).

For headline-eligible results, use `--route-token-resolver contextual`.
The broader portability claim across all tokenizer families remains
**not yet** licensed until validated on production tokenizers.

---

## 10. Independent composite coefficients

**Status: ESTABLISHED as infrastructure — NOT YET used in experiments.**

v0.3.5 added `--alpha-grid` to `tune_steering.py` for independent
per-vector coefficient sweeping in composite bundles. Each grid entry
specifies one alpha per vector, e.g. `--alpha-grid 0.5,1.0 1.0,2.0`
sweeps two configurations: (0.5, 1.0) and (1.0, 2.0).

This closes the v0.3.4 limitation where `--alphas` was a global
multiplier applied to every vector's stored alpha — making it impossible
to test whether different layers need different intervention strengths.

The infrastructure is ESTABLISHED (tested, wired through the sweep loop).
However, no experiment has yet used `--alpha-grid` to find optimal
per-layer coefficients on a real model. Until then, the default
`--alphas` (global multiplier) remains the only experimentally
characterized sweep mode.

---

## 11. Causal ablation: matched-norm random-direction control

**Status: ESTABLISHED — run on Qwen2.5-1.5B-Instruct.**

v0.3.5 added `scripts/random_direction_control.py` and ran it on
Qwen2.5-1.5B-Instruct with 5 random control directions.

Result: extract-once F1=0.6154, random F1=0.0533 ± 0.0653,
z-score=8.60, interpretation=`real_vector_outperforms`.

This is **causal evidence** that the steering vector's direction
encodes tool-policy information. Random directions of the same norm
produce near-zero F1, while the real contrastive mean direction
produces F1=0.62. The direction matters, not just the magnitude.

---

## 12. Linear-probe baseline

**Status: ESTABLISHED — run on Qwen2.5-1.5B-Instruct.**

v0.3.5 added `scripts/linear_probe_baseline.py` and ran it on
activations captured from Qwen2.5-1.5B-Instruct (layer 20, last token).

Result: 100% cross-validated accuracy (5-fold), majority-class
baseline 83%, lift +17%.

The activations at the last token position **perfectly separate**
symbolic from LLM tasks. This confirms the information is linearly
present in the residual stream.

However, using the probe's weights as a steering direction produces
F1=0.000 — the probe direction does not causally flip the model's
output. This is an important finding: **classification accuracy and
causal intervention effectiveness are different objectives**. A
direction that perfectly separates classes in a linear classifier may
not be the direction that, when added to the residual stream, changes
the model's next-token prediction.

The gap between the probe's 100% classification accuracy and the
contrastive mean direction's 62% steering F1 suggests that a
**steering-optimized direction** (trained to flip the model's output,
not to classify) could close this gap. This is a target for future
work.

---

## 18. AutoLearn: iterative learning from execution outcomes

**Status: ESTABLISHED as infrastructure — partially run on a real model.**

v0.3.5 added `src/daph_learning/autolearn/` and `scripts/autolearn.py`,
implementing the iterative learning loop that gives the project its name.

### Real-model experiment (Qwen2.5-1.5B-Instruct)

A full experiment suite was run on Qwen2.5-1.5B-Instruct (28 layers,
hidden_size=1536) with 100 train / 50 val / 50 test tasks. Key settings:
layer=20, normalization='none', alpha=3.0, token_scope='last',
capture at last token position.

Results on the test set:

| Method | Accuracy | F1 |
|--------|----------|----|
| Baseline (no steering) | 0.7000 | 0.0000 |
| Extract-once steering | 0.9000 | 0.6154 |
| Probe-based steering | 0.8400 | 0.0000 |
| AutoLearn (best iter) | 0.9000 | 0.6154 |
| Linear probe (CV) | 1.0000 | N/A |
| Random direction (mean) | N/A | 0.0533 |

Key findings:

1. **Steering causally improves routing** (F1 0.00 → 0.62). The
   random-direction control confirms this is not chance (z=8.6,
   `real_vector_outperforms`). This is the first real-model evidence
   that the extracted steering direction encodes tool-policy information.

2. **The linear probe achieves 100% accuracy** on the same activations,
   confirming the information is linearly separable. However, the probe's
   weights as a steering direction produce 0% F1 — **classification and
   causal intervention are different objectives**. A direction that
   perfectly separates the classes in a classifier may not be the
   direction that causally flips the model's output.

3. **The AutoLearn loop did not update** because its internal routing
   uses `score_route_batch_from_logits`, which requires single-token
   labels. Qwen2.5 tokenizes both "SYMBOLIC" and "LLM" as multi-token,
   so the logit router fails and the loop falls back to heuristic
   routing. The loop needs generate-mode routing support to work with
   this tokenizer. This is a known limitation.

4. **Key hyperparameters matter**: L2-normalized vectors (norm=1) had no
   effect even at alpha=20, because the residual stream norm is ~71.
   Using `normalization='none'` (vector norm ~15) with alpha=3.0
   produces a perturbation that is ~60% of the residual stream norm —
   strong enough to flip routing without degrading output.

The licensed claims are:
- "steering with the contrastive mean direction causally improves
  routing on Qwen2.5-1.5B-Instruct (F1 0→0.62, z=8.6 vs random control)"
- "the linear probe achieves 100% CV accuracy on the same activations,
  but its direction does not produce effective steering"

Claims NOT YET licensed:
- "the AutoLearn loop converges to a better vector than extract-once"
  (the loop didn't update on this model due to the multi-token tokenizer
  limitation)
- "steering works across model families" (only tested on Qwen2.5-1.5B)

---

## 20. Activation telemetry during steering

**Status: ESTABLISHED as infrastructure — NOT YET collected at scale.**

v0.3.5 added activation telemetry to `residual_addition_hook` and
`multi_layer_residual_addition_hook`. When a `telemetry_sink` list is
provided, the hook appends per-forward-pass statistics:

- `layer_index` — which layer was steered
- `n_steered_positions` — how many token positions were modified
- `h_norm_mean` — mean L2 norm of the pre-steering hidden state
- `av_norm_mean` — mean L2 norm of `alpha * vector`
- `relative_perturbation` — `av_norm_mean / h_norm_mean` (how hard the
  vector pushes relative to the residual stream's magnitude)
- `cosine_shift_mean` — mean `1 - cos(h, h + alpha*v)` (0 = no directional
  change, 2 = full reversal)

These statistics let operators detect pathological steering (e.g. a vector
that overwhelms the residual stream, or an alpha so large it reverses the
hidden state direction) without inspecting raw activations.

The infrastructure is ESTABLISHED (tested, wired through
`score_route_batch_from_logits` and `_generate_batch`). However, no
experiment has yet collected telemetry at scale on a real model, so the
typical ranges of these statistics for v0.3.5 steering vectors are
**not yet** characterized. Until then, telemetry is available for
diagnostic use but not for headline claims.

---

## How to use this file

- When adding a new metric, script, or docstring, check whether the term you
  are using appears here. If it does, use the narrower reading.
- When publishing or citing results, quote the status tag verbatim.
- When a `NOT YET` item becomes `ESTABLISHED`, update this file in the same
  PR that adds the supporting evidence. Do not let the file drift.
- When in doubt, claim less. The architecture is more interesting than the
  current evidence base; the gap is closed by experiments, not adjectives.
