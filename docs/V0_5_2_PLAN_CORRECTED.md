# DAPH Latent Memory v0.5.2 — Causal Specificity & Skill-State Architecture (Corrected)

> **Status of this document**: Corrected revision of the v0.5.2 build plan.
> Integrates review feedback into the objective, gates, losses, curriculum,
> sequencing, and scope. The original draft is superseded by this file.
>
> **What changed**: see `## Corrections applied` at the end of this document.

---

## 1. Primary objective and scope

Move from generic helpful latent conditioning to query-bound latent memory plus
reusable skill primitives. The release is successful only if matched latent
materially outperforms mismatched latents with statistically credible margins.

Current result (v0.5.1):

```
A_process   = 12.5%
A_shuffled  = 10.0%
A_corrupted = 13.5%
```

That is not good enough for a latent-memory claim.

### 1.1 Scope caveat — arithmetic is an infrastructure domain, not a claim domain

**Arithmetic cannot bear the weight of a memory claim.** The only reusable
skill is the operator, which is a single token extractable from the input.
Instance state encoding "4829 and 392" is a copy of two tokens already in the
context, not memory in any interesting sense. Once symbolic execution is the
answer authority for arithmetic (section 17), accuracy-based gates measure a
thing the latent is not doing.

Therefore v0.5.2 is scoped as **infrastructure validation on arithmetic**:

- Build and validate the causal evaluation harness, contrastive/functional
  losses, skill/instance split, router, and persistent memory on arithmetic,
  where ground truth is cheap and verifiable.
- **Do not** make a "latent memory" claim from arithmetic results alone.
- The memory *claim* is deferred to a follow-up release on a domain where
  state genuinely carries across subproblems: multi-hop reasoning, planning
  with branching, or equation rearrangement. Candidate domains are listed in
  `## Follow-up domain` below.

The release gate (section 25) is therefore an *infrastructure* gate: the
harness must produce the required contrasts, the losses must train without
degeneration, and the matched-vs-mismatched separation must be statistically
detectable on arithmetic. A passing arithmetic result licenses the harder
domain, not the memory claim.

### 1.2 Core gate ordering

```
A_matched  >  A_same-class mismatch  >  A_wrong-class mismatch  ≈  A_random
```

with statistically credible margins (section 24).

---

## 2. Architecture target

Two-channel latent state:

```
z = [z_skill, z_instance]
```

### 2.1 Skill state

`z_skill` represents reusable computational procedures (addition, subtraction,
multiplication, division, modular arithmetic, decomposition, verification,
backtracking, symbolic translation, answer checking). Generalizes across
instances.

Conceptually aligned with RISER's library of reusable reasoning vectors and
learned routing. **Caveat**: on arithmetic, `z_skill` will collapse to a
near-one-hot of the operator. This is expected and is acceptable *for
infrastructure validation*. The skill bank only becomes scientifically
interesting on the follow-up domain (section 1.1).

### 2.2 Instance state

`z_instance` encodes problem-specific information: operands, intermediate
values, partial derivation, current operation, remaining subproblem, local
constraints, execution history. This is the part that must fail when shuffled.

```
R(q_i, z_instance,i)  >>  R(q_i, z_instance,j)   for i ≠ j
```

**Caveat**: on arithmetic, instance state is largely a copy of operands. The
shuffle-failure property is therefore a *necessary but not sufficient*
condition for memory. Passing it on arithmetic licenses the harder domain.

---

## 3. Phase 0 — Produce v0.5.1 latents (prerequisite)

**The original plan's Phase 1 ("diagnose current representation") cannot run
without trained latents. No artifacts currently exist.** Phase 0 is the
actual first step.

1. Train v0.5.1 to completion on the existing dataset with the existing loss.
2. Persist per-example latents for every evaluation example under every
   condition the causal eval will need (matched, shuffled, corrupted at each
   α, etc.), keyed by `(example_id, condition, seed)`.
3. Persist model checkpoints and the exact decoding config used to produce
   them.
4. Emit a manifest with SHA-256 digests of all latent artifacts.

Exit criterion for Phase 0: a `latents_v0_5_1/` directory whose contents are
sufficient to run Phase 1 with no further training.

---

## 4. Phase 1 — Causal ablation framework

Do this before changing the model.

Create `scripts/evaluate_latent_causality.py`. For every evaluation example
`i`, generate the following conditions. All conditions must be reproducible
from a `(example_id, condition, seed)` triple.

**Baselines**

- `base` — no latent injection
- `matched` — `z_i`
- `zero` — zero vector
- `constant` — fixed learned constant
- `random_gaussian`
- `random_norm_matched` — gaussian rescaled to match `||z_i||` distribution

**Query-binding controls**

- `shuffled_global` — derangement across all examples
- `shuffled_same_class` — derangement within same operator
- `shuffled_wrong_class` — derangement across different operator
- `nearest_neighbor_wrong` — closest latent by cosine among wrong-class
- `farthest_neighbor` — farthest latent by cosine

**Geometry controls**

- `normalized` — `z_i / ||z_i||`
- `scaled` — `z_i * c` for fixed `c`
- `sign_flipped` — `-z_i`
- `mean_latent` — dataset mean
- `centroid_latent` — per-class centroid
- `pca_low_rank` — project onto top-k PCs
- `pca_residual` — project onto residual subspace

**Corruption sweep**

```
z'_i = z_i + α · ε,   ε ~ N(0, I) normalized to match ||z_i|| scale
```

```
α ∈ {0.00, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00}
```

Each `(α, seed)` pair is a separate reproducible condition. The current
single-`corrupted` control is removed; it hides geometry.

---

## 5. Phase 1 — Latent probes

Create `analysis/latent_probe.py`. Diagnostic only; never in the production
pathway.

**Problem properties**: `operation_type`, `operand_1`, `operand_2`, `answer`,
`difficulty`, `expression_length`, `number_of_steps`.

**Process properties**: `current_step`, `next_operation`, `intermediate_result`,
`remaining_steps`, `success/failure`.

**Measure**: linear-probe accuracy, nonlinear-probe accuracy, mutual
information estimates, nearest-neighbor label agreement, cosine clustering,
centered kernel alignment.

Purpose: distinguish skill representation from instance representation from
generic activation manifold.

---

## 6. Phase 2 — Loss redesign

### 6.1 What is being removed

The v0.5.1 loss is:

```python
L = answer_weight * answer_loss + alignment_weight * (1 - cos(student_h, teacher_h))
```

This is hidden-state distillation, not memory. It rewards the student for
matching the teacher's activation *regardless of which latent was injected*,
which fully explains why `A_corrupted > A_matched > A_shuffled`. **The
alignment term is removed.** It is not retained as `L_process`. Keeping it
would preserve the disease the new losses are meant to cure.

### 6.2 New objective

```
L = L_task + λ_f · L_functional + λ_d · L_disentangle
```

- `L_task` — answer NLL on matched latent
- `L_functional` — defined in section 7
- `L_disentangle` — light regularizer encouraging `z_skill` and `z_instance`
  to occupy disjoint subspaces (e.g., orthonormality penalty on the
  composition matrices `W_s`, `W_i`)

**`L_contrast` is dropped.** The original plan added both a representation-
similarity contrastive loss and a functional ranking loss while also stating
"representation similarity alone is insufficient." The two are in tension:
`L_contrast` pushes `z_i` to be geometrically close to `q_i`; `L_functional`
pushes `z_i` to be *useful* for `q_i`. A latent that is geometrically distant
but functionally perfect would be penalized by `L_contrast`. Keep only the
term that optimizes the property we actually want.

### 6.3 `L_process` is not in the objective

The original draft listed `λ_p · L_process` without defining it. It is
removed. If a process-alignment signal is needed later, it should be a
*consequence* of `L_functional` (the decoder behaving correctly), not a
separate target that recreates the v0.5.1 failure mode.

---

## 7. Functional mismatch loss (the most important change)

### 7.1 Definition

Let `R(q, z)` be the negative task loss (token NLL) of the decoder on query
`q` with latent `z` injected:

```
R(q, z) = − E_{y ~ target} [ log p(y | q, z) ]
```

Lower `R` is better. `R` is differentiable in `z` when `z` is injected
continuously into the residual stream.

For matched pair `(q_i, z_i)` and hard negative `z_j`:

```
r_i^+   = R(q_i, z_i)
r_ij^-  = R(q_i, z_j)

L_functional = mean_ij [ max(0, m − r_i^+ + r_ij^-) ]
```

The margin `m` is in units of nats. Calibrate per dataset by computing the
matched-vs-shuffled `R` gap on a held-out probe set at training start and
setting `m` to ~50% of that gap. Re-calibrate if the gap shifts by >2× during
training.

### 7.2 Hard negatives

Per matched example, sample one negative from each category with probability
uniform across categories:

- same operation, different numbers
- similar expression structure (same depth, different operators)
- same answer, different derivation
- same operands, different operator
- numerically close problems (operands within 10%)

Random gaussian negatives are *not* used in `L_functional`. They are too easy
and teach nothing.

### 7.3 Compute budget

Each `L_functional` step requires `1 + K` forward passes through the decoder,
where `K` is the number of hard negatives per example. With `K = 5` categories
and one negative per category, that is 6× the matched-only forward cost.

**Budget decision for v0.5.2**: `K = 3` hard negatives per step (sampled
uniformly from the 5 categories), giving 4× forward cost. On Qwen3-1.7B with
the existing ~4000-example dataset and a single GPU, this targets a
single-seed training run of approximately 4-8 hours. The 5-10 seed
requirement (section 24) is met by running seeds in parallel across GPUs or
sequentially across nights; if only one GPU is available, reduce to 5 seeds
and widen CIs accordingly. State the budget actually used in the release
report.

### 7.4 What `R` is not

`R` is not accuracy. Accuracy is non-differentiable and would force a
REINFORCE-style estimator with high variance. Token NLL is differentiable,
correlates with accuracy in this regime, and gives the margin `m` a
well-defined unit.

---

## 8. Phase 3 — Skill / instance split

### 8.1 Components

- `SkillBank` — holds `K` skill vectors `v_1 ... v_K`
- `InstanceEncoder` — produces `z_instance` from `q`
- `LatentComposer` — combines them

### 8.2 Composition

Start with linear composition:

```
z = W_s · z_skill + W_i · z_instance
```

Move to gated composition once linear is stable:

```
z = g_s(q) ⊙ z_skill + g_i(q) ⊙ z_instance
```

### 8.3 Staged curriculum

**Phase A — Skill primitives.** Train `SkillBank` with skill labels (operator
on arithmetic). Each skill vector is trained across many instances.

**Phase B — Instance state.** Freeze `SkillBank`. Train `InstanceEncoder` on
exact problem state. Desired: `z_instance,a ≠ z_instance,b` even when both
are multiplication.

**Phase C — Joint composition.** Unfreeze with an anneal schedule:

```
freeze_ratio(t) = max(0, 1 − t / T_unfreeze)
```

`T_unfreeze` = 30% of total Phase C steps. Linear anneal. Log the freeze
ratio per step. Joint training without an anneal schedule is exactly the
regime where entanglement was observed in v0.5.1; do not skip it.

### 8.4 Invariance loss (added; was missing)

The original plan asserted `z_multiply(q_a) ≈ z_multiply(q_b)` without a loss
to enforce it. Add an explicit invariance term during Phase A:

```
L_invariance = mean_{a,b: skill(a)=skill(b)} [ 1 − cos(z_skill,a, z_skill,b) ]
```

Without this, the skill bank absorbs operand-range information and the
"reusable" property is asserted but not enforced. `L_invariance` is added to
the Phase A objective only; it is removed in Phases B and C.

---

## 9. Phase 4 — Router

Create `daph_latent_memory/router.py`.

```
input:  h_q
output: α_1 ... α_K  over skill primitives
z_skill = Σ_k α_k · v_k
```

Start with soft routing. Move toward sparse `TopK(α)` with `K ∈ {1, 2}` only
after soft routing is stable.

**Training signal**: supervised on operation labels in v0.5.2 (arithmetic
gives us labels for free). Remove explicit labels and learn from task reward
in the follow-up domain, *not* in v0.5.2. The original plan implied RL
training of the router within v0.5.2; that is deferred.

---

## 10. Phase 5 — Persistent memory

### 10.1 Memory bank

```python
@dataclass
class MemoryEntry:
    key: Tensor
    latent: Tensor
    capability_id: str
    source_task: str
    timestamp: float
    confidence: float
    verifier_score: float
    lineage: dict
```

Query with the model's current hidden state, not only text embeddings:

```
k_q = f(h_q)
M_q = TopK(sim(k_q, k_i))
```

### 10.2 Learned selector — supervision specified

The original plan said "let a learned selector decide" without specifying
its training signal. **Specify**: the selector is a small MLP that takes
`(k_q, k_candidate)` and outputs a binary inject/skip decision. It is
trained in two stages:

1. **Heuristic warm-start** (first 50% of Phase 5 training): inject iff
   `sim(k_q, k_candidate) > τ` for a fixed `τ` calibrated on the probe set.
   The selector distills this heuristic.
2. **Reward fine-tuning** (second 50%): REINFORCE with reward
   `ΔR = R(q, z=skip) − R(q, z=inject)`, i.e., the NLL reduction from
   injecting. Baseline the return with a moving average of `ΔR` per
   capability_id.

The distinction `retrieval relevance ≠ downstream usefulness` (ElasticMem)
is handled by stage 2: the selector learns to skip retrieved memories that
don't reduce `R`.

### 10.3 Memory operations

```python
class MemoryAction(Enum):
    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    NOOP = "noop"
```

Per Memory-R1, learned operations beat heuristic append-only. `MERGE` is
added because learned skill latents produce near-duplicates.

### 10.4 Conflict detection before commit

Every new memory passes:

- novelty check
- contradiction check
- task verifier
- confidence threshold
- causal usefulness test

States: `verified`, `unverified`, `deprecated`, `contradicted`,
`superseded`. Memory is never a flat immutable cache (MOSAIC).

### 10.5 Verifier

```
Model generates candidate latent
        │
        ▼
   Latent verifier
        │
 ┌──────┴──────┐
 pass         fail
 │             │
 ▼             ▼
commit      repair / discard
```

For arithmetic, use deterministic verification. **No LLM judge for
arithmetic.** SATQuest and DeepVerifier motivate the architecture; the
arithmetic implementation uses exact symbolic checking.

### 10.6 Hidden-state success verification (CLUE)

```
Δh = h_end − h_start
c_success, c_failure  — centroids from training trajectories
d_s = ||Δh − c_success||
d_f = ||Δh − c_failure||
C = d_f − d_s
```

**OOD caveat (added)**: `c_success` and `c_failure` are computed from
training trajectories and encode training-distribution bias. On the OOD
splits (section 21) the centroids may not transfer. Therefore `C` is used
as a confidence signal **on IID only**. On OOD splits, `C` is logged for
analysis but not used as a gate or commit signal. A separate OOD-calibrated
centroid set is a follow-up task, not part of v0.5.2.

---

## 11. Latent subspace analysis

Run PCA/SVD on successful latent states per layer:

```
Z_l = [z_1, ..., z_N] = U Σ V^T
```

Test performance with `k ∈ {1, 2, 4, 8, 16, 32, 64, 128}` retained
components. This reveals the effective dimensionality of the useful
intervention. If the useful subspace is tiny, that is an important finding
and constrains all subsequent architecture work.

---

## 12. Symbolic / shared reasoning subspaces (CCA)

Generate paired representations:

- Natural: `"Multiply 4829 by 392..."`
- Symbolic: `4829 * 392`

Collect activations `X_NL`, `X_SYM`. Use CCA to identify `U_shared`. Test
steering along the shared subspace.

**Trivial-alignment caveat (added)**: on arithmetic, NL and symbolic forms
share the same numbers and operator, so CCA will find a shared subspace
dominated by operand embeddings — the *uninteresting* part. Interpret CCA
results on arithmetic accordingly. The CCA technique becomes informative on
the follow-up domain where NL and symbolic forms diverge in surface
structure. Run it on arithmetic for infrastructure validation; do not
over-interpret the shared subspace it finds.

---

## 13. Symbolic router integration

```
Input
  │
  ▼
Capability Router
  │
  ├── exact symbolic task ───────► Symbolic Engine
  │
  └── reasoning task
          │
          ▼
     Skill Router
          │
          ▼
     Latent Memory
          │
          ▼
        LLM
```

For exact arithmetic, symbolic execution is the answer authority. The latent
system handles decomposition, planning, strategy selection, explanation,
error recognition, tool selection — not deterministic computation.

**Consequence for gates**: because symbolic owns arithmetic correctness, the
accuracy-based gates (CSS, WMP, LUG) measure the *latent's effect on
non-symbolic subtasks*, not on arithmetic correctness per se. The release
report must state which subtasks each gate is computed on. A gate computed
on pure arithmetic accuracy is uninformative once symbolic is wired in.

---

## 14. Elastic memory budget

For each problem predict `B_i ∈ {0, 1, 2, 4, 8, 16}` latent tokens.

```
R_total = R_accuracy − β · B
```

**Gradient estimator (added; was unspecified)**: the discrete budget is
non-differentiable. Use Gumbel-softmax with a straight-through estimator
during training, annealed to hard one-hot over the first 50% of training.
`β` is in units of accuracy per token; calibrate per dataset by setting
`β` such that, at the matched latent's accuracy, the marginal accuracy gain
of one additional token equals `β`. Re-calibrate if the dataset changes.

---

## 15. Dataset redesign

The current arithmetic dataset is too easy to exploit via shortcuts.

**IID** — same structure and ranges as training.

**Operand OOD** — train `1-999`, test `1000-99999`.

**Structural OOD** — train `a op b`, test `(a op b) op c`.

**Operator OOD** — train `+ - *`, test `% // **`.

**Distractor OOD** — add irrelevant numbers and NL clutter.

**Counterfactual pairs** — same surface structure, different answer.

**Adversarial matched pairs** — same operands, different operators:

```
12 + 5
12 * 5
12 - 5
```

This forces the latent to encode actual operation semantics, not operand
identity.

---

## 16. Benchmark matrix

| Dimension | Required metric |
|---|---|
| Accuracy | exact match |
| Latent specificity | matched − shuffled |
| Skill transfer | same-skill OOD |
| Instance binding | correct memory vs wrong memory |
| Robustness | corruption curve |
| Memory necessity | memory-on vs memory-off |
| Memory efficiency | tokens / latent slots |
| Latency | wall-clock |
| Calibration | confidence vs correctness |
| Stability | multi-seed CI |
| Retention | old-task performance |
| Interference | cross-skill degradation |

---

## 17. Headline metrics

**Causal Specificity Score**

```
CSS = A_matched − A_shuffled
```

**Wrong-Memory Penalty**

```
WMP = A_matched − A_wrong-class
```

**Latent Utility Gain**

```
LUG = (A_latent − A_base) / (latent_token_cost + 1)
```

**Generalization Gain**

```
GG = A_OOD,latent − A_OOD,base
```

---

## 18. Statistical requirements

### 18.1 Paired tests

Each condition sees the same example, so use paired tests.

**Binary accuracy comparison (two conditions)**: McNemar

```
χ² = (|b − c| − 1)² / (b + c)
```

Also report exact McNemar, paired bootstrap, effect size, and CI on the
difference.

**Corruption sweep (continuous α, multiple levels)**: McNemar does not
apply. Use Cochran's Q across the discrete α levels, supplemented by a
paired Wilcoxon signed-rank test on per-example accuracy curves. Report
per-α-pair McNemar only for the endpoint comparisons.

### 18.2 Seeds — what is seeded

**Specify what is seeded**: the latent encoder init, the data shuffle order,
the hard-negative sampling, and the corruption RNG. The base model is frozen
in v0.5.2 and is *not* a seed axis. State this in every release report;
otherwise variance interpretation is ambiguous.

**Minimum**: 5 seeds. **Preferred for headline claims**: 10 seeds. If GPU
budget constrains to 5 seeds, widen CIs and report the constraint.

### 18.3 CSS threshold — statistically grounded

The original plan set `CSS ≥ 10pp` as a round number. **Replace with a
null-derived threshold.** Procedure:

1. For each seed, compute `CSS` under label permutation: shuffle the
   matched/mismatched assignment and recompute `CSS`. This is the null
   distribution of CSS when the latent carries no query-specific
   information.
2. Pool across seeds. Take the 99th percentile of the null distribution,
   `CSS_null_99`.
3. The gate threshold is `max(CSS_null_99, 5pp)`. The 5pp floor guards
   against a degenerate null in very small evaluation sets.

Report `CSS_null_99` in the release report. The threshold is no longer a
round number; it is a property of the experiment.

---

## 19. Release gates

v0.5.2 does not pass because its loss went down. It passes only if all of
the following hold.

**Gate 1 — matched specificity (conjunction)**

All three:

```
A_matched > A_shuffled          with significant paired improvement (McNemar p < 0.01)
A_matched > A_base              (matched beats no-latent; otherwise the latent is not useful)
LUG > 0                         (utility gain is positive after token cost)
```

The original plan's Gate 1 was CSS alone. CSS is gameable: a latent can hit
high CSS by making shuffled *actively harmful* without being useful when
matched. The conjunction prevents that failure mode.

**Gate 2 — corruption behavior (no monotonicity requirement)**

The original plan required strict monotonicity `A(0) > A(0.5) > A(1) > A(2)`.
Small-α noise often *helps* (regularization), and the current data already
shows `A_corrupted > A_matched`. Strict monotonicity would fail a working
system. Replace with:

```
A(α=0)   > A(α=2)            (endpoints: large corruption hurts)
|A(α=0.5) − A(α=0)| < ε      (robustness band: small perturbation is benign)
A(α=2)   < A(α=0) − δ        (large corruption materially degrades)
```

with `ε = 1pp` and `δ = 3pp` as starting values, recalibrated to the null
distribution if needed. No monotonicity requirement in the mid-range.

**Gate 3 — semantic mismatch**: wrong-operation latents degrade strongly
(`A_wrong-class < A_matched − 5pp`, McNemar p < 0.01).

**Gate 4 — OOD benefit**: latent intervention improves at least one
meaningful OOD split (`GG > 0` on at least one of Operand/Structural/
Operator/Distractor OOD, paired test p < 0.05).

**Gate 5 — no catastrophic degradation**: general tasks do not collapse
(`A_base` on a held-out general-purpose set degrades by < 2pp).

**Gate 6 — reproducibility**: at least 5 seeds reproduce the main effect
with the same sign and significant paired test on each seed, OR a
random-effects meta-analysis across seeds gives p < 0.01.

---

## 20. Implementation layout (v0.5.2 scoped)

Only modules built in v0.5.2 are listed. Aspirational modules for later
releases are listed separately.

```
experiments/daph_latent_memory_v0_5_2/
│
├── configs/
│   ├── qwen3_1_7b.yaml
│   ├── causal_eval.yaml
│   └── ablations.yaml
│
├── daph_latent/
│   ├── encoder.py            # Phase 3
│   ├── instance_state.py     # Phase 3
│   ├── skill_bank.py         # Phase 3
│   ├── router.py             # Phase 4
│   ├── injector.py           # Phase 1 (needed for eval)
│   ├── memory_bank.py        # Phase 5
│   ├── memory_manager.py     # Phase 5
│   ├── verifier.py           # Phase 5
│   ├── contrastive.py        # NOT BUILT — L_contrast dropped (section 6.2)
│   └── metrics.py
│
├── scripts/
│   ├── generate_dataset.py
│   ├── train_skill_bank.py        # Phase A
│   ├── train_instance_encoder.py  # Phase B
│   ├── train_joint.py             # Phase C
│   ├── evaluate.py
│   ├── evaluate_latent_causality.py
│   ├── evaluate_ood.py
│   └── analyze_latent_geometry.py
│
└── tests/
```

**Aspirational (not in v0.5.2)**: `recurrent_state.py`, `consolidation.py`,
`autolearn_bridge.py`. These belong to v0.6+ per the build order. The
original draft listed them in the v0.5.2 tree; they are removed from the
release scope.

---

## 21. Unit tests (v0.5.2 scoped)

Tests are scoped to modules actually built in v0.5.2. The original plan
listed tests for memory ADD/UPDATE/MERGE/DELETE and conflict rejection,
which belong to Phase 5 and are included; tests for recurrent state and
consolidation are removed (those modules are not built).

- matched latent lookup
- shuffle permutation correctness (derangement invariant)
- same-class negative sampling
- wrong-class negative sampling
- norm-matched random controls
- corruption reproducibility (per-α, per-seed)
- router probability normalization
- top-k routing
- memory ADD / UPDATE / MERGE / DELETE
- conflict rejection
- latent serialization
- checkpoint reproducibility
- paired evaluation alignment (same example set across conditions)
- `L_functional` margin sign (r_i^+ < r_ij^- on a constructed toy batch)
- invariance loss sign (cos same-skill > cos cross-skill on a toy batch)
- Gumbel-softmax straight-through gradient flow
- selector REINFORCE reward baseline subtracts moving average

A bad shuffled-control implementation can completely invalidate the
experiment. Treat these tests as scientific infrastructure, not cleanup.

---

## 22. Phase roadmap

**Phase 0 — Produce v0.5.1 latents** (prerequisite; added)

Train v0.5.1 to completion, persist latents and checkpoints, emit manifest.

**Phase 1 — Diagnose current representation**

Do not alter architecture. Run corruption sweep, norm sweep, shuffle
variants, PCA ablation, probe suite, multiple seeds.

**Phase 2 — Force causal specificity**

Implement hard negatives, `L_functional`, `L_disentangle`. Drop `L_contrast`
and the v0.5.1 alignment loss. Target: Gate 1 conjunction passes.

**Phase 3 — Split skill and instance state**

Implement `SkillBank`, `InstanceEncoder`, `LatentComposer` with the
invariance loss and the freeze-anneal schedule.

**Phase 4 — Dynamic routing**

Add lightweight router. Supervised on operation labels in v0.5.2. RL
training of the router is deferred to the follow-up domain.

**Phase 5 — Persistent latent memory**

Add retrieve, write, update, merge, delete, verify, with the two-stage
selector (heuristic warm-start → REINFORCE).

**Deferred to v0.6+**: recurrent working memory, memory consolidation,
AutoLearn integration. The original plan placed these in the v0.5.2
roadmap; they are removed. AutoLearn integration before Phase 5 is stable
would make debugging impossible.

---

## 23. Long-term architecture (aspirational, not v0.5.2)

```
                    ┌─────────────────────┐
                    │   Persistent Memory │
                    │ Episodic / Semantic │
                    │ Skill / Procedural  │
                    └──────────┬──────────┘
                               │
                               ▼
Input ──► Capability Encoder ──► Memory Retriever
                 │                    │
                 ▼                    ▼
            Skill Router       Instance Memory
                 │                    │
                 └────────┬───────────┘
                          ▼
                    Latent Composer
                          │
                          ▼
                   Recurrent State
                          │
                          ▼
                Transformer / GDN2
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Symbolic Executor            Generator
             │                         │
             └────────────┬────────────┘
                          ▼
                       Verifier
                          │
                    pass / repair
                          │
                          ▼
                   Memory Commit
```

This is the target. v0.5.2 delivers the skill router, instance memory,
latent composer, verifier, and memory commit. Recurrent state and
consolidation are v0.6+.

---

## 24. What not to do

- Do not move to a 7B or 14B model yet.
- Do not increase dataset size just because 4,000 examples seems small.
- Do not add more latent dimensions before determining whether the existing
  dimensions encode semantic information.
- Do not claim internal reasoning because visible output tokens decreased.
- Do not optimize benchmark accuracy until causal specificity is fixed.
- Do not remove corrupted or shuffled controls because they make the result
  look worse. Those controls are currently the most scientifically valuable
  part of the experiment.
- **Do not claim latent memory on arithmetic.** v0.5.2 on arithmetic is
  infrastructure validation. The memory claim requires a domain where state
  carries across subproblems (section 1.1).
- **Do not retain the v0.5.1 alignment loss.** It is the cause of the
  current non-specific behavior (section 6.1).
- **Do not add `L_contrast` alongside `L_functional`.** They optimize
  different and partially opposing properties (section 6.2).
- **Do not train the router with RL in v0.5.2.** Use supervised operation
  labels; defer RL to the follow-up domain (section 9).

---

## 25. Recommended immediate build order

```
Phase 0 (produce v0.5.1 latents)
  → Phase 1 (causal eval suite + latent geometry analysis)
  → Phase 2 (L_functional + hard negatives + drop alignment loss)
  → Phase 3 (skill / instance split with invariance loss + anneal)
  → Phase 4 (supervised router)
  → Phase 5 (persistent memory with two-stage selector)
```

The key milestone is not 20%, 30%, or 50% accuracy. It is reaching a result
where:

```
matched memory works and wrong memory reliably hurts
```

on a domain where that distinction is non-trivial.

---

## 26. Follow-up domain (post-v0.5.2)

Candidate domains where instance state is non-trivial and skill state is
more than an operator label:

- **Multi-hop reasoning** — state carries the partial derivation across
  hops; skill = hop type (lookup / compose / compare).
- **Planning with branching** — state carries the current branch and
  frontier; skill = planning primitive (expand / prune / backtrack).
- **Equation rearrangement** — state carries the current symbolic form;
  skill = algebraic move (isolate / substitute / factor).
- **Program synthesis** — state carries the partial program; skill =
  construct (loop / conditional / recursion).

Selection of the follow-up domain is a separate decision after v0.5.2
infrastructure passes its gates.

---

## Corrections applied

This document incorporates the following corrections to the original draft:

1. **Domain caveat (section 1.1)**: arithmetic is scoped as infrastructure
   validation, not a claim domain. The memory claim is deferred to a
   harder domain (section 26). The original draft claimed memory on
   arithmetic.

2. **Phase 0 added (section 3)**: the original Phase 1 diagnosis requires
   trained latents, which do not exist. Phase 0 trains v0.5.1 and
   persists artifacts first.

3. **Alignment loss removed (section 6.1)**: the v0.5.1 alignment term is
   the cause of non-specific behavior. The original draft kept it as
   `L_process` without definition.

4. **`L_contrast` dropped (section 6.2)**: it optimizes representation
   similarity, which the original draft itself called insufficient, and is
   in tension with `L_functional`. Only `L_functional` is retained.

5. **`L_functional` specified (section 7)**: `R` is defined as negative
   token NLL (differentiable, not accuracy). Margin `m` is calibrated
   from the matched-vs-shuffled gap. Compute budget is stated (K=3 hard
   negatives, 4× forward cost, ~4-8h per seed on 1.7B).

6. **Invariance loss added (section 8.4)**: the skill bank's reusability
   was asserted without a loss. `L_invariance` is added to Phase A.

7. **Freeze anneal schedule specified (section 8.3)**: joint training
   without an anneal schedule reproduces the v0.5.1 entanglement.

8. **Router supervision specified (section 9)**: supervised on operation
   labels in v0.5.2; RL deferred. The original draft implied RL within
   v0.5.2.

9. **Memory selector supervision specified (section 10.2)**: two-stage
   heuristic warm-start → REINFORCE with `ΔR` reward and per-capability
   baseline. The original draft said "learned selector" without a
   training signal.

10. **CLUE OOD caveat (section 10.6)**: `c_success`/`c_failure` centroids
    are training-distribution-biased. `C` is used as a confidence signal
    on IID only, not on OOD gates.

11. **CCA trivial-alignment caveat (section 12)**: on arithmetic, CCA
    finds a subspace dominated by operand embeddings. Run for
    infrastructure; do not over-interpret.

12. **Elastic memory gradient estimator specified (section 14)**:
    Gumbel-softmax with straight-through, annealed to hard one-hot. `β`
    calibration procedure defined.

13. **CSS gate replaced with conjunction (section 19, Gate 1)**: CSS
    alone is gameable. Gate 1 now requires
    `A_matched > A_shuffled` AND `A_matched > A_base` AND `LUG > 0`.

14. **CSS threshold grounded in null distribution (section 18.3)**:
    `CSS_null_99` from label permutation, with a 5pp floor. Replaces the
    unjustified `10pp` round number.

15. **Gate 2 monotonicity removed (section 19, Gate 2)**: strict
    monotonicity fails working systems because small-α noise often helps.
    Replaced with endpoints + robustness band + large-corruption-hurts.

16. **Corruption sweep statistical test specified (section 18.1)**:
    Cochran's Q across α levels + paired Wilcoxon on per-example curves.
    McNemar is for two-condition binary comparisons only.

17. **Seed scope specified (section 18.2)**: encoder init, data shuffle,
    hard-negative sampling, corruption RNG. Base model is frozen and is
    not a seed axis.

18. **Layout scoped to v0.5.2 (section 20)**: aspirational modules
    (`recurrent_state`, `consolidation`, `autolearn_bridge`) removed
    from the v0.5.2 tree. `contrastive.py` marked not-built.

19. **Tests scoped to v0.5.2 (section 21)**: tests for deferred modules
    removed. Tests for `L_functional` margin sign, invariance loss sign,
    Gumbel-softmax gradient flow, and REINFORCE baseline added.

20. **Recurrent state and consolidation removed from v0.5.2 (section 22,
    section 23)**: the original draft included them in the plan and then
    walked them back. They are v0.6+.

21. **Symbolic router consequence for gates (section 13)**: once symbolic
    owns arithmetic correctness, accuracy-based gates measure the
    latent's effect on non-symbolic subtasks. Release report must state
    which subtasks each gate is computed on.
