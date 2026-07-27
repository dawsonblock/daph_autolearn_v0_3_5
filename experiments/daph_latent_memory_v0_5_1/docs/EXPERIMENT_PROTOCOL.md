# DAPH Latent Memory Scientific Qualification Protocol

## Primary hypothesis

A frozen transformer supplied with meaningful, process-only latent working memory will solve unseen
multi-step tasks better than matched controls.

## Required conditions

1. Base model
2. Text process state
3. Process latent state
4. Random latent state
5. Learned constant latent state
6. Shuffled latent state
7. Corrupted latent state

A future extension should add a stricter compute-matched extra-reasoning baseline.

## Dataset separation

- Train: one- and two-step algebra
- IID: unseen values, familiar structures
- Composition: unseen composition of known operations
- OOD: variables on both sides / deeper algebraic structure

Do not randomly split near-duplicates and call it generalization.

## Leakage rules

The canonical process state must not contain the final answer field.
String-level answer-presence checks are performed during generation.

The leakage probe is diagnostic, not definitive. High linear decodability of answer class from latent
state is grounds for investigation, not automatic proof of leakage.

## Causal evidence

The strongest evidence is a pattern where:

- correct process latent improves performance
- shuffled memory loses that advantage
- targeted corruption degrades performance
- text state remains an approximate upper bound

This shows the model is using task-specific content rather than a generic learned prefix.

## Promotion gate

Promote to recurrent working-memory experiments only when process latent:

- beats base, random, and constant controls
- retains gains on composition and OOD
- has a measurable corruption sensitivity
- has acceptable latency / compute cost
- passes leakage review
