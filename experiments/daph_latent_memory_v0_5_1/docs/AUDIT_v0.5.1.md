# v0.5.1 Audit and Hardening Record

The original v0.5.0 scaffold passed its small unit suite, but deeper review found several issues that would have compromised the scientific result. v0.5.1 fixes them before publication.

## Corrected issues

1. **Non-reproducible dataset seeds** — Python's randomized `hash(split)` was used in seed construction. Replaced with fixed split offsets. Cross-process dataset hashes are now stable.
2. **False-positive leakage guard** — substring matching rejected innocent numerical overlap and biased the generated distribution. Replaced with direct-answer pattern detection (`x = answer`, `answer: value`, etc.).
3. **Weak corruption control** — the old corruption condition only prefixed text with `CORRUPTED:` while leaving the underlying state intact. Corruption now changes numerical/process semantics.
4. **Shuffled control self-matches** — random shuffle could return an example's own state. Replaced with a deterministic derangement.
5. **Mixed-dtype latent injection** — FP32 encoder output could be concatenated with BF16/FP16 model embeddings. Latents are explicitly cast to the embedding device/dtype.
6. **Answer padding contributed to loss** — padded answer IDs were treated as valid targets. Padding labels are now set to `-100`.
7. **Dropped final gradient accumulation remainder** — the final partial accumulation window was never stepped. It now performs an optimizer step.
8. **Constant baseline was not actually learned** — the prior evaluation used the mean process latent and called it a learned constant. v0.5.1 trains a true task-independent K-token soft prompt on the same training batches.
9. **Evaluation could score prompt numbers as predictions** — decoded base generations included the prompt, and answer extraction selected the last number. Evaluation now decodes continuation tokens only.
10. **Text-state prompt formatting** — removed the duplicated early `Answer:` marker before process state.
11. **Token-cost reporting** — evaluation now records generated continuation tokens instead of assuming `max_new_tokens` was always consumed.

## Local qualification performed

- Unit tests: 9 passed.
- Python bytecode compilation: passed.
- 500-example dataset smoke generation: passed.
- Cross-process deterministic generation: identical SHA-256 for equal seed/config.

## Not locally qualified

A full forward/backward run against `Qwen/Qwen3-1.7B` was not executed in this environment because the model checkpoint is not bundled and model download/GPU execution is external to this artifact. The first GPU run remains a required qualification gate.
