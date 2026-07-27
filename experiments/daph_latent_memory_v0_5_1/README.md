# DAPH Latent Working Memory v0.5.1

A research-grade scaffold for the first decisive DAPH Cognitive Machine experiment:

> Can a frozen transformer use **process-only latent working memory** to improve unseen multi-step reasoning under matched controls?

This build implements:

- structured, answer-excluding math working state
- synthetic algebra benchmark generation with IID / composition / OOD splits
- teacher hidden-state capture
- trainable structured-state encoder
- latent memory token projector
- frozen-transformer injection through `inputs_embeds`
- answer loss + hidden-state alignment loss
- controls: base, text-state, random latent, learned constant latent, shuffled latent, corrupted latent
- causal corruption tests
- linear answer-leakage probe
- exact-answer evaluation
- bootstrap confidence intervals
- utility / compute accounting
- checkpoint lineage metadata
- CLI scripts for end-to-end experiments
- unit tests for state leakage and dataset integrity

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/generate_dataset.py --out data/math_v1.jsonl --n 4000
python scripts/train_memory.py \
  --config configs/qwen3_1_7b.yaml \
  --dataset data/math_v1.jsonl \
  --output runs/latent_v1

python scripts/evaluate_all.py \
  --config configs/qwen3_1_7b.yaml \
  --dataset data/math_v1.jsonl \
  --checkpoint runs/latent_v1/best.pt \
  --output runs/latent_v1/eval
```

The default model is configurable. Model weights are not bundled.

## Scientific pass gate

Do **not** promote latent memory merely because it beats the bare base model.

A credible pass requires:

1. process-latent > base
2. process-latent > random latent
3. process-latent > learned constant latent
4. process-latent advantage persists on composition/OOD splits
5. shuffled/corrupted memory significantly reduces the gain
6. answer leakage remains below the configured threshold
7. benefit survives compute-normalized comparison

See `docs/EXPERIMENT_PROTOCOL.md`.

## Design invariant

Canonical structured state is authoritative. Latent memory is an ephemeral compiled representation.

```
structured state -> encoder -> latent tokens -> frozen transformer
```

The final answer is intentionally excluded from the state representation.

## Validation status

Local audit before publication:

- 9/9 unit tests pass
- Python compilation passes
- deterministic dataset generation verified across process hash seeds
- semantic corruption and deranged shuffle controls verified
- full Qwen3 GPU forward/backward qualification remains external and is intentionally not claimed as completed

See `docs/AUDIT_v0.5.1.md` for the defect record.
