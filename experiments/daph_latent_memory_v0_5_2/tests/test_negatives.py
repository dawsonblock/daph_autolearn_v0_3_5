import torch
from daph_latent_memory.training.negatives import sample_negatives, sample_same_skill_pairs, NegativePair
from daph_latent_memory.benchmarks.algebra import generate_example

def test_sample_negatives_returns_k():
    anchor = generate_example("train", 0, seed=1337)
    pool = [generate_example("train", i, seed=1337) for i in range(1, 50)]
    negs = sample_negatives(anchor, pool, k=3, seed=42)
    assert len(negs) == 3
    for neg in negs:
        assert isinstance(neg, NegativePair)
        assert neg.anchor.example_id == anchor.example_id
        assert neg.negative.example_id != anchor.example_id

def test_sample_negatives_excludes_anchor():
    anchor = generate_example("train", 0, seed=1337)
    pool = [generate_example("train", i, seed=1337) for i in range(1, 50)]
    negs = sample_negatives(anchor, pool, k=3, seed=42)
    for neg in negs:
        assert neg.negative.example_id != anchor.example_id

def test_sample_negatives_reproducible():
    anchor = generate_example("train", 0, seed=1337)
    pool = [generate_example("train", i, seed=1337) for i in range(1, 50)]
    negs1 = sample_negatives(anchor, pool, k=3, seed=42)
    negs2 = sample_negatives(anchor, pool, k=3, seed=42)
    for n1, n2 in zip(negs1, negs2):
        assert n1.negative.example_id == n2.negative.example_id

def test_sample_same_skill_pairs():
    examples = [generate_example("train", i, seed=1337) for i in range(50)]
    pairs = sample_same_skill_pairs(examples, seed=42)
    assert len(pairs) > 0
    for a, b in pairs:
        assert a.example_id != b.example_id
        assert a.skill_label == b.skill_label

def test_sample_same_skill_pairs_reproducible():
    examples = [generate_example("train", i, seed=1337) for i in range(50)]
    pairs1 = sample_same_skill_pairs(examples, seed=42)
    pairs2 = sample_same_skill_pairs(examples, seed=42)
    assert len(pairs1) == len(pairs2)
    for (a1, b1), (a2, b2) in zip(pairs1, pairs2):
        assert a1.example_id == a2.example_id
        assert b1.example_id == b2.example_id
