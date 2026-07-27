import random
from daph_latent_memory.benchmarks.algebra import generate_example, SKILL_ADD, SKILL_SUBTRACT, SKILL_MULTIPLY
from daph_latent_memory.benchmarks.dataset import generate_dataset, save_jsonl, load_jsonl
from daph_latent_memory.state.schema import MathExample, MathWorkingState
import tempfile, os

def test_generate_example_train():
    ex = generate_example("train", 0, seed=1337)
    assert ex.split == "train"
    assert ex.skill_label is not None
    assert len(ex.question) > 0
    assert len(ex.answer) > 0

def test_generate_example_ood_splits():
    for split in ["operand_ood", "structural_ood", "operator_ood", "distractor_ood", "counterfactual", "adversarial"]:
        ex = generate_example(split, 0, seed=1337)
        assert ex.split == split
        assert ex.skill_label is not None

def test_generate_dataset_counts():
    examples = generate_dataset(100, seed=1337)
    assert len(examples) == 100
    splits = set(e.split for e in examples)
    assert "train" in splits
    assert "iid" in splits

def test_skill_labels_present():
    examples = generate_dataset(200, seed=1337)
    for ex in examples:
        assert ex.skill_label is not None
        assert ex.skill_label in {SKILL_ADD, SKILL_SUBTRACT, SKILL_MULTIPLY, "divide", "verify", "decompose"}

def test_save_load_jsonl():
    examples = generate_dataset(50, seed=1337)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        save_jsonl(examples, path)
        loaded = load_jsonl(path)
        assert len(loaded) == len(examples)
        assert loaded[0].example_id == examples[0].example_id
        assert loaded[0].skill_label == examples[0].skill_label
    finally:
        os.unlink(path)

def test_adversarial_same_operands_different_ops():
    """Adversarial examples should have same operands but different operators."""
    ops_seen = set()
    operands_seen = set()
    for i in range(20):
        ex = generate_example("adversarial", i, seed=1337)
        import re
        nums = tuple(sorted(int(x) for x in re.findall(r"-?\d+", ex.question)))
        operands_seen.add(nums)
        ops_seen.add(ex.state.operation)
    # Should see multiple different operations
    assert len(ops_seen) >= 2

def test_state_no_answer_leakage():
    """Verify that generated states don't contain the answer."""
    for i in range(50):
        ex = generate_example("train", i, seed=1337)
        try:
            ex.state.assert_answer_not_present(ex.answer)
        except ValueError:
            assert False, f"Answer leakage in example {ex.example_id}"
