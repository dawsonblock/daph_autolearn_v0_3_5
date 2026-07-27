import random
from daph_latent_memory.state.schema import MathWorkingState
from daph_latent_memory.state.corruption import corrupt_state, corrupt_latent_vector
import torch

def test_final_answer_field_forbidden():
    import pytest
    with pytest.raises(ValueError):
        MathWorkingState(domain="algebra", operation="solve", variables={}, constraints=[], answer="6")

def test_direct_answer_encoding_rejected():
    import pytest
    s = MathWorkingState(domain="algebra", operation="solve", variables={"x": "unknown"}, constraints=["2*x = 12"], intermediate_steps=["x = 6"])
    with pytest.raises(ValueError):
        s.assert_no_direct_answer_encoding("6")

def test_incidental_numeric_overlap_is_allowed():
    s = MathWorkingState(domain="algebra", operation="solve", variables={"x": "unknown", "coefficient": 6}, constraints=["6*x + 2 = 20"], unresolved_subgoals=["divide by 6 after removing the constant"])
    s.assert_no_direct_answer_encoding("3")

def test_corruption_changes_semantics():
    s = MathWorkingState(domain="algebra", operation="two_step", variables={"x": "unknown"}, constraints=["2*x + 3 = 11"], intermediate_steps=["after removing 3, obtain 2*x = 8"], unresolved_subgoals=["divide by 2"])
    c = corrupt_state(s, random.Random(1))
    assert c.process_text() != s.process_text()
    assert "CORRUPTED:" not in c.process_text()

def test_corrupt_latent_vector_zero_alpha():
    z = torch.randn(4, 8, 128)
    c = corrupt_latent_vector(z, alpha=0.0, seed=42)
    assert torch.allclose(c, z)

def test_corrupt_latent_vector_nonzero_alpha():
    z = torch.randn(4, 8, 128)
    c = corrupt_latent_vector(z, alpha=1.0, seed=42)
    assert not torch.allclose(c, z)

def test_corrupt_latent_vector_reproducible():
    z = torch.randn(4, 8, 128)
    c1 = corrupt_latent_vector(z, alpha=0.5, seed=123)
    c2 = corrupt_latent_vector(z, alpha=0.5, seed=123)
    assert torch.allclose(c1, c2)

def test_corrupt_latent_vector_different_seeds():
    z = torch.randn(4, 8, 128)
    c1 = corrupt_latent_vector(z, alpha=0.5, seed=1)
    c2 = corrupt_latent_vector(z, alpha=0.5, seed=2)
    assert not torch.allclose(c1, c2)
