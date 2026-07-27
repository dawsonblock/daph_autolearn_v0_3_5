import torch
from daph_latent_memory.memory.clue import CLUEVerifier

def test_clue_initial_state():
    v = CLUEVerifier(hidden_dim=64)
    assert not v.is_calibrated()
    assert v.confidence(torch.randn(64)) == 0.0

def test_clue_update_success():
    v = CLUEVerifier(hidden_dim=64)
    v.update(torch.randn(64), success=True)
    assert v._success_count == 1
    assert v.c_success is not None

def test_clue_update_failure():
    v = CLUEVerifier(hidden_dim=64)
    v.update(torch.randn(64), success=False)
    assert v._failure_count == 1
    assert v.c_failure is not None

def test_clue_calibrated():
    v = CLUEVerifier(hidden_dim=64)
    v.update(torch.randn(64), success=True)
    v.update(torch.randn(64), success=False)
    assert v.is_calibrated()

def test_clue_confidence_positive_for_success_like():
    v = CLUEVerifier(hidden_dim=64)
    # Make success centroid at [1, 0, 0, ...] and failure at [-1, 0, 0, ...]
    success_vec = torch.zeros(64); success_vec[0] = 1.0
    failure_vec = torch.zeros(64); failure_vec[0] = -1.0
    v.update(success_vec, success=True)
    v.update(failure_vec, success=False)
    # A delta_h close to success should give positive confidence
    test_vec = torch.zeros(64); test_vec[0] = 0.9
    c = v.confidence(test_vec)
    assert c > 0, f"Expected positive confidence, got {c}"

def test_clue_confidence_negative_for_failure_like():
    v = CLUEVerifier(hidden_dim=64)
    success_vec = torch.zeros(64); success_vec[0] = 1.0
    failure_vec = torch.zeros(64); failure_vec[0] = -1.0
    v.update(success_vec, success=True)
    v.update(failure_vec, success=False)
    test_vec = torch.zeros(64); test_vec[0] = -0.9
    c = v.confidence(test_vec)
    assert c < 0, f"Expected negative confidence, got {c}"

def test_clue_compute_delta():
    v = CLUEVerifier(hidden_dim=64)
    h_start = torch.randn(2, 64)
    h_end = torch.randn(2, 64)
    delta = v.compute_delta(h_start, h_end)
    assert delta.shape == (2, 64)
    assert torch.allclose(delta, h_end - h_start)

def test_clue_reset():
    v = CLUEVerifier(hidden_dim=64)
    v.update(torch.randn(64), success=True)
    v.update(torch.randn(64), success=False)
    v.reset()
    assert not v.is_calibrated()
    assert v._success_count == 0

def test_clue_save_load(tmp_path):
    v = CLUEVerifier(hidden_dim=64)
    v.update(torch.randn(64), success=True)
    v.update(torch.randn(64), success=False)
    path = str(tmp_path / "clue.pt")
    v.save(path)
    v2 = CLUEVerifier(hidden_dim=64)
    v2.load(path)
    assert v2.is_calibrated()
    assert v2._success_count == 1
