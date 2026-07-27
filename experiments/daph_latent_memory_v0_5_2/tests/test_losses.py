import torch
from daph_latent_memory.training.losses import functional_mismatch_loss, disentangle_loss, invariance_loss, total_loss_v052

def test_functional_mismatch_loss_zero_when_margin_satisfied():
    # r_pos=1.0, r_neg=3.0, margin=1.0 -> r_neg - r_pos = 2 >= margin=1
    # hinge = max(0, margin - (r_neg - r_pos)) = max(0, 1 - 2) = 0
    r_pos = torch.tensor([1.0])
    r_neg = torch.tensor([[3.0]])  # r_neg - r_pos = 2 >= margin=1
    loss = functional_mismatch_loss(r_pos, r_neg, margin=1.0)
    assert loss.item() == 0.0, f"Expected 0, got {loss.item()}"

def test_functional_mismatch_loss_positive_when_margin_violated():
    # r_pos=1.0, r_neg=1.5, margin=1.0 -> r_neg - r_pos = 0.5 < margin=1
    # hinge = max(0, 1 - 0.5) = 0.5
    r_pos = torch.tensor([1.0])
    r_neg = torch.tensor([[1.5]])  # r_neg - r_pos = 0.5 < margin=1.0
    loss = functional_mismatch_loss(r_pos, r_neg, margin=1.0)
    assert loss.item() > 0, f"Expected positive, got {loss.item()}"
    assert abs(loss.item() - 0.5) < 1e-6, f"Expected 0.5, got {loss.item()}"

def test_functional_mismatch_loss_batch():
    r_pos = torch.tensor([1.0, 2.0, 0.5])
    r_neg = torch.tensor([[3.0, 2.0], [5.0, 1.0], [1.0, 0.6]])
    loss = functional_mismatch_loss(r_pos, r_neg, margin=1.0)
    # Example 0: max(0, 1-(3-1))=0, max(0, 1-(2-1))=0 -> mean=0
    # Example 1: max(0, 1-(5-2))=0, max(0, 1-(1-2))=max(0,2)=2 -> mean=1
    # Example 2: max(0, 1-(1-0.5))=0.5, max(0, 1-(0.6-0.5))=0.9 -> mean=0.7
    # Overall mean = (0+1+0.7)/3 = 0.5667
    assert abs(loss.item() - (0.0+1.0+0.7)/3) < 1e-5

def test_disentangle_loss_zero_for_orthogonal():
    z_skill = torch.tensor([[[1.0, 0.0, 0.0]]])
    z_instance = torch.tensor([[[0.0, 1.0, 0.0]]])
    loss = disentangle_loss(z_skill, z_instance)
    assert loss.item() < 1e-6, f"Expected ~0 for orthogonal, got {loss.item()}"

def test_disentangle_loss_positive_for_aligned():
    z_skill = torch.tensor([[[1.0, 0.0, 0.0]]])
    z_instance = torch.tensor([[[1.0, 0.0, 0.0]]])
    loss = disentangle_loss(z_skill, z_instance)
    assert loss.item() > 0.9, f"Expected ~1 for aligned, got {loss.item()}"

def test_invariance_loss_zero_for_identical():
    z = torch.tensor([[[1.0, 0.0, 0.0]]])
    loss = invariance_loss(z, z)
    assert loss.item() < 1e-6, f"Expected ~0 for identical, got {loss.item()}"

def test_invariance_loss_one_for_orthogonal():
    z_a = torch.tensor([[[1.0, 0.0, 0.0]]])
    z_b = torch.tensor([[[0.0, 1.0, 0.0]]])
    loss = invariance_loss(z_a, z_b)
    assert abs(loss.item() - 1.0) < 1e-6, f"Expected 1 for orthogonal, got {loss.item()}"

def test_total_loss_v052():
    task = torch.tensor(1.0)
    func = torch.tensor(0.5)
    dis = torch.tensor(0.3)
    loss = total_loss_v052(task, func, dis, answer_weight=1.0, functional_weight=0.5, disentangle_weight=0.1)
    expected = 1.0 * 1.0 + 0.5 * 0.5 + 0.1 * 0.3
    assert abs(loss.item() - expected) < 1e-6

def test_total_loss_v052_none_components():
    task = torch.tensor(1.0)
    loss = total_loss_v052(task, None, None, answer_weight=1.0, functional_weight=0.5, disentangle_weight=0.1)
    assert abs(loss.item() - 1.0) < 1e-6
