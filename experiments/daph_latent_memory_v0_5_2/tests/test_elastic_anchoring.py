import torch
from daph_latent_memory.memory.elastic import ElasticBudgetAllocator
from daph_latent_memory.memory.anchoring import AntiAnchoringControl

def test_elastic_allocator_output_shape():
    alloc = ElasticBudgetAllocator(hidden_dim=64)
    h_q = torch.randn(4, 64)
    logits, values = alloc(h_q)
    assert logits.shape == (4, len(alloc.budget_options))
    assert values.shape == (4,)

def test_elastic_allocator_soft_mode():
    alloc = ElasticBudgetAllocator(hidden_dim=64)
    h_q = torch.randn(4, 64)
    logits, values = alloc(h_q, hard=False)
    # Soft mode: logits should sum to ~1
    sums = logits.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)

def test_elastic_allocator_hard_mode():
    alloc = ElasticBudgetAllocator(hidden_dim=64)
    h_q = torch.randn(4, 64)
    logits, values = alloc(h_q, hard=True)
    # Hard mode: one-hot (approximately, due to straight-through)
    hard_idx = logits.argmax(dim=-1)
    onehot = torch.zeros_like(logits).scatter_(-1, hard_idx.unsqueeze(-1), 1.0)
    assert torch.allclose(logits.detach(), onehot, atol=1e-5)

def test_elastic_allocator_budget_options():
    alloc = ElasticBudgetAllocator(hidden_dim=64, budget_options=[0, 1, 2, 4])
    assert alloc.budget_options == [0, 1, 2, 4]
    assert alloc.num_options == 4

def test_elastic_allocator_get_budget():
    alloc = ElasticBudgetAllocator(hidden_dim=64)
    h_q = torch.randn(4, 64)
    budget = alloc.get_budget(h_q, hard=True)
    assert budget.shape == (4,)
    assert (budget >= 0).all()

def test_elastic_allocator_budget_penalty():
    alloc = ElasticBudgetAllocator(hidden_dim=64)
    values = torch.tensor([2.0, 4.0, 8.0])
    penalty = alloc.budget_penalty(values, beta=0.01)
    assert penalty.item() > 0

def test_elastic_allocator_anneal_tau():
    alloc = ElasticBudgetAllocator(hidden_dim=64, tau_init=2.0, tau_min=0.1)
    assert abs(alloc.anneal_tau(0, 100) - 2.0) < 1e-10
    assert abs(alloc.anneal_tau(100, 100) - 0.1) < 1e-10
    assert alloc.anneal_tau(50, 100) < 2.0

def test_elastic_allocator_calibrate_beta():
    alloc = ElasticBudgetAllocator(hidden_dim=64)
    beta = alloc.calibrate_beta(0.001)
    assert beta == 0.001

def test_anti_anchoring_default_rho():
    ctrl = AntiAnchoringControl()
    assert ctrl.get_rho() == 0.5

def test_anti_anchoring_set_rho():
    ctrl = AntiAnchoringControl()
    ctrl.set_rho(0.75)
    assert ctrl.get_rho() == 0.75

def test_anti_anchoring_clamp():
    ctrl = AntiAnchoringControl()
    ctrl.set_rho(2.0)
    assert ctrl.get_rho() == 1.0
    ctrl.set_rho(-1.0)
    assert ctrl.get_rho() == 0.0

def test_anti_anchoring_scale_latent():
    ctrl = AntiAnchoringControl(rho=0.5)
    z_mem = torch.ones(4, 8, 64)
    z_base = torch.zeros(4, 8, 64)
    z = ctrl.scale_latent(z_mem, z_base)
    assert torch.allclose(z, 0.5 * z_mem)

def test_anti_anchoring_scale_latent_rho_zero():
    ctrl = AntiAnchoringControl(rho=0.0)
    z_mem = torch.ones(4, 8, 64)
    z_base = torch.ones(4, 8, 64) * 2
    z = ctrl.scale_latent(z_mem, z_base)
    assert torch.allclose(z, z_base)  # memory fully ignored

def test_anti_anchoring_scale_latent_rho_one():
    ctrl = AntiAnchoringControl(rho=1.0)
    z_mem = torch.ones(4, 8, 64)
    z_base = torch.ones(4, 8, 64) * 2
    z = ctrl.scale_latent(z_mem, z_base)
    assert torch.allclose(z, z_mem)  # memory fully trusted

def test_anti_anchoring_scale_retrieved():
    ctrl = AntiAnchoringControl(rho=0.5)
    z = torch.ones(4, 8, 64)
    scaled = ctrl.scale_retrieved(z)
    assert torch.allclose(scaled, 0.5 * z)

def test_anti_anchoring_should_retrieve_high_rho():
    ctrl = AntiAnchoringControl(rho=0.8)
    assert ctrl.should_retrieve(0.6, threshold=0.5)

def test_anti_anchoring_should_not_retrieve_low_rho():
    ctrl = AntiAnchoringControl(rho=0.01)
    assert not ctrl.should_retrieve(0.6, threshold=0.5)
