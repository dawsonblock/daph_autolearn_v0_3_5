import torch
from daph_latent_memory.router.router import LatentRouter
from daph_latent_memory.latent.skill_bank import SkillBank

def test_router_soft_output_shape():
    router = LatentRouter(hidden_dim=512, num_skills=6)
    h_q = torch.randn(4, 512)
    alpha = router(h_q)
    assert alpha.shape == (4, 6)

def test_router_softmax_normalization():
    router = LatentRouter(hidden_dim=512, num_skills=6)
    h_q = torch.randn(4, 512)
    alpha = router(h_q)
    sums = alpha.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)

def test_router_topk_output_shape():
    router = LatentRouter(hidden_dim=512, num_skills=6, router_topk=2)
    h_q = torch.randn(4, 512)
    alpha = router(h_q)
    assert alpha.shape == (4, 6)
    # TopK should zero out non-top-k entries
    non_zero = (alpha > 0).sum(dim=-1)
    assert (non_zero <= 2).all()

def test_router_route_shape():
    router = LatentRouter(hidden_dim=512, num_skills=6)
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    h_q = torch.randn(4, 512)
    z_skill = router.route(h_q, bank)
    assert z_skill.shape == (4, 8, 256)

def test_router_supervised_loss():
    router = LatentRouter(hidden_dim=512, num_skills=6)
    h_q = torch.randn(4, 512)
    skill_ids = torch.tensor([0, 1, 2, 3])
    loss = router.supervised_loss(h_q, skill_ids)
    assert loss.item() > 0
    assert loss.requires_grad
