import torch
from daph_latent_memory.latent.skill_bank import SkillBank
from daph_latent_memory.latent.instance_state import InstanceEncoder
from daph_latent_memory.latent.composer import LinearComposer, GatedComposer, make_composer

def test_skill_bank_shape():
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    assert bank.skills.shape == (6, 8, 256)

def test_skill_bank_get_skill():
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    skill = bank.get_skill(0)
    assert skill.shape == (8, 256)

def test_skill_bank_get_batch():
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    ids = torch.tensor([0, 1, 2])
    skills = bank.get_skill_batch(ids)
    assert skills.shape == (3, 8, 256)

def test_skill_bank_freeze():
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    bank.freeze()
    assert not bank.skills.requires_grad

def test_skill_bank_unfreeze():
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    bank.freeze()
    bank.unfreeze()
    assert bank.skills.requires_grad

def test_skill_bank_partial_freeze():
    bank = SkillBank(num_skills=6, latent_tokens=8, skill_dim=256)
    bank.partial_freeze(0.5)  # freeze 3 of 6
    mask = bank.get_freeze_mask()
    assert mask is not None
    assert mask.sum().item() == 3
    assert mask[:3].all()
    assert not mask[3:].any()

def test_instance_encoder_shape():
    encoder = InstanceEncoder(vocab_size=1000, input_embed_dim=512, hidden_dim=256, latent_tokens=8, instance_dim=256)
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32)
    z = encoder(input_ids, attention_mask)
    assert z.shape == (4, 8, 256)

def test_linear_composer():
    composer = LinearComposer(skill_dim=256, instance_dim=256, output_dim=512, latent_tokens=8)
    z_skill = torch.randn(4, 8, 256)
    z_instance = torch.randn(4, 8, 256)
    z = composer(z_skill, z_instance)
    assert z.shape == (4, 8, 512)

def test_gated_composer():
    composer = GatedComposer(skill_dim=256, instance_dim=256, output_dim=512, latent_tokens=8)
    z_skill = torch.randn(4, 8, 256)
    z_instance = torch.randn(4, 8, 256)
    z = composer(z_skill, z_instance)
    assert z.shape == (4, 8, 512)

def test_make_composer_linear():
    composer = make_composer("linear", 256, 256, 512, 8)
    assert isinstance(composer, LinearComposer)

def test_make_composer_gated():
    composer = make_composer("gated", 256, 256, 512, 8)
    assert isinstance(composer, GatedComposer)

def test_make_composer_invalid():
    try:
        make_composer("invalid", 256, 256, 512, 8)
        assert False
    except ValueError:
        pass
