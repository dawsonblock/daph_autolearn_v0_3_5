import torch
from daph_latent_memory.memory.bank import MemoryBank, MemoryEntry, MemoryState, MemoryAction
from daph_latent_memory.memory.manager import MemoryManager
from daph_latent_memory.memory.verifier import LatentVerifier
from daph_latent_memory.memory.selector import MemorySelector

def test_memory_bank_add():
    bank = MemoryBank(key_dim=64, latent_dim=128)
    entry = MemoryEntry(key=torch.randn(64), latent=torch.randn(8, 128), capability_id="add", source_task="train-0")
    bank.add(entry)
    assert len(bank) == 1

def test_memory_bank_retrieve():
    bank = MemoryBank(key_dim=64, latent_dim=128)
    key = torch.randn(64)
    entry = MemoryEntry(key=key, latent=torch.randn(8, 128), capability_id="add", source_task="train-0")
    bank.add(entry)
    results = bank.retrieve(key, top_k=1)
    assert len(results) == 1
    assert results[0][0].capability_id == "add"

def test_memory_bank_max_size():
    bank = MemoryBank(key_dim=64, latent_dim=128, max_size=3)
    for i in range(5):
        bank.add(MemoryEntry(key=torch.randn(64), latent=torch.randn(8, 128), capability_id=f"add{i}", source_task=f"train-{i}"))
    assert len(bank) <= 3

def test_memory_manager_add_action():
    bank = MemoryBank(key_dim=64, latent_dim=128)
    manager = MemoryManager(bank)
    entry = MemoryEntry(key=torch.randn(64), latent=torch.randn(8, 128), capability_id="add", source_task="train-0", confidence=0.9)
    action = manager.propose_action(entry)
    assert action == MemoryAction.ADD
    modified = manager.execute(action, entry)
    assert modified
    assert len(bank) == 1

def test_memory_manager_merge_action():
    bank = MemoryBank(key_dim=64, latent_dim=128)
    manager = MemoryManager(bank, similarity_threshold=0.5)
    key = torch.randn(64)
    entry1 = MemoryEntry(key=key, latent=torch.randn(8, 128), capability_id="add", source_task="train-0", confidence=0.9, state=MemoryState.VERIFIED)
    bank.add(entry1)
    # Similar key, similar confidence -> should propose MERGE
    entry2 = MemoryEntry(key=key + 0.01, latent=torch.randn(8, 128), capability_id="add", source_task="train-1", confidence=0.9)
    action = manager.propose_action(entry2)
    assert action in (MemoryAction.MERGE, MemoryAction.UPDATE)
    modified = manager.execute(action, entry2)
    assert modified

def test_memory_manager_delete_action():
    bank = MemoryBank(key_dim=64, latent_dim=128)
    manager = MemoryManager(bank, similarity_threshold=0.5)
    key = torch.randn(64)
    entry = MemoryEntry(key=key, latent=torch.randn(8, 128), capability_id="add", source_task="train-0", confidence=0.5, state=MemoryState.CONTRADICTED)
    bank.add(entry)
    entry2 = MemoryEntry(key=key + 0.01, latent=torch.randn(8, 128), capability_id="add", source_task="train-1", confidence=0.9)
    action = manager.propose_action(entry2)
    assert action == MemoryAction.DELETE

def test_memory_verifier_arithmetic():
    verifier = LatentVerifier(latent_dim=128)
    assert verifier.verify_arithmetic("1+1", "2", "2")
    assert not verifier.verify_arithmetic("1+1", "3", "2")

def test_memory_verifier_network():
    verifier = LatentVerifier(latent_dim=64)
    q = torch.randn(64)
    z = torch.randn(8, 64)
    passed, score = verifier.verify(q, z)
    assert isinstance(passed, bool)
    assert 0.0 <= score <= 1.0

def test_memory_selector_heuristic():
    selector = MemorySelector(key_dim=64)
    assert selector.heuristic_decision(0.8, tau=0.5) == 1
    assert selector.heuristic_decision(0.3, tau=0.5) == 0

def test_memory_selector_forward():
    selector = MemorySelector(key_dim=64)
    q = torch.randn(64)
    c = torch.randn(64)
    prob = selector(q, c)
    assert 0.0 <= prob.item() <= 1.0

def test_memory_selector_reinforce_loss():
    selector = MemorySelector(key_dim=64)
    q = torch.randn(64)
    c = torch.randn(64)
    loss = selector.reinforce_loss(q, c, delta_r=0.5, capability_id="add")
    assert loss.requires_grad
    assert loss.item() != 0 or len(selector.baseline) > 0  # baseline should be updated
