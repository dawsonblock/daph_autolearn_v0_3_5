import torch, tempfile, os
from daph_latent_memory.training.checkpoint import save_checkpoint, load_checkpoint, file_sha256

def test_save_load_checkpoint():
    state = {"weights": torch.tensor([1.0, 2.0, 3.0])}
    meta = {"seed": 1337, "phase": "A"}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(path, state, meta)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["state_dict"]["weights"], state["weights"])
        assert loaded["meta"]["seed"] == 1337
        assert loaded["meta"]["phase"] == "A"
        # Check meta JSON was written
        json_path = path.replace(".pt", ".json")
        assert os.path.exists(json_path)
    finally:
        if os.path.exists(path): os.unlink(path)
        json_path = path.replace(".pt", ".json")
        if os.path.exists(json_path): os.unlink(json_path)

def test_file_sha256():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("hello world")
        path = f.name
    try:
        h = file_sha256(path)
        assert len(h) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in h)
    finally:
        os.unlink(path)
