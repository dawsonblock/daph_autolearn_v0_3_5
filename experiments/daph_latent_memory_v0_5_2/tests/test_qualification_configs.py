"""Tests for the v0.3.8 DEF-04 GPU qualification configs.

Validates that the two new production configs (Qwen2.5-1.5B-Instruct and
Qwen2.5-3B-Instruct) load correctly and contain all fields required by the
training/evaluation scripts, including the v0.3.8 additions:
  - training.latent_relative_norm_limit (DEF-02 safety clamp)
  - training.functional_margin (Task 3.2 calibration target)
  - training.functional_negatives == 3 (K=3 hard negatives)
  - evaluation.seed_count >= 5 (Gate 6 minimum)
  - memory.skill_dim / instance_dim / num_skills / composition (v0.5.2 two-channel)

Also verifies the configs are internally consistent (no legacy
alignment_weight, batch sizes match the repair plan §4 Task 4.1).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from daph_latent_memory.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


REQUIRED_MODEL_FIELDS = {"name", "dtype", "trust_remote_code", "device_map"}
REQUIRED_MEMORY_FIELDS = {
    "latent_tokens", "encoder_width", "encoder_layers", "dropout",
    "max_state_tokens", "skill_dim", "instance_dim", "num_skills", "composition",
}
REQUIRED_TRAINING_FIELDS = {
    "epochs", "batch_size", "learning_rate", "weight_decay",
    "answer_weight", "functional_weight", "disentangle_weight",
    "invariance_weight", "functional_margin", "functional_negatives",
    "max_prompt_tokens", "max_answer_tokens", "gradient_clip",
    "unfreeze_fraction", "latent_relative_norm_limit",
}
REQUIRED_EVAL_FIELDS = {
    "bootstrap_samples", "generation_max_new_tokens", "temperature",
    "corruption_alphas", "null_permutations", "null_percentile",
    "css_floor_pp", "seed_count",
}


@pytest.fixture(params=[
    "qwen25_1_5b.yaml",
    "qwen25_3b.yaml",
])
def qualification_config(request):
    path = CONFIGS_DIR / request.param
    if not path.exists():
        pytest.skip(f"config {request.param} not found at {path}")
    return load_config(path), request.param


def test_config_loads_as_dict(qualification_config):
    cfg, name = qualification_config
    assert isinstance(cfg, dict), f"{name} did not load as a dict"


def test_config_has_seed(qualification_config):
    cfg, name = qualification_config
    assert "seed" in cfg and isinstance(cfg["seed"], int), f"{name} missing seed"


def test_config_model_fields(qualification_config):
    cfg, name = qualification_config
    model = cfg.get("model", {})
    missing = REQUIRED_MODEL_FIELDS - set(model.keys())
    assert not missing, f"{name} model missing fields: {missing}"
    assert "Qwen2.5" in model["name"], f"{name} model.name should target Qwen2.5"
    assert model["dtype"] in ("bfloat16", "float16", "float32")


def test_config_memory_fields(qualification_config):
    cfg, name = qualification_config
    mem = cfg.get("memory", {})
    missing = REQUIRED_MEMORY_FIELDS - set(mem.keys())
    assert not missing, f"{name} memory missing fields: {missing}"
    assert mem["num_skills"] == 6, f"{name} num_skills should be 6 (add/sub/mul/div/verify/decompose)"
    assert mem["composition"] in ("gated", "linear"), f"{name} composition must be gated or linear"
    assert mem["skill_dim"] > 0 and mem["instance_dim"] > 0


def test_config_training_fields(qualification_config):
    cfg, name = qualification_config
    tcfg = cfg.get("training", {})
    missing = REQUIRED_TRAINING_FIELDS - set(tcfg.keys())
    assert not missing, f"{name} training missing fields: {missing}"


def test_config_no_legacy_alignment_weight(qualification_config):
    """DEF-03: the v0.5.1 alignment_weight must be absent from v0.5.2 configs."""
    cfg, name = qualification_config
    tcfg = cfg.get("training", {})
    assert "alignment_weight" not in tcfg, (
        f"{name} still has legacy alignment_weight — DEF-03 violation"
    )


def test_config_functional_negatives_is_three(qualification_config):
    """Repair plan §3 Task 3.2: K=3 hard negatives."""
    cfg, name = qualification_config
    assert cfg["training"]["functional_negatives"] == 3, (
        f"{name} functional_negatives must be 3 (K=3 hard negatives)"
    )


def test_config_latent_relative_norm_limit_set(qualification_config):
    """DEF-02: the latent injection safety clamp must be configured."""
    cfg, name = qualification_config
    limit = cfg["training"].get("latent_relative_norm_limit")
    assert limit is not None, f"{name} missing latent_relative_norm_limit"
    assert 0.0 <= limit <= 1.0, f"{name} latent_relative_norm_limit must be in [0,1]"
    # The production default is 0.65 (matches the residual-hook default).
    assert limit == 0.65, f"{name} latent_relative_norm_limit should be 0.65, got {limit}"


def test_config_unfreeze_fraction_is_0_3(qualification_config):
    """Repair plan §3 Task 3.3: Phase C freeze anneal over 30% of steps."""
    cfg, name = qualification_config
    assert cfg["training"]["unfreeze_fraction"] == 0.3, (
        f"{name} unfreeze_fraction should be 0.3"
    )


def test_config_evaluation_fields(qualification_config):
    cfg, name = qualification_config
    eval_cfg = cfg.get("evaluation", {})
    missing = REQUIRED_EVAL_FIELDS - set(eval_cfg.keys())
    assert not missing, f"{name} evaluation missing fields: {missing}"


def test_config_seed_count_meets_gate6_minimum(qualification_config):
    """Gate 6 requires >= 5 seeds; the repair plan §4 Task 4.1 specifies 10."""
    cfg, name = qualification_config
    sc = cfg["evaluation"]["seed_count"]
    assert sc >= 5, f"{name} seed_count {sc} below Gate 6 minimum (5)"
    assert sc == 10, f"{name} seed_count should be 10 per repair plan §4 Task 4.1, got {sc}"


def test_config_corruption_alphas_include_endpoints(qualification_config):
    """Gate 2 needs alpha=0 and alpha=2 endpoints, plus alpha=0.5 for the
    robustness band check."""
    cfg, name = qualification_config
    alphas = cfg["evaluation"]["corruption_alphas"]
    assert 0.0 in alphas, f"{name} corruption_alphas missing 0.0"
    assert 2.0 in alphas, f"{name} corruption_alphas missing 2.0"
    assert 0.5 in alphas, f"{name} corruption_alphas missing 0.5 (Gate 2 robustness band)"


# --- Per-model specific checks (repair plan §4 Task 4.1) ---

def test_qwen25_1_5b_batch_size_is_32():
    path = CONFIGS_DIR / "qwen25_1_5b.yaml"
    if not path.exists():
        pytest.skip("qwen25_1_5b.yaml not found")
    cfg = load_config(path)
    assert cfg["training"]["batch_size"] == 32, (
        "qwen25_1_5b batch_size should be 32 per repair plan §4 Task 4.1"
    )


def test_qwen25_3b_batch_size_is_16():
    path = CONFIGS_DIR / "qwen25_3b.yaml"
    if not path.exists():
        pytest.skip("qwen25_3b.yaml not found")
    cfg = load_config(path)
    assert cfg["training"]["batch_size"] == 16, (
        "qwen25_3b batch_size should be 16 (3B needs more memory per example)"
    )


def test_qwen25_1_5b_targets_correct_model():
    path = CONFIGS_DIR / "qwen25_1_5b.yaml"
    if not path.exists():
        pytest.skip("qwen25_1_5b.yaml not found")
    cfg = load_config(path)
    assert cfg["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_qwen25_3b_targets_correct_model():
    path = CONFIGS_DIR / "qwen25_3b.yaml"
    if not path.exists():
        pytest.skip("qwen25_3b.yaml not found")
    cfg = load_config(path)
    assert cfg["model"]["name"] == "Qwen/Qwen2.5-3B-Instruct"
