import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from daph_learning.steering.hooks import residual_addition_hook, validate_vector_for_model
from daph_learning.steering.types import SteeringSpec, SteeringVector


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([IdentityBlock(), IdentityBlock()])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.config = type("Cfg", (), {"hidden_size": 4, "_name_or_path": "fake"})()


def test_last_token_scope_only_changes_last_position():
    model = FakeModel()
    x = torch.zeros(1, 3, 4)
    vec = np.ones(4, dtype=np.float32)
    with residual_addition_hook(model, layer_index=0, vector=vec, alpha=2.0, token_scope="last"):
        y = model.model.layers[0](x)
    assert torch.all(y[:, :-1, :] == 0)
    assert torch.all(y[:, -1, :] == 2)


def test_vector_model_dimension_validation():
    model = FakeModel()
    good = SteeringVector(
        SteeringSpec("v", "tool_policy", "invoke", layer=1, hidden_size=4),
        np.zeros(4, dtype=np.float32),
    )
    validate_vector_for_model(model, good)
    bad = SteeringVector(
        SteeringSpec("v2", "tool_policy", "invoke", layer=1, hidden_size=3),
        np.zeros(3, dtype=np.float32),
    )
    with pytest.raises(ValueError):
        validate_vector_for_model(model, bad)
