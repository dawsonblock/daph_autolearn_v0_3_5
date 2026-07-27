from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from daph_learning.steering.hooks import residual_addition_hook


class Layer(nn.Module):
    def forward(self, x):
        return x


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer()])
        self.config = SimpleNamespace(hidden_size=3)


def test_last_scope_is_prompt_only_and_one_shot():
    model = FakeModel()
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    prompt = torch.zeros((1, 4, 3))
    decode = torch.zeros((1, 1, 3))
    growing_no_cache = torch.zeros((1, 5, 3))

    with residual_addition_hook(
        model, layer_index=0, vector=vector, alpha=1.0, token_scope="last"
    ):
        first = model.model.layers[0](prompt)
        second = model.model.layers[0](decode)
        third = model.model.layers[0](growing_no_cache)

    assert torch.allclose(first[0, -1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.count_nonzero(first[0, :-1]) == 0
    assert torch.count_nonzero(second) == 0
    # One-shot state prevents reapplication even if a later no-cache pass has seq_len > 1.
    assert torch.count_nonzero(third) == 0
