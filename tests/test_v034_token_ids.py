import pytest

from daph_learning.routing.steered_router import resolve_route_token_ids


class SingleToken:
    def encode(self, text, add_special_tokens=False):
        mapping = {" SYMBOLIC": [10], " LLM": [20]}
        return mapping[text]


class MultiToken:
    def encode(self, text, add_special_tokens=False):
        return [1, 2]


def test_resolve_route_token_ids_single_token():
    assert resolve_route_token_ids(SingleToken()) == (10, 20)


def test_resolve_route_token_ids_rejects_multitoken_labels():
    with pytest.raises(ValueError, match="single-token"):
        resolve_route_token_ids(MultiToken())
