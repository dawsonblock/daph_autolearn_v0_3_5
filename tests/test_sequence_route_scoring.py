"""Tests for v0.3.8 DEF-01 full-sequence logit route scoring.

Verifies that ``score_route_batch_sequence_from_logits`` resolves multi-token
route labels (e.g. Qwen2.5 ``"SYMBOLIC"`` → ``[" SY", "MBOL", "IC"]``) via the
full sequence log-probability contrast

    Score(Label | x) = sum_j log P(t_j | x, t_<j)

without falling back to autoregressive generation, and that the margin
``Score(SYMBOLIC|x) - Score(LLM|x)`` drives the route decision.

The pure helpers (``resolve_route_label_token_sequences``,
``route_action_from_sequence_scores``) are tested with a fake tokenizer that
emulates multi-token BPE behaviour. The full forward-pass path is tested with
the tiny fake model used by the steering-decay suite.
"""
from __future__ import annotations

import pytest

from daph_learning.routing.logit_router import score_route_batch_sequence_from_logits
from daph_learning.routing.steered_router import (
    resolve_route_label_token_sequences,
    route_action_from_sequence_scores,
)


# ---------------------------------------------------------------------------
# Fake tokenizer emulating multi-token BPE labels (Qwen2.5-style).
# ---------------------------------------------------------------------------

class _FakeMultiTokenTokenizer:
    """A toy tokenizer where 'SYMBOLIC' → 3 ids and 'LLM' → 1 id.

    Vocabulary is intentionally small but distinct for each sub-word so the
    sequence log-prob contrast is well-defined.
    """

    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<pad>"
    padding_side = "left"

    # vocab: 0=<pad>, 1=<bos>, 2="ACTION:", 3=":", 4=" SY", 5="MBOL", 6="IC",
    # 7=" LLM", 8="x", 9=" "
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(1)  # <bos>
        if text.endswith("ACTION: SYMBOLIC"):
            ids.extend([2, 3, 4, 5, 6])
        elif text.endswith("ACTION: LLM"):
            ids.extend([2, 3, 7])
        elif text.endswith("ACTION:"):
            ids.extend([2, 3])
        elif text.endswith("ACTION:SYMBOLIC"):
            ids.extend([2, 3, 4, 5, 6])
        elif text.endswith("ACTION:LLM"):
            ids.extend([2, 3, 7])
        else:
            # generic: tokenize whitespace-separated chunks
            for tok in text.split():
                ids.append(8)
        return ids

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True):
        # Not used by the sequence scorer (it uses encode + manual padding).
        if isinstance(texts, str):
            ids = self.encode(texts, add_special_tokens=add_special_tokens)
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        batch = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        return {"input_ids": batch, "attention_mask": [[1] * len(b) for b in batch]}


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

def test_resolve_route_label_token_sequences_multi_token():
    tok = _FakeMultiTokenTokenizer()
    prompt = "Choose backend.\nACTION:"
    sym_tail, llm_tail, used_space = resolve_route_label_token_sequences(
        tok, prompt, leading_space=True,
    )
    assert sym_tail == [4, 5, 6], f"SYMBOLIC tail={sym_tail}"
    assert llm_tail == [7], f"LLM tail={llm_tail}"
    assert used_space is True


def test_resolve_route_label_token_sequences_distinct_tails_required():
    tok = _FakeMultiTokenTokenizer()
    prompt = "Choose backend.\nACTION:"
    sym_tail, llm_tail, _ = resolve_route_label_token_sequences(tok, prompt)
    assert sym_tail != llm_tail, "tails must be distinct"


def test_resolve_route_label_token_sequences_empty_prompt_raises():
    tok = _FakeMultiTokenTokenizer()
    with pytest.raises(ValueError, match="non-empty rendered prompt"):
        resolve_route_label_token_sequences(tok, "")


def test_route_action_from_sequence_scores_symbolic_wins():
    action, margin = route_action_from_sequence_scores(-1.0, -5.0)
    assert action == "symbolic"
    assert margin == pytest.approx(4.0)


def test_route_action_from_sequence_scores_llm_wins():
    action, margin = route_action_from_sequence_scores(-5.0, -1.0)
    assert action == "llm"
    assert margin == pytest.approx(-4.0)


def test_route_action_from_sequence_scores_threshold():
    # margin = 0.5, threshold = 1.0 → llm (margin not > threshold)
    action, margin = route_action_from_sequence_scores(-2.0, -2.5, threshold=1.0)
    assert action == "llm"
    assert margin == pytest.approx(0.5)
    # margin = 0.5, threshold = 0.0 → symbolic
    action, _ = route_action_from_sequence_scores(-2.0, -2.5, threshold=0.0)
    assert action == "symbolic"


# ---------------------------------------------------------------------------
# Full forward-pass integration with the tiny fake model
# ---------------------------------------------------------------------------

def _make_fake_model(hidden=8, num_layers=2, vocab=16):
    import torch
    import torch.nn as nn

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden, hidden)
            self.linear.weight.data = torch.eye(hidden)
            self.linear.bias.data = torch.zeros(hidden)

        def forward(self, x):
            return (self.linear(x),)

    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            inner = FakeInner()
            self.model = inner
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            # Deterministic head: logit(v) = dot(embed(v), embed(token))
            self.lm_head.weight.data = torch.randn(vocab, hidden)

            class _Cfg:
                hidden_size = hidden
                num_hidden_layers = num_layers
                _name_or_path = "fake"

            self.config = _Cfg()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return type("O", (), {"logits": self.lm_head(x)})()

    return FakeModel()


class _FakeModelTokenizer:
    """Tokenizer whose vocab aligns with the fake model (vocab=16)."""

    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<pad>"
    padding_side = "left"

    # 0=pad, 1=bos, 2="ACTION:", 3=":", 4=" SY", 5="MBOL", 6="IC", 7=" LLM"
    def encode(self, text, add_special_tokens=True):
        ids: list[int] = []
        if add_special_tokens:
            ids.append(1)
        if text.endswith("ACTION: SYMBOLIC"):
            ids.extend([2, 3, 4, 5, 6])
        elif text.endswith("ACTION: LLM"):
            ids.extend([2, 3, 7])
        elif text.endswith("ACTION:"):
            ids.extend([2, 3])
        else:
            ids.extend([8] * len(text.split()))
        return ids

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True):
        if isinstance(texts, str):
            ids = self.encode(texts, add_special_tokens=add_special_tokens)
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        batch = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        return {"input_ids": batch, "attention_mask": [[1] * len(b) for b in batch]}


def test_score_route_batch_sequence_returns_one_decision_per_prompt():
    model = _make_fake_model()
    tok = _FakeModelTokenizer()
    prompts = ["Pick backend.\nACTION:", "Pick backend.\nACTION:"]
    results = score_route_batch_sequence_from_logits(prompts, model, tok)
    assert len(results) == 2
    for action, margin in results:
        assert action in ("symbolic", "llm")
        assert isinstance(margin, float)


def test_score_route_batch_sequence_multi_token_no_fallback_required():
    """The whole point of DEF-01: SYMBOLIC → 3 tokens must NOT raise and
    must NOT require autoregressive generation. The sequence scorer handles
    it with two teacher-forced forward passes."""
    model = _make_fake_model()
    tok = _FakeModelTokenizer()
    prompts = ["Pick backend.\nACTION:"]
    results = score_route_batch_sequence_from_logits(prompts, model, tok)
    assert len(results) == 1
    action, margin = results[0]
    # margin = Score(SYMBOLIC) - Score(LLM); both are sums of log-probs.
    # We only assert it is finite — the fake model has no semantic bias.
    assert isinstance(margin, float)


def test_score_route_batch_sequence_is_deterministic():
    import torch
    torch.manual_seed(123)
    model = _make_fake_model()
    tok = _FakeModelTokenizer()
    prompts = ["Pick backend.\nACTION:"]
    r1 = score_route_batch_sequence_from_logits(prompts, model, tok)
    r2 = score_route_batch_sequence_from_logits(prompts, model, tok)
    assert r1[0][0] == r2[0][0]
    assert r1[0][1] == pytest.approx(r2[0][1], abs=1e-5)


def test_score_route_batch_sequence_empty_prompts_raises():
    model = _make_fake_model()
    tok = _FakeModelTokenizer()
    with pytest.raises(ValueError, match="at least one rendered prompt"):
        score_route_batch_sequence_from_logits([], model, tok)


def test_score_route_batch_sequence_threshold_flips_decision():
    model = _make_fake_model()
    tok = _FakeModelTokenizer()
    prompts = ["Pick backend.\nACTION:"]
    # Compute the natural margin first (threshold=0).
    _, natural_margin = score_route_batch_sequence_from_logits(prompts, model, tok)[0]
    # Decision rule: action == "symbolic" iff margin > threshold.
    # threshold above margin → "llm"
    action_high, _ = score_route_batch_sequence_from_logits(
        prompts, model, tok, threshold=natural_margin + 1.0
    )[0]
    assert action_high == "llm"
    # threshold below margin → "symbolic"
    action_low, _ = score_route_batch_sequence_from_logits(
        prompts, model, tok, threshold=natural_margin - 1.0
    )[0]
    assert action_low == "symbolic"
