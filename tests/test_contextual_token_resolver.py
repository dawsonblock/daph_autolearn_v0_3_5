from __future__ import annotations

import pytest

from daph_learning.routing.steered_router import (
    resolve_route_token_ids,
    resolve_route_token_ids_contextual,
)
from daph_learning.routing.logit_router import score_route_batch_from_logits


class _ContextAwareTokenizer:
    """A fake tokenizer that demonstrates the boundary problem.

    Isolated encode(" SYMBOLIC") returns [10] (one token).
    But in context after "ACTION:", the continuation is [10, 99] (two tokens)
    because the BPE merge rules differ at the boundary.

    This is exactly the failure mode the contextual resolver is designed to
    catch: the isolated resolver would return token 10, but the actual
    continuation token after the rendered prompt is 10 followed by 99, so
    the isolated resolver happens to be right here but for the wrong reason.

    A more realistic fake would have the isolated token differ from the
    contextual continuation token. See test_contextual_resolver_detects_boundary_difference.
    """

    def __init__(self, base_prompt: str = "ACTION:"):
        self.base_prompt = base_prompt
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        # Isolated label encoding
        if text == " SYMBOLIC":
            return [10]
        if text == " LLM":
            return [20]
        if text == "SYMBOLIC":
            return [10, 99]
        if text == "LLM":
            return [20, 88]
        # Full prompt encoding: base prompt + label
        if text == self.base_prompt:
            return [1, 2, 3]  # base prompt tokens
        if text == self.base_prompt + " SYMBOLIC":
            return [1, 2, 3, 10]  # continuation is just [10]
        if text == self.base_prompt + " LLM":
            return [1, 2, 3, 20]
        if text == self.base_prompt + "SYMBOLIC":
            return [1, 2, 3, 10, 99]
        if text == self.base_prompt + "LLM":
            return [1, 2, 3, 20, 88]
        return [1, 2, 3]  # default: base prompt

    def __call__(self, text, **kwargs):
        import torch
        if isinstance(text, list):
            encoded = [self.encode(t, add_special_tokens=True) for t in text]
            max_len = max(len(e) for e in encoded)
            padded = [[0] * (max_len - len(e)) + e for e in encoded]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
            }
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _BoundaryDifferenceTokenizer:
    """A fake tokenizer where isolated and contextual resolutions DIFFER.

    Isolated: encode(" SYMBOLIC") = [10]
    Contextual: encode("ACTION: SYMBOLIC") = [1,2,3, 42]  (not 10!)

    This is the case the contextual resolver is designed to fix: the
    isolated resolver returns 10, but the actual continuation token after
    the rendered prompt is 42. Using 10 would read the wrong logit.
    """

    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        if text == " SYMBOLIC":
            return [10]  # isolated: token 10
        if text == " LLM":
            return [20]
        if text == "SYMBOLIC":
            return [10]
        if text == "LLM":
            return [20]
        if text == "ACTION:":
            return [1, 2, 3]
        if text == "ACTION: SYMBOLIC":
            return [1, 2, 3, 42]  # contextual: token 42 (different!)
        if text == "ACTION: LLM":
            return [1, 2, 3, 20]
        return [1, 2, 3]

    def __call__(self, text, **kwargs):
        import torch
        if isinstance(text, list):
            encoded = [self.encode(t, add_special_tokens=True) for t in text]
            max_len = max(len(e) for e in encoded)
            padded = [[0] * (max_len - len(e)) + e for e in encoded]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
            }
        # Single-string call: return lists (not tensors) so anchor code works.
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _SingleTokenTokenizer:
    """The simple case: isolated and contextual agree."""
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        mapping = {" SYMBOLIC": [10], " LLM": [20], "SYMBOLIC": [10], "LLM": [20]}
        if text in mapping:
            return mapping[text]
        if text == "ACTION:":
            return [1, 2, 3]
        if text == "ACTION: SYMBOLIC":
            return [1, 2, 3, 10]
        if text == "ACTION: LLM":
            return [1, 2, 3, 20]
        return [1, 2, 3]

    def __call__(self, text, **kwargs):
        import torch
        if isinstance(text, list):
            encoded = [self.encode(t, add_special_tokens=True) for t in text]
            max_len = max(len(e) for e in encoded)
            padded = [[0] * (max_len - len(e)) + e for e in encoded]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
            }
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_contextual_resolver_returns_continuation_token():
    tok = _ContextAwareTokenizer("ACTION:")
    sym_id, llm_id = resolve_route_token_ids_contextual(tok, "ACTION:")
    assert sym_id == 10
    assert llm == 20 if False else True  # llm_id
    assert llm_id == 20


def test_contextual_resolver_detects_boundary_difference():
    """The key test: when isolated and contextual resolutions differ,
    the contextual resolver returns the actual continuation token (42),
    not the isolated token (10)."""
    tok = _BoundaryDifferenceTokenizer()
    # Isolated resolver returns 10 (wrong for this prompt)
    iso_sym, iso_llm = resolve_route_token_ids(tok)
    assert iso_sym == 10
    # Contextual resolver returns 42 (correct for this prompt)
    ctx_sym, ctx_llm = resolve_route_token_ids_contextual(tok, "ACTION:")
    assert ctx_sym == 42
    assert ctx_llm == 20
    # They differ — that's the whole point
    assert ctx_sym != iso_sym


def test_contextual_resolver_with_leading_space_false():
    """When the prompt ends with a space, leading_space=False is the
    natural continuation (no extra space before the label)."""
    tok = _SingleTokenTokenizer()  # isolated and contextual agree here
    # base "ACTION: " encodes to [1,2,3] (default branch in fake)
    # "ACTION: " + "SYMBOLIC" = "ACTION: SYMBOLIC" -> [1,2,3,10], tail=[10]
    # But our _SingleTokenTokenizer doesn't have "ACTION: " handling.
    # Use a custom tokenizer for this case.
    class SpaceSuffixTok:
        def __init__(self):
            self.pad_token_id = 0
            self.pad_token = "<pad>"
            self.padding_side = "left"
        def encode(self, text, add_special_tokens=False):
            if text == "ACTION: ":
                return [1, 2, 3]
            if text == "ACTION: SYMBOLIC":
                return [1, 2, 3, 10]
            if text == "ACTION: LLM":
                return [1, 2, 3, 20]
            return [1, 2, 3]
        def __call__(self, *a, **k): pass
    tok = SpaceSuffixTok()
    sym_id, llm_id = resolve_route_token_ids_contextual(
        tok, "ACTION: ", leading_space=False
    )
    assert sym_id == 10
    assert llm_id == 20


def test_contextual_resolver_rejects_multitoken_continuation():
    """If the continuation is more than one token, the resolver must reject."""
    tok = _ContextAwareTokenizer("ACTION:")
    # With leading_space=False, "ACTION:" + "SYMBOLIC" = [1,2,3,10,99]
    # tail = [10, 99] -> two tokens -> reject
    with pytest.raises(ValueError, match="single continuation token"):
        resolve_route_token_ids_contextual(tok, "ACTION:", leading_space=False)


def test_contextual_resolver_rejects_same_token_ids():
    class SameTokenTok:
        def encode(self, text, add_special_tokens=False):
            if text == "ACTION:":
                return [1, 2, 3]
            # Both labels continue with the same token
            return [1, 2, 3, 99]
        def __call__(self, *a, **k): pass
    tok = SameTokenTok()
    with pytest.raises(ValueError, match="distinct"):
        resolve_route_token_ids_contextual(tok, "ACTION:")


def test_isolated_resolver_still_works():
    """Backward compat: isolated resolver must still work as before."""
    tok = _SingleTokenTokenizer()
    sym_id, llm_id = resolve_route_token_ids(tok)
    assert sym_id == 10
    assert llm_id == 20


def test_score_route_batch_supports_contextual_resolver():
    """score_route_batch_from_logits must accept token_resolver='contextual'."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    tok = _BoundaryDifferenceTokenizer()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4})()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, **kwargs):
            # Return logits where token 42 (contextual SYMBOLIC) has higher logit
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 100), -10.0)
            logits[:, -1, 42] = 5.0  # contextual SYMBOLIC token
            logits[:, -1, 20] = 1.0  # LLM token
            return type("O", (), {"logits": logits})()

    model = FakeModel()
    # With isolated resolver, it would try token 10 (logit -10) vs 20 (logit 1) -> LLM
    # With contextual resolver, it uses token 42 (logit 5) vs 20 (logit 1) -> SYMBOLIC
    results = score_route_batch_from_logits(
        ["ACTION:"],
        model,
        tok,
        token_resolver="contextual",
    )
    assert len(results) == 1
    action, margin = results[0]
    assert action == "symbolic"
    assert margin > 0  # 42 has higher logit than 20


def test_score_route_batch_contextual_differs_from_isolated_when_boundary_differs():
    """When the boundary differs, contextual and isolated resolvers must
    produce different routing decisions. This is the integration-level
    proof that the contextual resolver matters."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    tok = _BoundaryDifferenceTokenizer()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4})()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, **kwargs):
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 100), -10.0)
            # Isolated SYMBOLIC token (10) has low logit
            logits[:, -1, 10] = -5.0
            # Contextual SYMBOLIC token (42) has high logit
            logits[:, -1, 42] = 5.0
            # LLM token (20) has medium logit
            logits[:, -1, 20] = 1.0
            return type("O", (), {"logits": logits})()

    model = FakeModel()
    # Isolated: token 10 (logit -5) vs 20 (logit 1) -> LLM (margin -6)
    iso_results = score_route_batch_from_logits(
        ["ACTION:"], model, tok, token_resolver="isolated"
    )
    assert iso_results[0][0] == "llm"

    # Contextual: token 42 (logit 5) vs 20 (logit 1) -> SYMBOLIC (margin 4)
    ctx_results = score_route_batch_from_logits(
        ["ACTION:"], model, tok, token_resolver="contextual"
    )
    assert ctx_results[0][0] == "symbolic"


def test_score_route_batch_rejects_unknown_resolver():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    tok = _SingleTokenTokenizer()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4})()
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 100)
            return type("O", (), {"logits": logits})()

    model = FakeModel()
    with pytest.raises(ValueError, match="token_resolver"):
        score_route_batch_from_logits(
            ["ACTION:"], model, tok, token_resolver="magic"
        )


def test_contextual_resolver_requires_prompts():
    """Contextual resolver needs a non-empty rendered prompt to tokenize."""
    tok = _SingleTokenTokenizer()
    with pytest.raises(ValueError, match="non-empty rendered prompt"):
        resolve_route_token_ids_contextual(tok, "")


# --- v0.3.6 Phase 1.1: first-token fallback for multi-token labels ---

class _MultiTokenTokenizer:
    """A tokenizer where every route label tokenizes to multiple sub-words.

    This mirrors Qwen2.5's behaviour: ``" SYMBOLIC"`` → ``[10, 11, 12]`` and
    ``" LLM"`` → ``[20, 21]``. Neither the isolated nor the contextual
    resolver can find a single-token representation, so the strict path
    raises. The v0.3.6 first-token fallback reduces each label to its
    leading sub-word (10 vs 20) so a first-token logit contrast is still
    possible.
    """

    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        if text == " SYMBOLIC":
            return [10, 11, 12]
        if text == " LLM":
            return [20, 21]
        if text == "SYMBOLIC":
            return [10, 11, 12]
        if text == "LLM":
            return [20, 21]
        if text == "ACTION:":
            return [1, 2, 3]
        if text == "ACTION: SYMBOLIC":
            return [1, 2, 3, 10, 11, 12]
        if text == "ACTION: LLM":
            return [1, 2, 3, 20, 21]
        return [1, 2, 3]

    def __call__(self, text, **kwargs):
        import torch
        if isinstance(text, list):
            encoded = [self.encode(t, add_special_tokens=True) for t in text]
            max_len = max(len(e) for e in encoded)
            padded = [[0] * (max_len - len(e)) + e for e in encoded]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
            }
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_isolated_resolver_first_token_fallback_returns_first_subword():
    """allow_first_token_fallback reduces multi-token labels to their first
    sub-word instead of raising."""
    tok = _MultiTokenTokenizer()
    # Strict path raises (no single-token representation exists).
    with pytest.raises(ValueError, match="could not resolve"):
        resolve_route_token_ids(tok)
    # Fallback path returns the first sub-word of each label.
    sym_id, llm_id = resolve_route_token_ids(tok, allow_first_token_fallback=True)
    assert sym_id == 10
    assert llm_id == 20


def test_contextual_resolver_first_token_fallback_returns_first_continuation():
    """The contextual resolver's first-token fallback returns the first
    continuation token after the rendered prompt."""
    tok = _MultiTokenTokenizer()
    with pytest.raises(ValueError, match="could not resolve"):
        resolve_route_token_ids_contextual(tok, "ACTION:")
    sym_id, llm_id = resolve_route_token_ids_contextual(
        tok, "ACTION:", allow_first_token_fallback=True
    )
    # T("ACTION: SYMBOLIC") = [1,2,3,10,11,12], base=[1,2,3], tail=[10,11,12]
    # first continuation token = 10. Same for LLM → 20.
    assert sym_id == 10
    assert llm_id == 20


def test_first_token_fallback_prefers_single_token_when_available():
    """When one leading_space option yields a clean single-token resolution,
    the fallback is NOT used even if allow_first_token_fallback=True."""
    tok = _ContextAwareTokenizer("ACTION:")
    # _ContextAwareTokenizer: " SYMBOLIC" → [10] (single), "SYMBOLIC" → [10,99] (multi)
    # With leading_space=None, the " SYMBOLIC" option resolves cleanly to 10,
    # so the fallback must never be triggered.
    sym_id, llm_id = resolve_route_token_ids_contextual(
        tok, "ACTION:", allow_first_token_fallback=True
    )
    assert sym_id == 10
    assert llm_id == 20


def test_first_token_fallback_rejects_same_first_token():
    """If both labels share the same first sub-word, the fallback cannot
    produce a distinct contrast and must still raise."""
    class SameFirstTok:
        def __init__(self):
            self.pad_token_id = 0
            self.pad_token = "<pad>"
            self.padding_side = "left"
        def encode(self, text, add_special_tokens=False):
            if text in (" SYMBOLIC", " LLM", "SYMBOLIC", "LLM"):
                # Both labels start with the same sub-word 50.
                return [50, 60] if "SYM" in text else [50, 70]
            if text == "ACTION:":
                return [1, 2, 3]
            if text == "ACTION: SYMBOLIC":
                return [1, 2, 3, 50, 60]
            if text == "ACTION: LLM":
                return [1, 2, 3, 50, 70]
            return [1, 2, 3]
        def __call__(self, *a, **k):
            pass
    tok = SameFirstTok()
    with pytest.raises(ValueError, match="could not resolve"):
        resolve_route_token_ids_contextual(
            tok, "ACTION:", allow_first_token_fallback=True
        )


def test_score_route_batch_first_token_fallback_routes_on_first_subword():
    """score_route_batch_from_logits with allow_first_token_fallback=True
    must read the first-subword logits when full labels are multi-token."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    tok = _MultiTokenTokenizer()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4})()
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], 100), -10.0
            )
            # First sub-word of SYMBOLIC (10) has high logit; LLM (20) low.
            logits[:, -1, 10] = 5.0
            logits[:, -1, 20] = 1.0
            return type("O", (), {"logits": logits})()

    model = FakeModel()
    # Strict path raises because labels are multi-token.
    with pytest.raises(ValueError):
        score_route_batch_from_logits(
            ["ACTION:"], model, tok, token_resolver="contextual"
        )
    # Fallback path routes SYMBOLIC by reading the first-subword logit.
    results = score_route_batch_from_logits(
        ["ACTION:"],
        model,
        tok,
        token_resolver="contextual",
        allow_first_token_fallback=True,
    )
    assert results[0][0] == "symbolic"
    assert results[0][1] > 0


# --- v0.3.6 Phase 2.2: prompt terminal alignment + cross-family validation ---

from daph_learning.routing.steered_router import detect_route_prompt_alignment


def test_detect_alignment_trims_trailing_whitespace():
    """A prompt ending with 'ACTION: ' (trailing space) is trimmed to
    'ACTION:' and leading_space=True is returned (label appended as
    ' SYMBOLIC')."""
    aligned, leading_space = detect_route_prompt_alignment("blah\nACTION: ")
    assert aligned.endswith("ACTION:")
    assert not aligned.endswith("ACTION: ")
    assert leading_space is True


def test_detect_alignment_keeps_canonical_action_colon():
    """A prompt ending with 'ACTION:' (no trailing space) is unchanged
    and leading_space=True."""
    aligned, leading_space = detect_route_prompt_alignment("blah\nACTION:")
    assert aligned == "blah\nACTION:"
    assert leading_space is True


def test_detect_alignment_no_action_anchor_returns_unchanged():
    """A prompt without the ACTION: anchor is returned unchanged with the
    conservative leading_space=True default."""
    aligned, leading_space = detect_route_prompt_alignment("just a prompt")
    assert aligned == "just a prompt"
    assert leading_space is True


def test_detect_alignment_empty_prompt():
    aligned, leading_space = detect_route_prompt_alignment("")
    assert aligned == ""
    assert leading_space is True


def test_contextual_resolver_uses_alignment_when_no_leading_space_hint():
    """When the caller passes leading_space=None, score_route_batch uses
    detect_route_prompt_alignment to normalize a trailing-space prompt
    before resolving continuation tokens. This is the Phase 2.2
    integration: it eliminates the boundary ambiguity for tokenizers
    where 'ACTION: ' + 'SYMBOLIC' re-merges differently than
    'ACTION:' + ' SYMBOLIC'."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    # A tokenizer where the boundary matters: 'ACTION: SYMBOLIC' (with
    # the space inside the prompt) tokenizes to a different continuation
    # than 'ACTION:' + ' SYMBOLIC'.
    class BoundarySensitiveTok:
        def __init__(self):
            self.pad_token_id = 0
            self.pad_token = "<pad>"
            self.padding_side = "left"
        def encode(self, text, add_special_tokens=False):
            # Canonical aligned form: 'ACTION:' + ' SYMBOLIC' -> [1,2,3,10]
            if text == "ACTION:":
                return [1, 2, 3]
            if text == "ACTION: SYMBOLIC":
                return [1, 2, 3, 10]   # aligned form: clean continuation 10
            if text == "ACTION: LLM":
                return [1, 2, 3, 20]
            # Trailing-space form (pre-alignment): the space+label would
            # re-merge into a different token id 77.
            if text == "ACTION:  SYMBOLIC":
                return [1, 2, 3, 77]
            return [1, 2, 3]
        def __call__(self, text, **kwargs):
            import torch
            if isinstance(text, list):
                encoded = [self.encode(t, add_special_tokens=True) for t in text]
                max_len = max(len(e) for e in encoded)
                padded = [[0] * (max_len - len(e)) + e for e in encoded]
                return {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.ones(len(padded), max_len, dtype=torch.long),
                }
            ids = self.encode(text, add_special_tokens=True)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tok = BoundarySensitiveTok()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 4)
            self.config = type("C", (), {"hidden_size": 4})()
        def get_input_embeddings(self):
            return self.embed
        def forward(self, input_ids, **kwargs):
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 100), -10.0)
            logits[:, -1, 10] = 5.0   # aligned SYMBOLIC continuation
            logits[:, -1, 20] = 1.0
            logits[:, -1, 77] = -5.0  # the pre-alignment wrong token
            return type("O", (), {"logits": logits})()

    model = FakeModel()
    # Pass a trailing-space prompt with leading_space=None. The alignment
    # helper must trim it to 'ACTION:' so the resolver reads token 10
    # (logit 5) instead of 77 (logit -5).
    results = score_route_batch_from_logits(
        ["ACTION: "], model, tok, token_resolver="contextual",
    )
    assert results[0][0] == "symbolic"
    assert results[0][1] > 0


# --- Cross-tokenizer-family validation (Phase 2.2 requirement) ---
# Fake tokenizers representing the three families the plan calls out:
#   * BPE (Qwen2.5 / GPT-2): multi-token route labels, boundary-sensitive
#   * Llama-BPE (Llama-3): single-token with leading space
#   * SentencePiece (Mistral): single-token, no leading space variant

class _BpeFamilyTok:
    """Qwen2.5-style: 'SYMBOLIC' → [' SY','MBOL','IC'] (multi-token)."""
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"
    def encode(self, text, add_special_tokens=False):
        if text in (" SYMBOLIC", "SYMBOLIC"):
            return [10, 11, 12]
        if text in (" LLM", "LLM"):
            return [20, 21]
        if text == "ACTION:":
            return [1, 2, 3]
        if text == "ACTION: SYMBOLIC":
            return [1, 2, 3, 10, 11, 12]
        if text == "ACTION: LLM":
            return [1, 2, 3, 20, 21]
        return [1, 2, 3]
    def __call__(self, text, **kwargs):
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _LlamaBpeFamilyTok:
    """Llama-3-style: ' SYMBOLIC' → [10] (single token with leading space)."""
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"
    def encode(self, text, add_special_tokens=False):
        if text == " SYMBOLIC":
            return [10]
        if text == " LLM":
            return [20]
        if text in ("SYMBOLIC", "LLM"):
            return [10] if "SYM" in text else [20]   # also single-token without space
        if text == "ACTION:":
            return [1, 2, 3]
        if text == "ACTION: SYMBOLIC":
            return [1, 2, 3, 10]
        if text == "ACTION: LLM":
            return [1, 2, 3, 20]
        return [1, 2, 3]
    def __call__(self, text, **kwargs):
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _SentencePieceFamilyTok:
    """Mistral-style: 'SYMBOLIC' → [10] (single token, no leading space)."""
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.padding_side = "left"
    def encode(self, text, add_special_tokens=False):
        if text in (" SYMBOLIC", "SYMBOLIC"):
            return [10]
        if text in (" LLM", "LLM"):
            return [20]
        if text == "ACTION:":
            return [1, 2, 3]
        if text == "ACTION: SYMBOLIC":
            return [1, 2, 3, 10]
        if text == "ACTION: LLM":
            return [1, 2, 3, 20]
        return [1, 2, 3]
    def __call__(self, text, **kwargs):
        ids = self.encode(text, add_special_tokens=True)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.mark.parametrize("tok,allow_fallback,expected_sym,expected_llm", [
    (_BpeFamilyTok(), True, 10, 20),          # multi-token → first-token fallback
    (_LlamaBpeFamilyTok(), False, 10, 20),    # single-token, no fallback needed
    (_SentencePieceFamilyTok(), False, 10, 20),  # single-token, no fallback needed
])
def test_contextual_resolver_across_tokenizer_families(
    tok, allow_fallback, expected_sym, expected_llm
):
    """Phase 2.2 requirement: the contextual resolver must produce a
    distinct (sym, llm) pair across BPE (Qwen/GPT-2), Llama-BPE, and
    SentencePiece families. The BPE family requires the first-token
    fallback; the others resolve cleanly."""
    from daph_learning.routing.steered_router import resolve_route_token_ids_contextual
    sym_id, llm_id = resolve_route_token_ids_contextual(
        tok, "ACTION:", allow_first_token_fallback=allow_fallback,
    )
    assert sym_id == expected_sym
    assert llm_id == expected_llm
    assert sym_id != llm_id
