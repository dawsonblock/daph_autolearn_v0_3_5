from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from daph_learning.routing.errors import (
    ContextBoundaryError,
    MultiTokenRouteError,
)

RouteAction = Literal["symbolic", "llm"]
_ACTION_RE = re.compile(r"^\s*(?:ACTION:\s*)?(SYMBOLIC|LLM)\b", re.IGNORECASE)
# v0.3.6: detect the ACTION anchor and any trailing whitespace so the
# contextual resolver can pick the correct leading_space hint without
# relying on the caller to pass it.
_ACTION_TERMINAL_RE = re.compile(r"ACTION:(?P<trailing>\s*)$")


def build_route_prompt(task: Mapping[str, Any]) -> str:
    """Build a minimal call/no-call decision prompt with a stable ACTION anchor."""
    caps = ", ".join(map(str, task.get("capability_ids", []))) or "unknown"
    specification = str(task.get("specification", ""))
    return (
        "Choose the execution backend for the task.\n"
        "Use SYMBOLIC when exact arithmetic execution should be delegated to the "
        "typed symbolic engine. Use LLM otherwise.\n"
        f"CAPABILITIES: {caps}\n"
        f"TASK: {specification}\n"
        "Return exactly one token: SYMBOLIC or LLM.\n"
        "ACTION:"
    )


def parse_route_action(text: str) -> RouteAction | None:
    match = _ACTION_RE.search(text)
    if not match:
        return None
    return "symbolic" if match.group(1).upper() == "SYMBOLIC" else "llm"


def detect_route_prompt_alignment(rendered_prompt: str) -> tuple[str, bool]:
    """v0.3.6 Phase 2.2: normalize the routing-prompt terminal boundary.

    Returns ``(aligned_prompt, leading_space)`` where:

    - ``aligned_prompt`` is the rendered prompt with its trailing
      whitespace after the ``ACTION:`` anchor trimmed to a canonical
      form (no trailing space). The label is then appended with a
      single leading space, which is the most portable boundary across
      BPE / Llama-BPE / SentencePiece tokenizers.
    - ``leading_space`` is always ``True``: the function normalizes the
      prompt so the label should always be appended with a leading
      space (e.g. ``" SYMBOLIC"``). This eliminates the ambiguity where
      ``"ACTION: " + "SYMBOLIC"`` can re-merge with the space depending
      on the tokenizer's BPE rules.

    Prompts that do not end with the ``ACTION:`` anchor are returned
    unchanged with ``leading_space=True`` (the conservative default that
    matches the historical routing-prompt template).
    """
    if not rendered_prompt:
        return rendered_prompt, True
    match = _ACTION_TERMINAL_RE.search(rendered_prompt)
    if not match:
        return rendered_prompt, True
    trailing = match.group("trailing")
    if trailing:
        # Trim the trailing whitespace so the boundary is the canonical
        # "ACTION:" + " LABEL" form. This eliminates the ambiguity where
        # "ACTION: " + "SYMBOLIC" can re-merge with the space depending
        # on the tokenizer's BPE rules.
        aligned = rendered_prompt[: match.start("trailing")] + rendered_prompt[match.end("trailing"):]
        return aligned, True
    # No trailing space: prompt already ends with "ACTION:". The label
    # should be appended with a leading space.
    return rendered_prompt, True


def route_action_from_logits(
    logits,
    symbolic_token_id: int,
    llm_token_id: int,
    *,
    threshold: float = 0.0,
) -> tuple[RouteAction, float]:
    """Resolve a route from a single next-token logit vector.

    `logits` may be rank-1 `[vocab]` or rank-2 `[sequence, vocab]`.
    Returns `(action, margin)` where
    `margin = logit(SYMBOLIC) - logit(LLM)`.
    """
    if getattr(logits, "ndim", None) == 2:
        logits = logits[-1]
    if getattr(logits, "ndim", None) != 1:
        raise ValueError("logits must have shape [vocab] or [sequence, vocab]")
    vocab_size = int(logits.shape[-1])
    for name, token_id in (
        ("symbolic_token_id", symbolic_token_id),
        ("llm_token_id", llm_token_id),
    ):
        if not isinstance(token_id, int) or not (0 <= token_id < vocab_size):
            raise ValueError(f"{name}={token_id!r} outside vocabulary size {vocab_size}")

    sym_logit = float(logits[symbolic_token_id])
    llm_logit = float(logits[llm_token_id])
    margin = sym_logit - llm_logit
    return ("symbolic" if margin > threshold else "llm"), margin



def resolve_route_token_ids(
    tokenizer,
    *,
    symbolic_label: str = "SYMBOLIC",
    llm_label: str = "LLM",
    leading_space: bool | None = None,
    allow_first_token_fallback: bool = False,
) -> tuple[int, int]:
    """Resolve route labels to distinct single next-token IDs.

    If ``leading_space`` is ``None``, both ``" SYMBOLIC"/" LLM"`` and
    ``"SYMBOLIC"/"LLM"`` are tested, in that order. This keeps direct-logit
    routing portable across tokenizer families while preserving the strict
    requirement that each candidate be represented by exactly one token.

    .. warning::
        This resolver tokenizes labels **in isolation** via
        ``tokenizer.encode(label)``. For some BPE/SentencePiece tokenizers,
        ``T(prefix + label) != T(prefix) + T(label)`` at the token boundary,
        so the resolved ID is not guaranteed to be the actual next token
        following the rendered routing prompt. See CLAIMS.md §9.

        For headline-eligible results, use
        :func:`resolve_route_token_ids_contextual` instead, which derives
        the continuation token from the actual rendered prompt.

    When ``allow_first_token_fallback`` is ``True`` (v0.3.6), a label that
    tokenizes to multiple sub-words (e.g. Qwen2.5's ``"SYMBOLIC"`` →
    ``["SYM", "BOL", "IC"]``) is reduced to its **first continuation token**.
    This licenses a first-token logit contrast (``logit[" SY"] -
    logit[" LL"]``) which is a valid routing signal even when the full label
    is not a single token. The fallback is opt-in so the strict default
    behaviour (raise on multi-token) is preserved for callers that require
    the full-label guarantee.
    """
    options = [leading_space] if leading_space is not None else [True, False]
    attempts: list[str] = []

    fallback_candidate: tuple[int, int] | None = None
    for try_space in options:
        prefix = " " if try_space else ""
        sym_ids = tokenizer.encode(prefix + symbolic_label, add_special_tokens=False)
        llm_ids = tokenizer.encode(prefix + llm_label, add_special_tokens=False)
        attempts.append(
            f"{prefix + symbolic_label!r}->{sym_ids}, {prefix + llm_label!r}->{llm_ids}"
        )
        if len(sym_ids) == 1 and len(llm_ids) == 1:
            sym_id, llm_id = int(sym_ids[0]), int(llm_ids[0])
            if sym_id != llm_id:
                return sym_id, llm_id
            continue
        # Record a first-token fallback candidate if both labels produce at
        # least one token and their first tokens differ. We keep iterating
        # in case a later leading_space option yields a clean single-token
        # resolution, which is always preferred over the fallback.
        if (
            allow_first_token_fallback
            and fallback_candidate is None
            and sym_ids
            and llm_ids
            and int(sym_ids[0]) != int(llm_ids[0])
        ):
            fallback_candidate = (int(sym_ids[0]), int(llm_ids[0]))

    if fallback_candidate is not None:
        return fallback_candidate

    raise MultiTokenRouteError(
        f"could not resolve {symbolic_label!r} and {llm_label!r} to distinct "
        f"single-token representations using leading_space options {options}; attempts: "
        + "; ".join(attempts)
    )


def resolve_route_token_ids_contextual(
    tokenizer,
    rendered_prompt: str,
    *,
    symbolic_label: str = "SYMBOLIC",
    llm_label: str = "LLM",
    leading_space: bool | None = None,
    allow_first_token_fallback: bool = False,
) -> tuple[int, int]:
    """Resolve route labels to single next-token IDs **in context**.

    Unlike :func:`resolve_route_token_ids`, which tokenizes labels in
    isolation, this function derives the continuation token by diffing
    ``T(rendered_prompt + label)`` against ``T(rendered_prompt)``. This
    eliminates the boundary assumption that fails for some BPE/SentencePiece
    tokenizers. See CLAIMS.md §9 and the audit §7.

    The rendered prompt is the fully-formatted routing prompt (after chat
    template application if applicable). The label is appended directly;
    callers should pass the leading-space variant that matches the prompt's
    terminal character (typically ``" SYMBOLIC"`` / ``" LLM"`` when the
    prompt ends with ``"ACTION:"`` and ``"SYMBOLIC"`` / ``"LLM"`` when it
    ends with ``"ACTION: "``).

    Returns ``(symbolic_token_id, llm_token_id)``. Raises ``ValueError``
    if either label does not produce exactly one continuation token, or if
    the two labels resolve to the same token id.

    When ``allow_first_token_fallback`` is ``True`` (v0.3.6), a multi-token
    continuation tail (e.g. Qwen2.5's ``" SYMBOLIC"`` → tail ``[" SY",
    "MBOL", "IC"]``) is reduced to its **first continuation token**. This
    enables a first-token logit contrast for tokenizers that never produce
    a single-token route label, which is the common case for instruct-tuned
    BPE models. The fallback is opt-in: the strict default (raise on
    multi-token) is preserved so existing callers and tests are unaffected.
    """
    options = [leading_space] if leading_space is not None else [True, False]
    attempts: list[str] = []

    if not rendered_prompt:
        raise ContextBoundaryError(
            "contextual resolver requires a non-empty rendered prompt to "
            "derive the continuation token"
        )

    base_ids = tokenizer.encode(rendered_prompt, add_special_tokens=True)
    base_len = len(base_ids)

    fallback_candidate: tuple[int, int] | None = None
    for try_space in options:
        prefix = " " if try_space else ""
        sym_full = rendered_prompt + prefix + symbolic_label
        llm_full = rendered_prompt + prefix + llm_label
        sym_ids = tokenizer.encode(sym_full, add_special_tokens=True)
        llm_ids = tokenizer.encode(llm_full, add_special_tokens=True)
        sym_tail = sym_ids[base_len:]
        llm_tail = llm_ids[base_len:]
        attempts.append(
            f"{prefix + symbolic_label!r}: tail={sym_tail}, "
            f"{prefix + llm_label!r}: tail={llm_tail}"
        )
        if len(sym_tail) == 1 and len(llm_tail) == 1:
            sym_id, llm_id = int(sym_tail[0]), int(llm_tail[0])
            if sym_id != llm_id:
                return sym_id, llm_id
            continue
        # Record a first-token fallback candidate. A clean single-token
        # resolution from a later leading_space option is always preferred.
        if (
            allow_first_token_fallback
            and fallback_candidate is None
            and sym_tail
            and llm_tail
            and int(sym_tail[0]) != int(llm_tail[0])
        ):
            fallback_candidate = (int(sym_tail[0]), int(llm_tail[0]))

    if fallback_candidate is not None:
        return fallback_candidate

    raise MultiTokenRouteError(
        f"could not resolve {symbolic_label!r} and {llm_label!r} to distinct "
        f"single continuation tokens after the rendered prompt; attempts: "
        + "; ".join(attempts)
    )
