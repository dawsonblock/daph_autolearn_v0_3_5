from __future__ import annotations

import re
from typing import Any, Literal, Mapping

RouteAction = Literal["symbolic", "llm"]
_ACTION_RE = re.compile(r"^\s*(?:ACTION:\s*)?(SYMBOLIC|LLM)\b", re.IGNORECASE)


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
    """
    options = [leading_space] if leading_space is not None else [True, False]
    attempts: list[str] = []

    for try_space in options:
        prefix = " " if try_space else ""
        sym_ids = tokenizer.encode(prefix + symbolic_label, add_special_tokens=False)
        llm_ids = tokenizer.encode(prefix + llm_label, add_special_tokens=False)
        attempts.append(
            f"{prefix + symbolic_label!r}->{sym_ids}, {prefix + llm_label!r}->{llm_ids}"
        )
        if len(sym_ids) != 1 or len(llm_ids) != 1:
            continue
        sym_id, llm_id = int(sym_ids[0]), int(llm_ids[0])
        if sym_id != llm_id:
            return sym_id, llm_id

    raise ValueError(
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
    """
    options = [leading_space] if leading_space is not None else [True, False]
    attempts: list[str] = []

    if not rendered_prompt:
        raise ValueError(
            "contextual resolver requires a non-empty rendered prompt to "
            "derive the continuation token"
        )

    base_ids = tokenizer.encode(rendered_prompt, add_special_tokens=True)
    base_len = len(base_ids)

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
        if len(sym_tail) != 1 or len(llm_tail) != 1:
            continue
        sym_id, llm_id = int(sym_tail[0]), int(llm_tail[0])
        if sym_id != llm_id:
            return sym_id, llm_id

    raise ValueError(
        f"could not resolve {symbolic_label!r} and {llm_label!r} to distinct "
        f"single continuation tokens after the rendered prompt; attempts: "
        + "; ".join(attempts)
    )
