"""Reusable task / model-loading utilities (V037-002).

Moved out of ``scripts/generate_v0_outputs.py`` so the library layer no
longer depends on the CLI layer. The ``scripts/`` CLIs now import from here.

Public names (the leading underscores are dropped now that these are part of
the public library API):

- :func:`format_for_model`
- :func:`load_llm`

Backward-compat aliases (``_format_for_model``, ``_load_llm``) are exported
so existing callers inside ``scripts/`` and ``tests/`` keep working unchanged.
"""

from __future__ import annotations

from typing import Any


def format_for_model(prompt: str, tokenizer, prompt_format: str) -> str:
    """Render ``prompt`` for the model according to ``prompt_format``.

    ``"raw"`` returns the prompt unchanged. ``"chat"`` applies the
    tokenizer's chat template (raising if none is set) with a single user
    turn and ``add_generation_prompt=True``.
    """
    if prompt_format == "raw":
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError("--prompt-format chat requested but tokenizer has no chat template")
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_llm(model_id: str):
    """Load a HuggingFace causal LM + tokenizer for routing/inference.

    dtype is bfloat16 on CUDA, float32 otherwise; ``device_map="auto"`` is
    used only when CUDA is available.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


# --- Backward-compat aliases ---

_format_for_model = format_for_model
_load_llm = load_llm


__all__ = [
    "format_for_model",
    "load_llm",
    # backward-compat aliases:
    "_format_for_model",
    "_load_llm",
]
