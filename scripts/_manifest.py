"""Backward-compat shim: re-export manifest helpers from the library layer.

V037-002 moved the manifest emission logic from ``scripts/_manifest.py`` into
``daph_learning.experiments.manifest`` so the library no longer depends on
the CLI layer. This file remains so existing ``scripts/*.py`` and ``tests/``
imports (``from scripts._manifest import emit_manifest, ...``) keep working,
but it now simply re-exports the canonical implementation.

New code should import from :mod:`daph_learning.experiments.manifest` directly.
"""

from __future__ import annotations

from daph_learning.experiments.manifest import (
    GitInfo,
    MANIFEST_VERSION,
    _chat_template_hash,
    _config_hash,
    _count_jsonl_lines,
    _dataset_entry,
    _detect_environment,
    _detect_git,
    _detect_model_info,
    _detect_tokenizer_info,
    _dtype_name,
    _output_entry,
    _vector_entry_from_steering_vector,
    emit_manifest,
    enrich_model_info_from_loaded_model,
    enrich_tokenizer_info_from_loaded_tokenizer,
    manifest_reference_line,
    manifest_sha256,
)

__all__ = [
    "GitInfo",
    "MANIFEST_VERSION",
    "_chat_template_hash",
    "_config_hash",
    "_count_jsonl_lines",
    "_dataset_entry",
    "_detect_environment",
    "_detect_git",
    "_detect_model_info",
    "_detect_tokenizer_info",
    "_dtype_name",
    "_output_entry",
    "_vector_entry_from_steering_vector",
    "emit_manifest",
    "enrich_model_info_from_loaded_model",
    "enrich_tokenizer_info_from_loaded_tokenizer",
    "manifest_reference_line",
    "manifest_sha256",
]
