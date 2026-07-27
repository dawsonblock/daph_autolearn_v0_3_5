"""Shared helpers for emitting run manifests from CLI scripts.

This module is private to ``scripts`` (leading underscore) and exists so that
``generate_v0_outputs.py``, ``tune_steering.py``, and ``evaluate_routes.py``
can all emit a ``<output>.manifest.json`` sibling with consistent provenance
without duplicating the environment-detection and vector-encoding logic.

The helpers are best-effort: any field that cannot be detected is left as
``None`` so that :func:`daph_learning.evaluation.manifest.validate` can flag
it rather than silently passing. Vector capture provenance
(``capture_dataset_sha256``, ``capture_anchor``, ``capture_prompt_format``,
``extraction_method``, ``positive_n``, ``negative_n``) is read from
``SteeringSpec`` — the v0.3.5 manifest patch extended ``SteeringSpec`` to
carry these fields, and ``extract_steering_vector.py`` populates them from
CLI args. Older ``.npz`` files saved before the patch will load with
``None`` defaults, and headline manifest validation will flag the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from daph_learning import __version__ as DAPH_VERSION
from daph_learning.evaluation.manifest import (
    MANIFEST_VERSION,
    OutputEntry,
    RunManifest,
    VectorEntry,
    build_manifest,
    hash_file,
    hash_vector_values,
    manifest_sha256,
    validate,
    write_manifest,
)
from daph_learning.steering.types import SteeringVector


@dataclass
class GitInfo:
    commit: str | None
    dirty: bool | None


def _detect_git(repo_root: Path) -> GitInfo:
    """Best-effort git detection. Returns (None, None) if not a git repo."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        if not commit or len(commit) != 40:
            return GitInfo(commit=None, dirty=None)
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return GitInfo(commit=commit, dirty=bool(status))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return GitInfo(commit=None, dirty=None)


def _detect_environment() -> dict[str, Any]:
    """Best-effort environment detection. Never raises."""
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "os": f"{platform.system().lower()} {platform.release()}",
        "torch_version": None,
        "transformers_version": None,
        "cuda_version": None,
        "gpu_model": None,
    }
    try:
        import torch  # type: ignore[import-not-found]

        env["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            try:
                env["gpu_model"] = torch.cuda.get_device_name(0)
            except Exception:
                env["gpu_model"] = None
    except Exception:
        pass
    try:
        import transformers  # type: ignore[import-not-found]

        env["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return env


def _detect_model_info(model_id: str | None) -> dict[str, Any]:
    """Best-effort model info from a HuggingFace repo id.

    Revision / config_hash / dtype are not detectable without loading the
    model, which is too expensive to do twice. They are left as None so
    headline validation flags them; callers that want headline-eligible
    manifests should populate them explicitly.

    When ``model_id`` is None (e.g. a symbolic-only run), ``repo`` is set
    to the sentinel ``"none"`` so the manifest's REQUIRED validation passes
    while accurately recording that no model was used.
    """
    return {
        "repo": model_id if model_id is not None else "none",
        "revision": None,
        "config_hash": None,
        "hidden_size": None,
        "num_layers": None,
        "dtype": None,
        "device_map": None,
        "attention_impl": None,
    }


def _detect_tokenizer_info(model_id: str | None) -> dict[str, Any]:
    return {
        "repo": model_id if model_id is not None else "none",
        "revision": None,
        "chat_template_hash": None,
        "pad_token_id": None,
        "pad_side": None,
    }


def _vector_entry_from_steering_vector(
    vector: SteeringVector,
    *,
    alpha_override: float | None,
    path: str | None,
    token_scope: str,
) -> VectorEntry:
    """Build a :class:`VectorEntry` from a loaded ``SteeringVector``.

    Capture provenance is read from ``vector.spec`` (the v0.3.5 manifest
    patch extended ``SteeringSpec`` with ``extraction_method``,
    ``normalization``, ``positive_n``, ``negative_n``, ``capture_anchor``,
    ``capture_prompt_format``, ``capture_dataset_path``, and
    ``capture_dataset_sha256``). Older ``.npz`` files without these fields
    will load with ``None`` defaults, and headline manifest validation will
    flag the gap — which is the intended behavior.
    """
    alpha = (
        float(alpha_override)
        if alpha_override is not None
        else float(vector.spec.alpha)
    )
    return VectorEntry(
        path=path or "",
        vector_id=vector.spec.vector_id,
        family=vector.spec.family,
        layer=int(vector.spec.layer),
        alpha=alpha,
        token_scope=token_scope,
        hidden_size=int(vector.spec.hidden_size) if vector.spec.hidden_size else None,
        model_id=vector.spec.model_id,
        behavior=vector.spec.behavior,
        anchor=vector.spec.anchor,
        # Provenance read from spec:
        extraction_method=vector.spec.extraction_method,
        normalization=vector.spec.normalization,
        positive_n=vector.spec.positive_n,
        negative_n=vector.spec.negative_n,
        capture_anchor=vector.spec.capture_anchor,
        capture_prompt_format=vector.spec.capture_prompt_format,
        capture_dataset_sha256=vector.spec.capture_dataset_sha256,
        vector_sha256=hash_vector_values(vector.values),
    )


def _dataset_entry(
    path: str | Path,
    *,
    split: str,
    label_field: str | None,
    label_oracle_kind: str | None,
) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "sha256": hash_file(p) if p.exists() else None,
        "num_tasks": _count_jsonl_lines(p) if p.exists() else None,
        "split": split,
        "label_field": label_field,
        "label_oracle_kind": label_oracle_kind,
    }


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _output_entry(path: str | Path, *, kind: str) -> OutputEntry:
    p = Path(path)
    return OutputEntry(
        path=str(p),
        kind=kind,
        sha256=hash_file(p) if p.exists() else None,
    )


def emit_manifest(
    *,
    run_id: str,
    repo_root: Path,
    output_path: str | Path,
    output_kind: str,
    model_id: str | None,
    dataset_path: str | Path,
    dataset_split: str,
    label_field: str | None,
    label_oracle_kind: str | None,
    seed: int,
    prompt_format: str,
    route_decision_mode: str | None,
    route_token_resolver: str,
    batch_size: int | None,
    route_batch_size: int | None,
    max_new_tokens: int | None,
    extra_outputs: Sequence[tuple[str | Path, str]] = (),
    tool_vectors: Sequence[SteeringVector] = (),
    tool_vector_paths: Sequence[str | None] = (),
    tool_alphas: Sequence[float | None] = (),
    reasoning_vectors: Sequence[SteeringVector] = (),
    reasoning_vector_paths: Sequence[str | None] = (),
    reasoning_alphas: Sequence[float | None] = (),
    additional_environment: Mapping[str, Any] | None = None,
    additional_decoding: Mapping[str, Any] | None = None,
    validate_headline: bool = False,
) -> tuple[str, RunManifest]:
    """Build, validate, and write a run manifest next to ``output_path``.

    Returns ``(sha256, manifest)``. The manifest is written to
    ``<output_path>.manifest.json``. Validation is run at the
    non-headline tier by default; pass ``validate_headline=True`` to also
    check the headline-required fields (this will currently fail for any
    run that uses steering vectors, because v0.3.5 vectors do not carry
    capture provenance — that is the intended behavior).

    Validation failures raise
    :class:`daph_learning.evaluation.manifest.ManifestValidationError`.
    Scripts should catch this, print a warning naming the missing fields,
    and continue — the output file is still valid; only the manifest is
    flagged. Use ``validate_headline=False`` for engineering/regression
    outputs and ``validate_headline=True`` only when attempting a
    headline-eligible run.
    """
    git = _detect_git(repo_root)
    env = _detect_environment()
    if additional_environment:
        env.update(dict(additional_environment))

    outputs = [_output_entry(output_path, kind=output_kind)]
    for extra_path, extra_kind in extra_outputs:
        outputs.append(_output_entry(extra_path, kind=extra_kind))

    vector_entries: list[VectorEntry] = []
    for vector, path, alpha in zip(tool_vectors, tool_vector_paths, tool_alphas):
        vector_entries.append(
            _vector_entry_from_steering_vector(
                vector,
                alpha_override=alpha,
                path=str(path) if path else None,
                token_scope="last",
            )
        )
    for vector, path, alpha in zip(
        reasoning_vectors, reasoning_vector_paths, reasoning_alphas
    ):
        vector_entries.append(
            _vector_entry_from_steering_vector(
                vector,
                alpha_override=alpha,
                path=str(path) if path else None,
                token_scope="all",
            )
        )

    decoding: dict[str, Any] = {
        "seed": int(seed),
        "prompt_format": prompt_format,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "route_decision_mode": route_decision_mode,
        "route_token_resolver": route_token_resolver,
        "batch_size": batch_size,
        "route_batch_size": route_batch_size,
    }
    if additional_decoding:
        decoding.update(dict(additional_decoding))

    manifest = build_manifest(
        run_id=run_id,
        daph_version=DAPH_VERSION,
        model=_detect_model_info(model_id),
        tokenizer=_detect_tokenizer_info(model_id),
        dataset=_dataset_entry(
            dataset_path,
            split=dataset_split,
            label_field=label_field,
            label_oracle_kind=label_oracle_kind,
        ),
        decoding=decoding,
        outputs=outputs,
        environment=env,
        vectors=vector_entries,
        git_commit=git.commit,
        git_dirty=git.dirty,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    validate(manifest, headline=validate_headline)

    manifest_path = Path(str(output_path) + ".manifest.json")
    sha = write_manifest(manifest, manifest_path)
    return sha, manifest


def manifest_reference_line(sha: str, manifest_path: Path) -> str:
    """A one-line summary suitable for printing from a CLI script."""
    return f"manifest: {manifest_path} (sha256={sha})"


__all__ = [
    "emit_manifest",
    "manifest_reference_line",
    "MANIFEST_VERSION",
    "manifest_sha256",
]
