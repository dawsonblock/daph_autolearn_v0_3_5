"""Run-manifest emission helpers (V037-002).

Moved out of ``scripts/_manifest.py`` so the library layer no longer depends
on the CLI layer. ``scripts/_manifest.py`` now re-exports from here.

These helpers build, validate, and write a ``<output>.manifest.json`` sibling
next to a run output with consistent provenance (git, environment, model,
tokenizer, dataset, vectors, decoding). The helpers are best-effort: any
field that cannot be detected is left as ``None`` so that
:func:`daph_learning.evaluation.manifest.validate` can flag it rather than
silently passing. Vector capture provenance
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
    manifests should populate them explicitly via
    :func:`enrich_model_info_from_loaded_model`.

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


def _config_hash(model: Any) -> str | None:
    """Stable SHA-256 of the model's config dict (sorted keys).

    This is the v0.3.6 Phase 3.1 config_hash: it captures the model
    architecture (hidden_size, num_layers, num_heads, vocab_size,
    intermediate_size, max_position_embeddings, etc.) so two runs that
    load "the same" repo at different revisions can be distinguished
    even when the repo id matches.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    try:
        # to_dict is the HF PretrainedConfig API; sorted for stability.
        cfg_dict = cfg.to_dict()
    except Exception:
        return None
    return hashlib.sha256(
        json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _chat_template_hash(tokenizer: Any) -> str | None:
    """Stable SHA-256 of the tokenizer's chat template string.

    v0.3.6 Phase 3.1: a chat-template change silently alters routing
    decisions (the rendered prompt boundary moves), so the manifest must
    record the template hash to make such drift detectable across runs.
    Returns None when the tokenizer has no chat template.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _dtype_name(model: Any) -> str | None:
    """A stable string name for the model's parameter dtype."""
    try:
        dt = next(model.parameters()).dtype
        return str(dt).replace("torch.", "")
    except Exception:
        return None


def enrich_model_info_from_loaded_model(
    model_info: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    """v0.3.6 Phase 3.1: auto-fill headline-required model provenance
    fields (revision, config_hash, hidden_size, num_layers, dtype) from
    a loaded HuggingFace model.

    ``revision`` is read from ``model.config._commit_hash`` when present
    (HF sets this on local loads from the cache). ``config_hash`` is a
    SHA-256 of ``config.to_dict()``. ``hidden_size`` and ``num_layers``
    are read from the config (with GPT-2 fallbacks to ``n_embd`` /
    ``n_layer``). ``dtype`` is the dtype of the first parameter.

    The ``repo`` field is preserved from ``model_info`` (the caller
    already knows the repo id; the loaded model may not).
    """
    cfg = getattr(model, "config", None)
    out = dict(model_info)
    if cfg is not None:
        out["hidden_size"] = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
        )
        out["num_layers"] = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layer", None)
        )
        rev = getattr(cfg, "_commit_hash", None)
        if rev:
            out["revision"] = str(rev)
        out["config_hash"] = _config_hash(model)
    out["dtype"] = _dtype_name(model)
    return out


def enrich_tokenizer_info_from_loaded_tokenizer(
    tokenizer_info: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """v0.3.6 Phase 3.1: auto-fill headline-required tokenizer
    provenance fields (revision, chat_template_hash, pad_token_id,
    pad_side) from a loaded HuggingFace tokenizer.

    ``revision`` is read from ``tokenizer.vocab_files_paths`` when
    available, falling back to ``tokenizer.init_kwargs`` if present.
    ``chat_template_hash`` is a SHA-256 of the chat template string
    (None when no template is set). ``pad_token_id`` and ``pad_side``
    are read directly.
    """
    out = dict(tokenizer_info)
    out["chat_template_hash"] = _chat_template_hash(tokenizer)
    out["pad_token_id"] = getattr(tokenizer, "pad_token_id", None)
    out["pad_side"] = getattr(tokenizer, "padding_side", None)
    # Best-effort revision: HF sets _commit_hash on the tokenizer's
    # underlying tokenizer files when loaded from the cache.
    rev = getattr(tokenizer, "_commit_hash", None)
    if not rev:
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        rev = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
    if rev:
        out["revision"] = str(rev)
    return out


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
    loaded_model: Any | None = None,
    loaded_tokenizer: Any | None = None,
) -> tuple[str, RunManifest]:
    """Build, validate, and write a run manifest next to ``output_path``.

    Returns ``(sha256, manifest)``. The manifest is written to
    ``<output_path>.manifest.json``. Validation is run at the
    non-headline tier by default; pass ``validate_headline=True`` to also
    check the headline-required fields (this will currently fail for any
    run that uses steering vectors, because v0.3.5 vectors do not carry
    capture provenance — that is the intended behavior).

    v0.3.6 Phase 3.1: when ``loaded_model`` and ``loaded_tokenizer`` are
    supplied, the model and tokenizer info dicts are enriched with the
    headline-required provenance fields (``revision``, ``config_hash``,
    ``hidden_size``, ``num_layers``, ``dtype``, ``chat_template_hash``,
    ``pad_token_id``, ``pad_side``) that the v0.3.5 best-effort detector
    left as None. This makes a run manifest headline-eligible without
    the caller having to fill those fields manually.

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

    # v0.3.6 Phase 3.1: auto-fill headline model/tokenizer provenance
    # from the loaded objects when the caller passes them in.
    model_info = _detect_model_info(model_id)
    tokenizer_info = _detect_tokenizer_info(model_id)
    if loaded_model is not None:
        model_info = enrich_model_info_from_loaded_model(model_info, loaded_model)
    if loaded_tokenizer is not None:
        tokenizer_info = enrich_tokenizer_info_from_loaded_tokenizer(
            tokenizer_info, loaded_tokenizer
        )

    manifest = build_manifest(
        run_id=run_id,
        daph_version=DAPH_VERSION,
        model=model_info,
        tokenizer=tokenizer_info,
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
