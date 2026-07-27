"""Run manifest construction, hashing, and validation.

Implements the schema documented in ``docs/RUN_MANIFEST.md``. A run manifest
captures everything required to re-interpret an experimental output file at
activation-level resolution. See ``CLAIMS.md`` for why this matters: an
output file without a manifest is anecdote, not evidence.

The module is deliberately dependency-free (stdlib + hashlib only) so it can
run in any environment that imports ``daph_learning``, including ones
without torch or transformers installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MANIFEST_VERSION = "daph.run.v1"

# Splits that are touched exactly once after configuration freeze.
TEST_SPLITS = frozenset({"test", "final_test"})
# Splits used for vector capture.
CAPTURE_SPLITS = frozenset({"train", "capture"})
# All legal split values.
ALL_SPLITS = CAPTURE_SPLITS | TEST_SPLITS | frozenset(
    {"dev", "validation_alpha", "validation_threshold"}
)

REQUIRED_TOP_LEVEL = ("manifest_version", "run_id", "daph_version")
REQUIRED_MODEL = ("repo",)
REQUIRED_TOKENIZER = ("repo",)
REQUIRED_DATASET = ("path", "sha256", "split")
REQUIRED_DECODING = ("seed", "prompt_format", "route_token_resolver")
REQUIRED_VECTOR = (
    "path",
    "vector_id",
    "family",
    "layer",
    "alpha",
    "token_scope",
)
REQUIRED_FOR_HEADLINE_VECTOR = (
    "vector_sha256",
    "capture_dataset_sha256",
    "extraction_method",
    "capture_anchor",
    "capture_prompt_format",
)
REQUIRED_FOR_HEADLINE_MODEL = ("revision", "config_hash", "dtype")
REQUIRED_FOR_HEADLINE_TOKENIZER = ("revision", "chat_template_hash")
# V037-005: expanded environment provenance requirements. A headline
# manifest must record the installed versions of the packages that affect
# reproducibility, plus the CUDA/GPU info if CUDA is available. The
# accelerate/peft/safetensors fields are REQUIRED_FOR_HEADLINE only when
# the package is installed (the validator checks non-None when present in
# the environment dict); torch and transformers are always required.
REQUIRED_FOR_HEADLINE_ENV = ("torch_version", "transformers_version")

LEGAL_PROMPT_FORMATS = frozenset({"raw", "chat"})
LEGAL_ROUTE_TOKEN_RESOLVERS = frozenset({"isolated", "contextual"})
LEGAL_TOKEN_SCOPES = frozenset({"last", "all"})
LEGAL_VECTOR_FAMILIES = frozenset({"tool_policy", "reasoning_policy"})
LEGAL_LABEL_ORACLE_KINDS = frozenset(
    {"capability", "accuracy", "utility", "policy_heuristic", None}
)


class ManifestValidationError(ValueError):
    """Raised when a manifest fails validation. Aggregated messages joined by '; '."""


@dataclass
class VectorEntry:
    path: str
    vector_id: str
    family: str
    layer: int
    alpha: float
    token_scope: str
    # Optional-but-recommended provenance fields. Defaults preserve "null not
    # omitted" semantics from RUN_MANIFEST.md so headline validation can flag
    # missing provenance rather than silently passing.
    hidden_size: int | None = None
    model_id: str | None = None
    behavior: str | None = None
    anchor: str | None = None
    extraction_method: str | None = None
    normalization: str | None = None
    positive_n: int | None = None
    negative_n: int | None = None
    capture_anchor: str | None = None
    capture_prompt_format: str | None = None
    capture_dataset_sha256: str | None = None
    vector_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutputEntry:
    path: str
    kind: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """A run manifest. Construct via :func:`build_manifest` for normal use.

    All nested sub-objects are plain dicts in the public API to keep the
    surface small and to make serialization trivial. Use :func:`validate`
    to check completeness.
    """

    run_id: str
    daph_version: str
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    dataset: dict[str, Any]
    decoding: dict[str, Any]
    outputs: list[OutputEntry]
    environment: dict[str, Any] = field(default_factory=dict)
    vectors: list[VectorEntry] = field(default_factory=list)
    pytorch_determinism: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.run_id,
            "daph_version": self.daph_version,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dataset": self.dataset,
            "decoding": self.decoding,
            "environment": self.environment,
            "vectors": [v.to_dict() for v in self.vectors],
            "pytorch_determinism": self.pytorch_determinism,
            "outputs": [o.to_dict() for o in self.outputs],
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "created_at": self.created_at,
        }
        return payload


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes. Used for dataset and output hashing."""
    p = Path(path)
    return _sha256_hex(p.read_bytes())


def hash_vector_values(values: Sequence[float]) -> str:
    """SHA-256 of a steering vector's values array.

    The hash is taken over the float32 little-endian byte representation so
    that two arrays with the same values but different in-memory dtypes hash
    identically. This matches the storage convention in
    ``daph_learning.steering.io.save_vector`` (float32).
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    return _sha256_hex(arr.tobytes())


def build_manifest(
    *,
    run_id: str,
    daph_version: str,
    model: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    dataset: Mapping[str, Any],
    decoding: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any] | OutputEntry],
    environment: Mapping[str, Any] | None = None,
    vectors: Sequence[Mapping[str, Any] | VectorEntry] | None = None,
    pytorch_determinism: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    created_at: str | None = None,
) -> RunManifest:
    """Construct a :class:`RunManifest` from caller-supplied fields.

    The caller is responsible for filling in provenance fields it actually
    knows. Fields the caller cannot fill should be passed as ``None`` so
    that :func:`validate` can flag them rather than silently passing.
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_entries: list[OutputEntry] = []
    for o in outputs:
        if isinstance(o, OutputEntry):
            out_entries.append(o)
        else:
            out_entries.append(
                OutputEntry(path=str(o["path"]), kind=str(o["kind"]), sha256=str(o["sha256"]))
            )

    vec_entries: list[VectorEntry] = []
    for v in vectors or ():
        if isinstance(v, VectorEntry):
            vec_entries.append(v)
        else:
            # Accept a partial dict; missing provenance fields default to None.
            vec_entries.append(
                VectorEntry(
                    path=str(v["path"]),
                    vector_id=str(v["vector_id"]),
                    family=str(v["family"]),
                    layer=int(v["layer"]),
                    alpha=float(v["alpha"]),
                    token_scope=str(v["token_scope"]),
                    hidden_size=v.get("hidden_size"),
                    model_id=v.get("model_id"),
                    behavior=v.get("behavior"),
                    anchor=v.get("anchor"),
                    extraction_method=v.get("extraction_method"),
                    normalization=v.get("normalization"),
                    positive_n=v.get("positive_n"),
                    negative_n=v.get("negative_n"),
                    capture_anchor=v.get("capture_anchor"),
                    capture_prompt_format=v.get("capture_prompt_format"),
                    capture_dataset_sha256=v.get("capture_dataset_sha256"),
                    vector_sha256=v.get("vector_sha256"),
                )
            )

    return RunManifest(
        run_id=run_id,
        daph_version=daph_version,
        model=dict(model),
        tokenizer=dict(tokenizer),
        dataset=dict(dataset),
        decoding=dict(decoding),
        outputs=out_entries,
        environment=dict(environment or {}),
        vectors=vec_entries,
        pytorch_determinism=dict(pytorch_determinism or {}),
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=created_at,
    )


def _check_required(
    errors: list[str],
    obj: Mapping[str, Any],
    required: Iterable[str],
    path: str,
) -> None:
    for key in required:
        if key not in obj or obj[key] is None or obj[key] == "":
            errors.append(f"{path}.{key} is required")


def validate(manifest: RunManifest | Mapping[str, Any], *, headline: bool = False) -> None:
    """Validate a manifest.

    If ``headline`` is False, only REQUIRED fields are checked (the manifest
    is usable for engineering/regression outputs). If ``headline`` is True,
    REQUIRED_FOR_HEADLINE fields are also checked (the manifest is usable
    for scientific headline results).

    Raises :class:`ManifestValidationError` with all violations aggregated
    on a single message.
    """
    if isinstance(manifest, RunManifest):
        payload = manifest.to_dict()
    else:
        payload = dict(manifest)

    errors: list[str] = []

    _check_required(errors, payload, REQUIRED_TOP_LEVEL, "manifest")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        errors.append(
            f"manifest_version must be {MANIFEST_VERSION!r}, got {payload.get('manifest_version')!r}"
        )

    model = payload.get("model") or {}
    tokenizer = payload.get("tokenizer") or {}
    dataset = payload.get("dataset") or {}
    decoding = payload.get("decoding") or {}
    environment = payload.get("environment") or {}
    vectors = payload.get("vectors") or []
    outputs = payload.get("outputs") or []

    _check_required(errors, model, REQUIRED_MODEL, "model")
    _check_required(errors, tokenizer, REQUIRED_TOKENIZER, "tokenizer")
    _check_required(errors, dataset, REQUIRED_DATASET, "dataset")
    _check_required(errors, decoding, REQUIRED_DECODING, "decoding")

    if not outputs:
        errors.append("manifest.outputs must contain at least one entry")
    else:
        for i, o in enumerate(outputs):
            for key in ("path", "kind", "sha256"):
                if not o.get(key):
                    errors.append(f"outputs[{i}].{key} is required")

    # Enum / format checks.
    split = dataset.get("split")
    if split is not None and split not in ALL_SPLITS:
        errors.append(
            f"dataset.split {split!r} is not one of {sorted(ALL_SPLITS)}"
        )

    prompt_format = decoding.get("prompt_format")
    if prompt_format is not None and prompt_format not in LEGAL_PROMPT_FORMATS:
        errors.append(
            f"decoding.prompt_format {prompt_format!r} is not one of {sorted(LEGAL_PROMPT_FORMATS)}"
        )

    resolver = decoding.get("route_token_resolver")
    if resolver is not None and resolver not in LEGAL_ROUTE_TOKEN_RESOLVERS:
        errors.append(
            f"decoding.route_token_resolver {resolver!r} is not one of "
            f"{sorted(LEGAL_ROUTE_TOKEN_RESOLVERS)}"
        )

    label_oracle = dataset.get("label_oracle_kind")
    if label_oracle not in LEGAL_LABEL_ORACLE_KINDS:
        errors.append(
            f"dataset.label_oracle_kind {label_oracle!r} is not one of "
            f"{sorted(v for v in LEGAL_LABEL_ORACLE_KINDS if v is not None)} or null"
        )

    # Anti-circularity: a test-split manifest must not have a vector captured
    # on the same dataset SHA-256. This is the programmatic guard against the
    # most common split-leakage failure mode.
    if split in TEST_SPLITS and dataset.get("sha256"):
        ds_sha = dataset["sha256"]
        for i, v in enumerate(vectors):
            cap = v.get("capture_dataset_sha256")
            if cap is not None and cap == ds_sha:
                errors.append(
                    f"vectors[{i}].capture_dataset_sha256 equals dataset.sha256 "
                    f"on a {split!r} split — this is a split-leakage violation"
                )

    # Per-vector required + enum checks.
    for i, v in enumerate(vectors):
        _check_required(errors, v, REQUIRED_VECTOR, f"vectors[{i}]")
        family = v.get("family")
        if family is not None and family not in LEGAL_VECTOR_FAMILIES:
            errors.append(
                f"vectors[{i}].family {family!r} is not one of "
                f"{sorted(LEGAL_VECTOR_FAMILIES)}"
            )
        scope = v.get("token_scope")
        if scope is not None and scope not in LEGAL_TOKEN_SCOPES:
            errors.append(
                f"vectors[{i}].token_scope {scope!r} is not one of "
                f"{sorted(LEGAL_TOKEN_SCOPES)}"
            )

    if headline:
        _check_required(errors, model, REQUIRED_FOR_HEADLINE_MODEL, "model")
        _check_required(errors, tokenizer, REQUIRED_FOR_HEADLINE_TOKENIZER, "tokenizer")
        _check_required(errors, environment, REQUIRED_FOR_HEADLINE_ENV, "environment")
        if not payload.get("git_commit"):
            errors.append("manifest.git_commit is required for headline results")
        if not vectors:
            errors.append(
                "manifest.vectors must be non-empty for headline steering results"
            )
        for i, v in enumerate(vectors):
            _check_required(
                errors,
                v,
                REQUIRED_FOR_HEADLINE_VECTOR,
                f"vectors[{i}]",
            )

    if errors:
        raise ManifestValidationError("; ".join(errors))


def serialize(manifest: RunManifest) -> str:
    """Serialize a manifest to canonical JSON (sorted keys, 2-space indent)."""
    return json.dumps(manifest.to_dict(), sort_keys=True, indent=2, separators=(",", ": "))


def manifest_sha256(manifest: RunManifest | Mapping[str, Any]) -> str:
    """SHA-256 of the canonical serialized manifest. Used to reference a
    manifest from an output file.

    The canonical form is the same one :func:`serialize` produces (sorted
    keys, 2-space indent). A plain dict payload is re-serialized the same
    way so that an on-disk manifest and an in-memory manifest hash
    identically.
    """
    if isinstance(manifest, RunManifest):
        return _sha256_hex(serialize(manifest).encode("utf-8"))
    canonical = json.dumps(dict(manifest), sort_keys=True, indent=2, separators=(",", ": "))
    return _sha256_hex(canonical.encode("utf-8"))


def write_manifest(manifest: RunManifest, path: str | Path) -> str:
    """Write the manifest to ``path`` and return its SHA-256.

    The recommended convention is to write a sibling
    ``<output>.manifest.json`` next to each output file. The returned SHA-256
    should be embedded in the output file's first record when the format
    permits, so that the output can be checked against the manifest it
    claims to have been produced under.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # The on-disk file gets a trailing newline for editor friendliness, but
    # the returned SHA is taken over the canonical form (no newline) so it
    # matches :func:`manifest_sha256` and can be embedded in output files.
    canonical = serialize(manifest)
    p.write_text(canonical + "\n", encoding="utf-8")
    return _sha256_hex(canonical.encode("utf-8"))


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest from disk as a plain dict. Use :func:`validate` to check it."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
