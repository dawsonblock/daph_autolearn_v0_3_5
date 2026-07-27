"""Environment provenance capture (V037-005).

Replaces the v0.3.6 best-effort ``_detect_environment`` (which only captured
torch/transformers versions via ``__version__``) with a structured provenance
collector that:

1. Uses :func:`importlib.metadata.version` for installed-package versions
   (torch, transformers, accelerate, peft, safetensors). This is the
   *installed* version, not the *imported* version — they can differ in
   editable installs, and the installed version is what matters for
   reproducibility.
2. Captures CUDA runtime + driver versions, GPU name, and compute capability
   when torch + CUDA are available.
3. Captures attention implementation, dtype, and device map from a loaded
   model (when supplied).
4. **Fails closed** for headline manifests: if a REQUIRED_FOR_HEADLINE_ENV
   field cannot be captured, the caller gets a structured error, not a
   silent ``None``.

The captured environment is a plain dict suitable for embedding in a
``daph.run.v1`` manifest's ``environment`` block.
"""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass, field
from typing import Any, Mapping

# Packages whose installed versions we capture via importlib.metadata.
# These are the packages that affect reproducibility of a steering experiment.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "numpy",
    "daph-learning",
)


@dataclass
class EnvironmentProvenance:
    """Structured environment provenance for a run manifest."""

    python_version: str
    os: str
    packages: dict[str, str | None] = field(default_factory=dict)
    torch_version: str | None = None
    transformers_version: str | None = None
    accelerate_version: str | None = None
    peft_version: str | None = None
    safetensors_version: str | None = None
    numpy_version: str | None = None
    daph_version: str | None = None
    cuda_version: str | None = None
    cuda_driver_version: str | None = None
    gpu_model: str | None = None
    gpu_compute_capability: str | None = None
    attention_implementation: str | None = None
    dtype: str | None = None
    device_map: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a manifest-compatible dict.

        The ``packages`` sub-dict is included so consumers can see the full
        installed-version snapshot, not just the top-level convenience
        fields.
        """
        return {
            "python_version": self.python_version,
            "os": self.os,
            "packages": dict(self.packages),
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "accelerate_version": self.accelerate_version,
            "peft_version": self.peft_version,
            "safetensors_version": self.safetensors_version,
            "numpy_version": self.numpy_version,
            "daph_version": self.daph_version,
            "cuda_version": self.cuda_version,
            "cuda_driver_version": self.cuda_driver_version,
            "gpu_model": self.gpu_model,
            "gpu_compute_capability": self.gpu_compute_capability,
            "attention_implementation": self.attention_implementation,
            "dtype": self.dtype,
            "device_map": self.device_map,
        }


def _installed_version(package: str) -> str | None:
    """Return the installed version of a package via importlib.metadata.

    Returns ``None`` if the package is not installed (not an error — many
    packages are optional dependencies).
    """
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_info() -> tuple[str | None, str | None, str | None, str | None]:
    """Return (cuda_version, cuda_driver_version, gpu_model, compute_capability).

    Each element is ``None`` if torch is not installed, CUDA is not
    available, or the specific query failed.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        return None, None, None, None

    if not torch.cuda.is_available():
        return None, None, None, None

    cuda_version = torch.version.cuda
    try:
        cuda_driver_version = torch.cuda.get_driver_version()
        if cuda_driver_version is not None:
            cuda_driver_version = str(cuda_driver_version)
    except (RuntimeError, IndexError, AttributeError):
        cuda_driver_version = None

    try:
        gpu_model = torch.cuda.get_device_name(0)
    except (RuntimeError, IndexError, AttributeError):
        gpu_model = None

    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_compute_capability = f"{major}.{minor}"
    except (RuntimeError, IndexError, AttributeError):
        gpu_compute_capability = None

    return cuda_version, cuda_driver_version, gpu_model, gpu_compute_capability


def _model_info(model: Any) -> tuple[str | None, str | None, str | None]:
    """Return (attention_implementation, dtype, device_map) from a loaded model.

    Each element is ``None`` if the model doesn't expose the relevant
    attribute.
    """
    attention = None
    dtype = None
    device_map = None

    # Attention implementation: HF models store it in config._attn_implementation
    # (transformers >= 4.36) or config.attention_impl (older).
    cfg = getattr(model, "config", None)
    if cfg is not None:
        attention = (
            getattr(cfg, "_attn_implementation", None)
            or getattr(cfg, "attention_impl", None)
        )

    # Dtype: from the first parameter.
    try:
        dt = next(model.parameters()).dtype
        dtype = str(dt).replace("torch.", "")
    except (AttributeError, StopIteration, TypeError):
        pass

    # Device map: HF models store it as a dict or "auto" string.
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        device_map = str(device_map)
    elif device_map is not None:
        device_map = str(device_map)

    return attention, dtype, device_map


def capture_environment(
    *,
    model: Any | None = None,
    fail_closed: bool = False,
    required_for_headline: tuple[str, ...] = (
        "torch_version",
        "transformers_version",
    ),
) -> dict[str, Any]:
    """Capture structured environment provenance for a run manifest.

    Parameters
    ----------
    model : Any, optional
        A loaded HuggingFace model. If supplied, attention implementation,
        dtype, and device map are captured from it.
    fail_closed : bool
        If ``True``, raise :class:`EnvironmentProvenanceError` when any
        field in ``required_for_headline`` cannot be captured. This is the
        V037-005 fail-closed behavior for headline manifests.
    required_for_headline : tuple[str, ...]
        Fields that must be non-None when ``fail_closed`` is True.

    Returns
    -------
    dict[str, Any]
        A manifest-compatible environment dict.

    Raises
    ------
    EnvironmentProvenanceError
        If ``fail_closed`` is True and a required field is missing.
    """
    packages = {pkg: _installed_version(pkg) for pkg in _TRACKED_PACKAGES}
    cuda_version, cuda_driver_version, gpu_model, gpu_compute_capability = _cuda_info()

    if model is not None:
        attention, dtype, device_map = _model_info(model)
    else:
        attention = dtype = device_map = None

    prov = EnvironmentProvenance(
        python_version=platform.python_version(),
        os=f"{platform.system().lower()} {platform.release()}",
        packages=packages,
        torch_version=packages.get("torch"),
        transformers_version=packages.get("transformers"),
        accelerate_version=packages.get("accelerate"),
        peft_version=packages.get("peft"),
        safetensors_version=packages.get("safetensors"),
        numpy_version=packages.get("numpy"),
        daph_version=packages.get("daph-learning"),
        cuda_version=cuda_version,
        cuda_driver_version=cuda_driver_version,
        gpu_model=gpu_model,
        gpu_compute_capability=gpu_compute_capability,
        attention_implementation=attention,
        dtype=dtype,
        device_map=device_map,
    )

    env_dict = prov.to_dict()

    if fail_closed:
        missing = [
            f for f in required_for_headline if env_dict.get(f) is None
        ]
        if missing:
            raise EnvironmentProvenanceError(
                "fail-closed environment provenance capture failed; missing "
                f"required fields: {missing}"
            )

    return env_dict


class EnvironmentProvenanceError(ValueError):
    """Raised when fail-closed environment provenance capture is missing
    required fields."""


__all__ = [
    "EnvironmentProvenance",
    "EnvironmentProvenanceError",
    "capture_environment",
]
