from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


VectorFamily = Literal["reasoning_policy", "tool_policy"]


@dataclass(frozen=True)
class SteeringSpec:
    """Specification for a steering vector.

    The first block of fields (vector_id..hidden_size) is the original
    v0.3.4 schema and is required for the vector to be loadable and
    installable as a forward hook.

    The second block (capture_*, extraction_method, normalization,
    positive_n, negative_n, capture_dataset_sha256) is **capture
    provenance**: metadata recording where and how the vector was
    extracted. These fields are optional for engineering use but
    **required for headline-eligible run manifests** — see
    ``docs/RUN_MANIFEST.md`` and ``CLAIMS.md``. Without them, a steering
    vector cannot be reproduced or falsified, and any result that uses
    it is anecdote, not evidence.

    All provenance fields default to ``None`` so that older ``.npz``
    files without them still load. The manifest validator flags the gap
    rather than silently passing.
    """

    vector_id: str
    family: VectorFamily
    behavior: str
    layer: int
    alpha: float = 1.0
    anchor: str = "ACTION"
    model_id: str | None = None
    hidden_size: int | None = None
    # --- capture provenance (v0.3.5 manifest patch) ---
    extraction_method: str | None = None
    normalization: str | None = None
    positive_n: int | None = None
    negative_n: int | None = None
    capture_anchor: str | None = None
    capture_prompt_format: str | None = None
    capture_dataset_path: str | None = None
    capture_dataset_sha256: str | None = None


@dataclass(frozen=True)
class SteeringVector:
    spec: SteeringSpec
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("steering vector must be rank-1")
        if self.spec.hidden_size is not None and self.values.size != self.spec.hidden_size:
            raise ValueError(
                f"vector dim {self.values.size} != declared hidden_size {self.spec.hidden_size}"
            )
