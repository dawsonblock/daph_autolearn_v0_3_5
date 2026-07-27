from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence
import numpy as np

from .types import SteeringSpec, SteeringVector


def save_vector(vector: SteeringVector, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=vector.values.astype(np.float32),
        metadata=np.array(json.dumps(vector.spec.__dict__)),
    )


def load_vector(path: str | Path) -> SteeringVector:
    data = np.load(Path(path), allow_pickle=False)
    spec = SteeringSpec(**json.loads(str(data["metadata"].item())))
    return SteeringVector(spec=spec, values=np.asarray(data["values"], dtype=np.float32))



def load_vector_bundle(
    paths_or_pattern: str | Sequence[str | Path],
) -> list[SteeringVector]:
    """Load a validated multi-layer steering-vector bundle.

    Accepted forms:
    - comma-separated path string
    - sequence of paths
    - JSON manifest path containing either:
      - `["l20.npz", "l24.npz"]`
      - `{"vectors": ["l20.npz", ...]}`
      - `{"vectors": [{"path": "l20.npz", "alpha": 0.8}, ...]}`

    Relative paths inside a JSON manifest are resolved relative to the manifest.
    Optional manifest ``alpha`` values replace the vector's stored alpha.
    Duplicate layers are rejected.
    """
    entries: list[tuple[Path, float | None]] = []

    if isinstance(paths_or_pattern, str):
        candidate = Path(paths_or_pattern)
        if "," not in paths_or_pattern and candidate.suffix.lower() == ".json" and candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            raw_vectors = payload.get("vectors") if isinstance(payload, dict) else payload
            if not isinstance(raw_vectors, list):
                raise ValueError("vector bundle JSON must be a list or contain a 'vectors' list")
            for item in raw_vectors:
                if isinstance(item, str):
                    entries.append(((candidate.parent / item).resolve(), None))
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    alpha = item.get("alpha")
                    if alpha is not None:
                        alpha = float(alpha)
                    entries.append(((candidate.parent / item["path"]).resolve(), alpha))
                else:
                    raise ValueError(f"invalid vector-bundle entry: {item!r}")
        else:
            paths = [part.strip() for part in paths_or_pattern.split(",") if part.strip()]
            entries = [(Path(path), None) for path in paths]
    else:
        entries = [(Path(path), None) for path in paths_or_pattern]

    if not entries:
        raise ValueError("steering-vector bundle is empty")

    vectors: list[SteeringVector] = []
    for path, alpha_override in entries:
        vector = load_vector(path)
        if alpha_override is not None:
            vector = SteeringVector(
                spec=replace(vector.spec, alpha=alpha_override),
                values=vector.values,
            )
        vectors.append(vector)

    layers = [vector.spec.layer for vector in vectors]
    if len(layers) != len(set(layers)):
        raise ValueError(f"composite vector bundle contains duplicate layers: {layers}")
    return vectors
