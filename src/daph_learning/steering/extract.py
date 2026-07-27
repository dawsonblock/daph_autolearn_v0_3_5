from __future__ import annotations

from typing import Iterable

import numpy as np

from .types import SteeringSpec, SteeringVector


def contrastive_mean_direction(
    positives: Iterable[np.ndarray],
    negatives: Iterable[np.ndarray],
    spec: SteeringSpec,
    *,
    normalize: bool | None = None,
) -> SteeringVector:
    """First-order contrastive mean-difference direction.

    ``v = mean(positives) - mean(negatives)``, optionally L2-normalized.

    Normalization is determined by, in priority order:
    1. the explicit ``normalize`` argument if not ``None``;
    2. ``spec.normalization`` if set (``"l2"`` -> normalize, ``"none"`` ->
       do not normalize, ``"mean_centered"`` -> normalize, treated as l2
       for the direction itself since mean-centering is a capture-time
       concern);
    3. default ``True`` (L2 normalize) for backward compatibility.

    The recorded ``spec.normalization`` should reflect what was actually
    applied so that the run manifest's provenance is accurate.
    """
    pos = np.stack([np.asarray(x, dtype=np.float32) for x in positives])
    neg = np.stack([np.asarray(x, dtype=np.float32) for x in negatives])
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError("positive/negative activations must be [N, hidden] with matching hidden size")
    direction = pos.mean(axis=0) - neg.mean(axis=0)

    if normalize is None:
        if spec.normalization is None:
            normalize = True
        elif spec.normalization == "none":
            normalize = False
        else:
            # "l2" and "mean_centered" both normalize the final direction.
            normalize = True

    if normalize:
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("contrastive direction has zero norm")
        direction = direction / norm
    return SteeringVector(spec=spec, values=direction.astype(np.float32))
