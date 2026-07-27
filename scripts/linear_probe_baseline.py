"""Linear-probe baseline for routing prediction.

This script trains a logistic regression on captured router activations to
predict the symbolic/llm route label. It provides a **steering-free upper
bound** on what the activation direction alone can predict, which is the
baseline the audit (§9) requires for evaluating whether steering is actually
contributing beyond what a simple linear classifier could do.

Usage:

    python scripts/linear_probe_baseline.py \\
        --positive positive.npy \\
        --negative negative.npy \\
        --output artifacts/linear_probe.json

The output JSON contains:

- accuracy (5-fold cross-validated)
- accuracy_std
- precision / recall / f1 (macro-averaged)
- confusion matrix
- n_samples, n_features
- random_baseline (accuracy of always predicting the majority class)

See CLAIMS.md §9 (linear-probe baseline) for why this matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_import_sklearn():
    """Import sklearn lazily; the script errors clearly if unavailable."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )
        return (
            LogisticRegression,
            cross_val_predict,
            StratifiedKFold,
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the linear-probe baseline. "
            "Install it with: pip install scikit-learn"
        ) from exc


def run_linear_probe(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Train a logistic regression on activations and report cross-validated
    routing prediction metrics.

    Parameters
    ----------
    positive : np.ndarray
        Activations from tasks where the symbolic route was preferred,
        shape ``[N_pos, H]``.
    negative : np.ndarray
        Activations from tasks where the LLM route was preferred,
        shape ``[N_neg, H]``.
    n_splits : int
        Number of stratified cross-validation folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Cross-validated metrics: accuracy, accuracy_std, precision, recall,
        f1, confusion_matrix, n_samples, n_features, random_baseline.
    """
    (
        LogisticRegression,
        cross_val_predict,
        StratifiedKFold,
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    ) = _safe_import_sklearn()

    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError(
            f"activations must be 2D [N, H], got {positive.shape} and {negative.shape}"
        )
    if positive.shape[1] != negative.shape[1]:
        raise ValueError(
            f"positive hidden_size {positive.shape[1]} != negative hidden_size {negative.shape[1]}"
        )

    X = np.vstack([positive, negative]).astype(np.float32)
    y = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    n_samples, n_features = X.shape

    if n_samples < n_splits:
        raise ValueError(
            f"n_samples ({n_samples}) < n_splits ({n_splits}); reduce --n-splits"
        )

    # Majority-class baseline
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos
    random_baseline = max(n_pos, n_neg) / n_samples

    # Stratified k-fold cross-validated predictions
    n_splits = min(n_splits, n_pos, n_neg)
    if n_splits < 2:
        raise ValueError(
            f"need at least 2 samples per class for cross-validation; "
            f"got {n_pos} positive, {n_neg} negative"
        )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    accuracy = float(accuracy_score(y, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y, y_pred).tolist()

    # Also compute per-fold accuracy std for reporting
    fold_accs = []
    for train_idx, test_idx in cv.split(X, y):
        clf_fold = LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
        clf_fold.fit(X[train_idx], y[train_idx])
        fold_accs.append(float(accuracy_score(y[test_idx], clf_fold.predict(X[test_idx]))))
    accuracy_std = float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_std": accuracy_std,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "n_samples": int(n_samples),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_features": int(n_features),
        "n_splits": n_splits,
        "random_baseline": random_baseline,
        "lift_over_baseline": accuracy - random_baseline,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Linear-probe baseline: train a logistic regression on captured "
            "router activations to predict the symbolic/llm route label. "
            "Provides a steering-free upper bound for comparing against "
            "steering-based routing. See CLAIMS.md §9."
        )
    )
    ap.add_argument("--positive", required=True, help="Path to positive activations .npy [N_pos, H]")
    ap.add_argument("--negative", required=True, help="Path to negative activations .npy [N_neg, H]")
    ap.add_argument("--output", required=True, help="Path to output JSON")
    ap.add_argument("--n-splits", type=int, default=5, help="Cross-validation folds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    positive = np.load(args.positive)
    negative = np.load(args.negative)
    metrics = run_linear_probe(positive, negative, n_splits=args.n_splits, random_state=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
