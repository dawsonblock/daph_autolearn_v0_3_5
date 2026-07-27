"""Tests for the linear-probe baseline (scripts/linear_probe_baseline.py).

This baseline trains a logistic regression on captured activations to predict
the route label, providing a steering-free upper bound. See CLAIMS.md §9.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from scripts.linear_probe_baseline import run_linear_probe, main


def test_linear_probe_separable_classes():
    """When the two classes are linearly separable, the probe should
    achieve near-perfect accuracy."""
    rng = np.random.RandomState(42)
    positive = rng.randn(100, 8) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
    negative = rng.randn(100, 8) + np.array([-5, 0, 0, 0, 0, 0, 0, 0])
    metrics = run_linear_probe(positive, negative, n_splits=5, random_state=42)
    assert metrics["accuracy"] > 0.95
    assert metrics["lift_over_baseline"] > 0.45  # baseline is 0.5
    assert metrics["n_samples"] == 200
    assert metrics["n_features"] == 8
    assert metrics["n_positive"] == 100
    assert metrics["n_negative"] == 100


def test_linear_probe_random_classes_near_baseline():
    """When the classes are random (no signal), the probe should be near
    the majority-class baseline."""
    rng = np.random.RandomState(42)
    positive = rng.randn(100, 8)
    negative = rng.randn(100, 8)
    metrics = run_linear_probe(positive, negative, n_splits=5, random_state=42)
    # Accuracy should be near 0.5 (the baseline)
    assert abs(metrics["accuracy"] - metrics["random_baseline"]) < 0.2
    assert metrics["lift_over_baseline"] < 0.2


def test_linear_probe_imbalanced_classes():
    """The majority-class baseline should reflect class imbalance."""
    rng = np.random.RandomState(42)
    positive = rng.randn(20, 4) + 3
    negative = rng.randn(80, 4) - 3
    metrics = run_linear_probe(positive, negative, n_splits=5, random_state=42)
    # 80% of samples are negative, so random baseline is 0.8
    assert abs(metrics["random_baseline"] - 0.8) < 1e-6
    assert metrics["n_positive"] == 20
    assert metrics["n_negative"] == 80


def test_linear_probe_rejects_mismatched_dims():
    positive = np.zeros((10, 4))
    negative = np.zeros((10, 8))
    with pytest.raises(ValueError, match="hidden_size"):
        run_linear_probe(positive, negative)


def test_linear_probe_rejects_wrong_ndim():
    positive = np.zeros((10, 4, 2))  # 3D
    negative = np.zeros((10, 4))
    with pytest.raises(ValueError, match="2D"):
        run_linear_probe(positive, negative)


def test_linear_probe_rejects_too_few_samples():
    positive = np.zeros((1, 4))
    negative = np.zeros((1, 4))
    with pytest.raises(ValueError, match="n_samples"):
        run_linear_probe(positive, negative, n_splits=5)


def test_linear_probe_rejects_too_few_per_class_for_cv():
    """When one class has fewer than 2 samples, CV can't split."""
    positive = np.zeros((1, 4))
    negative = np.zeros((10, 4))
    with pytest.raises(ValueError, match="cross-validation|n_samples"):
        run_linear_probe(positive, negative, n_splits=5)


def test_linear_probe_confusion_matrix_shape():
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 4) + 2
    negative = rng.randn(50, 4) - 2
    metrics = run_linear_probe(positive, negative, n_splits=5, random_state=42)
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2  # 2x2 confusion matrix
    assert all(len(row) == 2 for row in cm)


def test_linear_probe_cli_end_to_end(tmp_path: Path, monkeypatch):
    """The CLI should produce a JSON file with the expected fields."""
    rng = np.random.RandomState(42)
    positive = rng.randn(60, 6) + 3
    negative = rng.randn(60, 6) - 3
    pos_path = tmp_path / "pos.npy"
    neg_path = tmp_path / "neg.npy"
    np.save(pos_path, positive)
    np.save(neg_path, negative)
    out_path = tmp_path / "probe.json"

    monkeypatch.setattr(sys, "argv", [
        "linear_probe_baseline.py",
        "--positive", str(pos_path),
        "--negative", str(neg_path),
        "--output", str(out_path),
        "--n-splits", "5",
        "--seed", "42",
    ])
    main()

    metrics = json.loads(out_path.read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert "accuracy_std" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "random_baseline" in metrics
    assert "lift_over_baseline" in metrics
    assert metrics["n_samples"] == 120
    assert metrics["n_features"] == 6
    # Separable data should give high accuracy
    assert metrics["accuracy"] > 0.9


def test_linear_probe_deterministic_with_seed():
    """Same seed should produce same results."""
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 4) + 2
    negative = rng.randn(50, 4) - 2
    m1 = run_linear_probe(positive, negative, n_splits=5, random_state=42)
    m2 = run_linear_probe(positive, negative, n_splits=5, random_state=42)
    assert m1["accuracy"] == m2["accuracy"]
