import numpy as np
from analysis.cca_analysis import cca, project_to_shared, shared_subspace_steering, cca_summary

def test_cca_identical_matrices():
    """CCA of identical matrices should give correlation ~1.0."""
    X = np.random.randn(50, 32)
    result = cca(X, X, n_components=5)
    assert result["max_correlation"] > 0.99

def test_cca_independent_matrices():
    """CCA of independent matrices should give low correlations with enough samples."""
    np.random.seed(42)
    X = np.random.randn(200, 16)
    Y = np.random.randn(200, 16)
    result = cca(X, Y, n_components=5, reg=1.0)
    assert result["mean_correlation"] < 0.5  # should be low for random with reg

def test_cca_linear_relationship():
    """CCA should find linear relationships."""
    X = np.random.randn(100, 16)
    W = np.random.randn(16, 8)
    Y = X @ W + 0.01 * np.random.randn(100, 8)  # Y is linear transform of X
    result = cca(X, Y, n_components=4)
    assert result["correlations"][0] > 0.9  # first canonical correlation should be high

def test_cca_shapes():
    X = np.random.randn(50, 20)
    Y = np.random.randn(50, 15)
    result = cca(X, Y, n_components=5)
    assert result["U"].shape == (20, 5)
    assert result["V"].shape == (15, 5)
    assert result["shared_subspace"].shape == (50, 5)

def test_cca_n_components():
    X = np.random.randn(50, 20)
    Y = np.random.randn(50, 15)
    result = cca(X, Y, n_components=3)
    assert len(result["correlations"]) == 3

def test_project_to_shared():
    X = np.random.randn(50, 20)
    result = cca(X, X.copy(), n_components=5)
    projected = project_to_shared(X, result["U"])
    assert projected.shape == (50, 5)

def test_shared_subspace_steering():
    X = np.random.randn(50, 20)
    result = cca(X, X.copy(), n_components=5)
    steered = shared_subspace_steering(X, result["U"], lam=0.5)
    assert steered.shape == X.shape

def test_cca_summary():
    X = np.random.randn(50, 20)
    Y = np.random.randn(50, 15)
    result = cca(X, Y, n_components=5)
    summary = cca_summary(result)
    assert "mean_correlation" in summary
    assert "max_correlation" in summary
    assert len(summary["correlations"]) == 5

def test_cca_mismatched_samples():
    X = np.random.randn(50, 20)
    Y = np.random.randn(30, 15)
    try:
        cca(X, Y)
        assert False, "Should have raised"
    except ValueError:
        pass
