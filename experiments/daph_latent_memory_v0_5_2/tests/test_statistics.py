from daph_latent_memory.evaluation.statistics import mcnemar_test, cochran_q_test, wilcoxon_signed_rank, paired_bootstrap_ci

def test_mcnemar_no_disagreement():
    matched = [1, 1, 0, 0, 1, 1]
    other = [1, 1, 0, 0, 1, 1]
    result = mcnemar_test(matched, other)
    assert result["p_value"] == 1.0
    assert result["b"] == 0 and result["c"] == 0

def test_mcnemar_significant_disagreement():
    matched = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    other = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result = mcnemar_test(matched, other)
    assert result["b"] == 10 and result["c"] == 0
    assert result["p_value"] < 0.05

def test_mcnemar_mixed():
    matched = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    other = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    result = mcnemar_test(matched, other)
    assert result["b"] + result["c"] > 0

def test_cochran_q_requires_3_conditions():
    try:
        cochran_q_test([[1, 0], [0, 1]])
        assert False, "Should have raised"
    except ValueError:
        pass

def test_cochran_q_identical_conditions():
    cond = [[1, 0, 1, 1, 0]] * 3
    result = cochran_q_test(cond)
    assert result["Q"] == 0.0
    assert result["p_value"] == 1.0

def test_wilcoxon_identical():
    result = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["n"] == 0
    assert result["p_value"] == 1.0

def test_wilcoxon_different():
    result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0])
    assert result["n"] > 0

def test_paired_bootstrap_ci():
    a = [0.8, 0.9, 0.7, 0.85, 0.95, 0.75, 0.8, 0.9, 0.85, 0.8]
    b = [0.7, 0.8, 0.6, 0.75, 0.85, 0.65, 0.7, 0.8, 0.75, 0.7]
    result = paired_bootstrap_ci(a, b, samples=500, seed=42)
    assert result["mean_diff"] > 0  # a consistently > b
    assert result["lo"] > 0 or abs(result["lo"]) < 0.1  # CI should not strongly cross 0
