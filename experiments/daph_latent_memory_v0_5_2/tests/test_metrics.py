from daph_latent_memory.evaluation.metrics import (
    normalize_answer, exact_match, causal_specificity_score, wrong_memory_penalty,
    latent_utility_gain, generalization_gain, css_null_distribution, css_threshold,
)

def test_normalize_answer_extracts_last_number():
    assert normalize_answer("The answer is 42") == "42"
    assert normalize_answer("x = -7") == "-7"
    assert normalize_answer("Result: 3.14") == "3.14"

def test_exact_match():
    assert exact_match("42", "42") == 1
    assert exact_match("The answer is 42", "42") == 1
    assert exact_match("42", "43") == 0

def test_css():
    assert abs(causal_specificity_score(0.15, 0.10) - 5.0) < 1e-10
    assert abs(causal_specificity_score(0.20, 0.10) - 10.0) < 1e-10

def test_wmp():
    assert abs(wrong_memory_penalty(0.20, 0.05) - 15.0) < 1e-10

def test_lug():
    assert abs(latent_utility_gain(0.15, 0.10, 8) - 0.05/9) < 1e-10

def test_gg():
    assert abs(generalization_gain(0.15, 0.10) - 0.05) < 1e-10

def test_css_null_distribution_mean_near_zero():
    # Under the null, CSS should be centered near 0
    matched = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10
    shuffled = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 10
    null = css_null_distribution(matched, shuffled, n_permutations=200, seed=42)
    mean_css = sum(null) / len(null)
    assert abs(mean_css) < 2.0, f"Null CSS mean should be near 0, got {mean_css}"

def test_css_threshold_respects_floor():
    null = [0.1, 0.2, 0.3, 0.5, 1.0, -0.1, -0.2]
    threshold = css_threshold(null, percentile=99, floor_pp=5.0)
    assert threshold >= 5.0

def test_css_threshold_uses_percentile():
    null = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    threshold = css_threshold(null, percentile=90, floor_pp=0.0)
    assert threshold >= 8.0  # 90th percentile should be >= 8
