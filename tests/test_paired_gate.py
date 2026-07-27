from daph_learning.evaluation.paired import exact_paired_binomial


def test_notebook_discordant_p_value():
    assert exact_paired_binomial(7, 2) == 0.1796875
    assert exact_paired_binomial(0, 0) == 1.0


def test_exact_paired_symmetry():
    assert exact_paired_binomial(2, 7) == exact_paired_binomial(7, 2)
