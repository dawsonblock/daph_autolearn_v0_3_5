from daph_learning.evaluation.scoring import parse_final_or_exact, parse_legacy_first_int


def test_final_parser():
    assert parse_final_or_exact("FINAL: -123") == -123
    assert parse_final_or_exact("-123") == -123
    assert parse_final_or_exact("First multiply 345 by 821. FINAL: 283245") == 283245
    assert parse_final_or_exact("First multiply 345 by 821. Answer 283245") is None


def test_legacy_parser_documents_old_behavior():
    assert parse_legacy_first_int("First multiply 345 by 821. Answer 283245") == 345
