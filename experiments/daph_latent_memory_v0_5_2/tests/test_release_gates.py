import json, tempfile, os
from pathlib import Path
from scripts.check_release_gates import (
    check_gate1, check_gate2, check_gate3, check_gate4, check_gate5, check_gate6, check_all_gates
)

def _make_causal_eval(a_matched=0.2, a_shuffled=0.1, a_wrong=0.05, a_base=0.1,
                     a_0=0.2, a_05=0.19, a_2=0.05, n=100):
    """Create a mock causal_eval.json structure."""
    matched_c = [1]*int(a_matched*n) + [0]*(n-int(a_matched*n))
    shuffled_c = [1]*int(a_shuffled*n) + [0]*(n-int(a_shuffled*n))
    wrong_c = [1]*int(a_wrong*n) + [0]*(n-int(a_wrong*n))
    base_c = [1]*int(a_base*n) + [0]*(n-int(a_base*n))
    return {
        "conditions": {
            "matched": {"accuracy": a_matched, "correct": matched_c},
            "shuffled_global": {"accuracy": a_shuffled, "correct": shuffled_c},
            "shuffled_wrong_class": {"accuracy": a_wrong, "correct": wrong_c},
            "base": {"accuracy": a_base, "correct": base_c},
        },
        "corruption_sweep": {
            "0.0": {"accuracy": a_0, "correct": [1]*int(a_0*n)+[0]*(n-int(a_0*n))},
            "0.5": {"accuracy": a_05, "correct": [1]*int(a_05*n)+[0]*(n-int(a_05*n))},
            "2.0": {"accuracy": a_2, "correct": [1]*int(a_2*n)+[0]*(n-int(a_2*n))},
        },
        "headline": {"css": (a_matched-a_shuffled)*100, "lug": (a_matched-a_base)/9},
        "statistics": {"css_threshold": 5.0},
    }

def test_gate1_passes_when_matched_better():
    results = _make_causal_eval(a_matched=0.3, a_shuffled=0.1, a_base=0.15)
    gate = check_gate1(results)
    assert gate["passed"]

def test_gate1_fails_when_matched_not_better_than_base():
    results = _make_causal_eval(a_matched=0.1, a_shuffled=0.1, a_base=0.15)
    gate = check_gate1(results)
    assert not gate["passed"]

def test_gate2_passes_when_corruption_degrades():
    results = _make_causal_eval(a_0=0.3, a_05=0.295, a_2=0.1)
    gate = check_gate2(results)
    assert gate["passed"]

def test_gate2_fails_when_corruption_doesnt_degrade():
    results = _make_causal_eval(a_0=0.2, a_05=0.21, a_2=0.2)
    gate = check_gate2(results)
    assert not gate["passed"]

def test_gate3_passes_when_wrong_class_degrades():
    results = _make_causal_eval(a_matched=0.3, a_wrong=0.1)
    gate = check_gate3(results)
    assert gate["passed"]

def test_gate3_fails_when_wrong_class_ok():
    results = _make_causal_eval(a_matched=0.2, a_wrong=0.18)
    gate = check_gate3(results)
    assert not gate["passed"]

def test_gate4_passes_with_ood_gain():
    ood_results = {"operand_ood": {"gg": 0.05, "mcnemar": {"p_value": 0.01}}}
    gate = check_gate4(ood_results)
    assert gate["passed"]

def test_gate4_fails_without_ood_gain():
    ood_results = {"operand_ood": {"gg": -0.05, "mcnemar": {"p_value": 0.5}}}
    gate = check_gate4(ood_results)
    assert not gate["passed"]

def test_gate5_passes_no_degradation():
    results = _make_causal_eval(a_base=0.2)
    gate = check_gate5(results, base_general_accuracy=0.2, max_degradation_pp=2.0)
    assert gate["passed"]

def test_gate5_fails_with_degradation():
    results = _make_causal_eval(a_base=0.15)
    gate = check_gate5(results, base_general_accuracy=0.2, max_degradation_pp=2.0)
    assert not gate["passed"]

def test_gate6_fails_with_too_few_seeds():
    seed_results = [{"css": 10.0, "mcnemar_p": 0.001}] * 3
    gate = check_gate6(seed_results)
    assert not gate["passed"]

def test_gate6_passes_with_5_seeds():
    seed_results = [{"css": 10.0, "mcnemar_p": 0.001}] * 5
    gate = check_gate6(seed_results)
    assert gate["passed"]

def test_check_all_gates():
    causal = _make_causal_eval(a_matched=0.3, a_shuffled=0.1, a_wrong=0.05, a_base=0.15,
                               a_0=0.3, a_05=0.29, a_2=0.1)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(causal, f); causal_path = f.name
    try:
        report = check_all_gates(causal_path)
        assert "gates" in report
        assert len(report["gates"]) == 6
    finally:
        os.unlink(causal_path)
