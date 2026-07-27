
from scripts.tune_steering import _score_key


def test_tuning_score_key_prefers_f1_then_coverage_then_accuracy():
    a = {"metrics": {"f1": 0.8, "decision_coverage": 0.9, "route_accuracy": 0.7}}
    b = {"metrics": {"f1": 0.8, "decision_coverage": 0.95, "route_accuracy": 0.6}}
    c = {"metrics": {"f1": 0.81, "decision_coverage": 0.5, "route_accuracy": 0.5}}
    assert _score_key(c) > _score_key(b) > _score_key(a)
