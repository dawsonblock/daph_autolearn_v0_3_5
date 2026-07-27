"""Tests for route normalization (AutoLearn v2 Phase 1A)."""

from __future__ import annotations

import pytest

from daph_learning.routing.normalizer import (
    EXECUTABLE_BACKENDS,
    NormalizedRoute,
    normalize_route_result,
    normalize_route_sequence,
)
from daph_learning.routing.errors import InvalidRouteDecisionError
from daph_learning.routing.policy import RouteDecision


def test_normalize_bare_string_symbolic():
    r = normalize_route_result("symbolic")
    assert r.route == "symbolic"
    assert r.is_executable


def test_normalize_bare_string_llm():
    r = normalize_route_result("llm")
    assert r.route == "llm"
    assert r.is_executable


def test_normalize_generate_tuple_unpacked():
    """The Phase 1A bug fix: ('symbolic', raw_text) must unpack the label,
    not treat the tuple as a string label."""
    r = normalize_route_result(("symbolic", "SYMBOLIC some text"))
    assert r.route == "symbolic"
    assert r.raw_text == "SYMBOLIC some text"
    assert r.parse_warning == "tuple_unpacked"


def test_normalize_generate_tuple_llm():
    r = normalize_route_result(("llm", "LLM because..."))
    assert r.route == "llm"
    assert r.raw_text == "LLM because..."


def test_normalize_tuple_label_case_insensitive():
    r = normalize_route_result(("SYMBOLIC", "raw"))
    assert r.route == "symbolic"


def test_normalize_unknown_label_defaults_to_abstain():
    """Fail-closed: an unparseable route does NOT silently become LLM."""
    r = normalize_route_result("banana")
    assert r.route == "abstain"
    assert "unknown_label" in (r.parse_warning or "")


def test_normalize_unknown_label_can_raise():
    with pytest.raises(InvalidRouteDecisionError):
        normalize_route_result("banana", on_unknown="raise")


def test_normalize_none_defaults_to_abstain():
    r = normalize_route_result(None)
    assert r.route == "abstain"


def test_normalize_none_can_become_llm():
    r = normalize_route_result(None, on_unknown="llm")
    assert r.route == "llm"


def test_normalize_policy_route_decision():
    rd = RouteDecision(task_id="t1", route="symbolic", source="steered")
    r = normalize_route_result(rd, source="logit")
    assert r.route == "symbolic"
    assert r.source == "logit"


def test_normalize_mapping_with_route_key():
    r = normalize_route_result({"route": "llm", "confidence": 0.8, "raw_scores": {"llm": 1.2}})
    assert r.route == "llm"
    assert r.confidence == 0.8
    assert r.raw_scores == {"llm": 1.2}


def test_normalize_mapping_with_backend_key():
    r = normalize_route_result({"backend": "symbolic"})
    assert r.route == "symbolic"


def test_normalize_mapping_missing_route_key_abstains():
    r = normalize_route_result({"foo": "bar"})
    assert r.route == "abstain"
    assert "missing_route_key" in (r.parse_warning or "")


def test_normalize_mapping_invalid_confidence_raises():
    with pytest.raises(InvalidRouteDecisionError):
        normalize_route_result({"route": "llm", "confidence": "high"})


def test_normalize_abstain_label():
    r = normalize_route_result("abstain")
    assert r.route == "abstain"
    assert not r.is_executable


def test_normalize_executable_backends_excludes_abstain():
    assert "abstain" not in EXECUTABLE_BACKENDS
    assert "symbolic" in EXECUTABLE_BACKENDS
    assert "llm" in EXECUTABLE_BACKENDS


def test_normalize_sequence():
    seq = normalize_route_sequence(["symbolic", ("llm", "raw"), None, "banana"])
    assert [r.route for r in seq] == ["symbolic", "llm", "abstain", "abstain"]


def test_normalized_route_rejects_invalid_label():
    with pytest.raises(InvalidRouteDecisionError):
        NormalizedRoute(route="banana")


def test_normalize_tuple_wrong_length_raises():
    with pytest.raises(InvalidRouteDecisionError):
        normalize_route_result(("a", "b", "c"))


def test_normalize_on_unknown_invalid_value_raises():
    with pytest.raises(ValueError):
        normalize_route_result("x", on_unknown="banana")


def test_normalize_logit_source_preserved():
    r = normalize_route_result("symbolic", source="logit")
    assert r.source == "logit"


def test_normalize_fallback_source():
    r = normalize_route_result(None, source="fallback")
    # None -> abstain path sets source to "abstain"
    assert r.route == "abstain"
