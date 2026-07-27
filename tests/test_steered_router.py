from daph_learning.routing.steered_router import build_route_prompt, parse_route_action


def test_route_action_parser():
    assert parse_route_action("SYMBOLIC") == "symbolic"
    assert parse_route_action("ACTION: LLM") == "llm"
    assert parse_route_action("I cannot decide") is None
    assert parse_route_action("I think SYMBOLIC") is None


def test_route_prompt_has_stable_anchor():
    prompt = build_route_prompt({"capability_ids": ["integer_arithmetic"], "specification": "Compute 2*3."})
    assert prompt.endswith("ACTION:")
