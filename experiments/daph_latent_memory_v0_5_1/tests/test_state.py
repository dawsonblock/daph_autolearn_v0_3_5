import pytest
from daph_latent_memory.state.schema import MathWorkingState

def test_final_answer_field_forbidden():
    with pytest.raises(ValueError):MathWorkingState(domain="algebra",operation="solve",variables={},constraints=[],answer="6")

def test_direct_answer_encoding_rejected():
    s=MathWorkingState(domain="algebra",operation="solve",variables={"x":"unknown"},constraints=["2*x = 12"],intermediate_steps=["x = 6"])
    with pytest.raises(ValueError):s.assert_no_direct_answer_encoding("6")

def test_incidental_numeric_overlap_is_allowed():
    s=MathWorkingState(domain="algebra",operation="solve",variables={"x":"unknown","coefficient":6},constraints=["6*x + 2 = 20"],unresolved_subgoals=["divide by 6 after removing the constant"]);s.assert_no_direct_answer_encoding("3")
