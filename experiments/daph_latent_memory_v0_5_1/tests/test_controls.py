import random
from daph_latent_memory.evaluation.controls import deranged_states
from daph_latent_memory.state.corruption import corrupt_state
from daph_latent_memory.state.schema import MathWorkingState

def test_derangement_has_no_fixed_points():
    states=list(range(20));shuffled=deranged_states(states,seed=1);assert len(shuffled)==len(states);assert all(a!=b for a,b in zip(states,shuffled))

def test_corruption_changes_semantics_text():
    s=MathWorkingState(domain="algebra",operation="two_step",variables={"x":"unknown"},constraints=["2*x + 3 = 11"],intermediate_steps=["after removing 3, obtain 2*x = 8"],unresolved_subgoals=["divide by 2"]);c=corrupt_state(s,random.Random(1));assert c.process_text()!=s.process_text();assert "CORRUPTED:" not in c.process_text()
