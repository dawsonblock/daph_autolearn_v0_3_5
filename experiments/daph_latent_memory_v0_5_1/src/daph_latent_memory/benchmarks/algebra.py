from __future__ import annotations
import random
from dataclasses import dataclass
from ..state.schema import MathExample, MathWorkingState


@dataclass(frozen=True)
class GeneratedProblem:
    question: str
    answer: str
    state: MathWorkingState
    template: str
    difficulty: int


def _linear_one_step(rng: random.Random) -> GeneratedProblem:
    x = rng.randint(-20, 20)
    b = rng.choice([i for i in range(-15, 16) if i != 0])
    c = x + b
    question = f"Solve for x: x + ({b}) = {c}"
    state = MathWorkingState(domain="algebra", operation="linear_equation", variables={"x":"unknown"}, constraints=[question.replace("Solve for x: ", "")], retrieved_concepts=["inverse operations","equality preservation"], intermediate_steps=[], unresolved_subgoals=[f"apply the inverse of adding {b} to both sides"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question, str(x), state, "linear_one_step", 1)


def _linear_two_step(rng: random.Random) -> GeneratedProblem:
    x = rng.choice([i for i in range(-20, 21) if i != 0])
    a = rng.choice([i for i in range(-9, 10) if i not in (0, 1, -1)])
    b = rng.choice([i for i in range(-15, 16) if i != 0])
    c = a * x + b
    question = f"Solve for x: {a}*x + ({b}) = {c}"
    transformed = c - b
    state = MathWorkingState(domain="algebra", operation="two_step_linear_equation", variables={"x":"unknown","coefficient":a}, constraints=[f"{a}*x + ({b}) = {c}"], retrieved_concepts=["inverse operations","equality preservation"], intermediate_steps=[f"after removing the constant term, the equation is {a}*x = {transformed}"], unresolved_subgoals=[f"divide both sides by the coefficient {a}"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question, str(x), state, "linear_two_step", 2)


def _linear_composed(rng: random.Random) -> GeneratedProblem:
    x = rng.choice([i for i in range(-15, 16) if i != 0])
    a = rng.choice([i for i in range(-7, 8) if i not in (0, 1, -1)])
    b = rng.choice([i for i in range(-10, 11) if i != 0])
    c = a * (x + b)
    question = f"Solve for x: {a}*(x + ({b})) = {c}"
    inner = c // a
    state = MathWorkingState(domain="algebra", operation="distributed_linear_equation", variables={"x":"unknown","outer_coefficient":a}, constraints=[f"{a}*(x + ({b})) = {c}"], retrieved_concepts=["inverse operations","function inversion"], intermediate_steps=[f"divide both sides by {a} to obtain x + ({b}) = {inner}"], unresolved_subgoals=[f"remove the additive offset {b}"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question, str(x), state, "linear_composed", 3)


def _ood_three_step(rng: random.Random) -> GeneratedProblem:
    x = rng.choice([i for i in range(-12, 13) if i != 0])
    a = rng.choice([i for i in range(-8, 9) if i not in (0, 1, -1)])
    c = rng.choice([i for i in range(-8, 9) if i not in (a, 0)])
    b = rng.randint(-10, 10)
    d = (a - c) * x + b
    question = f"Solve for x: {a}*x + ({b}) = {c}*x + ({d})"
    coeff = a - c
    offset = d - b
    state = MathWorkingState(domain="algebra", operation="variables_both_sides", variables={"x":"unknown"}, constraints=[f"{a}*x + ({b}) = {c}*x + ({d})"], retrieved_concepts=["collect like terms","inverse operations","equality preservation"], intermediate_steps=[f"collect x terms to obtain {coeff}*x = {offset}"], unresolved_subgoals=[f"isolate x by dividing by {coeff}"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question, str(x), state, "variables_both_sides", 4)

GENERATORS={"train":[_linear_one_step,_linear_two_step],"iid":[_linear_one_step,_linear_two_step],"composition":[_linear_composed],"ood":[_ood_three_step]}

def generate_example(split: str, index: int, seed: int = 1337) -> MathExample:
    split_offsets={"train":11,"iid":23,"composition":37,"ood":53}
    if split not in split_offsets: raise ValueError(f"Unknown split: {split}")
    rng=random.Random(seed*1_000_003+index*97+split_offsets[split]); funcs=GENERATORS[split]
    for _ in range(100):
        try:
            p=rng.choice(funcs)(rng)
            return MathExample(example_id=f"{split}-{index}",split=split,question=p.question,answer=p.answer,state=p.state,template=p.template,difficulty=p.difficulty)
        except ValueError: continue
    raise RuntimeError("Could not generate leakage-free example after 100 attempts")
