from __future__ import annotations
import random
from dataclasses import dataclass
from ..state.schema import MathExample,MathWorkingState

# =============================================================================
# v0.5.2 Dataset Generators
#
# Splits (corrected plan section 15):
#   train, iid         — same structure and ranges as training
#   composition        — (a op b) op c
#   ood                — variables on both sides (original v0.5.1 OOD)
#   operand_ood        — train 1-999, test 1000-99999
#   structural_ood     — train a op b, test (a op b) op c
#   operator_ood       — train + - *, test % // **
#   distractor_ood     — irrelevant numbers and NL clutter
#   counterfactual     — same surface structure, different answer
#   adversarial        — same operands, different operators
# =============================================================================

@dataclass(frozen=True)
class GeneratedProblem:
    question:str
    answer:str
    state:MathWorkingState
    template:str
    difficulty:int
    skill_label:str

# Skill labels
SKILL_ADD="add"
SKILL_SUBTRACT="subtract"
SKILL_MULTIPLY="multiply"
SKILL_DIVIDE="divide"
SKILL_VERIFY="verify"
SKILL_DECOMPOSE="decompose"

def _linear_one_step(rng:random.Random)->GeneratedProblem:
    x=rng.randint(-20,20);b=rng.choice([i for i in range(-15,16) if i!=0]);c=x+b
    question=f"Solve for x: x + ({b}) = {c}"
    state=MathWorkingState(domain="algebra",operation="linear_equation",variables={"x":"unknown"},constraints=[question.replace("Solve for x: ","")],retrieved_concepts=["inverse operations","equality preservation"],intermediate_steps=[],unresolved_subgoals=[f"apply the inverse of adding {b} to both sides"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question,str(x),state,"linear_one_step",1,SKILL_SUBTRACT)

def _linear_two_step(rng:random.Random)->GeneratedProblem:
    x=rng.choice([i for i in range(-20,21) if i!=0]);a=rng.choice([i for i in range(-9,10) if i not in (0,1,-1)]);b=rng.choice([i for i in range(-15,16) if i!=0]);c=a*x+b
    question=f"Solve for x: {a}*x + ({b}) = {c}"
    transformed=c-b
    state=MathWorkingState(domain="algebra",operation="two_step_linear_equation",variables={"x":"unknown","coefficient":a},constraints=[f"{a}*x + ({b}) = {c}"],retrieved_concepts=["inverse operations","equality preservation"],intermediate_steps=[f"after removing the constant term, the equation is {a}*x = {transformed}"],unresolved_subgoals=[f"divide both sides by the coefficient {a}"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question,str(x),state,"linear_two_step",2,SKILL_DIVIDE)

def _linear_composed(rng:random.Random)->GeneratedProblem:
    x=rng.choice([i for i in range(-15,16) if i!=0]);a=rng.choice([i for i in range(-7,8) if i not in (0,1,-1)]);b=rng.choice([i for i in range(-10,11) if i!=0]);c=a*(x+b)
    question=f"Solve for x: {a}*(x + ({b})) = {c}"
    inner=c//a
    state=MathWorkingState(domain="algebra",operation="distributed_linear_equation",variables={"x":"unknown","outer_coefficient":a},constraints=[f"{a}*(x + ({b})) = {c}"],retrieved_concepts=["inverse operations","function inversion"],intermediate_steps=[f"divide both sides by {a} to obtain x + ({b}) = {inner}"],unresolved_subgoals=[f"remove the additive offset {b}"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question,str(x),state,"linear_composed",3,SKILL_DECOMPOSE)

def _ood_three_step(rng:random.Random)->GeneratedProblem:
    x=rng.choice([i for i in range(-12,13) if i!=0]);a=rng.choice([i for i in range(-8,9) if i not in (0,1,-1)]);c=rng.choice([i for i in range(-8,9) if i not in (a,0)]);b=rng.randint(-10,10);d=(a-c)*x+b
    question=f"Solve for x: {a}*x + ({b}) = {c}*x + ({d})"
    coeff=a-c;offset=d-b
    state=MathWorkingState(domain="algebra",operation="variables_both_sides",variables={"x":"unknown"},constraints=[f"{a}*x + ({b}) = {c}*x + ({d}"],retrieved_concepts=["collect like terms","inverse operations","equality preservation"],intermediate_steps=[f"collect x terms to obtain {coeff}*x = {offset}"],unresolved_subgoals=[f"isolate x by dividing by {coeff}"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question,str(x),state,"variables_both_sides",4,SKILL_DECOMPOSE)

# --- v0.5.2 OOD splits ---

def _operand_ood(rng:random.Random)->GeneratedProblem:
    """Large operands: 1000-99999."""
    x=rng.randint(1000,99999);b=rng.randint(1000,99999);c=x+b
    question=f"Solve for x: x + ({b}) = {c}"
    state=MathWorkingState(domain="algebra",operation="operand_ood",variables={"x":"unknown"},constraints=[question.replace("Solve for x: ","")],retrieved_concepts=["inverse operations"],intermediate_steps=[],unresolved_subgoals=[f"apply the inverse of adding {b} to both sides"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question,str(x),state,"operand_ood",2,SKILL_SUBTRACT)

def _structural_ood(rng:random.Random)->GeneratedProblem:
    """Two-level nesting: (a op b) op c."""
    a=rng.randint(-50,50);b=rng.randint(-50,50);c=rng.randint(-20,20)
    inner=a+b;result=inner*c
    question=f"Solve for x: (a + b) * c = {result}, where a={a}, b={b}. What is c?"
    # Actually, let's make it a proper equation to solve
    a=rng.randint(1,50);b=rng.randint(1,50);c=rng.randint(2,20)
    inner=a+b;target=inner*c
    question=f"Find c: ({a} + {b}) * c = {target}"
    state=MathWorkingState(domain="algebra",operation="structural_ood",variables={"c":"unknown"},constraints=[f"({a} + {b}) * c = {target}"],retrieved_concepts=["distribution","inverse operations"],intermediate_steps=[f"compute inner sum: {a} + {b} = {inner}"],unresolved_subgoals=[f"divide {target} by {inner} to find c"])
    state.assert_no_direct_answer_encoding(str(c))
    return GeneratedProblem(question,str(c),state,"structural_ood",3,SKILL_DECOMPOSE)

def _operator_ood(rng:random.Random)->GeneratedProblem:
    """Modular arithmetic: train on + - *, test on %."""
    a=rng.randint(10,999);b=rng.randint(2,50)
    result=a%b
    question=f"Compute: {a} mod {b}"
    state=MathWorkingState(domain="arithmetic",operation="modular",variables={"a":a,"b":b},constraints=[f"{a} mod {b}"],retrieved_concepts=["modular arithmetic","division remainder"],intermediate_steps=[],unresolved_subgoals=[f"compute the remainder of {a} divided by {b}"])
    state.assert_no_direct_answer_encoding(str(result))
    return GeneratedProblem(question,str(result),state,"operator_ood",2,SKILL_DIVIDE)

def _distractor_ood(rng:random.Random)->GeneratedProblem:
    """Add irrelevant numbers and NL clutter."""
    x=rng.randint(-20,20);b=rng.choice([i for i in range(-15,16) if i!=0]);c=x+b
    distractors=[rng.randint(-100,100) for _ in range(3)]
    question=f"Given the numbers {distractors[0]}, {distractors[1]}, and {distractors[2]}, solve for x: x + ({b}) = {c}. Ignore the other numbers."
    state=MathWorkingState(domain="algebra",operation="distractor_linear",variables={"x":"unknown"},constraints=[f"x + ({b}) = {c}"],retrieved_concepts=["inverse operations","ignore distractors"],intermediate_steps=[],unresolved_subgoals=[f"apply the inverse of adding {b} to both sides"])
    state.assert_no_direct_answer_encoding(str(x))
    return GeneratedProblem(question,str(x),state,"distractor_ood",2,SKILL_SUBTRACT)

def _counterfactual(rng:random.Random)->GeneratedProblem:
    """Same surface structure, different answer."""
    x=rng.randint(-20,20);b=rng.choice([i for i in range(-15,16) if i!=0]);c=x+b
    # Use a different x to get a different answer with same structure
    x2=rng.choice([i for i in range(-20,21) if i!=x]);c2=x2+b
    question=f"Solve for x: x + ({b}) = {c2}"
    state=MathWorkingState(domain="algebra",operation="counterfactual_linear",variables={"x":"unknown"},constraints=[f"x + ({b}) = {c2}"],retrieved_concepts=["inverse operations"],intermediate_steps=[],unresolved_subgoals=[f"apply the inverse of adding {b} to both sides"])
    state.assert_no_direct_answer_encoding(str(x2))
    return GeneratedProblem(question,str(x2),state,"counterfactual",1,SKILL_SUBTRACT)

def _adversarial(rng:random.Random)->GeneratedProblem:
    """Same operands, different operators: 12 + 5, 12 * 5, 12 - 5."""
    a=rng.randint(1,50);b=rng.randint(1,50)
    op=rng.choice(["+","-","*"])
    if op=="+": result=a+b;skill=SKILL_ADD;op_name="addition"
    elif op=="-": result=a-b;skill=SKILL_SUBTRACT;op_name="subtraction"
    else: result=a*b;skill=SKILL_MULTIPLY;op_name="multiplication"
    question=f"Compute: {a} {op} {b}"
    state=MathWorkingState(domain="arithmetic",operation=op_name,variables={"a":a,"b":b},constraints=[f"{a} {op} {b}"],retrieved_concepts=[op_name],intermediate_steps=[],unresolved_subgoals=[f"compute {a} {op} {b}"])
    state.assert_no_direct_answer_encoding(str(result))
    return GeneratedProblem(question,str(result),state,"adversarial",1,skill)

GENERATORS={
    "train":[_linear_one_step,_linear_two_step],
    "iid":[_linear_one_step,_linear_two_step],
    "composition":[_linear_composed],
    "ood":[_ood_three_step],
    "operand_ood":[_operand_ood],
    "structural_ood":[_structural_ood],
    "operator_ood":[_operator_ood],
    "distractor_ood":[_distractor_ood],
    "counterfactual":[_counterfactual],
    "adversarial":[_adversarial],
}

def generate_example(split:str,index:int,seed:int=1337)->MathExample:
    split_offsets={"train":11,"iid":23,"composition":37,"ood":53,"operand_ood":67,"structural_ood":71,"operator_ood":79,"distractor_ood":83,"counterfactual":89,"adversarial":97}
    if split not in split_offsets: raise ValueError(f"Unknown split: {split}")
    rng=random.Random(seed*1_000_003+index*97+split_offsets[split]);funcs=GENERATORS[split]
    for _ in range(100):
        try:
            p=rng.choice(funcs)(rng)
            return MathExample(example_id=f"{split}-{index}",split=split,question=p.question,answer=p.answer,state=p.state,template=p.template,difficulty=p.difficulty,skill_label=p.skill_label)
        except ValueError: continue
    raise RuntimeError("Could not generate leakage-free example after 100 attempts")
