from __future__ import annotations
import copy
import random
import re
from .schema import MathWorkingState

_INT_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+")


def _perturb_first_integer(text: str, rng: random.Random) -> str:
    match = _INT_RE.search(text)
    if not match:
        return text + " [incorrect operation: multiply instead of invert]"
    value = int(match.group(0))
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    wrong = value + delta
    return text[: match.start()] + str(wrong) + text[match.end() :]


def corrupt_state(state: MathWorkingState, rng: random.Random) -> MathWorkingState:
    """Create a semantically wrong state rather than merely labeling content as corrupted."""
    out = copy.deepcopy(state.model_dump())
    if out["intermediate_steps"]:
        idx = rng.randrange(len(out["intermediate_steps"]))
        out["intermediate_steps"][idx] = _perturb_first_integer(out["intermediate_steps"][idx], rng)
    elif out["unresolved_subgoals"]:
        idx = rng.randrange(len(out["unresolved_subgoals"]))
        original = out["unresolved_subgoals"][idx]
        if "inverse" in original.lower() or "remove" in original.lower():
            out["unresolved_subgoals"][idx] = "apply the same operation again instead of its inverse"
        elif "divide" in original.lower():
            out["unresolved_subgoals"][idx] = original.lower().replace("divide", "multiply")
        else:
            out["unresolved_subgoals"][idx] = _perturb_first_integer(original, rng)
    elif out["constraints"]:
        idx = rng.randrange(len(out["constraints"]))
        out["constraints"][idx] = _perturb_first_integer(out["constraints"][idx], rng)
    else:
        out["unresolved_subgoals"].append("apply an invalid inverse operation")
    return MathWorkingState(**out)
