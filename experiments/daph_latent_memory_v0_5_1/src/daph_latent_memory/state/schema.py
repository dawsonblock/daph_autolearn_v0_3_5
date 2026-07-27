from __future__ import annotations
import re
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class MathWorkingState(BaseModel):
    domain: str
    operation: str
    variables: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    retrieved_concepts: list[str] = Field(default_factory=list)
    intermediate_steps: list[str] = Field(default_factory=list)
    unresolved_subgoals: list[str] = Field(default_factory=list)
    answer: str | None = None

    @model_validator(mode="after")
    def enforce_no_answer(self) -> "MathWorkingState":
        if self.answer not in (None, ""):
            raise ValueError("Final answers are forbidden in process-only working state")
        return self

    def process_text(self) -> str:
        parts = [f"Domain: {self.domain}",f"Operation: {self.operation}",f"Variables: {self.variables}","Constraints: " + " | ".join(self.constraints),"Concepts: " + " | ".join(self.retrieved_concepts),"Intermediate: " + " | ".join(self.intermediate_steps),"Unresolved: " + " | ".join(self.unresolved_subgoals)]
        return "\n".join(parts)

    def assert_no_direct_answer_encoding(self, final_answer: str, variable: str = "x") -> None:
        answer = re.escape(str(final_answer).strip())
        if not answer: return
        text = self.process_text().lower()
        patterns=[rf"\banswer\s*(?:is|=|:)\s*{answer}(?:\b|$)",rf"\bsolution\s*(?:is|=|:)\s*{answer}(?:\b|$)",rf"\b{re.escape(variable.lower())}\s*=\s*{answer}(?:\b|$)"]
        if any(re.search(pattern,text) for pattern in patterns): raise ValueError(f"Direct answer leakage detected for final answer {final_answer!r}")

    def assert_answer_not_present(self, final_answer: str) -> None:
        self.assert_no_direct_answer_encoding(final_answer)


class MathExample(BaseModel):
    example_id: str
    split: Literal["train", "iid", "composition", "ood"]
    question: str
    answer: str
    state: MathWorkingState
    template: str
    difficulty: int
