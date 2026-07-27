from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class SymbolicAction:
    name: str
    arguments: Mapping[str, Any]
    output_var: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    task_id: str
    route: Literal["llm", "symbolic"]
    planner_source: Literal[
        "structured_input", "rule_router", "llm", "steered_llm"
    ]
    actions: tuple[SymbolicAction, ...] = ()
    reason_code: str = ""
    result_var: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    value: int | str
    verified: bool
    actions_executed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
