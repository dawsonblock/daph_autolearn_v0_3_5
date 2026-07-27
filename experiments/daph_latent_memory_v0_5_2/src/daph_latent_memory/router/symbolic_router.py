from __future__ import annotations
import re
from enum import Enum
from dataclasses import dataclass
from typing import Any

class TaskType(str,Enum):
    EXACT_SYMBOLIC="exact_symbolic"
    REASONING="reasoning"
    UNKNOWN="unknown"

class SymbolicOp(str,Enum):
    ADD="+"
    SUB="-"
    MUL="*"
    DIV="/"
    MOD="%"
    FLOORDIV="//"
    POW="**"

# Operators that can be exactly computed symbolically
SYMBOLIC_OPERATORS={SymbolicOp.ADD,SymbolicOp.SUB,SymbolicOp.MUL,SymbolicOp.DIV,SymbolicOp.MOD,SymbolicOp.FLOORDIV,SymbolicOp.POW}

@dataclass
class SymbolicResult:
    success:bool
    value:str|None=None
    error:str|None=None
    steps:list[str]=None

class SymbolicEngine:
    """Deterministic symbolic arithmetic engine.

    For exact arithmetic, symbolic execution is the answer authority.
    The latent system handles decomposition, planning, strategy selection,
    explanation, error recognition, tool selection — not deterministic
    computation.

    No LLM judge for arithmetic. Uses exact symbolic checking.
    """
    def __init__(self)->None:
        self._int_re=re.compile(r"-?\d+")

    def classify(self,question:str)->TaskType:
        """Classify a question as exact_symbolic or reasoning."""
        q=question.lower().strip()
        # Check for arithmetic patterns
        if re.search(r"\d+\s*[+\-*/%]\s*\d+",q): return TaskType.EXACT_SYMBOLIC
        if re.search(r"compute[:\s].*\d+",q): return TaskType.EXACT_SYMBOLIC
        if re.search(r"solve for x[:\s].*\d+",q): return TaskType.EXACT_SYMBOLIC
        if re.search(r"\d+\s*mod\s*\d+",q): return TaskType.EXACT_SYMBOLIC
        return TaskType.REASONING

    def evaluate(self,question:str)->SymbolicResult:
        """Evaluate a question symbolically. Returns exact result."""
        try:
            result=self._safe_eval(question)
            if result is not None:
                return SymbolicResult(success=True,value=str(result),steps=[f"Computed: {question} = {result}"])
            return SymbolicResult(success=False,error="Could not parse question",steps=[])
        except Exception as e:
            return SymbolicResult(success=False,error=str(e),steps=[])

    def _safe_eval(self,question:str)->int|float|None:
        """Safely evaluate an arithmetic expression."""
        q=question.strip()

        # Pattern: "Compute: a op b"
        m=re.match(r"compute[:\s]*(.+)",q,re.IGNORECASE)
        if m:
            expr=m.group(1).strip()
            return self._eval_expr(expr)

        # Pattern: "a op b" directly
        m=re.match(r"^(-?\d+)\s*([+\-*/%]+)\s*(-?\d+)$",q)
        if m:
            a,op,b=int(m.group(1)),m.group(2),int(m.group(3))
            return self._apply_op(a,op,b)

        # Pattern: "a mod b"
        m=re.match(r"^(-?\d+)\s*mod\s*(-?\d+)$",q,re.IGNORECASE)
        if m:
            a,b=int(m.group(1)),int(m.group(2))
            return a%b

        # Pattern: "Solve for x: x + (b) = c"
        m=re.match(r"solve for x[:\s]*x\s*([+\-])\s*\((-?\d+)\)\s*=\s*(-?\d+)",q,re.IGNORECASE)
        if m:
            op,b,c=m.group(1),int(m.group(2)),int(m.group(3))
            if op=="+": return c-b
            else: return c+b

        # Pattern: "Solve for x: a*x + (b) = c"
        m=re.match(r"solve for x[:\s]*(-?\d+)\*x\s*([+\-])\s*\((-?\d+)\)\s*=\s*(-?\d+)",q,re.IGNORECASE)
        if m:
            a,op,b,c=int(m.group(1)),m.group(2),int(m.group(3)),int(m.group(4))
            if op=="+": transformed=c-b
            else: transformed=c+b
            if a!=0: return transformed//a if transformed%a==0 else transformed/a

        # Pattern: "Find c: (a + b) * c = target"
        m=re.match(r"find c[:\s]*\((-?\d+)\s*\+\s*(-?\d+)\)\s*\*\s*c\s*=\s*(-?\d+)",q,re.IGNORECASE)
        if m:
            a,b,target=int(m.group(1)),int(m.group(2)),int(m.group(3))
            inner=a+b
            if inner!=0: return target//inner if target%inner==0 else target/inner

        return None

    def _eval_expr(self,expr:str)->int|float|None:
        """Evaluate a simple arithmetic expression."""
        expr=expr.strip()
        m=re.match(r"^(-?\d+)\s*([+\-*/%]+)\s*(-?\d+)$",expr)
        if m:
            a,op,b=int(m.group(1)),m.group(2),int(m.group(3))
            return self._apply_op(a,op,b)
        return None

    def _apply_op(self,a:int|float,op:str,b:int|float)->int|float:
        if op=="+": return a+b
        elif op=="-": return a-b
        elif op=="*": return a*b
        elif op=="/":
            if b==0: raise ZeroDivisionError("Division by zero")
            return a/b if a%b!=0 else a//b
        elif op=="%": return a%b
        elif op=="//": return a//b
        elif op=="**": return a**b
        raise ValueError(f"Unknown operator: {op}")

    def verify(self,question:str,predicted_answer:str,expected_answer:str)->bool:
        """Verify a predicted answer against expected (deterministic, no LLM)."""
        from ..evaluation.metrics import normalize_answer
        return normalize_answer(predicted_answer)==normalize_answer(expected_answer)


class CapabilityRouter:
    """Routes tasks between symbolic engine and latent-memory LLM path.

    Architecture:
        Input
          |
        Capability Router
          |
          +-- exact symbolic task --> Symbolic Engine
          |
          +-- reasoning task --> Skill Router --> Latent Memory --> LLM

    For exact arithmetic, symbolic execution is the answer authority.
    """
    def __init__(self,engine:SymbolicEngine|None=None)->None:
        self.engine=engine or SymbolicEngine()

    def route(self,question:str)->tuple[TaskType,SymbolicResult|None]:
        """Classify and route a question.

        Returns (task_type, symbolic_result_if_exact).
        """
        task_type=self.engine.classify(question)
        if task_type==TaskType.EXACT_SYMBOLIC:
            result=self.engine.evaluate(question)
            if result.success:
                return task_type,result
            # If symbolic fails, fall back to reasoning
            return TaskType.REASONING,None
        return task_type,None

    def answer(self,question:str)->tuple[str,TaskType,bool]:
        """Get an answer for a question, routing appropriately.

        Returns (answer, task_type, was_symbolic).
        """
        task_type,symbolic_result=self.route(question)
        if symbolic_result and symbolic_result.success:
            return symbolic_result.value,task_type,True
        return "",task_type,False
