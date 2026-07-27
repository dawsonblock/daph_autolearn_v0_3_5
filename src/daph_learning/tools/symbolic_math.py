from __future__ import annotations

import ast
import operator
import re
from typing import Any, Mapping


class SymbolicMathError(ValueError):
    pass


class UnsafeExpressionError(SymbolicMathError):
    pass


class UnsupportedOperationError(SymbolicMathError):
    pass


class ResourceLimitError(SymbolicMathError):
    pass


MAX_EXPR_CHARS = 256
MAX_AST_NODES = 32
MAX_AST_DEPTH = 12
MAX_INT_DIGITS = 64
MAX_RESULT_BITS = 4096

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _require_int(value: Any, name: str = "value") -> int:
    # bool is a subclass of int; require the exact type.
    if type(value) is not int:
        raise UnsafeExpressionError(f"{name} must be an integer, got {type(value).__name__}")
    if len(str(abs(value))) > MAX_INT_DIGITS:
        raise ResourceLimitError(f"{name} exceeds {MAX_INT_DIGITS} decimal digits")
    return value


def _check_result(value: int) -> int:
    if type(value) is not int:
        raise SymbolicMathError("symbolic evaluator produced a non-integer result")
    if value.bit_length() > MAX_RESULT_BITS:
        raise ResourceLimitError(f"result exceeds {MAX_RESULT_BITS} bits")
    return value


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_depth(child) for child in children)


def _validate_tree(tree: ast.AST) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ResourceLimitError(f"expression exceeds {MAX_AST_NODES} AST nodes")
    if _depth(tree) > MAX_AST_DEPTH:
        raise ResourceLimitError(f"expression exceeds AST depth {MAX_AST_DEPTH}")


def _eval_node(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return _require_int(node.value, "constant")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _check_result(
            _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
        )

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Mod, ast.FloorDiv)) and right == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        return _check_result(_ALLOWED_BINOPS[type(node.op)](left, right))

    raise UnsafeExpressionError(
        f"disallowed expression node: {type(node).__name__}"
    )


def safe_eval_int_expr(expr: str) -> int:
    """Bounded fallback evaluator for a tiny exact-integer expression grammar.

    Deliberately unsupported: names, calls, attributes, subscripts, lists,
    comprehensions, floats, booleans, true division, exponentiation, bitwise
    operators, comparisons, lambdas, and assignment expressions.
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be str")
    if len(expr) > MAX_EXPR_CHARS:
        raise ResourceLimitError(f"expression exceeds {MAX_EXPR_CHARS} characters")
    tree = ast.parse(expr, mode="eval")
    _validate_tree(tree)
    return _eval_node(tree.body)


def execute_integer_arithmetic(inputs: Mapping[str, Any]) -> int:
    a = _require_int(inputs.get("a"), "a")
    b = _require_int(inputs.get("b"), "b")
    op = inputs.get("op")
    if op == "+":
        return _check_result(a + b)
    if op == "-":
        return _check_result(a - b)
    if op == "*":
        return _check_result(a * b)
    if op == "%":
        if b == 0:
            raise ZeroDivisionError("modulo by zero")
        return _check_result(a % b)
    if op == "//":
        if b == 0:
            raise ZeroDivisionError("floor division by zero")
        return _check_result(a // b)
    raise UnsupportedOperationError(f"unsupported integer operation: {op!r}")


def execute_modular_multiplication(inputs: Mapping[str, Any]) -> int:
    a = _require_int(inputs.get("a"), "a")
    b = _require_int(inputs.get("b"), "b")
    modulus = inputs.get("modulus", inputs.get("mod", inputs.get("m", inputs.get("c"))))
    modulus = _require_int(modulus, "modulus")
    if modulus == 0:
        raise ZeroDivisionError("modulus cannot be zero")
    return _check_result((a * b) % modulus)



def modular_inputs_from_specification(specification: str) -> dict[str, int]:
    """Extract only the narrow `a * b mod c` textual form.

    This is a fallback for supported modular tasks that lack structured inputs;
    it is not a general natural-language parser.
    """
    match = re.search(
        r"\bCompute\s+\(?\s*(-?\d+)\s*\*\s*(-?\d+)\s*\)?\s*(?:mod|%)\s*(-?\d+)(?:\.|\s|$)",
        specification,
        flags=re.IGNORECASE,
    )
    if not match:
        raise SymbolicMathError("could not extract modular multiplication operands")
    return {
        "a": _require_int(int(match.group(1)), "a"),
        "b": _require_int(int(match.group(2)), "b"),
        "modulus": _require_int(int(match.group(3)), "modulus"),
    }

def expression_from_specification(specification: str) -> str:
    """Extract the arithmetic prefix immediately following ``Compute``.

    Parsing stops before alphabetic trailing instructions, a period, or end of
    input. The extracted text is still passed through the bounded AST evaluator;
    this function is only a narrow text boundary isolator, not an evaluator.
    """
    match = re.search(
        r"\bCompute\s+([0-9+\-*%/()\s]+?)(?=[a-zA-Z]|\.|$)",
        specification,
        flags=re.IGNORECASE,
    )
    if not match or not match.group(1).strip():
        raise SymbolicMathError("could not extract a safe arithmetic expression")
    return match.group(1).strip()


def execute_task(task: Mapping[str, Any]) -> int:
    caps = set(task.get("capability_ids", []))
    inputs = task.get("inputs") or {}

    if "modular_multiplication" in caps:
        return execute_modular_multiplication(inputs)

    if "integer_arithmetic" in caps and {"a", "b", "op"} <= set(inputs):
        return execute_integer_arithmetic(inputs)

    # Fallback only for explicitly arithmetic capabilities.
    if caps & {"integer_arithmetic", "modular_multiplication"}:
        expr = expression_from_specification(str(task.get("specification", "")))
        return safe_eval_int_expr(expr)

    raise UnsupportedOperationError(
        f"task capabilities are not supported by symbolic executor: {sorted(caps)}"
    )
