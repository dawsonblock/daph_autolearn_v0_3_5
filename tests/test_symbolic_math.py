import pytest

from daph_learning.tools.symbolic_math import (
    ResourceLimitError,
    UnsafeExpressionError,
    execute_task,
    safe_eval_int_expr,
)


def test_safe_integer_arithmetic():
    assert safe_eval_int_expr("3591 - -5519") == 9110
    assert safe_eval_int_expr("(125 + 43) * 11") == 1848
    assert safe_eval_int_expr("345 * 821") == 283245
    assert safe_eval_int_expr("4829 * 392") == 1892968


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('id')",
        "(1).__class__",
        "x + 1",
        "[x for x in [1]]",
        "lambda: 1",
        "1.5 + 2",
        "True + 1",
        "2 ** 100",
        "4 / 2",
    ],
)
def test_unsafe_constructs_rejected(expr):
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        safe_eval_int_expr(expr)


def test_structured_inputs_preferred():
    task = {
        "task_id": "x",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": 3591, "b": -5519, "op": "-"},
        "specification": "Compute 1 + 1. Return only the integer.",
    }
    assert execute_task(task) == 9110


def test_modular_multiplication():
    task = {
        "capability_ids": ["modular_multiplication"],
        "inputs": {"a": 12345, "b": 6789, "modulus": 97},
    }
    assert execute_task(task) == (12345 * 6789) % 97


def test_mod_alias_supported():
    task = {
        "capability_ids": ["modular_multiplication"],
        "inputs": {"a": 12, "b": 13, "mod": 17},
    }
    assert execute_task(task) == (12 * 13) % 17


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        safe_eval_int_expr("5 % 0")
