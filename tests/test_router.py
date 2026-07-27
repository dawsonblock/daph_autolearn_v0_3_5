from daph_learning.routing.arithmetic_router import RoutingConfig, route_task


def task(op="+", cap="integer_arithmetic"):
    return {
        "task_id": "t",
        "capability_ids": [cap],
        "inputs": {"a": 2, "b": 3, "op": op},
        "specification": f"Compute 2 {op} 3.",
    }


def test_baseline_is_llm():
    d = route_task(task("*"), False, RoutingConfig("baseline"))
    assert d.backend == "llm"


def test_auto_multiply_is_symbolic():
    d = route_task(task("*"), False, RoutingConfig("auto"))
    assert d.backend == "symbolic"
    assert d.steering_policy is None


def test_auto_addition_stays_llm():
    d = route_task(task("+"), False, RoutingConfig("auto"))
    assert d.backend == "llm"


def test_hybrid_supported_is_symbolic():
    d = route_task(task("-"), False, RoutingConfig("hybrid"))
    assert d.backend == "symbolic"
