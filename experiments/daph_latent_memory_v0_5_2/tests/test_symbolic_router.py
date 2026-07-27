from daph_latent_memory.router.symbolic_router import SymbolicEngine, CapabilityRouter, TaskType

def test_symbolic_classify_arithmetic():
    engine = SymbolicEngine()
    assert engine.classify("Compute: 5 + 3") == TaskType.EXACT_SYMBOLIC
    assert engine.classify("3 * 4") == TaskType.EXACT_SYMBOLIC
    assert engine.classify("Solve for x: x + (3) = 7") == TaskType.EXACT_SYMBOLIC

def test_symbolic_classify_reasoning():
    engine = SymbolicEngine()
    assert engine.classify("Explain why the sky is blue") == TaskType.REASONING

def test_symbolic_evaluate_addition():
    engine = SymbolicEngine()
    result = engine.evaluate("Compute: 5 + 3")
    assert result.success
    assert result.value == "8"

def test_symbolic_evaluate_subtraction():
    engine = SymbolicEngine()
    result = engine.evaluate("Compute: 10 - 4")
    assert result.success
    assert result.value == "6"

def test_symbolic_evaluate_multiplication():
    engine = SymbolicEngine()
    result = engine.evaluate("Compute: 6 * 7")
    assert result.success
    assert result.value == "42"

def test_symbolic_evaluate_mod():
    engine = SymbolicEngine()
    result = engine.evaluate("17 mod 5")
    assert result.success
    assert result.value == "2"

def test_symbolic_evaluate_solve_x():
    engine = SymbolicEngine()
    result = engine.evaluate("Solve for x: x + (3) = 7")
    assert result.success
    assert result.value == "4"

def test_symbolic_evaluate_two_step():
    engine = SymbolicEngine()
    result = engine.evaluate("Solve for x: 2*x + (3) = 11")
    assert result.success
    assert result.value == "4"

def test_symbolic_verify():
    engine = SymbolicEngine()
    assert engine.verify("Compute: 5 + 3", "8", "8")
    assert not engine.verify("Compute: 5 + 3", "9", "8")

def test_symbolic_division_by_zero():
    engine = SymbolicEngine()
    result = engine.evaluate("Compute: 5 / 0")
    assert not result.success

def test_capability_router_routes_symbolic():
    router = CapabilityRouter()
    task_type, symbolic_result = router.route("Compute: 5 + 3")
    assert task_type == TaskType.EXACT_SYMBOLIC
    assert symbolic_result is not None
    assert symbolic_result.success

def test_capability_router_routes_reasoning():
    router = CapabilityRouter()
    task_type, symbolic_result = router.route("Explain quantum mechanics")
    assert task_type == TaskType.REASONING
    assert symbolic_result is None

def test_capability_router_answer():
    router = CapabilityRouter()
    answer, task_type, was_symbolic = router.answer("Compute: 5 + 3")
    assert answer == "8"
    assert was_symbolic

def test_capability_router_fallback():
    """If symbolic fails to parse, fall back to reasoning."""
    router = CapabilityRouter()
    answer, task_type, was_symbolic = router.answer("Compute: something complex")
    assert not was_symbolic
