from __future__ import annotations

from pathlib import Path

from daph_learning.memory.procedural import Procedure, ProceduralMemory


def main() -> None:
    out = Path("data/v0_memory")
    out.mkdir(parents=True, exist_ok=True)

    treatment = ProceduralMemory(
        [
            Procedure(
                procedure_id="rule_0",
                capability_id="integer_arithmetic",
                trigger_terms=["math", "integer_arithmetic"],
                action=(
                    "For integer arithmetic, identify the operation and signs. "
                    "Use a deterministic exact tool when the routing policy requests one."
                ),
            ),
            Procedure(
                procedure_id="rule_1",
                capability_id="modular_multiplication",
                trigger_terms=["math", "modular_multiplication"],
                action=(
                    "For modular multiplication, multiply exactly and reduce by the modulus. "
                    "Do not approximate."
                ),
            ),
        ]
    )
    treatment.to_jsonl(out / "procedures.jsonl")

    sham = ProceduralMemory(
        [
            Procedure(
                procedure_id="sham_0",
                capability_id="integer_arithmetic",
                trigger_terms=["math", "integer_arithmetic"],
                action=(
                    "The word digit comes from Latin digitus, referring to a finger or toe."
                ),
            ),
            Procedure(
                procedure_id="sham_1",
                capability_id="modular_multiplication",
                trigger_terms=["math", "modular_multiplication"],
                action=(
                    "The history of zero spans several ancient mathematical traditions."
                ),
            ),
        ]
    )
    sham.to_jsonl(out / "sham_procedures.jsonl")
    print("Wrote treatment and sham procedural-memory files.")


if __name__ == "__main__":
    main()
