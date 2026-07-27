from __future__ import annotations

import inspect
from pathlib import Path

try:
    from scripts import generate_v0_outputs as generation
except ModuleNotFoundError:
    import generate_v0_outputs as generation


def main() -> None:
    print("generate_v0_outputs module:", generation.__file__)
    print("\n_load_memory source:\n")
    print(inspect.getsource(generation._load_memory))

    default = Path("data/v0_memory/procedures.jsonl")
    if default.exists():
        memory = generation._load_memory(default)
        print(f"\nLoaded procedures: {len(memory.procedures)}")
        for proc in memory.procedures:
            print(f"- {proc.procedure_id}: {proc.capability_id}")
    else:
        print(f"\n{default} does not exist; run scripts/build_v0_memory.py first.")


if __name__ == "__main__":
    main()
