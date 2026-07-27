from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

# Load generate_v0_outputs.py by file path so this diagnostic does not depend
# on the scripts package being importable. V037-002: avoid `from scripts...`.
_gen_path = Path(__file__).resolve().parent / "generate_v0_outputs.py"
_spec = importlib.util.spec_from_file_location("_integrity_gen_v0", _gen_path)
generation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generation)


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
