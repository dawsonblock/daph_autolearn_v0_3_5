import json
from pathlib import Path

from daph_learning.memory.procedural import ProceduralMemory


def test_memory_load_and_retrieve(tmp_path: Path):
    p = tmp_path / "memory.jsonl"
    p.write_text(
        json.dumps(
            {
                "procedure_id": "rule_0",
                "capability_id": "integer_arithmetic",
                "trigger_terms": ["math"],
                "action": "verify",
                "confidence": 1.0,
                "positive_evidence": 1,
                "negative_evidence": 0,
                "source_episode_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    memory = ProceduralMemory.from_jsonl(p)
    assert len(memory.procedures) == 1
    found = memory.retrieve(["integer_arithmetic"], "Compute 4 * 5.")
    assert [x.procedure_id for x in found] == ["rule_0"]
