from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Procedure:
    procedure_id: str
    capability_id: str
    trigger_terms: list[str]
    action: str
    confidence: float = 1.0
    positive_evidence: int = 0
    negative_evidence: int = 0
    source_episode_ids: list[str] = field(default_factory=list)


class ProceduralMemory:
    """Small deterministic procedural-memory store used by the V0 experiment."""

    def __init__(self, procedures: Iterable[Procedure] | None = None) -> None:
        self.procedures: list[Procedure] = list(procedures or [])

    def upsert(self, procedure: Procedure) -> None:
        for idx, current in enumerate(self.procedures):
            if current.procedure_id == procedure.procedure_id:
                self.procedures[idx] = procedure
                return
        self.procedures.append(procedure)

    def retrieve(
        self,
        capability_ids: Iterable[str],
        specification: str = "",
        *,
        limit: int = 8,
    ) -> list[Procedure]:
        caps = set(capability_ids)
        spec = specification.casefold()
        scored: list[tuple[float, Procedure]] = []
        for proc in self.procedures:
            if proc.capability_id not in caps:
                continue
            trigger_hits = sum(
                1 for term in proc.trigger_terms
                if term and term.casefold() in spec
            )
            # Capability match is the primary gate. Trigger hits only rank ties.
            score = proc.confidence + 0.05 * trigger_hits
            scored.append((score, proc))
        scored.sort(key=lambda x: (-x[0], x[1].procedure_id))
        return [p for _, p in scored[:limit]]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ProceduralMemory":
        path = Path(path)
        memory = cls()
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    memory.upsert(Procedure(**data))
                except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid Procedure JSONL at {path}:{line_no}: {exc}"
                    ) from exc
        return memory

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for proc in self.procedures:
                fh.write(json.dumps(asdict(proc), ensure_ascii=False) + "\n")
