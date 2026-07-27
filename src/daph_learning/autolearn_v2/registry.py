"""Policy registry with immutable lineage (AutoLearn v2, Phase 8).

Every candidate and accepted policy is versioned and persisted with full
provenance:

* policy ID + parent policy ID
* creation timestamp
* base model hash + tokenizer hash
* layer configuration
* steering vector hash
* training experience IDs
* replay sample hash
* utility configuration
* random seed
* validation metrics
* acceptance decision
* source commit + environment metadata

Accepted policy artifacts are **immutable**: once a policy is accepted,
its on-disk record is never overwritten. The registry supports
``register_candidate``, ``accept``, ``reject``, and ``rollback``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .acceptance import CandidateDecision
from .policies.base import PolicyMetrics
from .policies.static_vector import SingleVectorPolicy


class RegistryError(Exception):
    """Base class for registry integrity violations."""


class PolicyNotFoundError(RegistryError):
    pass


class PolicyAlreadyAcceptedError(RegistryError):
    pass


@dataclass(frozen=True)
class PolicyRecord:
    """An immutable record describing one policy's lineage and status."""

    policy: SingleVectorPolicy
    status: str  # "candidate" | "accepted" | "rejected" | "rolled_back"
    parent_policy_id: str | None
    created_at: str
    accepted_at: str | None = None
    rejected_at: str | None = None
    acceptance_decision: CandidateDecision | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "status": self.status,
            "parent_policy_id": self.parent_policy_id,
            "created_at": self.created_at,
            "accepted_at": self.accepted_at,
            "rejected_at": self.rejected_at,
            "acceptance_decision": self.acceptance_decision.to_dict() if self.acceptance_decision else None,
            "environment": dict(self.environment),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyRecord":
        decision = None
        if d.get("acceptance_decision"):
            dec = d["acceptance_decision"]
            active_m = PolicyMetrics(**dec["active_metrics"]) if dec.get("active_metrics") else None
            cand_m = PolicyMetrics(**dec["candidate_metrics"]) if dec.get("candidate_metrics") else None
            decision = CandidateDecision(
                accepted=dec["accepted"],
                reason_codes=list(dec.get("reason_codes", [])),
                active_metrics=active_m,
                candidate_metrics=cand_m,
                utility_gain=dec.get("utility_gain", 0.0),
                accuracy_delta=dec.get("accuracy_delta", 0.0),
                abstain_delta=dec.get("abstain_delta", 0.0),
                displacement=dec.get("displacement", 0.0),
                detail=dec.get("detail"),
            )
        return cls(
            policy=SingleVectorPolicy.from_dict(d["policy"]),
            status=d["status"],
            parent_policy_id=d["parent_policy_id"],
            created_at=d["created_at"],
            accepted_at=d.get("accepted_at"),
            rejected_at=d.get("rejected_at"),
            acceptance_decision=decision,
            environment=dict(d.get("environment", {})),
        )


class PolicyRegistry:
    """An append-only registry of policy records.

    The registry keeps an in-memory index and optionally persists each
    record as an immutable JSON file under ``store_dir``. Accepted
    policies are written once and never overwritten.
    """

    def __init__(self, store_dir: str | None = None) -> None:
        self.store_dir = store_dir
        self._records: dict[str, PolicyRecord] = {}
        self._accepted_order: list[str] = []
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
            self._load_existing()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _record_path(self, policy_id: str) -> str:
        assert self.store_dir is not None
        return os.path.join(self.store_dir, f"{policy_id}.json")

    def _accepted_path(self, policy_id: str) -> str:
        assert self.store_dir is not None
        return os.path.join(self.store_dir, f"{policy_id}.accepted.json")

    def _rejected_path(self, policy_id: str) -> str:
        assert self.store_dir is not None
        return os.path.join(self.store_dir, f"{policy_id}.rejected.json")

    def _load_existing(self) -> None:
        assert self.store_dir is not None
        # Load candidate files first, then accepted/rejected companions so
        # the terminal status wins when both exist for the same policy_id.
        files = sorted(os.listdir(self.store_dir))
        candidate_files = [f for f in files if f.endswith(".json") and ".accepted." not in f and ".rejected." not in f and ".rollback." not in f]
        companion_files = [f for f in files if ".accepted." in f or ".rejected." in f or ".rollback." in f]
        for fname in candidate_files + companion_files:
            with open(os.path.join(self.store_dir, fname), "r", encoding="utf-8") as f:
                d = json.load(f)
            rec = PolicyRecord.from_dict(d)
            self._records[rec.policy.policy_id] = rec
            if rec.status == "accepted":
                self._accepted_order.append(rec.policy.policy_id)
        # Deduplicate accepted order and preserve chronological order.
        seen = set()
        deduped = []
        for pid in self._accepted_order:
            if pid not in seen:
                seen.add(pid)
                deduped.append(pid)
        self._accepted_order = deduped
        self._accepted_order.sort(key=lambda pid: self._records[pid].created_at)

    def _persist(self, record: PolicyRecord) -> None:
        if not self.store_dir:
            return
        path = self._record_path(record.policy.policy_id)
        # Accepted records are immutable: never overwrite once accepted.
        accepted_path = self._accepted_path(record.policy.policy_id)
        if os.path.exists(accepted_path):
            raise PolicyAlreadyAcceptedError(
                f"policy {record.policy.policy_id} is accepted; "
                f"its record is immutable"
            )
        # Candidate / rejected records may be updated (e.g. a rejected
        # candidate re-registered in a later iteration).
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def register_candidate(
        self,
        policy: SingleVectorPolicy,
        *,
        created_at: str | None = None,
        environment: dict[str, Any] | None = None,
    ) -> PolicyRecord:
        """Register a candidate policy (not yet accepted)."""
        ts = created_at or _now_iso()
        record = PolicyRecord(
            policy=policy,
            status="candidate",
            parent_policy_id=policy.parent_policy_id,
            created_at=ts,
            environment=dict(environment or {}),
        )
        if policy.policy_id in self._records:
            # Idempotent re-registration of the same candidate is allowed.
            # A previously rejected candidate may be re-registered in a
            # later iteration (the active policy may not have changed, so
            # the same candidate is reproduced). This is scientifically
            # correct: the same candidate is being reconsidered.
            existing = self._records[policy.policy_id]
            if existing.status == "accepted":
                raise RegistryError(
                    f"policy {policy.policy_id} already exists with status accepted"
                )
            # Reset to "candidate" status for re-evaluation.
            record = PolicyRecord(
                policy=policy,
                status="candidate",
                parent_policy_id=policy.parent_policy_id,
                created_at=ts,
                environment=dict(environment or {}),
            )
            self._records[policy.policy_id] = record
            self._persist(record)
            return record
        self._records[policy.policy_id] = record
        self._persist(record)
        return record

    def accept(
        self,
        policy: SingleVectorPolicy,
        decision: CandidateDecision,
        *,
        accepted_at: str | None = None,
    ) -> PolicyRecord:
        """Mark a candidate policy as accepted (immutable thereafter)."""
        if not decision.accepted:
            raise RegistryError(
                "cannot accept a policy whose CandidateDecision.accepted is False"
            )
        existing = self._records.get(policy.policy_id)
        if existing is None:
            # Register then accept in one step.
            existing = self.register_candidate(policy)
        if existing.status == "accepted":
            raise PolicyAlreadyAcceptedError(
                f"policy {policy.policy_id} is already accepted"
            )
        if existing.status == "rejected":
            raise RegistryError(
                f"policy {policy.policy_id} is rejected; cannot accept"
            )
        ts = accepted_at or _now_iso()
        # Attach validation metrics to the policy before sealing.
        sealed_policy = policy
        if decision.candidate_metrics is not None and policy.validation_metrics is None:
            sealed_policy = policy.with_metrics(decision.candidate_metrics)
        record = PolicyRecord(
            policy=sealed_policy,
            status="accepted",
            parent_policy_id=existing.parent_policy_id,
            created_at=existing.created_at,
            accepted_at=ts,
            acceptance_decision=decision,
            environment=dict(existing.environment),
        )
        # Replace the in-memory candidate record with the accepted one.
        self._records[policy.policy_id] = record
        self._accepted_order.append(policy.policy_id)
        # Persist: write to a *new* file (status changed). To preserve
        # immutability of the original candidate file, the accepted
        # record is persisted under the same id only if it did not exist
        # yet; otherwise we keep the candidate file and write the
        # accepted record to a companion ``*.accepted.json`` file.
        if self.store_dir:
            self._persist_accepted(record)
        return record

    def _persist_accepted(self, record: PolicyRecord) -> None:
        assert self.store_dir is not None
        # The candidate file is immutable; write the accepted record to a
        # separate companion file so the lineage trail is complete.
        path = os.path.join(self.store_dir, f"{record.policy.policy_id}.accepted.json")
        if os.path.exists(path):
            raise PolicyAlreadyAcceptedError(
                f"accepted record {path} already exists; refusing to overwrite"
            )
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def reject(
        self,
        policy: SingleVectorPolicy,
        decision: CandidateDecision,
        *,
        rejected_at: str | None = None,
    ) -> PolicyRecord:
        """Mark a candidate policy as rejected (kept for lineage)."""
        existing = self._records.get(policy.policy_id)
        if existing is None:
            existing = self.register_candidate(policy)
        if existing.status == "accepted":
            raise RegistryError(f"policy {policy.policy_id} is accepted; cannot reject")
        ts = rejected_at or _now_iso()
        record = PolicyRecord(
            policy=policy,
            status="rejected",
            parent_policy_id=existing.parent_policy_id,
            created_at=existing.created_at,
            rejected_at=ts,
            acceptance_decision=decision,
            environment=dict(existing.environment),
        )
        self._records[policy.policy_id] = record
        if self.store_dir:
            self._persist_rejected(record)
        return record

    def _persist_rejected(self, record: PolicyRecord) -> None:
        assert self.store_dir is not None
        path = os.path.join(self.store_dir, f"{record.policy.policy_id}.rejected.json")
        if os.path.exists(path):
            return  # idempotent
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def rollback(self, to_policy_id: str) -> PolicyRecord:
        """Roll the active policy back to a previously accepted policy.

        The current active policy is *not* deleted (immutability); a new
        accepted record is created that points back to ``to_policy_id``.
        Returns the rolled-back-to record (re-marked as the active
        accepted policy).
        """
        target = self._records.get(to_policy_id)
        if target is None:
            raise PolicyNotFoundError(f"policy {to_policy_id} not found in registry")
        if target.status != "accepted":
            raise RegistryError(
                f"can only roll back to an accepted policy; {to_policy_id} is {target.status}"
            )
        # Re-accept the target as the new active policy (new timestamp).
        ts = _now_iso()
        rollback_policy = target.policy
        record = PolicyRecord(
            policy=rollback_policy,
            status="accepted",
            parent_policy_id=rollback_policy.parent_policy_id,
            created_at=target.created_at,
            accepted_at=ts,
            acceptance_decision=target.acceptance_decision,
            environment={**target.environment, "rollback_from": to_policy_id, "rollback": True},
        )
        # Append a new active marker (the same policy_id; in-memory we
        # just refresh the accepted_at). On disk we write a rollback
        # marker file.
        self._records[to_policy_id] = record
        self._accepted_order.append(to_policy_id)
        if self.store_dir:
            marker = os.path.join(self.store_dir, f"{to_policy_id}.rollback.{ts}.json")
            with open(marker, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2, sort_keys=True)
        return record

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, policy_id: str) -> PolicyRecord:
        rec = self._records.get(policy_id)
        if rec is None:
            raise PolicyNotFoundError(policy_id)
        return rec

    def active_policy_id(self) -> str | None:
        return self._accepted_order[-1] if self._accepted_order else None

    def active_policy(self) -> SingleVectorPolicy | None:
        aid = self.active_policy_id()
        if aid is None:
            return None
        return self._records[aid].policy

    def lineage(self, policy_id: str) -> list[PolicyRecord]:
        """Return the chain parent -> ... -> policy_id (root first)."""
        chain: list[PolicyRecord] = []
        cur = self._records.get(policy_id)
        while cur is not None:
            chain.append(cur)
            if cur.parent_policy_id is None:
                break
            cur = self._records.get(cur.parent_policy_id)
        chain.reverse()
        return chain

    def all_records(self) -> list[PolicyRecord]:
        return list(self._records.values())

    def accepted_records(self) -> list[PolicyRecord]:
        return [self._records[pid] for pid in self._accepted_order]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "PolicyAlreadyAcceptedError",
    "PolicyNotFoundError",
    "PolicyRecord",
    "PolicyRegistry",
    "RegistryError",
]
