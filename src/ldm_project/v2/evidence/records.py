"""Versioned evidence records and hard downstream admission checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .io import canonical_sha256


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: Mapping[str, float | bool]
    reason: str | None = None


@dataclass(frozen=True)
class EvidenceRecordV2:
    schema_version: int
    evidence_kind: str
    tier: str
    status: str
    git_commit: str
    dirty_patch_hash: str | None
    config_hash: str
    input_hashes: Mapping[str, str]
    gates: tuple[GateResult, ...]
    permitted_next_actions: tuple[str, ...]
    invalidation_reason: str | None = None

    @property
    def evidence_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gates"] = [asdict(gate) for gate in self.gates]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecordV2":
        values = dict(payload)
        values["gates"] = tuple(GateResult(**gate) for gate in values.get("gates", ()))
        values["permitted_next_actions"] = tuple(values.get("permitted_next_actions", ()))
        return cls(**values)

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported evidence schema")
        if self.status not in {"passed", "failed", "diagnostic"}:
            raise ValueError("invalid evidence status")
        if self.tier == "acceptance" and self.dirty_patch_hash is not None:
            raise ValueError("acceptance evidence requires a clean commit")
        if self.tier == "diagnostic" and any(action in {"train", "promote", "export"} for action in self.permitted_next_actions):
            raise ValueError("diagnostic evidence cannot authorize train, promote, or export")
        if self.status == "passed" and not all(gate.passed for gate in self.gates):
            raise ValueError("passed evidence cannot contain a failed gate")
        if self.status != "passed" and self.permitted_next_actions and any(action in {"train", "promote", "export"} for action in self.permitted_next_actions):
            raise ValueError("failed or diagnostic evidence cannot authorize train, promote, or export")


def require_passed_evidence(record: EvidenceRecordV2, *, kind: str) -> None:
    record.validate()
    if record.evidence_kind != kind or record.status != "passed":
        raise ValueError(f"{kind} evidence must be passed before this operation")
