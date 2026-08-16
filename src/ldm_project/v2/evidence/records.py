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
class EvidenceContextV2:
    """Every downstream consumer compares this context before doing work."""

    system_signature: str
    config_hash: str
    teacher_manifest_hash: str | None = None
    dataset_hash: str | None = None
    calibration_hash: str | None = None
    checkpoint_hash: str | None = None
    basis_hash: str | None = None
    evaluator_checkpoint_hash: str | None = None
    dataset_split_manifest_hash: str | None = None
    exposure_scale: float | None = None
    candidate_dataset_manifest_hash: str | None = None

    def assert_matches(self, expected: "EvidenceContextV2", *, fields: tuple[str, ...] | None = None) -> None:
        names = fields or tuple(self.to_dict())
        for name in names:
            actual_value = getattr(self, name)
            expected_value = getattr(expected, name)
            if expected_value is not None and actual_value != expected_value:
                raise ValueError(f"evidence context mismatch for {name}")

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecordV2:
    schema_version: int
    evidence_kind: str
    tier: str
    status: str
    git_commit: str
    dirty_patch_hash: str | None
    config_hash: str
    context: EvidenceContextV2
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
        values["context"] = EvidenceContextV2(**values["context"])
        return cls(**values)

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported evidence schema")
        if self.status not in {
            "passed",
            "failed",
            "diagnostic",
            "pending_review",
            "invalid_harness",
            "valid_failed",
        }:
            raise ValueError("invalid evidence status")
        if self.tier == "acceptance" and self.dirty_patch_hash is not None:
            raise ValueError("acceptance evidence requires a clean commit")
        if self.config_hash != self.context.config_hash:
            raise ValueError("evidence config hash and context disagree")
        forbidden = {"train", "promote", "export", "evaluate"}
        if self.tier == "diagnostic" and any(action in forbidden for action in self.permitted_next_actions):
            raise ValueError("diagnostic evidence cannot authorize train, promote, or export")
        if self.status == "passed" and not all(gate.passed for gate in self.gates):
            raise ValueError("passed evidence cannot contain a failed gate")
        if self.status != "passed" and self.permitted_next_actions and any(action in forbidden for action in self.permitted_next_actions):
            raise ValueError("failed or diagnostic evidence cannot authorize train, promote, or export")


def require_passed_evidence(record: EvidenceRecordV2, *, kind: str) -> None:
    record.validate()
    if record.evidence_kind != kind or record.status != "passed":
        raise ValueError(f"{kind} evidence must be passed before this operation")


def require_evidence_action(record: EvidenceRecordV2, *, kind: str, action: str, expected_context: EvidenceContextV2, allow_diagnostic: bool = False) -> None:
    record.validate()
    if record.evidence_kind != kind:
        raise ValueError(f"expected {kind} evidence")
    record.context.assert_matches(expected_context)
    if allow_diagnostic:
        if record.tier != "diagnostic" or record.status != "diagnostic" or action not in record.permitted_next_actions:
            raise ValueError(f"{kind} diagnostic evidence cannot authorize {action}")
        return
    if record.status != "passed" or action not in record.permitted_next_actions:
        raise ValueError(f"{kind} evidence must pass before {action}")


def require_input_hashes(
    record: EvidenceRecordV2, expected: Mapping[str, str]
) -> None:
    """Require evidence to bind every supplied input hash before consumption."""
    record.validate()
    for name, value in expected.items():
        if record.input_hashes.get(name) != value:
            raise ValueError(f"evidence input hash mismatch for {name}")
