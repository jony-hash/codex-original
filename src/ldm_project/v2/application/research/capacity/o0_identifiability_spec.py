"""Contracts for the v2.4 D0 update-identifiability audit.

This is intentionally a new lifecycle.  It can inspect the v2.3 evidence, but
never treats that evidence as a scientific result or as a warm-start.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ldm_project.v2.evidence import canonical_sha256, file_sha256
from ldm_project.v2.application.research.shared.measurement import MEASUREMENT_CONTRACT


@dataclass(frozen=True)
class O0IdentifiabilitySpec:
    schema_version: int
    name: str
    version: str
    dataset_root: str
    output_root: str
    evaluator_checkpoint: str
    smooth_evaluator_checkpoint: str
    evaluator_bundle: str
    evaluator_qualification_evidence: str
    measurement_contract: str
    objective_id: str
    split_seed: int
    microbatch_size: int
    effective_batch_size: int
    audit_steps: int
    probe_steps: int
    seed: int
    candidate: str = "D0"
    harness_revision: int = 1
    require_clean_git: bool = False
    r1_interval: int = 16
    r1_weight: float = 10.0

    @property
    def config_hash(self) -> str:
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.schema_version != 1 or self.candidate != "D0":
            raise ValueError("v2.4 is a D0-only contract")
        if self.measurement_contract != MEASUREMENT_CONTRACT:
            raise ValueError("v2.4 requires the qualified measurement contract")
        if self.objective_id != "data_adversarial_identifiability":
            raise ValueError("v2.4 objective is fixed")
        if (self.split_seed, self.microbatch_size, self.effective_batch_size) != (
            1234,
            10,
            50,
        ):
            raise ValueError("v2.4 split and batch contract is fixed")
        if self.audit_steps != 2 or self.probe_steps != 100 or self.seed != 1234:
            raise ValueError("v2.4 budgets are fixed to audit=2, probe=100, seed=1234")
        if self.harness_revision < 1 or self.r1_interval <= 0 or self.r1_weight < 0:
            raise ValueError("invalid v2.4 training contract")


@dataclass(frozen=True)
class O0IdentifiabilityEvidence:
    schema_version: int
    action: str
    state: str
    candidate: str
    dataset: str
    seed: int
    config_hash: str
    inputs: dict[str, str]
    metrics: dict[str, Any]
    reason: str | None = None
    evidence_revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_hash"] = canonical_sha256(payload)
        return payload


def load_identifiability_spec(path: str | Path) -> O0IdentifiabilitySpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v2.4 config must be a mapping")
    spec = O0IdentifiabilitySpec(**payload)
    spec.validate()
    return spec


def identifiability_input_hashes(
    spec: O0IdentifiabilitySpec, config_path: str | Path
) -> dict[str, str]:
    result = {"config": file_sha256(config_path)}
    for key, path in {
        "primary_evaluator": spec.evaluator_checkpoint,
        "smooth_evaluator": spec.smooth_evaluator_checkpoint,
        "evaluator_bundle": spec.evaluator_bundle,
        "evaluator_qualification": spec.evaluator_qualification_evidence,
    }.items():
        result[key] = file_sha256(path) if Path(path).is_file() else "missing"
    return result
