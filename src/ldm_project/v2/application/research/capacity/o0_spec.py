"""Declarative contracts for the v2.3 electronic-oracle research lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from ldm_project.v2.evidence import canonical_sha256, file_sha256
from ldm_project.v2.application.research.shared.measurement import MEASUREMENT_CONTRACT


CandidateId = Literal["D0", "C0", "C1", "C2", "C3"]
ObjectiveId = Literal["data_adversarial"]
EvidenceState = Literal[
    "invalid_harness", "control_failed", "control_passed", "probe_failed",
    "probe_passed", "smoke_failed", "smoke_passed", "confirm_failed",
    "confirm_passed", "final_failed", "final_passed", "preflight_failed",
    "preflight_passed", "closed",
]
AggregateClassification = Literal[
    "d0_image_oracle_reachable",
    "electronic_oracle_reachable", "electronic_generator_capacity",
    "exact_anchor_bottleneck", "input_manifold_bottleneck",
    "training_objective_or_evaluator_inconclusive",
    "command_parameterization_inconclusive", "closed",
]


@dataclass(frozen=True)
class O0CandidateSpec:
    candidate_id: CandidateId
    domain: Literal["image28", "command200"]
    code_dim: int
    anchor: Literal["none", "exact", "soft"]
    diagnostic_only: bool = False
    parameter_budget: int = 5_000_000

    def validate(self) -> None:
        expected = {
            "D0": ("image28", 32, "none", True),
            "C0": ("command200", 8, "exact", False),
            "C1": ("command200", 8, "exact", False),
            "C2": ("command200", 8, "soft", False),
            "C3": ("command200", 32, "soft", True),
        }
        if self.candidate_id not in expected:
            raise ValueError(f"unknown v2.3 candidate {self.candidate_id}")
        if (self.domain, self.code_dim, self.anchor, self.diagnostic_only) != expected[self.candidate_id]:
            raise ValueError(f"candidate {self.candidate_id} does not match its pre-registered contract")
        if self.parameter_budget <= 0 or self.parameter_budget > 6_000_000:
            raise ValueError("v2.3 candidate parameter budget must be within 1..6M")
        if self.candidate_id == "D0" and self.parameter_budget > 5_000_000:
            raise ValueError("D0 parameter budget may not exceed 5M")


@dataclass(frozen=True)
class O0SplitManifest:
    schema_version: int
    dataset_root: str
    seed: int
    train_indices_hash: str
    dev_indices_hash: str
    locked_test_indices_hash: str
    dataset_content_hash: str
    train_count: int = 55_000
    dev_count: int = 5_000
    locked_test_count: int = 5_000

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.schema_version != 1 or self.seed != 1234:
            raise ValueError("v2.3 split manifest requires schema 1 and seed 1234")
        if (self.train_count, self.dev_count, self.locked_test_count) != (55_000, 5_000, 5_000):
            raise ValueError("v2.3 split counts are fixed at 55k/5k/5k")
        if not all(len(value) == 64 for value in (self.train_indices_hash, self.dev_indices_hash, self.locked_test_indices_hash, self.dataset_content_hash)):
            raise ValueError("v2.3 split hashes must be SHA-256 values")


@dataclass(frozen=True)
class O0FeasibilitySpec:
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
    objective_id: ObjectiveId
    split_seed: int
    microbatch_size: int
    effective_batch_size: int
    smoke_steps: int
    probe_steps: int
    confirm_epochs: int
    seeds: tuple[int, ...]
    candidates: tuple[O0CandidateSpec, ...]
    harness_revision: int = 1
    require_clean_git: bool = False
    stage_endpoint_checkpoint: bool = True
    probe_margin_epsilon: float = 1.0e-6

    @property
    def config_hash(self) -> str:
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.schema_version != 1 or self.objective_id != "data_adversarial":
            raise ValueError("only v2.3 schema 1 data_adversarial objective is supported")
        if self.measurement_contract != MEASUREMENT_CONTRACT:
            raise ValueError("v2.3 must use the qualified measurement contract")
        if self.split_seed != 1234 or self.microbatch_size != 10 or self.effective_batch_size != 50:
            raise ValueError("v2.3 split and batch contracts are fixed at seed 1234 and 10/50")
        if (self.probe_steps, self.smoke_steps, self.confirm_epochs) != (50, 400, 5):
            raise ValueError("v2.3 budgets are fixed at 50/400/5")
        if self.seeds != (1234, 4321, 2234):
            raise ValueError("v2.3 seeds are fixed at 1234/4321/2234")
        if self.harness_revision < 1 or self.probe_margin_epsilon < 0:
            raise ValueError("v2.3 harness revision and probe margin are invalid")
        if {item.candidate_id for item in self.candidates} != {"D0", "C0", "C1", "C2", "C3"}:
            raise ValueError("v2.3 must declare exactly D0 and C0-C3")
        for item in self.candidates:
            item.validate()


@dataclass(frozen=True)
class O0FeasibilityEvidence:
    schema_version: int
    action: str
    state: EvidenceState
    candidate: CandidateId | None
    dataset: str
    seed: int | None
    config_hash: str
    split_manifest_hash: str | None
    inputs: dict[str, str]
    metrics: dict[str, Any]
    reason: str | None = None
    evidence_revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_hash"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class O0AggregateDecision:
    """Seed-aggregated decision; never grants physical promotion itself."""

    candidate: CandidateId
    classification: AggregateClassification
    passed_seed_count: int
    required_seed_count: int = 2
    seed_results: tuple[dict[str, Any], ...] = ()
    physical_authorization: bool = False

    def validate(self) -> None:
        if not 0 <= self.passed_seed_count <= len(self.seed_results) <= 3:
            raise ValueError("v2.3 aggregate seed counts are inconsistent")
        if self.required_seed_count != 2:
            raise ValueError("v2.3 requires two of three seeds")
        if self.candidate == "C3" and self.physical_authorization:
            raise ValueError("C3 can never authorize physical follow-up")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def load_o0_spec(path: str | Path) -> O0FeasibilitySpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v2.3 config must be a YAML mapping")
    candidates = tuple(O0CandidateSpec(**item) for item in payload.pop("candidates"))
    payload["seeds"] = tuple(payload["seeds"])
    spec = O0FeasibilitySpec(candidates=candidates, **payload)
    spec.validate()
    return spec


def o0_input_hashes(spec: O0FeasibilitySpec, config_path: str | Path) -> dict[str, str]:
    result = {"o0_config": file_sha256(config_path)}
    for name, path in (
        ("primary_evaluator", spec.evaluator_checkpoint),
        ("smooth_evaluator", spec.smooth_evaluator_checkpoint),
        ("evaluator_bundle", spec.evaluator_bundle),
        ("evaluator_qualification_evidence", spec.evaluator_qualification_evidence),
    ):
        result[name] = file_sha256(path) if Path(path).is_file() else "missing"
    return result
