"""Pre-registered contracts for the electronic Objective E O0 lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

from ...evidence import canonical_sha256, file_sha256


O0Candidate = Literal["O0-I", "O0-G"]
O0Actions = ("validate", "preflight", "calibrate", "screen", "confirm", "locked-final", "closeout")
OBJECTIVE_O0_CONTRACT_VERSION = "electronic_objective_o0_v2"


@dataclass(frozen=True)
class ElectronicO0Spec:
    schema_version: int
    name: str
    version: str
    dataset_root: str
    output_root: str
    objective_id: str
    split_seed: int
    code_dim: int
    logical_batch_size: int
    warmup_steps: int
    calibration_steps: tuple[int, ...]
    screen_steps: int
    confirm_seeds: tuple[int, ...]
    final_steps: int
    candidate: O0Candidate
    parameter_budget: int
    objective_contract_version: str = OBJECTIVE_O0_CONTRACT_VERSION
    activation_step: int = 400
    camera_target: float = 0.30
    diversity_lower: float = 0.70
    diversity_upper: float = 1.30
    coverage_target: float = 1.35
    fixed_exposure_scale: float = 0.4913053810596466
    require_clean_git: bool = True
    harness_revision: int = 1

    @property
    def config_hash(self) -> str:
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.schema_version != 1 or self.objective_contract_version != OBJECTIVE_O0_CONTRACT_VERSION:
            raise ValueError("electronic O0 schema/contract version is unsupported")
        if self.objective_id != "objective_e_active_set_v1":
            raise ValueError("electronic O0 requires Objective E active-set semantics")
        if self.candidate not in {"O0-I", "O0-G"}:
            raise ValueError("electronic O0 candidate must be O0-I or O0-G")
        if (self.split_seed, self.code_dim, self.logical_batch_size) != (1234, 8, 50):
            raise ValueError("electronic O0 split/code/batch contract is fixed")
        if self.activation_step != self.warmup_steps or self.activation_step != 400:
            raise ValueError("electronic O0 activation is fixed at step 400")
        if self.calibration_steps != (400, 401, 402, 403):
            raise ValueError("electronic O0 calibration steps must be 400..403")
        if self.screen_steps != 984 or self.final_steps != 4920:
            raise ValueError("electronic O0 training budgets are fixed")
        if self.confirm_seeds != (1234, 1235, 1236):
            raise ValueError("electronic O0 confirmation seeds are fixed")
        if self.parameter_budget <= 0 or self.parameter_budget > 6_000_000:
            raise ValueError("electronic O0 parameter budget is invalid")
        if not (0.0 < self.camera_target < 1.0 and self.diversity_lower < self.diversity_upper and self.coverage_target > 0.0 and self.fixed_exposure_scale > 0.0):
            raise ValueError("electronic O0 acceptance targets are invalid")
        if self.harness_revision < 1:
            raise ValueError("electronic O0 harness revision is invalid")


def load_objective_o0_spec(path: str | Path) -> ElectronicO0Spec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("electronic O0 config must be a mapping")
    payload["calibration_steps"] = tuple(payload["calibration_steps"])
    payload["confirm_seeds"] = tuple(payload["confirm_seeds"])
    spec = ElectronicO0Spec(**payload)
    spec.validate()
    return spec


def objective_o0_input_hashes(
    config_path: str | Path,
    *,
    evaluator_checkpoint: str,
    primary_qualification_evidence: str,
    secondary_evaluator_checkpoint: str,
    secondary_training_evidence: str,
    measurement_bundle: str,
    measurement_evidence: str,
) -> dict[str, str]:
    return {
        "o0_config": file_sha256(config_path),
        "primary_evaluator": file_sha256(evaluator_checkpoint),
        "primary_qualification_evidence": file_sha256(primary_qualification_evidence),
        "secondary_evaluator": file_sha256(secondary_evaluator_checkpoint),
        "secondary_training_evidence": file_sha256(secondary_training_evidence),
        "measurement_bundle": file_sha256(measurement_bundle),
        "measurement_evidence": file_sha256(measurement_evidence),
    }
