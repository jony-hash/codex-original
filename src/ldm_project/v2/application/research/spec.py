"""Declarative contract for the conditional-generation capacity map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from ...evidence import canonical_sha256, file_sha256
from .measurement import MEASUREMENT_CONTRACT


MapState = Literal[
    "invalid_harness", "screened_out", "screen_passed", "smoke_failed",
    "smoke_passed", "confirmed",
]
ObjectiveId = Literal["primary_only", "dual_bundle"]


@dataclass(frozen=True)
class ElectronicVariantSpec:
    variant_id: str
    kind: Literal["shared_modes", "class_modes", "spatial_decoder"]
    code_dim: int
    parameter_budget: int = 1_500_000

    def validate(self) -> None:
        if self.variant_id not in {"E0", "E1", "E2", "E3", "E4"}:
            raise ValueError(f"unknown electronic capacity-map variant {self.variant_id}")
        if self.code_dim not in {8, 16, 32}:
            raise ValueError("capacity-map code_dim must be one of 8, 16, 32")
        if self.variant_id in {"E0", "E1", "E2"} and self.code_dim != 8:
            raise ValueError(f"{self.variant_id} is pre-registered as an 8D variant")
        if self.variant_id == "E3" and self.code_dim != 16:
            raise ValueError("E3 is pre-registered as 16D")
        if self.variant_id == "E4" and self.code_dim != 32:
            raise ValueError("E4 is pre-registered as 32D")
        if self.kind == "spatial_decoder" and self.parameter_budget > 1_500_000:
            raise ValueError("electronic seed decoder budget may not exceed 1.5M")


@dataclass(frozen=True)
class OpticalVariantSpec:
    variant_id: str
    kind: Literal["digital_oracle", "phase_bypass", "phase_only", "deep_phase_only", "spatial_upper_bound", "complex_upper_bound"]
    encoder_layers: int
    decoder_layers: int
    diagnostic_only: bool = False

    def validate(self) -> None:
        expected = {
            "O0": ("digital_oracle", 0, 0, False),
            "O1": ("phase_bypass", 0, 0, False),
            "O2": ("phase_only", 2, 2, False),
            "O3": ("deep_phase_only", 4, 4, False),
            "O4": ("spatial_upper_bound", 2, 2, True),
            "O5": ("complex_upper_bound", 2, 2, True),
        }
        if self.variant_id not in expected:
            raise ValueError(f"unknown optical capacity-map variant {self.variant_id}")
        kind, enc, dec, diagnostic = expected[self.variant_id]
        if (self.kind, self.encoder_layers, self.decoder_layers, self.diagnostic_only) != (kind, enc, dec, diagnostic):
            raise ValueError(f"{self.variant_id} does not match its pre-registered topology")


@dataclass(frozen=True)
class CapacityMapSpec:
    schema_version: int
    name: str
    dataset_root: str
    output_root: str
    evaluator_checkpoint: str | None
    smooth_evaluator_checkpoint: str | None
    evaluator_bundle: str | None
    evaluator_qualification_evidence: str | None
    electronic: tuple[ElectronicVariantSpec, ...]
    optical: tuple[OpticalVariantSpec, ...]
    screening_probe_steps: int = 50
    smoke_steps: int = 400
    confirm_epochs: int = 5
    seeds: tuple[int, ...] = (1234, 4321, 2234)
    batch_size: int = 50
    version: str = "capacity_map_v1"
    objective_id: ObjectiveId = "primary_only"
    measurement_contract: str | None = None

    @property
    def config_hash(self) -> str:
        return canonical_sha256(asdict(self))

    def variant(self, value: str) -> tuple[ElectronicVariantSpec, OpticalVariantSpec]:
        try:
            electronic_id, optical_id = value.split(":", 1)
        except ValueError as error:
            raise ValueError("variant must be formatted E#:O#") from error
        electronic = next((item for item in self.electronic if item.variant_id == electronic_id), None)
        optical = next((item for item in self.optical if item.variant_id == optical_id), None)
        if electronic is None or optical is None:
            raise ValueError(f"variant {value!r} is not declared by the capacity-map config")
        return electronic, optical

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only capacity-map schema_version=1 is supported")
        if not self.name or self.screening_probe_steps != 50 or self.smoke_steps != 400 or self.confirm_epochs != 5:
            raise ValueError("capacity-map budgets are pre-registered at 50/400/5")
        if self.seeds != (1234, 4321, 2234):
            raise ValueError("capacity-map smoke seeds are fixed at 1234/4321/2234")
        if self.batch_size % 10 or self.batch_size < 10:
            raise ValueError("capacity-map batch_size must be balanced across ten labels")
        if self.objective_id not in {"primary_only", "dual_bundle"}:
            raise ValueError("capacity-map objective_id must be primary_only or dual_bundle")
        if self.measurement_contract not in {None, MEASUREMENT_CONTRACT}:
            raise ValueError("capacity-map measurement_contract is not the v2.2 fixed-panel contract")
        if not all((self.evaluator_checkpoint, self.smooth_evaluator_checkpoint, self.evaluator_bundle, self.evaluator_qualification_evidence)):
            raise ValueError("capacity-map requires a content-bound qualified evaluator bundle")
        if {item.variant_id for item in self.electronic} != {"E0", "E1", "E2", "E3", "E4"}:
            raise ValueError("the pre-registered electronic matrix is E0--E4")
        if {item.variant_id for item in self.optical} != {"O0", "O1", "O2", "O3", "O4", "O5"}:
            raise ValueError("the pre-registered optical matrix is O0--O5")
        for item in (*self.electronic, *self.optical):
            item.validate()


@dataclass(frozen=True)
class CapacityMapEvidence:
    """Immutable per-stage research evidence; map state is separate from formal gates."""

    schema_version: int
    action: str
    state: MapState
    variant: str | None
    dataset: str
    seed: int | None
    config_hash: str
    inputs: dict[str, str]
    metrics: dict[str, Any]
    reason: str | None = None
    evidence_revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_hash"] = canonical_sha256(payload)
        return payload


def load_capacity_map_spec(path: str | Path) -> CapacityMapSpec:
    target = Path(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capacity-map config must be a YAML mapping")
    electronic = tuple(ElectronicVariantSpec(**item) for item in payload.pop("electronic"))
    optical = tuple(OpticalVariantSpec(**item) for item in payload.pop("optical"))
    if "seeds" in payload:
        payload["seeds"] = tuple(payload["seeds"])
    spec = CapacityMapSpec(electronic=electronic, optical=optical, **payload)
    spec.validate()
    return spec


def input_hashes(spec: CapacityMapSpec, config_path: str | Path) -> dict[str, str]:
    hashes = {"capacity_map_config": file_sha256(config_path)}
    for name, path in (
        ("primary_evaluator", spec.evaluator_checkpoint),
        ("smooth_evaluator", spec.smooth_evaluator_checkpoint),
        ("evaluator_bundle", spec.evaluator_bundle),
        ("evaluator_qualification_evidence", spec.evaluator_qualification_evidence),
    ):
        if path:
            target = Path(path)
            hashes[name] = file_sha256(target) if target.is_file() else "missing"
    return hashes
