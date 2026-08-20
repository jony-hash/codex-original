"""Synthetic, research-only fiber transport matrix pools."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch

PoolName = Literal["train", "validation", "holdout", "stress", "full_phase"]


@dataclass(frozen=True)
class FiberResearchSpec:
    grid_shape: tuple[int, int] = (10, 10)
    phase_rms_rad: float = 0.19634954
    mean_loss_db: float = 0.5
    differential_loss_sigma_db: float = 0.1
    offset_range_m: float = 1.0e-6
    model: str = "synthetic_smf_diagonal"
    physical_parameter_status: str = "synthetic diagnostic-only"

    @property
    def ports(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]


@dataclass(frozen=True)
class TransportRecord:
    matrix_id: str
    pool: PoolName
    seed: int
    parameters: dict[str, object]
    content_hash: str


def _sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class FiberTransportPool:
    """Deterministic diagonal complex matrices with strict passivity checks."""

    COUNTS: dict[PoolName, int] = {"train": 25, "validation": 4, "holdout": 8, "stress": 4, "full_phase": 4}
    SEEDS: dict[PoolName, tuple[int, ...]] = {
        "train": (0, *range(6101, 6125)),
        "validation": tuple(range(7101, 7105)),
        "holdout": tuple(range(7201, 7209)),
        "stress": tuple(range(8101, 8105)),
        "full_phase": tuple(range(8201, 8205)),
    }

    def __init__(self, spec: FiberResearchSpec = FiberResearchSpec()) -> None:
        self.spec = spec
        self._matrices: dict[PoolName, tuple[torch.Tensor, ...]] = {}
        self.records: dict[PoolName, tuple[TransportRecord, ...]] = {}
        for pool, seeds in self.SEEDS.items():
            matrices: list[torch.Tensor] = []
            records: list[TransportRecord] = []
            for index, seed in enumerate(seeds):
                matrix, parameters = self._make(pool, seed)
                if not torch.isfinite(matrix).all() or float(matrix.abs().amax()) > 1.0 + 1e-5:
                    raise ValueError("transport matrix is not finite/passive")
                matrix_id = f"{pool}-{index:02d}"
                records.append(TransportRecord(matrix_id, pool, seed, parameters, _sha(matrix)))
                matrices.append(matrix)
            self._matrices[pool] = tuple(matrices)
            self.records[pool] = tuple(records)
        self._validate_disjoint_ids()

    def _make(self, pool: PoolName, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
        if pool == "train" and seed == 0:
            return torch.eye(self.spec.ports, dtype=torch.complex64), {"kind": "nominal_identity"}
        if pool == "full_phase":
            generator = torch.Generator().manual_seed(seed)
            phase = torch.rand(self.spec.ports, generator=generator) * (2.0 * torch.pi) - torch.pi
            gain = torch.ones_like(phase)
            params = {"phase_distribution": "uniform", "phase_range_rad": [-3.141592653589793, 3.141592653589793]}
        else:
            stress = pool == "stress"
            phase_rms = 0.39269908 if stress else self.spec.phase_rms_rad
            mean_loss = 1.5 if stress else self.spec.mean_loss_db
            diff_sigma = 0.3 if stress else self.spec.differential_loss_sigma_db
            offset = 2.0e-6 if stress else self.spec.offset_range_m
            generator = torch.Generator().manual_seed(seed)
            phase = torch.randn(self.spec.ports, generator=generator) * phase_rms
            loss_db = mean_loss + torch.randn(self.spec.ports, generator=generator) * diff_sigma
            loss_db = loss_db.clamp_min(0.0)
            gain = 10.0 ** (-loss_db / 20.0)
            params = {"phase_rms_rad": phase_rms, "mean_loss_db": mean_loss, "differential_loss_sigma_db": diff_sigma, "offset_range_m": offset}
        matrix = torch.diag(gain.to(torch.complex64) * torch.exp(1j * phase.to(torch.float32)))
        return matrix, params

    def _validate_disjoint_ids(self) -> None:
        ids = [record.matrix_id for records in self.records.values() for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("transport pool IDs overlap")

    def matrix(self, pool: PoolName, index: int) -> torch.Tensor:
        return self._matrices[pool][index]

    def schedule(self, seed: int, batches: int, pool: PoolName = "train") -> tuple[str, ...]:
        if batches < 1:
            raise ValueError("batches must be positive")
        generator = torch.Generator().manual_seed(seed)
        count = len(self._matrices[pool])
        prefix = list(range(min(count, batches)))
        suffix = torch.randint(count, (batches - len(prefix),), generator=generator).tolist()
        indices = prefix + suffix
        return tuple(self.records[pool][index].matrix_id for index in indices)

    def evidence(self, schedule_seed: int = 5201, batches: int = 100) -> dict[str, object]:
        schedule = self.schedule(schedule_seed, batches)
        return {
            "schema": "fiber_transport_pool.v1",
            "status": self.spec.physical_parameter_status,
            "spec": self.spec.__dict__,
            "pools": {
                pool: [record.__dict__ for record in records]
                for pool, records in self.records.items()
            },
            "schedule": list(schedule),
            "schedule_hash": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest(),
        }
