"""Deterministic synthetic diagonal transport pools for fiber research."""
from __future__ import annotations

import hashlib
import json
import math
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
    offset_m: tuple[float, float]
    parameters: dict[str, object]
    content_hash: str


def _content_hash(diagonal: torch.Tensor, payload: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(diagonal.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


class FiberTransportPool:
    SEEDS: dict[PoolName, tuple[int, ...]] = {
        "train": (0, *range(6101, 6125)),
        "validation": tuple(range(7101, 7105)),
        "holdout": tuple(range(7201, 7209)),
        "stress": tuple(range(8101, 8105)),
        "full_phase": tuple(range(8201, 8205)),
    }

    def __init__(self, spec: FiberResearchSpec = FiberResearchSpec()) -> None:
        self.spec = spec
        self._matrices: dict[str, torch.Tensor] = {}
        self.records: dict[PoolName, tuple[TransportRecord, ...]] = {}
        for pool, seeds in self.SEEDS.items():
            rows: list[TransportRecord] = []
            for index, seed in enumerate(seeds):
                matrix, offset, parameters = self._make(pool, seed)
                if not torch.isfinite(matrix).all() or float(torch.linalg.svdvals(matrix).max()) > 1.0 + 1.0e-5:
                    raise ValueError("transport matrix is not finite/passive")
                matrix_id = f"{pool}-{index:02d}"
                payload = {"matrix_id": matrix_id, "pool": pool, "seed": seed, "offset_m": list(offset), "parameters": parameters}
                record = TransportRecord(matrix_id, pool, seed, offset, parameters, _content_hash(torch.diagonal(matrix), payload))
                self._matrices[matrix_id] = matrix
                rows.append(record)
            self.records[pool] = tuple(rows)
        self._validate_disjoint()

    def _make(self, pool: PoolName, seed: int) -> tuple[torch.Tensor, tuple[float, float], dict[str, object]]:
        if pool == "train" and seed == 0:
            return torch.eye(self.spec.ports, dtype=torch.complex64), (0.0, 0.0), {"kind": "nominal_identity"}
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if pool == "full_phase":
            phase = torch.rand(self.spec.ports, generator=generator, dtype=torch.float64) * (2.0 * torch.pi) - torch.pi
            gain = torch.ones_like(phase)
            parameters: dict[str, object] = {"phase_distribution": "uniform", "phase_range_rad": [-math.pi, math.pi]}
            # The negative control isolates phase mismatch; offsets remain zero.
            offset_range = 0.0
        else:
            stress = pool == "stress"
            target_rms = 0.39269908 if stress else self.spec.phase_rms_rad
            mean_loss = 1.5 if stress else self.spec.mean_loss_db
            sigma = 0.3 if stress else self.spec.differential_loss_sigma_db
            offset_range = 2.0e-6 if stress else self.spec.offset_range_m
            phase = torch.randn(self.spec.ports, generator=generator, dtype=torch.float64)
            phase = phase - phase.mean()
            phase = phase * (target_rms / phase.square().mean().sqrt()).clamp_min(1.0e-30)
            loss = (mean_loss + sigma * torch.randn(self.spec.ports, generator=generator, dtype=torch.float64)).clamp_min(0.0)
            gain = 10.0 ** (-loss / 20.0)
            parameters = {"phase_rms_rad_target": target_rms, "phase_mean_rad": float(phase.mean()), "phase_rms_rad_realized": float(phase.square().mean().sqrt()), "mean_loss_db_target": mean_loss, "mean_loss_db_realized": float(loss.mean()), "differential_loss_sigma_db": sigma, "offset_range_m": offset_range}
        offset = tuple(float(value) for value in (torch.rand(2, generator=generator, dtype=torch.float64) * (2.0 * offset_range) - offset_range).tolist())
        diagonal = gain.to(torch.complex64) * torch.exp(1j * phase.to(torch.float32))
        return torch.diag(diagonal), offset, parameters

    def _validate_disjoint(self) -> None:
        records = [record for rows in self.records.values() for record in rows]
        if len({record.matrix_id for record in records}) != len(records) or len({record.content_hash for record in records}) != len(records):
            raise ValueError("transport pool IDs or hashes overlap")

    def record(self, matrix_id: str) -> TransportRecord:
        for rows in self.records.values():
            for record in rows:
                if record.matrix_id == matrix_id:
                    return record
        raise KeyError(f"unknown transport ID: {matrix_id}")

    def matrix_by_id(self, matrix_id: str) -> torch.Tensor:
        return self._matrices[matrix_id]

    def matrix(self, pool: PoolName, index: int) -> torch.Tensor:
        return self.matrix_by_id(self.records[pool][index].matrix_id)

    def schedule(self, seed: int, batches: int, pool: PoolName = "train") -> tuple[str, ...]:
        if batches < 1:
            raise ValueError("batches must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ids = [record.matrix_id for record in self.records[pool]]
        result: list[str] = []
        while len(result) < batches:
            order = torch.randperm(len(ids), generator=generator).tolist()
            result.extend(ids[index] for index in order)
        return tuple(result[:batches])

    def evidence(self, schedule_seed: int = 5201, batches: int = 100) -> dict[str, object]:
        schedule = self.schedule(schedule_seed, batches)
        return {"schema": "fiber_transport_pool.v2", "status": self.spec.physical_parameter_status, "spec": self.spec.__dict__, "pools": {pool: [record.__dict__ for record in records] for pool, records in self.records.items()}, "schedule": list(schedule), "schedule_hash": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()}
