"""Final raw-CMOS acceptance evaluation; it never promotes a checkpoint itself."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..evidence import GateResult


@dataclass(frozen=True)
class CodeAcceptanceMetrics:
    camera_code_fraction: float
    teacher_corr: float
    teacher_dice: float
    paired_advantage_lower_bound: float
    bootstrap_seed: int
    bootstrap_resamples: int


def code_acceptance_gate(metrics: CodeAcceptanceMetrics, readable_labels: Sequence[int] | None) -> GateResult:
    checks = {
        "camera_code_fraction": metrics.camera_code_fraction >= 0.25,
        "teacher_corr": metrics.teacher_corr >= 0.70,
        "teacher_dice": metrics.teacher_dice >= 0.68,
        "paired_advantage": metrics.paired_advantage_lower_bound > 0.0,
        "bootstrap_seed": metrics.bootstrap_seed == 1234,
        "bootstrap_resamples": metrics.bootstrap_resamples == 10000,
        "manual_readability": list(readable_labels or ()) == list(range(10)),
    }
    values = {
        "camera_code_fraction": metrics.camera_code_fraction,
        "teacher_corr": metrics.teacher_corr,
        "teacher_dice": metrics.teacher_dice,
        "paired_advantage_lower_bound": metrics.paired_advantage_lower_bound,
        **checks,
    }
    return GateResult("code_conditioned_raw_cmos", all(checks.values()), values, None if all(checks.values()) else "code-conditioned acceptance failed")
