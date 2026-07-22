"""Final raw-CMOS acceptance evaluation; it never promotes a checkpoint itself."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .admission import balanced_code_fraction
from .metrics import correlation, foreground_dice

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


def paired_advantage_lower_bound(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, *, seed: int = 1234, resamples: int = 10000) -> float:
    """Bootstrap paired-vs-other-same-label correlation advantage."""
    paired = correlation(prediction, target)
    values = []
    for index in range(labels.numel()):
        peers = torch.nonzero((labels == labels[index]) & (torch.arange(labels.numel(), device=labels.device) != index), as_tuple=False).flatten()
        if peers.numel() == 0:
            raise ValueError("each label requires at least two paired samples")
        repeated = prediction[index : index + 1].expand(peers.numel(), *prediction.shape[1:])
        values.append(paired[index] - correlation(repeated, target.index_select(0, peers)).mean())
    advantage = torch.stack(values)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(0, advantage.numel(), (resamples, advantage.numel()), generator=generator)
    means = advantage.cpu().index_select(0, indices.flatten()).reshape(resamples, -1).mean(dim=1)
    return float(torch.quantile(means, 0.05).item())


def acceptance_metrics(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    rows = []
    for label in range(10):
        selected = prediction[labels == label]
        if selected.shape[0] < 2:
            raise ValueError("acceptance requires at least two codes for every label")
        rows.append(selected)
    return balanced_code_fraction(torch.stack(rows)), float(correlation(prediction, target).mean().item()), float(foreground_dice(prediction, target).mean().item())
