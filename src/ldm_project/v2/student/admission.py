"""Prototype and capacity gates shared by all v2 student stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..evidence import GateResult


@dataclass(frozen=True)
class PrototypeMetrics:
    clean_corr: float
    clean_dice: float
    prototype_accuracy: float
    high_saturation: float
    finite: bool
    gradients_finite: bool
    memory_pass: bool
    epoch_stable: bool


def prototype_gate(metrics: PrototypeMetrics, readable_labels: Sequence[int] | None) -> GateResult:
    labels = list(readable_labels or ())
    checks = {
        "clean_corr": metrics.clean_corr >= 0.70,
        "clean_dice": metrics.clean_dice >= 0.68,
        "prototype_accuracy": metrics.prototype_accuracy == 1.0,
        "high_saturation": metrics.high_saturation <= 0.08,
        "finite": metrics.finite,
        "gradients_finite": metrics.gradients_finite,
        "memory_pass": metrics.memory_pass,
        "epoch_stable": metrics.epoch_stable,
        "manual_readability": labels == list(range(10)),
    }
    values: Mapping[str, float | bool] = {
        "clean_corr": metrics.clean_corr,
        "clean_dice": metrics.clean_dice,
        "prototype_accuracy": metrics.prototype_accuracy,
        "high_saturation": metrics.high_saturation,
        **checks,
    }
    return GateResult("prototype_raw_cmos", all(checks.values()), values, None if all(checks.values()) else "prototype gate failed")


@dataclass(frozen=True)
class CapacityMetrics:
    physical_slm_code_fraction: float
    raw_camera_code_fraction: float
    prototype_support_energy: float
    clipping_fraction: float
    parameters_unchanged: bool
    optimizer_created: bool
    provenance_complete: bool


def capacity_gate(metrics: CapacityMetrics, prototype: GateResult) -> GateResult:
    checks = {
        "prototype_gate": prototype.passed,
        "physical_slm_code_fraction": metrics.physical_slm_code_fraction >= 0.25,
        "raw_camera_code_fraction": metrics.raw_camera_code_fraction >= 0.25,
        "prototype_support_energy": metrics.prototype_support_energy >= 0.95,
        "clipping_fraction": metrics.clipping_fraction <= 0.08,
        "parameters_unchanged": metrics.parameters_unchanged,
        "no_optimizer": not metrics.optimizer_created,
        "provenance_complete": metrics.provenance_complete,
    }
    values: Mapping[str, float | bool] = {
        "physical_slm_code_fraction": metrics.physical_slm_code_fraction,
        "raw_camera_code_fraction": metrics.raw_camera_code_fraction,
        "prototype_support_energy": metrics.prototype_support_energy,
        "clipping_fraction": metrics.clipping_fraction,
        **checks,
    }
    return GateResult("code_capacity", all(checks.values()), values, None if all(checks.values()) else "capacity gate failed")


def balanced_code_fraction(values: torch.Tensor) -> float:
    """Within-label code SS divided by total SS for [label, code, ...] tensors."""
    if values.ndim < 3:
        raise ValueError("code factor expects [label, code, spatial...] values")
    labels, codes = values.shape[:2]
    if labels != 10 or codes < 2:
        raise ValueError("code factor requires ten labels and at least two codes")
    label_mean = values.mean(dim=1, keepdim=True)
    code_ss = (values - label_mean).square().sum()
    total_ss = (values - values.mean()).square().sum().clamp_min(1.0e-12)
    return float((code_ss / total_ss).item())
