"""Small explicit training loops guarded by v2 evidence; no stage bypass exists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from ..evidence import EvidenceRecordV2, require_passed_evidence
from .generator import LabelPrototypeBasisGenerator
from .metrics import correlation, foreground_dice, high_saturation_ratio


class RawCameraSystem(Protocol):
    def forward_from_generator(self, generator, code: torch.Tensor, labels: torch.Tensor, **kwargs): ...


@dataclass(frozen=True)
class PrototypeCalibrationSummary:
    loss: float
    clean_corr: float
    clean_dice: float
    high_saturation: float
    gradients_finite: bool
    prototype_accuracy: float
    epoch_history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class CodeTrainingSummary:
    loss: float
    paired_corr: float
    paired_dice: float
    code_variance: float
    clipping_fraction: float


def _phase_regularization(system: nn.Module) -> torch.Tensor:
    terms = []
    for name, parameter in system.named_parameters():
        if name.endswith("phase") and parameter.ndim >= 3:
            terms.append((parameter[..., 1:, :] - parameter[..., :-1, :]).square().mean())
            terms.append((parameter[..., :, 1:] - parameter[..., :, :-1]).square().mean())
    return sum(terms) if terms else torch.tensor(0.0, device=next(system.parameters()).device)


def calibrate_prototypes(
    system: RawCameraSystem,
    generator: LabelPrototypeBasisGenerator,
    clean_targets: torch.Tensor,
    *,
    exposure_scale: float,
    epochs: int,
    learning_rate: float,
) -> PrototypeCalibrationSummary:
    """Optimize only prototypes and phase masks; coefficient-net parameters remain frozen."""
    if clean_targets.shape[0] != 10 or exposure_scale <= 0.0:
        raise ValueError("prototype calibration requires ten clean targets and fixed exposure")
    for parameter in generator.coefficient_net.parameters():
        parameter.requires_grad_(False)
    trainable = [generator.template_logits, *(parameter for name, parameter in system.named_parameters() if name.endswith("phase"))]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    labels = torch.arange(10, device=clean_targets.device)
    code = torch.zeros(10, generator.code_dim, device=clean_targets.device)
    last_loss = torch.tensor(float("nan"), device=clean_targets.device)
    gradients_finite = True
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        result = system.forward_from_generator(generator, code, labels)
        raw = result.raw_camera.tensor
        loss = torch.nn.functional.mse_loss((raw / exposure_scale).clamp(0.0, 1.0), clean_targets)
        loss = loss + 1.0e-5 * _phase_regularization(system)
        loss.backward()
        gradients_finite = gradients_finite and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainable)
        optimizer.step()
        last_loss = loss.detach()
        with torch.no_grad():
            observed = system.forward_from_generator(generator, code, labels).raw_camera.tensor
            normalized = (observed / exposure_scale).clamp(0.0, 1.0)
            history.append({"epoch": float(epoch + 1), "loss": float(last_loss.item()), "corr": float(correlation(normalized, clean_targets).mean().item()), "dice": float(foreground_dice(normalized, clean_targets).mean().item()), "high_saturation": float(high_saturation_ratio(observed, exposure_scale).item())})
    with torch.no_grad():
        result = system.forward_from_generator(generator, code, labels)
        normalized = (result.raw_camera.tensor / exposure_scale).clamp(0.0, 1.0)
        pair_prediction = normalized[:, None].expand(10, 10, *normalized.shape[1:]).reshape(100, *normalized.shape[1:])
        pair_target = clean_targets[None].expand(10, 10, *clean_targets.shape[1:]).reshape(100, *clean_targets.shape[1:])
        corr_matrix = correlation(pair_prediction, pair_target).reshape(10, 10)
        accuracy = (corr_matrix.argmax(dim=1) == labels).to(torch.float32).mean()
        return PrototypeCalibrationSummary(float(last_loss.item()), float(correlation(normalized, clean_targets).mean().item()), float(foreground_dice(normalized, clean_targets).mean().item()), float(high_saturation_ratio(result.raw_camera.tensor, exposure_scale).item()), gradients_finite, float(accuracy.item()), tuple(history))


def train_code_conditioned(
    system: RawCameraSystem,
    generator: LabelPrototypeBasisGenerator,
    codes: torch.Tensor,
    labels: torch.Tensor,
    teacher_targets: torch.Tensor,
    *,
    exposure_scale: float,
    prototype_evidence: EvidenceRecordV2,
    capacity_evidence: EvidenceRecordV2,
    expected_basis_hash: str,
    epochs: int,
    learning_rate: float,
) -> CodeTrainingSummary:
    require_passed_evidence(prototype_evidence, kind="prototype")
    require_passed_evidence(capacity_evidence, kind="capacity")
    if generator.basis_artifact_hash != expected_basis_hash:
        raise ValueError("generator basis artifact does not match approved capacity evidence")
    if codes.shape[0] != labels.shape[0] or teacher_targets.shape[0] != labels.shape[0]:
        raise ValueError("code, labels, and teacher targets must share batch size")
    for parameter in generator.coefficient_net.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam((parameter for parameter in list(generator.parameters()) + list(system.parameters()) if parameter.requires_grad), lr=learning_rate)
    last_loss = torch.tensor(float("nan"), device=codes.device)
    diagnostics = None
    raw = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        command, diagnostics = generator.forward_with_diagnostics(codes, labels)
        result = system.forward(system.command(command))
        raw = result.raw_camera.tensor
        normalized = (raw / exposure_scale).clamp(0.0, 1.0)
        paired = torch.nn.functional.mse_loss(normalized, teacher_targets)
        code_transport = -raw.var(dim=0, unbiased=False).mean()
        saturation = high_saturation_ratio(raw, exposure_scale)
        loss = paired + 0.01 * code_transport + 0.1 * diagnostics.clipping_fraction + 0.1 * saturation + 1.0e-5 * _phase_regularization(system)
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()
    assert diagnostics is not None and raw is not None
    normalized = (raw / exposure_scale).clamp(0.0, 1.0)
    return CodeTrainingSummary(float(last_loss.item()), float(correlation(normalized, teacher_targets).mean().item()), float(foreground_dice(normalized, teacher_targets).mean().item()), float(raw.var(dim=0, unbiased=False).mean().item()), float(diagnostics.clipping_fraction.item()))
