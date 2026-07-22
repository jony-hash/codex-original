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
    objective_history: tuple[dict[str, float], ...]


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
    batch_size: int,
    weights: dict[str, float],
    allow_diagnostic_evidence: bool = False,
) -> CodeTrainingSummary:
    if not allow_diagnostic_evidence:
        require_passed_evidence(prototype_evidence, kind="prototype")
        require_passed_evidence(capacity_evidence, kind="capacity")
    if generator.basis_artifact_hash != expected_basis_hash:
        raise ValueError("generator basis artifact does not match approved capacity evidence")
    if codes.shape[0] != labels.shape[0] or teacher_targets.shape[0] != labels.shape[0]:
        raise ValueError("code, labels, and teacher targets must share batch size")
    if batch_size < 20 or batch_size % 10:
        raise ValueError("code training requires a balanced batch size divisible by ten")
    if set(weights) != {"prototype", "paired_residual", "code_transport", "clipping", "saturation", "phase_tv"}:
        raise ValueError("code training requires all six explicit objective weights")
    counts = torch.bincount(labels, minlength=10)
    if (counts < 2).any() or not torch.equal(counts, torch.full_like(counts, counts[0])):
        raise ValueError("training requires an equal count of at least two codes for every label")
    if not torch.equal(generator(torch.zeros(1, generator.code_dim, device=codes.device), torch.zeros(1, dtype=torch.long, device=codes.device)), generator.prototype_command(torch.zeros(1, dtype=torch.long, device=codes.device))):
        raise ValueError("zero code must map exactly to the label prototype")
    generator.template_logits.requires_grad_(False)
    for parameter in generator.coefficient_net.parameters():
        parameter.requires_grad_(True)
    trainable = [*generator.coefficient_net.parameters(), *(parameter for name, parameter in system.named_parameters() if name.endswith("phase"))]
    if not trainable:
        raise ValueError("no allowed v2 code-training parameters")
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    all_labels = torch.arange(10, device=codes.device)
    with torch.no_grad():
        frozen_prototype_raw = (system.forward(system.command(generator.prototype_command(all_labels))).raw_camera.tensor / exposure_scale).clamp(0.0, 1.0).detach()
        teacher_prototypes = torch.stack([teacher_targets[labels == label].mean(dim=0) for label in range(10)])
    last_loss = torch.tensor(float("nan"), device=codes.device)
    diagnostics = None
    raw = None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        for index in _balanced_batch_indices(labels, batch_size=batch_size, seed=1234 + epoch):
            optimizer.zero_grad(set_to_none=True)
            batch_codes = codes.index_select(0, index)
            batch_labels = labels.index_select(0, index)
            batch_targets = teacher_targets.index_select(0, index)
            command, diagnostics = generator.forward_with_diagnostics(batch_codes, batch_labels)
            raw = system.forward(system.command(command)).raw_camera.tensor
            normalized = (raw / exposure_scale).clamp(0.0, 1.0)
            zero_raw = (system.forward(system.command(generator.prototype_command(all_labels))).raw_camera.tensor / exposure_scale).clamp(0.0, 1.0)
            prototype = torch.nn.functional.mse_loss(zero_raw, frozen_prototype_raw)
            paired = torch.nn.functional.mse_loss(normalized - zero_raw.index_select(0, batch_labels), batch_targets - teacher_prototypes.index_select(0, batch_labels))
            factor = _differentiable_within_label_factor(raw, batch_labels)
            saturation = torch.relu(raw / exposure_scale - 1.0).square().mean()
            phase_tv = _phase_regularization(system)
            loss = (weights["prototype"] * prototype + weights["paired_residual"] * paired + weights["code_transport"] * (1.0 - factor) + weights["clipping"] * diagnostics.clipping_penalty + weights["saturation"] * saturation + weights["phase_tv"] * phase_tv)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
            history.append({"epoch": float(epoch + 1), "loss": float(last_loss.item()), "prototype": float(prototype.detach().item()), "paired_residual": float(paired.detach().item()), "code_transport_factor": float(factor.detach().item()), "clipping_penalty": float(diagnostics.clipping_penalty.detach().item()), "saturation_penalty": float(saturation.detach().item()), "phase_tv": float(phase_tv.detach().item())})
    assert diagnostics is not None and raw is not None
    normalized = (raw / exposure_scale).clamp(0.0, 1.0)
    return CodeTrainingSummary(float(last_loss.item()), float(correlation(normalized, teacher_targets.index_select(0, index)).mean().item()), float(foreground_dice(normalized, teacher_targets.index_select(0, index)).mean().item()), float(_differentiable_within_label_factor(raw, labels.index_select(0, index)).item()), float(diagnostics.clipping_fraction.item()), tuple(history))


def _balanced_batch_indices(labels: torch.Tensor, *, batch_size: int, seed: int):
    per_label = batch_size // 10
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = []
    for label in range(10):
        matches = torch.nonzero(labels == label, as_tuple=False).flatten().cpu()
        groups.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)))
    for start in range(0, groups[0].numel(), per_label):
        yield torch.cat([group[start : start + per_label] for group in groups]).to(labels.device)


def _differentiable_within_label_factor(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_mean = torch.stack([values[labels == label].mean(dim=0) for label in range(10)])
    centered = values - label_mean.index_select(0, labels)
    return centered.square().sum() / (values - values.mean()).square().sum().clamp_min(1.0e-12)
