"""Small explicit training loops guarded by v2 evidence; no stage bypass exists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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


def _raw_from_command(system: RawCameraSystem, command: torch.Tensor, *, activation_checkpoint: bool) -> torch.Tensor:
    def forward(value: torch.Tensor) -> torch.Tensor:
        return system.forward(system.command(value)).raw_camera.tensor

    return checkpoint(forward, command, use_reentrant=False) if activation_checkpoint and torch.is_grad_enabled() else forward(command)


def _differentiable_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.flatten(1)
    truth = target.flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    truth = truth - truth.mean(dim=1, keepdim=True)
    pred_norm = pred.square().sum(dim=1).clamp_min(1.0e-12).sqrt()
    truth_norm = truth.square().sum(dim=1).clamp_min(1.0e-12).sqrt()
    return (pred * truth).sum(dim=1) / (pred_norm * truth_norm)


def _prototype_raw(system: RawCameraSystem, generator: LabelPrototypeBasisGenerator, labels: torch.Tensor, *, microbatch_size: int, activation_checkpoint: bool) -> torch.Tensor:
    rows = []
    for start in range(0, labels.numel(), microbatch_size):
        command = generator.prototype_command(labels[start : start + microbatch_size])
        rows.append(_raw_from_command(system, command, activation_checkpoint=activation_checkpoint))
    return torch.cat(rows)


def calibrate_prototypes(
    system: RawCameraSystem,
    generator: LabelPrototypeBasisGenerator,
    clean_targets: torch.Tensor,
    *,
    exposure_scale: float,
    epochs: int,
    learning_rate: float,
    microbatch_size: int = 10,
    activation_checkpoint: bool = False,
    initialization: str = "uniform_half",
    optimizer_steps_per_epoch: int = 1,
    correlation_weight: float = 0.0,
    dice_weight: float = 0.0,
    ce_weight: float = 0.0,
    ce_temperature: float = 0.07,
) -> PrototypeCalibrationSummary:
    """Optimize only prototypes and phase masks; coefficient-net parameters remain frozen."""
    if clean_targets.shape[0] != 10 or exposure_scale <= 0.0:
        raise ValueError("prototype calibration requires ten clean targets and fixed exposure")
    if microbatch_size < 1 or optimizer_steps_per_epoch < 1:
        raise ValueError("prototype microbatch must be positive")
    if initialization not in {"uniform_half", "clean_intensity_to_amplitude"}:
        raise ValueError("unsupported prototype initialization")
    if initialization == "clean_intensity_to_amplitude":
        with torch.no_grad():
            amplitude = clean_targets[:, 0].clamp(0.0, 1.0).sqrt().clamp(1.0e-4, 1.0 - 1.0e-4)
            generator.template_logits.copy_(torch.logit(amplitude))
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
        for _ in range(optimizer_steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            predictions = []
            for start in range(0, labels.numel(), microbatch_size):
                part_labels = labels[start : start + microbatch_size]
                command = generator(code[start : start + microbatch_size], part_labels)
                raw = _raw_from_command(system, command, activation_checkpoint=activation_checkpoint)
                normalized_part = (raw / exposure_scale).clamp(0.0, 1.0)
                predictions.append(normalized_part)
                losses.append(torch.nn.functional.mse_loss(normalized_part, clean_targets[start : start + microbatch_size], reduction="sum"))
            normalized_step = torch.cat(predictions)
            corr = _differentiable_correlation(normalized_step, clean_targets)
            soft_dice = (2.0 * (normalized_step * clean_targets).flatten(1).sum(dim=1) + 1.0e-6) / (normalized_step.square().flatten(1).sum(dim=1) + clean_targets.square().flatten(1).sum(dim=1) + 1.0e-6)
            pair_prediction = normalized_step[:, None].expand(10, 10, *normalized_step.shape[1:]).reshape(100, *normalized_step.shape[1:])
            pair_target = clean_targets[None].expand(10, 10, *clean_targets.shape[1:]).reshape(100, *clean_targets.shape[1:])
            correlations = _differentiable_correlation(pair_prediction, pair_target).reshape(10, 10)
            ce = torch.nn.functional.cross_entropy(correlations / ce_temperature, labels)
            loss = sum(losses) / clean_targets.numel()
            loss = loss + correlation_weight * (1.0 - corr.mean()) + dice_weight * (1.0 - soft_dice.mean()) + ce_weight * ce + 1.0e-5 * _phase_regularization(system)
            loss.backward()
            gradients_finite = gradients_finite and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainable)
            optimizer.step()
            last_loss = loss.detach()
        with torch.no_grad():
            observed = _prototype_raw(system, generator, labels, microbatch_size=microbatch_size, activation_checkpoint=False)
            normalized = (observed / exposure_scale).clamp(0.0, 1.0)
            history.append({"epoch": float(epoch + 1), "loss": float(last_loss.item()), "corr": float(correlation(normalized, clean_targets).mean().item()), "dice": float(foreground_dice(normalized, clean_targets).mean().item()), "prototype_accuracy": float(((correlation(normalized[:, None].expand(10, 10, *normalized.shape[1:]).reshape(100, *normalized.shape[1:]), clean_targets[None].expand(10, 10, *clean_targets.shape[1:]).reshape(100, *clean_targets.shape[1:])).reshape(10, 10).argmax(dim=1)) == labels).to(torch.float32).mean().item()), "high_saturation": float(high_saturation_ratio(observed, exposure_scale).item())})
    with torch.no_grad():
        raw_final = _prototype_raw(system, generator, labels, microbatch_size=microbatch_size, activation_checkpoint=False)
        normalized = (raw_final / exposure_scale).clamp(0.0, 1.0)
        pair_prediction = normalized[:, None].expand(10, 10, *normalized.shape[1:]).reshape(100, *normalized.shape[1:])
        pair_target = clean_targets[None].expand(10, 10, *clean_targets.shape[1:]).reshape(100, *clean_targets.shape[1:])
        corr_matrix = correlation(pair_prediction, pair_target).reshape(10, 10)
        accuracy = (corr_matrix.argmax(dim=1) == labels).to(torch.float32).mean()
        return PrototypeCalibrationSummary(float(last_loss.item()), float(correlation(normalized, clean_targets).mean().item()), float(foreground_dice(normalized, clean_targets).mean().item()), float(high_saturation_ratio(raw_final, exposure_scale).item()), gradients_finite, float(accuracy.item()), tuple(history))


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
    microbatch_size: int = 20,
    activation_checkpoint: bool = False,
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
    if microbatch_size < 1:
        raise ValueError("training microbatch must be positive")
    device = next(generator.parameters()).device
    if set(weights) != {"prototype", "paired_residual", "code_transport", "clipping", "saturation", "phase_tv"}:
        raise ValueError("code training requires all six explicit objective weights")
    counts = torch.bincount(labels, minlength=10)
    if (counts < 2).any() or not torch.equal(counts, torch.full_like(counts, counts[0])):
        raise ValueError("training requires an equal count of at least two codes for every label")
    if not torch.equal(generator(torch.zeros(1, generator.code_dim, device=device), torch.zeros(1, dtype=torch.long, device=device)), generator.prototype_command(torch.zeros(1, dtype=torch.long, device=device))):
        raise ValueError("zero code must map exactly to the label prototype")
    generator.template_logits.requires_grad_(False)
    for parameter in generator.coefficient_net.parameters():
        parameter.requires_grad_(True)
    trainable = [*generator.coefficient_net.parameters(), *(parameter for name, parameter in system.named_parameters() if name.endswith("phase"))]
    if not trainable:
        raise ValueError("no allowed v2 code-training parameters")
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    all_labels = torch.arange(10, device=device)
    with torch.no_grad():
        frozen_prototype_raw = (_prototype_raw(system, generator, all_labels, microbatch_size=microbatch_size, activation_checkpoint=False) / exposure_scale).clamp(0.0, 1.0).detach()
        teacher_prototypes = torch.stack([teacher_targets[labels == label].mean(dim=0) for label in range(10)])
    last_loss = torch.tensor(float("nan"), device=device)
    raw = None
    unclamped = None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        for index in _balanced_batch_indices(labels, batch_size=batch_size, seed=1234 + epoch):
            optimizer.zero_grad(set_to_none=True)
            batch_codes = codes.index_select(0, index)
            batch_labels = labels.index_select(0, index)
            batch_targets = teacher_targets.index_select(0, index)
            raw_rows, unclamped_rows = [], []
            clipping_penalties = []
            for start in range(0, batch_codes.shape[0], microbatch_size):
                part_codes = batch_codes[start : start + microbatch_size].to(device)
                part_labels = batch_labels[start : start + microbatch_size].to(device)
                command, part_diagnostics = generator.forward_with_diagnostics(part_codes, part_labels)
                raw_rows.append(_raw_from_command(system, command, activation_checkpoint=activation_checkpoint))
                unclamped_rows.append(part_diagnostics.unclamped_command)
                clipping_penalties.append(part_diagnostics.clipping_penalty * part_codes.shape[0])
            raw = torch.cat(raw_rows)
            unclamped = torch.cat(unclamped_rows)
            clipping_penalty = sum(clipping_penalties) / batch_codes.shape[0]
            normalized = (raw / exposure_scale).clamp(0.0, 1.0)
            zero_raw = (_prototype_raw(system, generator, all_labels, microbatch_size=microbatch_size, activation_checkpoint=activation_checkpoint) / exposure_scale).clamp(0.0, 1.0)
            prototype = torch.nn.functional.mse_loss(zero_raw, frozen_prototype_raw)
            batch_labels_device = batch_labels.to(device)
            paired = torch.nn.functional.mse_loss(normalized - zero_raw.index_select(0, batch_labels_device), batch_targets.to(device) - teacher_prototypes.index_select(0, batch_labels).to(device))
            factor = _differentiable_within_label_factor(raw, batch_labels_device)
            saturation = torch.relu(raw / exposure_scale - 1.0).square().mean()
            phase_tv = _phase_regularization(system)
            loss = (weights["prototype"] * prototype + weights["paired_residual"] * paired + weights["code_transport"] * (1.0 - factor) + weights["clipping"] * clipping_penalty + weights["saturation"] * saturation + weights["phase_tv"] * phase_tv)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
            history.append({"epoch": float(epoch + 1), "loss": float(last_loss.item()), "prototype": float(prototype.detach().item()), "paired_residual": float(paired.detach().item()), "code_transport_factor": float(factor.detach().item()), "clipping_penalty": float(clipping_penalty.detach().item()), "saturation_penalty": float(saturation.detach().item()), "phase_tv": float(phase_tv.detach().item())})
    assert raw is not None and unclamped is not None
    normalized = (raw / exposure_scale).clamp(0.0, 1.0)
    final_labels = labels.index_select(0, index).to(device)
    return CodeTrainingSummary(float(last_loss.item()), float(correlation(normalized, teacher_targets.index_select(0, index).to(device)).mean().item()), float(foreground_dice(normalized, teacher_targets.index_select(0, index).to(device)).mean().item()), float(_differentiable_within_label_factor(raw, final_labels).item()), float(((unclamped < 0.0) | (unclamped > 1.0)).to(raw.dtype).mean().item()), tuple(history))


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
