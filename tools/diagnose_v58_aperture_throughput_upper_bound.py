#!/usr/bin/env python3
"""v58 diagnostic-only aperture-throughput broad-isolation upper bound.

This never changes a training checkpoint or the formal hardware contract.  It
reruns the v56 class-balanced sample twice with identical checkpoint, codes,
labels, and relays.  The counterfactual retains the full padded propagation
field between D2NN layers; each trained 100x100 metasurface remains centered
in that larger field and is identity outside its physical aperture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import nn

from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config
from ldm_project.one_shot_optical_layers import angular_spectrum_propagate


PLANE_ORDER = (
    "slm_output", "pre_encoder_relay_image", "encoder_input", "encoder_layer_1",
    "encoder_layer_2", "encoder_output", "ols_input", "ols_output", "decoder_input",
    "decoder_layer_1", "decoder_layer_2", "decoder_output", "chip", "camera_field",
)
METRICS = ("amplitude", "complex", "intensity")


def _energy(field: torch.Tensor) -> torch.Tensor:
    return field.abs().square().flatten(1).sum(1)


def _center_embed(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    if height > size or width > size:
        raise ValueError(f"cannot embed {height}x{width} field into {size}x{size}")
    output = field.new_zeros(*field.shape[:-2], size, size)
    top, left = (size - height) // 2, (size - width) // 2
    output[..., top : top + height, left : left + width] = field
    return output


class _NoInterlayerCropStack(nn.Module):
    """Diagnostic stack that removes only the numerical finite-crop boundary."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source
        self.num_layers = int(source.num_layers)
        self.grid_size = int(source.grid_size)
        self.propagation_grid_size = int(source.propagation_grid_size)
        self.layer_spacing = float(source.layer_spacing)
        self.wavelength = float(source.wavelength)
        self.pixel_size = float(source.pixel_size)

    def phase_transfers(self, *args, **kwargs):
        return self.source.phase_transfers(*args, **kwargs)

    def forward(self, field: torch.Tensor, *, mask_component: str = "trained", phase_layer_mask=()):
        if field.shape[-2:] == (self.grid_size, self.grid_size):
            field = _center_embed(field, self.propagation_grid_size)
        elif field.shape[-2:] != (self.propagation_grid_size, self.propagation_grid_size):
            raise ValueError(f"unexpected diagnostic field shape {tuple(field.shape[-2:])}")
        transfers = self.source.phase_transfers(mask_component, tuple(phase_layer_mask)).to(field.dtype)
        layers, values = [], {key: [] for key in (
            "input_energy", "padded_input_energy", "propagated_full_energy", "cropped_energy",
            "crop_retention", "propagation_energy_ratio",
        )}
        for layer in range(self.num_layers):
            transfer = _center_embed(transfers[layer].unsqueeze(0), self.propagation_grid_size)
            # Outside the trained finite device, the upper-bound propagation is identity modulation.
            transfer = transfer + (transfer == 0).to(transfer.dtype)
            field = field * transfer
            input_energy = _energy(field)
            field = angular_spectrum_propagate(field, self.layer_spacing, self.wavelength, self.pixel_size)
            propagated_energy = _energy(field)
            layers.append(field)
            values["input_energy"].append(input_energy)
            values["padded_input_energy"].append(input_energy)
            values["propagated_full_energy"].append(propagated_energy)
            values["cropped_energy"].append(propagated_energy)
            values["crop_retention"].append(torch.ones_like(propagated_energy))
            values["propagation_energy_ratio"].append(propagated_energy / input_energy.clamp_min(1e-12))
        return field, torch.stack(layers, dim=1), {key: torch.stack(value, dim=1) for key, value in values.items()}


def _field(record: dict[str, Any]) -> torch.Tensor:
    return torch.complex(record["real"], record["imag"]).detach().to(torch.complex128)


def _pair_distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.to(torch.complex128), b.to(torch.complex128)
    a_amp, b_amp = a.abs(), b.abs()
    def distance(left, right):
        return float((left - right).norm() / ((left.norm() + right.norm()).mul(0.5).clamp_min(1e-12)))
    phase = torch.angle((a.conj() * b).sum())
    aligned = a * torch.exp(1j * phase).to(a.dtype)
    return {"amplitude": distance(a_amp, b_amp), "complex": distance(aligned, b),
            "intensity": distance(a_amp.square(), b_amp.square())}


def _summarize(fields: dict[str, dict[str, object]], labels: torch.Tensor) -> list[dict[str, object]]:
    summary = []
    for name in PLANE_ORDER:
        values = _field(fields[name])
        same, cross = {metric: [] for metric in METRICS}, {metric: [] for metric in METRICS}
        for left in range(values.shape[0]):
            for right in range(left + 1, values.shape[0]):
                target = same if int(labels[left]) == int(labels[right]) else cross
                for metric, value in _pair_distance(values[left, 0], values[right, 0]).items():
                    target[metric].append(value)
        same_median = {metric: median(value) for metric, value in same.items()}
        cross_median = {metric: median(value) for metric, value in cross.items()}
        summary.append({
            "plane": name,
            "energy_mean": float(_energy(values).mean()),
            "same_class_median": same_median,
            "cross_class_median": cross_median,
            "separation_signal": {metric: cross_median[metric] - same_median[metric] for metric in METRICS},
        })
    return summary


def _validate_sample(payload: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    codes, labels = payload.get("codes"), payload.get("class_labels")
    if not torch.is_tensor(codes) or not torch.is_tensor(labels) or codes.shape[0] != labels.shape[0]:
        raise ValueError("sample tensor bundle must provide aligned codes and class_labels")
    counts = {label: int((labels == label).sum()) for label in range(10)}
    if counts != {label: 2 for label in range(10)}:
        raise ValueError(f"expected exactly two samples for each class 0-9, got {counts}")
    return codes, labels


def _camera_signal(planes: list[dict[str, object]]) -> dict[str, float]:
    item = next(item for item in planes if item["plane"] == "camera_field")
    return dict(item["separation_signal"])


def run(*, train_config: str | Path, checkpoint: str | Path, sample_tensors: str | Path,
        output: str | Path, device: str = "cuda") -> dict[str, object]:
    train_config, checkpoint, sample_tensors, output = map(Path, (train_config, checkpoint, sample_tensors, output))
    if not checkpoint.is_file() or not sample_tensors.is_file():
        raise ValueError("checkpoint and sample tensor bundle must exist")
    train_cfg = load_training_config(train_config)
    if train_cfg.physics is None:
        raise ValueError("v58 requires physics embedded in the training config")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-geometry diagnostic; no diagnostic-only CPU substitute is run")
    sample_payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
    codes, labels = _validate_sample(sample_payload)
    model = OneShotOpticalGenerator(train_cfg.physics, train_cfg.one_shot, image_size=train_cfg.first_stage.image_size).to(resolved_device)
    load_one_shot_checkpoint(model, torch.load(checkpoint, map_location=resolved_device, weights_only=False))
    model.eval()
    with torch.no_grad():
        baseline = model(codes.to(resolved_device), labels.to(resolved_device), return_optical_fields=True)
        model.encoder = _NoInterlayerCropStack(model.encoder)
        model.decoder = _NoInterlayerCropStack(model.decoder)
        upper_bound = model(codes.to(resolved_device), labels.to(resolved_device), return_optical_fields=True)
    baseline_planes = _summarize(baseline["optical_fields"], labels)
    upper_planes = _summarize(upper_bound["optical_fields"], labels)
    baseline_signal, upper_signal = _camera_signal(baseline_planes), _camera_signal(upper_planes)
    gains = {metric: upper_signal[metric] / baseline_signal[metric] if baseline_signal[metric] > 0.0 else None for metric in METRICS}
    restored = any(gain is not None and gain >= 2.0 for gain in gains.values())
    report: dict[str, object] = {
        "status": "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "counterfactual": "no_interlayer_d2nn_finite_crop; centered trained masks, identity outside aperture",
        "inputs": {"train_config": str(train_config), "checkpoint": str(checkpoint), "sample_tensors": str(sample_tensors), "device": str(resolved_device)},
        "invariants": {"zero_training": True, "same_codes_and_labels": True, "same_front_and_back_relays": True, "formal_hardware_contract_changed": False},
        "baseline": {"planes": baseline_planes, "camera_separation_signal": baseline_signal},
        "upper_bound": {"planes": upper_planes, "camera_separation_signal": upper_signal},
        "camera_separation_gain": gains,
        "decision": ("aperture-upper-bound-restores-separation-manual-CMOS-review-required" if restored
                     else "aperture-upper-bound-no-separation-recovery"),
        "next_decision_gate": ("manual review of saved counterfactual true CMOS intensity before discussing formal aperture redesign"
                               if restored else "investigate weak upstream SLM/generator/objective cross-class separation; do not tune decoder or losses"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"class_labels": labels, "baseline_physical_camera_plane": baseline["physical_camera_plane"].detach().cpu(),
                "upper_bound_physical_camera_plane": upper_bound["physical_camera_plane"].detach().cpu()}, output.with_suffix(".pt"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-tensors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
