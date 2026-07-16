"""v67.1 diagnostic-only convergence gate for transparent thin-phase propagation."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPORT_SCHEMA = "v67_transparent_convergence_v2"
ACTIVE_SIZE = 100


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    value, reference = value, reference.to(value.dtype)
    return float((value - reference).norm() / reference.norm().clamp_min(1e-12))


def correlation(value: torch.Tensor, reference: torch.Tensor) -> float:
    x, y = value.real.float().reshape(-1), reference.real.float().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))


def guard_band_energy(field: torch.Tensor, guard_fraction: float = 0.05) -> float:
    """Return energy outside the centered guard region of a full grid."""
    return float(_per_sample_guard_band_energy(field, guard_fraction).mean())


def _per_sample_guard_band_energy(field: torch.Tensor, guard_fraction: float = 0.05) -> torch.Tensor:
    height, width = field.shape[-2:]
    active_h = max(1, int(round(height * (1.0 - 2.0 * guard_fraction))))
    active_w = max(1, int(round(width * (1.0 - 2.0 * guard_fraction))))
    top, left = (height - active_h) // 2, (width - active_w) // 2
    mask = torch.ones((height, width), dtype=torch.bool, device=field.device)
    mask[top : top + active_h, left : left + active_w] = False
    energy = field.abs().square().reshape(-1, height, width)
    return energy[:, mask].sum(dim=1) / energy.flatten(start_dim=1).sum(dim=1).clamp_min(1e-12)


def center_crop(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds field {(height, width)}")
    top, left = (height - size) // 2, (width - size) // 2
    return field[..., top : top + size, left : left + size]


def camera_contract_view(
    raw_intensity: torch.Tensor,
    *,
    input_pixel_size: float,
    output_pixel_size: float,
    output_size: int,
) -> torch.Tensor:
    """Extract the same physical camera FOV before comparing grids."""
    if tuple(raw_intensity.shape[-2:]) == (output_size, output_size):
        return raw_intensity
    target_pixels = max(1, int(round(output_size * output_pixel_size / input_pixel_size)))
    view = center_crop(raw_intensity, target_pixels)
    return F.interpolate(view, size=(output_size, output_size), mode="bilinear", align_corners=False)


def phase_circular_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean absolute wrapped phase error in radians."""
    return float(torch.angle(value * reference.conj()).abs().mean())


def total_energy(field: torch.Tensor) -> torch.Tensor:
    """Per-sample total energy; sum, not mean, preserves grid-area semantics."""
    return field.abs().square().reshape(field.shape[0], -1).sum(dim=1)


def _bootstrap_ci(values: torch.Tensor, *, seed: int = 1234, draws: int = 2000) -> list[float]:
    values = values.detach().float().cpu().flatten()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values.numel(), (draws, values.numel()), generator=generator)
    means = values[indices].mean(dim=1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def _center_embed(field: torch.Tensor, size: int) -> torch.Tensor:
    padded = torch.zeros((*field.shape[:-2], size, size), dtype=field.dtype, device=field.device)
    top, left = (size - field.shape[-2]) // 2, (size - field.shape[-1]) // 2
    padded[..., top : top + field.shape[-2], left : left + field.shape[-1]] = field
    return padded


def _independent_asm(field: torch.Tensor, distance: float, wavelength: float, pixel_size: float) -> torch.Tensor:
    """Reference ASM written independently of the model stack implementation."""
    height, width = field.shape[-2:]
    fy = torch.fft.fftfreq(height, pixel_size, device=field.device, dtype=field.real.dtype)
    fx = torch.fft.fftfreq(width, pixel_size, device=field.device, dtype=field.real.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radial = (wavelength * xx).square() + (wavelength * yy).square()
    root = (1.0 - radial).clamp_min(0.0).sqrt()
    transfer = torch.exp(1j * 2.0 * math.pi * distance * root / wavelength).to(field.dtype)
    transfer = torch.where(radial <= 1.0, transfer, torch.zeros_like(transfer))
    return torch.fft.ifft2(torch.fft.fft2(field) * transfer)


def zero_phase_reference_check(*, device: torch.device, grid: int, layers: int = 4) -> dict[str, float | bool]:
    from ldm_project.student.experimental.transparent_phase import TransparentIdentityDiffractiveStack

    torch.manual_seed(1234)
    input_field = torch.complex(
        torch.randn(1, 1, ACTIVE_SIZE, ACTIVE_SIZE, device=device),
        torch.randn(1, 1, ACTIVE_SIZE, ACTIVE_SIZE, device=device),
    )
    stack = TransparentIdentityDiffractiveStack(
        num_layers=layers,
        grid_size=ACTIVE_SIZE,
        layer_spacing=1.5e-3,
        wavelength=532e-9,
        pixel_size=2e-6,
        propagation_grid_size=grid,
        phase_init="small",
    ).to(device)
    with torch.no_grad():
        stack.phase.zero_()
        actual, _, _ = stack(input_field)
    expected = _center_embed(input_field, grid)
    for _ in range(layers):
        expected = _independent_asm(expected, 1.5e-3, 532e-9, 2e-6)
    error = relative_l2(actual, expected)
    return {"passed": error <= 1e-6, "relative_l2": error}


def _backend_relay_reference_check(*, base, device: torch.device) -> dict[str, Any]:
    """Compare readout grids with the same physical central chip field."""
    from ldm_project.student.experimental.model import OneShotOpticalGenerator

    if base.physics is None or base.one_shot.d2nn is None:
        raise ValueError("v67.1 requires physics and nested d2nn config")
    torch.manual_seed(4321)
    active = torch.complex(
        torch.randn(1, 1, ACTIVE_SIZE, ACTIVE_SIZE, device=device),
        torch.randn(1, 1, ACTIVE_SIZE, ACTIVE_SIZE, device=device),
    )
    cameras: dict[int, torch.Tensor] = {}
    for grid in (1024, 1536, 2048):
        physics = replace(base.physics, physical_forward_contract=True)
        d2nn = replace(base.one_shot.d2nn, propagation_grid_size=grid, exterior_boundary="transparent_identity")
        one_shot = replace(base.one_shot, physical_forward_contract=True, d2nn=d2nn)
        model = OneShotOpticalGenerator(physics, one_shot, image_size=base.first_stage.image_size).to(device).eval()
        chip = _center_embed(active, grid)
        with torch.no_grad():
            readout = model.back_path(
                chip,
                input_pixel_size=d2nn.pixel_size,
                output_size=(base.first_stage.image_size, base.first_stage.image_size),
                output_pixel_size=base.physics.optics.pixel_size,
            )
            cameras[grid] = camera_contract_view(
                readout["camera_intensity_raw"],
                input_pixel_size=d2nn.pixel_size,
                output_pixel_size=base.physics.optics.pixel_size,
                output_size=base.first_stage.image_size,
            ).detach().cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference = cameras[2048]
    values = torch.stack([
        torch.tensor(correlation(cameras[grid], reference))
        for grid in (1024, 1536, 2048)
    ])
    return {str(grid): correlation(cameras[grid], reference) for grid in (1024, 1536, 2048)} | {
        "all_grids_pass": bool(torch.all(values >= 0.999)),
    }


def _metric_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": _bootstrap_ci(values),
    }


def choose_grid(results: dict[int, dict[str, float]], *, reference: int = 2048) -> dict[str, Any]:
    """Select the smallest grid meeting all v67.1 gates."""
    for grid in (1024, 1536):
        item = results.get(grid, {})
        if (
            item.get("camera_intensity_corr_mean", item.get("camera_intensity_corr", 0.0)) >= 0.999
            and item.get("central_complex_field_relative_l2_mean", item.get("central_field_relative_l2", float("inf"))) <= 1e-3
            and item.get("total_energy_relative_difference_mean", item.get("energy_relative_difference", float("inf"))) <= 1e-3
            and item.get("guard_band_energy_ratio", float("inf")) <= 1e-4
        ):
            return {"status": "converged", "selected_grid": grid, "reference_grid": reference}
    return {"status": "grid-not-converged", "selected_grid": None, "reference_grid": reference}


def run(*, output: str | Path, device: str = "cuda", train_config: str | Path | None = None,
        checkpoint: str | Path | None = None, sample_tensors: str | Path | None = None) -> dict[str, Any]:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v67.1 full-geometry convergence")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "preflight-input-required" if not (train_config and checkpoint and sample_tensors) else "running",
        "diagnostic_only": True,
        "device": str(resolved),
        "grids": [1024, 1536, 2048],
        "active_window": {"shape": [ACTIVE_SIZE, ACTIVE_SIZE], "pixel_pitch_m": 2e-6},
        "camera_observation_contract": {"shape": [200, 200], "pixel_pitch_m": 2.5e-6, "fov_m": 500e-6},
        "gates": {
            "camera_intensity_corr_mean": 0.999,
            "central_complex_field_relative_l2_mean": 1e-3,
            "total_energy_relative_difference_mean": 1e-3,
            "guard_band_energy_ratio": 1e-4,
        },
        "decision": "do-not-train-until-v67.1-results-are-recorded",
    }
    if train_config and checkpoint and sample_tensors:
        from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config

        payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
        codes, labels = payload["codes"], payload["class_labels"]
        base = load_training_config(train_config)
        if base.physics is None or base.one_shot.d2nn is None:
            raise ValueError("v67.1 requires physics and nested d2nn config")
        raw: dict[int, dict[str, torch.Tensor]] = {}
        for grid in (1024, 1536, 2048):
            physics = replace(base.physics, physical_forward_contract=True)
            d2nn = replace(base.one_shot.d2nn, propagation_grid_size=grid, exterior_boundary="transparent_identity")
            one_shot = replace(base.one_shot, physical_forward_contract=True, d2nn=d2nn)
            model = OneShotOpticalGenerator(physics, one_shot, image_size=base.first_stage.image_size).to(resolved).eval()
            load_one_shot_checkpoint(model, torch.load(checkpoint, map_location=resolved, weights_only=False))
            cameras, chips = [], []
            with torch.no_grad():
                for code, label in zip(codes, labels):
                    result = model(code[None].to(resolved), label[None].to(resolved), return_optical_fields=False)
                    cameras.append(
                        camera_contract_view(
                            result["physical_camera_intensity_raw"],
                            input_pixel_size=d2nn.pixel_size,
                            output_pixel_size=base.physics.optics.pixel_size,
                            output_size=base.first_stage.image_size,
                        ).detach().cpu()
                    )
                    chips.append(torch.complex(result["chip_real"], result["chip_imag"]).detach().cpu())
            raw[grid] = {"camera": torch.cat(cameras), "chip": torch.cat(chips)}
            del model
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
        reference = raw[2048]
        results: dict[int, dict[str, Any]] = {}
        for grid, values in raw.items():
            value_chip = center_crop(values["chip"], ACTIVE_SIZE)
            reference_chip = center_crop(reference["chip"], ACTIVE_SIZE)
            camera_corr = torch.stack([
                torch.tensor(correlation(value, target))
                for value, target in zip(values["camera"], reference["camera"])
            ])
            complex_l2 = torch.stack([
                torch.tensor(relative_l2(value, target))
                for value, target in zip(value_chip, reference_chip)
            ])
            intensity_corr = torch.stack([
                torch.tensor(correlation(value.abs().square(), target.abs().square()))
                for value, target in zip(value_chip, reference_chip)
            ])
            phase_error = torch.stack([
                torch.tensor(phase_circular_error(value, target))
                for value, target in zip(value_chip, reference_chip)
            ])
            energy_difference = (total_energy(values["chip"]) / total_energy(reference["chip"]).clamp_min(1e-12) - 1.0).abs()
            guard = _per_sample_guard_band_energy(values["chip"])
            results[grid] = {
                "camera_intensity_corr": _metric_summary(camera_corr),
                "camera_intensity_corr_mean": float(camera_corr.mean()),
                "central_complex_field_relative_l2": _metric_summary(complex_l2),
                "central_complex_field_relative_l2_mean": float(complex_l2.mean()),
                "central_intensity_corr": _metric_summary(intensity_corr),
                "central_phase_circular_error_rad": _metric_summary(phase_error),
                "total_energy_relative_difference": _metric_summary(energy_difference),
                "total_energy_relative_difference_mean": float(energy_difference.mean()),
                "guard_band_energy": _metric_summary(guard),
                "guard_band_energy_ratio": float(guard.mean()),
            }
        report["results"] = {str(grid): value for grid, value in results.items()}
        report["reference_checks"] = {
            str(grid): zero_phase_reference_check(device=resolved, grid=grid)
            for grid in (1024, 1536, 2048)
        }
        report["backend_relay_reference_check"] = _backend_relay_reference_check(base=base, device=resolved)
        decision = choose_grid(results)
        report.update(decision)
        report["status"] = decision["status"]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-tensors", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
