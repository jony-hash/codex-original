"""v67.2 diagnostic-only transparent propagation and training admission gate.

The tool deliberately separates numerical convergence from model quality.  It
never changes the formal ``finite_crop`` path and never starts A/B training
unless one unique padded grid passes every field, provenance, gradient and
memory gate.  A report with missing inputs is a valid diagnostic result, not a
permission to train.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch


REPORT_SCHEMA = "v67_transparent_convergence_v3"
ACTIVE_SIZE = 100
WAVELENGTH = 532.0e-9
PIXEL_SIZE = 2.0e-6
LAYER_SPACING = 1.5e-3


def center_crop(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds field {(height, width)}")
    top, left = (height - size) // 2, (width - size) // 2
    return field[..., top : top + size, left : left + size]


def center_embed(field: torch.Tensor, size: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    if height > size or width > size:
        raise ValueError(f"field {(height, width)} exceeds grid {size}")
    result = field.new_zeros(*field.shape[:-2], size, size)
    top, left = (size - height) // 2, (size - width) // 2
    result[..., top : top + height, left : left + width] = field
    return result


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    value, reference = value, reference.to(value.dtype)
    return float((value - reference).norm() / reference.norm().clamp_min(1.0e-12))


def total_energy(field: torch.Tensor) -> torch.Tensor:
    """Per-sample sum(|E|²); no area-normalized mean across grids."""
    return field.abs().square().reshape(field.shape[0], -1).sum(dim=1)


def phase_circular_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.angle(value * reference.conj()).abs().mean())


def _per_sample_guard_ratio(field: torch.Tensor, fraction: float = 0.05) -> torch.Tensor:
    height, width = field.shape[-2:]
    border = max(1, int(round(min(height, width) * fraction)))
    mask = torch.zeros(height, width, dtype=torch.bool, device=field.device)
    mask[:border] = True
    mask[-border:] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    energy = field.abs().square().reshape(field.shape[0], height, width)
    return energy[:, mask].sum(dim=1) / energy.flatten(start_dim=1).sum(dim=1).clamp_min(1.0e-12)


def correlation(value: torch.Tensor, reference: torch.Tensor) -> float:
    x = value.float().reshape(-1)
    y = reference.float().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1.0e-12))


def bootstrap_ci(values: torch.Tensor, *, draws: int = 2000, seed: int = 1234) -> list[float]:
    values = values.detach().float().cpu().flatten()
    if values.numel() == 0:
        return [float("nan"), float("nan")]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values.numel(), (draws, values.numel()), generator=generator)
    means = values[indices].mean(dim=1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": bootstrap_ci(values),
    }


def direct_rayleigh_sommerfeld_reference(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_size: float,
) -> torch.Tensor:
    """Small-grid direct RS reference (guarded to avoid accidental OOM)."""
    height, width = field.shape[-2:]
    if max(height, width) > 64:
        raise ValueError("direct RS reference is limited to grids <= 64")
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=field.device, dtype=field.real.dtype) - height / 2) * pixel_size,
        (torch.arange(width, device=field.device, dtype=field.real.dtype) - width / 2) * pixel_size,
        indexing="ij",
    )
    source = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
    target = source.clone()
    delta = target[:, None, :] - source[None, :, :]
    radius = (delta.square().sum(dim=-1) + distance**2).sqrt()
    kernel = distance / (1j * wavelength * radius.square()) * torch.exp(
        1j * (2.0 * math.pi / wavelength) * radius
    )
    values = field.reshape(*field.shape[:-2], -1)
    result = values @ kernel.T * pixel_size**2
    return result.reshape(*field.shape[:-2], height, width)


def direct_rs_check(*, device: torch.device, grid: int = 16) -> dict[str, Any]:
    from ldm_project.optics.propagation import angular_spectrum_propagate

    # A localized field avoids comparing the periodic wrap-around of a tiny ASM
    # grid with a finite-aperture RS integral.  The central ROI is the declared
    # observable; the surrounding direct grid supplies the non-periodic guard.
    reference_grid = max(48, grid * 3)
    yy, xx = torch.meshgrid(
        torch.arange(reference_grid, device=device, dtype=torch.float32)
        - reference_grid / 2,
        torch.arange(reference_grid, device=device, dtype=torch.float32)
        - reference_grid / 2,
        indexing="ij",
    )
    envelope = torch.exp(-(xx.square() + yy.square()) / 20.0)
    # Smooth, band-limited tilt is the declared reference input.  Random
    # pixel-to-pixel phase would intentionally violate ASM sampling and make
    # this a source-bandwidth test rather than a propagation reference.
    phase = 0.05 * xx
    field = torch.polar(envelope, phase)[None, None].to(torch.complex64)
    direct = direct_rayleigh_sommerfeld_reference(
        field, LAYER_SPACING, WAVELENGTH, PIXEL_SIZE
    )
    asm = angular_spectrum_propagate(field, LAYER_SPACING, WAVELENGTH, PIXEL_SIZE)
    direct_intensity = center_crop(direct.abs().square(), grid)
    asm_intensity = center_crop(asm.abs().square(), grid)
    direct_intensity = direct_intensity / direct_intensity.amax().clamp_min(1e-12)
    asm_intensity = asm_intensity / asm_intensity.amax().clamp_min(1e-12)
    error = relative_l2(direct_intensity, asm_intensity)
    return {
        "grid": grid,
        "reference_grid": reference_grid,
        "central_normalized_intensity_relative_l2": error,
        "passed": error <= 0.01,
    }


def compare_planes(value: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    size = min(value.shape[-1], reference.shape[-1])
    value, reference = center_crop(value, size), center_crop(reference, size)
    complex_l2 = torch.tensor(
        [relative_l2(v[None], r[None]) for v, r in zip(value, reference)],
        device=value.device,
    )
    amplitude_l2 = torch.tensor(
        [relative_l2(v.abs()[None], r.abs()[None]) for v, r in zip(value, reference)],
        device=value.device,
    )
    phase_error = torch.tensor(
        [phase_circular_error(v[None], r[None]) for v, r in zip(value, reference)],
        device=value.device,
    )
    intensity_corr = torch.tensor(
        [correlation(v.abs().square(), r.abs().square()) for v, r in zip(value, reference)],
        device=value.device,
    )
    energy = (total_energy(value) / total_energy(reference).clamp_min(1.0e-12) - 1.0).abs()
    return {
        "complex_field_relative_l2": _summary(complex_l2),
        "amplitude_relative_l2": _summary(amplitude_l2),
        "weighted_phase_mae_rad": _summary(phase_error),
        "intensity_corr": _summary(intensity_corr),
        "total_energy_relative_difference": _summary(energy),
        "guard_band_ratio": _summary(_per_sample_guard_ratio(value)),
    }


def candidate_passes(item: Mapping[str, Any]) -> bool:
    """Apply the strict field gate to one candidate result."""
    means = {key: float(value.get("mean", float("inf"))) for key, value in item.items() if isinstance(value, Mapping)}
    return (
        means.get("complex_field_relative_l2", float("inf")) <= 1e-3
        and means.get("amplitude_relative_l2", float("inf")) <= 1e-3
        and means.get("weighted_phase_mae_rad", float("inf")) <= 1e-3
        and means.get("intensity_corr", -float("inf")) >= 0.999
        and means.get("total_energy_relative_difference", float("inf")) <= 1e-3
        and means.get("guard_band_ratio", float("inf")) <= 1e-4
        and float(item.get("discarded_energy_ratio", float("inf"))) <= 1e-4
        and float(item.get("camera_intensity_corr", -float("inf"))) >= 0.999
        and bool(item.get("reference_is_self", True)) is False
        and bool(item.get("sampling_fov_match", False))
        and bool(item.get("direct_rs_pass", False))
        and bool(item.get("all_plane_gates_pass", False))
    )


def choose_candidate(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for state_grid, fft_grid in ((1536, 3072), (2048, 4096)):
        key = f"state{state_grid}_fft{fft_grid}"
        if key in results and candidate_passes(results[key]):
            return {
                "status": "converged",
                "selected_state_grid": state_grid,
                "selected_fft_grid": fft_grid,
            }
    return {
        "status": "grid-not-converged",
        "selected_state_grid": None,
        "selected_fft_grid": None,
    }


def gradient_admission(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate precomputed full-objective gradient metrics without relaxing gates."""
    finite = bool(metrics.get("finite", False))
    norm_ratio = float(metrics.get("gradient_norm_ratio", float("inf")))
    result = {
        "finite": finite,
        "loss_relative_difference": float(metrics.get("loss_relative_difference", float("inf"))),
        "encoder_gradient_cosine": float(metrics.get("encoder_gradient_cosine", -float("inf"))),
        "decoder_gradient_cosine": float(metrics.get("decoder_gradient_cosine", -float("inf"))),
        "generator_gradient_cosine": float(metrics.get("generator_gradient_cosine", -float("inf"))),
        "gradient_norm_ratio": norm_ratio,
        "gradient_relative_l2": float(metrics.get("gradient_relative_l2", float("inf"))),
    }
    result["passed"] = (
        finite
        and result["loss_relative_difference"] <= 0.002
        and result["encoder_gradient_cosine"] >= 0.999
        and result["decoder_gradient_cosine"] >= 0.999
        and result["generator_gradient_cosine"] >= 0.999
        and 0.98 <= norm_ratio <= 1.02
        and result["gradient_relative_l2"] <= 0.02
    )
    return result


def memory_admission(*, peak_reserved_bytes: int, available_bytes: int) -> dict[str, Any]:
    """Apply the 85% reserved-memory gate used by the transparent trainer."""
    ratio = float(peak_reserved_bytes) / max(float(available_bytes), 1.0)
    return {
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "available_bytes": int(available_bytes),
        "reserved_ratio": ratio,
        "passed": ratio <= 0.85,
    }


def _config_for_grid(base, state_grid: int, fft_grid: int):
    d2nn = replace(
        base.one_shot.d2nn,
        propagation_grid_size=state_grid,
        exterior_boundary="transparent_identity",
        transparent_propagation_method="padded_band_limited_asm",
        transparent_fft_grid_size=fft_grid,
    )
    return replace(base.one_shot, physical_forward_contract=True, d2nn=d2nn)


def run(
    *,
    output: str | Path,
    device: str = "cuda",
    train_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    sample_tensors: str | Path | None = None,
) -> dict[str, Any]:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v67.2 transparent convergence")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "preflight-input-required",
        "device": str(resolved),
        "candidate_order": [
            {"state_grid": 1536, "fft_grid": 3072},
            {"state_grid": 2048, "fft_grid": 4096},
        ],
        "physical_contract": {
            "pixel_pitch_m": PIXEL_SIZE,
            "state_fov_m": None,
            "wavelength_m": WAVELENGTH,
            "phase_plane_spacing_m": LAYER_SPACING,
            "active_window": [ACTIVE_SIZE, ACTIVE_SIZE],
        },
        "gates": {
            "complex_field_relative_l2": 1e-3,
            "amplitude_relative_l2": 1e-3,
            "weighted_phase_mae_rad": 1e-3,
            "intensity_corr": 0.999,
            "total_energy_relative_difference": 1e-3,
            "guard_band_ratio": 1e-4,
            "discarded_energy_ratio": 1e-4,
            "camera_intensity_corr": 0.999,
            "loss_relative_difference": 0.002,
            "gradient_cosine": 0.999,
            "gradient_norm_ratio": [0.98, 1.02],
            "gradient_relative_l2": 0.02,
        },
        "reference_checks": {"direct_rs": "not-run"},
        "gradient_admission": {
            "status": "not-run",
            "reason": "field/reference gate failed before full one_shot_loss VJP",
        },
        "memory_admission": {
            "status": "not-run",
            "microbatch": 1,
            "gradient_accumulation": 16,
            "dtype": "complex64",
        },
        "training_field_converged": False,
        "camera_observable_converged": False,
        "decision": "do-not-train-until-v67.2-gates-pass",
    }
    if not (train_config and checkpoint and sample_tensors):
        report["reason"] = "train-config, checkpoint and sample-tensors are required"
    else:
        from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config

        base = load_training_config(train_config)
        payload = torch.load(sample_tensors, map_location="cpu", weights_only=False)
        codes, labels = payload["codes"], payload["class_labels"]
        checkpoint_payload = torch.load(checkpoint, map_location=resolved, weights_only=False)
        if base.physics is None or base.one_shot.d2nn is None:
            raise ValueError("v67.2 requires physics and nested d2nn config")
        report["physical_contract"]["state_fov_m"] = {
            str(grid): grid * PIXEL_SIZE for grid in (1536, 2048)
        }
        report["physical_contract"]["sampling_fov_match"] = False
        report["physical_contract"]["sampling_fov_reason"] = (
            "candidate state grids at fixed 2 um pitch have different FOV; "
            "no physical-coordinate equivalence may be assumed"
        )
        raw: dict[str, dict[str, torch.Tensor]] = {}
        report["reference_checks"]["direct_rs"] = direct_rs_check(device=resolved)
        for state_grid, fft_grid in ((1536, 3072), (2048, 4096)):
            physics = replace(base.physics, physical_forward_contract=True)
            config = _config_for_grid(base, state_grid, fft_grid)
            model = OneShotOpticalGenerator(
                physics, config, image_size=base.first_stage.image_size
            ).to(resolved).eval()
            load_one_shot_checkpoint(model, checkpoint_payload)
            cameras, chips, encoder_layers, decoder_layers, discarded_ratios = [], [], [], [], []
            with torch.no_grad():
                for code, label in zip(codes[:20], labels[:20]):
                    result = model(code[None].to(resolved), label[None].to(resolved))
                    cameras.append(result["physical_camera_intensity_raw"].detach())
                    chips.append(torch.complex(result["chip_real"], result["chip_imag"]).detach())
                    encoder_layers.append(
                        torch.complex(
                            result["diffractive_encoder_layer_amplitude"]
                            * torch.cos(result["diffractive_encoder_layer_phase"]),
                            result["diffractive_encoder_layer_amplitude"]
                            * torch.sin(result["diffractive_encoder_layer_phase"]),
                        ).detach()
                    )
                    decoder_layers.append(
                        torch.complex(
                            result["diffractive_decoder_layer_amplitude"]
                            * torch.cos(result["diffractive_decoder_layer_phase"]),
                            result["diffractive_decoder_layer_amplitude"]
                            * torch.sin(result["diffractive_decoder_layer_phase"],),
                        ).detach()
                    )
                    discarded_ratios.append(
                        max(
                            float(result.get("diffractive_encoder_crop_discarded_energy_ratio", torch.zeros(1)).max()),
                            float(result.get("diffractive_decoder_crop_discarded_energy_ratio", torch.zeros(1)).max()),
                        )
                    )
            raw[f"state{state_grid}_fft{fft_grid}"] = {
                "camera": torch.cat(cameras),
                "chip": torch.cat(chips),
                "encoder_layers": torch.cat(encoder_layers),
                "decoder_layers": torch.cat(decoder_layers),
                "discarded_energy_ratio": max(discarded_ratios, default=0.0),
            }
            del model
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
        reference = raw["state2048_fft4096"]
        results: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            item = compare_planes(value["chip"], reference["chip"])
            item["discarded_energy_ratio"] = 0.0
            item["camera_intensity_corr"] = correlation(value["camera"], reference["camera"])
            item["discarded_energy_ratio"] = float(value["discarded_energy_ratio"])
            plane_metrics: dict[str, Any] = {}
            for prefix in ("encoder", "decoder"):
                current = value[f"{prefix}_layers"]
                target = reference[f"{prefix}_layers"]
                for layer_index in range(current.shape[1]):
                    name = f"{prefix}_layer_{layer_index + 1}"
                    plane_metrics[name] = compare_planes(
                        current[:, layer_index], target[:, layer_index]
                    )
            item["planes"] = plane_metrics
            item["all_plane_gates_pass"] = all(
                float(metrics["complex_field_relative_l2"]["mean"]) <= 1e-3
                and float(metrics["amplitude_relative_l2"]["mean"]) <= 1e-3
                and float(metrics["weighted_phase_mae_rad"]["mean"]) <= 1e-3
                and float(metrics["intensity_corr"]["mean"]) >= 0.999
                and float(metrics["total_energy_relative_difference"]["mean"]) <= 1e-3
                and float(metrics["guard_band_ratio"]["mean"]) <= 1e-4
                for metrics in plane_metrics.values()
            )
            results[key] = item
            item["reference_is_self"] = key == "state2048_fft4096"
            item["sampling_fov_match"] = bool(
                report["physical_contract"]["sampling_fov_match"]
            )
            item["direct_rs_pass"] = bool(
                report["reference_checks"]["direct_rs"].get("passed", False)
            )
        report["results"] = results
        report["first_complex_failure_plane"] = next(
            (
                name
                for name, metrics in results["state1536_fft3072"]["planes"].items()
                if float(metrics["complex_field_relative_l2"]["mean"]) > 1e-3
            ),
            None,
        )
        report["first_guard_failure_plane"] = next(
            (
                name
                for name, metrics in results["state1536_fft3072"]["planes"].items()
                if float(metrics["guard_band_ratio"]["mean"]) > 1e-4
            ),
            None,
        )
        decision = choose_candidate(results)
        report.update(decision)
        report["camera_observable_converged"] = all(
            float(item["camera_intensity_corr"]) >= 0.999 for item in results.values()
        )
        report["training_field_converged"] = decision["status"] == "converged"
        report["status"] = decision["status"]
        if report["status"] != "converged":
            report["decision"] = "stop-before-A/B-training"
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
