"""Read-only v73 semantic-support direct-command upper bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.audit_v70_slm_transport import _load_generator, build_lowfreq_code_pattern
from tools.audit_v72_d2nn_phase_attribution import gauge_align_fields
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import balanced_factor_decomposition, metric_bundle
from tools.audit_v69_code_distill import _StreamingFactor

REPORT_SCHEMA = "v73_semantic_support_audit_v1"
CPU_SCHEMA = "v73_semantic_support_cpu_admission_v1"
SCALE = 0.4913053810596466
TARGET = 0.25
BACKGROUND_FLOOR = 0.05
EXPECTED_AMPLITUDE = 0.4027690887451172
FOREGROUND_THRESHOLD = 0.2
FOREGROUND_ENERGY_MIN = 0.95
LOW_FREQUENCY_ENERGY_MIN = 0.99
ONE_CLIPPING_MAX = 0.08
BASELINE_CODE_SS = 53141.618758854296
BASELINE_LABEL_SS = 182040.41204567475
PLANE_ORDER = (
    "slm_physical_intensity",
    "pre_encoder",
    "encoder_layer_1",
    "encoder_layer_2",
    "ols_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "chip",
    "camera",
)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_data(codes: torch.Tensor, labels: torch.Tensor) -> None:
    if tuple(codes.shape) != (100, 128) or tuple(labels.shape) != (100,):
        raise ValueError("v73 requires the fixed 100-sample validation split")
    if set(labels.tolist()) != set(range(10)):
        raise ValueError("v73 validation split must contain all ten labels")


def semantic_support_pattern(
    codes: torch.Tensor, prototypes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build centered, prototype-supported code patterns for every code×label."""
    if tuple(codes.shape) != (100, 128) or tuple(prototypes.shape[-2:]) != (200, 200):
        raise ValueError("expected codes [100,128] and prototypes [10,1,200,200]")
    lowfreq = build_lowfreq_code_pattern(codes).reshape(100, 1, 1, 200, 200)
    support = prototypes.float().reshape(1, 10, 1, 200, 200).expand(100, -1, -1, -1, -1)
    pattern = lowfreq.expand(100, 10, 1, 200, 200)
    weighted_mean = (support * pattern).flatten(2).sum(2, keepdim=True)
    weighted_mean = weighted_mean / support.flatten(2).sum(2, keepdim=True).clamp_min(1.0e-8)
    centered = support * (pattern - weighted_mean.view(100, 10, 1, 1, 1))
    maximum = centered.flatten(2).abs().amax(2).view(100, 10, 1, 1, 1).clamp_min(1.0e-8)
    return centered / maximum, support


def _physical_factor(command: torch.Tensor, background_floor: float) -> tuple[dict[str, Any], torch.Tensor]:
    amplitude = background_floor + (1.0 - background_floor) * command
    intensity = amplitude.square()
    return balanced_factor_decomposition(intensity)[0], intensity


def calibrate_semantic_support_amplitude(
    base_command: torch.Tensor,
    pattern: torch.Tensor,
    *,
    background_floor: float = BACKGROUND_FLOOR,
    iterations: int = 30,
    target: float = TARGET,
) -> dict[str, Any]:
    if base_command.shape != pattern.shape or base_command.ndim != 5:
        raise ValueError("expected balanced [100,10,1,H,W] command and pattern")

    def evaluate(amplitude: float) -> tuple[dict[str, Any], torch.Tensor]:
        command = (base_command + amplitude * pattern).clamp(0.0, 1.0)
        return _physical_factor(command, background_floor)

    low, high = 0.0, 1.0
    factor, _ = evaluate(high)
    while float(factor["code_fraction"]) < target and high < 16.0:
        high *= 2.0
        factor, _ = evaluate(high)
    if float(factor["code_fraction"]) < target:
        raise ValueError("semantic-support intervention cannot reach target")
    for _ in range(iterations):
        middle = (low + high) / 2.0
        factor, _ = evaluate(middle)
        if float(factor["code_fraction"]) < target:
            low = middle
        else:
            high = middle
    factor, intensity = evaluate(high)
    command = (base_command + high * pattern).clamp(0.0, 1.0)
    return {
        "target_code_fraction": target,
        "amplitude": high,
        "background_floor": background_floor,
        "iterations": iterations,
        "achieved_code_fraction": float(factor["code_fraction"]),
        "absolute_error": abs(float(factor["code_fraction"]) - target),
        "finite": bool(torch.isfinite(command).all() and torch.isfinite(intensity).all()),
        "command_zero_fraction": float((command == 0).float().mean()),
        "command_one_fraction": float((command == 1).float().mean()),
        "factor": factor,
    }


def _frequency_energy(values: torch.Tensor, *, pixel_pitch: float = 2.5e-6) -> dict[str, float]:
    centered = values.float() - values.float().mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered), dim=(-2, -1)).abs().square()
    fy = torch.fft.fftshift(torch.fft.fftfreq(values.shape[-2], d=pixel_pitch)) / 1000.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(values.shape[-1], d=pixel_pitch)) / 1000.0
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low = spectrum[..., radius <= 20.0].sum()
    mid = spectrum[..., (radius > 20.0) & (radius <= 80.0)].sum()
    high = spectrum[..., radius > 80.0].sum()
    total = float((low + mid + high).item())
    if total <= 0.0:
        return {"semantic_0_20_cycles_mm": 0.0, "mid_20_80_cycles_mm": 0.0, "high_above_80_cycles_mm": 0.0}
    return {
        "semantic_0_20_cycles_mm": float(low.item()) / total,
        "mid_20_80_cycles_mm": float(mid.item()) / total,
        "high_above_80_cycles_mm": float(high.item()) / total,
    }


def _effect_summary(factors: dict[str, Any]) -> dict[str, Any]:
    total = float(factors["total_energy"])
    return {
        **factors,
        "code_effect_ss": total * float(factors["code_fraction"]),
        "label_effect_ss": total * float(factors["label_fraction"]),
        "interaction_effect_ss": total * float(factors["interaction_fraction"]),
    }


def _layer_values(outputs: dict[str, Any], prefix: str, count: int) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    amplitudes = outputs[f"diffractive_{prefix}_layer_amplitude"]
    phases = outputs[f"diffractive_{prefix}_layer_phase"]
    values = []
    for index in range(count):
        amplitude = amplitudes[:, index]
        field = torch.polar(amplitude, phases[:, index])
        values.append((
            f"{prefix}_layer_{index + 1}",
            amplitude.square(),
            torch.stack((field.real, field.imag), dim=1),
        ))
    return values


def _energy(outputs: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean())
        for key, value in outputs.items()
        if torch.is_tensor(value) and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
    }


def _make_preview(images: torch.Tensor, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(10, 10, figsize=(12, 12), squeeze=False)
    for row in range(10):
        for column in range(10):
            axes[row][column].imshow(images[row, column, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row][column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _forward_with_support(
    model: OneShotOpticalGenerator,
    code: torch.Tensor,
    labels: torch.Tensor,
    support: torch.Tensor,
    amplitude: float,
) -> dict[str, torch.Tensor]:
    original_override = model._diagnostic_slm_override

    def semantic_override(
        slm_command: torch.Tensor,
        code_value: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del code_value
        pattern = build_lowfreq_code_pattern(code.detach().cpu()).to(slm_command.device)
        mask = support.to(slm_command.device).index_select(0, labels)
        weighted_mean = (mask * pattern).flatten(1).sum(1, keepdim=True)
        weighted_mean = weighted_mean / mask.flatten(1).sum(1, keepdim=True).clamp_min(1.0e-8)
        perturbation = mask * (pattern - weighted_mean.view(-1, 1, 1, 1))
        perturbation = perturbation / perturbation.flatten(1).abs().amax(1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1.0e-8)
        command = (diagnostics["generator_base_command"] + amplitude * perturbation).clamp(0.0, 1.0)
        return command, {
            "diagnostic_slm_override_active": torch.ones((), device=command.device, dtype=torch.bool),
            "diagnostic_slm_override_amplitude": command.new_tensor(amplitude),
            "diagnostic_slm_override_pattern": perturbation,
            "diagnostic_slm_override_base_command": diagnostics["generator_base_command"],
            "semantic_support_mask": mask,
        }

    model._diagnostic_slm_override = semantic_override
    try:
        return model(code, labels, return_optical_fields=False)
    finally:
        model._diagnostic_slm_override = original_override


def _base_payload(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    v72_decision: str | Path,
    v72_report: str | Path,
) -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], str]:
    cfg = load_training_config(config)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, labels = data["codes"].float(), data["class_labels"].long()
    _validate_data(codes, labels)
    prototypes = _build_prototype_bank(cfg).float().contiguous()
    generator = _load_generator(cfg, checkpoint)
    with torch.no_grad():
        all_codes = codes[:, None, :].expand(100, 10, 128).reshape(1000, 128)
        requested = torch.arange(10).repeat(100)
        _, diagnostics = generator.forward_with_diagnostics(all_codes, requested)
        base = diagnostics["generator_base_command"].reshape(100, 10, 1, 200, 200)
    pattern, support = semantic_support_pattern(codes, prototypes)
    v72 = json.loads(Path(v72_decision).read_text(encoding="utf-8"))
    v72_report_data = json.loads(Path(v72_report).read_text(encoding="utf-8"))
    if v72.get("status") != "phase-readability-code-tradeoff" or v72_report_data.get("status") != "v72-phase-attribution-complete":
        raise ValueError("v73 requires the frozen valid v72 provenance")
    return cfg, v72, codes, labels, prototypes, base, {"pattern": pattern, "support": support, "v72_report": v72_report_data}, state_hash(generator)


def run_cpu_admission(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    v72_decision: str | Path,
    v72_report: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    cfg, v72, codes, labels, prototypes, base, aux, generator_hash = _base_payload(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v72_decision=v72_decision, v72_report=v72_report,
    )
    pattern, support = aux["pattern"], aux["support"]
    calibration = calibrate_semantic_support_amplitude(
        base, pattern, background_floor=float(cfg.one_shot.slm_background_floor)
    )
    command = (base + float(calibration["amplitude"]) * pattern).clamp(0.0, 1.0)
    delta = command - base
    foreground = support >= FOREGROUND_THRESHOLD
    delta_energy = delta.square()
    foreground_energy_fraction = float(delta_energy[foreground].sum() / delta_energy.sum().clamp_min(1.0e-8))
    frequency_energy = _frequency_energy(delta.flatten(0, 1))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v72_baseline_hash = aux["v72_report"]["arms"]["baseline"]["arm_state_hash"]
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size)
    load_one_shot_checkpoint(model, checkpoint_payload)
    model_hash = state_hash(model)
    checks = {
        "calibration": bool(calibration["finite"] and calibration["absolute_error"] <= 1.0e-4),
        "foreground_energy": foreground_energy_fraction >= FOREGROUND_ENERGY_MIN,
        "low_frequency_energy": frequency_energy["semantic_0_20_cycles_mm"] >= LOW_FREQUENCY_ENERGY_MIN,
        "one_clipping": calibration["command_one_fraction"] <= ONE_CLIPPING_MAX,
        "finite": bool(torch.isfinite(delta).all()),
        "v72_hash_match": model_hash == v72_baseline_hash,
        "validation_contract": tuple(codes.shape) == (100, 128) and set(labels.tolist()) == set(range(10)),
        "prototype_hash_present": bool(tensor_hash(prototypes)),
    }
    report = {
        "schema": CPU_SCHEMA,
        "diagnostic_only": True,
        "status": "v73-cpu-admission-passed" if all(checks.values()) else "harness-invalid",
        "checkpoint": str(checkpoint),
        "validation_dataset": str(validation_dataset),
        "v72_decision": str(v72_decision),
        "v72_report": str(v72_report),
        "validation_sample_count": 100,
        "crossed_output_count": 1000,
        "test_split_used": False,
        "return_optical_fields": False,
        "intervention": "prototype_soft_weighted_centered_lowfreq_additive",
        "calibration": calibration,
        "expected_amplitude": EXPECTED_AMPLITUDE,
        "amplitude_delta_vs_expected": float(calibration["amplitude"]) - EXPECTED_AMPLITUDE,
        "prototype_hash": tensor_hash(prototypes),
        "codes_hash": tensor_hash(codes),
        "checkpoint_model_hash": model_hash,
        "v72_baseline_hash": v72_baseline_hash,
        "foreground_threshold": FOREGROUND_THRESHOLD,
        "foreground_energy_fraction": foreground_energy_fraction,
        "frequency_energy": frequency_energy,
        "delta_energy_finite": bool(torch.isfinite(delta_energy).all()),
        "checks": checks,
        "diagnostic_only_guarantees": {
            "optimizer_created": False,
            "parameter_update_count": 0,
            "full_field_saved": False,
        },
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _transitions(factors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for index, name in enumerate(PLANE_ORDER[1:], start=1):
        previous = factors[PLANE_ORDER[index - 1]]
        current = factors[name]
        result[name] = {
            "from": PLANE_ORDER[index - 1],
            "signed_delta": float(current["code_fraction"] - previous["code_fraction"]),
            "retention_ratio": float(current["code_fraction"] / max(previous["code_fraction"], 1.0e-12)),
            "code_effect_ss_signed_delta": float(current["code_effect_ss"] - previous["code_effect_ss"]),
            "code_effect_ss_retention": float(current["code_effect_ss"] / max(previous["code_effect_ss"], 1.0e-12)),
            "label_effect_ss_signed_delta": float(current["label_effect_ss"] - previous["label_effect_ss"]),
            "label_effect_ss_retention": float(current["label_effect_ss"] / max(previous["label_effect_ss"], 1.0e-12)),
            "interaction_effect_ss_signed_delta": float(current["interaction_effect_ss"] - previous["interaction_effect_ss"]),
            "interaction_effect_ss_retention": float(current["interaction_effect_ss"] / max(previous["interaction_effect_ss"], 1.0e-12)),
        }
    return result


def run_cuda(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    v72_decision: str | Path,
    v72_report: str | Path,
    cpu_admission: str | Path,
    output: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    admission = json.loads(Path(cpu_admission).read_text(encoding="utf-8"))
    if admission.get("status") != "v73-cpu-admission-passed":
        raise ValueError("v73 CUDA audit requires a passing CPU admission")
    cfg, v72, codes, labels, prototypes, base, aux, _ = _base_payload(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v72_decision=v72_decision, v72_report=v72_report,
    )
    support = aux["support"][0]
    amplitude = float(admission["calibration"]["amplitude"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v73 CUDA audit requires CUDA")
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    load_one_shot_checkpoint(model, payload)
    before_hash = state_hash(model)
    streams: dict[str, _StreamingFactor] = {}
    complex_streams: dict[str, _StreamingFactor] = {}
    compact_fields: dict[str, torch.Tensor] = {}
    camera_raw = torch.empty((100, 10, 1, 200, 200), dtype=torch.float32)
    preview_raw = torch.empty((10, 10, 1, 200, 200), dtype=torch.float32)
    chosen = [int(torch.nonzero(labels == label, as_tuple=False)[0]) for label in range(10)]
    chosen_lookup = {index: row for row, index in enumerate(chosen)}
    energies: list[dict[str, float]] = []
    finite = True
    with torch.no_grad():
        for code_index, code in enumerate(codes):
            for requested_label in range(10):
                label = torch.tensor([requested_label], device=dev, dtype=torch.long)
                outputs = _forward_with_support(model, code[None].to(dev), label, support, amplitude)
                finite = finite and all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_tensor(value))
                values: list[tuple[str, torch.Tensor, torch.Tensor | None]] = [
                    ("slm_physical_intensity", outputs["slm_amplitude"].square(), None),
                    ("pre_encoder", outputs["encoder_input_grid_amplitude"].square(), torch.stack((outputs["encoder_input_grid_amplitude"] * torch.cos(outputs["encoder_input_grid_phase"]), outputs["encoder_input_grid_amplitude"] * torch.sin(outputs["encoder_input_grid_phase"])), dim=1)),
                    ("encoded", outputs["encoded_amplitude"].square(), torch.stack((outputs["encoded_amplitude"] * torch.cos(outputs["encoded_phase"]), outputs["encoded_amplitude"] * torch.sin(outputs["encoded_phase"])), dim=1)),
                    ("ols_output", outputs["ols_output_amplitude"].square(), torch.stack((outputs["ols_output_amplitude"] * torch.cos(outputs["ols_output_phase"]), outputs["ols_output_amplitude"] * torch.sin(outputs["ols_output_phase"])), dim=1)),
                    ("chip", outputs["chip_real"].square() + outputs["chip_imag"].square(), torch.stack((outputs["chip_real"], outputs["chip_imag"]), dim=1)),
                    ("camera", outputs["physical_camera_intensity_raw"], None),
                ]
                values.extend(_layer_values(outputs, "encoder", 2))
                values.extend(_layer_values(outputs, "decoder", 2))
                for name, value, complex_value in values:
                    streams.setdefault(name, _StreamingFactor(tuple(value.shape[1:]))).add(code_index, requested_label, value)
                    if complex_value is not None:
                        complex_cpu = complex_value.detach().cpu().float().squeeze(0)
                        if name not in compact_fields:
                            compact_fields[name] = torch.empty((100, 10, *complex_cpu.shape), dtype=torch.float32)
                        compact_fields[name][code_index, requested_label] = complex_cpu
                        complex_streams.setdefault(name, _StreamingFactor(tuple(complex_value.shape[1:]))).add(code_index, requested_label, complex_value)
                raw = outputs["physical_camera_intensity_raw"].detach().cpu()
                camera_raw[code_index, requested_label] = raw
                if code_index in chosen_lookup:
                    preview_raw[chosen_lookup[code_index], requested_label] = raw
                energies.append(_energy(outputs))
    factors = {name: _effect_summary(stream.report()) for name, stream in streams.items()}
    raw_complex = {name: _effect_summary(stream.report()) for name, stream in complex_streams.items()}
    gauge_complex = {
        name: _effect_summary(balanced_factor_decomposition(gauge_align_fields(fields))[0])
        for name, fields in compact_fields.items()
    }
    intensity = {name: factors[name]["code_fraction"] for name in PLANE_ORDER}
    transitions = _transitions(factors)
    first_below = next((name for name in PLANE_ORDER if intensity[name] < TARGET - 1.0e-4), None)
    camera = fixed_exposure_view(camera_raw, SCALE)
    targets = prototypes[None].expand(100, -1, *prototypes.shape[1:])
    crossed_metrics = metric_bundle(camera.flatten(0, 1), targets.flatten(0, 1), torch.arange(10).repeat(100), prototypes)
    diagonal = camera[torch.arange(100), labels]
    diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.parent / "v73_semantic_support_raw_preview.png"
    _make_preview(fixed_exposure_view(preview_raw, SCALE), preview_path)
    energy_mean = {key: sum(item.get(key, 0.0) for item in energies) / len(energies) for key in sorted({key for item in energies for key in item})}
    energy_values = [value for key, value in energy_mean.items() if "guard" in key or "discarded" in key]
    camera_code_ss = factors["camera"]["code_effect_ss"]
    camera_label_ss = factors["camera"]["label_effect_ss"]
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "v73-semantic-support-audit-complete",
        "checkpoint": str(checkpoint),
        "validation_dataset": str(validation_dataset),
        "v72_decision": str(v72_decision),
        "v72_report": str(v72_report),
        "cpu_admission": str(cpu_admission),
        "test_split_used": False,
        "validation_sample_count": 100,
        "crossed_output_count": 1000,
        "batch_size": 1,
        "return_optical_fields": False,
        "full_field_saved": False,
        "active_view_only": True,
        "exposure_mode": "locked",
        "exposure_scale": SCALE,
        "intervention": "prototype_soft_weighted_centered_lowfreq_additive",
        "calibration": admission["calibration"],
        "factor_decomposition": factors,
        "intensity_code_factors": intensity,
        "transitions": transitions,
        "complex_factor_decomposition": raw_complex,
        "gauge_aligned_complex_factor_decomposition": gauge_complex,
        "first_intensity_plane_below_25_percent": first_below,
        "camera_code_factor": intensity["camera"],
        "camera_code_effect_ss": camera_code_ss,
        "camera_label_effect_ss": camera_label_ss,
        "v72_baseline_camera_code_effect_ss": BASELINE_CODE_SS,
        "v72_baseline_camera_label_effect_ss": BASELINE_LABEL_SS,
        "absolute_code_ss_pass": camera_code_ss >= BASELINE_CODE_SS,
        "label_ss_retention": camera_label_ss / BASELINE_LABEL_SS,
        "label_ss_retention_pass": camera_label_ss >= 0.9 * BASELINE_LABEL_SS,
        "crossed_metrics": crossed_metrics,
        "diagonal_metrics": diagonal_metrics,
        "energy_records_mean": energy_mean,
        "energy_pass": bool(energy_values) and max(energy_values) <= 1.0e-4,
        "finite": finite and all(item["finite"] for item in factors.values()) and all(item["finite"] for item in raw_complex.values()) and all(item["finite"] for item in gauge_complex.values()),
        "parameters_unchanged": before_hash == state_hash(model),
        "optimizer_created": False,
        "parameter_update_count": 0,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "preview": str(preview_path),
        "preview_shape": [10, 10, 1, 200, 200],
        "teacher_alignment_computed": False,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--v72-decision", required=True, type=Path)
    parser.add_argument("--v72-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-admission", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        result = run_cpu_admission(
            config=args.config, checkpoint=args.checkpoint, validation_dataset=args.validation_dataset,
            v72_decision=args.v72_decision, v72_report=args.v72_report, output=args.output,
        )
    else:
        if args.cpu_admission is None:
            raise ValueError("--cpu-admission is required for CUDA audit")
        result = run_cuda(
            config=args.config, checkpoint=args.checkpoint, validation_dataset=args.validation_dataset,
            v72_decision=args.v72_decision, v72_report=args.v72_report,
            cpu_admission=args.cpu_admission, output=args.output, device=args.device,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
