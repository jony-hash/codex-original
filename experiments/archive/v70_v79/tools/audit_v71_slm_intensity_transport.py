"""Read-only v71 physical SLM-intensity to D2NN transport isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.audit_v69_code_distill import _StreamingFactor
from tools.audit_v70_slm_transport import _load_generator, build_lowfreq_code_pattern
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import balanced_factor_decomposition, metric_bundle

REPORT_SCHEMA = "v71_slm_intensity_transport_v1"
SCALE = 0.4913053810596466
TARGET = 0.25
PLANE_ORDER = (
    "slm_physical_intensity",
    "pre_encoder",
    "encoder_layer_1",
    "encoder_layer_2",
    "encoded",
    "ols_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "chip",
    "camera",
)


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def calibrate_physical_intensity_amplitude(
    base_command: torch.Tensor,
    pattern: torch.Tensor,
    *,
    background_floor: float = 0.05,
    iterations: int = 30,
    target: float = TARGET,
) -> dict[str, Any]:
    """Calibrate only ``slm_amplitude.square()``; no optical forward occurs."""
    if base_command.shape != pattern.shape or base_command.ndim != 5:
        raise ValueError("expected balanced [100,10,1,H,W] command and pattern")

    def evaluate(amplitude: float) -> tuple[float, torch.Tensor, torch.Tensor]:
        command = (base_command + amplitude * pattern).clamp(0.0, 1.0)
        physical_amplitude = background_floor + (1.0 - background_floor) * command
        physical_intensity = physical_amplitude.square()
        factor = balanced_factor_decomposition(physical_intensity)[0]["code_fraction"]
        return factor, command, physical_intensity

    low, high = 0.0, 1.0
    high_factor, _, _ = evaluate(high)
    while high_factor < target and high < 16.0:
        high *= 2.0
        high_factor, _, _ = evaluate(high)
    if high_factor < target:
        raise ValueError("physical SLM intensity cannot reach target code factor")
    for _ in range(iterations):
        middle = (low + high) / 2.0
        factor, _, _ = evaluate(middle)
        if factor < target:
            low = middle
        else:
            high = middle
    factor, command, intensity = evaluate(high)
    return {
        "target_code_fraction": target,
        "amplitude": high,
        "background_floor": background_floor,
        "iterations": iterations,
        "achieved_code_fraction": factor,
        "absolute_error": abs(factor - target),
        "finite": bool(torch.isfinite(command).all() and torch.isfinite(intensity).all()),
        "command_zero_fraction": float((command == 0).float().mean()),
        "command_one_fraction": float((command == 1).float().mean()),
    }


def _factor(value: torch.Tensor) -> dict[str, Any]:
    return balanced_factor_decomposition(value)[0]


def _complex_from_amplitude_phase(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    field = torch.polar(amplitude, phase)
    return torch.stack((field.real, field.imag), dim=1)


def _energy(outputs: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean())
        for key, value in outputs.items()
        if torch.is_tensor(value)
        and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
    }


def _layer_values(outputs: dict[str, Any], prefix: str, count: int) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    amplitudes = outputs[f"diffractive_{prefix}_layer_amplitude"]
    phases = outputs[f"diffractive_{prefix}_layer_phase"]
    return [
        (
            f"{prefix}_layer_{index + 1}",
            amplitudes[:, index].square(),
            _complex_from_amplitude_phase(amplitudes[:, index], phases[:, index]),
        )
        for index in range(count)
    ]


def run(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    v69_decision: str | Path,
    v70_decision: str | Path,
    v70_report: str | Path,
    output: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    prior = json.loads(Path(v69_decision).read_text(encoding="utf-8"))
    v70 = json.loads(Path(v70_decision).read_text(encoding="utf-8"))
    v70_report_data = json.loads(Path(v70_report).read_text(encoding="utf-8"))
    if prior.get("status") != "v69-prototype-only":
        raise ValueError("v71 requires frozen v69-prototype-only decision")
    if v70.get("status") != "post-slm-code-factor-attenuated":
        raise ValueError("v71 requires frozen v70 factor-attenuated decision")
    if v70_report_data.get("status") != "v70-slm-intervention-complete":
        raise ValueError("v71 requires complete v70 report")
    cfg = load_training_config(config)
    if cfg.one_shot.generator_e_arch != "label_template_code_delta":
        raise ValueError("v71 requires the frozen v69 code-delta config")
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, labels = data["codes"].float(), data["class_labels"].long()
    if codes.shape != (100, 128) or labels.shape != (100,) or set(labels.tolist()) != set(range(10)):
        raise ValueError("v71 requires the fixed 100-sample validation split")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prototypes = _build_prototype_bank(cfg).float()
    generator = _load_generator(cfg, checkpoint)
    with torch.no_grad():
        all_codes = codes[:, None, :].expand(100, 10, 128).reshape(1000, 128)
        requested = torch.arange(10).repeat(100)
        _, diagnostics = generator.forward_with_diagnostics(all_codes, requested)
        base = diagnostics["generator_base_command"].reshape(100, 10, 1, 200, 200)
    pattern = build_lowfreq_code_pattern(codes).reshape(100, 1, 1, 200, 200).expand_as(base)
    calibration = calibrate_physical_intensity_amplitude(
        base,
        pattern,
        background_floor=float(cfg.one_shot.slm_background_floor),
    )
    overridden_one_shot = replace(
        cfg.one_shot,
        diagnostic_slm_override_mode="lowfreq_additive",
        diagnostic_slm_override_amplitude=calibration["amplitude"],
    )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v71 isolation requires CUDA")
    model = OneShotOpticalGenerator(cfg.physics, overridden_one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_one_shot_checkpoint(model, payload)
    state_before = _state_hash(model)
    streams = {name: _StreamingFactor((1, 200, 200)) for name in ("slm_command", "slm_amplitude", "slm_physical_intensity")}
    streams.update({name: _StreamingFactor((1, 100, 100)) for name in PLANE_ORDER[1:] if name not in {"camera", "chip"}})
    streams.update({"chip": _StreamingFactor((1, 100, 100)), "camera": _StreamingFactor((1, 200, 200))})
    complex_streams: dict[str, _StreamingFactor] = {}
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
                outputs = model(code[None].to(dev), label, return_optical_fields=False)
                finite = finite and all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_tensor(value))
                amplitude = outputs["slm_amplitude"]
                values: list[tuple[str, torch.Tensor, torch.Tensor | None]] = [
                    ("slm_command", outputs["slm_command"], None),
                    ("slm_amplitude", amplitude, None),
                    ("slm_physical_intensity", amplitude.square(), None),
                    ("pre_encoder", outputs["encoder_input_grid_amplitude"].square(), _complex_from_amplitude_phase(outputs["encoder_input_grid_amplitude"], outputs["encoder_input_grid_phase"])),
                    ("encoded", outputs["encoded_amplitude"].square(), _complex_from_amplitude_phase(outputs["encoded_amplitude"], outputs["encoded_phase"])),
                    ("ols_output", outputs["ols_output_amplitude"].square(), _complex_from_amplitude_phase(outputs["ols_output_amplitude"], outputs["ols_output_phase"])),
                    ("chip", outputs["chip_real"].square() + outputs["chip_imag"].square(), torch.stack((outputs["chip_real"], outputs["chip_imag"]), dim=1)),
                    ("camera", outputs["physical_camera_intensity_raw"], None),
                ]
                values.extend(_layer_values(outputs, "encoder", 2))
                values.extend(_layer_values(outputs, "decoder", 2))
                for name, value, complex_value in values:
                    streams[name].add(code_index, requested_label, value)
                    if complex_value is not None:
                        complex_streams.setdefault(name, _StreamingFactor(tuple(complex_value.shape[1:]))).add(code_index, requested_label, complex_value)
                camera_raw[code_index, requested_label] = outputs["physical_camera_intensity_raw"].detach().cpu()
                if code_index in chosen_lookup:
                    preview_raw[chosen_lookup[code_index], requested_label] = camera_raw[code_index, requested_label]
                energies.append(_energy(outputs))
    factors = {name: stream.report() for name, stream in streams.items()}
    complex_factors = {name: stream.report() for name, stream in complex_streams.items()}
    intensity = {name: factors[name]["code_fraction"] for name in PLANE_ORDER}
    transitions = {
        name: {
            "from": PLANE_ORDER[index - 1],
            "absolute_drop": intensity[PLANE_ORDER[index]] - intensity[PLANE_ORDER[index - 1]],
            "retention_ratio": intensity[PLANE_ORDER[index]] / max(intensity[PLANE_ORDER[index - 1]], 1e-12),
        }
        for index, name in enumerate(PLANE_ORDER[1:], start=1)
    }
    # Treat the calibrated source plane as passing within the registered
    # 1e-4 calibration tolerance; otherwise floating-point bisection noise
    # would falsely label the source itself as the attenuation plane.
    below = [name for name in PLANE_ORDER if intensity[name] < TARGET - 1e-4]
    camera = fixed_exposure_view(camera_raw, SCALE)
    targets = prototypes[None].expand(100, -1, *prototypes.shape[1:])
    crossed_metrics = metric_bundle(camera.flatten(0, 1), targets.flatten(0, 1), torch.arange(10).repeat(100), prototypes)
    diagonal = camera[torch.arange(100), labels]
    diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    preview = fixed_exposure_view(preview_raw, SCALE)
    preview_path = output_path.with_name(output_path.stem + "_raw_preview.png")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(10, 10, figsize=(12, 12), squeeze=False)
        for row in range(10):
            for col in range(10):
                axes[row][col].imshow(preview[row, col, 0], cmap="gray", vmin=0, vmax=1)
                axes[row][col].axis("off")
        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        preview_path = Path(str(preview_path) + ".unavailable")
        preview_path.write_text(str(exc), encoding="utf-8")
    energy_mean = {key: sum(item.get(key, 0.0) for item in energies) / len(energies) for key in sorted({key for item in energies for key in item})}
    energy_values = [value for key, value in energy_mean.items() if "guard" in key or "discarded" in key]
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "v71-slm-intensity-isolation-complete",
        "v69_decision": str(v69_decision),
        "v70_decision": str(v70_decision),
        "v70_report": str(v70_report),
        "test_split_used": False,
        "validation_sample_count": 100,
        "crossed_output_count": 1000,
        "batch_size": 1,
        "return_optical_fields": False,
        "override_mode": "lowfreq_additive",
        "exposure_mode": "locked",
        "exposure_scale": SCALE,
        "calibration_semantics": "slm_amplitude_squared",
        "calibration": calibration,
        "factor_decomposition": factors,
        "complex_factor_decomposition": complex_factors,
        "intensity_code_factors": intensity,
        "transitions": transitions,
        "camera_code_factor": intensity["camera"],
        "first_intensity_plane_below_25_percent": below[0] if below else None,
        "crossed_metrics": crossed_metrics,
        "diagonal_metrics": diagonal_metrics,
        "energy_records_mean": energy_mean,
        "energy_pass": bool(energy_values) and max(energy_values) <= 1e-4,
        "finite": finite and all(item["finite"] for item in factors.values()) and all(item["finite"] for item in complex_factors.values()),
        "parameters_unchanged": state_before == _state_hash(model),
        "optimizer_created": False,
        "parameter_update_count": 0,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "preview": str(preview_path),
        "preview_shape": list(preview.shape),
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "validation-dataset", "v69-decision", "v70-decision", "v70-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
