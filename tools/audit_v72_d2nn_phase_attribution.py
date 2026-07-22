"""Read-only v72 causal attribution of frozen D2NN phase masks."""

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
from tools.audit_v69_code_distill import _StreamingFactor
from tools.audit_v70_slm_transport import _load_generator, build_lowfreq_code_pattern
from tools.audit_v71_slm_intensity_transport import calibrate_physical_intensity_amplitude
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import balanced_factor_decomposition, metric_bundle

REPORT_SCHEMA = "v72_d2nn_phase_attribution_v1"
SCALE = 0.4913053810596466
TARGET = 0.25
EXPECTED_INTERVENTION_AMPLITUDE = 0.27918721828609705
PLANE_ORDER = (
    "pre_encoder",
    "encoder_layer_1",
    "encoder_layer_2",
    "ols_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "chip",
    "camera",
)
ARMS = {
    "baseline": (),
    "encoder2_identity": (("encoder.phase", 1),),
    "decoder1_identity": (("decoder.phase", 0),),
    "joint_identity": (("encoder.phase", 1), ("decoder.phase", 0)),
}


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _effect_summary(factors: dict[str, Any]) -> dict[str, Any]:
    total = float(factors["total_energy"])
    return {
        **factors,
        "code_effect_ss": total * float(factors["code_fraction"]),
        "label_effect_ss": total * float(factors["label_fraction"]),
        "interaction_effect_ss": total * float(factors["interaction_fraction"]),
    }


def gauge_align_fields(fields: torch.Tensor) -> torch.Tensor:
    """Align each code field to its requested-label leave-one-out mean phase.

    ``fields`` is real/imag with shape ``[100, 10, 2, ...]``. The returned
    tensor has the same shape and is still compact active-view data.
    """
    if fields.ndim < 5 or tuple(fields.shape[:3]) != (100, 10, 2):
        raise ValueError("expected fields with shape [100,10,2,...]")
    complex_fields = torch.complex(fields[:, :, 0], fields[:, :, 1])
    sums = complex_fields.sum(dim=0)
    aligned = torch.empty_like(complex_fields)
    for label in range(10):
        current = complex_fields[:, label]
        reference = (sums[label][None] - current) / 99.0
        inner = (reference.conj() * current).flatten(1).sum(1)
        phase = torch.angle(inner)
        aligned[:, label] = current * torch.exp(-1j * phase).view(-1, *([1] * (current.ndim - 1)))
    return torch.stack((aligned.real, aligned.imag), dim=2)


def _complex_from_amplitude_phase(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    field = torch.polar(amplitude, phase)
    return torch.stack((field.real, field.imag), dim=1)


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


def _energy(outputs: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean())
        for key, value in outputs.items()
        if torch.is_tensor(value)
        and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
    }


def _make_preview(images: torch.Tensor, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(10, 10, figsize=(12, 12), squeeze=False)
    for row in range(10):
        for col in range(10):
            axes[row][col].imshow(images[row, col, 0], cmap="gray", vmin=0, vmax=1)
            axes[row][col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _validate_non_target_hash(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], targeted: tuple[tuple[str, int], ...]) -> bool:
    targeted_by_key: dict[str, set[int]] = {}
    for key, index in targeted:
        targeted_by_key.setdefault(key, set()).add(index)
    for key, value in before.items():
        if key not in targeted_by_key:
            if not torch.equal(value, after[key]):
                return False
            continue
        for index in range(value.shape[0]):
            if index not in targeted_by_key[key] and not torch.equal(value[index], after[key][index]):
                return False
        if value.ndim == 0 and not torch.equal(value, after[key]):
            return False
    return True


def _run_arm(
    *,
    arm: str,
    zero_specs: tuple[tuple[str, int], ...],
    cfg: Any,
    payload: dict[str, Any],
    codes: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    overridden = replace(
        cfg.one_shot,
        diagnostic_slm_override_mode="lowfreq_additive",
        diagnostic_slm_override_amplitude=EXPECTED_INTERVENTION_AMPLITUDE,
    )
    model = OneShotOpticalGenerator(cfg.physics, overridden, image_size=cfg.first_stage.image_size).to(device).eval()
    load_one_shot_checkpoint(model, payload)
    before_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key, index in zero_specs:
        parameter = dict(model.named_parameters())[key]
        with torch.no_grad():
            parameter[index].zero_()
    after_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    non_target_unchanged = _validate_non_target_hash(before_state, after_state, zero_specs)
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
                label = torch.tensor([requested_label], device=device, dtype=torch.long)
                outputs = model(code[None].to(device), label, return_optical_fields=False)
                finite = finite and all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_tensor(value))
                values: list[tuple[str, torch.Tensor, torch.Tensor | None]] = [
                    ("pre_encoder", outputs["encoder_input_grid_amplitude"].square(), _complex_from_amplitude_phase(outputs["encoder_input_grid_amplitude"], outputs["encoder_input_grid_phase"])),
                    ("encoded", outputs["encoded_amplitude"].square(), _complex_from_amplitude_phase(outputs["encoded_amplitude"], outputs["encoded_phase"])),
                    ("ols_output", outputs["ols_output_amplitude"].square(), _complex_from_amplitude_phase(outputs["ols_output_amplitude"], outputs["ols_output_phase"])),
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
    camera = fixed_exposure_view(camera_raw, SCALE)
    targets = prototypes[None].expand(100, -1, *prototypes.shape[1:])
    crossed_metrics = metric_bundle(camera.flatten(0, 1), targets.flatten(0, 1), torch.arange(10).repeat(100), prototypes)
    diagonal = camera[torch.arange(100), labels]
    diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    preview_path = output_path.parent / f"v72_{arm}_raw_preview.png"
    _make_preview(fixed_exposure_view(preview_raw, SCALE), preview_path)
    energy_mean = {key: sum(item.get(key, 0.0) for item in energies) / len(energies) for key in sorted({key for item in energies for key in item})}
    energy_values = [value for key, value in energy_mean.items() if "guard" in key or "discarded" in key]
    intensity = {name: factors[name]["code_fraction"] for name in PLANE_ORDER}
    transitions = {
        name: {
            "from": PLANE_ORDER[index - 1],
            "absolute_drop": intensity[name] - intensity[PLANE_ORDER[index - 1]],
            "retention_ratio": intensity[name] / max(intensity[PLANE_ORDER[index - 1]], 1e-12),
        }
        for index, name in enumerate(PLANE_ORDER[1:], start=1)
    }
    return {
        "zeroed_phase_keys": [f"{key}[{index}]" for key, index in zero_specs],
        "arm_state_hash": _state_hash(model),
        "non_target_parameters_unchanged": non_target_unchanged,
        "targeted_parameters_zeroed": all(bool(torch.count_nonzero(after_state[key][index]).item() == 0) for key, index in zero_specs),
        "optimizer_created": False,
        "parameter_update_count": 0,
        "finite": finite and all(item["finite"] for item in factors.values()) and all(item["finite"] for item in raw_complex.values()) and all(item["finite"] for item in gauge_complex.values()),
        "factor_decomposition": factors,
        "intensity_code_factors": intensity,
        "transitions": transitions,
        "complex_factor_decomposition": raw_complex,
        "gauge_aligned_complex_factor_decomposition": gauge_complex,
        "camera_code_factor": intensity["camera"],
        "crossed_metrics": crossed_metrics,
        "diagonal_metrics": diagonal_metrics,
        "energy_records_mean": energy_mean,
        "energy_pass": bool(energy_values) and max(energy_values) <= 1e-4,
        "preview": str(preview_path),
        "preview_shape": [10, 10, 1, 200, 200],
    }


def run(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    v69_decision: str | Path,
    v70_decision: str | Path,
    v71_decision: str | Path,
    v71_report: str | Path,
    output: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    prior = json.loads(Path(v69_decision).read_text(encoding="utf-8"))
    v70 = json.loads(Path(v70_decision).read_text(encoding="utf-8"))
    v71 = json.loads(Path(v71_decision).read_text(encoding="utf-8"))
    v71_report_data = json.loads(Path(v71_report).read_text(encoding="utf-8"))
    if prior.get("status") != "v69-prototype-only" or v70.get("status") != "post-slm-code-factor-attenuated":
        raise ValueError("v72 requires frozen v69/v70 provenance")
    if v71.get("status") != "d2nn-code-factor-attenuated" or v71_report_data.get("status") != "v71-slm-intensity-isolation-complete":
        raise ValueError("v72 requires the valid v71 d2nn attribution result")
    cfg = load_training_config(config)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, labels = data["codes"].float(), data["class_labels"].long()
    if codes.shape != (100, 128) or labels.shape != (100,) or set(labels.tolist()) != set(range(10)):
        raise ValueError("v72 requires the fixed 100-sample validation split")
    if cfg.one_shot.generator_e_arch != "label_template_code_delta":
        raise ValueError("v72 requires label_template_code_delta")
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
    replay = calibrate_physical_intensity_amplitude(base, pattern, background_floor=float(cfg.one_shot.slm_background_floor))
    replay_ok = abs(float(replay["amplitude"]) - EXPECTED_INTERVENTION_AMPLITUDE) <= 1e-8 and float(replay["absolute_error"]) <= 1e-4
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v72 requires CUDA")
    arms = {
        name: _run_arm(arm=name, zero_specs=keys, cfg=cfg, payload=payload, codes=codes, labels=labels, prototypes=prototypes, device=dev, output_path=output_path)
        for name, keys in ARMS.items()
    }
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "v72-phase-attribution-complete",
        "v69_decision": str(v69_decision),
        "v70_decision": str(v70_decision),
        "v71_decision": str(v71_decision),
        "v71_report": str(v71_report),
        "test_split_used": False,
        "validation_sample_count": 100,
        "crossed_output_count_per_arm": 1000,
        "batch_size": 1,
        "return_optical_fields": False,
        "exposure_mode": "locked",
        "exposure_scale": SCALE,
        "calibration_semantics": "slm_amplitude_squared",
        "calibration": replay,
        "calibration_replay_pass": replay_ok,
        "arms": arms,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "phase_ablation_only": True,
        "teacher_alignment_computed": False,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
        "finite": replay_ok and all(item["finite"] for item in arms.values()),
        "parameters_unchanged": all(item["non_target_parameters_unchanged"] and item["targeted_parameters_zeroed"] for item in arms.values()),
        "optimizer_created": False,
        "parameter_update_count": 0,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "validation-dataset", "v69-decision", "v70-decision", "v71-decision", "v71-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
