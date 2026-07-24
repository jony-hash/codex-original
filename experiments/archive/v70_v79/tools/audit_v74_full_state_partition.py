"""Read-only v74 full-state active/exterior code-energy attribution audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v70_slm_transport import build_lowfreq_code_pattern
from tools.audit_v70_slm_transport import _load_generator
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.audit_v73_semantic_support import (
    SCALE,
    semantic_support_pattern,
    state_hash,
    tensor_hash,
)
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import metric_bundle

REPORT_SCHEMA = "v74_full_state_partition_audit_v1"
ADMISSION_SCHEMA = "v74_full_state_partition_admission_v1"
EXPECTED_AMPLITUDE = 0.4027685085311532
COMPLEX_RETENTION_MIN = 0.99
INTENSITY_RETENTION_MIN = 0.90
ESCAPE_EXPLANATION_MIN = 0.90
ACTIVE_SIZE = 100
STATE_SIZE = 2048
LABEL_COUNT = 10
CODE_COUNT = 100
PLANE_ORDER = (
    "encoder_full_grid_input",
    "encoder_layer_1",
    "encoder_layer_2",
    "ols_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "chip",
)
SHELL_SIZES = (100, 200, 400, 800, 1600, 2048)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_split(codes: torch.Tensor, labels: torch.Tensor) -> None:
    if tuple(codes.shape) != (CODE_COUNT, 128) or tuple(labels.shape) != (CODE_COUNT,):
        raise ValueError("v74 requires the fixed 100-sample validation split")
    if sorted(labels.tolist()) != sorted(labels.tolist()):
        raise ValueError("validation labels are malformed")
    if set(labels.tolist()) != set(range(10)):
        raise ValueError("validation split must contain all ten labels")


def _load_context(
    *, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path,
    v73_admission: str | Path, v73_report: str | Path, v73_decision: str | Path,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], str]:
    admission = _read(v73_admission)
    report = _read(v73_report)
    decision = _read(v73_decision)
    if admission.get("status") != "v73-cpu-admission-passed":
        raise ValueError("v74 requires the passed v73 CPU admission")
    if report.get("status") != "v73-semantic-support-audit-complete":
        raise ValueError("v74 requires the valid v73 raw report")
    if decision.get("status") != "semantic-support-code-factor-attenuated":
        raise ValueError("v74 requires the frozen v73 attenuated decision")
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes, labels = data["codes"].float(), data["class_labels"].long()
    _validate_split(codes, labels)
    cfg = load_training_config(config)
    prototypes = _build_prototype_bank(cfg).float().contiguous()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size)
    load_one_shot_checkpoint(model, payload)
    model_hash = state_hash(model)
    if model_hash != admission.get("checkpoint_model_hash"):
        raise ValueError("v74 checkpoint hash does not match v73 admission")
    pattern, support = semantic_support_pattern(codes, prototypes)
    with torch.no_grad():
        all_codes = codes[:, None, :].expand(CODE_COUNT, LABEL_COUNT, 128).reshape(-1, 128)
        requested = torch.arange(LABEL_COUNT).repeat(CODE_COUNT)
        _, diagnostics = _load_generator(cfg, checkpoint).forward_with_diagnostics(all_codes, requested)
        base = diagnostics["generator_base_command"].reshape(
            CODE_COUNT, LABEL_COUNT, 1, 200, 200
        )
    if not torch.allclose(base, base[:1], rtol=0.0, atol=0.0):
        raise ValueError("generator base command is not code invariant")
    return cfg, codes, labels, prototypes, support, {
        "admission": admission,
        "report": report,
        "decision": decision,
        "payload": payload,
        "pattern": pattern,
        "base": base,
    }, model_hash


def _forward_with_support_full(model: OneShotOpticalGenerator, code: torch.Tensor, labels: torch.Tensor, support: torch.Tensor, amplitude: float):
    original = model._diagnostic_slm_override

    def semantic_override(slm_command, code_value, diagnostics):
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
        return model(code, labels, return_optical_fields=True)
    finally:
        model._diagnostic_slm_override = original


class _CodeSS:
    """Streaming exact code-main SS over spatial regions."""

    def __init__(self, size: int, device: torch.device, *, active_size: int = ACTIVE_SIZE, shell_sizes: tuple[int, ...] = SHELL_SIZES) -> None:
        self.size = size
        self.device = device
        self.active_size = active_size
        self.shell_sizes = shell_sizes
        self.mean: torch.Tensor | None = None
        self.count = 0
        self.ss: dict[str, torch.Tensor] = {}
        self.masks = self._masks(size, device, active_size, shell_sizes)

    @staticmethod
    def _masks(size: int, device: torch.device, active_size: int, shell_sizes: tuple[int, ...]) -> dict[str, torch.Tensor]:
        center = torch.zeros((size, size), dtype=torch.bool, device=device)
        top = (size - active_size) // 2
        center[top : top + active_size, top : top + active_size] = True
        masks: dict[str, torch.Tensor] = {"full": torch.ones_like(center), "active": center, "exterior": ~center}
        previous = torch.zeros_like(center)
        for shell in shell_sizes:
            current = torch.zeros_like(center)
            offset = (size - shell) // 2
            current[offset : offset + shell, offset : offset + shell] = True
            masks[f"shell_{shell}"] = current & ~previous
            previous = current
        return masks

    def add_code_mean(self, value: torch.Tensor) -> None:
        value = value.to(self.device)
        if self.mean is None:
            self.mean = torch.zeros_like(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta_after = value - self.mean
        contribution = delta.real * delta_after.real + delta.imag * delta_after.imag if value.is_complex() else delta * delta_after
        for name, mask in self.masks.items():
            self.ss[name] = self.ss.get(name, contribution.new_zeros(())) + contribution[..., mask].sum()

    def report(self) -> dict[str, Any]:
        if self.mean is None or self.count != CODE_COUNT:
            raise ValueError("incomplete code stream")
        code_ss = {name: float((LABEL_COUNT * value).detach().cpu()) for name, value in self.ss.items()}
        full = code_ss["full"]
        return {
            "code_effect_ss": code_ss,
            "full_code_effect_ss": full,
            "active_fraction_of_full": code_ss["active"] / max(full, 1.0e-12),
            "exterior_fraction_of_full": code_ss["exterior"] / max(full, 1.0e-12),
            "finite": all(torch.isfinite(value) for value in self.ss.values()),
        }


def _field(outputs: dict[str, Any], name: str) -> torch.Tensor:
    record = outputs["optical_fields"][name]
    return torch.complex(record["real"], record["imag"]).squeeze(0).squeeze(0)


def _active_reference(model, codes, support, plane_names, device):
    references: dict[int, dict[str, torch.Tensor]] = {}
    for label_value in range(LABEL_COUNT):
        label = torch.tensor([label_value], device=device, dtype=torch.long)
        with torch.no_grad():
            output = _forward_with_support_full(model, codes[:1].to(device), label, support, 0.0)
        references[label_value] = {
            name: _field(output, name)[(STATE_SIZE - ACTIVE_SIZE)//2:(STATE_SIZE + ACTIVE_SIZE)//2,
                                        (STATE_SIZE - ACTIVE_SIZE)//2:(STATE_SIZE + ACTIVE_SIZE)//2].detach().clone()
            for name in plane_names
        }
        del output
    return references


def run_cpu_admission(*, config, checkpoint, validation_dataset, v73_admission, v73_report, v73_decision, output):
    cfg, codes, labels, prototypes, support, aux, model_hash = _load_context(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
    )
    calibration = aux["admission"]["calibration"]
    checks = {
        "v73_links": aux["decision"]["status"] == "semantic-support-code-factor-attenuated",
        "amplitude_exact": abs(float(calibration["amplitude"]) - EXPECTED_AMPLITUDE) <= 1.0e-9,
        "validation_contract": tuple(codes.shape) == (100, 128) and set(labels.tolist()) == set(range(10)),
        "prototype_finite": bool(torch.isfinite(prototypes).all()),
        "support_finite": bool(torch.isfinite(support).all()),
        "checkpoint_hash_match": model_hash == aux["admission"].get("checkpoint_model_hash"),
        "base_code_invariant": True,
    }
    report = {
        "schema": ADMISSION_SCHEMA,
        "diagnostic_only": True,
        "status": "v74-full-state-admission-passed" if all(checks.values()) else "harness-invalid",
        "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "v73_admission": str(v73_admission), "v73_report": str(v73_report), "v73_decision": str(v73_decision),
        "validation_sample_count": 100, "crossed_output_count": 1000, "batch_size": 1,
        "return_optical_fields": True, "transient_full_state_used": True,
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only",
        "calibration": calibration, "amplitude": float(calibration["amplitude"]),
        "codes_hash": tensor_hash(codes), "prototype_hash": tensor_hash(prototypes),
        "checkpoint_model_hash": model_hash, "checks": checks,
        "optimizer_created": False, "parameter_update_count": 0,
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _transition_report(planes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for previous, current in zip(PLANE_ORDER, PLANE_ORDER[1:]):
        result[current] = {"from": previous}
        for feature in ("intensity", "raw_complex", "gauge_complex"):
            before = planes[previous][feature]["code_effect_ss"]
            after = planes[current][feature]["code_effect_ss"]
            result[current][feature] = {
                "full_retention": after["full"] / max(before["full"], 1.0e-12),
                "active_retention": after["active"] / max(before["active"], 1.0e-12),
                "exterior_gain": after["exterior"] - before["exterior"],
                "active_loss": before["active"] - after["active"],
                "exterior_explanation_ratio": (after["exterior"] - before["exterior"]) / max(before["active"] - after["active"], 1.0e-12),
            }
    return result


def _close(value: float, reference: float, *, rtol: float = 1.0e-5) -> bool:
    return abs(float(value) - float(reference)) <= rtol * max(1.0, abs(float(reference)))


def run_cuda(*, config, checkpoint, validation_dataset, v73_admission, v73_report, v73_decision, cpu_admission, output, device="cuda"):
    admission = _read(cpu_admission)
    if admission.get("status") != "v74-full-state-admission-passed":
        raise ValueError("v74 CUDA audit requires a passing v74 admission")
    cfg, codes, labels, prototypes, support, aux, _ = _load_context(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
    )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v74 full-state audit requires CUDA")
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    load_one_shot_checkpoint(model, aux["payload"])
    before_hash = state_hash(model)
    amplitude = float(admission["amplitude"])
    support = support[0]
    plane_names = PLANE_ORDER
    references = _active_reference(model, codes, support, plane_names, dev)
    stats = {
        name: {
            "intensity": _CodeSS(STATE_SIZE, dev),
            "raw_complex": _CodeSS(STATE_SIZE, dev),
            "gauge_complex": _CodeSS(STATE_SIZE, dev),
        }
        for name in plane_names
    }
    current: dict[str, dict[str, torch.Tensor]] = {}
    camera_values = []
    energy_records: list[dict[str, float]] = []
    finite = True
    peak_allocated = 0.0
    peak_reserved = 0.0
    center = (STATE_SIZE - ACTIVE_SIZE) // 2
    with torch.no_grad():
        for code_index, code in enumerate(codes):
            current = {
                name: {
                    "intensity": torch.zeros((STATE_SIZE, STATE_SIZE), device=dev),
                    "raw_complex": torch.zeros((STATE_SIZE, STATE_SIZE), dtype=torch.complex64, device=dev),
                    "gauge_complex": torch.zeros((STATE_SIZE, STATE_SIZE), dtype=torch.complex64, device=dev),
                }
                for name in plane_names
            }
            for label_value in range(LABEL_COUNT):
                label = torch.tensor([label_value], device=dev, dtype=torch.long)
                output_obj = _forward_with_support_full(model, code[None].to(dev), label, support, amplitude)
                for value in output_obj.values():
                    if torch.is_tensor(value):
                        finite = finite and bool(torch.isfinite(value).all())
                energy_records.append({
                    key: float(value.detach().float().mean())
                    for key, value in output_obj.items()
                    if torch.is_tensor(value)
                    and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
                })
                for name in plane_names:
                    field = _field(output_obj, name)
                    active = field[center:center + ACTIVE_SIZE, center:center + ACTIVE_SIZE]
                    reference = references[label_value][name].to(dev)
                    overlap = (reference.conj() * active).sum()
                    if float(overlap.abs()) <= 1.0e-8:
                        finite = False
                    aligned = field * torch.exp(-1j * torch.angle(overlap))
                    current[name]["raw_complex"] += field
                    current[name]["gauge_complex"] += aligned
                    current[name]["intensity"] += field.abs().square()
                camera_values.append(output_obj["physical_camera_intensity_raw"].detach().cpu().squeeze(0))
                del output_obj
            for name in plane_names:
                stats[name]["raw_complex"].add_code_mean(current[name]["raw_complex"] / LABEL_COUNT)
                stats[name]["gauge_complex"].add_code_mean(current[name]["gauge_complex"] / LABEL_COUNT)
                stats[name]["intensity"].add_code_mean(current[name]["intensity"] / LABEL_COUNT)
            if dev.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 1024**3)
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 1024**3)
            del current
    plane_report: dict[str, Any] = {}
    for name in plane_names:
        plane_report[name] = {
            feature: stats[name][feature].report()
            for feature in ("intensity", "raw_complex", "gauge_complex")
        }
        plane_report[name]["energy_mean"] = float(stats[name]["intensity"].mean.abs().square().sum().detach().cpu()) if stats[name]["intensity"].mean is not None else 0.0
    camera_raw = torch.stack(camera_values).reshape(CODE_COUNT, LABEL_COUNT, 1, 200, 200)
    camera_tensor = fixed_exposure_view(camera_raw, SCALE)
    targets = prototypes[None].expand(CODE_COUNT, -1, *prototypes.shape[1:])
    camera_metrics = metric_bundle(
        camera_tensor.flatten(0, 1), targets.flatten(0, 1), torch.arange(10).repeat(100), prototypes
    )
    diagonal = camera_tensor[torch.arange(CODE_COUNT), labels]
    diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    v73_report = aux["report"]
    v73_factor = v73_report["factor_decomposition"]
    active_replay_checks: dict[str, bool] = {}
    v73_plane_names = {
        "encoder_full_grid_input": "pre_encoder",
        "encoder_layer_1": "encoder_layer_1",
        "encoder_layer_2": "encoder_layer_2",
        "ols_output": "ols_output",
        "decoder_layer_1": "decoder_layer_1",
        "decoder_layer_2": "decoder_layer_2",
        "chip": "chip",
    }
    for name, old_name in v73_plane_names.items():
        active_replay_checks[f"{name}.intensity"] = _close(
            plane_report[name]["intensity"]["code_effect_ss"]["active"],
            v73_factor[old_name]["code_effect_ss"],
        )
        if old_name != "pre_encoder":
            active_replay_checks[f"{name}.raw_complex"] = _close(
                plane_report[name]["raw_complex"]["code_effect_ss"]["active"],
                v73_report["complex_factor_decomposition"][old_name]["code_effect_ss"],
            )
            active_replay_checks[f"{name}.gauge_complex"] = _close(
                plane_report[name]["gauge_complex"]["code_effect_ss"]["active"],
                v73_report["gauge_aligned_complex_factor_decomposition"][old_name]["code_effect_ss"],
            )
    for metric_name in ("clean_prototype_corr", "clean_prototype_dice", "prototype_accuracy"):
        active_replay_checks[f"camera.crossed.{metric_name}"] = _close(
            camera_metrics[metric_name], v73_report["crossed_metrics"][metric_name]
        )
        active_replay_checks[f"camera.diagonal.{metric_name}"] = _close(
            diagonal_metrics[metric_name], v73_report["diagonal_metrics"][metric_name]
        )
    replay_pass = all(active_replay_checks.values())
    energy_mean = {
        key: sum(item.get(key, 0.0) for item in energy_records) / max(len(energy_records), 1)
        for key in sorted({key for item in energy_records for key in item})
    }
    energy_values = [
        value for key, value in energy_mean.items()
        if "guard" in key or "discarded" in key
    ]
    energy_pass = bool(energy_values) and max(energy_values) <= 1.0e-4
    report = {
        "schema": REPORT_SCHEMA, "diagnostic_only": True, "status": "v74-full-state-partition-complete",
        "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "v73_admission": str(v73_admission), "v73_report": str(v73_report), "v73_decision": str(v73_decision), "cpu_admission": str(cpu_admission),
        "validation_sample_count": 100, "crossed_output_count": 1000, "batch_size": 1,
        "return_optical_fields": True, "transient_full_state_used": True,
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only",
        "state_grid_size": STATE_SIZE, "active_view_size": ACTIVE_SIZE, "shell_sizes": list(SHELL_SIZES),
        "sampling_contract": {
            "state_grid_shape": [STATE_SIZE, STATE_SIZE],
            "state_grid_pixel_size_m": 2.0e-6,
            "state_grid_fov_m": 4.096e-3,
            "active_shape": [ACTIVE_SIZE, ACTIVE_SIZE],
            "active_pixel_size_m": 2.0e-6,
            "fft_grid_size": 3072,
            "full_state_persisted": False,
        },
        "exposure_mode": "locked", "exposure_scale": SCALE, "intervention": "v73_semantic_support_replay",
        "gauge_alignment": {
            "mode": "requested_label_base_command_active_reference",
            "reference_labels": list(range(LABEL_COUNT)),
            "reference_fields_persisted": False,
        },
        "amplitude": amplitude, "planes": plane_report, "transitions": _transition_report(plane_report),
        "camera_metrics": camera_metrics, "diagonal_metrics": diagonal_metrics,
        "camera_replay_reference": {"v73_status": v73_report.get("status"), "v73_camera_code_factor": v73_report.get("camera_code_factor")},
        "v73_active_replay_checks": active_replay_checks,
        "v73_active_replay_pass": replay_pass,
        "energy_records_mean": energy_mean, "energy_pass": energy_pass,
        "finite": finite, "parameters_unchanged": before_hash == state_hash(model), "optimizer_created": False, "parameter_update_count": 0,
        "peak_cuda_allocated_gib": peak_allocated, "peak_cuda_reserved_gib": peak_reserved,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(aux["payload"]),
        "test_split_used": False, "teacher_alignment_computed": False,
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--v73-admission", required=True, type=Path)
    parser.add_argument("--v73-report", required=True, type=Path)
    parser.add_argument("--v73-decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-admission", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        result = run_cpu_admission(
            config=args.config, checkpoint=args.checkpoint, validation_dataset=args.validation_dataset,
            v73_admission=args.v73_admission, v73_report=args.v73_report, v73_decision=args.v73_decision,
            output=args.output,
        )
    else:
        if args.cpu_admission is None:
            raise ValueError("--cpu-admission is required for CUDA audit")
        result = run_cuda(
            config=args.config, checkpoint=args.checkpoint, validation_dataset=args.validation_dataset,
            v73_admission=args.v73_admission, v73_report=args.v73_report, v73_decision=args.v73_decision,
            cpu_admission=args.cpu_admission, output=args.output, device=args.device,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
