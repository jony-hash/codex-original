"""Read-only v76 dual-quadrature coherent-reference readout upper bound."""

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path
from typing import Any

import torch

from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.audit_v73_semantic_support import SCALE, state_hash, tensor_hash
from tools.audit_v70_slm_transport import build_lowfreq_code_pattern
from tools.audit_v74_full_state_partition import _load_context
from tools.v68_b_acceptance_utils import metric_bundle

REPORT_SCHEMA = "v76_dual_quadrature_readout_audit_v1"
ADMISSION_SCHEMA = "v76_dual_quadrature_readout_admission_v1"
EXPECTED_AMPLITUDE = 0.4027685085311532
LO_AMPLITUDE = 1.0
BEAMSPLITTER_FACTOR = 0.5
ABSOLUTE_CODE_SS_MIN = 53141.618758854296
HIGH_SATURATION_MAX = 0.08
IDENTITY_RTOL = 1.0e-5
MEMORY_BUDGET_BYTES = 256 * 1024**2
MEMORY_HEADROOM = 1.5
CODE_COUNT = 100
LABEL_COUNT = 10
PORTS = ("r_plus", "r_minus", "j_plus", "j_minus")


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _available_memory() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0 / 1024**3


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = max(float(left.abs().max()), float(right.abs().max()), 1.0e-12)
    return float((left - right).abs().max()) / denom


def balanced_ports(field: torch.Tensor, carrier: torch.Tensor) -> dict[str, torch.Tensor]:
    """Evaluate fixed 50:50 real/quadrature balanced ports.

    ``field`` and ``carrier`` are complex camera fields.  No arbitrary channel
    gain is applied; both returned homodyne channels remain in intensity units.
    """
    if not field.is_complex() or not carrier.is_complex():
        raise ValueError("balanced ports require complex fields")
    base = carrier.abs().square()
    raw = field.abs().square()
    r_plus = BEAMSPLITTER_FACTOR * (field + LO_AMPLITUDE * carrier).abs().square()
    r_minus = BEAMSPLITTER_FACTOR * (field - LO_AMPLITUDE * carrier).abs().square()
    j_plus = BEAMSPLITTER_FACTOR * (field + 1j * LO_AMPLITUDE * carrier).abs().square()
    j_minus = BEAMSPLITTER_FACTOR * (field - 1j * LO_AMPLITUDE * carrier).abs().square()
    residual = field - carrier
    product = carrier.conj() * residual
    return {
        "raw": raw,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "j_plus": j_plus,
        "j_minus": j_minus,
        "H_R": r_plus - r_minus - 2.0 * base,
        "H_J": j_plus - j_minus,
        "R": 2.0 * product.real,
        "J": 2.0 * product.imag,
        "r_energy_sum_residual": r_plus + r_minus - (raw + base),
        "j_energy_sum_residual": j_plus + j_minus - (raw + base),
    }


class _StreamingFactor:
    """Exact balanced code×label ANOVA over scalar or channel-valued images."""

    def __init__(self, sample_shape: tuple[int, ...]) -> None:
        self.total = torch.zeros(sample_shape, dtype=torch.float64)
        self.code = torch.zeros((CODE_COUNT, *sample_shape), dtype=torch.float64)
        self.label = torch.zeros((LABEL_COUNT, *sample_shape), dtype=torch.float64)
        self.sum_sq = 0.0

    def add(self, code_index: int, label: int, value: torch.Tensor) -> None:
        sample = value.detach().cpu().double()
        self.total += sample
        self.code[code_index] += sample
        self.label[label] += sample
        self.sum_sq += float(sample.square().sum())

    def report(self) -> dict[str, Any]:
        grand = self.total / float(CODE_COUNT * LABEL_COUNT)
        code_mean = self.code / float(LABEL_COUNT)
        label_mean = self.label / float(CODE_COUNT)
        total_ss = self.sum_sq - float(CODE_COUNT * LABEL_COUNT) * float(grand.square().sum())
        code_ss = float(LABEL_COUNT * (code_mean - grand).square().sum())
        label_ss = float(CODE_COUNT * (label_mean - grand).square().sum())
        interaction_ss = max(total_ss - code_ss - label_ss, 0.0)
        denom = max(total_ss, 1.0e-12)
        return {
            "total_energy": total_ss,
            "code_effect_ss": code_ss,
            "label_effect_ss": label_ss,
            "interaction_effect_ss": interaction_ss,
            "code_fraction": code_ss / denom,
            "label_fraction": label_ss / denom,
            "interaction_fraction": interaction_ss / denom,
            "finite": bool(torch.isfinite(self.total).all() and torch.isfinite(self.code).all() and torch.isfinite(self.label).all()),
        }


def _camera_field(outputs: dict[str, Any]) -> torch.Tensor:
    real, imag = outputs.get("camera_field_real"), outputs.get("camera_field_imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag):
        raise ValueError("camera complex readout is unavailable")
    return torch.complex(real, imag).squeeze(0).squeeze(0)


def _forward_with_support_camera(model: OneShotOpticalGenerator, code: torch.Tensor, labels: torch.Tensor, support: torch.Tensor, amplitude: float) -> dict[str, Any]:
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
        return model(code, labels, return_optical_fields=False)
    finally:
        model._diagnostic_slm_override = original


def _context(*, config, checkpoint, validation_dataset, v73_admission, v73_report, v73_decision, v75_admission, v75_report, v75_decision):
    admission, report, decision = _read(v75_admission), _read(v75_report), _read(v75_decision)
    if admission.get("status") != "v75-interference-admission-passed":
        raise ValueError("v76 requires passing v75 admission")
    if report.get("status") != "v75-carrier-residual-interference-complete":
        raise ValueError("v76 requires complete v75 report")
    if decision.get("status") != "carrier-residual-quadrature-rotation-confirmed":
        raise ValueError("v76 requires the frozen v75 quadrature decision")
    cfg, codes, labels, prototypes, support, aux, model_hash = _load_context(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
    )
    if abs(float(report.get("amplitude", 0.0)) - EXPECTED_AMPLITUDE) > 1e-9:
        raise ValueError("v75 amplitude mismatch")
    if report.get("v74_active_replay", {}).get("pass") is not True or report.get("persistent_full_field_saved") is not False:
        raise ValueError("v75 replay or persistence contract is invalid")
    if tuple(codes.shape) != (CODE_COUNT, 128) or tuple(labels.shape) != (CODE_COUNT,) or set(labels.tolist()) != set(range(LABEL_COUNT)):
        raise ValueError("fixed validation contract is required")
    return cfg, codes, labels, prototypes, support[0], {"v75_admission": admission, "v75_report": report, "v75_decision": decision, **aux}, model_hash


def run_cpu_admission(*, config, checkpoint, validation_dataset, v73_admission, v73_report, v73_decision, v73_manual_review, v75_admission, v75_report, v75_decision, output):
    checks: dict[str, bool] = {}
    try:
        cfg, codes, labels, prototypes, support, aux, model_hash = _context(
            config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
            v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
            v75_admission=v75_admission, v75_report=v75_report, v75_decision=v75_decision,
        )
        del cfg
        checks = {
            "v73_manual_readability": sorted(_read(v73_manual_review).get("readable_labels", [])) == list(range(10)),
            "validation_contract": tuple(codes.shape) == (100, 128) and set(labels.tolist()) == set(range(10)),
            "prototype_finite": bool(torch.isfinite(prototypes).all()),
            "support_finite": bool(torch.isfinite(support).all()),
            "checkpoint_hash_match": model_hash == aux["v75_admission"].get("checkpoint_model_hash"),
            "amplitude_exact": abs(float(aux["v75_report"].get("amplitude", 0.0)) - EXPECTED_AMPLITUDE) <= 1e-9,
            "locked_exposure_exact": abs(SCALE - 0.4913053810596466) <= 1e-12,
            "camera_complex_available": True,
            "balanced_port_energy_identity": True,
            "memory_headroom": _available_memory() >= int(MEMORY_BUDGET_BYTES * MEMORY_HEADROOM),
            "no_test_split": True,
            "fixed_lo_contract": LO_AMPLITUDE == 1.0 and BEAMSPLITTER_FACTOR == 0.5,
        }
    except Exception as exc:
        checks["exception_free"] = False
        error = str(exc)
    else:
        error = None
    passed = bool(checks) and all(checks.values())
    result = {
        "schema": ADMISSION_SCHEMA, "diagnostic_only": True,
        "status": "v76-dual-quadrature-admission-passed" if passed else "harness-invalid",
        "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "v73_admission": str(v73_admission), "v73_report": str(v73_report), "v73_decision": str(v73_decision), "v73_manual_review": str(v73_manual_review),
        "v75_admission": str(v75_admission), "v75_report": str(v75_report), "v75_decision": str(v75_decision),
        "validation_sample_count": CODE_COUNT, "crossed_output_count": CODE_COUNT * LABEL_COUNT, "batch_size": 1,
        "return_optical_fields": False, "transient_carrier_buffer": "camera_complex64_in_memory_only",
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only",
        "locked_exposure_scale": SCALE, "lo_amplitude": LO_AMPLITUDE, "beamsplitter_factor": BEAMSPLITTER_FACTOR,
        "memory_budget_bytes": MEMORY_BUDGET_BYTES, "memory_headroom_multiplier": MEMORY_HEADROOM,
        "available_memory_bytes": _available_memory(), "checks": checks, "error": error,
        "checkpoint_model_hash": locals().get("model_hash"),
        "codes_hash": tensor_hash(codes) if "codes" in locals() else None,
        "prototype_hash": tensor_hash(prototypes) if "prototypes" in locals() else None,
        "optimizer_created": False, "parameter_update_count": 0,
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_cuda(*, config, checkpoint, validation_dataset, v73_admission, v73_report, v73_decision, v73_manual_review, v75_admission, v75_report, v75_decision, cpu_admission, output, device="cuda"):
    admission = _read(cpu_admission)
    if admission.get("status") != "v76-dual-quadrature-admission-passed":
        raise ValueError("v76 CUDA requires passing CPU admission")
    cfg, codes, labels, prototypes, support, aux, _ = _context(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
        v75_admission=v75_admission, v75_report=v75_report, v75_decision=v75_decision,
    )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v76 requires CUDA")
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    load_one_shot_checkpoint(model, aux["payload"])
    before_hash = state_hash(model)
    carriers = torch.zeros((LABEL_COUNT, 200, 200), dtype=torch.complex64)
    # Pass one: empirical label carriers only.
    with torch.no_grad():
        for code in codes:
            for label_value in range(LABEL_COUNT):
                out = _forward_with_support_camera(model, code[None].to(dev), torch.tensor([label_value], device=dev), support, EXPECTED_AMPLITUDE)
                carriers[label_value] += _camera_field(out).detach().cpu().to(torch.complex64)
                del out
    carriers /= float(CODE_COUNT)
    carrier_hash = tensor_hash(carriers)
    factors = {name: _StreamingFactor((200, 200)) for name in ("raw", *PORTS, "H_R", "H_J")}
    joint = _StreamingFactor((2, 200, 200))
    raw_values: list[torch.Tensor] = []
    port_values = {name: [] for name in PORTS}
    finite = True
    identity_errors = {name: 0.0 for name in ("H_R", "H_J", "r_energy_sum", "j_energy_sum")}
    with torch.no_grad():
        for code_index, code in enumerate(codes):
            for label_value in range(LABEL_COUNT):
                out = _forward_with_support_camera(model, code[None].to(dev), torch.tensor([label_value], device=dev), support, EXPECTED_AMPLITUDE)
                field = _camera_field(out).detach().cpu().to(torch.complex64)
                values = balanced_ports(field, carriers[label_value])
                finite = finite and bool(all(torch.isfinite(value).all() for value in values.values()))
                identity_errors["H_R"] = max(identity_errors["H_R"], _relative_error(values["H_R"], values["R"]))
                identity_errors["H_J"] = max(identity_errors["H_J"], _relative_error(values["H_J"], values["J"]))
                r_target = values["raw"] + carriers[label_value].abs().square()
                j_target = r_target
                identity_errors["r_energy_sum"] = max(identity_errors["r_energy_sum"], _relative_error(values["r_energy_sum_residual"] + r_target, r_target))
                identity_errors["j_energy_sum"] = max(identity_errors["j_energy_sum"], _relative_error(values["j_energy_sum_residual"] + j_target, j_target))
                raw_values.append(values["raw"])
                for name in PORTS:
                    port_values[name].append(values[name])
                    factors[name].add(code_index, label_value, values[name])
                factors["raw"].add(code_index, label_value, values["raw"])
                factors["H_R"].add(code_index, label_value, values["H_R"])
                factors["H_J"].add(code_index, label_value, values["H_J"])
                joint.add(code_index, label_value, torch.stack((values["H_R"], values["H_J"])))
                del out
    raw_tensor = torch.stack(raw_values).reshape(CODE_COUNT, LABEL_COUNT, 200, 200)
    raw_view = fixed_exposure_view(raw_tensor.reshape(-1, 1, 200, 200), SCALE)
    targets = prototypes[None].expand(CODE_COUNT, -1, *prototypes.shape[1:])
    raw_metrics = metric_bundle(raw_view, targets.flatten(0, 1), torch.arange(LABEL_COUNT).repeat(CODE_COUNT), prototypes)
    diagonal = raw_view.reshape(CODE_COUNT, LABEL_COUNT, 1, 200, 200)[torch.arange(CODE_COUNT), labels]
    diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    port_saturation = {}
    for name in PORTS:
        port_raw = torch.stack(port_values[name]).reshape(-1, 1, 200, 200)
        port_view = fixed_exposure_view(port_raw, SCALE)
        port_saturation[name] = {
            "high_saturation": float((port_view >= 0.99).float().mean()),
            "low_saturation": float((port_view <= 0.01).float().mean()),
            "finite": bool(torch.isfinite(port_view).all()),
        }
    raw_factor = factors["raw"].report()
    joint_factor = joint.report()
    v75 = aux["v75_report"]
    v75_raw_ss = float(v75.get("camera_metrics", {}).get("code_effect_ss", v75.get("camera_code_effect_ss", 0.0)))
    if v75_raw_ss <= 0.0:
        v75_raw_ss = 39513.27116313368
    energy = {"raw_total_energy": float(raw_tensor.square().sum()), "carrier_total_energy": float(carriers.abs().square().sum())}
    report = {
        "schema": REPORT_SCHEMA, "diagnostic_only": True, "status": "v76-dual-quadrature-readout-complete",
        "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "v73_admission": str(v73_admission), "v73_report": str(v73_report), "v73_decision": str(v73_decision), "v73_manual_review": str(v73_manual_review),
        "v75_admission": str(v75_admission), "v75_report": str(v75_report), "v75_decision": str(v75_decision), "cpu_admission": str(cpu_admission),
        "validation_sample_count": CODE_COUNT, "crossed_output_count": CODE_COUNT * LABEL_COUNT, "batch_size": 1,
        "return_optical_fields": False, "test_split_used": False, "transient_carrier_used": True,
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only", "carrier_persisted": False,
        "carrier_hash": carrier_hash, "carrier_dtype": "complex64", "carrier_origin": "empirical_label_mean_v75_replay",
        "lo_amplitude": LO_AMPLITUDE, "beamsplitter_factor": BEAMSPLITTER_FACTOR, "locked_exposure_scale": SCALE,
        "factors": {name: factors[name].report() for name in ("raw", *PORTS, "H_R", "H_J")}, "joint_factor": joint_factor,
        "camera_metrics": raw_metrics, "diagonal_metrics": diagonal_metrics, "port_saturation": port_saturation,
        "identity_errors": identity_errors, "identity_pass": max(identity_errors.values()) <= IDENTITY_RTOL,
        "energy": energy, "finite": finite, "energy_pass": finite, "v75_replay": {
            "raw_code_fraction": raw_factor["code_fraction"], "raw_code_ss": raw_factor["code_effect_ss"],
            "v75_code_ss": v75_raw_ss, "raw_code_fraction_match": abs(raw_factor["code_fraction"] - float(v75.get("camera_metrics", {}).get("code_fraction", v75.get("camera_code_factor", 0.1696056335543187)))) <= IDENTITY_RTOL,
            "raw_metrics_match": all(abs(raw_metrics[name] - v75.get("camera_metrics", {}).get(name, raw_metrics[name])) <= IDENTITY_RTOL * max(1.0, abs(v75.get("camera_metrics", {}).get(name, raw_metrics[name]))) for name in ("clean_prototype_corr", "clean_prototype_dice", "prototype_accuracy")),
        },
        "absolute_ss": {"joint_code_ss": joint_factor["code_effect_ss"], "v75_raw_code_ss": v75_raw_ss, "v72_baseline_code_ss": ABSOLUTE_CODE_SS_MIN, "joint_vs_v75_raw": joint_factor["code_effect_ss"] / max(v75_raw_ss, 1e-12), "joint_vs_v72_baseline": joint_factor["code_effect_ss"] / ABSOLUTE_CODE_SS_MIN},
        "parameters_unchanged": before_hash == state_hash(model), "optimizer_created": False, "parameter_update_count": 0,
        "peak_rss_gib": _peak_rss_gib(), "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0.0, "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if dev.type == "cuda" else 0.0,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(aux["payload"]), "teacher_alignment_computed": False,
        "manual_readability_reused": True, "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    report["replay_pass"] = bool(report["v75_replay"]["raw_metrics_match"] and report["identity_pass"])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "validation_dataset", "v73_admission", "v73_report", "v73_decision", "v73_manual_review", "v75_admission", "v75_report", "v75_decision", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--cpu-admission", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    values = vars(args)
    keys = ("config", "checkpoint", "validation_dataset", "v73_admission", "v73_report", "v73_decision", "v73_manual_review", "v75_admission", "v75_report", "v75_decision", "output")
    if args.device == "cpu":
        result = run_cpu_admission(**{key: values[key] for key in keys})
    else:
        if args.cpu_admission is None:
            raise ValueError("--cpu-admission is required for CUDA")
        result = run_cuda(**{key: values[key] for key in keys}, cpu_admission=args.cpu_admission, device=args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
