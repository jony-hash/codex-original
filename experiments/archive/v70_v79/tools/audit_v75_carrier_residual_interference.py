"""Read-only v75 carrier/residual phase-to-intensity attribution audit.

The audit deliberately keeps the full optical fields transient.  Pass one builds
an empirical complex carrier per requested label; pass two reduces the
carrier/residual decomposition to scalar ANOVA statistics.
"""

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
from tools.audit_v74_full_state_partition import (
    ACTIVE_SIZE,
    CODE_COUNT,
    LABEL_COUNT,
    STATE_SIZE,
    _CodeSS,
    _field,
    _forward_with_support_full,
    _load_context,
)
from tools.v68_b_acceptance_utils import metric_bundle

REPORT_SCHEMA = "v75_carrier_residual_interference_audit_v1"
ADMISSION_SCHEMA = "v75_interference_admission_v1"
EXPECTED_AMPLITUDE = 0.4027685085311532
MEMORY_BUDGET_BYTES = int(1.6 * 1024**3)
MEMORY_HEADROOM = 1.5
PLANES = (
    "encoder_full_grid_input",
    "encoder_layer_1",
    "encoder_layer_2",
    "decoder_layer_1",
    "decoder_layer_2",
    "camera_field",
)
ALIASES = {"ols_output": "encoder_layer_2", "chip": "decoder_layer_2"}


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _available_memory() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def carrier_residual_terms(field: torch.Tensor, carrier: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return D, R, J, Q and delta-I with the exact algebraic identity."""
    field = torch.complex(field, torch.zeros_like(field)) if not field.is_complex() else field
    carrier = torch.complex(carrier, torch.zeros_like(carrier)) if not carrier.is_complex() else carrier
    residual = field - carrier
    product = carrier.conj() * residual
    real_interference = 2.0 * product.real
    quadrature_interference = 2.0 * product.imag
    residual_self = residual.abs().square()
    delta_intensity = field.abs().square() - carrier.abs().square()
    return {
        "residual": residual,
        "R": real_interference,
        "J": quadrature_interference,
        "Q": residual_self,
        "delta_I": delta_intensity,
    }


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = max(float(left.abs().max()), float(right.abs().max()), 1.0e-12)
    return float((left - right).abs().max()) / denom


def _peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0 / 1024**3


def _validate_context(
    *, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path,
    v73_admission: str | Path, v73_report: str | Path, v73_decision: str | Path,
    v74_admission: str | Path, v74_report: str | Path, v74_decision: str | Path,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], str]:
    v74a, v74r, v74d = _read(v74_admission), _read(v74_report), _read(v74_decision)
    if v74a.get("status") != "v74-full-state-admission-passed":
        raise ValueError("v75 requires v74 CPU admission")
    if v74r.get("status") != "v74-full-state-partition-complete":
        raise ValueError("v75 requires a complete v74 report")
    if v74d.get("status") != "full-state-phase-to-intensity-redistribution-confirmed":
        raise ValueError("v75 requires the frozen v74 phase/intensity result")
    cfg, codes, labels, prototypes, support, aux, model_hash = _load_context(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
    )
    if abs(float(v74r.get("amplitude", 0.0)) - EXPECTED_AMPLITUDE) > 1.0e-9:
        raise ValueError("v74 amplitude mismatch")
    if not bool(v74r.get("v73_active_replay_pass")) or not bool(v74r.get("energy_pass")):
        raise ValueError("v74 replay/energy gate is not valid")
    if v74r.get("persistent_full_field_saved") is not False:
        raise ValueError("v74 persisted full-field artifact")
    if tuple(codes.shape) != (CODE_COUNT, 128) or tuple(labels.shape) != (CODE_COUNT,):
        raise ValueError("fixed validation split is required")
    if set(labels.tolist()) != set(range(LABEL_COUNT)):
        raise ValueError("validation split must contain all labels")
    return cfg, codes, labels, prototypes, support, {"v74a": v74a, "v74r": v74r, "v74d": v74d, **aux}, model_hash


def run_cpu_admission(*, config, checkpoint, validation_dataset, v73_admission, v73_report,
                      v73_decision, v74_admission, v74_report, v74_decision, output, v73_manual_review=None):
    checks: dict[str, bool] = {}
    try:
        cfg, codes, labels, prototypes, support, aux, model_hash = _validate_context(
            config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
            v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
            v74_admission=v74_admission, v74_report=v74_report, v74_decision=v74_decision,
        )
        checks = {
            "provenance": True,
            "validation_contract": tuple(codes.shape) == (100, 128) and set(labels.tolist()) == set(range(10)),
            "prototype_finite": bool(torch.isfinite(prototypes).all()),
            "support_finite": bool(torch.isfinite(support).all()),
            "checkpoint_hash_match": model_hash == aux["v74a"].get("checkpoint_model_hash"),
            "amplitude_exact": abs(EXPECTED_AMPLITUDE - float(aux["v74r"].get("amplitude"))) <= 1e-9,
            "sampling_2048_3072": aux["v74r"].get("sampling_contract", {}).get("state_grid_shape") == [2048, 2048]
            and aux["v74r"].get("sampling_contract", {}).get("fft_grid_size") == 3072,
            "camera_complex_available": bool(aux["v74r"].get("camera_field_available", True)),
            "memory_headroom": _available_memory() >= int(MEMORY_BUDGET_BYTES * MEMORY_HEADROOM),
        }
        del cfg
    except Exception as exc:  # admission is intentionally fail-closed
        checks["exception_free"] = False
        error = str(exc)
    else:
        error = None
    passed = bool(checks) and all(checks.values())
    report = {
        "schema": ADMISSION_SCHEMA, "diagnostic_only": True,
        "status": "v75-interference-admission-passed" if passed else "harness-invalid",
        "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "v73_admission": str(v73_admission), "v73_report": str(v73_report), "v73_decision": str(v73_decision),
        "v74_admission": str(v74_admission), "v74_report": str(v74_report), "v74_decision": str(v74_decision),
        "v73_manual_review": str(v73_manual_review) if v73_manual_review else None,
        "validation_sample_count": 100, "crossed_output_count": 1000, "batch_size": 1,
        "return_optical_fields": True, "transient_carrier_buffer": "complex64_in_memory_only",
        "transient_carrier_budget_bytes": MEMORY_BUDGET_BYTES, "memory_headroom_multiplier": MEMORY_HEADROOM,
        "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only",
        "amplitude": EXPECTED_AMPLITUDE, "checks": checks, "available_memory_bytes": _available_memory(),
        "checkpoint_model_hash": locals().get("model_hash"),
        "codes_hash": tensor_hash(codes) if "codes" in locals() else None,
        "prototype_hash": tensor_hash(prototypes) if "prototypes" in locals() else None,
        "error": error, "optimizer_created": False, "parameter_update_count": 0,
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _new_stats(size: int) -> dict[str, _CodeSS]:
    shells = (100, 200, 400, 800, 1600, 2048) if size == STATE_SIZE else ()
    return {term: _CodeSS(size, torch.device("cpu"), shell_sizes=shells) for term in ("R", "J", "Q", "delta_I")}


def _region_sum(value: torch.Tensor, size: int) -> dict[str, float]:
    if size == STATE_SIZE:
        top = (size - ACTIVE_SIZE) // 2
        active = value[top:top + ACTIVE_SIZE, top:top + ACTIVE_SIZE]
        return {"full": float(value.sum()), "active": float(active.sum()), "exterior": float(value.sum() - active.sum())}
    return {"full": float(value.sum()), "active": float(value[(size - ACTIVE_SIZE)//2:(size + ACTIVE_SIZE)//2, (size - ACTIVE_SIZE)//2:(size + ACTIVE_SIZE)//2].sum()), "exterior": float(value.sum() - value[(size - ACTIVE_SIZE)//2:(size + ACTIVE_SIZE)//2, (size - ACTIVE_SIZE)//2:(size + ACTIVE_SIZE)//2].sum())}


def _plane_summary(stats: dict[str, _CodeSS], within_r: dict[str, float], within_j: dict[str, float], residual_energy: dict[str, float], recon_error: dict[str, float]) -> dict[str, Any]:
    result = {term: stats[term].report() for term in stats}
    r_ss = result["R"]["code_effect_ss"]["full"]
    j_ss = result["J"]["code_effect_ss"]["full"]
    delta_ss = result["delta_I"]["code_effect_ss"]["full"]
    q_ss = result["Q"]["code_effect_ss"]["full"]
    covariance = (delta_ss - r_ss - q_ss) / 2.0
    result.update({
        "residual_complex_energy": residual_energy,
        "within_label_R_ss": within_r, "within_label_J_ss": within_j,
        "quadrature_share": j_ss / max(r_ss + j_ss, 1.0e-12),
        "label_coherence": r_ss / max(within_r["full"], 1.0e-12),
        "covariance_ss": covariance,
        "reconstruction_error": {"max": float(recon_error)},
        "identity_delta_equals_R_plus_Q": abs(delta_ss - (r_ss + q_ss + 2.0 * covariance)) / max(abs(delta_ss), 1.0e-12),
    })
    return result


def run_cuda(*, config, checkpoint, validation_dataset, v73_admission, v73_report, v73_decision,
             v74_admission, v74_report, v74_decision, cpu_admission, output, device="cuda", v73_manual_review=None):
    admission = _read(cpu_admission)
    if admission.get("status") != "v75-interference-admission-passed":
        raise ValueError("v75 CUDA audit requires passing CPU admission")
    cfg, codes, labels, prototypes, support, aux, _ = _validate_context(
        config=config, checkpoint=checkpoint, validation_dataset=validation_dataset,
        v73_admission=v73_admission, v73_report=v73_report, v73_decision=v73_decision,
        v74_admission=v74_admission, v74_report=v74_report, v74_decision=v74_decision,
    )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v75 requires CUDA")
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    load_one_shot_checkpoint(model, aux["payload"])
    before_hash = state_hash(model)
    amplitude = EXPECTED_AMPLITUDE
    # semantic_support_pattern carries a leading broadcast dimension; the
    # diagnostic override expects the ten requested-label masks directly.
    support = support[0]
    carriers: dict[str, torch.Tensor] = {}
    # Pass one: empirical carrier B_l, retained only in memory.
    with torch.no_grad():
        for name in PLANES:
            shape = (LABEL_COUNT, 200, 200) if name == "camera_field" else (LABEL_COUNT, STATE_SIZE, STATE_SIZE)
            carriers[name] = torch.zeros(shape, dtype=torch.complex64)
        for code in codes:
            for label_value in range(LABEL_COUNT):
                output_obj = _forward_with_support_full(model, code[None].to(dev), torch.tensor([label_value], device=dev), support, amplitude)
                for name in PLANES:
                    carriers[name][label_value] += _field(output_obj, name).detach().cpu().to(torch.complex64)
                del output_obj
        for name in PLANES:
            carriers[name] /= float(CODE_COUNT)

    stats = {name: _new_stats(200 if name == "camera_field" else STATE_SIZE) for name in PLANES}
    within_r = {name: {region: 0.0 for region in ("full", "active", "exterior")} for name in PLANES}
    within_j = {name: {region: 0.0 for region in ("full", "active", "exterior")} for name in PLANES}
    residual_energy = {name: 0.0 for name in PLANES}
    recon_error = {name: 0.0 for name in PLANES}
    camera_values: list[torch.Tensor] = []
    energy_records: list[dict[str, float]] = []
    finite = True
    alias_pass = True
    peak_allocated = peak_reserved = 0.0
    with torch.no_grad():
        for code in codes:
            code_acc = {name: {term: torch.zeros_like(carriers[name][0].real) for term in ("R", "J", "Q", "delta_I")} for name in PLANES}
            for label_value in range(LABEL_COUNT):
                output_obj = _forward_with_support_full(model, code[None].to(dev), torch.tensor([label_value], device=dev), support, amplitude)
                energy_records.append({
                    key: float(value.detach().float().mean())
                    for key, value in output_obj.items()
                    if torch.is_tensor(value) and (key.endswith("energy_ratio") or key.endswith("guard_band_energy_ratio"))
                })
                if "physical_camera_intensity_raw" in output_obj:
                    camera_values.append(output_obj["physical_camera_intensity_raw"].detach().cpu().squeeze(0))
                for name in PLANES:
                    field = _field(output_obj, name).detach().cpu().to(torch.complex64)
                    terms = carrier_residual_terms(field, carriers[name][label_value])
                    finite = finite and bool(all(torch.isfinite(value).all() for value in terms.values()))
                    identity = terms["R"] + terms["Q"]
                    recon_error[name] = max(recon_error[name], _relative_error(terms["delta_I"], identity))
                    for term in ("R", "J", "Q", "delta_I"):
                        code_acc[name][term] += terms[term]
                    residual_energy[name] += float(terms["residual"].abs().square().sum())
                    sums_r, sums_j = _region_sum(terms["R"].square(), terms["R"].shape[-1]), _region_sum(terms["J"].square(), terms["J"].shape[-1])
                    for region in sums_r:
                        within_r[name][region] += sums_r[region]
                        within_j[name][region] += sums_j[region]
                # strict alias replay of OLS/chip is checked on every sample
                if "ols_output" in output_obj["optical_fields"]:
                    alias_pass = alias_pass and _relative_error(_field(output_obj, "ols_output"), _field(output_obj, "encoder_layer_2")) <= 1.0e-5
                if "chip" in output_obj["optical_fields"]:
                    alias_pass = alias_pass and _relative_error(_field(output_obj, "chip"), _field(output_obj, "decoder_layer_2")) <= 1.0e-5
                del output_obj
            for name in PLANES:
                for term in ("R", "J", "Q", "delta_I"):
                    stats[name][term].add_code_mean(code_acc[name][term] / float(LABEL_COUNT))
            if dev.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 1024**3)
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 1024**3)

    planes = {name: _plane_summary(stats[name], within_r[name], within_j[name], residual_energy[name], recon_error[name]) for name in PLANES}
    # Aliases are scalar replay gates; their plane statistics must match by construction.
    planes["ols_output"] = planes["encoder_layer_2"]
    planes["chip"] = planes["decoder_layer_2"]
    camera_metrics, diagonal_metrics = {}, {}
    if camera_values:
        camera = fixed_exposure_view(torch.stack(camera_values).reshape(CODE_COUNT, LABEL_COUNT, 1, 200, 200), SCALE)
        targets = prototypes[None].expand(CODE_COUNT, -1, *prototypes.shape[1:])
        camera_metrics = metric_bundle(camera.flatten(0, 1), targets.flatten(0, 1), torch.arange(10).repeat(100), prototypes)
        diagonal = camera[torch.arange(CODE_COUNT), labels]
        diagonal_metrics = metric_bundle(diagonal, prototypes[labels], labels, prototypes)
    v74 = aux["v74r"]
    def close(value: float, reference: float, rtol: float = 1.0e-5) -> bool:
        return abs(float(value) - float(reference)) <= rtol * max(1.0, abs(float(reference)))
    replay_checks = {
        "v74_status_linked": v74.get("status") == "v74-full-state-partition-complete",
        "camera_metrics_finite": bool(camera_metrics) and all(torch.isfinite(torch.tensor(value)) for value in camera_metrics.values() if isinstance(value, (int, float))),
    }
    if camera_metrics:
        for metric_name in ("clean_prototype_corr", "clean_prototype_dice", "prototype_accuracy"):
            replay_checks[f"camera.{metric_name}"] = close(camera_metrics[metric_name], v74.get("camera_metrics", {}).get(metric_name, float("nan")))
        for metric_name in ("clean_prototype_corr", "clean_prototype_dice", "prototype_accuracy"):
            replay_checks[f"diagonal.{metric_name}"] = close(diagonal_metrics[metric_name], v74.get("diagonal_metrics", {}).get(metric_name, float("nan")))
    energy_mean = {
        key: sum(item.get(key, 0.0) for item in energy_records) / max(len(energy_records), 1)
        for key in sorted({key for item in energy_records for key in item})
    }
    energy_values = [value for key, value in energy_mean.items() if "guard" in key or "discarded" in key]
    energy_pass = bool(energy_values) and max(energy_values) <= 1.0e-4
    replay = {**replay_checks, "pass": all(replay_checks.values())}
    finite = finite and all(item["reconstruction_error"]["max"] <= 1.0e-5 for item in planes.values())
    report = {
        "schema": REPORT_SCHEMA, "diagnostic_only": True, "status": "v75-carrier-residual-interference-complete",
        "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "v73_admission": str(v73_admission), "v73_report": str(v73_report), "v73_decision": str(v73_decision),
        "v74_admission": str(v74_admission), "v74_report": str(v74_report), "v74_decision": str(v74_decision), "cpu_admission": str(cpu_admission),
        "validation_sample_count": CODE_COUNT, "crossed_output_count": CODE_COUNT * LABEL_COUNT, "batch_size": 1,
        "return_optical_fields": True, "transient_full_state_used": True, "persistent_full_field_saved": False,
        "saved_artifacts": "scalar_statistics_only", "carrier_pass": 1, "residual_pass": 2,
        "carrier_dtype": "complex64", "carrier_persisted": False, "amplitude": amplitude,
        "planes": planes, "aliases": ALIASES, "alias_replay_pass": alias_pass,
        "camera_metrics": camera_metrics, "diagonal_metrics": diagonal_metrics,
        "v74_active_replay": replay, "finite": finite, "energy_records_mean": energy_mean, "energy_pass": energy_pass,
        "parameters_unchanged": before_hash == state_hash(model), "optimizer_created": False, "parameter_update_count": 0,
        "peak_rss_gib": _peak_rss_gib(), "peak_cuda_allocated_gib": peak_allocated, "peak_cuda_reserved_gib": peak_reserved,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(aux["payload"]), "test_split_used": False,
        "teacher_alignment_computed": False, "manual_readability_reused": True, "manual_review": str(v73_manual_review or v73_report),
        "reconstruction_pass": all(item["reconstruction_error"]["max"] <= 1.0e-5 for item in planes.values()),
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "validation_dataset", "v73_admission", "v73_report", "v73_decision", "v74_admission", "v74_report", "v74_decision", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--v73-manual-review", type=Path)
    parser.add_argument("--cpu-admission", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    values = vars(args)
    if args.device == "cpu":
        result = run_cpu_admission(**{key: values[key] for key in ("config", "checkpoint", "validation_dataset", "v73_admission", "v73_report", "v73_decision", "v74_admission", "v74_report", "v74_decision", "output")}, v73_manual_review=args.v73_manual_review)
    else:
        if args.cpu_admission is None:
            raise ValueError("--cpu-admission is required for CUDA")
        result = run_cuda(**{key: values[key] for key in ("config", "checkpoint", "validation_dataset", "v73_admission", "v73_report", "v73_decision", "v74_admission", "v74_report", "v74_decision", "output")}, cpu_admission=args.cpu_admission, device=args.device, v73_manual_review=args.v73_manual_review)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
