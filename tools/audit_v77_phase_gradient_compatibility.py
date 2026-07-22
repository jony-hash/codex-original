"""Read-only v77 phase-aware D2NN local co-design admission/audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from tools.audit_v73_semantic_support import (
    SCALE,
    semantic_support_pattern,
    state_hash,
    tensor_hash,
)
from tools.audit_v73_semantic_support import _build_prototype_bank, _forward_with_support

REPORT_SCHEMA = "v77_phase_gradient_compatibility_audit_v1"
ADMISSION_SCHEMA = "v77_phase_gradient_admission_v1"
SMOKE_SCHEMA = "v77_phase_gradient_smoke_v1"
EXPECTED_AMPLITUDE = 0.4027685085311532
CODE_COUNT = 100
LABEL_COUNT = 10
MIN_PROJECTED_RATIO = 0.25
GRADIENT_TOLERANCE = 1.0e-12
CODE_SS_REFERENCE = 53141.618758854296
READABILITY_DICE_WEIGHT = 0.30


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0 / 1024**3


def selective_state_hash(model: torch.nn.Module, *, include_phase: bool) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        is_phase = key in {"encoder.phase", "decoder.phase"}
        if is_phase != include_phase:
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def balanced_code_stats(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return exact balanced code-main SS statistics for [C,L,...] values."""
    if values.ndim < 3 or values.shape[0] != CODE_COUNT or values.shape[1] != LABEL_COUNT:
        raise ValueError("expected [100,10,...] balanced values")
    values = values.float()
    code_mean = values.mean(dim=1)
    grand = values.mean(dim=(0, 1))
    centered = code_mean - grand
    code_ss = float(LABEL_COUNT) * centered.square().sum()
    return {
        "code_mean": code_mean,
        "grand_mean": grand,
        "weights": 2.0 * centered,
        "code_effect_ss": code_ss,
    }


def streaming_code_weights(
    code_sums: torch.Tensor, total_sum: torch.Tensor, *, label_count: int = LABEL_COUNT, sample_count: int = CODE_COUNT
) -> dict[str, torch.Tensor]:
    """Build ANOVA VJP weights without retaining individual fields."""
    if code_sums.shape[0] != sample_count or total_sum.shape != code_sums.shape[1:]:
        raise ValueError("invalid streaming code sufficient statistics")
    code_mean = code_sums / float(label_count)
    grand = total_sum / float(sample_count * label_count)
    centered = code_mean - grand
    return {
        "code_mean": code_mean,
        "grand_mean": grand,
        "weights": 2.0 * centered,
        "code_effect_ss": float(label_count) * centered.square().sum(),
    }


def fixed_readability_loss(raw: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    """Locked-exposure prototype loss used only for local gradient compatibility."""
    prediction = (raw / SCALE).clamp(0.0, 1.0)
    target = prototype.to(device=prediction.device, dtype=prediction.dtype)
    mse = (prediction - target).square().mean()
    dice = 2.0 * (prediction * target).sum() / (prediction.sum() + target.sum()).clamp_min(1.0e-8)
    return mse + READABILITY_DICE_WEIGHT * (1.0 - dice)


def project_code_ascent(code_grad: torch.Tensor, readability_grad: torch.Tensor) -> dict[str, float | bool]:
    """Project code-SS ascent onto the half-space with non-increasing readability."""
    code = code_grad.detach().double().flatten()
    readability = readability_grad.detach().double().flatten()
    code_norm = float(code.norm())
    readability_norm_sq = float(readability.square().sum())
    if code_norm <= GRADIENT_TOLERANCE:
        return {
            "finite": bool(torch.isfinite(code).all() and torch.isfinite(readability).all()),
            "code_gradient_norm": code_norm,
            "readability_gradient_norm": float(readability.norm()),
            "cosine": 0.0,
            "projected_code_drive_ratio": 0.0,
            "first_order_code_drive": 0.0,
            "first_order_readability_change": 0.0,
            "code_gradient_suppressed": True,
        }
    dot = float(torch.dot(readability, code))
    if readability_norm_sq <= GRADIENT_TOLERANCE or dot <= 0.0:
        projected = code
    else:
        projected = code - (dot / readability_norm_sq) * readability
    projected_norm = float(projected.norm())
    cosine = float(torch.dot(code, readability) / (code.norm() * readability.norm()).clamp_min(1.0e-12))
    return {
        "finite": bool(torch.isfinite(code).all() and torch.isfinite(readability).all() and torch.isfinite(projected).all()),
        "code_gradient_norm": code_norm,
        "readability_gradient_norm": float(readability.norm()),
        "code_gradient_rms": float(code.square().mean().sqrt()),
        "readability_gradient_rms": float(readability.square().mean().sqrt()),
        "cosine": cosine,
        "projected_code_drive_ratio": projected_norm / code_norm,
        "first_order_code_drive": float(torch.dot(code, projected)),
        "first_order_readability_change": float(torch.dot(readability, projected)),
        "code_gradient_suppressed": False,
    }


def _validate_fixed_inputs(
    *, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path,
    v73_admission: str | Path, v73_report: str | Path, v73_decision: str | Path,
    v73_manual_review: str | Path, v75_admission: str | Path, v75_report: str | Path,
    v75_decision: str | Path, v76_admission: str | Path, v76_report: str | Path,
    v76_decision: str | Path,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    cfg = load_training_config(config)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes = data.get("codes", torch.empty(0)).float()
    labels = data.get("class_labels", torch.empty(0, dtype=torch.long)).long()
    prototypes = _build_prototype_bank(cfg).float().contiguous()
    reports = {"v73_admission": _read(v73_admission), "v73_report": _read(v73_report), "v73_decision": _read(v73_decision),
               "v73_manual": _read(v73_manual_review), "v75_admission": _read(v75_admission), "v75_report": _read(v75_report),
               "v75_decision": _read(v75_decision), "v76_admission": _read(v76_admission), "v76_report": _read(v76_report),
               "v76_decision": _read(v76_decision)}
    checks = {
        "validation_shape": tuple(codes.shape) == (CODE_COUNT, 128) and tuple(labels.shape) == (CODE_COUNT,),
        "all_labels": set(labels.tolist()) == set(range(LABEL_COUNT)),
        "v73_admission": reports["v73_admission"].get("status") == "v73-cpu-admission-passed",
        "v73_report": reports["v73_report"].get("status") == "v73-semantic-support-audit-complete",
        "v73_decision": reports["v73_decision"].get("status") == "semantic-support-code-factor-attenuated",
        "v75_chain": reports["v75_admission"].get("status") == "v75-interference-admission-passed" and reports["v75_report"].get("status") == "v75-carrier-residual-interference-complete" and reports["v75_decision"].get("status") == "carrier-residual-quadrature-rotation-confirmed",
        "v76_chain": reports["v76_admission"].get("status") == "v76-dual-quadrature-admission-passed" and reports["v76_report"].get("status") == "v76-dual-quadrature-readout-complete" and reports["v76_decision"].get("status") == "dual-quadrature-dynamic-range-tradeoff",
        "manual_readability": sorted(int(x) for x in reports["v73_manual"].get("readable_labels", [])) == list(range(LABEL_COUNT)),
        "amplitude": abs(float(reports["v73_report"].get("calibration", {}).get("amplitude", -1.0)) - EXPECTED_AMPLITUDE) <= 1.0e-12,
        "exposure": abs(float(reports["v73_report"].get("exposure_scale", -1.0)) - SCALE) <= 1.0e-12 and abs(float(reports["v76_report"].get("locked_exposure_scale", -1.0)) - SCALE) <= 1.0e-12,
        "no_test": all(not bool(reports[key].get("test_split_used", False)) for key in ("v73_report", "v75_report", "v76_report")),
        "v76_replay": bool(reports["v76_report"].get("replay_pass")) and bool(reports["v76_report"].get("identity_pass")),
        "v76_no_mutation": bool(reports["v76_report"].get("parameters_unchanged")) and reports["v76_report"].get("parameter_update_count") == 0 and not bool(reports["v76_report"].get("optimizer_created")),
        "complex_field_available": bool(reports["v76_report"].get("carrier_dtype") == "complex64"),
        "finite_data": bool(torch.isfinite(codes).all() and torch.isfinite(prototypes).all()),
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size)
    load_one_shot_checkpoint(model, payload)
    model_hash = state_hash(model)
    expected_hashes = {reports["v73_admission"].get("checkpoint_model_hash"), reports["v75_admission"].get("checkpoint_model_hash"), reports["v73_admission"].get("v72_baseline_hash")}
    checks["checkpoint_hash"] = model_hash in expected_hashes and None not in expected_hashes
    checks["checkpoint_contract"] = checkpoint_boundary_contract(payload) == "ideal_transparent_thin_phase_v2"
    checks["prototype_hash"] = bool(tensor_hash(prototypes))
    if not all(checks.values()):
        raise ValueError("v77 fixed-input admission failed: " + ", ".join(k for k, ok in checks.items() if not ok))
    pattern, support = semantic_support_pattern(codes, prototypes)
    # The prototype support is identical across code rows; forward overrides
    # consume the canonical [label,1,H,W] view, not the repeated [code,label,...]
    # construction used for admission bookkeeping.
    return cfg, codes, labels, prototypes, pattern, support[0], {"checks": checks, "model_hash": model_hash, "reports": reports}


def run_cpu_admission(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    cfg, codes, labels, prototypes, pattern, support, aux = _validate_fixed_inputs(**kwargs)
    report = {
        "schema": ADMISSION_SCHEMA, "diagnostic_only": True, "status": "v77-phase-gradient-admission-passed",
        "checkpoint": str(kwargs["checkpoint"]), "validation_dataset": str(kwargs["validation_dataset"]),
        "validation_sample_count": CODE_COUNT, "crossed_output_count": CODE_COUNT * LABEL_COUNT, "batch_size": 1,
        "test_split_used": False, "return_optical_fields": False, "semantic_support_amplitude": EXPECTED_AMPLITUDE,
        "locked_exposure_scale": SCALE, "codes_hash": tensor_hash(codes), "prototype_hash": tensor_hash(prototypes),
        "checkpoint_model_hash": aux["model_hash"], "checks": aux["checks"],
        "phase_groups": ["encoder_layer_1", "encoder_layer_2", "decoder_layer_1", "decoder_layer_2"],
        "receiver_branch": "closed_after_v76_dynamic_range_tradeoff", "learned_code_transport_proven": False,
        "cuda_smoke_required": True, "diagnostic_only_guarantees": {"optimizer_created": False, "parameter_update_count": 0, "full_field_saved": False},
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _phase_parameters(model: OneShotOpticalGenerator) -> dict[str, torch.Tensor]:
    return {
        "encoder_layer_1": model.encoder.phase[0], "encoder_layer_2": model.encoder.phase[1],
        "decoder_layer_1": model.decoder.phase[0], "decoder_layer_2": model.decoder.phase[1],
    }


def _select_code_indices(codes: torch.Tensor, code_count: int | None) -> list[int]:
    if code_count is None or code_count == CODE_COUNT:
        return list(range(CODE_COUNT))
    if code_count <= 0 or code_count > CODE_COUNT:
        raise ValueError("code-count must be in [1,100]")
    return torch.linspace(0, CODE_COUNT - 1, code_count).round().long().tolist()


def run_cuda(*, output: str | Path, cpu_admission: str | Path, smoke: bool, device: str = "cuda", code_count: int | None = None, **kwargs: Any) -> dict[str, Any]:
    admission = _read(cpu_admission)
    if admission.get("status") != "v77-phase-gradient-admission-passed":
        raise ValueError("v77 CUDA audit requires passing CPU admission")
    cfg, codes, labels, prototypes, pattern, support, aux = _validate_fixed_inputs(**kwargs)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v77 requires CUDA")
    payload = torch.load(kwargs["checkpoint"], map_location="cpu", weights_only=False)
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).to(dev).eval()
    load_one_shot_checkpoint(model, payload)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phases = _phase_parameters(model)
    phase_bases = (model.encoder.phase, model.decoder.phase)
    for parameter in phase_bases:
        parameter.requires_grad_(True)
    before_full = state_hash(model)
    before_phase = {name: tensor_hash(value) for name, value in phases.items()}
    before_non_phase = selective_state_hash(model, include_phase=False)
    indices = _select_code_indices(codes, 4 if smoke else code_count)
    # First pass: only sufficient statistics, never individual fields.
    code_sums = torch.zeros((len(indices), 200, 200), dtype=torch.float64)
    total_sum = torch.zeros((200, 200), dtype=torch.float64)
    finite = True
    with torch.no_grad():
        for local_index, code_index in enumerate(indices):
            for label_value in range(LABEL_COUNT):
                out = _forward_with_support(model, codes[code_index:code_index + 1].to(dev), torch.tensor([label_value], device=dev), support, EXPECTED_AMPLITUDE)
                raw = out["physical_camera_intensity_raw"].detach().float().cpu().squeeze(0).squeeze(0).double()
                finite = finite and bool(torch.isfinite(raw).all())
                code_sums[local_index] += raw
                total_sum += raw
                del out
    stats = streaming_code_weights(code_sums, total_sum, label_count=LABEL_COUNT, sample_count=len(indices))
    code_grad_acc = {name: torch.zeros_like(value, device="cpu") for name, value in phases.items()}
    read_grad_acc = {name: torch.zeros_like(value, device="cpu") for name, value in phases.items()}
    code_ss_samples = 0
    read_losses: list[float] = []
    with torch.enable_grad():
        for local_index, code_index in enumerate(indices):
            for label_value in range(LABEL_COUNT):
                out = _forward_with_support(model, codes[code_index:code_index + 1].to(dev), torch.tensor([label_value], device=dev), support, EXPECTED_AMPLITUDE)
                raw = out["physical_camera_intensity_raw"]
                weight = stats["weights"][local_index].to(device=dev, dtype=raw.dtype).view(1, 1, 200, 200)
                code_scalar = (raw * weight).sum()
                prototype = prototypes[label_value:label_value + 1].to(dev)
                read_loss = fixed_readability_loss(raw, prototype)
                code_grads = torch.autograd.grad(code_scalar, phase_bases, retain_graph=True, allow_unused=True)
                read_grads = torch.autograd.grad(read_loss, phase_bases, retain_graph=False, allow_unused=True)
                phase_slices = {
                    "encoder_layer_1": (0, 0), "encoder_layer_2": (0, 1),
                    "decoder_layer_1": (1, 0), "decoder_layer_2": (1, 1),
                }
                for name, (base_index, layer_index) in phase_slices.items():
                    parameter = phases[name]
                    code_grad = code_grads[base_index]
                    read_grad = read_grads[base_index]
                    cg = torch.zeros_like(parameter) if code_grad is None else code_grad[layer_index].detach()
                    rg = torch.zeros_like(parameter) if read_grad is None else read_grad[layer_index].detach()
                    finite = finite and bool(torch.isfinite(cg).all() and torch.isfinite(rg).all())
                    code_grad_acc[name] += cg.cpu()
                    read_grad_acc[name] += rg.cpu()
                read_losses.append(float(read_loss.detach().cpu()))
                code_ss_samples += 1
                del out, code_scalar, read_loss, code_grads, read_grads
    metrics = {name: project_code_ascent(code_grad_acc[name], read_grad_acc[name]) for name in phases}
    all_code = torch.cat([code_grad_acc[name].double().flatten() for name in phases])
    all_read = torch.cat([read_grad_acc[name].double().flatten() for name in phases])
    metrics["joint"] = project_code_ascent(all_code, all_read)
    after_full = state_hash(model)
    after_phase = {name: tensor_hash(value) for name, value in phases.items()}
    after_non_phase = selective_state_hash(model, include_phase=False)
    phase_gradient_nonzero = all(metrics[name]["code_gradient_norm"] > GRADIENT_TOLERANCE and not metrics[name]["code_gradient_suppressed"] for name in phases)
    result = {
        "schema": SMOKE_SCHEMA if smoke else REPORT_SCHEMA, "diagnostic_only": True,
        "status": "v77-phase-gradient-smoke-passed" if smoke and finite and phase_gradient_nonzero and before_full == after_full else "v77-phase-gradient-compatibility-complete",
        "checkpoint": str(kwargs["checkpoint"]), "validation_dataset": str(kwargs["validation_dataset"]),
        "cpu_admission": str(cpu_admission), "validation_sample_count": len(indices), "crossed_output_count": code_ss_samples,
        "code_indices": indices, "batch_size": 1, "return_optical_fields": False, "test_split_used": False,
        "semantic_support_amplitude": EXPECTED_AMPLITUDE, "locked_exposure_scale": SCALE,
        "streaming_code_effect_ss": float(stats["code_effect_ss"]), "readability_loss_mean": sum(read_losses) / max(len(read_losses), 1),
        "gradient_metrics": metrics, "phase_gradient_nonzero": phase_gradient_nonzero, "finite": bool(finite),
        "phase_hash_before": before_phase, "phase_hash_after": after_phase, "non_phase_hash_before": before_non_phase, "non_phase_hash_after": after_non_phase,
        "parameters_unchanged": before_full == after_full and before_phase == after_phase and before_non_phase == after_non_phase,
        "optimizer_created": False, "parameter_update_count": 0, "full_field_saved": False, "persistent_full_field_saved": False, "saved_artifacts": "scalar_gradient_statistics_only",
        "peak_rss_gib": _peak_rss_gib(), "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0.0,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if dev.type == "cuda" else 0.0,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload), "teacher_alignment_computed": False,
        "coherent_receiver_branch_closed": True, "learned_code_transport_proven": False,
        "test_allowed": False, "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
    }
    if smoke and (not finite or not phase_gradient_nonzero or not result["parameters_unchanged"]):
        result["status"] = "harness-invalid"
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("admission", "smoke", "audit"), required=True)
    for name in ("config", "checkpoint", "validation_dataset", "v73_admission", "v73_report", "v73_decision", "v73_manual_review", "v75_admission", "v75_report", "v75_decision", "v76_admission", "v76_report", "v76_decision", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--cpu-admission", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-count", type=int, default=100)
    args = parser.parse_args()
    values = vars(args)
    mode = values.pop("mode")
    output = values.pop("output")
    device = values.pop("device")
    code_count = values.pop("code_count")
    cpu_admission = values.pop("cpu_admission")
    if mode == "admission":
        result = run_cpu_admission(output=output, **values)
    else:
        if cpu_admission is None:
            raise SystemExit("--cpu-admission is required for smoke/audit")
        result = run_cuda(output=output, cpu_admission=cpu_admission, smoke=mode == "smoke", device=device, code_count=code_count, **values)
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
