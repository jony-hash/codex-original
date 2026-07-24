"""v78 semantic-gated direct-command contract admission and CUDA smoke.

This tool intentionally performs no optical audit or training.  Admission is
generator-only; smoke only checks the finite paired gradient contract.
"""

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
from ldm_project.student.experimental.semantic_command import project_residual_to_basis, semantic_basis
from tools.audit_v73_semantic_support import tensor_hash
from tools.diagnose_v68_transparent_memory import _build_prototype_bank
from tools.v68_b_acceptance_utils import balanced_factor_decomposition

AMPLITUDE = 0.4027685085311532
BACKGROUND_FLOOR = 0.05
SCALE = 0.4913053810596466
CODE_COUNT = 100
LABEL_COUNT = 10


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_hash(module: torch.nn.Module, prefixes: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if prefixes and not key.startswith(prefixes):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _frequency_fraction(delta: torch.Tensor) -> dict[str, float]:
    centered = delta - delta.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered), dim=(-2, -1)).abs().square()
    fy = torch.fft.fftshift(torch.fft.fftfreq(delta.shape[-2], d=2.0e-6)) / 1000.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(delta.shape[-1], d=2.0e-6)) / 1000.0
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low = spectrum[..., radius <= 20.0].sum()
    total = spectrum.sum().clamp_min(1.0e-12)
    return {"low_0_20_cycles_mm_fraction": float((low / total).item())}


def _load_inputs(config: str | Path, checkpoint: str | Path, validation_dataset: str | Path, v77_decision: str | Path):
    cfg = load_training_config(config)
    decision = _read(v77_decision)
    if decision.get("status") != "phase-aware-local-co-design-admissible":
        raise ValueError("v78 requires the final admissible v77 decision")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes = data["codes"].float()
    labels = data["class_labels"].long()
    if tuple(codes.shape) != (CODE_COUNT, 128) or tuple(labels.shape) != (CODE_COUNT,):
        raise ValueError("v78 requires the fixed validation split (100 codes)")
    if set(labels.tolist()) != set(range(LABEL_COUNT)):
        raise ValueError("v78 validation labels must cover 0..9")
    prototypes = _build_prototype_bank(cfg).float().contiguous()
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size)
    load_one_shot_checkpoint(model, payload)
    return cfg, payload, data, codes, labels, prototypes, model


def _admission_metrics(model: OneShotOpticalGenerator, codes: torch.Tensor, prototypes: torch.Tensor) -> dict[str, Any]:
    commands = []
    deltas = []
    coefficients = []
    residuals = []
    bases = []
    base_commands = []
    weighted_means = []
    with torch.no_grad():
        for code_start in range(0, CODE_COUNT, 10):
            code_batch = codes[code_start:code_start + 10]
            code_input = code_batch.repeat_interleave(LABEL_COUNT, dim=0)
            label_input = torch.arange(LABEL_COUNT).repeat(code_batch.shape[0])
            output, diag = model.generator_e.forward_with_diagnostics(code_input, label_input)
            commands.append(output.cpu().reshape(-1, LABEL_COUNT, 1, 200, 200))
            deltas.append(diag["generator_command_delta"].cpu().reshape(-1, LABEL_COUNT, 1, 200, 200))
            coefficients.append(diag["generator_basis_coefficients"].cpu().reshape(-1, LABEL_COUNT, 8))
            residuals.append(diag["generator_code_delta_logits"].cpu().reshape(-1, LABEL_COUNT, 1, 200, 200))
            bases.append(diag["generator_semantic_pattern"].cpu().reshape(-1, LABEL_COUNT, 1, 200, 200))
            base_commands.append(diag["generator_base_command"].cpu().reshape(-1, LABEL_COUNT, 1, 200, 200))
            weighted_means.append(diag["generator_weighted_mean"].cpu().reshape(-1, LABEL_COUNT, 1, 1, 1))
    command = torch.cat(commands, dim=0)
    delta = torch.cat(deltas, dim=0)
    coeff = torch.cat(coefficients, dim=0)
    residual_logits = torch.cat(residuals, dim=0)
    semantic_pattern = torch.cat(bases, dim=0)
    base_command = torch.cat(base_commands, dim=0)
    weighted_mean = torch.cat(weighted_means, dim=0)
    intensity = (BACKGROUND_FLOOR + (1.0 - BACKGROUND_FLOOR) * command).square()
    factors, _ = balanced_factor_decomposition(intensity)
    proto_support = prototypes[:, 0].reshape(1, LABEL_COUNT, 1, 200, 200) >= 0.2
    total_delta = delta.square().sum().clamp_min(1.0e-12)
    support_energy = (delta.square() * proto_support).sum() / total_delta
    freq = _frequency_fraction(delta)
    basis = semantic_basis(200)
    projected, _ = project_residual_to_basis(
        residual_logits.reshape(-1, 1, 200, 200), basis
    )
    # The projected pattern is normalized by the semantic-support equation;
    # compare the emitted command against its exact fixed formula.
    support = torch.sigmoid(model.generator_e.template_logits.detach()).index_select(
        0, torch.arange(LABEL_COUNT)
    ).reshape(1, LABEL_COUNT, 1, 200, 200).expand(CODE_COUNT, -1, -1, -1, -1)
    centered = support * (projected.reshape(CODE_COUNT, LABEL_COUNT, 1, 200, 200) - weighted_mean)
    normalized = centered / centered.flatten(2).abs().amax(2).clamp_min(1.0e-8).view(CODE_COUNT, LABEL_COUNT, 1, 1, 1)
    command_rebuilt = (base_command + AMPLITUDE * normalized).clamp(0.0, 1.0)
    formula_error = float((command_rebuilt - command).abs().max())
    pattern_error = float((normalized - semantic_pattern).abs().max())
    direct_mean = (
        (support * projected.reshape(CODE_COUNT, LABEL_COUNT, 1, 200, 200)).flatten(2).sum(2)
        / support.flatten(2).sum(2).clamp_min(1.0e-8)
    )
    weighted_error = float((direct_mean - weighted_mean.flatten(2).squeeze(-1)).abs().max())
    expected_projected = torch.einsum("bk,khw->bhw", coeff.reshape(-1, 8), basis)[:, None]
    basis_error = float((projected - expected_projected).abs().max())
    return {
        "physical_slm_intensity_code_factor": factors["code_fraction"],
        "physical_slm_intensity_code_ss": factors["total_energy"] * factors["code_fraction"],
        "support_energy_fraction": float(support_energy),
        "low_frequency_energy_fraction": freq["low_0_20_cycles_mm_fraction"],
        "one_clipping_fraction": float(((command <= 0.0) | (command >= 1.0)).float().mean()),
        "weighted_mean_formula_max_abs_error": weighted_error,
        "basis_reconstruction_max_abs_error": basis_error,
        "command_formula_max_abs_error": formula_error,
        "semantic_pattern_max_abs_error": pattern_error,
        "finite": bool(torch.isfinite(command).all() and torch.isfinite(intensity).all() and torch.isfinite(coeff).all()),
        "coefficients_hash": tensor_hash(coeff),
        "command_hash": tensor_hash(command_rebuilt),
    }


def run_admission(*, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path, v77_decision: str | Path, output: str | Path) -> dict[str, Any]:
    cfg, payload, data, codes, labels, prototypes, model = _load_inputs(config, checkpoint, validation_dataset, v77_decision)
    before = state_hash(model)
    metrics = _admission_metrics(model, codes, prototypes)
    checkpoint_keys = set(payload.get("model_state_dict", {}))
    model_keys = set(model.state_dict())
    state_compatible = checkpoint_keys == model_keys
    checks = {
        "v77_status": _read(v77_decision).get("status") == "phase-aware-local-co-design-admissible",
        "validation_contract": tuple(codes.shape) == (100, 128) and set(labels.tolist()) == set(range(10)),
        "checkpoint_contract": checkpoint_boundary_contract(payload) == "ideal_transparent_thin_phase_v2",
        "checkpoint_state_compatible": state_compatible,
        "physical_intensity_code_factor": metrics["physical_slm_intensity_code_factor"] >= 0.25,
        "support_energy": metrics["support_energy_fraction"] >= 0.95,
        "low_frequency_energy": metrics["low_frequency_energy_fraction"] >= 0.99,
        "clipping": metrics["one_clipping_fraction"] <= 0.08,
        "formula": metrics["weighted_mean_formula_max_abs_error"] <= 1.0e-6 and metrics["basis_reconstruction_max_abs_error"] <= 1.0e-6 and metrics["semantic_pattern_max_abs_error"] <= 1.0e-6 and metrics["command_formula_max_abs_error"] <= 1.0e-6,
        "finite": metrics["finite"],
        "parameters_unchanged": state_hash(model) == before,
        "optimizer_created": False,
        "calibration_or_sweep": False,
    }
    status = "v78-cpu-admission-passed" if all(checks.values()) else "v78-cpu-admission-failed"
    report = {
        "schema": "v78_semantic_command_admission_v1", "status": status, "diagnostic_only": True,
        "config": str(config), "checkpoint": str(checkpoint), "validation_dataset": str(validation_dataset),
        "validation_sample_count": 100, "crossed_output_count": 1000, "batch_size": 1,
        "test_split_used": False, "return_optical_fields": False, "persistent_full_field_saved": False,
        "semantic_amplitude": AMPLITUDE, "locked_exposure_scale": SCALE,
        "checkpoint_model_hash": before, "codes_hash": tensor_hash(codes), "prototype_hash": tensor_hash(prototypes),
        "metrics": metrics, "checks": checks, "no_optimizer_or_update": True,
        "test_allowed": False, "formal_promotion_allowed": False, "cuda_smoke_required": bool(status == "v78-cpu-admission-passed"),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_smoke(*, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path, admission: str | Path, output: str | Path, device: str = "cuda") -> dict[str, Any]:
    adm = _read(admission)
    if adm.get("status") != "v78-cpu-admission-passed":
        raise ValueError("v78 smoke requires passing CPU admission")
    cfg, payload, data, codes, labels, prototypes, model = _load_inputs(config, checkpoint, validation_dataset, adm.get("v77_decision", validation_dataset) if False else _v77_path_from_admission(adm))
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("v78 smoke requires CUDA")
    model = model.to(dev).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.generator_e.residual_net.requires_grad_(True)
    model.encoder.phase.requires_grad_(True)
    model.decoder.phase.requires_grad_(True)
    frozen_before = state_hash(model, ("generator_e.template_logits", "generator_e.label_embedding."))
    idx = torch.arange(4)
    target = data.get("teacher_camera", data.get("targets"))[idx].to(dev).float()
    finite = True
    for i in idx.tolist():
        out = model(codes[i:i+1].to(dev), labels[i:i+1].to(dev), return_optical_fields=False)
        pred = out.get("physical_camera_intensity_raw", out["x_hat"])
        loss = (pred - target[i:i+1]).square().mean()
        finite = finite and bool(torch.isfinite(loss))
        loss.backward()
    residual_grads = [p.grad for p in model.generator_e.residual_net.parameters() if p.grad is not None]
    phase_grads = [model.encoder.phase.grad, model.decoder.phase.grad]
    nonzero = all(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0.0 for g in residual_grads + phase_grads)
    report = {"schema": "v78_semantic_command_smoke_v1", "status": "v78-semantic-command-smoke-passed" if finite and nonzero else "v78-residual-phase-gradient-invalid", "diagnostic_only": True, "sample_count": 4, "finite": finite, "residual_phase_gradients_nonzero": nonzero, "template_embedding_frozen": state_hash(model, ("generator_e.template_logits", "generator_e.label_embedding.")) == frozen_before, "optimizer_created": False, "parameter_update_count": 0, "return_optical_fields": False, "persistent_full_field_saved": False, "cuda_allocated_gib": float(torch.cuda.memory_allocated(dev) / 1024**3) if dev.type == "cuda" else 0.0, "cuda_reserved_gib": float(torch.cuda.memory_reserved(dev) / 1024**3) if dev.type == "cuda" else 0.0}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _v77_path_from_admission(admission: dict[str, Any]) -> str:
    # The CPU report stores the chain only as a provenance flag.  The fixed
    # repository path is resolved explicitly by the CLI instead of guessing.
    return admission.get("v77_decision_path", "outputs/20260720_v77/runs/isolation/v77_joint_decision_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("admission", "smoke"), required=True)
    parser.add_argument("--config", default="configs/train_v78_semantic_command_contract.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--v77-decision", default="outputs/20260720_v77/runs/isolation/v77_joint_decision_v1.json")
    parser.add_argument("--admission", default="outputs/20260721_v78/runs/admission/v78_semantic_command_admission_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "admission":
        run_admission(config=args.config, checkpoint=args.checkpoint, validation_dataset=args.validation_dataset, v77_decision=args.v77_decision, output=args.output)
    else:
        run_smoke(config=args.config, checkpoint=args.checkpoint, validation_dataset=args.validation_dataset, admission=args.admission, output=args.output, device=args.device)


if __name__ == "__main__":
    main()
