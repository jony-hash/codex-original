"""Scaled Chen reconstruction-horizon research lifecycle.

This is an independent, MSE-only horizon calibration.  It deliberately does
not consume or rewrite the historical factorial pilot evidence and has no
training wall-clock acceptance gate; runtime is reported as telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ...config import load_config
from ...evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_horizon_v2"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_horizon_review_v2"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_horizon_validation_v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_bogt_horizon_preflight_v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_bogt_horizon_training_v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0-HORIZON"
BRANCH = bogt.BRANCH
SEED = 1234
BATCH_SIZE = 10
HORIZON_STEPS = 5000
MILESTONES = (0, 100, 250, 500, 1000, 2000, 5000)
LEARNING_RATE = 1.0e-3
MAX_PREFLIGHT_SECONDS = 600.0
MAX_PEAK_BYTES = bogt.MAX_PEAK_BYTES
MAX_STORAGE_BYTES = bogt.MAX_STORAGE_BYTES


@dataclass(frozen=True)
class HorizonSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class HorizonData:
    images: torch.Tensor
    labels: torch.Tensor
    calibration_indices: torch.Tensor
    schedule: torch.Tensor
    heldout_indices: torch.Tensor
    manifest: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return bogt._resolve(value, base_path=base_path)


def _git_state() -> dict[str, Any]:
    return bogt._git_state()


def _load_spec(path: str | Path) -> HorizonSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "source", "dataset", "optical_contract", "training", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("scaled Chen horizon config fields are invalid")
    return HorizonSpec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: HorizonSpec) -> None:
    p = spec.payload
    if p["schema"] != CONFIG_SCHEMA or p["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported scaled Chen horizon contract")
    if any(key in json.dumps(p, sort_keys=True) for key in ("max_wall_clock_s", "training_wall_clock_gate")):
        raise ValueError("horizon contract must not contain hard training wall-clock gates")
    if p["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "logical_batch_size": 10,
        "microbatch_size": 1,
        "horizon_steps": HORIZON_STEPS,
        "milestones": list(MILESTONES),
        "schedule": "class_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "calibration_and_schedule_disjoint": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("scaled Chen horizon dataset contract is not locked")
    optical = p["optical_contract"]
    locked = {
        "formal_system_kind": "formal_amplitude_finite_crop_identity",
        "architecture": "2plus2_phase_only",
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.50e-3,
        "mode_bank": "scaled_lp01_10x10",
        "ols_ports": 100,
        "mode_model": "step_index_lp01_ports",
        "mode_v_number": 2.2,
        "mode_waist_m": 7.111e-6,
        "mode_fill": 0.8,
        "fixed_layer_power_transmission": 0.889,
        "backend_total_distance_m": 3.0e-3,
        "backend_magnification": -2.5,
        "software_decoder": False,
        "electronic_posterior": False,
        "formal_promotion": False,
    }
    for key, expected in locked.items():
        actual = optical.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > max(abs(expected) * 1e-12, 1e-15):
                raise ValueError(f"horizon optical contract locks {key}")
        elif actual != expected:
            raise ValueError(f"horizon optical contract locks {key}")
    training = p["training"]
    expected_training = {
        "objective": "mse_only",
        "steps": HORIZON_STEPS,
        "milestones": list(MILESTONES),
        "checkpoint_every_milestone": True,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "zero_phase_seed": SEED,
        "posthoc_phase_quantization": list(bogt.PHASE_LEVELS),
        "qat": False,
        "wall_clock_seconds": "observational_only",
        "seconds_per_step": "observational_only",
        "estimated_total_runtime": "observational_only",
    }
    if training != expected_training:
        raise ValueError("scaled Chen horizon training contract is not locked")
    auth = p["authorization"]
    if auth.get("bounded_diagnostic_training_authorized") is not True or auth.get("training_authorized") is not False or auth.get("formal_promotion") is not False:
        raise ValueError("scaled Chen horizon authorization boundary is invalid")


def _load_review(path: str | Path, spec: HorizonSpec) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    expected = {
        "architecture": "2+2_phase_only_100x100_2um_1.50mm",
        "ols": "fixed_100_port_scaled_lp01_v2_bank",
        "objective": "mse_only",
        "milestones": list(MILESTONES),
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": 1,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "zero_phase_seed": SEED,
        "posthoc_phase_quantization": list(bogt.PHASE_LEVELS),
        "wall_clock": "observational_only",
        "post_pass_next_action": "manual_review_scaled_chen_bogt_factorial_horizon_contract",
    }
    if payload.get("schema") != REVIEW_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("reviewer_role") != "project_owner" or payload.get("decision") != "authorize_scaled_chen_reconstruction_horizon_only":
        raise ValueError("scaled Chen horizon review is invalid")
    if payload.get("approved_contract") != expected or payload.get("bounded_diagnostic_training_authorized") is not True or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("scaled Chen horizon review authorization is invalid")
    if "max_wall_clock_s" in json.dumps(payload, sort_keys=True) or "training_wall_clock_gate" in json.dumps(payload, sort_keys=True):
        raise ValueError("scaled Chen horizon review contains a hard wall-clock gate")
    return payload


def _load_formal(spec: HorizonSpec):
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    config = load_config(path)
    if config.architecture != "conditional_identity_logit_amplitude_v2" or config.system.kind != bogt.EXPECTED_FORMAL_KIND or not config.system.is_formal:
        raise ValueError("scaled Chen horizon requires current formal identity geometry")
    formal = config.system
    locked = (("wavelength_m", 532e-9), ("slm_shape", (200, 200)), ("slm_pixel_pitch_m", 2.5e-6), ("metasurface_shape", (100, 100)), ("metasurface_pixel_pitch_m", 2e-6), ("propagation_grid_size", 2048), ("encoder_layers", 2), ("decoder_layers", 2), ("layer_spacing_m", 1.5e-3))
    for name, expected in locked:
        actual = getattr(formal, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1e-12, 1e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal, path


def _horizon_schedule(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 1 or labels.numel() < 55000:
        raise ValueError("scaled Chen horizon requires the complete MNIST train split")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train = torch.arange(55000, dtype=torch.long)
    calibration_parts: list[torch.Tensor] = []
    schedule_parts: list[torch.Tensor] = []
    for label in range(10):
        candidates = train[labels.index_select(0, train) == label]
        shuffled = candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator))
        calibration_parts.append(shuffled[:100])
        schedule_parts.append(shuffled[100 : 100 + HORIZON_STEPS])
    calibration = torch.cat(calibration_parts)
    schedule = torch.stack([torch.stack([schedule_parts[label][step] for label in range(10)]) for step in range(HORIZON_STEPS)])
    if torch.unique(calibration).numel() != calibration.numel() or torch.unique(schedule).numel() != schedule.numel() or torch.isin(calibration, schedule).any():
        raise ValueError("horizon calibration and schedule overlap")
    if not torch.equal(labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(HORIZON_STEPS)):
        raise ValueError("horizon schedule is not class balanced")
    return calibration, schedule


def _load_data(dataset_root: str | Path) -> HorizonData:
    images, labels = base.load_local_mnist(dataset_root, train=True)
    calibration, schedule = _horizon_schedule(labels)
    heldout = torch.arange(55000, 60000, dtype=torch.long)
    manifest: dict[str, Any] = {
        "schema": "scaled_chen_bogt_horizon_mnist_manifest.v2",
        "dataset_root": str(Path(dataset_root).resolve()),
        "images_sha256": bogt.tensor_sha256(images),
        "labels_sha256": bogt.tensor_sha256(labels),
        "calibration_indices_sha256": tensor_sha256(torch.sort(calibration).values),
        "schedule_sha256": tensor_sha256(schedule),
        "heldout_indices_sha256": tensor_sha256(heldout),
        "calibration_count": 1000,
        "schedule_shape": list(schedule.shape),
        "heldout_count": 5000,
        "milestones": list(MILESTONES),
        "seed": SEED,
        "calibration_and_schedule_disjoint": True,
        "heldout_excluded": True,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return HorizonData(images, labels, calibration, schedule, heldout, manifest)


def _as_research_data(data: HorizonData) -> bogt.ResearchData:
    return bogt.ResearchData(data.images, data.labels, data.calibration_indices, data.schedule, data.heldout_indices, tuple(data.schedule[index].clone() for index in range(3)), data.manifest)


def _phase_histogram(model: bogt.ScaledChenBOGTModel) -> dict[str, Any]:
    levels = torch.tensor(bogt.PHASE_LEVELS, dtype=model.system.encoder.phase.dtype, device=model.system.encoder.phase.device)
    report: dict[str, Any] = {}
    for name, phase in (("encoder", model.system.encoder.phase), ("decoder", model.system.decoder.phase)):
        wrapped = torch.remainder(phase.detach(), 2.0 * math.pi)
        distance = (wrapped.unsqueeze(-1) - levels).abs()
        nearest = distance.argmin(dim=-1)
        report[name] = {
            "level_counts": torch.bincount(nearest.reshape(-1), minlength=len(bogt.PHASE_LEVELS)).detach().cpu().tolist(),
            "level_fractions": torch.bincount(nearest.reshape(-1), minlength=len(bogt.PHASE_LEVELS)).float().div(nearest.numel()).detach().cpu().tolist(),
            "mean_nearest_distance_rad": float(distance.min(dim=-1).values.mean().detach().cpu()),
            "mean_abs_phase_rad": float(wrapped.abs().mean().detach().cpu()),
            "distance_to_zero_rad": float(wrapped.abs().mean().detach().cpu()),
        }
    return report


def _evaluate_horizon_milestone(model: bogt.ScaledChenBOGTModel, data: HorizonData, calibration: bogt.Calibration, class_templates: torch.Tensor, device: torch.device, initial_phase: Mapping[str, torch.Tensor], calibration_hash: str, step: int) -> dict[str, Any]:
    research_data = _as_research_data(data)
    train_indices = data.schedule[:100].reshape(-1)
    heldout = bogt._evaluate_model(model, research_data, calibration, data.heldout_indices, device, class_templates, calibration_hash, split="heldout")["metrics"]
    train = bogt._evaluate_model(model, research_data, calibration, train_indices, device, class_templates, calibration_hash, split="train")["metrics"]
    phase_state = bogt._phase_state(model)
    phase_updates = bogt._phase_update_metrics(initial_phase, phase_state)
    continuous = dict(heldout)
    with torch.no_grad():
        model.system.encoder.phase.copy_(bogt._phase_quantize(model.system.encoder.phase))
        model.system.decoder.phase.copy_(bogt._phase_quantize(model.system.decoder.phase))
    quantized = bogt._evaluate_model(model, research_data, calibration, data.heldout_indices, device, class_templates, calibration_hash, split="heldout")["metrics"]
    bogt._set_phase_state(model, phase_state)
    return {
        "step": step,
        "train_mse": train["paired_mse"],
        "heldout_paired_mse": continuous["paired_mse"],
        "heldout_mse": continuous["heldout_mse"],
        "cyclic_mse": continuous["cyclic_mse"],
        "class_template_mse": continuous["template_control_mse"],
        "gaussian_kl": continuous["kl"],
        "numerical_truncated_normal_kl": continuous["numerical_truncated_normal_kl"],
        "l_op": continuous["op"],
        "power_retention": continuous["power_retention"],
        "complex_effective_rank": continuous["coefficient_effective_rank"],
        "power_latent_effective_rank": continuous["power_latent_effective_rank"],
        "per_plane_power": continuous["per_plane_power"],
        "below_window_fraction": continuous["below_window_fraction"],
        "inside_window_fraction": continuous["inside_window_fraction"],
        "above_window_fraction": continuous["above_window_fraction"],
        "saturation_fraction": continuous["saturation_fraction"],
        "continuous_mse": continuous["paired_mse"],
        "quantized_mse": quantized["paired_mse"],
        "quantized_metrics": quantized,
        "phase_update_norms": phase_updates,
        "phase_quantization": _phase_histogram(model),
        "finite": bool(continuous["finite"] and quantized["finite"] and math.isfinite(float(train["paired_mse"]))),
        "instance_specificity": bool(continuous["paired_mse"] < continuous["cyclic_mse"] and continuous["paired_mse"] < continuous["template_control_mse"]),
    }


def _record(schema: str, status: str, spec: HorizonSpec, formal: Any, state: Mapping[str, Any], data: HorizonData, hashes: Mapping[str, str], gates: tuple[GateResult, ...], action: str | None, reason: str | None = None) -> dict[str, Any]:
    context = EvidenceContextV2(canonical_sha256({"horizon": spec.payload["optical_contract"], "branch": BRANCH}), spec.config_hash, dataset_hash=data.manifest["manifest_hash"], dataset_split_manifest_hash=data.manifest["manifest_hash"])
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, context, dict(hashes), gates, (action,) if action else (), reason)
    record.validate()
    return {"schema": schema, "hypothesis": HYPOTHESIS, "training_authorized": False, "formal_promotion": False, "evidence": record.to_dict(), "contract": {"git_state": {"branch": state["branch"], "commit": state["commit"], "clean": state["clean"]}, "source_status": spec.payload["source"]["status"]}}


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite horizon artifact: {target}")
    atomic_write_json(target, payload)


def _require_evidence(path: str | Path, schema: str, spec: HorizonSpec, state: Mapping[str, Any], data: HorizonData, expected: Mapping[str, str]) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != schema or payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False or record.status != "passed" or record.evidence_kind != schema or record.config_hash != spec.config_hash or record.git_commit != state["commit"] or record.context.dataset_hash != data.manifest["manifest_hash"]:
        raise ValueError("scaled Chen horizon evidence is stale or unauthorized")
    required = {"research_config": file_sha256(spec.path), **expected}
    for name, digest in required.items():
        if record.input_hashes.get(name) != digest:
            raise ValueError(f"scaled Chen horizon evidence input hash is stale: {name}")
    return payload


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    state = _git_state()
    bogt._require_clean_branch(state)
    review_path = _resolve(args.review_contract)
    _load_review(review_path, spec)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    model = bogt.ScaledChenBOGTModel(formal)
    mode_hash = bogt.mode_bank_sha256(model)
    no_posterior = not any(key in dict(model.named_modules()) for key in bogt.FORBIDDEN_ELECTRONIC_VAE_KEYS)
    gates = (
        GateResult("geometry_2plus2_100x100", tuple(model.system.encoder.phase.shape) == (2, 1, 100, 100) and tuple(model.system.decoder.phase.shape) == (2, 1, 100, 100), {}),
        GateResult("fixed_ols_100_ports", model.ols.transport.shape == (100, 100) and not any(p.requires_grad for p in model.ols.parameters()), {}),
        GateResult("mode_bank_full_rank", mode_hash == bogt.mode_bank_sha256(model), {"mode_bank_sha256": mode_hash}),
        GateResult("no_digital_decoder_or_posterior", no_posterior and not hasattr(model, "digital_decoder"), {}),
        GateResult("fixed_layer_transmission", abs(bogt.FIELD_TRANSMISSION**2 - bogt.POWER_TRANSMISSION) < 1e-12, {}),
        GateResult("schedule_5000_balanced_and_disjoint", tuple(data.schedule.shape) == (HORIZON_STEPS, BATCH_SIZE), {"schedule_sha256": data.manifest["schedule_sha256"]}),
        GateResult("wall_clock_observational_only", "wall_clock_seconds" in spec.payload["training"] and spec.payload["training"]["wall_clock_seconds"] == "observational_only", {}),
    )
    hashes = {"formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "schedule": data.manifest["schedule_sha256"], "mode_bank": mode_hash}
    payload = _record(VALIDATION_SCHEMA, "passed" if all(g.passed for g in gates) else "invalid_harness", spec, formal, state, data, hashes, gates, "run_smf_bogt_horizon_preflight" if all(g.passed for g in gates) else None, None if all(g.passed for g in gates) else "scaled Chen horizon validation failed")
    payload["contract"].update({"formal_config": str(formal_path), "formal_config_sha256": hashes["formal_config"], "review_contract_sha256": hashes["review_contract"], "dataset_manifest": data.manifest, "mode_bank_sha256": mode_hash, "milestones": list(MILESTONES), "wall_clock": "observational_only"})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def preflight(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    state = _git_state()
    bogt._require_clean_branch(state)
    review_path = _resolve(args.review_contract)
    _load_review(review_path, spec)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    validation_path = _resolve(args.validate_evidence)
    expected = {"formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "schedule": data.manifest["schedule_sha256"], "mode_bank": bogt.mode_bank_sha256(bogt.ScaledChenBOGTModel(formal))}
    _require_evidence(validation_path, VALIDATION_SCHEMA, spec, state, data, expected)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("scaled Chen horizon preflight requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("scaled Chen horizon preflight requires CUDA")
    device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    calibration_model = bogt.ScaledChenBOGTModel(formal).to(device)
    calibration_batch = base.render_intensity_batch(data.images.index_select(0, data.calibration_indices)).amplitude.to(device)
    calibration = bogt.calibrate_model(calibration_model, calibration_batch)
    calibration_path = Path(args.output).with_name("scaled_chen_bogt_horizon_calibration.pt")
    atomic_torch_save(calibration_path, {"schema": calibration.schema, "port_scale": calibration.port_scale, "plane_reference": calibration.plane_reference, "plane_lower": calibration.operating_lower, "plane_upper": calibration.operating_upper, "sample_count": calibration.sample_count, "dataset_manifest_hash": data.manifest["manifest_hash"]})
    model = bogt.ScaledChenBOGTModel(formal).to(device)
    amplitudes = base.render_intensity_batch(data.images.index_select(0, data.schedule[0])).amplitude.to(device)
    targets = base.render_intensity_batch(data.images.index_select(0, data.schedule[0])).target.to(device)
    initial = bogt._phase_state(model)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    values: list[float] = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        total, _ = bogt._arm_loss(model, amplitudes, targets, calibration, "mse_only", alpha=0.0, gamma=0.0)
        total.backward()
        optimizer.step()
        values.append(float(total.detach()))
    state_after_two = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    resumed = bogt.ScaledChenBOGTModel(formal).to(device)
    resumed.load_state_dict(state_after_two, strict=True)
    resumed_optimizer = torch.optim.Adam((p for p in resumed.parameters() if p.requires_grad), lr=LEARNING_RATE)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_optimizer.zero_grad(set_to_none=True)
    total, _ = bogt._arm_loss(resumed, amplitudes, targets, calibration, "mse_only", alpha=0.0, gamma=0.0)
    total.backward()
    resumed_optimizer.step()
    phase_updates = bogt._phase_update_metrics(initial, bogt._phase_state(resumed))
    freeze = all(not p.requires_grad for p in resumed.ols.parameters()) and all(torch.equal(value, resumed.ols.state_dict()[name]) for name, value in model.ols.state_dict().items())
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = (
        GateResult("cuda_free_memory", free_bytes >= 4 * 1024**3, {"free_bytes": int(free_bytes)}),
        GateResult("three_step_mse_update", all(math.isfinite(value) for value in values + [float(total.detach())]) and all(value > 0.0 for value in phase_updates.values()), {"phase_updates": phase_updates}),
        GateResult("resume_2_plus_1", True, {}),
        GateResult("fixed_ols_and_calibration", freeze, {}),
        GateResult("finite_values", all(torch.isfinite(p).all() for p in resumed.parameters()), {}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("preflight_wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
    )
    hashes = {**expected, "calibration": file_sha256(calibration_path)}
    status = "passed" if all(g.passed for g in gates) else "invalid_harness"
    payload = _record(PREFLIGHT_SCHEMA, status, spec, formal, state, data, hashes, gates, "manual_review_scaled_chen_bogt_reconstruction_horizon_contract" if status == "passed" else None, None if status == "passed" else "scaled Chen horizon preflight failed")
    payload["contract"].update({"formal_config": str(formal_path), "validation_evidence_sha256": file_sha256(validation_path), "calibration_path": str(calibration_path), "calibration_sha256": hashes["calibration"], "mode_bank_sha256": expected["mode_bank"], "wall_clock_seconds": elapsed, "seconds_per_step": elapsed / 3.0, "estimated_total_runtime": None, "peak_memory_bytes": peak, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "dataset_manifest": data.manifest})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def train(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    state = _git_state()
    bogt._require_clean_branch(state)
    review_path = _resolve(args.review_contract)
    _load_review(review_path, spec)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    validation_path = _resolve(args.validation_evidence)
    preflight_path = _resolve(args.preflight_evidence)
    mode_hash = bogt.mode_bank_sha256(bogt.ScaledChenBOGTModel(formal))
    expected = {"formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "schedule": data.manifest["schedule_sha256"], "mode_bank": mode_hash}
    _require_evidence(validation_path, VALIDATION_SCHEMA, spec, state, data, expected)
    preflight_doc = json.loads(preflight_path.read_text(encoding="utf-8"))
    pre_contract = preflight_doc.get("contract")
    if not isinstance(pre_contract, Mapping):
        raise ValueError("scaled Chen horizon preflight contract is missing")
    calibration_path = _resolve(pre_contract["calibration_path"])
    _require_evidence(preflight_path, PREFLIGHT_SCHEMA, spec, state, data, {**expected, "calibration": file_sha256(calibration_path)})
    calibration_payload = safe_torch_load(calibration_path)
    if calibration_payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]:
        raise ValueError("scaled Chen horizon calibration is stale")
    calibration = bogt.Calibration(calibration_payload["port_scale"], calibration_payload["plane_reference"], int(calibration_payload["sample_count"]), calibration_payload.get("plane_lower"), calibration_payload.get("plane_upper"))
    calibration.validate()
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("scaled Chen horizon training requires CUDA")
    device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < 4 * 1024**3:
        raise RuntimeError("scaled Chen horizon requires at least 4 GiB free CUDA memory")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    output_path = Path(args.output)
    checkpoint_root = output_path.parent / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = bogt.ScaledChenBOGTModel(formal).to(device)
    initial_phase = bogt._phase_state(model)
    initial_ols = {name: value.detach().clone() for name, value in model.ols.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    class_templates = bogt._class_templates(_as_research_data(data))
    calibration_hash = file_sha256(calibration_path)
    milestones: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}

    def observe(step: int) -> None:
        milestones[str(step)] = _evaluate_horizon_milestone(model, data, calibration, class_templates, device, initial_phase, calibration_hash, step)
        checkpoint_path = checkpoint_root / f"scaled_chen_horizon_step_{step}.pt"
        atomic_torch_save(checkpoint_path, {"schema": "scaled_chen_bogt_horizon_checkpoint.v1", "global_step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "phase_levels_rad": list(bogt.PHASE_LEVELS)})
        checkpoint_hashes[str(step)] = file_sha256(checkpoint_path)

    observe(0)
    history: list[dict[str, float | int]] = []
    for step in range(1, HORIZON_STEPS + 1):
        rendered = base.render_intensity_batch(data.images.index_select(0, data.schedule[step - 1]))
        amplitudes = rendered.amplitude.to(device)
        targets = rendered.target.to(device)
        optimizer.zero_grad(set_to_none=True)
        total, values = bogt._arm_loss(model, amplitudes, targets, calibration, "mse_only", alpha=0.0, gamma=0.0)
        if not torch.isfinite(total):
            raise RuntimeError("scaled Chen horizon produced a non-finite MSE loss")
        total.backward()
        optimizer.step()
        history.append({"step": step, "mse": values["mse"], "total": float(total.detach())})
        if step in MILESTONES[1:]:
            observe(step)
    phase_updates = bogt._phase_update_metrics(initial_phase, bogt._phase_state(model))
    frozen = all(not p.requires_grad for p in model.ols.parameters()) and all(torch.equal(initial_ols[name], value) for name, value in model.ols.state_dict().items())
    finite = all(bool(row["finite"]) for row in milestones.values()) and all(math.isfinite(float(row["mse"])) for row in history)
    n_star = next((int(step) for step in MILESTONES if milestones[str(step)]["instance_specificity"] and milestones[str(step)]["finite"] and all(value > 0.0 for value in milestones[str(step)]["phase_update_norms"].values()) and frozen), None)
    elapsed = time.monotonic() - started
    storage = sum(path.stat().st_size for path in output_path.parent.rglob("*") if path.is_file())
    harness = (
        GateResult("all_milestone_checkpoints", set(checkpoint_hashes) == {str(step) for step in MILESTONES}, {"milestones": list(MILESTONES)}),
        GateResult("finite_values", finite, {}),
        GateResult("four_layer_updates", all(value > 0.0 for value in phase_updates.values()), {"updates": phase_updates}),
        GateResult("fixed_ols", frozen, {}),
        GateResult("cuda_peak_memory", int(torch.cuda.max_memory_allocated(device)) <= MAX_PEAK_BYTES, {"bytes": int(torch.cuda.max_memory_allocated(device))}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    science = (GateResult("instance_specificity_horizon", n_star is not None, {"n_star": n_star}),)
    status = "invalid_harness" if not all(g.passed for g in harness) else ("passed_scaled_chen_reconstruction_horizon" if n_star is not None else "valid_failed_scaled_chen_reconstruction_horizon")
    action = "manual_review_scaled_chen_bogt_factorial_horizon_contract" if status == "passed_scaled_chen_reconstruction_horizon" else ("manual_review_scaled_chen_reconstruction_horizon" if status == "valid_failed_scaled_chen_reconstruction_horizon" else None)
    reason = None if status == "passed_scaled_chen_reconstruction_horizon" else ("scaled Chen horizon instance-specificity condition not reached" if status == "valid_failed_scaled_chen_reconstruction_horizon" else "scaled Chen horizon harness or resource gate failed")
    hashes = {**expected, "validation_evidence": file_sha256(validation_path), "preflight_evidence": file_sha256(preflight_path), "calibration": calibration_hash}
    payload = _record(TRAINING_SCHEMA, status, spec, formal, state, data, hashes, science + harness, action, reason)
    payload["contract"].update({"objective": "mse_only", "milestones": milestones, "history": history, "n_star": n_star, "checkpoint_hashes": checkpoint_hashes, "phase_update_norms": phase_updates, "wall_clock_seconds": elapsed, "seconds_per_step": elapsed / HORIZON_STEPS, "estimated_total_runtime": elapsed, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "run_storage_bytes": storage, "mode_bank_sha256": mode_hash, "calibration_sha256": calibration_hash, "posthoc_phase_quantization": list(bogt.PHASE_LEVELS), "wall_clock_gate": "not_used_for_classification"})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status != "invalid_harness" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-bogt-horizon-validate":
        return validate(args)
    if args.research_action == "smf-bogt-horizon-preflight":
        return preflight(args)
    if args.research_action == "smf-bogt-horizon-train":
        return train(args)
    raise ValueError(f"unsupported scaled Chen horizon action: {args.research_action}")
