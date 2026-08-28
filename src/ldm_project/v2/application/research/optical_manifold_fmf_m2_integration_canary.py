"""Diagnostic-only PLAN 7a FMF M=2 integration canary.

This module adds the first two-mode research boundary without changing the
formal Identity system or the validated SMF lifecycle.  It intentionally
uses the PLAN 6 SMF checkpoint as a common starting point and only measures
whether an LP01+LP11a path can execute and update the four phase layers.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ...config import FormalSystemConfig, load_config
from ...contracts import ComplexField, IntensityFrame, OperationProvenance
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...optics.devices import IdentityOLS
from ...optics.fiber_operator import FiberArrayOLS
from ...optics.few_mode_fiber_operator import FewModeFiberArrayOLS, FewModeFiberArraySpec
from . import optical_manifold as base
from . import optical_manifold_full_epoch as full_epoch
from . import optical_manifold_smf_integration_canary as smf_canary
from . import optical_manifold_smf_reconstruction_pilot as plan6


BRANCH = "research/v2-lightgen-optical-manifold-fmf-m2-integration-canary-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_fmf_m2_integration_canary_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_fmf_m2_integration_canary_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_fmf_m2_canary_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_fmf_m2_canary_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_fmf_m2_canary_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_fmf_m2_canary_resume.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_fmf_m2_canary_panel.v1"

SOURCE_VALIDATION_HASH = "d548cf93f2cfefb2258239b066b14f574b098bfa17edc0f377f718fd2cba3ca5"
SOURCE_PREFLIGHT_HASH = "722557fc0bfc058edafa8666772ee87bb927b0a37924b55087467303d3a56623"
SOURCE_TRAINING_HASH = "a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e"
SOURCE_PANEL_HASH = "9de2fc9d9c2d43ed9752d8609afcaa5455cfb05b9c411d7d0cca9b2a8cf50fdf"
SOURCE_CHECKPOINT_HASH = "949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_STEP = 1000
STEPS = 20
PREFLIGHT_STEPS = 3
BATCH_SIZE = 10
MICROBATCH = 1
CHECKPOINT_INTERVAL = 10
SEED = 1234
LEARNING_RATE = 1.0e-3
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 1800.0
MAX_STORAGE_BYTES = 512 * 1024**2
SAFETY_MSE_RATIO = 2.0
NEXT_ACTION = "manual_review_fmf_m2_reconstruction_pilot_contract"
ARMS = ("fmf_m2_mse_canary", "smf_m1_mse_control")
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


@dataclass(frozen=True)
class CanarySpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class CanaryData:
    images: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    heldout_indices: torch.Tensor
    schedule: torch.Tensor
    visual_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    checkpoint: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


class _SafetyAbort(RuntimeError):
    def __init__(self, arm: str, step: int, loss: float, baseline: float) -> None:
        super().__init__(f"FMF M=2 canary safety bound exceeded for {arm} at step {step}")
        self.arm = arm
        self.step = step
        self.loss = loss
        self.baseline = baseline


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"FMF M=2 canary requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("FMF M=2 canary requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")


def _load_spec(path: str | Path) -> CanarySpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {"schema", "hypothesis", "formal_system_config", "source", "dataset", "optical_contract", "training", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("FMF M=2 config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported FMF M=2 config")
    return CanarySpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: CanarySpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_fmf_m2_integration_canary",
        "approved_contract": {
            "arms": list(ARMS),
            "steps_per_arm": STEPS,
            "logical_batch_size": BATCH_SIZE,
            "microbatch_size": MICROBATCH,
            "source_global_step": SOURCE_STEP,
            "preflight_steps": PREFLIGHT_STEPS,
            "resume_segment": "2+1",
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "m2_modes": ["LP01", "LP11a"],
            "m2_transport": "identity_200",
            "post_diagnostic_next_action": NEXT_ACTION,
        },
        "bounded_diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("FMF M=2 review contract does not match scope")
    return payload, target


def _assert_contract(spec: CanarySpec) -> None:
    p = spec.payload
    source = p["source"]
    expected_source = {
        "plan6_validation_sha256": SOURCE_VALIDATION_HASH,
        "plan6_preflight_sha256": SOURCE_PREFLIGHT_HASH,
        "plan6_training_sha256": SOURCE_TRAINING_HASH,
        "plan6_panel_sha256": SOURCE_PANEL_HASH,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "mnist_manifest_sha256": SOURCE_MANIFEST_HASH,
        "source_global_step": SOURCE_STEP,
    }
    if any(source.get(k) != v for k, v in expected_source.items()):
        raise ValueError("FMF M=2 source binding is not locked")
    if p["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH,
        "schedule": "class_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "excluded_prior_panels": True,
        "excluded_prior_schedules": True,
        "heldout_excluded": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "heldout_panel_count": 1000,
        "visual_panel_count": 100,
    }:
        raise ValueError("FMF M=2 dataset contract is not locked")
    required_optical = {
        "formal_system_kind": base.EXPECTED_FORMAL_KIND,
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "boundary": "finite_crop",
        "formal_ols": "identity",
        "m1_research_ols": "100_port_lp01_smf",
        "m2_research_ols": "100_port_two_mode_fmf",
        "m2_modes": ["LP01", "LP11a"],
        "m2_transport": "identity_200",
        "m2_mode_provenance": "synthetic_fmf_nested_lp01_lp11a_identity_transport",
        "m2_parameter_status": "diagnostic_project_approximation",
        "front_relay": "coherent_imaging_4f",
        "back_relay": "free_space",
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    }
    if p["optical_contract"] != required_optical:
        raise ValueError("FMF M=2 optical contract is not locked")
    if p["training"] != {
        "arms": list(ARMS),
        "optimizer": "Adam",
        "optimizer_update_device": "cpu_shadow",
        "learning_rate": LEARNING_RATE,
        "reset_adam_from_source": True,
        "steps_per_arm": STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "reconstruction_activation_checkpoint": True,
        "deterministic": True,
        "fixed_execution_order": list(ARMS),
        "no_early_stop": True,
        "no_tuning": True,
        "no_checkpoint_selection": True,
    }:
        raise ValueError("FMF M=2 training contract is not locked")
    if p["gates"] != {
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "max_preflight_wall_seconds": MAX_PREFLIGHT_SECONDS,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "safety_mse_ratio": SAFETY_MSE_RATIO,
        "evaluator_calls": 0,
        "require_lp01_nested_exact": True,
        "require_lp11a_nonzero": True,
        "require_mode_bank_orthonormal": True,
        "require_identity_transport": True,
    }:
        raise ValueError("FMF M=2 gate contract is not locked")
    auth = p["authorization"]
    if auth.get("bounded_diagnostic_training_authorized") is not True or auth.get("training_authorized") is not False or auth.get("formal_promotion") is not False or auth.get("post_diagnostic_next_action") != NEXT_ACTION:
        raise ValueError("FMF M=2 authorization boundary is invalid")


def _load_formal(spec: CanarySpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2" or formal.system.kind != base.EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("FMF M=2 requires current formal Identity optics")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, current], cwd=base._repo_root(), check=False)
    if result.returncode != 0:
        raise ValueError("PLAN 6 evidence commit is not an ancestor")


def _load_source(spec: CanarySpec, formal_path: Path, current_commit: str) -> SourceBundle:
    source = spec.payload["source"]
    entries = {
        "plan6_validation": (_resolve(source["plan6_validation"]), SOURCE_VALIDATION_HASH),
        "plan6_preflight": (_resolve(source["plan6_preflight"]), SOURCE_PREFLIGHT_HASH),
        "plan6_training": (_resolve(source["plan6_training"]), SOURCE_TRAINING_HASH),
        "plan6_panel": (_resolve(source["plan6_panel"]), SOURCE_PANEL_HASH),
        "source_checkpoint": (_resolve(source["source_checkpoint"]), SOURCE_CHECKPOINT_HASH),
    }
    for role, (path, expected) in entries.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    for role in ("plan6_validation", "plan6_preflight", "plan6_training"):
        payload = json.loads(entries[role][0].read_text(encoding="utf-8"))
        record = EvidenceRecordV2.from_dict(payload["evidence"])
        record.validate()
        if record.status != "passed" or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
            raise ValueError("PLAN 6 source evidence is not an authorized passed artifact")
        _assert_ancestor(record.git_commit, current_commit)
    panel = safe_torch_load(entries["plan6_panel"][0], expected_sha256=SOURCE_PANEL_HASH, map_location="cpu")
    if panel.get("schema") != plan6.PANEL_SCHEMA or panel.get("indices_are_global_mnist") is not True:
        raise ValueError("PLAN 6 source panel is invalid")
    checkpoint = safe_torch_load(entries["source_checkpoint"][0], expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if checkpoint.get("schema") != plan6.RESUME_SCHEMA or checkpoint.get("arm") != "smf_mse_reconstruction" or checkpoint.get("step") != SOURCE_STEP:
        raise ValueError("PLAN 6 SMF source checkpoint is invalid")
    if not isinstance(checkpoint.get("model_state"), Mapping) or not isinstance(checkpoint.get("optimizer_state"), Mapping) or not isinstance(checkpoint.get("rng_state"), Mapping):
        raise ValueError("PLAN 6 source checkpoint is incomplete")
    return SourceBundle(
        checkpoint=checkpoint,
        formal_path=formal_path,
        hashes={
            "plan6_validation": SOURCE_VALIDATION_HASH,
            "plan6_preflight": SOURCE_PREFLIGHT_HASH,
            "plan6_training": SOURCE_TRAINING_HASH,
            "plan6_panel": SOURCE_PANEL_HASH,
            "source_checkpoint": SOURCE_CHECKPOINT_HASH,
            "mnist_manifest": SOURCE_MANIFEST_HASH,
            "formal_config": file_sha256(formal_path),
        },
    )


def _load_data(dataset_root: str | Path) -> CanaryData:
    source_data = plan6._load_data(dataset_root)
    schedule = source_data.schedule[:STEPS].clone()
    if schedule.shape != (STEPS, BATCH_SIZE) or torch.unique(schedule).numel() != schedule.numel():
        raise ValueError("FMF M=2 schedule is not fresh and without replacement")
    if not torch.equal(source_data.labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(STEPS)):
        raise ValueError("FMF M=2 schedule is not balanced one-per-label")
    heldout_labels = source_data.labels.index_select(0, source_data.heldout_indices)
    heldout = torch.cat([source_data.heldout_indices[heldout_labels == label] for label in range(10)])
    visual = torch.cat([source_data.heldout_indices[heldout_labels == label][:10] for label in range(10)])
    manifest = {
        "schema": "lightgen_optical_manifold_fmf_m2_canary_mnist_manifest.v1",
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "source_plan6_manifest_hash": source_data.manifest["manifest_hash"],
        "schedule_sha256": tensor_sha256(schedule),
        "schedule_shape": list(schedule.shape),
        "schedule_count": int(schedule.numel()),
        "schedule_seed": SEED,
        "heldout_indices_sha256": tensor_sha256(heldout),
        "heldout_count": int(heldout.numel()),
        "visual_indices_sha256": tensor_sha256(visual),
        "visual_count": int(visual.numel()),
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "schedule_balance_and_offline_metrics_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return CanaryData(source_data.images, source_data.labels, source_data.train_indices, heldout, schedule, visual, manifest)


def _context(spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _model_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    contract = base._parameter_contract(model)
    if tuple(contract["trainable_parameter_names"]) != EXPECTED_TRAINABLE:
        raise ValueError("FMF M=2 trainable parameter contract failed")
    return contract


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema": schema, "evidence": record.to_dict(), "hypothesis": HYPOTHESIS, "training_authorized": False, "formal_promotion": False, "contract": dict(contract)}
    if extra:
        payload.update(extra)
    return payload


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite FMF M=2 artifact: {target}")
    atomic_write_json(target, payload)


def _state_cpu(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.nn.Module):
        return {name: item.detach().cpu().clone() for name, item in value.state_dict().items()}
    if isinstance(value, IdentityOLS):
        return {"type": type(value).__qualname__, "attributes": copy.deepcopy(vars(value))}
    raise TypeError(f"unsupported state snapshot {type(value)!r}")


def _tree_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _tree_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_clone(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _batch(data: CanaryData, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude.to(device), rendered.target.to(device)


def _smf_operator(model: base.IdentityOpticalManifoldAutoencoder, device: torch.device) -> FiberArrayOLS:
    return smf_canary._smf_operator(model.system, device)


def _fmf_operator(model: base.IdentityOpticalManifoldAutoencoder, device: torch.device, *, zero_transport: bool = False) -> FewModeFiberArrayOLS:
    spec = FewModeFiberArraySpec(modes=("LP01", "LP11a"))
    transport = torch.zeros(200, 200, dtype=torch.complex64) if zero_transport else None
    return FewModeFiberArrayOLS(model.system.encoder_plane, spec, transport=transport).to(device)


def _runtime(formal: FormalSystemConfig, source: SourceBundle, device: torch.device, arm: str) -> tuple[base.IdentityOpticalManifoldAutoencoder, full_epoch._DeterministicCpuAdam, FiberArrayOLS | FewModeFiberArrayOLS]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(source.checkpoint["model_state"], strict=True)
    optimizer = full_epoch._make_full_optimizer(model)
    base._restore_rng(source.checkpoint["rng_state"])
    operator = _fmf_operator(model, device) if arm == "fmf_m2_mse_canary" else _smf_operator(model, device)
    return model, optimizer, operator


def _forward(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, amplitude: torch.Tensor, *, trace: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    system = model.system
    slm_field = system.slm(system.command(amplitude))
    front = system.front(slm_field)
    encoded, encoder_records, encoder_stats = system.encoder(front, "encoder")
    transformed = operator(encoded)
    decoded, _decoder_records, decoder_stats = system.decoder(transformed, "decoder")
    chip = ComplexField(decoded.tensor, system.chip_plane, decoded.provenance)
    camera_field = system.back(chip)
    smf_canary._relay_stats(system)
    raw = IntensityFrame(camera_field.tensor.abs().square(), system.camera_plane, camera_field.provenance, model.exposure.calibration_id)
    diagnostics = {
        "provenance": transformed.provenance.to_dict(),
        "coefficients": operator.last_diagnostics.get("port_input_coefficients"),
        "mode_coefficients": operator.last_diagnostics.get("mode_coefficients"),
        "mode_power": operator.last_diagnostics.get("mode_power"),
        "active_mode_fraction": operator.last_diagnostics.get("active_mode_fraction"),
        "coupling_efficiency": operator.last_diagnostics.get("coupling_efficiency"),
        "input_energy": operator.last_diagnostics.get("input_energy"),
        "output_energy": operator.last_diagnostics.get("output_energy"),
        "encoder_crop_retention": encoder_stats["crop_retention"],
        "decoder_crop_retention": decoder_stats["crop_retention"],
    }
    return raw.tensor, encoder_records[-1].tensor if trace else None, diagnostics


def _clear_grad(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _add_grads(total: dict[str, torch.Tensor], model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            total[name].add_(parameter.grad.detach().cpu())


def _grad_norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return smf_canary._layer_norms(values)


def _gradients(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, amplitudes: torch.Tensor, targets: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    gradients = {name: torch.zeros_like(value, device="cpu") for name, value in model.named_parameters()}
    losses: list[torch.Tensor] = []
    cameras: list[torch.Tensor] = []
    coefficient_shape: list[int] | None = None
    _clear_grad(model)
    for start in range(0, BATCH_SIZE, MICROBATCH):
        def run(value: torch.Tensor) -> torch.Tensor:
            return _forward(model, operator, value)[0]
        raw = checkpoint(run, amplitudes[start : start + 1], use_reentrant=False)
        coeff = operator.last_diagnostics.get("port_input_coefficients")
        if isinstance(coeff, torch.Tensor):
            coefficient_shape = list(coeff.shape)
        loss = F.mse_loss(raw, targets[start : start + 1])
        losses.append(loss.detach())
        cameras.append(raw.detach())
        (loss / BATCH_SIZE).backward()
    _add_grads(gradients, model)
    _clear_grad(model)
    camera = torch.cat(cameras)
    return gradients, {
        "loss": float(torch.stack(losses).mean().cpu()),
        "camera_finite": bool(torch.isfinite(camera).all()),
        "camera_min": float(camera.min().cpu()),
        "camera_max": float(camera.max().cpu()),
        "coefficient_shape": coefficient_shape,
        "gradient_norms": _grad_norms(gradients),
    }


def _step(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, operator: FiberArrayOLS | FewModeFiberArrayOLS, amplitudes: torch.Tensor, targets: torch.Tensor, arm: str) -> dict[str, Any]:
    before = {name: value.detach().clone() for name, value in model.named_parameters() if name in EXPECTED_TRAINABLE}
    gradients, info = _gradients(model, operator, amplitudes, targets)
    for name, value in model.named_parameters():
        value.grad = gradients[name].to(value.device)
    optimizer.step()
    parameters = dict(model.named_parameters())
    updates: dict[str, float] = {}
    for name in EXPECTED_TRAINABLE:
        updates[name] = float((parameters[name].detach() - before[name]).abs().sum().cpu())
        for index in range(parameters[name].shape[0]):
            updates[f"{name}[{index}]"] = float((parameters[name][index].detach() - before[name][index]).abs().sum().cpu())
    finite_values = [info["loss"], info["camera_min"], info["camera_max"], *_grad_norms(gradients).values(), *updates.values()]
    _clear_grad(model)
    return {
        "arm": arm,
        "loss": info["loss"],
        "camera_finite": info["camera_finite"],
        "camera_min": info["camera_min"],
        "camera_max": info["camera_max"],
        "coefficient_shape": info["coefficient_shape"],
        "gradient_norms": _grad_norms(gradients),
        "update_norms": updates,
        "activation_checkpoint": True,
        "finite": info["camera_finite"] and all(bool(torch.isfinite(torch.tensor(value))) for value in finite_values),
    }


def _structure_metrics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return plan6._structure_metrics(features, labels)


def _decode_pre_ols(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, field: torch.Tensor) -> torch.Tensor:
    system = model.system
    value = ComplexField(field, system.encoder_plane, OperationProvenance("fmf_m2_interpolation_pre_ols", "encoder_layer_2", None))
    transformed = operator(value)
    decoded, _, _ = system.decoder(transformed, "decoder")
    camera_field = system.back(ComplexField(decoded.tensor, system.chip_plane, decoded.provenance))
    return camera_field.tensor.abs().square()


def _interpolation(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, pre_visual: torch.Tensor, camera_visual: torch.Tensor, device: torch.device) -> dict[str, Any]:
    values: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    for label in range(10):
        first = pre_visual[label * 10 : label * 10 + 1].to(device)
        last = pre_visual[label * 10 + 9 : label * 10 + 10].to(device)
        outputs = [_decode_pre_ols(model, operator, first * (1.0 - t) + last * t).detach() for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        errors.extend((F.mse_loss(outputs[0], camera_visual[label * 10 : label * 10 + 1].to(device)), F.mse_loss(outputs[-1], camera_visual[label * 10 + 9 : label * 10 + 10].to(device))))
        values.extend(outputs)
    stacked = torch.stack(values)
    adjacent = (stacked[1:] - stacked[:-1]).flatten(1).norm(dim=1)
    return {"endpoint_max_mse": float(torch.stack(errors).max().cpu()), "adjacent_mean_l2": float(adjacent.mean().cpu()), "finite": bool(torch.isfinite(stacked).all()), "endpoint_count": 20, "interpolation_count": 50, "t_values": [0.0, 0.25, 0.5, 0.75, 1.0]}


def _effective_rank(features: torch.Tensor) -> float:
    work = torch.cat((features.real, features.imag), dim=1).to(torch.float64) if torch.is_complex(features) else features.to(torch.float64)
    centered = work - work.mean(0, keepdim=True)
    values = torch.linalg.svdvals(centered)
    weights = values.square() / values.square().sum().clamp_min(1.0e-30)
    return float(1.0 / weights.square().sum().clamp_min(1.0e-30))


def _evaluate(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, data: CanaryData, indices: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    cameras: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    couplings: list[torch.Tensor] = []
    active_modes: list[torch.Tensor] = []
    optical_powers: list[torch.Tensor] = []
    pres: list[torch.Tensor] = []
    visual_cameras: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            amplitudes, target = _batch(data, batch_indices, device)
            raw, pre, diagnostics = _forward(model, operator, amplitudes, trace=True)
            coefficient = diagnostics["coefficients"]
            if not isinstance(coefficient, torch.Tensor):
                raise ValueError("FMF evaluation did not expose mode coefficients")
            cameras.append(raw.cpu())
            targets.append(target.cpu())
            features.append(coefficient.cpu())
            mode_power = diagnostics.get("mode_power")
            if isinstance(mode_power, torch.Tensor):
                powers.append(mode_power.cpu())
            coupling = diagnostics.get("coupling_efficiency")
            if isinstance(coupling, torch.Tensor):
                couplings.append(coupling.cpu().reshape(-1))
            active = diagnostics.get("active_mode_fraction")
            if isinstance(active, torch.Tensor):
                active_modes.append(active.cpu().reshape(-1))
            input_energy = diagnostics.get("input_energy")
            output_energy = diagnostics.get("output_energy")
            if isinstance(input_energy, torch.Tensor) and isinstance(output_energy, torch.Tensor):
                optical_powers.append((output_energy / input_energy.clamp_min(1.0e-30)).cpu().reshape(-1))
            if sum(item.shape[0] for item in cameras) <= 100:
                pres.append(pre.cpu())
                visual_cameras.append(raw.cpu())
    camera = torch.cat(cameras)
    target = torch.cat(targets)
    feature = torch.cat(features)
    flat_camera = camera.flatten(1)
    flat_target = target.flatten(1)
    c0 = flat_camera - flat_camera.mean(1, keepdim=True)
    t0 = flat_target - flat_target.mean(1, keepdim=True)
    correlation = (c0 * t0).sum(1) / (c0.norm(1) * t0.norm(1)).clamp_min(1.0e-30)
    labels = data.labels.index_select(0, data.visual_indices).cpu()
    mode_power = torch.cat(powers) if powers else torch.empty(0)
    power_fraction = (mode_power.mean(0) / mode_power.mean(0).sum().clamp_min(1.0e-30)).tolist() if mode_power.numel() else []
    visual_pre = torch.cat(pres)[:100]
    visual_camera = torch.cat(visual_cameras)[:100]
    metrics = {
        "paired_mse": float(F.mse_loss(camera, target)),
        "cyclic_mse": float(F.mse_loss(camera, target.roll(1, 0))),
        "centered_correlation": float(correlation.mean()),
        "variance_retention": float(flat_camera.var(0, unbiased=False).mean() / flat_target.var(0, unbiased=False).mean().clamp_min(1.0e-30)),
        "finite": bool(torch.isfinite(camera).all() and torch.isfinite(target).all() and torch.isfinite(feature).all()),
        "structure": _structure_metrics(feature[:100], labels),
        "coefficient_effective_rank": _effective_rank(feature[:100]),
        "evaluation_count": int(indices.numel()),
        "coefficient_shape": list(feature.shape),
        "mode_power_fraction": power_fraction,
        "active_mode_fraction_mean": float(torch.cat(active_modes).mean()) if active_modes else None,
        "coupling_efficiency_mean": float(torch.cat(couplings).mean()) if couplings else None,
        "optical_power_retention": float(torch.cat(optical_powers).mean()) if optical_powers else None,
    }
    metrics["interpolation"] = _interpolation(model, operator, visual_pre, visual_camera, device)
    return metrics, {"camera": camera, "target": target, "pre_visual": visual_pre, "camera_visual": visual_camera, "features": feature}


def _initial_losses(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, data: CanaryData, device: torch.device) -> list[float]:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for indices in data.schedule:
            amplitudes, target = _batch(data, indices, device)
            raw, _pre, _diagnostics = _forward(model, operator, amplitudes)
            values.append(float(F.mse_loss(raw, target).cpu()))
    return values


def _resume_payload(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, operator: FiberArrayOLS | FewModeFiberArrayOLS, *, arm: str, step: int, history: list[dict[str, Any]], initial_losses: list[float], spec: CanarySpec, data: CanaryData, hashes: Mapping[str, str], completed_arms: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "arm": arm,
        "step": step,
        "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "optimizer_state": _tree_clone(optimizer.state_dict()),
        "operator_state": _state_cpu(operator),
        "rng_state": _tree_clone(base._rng_state()),
        "history": list(history),
        "initial_batch_losses": list(initial_losses),
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": data.manifest["schedule_sha256"],
        "evidence_hashes": dict(hashes),
        "data_cursor": {"arm": arm, "next_step": step, "logical_batch_size": BATCH_SIZE, "microbatch_size": MICROBATCH},
        "completed_arms": dict(completed_arms or {}),
        "formal_candidate_checkpoint": False,
        "training_authorized": False,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite FMF M=2 checkpoint: {path}")
    atomic_torch_save(path, payload)


def _load_resume(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, operator: FiberArrayOLS | FewModeFiberArrayOLS, *, arm: str, spec: CanarySpec, data: CanaryData, hashes: Mapping[str, str], output_dir: Path) -> dict[str, Any]:
    target = _resolve(path)
    if target.parent.resolve() != output_dir.resolve():
        raise ValueError("FMF M=2 resume belongs to a different attempt")
    payload = safe_torch_load(target, map_location="cpu")
    if payload.get("schema") != RESUME_SCHEMA or payload.get("arm") != arm or not isinstance(payload.get("step"), int) or not 0 < payload["step"] < STEPS:
        raise ValueError("invalid FMF M=2 resume artifact")
    if payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"] or payload.get("evidence_hashes") != dict(hashes):
        raise ValueError("FMF M=2 resume lineage mismatch")
    if payload.get("data_cursor") != {"arm": arm, "next_step": payload["step"], "logical_batch_size": BATCH_SIZE, "microbatch_size": MICROBATCH}:
        raise ValueError("FMF M=2 resume cursor mismatch")
    if not isinstance(payload.get("history"), list) or len(payload["history"]) != payload["step"] or len(payload.get("initial_batch_losses", [])) != STEPS:
        raise ValueError("FMF M=2 resume history mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    operator.load_state_dict(payload["operator_state"], strict=True)
    base._restore_rng(payload["rng_state"])
    return dict(payload)


def _validation_record(state: Mapping[str, Any], spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, source: SourceBundle, review_path: Path, contract: Mapping[str, Any]) -> EvidenceRecordV2:
    gates = tuple(GateResult(name, True, metrics) for name, metrics in (
        ("clean_lineage", {"clean": True}),
        ("independent_fmf_branch", {"branch_matches": True}),
        ("plan6_source_complete", {"global_step": SOURCE_STEP}),
        ("fresh_balanced_schedule", {"steps": STEPS, "batch": BATCH_SIZE, "count": STEPS * BATCH_SIZE}),
        ("heldout_excluded", {"count": 1000}),
        ("formal_geometry", {"grid": 2048, "identity_formal_ols": True}),
        ("nested_fmf_contract", {"modes": ["LP01", "LP11a"], "coefficients": 200}),
        ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
        ("four_phase_parameter_scope", {"parameters": contract["trainable_parameter_names"]}),
        ("diagnostic_boundary", {"training_authorized": False, "formal_promotion": False}),
    ))
    hashes = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"]}
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_fmf_m2_canary_preflight",))
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    contract = _model_contract(model)
    record = _validation_record(state, spec, formal, data, source, review_path, contract)
    payload = _record_payload(VALIDATION_SCHEMA, record, {"formal_config": str(formal_path.resolve()), "source_hashes": source.hashes, "dataset_manifest": data.manifest, "model_contract": contract, "optimizer_created": False, "cuda_context_created": False})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(path: str | Path, spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, state: Mapping[str, Any], source: SourceBundle, review_path: Path) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("FMF M=2 validation schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("FMF M=2 validation is stale or invalid")
    expected = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"]}
    if record.input_hashes != expected or record.permitted_next_actions != ("run_fmf_m2_canary_preflight",):
        raise ValueError("FMF M=2 validation input hashes or next action mismatch")
    return payload, digest


def _preflight_controls(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, fmf: FewModeFiberArrayOLS, data: CanaryData, device: torch.device) -> dict[str, Any]:
    amplitudes, _target = _batch(data, data.schedule[0], device)
    smf_raw, _smf_pre, smf_diag = _forward(model, smf, amplitudes, trace=True)
    fmf_raw, _fmf_pre, fmf_diag = _forward(model, fmf, amplitudes, trace=True)
    zero = _fmf_operator(model, device, zero_transport=True)
    zero_raw, _zero_pre, zero_diag = _forward(model, zero, amplitudes, trace=True)
    smf_lp01 = smf.input_modes
    fmf_lp01 = fmf.input_modes[0::2]
    area = model.system.encoder_plane.pixel_pitch_m[0] * model.system.encoder_plane.pixel_pitch_m[1]
    gram = fmf.input_modes.flatten(1).conj() @ fmf.input_modes.flatten(1).T * area
    coeff = fmf_diag.get("coefficients")
    mode_coeff = fmf_diag.get("mode_coefficients")
    input_energy = fmf_diag.get("input_energy")
    output_energy = fmf_diag.get("output_energy")
    return {
        "smf_coefficients_shape": list(smf_diag["coefficients"].shape),
        "fmf_coefficients_shape": list(coeff.shape) if isinstance(coeff, torch.Tensor) else None,
        "fmf_provenance": fmf_diag["provenance"].get("operation"),
        "lp01_nested_exact": bool(torch.equal(smf_lp01.cpu(), fmf_lp01.cpu())),
        "mode_bank_orthonormal": bool((gram - torch.eye(200, dtype=gram.dtype)).abs().max() <= 2.0e-4),
        "mode_bank_rank": int(torch.linalg.matrix_rank(gram).item()),
        "identity_transport": bool(torch.equal(fmf.transport.cpu(), torch.eye(200, dtype=fmf.transport.dtype))),
        "lp11a_coefficient_nonzero": bool(isinstance(mode_coeff, torch.Tensor) and mode_coeff[..., 1].abs().max() > 1.0e-8),
        "smf_fmf_camera_not_equal": not torch.equal(smf_raw, fmf_raw),
        "zero_port_finite": bool(torch.isfinite(zero_raw).all()),
        "zero_port_is_zero": bool(zero_raw.abs().max() <= 1.0e-12),
        "passivity": isinstance(input_energy, torch.Tensor) and isinstance(output_energy, torch.Tensor) and bool((output_energy <= input_energy * (1.0 + 1.0e-5) + 1.0e-12).all()),
        "fmf_input_energy_max": float(input_energy.max().cpu()) if isinstance(input_energy, torch.Tensor) else None,
        "fmf_output_energy_max": float(output_energy.max().cpu()) if isinstance(output_energy, torch.Tensor) else None,
        "zero_output_energy_max": float(zero_diag["output_energy"].max().cpu()),
        "formal_identity_ols": isinstance(model.system.ols, IdentityOLS),
        "smf_provenance": smf_diag["provenance"].get("operation"),
    }


def _exact_resume(formal: FormalSystemConfig, source: SourceBundle, data: CanaryData, arm: str, spec: CanarySpec, hashes: Mapping[str, str], output_dir: Path, device: torch.device) -> dict[str, Any]:
    continuous_model, continuous_optimizer, continuous_operator = _runtime(formal, source, device, arm)
    split_model, split_optimizer, split_operator = _runtime(formal, source, device, arm)
    continuous_history: list[dict[str, Any]] = []
    split_history: list[dict[str, Any]] = []
    for step in range(PREFLIGHT_STEPS):
        amp, target = _batch(data, data.schedule[step], device)
        continuous_history.append(_step(continuous_model, continuous_optimizer, continuous_operator, amp, target, arm))
    continuous_rng = _tree_clone(base._rng_state())
    for step in range(PREFLIGHT_STEPS - 1):
        amp, target = _batch(data, data.schedule[step], device)
        split_history.append(_step(split_model, split_optimizer, split_operator, amp, target, arm))
    path = output_dir / f"preflight_{arm}_step_00002.pt"
    _save_checkpoint(path, _resume_payload(split_model, split_optimizer, split_operator, arm=arm, step=2, history=split_history, initial_losses=[0.0] * STEPS, spec=spec, data=data, hashes=hashes))
    resumed_model, resumed_optimizer, resumed_operator = _runtime(formal, source, device, arm)
    _load_resume(path, resumed_model, resumed_optimizer, resumed_operator, arm=arm, spec=spec, data=data, hashes=hashes, output_dir=output_dir)
    amp, target = _batch(data, data.schedule[2], device)
    split_history.append(_step(resumed_model, resumed_optimizer, resumed_operator, amp, target, arm))
    resumed_rng = _tree_clone(base._rng_state())
    return {
        "continuous": continuous_history,
        "resumed": split_history,
        "model_exact": all(torch.equal(continuous_model.state_dict()[name].cpu(), resumed_model.state_dict()[name].cpu()) for name in continuous_model.state_dict()),
        "optimizer_exact": _tree_equal(continuous_optimizer.state_dict(), resumed_optimizer.state_dict()),
        "operator_exact": _tree_equal(_state_cpu(continuous_operator), _state_cpu(resumed_operator)),
        "rng_exact": _tree_equal(continuous_rng, resumed_rng),
        "finite": all(item["finite"] for item in continuous_history + split_history),
        "resume_artifact": str(path.resolve()),
        "resume_artifact_sha256": file_sha256(path),
    }


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("FMF M=2 preflight requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    _validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source, review_path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite FMF M=2 preflight: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for FMF M=2 preflight")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash}
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        source_model, _source_optimizer, source_smf = _runtime(formal, source, device, "smf_m1_mse_control")
        source_fmf = _fmf_operator(source_model, device)
        source_model_state = _state_cpu(source_model)
        source_formal_state = _state_cpu(source_model.system.ols)
        source_smf_state = _state_cpu(source_smf)
        source_fmf_state = _state_cpu(source_fmf)
        exact = {arm: _exact_resume(formal, source, data, arm, spec, hashes, output.parent, device) for arm in ARMS}
        reports: dict[str, Any] = {}
        for arm in ARMS:
            model, optimizer, operator = _runtime(formal, source, device, arm)
            amp, target = _batch(data, data.schedule[0], device)
            reports[arm] = {"step": _step(model, optimizer, operator, amp, target, arm), "exact_resume": exact[arm]}
        controls = _preflight_controls(source_model, source_smf, source_fmf, data, device)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        gates = (
            GateResult("finite_values", all(item["step"]["finite"] and item["exact_resume"]["finite"] for item in reports.values()), {}),
            GateResult("four_phase_updates", all(smf_canary._four_phase_nonzero(reports[arm]["step"]["update_norms"]) for arm in ARMS), {"layers": 4}),
            GateResult("activation_checkpoint", all(reports[arm]["step"]["activation_checkpoint"] for arm in ARMS), {}),
            GateResult("exact_resume", all(item["model_exact"] and item["optimizer_exact"] and item["operator_exact"] and item["rng_exact"] for item in exact.values()), {}),
            GateResult("nested_mode_bank", controls["lp01_nested_exact"] and controls["mode_bank_orthonormal"] and controls["mode_bank_rank"] == 200, controls),
            GateResult("fmf_trace_and_coefficients", controls["fmf_provenance"] == "synthetic_fmf_nested_lp01_lp11a_identity_transport" and controls["fmf_coefficients_shape"] == [1, 200] and controls["lp11a_coefficient_nonzero"], controls),
            GateResult("smf_trace_and_coefficients", controls["smf_provenance"] == "synthetic_smf_lp01_port_transport" and controls["smf_coefficients_shape"] == [1, 100], controls),
            GateResult("passivity_and_no_bypass", controls["passivity"] and controls["zero_port_finite"] and controls["zero_port_is_zero"] and controls["smf_fmf_camera_not_equal"], controls),
            GateResult("formal_and_frozen_state", controls["formal_identity_ols"] and _tree_equal(source_model_state, _state_cpu(source_model)) and _tree_equal(source_formal_state, _state_cpu(source_model.system.ols)) and _tree_equal(source_smf_state, _state_cpu(source_smf)) and _tree_equal(source_fmf_state, _state_cpu(source_fmf)), {}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        passed = all(gate.passed for gate in gates)
        record = EvidenceRecordV2(2, PREFLIGHT_SCHEMA, "diagnostic", "passed" if passed else "invalid_harness", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_fmf_m2_canary_train",) if passed else (), None if passed else "FMF M=2 preflight gate failed")
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"validation_hash": validation_hash, "source_hashes": source.hashes, "training_hashes": hashes, "arms": reports, "controls": controls, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "optimizer_created": True})
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if passed else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def _load_preflight(path: str | Path, spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, state: Mapping[str, Any], validation_hash: str, source: SourceBundle, review_path: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != PREFLIGHT_SCHEMA or record.status != "passed" or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("FMF M=2 preflight is not passed")
    if record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("FMF M=2 preflight is stale")
    expected = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash}
    if record.input_hashes != expected or record.permitted_next_actions != ("run_fmf_m2_canary_train",) or payload.get("contract", {}).get("training_hashes") != expected:
        raise ValueError("FMF M=2 preflight provenance mismatch")
    return payload, digest, {**expected, "preflight_evidence": digest}


def _run_steps(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, operator: FiberArrayOLS | FewModeFiberArrayOLS, data: CanaryData, arm: str, start: int, history: list[dict[str, Any]], initial_losses: list[float], output_dir: Path, spec: CanarySpec, hashes: Mapping[str, str], completed_arms: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], Path | None, str | None]:
    final_path: Path | None = None
    final_hash: str | None = None
    for index in range(start, STEPS):
        amplitudes, targets = _batch(data, data.schedule[index], next(model.parameters()).device)
        result = _step(model, optimizer, operator, amplitudes, targets, arm)
        result["step"] = index + 1
        history.append(result)
        if not result["finite"]:
            raise ValueError("FMF M=2 canary encountered non-finite values")
        if result["loss"] > initial_losses[index] * SAFETY_MSE_RATIO:
            raise _SafetyAbort(arm, index + 1, result["loss"], initial_losses[index])
        if (index + 1) % CHECKPOINT_INTERVAL == 0:
            path = output_dir / f"{arm}_step_{index + 1:04d}.pt"
            _save_checkpoint(path, _resume_payload(model, optimizer, operator, arm=arm, step=index + 1, history=history, initial_losses=initial_losses, spec=spec, data=data, hashes=hashes, completed_arms=completed_arms))
            final_path, final_hash = path, file_sha256(path)
    return history, final_path, final_hash


def _load_completed(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, operator: FiberArrayOLS | FewModeFiberArrayOLS, *, spec: CanarySpec, data: CanaryData, hashes: Mapping[str, str], output_dir: Path, expected_sha256: str, expected_arm: str) -> dict[str, Any]:
    target = _resolve(path)
    if target.parent.resolve() != output_dir.resolve() or file_sha256(target) != expected_sha256:
        raise ValueError("completed FMF M=2 arm checkpoint path or hash is invalid")
    payload = safe_torch_load(target, expected_sha256=expected_sha256, map_location="cpu")
    if payload.get("schema") != RESUME_SCHEMA or payload.get("arm") != expected_arm or payload.get("step") != STEPS:
        raise ValueError("completed FMF M=2 arm checkpoint is invalid")
    if payload.get("evidence_hashes") != dict(hashes) or payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]:
        raise ValueError("completed FMF M=2 arm checkpoint lineage mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    operator.load_state_dict(payload["operator_state"], strict=True)
    base._restore_rng(payload["rng_state"])
    return dict(payload)


def _save_panel(path: Path, data: CanaryData, source_hashes: Mapping[str, str], m1_initial: Mapping[str, torch.Tensor], m2_initial: Mapping[str, torch.Tensor], m1_final: Mapping[str, torch.Tensor], m2_final: Mapping[str, torch.Tensor], mode_metrics: Mapping[str, Any]) -> str:
    tensors = {
        "indices": data.visual_indices.cpu(),
        "input_28": data.images.index_select(0, data.visual_indices).cpu(),
        "target_200": m1_initial["target"][:100].cpu(),
        "smf_m1_initial": m1_initial["camera"][:100].cpu(),
        "smf_m1_final": m1_final["camera"][:100].cpu(),
        "fmf_m2_initial": m2_initial["camera"][:100].cpu(),
        "fmf_m2_final": m2_final["camera"][:100].cpu(),
        "smf_m1_features": m1_initial["features"][:100].cpu(),
        "fmf_m2_features": m2_initial["features"][:100].cpu(),
    }
    payload = {"schema": PANEL_SCHEMA, "source_hashes": dict(source_hashes), "indices_are_global_mnist": True, "tensors": tensors, "mode_metrics": dict(mode_metrics)}
    atomic_torch_save(path, payload)
    try:
        from PIL import Image
        rows = []
        for key in ("target_200", "smf_m1_initial", "smf_m1_final", "fmf_m2_initial", "fmf_m2_final"):
            images = tensors[key].squeeze(1).clamp(0.0, 1.0).mul(255).to(torch.uint8)
            thumb = F.interpolate(images.unsqueeze(1).float(), size=(32, 32), mode="bilinear", align_corners=False).squeeze(1).to(torch.uint8)
            rows.append(torch.cat([thumb[i] for i in range(100)], dim=1))
        Image.fromarray(torch.cat(rows, 0).numpy(), mode="L").save(path.with_suffix(".png"))
    except ImportError:
        pass
    return file_sha256(path)


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("FMF M=2 training requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    _validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source, review_path)
    _preflight_payload, preflight_hash, training_hashes = _load_preflight(args.preflight_evidence, spec, formal, data, state, validation_hash, source, review_path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite FMF M=2 training evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for FMF M=2 training")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        reference_model, _reference_optimizer, reference_operator = _runtime(formal, source, device, "smf_m1_mse_control")
        reference_state = _state_cpu(reference_model)
        reference_formal_state = _state_cpu(reference_model.system.ols)
        reference_operator_state = _state_cpu(reference_operator)
        m1_initial_model, _m1_initial_optimizer, m1_initial_operator = _runtime(formal, source, device, "smf_m1_mse_control")
        m2_initial_model, _m2_initial_optimizer, m2_initial_operator = _runtime(formal, source, device, "fmf_m2_mse_canary")
        m1_initial_metrics, m1_initial_panel = _evaluate(m1_initial_model, m1_initial_operator, data, data.heldout_indices, device)
        m2_initial_metrics, m2_initial_panel = _evaluate(m2_initial_model, m2_initial_operator, data, data.heldout_indices, device)
        initial_losses = {"smf_m1_mse_control": _initial_losses(m1_initial_model, m1_initial_operator, data, device), "fmf_m2_mse_canary": _initial_losses(m2_initial_model, m2_initial_operator, data, device)}
        arm_results: dict[str, Any] = {}
        checkpoints: dict[str, Any] = {}
        resume_payload: Mapping[str, Any] | None = None
        resume_arm: str | None = None
        if args.resume_checkpoint:
            resume_target = _resolve(args.resume_checkpoint)
            if resume_target.parent.resolve() != output.parent.resolve():
                raise ValueError("explicit FMF M=2 resume belongs to a different attempt")
            resume_payload = safe_torch_load(resume_target, map_location="cpu")
            resume_arm = str(resume_payload.get("arm"))
            if resume_arm not in ARMS:
                raise ValueError("explicit FMF M=2 resume has invalid arm")

        def run_arm(arm: str, resume_path: Path | None = None, completed_arms: Mapping[str, Any] | None = None) -> None:
            model, optimizer, operator = _runtime(formal, source, device, arm)
            history: list[dict[str, Any]] = []
            start = 0
            safety = initial_losses[arm]
            if resume_path is not None:
                restored = _load_resume(resume_path, model, optimizer, operator, arm=arm, spec=spec, data=data, hashes=training_hashes, output_dir=output.parent)
                start = int(restored["step"])
                history = list(restored["history"])
                safety = [float(value) for value in restored["initial_batch_losses"]]
            history, final_path, final_hash = _run_steps(model, optimizer, operator, data, arm, start, history, safety, output.parent, spec, training_hashes, completed_arms)
            metrics, panel = _evaluate(model, operator, data, data.heldout_indices, device)
            arm_results[arm] = {"steps_completed": len(history), "history": history, "metrics": metrics, "final_checkpoint": str(final_path.resolve()) if final_path else None, "final_checkpoint_sha256": final_hash, "panel": panel}
            checkpoints[arm] = {"path": arm_results[arm]["final_checkpoint"], "sha256": final_hash}
            _clear_grad(model)
            torch.cuda.empty_cache()

        if resume_arm == "smf_m1_mse_control":
            if not isinstance(resume_payload, Mapping) or not isinstance(resume_payload.get("completed_arms"), Mapping):
                raise ValueError("M1 resume is missing completed M2 binding")
            completed = resume_payload["completed_arms"].get("fmf_m2_mse_canary")
            if not isinstance(completed, Mapping):
                raise ValueError("M1 resume has no completed M2 checkpoint")
            m2_model, m2_optimizer, m2_operator = _runtime(formal, source, device, "fmf_m2_mse_canary")
            completed_payload = _load_completed(completed["path"], m2_model, m2_optimizer, m2_operator, spec=spec, data=data, hashes=training_hashes, output_dir=output.parent, expected_sha256=completed["sha256"], expected_arm="fmf_m2_mse_canary")
            m2_metrics, m2_panel = _evaluate(m2_model, m2_operator, data, data.heldout_indices, device)
            arm_results["fmf_m2_mse_canary"] = {"steps_completed": STEPS, "history": list(completed_payload["history"]), "metrics": m2_metrics, "final_checkpoint": str(_resolve(completed["path"]).resolve()), "final_checkpoint_sha256": completed["sha256"], "panel": m2_panel}
            checkpoints["fmf_m2_mse_canary"] = {"path": arm_results["fmf_m2_mse_canary"]["final_checkpoint"], "sha256": completed["sha256"]}
            run_arm("smf_m1_mse_control", _resolve(args.resume_checkpoint), {"fmf_m2_mse_canary": checkpoints["fmf_m2_mse_canary"]})
        else:
            run_arm("fmf_m2_mse_canary", _resolve(args.resume_checkpoint) if resume_arm == "fmf_m2_mse_canary" else None)
            run_arm("smf_m1_mse_control", completed_arms={"fmf_m2_mse_canary": checkpoints["fmf_m2_mse_canary"]})
        panel_path = output.parent / "fmf_m2_comparison_panel.pt"
        panel_hash = _save_panel(panel_path, data, training_hashes, m1_initial_panel, m2_initial_panel, arm_results["smf_m1_mse_control"]["panel"], arm_results["fmf_m2_mse_canary"]["panel"], {arm: arm_results[arm]["metrics"].get("mode_power_fraction") for arm in ARMS})
        finite = all(result["metrics"]["finite"] and all(item["finite"] for item in result["history"]) for result in arm_results.values())
        all_steps = all(result["steps_completed"] == STEPS for result in arm_results.values())
        updates = all(smf_canary._four_phase_nonzero(item["update_norms"]) for result in arm_results.values() for item in result["history"])
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        gates = (
            GateResult("finite_values", finite, {}),
            GateResult("both_arms_completed_20_steps", all_steps, {"steps_per_arm": STEPS}),
            GateResult("four_phase_updates_nonzero", updates, {"layers": 4}),
            GateResult("source_and_formal_state_unchanged", _tree_equal(reference_state, _state_cpu(reference_model)) and _tree_equal(reference_formal_state, _state_cpu(reference_model.system.ols)) and _tree_equal(reference_operator_state, _state_cpu(reference_operator)), {}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("panel_provenance", panel_hash is not None, {"panel_sha256": panel_hash}),
            GateResult("safety_mse_bound", True, {"ratio": SAFETY_MSE_RATIO}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        record = EvidenceRecordV2(2, TRAINING_SCHEMA, "diagnostic", "diagnostic", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), training_hashes, gates, (NEXT_ACTION,))
        record.validate()
        payload = _record_payload(TRAINING_SCHEMA, record, {"outcome": "diagnostic_completed", "validation_hash": validation_hash, "preflight_hash": preflight_hash, "source_hashes": source.hashes, "training_hashes": training_hashes, "initial_metrics": {"smf_m1_mse_control": m1_initial_metrics, "fmf_m2_mse_canary": m2_initial_metrics}, "arms": {arm: {key: value for key, value in result.items() if key != "panel"} for arm, result in arm_results.items()}, "checkpoints": checkpoints, "panel_path": str(panel_path.resolve()), "panel_sha256": panel_hash, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "evaluator_calls": 0, "formal_candidate_checkpoint": False})
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    except _SafetyAbort as exc:
        raise RuntimeError(f"safety_aborted:{exc}") from exc
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "fmf-m2-integration-canary-validate":
        return validate(args)
    if args.research_action == "fmf-m2-integration-canary-preflight":
        return preflight(args)
    if args.research_action == "fmf-m2-integration-canary-train":
        return train(args)
    raise ValueError(f"unsupported FMF M=2 action: {args.research_action}")
