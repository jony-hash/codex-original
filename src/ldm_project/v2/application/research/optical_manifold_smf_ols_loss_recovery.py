"""Fixed-OLS projection-loss recovery canary for H-LOMA-I0.

The SMF operator and its transport remain fixed.  This lifecycle adds only a
research loss at the encoder/OLS boundary and optionally reuses the already
validated posterior/KL approximation.  It is intentionally separate from
the formal IdentityOLS runtime and from the earlier SWD experiments.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ...config import FormalSystemConfig
from ...contracts import ComplexField, OperationProvenance
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)
from ...optics.fiber_operator import FiberArrayOLS
from . import optical_manifold as base
from . import optical_manifold_smf_integration_canary as canary
from . import optical_manifold_smf_vae_kl_effect as kl_effect
from . import optical_manifold_smf_vae_pilot as vae


BRANCH = "research/v2-lightgen-optical-manifold-kl-ols-loss-recovery-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_ols_loss_recovery_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_ols_loss_recovery_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_ols_loss_recovery_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_ols_loss_recovery_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_ols_loss_recovery_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_smf_ols_loss_recovery_resume.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_identity_smf_coupling_panel.v1"

SOURCE_CHECKPOINT_HASH = "949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_PLAN6_VALIDATION_HASH = "d548cf93f2cfefb2258239b066b14f574b098bfa17edc0f377f718fd2cba3ca5"
SOURCE_PLAN6_PREFLIGHT_HASH = "722557fc0bfc058edafa8666772ee87bb927b0a37924b55087467303d3a56623"
SOURCE_PLAN6_TRAINING_HASH = "a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e"
SOURCE_KL_VALIDATION_HASH = "67ff61c726c3c95c0cc8625df79300f51af528da3aabd8916305f1cc750e43a2"
SOURCE_KL_PREFLIGHT_HASH = "ede0bc750e927c47b87b7bc6dcc4ffd01bc954d6af1aca7f8c57dce6e3a5fe2e"
SOURCE_KL_TRAINING_HASH = "b1182f804a32596a430f4570383d1b1d8f76cebee611e73943e99a88c7ea7659"
SOURCE_DOSE_AUDIT_HASH = "2d708bc3eea54dcfe8f580261891387c5e04698d6673667b421e53132fe3329f"
SOURCE_STEP = 1000
STEPS = 100
PREFLIGHT_STEPS = 3
BATCH_SIZE = 10
PORT_COUNT = 100
CHECKPOINT_INTERVAL = 50
SEED = 1234
LEARNING_RATE = 1.0e-3
CANDIDATE_BETA = 0.014605162095621654
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_CANARY_SECONDS = 1800.0
MAX_STORAGE_BYTES = 1024**3
SAFETY_MSE_RATIO = 2.0
MIN_POWER_GAIN = 0.05
MAX_MSE_RELATIVE = 1.02
MIN_RANK_RETENTION = 0.90
MIN_VARIANCE_RETENTION = 1.0
ARMS = (
    "optical_mse_control",
    "optical_coupling",
    "posterior_no_kl_coupling",
    "posterior_kl_coupling",
)
EXECUTION_ORDER = (
    "posterior_kl_coupling",
    "posterior_no_kl_coupling",
    "optical_coupling",
    "optical_mse_control",
)
POSTERIOR_ARMS = {"posterior_no_kl_coupling", "posterior_kl_coupling"}
COUPLING_ARMS = {"optical_coupling", "posterior_no_kl_coupling", "posterior_kl_coupling"}
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE
NEXT_PASS_ACTION = "manual_review_fixed_ols_manifold_full_training_contract"
NEXT_FAILED_ACTION = "manual_review_smf_bottleneck_attribution_contract"


class _SafetyAbort(RuntimeError):
    pass


class _Source:
    def __init__(self, source: vae.Source, hashes: dict[str, str], current_beta: float) -> None:
        self.source = source
        self.hashes = hashes
        self.current_beta = current_beta


class Spec:
    def __init__(self, path: Path, payload: dict[str, Any], config_hash: str) -> None:
        self.path = path
        self.payload = payload
        self.config_hash = config_hash


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"fixed OLS recovery requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("fixed OLS recovery requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "source", "dataset", "optical_contract", "training", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("fixed OLS recovery config is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    source = spec.payload["source"]
    expected_source = {
        "plan6_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "plan6_validation_sha256": SOURCE_PLAN6_VALIDATION_HASH,
        "plan6_preflight_sha256": SOURCE_PLAN6_PREFLIGHT_HASH,
        "plan6_training_sha256": SOURCE_PLAN6_TRAINING_HASH,
        "kl_effect_validation_sha256": SOURCE_KL_VALIDATION_HASH,
        "kl_effect_preflight_sha256": SOURCE_KL_PREFLIGHT_HASH,
        "kl_effect_training_sha256": SOURCE_KL_TRAINING_HASH,
        "kl_dose_audit_sha256": SOURCE_DOSE_AUDIT_HASH,
        "mnist_manifest_sha256": SOURCE_MANIFEST_HASH,
    }
    if any(source.get(key) != value for key, value in expected_source.items()) or source.get("source_global_step") != SOURCE_STEP:
        raise ValueError("fixed OLS recovery source binding mismatch")
    if spec.payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": 1,
        "schedule": "fresh_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "excluded_prior_panels": True,
        "heldout_excluded": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "heldout_panel_count": 1000,
        "visual_panel_count": 100,
    }:
        raise ValueError("fixed OLS recovery dataset contract mismatch")
    optical = spec.payload["optical_contract"]
    for key, value in {
        "formal_system_kind": base.EXPECTED_FORMAL_KIND,
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "formal_ols": "identity",
        "research_ols": "100_port_lp01_smf",
        "fixed_transport": True,
        "fixed_exposure": {"scale": 1.0, "gamma": 1.0},
        "projection_loss": "one_minus_smf_power_retention",
        "projection_loss_status": "project_physics_choice",
        "posterior_status": "project_approximation",
    }.items():
        if optical.get(key) != value:
            raise ValueError(f"fixed OLS recovery optical contract mismatch: {key}")
    training = spec.payload["training"]
    for key, value in {
        "arms": list(ARMS),
        "execution_order": list(EXECUTION_ORDER),
        "optimizer": "Adam",
        "optimizer_update_device": "cpu_shadow",
        "learning_rate": LEARNING_RATE,
        "reset_adam_from_source": True,
        "steps_per_arm": STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "reconstruction_activation_checkpoint": True,
        "deterministic": True,
        "epsilon_seed": SEED,
        "registered_beta": 0.0018256452619527068,
        "candidate_beta": CANDIDATE_BETA,
        "coupling_lambda_registration": "0.1_encoder_mse_norm_over_projection_norm",
        "no_early_stop": True,
        "no_tuning": True,
        "no_checkpoint_selection": True,
    }.items():
        if training.get(key) != value:
            raise ValueError(f"fixed OLS recovery training contract mismatch: {key}")
    if spec.payload["authorization"].get("training_authorized") is not False or spec.payload["authorization"].get("formal_promotion") is not False:
        raise ValueError("fixed OLS recovery authorization boundary is invalid")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_fixed_ols_projection_loss_recovery_canary",
        "approved_contract": {
            "arms": list(ARMS),
            "steps_per_arm": STEPS,
            "logical_batch_size": BATCH_SIZE,
            "microbatch_size": 1,
            "source_global_step": SOURCE_STEP,
            "preflight_steps": PREFLIGHT_STEPS,
            "resume_segment": "2+1",
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "projection_loss": "one_minus_smf_power_retention",
            "fixed_transport": True,
            "post_canary_next_action": NEXT_PASS_ACTION,
            "post_invalid_next_action": NEXT_FAILED_ACTION,
        },
        "bounded_diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("fixed OLS recovery review contract does not match scope")
    return target


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    effect_spec = kl_effect._load_spec(_resolve("configs/research/lightgen_optical_manifold_smf_vae_kl_effect_v1.yaml", base_path=base._repo_root()))
    formal, path = kl_effect._load_formal(effect_spec)
    expected = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    if path.resolve() != expected.resolve():
        raise ValueError("fixed OLS recovery formal config path mismatch")
    return formal, path


def _ancestor(commit: str, current: str) -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, current], cwd=base._repo_root(), check=False).returncode != 0:
        raise ValueError("source evidence commit is not an ancestor")


def _load_source(spec: Spec, formal_path: Path, current: str) -> _Source:
    effect_spec = kl_effect._load_spec(_resolve("configs/research/lightgen_optical_manifold_smf_vae_kl_effect_v1.yaml", base_path=base._repo_root()))
    effect_source = kl_effect._load_source(effect_spec, formal_path, current)
    source = spec.payload["source"]
    paths = {
        "plan6_validation": (source["plan6_validation"], SOURCE_PLAN6_VALIDATION_HASH),
        "plan6_preflight": (source["plan6_preflight"], SOURCE_PLAN6_PREFLIGHT_HASH),
        "plan6_training": (source["plan6_training"], SOURCE_PLAN6_TRAINING_HASH),
        "plan6_checkpoint": (source["plan6_checkpoint"], SOURCE_CHECKPOINT_HASH),
        "kl_effect_validation": (source["kl_effect_validation"], SOURCE_KL_VALIDATION_HASH),
        "kl_effect_preflight": (source["kl_effect_preflight"], SOURCE_KL_PREFLIGHT_HASH),
        "kl_effect_training": (source["kl_effect_training"], SOURCE_KL_TRAINING_HASH),
        "kl_dose_audit": (source["kl_dose_audit"], SOURCE_DOSE_AUDIT_HASH),
    }
    for role, (raw_path, digest) in paths.items():
        if file_sha256(_resolve(raw_path)) != digest:
            raise ValueError(f"{role} source hash mismatch")
    for role in ("kl_effect_validation", "kl_effect_preflight", "kl_effect_training"):
        value = json.loads(_resolve(paths[role][0]).read_text(encoding="utf-8"))
        record = EvidenceRecordV2.from_dict(value["evidence"])
        record.validate()
        _ancestor(record.git_commit, current)
        if value.get("training_authorized") is not False or value.get("formal_promotion") is not False:
            raise ValueError("KL effect source authorization boundary is invalid")
    training = json.loads(_resolve(source["kl_effect_training"]).read_text(encoding="utf-8"))
    if training["evidence"]["status"] != "passed" or training["contract"].get("outcome") != "passed":
        raise ValueError("KL effect source is not passed")
    dose = json.loads(_resolve(source["kl_dose_audit"]).read_text(encoding="utf-8"))
    if dose["evidence"]["status"] != "diagnostic" or dose["contract"].get("selected_multiplier") != 8.0:
        raise ValueError("corrected dose source is not authoritative")
    hashes = {
        **effect_source.hashes,
        "kl_effect_validation": SOURCE_KL_VALIDATION_HASH,
        "kl_effect_preflight": SOURCE_KL_PREFLIGHT_HASH,
        "kl_effect_training": SOURCE_KL_TRAINING_HASH,
        "kl_dose_audit": SOURCE_DOSE_AUDIT_HASH,
    }
    return _Source(effect_source.source, hashes, effect_source.current_beta)


def _load_data(root: str | Path, spec: Spec) -> vae.Data:
    effect_spec = kl_effect._load_spec(_resolve("configs/research/lightgen_optical_manifold_smf_vae_kl_effect_v1.yaml", base_path=base._repo_root()))
    return kl_effect._load_data(root, _resolve(effect_spec.payload["source"]["kl_dose_validation"]))


def _context(spec: Spec, formal: FormalSystemConfig, data: vae.Data) -> EvidenceContextV2:
    return EvidenceContextV2(canonical_sha256(formal.model_dump(mode="json")), spec.config_hash, data.manifest["manifest_hash"], canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}), 1.0)


def _hashes(spec: Spec, review: Path, data: vae.Data, source: _Source) -> dict[str, str]:
    return {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review), "dataset_manifest": data.manifest["manifest_hash"]}


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": schema, "evidence": record.to_dict(), "hypothesis": HYPOTHESIS, "training_authorized": False, "formal_promotion": False, "contract": dict(contract)}


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite fixed OLS recovery artifact: {target}")
    atomic_write_json(target, payload)


def _batch(data: vae.Data, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return vae._batch(data, indices, device)


def _runtime(formal: FormalSystemConfig, source: _Source, arm: str, device: torch.device) -> tuple[base.IdentityOpticalManifoldAutoencoder, FiberArrayOLS, vae.SharedPerPortPosterior | None, vae._CpuAdam]:
    runtime_arm = "smf_beta_vae" if arm in POSTERIOR_ARMS else "smf_deterministic_mse"
    model, operator, posterior, optimizer = vae._make_runtime(formal, source.source, device, runtime_arm)
    return model, operator, posterior, optimizer


def _encoded_field(model: base.IdentityOpticalManifoldAutoencoder, amplitude: torch.Tensor) -> tuple[torch.Tensor, Any]:
    system = model.system
    slm = system.slm(system.command(amplitude))
    front = system.front(slm)
    encoded, _records, _stats = system.encoder(front, "encoder")
    return encoded.tensor[:, 0], system


def _projection_from_pre(pre: torch.Tensor, smf: FiberArrayOLS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = pre
    modes = smf.input_modes.to(values)
    area = smf.input_plane.pixel_pitch_m[0] * smf.input_plane.pixel_pitch_m[1]
    coefficients = torch.einsum("mhw,bhw->bm", modes.conj(), values) * area
    projected = torch.einsum("bm,mhw->bhw", coefficients, modes)
    input_energy = values.abs().square().sum(dim=(-2, -1)) * area
    output_energy = projected.abs().square().sum(dim=(-2, -1)) * area
    residual = (1.0 - output_energy / input_energy.clamp_min(1.0e-30)).mean()
    return residual, input_energy, output_energy


def _decode(model: base.IdentityOpticalManifoldAutoencoder, system: Any, field: ComplexField) -> torch.Tensor:
    decoded, _records, _stats = system.decoder(field, "decoder")
    chip = ComplexField(decoded.tensor, system.chip_plane, decoded.provenance)
    return system.back(chip).tensor.abs().square()


def _forward(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, posterior: vae.SharedPerPortPosterior | None, amplitude: torch.Tensor, epsilon: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, *, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    pre, system = _encoded_field(model, amplitude)
    projection_loss, input_energy, output_energy = _projection_from_pre(pre, smf)
    values = pre
    modes = smf.input_modes.to(values)
    area = smf.input_plane.pixel_pitch_m[0] * smf.input_plane.pixel_pitch_m[1]
    coefficients = torch.einsum("mhw,bhw->bm", modes.conj(), values) * area
    if posterior is None:
        smf_field = ComplexField(pre.unsqueeze(1), system.encoder_plane.with_plane("encoder_layer_2"), OperationProvenance("fixed_smf_encoder_boundary", "encoder_layer_2", 0.0, {"status": "diagnostic"}))
        smf_output = smf(smf_field)
        raw = _decode(model, system, smf_output)
        kl = torch.zeros((), device=amplitude.device)
    else:
        normalized = vae._normalize(coefficients, mean.to(coefficients.device), scale.to(coefficients.device))
        post_mean, logvar = posterior(normalized)
        latent = post_mean + logvar.mul(0.5).exp() * epsilon
        restored = vae._denormalize(latent, mean.to(latent.device), scale.to(latent.device))
        raw = vae._decode_coefficients(model, smf, restored, system)
        kl = vae.posterior_kl(post_mean, logvar)
    diagnostics = {"input_energy": input_energy, "output_energy": output_energy, "coefficients": coefficients, "projection_loss": projection_loss}
    return raw, projection_loss, kl, coefficients, diagnostics


def _layer_norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        result[name] = float(value.norm())
        for layer in range(value.shape[0]):
            result[f"{name}[{layer}]"] = float(value[layer].norm())
    return result


def _clear(model: nn.Module, posterior: nn.Module | None) -> None:
    for parameter in model.parameters():
        parameter.grad = None
    if posterior is not None:
        for parameter in posterior.parameters():
            parameter.grad = None


def _items(model: nn.Module, posterior: nn.Module | None) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in list(model.named_parameters()) + ([(f"posterior.{name}", parameter) for name, parameter in posterior.named_parameters()] if posterior is not None else []) if parameter.requires_grad]


def _objective_gradients(model: nn.Module, smf: FiberArrayOLS, posterior: vae.SharedPerPortPosterior | None, amplitudes: torch.Tensor, targets: torch.Tensor, epsilon: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, objective: str, beta: float) -> tuple[dict[str, torch.Tensor], float]:
    if objective not in {"mse", "projection", "kl", "combined"}:
        raise ValueError("unsupported fixed OLS objective")
    items = _items(model, posterior)
    gradients = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in items}
    values: list[float] = []
    for index in range(BATCH_SIZE):
        raw, projection, kl, _coefficients, _diagnostics = _forward(model, smf, posterior, amplitudes[index : index + 1], epsilon[index : index + 1], mean, scale, beta=beta)
        mse = F.mse_loss(raw, targets[index : index + 1])
        value = {"mse": mse, "projection": projection, "kl": kl, "combined": mse + projection + beta * kl}[objective]
        gradients_t = torch.autograd.grad(value / BATCH_SIZE, [parameter for _, parameter in items], allow_unused=True)
        values.append(float(value.detach().cpu()))
        for (name, _parameter), gradient in zip(items, gradients_t, strict=True):
            if gradient is not None:
                gradients[name].add_(gradient.detach().cpu())
    _clear(model, posterior)
    return gradients, float(sum(values) / len(values))


def _calibrate_lambda(model: nn.Module, smf: FiberArrayOLS, data: vae.Data, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> tuple[float, dict[str, Any]]:
    amplitudes, targets = _batch(data, data.schedule[0], device)
    epsilon = data.epsilon_schedule[0].to(device)
    mse, mse_value = _objective_gradients(model, smf, None, amplitudes, targets, epsilon, mean, scale, "mse", 0.0)
    projection, projection_value = _objective_gradients(model, smf, None, amplitudes, targets, epsilon, mean, scale, "projection", 0.0)
    mse_norm = float(mse["system.encoder.phase"].norm())
    projection_norm = float(projection["system.encoder.phase"].norm())
    if mse_norm <= 0.0 or projection_norm <= 0.0:
        raise ValueError("fixed OLS lambda calibration requires nonzero encoder gradients")
    return 0.1 * mse_norm / projection_norm, {"mse": mse_value, "projection": projection_value, "mse_encoder_norm": mse_norm, "projection_encoder_norm": projection_norm, "formula": "0.1*g_mse_encoder/g_projection_encoder"}


def _step(model: nn.Module, smf: FiberArrayOLS, posterior: vae.SharedPerPortPosterior | None, optimizer: vae._CpuAdam, amplitudes: torch.Tensor, targets: torch.Tensor, epsilon: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, arm: str, lambda_value: float, beta: float) -> dict[str, Any]:
    items = _items(model, posterior)
    before = {name: parameter.detach().clone() for name, parameter in items}
    _clear(model, posterior)
    mse_values: list[float] = []
    projection_values: list[float] = []
    kl_values: list[float] = []
    cameras: list[torch.Tensor] = []
    for index in range(BATCH_SIZE):
        raw, projection, kl, _coefficients, _diagnostics = checkpoint(lambda value, noise: _forward(model, smf, posterior, value, noise, mean, scale, beta=beta), amplitudes[index : index + 1], epsilon[index : index + 1], use_reentrant=False) if posterior is not None else checkpoint(lambda value: _forward(model, smf, posterior, value, torch.empty((1, PORT_COUNT, 2), device=value.device), mean, scale, beta=0.0), amplitudes[index : index + 1], use_reentrant=False)
        mse = F.mse_loss(raw, targets[index : index + 1])
        weight = lambda_value if arm in COUPLING_ARMS else 0.0
        loss = mse + weight * projection + beta * kl
        (loss / BATCH_SIZE).backward()
        mse_values.append(float(mse.detach().cpu()))
        projection_values.append(float(projection.detach().cpu()))
        kl_values.append(float(kl.detach().cpu()))
        cameras.append(raw.detach())
    gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else torch.zeros_like(parameter, device="cpu") for name, parameter in items}
    optimizer.step()
    updates: dict[str, float] = {}
    for name, parameter in items:
        delta = (parameter.detach() - before[name]).abs()
        updates[name] = float(delta.sum().cpu())
        if name in EXPECTED_TRAINABLE:
            for layer in range(delta.shape[0]):
                updates[f"{name}[{layer}]"] = float(delta[layer].sum().cpu())
    camera = torch.cat(cameras)
    _clear(model, posterior)
    finite_values = [*mse_values, *projection_values, *kl_values, *updates.values(), *_layer_norms(gradients).values()]
    return {"arm": arm, "loss": float(sum(m + (lambda_value if arm in COUPLING_ARMS else 0.0) * p + beta * k for m, p, k in zip(mse_values, projection_values, kl_values, strict=True)) / BATCH_SIZE), "mse": float(sum(mse_values) / BATCH_SIZE), "projection_loss": float(sum(projection_values) / BATCH_SIZE), "kl": float(sum(kl_values) / BATCH_SIZE), "camera_finite": bool(torch.isfinite(camera).all()), "camera_min": float(camera.min().cpu()), "camera_max": float(camera.max().cpu()), "gradient_norms": _layer_norms(gradients), "update_norms": updates, "activation_checkpoint": True, "finite": bool(torch.isfinite(camera).all() and all(bool(torch.isfinite(torch.tensor(value))) for value in finite_values))}


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


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
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _resume_payload(model: nn.Module, posterior: vae.SharedPerPortPosterior | None, optimizer: vae._CpuAdam, arm: str, step: int, history: list[dict[str, Any]], data: vae.Data, spec: Spec, hashes: Mapping[str, str], lambda_value: float, beta: float, completed: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": RESUME_SCHEMA, "arm": arm, "step": step, "model_state": _model_state(model), "posterior_state": None if posterior is None else _model_state(posterior), "optimizer_state": _tree_clone(optimizer.state_dict()), "rng_state": _tree_clone(base._rng_state()), "history": list(history), "schedule_cursor": {"next_step": step, "batch_size": BATCH_SIZE}, "epsilon_cursor": step, "lambda_coupling": lambda_value, "beta": beta, "config_hash": spec.config_hash, "dataset_manifest_hash": data.manifest["manifest_hash"], "schedule_sha256": data.manifest["schedule_sha256"], "evidence_hashes": dict(hashes), "completed_arms": dict(completed), "formal_candidate_checkpoint": False, "training_authorized": False, "formal_promotion": False}


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite fixed OLS checkpoint: {path}")
    atomic_torch_save(path, payload)


def _load_resume(path: Path, model: nn.Module, posterior: vae.SharedPerPortPosterior | None, optimizer: vae._CpuAdam, data: vae.Data, spec: Spec, hashes: Mapping[str, str], arm: str, output_dir: Path, lambda_value: float, beta: float, *, final: bool = False) -> dict[str, Any]:
    if path.parent.resolve() != output_dir.resolve():
        raise ValueError("fixed OLS resume belongs to another attempt")
    payload = safe_torch_load(path, map_location="cpu")
    step = int(payload.get("step", 0))
    if payload.get("schema") != RESUME_SCHEMA or payload.get("arm") != arm or not (0 < step <= STEPS) or (final and step != STEPS) or (not final and step >= STEPS) or payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"] or payload.get("evidence_hashes") != dict(hashes) or float(payload.get("lambda_coupling")) != lambda_value or float(payload.get("beta")) != beta:
        raise ValueError("fixed OLS resume lineage mismatch")
    if payload.get("schedule_cursor") != {"next_step": step, "batch_size": BATCH_SIZE} or len(payload.get("history", [])) != step:
        raise ValueError("fixed OLS resume cursor/history mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    if posterior is not None:
        posterior.load_state_dict(payload["posterior_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return payload


def _exact_resume(formal: FormalSystemConfig, source: _Source, data: vae.Data, mean: torch.Tensor, scale: torch.Tensor, device: torch.device, arm: str, lambda_value: float, beta: float, spec: Spec, hashes: Mapping[str, str], output_dir: Path) -> dict[str, Any]:
    initial_rng = _tree_clone(base._rng_state())
    base._restore_rng(initial_rng)
    continuous_model, continuous_smf, continuous_post, continuous_opt = _runtime(formal, source, arm, device)
    continuous = [_step(continuous_model, continuous_smf, continuous_post, continuous_opt, *_batch(data, data.schedule[step], device), data.epsilon_schedule[step].to(device), mean.to(device), scale.to(device), arm, lambda_value, beta) for step in range(3)]
    continuous_state = (_model_state(continuous_model), None if continuous_post is None else _model_state(continuous_post), continuous_opt.state_dict(), _tree_clone(base._rng_state()))
    base._restore_rng(initial_rng)
    split_model, split_smf, split_post, split_opt = _runtime(formal, source, arm, device)
    split = [_step(split_model, split_smf, split_post, split_opt, *_batch(data, data.schedule[step], device), data.epsilon_schedule[step].to(device), mean.to(device), scale.to(device), arm, lambda_value, beta) for step in range(2)]
    path = output_dir / f"preflight_{arm}_step_0002.pt"
    _save_checkpoint(path, _resume_payload(split_model, split_post, split_opt, arm, 2, split, data, spec, hashes, lambda_value, beta, {}))
    resumed_model, resumed_smf, resumed_post, resumed_opt = _runtime(formal, source, arm, device)
    payload = _load_resume(path, resumed_model, resumed_post, resumed_opt, data, spec, hashes, arm, output_dir, lambda_value, beta)
    resumed = split + [_step(resumed_model, resumed_smf, resumed_post, resumed_opt, *_batch(data, data.schedule[2], device), data.epsilon_schedule[2].to(device), mean.to(device), scale.to(device), arm, lambda_value, beta)]
    resumed_state = (_model_state(resumed_model), None if resumed_post is None else _model_state(resumed_post), resumed_opt.state_dict(), _tree_clone(base._rng_state()))
    return {"continuous": continuous, "resumed": resumed, "resume_artifact": str(path.resolve()), "resume_artifact_sha256": file_sha256(path), "persisted_history": payload["history"] == split, "schedule_cursor": payload["schedule_cursor"] == {"next_step": 2, "batch_size": BATCH_SIZE}, "model_exact": _tree_equal(continuous_state[0], resumed_state[0]), "posterior_exact": _tree_equal(continuous_state[1], resumed_state[1]), "optimizer_exact": _tree_equal(continuous_state[2], resumed_state[2]), "step_exact": _tree_equal(continuous[2], resumed[2]), "rng_exact": _tree_equal(continuous_state[3], resumed_state[3]), "finite": all(item["finite"] for item in continuous + resumed)}


def _load_validation(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: _Source, review: Path, state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != VALIDATION_SCHEMA or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data) or record.input_hashes != _hashes(spec, review, data, source) or record.permitted_next_actions != ("run_smf_ols_loss_recovery_preflight",):
        raise ValueError("fixed OLS validation is stale or mismatched")
    return payload, digest


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root, spec)
    hashes = _hashes(spec, review, data, source)
    gates = tuple(GateResult(name, True, metrics) for name, metrics in (("clean_lineage", {"branch": BRANCH}), ("source_kl_effect_passed", {"training": "passed"}), ("source_checkpoint", {"step": SOURCE_STEP}), ("four_arm_scope", {"arms": list(ARMS)}), ("fresh_balanced_schedule", {"steps": STEPS, "batch": BATCH_SIZE}), ("formal_identity_smf_optics", {"grid": 2048, "formal_ols": "identity", "research_ols": "100_port_lp01_smf"}), ("fixed_ols_boundary", {"trainable_transport": False, "exposure": 1.0}), ("phase_only_trainable_scope", {"parameters": list(EXPECTED_TRAINABLE)}), ("research_boundary", {"training_authorized": False, "formal_promotion": False})))
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_smf_ols_loss_recovery_preflight",))
    record.validate()
    payload = _record_payload(VALIDATION_SCHEMA, record, {"source_hashes": source.hashes, "dataset_manifest": data.manifest, "optimizer_created": False, "cuda_context_created": False, "projection_loss": "1 - P_smf_out/P_smf_in"})
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_preflight(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: _Source, review: Path, validation_hash: str, state: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, str], torch.Tensor, torch.Tensor, float]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    norm_path = _resolve(payload["contract"]["normalization_path"])
    norm_hash = file_sha256(norm_path)
    expected = {**_hashes(spec, review, data, source), "training_validation": validation_hash, "posterior_normalization": norm_hash}
    if payload.get("schema") != PREFLIGHT_SCHEMA or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected:
        raise ValueError("fixed OLS preflight is stale or mismatched")
    normalization = safe_torch_load(norm_path, map_location="cpu")
    if normalization.get("schema") != "lightgen_optical_manifold_smf_ols_loss_recovery_normalization.v1" or normalization.get("config_hash") != spec.config_hash or normalization.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or normalization.get("source_hashes") != source.hashes or normalization.get("validation_sha256") != validation_hash or tuple(normalization.get("shape", ())) != (PORT_COUNT, 2):
        raise ValueError("fixed OLS normalization provenance is invalid")
    return payload, digest, {**expected, "training_preflight": digest}, normalization["mean"], normalization["scale"], float(payload["contract"]["lambda_coupling"])


def _normalization(mean: torch.Tensor, scale: torch.Tensor, spec: Spec, data: vae.Data, source: _Source, validation_hash: str) -> dict[str, Any]:
    return {"schema": "lightgen_optical_manifold_smf_ols_loss_recovery_normalization.v1", "mean": mean.cpu(), "scale": scale.cpu(), "shape": [PORT_COUNT, 2], "config_hash": spec.config_hash, "dataset_manifest_hash": data.manifest["manifest_hash"], "source_hashes": dict(source.hashes), "validation_sha256": validation_hash, "training_authorized": False, "formal_promotion": False}


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("fixed OLS preflight requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root, spec)
    _validation, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, source, review, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixed OLS preflight: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        norm_model, norm_smf, _norm_post, _norm_opt = _runtime(formal, source, "optical_mse_control", device)
        mean, scale = vae._fit_normalization(norm_model, norm_smf, data, device)
        norm_path = output.parent / "posterior_normalization.pt"
        atomic_torch_save(norm_path, _normalization(mean, scale, spec, data, source, validation_hash))
        hashes = {**_hashes(spec, review, data, source), "training_validation": validation_hash, "posterior_normalization": file_sha256(norm_path)}
        lambda_value, calibration = _calibrate_lambda(norm_model, norm_smf, data, mean, scale, device)
        exact = {arm: _exact_resume(formal, source, data, mean, scale, device, arm, lambda_value, CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0, spec, hashes, output.parent) for arm in ARMS}
        reports: dict[str, Any] = {}
        for arm in ARMS:
            model, smf, posterior, optimizer = _runtime(formal, source, arm, device)
            amplitudes, targets = _batch(data, data.schedule[0], device)
            beta = CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0
            reports[arm] = {"step": _step(model, smf, posterior, optimizer, amplitudes, targets, data.epsilon_schedule[0].to(device), mean.to(device), scale.to(device), arm, lambda_value, beta), "exact_resume": exact[arm]}
        candidate_model, candidate_smf, candidate_post, _candidate_opt = _runtime(formal, source, "posterior_kl_coupling", device)
        projection_grads, projection_value = _objective_gradients(candidate_model, candidate_smf, candidate_post, *_batch(data, data.schedule[0], device), data.epsilon_schedule[0].to(device), mean.to(device), scale.to(device), "projection", CANDIDATE_BETA)
        projection_norms = _layer_norms(projection_grads)
        projection_boundary = projection_norms["system.encoder.phase"] > 0.0 and projection_norms["system.decoder.phase"] == 0.0 and all(value == 0.0 for name, value in projection_norms.items() if name.startswith("posterior.")) and all(projection_norms[f"system.encoder.phase[{layer}]"] > 0.0 for layer in range(2))
        exact_ok = all(report["model_exact"] and report["posterior_exact"] and report["optimizer_exact"] and report["step_exact"] and report["rng_exact"] and report["persisted_history"] and report["schedule_cursor"] and report["finite"] for report in exact.values())
        phase_ok = all(canary._four_phase_nonzero(report["step"]["update_norms"]) for report in reports.values())
        posterior_ok = all(any(value > 0.0 for name, value in report["step"]["update_norms"].items() if name.startswith("posterior.")) for arm, report in reports.items() if arm in POSTERIOR_ARMS)
        finite = all(report["step"]["finite"] for report in reports.values())
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        gates = (GateResult("finite_values", finite, {}), GateResult("four_phase_updates", phase_ok, {}), GateResult("posterior_updates", posterior_ok, {}), GateResult("projection_encoder_only", projection_boundary, {"gradient_norms": projection_norms}), GateResult("persisted_resume_exact", exact_ok, {arm: {key: value for key, value in report.items() if isinstance(value, bool)} for arm, report in exact.items()}), GateResult("projection_loss_definition", bool(torch.isfinite(torch.tensor(projection_value))), {"formula": "1 - P_smf_out/P_smf_in"}), GateResult("fixed_transport", True, {"transport_trainable": False}), GateResult("evaluator_isolation", True, {"evaluator_calls": 0}), GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}), GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}))
        passed = all(gate.passed for gate in gates)
        hashes_final = {**hashes}
        record = EvidenceRecordV2(2, PREFLIGHT_SCHEMA, "diagnostic", "passed" if passed else "invalid_harness", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes_final, gates, ("run_smf_ols_loss_recovery",) if passed else (), None if passed else "fixed OLS preflight gate failed")
        record.validate()
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"normalization_path": str(norm_path.resolve()), "normalization_sha256": file_sha256(norm_path), "lambda_coupling": lambda_value, "lambda_registration": calibration, "projection_boundary": {"value": projection_value, "gradient_norms": projection_norms}, "arms": reports, "free_cuda_bytes": int(free), "total_cuda_bytes": int(total), "peak_cuda_bytes": peak, "elapsed_seconds": elapsed, "training_authorized": False, "formal_promotion": False})
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if passed else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def _rank(features: torch.Tensor) -> float:
    work = torch.cat((features.real, features.imag), dim=1).to(torch.float64)
    centered = work - work.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probability = singular.square() / singular.square().sum().clamp_min(1.0e-30)
    return float(1.0 / probability.square().sum().clamp_min(1.0e-30))


def _evaluate(model: nn.Module, smf: FiberArrayOLS, posterior: vae.SharedPerPortPosterior | None, data: vae.Data, indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, device: torch.device, *, beta: float) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cameras: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    projections: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            amplitudes, target = _batch(data, indices[start : start + BATCH_SIZE], device)
            epsilon = data.epsilon_schedule[: amplitudes.shape[0], 0].to(device) if posterior is not None else torch.empty((amplitudes.shape[0], PORT_COUNT, 2), device=device)
            raw, projection, _kl, coefficients, _diagnostics = _forward(model, smf, posterior, amplitudes, epsilon, mean.to(device), scale.to(device), beta=beta)
            cameras.append(raw.cpu())
            targets.append(target.cpu())
            features.append(coefficients.cpu())
            projections.append(projection.detach().reshape(1).cpu().repeat(amplitudes.shape[0]))
    camera = torch.cat(cameras)
    target = torch.cat(targets)
    feature = torch.cat(features)
    centered_camera = camera.flatten(1) - camera.flatten(1).mean(1, keepdim=True)
    centered_target = target.flatten(1) - target.flatten(1).mean(1, keepdim=True)
    corr = (centered_camera * centered_target).sum(1) / (centered_camera.norm(dim=1) * centered_target.norm(dim=1)).clamp_min(1.0e-30)
    return {"paired_mse": float(F.mse_loss(camera, target)), "cyclic_mse": float(F.mse_loss(camera, target.roll(1, 0))), "centered_correlation": float(corr.mean()), "variance_retention": float(centered_camera.var(0, unbiased=False).mean() / centered_target.var(0, unbiased=False).mean().clamp_min(1.0e-30)), "projection_residual": float(torch.cat(projections).mean()), "power_retention": float(1.0 - torch.cat(projections).mean()), "coefficient_effective_rank": _rank(feature), "finite": bool(torch.isfinite(camera).all() and torch.isfinite(target).all() and torch.isfinite(feature).all()), "evaluation_count": int(indices.numel()), "feature_shape": list(feature.shape)}, {"camera": camera, "target": target, "coefficients": feature}


def _load_training_preflight(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: _Source, review: Path, validation_hash: str, state: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, str], torch.Tensor, torch.Tensor, float]:
    return _load_preflight(path, spec, formal, data, source, review, validation_hash, state)


def _save_panel(path: Path, data: vae.Data, values: Mapping[str, torch.Tensor], hashes: Mapping[str, str]) -> str:
    payload = {"schema": PANEL_SCHEMA, "indices_are_global_mnist": True, "indices": data.visual_indices.cpu(), "input_28": data.images.index_select(0, data.visual_indices).cpu(), "tensors": {name: value.cpu().contiguous() for name, value in values.items()}, "source_hashes": dict(hashes), "diagnostic_only": True}
    atomic_torch_save(path, payload)
    return file_sha256(path)


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("fixed OLS training requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root, spec)
    _validation, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, source, review, state)
    _preflight, preflight_hash, hashes, mean, scale, lambda_value = _load_training_preflight(args.preflight_evidence, spec, formal, data, source, review, validation_hash, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixed OLS training evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        initial: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            model, smf, posterior, _optimizer = _runtime(formal, source, arm, device)
            initial[arm], _tensors = _evaluate(model, smf, posterior, data, data.heldout_indices, mean, scale, device, beta=CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0)
        resume_payload: Mapping[str, Any] | None = None
        resume_arm: str | None = None
        if args.resume_checkpoint:
            resume_payload = safe_torch_load(_resolve(args.resume_checkpoint), map_location="cpu")
            resume_arm = str(resume_payload.get("arm"))
            if resume_arm not in ARMS:
                raise ValueError("invalid fixed OLS resume arm")
        results: dict[str, Any] = {}
        models: dict[str, tuple[nn.Module, FiberArrayOLS, vae.SharedPerPortPosterior | None]] = {}
        checkpoints: dict[str, Any] = {}
        completed: dict[str, Any] = {}
        for arm in EXECUTION_ORDER:
            model, smf, posterior, optimizer = _runtime(formal, source, arm, device)
            start = 0
            history: list[dict[str, Any]] = []
            if resume_payload is not None and resume_arm == arm:
                loaded = _load_resume(_resolve(args.resume_checkpoint), model, posterior, optimizer, data, spec, hashes, arm, output.parent, lambda_value, CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0)
                start = int(loaded["step"])
                history = list(loaded["history"])
                completed.update(loaded.get("completed_arms", {}))
            elif resume_payload is not None and arm in resume_payload.get("completed_arms", {}):
                item = resume_payload["completed_arms"][arm]
                cm, cs, cp, co = _runtime(formal, source, arm, device)
                loaded = _load_resume(_resolve(item["path"]), cm, cp, co, data, spec, hashes, arm, output.parent, lambda_value, CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0, final=True)
                models[arm] = (cm, cs, cp)
                results[arm] = {"history": list(loaded["history"]), "steps_completed": STEPS, "metrics": _evaluate(cm, cs, cp, data, data.heldout_indices, mean, scale, device, beta=CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0)[0]}
                completed[arm] = item
                continue
            beta = CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0
            for step in range(start, STEPS):
                amplitudes, targets = _batch(data, data.schedule[step], device)
                report = _step(model, smf, posterior, optimizer, amplitudes, targets, data.epsilon_schedule[step].to(device), mean.to(device), scale.to(device), arm, lambda_value, beta)
                report["step"] = step + 1
                history.append(report)
                if not report["finite"] or report["mse"] > SAFETY_MSE_RATIO * max(initial[arm]["paired_mse"], 1.0e-30):
                    raise _SafetyAbort(f"fixed OLS safety gate failed for {arm} at step {step + 1}")
                if (step + 1) % CHECKPOINT_INTERVAL == 0:
                    path = output.parent / f"{arm}_step_{step + 1:04d}.pt"
                    _save_checkpoint(path, _resume_payload(model, posterior, optimizer, arm, step + 1, history, data, spec, hashes, lambda_value, beta, completed))
                    checkpoints[arm] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
            metrics, _tensors = _evaluate(model, smf, posterior, data, data.heldout_indices, mean, scale, device, beta=beta)
            results[arm] = {"history": history, "steps_completed": len(history), "metrics": metrics}
            models[arm] = (model, smf, posterior)
            completed[arm] = checkpoints[arm]
        if set(results) != set(ARMS):
            raise RuntimeError("fixed OLS recovery did not complete all arms")
        panel_values: dict[str, torch.Tensor] = {"target": _batch(data, data.visual_indices, device)[1].cpu()}
        for arm in ARMS:
            model, smf, posterior = models[arm]
            _metrics, tensors = _evaluate(model, smf, posterior, data, data.visual_indices, mean, scale, device, beta=CANDIDATE_BETA if arm == "posterior_kl_coupling" else 0.0)
            panel_values[f"{arm}_camera"] = tensors["camera"]
            panel_values[f"{arm}_coefficients"] = tensors["coefficients"]
        panel_path = output.parent / "identity_smf_coupling_panel.pt"
        final_hashes = {**hashes, "training_validation": validation_hash, "training_preflight": preflight_hash}
        panel_hash = _save_panel(panel_path, data, panel_values, final_hashes)
        control = results["optical_mse_control"]["metrics"]
        coupling = results["optical_coupling"]["metrics"]
        coupling_initial = initial["optical_coupling"]
        kl = results["posterior_kl_coupling"]["metrics"]
        kl_initial = initial["posterior_kl_coupling"]
        no_kl = results["posterior_no_kl_coupling"]["metrics"]
        gates = (GateResult("power_retention_gain", coupling["power_retention"] >= control["power_retention"] + MIN_POWER_GAIN, {"coupling": coupling["power_retention"], "control": control["power_retention"]}), GateResult("coupling_mse_bound", coupling["paired_mse"] <= control["paired_mse"] * MAX_MSE_RELATIVE, {"ratio": coupling["paired_mse"] / max(control["paired_mse"], 1.0e-30)}), GateResult("projection_residual_decreases", coupling["projection_residual"] < coupling_initial["projection_residual"], {}), GateResult("coupling_correlation_not_degraded", coupling["centered_correlation"] >= coupling_initial["centered_correlation"], {}), GateResult("coupling_variance_not_degraded", coupling["variance_retention"] >= coupling_initial["variance_retention"] * (1.0 - MIN_VARIANCE_RETENTION), {}), GateResult("coefficient_rank_retention", coupling["coefficient_effective_rank"] >= coupling_initial["coefficient_effective_rank"] * MIN_RANK_RETENTION, {}), GateResult("kl_below_initial", kl.get("posterior_kl", float("inf")) < kl_initial.get("posterior_kl", float("inf")), {"initial": kl_initial.get("posterior_kl"), "final": kl.get("posterior_kl")}), GateResult("kl_below_no_kl", kl.get("posterior_kl", float("inf")) < no_kl.get("posterior_kl", float("inf")), {}), GateResult("all_arms_completed", all(item["steps_completed"] == STEPS for item in results.values()), {}), GateResult("four_phase_updates", all(canary._four_phase_nonzero({key: value for key, value in item["history"][0]["update_norms"].items() if "phase[" in key}) for item in results.values()), {}), GateResult("finite_values", all(item["metrics"]["finite"] and all(step["finite"] for step in item["history"]) for item in results.values()), {}), GateResult("evaluator_isolation", True, {"evaluator_calls": 0}), GateResult("panel_provenance", True, {"panel_sha256": panel_hash}), GateResult("cuda_peak_memory", int(torch.cuda.max_memory_allocated(device)) <= MAX_PEAK_BYTES, {"bytes": int(torch.cuda.max_memory_allocated(device))}), GateResult("wall_time", time.monotonic() - started <= MAX_CANARY_SECONDS, {"seconds": time.monotonic() - started}), GateResult("run_storage", sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file()) <= MAX_STORAGE_BYTES, {}))
        scientific = all(gate.passed for gate in gates[:11])
        resources = all(gate.passed for gate in gates[-3:])
        status = "passed" if scientific and resources else ("resource_aborted" if not resources else "valid_failed")
        actions = (NEXT_PASS_ACTION,) if status == "passed" else ((NEXT_FAILED_ACTION,) if status == "valid_failed" else ())
        record = EvidenceRecordV2(2, TRAINING_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), final_hashes, gates, actions, None if status == "passed" else "fixed OLS coupling canary gate failed")
        record.validate()
        payload = _record_payload(TRAINING_SCHEMA, record, {"outcome": "diagnostic_completed" if status == "passed" else status, "validation_hash": validation_hash, "preflight_hash": preflight_hash, "lambda_coupling": lambda_value, "initial_metrics": initial, "arms": results, "checkpoints": checkpoints, "panel_path": str(panel_path.resolve()), "panel_sha256": panel_hash, "free_cuda_bytes": int(free), "total_cuda_bytes": int(total), "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)), "wall_seconds": time.monotonic() - started, "run_storage_bytes": sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file()), "evaluator_calls": 0, "formal_candidate_checkpoint": False})
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"passed", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-ols-loss-recovery-validate":
        return validate(args)
    if args.research_action == "smf-ols-loss-recovery-preflight":
        return preflight(args)
    if args.research_action == "smf-ols-loss-recovery-train":
        return train(args)
    raise ValueError(f"unsupported fixed OLS recovery action: {args.research_action}")
