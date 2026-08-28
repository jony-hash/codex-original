"""Diagnostic SMF hybrid-VAE pilot for the H-LOMA-I0 research line.

The formal optical system is deliberately untouched.  A fixed LP01 SMF
projection supplies 100 complex coefficients; a small research-only posterior
head operates on their real and imaginary parts, samples a real diagonal
Gaussian, and injects the coefficients back into the fixed mode bank.  This
module is not a formal all-optical inference path.
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
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ...config import FormalSystemConfig, load_config
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
    tensor_sha256,
)
from ...optics.fiber_operator import FiberArrayOLS
from . import optical_manifold as base
from . import optical_manifold_smf_integration_canary as canary
from . import optical_manifold_smf_reconstruction_pilot as plan6


BRANCH = "research/v2-lightgen-optical-manifold-smf-vae-pilot-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_vae_pilot_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_vae_pilot_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_vae_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_vae_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_vae_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_smf_vae_resume.v1"
NORMALIZATION_SCHEMA = "lightgen_optical_manifold_smf_vae_posterior_normalization.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_vae_panel.v1"

SOURCE_VALIDATION_HASH = (
    "d548cf93f2cfefb2258239b066b14f574b098bfa17edc0f377f718fd2cba3ca5"
)
SOURCE_PREFLIGHT_HASH = (
    "722557fc0bfc058edafa8666772ee87bb927b0a37924b55087467303d3a56623"
)
SOURCE_TRAINING_HASH = (
    "a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e"
)
SOURCE_PANEL_HASH = "9de2fc9d9c2d43ed9752d8609afcaa5455cfb05b9c411d7d0cca9b2a8cf50fdf"
SOURCE_CHECKPOINT_HASH = (
    "949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94"
)
SOURCE_MANIFEST_HASH = (
    "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
)
SOURCE_STEP = 1000
STEPS = 100
PREFLIGHT_STEPS = 3
BATCH_SIZE = 10
PORT_COUNT = 100
REAL_LATENT_DIM = 200
MICROBATCH = 1
CHECKPOINT_INTERVAL = 50
SEED = 1234
LEARNING_RATE = 1.0e-3
INITIAL_LOGVAR = -6.0
MIN_LOGVAR = -12.0
MAX_LOGVAR = 2.0
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 1800.0
MAX_STORAGE_BYTES = 1024**3
SAFETY_MSE_RATIO = 2.0
MAX_VAE_MSE_RELATIVE = 1.10
MIN_VARIANCE_RETENTION = 0.50
NEXT_PASS_ACTION = "manual_review_fmf_m2_reconstruction_pilot_contract"
NEXT_VALID_FAILED_ACTION = "manual_review_identity_posterior_collapse_contract"
ARMS = ("smf_beta_vae", "smf_stochastic_no_kl", "smf_deterministic_mse")


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Data:
    images: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    heldout_indices: torch.Tensor
    calibration_indices: torch.Tensor
    schedule: torch.Tensor
    epsilon_schedule: torch.Tensor
    visual_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Source:
    checkpoint: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


class SharedPerPortPosterior(nn.Module):
    """Shared two-real-input posterior head with an identity-residual mean."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2, 16), nn.SiLU())
        self.mean_delta = nn.Linear(16, 2)
        self.logvar_head = nn.Linear(16, 2)
        nn.init.zeros_(self.mean_delta.weight)
        nn.init.zeros_(self.mean_delta.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, INITIAL_LOGVAR)

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized.ndim != 3 or tuple(normalized.shape[-2:]) != (PORT_COUNT, 2):
            raise ValueError("posterior input must have shape [N,100,2]")
        hidden = self.trunk(normalized)
        mean = normalized + self.mean_delta(hidden)
        logvar = self.logvar_head(hidden).clamp(MIN_LOGVAR, MAX_LOGVAR)
        return mean, logvar


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"SMF VAE pilot requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "SMF VAE pilot requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup"
        )


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "source",
        "dataset",
        "optical_contract",
        "training",
        "gates",
        "authorization",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload["schema"] != CONFIG_SCHEMA
        or payload["hypothesis"] != HYPOTHESIS
    ):
        raise ValueError("SMF VAE config is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_smf_hybrid_vae_pilot",
        "approved_contract": {
            "arms": list(ARMS),
            "steps_per_arm": STEPS,
            "logical_batch_size": BATCH_SIZE,
            "microbatch_size": MICROBATCH,
            "source_global_step": SOURCE_STEP,
            "preflight_steps": PREFLIGHT_STEPS,
            "resume_segment": "2+1",
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "posterior_real_dimension": REAL_LATENT_DIM,
            "posterior_head": "shared_per_port_linear_16_silu",
            "post_pass_next_action": NEXT_PASS_ACTION,
            "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
        },
        "bounded_diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("SMF VAE review contract does not match scope")
    return target


def _assert_contract(spec: Spec) -> None:
    p = spec.payload
    expected_source = {
        "plan6_validation_sha256": SOURCE_VALIDATION_HASH,
        "plan6_preflight_sha256": SOURCE_PREFLIGHT_HASH,
        "plan6_training_sha256": SOURCE_TRAINING_HASH,
        "plan6_panel_sha256": SOURCE_PANEL_HASH,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "mnist_manifest_sha256": SOURCE_MANIFEST_HASH,
        "source_global_step": SOURCE_STEP,
    }
    if p["source"] != {
        **{
            k: p["source"][k]
            for k in p["source"]
            if k.endswith("_validation")
            or k.endswith("_preflight")
            or k.endswith("_training")
            or k.endswith("_panel")
            or k == "source_checkpoint"
        },
        **expected_source,
    }:
        # Keep the comparison explicit without allowing an extra mutable source field.
        for key, value in expected_source.items():
            if p["source"].get(key) != value:
                raise ValueError(f"SMF VAE source binding mismatch: {key}")
    for key, value in expected_source.items():
        if p["source"].get(key) != value:
            raise ValueError(f"SMF VAE source binding mismatch: {key}")
    if p["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "training_count": 1000,
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH,
        "schedule": "class_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "calibration_and_schedule_disjoint": True,
        "prior_schedules_excluded": True,
        "heldout_excluded": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "heldout_panel_count": 1000,
        "visual_panel_count": 100,
    }:
        raise ValueError("SMF VAE dataset contract mismatch")
    optical = p["optical_contract"]
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
        "boundary": "finite_crop",
        "formal_ols": "identity",
        "research_ols": "100_port_lp01_smf",
        "posterior_domain": "100_complex_coefficients_as_200_real",
        "posterior_head": "shared_per_port_linear_16_silu",
        "posterior_status": "project_approximation",
        "injector_status": "research_hybrid_active_latent_injector",
        "front_relay": "coherent_imaging_4f",
        "back_relay": "free_space",
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    }.items():
        if optical.get(key) != value:
            raise ValueError(f"SMF VAE optical contract mismatch: {key}")
    training = p["training"]
    if (
        training.get("arms") != list(ARMS)
        or training.get("steps_per_arm") != STEPS
        or training.get("logical_batch_size", BATCH_SIZE) != BATCH_SIZE
        or training.get("microbatch_size", MICROBATCH) != MICROBATCH
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("checkpoint_interval") != CHECKPOINT_INTERVAL
        or training.get("initial_logvar") != INITIAL_LOGVAR
    ):
        raise ValueError("SMF VAE training contract mismatch")
    if (
        p["authorization"]["training_authorized"] is not False
        or p["authorization"]["formal_promotion"] is not False
    ):
        raise ValueError("SMF VAE authorization boundary is invalid")


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(path)
    if (
        formal.architecture != "conditional_identity_logit_amplitude_v2"
        or formal.system.kind != base.EXPECTED_FORMAL_KIND
        or not formal.system.is_formal
    ):
        raise ValueError("SMF VAE requires current formal Identity optics")
    for key, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, key)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1e-12, 1e-15):
                raise ValueError(f"formal geometry mismatch: {key}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch: {key}")
    return formal.system, path


def _ancestor(commit: str, current: str) -> None:
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, current],
            cwd=base._repo_root(),
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("source evidence commit is not an ancestor")


def _load_source(spec: Spec, formal_path: Path, current: str) -> Source:
    src = spec.payload["source"]
    paths = {
        "plan6_validation": (_resolve(src["plan6_validation"]), SOURCE_VALIDATION_HASH),
        "plan6_preflight": (_resolve(src["plan6_preflight"]), SOURCE_PREFLIGHT_HASH),
        "plan6_training": (_resolve(src["plan6_training"]), SOURCE_TRAINING_HASH),
        "plan6_panel": (_resolve(src["plan6_panel"]), SOURCE_PANEL_HASH),
        "source_checkpoint": (
            _resolve(src["source_checkpoint"]),
            SOURCE_CHECKPOINT_HASH,
        ),
    }
    for role, (path, digest) in paths.items():
        if file_sha256(path) != digest:
            raise ValueError(f"{role} source hash mismatch")
    records: dict[str, Mapping[str, Any]] = {}
    for role in ("plan6_validation", "plan6_preflight", "plan6_training"):
        value = json.loads(paths[role][0].read_text(encoding="utf-8"))
        record = EvidenceRecordV2.from_dict(value["evidence"])
        record.validate()
        records[role] = value
        _ancestor(record.git_commit, current)
        if (
            value.get("training_authorized") is not False
            or value.get("formal_promotion") is not False
        ):
            raise ValueError("source authorization boundary is invalid")
    if (
        EvidenceRecordV2.from_dict(records["plan6_validation"]["evidence"]).status
        != "passed"
        or EvidenceRecordV2.from_dict(records["plan6_preflight"]["evidence"]).status
        != "passed"
        or EvidenceRecordV2.from_dict(records["plan6_training"]["evidence"]).status
        != "passed"
    ):
        raise ValueError("PLAN 6 source evidence is not authoritative")
    panel = safe_torch_load(
        paths["plan6_panel"][0], expected_sha256=SOURCE_PANEL_HASH, map_location="cpu"
    )
    if (
        panel.get("schema") != plan6.PANEL_SCHEMA
        or panel.get("indices_are_global_mnist") is not True
    ):
        raise ValueError("PLAN 6 source panel is invalid")
    checkpoint_payload = safe_torch_load(
        paths["source_checkpoint"][0],
        expected_sha256=SOURCE_CHECKPOINT_HASH,
        map_location="cpu",
    )
    if (
        checkpoint_payload.get("schema") != plan6.RESUME_SCHEMA
        or checkpoint_payload.get("step") != SOURCE_STEP
        or not isinstance(checkpoint_payload.get("model_state"), Mapping)
        or not isinstance(checkpoint_payload.get("rng_state"), Mapping)
    ):
        raise ValueError("PLAN 6 source checkpoint is invalid")
    hashes = {
        "plan6_validation": SOURCE_VALIDATION_HASH,
        "plan6_preflight": SOURCE_PREFLIGHT_HASH,
        "plan6_training": SOURCE_TRAINING_HASH,
        "plan6_panel": SOURCE_PANEL_HASH,
        "source_checkpoint": SOURCE_CHECKPOINT_HASH,
        "mnist_manifest": SOURCE_MANIFEST_HASH,
        "formal_config": file_sha256(formal_path),
    }
    return Source(checkpoint_payload, hashes, formal_path)


def _load_data(root: str | Path) -> Data:
    source_data = plan6._load_data(root)
    images, labels = source_data.images, source_data.labels
    excluded = torch.cat((source_data.schedule.flatten(), source_data.heldout_indices))
    calibration: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    train_labels = labels.index_select(0, source_data.train_indices)
    for label in range(10):
        candidates = source_data.train_indices[train_labels == label]
        candidates = candidates[~torch.isin(candidates, excluded)].sort().values
        if candidates.numel() < 200:
            raise ValueError(
                "insufficient fresh MNIST samples for posterior calibration"
            )
        generator = torch.Generator(device="cpu").manual_seed(SEED + label)
        shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
        calibration.append(shuffled[:100])
        selected.append(shuffled[100:200])
    calibration_indices = torch.stack(calibration, dim=1)
    schedule = torch.stack(selected, dim=1)
    if (
        torch.unique(calibration_indices).numel() != calibration_indices.numel()
        or torch.unique(schedule).numel() != schedule.numel()
        or torch.isin(calibration_indices.flatten(), schedule.flatten()).any()
        or torch.isin(schedule.flatten(), excluded).any()
    ):
        raise ValueError("SMF VAE calibration or schedule is not disjoint")
    epsilon = torch.Generator(device="cpu").manual_seed(SEED)
    epsilon_schedule = torch.randn(
        (STEPS, BATCH_SIZE, PORT_COUNT, 2), generator=epsilon, dtype=torch.float32
    )
    if not torch.equal(
        labels.index_select(0, schedule.flatten()).reshape(STEPS, BATCH_SIZE)[0],
        torch.arange(10),
    ):
        raise ValueError("SMF VAE schedule is not ordered one-per-label")
    heldout_labels = labels.index_select(0, source_data.heldout_indices)
    heldout = torch.cat(
        [source_data.heldout_indices[heldout_labels == label] for label in range(10)]
    )
    visual = torch.cat(
        [heldout[labels.index_select(0, heldout) == label][:10] for label in range(10)]
    )
    manifest = {
        "schema": NORMALIZATION_SCHEMA.replace(
            "posterior_normalization", "mnist_manifest"
        ),
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "calibration_indices_sha256": tensor_sha256(calibration_indices),
        "calibration_count": int(calibration_indices.numel()),
        "schedule_sha256": tensor_sha256(schedule),
        "schedule_shape": list(schedule.shape),
        "schedule_count": int(schedule.numel()),
        "epsilon_schedule_sha256": tensor_sha256(epsilon_schedule),
        "heldout_indices_sha256": tensor_sha256(heldout),
        "visual_indices_sha256": tensor_sha256(visual),
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "schedule_balance_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return Data(
        images,
        labels,
        source_data.train_indices,
        heldout,
        calibration_indices,
        schedule,
        epsilon_schedule,
        visual,
        manifest,
    )


def _context(spec: Spec, formal: FormalSystemConfig, data: Data) -> EvidenceContextV2:
    return EvidenceContextV2(
        canonical_sha256(formal.model_dump(mode="json")),
        spec.config_hash,
        data.manifest["manifest_hash"],
        canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        1.0,
    )


def _record_payload(
    schema: str,
    record: EvidenceRecordV2,
    contract: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        value.update(extra)
    return value


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite SMF VAE artifact: {target}")
    atomic_write_json(target, payload)


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


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
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


class _CpuAdam:
    def __init__(
        self, model: nn.Module, posterior: SharedPerPortPosterior | None
    ) -> None:
        params = [
            (name, parameter)
            for name, parameter in list(model.named_parameters())
            + (list(posterior.named_parameters()) if posterior is not None else [])
            if parameter.requires_grad
        ]
        if not params:
            raise ValueError("SMF VAE optimizer has no parameters")
        self.names = tuple(name for name, _ in params)
        self.gpu = tuple(parameter for _, parameter in params)
        self.cpu = tuple(
            nn.Parameter(parameter.detach().cpu().clone()) for parameter in self.gpu
        )
        self.optimizer = torch.optim.Adam(self.cpu, lr=LEARNING_RATE)

    def step(self) -> None:
        for gpu, cpu in zip(self.gpu, self.cpu, strict=True):
            cpu.data.copy_(gpu.detach().cpu())
            cpu.grad = None if gpu.grad is None else gpu.grad.detach().cpu().clone()
        self.optimizer.step()
        for gpu, cpu in zip(self.gpu, self.cpu, strict=True):
            gpu.data.copy_(cpu.data)

    def state_dict(self) -> dict[str, Any]:
        return {"names": self.names, "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if tuple(value.get("names", ())) != self.names:
            raise ValueError("SMF VAE optimizer parameter names mismatch")
        self.optimizer.load_state_dict(value["optimizer"])


def _set_requires(
    model: base.IdentityOpticalManifoldAutoencoder,
    posterior: SharedPerPortPosterior | None,
) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in base.EXPECTED_TRAINABLE)
    if posterior is not None:
        for parameter in posterior.parameters():
            parameter.requires_grad_(True)


def _make_runtime(
    formal: FormalSystemConfig, source: Source, device: torch.device, arm: str
) -> tuple[
    base.IdentityOpticalManifoldAutoencoder,
    FiberArrayOLS,
    SharedPerPortPosterior | None,
    _CpuAdam,
]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(source.checkpoint["model_state"], strict=True)
    posterior = (
        None if arm == "smf_deterministic_mse" else SharedPerPortPosterior().to(device)
    )
    _set_requires(model, posterior)
    operator = canary._smf_operator(model.system, device)
    return model, operator, posterior, _CpuAdam(model, posterior)


def _encoded_coefficients(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    amplitude: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    system = model.system
    slm = system.slm(system.command(amplitude))
    front = system.front(slm)
    encoded, _, _ = system.encoder(front, "encoder")
    values = encoded.tensor[:, 0]
    modes = operator.input_modes.to(values)
    area = system.encoder_plane.pixel_pitch_m[0] * system.encoder_plane.pixel_pitch_m[1]
    return torch.einsum("mhw,bhw->bm", modes.conj(), values) * area, system


def _decode_coefficients(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    coefficients: torch.Tensor,
    system: Any,
) -> torch.Tensor:
    values = torch.einsum(
        "bm,mhw->bhw", coefficients, operator.output_modes.to(coefficients)
    ).unsqueeze(1)
    field = ComplexField(
        values,
        system.encoder_plane.with_plane("ols_output"),
        OperationProvenance(
            "research_smf_latent_injection",
            system.encoder_plane.plane_id,
            0.0,
            {"status": "project_approximation"},
        ),
    )
    decoded, _, _ = system.decoder(field, "decoder")
    chip = ComplexField(decoded.tensor, system.chip_plane, decoded.provenance)
    camera_field = system.back(chip)
    return camera_field.tensor.abs().square()


def _normalize(
    coefficients: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        (
            (coefficients.real - mean[..., 0]) / scale[..., 0],
            (coefficients.imag - mean[..., 1]) / scale[..., 1],
        ),
        dim=-1,
    )


def _denormalize(
    values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return torch.complex(
        values[..., 0] * scale[..., 0] + mean[..., 0],
        values[..., 1] * scale[..., 1] + mean[..., 1],
    )


def posterior_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mean.shape != logvar.shape or mean.shape[-2:] != (PORT_COUNT, 2):
        raise ValueError("posterior mean/logvar shape mismatch")
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()


def _calibrate_beta(
    reconstruction_encoder_norm: float, kl_encoder_norm: float
) -> float:
    """Register the single preflight beta without searching or tuning it."""
    if (
        not torch.isfinite(torch.tensor(reconstruction_encoder_norm))
        or not torch.isfinite(torch.tensor(kl_encoder_norm))
        or reconstruction_encoder_norm <= 0.0
        or kl_encoder_norm <= 0.0
    ):
        raise ValueError("beta calibration requires finite, non-zero gradient norms")
    return 0.1 * reconstruction_encoder_norm / kl_encoder_norm


def _forward_latent(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    posterior: SharedPerPortPosterior,
    amplitude: torch.Tensor,
    epsilon: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    sample: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients, system = _encoded_coefficients(model, operator, amplitude)
    normalized = _normalize(coefficients, mean, scale)
    posterior_mean, logvar = posterior(normalized)
    latent = (
        posterior_mean + logvar.mul(0.5).exp() * epsilon if sample else posterior_mean
    )
    raw = _decode_coefficients(
        model, operator, _denormalize(latent, mean, scale), system
    )
    return raw, posterior_mean, logvar, coefficients


def _forward_direct(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    return canary._smf_forward(model, operator, amplitude)[0]


def _clear(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


def _trainable_items(
    model: nn.Module, posterior: SharedPerPortPosterior | None
) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, value)
        for name, value in list(model.named_parameters())
        + (
            [(f"posterior.{n}", v) for n, v in posterior.named_parameters()]
            if posterior is not None
            else []
        )
        if value.requires_grad
    ]


def _objective_gradient_report(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    posterior: SharedPerPortPosterior,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    epsilon: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    objective: str,
) -> dict[str, Any]:
    """Measure one objective's gradients without changing model or optimizer state."""
    if objective not in {"reconstruction", "kl"}:
        raise ValueError("unsupported SMF VAE gradient objective")
    all_items = _trainable_items(model, posterior)
    for _, value in all_items:
        value.grad = None
    values: list[float] = []
    for index in range(BATCH_SIZE):
        amp = amplitudes[index : index + 1]
        target = targets[index : index + 1]
        eps = epsilon[index : index + 1]
        raw, post_mean, logvar, _ = checkpoint(
            lambda value, noise: _forward_latent(
                model, operator, posterior, value, noise, mean, scale, sample=True
            ),
            amp,
            eps,
            use_reentrant=False,
        )
        value = (
            F.mse_loss(raw, target)
            if objective == "reconstruction"
            else posterior_kl(post_mean, logvar)
        )
        (value / BATCH_SIZE).backward()
        values.append(float(value.detach().cpu()))
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter, device="cpu")
        for name, parameter in all_items
    }
    norms = {name: float(value.norm()) for name, value in gradients.items()}
    _clear(model)
    _clear(posterior)
    finite = all(
        torch.isfinite(torch.tensor(value)) for value in (*values, *norms.values())
    )
    return {
        "objective": objective,
        "value": float(sum(values) / len(values)),
        "gradient_norms": norms,
        "finite": finite,
    }


def _step(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    posterior: SharedPerPortPosterior | None,
    optimizer: _CpuAdam,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    epsilon: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    arm: str,
    beta: float,
) -> dict[str, Any]:
    all_items = _trainable_items(model, posterior)
    before = {name: value.detach().clone() for name, value in all_items}
    for _, value in all_items:
        value.grad = None
    mse_values: list[float] = []
    kl_values: list[float] = []
    cameras: list[torch.Tensor] = []
    for index in range(BATCH_SIZE):
        amp = amplitudes[index : index + 1]
        target = targets[index : index + 1]
        eps = epsilon[index : index + 1]
        if posterior is None:
            raw = checkpoint(
                lambda value: _forward_direct(model, operator, value),
                amp,
                use_reentrant=False,
            )
            kl = torch.zeros((), device=amp.device)
        else:
            raw, post_mean, logvar, _ = checkpoint(
                lambda value, noise: _forward_latent(
                    model, operator, posterior, value, noise, mean, scale, sample=True
                ),
                amp,
                eps,
                use_reentrant=False,
            )
            kl = posterior_kl(post_mean, logvar)
        mse = F.mse_loss(raw, target)
        loss = mse + beta * kl
        (loss / BATCH_SIZE).backward()
        mse_values.append(float(mse.detach().cpu()))
        kl_values.append(float(kl.detach().cpu()))
        cameras.append(raw.detach())
    gradients = {
        name: value.grad.detach().cpu().clone()
        if value.grad is not None
        else torch.zeros_like(value, device="cpu")
        for name, value in all_items
    }
    optimizer.step()
    updates = {
        name: float((value.detach() - before[name]).abs().sum().cpu())
        for name, value in all_items
    }
    norms = {name: float(value.norm().cpu()) for name, value in gradients.items()}
    _clear(model)
    if posterior is not None:
        _clear(posterior)
    camera = torch.cat(cameras)
    finite = bool(
        torch.isfinite(camera).all()
        and all(
            torch.isfinite(torch.tensor(value))
            for value in (*mse_values, *kl_values, *norms.values(), *updates.values())
        )
    )
    return {
        "arm": arm,
        "loss": float(
            sum(m + beta * k for m, k in zip(mse_values, kl_values, strict=True))
            / BATCH_SIZE
        ),
        "mse": float(sum(mse_values) / BATCH_SIZE),
        "kl": float(sum(kl_values) / BATCH_SIZE),
        "gradient_norms": norms,
        "update_norms": updates,
        "camera_finite": bool(torch.isfinite(camera).all()),
        "camera_min": float(camera.min().cpu()),
        "camera_max": float(camera.max().cpu()),
        "activation_checkpoint": True,
        "finite": finite,
    }


def _fit_normalization(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    data: Data,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in data.calibration_indices:
            amp, _target = canary._batch(
                canary.CanaryData(
                    data.images,
                    data.labels,
                    data.train_indices,
                    data.heldout_indices,
                    data.schedule,
                    data.visual_indices,
                    data.manifest,
                ),
                indices,
                device,
            )
            coefficients, _ = _encoded_coefficients(model, operator, amp)
            values.append(
                torch.stack((coefficients.real, coefficients.imag), dim=-1).cpu()
            )
    flat = torch.cat(values).to(torch.float64)
    mean = flat.mean(dim=0).to(torch.float32)
    scale = flat.std(dim=0, unbiased=False).clamp_min(1.0e-6).to(torch.float32)
    return mean, scale


def _batch(
    data: Data, indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude.to(device), rendered.target.to(device)


def _evaluate(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    posterior: SharedPerPortPosterior | None,
    data: Data,
    indices: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    *,
    sample: bool = False,
    epsilon: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    if posterior is not None:
        posterior.eval()
    cameras: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            amplitudes, target = _batch(data, batch_indices, device)
            if posterior is None:
                raw = _forward_direct(model, operator, amplitudes)
                coefficient, _ = _encoded_coefficients(model, operator, amplitudes)
            else:
                noise = (
                    torch.zeros((amplitudes.shape[0], PORT_COUNT, 2), device=device)
                    if epsilon is None
                    else epsilon[start : start + amplitudes.shape[0]].to(device)
                )
                raw, post_mean, logvar, coefficient = _forward_latent(
                    model,
                    operator,
                    posterior,
                    amplitudes,
                    noise,
                    mean,
                    scale,
                    sample=sample,
                )
                means.append(post_mean.cpu())
                logvars.append(logvar.cpu())
            cameras.append(raw.cpu())
            targets.append(target.cpu())
            features.append(coefficient.cpu())
    camera = torch.cat(cameras)
    target = torch.cat(targets)
    centered_camera = camera.flatten(1) - camera.flatten(1).mean(1, keepdim=True)
    centered_target = target.flatten(1) - target.flatten(1).mean(1, keepdim=True)
    corr = (centered_camera * centered_target).sum(1) / (
        centered_camera.norm(dim=1) * centered_target.norm(dim=1)
    ).clamp_min(1.0e-30)
    result = {
        "paired_mse": float(F.mse_loss(camera, target)),
        "cyclic_mse": float(F.mse_loss(camera, target.roll(1, 0))),
        "centered_correlation": float(corr.mean()),
        "variance_retention": float(
            centered_camera.var(0, unbiased=False).mean()
            / centered_target.var(0, unbiased=False).mean().clamp_min(1.0e-30)
        ),
        "finite": bool(torch.isfinite(camera).all() and torch.isfinite(target).all()),
        "evaluation_count": int(indices.numel()),
        "feature_shape": list(torch.cat(features).shape),
        "sampled": sample,
    }
    if means:
        posterior_mean = torch.cat(means)
        posterior_logvar = torch.cat(logvars)
        result.update(
            {
                "posterior_kl": float(posterior_kl(posterior_mean, posterior_logvar)),
                "posterior_mean_norm": float(posterior_mean.norm(dim=-1).mean()),
                "posterior_variance_mean": float(posterior_logvar.exp().mean()),
                "posterior_mean_variance": float(
                    posterior_mean.flatten(1).var(0, unbiased=False).mean()
                ),
            }
        )
    return result, {"camera": camera, "target": target, "features": torch.cat(features)}


def _normalization_payload(
    mean: torch.Tensor,
    scale: torch.Tensor,
    spec: Spec,
    data: Data,
    source: Source,
    validation_hash: str,
) -> dict[str, Any]:
    return {
        "schema": NORMALIZATION_SCHEMA,
        "mean": mean.cpu(),
        "scale": scale.cpu(),
        "shape": [PORT_COUNT, 2],
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "source_hashes": dict(source.hashes),
        "validation_sha256": validation_hash,
        "training_authorized": False,
        "formal_promotion": False,
    }


def _resume_payload(
    model: nn.Module,
    posterior: SharedPerPortPosterior | None,
    optimizer: _CpuAdam,
    arm: str,
    step: int,
    history: list[dict[str, Any]],
    data: Data,
    spec: Spec,
    hashes: Mapping[str, str],
    beta: float,
    completed_arms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "arm": arm,
        "step": step,
        "model_state": _model_state(model),
        "posterior_state": None if posterior is None else _model_state(posterior),
        "optimizer_state": _tree_clone(optimizer.state_dict()),
        "rng_state": _tree_clone(base._rng_state()),
        "history": list(history),
        "schedule_cursor": {"next_step": step, "batch_size": BATCH_SIZE},
        "epsilon_cursor": step,
        "beta": beta,
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": data.manifest["schedule_sha256"],
        "evidence_hashes": dict(hashes),
        "completed_arms": dict(completed_arms or {}),
        "formal_candidate_checkpoint": False,
        "training_authorized": False,
        "formal_promotion": False,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite SMF VAE checkpoint: {path}")
    atomic_torch_save(path, payload)


def _validation_record(
    state: Mapping[str, Any],
    spec: Spec,
    formal: FormalSystemConfig,
    data: Data,
    source: Source,
    review_path: Path,
) -> EvidenceRecordV2:
    hashes = {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
    }
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_branch", {"branch": BRANCH}),
            ("source_step", {"step": SOURCE_STEP}),
            ("plan6_evidence_complete", {"training": "passed"}),
            ("mnist_disjoint_schedule", {"calibration": 1000, "training": 1000}),
            ("formal_geometry", {"grid": 2048, "identity_formal_ols": True}),
            ("smf_posterior_contract", {"ports": 100, "real_dimension": 200}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            (
                "research_boundary",
                {"training_authorized": False, "formal_promotion": False},
            ),
        )
    )
    record = EvidenceRecordV2(
        2,
        VALIDATION_SCHEMA,
        "diagnostic",
        "passed",
        str(state["commit"]),
        None,
        spec.config_hash,
        _context(spec, formal, data),
        hashes,
        gates,
        ("run_smf_vae_preflight",),
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    if tuple(data.images.shape) != (60000, 1, 28, 28) or tuple(data.labels.shape) != (
        60000,
    ):
        raise ValueError("SMF VAE requires complete MNIST tensors")
    record = _validation_record(state, spec, formal, data, source, review_path)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "formal_config": str(formal_path.resolve()),
            "source_hashes": source.hashes,
            "dataset_manifest": data.manifest,
            "optimizer_created": False,
            "cuda_context_created": False,
        },
    )
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path,
    spec: Spec,
    formal: FormalSystemConfig,
    data: Data,
    state: Mapping[str, Any],
    source: Source,
    review_path: Path,
) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected = {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
    }
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or record.status != "passed"
        or record.evidence_kind != VALIDATION_SCHEMA
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.context != _context(spec, formal, data)
        or record.input_hashes != expected
        or record.permitted_next_actions != ("run_smf_vae_preflight",)
    ):
        raise ValueError("SMF VAE validation is stale or mismatched")
    return payload, digest


def _load_preflight(
    path: str | Path,
    spec: Spec,
    formal: FormalSystemConfig,
    data: Data,
    state: Mapping[str, Any],
    validation_hash: str,
    source: Source,
    review_path: Path,
) -> tuple[dict[str, Any], str, dict[str, str], torch.Tensor, torch.Tensor]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    normalization_path = _resolve(payload["contract"]["normalization_path"])
    normalization_hash = file_sha256(normalization_path)
    expected = {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
        "training_validation": validation_hash,
        "posterior_normalization": normalization_hash,
    }
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.input_hashes != expected
    ):
        raise ValueError("SMF VAE preflight is stale or mismatched")
    normalization = safe_torch_load(normalization_path, map_location="cpu")
    if (
        normalization.get("schema") != NORMALIZATION_SCHEMA
        or normalization.get("config_hash") != spec.config_hash
        or normalization.get("dataset_manifest_hash") != data.manifest["manifest_hash"]
        or normalization.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_HASH
        or normalization.get("source_hashes") != source.hashes
        or normalization.get("validation_sha256") != validation_hash
        or payload["contract"].get("normalization_sha256") != normalization_hash
        or tuple(normalization.get("shape", ())) != (PORT_COUNT, 2)
        or not torch.isfinite(normalization["mean"]).all()
        or not torch.isfinite(normalization["scale"]).all()
    ):
        raise ValueError("SMF VAE normalization artifact is stale")
    hashes = {
        **expected,
        "training_preflight": digest,
    }
    return payload, digest, hashes, normalization["mean"], normalization["scale"]


def _run_exact(
    model: base.IdentityOpticalManifoldAutoencoder,
    operator: FiberArrayOLS,
    posterior: SharedPerPortPosterior | None,
    optimizer: _CpuAdam,
    data: Data,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    arm: str,
    beta: float,
    steps: int,
    *,
    start_step: int = 0,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    if start_step < 0 or start_step + steps > data.schedule.shape[0]:
        raise ValueError("SMF VAE exact segment is outside the schedule")
    for step in range(start_step, start_step + steps):
        amp, target = _batch(data, data.schedule[step], device)
        report = _step(
            model,
            operator,
            posterior,
            optimizer,
            amp,
            target,
            data.epsilon_schedule[step].to(device),
            mean.to(device),
            scale.to(device),
            arm,
            beta,
        )
        report["step"] = step + 1
        history.append(report)
    return history


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("SMF VAE preflight requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    validation_payload, validation_hash = _load_validation(
        args.validation_evidence, spec, formal, data, state, source, review_path
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SMF VAE preflight: {output}")
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
        norm_model, norm_operator, _norm_post, _norm_opt = _make_runtime(
            formal, source, device, "smf_deterministic_mse"
        )
        mean, scale = _fit_normalization(norm_model, norm_operator, data, device)
        norm_path = output.parent / "posterior_normalization.pt"
        atomic_torch_save(
            norm_path,
            _normalization_payload(mean, scale, spec, data, source, validation_hash),
        )
        beta_model, beta_operator, beta_post, beta_opt = _make_runtime(
            formal, source, device, "smf_beta_vae"
        )
        calibration_amplitudes, calibration_targets = _batch(
            data, data.schedule[0], device
        )
        calibration_epsilon = data.epsilon_schedule[0].to(device)
        reconstruction_calibration = _objective_gradient_report(
            beta_model,
            beta_operator,
            beta_post,
            calibration_amplitudes,
            calibration_targets,
            calibration_epsilon,
            mean.to(device),
            scale.to(device),
            "reconstruction",
        )
        kl_calibration = _objective_gradient_report(
            beta_model,
            beta_operator,
            beta_post,
            calibration_amplitudes,
            calibration_targets,
            calibration_epsilon,
            mean.to(device),
            scale.to(device),
            "kl",
        )
        reconstruction_encoder_norm = reconstruction_calibration["gradient_norms"][
            "system.encoder.phase"
        ]
        kl_encoder_norm = kl_calibration["gradient_norms"]["system.encoder.phase"]
        beta = _calibrate_beta(reconstruction_encoder_norm, kl_encoder_norm)
        beta_history = _run_exact(
            beta_model,
            beta_operator,
            beta_post,
            beta_opt,
            data,
            mean,
            scale,
            device,
            "smf_beta_vae",
            beta,
            PREFLIGHT_STEPS,
        )
        uninterrupted_rng = _tree_clone(base._rng_state())
        control_model, control_operator, control_post, control_opt = _make_runtime(
            formal, source, device, "smf_stochastic_no_kl"
        )
        det_model, det_operator, det_post, det_opt = _make_runtime(
            formal, source, device, "smf_deterministic_mse"
        )
        control_history = _run_exact(
            control_model,
            control_operator,
            control_post,
            control_opt,
            data,
            mean,
            scale,
            device,
            "smf_stochastic_no_kl",
            0.0,
            PREFLIGHT_STEPS,
        )
        det_history = _run_exact(
            det_model,
            det_operator,
            det_post,
            det_opt,
            data,
            mean,
            scale,
            device,
            "smf_deterministic_mse",
            0.0,
            PREFLIGHT_STEPS,
        )
        interrupted_model, interrupted_operator, interrupted_post, interrupted_opt = (
            _make_runtime(formal, source, device, "smf_beta_vae")
        )
        interrupted_history = _run_exact(
            interrupted_model,
            interrupted_operator,
            interrupted_post,
            interrupted_opt,
            data,
            mean,
            scale,
            device,
            "smf_beta_vae",
            beta,
            2,
        )
        resume_state = _resume_payload(
            interrupted_model,
            interrupted_post,
            interrupted_opt,
            "smf_beta_vae",
            2,
            interrupted_history,
            data,
            spec,
            {
                **source.hashes,
                "training_validation": validation_hash,
                "posterior_normalization": file_sha256(norm_path),
            },
            beta,
        )
        resume_path = output.parent / "preflight_smf_beta_vae_step_0002.pt"
        _save_checkpoint(resume_path, resume_state)
        resumed_model, resumed_operator, resumed_post, resumed_opt = _make_runtime(
            formal, source, device, "smf_beta_vae"
        )
        resumed_model.load_state_dict(resume_state["model_state"], strict=True)
        resumed_post.load_state_dict(resume_state["posterior_state"], strict=True)
        resumed_opt.load_state_dict(resume_state["optimizer_state"])
        base._restore_rng(resume_state["rng_state"])
        resumed = _run_exact(
            resumed_model,
            resumed_operator,
            resumed_post,
            resumed_opt,
            data,
            mean,
            scale,
            device,
            "smf_beta_vae",
            beta,
            1,
            start_step=2,
        )
        resumed_rng = _tree_clone(base._rng_state())
        resume_checks = {
            "model_state": _tree_equal(
                _model_state(beta_model), _model_state(resumed_model)
            ),
            "optimizer_state": _tree_equal(
                beta_opt.state_dict(), resumed_opt.state_dict()
            ),
            "step_report": _tree_equal(beta_history[2], resumed[0]),
            "rng_state": _tree_equal(uninterrupted_rng, resumed_rng),
            "persisted_history": resume_state["history"] == interrupted_history,
            "schedule_cursor": resume_state["schedule_cursor"]
            == {"next_step": 2, "batch_size": BATCH_SIZE},
        }
        exact = all(resume_checks.values())
        finite = all(
            item["finite"]
            for item in beta_history + control_history + det_history + resumed
        )
        phase_updates = all(
            item["update_norms"].get(name, 0.0) > 0.0
            for item in beta_history + control_history + det_history
            for name in base.EXPECTED_TRAINABLE
        )
        posterior_updates = all(
            any(
                value > 0.0
                for name, value in item["update_norms"].items()
                if name.startswith("posterior.")
            )
            for item in beta_history + control_history
        )
        elapsed = time.monotonic() - started
        gates = (
            GateResult("finite_values", finite, {}),
            GateResult("four_phase_updates", phase_updates, {}),
            GateResult("posterior_updates", posterior_updates, {}),
            GateResult("persisted_resume_exact", exact, resume_checks),
            GateResult(
                "fixed_epsilon_and_schedule",
                True,
                {"epsilon_seed": SEED, "schedule_cursor": 3},
            ),
            GateResult(
                "beta_calibration",
                reconstruction_calibration["finite"]
                and kl_calibration["finite"]
                and beta > 0.0,
                {
                    "beta": beta,
                    "reconstruction_encoder_norm": reconstruction_encoder_norm,
                    "kl_encoder_norm": kl_encoder_norm,
                    "formula": "0.1*g_reconstruction_encoder/g_kl_encoder",
                },
            ),
            GateResult(
                "normalization_provenance", True, {"path": str(norm_path.resolve())}
            ),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult(
                "wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}
            ),
        )
        status = "passed" if all(gate.passed for gate in gates) else "invalid_harness"
        actions = ("run_smf_vae_pilot",) if status == "passed" else ()
        hashes = {
            **source.hashes,
            "research_config": file_sha256(spec.path),
            "review_contract": file_sha256(review_path),
            "dataset_manifest": data.manifest["manifest_hash"],
            "training_validation": validation_hash,
            "posterior_normalization": file_sha256(norm_path),
        }
        record = EvidenceRecordV2(
            2,
            PREFLIGHT_SCHEMA,
            "diagnostic",
            status,
            str(state["commit"]),
            None,
            spec.config_hash,
            _context(spec, formal, data),
            hashes,
            gates,
            actions,
            None if status == "passed" else "SMF VAE preflight gate failed",
        )
        record.validate()
        payload = _record_payload(
            PREFLIGHT_SCHEMA,
            record,
            {
                "normalization_path": str(norm_path.resolve()),
                "normalization_sha256": file_sha256(norm_path),
                "beta": beta,
                "beta_calibration": {
                    "reconstruction": reconstruction_calibration,
                    "kl": kl_calibration,
                    "formula": "0.1*g_reconstruction_encoder/g_kl_encoder",
                },
                "training_hashes": hashes,
                "arms": {
                    "smf_beta_vae": beta_history,
                    "smf_stochastic_no_kl": control_history,
                    "smf_deterministic_mse": det_history,
                },
                "resume_artifact": str(resume_path.resolve()),
                "elapsed_seconds": elapsed,
                "free_cuda_bytes": int(free),
                "total_cuda_bytes": int(total),
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
                "training_authorized": False,
                "formal_promotion": False,
            },
        )
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status == "passed" else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def _load_resume(
    path: Path,
    model: nn.Module,
    posterior: SharedPerPortPosterior | None,
    optimizer: _CpuAdam,
    data: Data,
    spec: Spec,
    hashes: Mapping[str, str],
    arm: str,
    output_dir: Path,
) -> dict[str, Any]:
    if path.parent.resolve() != output_dir.resolve():
        raise ValueError("SMF VAE resume belongs to another attempt")
    payload = safe_torch_load(path, map_location="cpu")
    if (
        payload.get("schema") != RESUME_SCHEMA
        or payload.get("arm") != arm
        or not 0 < payload.get("step", 0) < STEPS
        or payload.get("config_hash") != spec.config_hash
        or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]
        or payload.get("schedule_sha256") != data.manifest["schedule_sha256"]
        or payload.get("evidence_hashes") != dict(hashes)
    ):
        raise ValueError("SMF VAE resume lineage mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    if posterior is not None:
        posterior.load_state_dict(payload["posterior_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return payload


def _save_panel(
    path: Path,
    data: Data,
    results: Mapping[str, Mapping[str, torch.Tensor]],
    source_hashes: Mapping[str, str],
) -> str:
    tensors = {
        "indices": data.visual_indices.cpu(),
        "input_28": data.images.index_select(0, data.visual_indices).cpu(),
    }
    for key, value in results.items():
        tensors[key] = value[:100].contiguous().clone().cpu()
    payload = {
        "schema": PANEL_SCHEMA,
        "source_hashes": dict(source_hashes),
        "indices_are_global_mnist": True,
        "tensors": tensors,
        "posterior_sample_is_diagnostic": True,
    }
    atomic_torch_save(path, payload)
    return file_sha256(path)


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("SMF VAE training requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    validation_payload, validation_hash = _load_validation(
        args.validation_evidence, spec, formal, data, state, source, review_path
    )
    preflight_payload, preflight_hash, hashes, mean, scale = _load_preflight(
        args.preflight_evidence,
        spec,
        formal,
        data,
        state,
        validation_hash,
        source,
        review_path,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite SMF VAE training evidence: {output}"
        )
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
        beta = float(preflight_payload["contract"]["beta"])
        initial: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            model, operator, posterior, optimizer = _make_runtime(
                formal, source, device, arm
            )
            initial[arm], _ = _evaluate(
                model,
                operator,
                posterior,
                data,
                data.heldout_indices,
                mean.to(device),
                scale.to(device),
                device,
                sample=posterior is not None,
                epsilon=torch.zeros((data.heldout_indices.numel(), PORT_COUNT, 2)),
            )
        results: dict[str, Any] = {}
        models: dict[str, tuple[Any, Any, Any]] = {}
        checkpoints: dict[str, Any] = {}
        resume_arm: str | None = None
        resume_payload: Mapping[str, Any] | None = None
        if args.resume_checkpoint:
            resume_path = _resolve(args.resume_checkpoint)
            resume_payload = safe_torch_load(resume_path, map_location="cpu")
            resume_arm = str(resume_payload.get("arm"))
            if resume_arm not in ARMS:
                raise ValueError("invalid SMF VAE resume arm")
        ordered = list(ARMS)
        if resume_arm is not None:
            ordered = [resume_arm] + [arm for arm in ARMS if arm != resume_arm]
        for arm in ordered:
            model, operator, posterior, optimizer = _make_runtime(
                formal, source, device, arm
            )
            start = 0
            history: list[dict[str, Any]] = []
            if resume_arm == arm and resume_payload is not None:
                resume = _load_resume(
                    _resolve(args.resume_checkpoint),
                    model,
                    posterior,
                    optimizer,
                    data,
                    spec,
                    hashes,
                    arm,
                    output.parent,
                )
                start = int(resume["step"])
                history = list(resume["history"])
            beta_arm = beta if arm == "smf_beta_vae" else 0.0
            for step in range(start, STEPS):
                amplitudes, targets = _batch(data, data.schedule[step], device)
                report = _step(
                    model,
                    operator,
                    posterior,
                    optimizer,
                    amplitudes,
                    targets,
                    data.epsilon_schedule[step].to(device),
                    mean.to(device),
                    scale.to(device),
                    arm,
                    beta_arm,
                )
                report["step"] = step + 1
                history.append(report)
                if (
                    not report["finite"]
                    or report["loss"] > 2.0 * initial[arm]["paired_mse"]
                ):
                    raise RuntimeError(
                        f"SMF VAE safety gate failed for {arm} at step {step + 1}"
                    )
                if (step + 1) % CHECKPOINT_INTERVAL == 0:
                    checkpoint_path = output.parent / f"{arm}_step_{step + 1:04d}.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        _resume_payload(
                            model,
                            posterior,
                            optimizer,
                            arm,
                            step + 1,
                            history,
                            data,
                            spec,
                            hashes,
                            beta_arm,
                        ),
                    )
                    checkpoints[arm] = {
                        "path": str(checkpoint_path.resolve()),
                        "sha256": file_sha256(checkpoint_path),
                    }
            metrics, panel = _evaluate(
                model,
                operator,
                posterior,
                data,
                data.heldout_indices,
                mean.to(device),
                scale.to(device),
                device,
                sample=posterior is not None,
                epsilon=torch.zeros((data.heldout_indices.numel(), PORT_COUNT, 2)),
            )
            results[arm] = {
                "history": history,
                "metrics": metrics,
                "panel": panel,
                "steps_completed": len(history),
            }
            models[arm] = (model, operator, posterior)
        deterministic_mse = results["smf_deterministic_mse"]["metrics"]["paired_mse"]
        vae_mse = results["smf_beta_vae"]["metrics"]["paired_mse"]
        vae_metrics = results["smf_beta_vae"]["metrics"]
        no_kl_metrics = results["smf_stochastic_no_kl"]["metrics"]
        panel_results = {
            "target": _evaluate(
                models["smf_deterministic_mse"][0],
                models["smf_deterministic_mse"][1],
                None,
                data,
                data.heldout_indices[:100],
                mean.to(device),
                scale.to(device),
                device,
            )[1]["target"],
            "smf_deterministic_mse": results["smf_deterministic_mse"]["panel"][
                "camera"
            ],
            "smf_stochastic_no_kl": results["smf_stochastic_no_kl"]["panel"]["camera"],
            "smf_beta_vae": results["smf_beta_vae"]["panel"]["camera"],
        }
        panel_path = output.parent / "smf_vae_panel.pt"
        panel_hash = _save_panel(panel_path, data, panel_results, hashes)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(
            item.stat().st_size for item in output.parent.rglob("*") if item.is_file()
        )
        gates = (
            GateResult(
                "finite_values",
                all(
                    item["metrics"]["finite"]
                    and all(step["finite"] for step in item["history"])
                    for item in results.values()
                ),
                {},
            ),
            GateResult(
                "all_arms_completed",
                all(item["steps_completed"] == STEPS for item in results.values()),
                {"steps": STEPS},
            ),
            GateResult(
                "vae_kl_below_no_kl",
                vae_metrics.get("posterior_kl", float("inf"))
                < no_kl_metrics.get("posterior_kl", float("inf")),
                {},
            ),
            GateResult("vae_mse_below_cyclic", vae_mse < vae_metrics["cyclic_mse"], {}),
            GateResult(
                "vae_mse_relative_bound",
                vae_mse <= deterministic_mse * MAX_VAE_MSE_RELATIVE,
                {"ratio": vae_mse / max(deterministic_mse, 1e-30)},
            ),
            GateResult(
                "posterior_variance_retention",
                vae_metrics.get("posterior_mean_variance", 0.0)
                >= MIN_VARIANCE_RETENTION
                * max(initial["smf_beta_vae"]["posterior_mean_variance"], 1e-30),
                {},
            ),
            GateResult(
                "four_phase_updates",
                all(
                    all(
                        any(
                            key.startswith(name) and value > 0.0
                            for key, value in step["update_norms"].items()
                        )
                        for name in base.EXPECTED_TRAINABLE
                    )
                    for item in results.values()
                    for step in item["history"]
                ),
                {},
            ),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("panel_provenance", True, {"panel_sha256": panel_hash}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        scientific = all(gate.passed for gate in gates[2:8])
        resource = all(gate.passed for gate in gates[-3:])
        status = (
            "passed"
            if scientific and resource
            else ("resource_aborted" if not resource else "valid_failed")
        )
        outcome = status
        actions = (
            (NEXT_PASS_ACTION,)
            if status == "passed"
            else ((NEXT_VALID_FAILED_ACTION,) if status == "valid_failed" else ())
        )
        record = EvidenceRecordV2(
            2,
            TRAINING_SCHEMA,
            "diagnostic",
            status,
            str(state["commit"]),
            None,
            spec.config_hash,
            _context(spec, formal, data),
            hashes,
            gates,
            actions,
            None if status == "passed" else "SMF VAE pilot gate failed",
        )
        record.validate()
        payload = _record_payload(
            TRAINING_SCHEMA,
            record,
            {
                "outcome": outcome,
                "validation_hash": validation_hash,
                "preflight_hash": preflight_hash,
                "beta": beta,
                "initial_metrics": initial,
                "arms": {
                    arm: {key: value for key, value in result.items() if key != "panel"}
                    for arm, result in results.items()
                },
                "checkpoints": checkpoints,
                "panel_path": str(panel_path.resolve()),
                "panel_sha256": panel_hash,
                "free_cuda_bytes_before": int(free),
                "total_cuda_bytes": int(total),
                "peak_cuda_bytes": peak,
                "wall_seconds": elapsed,
                "run_storage_bytes": storage,
                "evaluator_calls": 0,
                "formal_candidate_checkpoint": False,
            },
        )
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"passed", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-vae-pilot-validate":
        return validate(args)
    if args.research_action == "smf-vae-pilot-preflight":
        return preflight(args)
    if args.research_action == "smf-vae-pilot-train":
        return train(args)
    raise ValueError(f"unsupported SMF VAE action: {args.research_action}")
