"""Independent Chen-BOGT consumer for the LightGen complex-OLS runtime.

This module is the implementation boundary for the two previously resolved
definition-only contracts.  It deliberately does not import the older MMD
consumer and does not expose a long-training command.  Scientific execution
is a separate, explicitly authorized action; the APIs here are also used by
the bounded engineering preflight and its synthetic tests.
"""

from __future__ import annotations

import copy
import hashlib
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from ldm_project.v2.evidence import atomic_torch_save_new, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import GAUSSIAN_OLS_MODEL, LP01_OLS_MODEL, OLS_MODELS, ComplexOLSState, LightGen2DAlignmentRuntimeConfig, LightGen2DForwardResult, LightGen2DSemanticGenerationInputOLSAlignment, LightGenPhysicalOpticsConfig, OLS_INTERFACE_ID, OLS_PORT_COUNT
from ldm_project.v2.application.research.shared.research_run_control import RECOVERY_CHECKPOINT_SCHEMA, ResearchRunIdentity, check_resume_compatibility


TASK_ID = "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_CONSUMER_V1"
OUTER_GATE = "CHEN_BOGT_TRAINING_CONSUMER_IMPLEMENTATION_PREFLIGHT_READY_V1"
IMPLEMENTATION_CLASSIFICATION = "CHEN_BOGT_TRAINING_CONSUMER_V1_IMPLEMENTED"

MODEL_CONTRACT_ID = "CHEN_BOGT_ON_COMPLEX_OLS_MODEL_CONTRACT_V1"
MODEL_CONTRACT_GATE = "BOGT_ON_COMPLEX_OLS_CONTRACT_RESOLVED"
TRAINING_CONSUMER_CONTRACT_ID = "CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_CONSUMER_CONTRACT_V1"
TRAINING_CONSUMER_CONTRACT_GATE = "DEFINE_CHEN_BOGT_ON_COMPLEX_OLS_TRAINING_CONSUMER_V1_READY"

OLS_GRID_SHAPE = (10, 10)
LOGICAL_BATCH_SIZE = 10
TRAIN_COUNT = 55_000
DEVELOPMENT_COUNT = 5_000
PROTECTED_COUNT = 10_000
SCIENTIFIC_UPDATES = 300
CHECKPOINT_STEPS = (0, 100, 200, 300)
SCHEDULE_SEED = 1234
PRODUCTION_PROPAGATION_GRID = 2048
ENGINEERING_PROPAGATION_GRID = 128

EPS_S = 1.0e-12
EPS_OP = 1.0e-12
VARIANCE_FLOOR = 1.0e-8
PHASE_LOG_EPS = 1.0e-12
PHASE_COVARIANCE_EPS = 1.0e-8
PHASE_MEAN_EPS = 1.0e-8

LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False

ALLOWED_TRAINABLE_PARAMETER_NAMES = ("encoder.phase", "generator.phase")
EXPECTED_PARAMETER_SHAPES = {
    "encoder.phase": (2, 1, 100, 100),
    "generator.phase": (2, 1, 100, 100),
}

TRAIN_ROLE = "train"
TRAIN_SMOKE_ROLE = "train_smoke"
SYNTHETIC_ROLE = "synthetic"
DEV_ROLE = "dev"
PROTECTED_ROLE = "protected"
TEST_ROLE = "test"
SEALED_ROLE = "sealed"
FORBIDDEN_ROLES = frozenset({DEV_ROLE, "development", PROTECTED_ROLE, TEST_ROLE, SEALED_ROLE})

REFERENCES_SCHEMA = "chen_bogt_complex_ols_train_references.v1"
WEIGHTS_SCHEMA = "chen_bogt_complex_ols_loss_weights.v1"
SAMPLER_SCHEMA = "chen_bogt_train_role_sampler.v1"
CHECKPOINT_SCHEMA = "chen_bogt_complex_ols_training_checkpoint.v1"
PHASE_MODEL_SCHEMA = "chen_bogt_wrapped_conditional_phase_model.v1"


# These mappings are the compact, machine-readable transcription of the two
# resolved definition-only authorities.  Their hashes are used in checkpoint
# rejection; no automatic latest-artifact lookup is involved.
MODEL_CONTRACT_PAYLOAD: dict[str, Any] = {
    "id": MODEL_CONTRACT_ID,
    "version": 1,
    "gate": MODEL_CONTRACT_GATE,
    "physical_latent": {
        "space": "C^100",
        "meaning": "raw aligned complex OLS coefficients",
        "port_order": "row-major j=10*r+c",
    },
    "statistical_variable": {
        "formula": "u_j = |z_j|^2 / (s_j + eps_s)",
        "eps_s": EPS_S,
        "support": "u_j >= 0",
    },
    "normalization": {
        "scale": "TRAIN-only mean raw modal power at theta0",
        "accumulation": "float64, canonical source order",
        "frozen_before_updates": True,
    },
    "q": {
        "family": "factorized positive-support truncated normal",
        "source": "class-balanced batch moments",
        "batch_size": LOGICAL_BATCH_SIZE,
        "mean": "sample mean over batch axis",
        "variance": "population variance over batch axis",
        "variance_floor": VARIANCE_FLOOR,
        "class_conditioned": False,
        "full_covariance": False,
    },
    "p": {
        "family": "factorized positive-support truncated normal",
        "loc": 1.0,
        "variance": 0.25,
        "lower": 0.0,
        "fixed": True,
    },
    "loss": {
        "distribution": "closed-form KL(Q||P) for positive truncated normals",
        "reconstruction": "raw CMOS intensity MSE versus resized/clamped SLM intensity",
        "optical": "five-plane relative log-power deviation",
        "mmd": False,
    },
    "trainable": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
    "frozen": ["OLS mode bank", "T=I_100", "W_OLS", "relays", "prior", "normalization", "power references"],
    "phase_model": {
        "timing": "POST_TRAIN_PHASE_FIT",
        "fit_data": "final encoder real-path TRAIN OLS",
        "optimizer_participation": False,
    },
    "paths": {
        "real": "image -> amplitude SLM -> encoder -> OLS -> W_OLS -> generator -> raw CMOS",
        "generated": "u -> conditional phase -> z -> T -> W_OLS -> generator -> raw CMOS",
    },
}

TRAINING_CONSUMER_CONTRACT_PAYLOAD: dict[str, Any] = {
    "id": TRAINING_CONSUMER_CONTRACT_ID,
    "version": 1,
    "gate": TRAINING_CONSUMER_CONTRACT_GATE,
    "model_contract_id": MODEL_CONTRACT_ID,
    "initialization": {
        "kind": "fresh canonical LightGen-aligned constructor",
        "encoder_layers": 2,
        "generator_layers": 2,
        "phase_initialization": "exact zeros",
        "production_grid": PRODUCTION_PROPAGATION_GRID,
    },
    "data_roles": {
        "train_count": TRAIN_COUNT,
        "development_count": DEVELOPMENT_COUNT,
        "protected_count": PROTECTED_COUNT,
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "schedule_seed": SCHEDULE_SEED,
        "development_use": "diagnostics only",
        "protected_test_sealed_use": "forbidden",
    },
    "statistics": {
        "pass": "one no-grad TRAIN pass at theta0",
        "outputs": ["s_j", "r_p"],
        "accumulation": "float64",
        "frozen_before_optimizer": True,
    },
    "weight_calibration": {
        "rule": "3*(1/g_k)/sum(1/g_i)",
        "batch": "first seed-1234 class-balanced TRAIN batch",
        "frozen": True,
    },
    "optimizer": {
        "name": "Adam",
        "learning_rate": LEARNING_RATE,
        "betas": list(ADAM_BETAS),
        "eps": ADAM_EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": AMSGRAD,
        "gradient_accumulation": False,
        "clipping": None,
        "autocast": False,
        "grad_scaler": False,
    },
    "budget": {"updates": SCIENTIFIC_UPDATES, "checkpoints": list(CHECKPOINT_STEPS)},
    "selection": "fixed final update step 300",
    "phase_fit": "once after final checkpoint freeze, TRAIN only",
    "one_shot": True,
}

MODEL_CONTRACT_SHA256 = canonical_sha256(MODEL_CONTRACT_PAYLOAD)
TRAINING_CONSUMER_CONTRACT_SHA256 = canonical_sha256(TRAINING_CONSUMER_CONTRACT_PAYLOAD)


class TrainingContractError(ValueError):
    """Raised when a frozen contract or runtime boundary is invalid."""


class DataRoleError(TrainingContractError):
    """Raised before a forbidden data role can cross a consumer boundary."""


class CheckpointError(TrainingContractError):
    """Raised when a checkpoint cannot be safely consumed."""


def _require(condition: bool, message: str, error_type: type[Exception] = TrainingContractError) -> None:
    if not condition:
        raise error_type(message)


def _mapping(value: object, name: str, error_type: type[Exception] = TrainingContractError) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping", error_type)
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_finite(value: torch.Tensor, name: str) -> None:
    _require(torch.isfinite(value).all().item(), f"{name} is nonfinite")


def _tensor_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Hash one state boundary without returning a child-hash inventory."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_identity(repo_root: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _validate_contract_payloads() -> None:
    _require(MODEL_CONTRACT_PAYLOAD["id"] == MODEL_CONTRACT_ID, "model contract identity changed")
    _require(TRAINING_CONSUMER_CONTRACT_PAYLOAD["model_contract_id"] == MODEL_CONTRACT_ID, "training consumer model binding changed")
    _require(MODEL_CONTRACT_PAYLOAD["trainable"] == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "model trainable surface changed")
    _require(TRAINING_CONSUMER_CONTRACT_PAYLOAD["budget"]["updates"] == SCIENTIFIC_UPDATES, "training budget changed")
    _require(TRAINING_CONSUMER_CONTRACT_PAYLOAD["budget"]["checkpoints"] == list(CHECKPOINT_STEPS), "checkpoint schedule changed")
    _require(TRAINING_CONSUMER_CONTRACT_PAYLOAD["selection"] == "fixed final update step 300", "checkpoint selection changed")


def frozen_contract_identity() -> dict[str, Any]:
    _validate_contract_payloads()
    return {
        "model": {
            "id": MODEL_CONTRACT_ID,
            "gate": MODEL_CONTRACT_GATE,
            "sha256": MODEL_CONTRACT_SHA256,
        },
        "training_consumer": {
            "id": TRAINING_CONSUMER_CONTRACT_ID,
            "gate": TRAINING_CONSUMER_CONTRACT_GATE,
            "sha256": TRAINING_CONSUMER_CONTRACT_SHA256,
        },
    }


def _validate_complex_ols(z: torch.Tensor, name: str, *, batch_size: int | None = None) -> None:
    _require(isinstance(z, torch.Tensor), f"{name} must be a tensor")
    _require(torch.is_complex(z) and z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT, f"{name} must have complex shape [B,100]")
    if batch_size is not None:
        _require(z.shape[0] == batch_size, f"{name} must have batch size {batch_size}")
    _validate_finite(z, name)


def _validate_image_batch(images: torch.Tensor, *, exact_batch: bool) -> None:
    _require(isinstance(images, torch.Tensor), "images must be a tensor")
    _require(images.ndim == 4 and tuple(images.shape[1:]) == (1, 28, 28), "images must have shape [B,1,28,28]")
    if exact_batch:
        _require(images.shape[0] == LOGICAL_BATCH_SIZE, "training update requires logical batch size 10")
    _require(images.dtype == torch.float32 and not torch.is_complex(images), "images must be real float32")
    _validate_finite(images, "images")
    _require(images.numel() == 0 or bool((images.min() >= 0.0).item() and (images.max() <= 1.0).item()), "images must be normalized to [0,1]")


@dataclass
class DataReadLedger:
    """Counts reads by role so engineering preflight cannot hide leakage."""

    counts: dict[str, int] = field(
        default_factory=lambda: {
            TRAIN_ROLE: 0,
            DEV_ROLE: 0,
            PROTECTED_ROLE: 0,
            TEST_ROLE: 0,
            SEALED_ROLE: 0,
            TRAIN_SMOKE_ROLE: 0,
            SYNTHETIC_ROLE: 0,
        }
    )

    def record(self, role: str, count: int) -> None:
        _require(count >= 0, "read count must be nonnegative", DataRoleError)
        key = role if role in self.counts else role
        self.counts[key] = self.counts.get(key, 0) + int(count)

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)

    def assert_forbidden_zero(self) -> None:
        for role in (DEV_ROLE, PROTECTED_ROLE, TEST_ROLE, SEALED_ROLE):
            _require(self.counts.get(role, 0) == 0, f"forbidden role was read: {role}", DataRoleError)


def guard_data_role(
    role: str,
    *,
    operation: str,
    engineering_only: bool,
    ledger: DataReadLedger | None = None,
    count: int = 0,
) -> None:
    role = str(role)
    if role in FORBIDDEN_ROLES:
        raise DataRoleError(f"{operation} rejects forbidden data role: {role}")
    _require(role in {TRAIN_ROLE, TRAIN_SMOKE_ROLE, SYNTHETIC_ROLE}, f"{operation} data role is not registered: {role}", DataRoleError)
    if role in {TRAIN_SMOKE_ROLE, SYNTHETIC_ROLE}:
        _require(engineering_only, f"{operation} smoke/synthetic role requires engineering_only=true", DataRoleError)
    if ledger is not None and count:
        ledger.record(role, count)


@dataclass(frozen=True)
class FrozenBogtReferences:
    """Frozen TRAIN-derived modal scales and five physical-plane references."""

    scale: torch.Tensor
    power_reference: torch.Tensor
    data_role: str
    count: int
    engineering_only: bool
    theta0_identity: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    basis_id: str = LP01_OLS_MODEL
    mode_geometry: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        scale = self.scale.detach().to(dtype=torch.float64, device="cpu").clone()
        reference = self.power_reference.detach().to(dtype=torch.float64, device="cpu").clone()
        _require(scale.shape == (OLS_PORT_COUNT,), "s_j must have shape [100]")
        _require(reference.shape == (5,), "r_p must have shape [5]")
        _validate_finite(scale, "s_j")
        _validate_finite(reference, "r_p")
        _require(bool((scale >= 0.0).all().item()), "s_j must be nonnegative")
        _require(bool((reference >= 0.0).all().item()), "r_p must be nonnegative")
        _require(self.data_role in {TRAIN_ROLE, TRAIN_SMOKE_ROLE, SYNTHETIC_ROLE}, "reference data role is invalid", DataRoleError)
        _require(self.count > 0, "reference count must be positive")
        if self.data_role == TRAIN_ROLE:
            _require(self.count == TRAIN_COUNT and not self.engineering_only, "scientific TRAIN references require exactly 55,000 rows", DataRoleError)
        else:
            _require(self.engineering_only, "engineering references must be explicitly marked", DataRoleError)
        _require(isinstance(self.theta0_identity, Mapping), "theta0 identity is required")
        _require(isinstance(self.data_identity, Mapping), "data identity is required")
        _require(self.basis_id in OLS_MODELS, "reference basis_id is unsupported", CheckpointError)
        if self.basis_id == GAUSSIAN_OLS_MODEL:
            _require(isinstance(self.mode_geometry, Mapping), "Gaussian references require explicit mode geometry", CheckpointError)
            _require(self.mode_geometry.get("basis_id") == self.basis_id, "Gaussian reference geometry/basis mismatch", CheckpointError)
            object.__setattr__(self, "mode_geometry", copy.deepcopy(dict(self.mode_geometry)))
        elif self.mode_geometry is not None:
            _require(isinstance(self.mode_geometry, Mapping) and self.mode_geometry.get("basis_id") == self.basis_id, "LP01 reference geometry/basis mismatch", CheckpointError)
        scale.requires_grad_(False)
        reference.requires_grad_(False)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "power_reference", reference)

    @property
    def scale_sha256(self) -> str:
        return tensor_sha256(self.scale)

    @property
    def power_reference_sha256(self) -> str:
        return tensor_sha256(self.power_reference)

    def identity(self) -> dict[str, Any]:
        identity = {
            "schema": REFERENCES_SCHEMA,
            "data_role": self.data_role,
            "count": self.count,
            "engineering_only": self.engineering_only,
            "scale_sha256": self.scale_sha256,
            "power_reference_sha256": self.power_reference_sha256,
            "theta0_state_identity_sha256": self.theta0_identity.get("state_identity_sha256"),
            "data_identity": dict(self.data_identity),
        }
        if self.basis_id == GAUSSIAN_OLS_MODEL:
            identity["basis_id"] = self.basis_id
            identity["mode_geometry"] = dict(self.mode_geometry or {})
        return identity

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFERENCES_SCHEMA,
            "scale": self.scale.clone(),
            "power_reference": self.power_reference.clone(),
            "data_role": self.data_role,
            "count": self.count,
            "engineering_only": self.engineering_only,
            "theta0_identity": dict(self.theta0_identity),
            "data_identity": dict(self.data_identity),
            "identity": self.identity(),
        }
        if self.basis_id == GAUSSIAN_OLS_MODEL:
            payload["basis_id"] = self.basis_id
            payload["mode_geometry"] = dict(self.mode_geometry or {})
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenBogtReferences":
        _require(payload.get("schema") == REFERENCES_SCHEMA, "references schema mismatch", CheckpointError)
        scale = payload.get("scale")
        power = payload.get("power_reference")
        _require(isinstance(scale, torch.Tensor) and isinstance(power, torch.Tensor), "references tensors are missing", CheckpointError)
        return cls(
            scale,
            power,
            str(payload.get("data_role")),
            int(payload.get("count")),
            bool(payload.get("engineering_only")),
            _mapping(payload.get("theta0_identity"), "theta0 identity", CheckpointError),
            _mapping(payload.get("data_identity"), "data identity", CheckpointError),
            str(payload.get("basis_id", LP01_OLS_MODEL)),
            payload.get("mode_geometry"),
        )


@dataclass(frozen=True)
class BogtLossWeights:
    alpha: float
    beta: float
    gamma: float
    gradient_norms: Mapping[str, float]
    data_role: str
    engineering_only: bool
    calibration_batch_identity: Mapping[str, Any]
    references_identity_sha256: str

    def __post_init__(self) -> None:
        values = (self.alpha, self.beta, self.gamma)
        _require(all(math.isfinite(float(value)) and float(value) > 0.0 for value in values), "loss weights must be finite and positive")
        _require(math.isclose(sum(values), 3.0, rel_tol=0.0, abs_tol=1.0e-12), "loss weights must sum to 3")
        _require(set(self.gradient_norms) == {"dist", "rec", "op"}, "gradient norm keys are incomplete")
        _require(all(math.isfinite(float(value)) and float(value) > 0.0 for value in self.gradient_norms.values()), "gradient norms must be finite and positive")
        _require(self.data_role in {TRAIN_ROLE, TRAIN_SMOKE_ROLE, SYNTHETIC_ROLE}, "weight calibration data role is invalid", DataRoleError)
        if self.data_role != TRAIN_ROLE:
            _require(self.engineering_only, "smoke weights must be engineering-only", DataRoleError)
        _require(isinstance(self.calibration_batch_identity, Mapping), "calibration batch identity is required")

    def identity(self) -> dict[str, Any]:
        return {
            "schema": WEIGHTS_SCHEMA,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "gradient_norms": dict(self.gradient_norms),
            "data_role": self.data_role,
            "engineering_only": self.engineering_only,
            "calibration_batch_identity": dict(self.calibration_batch_identity),
            "references_identity_sha256": self.references_identity_sha256,
        }

    def to_payload(self) -> dict[str, Any]:
        return self.identity()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BogtLossWeights":
        _require(payload.get("schema") == WEIGHTS_SCHEMA, "weights schema mismatch", CheckpointError)
        return cls(
            float(payload["alpha"]),
            float(payload["beta"]),
            float(payload["gamma"]),
            {str(key): float(value) for key, value in _mapping(payload.get("gradient_norms"), "gradient norms", CheckpointError).items()},
            str(payload["data_role"]),
            bool(payload["engineering_only"]),
            _mapping(payload.get("calibration_batch_identity"), "calibration batch identity", CheckpointError),
            str(payload["references_identity_sha256"]),
        )


@dataclass(frozen=True)
class QBatchMoments:
    mean: torch.Tensor
    variance: torch.Tensor

    def __post_init__(self) -> None:
        _require(self.mean.shape == (OLS_PORT_COUNT,) and self.variance.shape == (OLS_PORT_COUNT,), "Q moments must have shape [100]")
        _validate_finite(self.mean, "Q mean")
        _validate_finite(self.variance, "Q variance")
        _require(bool((self.mean >= 0.0).all().item()), "Q mean must be nonnegative")
        _require(bool((self.variance >= VARIANCE_FLOOR).all().item()), "Q variance floor was not applied")


@dataclass(frozen=True)
class PositiveTruncatedNormal:
    """The underlying loc/variance parameters of TN_+(loc, variance)."""

    loc: torch.Tensor
    variance: torch.Tensor

    def __post_init__(self) -> None:
        _require(self.loc.shape == self.variance.shape, "truncated-normal parameter shapes differ")
        _validate_finite(self.loc, "truncated-normal loc")
        _validate_finite(self.variance, "truncated-normal variance")
        _require(bool((self.variance > 0.0).all().item()), "truncated-normal variance must be positive")

    def to(self, *, device: torch.device, dtype: torch.dtype | None = None) -> "PositiveTruncatedNormal":
        target_dtype = self.loc.dtype if dtype is None else dtype
        return PositiveTruncatedNormal(self.loc.to(device=device, dtype=target_dtype), self.variance.to(device=device, dtype=target_dtype))


def statistical_variable(z: torch.Tensor, references: FrozenBogtReferences | torch.Tensor) -> torch.Tensor:
    """Compute the contract's nonnegative normalized modal-power variable."""
    _validate_complex_ols(z, "z")
    scale = references.scale if isinstance(references, FrozenBogtReferences) else references
    _require(isinstance(scale, torch.Tensor) and scale.shape == (OLS_PORT_COUNT,), "s_j must have shape [100]")
    scale = scale.detach().to(device=z.device, dtype=torch.float64)
    _validate_finite(scale, "s_j")
    _require(bool((scale >= 0.0).all().item()), "s_j must be nonnegative")
    power = z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()
    value = power / (scale + EPS_S)
    _validate_finite(value, "u")
    _require(bool((value >= 0.0).all().item()), "u must be nonnegative")
    return value


def q_batch_moments(u: torch.Tensor) -> QBatchMoments:
    """Return batch-axis mean and population variance for factorized Q."""
    _require(isinstance(u, torch.Tensor) and u.ndim == 2 and u.shape[1] == OLS_PORT_COUNT, "u must have shape [B,100]")
    _validate_finite(u, "u")
    _require(bool((u >= 0.0).all().item()), "u must be nonnegative")
    mean = u.to(torch.float64).mean(dim=0)
    variance = (u.to(torch.float64) - mean).square().mean(dim=0).clamp_min(VARIANCE_FLOOR)
    return QBatchMoments(mean, variance)


def q_from_batch(u: torch.Tensor) -> PositiveTruncatedNormal:
    _require(u.shape[0] == LOGICAL_BATCH_SIZE, "Q requires logical batch size 10")
    moments = q_batch_moments(u)
    return PositiveTruncatedNormal(moments.mean, moments.variance)


def fixed_prior(*, device: torch.device, dtype: torch.dtype = torch.float64) -> PositiveTruncatedNormal:
    return PositiveTruncatedNormal(
        torch.full((OLS_PORT_COUNT,), 1.0, device=device, dtype=dtype),
        torch.full((OLS_PORT_COUNT,), 0.25, device=device, dtype=dtype),
    )


def _tn_log_normalizer(loc: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    sigma = variance.sqrt()
    return torch.special.log_ndtr(loc / sigma)


def truncated_normal_positive_log_prob(
    x: torch.Tensor,
    loc: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """Log-density of the normal distribution truncated to [0, infinity)."""
    x, loc, variance = torch.broadcast_tensors(
        x.to(torch.float64), loc.to(torch.float64), variance.to(torch.float64)
    )
    _validate_finite(loc, "TN loc")
    _validate_finite(variance, "TN variance")
    _require(bool((variance > 0.0).all().item()), "TN variance must be positive")
    sigma = variance.sqrt()
    standardized = (x - loc) / sigma
    log_pdf = -0.5 * standardized.square() - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    log_density = log_pdf - _tn_log_normalizer(loc, variance)
    return torch.where(x >= 0.0, log_density, torch.full_like(log_density, -torch.inf))


def _tn_truncated_moments(loc: torch.Tensor, variance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = variance.sqrt()
    standardized_loc = loc / sigma
    log_z = _tn_log_normalizer(loc, variance)
    log_lambda = -0.5 * standardized_loc.square() - 0.5 * math.log(2.0 * math.pi) - log_z
    lam = torch.exp(log_lambda)
    mean = loc + sigma * lam
    truncated_variance = variance * (1.0 - standardized_loc * lam - lam.square())
    truncated_variance = truncated_variance.clamp_min(torch.finfo(torch.float64).tiny)
    return mean, truncated_variance


def tn_positive_kl(
    q_loc: torch.Tensor,
    q_variance: torch.Tensor,
    p_loc: torch.Tensor,
    p_variance: torch.Tensor,
) -> torch.Tensor:
    """Closed-form elementwise KL(TN_+(q)||TN_+(p))."""
    tensors = [value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in (q_loc, q_variance, p_loc, p_variance)]
    device = tensors[0].device
    q_loc, q_variance, p_loc, p_variance = [value.to(device=device, dtype=torch.float64) for value in tensors]
    q_loc, q_variance, p_loc, p_variance = torch.broadcast_tensors(q_loc, q_variance, p_loc, p_variance)
    _validate_finite(q_loc, "Q loc")
    _validate_finite(q_variance, "Q variance")
    _validate_finite(p_loc, "P loc")
    _validate_finite(p_variance, "P variance")
    _require(bool((q_variance > 0.0).all().item()), "Q variance must be positive")
    _require(bool((p_variance > 0.0).all().item()), "P variance must be positive")
    q_mean, q_var = _tn_truncated_moments(q_loc, q_variance)
    log_sigma_ratio = 0.5 * (torch.log(p_variance) - torch.log(q_variance))
    log_normalizer_ratio = _tn_log_normalizer(p_loc, p_variance) - _tn_log_normalizer(q_loc, q_variance)
    p_quadratic = (q_var + (q_mean - p_loc).square()) / (2.0 * p_variance)
    q_quadratic = (q_var + (q_mean - q_loc).square()) / (2.0 * q_variance)
    value = log_sigma_ratio + log_normalizer_ratio + p_quadratic - q_quadratic
    _validate_finite(value, "TN KL")
    return value


def truncated_normal_positive_kl(q: PositiveTruncatedNormal, p: PositiveTruncatedNormal) -> torch.Tensor:
    return tn_positive_kl(q.loc, q.variance, p.loc, p.variance)


def physical_plane_power(field: Any) -> torch.Tensor:
    """Return P=dy*dx*sum(|E|^2) for each batch item."""
    tensor = getattr(field, "tensor", None)
    plane = getattr(field, "plane", None)
    _require(isinstance(tensor, torch.Tensor) and torch.is_complex(tensor), "optical field must be complex")
    _require(plane is not None, "optical field plane metadata is required")
    _require(tensor.ndim >= 3, "optical field must include a batch dimension")
    intensity = tensor.real.to(torch.float64).square() + tensor.imag.to(torch.float64).square()
    pitch_area = float(plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1])
    power = intensity.reshape(intensity.shape[0], -1).sum(dim=1) * pitch_area
    _validate_finite(power, "optical plane power")
    _require(bool((power >= 0.0).all().item()), "optical plane power must be nonnegative")
    return power


def contracted_plane_powers(result: LightGen2DForwardResult) -> torch.Tensor:
    _require(len(result.encoder_records) == 2, "BOGT requires two encoder records")
    _require(len(result.generator_records) == 2, "BOGT requires two generator records")
    fields = (*result.encoder_records, result.generator_input_field, *result.generator_records)
    powers = torch.stack([physical_plane_power(field) for field in fields], dim=1)
    _require(powers.shape[1] == 5, "BOGT requires five optical planes")
    return powers


def optical_power_loss_from_powers(power: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    _require(power.ndim == 2 and power.shape[1] == 5, "power must have shape [B,5]")
    _require(reference.shape == (5,), "power reference must have shape [5]")
    _validate_finite(power, "power")
    _validate_finite(reference, "power reference")
    _require(bool((power >= 0.0).all().item()), "power must be nonnegative")
    _require(bool((reference >= 0.0).all().item()), "power reference must be nonnegative")
    ref = reference.to(device=power.device, dtype=torch.float64)
    ratio = (power.to(torch.float64) + EPS_OP) / (ref + EPS_OP)
    value = torch.log(ratio).square().mean()
    _validate_finite(value, "L_op")
    return value


def optical_power_loss(result: LightGen2DForwardResult, references: FrozenBogtReferences) -> torch.Tensor:
    return optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)


@dataclass(frozen=True)
class BogtLossBreakdown:
    l_dist: torch.Tensor
    l_rec: torch.Tensor
    l_op: torch.Tensor
    l_total: torch.Tensor
    q_moments: QBatchMoments
    plane_powers: torch.Tensor
    result: LightGen2DForwardResult

    @property
    def L_dist(self) -> torch.Tensor:
        return self.l_dist

    @property
    def L_rec(self) -> torch.Tensor:
        return self.l_rec

    @property
    def L_op(self) -> torch.Tensor:
        return self.l_op

    @property
    def L_total(self) -> torch.Tensor:
        return self.l_total

    def metrics(self) -> dict[str, float]:
        return {
            "L_dist": float(self.l_dist.detach().cpu()),
            "L_rec": float(self.l_rec.detach().cpu()),
            "L_op": float(self.l_op.detach().cpu()),
            "L_total": float(self.l_total.detach().cpu()),
        }


@dataclass(frozen=True)
class AutogradDiagnostics:
    projection_active: bool
    writer_active: bool
    gradient_norms: Mapping[str, float]
    trainable_gradients_finite: bool
    forbidden_gradients_absent: bool


@dataclass(frozen=True)
class BogtUpdateResult:
    completed_updates: int
    loss: BogtLossBreakdown
    autograd: AutogradDiagnostics
    parameter_update_norms: Mapping[str, float]
    frozen_state_unchanged: bool

    def metrics(self) -> dict[str, Any]:
        q_mean = self.loss.q_moments.mean.detach().to(torch.float64)
        q_variance = self.loss.q_moments.variance.detach().to(torch.float64)
        plane_power = self.loss.plane_powers.detach().to(torch.float64)
        return {
            "completed_updates": self.completed_updates,
            "loss": self.loss.metrics(),
            "gradient_norms": dict(self.autograd.gradient_norms),
            "parameter_update_norms": dict(self.parameter_update_norms),
            "frozen_state_unchanged": self.frozen_state_unchanged,
            "q_moment_summary": {
                "mean_mean": float(q_mean.mean().cpu()),
                "mean_min": float(q_mean.min().cpu()),
                "mean_max": float(q_mean.max().cpu()),
                "variance_mean": float(q_variance.mean().cpu()),
                "variance_min": float(q_variance.min().cpu()),
                "variance_max": float(q_variance.max().cpu()),
            },
            "power_summary": {
                "mean": [float(value) for value in plane_power.mean(dim=0).cpu().tolist()],
                "min": [float(value) for value in plane_power.min(dim=0).values.cpu().tolist()],
                "max": [float(value) for value in plane_power.max(dim=0).values.cpu().tolist()],
            },
            "finite_state": True,
        }


def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            if not torch.equal(first, second):
                return False
        elif isinstance(first, Mapping) and isinstance(second, Mapping):
            if not _state_equal(first, second):
                return False
        elif isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
            if len(first) != len(second) or any(not _state_equal({"v": a}, {"v": b}) for a, b in zip(first, second, strict=True)):
                return False
        elif first != second:
            return False
    return True


def _resume_identity(value: Any) -> Any:
    """Remove provenance fields from resume comparisons, not loaded state."""
    if isinstance(value, Mapping):
        provenance_keys = {
            "branch",
            "head",
            "dirty",
            "worktree_dirty",
            "git",
            "runtime_module",
            "source_manifest",
            "source_manifest_id",
            "execution_source_semantics_id",
            "source_semantics_id",
            "status_sha256",
        }
        return {
            key: _resume_identity(item)
            for key, item in value.items()
            if str(key).lower() not in provenance_keys
            and str(key).lower() not in {"sha256", "hash", "digest", "dirty_patch_digest"}
            and not str(key).lower().endswith(("_sha256", "_hash", "_digest"))
        }
    if isinstance(value, (list, tuple)):
        return [_resume_identity(item) for item in value]
    return value


class LightGen2DChenBOGTComplexOLSTrainingConsumer(nn.Module):
    """The unique optimizer-facing consumer for the resolved Chen contract."""

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        propagation_grid_size: int = PRODUCTION_PROPAGATION_GRID,
        ols_model: str = LP01_OLS_MODEL,
        physical_optics: LightGenPhysicalOpticsConfig | None = None,
    ) -> None:
        super().__init__()
        _validate_contract_payloads()
        _require(propagation_grid_size in {ENGINEERING_PROPAGATION_GRID, PRODUCTION_PROPAGATION_GRID}, "only declared production or engineering grid is supported")
        self.device = torch.device(device)
        self.runtime = LightGen2DSemanticGenerationInputOLSAlignment(
            LightGen2DAlignmentRuntimeConfig(
                propagation_grid_size=propagation_grid_size,
                ols_model=ols_model,
                physical_optics=physical_optics,
            )
        ).to(self.device)
        self.runtime.train()
        self._references: FrozenBogtReferences | None = None
        self._weights: BogtLossWeights | None = None
        self._enforce_parameter_firewall()
        self._theta0_trainable_state = self.trainable_state_dict()
        self._theta0_frozen_state = self._frozen_state_snapshot()
        self._theta0_identity = self._make_theta0_identity()

    def _named_runtime_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(self.runtime.named_parameters())

    def _enforce_parameter_firewall(self) -> None:
        named = self._named_runtime_parameters()
        names = [name for name, _ in named]
        _require(names == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), f"runtime parameter surface changed: {names}")
        _require(len({id(parameter) for _, parameter in named}) == len(named), "runtime parameter objects are duplicated")
        for name, parameter in named:
            _require(name in EXPECTED_PARAMETER_SHAPES, f"unexpected runtime parameter: {name}")
            _require(tuple(parameter.shape) == EXPECTED_PARAMETER_SHAPES[name], f"parameter shape changed: {name}")
            _require(parameter.dtype == torch.float32, f"parameter dtype changed: {name}")
            parameter.requires_grad_(True)
        for name, buffer in self.runtime.named_buffers():
            if buffer.requires_grad:
                raise TrainingContractError(f"frozen runtime buffer unexpectedly requires gradients: {name}")
        _require(torch.equal(self.runtime.ols.transport.detach().cpu(), torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "OLS transport is not T=I_100")

    def _frozen_state_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.runtime.state_dict().items()
            if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES
        }

    @property
    def frozen_state_sha256(self) -> str:
        return _tensor_state_digest(self._frozen_state_snapshot())

    @property
    def frozen_state_unchanged(self) -> bool:
        return _state_equal(self._frozen_state_snapshot(), self._theta0_frozen_state)

    @property
    def references(self) -> FrozenBogtReferences | None:
        return self._references

    @property
    def weights(self) -> BogtLossWeights | None:
        return self._weights

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._named_runtime_parameters()
            if name in ALLOWED_TRAINABLE_PARAMETER_NAMES
        }

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(set(state) == set(ALLOWED_TRAINABLE_PARAMETER_NAMES), "trainable checkpoint keys differ from allowlist", CheckpointError)
        parameters = dict(self._named_runtime_parameters())
        with torch.no_grad():
            for name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
                value = state[name]
                _require(isinstance(value, torch.Tensor), f"checkpoint state is not a tensor: {name}", CheckpointError)
                _require(tuple(value.shape) == tuple(parameters[name].shape) and value.dtype == parameters[name].dtype, f"checkpoint parameter shape/dtype changed: {name}", CheckpointError)
                _validate_finite(value, f"checkpoint parameter {name}")
                parameters[name].copy_(value.to(self.device))
        self._enforce_parameter_firewall()

    def _make_theta0_identity(self) -> dict[str, Any]:
        trainable_digest = _tensor_state_digest(self._theta0_trainable_state)
        frozen_digest = _tensor_state_digest(self._theta0_frozen_state)
        state_identity = canonical_sha256({"trainable": trainable_digest, "frozen": frozen_digest})
        return {
            "schema": "chen_bogt_theta0_initialization_fingerprint.v1",
            "runtime_semantic_identity": "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1",
            "runtime_module": _relative_path(Path(__file__).with_name("lightgen_2d_semantic_generation_input_ols_alignment.py"), _repo_root()),
            "config": self._runtime_config_payload(),
            "phase_zero": {
                name: bool(torch.equal(self._theta0_trainable_state[name], torch.zeros_like(self._theta0_trainable_state[name])))
                for name in ALLOWED_TRAINABLE_PARAMETER_NAMES
            },
            "trainable_state_sha256": trainable_digest,
            "frozen_runtime_state_sha256": frozen_digest,
            "state_identity_sha256": state_identity,
            "ols_interface": OLS_INTERFACE_ID,
            "port_order": "row-major j=10*r+c",
            "transport": "T=I_100 coefficient space",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "git": git_identity(),
        }

    def initialization_fingerprint(self) -> dict[str, Any]:
        return copy.deepcopy(self._theta0_identity)

    def assert_theta0(self) -> None:
        _require(_state_equal(self.trainable_state_dict(), self._theta0_trainable_state), "consumer is not at canonical theta0")
        _require(_state_equal(self._frozen_state_snapshot(), self._theta0_frozen_state), "frozen runtime differs from canonical theta0")
        _require(all(self._theta0_identity["phase_zero"].values()), "canonical theta0 phase state is not zero")

    def runtime_identity(self) -> dict[str, Any]:
        root = _repo_root()
        runtime_path = Path(__file__).with_name("lightgen_2d_semantic_generation_input_ols_alignment.py")
        return {
            "identity": "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1",
            "module": _relative_path(runtime_path, root),
            "module_sha256": file_sha256(runtime_path),
            "config": self._runtime_config_payload(),
            "ols_interface": OLS_INTERFACE_ID,
            "ols_shape": [OLS_PORT_COUNT],
            "ols_grid_shape": list(OLS_GRID_SHAPE),
            "port_order": "row-major j=10*r+c",
            "transport": "T=I_100 coefficient space",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "frozen_runtime_state_sha256": self.frozen_state_sha256,
            "transport_sha256": tensor_sha256(self.runtime.ols.transport.detach().cpu()),
            "production_propagation_grid": PRODUCTION_PROPAGATION_GRID,
        }

    def _runtime_config_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "propagation_grid_size": self.runtime.config.propagation_grid_size,
        }
        if self.runtime.config.ols_model == GAUSSIAN_OLS_MODEL:
            payload["ols_model"] = self.runtime.config.ols_model
            payload["mode_geometry"] = self.runtime.ols.mode_identity()
        if self.runtime.config.physical_optics is not None:
            payload["ols_model"] = self.runtime.config.ols_model
            payload["mode_geometry"] = self.runtime.ols.mode_identity()
            payload["physical_optics"] = self.runtime.config.physical_optics.to_dict()
        return payload

    def parameter_manifest(self) -> dict[str, Any]:
        self._enforce_parameter_firewall()
        named = self._named_runtime_parameters()
        state = self.runtime.state_dict()
        frozen = {name: value for name, value in state.items() if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES}
        return {
            "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
            "optimizer_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
            "parameter_shapes": {name: list(parameter.shape) for name, parameter in named},
            "parameter_dtypes": {name: str(parameter.dtype) for name, parameter in named},
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in named),
            "frozen_runtime_state_tensor_count": len(frozen),
            "frozen_runtime_state_numel": sum(value.numel() for value in frozen.values() if torch.is_tensor(value)),
            "frozen_runtime_state_sha256": self.frozen_state_sha256,
        }

    def create_optimizer(self, candidate_parameters: Iterable[nn.Parameter] | None = None) -> torch.optim.Adam:
        self._enforce_parameter_firewall()
        expected = [parameter for _, parameter in self._named_runtime_parameters()]
        candidates = expected if candidate_parameters is None else list(candidate_parameters)
        _require(len(candidates) == len(expected) and all(actual is wanted for actual, wanted in zip(candidates, expected, strict=True)), "optimizer candidate list differs from exact allowlist")
        _require(all(parameter.requires_grad for parameter in candidates), "optimizer candidate is not trainable")
        optimizer = torch.optim.Adam(
            candidates,
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPSILON,
            weight_decay=WEIGHT_DECAY,
            amsgrad=AMSGRAD,
        )
        self.optimizer_manifest(optimizer)
        return optimizer

    def optimizer_manifest(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        _require(isinstance(optimizer, torch.optim.Adam), "optimizer must be Adam")
        _require(len(optimizer.param_groups) == 1, "optimizer must have one parameter group")
        expected = [parameter for _, parameter in self._named_runtime_parameters()]
        actual = list(optimizer.param_groups[0]["params"])
        _require(len(actual) == len(expected) and all(a is b for a, b in zip(actual, expected, strict=True)), "optimizer IDs differ from allowlist")
        group = optimizer.param_groups[0]
        _require(
            group["lr"] == LEARNING_RATE
            and tuple(group["betas"]) == ADAM_BETAS
            and group["eps"] == ADAM_EPSILON
            and group["weight_decay"] == WEIGHT_DECAY
            and group["amsgrad"] is AMSGRAD,
            "optimizer configuration differs from contract",
        )
        return {
            "name": "Adam",
            "parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
            "learning_rate": group["lr"],
            "betas": list(group["betas"]),
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
            "amsgrad": group["amsgrad"],
            "gradient_accumulation": False,
            "clipping": None,
            "autocast": False,
            "grad_scaler": False,
        }

    def bind_references(self, references: FrozenBogtReferences, *, for_scientific: bool = False) -> None:
        _require(isinstance(references, FrozenBogtReferences), "consumer requires FrozenBogtReferences")
        _require(references.basis_id == self.runtime.ols.basis_id, "references belong to another OLS modal basis", CheckpointError)
        if self.runtime.ols.basis_id == GAUSSIAN_OLS_MODEL:
            _require(dict(references.mode_geometry or {}) == self.runtime.ols.mode_identity(), "Gaussian reference geometry differs from the runtime", CheckpointError)
        elif references.mode_geometry is not None:
            _require(dict(references.mode_geometry) == self.runtime.ols.mode_identity(), "LP01 reference geometry differs from the runtime", CheckpointError)
        if for_scientific:
            _require(references.data_role == TRAIN_ROLE and not references.engineering_only, "scientific consumer cannot bind engineering references", DataRoleError)
        _require(
            _resume_identity(references.theta0_identity) == _resume_identity(self._theta0_identity),
            "references were collected from another theta0 configuration",
        )
        if self._references is not None:
            same_values = (
                torch.equal(self._references.scale, references.scale)
                and torch.equal(self._references.power_reference, references.power_reference)
                and self._references.data_role == references.data_role
                and self._references.count == references.count
                and self._references.engineering_only is references.engineering_only
                and _resume_identity(self._references.theta0_identity) == _resume_identity(references.theta0_identity)
                and _resume_identity(self._references.data_identity) == _resume_identity(references.data_identity)
            )
            _require(same_values, "frozen references were already bound to different loaded values")
        self._references = references

    def bind_weights(self, weights: BogtLossWeights, *, for_scientific: bool = False) -> None:
        _require(isinstance(weights, BogtLossWeights), "consumer requires BogtLossWeights")
        _require(self._references is not None, "bind references before weights")
        if for_scientific:
            _require(weights.data_role == TRAIN_ROLE and not weights.engineering_only, "scientific consumer cannot bind engineering weights", DataRoleError)
        if self._weights is not None:
            same_values = (
                (self._weights.alpha, self._weights.beta, self._weights.gamma)
                == (weights.alpha, weights.beta, weights.gamma)
                and _resume_identity(self._weights.gradient_norms) == _resume_identity(weights.gradient_norms)
                and self._weights.data_role == weights.data_role
                and self._weights.engineering_only is weights.engineering_only
                and _resume_identity(self._weights.calibration_batch_identity)
                == _resume_identity(weights.calibration_batch_identity)
            )
            _require(same_values, "loss weights were already bound to different loaded values")
        self._weights = weights

    def _bound_for_role(self, role: str, operation: str) -> tuple[FrozenBogtReferences, BogtLossWeights]:
        _require(self._references is not None and self._weights is not None, f"{operation} requires frozen references and weights")
        references = self._references
        weights = self._weights
        guard_data_role(role, operation=operation, engineering_only=references.engineering_only)
        _require(role == references.data_role, f"{operation} role does not match bound references", DataRoleError)
        _require(weights.data_role == role and weights.engineering_only == references.engineering_only, f"{operation} role does not match bound weights", DataRoleError)
        return references, weights

    def _loss_from_result(self, result: LightGen2DForwardResult, references: FrozenBogtReferences, weights: BogtLossWeights) -> BogtLossBreakdown:
        _require(result.image_input is not None and result.ols_input_field is not None, "real path did not expose required image/OLS fields")
        raw = result.raw_intensity
        target = result.image_input.slm_intensity
        _require(raw.shape == target.shape and tuple(raw.shape[-2:]) == (200, 200), "raw CMOS/SLM target shape mismatch")
        _validate_finite(raw, "raw CMOS intensity")
        _validate_finite(target, "SLM intensity target")
        l_rec = (raw - target).square().mean()
        u = statistical_variable(result.z_ols.coefficients, references)
        q = q_from_batch(u)
        p = fixed_prior(device=q.loc.device, dtype=q.loc.dtype)
        l_dist = truncated_normal_positive_kl(q, p).mean()
        powers = contracted_plane_powers(result)
        l_op = optical_power_loss_from_powers(powers, references.power_reference)
        l_total = weights.alpha * l_dist + weights.beta * l_rec + weights.gamma * l_op
        _validate_finite(l_dist, "L_dist")
        _validate_finite(l_rec, "L_rec")
        _validate_finite(l_op, "L_op")
        _validate_finite(l_total, "L_total")
        return BogtLossBreakdown(l_dist, l_rec, l_op, l_total, q_batch_moments(u), powers, result)

    def forward_loss(self, images: torch.Tensor, *, data_role: str) -> BogtLossBreakdown:
        references, weights = self._bound_for_role(data_role, "BOGT loss")
        _validate_image_batch(images, exact_batch=True)
        _require(images.device == self.device, "images must already be on consumer device")
        result = self.runtime.forward_from_image(images)
        return self._loss_from_result(result, references, weights)

    @staticmethod
    def _retain_autograd_probes(loss: BogtLossBreakdown) -> None:
        result = loss.result
        tensors = (
            result.ols_input_field.tensor if result.ols_input_field is not None else None,
            result.z_ols.coefficients,
            result.generator_input_field.tensor,
        )
        for tensor in tensors:
            _require(isinstance(tensor, torch.Tensor) and tensor.requires_grad, "required modal/W_OLS tensor is detached")
            tensor.retain_grad()

    def _validate_autograd(self, loss: BogtLossBreakdown) -> AutogradDiagnostics:
        gradient_norms: dict[str, float] = {}
        for name, parameter in self._named_runtime_parameters():
            if name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
                _require(parameter.grad is not None, f"trainable parameter has no gradient: {name}")
                _validate_finite(parameter.grad, f"gradient {name}")
                norm = float(parameter.grad.detach().to(torch.float64).square().sum().sqrt().cpu())
                _require(math.isfinite(norm) and norm > 0.0, f"gradient norm is invalid: {name}")
                gradient_norms[name] = norm
            elif parameter.grad is not None:
                raise TrainingContractError(f"forbidden parameter received a gradient: {name}")
        result = loss.result
        _require(result.ols_input_field is not None, "OLS input field is missing")
        z_grad = result.z_ols.coefficients.grad
        ols_grad = result.ols_input_field.tensor.grad
        writer_grad = result.generator_input_field.tensor.grad
        _require(z_grad is not None and ols_grad is not None and writer_grad is not None, "modal projection/W_OLS gradient probes are missing")
        _validate_finite(z_grad, "z gradient")
        _validate_finite(ols_grad, "OLS input gradient")
        _validate_finite(writer_grad, "W_OLS gradient")
        projection_active = bool(
            result.z_ols.coefficients.grad_fn is not None
            and result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True
            and result.ols_input_field.tensor.grad_fn is not None
        )
        writer_active = bool(
            result.generator_input_field.tensor.grad_fn is not None
            and result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission"
            and result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space"
        )
        _require(projection_active and writer_active, "modal projection/W_OLS structural proof failed")
        return AutogradDiagnostics(projection_active, writer_active, gradient_norms, True, True)

    def bogt_training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        global_step: int,
        data_role: str,
        ledger: DataReadLedger | None = None,
    ) -> BogtUpdateResult:
        _require(0 <= global_step < SCIENTIFIC_UPDATES, "global_step is outside the fixed 300-update budget")
        references, _ = self._bound_for_role(data_role, "BOGT training step")
        _validate_image_batch(batch, exact_batch=True)
        _require(batch.device == self.device, "training batch must already be on consumer device")
        if ledger is not None:
            ledger.record(data_role, int(batch.shape[0]))
        self._enforce_parameter_firewall()
        self.optimizer_manifest(optimizer)
        before_parameters = {name: parameter.detach().clone() for name, parameter in self._named_runtime_parameters()}
        before_frozen = self._frozen_state_snapshot()
        reference_identity = references.identity()
        weights_identity = self._weights.identity() if self._weights is not None else None
        optimizer.zero_grad(set_to_none=True)
        loss = self.forward_loss(batch, data_role=data_role)
        self._retain_autograd_probes(loss)
        loss.l_total.backward()
        autograd = self._validate_autograd(loss)
        optimizer.step()
        self._enforce_parameter_firewall()
        after_parameters = {name: parameter.detach().clone() for name, parameter in self._named_runtime_parameters()}
        for name, parameter in self._named_runtime_parameters():
            _validate_finite(parameter, f"updated parameter {name}")
        update_norms = {
            name: float((after_parameters[name] - before).to(torch.float64).square().sum().sqrt().cpu())
            for name, before in before_parameters.items()
        }
        _require(all(value > 0.0 for value in update_norms.values()), "both allowed phase groups must change on every update")
        frozen_unchanged = _state_equal(before_frozen, self._frozen_state_snapshot())
        _require(frozen_unchanged, "a frozen runtime state changed during optimizer step")
        _require(self._references is not None and self._references.identity() == reference_identity, "frozen references changed during update")
        _require(self._weights is not None and self._weights.identity() == weights_identity, "frozen weights changed during update")
        return BogtUpdateResult(global_step + 1, loss, autograd, update_norms, frozen_unchanged)

    def generated_forward(self, state: ComplexOLSState | torch.Tensor) -> LightGen2DForwardResult:
        """Use the runtime's existing direct OLS path; no second writer exists."""
        if isinstance(state, torch.Tensor):
            _require(self.runtime.ols.basis_id == LP01_OLS_MODEL, "Gaussian generated input requires an explicit ComplexOLSState basis and geometry")
            state = ComplexOLSState.from_external(state)
        _require(isinstance(state, ComplexOLSState), "generated state must be ComplexOLSState")
        return self.runtime.forward_from_ols(state)

    def loss_configuration(self) -> dict[str, Any]:
        return {
            "distribution": {
                "name": "POSITIVE_TRUNCATED_NORMAL_KL_V1",
                "direction": "KL(Q||P)",
                "reduction": "mean over 100 ports",
            },
            "reconstruction": {
                "name": "RAW_CMOS_TO_SLM_INTENSITY_MSE_V1",
                "reduction": "mean over batch and spatial axes",
            },
            "optical": {
                "name": "FIVE_PLANE_RELATIVE_LOG_POWER_DEVIATION_V1",
                "epsilon": EPS_OP,
                "reduction": "mean over batch and five planes",
            },
            "weights": None if self._weights is None else self._weights.identity(),
        }


class DeterministicTrainRoleSampler:
    """One-row-per-class sampler for the exact 300-update schedule."""

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int = SCHEDULE_SEED, steps: int = SCIENTIFIC_UPDATES) -> None:
        _require(train_indices.ndim == 1 and labels.ndim == 1 and train_indices.dtype == torch.long and labels.dtype == torch.long, "sampler requires long index/label vectors", DataRoleError)
        _require(train_indices.numel() == TRAIN_COUNT and labels.numel() == 60_000, "sampler requires the frozen 55,000/60,000 roles", DataRoleError)
        _require(seed == SCHEDULE_SEED and steps == SCIENTIFIC_UPDATES, "sampler seed/steps differ from contract", DataRoleError)
        self.seed = int(seed)
        self.steps = int(steps)
        self._cursor = 0
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._candidates: list[torch.Tensor] = []
        self._permutations: list[torch.Tensor] = []
        train_indices = train_indices.cpu()
        labels = labels.cpu()
        for digit in range(10):
            candidates = train_indices[labels.index_select(0, train_indices) == digit]
            _require(candidates.numel() >= steps, f"class {digit} cannot support the fixed schedule", DataRoleError)
            self._candidates.append(candidates.clone())
            self._permutations.append(candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator)))

    @property
    def cursor(self) -> int:
        return self._cursor

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self._cursor < self.steps, "train sampler is exhausted", DataRoleError)
        indices = torch.stack([permutation[self._cursor] for permutation in self._permutations])
        labels = torch.arange(10, dtype=torch.long)
        self._cursor += 1
        return indices, labels

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SAMPLER_SCHEMA,
            "seed": self.seed,
            "steps": self.steps,
            "cursor": self._cursor,
            "class_count": 10,
            "rows_per_class": [int(value.numel()) for value in self._permutations],
            "permutations": [value.clone() for value in self._permutations],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(state.get("schema") == SAMPLER_SCHEMA and state.get("seed") == self.seed and state.get("steps") == self.steps, "sampler checkpoint identity mismatch", CheckpointError)
        permutations = state.get("permutations")
        _require(isinstance(permutations, list) and len(permutations) == 10, "sampler checkpoint permutations are invalid", CheckpointError)
        expected_counts = [int(value.numel()) for value in self._permutations]
        _require(state.get("rows_per_class") == expected_counts, "sampler checkpoint class counts changed", CheckpointError)
        cursor = state.get("cursor")
        _require(isinstance(cursor, int) and 0 <= cursor <= self.steps, "sampler checkpoint cursor is invalid", CheckpointError)
        for index, value in enumerate(permutations):
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.long and value.shape == (expected_counts[index],), "sampler permutation shape changed", CheckpointError)
            _require(torch.equal(torch.sort(value).values, torch.sort(self._candidates[index]).values), "sampler permutation contents changed", CheckpointError)
        self._permutations = [value.cpu().clone() for value in permutations]
        self._cursor = cursor


def collect_bogt_train_references(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    images: torch.Tensor,
    *,
    data_role: str,
    engineering_only: bool,
    source_indices: torch.Tensor | None = None,
    data_identity: Mapping[str, Any] | None = None,
    ledger: DataReadLedger | None = None,
) -> FrozenBogtReferences:
    """Collect s_j and r_p in one frozen pass; smoke results stay marked."""
    guard_data_role(data_role, operation="BOGT reference collection", engineering_only=engineering_only)
    _validate_image_batch(images, exact_batch=False)
    _require(images.shape[0] > 0, "reference collection requires at least one row")
    if data_role == TRAIN_ROLE:
        _require(images.shape[0] == TRAIN_COUNT and not engineering_only, "scientific reference collection requires exactly 55,000 TRAIN rows", DataRoleError)
        _require(source_indices is not None, "scientific TRAIN reference collection requires canonical source indices", DataRoleError)
    else:
        _require(engineering_only, "non-scientific reference collection must be engineering-only", DataRoleError)
    if source_indices is not None:
        _require(source_indices.ndim == 1 and source_indices.dtype == torch.long and source_indices.numel() == images.shape[0], "source indices are invalid", DataRoleError)
        _require(bool((source_indices[1:] > source_indices[:-1]).all().item()) if source_indices.numel() > 1 else True, "source indices must be canonical increasing order", DataRoleError)
    consumer.assert_theta0()
    scale_sum = torch.zeros(OLS_PORT_COUNT, dtype=torch.float64)
    power_sum = torch.zeros(5, dtype=torch.float64)
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    try:
        with torch.no_grad():
            for batch in images.split(LOGICAL_BATCH_SIZE, dim=0):
                result = consumer.runtime.forward_from_image(batch.to(consumer.device))
                z = result.z_ols.coefficients
                scale_sum += (z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()).sum(dim=0).cpu()
                power_sum += contracted_plane_powers(result).to(torch.float64).sum(dim=0).cpu()
    finally:
        consumer.runtime.train(was_training)
    if ledger is not None:
        ledger.record(data_role, int(images.shape[0]))
    count = int(images.shape[0])
    scale = scale_sum / float(count)
    power_reference = power_sum / float(count)
    if data_role == TRAIN_ROLE:
        _require(bool((scale > EPS_S).all().item()), "scientific s_j must be strictly greater than eps_s", DataRoleError)
        _require(bool((power_reference > 0.0).all().item()), "scientific r_p must be strictly positive", DataRoleError)
    identity = dict(data_identity or {})
    identity.update({"role": data_role, "count": count, "canonical_order": "source_indices_in_input_order", "engineering_only": engineering_only})
    return FrozenBogtReferences(
        scale,
        power_reference,
        data_role,
        count,
        engineering_only,
        consumer.initialization_fingerprint(),
        identity,
        consumer.runtime.ols.basis_id,
        consumer.runtime.ols.mode_identity(),
    )


def calibrate_initial_gradient_weights(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    batch: torch.Tensor,
    references: FrozenBogtReferences,
    *,
    data_role: str,
    calibration_batch_identity: Mapping[str, Any],
    engineering_only: bool,
    ledger: DataReadLedger | None = None,
) -> BogtLossWeights:
    """Compute one-shot inverse initial-gradient balancing at canonical theta0."""
    guard_data_role(data_role, operation="BOGT weight calibration", engineering_only=engineering_only, ledger=ledger, count=int(batch.shape[0]))
    _validate_image_batch(batch, exact_batch=True)
    _require(batch.device == consumer.device, "calibration batch must already be on consumer device")
    _require(data_role == references.data_role, "calibration role does not match references", DataRoleError)
    _require(engineering_only == references.engineering_only, "calibration engineering flag does not match references", DataRoleError)
    _require(isinstance(calibration_batch_identity, Mapping), "calibration batch identity is required", DataRoleError)
    _require(calibration_batch_identity.get("seed") == SCHEDULE_SEED, "calibration batch seed differs from contract", DataRoleError)
    _require(calibration_batch_identity.get("batch_size") == LOGICAL_BATCH_SIZE, "calibration batch size differs from contract", DataRoleError)
    _require(calibration_batch_identity.get("class_balanced") is True, "calibration batch is not declared class-balanced", DataRoleError)
    consumer.assert_theta0()
    consumer.bind_references(references)
    consumer._enforce_parameter_firewall()
    parameters = [parameter for _, parameter in consumer._named_runtime_parameters()]
    for parameter in parameters:
        parameter.grad = None
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "calibration real path is incomplete")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    l_rec = (raw - target).square().mean()
    u = statistical_variable(result.z_ols.coefficients, references)
    q = q_from_batch(u)
    p = fixed_prior(device=q.loc.device, dtype=q.loc.dtype)
    l_dist = truncated_normal_positive_kl(q, p).mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    component_values = {"dist": l_dist, "rec": l_rec, "op": l_op}
    norms: dict[str, float] = {}
    for index, (name, value) in enumerate(component_values.items()):
        gradients = torch.autograd.grad(value, parameters, retain_graph=index < len(component_values) - 1, allow_unused=True)
        squared = torch.zeros((), device=consumer.device, dtype=torch.float64)
        for gradient in gradients:
            if gradient is None:
                continue
            _validate_finite(gradient, f"initial gradient {name}")
            squared = squared + gradient.detach().to(torch.float64).square().sum()
        norm = float(squared.sqrt().cpu())
        _require(math.isfinite(norm) and norm > 0.0, f"initial gradient norm is invalid: {name}")
        norms[name] = norm
    inverse = {name: 1.0 / value for name, value in norms.items()}
    normalizer = sum(inverse.values())
    weights = {name: 3.0 * inverse[name] / normalizer for name in inverse}
    return BogtLossWeights(
        weights["dist"],
        weights["rec"],
        weights["op"],
        norms,
        data_role,
        engineering_only,
        dict(calibration_batch_identity),
        canonical_sha256(references.identity()),
    )


def _rng_state_payload() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "cpu": torch.random.get_rng_state().cpu().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()] if cuda_available else [],
        "cuda_available": cuda_available,
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    cpu = payload.get("cpu")
    cuda = payload.get("cuda")
    _require(isinstance(cpu, torch.Tensor) and isinstance(cuda, list), "checkpoint RNG state is malformed", CheckpointError)
    torch.random.set_rng_state(cpu.cpu())
    if cuda:
        _require(torch.cuda.is_available(), "checkpoint contains CUDA RNG state but CUDA is unavailable", CheckpointError)
        _require(all(isinstance(state, torch.Tensor) for state in cuda), "checkpoint CUDA RNG state is malformed", CheckpointError)
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])


def checkpoint_payload(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
    *,
    completed_updates: int,
    sampler_state: Mapping[str, Any],
    training_history: Sequence[Mapping[str, Any]],
    engineering_only: bool,
    git: Mapping[str, Any] | None = None,
    recovery: bool = False,
    run_identity: ResearchRunIdentity | Mapping[str, Any] | None = None,
    source_manifest: Mapping[str, Any] | None = None,
    authorization_id: str | None = None,
) -> dict[str, Any]:
    _require(0 <= completed_updates <= SCIENTIFIC_UPDATES, "checkpoint update count is outside contract", CheckpointError)
    _require(isinstance(engineering_only, bool), "checkpoint engineering scope is invalid", CheckpointError)
    _require(isinstance(recovery, bool), "checkpoint operational scope is invalid", CheckpointError)
    _require(not (engineering_only and recovery), "operational recovery cannot be marked engineering-only", CheckpointError)
    if authorization_id is not None:
        _require(bool(authorization_id), "checkpoint authorization identity is empty", CheckpointError)
    if engineering_only:
        _require(completed_updates <= 3, "engineering checkpoint may contain at most three updates", CheckpointError)
    elif not recovery:
        _require(completed_updates in CHECKPOINT_STEPS, "scientific checkpoint is not on fixed schedule", CheckpointError)
    _require(consumer.references is not None and consumer.weights is not None, "checkpoint requires frozen references and weights", CheckpointError)
    if run_identity is None:
        identity_payload: dict[str, Any] = {}
    elif isinstance(run_identity, ResearchRunIdentity):
        identity_payload = run_identity.to_dict()
    else:
        identity_payload = ResearchRunIdentity.from_mapping(run_identity).to_dict()
    if source_manifest is None:
        source_manifest_payload: dict[str, Any] = {}
    else:
        _require(isinstance(source_manifest, Mapping), "checkpoint source manifest is invalid", CheckpointError)
        source_manifest_payload = dict(source_manifest)
    consumer.optimizer_manifest(optimizer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "engineering_only": engineering_only,
        "scientific_artifact": not engineering_only and not recovery,
        "operational_artifact": recovery,
        "artifact_kind": "OPERATIONAL_RECOVERY_CHECKPOINT" if recovery else ("ENGINEERING_CHECKPOINT" if engineering_only else "SCIENTIFIC_CHECKPOINT"),
        "completed_updates": completed_updates,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "selection_rule": "fixed final update step 300",
        "contracts": frozen_contract_identity(),
        "initialization": consumer.initialization_fingerprint(),
        "runtime": consumer.runtime_identity(),
        "frozen_runtime_state_sha256": consumer.frozen_state_sha256,
        "trainable_parameter_state": consumer.trainable_state_dict(),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "optimizer_configuration": consumer.optimizer_manifest(optimizer),
        "sampler_state": copy.deepcopy(dict(sampler_state)),
        "rng_state": _rng_state_payload(),
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": [dict(record) for record in training_history],
        "git": dict(git or git_identity()),
        "run_identity": identity_payload,
        "source_manifest": source_manifest_payload,
        "source_manifest_id": identity_payload.get("source_manifest_id") or source_manifest_payload.get("source_manifest_id"),
        "authorization_id": authorization_id,
    }


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    _require(not target.exists(), f"refusing to overwrite checkpoint: {target}", CheckpointError)
    atomic_torch_save_new(target, dict(payload), artifact_kind="Chen BOGT training checkpoint")


@dataclass(frozen=True)
class LoadedCheckpoint:
    completed_updates: int
    sampler_state: Mapping[str, Any]
    training_history: Sequence[Mapping[str, Any]]
    engineering_only: bool
    recovery: bool = False
    run_identity: ResearchRunIdentity | None = None
    source_manifest: Mapping[str, Any] = field(default_factory=dict)


def _load_checkpoint_payload(
    payload: Mapping[str, Any],
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
    *,
    expected_engineering_only: bool | None,
    expected_run_identity: ResearchRunIdentity | None,
    expected_authorization_id: str | None,
) -> LoadedCheckpoint:
    _require(payload.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch", CheckpointError)
    engineering_value = payload.get("engineering_only")
    _require(isinstance(engineering_value, bool), "checkpoint engineering scope is invalid", CheckpointError)
    engineering_only = engineering_value
    operational_value = payload.get("operational_artifact", False)
    _require(isinstance(operational_value, bool), "checkpoint operational scope is invalid", CheckpointError)
    recovery = operational_value or payload.get("artifact_kind") == "OPERATIONAL_RECOVERY_CHECKPOINT"
    _require(payload.get("scientific_artifact") is (not engineering_only and not recovery), "checkpoint scientific/engineering scope is inconsistent", CheckpointError)
    if "operational_artifact" in payload:
        _require(payload.get("operational_artifact") is recovery, "checkpoint operational scope is inconsistent", CheckpointError)
    if "artifact_kind" in payload:
        expected_kind = "OPERATIONAL_RECOVERY_CHECKPOINT" if recovery else ("ENGINEERING_CHECKPOINT" if engineering_only else "SCIENTIFIC_CHECKPOINT")
        _require(payload.get("artifact_kind") == expected_kind, "checkpoint artifact kind is inconsistent", CheckpointError)
    _require(payload.get("checkpoint_schedule") == list(CHECKPOINT_STEPS), "checkpoint schedule changed", CheckpointError)
    _require(payload.get("selection_rule") == "fixed final update step 300", "checkpoint selection rule changed", CheckpointError)
    if expected_engineering_only is not None:
        _require(engineering_only is expected_engineering_only, "checkpoint engineering boundary changed", CheckpointError)
    if expected_authorization_id is not None:
        _require(payload.get("authorization_id") == expected_authorization_id, "checkpoint authorization identity changed", CheckpointError)
    stored_identity: ResearchRunIdentity | None = None
    identity_payload = payload.get("run_identity")
    if identity_payload:
        _require(isinstance(identity_payload, Mapping), "checkpoint run identity is invalid", CheckpointError)
        stored_identity = ResearchRunIdentity.from_mapping(identity_payload)
    if expected_run_identity is not None:
        _require(stored_identity is not None, "checkpoint is missing required run identity", CheckpointError)
        check_resume_compatibility(expected_run_identity, stored_identity)
    source_manifest = payload.get("source_manifest", {})
    _require(isinstance(source_manifest, Mapping), "checkpoint source manifest is invalid", CheckpointError)
    contracts = _mapping(payload.get("contracts"), "checkpoint contracts", CheckpointError)
    current_contracts = frozen_contract_identity()
    _require(
        _resume_identity(contracts) == _resume_identity(current_contracts),
        "checkpoint model/training contract changed",
        CheckpointError,
    )
    runtime = _mapping(payload.get("runtime"), "checkpoint runtime", CheckpointError)
    current_runtime = consumer.runtime_identity()
    for key in ("identity", "config", "ols_interface", "transport", "writer"):
        _require(runtime.get(key) == current_runtime.get(key), f"checkpoint runtime identity changed: {key}", CheckpointError)
    initialization = _mapping(payload.get("initialization"), "checkpoint initialization", CheckpointError)
    _require(
        _resume_identity(initialization) == _resume_identity(consumer._theta0_identity),
        "checkpoint theta0 model configuration changed",
        CheckpointError,
    )
    completed_updates = payload.get("completed_updates")
    _require(isinstance(completed_updates, int) and 0 <= completed_updates <= SCIENTIFIC_UPDATES, "checkpoint update count is invalid", CheckpointError)
    if engineering_only:
        _require(completed_updates <= 3, "engineering checkpoint update count is invalid", CheckpointError)
    elif not recovery:
        _require(completed_updates in CHECKPOINT_STEPS, "scientific checkpoint is off schedule", CheckpointError)
    references = FrozenBogtReferences.from_payload(_mapping(payload.get("references"), "checkpoint references", CheckpointError))
    consumer.bind_references(references)
    weights = BogtLossWeights.from_payload(_mapping(payload.get("weights"), "checkpoint weights", CheckpointError))
    consumer.bind_weights(weights)
    consumer.load_trainable_state_dict(_mapping(payload.get("trainable_parameter_state"), "trainable state", CheckpointError))
    optimizer_state = payload.get("optimizer_state")
    _require(isinstance(optimizer_state, Mapping), "checkpoint optimizer state is invalid", CheckpointError)
    optimizer.load_state_dict(dict(optimizer_state))
    consumer.optimizer_manifest(optimizer)
    restore_rng_state(_mapping(payload.get("rng_state"), "checkpoint RNG", CheckpointError))
    sampler_state = _mapping(payload.get("sampler_state"), "checkpoint sampler", CheckpointError)
    history = payload.get("training_history")
    _require(isinstance(history, list), "checkpoint history is invalid", CheckpointError)
    return LoadedCheckpoint(completed_updates, sampler_state, history, engineering_only, recovery, stored_identity, dict(source_manifest))


def load_checkpoint(
    path: str | Path,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
    *,
    expected_engineering_only: bool | None = None,
    expected_run_identity: ResearchRunIdentity | None = None,
    expected_authorization_id: str | None = None,
) -> LoadedCheckpoint:
    payload = safe_torch_load(path, map_location="cpu")
    _require(isinstance(payload, Mapping), "checkpoint schema mismatch", CheckpointError)
    if payload.get("schema") == RECOVERY_CHECKPOINT_SCHEMA:
        wrapper_identity = payload.get("run_identity")
        _require(isinstance(wrapper_identity, Mapping), "recovery run identity is invalid", CheckpointError)
        if expected_authorization_id is not None:
            _require(payload.get("authorization_id") == expected_authorization_id, "recovery authorization identity changed", CheckpointError)
        if expected_run_identity is not None:
            check_resume_compatibility(expected_run_identity, wrapper_identity)
        inner = payload.get("payload")
        _require(isinstance(inner, Mapping), "recovery payload is invalid", CheckpointError)
        return _load_checkpoint_payload(
            inner,
            consumer,
            optimizer,
            expected_engineering_only=expected_engineering_only,
            expected_run_identity=expected_run_identity,
            expected_authorization_id=expected_authorization_id,
        )
    return _load_checkpoint_payload(
        payload,
        consumer,
        optimizer,
        expected_engineering_only=expected_engineering_only,
        expected_run_identity=expected_run_identity,
        expected_authorization_id=expected_authorization_id,
    )


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    _validate_finite(phase, "phase")
    return torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi


def max_q_gauge(z: torch.Tensor, references: FrozenBogtReferences | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the deterministic max-q global phase gauge with lowest-index ties."""
    _validate_complex_ols(z, "z")
    q = statistical_variable(z, references)
    maximum, indices = q.max(dim=1)
    _require(bool((maximum > 0.0).all().item()), "phase gauge requires a positive max-q anchor")
    anchor = z.gather(1, indices.unsqueeze(1)).squeeze(1)
    _require(bool((anchor.abs() > 0.0).all().item()), "phase gauge anchor is zero")
    unit = anchor / anchor.abs()
    gauged = z * unit.conj().unsqueeze(1)
    _validate_finite(gauged, "gauged z")
    return gauged, q, indices


@dataclass(frozen=True)
class WrappedConditionalPhaseModel:
    coefficients: torch.Tensor
    residual_mean: torch.Tensor
    covariance: torch.Tensor
    cholesky: torch.Tensor
    scale_sha256: str
    fit_data_role: str
    fit_count: int
    engineering_only: bool
    data_identity: Mapping[str, Any]
    log_epsilon: float = PHASE_LOG_EPS
    covariance_epsilon: float = PHASE_COVARIANCE_EPS

    def __post_init__(self) -> None:
        coefficients = self.coefficients.detach().to(dtype=torch.complex128, device="cpu").clone()
        residual_mean = self.residual_mean.detach().to(dtype=torch.float64, device="cpu").clone()
        covariance = self.covariance.detach().to(dtype=torch.float64, device="cpu").clone()
        cholesky = self.cholesky.detach().to(dtype=torch.float64, device="cpu").clone()
        _require(coefficients.shape == (OLS_PORT_COUNT + 1, OLS_PORT_COUNT), "phase coefficients must have shape [101,100]")
        _require(residual_mean.shape == (OLS_PORT_COUNT,), "phase residual mean must have shape [100]")
        _require(covariance.shape == (OLS_PORT_COUNT, OLS_PORT_COUNT), "phase covariance must have shape [100,100]")
        _require(cholesky.shape == covariance.shape, "phase Cholesky shape differs")
        for value, name in ((coefficients, "phase coefficients"), (residual_mean, "phase residual mean"), (covariance, "phase covariance"), (cholesky, "phase Cholesky")):
            _validate_finite(value, name)
        _require(torch.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12), "phase covariance is not symmetric")
        _require(self.fit_count > 0, "phase fit count must be positive")
        if self.fit_data_role != TRAIN_ROLE:
            _require(self.engineering_only, "non-TRAIN phase fit must be engineering-only", DataRoleError)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "residual_mean", residual_mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "cholesky", cholesky)

    def _design(self, u: torch.Tensor) -> torch.Tensor:
        _require(u.ndim == 2 and u.shape[1] == OLS_PORT_COUNT, "phase model u must have shape [N,100]")
        _validate_finite(u, "phase model u")
        _require(bool((u >= 0.0).all().item()), "phase model u must be nonnegative")
        h = torch.log(u.to(torch.float64) + self.log_epsilon)
        return torch.cat((torch.ones((u.shape[0], 1), device=u.device, dtype=torch.float64), h), dim=1)

    def conditional_mean_phase(self, u: torch.Tensor) -> torch.Tensor:
        design = self._design(u)
        coefficients = self.coefficients.to(device=u.device)
        predicted = design.to(torch.complex128) @ coefficients
        _require(bool((predicted.abs() > PHASE_MEAN_EPS).all().item()), "conditional phase mean phasor is degenerate")
        residual_mean = self.residual_mean.to(device=u.device)
        return wrap_phase(torch.angle(predicted) + residual_mean)

    def sample(self, u: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        mean = self.conditional_mean_phase(u)
        if generator is None:
            noise = torch.randn(u.shape, device=u.device, dtype=torch.float64)
        else:
            noise = torch.randn(u.shape, device="cpu", dtype=torch.float64, generator=generator).to(u.device)
        chol = self.cholesky.to(device=u.device)
        residual_mean = self.residual_mean.to(device=u.device)
        residual = noise @ chol.T + residual_mean
        return wrap_phase(mean + residual)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": PHASE_MODEL_SCHEMA,
            "coefficients": self.coefficients.clone(),
            "residual_mean": self.residual_mean.clone(),
            "covariance": self.covariance.clone(),
            "cholesky": self.cholesky.clone(),
            "scale_sha256": self.scale_sha256,
            "fit_data_role": self.fit_data_role,
            "fit_count": self.fit_count,
            "engineering_only": self.engineering_only,
            "data_identity": dict(self.data_identity),
            "log_epsilon": self.log_epsilon,
            "covariance_epsilon": self.covariance_epsilon,
        }


def fit_wrapped_conditional_phase_model(
    z: torch.Tensor,
    references: FrozenBogtReferences,
    *,
    data_role: str,
    engineering_only: bool,
    data_identity: Mapping[str, Any] | None = None,
) -> WrappedConditionalPhaseModel:
    """Fit the post-training q-conditioned wrapped multivariate phase model."""
    guard_data_role(data_role, operation="phase model fit", engineering_only=engineering_only)
    _validate_complex_ols(z, "phase-fit z")
    if data_role == TRAIN_ROLE:
        _require(z.shape[0] == TRAIN_COUNT and not engineering_only, "scientific phase fit requires exactly 55,000 TRAIN rows", DataRoleError)
    with torch.no_grad():
        gauged, q, _ = max_q_gauge(z, references)
        phi = torch.angle(gauged).to(torch.float64)
        h = torch.log(q.to(torch.float64) + PHASE_LOG_EPS)
        design = torch.cat((torch.ones((z.shape[0], 1), device=z.device, dtype=torch.float64), h), dim=1)
        _require(int(torch.linalg.matrix_rank(design).item()) >= 2, "phase design matrix is rank deficient")
        unit_phasor = torch.complex(torch.cos(phi), torch.sin(phi)).to(torch.complex128)
        coefficients = torch.linalg.pinv(design.to(torch.complex128)) @ unit_phasor
        predicted = design.to(torch.complex128) @ coefficients
        _require(bool((predicted.abs() > PHASE_MEAN_EPS).all().item()), "phase conditional mean phasor is degenerate")
        conditional_mean = torch.angle(predicted)
        residual = wrap_phase(phi - conditional_mean)
        residual_mean = torch.atan2(residual.sin().mean(dim=0), residual.cos().mean(dim=0))
        centered = wrap_phase(residual - residual_mean)
        covariance = centered.T @ centered / float(z.shape[0])
        covariance = 0.5 * (covariance + covariance.T) + PHASE_COVARIANCE_EPS * torch.eye(OLS_PORT_COUNT, device=z.device, dtype=torch.float64)
        try:
            cholesky = torch.linalg.cholesky(covariance)
        except RuntimeError as error:
            raise TrainingContractError("phase covariance is not positive definite") from error
    return WrappedConditionalPhaseModel(
        coefficients,
        residual_mean,
        covariance,
        cholesky,
        references.scale_sha256,
        data_role,
        int(z.shape[0]),
        engineering_only,
        dict(data_identity or {"role": data_role, "count": int(z.shape[0]), "engineering_only": engineering_only}),
    )


def construct_sampled_ols(
    u_sample: torch.Tensor,
    references: FrozenBogtReferences,
    phase_model: WrappedConditionalPhaseModel,
    *,
    generator: torch.Generator | None = None,
) -> ComplexOLSState:
    """Construct z from u and the fitted phase support without changing magnitude."""
    _require(u_sample.ndim == 2 and u_sample.shape[1] == OLS_PORT_COUNT, "u_sample must have shape [N,100]")
    _validate_finite(u_sample, "u_sample")
    _require(bool((u_sample >= 0.0).all().item()), "u_sample must be nonnegative")
    phase = phase_model.sample(u_sample, generator=generator)
    scale = references.scale.to(device=u_sample.device, dtype=torch.float64)
    amplitude = ((scale + EPS_S) * u_sample.to(torch.float64)).clamp_min(0.0).sqrt()
    z = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase)).to(torch.complex64)
    gauged, _, _ = max_q_gauge(z, references)
    _validate_finite(gauged, "sampled z")
    return ComplexOLSState.from_external(
        gauged,
        basis_id=references.basis_id,
        mode_geometry=references.mode_geometry,
    )


__all__ = [
    "ADAM_BETAS",
    "ADAM_EPSILON",
    "ALLOWED_TRAINABLE_PARAMETER_NAMES",
    "AutogradDiagnostics",
    "BogtLossBreakdown",
    "BogtLossWeights",
    "BogtUpdateResult",
    "CHECKPOINT_STEPS",
    "CHECKPOINT_SCHEMA",
    "ChenBogtComplexOLSTrainingConsumer",
    "LightGen2DChenBOGTComplexOLSTrainingConsumer",
    "DataReadLedger",
    "DataRoleError",
    "DeterministicTrainRoleSampler",
    "DEV_ROLE",
    "ENGINEERING_PROPAGATION_GRID",
    "EPS_OP",
    "EPS_S",
    "FrozenBogtReferences",
    "MODEL_CONTRACT_ID",
    "MODEL_CONTRACT_SHA256",
    "MODEL_CONTRACT_GATE",
    "OUTER_GATE",
    "PHASE_COVARIANCE_EPS",
    "PHASE_LOG_EPS",
    "PositiveTruncatedNormal",
    "PRODUCTION_PROPAGATION_GRID",
    "QBatchMoments",
    "SCIENTIFIC_UPDATES",
    "SCHEDULE_SEED",
    "SYNTHETIC_ROLE",
    "TRAINING_CONSUMER_CONTRACT_ID",
    "TRAINING_CONSUMER_CONTRACT_SHA256",
    "TRAINING_CONSUMER_CONTRACT_GATE",
    "TRAIN_COUNT",
    "TRAIN_ROLE",
    "TRAIN_SMOKE_ROLE",
    "TrainingContractError",
    "WrappedConditionalPhaseModel",
    "calibrate_initial_gradient_weights",
    "checkpoint_payload",
    "collect_bogt_train_references",
    "construct_sampled_ols",
    "contracted_plane_powers",
    "fixed_prior",
    "fit_wrapped_conditional_phase_model",
    "frozen_contract_identity",
    "git_identity",
    "guard_data_role",
    "load_checkpoint",
    "max_q_gauge",
    "optical_power_loss",
    "optical_power_loss_from_powers",
    "physical_plane_power",
    "q_batch_moments",
    "q_from_batch",
    "save_checkpoint",
    "statistical_variable",
    "tn_positive_kl",
    "truncated_normal_positive_kl",
    "truncated_normal_positive_log_prob",
    "wrap_phase",
]


# Public spelling kept next to the longer runtime-specific name for scripts
# and future callers; both names resolve to this one implementation.
ChenBogtComplexOLSTrainingConsumer = LightGen2DChenBOGTComplexOLSTrainingConsumer
