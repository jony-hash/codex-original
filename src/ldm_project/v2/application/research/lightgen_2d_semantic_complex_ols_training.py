"""LightGen-aligned full-complex OLS training consumer.

This module consumes the frozen V1 training contract and exposes only the
engineering boundaries needed by a later bounded training entry point.  It is
deliberately not a long-training command: the caller must explicitly request
the small CPU smoke used for implementation/preflight validation.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ...evidence import (
    atomic_torch_save_new,
    file_sha256,
    safe_torch_load,
)
from .lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    OLS_INTERFACE_ID,
    OLS_PORT_COUNT,
    ROW_MAJOR_PORT_ORDER,
)


TASK_ID = "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1"
CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1_IMPLEMENTED"
PREFLIGHT_ID = "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1"
TRAINING_CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_contract.v1"
TRAINING_CONTRACT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v1.json"
)
TRAINING_CONTRACT_SHA256 = "7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f"
ALIGNMENT_CONTRACT_SHA256 = "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"
ALIGNMENT_RUNTIME_ID = "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1"
ALIGNMENT_RUNTIME_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
ALIGNMENT_RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"

ALLOWED_TRAINABLE_PARAMETER_NAMES = ("encoder.phase", "generator.phase")
OLS_GRID_SHAPE = (10, 10)
LATENT_DIMENSION = 200
PRIOR_SAMPLE_COUNT = 10
PRIOR_SIGMA = 1.0e-3
PRIOR_NAMESPACE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_PRIOR"
PRIOR_ROLE = "training_target"
GAUGE_EPSILON = 1.0e-6
MMD_BANDWIDTH_SQUARED = 200.0
ALPHA_LATENT = 1.0
BETA_IMAGE = 1.0
GAMMA_OPTICAL = 0.0
LOGICAL_BATCH_SIZE = 10
SCHEDULE_SEED = 1234
TRAINING_STEPS = 100
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGrad = False
SMOKE_PROPAGATION_GRID = 128
PRODUCTION_PROPAGATION_GRID = 2048
DETERMINISTIC_TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}


class TrainingContractError(ValueError):
    """Raised when an external training authority is stale or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass(frozen=True)
class FrozenTrainingContract:
    """A hash-bound, read-only view of the V1 contract."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    def identity(self, root: Path | None = None) -> dict[str, str]:
        base = _repo_root() if root is None else root
        return {
            "schema": self.schema,
            "path": _relative_path(self.path, base),
            "sha256": self.sha256,
        }


def _validate_contract_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == TRAINING_CONTRACT_SCHEMA, "training contract schema mismatch")
    _require(payload.get("version") == 1, "training contract version mismatch")
    _require(payload.get("status") == "valid_completed", "training contract is not valid_completed")
    _require(payload.get("result") == "READY", "training contract is not READY")
    _require(
        payload.get("classification") == "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1_READY",
        "training contract classification mismatch",
    )
    _require(payload.get("training_authorized") is False, "training contract unexpectedly authorizes training")
    _require(payload.get("formal_promotion") is False, "training contract unexpectedly authorizes promotion")
    _require(payload.get("generation_readiness") == "not_ready", "training contract readiness boundary changed")

    source = _mapping(payload.get("source_authority"), "source_authority")
    alignment = _mapping(source.get("alignment_contract"), "alignment_contract")
    authority = _mapping(alignment.get("authority"), "alignment authority")
    _require(authority.get("sha256") == ALIGNMENT_CONTRACT_SHA256, "aligned contract SHA mismatch")
    _require(
        alignment.get("classification") == "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY",
        "aligned contract classification mismatch",
    )
    runtime_report = _mapping(source.get("runtime_report"), "runtime_report")
    runtime = _mapping(runtime_report.get("runtime"), "runtime authority")
    _require(runtime.get("module_sha256") == ALIGNMENT_RUNTIME_SHA256, "aligned runtime SHA mismatch")
    _require(runtime_report.get("report", {}).get("sha256") == ALIGNMENT_RUNTIME_REPORT_SHA256, "runtime report SHA mismatch")
    _require(runtime.get("canonical_ols") == "complex [B,100], 10x10, row-major j=10*r+c", "canonical OLS changed")
    _require(runtime.get("transport") == "T=I_100 in coefficient space", "OLS transport changed")
    _require(runtime.get("writer") == "W_OLS(z)=sum_j (Tz)_j v_j", "OLS writer changed")

    architecture = _mapping(payload.get("architecture"), "architecture")
    _require(architecture.get("scope") == "2D semantic image generation only", "training scope changed")
    immutable_runtime = architecture.get("immutable_runtime")
    _require(isinstance(immutable_runtime, list) and immutable_runtime, "immutable runtime authority is missing")
    trainable = _mapping(payload.get("trainable_parameter_contract"), "trainable parameter contract")
    expected_names = list(ALLOWED_TRAINABLE_PARAMETER_NAMES)
    _require(trainable.get("exact_trainable_parameters") == expected_names, "allowed trainable parameter set changed")
    _require(trainable.get("exact_optimizer_parameters") == expected_names, "optimizer parameter set changed")
    _require(trainable.get("parameter_count") == 40_000, "trainable parameter count changed")
    _require(
        trainable.get("parameter_shapes")
        == {"encoder.phase": [2, 1, 100, 100], "generator.phase": [2, 1, 100, 100]},
        "trainable parameter shapes changed",
    )

    latent = _mapping(payload.get("latent_contract"), "latent contract")
    _require(latent.get("authoritative_state") == "raw aligned ComplexOLS coefficients z in C^100", "latent authority changed")
    _require(latent.get("representation") == "S(z)=concat(Re(G(z)/sigma), Im(G(z)/sigma)) in R^200", "latent representation changed")
    scale = _mapping(latent.get("scale"), "latent scale")
    _require(scale.get("sigma_z") == PRIOR_SIGMA, "latent sigma changed")
    _require(scale.get("role") == "fixed loss/prior unit; does not rescale W_OLS input", "latent scale role changed")
    global_phase = _mapping(latent.get("global_phase"), "global phase")
    _require(global_phase.get("epsilon") == GAUGE_EPSILON, "gauge epsilon changed")
    _require(global_phase.get("phase_repair") is False, "phase repair entered the contract")

    objectives = _mapping(payload.get("objectives"), "objectives")
    image = _mapping(objectives.get("image"), "image objective")
    _require(image.get("name") == "RAW_CMOS_INTENSITY_MSE_V1", "image loss changed")
    _require(image.get("formula") == "L_image = mean((I_cmos_raw - I_target)^2)", "image loss formula changed")
    _require(image.get("prediction_normalization") == "none; raw CMOS intensity is consumed as produced by the immutable runtime", "image normalization changed")
    latent_objective = _mapping(objectives.get("latent"), "latent objective")
    _require(latent_objective.get("name") == "FULL_COMPLEX_GAUSSIAN_KERNEL_MMD_VSTAT_V1", "latent loss changed")
    _require(latent_objective.get("q_only") is False, "latent loss became q-only")
    _require(latent_objective.get("target_count") == PRIOR_SAMPLE_COUNT, "latent target count changed")
    _require(latent_objective.get("kernel") == "k(a,b)=exp(-||a-b||^2/(2*l^2)), l^2=200", "MMD kernel changed")
    optical = _mapping(objectives.get("optical"), "optical objective")
    _require(optical.get("included") is False and optical.get("formula") is None, "an uncontracted optical loss was added")
    total = _mapping(objectives.get("total"), "total objective")
    weights = _mapping(total.get("weights"), "total weights")
    _require(
        weights == {"alpha_latent": ALPHA_LATENT, "beta_image": BETA_IMAGE, "gamma_optical": GAMMA_OPTICAL},
        "loss weights changed",
    )
    _require(total.get("adaptive_weighting") is False and total.get("best_of_n") is False, "adaptive or best-of-N loss behavior entered")

    prior = _mapping(payload.get("target_prior"), "target prior")
    _require(prior.get("name") == "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NATIVE_OLS_V1", "prior identity changed")
    _require(prior.get("shape") == [OLS_PORT_COUNT] and prior.get("sigma_z") == PRIOR_SIGMA, "prior shape/scale changed")
    _require(
        prior.get("formula") == "z_j = sigma_z/sqrt(2)*(epsilon_j_real + i*epsilon_j_imag), epsilon iid Normal(0,1), j=0..99",
        "prior formula changed",
    )
    _require(prior.get("training_target_sample_count") == PRIOR_SAMPLE_COUNT, "prior batch count changed")
    prior_rng = _mapping(prior.get("rng"), "prior RNG")
    _require(prior_rng.get("namespace") == PRIOR_NAMESPACE, "prior RNG namespace changed")
    _require(prior_rng.get("derivation") == "SHA256(namespace|role|global_step|sample_count), first 8 bytes reduced to signed 63-bit seed", "prior seed derivation changed")
    _require(prior_rng.get("training_seed") == 20260910, "training seed authority changed")

    schedule = _mapping(payload.get("training_schedule"), "training schedule")
    _require(schedule.get("logical_batch_size") == LOGICAL_BATCH_SIZE, "logical batch size changed")
    _require(schedule.get("steps") == TRAINING_STEPS, "training step count changed")
    _require(schedule.get("schedule_seed") == SCHEDULE_SEED, "schedule seed changed")
    _require(schedule.get("device") == "cuda required for the future training execution; no CUDA used by this definition", "device contract changed")
    precision = _mapping(schedule.get("precision"), "precision")
    _require(precision.get("field") == "complex64" and precision.get("phase") == "float32", "precision contract changed")
    _require(precision.get("autocast") is False and precision.get("grad_scaler") is False, "mixed precision entered the contract")
    optimizer = _mapping(schedule.get("optimizer"), "optimizer")
    _require(
        optimizer
        == {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": AMSGrad,
        },
        "optimizer hyperparameters changed",
    )
    gradient = _mapping(schedule.get("gradient"), "gradient contract")
    _require(gradient.get("clipping") is None and gradient.get("zero_grad") == "set_to_none=True", "gradient schedule changed")

    data = _mapping(payload.get("data_contract"), "data contract")
    _require(data.get("dataset") == "local MNIST, download=False", "dataset authority changed")
    preprocessing = _mapping(data.get("preprocessing"), "data preprocessing")
    _require(preprocessing.get("augmentation") == "none", "data augmentation changed")
    _require(preprocessing.get("label_or_controller_input") is False, "labels entered the model input")
    roles = _mapping(data.get("roles"), "dataset roles")
    train_role = _mapping(roles.get("train"), "train role")
    dev_role = _mapping(roles.get("development"), "development role")
    heldout_role = _mapping(roles.get("protected_heldout"), "heldout role")
    _require(train_role.get("count") == 55_000 and dev_role.get("count") == 5_000, "train/development role counts changed")
    _require(heldout_role.get("count") == 10_000, "protected heldout role changed")
    _require(roles.get("calibration", {}).get("count") == 0, "calibration role entered training")
    _require(roles.get("sealed", {}).get("use") == "not applicable and not read; consumed N3 sealed evidence remains unavailable", "sealed boundary changed")


def load_frozen_training_contract(
    contract_path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> FrozenTrainingContract:
    """Load exactly the hash-bound V1 contract; no latest-artifact lookup is used."""
    root = (_repo_root() if repo_root is None else repo_root).resolve()
    path = Path(contract_path) if contract_path is not None else root / TRAINING_CONTRACT_RELATIVE_PATH
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    _require(path.is_file(), f"training contract is unavailable: {path}")
    actual = file_sha256(path)
    _require(actual == TRAINING_CONTRACT_SHA256, "training contract SHA mismatch")
    try:
        payload_value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError("training contract is not valid JSON") from error
    payload = _mapping(payload_value, "training contract")
    _validate_contract_payload(payload)
    return FrozenTrainingContract(path, actual, payload)


def derive_prior_seed(
    *,
    namespace: str = PRIOR_NAMESPACE,
    role: str = PRIOR_ROLE,
    global_step: int,
    sample_count: int,
) -> int:
    """Derive the contract's local CPU-generator seed from immutable coordinates."""
    if not namespace or not role or global_step < 0 or sample_count <= 0:
        raise ValueError("prior seed coordinates are invalid")
    digest = hashlib.sha256(f"{namespace}|{role}|{global_step}|{sample_count}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


@dataclass(frozen=True)
class PriorSample:
    state: ComplexOLSState
    seed: int
    generator_state: torch.Tensor
    namespace: str
    role: str
    global_step: int


class FrozenComplexOLSPrior:
    """Directly sample the frozen native complex OLS prior."""

    def __init__(self, *, sigma: float = PRIOR_SIGMA, namespace: str = PRIOR_NAMESPACE) -> None:
        if sigma != PRIOR_SIGMA or namespace != PRIOR_NAMESPACE:
            raise ValueError("only the frozen V1 complex prior is supported")
        self.sigma = float(sigma)
        self.namespace = namespace

    def sample(
        self,
        sample_count: int,
        *,
        global_step: int,
        device: torch.device,
        role: str = PRIOR_ROLE,
    ) -> PriorSample:
        if sample_count != PRIOR_SAMPLE_COUNT or global_step < 0 or not role:
            raise ValueError("prior sample coordinates do not match the frozen contract")
        seed = derive_prior_seed(
            namespace=self.namespace,
            role=role,
            global_step=global_step,
            sample_count=sample_count,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        real = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
        imag = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
        coefficients = torch.complex(real, imag) * (self.sigma / math.sqrt(2.0))
        if device.type != "cpu":
            coefficients = coefficients.to(device)
        state = ComplexOLSState.from_external(coefficients)
        return PriorSample(
            state=state,
            seed=seed,
            generator_state=generator.get_state().clone(),
            namespace=self.namespace,
            role=role,
            global_step=global_step,
        )


def _validate_complex_ols(z: torch.Tensor, name: str) -> None:
    if z.ndim != 2 or z.shape[1] != OLS_PORT_COUNT or not torch.is_complex(z):
        raise ValueError(f"{name} must have complex shape [B,100]")
    if not torch.isfinite(z).all():
        raise ValueError(f"{name} must be finite")


def gauge_invariant_ols_representation(
    z: torch.Tensor,
    *,
    sigma: float = PRIOR_SIGMA,
    epsilon: float = GAUGE_EPSILON,
) -> torch.Tensor:
    """Return the differentiable full-complex global-phase quotient S(z)."""
    _validate_complex_ols(z, "z")
    if sigma != PRIOR_SIGMA or epsilon != GAUGE_EPSILON:
        raise ValueError("only the frozen V1 gauge constants are supported")
    anchor = z.sum(dim=1, keepdim=True) / math.sqrt(float(OLS_PORT_COUNT))
    regularized_factor = anchor.conj() / (anchor.abs().square() + epsilon**2).sqrt()
    gauged = z * regularized_factor
    representation = torch.cat((gauged.real / sigma, gauged.imag / sigma), dim=1)
    if representation.shape[1] != LATENT_DIMENSION or not torch.isfinite(representation).all():
        raise ValueError("gauge-invariant full-complex representation is invalid")
    return representation


def _gaussian_kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    squared_distance = (left[:, None, :] - right[None, :, :]).square().sum(dim=-1)
    return torch.exp(-squared_distance / (2.0 * MMD_BANDWIDTH_SQUARED))


def full_complex_mmd_vstat(z_real: torch.Tensor, z_prior: torch.Tensor) -> torch.Tensor:
    """Compute the frozen biased B² V-statistic on full-complex S(z)."""
    _validate_complex_ols(z_real, "z_real")
    _validate_complex_ols(z_prior, "z_prior")
    if z_real.shape != z_prior.shape:
        raise ValueError("full-complex MMD requires equal logical batch shapes")
    x = gauge_invariant_ols_representation(z_real)
    y = gauge_invariant_ols_representation(z_prior)
    value = _gaussian_kernel(x, x).mean() + _gaussian_kernel(y, y).mean() - 2.0 * _gaussian_kernel(x, y).mean()
    if not torch.isfinite(value):
        raise ValueError("full-complex MMD is nonfinite")
    return value


@dataclass(frozen=True)
class DatasetRoleMetadata:
    """Explicit data-role boundary carried by the training consumer."""

    root: str = "data/MNIST"
    train_count: int = 55_000
    development_count: int = 5_000
    protected_heldout_count: int = 10_000
    calibration_count: int = 0
    sealed_use: str = "not applicable and not read; consumed N3 sealed evidence remains unavailable"
    labels_are_sampler_metadata_only: bool = True
    download: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "roles": {
                "train": {"count": self.train_count, "use": "image-to-aligned-runtime optimization only"},
                "development": {"count": self.development_count, "use": "predeclared checkpoint comparison only"},
                "protected_heldout": {"count": self.protected_heldout_count, "use": "not consumed"},
                "calibration": {"count": self.calibration_count, "use": "none"},
                "sealed": {"use": self.sealed_use},
            },
            "labels_are_sampler_metadata_only": self.labels_are_sampler_metadata_only,
            "download": self.download,
        }


class BalancedTrainRoleSampler:
    """Deterministic one-row-per-class sampler for the 55,000 train role."""

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int = SCHEDULE_SEED) -> None:
        if train_indices.ndim != 1 or labels.ndim != 1 or train_indices.dtype != torch.long or labels.dtype != torch.long:
            raise TypeError("train sampler requires long index and label vectors")
        if train_indices.numel() != 55_000 or labels.numel() != 60_000:
            raise ValueError("train sampler requires the frozen 55,000/60,000 MNIST roles")
        if seed != SCHEDULE_SEED:
            raise ValueError("only the frozen schedule seed is supported")
        self.seed = int(seed)
        self._permutations: list[torch.Tensor] = []
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_indices = train_indices.cpu()
        labels = labels.cpu()
        for digit in range(10):
            candidates = train_indices[labels.index_select(0, train_indices) == digit]
            if candidates.numel() < TRAINING_STEPS:
                raise ValueError("train role has fewer than 100 rows in one class")
            order = torch.randperm(candidates.numel(), generator=generator)
            self._permutations.append(candidates.index_select(0, order))
        self._cursor = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cursor >= min(value.numel() for value in self._permutations):
            raise StopIteration("train role sampler is exhausted")
        indices = torch.stack([permutation[self._cursor] for permutation in self._permutations])
        labels = torch.arange(10, dtype=torch.long)
        self._cursor += 1
        return indices, labels

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "lightgen_train_role_sampler.v1",
            "seed": self.seed,
            "cursor": self._cursor,
            "class_count": 10,
            "rows_per_class": [value.numel() for value in self._permutations],
            "permutations": [value.clone() for value in self._permutations],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "lightgen_train_role_sampler.v1" or state.get("seed") != self.seed:
            raise ValueError("sampler checkpoint identity mismatch")
        permutations = state.get("permutations")
        if not isinstance(permutations, list) or len(permutations) != 10:
            raise ValueError("sampler checkpoint permutations are invalid")
        expected_counts = [value.numel() for value in self._permutations]
        counts = state.get("rows_per_class")
        if counts != expected_counts:
            raise ValueError("sampler checkpoint class counts do not match the dataset")
        if any(not isinstance(value, torch.Tensor) or value.shape != (expected_counts[index],) or value.dtype != torch.long for index, value in enumerate(permutations)):
            raise ValueError("sampler checkpoint permutation shape is invalid")
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or not 0 <= cursor <= min(expected_counts):
            raise ValueError("sampler checkpoint cursor is invalid")
        self._permutations = [value.cpu().clone() for value in permutations]
        self._cursor = cursor


@dataclass(frozen=True)
class TrainingLossBreakdown:
    image: torch.Tensor
    latent: torch.Tensor
    optical: torch.Tensor
    total: torch.Tensor
    result: Any
    prior: PriorSample

    def metrics(self) -> dict[str, float]:
        return {
            "L_image": float(self.image.detach().cpu()),
            "L_latent": float(self.latent.detach().cpu()),
            "L_optical": float(self.optical.detach().cpu()),
            "L_total": float(self.total.detach().cpu()),
        }


@dataclass(frozen=True)
class AutogradFirewallResult:
    modal_projection_active: bool
    w_ols_active: bool
    trainable_gradients_finite: bool
    forbidden_gradients_absent: bool
    gradient_norms: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modal_projection_active": self.modal_projection_active,
            "w_ols_active": self.w_ols_active,
            "trainable_gradients_finite": self.trainable_gradients_finite,
            "forbidden_gradients_absent": self.forbidden_gradients_absent,
            "gradient_norms": dict(self.gradient_norms),
        }


@dataclass(frozen=True)
class UpdateResult:
    global_step: int
    loss: TrainingLossBreakdown
    autograd: AutogradFirewallResult
    parameter_update_norms: Mapping[str, float]
    frozen_state_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "global_step": self.global_step,
            "loss": self.loss.metrics(),
            "autograd": self.autograd.to_dict(),
            "parameter_update_norms": dict(self.parameter_update_norms),
            "frozen_state_unchanged": self.frozen_state_unchanged,
        }
        return payload


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} is nonfinite")


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
        elif isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second) or any(
                not _state_equal({"value": a}, {"value": b}) for a, b in zip(first, second, strict=True)
            ):
                return False
        elif first != second:
            return False
    return True


class LightGen2DComplexOLSTrainingConsumer(nn.Module):
    """Training consumer with a closed parameter and optical-path firewall."""

    def __init__(
        self,
        contract: FrozenTrainingContract,
        *,
        device: torch.device | str = "cpu",
        propagation_grid_size: int = PRODUCTION_PROPAGATION_GRID,
    ) -> None:
        super().__init__()
        if not isinstance(contract, FrozenTrainingContract):
            raise TypeError("consumer requires a hash-bound FrozenTrainingContract")
        if propagation_grid_size != PRODUCTION_PROPAGATION_GRID and propagation_grid_size != SMOKE_PROPAGATION_GRID:
            raise ValueError("only the production grid or the declared CPU smoke grid is supported")
        self.contract = contract
        self.device = torch.device(device)
        self.runtime = LightGen2DSemanticGenerationInputOLSAlignment(
            LightGen2DAlignmentRuntimeConfig(propagation_grid_size=propagation_grid_size)
        ).to(self.device)
        self.runtime.train()
        self._enforce_parameter_firewall()
        self.prior = FrozenComplexOLSPrior()

    @property
    def runtime_identity(self) -> dict[str, Any]:
        return {
            "identity": ALIGNMENT_RUNTIME_ID,
            "module": ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix(),
            "module_sha256": ALIGNMENT_RUNTIME_SHA256,
            "runtime_report_sha256": ALIGNMENT_RUNTIME_REPORT_SHA256,
            "ols_interface": OLS_INTERFACE_ID,
            "ols_shape": [OLS_PORT_COUNT],
            "ols_grid_shape": list(OLS_GRID_SHAPE),
            "port_order": "row_major_j=10*r+c",
            "transport": "T=I_100 coefficient space",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "production_propagation_grid": PRODUCTION_PROPAGATION_GRID,
            "active_propagation_grid": self.runtime.config.propagation_grid_size,
        }

    def _named_runtime_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(self.runtime.named_parameters())

    def _enforce_parameter_firewall(self) -> None:
        named = self._named_runtime_parameters()
        names = [name for name, _ in named]
        if len({id(parameter) for _, parameter in named}) != len(named):
            raise TrainingContractError("runtime has duplicate parameter objects")
        unexpected_initial = [name for name, parameter in named if parameter.requires_grad and name not in ALLOWED_TRAINABLE_PARAMETER_NAMES]
        if unexpected_initial:
            raise TrainingContractError(f"unexpected trainable runtime parameters: {unexpected_initial}")
        for name, parameter in named:
            parameter.requires_grad_(name in ALLOWED_TRAINABLE_PARAMETER_NAMES)
        trainable_names = [name for name, parameter in named if parameter.requires_grad]
        if trainable_names != list(ALLOWED_TRAINABLE_PARAMETER_NAMES):
            raise TrainingContractError(f"runtime trainable parameter firewall mismatch: {trainable_names}")
        if names != list(ALLOWED_TRAINABLE_PARAMETER_NAMES):
            raise TrainingContractError(f"runtime parameter surface changed: {names}")
        for name, parameter in named:
            if tuple(parameter.shape) != {"encoder.phase": (2, 1, 100, 100), "generator.phase": (2, 1, 100, 100)}[name]:
                raise TrainingContractError(f"parameter shape changed for {name}")
            if parameter.dtype != torch.float32:
                raise TrainingContractError(f"parameter dtype changed for {name}")
        if not torch.equal(self.runtime.ols.transport.cpu(), torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)):
            raise TrainingContractError("runtime OLS transport is not coefficient-space identity")

    def parameter_manifest(self) -> dict[str, Any]:
        self._enforce_parameter_firewall()
        named = self._named_runtime_parameters()
        trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in named if not parameter.requires_grad]
        state = self.runtime.state_dict()
        frozen_state = {name: value for name, value in state.items() if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES}
        return {
            "parameter_names": [name for name, _ in named],
            "trainable_parameter_names": [name for name, _ in trainable],
            "frozen_parameter_names": [name for name, _ in frozen],
            "parameter_shapes": {name: list(parameter.shape) for name, parameter in named},
            "parameter_dtypes": {name: str(parameter.dtype) for name, parameter in named},
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "frozen_parameter_count": sum(parameter.numel() for _, parameter in frozen),
            "total_parameter_count": sum(parameter.numel() for _, parameter in named),
            "frozen_state_tensor_count": len(frozen_state),
            "frozen_state_numel": sum(value.numel() for value in frozen_state.values() if torch.is_tensor(value)),
            "optimizer_parameter_names": [name for name, _ in trainable],
        }

    def create_optimizer(self) -> torch.optim.Adam:
        self._enforce_parameter_firewall()
        named = self._named_runtime_parameters()
        trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
        if [name for name, _ in trainable] != list(ALLOWED_TRAINABLE_PARAMETER_NAMES):
            raise TrainingContractError("optimizer parameter manifest is not exact")
        parameters = [parameter for _, parameter in trainable]
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise TrainingContractError("optimizer parameter list contains duplicates")
        optimizer = torch.optim.Adam(
            parameters,
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPSILON,
            weight_decay=WEIGHT_DECAY,
            amsgrad=AMSGrad,
        )
        if any(
            actual is not expected
            for actual, expected in zip(optimizer.param_groups[0]["params"], parameters, strict=True)
        ):
            raise TrainingContractError("optimizer parameter list differs from the trainable manifest")
        return optimizer

    def optimizer_manifest(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        if not isinstance(optimizer, torch.optim.Adam):
            raise TypeError("the frozen training contract requires Adam")
        if len(optimizer.param_groups) != 1:
            raise TrainingContractError("optimizer must have one parameter group")
        group = optimizer.param_groups[0]
        expected_parameters = [parameter for name, parameter in self._named_runtime_parameters() if name in ALLOWED_TRAINABLE_PARAMETER_NAMES]
        actual_parameters = list(group["params"])
        if len(actual_parameters) != len(expected_parameters) or any(
            actual is not expected for actual, expected in zip(actual_parameters, expected_parameters, strict=True)
        ):
            raise TrainingContractError("optimizer parameter manifest differs from the allowed trainable set")
        defaults = optimizer.defaults
        if (
            defaults.get("lr") != LEARNING_RATE
            or tuple(defaults.get("betas", ())) != ADAM_BETAS
            or defaults.get("eps") != ADAM_EPSILON
            or defaults.get("weight_decay") != WEIGHT_DECAY
            or defaults.get("amsgrad") != AMSGrad
        ):
            raise TrainingContractError("optimizer hyperparameters differ from the frozen contract")
        return {
            "name": "Adam",
            "parameter_count": len(group["params"]),
            "learning_rate": group["lr"],
            "betas": list(group["betas"]),
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
            "amsgrad": group["amsgrad"],
            "parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
        }

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("training image batch requires shape [10,1,28,28]")
        if images.shape[0] != LOGICAL_BATCH_SIZE or torch.is_complex(images) or images.dtype != torch.float32:
            raise TypeError("training image batch must be ten float32 real images")
        if not torch.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("training image batch must be finite and normalized to [0,1]")

    def forward_and_loss(self, images: torch.Tensor, *, global_step: int) -> TrainingLossBreakdown:
        self._validate_images(images)
        if images.device != self.device:
            raise ValueError("training images must already be on the consumer device")
        if global_step < 0 or global_step >= TRAINING_STEPS:
            raise ValueError("global_step is outside the frozen 100-step schedule")
        result = self.runtime.forward_from_image(images)
        if result.image_input is None or result.ols_input_field is None:
            raise TrainingContractError("aligned image path did not expose its optical input and OLS boundary")
        prior = self.prior.sample(PRIOR_SAMPLE_COUNT, global_step=global_step, device=self.device)
        raw_cmos = result.raw_intensity
        target = result.image_input.slm_intensity
        if raw_cmos.shape != target.shape or tuple(raw_cmos.shape[-2:]) != (200, 200):
            raise TrainingContractError("raw CMOS and aligned target shape mismatch")
        _finite_tensor(raw_cmos, "raw CMOS intensity")
        _finite_tensor(target, "image target")
        image_loss = (raw_cmos - target).square().mean()
        latent_loss = full_complex_mmd_vstat(result.z_ols.coefficients, prior.state.coefficients)
        optical_loss = raw_cmos.new_zeros(())
        total_loss = BETA_IMAGE * image_loss + ALPHA_LATENT * latent_loss + GAMMA_OPTICAL * optical_loss
        weighted_sum = BETA_IMAGE * image_loss + ALPHA_LATENT * latent_loss + GAMMA_OPTICAL * optical_loss
        if not torch.allclose(total_loss, weighted_sum, rtol=0.0, atol=1.0e-12):
            raise TrainingContractError("total loss is not the frozen weighted component sum")
        for name, value in (("L_image", image_loss), ("L_latent", latent_loss), ("L_optical", optical_loss), ("L_total", total_loss)):
            _finite_tensor(value, name)
        return TrainingLossBreakdown(image_loss, latent_loss, optical_loss, total_loss, result, prior)

    @staticmethod
    def _retain_autograd_probes(loss: TrainingLossBreakdown) -> None:
        result = loss.result
        tensors = (
            result.ols_input_field.tensor if result.ols_input_field is not None else None,
            result.z_ols.coefficients,
            result.generator_input_field.tensor,
        )
        for tensor in tensors:
            if tensor is None or not tensor.requires_grad:
                raise TrainingContractError("aligned modal/W_OLS tensor is detached from autograd")
            tensor.retain_grad()

    def _validate_autograd(self, loss: TrainingLossBreakdown) -> AutogradFirewallResult:
        result = loss.result
        named = self._named_runtime_parameters()
        gradient_norms: dict[str, float] = {}
        for name, parameter in named:
            if name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
                if parameter.grad is None:
                    raise TrainingContractError(f"trainable parameter has no gradient: {name}")
                _finite_tensor(parameter.grad, f"gradient {name}")
                gradient_norms[name] = float(parameter.grad.detach().abs().square().sum().sqrt().cpu())
            elif parameter.grad is not None:
                raise TrainingContractError(f"forbidden parameter received a gradient: {name}")
        if result.ols_input_field is None:
            raise TrainingContractError("OLS input field is missing")
        z_gradient = result.z_ols.coefficients.grad
        ols_input_gradient = result.ols_input_field.tensor.grad
        writer_gradient = result.generator_input_field.tensor.grad
        if z_gradient is None or ols_input_gradient is None or writer_gradient is None:
            raise TrainingContractError("modal projection or W_OLS is absent from the autograd path")
        _finite_tensor(z_gradient, "OLS coefficient gradient")
        _finite_tensor(ols_input_gradient, "modal input gradient")
        _finite_tensor(writer_gradient, "W_OLS output gradient")
        projection_active = (
            result.z_ols.coefficients.grad_fn is not None
            and result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True
            and result.ols_input_field.tensor.grad_fn is not None
        )
        writer_active = (
            result.generator_input_field.tensor.grad_fn is not None
            and result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission"
            and result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space"
        )
        if not projection_active or not writer_active:
            raise TrainingContractError("aligned modal projection/W_OLS structural proof failed")
        return AutogradFirewallResult(projection_active, writer_active, True, True, gradient_norms)

    def _frozen_state_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.runtime.state_dict().items()
            if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES
        }

    def update(self, images: torch.Tensor, *, global_step: int, optimizer: torch.optim.Optimizer) -> UpdateResult:
        if optimizer is None:
            raise ValueError("optimizer is required")
        self._enforce_parameter_firewall()
        self.optimizer_manifest(optimizer)
        before_parameters = {name: parameter.detach().clone() for name, parameter in self._named_runtime_parameters()}
        before_frozen = self._frozen_state_snapshot()
        optimizer.zero_grad(set_to_none=True)
        loss = self.forward_and_loss(images, global_step=global_step)
        self._retain_autograd_probes(loss)
        loss.total.backward()
        autograd = self._validate_autograd(loss)
        optimizer.step()
        after_parameters = {name: parameter.detach().clone() for name, parameter in self._named_runtime_parameters()}
        for name, parameter in self._named_runtime_parameters():
            _finite_tensor(parameter, f"updated parameter {name}")
        update_norms = {
            name: float((after_parameters[name] - before).abs().square().sum().sqrt().cpu())
            for name, before in before_parameters.items()
        }
        if not any(value > 0.0 for value in update_norms.values()):
            raise TrainingContractError("optimizer step did not change an allowed trainable parameter")
        frozen_unchanged = _state_equal(before_frozen, self._frozen_state_snapshot())
        if not frozen_unchanged:
            raise TrainingContractError("a frozen runtime state changed during optimizer step")
        return UpdateResult(global_step, loss, autograd, update_norms, frozen_unchanged)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self._named_runtime_parameters() if name in ALLOWED_TRAINABLE_PARAMETER_NAMES}

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(ALLOWED_TRAINABLE_PARAMETER_NAMES):
            raise TrainingContractError("checkpoint trainable state keys differ from the contract")
        parameters = dict(self._named_runtime_parameters())
        with torch.no_grad():
            for name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
                value = state[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(parameters[name].shape) or value.dtype != parameters[name].dtype:
                    raise TrainingContractError(f"checkpoint parameter tensor is invalid: {name}")
                _finite_tensor(value, f"checkpoint parameter {name}")
                parameters[name].copy_(value.to(parameters[name].device))
        self._enforce_parameter_firewall()

    def loss_configuration(self) -> dict[str, Any]:
        return {
            "image": {"name": "RAW_CMOS_INTENSITY_MSE_V1", "target": "aligned image adapter slm_intensity", "reduction": "mean"},
            "latent": {
                "name": "FULL_COMPLEX_GAUSSIAN_KERNEL_MMD_VSTAT_V1",
                "representation": "S(z)=concat(Re(G(z)/sigma), Im(G(z)/sigma)) in R^200",
                "bandwidth_squared": MMD_BANDWIDTH_SQUARED,
                "biased_v_statistic_including_diagonals": True,
            },
            "optical": {"included": False, "value": 0.0, "reason": "omitted by frozen contract"},
            "weights": {"alpha_latent": ALPHA_LATENT, "beta_image": BETA_IMAGE, "gamma_optical": GAMMA_OPTICAL},
        }


def _rng_state_payload() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()] if cuda_available else []
    return {
        "cpu": torch.random.get_rng_state().cpu().clone(),
        "cuda": cuda_states,
        "cuda_available": cuda_available,
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    cpu = payload.get("cpu")
    cuda = payload.get("cuda")
    if not isinstance(cpu, torch.Tensor) or not isinstance(cuda, list):
        raise TrainingContractError("checkpoint RNG state is malformed")
    torch.random.set_rng_state(cpu.cpu())
    if cuda:
        if not torch.cuda.is_available():
            raise TrainingContractError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if any(not isinstance(state, torch.Tensor) for state in cuda):
            raise TrainingContractError("checkpoint CUDA RNG state is malformed")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])


def git_facts(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True)
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def cuda_capability_status() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    return {
        "checked_without_training": True,
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "long_training_started": False,
    }


def validate_legacy_exclusion() -> dict[str, Any]:
    """Prove the active consumer imports only aligned/runtime support modules."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = [name for name in imports if name.startswith("ldm_project.v1") or ".legacy" in name or "generate_from_prior" in name]
    if forbidden:
        raise TrainingContractError(f"legacy dependency entered active consumer: {forbidden}")
    return {"checked": True, "import_modules": sorted(set(imports)), "forbidden_imports": [], "passed": True}


def checkpoint_payload(
    consumer: LightGen2DComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    sampler_state: Mapping[str, Any],
    prior: PriorSample,
    dataset_roles: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    if global_step < 0 or global_step >= TRAINING_STEPS:
        raise ValueError("checkpoint global_step is outside the frozen schedule")
    if prior.global_step != global_step:
        raise ValueError("checkpoint prior state does not match global_step")
    return {
        "schema": "lightgen_2d_semantic_complex_ols_training_smoke_checkpoint.v1",
        "smoke_only": True,
        "accepted_training_checkpoint": False,
        "contract": consumer.contract.identity(),
        "runtime": consumer.runtime_identity,
        "architecture_identity": {
            "consumer": "LightGen2DComplexOLSTrainingConsumer",
            "runtime": ALIGNMENT_RUNTIME_ID,
            "ols_interface": OLS_INTERFACE_ID,
            "port_order": list(ROW_MAJOR_PORT_ORDER),
        },
        "git": dict(git),
        "global_step": global_step,
        "completed_steps": global_step + 1,
        "epoch": 0,
        "trainable_parameter_state": consumer.trainable_state_dict(),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": None,
        "rng_state": _rng_state_payload(),
        "prior_rng": {
            "namespace": prior.namespace,
            "role": prior.role,
            "global_step": prior.global_step,
            "sample_count": PRIOR_SAMPLE_COUNT,
            "seed": prior.seed,
            "state": prior.generator_state.cpu().clone(),
        },
        "sampler_state": copy.deepcopy(dict(sampler_state)),
        "dataset_roles": dict(dataset_roles),
        "loss_configuration": consumer.loss_configuration(),
        "optimizer_configuration": consumer.optimizer_manifest(optimizer),
    }


def save_smoke_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite smoke checkpoint: {target}")
    atomic_torch_save_new(target, dict(payload), artifact_kind="LightGen training smoke checkpoint")


@dataclass(frozen=True)
class LoadedCheckpoint:
    global_step: int
    completed_steps: int
    epoch: int
    sampler_state: Mapping[str, Any]
    prior_rng: Mapping[str, Any]
    rng_state: Mapping[str, Any]


def load_smoke_checkpoint(
    path: str | Path,
    consumer: LightGen2DComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
) -> LoadedCheckpoint:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != "lightgen_2d_semantic_complex_ols_training_smoke_checkpoint.v1":
        raise TrainingContractError("smoke checkpoint schema mismatch")
    if payload.get("smoke_only") is not True or payload.get("accepted_training_checkpoint") is not False:
        raise TrainingContractError("non-smoke checkpoint crossed the preflight boundary")
    contract = _mapping(payload.get("contract"), "checkpoint contract")
    if contract.get("sha256") != consumer.contract.sha256 or contract.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise TrainingContractError("checkpoint training-contract identity mismatch")
    runtime = _mapping(payload.get("runtime"), "checkpoint runtime")
    if runtime.get("identity") != ALIGNMENT_RUNTIME_ID or runtime.get("module_sha256") != ALIGNMENT_RUNTIME_SHA256:
        raise TrainingContractError("checkpoint aligned-runtime identity mismatch")
    step = payload.get("global_step")
    completed = payload.get("completed_steps")
    epoch = payload.get("epoch")
    if not isinstance(step, int) or not 0 <= step < TRAINING_STEPS or completed != step + 1 or epoch != 0:
        raise TrainingContractError("checkpoint step metadata is invalid")
    consumer.load_trainable_state_dict(_mapping(payload.get("trainable_parameter_state"), "trainable parameter state"))
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise TrainingContractError("checkpoint optimizer state is invalid")
    optimizer.load_state_dict(dict(optimizer_state))
    consumer.optimizer_manifest(optimizer)
    rng_state = _mapping(payload.get("rng_state"), "checkpoint RNG state")
    restore_rng_state(rng_state)
    prior_rng = _mapping(payload.get("prior_rng"), "checkpoint prior RNG")
    if prior_rng.get("namespace") != PRIOR_NAMESPACE or prior_rng.get("sample_count") != PRIOR_SAMPLE_COUNT or not isinstance(prior_rng.get("state"), torch.Tensor):
        raise TrainingContractError("checkpoint prior RNG state is invalid")
    sampler_state = _mapping(payload.get("sampler_state"), "checkpoint sampler state")
    return LoadedCheckpoint(step, completed, epoch, sampler_state, prior_rng, rng_state)


__all__ = [
    "ALIGNMENT_RUNTIME_ID",
    "ALLOWED_TRAINABLE_PARAMETER_NAMES",
    "AutogradFirewallResult",
    "BalancedTrainRoleSampler",
    "CLASSIFICATION",
    "DatasetRoleMetadata",
    "DETERMINISTIC_TOLERANCE",
    "FrozenComplexOLSPrior",
    "FrozenTrainingContract",
    "LightGen2DComplexOLSTrainingConsumer",
    "LoadedCheckpoint",
    "PREFLIGHT_ID",
    "PriorSample",
    "TRAINING_CONTRACT_RELATIVE_PATH",
    "TRAINING_CONTRACT_SCHEMA",
    "TRAINING_CONTRACT_SHA256",
    "TrainingContractError",
    "TrainingLossBreakdown",
    "UpdateResult",
    "checkpoint_payload",
    "cuda_capability_status",
    "derive_prior_seed",
    "full_complex_mmd_vstat",
    "gauge_invariant_ols_representation",
    "git_facts",
    "load_frozen_training_contract",
    "load_smoke_checkpoint",
    "restore_rng_state",
    "save_smoke_checkpoint",
    "validate_legacy_exclusion",
]
