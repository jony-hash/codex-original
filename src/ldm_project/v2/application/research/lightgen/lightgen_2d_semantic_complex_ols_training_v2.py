"""Bounded two-stage LightGen complex-OLS training consumer.

The consumer implements the already-approved V2 contract but does not expose a
long-training command.  It provides the exact losses, train-only scale
boundary, deterministic prior, state machine, and checkpoint/resume machinery
that a later explicitly authorized training entry point will consume.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ldm_project.v2.evidence import atomic_torch_save_new, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import GAUSSIAN_OLS_MODEL, LP01_OLS_MODEL, OLS_MODELS, ComplexOLSState, LightGen2DAlignmentRuntimeConfig, LightGen2DForwardResult, LightGen2DSemanticGenerationInputOLSAlignment, OLS_INTERFACE_ID, OLS_PORT_COUNT


TASK_ID = "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2"
CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2_IMPLEMENTED"
V2_CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_contract.v2"
V2_CONTRACT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
ALIGNMENT_RUNTIME_ID = "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1"
ALIGNMENT_RUNTIME_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
ALIGNMENT_RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"
OLS_GRID_SHAPE = (10, 10)
LATENT_DIMENSION = 2 * OLS_PORT_COUNT
LOGICAL_BATCH_SIZE = 10
STAGE_A_STEPS = 300
STAGE_B_STEPS = 300
CHECKPOINT_STEPS = (100, 200, 300)
SEMANTIC_THRESHOLD = 0.937
EPSILON_SCALE = 1.0e-12
GAUGE_EPSILON = 1.0e-6
MMD_BANDWIDTH_SQUARED = 200.0
PRIOR_NAMESPACE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_PRIOR"
SCHEDULE_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGrad = False
SMOKE_PROPAGATION_GRID = 128
PRODUCTION_PROPAGATION_GRID = 2048
DETERMINISTIC_TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}
ALLOWED_TRAINABLE_PARAMETER_NAMES = ("encoder.phase", "generator.phase")

SCALE_SCHEMA = "lightgen_2d_semantic_complex_ols_training_scale.v2"
CHECKPOINT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_checkpoint.v2"
SCALE_SMOKE_CLASSIFICATION = "NON_SCIENTIFIC_SMOKE"

STAGE_A_TRAIN = "STAGE_A_TRAIN"
STAGE_A_SEMANTIC_CHECK = "STAGE_A_SEMANTIC_CHECK"
STAGE_A_SELECT = "STAGE_A_SELECT"
SCALE_EXTRACT = "SCALE_EXTRACT"
SCALE_FREEZE = "SCALE_FREEZE"
STAGE_B_TRAIN = "STAGE_B_TRAIN"
STAGE_B_SEMANTIC_CHECK = "STAGE_B_SEMANTIC_CHECK"
FINAL_SELECT = "FINAL_SELECT"
COMPLETE = "COMPLETE"
STAGE_A_FAILURE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED"
STAGE_B_FAILURE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_PRESERVATION_NOT_ESTABLISHED"
STAGE_A_CHECKPOINT_RULE = "minimum development L_image among eligible checkpoints"
STAGE_B_CHECKPOINT_RULE = "among semantic-eligible checkpoints, minimum frozen development L_B"


class TrainingContractError(ValueError):
    """Raised when a frozen external authority or runtime invariant is invalid."""


class DataRoleError(TrainingContractError):
    """Raised when a forbidden data role crosses a consumer boundary."""


class StateMachineError(TrainingContractError):
    """Raised when a V2 state transition is not permitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} is nonfinite")


def _validate_complex(z: torch.Tensor, name: str, *, batch_size: int | None = None) -> None:
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[1] != OLS_PORT_COUNT or not torch.is_complex(z):
        raise ValueError(f"{name} must be complex with shape [B,100]")
    if batch_size is not None and z.shape[0] != batch_size:
        raise ValueError(f"{name} must have batch size {batch_size}")
    _finite(z, name)


@dataclass(frozen=True)
class FrozenV2TrainingContract:
    """Read-only, schema-validated view of the materialized V2 contract."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def identity(self, root: Path | None = None) -> dict[str, str]:
        base = _repo_root() if root is None else root.resolve()
        return {"schema": str(self.payload["schema"]), "path": _relative(self.path, base), "sha256": self.sha256}


def _validate_contract_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == V2_CONTRACT_SCHEMA, "V2 contract schema mismatch")
    _require(payload.get("version") == 2, "V2 contract version mismatch")
    _require(payload.get("status") == "valid_completed" and payload.get("result") == "READY", "V2 contract is not READY")
    _require(payload.get("classification") == "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V2_READY", "V2 contract classification mismatch")
    _require(payload.get("training_authorized") is False, "V2 contract authorized training")
    _require(payload.get("formal_promotion") is False, "V2 contract authorized promotion")
    _require(payload.get("generation_readiness") == "not_ready", "V2 generation boundary changed")
    _require(payload.get("approved_r0_transcription") is True, "V2 contract is not the approved R0 transcription")

    architecture = _mapping(payload.get("architecture"), "architecture")
    _require(architecture.get("scope") == "2D semantic image generation only", "V2 scope changed")
    _require(architecture.get("same_real_and_generated_interface") is True, "shared real/generated interface changed")
    _require(architecture.get("whole_field_identity_bypass") is False, "whole-field identity bypass entered V2")

    trainable = _mapping(payload.get("trainable_parameter_contract"), "trainable parameter contract")
    names = trainable.get("exact_trainable_parameters")
    _require(names == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "V2 trainable parameter names changed")
    _require(trainable.get("exact_optimizer_parameters") == names, "V2 optimizer parameter names changed")
    shapes = trainable.get("parameter_shapes")
    _require(isinstance(shapes, Mapping), "V2 parameter shapes are missing")
    _require(trainable.get("parameter_count") == 40_000, "V2 parameter count changed")

    evaluator = _mapping(payload.get("semantic_evaluator_contract"), "semantic evaluator contract")
    _require(evaluator.get("gradient_enabled") is False, "semantic evaluator received a gradient boundary")
    threshold = _mapping(evaluator.get("threshold"), "semantic threshold")
    _require(threshold.get("value") == SEMANTIC_THRESHOLD, "semantic threshold changed")
    _require(threshold.get("tuned_from_v2_results") is False, "semantic threshold was tuned from V2")

    data = _mapping(payload.get("data_contract"), "data contract")
    roles = _mapping(data.get("roles"), "data roles")
    _require(_mapping(roles.get("train"), "train role").get("count") == 55_000, "train role count changed")
    development = _mapping(roles.get("development"), "development role")
    _require(development.get("count") == 5_000 and development.get("selection_count") == 1_000, "development role changed")
    _require(_mapping(roles.get("protected"), "protected role").get("use") == "forbidden", "protected role opened")
    _require(_mapping(roles.get("test"), "test role").get("use") == "forbidden", "test role opened")
    _require(_mapping(roles.get("sealed"), "sealed role").get("use", "").startswith("forbidden"), "sealed role opened")

    stage_a = _mapping(payload.get("stage_a"), "Stage-A contract")
    _require(stage_a.get("steps") == STAGE_A_STEPS and stage_a.get("checkpoint_steps") == list(CHECKPOINT_STEPS), "Stage-A budget/cadence changed")
    loss_a = _mapping(stage_a.get("loss"), "Stage-A loss")
    _require(loss_a.get("alpha_latent") == 0.0 and loss_a.get("beta_image") == 1.0 and loss_a.get("gamma_optical") == 0.0, "Stage-A weights changed")
    _require(loss_a.get("latent_mmd_enabled") is False, "Stage-A latent MMD was enabled")
    selection_a = _mapping(stage_a.get("selection"), "Stage-A selection")
    _require(selection_a.get("rule") == STAGE_A_CHECKPOINT_RULE, "Stage-A selection rule changed")

    scale = _mapping(payload.get("scale_contract"), "scale contract")
    _require(scale.get("formula") == "s_j = mean_{i in train} |z_{i,j}|^2, j=0..99", "scale formula changed")
    _require(scale.get("epsilon_scale") == EPSILON_SCALE and scale.get("dtype").startswith("raw z complex64"), "scale semantics changed")
    _require(scale.get("artifact_schema") == SCALE_SCHEMA and scale.get("stage_b_frozen") is True, "scale artifact contract changed")

    normalized = _mapping(payload.get("normalized_latent_contract"), "normalized latent contract")
    _require(normalized.get("authoritative_physical_state") == "z in C^100", "physical latent authority changed")
    _require(normalized.get("formula") == "z_tilde_j = z_j / sqrt(s_j + epsilon_scale)", "normalized coordinate changed")
    prior = _mapping(normalized.get("prior"), "normalized prior")
    _require(prior.get("name") == "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NORMALIZED_OLS_V2", "prior identity changed")
    _require(prior.get("formula") == "z_tilde_prior,j=(epsilon_R+i*epsilon_I)/sqrt(2)", "prior formula changed")
    mmd = _mapping(normalized.get("mmd"), "normalized MMD")
    _require(mmd.get("raw_z_direct_mmd") is False and mmd.get("q_only") is False, "raw/q-only MMD entered V2")

    stage_b = _mapping(payload.get("stage_b"), "Stage-B contract")
    _require(stage_b.get("steps") == STAGE_B_STEPS and stage_b.get("checkpoint_steps") == list(CHECKPOINT_STEPS), "Stage-B budget/cadence changed")
    _require(stage_b.get("optimizer_reset") is True, "Stage-B optimizer reset changed")
    loss_b = _mapping(stage_b.get("loss"), "Stage-B loss")
    _require(loss_b.get("formula") == "L_B = 1.0*L_image + 1.0*L_latent_normalized + 1.0*L_scale", "Stage-B loss changed")
    _require(loss_b.get("one_scale_penalty_only") is True, "additional scale penalty entered V2")
    selection_b = _mapping(stage_b.get("selection"), "Stage-B selection")
    _require(selection_b.get("rule") == STAGE_B_CHECKPOINT_RULE, "Stage-B selection rule changed")

    optimizer = _mapping(payload.get("optimizer_and_precision"), "optimizer contract")
    _require(optimizer.get("logical_batch_size") == LOGICAL_BATCH_SIZE, "logical batch size changed")
    _require(optimizer.get("schedule_seed") == SCHEDULE_SEED, "schedule seed changed")
    precision = _mapping(optimizer.get("precision"), "precision contract")
    _require(precision.get("phase") == "float32" and precision.get("field") == "complex64", "precision changed")
    _require(precision.get("autocast") is False and precision.get("grad_scaler") is False, "mixed precision entered V2")
    adam = _mapping(optimizer.get("optimizer"), "Adam contract")
    _require(adam == {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGrad}, "Adam configuration changed")

    checkpoint = _mapping(payload.get("checkpoint_contract"), "checkpoint contract")
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema changed")
    _require(isinstance(checkpoint.get("required"), list) and isinstance(checkpoint.get("stage_b_additional"), list), "checkpoint fields are incomplete")


def load_v2_training_contract(
    contract_path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> FrozenV2TrainingContract:
    root = (_repo_root() if repo_root is None else repo_root.resolve())
    path = Path(contract_path) if contract_path is not None else root / V2_CONTRACT_RELATIVE_PATH
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    _require(path.is_file(), f"V2 contract is unavailable: {path}")
    actual = file_sha256(path)
    try:
        payload_value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError("V2 contract is not valid JSON") from error
    payload = _mapping(payload_value, "V2 contract")
    _validate_contract_payload(payload)
    runtime_path = root / ALIGNMENT_RUNTIME_RELATIVE_PATH
    _require(runtime_path.is_file(), "aligned runtime source is unavailable")
    return FrozenV2TrainingContract(path, actual, payload)


def derive_prior_seed(
    *,
    stage: str,
    role: str,
    stage_local_step: int,
    sample_count: int,
    namespace: str = PRIOR_NAMESPACE,
) -> int:
    if not stage or not role or stage_local_step < 0 or sample_count <= 0 or namespace != PRIOR_NAMESPACE:
        raise ValueError("invalid normalized-prior seed coordinates")
    digest = hashlib.sha256(f"{namespace}|{stage}|{role}|{stage_local_step}|{sample_count}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


@dataclass(frozen=True)
class NormalizedPriorSample:
    coefficients: torch.Tensor
    seed: int
    generator_state: torch.Tensor
    stage: str
    role: str
    stage_local_step: int

    def __post_init__(self) -> None:
        _validate_complex(self.coefficients, "normalized prior")
        if self.coefficients.dtype != torch.complex64:
            raise TypeError("normalized prior must be complex64")
        if self.generator_state.dtype != torch.uint8:
            raise TypeError("normalized prior RNG state must be uint8")


def sample_normalized_prior(
    sample_count: int,
    *,
    stage: str,
    stage_local_step: int,
    device: torch.device | str,
    role: str = "training_target",
) -> NormalizedPriorSample:
    if sample_count <= 0 or stage_local_step < 0 or not stage or not role:
        raise ValueError("normalized prior coordinates are invalid")
    seed = derive_prior_seed(stage=stage, role=role, stage_local_step=stage_local_step, sample_count=sample_count)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
    imag = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
    coefficients = torch.complex(real, imag) / math.sqrt(2.0)
    target_device = torch.device(device)
    if target_device.type != "cpu":
        coefficients = coefficients.to(target_device)
    return NormalizedPriorSample(
        coefficients=coefficients,
        seed=seed,
        generator_state=generator.get_state().clone(),
        stage=stage,
        role=role,
        stage_local_step=stage_local_step,
    )


@dataclass(frozen=True)
class FrozenPhysicalOLSScale:
    """Immutable physical per-port mean-power scale used by Stage B."""

    values: torch.Tensor
    checkpoint_identity: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    data_role: str = "train"
    epsilon_scale: float = EPSILON_SCALE
    classification: str = "SCIENTIFIC_TRAIN_ONLY"
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    basis_id: str = LP01_OLS_MODEL
    mode_geometry: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 1 or self.values.shape[0] != OLS_PORT_COUNT:
            raise ValueError("physical OLS scale must have shape [100]")
        if self.values.dtype != torch.float64:
            raise TypeError("physical OLS scale must be float64")
        if self.values.requires_grad:
            raise ValueError("physical OLS scale must not require gradients")
        _finite(self.values, "physical OLS scale")
        if self.epsilon_scale != EPSILON_SCALE or bool((self.values <= EPSILON_SCALE).any()):
            raise ValueError("physical OLS scale contains a non-positive or below-epsilon port")
        if self.data_role not in {"train", SCALE_SMOKE_CLASSIFICATION}:
            raise DataRoleError("scale extraction accepts only train or NON_SCIENTIFIC_SMOKE")
        _require(self.basis_id in OLS_MODELS, "physical OLS scale basis_id is unsupported")
        if self.basis_id == GAUSSIAN_OLS_MODEL:
            _require(isinstance(self.mode_geometry, Mapping), "Gaussian physical OLS scale requires mode geometry")
            _require(self.mode_geometry.get("basis_id") == self.basis_id, "Gaussian physical OLS scale geometry/basis mismatch")
            object.__setattr__(self, "mode_geometry", copy.deepcopy(dict(self.mode_geometry)))
        elif self.mode_geometry is not None:
            _require(isinstance(self.mode_geometry, Mapping) and self.mode_geometry.get("basis_id") == self.basis_id, "LP01 physical OLS scale geometry/basis mismatch")
            object.__setattr__(self, "mode_geometry", copy.deepcopy(dict(self.mode_geometry)))

    @classmethod
    def from_tensor(
        cls,
        values: torch.Tensor,
        *,
        checkpoint_identity: Mapping[str, Any] | None = None,
        data_identity: Mapping[str, Any] | None = None,
        data_role: str = SCALE_SMOKE_CLASSIFICATION,
        basis_id: str = LP01_OLS_MODEL,
        mode_geometry: Mapping[str, Any] | None = None,
    ) -> "FrozenPhysicalOLSScale":
        value = values.detach().to(dtype=torch.float64, device="cpu").clone()
        return cls(value, checkpoint_identity or {"role": "synthetic smoke"}, data_identity or {"role": "synthetic smoke"}, data_role=data_role, classification=SCALE_SMOKE_CLASSIFICATION if data_role == SCALE_SMOKE_CLASSIFICATION else "SCIENTIFIC_TRAIN_ONLY", basis_id=basis_id, mode_geometry=mode_geometry)

    @property
    def vector_sha256(self) -> str:
        return tensor_sha256(self.values)

    def identity(self) -> dict[str, Any]:
        identity = {
            "schema": SCALE_SCHEMA,
            "vector_sha256": self.vector_sha256,
            "data_role": self.data_role,
            "classification": self.classification,
            "epsilon_scale": self.epsilon_scale,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }
        if self.basis_id == GAUSSIAN_OLS_MODEL:
            identity["basis_id"] = self.basis_id
            identity["mode_geometry"] = dict(self.mode_geometry or {})
        return identity

    def with_artifact(self, path: str | Path, sha256: str) -> "FrozenPhysicalOLSScale":
        return FrozenPhysicalOLSScale(
            self.values,
            self.checkpoint_identity,
            self.data_identity,
            self.data_role,
            self.epsilon_scale,
            self.classification,
            str(path),
            sha256,
            self.basis_id,
            self.mode_geometry,
        )


def save_scale_artifact(path: str | Path, scale: FrozenPhysicalOLSScale) -> FrozenPhysicalOLSScale:
    target = Path(path)
    payload = {
        "schema": SCALE_SCHEMA,
        "version": 2,
        "classification": scale.classification,
        "data_role": scale.data_role,
        "epsilon_scale": EPSILON_SCALE,
        "port_count": OLS_PORT_COUNT,
        "port_order": "row-major j=10*r+c",
        "values": scale.values.detach().cpu().clone(),
        "vector_sha256": scale.vector_sha256,
        "checkpoint_identity": dict(scale.checkpoint_identity),
        "data_identity": dict(scale.data_identity),
        "stage_b_frozen": True,
        "artifact_self_hash": "not_embedded_to_avoid_self_reference",
    }
    if scale.basis_id == GAUSSIAN_OLS_MODEL:
        payload["basis_id"] = scale.basis_id
        payload["mode_geometry"] = dict(scale.mode_geometry or {})
    atomic_torch_save_new(target, payload, artifact_kind="LightGen V2 physical OLS scale")
    return scale.with_artifact(target, file_sha256(target))


def load_scale_artifact(path: str | Path, *, expected_sha256: str | None = None) -> FrozenPhysicalOLSScale:
    target = Path(path).resolve()
    _require(target.is_file(), f"scale artifact is unavailable: {target}")
    actual = file_sha256(target)
    # Digests remain provenance only. Admission is based on the explicit path,
    # artifact schema, role, and validated numeric scale values below.
    payload = _mapping(safe_torch_load(target, map_location="cpu"), "scale artifact")
    _require(payload.get("schema") == SCALE_SCHEMA and payload.get("version") == 2, "scale artifact schema mismatch")
    _require(payload.get("epsilon_scale") == EPSILON_SCALE and payload.get("port_count") == OLS_PORT_COUNT, "scale artifact convention changed")
    values = payload.get("values")
    _require(isinstance(values, torch.Tensor), "scale artifact values are missing")
    data_role = payload.get("data_role")
    classification = payload.get("classification")
    _require(data_role in {"train", SCALE_SMOKE_CLASSIFICATION}, "scale artifact data role is invalid")
    _require(classification in {"SCIENTIFIC_TRAIN_ONLY", SCALE_SMOKE_CLASSIFICATION}, "scale artifact classification is invalid")
    scale = FrozenPhysicalOLSScale(
        values.detach().to(dtype=torch.float64, device="cpu"),
        _mapping(payload.get("checkpoint_identity"), "scale checkpoint identity"),
        _mapping(payload.get("data_identity"), "scale data identity"),
        data_role,
        EPSILON_SCALE,
        classification,
        str(target),
        actual,
        str(payload.get("basis_id", LP01_OLS_MODEL)),
        payload.get("mode_geometry"),
    )
    return scale


def extract_train_only_ols_scale(
    consumer: "LightGen2DComplexOLSTrainingConsumerV2",
    images: torch.Tensor,
    *,
    data_role: str,
    checkpoint_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    batch_size: int = LOGICAL_BATCH_SIZE,
    basis_id: str = LP01_OLS_MODEL,
    mode_geometry: Mapping[str, Any] | None = None,
) -> FrozenPhysicalOLSScale:
    """Extract ``s_j`` in canonical row order from an explicit data role.

    The production role requires all 55,000 train rows.  A smaller input is
    accepted only when explicitly labelled ``NON_SCIENTIFIC_SMOKE``.
    """

    if data_role not in {"train", SCALE_SMOKE_CLASSIFICATION}:
        raise DataRoleError("scale extraction rejects development/calibration/protected/test/sealed roles")
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("scale images require shape [N,1,28,28]")
    if images.dtype != torch.float32 or torch.is_complex(images) or not torch.isfinite(images).all():
        raise TypeError("scale images must be finite float32")
    if images.numel() and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("scale images must be in [0,1]")
    if batch_size != LOGICAL_BATCH_SIZE:
        raise ValueError("scale extraction batch size is frozen at 10")
    if data_role == "train" and images.shape[0] != 55_000:
        raise DataRoleError("scientific train scale extraction requires exactly 55,000 rows")
    if data_role == SCALE_SMOKE_CLASSIFICATION and images.shape[0] == 0:
        raise ValueError("smoke scale extraction requires at least one row")
    if basis_id != consumer.runtime.ols.basis_id:
        raise TrainingContractError("scale extraction basis differs from the active OLS runtime")
    active_geometry = consumer.runtime.ols.mode_identity()
    if mode_geometry is not None and dict(mode_geometry) != active_geometry:
        raise TrainingContractError("scale extraction geometry differs from the active OLS runtime")

    was_training = consumer.runtime.training
    consumer.runtime.eval()
    total = torch.zeros((OLS_PORT_COUNT,), dtype=torch.float64)
    count = 0
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                batch = images[start : start + batch_size].to(consumer.device)
                result = consumer.runtime.forward_from_image(batch)
                if result.z_ols.source != "real_encoded" or result.ols_input_field is None:
                    raise TrainingContractError("scale extraction did not use the mandatory real modal projection")
                z = result.z_ols.coefficients
                _validate_complex(z, "real encoded OLS")
                power = (z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()).sum(dim=0).cpu()
                total += power
                count += z.shape[0]
    finally:
        consumer.runtime.train(was_training)
    if count != images.shape[0]:
        raise TrainingContractError("scale extraction row count changed")
    values = total / float(count)
    return FrozenPhysicalOLSScale(values, checkpoint_identity, data_identity, data_role=data_role, classification="SCIENTIFIC_TRAIN_ONLY" if data_role == "train" else SCALE_SMOKE_CLASSIFICATION, basis_id=basis_id, mode_geometry=active_geometry)


def normalized_ols(z: torch.Tensor, scale: FrozenPhysicalOLSScale) -> torch.Tensor:
    """Return the derived normalized complex coordinate without changing raw z."""

    _validate_complex(z, "physical OLS")
    if scale.values.requires_grad:
        raise TrainingContractError("frozen physical scale unexpectedly requires gradients")
    denominator = torch.sqrt(scale.values.to(device=z.device, dtype=z.real.dtype) + EPSILON_SCALE)
    normalized = z / denominator.unsqueeze(0)
    _validate_complex(normalized, "normalized OLS")
    return normalized


def physical_ols_from_normalized(z_tilde: torch.Tensor, scale: FrozenPhysicalOLSScale) -> torch.Tensor:
    """Map a normalized prior sample back to physical OLS coefficients."""

    _validate_complex(z_tilde, "normalized OLS sample")
    denominator = torch.sqrt(scale.values.to(device=z_tilde.device, dtype=z_tilde.real.dtype) + EPSILON_SCALE)
    physical = z_tilde * denominator.unsqueeze(0)
    _validate_complex(physical, "physical OLS sample")
    return physical


def sample_physical_prior(
    scale: FrozenPhysicalOLSScale,
    sample_count: int,
    *,
    stage_local_step: int = 0,
    device: torch.device | str = "cpu",
    role: str = "future_generation",
) -> tuple[ComplexOLSState, NormalizedPriorSample]:
    prior = sample_normalized_prior(
        sample_count,
        stage="FUTURE_GENERATION",
        stage_local_step=stage_local_step,
        device=device,
        role=role,
    )
    physical = physical_ols_from_normalized(prior.coefficients, scale)
    return ComplexOLSState.from_external(
        physical,
        basis_id=scale.basis_id,
        mode_geometry=scale.mode_geometry,
    ), prior


def normalized_gauge_representation(z_tilde: torch.Tensor) -> torch.Tensor:
    """Use the frozen smooth global-phase quotient on normalized z only."""

    _validate_complex(z_tilde, "normalized OLS for gauge")
    anchor = z_tilde.sum(dim=1, keepdim=True) / math.sqrt(float(OLS_PORT_COUNT))
    factor = anchor.conj() / torch.sqrt(anchor.abs().square() + GAUGE_EPSILON**2)
    gauged = z_tilde * factor
    representation = torch.cat((gauged.real, gauged.imag), dim=1)
    if representation.shape[1] != LATENT_DIMENSION:
        raise TrainingContractError("normalized gauge representation dimension changed")
    _finite(representation, "normalized gauge representation")
    return representation


def _gaussian_kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    squared = (left[:, None, :] - right[None, :, :]).square().sum(dim=-1)
    return torch.exp(-squared / (2.0 * MMD_BANDWIDTH_SQUARED))


def normalized_full_complex_mmd_vstat(encoded: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Frozen biased B² V-statistic on normalized full-complex coordinates."""

    _validate_complex(encoded, "encoded normalized OLS", batch_size=LOGICAL_BATCH_SIZE)
    _validate_complex(prior, "normalized prior", batch_size=LOGICAL_BATCH_SIZE)
    x = normalized_gauge_representation(encoded)
    y = normalized_gauge_representation(prior)
    value = _gaussian_kernel(x, x).mean() + _gaussian_kernel(y, y).mean() - 2.0 * _gaussian_kernel(x, y).mean()
    _finite(value, "normalized full-complex MMD")
    return value


def physical_scale_anchor_loss(z: torch.Tensor, scale: FrozenPhysicalOLSScale) -> torch.Tensor:
    """Freeze the one permitted raw-physical-power anchor."""

    _validate_complex(z, "physical OLS for scale anchor")
    batch_power = (z.real.square() + z.imag.square()).mean(dim=0)
    target = scale.values.to(device=z.device, dtype=batch_power.dtype)
    ratio = (batch_power + EPSILON_SCALE) / (target + EPSILON_SCALE)
    value = torch.log(ratio).square().mean()
    _finite(value, "physical scale anchor")
    return value


@dataclass(frozen=True)
class V2LossBreakdown:
    stage: str
    image: torch.Tensor
    latent: torch.Tensor
    scale: torch.Tensor
    total: torch.Tensor
    result: LightGen2DForwardResult
    prior: NormalizedPriorSample | None
    normalized: torch.Tensor | None

    @property
    def optical(self) -> torch.Tensor:
        return self.scale

    def metrics(self) -> dict[str, float]:
        return {
            "L_image": float(self.image.detach().cpu()),
            "L_latent_normalized": float(self.latent.detach().cpu()),
            "L_scale": float(self.scale.detach().cpu()),
            "L_optical": float(self.scale.detach().cpu()),
            "L_total": float(self.total.detach().cpu()),
        }


@dataclass(frozen=True)
class V2AutogradResult:
    modal_projection_active: bool
    w_ols_active: bool
    trainable_gradients_finite: bool
    forbidden_gradients_absent: bool
    stage_a_no_mmd_gradient: bool
    stage_b_normalized_mmd_active: bool
    scale_gradient_active: bool
    frozen_scale_no_grad: bool
    gradient_norms: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modal_projection_active": self.modal_projection_active,
            "w_ols_active": self.w_ols_active,
            "trainable_gradients_finite": self.trainable_gradients_finite,
            "forbidden_gradients_absent": self.forbidden_gradients_absent,
            "stage_a_no_mmd_gradient": self.stage_a_no_mmd_gradient,
            "stage_b_normalized_mmd_active": self.stage_b_normalized_mmd_active,
            "scale_gradient_active": self.scale_gradient_active,
            "frozen_scale_no_grad": self.frozen_scale_no_grad,
            "gradient_norms": dict(self.gradient_norms),
        }


@dataclass(frozen=True)
class V2UpdateResult:
    stage: str
    global_step: int
    stage_local_step: int
    loss: V2LossBreakdown
    autograd: V2AutogradResult
    parameter_update_norms: Mapping[str, float]
    frozen_state_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "global_step": self.global_step,
            "stage_local_step": self.stage_local_step,
            "loss": self.loss.metrics(),
            "autograd": self.autograd.to_dict(),
            "parameter_update_norms": dict(self.parameter_update_norms),
            "frozen_state_unchanged": self.frozen_state_unchanged,
        }


class LightGen2DComplexOLSTrainingConsumerV2(nn.Module):
    """V2 loss/update consumer with an explicit trainable and optical firewall."""

    def __init__(
        self,
        contract: FrozenV2TrainingContract,
        *,
        device: torch.device | str = "cpu",
        propagation_grid_size: int = PRODUCTION_PROPAGATION_GRID,
        ols_model: str = LP01_OLS_MODEL,
    ) -> None:
        super().__init__()
        if not isinstance(contract, FrozenV2TrainingContract):
            raise TypeError("V2 consumer requires a validated FrozenV2TrainingContract")
        if propagation_grid_size not in {SMOKE_PROPAGATION_GRID, PRODUCTION_PROPAGATION_GRID}:
            raise ValueError("only the declared smoke or production propagation grid is supported")
        self.contract = contract
        self.device = torch.device(device)
        self.runtime = LightGen2DSemanticGenerationInputOLSAlignment(
            LightGen2DAlignmentRuntimeConfig(propagation_grid_size=propagation_grid_size, ols_model=ols_model)
        ).to(self.device)
        self.runtime.train()
        self._enforce_parameter_firewall()
        self._frozen_scale: FrozenPhysicalOLSScale | None = None

    @property
    def runtime_identity(self) -> dict[str, Any]:
        identity = {
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
        if self.runtime.ols.basis_id == GAUSSIAN_OLS_MODEL:
            identity["ols_model"] = GAUSSIAN_OLS_MODEL
            identity["mode_geometry"] = self.runtime.ols.mode_identity()
        return identity

    def _named_runtime_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(self.runtime.named_parameters())

    def _contract_trainable(self) -> Mapping[str, Any]:
        return _mapping(self.contract.payload.get("trainable_parameter_contract"), "contract trainable surface")

    def _enforce_parameter_firewall(self) -> None:
        named = self._named_runtime_parameters()
        names = [name for name, _ in named]
        _require(names == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), f"runtime parameter surface changed: {names}")
        _require(len({id(parameter) for _, parameter in named}) == len(named), "runtime parameter objects are duplicated")
        contract = self._contract_trainable()
        contract_shapes = {str(name): list(shape) for name, shape in _mapping(contract.get("parameter_shapes"), "contract shapes").items()}
        actual_shapes = {name: list(parameter.shape) for name, parameter in named}
        _require(actual_shapes == contract_shapes, "runtime parameter shapes differ from the V2 contract")
        _require(sum(parameter.numel() for _, parameter in named) == int(contract.get("parameter_count")), "runtime parameter count differs from V2 contract")
        for name, parameter in named:
            parameter.requires_grad_(name in ALLOWED_TRAINABLE_PARAMETER_NAMES)
            _require(parameter.dtype == torch.float32, f"runtime parameter dtype changed: {name}")
        trainable = [name for name, parameter in named if parameter.requires_grad]
        _require(trainable == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "runtime requires_grad firewall changed")
        _require(torch.equal(self.runtime.ols.transport.cpu(), torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "OLS transport is not T=I_100")

    def parameter_manifest(self) -> dict[str, Any]:
        self._enforce_parameter_firewall()
        named = self._named_runtime_parameters()
        state = self.runtime.state_dict()
        frozen = {name: value for name, value in state.items() if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES}
        return {
            "parameter_names": [name for name, _ in named],
            "trainable_parameter_names": [name for name, parameter in named if parameter.requires_grad],
            "parameter_shapes": {name: list(parameter.shape) for name, parameter in named},
            "parameter_dtypes": {name: str(parameter.dtype) for name, parameter in named},
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
            "total_parameter_count": sum(parameter.numel() for _, parameter in named),
            "frozen_state_tensor_count": len(frozen),
            "frozen_state_numel": sum(value.numel() for value in frozen.values() if torch.is_tensor(value)),
            "optimizer_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
        }

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self._named_runtime_parameters() if name in ALLOWED_TRAINABLE_PARAMETER_NAMES}

    def load_trainable_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(ALLOWED_TRAINABLE_PARAMETER_NAMES):
            raise TrainingContractError("V2 checkpoint trainable state keys changed")
        parameters = dict(self._named_runtime_parameters())
        with torch.no_grad():
            for name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
                value = state[name]
                _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(parameters[name].shape) and value.dtype == parameters[name].dtype, f"invalid V2 parameter state: {name}")
                _finite(value, f"checkpoint parameter {name}")
                parameters[name].copy_(value.to(self.device))
        self._enforce_parameter_firewall()

    def create_optimizer(self) -> torch.optim.Adam:
        self._enforce_parameter_firewall()
        parameters = [parameter for name, parameter in self._named_runtime_parameters() if name in ALLOWED_TRAINABLE_PARAMETER_NAMES]
        _require(len({id(parameter) for parameter in parameters}) == len(parameters), "optimizer parameter objects are duplicated")
        optimizer = torch.optim.Adam(
            parameters,
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPSILON,
            weight_decay=WEIGHT_DECAY,
            amsgrad=AMSGrad,
        )
        self.optimizer_manifest(optimizer)
        return optimizer

    def optimizer_manifest(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
            raise TrainingContractError("V2 requires one Adam parameter group")
        expected = [parameter for name, parameter in self._named_runtime_parameters() if name in ALLOWED_TRAINABLE_PARAMETER_NAMES]
        actual = list(optimizer.param_groups[0]["params"])
        _require(len(actual) == len(expected) and all(a is b for a, b in zip(actual, expected, strict=True)), "optimizer parameters differ from V2 trainable manifest")
        defaults = optimizer.defaults
        _require(defaults.get("lr") == LEARNING_RATE and tuple(defaults.get("betas", ())) == ADAM_BETAS and defaults.get("eps") == ADAM_EPSILON and defaults.get("weight_decay") == WEIGHT_DECAY and defaults.get("amsgrad") is AMSGrad, "optimizer hyperparameters differ from V2 contract")
        return {
            "name": "Adam",
            "parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "betas": list(optimizer.param_groups[0]["betas"]),
            "eps": optimizer.param_groups[0]["eps"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "amsgrad": optimizer.param_groups[0]["amsgrad"],
        }

    def reset_stage_b_optimizer(self) -> tuple[torch.optim.Adam, dict[str, Any]]:
        """Discard Stage-A moments while retaining the selected parameters."""

        optimizer = self.create_optimizer()
        return optimizer, {
            "optimizer_reset": True,
            "stage_a_optimizer_moments_restored": False,
            "new_optimizer_state_empty": len(optimizer.state) == 0,
            "source_stage": "STAGE_A_SELECT",
            "target_stage": STAGE_B_TRAIN,
        }

    def set_scale(self, scale: FrozenPhysicalOLSScale) -> None:
        if scale.data_role not in {"train", SCALE_SMOKE_CLASSIFICATION}:
            raise DataRoleError("V2 scale may only originate from train or explicit smoke")
        if scale.values.requires_grad:
            raise TrainingContractError("cannot bind a differentiable physical scale")
        if scale.basis_id != self.runtime.ols.basis_id or (scale.basis_id == GAUSSIAN_OLS_MODEL and dict(scale.mode_geometry or {}) != self.runtime.ols.mode_identity()):
            raise TrainingContractError("physical scale modal basis/geometry differs from the active OLS runtime")
        self._frozen_scale = scale

    @property
    def frozen_scale(self) -> FrozenPhysicalOLSScale | None:
        return self._frozen_scale

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("V2 training image batch requires shape [10,1,28,28]")
        if images.shape[0] != LOGICAL_BATCH_SIZE or images.dtype != torch.float32 or torch.is_complex(images):
            raise TypeError("V2 training images must be ten float32 real images")
        if not torch.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("V2 training images must be finite and in [0,1]")

    def forward_loss(
        self,
        images: torch.Tensor,
        *,
        stage: str,
        stage_local_step: int,
        scale: FrozenPhysicalOLSScale | None = None,
    ) -> V2LossBreakdown:
        self._validate_images(images)
        if images.device != self.device:
            raise ValueError("V2 training images must already be on the consumer device")
        if stage not in {"A", "B"} or stage_local_step < 0:
            raise ValueError("stage or stage-local step is invalid")
        if stage == "B" and scale is None:
            scale = self._frozen_scale
        if stage == "B" and scale is None:
            raise TrainingContractError("Stage B requires a frozen train-only physical scale")
        if scale is not None and scale.values.requires_grad:
            raise TrainingContractError("frozen physical scale requires no gradients")

        result = self.runtime.forward_from_image(images)
        if result.image_input is None or result.ols_input_field is None or result.z_ols.source != "real_encoded":
            raise TrainingContractError("V2 image path did not expose the mandatory real OLS projection")
        raw = result.raw_intensity
        target = result.image_input.slm_intensity
        if raw.shape != target.shape or tuple(raw.shape[-2:]) != (200, 200):
            raise TrainingContractError("V2 raw CMOS and image target shape mismatch")
        _finite(raw, "V2 raw CMOS")
        _finite(target, "V2 image target")
        image_loss = (raw - target).square().mean()
        _finite(image_loss, "V2 image loss")

        if stage == "A":
            latent_loss = raw.new_zeros(())
            scale_loss = raw.new_zeros(())
            total = image_loss
            prior = None
            normalized = None
        else:
            assert scale is not None
            normalized = normalized_ols(result.z_ols.coefficients, scale)
            prior = sample_normalized_prior(
                LOGICAL_BATCH_SIZE,
                stage="STAGE_B",
                stage_local_step=stage_local_step,
                device=self.device,
                role="training_target",
            )
            latent_loss = normalized_full_complex_mmd_vstat(normalized, prior.coefficients)
            scale_loss = physical_scale_anchor_loss(result.z_ols.coefficients, scale)
            total = image_loss + latent_loss + scale_loss
        for name, value in (("L_image", image_loss), ("L_latent_normalized", latent_loss), ("L_scale", scale_loss), ("L_total", total)):
            _finite(value, name)
        return V2LossBreakdown(stage, image_loss, latent_loss, scale_loss, total, result, prior, normalized)

    def _frozen_state_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.runtime.state_dict().items()
            if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES
        }

    @staticmethod
    def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        if left.keys() != right.keys():
            return False
        for key in left:
            first, second = left[key], right[key]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                if not torch.equal(first, second):
                    return False
            elif isinstance(first, Mapping) and isinstance(second, Mapping):
                if not LightGen2DComplexOLSTrainingConsumerV2._state_equal(first, second):
                    return False
            elif isinstance(first, list) and isinstance(second, list):
                if len(first) != len(second) or any(first_item != second_item for first_item, second_item in zip(first, second, strict=True)):
                    return False
            elif first != second:
                return False
        return True

    def _retain_probes(self, loss: V2LossBreakdown) -> None:
        result = loss.result
        tensors = [
            result.ols_input_field.tensor if result.ols_input_field is not None else None,
            result.z_ols.coefficients,
            result.generator_input_field.tensor,
        ]
        if loss.normalized is not None:
            tensors.append(loss.normalized)
        for tensor in tensors:
            if tensor is None or not tensor.requires_grad:
                raise TrainingContractError("V2 modal/W_OLS tensor is detached from autograd")
            tensor.retain_grad()

    def _validate_autograd(
        self,
        loss: V2LossBreakdown,
        *,
        stage_a_scale_grads: Sequence[torch.Tensor | None],
        stage_b_mmd_grads: Sequence[torch.Tensor | None],
    ) -> V2AutogradResult:
        gradient_norms: dict[str, float] = {}
        for name, parameter in self._named_runtime_parameters():
            if name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
                if parameter.grad is None:
                    raise TrainingContractError(f"V2 trainable parameter has no gradient: {name}")
                _finite(parameter.grad, f"V2 gradient {name}")
                gradient_norms[name] = float(parameter.grad.detach().square().sum().sqrt().cpu())
            elif parameter.grad is not None:
                raise TrainingContractError(f"forbidden V2 parameter received a gradient: {name}")
        result = loss.result
        z_gradient = result.z_ols.coefficients.grad
        ols_input_gradient = result.ols_input_field.tensor.grad if result.ols_input_field is not None else None
        writer_gradient = result.generator_input_field.tensor.grad
        if z_gradient is None or ols_input_gradient is None or writer_gradient is None:
            raise TrainingContractError("V2 modal projection or W_OLS is absent from autograd")
        _finite(z_gradient, "V2 z gradient")
        _finite(ols_input_gradient, "V2 OLS input gradient")
        _finite(writer_gradient, "V2 W_OLS gradient")
        projection_active = result.z_ols.coefficients.grad_fn is not None and result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True
        writer_active = result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission" and result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space"
        _require(projection_active and writer_active, "V2 modal/W_OLS structural proof failed")
        scale_active = loss.stage == "B" and any(gradient is not None and torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0.0 for gradient in stage_a_scale_grads)
        mmd_active = loss.stage == "B" and any(gradient is not None and torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0.0 for gradient in stage_b_mmd_grads)
        return V2AutogradResult(
            projection_active,
            writer_active,
            True,
            True,
            loss.stage == "A" and loss.prior is None and loss.latent.requires_grad is False,
            mmd_active,
            scale_active,
            self._frozen_scale is None or not self._frozen_scale.values.requires_grad,
            gradient_norms,
        )

    def update(
        self,
        images: torch.Tensor,
        *,
        stage: str,
        global_step: int,
        stage_local_step: int,
        optimizer: torch.optim.Optimizer,
        scale: FrozenPhysicalOLSScale | None = None,
    ) -> V2UpdateResult:
        if optimizer is None:
            raise ValueError("V2 optimizer is required")
        if global_step < 0 or stage_local_step < 0:
            raise ValueError("V2 step coordinates are invalid")
        self._enforce_parameter_firewall()
        self.optimizer_manifest(optimizer)
        before_parameters = {name: parameter.detach().clone() for name, parameter in self._named_runtime_parameters()}
        before_frozen = self._frozen_state_snapshot()
        optimizer.zero_grad(set_to_none=True)
        loss = self.forward_loss(images, stage=stage, stage_local_step=stage_local_step, scale=scale)
        self._retain_probes(loss)
        parameters = [parameter for _, parameter in self._named_runtime_parameters() if parameter.requires_grad]
        scale_grads: Sequence[torch.Tensor | None] = ()
        mmd_grads: Sequence[torch.Tensor | None] = ()
        if loss.stage == "B":
            scale_grads = torch.autograd.grad(loss.scale, parameters, retain_graph=True, allow_unused=True)
            mmd_grads = torch.autograd.grad(loss.latent, parameters, retain_graph=True, allow_unused=True)
        loss.total.backward()
        autograd = self._validate_autograd(loss, stage_a_scale_grads=scale_grads, stage_b_mmd_grads=mmd_grads)
        optimizer.step()
        after_parameters = {name: parameter.detach().clone() for name, parameter in self._named_runtime_parameters()}
        for name, parameter in self._named_runtime_parameters():
            _finite(parameter, f"updated V2 parameter {name}")
        update_norms = {
            name: float((after_parameters[name] - before_parameters[name]).abs().square().sum().sqrt().cpu())
            for name in before_parameters
        }
        if not any(value > 0.0 for name, value in update_norms.items() if name in ALLOWED_TRAINABLE_PARAMETER_NAMES):
            raise TrainingContractError("V2 optimizer step did not change a trainable parameter")
        frozen_unchanged = self._state_equal(before_frozen, self._frozen_state_snapshot())
        if not frozen_unchanged:
            raise TrainingContractError("V2 frozen runtime state changed")
        return V2UpdateResult(stage, global_step, stage_local_step, loss, autograd, update_norms, frozen_unchanged)


class DeterministicTrainRoleSampler:
    """One-row-per-class train sampler for each frozen stage-local schedule."""

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int = SCHEDULE_SEED, steps: int = STAGE_A_STEPS) -> None:
        if train_indices.ndim != 1 or labels.ndim != 1 or train_indices.dtype != torch.long or labels.dtype != torch.long:
            raise TypeError("V2 sampler requires long indices and labels")
        if train_indices.numel() != 55_000 or labels.numel() != 60_000:
            raise ValueError("V2 sampler requires the frozen 55,000/60,000 roles")
        if seed != SCHEDULE_SEED or steps not in {STAGE_A_STEPS, STAGE_B_STEPS}:
            raise ValueError("V2 sampler seed/steps differ from the contract")
        self.seed = int(seed)
        self.steps = int(steps)
        self._cursor = 0
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._permutations: list[torch.Tensor] = []
        for digit in range(10):
            candidates = train_indices[labels.index_select(0, train_indices) == digit].cpu()
            if candidates.numel() < steps:
                raise ValueError("train role cannot support the V2 stage budget")
            self._permutations.append(candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator)))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cursor >= self.steps:
            raise StopIteration("V2 train sampler is exhausted")
        indices = torch.stack([permutation[self._cursor] for permutation in self._permutations])
        labels = torch.arange(10, dtype=torch.long)
        self._cursor += 1
        return indices, labels

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "lightgen_v2_train_role_sampler.v2",
            "seed": self.seed,
            "steps": self.steps,
            "cursor": self._cursor,
            "class_count": 10,
            "rows_per_class": [int(value.numel()) for value in self._permutations],
            "permutations": [value.clone() for value in self._permutations],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(state.get("schema") == "lightgen_v2_train_role_sampler.v2" and state.get("seed") == self.seed and state.get("steps") == self.steps, "V2 sampler identity changed")
        permutations = state.get("permutations")
        _require(isinstance(permutations, list) and len(permutations) == 10, "V2 sampler permutations are invalid")
        _require(state.get("rows_per_class") == [int(value.numel()) for value in self._permutations], "V2 sampler row counts changed")
        cursor = state.get("cursor")
        _require(isinstance(cursor, int) and 0 <= cursor <= self.steps, "V2 sampler cursor is invalid")
        for index, value in enumerate(permutations):
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.long and value.shape == self._permutations[index].shape, "V2 sampler permutation shape changed")
        self._permutations = [value.cpu().clone() for value in permutations]
        self._cursor = cursor


@dataclass
class V2TrainingStateMachine:
    """Fail-closed state machine for the two-stage protocol."""

    stage: str = STAGE_A_TRAIN
    global_step: int = 0
    stage_local_step: int = 0
    selected_stage_a_checkpoint: Mapping[str, Any] | None = None
    scale_identity: Mapping[str, Any] | None = None
    terminal_classification: str | None = None
    optimizer_reset_marker: Mapping[str, Any] | None = None

    _transitions = {
        STAGE_A_TRAIN: {STAGE_A_SEMANTIC_CHECK},
        STAGE_A_SEMANTIC_CHECK: {STAGE_A_TRAIN, STAGE_A_SELECT},
        STAGE_A_SELECT: {SCALE_EXTRACT},
        SCALE_EXTRACT: {SCALE_FREEZE},
        SCALE_FREEZE: {STAGE_B_TRAIN},
        STAGE_B_TRAIN: {STAGE_B_SEMANTIC_CHECK},
        STAGE_B_SEMANTIC_CHECK: {STAGE_B_TRAIN, FINAL_SELECT},
        FINAL_SELECT: {COMPLETE},
        COMPLETE: set(),
    }

    def transition(self, target: str) -> None:
        if self.terminal_classification is not None:
            raise StateMachineError("V2 state machine is terminal")
        if target not in self._transitions.get(self.stage, set()):
            raise StateMachineError(f"invalid V2 transition {self.stage} -> {target}")
        self.stage = target

    def stage_a_checkpoint_checked(self, *, final_checkpoint: bool, eligible: bool) -> None:
        _require(self.stage == STAGE_A_SEMANTIC_CHECK, "Stage-A semantic check is not active")
        if final_checkpoint and not eligible:
            self.terminal_classification = STAGE_A_FAILURE
            return
        if final_checkpoint:
            self.transition(STAGE_A_SELECT)
        else:
            self.transition(STAGE_A_TRAIN)

    def stage_b_checkpoint_checked(self, *, final_checkpoint: bool, eligible: bool) -> None:
        _require(self.stage == STAGE_B_SEMANTIC_CHECK, "Stage-B semantic check is not active")
        if final_checkpoint and not eligible:
            self.terminal_classification = STAGE_B_FAILURE
            return
        if final_checkpoint:
            self.transition(FINAL_SELECT)
        else:
            self.transition(STAGE_B_TRAIN)

    def fail_stage_a(self) -> None:
        _require(self.stage == STAGE_A_SEMANTIC_CHECK, "Stage-A failure can only occur at semantic check")
        self.terminal_classification = STAGE_A_FAILURE

    def fail_stage_b(self) -> None:
        _require(self.stage == STAGE_B_SEMANTIC_CHECK, "Stage-B failure can only occur at semantic check")
        self.terminal_classification = STAGE_B_FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lightgen_v2_training_state.v2",
            "stage": self.stage,
            "global_step": self.global_step,
            "stage_local_step": self.stage_local_step,
            "selected_stage_a_checkpoint": dict(self.selected_stage_a_checkpoint) if self.selected_stage_a_checkpoint is not None else None,
            "scale_identity": dict(self.scale_identity) if self.scale_identity is not None else None,
            "terminal_classification": self.terminal_classification,
            "optimizer_reset_marker": dict(self.optimizer_reset_marker) if self.optimizer_reset_marker is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V2TrainingStateMachine":
        _require(value.get("schema") == "lightgen_v2_training_state.v2", "V2 state schema mismatch")
        stage = str(value.get("stage"))
        _require(stage in cls._transitions, "V2 state stage is invalid")
        return cls(
            stage=stage,
            global_step=int(value.get("global_step")),
            stage_local_step=int(value.get("stage_local_step")),
            selected_stage_a_checkpoint=value.get("selected_stage_a_checkpoint"),
            scale_identity=value.get("scale_identity"),
            terminal_classification=value.get("terminal_classification"),
            optimizer_reset_marker=value.get("optimizer_reset_marker"),
        )


def select_stage_a_checkpoint(records: Sequence[Mapping[str, Any]], *, threshold: float = SEMANTIC_THRESHOLD) -> Mapping[str, Any] | None:
    if threshold != SEMANTIC_THRESHOLD:
        raise ValueError("semantic threshold is frozen at 0.937")
    by_step = {int(record.get("stage_local_step")): record for record in records}
    if set(by_step) != set(CHECKPOINT_STEPS):
        raise StateMachineError("Stage-A checkpoint cadence is incomplete or changed")
    eligible = [record for record in by_step.values() if float(record.get("development_macro_recall")) >= threshold]
    if not eligible:
        return None
    return min(eligible, key=lambda record: (float(record["development_L_image"]), int(record["stage_local_step"])))


def select_stage_b_checkpoint(records: Sequence[Mapping[str, Any]], *, threshold: float = SEMANTIC_THRESHOLD) -> Mapping[str, Any] | None:
    if threshold != SEMANTIC_THRESHOLD:
        raise ValueError("semantic threshold is frozen at 0.937")
    by_step = {int(record.get("stage_local_step")): record for record in records}
    if set(by_step) != set(CHECKPOINT_STEPS):
        raise StateMachineError("Stage-B checkpoint cadence is incomplete or changed")
    eligible = [record for record in by_step.values() if float(record.get("development_macro_recall")) >= threshold]
    if not eligible:
        return None
    return min(eligible, key=lambda record: (float(record["development_L_total"]), int(record["stage_local_step"])))


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
    _require(isinstance(cpu, torch.Tensor) and isinstance(cuda, list), "V2 RNG payload is malformed")
    torch.random.set_rng_state(cpu.cpu())
    if cuda:
        _require(torch.cuda.is_available(), "V2 checkpoint contains CUDA RNG state but CUDA is unavailable")
        _require(all(isinstance(state, torch.Tensor) for state in cuda), "V2 CUDA RNG state is malformed")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])


def _parameter_state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def checkpoint_payload(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Optimizer,
    *,
    state: V2TrainingStateMachine,
    sampler_state: Mapping[str, Any],
    semantic_selection_state: Mapping[str, Any],
    git: Mapping[str, Any],
    scale: FrozenPhysicalOLSScale | None = None,
    selected_stage_a_checkpoint: Mapping[str, Any] | None = None,
    optimizer_reset_marker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    consumer.optimizer_manifest(optimizer)
    stage_b = state.stage in {STAGE_B_TRAIN, STAGE_B_SEMANTIC_CHECK, FINAL_SELECT, COMPLETE}
    if stage_b:
        _require(scale is not None, "Stage-B checkpoint requires a frozen scale")
        _require(scale.artifact_sha256 is not None and scale.artifact_path is not None, "Stage-B checkpoint requires a published scale artifact identity")
        _require(scale.basis_id == consumer.runtime.ols.basis_id, "Stage-B checkpoint scale basis differs from the active OLS runtime")
        if scale.basis_id == GAUSSIAN_OLS_MODEL:
            _require(dict(scale.mode_geometry or {}) == consumer.runtime.ols.mode_identity(), "Stage-B checkpoint scale geometry differs from the active OLS runtime")
        _require(optimizer_reset_marker is not None and optimizer_reset_marker.get("optimizer_reset") is True, "Stage-B checkpoint lacks optimizer-reset marker")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 2,
        "accepted_training_checkpoint": False,
        "contract": consumer.contract.identity(),
        "runtime": consumer.runtime_identity,
        "stage": state.stage,
        "global_step": int(state.global_step),
        "stage_local_step": int(state.stage_local_step),
        "trainable_parameter_state": consumer.trainable_state_dict(),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "optimizer_configuration": consumer.optimizer_manifest(optimizer),
        "optimizer_reset_marker": dict(optimizer_reset_marker) if optimizer_reset_marker is not None else None,
        "rng_state": _rng_state_payload(),
        "sampler_state": copy.deepcopy(dict(sampler_state)),
        "semantic_evaluator": {
            "identity": "FrozenMNISTEvaluator v1",
            "preprocessing": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
            "threshold": SEMANTIC_THRESHOLD,
            "gradient_enabled": False,
        },
        "checkpoint_selection_state": copy.deepcopy(dict(semantic_selection_state)),
        "selected_stage_a_checkpoint": dict(selected_stage_a_checkpoint or state.selected_stage_a_checkpoint or {}) or None,
        "scale_artifact": {
            "schema": SCALE_SCHEMA,
            "path": scale.artifact_path,
            "sha256": scale.artifact_sha256,
            "vector_sha256": scale.vector_sha256,
            "epsilon_scale": EPSILON_SCALE,
            "data_role": scale.data_role,
            **({"basis_id": scale.basis_id, "mode_geometry": dict(scale.mode_geometry or {})} if scale.basis_id == GAUSSIAN_OLS_MODEL else {}),
        } if scale is not None else None,
        "scale_vector": scale.values.detach().cpu().clone() if scale is not None else None,
        "normalized_prior": {
            "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NORMALIZED_OLS_V2",
            "namespace": PRIOR_NAMESPACE,
            "formula": "(epsilon_R+i*epsilon_I)/sqrt(2)",
        },
        "data_roles": {
            "train_used": True,
            "development_used": True,
            "calibration_used": False,
            "protected_used": False,
            "test_used": False,
            "sealed_used": False,
        },
        "git": dict(git),
    }


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise TrainingContractError("cannot save a non-V2 checkpoint payload")
    atomic_torch_save_new(path, dict(payload), artifact_kind="LightGen V2 training checkpoint")


@dataclass(frozen=True)
class LoadedV2Checkpoint:
    stage: str
    global_step: int
    stage_local_step: int
    sampler_state: Mapping[str, Any]
    checkpoint_selection_state: Mapping[str, Any]
    selected_stage_a_checkpoint: Mapping[str, Any] | None
    scale: FrozenPhysicalOLSScale | None
    optimizer_reset_marker: Mapping[str, Any] | None


def load_checkpoint(
    path: str | Path,
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Optimizer,
    *,
    expected_scale: FrozenPhysicalOLSScale | None = None,
) -> LoadedV2Checkpoint:
    payload = _mapping(safe_torch_load(path, map_location="cpu"), "V2 checkpoint")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("version") == 2, "V2 checkpoint schema mismatch")
    contract = _mapping(payload.get("contract"), "V2 checkpoint contract")
    _require(contract.get("schema") == V2_CONTRACT_SCHEMA and contract.get("path") == consumer.contract.path.relative_to(_repo_root()).as_posix(), "V2 checkpoint contract identity mismatch")
    runtime = _mapping(payload.get("runtime"), "V2 checkpoint runtime")
    _require(runtime.get("identity") == ALIGNMENT_RUNTIME_ID and runtime.get("module") == ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix() and runtime.get("transport") == "T=I_100 coefficient space", "V2 checkpoint runtime identity mismatch")
    current_runtime = consumer.runtime_identity
    for key in ("ols_model", "mode_geometry"):
        _require(runtime.get(key) == current_runtime.get(key), f"V2 checkpoint OLS basis/geometry mismatch: {key}")
    stage = str(payload.get("stage"))
    _require(stage in V2TrainingStateMachine._transitions, "V2 checkpoint stage is invalid")
    scale: FrozenPhysicalOLSScale | None = None
    marker_value = payload.get("optimizer_reset_marker")
    stage_b = stage in {STAGE_B_TRAIN, STAGE_B_SEMANTIC_CHECK, FINAL_SELECT, COMPLETE}
    if stage_b:
        marker = _mapping(marker_value, "V2 optimizer reset marker")
        _require(marker.get("optimizer_reset") is True and marker.get("stage_a_optimizer_moments_restored") is False, "V2 Stage-B optimizer reset semantics changed")
        artifact = _mapping(payload.get("scale_artifact"), "V2 scale artifact binding")
        values = payload.get("scale_vector")
        artifact_path = artifact.get("path")
        _require(
            artifact.get("schema") == SCALE_SCHEMA
            and isinstance(values, torch.Tensor)
            and isinstance(artifact_path, str)
            and bool(artifact_path.strip()),
            "V2 Stage-B scale binding is incomplete",
        )
        scale = load_scale_artifact(artifact_path)
        _require(torch.equal(scale.values, values.detach().to(dtype=torch.float64, device="cpu")), "V2 checkpoint scale vector differs from artifact")
        _require(artifact.get("basis_id", LP01_OLS_MODEL) == scale.basis_id, "V2 checkpoint scale artifact basis differs from artifact")
        if scale.basis_id == GAUSSIAN_OLS_MODEL:
            _require(artifact.get("mode_geometry") == dict(scale.mode_geometry or {}), "V2 checkpoint scale artifact geometry differs from artifact")
        _require(scale.basis_id == consumer.runtime.ols.basis_id, "V2 checkpoint scale modal basis differs from the active runtime")
        if scale.basis_id == GAUSSIAN_OLS_MODEL:
            _require(dict(scale.mode_geometry or {}) == consumer.runtime.ols.mode_identity(), "V2 checkpoint scale geometry differs from the active runtime")
        if expected_scale is not None:
            _require(torch.equal(scale.values, expected_scale.values), "V2 checkpoint scale values differ from the expected scale")
            _require(scale.basis_id == expected_scale.basis_id, "V2 checkpoint scale basis differs from the expected scale")
            if scale.basis_id == GAUSSIAN_OLS_MODEL:
                _require(dict(scale.mode_geometry or {}) == dict(expected_scale.mode_geometry or {}), "V2 checkpoint scale geometry differs from the expected scale")

    # Reject modal and artifact mismatches before touching model, optimizer, RNG, or scale state.
    trainable_state = _mapping(payload.get("trainable_parameter_state"), "V2 trainable parameter state")
    optimizer_state = payload.get("optimizer_state")
    _require(isinstance(optimizer_state, Mapping), "V2 optimizer state is missing")
    rng_state = _mapping(payload.get("rng_state"), "V2 RNG state")
    sampler_state = _mapping(payload.get("sampler_state"), "V2 sampler state")
    selection_state = _mapping(payload.get("checkpoint_selection_state"), "V2 selection state")
    consumer.optimizer_manifest(optimizer)
    consumer.load_trainable_state_dict(trainable_state)
    optimizer.load_state_dict(dict(optimizer_state))
    consumer.optimizer_manifest(optimizer)
    restore_rng_state(rng_state)
    if scale is not None:
        consumer.set_scale(scale)
    return LoadedV2Checkpoint(
        stage=stage,
        global_step=int(payload.get("global_step")),
        stage_local_step=int(payload.get("stage_local_step")),
        sampler_state=sampler_state,
        checkpoint_selection_state=selection_state,
        selected_stage_a_checkpoint=payload.get("selected_stage_a_checkpoint"),
        scale=scale,
        optimizer_reset_marker=marker_value,
    )


def raw_cmos_evaluator_view(raw_cmos: torch.Tensor) -> torch.Tensor:
    if not isinstance(raw_cmos, torch.Tensor) or raw_cmos.ndim != 4 or raw_cmos.shape[1] != 1:
        raise ValueError("raw CMOS evaluator view requires [B,1,H,W]")
    if torch.is_complex(raw_cmos) or not torch.is_floating_point(raw_cmos):
        raise TypeError("raw CMOS evaluator view requires real floating intensity")
    _finite(raw_cmos, "raw CMOS evaluator input")
    return torch.nn.functional.adaptive_avg_pool2d(raw_cmos.clamp(0.0, 1.0), (28, 28))


@torch.no_grad()
def development_macro_recall(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    evaluator: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = LOGICAL_BATCH_SIZE,
) -> float:
    """Evaluate only a permitted development role; evaluator remains frozen."""

    if images.ndim != 4 or images.shape[1:] != (1, 28, 28) or labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("development evaluator inputs have incompatible shapes")
    if images.dtype != torch.float32 or labels.dtype != torch.long:
        raise TypeError("development evaluator inputs have incompatible dtypes")
    if batch_size <= 0:
        raise ValueError("development evaluator batch size must be positive")
    evaluator.eval()
    correct = torch.zeros(10, dtype=torch.float64)
    counts = torch.zeros(10, dtype=torch.float64)
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    try:
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(consumer.device)
            batch_labels = labels[start : start + batch_size].to(consumer.device)
            result = consumer.runtime.forward_from_image(batch)
            view = raw_cmos_evaluator_view(result.raw_intensity)
            _, logits = evaluator(view)
            predictions = logits.argmax(dim=1)
            for digit in range(10):
                mask = batch_labels == digit
                counts[digit] += mask.sum().to(torch.float64).cpu()
                correct[digit] += ((predictions == digit) & mask).sum().to(torch.float64).cpu()
    finally:
        consumer.runtime.train(was_training)
    if bool((counts == 0).any()):
        raise ValueError("development evaluator requires every class")
    value = (correct / counts).mean()
    _finite(value, "development macro recall")
    return float(value)


def validate_legacy_exclusion(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else Path(__file__)
    tree = ast.parse(target.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden_fragments = ("ldm_project.v1", "phase_producer", "q_producer", "generate_from_prior", "IdentityOLS")
    forbidden = [name for name in imports if any(fragment in name for fragment in forbidden_fragments)]
    if forbidden:
        raise TrainingContractError(f"legacy dependency entered V2 consumer: {forbidden}")
    return {"passed": True, "forbidden_imports": [], "imports": sorted(set(imports))}


def git_facts(repo_root: Path | None = None) -> dict[str, Any]:
    import subprocess

    root = _repo_root() if repo_root is None else repo_root.resolve()

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


__all__ = [
    "ALLOWED_TRAINABLE_PARAMETER_NAMES",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_STEPS",
    "CLASSIFICATION",
    "COMPLETE",
    "DETERMINISTIC_TOLERANCE",
    "DeterministicTrainRoleSampler",
    "DataRoleError",
    "EPSILON_SCALE",
    "FINAL_SELECT",
    "FrozenPhysicalOLSScale",
    "FrozenV2TrainingContract",
    "GAUGE_EPSILON",
    "LightGen2DComplexOLSTrainingConsumerV2",
    "LoadedV2Checkpoint",
    "LOGICAL_BATCH_SIZE",
    "MMD_BANDWIDTH_SQUARED",
    "NormalizedPriorSample",
    "OLS_PORT_COUNT",
    "PRIOR_NAMESPACE",
    "PRODUCTION_PROPAGATION_GRID",
    "SCALE_EXTRACT",
    "SCALE_FREEZE",
    "SCALE_SCHEMA",
    "SCALE_SMOKE_CLASSIFICATION",
    "SEMANTIC_THRESHOLD",
    "STAGE_A_CHECKPOINT_RULE",
    "STAGE_A_FAILURE",
    "STAGE_A_SELECT",
    "STAGE_A_SEMANTIC_CHECK",
    "STAGE_A_STEPS",
    "STAGE_A_TRAIN",
    "STAGE_B_CHECKPOINT_RULE",
    "STAGE_B_FAILURE",
    "STAGE_B_SEMANTIC_CHECK",
    "STAGE_B_STEPS",
    "STAGE_B_TRAIN",
    "StateMachineError",
    "TrainingContractError",
    "V2AutogradResult",
    "V2LossBreakdown",
    "V2TrainingStateMachine",
    "V2UpdateResult",
    "V2_CONTRACT_RELATIVE_PATH",
    "V2_CONTRACT_SCHEMA",
    "V2_CONTRACT_SHA256",
    "checkpoint_payload",
    "development_macro_recall",
    "derive_prior_seed",
    "extract_train_only_ols_scale",
    "file_sha256",
    "git_facts",
    "load_checkpoint",
    "load_scale_artifact",
    "load_v2_training_contract",
    "normalized_full_complex_mmd_vstat",
    "normalized_gauge_representation",
    "normalized_ols",
    "physical_ols_from_normalized",
    "physical_scale_anchor_loss",
    "raw_cmos_evaluator_view",
    "restore_rng_state",
    "sample_normalized_prior",
    "sample_physical_prior",
    "save_checkpoint",
    "save_scale_artifact",
    "select_stage_a_checkpoint",
    "select_stage_b_checkpoint",
    "validate_legacy_exclusion",
]
