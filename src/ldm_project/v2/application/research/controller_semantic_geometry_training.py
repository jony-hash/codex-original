"""Fail-closed controller-only S3 semantic-geometry repair harness.

This module contains the pre-registered, task-scoped intervention after the
S3 ``CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`` result.  It intentionally
stops at the frozen encoder-to-OLS R1 boundary: decoder, relay, CMOS, and the
frozen evaluator are not part of the training objective.

The module is also deliberately usable without CUDA.  The no-update
preflight and all contract tests operate on hashes, CPU tensors, and isolated
fixtures.  The authorized trainer is dependency-injected so that constructing
this module cannot silently load or mutate the registered optical controller.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.evidence.io import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.identity_generator import (
    IdentityAnchoredConditionalAmplitudeGenerator,
)


SCHEMA = "controller_semantic_geometry_training_harness.v1"
TARGET_BANK_SCHEMA = "controller_semantic_geometry_r1_target_bank.v1"
TARGET_BANK_MANIFEST_SCHEMA = "controller_semantic_geometry_r1_target_bank_manifest.v1"
PARTITION_SCHEMA = "controller_semantic_geometry_r1_partition.v1"
OBJECTIVE_FAMILY = "CLASS_CONDITIONED_R1_FIRST_SECOND_ORDER_GEOMETRY_ALIGNMENT"
AUTHORIZATION_TOKEN = "AUTHORIZE_ONE_SHOT_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_CANARY"
IMPLEMENTATION_READY = "IMPLEMENTATION_READY_FOR_ONE_SHOT_AUTHORIZATION"
PREFLIGHT_PASS = "PRETRAINING_HARNESS_PREFLIGHT_PASS"
INVALID_HARNESS = "invalid_harness"

CLASS_COUNT = 10
CODE_DIM = 8
R1_DIM = 100
SAMPLES_PER_CLASS = 50
REAL_COUNT = CLASS_COUNT * SAMPLES_PER_CLASS
HORIZON = 492
PHYSICAL_MICROBATCH_SIZE = 1

TRAINING_CODE_SEED = 1_239_212_442_827_321_870
TRAINING_CODE_SEED_MATERIAL_SHA256 = (
    "11329126aff7720ef6352006f9eb29627643c2fe1ff86819f0e8dc9a86efa277"
)
TRAINING_FIRST_BLOCK_SHA256 = (
    "a1fa3261b7f82756aaee4f3b62127bccee0bc2c0243f476b9956beeb4b92b0ce"
)
INTERNAL_VALIDATION_CODE_SEED = 100_000
S3_CODE_SEED = 1234
S3_BASE_CODE_SHA256 = (
    "2c83f8fca3d1fde7d5c694f69b602dbc3d43b4e41cfb8159d1af1914efef8c86"
)
PARTITION_IDENTITY = (
    "b07a04d3c779e5952c06209c9d8cc8421b8193cd41f037db88f95c736ab35594"
)
OPTIMIZER_CONTRACT_SHA256 = (
    "c83a0232a418ca9faa0e3d74d16d62af39ea67a21312b1255896c372b944b2f7"
)
OBJECTIVE_CONTRACT_SHA256 = (
    "bfc372e6803140c68b739656332b2d6f47ade06ba44044bf852ebead9baabfcb"
)

CONTROLLER_PARAMETER_NAMES = (
    "generator.class_carrier_logits",
    "generator.global_modes",
    "generator.code_projection.weight",
    "generator.class_heads",
)
_GENERATOR_PARAMETER_NAMES = tuple(name.removeprefix("generator.") for name in CONTROLLER_PARAMETER_NAMES)

S1_FIT_INDICES_SHA256 = (
    "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
)
S1_FIT_LABELS_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
S1_SELECTION_INDICES_SHA256 = (
    "11c076affa8511e883f595a0d1ee91e40f6d94747a351e21269520344d3f594f"
)
S1_SELECTION_LABELS_SHA256 = (
    "31fce460255bda46afc2bd6b6777793873e2e6ee33e23a681def3ab8af6693cb"
)
S1_CONFIRMATION_INDICES_SHA256 = (
    "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a"
)
S1_CONFIRMATION_LABELS_SHA256 = (
    "b05112caa3014d6c2081423b371361eefe1ec7a9f531e54db016b89bfbb77b18"
)
S2_VALIDATION_INDICES_SHA256 = (
    "5370b35f304a779cde7f978df6392b9974faaf4296e9edf75f9edf8202db02fa"
)
S2_VALIDATION_LABELS_SHA256 = (
    "13d90706eef49c3c9f17ccd39c002dc18a79eca1378e780548ea6b29adc6ec29"
)

_EXPECTED_TRAINING_LABELS = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(
    SAMPLES_PER_CLASS
)


class HarnessFailure(ValueError):
    """A fail-closed contract violation at the harness boundary."""


@dataclass(frozen=True)
class S1Reference:
    fit_r1: torch.Tensor
    fit_labels: torch.Tensor
    fit_indices: torch.Tensor
    selection_labels: torch.Tensor
    selection_indices: torch.Tensor
    confirmation_labels: torch.Tensor
    confirmation_indices: torch.Tensor
    source_artifact_sha256: str


@dataclass(frozen=True)
class TargetBank:
    training_r1: torch.Tensor
    training_labels: torch.Tensor
    training_indices: torch.Tensor
    validation_r1: torch.Tensor
    validation_labels: torch.Tensor
    validation_indices: torch.Tensor
    manifest: Mapping[str, Any]
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class LogicalGeometryBatch:
    real_r1: torch.Tensor
    real_labels: torch.Tensor
    controller_codes: torch.Tensor
    controller_labels: torch.Tensor
    base_codes: torch.Tensor
    physical_microbatch_size: int


@dataclass(frozen=True)
class GeometryObjectiveResult:
    loss: torch.Tensor
    anchor_terms: torch.Tensor
    covariance_terms: torch.Tensor
    class_terms: torch.Tensor
    mean_anchor: torch.Tensor
    mean_covariance: torch.Tensor


@dataclass(frozen=True)
class FrozenStateSnapshot:
    controller_state_hash: str
    frozen_model_state_hash: str
    frozen_parameter_hashes: Mapping[str, str]
    file_hashes: Mapping[str, str]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessFailure(f"{name} must be a mapping")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HarnessFailure(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HarnessFailure(f"{name} must be a lowercase SHA-256") from exc
    if value != value.lower():
        raise HarnessFailure(f"{name} must be a lowercase SHA-256")
    return value


def resolve_path(repo_root: str | Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate


def load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise HarnessFailure("harness configuration must be a mapping")
    validate_config(payload)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "governance",
        "data_firewall",
        "training_code",
        "optimizer",
        "batch",
        "objective",
        "horizon",
        "checkpoint",
        "optical_authority",
        "source",
        "output",
    }
    if set(config) != required:
        raise HarnessFailure("controller geometry harness config fields are invalid")
    if config["schema"] != SCHEMA:
        raise HarnessFailure("controller geometry harness schema is invalid")
    if config["task_id"] != "H-LOMA-GEN-S3-CONTROLLER-SEMANTIC-GEOMETRY-REPAIR":
        raise HarnessFailure("controller geometry harness task identity is invalid")

    governance = _mapping(config["governance"], "governance")
    if governance.get("training_authorized") is not False:
        raise HarnessFailure("training_authorized must remain false before explicit authorization")
    if governance.get("formal_promotion") is not False:
        raise HarnessFailure("formal_promotion must remain false")
    if governance.get("s3_classification") != "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED":
        raise HarnessFailure("S3 classification may not be changed by the repair harness")

    firewall = _mapping(config["data_firewall"], "data_firewall")
    if firewall.get("partition_identity") != PARTITION_IDENTITY:
        raise HarnessFailure("data-firewall partition identity is not frozen")
    if firewall.get("partition_manifest_sha256") != PARTITION_IDENTITY:
        raise HarnessFailure("data-firewall partition manifest identity is not frozen")
    if firewall.get("runtime_resplit") is not False or firewall.get("reselection") is not False:
        raise HarnessFailure("data firewall permits runtime reselection or resplitting")
    if firewall.get("training_count") != REAL_COUNT or firewall.get("validation_count") != REAL_COUNT:
        raise HarnessFailure("data-firewall counts are not 500/500")
    if firewall.get("samples_per_class") != SAMPLES_PER_CLASS:
        raise HarnessFailure("data-firewall class count is not 50")

    code = _mapping(config["training_code"], "training_code")
    exact_code = {
        "seed": TRAINING_CODE_SEED,
        "seed_material_sha256": TRAINING_CODE_SEED_MATERIAL_SHA256,
        "distribution": "uniform[-1,1]^8",
        "dtype": "torch.float32",
        "code_dim": CODE_DIM,
        "base_codes_per_logical_step": SAMPLES_PER_CLASS,
        "generator_device": "cpu",
        "initialization": "single_generator_manual_seed_once",
        "reseed": False,
        "skip": False,
        "replay": False,
        "fallback": False,
        "seed_sweep": False,
    }
    if dict(code) != exact_code:
        raise HarnessFailure("training-code contract differs from frozen authority")

    optimizer = _mapping(config["optimizer"], "optimizer")
    expected_optimizer = {
        "contract_sha256": OPTIMIZER_CONTRACT_SHA256,
        "family": "Adam",
        "parameter_names": list(CONTROLLER_PARAMETER_NAMES),
        "lr": 0.0002,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": True,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "state": "fresh",
        "scheduler": "none",
        "constant_lr": True,
        "no_sweep": True,
    }
    if dict(optimizer) != expected_optimizer:
        raise HarnessFailure("optimizer contract differs from frozen authority")

    batch = _mapping(config["batch"], "batch")
    expected_batch = {
        "real_samples_per_class": SAMPLES_PER_CLASS,
        "real_count": REAL_COUNT,
        "controller_base_code_count": SAMPLES_PER_CLASS,
        "controller_count": REAL_COUNT,
        "controller_layout": "label_major_row=label*50+code_index",
        "paired": False,
        "physical_microbatch_size": PHYSICAL_MICROBATCH_SIZE,
        "statistics_before_backward": True,
        "optimizer_steps_per_logical_batch": 1,
    }
    if dict(batch) != expected_batch:
        raise HarnessFailure("logical-batch contract differs from frozen authority")

    objective = _mapping(config["objective"], "objective")
    if objective.get("family") != OBJECTIVE_FAMILY:
        raise HarnessFailure("objective family is not the registered geometry-only family")
    if objective.get("contract_sha256") != OBJECTIVE_CONTRACT_SHA256:
        raise HarnessFailure("objective contract identity is stale")
    if objective.get("r1_observer") != "z_j=abs(c_j)^2/s_j":
        raise HarnessFailure("R1 observer identity is invalid")
    if objective.get("dimension") != R1_DIM or objective.get("samples_per_class") != SAMPLES_PER_CLASS:
        raise HarnessFailure("R1 objective dimensions are invalid")
    if objective.get("covariance_denominator") != "N_MINUS_1":
        raise HarnessFailure("R1 covariance denominator is not N_MINUS_1")
    if objective.get("dtype") != "torch.float32" or objective.get("autocast") is not False:
        raise HarnessFailure("R1 numerical dtype contract is invalid")
    if objective.get("global_multiplier") != 1.0 or objective.get("class_reduction") != "mean":
        raise HarnessFailure("R1 objective reduction is invalid")
    if objective.get("mechanism_weights") != [0.5, 0.5]:
        raise HarnessFailure("R1 objective mechanism weights are invalid")
    if objective.get("prohibited_auxiliary") != [
        "pixel_mse",
        "decoder_loss",
        "cmos_loss",
        "classification",
        "evaluator_loss",
        "sinkhorn",
        "energy_distance",
        "mmd",
        "swd",
        "chamfer",
        "diversity",
        "saturation",
        "phase_tv",
        "command_gain_regularizer",
        "code_norm_penalty",
        "global_prior",
        "class9_term",
    ]:
        raise HarnessFailure("objective auxiliary-loss firewall is invalid")

    horizon = _mapping(config["horizon"], "horizon")
    if dict(horizon) != {
        "logical_steps": HORIZON,
        "reference_formal_steps": 4920,
        "reference_formal_logical_batch": 50,
        "reference_controller_state_exposure": 246000,
        "geometry_logical_batch": REAL_COUNT,
        "basis": "minimal_delta_preserve_current_formal_controller_state_exposure",
        "sweep": False,
        "extension": False,
        "early_stop": False,
        "train_until_s3_pass": False,
    }:
        raise HarnessFailure("horizon contract differs from frozen authority")

    checkpoint = _mapping(config["checkpoint"], "checkpoint")
    if dict(checkpoint) != {
        "selection": "FINAL_AUTHORIZED_STEP_ONLY",
        "step": HORIZON,
        "best_loss": False,
        "best_validation": False,
        "s3_selected": False,
        "rollback": False,
    }:
        raise HarnessFailure("checkpoint selection is not final-step-only")


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _module_state_hash(module: torch.nn.Module) -> str:
    return _state_hash(module.state_dict())


def _assert_tensor(value: Any, *, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
        raise HarnessFailure(f"{name} must have shape {shape} and dtype {dtype}")
    if not torch.isfinite(value).all():
        raise HarnessFailure(f"{name} contains non-finite values")
    return value.detach().cpu().contiguous()


def _class_labels(count: int) -> torch.Tensor:
    if count != REAL_COUNT:
        raise HarnessFailure("registered class-balanced count is invalid")
    return _EXPECTED_TRAINING_LABELS.clone()


def _validate_balanced_labels(labels: torch.Tensor, *, name: str) -> None:
    if labels.ndim != 1 or labels.dtype != torch.long or labels.numel() != REAL_COUNT:
        raise HarnessFailure(f"{name} labels are not [500] int64")
    if not torch.equal(labels.cpu(), _EXPECTED_TRAINING_LABELS):
        raise HarnessFailure(f"{name} labels are not canonical label-major 50/class")


def _identity_set(source: str, indices: torch.Tensor) -> set[tuple[str, int]]:
    return {(source, int(value)) for value in indices.tolist()}


def _load_s1_reference(config: Mapping[str, Any], repo_root: str | Path) -> S1Reference:
    firewall = _mapping(config["data_firewall"], "data_firewall")
    s1 = _mapping(firewall["s1"], "data_firewall.s1")
    semantic_path = resolve_path(repo_root, s1["semantic_manifest"])
    embeddings_path = resolve_path(repo_root, s1["embeddings"])
    if file_sha256(semantic_path) != s1["semantic_manifest_sha256"]:
        raise HarnessFailure("S1 semantic manifest hash mismatch")
    if file_sha256(embeddings_path) != s1["embeddings_sha256"]:
        raise HarnessFailure("S1 embedding artifact hash mismatch")
    with semantic_path.open("r", encoding="utf-8") as handle:
        semantic_manifest = json.load(handle)
    if semantic_manifest.get("gate") != "S1_POWER_SEMANTICS_SUPPORTED":
        raise HarnessFailure("S1 source is not the supported R1 authority")
    payload = safe_torch_load(embeddings_path, expected_sha256=s1["embeddings_sha256"])
    if not isinstance(payload, Mapping):
        raise HarnessFailure("S1 embedding artifact is not a mapping")
    representations = _mapping(payload.get("representations"), "S1 representations")
    labels = _mapping(payload.get("labels"), "S1 labels")
    indices = _mapping(payload.get("indices"), "S1 indices")
    fit_r1 = _assert_tensor(representations["fit"]["R1_POWER"], name="S1 fit R1", shape=(1000, R1_DIM), dtype=torch.float32)
    fit_labels = _assert_tensor(labels["fit"], name="S1 fit labels", shape=(1000,), dtype=torch.long)
    fit_indices = _assert_tensor(indices["fit"], name="S1 fit indices", shape=(1000,), dtype=torch.long)
    selection_labels = _assert_tensor(labels["selection"], name="S1 selection labels", shape=(1000,), dtype=torch.long)
    selection_indices = _assert_tensor(indices["selection"], name="S1 selection indices", shape=(1000,), dtype=torch.long)
    confirmation_labels = _assert_tensor(labels["confirmation"], name="S1 confirmation labels", shape=(5000,), dtype=torch.long)
    confirmation_indices = _assert_tensor(indices["confirmation"], name="S1 confirmation indices", shape=(5000,), dtype=torch.long)
    for value, expected, name in (
        (fit_indices, S1_FIT_INDICES_SHA256, "S1 fit indices"),
        (fit_labels, S1_FIT_LABELS_SHA256, "S1 fit labels"),
        (selection_indices, S1_SELECTION_INDICES_SHA256, "S1 selection indices"),
        (selection_labels, S1_SELECTION_LABELS_SHA256, "S1 selection labels"),
        (confirmation_indices, S1_CONFIRMATION_INDICES_SHA256, "S1 confirmation indices"),
        (confirmation_labels, S1_CONFIRMATION_LABELS_SHA256, "S1 confirmation labels"),
    ):
        if tensor_sha256(value) != expected:
            raise HarnessFailure(f"{name} identity changed")
    if not torch.equal(torch.bincount(fit_labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), 100, dtype=torch.long)):
        raise HarnessFailure("S1 fit is not 100/class")
    if torch.unique(fit_indices).numel() != fit_indices.numel():
        raise HarnessFailure("S1 fit indices contain duplicates")
    if _identity_set("mnist_train", fit_indices) & _identity_set("mnist_train", confirmation_indices):
        raise HarnessFailure("S1 fit overlaps protected confirmation")
    return S1Reference(
        fit_r1=fit_r1,
        fit_labels=fit_labels,
        fit_indices=fit_indices,
        selection_labels=selection_labels,
        selection_indices=selection_indices,
        confirmation_labels=confirmation_labels,
        confirmation_indices=confirmation_indices,
        source_artifact_sha256=s1["embeddings_sha256"],
    )


def _protected_sample_sets(
    reference: S1Reference,
    config: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> dict[str, set[tuple[str, int]]]:
    firewall = _mapping(config["data_firewall"], "data_firewall")
    s2 = _mapping(firewall["s2_validation"], "data_firewall.s2_validation")
    validation = torch.arange(int(s2["start"]), int(s2["stop"]), dtype=torch.long)
    if tensor_sha256(validation) != s2["indices_sha256"]:
        raise HarnessFailure("S2 validation index range does not match authority")
    labels_artifact = resolve_path(repo_root, s2["labels_artifact"])
    if file_sha256(labels_artifact) != s2["labels_artifact_sha256"]:
        raise HarnessFailure("S2 validation label source hash mismatch")
    dataset_payload = safe_torch_load(labels_artifact, expected_sha256=s2["labels_artifact_sha256"])
    if not isinstance(dataset_payload, (tuple, list)) or len(dataset_payload) != 2:
        raise HarnessFailure("MNIST training source is not the registered pair")
    dataset_labels = dataset_payload[1]
    if not torch.is_tensor(dataset_labels) or tensor_sha256(dataset_labels[int(s2["start"]):int(s2["stop"])]) != s2["labels_sha256"]:
        raise HarnessFailure("S2 validation labels identity changed")
    return {
        "s1_selection": _identity_set("mnist_test", reference.selection_indices),
        "s1_confirmation": _identity_set("mnist_train", reference.confirmation_indices),
        "s2_validation": _identity_set("mnist_train", validation),
    }


def _validate_frozen_authority_files(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, str]:
    binding = _mapping(config["optical_authority"], "optical_authority")
    checks = {
        "formal_config": (binding["formal_config"], binding["formal_config_sha256"]),
        "checkpoint": (binding["checkpoint"], binding["checkpoint_sha256"]),
        "calibration": (binding["calibration"], binding["calibration_sha256"]),
        "calibration_manifest": (binding["calibration_manifest"], binding["calibration_manifest_sha256"]),
        "evaluator_identity_manifest": (
            binding["evaluator_identity_manifest"],
            binding["evaluator_identity_manifest_file_sha256"],
        ),
    }
    result: dict[str, str] = {}
    for name, (path, expected) in checks.items():
        target = resolve_path(repo_root, path)
        actual = file_sha256(target)
        if actual != expected:
            raise HarnessFailure(f"frozen {name} identity mismatch")
        result[name] = actual
    with resolve_path(repo_root, binding["evaluator_identity_manifest"]).open("r", encoding="utf-8") as handle:
        evaluator_identity = json.load(handle)
    if evaluator_identity.get("identity_manifest_hash") != binding["evaluator_identity_manifest_sha256"]:
        raise HarnessFailure("evaluator identity manifest binding mismatch")
    return result


def _load_s3_base_codes(config: Mapping[str, Any], repo_root: str | Path) -> torch.Tensor:
    panel = _mapping(_mapping(config["data_firewall"], "data_firewall")["s3_panel"], "data_firewall.s3_panel")
    manifest_path = resolve_path(repo_root, panel["manifest"])
    if file_sha256(manifest_path) != panel["manifest_sha256"]:
        raise HarnessFailure("S3 manifest hash mismatch")
    panel_identity_path = resolve_path(repo_root, panel["panel_identity_artifact"])
    if file_sha256(panel_identity_path) != panel["panel_identity_artifact_sha256"]:
        raise HarnessFailure("S3 panel identity artifact hash mismatch")
    with panel_identity_path.open("r", encoding="utf-8") as handle:
        panel_identity = json.load(handle)
    if panel_identity.get("panel_identity_sha256") != panel["panel_identity_sha256"]:
        raise HarnessFailure("S3 panel identity binding mismatch")
    state_path = resolve_path(repo_root, panel["states"])
    if file_sha256(state_path) != panel["states_sha256"]:
        raise HarnessFailure("S3 frozen state artifact hash mismatch")
    payload = safe_torch_load(state_path, expected_sha256=panel["states_sha256"])
    if not isinstance(payload, Mapping) or not torch.is_tensor(payload.get("codes")):
        raise HarnessFailure("S3 frozen panel artifact has no codes tensor")
    codes = payload["codes"]
    if tuple(codes.shape) != (1000, CODE_DIM) or codes.dtype != torch.float32:
        raise HarnessFailure("S3 frozen panel code tensor identity is invalid")
    base_codes = codes[:100].detach().cpu().contiguous()
    if tensor_sha256(base_codes) != S3_BASE_CODE_SHA256:
        raise HarnessFailure("S3 protected base-code hash mismatch")
    return base_codes


def internal_validation_codes() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(INTERNAL_VALIDATION_CODE_SEED)
    codes = torch.rand((100, CODE_DIM), generator=generator, dtype=torch.float32)
    return codes.mul(2.0).sub(1.0).contiguous()


def _validate_codes(codes: torch.Tensor, *, name: str) -> None:
    if codes.ndim != 2 or tuple(codes.shape[1:]) != (CODE_DIM,) or codes.dtype != torch.float32:
        raise HarnessFailure(f"{name} must be float32 [N,8]")
    if not torch.isfinite(codes).all() or bool((codes < -1.0).any()) or bool((codes > 1.0).any()):
        raise HarnessFailure(f"{name} is outside Uniform([-1,1]^8)")


def code_overlap_count(codes: torch.Tensor, protected: torch.Tensor) -> int:
    _validate_codes(codes, name="codes")
    _validate_codes(protected, name="protected codes")
    if protected.numel() == 0 or codes.numel() == 0:
        return 0
    matches = (codes[:, None, :] == protected[None, :, :]).all(dim=2)
    return int(matches.sum().item())


class TrainingCodeStream:
    """One sequential CPU RNG stream; constructing one is the only reset."""

    def __init__(self, seed: int = TRAINING_CODE_SEED) -> None:
        if seed != TRAINING_CODE_SEED:
            raise HarnessFailure("training code stream seed is not the registered derived seed")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.blocks_consumed = 0

    def next_block(self) -> torch.Tensor:
        codes = torch.rand(
            (SAMPLES_PER_CLASS, CODE_DIM),
            generator=self.generator,
            dtype=torch.float32,
        )
        self.blocks_consumed += 1
        return codes.mul(2.0).sub(1.0).contiguous()

    def materialize(self, logical_steps: int) -> torch.Tensor:
        if logical_steps <= 0:
            raise HarnessFailure("logical_steps must be positive")
        blocks = [self.next_block() for _ in range(logical_steps)]
        return torch.stack(blocks, dim=0)


def finite_training_stream(logical_steps: int = HORIZON) -> tuple[torch.Tensor, dict[str, Any]]:
    stream = TrainingCodeStream()
    blocks = stream.materialize(logical_steps)
    first_hash = tensor_sha256(blocks[0])
    if logical_steps == HORIZON and first_hash != TRAINING_FIRST_BLOCK_SHA256:
        raise HarnessFailure("training RNG first block identity changed")
    flat = blocks.reshape(-1, CODE_DIM)
    internal = internal_validation_codes()
    metadata = {
        "seed": TRAINING_CODE_SEED,
        "seed_material_sha256": TRAINING_CODE_SEED_MATERIAL_SHA256,
        "generator_device": "cpu",
        "dtype": "torch.float32",
        "shape": list(blocks.shape),
        "logical_steps": logical_steps,
        "base_codes": int(flat.shape[0]),
        "first_block_sha256": first_hash,
        "finite_training_stream_sha256": tensor_sha256(blocks),
        "s3_overlap_count": None,
        "internal_validation_overlap_count": code_overlap_count(flat, internal),
    }
    return blocks, metadata


def validate_finite_stream_against_protected(
    blocks: torch.Tensor,
    s3_base_codes: torch.Tensor,
    internal_codes: torch.Tensor | None = None,
) -> dict[str, Any]:
    if blocks.ndim != 3 or tuple(blocks.shape[1:]) != (SAMPLES_PER_CLASS, CODE_DIM):
        raise HarnessFailure("finite training stream shape is invalid")
    flat = blocks.reshape(-1, CODE_DIM)
    internal = internal_validation_codes() if internal_codes is None else internal_codes
    s3_overlap = code_overlap_count(flat, s3_base_codes)
    internal_overlap = code_overlap_count(flat, internal)
    if s3_overlap or internal_overlap:
        raise HarnessFailure("protected code overlap makes the harness invalid")
    return {
        "finite_training_stream_sha256": tensor_sha256(blocks),
        "s3_overlap_count": s3_overlap,
        "internal_validation_overlap_count": internal_overlap,
    }


def build_logical_batch(real_r1: torch.Tensor, real_labels: torch.Tensor, base_codes: torch.Tensor) -> LogicalGeometryBatch:
    real_r1 = _assert_tensor(real_r1, name="real R1 logical bank", shape=(REAL_COUNT, R1_DIM), dtype=torch.float32)
    _validate_balanced_labels(real_labels, name="real")
    base_codes = _assert_tensor(base_codes, name="controller base codes", shape=(SAMPLES_PER_CLASS, CODE_DIM), dtype=torch.float32)
    _validate_codes(base_codes, name="controller base codes")
    controller_codes = base_codes.repeat((CLASS_COUNT, 1)).contiguous()
    controller_labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(SAMPLES_PER_CLASS)
    return LogicalGeometryBatch(
        real_r1=real_r1,
        real_labels=real_labels.detach().cpu().contiguous(),
        controller_codes=controller_codes,
        controller_labels=controller_labels,
        base_codes=base_codes,
        physical_microbatch_size=PHYSICAL_MICROBATCH_SIZE,
    )


def load_target_bank(path: str | Path, *, expected_sha256: str | None = None) -> TargetBank:
    artifact = Path(path)
    payload = safe_torch_load(artifact, expected_sha256=expected_sha256)
    if not isinstance(payload, Mapping):
        raise HarnessFailure("target bank artifact is not a mapping")
    if set(payload) != {"schema", "manifest", "training", "internal_validation"}:
        raise HarnessFailure("target bank contains data outside training/internal_validation")
    if payload["schema"] != TARGET_BANK_SCHEMA:
        raise HarnessFailure("target bank schema is invalid")
    manifest = _mapping(payload["manifest"], "target bank manifest")
    sidecar = artifact.with_suffix(".manifest.json")
    if not sidecar.is_file():
        raise HarnessFailure("target bank sidecar manifest is missing")
    with sidecar.open("r", encoding="utf-8") as handle:
        sidecar_payload = json.load(handle)
    sidecar_artifact = _mapping(sidecar_payload.get("artifact"), "target bank sidecar artifact")
    if sidecar_artifact.get("sha256") != file_sha256(artifact) or sidecar_artifact.get("schema") != TARGET_BANK_SCHEMA:
        raise HarnessFailure("target bank sidecar artifact identity mismatch")
    for key, value in manifest.items():
        if sidecar_payload.get(key) != value:
            raise HarnessFailure("target bank sidecar manifest does not match payload manifest")
    training = _mapping(payload["training"], "target bank training")
    validation = _mapping(payload["internal_validation"], "target bank internal_validation")
    if set(training) != {"r1", "labels", "indices"} or set(validation) != {"r1", "labels", "indices"}:
        raise HarnessFailure("target bank split contains unregistered fields")
    return TargetBank(
        training_r1=_assert_tensor(training["r1"], name="target bank training R1", shape=(REAL_COUNT, R1_DIM), dtype=torch.float32),
        training_labels=_assert_tensor(training["labels"], name="target bank training labels", shape=(REAL_COUNT,), dtype=torch.long),
        training_indices=_assert_tensor(training["indices"], name="target bank training indices", shape=(REAL_COUNT,), dtype=torch.long),
        validation_r1=_assert_tensor(validation["r1"], name="target bank validation R1", shape=(REAL_COUNT, R1_DIM), dtype=torch.float32),
        validation_labels=_assert_tensor(validation["labels"], name="target bank validation labels", shape=(REAL_COUNT,), dtype=torch.long),
        validation_indices=_assert_tensor(validation["indices"], name="target bank validation indices", shape=(REAL_COUNT,), dtype=torch.long),
        manifest=manifest,
        artifact_sha256=file_sha256(artifact),
    )


def validate_target_bank(
    bank: TargetBank,
    reference: S1Reference,
    config: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
    repo_root: str | Path = ".",
) -> None:
    if expected_artifact_sha256 is not None and bank.artifact_sha256 != expected_artifact_sha256:
        raise HarnessFailure("target bank artifact hash mismatch")
    _validate_balanced_labels(bank.training_labels, name="target bank training")
    _validate_balanced_labels(bank.validation_labels, name="target bank internal_validation")
    if torch.unique(bank.training_indices).numel() != REAL_COUNT or torch.unique(bank.validation_indices).numel() != REAL_COUNT:
        raise HarnessFailure("target bank indices contain duplicates")
    training_set = _identity_set("mnist_train", bank.training_indices)
    validation_set = _identity_set("mnist_train", bank.validation_indices)
    fit_set = _identity_set("mnist_train", reference.fit_indices)
    if training_set & validation_set or training_set | validation_set != fit_set:
        raise HarnessFailure("target bank partition is not a disjoint full S1-fit partition")
    protected = _protected_sample_sets(reference, config, repo_root)
    for name, protected_set in protected.items():
        if (training_set | validation_set) & protected_set:
            raise HarnessFailure(f"target bank overlaps protected sample split {name}")
    manifest = bank.manifest
    if manifest.get("schema") != TARGET_BANK_MANIFEST_SCHEMA:
        raise HarnessFailure("target bank manifest schema is invalid")
    if manifest.get("partition_identity") != PARTITION_IDENTITY:
        raise HarnessFailure("target bank partition identity mismatch")
    if manifest.get("partition_manifest_sha256") != PARTITION_IDENTITY:
        raise HarnessFailure("target bank partition manifest binding mismatch")
    if manifest.get("observer_identity") != "R1_POWER:z_j=abs(c_j)^2/s_j":
        raise HarnessFailure("target bank observer identity mismatch")
    if manifest.get("source_artifact_sha256") != reference.source_artifact_sha256:
        raise HarnessFailure("target bank source lineage mismatch")
    expected_hashes = {
        "training_r1_sha256": tensor_sha256(bank.training_r1),
        "training_labels_sha256": tensor_sha256(bank.training_labels),
        "training_indices_sha256": tensor_sha256(bank.training_indices),
        "internal_validation_r1_sha256": tensor_sha256(bank.validation_r1),
        "internal_validation_labels_sha256": tensor_sha256(bank.validation_labels),
        "internal_validation_indices_sha256": tensor_sha256(bank.validation_indices),
    }
    if any(manifest.get(name) != value for name, value in expected_hashes.items()):
        raise HarnessFailure("target bank tensor identity mismatch")
    if manifest.get("protected_sample_identities") != {
        "s1_selection_indices_sha256": S1_SELECTION_INDICES_SHA256,
        "s1_confirmation_indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
        "s2_validation_indices_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s2_validation"], "data_firewall.s2_validation")["indices_sha256"],
    }:
        raise HarnessFailure("target bank protected sample identity binding mismatch")


def register_target_bank(
    config: Mapping[str, Any],
    repo_root: str | Path,
    partition_manifest_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    reference = _load_s1_reference(config, repo_root)
    partition_path = Path(partition_manifest_path)
    with partition_path.open("r", encoding="utf-8") as handle:
        partition = json.load(handle)
    if file_sha256(partition_path) != PARTITION_IDENTITY:
        raise HarnessFailure("partition manifest bytes do not match the registered identity")
    if not isinstance(partition, Mapping) or partition.get("schema") != PARTITION_SCHEMA:
        raise HarnessFailure("partition manifest schema is invalid")
    if partition.get("partition_identity") != PARTITION_IDENTITY:
        raise HarnessFailure("partition manifest identity mismatch")
    positions = {}
    for key in ("training_positions", "internal_validation_positions"):
        values = partition.get(key)
        if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
            raise HarnessFailure(f"{key} must be an explicit integer list")
        positions[key] = torch.tensor(values, dtype=torch.long)
        if positions[key].numel() != REAL_COUNT or bool((positions[key] < 0).any()) or bool((positions[key] >= 1000).any()):
            raise HarnessFailure(f"{key} is not a 500-row S1-fit position list")
        if torch.unique(positions[key]).numel() != REAL_COUNT:
            raise HarnessFailure(f"{key} contains duplicate positions")
    if torch.isin(positions["training_positions"], positions["internal_validation_positions"]).any():
        raise HarnessFailure("partition positions overlap")
    if torch.cat(tuple(positions.values())).unique().numel() != 1000:
        raise HarnessFailure("partition does not cover the full S1 fit bank")
    training_r1 = reference.fit_r1.index_select(0, positions["training_positions"])
    validation_r1 = reference.fit_r1.index_select(0, positions["internal_validation_positions"])
    training_labels = reference.fit_labels.index_select(0, positions["training_positions"])
    validation_labels = reference.fit_labels.index_select(0, positions["internal_validation_positions"])
    training_indices = reference.fit_indices.index_select(0, positions["training_positions"])
    validation_indices = reference.fit_indices.index_select(0, positions["internal_validation_positions"])
    manifest = {
        "schema": TARGET_BANK_MANIFEST_SCHEMA,
        "partition_identity": PARTITION_IDENTITY,
        "partition_manifest_sha256": PARTITION_IDENTITY,
        "source_artifact_sha256": reference.source_artifact_sha256,
        "observer_identity": "R1_POWER:z_j=abs(c_j)^2/s_j",
        "dataset_identity": "mnist_train",
        "per_class_count": SAMPLES_PER_CLASS,
        "protected_sample_identities": {
            "s1_selection_indices_sha256": S1_SELECTION_INDICES_SHA256,
            "s1_confirmation_indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
            "s2_validation_indices_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s2_validation"], "data_firewall.s2_validation")["indices_sha256"],
        },
        "optical_authority": dict(_mapping(config["optical_authority"], "optical_authority")),
        "training_r1_sha256": tensor_sha256(training_r1),
        "training_labels_sha256": tensor_sha256(training_labels),
        "training_indices_sha256": tensor_sha256(training_indices),
        "internal_validation_r1_sha256": tensor_sha256(validation_r1),
        "internal_validation_labels_sha256": tensor_sha256(validation_labels),
        "internal_validation_indices_sha256": tensor_sha256(validation_indices),
    }
    payload = {
        "schema": TARGET_BANK_SCHEMA,
        "manifest": manifest,
        "training": {"r1": training_r1, "labels": training_labels, "indices": training_indices},
        "internal_validation": {"r1": validation_r1, "labels": validation_labels, "indices": validation_indices},
    }
    validate_target_bank(
        TargetBank(
            training_r1,
            training_labels,
            training_indices,
            validation_r1,
            validation_labels,
            validation_indices,
            manifest,
        ),
        reference,
        config,
        repo_root=repo_root,
    )
    output = Path(output_path)
    atomic_torch_save_new(output, payload, artifact_kind="controller semantic geometry target bank")
    manifest_target = Path(manifest_path) if manifest_path is not None else output.with_suffix(".manifest.json")
    manifest_payload = {
        **manifest,
        "artifact": {"path": str(output), "sha256": file_sha256(output), "schema": TARGET_BANK_SCHEMA},
        "partition_manifest_sha256": file_sha256(partition_path),
    }
    atomic_write_new_json(manifest_target, manifest_payload, artifact_kind="controller semantic geometry target bank manifest")
    return output, manifest_target


def r1_power(coefficients: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(coefficients) or coefficients.dtype != torch.complex64:
        raise HarnessFailure("R1 observer requires complex64 OLS coefficients")
    if tuple(coefficients.shape[1:]) != (R1_DIM,):
        raise HarnessFailure("OLS coefficient dimension is not 100")
    if scale.ndim != 1 or scale.shape[0] != R1_DIM or scale.dtype != torch.float32:
        raise HarnessFailure("OLS scale must be float32 [100]")
    if not torch.isfinite(coefficients).all() or not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise HarnessFailure("R1 observer scale or coefficients are invalid")
    values = coefficients.abs().square() / scale.to(coefficients.device).unsqueeze(0)
    if values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise HarnessFailure("R1 observer produced invalid float32 values")
    return values


def class_moments(values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or tuple(values.shape) != (REAL_COUNT, R1_DIM) or values.dtype != torch.float32:
        raise HarnessFailure("class-moment values must be float32 [500,100]")
    _validate_balanced_labels(labels, name="moment")
    if not torch.isfinite(values).all():
        raise HarnessFailure("class-moment values contain non-finite values")
    means: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    for label in range(CLASS_COUNT):
        class_values = values[labels == label]
        if class_values.shape != (SAMPLES_PER_CLASS, R1_DIM):
            raise HarnessFailure("class-conditioned moment population is not 50")
        mean = class_values.sum(dim=0) / float(SAMPLES_PER_CLASS)
        centered = class_values - mean
        covariance = centered.T @ centered / float(SAMPLES_PER_CLASS - 1)
        means.append(mean)
        covariances.append(covariance)
    return torch.stack(means), torch.stack(covariances)


def _validate_real_moments(means: torch.Tensor, covariances: torch.Tensor) -> None:
    if means.dtype != torch.float32 or covariances.dtype != torch.float32:
        raise HarnessFailure("geometry moments must remain float32")
    if not torch.isfinite(means).all() or not torch.isfinite(covariances).all():
        raise HarnessFailure("real geometry moments are non-finite")
    traces = covariances.diagonal(dim1=1, dim2=2).sum(dim=1)
    covariance_norms = covariances.square().sum(dim=(1, 2))
    if bool((traces <= 0).any()) or bool((covariance_norms <= 0).any()):
        raise HarnessFailure("real geometry normalization denominator is zero or negative")


def geometry_alignment_loss(
    real_r1: torch.Tensor,
    controller_r1: torch.Tensor,
    real_labels: torch.Tensor,
    controller_labels: torch.Tensor,
) -> GeometryObjectiveResult:
    real_means, real_covariances = class_moments(real_r1.detach(), real_labels)
    if not torch.isfinite(controller_r1).all():
        raise HarnessFailure(INVALID_HARNESS)
    controller_means, controller_covariances = class_moments(controller_r1, controller_labels)
    _validate_real_moments(real_means, real_covariances)
    if not torch.isfinite(controller_means).all() or not torch.isfinite(controller_covariances).all():
        raise HarnessFailure(INVALID_HARNESS)
    anchor_scale = real_covariances.diagonal(dim1=1, dim2=2).sum(dim=1)
    covariance_scale = real_covariances.square().sum(dim=(1, 2))
    anchor_terms = (controller_means - real_means).square().sum(dim=1) / anchor_scale
    covariance_terms = (controller_covariances - real_covariances).square().sum(dim=(1, 2)) / covariance_scale
    class_terms = (anchor_terms + covariance_terms) / 2.0
    loss = class_terms.mean()
    all_values = (anchor_terms, covariance_terms, class_terms, loss)
    if any(not torch.isfinite(value).all() for value in all_values):
        raise HarnessFailure(INVALID_HARNESS)
    return GeometryObjectiveResult(
        loss=loss,
        anchor_terms=anchor_terms,
        covariance_terms=covariance_terms,
        class_terms=class_terms,
        mean_anchor=anchor_terms.mean(),
        mean_covariance=covariance_terms.mean(),
    )


def controller_parameter_items(generator: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    named = dict(generator.named_parameters())
    if set(named) != set(_GENERATOR_PARAMETER_NAMES):
        raise HarnessFailure(f"generator parameter set changed: {tuple(named)}")
    return tuple((name, named[name.removeprefix("generator.")]) for name in CONTROLLER_PARAMETER_NAMES)


def build_controller_optimizer(generator: torch.nn.Module) -> torch.optim.Adam:
    items = controller_parameter_items(generator)
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in items],
        lr=2.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=True,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if len(optimizer.param_groups) != 1 or optimizer.state:
        raise HarnessFailure("controller optimizer is not fresh and one-group")
    defaults = optimizer.defaults
    expected = {
        "lr": 2.0e-4,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": True,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    if any(defaults.get(key) != value for key, value in expected.items()):
        raise HarnessFailure("Adam defaults differ from the registered optimizer authority")
    return optimizer


def optimizer_parameter_names(optimizer: torch.optim.Optimizer, generator: torch.nn.Module) -> tuple[str, ...]:
    items = controller_parameter_items(generator)
    by_id = {id(parameter): name for name, parameter in items}
    names = tuple(by_id.get(id(parameter), "") for parameter in optimizer.param_groups[0]["params"])
    if names != CONTROLLER_PARAMETER_NAMES:
        raise HarnessFailure("optimizer parameter coverage is not exactly controller-only")
    return names


def freeze_optical_model(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def capture_state_snapshot(
    generator: torch.nn.Module,
    optical_model: torch.nn.Module,
    file_paths: Mapping[str, str | Path] | None = None,
) -> FrozenStateSnapshot:
    files = {
        name: file_sha256(path) for name, path in (file_paths or {}).items()
    }
    frozen_parameters = {
        name: tensor_sha256(parameter.detach())
        for name, parameter in optical_model.named_parameters()
    }
    return FrozenStateSnapshot(
        controller_state_hash=_module_state_hash(generator),
        frozen_model_state_hash=_module_state_hash(optical_model),
        frozen_parameter_hashes=frozen_parameters,
        file_hashes=files,
    )


def validate_frozen_state(
    before: FrozenStateSnapshot,
    after: FrozenStateSnapshot,
    generator: torch.nn.Module,
    optical_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    if before.frozen_model_state_hash != after.frozen_model_state_hash:
        raise HarnessFailure("frozen optical model state changed")
    if before.frozen_parameter_hashes != after.frozen_parameter_hashes:
        raise HarnessFailure("frozen optical parameter state changed")
    if before.file_hashes != after.file_hashes:
        raise HarnessFailure("frozen artifact identity changed")
    optimizer_parameter_names(optimizer, generator)
    controller_ids = {id(parameter) for _, parameter in controller_parameter_items(generator)}
    if any(id(parameter) not in controller_ids for parameter in optimizer.state):
        raise HarnessFailure("optimizer acquired non-controller state")
    for parameter in optical_model.parameters():
        if parameter.requires_grad:
            raise HarnessFailure("optical parameter is trainable")


def controller_to_ols_r1(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    codes: torch.Tensor,
    labels: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Run only controller -> amplitude -> frozen front-end -> complex OLS."""
    if codes.ndim != 2 or codes.shape[1] != CODE_DIM or codes.dtype != torch.float32:
        raise HarnessFailure("controller codes are not float32 [N,8]")
    amplitude = generator(codes, labels)
    if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != (1, 200, 200):
        raise HarnessFailure("controller amplitude shape is invalid")
    from ldm_project.v2.application.research.optical_manifold_smf_bogt import (
        FIELD_TRANSMISSION,
        _attenuate,
    )

    slm = optical_model.system.slm(optical_model.system.command(amplitude))
    front = optical_model.system.front(slm)
    encoded, records, _ = optical_model.system.encoder(front, "encoder")
    if len(records) != 2:
        raise HarnessFailure("frozen encoder trace does not match the registered two-layer prefix")
    encoded = _attenuate(encoded, FIELD_TRANSMISSION**2, "fixed_encoder_group_transmission")
    coefficients = optical_model._coefficients(encoded)
    return r1_power(coefficients, scale)


def controller_logical_r1(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    batch: LogicalGeometryBatch,
    scale: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    for start in range(0, REAL_COUNT, batch.physical_microbatch_size):
        stop = start + batch.physical_microbatch_size
        chunks.append(
            controller_to_ols_r1(
                generator,
                optical_model,
                batch.controller_codes[start:stop].to(next(generator.parameters()).device),
                batch.controller_labels[start:stop].to(next(generator.parameters()).device),
                scale,
            )
        )
    return torch.cat(chunks, dim=0)


def train_authorized_controller(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    target_bank: TargetBank,
    scale: torch.Tensor,
    output_path: str | Path,
    *,
    authorization: str | None,
    frozen_file_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    if authorization != AUTHORIZATION_TOKEN:
        raise HarnessFailure("explicit one-shot training authorization is required")
    freeze_optical_model(optical_model)
    for _, parameter in controller_parameter_items(generator):
        parameter.requires_grad_(True)
    optimizer = build_controller_optimizer(generator)
    if optimizer_parameter_names(optimizer, generator) != CONTROLLER_PARAMETER_NAMES:
        raise HarnessFailure("controller optimizer coverage failed")
    before = capture_state_snapshot(generator, optical_model, frozen_file_paths)
    real_r1 = target_bank.training_r1.to(next(generator.parameters()).device)
    real_labels = target_bank.training_labels.to(next(generator.parameters()).device)
    real_means, real_covariances = class_moments(real_r1, real_labels)
    _validate_real_moments(real_means.detach(), real_covariances.detach())
    stream = TrainingCodeStream()
    last_result: GeometryObjectiveResult | None = None
    for _step in range(1, HORIZON + 1):
        base_codes = stream.next_block().to(next(generator.parameters()).device)
        batch = build_logical_batch(real_r1, real_labels, base_codes)
        controller_r1 = controller_logical_r1(generator, optical_model, batch, scale.to(real_r1.device))
        last_result = geometry_alignment_loss(
            batch.real_r1.to(real_r1.device),
            controller_r1,
            batch.real_labels.to(real_r1.device),
            batch.controller_labels.to(real_r1.device),
        )
        optimizer.zero_grad(set_to_none=True)
        last_result.loss.backward()
        optimizer.step()
    if last_result is None:
        raise HarnessFailure("authorized horizon produced no update")
    after = capture_state_snapshot(generator, optical_model, frozen_file_paths)
    validate_frozen_state(before, after, generator, optical_model, optimizer)
    payload = {
        "schema": "controller_semantic_geometry_final_controller_state.v1",
        "objective_family": OBJECTIVE_FAMILY,
        "checkpoint_selection": "FINAL_AUTHORIZED_STEP_ONLY",
        "step": HORIZON,
        "controller_state": {name: value.detach().cpu() for name, value in generator.state_dict().items()},
        "controller_state_hash": _module_state_hash(generator),
        "training_authorization": AUTHORIZATION_TOKEN,
        "training_authorized": True,
        "formal_promotion": False,
        "s3_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
        "mean_anchor": float(last_result.mean_anchor.detach().cpu()),
        "mean_covariance": float(last_result.mean_covariance.detach().cpu()),
        "loss": float(last_result.loss.detach().cpu()),
    }
    atomic_torch_save_new(output_path, payload, artifact_kind="controller semantic geometry final checkpoint")
    return {"checkpoint": str(output_path), "checkpoint_sha256": file_sha256(output_path), **{key: value for key, value in payload.items() if key != "controller_state"}}


def _git_state(repo_root: str | Path) -> tuple[str, tuple[str, ...]]:
    root = Path(repo_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return head, tuple(status)


def no_update_preflight(
    config_path: str | Path,
    repo_root: str | Path,
    target_bank_path: str | Path,
    output_path: str | Path,
    *,
    expected_head: str | None = None,
) -> dict[str, Any]:
    config_file = resolve_path(repo_root, config_path)
    config = load_config(config_file)
    head, dirty = _git_state(repo_root)
    if dirty:
        raise HarnessFailure("clean Git worktree is required for the no-update preflight")
    if expected_head is not None and head != expected_head:
        raise HarnessFailure("implementation HEAD does not match the expected clean preflight HEAD")
    frozen_authorities = _validate_frozen_authority_files(config, repo_root)
    reference = _load_s1_reference(config, repo_root)
    bank = load_target_bank(target_bank_path)
    validate_target_bank(bank, reference, config, repo_root=repo_root)
    s3_codes = _load_s3_base_codes(config, repo_root)
    blocks, stream_meta = finite_training_stream(HORIZON)
    stream_meta.update(validate_finite_stream_against_protected(blocks, s3_codes))
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    optimizer = build_controller_optimizer(generator)
    optimizer_names = optimizer_parameter_names(optimizer, generator)
    real_means, real_covariances = class_moments(bank.training_r1, bank.training_labels)
    _validate_real_moments(real_means, real_covariances)
    output = Path(output_path)
    if output.exists():
        raise HarnessFailure("preflight output target is not a fresh write-once path")
    if config["governance"]["training_authorized"] is not False:
        raise HarnessFailure("training authorization is not explicitly absent")
    return {
        "gate": PREFLIGHT_PASS,
        "implementation_status": IMPLEMENTATION_READY,
        "training_authorized": False,
        "formal_promotion": False,
        "s3_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
        "git_head": head,
        "dirty_paths": list(dirty),
        "config_sha256": file_sha256(config_file),
        "frozen_authorities": frozen_authorities,
        "target_bank_sha256": bank.artifact_sha256,
        "partition_identity": PARTITION_IDENTITY,
        "s3_base_code_sha256": tensor_sha256(s3_codes),
        "internal_validation_code_sha256": tensor_sha256(internal_validation_codes()),
        "optimizer_contract_sha256": OPTIMIZER_CONTRACT_SHA256,
        "optimizer_parameter_names": list(optimizer_names),
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "real_anchor_denominators": [float(value) for value in real_covariances.diagonal(dim1=1, dim2=2).sum(dim=1)],
        "real_covariance_denominators": [float(value) for value in real_covariances.square().sum(dim=(1, 2))],
        "stream": stream_meta,
        "checkpoint_selection": "FINAL_AUTHORIZED_STEP_ONLY",
        "output_target": str(output),
        "optimizer_step_count": 0,
        "backward_count": 0,
        "checkpoint_writes": 0,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--repo-root", required=True)
    preflight.add_argument("--target-bank", required=True)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--expected-head")
    args = parser.parse_args()
    if args.command == "preflight":
        result = no_update_preflight(
            args.config,
            args.repo_root,
            args.target_bank,
            args.output,
            expected_head=args.expected_head,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


__all__ = [
    "AUTHORIZATION_TOKEN",
    "CLASS_COUNT",
    "CONTROLLER_PARAMETER_NAMES",
    "GeometryObjectiveResult",
    "HarnessFailure",
    "HORIZON",
    "INTERNAL_VALIDATION_CODE_SEED",
    "LogicalGeometryBatch",
    "OBJECTIVE_CONTRACT_SHA256",
    "OBJECTIVE_FAMILY",
    "OPTIMIZER_CONTRACT_SHA256",
    "PARTITION_IDENTITY",
    "PREFLIGHT_PASS",
    "REAL_COUNT",
    "S1Reference",
    "SAMPLES_PER_CLASS",
    "SCHEMA",
    "TARGET_BANK_SCHEMA",
    "TargetBank",
    "TRAINING_CODE_SEED",
    "TRAINING_FIRST_BLOCK_SHA256",
    "TrainingCodeStream",
    "build_controller_optimizer",
    "build_logical_batch",
    "capture_state_snapshot",
    "class_moments",
    "code_overlap_count",
    "controller_parameter_items",
    "controller_to_ols_r1",
    "finite_training_stream",
    "freeze_optical_model",
    "geometry_alignment_loss",
    "internal_validation_codes",
    "load_config",
    "load_target_bank",
    "no_update_preflight",
    "optimizer_parameter_names",
    "r1_power",
    "register_target_bank",
    "resolve_path",
    "train_authorized_controller",
    "validate_config",
    "validate_finite_stream_against_protected",
    "validate_frozen_state",
    "validate_target_bank",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
