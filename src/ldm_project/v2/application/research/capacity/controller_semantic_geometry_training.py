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
import time
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
DATA_AUTHORITY_SCHEMA = "controller_semantic_geometry_data_firewall_manifest.v1"
OBJECTIVE_FAMILY = "CLASS_CONDITIONED_R1_FIRST_SECOND_ORDER_GEOMETRY_ALIGNMENT"
AUTHORIZATION_TOKEN = "AUTHORIZE_ONE_SHOT_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_CANARY"
IMPLEMENTATION_READY = "IMPLEMENTATION_READY_FOR_ONE_SHOT_REAUTHORIZATION"
PREFLIGHT_PASS = "PRETRAINING_HARNESS_PREFLIGHT_PASS"
TRAINING_PREFLIGHT_SCHEMA = "controller_semantic_geometry_training_preflight.v1"
TRAINING_PREFLIGHT_GATE = "CONTROLLER_SEMANTIC_GEOMETRY_3_STEP_PREFLIGHT_PASS"
TRAINING_PREFLIGHT_STEPS = 3
TRAINING_PREFLIGHT_MAX_SECONDS = 600.0
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
PARTITION_TRAINING_POSITION_SHA256 = (
    "54fe39a51036f6b69cf4cb65cfc614e4043a92fb8a601c75f529da8f87122243"
)
PARTITION_VALIDATION_POSITION_SHA256 = (
    "964724eddd503d596ec870ceea3b0145f151ade64f5903307352297372763a75"
)
PARTITION_TRAINING_INDEX_SHA256 = (
    "99285b32a2c098de1788cd405bc317c3c7877bbd07bd1b63260fd9fd1cc93d18"
)
PARTITION_VALIDATION_INDEX_SHA256 = (
    "96a367399d53a48078d7a3328a6063174c2f58031177da0f14953191620b5d80"
)
PARTITION_TRAINING_LABEL_SHA256 = (
    "c8bacf260ae894e7da623586b9e04c67d95f628c16f52766cee0994ca2ceac19"
)
PARTITION_VALIDATION_LABEL_SHA256 = PARTITION_TRAINING_LABEL_SHA256
PARTITION_TRAINING_R1_SHA256 = (
    "453d207c1b87468f6ed0f3d36d27c650207019f25577ac0484e91be234e911ab"
)
PARTITION_VALIDATION_R1_SHA256 = (
    "54cc0ae18a07e50fa11b6fe3aa7adba21c0f6651667f963054046d80e482143d"
)
PARTITION_ALGORITHM = {
    "class_order": list(range(CLASS_COUNT)),
    "within_class_order": "source_index_ascending",
    "training_rule": "first_50",
    "internal_validation_rule": "last_50",
    "concatenation": "label_major",
    "rng": "none",
}
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
class FrozenPartition:
    training_positions: torch.Tensor
    internal_validation_positions: torch.Tensor
    training_indices: torch.Tensor
    internal_validation_indices: torch.Tensor
    training_labels: torch.Tensor
    internal_validation_labels: torch.Tensor
    training_r1_sha256: str
    internal_validation_r1_sha256: str
    identity_material_sha256: str


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
    if firewall.get("partition_identity_sha256") != PARTITION_IDENTITY:
        raise HarnessFailure("data-firewall partition semantic identity is not frozen")
    authority = _mapping(firewall.get("authority"), "data_firewall.authority")
    if set(authority) != {
        "run_directory",
        "partition_manifest",
        "partition_manifest_file_sha256",
        "target_bank",
        "target_bank_sha256",
        "target_bank_manifest",
        "target_bank_manifest_sha256",
        "data_firewall_manifest",
        "data_firewall_manifest_sha256",
    }:
        raise HarnessFailure("data-authority binding fields are invalid")
    for name in (
        "partition_manifest_file_sha256",
        "target_bank_sha256",
        "target_bank_manifest_sha256",
        "data_firewall_manifest_sha256",
    ):
        value = authority.get(name)
        if value is not None:
            _sha(value, f"data_firewall.authority.{name}")
    for name in (
        "run_directory",
        "partition_manifest",
        "target_bank",
        "target_bank_manifest",
        "data_firewall_manifest",
    ):
        if not isinstance(authority.get(name), str) or not authority[name]:
            raise HarnessFailure(f"data-firewall authority {name} is invalid")
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


def build_frozen_partition(reference: S1Reference) -> FrozenPartition:
    """Materialize the registered S1-fit split without RNG or reselection."""
    training_positions: list[torch.Tensor] = []
    validation_positions: list[torch.Tensor] = []
    for label in range(CLASS_COUNT):
        rows = torch.nonzero(reference.fit_labels == label, as_tuple=False).flatten()
        order = torch.argsort(reference.fit_indices[rows], stable=True)
        rows = rows.index_select(0, order)
        if rows.numel() != SAMPLES_PER_CLASS * 2:
            raise HarnessFailure("DATA_FIREWALL_PARTITION_IDENTITY_MISMATCH")
        training_positions.append(rows[:SAMPLES_PER_CLASS])
        validation_positions.append(rows[SAMPLES_PER_CLASS:])
    training_positions_tensor = torch.cat(training_positions).to(dtype=torch.long).contiguous()
    validation_positions_tensor = torch.cat(validation_positions).to(dtype=torch.long).contiguous()
    if torch.isin(training_positions_tensor, validation_positions_tensor).any():
        raise HarnessFailure("DATA_FIREWALL_PARTITION_IDENTITY_MISMATCH")
    if torch.cat((training_positions_tensor, validation_positions_tensor)).unique().numel() != len(reference.fit_indices):
        raise HarnessFailure("DATA_FIREWALL_PARTITION_IDENTITY_MISMATCH")

    training_indices = reference.fit_indices.index_select(0, training_positions_tensor)
    validation_indices = reference.fit_indices.index_select(0, validation_positions_tensor)
    training_labels = reference.fit_labels.index_select(0, training_positions_tensor)
    validation_labels = reference.fit_labels.index_select(0, validation_positions_tensor)
    training_r1 = reference.fit_r1.index_select(0, training_positions_tensor)
    validation_r1 = reference.fit_r1.index_select(0, validation_positions_tensor)
    expected = {
        "training_positions": (tensor_sha256(training_positions_tensor), PARTITION_TRAINING_POSITION_SHA256),
        "internal_validation_positions": (tensor_sha256(validation_positions_tensor), PARTITION_VALIDATION_POSITION_SHA256),
        "training_indices": (tensor_sha256(training_indices), PARTITION_TRAINING_INDEX_SHA256),
        "internal_validation_indices": (tensor_sha256(validation_indices), PARTITION_VALIDATION_INDEX_SHA256),
        "training_labels": (tensor_sha256(training_labels), PARTITION_TRAINING_LABEL_SHA256),
        "internal_validation_labels": (tensor_sha256(validation_labels), PARTITION_VALIDATION_LABEL_SHA256),
        "training_r1": (tensor_sha256(training_r1), PARTITION_TRAINING_R1_SHA256),
        "internal_validation_r1": (tensor_sha256(validation_r1), PARTITION_VALIDATION_R1_SHA256),
    }
    if any(actual != frozen for actual, frozen in expected.values()):
        raise HarnessFailure("DATA_FIREWALL_PARTITION_IDENTITY_MISMATCH")
    identity_material = {
        "schema": PARTITION_SCHEMA,
        "source": {
            "dataset_identity": "mnist_train",
            "semantic_source": "S1_fit_R1_POWER",
            "source_artifact_sha256": reference.source_artifact_sha256,
            "fit_indices_sha256": S1_FIT_INDICES_SHA256,
            "fit_labels_sha256": S1_FIT_LABELS_SHA256,
        },
        "algorithm": PARTITION_ALGORITHM,
        "training": {
            "count": REAL_COUNT,
            "per_class_count": SAMPLES_PER_CLASS,
            "positions_sha256": PARTITION_TRAINING_POSITION_SHA256,
            "indices_sha256": PARTITION_TRAINING_INDEX_SHA256,
            "labels_sha256": PARTITION_TRAINING_LABEL_SHA256,
            "r1_sha256": PARTITION_TRAINING_R1_SHA256,
        },
        "internal_validation": {
            "count": REAL_COUNT,
            "per_class_count": SAMPLES_PER_CLASS,
            "positions_sha256": PARTITION_VALIDATION_POSITION_SHA256,
            "indices_sha256": PARTITION_VALIDATION_INDEX_SHA256,
            "labels_sha256": PARTITION_VALIDATION_LABEL_SHA256,
            "r1_sha256": PARTITION_VALIDATION_R1_SHA256,
        },
        "runtime_resplit": False,
        "reselection": False,
    }
    return FrozenPartition(
        training_positions=training_positions_tensor,
        internal_validation_positions=validation_positions_tensor,
        training_indices=training_indices,
        internal_validation_indices=validation_indices,
        training_labels=training_labels,
        internal_validation_labels=validation_labels,
        training_r1_sha256=PARTITION_TRAINING_R1_SHA256,
        internal_validation_r1_sha256=PARTITION_VALIDATION_R1_SHA256,
        identity_material_sha256=canonical_sha256(identity_material),
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
    if not artifact.is_file():
        raise HarnessFailure(f"target bank artifact is missing: {artifact}")
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
    expected_partition_manifest_file_sha256: str | None = None,
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
    if manifest.get("partition_identity_sha256") != PARTITION_IDENTITY:
        raise HarnessFailure("target bank partition identity mismatch")
    if not isinstance(manifest.get("partition_identity_material_sha256"), str):
        raise HarnessFailure("target bank partition material identity is missing")
    partition_file_sha256 = manifest.get("partition_manifest_file_sha256")
    if expected_partition_manifest_file_sha256 is not None and partition_file_sha256 != expected_partition_manifest_file_sha256:
        raise HarnessFailure("target bank partition manifest binding mismatch")
    if not isinstance(partition_file_sha256, str) or len(partition_file_sha256) != 64:
        raise HarnessFailure("target bank partition manifest file identity is invalid")
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
        "s1_selection_labels_sha256": S1_SELECTION_LABELS_SHA256,
        "s1_confirmation_indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
        "s1_confirmation_labels_sha256": S1_CONFIRMATION_LABELS_SHA256,
        "s2_validation_indices_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s2_validation"], "data_firewall.s2_validation")["indices_sha256"],
        "s2_validation_labels_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s2_validation"], "data_firewall.s2_validation")["labels_sha256"],
    }:
        raise HarnessFailure("target bank protected sample identity binding mismatch")


def _relative_artifact_path(repo_root: str | Path, path: str | Path) -> str:
    root = Path(repo_root).resolve()
    target = Path(path).resolve()
    try:
        return str(target.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(target)


def _authority_config_contract(config: Mapping[str, Any]) -> str:
    firewall = dict(_mapping(config["data_firewall"], "data_firewall"))
    authority = dict(_mapping(firewall["authority"], "data_firewall.authority"))
    for name in (
        "partition_manifest_file_sha256",
        "target_bank_sha256",
        "target_bank_manifest_sha256",
        "data_firewall_manifest_sha256",
    ):
        authority[name] = None
    firewall["authority"] = authority
    return canonical_sha256(
        {
            "schema": config["schema"],
            "task_id": config["task_id"],
            "data_firewall": firewall,
        }
    )


def _authority_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(config["data_firewall"], "data_firewall")["authority"], "data_firewall.authority")


def _require_materialized_sha(value: Any, name: str) -> str:
    if value is None:
        raise HarnessFailure(f"{name} is not materialized")
    return _sha(value, name)


def _partition_manifest_payload(
    reference: S1Reference,
    partition: FrozenPartition,
    *,
    config_path: Path,
    config_file_sha256: str,
    config_contract_sha256: str,
    git_head: str,
) -> dict[str, Any]:
    return {
        "schema": PARTITION_SCHEMA,
        "artifact_kind": "controller semantic geometry deterministic S1-fit partition",
        "source": {
            "dataset_identity": "mnist_train",
            "semantic_source": "S1_fit_R1_POWER",
            "observer_identity": "R1_POWER:z_j=abs(c_j)^2/s_j",
            "source_artifact_sha256": reference.source_artifact_sha256,
            "fit_indices_sha256": S1_FIT_INDICES_SHA256,
            "fit_labels_sha256": S1_FIT_LABELS_SHA256,
            "fit_count": 1000,
            "fit_per_class_count": 100,
        },
        "algorithm": dict(PARTITION_ALGORITHM),
        "runtime_resplit": False,
        "reselection": False,
        "class_order": list(range(CLASS_COUNT)),
        "within_class_order": "source_index_ascending",
        "training_positions": partition.training_positions.tolist(),
        "internal_validation_positions": partition.internal_validation_positions.tolist(),
        "training_indices": partition.training_indices.tolist(),
        "internal_validation_indices": partition.internal_validation_indices.tolist(),
        "training_labels": partition.training_labels.tolist(),
        "internal_validation_labels": partition.internal_validation_labels.tolist(),
        "counts": {
            "training": REAL_COUNT,
            "internal_validation": REAL_COUNT,
            "per_class": SAMPLES_PER_CLASS,
        },
        "hashes": {
            "training_positions_sha256": PARTITION_TRAINING_POSITION_SHA256,
            "internal_validation_positions_sha256": PARTITION_VALIDATION_POSITION_SHA256,
            "training_indices_sha256": PARTITION_TRAINING_INDEX_SHA256,
            "internal_validation_indices_sha256": PARTITION_VALIDATION_INDEX_SHA256,
            "training_labels_sha256": PARTITION_TRAINING_LABEL_SHA256,
            "internal_validation_labels_sha256": PARTITION_VALIDATION_LABEL_SHA256,
            "training_r1_sha256": PARTITION_TRAINING_R1_SHA256,
            "internal_validation_r1_sha256": PARTITION_VALIDATION_R1_SHA256,
        },
        "partition_identity_sha256": PARTITION_IDENTITY,
        "partition_identity_material_sha256": partition.identity_material_sha256,
        "producer": {
            "config_path": _relative_artifact_path(config_path.parent.parent.parent, config_path),
            "config_file_sha256": config_file_sha256,
            "config_contract_sha256": config_contract_sha256,
            "git_head": git_head,
            "module": "ldm_project.v2.application.research.capacity.controller_semantic_geometry_training",
        },
    }


def _validate_partition_manifest(
    partition_payload: Mapping[str, Any],
    reference: S1Reference,
    expected: FrozenPartition,
    *,
    expected_file_sha256: str,
    expected_config_contract_sha256: str,
) -> None:
    if partition_payload.get("schema") != PARTITION_SCHEMA:
        raise HarnessFailure("partition manifest schema is invalid")
    if partition_payload.get("partition_identity_sha256") != PARTITION_IDENTITY:
        raise HarnessFailure("partition manifest semantic identity mismatch")
    if partition_payload.get("partition_identity_material_sha256") != expected.identity_material_sha256:
        raise HarnessFailure("partition manifest canonical material identity mismatch")
    if partition_payload.get("algorithm") != PARTITION_ALGORITHM:
        raise HarnessFailure("partition manifest algorithm identity mismatch")
    if partition_payload.get("runtime_resplit") is not False or partition_payload.get("reselection") is not False:
        raise HarnessFailure("partition manifest permits runtime resplit or reselection")
    source = _mapping(partition_payload.get("source"), "partition manifest source")
    if source.get("dataset_identity") != "mnist_train" or source.get("semantic_source") != "S1_fit_R1_POWER":
        raise HarnessFailure("partition manifest source identity mismatch")
    if source.get("source_artifact_sha256") != reference.source_artifact_sha256:
        raise HarnessFailure("partition manifest source artifact mismatch")
    if partition_payload.get("training_positions") != expected.training_positions.tolist():
        raise HarnessFailure("partition manifest training positions mismatch")
    if partition_payload.get("internal_validation_positions") != expected.internal_validation_positions.tolist():
        raise HarnessFailure("partition manifest internal-validation positions mismatch")
    if partition_payload.get("training_indices") != expected.training_indices.tolist():
        raise HarnessFailure("partition manifest training indices mismatch")
    if partition_payload.get("internal_validation_indices") != expected.internal_validation_indices.tolist():
        raise HarnessFailure("partition manifest internal-validation indices mismatch")
    if partition_payload.get("training_labels") != expected.training_labels.tolist():
        raise HarnessFailure("partition manifest training labels mismatch")
    if partition_payload.get("internal_validation_labels") != expected.internal_validation_labels.tolist():
        raise HarnessFailure("partition manifest internal-validation labels mismatch")
    expected_hashes = {
        "training_positions_sha256": PARTITION_TRAINING_POSITION_SHA256,
        "internal_validation_positions_sha256": PARTITION_VALIDATION_POSITION_SHA256,
        "training_indices_sha256": PARTITION_TRAINING_INDEX_SHA256,
        "internal_validation_indices_sha256": PARTITION_VALIDATION_INDEX_SHA256,
        "training_labels_sha256": PARTITION_TRAINING_LABEL_SHA256,
        "internal_validation_labels_sha256": PARTITION_VALIDATION_LABEL_SHA256,
        "training_r1_sha256": PARTITION_TRAINING_R1_SHA256,
        "internal_validation_r1_sha256": PARTITION_VALIDATION_R1_SHA256,
    }
    if partition_payload.get("hashes") != expected_hashes:
        raise HarnessFailure("partition manifest component hash mismatch")
    producer = _mapping(partition_payload.get("producer"), "partition manifest producer")
    if producer.get("config_contract_sha256") != expected_config_contract_sha256:
        raise HarnessFailure("partition manifest config contract mismatch")
    _sha(expected_file_sha256, "partition manifest file SHA-256")


def _target_bank_manifest(
    config: Mapping[str, Any],
    reference: S1Reference,
    partition: FrozenPartition,
    *,
    partition_manifest_file_sha256: str,
    config_contract_sha256: str,
    git_head: str,
) -> dict[str, Any]:
    training_r1 = reference.fit_r1.index_select(0, partition.training_positions)
    validation_r1 = reference.fit_r1.index_select(0, partition.internal_validation_positions)
    return {
        "schema": TARGET_BANK_MANIFEST_SCHEMA,
        "artifact_kind": "redacted controller semantic geometry R1 target bank",
        "partition_identity_sha256": PARTITION_IDENTITY,
        "partition_identity_material_sha256": partition.identity_material_sha256,
        "partition_manifest_file_sha256": partition_manifest_file_sha256,
        "source_artifact_sha256": reference.source_artifact_sha256,
        "observer_identity": "R1_POWER:z_j=abs(c_j)^2/s_j",
        "dataset_identity": "mnist_train",
        "per_class_count": SAMPLES_PER_CLASS,
        "protected_sample_identities": {
            "s1_selection_indices_sha256": S1_SELECTION_INDICES_SHA256,
            "s1_selection_labels_sha256": S1_SELECTION_LABELS_SHA256,
            "s1_confirmation_indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
            "s1_confirmation_labels_sha256": S1_CONFIRMATION_LABELS_SHA256,
            "s2_validation_indices_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s2_validation"], "data_firewall.s2_validation")["indices_sha256"],
            "s2_validation_labels_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s2_validation"], "data_firewall.s2_validation")["labels_sha256"],
        },
        "training_r1_sha256": tensor_sha256(training_r1),
        "training_labels_sha256": tensor_sha256(partition.training_labels),
        "training_indices_sha256": tensor_sha256(partition.training_indices),
        "internal_validation_r1_sha256": tensor_sha256(validation_r1),
        "internal_validation_labels_sha256": tensor_sha256(partition.internal_validation_labels),
        "internal_validation_indices_sha256": tensor_sha256(partition.internal_validation_indices),
        "producer": {
            "config_contract_sha256": config_contract_sha256,
            "git_head": git_head,
            "module": "ldm_project.v2.application.research.capacity.controller_semantic_geometry_training",
        },
    }


def _build_target_bank_payload(
    reference: S1Reference,
    partition: FrozenPartition,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TARGET_BANK_SCHEMA,
        "manifest": dict(manifest),
        "training": {
            "r1": reference.fit_r1.index_select(0, partition.training_positions),
            "labels": partition.training_labels,
            "indices": partition.training_indices,
        },
        "internal_validation": {
            "r1": reference.fit_r1.index_select(0, partition.internal_validation_positions),
            "labels": partition.internal_validation_labels,
            "indices": partition.internal_validation_indices,
        },
    }


def register_data_authority(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    expected_head: str | None = None,
) -> dict[str, Any]:
    """Write the dedicated immutable data authority without creating an optimizer."""
    root = Path(repo_root).resolve()
    head, dirty = _git_state(root)
    if dirty:
        raise HarnessFailure("clean Git worktree is required for data-authority registration")
    if expected_head is not None and head != expected_head:
        raise HarnessFailure("registration HEAD does not match the expected clean HEAD")
    reference = _load_s1_reference(config, root)
    partition = build_frozen_partition(reference)
    authority = _authority_config(config)
    configured_dir = resolve_path(root, authority["run_directory"]).resolve()
    target_dir = configured_dir if output_dir is None else Path(output_dir).resolve()
    if target_dir != configured_dir:
        raise HarnessFailure("registration output must match the configured dedicated authority run")
    future_dir = resolve_path(root, _mapping(config["output"], "output")["target_directory"]).resolve()
    if target_dir == future_dir or target_dir.is_relative_to(future_dir):
        raise HarnessFailure("data authority cannot be stored in the future training run directory")
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite data-authority run: {target_dir}")

    config_target = Path(config_path).resolve()
    config_file_sha256 = file_sha256(config_target)
    config_contract_sha256 = _authority_config_contract(config)
    target_dir.mkdir(parents=True, exist_ok=False)
    partition_path = resolve_path(root, authority["partition_manifest"])
    target_bank_path = resolve_path(root, authority["target_bank"])
    target_bank_manifest_path = resolve_path(root, authority["target_bank_manifest"])
    data_firewall_path = resolve_path(root, authority["data_firewall_manifest"])
    expected_paths = (partition_path, target_bank_path, target_bank_manifest_path, data_firewall_path)
    if any(not path.resolve().is_relative_to(target_dir) for path in expected_paths):
        raise HarnessFailure("data-authority artifacts must remain inside their dedicated run")

    partition_payload = _partition_manifest_payload(
        reference,
        partition,
        config_path=config_target,
        config_file_sha256=config_file_sha256,
        config_contract_sha256=config_contract_sha256,
        git_head=head,
    )
    atomic_write_new_json(partition_path, partition_payload, artifact_kind="controller semantic geometry partition manifest")
    partition_file_sha256 = file_sha256(partition_path)
    _validate_partition_manifest(
        partition_payload,
        reference,
        partition,
        expected_file_sha256=partition_file_sha256,
        expected_config_contract_sha256=config_contract_sha256,
    )

    target_manifest = _target_bank_manifest(
        config,
        reference,
        partition,
        partition_manifest_file_sha256=partition_file_sha256,
        config_contract_sha256=config_contract_sha256,
        git_head=head,
    )
    target_payload = _build_target_bank_payload(reference, partition, target_manifest)
    validate_target_bank(
        TargetBank(
            target_payload["training"]["r1"],
            target_payload["training"]["labels"],
            target_payload["training"]["indices"],
            target_payload["internal_validation"]["r1"],
            target_payload["internal_validation"]["labels"],
            target_payload["internal_validation"]["indices"],
            target_manifest,
        ),
        reference,
        config,
        expected_partition_manifest_file_sha256=partition_file_sha256,
        repo_root=root,
    )
    atomic_torch_save_new(target_bank_path, target_payload, artifact_kind="controller semantic geometry target bank")
    target_bank_sha256 = file_sha256(target_bank_path)
    target_sidecar_payload = {
        **target_manifest,
        "artifact": {
            "path": _relative_artifact_path(root, target_bank_path),
            "sha256": target_bank_sha256,
            "schema": TARGET_BANK_SCHEMA,
        },
    }
    atomic_write_new_json(
        target_bank_manifest_path,
        target_sidecar_payload,
        artifact_kind="controller semantic geometry target bank manifest",
    )
    target_bank_manifest_sha256 = file_sha256(target_bank_manifest_path)
    reloaded = load_target_bank(target_bank_path, expected_sha256=target_bank_sha256)
    validate_target_bank(
        reloaded,
        reference,
        config,
        expected_artifact_sha256=target_bank_sha256,
        expected_partition_manifest_file_sha256=partition_file_sha256,
        repo_root=root,
    )

    protected = _protected_sample_sets(reference, config, root)
    s3_codes = _load_s3_base_codes(config, root)
    internal_codes = internal_validation_codes()
    stream_blocks, stream_meta = finite_training_stream(HORIZON)
    stream_meta.update(validate_finite_stream_against_protected(stream_blocks, s3_codes, internal_codes))
    data_firewall_payload = {
        "schema": DATA_AUTHORITY_SCHEMA,
        "artifact_kind": "controller semantic geometry immutable data firewall",
        "authority_scope": "CONTROLLER_SEMANTIC_GEOMETRY_R1",
        "source": {
            "dataset_identity": "mnist_train",
            "semantic_source": "S1_fit_R1_POWER",
            "source_artifact_sha256": reference.source_artifact_sha256,
            "semantic_manifest_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s1"], "data_firewall.s1")["semantic_manifest_sha256"],
            "fit_indices_sha256": S1_FIT_INDICES_SHA256,
            "fit_labels_sha256": S1_FIT_LABELS_SHA256,
        },
        "partition": {
            "semantic_identity_sha256": PARTITION_IDENTITY,
            "identity_material_sha256": partition.identity_material_sha256,
            "path": _relative_artifact_path(root, partition_path),
            "file_sha256": partition_file_sha256,
        },
        "target_bank": {
            "path": _relative_artifact_path(root, target_bank_path),
            "file_sha256": target_bank_sha256,
            "manifest_path": _relative_artifact_path(root, target_bank_manifest_path),
            "manifest_file_sha256": target_bank_manifest_sha256,
            "schema": TARGET_BANK_SCHEMA,
            "redacted_top_level_fields": ["schema", "manifest", "training", "internal_validation"],
        },
        "protected_sample_identities": {
            "s1_selection": {
                "dataset_namespace": "mnist_test",
                "indices_sha256": S1_SELECTION_INDICES_SHA256,
                "labels_sha256": S1_SELECTION_LABELS_SHA256,
            },
            "s1_confirmation": {
                "dataset_namespace": "mnist_train",
                "indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
                "labels_sha256": S1_CONFIRMATION_LABELS_SHA256,
            },
            "s2_validation": {
                "dataset_namespace": "mnist_train",
                "indices_sha256": S2_VALIDATION_INDICES_SHA256,
                "labels_sha256": S2_VALIDATION_LABELS_SHA256,
            },
        },
        "protected_code_identities": {
            "s3_base_codes_sha256": tensor_sha256(s3_codes),
            "s3_panel_identity_sha256": _mapping(_mapping(config["data_firewall"], "data_firewall")["s3_panel"], "data_firewall.s3_panel")["panel_identity_sha256"],
            "internal_validation_codes_sha256": tensor_sha256(internal_codes),
        },
        "overlap_counts": {
            "training_samples_s1_selection": 0,
            "training_samples_s1_confirmation": 0,
            "training_samples_s2_validation": 0,
            "internal_validation_samples_s1_selection": 0,
            "internal_validation_samples_s1_confirmation": 0,
            "internal_validation_samples_s2_validation": 0,
            "training_codes_s3_base": stream_meta["s3_overlap_count"],
            "training_codes_internal_validation": stream_meta["internal_validation_overlap_count"],
        },
        "training_code_stream": stream_meta,
        "producer": {
            "config_path": _relative_artifact_path(root, config_target),
            "config_file_sha256": config_file_sha256,
            "config_contract_sha256": config_contract_sha256,
            "git_head": head,
            "module": "ldm_project.v2.application.research.capacity.controller_semantic_geometry_training",
        },
        "registration": {
            "classification": "CONTROLLER_SEMANTIC_GEOMETRY_DATA_AUTHORITY_REGISTERED",
            "optimizer_step_count": 0,
            "backward_count": 0,
            "forward_count": 0,
            "cuda_training": False,
            "write_once": True,
        },
        "consumer": "controller semantic geometry training no-update preflight",
    }
    bank_identity = _identity_set("mnist_train", partition.training_indices) | _identity_set(
        "mnist_train", partition.internal_validation_indices
    )
    if any(bank_identity & protected_set for protected_set in protected.values()):
        raise HarnessFailure("data authority protected sample overlap")
    atomic_write_new_json(data_firewall_path, data_firewall_payload, artifact_kind="controller semantic geometry data firewall manifest")
    data_firewall_manifest_sha256 = file_sha256(data_firewall_path)
    return {
        "classification": "CONTROLLER_SEMANTIC_GEOMETRY_DATA_AUTHORITY_REGISTERED",
        "git_head": head,
        "target_bank": str(target_bank_path),
        "target_bank_sha256": target_bank_sha256,
        "target_bank_manifest": str(target_bank_manifest_path),
        "target_bank_manifest_sha256": target_bank_manifest_sha256,
        "partition_manifest": str(partition_path),
        "partition_manifest_file_sha256": partition_file_sha256,
        "partition_identity_sha256": PARTITION_IDENTITY,
        "partition_identity_material_sha256": partition.identity_material_sha256,
        "data_firewall_manifest": str(data_firewall_path),
        "data_firewall_manifest_sha256": data_firewall_manifest_sha256,
        "optimizer_step_count": 0,
        "backward_count": 0,
        "forward_count": 0,
    }


def register_target_bank(
    config: Mapping[str, Any],
    repo_root: str | Path,
    partition_manifest_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Legacy target-bank-only entrypoint; dedicated authority is mandatory."""
    raise HarnessFailure("target-bank-only registration is disabled; use register-data-authority")


def _load_data_authority(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    target_bank_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the dedicated immutable data-authority root."""
    root = Path(repo_root).resolve()
    authority = _authority_config(config)
    hashes = {
        name: _require_materialized_sha(authority.get(name), f"data_firewall.authority.{name}")
        for name in (
            "partition_manifest_file_sha256",
            "target_bank_sha256",
            "target_bank_manifest_sha256",
            "data_firewall_manifest_sha256",
        )
    }
    run_directory = resolve_path(root, authority["run_directory"]).resolve()
    future_directory = resolve_path(root, _mapping(config["output"], "output")["target_directory"]).resolve()
    if run_directory == future_directory or run_directory.is_relative_to(future_directory):
        raise HarnessFailure("data authority is stored in the future training run directory")
    paths = {
        "partition": resolve_path(root, authority["partition_manifest"]).resolve(),
        "target_bank": resolve_path(root, authority["target_bank"]).resolve(),
        "target_bank_manifest": resolve_path(root, authority["target_bank_manifest"]).resolve(),
        "data_firewall": resolve_path(root, authority["data_firewall_manifest"]).resolve(),
    }
    if any(not path.is_relative_to(run_directory) for path in paths.values()):
        raise HarnessFailure("data-authority artifact escaped its dedicated run directory")
    if target_bank_path is not None and Path(target_bank_path).resolve() != paths["target_bank"]:
        raise HarnessFailure("consumer target-bank path is not the registered dedicated authority")
    if file_sha256(paths["data_firewall"]) != hashes["data_firewall_manifest_sha256"]:
        raise HarnessFailure("data-firewall manifest file hash mismatch")
    with paths["data_firewall"].open("r", encoding="utf-8") as handle:
        firewall_manifest = json.load(handle)
    if not isinstance(firewall_manifest, Mapping) or firewall_manifest.get("schema") != DATA_AUTHORITY_SCHEMA:
        raise HarnessFailure("data-firewall manifest schema is invalid")
    if firewall_manifest.get("authority_scope") != "CONTROLLER_SEMANTIC_GEOMETRY_R1":
        raise HarnessFailure("data-firewall authority scope is invalid")
    if firewall_manifest.get("consumer") != "controller semantic geometry training no-update preflight":
        raise HarnessFailure("data-firewall consumer identity is invalid")
    producer = _mapping(firewall_manifest.get("producer"), "data-firewall producer")
    if producer.get("config_contract_sha256") != _authority_config_contract(config):
        raise HarnessFailure("data-firewall config contract identity mismatch")
    partition_entry = _mapping(firewall_manifest.get("partition"), "data-firewall partition")
    if partition_entry.get("path") != _relative_artifact_path(root, paths["partition"]):
        raise HarnessFailure("data-firewall partition path binding mismatch")
    if partition_entry.get("semantic_identity_sha256") != PARTITION_IDENTITY:
        raise HarnessFailure("data-firewall partition semantic identity mismatch")
    if partition_entry.get("file_sha256") != hashes["partition_manifest_file_sha256"]:
        raise HarnessFailure("data-firewall partition file hash binding mismatch")
    if file_sha256(paths["partition"]) != hashes["partition_manifest_file_sha256"]:
        raise HarnessFailure("partition manifest file hash mismatch")

    reference = _load_s1_reference(config, root)
    expected_partition = build_frozen_partition(reference)
    with paths["partition"].open("r", encoding="utf-8") as handle:
        partition_manifest = json.load(handle)
    _validate_partition_manifest(
        partition_manifest,
        reference,
        expected_partition,
        expected_file_sha256=hashes["partition_manifest_file_sha256"],
        expected_config_contract_sha256=_authority_config_contract(config),
    )
    target_entry = _mapping(firewall_manifest.get("target_bank"), "data-firewall target_bank")
    if target_entry.get("path") != _relative_artifact_path(root, paths["target_bank"]):
        raise HarnessFailure("data-firewall target-bank path binding mismatch")
    if target_entry.get("file_sha256") != hashes["target_bank_sha256"]:
        raise HarnessFailure("data-firewall target-bank hash binding mismatch")
    if target_entry.get("manifest_path") != _relative_artifact_path(root, paths["target_bank_manifest"]):
        raise HarnessFailure("data-firewall target-bank sidecar path binding mismatch")
    if target_entry.get("manifest_file_sha256") != hashes["target_bank_manifest_sha256"]:
        raise HarnessFailure("data-firewall target-bank sidecar hash binding mismatch")
    if not paths["target_bank"].is_file() or file_sha256(paths["target_bank"]) != hashes["target_bank_sha256"]:
        raise HarnessFailure("target-bank file hash mismatch")
    if not paths["target_bank_manifest"].is_file() or file_sha256(paths["target_bank_manifest"]) != hashes["target_bank_manifest_sha256"]:
        raise HarnessFailure("target-bank sidecar hash mismatch")
    with paths["target_bank_manifest"].open("r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    sidecar_artifact = _mapping(sidecar.get("artifact"), "target-bank sidecar artifact")
    if sidecar_artifact.get("sha256") != hashes["target_bank_sha256"] or sidecar_artifact.get("schema") != TARGET_BANK_SCHEMA:
        raise HarnessFailure("target-bank sidecar artifact binding mismatch")

    protected = _mapping(firewall_manifest.get("protected_sample_identities"), "protected sample identities")
    expected_protected = {
        "s1_selection": {
            "dataset_namespace": "mnist_test",
            "indices_sha256": S1_SELECTION_INDICES_SHA256,
            "labels_sha256": S1_SELECTION_LABELS_SHA256,
        },
        "s1_confirmation": {
            "dataset_namespace": "mnist_train",
            "indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
            "labels_sha256": S1_CONFIRMATION_LABELS_SHA256,
        },
        "s2_validation": {
            "dataset_namespace": "mnist_train",
            "indices_sha256": S2_VALIDATION_INDICES_SHA256,
            "labels_sha256": S2_VALIDATION_LABELS_SHA256,
        },
    }
    if protected != expected_protected:
        raise HarnessFailure("protected sample identity binding mismatch")
    code_ids = _mapping(firewall_manifest.get("protected_code_identities"), "protected code identities")
    s3_panel = _mapping(_mapping(config["data_firewall"], "data_firewall")["s3_panel"], "data_firewall.s3_panel")
    if code_ids.get("s3_base_codes_sha256") != S3_BASE_CODE_SHA256 or code_ids.get("s3_panel_identity_sha256") != s3_panel["panel_identity_sha256"]:
        raise HarnessFailure("protected S3 code identity binding mismatch")
    if code_ids.get("internal_validation_codes_sha256") != tensor_sha256(internal_validation_codes()):
        raise HarnessFailure("protected internal-validation code identity mismatch")
    overlap_counts = _mapping(firewall_manifest.get("overlap_counts"), "data-firewall overlap counts")
    if any(value != 0 for value in overlap_counts.values()):
        raise HarnessFailure("protected overlap is nonzero")
    return {
        "manifest": firewall_manifest,
        "reference": reference,
        "partition": partition_manifest,
        "paths": paths,
        "hashes": hashes,
    }


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
    from ldm_project.v2.application.research.optical_manifold.optical_manifold_smf_bogt import FIELD_TRANSMISSION, _attenuate

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


def _controller_clone(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
) -> IdentityAnchoredConditionalAmplitudeGenerator:
    device = next(generator.parameters()).device
    clone = IdentityAnchoredConditionalAmplitudeGenerator().to(device)
    clone.load_state_dict(
        {name: value.detach().clone() for name, value in generator.state_dict().items()},
        strict=True,
    )
    clone.train(generator.training)
    return clone


def _controller_geometry_update_step(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    real_r1: torch.Tensor,
    real_labels: torch.Tensor,
    base_codes: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[GeometryObjectiveResult, dict[str, Any]]:
    parameter_items = controller_parameter_items(generator)
    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in parameter_items
    }
    batch = build_logical_batch(real_r1, real_labels, base_codes)
    controller_r1 = controller_logical_r1(generator, optical_model, batch, scale)
    result = geometry_alignment_loss(
        batch.real_r1.to(real_r1.device),
        controller_r1,
        batch.real_labels.to(real_r1.device),
        batch.controller_labels.to(real_r1.device),
    )
    if not torch.isfinite(result.loss):
        raise HarnessFailure(INVALID_HARNESS)
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    gradient_norms: dict[str, float] = {}
    for name, parameter in parameter_items:
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise HarnessFailure(f"missing or non-finite gradient for {name}")
        gradient_norms[name] = float(torch.linalg.vector_norm(gradient.detach().float()).cpu())
    optimizer.step()
    update_norms: dict[str, float] = {}
    for name, parameter in parameter_items:
        if not torch.isfinite(parameter).all():
            raise HarnessFailure(f"non-finite controller parameter after optimizer step: {name}")
        update_norms[name] = float(
            torch.linalg.vector_norm(parameter.detach() - parameter_before[name]).cpu()
        )
    metrics = {
        "loss": float(result.loss.detach().cpu()),
        "mean_anchor": float(result.mean_anchor.detach().cpu()),
        "mean_covariance": float(result.mean_covariance.detach().cpu()),
        "gradient_norms": gradient_norms,
        "update_norms": update_norms,
        "finite": True,
    }
    return result, metrics


def _controller_geometry_frozen_step(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    real_r1: torch.Tensor,
    real_labels: torch.Tensor,
    base_codes: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, Any]:
    batch = build_logical_batch(real_r1, real_labels, base_codes)
    with torch.no_grad():
        controller_r1 = controller_logical_r1(generator, optical_model, batch, scale)
        result = geometry_alignment_loss(
            batch.real_r1.to(real_r1.device),
            controller_r1,
            batch.real_labels.to(real_r1.device),
            batch.controller_labels.to(real_r1.device),
        )
    if not torch.isfinite(result.loss):
        raise HarnessFailure(INVALID_HARNESS)
    return {
        "loss": float(result.loss.detach().cpu()),
        "mean_anchor": float(result.mean_anchor.detach().cpu()),
        "mean_covariance": float(result.mean_covariance.detach().cpu()),
        "finite": True,
    }


def training_update_preflight(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    target_bank: TargetBank,
    scale: torch.Tensor,
    *,
    authorization: str | None,
    frozen_file_paths: Mapping[str, str | Path],
    s3_base_codes: torch.Tensor,
    max_peak_cuda_bytes: int,
) -> dict[str, Any]:
    """Run the fixed CUDA representative update and its frozen-control check."""
    if authorization != AUTHORIZATION_TOKEN:
        raise HarnessFailure("explicit one-shot training authorization is required")
    if max_peak_cuda_bytes <= 0:
        raise HarnessFailure("preflight memory authority is invalid")
    device = next(generator.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise HarnessFailure("BLOCKED_CUDA_UNAVAILABLE")
    freeze_optical_model(optical_model)
    for _, parameter in controller_parameter_items(generator):
        parameter.requires_grad_(False)
    original_before = capture_state_snapshot(generator, optical_model, frozen_file_paths)
    real_r1 = target_bank.training_r1.to(device)
    real_labels = target_bank.training_labels.to(device)
    real_means, real_covariances = class_moments(real_r1, real_labels)
    _validate_real_moments(real_means, real_covariances)
    stream = TrainingCodeStream()
    stream_blocks = torch.stack(
        [stream.next_block() for _ in range(TRAINING_PREFLIGHT_STEPS)],
        dim=0,
    )
    stream_meta = validate_finite_stream_against_protected(
        stream_blocks,
        s3_base_codes,
        internal_validation_codes(),
    )
    update_generator = _controller_clone(generator)
    for _, parameter in controller_parameter_items(update_generator):
        parameter.requires_grad_(True)
    optimizer = build_controller_optimizer(update_generator)
    if optimizer_parameter_names(optimizer, update_generator) != CONTROLLER_PARAMETER_NAMES:
        raise HarnessFailure("controller optimizer coverage failed")
    update_before = capture_state_snapshot(update_generator, optical_model, frozen_file_paths)
    update_history: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, TRAINING_PREFLIGHT_STEPS + 1):
        _, metrics = _controller_geometry_update_step(
            update_generator,
            optical_model,
            optimizer,
            real_r1,
            real_labels,
            stream_blocks[step - 1].to(device),
            scale.to(device=device, dtype=torch.float32),
        )
        metrics["step"] = step
        update_history.append(metrics)
    update_after = capture_state_snapshot(update_generator, optical_model, frozen_file_paths)
    validate_frozen_state(update_before, update_after, update_generator, optical_model, optimizer)

    frozen_generator = _controller_clone(generator)
    for _, parameter in controller_parameter_items(frozen_generator):
        parameter.requires_grad_(False)
    frozen_generator.eval()
    frozen_before = capture_state_snapshot(frozen_generator, optical_model, frozen_file_paths)
    frozen_history = [
        {
            "step": step,
            **_controller_geometry_frozen_step(
                frozen_generator,
                optical_model,
                real_r1,
                real_labels,
                stream_blocks[step - 1].to(device),
                scale.to(device=device, dtype=torch.float32),
            ),
        }
        for step in range(1, TRAINING_PREFLIGHT_STEPS + 1)
    ]
    frozen_after = capture_state_snapshot(frozen_generator, optical_model, frozen_file_paths)
    original_after = capture_state_snapshot(generator, optical_model, frozen_file_paths)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    update_gradients_nonzero = all(
        all(value > 0.0 for value in step["gradient_norms"].values())
        for step in update_history
    )
    update_norms_nonzero = all(
        all(value > 0.0 for value in step["update_norms"].values())
        for step in update_history
    )
    update_state_changed = update_before.controller_state_hash != update_after.controller_state_hash
    frozen_control_unchanged = (
        frozen_before.controller_state_hash == frozen_after.controller_state_hash
        and frozen_before.frozen_model_state_hash == frozen_after.frozen_model_state_hash
        and frozen_before.frozen_parameter_hashes == frozen_after.frozen_parameter_hashes
        and frozen_before.file_hashes == frozen_after.file_hashes
    )
    original_unchanged = (
        original_before.controller_state_hash == original_after.controller_state_hash
        and original_before.frozen_model_state_hash == original_after.frozen_model_state_hash
        and original_before.frozen_parameter_hashes == original_after.frozen_parameter_hashes
        and original_before.file_hashes == original_after.file_hashes
    )
    frozen_optical_unchanged = (
        update_before.frozen_model_state_hash == update_after.frozen_model_state_hash
        and update_before.frozen_parameter_hashes == update_after.frozen_parameter_hashes
        and update_before.file_hashes == update_after.file_hashes
    )
    memory_passed = peak_cuda_bytes <= max_peak_cuda_bytes
    wall_passed = elapsed <= TRAINING_PREFLIGHT_MAX_SECONDS
    passed = all(
        (
            all(step["finite"] for step in update_history),
            all(step["finite"] for step in frozen_history),
            update_gradients_nonzero,
            update_norms_nonzero,
            update_state_changed,
            frozen_control_unchanged,
            original_unchanged,
            frozen_optical_unchanged,
            memory_passed,
            wall_passed,
        )
    )
    return {
        "schema": TRAINING_PREFLIGHT_SCHEMA,
        "gate": TRAINING_PREFLIGHT_GATE if passed else INVALID_HARNESS,
        "status": "passed" if passed else INVALID_HARNESS,
        "passed": passed,
        "objective_family": OBJECTIVE_FAMILY,
        "authorization": AUTHORIZATION_TOKEN,
        "preflight_steps": TRAINING_PREFLIGHT_STEPS,
        "authorized_horizon": HORIZON,
        "optimizer_contract_sha256": OPTIMIZER_CONTRACT_SHA256,
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "training_code_stream": {
            **stream_meta,
            "shape": list(stream_blocks.shape),
            "first_block_sha256": tensor_sha256(stream_blocks[0]),
            "finite_training_stream_sha256": tensor_sha256(stream_blocks),
        },
        "update_arm": {
            "optimizer_parameter_names": list(CONTROLLER_PARAMETER_NAMES),
            "optimizer_step_count": TRAINING_PREFLIGHT_STEPS,
            "controller_state_hash_before": update_before.controller_state_hash,
            "controller_state_hash_after": update_after.controller_state_hash,
            "state_changed": update_state_changed,
            "gradient_nonzero": update_gradients_nonzero,
            "update_nonzero": update_norms_nonzero,
            "steps": update_history,
        },
        "frozen_controller_negative_control": {
            "kind": "frozen_controller_same_batches",
            "expected_decision": "reject_primary_update",
            "controller_state_hash_before": frozen_before.controller_state_hash,
            "controller_state_hash_after": frozen_after.controller_state_hash,
            "unchanged": frozen_control_unchanged,
            "steps": frozen_history,
            "passed": frozen_control_unchanged,
        },
        "state_isolation": {
            "original_controller_state_hash_before": original_before.controller_state_hash,
            "original_controller_state_hash_after": original_after.controller_state_hash,
            "original_controller_unchanged": original_unchanged,
            "frozen_optical_unchanged": frozen_optical_unchanged,
            "frozen_model_state_hash_before": update_before.frozen_model_state_hash,
            "frozen_model_state_hash_after": update_after.frozen_model_state_hash,
            "frozen_parameter_hashes_before": dict(update_before.frozen_parameter_hashes),
            "frozen_parameter_hashes_after": dict(update_after.frozen_parameter_hashes),
            "frozen_file_hashes_before": dict(update_before.file_hashes),
            "frozen_file_hashes_after": dict(update_after.file_hashes),
        },
        "resource": {
            "device": str(device),
            "peak_cuda_bytes": peak_cuda_bytes,
            "max_peak_cuda_bytes": max_peak_cuda_bytes,
            "memory_passed": memory_passed,
            "wall_clock_seconds": elapsed,
            "max_wall_clock_seconds": TRAINING_PREFLIGHT_MAX_SECONDS,
            "wall_passed": wall_passed,
        },
        "evaluator_calls": 0,
        "decoder_entered": False,
        "checkpoint_writes": 0,
    }


def validate_training_preflight(
    path: str | Path,
    *,
    expected_git_head: str,
    expected_config_sha256: str,
    expected_s3_config_sha256: str,
    expected_target_bank_sha256: str,
    expected_data_firewall_manifest_sha256: str,
    expected_s3_base_code_sha256: str,
    expected_internal_validation_code_sha256: str,
    expected_max_peak_cuda_bytes: int,
    frozen_file_paths: Mapping[str, str | Path],
) -> str:
    evidence_path = Path(path)
    if not evidence_path.is_file():
        raise HarnessFailure(f"training preflight evidence is missing: {evidence_path}")
    with evidence_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise HarnessFailure("training preflight evidence is not a mapping")
    if (
        payload.get("schema") != TRAINING_PREFLIGHT_SCHEMA
        or payload.get("gate") != TRAINING_PREFLIGHT_GATE
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("authorization") != AUTHORIZATION_TOKEN
        or payload.get("preflight_steps") != TRAINING_PREFLIGHT_STEPS
        or payload.get("authorized_horizon") != HORIZON
        or payload.get("objective_family") != OBJECTIVE_FAMILY
        or payload.get("optimizer_contract_sha256") != OPTIMIZER_CONTRACT_SHA256
        or payload.get("objective_contract_sha256") != OBJECTIVE_CONTRACT_SHA256
    ):
        raise HarnessFailure("training preflight evidence is stale or unauthorized")
    for name, expected in (
        ("git_head", expected_git_head),
        ("config_sha256", expected_config_sha256),
        ("s3_config_sha256", expected_s3_config_sha256),
        ("target_bank_sha256", expected_target_bank_sha256),
        ("data_firewall_manifest_sha256", expected_data_firewall_manifest_sha256),
        ("s3_base_code_sha256", expected_s3_base_code_sha256),
        ("internal_validation_code_sha256", expected_internal_validation_code_sha256),
    ):
        if payload.get(name) != expected:
            raise HarnessFailure(f"training preflight {name} binding mismatch")
    update_arm = _mapping(payload.get("update_arm"), "training preflight update arm")
    if payload.get("task_id") != "H-LOMA-GEN-S3-CONTROLLER-SEMANTIC-GEOMETRY-REPAIR":
        raise HarnessFailure("training preflight task identity mismatch")
    if update_arm.get("optimizer_parameter_names") != list(CONTROLLER_PARAMETER_NAMES):
        raise HarnessFailure("training preflight optimizer parameter boundary mismatch")
    update_steps = update_arm.get("steps")
    if not isinstance(update_steps, list) or any(not isinstance(step, Mapping) for step in update_steps):
        raise HarnessFailure("training preflight update steps are invalid")
    if (
        update_arm.get("optimizer_step_count") != TRAINING_PREFLIGHT_STEPS
        or update_arm.get("state_changed") is not True
        or update_arm.get("gradient_nonzero") is not True
        or update_arm.get("update_nonzero") is not True
        or len(update_steps) != TRAINING_PREFLIGHT_STEPS
        or any(
            step.get("finite") is not True
            or any(float(value) <= 0.0 for value in step.get("gradient_norms", {}).values())
            or any(float(value) <= 0.0 for value in step.get("update_norms", {}).values())
            for step in update_steps
        )
    ):
        raise HarnessFailure("training preflight optimizer step count mismatch")
    stream = _mapping(payload.get("training_code_stream"), "training preflight code stream")
    if stream.get("s3_overlap_count") != 0 or stream.get("internal_validation_overlap_count") != 0:
        raise HarnessFailure("training preflight code firewall is invalid")
    negative = _mapping(
        payload.get("frozen_controller_negative_control"),
        "training preflight frozen-controller negative control",
    )
    if (
        negative.get("expected_decision") != "reject_primary_update"
        or negative.get("unchanged") is not True
        or negative.get("passed") is not True
    ):
        raise HarnessFailure("training preflight frozen-controller negative control is invalid")
    isolation = _mapping(payload.get("state_isolation"), "training preflight state isolation")
    if any(
        isolation.get(name) is not True
        for name in ("original_controller_unchanged", "frozen_optical_unchanged")
    ):
        raise HarnessFailure("training preflight state isolation is invalid")
    expected_files = {name: file_sha256(file_path) for name, file_path in frozen_file_paths.items()}
    if isolation.get("frozen_file_hashes_before") != expected_files or isolation.get("frozen_file_hashes_after") != expected_files:
        raise HarnessFailure("training preflight frozen file identity changed")
    if payload.get("evaluator_calls") != 0 or payload.get("decoder_entered") is not False or payload.get("checkpoint_writes") != 0:
        raise HarnessFailure("training preflight crossed a prohibited boundary")
    resource = _mapping(payload.get("resource"), "training preflight resource")
    resource_device = resource.get("device")
    if (
        resource_device not in ("cuda", "cuda:0")
        or resource.get("max_peak_cuda_bytes") != expected_max_peak_cuda_bytes
        or int(resource.get("peak_cuda_bytes", -1)) > expected_max_peak_cuda_bytes
        or resource.get("memory_passed") is not True
        or resource.get("wall_passed") is not True
    ):
        raise HarnessFailure("training preflight resource gate is invalid")
    return file_sha256(evidence_path)


def train_authorized_controller(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    optical_model: torch.nn.Module,
    target_bank: TargetBank,
    scale: torch.Tensor,
    output_path: str | Path,
    *,
    authorization: str | None,
    frozen_file_paths: Mapping[str, str | Path] | None = None,
    preflight_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    if authorization != AUTHORIZATION_TOKEN:
        raise HarnessFailure("explicit one-shot training authorization is required")
    if preflight_evidence_sha256 is None:
        raise HarnessFailure("passed 3-step training preflight evidence is required")
    output = Path(output_path)
    if output.exists():
        raise HarnessFailure("training output target is not a fresh write-once path")
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
        last_result, _ = _controller_geometry_update_step(
            generator,
            optical_model,
            optimizer,
            real_r1,
            real_labels,
            base_codes,
            scale.to(real_r1.device),
        )
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
        "preflight_evidence_sha256": preflight_evidence_sha256,
        "training_authorized": True,
        "formal_promotion": False,
        "s3_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
        "mean_anchor": float(last_result.mean_anchor.detach().cpu()),
        "mean_covariance": float(last_result.mean_covariance.detach().cpu()),
        "loss": float(last_result.loss.detach().cpu()),
    }
    atomic_torch_save_new(output, payload, artifact_kind="controller semantic geometry final checkpoint")
    return {"checkpoint": str(output), "checkpoint_sha256": file_sha256(output), **{key: value for key, value in payload.items() if key != "controller_state"}}


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
    authority = _load_data_authority(config, repo_root, target_bank_path=target_bank_path)
    reference = authority["reference"]
    bank = load_target_bank(target_bank_path, expected_sha256=authority["hashes"]["target_bank_sha256"])
    validate_target_bank(
        bank,
        reference,
        config,
        expected_artifact_sha256=authority["hashes"]["target_bank_sha256"],
        expected_partition_manifest_file_sha256=authority["hashes"]["partition_manifest_file_sha256"],
        repo_root=repo_root,
    )
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
        "partition_manifest_file_sha256": authority["hashes"]["partition_manifest_file_sha256"],
        "data_firewall_manifest_sha256": authority["hashes"]["data_firewall_manifest_sha256"],
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
    "DATA_AUTHORITY_SCHEMA",
    "FrozenPartition",
    "GeometryObjectiveResult",
    "HarnessFailure",
    "HORIZON",
    "INTERNAL_VALIDATION_CODE_SEED",
    "LogicalGeometryBatch",
    "OBJECTIVE_CONTRACT_SHA256",
    "OBJECTIVE_FAMILY",
    "OPTIMIZER_CONTRACT_SHA256",
    "PARTITION_IDENTITY",
    "PARTITION_TRAINING_INDEX_SHA256",
    "PARTITION_TRAINING_POSITION_SHA256",
    "PARTITION_TRAINING_R1_SHA256",
    "PARTITION_VALIDATION_INDEX_SHA256",
    "PARTITION_VALIDATION_POSITION_SHA256",
    "PARTITION_VALIDATION_R1_SHA256",
    "PREFLIGHT_PASS",
    "TRAINING_PREFLIGHT_GATE",
    "TRAINING_PREFLIGHT_SCHEMA",
    "TRAINING_PREFLIGHT_STEPS",
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
    "build_frozen_partition",
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
    "register_data_authority",
    "resolve_path",
    "train_authorized_controller",
    "training_update_preflight",
    "validate_training_preflight",
    "validate_config",
    "validate_finite_stream_against_protected",
    "validate_frozen_state",
    "validate_target_bank",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
