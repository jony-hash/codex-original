"""Materialize the bounded V4 fixed-OLS generator-recoverability contract.

This definition step is read-only with respect to scientific state.  It binds
the completed V3 evidence, the exact V3 control encoder checkpoint, and the
fresh zero-phase generator identity before a separate three-update engineering
preflight may run.  It does not authorize scientific training or downstream
stages.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    LOGICAL_BATCH_SIZE,
    PRODUCTION_PROPAGATION_GRID,
    STAGE_A_STEPS,
    CHECKPOINT_STEPS,
    file_sha256,
    git_facts,
    load_v2_training_contract,
    tensor_sha256,
    LightGen2DComplexOLSTrainingConsumerV2,
)
from ldm_project.v2.evidence import atomic_write_new_json, safe_torch_load


TASK_ID = "DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4"
CONTRACT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_contract.v4"
RESULT_CLASSIFICATION = "DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4_READY"
OUTER_GATE = "FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED"
STAGE_A_FIREWALL = 0.937

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "383197a930dd48f98294120f8cf867444f994f85"

V3_CONTRACT_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract_v3_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json"
)
V3_CONTRACT_SHA256 = "e537fbf4b856519a71e48c16a48c149fadd327e2ad12db84f64c953413fb53f6"
V3_SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json"
)
V3_SCIENTIFIC_REPORT_SHA256 = "8d7930449a1872e685ad39dafa499d89fdcc1d6bb5a9317e7ba06c554f54f31a"
V3_SCIENTIFIC_INPUT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_input.v1.json"
)
V3_SCIENTIFIC_INPUT_SHA256 = "5d8ddb2918dd6706d0a61bf0ce150b17d9ae1c8801af9695dbeeceae5f970943"
V3_SCIENTIFIC_LINEAGE_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_lineage.v1.json"
)
V3_SCIENTIFIC_LINEAGE_SHA256 = "00127edfad105e077342a8eb270447ea577cd4c48b816fb0435c0e6edbfc237d"
V3_ATTRIBUTION_REPORT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_report.v1.json"
)
V3_ATTRIBUTION_REPORT_SHA256 = "b3175da21e50448c5e6eeb0c2c36042f6b620c2510865c5059de26eb43277393"
V3_ATTRIBUTION_INPUT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_input.v1.json"
)
V3_ATTRIBUTION_INPUT_SHA256 = "0c4945ca91641e89de44af3c131106d5b8fa55d42dbe842244b57e1b40ced794"
V3_ATTRIBUTION_LINEAGE_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_lineage.v1.json"
)
V3_ATTRIBUTION_LINEAGE_SHA256 = "21c095047ce119f69795cb543acb5ac5fac5199380c383024ec8e6638cda9be5"

V3_CONTROL_CHECKPOINT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
V3_CONTROL_CHECKPOINT_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"
V3_CONTROL_ENCODER_PHASE_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V3_CONTROL_GENERATOR_PHASE_SHA256 = "1bb3a3e6b641aae69eef5c22614326795bc300fc26c91b6569fbf742df1c351c"
V3_CONTROL_OLS_TRANSPORT_SHA256 = "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293"

V2_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
ALIGNED_RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNED_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
EVALUATOR_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
SMOOTH_CHECKPOINT_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
FROZEN_CHECKPOINT_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
FROZEN_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"

INITIAL_RUNTIME_STATE_SHA256 = "0f90086cf26a179905a27a5a519f7599de256e4f4225f7798f24aeed086eb813"
INITIAL_RNG_STATE_SHA256 = "6241d4343843975f8306769ff6ff05c0cb6af5b86956b545892592fc0e65ea30"
PHASE_SHAPE = [2, 1, 100, 100]
PHASE_DTYPE = "torch.float32"
OLS_TRANSPORT_SHAPE = [100, 100]
OLS_TRANSPORT_DTYPE = "torch.complex64"
GENERATOR_PARAM_COUNT = 20_000
ENCODER_PARAM_COUNT = 20_000
OLS_FROZEN_STATE_NUMEL = 2_010_000
LOGICAL_BATCH = 10
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
LEARNING_RATE = 1.0e-3
ADAM_BETAS = [0.9, 0.999]
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
DETERMINISTIC_RTOL = 1.0e-5
DETERMINISTIC_ATOL = 1.0e-6

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract_v4_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json"


class V4DefinitionError(RuntimeError):
    """Raised when a V4 authority or contract invariant is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V4DefinitionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute path is not allowed at trust boundary: {target}")
    return (root / target).resolve()


def _read_json(root: Path, relative: str | Path, expected_sha256: str, name: str) -> dict[str, Any]:
    path = _path(root, relative)
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected_sha256, f"{name} SHA mismatch: expected {expected_sha256}, observed {observed}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4DefinitionError(f"{name} is not valid JSON") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _ref(relative: str | Path, sha256: str, role: str, **extra: Any) -> dict[str, Any]:
    return {"path": Path(relative).as_posix(), "sha256": sha256, "role": role, **extra}


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _digest_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if torch.is_tensor(item):
            digest.update(prefix.encode("utf-8"))
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{prefix}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")
        else:
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))

    visit(value, "root")
    return digest.hexdigest()


def _validate_control_checkpoint(root: Path) -> dict[str, Any]:
    path = _path(root, V3_CONTROL_CHECKPOINT_PATH)
    _require(path.is_file(), f"V3 control checkpoint is unavailable: {path}")
    _require(file_sha256(path) == V3_CONTROL_CHECKPOINT_SHA256, "V3 control checkpoint SHA mismatch")
    payload = _mapping(
        safe_torch_load(path, expected_sha256=V3_CONTROL_CHECKPOINT_SHA256, map_location="cpu"),
        "V3 control checkpoint",
    )
    _require(payload.get("schema") == "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_checkpoint.v1", "V3 control checkpoint schema changed")
    _require(payload.get("checkpoint_step") == 300 and payload.get("arm") == "control", "V3 control endpoint changed")
    _require(payload.get("scientific_training_checkpoint") is True, "V3 control checkpoint is not a scientific checkpoint")
    _require(payload.get("training_authorized") is False, "V3 control checkpoint opened a future training boundary")
    for key in ("stage_b_executed", "generation_executed", "formal_promotion"):
        _require(payload.get(key) is False, f"V3 control checkpoint crossed forbidden boundary: {key}")
    _require(payload.get("base_initial_state_digest") == INITIAL_RUNTIME_STATE_SHA256, "V3 initial-state identity changed")
    runtime_state = _mapping(payload.get("runtime_state"), "V3 control runtime state")
    expected_keys = {
        "encoder.phase",
        "generator.phase",
        "ols.operator.input_modes",
        "ols.operator.output_modes",
        "ols.operator.transport",
    }
    _require(set(runtime_state) == expected_keys, "V3 control runtime state keys changed")
    encoder = runtime_state["encoder.phase"]
    generator = runtime_state["generator.phase"]
    transport = runtime_state["ols.operator.transport"]
    _require(torch.is_tensor(encoder) and list(encoder.shape) == PHASE_SHAPE and str(encoder.dtype) == PHASE_DTYPE, "V3 control encoder phase shape/dtype changed")
    _require(torch.is_tensor(generator) and list(generator.shape) == PHASE_SHAPE and str(generator.dtype) == PHASE_DTYPE, "V3 control generator phase shape/dtype changed")
    _require(torch.is_tensor(transport) and list(transport.shape) == OLS_TRANSPORT_SHAPE and str(transport.dtype) == OLS_TRANSPORT_DTYPE, "V3 control OLS transport shape/dtype changed")
    _require(tensor_sha256(encoder) == V3_CONTROL_ENCODER_PHASE_SHA256, "V3 control encoder tensor changed")
    _require(tensor_sha256(generator) == V3_CONTROL_GENERATOR_PHASE_SHA256, "V3 control generator tensor changed")
    _require(tensor_sha256(transport) == V3_CONTROL_OLS_TRANSPORT_SHA256, "V3 control OLS transport changed")
    _require(torch.equal(transport, torch.eye(100, dtype=torch.complex64)), "V3 control transport is not T=I_100")
    data_roles = _mapping(payload.get("data_roles"), "V3 control data roles")
    _require(data_roles.get("train") is True and data_roles.get("development") is False, "V3 control TRAIN/DEV roles changed")
    for key in ("protected", "test", "sealed"):
        _require(data_roles.get(key) is False, f"V3 control opened forbidden data role: {key}")
    return {
        "checkpoint": _ref(V3_CONTROL_CHECKPOINT_PATH, V3_CONTROL_CHECKPOINT_SHA256, "V3 CONTROL step-300 checkpoint", step=300),
        "runtime_state_keys": sorted(runtime_state),
        "encoder_phase": {"shape": PHASE_SHAPE, "dtype": PHASE_DTYPE, "tensor_sha256": V3_CONTROL_ENCODER_PHASE_SHA256},
        "generator_phase": {"shape": PHASE_SHAPE, "dtype": PHASE_DTYPE, "tensor_sha256": V3_CONTROL_GENERATOR_PHASE_SHA256, "not_reused": True},
        "ols_transport": {"shape": OLS_TRANSPORT_SHAPE, "dtype": OLS_TRANSPORT_DTYPE, "tensor_sha256": V3_CONTROL_OLS_TRANSPORT_SHA256, "identity": "T=I_100"},
        "base_initial_state_sha256": payload["base_initial_state_digest"],
        "data_roles": {key: data_roles.get(key) for key in ("train", "development", "protected", "test", "sealed")},
    }


def validate_v4_authorities(root: Path) -> dict[str, Any]:
    """Validate all immutable V3/V2 authorities used by the V4 contract."""

    git = git_facts(root)
    _require(git.get("branch") == EXPECTED_BRANCH, f"active branch differs from frozen plan: {git.get('branch')}")
    _require(git.get("head") == EXPECTED_HEAD, f"active HEAD differs from frozen plan: {git.get('head')}")

    v3_contract = _read_json(root, V3_CONTRACT_PATH, V3_CONTRACT_SHA256, "V3 contract")
    _require(v3_contract.get("task_id") == "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3", "V3 contract identity changed")
    _require(v3_contract.get("result") == "READY" and v3_contract.get("status") == "valid_completed", "V3 contract is not READY")
    _require(v3_contract.get("training_authorized") is False, "V3 contract opened training")
    v3_governance = _mapping(v3_contract.get("governance"), "V3 governance")
    _require(v3_governance.get("formal_promotion") is False and v3_governance.get("generation_readiness") == "not_ready", "V3 contract opened promotion/generation")
    _require(v3_contract.get("gate") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED", "V3 gate identity changed")

    v3_report = _read_json(root, V3_SCIENTIFIC_REPORT_PATH, V3_SCIENTIFIC_REPORT_SHA256, "V3 scientific report")
    _require(v3_report.get("result") == "VALID_COMPLETED" and v3_report.get("status") == "valid_completed", "V3 scientific result is not valid completed")
    _require(v3_report.get("classification") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED", "V3 scientific classification changed")
    _require(v3_report.get("outer_gate") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED", "V3 scientific outer gate changed")
    initial_state = _mapping(v3_report.get("initial_state"), "V3 scientific initial state")
    _require(initial_state.get("runtime_state_sha256") == INITIAL_RUNTIME_STATE_SHA256, "V3 original runtime identity changed")
    _require(initial_state.get("rng_state_sha256") == INITIAL_RNG_STATE_SHA256 and initial_state.get("zero_phase") is True, "V3 original zero-phase identity changed")
    manifest = _mapping(initial_state.get("parameter_manifest"), "V3 initial parameter manifest")
    _require(manifest.get("parameter_shapes") == {"encoder.phase": PHASE_SHAPE, "generator.phase": PHASE_SHAPE}, "V3 phase shapes changed")
    _require(manifest.get("parameter_dtypes") == {"encoder.phase": PHASE_DTYPE, "generator.phase": PHASE_DTYPE}, "V3 phase dtypes changed")
    _require(manifest.get("total_parameter_count") == 40_000 and manifest.get("frozen_state_numel") == OLS_FROZEN_STATE_NUMEL, "V3 parameter manifest changed")
    decision = _mapping(v3_report.get("decision"), "V3 decision")
    _require(decision.get("classification") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED", "V3 decision classification changed")
    _require(float(decision.get("delta_smooth")) == 0.003999999910593036, "V3 Smooth endpoint changed")
    _require(float(decision.get("image_loss_delta_candidate_minus_control")) == 0.005275583900511263, "V3 image-loss endpoint changed")
    firewall = _mapping(v3_report.get("stage_a_firewall"), "V3 Stage-A firewall")
    _require(float(firewall.get("threshold")) == STAGE_A_FIREWALL and firewall.get("passed_at_recorded_checkpoint") is False, "V3 firewall boundary changed")

    v3_input = _read_json(root, V3_SCIENTIFIC_INPUT_PATH, V3_SCIENTIFIC_INPUT_SHA256, "V3 scientific input")
    _require(v3_input.get("task_id") == "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3", "V3 scientific input identity changed")
    v3_lineage = _read_json(root, V3_SCIENTIFIC_LINEAGE_PATH, V3_SCIENTIFIC_LINEAGE_SHA256, "V3 scientific lineage")
    _require(v3_lineage.get("result") == "VALID_COMPLETED" and v3_lineage.get("classification") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED", "V3 scientific lineage changed")

    attribution = _read_json(root, V3_ATTRIBUTION_REPORT_PATH, V3_ATTRIBUTION_REPORT_SHA256, "V3 failure-attribution report")
    _require(attribution.get("result") == "VALID_COMPLETED" and attribution.get("status") == "valid_completed", "V3 failure-attribution result is not valid completed")
    _require(attribution.get("classification") == "PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED", "V3 failure-attribution classification changed")
    _require(attribution.get("outer_gate") == "V3_OBJECTIVE_FAILURE_ATTRIBUTION_RESOLVED", "V3 failure-attribution outer gate changed")
    attribution_input = _read_json(root, V3_ATTRIBUTION_INPUT_PATH, V3_ATTRIBUTION_INPUT_SHA256, "V3 failure-attribution input")
    _require(attribution_input.get("task_id") == "RESOLVE_V3_OBJECTIVE_CORRECTION_FAILURE_ATTRIBUTION_V1", "V3 attribution input identity changed")
    attribution_lineage = _read_json(root, V3_ATTRIBUTION_LINEAGE_PATH, V3_ATTRIBUTION_LINEAGE_SHA256, "V3 failure-attribution lineage")
    _require(attribution_lineage.get("result") == "VALID_COMPLETED" and attribution_lineage.get("classification") == "PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED", "V3 failure-attribution lineage changed")

    v2_contract = load_v2_training_contract(_path(root, V2_CONTRACT_PATH), repo_root=root)
    _require(v2_contract.sha256 == V2_CONTRACT_SHA256, "corrected V2 contract SHA binding changed")
    _require(_path(root, TRAINING_SOURCE_PATH).is_file() and file_sha256(_path(root, TRAINING_SOURCE_PATH)) == TRAINING_SOURCE_SHA256, "canonical TRAIN source changed")
    _require(_path(root, ALIGNED_RUNTIME_PATH).is_file() and file_sha256(_path(root, ALIGNED_RUNTIME_PATH)) == ALIGNED_RUNTIME_SHA256, "aligned LightGen runtime changed")
    _require(_path(root, EVALUATOR_PATH).is_file() and file_sha256(_path(root, EVALUATOR_PATH)) == EVALUATOR_SHA256, "evaluator implementation changed")
    _require(_path(root, SMOOTH_CHECKPOINT_PATH).is_file() and file_sha256(_path(root, SMOOTH_CHECKPOINT_PATH)) == SMOOTH_CHECKPOINT_SHA256, "Smooth evaluator checkpoint changed")
    _require(_path(root, FROZEN_CHECKPOINT_PATH).is_file() and file_sha256(_path(root, FROZEN_CHECKPOINT_PATH)) == FROZEN_CHECKPOINT_SHA256, "Frozen evaluator checkpoint changed")

    control = _validate_control_checkpoint(root)

    # The control checkpoint's encoder is the only starting optical encoder;
    # the generator is explicitly reset to the original zero-phase identity.
    zero_generator = torch.zeros(PHASE_SHAPE, dtype=torch.float32)
    zero_generator_sha256 = tensor_sha256(zero_generator)
    fresh = LightGen2DComplexOLSTrainingConsumerV2(v2_contract, device="cpu", propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    fresh_state = {key: value.detach().cpu().clone() for key, value in fresh.runtime.state_dict().items()}
    fresh_state_sha256 = _digest_tree(fresh_state)
    _require(fresh_state_sha256 == INITIAL_RUNTIME_STATE_SHA256, "fresh runtime does not reproduce the V3 original zero-phase identity")
    _require(torch.equal(fresh_state["generator.phase"], zero_generator), "fresh generator is not the original zero-phase generator")
    _require(tensor_sha256(fresh_state["generator.phase"]) == zero_generator_sha256, "zero-phase generator tensor binding changed")

    return {
        "git": git,
        "v3_contract": _ref(V3_CONTRACT_PATH, V3_CONTRACT_SHA256, "completed V3 objective-correction contract"),
        "v3_scientific": {
            "report": _ref(V3_SCIENTIFIC_REPORT_PATH, V3_SCIENTIFIC_REPORT_SHA256, "completed V3 scientific report"),
            "input": _ref(V3_SCIENTIFIC_INPUT_PATH, V3_SCIENTIFIC_INPUT_SHA256, "V3 scientific input"),
            "lineage": _ref(V3_SCIENTIFIC_LINEAGE_PATH, V3_SCIENTIFIC_LINEAGE_SHA256, "completed V3 scientific lineage"),
            "classification": v3_report["classification"],
            "outer_gate": v3_report["outer_gate"],
            "initial_state": {
                "runtime_state_sha256": INITIAL_RUNTIME_STATE_SHA256,
                "rng_state_sha256": INITIAL_RNG_STATE_SHA256,
                "zero_phase": True,
                "parameter_manifest": manifest,
            },
            "control_endpoint": {
                "smooth_macro_recall": 0.103,
                "frozen_macro_recall": 0.136,
                "image_loss": 0.0673580589145422,
            },
        },
        "v3_attribution": {
            "report": _ref(V3_ATTRIBUTION_REPORT_PATH, V3_ATTRIBUTION_REPORT_SHA256, "completed V3 failure-attribution report"),
            "input": _ref(V3_ATTRIBUTION_INPUT_PATH, V3_ATTRIBUTION_INPUT_SHA256, "V3 failure-attribution input"),
            "lineage": _ref(V3_ATTRIBUTION_LINEAGE_PATH, V3_ATTRIBUTION_LINEAGE_SHA256, "completed V3 failure-attribution lineage"),
            "classification": attribution["classification"],
            "outer_gate": attribution["outer_gate"],
        },
        "v2_contract": _ref(V2_CONTRACT_PATH, V2_CONTRACT_SHA256, "corrected V2 training contract"),
        "training_source": _ref(TRAINING_SOURCE_PATH, TRAINING_SOURCE_SHA256, "canonical TRAIN source"),
        "aligned_runtime": _ref(ALIGNED_RUNTIME_PATH, ALIGNED_RUNTIME_SHA256, "LightGen-aligned input/OLS runtime"),
        "evaluator_implementation": _ref(EVALUATOR_PATH, EVALUATOR_SHA256, "frozen evaluator implementation"),
        "evaluators": {
            "frozen_checkpoint": _ref(FROZEN_CHECKPOINT_PATH, FROZEN_CHECKPOINT_SHA256, "Frozen evaluator checkpoint"),
            "smooth_checkpoint": _ref(SMOOTH_CHECKPOINT_PATH, SMOOTH_CHECKPOINT_SHA256, "Smooth v2.1 evaluator checkpoint"),
            "gradient_role": "evaluation-only; no training gradient and no state update",
        },
        "v3_control": control,
        "original_zero_phase_generator": {
            "source": "fresh constructor state reproduced from the corrected V2 runtime and independently matched to V3 scientific initial_state",
            "runtime_state_sha256": INITIAL_RUNTIME_STATE_SHA256,
            "rng_state_sha256": INITIAL_RNG_STATE_SHA256,
            "tensor_name": "generator.phase",
            "shape": PHASE_SHAPE,
            "dtype": PHASE_DTYPE,
            "value": "all zeros",
            "tensor_sha256": zero_generator_sha256,
        },
        "fresh_runtime_reproduction": {
            "runtime_state_sha256": fresh_state_sha256,
            "matches_v3_initial_state": True,
            "constructor_grid": PRODUCTION_PROPAGATION_GRID,
        },
        "hash_on_trust_boundary": True,
    }


def _build_contract(root: Path, authorities: Mapping[str, Any], definition_sha256: str) -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 4,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "result": "READY",
        "classification": RESULT_CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": authorities["git"],
        "definition_entrypoint": {
            "path": "scripts/research/define_lightgen_2d_fixed_ols_generator_recoverability_contract_v4.py",
            "sha256": definition_sha256,
        },
        "question": "Given the immutable V3 CONTROL step-300 encoder and its fixed real complex OLS representations, can the existing LightGen-aligned optical generator learn to decode fixed OLS states into independently recognizable raw-CMOS outputs when only generator.phase is trained?",
        "claim_boundary": {
            "diagnostic_only": True,
            "does_not_claim": [
                "final LightGen readiness",
                "BOGT readiness",
                "native generation readiness",
                "Stage-A semantic eligibility",
                "Stage-B authorization",
                "scale extraction",
                "prior generation",
                "formal promotion",
            ],
        },
        "facts": {
            "v3_objective_correction": "VALID_COMPLETED / V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED",
            "v3_failure_attribution": "VALID_COMPLETED / PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED",
            "v3_control_endpoint": authorities["v3_scientific"]["control_endpoint"],
            "v3_control_encoder_checkpoint": "step-300, exact hash-bound runtime tensor",
            "existing_runtime": "LightGen input -> encoder -> C^100 OLS projection -> T=I_100 -> shared W_OLS -> generator -> raw CMOS",
            "stage_a_firewall": STAGE_A_FIREWALL,
        },
        "model_assumptions": [
            "A frozen V3 control encoder can provide a stable fixed real complex C^100 representation for a generator-only recoverability test.",
            "A generator-only image objective can test whether the existing LightGen optical generator can decode fixed OLS states without changing OLS geometry.",
            "The independently qualified Smooth v2.1 evaluator is suitable for bounded diagnostic endpoint evidence but is not training authority.",
        ],
        "design_decisions": {
            "control": "No new scientific control is trained in V4; the matched comparator is the existing V3 CONTROL step-300 raw-CMOS output on the same DEV selection.",
            "candidate": "One fixed-OLS generator-recoverability candidate with V3 CONTROL encoder frozen and only generator.phase trainable.",
            "objective": "L_generator = L_image only",
            "path": "image -> frozen V3 CONTROL encoder -> z_control in C^100 -> mandatory modal projection -> T=I_100 -> shared W_OLS -> trainable LightGen generator -> raw CMOS",
            "zero_generator_reset": "fresh exact original zero-phase generator initialization; V3 step-300 generator is forbidden as an initialization source",
            "semantic_provider": "no semantic provider in the optimization path; FrozenMNISTEvaluator and SmoothMNISTEvaluator remain evaluation-only",
            "ols": "100-port complex OLS, mandatory modal projection/re-emission, coefficient-space T=I_100, shared W_OLS; no geometry or transport modification",
        },
        "implementation_identity": {
            "runtime": authorities["aligned_runtime"],
            "v2_training_contract": authorities["v2_contract"],
            "propagation_grid": PRODUCTION_PROPAGATION_GRID,
            "complex_field_dtype": "torch.complex64",
            "phase_dtype": PHASE_DTYPE,
            "ols_port_count": 100,
            "w_ols": "lightgen_ols_coefficient_transmission_and_modal_reemission",
            "transport": "T=I_100_coefficient_space",
        },
        "parameter_firewall": {
            "trainable_exact": ["generator.phase"],
            "trainable_count": GENERATOR_PARAM_COUNT,
            "frozen_exact": ["encoder.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"],
            "frozen_encoder_parameter_count": ENCODER_PARAM_COUNT,
            "frozen_ols_state_numel": OLS_FROZEN_STATE_NUMEL,
            "encoder_gradient_required": False,
            "generator_gradient_required": True,
            "semantic_provider_gradient_required": False,
        },
        "objective": {
            "formula": "L_generator = L_image",
            "image_loss": "mean((raw_cmos - slm_intensity_target)^2)",
            "semantic_loss": "disabled; no semantic-provider gradient",
            "lambda_rule": "not applicable; no semantic weight is selected or tuned",
        },
        "optimizer_and_budget": {
            "optimizer": {
                "name": "Adam",
                "learning_rate": LEARNING_RATE,
                "betas": ADAM_BETAS,
                "eps": ADAM_EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "amsgrad": AMSGRAD,
                "parameter_names": ["generator.phase"],
            },
            "initialization": {
                "encoder": authorities["v3_control"]["encoder_phase"],
                "generator": authorities["original_zero_phase_generator"],
                "optimizer_moments": "fresh empty Adam state; no V3 optimizer state reuse",
            },
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "scientific_updates": SCIENTIFIC_STEPS,
            "checkpoint_steps": [100, 200, 300],
            "engineering_preflight_updates": PRECHECK_STEPS,
            "train_order": "same DeterministicTrainRoleSampler seed=1234 and current TRAIN ordering; first three real batches only in preflight",
            "matched_conditions": "same production 2048 propagation grid, same TRAIN source/role, same batch size, same Adam configuration, same real entrypoint boundaries",
        },
        "data_contract": {
            "source": authorities["training_source"],
            "train": {"count": 55_000, "updates": "TRAIN only", "preflight_rows_consumed": 30},
            "development": {"count": 5_000, "use": "evaluation only; no preflight values consumed"},
            "protected": {"use": "forbidden", "rows_read": 0},
            "test": {"use": "forbidden", "rows_read": 0},
            "sealed": {"use": "forbidden and unavailable", "rows_read": 0},
            "labels_as_model_input": False,
        },
        "comparison_and_gate": {
            "comparator": "existing V3 CONTROL step-300 output on the same fixed DEV selection; no matched V4 control retraining",
            "endpoint": "step 300 only",
            "metrics": [
                "Smooth v2.1 macro recall and per-class recall",
                "Frozen evaluator macro/per-class recall as descriptive-only secondary evidence",
                "DEV image loss against the same SLM intensity target",
                "raw-CMOS finite/shape/structured/nonconstant panel",
                "generator/encoder/frozen-OLS gradient and update diagnostics",
            ],
            "support_conditions": [
                "Smooth macro recall delta candidate-minus-V3-control >= 0.05",
                "paired bootstrap 95% lower bound for the Smooth delta > 0",
                "no per-class Smooth recall drop > 0.10",
                "candidate DEV image-loss degradation versus V3 control <= +0.005",
                "raw panel finite, shape-valid, structured, and nonconstant",
            ],
            "failure": "valid_failed / FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED when a complete scientific endpoint fails any support condition",
            "invalid_harness": "invalid_harness when a preflight or measurement boundary is incomplete or technically invalid; no scientific gate is emitted",
        },
        "stage_a_boundary": {
            "firewall": "DEV Smooth macro recall >= 0.937",
            "threshold": STAGE_A_FIREWALL,
            "unchanged": True,
            "recoverability_is_not_eligibility": True,
            "a_passing_V4_gate_does_not_authorize_stage_b": True,
        },
        "preflight": {
            "required": [
                "exact V3 CONTROL step-300 encoder checkpoint verified",
                "exact original zero-phase generator identity reproduced and verified",
                "encoder.phase frozen and unchanged",
                "generator-only finite nonzero gradient positive control",
                "encoder-gradient negative control",
                "no semantic-provider gradients or state updates",
                "cached fixed-z path equals canonical frozen-encoder path within declared tolerance",
                "mandatory modal projection, T=I_100, and shared W_OLS provenance active",
                "three real CUDA generator-only updates through the production 2048-grid path",
                "checkpoint/save/reload/resume identity within deterministic tolerance",
                "TRAIN-only updates and zero protected/test/sealed reads",
            ],
            "tolerance": {"rtol": DETERMINISTIC_RTOL, "atol": DETERMINISTIC_ATOL},
            "result": "AUTHORIZABLE only when all checks complete; otherwise BLOCKED or invalid_harness",
        },
        "one_shot_and_stop_conditions": {
            "one_shot": True,
            "scientific_run_not_authorized_by_this_contract": True,
            "stop_on": [
                "missing or hash-mismatched V3/V2 authority",
                "runtime identity conflict",
                "zero-phase generator identity mismatch",
                "fixed-z/canonical-z equivalence failure",
                "encoder or OLS update/gradient leakage",
                "semantic-provider gradient/state leakage",
                "nonfinite/undefined generator objective or raw CMOS",
                "TRAIN/DEV/protected/test/sealed role violation",
                "checkpoint/resume identity failure",
            ],
        },
        "authorities": authorities,
        "governance": {
            "engineering_preflight_executed": False,
            "scientific_training_executed": False,
            "stage_b_authorized": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_generation_executed": False,
            "generation_executed": False,
            "protected_read": False,
            "test_read": False,
            "sealed_read": False,
            "formal_promotion": False,
            "training_authorized": False,
            "generation_readiness": "not_ready",
        },
        "claim": "V4 defines one bounded fixed-OLS generator-recoverability experiment; it does not claim that the experiment has run or that any downstream stage is open.",
    }


def define_contract(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    output = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    definition_path = (root / "scripts/research/define_lightgen_2d_fixed_ols_generator_recoverability_contract_v4.py").resolve()
    definition_sha256 = file_sha256(definition_path)
    authorities = validate_v4_authorities(root)
    payload = _build_contract(root, authorities, definition_sha256)
    target = output / CONTRACT_NAME
    atomic_write_new_json(target, payload, artifact_kind="V4 fixed-OLS generator-recoverability contract")
    payload["artifact"] = {"path": target.relative_to(root).as_posix(), "sha256": file_sha256(target)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        payload = define_contract(args.repo_root, args.output_dir)
    except Exception as error:
        print(json.dumps({"task_id": TASK_ID, "status": "blocked", "result": "BLOCKED", "classification": "V4_CONTRACT_AUTHORITY_BLOCKED", "error": str(error)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({"task_id": payload["task_id"], "status": payload["status"], "result": payload["result"], "classification": payload["classification"], "outer_gate": payload["outer_gate"], "artifact": payload["artifact"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
