"""Materialize the bounded V5 direct complex-OLS semantic-alignment contract.

This definition consumer binds the existing V3 CONTROL encoder, the valid V4
step-300 fixed-OLS generator, the class-direction transfer authority, and the
current LightGen-aligned runtime.  It does not load TRAIN values, execute an
optical forward, create an optimizer, run CUDA, or authorize the V5 scientific
comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    file_sha256,
    git_facts,
    load_v2_training_contract,
    tensor_sha256,
)
from ldm_project.v2.evidence import atomic_write_new_json, safe_torch_load


TASK_ID = "DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5"
CONTRACT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_contract.v5"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5_READY"
OUTER_GATE = "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "9a72e0bcf2de12ab9a0caa7151f273de76768dc7"

V2_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"

RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"

# The existing differentiable global-phase quotient is the project geometry
# authority used by the V2 latent diagnostics.  It is read-only in V5.
GAUGE_HELPER_PATH = Path(
    "src/ldm_project/v2/application/research/"
    "lightgen_2d_semantic_complex_ols_training.py"
)
GAUGE_HELPER_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"
GAUGE_SIGMA = 1.0e-3
GAUGE_EPSILON = 1.0e-6

EVALUATOR_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
SMOOTH_CHECKPOINT_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"

V3_CONTROL_CHECKPOINT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
V3_CONTROL_CHECKPOINT_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"
V3_CONTROL_ENCODER_PHASE_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V3_CONTROL_OLS_TRANSPORT_SHA256 = "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293"

V3_SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json"
)
V3_SCIENTIFIC_REPORT_SHA256 = "8d7930449a1872e685ad39dafa499d89fdcc1d6bb5a9317e7ba06c554f54f31a"

V4_SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1.json"
)
V4_SCIENTIFIC_REPORT_SHA256 = "fdfeb8d800a2d806f3ecc56503ef8d66cc68f06de6cd9e47aa4eab7c93ca2784"
V4_SCIENTIFIC_LINEAGE_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1.json"
)
V4_SCIENTIFIC_LINEAGE_SHA256 = "6089bc41797580aa95149b0478a0e25190a450e00ec4a47c886b756f0187f9e6"
V4_GENERATOR_CHECKPOINT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_v4_step_300_checkpoint.v1.pt"
)
V4_GENERATOR_CHECKPOINT_SHA256 = "de80f2826419ca6ced28c8e1fc6d8804ef513493a49474e5997cfb4cb59bdf80"

CLASS_DIRECTION_REPORT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_report.v1.json"
)
CLASS_DIRECTION_REPORT_SHA256 = "f5b1b252b5815578fd09367258df66e7ba8f3ea008fc889cb30c23ec8d18c249"
CLASS_DIRECTION_LINEAGE_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_lineage.v1.json"
)
CLASS_DIRECTION_LINEAGE_SHA256 = "7a75ddb6008a65057c86e658104b6ccf61b14bd9091382f2101d55dae70440e7"

PHASE_SHAPE = [2, 1, 100, 100]
PHASE_DTYPE = "torch.float32"
OLS_TRANSPORT_SHAPE = [100, 100]
OLS_TRANSPORT_DTYPE = "torch.complex64"
OLS_PORT_COUNT = 100
LOGICAL_BATCH_SIZE = 10
STAGE_A_STEPS = 300
CHECKPOINT_STEPS = [100, 200, 300]
PRECHECK_STEPS = 3
PRODUCTION_GRID = 2048
SMOKE_GRID = 128
LEARNING_RATE = 1.0e-3
ADAM_BETAS = [0.9, 0.999]
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STAGE_A_FIREWALL = 0.937
EPSILON_DENOMINATOR = 1.0e-12

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_complex_ols_semantic_alignment_contract.v5.json"


class V5DefinitionError(RuntimeError):
    """Raised when a required V5 authority or invariant is not resolved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V5DefinitionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


def _ref(root: Path, relative: str | Path, expected_sha256: str, role: str, **extra: Any) -> dict[str, Any]:
    path = _path(root, relative)
    _require(path.is_file(), f"{role} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected_sha256, f"{role} SHA mismatch: expected {expected_sha256}, observed {observed}")
    return {"path": Path(relative).as_posix(), "sha256": observed, "role": role, **extra}


def _read_json(root: Path, relative: str | Path, expected_sha256: str, role: str) -> dict[str, Any]:
    _ref(root, relative, expected_sha256, role)
    try:
        payload = json.loads(_path(root, relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V5DefinitionError(f"{role} is not valid JSON") from error
    _require(isinstance(payload, dict), f"{role} must be a JSON object")
    return payload


def _git_state(root: Path) -> dict[str, Any]:
    try:
        facts = git_facts(root)
    except (OSError, subprocess.SubprocessError) as error:
        raise V5DefinitionError("Git identity could not be read") from error
    _require(facts.get("branch") == EXPECTED_BRANCH, f"active branch differs from V5 plan: {facts.get('branch')}")
    _require(facts.get("head") == EXPECTED_HEAD, f"active HEAD differs from V5 plan: {facts.get('head')}")
    return facts


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


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


def _validate_data_roles(payload: Mapping[str, Any], name: str) -> None:
    roles = _mapping(payload.get("data_roles"), f"{name} data roles")
    _require(roles.get("train") is True or roles.get("train_used") is True, f"{name} did not record TRAIN use")
    for key in ("development", "protected", "test", "sealed"):
        if key in roles:
            _require(roles.get(key) is False, f"{name} opened forbidden data role: {key}")
    for key in ("development_used", "protected_used", "test_used", "sealed_used"):
        if key in roles:
            _require(roles.get(key) is False, f"{name} opened forbidden data role: {key}")


def _validate_v3_control(root: Path) -> dict[str, Any]:
    path = _path(root, V3_CONTROL_CHECKPOINT_PATH)
    payload = _mapping(
        safe_torch_load(path, expected_sha256=V3_CONTROL_CHECKPOINT_SHA256, map_location="cpu"),
        "V3 CONTROL checkpoint",
    )
    _require(payload.get("schema") == "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_checkpoint.v1", "V3 CONTROL checkpoint schema changed")
    _require(payload.get("checkpoint_step") == 300 and payload.get("arm") == "control", "V3 CONTROL endpoint changed")
    _require(payload.get("scientific_training_checkpoint") is True, "V3 CONTROL is not a scientific checkpoint")
    _require(payload.get("training_authorized") is False, "V3 CONTROL opened training authorization")
    for key in ("stage_b_executed", "generation_executed", "formal_promotion"):
        _require(payload.get(key) is False, f"V3 CONTROL crossed forbidden boundary: {key}")
    state = _mapping(payload.get("runtime_state"), "V3 CONTROL runtime state")
    expected_keys = {
        "encoder.phase",
        "generator.phase",
        "ols.operator.input_modes",
        "ols.operator.output_modes",
        "ols.operator.transport",
    }
    _require(set(state) == expected_keys, "V3 CONTROL runtime state keys changed")
    encoder = state["encoder.phase"]
    transport = state["ols.operator.transport"]
    _require(torch.is_tensor(encoder) and list(encoder.shape) == PHASE_SHAPE and str(encoder.dtype) == PHASE_DTYPE, "V3 CONTROL encoder shape/dtype changed")
    _require(torch.is_tensor(transport) and list(transport.shape) == OLS_TRANSPORT_SHAPE and str(transport.dtype) == OLS_TRANSPORT_DTYPE, "V3 CONTROL transport shape/dtype changed")
    _require(tensor_sha256(encoder) == V3_CONTROL_ENCODER_PHASE_SHA256, "V3 CONTROL encoder tensor changed")
    _require(tensor_sha256(transport) == V3_CONTROL_OLS_TRANSPORT_SHA256, "V3 CONTROL OLS transport changed")
    _require(torch.equal(transport, torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V3 CONTROL transport is not T=I_100")
    _validate_data_roles(payload, "V3 CONTROL")
    return {
        "checkpoint": _ref(root, V3_CONTROL_CHECKPOINT_PATH, V3_CONTROL_CHECKPOINT_SHA256, "V3 CONTROL step-300 encoder checkpoint", step=300),
        "encoder_phase": {
            "shape": PHASE_SHAPE,
            "dtype": PHASE_DTYPE,
            "tensor_sha256": V3_CONTROL_ENCODER_PHASE_SHA256,
            "role": "initial trainable encoder state for both V5 arms",
        },
        "ols_transport": {
            "shape": OLS_TRANSPORT_SHAPE,
            "dtype": OLS_TRANSPORT_DTYPE,
            "tensor_sha256": V3_CONTROL_OLS_TRANSPORT_SHA256,
            "identity": "T=I_100 coefficient space",
        },
        "base_initial_state_digest": payload.get("base_initial_state_digest"),
        "producer_task": payload.get("task_id"),
        "producer_attempt": "v3_scientific_attempt_01",
    }


def _validate_v4_generator(root: Path, v3_control: Mapping[str, Any]) -> dict[str, Any]:
    report = _read_json(root, V4_SCIENTIFIC_REPORT_PATH, V4_SCIENTIFIC_REPORT_SHA256, "V4 scientific report")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "V4 scientific result is not valid completed")
    _require(report.get("classification") == "FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED", "V4 scientific classification changed")
    boundary = _mapping(report.get("claim_boundary"), "V4 claim boundary")
    for key in ("stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "sampled_ols_generation_executed", "protected_accessed", "test_accessed", "sealed_accessed", "formal_promotion"):
        _require(boundary.get(key) is False, f"V4 claim boundary opened: {key}")
    report_checkpoints = _mapping(report.get("checkpoints"), "V4 checkpoints")
    report_checkpoint = _mapping(report_checkpoints.get("300"), "V4 step-300 checkpoint authority")
    _require(report_checkpoint.get("path") == V4_GENERATOR_CHECKPOINT_PATH.as_posix(), "V4 checkpoint producer path changed")
    _require(report_checkpoint.get("sha256") == V4_GENERATOR_CHECKPOINT_SHA256, "V4 checkpoint producer SHA changed")
    lineage = _read_json(root, V4_SCIENTIFIC_LINEAGE_PATH, V4_SCIENTIFIC_LINEAGE_SHA256, "V4 scientific lineage")
    _require(lineage.get("result") == "VALID_COMPLETED" and lineage.get("classification") == report.get("classification"), "V4 scientific lineage changed")

    payload = _mapping(
        safe_torch_load(_path(root, V4_GENERATOR_CHECKPOINT_PATH), expected_sha256=V4_GENERATOR_CHECKPOINT_SHA256, map_location="cpu"),
        "V4 step-300 generator checkpoint",
    )
    _require(payload.get("schema") == "lightgen_2d_fixed_ols_generator_recoverability_scientific_checkpoint.v1", "V4 generator checkpoint schema changed")
    _require(payload.get("checkpoint_step") == 300 and payload.get("objective") == "L_generator=L_image", "V4 generator endpoint changed")
    _require(payload.get("scientific_training_checkpoint") is True and payload.get("training_authorized") is False, "V4 generator boundary changed")
    for key in ("stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "generation_executed", "formal_promotion"):
        _require(payload.get(key) is False, f"V4 generator crossed forbidden boundary: {key}")
    _validate_data_roles(payload, "V4 generator")
    state = _mapping(payload.get("runtime_state"), "V4 generator runtime state")
    _require(set(state) == {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "V4 generator runtime state keys changed")
    generator = state["generator.phase"]
    encoder = state["encoder.phase"]
    transport = state["ols.operator.transport"]
    _require(torch.is_tensor(generator) and list(generator.shape) == PHASE_SHAPE and str(generator.dtype) == PHASE_DTYPE, "V4 generator phase shape/dtype changed")
    _require(torch.is_tensor(encoder) and tensor_sha256(encoder) == v3_control["encoder_phase"]["tensor_sha256"], "V4 frozen encoder is not the V3 CONTROL encoder")
    _require(torch.is_tensor(transport) and torch.equal(transport, torch.eye(OLS_PORT_COUNT, dtype=torch.complex64),), "V4 generator transport is not T=I_100")
    _require(tensor_sha256(transport) == V3_CONTROL_OLS_TRANSPORT_SHA256, "V4 generator OLS transport changed")
    generator_sha = tensor_sha256(generator)
    return {
        "scientific_report": _ref(root, V4_SCIENTIFIC_REPORT_PATH, V4_SCIENTIFIC_REPORT_SHA256, "valid V4 fixed-OLS scientific report", classification=report["classification"]),
        "scientific_lineage": _ref(root, V4_SCIENTIFIC_LINEAGE_PATH, V4_SCIENTIFIC_LINEAGE_SHA256, "valid V4 fixed-OLS scientific lineage"),
        "checkpoint": _ref(root, V4_GENERATOR_CHECKPOINT_PATH, V4_GENERATOR_CHECKPOINT_SHA256, "valid V4 step-300 fixed-OLS generator checkpoint", step=300, tensor_domain="checkpoint file bytes"),
        "generator_phase": {"shape": PHASE_SHAPE, "dtype": PHASE_DTYPE, "tensor_sha256": generator_sha, "role": "frozen generator for both V5 arms"},
        "encoder_phase_match": True,
        "producer_task": payload.get("task_id"),
        "producer_attempt": "v4_scientific_attempt_02",
        "scientific_classification": report.get("classification"),
    }


def _validate_class_direction_authority(root: Path) -> dict[str, Any]:
    report = _read_json(root, CLASS_DIRECTION_REPORT_PATH, CLASS_DIRECTION_REPORT_SHA256, "class-direction replication report")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "class-direction replication is not valid completed")
    _require(report.get("classification") == "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED", "class-direction replication classification changed")
    _require(report.get("outer_gate") == "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED", "class-direction replication outer gate changed")
    boundary = _mapping(report.get("claim_boundary"), "class-direction replication claim boundary")
    for key in ("optimizer_created", "backward_executed", "optical_update_executed", "training_authorized", "stage_b_authorized", "stage_b_executed", "scale_extraction_executed", "prior_work_executed", "sampled_generation_executed", "protected_accessed", "test_accessed", "sealed_accessed", "formal_promotion"):
        if key in boundary:
            _require(boundary.get(key) is False, f"class-direction replication boundary opened: {key}")
    lineage = _read_json(root, CLASS_DIRECTION_LINEAGE_PATH, CLASS_DIRECTION_LINEAGE_SHA256, "class-direction replication lineage")
    _require(lineage.get("result") == "VALID_COMPLETED" and lineage.get("classification") == report.get("classification"), "class-direction replication lineage changed")
    return {
        "report": _ref(root, CLASS_DIRECTION_REPORT_PATH, CLASS_DIRECTION_REPORT_SHA256, "valid class-direction transfer replication report"),
        "lineage": _ref(root, CLASS_DIRECTION_LINEAGE_PATH, CLASS_DIRECTION_LINEAGE_SHA256, "valid class-direction transfer replication lineage"),
        "semantic_authority": "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED",
        "classification": report.get("classification"),
        "producer_task": report.get("task_id"),
        "producer_attempt": "replication_v1_attempt_01",
    }


def validate_v5_authorities(root: Path) -> dict[str, Any]:
    git = _git_state(root)
    v2_path = _path(root, V2_CONTRACT_PATH)
    v2 = load_v2_training_contract(v2_path, repo_root=root)
    _require(v2.sha256 == V2_CONTRACT_SHA256, "corrected V2 contract SHA changed")
    _require(v2.payload.get("result") == "READY" and v2.payload.get("status") == "valid_completed", "corrected V2 contract is not READY")
    _require(v2.payload.get("training_authorized") is False, "corrected V2 contract opened training")
    _require(_mapping(v2.payload.get("stage_a"), "V2 Stage-A").get("steps") == STAGE_A_STEPS, "V2 Stage-A budget changed")
    _require(_mapping(_mapping(v2.payload.get("stage_a"), "V2 Stage-A").get("loss"), "V2 Stage-A loss").get("formula") == "L_A = 1.0*L_image + 0.0*L_optical", "V2 pure-image control changed")

    source = {
        "v2_contract": _ref(root, V2_CONTRACT_PATH, V2_CONTRACT_SHA256, "corrected V2 contract"),
        "training_source": _ref(root, TRAINING_SOURCE_PATH, TRAINING_SOURCE_SHA256, "canonical TRAIN source", values_read_by_definition=False),
        "aligned_runtime": _ref(root, RUNTIME_PATH, RUNTIME_SHA256, "LightGen-aligned input/OLS runtime"),
        "gauge_helper": _ref(root, GAUGE_HELPER_PATH, GAUGE_HELPER_SHA256, "existing complex-OLS gauge helper"),
        "evaluator_implementation": _ref(root, EVALUATOR_PATH, EVALUATOR_SHA256, "evaluator implementation"),
        "smooth_checkpoint": _ref(root, SMOOTH_CHECKPOINT_PATH, SMOOTH_CHECKPOINT_SHA256, "SmoothMNISTEvaluator v2.1 checkpoint", identity_domain="checkpoint file bytes"),
    }
    runtime_text = _path(root, RUNTIME_PATH).read_text(encoding="utf-8")
    for marker in ("forward_from_image", "forward_from_ols", "modal_projection", "W_OLS"):
        _require(marker in runtime_text, f"LightGen runtime marker is missing: {marker}")
    gauge_text = _path(root, GAUGE_HELPER_PATH).read_text(encoding="utf-8")
    for marker in ("gauge_invariant_ols_representation", "GAUGE_EPSILON", "PRIOR_SIGMA"):
        _require(marker in gauge_text, f"complex-OLS gauge helper marker is missing: {marker}")

    v3_control = _validate_v3_control(root)
    v4_generator = _validate_v4_generator(root, v3_control)
    class_direction = _validate_class_direction_authority(root)
    return {
        "consumer_git": git,
        "hash_governance": {
            "policy": "HASH_ON_TRUST_BOUNDARY_V2",
            "routine_identity": "semantic authority + producer task/attempt + consumer Git branch/HEAD",
            "file_sha256": "checkpoint/file-byte identity when exact file transfer is required",
            "state_dict_tree_sha256": "loaded tree identity; not equal-gated against file SHA",
            "tensor_content_sha256": "tensor-content identity for explicitly named tensor fields",
            "manifest_sha256": "machine-manifest identity only",
            "cross_domain_equality_required": False,
            "producer_head_equals_consumer_head_required": False,
        },
        "source": source,
        "v3_control": v3_control,
        "v4_generator": v4_generator,
        "class_direction_replication": class_direction,
        "semantic_evaluator": {
            "provider_for_scientific_evaluation": "SmoothMNISTEvaluator v2.1",
            "training_provider": "none",
            "gradient_enabled_during_v5_training": False,
            "role": "independent evaluation-only provider; not sole acceptance evidence",
        },
        "current_v4_conditionality": {
            "outer_gate": "V4_GENERATOR_CONDITIONALITY_RESOLVED",
            "status": "unresolved",
            "consumed_as": "context only; no V4 diagnosis retry",
        },
    }


def _build_contract(root: Path, authorities: Mapping[str, Any], definition_sha256: str) -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 5,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "result": "READY",
        "classification": CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": authorities["consumer_git"],
        "definition_entrypoint": {
            "path": "scripts/research/define_lightgen_2d_complex_ols_semantic_alignment_contract_v5.py",
            "sha256": definition_sha256,
        },
        "question": "With the V4 step-300 generator frozen, can direct TRAIN-label-grounded alignment of encoder-produced complex OLS states to frozen class prototypes improve independent raw-CMOS semantics over the matched pure-image encoder objective?",
        "facts": [
            "The valid class-direction replication supports broad target-consistent optical-output movement under fixed complex OLS and fixed V4 generator.",
            "The existing V3 output-feature objective was not supported and was attributed to provider-specific alignment.",
            "The existing valid V4 image-only fixed-OLS generator-recoverability run was not supported.",
            "The V4 step-300 generator checkpoint is a valid scientific artifact but is not a promoted generator or a Stage-B authority.",
        ],
        "model_assumptions": [
            "The learned OLS semantic alignment, rather than proven absolute generator capacity insufficiency, may be the current bottleneck.",
            "Direct label-grounded prototype geometry may align the trainable encoder with class directions that the frozen generator can transfer.",
        ],
        "design_decisions": {
            "single_experiment": True,
            "control": "L_control = L_image",
            "candidate": "L_candidate = L_image + lambda_proto * L_proto",
            "frozen_generator": "exact valid V4 step-300 fixed-OLS generator; all generator.phase parameters frozen and excluded from the optimizer",
            "trainable_surface": ["encoder.phase"],
            "initial_encoder": "exact V3 CONTROL step-300 encoder.phase for both arms; V3 feature-objective candidate is forbidden",
            "ols": "100-port complex OLS, mandatory modal projection/re-emission, coefficient-space T=I_100, shared W_OLS",
            "matched_difference": "the frozen prototype loss is the only control/candidate objective difference",
            "v3_output_feature_objective": "not reused",
        },
        "prototype_geometry": {
            "authority": "existing project complex-OLS gauge convention",
            "latent_domain": "full complex C^100",
            "construction_role": "TRAIN only, from the frozen initial V3 CONTROL encoder",
            "aggregation": "for each class k, arithmetic mean of all gauge-canonicalized initial encoder z rows with TRAIN label k",
            "prototype_names": [f"mu_{index}" for index in range(10)],
            "gauge": {
                "formula": "G(z)=z*conj(a(z))/sqrt(|a(z)|^2+epsilon_g^2)",
                "anchor": "a(z)=sum_j z_j/sqrt(100)",
                "epsilon_g": GAUGE_EPSILON,
                "sigma": GAUGE_SIGMA,
                "representation": "S(z)=concat(Re(G(z)/sigma), Im(G(z)/sigma)) in R^200",
                "role": "distance representation only; runtime z and W_OLS are not transformed",
            },
            "distance": "d(z,mu)=sum over the 200 coordinates of (S(z)-S(mu))^2",
            "scale": {
                "formula": "s_mu=median_{0<=k<l<10} d(mu_k,mu_l)",
                "epsilon_denominator": EPSILON_DENOMINATOR,
                "required": "finite and strictly positive; otherwise BLOCKED",
            },
            "logits": "a_k(z)=-d(z,mu_k)/(s_mu+1e-12)",
            "loss": "L_proto=CrossEntropy([a_0(z),...,a_9(z)], y)",
            "frozen_after_construction": True,
            "forbidden": ["DEV prototypes", "evolving candidate prototypes", "FrozenMNISTEvaluator gradients", "SmoothMNISTEvaluator gradients", "output classifier features"],
        },
        "lambda_rule": {
            "formula": "lambda_proto=g_image/(g_proto+1e-12)",
            "batch": "the exact first TRAIN batch used by both arms",
            "g_image": "L2 norm of grad_encoder_phase(L_image)",
            "g_proto": "L2 norm of grad_encoder_phase(L_proto)",
            "timing": "once before any optimizer update",
            "clipping": False,
            "tuning": False,
            "recomputation_after_start": False,
            "invalid_if": ["nonfinite/zero/undefined required gradient", "nonfinite lambda", "g_proto <= 0"],
        },
        "matched_protocol": {
            "initial_state": "exactly equal combined V3 CONTROL encoder + V4 step-300 generator + OLS state",
            "rng_and_sampler": "same start state and deterministic TRAIN sampler order",
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": ADAM_BETAS, "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "propagation_grid": PRODUCTION_GRID,
            "scientific_updates": STAGE_A_STEPS,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "engineering_updates_per_arm": PRECHECK_STEPS,
            "precision": {"phase": PHASE_DTYPE, "field": "torch.complex64"},
        },
        "dev_diagnostic": {
            "name": "M_OLS",
            "formula": "min_{k != y} d(z,mu_k)-d(z,mu_y)",
            "direction": "higher is better",
            "electronic_classifier": False,
            "preflight_values_consumed": False,
        },
        "scientific_gate": {
            "outer_gate": OUTER_GATE,
            "endpoint": "step 300",
            "criteria": [
                "Smooth macro recall candidate-control >= +0.05",
                "paired class-stratified bootstrap 95% CI lower bound for Smooth recall delta > 0",
                "mean Delta_M > 0 and paired bootstrap 95% CI lower bound > 0",
                "no Smooth per-class recall degradation > 0.10",
                "DEV image-loss degradation <= +0.005",
                "raw CMOS finite, shape-valid, structured, and nonconstant",
            ],
            "support_classification": "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED",
            "complete_failure_classification": "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED",
            "incomplete_or_broken_harness": "invalid_harness; no scientific classification",
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired class-stratified", "quantile": "torch.quantile interpolation=linear"},
        },
        "stage_a_boundary": {
            "firewall_metric": "Smooth DEV macro recall",
            "threshold": STAGE_A_FIREWALL,
            "unchanged": True,
            "passing_v5_does_not_authorize_stage_b": True,
            "stage_b_entry_requires_independent_firewall_pass": True,
        },
        "data_contract": {
            "source": authorities["source"]["training_source"],
            "train": {"rows": 55_000, "prototype_values_allowed": True, "preflight_update_rows": PRECHECK_STEPS * LOGICAL_BATCH_SIZE},
            "development": {"rows": 5_000, "prototype_values_allowed": False, "preflight_scientific_values_consumed": False, "future_evaluation_only": True},
            "protected": {"rows_read": 0, "use": "forbidden"},
            "test": {"rows_read": 0, "use": "forbidden"},
            "sealed": {"rows_read": 0, "use": "forbidden and unavailable"},
            "labels_as_model_input": False,
        },
        "preflight": {
            "required_before_scientific_authorization": [
                "exact V3 CONTROL encoder and exact V4 generator authority resolved",
                "all ten TRAIN-only complex prototypes constructed once",
                "s_mu finite and positive",
                "prototype logits and loss finite",
                "one-shot g_image/g_proto/lambda_proto calibration valid",
                "generator frozen and excluded from optimizer; encoder.phase is the sole trainable parameter",
                "candidate prototype gradient finite and nonzero; control prototype term disabled",
                "three true CUDA updates through production 2048-grid path for each arm",
                "modal projection, T=I_100 and W_OLS provenance unchanged",
                "checkpoint save/reload/resume identity valid",
                "full synthetic exact-core path passes through gate/statistics/report/lineage",
                "production-grid CUDA no-data smoke passes",
                "zero protected/test/sealed reads and no scientific DEV values consumed",
            ],
            "synthetic_result_has_no_scientific_meaning": True,
            "result": "AUTHORIZABLE only after every bounded check passes",
        },
        "one_shot_and_stop_conditions": {
            "scientific_one_shot": True,
            "scientific_comparison_authorized_by_this_task": False,
            "stop_on": [
                "missing V3/V4/class-direction authority",
                "undefined prototype or invalid s_mu",
                "nonfinite/zero prototype gradient or lambda",
                "generator/encoder firewall failure",
                "matched-arm state or checkpoint/resume mismatch",
                "modal/T/W_OLS boundary change",
                "data-role leakage",
                "synthetic full-path failure",
                "any request to tune lambda, recompute prototypes, or execute 300-step science in this task",
            ],
        },
        "authorities": authorities,
        "governance": {
            "definition_only": True,
            "engineering_preflight_executed": False,
            "scientific_training_executed": False,
            "scientific_comparison_executed": False,
            "training_authorized": False,
            "stage_b_authorized": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_generation_executed": False,
            "generation_executed": False,
            "protected_read": False,
            "test_read": False,
            "sealed_read": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "claim": "V5 defines one falsifiable direct complex-OLS semantic-alignment experiment and its bounded engineering preflight; it does not claim V5 science has run or that any downstream stage is open.",
    }


def define_contract(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    authorities = validate_v5_authorities(root)
    output = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(output.is_relative_to(outputs_root), "V5 contract output must remain under outputs/")
    _require(not output.exists(), f"refusing to overwrite existing V5 contract output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    definition_path = (root / "scripts/research/define_lightgen_2d_complex_ols_semantic_alignment_contract_v5.py").resolve()
    payload = _build_contract(root, authorities, file_sha256(definition_path))
    target = output / CONTRACT_NAME
    atomic_write_new_json(target, payload, artifact_kind="V5 complex-OLS semantic-alignment contract")
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
        print(json.dumps({"task_id": TASK_ID, "result": "BLOCKED", "status": "blocked", "classification": "V5_CONTRACT_AUTHORITY_BLOCKED", "error": str(error)}, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps({"task_id": payload["task_id"], "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": payload["outer_gate"], "artifact": payload["artifact"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
