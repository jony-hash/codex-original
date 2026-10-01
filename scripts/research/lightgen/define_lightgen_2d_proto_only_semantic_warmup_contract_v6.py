"""Freeze the one-shot V6 prototype-only semantic-warmup contract.

This definition consumer resolves the already-authorized V3/V4/V5 authorities,
the historical cyclic-transfer templates, and the public LightGen/V2 runtime.
It reads no TRAIN or DEV values and never creates an optimizer or performs a
forward pass.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import file_sha256, git_facts
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import load_v2_training_contract, tensor_sha256
from ldm_project.v2.evidence import atomic_write_new_json, safe_torch_load


TASK_ID = "DEFINE_LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_CONTRACT_V6"
CONTRACT_SCHEMA = "lightgen_2d_proto_only_semantic_warmup_contract.v6"
CLASSIFICATION = "DEFINE_LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_CONTRACT_V6_READY"
OUTER_GATE = "V6_PROTO_ONLY_SEMANTIC_WARMUP_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V2_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"

V3_CONTROL_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
V3_CONTROL_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"

V4_GENERATOR_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_v4_step_300_checkpoint.v1.pt"
)
V4_GENERATOR_SHA256 = "de80f2826419ca6ced28c8e1fc6d8804ef513493a49474e5997cfb4cb59bdf80"
V4_REPORT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1.json"
)
V4_REPORT_SHA256 = "fdfeb8d800a2d806f3ecc56503ef8d66cc68f06de6cd9e47aa4eab7c93ca2784"
V4_LINEAGE_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1.json"
)
V4_LINEAGE_SHA256 = "6089bc41797580aa95149b0478a0e25190a450e00ec4a47c886b756f0187f9e6"

V5_CONTRACT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_contract.v5.json"
)
V5_CONTRACT_SHA256 = "e2d57fa96e0a020abf23cbd889188d74d4893a6e9c48ac7a1d24d7156cf21b61"
V5_REPORT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1.json"
)
V5_REPORT_SHA256 = "41aeb6e5ae5d1736521b164f2c8e9267bc3687801ab99b7142692112f8d85ce8"
V5_PROTOTYPE_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_control_step_300_checkpoint.v1.pt"
)
V5_PROTOTYPE_SHA256 = "abf62784d73a3ef7a5d51143642c42baf1472ce8da20af3f3f7a8392af32549f"

SMOOTH_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
SMOOTH_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
GAUGE_HELPER_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_complex_ols_training.py"
)
GAUGE_HELPER_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"
EVALUATOR_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"

HISTORICAL_INPUT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1.json"
)
HISTORICAL_INPUT_SHA256 = "c43b333822cab7c5ebafb5e64cdce7a50f386e662388f071d9378c740ed04b1d"
HISTORICAL_REPORT_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_report.v1.json"
)
HISTORICAL_REPORT_SHA256 = "f5b1b252b5815578fd09367258df66e7ba8f3ea008fc889cb30c23ec8d18c249"
HISTORICAL_LINEAGE_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_lineage.v1.json"
)
HISTORICAL_LINEAGE_SHA256 = "7a75ddb6008a65057c86e658104b6ccf61b14bd9091382f2101d55dae70440e7"
HISTORICAL_RAW_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1.pt"
)
HISTORICAL_RAW_SHA256 = "e985f6f030591a1aa38d33517d35da6c831806e121c9f73c3162395c23336d00"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v6/runs/"
    "lightgen_2d_proto_only_semantic_warmup_contract_v6_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_proto_only_semantic_warmup_contract.v6.json"
CLASS_COUNT = 10
OLS_PORT_COUNT = 100
PROTOTYPE_SCALE = 0.00555040966719389
PROTOTYPE_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
V3_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
OLS_TRANSPORT_SHA256 = "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293"
V4_GENERATOR_TENSOR_SHA256 = "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe"


class ContractError(RuntimeError):
    """Raised when a frozen V6 authority cannot be resolved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"{relative} must be relative")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ref(root: Path, relative: str | Path, expected: str, role: str, **extra: Any) -> dict[str, Any]:
    path = _path(root, relative)
    _require(path.is_file(), f"{role} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected, f"{role} SHA mismatch: expected {expected}, observed {observed}")
    return {"path": Path(relative).as_posix(), "sha256": observed, "role": role, **extra}


def _read_json(root: Path, relative: str | Path, expected: str, role: str) -> dict[str, Any]:
    _ref(root, relative, expected, role)
    try:
        payload = json.loads(_path(root, relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{role} is not valid JSON") from error
    _require(isinstance(payload, dict), f"{role} must be a JSON object")
    return payload


def _load_tensor_artifact(root: Path, relative: str | Path, expected: str, role: str) -> dict[str, Any]:
    _ref(root, relative, expected, role)
    payload = safe_torch_load(_path(root, relative), expected_sha256=expected, map_location="cpu")
    return dict(_mapping(payload, role))


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    schema: str,
    step: int,
    role: str,
) -> Mapping[str, Any]:
    _require(payload.get("schema") == schema, f"{role} schema changed")
    _require(payload.get("checkpoint_step") == step, f"{role} step changed")
    _require(payload.get("scientific_training_checkpoint") is True, f"{role} is not a scientific checkpoint")
    _require(payload.get("training_authorized") is False, f"{role} opened training authorization")
    return _mapping(payload.get("runtime_state"), f"{role} runtime state")


def _validate_git(root: Path) -> dict[str, Any]:
    facts = git_facts(root)
    _require(facts.get("branch") == EXPECTED_BRANCH, f"active branch differs: {facts.get('branch')}")
    _require(facts.get("head") == EXPECTED_HEAD, f"active HEAD differs: {facts.get('head')}")
    return facts


def _validate_v2(root: Path) -> dict[str, Any]:
    _ref(root, V2_CONTRACT_PATH, V2_CONTRACT_SHA256, "corrected V2 contract")
    contract = load_v2_training_contract(_path(root, V2_CONTRACT_PATH), repo_root=root)
    _require(contract.sha256 == V2_CONTRACT_SHA256, "V2 contract identity changed")
    _require(contract.payload.get("training_authorized") is False, "V2 contract opened training")
    architecture = _mapping(contract.payload.get("architecture"), "V2 architecture")
    immutable = [str(value) for value in architecture.get("immutable_components", [])]
    _require(architecture.get("same_real_and_generated_interface") is True, "real/generated interface diverged")
    _require(architecture.get("whole_field_identity_bypass") is False, "whole-field identity bypass changed")
    _require(any("T=I_100" in value and "W_OLS" in value for value in immutable), "T=I_100/W_OLS boundary changed")
    return {"path": V2_CONTRACT_PATH.as_posix(), "sha256": V2_CONTRACT_SHA256, "architecture": dict(architecture)}


def _validate_authorities(root: Path) -> dict[str, Any]:
    _validate_v2(root)
    v3 = _load_tensor_artifact(root, V3_CONTROL_PATH, V3_CONTROL_SHA256, "V3 CONTROL checkpoint")
    v3_state = _validate_checkpoint(
        v3,
        schema="lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_checkpoint.v1",
        step=300,
        role="V3 CONTROL checkpoint",
    )
    _require(v3.get("arm") == "control", "V3 checkpoint is not CONTROL")
    _require(set(v3_state) == {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "V3 runtime state keys changed")
    _require(tensor_sha256(v3_state["encoder.phase"]) == V3_ENCODER_SHA256, "V3 CONTROL encoder tensor changed")
    _require(torch.equal(v3_state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V3 transport is not T=I_100")
    _require(tensor_sha256(v3_state["ols.operator.transport"]) == OLS_TRANSPORT_SHA256, "V3 OLS transport changed")

    v4 = _load_tensor_artifact(root, V4_GENERATOR_PATH, V4_GENERATOR_SHA256, "V4 generator checkpoint")
    v4_state = _validate_checkpoint(
        v4,
        schema="lightgen_2d_fixed_ols_generator_recoverability_scientific_checkpoint.v1",
        step=300,
        role="V4 generator checkpoint",
    )
    _require(set(v4_state) == set(v3_state), "V4 runtime state keys differ from V3")
    _require(tensor_sha256(v4_state["encoder.phase"]) == V3_ENCODER_SHA256, "V4 encoder does not match V3 CONTROL")
    _require(tensor_sha256(v4_state["generator.phase"]) == V4_GENERATOR_TENSOR_SHA256, "V4 generator tensor changed")
    _require(torch.equal(v4_state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V4 transport is not T=I_100")
    _ref(root, V4_REPORT_PATH, V4_REPORT_SHA256, "V4 scientific report")
    _ref(root, V4_LINEAGE_PATH, V4_LINEAGE_SHA256, "V4 scientific lineage")

    v5_report = _read_json(root, V5_REPORT_PATH, V5_REPORT_SHA256, "V5 scientific report")
    _require(v5_report.get("result") == "VALID_COMPLETED" and v5_report.get("classification") == "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED", "V5 scientific authority changed")
    prototype_report = _mapping(v5_report.get("prototype_setup"), "V5 prototype setup")
    prototype_summary = _mapping(prototype_report.get("summary"), "V5 prototype summary")
    scale = _mapping(prototype_report.get("scale"), "V5 prototype scale")
    _require(prototype_summary.get("source_role") == "TRAIN only" and prototype_summary.get("prototypes_tensor_sha256") == PROTOTYPE_SHA256, "V5 prototype source changed")
    _require(scale.get("s_mu") == PROTOTYPE_SCALE and scale.get("pair_distance_tensor_sha256") == "85c1514fd20934b425120327fa900cc47b764ed58000edbb065107a9741166f5", "V5 prototype scale changed")
    v5_prototype = _load_tensor_artifact(root, V5_PROTOTYPE_PATH, V5_PROTOTYPE_SHA256, "V5 frozen prototype checkpoint")
    prototypes = v5_prototype.get("prototypes")
    _require(torch.is_tensor(prototypes) and tuple(prototypes.shape) == (CLASS_COUNT, OLS_PORT_COUNT) and prototypes.dtype == torch.complex64, "V5 prototypes shape/dtype changed")
    _require(v5_prototype.get("prototypes_sha256") == PROTOTYPE_SHA256 and tensor_sha256(prototypes) == PROTOTYPE_SHA256, "V5 prototypes tensor changed")
    _require(float(v5_prototype.get("s_mu")) == PROTOTYPE_SCALE, "V5 s_mu changed")
    _require(v5_prototype.get("arm") == "control", "V5 prototype source must be CONTROL checkpoint only")

    historical_input = _read_json(root, HISTORICAL_INPUT_PATH, HISTORICAL_INPUT_SHA256, "historical class-direction input")
    historical_report = _read_json(root, HISTORICAL_REPORT_PATH, HISTORICAL_REPORT_SHA256, "historical class-direction report")
    historical_lineage = _read_json(root, HISTORICAL_LINEAGE_PATH, HISTORICAL_LINEAGE_SHA256, "historical class-direction lineage")
    _require(historical_report.get("result") == "VALID_COMPLETED" and historical_report.get("classification") == "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED", "historical class-direction authority is not valid")
    _require(historical_lineage.get("result") == "VALID_COMPLETED" and historical_lineage.get("classification") == historical_report.get("classification"), "historical class-direction lineage changed")
    protocol = _mapping(historical_input.get("protocol"), "historical class-direction protocol")
    _require(protocol.get("target_rule") == "t=(y+1) mod 10", "historical cyclic target rule changed")
    _require(protocol.get("template_direction") == "M_template=d(output,tau_source)-d(output,tau_target); Delta=M_shift-M_base", "historical template convention changed")
    _require(protocol.get("smooth_direction") == "M_smooth=logit_target-logit_source; Delta=M_shift-M_base", "historical Smooth convention changed")
    _require(protocol.get("transport") == "T=I_100 coefficient space", "historical transport convention changed")
    historical_raw = _load_tensor_artifact(root, HISTORICAL_RAW_PATH, HISTORICAL_RAW_SHA256, "historical class-direction raw artifact")
    templates = historical_raw.get("templates")
    _require(torch.is_tensor(templates) and tuple(templates.shape) == (CLASS_COUNT, 1, 28, 28) and templates.dtype == torch.float32, "historical TRAIN templates changed")
    _require(bool(torch.isfinite(templates).all()), "historical TRAIN templates are nonfinite")

    smooth = _load_tensor_artifact(root, SMOOTH_PATH, SMOOTH_SHA256, "SmoothMNISTEvaluator v2.1 checkpoint")
    _require(smooth.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    for relative, expected, role in (
        (TRAINING_SOURCE_PATH, TRAINING_SOURCE_SHA256, "canonical TRAIN source"),
        (RUNTIME_PATH, RUNTIME_SHA256, "LightGen-aligned runtime"),
        (GAUGE_HELPER_PATH, GAUGE_HELPER_SHA256, "gauge helper"),
        (EVALUATOR_PATH, EVALUATOR_SHA256, "evaluator implementation"),
    ):
        _ref(root, relative, expected, role)

    return {
        "v2_contract": {"path": V2_CONTRACT_PATH.as_posix(), "sha256": V2_CONTRACT_SHA256},
        "v3_control": {
            "checkpoint": {"path": V3_CONTROL_PATH.as_posix(), "sha256": V3_CONTROL_SHA256, "step": 300},
            "encoder_phase": {"tensor_sha256": V3_ENCODER_SHA256, "shape": [2, 1, 100, 100], "role": "exact V3 CONTROL step-300 encoder"},
            "initialization_role": "V6 initial encoder; never V5 CONTROL/CANDIDATE step-300",
        },
        "v4_generator": {
            "checkpoint": {"path": V4_GENERATOR_PATH.as_posix(), "sha256": V4_GENERATOR_SHA256, "step": 300},
            "generator_phase": {"tensor_sha256": V4_GENERATOR_TENSOR_SHA256, "shape": [2, 1, 100, 100], "role": "fully frozen"},
            "scientific_report": {"path": V4_REPORT_PATH.as_posix(), "sha256": V4_REPORT_SHA256},
            "scientific_lineage": {"path": V4_LINEAGE_PATH.as_posix(), "sha256": V4_LINEAGE_SHA256},
        },
        "v5_prototypes": {
            "source_checkpoint": {"path": V5_PROTOTYPE_PATH.as_posix(), "sha256": V5_PROTOTYPE_SHA256, "role": "frozen V5 prototype payload only"},
            "tensor_sha256": PROTOTYPE_SHA256,
            "s_mu": PROTOTYPE_SCALE,
            "source_role": "TRAIN only; same initial V3 CONTROL encoder",
            "recompute_during_v6": False,
        },
        "historical_class_direction": {
            "input": {"path": HISTORICAL_INPUT_PATH.as_posix(), "sha256": HISTORICAL_INPUT_SHA256},
            "report": {"path": HISTORICAL_REPORT_PATH.as_posix(), "sha256": HISTORICAL_REPORT_SHA256},
            "lineage": {"path": HISTORICAL_LINEAGE_PATH.as_posix(), "sha256": HISTORICAL_LINEAGE_SHA256},
            "raw_templates": {"path": HISTORICAL_RAW_PATH.as_posix(), "sha256": HISTORICAL_RAW_SHA256, "template_tensor_shape": [10, 1, 28, 28]},
            "semantic_authority": "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED",
        },
        "smooth_evaluator": {"path": SMOOTH_PATH.as_posix(), "sha256": SMOOTH_SHA256, "role": "frozen evaluation-only SmoothMNISTEvaluator v2.1", "gradient_enabled": False},
        "source": {
            "training_source": _ref(root, TRAINING_SOURCE_PATH, TRAINING_SOURCE_SHA256, "canonical TRAIN source"),
            "aligned_runtime": _ref(root, RUNTIME_PATH, RUNTIME_SHA256, "LightGen-aligned runtime"),
            "gauge_helper": _ref(root, GAUGE_HELPER_PATH, GAUGE_HELPER_SHA256, "gauge helper"),
            "evaluator_implementation": _ref(root, EVALUATOR_PATH, EVALUATOR_SHA256, "evaluator implementation"),
        },
    }


def build_contract(root: Path) -> dict[str, Any]:
    git = _validate_git(root)
    authorities = _validate_authorities(root)
    definition_path = Path(__file__).resolve()
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "result": "READY",
        "status": "valid_completed",
        "classification": CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "definition_entrypoint": {"path": _relative(root, definition_path), "sha256": file_sha256(definition_path)},
        "authorities": authorities,
        "objective": {
            "question": "Can encoder-only prototype semantic warmup improve frozen OLS organization and independent natural raw-CMOS semantics while retaining the demonstrated cyclic class-direction geometry?",
            "training_objective": "L_V6=L_proto",
            "image_loss_gradient_contribution": "ZERO",
            "image_loss_role": "evaluation-only",
            "prototype_logits": "a_k(z)=-d(z,mu_k)/(s_mu+1e-12)",
            "prototype_loss": "cross_entropy([a_0(z),...,a_9(z)], y)",
            "distance": "existing gauge-invariant complex-OLS squared distance",
        },
        "model_assumption": "Removing reconstruction gradient during a frozen-prototype semantic warmup may strengthen latent semantic alignment without destroying transferable geometry.",
        "design_decisions": [
            "Use exact V3 CONTROL step-300 encoder as the only V6 initial encoder.",
            "Use exact V4 valid step-300 generator as fully frozen; do not initialize from V5 CONTROL/CANDIDATE.",
            "Reuse frozen V5 TRAIN-only prototypes and s_mu without recomputation.",
            "Do not train a separate image-MSE control arm; step 0 is the paired scientific baseline.",
            "Use fixed cyclic target t=(y+1) mod 10 and the previously validated balanced 1000-sample DEV selection.",
        ],
        "protocol": {
            "trainable": ["encoder.phase"],
            "frozen": ["generator.phase", "prototypes", "OLS", "T=I_100", "W_OLS", "SmoothMNISTEvaluator", "all other parameters"],
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False},
            "logical_batch_size": 10,
            "updates": 300,
            "checkpoints": [0, 100, 200, 300],
            "propagation_grid": 2048,
            "rng_convention": {"initialization_seed": 1234, "schedule_seed": 1234},
            "prototype_scale": PROTOTYPE_SCALE,
            "cyclic_transfer": {"target_rule": "t=(y+1) mod 10", "selection_rows": 1000, "selection_per_class": 100, "selection_seed": 4321, "no_nearest_wrong": True, "no_target_search": True, "no_magnitude_sweep": True},
            "bootstrap": {"replicates": 999, "seed": 271828, "type": "paired class-stratified"},
        },
        "metrics": {
            "primary_ols": {"name": "Delta_M_OLS", "formula": "M_step300-M_step0; M=min_{k!=y}d(z,mu_k)-d(z,mu_y)", "pass": "mean > 0 and paired bootstrap 95% CI lower > 0"},
            "primary_smooth": {"name": "Delta_R_Smooth", "formula": "R300-R0 using frozen SmoothMNISTEvaluator v2.1", "pass": "Delta >= +0.05 and paired bootstrap 95% CI lower > 0"},
            "known_transfer": {"template": "M_template=d(raw,tau_y)-d(raw,tau_t); Delta=M_shift-M_base", "smooth": "M_smooth=score_t-score_y; Delta=M_shift-M_base", "pass": "both mean > 0, both CI lower > 0, and not isolated-pair-only"},
            "image_loss": "report TRAIN/DEV L_image at 0/100/200/300; evaluation-only and not a Gate",
            "stage_a_firewall": "R300 >= 0.937, independently reported and does not authorize Stage B",
        },
        "data_roles": {
            "TRAIN": {"rows": 55000, "used_for": "optimization and existing prototype authority only"},
            "DEV": {"rows": 5000, "used_for": "frozen evaluation; balanced 1000-row cyclic-transfer selection"},
            "protected": 0,
            "official_test": 0,
            "sealed": 0,
            "labels_as_model_input": False,
        },
        "engineering_preflight": {
            "required": [
                "resolve all authorities",
                "verify exact V3 CONTROL encoder and V4 generator",
                "verify frozen V5 prototypes and s_mu",
                "verify encoder.phase sole trainable parameter",
                "prove L_image is not in the differentiated V6 objective",
                "prove finite nonzero L_proto gradient",
                "three-step CUDA smoke through the real entrypoint from a disposable copy",
                "checkpoint/reload/resume identity",
                "exact-core synthetic train/eval/transfer/bootstrap/sufficient/report/lineage reload",
            ],
            "scientific_meaning": "none for synthetic path",
        },
        "governance": {
            "one_shot": True,
            "scientific_retry": False,
            "second_hyperparameter_attempt": False,
            "training_authorized": False,
            "stage_b_authorized": False,
            "generation_readiness": "not_ready",
            "formal_promotion": False,
            "protected_read": False,
            "test_read": False,
            "sealed_read": False,
            "hash_policy": "HASH_ON_TRUST_BOUNDARY_V2",
        },
        "claim_boundary": {
            "contract_only": True,
            "scientific_training_executed": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_shaping_executed": False,
            "generation_executed": False,
            "formal_promotion": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = (args.output_dir or root / DEFAULT_OUTPUT_DIR).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "V6 contract output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite V6 contract output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload = build_contract(root)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="V6 proto-only semantic-warmup contract")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": payload["outer_gate"],
        "contract": {"path": _relative(root, path), "sha256": file_sha256(path)},
        "scientific_training_executed": False,
        "training_authorized": False,
        "stage_b_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
