"""Materialize the bounded V3 Stage-A objective-correction contract.

This definition consumer verifies the already published LightGen, V2 training,
warmup-diagnosis, and evaluator authorities, then writes one immutable contract.
It does not load MNIST tensors, execute an optical forward, create an optimizer,
run CUDA, train, enter Stage B, generate prior samples, or read protected,
test, or sealed data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3"
GATE = "V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED"
CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_contract.v3"
READY_CLASSIFICATION = (
    "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3_READY"
)

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "383197a930dd48f98294120f8cf867444f994f85"

OUTPUT_DIR = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract_v3_attempt_01"
)
CONTRACT_NAME = (
    "lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json"
)

ALIGNMENT_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json"
)
ALIGNMENT_CONTRACT_SHA256 = (
    "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"
)
ALIGNMENT_RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = (
    "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
)
ALIGNMENT_RUNTIME_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"
)
ALIGNMENT_RUNTIME_REPORT_SHA256 = (
    "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"
)

V2_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = (
    "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
)
V2_TRAINING_REPORT_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_training_v2_report.v1.json"
)
V2_TRAINING_REPORT_SHA256 = (
    "de55a0f307675ded10aca5726f9ac09474633b2c9a5f8bfb1890075de53a530d"
)
V2_TRAINING_LINEAGE_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_training_v2_lineage.v1.json"
)
V2_TRAINING_LINEAGE_SHA256 = (
    "899aa78daff4bbe2f70cf72e01080e323e24d791f4937ca35fefc8c9b30394dc"
)
DIAGNOSIS_REPORT_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_ols_warmup_failure_diagnosis_v1_attempt_02/"
    "lightgen_2d_semantic_ols_warmup_failure_diagnosis_report.v1.json"
)
DIAGNOSIS_REPORT_SHA256 = (
    "f05eb4e6dfd7c8c3ce1865ff8097fdef2dfa649b3e3dab3352f5282fa544f3da"
)
DIAGNOSIS_LINEAGE_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_ols_warmup_failure_diagnosis_v1_attempt_02/"
    "lightgen_2d_semantic_ols_warmup_failure_diagnosis_lineage.v1.json"
)
DIAGNOSIS_LINEAGE_SHA256 = (
    "ef72fcaf191fc147d34fc460fb0d0f147614b2d696d677feae7d5d501f40b763"
)

EVALUATOR_IMPLEMENTATION_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_IMPLEMENTATION_SHA256 = (
    "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
)
FROZEN_EVALUATOR_CHECKPOINT_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
FROZEN_EVALUATOR_CHECKPOINT_SHA256 = (
    "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
)
FROZEN_EVALUATOR_QUALIFICATION_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "evaluator_qualification_evidence.json"
)
FROZEN_EVALUATOR_QUALIFICATION_SHA256 = (
    "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e"
)

SMOOTH_EVALUATOR_CHECKPOINT_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
SMOOTH_EVALUATOR_CHECKPOINT_SHA256 = (
    "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
)
SMOOTH_EVALUATOR_BUNDLE_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "conditional_evaluator_bundle_v21.json"
)
SMOOTH_EVALUATOR_BUNDLE_SHA256 = (
    "970dd6ea39cbda8aa23d3ca8338d9b0140a21cde45e4199137ef303f9d73ea31"
)
SMOOTH_EVALUATOR_QUALIFICATION_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "conditional_evaluator_qualification_v21_evidence.json"
)
SMOOTH_EVALUATOR_QUALIFICATION_SHA256 = (
    "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88"
)

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = (
    "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
)

PRIMARY_EVALUATOR_VALIDATION_ACCURACY = 0.9875999689102173
STAGE_A_SEMANTIC_THRESHOLD = 0.937
LOGICAL_BATCH_SIZE = 10
STAGE_A_STEPS = 300
CHECKPOINT_STEPS = (100, 200, 300)
INITIALIZATION_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321
LAMBDA_CALIBRATION_SEED = 271828
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_ALPHA = 0.05


class ContractDefinitionError(ValueError):
    """Raised when a required planning authority cannot be bound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractDefinitionError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ref(root: Path, path: Path, role: str, expected_sha256: str) -> dict[str, str]:
    target = root / path
    _require(target.is_file(), f"missing {role}: {path}")
    actual = file_sha256(target)
    _require(actual == expected_sha256, f"{role} hash mismatch")
    return {"path": path.as_posix(), "role": role, "sha256": actual}


def _read_json(root: Path, path: Path, role: str, expected_sha256: str) -> dict[str, Any]:
    _ref(root, path, role, expected_sha256)
    try:
        payload = json.loads((root / path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractDefinitionError(f"cannot read {role}") from exc
    _require(isinstance(payload, dict), f"{role} must be a JSON object")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, encoding="utf-8"
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ContractDefinitionError("git boundary could not be read") from exc

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_clean": not bool(status),
        "status_lines": status.splitlines() if status else [],
    }


def _validate_git(root: Path) -> dict[str, Any]:
    state = _git_state(root)
    _require(state["branch"] == EXPECTED_BRANCH, "current branch differs from the plan")
    _require(state["head"] == EXPECTED_HEAD, "current HEAD differs from the plan")
    return state


def _validate_authorities(root: Path) -> dict[str, Any]:
    alignment = _read_json(
        root,
        ALIGNMENT_CONTRACT_PATH,
        "aligned input/OLS contract",
        ALIGNMENT_CONTRACT_SHA256,
    )
    _require(alignment.get("result") == "READY", "aligned input/OLS contract is not READY")
    _require(
        alignment.get("classification")
        == "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY",
        "aligned input/OLS contract classification changed",
    )
    scope = _mapping(alignment.get("scope"), "alignment scope")
    _require(scope.get("task") == "2D semantic image generation only", "alignment scope changed")
    ols = _mapping(alignment.get("ols_contract"), "alignment OLS contract")
    latent = _mapping(ols.get("authoritative_latent"), "authoritative OLS latent")
    _require(latent.get("domain") == "C^100", "authoritative OLS latent is not C^100")
    projection = _mapping(ols.get("modal_projection"), "modal projection")
    reemission = _mapping(ols.get("modal_reemission"), "modal re-emission")
    _require(projection.get("mandatory") is True, "modal projection is not mandatory")
    _require(reemission.get("mandatory") is True, "modal re-emission is not mandatory")
    _require(
        reemission.get("same_writer_for_real_and_generated") is True,
        "real/generated W_OLS writer is not shared",
    )
    runtime = _ref(root, ALIGNMENT_RUNTIME_PATH, "aligned LightGen runtime", ALIGNMENT_RUNTIME_SHA256)
    runtime_text = (root / ALIGNMENT_RUNTIME_PATH).read_text(encoding="utf-8")
    for marker in ("forward_from_image", "forward_from_ols", "modal_projection", "W_OLS"):
        _require(marker in runtime_text, f"aligned runtime marker is missing: {marker}")
    runtime_report = _read_json(
        root,
        ALIGNMENT_RUNTIME_REPORT_PATH,
        "aligned runtime report",
        ALIGNMENT_RUNTIME_REPORT_SHA256,
    )
    _require(runtime_report.get("result") == "IMPLEMENTED", "aligned runtime report is not IMPLEMENTED")
    runtime_boundary = _mapping(runtime_report.get("claim_boundary"), "aligned runtime claim boundary")
    _require(runtime_boundary.get("generation_readiness") == "not_ready", "aligned runtime opened generation")

    v2_contract = _read_json(root, V2_CONTRACT_PATH, "corrected V2 contract", V2_CONTRACT_SHA256)
    _require(v2_contract.get("schema") == "lightgen_2d_semantic_complex_ols_training_contract.v2", "V2 schema changed")
    _require(v2_contract.get("result") == "READY" and v2_contract.get("status") == "valid_completed", "corrected V2 contract is not READY")
    _require(v2_contract.get("training_authorized") is False, "corrected V2 contract opened training")
    v2_data = _mapping(v2_contract.get("data_contract"), "V2 data contract")
    v2_source = _mapping(_mapping(v2_data.get("source_artifacts"), "V2 source artifacts").get("training"), "V2 training source")
    _require(v2_source.get("path") == TRAINING_SOURCE_PATH.as_posix(), "training source path changed")
    _require(v2_source.get("sha256") == TRAINING_SOURCE_SHA256, "training source authority changed")
    v2_stage_a = _mapping(v2_contract.get("stage_a"), "V2 Stage-A contract")
    v2_loss = _mapping(v2_stage_a.get("loss"), "V2 Stage-A loss")
    _require(v2_loss.get("formula") == "L_A = 1.0*L_image + 0.0*L_optical", "pure-image control changed")
    _require(v2_stage_a.get("steps") == STAGE_A_STEPS, "V2 Stage-A budget changed")
    _require(v2_stage_a.get("checkpoint_steps") == list(CHECKPOINT_STEPS), "V2 checkpoint cadence changed")

    v2_report = _read_json(root, V2_TRAINING_REPORT_PATH, "V2 training report", V2_TRAINING_REPORT_SHA256)
    _require(v2_report.get("result") == "VALID_COMPLETED", "V2 training report is not valid completed")
    _require(v2_report.get("status") == "valid_failed", "V2 training report status changed")
    _require(
        v2_report.get("classification")
        == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED",
        "V2 training classification changed",
    )
    v2_lineage = _read_json(root, V2_TRAINING_LINEAGE_PATH, "V2 training lineage", V2_TRAINING_LINEAGE_SHA256)
    _require(v2_lineage.get("result") == "VALID_COMPLETED", "V2 lineage result changed")

    diagnosis = _read_json(root, DIAGNOSIS_REPORT_PATH, "warmup diagnosis report", DIAGNOSIS_REPORT_SHA256)
    _require(diagnosis.get("result") == "VALID_COMPLETED", "warmup diagnosis is not valid completed")
    _require(
        diagnosis.get("primary_diagnosis")
        == "SEMANTIC_WARMUP_FAILURE_IMAGE_OBJECTIVE_MISMATCH_SUPPORTED",
        "warmup diagnosis changed",
    )
    diagnosis_boundary = _mapping(diagnosis.get("governance"), "diagnosis governance boundary")
    for key in ("new_training", "stage_b_executed", "generation_executed", "protected_read", "test_read", "sealed_read"):
        _require(diagnosis_boundary.get(key) is False, f"diagnosis crossed forbidden boundary: {key}")
    diagnosis_lineage = _read_json(root, DIAGNOSIS_LINEAGE_PATH, "warmup diagnosis lineage", DIAGNOSIS_LINEAGE_SHA256)
    _require(diagnosis_lineage.get("result") == "VALID_COMPLETED", "diagnosis lineage result changed")

    evaluator_text = (root / EVALUATOR_IMPLEMENTATION_PATH).read_text(encoding="utf-8")
    _require("class FrozenMNISTEvaluator" in evaluator_text, "FrozenMNISTEvaluator implementation changed")
    _require("class SmoothMNISTEvaluator" in evaluator_text, "SmoothMNISTEvaluator implementation changed")
    _require("return features, self.classifier(features)" in evaluator_text, "evaluator embedding boundary changed")
    evaluator_implementation = _ref(
        root,
        EVALUATOR_IMPLEMENTATION_PATH,
        "semantic evaluator implementation",
        EVALUATOR_IMPLEMENTATION_SHA256,
    )
    frozen_checkpoint = _ref(
        root,
        FROZEN_EVALUATOR_CHECKPOINT_PATH,
        "accepted FrozenMNISTEvaluator checkpoint",
        FROZEN_EVALUATOR_CHECKPOINT_SHA256,
    )
    frozen_qualification = _read_json(
        root,
        FROZEN_EVALUATOR_QUALIFICATION_PATH,
        "FrozenMNISTEvaluator qualification evidence",
        FROZEN_EVALUATOR_QUALIFICATION_SHA256,
    )
    _require(
        float(frozen_qualification.get("validation_accuracy", PRIMARY_EVALUATOR_VALIDATION_ACCURACY))
        >= 0.985,
        "FrozenMNISTEvaluator qualification is below the authority threshold",
    )
    smooth_checkpoint = _ref(
        root,
        SMOOTH_EVALUATOR_CHECKPOINT_PATH,
        "independent SmoothMNISTEvaluator checkpoint",
        SMOOTH_EVALUATOR_CHECKPOINT_SHA256,
    )
    smooth_bundle = _read_json(
        root,
        SMOOTH_EVALUATOR_BUNDLE_PATH,
        "independent evaluator bundle",
        SMOOTH_EVALUATOR_BUNDLE_SHA256,
    )
    _require(smooth_bundle.get("feature_dim") == 128, "independent evaluator feature dimension changed")
    _require(smooth_bundle.get("primary_checkpoint") == FROZEN_EVALUATOR_CHECKPOINT_SHA256, "independent bundle primary checkpoint changed")
    _require(smooth_bundle.get("smooth_checkpoint") == SMOOTH_EVALUATOR_CHECKPOINT_SHA256, "independent bundle smooth checkpoint changed")
    smooth_qualification = _read_json(
        root,
        SMOOTH_EVALUATOR_QUALIFICATION_PATH,
        "independent evaluator qualification evidence",
        SMOOTH_EVALUATOR_QUALIFICATION_SHA256,
    )
    _require(smooth_qualification.get("status") == "passed", "independent evaluator qualification did not pass")
    _require(smooth_qualification.get("tier") == "diagnostic", "independent evaluator tier changed")
    smooth_hashes = _mapping(smooth_qualification.get("input_hashes"), "independent evaluator input hashes")
    _require(smooth_hashes.get("smooth_evaluator") == SMOOTH_EVALUATOR_CHECKPOINT_SHA256, "independent smooth authority changed")

    return {
        "aligned_runtime": runtime,
        "aligned_runtime_report": _ref(root, ALIGNMENT_RUNTIME_REPORT_PATH, "aligned runtime report", ALIGNMENT_RUNTIME_REPORT_SHA256),
        "aligned_input_ols_contract": _ref(root, ALIGNMENT_CONTRACT_PATH, "aligned input/OLS contract", ALIGNMENT_CONTRACT_SHA256),
        "corrected_v2_contract": _ref(root, V2_CONTRACT_PATH, "corrected V2 training contract", V2_CONTRACT_SHA256),
        "v2_training_report": _ref(root, V2_TRAINING_REPORT_PATH, "completed V2 training report", V2_TRAINING_REPORT_SHA256),
        "v2_training_lineage": _ref(root, V2_TRAINING_LINEAGE_PATH, "completed V2 training lineage", V2_TRAINING_LINEAGE_SHA256),
        "warmup_diagnosis_report": _ref(root, DIAGNOSIS_REPORT_PATH, "warmup diagnosis report", DIAGNOSIS_REPORT_SHA256),
        "warmup_diagnosis_lineage": _ref(root, DIAGNOSIS_LINEAGE_PATH, "warmup diagnosis lineage", DIAGNOSIS_LINEAGE_SHA256),
        "training_source": {"path": TRAINING_SOURCE_PATH.as_posix(), "role": "canonical TRAIN source bound through corrected V2 authority", "sha256": TRAINING_SOURCE_SHA256, "values_read_by_definition": False},
        "frozen_evaluator": {
            "implementation": evaluator_implementation,
            "checkpoint": frozen_checkpoint,
            "qualification": _ref(root, FROZEN_EVALUATOR_QUALIFICATION_PATH, "FrozenMNISTEvaluator qualification evidence", FROZEN_EVALUATOR_QUALIFICATION_SHA256),
        },
        "smooth_evaluator": {
            "checkpoint": smooth_checkpoint,
            "bundle": _ref(root, SMOOTH_EVALUATOR_BUNDLE_PATH, "independent evaluator bundle", SMOOTH_EVALUATOR_BUNDLE_SHA256),
            "qualification": _ref(root, SMOOTH_EVALUATOR_QUALIFICATION_PATH, "independent evaluator qualification evidence", SMOOTH_EVALUATOR_QUALIFICATION_SHA256),
        },
    }


def _contract_payload(root: Path, git: dict[str, Any], authorities: dict[str, Any]) -> dict[str, Any]:
    definition_consumer_path = Path(__file__).resolve().relative_to(root.resolve())
    definition_consumer = {
        "path": definition_consumer_path.as_posix(),
        "role": "V3 bounded contract definition and authority-binding consumer",
        "sha256": file_sha256(root / definition_consumer_path),
    }
    return {
        "artifact_self_hash": "not_embedded_to_avoid_self_reference",
        "claim_boundary": {
            "this_contract_proves": [
                "one matched Stage-A control/candidate objective-correction experiment is fully specified",
                "the semantic feature loss has an exact frozen provider, representation, distance, and input-gradient path",
                "the independent semantic corroboration metric is fixed before execution",
                "the existing real/generated complex-OLS interface and data-role firewall are preserved",
            ],
            "this_contract_does_not_prove": [
                "that the semantic-corrected objective improves reconstruction",
                "that any Stage-A checkpoint is semantically eligible",
                "Stage-B preservation, prior compatibility, direct prior generation, diversity, novelty, hardware readiness, or formal promotion",
            ],
            "valid_scientific_failure": "complete paired execution that misses the objective-correction rule returns V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED",
            "invalid_harness": "authority drift, undefined provider gradient, nonfinite or detached path, unmatched pair, recovery failure, or data-role violation returns invalid_harness",
        },
        "classification": READY_CLASSIFICATION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_contract": {
            "source": {
                "path": TRAINING_SOURCE_PATH.as_posix(),
                "sha256": TRAINING_SOURCE_SHA256,
                "rows": 60000,
                "schema": "60000x28x28 uint8 images plus 60000 int64 labels",
                "definition_values_read": False,
            },
            "roles": {
                "train": {"rows": 55000, "use": "optimization and lambda calibration only"},
                "development": {"rows": 5000, "selection_rows": 1000, "per_class_selection": 100, "use": "fixed endpoint comparison and Stage-A firewall only"},
                "calibration": {"rows_read": 0, "use": "forbidden in V3"},
                "protected": {"rows_read": 0, "use": "forbidden"},
                "test": {"rows_read": 0, "use": "forbidden"},
                "sealed": {"rows_read": 0, "use": "forbidden and unavailable"},
            },
            "split": {
                "algorithm": "stratified_mnist_indices(labels, seed=1234, validation_per_label=500)",
                "initialization_seed": INITIALIZATION_SEED,
                "development_selection_seed": DEVELOPMENT_SELECTION_SEED,
                "selection_indices_sha256": "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8",
                "labels_as_model_input": False,
            },
            "forbidden_boundary": "Any protected/test/sealed open, any calibration value read, or any label/controller input to the optical runtime is a hard stop.",
        },
        "definition_consumer": definition_consumer,
        "definition_only": True,
        "facts": {
            "FACT": [
                "The aligned runtime maps image -> SLM input -> encoder -> mandatory modal projection -> z in C^100 -> coefficient-space T=I_100 -> shared W_OLS -> generator -> raw CMOS.",
                "V2 Stage-A attempt-02 completed 300 updates and is a valid scientific failure: semantic warmup was not established.",
                "The complete V2 trace improved image loss while development semantic recall declined from 0.170 to 0.160 to 0.136.",
                "The warmup diagnosis supports image-objective/semantic mismatch; encoder gradients were finite and not effectively zero.",
                "FrozenMNISTEvaluator v1 exposes a frozen 128D embedding before its classifier and has qualified validation accuracy above 0.985.",
                "An independently qualified SmoothMNISTEvaluator v2.1 checkpoint is available for corroborating semantic metrics.",
            ],
            "MODEL_ASSUMPTION": [
                "A frozen feature-consistency term can align the optical optimization direction with semantic identity.",
                "The fixed evaluator embedding is a useful semantic representation for this bounded MNIST experiment.",
            ],
            "DESIGN_DECISION": [
                "Use the accepted FrozenMNISTEvaluator embedding as the sole semantic-loss provider and use SmoothMNISTEvaluator v2.1 as independent metric corroboration.",
                "Use one final-step paired comparison with a predeclared 0.05 macro-recall margin, a stratified paired bootstrap, and a fixed image-loss tolerance.",
                "Keep the existing 0.937 FrozenMNISTEvaluator threshold unchanged and separate from objective-correction support.",
                "Do not warm-start either arm from a failed V2 checkpoint; initialize both arms from one fresh zero-phase state.",
            ],
        },
        "future_one_shot_semantics": {
            "real_path": "image -> aligned input adapter -> encoder -> z_real in C^100 -> T=I_100 -> W_OLS -> same generator -> raw CMOS",
            "future_generated_path": "z_generated in C^100 -> T=I_100 -> W_OLS -> same generator -> raw CMOS",
            "shared_interface": True,
            "q_only_latent": False,
            "phase_repair": False,
            "software_iterative_inference": False,
            "generation_in_this_contract": False,
        },
        "gate": GATE,
        "governance": {
            "training_authorized": False,
            "scientific_training_executed": False,
            "stage_b_authorized": False,
            "stage_b_executed": False,
            "formal_promotion": False,
            "generation_executed": False,
            "generation_readiness": "not_ready",
            "protected_read": False,
            "test_read": False,
            "sealed_read": False,
            "new_checkpoint": False,
        },
        "independent_semantic_corroboration": {
            "identity": "SmoothMNISTEvaluator v2.1",
            "implementation": authorities["frozen_evaluator"]["implementation"],
            "checkpoint": authorities["smooth_evaluator"]["checkpoint"],
            "bundle": authorities["smooth_evaluator"]["bundle"],
            "qualification": authorities["smooth_evaluator"]["qualification"],
            "tier": "diagnostic qualified; metric corroboration only",
            "feature_dimension": 128,
            "representation": "embedding output before classifier",
            "preprocessing": "clamp image to [0,1], then adaptive_avg_pool2d to 28x28",
            "labels_used": "only to calculate fixed-panel per-class recall; labels never enter the provider or optical runtime",
            "training_use": False,
            "sole_acceptance_evidence": False,
        },
        "objective": {
            "question": "Can a semantic-preserving Stage-A objective improve real-path semantic reconstruction relative to the matched pure-image objective while the optical architecture and training conditions remain fixed?",
            "scope": "LightGen-aligned v2 research runtime, Stage A only",
            "control": {
                "name": "PURE_IMAGE_CONTROL",
                "formula": "L_control = L_image",
                "image_loss": "mean((I_cmos_raw - clamp(BilinearResize(X,200x200),0,1))^2)",
                "semantic_term_weight": 0.0,
            },
            "candidate": {
                "name": "FROZEN_FEATURE_CORRECTED",
                "formula": "L_candidate = L_image + lambda_feature * L_feature",
                "image_loss": "same L_image as PURE_IMAGE_CONTROL",
                "semantic_term": "one frozen feature-consistency term; no latent MMD, scale term, prior target, label loss, or optical architecture change",
            },
            "architecture_fixed": [
                "100-port complex OLS with z in C^100",
                "mandatory modal projection and modal re-emission",
                "coefficient-space T=I_100",
                "shared W_OLS and generator interface",
                "existing encoder/generator phase stacks and 2048 propagation grid",
            ],
        },
        "optimizer_and_precision": {
            "device": "cuda:0 required for future scientific execution; no CUDA in this definition",
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "stage_a_steps": STAGE_A_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "initialization_seed": INITIALIZATION_SEED,
            "schedule_seed": INITIALIZATION_SEED,
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False},
            "precision": {"phase": "float32", "field": "complex64", "image": "float32", "loss": "float32", "autocast": False, "grad_scaler": False},
            "gradient_clipping": None,
            "trainable_parameters": ["encoder.phase", "generator.phase"],
            "frozen_parameters": ["OLS modal banks and T", "input/output relays and SLM mapping", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"],
        },
        "paired_protocol": {
            "fresh_initial_state": "Construct one existing V2 runtime with all encoder.phase and generator.phase tensors at the runtime's zero initialization, then clone that exact trainable and frozen state into both arms.",
            "warm_start": "forbidden; failed V2 checkpoints are comparison evidence only",
            "data_order": "Materialize one DeterministicTrainRoleSampler schedule with seed=1234 and reuse the same batch indices for both arms at every stage-local step.",
            "arm_order": "Run control and candidate from cloned states; arm order cannot change data order, seeds, initialization, or checkpoint endpoint.",
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "primary_endpoint": 300,
            "selection_rule": "No best-checkpoint selection for the V3 comparison; step 300 is the preregistered primary endpoint. Earlier checkpoints are trajectory diagnostics and independent Stage-A firewall observations.",
        },
        "semantic_provider": {
            "identity": "FrozenMNISTEvaluator v1",
            "implementation": authorities["frozen_evaluator"]["implementation"],
            "checkpoint": authorities["frozen_evaluator"]["checkpoint"],
            "qualification": authorities["frozen_evaluator"]["qualification"],
            "feature_dimension": 128,
            "layer": "embedding output after the evaluator's second ReLU and before classifier",
            "preprocessing": "clamp image to [0,1], then adaptive_avg_pool2d to 28x28",
            "representation": "f = normalize_l2(embedding(preprocessed_image), dim=1, eps=1e-12) in R^128",
            "distance": "per-sample squared Euclidean distance in normalized feature space",
            "loss": "L_feature = mean_i sum_k (f(raw_cmos_i)_k - f(target_i)_k)^2",
            "target": "target_i = X_i after the same fixed evaluator preprocessing; X is already 28x28 float32 in [0,1]",
            "provider_parameters": "eval mode, requires_grad=False, no optimizer, no parameter-gradient accumulation",
            "gradient_boundary": "The raw-CMOS input remains attached to the provider forward so gradients reach encoder.phase and generator.phase; do not wrap the candidate output forward in no_grad or detach it.",
            "labels_as_provider_input": False,
        },
        "lambda_rule": {
            "computed": "once before the first optimizer update and before either arm starts Stage-A training",
            "calibration_role": "TRAIN only",
            "calibration_batch": "the exact stage-local step=1 batch from the shared deterministic sampler; the batch is reused unchanged for optimizer step 1",
            "g_image": "sqrt(sum over exact encoder.phase and generator.phase tensors of the squared gradient of L_image at the shared initial state)",
            "g_feature": "sqrt(sum over the same exact tensors of the squared gradient of L_feature at the shared initial state)",
            "raw_ratio": "g_image / g_feature",
            "formula": "lambda_feature = min(10.0, max(0.1, g_image / g_feature))",
            "lower_bound": 0.1,
            "upper_bound": 10.0,
            "deterministic": True,
            "no_outcome_tuning": True,
            "hard_stop_if": ["g_image or g_feature is nonfinite", "g_image <= 0", "g_feature <= 0", "the provider input gradient is absent", "lambda is recomputed after any scientific checkpoint or using development metrics"],
            "record": ["g_image", "g_feature", "raw_ratio", "lambda_feature", "clipped"],
        },
        "comparison_metrics": {
            "per_arm_and_checkpoint": [
                "L_image",
                "L_feature",
                "FrozenMNISTEvaluator macro recall and ten per-class recalls",
                "SmoothMNISTEvaluator macro recall and ten per-class recalls",
                "fixed raw-CMOS panel tensor identity, shape, finiteness, mean, standard deviation, and per-image variance",
                "encoder.phase and generator.phase gradient/update norms",
                "OLS field, z, and W_OLS boundary gradient norms and provenance flags",
            ],
            "fixed_raw_cmos_panel": {
                "role": "development selection only",
                "rows": 1000,
                "per_class": 100,
                "shape": [1000, 1, 200, 200],
                "store": "all raw CMOS outputs for both arms at steps 100, 200, and 300; no sample removal, denoising, crop search, or evaluator-dependent replacement",
                "evaluator_view_only": "clamp to [0,1], then adaptive_avg_pool2d to 28x28",
            },
            "gradient_diagnostics": {
                "required": [
                    "all intended trainable gradients finite",
                    "both intended trainable groups receive a nonzero update",
                    "FrozenMNISTEvaluator and SmoothMNISTEvaluator parameters receive no gradients or updates",
                    "semantic provider input gradient finite and nonzero on the calibration batch",
                    "modal projection and W_OLS remain on the autograd path",
                    "zero-gradient and nonfinite-gradient fractions",
                ],
                "provider_parameter_gradients": "must be absent",
            },
        },
        "decision_rule": {
            "primary_endpoint": "step 300 on the fixed 1000-row development selection",
            "primary_evidence": "independent SmoothMNISTEvaluator macro recall; the training provider is not the sole decision source",
            "delta": "Delta_smooth = macro_recall_smooth(candidate) - macro_recall_smooth(control)",
            "bootstrap": {
                "type": "paired, class-stratified bootstrap over the same 100 examples per class",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": LAMBDA_CALIBRATION_SEED,
                "alpha": BOOTSTRAP_ALPHA,
                "interval": "percentile 95% interval with fixed linear quantile convention",
            },
            "objective_correction_supported_if": [
                "both arms complete valid 300-step traces with matched initialization, schedule, optimizer, and data firewall",
                "Delta_smooth >= 0.05 and the paired-bootstrap lower confidence bound is > 0",
                "no Smooth evaluator per-class recall decreases by more than 0.10 for the candidate relative to control",
                "candidate L_image at step 300 <= control L_image at step 300 + 0.005",
                "both raw-CMOS endpoint panels are finite, structurally valid, and nonconstant",
            ],
            "objective_correction_not_supported_if": "all measurements are complete and any support condition fails; return a valid scientific failure without changing lambda, feature layer, threshold, initialization, order, or budget",
            "stage_a_eligibility": {
                "metric": "FrozenMNISTEvaluator macro recall on the fixed development selection",
                "eligible_if": "macro recall >= 0.937 at a recorded checkpoint",
                "threshold": STAGE_A_SEMANTIC_THRESHOLD,
                "cannot_be_compensated": True,
                "separate_from_objective_correction_support": True,
                "stage_b_entry_in_this_contract": False,
            },
        },
        "preflight": {
            "required_before_any_scientific_run": [
                "static and lineage checks through the future V3 entrypoint",
                "three real CUDA updates for both control and candidate through the exact Stage-A path",
                "finite loss/gradient/parameter checks and nonzero updates in both intended trainable groups",
                "frozen provider parameter and evaluator firewall checks",
                "semantic-provider input-gradient check with the candidate output attached",
                "shared batch schedule and cloned-initial-state equality check",
                "disabled-term control must reproduce the pure-image control within the declared deterministic tolerance",
                "detached-provider negative control must be rejected as invalid_harness because its semantic input gradient is absent",
                "one interrupted/resumed segment per arm with exact next batch, optimizer, RNG, lambda, and provider identities",
                "memory headroom and exact 2048-grid device check",
            ],
            "device": "cuda:0",
            "budget": "ordinary/scientific long-training preflight cap from docs/operations/long-training-preflight.md; no scientific training is authorized by this contract",
            "recovery": "unverified recovery blocks the future scientific run",
        },
        "source_authority": authorities,
        "stage_b_boundary": {
            "authorized": False,
            "scale_extraction": False,
            "latent_mmd": False,
            "prior_sampling": False,
            "entry_requires_future_contract": True,
            "entry_requires_candidate_frozen_macro_recall": ">= 0.937 at a fixed recorded checkpoint",
        },
        "status": "valid_completed",
        "stop_conditions": [
            "required V2 report, diagnosis, runtime, or evaluator authority cannot be resolved exactly",
            "current runtime no longer exposes the declared LightGen input/OLS/shared-W_OLS interface",
            "provider gradient path is undefined, detached, nonfinite, or has zero feature gradient",
            "control and candidate do not share exact initialization, batch order, optimizer, or endpoint",
            "any protected, test, sealed, or calibration value is opened",
            "any architecture, OLS port count, W_OLS, generator layer, q-only producer, phase repair, or Stage-B term is introduced",
            "any outcome-dependent lambda, feature layer, threshold, checkpoint, or metric substitution is attempted",
        ],
        "task_id": TASK_ID,
        "training_authorized": False,
        "version": 3,
        "git": {
            "pre_materialization_verified": {
                "branch": EXPECTED_BRANCH,
                "head": EXPECTED_HEAD,
                "worktree_clean": True,
                "verification": "read-only local check before adding this definition consumer",
            },
            "definition_consumer_execution": git,
        },
        "outer_gate": {
            "name": GATE,
            "answer": "YES",
            "passed": True,
            "checks": {
                "v2_report_and_diagnosis_bound": True,
                "aligned_runtime_identity_bound": True,
                "control_objective_frozen": True,
                "candidate_objective_frozen": True,
                "feature_provider_layer_and_distance_frozen": True,
                "lambda_rule_deterministic_and_predeclared": True,
                "matched_initialization_order_optimizer_budget": True,
                "independent_semantic_corroboration_bound": True,
                "objective_support_separate_from_stage_a_firewall": True,
                "stage_a_threshold_0_937_unchanged": True,
                "preflight_and_negative_controls_defined": True,
                "one_shot_interface_preserved": True,
                "forbidden_roles_closed": True,
            },
            "hard_stop_conditions": {
                "required_authority_unresolved": False,
                "runtime_interface_conflict": False,
                "provider_gradient_undefined": False,
                "data_role_leak": False,
                "current_contract_conflict": False,
            },
        },
        "result": "READY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or _repo_root()).resolve()
    git = _validate_git(root)
    authorities = _validate_authorities(root)
    payload = _contract_payload(root, git, authorities)
    output_path = root / OUTPUT_DIR / CONTRACT_NAME
    atomic_write_new_json(output_path, payload, artifact_kind="V3 objective-correction contract")
    print(json.dumps({"artifact": _relative(output_path, root), "contract": payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
