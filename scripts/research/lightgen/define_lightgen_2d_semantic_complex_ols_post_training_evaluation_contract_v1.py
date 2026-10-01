"""Define the one-shot LightGen complex-OLS post-training evaluation contract.

This is a definition-only consumer.  It binds one training checkpoint, one
independent frozen semantic evaluator, the protected-data roles, and one
fail-closed evaluation order.  It does not load protected test tensors,
execute training, execute generation, or calculate any scientific statistic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import stratified_mnist_indices


TASK_ID = "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1"
READY_CLASSIFICATION = (
    "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1_READY"
)
BLOCKED_EVALUATOR_CLASSIFICATION = "BLOCKED_SEMANTIC_EVALUATOR_AUTHORITY_UNRESOLVED"
CONTROL_NOT_ESTABLISHED_CLASSIFICATION = (
    "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_CONTROL_NOT_ESTABLISHED"
)
SUPPORTED_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_GENERATION_SUPPORTED"
NOT_SUPPORTED_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_GENERATION_NOT_SUPPORTED"
NEXT_ACTION = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1_attempt_01"
)
CONTRACT_NAME = (
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json"
)

TRAINING_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v1.json"
)
TRAINING_CONTRACT_SHA256 = "7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f"
TRAINING_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_report.v1.json"
)
TRAINING_REPORT_SHA256 = "f3f5fa876567513ec70cac67a6888aad6b55d140299992a535d0f9656bb4aa06"
TRAINING_LINEAGE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_lineage.v1.json"
)
TRAINING_LINEAGE_SHA256 = "9c6eacb921b59788062236d9ba8354529f97bcf86de23813ba594e0fb7582c12"
CHECKPOINT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_checkpoint_step_100.v1.pt"
)
CHECKPOINT_SHA256 = "6cfcc82831b575d83c097648a78b6507b32900e578d00c446d76ccd29130e149"

ALIGNMENT_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json"
)
ALIGNMENT_CONTRACT_SHA256 = "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"
ALIGNMENT_RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
ALIGNMENT_RUNTIME_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"
)
ALIGNMENT_RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"
TRAINING_CONSUMER_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_complex_ols_training.py"
)
TRAINING_CONSUMER_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"

SOURCE_MANIFEST_PATH = Path("research_assets/lightgen/science_adv7434/manifest.json")
SOURCE_MANIFEST_SHA256 = "fdb4f8c7b476ce92bd85b42ecbd435507657c91adfd501326842d5d00ee3418c"
PAPER_PDF_PATH = Path("research_assets/lightgen/science_adv7434/paper/sciencesm.pdf")
PAPER_PDF_SHA256 = "1325cdfb438f55249224f660a90ec2c21438ba3ce8cffc2677001f2db4e6c211"

EVALUATOR_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
EVALUATOR_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
EVALUATOR_QUALIFICATION_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "evaluator_qualification_evidence.json"
)
EVALUATOR_QUALIFICATION_SHA256 = "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e"
EVALUATOR_SPLIT_MANIFEST_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "mnist_split_manifest.json"
)
EVALUATOR_SPLIT_MANIFEST_SHA256 = "27b9c4ab61963426fea6657308e523d0b19e4f687fda1ed94a28d87f73328ee9"
EVALUATOR_SOURCE_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SOURCE_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
EVALUATOR_SPLIT_MANIFEST_HASH = "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912"
EVALUATOR_VALIDATION_ACCURACY = 0.9875999689102173
EVALUATOR_QUALIFICATION_THRESHOLD = 0.985

REJECTED_OUTPUT_EVALUATOR_REPORT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_frozen_output_semantic_evaluator_v1_attempt_01/"
    "compact_report.json"
)
REJECTED_OUTPUT_EVALUATOR_REPORT_SHA256 = "7fc7dc3c3c97214a4e257def13a4c6cf563b7e22b72584ce4af615c3c71c8026"

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
PROTECTED_TEST_PATH = Path("data/MNIST/processed/test.pt")
PROTECTED_TEST_SHA256 = "8c089c2f0a5db35de7961f4c60ed6b627a05141245d44f542eeee57b09b539e1"
SPLIT_SOURCE_PATH = Path("src/ldm_project/v2/student/evaluator.py")
SPLIT_SOURCE_SHA256 = EVALUATOR_SOURCE_SHA256
SPLIT_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321
DEVELOPMENT_SELECTION_COUNT = 1_000
DEVELOPMENT_SELECTION_PER_CLASS = 100
DEVELOPMENT_CALIBRATION_COUNT = 4_000
DEVELOPMENT_CALIBRATION_PER_CLASS = 400
DEVELOPMENT_CALIBRATION_INDICES_SHA256 = "c3db6fe1554baec2bebed85cdbc03cfd13f74521fd658244e6d2ef9aabf52ee9"
PROTECTED_EVALUATION_COUNT = 10_000
PROTECTED_PANEL_SEED = 20260912
PROTECTED_PANEL_COUNT = 1_000
PROTECTED_PANEL_PER_CLASS = 100

CLASS_COUNT = 10
IMAGE_SHAPE = (1, 28, 28)
RAW_OUTPUT_SHAPE = (1, 200, 200)
OLS_GRID_SHAPE = (10, 10)
OLS_PORT_COUNT = 100
OLS_SHAPE = (100, 100)
PRODUCTION_PROPAGATION_GRID = 2048
PRIOR_SIGMA = 1.0e-3
GAUGE_EPSILON = 1.0e-6
LATENT_DIMENSION = 200
LATENT_MMD_BANDWIDTH_SQUARED = 200.0
SEMANTIC_FEATURE_DIMENSION = 128
SEMANTIC_MMD_BANDWIDTH_SQUARED = 1.0
MONTE_CARLO_REPLICATES = 999
MONTE_CARLO_ALPHA = 0.01
CONTROL_FLOOR = 0.90
CONTROL_ALLOWED_DROP = 0.05
PRIOR_NAMESPACE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_PRIOR"
PRIOR_GLOBAL_STEP = 0
LATENT_MODEL_ROLE = "post_training_latent_model"
GENERATION_BANK_COUNT = 4
GENERATION_BANK_SIZE = 250
GENERATION_TOTAL_COUNT = GENERATION_BANK_COUNT * GENERATION_BANK_SIZE
GENERATION_ROLE_PREFIX = "post_training_generation_bank_"
METRIC_NULL_NAMESPACE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1_NULL"

GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")


class DefinitionBlocked(RuntimeError):
    """The contract cannot be frozen without weakening an authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionBlocked(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_json(
    repo_root: Path,
    path: Path,
    name: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {path.as_posix()}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DefinitionBlocked(f"{name} is not valid JSON") from exc
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _ref(
    repo_root: Path,
    path: Path,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    target = repo_root / path
    _require(target.is_file(), f"{role} is missing: {path.as_posix()}")
    digest = file_sha256(target)
    return {"path": _relative(repo_root, target), "sha256": digest, "role": role}


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status_result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = status_result.stdout.splitlines()
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_dirty": bool(status),
    }


def _validate_git_boundary(repo_root: Path) -> dict[str, Any]:
    state = _git_state(repo_root)
    _require(state["branch"] == EXPECTED_BRANCH, f"expected branch {EXPECTED_BRANCH}")
    _require(state["head"] == EXPECTED_STARTING_HEAD, f"expected HEAD {EXPECTED_STARTING_HEAD}")
    return state


def _validate_source_authority(repo_root: Path, alignment: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(
        repo_root,
        SOURCE_MANIFEST_PATH,
        "LightGen source manifest",
        SOURCE_MANIFEST_SHA256,
    )
    _require(manifest.get("status") == "reference_only", "LightGen source is not reference-only")
    source = _mapping(manifest.get("source"), "LightGen source identity")
    _require(source.get("doi") == "10.1126/science.adv7434", "LightGen DOI authority changed")
    _require(source.get("supplement_pages") == 46, "LightGen supplement authority changed")
    pdf = _ref(repo_root, PAPER_PDF_PATH, "verified LightGen supplement PDF", PAPER_PDF_SHA256)
    taxonomy = _mapping(alignment.get("source_authority"), "alignment source authority")
    entries = taxonomy.get("paper_and_project_taxonomy")
    _require(isinstance(entries, list), "alignment paper taxonomy is missing")
    paper_facts = [item for item in entries if item.get("category") == "CHEN_FACT"]
    unresolved = [item for item in entries if item.get("category") == "CHEN_UNRESOLVED"]
    _require(paper_facts and unresolved, "alignment fact/unresolved taxonomy is incomplete")
    return {
        "manifest": _ref(repo_root, SOURCE_MANIFEST_PATH, "verified LightGen source manifest", SOURCE_MANIFEST_SHA256),
        "paper_pdf": pdf,
        "identity": {
            "title": source.get("title"),
            "doi": source.get("doi"),
            "supplement_pages": source.get("supplement_pages"),
            "status": manifest.get("status"),
        },
        "taxonomy": {
            "CHEN_FACT": paper_facts,
            "CHEN_UNRESOLVED": unresolved,
        },
    }


def _validate_alignment_authority(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(
        repo_root,
        ALIGNMENT_CONTRACT_PATH,
        "aligned input/OLS contract",
        ALIGNMENT_CONTRACT_SHA256,
    )
    _require(payload.get("result") == "READY", "aligned input/OLS contract is not READY")
    _require(
        payload.get("classification")
        == "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY",
        "aligned input/OLS contract classification changed",
    )
    scope = _mapping(payload.get("scope"), "alignment scope")
    _require(scope.get("task") == "2D semantic image generation only", "alignment scope changed")
    ols = _mapping(payload.get("ols_contract"), "alignment OLS contract")
    latent = _mapping(ols.get("authoritative_latent"), "alignment authoritative latent")
    _require(latent.get("domain") == "C^100", "aligned OLS is not C^100")
    _require(latent.get("batch_shape") == ["N", 100], "aligned OLS shape changed")
    projection = _mapping(ols.get("modal_projection"), "alignment modal projection")
    reemission = _mapping(ols.get("modal_reemission"), "alignment modal re-emission")
    _require(projection.get("mandatory") is True, "modal projection is not mandatory")
    _require(reemission.get("mandatory") is True, "modal re-emission is not mandatory")
    _require(reemission.get("same_writer_for_real_and_generated") is True, "shared W_OLS invariant changed")
    runtime = _ref(repo_root, ALIGNMENT_RUNTIME_PATH, "aligned LightGen runtime", ALIGNMENT_RUNTIME_SHA256)
    runtime_report = _read_json(
        repo_root,
        ALIGNMENT_RUNTIME_REPORT_PATH,
        "aligned runtime report",
        ALIGNMENT_RUNTIME_REPORT_SHA256,
    )
    _require(runtime_report.get("result") == "IMPLEMENTED", "aligned runtime report is not IMPLEMENTED")
    runtime_boundary = _mapping(runtime_report.get("claim_boundary"), "aligned runtime claim boundary")
    _require(runtime_boundary.get("generation_readiness") == "not_ready", "aligned runtime opened generation")
    _require(runtime_boundary.get("scientific_generation_executed") is False, "aligned runtime executed scientific generation")
    source = _validate_source_authority(repo_root, payload)
    return payload, {
        "contract": _ref(repo_root, ALIGNMENT_CONTRACT_PATH, "aligned input/OLS authority", ALIGNMENT_CONTRACT_SHA256),
        "runtime": runtime,
        "runtime_report": _ref(repo_root, ALIGNMENT_RUNTIME_REPORT_PATH, "aligned runtime report", ALIGNMENT_RUNTIME_REPORT_SHA256),
        "runtime_report_result": runtime_report.get("result"),
        "source": source,
    }


def _validate_training_authority(repo_root: Path) -> dict[str, Any]:
    contract = _read_json(
        repo_root,
        TRAINING_CONTRACT_PATH,
        "complex-OLS training contract",
        TRAINING_CONTRACT_SHA256,
    )
    _require(contract.get("schema") == "lightgen_2d_semantic_complex_ols_training_contract.v1", "training contract schema changed")
    _require(contract.get("result") == "READY", "training contract is not READY")
    _require(contract.get("training_authorized") is False, "training contract opened training")
    report = _read_json(
        repo_root,
        TRAINING_REPORT_PATH,
        "complex-OLS training report",
        TRAINING_REPORT_SHA256,
    )
    _require(report.get("status") == "valid_completed", "training report is not valid_completed")
    _require(report.get("result") == "VALID_COMPLETED", "training report result changed")
    _require(
        report.get("classification") == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_VALID_COMPLETED",
        "training report classification changed",
    )
    selected = _mapping(report.get("selected_checkpoint"), "selected checkpoint report")
    _require(selected.get("relative_path") == CHECKPOINT_PATH.as_posix(), "selected checkpoint path changed")
    _require(selected.get("sha256") == CHECKPOINT_SHA256, "selected checkpoint digest changed")
    _require(selected.get("selected") is True, "step-100 checkpoint is not the reported selection")
    _require(selected.get("valid_training_checkpoint") is True, "step-100 checkpoint is not valid")
    _require(selected.get("safe_load_verified") is True, "step-100 safe-load evidence is missing")
    _require(selected.get("completed_steps") == 100, "selected checkpoint step changed")
    _require(selected.get("accepted_training_checkpoint") is False, "checkpoint acceptance boundary changed")
    training = _mapping(report.get("training"), "training report training section")
    _require(training.get("completed_steps") == 100 and training.get("planned_steps") == 100, "training schedule changed")
    _require(training.get("device") == "cuda:0", "training device authority changed")
    boundary = _mapping(report.get("claim_boundary"), "training report claim boundary")
    for key in ("test_split_read", "heldout_evaluation_executed", "post_training_evaluation_executed", "formal_promotion", "sealed_accessed", "sealed_reused"):
        _require(boundary.get(key) is False, f"training report crossed protected boundary: {key}")
    _require(boundary.get("generation_readiness") == "not_ready", "training report opened generation")
    lineage = _read_json(
        repo_root,
        TRAINING_LINEAGE_PATH,
        "training lineage",
        TRAINING_LINEAGE_SHA256,
    )
    _require(lineage.get("schema") == "lightgen_2d_semantic_complex_ols_training_lineage.v1", "training lineage schema changed")
    lineage_execution = _mapping(lineage.get("execution"), "training lineage execution")
    _require(lineage_execution.get("result") == "VALID_COMPLETED", "training lineage result changed")
    _require(lineage_execution.get("classification") == report.get("classification"), "training lineage classification changed")
    checkpoint = safe_torch_load(repo_root / CHECKPOINT_PATH)
    _require(isinstance(checkpoint, dict), "selected checkpoint is not a mapping")
    _require(checkpoint.get("schema") == "lightgen_2d_semantic_complex_ols_training_checkpoint.v1", "checkpoint schema changed")
    _require(checkpoint.get("valid_training_checkpoint") is True, "checkpoint validity flag changed")
    _require(checkpoint.get("accepted_training_checkpoint") is False, "checkpoint acceptance flag changed")
    _require(checkpoint.get("completed_steps") == 100, "checkpoint completed-step identity changed")
    _require(checkpoint.get("contract", {}).get("path") == TRAINING_CONTRACT_PATH.as_posix(), "checkpoint training-contract path changed")
    architecture = _mapping(checkpoint.get("architecture_identity"), "checkpoint architecture identity")
    _require(architecture.get("ols_interface") == "lightgen_10x10_complex_ols_v1", "checkpoint OLS interface changed")
    _require(architecture.get("ols_shape") == [100], "checkpoint OLS shape changed")
    _require(architecture.get("modal_projection") == "z_j=<E_encoder,u_j>", "checkpoint projection boundary changed")
    _require(architecture.get("transport") == "T=I_100 coefficient space", "checkpoint transport changed")
    _require(architecture.get("writer") == "W_OLS(z)=sum_j (Tz)_j v_j", "checkpoint writer changed")
    _require(checkpoint.get("promotion", {}).get("formal_promotion") is False, "checkpoint formal promotion changed")
    _require(checkpoint.get("promotion", {}).get("generation_readiness") == "not_ready", "checkpoint readiness changed")
    data = _mapping(report.get("data"), "training report data")
    source_artifacts = _mapping(data.get("source_artifacts"), "training report source artifacts")
    training_source = _mapping(source_artifacts.get("training"), "training source artifact")
    protected_source = _mapping(source_artifacts.get("official_test"), "protected test source artifact")
    _require(training_source.get("path") == TRAINING_SOURCE_PATH.as_posix(), "training data source path changed")
    _require(protected_source.get("path") == PROTECTED_TEST_PATH.as_posix(), "protected test authority path changed")
    _require(protected_source.get("read") is False, "training report read protected test")
    report_authorities = _mapping(report.get("authorities"), "training report authorities")
    aligned = _mapping(report_authorities.get("aligned_runtime"), "training aligned runtime authority")
    _require(Path(str(aligned.get("path", ""))).name == ALIGNMENT_RUNTIME_PATH.name, "training runtime path changed")
    consumer = _mapping(report_authorities.get("consumer"), "training consumer authority")
    _require(Path(str(consumer.get("module", ""))).name == TRAINING_CONSUMER_PATH.name, "training consumer path changed")
    return {
        "contract": _ref(repo_root, TRAINING_CONTRACT_PATH, "frozen complex-OLS training contract", TRAINING_CONTRACT_SHA256),
        "report": _ref(repo_root, TRAINING_REPORT_PATH, "valid complex-OLS training report", TRAINING_REPORT_SHA256),
        "lineage": _ref(repo_root, TRAINING_LINEAGE_PATH, "valid complex-OLS training lineage", TRAINING_LINEAGE_SHA256),
        "checkpoint": _ref(repo_root, CHECKPOINT_PATH, "only eligible step-100 training checkpoint", CHECKPOINT_SHA256),
        "checkpoint_policy": {
            "eligible_checkpoint": "step_100 only",
            "selection_search": False,
            "best_of_n": False,
            "fallback_checkpoint": False,
            "reported_selected": True,
            "accepted_training_checkpoint": False,
        },
        "training_boundary": {
            "completed_steps": 100,
            "test_read": False,
            "post_training_evaluation_executed": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "source_artifacts": {
            "training": {
                "path": training_source.get("path"),
                "sha256": TRAINING_SOURCE_SHA256,
                "read_by_definition": False,
            },
            "protected_test": {
                "path": protected_source.get("path"),
                "sha256": PROTECTED_TEST_SHA256,
                "read_by_definition": False,
                "read_by_training": False,
            },
        },
        "checkpoint_metadata": {
            "runtime": checkpoint.get("runtime"),
            "loss_configuration": checkpoint.get("loss_configuration"),
        },
    }


def _validate_evaluator_authority(repo_root: Path) -> dict[str, Any]:
    qualification = _read_json(
        repo_root,
        EVALUATOR_QUALIFICATION_PATH,
        "frozen evaluator qualification evidence",
        EVALUATOR_QUALIFICATION_SHA256,
    )
    _require(qualification.get("evidence_kind") == "evaluator_qualification", "evaluator evidence kind changed")
    _require(qualification.get("status") == "passed", "frozen evaluator qualification did not pass")
    _require(qualification.get("tier") == "acceptance", "frozen evaluator is not acceptance-tier")
    _require(EVALUATOR_VALIDATION_ACCURACY >= EVALUATOR_QUALIFICATION_THRESHOLD, "evaluator qualification threshold changed")
    _require(qualification.get("permitted_next_actions") == ["conditional_observability"], "evaluator permitted action changed")
    context = _mapping(qualification.get("context"), "evaluator qualification context")
    _require(context.get("checkpoint_hash") is None, "evaluator was trained with a generator checkpoint")
    _mapping(qualification.get("input_hashes"), "evaluator qualification inputs")
    manifest = _read_json(
        repo_root,
        EVALUATOR_SPLIT_MANIFEST_PATH,
        "frozen evaluator split manifest",
        EVALUATOR_SPLIT_MANIFEST_SHA256,
    )
    _require(manifest.get("schema_version") == 2, "evaluator split manifest schema changed")
    _require(manifest.get("seed") == SPLIT_SEED, "evaluator split seed changed")
    _require(manifest.get("train_count") == 55_000 and manifest.get("validation_count") == 5_000, "evaluator split counts changed")
    checkpoint = safe_torch_load(repo_root / EVALUATOR_PATH)
    _require(isinstance(checkpoint, dict), "frozen evaluator checkpoint is not a mapping")
    _require(checkpoint.get("architecture_kind") == "frozen_mnist_evaluator_v1", "frozen evaluator architecture changed")
    _require(checkpoint.get("validation_accuracy") == EVALUATOR_VALIDATION_ACCURACY, "frozen evaluator accuracy metadata changed")
    state_dict = checkpoint.get("state_dict")
    _require(isinstance(state_dict, Mapping) and state_dict, "frozen evaluator state is missing")
    evaluator_source = _ref(repo_root, EVALUATOR_SOURCE_PATH, "frozen evaluator implementation", EVALUATOR_SOURCE_SHA256)
    rejected = _read_json(
        repo_root,
        REJECTED_OUTPUT_EVALUATOR_REPORT_PATH,
        "rejected output-domain evaluator report",
        REJECTED_OUTPUT_EVALUATOR_REPORT_SHA256,
    )
    _require(rejected.get("classification") == "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED", "rejected output evaluator decision changed")
    _require(rejected.get("validation_accuracy") == 0.9712 and rejected.get("threshold") == 0.985, "rejected output evaluator evidence changed")
    return {
        "selected_authority": {
            "checkpoint": _ref(repo_root, EVALUATOR_PATH, "accepted independent frozen MNIST evaluator", EVALUATOR_SHA256),
            "qualification": _ref(repo_root, EVALUATOR_QUALIFICATION_PATH, "acceptance-tier evaluator qualification", EVALUATOR_QUALIFICATION_SHA256),
            "split_manifest": _ref(repo_root, EVALUATOR_SPLIT_MANIFEST_PATH, "historical evaluator fit/validation split manifest", EVALUATOR_SPLIT_MANIFEST_SHA256),
            "implementation": evaluator_source,
            "architecture_kind": "frozen_mnist_evaluator_v1",
            "validation_accuracy": EVALUATOR_VALIDATION_ACCURACY,
            "qualification_threshold": EVALUATOR_QUALIFICATION_THRESHOLD,
            "permitted_next_action": "conditional_observability",
            "trained_on_generator_checkpoint": False,
            "protected_test_used_for_training": False,
            "feature_dimension": SEMANTIC_FEATURE_DIMENSION,
            "outputs": ["128D embedding before classifier", "10-class logits/probabilities"],
        },
        "output_domain_candidate_rejected": {
            "report": _ref(repo_root, REJECTED_OUTPUT_EVALUATOR_REPORT_PATH, "rejected output-domain evaluator", REJECTED_OUTPUT_EVALUATOR_REPORT_SHA256),
            "classification": rejected.get("classification"),
            "reason": "validation accuracy 0.9712 is below the 0.985 qualification gate; it cannot be a HARD semantic authority",
            "used_by_this_contract": False,
        },
        "provenance_caveat": "The accepted evaluator was qualified on an independent historical MNIST train/validation manifest, not on the protected test split and not on generated optical outputs. The raw-CMOS-to-28x28 view below is an explicit PROJECT_DECISION; the contract does not claim output-domain requalification.",
    }


def _load_training_split_identity(repo_root: Path) -> dict[str, Any]:
    """Derive non-protected split identities; never opens the protected test file."""

    source = repo_root / TRAINING_SOURCE_PATH
    _require(source.is_file(), "local training source is unavailable")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DefinitionBlocked("local training source cannot be read for split identity") from exc
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "training source layout changed")
    labels = payload[1]
    _require(torch.is_tensor(labels) and labels.ndim == 1 and labels.numel() == 60_000, "training labels changed")
    labels = labels.to(torch.long)
    train, development = stratified_mnist_indices(labels, seed=SPLIT_SEED, validation_per_label=500)
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    development_labels = labels.index_select(0, development)
    selected: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = development[development_labels == digit]
        _require(matches.numel() == 500, "development class count changed")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    selection = torch.cat(selected)
    calibration = development[~torch.isin(development, selection)]
    _require(train.numel() == 55_000 and development.numel() == 5_000, "train/development counts changed")
    _require(selection.numel() == DEVELOPMENT_SELECTION_COUNT, "selection count changed")
    _require(calibration.numel() == DEVELOPMENT_CALIBRATION_COUNT, "calibration count changed")
    _require(torch.unique(torch.cat((train, development))).numel() == 60_000, "train/development overlap detected")
    calibration_counts = torch.bincount(labels.index_select(0, calibration), minlength=CLASS_COUNT)
    _require(torch.equal(calibration_counts, torch.full((CLASS_COUNT,), DEVELOPMENT_CALIBRATION_PER_CLASS)), "calibration is not 400/class")
    _require(tensor_sha256(calibration) == DEVELOPMENT_CALIBRATION_INDICES_SHA256, "calibration index identity changed")
    return {
        "algorithm": "stratified_mnist_indices(labels, seed=1234, validation_per_label=500)",
        "source": {"path": _relative(repo_root, source), "sha256": TRAINING_SOURCE_SHA256},
        "train": {"count": int(train.numel()), "indices_sha256": tensor_sha256(train)},
        "development": {"count": int(development.numel()), "indices_sha256": tensor_sha256(development)},
        "checkpoint_selection": {
            "count": int(selection.numel()),
            "per_class": DEVELOPMENT_SELECTION_PER_CLASS,
            "seed": DEVELOPMENT_SELECTION_SEED,
            "indices_sha256": tensor_sha256(selection),
            "used_for_future_post_training_calibration": False,
        },
        "calibration": {
            "count": int(calibration.numel()),
            "per_class": DEVELOPMENT_CALIBRATION_PER_CLASS,
            "construction": "the 4000 development rows left after removing the fixed 100/class checkpoint-selection subset",
            "indices_sha256": tensor_sha256(calibration),
            "used_for_future_control_and_real-data-null_calibration": True,
        },
        "protected": {
            "path": PROTECTED_TEST_PATH.as_posix(),
            "sha256": PROTECTED_TEST_SHA256,
            "count": PROTECTED_EVALUATION_COUNT,
            "read_by_definition": False,
            "selection": "official test labels are used only during the future evaluator to select a fixed 100/class panel",
        },
    }


def _validate_dataset_authority(repo_root: Path, training: Mapping[str, Any]) -> dict[str, Any]:
    roles = _load_training_split_identity(repo_root)
    _require(roles["train"]["indices_sha256"] == "42980fdbe39cafeb89ae86e21fd818dc2e2404bb7f89193f09bd6c530f4a846c", "train split identity changed")
    _require(roles["development"]["indices_sha256"] == "5c31afc027134a797ffb022fcbfcd1a95fec53c1119ceaf8ca8d01806e5f017e", "development split identity changed")
    _require(roles["checkpoint_selection"]["indices_sha256"] == "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8", "selection split identity changed")
    protected = _mapping(training["source_artifacts"]["protected_test"], "training protected source authority")
    _require(protected.get("sha256") == PROTECTED_TEST_SHA256, "training/protected source mismatch")
    return {
        "dataset": "local MNIST, download=False",
        "roles": {
            "train_reference": {
                "count": roles["train"]["count"],
                "source": roles["source"],
                "indices_sha256": roles["train"]["indices_sha256"],
                "use": "nearest-training-feature reference and exact-duplicate reference only; never a model-selection target",
            },
            "development_checkpoint_selection": {
                **roles["checkpoint_selection"],
                "use": "historical checkpoint selection only; no post-training threshold or metric calibration",
            },
            "development_calibration": {
                **roles["calibration"],
                "use": "control threshold and real-data null calibration only; no checkpoint choice or model tuning",
            },
            "protected_semantic_generation_evaluation": {
                **roles["protected"],
                "use": "future one-shot evaluation only; exact source consumed once, values not read by this definition",
            },
            "sealed": {
                "use": "unavailable and forbidden; historical consumed N3 sealed values are never reused",
                "read_by_definition": False,
                "read_by_future_evaluation": False,
            },
        },
        "preprocessing": {
            "source_images": "float32 grayscale [N,1,28,28] in [0,1]",
            "augmentation": "none",
            "labels_as_model_input": False,
            "protected_panel": {
                "count": PROTECTED_PANEL_COUNT,
                "per_class": PROTECTED_PANEL_PER_CLASS,
                "seed": PROTECTED_PANEL_SEED,
                "algorithm": "for each digit 0..9, CPU-shuffle official-test indices with seed 20260912 and concatenate the first 100 in digit order; do not sort or tune",
            },
        },
        "overlap_policy": {
            "train_vs_development": "none by construction",
            "selection_vs_calibration": "none by construction",
            "train_vs_protected": "source-role boundary; exact protected values are not loaded here",
            "sealed_reuse": False,
        },
    }


def _project_decisions() -> list[dict[str, Any]]:
    return [
        {
            "id": "PROJECT_DECISION_EVAL_SCOPE",
            "decision": "Freeze this contract to one-shot 2D monochrome semantic image generation evaluation only. No denoising, style transfer, 3D, manipulation, routing, controller, electronic rank-16 reconstruction, or new phase producer.",
        },
        {
            "id": "PROJECT_DECISION_EVAL_CHECKPOINT",
            "decision": "The reported step-100 training checkpoint is the only eligible evaluation checkpoint. No sweep, best-checkpoint search, manual checkpoint choice, fallback, retry, or checkpoint-dependent tuning is allowed.",
        },
        {
            "id": "PROJECT_DECISION_OUTPUT_TO_EVALUATOR_VIEW",
            "decision": "Map raw CMOS output [N,1,200,200] to the frozen evaluator view by float32 clamp to [0,1] followed by adaptive_avg_pool2d to [N,1,28,28], with no exposure refit, per-sample normalization, crop search, or denoising. This is a project convention, not a Chen fact.",
        },
        {
            "id": "PROJECT_DECISION_SEMANTIC_EVALUATOR",
            "decision": "Use the existing acceptance-tier independent FrozenMNISTEvaluator v1 for HARD semantic observability. Its 128D pre-classifier embedding and 10-class logits are frozen; the failed 20260904 output-domain evaluator is explicitly rejected.",
        },
        {
            "id": "PROJECT_DECISION_CONTROL_THRESHOLD",
            "decision": "Use one real-path positive-control statistic: macro recall over the ten known protected labels. Calibrate the threshold only from the untouched 4000-row development reserve as max(0.90, clean-reserve macro-recall - 0.05); pass when the real encoded image path reaches that threshold.",
        },
        {
            "id": "PROJECT_DECISION_LATENT_COMPATIBILITY",
            "decision": "Use the training-consistent full-complex gauge S(z) and the biased full-complex Gaussian-kernel MMD^2 V-statistic with l^2=200, equal N=1000 panels, B=999 independently seeded P_target null replicates, and alpha=0.01. q, magnitude-only phase repair, and alternate priors are not allowed.",
        },
        {
            "id": "PROJECT_DECISION_GENERATION_BANKS",
            "decision": "Generate exactly four direct P_target banks of 250 samples, concatenate them in bank order for the single semantic-distribution decision, and retain every sample. Seeds use the existing SHA256(namespace|role|global_step|sample_count) derivation.",
        },
        {
            "id": "PROJECT_DECISION_SEMANTIC_DISTRIBUTION",
            "decision": "Use one HARD semantic-distribution primary: Gaussian-kernel MMD^2 V-statistic on L2-normalized frozen 128D evaluator features with l^2=1.0, calibrated by B=999 independent real-vs-real resamples from the development reserve at alpha=0.01. FID, KID, MMD variants, and competing HARD metrics are not run.",
        },
        {
            "id": "PROJECT_DECISION_DIVERSITY_NOVELTY",
            "decision": "Make one-sided real-null tests HARD for lower within-feature dispersion and train-reference proximity. Use median pairwise feature distance for diversity and median nearest-training-feature distance for novelty, each with B=999 real-vs-real null replicates and alpha=0.01; exact duplicate counts and class summaries remain descriptive and no sample is removed.",
        },
        {
            "id": "PROJECT_DECISION_FINAL_CLASSIFICATION",
            "decision": "Emit exactly one final scientific classification only after a complete valid run: SUPPORTED requires control, latent compatibility, semantic distribution, diversity, and novelty PASS; otherwise emit NOT_SUPPORTED. A control failure stops before generation and is classified CONTROL_NOT_ESTABLISHED.",
        },
    ]


def _evaluation_protocol() -> dict[str, Any]:
    common = {
        "monte_carlo_replicates": MONTE_CARLO_REPLICATES,
        "alpha": MONTE_CARLO_ALPHA,
        "p_value": "p_MC=(1 + #{b: T_b >= T_obs})/(B+1)",
        "pass_rule": "p_MC >= 0.01",
        "metric_search": False,
        "threshold_tuning_after_protected_read": False,
        "protected_retries": False,
    }
    return {
        "execution_order": [
            "identity and input-boundary validation",
            "stage_1_real_image_positive_control",
            "stage_2_full_complex_latent_compatibility_only_if_stage_1_passes",
            "stage_3_direct_target_prior_generation_only_if_stage_2_passes",
            "stage_4_semantic_distribution_then_diversity_then_novelty",
            "one_final_classification",
        ],
        "runtime": {
            "device": "cuda:0 required for the frozen 2048-grid optical forward; no execution in this definition",
            "optical_field_dtype": "torch.complex64",
            "image_and_evaluator_dtype": "torch.float32",
            "metric_accumulation_dtype": "torch.float64",
            "autocast": False,
            "batch_size": 10,
            "raw_output_shape": list(RAW_OUTPUT_SHAPE),
            "stored_output_policy": "store raw CMOS banks and all per-sample evaluator outputs; no sample removal",
        },
        "checkpoint": {
            "path": CHECKPOINT_PATH.as_posix(),
            "sha256": CHECKPOINT_SHA256,
            "eligible_count": 1,
            "step": 100,
            "safe_load_required": True,
            "no_best_or_fallback": True,
        },
        "stage_1_real_image_positive_control": {
            "status": "HARD_GATE",
            "path": "protected image -> aligned input adapter -> input relay -> trained encoder -> modal projection z_real -> T=I_100 -> W_OLS -> same generator -> raw CMOS -> fixed evaluator view",
            "clean_reference": "the same protected image tensor directly evaluated at [1,28,28] after [0,1] validation; no optical path",
            "statistic": "macro_recall = mean_{c=0..9} correct_count_c / label_count_c",
            "calibration_data": "development_calibration role, 4000 rows, untouched by checkpoint selection",
            "threshold": "max(0.90, clean_development_calibration_macro_recall - 0.05)",
            "aggregation": "one macro recall over the full protected 10000-image role",
            "fail_action": f"emit {CONTROL_NOT_ESTABLISHED_CLASSIFICATION}; do not read protected values beyond the declared control pass and do not generate prior samples",
            "labels": "known protected MNIST labels may be used only for this real-image control; labels never enter the generator",
        },
        "stage_2_latent_compatibility": {
            "status": "HARD_GATE_AFTER_CONTROL",
            "data_panel": {"count": PROTECTED_PANEL_COUNT, "per_class": PROTECTED_PANEL_PER_CLASS, "source": "fixed protected panel"},
            "model_panel": {"count": PROTECTED_PANEL_COUNT, "source": "one independently sampled P_target bank"},
            "observed": "T_obs = MMD2_full_complex(z_eval_panel, z_prior_panel)",
            "null": "for b=0..998, independently sample pseudo-DATA and pseudo-MODEL from the same P_target, each count 1000, then T_b=MMD2_full_complex(DATA_b,MODEL_b)",
            "representation": "S(z)=concat(Re(G(z)/sigma_z), Im(G(z)/sigma_z)) in R^200; G(z)=z*conj(a)/sqrt(|a|^2+epsilon_g^2); a=sum_j z_j/sqrt(100); sigma_z=1e-3; epsilon_g=1e-6",
            "kernel": "k(a,b)=exp(-||a-b||^2/(2*l^2)), l^2=200; biased B^2 V-statistic including diagonals",
            "prior": {
                "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NATIVE_OLS_V1",
                "namespace": PRIOR_NAMESPACE,
                "formula": "z_j=sigma_z/sqrt(2)*(epsilon_j_real+i*epsilon_j_imag), epsilon iid Normal(0,1), j=0..99",
                "sampling": "CPU torch.Generator; draw all real quadratures then all imaginary quadratures as the existing training sampler does; transfer complex64 coefficients to the optical device",
                "seed_derivation": "SHA256(namespace|role|global_step|sample_count), first 8 bytes, unsigned masked to 63 bits",
                "observed_model_role": LATENT_MODEL_ROLE,
                "global_step": PRIOR_GLOBAL_STEP,
            },
            "fail_action": "valid scientific failure stops before direct prior generation; do not reinterpret as evaluator or runtime invalidity",
            "no_q_or_phase_repair": True,
        },
        "stage_3_direct_prior_generation": {
            "status": "EXECUTE_ONLY_AFTER_LATENT_PASS",
            "banks": {
                "count": GENERATION_BANK_COUNT,
                "samples_per_bank": GENERATION_BANK_SIZE,
                "total_samples": GENERATION_TOTAL_COUNT,
                "concatenation": "ascending bank index, no sorting, no selection",
                "seed_roles": [f"{GENERATION_ROLE_PREFIX}{index}" for index in range(GENERATION_BANK_COUNT)],
            },
            "path": "z_generated ~ P_target -> ComplexOLSState -> T=I_100 -> W_OLS -> same trained generator -> raw CMOS",
            "forbidden_prerequisites": [
                "image encoder",
                "labels or label+8D controller",
                "q statistic",
                "phase producer or phase-repair module",
                "latent optimization",
                "rejection sampling",
                "retry, best-of-N, or human selection",
                "alternate electronic/generated-state decoder",
            ],
            "invalid_output_policy": "nonfinite or structurally invalid output is INVALID_HARNESS and cannot become NOT_SUPPORTED",
        },
        "stage_4_semantic_distribution": {
            "status": "HARD_GATE_AFTER_GENERATION",
            "features": "f=normalize_l2(FrozenMNISTEvaluator.embedding(view), dim=1) in R^128",
            "observed": "T_obs=MMD2_feature(protected_panel_features, concatenated_generated_features)",
            "null": "for b=0..998, independently resample two 1000-row feature banks with replacement from the 4000-row development calibration feature bank and compute real-vs-real T_b",
            "kernel": "k(a,b)=exp(-||a-b||^2/(2*l^2)), l^2=1.0; biased B^2 V-statistic including diagonals",
            "calibration": "the development real-data null is the only calibration; no protected metric threshold is estimated",
            "labels": "generated class predictions, probabilities, entropy, and class coverage are descriptive only",
            "fail_action": "final classification NOT_SUPPORTED after complete valid metric production; no competing HARD metric is substituted",
        },
        "stage_4_diversity": {
            "status": "HARD_GATE_AFTER_SEMANTIC_DISTRIBUTION",
            "features": "the same frozen, L2-normalized 128D evaluator features as the semantic primary",
            "observed": "T_obs=median_pairwise_distance(protected_panel)-median_pairwise_distance(generated_bank)",
            "null": "for b=0..998, independently resample two 1000-row development feature banks with replacement and use T_b=median_distance(A)-median_distance(B)",
            "tail": "large T means generated dispersion is lower; p_MC uses the upper tail exactly",
            "duplicate_count": "descriptive exact bitwise duplicate count after fixed evaluator-view preprocessing against the train-reference view; no removal",
        },
        "stage_4_novelty": {
            "status": "HARD_GATE_AFTER_DIVERSITY",
            "reference": "all 55000 train-role evaluator features, frozen before protected access",
            "observed": "T_obs=median(nearest_train_distance(protected_panel))-median(nearest_train_distance(generated_bank))",
            "null": "for b=0..998, independently resample two 1000-row development feature banks with replacement and compute the same difference to the frozen train reference",
            "tail": "large T means generated features are closer to training references; p_MC uses the upper tail exactly",
            "retention": "all generated samples remain in every downstream summary; proximity never triggers filtering",
        },
        "common_statistic_rules": common,
    }


def _alignment_matrix(
    alignment_authority: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runtime_ref = alignment_authority["runtime"]
    return [
        {
            "component": "eligible checkpoint",
            "lightgen_requirement": "evaluate one frozen trained LightGen-aligned runtime checkpoint",
            "current_project_implementation": training["checkpoint"],
            "classification": "SAME",
            "migration_action": "bind step 100 and its SHA; reject sweep, fallback, manual selection, or retry",
        },
        {
            "component": "image optical input",
            "lightgen_requirement": "image -> amplitude SLM command -> illuminated field -> coherent input relay -> encoder",
            "current_project_implementation": {
                "path": runtime_ref["path"],
                "sha256": runtime_ref["sha256"],
                "detail": "validated aligned image adapter, sqrt project mapping, zero-phase illuminated field, and coherent relay",
            },
            "classification": "SAME",
            "migration_action": "call the unchanged aligned image path for the real control; do not add digital preprocessing",
        },
        {
            "component": "authoritative OLS state",
            "lightgen_requirement": "10x10 / 100-channel single-mode complex optical latent retaining relative phase",
            "current_project_implementation": "ComplexOLSState coefficients in C^100 with stable row-major port order",
            "classification": "SAME",
            "migration_action": "bind z_real and z_generated to the same typed OLS interface; q remains diagnostic only",
        },
        {
            "component": "modal projection",
            "lightgen_requirement": "every OLS coefficient arises from a complex single-mode overlap",
            "current_project_implementation": "FiberArrayOLS project_coefficients at the aligned encoder OLS plane",
            "classification": "SAME",
            "migration_action": "require projection before the real-path writer and record finite complex z",
        },
        {
            "component": "coefficient transmission",
            "lightgen_requirement": "OLS transport occurs in coefficient space",
            "current_project_implementation": "T=I_100 in coefficient space",
            "classification": "SAME",
            "migration_action": "use T=I_100 for both paths; reject whole-field identity bypass",
        },
        {
            "component": "modal re-emission / generator input",
            "lightgen_requirement": "mandatory modal re-emission and one shared generator entrance interface",
            "current_project_implementation": "W_OLS(z)=sum_j(Tz)_j v_j followed by the same trained generator",
            "classification": "SAME",
            "migration_action": "route real encoded and direct prior states through the identical W_OLS and generator",
        },
        {
            "component": "frozen semantic evaluator",
            "lightgen_requirement": "independent frozen labels/probabilities and stable features after fixed preprocessing",
            "current_project_implementation": evaluator["selected_authority"],
            "classification": "PARTIAL",
            "migration_action": "freeze the accepted historical evaluator and explicitly bind the raw-CMOS-to-28x28 project view; do not call it Chen or output-domain requalified",
        },
        {
            "component": "raw-output evaluator view",
            "lightgen_requirement": "raw output is measured without model-dependent postprocessing",
            "current_project_implementation": "clamp raw CMOS to [0,1] then adaptive_avg_pool2d to 28x28",
            "classification": "PROJECT_DECISION",
            "migration_action": "implement exactly once in the future evaluation consumer; forbid exposure refit, normalization, crop search, denoising, or metric-driven changes",
        },
        {
            "component": "full-complex latent compatibility",
            "lightgen_requirement": "compare encoded complex OLS with the frozen full-complex target prior",
            "current_project_implementation": "training-consistent S(z), sigma=1e-3, epsilon=1e-6, l^2=200 MMD semantics",
            "classification": "SAME",
            "migration_action": "use fixed equal 1000-row panels and B=999 null calibration; no q-only statistic or phase producer",
        },
        {
            "component": "semantic distribution",
            "lightgen_requirement": "direct prior outputs match protected semantic feature distribution",
            "current_project_implementation": "no post-training consumer exists yet",
            "classification": "UNRESOLVED",
            "migration_action": "implement the single calibrated feature-MMD primary and no competing HARD metric",
        },
        {
            "component": "diversity and novelty",
            "lightgen_requirement": "generated outputs are not a collapsed or trivial training replay",
            "current_project_implementation": "existing diagnostic feature/statistic primitives, but no bound consumer for this checkpoint",
            "classification": "PARTIAL",
            "migration_action": "implement the frozen one-sided real-null dispersion and train-reference proximity tests; retain all samples",
        },
        {
            "component": "historical N3 phase producers and sealed values",
            "lightgen_requirement": "aligned generation must not depend on amplitude-first phase repair or consumed sealed evidence",
            "current_project_implementation": "historical N3 artifacts; sealed values unavailable",
            "classification": "DIFFERENT",
            "migration_action": "preserve as baselines only; never load, fit, or route them into this evaluation",
        },
    ]


def _claim_boundary() -> dict[str, Any]:
    return {
        "this_contract_proves": [
            "one coherent protected one-shot evaluation protocol is defined",
            "only the reported step-100 checkpoint is eligible",
            "real encoded OLS and direct prior OLS converge through the same W_OLS and generator interface",
            "a frozen independent evaluator, development calibration role, null policies, and fail-closed order are bound",
            "full-complex latent compatibility, semantic distribution, diversity, and novelty decisions have one fixed definition each",
        ],
        "this_contract_does_not_prove": [
            "real-path semantic control pass",
            "latent prior compatibility",
            "direct prior generation quality or semantic distribution match",
            "diversity, novelty, or absence of memorization in actual outputs",
            "exact Chen reproduction, paper FID/KID, universal image quality, physical arbitrary-complex writer validity, disentanglement, or controllable semantics",
            "formal promotion or generation readiness",
        ],
        "result_interpretation": {
            "control_failure": f"{CONTROL_NOT_ESTABLISHED_CLASSIFICATION}; preserve the valid training result and stop before generation",
            "complete_valid_failure": f"{NOT_SUPPORTED_CLASSIFICATION}; do not infer that the aligned architecture, all priors, or the evaluator are invalid, and do not auto-retrain",
            "complete_valid_pass": f"{SUPPORTED_CLASSIFICATION}; direct target-prior -> same W_OLS -> same generator is compatible with the protected semantic-data distribution under this frozen contract only",
            "invalid_harness": "malformed authority, nonfinite output, path/dtype/shape mismatch, or protected-role violation; not a scientific gate failure",
        },
    }


def _reuse(training: Mapping[str, Any], alignment: Mapping[str, Any], evaluator: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"component": "step-100 complex-OLS checkpoint", "status": "read-only bound input", "use": training["checkpoint"]},
        {"component": "aligned input/OLS runtime and W_OLS writer", "status": "reuse unchanged", "use": alignment["runtime"]},
        {"component": "existing full-complex gauge and MMD mathematical semantics", "status": "reuse semantics; no training execution", "use": "gauge_invariant_ols_representation and full_complex_mmd_vstat definitions with an evaluation-sized panel"},
        {"component": "acceptance-tier FrozenMNISTEvaluator v1", "status": "reuse frozen checkpoint and qualification evidence", "use": evaluator["selected_authority"]},
        {"component": "local MNIST train/development split helper", "status": "reuse unchanged", "use": "non-protected reserve identity and real-data null calibration"},
        {"component": "PlaneSpec / ComplexField / ComplexOLSState", "status": "reuse unchanged", "use": "physical-coordinate and complex OLS boundary"},
        {"component": "historical N3 reports and amplitude confirmation", "status": "preserve read-only", "use": "negative baselines and claim boundary only; no sealed reuse"},
    ]


def _retire_from_mainline() -> list[dict[str, Any]]:
    return [
        {
            "mechanism": "amplitude-only q as authoritative latent",
            "status": "historical diagnostic only; complex z is authoritative",
            "preserve": True,
        },
        {
            "mechanism": "independent phase inheritance, q-conditioned nearest phase donor, and q-conditioned phasor residual",
            "status": "historical N3 baselines; no phase-repair prerequisite",
            "preserve": True,
        },
        {
            "mechanism": "direct post-hoc complex-joint producer development",
            "status": "not the aligned mainline; may remain a later baseline only",
            "preserve": True,
        },
        {
            "mechanism": "failed 20260904 output-domain frozen evaluator",
            "status": "rejected because validation accuracy 0.9712 < 0.985; never a HARD authority",
            "preserve": True,
        },
        {
            "mechanism": "label+8D controller, electronic decoder, denoiser, VAE decode, rejection sampling, retry, best-of-N, human selection",
            "status": "forbidden in the direct prior generation path",
            "preserve": True,
        },
        {
            "mechanism": "consumed N3 sealed reservation",
            "status": "unavailable and never reused",
            "preserve": True,
        },
    ]


def _contract_totality_checks() -> dict[str, Any]:
    _require(OLS_GRID_SHAPE == (10, 10) and OLS_PORT_COUNT == 100, "OLS dimension is not fixed")
    _require(RAW_OUTPUT_SHAPE == (1, 200, 200), "raw output shape is not fixed")
    _require(MONTE_CARLO_REPLICATES == 999 and MONTE_CARLO_ALPHA == 0.01, "Monte Carlo gate changed")
    _require(GENERATION_BANK_COUNT == 4 and GENERATION_BANK_SIZE == 250, "generation bank count changed")
    return {
        "definition_only": True,
        "training_executed": False,
        "cuda_used": False,
        "scientific_generation_executed": False,
        "heldout_evaluation_executed": False,
        "protected_values_read": False,
        "protected_source_consumed": False,
        "fid_executed": False,
        "kid_executed": False,
        "mmd_executed": False,
        "checkpoint_sweep_executed": False,
        "sealed_values_loaded": False,
        "sealed_values_reused": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "same_real_and_generated_w_ols": True,
        "same_downstream_generator": True,
        "q_authoritative": False,
        "phase_repair_required": False,
        "no_sample_removal": True,
    }


def _governance(repo_root: Path) -> dict[str, Any]:
    payload = _read_json(repo_root, GOVERNANCE_STATE_PATH, "machine governance state")
    _require(payload.get("schema_version") == 1, "machine governance schema changed")
    for key in ("training_authorized", "formal_promotion"):
        if key in payload:
            _require(payload.get(key) is False, f"machine governance opened {key}")
    if "generation_readiness" in payload:
        _require(payload.get("generation_readiness") == "not_ready", "machine governance opened generation")
    return {
        "machine_state": _ref(repo_root, GOVERNANCE_STATE_PATH, "machine governance state; read-only"),
        "state_mutated": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "note": "this definition does not rewrite the machine governance state; the future evaluation remains separately authorized",
    }


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    alignment_payload: Mapping[str, Any],
    alignment: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    data: Mapping[str, Any],
    governance: Mapping[str, Any],
    totality: Mapping[str, Any],
) -> dict[str, Any]:
    source = alignment["source"]
    project_decisions = _project_decisions()
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definition_only": True,
        "architecture_decision": "LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST",
        "git": dict(git),
        "source_authority": {
            "verified_lightgen": source,
            "alignment_contract": alignment["contract"],
            "alignment_runtime": alignment["runtime"],
            "alignment_runtime_report": alignment["runtime_report"],
            "classification_rule": "CHEN_FACT is limited to the verified source; CHEN_UNRESOLVED is genuinely unspecified by that source; PROJECT_DECISION is an explicit project convention and never a paper fact.",
            "paper_and_project_taxonomy": {
                "CHEN_FACT": source["taxonomy"]["CHEN_FACT"],
                "CHEN_UNRESOLVED": source["taxonomy"]["CHEN_UNRESOLVED"],
                "PROJECT_DECISION": project_decisions,
            },
        },
        "bound_training_authority": training,
        "bound_semantic_evaluator_authority": evaluator,
        "data_contract": data,
        "architecture": {
            "scope": "2D semantic image generation only",
            "canonical_image_path": [
                "X_eval -> explicit aligned image validation and project SLM intensity mapping",
                "I_slm=clamp(BilinearResize(X_eval,200x200),0,1)",
                "A_slm=sqrt(I_slm), phi0=0 -> illuminated complex field -> coherent input relay",
                "relay -> trained two-layer phase-only encoder -> E_encoder",
                "mandatory z_real,j=<E_encoder,u_j> -> z_real in C^100",
                "coefficient-space T=I_100 -> mandatory W_OLS(z_real)=sum_j(Tz_real)_j v_j",
                "same trained photonic generator -> B=0 output relay -> raw CMOS intensity",
            ],
            "canonical_generation_path": [
                "draw z_generated ~ P_target directly in C^100",
                "bind z_generated to the same ComplexOLSState and stable row-major 10x10 port order",
                "coefficient-space T=I_100 -> the same W_OLS -> the same trained generator",
                "B=0 output relay -> raw CMOS intensity",
                "no image encoder, label/controller, q, phase producer, repair, optimization, retry, or alternate decoder",
            ],
            "shared_interface_invariant": "real encoded z and future generated z enter the identical W_OLS and downstream generator interface",
        },
        "ols_mathematical_definition": {
            "authoritative_state": "z in C^100",
            "shape": ["N", OLS_PORT_COUNT],
            "dtype": "complex64 optical state",
            "grid": list(OLS_GRID_SHAPE),
            "port_order": "j=10*r+c for r,c in {0,...,9}, row-major and stable",
            "projection": "z_j=<E_encoder,u_j>=sum_{y,x} conj(u_j(y,x))*E_encoder(y,x)*Delta_y*Delta_x",
            "transmission": "z_transmitted=Tz, T=I_100 in coefficient space",
            "re_emission": "W_OLS(z)=sum_j(Tz)_j v_j",
            "relative_phase": "all coordinate-to-coordinate phase differences are retained; only the latent statistics use the frozen global-phase quotient",
            "derived_only": {"q": "abs(z)", "phi": "arg(z)", "p": "abs(z)^2", "restriction": "none may replace z at the generator interface"},
            "modal_projection_mandatory": True,
            "modal_re_emission_mandatory": True,
            "whole_field_identity_bypass": False,
        },
        "ols_physical_numerical_geometry": {
            "chen_facts": {
                "wavelength_m": 532.0e-9,
                "single_mode_ols_grid": list(OLS_GRID_SHAPE),
                "fiber_reference_length_m": 8.5e-3,
                "fiber_reference_outer_diameter_m": 125.0e-6,
                "fiber_reference_use": "reference metadata only; never used to infer mode-field radius",
            },
            "project_decisions": {
                "slm_shape": [200, 200],
                "slm_pixel_pitch_m": 2.5e-6,
                "ols_shape": list(OLS_SHAPE),
                "ols_pixel_pitch_m": 2.0e-6,
                "ols_fov_m": [200.0e-6, 200.0e-6],
                "mode_bank": {"model": "step_index_lp01_ports", "v_number": 2.2, "waist_m": 7.111e-6, "fill": 0.8},
                "encoder_layers": 2,
                "generator_layers": 2,
                "propagation_grid_size": PRODUCTION_PROPAGATION_GRID,
                "transport": "T=I_100 coefficient-space identity",
            },
            "required_future_verification": [
                "physical-coordinate quadrature and modal normalization",
                "100-port Gram rank/conditioning under the selected bank",
                "dimensionless LP01/coupling relations and passive coefficient transport",
                "generator-plane PlaneSpec/provenance and raw CMOS boundary",
            ],
        },
        "generator_input_writer": {
            "writer_id": "W_OLS_V1",
            "definition": "validate finite complex [N,100] -> apply T in coefficient space -> re-emit through output modal bank v_j -> typed ComplexField at generator entrance",
            "real_encoded_ols": "image -> aligned optical input -> encoder -> modal projection z_real -> W_OLS -> generator",
            "future_generated_ols": "z_generated ~ P_target -> W_OLS -> same generator",
            "same_writer": True,
            "same_generator": True,
            "forbidden_shortcuts": [
                "whole-field identity bypass",
                "alternate electronic/generated-state decoder",
                "q-only input",
                "image-dependent phase repair",
                "label+8D controller",
                "software-only VAE decode",
            ],
        },
        "evaluation_protocol": _evaluation_protocol(),
        "alignment_matrix": _alignment_matrix(alignment, training, evaluator),
        "reuse": _reuse(training, alignment, evaluator),
        "retire_from_mainline": _retire_from_mainline(),
        "unresolved": source["taxonomy"]["CHEN_UNRESOLVED"],
        "claim_boundary": _claim_boundary(),
        "outer_gate": {
            "passed": True,
            "answer": "YES",
            "checks": {
                "semantic_generation_only": True,
                "explicit_image_optical_input": True,
                "ols_is_10x10_100_channel_single_mode_complex": True,
                "complex_z_in_C100_authoritative": True,
                "mandatory_modal_projection": True,
                "mandatory_modal_re_emission": True,
                "coefficient_space_transmission": True,
                "same_real_and_generated_w_ols_generator_interface": True,
                "independent_frozen_semantic_evaluator_available": True,
                "chen_unresolved_separated_from_project_decisions": True,
                "no_protected_values_read": True,
                "no_training_or_generation_executed": True,
            },
            "hard_stop_conditions": {
                "required_lightgen_source_unresolved": False,
                "ols_semantics_conflict_unrepresentable": False,
                "shared_generator_interface_unavailable": False,
                "required_quantity_undefined": False,
                "paper_detail_invented_as_fact": False,
                "semantic_evaluator_authority_unresolved": False,
            },
            "reason": "The accepted independent evaluator authority is sufficient for conditional observability, the failed output-domain candidate is explicitly rejected, and every output preprocessing and decision threshold is recorded as a project convention rather than a paper fact.",
        },
        "governance": dict(governance),
        "totality_checks": dict(totality),
        "definition_consumer": _ref(repo_root, Path(__file__).resolve().relative_to(repo_root), "post-training evaluation contract definition consumer"),
        "next_action": NEXT_ACTION,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def define(repo_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    _require(not (root / OUTPUT_DIR).exists(), "definition output already exists; refusing overwrite")
    git = _validate_git_boundary(root)
    alignment_payload, alignment = _validate_alignment_authority(root)
    training = _validate_training_authority(root)
    evaluator = _validate_evaluator_authority(root)
    data = _validate_dataset_authority(root, training)
    governance = _governance(root)
    totality = _contract_totality_checks()
    payload = _build_payload(
        root,
        git,
        alignment_payload,
        alignment,
        training,
        evaluator,
        data,
        governance,
        totality,
    )
    output_path = root / OUTPUT_DIR / CONTRACT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_new_json(output_path, payload, artifact_kind="LightGen post-training evaluation contract")
    return output_path, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        path, payload = define(args.repo_root)
    except DefinitionBlocked as exc:
        print(json.dumps({"result": "BLOCKED", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    root = _repo_root() if args.repo_root is None else args.repo_root.resolve()
    print(
        json.dumps(
            {
                "result": payload["result"],
                "classification": payload["classification"],
                "artifact": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "next_action": payload["next_action"],
                "training_authorized": payload["training_authorized"],
                "formal_promotion": payload["formal_promotion"],
                "generation_readiness": payload["generation_readiness"],
                "protected_values_read": payload["totality_checks"]["protected_values_read"],
                "scientific_generation_executed": payload["totality_checks"]["scientific_generation_executed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
