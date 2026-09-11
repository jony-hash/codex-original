"""Define the LightGen-aligned complex-OLS semantic training contract.

This is a definition-only consumer.  It binds the already verified aligned
runtime, freezes one complete full-complex target prior and one differentiable
latent objective, and records a bounded future training schedule.  It does not
create an optimizer, execute a forward pass, load q/complex/sealed values, run
training, or authorize generation/promotion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_contract.v1"
READY_CLASSIFICATION = "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"
ARCHITECTURE_DECISION = "LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST"
NEXT_ACTION = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_semantic_complex_ols_training_contract.v1.json"

OLS_GRID_SHAPE = (10, 10)
OLS_PORT_COUNT = 100
SLM_SHAPE = (200, 200)
OLS_SHAPE = (100, 100)
IMAGE_SHAPE = (1, 28, 28)
WAVELENGTH_M = 532.0e-9
SLM_PIXEL_PITCH_M = 2.5e-6
OLS_PIXEL_PITCH_M = 2.0e-6
OLS_FOV_M = (200.0e-6, 200.0e-6)
OLS_V_NUMBER = 2.2
OLS_WAIST_M = 7.111e-6
OLS_FILL = 0.8
ENCODER_LAYERS = 2
GENERATOR_LAYERS = 2
PRODUCTION_PROPAGATION_GRID = 2048

CLASS_COUNT = 10
TRAIN_COUNT = 55_000
DEVELOPMENT_COUNT = 5_000
PROTECTED_TEST_COUNT = 10_000
LOGICAL_BATCH_SIZE = 10
TRAINING_STEPS = 100
CHECKPOINT_STEPS = (50, 100)
SPLIT_SEED = 1234
SCHEDULE_SEED = 1234
DEVELOPMENT_SUBSET_SEED = 4321
DEVELOPMENT_SUBSET_PER_CLASS = 100

PRIOR_SIGMA = 1.0e-3
GAUGE_EPSILON = 1.0e-6
GAUGE_WEIGHT = 1.0 / math.sqrt(OLS_PORT_COUNT)
LATENT_EMBEDDING_DIMENSION = 2 * OLS_PORT_COUNT
MMD_BANDWIDTH_SQUARED = float(LATENT_EMBEDDING_DIMENSION)
LATENT_ALPHA = 1.0
IMAGE_BETA = 1.0
OPTICAL_GAMMA = 0.0
PRIOR_TRAIN_SEED = 20260910
PRIOR_DEVELOPMENT_SEED = 20260911
PRIOR_RNG_NAMESPACE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_PRIOR"

ALIGNMENT_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json"
)
ALIGNMENT_CONTRACT_SHA256 = "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"

RUNTIME_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"
)
RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"

LATEST_N3_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/"
    "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1.json"
)
LATEST_N3_REPORT_SHA256 = "ae60752a27d3fb655b9354ace5591cb7f092db2aebb6206774853fa82cd025fe"
LATEST_N3_LINEAGE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/"
    "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_lineage.v1.json"
)
LATEST_N3_LINEAGE_SHA256 = "503c5afa463403d237768e5eee1c5e130939394486e1c8348716ef553041a9de"

SOURCE_MANIFEST_PATH = Path("research_assets/lightgen/science_adv7434/manifest.json")
SOURCE_MANIFEST_SHA256 = "fdb4f8c7b476ce92bd85b42ecbd435507657c91adfd501326842d5d00ee3418c"
PAPER_PDF_PATH = Path("research_assets/lightgen/science_adv7434/paper/sciencesm.pdf")
PAPER_PDF_SHA256 = "1325cdfb438f55249224f660a90ec2c21438ba3ce8cffc2677001f2db4e6c211"

PROJECT_COMPLEX_REPRESENTATION_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "ols_representation_contract.json"
)
PROJECT_COMPLEX_REPRESENTATION_SHA256 = "f99c2ced713005fe3f4a5a0fa91a2cd7648f5647e2360f122bc68e77f225f10d"
PROJECT_COMPLEX_PHASE_GAUGE_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "phase_gauge_audit.json"
)
PROJECT_COMPLEX_PHASE_GAUGE_SHA256 = "5ee1a81ebdc054937f70b2e206d0da95aeae45d8a14f6f3812f265000563abb2"

GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")

DATA_ARTIFACTS = {
    "data/MNIST/processed/training.pt": {
        "sha256": "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd",
        "role": "local MNIST train split source; read-only",
    },
    "data/MNIST/processed/test.pt": {
        "sha256": "8c089c2f0a5db35de7961f4c60ed6b627a05141245d44f542eeee57b09b539e1",
        "role": "official MNIST test split; protected heldout source; not consumed here",
    },
}

RUNTIME_SOURCE_HASHES = {
    "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py": (
        "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce",
        "aligned LightGen input/OLS/generator runtime",
    ),
    "src/ldm_project/v2/optics/fiber_operator.py": (
        "6e941d6610c282aeb9f54cb39ea200c868f7b0eb14c3bc41b692ee505f4cf295",
        "modal projection, coefficient transport, and modal re-emission primitive",
    ),
    "src/ldm_project/v2/optics/d2nn.py": (
        "25bc45ea0bb6ea1ecf003010c9a62e88a75903738abc9b69e51062b89ed37cf0",
        "phase-only finite-crop encoder/generator primitive",
    ),
    "src/ldm_project/v2/optics/devices.py": (
        "bbe1dc78a977bc8c143ba9ebbf663e78a4a206e1ac18759481f815c7dc815a0d",
        "amplitude SLM primitive",
    ),
    "src/ldm_project/v2/optics/relays.py": (
        "8f777f206f16b51a1832047c696a5cde6bb77d48c0c18582eb0fc4fb971bc54d",
        "coherent input/output relay primitive",
    ),
    "src/ldm_project/v2/contracts/__init__.py": (
        "e926360bd26df68deb3dc9e99e4e1131013e5078e2e41d74abe7e4f3e988e82a",
        "typed field and plane exports",
    ),
    "src/ldm_project/v2/contracts/planes.py": (
        "6692d42315ce9af30fb7f2dfb1160d749ff1254f1d8ed5235304fb39fd374179",
        "PlaneSpec, ComplexField, IntensityFrame, and provenance types",
    ),
}

DATA_SPLIT_SOURCE_PATH = Path("src/ldm_project/v2/student/evaluator.py")
DATA_SPLIT_SOURCE_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"


class DefinitionBlocked(RuntimeError):
    """The contract cannot be frozen without weakening an authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionBlocked(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _read_json(
    repo_root: Path,
    path: Path,
    name: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {path.as_posix()}")
    if expected_sha256 is not None:
        _require(file_sha256(target) == expected_sha256, f"{name} SHA256 changed")
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
    _require(target.is_file(), f"authority is missing: {path.as_posix()}")
    digest = file_sha256(target)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"authority SHA256 changed: {path.as_posix()}")
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
        "worktree_dirty": bool(status),
        "status_line_count": len(status),
    }


def _validate_git_boundary(repo_root: Path) -> dict[str, Any]:
    state = _git_state(repo_root)
    _require(state["branch"] == EXPECTED_BRANCH, f"expected branch {EXPECTED_BRANCH}")
    _require(state["head"] == EXPECTED_STARTING_HEAD, f"expected HEAD {EXPECTED_STARTING_HEAD}")
    return state


def _validate_source_identity(repo_root: Path) -> dict[str, Any]:
    manifest = _read_json(
        repo_root,
        SOURCE_MANIFEST_PATH,
        "LightGen source identity manifest",
        SOURCE_MANIFEST_SHA256,
    )
    _require(manifest.get("status") == "reference_only", "LightGen source status changed")
    source = _mapping(manifest.get("source"), "LightGen source identity")
    _require(
        source.get("doi") == "10.1126/science.adv7434"
        and source.get("title") == "All-optical synthesis chip for large-scale intelligent semantic vision generation"
        and source.get("supplement_pages") == 46,
        "LightGen source identity is not the verified Chen source",
    )
    paper_ref = _ref(repo_root, PAPER_PDF_PATH, "verified LightGen Science supplement PDF", PAPER_PDF_SHA256)
    files = manifest.get("files")
    _require(isinstance(files, list), "LightGen source file list is invalid")
    pdf_entry = next(
        (item for item in files if isinstance(item, Mapping) and item.get("path") == "paper/sciencesm.pdf"),
        None,
    )
    _require(isinstance(pdf_entry, Mapping) and pdf_entry.get("sha256") == PAPER_PDF_SHA256, "source manifest/PDF binding changed")
    return {
        "manifest": _ref(repo_root, SOURCE_MANIFEST_PATH, "verified LightGen source identity manifest", SOURCE_MANIFEST_SHA256),
        "paper_pdf": paper_ref,
        "identity": {
            "doi": source["doi"],
            "title": source["title"],
            "supplement_pages": source["supplement_pages"],
            "status": manifest["status"],
        },
    }


def _validate_alignment_authority(repo_root: Path) -> dict[str, Any]:
    payload = _read_json(
        repo_root,
        ALIGNMENT_CONTRACT_PATH,
        "aligned input/OLS contract",
        ALIGNMENT_CONTRACT_SHA256,
    )
    _require(
        payload.get("schema") == "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1"
        and payload.get("status") == "valid_completed"
        and payload.get("result") == "READY"
        and payload.get("classification") == "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY",
        "aligned input/OLS contract is not the verified READY authority",
    )
    outer = _mapping(payload.get("outer_gate"), "aligned input/OLS outer gate")
    _require(outer.get("passed") is True, "aligned input/OLS outer gate is not passed")
    totality = _mapping(payload.get("totality_checks"), "aligned input/OLS totality")
    for key in (
        "projection_required",
        "reemission_required",
        "coefficient_transmission_required",
        "same_writer_for_real_and_generated_ols",
    ):
        _require(totality.get(key) is True, f"aligned runtime requirement is missing: {key}")
    _require(totality.get("q_is_authoritative_latent") is False, "aligned contract reopened q as latent")
    _require(totality.get("whole_field_identity_bypass") is False, "aligned contract permits whole-field identity bypass")
    return {
        "authority": _ref(repo_root, ALIGNMENT_CONTRACT_PATH, "verified aligned input/OLS contract", ALIGNMENT_CONTRACT_SHA256),
        "classification": payload["classification"],
        "architecture_decision": payload.get("architecture_decision"),
        "outer_gate": dict(outer),
        "totality": dict(totality),
    }


def _validate_runtime_authority(repo_root: Path) -> dict[str, Any]:
    payload = _read_json(
        repo_root,
        RUNTIME_REPORT_PATH,
        "aligned runtime report",
        RUNTIME_REPORT_SHA256,
    )
    _require(
        payload.get("schema") == "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1"
        and payload.get("status") == "valid_completed"
        and payload.get("result") == "IMPLEMENTED"
        and payload.get("classification") == "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1_IMPLEMENTED",
        "aligned runtime report is not the verified implementation authority",
    )
    contract = _mapping(payload.get("contract"), "aligned runtime contract reference")
    _require(
        contract.get("path") == ALIGNMENT_CONTRACT_PATH.as_posix()
        and contract.get("sha256") == ALIGNMENT_CONTRACT_SHA256,
        "runtime report is not bound to the verified alignment contract",
    )
    runtime = _mapping(payload.get("runtime"), "aligned runtime metadata")
    _require(
        runtime.get("module") == "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py"
        and runtime.get("module_sha256") == RUNTIME_SOURCE_HASHES[runtime.get("module")][0]
        and runtime.get("fiber_operator_sha256") == RUNTIME_SOURCE_HASHES["src/ldm_project/v2/optics/fiber_operator.py"][0]
        and runtime.get("production_propagation_grid") == PRODUCTION_PROPAGATION_GRID
        and runtime.get("canonical_ols") == "complex [B,100], 10x10, row-major j=10*r+c"
        and runtime.get("transport") == "T=I_100 in coefficient space"
        and runtime.get("writer") == "W_OLS(z)=sum_j (Tz)_j v_j",
        "aligned runtime metadata changed",
    )
    structural = _mapping(payload.get("structural_verification"), "aligned runtime structural verification")
    for key in (
        "stable_row_major_order",
        "modal_projection_active",
        "t_identity",
        "w_ols_output_plane_matches_generator_input",
        "same_writer_path_equivalence",
        "same_generator_raw_equivalence",
        "direct_path_skips_image_encoder",
        "q_only_rejected",
    ):
        _require(structural.get(key) is True, f"aligned runtime invariant failed: {key}")
    claim = _mapping(payload.get("claim_boundary"), "aligned runtime claim boundary")
    _require(
        claim.get("training_executed") is False
        and claim.get("scientific_generation_executed") is False
        and claim.get("heldout_evaluation") is False
        and claim.get("cuda_used") is False
        and claim.get("sealed_values_loaded") is False
        and claim.get("formal_promotion") is False
        and claim.get("generation_readiness") == "not_ready",
        "aligned runtime claim boundary opened",
    )
    source_refs = {
        path: _ref(repo_root, Path(path), role, digest)
        for path, (digest, role) in RUNTIME_SOURCE_HASHES.items()
    }
    return {
        "report": _ref(repo_root, RUNTIME_REPORT_PATH, "verified aligned runtime report", RUNTIME_REPORT_SHA256),
        "runtime": dict(runtime),
        "structural_verification": dict(structural),
        "claim_boundary": dict(claim),
        "source_implementations": source_refs,
    }


def _validate_project_complex_ols_authority(repo_root: Path) -> dict[str, Any]:
    representation = _read_json(
        repo_root,
        PROJECT_COMPLEX_REPRESENTATION_PATH,
        "PROJECT_COMPLEX_OLS representation contract",
        PROJECT_COMPLEX_REPRESENTATION_SHA256,
    )
    _require(
        representation.get("schema") == "lightgen_scaled_chen_ols_representation_contract.v1"
        and representation.get("representations", {}).get("R3_COMPLEX") == "c/sqrt(s), gauge-aware complex geometry"
        and representation.get("complex_primary_geometry") == "global_phase_gauge_distance"
        and representation.get("raw_cartesian_primary_geometry") is False,
        "PROJECT_COMPLEX_OLS representation authority changed",
    )
    gauge = _read_json(
        repo_root,
        PROJECT_COMPLEX_PHASE_GAUGE_PATH,
        "PROJECT_COMPLEX_OLS phase-gauge audit",
        PROJECT_COMPLEX_PHASE_GAUGE_SHA256,
    )
    _require(
        gauge.get("schema") == "lightgen_scaled_chen_ols_phase_gauge_audit.v1"
        and gauge.get("classification") == "GLOBAL_PHASE_GAUGE_CONFIRMED"
        and gauge.get("formal_phase_reference") is False
        and gauge.get("raw_cartesian_primary") is False,
        "PROJECT_COMPLEX_OLS global-phase authority changed",
    )
    return {
        "semantic_authority_id": "PROJECT_COMPLEX_OLS_V1",
        "historical_coordinate": "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX = c/sqrt(s)",
        "representation_contract": _ref(repo_root, PROJECT_COMPLEX_REPRESENTATION_PATH, "historical complex representation authority", PROJECT_COMPLEX_REPRESENTATION_SHA256),
        "phase_gauge_audit": _ref(repo_root, PROJECT_COMPLEX_PHASE_GAUGE_PATH, "historical global-phase gauge authority", PROJECT_COMPLEX_PHASE_GAUGE_SHA256),
        "reused_boundary": "common global phase is an unobservable quotient for intensity/power and relative phase remains meaningful",
        "native_runtime_scaling": "not reused: historical s_j calibration is not part of the aligned runtime W_OLS interface",
    }


def _validate_latest_scientific_boundary(repo_root: Path) -> dict[str, Any]:
    report = _read_json(repo_root, LATEST_N3_REPORT_PATH, "latest N3 result", LATEST_N3_REPORT_SHA256)
    _require(
        report.get("schema") == "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1"
        and report.get("task_id") == "EXECUTE_N3_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1"
        and report.get("status") == "valid_completed"
        and report.get("result") == "VALID_COMPLETED"
        and report.get("classification") == "N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED",
        "latest N3 result is not the verified q-conditioned complex-phasor failure",
    )
    boundaries = _mapping(report.get("boundaries"), "latest N3 boundaries")
    _require(
        boundaries.get("n3_amplitude_producer_confirmation") == "PASSED"
        and boundaries.get("training_authorized") is False
        and boundaries.get("formal_promotion") is False
        and boundaries.get("generation_readiness") == "not_ready",
        "latest N3 changed the authorization boundary",
    )
    candidate = _mapping(report.get("candidate"), "latest N3 candidate")
    candidate_aggregate = _mapping(candidate.get("aggregate"), "latest N3 candidate aggregate")
    _require(
        candidate_aggregate.get("fold_count") == 5
        and candidate_aggregate.get("passed_fold_count") == 0
        and candidate_aggregate.get("fold_pass_pattern") == [False] * 5
        and candidate_aggregate.get("primary_metric") == "energy_distance",
        "latest N3 candidate is not 0/5",
    )
    folds = candidate.get("folds")
    _require(isinstance(folds, list) and len(folds) == 5, "latest N3 fold detail is incomplete")
    for index, fold in enumerate(folds):
        fold_mapping = _mapping(fold, f"latest N3 fold {index}")
        primary = _mapping(fold_mapping.get("primary"), f"latest N3 fold {index} primary")
        _require(primary.get("p_MC") == 0.001, f"latest N3 fold {index} p_MC changed")
        fit = _mapping(fold_mapping.get("phase_model_fit"), f"latest N3 fold {index} fit")
        _require(fit.get("numerical_validity") is True, f"latest N3 fold {index} fit is not numerically valid")
        projection = _mapping(fold_mapping.get("projection_diagnostics"), f"latest N3 fold {index} projection")
        _require(
            projection.get("projection_invalid_coordinate_count") == 0
            and projection.get("projection_invalid_sample_count") == 0,
            f"latest N3 fold {index} has projection invalids",
        )
    control = _mapping(report.get("control"), "latest N3 full-complex control")
    control_aggregate = _mapping(control.get("aggregate"), "latest N3 control aggregate")
    _require(
        control_aggregate.get("fold_count") == 5
        and control_aggregate.get("passed_fold_count") == 5
        and control_aggregate.get("fold_pass_pattern") == [True] * 5,
        "latest N3 full-complex control is not 5/5",
    )
    sealed = _mapping(report.get("sealed_confirmation"), "latest N3 sealed boundary")
    _require(
        sealed.get("consumed") is True
        and sealed.get("values_loaded") is False
        and sealed.get("reused") is False,
        "latest N3 sealed boundary changed",
    )
    governance = _mapping(report.get("governance"), "latest N3 governance")
    _require(
        governance.get("state_mutated") is False
        and governance.get("training_authorized") is False
        and governance.get("formal_promotion") is False
        and governance.get("generation_readiness") == "not_ready",
        "latest N3 governance boundary changed",
    )
    lineage = _read_json(repo_root, LATEST_N3_LINEAGE_PATH, "latest N3 lineage", LATEST_N3_LINEAGE_SHA256)
    _require(
        lineage.get("schema") == "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_lineage.v1"
        and lineage.get("classification") == report["classification"]
        and lineage.get("result") == "VALID_COMPLETED",
        "latest N3 lineage is not the matching completed authority",
    )
    return {
        "report": _ref(repo_root, LATEST_N3_REPORT_PATH, "latest N3 scientific result metadata", LATEST_N3_REPORT_SHA256),
        "lineage": _ref(repo_root, LATEST_N3_LINEAGE_PATH, "latest N3 result lineage", LATEST_N3_LINEAGE_SHA256),
        "classification": report["classification"],
        "candidate": "0/5",
        "candidate_p_mc": [0.001] * 5,
        "control": "5/5",
        "numerical_fit_valid": True,
        "projection_invalids": 0,
        "amplitude_producer_confirmation": "PASSED",
        "sealed": {"consumed": True, "values_loaded": False, "reused": False},
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def _validate_governance(repo_root: Path) -> dict[str, Any]:
    payload = _read_json(repo_root, GOVERNANCE_STATE_PATH, "machine governance state")
    _require(
        payload.get("schema_version") == 1
        and payload.get("policy_id") == "RESEARCH_GOVERNANCE_SIMPLIFICATION_V1"
        and payload.get("authorization_tier") == "R3",
        "machine governance schema or tier changed",
    )
    for key in ("training_authorized", "formal_promotion", "generation_readiness"):
        if key in payload:
            expected = "not_ready" if key == "generation_readiness" else False
            _require(payload.get(key) == expected, f"machine governance opened {key}")
    return {
        "machine_state": _ref(repo_root, GOVERNANCE_STATE_PATH, "machine governance state; read-only"),
        "state_mutated": False,
        "authorization_tier": payload["authorization_tier"],
        "active_objective": payload.get("active_objective"),
        "hard_blocker": payload.get("hard_blocker"),
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "note": "the current machine schema has no top-level authorization flags; result/report flags remain fail-closed and this definition does not rewrite the state",
    }


def _validate_dataset_authority(repo_root: Path) -> dict[str, Any]:
    artifacts = {
        path: _ref(repo_root, Path(path), details["role"], details["sha256"])
        for path, details in DATA_ARTIFACTS.items()
    }
    split_ref = _ref(
        repo_root,
        DATA_SPLIT_SOURCE_PATH,
        "canonical local MNIST loader and stratified split implementation",
        DATA_SPLIT_SOURCE_SHA256,
    )
    return {
        "dataset": "local MNIST, download=False",
        "root": "data/MNIST",
        "loader": "ldm_project.v2.student.evaluator.load_local_mnist",
        "source_artifacts": artifacts,
        "split_implementation": split_ref,
        "split": {
            "algorithm": "stratified_mnist_indices(labels, seed=1234, validation_per_label=500)",
            "source_train_count": 60_000,
            "train_count": TRAIN_COUNT,
            "development_count": DEVELOPMENT_COUNT,
            "classes": CLASS_COUNT,
            "train_per_class": 5_500,
            "development_per_class": 500,
            "split_seed": SPLIT_SEED,
            "overlap": "none by construction; validation is selected before train role is consumed",
        },
        "roles": {
            "train": {
                "count": TRAIN_COUNT,
                "use": "image-to-aligned-runtime optimization only",
                "labels": "sampler balancing only; never a model input",
            },
            "development": {
                "count": DEVELOPMENT_COUNT,
                "selection_subset": DEVELOPMENT_SUBSET_PER_CLASS * CLASS_COUNT,
                "selection_subset_rule": "100 per class from the fixed 500-per-class development split, CPU seed 4321",
                "use": "predeclared checkpoint comparison only; no optimizer or hyperparameter search",
            },
            "protected_heldout": {
                "count": PROTECTED_TEST_COUNT,
                "source": "official MNIST test split",
                "use": "not consumed by this definition or the future training run; separate post-training evaluation only",
            },
            "calibration": {
                "count": 0,
                "use": "none; no empirical q scale, phase calibration, or historical N3 artifact is imported",
            },
            "sealed": {
                "use": "not applicable and not read; consumed N3 sealed evidence remains unavailable",
            },
        },
        "preprocessing": {
            "loader": "uint8 image data converted to float32 and divided by 255",
            "runtime_input": "finite real [0,1] [B,1,28,28]",
            "runtime_resize": "the immutable aligned runtime performs bilinear resize to 200x200 and clamp [0,1]",
            "augmentation": "none",
            "label_or_controller_input": False,
        },
    }


def _inspect_trainable_parameter_surface() -> dict[str, Any]:
    """Inspect module parameters without a forward pass, optimizer, or data."""

    import torch

    from ldm_project.v2.application.research.lightgen_2d_semantic_generation_input_ols_alignment import (
        LightGen2DAlignmentRuntimeConfig,
        LightGen2DSemanticGenerationInputOLSAlignment,
    )

    model = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=128)
    )
    named = tuple(model.named_parameters())
    names = tuple(name for name, _ in named)
    expected = ("encoder.phase", "generator.phase")
    _require(names == expected, f"aligned runtime parameter surface changed: {names!r}")
    shapes = {name: list(parameter.shape) for name, parameter in named}
    _require(shapes == {name: [2, 1, 100, 100] for name in expected}, "phase parameter shapes changed")
    _require(all(parameter.dtype == torch.float32 for _, parameter in named), "phase parameter dtype changed")
    _require(all(parameter.requires_grad for _, parameter in named), "runtime phase parameters are not available to the future trainer")
    return {
        "probe": "CPU module construction with propagation_grid_size=128; no forward, optimizer, backward, data, or CUDA",
        "all_parameter_names": list(names),
        "trainable_parameter_names": list(expected),
        "optimizer_parameter_names": list(expected),
        "parameter_shapes": shapes,
        "parameter_dtype": "torch.float32",
        "parameter_count": 40_000,
        "frozen_scope": [
            "image adapter and amplitude SLM mapping",
            "input coherent relay and output B=0 relay",
            "OLS input/output modal banks, coefficient transport T=I_100, and provenance",
            "encoder/generator topology, propagation grid, layer count, spacing, and active shape",
            "raw CMOS intensity construction",
            "all q/phase producers, controllers, electronic decoders, and prior state",
        ],
    }


def _contract_totality_checks() -> dict[str, Any]:
    _require(OLS_GRID_SHAPE == (10, 10) and OLS_PORT_COUNT == 100, "OLS dimension is not fixed")
    _require(SLM_SHAPE == (200, 200) and OLS_SHAPE == (100, 100), "project plane shapes changed")
    _require(WAVELENGTH_M == 532.0e-9 and OLS_FOV_M == (200.0e-6, 200.0e-6), "project optical sampling changed")
    _require(ENCODER_LAYERS == 2 and GENERATOR_LAYERS == 2, "project layer counts changed")
    _require(PRIOR_SIGMA > 0.0 and GAUGE_EPSILON > 0.0 and MMD_BANDWIDTH_SQUARED == 200.0, "latent constants are invalid")
    _require((IMAGE_BETA, LATENT_ALPHA, OPTICAL_GAMMA) == (1.0, 1.0, 0.0), "loss weights are not frozen")
    return {
        "status": "passed",
        "definition_only": True,
        "training_executed": False,
        "cuda_used": False,
        "optimizer_created": False,
        "backward_executed": False,
        "scientific_generation_executed": False,
        "heldout_evaluation_executed": False,
        "sealed_values_loaded": False,
        "governance_mutated": False,
        "new_phase_producer_defined": False,
        "q_authoritative": False,
        "whole_field_identity_bypass": False,
        "same_real_and_generated_w_ols": True,
    }


def _chen_facts(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    authority = source["paper_pdf"]
    return [
        {
            "id": "CHEN_FACT_INPUT_OPTICS",
            "category": "CHEN_FACT",
            "statement": "The LightGen setup uses a single-mode 532 nm laser, an amplitude-modulation SLM, and a 4f relay that conjugates the image to the LightGen input plane.",
            "source": "Science supplement page 2, Experimental set-up of LightGen chip.",
            "authority": authority,
        },
        {
            "id": "CHEN_FACT_COMPLEX_MODAL_OLS",
            "category": "CHEN_FACT",
            "statement": "The supplement describes complex optical fields through a 10x10 OLS and coupling to a fundamental single-mode field by a complex overlap integral.",
            "source": "Science supplement pages 6-9 and Figure S2 caption.",
            "authority": authority,
        },
        {
            "id": "CHEN_FACT_OLS_DIMENSION",
            "category": "CHEN_FACT",
            "statement": "The supplement reports that OLS performance improves up to dimension 100, corresponding to a 10x10 OLS.",
            "source": "Science supplement page 2 and Figure S11.",
            "authority": authority,
        },
        {
            "id": "CHEN_FACT_GENERATOR_ONLY_RECONFIGURATION",
            "category": "CHEN_FACT",
            "statement": "After OLS adjustment, the described generation path uses only the photonic generator.",
            "source": "Science supplement pages 2-3, Chip design of LightGen.",
            "authority": authority,
        },
        {
            "id": "CHEN_FACT_BOGT_LOSS_SHAPE",
            "category": "CHEN_FACT",
            "statement": "BOGT is described as a distribution constraint between Q(Z|X) and P(Z), combined with reconstruction and optical-power terms in a weighted loss; the source description is not a complete estimator contract.",
            "source": "Science supplement page 2, Loss function of BOGT.",
            "authority": authority,
        },
    ]


def _chen_unresolved(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    authority = source["paper_pdf"]
    return [
        {
            "id": "CHEN_UNRESOLVED_IMAGE_TO_SLM_MAPPING",
            "category": "CHEN_UNRESOLVED",
            "statement": "The available source does not establish the exact digital normalization, interpolation, SLM command transfer curve, gamma, or illumination phase convention.",
            "authority": authority,
        },
        {
            "id": "CHEN_UNRESOLVED_FULL_COMPLEX_PRIOR",
            "category": "CHEN_UNRESOLVED",
            "statement": "The source does not uniquely establish a directly sampleable full-complex 100-channel prior in the aligned coefficient coordinate; the positive-intensity truncated biased Gaussian description is not a full-complex prior specification.",
            "authority": authority,
        },
        {
            "id": "CHEN_UNRESOLVED_BOGT_ESTIMATOR",
            "category": "CHEN_UNRESOLVED",
            "statement": "The exact BOGT statistic axes, normalization, alpha/beta/gamma values, batch estimator, numerical stabilizers, and sampling implementation are not established.",
            "authority": authority,
        },
        {
            "id": "CHEN_UNRESOLVED_OLS_NORMALIZATION_AND_PORT_ORDER",
            "category": "CHEN_UNRESOLVED",
            "statement": "The machine-readable 10x10 port order, discrete modal-basis normalization, and output-plane writer are not specified by the available authority.",
            "authority": authority,
        },
        {
            "id": "CHEN_UNRESOLVED_PHYSICAL_GEOMETRY",
            "category": "CHEN_UNRESOLVED",
            "statement": "Core diameter, NA, mode-field radius, port pitch, crosstalk, insertion loss, measured complex transmission, and complete relay/layer/generator numerical details remain unresolved; outer fiber diameter is not a mode-field radius.",
            "authority": authority,
        },
    ]


def _project_decisions() -> list[dict[str, Any]]:
    return [
        {
            "id": "PROJECT_DECISION_TRAINING_SCOPE",
            "decision": "Freeze this contract to unconditional 2D monochrome semantic image generation. Labels can balance the sampler but cannot enter the model; no denoising, style transfer, 3D, manipulation, routing, controller, rank-16 readout, or new phase family.",
        },
        {
            "id": "PROJECT_DECISION_NATIVE_COMPLEX_COORDINATE",
            "decision": "Use the aligned runtime's raw modal coefficients z in C^100 as the generator interface. Do not import historical PROJECT_COMPLEX_OLS s_j calibration into W_OLS; the old c/sqrt(s) coordinate is retained only for compatible gauge geometry and historical evidence.",
        },
        {
            "id": "PROJECT_DECISION_FULL_COMPLEX_PRIOR",
            "decision": "Freeze P_target as the raw complex isotropic prior z in C^100, with z_j = sigma/sqrt(2)*(epsilon_j^R+i epsilon_j^I), sigma=1e-3, and independent standard-normal real and imaginary draws for j=0..99. This is directly sampleable and is not a Chen fact or a q-only prior.",
        },
        {
            "id": "PROJECT_DECISION_DIFFERENTIABLE_GAUGE",
            "decision": "For latent statistics only, use the smooth global-phase-invariant representative G(z)=z*conj(a(z))/sqrt(|a(z)|^2+eps_g^2), a(z)=sum_j z_j/sqrt(100), eps_g=1e-6. It preserves relative phase, maps z=0 deterministically to zero, and is not a phase producer or a runtime input transformation.",
        },
        {
            "id": "PROJECT_DECISION_LATENT_MMD",
            "decision": "Because the exact BOGT estimator is unresolved, use one fixed full-complex Gaussian-kernel MMD^2 V-statistic on S(z) in R^200, with bandwidth squared 200, biased B^2 reductions including diagonals, and no metric search.",
        },
        {
            "id": "PROJECT_DECISION_IMAGE_LOSS",
            "decision": "Use raw CMOS intensity MSE against the aligned runtime's clamped 200x200 bilinear image target, with mean reduction and no prediction clamp, per-sample normalization, exposure refit, perceptual term, GAN, classifier, or denoiser.",
        },
        {
            "id": "PROJECT_DECISION_NO_OPTICAL_TERM",
            "decision": "Omit L_optical in V1 because no source-authorized physical target and normalization can be frozen without inventing a paper detail; gamma=0 is explicit, not an unrecorded default.",
        },
        {
            "id": "PROJECT_DECISION_SINGLE_BOUNDED_RUN",
            "decision": "Freeze one 100-step Adam run at lr=1e-3, logical batch 10, zero phase initialization, float32/no AMP, CUDA required for the future run, and no arms, sweep, retries, adaptive weights, or best-of-N.",
        },
    ]


def _alignment_matrix(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    def impl(detail: str) -> dict[str, Any]:
        return {
            "path": "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py",
            "detail": detail,
            "authority": runtime["source_implementations"]["src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py"],
        }

    return [
        {
            "component": "image optical input",
            "lightgen_requirement": "amplitude SLM plus coherent 4f relay from image input to encoder entrance",
            "current_project_implementation": impl("real [B,1,H,W] -> bilinear/clamp -> sqrt amplitude -> zero-phase IdealAmplitudeSLM -> CoherentImagingRelay"),
            "classification": "SAME",
            "migration_action": "freeze the existing mapping as PROJECT_DECISION; do not call it a Chen fact",
        },
        {
            "component": "photonic encoder",
            "lightgen_requirement": "project-specific diffractive encoder is permitted",
            "current_project_implementation": impl("two-layer FiniteCropPhaseStack, phase-only, active 100x100 plane"),
            "classification": "SAME",
            "migration_action": "optimize encoder.phase only",
        },
        {
            "component": "authoritative OLS latent",
            "lightgen_requirement": "10x10 / 100-channel complex single-mode OLS state with relative phase",
            "current_project_implementation": impl("ComplexOLSState.coefficients with shape [B,100] and complex dtype"),
            "classification": "SAME",
            "migration_action": "keep z as the only generator-facing latent; q/phi/p remain derived diagnostics",
        },
        {
            "component": "modal projection",
            "lightgen_requirement": "each coefficient is a complex single-mode overlap",
            "current_project_implementation": {
                "path": "src/ldm_project/v2/optics/fiber_operator.py",
                "detail": "FiberArrayOLS.project_coefficients uses complex overlap and physical-coordinate quadrature",
                "authority": runtime["source_implementations"]["src/ldm_project/v2/optics/fiber_operator.py"],
            },
            "classification": "SAME",
            "migration_action": "freeze projection boundary; never replace it with amplitude pooling or whole-field identity",
        },
        {
            "component": "modal transmission",
            "lightgen_requirement": "coefficient-space OLS transport",
            "current_project_implementation": impl("T=I_100 in coefficient space"),
            "classification": "SAME",
            "migration_action": "freeze T and reject any runtime transport change for this training contract",
        },
        {
            "component": "modal re-emission and generator input",
            "lightgen_requirement": "mandatory W_OLS re-emission; real and generated OLS share the interface",
            "current_project_implementation": impl("w_ols -> coefficient transmission -> output modal bank -> same generator entrance plane"),
            "classification": "SAME",
            "migration_action": "all training and future generation must call the same w_ols and generator objects",
        },
        {
            "component": "trainable scope",
            "lightgen_requirement": "only project photonic encoder/generator parameters are trainable",
            "current_project_implementation": impl("only encoder.phase and generator.phase exist in the verified parameter surface"),
            "classification": "SAME",
            "migration_action": "optimizer manifest must equal exactly those two names; fail closed on any added parameter",
        },
        {
            "component": "historical q/phase producers",
            "lightgen_requirement": "no q->phase reconstruction in the aligned mainline",
            "current_project_implementation": "N3 q-conditioned phase candidates are historical evidence only",
            "classification": "DIFFERENT",
            "migration_action": "do not import, fit, or select them in the training runtime",
        },
        {
            "component": "formal v2 identity OLS/controller",
            "lightgen_requirement": "real/generated OLS must use the aligned complex modal interface",
            "current_project_implementation": "formal label+8D and identity-OLS chain remains a separate compatibility baseline",
            "classification": "DIFFERENT",
            "migration_action": "preserve formally; do not use it as this research runtime or latent target",
        },
    ]


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    source: Mapping[str, Any],
    alignment: Mapping[str, Any],
    runtime: Mapping[str, Any],
    complex_ols: Mapping[str, Any],
    latest_result: Mapping[str, Any],
    governance: Mapping[str, Any],
    dataset: Mapping[str, Any],
    trainable: Mapping[str, Any],
    totality: Mapping[str, Any],
) -> dict[str, Any]:
    source_taxonomy = _chen_facts(source) + _chen_unresolved(source) + _project_decisions()
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
        "architecture_decision": ARCHITECTURE_DECISION,
        "git": dict(git),
        "source_authority": {
            "verified_lightgen": source,
            "alignment_contract": dict(alignment),
            "runtime_report": dict(runtime),
            "project_complex_ols": dict(complex_ols),
            "classification_rule": "CHEN_FACT is limited to the verified source; CHEN_UNRESOLVED is genuinely unspecified by that source; PROJECT_DECISION is an explicit project convention and never a paper fact.",
            "paper_and_project_taxonomy": source_taxonomy,
        },
        "current_scientific_boundary": dict(latest_result),
        "architecture": {
            "scope": "2D semantic image generation only",
            "immutable_runtime": [
                "image input validation and explicit SLM mapping",
                "input coherent relay and encoder entrance plane",
                "two-layer phase-only encoder topology and propagation geometry",
                "10x10/100-port single-mode OLS plane, row-major port order, modal banks, projection, T=I_100, and re-emission",
                "generator entrance plane, two-layer phase-only generator topology, output B=0 relay, and raw CMOS intensity",
            ],
            "allowed_trainable_change": "values of encoder.phase and generator.phase only; no topology or boundary change",
            "canonical_image_path": [
                "X real finite [B,1,28,28] in [0,1]",
                "X -> aligned SLM intensity target via immutable bilinear 200x200 resize and clamp",
                "SLM intensity -> sqrt amplitude command -> illuminated complex field with phi0=0",
                "coherent input relay -> encoder entrance field",
                "two-layer phase-only photonic encoder -> final encoder field",
                "mandatory modal projection z_real_j=<E_encoder,u_j> -> z_real in C^100",
                "shared W_OLS(z_real) -> same photonic generator -> B=0 output relay -> raw CMOS intensity",
            ],
            "canonical_generation_path": [
                "draw raw z_sample ~ P_target in C^100",
                "bind z_sample to ComplexOLSState / aligned OLS interface",
                "shared W_OLS(z_sample) -> same photonic generator -> B=0 output relay -> raw CMOS intensity",
                "no image encoder, label+8D controller, amplitude producer, phase producer, or phase repair prerequisite",
            ],
        },
        "ols_mathematical_definition": {
            "authoritative_state": "z in C^100",
            "port_order": "j=10*r+c for r,c in {0,...,9}, row-major",
            "projection": "z_j = <E_encoder,u_j> = sum_{y,x} conj(u_j(y,x))*E_encoder(y,x)*Delta_y*Delta_x",
            "transmission": "z_transmitted = T z, T=I_100 in coefficient space",
            "re_emission": "W_OLS(z) = sum_j (Tz)_j v_j",
            "derived_only": {
                "q": "abs(z)",
                "phi": "arg(z)",
                "p": "abs(z)^2",
                "restriction": "none may replace z in the generator interface or training target",
            },
            "relative_phase": "all coordinate-to-coordinate phase differences in z are retained; common phase is handled only by the loss representation gauge",
            "interface": "ComplexOLSState / lightgen_10x10_complex_ols_v1",
        },
        "ols_physical_numerical_geometry": {
            "chen_facts": {
                "wavelength_m": WAVELENGTH_M,
                "single_mode_ols_grid": list(OLS_GRID_SHAPE),
                "fiber_reference_length_m": 8.5e-3,
                "fiber_reference_outer_diameter_m": 125.0e-6,
                "fiber_reference_use": "metadata only; never used to infer mode-field radius",
            },
            "project_decisions": {
                "slm": {"shape": list(SLM_SHAPE), "pixel_pitch_m": SLM_PIXEL_PITCH_M},
                "ols_plane": {"shape": list(OLS_SHAPE), "pixel_pitch_m": OLS_PIXEL_PITCH_M, "fov_m": list(OLS_FOV_M)},
                "mode_bank": {"model": "step_index_lp01_ports", "v_number": OLS_V_NUMBER, "waist_m": OLS_WAIST_M, "fill": OLS_FILL},
                "encoder_layers": ENCODER_LAYERS,
                "generator_layers": GENERATOR_LAYERS,
                "propagation_grid_size": PRODUCTION_PROPAGATION_GRID,
                "transport": "T=I_100 coefficient-space identity",
                "port_order": "row-major j=10*r+c",
            },
            "required_future_verification": [
                "physical-coordinate quadrature and modal normalization",
                "100-port Gram rank/conditioning and declared passivity",
                "dimensionless LP01/coupling relations",
                "input/output relay provenance and generator-plane normalization",
            ],
        },
        "generator_input_writer": {
            "writer_id": "W_OLS_V1",
            "definition": "validate complex [B,100] -> apply T in coefficient space -> re-emit through v_j -> typed ComplexField at generator entrance",
            "real_encoded_ols": "image -> aligned input -> encoder -> z_real -> W_OLS -> generator",
            "future_generated_ols": "z_generated -> W_OLS -> same generator",
            "convergence_invariant": "same W_OLS, T, output modal basis, generator entrance PlaneSpec, generator parameters, output relay, and raw CMOS readout",
            "forbidden_shortcuts": [
                "whole-field identity bypass",
                "alternate electronic/generated-state decoder",
                "q-only generator input",
                "image-dependent phase repair",
                "label+8D controller",
                "software-only VAE decode",
            ],
        },
        "trainable_parameter_contract": {
            "exact_trainable_parameters": list(trainable["trainable_parameter_names"]),
            "exact_optimizer_parameters": list(trainable["optimizer_parameter_names"]),
            "parameter_shapes": dict(trainable["parameter_shapes"]),
            "parameter_count": trainable["parameter_count"],
            "initialization": {
                "seed": SPLIT_SEED,
                "encoder.phase": "all zeros",
                "generator.phase": "all zeros",
                "warm_start": False,
            },
            "frozen_parameters_and_state": list(trainable["frozen_scope"]),
            "forbidden_trainables": [
                "OLS modal banks or T",
                "input/output relays and SLM mapping",
                "q producer",
                "phase producer or phase repair",
                "electronic controller, label embedding, or rank-16 readout",
                "prior parameters or any adaptive normalization",
            ],
            "parameter_surface_probe": trainable["probe"],
        },
        "latent_contract": {
            "authoritative_state": "raw aligned ComplexOLS coefficients z in C^100",
            "representation": "S(z)=concat(Re(G(z)/sigma), Im(G(z)/sigma)) in R^200",
            "scale": {"sigma_z": PRIOR_SIGMA, "role": "fixed loss/prior unit; does not rescale W_OLS input"},
            "global_phase": {
                "reused_authority": complex_ols["reused_boundary"],
                "smooth_gauge_formula": "G(z)=z*conj(a(z))/sqrt(|a(z)|^2+eps_g^2)",
                "anchor": "a(z)=sum_j z_j/sqrt(100)",
                "epsilon": GAUGE_EPSILON,
                "zero_behavior": "G(0)=0; no arg(0), retry, or arbitrary phase fill",
                "near_zero_behavior": "the regularized factor is deterministic and differentiable; it smoothly attenuates rather than reconstructing phase",
                "future_diagnostics": "apply the same G and S definitions; never compare raw Cartesian phase against a new arbitrary reference",
                "phase_repair": False,
            },
            "full_complex_requirement": "real and imaginary quadratures for all 100 channels are present in the authoritative state and in S(z); q is not a substitute",
        },
        "target_prior": {
            "authority": "PROJECT_DECISION",
            "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NATIVE_OLS_V1",
            "directly_sampleable": True,
            "formula": "z_j = sigma_z/sqrt(2)*(epsilon_j_real + i*epsilon_j_imag), epsilon iid Normal(0,1), j=0..99",
            "sigma_z": PRIOR_SIGMA,
            "shape": [OLS_PORT_COUNT],
            "dtype": "complex64-compatible complex state",
            "global_phase": "raw prior retains a sampled common phase; the same smooth G is applied only by latent statistics and future diagnostics",
            "training_target_sample_count": LOGICAL_BATCH_SIZE,
            "development_target_sample_count": DEVELOPMENT_SUBSET_PER_CLASS * CLASS_COUNT,
            "rng": {
                "namespace": PRIOR_RNG_NAMESPACE,
                "training_seed": PRIOR_TRAIN_SEED,
                "development_seed": PRIOR_DEVELOPMENT_SEED,
                "derivation": "SHA256(namespace|role|global_step|sample_count), first 8 bytes reduced to signed 63-bit seed",
                "search_or_refit": False,
            },
            "future_generator_only_use": "the exact same P_target is sampled and passed directly to forward_from_ols; no fitted q marginal, empirical copula, phase donor, residual, or post-hoc repair",
            "not_chen_fact": True,
            "historical_q_prior_reused": False,
        },
        "objectives": {
            "latent": {
                "name": "FULL_COMPLEX_GAUSSIAN_KERNEL_MMD_VSTAT_V1",
                "status": "PROJECT_DECISION surrogate because exact BOGT estimator is CHEN_UNRESOLVED",
                "embedding": "x_i=S(z_real_i), y_i=S(z_target_i), x,y in R^200",
                "kernel": "k(a,b)=exp(-||a-b||^2/(2*l^2)), l^2=200",
                "formula": "L_latent = mean_ij k(x_i,x_j) + mean_ij k(y_i,y_j) - 2*mean_ij k(x_i,y_j)",
                "reduction": "biased B^2 V-statistic including diagonal terms; batch axis is logical batch rows",
                "target_count": LOGICAL_BATCH_SIZE,
                "differentiable": True,
                "stabilizers": "finite complex-state validation; eps_g regularized gauge; no clamp, metric search, or adaptive bandwidth",
                "q_only": False,
            },
            "image": {
                "name": "RAW_CMOS_INTENSITY_MSE_V1",
                "formula": "L_image = mean((I_cmos_raw - I_target)^2)",
                "target": "I_target=clamp(BilinearResize(X,200x200),0,1), exactly the aligned input adapter's SLM-intensity target",
                "reduction": "mean over batch, channel, and 200x200 pixels",
                "prediction_normalization": "none; raw CMOS intensity is consumed as produced by the immutable runtime",
                "differentiable": True,
            },
            "optical": {
                "included": False,
                "formula": None,
                "reason": "No source-authorized physical power target/normalization can be frozen without inventing a Chen detail; crop retention remains diagnostic only.",
            },
            "total": {
                "formula": "L_total = beta*L_image + alpha*L_latent + gamma*L_optical",
                "weights": {"beta_image": IMAGE_BETA, "alpha_latent": LATENT_ALPHA, "gamma_optical": OPTICAL_GAMMA},
                "reduced_formula": "L_total = 1.0*L_image + 1.0*L_latent",
                "weight_search": False,
                "best_of_n": False,
                "adaptive_weighting": False,
            },
        },
        "data_contract": dataset,
        "training_schedule": {
            "one_run_only": True,
            "steps": TRAINING_STEPS,
            "epochs": None,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "batch_order": "one sample per digit, labels 0..9; labels are sampler metadata only",
            "sampling": "from the fixed 55,000 train-role rows, one independently permuted row per class per step, without replacement across the 100 steps",
            "schedule_seed": SCHEDULE_SEED,
            "optimizer": {
                "name": "Adam",
                "learning_rate": 1.0e-3,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "precision": {"phase": "float32", "field": "complex64", "autocast": False, "grad_scaler": False},
            "device": "cuda required for the future training execution; no CUDA used by this definition",
            "gradient": {
                "zero_grad": "set_to_none=True",
                "clipping": None,
                "nonfinite_policy": "fail closed before update and write no accepted checkpoint",
            },
            "initialization": "zero phase; no historical checkpoint and no warm start",
            "checkpointing": {
                "steps": list(CHECKPOINT_STEPS),
                "save_optimizer_state": True,
                "selection": "among valid checkpoints at steps 50 and 100, choose lowest fixed development L_total; ties choose earlier step; no heldout input",
                "development_subset": DEVELOPMENT_SUBSET_PER_CLASS * CLASS_COUNT,
                "development_target_seed": PRIOR_DEVELOPMENT_SEED,
            },
            "stop_rule": "run exactly 100 updates; stop immediately and fail closed on nonfinite loss/gradient/parameter or any structural invariant violation; no early-stop, retry, sweep, or automatic continuation",
            "preflight_required_later": "root long-training gate: static/lineage checks, representative three-step real update through the same entrypoint, and domain-appropriate negative control before this run",
        },
        "structural_invariants_per_step": [
            "real image -> explicit aligned optical input -> input relay -> phase-only encoder",
            "encoder output -> mandatory modal projection -> finite complex z in C^100",
            "z -> coefficient-space T=I_100 -> mandatory W_OLS modal re-emission",
            "W_OLS output -> the same phase-only generator -> B=0 output relay -> raw CMOS intensity",
            "no alternate decoder, q-only path, phase repair, controller, or whole-field identity bypass",
            "only encoder.phase and generator.phase receive gradients and optimizer updates",
        ],
        "future_generation": {
            "conceptual_only": True,
            "path": "z_sample ~ P_target -> ComplexOLSState -> forward_from_ols(z_sample) -> W_OLS -> same generator -> raw CMOS",
            "image_encoder_required": False,
            "label_controller_required": False,
            "amplitude_producer_required": False,
            "phase_producer_required": False,
            "execution_in_this_task": False,
        },
        "claim_boundary": {
            "this_contract_proves": [
                "one complete future training interface is mathematically defined",
                "the target is full-complex and directly sampleable",
                "the latent loss is differentiable and not q-only",
                "the future optimizer parameter set and bounded schedule are frozen",
                "real encoded and future generated OLS states converge at the same W_OLS/generator interface",
            ],
            "this_contract_does_not_prove": [
                "training convergence or image reconstruction",
                "semantic validity, diversity, novelty, or prior accessibility",
                "generation acceptance or generation readiness",
                "physical Chen reproduction or hardware validity",
                "formal v2 promotion",
            ],
            "post_training_evaluation_separate": [
                "real encoded OLS reconstruction",
                "prior-sampled OLS raw CMOS outputs",
                "same-W_OLS/generator structural equivalence",
                "semantic validity, diversity, novelty, and distribution comparison",
            ],
        },
        "alignment_matrix": _alignment_matrix(runtime),
        "reuse": [
            "verified LightGen source identity and page-level taxonomy, read-only",
            "LightGen2DSemanticGenerationInputOLSAlignment runtime and its single W_OLS writer",
            "PlaneSpec, ComplexField, IntensityFrame, and OperationProvenance",
            "IdealAmplitudeSLM and CoherentImagingRelay",
            "FiniteCropPhaseStack for the project encoder/generator",
            "FiberArrayOLS complex modal overlap, coefficient transport, and re-emission",
            "local MNIST loader and fixed stratified split helper",
            "PROJECT_COMPLEX_OLS global-phase quotient conclusion for diagnostics only",
        ],
        "retire_from_mainline": [
            {
                "mechanism": "amplitude-only q as authoritative latent",
                "status": "historical baseline; not used by aligned training or generation",
                "preserve": True,
            },
            {
                "mechanism": "historical truncated q prior and q-derived Gaussian/coupla producer",
                "status": "not reused; not a full-complex target prior",
                "preserve": True,
            },
            {
                "mechanism": "independent phase inheritance, nearest phase donor, and q-conditioned complex-phasor residual",
                "status": "historical N3 baselines only; no phase repair prerequisite",
                "preserve": True,
            },
            {
                "mechanism": "formal label+8D controller and identity OLS",
                "status": "separate formal compatibility baseline; not this LightGen-aligned research runtime",
                "preserve": True,
            },
            {
                "mechanism": "electronic rank-16 reconstruction, software VAE decoder, whole-field identity bypass",
                "status": "outside aligned semantic-generation mainline",
                "preserve": True,
            },
        ],
        "unresolved": _chen_unresolved(source),
        "outer_gate": {
            "passed": True,
            "answer": "YES",
            "checks": {
                "aligned_runtime_immutable": True,
                "only_encoder_and_generator_phase_parameters_trainable": True,
                "authoritative_z_in_C100": True,
                "full_complex_sampleable_target": True,
                "latent_objective_directly_matches_full_complex_distribution": True,
                "same_target_prior_for_training_and_future_sampling": True,
                "no_q_to_phase_reconstruction": True,
                "no_training_or_generation_executed": True,
            },
            "hard_stop_conditions": {
                "aligned_authority_unavailable": False,
                "trainable_scope_unavailable": False,
                "complex_state_undefined": False,
                "direct_prior_undefined": False,
                "full_complex_loss_undefined": False,
                "shared_w_ols_unavailable": False,
                "data_roles_unavailable": False,
                "invented_chen_fact": False,
            },
            "reason": "The aligned runtime is structurally verified, its native C^100 OLS interface is explicit, and all unresolved BOGT details can be replaced by frozen project decisions without presenting them as Chen facts.",
        },
        "governance": dict(governance),
        "totality_checks": dict(totality),
        "definition_consumer": _ref(repo_root, Path(__file__).resolve().relative_to(repo_root), "training contract definition consumer"),
        "next_action": NEXT_ACTION,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def define(repo_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    output_path = root / OUTPUT_DIR / CONTRACT_NAME
    _require(not (root / OUTPUT_DIR).exists(), "definition output already exists; refusing overwrite")
    git = _validate_git_boundary(root)
    source = _validate_source_identity(root)
    alignment = _validate_alignment_authority(root)
    runtime = _validate_runtime_authority(root)
    complex_ols = _validate_project_complex_ols_authority(root)
    latest_result = _validate_latest_scientific_boundary(root)
    governance = _validate_governance(root)
    dataset = _validate_dataset_authority(root)
    trainable = _inspect_trainable_parameter_surface()
    totality = _contract_totality_checks()
    payload = _build_payload(
        root,
        git,
        source,
        alignment,
        runtime,
        complex_ols,
        latest_result,
        governance,
        dataset,
        trainable,
        totality,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_new_json(output_path, payload, artifact_kind="LightGen complex OLS semantic training contract")
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
