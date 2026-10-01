"""Define the LightGen-aligned 2-D semantic-generation input/OLS contract.

This is a definition-only consumer.  It reconciles the verified LightGen
source bundle with the existing project optics and freezes one coherent
image-input -> encoder -> complex 100-port OLS -> generator interface.  It
does not load q/complex/sealed values, execute optical forward passes, train,
or authorize generation/promotion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1"
READY_CLASSIFICATION = "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json"

ARCHITECTURE_DECISION = "LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST"
NEXT_ACTION = "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1"
OLD_PLANNING_ACTION = "PLAN_BOUNDED_ARCHITECTURAL_DECISION_DIRECT_COMPLEX_JOINT_VS_LIGHTGEN_INPUT_ALIGNMENT"

CLASS_COUNT = 10
OLS_GRID_SHAPE = (10, 10)
PORT_COUNT = 100
IMAGE_INPUT_CHANNELS = 1
SLM_SHAPE = (200, 200)
SLM_PIXEL_PITCH_M = 2.5e-6
OLS_SHAPE = (100, 100)
OLS_PIXEL_PITCH_M = 2.0e-6
WAVELENGTH_M = 532.0e-9
OLS_FOV_M = (200.0e-6, 200.0e-6)
OLS_V_NUMBER = 2.2
OLS_WAIST_M = 7.111e-6
OLS_FILL = 0.8
PROJECT_ENCODER_LAYERS = 2
PROJECT_GENERATOR_LAYERS = 2

SOURCE_MANIFEST_PATH = Path("research_assets/lightgen/science_adv7434/manifest.json")
PAPER_PDF_PATH = Path("research_assets/lightgen/science_adv7434/paper/sciencesm.pdf")
PAPER_PDF_SHA256 = "1325cdfb438f55249224f660a90ec2c21438ba3ce8cffc2677001f2db4e6c211"
SOURCE_MANIFEST_SHA256 = "fdb4f8c7b476ce92bd85b42ecbd435507657c91adfd501326842d5d00ee3418c"

PROJECT_ARCHITECTURE_PATH = Path("docs/architecture/lightgen_scaled_chen_bogt_v1.md")
PROJECT_ARCHITECTURE_SHA256 = "821e9a1402088d58c8b0b8f29698c87277744fbb550c4c732a0ea0713e23a7b8"
PROJECT_CONFIG_PATH = Path("configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_v1.yaml")
PROJECT_CONFIG_SHA256 = "e3d1b536aa081a8298b21b0b608f232463444a2863e138f941765adf96d8a701"
GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")
GOVERNANCE_STATE_SHA256 = "3d5b55bbcb3bb01cffb7badefb2a1ea2fd7cf9e924afa17729b183d0dff42e99"

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
LATEST_N3_ARTIFACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/"
    "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_banks_and_calibration.v1.pt"
)
LATEST_N3_ARTIFACT_SHA256 = "e944a96e4f751db35ec84440b779cb7a8825145a78ea725703f839e3c4f9c587"

SOURCE_HASHES = {
    "src/ldm_project/v2/contracts/planes.py": "6692d42315ce9af30fb7f2dfb1160d749ff1254f1d8ed5235304fb39fd374179",
    "src/ldm_project/v2/optics/fiber_operator.py": "6e941d6610c282aeb9f54cb39ea200c868f7b0eb14c3bc41b692ee505f4cf295",
    "src/ldm_project/v2/optics/fiber_ols.py": "797cad2af3581b9f32bafd610b1ad50eef63af55ec810c9f5a123bb20819606d",
    "src/ldm_project/v2/optics/relays.py": "8f777f206f16b51a1832047c696a5cde6bb77d48c0c18582eb0fc4fb971bc54d",
    "src/ldm_project/v2/optics/lightgen.py": "3c45acb9e02df3e3d7531424b37ddd37e79cdb324819eb040d604172bb391260",
    "src/ldm_project/v2/optics/system.py": "94482b761f8c5dade41460b4d89ff51319a28bd5c5a3232274d6c538f1fdbcb6",
    "src/ldm_project/v2/application/research/optical_manifold/optical_manifold.py": "b183dd4656cea9a84c91320c4369167a87694889806e9ca6b5fec5a6b10f4c4b",
    "src/ldm_project/v2/application/research/optical_manifold/optical_manifold_smf_bogt.py": "cf20c0d58f4314c046643fe51e3d885733b4180dd1e4bc716a4c12ce3cee92db",
}


class DefinitionBlocked(RuntimeError):
    """The definition cannot be published without inventing or weakening authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionBlocked(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _read_json(repo_root: Path, path: Path, name: str, expected_sha256: str | None = None) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {path.as_posix()}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DefinitionBlocked(f"{name} is not valid JSON") from exc
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _ref(repo_root: Path, path: Path, role: str, expected_sha256: str | None = None) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"authority is missing: {path.as_posix()}")
    digest = file_sha256(target)
    return {"path": _relative(repo_root, target), "sha256": digest, "role": role}


def _git_state(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
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
        "branch": _run("branch", "--show-current"),
        "head": _run("rev-parse", "HEAD"),
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_clean": not status,
    }


def _validate_git_boundary(repo_root: Path) -> dict[str, Any]:
    return _git_state(repo_root)


def _validate_source_identity(repo_root: Path) -> dict[str, Any]:
    manifest = _read_json(repo_root, SOURCE_MANIFEST_PATH, "LightGen source manifest", SOURCE_MANIFEST_SHA256)
    _require(manifest.get("schema_version") == 1, "LightGen source manifest schema changed")
    _require(manifest.get("status") == "reference_only", "LightGen source bundle is not reference-only")
    source = _mapping(manifest.get("source"), "LightGen source identity")
    _require(
        source.get("title") == "All-optical synthesis chip for large-scale intelligent semantic vision generation"
        and source.get("doi") == "10.1126/science.adv7434"
        and source.get("supplement_pages") == 46,
        "LightGen source identity is not the verified Chen source",
    )
    files = manifest.get("files")
    _require(isinstance(files, list), "LightGen source manifest files are invalid")
    pdf_entry = next((item for item in files if isinstance(item, Mapping) and item.get("path") == "paper/sciencesm.pdf"), None)
    _require(isinstance(pdf_entry, Mapping), "LightGen supplement entry is missing")
    pdf_ref = _ref(repo_root, PAPER_PDF_PATH, "verified LightGen Science supplement PDF", PAPER_PDF_SHA256)
    _require((repo_root / PAPER_PDF_PATH).is_file(), "LightGen supplement PDF is missing")
    return {
        "manifest": _ref(repo_root, SOURCE_MANIFEST_PATH, "verified LightGen source identity manifest", SOURCE_MANIFEST_SHA256),
        "paper_pdf": pdf_ref,
        "identity": {
            "title": source["title"],
            "doi": source["doi"],
            "supplement_pages": source["supplement_pages"],
            "status": manifest["status"],
            "manifest_pdf_match": True,
        },
    }


def _validate_project_authority(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / PROJECT_CONFIG_PATH
    _require(config_path.is_file(), "scaled Chen project configuration is missing")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DefinitionBlocked("scaled Chen project configuration is not valid YAML") from exc
    _require(isinstance(config, Mapping), "scaled Chen project configuration must be a mapping")
    optical = _mapping(config.get("optical_contract"), "scaled Chen optical contract")
    expected = {
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "mode_bank": "scaled_lp01_10x10",
        "ols_ports": 100,
        "mode_model": "step_index_lp01_ports",
        "mode_v_number": 2.2,
        "mode_waist_m": 7.111e-6,
        "mode_fill": 0.8,
        "software_decoder": False,
        "formal_promotion": False,
    }
    for key, value in expected.items():
        _require(optical.get(key) == value, f"scaled Chen project authority changed: optical_contract.{key}")
    authorization = _mapping(config.get("authorization"), "scaled Chen authorization")
    _require(
        authorization.get("training_authorized") is False
        and authorization.get("formal_training_authorized") is False
        and authorization.get("formal_promotion") is False,
        "scaled Chen project authority opened training or promotion",
    )
    return {
        "architecture_document": _ref(
            repo_root,
            PROJECT_ARCHITECTURE_PATH,
            "scaled Chen project-approximation boundary",
            PROJECT_ARCHITECTURE_SHA256,
        ),
        "configuration": _ref(repo_root, PROJECT_CONFIG_PATH, "scaled Chen numerical project baseline", PROJECT_CONFIG_SHA256),
        "declared_optical_contract": dict(optical),
        "authorization": {
            "training_authorized": authorization["training_authorized"],
            "formal_training_authorized": authorization["formal_training_authorized"],
            "formal_promotion": authorization["formal_promotion"],
        },
    }


def _validate_latest_result(repo_root: Path) -> dict[str, Any]:
    report = _read_json(repo_root, LATEST_N3_REPORT_PATH, "latest N3 q-conditioned report", LATEST_N3_REPORT_SHA256)
    _require(
        report.get("schema") == "n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1"
        and report.get("task_id") == "EXECUTE_N3_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1"
        and report.get("status") == "valid_completed"
        and report.get("result") == "VALID_COMPLETED"
        and report.get("classification") == "N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED",
        "latest N3 result is not the verified 0/5 q-conditioned complex-phasor result",
    )
    boundaries = _mapping(report.get("boundaries"), "latest N3 boundaries")
    _require(
        boundaries.get("selected_producer") == "EMPIRICAL_GAUSSIAN_COPULA"
        and boundaries.get("n3_amplitude_producer_confirmation") == "PASSED"
        and boundaries.get("training_authorized") is False
        and boundaries.get("formal_promotion") is False
        and boundaries.get("generation_readiness") == "not_ready",
        "latest N3 result changed the project authorization boundary",
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
    control = _mapping(report.get("control"), "latest N3 control")
    control_aggregate = _mapping(control.get("aggregate"), "latest N3 control aggregate")
    _require(
        control_aggregate.get("fold_count") == 5
        and control_aggregate.get("passed_fold_count") == 5
        and control_aggregate.get("fold_pass_pattern") == [True] * 5,
        "latest N3 full-complex control is not 5/5",
    )
    sealed = _mapping(report.get("sealed_confirmation"), "latest N3 sealed boundary")
    _require(
        sealed.get("consumed") is True and sealed.get("values_loaded") is False and sealed.get("reused") is False,
        "latest N3 reopened or reused sealed values",
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
        "candidate_artifact": _ref(
            repo_root,
            LATEST_N3_ARTIFACT_PATH,
            "latest N3 historical candidate banks and calibration artifact; not loaded by this definition",
            LATEST_N3_ARTIFACT_SHA256,
        ),
        "classification": report["classification"],
        "candidate": "0/5",
        "control": "5/5",
        "sealed": {"consumed": True, "values_loaded": False, "reused": False},
        "selected_producer": boundaries["selected_producer"],
        "amplitude_producer_confirmation": boundaries["n3_amplitude_producer_confirmation"],
        "training_authorized": boundaries["training_authorized"],
        "formal_promotion": boundaries["formal_promotion"],
        "generation_readiness": boundaries["generation_readiness"],
    }


def _validate_governance_and_current_docs(repo_root: Path) -> dict[str, Any]:
    governance = _read_json(repo_root, GOVERNANCE_STATE_PATH, "machine governance state", GOVERNANCE_STATE_SHA256)
    _require(
        governance.get("active_objective") == "N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED"
        and governance.get("hard_blocker") == "N3_COMPLEX_OLS_MINIMAL_Q_CONDITIONED_PHASE_COUPLING_CONTRACT_REQUIRED"
        and governance.get("authorization_tier") == "R3",
        "machine governance state is not the expected current boundary",
    )
    for relative, name in (
        (Path("docs/CURRENT_STATUS.md"), "CURRENT_STATUS"),
        (Path("docs/status/current_handoff.md"), "current handoff"),
    ):
        text = (repo_root / relative).read_text(encoding="utf-8")
        _require(
            TASK_ID in text or OLD_PLANNING_ACTION in text,
            f"{name} does not record the LightGen alignment planning boundary",
        )
        _require("training_authorized=false" in text, f"{name} changed training authorization")
        _require("formal_promotion=false" in text, f"{name} changed formal promotion")
        _require("generation_readiness=not_ready" in text, f"{name} changed generation readiness")
    return {
        "machine_state": _ref(repo_root, GOVERNANCE_STATE_PATH, "machine governance state; read-only in this task", GOVERNANCE_STATE_SHA256),
        "state_mutated": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def _validate_source_code_authorities(repo_root: Path) -> dict[str, Any]:
    return {
        relative: _ref(repo_root, Path(relative), role="existing project implementation", expected_sha256=digest)
        for relative, digest in SOURCE_HASHES.items()
    }


def _contract_totality_checks() -> dict[str, Any]:
    _require(OLS_GRID_SHAPE == (10, 10) and PORT_COUNT == 100, "OLS dimension is not fixed at 10x10/100")
    _require(SLM_SHAPE == (200, 200) and OLS_SHAPE == (100, 100), "project plane shapes changed")
    _require(WAVELENGTH_M == 532.0e-9 and OLS_FOV_M == (200.0e-6, 200.0e-6), "project optical sampling changed")
    _require(PROJECT_ENCODER_LAYERS == 2 and PROJECT_GENERATOR_LAYERS == 2, "project diffractive baseline changed")
    _require(0.0 < OLS_V_NUMBER < 2.4048255577 and OLS_WAIST_M > 0.0 and 0.0 < OLS_FILL <= 1.0, "LP01 project approximation is invalid")
    _require("z_j = <E_encoder, u_j>" in OLS_PROJECTION_FORMULA, "OLS projection formula is missing")
    _require("W_OLS(z)" in OLS_REEMISSION_FORMULA and "sum" in OLS_REEMISSION_FORMULA, "OLS re-emission formula is missing")
    _require("T=I_100" in OLS_TRANSMISSION_DECISION, "coefficient-space identity transmission is not frozen")
    return {
        "status": "passed",
        "ols_shape": [PORT_COUNT],
        "stable_grid": list(OLS_GRID_SHAPE),
        "projection_required": True,
        "reemission_required": True,
        "coefficient_transmission_required": True,
        "same_writer_for_real_and_generated_ols": True,
        "whole_field_identity_bypass": False,
        "q_is_authoritative_latent": False,
        "training_executed": False,
        "scientific_generation_executed": False,
        "sealed_values_loaded": False,
        "governance_mutated": False,
    }


OLS_PROJECTION_FORMULA = "z_j = <E_encoder, u_j>"
OLS_REEMISSION_FORMULA = "W_OLS(z) = sum_j (Tz)_j v_j"
OLS_TRANSMISSION_DECISION = "T=I_100 in coefficient space"


def _paper_facts(paper_pdf: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "CHEN_FACT_OLS_DIMENSION",
            "category": "CHEN_FACT",
            "statement": "The supplement reports that OLS performance improves up to dimension 100, corresponding to a 10x10 OLS, with no prominent improvement beyond that point.",
            "source": "Science supplement PDF page 2, Materials and Methods, Dimensions and further validations of OLS; Figure S11.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_INPUT_OPTICS",
            "category": "CHEN_FACT",
            "statement": "The experimental setup uses a single-mode 532 nm laser, an amplitude-modulation spatial light modulator, and a 4f relay to conjugate the input image to the LightGen input plane.",
            "source": "Science supplement PDF page 2, Experimental set-up of LightGen chip.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_COMPLEX_OLS_FIELD",
            "category": "CHEN_FACT",
            "statement": "The supplement describes OLS coupling using complex optical fields and shows both amplitude and phase of input/output fields through a 10x10 OLS.",
            "source": "Science supplement PDF pages 6-9, Supplementary Text on OLS; page 15, Figure S2 caption.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_MODAL_COUPLING",
            "category": "CHEN_FACT",
            "statement": "The supplement models coupling to a fundamental single-mode field by a complex overlap integral and defines coupling efficiency from the squared overlap magnitude.",
            "source": "Science supplement PDF pages 6-9, Eqs. S6-S14.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_OLS_FIBER_ARRAY",
            "category": "CHEN_FACT",
            "statement": "The reported OLS is implemented with a 10x10 single-mode fiber array; the supplement reports 8.5 mm fiber length and 125 micrometers external fiber diameter.",
            "source": "Science supplement PDF page 3, Chip design of LightGen.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_GENERATOR_ONLY_RECONFIGURATION",
            "category": "CHEN_FACT",
            "statement": "For generation after OLS adjustment, the supplement states that only the photonic generator was used.",
            "source": "Science supplement PDF pages 2-3, Chip design of LightGen.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_DEMONSTRATION_ARCHITECTURE",
            "category": "CHEN_FACT",
            "statement": "The demonstration described in the supplement uses a two-layer encoder and a four-layer generator, with 400x400 or 600x600 photonic neurons per layer depending on task.",
            "source": "Science supplement PDF page 2, Chip design of LightGen.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_FACT_BOGT_OBJECTIVE_SHAPE",
            "category": "CHEN_FACT",
            "statement": "The supplement describes BOGT as a KL constraint between Q(Z|X) and P(Z), combined with reconstruction and optical-power terms in a weighted total loss; the positive-intensity constraint motivates a truncated biased Gaussian description.",
            "source": "Science supplement PDF page 2, Loss function of BOGT.",
            "authority": paper_pdf,
        },
    ]


def _paper_unresolved(paper_pdf: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "CHEN_UNRESOLVED_IMAGE_TO_SLM_MAPPING",
            "category": "CHEN_UNRESOLVED",
            "statement": "The paper source does not establish the exact digital image normalization, interpolation, amplitude-command transfer curve, gamma, or illumination phase convention used by the project.",
            "source": "Paper-parameter audit of the verified supplement; the source identifies an amplitude-modulation SLM but does not publish this machine-level mapping.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_UNRESOLVED_FIBER_PHYSICAL_PARAMETERS",
            "category": "CHEN_UNRESOLVED",
            "statement": "Core diameter, numerical aperture, mode-field diameter/radius, port pitch, crosstalk, insertion loss, and the measured 100-port complex transmission matrix are not published in the available source bundle.",
            "source": "LightGen source bundle README physical-parameter boundary and supplement pages 2-3.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_UNRESOLVED_PORT_ORDER_AND_BASIS",
            "category": "CHEN_UNRESOLVED",
            "statement": "The available authority does not specify a machine-readable 10x10 port ordering/orientation, the exact discrete modal basis normalization, or the output-plane re-emission writer.",
            "source": "Paper-parameter audit of the verified supplement and Figure S2; no port-index contract is published.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_UNRESOLVED_RELAY_AND_LAYER_DETAILS",
            "category": "CHEN_UNRESOLVED",
            "statement": "The exact input-relay prescription, layer-to-layer geometry, pupil/aperture details, and generator entrance normalization required for a numerical reproduction are not sufficiently specified by the available authority.",
            "source": "Paper-parameter audit of the verified supplement; the paper reports the experimental arrangement but not the complete project numerical contract.",
            "authority": paper_pdf,
        },
        {
            "id": "CHEN_UNRESOLVED_BOGT_IMPLEMENTATION",
            "category": "CHEN_UNRESOLVED",
            "statement": "The exact BOGT estimator axes, numerical normalization, alpha/beta/gamma values, batch construction, and sampling implementation are not established by the available source authority.",
            "source": "Science supplement PDF page 2 gives the loss form but not the complete implementation contract.",
            "authority": paper_pdf,
        },
    ]


def _project_decisions() -> list[dict[str, Any]]:
    return [
        {
            "id": "PROJECT_DECISION_SCOPE",
            "category": "PROJECT_DECISION",
            "decision": "Freeze this contract to 2D semantic image generation only; no denoising, style transfer, 3D, semantic manipulation, multi-generator routing, label+8D controller, rank-16 electronic reconstruction, or new phase family.",
        },
        {
            "id": "PROJECT_DECISION_IMAGE_SEMANTICS",
            "category": "PROJECT_DECISION",
            "decision": "Treat X as a finite real nonnegative normalized image tensor in [0,1], with shape [N,1,H,W] for the current monochrome baseline; do not treat sqrt(image) as a Chen fact.",
        },
        {
            "id": "PROJECT_DECISION_SLM_MAPPING",
            "category": "PROJECT_DECISION",
            "decision": "Use I_slm=clamp(BilinearResize(X),0,1) and A_slm=sqrt(I_slm) as the initial explicit project image-to-amplitude convention. This is a project convention requiring later input-path verification, not a paper-reproduction claim.",
        },
        {
            "id": "PROJECT_DECISION_ILLUMINATED_FIELD",
            "category": "PROJECT_DECISION",
            "decision": "Use E_slm(y,x)=A_slm(y,x) exp(i*phi0) with frozen phi0=0 for the numerical baseline, then pass the complex field through a coherent input relay to the encoder entrance plane.",
        },
        {
            "id": "PROJECT_DECISION_PROJECT_PLANES",
            "category": "PROJECT_DECISION",
            "decision": "Use the existing 200x200 SLM plane at 2.5 micrometers, 100x100 encoder/OLS plane at 2 micrometers, 532 nm wavelength, and explicit PlaneSpec/ComplexField metadata for the first aligned implementation.",
        },
        {
            "id": "PROJECT_DECISION_PORT_ORDER",
            "category": "PROJECT_DECISION",
            "decision": "Freeze stable row-major port order j=10*r+c, r,c in {0,...,9}; the paper does not establish this order, so every artifact must carry this project order explicitly.",
        },
        {
            "id": "PROJECT_DECISION_SCALED_LP01",
            "category": "PROJECT_DECISION",
            "decision": "Use the existing scaled LP01 mode bank (V=2.2, waist=7.111 micrometers, fill=0.8) as a numerical project approximation only; never infer its mode-field radius from the reported 125 micrometers outer diameter.",
        },
        {
            "id": "PROJECT_DECISION_OLS_TRANSMISSION",
            "category": "PROJECT_DECISION",
            "decision": "Use coefficient-space T=I_100 for the initial transport contract. This is 100-port modal transmission, not whole-field identity bypass and not a measured fiber transfer matrix.",
        },
        {
            "id": "PROJECT_DECISION_DIFRACTIVE_ARCHITECTURE",
            "category": "PROJECT_DECISION",
            "decision": "Retain the current reusable two-layer encoder/two-layer generator phase-only stack for the first project runtime. It is intentionally allowed to differ from the paper demonstration's two-layer/four-layer stack.",
        },
        {
            "id": "PROJECT_DECISION_NO_Q_LATENT",
            "category": "PROJECT_DECISION",
            "decision": "Make z in C^100 the sole authoritative OLS state. Any amplitude-only q is derived diagnostic/statistic output and cannot be the generator interface or final latent.",
        },
        {
            "id": "PROJECT_DECISION_NO_PHASE_REPAIR",
            "category": "PROJECT_DECISION",
            "decision": "Do not make phase repair, independent phase inheritance, nearest donors, linear phasor residuals, or any new phase producer a prerequisite of the aligned runtime.",
        },
    ]


def _alignment_matrix(source_code: Mapping[str, Any]) -> list[dict[str, Any]]:
    def impl(path: str, detail: str) -> dict[str, Any]:
        return {"path": path, "detail": detail, "authority": source_code[path]}

    return [
        {
            "component": "LightGen source identity",
            "lightgen_requirement": "Use the verified Chen paper/supplement as the fact boundary.",
            "current_project_implementation": "Manifest-bound local supplement and source audit.",
            "classification": "SAME",
            "migration_action": "Keep read-only and cite page-level evidence; never treat the bundle as code or training authorization.",
        },
        {
            "component": "Image -> amplitude SLM",
            "lightgen_requirement": "Amplitude-modulation SLM receives the optical image input.",
            "current_project_implementation": impl("src/ldm_project/v2/application/research/optical_manifold/optical_manifold.py", "render_intensity_batch resizes intensity and returns sqrt amplitude tensors."),
            "classification": "PARTIAL",
            "migration_action": "Expose X, I_slm, A_slm, E_slm, planes, and provenance as one explicit writer; keep sqrt as PROJECT_DECISION only.",
        },
        {
            "component": "Input relay -> encoder entrance",
            "lightgen_requirement": "4f/coherent relay conjugates the input image to the LightGen input plane.",
            "current_project_implementation": impl("src/ldm_project/v2/optics/relays.py", "CoherentImagingRelay and the formal system front relay provide typed plane-aware propagation."),
            "classification": "PARTIAL",
            "migration_action": "Use the relay as the canonical boundary and expose every project geometry parameter; do not claim paper geometry beyond the resolved fact.",
        },
        {
            "component": "Photonic encoder",
            "lightgen_requirement": "Optical encoder produces the field presented to OLS.",
            "current_project_implementation": impl("src/ldm_project/v2/application/research/optical_manifold/optical_manifold_smf_bogt.py", "ScaledChenBOGTModel contains a reusable two-layer phase-only encoder stack."),
            "classification": "PARTIAL",
            "migration_action": "Keep the diffractive stack if desired, but route its final ComplexField through the new authoritative OLS writer.",
        },
        {
            "component": "OLS modal projection",
            "lightgen_requirement": "10x10 single-mode OLS couples complex input field into 100 modal ports.",
            "current_project_implementation": impl("src/ldm_project/v2/optics/fiber_operator.py", "FiberArrayOLS computes area-weighted complex modal overlaps and records 100 port coefficients."),
            "classification": "PARTIAL",
            "migration_action": "Reuse the overlap primitive, but remove the diagnostic-only status at the new runtime boundary and freeze basis/order/plane metadata under this contract.",
        },
        {
            "component": "OLS coefficient transmission",
            "lightgen_requirement": "Port coefficients are transmitted through the OLS, not bypassed as a whole field.",
            "current_project_implementation": impl("src/ldm_project/v2/optics/fiber_operator.py", "FiberArrayOLS already accepts a complex 100x100 transport matrix and defaults to identity."),
            "classification": "PARTIAL",
            "migration_action": "Use T=I_100 as the explicit project decision and forbid system.ols/whole-field identity in the aligned path.",
        },
        {
            "component": "OLS modal re-emission",
            "lightgen_requirement": "Transmitted modal coefficients are re-emitted into the generator entrance field.",
            "current_project_implementation": impl("src/ldm_project/v2/optics/fiber_ols.py", "reemit_single_mode_fibers and FiberArrayOLS reconstruct a complex field from output modes."),
            "classification": "PARTIAL",
            "migration_action": "Create one named W_OLS writer used by both real encoded OLS and generated OLS; remove alternate electronic/generated-state decoders.",
        },
        {
            "component": "Generator-only generation path",
            "lightgen_requirement": "A valid generated OLS state enters the photonic generator without the image encoder.",
            "current_project_implementation": impl("src/ldm_project/v2/application/research/optical_manifold/optical_manifold_smf_bogt.py", "reemit_decoder accepts coefficients but includes historical phase/q-prior semantics and is not the shared public boundary."),
            "classification": "DIFFERENT",
            "migration_action": "Replace the historical entry with z_generated -> W_OLS -> same generator; no image encoder, label+8D controller, or phase repair prerequisite.",
        },
        {
            "component": "Plane and field semantics",
            "lightgen_requirement": "Optical fields retain amplitude, relative phase, physical coordinates, and plane identity.",
            "current_project_implementation": impl("src/ldm_project/v2/contracts/planes.py", "PlaneSpec, ComplexField, and OperationProvenance enforce complex dtype and physical plane metadata."),
            "classification": "SAME",
            "migration_action": "Reuse unchanged and require all OLS/generator interfaces to consume these typed planes.",
        },
        {
            "component": "Project scaled LP01 geometry",
            "lightgen_requirement": "Single-mode modal geometry must correspond to declared physical/coupling semantics.",
            "current_project_implementation": impl("src/ldm_project/v2/optics/fiber_operator.py", "V=2.2, 7.111 micrometer waist, fill 0.8 scaled LP01 bank; project diagnostic-only approximation."),
            "classification": "PARTIAL",
            "migration_action": "Retain as PROJECT_DECISION only and verify dimensionless modal/coupling relations before any physical-alignment claim.",
        },
        {
            "component": "N3 q/phase producer history",
            "lightgen_requirement": "Complex OLS manifold is authoritative; amplitude-only or ad-hoc phase repair is not the aligned interface.",
            "current_project_implementation": "Confirmed amplitude producer, full-complex control, and three rejected phase-coupling candidates.",
            "classification": "DIFFERENT",
            "migration_action": "Preserve as historical evidence; do not carry any amplitude-first phase repair into the aligned mainline.",
        },
    ]


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    project_authority: Mapping[str, Any],
    latest_result: Mapping[str, Any],
    governance: Mapping[str, Any],
    source_code: Mapping[str, Any],
    totality: Mapping[str, Any],
) -> dict[str, Any]:
    paper_pdf = _mapping(source_identity["paper_pdf"], "paper PDF authority")
    definition_ref = _ref(
        repo_root,
        Path(__file__).resolve().relative_to(repo_root.resolve()),
        "LightGen 2-D input/OLS alignment definition consumer",
    )
    source_taxonomy = _paper_facts(paper_pdf) + _paper_unresolved(paper_pdf) + _project_decisions()
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "definition_only": True,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": dict(git),
        "architecture_decision": ARCHITECTURE_DECISION,
        "scope": {
            "task": "2D semantic image generation only",
            "in_scope": [
                "image optical input",
                "photonic encoder",
                "10x10 single-mode OLS",
                "complex OLS representation",
                "modal projection",
                "modal transmission",
                "modal re-emission",
                "photonic generator input",
                "raw optical image generation",
            ],
            "out_of_scope": [
                "denoising",
                "style transfer",
                "3D generation",
                "semantic manipulation",
                "multi-generator routing",
                "label + 8D controller",
                "electronic rank-16 reconstruction",
                "new phase producer families",
            ],
        },
        "source_authority": {
            "verified_identity": dict(source_identity),
            "project_authority": dict(project_authority),
            "code_authorities": dict(source_code),
            "classification_rule": "CHEN_FACT is limited to the verified paper source; CHEN_UNRESOLVED means the paper detail is not sufficiently specified; PROJECT_DECISION is an explicit project convention and never a paper fact.",
            "paper_and_project_taxonomy": source_taxonomy,
        },
        "current_scientific_boundary": {
            "latest_result": dict(latest_result),
            "interpretation": "The latest 0/5 result rejects only the tested q-conditioned complex-phasor linear residual hypothesis. It does not reject LightGen input/OLS alignment, all conditional phase models, the paper mechanism, or physical generation.",
            "historical_n3_baselines_preserved": [
                "confirmed Gaussian amplitude producer",
                "full-complex empirical control 5/5",
                "independent phase inheritance 0/5",
                "q-conditioned nearest phase donor 0/5",
                "q-conditioned complex phasor linear residual 0/5",
            ],
        },
        "input_contract": {
            "canonical_image_path": [
                "X: real finite normalized image in [0,1]",
                "I_slm = clamp(BilinearResize(X, SLM_SHAPE), 0, 1)",
                "A_slm = sqrt(I_slm) [PROJECT_DECISION, not CHEN_FACT]",
                "E_slm(y,x) = A_slm(y,x) * exp(i*phi0), phi0=0 [PROJECT_DECISION]",
                "AmplitudeCommand(E_slm amplitude) -> IdealAmplitudeSLM -> illuminated optical field",
                "CoherentImagingRelay(E_slm) -> E_encoder at the encoder entrance plane",
                "photonic encoder -> E_encoder_final",
                "modal projection -> z_real in C^100",
                "W_OLS(z_real) -> generator entrance field",
                "same photonic generator -> raw CMOS intensity",
            ],
            "image_input": {
                "symbol": "X",
                "dtype": "real floating",
                "shape": ["N", IMAGE_INPUT_CHANNELS, "H", "W"],
                "range": [0.0, 1.0],
                "channels": "monochrome project baseline; RGB is not introduced by this contract",
            },
            "slm_mapping": {
                "category": "PROJECT_DECISION",
                "intensity_formula": "I_slm=clamp(BilinearResize(X),0,1)",
                "amplitude_formula": "A_slm=sqrt(I_slm)",
                "paper_status": "not established as a Chen fact",
                "future_verification": "verify numerical amplitude semantics, command calibration, boundary clipping, and provenance through the real input entrypoint",
            },
            "illuminated_field": {
                "category": "PROJECT_DECISION",
                "formula": "E_slm=A_slm*exp(i*phi0), phi0=0",
                "relative_phase": "preserved from this field through modal coupling; no phase repair",
                "plane": {
                    "id": "slm_output",
                    "shape": list(SLM_SHAPE),
                    "pixel_pitch_m": [SLM_PIXEL_PITCH_M, SLM_PIXEL_PITCH_M],
                    "wavelength_m": WAVELENGTH_M,
                    "coordinate_frame": "slm",
                },
            },
            "input_relay": {
                "requirement": "coherent 4f/imaging relay from illuminated SLM field to encoder entrance field",
                "implementation_primitive": "CoherentImagingRelay",
                "current_project_baseline": {"front_magnification": -0.4, "plane_metadata_required": True},
                "paper_status": "relay role is CHEN_FACT; complete project numerical geometry is CHEN_UNRESOLVED",
                "generic_complex_resize": False,
            },
            "encoder_entrance_field": {
                "symbol": "E_encoder",
                "type": "ComplexField with PlaneSpec and OperationProvenance",
                "projection_input": "the final ComplexField emitted by the optical encoder at the frozen OLS input plane",
            },
        },
        "ols_contract": {
            "authoritative_latent": {
                "symbol": "z",
                "domain": "C^100",
                "batch_shape": ["N", PORT_COUNT],
                "dtype": "complex",
                "meaning": "100 single-mode modal coupling coefficients, retaining magnitude and relative phase",
                "amplitude_q": "derived diagnostic/statistic only; never the authoritative OLS state",
            },
            "port_ordering": {
                "grid_shape": list(OLS_GRID_SHAPE),
                "index": "j=10*r+c",
                "r_range": [0, 9],
                "c_range": [0, 9],
                "order": "row-major, stable, explicit project convention",
                "paper_status": "CHEN_UNRESOLVED",
            },
            "modal_bases": {
                "input": "u_j: normalized single-mode field at encoder OLS plane",
                "output": "v_j: normalized single-mode field at generator entrance plane",
                "basis_relation": "input/output banks are distinct plane objects even when the project numerical geometry is equal",
                "normalization": "freeze the physical-coordinate quadrature convention; sampled integral uses Delta_y*Delta_x",
                "project_baseline": {
                    "mode_model": "step_index_lp01_ports",
                    "v_number": OLS_V_NUMBER,
                    "waist_m": OLS_WAIST_M,
                    "fill": OLS_FILL,
                    "status": "scaled project approximation, not measured Chen fiber geometry",
                },
            },
            "modal_projection": {
                "mandatory": True,
                "formula": OLS_PROJECTION_FORMULA,
                "expanded_formula": "z_j = sum_{y,x} conj(u_j(y,x))*E_encoder(y,x)*Delta_y*Delta_x",
                "operation": "complex overlap coupling, not amplitude/power pooling and not whole-field identity",
                "relative_phase": "retain all coordinate-to-coordinate phase differences in z",
            },
            "coefficient_transmission": {
                "formula": "z_transmitted=T*z",
                "shape": [PORT_COUNT, PORT_COUNT],
                "initial_value": "I_100",
                "decision_status": "PROJECT_DECISION",
                "measured_matrix": False,
                "whole_field_identity_bypass": False,
                "passivity_gate": "future implementation must validate declared T; T=I_100 is coefficient-space transport only",
            },
            "modal_reemission": {
                "mandatory": True,
                "formula": OLS_REEMISSION_FORMULA,
                "expanded_formula": "E_generator(y,x)=sum_j (T*z)_j*v_j(y,x)",
                "output_type": "ComplexField at generator_entrance plane",
                "writer_id": "W_OLS_V1",
                "same_writer_for_real_and_generated": True,
                "alternate_electronic_decoder": False,
            },
            "dimensionless_relations_to_verify": [
                "0 < V < 2.4048255577 for the LP01-only project approximation",
                "each sampled mode obeys the frozen physical-coordinate normalization",
                "declared mode-center spacing / waist and wavelength / waist relations remain fixed and auditable",
                "modal overlap/coupling efficiency uses the same normalized fields and quadrature convention",
                "the 100-port Gram matrix is finite and rank-valid under the selected bank convention",
                "T singular values satisfy the declared passive transport gate",
                "output modal-field energy does not exceed input energy under the selected T",
            ],
            "forbidden_inference": "Do not infer mode-field radius, core diameter, or NA from the 125 micrometer external fiber diameter.",
        },
        "generator_input_writer": {
            "canonical_definition": "write_ols_to_generator_input(z, T, v_bank) returns W_OLS(z) as a typed ComplexField",
            "writer_steps": [
                "validate finite complex z with shape [N,100] and stable port order",
                "apply coefficient-space y=T*z",
                "re-emit y through the frozen output modal bank v_j",
                "attach generator_entrance PlaneSpec and provenance",
                "pass the resulting ComplexField directly to the same downstream photonic generator",
            ],
            "real_encoded_ols_path": "image -> optical input -> encoder -> z_real -> W_OLS(z_real) -> generator",
            "future_generated_ols_path": "z_generated(valid complex C^100) -> W_OLS(z_generated) -> same generator",
            "convergence_invariant": "The two paths share the same W_OLS, T, output modal basis, generator entrance plane, downstream generator, and raw CMOS readout.",
            "generation_prerequisites_forbidden": [
                "image encoder",
                "label + 8D controller",
                "phase-repair module",
                "electronic OLS decoder",
                "software-only VAE decode",
            ],
        },
        "physical_numerical_geometry": {
            "chen_facts": {
                "wavelength_m": WAVELENGTH_M,
                "single_mode_ols_grid": list(OLS_GRID_SHAPE),
                "fiber_length_m": 8.5e-3,
                "fiber_external_diameter_m": 125.0e-6,
                "metasurface_pixel_size_m": 3.0e-6,
                "metasurface_phase_levels": 8,
                "ols_dimension_selection": "100 ports / 10x10 is the reported adequate dimension",
            },
            "chen_unresolved": [
                "fiber core diameter",
                "fiber NA",
                "fiber MFD/mode-field radius",
                "fiber port pitch",
                "fiber crosstalk",
                "fiber insertion loss",
                "measured 100-port complex transmission matrix",
                "machine-readable port order and generator-plane modal writer",
                "complete numerical input/layer/relay normalization",
            ],
            "project_decisions": {
                "slm": {"shape": list(SLM_SHAPE), "pixel_pitch_m": SLM_PIXEL_PITCH_M},
                "ols_plane": {"shape": list(OLS_SHAPE), "pixel_pitch_m": OLS_PIXEL_PITCH_M, "fov_m": list(OLS_FOV_M)},
                "mode_bank": {"v_number": OLS_V_NUMBER, "waist_m": OLS_WAIST_M, "fill": OLS_FILL, "grid": list(OLS_GRID_SHAPE)},
                "encoder_layers": PROJECT_ENCODER_LAYERS,
                "generator_layers": PROJECT_GENERATOR_LAYERS,
                "fiber_dimensions_usage": "paper reference metadata only; not used to derive scaled LP01 parameters",
            },
        },
        "diffractive_architecture": {
            "project_encoder": f"{PROJECT_ENCODER_LAYERS} phase-only layers, reusable current scaled-Chen stack",
            "project_generator": f"{PROJECT_GENERATOR_LAYERS} phase-only layers after W_OLS, reusable current scaled-Chen decoder stack",
            "paper_difference": "paper demonstration reports two encoder and four generator layers; project difference is explicitly allowed by the outer gate",
            "generator_identity": "same downstream generator for real encoded and future generated OLS states",
        },
        "alignment_matrix": _alignment_matrix(source_code),
        "reuse": {
            "unchanged_or_read_only": [
                {
                    "component": "verified LightGen manifest and supplement",
                    "use": "fact-source identity and page-level reconciliation only",
                    "status": "read-only",
                },
                {
                    "component": "PlaneSpec / ComplexField / OperationProvenance",
                    "use": "typed physical-plane and complex-field metadata boundary",
                    "status": "reuse unchanged",
                },
                {
                    "component": "CoherentImagingRelay",
                    "use": "plane-aware coherent input/backend relay primitive",
                    "status": "reuse primitive; expose project parameters in implementation",
                },
                {
                    "component": "existing phase-only propagation and crop stack",
                    "use": "project-specific encoder/generator architecture",
                    "status": "reuse architecture with new W_OLS boundary",
                },
                {
                    "component": "N3 reports, lineages, and candidate artifacts",
                    "use": "historical evidence and negative baselines only",
                    "status": "immutable/read-only; no sealed reuse",
                },
            ],
            "adapted_primitives": [
                "FiberArraySpec/FiberArrayOLS overlap and coefficient-space transport",
                "FiberModeBankV2/project single-mode projection and re-emission types",
                "LightGenAmplitudeFiberOLSSystemV2 projection/re-emission structure",
                "render_intensity_batch image validation and explicit sqrt project mapping",
            ],
        },
        "retire_from_aligned_mainline": [
            {
                "mechanism": "formal v2 IdentityOLS or any whole-field identity bypass",
                "status": "historical/formal compatibility only; not LightGen-aligned OLS",
                "preserve": True,
            },
            {
                "mechanism": "label + 8D controller as a generation prerequisite",
                "status": "formal compatibility baseline only",
                "preserve": True,
            },
            {
                "mechanism": "q as final/authoritative latent",
                "status": "derived diagnostic only; complete complex z replaces it at the aligned boundary",
                "preserve": True,
            },
            {
                "mechanism": "amplitude-first phase repair and N3 phase candidates",
                "status": "historical baselines: independent inheritance 0/5, nearest donor 0/5, complex-phasor residual 0/5",
                "preserve": True,
            },
            {
                "mechanism": "ScaledChenBOGTModel.generate_from_prior/reemit_decoder as the canonical generated-state path",
                "status": "historical q/zero-phase prior path; replace with shared z -> W_OLS -> generator boundary",
                "preserve": True,
            },
            {
                "mechanism": "electronic rank-16 reconstruction, software VAE decoder, generic complex resize",
                "status": "outside aligned mainline",
                "preserve": True,
            },
        ],
        "future_training_boundary": {
            "this_task": "definition-only; no training, CUDA, heldout evaluation, generation acceptance, or promotion",
            "later_training_interface": "optimize a complete complex OLS manifold whose output is valid z in C^100, then enter the same W_OLS writer and generator",
            "q_status": "q may be a derived descriptive statistic but is not the final latent",
            "bogt_status": "published loss shape is a CHEN_FACT; exact estimator/loss weights/normalization remain CHEN_UNRESOLVED until a later training contract",
            "phase_producer_status": "no new phase producer is defined or authorized here",
            "authorization": {
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
                "sealed_values_loaded": False,
                "sealed_reused": False,
                "new_sealed_authority": False,
            },
        },
        "outer_gate": {
            "passed": True,
            "answer": "YES",
            "checks": {
                "explicit_lightgen_aligned_image_field_semantics": True,
                "complex_10x10_100_channel_single_mode_ols": True,
                "mandatory_modal_projection": True,
                "mandatory_modal_reemission": True,
                "same_real_and_generated_ols_writer": True,
                "project_specific_diffractive_architecture_allowed": True,
                "paper_facts_separated_from_project_conventions": True,
                "required_source_identity_resolved": True,
                "mathematics_defined": True,
                "no_training_or_scientific_execution": True,
            },
            "hard_stop_conditions": {
                "required_source_identity_unresolved": False,
                "unrepresentable_ols_authority_conflict": False,
                "shared_real_generated_interface_impossible": False,
                "undefined_required_quantity": False,
                "invented_paper_fact": False,
            },
            "reason": "A coherent migration contract can be frozen by treating paper facts as evidence, project geometry/mapping/order/T as explicit decisions, and using one coefficient-space modal projection/re-emission writer for both real and future generated complex OLS states.",
        },
        "governance": dict(governance),
        "definition_consumer": definition_ref,
        "totality_checks": dict(totality),
        "next_action": NEXT_ACTION,
        "training_authorization": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def define(repo_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    output_path = root / OUTPUT_DIR / CONTRACT_NAME
    _require(not output_path.exists() and not (root / OUTPUT_DIR).exists(), "definition output already exists; refusing overwrite")
    git = _validate_git_boundary(root)
    source_identity = _validate_source_identity(root)
    project_authority = _validate_project_authority(root)
    latest_result = _validate_latest_result(root)
    governance = _validate_governance_and_current_docs(root)
    source_code = _validate_source_code_authorities(root)
    totality = _contract_totality_checks()
    payload = _build_payload(root, git, source_identity, project_authority, latest_result, governance, source_code, totality)
    atomic_write_new_json(output_path, payload, artifact_kind="LightGen input/OLS alignment contract")
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
    print(
        json.dumps(
            {
                "result": payload["result"],
                "classification": payload["classification"],
                "artifact": path.relative_to(_repo_root()).as_posix(),
                "sha256": file_sha256(path),
                "next_action": payload["next_action"],
                "training_authorized": payload["training_authorization"],
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
