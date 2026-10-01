"""Materialize the already-approved LightGen V2 training contract.

This is a definition consumer only.  It validates the existing aligned runtime
and prior evidence, inspects the local trainable surface without a forward pass,
and writes one immutable machine-readable contract.  It never loads dataset
values, creates an optimizer, performs a backward pass, or executes training.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V2"
GATE = TASK_ID
CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_contract.v2"
READY_CLASSIFICATION = "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V2_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"
OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_semantic_complex_ols_training_contract.v2.json"

ALIGNMENT_CONTRACT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json"
)
ALIGNMENT_CONTRACT_SHA256 = "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"
ALIGNMENT_RUNTIME = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
ALIGNMENT_RUNTIME_REPORT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"
)
ALIGNMENT_RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"
V1_CONTRACT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v1.json"
)
V1_CONTRACT_SHA256 = "7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f"
DIAGNOSIS_REPORT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_real_path_control_failure_diagnosis_v1_attempt_01/"
    "lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1.json"
)
DIAGNOSIS_REPORT_SHA256 = "8b98dbcb2955417ca4dacd2298144bfdc006fd697dc1dc36eb49ff63dadcda86"
POST_TRAINING_CONTRACT = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json"
)
POST_TRAINING_CONTRACT_SHA256 = "b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d"
TRAINING_DATA = Path("data/MNIST/processed/training.pt")
TRAINING_DATA_SHA256 = "8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
PROTECTED_DATA = Path("data/MNIST/processed/test.pt")
PROTECTED_DATA_SHA256 = "8c089c2f0a5db35de7961f4c60ed6b627a05141245d44f542eeee57b09b539e1"
EVALUATOR_IMPLEMENTATION = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_IMPLEMENTATION_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
EVALUATOR_CHECKPOINT = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
EVALUATOR_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
SEMANTIC_THRESHOLD = 0.937
SEMANTIC_EVALUATOR_QUALIFICATION_THRESHOLD = 0.985
OLS_PORT_COUNT = 100
OLS_GRID_SHAPE = (10, 10)
LOGICAL_BATCH_SIZE = 10
STAGE_A_STEPS = 300
STAGE_B_STEPS = 300
CHECKPOINT_STEPS = (100, 200, 300)
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
EPSILON_SCALE = 1.0e-12
GAUGE_EPSILON = 1.0e-6
MMD_BANDWIDTH_SQUARED = 200.0
PRIOR_NAMESPACE = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_PRIOR"
SPLIT_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321


class DefinitionBlocked(RuntimeError):
    """Raised when the approved definition cannot be materialized safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionBlocked(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(root: Path, relative: Path, expected_sha: str, name: str) -> dict[str, Any]:
    path = root / relative
    _require(path.is_file(), f"{name} is missing: {relative.as_posix()}")
    actual = file_sha256(path)
    _require(actual == expected_sha, f"{name} SHA mismatch: {actual}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DefinitionBlocked(f"{name} is not valid JSON") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _ref(root: Path, path: Path, role: str, sha: str) -> dict[str, str]:
    target = root / path
    _require(target.is_file(), f"{role} is missing: {path.as_posix()}")
    actual = file_sha256(target)
    _require(actual == sha, f"{role} SHA mismatch: {actual}")
    return {"path": _relative(root, target), "role": role, "sha256": actual}


def _git(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    branch = run("branch", "--show-current")
    head = run("rev-parse", "HEAD")
    status = run("status", "--short")
    _require(branch == EXPECTED_BRANCH, f"unexpected branch: {branch}")
    _require(head == EXPECTED_HEAD, f"unexpected HEAD: {head}")
    return {
        "branch": branch,
        "head": head,
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _authority(root: Path) -> dict[str, Any]:
    alignment = _read_json(root, ALIGNMENT_CONTRACT, ALIGNMENT_CONTRACT_SHA256, "aligned contract")
    _require(
        alignment.get("classification")
        == "DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY",
        "aligned contract is not READY",
    )
    totality = alignment.get("totality_checks")
    _require(isinstance(totality, dict), "aligned contract totality is missing")
    _require(totality.get("ols_shape") == [100], "aligned contract OLS shape changed")
    for key in ("projection_required", "reemission_required", "same_writer_for_real_and_generated_ols", "coefficient_transmission_required"):
        _require(totality.get(key) is True, f"aligned contract requirement missing: {key}")
    _require(totality.get("whole_field_identity_bypass") is False, "identity bypass reopened")

    runtime_report = _read_json(
        root, ALIGNMENT_RUNTIME_REPORT, ALIGNMENT_RUNTIME_REPORT_SHA256, "aligned runtime report"
    )
    runtime = runtime_report.get("runtime")
    _require(isinstance(runtime, dict), "runtime authority is missing")
    _require(runtime.get("module_sha256") == ALIGNMENT_RUNTIME_SHA256, "runtime source identity changed")
    _require(runtime.get("canonical_ols") == "complex [B,100], 10x10, row-major j=10*r+c", "OLS authority changed")
    _require(runtime.get("transport") == "T=I_100 in coefficient space", "OLS transport changed")
    _require(runtime.get("writer") == "W_OLS(z)=sum_j (Tz)_j v_j", "OLS writer changed")

    v1 = _read_json(root, V1_CONTRACT, V1_CONTRACT_SHA256, "V1 training contract")
    _require(v1.get("result") == "READY", "V1 training contract is not READY")
    _require(v1.get("training_authorized") is False, "V1 contract authorized training")

    diagnosis = _read_json(root, DIAGNOSIS_REPORT, DIAGNOSIS_REPORT_SHA256, "real-path diagnosis")
    _require(
        diagnosis.get("primary_diagnosis") == "REAL_PATH_FAILURE_ENCODER_OLS_SEMANTIC_COLLAPSE_SUPPORTED",
        "diagnosis classification changed",
    )

    post = _read_json(root, POST_TRAINING_CONTRACT, POST_TRAINING_CONTRACT_SHA256, "post-training contract")
    post_text = json.dumps(post, ensure_ascii=False, sort_keys=True)
    _require("clean_development_calibration_macro_recall - 0.05" in post_text, "semantic threshold authority changed")
    _require("FrozenMNISTEvaluator" in post_text, "semantic evaluator authority changed")

    return {
        "alignment_contract": _ref(root, ALIGNMENT_CONTRACT, "approved aligned input/OLS contract", ALIGNMENT_CONTRACT_SHA256),
        "runtime_report": _ref(root, ALIGNMENT_RUNTIME_REPORT, "aligned runtime report", ALIGNMENT_RUNTIME_REPORT_SHA256),
        "runtime": _ref(root, ALIGNMENT_RUNTIME, "aligned LightGen runtime", ALIGNMENT_RUNTIME_SHA256),
        "v1_training_contract": _ref(root, V1_CONTRACT, "historical V1 training contract", V1_CONTRACT_SHA256),
        "diagnosis_report": _ref(root, DIAGNOSIS_REPORT, "semantic-collapse diagnosis", DIAGNOSIS_REPORT_SHA256),
        "post_training_contract": _ref(root, POST_TRAINING_CONTRACT, "semantic threshold authority", POST_TRAINING_CONTRACT_SHA256),
        "evaluator": {
            "implementation": _ref(root, EVALUATOR_IMPLEMENTATION, "FrozenMNISTEvaluator implementation", EVALUATOR_IMPLEMENTATION_SHA256),
            "checkpoint": _ref(root, EVALUATOR_CHECKPOINT, "accepted FrozenMNISTEvaluator checkpoint", EVALUATOR_CHECKPOINT_SHA256),
            "qualification_threshold": SEMANTIC_EVALUATOR_QUALIFICATION_THRESHOLD,
            "validation_accuracy": 0.9875999689102173,
            "view": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
        },
        "semantic_threshold": {
            "value": SEMANTIC_THRESHOLD,
            "source_formula": "max(0.90, clean_development_calibration_macro_recall - 0.05)",
            "source_clean_development_macro_recall": 0.987,
            "role": "development Stage-A transition and Stage-B semantic firewall",
            "frozen_before_v2_execution": True,
            "tuned_from_v2_results": False,
        },
    }


def _parameter_surface() -> dict[str, Any]:
    from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import LightGen2DAlignmentRuntimeConfig, LightGen2DSemanticGenerationInputOLSAlignment

    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=128)
    )
    named = tuple(runtime.named_parameters())
    names = [name for name, _ in named]
    _require(names == ["encoder.phase", "generator.phase"], f"unexpected parameter surface: {names}")
    shapes = {name: list(parameter.shape) for name, parameter in named}
    dtypes = {name: str(parameter.dtype) for name, parameter in named}
    _require(all(parameter.requires_grad for _, parameter in named), "runtime phase parameters are not trainable")
    count = sum(parameter.numel() for _, parameter in named)
    _require(count == 40_000, f"unexpected trainable parameter count: {count}")
    return {
        "exact_trainable_parameters": names,
        "exact_optimizer_parameters": names,
        "parameter_shapes": shapes,
        "parameter_dtypes": dtypes,
        "parameter_count": count,
        "surface_probe": "runtime construction at grid=128; no forward, optimizer, backward, CUDA, or data",
    }


def _payload(root: Path, git: Mapping[str, Any], authority: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 2,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definition_only": True,
        "architecture_decision": "LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST",
        "approved_r0_transcription": True,
        "git": dict(git),
        "source_authority": dict(authority),
        "architecture": {
            "scope": "2D semantic image generation only",
            "immutable_chain": [
                "X -> explicit SLM amplitude mapping -> illuminated complex field -> coherent relay",
                "coherent relay -> trainable optical encoder -> mandatory 10x10 single-mode modal projection",
                "raw physical z in C^100 -> coefficient-space T=I_100 -> mandatory W_OLS",
                "W_OLS -> trainable photonic generator -> raw CMOS intensity",
            ],
            "immutable_components": [
                "input semantics and PlaneSpec",
                "FiberModeBank, modal normalization, and row-major port order j=10*r+c",
                "10x10/100-port OLS topology, T=I_100, W_OLS, generator entrance plane, and detector",
            ],
            "same_real_and_generated_interface": True,
            "whole_field_identity_bypass": False,
        },
        "trainable_parameter_contract": {
            **dict(surface),
            "forbidden_trainables": [
                "OLS modal banks or T",
                "input/output relay and SLM mapping",
                "q/amplitude/phase producer or phase repair",
                "label+8D controller, electronic decoder, rank-16 readout, or prior parameters",
            ],
        },
        "semantic_evaluator_contract": {
            "identity": "FrozenMNISTEvaluator v1",
            "authority": authority["evaluator"],
            "preprocessing": authority["evaluator"]["view"],
            "threshold": authority["semantic_threshold"],
            "gradient_enabled": False,
            "labels_as_model_input": False,
        },
        "data_contract": {
            "dataset": "local MNIST, download=False",
            "source_artifacts": {
                "training": {
                    "path": _relative(root, root / TRAINING_DATA),
                    "sha256": TRAINING_DATA_SHA256,
                    "read_by_definition": False,
                    "read_by_future_training": True,
                },
                "protected_test": {
                    "path": _relative(root, root / PROTECTED_DATA),
                    "sha256": PROTECTED_DATA_SHA256,
                    "read_by_definition": False,
                    "read_by_future_training": False,
                },
            },
            "split": {
                "algorithm": "stratified_mnist_indices(labels, seed=1234, validation_per_label=500)",
                "seed": SPLIT_SEED,
                "train_count": 55_000,
                "development_count": 5_000,
                "development_selection": {
                    "count": 1_000,
                    "per_class": 100,
                    "seed": DEVELOPMENT_SELECTION_SEED,
                    "use": "Stage-A transition, Stage-B firewall, and frozen checkpoint selection only",
                },
            },
            "roles": {
                "train": {
                    "count": 55_000,
                    "use": "optimization and Stage-A scale extraction only",
                },
                "development": {
                    "count": 5_000,
                    "selection_count": 1_000,
                    "use": "semantic transition/firewall and checkpoint selection only",
                },
                "calibration": {
                    "count": 4_000,
                    "use": "not consumed; prior authority only",
                },
                "protected": {"count": 10_000, "use": "forbidden"},
                "test": {"use": "forbidden"},
                "sealed": {"use": "forbidden; consumed historical N3 evidence unavailable"},
            },
            "forbidden_roles_fail_closed": True,
        },
        "state_machine": {
            "states": [
                "STAGE_A_TRAIN",
                "STAGE_A_SEMANTIC_CHECK",
                "STAGE_A_SELECT",
                "SCALE_EXTRACT",
                "SCALE_FREEZE",
                "STAGE_B_TRAIN",
                "STAGE_B_SEMANTIC_CHECK",
                "FINAL_SELECT",
                "COMPLETE",
            ],
            "transitions": [
                "STAGE_A_TRAIN -> STAGE_A_SEMANTIC_CHECK -> STAGE_A_TRAIN",
                "STAGE_A_SEMANTIC_CHECK -> STAGE_A_SELECT at final checkpoint",
                "STAGE_A_SELECT -> SCALE_EXTRACT -> SCALE_FREEZE -> STAGE_B_TRAIN",
                "STAGE_B_TRAIN -> STAGE_B_SEMANTIC_CHECK -> STAGE_B_TRAIN",
                "STAGE_B_SEMANTIC_CHECK -> FINAL_SELECT at final checkpoint",
                "FINAL_SELECT -> COMPLETE only with a semantic-eligible checkpoint",
            ],
            "terminal_failures": {
                "stage_a": "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED",
                "stage_b": "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_PRESERVATION_NOT_ESTABLISHED",
            },
        },
        "stage_a": {
            "purpose": "semantic complex-OLS warmup before any strong prior pressure",
            "steps": STAGE_A_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "loss": {
                "formula": "L_A = 1.0*L_image + 0.0*L_optical",
                "alpha_latent": 0.0,
                "beta_image": 1.0,
                "gamma_optical": 0.0,
                "latent_mmd_enabled": False,
                "image": "mean((I_cmos_raw - clamp(BilinearResize(X,200x200),0,1))^2)",
                "optical": "V1 optical objective remains explicitly omitted; no scale anchor before Stage-B",
            },
            "semantic_check": {
                "role": "fixed development checkpoint-selection subset, 100/class",
                "macro_recall_threshold": SEMANTIC_THRESHOLD,
                "evaluator": "FrozenMNISTEvaluator v1 with frozen raw-CMOS view",
                "failure": "stop before scale extraction and Stage B",
            },
            "selection": {
                "eligible_if": "development macro recall >= 0.937",
                "rule": "minimum development L_image among eligible checkpoints",
                "tie_break": "earlier stage-local checkpoint",
                "protected_or_generated_inputs": False,
            },
        },
        "scale_contract": {
            "authority": "train-only raw physical OLS from selected Stage-A checkpoint",
            "formula": "s_j = mean_{i in train} |z_{i,j}|^2, j=0..99",
            "port_count": OLS_PORT_COUNT,
            "port_order": "row-major j=10*r+c",
            "epsilon_scale": EPSILON_SCALE,
            "dtype": "raw z complex64; per-port power accumulated in float64 in canonical train-row order",
            "train_count": 55_000,
            "batch_size": LOGICAL_BATCH_SIZE,
            "requirements": ["finite", "exactly 100 entries", "every s_j > epsilon_scale", "no clipping or rescaling"],
            "artifact_schema": "lightgen_2d_semantic_complex_ols_training_scale.v2",
            "artifact_identity": "bind selected Stage-A checkpoint, V2 contract, aligned runtime, train source, port order, epsilon, dtype, accumulation, and vector SHA",
            "stage_b_frozen": True,
        },
        "normalized_latent_contract": {
            "authoritative_physical_state": "z in C^100",
            "formula": "z_tilde_j = z_j / sqrt(s_j + epsilon_scale)",
            "role": "derived statistical coordinate only; never passed to W_OLS as physical z",
            "gauge": {
                "formula": "G(a)=a*conj(sum_j a_j/sqrt(100))/sqrt(|sum_j a_j/sqrt(100)|^2+epsilon_g^2)",
                "epsilon_g": GAUGE_EPSILON,
                "applied_to": "normalized full-complex statistics only",
                "relative_phase_preserved": True,
            },
            "prior": {
                "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NORMALIZED_OLS_V2",
                "formula": "z_tilde_prior,j=(epsilon_R+i*epsilon_I)/sqrt(2)",
                "real_imag_distribution": "epsilon_R,epsilon_I iid Normal(0,1); real/imag variance=1/2; E|z_j|^2=1",
                "shape": [LOGICAL_BATCH_SIZE, OLS_PORT_COUNT],
                "dtype": "torch.complex64",
                "namespace": PRIOR_NAMESPACE,
                "seed_derivation": "SHA256(namespace|stage|role|stage_local_step|sample_count), first 8 bytes reduced to signed 63-bit seed",
            },
            "mmd": {
                "name": "FULL_COMPLEX_NORMALIZED_GAUSSIAN_KERNEL_MMD_VSTAT_V2",
                "embedding": "concat(Re(G(z_tilde)), Im(G(z_tilde))) in R^200",
                "kernel": "exp(-||a-b||^2/(2*l^2)), l^2=200",
                "reduction": "biased B^2 V-statistic including diagonals",
                "q_only": False,
                "raw_z_direct_mmd": False,
            },
        },
        "stage_b": {
            "purpose": "semantic-preserving normalized full-complex distribution shaping",
            "steps": STAGE_B_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "initialization": "selected semantic-eligible Stage-A checkpoint parameters",
            "optimizer_reset": True,
            "loss": {
                "formula": "L_B = 1.0*L_image + 1.0*L_latent_normalized + 1.0*L_scale",
                "alpha_latent": 1.0,
                "beta_image": 1.0,
                "gamma_optical": 1.0,
                "optical_definition": "L_optical := L_scale",
                "scale_anchor": "mean_j[log((mean_batch |z_j|^2 + epsilon_scale)/(s_j + epsilon_scale))]^2",
                "one_scale_penalty_only": True,
            },
            "semantic_firewall": {
                "eligible_if": "development macro recall >= 0.937",
                "cannot_be_compensated": True,
                "failure": "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_PRESERVATION_NOT_ESTABLISHED",
            },
            "selection": {
                "rule": "among semantic-eligible checkpoints, minimum frozen development L_B",
                "tie_break": "earlier stage-local checkpoint",
                "ineligible_checkpoints_can_win": False,
            },
        },
        "future_generation": {
            "conceptual_only": True,
            "normalized_sampling": "sample z_tilde_sample ~ CN(0,I_100) with the frozen V2 RNG convention",
            "physical_mapping": "z_sample,j=sqrt(s_j+epsilon_scale)*z_tilde_sample,j",
            "path": "z_sample -> ComplexOLSState -> forward_from_ols(z_sample) -> same T -> same W_OLS -> same generator",
            "gauge_before_w_ols": False,
            "image_encoder_required": False,
            "phase_repair_required": False,
            "execution_in_this_task": False,
        },
        "optimizer_and_precision": {
            "optimizer": {
                "name": "Adam",
                "learning_rate": LEARNING_RATE,
                "betas": list(ADAM_BETAS),
                "eps": ADAM_EPSILON,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "schedule_seed": SPLIT_SEED,
            "precision": {"phase": "float32", "field": "complex64", "autocast": False, "grad_scaler": False},
            "device": "cuda required for future 300+300 scientific training; no CUDA training in definition/materialization",
            "gradient_clipping": None,
            "nonfinite_policy": "fail closed before update and write no accepted checkpoint",
            "stage_b_optimizer_state": "reset to a fresh Adam state at the selected Stage-A checkpoint",
        },
        "checkpoint_contract": {
            "schema": "lightgen_2d_semantic_complex_ols_training_checkpoint.v2",
            "required": [
                "V2 contract identity",
                "aligned runtime identity",
                "stage and global/stage-local steps",
                "encoder.phase and generator.phase",
                "optimizer state",
                "CPU/CUDA RNG state",
                "train sampler state",
                "semantic evaluator/preprocessing identity and 0.937 threshold",
                "checkpoint selection state",
            ],
            "stage_b_additional": [
                "selected Stage-A checkpoint identity",
                "frozen scale artifact identity and SHA",
                "exact s vector",
                "epsilon_scale",
                "normalized prior identity",
                "optimizer_reset_marker",
            ],
            "resume": [
                "inside Stage A",
                "after Stage-A selection",
                "at scale extraction/freeze boundary",
                "inside Stage B",
            ],
            "reject_on": ["contract/runtime/scale identity mismatch", "forbidden data role", "Stage-A optimizer restored into Stage B"],
        },
        "claim_boundary": {
            "future_success_means": "a bounded V2 run produced a development-semantic-eligible checkpoint while optimizing the normalized full-complex OLS objective under a frozen train-only physical scale",
            "warmup_failure_means": "semantic warmup was not established; Stage B did not run and no conclusion about the prior is permitted",
            "stage_b_failure_means": "the frozen normalized distribution-shaping configuration did not preserve development semantics after Stage A",
            "does_not_prove": [
                "direct semantic generation",
                "protected/test generalization",
                "hardware or exact Chen reproduction",
                "generation acceptance, formal promotion, or generation readiness",
            ],
        },
        "legacy_exclusion": [
            "historical Gaussian q producer",
            "independent phase inheritance",
            "q-conditioned nearest phase donor",
            "q-conditioned complex-phasor residual",
            "label+8D controller and whole-field IdentityOLS",
            "electronic rank-16 reconstruction and software VAE decoder",
        ],
        "outer_gate": {
            "passed": True,
            "answer": "YES",
            "checks": {
                "aligned_runtime_unchanged": True,
                "exact_trainable_scope": True,
                "stage_a_no_mmd": True,
                "semantic_checkpoint_firewall": True,
                "train_only_scale": True,
                "normalized_full_complex_prior": True,
                "same_scale_future_sampling": True,
                "optimizer_reset_reproducible": True,
                "checkpoint_resume_boundary_defined": True,
            },
            "hard_stop_conditions": {
                "approved_semantics_unresolved": False,
                "runtime_change_required": False,
                "evaluator_or_threshold_unresolved": False,
                "selection_or_cadence_unresolved": False,
                "scale_undefined": False,
                "protected_or_sealed_required": False,
            },
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "scientific_training_executed": False,
            "protected_or_sealed_accessed": False,
        },
        "next_action": "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2",
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "definition_consumer": {
            "path": _relative(root, Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "role": "V2 contract materialization consumer",
        },
        "artifact_self_hash": "not_embedded_to_avoid_self_reference",
    }


def define(repo_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = _root() if repo_root is None else repo_root.resolve()
    output_dir = root / OUTPUT_DIR
    output_path = output_dir / CONTRACT_NAME
    _require(not output_dir.exists(), "V2 contract output already exists; refusing overwrite")
    git = _git(root)
    authority = _authority(root)
    surface = _parameter_surface()
    payload = _payload(root, git, authority, surface)
    atomic_write_new_json(output_path, payload, artifact_kind="LightGen V2 training contract")
    return output_path, payload


def main() -> int:
    path, payload = define()
    print(json.dumps({"result": payload["result"], "classification": payload["classification"], "path": str(path), "sha256": file_sha256(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
