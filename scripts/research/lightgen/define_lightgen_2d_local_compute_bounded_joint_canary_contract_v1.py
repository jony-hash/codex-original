"""Define the bounded local-compute joint Canary after the stale HEAD repair.

This definition consumer records the accepted step-9000 inputs and a fixed
300-update Stage-1 contract.  It performs only immutable identity/schema
checks; it does not load models, run a forward pass, profile CUDA, or create a
scientific attempt.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import git_identity
from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, safe_torch_load
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as joint


TASK_ID = "DEFINE_LIGHTGEN_2D_LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_CONTRACT_V1"
CONTRACT_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_contract.v1"
CLASSIFICATION = "LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_CONTRACT_DEFINED"
OUTER_GATE = "LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_CONTRACT_DEFINED"
PROPOSED_AUTHORIZATION = "AUTHORIZED_EXECUTE_LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_V1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_CONSUMER_HEAD = "500045d6a92843c81ffeda010e01351f559e453b"
HISTORICAL_PRODUCER_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

START_CHECKPOINT = joint.START_CHECKPOINT
START_CHECKPOINT_SHA256 = joint.START_CHECKPOINT_SHA256.lower()
ATTRIBUTION_REPORT = joint.ATTRIBUTION_REPORT
ATTRIBUTION_REPORT_SHA256 = joint.ATTRIBUTION_REPORT_SHA256.lower()
ATTRIBUTION_GRADIENTS = joint.ATTRIBUTION_GRADIENTS
ATTRIBUTION_GRADIENTS_SHA256 = joint.ATTRIBUTION_GRADIENTS_SHA256.lower()
REUSED_STEP9000_DEV = joint.REUSED_STEP9000_DEV
REUSED_STEP9000_DEV_SHA256 = joint.REUSED_STEP9000_DEV_SHA256.lower()
REUSED_STEP9000_S1 = joint.REUSED_STEP9000_S1
REUSED_STEP9000_S1_SHA256 = joint.REUSED_STEP9000_S1_SHA256.lower()
ACCEPTED_STRUCTURAL_REPORT = joint.ACCEPTED_STRUCTURAL_REPORT
ACCEPTED_STRUCTURAL_REPORT_SHA256 = joint.ACCEPTED_STRUCTURAL_REPORT_SHA256.lower()
STRUCTURAL_REFERENCE = joint.STRUCTURAL_REFERENCE
STRUCTURAL_REFERENCE_SHA256 = joint.STRUCTURAL_REFERENCE_SHA256.lower()
SURROGATE_NORMALIZATION = joint.SURROGATE_NORMALIZATION
SURROGATE_NORMALIZATION_SHA256 = joint.SURROGATE_NORMALIZATION_SHA256.lower()
S1_WINNER = joint.S1_WINNER
S1_WINNER_SHA256 = joint.S1_WINNER_SHA256.lower()

SCIENTIFIC_ATTEMPT_DIR = joint.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260922_local_compute_bounded_joint_canary_v1/runs/contract_definition_07"
)
CONTRACT_NAME = "lightgen_2d_local_compute_bounded_joint_canary_contract.v1.json"

START_STEP = 9000
MAX_UPDATES = 300
PRIMARY_ENDPOINT = 9300
CHECKPOINT_STEPS = (9100, 9200, 9300)
LOGICAL_BATCH_SIZE = 10
GRADIENT_ENGINE = "SCALAR_REFERENCE_VJP"
EQUIVALENT_BATCHED_ENGINE = "HYBRID_SEMANTIC_BATCHED_VJP"
GRADIENT_RTOL = 2.0e-5
GRADIENT_ATOL = 2.0e-7
DELTA_JOINT = joint.DELTA_JOINT
BETA = joint.BETA
GAMMA = joint.GAMMA
EPS_PROJ = joint.EPS_PROJ


class ContractError(RuntimeError):
    """A bounded Canary authority cannot be defined safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> Any:
    _require(hasattr(value, "shape") and hasattr(value, "dtype"), f"{name} must be a tensor")
    return value


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"{value} must be a repository-relative path")
    return (root / relative).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, relative: str | Path, expected_sha256: str, role: str) -> dict[str, Any]:
    target = _path(root, relative)
    _require(target.is_file(), f"{role} is missing: {target}")
    actual = file_sha256(target)
    _require(actual == expected_sha256.lower(), f"{role} hash changed")
    return {"path": _relative(root, target), "sha256": expected_sha256.lower(), "role": role}


def _load_start_checkpoint(root: Path) -> dict[str, Any]:
    checkpoint = _mapping(
        safe_torch_load(
            _path(root, START_CHECKPOINT),
            expected_sha256=START_CHECKPOINT_SHA256,
            map_location="cpu",
        ),
        "accepted step9000 checkpoint",
    )
    _require(checkpoint.get("schema") == "evaluator_agnostic_structural_image_alignment_checkpoint.v1", "accepted checkpoint schema changed")
    _require(checkpoint.get("completed_updates") == START_STEP, "accepted checkpoint cursor changed")
    _require(checkpoint.get("scientific_artifact") is True and checkpoint.get("operational_artifact") is False, "accepted checkpoint role changed")
    _require(checkpoint.get("trainable_parameter_names") == ["generator.phase"], "accepted trainable boundary changed")
    _require(
        checkpoint.get("frozen_parameter_names") == [
            "encoder.phase",
            "OLS",
            "W_OLS",
            "scale",
            "T=I_100",
            "S1",
            "E_S",
            "E_I",
            "E_C",
            "E_D",
            "f_struct",
        ],
        "accepted frozen boundary changed",
    )
    for name in ("encoder_state", "generator_state"):
        value = _tensor(checkpoint.get(name), name)
        _require(tuple(value.shape) == (2, 1, 100, 100), f"accepted {name} geometry changed")
        _require(bool(value.is_floating_point()), f"accepted {name} dtype changed")
    optimizer = _mapping(checkpoint.get("optimizer_configuration"), "accepted Adam configuration")
    _require(optimizer.get("name") == "Adam", "accepted optimizer changed")
    _require(optimizer.get("parameter_names") == ["generator.phase"], "accepted optimizer parameter boundary changed")
    _require(float(optimizer.get("learning_rate", -1.0)) == 1.0e-3, "accepted learning rate changed")
    _require(list(optimizer.get("betas", [])) == [0.9, 0.999], "accepted Adam betas changed")
    _require(float(optimizer.get("eps", -1.0)) == 1.0e-8 and float(optimizer.get("weight_decay", -1.0)) == 0.0, "accepted Adam options changed")
    _require(optimizer.get("amsgrad") is False and float(optimizer.get("internal_step", -1.0)) == 9300.0, "accepted Adam state cursor changed")
    _require(len(_mapping(checkpoint.get("optimizer_state"), "accepted optimizer state").get("state", {})) == 1, "accepted Adam moments are missing")
    sampler = _mapping(checkpoint.get("sampler_state"), "accepted sampler state")
    _require(sampler.get("seed") == 1234 and sampler.get("cursor") == START_STEP and sampler.get("steps") == START_STEP, "accepted sampler cursor changed")
    _require(sampler.get("wraparound") is True, "accepted sampler wraparound changed")
    rng = _mapping(checkpoint.get("rng_state"), "accepted RNG state")
    _tensor(rng.get("cpu"), "accepted CPU RNG")
    _require(isinstance(rng.get("cuda"), list) and len(rng["cuda"]) > 0, "accepted CUDA RNG state is missing")
    return dict(checkpoint)


def _load_attribution_authorities(root: Path) -> dict[str, Any]:
    report_path = _path(root, ATTRIBUTION_REPORT)
    _require(file_sha256(report_path) == ATTRIBUTION_REPORT_SHA256, "step9000 attribution report hash changed")
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "step9000 attribution report")
    _require(report.get("result") == "VALID_COMPLETED", "step9000 attribution is not complete")
    _require(report.get("outer_gate") == "STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_ATTRIBUTED", "step9000 attribution gate changed")
    _require(report.get("classification") == "STRUCTURAL_VS_CLASSIFIER_OBJECTIVE_CONFLICT_SUPPORTED", "step9000 attribution classification changed")
    gradients = _mapping(
        safe_torch_load(_path(root, ATTRIBUTION_GRADIENTS), expected_sha256=ATTRIBUTION_GRADIENTS_SHA256, map_location="cpu"),
        "step9000 attribution gradients",
    )
    key_steps = _mapping(gradients.get("key_steps"), "attribution key steps")
    _require("9000" in key_steps, "step9000 attribution gradient is missing")
    return {"report": dict(report), "gradients": dict(gradients)}


def _git_contract(root: Path) -> dict[str, Any]:
    current = git_identity(root)
    binding = robust._bind_git_provenance(current)
    _require(binding["producer"]["head"] == HISTORICAL_PRODUCER_HEAD, "historical producer HEAD changed")
    return {
        "producer": binding["producer"],
        "consumer": {**binding["consumer"], "worktree_dirty": current["worktree_dirty"], "status_sha256": current["status_sha256"]},
        "consumer_head_exact_required": False,
        "required_consumer_head": None,
        "head_equal": binding["head_equal"],
        "producer_head_is_provenance_only": True,
        "semantics": "historical producer and consumer Git fields are provenance only",
    }


def build_contract(root: Path) -> dict[str, Any]:
    root = root.resolve()
    git = _git_contract(root)
    checkpoint = _load_start_checkpoint(root)
    attribution = _load_attribution_authorities(root)
    _require(not _path(root, SCIENTIFIC_ATTEMPT_DIR).exists(), "scientific attempt_01 already exists")
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
        "authorities": {
            "accepted_step9000_checkpoint": _artifact_ref(root, START_CHECKPOINT, START_CHECKPOINT_SHA256, "accepted step9000 checkpoint"),
            "step9000_attribution_report": _artifact_ref(root, ATTRIBUTION_REPORT, ATTRIBUTION_REPORT_SHA256, "step9000 attribution report"),
            "step9000_attribution_gradients": _artifact_ref(root, ATTRIBUTION_GRADIENTS, ATTRIBUTION_GRADIENTS_SHA256, "step9000 attribution gradients"),
            "accepted_structural_report": _artifact_ref(root, ACCEPTED_STRUCTURAL_REPORT, ACCEPTED_STRUCTURAL_REPORT_SHA256, "accepted structural report"),
            "frozen_structural_reference_q95": _artifact_ref(root, STRUCTURAL_REFERENCE, STRUCTURAL_REFERENCE_SHA256, "frozen structural q95 reference"),
            "frozen_surrogate_normalization": _artifact_ref(root, SURROGATE_NORMALIZATION, SURROGATE_NORMALIZATION_SHA256, "frozen structural surrogate normalization"),
            "frozen_s1_winner": _artifact_ref(root, S1_WINNER, S1_WINNER_SHA256, "frozen S1 winner"),
            "reused_step9000_dev": _artifact_ref(root, REUSED_STEP9000_DEV, REUSED_STEP9000_DEV_SHA256, "reused step9000 DEV baseline"),
            "reused_step9000_s1": _artifact_ref(root, REUSED_STEP9000_S1, REUSED_STEP9000_S1_SHA256, "reused step9000 S1 baseline"),
            "checkpoint_validation": {
                "schema": checkpoint["schema"],
                "completed_updates": checkpoint["completed_updates"],
                "optimizer_internal_step": checkpoint["optimizer_configuration"]["internal_step"],
                "sampler_cursor": checkpoint["sampler_state"]["cursor"],
                "rng_restored": True,
            },
            "attribution_validation": {
                "result": attribution["report"]["result"],
                "outer_gate": attribution["report"]["outer_gate"],
                "gradient_key_steps": ["9000"],
            },
        },
        "protocol": {
            "start_step": START_STEP,
            "max_updates": MAX_UPDATES,
            "primary_endpoint": PRIMARY_ENDPOINT,
            "checkpoints": list(CHECKPOINT_STEPS),
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "losses_unchanged": ["L_struct", "L_S", "L_I", "L_C"],
            "joint_normalization_unchanged": {"delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "formula": "existing accepted joint normalization and auxiliary projection"},
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "restore_step9000_state": True},
            "sampler_and_data_order_unchanged": True,
            "step9046_recovery_consumed": False,
            "stage2": {"executed": False, "auto_continuation": False, "decision_after": PRIMARY_ENDPOINT},
        },
        "gradient_backend": {
            "selected": GRADIENT_ENGINE,
            "reason": "The existing scalar reference path is the lowest-cost measured mathematically equivalent backend for this bounded Canary; the prior batched +10% gate is not carried forward.",
            "equivalent_batched_engine": EQUIVALENT_BATCHED_ENGINE,
            "engineering_measurement_basis": {"scalar_median_seconds": 0.43308600000091, "batched_median_seconds": 0.45924589999959, "historical_speedup_fraction": -0.0604, "scientific_evidence": False, "role": "historical implementation diagnostic only"},
            "equivalence_basis": {"semantic_rows": ["L_S->g_S", "L_I->g_I", "L_C->g_C"], "comparison": "scalar reference versus one is_grads_batched=True three-row mapping", "rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "finite_validation": "all three rows fail closed together"},
            "performance_gate": "none beyond technical executability and exact equivalence; no +10% speedup requirement",
        },
        "acceptance": {
            "primary": {"endpoint": PRIMARY_ENDPOINT, "structural": {"metric": "macro_log_rho", "improvement": "macro_log_rho(9000)-macro_log_rho(9300) > 0", "paired_bootstrap_ci95_lower": "> 0"}, "semantic": {"L_S_9300": "<= L_S_9000", "L_I_9300": "<= L_I_9000", "L_C_9300": "<= L_C_9000"}},
            "additional": ["no severe per-class structural reversal", "joint direction nondegenerate", "reconstruction pass", "optical validity pass", "protected/test/sealed reads are zero"],
            "secondary_descriptive_only": ["SSR_struct", "intermediate checkpoint diagnostics"],
            "checkpoint_selection": "9300 is the fixed primary endpoint; 9100 and 9200 cannot replace it",
        },
        "fail_fast": {"stop_at": [9100, 9200, 9300], "safety_or_validity_failures": ["nonfinite", "joint direction degeneracy", "severe reconstruction or optical failure", "frozen-state violation", "data-firewall violation", "catastrophic semantic reversal"], "early_stop_cannot_declare_success": True, "always_stop_after": 9300},
        "execution_boundary": {"scientific_training_executed": False, "scientific_attempt_created": False, "attempt_path": SCIENTIFIC_ATTEMPT_DIR.as_posix(), "cuda_or_profiler_executed": False, "model_selection": False, "promotion": False, "proposed_authorization": PROPOSED_AUTHORIZATION},
        "claim_boundary": {"bounded_canary_only": True, "stage2_not_authorized": True, "no_checkpoint_promotion": True, "no_hardware_readiness": True, "new_user_decision_required_after_9300": True},
    }


def write_contract(root: Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require(not target.exists(), f"contract output already exists: {target}")
    payload = build_contract(root)
    target.mkdir(parents=True, exist_ok=False)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="local bounded joint Canary contract")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    path = write_contract(args.repo_root.resolve(), args.output_dir)
    print(json.dumps({"result": "READY", "outer_gate": OUTER_GATE, "contract": path.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
