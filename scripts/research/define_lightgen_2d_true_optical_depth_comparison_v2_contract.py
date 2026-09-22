"""Materialize the independent successor contract for N5 V2.

N5 V2 preserves the frozen V1 scientific design while giving it a fresh
successor authority and fresh output namespace.  The invalid V1 attempt is
recorded as historical provenance only; none of its unpersisted observations
are an input to this contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2_CONTRACT"
SCHEMA = "lightgen_2d_true_optical_depth_comparison_v2_contract.v1"
CLASSIFICATION = "N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_READY"
OUTER_GATE = "N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_RESOLVED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

V1_CONTRACT = Path(
    "outputs/20260914_n5/runs/"
    "lightgen_2d_true_optical_depth_comparison_contract_v1_attempt_01/"
    "lightgen_2d_true_optical_depth_comparison_contract.v1.json"
)
V1_CONTRACT_SHA256 = "8d305b59216ec5ccccc6af50d83dce5d30d40c231bca1e50da104ebb0406cb6d"
V1_A_REPORT = Path(
    "outputs/20260914_n5/runs/"
    "lightgen_2d_true_optical_depth_comparison_candidate_a_attempt_01/"
    "lightgen_2d_true_optical_depth_comparison_a_continued_c1_scientific_report.v1.json"
)
V1_A_REPORT_SHA256 = "35ba83b24f72b79f2a4394324f1cff33f6f356c9252de6f85390df517c8bf51c"
V1_A_LINEAGE = Path(
    "outputs/20260914_n5/runs/"
    "lightgen_2d_true_optical_depth_comparison_candidate_a_attempt_01/"
    "lightgen_2d_true_optical_depth_comparison_a_continued_c1_scientific_lineage.v1.json"
)
V1_A_LINEAGE_SHA256 = "3aea6062ae87b01569daf20d3d1986d8babaa61767393dabcf182257f2b5e2ab"
N4_ATTRIBUTION_REPORT = Path(
    "outputs/20260914_n4_endpoint_attribution/runs/"
    "lightgen_2d_n4_endpoint_attribution_evaluation_v1_attempt_01/"
    "lightgen_2d_n4_endpoint_attribution_report.v1.json"
)
N4_ATTRIBUTION_REPORT_SHA256 = "18cef70e3ad1a2a7f81144663f6e55373d08265c4d387d2e39df2aec0c4a3388"
C1_CHECKPOINT = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_step_300_checkpoint.v1.pt"
)
C1_CHECKPOINT_SHA256 = "d7a5581df3fb460337e8655865a200886739cdeab0fb6074213cc4b2cc66c6e9"
V11_REPORT = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1.json"
)
V11_REPORT_SHA256 = "bc603b7a3b7ef1590bc354cb0b62464f3ececcb2390834d864d58bfbe9158184"
N3_AGGREGATE = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_aggregate_v1_attempt_01/n3_low_rank_model_order_aggregate.v1.json"
)
N3_AGGREGATE_SHA256 = "1b6bd6af18462f717fa19532e1a981e872a4e48fe6498ed561890f6570ae89de"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260914_n5_v2/runs/"
    "lightgen_2d_true_optical_depth_comparison_v2_contract_v1_attempt_01"
)
CONTRACT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_contract.v1.json"


class ContractBlocked(RuntimeError):
    """The successor contract cannot be made valid from local authorities."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractBlocked(message)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractBlocked(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ContractBlocked(f"Git state cannot be resolved: {args}") from error
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def _ref(root: Path, path: Path, expected: str, name: str) -> dict[str, str]:
    target = (root / path).resolve()
    _require(target.is_file(), f"{name} is unavailable: {target}")
    observed = file_sha256(target)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")
    return {"path": path.as_posix(), "sha256": observed}


def _validate_inputs(root: Path) -> dict[str, Any]:
    git = _git_state(root)
    _require(git["branch"] == EXPECTED_BRANCH, f"unexpected branch: {git['branch']}")
    _require(git["head"] == EXPECTED_HEAD, f"unexpected HEAD: {git['head']}")
    v1_ref = _ref(root, V1_CONTRACT, V1_CONTRACT_SHA256, "V1 design contract")
    v1 = _read_json((root / V1_CONTRACT).resolve(), "V1 design contract")
    _require(v1.get("result") == "READY", "V1 design contract is not READY")
    _require(v1.get("classification") == "N5_TRUE_OPTICAL_DEPTH_COMPARISON_READY", "V1 design classification changed")
    _require(v1.get("outer_gate") == "N5_TRUE_OPTICAL_DEPTH_COMPARISON_READY", "V1 design gate changed")
    _require(v1.get("common_execution", {}).get("additional_updates") == 300, "V1 update budget changed")
    _require(v1.get("common_execution", {}).get("batch") == 10, "V1 batch changed")
    _require(v1.get("common_objective", {}).get("tuning") is False, "V1 objective tuning boundary changed")
    _require(v1.get("stage_a", {}).get("firewall") == "R300>=0.937", "Stage-A firewall changed")
    n4_ref = _ref(root, N4_ATTRIBUTION_REPORT, N4_ATTRIBUTION_REPORT_SHA256, "N4 attribution report")
    c1_ref = _ref(root, C1_CHECKPOINT, C1_CHECKPOINT_SHA256, "C1 step300 checkpoint")
    v11_ref = _ref(root, V11_REPORT, V11_REPORT_SHA256, "V11 report")
    n3_ref = _ref(root, N3_AGGREGATE, N3_AGGREGATE_SHA256, "N3 aggregate")
    v1_a_ref = _ref(root, V1_A_REPORT, V1_A_REPORT_SHA256, "historical V1 A report")
    v1_lineage_ref = _ref(root, V1_A_LINEAGE, V1_A_LINEAGE_SHA256, "historical V1 A lineage")
    v1_a = _read_json((root / V1_A_REPORT).resolve(), "historical V1 A report")
    _require(v1_a.get("result") == "INVALID", "historical V1 A result changed")
    _require(v1_a.get("classification") == "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "historical V1 A classification changed")
    _require(v1_a.get("claim_boundary", {}).get("scientific_training_executed") is True, "historical V1 claim boundary changed")
    _require(v1_a.get("claim_boundary", {}).get("scientific_updates") == 0, "historical V1 update boundary changed")
    return {
        "git": git,
        "v1_contract": v1_ref,
        "n4_endpoint_attribution": n4_ref,
        "c1_step300_checkpoint": c1_ref,
        "v11": v11_ref,
        "n3": n3_ref,
        "historical_v1_attempt": {
            "result": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR",
            "report": v1_a_ref,
            "lineage": v1_lineage_ref,
            "scientific_observations_consumed": False,
            "scientific_observations_reused": False,
            "scientific_updates": 0,
            "retry_or_continuation": False,
        },
    }


def _candidate_definitions() -> dict[str, Any]:
    adapter = 4 * 100 * 9
    phase = 2 * 100 * 100
    depth = 100 * 100
    names = [
        "adapter.u_real",
        "adapter.u_imag",
        "adapter.v_real",
        "adapter.v_imag",
        "generator.phase",
    ]
    return {
        "A_CONTINUED_C1": {
            "label": "CONTINUED_C1",
            "architecture": "rank9 OLS interface + existing 2-layer generator",
            "initialization": "exact frozen C1 step300 checkpoint; restore shared adapter/generator, optimizer and RNG state",
            "trainable": names,
            "frozen": ["encoder", "OLS", "W_OLS", "Smooth v2.1", "prototypes", "templates"],
            "parameter_counts": {"adapter": adapter, "existing_generator_phase": phase, "total_trainable": adapter + phase},
        },
        "B_TRUE_DEPTH_C1": {
            "label": "TRUE_DEPTH_C1",
            "architecture": "rank9 OLS interface + existing 2-layer generator with the second propagation interval factorized by a separated midpoint phase plane",
            "initialization": "same exact C1 step300 checkpoint; shared parameters identical to A and depth_phase=0",
            "trainable": names + ["generator.depth_phase"],
            "frozen": ["encoder", "OLS", "W_OLS", "Smooth v2.1", "prototypes", "templates"],
            "midpoint_phase_shape": [1, 1, 100, 100],
            "midpoint_phase_support": "central active support embedded in full 2048x2048 propagation grid",
            "parameter_counts": {"adapter": adapter, "existing_generator_phase": phase, "propagation_separated_depth_phase": depth, "total_trainable": adapter + phase + depth},
        },
    }


def _payload(root: Path, authorities: dict[str, Any]) -> dict[str, Any]:
    definition_path = Path(__file__).resolve().relative_to(root.resolve()).as_posix()
    definition_sha = file_sha256(Path(__file__).resolve())
    return {
        "schema": SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "READY",
        "status": "definition_ready",
        "classification": CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "predecessor": {
            "task_id": "EXECUTE_N5_TRUE_OPTICAL_DEPTH_COMPARISON_V1",
            "status": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR",
            "scientific_metrics": "not available and not consumed",
            "observations_reused": False,
            "report": authorities["historical_v1_attempt"]["report"],
            "lineage": authorities["historical_v1_attempt"]["lineage"],
        },
        "input_authorities": {
            "predecessor_design_contract": authorities["v1_contract"],
            "n4_endpoint_attribution": authorities["n4_endpoint_attribution"],
            "c1_step300_frozen_endpoint": authorities["c1_step300_checkpoint"],
            "v11_baseline": authorities["v11"],
            "n3_model_order": authorities["n3"],
            "propagation_equivalence": "the persisted V1 contract synthetic_preflight only; no V1 scientific observation",
        },
        "definition_entrypoint": {"path": definition_path, "sha256": definition_sha},
        "candidates": _candidate_definitions(),
        "common_execution": {
            "additional_updates": 300,
            "batch": 10,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "same_rng_convention": True,
            "same_train_order": True,
            "shared_initial_checkpoint": True,
            "shared_parameter_optimizer_state_restored": True,
            "new_depth_phase_optimizer_state": "zero-initialized Adam moments",
            "evaluation_steps_relative_to_continuation": [0, 100, 200, 300],
            "sampler": "fresh deterministic continuation view at global offset 300; terminal step saves cursor without requesting another batch",
            "checkpoint_resume": "reload and resume identity is checked without consuming a scientific batch after step300",
        },
        "common_objective": {
            "formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
            "lambda_transfer": 28.48464149920509,
            "lambda_image": 13.353282014834436,
            "tuning": False,
            "recalibration": False,
        },
        "propagation_factorization": {
            "candidate_b_segment": "S_depth(phi,psi)=C_active P_(d/2) M_(E_active psi) P_(d/2) E_active M_phi",
            "current_segment": "S_d(phi)=C_active P_d E_active M_phi",
            "d_m": 0.0015,
            "half_d_m": 0.00075,
            "midpoint_support": [100, 100],
            "production_grid": [2048, 2048],
            "midpoint_crop": False,
            "midpoint_outside_support": "fixed_transparent",
            "neutral_identity": "psi0=0 and production ASM P_(d/2)P_(d/2)=P_d within the frozen tolerance",
        },
        "evidence_durability": {
            "at_each_evaluation": [
                "scientific forward",
                "construct canonical sufficient observations immediately",
                "atomically persist write-once sufficient observations",
                "reload and verify SHA/schema/shape",
                "then run optional diagnostics",
                "then construct bootstrap/report inputs",
            ],
            "required_recovery_inputs": [
                "per-example Smooth records",
                "DEV image statistics",
                "template anchor and cyclic transfer records",
                "raw/diversity sufficient observations",
                "adapter/generator diagnostics from captured state",
                "midpoint depth diagnostics for B",
            ],
        },
        "preflight": {
            "required": True,
            "production_entrypoint": True,
            "synthetic_disposable_chain": "forward-result -> sufficient builder -> persistence -> reload -> adapter diagnostic -> generator diagnostic -> cyclic diagnostic -> bootstrap input -> report -> lineage",
            "missing_helper_regression": "V10._adapter_observation must exist; N4._adapter_observation is not a required dependency",
            "real_disposable_updates_per_candidate": 3,
            "cuda": True,
            "checkpoint_reload": True,
            "negative_controls": True,
        },
        "metrics": {
            "primary": "Delta_depth = R_B300-R_A300",
            "supported": "Delta_depth>=+0.05 AND paired CI lower>0 AND B reconstruction delta<=+0.005 AND cyclic template/Smooth retained AND raw validity passes",
            "diagnostic": "Delta_A = R_A300-0.4838; no second Gate",
            "reported": ["Smooth at 0/100/200/300", "DEV image loss", "cyclic template transfer", "cyclic Smooth transfer", "raw/diversity", "adapter residual", "generator phase", "midpoint depth phase", "per-class recall", "factor singular values"],
            "bootstrap": {"replicates": 999, "seed": 271828, "unit": "paired identical DEV rows"},
        },
        "stage_a": {"firewall": "R300>=0.937", "independent": True, "stage_b_authorized_by_n5_v2": False},
        "claim_boundary": {"scientific_data_consumed": False, "scientific_forward_executed": False, "scientific_training_executed": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "physical_realizability_claim": False},
        "one_shot": {"preflight": 1, "candidate_a_scientific_attempts": 1, "candidate_b_scientific_attempts": 1, "retry": False, "second_arm": False, "v1_retry_or_continuation": False, "v1_observations_reused": False, "poor_performance_is": "valid scientific outcome, not a harness blocker"},
        "hard_stop_conditions": ["required authority cannot be resolved", "A/B symmetry cannot be established", "production true-depth equivalence fails", "sufficient-observation persistence/reload preflight fails", "trainable/frozen firewall fails", "CUDA or checkpoint validation fails", "data leakage", "required statistic is undefined"],
        "design_decisions": ["V2 is a successor experiment, not a repair, retry or continuation of V1.", "A and B start from the exact C1 step300 endpoint and use the same objective, data order, RNG convention, optimizer and budget.", "Only B receives the propagation-separated midpoint phase plane; no rank/loss/lambda/lr redesign is permitted.", "A post-forward diagnostic failure cannot trigger another scientific forward; recovery is report-only from persisted sufficient observations."],
        "git": authorities["git"],
        "future_authorization_scope": "EXECUTE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2: one bounded preflight, one A 300-update run, one B 300-update run, frozen paired aggregate; no retry or downstream stage",
        "next": "requires the separately authorized V2 execution consumer; this contract does not itself execute training",
    }


def materialize(root: Path, output_dir: Path) -> dict[str, Any]:
    target_dir = (root / output_dir).resolve()
    _require(not target_dir.exists(), f"refusing to reuse successor contract directory: {target_dir}")
    authorities = _validate_inputs(root)
    target_dir.mkdir(parents=True, exist_ok=False)
    path = target_dir / CONTRACT_NAME
    payload = _payload(root, authorities)
    atomic_write_new_json(path, payload, artifact_kind="N5 V2 successor contract")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": payload["outer_gate"], "contract": {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}, "scientific_data_consumed": False, "scientific_forward": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        result = materialize(root, args.output_dir)
    except Exception as error:
        result = {"result": "BLOCKED", "status": "invalid_harness", "classification": "BLOCKED_N5_V2_SUCCESSOR_CONTRACT", "error": {"type": type(error).__name__, "message": str(error)}}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("result") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
