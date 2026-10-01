"""Define the bounded class-balanced structural-gradient Canary contract.

This definition consumer is deliberately CPU/read-only.  It records that the
accepted production sampler already supplies one TRAIN row per class and
freezes the only scientific intervention: equal normalized classwise
structural-gradient aggregation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import git_identity
from ldm_project.v2.evidence import atomic_write_new_json, file_sha256
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as joint


TASK_ID = "DEFINE_LIGHTGEN_2D_CLASS_BALANCED_JOINT_CANARY_CONTRACT_V1"
CONTRACT_SCHEMA = "lightgen_2d_class_balanced_joint_canary_contract.v1"
CLASSIFICATION = "CLASS_BALANCED_JOINT_CANARY_CONTRACT_DEFINED"
OUTER_GATE = CLASSIFICATION
PROPOSED_AUTHORIZATION = "AUTHORIZED_EXECUTE_CLASS_BALANCED_JOINT_CANARY_V1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_CONSUMER_HEAD = "500045d6a92843c81ffeda010e01351f559e453b"
HISTORICAL_PRODUCER_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

START_CHECKPOINT = joint.START_CHECKPOINT
START_CHECKPOINT_SHA256 = joint.START_CHECKPOINT_SHA256.lower()
ATTRIBUTION_REPORT = joint.ATTRIBUTION_REPORT
ATTRIBUTION_REPORT_SHA256 = joint.ATTRIBUTION_REPORT_SHA256.lower()
ATTRIBUTION_GRADIENTS = joint.ATTRIBUTION_GRADIENTS
ATTRIBUTION_GRADIENTS_SHA256 = joint.ATTRIBUTION_GRADIENTS_SHA256.lower()
ACCEPTED_STRUCTURAL_REPORT = joint.ACCEPTED_STRUCTURAL_REPORT
ACCEPTED_STRUCTURAL_REPORT_SHA256 = joint.ACCEPTED_STRUCTURAL_REPORT_SHA256.lower()
STRUCTURAL_REFERENCE = joint.STRUCTURAL_REFERENCE
STRUCTURAL_REFERENCE_SHA256 = joint.STRUCTURAL_REFERENCE_SHA256.lower()
SURROGATE_NORMALIZATION = joint.SURROGATE_NORMALIZATION
SURROGATE_NORMALIZATION_SHA256 = joint.SURROGATE_NORMALIZATION_SHA256.lower()
S1_WINNER = joint.S1_WINNER
S1_WINNER_SHA256 = joint.S1_WINNER_SHA256.lower()
REUSED_STEP9000_DEV = joint.REUSED_STEP9000_DEV
REUSED_STEP9000_DEV_SHA256 = joint.REUSED_STEP9000_DEV_SHA256.lower()
REUSED_STEP9000_S1 = joint.REUSED_STEP9000_S1
REUSED_STEP9000_S1_SHA256 = joint.REUSED_STEP9000_S1_SHA256.lower()

START_STEP = 9000
MAX_UPDATES = 300
PRIMARY_ENDPOINT = 9300
CHECKPOINT_STEPS = (9100, 9200, 9300)
LOGICAL_BATCH_SIZE = 10
GRADIENT_ENGINE = joint.SCALAR_REFERENCE_ENGINE
STRUCTURAL_AGGREGATION_MODE = "CLASS_BALANCED_EQUAL_NORMALIZED"
CLASS_COUNT = 10
CLASS_WEIGHT = 1.0 / CLASS_COUNT
EPS_PROJ = joint.EPS_PROJ
DELTA_JOINT = joint.DELTA_JOINT
BETA = joint.BETA
GAMMA = joint.GAMMA
GRADIENT_RTOL = joint.GRADIENT_RTOL
GRADIENT_ATOL = joint.GRADIENT_ATOL

OUTPUT_ROOT = Path("outputs/20260922_class_balanced_joint_canary_v1")
SCIENTIFIC_ATTEMPT_DIR = OUTPUT_ROOT / "runs" / "attempt_01"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "runs" / "contract_definition_01"
CONTRACT_NAME = "lightgen_2d_class_balanced_joint_canary_contract.v1.json"


class ContractError(RuntimeError):
    """A class-balanced Canary authority cannot be defined safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"{value} must be repository-relative")
    return (root / relative).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, relative: str | Path, expected: str, role: str) -> dict[str, str]:
    target = _path(root, relative)
    _require(target.is_file(), f"{role} is missing: {target}")
    _require(file_sha256(target).lower() == expected.lower(), f"{role} hash changed")
    return {"path": _relative(root, target), "sha256": expected.lower(), "role": role}


def _git_contract(root: Path) -> dict[str, Any]:
    current = git_identity(root)
    _require(current.get("branch") == EXPECTED_BRANCH, "current consumer branch changed")
    _require(current.get("head") == EXPECTED_CONSUMER_HEAD, "current consumer HEAD changed")
    binding = robust._bind_git_provenance(current)
    _require(binding["producer"]["head"] == HISTORICAL_PRODUCER_HEAD, "historical producer HEAD changed")
    return {
        "producer": binding["producer"],
        "consumer": {**binding["consumer"], "worktree_dirty": current["worktree_dirty"], "status_sha256": current["status_sha256"]},
        "consumer_head_exact_required": False,
        "required_consumer_head": None,
        "head_equal": binding["head_equal"],
        "producer_head_is_provenance_only": True,
        "semantics": "HASH_ON_TRUST_BOUNDARY_V2: producer HEAD is provenance unless an explicit consumer exact lock is consumed",
    }


def production_batch_semantics(root: Path, authorities: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Verify the accepted production sampler is already one-row-per-class."""

    loaded = joint._load_authorities(root) if authorities is None else authorities
    checkpoint = _mapping(loaded["step9000"], "accepted step9000 checkpoint")
    sampler = _mapping(checkpoint.get("sampler_state"), "accepted sampler state")
    values = sampler.get("permutations")
    labels = loaded["labels"].cpu()
    _require(isinstance(values, list) and len(values) == CLASS_COUNT, "production sampler must expose ten class permutations")
    permutations = [value.cpu() if torch.is_tensor(value) else None for value in values]
    _require(all(value is not None and value.ndim == 1 and value.numel() > 0 for value in permutations), "production sampler permutations are malformed")
    records: list[dict[str, Any]] = []
    for cursor in (START_STEP, START_STEP + 1, START_STEP + 2):
        indices = torch.stack([permutations[c][cursor % permutations[c].numel()] for c in range(CLASS_COUNT)])
        batch_labels = labels.index_select(0, indices)
        counts = torch.bincount(batch_labels, minlength=CLASS_COUNT)
        _require(torch.equal(batch_labels, torch.arange(CLASS_COUNT)), f"production batch labels are not class ordered at cursor {cursor}")
        _require(torch.equal(counts, torch.ones(CLASS_COUNT, dtype=counts.dtype)), f"production batch is not exactly one row per class at cursor {cursor}")
        records.append({"cursor": cursor, "indices": indices.tolist(), "labels": batch_labels.tolist(), "counts": counts.tolist()})
    return {
        "verified": True,
        "source": "joint.ContinuationSampler.next_batch",
        "rows_per_update": LOGICAL_BATCH_SIZE,
        "classes": list(range(CLASS_COUNT)),
        "per_class_counts": [1] * CLASS_COUNT,
        "scientific_semantics": "accepted production batch already contains exactly one canonical TRAIN row per class",
        "verified_cursors": records,
        "new_sampler_required": False,
    }


def build_contract(root: Path) -> dict[str, Any]:
    root = root.resolve()
    authorities = joint._load_authorities(root)
    _require(not _path(root, SCIENTIFIC_ATTEMPT_DIR).exists(), "class-balanced scientific attempt_01 already exists")
    batch_semantics = production_batch_semantics(root, authorities)
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
        "git": _git_contract(root),
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
            "production_batch_semantics": batch_semantics,
        },
        "protocol": {
            "start_step": START_STEP,
            "max_updates": MAX_UPDATES,
            "primary_endpoint": PRIMARY_ENDPOINT,
            "checkpoints": list(CHECKPOINT_STEPS),
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "production_batch_semantics": "exactly_one_row_per_class",
            "losses_unchanged": ["L_struct", "L_S", "L_I", "L_C"],
            "joint_normalization_unchanged": {"delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "restore_step9000_state": True},
            "sampler_and_data_order_unchanged": True,
            "step9046_recovery_consumed": False,
            "stage2": {"executed": False, "auto_continuation": False, "decision_after": PRIMARY_ENDPOINT},
        },
        "structural_aggregation": {
            "mode": STRUCTURAL_AGGREGATION_MODE,
            "class_count": CLASS_COUNT,
            "class_weight": CLASS_WEIGHT,
            "epsilon": EPS_PROJ,
            "formula": "g_struct_c=grad(L_struct_c); u_struct_c=g_struct_c/(||g_struct_c||+eps); v_struct_CB=(1/10)sum_c u_struct_c; u_struct_CB=v_struct_CB/(||v_struct_CB||+eps)",
            "semantic_formula": "v_SIC=u_S+u_I+u_C; u_SIC=v_SIC/||v_SIC||",
            "joint_formula": "q_joint_CB=(u_struct_CB+u_SIC)/RMS(u_struct_CB+u_SIC)",
            "outcome_independent_equal_weights": True,
            "trainable_change_only": "structural_gradient_aggregation",
        },
        "gradient_backend": {
            "selected": GRADIENT_ENGINE,
            "training_backend": GRADIENT_ENGINE,
            "reason": "exact scalar classwise reference VJP; no superseded batched speed gate",
            "performance_gate": "technical executability only; no +10% batched requirement",
        },
        "acceptance": {
            "primary": {"endpoint": PRIMARY_ENDPOINT, "structural": {"metric": "macro_log_rho", "improvement": "macro_log_rho(9000)-macro_log_rho(9300) > 0", "paired_bootstrap_ci95_lower": "> 0"}, "semantic": {"L_S_9300": "<= L_S_9000", "L_I_9300": "<= L_I_9000", "L_C_9300": "<= L_C_9000"}},
            "additional": ["no severe per-class structural reversal", "class-balanced aggregate nondegenerate", "joint direction nondegenerate", "reconstruction pass", "optical validity pass", "protected/test/sealed reads are zero"],
            "secondary_descriptive_only": ["SSR_struct", "intermediate checkpoint diagnostics"],
            "checkpoint_selection": "9300 is the fixed primary endpoint; 9100 and 9200 cannot replace it",
        },
        "execution_boundary": {"scientific_training_executed": False, "scientific_attempt_created": False, "attempt_path": SCIENTIFIC_ATTEMPT_DIR.as_posix(), "cuda_or_profiler_executed": False, "model_selection": False, "promotion": False, "proposed_authorization": PROPOSED_AUTHORIZATION},
        "claim_boundary": {"bounded_canary_only": True, "stage2_not_authorized": True, "no_checkpoint_promotion": True, "no_hardware_readiness": True, "new_authorization_required": True},
    }


def write_contract(root: Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    root = root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "contract output must be a distinct child under outputs")
    _require(not target.exists(), f"contract output already exists: {target}")
    payload = build_contract(root)
    target.mkdir(parents=True, exist_ok=False)
    path = target / CONTRACT_NAME
    atomic_write_new_json(path, payload, artifact_kind="class-balanced joint Canary contract")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    path = write_contract(args.repo_root, args.output_dir)
    print(json.dumps({"result": "READY", "outer_gate": OUTER_GATE, "contract": path.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
