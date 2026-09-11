"""Execute the N3 empirical ceiling once under the frozen balanced protocol.

The fold authority is verified before q is loaded.  Sampling and compatibility
metrics are delegated to the existing empirical-ceiling implementation, so
the new consumer changes only the development row-to-fold assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, tensor_sha256
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison
from scripts.research import define_n3_source_position_balanced_development_protocol_v1 as definition
from scripts.research import execute_n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1 as ceiling_execution


TASK_ID = "N3_SOURCE_POSITION_BALANCED_EMPIRICAL_CEILING_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_source_position_balanced_empirical_ceiling_report.v1"
REPORT_NAME = "n3_source_position_balanced_empirical_ceiling_report.v1.json"
EXPECTED_BRANCH = definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = definition.EXPECTED_STARTING_HEAD

PROTOCOL_PATH = definition.OUTPUT_DIR / definition.ARTIFACT_NAME
# Filled only after the write-once protocol artifact is created and hashed.
EXPECTED_PROTOCOL_SHA256 = "5700ebb3b4b163d6d911763ce70ef77ca85f1c3ef387b7b0fdaed153e36e0904"
REPORT_OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_source_position_balanced_empirical_ceiling_v1_attempt_01"
)

CLASSIFICATION_SUPPORTED = "SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_SUPPORTED"
CLASSIFICATION_NOT_SUPPORTED = "SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED"

CLASS_COUNT = definition.CLASS_COUNT
DIMENSION = definition.DIMENSION
POOL_COUNT = definition.POOL_COUNT
FOLD_COUNT = definition.FOLD_COUNT
FIT_COUNT_PER_FOLD = definition.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = definition.HELDOUT_COUNT_PER_FOLD
RNG_NAMESPACE = ceiling_execution.RNG_NAMESPACE
DATA_VS_MODEL_REPLICATES = ceiling_execution.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = ceiling_execution.MODEL_VS_MODEL_REPLICATES
GENERATED_BANK_SIZE = ceiling_execution.GENERATED_BANK_SIZE
MODEL_REFERENCE_PAIRS = ceiling_execution.MODEL_REFERENCE_PAIRS
COMPATIBILITY_Q = ceiling_execution.COMPATIBILITY_Q
SCALAR_METRICS = ceiling_execution.SCALAR_METRICS

SAFE_DIRTY_PATHS = set(definition.SAFE_DIRTY_PATHS) | {
    "scripts/research/execute_n3_source_position_balanced_empirical_ceiling_v1.py",
}


class _InvalidHarnessError(RuntimeError):
    """A declared input or numerical boundary failed during measurement."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _invalid(condition: bool, message: str) -> None:
    if not condition:
        raise _InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be a list")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_json_exact(repo_root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {target}")
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    return dict(_mapping(json.loads(target.read_text(encoding="utf-8")), name))


def _git_state(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "branch": branch,
        "head": head,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_clean": not status,
    }


def _validate_protocol(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(EXPECTED_PROTOCOL_SHA256 != "PENDING_PROTOCOL_SHA256", "protocol hash is not frozen")
    protocol = _read_json_exact(repo_root, PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "frozen source-position protocol")
    definition.validate_protocol_payload(protocol)
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "current HEAD changed from frozen authority base")
    frozen_git = _mapping(protocol.get("git"), "protocol git")
    _require(frozen_git.get("branch") == EXPECTED_BRANCH, "protocol branch changed")
    _require(frozen_git.get("starting_committed_head") == EXPECTED_STARTING_HEAD, "protocol starting HEAD changed")
    unexpected_dirty = [path for path in git["dirty_paths"] if path not in SAFE_DIRTY_PATHS]
    _require(not unexpected_dirty, f"unrelated local changes invalidate provenance: {unexpected_dirty}")
    _require(not REPORT_OUTPUT_DIR.exists(), f"write-once report output already exists: {REPORT_OUTPUT_DIR}")

    # All external JSON/manifest authorities are checked before q loading.
    definition._validate_current_state(repo_root)
    stationarity = definition._validate_stationarity_report(repo_root)
    empirical = definition._validate_empirical_authorities(repo_root)
    source_position = definition._validate_source_position_authority(repo_root)
    source_indices, labels, _row_positions, metadata = definition._load_row_metadata(repo_root)
    fold_ids, rows, fold_identity = definition.assign_source_position_balanced_folds(source_indices, labels)

    _require(
        protocol["protocol"]["assignment_sha256"] == fold_identity["assignment_sha256"],
        "runtime source-position assignment differs from frozen protocol",
    )
    _require(protocol["rows"] == rows, "runtime row-to-fold mapping differs from frozen protocol")
    _require(protocol["fold_identity"] == fold_identity, "runtime fold identity differs from frozen protocol")
    frozen_empirical = _mapping(protocol.get("empirical_ceiling_authority"), "frozen empirical authority")
    _require(frozen_empirical.get("contract") == empirical["contract"], "empirical contract authority changed")
    _require(frozen_empirical.get("report") == empirical["report"], "empirical report authority changed")
    _require(frozen_empirical.get("execution_source") == empirical["execution_source"], "empirical execution source changed")
    _require(frozen_empirical.get("contract_source") == empirical["contract_source"], "empirical contract source changed")
    _require(
        file_sha256(repo_root / Path(str(empirical["execution_source"]["path"])))
        == str(empirical["execution_source"]["sha256"]),
        "reused empirical execution source hash changed",
    )
    _require(stationarity["classification"] == "N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED", "stationarity authority changed")
    _require(source_position["source_index_sha256"] == metadata["source_indices_sha256"], "source-position metadata identity changed")
    return protocol, metadata, {
        "source_indices": source_indices,
        "labels": labels,
        "fold_ids": fold_ids,
        "rows": rows,
        "fold_identity": fold_identity,
        "empirical": empirical,
    }


def _validate_loaded_identity(
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    _invalid(tuple(q.shape) == (POOL_COUNT, DIMENSION), "development q shape changed")
    _invalid(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "development q is invalid")
    _invalid(tuple(source_indices.shape) == (POOL_COUNT,), "source-index shape changed")
    _invalid(tuple(labels.shape) == (POOL_COUNT,), "label shape changed")
    expected_indices = torch.tensor([int(row["source_index"]) for row in rows], dtype=torch.int64)
    expected_labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.int64)
    _invalid(torch.equal(source_indices, expected_indices), "loaded source-index identity differs from protocol")
    _invalid(torch.equal(labels, expected_labels), "loaded class identity differs from protocol")
    _invalid(tensor_sha256(source_indices) == metadata["source_indices_sha256"], "loaded source-index hash changed")
    _invalid(tensor_sha256(labels) == metadata["labels_sha256"], "loaded label hash changed")
    fold_ids = torch.tensor([int(row["heldout_fold"]) for row in rows], dtype=torch.int64)
    _invalid(bool(fold_ids.ge(0).all()) and bool(fold_ids.lt(FOLD_COUNT).all()), "protocol fold ids are invalid")
    return fold_ids, {
        "count": POOL_COUNT,
        "dimension": DIMENSION,
        "coordinate": "R2_AMPLITUDE",
        "q_r2_amplitude_sha256": tensor_sha256(q),
        "source_indices_sha256": tensor_sha256(source_indices),
        "labels_sha256": tensor_sha256(labels),
        "q_values_loaded_after_protocol_verification": True,
        "protocol_assignment_sha256": protocol["protocol"]["assignment_sha256"],
    }


def _duplicate_summary(fold_result: Mapping[str, Any]) -> dict[str, Any]:
    records = list(_sequence(fold_result.get("data_sampling"), "data sampling")) + list(
        _sequence(fold_result.get("model_sampling"), "model sampling")
    )
    rates = [float(_mapping(record, "sampling record")["duplicate_row_rate"]) for record in records]
    _invalid(len(rates) == DATA_VS_MODEL_REPLICATES + MODEL_VS_MODEL_REPLICATES, "sampling replicate count changed")
    _invalid(all(0.0 <= rate <= 1.0 for rate in rates), "duplicate rate is outside its descriptive domain")
    return {
        "role": "descriptive only; not a compatibility gate",
        "replicate_count": len(rates),
        "min": min(rates),
        "mean": sum(rates) / len(rates),
        "max": max(rates),
    }


def _fold_result(
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
    protocol_fold: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        result = dict(ceiling_execution._evaluate_fold(q, source_indices, fold_ids, fold))
    except ceiling_execution._InvalidHarnessError as error:
        raise _InvalidHarnessError(str(error)) from error
    result["protocol_fold"] = dict(protocol_fold)
    result["class_counts"] = {
        "heldout": [int(((labels == label) & (fold_ids == fold)).sum()) for label in range(CLASS_COUNT)],
        "fit": [int(((labels == label) & (fold_ids != fold)).sum()) for label in range(CLASS_COUNT)],
    }
    result["source_position_rank_invariant"] = {
        "heldout_count": int((fold_ids == fold).sum()),
        "per_class_count": definition.ROWS_PER_CLASS_PER_FOLD,
        "full_observed_rank_range": True,
        "round_robin_residue": fold,
    }
    result["duplicate_row_rate"] = _duplicate_summary(result)
    result["raw_q_summary"] = {
        "data_vs_model_energy_distance": result["primary"]["data_vs_model_mean_energy_distance"],
        "model_vs_model_energy_distance_q95": result["primary"]["model_vs_model_q95_energy_distance"],
        "all_scalar_guardrails": dict(result["guardrails"]["compatibility_components"]),
        "compatible": bool(result["compatibility"]),
    }
    return result


def _empirical_result(fold_results: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(fold_results) == FOLD_COUNT, "all five new folds were not evaluated")
    compatible_count = sum(bool(item["compatibility"]) for item in fold_results)
    return {
        "candidate": "P_EMPIRICAL_FIT",
        "family": "UNIFORM_EMPIRICAL_FIT_JOINT_ROW_BOOTSTRAP_V1",
        "fit_definition": "uniform empirical distribution over each new fold's exact 800 fitting q rows",
        "joint_structure_preserved": True,
        "operationally_valid": all(bool(item["operationally_valid"]) for item in fold_results),
        "compatible_fold_count": compatible_count,
        "fold_count": FOLD_COUNT,
        "development_supported": compatible_count == FOLD_COUNT,
        "support_rule": "all five new folds must be operationally valid and complete-compatible",
        "fold_results": [dict(item) for item in fold_results],
    }


def _interpretation(classification: str, compatible_count: int) -> dict[str, str]:
    if classification == CLASSIFICATION_SUPPORTED:
        return {
            "fact": f"FACT: the deterministic class/source-position-balanced protocol restored empirical fit-to-heldout compatibility on {compatible_count}/{FOLD_COUNT} folds under the frozen ceiling semantics.",
            "model_assumption": "MODEL_ASSUMPTION: within the observed 1000-row development domain, explicit class/source-position balancing is sufficient for this empirical compatibility diagnostic; this authorizes only future producer development planning under the new protocol and does not test extrapolation.",
        }
    return {
        "fact": f"FACT: the deterministic class/source-position-balanced protocol achieved empirical fit-to-heldout compatibility on {compatible_count}/{FOLD_COUNT} folds under the frozen ceiling semantics.",
        "model_assumption": "MODEL_ASSUMPTION: source-position balancing alone is insufficient to make the existing 1000-row bank a stable development authority; the next target is a replacement or expanded N3 development-data authority, and producer-family escalation remains stopped.",
    }


def _valid_report(
    repo_root: Path,
    git_before: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_metadata: Mapping[str, Any],
    loaded_identity: Mapping[str, Any],
    fold_identity: Mapping[str, Any],
    empirical: Mapping[str, Any],
) -> dict[str, Any]:
    compatible_count = int(empirical["compatible_fold_count"])
    classification = CLASSIFICATION_SUPPORTED if compatible_count == FOLD_COUNT else CLASSIFICATION_NOT_SUPPORTED
    old_empirical = _mapping(protocol.get("empirical_ceiling_authority"), "protocol empirical authority")
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": classification,
        "protocol": {
            "path": _relative(repo_root, repo_root / PROTOCOL_PATH),
            "schema": protocol["schema"],
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "semantic_authority_id": protocol["semantic_authority_id"],
            "classification": protocol["classification"],
            "assignment_sha256": protocol["protocol"]["assignment_sha256"],
        },
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": _git_state(repo_root)["head"],
            "dirty_paths": git_before["dirty_paths"],
            "worktree_clean": git_before["worktree_clean"],
        },
        "input_verification": {
            "protocol_verified_before_q_load": True,
            "row_identity_verified_before_q_load": True,
            "old_fold_authority_immutable": True,
            "old_empirical_ceiling_contract_sha256": old_empirical["contract"]["sha256"],
            "old_empirical_ceiling_report_sha256": old_empirical["report"]["sha256"],
            "old_empirical_ceiling_fold_pattern": old_empirical["report"]["fold_pattern"],
            "source_position_authority_verified": True,
            "sealed_confirmation_values_loaded": False,
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
        },
        "development_lineage": {
            "pool": dict(loaded_identity),
            "protocol_fold_identity": dict(fold_identity),
            "protocol_assignment_sha256": protocol["protocol"]["assignment_sha256"],
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "per_class_fit_rows_per_fold": definition.FIT_ROWS_PER_CLASS_PER_FOLD,
            "per_class_heldout_rows_per_fold": definition.ROWS_PER_CLASS_PER_FOLD,
            "protocol_source_position_coverage": "each class uses ranks r=0..99 and heldout fold=r mod 5",
            "old_fold_assignment_sha256": definition.OLD_FOLD_ASSIGNMENT_SHA256,
            "old_fold_reinterpreted": False,
            "metadata_authority": dict(protocol_metadata),
        },
        "execution_settings": {
            "device": "cpu",
            "dtype": "float64 q; int64 row indices",
            "rng_namespace": RNG_NAMESPACE,
            "rng_generator": "torch.Generator(device='cpu')",
            "seed_derivation": "sha256(namespace|role|fold|replicate) first 64 bits masked to signed 63-bit",
            "fold_count": FOLD_COUNT,
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
            "compatibility_q": COMPATIBILITY_Q,
            "complete_row_sampling": True,
            "with_replacement": True,
            "independent_dimension_sampling": False,
            "cross_fold_pooling": False,
            "seed_retry": False,
            "best_bank_selection": False,
            "samplewise_normalization": False,
            "guardrails": list(SCALAR_METRICS),
        },
        "empirical_ceiling": {
            **dict(empirical),
            "protocol_reused_exactly": True,
            "sampling_and_metric_source": dict(old_empirical["execution_source"]),
            "compatibility_contract_source": dict(old_empirical["contract"]),
            "duplicate_rate_role": "descriptive only; no effect on compatibility or classification",
        },
        "aggregate": {
            "classification": classification,
            "fold_count": FOLD_COUNT,
            "operationally_valid": bool(empirical["operationally_valid"]),
            "compatible_fold_count": compatible_count,
            "all_five_folds_required": True,
            "development_supported": compatible_count == FOLD_COUNT,
            "producer_family_evaluated": False,
        },
        "interpretation": _interpretation(classification, compatible_count),
        "claim_boundary": {
            "diagnostic_only": True,
            "development_domain_only": True,
            "not_extrapolation": True,
            "not_a_producer": True,
            "not_confirmation_evidence": True,
            "not_a_reinterpretation_of_old_folds": True,
            "population_truth": False,
        },
        "governance": {
            "diagnostic_executed": True,
            "producer_family_evaluated": False,
            "producer_selection": False,
            "selected_producer": None,
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "project_state": {
            "old_stationarity_classification": "N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED",
            "old_empirical_ceiling_compatible_folds": 3,
            "old_empirical_ceiling_fold_pattern": [False, True, False, True, True],
            "old_minimal_intervention_classification": "NO_MINIMAL_INTERVENTION_SUPPORTED",
            "selected_producer": None,
            "sealed_confirmation_unchanged": True,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "not_executed": [
            "learnable producer",
            "M2",
            "Student-t",
            "KDE/flow/mixture",
            "alternate split candidates",
            "split optimization",
            "sealed confirmation",
            "CUDA/training",
            "N4/N5/N6",
            "generation",
        ],
        "artifact_policy": {
            "write_once": True,
            "immutable_after_publish": True,
            "bootstrap_tensors_embedded": False,
            "existing_authorities_modified": False,
            "old_fold_authority_modified": False,
            "sealed_confirmation_modified": False,
        },
    }


def _blocked_report(
    repo_root: Path,
    git_before: Mapping[str, Any],
    protocol: Mapping[str, Any],
    error: _InvalidHarnessError,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "blocked",
        "result": "BLOCKED",
        "classification": None,
        "blocker": {
            "type": "INVALID_HARNESS",
            "message": str(error),
            "action": "stop before scientific compatibility classification",
        },
        "protocol": {
            "path": _relative(repo_root, repo_root / PROTOCOL_PATH),
            "schema": protocol["schema"],
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "assignment_sha256": protocol["protocol"]["assignment_sha256"],
        },
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": _git_state(repo_root)["head"],
            "dirty_paths": git_before["dirty_paths"],
        },
        "scientific_classification_emitted": False,
        "empirical_ceiling": None,
        "governance": {
            "diagnostic_executed": False,
            "producer_family_evaluated": False,
            "producer_selection": False,
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }


def _write_report(repo_root: Path, output_dir: Path, report: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once report output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / REPORT_NAME,
            report,
            artifact_kind="N3 source-position-balanced empirical ceiling report",
        )
        if output_dir.exists():
            raise FileExistsError(f"write-once report output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path = REPORT_OUTPUT_DIR) -> dict[str, Any]:
    git_before = _git_state(repo_root)
    protocol, metadata, runtime = _validate_protocol(repo_root)
    source_indices_list = runtime["source_indices"]
    labels_list = runtime["labels"]
    fold_ids_list = runtime["fold_ids"]
    rows = runtime["rows"]

    try:
        q, source_indices, labels, _pool_identity = comparison._load_development_pool(repo_root)
        fold_ids, loaded_identity = _validate_loaded_identity(
            q,
            source_indices,
            labels,
            protocol,
            metadata,
            rows,
        )
        expected_fold_ids = torch.tensor(fold_ids_list, dtype=torch.int64)
        _invalid(torch.equal(fold_ids, expected_fold_ids), "loaded fold identity differs from frozen protocol")
        _invalid(source_indices.tolist() == source_indices_list, "loaded source indices differ from frozen protocol")
        _invalid(labels.tolist() == labels_list, "loaded labels differ from frozen protocol")
        with torch.no_grad():
            fold_results = [
                _fold_result(
                    q,
                    source_indices,
                    labels,
                    fold_ids,
                    fold,
                    runtime["fold_identity"]["folds"][fold],
                )
                for fold in range(FOLD_COUNT)
            ]
    except (ceiling_execution._InvalidHarnessError, _InvalidHarnessError) as error:
        blocked = _InvalidHarnessError(str(error))
        report_path = _write_report(repo_root, output_dir, _blocked_report(repo_root, git_before, protocol, blocked))
        return {
            "result": "BLOCKED",
            "classification": None,
            "protocol": _relative(repo_root, repo_root / PROTOCOL_PATH),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "report": _relative(repo_root, report_path),
            "report_sha256": file_sha256(report_path),
            "blocker": "INVALID_HARNESS",
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        }

    empirical = _empirical_result(fold_results)
    report = _valid_report(
        repo_root,
        git_before,
        protocol,
        metadata,
        loaded_identity,
        runtime["fold_identity"],
        empirical,
    )
    report_path = _write_report(repo_root, output_dir, report)
    return {
        "result": "VALID_COMPLETED",
        "classification": report["classification"],
        "protocol": _relative(repo_root, repo_root / PROTOCOL_PATH),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "assignment_sha256": protocol["protocol"]["assignment_sha256"],
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "compatible_folds": empirical["compatible_fold_count"],
        "fold_count": FOLD_COUNT,
        "git_branch": git_before["branch"],
        "starting_head": git_before["head"],
        "ending_head": report["git"]["ending_head"],
        "sealed_confirmation_consumed": False,
        "producer_family_evaluated": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / REPORT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
