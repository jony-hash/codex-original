"""Freeze the N3 class/source-position-balanced development protocol.

This definition-only consumer reads the existing row and class metadata from
the immutable partition manifest.  It does not load q values, sample banks,
evaluate a producer, or consume sealed confirmation values.
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

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison
from scripts.research import define_n3_development_sampling_stationarity_diagnostic_v1 as stationarity
from scripts.research import define_n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1 as ceiling


TASK_ID = "N3_SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_V1"
GATE = TASK_ID
SCHEMA = "n3_source_position_balanced_development_protocol.v1"
READY_CLASSIFICATION = "DEFINE_N3_SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_V1_READY"
CLAIM_CLASSIFICATION = "N3_SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_DEFINED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

STATIONARITY_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_sampling_stationarity_diagnostic_execution_v1_attempt_02/"
    "n3_development_sampling_stationarity_diagnostic_report.v1.json"
)
STATIONARITY_REPORT_SHA256 = "625b4edda39066a4c414337fc0ba7d06d64fda77b6d41d8241806c715ee871a3"

OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_source_position_balanced_development_protocol_v1_attempt_01"
)
ARTIFACT_NAME = "n3_source_position_balanced_development_protocol.v1.json"

CLASS_COUNT = comparison.CLASS_COUNT
DIMENSION = comparison.DIMENSION
POOL_COUNT = comparison.POOL_COUNT
ROWS_PER_CLASS = comparison.ROWS_PER_CLASS
FOLD_COUNT = comparison.FOLD_COUNT
FIT_COUNT_PER_FOLD = comparison.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = comparison.HELDOUT_COUNT_PER_FOLD
ROWS_PER_CLASS_PER_FOLD = comparison.HELDOUT_ROWS_PER_CLASS_PER_FOLD
FIT_ROWS_PER_CLASS_PER_FOLD = comparison.FIT_ROWS_PER_CLASS_PER_FOLD

OLD_FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
OLD_EMPIRICAL_CONTRACT_SHA256 = "fc24ca4c863fded5f23dacc69b13bb16f44224b96d47888d62cb9a3f0f70a38d"
OLD_EMPIRICAL_REPORT_SHA256 = "c4557d44c8d1b40adeb2f362a4f81f0f28d9feb398428fbb1b13c035b043e12c"
OLD_EMPIRICAL_CONTRACT_PATH = ceiling.OUTPUT_DIR / ceiling.CONTRACT_NAME
OLD_EMPIRICAL_REPORT_PATH = stationarity.EMPIRICAL_CEILING_REPORT
SOURCE_POSITION_REPORT_PATH = stationarity.SOURCE_POSITION_REPORT
SOURCE_POSITION_REPORT_SHA256 = stationarity.SOURCE_POSITION_REPORT_SHA256

POSITION_ORDERING = "source_position ascending; exact ties by original development-row identity ascending"
POSITION_IDENTITY = "frozen MNIST source_index"
TIE_BREAK_IDENTITY = "original development row position in the 1000-row authority"
ASSIGNMENT_ALGORITHM = "within_class_source_position_round_robin_v1"

SAFE_DIRTY_PATHS = set(ceiling.SAFE_DIRTY_PATHS) | set(stationarity.SAFE_DIRTY_PATHS) | {
    "scripts/research/define_n3_source_position_balanced_development_protocol_v1.py",
    "scripts/research/execute_n3_source_position_balanced_empirical_ceiling_v1.py",
    "tests/v2/test_n3_source_position_balanced_development_protocol_v1.py",
    "tests/v2/test_n3_source_position_balanced_empirical_ceiling_v1.py",
    "docs/CURRENT_STATUS.md",
    "docs/status/current_handoff.md",
    "docs/status/experiment_history.md",
    "docs/status/research_governance_state.json",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _validate_current_state(repo_root: Path) -> dict[str, Any]:
    stationarity._validate_current_state(repo_root)
    required = (
        "N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED",
        "training_authorized=false",
        "formal_promotion=false",
        "generation_readiness=not_ready",
    )
    for path in (Path("docs/CURRENT_STATUS.md"), Path("docs/status/current_handoff.md")):
        text = (repo_root / path).read_text(encoding="utf-8")
        for marker in required:
            _require(marker in text, f"current state marker is missing from {path}: {marker}")
    return {
        "paths": ["docs/CURRENT_STATUS.md", "docs/status/current_handoff.md"],
        "required_markers": list(required),
        "stationarity_classification": "N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED",
        "state_conflict": False,
    }


def _validate_stationarity_report(repo_root: Path) -> dict[str, Any]:
    report = _read_json_exact(repo_root, STATIONARITY_REPORT, STATIONARITY_REPORT_SHA256, "stationarity report")
    _require(report.get("schema") == "n3_development_sampling_stationarity_diagnostic_report.v1", "stationarity schema changed")
    _require(report.get("status") == "valid_completed" and report.get("result") == "VALID_COMPLETED", "stationarity report is incomplete")
    _require(report.get("classification") == "N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED", "stationarity classification changed")
    governance = _mapping(report.get("governance"), "stationarity governance")
    for key, expected in {
        "producer_family_evaluated": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(governance.get(key) == expected, f"stationarity governance changed: {key}")
    return {
        "path": _relative(repo_root, repo_root / STATIONARITY_REPORT),
        "sha256": STATIONARITY_REPORT_SHA256,
        "schema": report["schema"],
        "classification": report["classification"],
        "result": report["result"],
        "exceptional_fold_count": report.get("aggregate", {}).get("exceptional_fold_count"),
    }


def _validate_empirical_authorities(repo_root: Path) -> dict[str, Any]:
    old_report = _read_json_exact(repo_root, OLD_EMPIRICAL_REPORT_PATH, OLD_EMPIRICAL_REPORT_SHA256, "old empirical ceiling report")
    _require(old_report.get("schema") == "n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1", "old empirical schema changed")
    _require(old_report.get("status") == "valid_completed" and old_report.get("result") == "VALID_COMPLETED", "old empirical report is incomplete")
    _require(old_report.get("classification") == "N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED", "old empirical classification changed")
    old_contract = _read_json_exact(repo_root, OLD_EMPIRICAL_CONTRACT_PATH, OLD_EMPIRICAL_CONTRACT_SHA256, "old empirical ceiling contract")
    ceiling.validate_contract_payload(old_contract)
    _require(old_report.get("contract", {}).get("sha256") == OLD_EMPIRICAL_CONTRACT_SHA256, "old empirical contract link changed")
    lineage = _mapping(old_report.get("development_lineage"), "old empirical development lineage")
    _require(lineage.get("fold_assignment_sha256") == OLD_FOLD_ASSIGNMENT_SHA256, "old fold authority changed")
    _require(old_report.get("aggregate", {}).get("compatible_fold_count") == 3, "old empirical compatible count changed")
    old_pattern = [bool(item.get("compatibility")) for item in _sequence(old_report["empirical_fit"]["fold_results"], "old empirical fold results")]
    _require(old_pattern == [False, True, False, True, True], "old empirical fold pattern changed")
    old_execution_path = Path("scripts/research/execute_n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1.py")
    old_contract_path = Path("scripts/research/define_n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1.py")
    return {
        "contract": {
            "path": _relative(repo_root, repo_root / OLD_EMPIRICAL_CONTRACT_PATH),
            "sha256": OLD_EMPIRICAL_CONTRACT_SHA256,
            "schema": old_contract["schema"],
            "sampling_contract": dict(old_contract["sampling_contract"]),
            "compatibility_framework": dict(old_contract["compatibility_framework"]),
            "contract_script": {
                "path": old_contract["git"]["contract_script"],
                "sha256": old_contract["git"]["contract_script_sha256"],
            },
        },
        "report": {
            "path": _relative(repo_root, repo_root / OLD_EMPIRICAL_REPORT_PATH),
            "sha256": OLD_EMPIRICAL_REPORT_SHA256,
            "schema": old_report["schema"],
            "classification": old_report["classification"],
            "compatible_fold_count": 3,
            "fold_pattern": old_pattern,
            "fold_assignment_sha256": OLD_FOLD_ASSIGNMENT_SHA256,
        },
        "execution_source": {
            "path": old_execution_path.as_posix(),
            "sha256": file_sha256(repo_root / old_execution_path),
        },
        "contract_source": {
            "path": old_contract_path.as_posix(),
            "sha256": file_sha256(repo_root / old_contract_path),
        },
    }


def _validate_source_position_authority(repo_root: Path) -> dict[str, Any]:
    source = stationarity._validate_source_position_authority(repo_root)
    _require(source["path"] == _relative(repo_root, repo_root / SOURCE_POSITION_REPORT_PATH), "source-position authority path changed")
    return dict(source)


def _load_row_metadata(repo_root: Path) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    authority = comparison._verify_authorities(repo_root)
    manifest = _read_json_exact(
        repo_root,
        comparison.PARTITION_MANIFEST,
        comparison.EXPECTED_PARTITION_MANIFEST_SHA256,
        "development partition manifest",
    )
    records: list[tuple[int, int, int]] = []
    for section_name in ("fit", "internal_validation"):
        section = _mapping(manifest.get(section_name), f"partition {section_name}")
        positions = _sequence(section.get("row_positions"), f"partition {section_name} row positions")
        indices = _sequence(section.get("source_indices"), f"partition {section_name} source indices")
        labels = _sequence(section.get("labels"), f"partition {section_name} labels")
        _require(len(positions) == len(indices) == len(labels) == POOL_COUNT // 2, f"partition {section_name} metadata count changed")
        records.extend((int(position), int(index), int(label)) for position, index, label in zip(positions, indices, labels))
    records.sort(key=lambda item: item[0])
    _require([item[0] for item in records] == list(range(POOL_COUNT)), "development row identity is incomplete")
    source_indices = [item[1] for item in records]
    labels = [item[2] for item in records]
    _require(len(set(source_indices)) == POOL_COUNT, "source-position identity is not unique")
    _require(all(0 <= label < CLASS_COUNT for label in labels), "class identity is outside the declared range")
    _require(
        [labels.count(label) for label in range(CLASS_COUNT)] == [ROWS_PER_CLASS] * CLASS_COUNT,
        "class identity is not balanced",
    )
    index_tensor = torch.tensor(source_indices, dtype=torch.int64)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    _require(tensor_sha256(index_tensor) == comparison.EXPECTED_SOURCE_INDEX_SHA256, "source-index metadata hash changed")
    _require(tensor_sha256(label_tensor) == comparison.EXPECTED_SOURCE_LABEL_SHA256, "class metadata hash changed")
    return source_indices, labels, [item[0] for item in records], {
        "development_authority": authority,
        "partition_manifest": {
            "path": _relative(repo_root, repo_root / comparison.PARTITION_MANIFEST),
            "sha256": comparison.EXPECTED_PARTITION_MANIFEST_SHA256,
            "schema": manifest["schema"],
            "historical_algorithm": manifest["algorithm"],
            "historical_runtime_resplit": manifest["runtime_resplit"],
            "historical_runtime_reselection": manifest["runtime_reselection"],
        },
        "count": POOL_COUNT,
        "dimension": DIMENSION,
        "source_indices_sha256": tensor_sha256(index_tensor),
        "labels_sha256": tensor_sha256(label_tensor),
        "q_values_used": False,
    }


def _assignment_payload(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "algorithm": ASSIGNMENT_ALGORITHM,
        "position_identity": POSITION_IDENTITY,
        "tie_break_identity": TIE_BREAK_IDENTITY,
        "class_count": CLASS_COUNT,
        "rows_per_class": ROWS_PER_CLASS,
        "fold_count": FOLD_COUNT,
        "rows": [dict(row) for row in rows],
    }


def assign_source_position_balanced_folds(
    source_indices: list[int],
    labels: list[int],
    row_positions: list[int] | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    """Assign deterministic folds from metadata only; never uses q values or RNG."""

    _require(len(source_indices) == len(labels) == POOL_COUNT, "assignment metadata count changed")
    if row_positions is None:
        row_positions = list(range(POOL_COUNT))
    _require(len(row_positions) == POOL_COUNT, "row-position metadata count changed")
    rows_by_position = {
        int(position): {
            "development_row_position": int(position),
            "source_index": int(source_index),
            "label": int(label),
        }
        for position, source_index, label in zip(row_positions, source_indices, labels)
    }
    _require(set(rows_by_position) == set(range(POOL_COUNT)), "row identity is not a complete 0..999 authority")
    fold_ids = [-1] * POOL_COUNT
    rank_by_position: dict[int, int] = {}
    for label in range(CLASS_COUNT):
        positions = [position for position, row in rows_by_position.items() if row["label"] == label]
        _require(len(positions) == ROWS_PER_CLASS, f"class {label} row count changed")
        ordered = sorted(positions, key=lambda position: (rows_by_position[position]["source_index"], position))
        for rank, position in enumerate(ordered):
            rank_by_position[position] = rank
            fold_ids[position] = rank % FOLD_COUNT
    _require(all(fold_id >= 0 for fold_id in fold_ids), "source-position fold assignment is incomplete")

    rows: list[dict[str, Any]] = []
    for position in range(POOL_COUNT):
        row = dict(rows_by_position[position])
        row["within_class_rank"] = rank_by_position[position]
        row["heldout_fold"] = fold_ids[position]
        rows.append(row)

    folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        held_positions = [position for position, assigned in enumerate(fold_ids) if assigned == fold]
        fit_positions = [position for position, assigned in enumerate(fold_ids) if assigned != fold]
        _require(len(held_positions) == HELDOUT_COUNT_PER_FOLD, f"held-out count changed at fold {fold}")
        _require(len(fit_positions) == FIT_COUNT_PER_FOLD, f"fit count changed at fold {fold}")
        class_summaries: list[dict[str, Any]] = []
        for label in range(CLASS_COUNT):
            held_rows = [row for row in rows if row["label"] == label and row["heldout_fold"] == fold]
            fit_rows = [row for row in rows if row["label"] == label and row["heldout_fold"] != fold]
            held_ranks = [int(row["within_class_rank"]) for row in held_rows]
            _require(len(held_rows) == ROWS_PER_CLASS_PER_FOLD, f"class {label} held-out count changed at fold {fold}")
            _require(len(fit_rows) == FIT_ROWS_PER_CLASS_PER_FOLD, f"class {label} fit count changed at fold {fold}")
            _require(held_ranks == list(range(fold, ROWS_PER_CLASS, FOLD_COUNT)), f"source-position coverage changed at fold {fold}, class {label}")
            class_summaries.append(
                {
                    "label": label,
                    "heldout_count": len(held_rows),
                    "fit_count": len(fit_rows),
                    "heldout_rank_min": min(held_ranks),
                    "heldout_rank_max": max(held_ranks),
                    "heldout_rank_step": FOLD_COUNT,
                    "heldout_rank_count": len(held_ranks),
                    "heldout_rank_residue": fold,
                    "heldout_rank_distribution": held_ranks,
                    "fit_rank_count": len(fit_rows),
                    "fit_complement_covers_other_ranks": True,
                }
            )
        held_position_tensor = torch.tensor(held_positions, dtype=torch.int64)
        fit_position_tensor = torch.tensor(fit_positions, dtype=torch.int64)
        source_tensor = torch.tensor(source_indices, dtype=torch.int64)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        folds.append(
            {
                "fold": fold,
                "heldout_count": len(held_positions),
                "fit_count": len(fit_positions),
                "heldout_per_class": ROWS_PER_CLASS_PER_FOLD,
                "fit_per_class": FIT_ROWS_PER_CLASS_PER_FOLD,
                "heldout_row_identity_sha256": tensor_sha256(held_position_tensor),
                "fit_row_identity_sha256": tensor_sha256(fit_position_tensor),
                "heldout_source_index_sha256": tensor_sha256(source_tensor.index_select(0, held_position_tensor)),
                "fit_source_index_sha256": tensor_sha256(source_tensor.index_select(0, fit_position_tensor)),
                "heldout_label_sha256": tensor_sha256(label_tensor.index_select(0, held_position_tensor)),
                "fit_label_sha256": tensor_sha256(label_tensor.index_select(0, fit_position_tensor)),
                "source_position_coverage": {
                    "within_class_rank_min": 0,
                    "within_class_rank_max": ROWS_PER_CLASS - 1,
                    "all_classes_cover_full_observed_range": True,
                    "round_robin_residue": fold,
                },
                "class_summaries": class_summaries,
                "heldout_fit_disjoint": True,
            }
        )
    assignment = _assignment_payload(rows)
    return fold_ids, rows, {
        "algorithm": ASSIGNMENT_ALGORITHM,
        "position_identity": POSITION_IDENTITY,
        "ordering": POSITION_ORDERING,
        "tie_break": TIE_BREAK_IDENTITY,
        "fold_count": FOLD_COUNT,
        "fold_id_by_development_row": fold_ids,
        "assignment_sha256": canonical_sha256(assignment),
        "folds": folds,
        "rng_used": False,
        "q_values_used": False,
    }


def validate_protocol_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "protocol schema changed")
    _require(payload.get("version") == 1, "protocol version changed")
    _require(payload.get("task_id") == TASK_ID and payload.get("gate") == GATE, "protocol task changed")
    _require(payload.get("classification") == READY_CLASSIFICATION, "protocol readiness changed")
    _require(payload.get("claim_classification") == CLAIM_CLASSIFICATION, "protocol claim changed")
    _require(payload.get("definition_only") is True, "protocol definition boundary changed")
    protocol = _mapping(payload.get("protocol"), "protocol")
    for key, expected in {
        "class_count": CLASS_COUNT,
        "rows_per_class": ROWS_PER_CLASS,
        "pool_count": POOL_COUNT,
        "fold_count": FOLD_COUNT,
        "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
        "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
        "heldout_rows_per_class_per_fold": ROWS_PER_CLASS_PER_FOLD,
        "fit_rows_per_class_per_fold": FIT_ROWS_PER_CLASS_PER_FOLD,
        "algorithm": ASSIGNMENT_ALGORITHM,
        "position_identity": POSITION_IDENTITY,
        "ordering": POSITION_ORDERING,
        "tie_break": TIE_BREAK_IDENTITY,
        "rng_used": False,
    }.items():
        _require(protocol.get(key) == expected, f"protocol semantic changed: {key}")
    rows = _sequence(payload.get("rows"), "protocol rows")
    _require(len(rows) == POOL_COUNT, "protocol row assignment is incomplete")
    normalized_rows: list[dict[str, Any]] = []
    for expected_position, item in enumerate(rows):
        row = _mapping(item, f"protocol row {expected_position}")
        _require(row.get("development_row_position") == expected_position, "protocol row order changed")
        for key in ("source_index", "label", "within_class_rank", "heldout_fold"):
            _require(isinstance(row.get(key), int), f"protocol row field is invalid: {key}")
        normalized_rows.append(dict(row))
    source_indices = [int(row["source_index"]) for row in normalized_rows]
    labels = [int(row["label"]) for row in normalized_rows]
    _require(len(set(source_indices)) == POOL_COUNT, "protocol source-position identity is not unique")
    _require(all(0 <= label < CLASS_COUNT for label in labels), "protocol class identity is invalid")
    _require([labels.count(label) for label in range(CLASS_COUNT)] == [ROWS_PER_CLASS] * CLASS_COUNT, "protocol class balance changed")
    assignment = _assignment_payload(normalized_rows)
    _require(protocol.get("assignment_sha256") == canonical_sha256(assignment), "protocol assignment hash changed")
    fold_summary = _sequence(protocol.get("folds"), "protocol fold summaries")
    _require(len(fold_summary) == FOLD_COUNT, "protocol fold summaries are incomplete")
    for fold, item in enumerate(fold_summary):
        record = _mapping(item, f"protocol fold {fold}")
        _require(record.get("fold") == fold, f"protocol fold order changed at {fold}")
        _require(record.get("heldout_count") == HELDOUT_COUNT_PER_FOLD, f"protocol held-out count changed at {fold}")
        _require(record.get("fit_count") == FIT_COUNT_PER_FOLD, f"protocol fit count changed at {fold}")
        _require(record.get("heldout_per_class") == ROWS_PER_CLASS_PER_FOLD, f"protocol held-out class count changed at {fold}")
        _require(record.get("fit_per_class") == FIT_ROWS_PER_CLASS_PER_FOLD, f"protocol fit class count changed at {fold}")
        _require(record.get("heldout_fit_disjoint") is True, f"protocol overlap at {fold}")
        coverage = _mapping(record.get("source_position_coverage"), f"protocol source-position coverage {fold}")
        for key, expected in {
            "within_class_rank_min": 0,
            "within_class_rank_max": ROWS_PER_CLASS - 1,
            "all_classes_cover_full_observed_range": True,
            "round_robin_residue": fold,
        }.items():
            _require(coverage.get(key) == expected, f"source-position coverage changed at fold {fold}: {key}")
        classes = _sequence(record.get("class_summaries"), f"protocol classes {fold}")
        _require(len(classes) == CLASS_COUNT, f"protocol class summaries incomplete at fold {fold}")
        for label, item_class in enumerate(classes):
            item_mapping = _mapping(item_class, f"protocol class {fold}/{label}")
            _require(item_mapping.get("label") == label, f"protocol class order changed at fold {fold}")
            _require(item_mapping.get("heldout_count") == ROWS_PER_CLASS_PER_FOLD, f"protocol class held-out count changed at fold {fold}")
            _require(item_mapping.get("fit_count") == FIT_ROWS_PER_CLASS_PER_FOLD, f"protocol class fit count changed at fold {fold}")
            _require(item_mapping.get("heldout_rank_distribution") == list(range(fold, ROWS_PER_CLASS, FOLD_COUNT)), f"protocol rank distribution changed at fold {fold}, class {label}")

    old = _mapping(payload.get("empirical_ceiling_authority"), "empirical ceiling authority")
    _require(old.get("contract", {}).get("sha256") == OLD_EMPIRICAL_CONTRACT_SHA256, "old empirical contract identity changed")
    _require(old.get("report", {}).get("sha256") == OLD_EMPIRICAL_REPORT_SHA256, "old empirical report identity changed")
    _require(old.get("report", {}).get("fold_assignment_sha256") == OLD_FOLD_ASSIGNMENT_SHA256, "old fold authority was replaced")
    sampling = _mapping(old.get("contract", {}).get("sampling_contract"), "reused sampling contract")
    for key, expected in {
        "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
        "bank_size": ceiling.GENERATED_BANK_SIZE,
        "data_vs_model_replicates": ceiling.DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": ceiling.MODEL_VS_MODEL_REPLICATES,
        "rng_device": "cpu",
        "rng_namespace": ceiling.RNG_NAMESPACE,
        "seed_retry": False,
        "best_bank_selection": False,
        "samplewise_normalization": False,
    }.items():
        _require(sampling.get(key) == expected, f"reused empirical sampling changed: {key}")
    framework = _mapping(old.get("contract", {}).get("compatibility_framework"), "reused compatibility framework")
    _require(framework.get("scalar_metric_count") == 8, "guardrail count changed")
    _require(framework.get("all_eight_scalar_guardrails_required") is True, "guardrails were weakened")
    _require(framework.get("omitted_guardrails") == [], "a guardrail was omitted")
    _require(framework.get("all_five_folds_required") is True, "five-fold aggregate rule changed")
    _require(framework.get("q") == ceiling.COMPATIBILITY_Q, "compatibility q changed")
    _require(payload.get("contract_totality", {}).get("status") == "passed", "protocol totality did not pass")
    execution = _mapping(payload.get("execution_boundary"), "execution boundary")
    for key, expected in {
        "q_values_used": False,
        "scientific_execution": False,
        "sealed_confirmation_consumed": False,
        "producer_evaluated": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(execution.get(key) == expected, f"execution boundary changed: {key}")
    governance = _mapping(payload.get("governance"), "governance")
    for key, expected in {
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
        "producer_selection": False,
    }.items():
        _require(governance.get(key) == expected, f"governance changed: {key}")


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    state: Mapping[str, Any],
    stationarity_report: Mapping[str, Any],
    empirical: Mapping[str, Any],
    source_position: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_indices: list[int],
    labels: list[int],
    rows: list[dict[str, Any]],
    fold_identity: Mapping[str, Any],
) -> dict[str, Any]:
    old = empirical["contract"]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "semantic_authority_id": TASK_ID,
        "definition_only": True,
        "git": {
            "branch": git["branch"],
            "starting_committed_head": git["head"],
            "definition_script": _relative(repo_root, Path(__file__).resolve()),
            "definition_script_sha256": file_sha256(Path(__file__).resolve()),
            "dirty_paths": git["dirty_paths"],
            "worktree_clean": git["worktree_clean"],
        },
        "authorities": {
            "current_project_state": dict(state),
            "prior_stationarity_report": dict(stationarity_report),
            "source_position": dict(source_position),
            "development_metadata": dict(metadata),
            "old_fold_authority": {
                "assignment_sha256": OLD_FOLD_ASSIGNMENT_SHA256,
                "status": "immutable_historical_evidence",
                "reinterpreted": False,
            },
        },
        "protocol": {
            "name": TASK_ID,
            "scope": "existing N3 1000-row development domain only",
            "class_count": CLASS_COUNT,
            "rows_per_class": ROWS_PER_CLASS,
            "pool_count": POOL_COUNT,
            "fold_count": FOLD_COUNT,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_class_per_fold": ROWS_PER_CLASS_PER_FOLD,
            "fit_rows_per_class_per_fold": FIT_ROWS_PER_CLASS_PER_FOLD,
            "algorithm": ASSIGNMENT_ALGORITHM,
            "position_identity": POSITION_IDENTITY,
            "ordering": POSITION_ORDERING,
            "tie_break": TIE_BREAK_IDENTITY,
            "round_robin_rule": "heldout_fold = within_class_rank mod 5",
            "rng_used": False,
            "assignment_sha256": fold_identity["assignment_sha256"],
            "source_index_sha256": metadata["source_indices_sha256"],
            "label_sha256": metadata["labels_sha256"],
            "folds": list(fold_identity["folds"]),
        },
        "rows": rows,
        "row_identity": {
            "development_row_positions": list(range(POOL_COUNT)),
            "source_indices_sha256": metadata["source_indices_sha256"],
            "labels_sha256": metadata["labels_sha256"],
            "q_values_used": False,
            "source_indices_complete_and_unique": True,
            "class_counts": [labels.count(label) for label in range(CLASS_COUNT)],
        },
        "fold_identity": dict(fold_identity),
        "empirical_ceiling_authority": {
            "contract": dict(old),
            "report": dict(empirical["report"]),
            "execution_source": dict(empirical["execution_source"]),
            "contract_source": dict(empirical["contract_source"]),
            "protocol_reuse": {
                "complete_row_bootstrap_with_replacement": True,
                "dimensionwise_resampling": False,
                "cross_fold_pooling": False,
                "same_bank_sizes": True,
                "same_raw_q_energy_distance": True,
                "same_model_vs_model_reference_semantics": True,
                "same_guardrails": True,
                "same_fold_compatibility_rule": True,
                "same_strict_five_of_five_rule": True,
            },
        },
        "protocol_claim": {
            "claim": "estimate producer compatibility within the observed development domain balanced over class and source-position range",
            "not_claimed": [
                "extrapolation beyond the observed source-position domain",
                "erasure or supersession of old five-fold evidence",
                "producer quality or selection",
                "sealed confirmation evidence",
            ],
        },
        "contract_totality": {
            "status": "passed",
            "validated_domains": [
                "exact 1000-row metadata identity",
                "exact class counts and source-position tie rule",
                "deterministic 20-per-class-per-fold round robin",
                "full within-class rank coverage",
                "old fold authority immutability",
                "old empirical ceiling semantics and guardrails",
                "sealed confirmation exclusion",
            ],
            "scientific_execution_requires": "one later empirical ceiling execution after this artifact is frozen and hashed",
        },
        "execution_boundary": {
            "q_values_used": False,
            "scientific_execution": False,
            "sealed_confirmation_consumed": False,
            "producer_evaluated": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_confirmation_consumed": False,
            "producer_selection": False,
        },
        "not_executed": [
            "q value loading",
            "empirical ceiling sampling",
            "scientific compatibility metrics",
            "producer model",
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
            "q_values_embedded": False,
            "existing_authorities_modified": False,
            "old_fold_authority_modified": False,
            "sealed_confirmation_modified": False,
        },
    }
    validate_protocol_payload(payload)
    return payload


def _write_artifact(repo_root: Path, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once protocol output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / ARTIFACT_NAME,
            payload,
            artifact_kind="N3 source-position-balanced development protocol",
        )
        if output_dir.exists():
            raise FileExistsError(f"write-once protocol output appeared during definition: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / ARTIFACT_NAME


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "starting committed HEAD changed")
    unexpected_dirty = [path for path in git["dirty_paths"] if path not in SAFE_DIRTY_PATHS]
    _require(not unexpected_dirty, f"unrelated local changes invalidate provenance: {unexpected_dirty}")
    _require(not output_dir.exists(), f"write-once protocol output already exists: {output_dir}")

    state = _validate_current_state(repo_root)
    stationarity_report = _validate_stationarity_report(repo_root)
    empirical = _validate_empirical_authorities(repo_root)
    source_position = _validate_source_position_authority(repo_root)
    source_indices, labels, _row_positions, metadata = _load_row_metadata(repo_root)
    fold_ids, rows, fold_identity = assign_source_position_balanced_folds(source_indices, labels)
    _require(len(fold_ids) == POOL_COUNT, "assignment size changed")
    payload = _build_payload(
        repo_root,
        git,
        state,
        stationarity_report,
        empirical,
        source_position,
        metadata,
        source_indices,
        labels,
        rows,
        fold_identity,
    )
    artifact_path = _write_artifact(repo_root, output_dir, payload)
    artifact = _read_json_exact(repo_root, output_dir / ARTIFACT_NAME, file_sha256(artifact_path), "new protocol artifact")
    validate_protocol_payload(artifact)
    return {
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "semantic_authority": TASK_ID,
        "artifact": _relative(repo_root, artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "assignment_sha256": fold_identity["assignment_sha256"],
        "git_branch": git["branch"],
        "git_starting_head": git["head"],
        "git_ending_head": _git_state(repo_root)["head"],
        "q_values_used": False,
        "scientific_execution": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
