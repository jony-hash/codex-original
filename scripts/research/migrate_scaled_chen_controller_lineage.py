"""Migrate the selected trained controller into the current v2 lineage.

The migration is deliberately state preserving.  It changes only the
checkpoint configuration binding from the historical config to the current
canonical config after proving that the two configs and controller source are
semantically identical.  The source checkpoint is fixed by contract; this
module never searches for or ranks alternative candidates.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2 import load_config
from ldm_project.v2.application.current.runtime import (
    BALANCED_TRAINING_SCHEDULE_CONTRACT,
    CONDITIONAL_CONTROL_METRIC_CONTRACT,
    IDENTITY_GENERATOR_CONTRACT,
    _load_bundle,
    _seeded_conditional_bundle,
    _system_signature,
    _validate_balanced_training_payload,
)
from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.checkpoints import (
    ConditionalCheckpointManifestV2,
    load_conditional_checkpoint_payload,
    save_conditional_checkpoint,
    state_content_hash,
    state_signature,
)


TASK_ID = "H-LOMA-GEN-S3-SPCLM"
MIGRATION_SCHEMA = "controller_lineage_migration.v1"
EXPECTED_BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
SOURCE_RELATIVE = Path(
    "outputs/20260812_v2/runs/20260812_012817_770553_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "conditional_optical_generator.pt"
)
SOURCE_COMMIT = "ae3273da6f26dd2e8dab5e6ed4dd461cd86c5dca"
SOURCE_SHA256 = "423328bf472bb7468f131b351850d1b6f7c5a282796444ecbec5e10bad12c2b9"
SOURCE_STATE_HASH = "d36165ee1d5e9c2a66734ffdea0f8291cb8699e86df667e6a66ac169cd75e2fc"
SOURCE_STATE_SIGNATURE = "88d31ec03d88a7e471d651b49c6eb20770e0fd6db5afb71798486499f62068fa"
SOURCE_CONFIG_HASH = "77db9b7fcfe47cb53f2d089c40454e811a898a329a8f694b8370b8d17d81706d"
CURRENT_CONFIG_HASH = "edeefb801b563a4657cd621c86c725ed428bbd71678cddafb93fb629eb6ee030"
CURRENT_CONFIG_FILE_SHA256 = (
    "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
)
EXPECTED_SYSTEM_SIGNATURE = (
    "f6aef9eca2bf789bbce9b0522f7c2618fd0f6adefb0f8b081c4a14c1df01ec63"
)
EXPECTED_GENERATOR_CONTRACT = "3c9855ff473de172f704d83b00637c09a05da230be4d50cd6807d86539f49e11"
EXPECTED_INITIALIZATION_STATE_HASH = (
    "07def7b8eec10f4eb99b846afa4cfd3e6bb5c50fdaa3c2a80d8dbe4d64e71545"
)
EXPECTED_TRAINING_SCHEDULE_HASH = (
    "5c8b75750714580cfa28237cd4bd15fe80615a7a0e6c618502cb142b0184d9d6"
)
EXPECTED_SPLIT_HASH = "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912"
EXPECTED_EVALUATOR_HASH = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
EXPECTED_STEPS_PER_EPOCH = 984
EXPECTED_TOTAL_STEPS = 4920

CONFIG_RELATIVE = Path(
    "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
)
GENERATOR_RELATIVE = Path("src/ldm_project/v2/student/identity_generator.py")
RUNTIME_RELATIVE = Path("src/ldm_project/v2/application/current/runtime.py")
MIGRATED_ENVELOPE_NAME = "conditional_optical_generator_current_lineage_migrated.pt"
EXPECTED_ARTIFACT_NAMES = {
    "controller_lineage_migration_manifest.json",
    "source_checkpoint_authority.json",
    "semantic_config_equivalence.json",
    "tensor_identity_audit.json",
    "runtime_binding.json",
    "strict_loader_replay.json",
    "state_replay.json",
    "historical_immutability.json",
    "compact_report.json",
    MIGRATED_ENVELOPE_NAME,
}


class MigrationError(ValueError):
    """A fail-closed migration rejection with a registered classification."""

    def __init__(self, classification: str, message: str) -> None:
        self.classification = classification
        super().__init__(f"{classification}: {message}")


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _require(condition: bool, classification: str, message: str) -> None:
    if not condition:
        raise MigrationError(classification, message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MigrationError(
            "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
            f"JSON artifact is not an object: {path}",
        )
    return payload


def _compact_training_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = payload.get("context")
    if not isinstance(context, Mapping):
        context = {}
    return {
        "schema_version": payload.get("schema_version"),
        "evidence_kind": payload.get("evidence_kind"),
        "status": payload.get("status"),
        "tier": payload.get("tier"),
        "permitted_next_actions": payload.get("permitted_next_actions"),
        "steps": payload.get("steps"),
        "input_hashes": payload.get("input_hashes"),
        "context": {
            "config_hash": context.get("config_hash"),
            "system_signature": context.get("system_signature"),
            "dataset_split_manifest_hash": context.get("dataset_split_manifest_hash"),
            "evaluator_checkpoint_hash": context.get("evaluator_checkpoint_hash"),
            "checkpoint_hash": context.get("checkpoint_hash"),
        },
    }


def _validate_source_provenance(
    source: Path, source_sha256: str, source_commit: str, repo_root: Path
) -> dict[str, Any]:
    run_root = source.parent
    run_meta_path = run_root / "run_meta.json"
    evidence_path = run_root / "conditional_training_evidence.json"
    _require(
        run_meta_path.is_file() and evidence_path.is_file(),
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "selected source run lacks fixed run metadata or training evidence",
    )
    run_meta = _read_json(run_meta_path)
    evidence = _read_json(evidence_path)
    _require(
        run_meta.get("git_commit") == source_commit,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source run commit does not match the registered source commit",
    )
    _require(
        run_meta.get("config_hash") == SOURCE_CONFIG_HASH
        and run_meta.get("dirty_patch_hash") is None,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source run metadata is not the registered clean historical lineage",
    )
    input_hashes = evidence.get("input_hashes")
    context = evidence.get("context")
    _require(
        evidence.get("evidence_kind") == "conditional_training"
        and evidence.get("status") == "passed"
        and evidence.get("tier") == "acceptance"
        and isinstance(input_hashes, Mapping)
        and isinstance(context, Mapping),
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source training evidence is not a passed acceptance record",
    )
    _require(
        context.get("config_hash") == SOURCE_CONFIG_HASH
        and context.get("checkpoint_hash") == source_sha256
        and context.get("system_signature") == EXPECTED_SYSTEM_SIGNATURE
        and context.get("dataset_split_manifest_hash") == EXPECTED_SPLIT_HASH
        and context.get("evaluator_checkpoint_hash") == EXPECTED_EVALUATOR_HASH,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source training evidence context does not bind the selected source",
    )
    _require(
        input_hashes.get("initialization_state_hash")
        == EXPECTED_INITIALIZATION_STATE_HASH
        and input_hashes.get("training_schedule_hash") == EXPECTED_TRAINING_SCHEDULE_HASH,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source training evidence lineage hashes do not match the registration",
    )
    return {
        "run_meta_path": _repo_relative(run_meta_path, repo_root),
        "evidence_path": _repo_relative(evidence_path, repo_root),
        "run_meta_sha256": file_sha256(run_meta_path),
        "evidence_sha256": file_sha256(evidence_path),
        "run_meta": {
            "schema_version": run_meta.get("schema_version"),
            "config_hash": run_meta.get("config_hash"),
            "git_commit": run_meta.get("git_commit"),
            "dirty_patch_hash": run_meta.get("dirty_patch_hash"),
        },
        "training_evidence": _compact_training_evidence(evidence),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def _semantic_config_equivalence(repo_root: Path, config_path: Path) -> dict[str, Any]:
    old_yaml = _git(repo_root, "show", f"{SOURCE_COMMIT}:{CONFIG_RELATIVE.as_posix()}")
    current_yaml = config_path.read_text(encoding="utf-8")
    old_payload = yaml.safe_load(old_yaml)
    current_payload = yaml.safe_load(current_yaml)
    _require(
        isinstance(old_payload, Mapping) and isinstance(current_payload, Mapping),
        "CONTROLLER_LINEAGE_MIGRATION_CONFIG_EQUIVALENCE_INVALID",
        "historical or current canonical config is not a YAML mapping",
    )
    old_flat = _flatten(old_payload)
    current_flat = _flatten(current_payload)
    fields = sorted(set(old_flat) | set(current_flat))
    differences = [
        field
        for field in fields
        if old_flat.get(field) != current_flat.get(field)
    ]
    old_generator = _git(repo_root, "show", f"{SOURCE_COMMIT}:{GENERATOR_RELATIVE.as_posix()}")
    current_generator = (repo_root / GENERATOR_RELATIVE).read_text(encoding="utf-8")
    result = {
        "schema": "semantic_config_equivalence.v1",
        "historical_config_hash": SOURCE_CONFIG_HASH,
        "current_config_hash": CURRENT_CONFIG_HASH,
        "current_config_file_sha256": file_sha256(config_path),
        "yaml_field_count": len(fields),
        "field_diff_count": len(differences),
        "field_differences": differences,
        "controller_source_path": GENERATOR_RELATIVE.as_posix(),
        "historical_controller_source_sha256": canonical_sha256(old_generator),
        "current_controller_source_sha256": canonical_sha256(current_generator),
        "controller_source_content_equal": old_generator == current_generator,
        "system_signature_equal": True,
        "classification": "NONSEMANTIC_LINEAGE_PACKAGING_DRIFT",
    }
    _require(
        result["current_config_file_sha256"] == CURRENT_CONFIG_FILE_SHA256
        and len(differences) == 0
        and result["controller_source_content_equal"],
        "CONTROLLER_LINEAGE_MIGRATION_CONFIG_EQUIVALENCE_INVALID",
        "historical and current controller/config semantics are not equivalent",
    )
    return result


def _state_inventory(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    tensor_hashes = {
        key: tensor_sha256(value) for key, value in sorted(state.items())
    }
    return {
        "key_count": len(state),
        "keys": sorted(state),
        "shape_dtype": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(state.items())
        },
        "tensor_sha256": tensor_hashes,
        "state_signature": state_signature(state),
        "state_content_hash": state_content_hash(state),
        "parameter_count": sum(value.numel() for value in state.values()),
    }


def _require_state_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    classification: str,
) -> None:
    _require(
        sorted(left) == sorted(right),
        classification,
        "state-dict keys changed during migration",
    )
    for key in sorted(left):
        _require(
            left[key].shape == right[key].shape
            and left[key].dtype == right[key].dtype
            and torch.equal(left[key], right[key])
            and tensor_sha256(left[key]) == tensor_sha256(right[key]),
            classification,
            f"state tensor changed during migration: {key}",
        )


def _strict_load(
    config: Any, envelope_path: Path, envelope_sha256: str
) -> tuple[torch.nn.Module, ConditionalCheckpointManifestV2]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = _load_bundle(
            config,
            str(envelope_path),
            device=torch.device("cpu"),
            allow_balanced_schedule=True,
            checkpoint_sha256=envelope_sha256,
        )
    _require(
        not caught,
        "CONTROLLER_LINEAGE_MIGRATION_RUNTIME_BINDING_FAILED",
        "strict current loader emitted a warning",
    )
    return loaded


def _manifest_dict(manifest: ConditionalCheckpointManifestV2) -> dict[str, Any]:
    return manifest.to_dict()


def _write_json(staging: Path, name: str, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(staging / name, payload, artifact_kind=name)


def _audit_tree(directory: Path) -> None:
    names = {path.name for path in directory.iterdir() if path.is_file()}
    _require(
        names == EXPECTED_ARTIFACT_NAMES,
        "CONTROLLER_LINEAGE_MIGRATION_OUTPUT_FAILURE",
        f"migration output tree differs from the registered artifact set: {sorted(names)}",
    )
    _require(
        not any(
            path.name.startswith(".")
            or path.name.endswith(".tmp")
            or path.suffix.lower() == ".tiff"
            for path in directory.iterdir()
        ),
        "CONTROLLER_LINEAGE_MIGRATION_OUTPUT_FAILURE",
        "migration output contains a hidden, temporary, or TIFF artifact",
    )


def migrate_controller_state(
    source: str | Path,
    output_dir: str | Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create and validate the immutable state-preserving migration envelope."""
    root = _repo_root(repo_root)
    source_path = _resolve_path(source, root)
    output_path = _resolve_path(output_dir, root)
    expected_source = (root / SOURCE_RELATIVE).resolve()
    _require(
        source_path == expected_source,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source must be the fixed registered trained-final checkpoint",
    )
    _require(
        not output_path.exists(),
        "CONTROLLER_LINEAGE_MIGRATION_OUTPUT_FAILURE",
        f"write-once migration output already exists: {output_path}",
    )
    branch = _git(root, "branch", "--show-current")
    current_head = _git(root, "rev-parse", "HEAD")
    _require(
        branch == EXPECTED_BRANCH,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        f"unexpected branch: {branch}",
    )
    _require(
        not _git(root, "status", "--porcelain"),
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "migration requires a clean worktree before publishing authority",
    )
    _require(
        source_path.is_file(),
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        f"fixed source checkpoint is missing: {source_path}",
    )
    source_before = {
        "path": SOURCE_RELATIVE.as_posix(),
        "size_bytes": source_path.stat().st_size,
        "sha256": file_sha256(source_path),
    }
    _require(
        source_before["sha256"] == SOURCE_SHA256,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "fixed source checkpoint bytes do not match the registered authority",
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, current_head],
        cwd=root,
        capture_output=True,
    )
    _require(
        ancestry.returncode == 0,
        "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
        "source training commit is not an ancestor of the current runtime HEAD",
    )

    config_path = root / CONFIG_RELATIVE
    config = load_config(config_path)
    current_config_hash = canonical_sha256(config.model_dump(mode="json"))
    _require(
        current_config_hash == CURRENT_CONFIG_HASH
        and file_sha256(config_path) == CURRENT_CONFIG_FILE_SHA256,
        "CONTROLLER_LINEAGE_MIGRATION_CONFIG_EQUIVALENCE_INVALID",
        "current canonical formal config does not match the registered identity",
    )
    system_signature = _system_signature(config)
    _require(
        system_signature == EXPECTED_SYSTEM_SIGNATURE,
        "CONTROLLER_LINEAGE_MIGRATION_CONFIG_EQUIVALENCE_INVALID",
        "current system signature does not match the registered identity",
    )
    settings = config.conditional_generation
    _require(
        settings is not None
        and settings.code_dim == 8
        and settings.hidden_dim == 64
        and settings.command_gain == 0.5
        and IDENTITY_GENERATOR_CONTRACT == EXPECTED_GENERATOR_CONTRACT,
        "CONTROLLER_LINEAGE_MIGRATION_CONFIG_EQUIVALENCE_INVALID",
        "current controller architecture does not match the registered identity",
    )
    equivalence = _semantic_config_equivalence(root, config_path)
    provenance = _validate_source_provenance(
        source_path, SOURCE_SHA256, SOURCE_COMMIT, root
    )
    payload = safe_torch_load(source_path, expected_sha256=SOURCE_SHA256)
    _require(
        isinstance(payload, dict)
        and isinstance(payload.get("manifest"), dict)
        and isinstance(payload.get("state_dict"), dict),
        "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
        "fixed source is not a valid conditional checkpoint payload",
    )
    source_manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    source_state = payload["state_dict"]
    source_inventory = _state_inventory(source_state)
    _require(
        source_manifest.config_hash == SOURCE_CONFIG_HASH
        and source_manifest.system_signature == EXPECTED_SYSTEM_SIGNATURE
        and source_manifest.architecture_kind == "conditional_identity_logit_amplitude_v2"
        and source_manifest.generator_contract_hash == EXPECTED_GENERATOR_CONTRACT
        and source_manifest.metric_contract_hash
        == canonical_sha256(CONDITIONAL_CONTROL_METRIC_CONTRACT)
        and source_manifest.training_schedule_hash == EXPECTED_TRAINING_SCHEDULE_HASH
        and source_manifest.initialization_seed == 1234
        and source_manifest.initialization_state_hash == EXPECTED_INITIALIZATION_STATE_HASH
        and source_manifest.dataset_split_manifest_hash == EXPECTED_SPLIT_HASH
        and source_manifest.evaluator_checkpoint_hash == EXPECTED_EVALUATOR_HASH
        and source_inventory["state_content_hash"] == SOURCE_STATE_HASH
        and source_inventory["state_signature"] == SOURCE_STATE_SIGNATURE,
        "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
        "source checkpoint manifest or state identity does not match registration",
    )

    source_bundle = _seeded_conditional_bundle(config)
    load_conditional_checkpoint_payload(payload, source_bundle, source_manifest)
    loaded_source_state = source_bundle.state_dict()
    _require_state_equal(
        source_state,
        loaded_source_state,
        "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
    )

    migrated_manifest = replace(source_manifest, config_hash=CURRENT_CONFIG_HASH)
    migrated_extra = copy.deepcopy(payload.get("extra", {}))
    _require(
        isinstance(migrated_extra, dict),
        "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
        "source checkpoint extra metadata is not a mapping",
    )
    runtime_contract = {
        "entrypoint": "ldm_project.v2.application.current.runtime._load_bundle",
        "checkpoint_schema": "ConditionalCheckpointManifestV2",
        "architecture_kind": migrated_manifest.architecture_kind,
        "controller_class": "IdentityAnchoredConditionalAmplitudeGenerator",
        "code_dim": 8,
        "hidden_dim": 64,
        "command_gain": 0.5,
        "generator_contract_hash": EXPECTED_GENERATOR_CONTRACT,
        "system_signature": EXPECTED_SYSTEM_SIGNATURE,
        "config_hash": CURRENT_CONFIG_HASH,
        "balanced_schedule_allowed": True,
        "balanced_schedule_contract": BALANCED_TRAINING_SCHEDULE_CONTRACT,
        "reason": "existing formal evaluation path consumes the registered balanced final schedule",
    }
    migration_binding = {
        "schema": MIGRATION_SCHEMA,
        "task_id": TASK_ID,
        "source_checkpoint": SOURCE_RELATIVE.as_posix(),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_state_content_hash": SOURCE_STATE_HASH,
        "source_state_signature": SOURCE_STATE_SIGNATURE,
        "source_commit": SOURCE_COMMIT,
        "historical_config_hash": SOURCE_CONFIG_HASH,
        "current_config_hash": CURRENT_CONFIG_HASH,
        "current_config_file_sha256": CURRENT_CONFIG_FILE_SHA256,
        "current_runtime_commit": current_head,
        "current_runtime_module_sha256": file_sha256(root / RUNTIME_RELATIVE),
        "current_runtime_contract_hash": canonical_sha256(runtime_contract),
        "rationale": "bind the unchanged historical trained-final state to the semantically equivalent current formal runtime",
        "state_preserving": True,
        "parameter_mutation": False,
        "migration_authority": "controller_lineage_migration_manifest.json",
    }
    migrated_extra["controller_lineage_migration"] = migration_binding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.", dir=output_path.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        envelope_path = staging / MIGRATED_ENVELOPE_NAME
        save_conditional_checkpoint(
            envelope_path,
            source_bundle,
            migrated_manifest,
            extra=migrated_extra,
        )
        migrated_sha256 = file_sha256(envelope_path)
        migrated_payload = safe_torch_load(
            envelope_path, expected_sha256=migrated_sha256
        )
        migrated_bundle = _seeded_conditional_bundle(config)
        load_conditional_checkpoint_payload(
            migrated_payload, migrated_bundle, migrated_manifest
        )
        _validate_balanced_training_payload(
            migrated_payload,
            steps_per_epoch=EXPECTED_STEPS_PER_EPOCH,
            total_steps=EXPECTED_TOTAL_STEPS,
        )
        _require_state_equal(
            source_state,
            migrated_payload["state_dict"],
            "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
        )
        migrated_inventory = _state_inventory(migrated_payload["state_dict"])
        _require(
            migrated_inventory == source_inventory,
            "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
            "migrated envelope does not preserve the source tensor identity audit",
        )

        strict_records: list[dict[str, Any]] = []
        replay_inventories: list[dict[str, Any]] = []
        for attempt in range(1, 3):
            strict_bundle, strict_manifest = _strict_load(
                config, envelope_path, migrated_sha256
            )
            strict_state = strict_bundle.state_dict()
            strict_inventory = _state_inventory(strict_state)
            _require(
                strict_manifest == migrated_manifest,
                "CONTROLLER_LINEAGE_MIGRATION_RUNTIME_BINDING_FAILED",
                "strict current loader returned a different manifest",
            )
            _require_state_equal(
                source_state,
                strict_state,
                "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
            )
            replay_inventories.append(strict_inventory)
            strict_records.append(
                {
                    "attempt": attempt,
                    "result": "PASS",
                    "device": "cpu",
                    "loader": "ldm_project.v2.application.current.runtime._load_bundle",
                    "strict_state_load": True,
                    "config_hash_override": False,
                    "checkpoint_sha256_verified": True,
                    "allow_balanced_schedule": True,
                    "state_content_hash": strict_inventory["state_content_hash"],
                    "state_signature": strict_inventory["state_signature"],
                }
            )

        _require(
            replay_inventories[0] == replay_inventories[1]
            and replay_inventories[0] == source_inventory,
            "CONTROLLER_LINEAGE_MIGRATION_STATE_IDENTITY_FAILED",
            "independent strict loader replays are not identical to source state",
        )

        source_after = {
            "path": SOURCE_RELATIVE.as_posix(),
            "size_bytes": source_path.stat().st_size,
            "sha256": file_sha256(source_path),
        }
        source_after_payload = safe_torch_load(
            source_path, expected_sha256=SOURCE_SHA256
        )
        source_after_inventory = _state_inventory(source_after_payload["state_dict"])
        _require(
            source_after == source_before
            and source_after_inventory == source_inventory,
            "CONTROLLER_LINEAGE_MIGRATION_PROVENANCE_INCOMPLETE",
            "historical source changed during migration",
        )

        source_authority = {
            "schema": "controller_lineage_migration_source_authority.v1",
            "task_id": TASK_ID,
            "selection_authority": "fixed registered trained-final source; no candidate search",
            "source": source_before,
            "source_commit": SOURCE_COMMIT,
            "source_manifest": _manifest_dict(source_manifest),
            "source_state": source_inventory,
            "provenance": provenance,
        }
        tensor_audit = {
            "schema": "tensor_identity_audit.v1",
            "source": source_inventory,
            "migrated": migrated_inventory,
            "state_content_hash_equal": True,
            "state_signature_equal": True,
            "keys_shapes_dtypes_bytes_equal": True,
            "parameter_mutation": False,
        }
        runtime_binding = {
            "schema": "runtime_binding.v1",
            "loader": runtime_contract,
            "current_head": current_head,
            "current_config_path": CONFIG_RELATIVE.as_posix(),
            "current_config_hash": CURRENT_CONFIG_HASH,
            "migrated_checkpoint": MIGRATED_ENVELOPE_NAME,
            "migrated_checkpoint_sha256": migrated_sha256,
            "manifest_config_hash": migrated_manifest.config_hash,
            "manifest_system_signature": migrated_manifest.system_signature,
            "strict_current_loader": "PASS",
            "governance": {
                "generative_manifold_supported": False,
                "generation_readiness": "not_ready",
                "training_authorized": False,
                "formal_promotion": False,
            },
        }
        strict_replay = {
            "schema": "strict_loader_replay.v1",
            "result": "PASS",
            "attempts": strict_records,
            "normal_current_loader": True,
            "config_hash_override_used": False,
            "manual_state_remap": False,
            "forward": False,
            "optimizer": False,
            "backward": False,
            "training": False,
        }
        state_replay = {
            "schema": "state_replay.v1",
            "result": "PASS",
            "replay_count": len(replay_inventories),
            "replays": replay_inventories,
            "matches_source": True,
            "matches_each_other": True,
        }
        immutability = {
            "schema": "historical_immutability.v1",
            "result": "PASS",
            "source_before": source_before,
            "source_after": source_after,
            "state_before": source_inventory,
            "state_after": source_after_inventory,
            "bytes_unchanged": True,
            "state_unchanged": True,
            "source_overwritten": False,
        }
        compact_report = {
            "schema": "controller_lineage_migration_compact_report.v1",
            "task_id": TASK_ID,
            "classification": "NONSEMANTIC_LINEAGE_PACKAGING_DRIFT",
            "gate": "CONTROLLER_LINEAGE_MIGRATION_PASS",
            "source": SOURCE_RELATIVE.as_posix(),
            "migrated_envelope": MIGRATED_ENVELOPE_NAME,
            "state_origin": "HISTORICAL_TRAINED_STATE_MIGRATED_WITHOUT_PARAMETER_CHANGE",
            "current_formal_controller_state": "ESTABLISHED_BY_MIGRATION",
            "strict_loader_replay": "PASS",
            "historical_immutability": "PASS",
            "formal_promotion": False,
            "training_authorized": False,
            "next_action": "RERUN_H-LOMA-GEN-S3-CSAR",
        }
        _write_json(staging, "source_checkpoint_authority.json", source_authority)
        _write_json(staging, "semantic_config_equivalence.json", equivalence)
        _write_json(staging, "tensor_identity_audit.json", tensor_audit)
        _write_json(staging, "runtime_binding.json", runtime_binding)
        _write_json(staging, "strict_loader_replay.json", strict_replay)
        _write_json(staging, "state_replay.json", state_replay)
        _write_json(staging, "historical_immutability.json", immutability)
        _write_json(staging, "compact_report.json", compact_report)

        artifact_hashes = {
            name: file_sha256(staging / name)
            for name in sorted(EXPECTED_ARTIFACT_NAMES - {"controller_lineage_migration_manifest.json"})
        }
        manifest_without_hash = {
            "schema": MIGRATION_SCHEMA,
            "task_id": TASK_ID,
            "gate": "CONTROLLER_LINEAGE_MIGRATION_PASS",
            "classification": "NONSEMANTIC_LINEAGE_PACKAGING_DRIFT",
            "source_checkpoint": source_before,
            "source_commit": SOURCE_COMMIT,
            "source_state_content_hash": SOURCE_STATE_HASH,
            "source_state_signature": SOURCE_STATE_SIGNATURE,
            "historical_config_hash": SOURCE_CONFIG_HASH,
            "current_config_path": CONFIG_RELATIVE.as_posix(),
            "current_config_hash": CURRENT_CONFIG_HASH,
            "current_config_file_sha256": CURRENT_CONFIG_FILE_SHA256,
            "current_runtime_commit": current_head,
            "current_runtime_module_sha256": migration_binding["current_runtime_module_sha256"],
            "controller_contract_hash": EXPECTED_GENERATOR_CONTRACT,
            "migration_rationale": migration_binding["rationale"],
            "migrated_checkpoint": {
                "path": MIGRATED_ENVELOPE_NAME,
                "sha256": migrated_sha256,
                "state_content_hash": migrated_manifest.state_content_hash,
                "state_signature": migrated_manifest.state_signature,
                "manifest": _manifest_dict(migrated_manifest),
            },
            "state_preserving": True,
            "parameter_mutation": False,
            "semantic_equivalence": {
                "field_diff_count": equivalence["field_diff_count"],
                "controller_source_content_equal": equivalence[
                    "controller_source_content_equal"
                ],
                "classification": equivalence["classification"],
            },
            "runtime_binding": runtime_contract,
            "artifact_hashes": artifact_hashes,
            "artifacts": sorted(EXPECTED_ARTIFACT_NAMES),
            "state_origin": "HISTORICAL_TRAINED_STATE_MIGRATED_WITHOUT_PARAMETER_CHANGE",
            "next_action": "RERUN_H-LOMA-GEN-S3-CSAR",
            "governance": {
                "mainline_state": "closed_at_current_evidence_boundary",
                "reconstruction_manifold_supported": False,
                "generative_manifold_supported": False,
                "generation_readiness": "not_ready",
                "training_authorized": False,
                "formal_promotion": False,
            },
        }
        manifest = dict(manifest_without_hash)
        manifest["manifest_hash"] = canonical_sha256(manifest_without_hash)
        _write_json(staging, "controller_lineage_migration_manifest.json", manifest)
        _audit_tree(staging)
        loaded_manifest = _read_json(staging / "controller_lineage_migration_manifest.json")
        recorded_hash = loaded_manifest.pop("manifest_hash", None)
        _require(
            isinstance(recorded_hash, str)
            and recorded_hash == canonical_sha256(loaded_manifest),
            "CONTROLLER_LINEAGE_MIGRATION_OUTPUT_FAILURE",
            "migration manifest self-hash verification failed",
        )
        staging.replace(output_path)

    _audit_tree(output_path)
    final_manifest = _read_json(output_path / "controller_lineage_migration_manifest.json")
    _require(
        final_manifest.get("migrated_checkpoint", {}).get("sha256")
        == file_sha256(output_path / MIGRATED_ENVELOPE_NAME)
        and final_manifest.get("manifest_hash")
        == canonical_sha256(
            {key: value for key, value in final_manifest.items() if key != "manifest_hash"}
        ),
        "CONTROLLER_LINEAGE_MIGRATION_OUTPUT_FAILURE",
        "published migration manifest or envelope hash does not verify",
    )
    return {
        "output_dir": str(output_path),
        "manifest": final_manifest,
        "source": source_before,
        "migrated_checkpoint_sha256": migrated_sha256,
        "gate": "CONTROLLER_LINEAGE_MIGRATION_PASS",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="fixed registered source checkpoint")
    parser.add_argument("--output-dir", required=True, help="new write-once migration output directory")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    result = migrate_controller_state(arguments.source, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
