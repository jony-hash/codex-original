"""Register the migrated Scaled-Chen controller as the current state authority.

This command is intentionally read-only with respect to the migrated
checkpoint.  It validates the existing migration authority, loads that one
explicit envelope through the current strict loader twice, checks that no
second current-config-bound controller checkpoint exists, and publishes only
JSON authority records.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2 import load_config
from ldm_project.v2.application.current.runtime import (
    IDENTITY_GENERATOR_CONTRACT,
    _load_bundle,
    _system_signature,
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
    state_content_hash,
    state_signature,
)


TASK_ID = "H-LOMA-GEN-S3-CSAR"
CURRENT_CONFIG_FILE_SHA256 = (
    "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
)
CURRENT_CONFIG_HASH = "edeefb801b563a4657cd621c86c725ed428bbd71678cddafb93fb629eb6ee030"
EXPECTED_GENERATOR_CONTRACT = "3c9855ff473de172f704d83b00637c09a05da230be4d50cd6807d86539f49e11"
EXPECTED_SYSTEM_SIGNATURE = (
    "f6aef9eca2bf789bbce9b0522f7c2618fd0f6adefb0f8b081c4a14c1df01ec63"
)
MIGRATED_ENVELOPE_NAME = "conditional_optical_generator_current_lineage_migrated.pt"
SOURCE_RELATIVE = Path(
    "outputs/20260812_v2/runs/20260812_012817_770553_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "conditional_optical_generator.pt"
)
SOURCE_SHA256 = "423328bf472bb7468f131b351850d1b6f7c5a282796444ecbec5e10bad12c2b9"
SOURCE_STATE_HASH = "d36165ee1d5e9c2a66734ffdea0f8291cb8699e86df667e6a66ac169cd75e2fc"
SOURCE_STATE_SIGNATURE = "88d31ec03d88a7e471d651b49c6eb20770e0fd6db5afb71798486499f62068fa"
CONFIG_RELATIVE = Path(
    "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
)
MIGRATION_RELATIVE = Path(
    "outputs/20260906_v2/runs/lightgen_controller_lineage_migration_v1_attempt_01"
)
AUTHORITY_RELATIVE = Path(
    "outputs/20260906_v2/runs/lightgen_controller_state_authority_v1_attempt_01"
)
MIGRATION_MANIFEST_NAME = "controller_lineage_migration_manifest.json"
MIGRATION_MANIFEST_SHA256 = (
    "f41f2ac04462b8be8be77f20833e6dadc21133f2473244143bb32eda4cddd212"
)
MIGRATION_MANIFEST_SELF_HASH = (
    "94b9e9cfc775d089e759627c16e3d829c45993861ca699618b8900c0bfed0569"
)
AUTHORITY_SCHEMA = "controller_state_authority.v1"
EXPECTED_AUTHORITY_NAMES = {
    "controller_state_authority.json",
    "runtime_binding.json",
    "migration_binding.json",
    "provenance_lineage.json",
    "state_replay.json",
    "state_isolation.json",
    "compact_report.json",
}
FORMAL_CONTROLLER_CHECKPOINT_NAMES = {
    "conditional_optical_generator.pt",
    "conditional_optical_generator_balanced.pt",
    MIGRATED_ENVELOPE_NAME,
}


class AuthorityError(ValueError):
    """A fail-closed CSAR rejection with its registered classification."""

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


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _require(condition: bool, classification: str, message: str) -> None:
    if not condition:
        raise AuthorityError(classification, message)


def _read_json(path: Path, classification: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict),
        classification,
        f"JSON authority is not an object: {path}",
    )
    return payload


def _state_inventory(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "key_count": len(state),
        "keys": sorted(state),
        "shape_dtype": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(state.items())
        },
        "tensor_sha256": {
            key: tensor_sha256(value) for key, value in sorted(state.items())
        },
        "state_content_hash": state_content_hash(state),
        "state_signature": state_signature(state),
        "parameter_count": sum(value.numel() for value in state.values()),
    }


def _audit_migration_authority(
    migration_dir: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _require(
        migration_dir.is_dir(),
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        f"migration authority directory is missing: {migration_dir}",
    )
    names = {path.name for path in migration_dir.iterdir() if path.is_file()}
    expected_names = {
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
    _require(
        names == expected_names
        and not any(
            path.name.startswith(".")
            or path.name.endswith(".tmp")
            or path.suffix.lower() == ".tiff"
            for path in migration_dir.iterdir()
        ),
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        "migration authority tree contains unexpected or temporary artifacts",
    )
    manifest_path = migration_dir / MIGRATION_MANIFEST_NAME
    _require(
        file_sha256(manifest_path) == MIGRATION_MANIFEST_SHA256,
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        "migration manifest file hash does not match the registered authority",
    )
    manifest = _read_json(
        manifest_path, "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID"
    )
    recorded_hash = manifest.pop("manifest_hash", None)
    _require(
        isinstance(recorded_hash, str)
        and recorded_hash == MIGRATION_MANIFEST_SELF_HASH
        and recorded_hash == canonical_sha256(manifest),
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        "migration manifest self-hash is invalid",
    )
    manifest["manifest_hash"] = recorded_hash
    artifact_hashes = manifest.get("artifact_hashes")
    _require(
        isinstance(artifact_hashes, Mapping)
        and set(artifact_hashes) == expected_names - {MIGRATION_MANIFEST_NAME}
        and all(
            isinstance(value, str)
            and file_sha256(migration_dir / name) == value
            for name, value in artifact_hashes.items()
        ),
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        "migration artifact tree hash binding is invalid",
    )
    migrated = manifest.get("migrated_checkpoint")
    semantic = manifest.get("semantic_equivalence")
    _require(
        manifest.get("gate") == "CONTROLLER_LINEAGE_MIGRATION_PASS"
        and manifest.get("state_preserving") is True
        and manifest.get("parameter_mutation") is False
        and manifest.get("source_checkpoint", {}).get("path")
        == SOURCE_RELATIVE.as_posix()
        and manifest.get("source_checkpoint", {}).get("sha256") == SOURCE_SHA256
        and manifest.get("source_state_content_hash") == SOURCE_STATE_HASH
        and manifest.get("source_state_signature") == SOURCE_STATE_SIGNATURE
        and manifest.get("current_config_hash") == CURRENT_CONFIG_HASH
        and manifest.get("current_config_file_sha256") == CURRENT_CONFIG_FILE_SHA256
        and manifest.get("controller_contract_hash") == EXPECTED_GENERATOR_CONTRACT
        and isinstance(migrated, Mapping)
        and migrated.get("path") == MIGRATED_ENVELOPE_NAME
        and migrated.get("sha256")
        == file_sha256(migration_dir / MIGRATED_ENVELOPE_NAME)
        and migrated.get("state_content_hash") == SOURCE_STATE_HASH
        and migrated.get("state_signature") == SOURCE_STATE_SIGNATURE
        and isinstance(semantic, Mapping)
        and semantic.get("field_diff_count") == 0
        and semantic.get("classification")
        == "NONSEMANTIC_LINEAGE_PACKAGING_DRIFT",
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        "migration manifest has an invalid source, state, or equivalence binding",
    )
    sidecars = {
        name: _read_json(
            migration_dir / name, "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID"
        )
        for name in (
            "tensor_identity_audit.json",
            "runtime_binding.json",
            "strict_loader_replay.json",
            "historical_immutability.json",
        )
    }
    tensor_audit = sidecars["tensor_identity_audit.json"]
    runtime_binding = sidecars["runtime_binding.json"]
    strict_replay = sidecars["strict_loader_replay.json"]
    immutability = sidecars["historical_immutability.json"]
    _require(
        tensor_audit.get("state_content_hash_equal") is True
        and tensor_audit.get("state_signature_equal") is True
        and tensor_audit.get("keys_shapes_dtypes_bytes_equal") is True
        and tensor_audit.get("parameter_mutation") is False
        and runtime_binding.get("strict_current_loader") == "PASS"
        and runtime_binding.get("current_config_hash") == CURRENT_CONFIG_HASH
        and strict_replay.get("result") == "PASS"
        and strict_replay.get("config_hash_override_used") is False
        and strict_replay.get("manual_state_remap") is False
        and immutability.get("result") == "PASS"
        and immutability.get("bytes_unchanged") is True
        and immutability.get("state_unchanged") is True,
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
        "one or more migration gate sidecars is not a PASS authority",
    )
    envelope_path = migration_dir / MIGRATED_ENVELOPE_NAME
    return manifest, sidecars, envelope_path


def _strict_load(
    config: Any, checkpoint: Path, checkpoint_sha256: str
) -> tuple[torch.nn.Module, ConditionalCheckpointManifestV2]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = _load_bundle(
            config,
            str(checkpoint),
            device=torch.device("cpu"),
            allow_balanced_schedule=True,
            checkpoint_sha256=checkpoint_sha256,
        )
    _require(
        not caught,
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        "current strict loader emitted a warning",
    )
    return loaded


def _discover_current_bindings(
    repo_root: Path, envelope_path: Path
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    outputs = repo_root / "outputs"
    if not outputs.is_dir():
        return bindings
    for path in sorted(outputs.rglob("*.pt")):
        if path.name not in FORMAL_CONTROLLER_CHECKPOINT_NAMES:
            continue
        payload = safe_torch_load(path)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("manifest"), dict
        ):
            raise AuthorityError(
                "CONTROLLER_STATE_AUTHORITY_AMBIGUOUS",
                f"formal controller checkpoint has no valid manifest: {path}",
            )
        checkpoint_manifest = payload["manifest"]
        if (
            checkpoint_manifest.get("config_hash") == CURRENT_CONFIG_HASH
            and checkpoint_manifest.get("architecture_kind")
            == "conditional_identity_logit_amplitude_v2"
        ):
            bindings.append(
                {
                    "path": _relative(path, repo_root),
                    "sha256": file_sha256(path),
                    "state_content_hash": checkpoint_manifest.get("state_content_hash"),
                    "state_signature": checkpoint_manifest.get("state_signature"),
                    "is_selected_envelope": path.resolve() == envelope_path.resolve(),
                }
            )
    return bindings


def _write_json(staging: Path, name: str, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(staging / name, payload, artifact_kind=name)


def _audit_authority_tree(directory: Path) -> None:
    names = {path.name for path in directory.iterdir() if path.is_file()}
    _require(
        names == EXPECTED_AUTHORITY_NAMES,
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        "CSAR output tree differs from the registered authority artifact set",
    )
    _require(
        not any(
            path.name.startswith(".")
            or path.name.endswith(".tmp")
            or path.suffix.lower() == ".tiff"
            for path in directory.iterdir()
        ),
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        "CSAR output contains a hidden, temporary, or TIFF artifact",
    )


def register_controller_state_authority(
    output_dir: str | Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Publish the unique migrated controller state authority."""
    root = _resolve_repo_root(repo_root)
    output_path = _resolve_path(output_dir, root)
    _require(
        not output_path.exists(),
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        f"refusing to overwrite write-once CSAR output: {output_path}",
    )
    current_head = _git(root, "rev-parse", "HEAD")
    migration_dir = root / MIGRATION_RELATIVE
    migration_manifest, migration_sidecars, migration_envelope = (
        _audit_migration_authority(migration_dir, root)
    )
    config_path = root / CONFIG_RELATIVE
    config = load_config(config_path)
    _require(
        canonical_sha256(config.model_dump(mode="json")) == CURRENT_CONFIG_HASH
        and file_sha256(config_path) == CURRENT_CONFIG_FILE_SHA256
        and _system_signature(config) == EXPECTED_SYSTEM_SIGNATURE,
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        "current formal config/system binding is not registered",
    )
    settings = config.conditional_generation
    _require(
        settings is not None
        and settings.code_dim == 8
        and settings.hidden_dim == 64
        and settings.command_gain == 0.5
        and IDENTITY_GENERATOR_CONTRACT == EXPECTED_GENERATOR_CONTRACT,
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        "current controller architecture contract is not registered",
    )
    envelope_sha256 = file_sha256(migration_envelope)
    _require(
        envelope_sha256
        == migration_manifest["migrated_checkpoint"]["sha256"]
        == "572116dbb88464d0999a0391e7e94675cacd64ad5ae303bc2e0efb16e1f63ba2",
        "CONTROLLER_STATE_AUTHORITY_IDENTITY_MISMATCH",
        "migrated envelope SHA does not match migration authority",
    )
    payload = safe_torch_load(migration_envelope, expected_sha256=envelope_sha256)
    _require(
        isinstance(payload, dict)
        and isinstance(payload.get("manifest"), dict)
        and isinstance(payload.get("state_dict"), dict),
        "CONTROLLER_STATE_AUTHORITY_IDENTITY_MISMATCH",
        "migrated envelope payload is invalid",
    )
    envelope_manifest = ConditionalCheckpointManifestV2(**payload["manifest"])
    inventory = _state_inventory(payload["state_dict"])
    _require(
        envelope_manifest.config_hash == CURRENT_CONFIG_HASH
        and envelope_manifest.architecture_kind
        == "conditional_identity_logit_amplitude_v2"
        and envelope_manifest.generator_contract_hash == EXPECTED_GENERATOR_CONTRACT
        and envelope_manifest.state_content_hash == SOURCE_STATE_HASH
        and envelope_manifest.state_signature == SOURCE_STATE_SIGNATURE
        and inventory["state_content_hash"] == SOURCE_STATE_HASH
        and inventory["state_signature"] == SOURCE_STATE_SIGNATURE,
        "CONTROLLER_STATE_AUTHORITY_IDENTITY_MISMATCH",
        "migrated envelope state or architecture identity differs from migration",
    )

    replay_inventories: list[dict[str, Any]] = []
    for _ in range(2):
        bundle, loaded_manifest = _strict_load(
            config, migration_envelope, envelope_sha256
        )
        loaded_inventory = _state_inventory(bundle.state_dict())
        _require(
            loaded_manifest == envelope_manifest
            and loaded_inventory == inventory,
            "CONTROLLER_STATE_REPLAY_INVALID",
            "strict replay does not match the migrated state inventory",
        )
        replay_inventories.append(loaded_inventory)
    _require(
        replay_inventories[0] == replay_inventories[1] == inventory,
        "CONTROLLER_STATE_REPLAY_INVALID",
        "independent strict replays are not identical",
    )

    bindings = _discover_current_bindings(root, migration_envelope)
    _require(
        len(bindings) == 1 and bindings[0]["is_selected_envelope"],
        "CONTROLLER_STATE_AUTHORITY_AMBIGUOUS",
        f"current formal controller bindings are not unique: {bindings}",
    )

    migration_provenance = migration_sidecars["strict_loader_replay.json"]
    source_authority = _read_json(
        migration_dir / "source_checkpoint_authority.json",
        "CONTROLLER_STATE_MIGRATION_AUTHORITY_INVALID",
    )
    source_training = source_authority.get("provenance", {}).get("training_evidence", {})
    state_replay = {
        "schema": "controller_state_replay.v1",
        "gate": "CONTROLLER_STATE_REPLAY_PASS",
        "replay_count": len(replay_inventories),
        "replays": replay_inventories,
        "same_state_hash": True,
        "same_signature": True,
        "same_key_shape_dtype_inventory": True,
        "scientific_forward": False,
    }
    state_isolation = {
        "schema": "controller_state_isolation.v1",
        "gate": "CONTROLLER_STATE_ISOLATION_PASS",
        "optimizer": "absent",
        "backward": 0,
        "training_steps": 0,
        "parameter_mutation": 0,
        "checkpoint_write": 0,
        "migration_envelope_modified": False,
        "scientific_forward": False,
    }
    runtime_binding = {
        "schema": "controller_state_runtime_binding.v1",
        "gate": "CURRENT_CONTROLLER_RUNTIME_BINDING_PASS",
        "runtime_entrypoint": "ldm_project.v2.application.current.runtime._load_bundle",
        "current_head": current_head,
        "cli_checkpoint_argument": "--checkpoint",
        "explicit_checkpoint": _relative(migration_envelope, root),
        "cli_binding": (
            "python -m ldm_project.v2 student evaluate "
            f"--config {CONFIG_RELATIVE.as_posix()} "
            f"--checkpoint {_relative(migration_envelope, root)}"
        ),
        "config_path": CONFIG_RELATIVE.as_posix(),
        "config_hash": CURRENT_CONFIG_HASH,
        "checkpoint_sha256": envelope_sha256,
        "loaded_state_content_hash": inventory["state_content_hash"],
        "loaded_state_signature": inventory["state_signature"],
        "default_fallback": False,
        "alternate_controller_checkpoint": None,
        "silent_initialization_fallback": False,
        "strict_loader_result": "PASS",
        "migration_strict_replay_sidecar": migration_provenance,
    }
    migration_binding = {
        "schema": "controller_state_migration_binding.v1",
        "gate": "CONTROLLER_STATE_MIGRATION_AUTHORITY_VALID",
        "migration_manifest": _relative(
            migration_dir / MIGRATION_MANIFEST_NAME, root
        ),
        "migration_manifest_sha256": MIGRATION_MANIFEST_SHA256,
        "migration_manifest_self_hash": MIGRATION_MANIFEST_SELF_HASH,
        "migration_gate": migration_manifest["gate"],
        "migration_classification": migration_manifest["classification"],
        "migrated_checkpoint": _relative(migration_envelope, root),
        "migrated_checkpoint_sha256": envelope_sha256,
        "state_content_hash": SOURCE_STATE_HASH,
        "state_signature": SOURCE_STATE_SIGNATURE,
        "historical_immutability": "PASS",
        "tensor_identity": "PASS",
        "strict_loader": "PASS",
        "runtime_binding": "PASS",
        "migration_not_reproved": True,
    }
    provenance_lineage = {
        "schema": "controller_state_provenance_lineage.v1",
        "state_origin": "HISTORICAL_TRAINED_STATE_MIGRATED_WITHOUT_PARAMETER_CHANGE",
        "trained": True,
        "migration_training_steps": 0,
        "historical_source_checkpoint": source_authority.get("source"),
        "historical_source_commit": migration_manifest["source_commit"],
        "historical_source_run": source_authority.get("provenance", {}).get(
            "run_meta_path"
        ),
        "historical_training_evidence": source_training,
        "lineage": [
            "20260812 trained-final historical checkpoint",
            "state-preserving lineage migration",
            "migrated checkpoint envelope",
            "current runtime/config binding",
            "current formal controller state",
        ],
    }
    compact_report = {
        "schema": "controller_state_authority_compact_report.v1",
        "task_id": TASK_ID,
        "gate": "CONTROLLER_STATE_AUTHORITY_PASS",
        "classification": "CURRENT_FORMAL_CONTROLLER_STATE_AUTHORITY_REGISTERED",
        "source_type": "MIGRATED_HISTORICAL_TRAINED_STATE",
        "state_content_hash": inventory["state_content_hash"],
        "state_signature": inventory["state_signature"],
        "current_state_unique": True,
        "s3_executed": False,
        "controller_writability_supported": False,
        "label_plus_8d_adequate": False,
        "generation_ready": False,
        "next_action": "IMPLEMENT_S3_CONTROL_TO_OLS_WRITABILITY_AUDIT",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.", dir=output_path.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        _write_json(staging, "runtime_binding.json", runtime_binding)
        _write_json(staging, "migration_binding.json", migration_binding)
        _write_json(staging, "provenance_lineage.json", provenance_lineage)
        _write_json(staging, "state_replay.json", state_replay)
        _write_json(staging, "state_isolation.json", state_isolation)
        _write_json(staging, "compact_report.json", compact_report)
        artifact_hashes = {
            name: file_sha256(staging / name)
            for name in sorted(EXPECTED_AUTHORITY_NAMES - {"controller_state_authority.json"})
        }
        authority_without_hash = {
            "schema": AUTHORITY_SCHEMA,
            "task_id": TASK_ID,
            "gate": "CONTROLLER_STATE_AUTHORITY_PASS",
            "classification": "CURRENT_FORMAL_CONTROLLER_STATE_AUTHORITY_REGISTERED",
            "controller": {
                "class": "IdentityAnchoredConditionalAmplitudeGenerator",
                "architecture": "conditional_identity_logit_amplitude_v2",
                "code_dim": 8,
                "hidden_dim": 64,
                "command_gain": 0.5,
                "contract_hash": EXPECTED_GENERATOR_CONTRACT,
            },
            "state_source_type": "MIGRATED_HISTORICAL_TRAINED_STATE",
            "migrated_checkpoint": {
                "path": _relative(migration_envelope, root),
                "sha256": envelope_sha256,
                "state_content_hash": inventory["state_content_hash"],
                "state_signature": inventory["state_signature"],
            },
            "controller_state": inventory,
            "current_config": {
                "path": CONFIG_RELATIVE.as_posix(),
                "canonical_hash": CURRENT_CONFIG_HASH,
                "file_sha256": CURRENT_CONFIG_FILE_SHA256,
            },
            "runtime_binding": runtime_binding,
            "migration_binding": migration_binding,
            "historical_provenance": provenance_lineage,
            "uniqueness": {
                "gate": "CURRENT_CONTROLLER_STATE_UNIQUE_PASS",
                "current_bindings": bindings,
                "historical_checkpoints_counted": False,
            },
            "state_replay": state_replay,
            "state_isolation": state_isolation,
            "artifact_hashes": artifact_hashes,
            "artifacts": sorted(EXPECTED_AUTHORITY_NAMES),
            "governance": {
                "mainline_state": "closed_at_current_evidence_boundary",
                "reconstruction_manifold_supported": False,
                "generative_manifold_supported": False,
                "generation_readiness": "not_ready",
                "training_authorized": False,
                "formal_promotion": False,
            },
            "next_action": "IMPLEMENT_S3_CONTROL_TO_OLS_WRITABILITY_AUDIT",
        }
        authority = dict(authority_without_hash)
        authority["authority_hash"] = canonical_sha256(authority_without_hash)
        _write_json(staging, "controller_state_authority.json", authority)
        _audit_authority_tree(staging)
        loaded_authority = _read_json(
            staging / "controller_state_authority.json",
            "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        )
        recorded_authority_hash = loaded_authority.pop("authority_hash", None)
        _require(
            isinstance(recorded_authority_hash, str)
            and recorded_authority_hash == canonical_sha256(loaded_authority),
            "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
            "CSAR authority self-hash verification failed",
        )
        staging.replace(output_path)

    _audit_authority_tree(output_path)
    final_authority = _read_json(
        output_path / "controller_state_authority.json",
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
    )
    authority_hash = final_authority.pop("authority_hash", None)
    _require(
        isinstance(authority_hash, str)
        and authority_hash == canonical_sha256(final_authority),
        "CONTROLLER_STATE_RUNTIME_BINDING_FAILED",
        "published CSAR authority hash verification failed",
    )
    return {
        "output_dir": str(output_path),
        "authority_hash": authority_hash,
        "checkpoint_sha256": envelope_sha256,
        "state_content_hash": inventory["state_content_hash"],
        "state_signature": inventory["state_signature"],
        "gate": "CONTROLLER_STATE_AUTHORITY_PASS",
        "classification": "CURRENT_FORMAL_CONTROLLER_STATE_AUTHORITY_REGISTERED",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new write-once controller state authority directory",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    result = register_controller_state_authority(arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
