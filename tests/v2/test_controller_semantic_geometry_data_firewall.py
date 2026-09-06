"""Fail-closed data and code firewall tests for the S3 repair harness."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

import ldm_project.v2.application.research.controller_semantic_geometry_training as harness
from ldm_project.v2.application.research.controller_semantic_geometry_training import (
    PARTITION_IDENTITY,
    S1Reference,
    S1_CONFIRMATION_INDICES_SHA256,
    S1_SELECTION_INDICES_SHA256,
    TargetBank,
    TARGET_BANK_MANIFEST_SCHEMA,
    TARGET_BANK_SCHEMA,
    build_logical_batch,
    build_frozen_partition,
    code_overlap_count,
    finite_training_stream,
    internal_validation_codes,
    load_config,
    load_target_bank,
    no_update_preflight,
    register_data_authority,
    validate_finite_stream_against_protected,
    validate_target_bank,
)
from ldm_project.v2.evidence.io import atomic_torch_save_new, file_sha256, tensor_sha256


def _reference(*, protected_overlap: bool = False) -> S1Reference:
    indices = torch.arange(1000, dtype=torch.long)
    if protected_overlap:
        indices[0] = 50000
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(100)
    r1 = torch.arange(1000 * 100, dtype=torch.float32).reshape(1000, 100) / 1000.0
    return S1Reference(
        fit_r1=r1,
        fit_labels=labels,
        fit_indices=indices,
        selection_labels=torch.arange(10, dtype=torch.long).repeat_interleave(100),
        selection_indices=torch.arange(1000, dtype=torch.long),
        confirmation_labels=torch.arange(10, dtype=torch.long).repeat_interleave(500),
        confirmation_indices=torch.arange(55000, 60000, dtype=torch.long),
        source_artifact_sha256="synthetic-s1-artifact",
    )


def _config() -> dict:
    return load_config(
        "configs/research/lightgen_controller_semantic_geometry_training_v1.yaml"
    )


def _bank(reference: S1Reference, *, bad_partition: bool = False) -> TargetBank:
    train_positions = torch.cat(
        [torch.arange(label * 100, label * 100 + 50) for label in range(10)]
    )
    validation_positions = torch.cat(
        [torch.arange(label * 100 + 50, label * 100 + 100) for label in range(10)]
    )
    training_indices = reference.fit_indices[train_positions]
    validation_indices = reference.fit_indices[validation_positions]
    training_labels = reference.fit_labels[train_positions]
    validation_labels = reference.fit_labels[validation_positions]
    manifest = {
        "schema": TARGET_BANK_MANIFEST_SCHEMA,
        "partition_identity_sha256": "wrong" if bad_partition else PARTITION_IDENTITY,
        "partition_identity_material_sha256": "0" * 64,
        "partition_manifest_file_sha256": "1" * 64,
        "source_artifact_sha256": reference.source_artifact_sha256,
        "observer_identity": "R1_POWER:z_j=abs(c_j)^2/s_j",
        "protected_sample_identities": {
            "s1_selection_indices_sha256": S1_SELECTION_INDICES_SHA256,
            "s1_selection_labels_sha256": harness.S1_SELECTION_LABELS_SHA256,
            "s1_confirmation_indices_sha256": S1_CONFIRMATION_INDICES_SHA256,
            "s1_confirmation_labels_sha256": harness.S1_CONFIRMATION_LABELS_SHA256,
            "s2_validation_indices_sha256": _config()["data_firewall"]["s2_validation"]["indices_sha256"],
            "s2_validation_labels_sha256": _config()["data_firewall"]["s2_validation"]["labels_sha256"],
        },
    }
    training_r1 = reference.fit_r1[train_positions]
    validation_r1 = reference.fit_r1[validation_positions]
    manifest.update(
        {
            "training_r1_sha256": tensor_sha256(training_r1),
            "training_labels_sha256": tensor_sha256(training_labels),
            "training_indices_sha256": tensor_sha256(training_indices),
            "internal_validation_r1_sha256": tensor_sha256(validation_r1),
            "internal_validation_labels_sha256": tensor_sha256(validation_labels),
            "internal_validation_indices_sha256": tensor_sha256(validation_indices),
        }
    )
    return TargetBank(
        training_r1,
        training_labels,
        training_indices,
        validation_r1,
        validation_labels,
        validation_indices,
        manifest,
    )


def test_valid_target_bank_is_disjoint_and_class_balanced() -> None:
    reference = _reference()
    validate_target_bank(_bank(reference), reference, _config())


def test_real_frozen_partition_reproduces_registered_component_hashes() -> None:
    config = _config()
    reference = harness._load_s1_reference(config, ".")
    partition = build_frozen_partition(reference)
    assert harness.tensor_sha256(partition.training_positions) == harness.PARTITION_TRAINING_POSITION_SHA256
    assert harness.tensor_sha256(partition.internal_validation_positions) == harness.PARTITION_VALIDATION_POSITION_SHA256
    assert harness.tensor_sha256(partition.training_indices) == harness.PARTITION_TRAINING_INDEX_SHA256
    assert harness.tensor_sha256(partition.internal_validation_indices) == harness.PARTITION_VALIDATION_INDEX_SHA256
    assert partition.training_r1_sha256 == harness.PARTITION_TRAINING_R1_SHA256
    assert partition.internal_validation_r1_sha256 == harness.PARTITION_VALIDATION_R1_SHA256
    assert partition.training_positions.numel() == 500
    assert partition.internal_validation_positions.numel() == 500


def test_missing_target_bank_is_rejected_before_safe_load(tmp_path) -> None:
    with pytest.raises(ValueError, match="target bank artifact is missing"):
        load_target_bank(tmp_path / "missing-target-bank.pt")


def test_protected_sample_overlap_is_fail_closed() -> None:
    reference = _reference(protected_overlap=True)
    with pytest.raises(ValueError, match="protected sample split"):
        validate_target_bank(_bank(reference), reference, _config())


def test_partition_identity_mismatch_is_fail_closed() -> None:
    reference = _reference()
    with pytest.raises(ValueError, match="partition identity"):
        validate_target_bank(_bank(reference, bad_partition=True), reference, _config())


def test_target_bank_loader_rejects_unregistered_scientific_split(tmp_path) -> None:
    reference = _reference()
    bank = _bank(reference)
    payload = {
        "schema": TARGET_BANK_SCHEMA,
        "manifest": dict(bank.manifest),
        "training": {
            "r1": bank.training_r1,
            "labels": bank.training_labels,
            "indices": bank.training_indices,
        },
        "internal_validation": {
            "r1": bank.validation_r1,
            "labels": bank.validation_labels,
            "indices": bank.validation_indices,
        },
        "selection": torch.zeros(1),
    }
    path = tmp_path / "bad_target_bank.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="outside training/internal_validation"):
        load_target_bank(path)


def test_write_once_target_bank_path_cannot_be_replaced(tmp_path) -> None:
    path = tmp_path / "artifact.pt"
    atomic_torch_save_new(path, {"value": torch.tensor(1)}, artifact_kind="test artifact")
    with pytest.raises(FileExistsError):
        atomic_torch_save_new(path, {"value": torch.tensor(2)}, artifact_kind="test artifact")


def test_label_major_logical_batch_reuses_base_codes_without_pairing() -> None:
    real = torch.randn(500, 100, dtype=torch.float32)
    real_labels = torch.arange(10, dtype=torch.long).repeat_interleave(50)
    base_codes = torch.linspace(-1.0, 1.0, 50 * 8, dtype=torch.float32).reshape(50, 8)
    batch = build_logical_batch(real, real_labels, base_codes)
    assert batch.controller_codes.shape == (500, 8)
    assert batch.controller_labels.shape == (500,)
    assert torch.equal(batch.controller_labels, real_labels)
    for label in range(10):
        start = label * 50
        assert torch.equal(batch.controller_codes[start : start + 50], base_codes)
    assert batch.physical_microbatch_size == 1
    assert batch.real_r1.requires_grad is False


def test_protected_code_overlap_is_fail_closed() -> None:
    blocks, _ = finite_training_stream(1)
    internal = internal_validation_codes()
    with pytest.raises(ValueError, match="protected code overlap"):
        validate_finite_stream_against_protected(blocks, blocks[0], internal)
    assert code_overlap_count(blocks.reshape(-1, 8), internal) == 0


def test_finite_stream_has_fixed_horizon_and_first_block_identity() -> None:
    blocks, metadata = finite_training_stream(492)
    assert blocks.shape == (492, 50, 8)
    assert metadata["base_codes"] == 24600
    assert metadata["first_block_sha256"] == (
        "a1fa3261b7f82756aaee4f3b62127bccee0bc2c0243f476b9956beeb4b92b0ce"
    )
    assert metadata["internal_validation_overlap_count"] == 0


def test_training_stream_consumes_sequential_blocks_without_reseed() -> None:
    first, _ = finite_training_stream(2)
    second, _ = finite_training_stream(2)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_no_update_preflight_consumes_real_authority_without_update(tmp_path, monkeypatch) -> None:
    config = _config()
    reference = harness._load_s1_reference(config, ".")
    bank = _bank(reference)
    payload = {
        "schema": TARGET_BANK_SCHEMA,
        "manifest": dict(bank.manifest),
        "training": {
            "r1": bank.training_r1,
            "labels": bank.training_labels,
            "indices": bank.training_indices,
        },
        "internal_validation": {
            "r1": bank.validation_r1,
            "labels": bank.validation_labels,
            "indices": bank.validation_indices,
        },
    }
    target_bank = tmp_path / "preflight-target-bank.pt"
    torch.save(payload, target_bank)
    (tmp_path / "preflight-target-bank.manifest.json").write_text(
        json.dumps(
            {
                **bank.manifest,
                "artifact": {
                    "path": str(target_bank),
                    "sha256": file_sha256(target_bank),
                    "schema": TARGET_BANK_SCHEMA,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "_git_state", lambda _root: ("synthetic-head", ()))
    monkeypatch.setattr(
        harness,
        "_load_data_authority",
        lambda _config, _root, target_bank_path=None: {
            "reference": reference,
            "hashes": {
                "target_bank_sha256": file_sha256(target_bank),
                "partition_manifest_file_sha256": "1" * 64,
                "data_firewall_manifest_sha256": "2" * 64,
            },
        },
    )
    result = no_update_preflight(
        "configs/research/lightgen_controller_semantic_geometry_training_v1.yaml",
        ".",
        target_bank,
        tmp_path / "future-output.pt",
        expected_head="synthetic-head",
    )
    assert result["gate"] == "PRETRAINING_HARNESS_PREFLIGHT_PASS"
    assert result["optimizer_step_count"] == 0
    assert result["backward_count"] == 0
    assert result["checkpoint_writes"] == 0


def test_registration_writes_dedicated_redacted_authority_and_reloads(tmp_path, monkeypatch) -> None:
    config = deepcopy(_config())
    authority_dir = tmp_path / "data-authority"
    future_dir = tmp_path / "future-training"
    authority = config["data_firewall"]["authority"]
    authority.update(
        {
            "run_directory": str(authority_dir),
            "partition_manifest": str(authority_dir / "controller_semantic_geometry_partition_manifest.v1.json"),
            "target_bank": str(authority_dir / "controller_semantic_geometry_r1_target_bank.v1.pt"),
            "target_bank_manifest": str(authority_dir / "controller_semantic_geometry_r1_target_bank.v1.manifest.json"),
            "data_firewall_manifest": str(authority_dir / "controller_semantic_geometry_data_firewall_manifest.v1.json"),
        }
    )
    config["output"]["target_directory"] = str(future_dir)
    monkeypatch.setattr(harness, "_git_state", lambda _root: ("synthetic-head", ()))
    result = register_data_authority(
        config,
        ".",
        config_path="configs/research/lightgen_controller_semantic_geometry_training_v1.yaml",
        output_dir=authority_dir,
    )
    assert result["classification"] == "CONTROLLER_SEMANTIC_GEOMETRY_DATA_AUTHORITY_REGISTERED"
    assert result["partition_identity_sha256"] == PARTITION_IDENTITY
    assert result["partition_manifest_file_sha256"] != PARTITION_IDENTITY
    assert result["optimizer_step_count"] == 0
    assert result["backward_count"] == 0
    config["data_firewall"]["authority"].update(
        {
            "partition_manifest_file_sha256": result["partition_manifest_file_sha256"],
            "target_bank_sha256": result["target_bank_sha256"],
            "target_bank_manifest_sha256": result["target_bank_manifest_sha256"],
            "data_firewall_manifest_sha256": result["data_firewall_manifest_sha256"],
        }
    )
    authority_loaded = harness._load_data_authority(config, ".", target_bank_path=result["target_bank"])
    bank = load_target_bank(result["target_bank"], expected_sha256=result["target_bank_sha256"])
    validate_target_bank(
        bank,
        authority_loaded["reference"],
        config,
        expected_artifact_sha256=result["target_bank_sha256"],
        expected_partition_manifest_file_sha256=result["partition_manifest_file_sha256"],
        repo_root=".",
    )
    payload = harness.safe_torch_load(result["target_bank"], expected_sha256=result["target_bank_sha256"])
    assert set(payload) == {"schema", "manifest", "training", "internal_validation"}
