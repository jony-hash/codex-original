"""Regression tests for current controller state authority registration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.register_scaled_chen_controller_state_authority import (
    EXPECTED_AUTHORITY_NAMES,
    MIGRATION_RELATIVE,
    MIGRATED_ENVELOPE_NAME,
    SOURCE_STATE_HASH,
    SOURCE_STATE_SIGNATURE,
    AuthorityError,
    _audit_migration_authority,
    _discover_current_bindings,
    register_controller_state_authority,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_registered_migration_authority_is_complete() -> None:
    manifest, sidecars, envelope = _audit_migration_authority(
        REPO_ROOT / MIGRATION_RELATIVE, REPO_ROOT
    )

    assert manifest["gate"] == "CONTROLLER_LINEAGE_MIGRATION_PASS"
    assert manifest["migrated_checkpoint"]["path"] == MIGRATED_ENVELOPE_NAME
    assert manifest["migrated_checkpoint"]["state_content_hash"] == SOURCE_STATE_HASH
    assert manifest["migrated_checkpoint"]["state_signature"] == SOURCE_STATE_SIGNATURE
    assert envelope.is_file()
    assert sidecars["strict_loader_replay.json"]["result"] == "PASS"


def test_only_migrated_envelope_is_current_config_bound() -> None:
    envelope = REPO_ROOT / MIGRATION_RELATIVE / MIGRATED_ENVELOPE_NAME
    bindings = _discover_current_bindings(REPO_ROOT, envelope)

    assert len(bindings) == 1
    assert bindings[0]["is_selected_envelope"] is True


def test_registration_rejects_existing_write_once_output(tmp_path: Path) -> None:
    output = tmp_path / "authority"
    output.mkdir()

    with pytest.raises(AuthorityError, match="refusing to overwrite"):
        register_controller_state_authority(output, repo_root=REPO_ROOT)


def test_registers_current_controller_authority(tmp_path: Path) -> None:
    output = tmp_path / "authority"
    result = register_controller_state_authority(output, repo_root=REPO_ROOT)

    assert result["gate"] == "CONTROLLER_STATE_AUTHORITY_PASS"
    assert result["classification"] == "CURRENT_FORMAL_CONTROLLER_STATE_AUTHORITY_REGISTERED"
    assert {path.name for path in output.iterdir()} == EXPECTED_AUTHORITY_NAMES

    authority = json.loads(
        (output / "controller_state_authority.json").read_text(encoding="utf-8")
    )
    assert authority["state_source_type"] == "MIGRATED_HISTORICAL_TRAINED_STATE"
    assert authority["controller_state"]["state_content_hash"] == SOURCE_STATE_HASH
    assert authority["controller_state"]["state_signature"] == SOURCE_STATE_SIGNATURE
    assert authority["runtime_binding"]["default_fallback"] is False
    assert authority["uniqueness"]["gate"] == "CURRENT_CONTROLLER_STATE_UNIQUE_PASS"
    assert authority["state_replay"]["gate"] == "CONTROLLER_STATE_REPLAY_PASS"
    assert authority["state_isolation"]["checkpoint_write"] == 0
    assert authority["next_action"] == "IMPLEMENT_S3_CONTROL_TO_OLS_WRITABILITY_AUDIT"
