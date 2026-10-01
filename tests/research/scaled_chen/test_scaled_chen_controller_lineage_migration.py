"""Regression tests for the explicit state-preserving controller migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2 import load_config
from ldm_project.v2.application.current.runtime import _load_bundle
from scripts.research.scaled_chen.migrate_scaled_chen_controller_lineage import EXPECTED_ARTIFACT_NAMES, MIGRATED_ENVELOPE_NAME, SOURCE_RELATIVE, MigrationError, migrate_controller_state


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / SOURCE_RELATIVE
CONFIG = REPO_ROOT / "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"


def test_current_loader_rejects_historical_config_binding() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        _load_bundle(config, str(SOURCE), device=torch.device("cpu"))


def test_migration_rejects_non_registered_source(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="fixed registered trained-final checkpoint"):
        migrate_controller_state(tmp_path / "other.pt", tmp_path / "migration", repo_root=REPO_ROOT)


def test_migration_rejects_existing_write_once_output(tmp_path: Path) -> None:
    output = tmp_path / "migration"
    output.mkdir()
    with pytest.raises(MigrationError, match="write-once migration output already exists"):
        migrate_controller_state(SOURCE, output, repo_root=REPO_ROOT)


def test_migration_produces_strict_state_preserving_envelope(tmp_path: Path) -> None:
    output = tmp_path / "migration"
    result = migrate_controller_state(SOURCE, output, repo_root=REPO_ROOT)

    assert result["gate"] == "CONTROLLER_LINEAGE_MIGRATION_PASS"
    assert {path.name for path in output.iterdir()} == EXPECTED_ARTIFACT_NAMES
    manifest = json.loads(
        (output / "controller_lineage_migration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["migrated_checkpoint"]["path"] == MIGRATED_ENVELOPE_NAME
    assert manifest["state_preserving"] is True
    assert manifest["parameter_mutation"] is False
    assert manifest["current_config_hash"] == manifest["migrated_checkpoint"]["manifest"]["config_hash"]
    assert manifest["source_checkpoint"]["path"] == SOURCE_RELATIVE.as_posix()
    assert manifest["next_action"] == "RERUN_H-LOMA-GEN-S3-CSAR"

    equivalence = json.loads(
        (output / "semantic_config_equivalence.json").read_text(encoding="utf-8")
    )
    assert equivalence["field_diff_count"] == 0
    assert equivalence["controller_source_content_equal"] is True

    replay = json.loads(
        (output / "strict_loader_replay.json").read_text(encoding="utf-8")
    )
    assert replay["result"] == "PASS"
    assert replay["config_hash_override_used"] is False
    assert len(replay["attempts"]) == 2

    immutability = json.loads(
        (output / "historical_immutability.json").read_text(encoding="utf-8")
    )
    assert immutability["result"] == "PASS"
    assert immutability["source_before"] == immutability["source_after"]
