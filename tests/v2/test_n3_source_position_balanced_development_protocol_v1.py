"""Focused tests for the deterministic N3 source-position protocol."""

from __future__ import annotations

import json

import pytest

from scripts.research import define_n3_source_position_balanced_development_protocol_v1 as definition


def test_source_position_round_robin_is_deterministic_and_balanced() -> None:
    source_indices = list(range(definition.POOL_COUNT))
    labels = [position % definition.CLASS_COUNT for position in range(definition.POOL_COUNT)]
    first = definition.assign_source_position_balanced_folds(source_indices, labels)
    second = definition.assign_source_position_balanced_folds(source_indices, labels)
    assert first == second
    fold_ids, rows, identity = first
    assert len(rows) == definition.POOL_COUNT
    assert len(fold_ids) == definition.POOL_COUNT
    assert identity["rng_used"] is False
    for label in range(definition.CLASS_COUNT):
        class_rows = [row for row in rows if row["label"] == label]
        assert [row["within_class_rank"] for row in class_rows if row["heldout_fold"] == 0] == list(range(0, 100, 5))
        for fold in range(definition.FOLD_COUNT):
            assert sum(row["heldout_fold"] == fold for row in class_rows) == definition.ROWS_PER_CLASS_PER_FOLD
    for fold in range(definition.FOLD_COUNT):
        assert fold_ids.count(fold) == definition.HELDOUT_COUNT_PER_FOLD


def test_exact_source_position_ties_use_original_row_identity() -> None:
    source_indices = [position % definition.CLASS_COUNT for position in range(definition.POOL_COUNT)]
    labels = [position % definition.CLASS_COUNT for position in range(definition.POOL_COUNT)]
    _fold_ids, rows, _identity = definition.assign_source_position_balanced_folds(source_indices, labels)
    for label in range(definition.CLASS_COUNT):
        class_rows = [row for row in rows if row["label"] == label]
        assert [row["development_row_position"] for row in class_rows] == list(range(label, definition.POOL_COUNT, definition.CLASS_COUNT))


def test_published_protocol_replays_totality_when_available() -> None:
    path = definition.OUTPUT_DIR / definition.ARTIFACT_NAME
    if not path.is_file():
        pytest.skip("protocol is published by the bounded implementation task")
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition.validate_protocol_payload(payload)
    assert payload["contract_totality"]["status"] == "passed"
    assert payload["protocol"]["rng_used"] is False
    assert payload["execution_boundary"]["q_values_used"] is False
