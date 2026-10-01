"""Focused tests for the immutable N3 producer data contract."""

from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen.materialize_ols_producer_data_authority import build_fixed_partition, validate_firewall_entries


def _synthetic_source() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.arange(10, dtype=torch.int64).repeat_interleave(100)
    indices = torch.cat(
        [torch.arange(label * 100, label * 100 + 100, dtype=torch.int64) for label in range(10)]
    )
    positions = torch.arange(1000, dtype=torch.int64)
    permutation = torch.flip(positions, dims=(0,))
    return indices[permutation], labels[permutation], torch.zeros((1000, 100), dtype=torch.complex64)


def test_fixed_partition_is_label_balanced_and_deterministic() -> None:
    indices, labels, r3 = _synthetic_source()
    first = build_fixed_partition(indices, labels, r3)
    second = build_fixed_partition(indices, labels, r3)

    assert torch.equal(first.fit_positions, second.fit_positions)
    assert torch.equal(first.validation_positions, second.validation_positions)
    assert torch.equal(first.fit_labels, torch.arange(10).repeat_interleave(50))
    assert torch.equal(first.validation_labels, torch.arange(10).repeat_interleave(50))
    assert set(first.fit_positions.tolist()).isdisjoint(first.validation_positions.tolist())
    assert set(first.fit_positions.tolist()) | set(first.validation_positions.tolist()) == set(range(1000))


def test_fixed_partition_rejects_unbalanced_source() -> None:
    indices, labels, r3 = _synthetic_source()
    labels[0] = 1

    with pytest.raises(ValueError, match="source count changed"):
        build_fixed_partition(indices, labels, r3)


def test_firewall_accepts_zero_overlap_and_rejects_overlap() -> None:
    valid = [
        {
            "independently_required": True,
            "status": "PASS_ZERO_OVERLAP",
            "fit_overlap_count": 0,
            "validation_overlap_count": 0,
        }
    ]
    validate_firewall_entries(valid)

    invalid = [dict(valid[0], validation_overlap_count=1)]
    with pytest.raises(ValueError, match="BLOCKED_OLS_PRODUCER_DATA_FIREWALL"):
        validate_firewall_entries(invalid)
