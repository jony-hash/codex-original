from __future__ import annotations

import torch

from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.diagnostic.objective import _code_seed, _companion_indices, _scheduled_batches
from ldm_project.v2.evidence import canonical_sha256
from ldm_project.v2.student.conditional import coverage_ratio_upper_loss


def test_companion_schedule_is_deterministic_and_disjoint() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    primary = torch.cat([torch.nonzero(labels == label, as_tuple=False).flatten()[:5] for label in range(10)])
    first = _companion_indices(labels, primary, seed=9012)
    second = _companion_indices(labels, primary, seed=9012)
    assert torch.equal(first, second)
    assert not torch.isin(first, primary).any()
    assert torch.bincount(labels.index_select(0, first), minlength=10).tolist() == [5] * 10


def test_registered_confirmation_seeds_reproduce_formal_code_schedule() -> None:
    assert [_code_seed(seed, 0) for seed in (1234, 1235, 1236)] == [100000, 200000, 300000]
    assert _code_seed(1234, 99) == 100099


def test_segmented_confirmation_schedule_is_contiguous_and_reproducible() -> None:
    labels = torch.arange(10).repeat_interleave(2500)
    whole = _scheduled_batches(labels, batch_size=50, seed=1234, start_step=0, steps=400)
    segmented = _scheduled_batches(labels, batch_size=50, seed=1234, start_step=0, steps=100)
    segmented += _scheduled_batches(labels, batch_size=50, seed=1234, start_step=100, steps=300)
    assert canonical_sha256(whole) == canonical_sha256(segmented)
    assert all(torch.equal(left, right) for left, right in zip(whole, segmented, strict=True))


def test_confirmation_primary_schedule_is_shared_by_candidates() -> None:
    labels = torch.arange(10).repeat_interleave(2500)
    a = _scheduled_batches(labels, batch_size=50, seed=1235, start_step=0, steps=400)
    winner = _scheduled_batches(labels, batch_size=50, seed=1235, start_step=0, steps=100)
    winner += _scheduled_batches(labels, batch_size=50, seed=1235, start_step=100, steps=300)
    assert canonical_sha256(a) == canonical_sha256(winner)


def test_coverage_companion_contract_uses_matching_primary_counts() -> None:
    torch.manual_seed(7)
    labels = torch.arange(10).repeat_interleave(5)
    generated_features = torch.randn(50, 8)
    real_features = torch.randn(50, 8)
    generated_view = torch.rand(50, 1, 8, 8)
    real_view = torch.rand(50, 1, 8, 8)
    companion_features = torch.randn(50, 8)
    companion_view = torch.rand(50, 1, 8, 8)
    value = coverage_ratio_upper_loss(
        generated_features,
        generated_view,
        real_features,
        real_view,
        labels,
        epsilon=0.05,
        iterations=4,
        companion_features=companion_features,
        companion_view=companion_view,
        companion_labels=labels,
    )
    assert torch.isfinite(value)


def test_objective_commands_are_content_bound() -> None:
    parser = _parser()
    screen = parser.parse_args(
        [
            "student", "objective-alignment-screen", "--config", "c", "--dataset-root", "d",
            "--evaluator-checkpoint", "p", "--acceptance-evaluator-checkpoint", "s",
            "--acceptance-measurement-bundle", "b", "--acceptance-measurement-evidence", "m",
            "--initialization-checkpoint", "i", "--injection-evidence", "j", "--weight-calibration", "w",
        ]
    )
    assert screen.weight_calibration == "w"
    assert not hasattr(screen, "baseline_measurement")
    confirm = parser.parse_args(
        [
            "student", "objective-alignment-confirm", "--config", "c", "--dataset-root", "d",
            "--evaluator-checkpoint", "p", "--acceptance-evaluator-checkpoint", "s",
            "--acceptance-measurement-bundle", "b", "--acceptance-measurement-evidence", "m",
            "--initialization-checkpoint", "i", "--injection-evidence", "j", "--screen-evidence", "e",
        ]
    )
    assert confirm.screen_evidence == "e"
    assert not hasattr(confirm, "candidate")
