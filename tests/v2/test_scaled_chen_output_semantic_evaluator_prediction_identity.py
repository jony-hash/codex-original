from __future__ import annotations

import torch

from scripts.research import (
    materialize_scaled_chen_output_semantic_evaluator_prediction_identity as vpim,
)


def test_aggregate_predictions_reconstructs_registered_qualification() -> None:
    confusion = torch.tensor(
        [
            [497, 0, 0, 0, 1, 0, 1, 0, 2, 1],
            [0, 529, 1, 2, 1, 0, 1, 0, 0, 0],
            [1, 1, 484, 4, 0, 1, 0, 2, 3, 1],
            [1, 1, 3, 506, 0, 2, 0, 0, 5, 3],
            [0, 3, 0, 0, 475, 0, 0, 2, 0, 4],
            [0, 0, 0, 13, 2, 429, 6, 1, 4, 2],
            [1, 0, 1, 0, 2, 1, 479, 0, 1, 0],
            [0, 0, 0, 2, 0, 0, 0, 521, 0, 4],
            [1, 2, 1, 12, 0, 0, 3, 4, 489, 3],
            [3, 2, 0, 6, 10, 1, 0, 8, 1, 447],
        ],
        dtype=torch.long,
    )
    labels = torch.repeat_interleave(torch.arange(10), confusion.sum(dim=1))
    predictions = torch.cat(
        [
            torch.cat(
                [torch.full((int(count),), column, dtype=torch.long) for column, count in enumerate(row) if count]
            )
            for row in confusion
        ]
    )

    result = vpim.aggregate_predictions(labels, predictions)

    assert result["sample_count"] == 5000
    assert result["overall_accuracy"] == 0.9712
    assert result["error_count"] == 144
    assert torch.equal(torch.tensor(result["confusion_matrix"]), confusion)


def test_validation_record_order_rejects_mismatch() -> None:
    indices = torch.tensor([10, 11, 12], dtype=torch.long)
    labels = torch.tensor([3, 4, 5], dtype=torch.long)
    records = [
        {"source_dataset": "mnist_train", "source_index": 10, "label": 3},
        {"source_dataset": "mnist_train", "source_index": 11, "label": 4},
        {"source_dataset": "mnist_train", "source_index": 12, "label": 5},
    ]

    assert vpim.validate_record_order(records, indices, labels)

    records[1] = {"source_dataset": "mnist_train", "source_index": 12, "label": 4}
    assert not vpim.validate_record_order(records, indices, labels)


def test_null_permutation_is_deterministic_and_preserves_class_counts() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)

    first = vpim.build_null_permutation(labels, seed=2053)
    second = vpim.build_null_permutation(labels, seed=2053)

    assert torch.equal(first, second)
    assert torch.equal(torch.bincount(labels), torch.bincount(labels.index_select(0, first)))
    assert vpim.build_null_permutation(labels, seed=2053).numel() == labels.numel()


def test_cpu_replay_is_deterministic_and_does_not_change_state() -> None:
    torch.manual_seed(7)
    model = vpim.FrozenMNISTEvaluator()
    views = torch.rand((6, 1, 28, 28), dtype=torch.float32)
    before = vpim.evaluator_state_digest(model)

    first_logits, first_predictions = vpim.replay_predictions(model, views, batch_size=2)
    second_logits, second_predictions = vpim.replay_predictions(model, views, batch_size=2)

    assert model.training is False
    assert torch.equal(first_logits, second_logits)
    assert torch.equal(first_predictions, second_predictions)
    assert torch.isfinite(first_logits).all()
    assert vpim.evaluator_state_digest(model) == before


def test_write_once_path_selects_next_attempt_without_overwrite(tmp_path) -> None:
    requested = tmp_path / "authority_attempt_01"
    requested.mkdir()

    selected = vpim.next_write_once_output(requested)

    assert selected.name == "authority_attempt_02"
    assert not selected.exists()
