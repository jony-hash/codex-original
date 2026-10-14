from __future__ import annotations

import csv
from pathlib import Path

import torch

from scripts.research.lightgen import audit_lightgen_2d_generator_frozen_readability_screen_v1 as task


def _synthetic_data() -> dict[str, object]:
    requested = torch.arange(10, dtype=torch.long).repeat_interleave(200)
    sample_index = torch.arange(200, dtype=torch.long).repeat(10)
    sample_views = torch.zeros((2_000, 1, 28, 28), dtype=torch.float32)
    for digit in range(10):
        sample_views[digit * 200 : (digit + 1) * 200] = digit / 10.0
    empirical_labels = torch.arange(10, dtype=torch.long).repeat_interleave(20)
    empirical_views = torch.zeros((200, 1, 28, 28), dtype=torch.float32)
    for position, digit in enumerate(empirical_labels.tolist()):
        empirical_views[position] = digit / 10.0
    population_labels = torch.cat(
        (
            torch.arange(10, dtype=torch.long).repeat_interleave(300),
            torch.arange(10, dtype=torch.long).repeat_interleave(30),
        )
    )
    target_views = torch.zeros((3_300, 1, 28, 28), dtype=torch.float32)
    for position, digit in enumerate(population_labels.tolist()):
        target_views[position] = digit / 10.0
    return {
        "sample_views": sample_views,
        "sample_requested": requested,
        "sample_index": sample_index,
        "sample_e_d_prediction": requested.clone(),
        "empirical_views": empirical_views,
        "empirical_labels": empirical_labels,
        "empirical_rows": torch.arange(20_000, 20_200, dtype=torch.long),
        "target_views": target_views,
        "labels": population_labels,
        "source_rows": torch.arange(30_000, 33_300, dtype=torch.long),
        "generation_dir": Path("generation"),
        "target_readout_path": Path("readouts_A4700.pt"),
    }


def _records() -> list[dict[str, object]]:
    counter = iter(f"item{index:04d}" for index in range(task.ITEM_COUNT))
    _, mapping = task._build_records(_synthetic_data(), id_factory=lambda: next(counter))
    return mapping


def _ratings_for(mapping: list[dict[str, object]]) -> dict[str, str]:
    return {str(record["item_id"]): str(record["true_class"]) for record in mapping}


def _write_ratings(path: Path, mapping: list[dict[str, object]], answers: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "answer"])
        writer.writerows((str(record["item_id"]), answers[str(record["item_id"])]) for record in mapping)


def _answers_with_generated_counts(
    mapping: list[dict[str, object]], counts: list[int]
) -> dict[str, str]:
    answers = _ratings_for(mapping)
    repeat_index = {
        int(record["true_class"]): int(record["sample_index"])
        for record in mapping
        if record["source_type"] == "generated_repeat"
    }
    for digit, correct_count in enumerate(counts):
        rows = [
            record
            for record in mapping
            if record["source_type"] == "generated"
            and record["true_class"] == digit
            and record["sample_index"] != repeat_index[digit]
        ]
        errors = 20 - correct_count
        for record in rows[:errors]:
            answers[str(record["item_id"])] = str((digit + 1) % 10)
    for record in mapping:
        if record["source_type"] == "generated_repeat":
            answers[str(record["item_id"])] = answers[str(record["repeat_of"])]
    return answers


def test_blind_producer_balances_sources_hides_answers_and_scores_roundtrip() -> None:
    data = _synthetic_data()
    public_items, mapping = task._build_records(data, id_factory=iter(f"blind{index:04d}" for index in range(task.ITEM_COUNT)).__next__)
    assert len(public_items) == len(mapping) == 310
    assert {name: sum(row["source_type"] == name for row in mapping) for name in ("generated", "empirical_ols", "real_target", "generated_repeat")} == {
        "generated": 200,
        "empirical_ols": 50,
        "real_target": 50,
        "generated_repeat": 10,
    }
    assert [sum(row["source_type"] == "generated" and row["true_class"] == digit for row in mapping) for digit in range(10)] == [20] * 10
    page = task._render_html(public_items, "reviewtoken")
    assert "requested_class" not in page
    assert "source_type" not in page
    assert "e_d_prediction" not in page
    assert "true_class" not in page
    assert "source_row_index" not in page
    assert all(record["item_id"] in page for record in mapping)
    public_by_id = {str(record["item_id"]): record for record in public_items}
    for record in mapping:
        if record["source_type"] == "generated_repeat":
            assert public_by_id[str(record["item_id"])]["image"] == public_by_id[str(record["repeat_of"])]["image"]
    answers = _ratings_for(mapping)
    result = task._score_review(mapping, answers)
    assert result["reviewer_quality"]["passed"] is True
    assert result["generated_primary"]["correct_total"] == 200
    assert result["decision"]["classification"] == "SINGLE_REVIEWER_READABILITY_SCREEN_SUPPORTED"


def test_single_reviewer_thresholds_and_quality_failure_are_separate() -> None:
    mapping = _records()
    boundary = task._score_review(mapping, _answers_with_generated_counts(mapping, [15] * 10))
    assert boundary["generated_primary"]["correct_total"] == 150
    assert boundary["generated_primary"]["minimum_per_class_correct"] == 15
    assert boundary["decision"]["classification"] == "SINGLE_REVIEWER_READABILITY_SCREEN_SUPPORTED"

    class_boundary = task._score_review(mapping, _answers_with_generated_counts(mapping, [20, 18] + [14] * 8))
    assert class_boundary["generated_primary"]["correct_total"] == 150
    assert class_boundary["generated_primary"]["minimum_per_class_correct"] == 14
    assert class_boundary["decision"]["classification"] == "SINGLE_REVIEWER_READABILITY_SCREEN_SUPPORTED"

    class_failure = task._score_review(mapping, _answers_with_generated_counts(mapping, [20, 13] + [15] * 8))
    assert class_failure["generated_primary"]["correct_total"] == 153
    assert class_failure["generated_primary"]["minimum_per_class_correct"] == 13
    assert class_failure["decision"]["classification"] == "TEN_CLASS_READABILITY_SCREEN_NOT_SUPPORTED"

    total_failure = task._score_review(mapping, _answers_with_generated_counts(mapping, [14] * 10))
    assert total_failure["generated_primary"]["correct_total"] == 140
    assert total_failure["generated_primary"]["minimum_per_class_correct"] == 14
    assert total_failure["decision"]["classification"] == "TEN_CLASS_READABILITY_SCREEN_NOT_SUPPORTED"

    answers = _ratings_for(mapping)
    anchors = [row for row in mapping if row["source_type"] == "real_target"]
    for row in anchors[:6]:
        answers[str(row["item_id"])] = str((int(row["true_class"]) + 1) % 10)
    invalid_review = task._score_review(mapping, answers)
    assert invalid_review["reviewer_quality"]["real_target_anchor_correct"] == 44
    assert invalid_review["decision"]["status"] == "REVIEW_INCONCLUSIVE"
    assert invalid_review["decision"]["classification"] == "REVIEW_EVIDENCE_INSUFFICIENT"

    repeat_failure = _ratings_for(mapping)
    repeats = [row for row in mapping if row["source_type"] == "generated_repeat"]
    for row in repeats[:2]:
        repeat_failure[str(row["item_id"])] = str((int(row["true_class"]) + 1) % 10)
    repeat_review = task._score_review(mapping, repeat_failure)
    assert repeat_review["reviewer_quality"]["duplicate_agreement"] == 8
    assert repeat_review["decision"]["status"] == "REVIEW_INCONCLUSIVE"


def test_ratings_csv_rejects_missing_duplicate_unknown_and_invalid_answers(tmp_path: Path) -> None:
    mapping = _records()
    valid = _ratings_for(mapping)
    path = tmp_path / "ratings.csv"
    _write_ratings(path, mapping, valid)
    assert task._parse_ratings(path, mapping) == valid

    path.write_text("item_id,answer\nitem0000,0\n", encoding="utf-8")
    try:
        task._parse_ratings(path, mapping)
    except task.ReadabilityScreenError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("incomplete ratings were accepted")

    path.write_text("item_id,answer\nitem0000,0\nitem0000,0\n", encoding="utf-8")
    try:
        task._parse_ratings(path, mapping)
    except task.ReadabilityScreenError as error:
        assert "repeats" in str(error)
    else:
        raise AssertionError("duplicate rating IDs were accepted")

    path.write_text("item_id,answer\nunknown,0\n", encoding="utf-8")
    try:
        task._parse_ratings(path, mapping)
    except task.ReadabilityScreenError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown rating IDs were accepted")

    path.write_text("item_id,answer\nitem0000,10\n", encoding="utf-8")
    try:
        task._parse_ratings(path, mapping)
    except task.ReadabilityScreenError as error:
        assert "invalid rating" in str(error)
    else:
        raise AssertionError("out-of-range ratings were accepted")


def test_scoring_rejects_corrupted_unblinding_mapping() -> None:
    mapping = [dict(record) for record in _records()]
    answers = _ratings_for(mapping)
    first = next(record for record in mapping if record["source_type"] == "generated")
    second = next(
        record
        for record in mapping
        if record["source_type"] == "generated" and record["true_class"] == first["true_class"] and record is not first
    )
    second["sample_index"] = first["sample_index"]
    try:
        task._score_review(mapping, answers)
    except task.ReadabilityScreenError as error:
        assert "indices are missing, duplicated" in str(error)
    else:
        raise AssertionError("corrupted generated sample mapping was accepted")


def test_diversity_summary_is_class_conditional_and_explicitly_descriptive() -> None:
    samples = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
    requested = torch.arange(10).repeat_interleave(2)
    for digit in range(10):
        samples[digit * 2 + 1] = 0.1
    target_labels = torch.arange(10).repeat_interleave(3)
    targets = torch.zeros((30, 1, 28, 28), dtype=torch.float32)
    for position, digit in enumerate(target_labels.tolist()):
        targets[position] = digit / 10.0
    summary = task._diversity_summary({
        "sample_views": samples,
        "sample_requested": requested,
        "target_views": targets,
        "labels": target_labels,
        "source_rows": torch.arange(30),
    })
    assert summary["novelty_or_nonmemorization_claim"] is False
    assert summary["per_class"]["0"]["sample_count"] == 2
    assert summary["per_class"]["0"]["exact_8bit_duplicate_excess"] == 0
    assert summary["five_closest_generated_pairs_per_class"]["0"][0]["sample_index_a"] == 0


def test_a4700_target_readout_uses_saved_checkpoint_lineage(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generation"
    capacity_dir = tmp_path / "capacity"
    generation_dir.mkdir()
    capacity_dir.mkdir()
    checkpoint = capacity_dir / "arm_a_step4700.pt"
    checkpoint.touch()
    expected = capacity_dir / "readouts_A4700.pt"
    expected.touch()
    population = {"source_checkpoint": str(checkpoint)}
    assert task._resolve_a4700_target_readout(generation_dir, population) == expected.resolve()
