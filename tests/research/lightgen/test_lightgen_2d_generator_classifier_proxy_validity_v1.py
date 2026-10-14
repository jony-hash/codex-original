from __future__ import annotations

import csv
import itertools
import json
import random
from pathlib import Path

import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_classifier_proxy_validity_v1 as task,
)


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
    target_labels = torch.cat(
        (
            torch.arange(10, dtype=torch.long).repeat_interleave(300),
            torch.arange(10, dtype=torch.long).repeat_interleave(30),
        )
    )
    target_views = torch.zeros((3_300, 1, 28, 28), dtype=torch.float32)
    for position, digit in enumerate(target_labels.tolist()):
        target_views[position] = digit / 10.0
    e_d_prediction = requested.clone()
    selection = random.Random(task.SELECTION_SEED).sample(range(2_000), 200)
    selected_negative = set(selection[150:])
    outside_selection = [
        position for position in range(2_000) if position not in selection
    ]
    selected_negative.update(random.Random(91).sample(outside_selection, 437))
    for position in selected_negative:
        e_d_prediction[position] = (requested[position] + 1) % 10
    return {
        "sample_views": sample_views,
        "sample_requested": requested,
        "sample_index": sample_index,
        "sample_e_d_prediction": e_d_prediction,
        "empirical_views": empirical_views,
        "empirical_labels": empirical_labels,
        "empirical_rows": torch.arange(20_000, 20_200, dtype=torch.long),
        "empirical_e_d_prediction": empirical_labels.clone(),
        "target_views": target_views,
        "labels": target_labels,
        "source_rows": torch.arange(30_000, 33_300, dtype=torch.long),
        "generation_dir": Path("generation"),
        "target_readout_path": Path("readouts_A4700.pt"),
        "population_positive_count": 1_513,
    }


def _records() -> list[dict[str, object]]:
    counter = iter(f"item{index:04d}" for index in range(task.ITEM_COUNT))
    _, mapping = task._build_records(
        _synthetic_data(), id_factory=lambda: next(counter)
    )
    return mapping


def _ratings_for(mapping: list[dict[str, object]]) -> dict[str, str]:
    answers = {}
    by_id = {str(row["item_id"]): row for row in mapping}
    for row in mapping:
        kind = row["source_type"]
        if kind == "generated" or kind == "empirical_ols":
            answers[str(row["item_id"])] = str(row["requested_class"])
        elif kind == "real_target":
            answers[str(row["item_id"])] = str(row["true_class"])
        elif kind == "blank":
            answers[str(row["item_id"])] = "unclear"
    for row in mapping:
        if row["source_type"] == "generated_repeat":
            answers[str(row["item_id"])] = answers[str(row["repeat_of"])]
    assert set(answers) == set(by_id)
    return answers


def _mapping_with_150_proxy_positives() -> list[dict[str, object]]:
    mapping = [dict(row) for row in _records()]
    primary = [row for row in mapping if row["source_type"] == "generated"]
    positive_ids = {str(row["item_id"]) for row in primary[:150]}
    source_for_repeat = {str(row["item_id"]): row for row in primary}
    for row in primary:
        row["e_d_prediction"] = (
            int(row["requested_class"])
            if str(row["item_id"]) in positive_ids
            else (int(row["requested_class"]) + 1) % 10
        )
    for row in mapping:
        if row["source_type"] == "generated_repeat":
            source = source_for_repeat[str(row["repeat_of"])]
            row["e_d_prediction"] = source["e_d_prediction"]
    return mapping


def _answers_with_mismatches(
    mapping: list[dict[str, object]], mismatch_count: int
) -> dict[str, str]:
    answers = _ratings_for(mapping)
    positives = [
        row
        for row in mapping
        if row["source_type"] == "generated"
        and row["e_d_prediction"] == row["requested_class"]
    ]
    assert len(positives) == 150
    for row in positives[:mismatch_count]:
        answers[str(row["item_id"])] = str((int(row["requested_class"]) + 1) % 10)
    for row in mapping:
        if row["source_type"] == "generated_repeat":
            answers[str(row["item_id"])] = answers[str(row["repeat_of"])]
    return answers


def _write_ratings(
    path: Path, mapping: list[dict[str, object]], answers: dict[str, str]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item_id", "answer"])
        writer.writerows(
            (str(row["item_id"]), answers[str(row["item_id"])]) for row in mapping
        )


def test_hypergeometric_interval_matches_independent_small_population_enumeration() -> (
    None
):
    population_size, draws, observed = 10, 4, 1
    alpha_tail = (1.0 - task.CONFIDENCE_LEVEL) / 2.0
    outcomes: dict[int, list[int]] = {}
    for positive_count in range(population_size + 1):
        counts = [0] * (draws + 1)
        for selected in itertools.combinations(range(population_size), draws):
            counts[sum(index < positive_count for index in selected)] += 1
        outcomes[positive_count] = counts
    eligible = [
        positive_count
        for positive_count, counts in outcomes.items()
        if sum(counts[observed:]) / sum(counts) >= alpha_tail
    ]
    lower = min(eligible) / population_size
    eligible = [
        positive_count
        for positive_count, counts in outcomes.items()
        if sum(counts[: observed + 1]) / sum(counts) >= alpha_tail
    ]
    upper = max(eligible) / population_size
    assert task._hypergeom_interval(observed, draws, population_size) == [lower, upper]


def test_declared_decision_examples_use_exact_finite_population_bounds() -> None:
    within_limit = task._hypergeom_interval(8, 150)
    above_limit = task._hypergeom_interval(30, 150)
    assert within_limit[1] <= 0.10
    assert round(within_limit[1] * 100, 2) == 9.98
    assert above_limit[0] > 0.10
    assert round(above_limit[0] * 100, 2) == 14.21


def test_producer_blinds_labels_and_places_repeats_apart() -> None:
    counter = iter(f"blind{index:04d}" for index in range(task.ITEM_COUNT))
    public_items, mapping = task._build_records(
        _synthetic_data(), id_factory=lambda: next(counter)
    )
    assert len(public_items) == len(mapping) == 340
    assert [
        sum(
            row["source_type"] == "generated" and row["requested_class"] == digit
            for row in mapping
        )
        for digit in range(10)
    ] == task.GENERATED_CLASS_COUNTS
    assert sum(row["source_type"] == "empirical_ols" for row in mapping) == 50
    assert sum(row["source_type"] == "real_target" for row in mapping) == 50
    assert sum(row["source_type"] == "generated_repeat" for row in mapping) == 30
    assert sum(row["source_type"] == "blank" for row in mapping) == 10
    page = task._render_html(public_items, "reviewtoken")
    for hidden_value in (
        "requested_class",
        "true_class",
        "e_d_prediction",
        "source_type",
        "source_row_index",
        "review_mapping",
    ):
        assert hidden_value not in page
    positions = {str(row["item_id"]): index for index, row in enumerate(mapping)}
    public_by_id = {str(row["item_id"]): row for row in public_items}
    for row in mapping:
        if row["source_type"] == "generated_repeat":
            assert (
                abs(positions[str(row["item_id"])] - positions[str(row["repeat_of"])])
                >= 50
            )
            assert (
                public_by_id[str(row["item_id"])]["image"]
                == public_by_id[str(row["repeat_of"])]["image"]
            )
    assert all(
        "true_class" not in row for row in mapping if row["source_type"] == "generated"
    )


def test_scoring_drives_support_mismatch_and_inconclusive_decisions() -> None:
    mapping = _mapping_with_150_proxy_positives()
    supported = task._score_review(mapping, _answers_with_mismatches(mapping, 8))
    assert supported["review_quality"]["passed"] is True
    assert supported["proxy_positive_validity"]["e_d_positive_count_in_sample"] == 150
    assert (
        supported["decision"]["classification"] == "CLASSIFIER_PROXY_LOCALLY_SUPPORTED"
    )

    mismatched = task._score_review(mapping, _answers_with_mismatches(mapping, 30))
    assert (
        mismatched["decision"]["classification"]
        == "CLASSIFIER_PROXY_MISMATCH_SUPPORTED"
    )

    inconclusive = task._score_review(mapping, _answers_with_mismatches(mapping, 15))
    assert (
        inconclusive["decision"]["classification"]
        == "CLASSIFIER_PROXY_EVIDENCE_INSUFFICIENT"
    )

    bad_quality_answers = _answers_with_mismatches(mapping, 8)
    anchors = [row for row in mapping if row["source_type"] == "real_target"]
    for row in anchors[:8]:
        bad_quality_answers[str(row["item_id"])] = str(
            (int(row["true_class"]) + 1) % 10
        )
    invalid = task._score_review(mapping, bad_quality_answers)
    assert invalid["decision"]["status"] == "REVIEW_INCONCLUSIVE"
    assert invalid["decision"]["classification"] == "REVIEW_QUALITY_CHECK_FAILED"
    closeout = task._render_closeout_markdown(invalid)
    assert "效度结论证据不足" in closeout
    assert "仅作描述性结果" in closeout
    assert "不据此认定代理失效" in closeout
    assert "测量无效" not in closeout
    assert "8/150" in closeout


def test_rating_import_rejects_missing_duplicate_and_invalid_answers(
    tmp_path: Path,
) -> None:
    mapping = _records()
    answers = _ratings_for(mapping)
    valid = tmp_path / "valid.csv"
    _write_ratings(valid, mapping, answers)
    assert task._parse_ratings(valid, mapping) == answers

    invalid_cases = [
        ("item_id,answer\nitem0000,0\n", "missing"),
        ("item_id,answer\nitem0000,0\nitem0000,1\n", "repeats"),
        ("item_id,answer\nitem0000,10\n", "invalid rating"),
        ("item_id,answer\nunknown,0\n", "unknown"),
    ]
    for index, (content, expected_text) in enumerate(invalid_cases):
        path = tmp_path / f"invalid_{index}.csv"
        path.write_text(content, encoding="utf-8")
        try:
            task._parse_ratings(path, mapping)
        except task.ProxyValidityError as error:
            assert expected_text in str(error)
        else:
            raise AssertionError(f"invalid CSV was accepted: {content!r}")


def test_full_prepare_and_score_roundtrip_archives_actual_answers(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "attempt_01"
    output_dir.mkdir()
    prepared = task._write_blind_packet(output_dir, _synthetic_data())
    assert prepared["state"] == "PREPARED_AWAITING_RATINGS"
    mapping_payload = json.loads(
        (output_dir / "analysis" / "review_mapping.json").read_text(encoding="utf-8")
    )
    mapping = mapping_payload["items"]
    answers = _answers_with_mismatches(mapping, 8)
    ratings = tmp_path / "complete_ratings.csv"
    _write_ratings(ratings, mapping, answers)
    report = task._finalize(output_dir, ratings)
    assert report["decision"]["classification"] == "CLASSIFIER_PROXY_LOCALLY_SUPPORTED"
    assert (
        output_dir / "analysis" / "raw_answers.csv"
    ).read_bytes() == ratings.read_bytes()
    with (output_dir / "analysis" / "paired_rows.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        paired_rows = list(csv.DictReader(handle))
    assert len(paired_rows) == 200
    assert sum(row["human_matches_requested"] == "0" for row in paired_rows) == 8
    closeout = (output_dir / "report.md").read_text(encoding="utf-8")
    headings = [
        "结果与状态：",
        "科学认识：",
        "目标缺口：",
        "工程完成与验证：",
        "决策影响与下一步：",
    ]
    positions = [closeout.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert (
        json.loads((output_dir / "report.json").read_text(encoding="utf-8"))["decision"]
        == report["decision"]
    )
    assert (
        json.loads((output_dir / "completed.json").read_text(encoding="utf-8"))["state"]
        == "VALID_COMPLETED"
    )
