from __future__ import annotations

import csv
from collections import Counter

import pytest
import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_measurement_stability_v1 as task,
)


def _synthetic_data() -> dict[str, object]:
    requested = torch.arange(10, dtype=torch.long).repeat_interleave(200)
    sample_index = torch.arange(200, dtype=torch.long).repeat(10)
    sample_views = torch.zeros((2000, 1, 28, 28), dtype=torch.float32)
    for digit in range(10):
        for sample in range(200):
            sample_views[digit * 200 + sample, 0, digit : digit + 2, sample % 26 : sample % 26 + 2] = 0.75
    empirical_labels = torch.arange(10, dtype=torch.long).repeat_interleave(20)
    empirical_views = torch.zeros((200, 1, 28, 28), dtype=torch.float32)
    for position, digit in enumerate(empirical_labels.tolist()):
        empirical_views[position, 0, digit : digit + 2, position % 26 : position % 26 + 2] = 0.8
    target_labels = torch.cat((torch.arange(10).repeat_interleave(300), torch.arange(10).repeat_interleave(30))).long()
    target_views = torch.zeros((3300, 1, 28, 28), dtype=torch.float32)
    for position, digit in enumerate(target_labels.tolist()):
        target_views[position, 0, digit : digit + 2, position % 26 : position % 26 + 2] = 0.9
    return {
        "sample_views": sample_views,
        "sample_requested": requested,
        "sample_index": sample_index,
        "sample_e_d_prediction": requested.clone(),
        "empirical_views": empirical_views,
        "empirical_labels": empirical_labels,
        "empirical_rows": torch.arange(20_000, 20_200),
        "empirical_e_d_prediction": empirical_labels.clone(),
        "target_views": target_views,
        "labels": target_labels,
        "source_rows": torch.arange(30_000, 33_300),
        "generation_dir": "synthetic-generation",
        "target_readout_path": "synthetic-readout.pt",
    }


def _packet() -> list[dict[str, object]]:
    original = task.proxy.saved_evidence._image_data_uri
    task.proxy.saved_evidence._image_data_uri = lambda image: "data:image/png;base64," + __import__("base64").b64encode(
        bytes([int(float(image.max().item()) * 100) % 256])
    ).decode("ascii")
    try:
        (_, mapping), _ = task._select_records(
            _synthetic_data(),
            {"conditional": set(), "empirical_ols": set(), "real_target": set()},
        )
        return mapping
    finally:
        task.proxy.saved_evidence._image_data_uri = original


def _answers(mapping: list[dict[str, object]]) -> dict[str, str]:
    by_id = {str(row["item_id"]): row for row in mapping}
    result = {}
    for row in mapping:
        kind = row["source_type"]
        if kind == "conditional_sample" or kind == "empirical_ols":
            result[str(row["item_id"])] = str(row["requested_class"])
        elif kind == "real_target":
            result[str(row["item_id"])] = str(row["true_class"])
        elif kind == "black_control":
            result[str(row["item_id"])] = "unclear"
        elif str(kind).endswith("_repeat"):
            result[str(row["item_id"])] = result[str(row["repeat_of"])]
    assert set(result) == set(by_id)
    return result


def test_packet_has_fixed_composition_blindness_and_spaced_byte_identical_repeats() -> None:
    mapping = _packet()
    assert len(mapping) == 100
    assert Counter(str(row["source_type"]) for row in mapping) == Counter(
        {
            "conditional_sample": 30,
            "empirical_ols": 30,
            "real_target": 10,
            "black_control": 10,
            "conditional_sample_repeat": 10,
            "empirical_ols_repeat": 10,
        }
    )
    public, private = task._select_records(
        _synthetic_data(), {"conditional": set(), "empirical_ols": set(), "real_target": set()}
    )[0]
    html = task._render_html(public, "test-review")
    assert all(set(item) == {"item_id", "image"} for item in public)
    assert not any(secret in html for secret in ("requested_class", "true_class", "e_d_prediction", "source_row_index", "sample_index"))
    assert "if(answer&&!allowed.has(answer))" in html
    assert "else delete answers[id]" in html
    del private
    positions = {str(row["item_id"]): index for index, row in enumerate(mapping)}
    by_id = {str(row["item_id"]): row for row in mapping}
    repeats = [row for row in mapping if str(row["source_type"]).endswith("_repeat")]
    assert min(positions[str(row["item_id"])] - positions[str(row["repeat_of"])] for row in repeats) >= 50
    assert all(by_id[str(row["repeat_of"])]["image_sha256"] == row["image_sha256"] for row in repeats)
    assert all(positions[str(row["repeat_of"])] < 40 for row in repeats)


def test_historical_image_identity_exclusions_are_applied_before_sampling() -> None:
    data = _synthetic_data()
    excluded = {
        "conditional": {(0, 0), (0, 1)},
        "empirical_ols": {20_000, 20_001},
        "real_target": {30_000, 30_300},
    }
    (_, mapping), _ = task._select_records(data, excluded)
    selected_conditional = {(int(row["requested_class"]), int(row["sample_index"])) for row in mapping if row["source_type"] == "conditional_sample"}
    selected_empirical = {int(row["source_row_index"]) for row in mapping if row["source_type"] == "empirical_ols"}
    selected_targets = {int(row["source_row_index"]) for row in mapping if row["source_type"] == "real_target"}
    assert not selected_conditional.intersection(excluded["conditional"])
    assert not selected_empirical.intersection(excluded["empirical_ols"])
    assert not selected_targets.intersection(excluded["real_target"])


def test_selection_fails_closed_when_any_class_is_ineligible() -> None:
    data = _synthetic_data()
    excluded = {
        "conditional": {(0, index) for index in range(200)},
        "empirical_ols": set(),
        "real_target": set(),
    }
    with pytest.raises(task.StabilityError, match="too few eligible conditional samples"):
        task._select_records(data, excluded)


def test_ratings_parser_rejects_missing_duplicate_and_illegal_answers(tmp_path) -> None:
    mapping = _packet()
    valid = _answers(mapping)
    path = tmp_path / "ratings.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "answer"])
        writer.writerows(valid.items())
    parsed, preserved = task._parse_ratings(path, mapping)
    assert parsed == valid and len(preserved) == 100

    missing = tmp_path / "missing.csv"
    with missing.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "answer"])
        writer.writerows(list(valid.items())[:-1])
    with pytest.raises(task.StabilityError, match="every item exactly once"):
        task._parse_ratings(missing, mapping)

    duplicate = tmp_path / "duplicate.csv"
    first = next(iter(valid.items()))
    with duplicate.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "answer"])
        writer.writerows(list(valid.items())[:-1] + [first])
    with pytest.raises(task.StabilityError, match="duplicate"):
        task._parse_ratings(duplicate, mapping)

    invalid = tmp_path / "invalid.csv"
    invalid_answers = dict(valid)
    invalid_answers[next(iter(invalid_answers))] = "10"
    with invalid.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "answer"])
        writer.writerows(invalid_answers.items())
    with pytest.raises(task.StabilityError, match="invalid rating"):
        task._parse_ratings(invalid, mapping)


def test_pilot_quality_checks_use_separate_9_of_10_boundaries() -> None:
    mapping = _packet()
    answers = _answers(mapping)
    by_id = {str(row["item_id"]): row for row in mapping}
    black_changed = False
    for row in mapping:
        if row["source_type"] == "real_target" and row["true_class"] == 0:
            answers[str(row["item_id"])] = "1"
        if row["source_type"] == "black_control" and not black_changed:
            answers[str(row["item_id"])] = "0"
            black_changed = True
    for source_type in ("conditional_sample_repeat", "empirical_ols_repeat"):
        repeat = next(row for row in mapping if row["source_type"] == source_type)
        parent = by_id[str(repeat["repeat_of"])]
        parent_answer = answers[str(parent["item_id"])]
        answers[str(repeat["item_id"])] = "unclear" if parent_answer != "unclear" else "0"
    result = task._score(mapping, answers)
    assert result["classification"] == "PILOT_QUALITY_CHECKS_PASSED"
    assert result["quality_checks"]["real_target_correct"] == 9
    assert result["quality_checks"]["black_answered_unclear"] == 9
    assert result["quality_checks"]["repeat_consistency_by_source"]["conditional_sample"]["agree"] == 9
    assert result["quality_checks"]["repeat_consistency_by_source"]["empirical_ols"]["agree"] == 9

    for source_type in ("conditional_sample_repeat", "empirical_ols_repeat"):
        candidates = [row for row in mapping if row["source_type"] == source_type][:2]
        for repeat in candidates:
            parent_answer = answers[str(by_id[str(repeat["repeat_of"])]["item_id"])]
            answers[str(repeat["item_id"])] = "1" if parent_answer != "1" else "2"
    failed = task._score(mapping, answers)
    assert failed["classification"] == "PILOT_QUALITY_CHECKS_FAILED"
    assert failed["quality_checks"]["passed"] is False
