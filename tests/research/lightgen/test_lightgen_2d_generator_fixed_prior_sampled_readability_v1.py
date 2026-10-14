from __future__ import annotations

import csv
import json

import pytest
import torch

from scripts.research.lightgen import audit_lightgen_2d_generator_fixed_prior_sampled_readability_v1 as subject


def _review_fixture() -> tuple[list[dict[str, object]], dict[str, str]]:
    mapping: list[dict[str, object]] = []
    answers: dict[str, str] = {}
    primaries: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    next_id = 1
    for digit in range(10):
        for within_class in range(10):
            pair_id = f"{2026100961 + within_class // 2}:{digit}:{within_class % 2}"
            seed = 2026100961 + within_class // 2
            draw = within_class % 2
            pair = {}
            for arm in ("A5400", "B5400"):
                item_id = f"FJ{next_id:03d}"
                next_id += 1
                row = {
                    "item_id": item_id, "source_type": f"{arm}_primary", "arm": arm,
                    "pair_id": pair_id, "requested_class": digit, "seed": seed,
                    "draw_index": draw, "image_sha256": f"{pair_id}-{arm}",
                }
                mapping.append(row)
                pair[arm] = row
            primaries[(digit, within_class)] = pair

    repeated_pairs = [(digit, pair_index) for digit in range(10) for pair_index in range(2)]
    for digit, pair_index in repeated_pairs:
        pair = primaries[(digit, pair_index)]
        for arm in ("A5400", "B5400"):
            original = pair[arm]
            item_id = f"FJ{next_id:03d}"
            next_id += 1
            mapping.append({
                "item_id": item_id, "source_type": f"{arm}_repeat", "arm": arm,
                "pair_id": original["pair_id"], "requested_class": digit,
                "seed": original["seed"], "draw_index": original["draw_index"],
                "repeat_of": original["item_id"], "image_sha256": original["image_sha256"],
            })

    for digit in range(10):
        for _ in range(2):
            item_id = f"FJ{next_id:03d}"
            next_id += 1
            mapping.append({"item_id": item_id, "source_type": "real_control", "true_class": digit, "image_sha256": f"real-{item_id}"})
            answers[item_id] = str(digit)
    for _ in range(10):
        item_id = f"FJ{next_id:03d}"
        next_id += 1
        mapping.append({"item_id": item_id, "source_type": "black_control", "image_sha256": "black"})
        answers[item_id] = "unclear"

    for digit in range(10):
        for within_class in range(10):
            pair = primaries[(digit, within_class)]
            b_only = within_class < 2
            answers[str(pair["A5400"]["item_id"])] = "unclear"
            answers[str(pair["B5400"]["item_id"])] = str(digit) if b_only else "unclear"
    for row in mapping:
        if str(row["source_type"]).endswith("_repeat"):
            answers[str(row["item_id"])] = answers[str(row["repeat_of"])]
    ordered_pairs = [(digit, index) for digit in range(10) for index in range(2)]
    ordered_pairs.extend((digit, index) for digit in range(10) for index in range(2, 10))
    for index, (digit, within_class) in enumerate(ordered_pairs):
        pair = primaries[(digit, within_class)]
        early_arm = "A5400" if index < 50 else "B5400"
        late_arm = "B5400" if early_arm == "A5400" else "A5400"
        pair[early_arm]["question_number"] = index + 1
        pair[late_arm]["question_number"] = 151 + index
        if index < 20:
            for row in mapping:
                if row.get("pair_id") == pair[early_arm]["pair_id"] and row["source_type"].endswith("_repeat"):
                    row["question_number"] = 251 + index if row["arm"] == "A5400" else 101 + index
    control_index = 0
    for row in mapping:
        if row["source_type"] in {"real_control", "black_control"}:
            row["question_number"] = 121 + control_index
            control_index += 1
    mapping.sort(key=lambda row: int(row["question_number"]))
    id_remap = {str(row["item_id"]): f"FJ{int(row['question_number']):03d}" for row in mapping}
    for row in mapping:
        row["item_id"] = id_remap[str(row["item_id"])]
        if "repeat_of" in row:
            row["repeat_of"] = id_remap[str(row["repeat_of"])]
    answers = {id_remap[item_id]: answer for item_id, answer in answers.items()}
    return mapping, answers


def test_exact_mcnemar_uses_only_discordant_pairs() -> None:
    assert subject._exact_mcnemar_p(0, 0) == 1.0
    assert subject._exact_mcnemar_p(10, 0) == pytest.approx(2 / 2**10)
    assert subject._exact_mcnemar_p(10, 10) == 1.0


def test_human_pair_decision_requires_qc_and_ignores_classifier_proxy() -> None:
    mapping, answers = _review_fixture()
    result = subject._score_results(mapping, answers)
    assert result["quality_checks"]["passed"] is True
    assert result["primary"]["B_only"] == 20
    assert result["primary"]["A_only"] == 0
    assert result["primary"]["D_human"] == pytest.approx(0.2)
    assert result["classification"] == "B5400_SINGLE_REVIEWER_RELATIVE_READABILITY_SUPPORTED"
    assert result["classifier_proxy_validity_claim"] is False

    failed_answers = dict(answers)
    real_ids = [str(row["item_id"]) for row in mapping if row["source_type"] == "real_control"]
    for item_id in real_ids[:3]:
        failed_answers[item_id] = "0" if failed_answers[item_id] != "0" else "1"
    failed = subject._score_results(mapping, failed_answers)
    assert failed["quality_checks"]["real_digit_correct"] == 17
    assert failed["classification"] == "REVIEW_INCONCLUSIVE"


def test_generated_review_page_escapes_javascript_line_endings() -> None:
    page = subject._render_review_html(
        [{"item_id": "FJ001", "image": "data:image/png;base64,AA=="}],
        "test-review",
    )
    assert 'lines.join("\\r\\n")' in page
    assert 'split(/\\r?\\n/)' in page


def test_blind_packet_is_balanced_spaced_and_contains_no_labels(tmp_path, monkeypatch) -> None:
    def fake_views(_target, selected):
        views = {}
        for row in selected:
            seed, digit, draw = int(row["seed"]), int(row["label"]), int(row["draw_index"])
            for arm_index, arm in enumerate(("A5400", "B5400")):
                level = ((seed + digit * 7 + draw * 11 + arm_index * 13) % 70 + 10) / 100
                views[(seed, digit, draw, arm)] = torch.full((1, 28, 28), level, dtype=torch.float32)
        return views

    monkeypatch.setattr(subject, "_load_review_views", fake_views)
    monkeypatch.setattr(
        subject.generation.capacity.canonical,
        "cmos_view",
        lambda raw: torch.full((raw.shape[0], 1, 28, 28), 0.25, dtype=torch.float32),
    )
    labels = torch.tensor([digit for digit in range(10) for _ in range(20)] + [digit for digit in range(10) for _ in range(30)])
    inputs = {
        "labels": labels,
        "population": {"source_row_indices": list(range(500))},
        "readout": {"target_slm_intensity": torch.empty((500, 1, 200, 200))},
    }
    result = subject._prepare_review(tmp_path, inputs)
    mapping = result["mapping"]
    validated = subject._validate_review_mapping(mapping)
    assert len(mapping) == 270
    assert result["eligibility"]["page_counts"] == {str(page + 1): 45 for page in range(6)}
    assert result["eligibility"]["first_presented_endpoint_counts"] == {"A5400": 50, "B5400": 50}
    assert result["eligibility"]["repeat_png_bytes_identical"] is True
    assert set(result["public_items"][0]) == {"item_id", "image"}
    html = (tmp_path / "blind_review.html").read_text(encoding="utf-8")
    assert "requested_class" not in html
    assert "e_d_prediction" not in html
    assert "A5400" not in html and "B5400" not in html
    for row in mapping:
        if str(row["source_type"]).endswith("_repeat"):
            assert abs(int(row["item_id"][2:]) - int(str(row["repeat_of"])[2:])) >= subject.MIN_SPACING
    for pair in validated["pairs"].values():
        assert abs(int(pair["A5400"]["question_number"]) - int(pair["B5400"]["question_number"])) >= subject.MIN_SPACING


def test_review_mapping_rejects_wrong_task_ids_and_mismatched_pair_identity(tmp_path) -> None:
    mapping, _ = _review_fixture()
    with pytest.raises(subject.ReadabilityComparisonError, match="unique item IDs"):
        subject._validate_review_mapping([{**mapping[0], "item_id": "Q001"}, *mapping[1:]])

    mismatched = [dict(row) for row in mapping]
    b_primary = next(row for row in mismatched if row["source_type"] == "B5400_primary")
    b_primary["requested_class"] = (int(b_primary["requested_class"]) + 1) % 10
    with pytest.raises(subject.ReadabilityComparisonError, match="paired images do not share"):
        subject._validate_review_mapping(mismatched)

    path = tmp_path / "ratings.csv"
    path.write_text("item_id,answer\nQ001,1\n", encoding="utf-8")
    with pytest.raises(subject.ReadabilityComparisonError, match="unknown rating item ID"):
        subject._parse_ratings(path, mapping)


@pytest.mark.parametrize("first_presentation_boundary", [False, True])
def test_complete_rating_submission_is_archived_through_final_report(tmp_path, first_presentation_boundary) -> None:
    mapping, answers = _review_fixture()
    if first_presentation_boundary:
        for row in mapping:
            if row["source_type"] == "B5400_repeat" and int(row["question_number"]) >= 107:
                answers[str(row["item_id"])] = "unclear"
                answers[str(row["repeat_of"])] = "unclear"
        answers["FJ101"] = "unclear"
        answers["FJ102"] = "unclear"
    output = tmp_path / "attempt_01"
    (output / "private").mkdir(parents=True)
    (output / "prepared.json").write_text(
        json.dumps({"task_id": subject.TASK_ID, "state": "PREPARED_AWAITING_RATINGS"}),
        encoding="utf-8",
    )
    (output / "private" / "review_mapping.json").write_text(
        json.dumps({"task_id": subject.TASK_ID, "items": mapping}),
        encoding="utf-8",
    )
    measurement_report = {
        "numerical_auxiliary": {"shape_statistics": {"confirmation": {"28": {"B_minus_A": {"energy_distance": -0.01}}}}},
        "scope": {"reviewer_count": 1}, "preflight": {"status": "READY"},
    }
    (output / "measurement_report.json").write_text(json.dumps(measurement_report), encoding="utf-8")
    ratings = tmp_path / "complete_ratings.csv"
    submitted_answers = dict(answers)
    chinese_unclear_id = next(str(row["item_id"]) for row in mapping if row["source_type"] == "black_control")
    submitted_answers[chinese_unclear_id] = "无法辨认"
    with ratings.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "answer"])
        writer.writerows((row["item_id"], submitted_answers[str(row["item_id"])]) for row in mapping)

    report = subject._score_output(output, ratings)

    assert report["status"] == "VALID_COMPLETED"
    expected = "EVIDENCE_INSUFFICIENT" if first_presentation_boundary else "B5400_SINGLE_REVIEWER_RELATIVE_READABILITY_SUPPORTED"
    assert report["classification"] == expected
    if first_presentation_boundary:
        assert report["human_results"]["quality_checks"]["passed"] is True
        assert report["human_results"]["primary"]["B_only"] == 4
        assert report["human_results"]["primary"]["mcnemar_exact_two_sided_p"] == 0.125
        paired_rows = subject._read_csv(output / "analysis" / "paired_rows.csv")
        assert next(row for row in paired_rows if row["B_item_id"] == "FJ101")["B_answer"] == "unclear"
    assert len(subject._read_csv(output / "analysis" / "raw_answers.csv")) == subject.ITEM_COUNT
    normalized_answers = subject._read_csv(output / "analysis" / "raw_answers.csv")
    assert next(row["answer"] for row in normalized_answers if row["item_id"] == chinese_unclear_id) == "unclear"
    assert len(subject._read_csv(output / "analysis" / "paired_rows.csv")) == subject.PRIMARY_PAIR_COUNT
    assert (output / "ratings_submission.csv").is_file()
    assert json.loads((output / "completed.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"
    assert "## 科学认识" in (output / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "csv_text",
    [
        "item_id,answer\nQ001,1\n",
        "item_id,answer\nQ001,1\nQ001,2\n",
        "item_id,answer\nQ001,?\nQ002,2\n",
    ],
)
def test_rating_parser_rejects_missing_duplicate_and_illegal_answers(tmp_path, csv_text) -> None:
    path = tmp_path / "ratings.csv"
    path.write_text(csv_text, encoding="utf-8")
    mapping = [{"item_id": "Q001"}, {"item_id": "Q002"}]
    with pytest.raises(subject.ReadabilityComparisonError):
        subject._parse_ratings(path, mapping)
