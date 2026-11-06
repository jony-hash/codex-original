"""Focused population and rating-consumer checks for the fixed C45k review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_c45k_phase_chart_readability_spec_v1 as task


def _selection_sources():
    cohorts = {}
    identity_to_row = {}
    for method in task.METHODS:
        for stream in task.STREAMS:
            labels = torch.arange(10).repeat_interleave(1_000)
            proposals = torch.arange(10_000)
            cohorts[(method, stream)] = {
                "requested_labels": labels,
                "proposal_ids": proposals,
                "component_ids": torch.zeros(10_000, dtype=torch.long),
            }
            for row, label in enumerate(labels.tolist()):
                identity_to_row[(method, stream, label, row)] = row
    return cohorts, identity_to_row


def _ratings_config():
    return {"review": {"bootstrap_seed": 2026102313, "bootstrap_resamples": 10_000,
                       "minimum_duplicate_consistency": 18, "minimum_anchor_correct": 18,
                       "minimum_duplicate_card_distance": 40}}


def test_task_input_overlay_redirects_only_local_supporting_evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    input_root = tmp_path / "isolated-inputs"
    input_root.mkdir()
    monkeypatch.setattr(task, "ROOT", root)

    config, paths = task._load_config(
        Path(r"D:\codex-original\configs\research\lightgen\c45k_phase_chart_readability_spec_v1.yaml"),
        input_root)

    assert config["task_id"] == task.TASK_ID
    assert paths["replication_dir"] == input_root / "replication"
    assert paths["prior_readability_key"] == input_root / "prior_readability_key.json"
    assert paths["decoder_readability_key"] == input_root / "decoder_readability_key.json"
    assert paths["phase_readback_receipt"] == input_root / "phase_readback_receipt.json"
    assert paths["phase_evidence_dir"].is_relative_to(root)
    assert paths["output_dir"].is_relative_to(root)


def test_selection_preserves_every_stratum_and_excludes_known_rows():
    cohorts, identity_to_row = _selection_sources()
    excluded = {("original", 0, 0, row): {"historically_reviewed"} for row in range(12)}
    selected = task._select_population(cohorts, identity_to_row, excluded, seed=2026102311)

    assert len(selected) == 1_200
    assert len({row["sample_id"] for row in selected}) == 1_200
    assert len({row["display_id"] for row in selected}) == 1_200
    assert all((row["method"], row["stream"], row["requested_label"], row["proposal_id"]) not in excluded
               for row in selected)
    assert sum(row["preflight"] for row in selected) == 120
    assert sum(row["stream"] == 0 and row["preflight_rank"] == 0 for row in selected) == 20
    for method in task.METHODS:
        for stream in task.STREAMS:
            for label in task.CLASSES:
                assert sum(row["method"] == method and row["stream"] == stream
                           and row["requested_label"] == label for row in selected) == 20


def test_rating_consumer_rejects_missing_duplicate_or_foreign_cards(tmp_path):
    key = {"review_id": "review-1", "cards": [{"card_id": "R0001"}, {"card_id": "R0002"}]}
    path = tmp_path / "ratings.json"

    for cards in (
        [{"card_id": "R0001", "rating": 3}],
        [{"card_id": "R0001", "rating": 3}, {"card_id": "R0001", "rating": 4}],
        [{"card_id": "R0001", "rating": 3}, {"card_id": "R9999", "rating": 4}],
    ):
        path.write_text(json.dumps({"schema": task.RATING_SCHEMA, "task_id": task.TASK_ID,
                                    "review_id": "review-1", "cards": cards}), encoding="utf-8")
        with pytest.raises(task.ReviewError):
            task._read_ratings(path, key)


def _valid_review_key_and_ratings():
    cards = []
    ratings = {}
    row_id = 0
    duplicate_sources = []
    for method in task.METHODS:
        for stream in task.STREAMS:
            for label in task.CLASSES:
                for rank in range(20):
                    sample_id = f"{method}:G4:{stream}:{label}:{row_id}"
                    card = {"card_id": f"R{row_id + 1:04d}", "kind": "generated", "method": method,
                            "stream": stream, "requested_label": label, "sample_id": sample_id,
                            "saved_metrics": {"E_D_prediction": label, "E_C_prediction": label}}
                    cards.append(card)
                    # The template arm is always read correctly; original is always a different digit.
                    ratings[card["card_id"]] = label if method == "template" else (label + 1) % 10
                    if stream == 0 and rank == 0:
                        duplicate_sources.append((sample_id, ratings[card["card_id"]]))
                    row_id += 1
    for index, (sample_id, rating) in enumerate(duplicate_sources):
        card_id = f"R{row_id + 1:04d}"
        cards.append({"card_id": card_id, "kind": "duplicate", "duplicate_group": f"D{index:02d}",
                      "source_sample_id": sample_id})
        ratings[card_id] = rating
        row_id += 1
    for index in range(20):
        card_id = f"R{row_id + 1:04d}"
        label = index // 2
        cards.append({"card_id": card_id, "kind": "real_anchor", "target_label": label})
        ratings[card_id] = label
        row_id += 1
    return {"cards": cards}, ratings


def test_review_scoring_consumes_one_primary_comparison_and_qc():
    key, ratings = _valid_review_key_and_ratings()
    result = task._summarize_review(key, ratings, _ratings_config())

    assert result["status"] == "COMPLETE"
    assert result["decision"] == "TEMPLATE_HUMAN_GAIN_SUPPORTED"
    assert result["primary_comparison"]["difference"] == 1.0
    assert result["primary_comparison"]["ci_95"][0] > 0.0
    assert result["human_results"]["original"]["count"] == 600
    assert result["human_results"]["template"]["count"] == 600
    assert result["quality_control"]["passed"] is True
    assert set(result["evaluator_disagreements"]) == {"original", "template"}


def test_review_quality_control_failure_blocks_scientific_interpretation():
    key, ratings = _valid_review_key_and_ratings()
    anchors = [card for card in key["cards"] if card["kind"] == "real_anchor"]
    for card in anchors[:3]:
        ratings[card["card_id"]] = (card["target_label"] + 1) % 10

    result = task._summarize_review(key, ratings, _ratings_config())

    assert result["status"] == "REVIEW_INVALID_QC"
    assert result["decision"] == "REVIEW_INVALID_QC"
    assert result["quality_control"]["real_target_correct"]["count"] == 17
    assert result["quality_control"]["passed"] is False


def test_bootstrap_difference_preserves_full_effect_magnitude_and_matches_independent_recalculation():
    key, ratings = _valid_review_key_and_ratings()
    for card in key["cards"]:
        if card.get("kind") != "generated":
            continue
        label = int(card["requested_label"])
        ratings[card["card_id"]] = label if card["method"] == "template" else (label + 1) % 10

    config = _ratings_config()
    production = task._summarize_review(key, ratings, config)

    assert production["primary_comparison"]["difference"] == 1.0
    assert np.allclose(production["primary_comparison"]["ci_95"], [1.0, 1.0], rtol=0.0, atol=1e-15)
    independent = task._independently_recompute_primary_review(key, ratings, config, production)
    assert independent["status"] == "PASSED"
    assert independent["production_difference_and_interval_match"] is True


def test_review_package_has_exact_population_and_separated_hidden_repeats(tmp_path):
    rows = []
    metric_rows = {}
    image = torch.zeros((1, 200, 200))
    for method in task.METHODS:
        for stream in task.STREAMS:
            for label in task.CLASSES:
                for rank in range(20):
                    sample_id = f"{method}:G4:{stream}:{label}:{rank}"
                    display_id = f"generated_{len(rows) + 1:04d}"
                    row = {"method": method, "stream": stream, "requested_label": label,
                           "cohort_row": rank, "proposal_id": rank, "component_id": 0,
                           "sample_id": sample_id, "display_id": display_id,
                           "preflight": rank < 2, "preflight_rank": rank}
                    rows.append(row)
                    task._save_png(tmp_path / "review_cards" / f"{display_id}.png", image)
                    metric_rows[(method, stream, label, rank)] = {"E_D_prediction": label,
                                                                  "E_C_prediction": label}
    examples = {"targets28": torch.zeros((20, 1, 28, 28)),
                "labels": torch.arange(10).repeat_interleave(2),
                "rows": torch.arange(20)}
    config = {"review": {"shuffle_seed": 2026102312, "minimum_duplicate_card_distance": 40}}

    package = task._make_review_package(rows, examples, metric_rows, tmp_path, config, "run-1")
    task._validate_review_package(tmp_path, package["key"])

    assert package["public"]["card_count"] == 1_240
    assert len(list((tmp_path / "review_cards").glob("*.png"))) == 1_240
    assert len({card["card_id"] for card in package["key"]["cards"]}) == 1_240


def test_public_review_page_has_no_sample_identity_or_arm_labels(tmp_path):
    cards = [{"card_id": "R0001", "image": "review_cards/generated_0001.png"}]
    page = task._build_page("review-1", cards)
    key = {"review_id": "review-1", "cards": [{"card_id": "R0001", "image": cards[0]["image"],
                                                    "method": "template", "requested_label": 7}]}
    (tmp_path / "blind_review.html").write_text(page, encoding="utf-8")
    task._validate_public_page(tmp_path, key)
    leaked = [{"card_id": "R0001", "image": "review_cards/generated_0001.png", "requested_label": 7}]
    (tmp_path / "blind_review.html").write_text(task._build_page("review-1", leaked), encoding="utf-8")
    with pytest.raises(task.ReviewError):
        task._validate_public_page(tmp_path, key)

