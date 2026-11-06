from __future__ import annotations

import csv
import math
import base64
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import audit_lightgen_small_class9_saved_defect_diagnosis_v1 as audit
from PIL import Image


def _conditions() -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in audit.CONDITIONS:
        logits = np.zeros((audit.SAMPLES_PER_CLASS, 10), dtype=np.float64)
        logits[:, 9] = 1.0
        logits[0, 9] = 0.0
        logits[0, 4] = 2.0
        predictions, margins, correct = audit._recompute_predictions(
            logits, np.full(audit.SAMPLES_PER_CLASS, 9, dtype=np.int64)
        )
        result[name] = {
            "images": np.zeros((audit.SAMPLES_PER_CLASS, 1, 28, 28), dtype=np.float32),
            "logits": logits,
            "predicted_labels": predictions,
            "identity_margin": margins,
            "view_mean": np.full(audit.SAMPLES_PER_CLASS, 0.2, dtype=np.float64),
            "camera_total_power_w": np.full(audit.SAMPLES_PER_CLASS, 1e-6, dtype=np.float64),
            "peak_w_per_pixel": np.full(audit.SAMPLES_PER_CLASS, 1e-9, dtype=np.float64),
            "scaled_peak": np.full(audit.SAMPLES_PER_CLASS, 0.5, dtype=np.float64),
            "clipping_area_fraction": np.zeros(audit.SAMPLES_PER_CLASS, dtype=np.float64),
            "clipping_region_power_share": np.zeros(audit.SAMPLES_PER_CLASS, dtype=np.float64),
            "excess_removed_power_share": np.zeros(audit.SAMPLES_PER_CLASS, dtype=np.float64),
        }
        assert correct.sum() == audit.SAMPLES_PER_CLASS - 1
    return result


def _valid_block(cell: str = "AA", label: int = 9) -> dict[str, object]:
    identity = {
        "schema": "lightgen_small_cmos_visibility_block.v1",
        "task_id": audit.ATTRIBUTION_TASK_ID,
        "cell": cell,
        "input_source": audit.CELL_INPUT[cell],
        "decoder": audit.CELL_DECODER[cell],
        "class_id": label,
        "sample_count": audit.SAMPLES_PER_CLASS,
        "input_run_id": "run-fixed",
        "decoder_run_id": "run-fixed",
        "conditions": list(audit._condition_names(cell)),
    }
    variants = {}
    for condition in audit._condition_names(cell):
        variants[condition] = {
            "images": torch.zeros((1, 1, 1, 1)).expand(audit.SAMPLES_PER_CLASS, 1, 28, 28),
            "logits": torch.zeros((1, 10)).expand(audit.SAMPLES_PER_CLASS, 10),
            "predicted_labels": torch.full((audit.SAMPLES_PER_CLASS,), label),
            "identity_margin": torch.zeros(audit.SAMPLES_PER_CLASS),
            "view_mean": torch.zeros(audit.SAMPLES_PER_CLASS),
            "camera_total_power_w": torch.ones(audit.SAMPLES_PER_CLASS),
            "peak_w_per_pixel": torch.ones(audit.SAMPLES_PER_CLASS),
            "scaled_peak": torch.ones(audit.SAMPLES_PER_CLASS),
            "clipping_area_fraction": torch.zeros(audit.SAMPLES_PER_CLASS),
            "clipping_region_power_share": torch.zeros(audit.SAMPLES_PER_CLASS),
            "excess_removed_power_share": torch.zeros(audit.SAMPLES_PER_CLASS),
        }
    block: dict[str, object] = {
        "identity": identity,
        "requested_labels": torch.full((audit.SAMPLES_PER_CLASS,), label),
        "proposal_ids": torch.arange(audit.SAMPLES_PER_CLASS),
        "conditions": variants,
    }
    if label == 9:
        block["native_raw_examples"] = torch.zeros((1, 1, 1, 1)).expand(10, 1, 200, 200)
    return block


def test_numpy_readout_consumed_by_native_report_check_and_rejects_wrong_order() -> None:
    gain = 2.0
    raw = np.full((200, 200), 0.25, dtype=np.float32)
    expected = np.full((28, 28), 0.5, dtype=np.float64)
    np.testing.assert_allclose(audit._numpy_production_readout(raw, gain), expected, rtol=0, atol=1e-7)

    ramp = np.zeros((200, 200), dtype=np.float32)
    ramp[:, 97:] = 1.0
    production = audit._numpy_production_readout(ramp, gain)
    already_clipped_then_rescaled = np.clip(ramp * gain, 0, 1) * 0.5
    correctly_rescaled_before_clip = np.clip(ramp * (gain * 0.5), 0, 1)
    pool_before_clamp = np.clip(audit._numpy_adaptive_average_pool(ramp * gain), 0, 1)
    assert not np.allclose(already_clipped_then_rescaled, correctly_rescaled_before_clip, rtol=1e-5, atol=1e-5)
    assert not np.allclose(production, pool_before_clamp, rtol=1e-5, atol=1e-5)

    blocks = {}
    for cell in audit.CELLS:
        block = _valid_block(cell)
        block["native_raw_examples"] = torch.full((10, 1, 200, 200), 0.25)
        block["conditions"][f"{cell}_g1p00"]["images"] = torch.full((audit.SAMPLES_PER_CLASS, 1, 28, 28), 0.5)
        blocks[(cell, 9)] = block
    checked = audit._validate_native_readout(blocks, gain)
    assert checked["status"] == "PASSED"
    assert checked["samples_checked"] == 40


def test_effective_channel_and_phase_invariant_overlap_match_analytic_values() -> None:
    coefficients = np.asarray([[1, 1, 0], [3, 0, 0]], dtype=np.complex128)
    total, effective = audit._effective_channels(coefficients)
    np.testing.assert_allclose(total, [2.0, 9.0])
    np.testing.assert_allclose(effective, [2.0, 1.0])
    phase = np.exp(1j * 0.73)
    overlap = audit._phase_invariant_overlap(coefficients, coefficients * phase)
    np.testing.assert_allclose(overlap, [1.0, 1.0], atol=1e-14)


def test_unit_mass_shape_is_scale_invariant_and_detects_spatial_perturbation() -> None:
    image = np.zeros((28, 28), dtype=np.float64)
    image[8:12, 7:10] = 1.0
    scaled = image * 5.0
    moved = np.roll(image, 5, axis=1)
    a = audit._spatial_features(image)
    b = audit._spatial_features(scaled)
    c = audit._spatial_features(moved)
    np.testing.assert_allclose(a["unit_mass"], b["unit_mass"])
    assert math.isclose(a["centroid_x"], b["centroid_x"])
    assert math.isclose(a["spread_rms"], b["spread_rms"])
    assert float(0.5 * np.abs(a["unit_mass"] - c["unit_mass"]).sum()) > 0
    assert c["centroid_x"] > a["centroid_x"]


def test_saved_logits_flow_through_csv_and_report_consumers(tmp_path: Path) -> None:
    conditions = _conditions()
    proposal_ids = np.arange(audit.SAMPLES_PER_CLASS)
    path = tmp_path / "metrics.csv"
    audit._write_metrics_csv(path, conditions, proposal_ids)
    audit._validate_metrics_csv(path, conditions, proposal_ids)

    report_conditions = {}
    for name, values in conditions.items():
        predictions, margins, correct = audit._recompute_predictions(
            values["logits"], np.full(audit.SAMPLES_PER_CLASS, 9, dtype=np.int64)
        )
        report_conditions[name] = {
            "recall_by_class": [0.0] * 9 + [float(correct.mean())],
            "identity_margin_by_class": [0.0] * 9 + [float(margins.mean())],
            "clipping_area_fraction_by_class": [0.0] * 9 + [float(values["clipping_area_fraction"].mean())],
        }
        assert np.array_equal(predictions, values["predicted_labels"])
    audit._validate_report_summaries({"results": {"conditions": report_conditions}}, conditions)

    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    rows[0]["predicted_label"] = "9"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(audit.EvidenceError, match="prediction"):
        audit._validate_metrics_csv(path, conditions, proposal_ids)


@pytest.mark.parametrize("corruption", ["cell", "class", "duplicate", "missing", "missing_proposal", "nonfinite"])
def test_saved_block_consumer_rejects_identity_and_completeness_defects(corruption: str) -> None:
    block = _valid_block()
    label = 9
    if corruption == "cell":
        block["identity"]["cell"] = "BB"
    elif corruption == "class":
        block["identity"]["class_id"] = 8
        block["requested_labels"][0] = 8
    elif corruption == "duplicate":
        block["proposal_ids"][1] = 0
    elif corruption == "missing":
        del block["conditions"]["AA_g1p00"]
    elif corruption == "missing_proposal":
        block["proposal_ids"] = block["proposal_ids"][:-1]
    elif corruption == "nonfinite":
        block["conditions"]["AA_g1p00"]["logits"] = torch.zeros((audit.SAMPLES_PER_CLASS, 10))
        block["conditions"]["AA_g1p00"]["logits"][0, 0] = float("nan")
    with pytest.raises(audit.EvidenceError):
        audit._validate_block(block, "AA", label, "run-fixed")


def _review_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict[tuple[str, int], dict[str, object]], np.ndarray]:
    conditions = {}
    blocks = {}
    proposals = np.arange(audit.SAMPLES_PER_CLASS)
    for cell in audit.CELLS:
        predicted = np.full(audit.SAMPLES_PER_CLASS, 9, dtype=np.int64)
        predicted[:400] = 4
        predicted[400:800] = 8
        values = {
            "images": np.zeros((audit.SAMPLES_PER_CLASS, 1, 28, 28), dtype=np.float32),
            "logits": np.zeros((audit.SAMPLES_PER_CLASS, 10), dtype=np.float64),
            "predicted_labels": predicted,
            "identity_margin": np.zeros(audit.SAMPLES_PER_CLASS),
            "view_mean": np.zeros(audit.SAMPLES_PER_CLASS),
            "camera_total_power_w": np.ones(audit.SAMPLES_PER_CLASS),
            "peak_w_per_pixel": np.ones(audit.SAMPLES_PER_CLASS),
            "scaled_peak": np.ones(audit.SAMPLES_PER_CLASS),
            "clipping_area_fraction": np.zeros(audit.SAMPLES_PER_CLASS),
            "clipping_region_power_share": np.zeros(audit.SAMPLES_PER_CLASS),
            "excess_removed_power_share": np.zeros(audit.SAMPLES_PER_CLASS),
        }
        conditions[f"{cell}_g1p00"] = values
        block9 = _valid_block(cell, 9)
        block9["conditions"][f"{cell}_g1p00"]["images"] = torch.zeros((audit.SAMPLES_PER_CLASS, 1, 28, 28))
        blocks[(cell, 9)] = block9
        for label in (4, 8):
            block = _valid_block(cell, label)
            block["conditions"][f"{cell}_g1p00"]["images"] = torch.zeros((audit.SAMPLES_PER_CLASS, 1, 28, 28))
            blocks[(cell, label)] = block
    return conditions, blocks, proposals


def test_blind_sample_is_fixed_shuffled_and_has_no_duplicate_class_proposal() -> None:
    conditions, blocks, proposals = _review_fixture()
    first = audit._selection_entries(conditions, blocks, proposals)
    second = audit._selection_entries(conditions, blocks, proposals)
    assert first == second
    assert len(first) == 60
    assert len({(row["requested_label"], row["proposal_id"]) for row in first}) == 60
    assert len({row["display_order"] for row in first}) == 60
    for cell in audit.CELLS:
        rows = [row for row in first if row["cell"] == cell]
        assert len(rows) == 15
        assert sum(row["stratum"] == "requested9_ed4" for row in rows) == 5
        assert sum(row["stratum"] == "requested9_ed8" for row in rows) == 5
        assert sum(row["stratum"] == "requested9_ed9" for row in rows) == 3
        assert sum(row["stratum"].endswith("unfiltered") for row in rows) == 2


def test_blind_card_image_matches_hidden_mapping() -> None:
    conditions, blocks, proposals = _review_fixture()
    entries = audit._selection_entries(conditions, blocks, proposals)
    entry = entries[0]
    images = torch.zeros((audit.SAMPLES_PER_CLASS, 1, 28, 28))
    images[entry["cohort_index"], 0, 2, 3] = 0.75
    blocks[(entry["cell"], entry["requested_label"])]["conditions"][entry["condition_id"]]["images"] = images
    card = audit._public_cards([entry], blocks)[0]
    assert set(card) == {"blind_id", "image"}
    data = base64.b64decode(card["image"].split(",", 1)[1])
    shown = np.asarray(Image.open(io.BytesIO(data)))
    assert shown.shape == (28 * 12, 28 * 12)
    assert int(shown[2 * 12, 3 * 12]) == round(0.75 * 255)


def _valid_manifest() -> dict[str, object]:
    conditions, blocks, proposal_ids = _review_fixture()
    entries = audit._selection_entries(conditions, blocks, proposal_ids)
    return audit._blind_manifest(entries, "review-fixed", conditions)


def test_ratings_consumer_requires_full_valid_blind_ids_and_routes_primary20() -> None:
    manifest = _valid_manifest()
    ratings = []
    for entry in manifest["entries"]:
        if entry["requested_label"] == 9 and entry["cell"] in ("AA", "BB") and entry["ed_prediction"] in (4, 8):
            rating = "unreadable" if entry["cell"] == "AA" and entry["selection_index"] % 2 == 0 else "3"
        else:
            rating = "9"
        ratings.append({"blind_id": entry["blind_id"], "rating": rating})
    payload = {"schema": "lightgen_blind_ratings.v1", "task_id": audit.TASK_ID,
               "review_id": "review-fixed", "ratings": ratings}
    validated = audit._validate_ratings(payload, manifest)
    diagnosis = audit._human_diagnosis(manifest, validated)
    assert sum(diagnosis["primary_20_AA_BB_ED_errors"]["counts"].values()) == 20
    assert diagnosis["primary_20_AA_BB_ED_errors"]["rule_satisfied"] is False
    assert diagnosis["primary_20_AA_BB_ED_errors"]["counts"]["unreadable"] > 0


@pytest.mark.parametrize("rating,expected", [("3", "readable_other"), ("9", "readable9"),
                                               ("unreadable", "unreadable")])
def test_primary20_majority_rule_routes_only_a_supported_human_outcome(rating: str, expected: str) -> None:
    manifest = _valid_manifest()
    ratings = []
    for entry in manifest["entries"]:
        primary = (entry["requested_label"] == 9 and entry["cell"] in ("AA", "BB")
                   and entry["ed_prediction"] in (4, 8))
        ratings.append({"blind_id": entry["blind_id"], "rating": rating if primary else "0"})
    payload = {"schema": "lightgen_blind_ratings.v1", "task_id": audit.TASK_ID,
               "review_id": manifest["review_id"], "ratings": ratings}
    diagnosis = audit._human_diagnosis(manifest, audit._validate_ratings(payload, manifest))
    primary_result = diagnosis["primary_20_AA_BB_ED_errors"]
    assert primary_result["rule_satisfied"] is True
    assert primary_result["decision_category"] == expected


def test_no_human_labels_cannot_complete_the_identity_diagnosis() -> None:
    conditions = _conditions()
    manifest = _valid_manifest()
    inputs = audit.Inputs(
        attribution_dir=Path("attribution"), ab_evidence_dir=Path("ab"),
        attribution_report={}, ab_report={}, source_cohorts={}, checkpoints={
            "A": {"source_run_id": "source"}, "B": {"source_run_id": "source"},
        }, coefficient_rows={"A": np.ones((1000, 100), dtype=np.complex64),
                             "B": np.ones((1000, 100), dtype=np.complex64)},
        blocks={}, conditions=conditions, cohort_images={}, proposal_ids=np.arange(1000),
        camera_gain=1.0, preflight={"native_readout": {"status": "PASSED"}},
    )
    report = audit._build_report(inputs, manifest, 0.0, ratings=None)
    assert report["status"] == "AWAITING_HUMAN_RATINGS"
    assert report["decision"]["code"] == "PENDING_HUMAN_RATINGS"
    assert "human_diagnosis" not in report


def test_bootstrap_is_fixed_seeded_and_proposal_paired() -> None:
    conditions = _conditions()
    result = audit._bootstrap(conditions)
    again = audit._bootstrap(conditions)
    assert result == again
    assert result["resamples"] == 2_000
    assert result["seed"] == 2026102102
    assert result["contrasts"]["decoder_B_minus_A_fixed_zA"]["recall9"]["estimate"] == 0.0
    assert result["contrasts"]["AA_gain_1p15_minus_1p00"]["recall9"]["estimate"] == 0.0


@pytest.mark.parametrize("corruption", ["wrong_review", "duplicate", "missing", "invalid_label", "metadata_leak"])
def test_ratings_consumer_rejects_wrong_incomplete_or_unblind_input(corruption: str) -> None:
    manifest = _valid_manifest()
    ratings = [{"blind_id": entry["blind_id"], "rating": "9"} for entry in manifest["entries"]]
    payload: dict[str, object] = {"schema": "lightgen_blind_ratings.v1", "task_id": audit.TASK_ID,
                                  "review_id": manifest["review_id"], "ratings": ratings}
    if corruption == "wrong_review":
        payload["review_id"] = "wrong"
    elif corruption == "duplicate":
        payload["ratings"][1]["blind_id"] = payload["ratings"][0]["blind_id"]
    elif corruption == "missing":
        payload["ratings"] = payload["ratings"][:-1]
    elif corruption == "invalid_label":
        payload["ratings"][0]["rating"] = "unknown"
    elif corruption == "metadata_leak":
        payload["ratings"][0]["expected_label"] = 9
    with pytest.raises(audit.EvidenceError):
        audit._validate_ratings(payload, manifest)


def test_resume_consumer_rejects_incomplete_saved_task(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text('{"task_id":"LIGHTGEN_SMALL_CLASS9_SAVED_DEFECT_DIAGNOSIS_V1"}', encoding="utf-8")
    with pytest.raises(audit.EvidenceError, match="incomplete"):
        audit._expected_output_files(tmp_path)
