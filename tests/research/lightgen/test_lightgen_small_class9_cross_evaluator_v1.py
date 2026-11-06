from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.confirmation_evaluators import ThirdConfirmationEvaluator
from scripts.research.lightgen import audit_lightgen_small_class9_cross_evaluator_v1 as audit
from scripts.research.lightgen import audit_lightgen_small_class9_saved_defect_diagnosis_v1 as base


def _review_fixture() -> tuple[dict, dict, dict[str, str], dict[str, str]]:
    proposal_by_label = {4: 0, 8: 0, 9: 0}
    def entries_for(wave: str, cell: str, label: int, predictions: list[int],
                    ratings: list[str], rows: list[dict], labels: dict[str, str]) -> None:
        for prediction, rating in zip(predictions, ratings, strict=True):
            proposal = proposal_by_label[label]
            proposal_by_label[label] += 1
            card = f"{wave}_{cell}_{label}_{proposal:03d}"
            row = {"blind_id": card, "cell": cell, "requested_label": label,
                   "proposal_id": proposal, "ed_prediction": prediction}
            rows.append(row)
            labels[card] = rating

    old_labels: dict[str, str] = {}
    new_labels: dict[str, str] = {}
    old_rows: list[dict] = []
    new_rows: list[dict] = []
    for cell, readable in (("AA", 4), ("BB", 6)):
        ratings = ["9"] * readable + ["unreadable"] * (10 - readable)
        entries_for("old", cell, 9, [4, 8] * 5, ratings, old_rows, old_labels)
    # Twenty cross-cell E_D-error cards.
    for cell in ("BA", "AB"):
        entries_for("old", cell, 9, [4, 8] * 5, ["1"] * 10, old_rows, old_labels)
    # Twelve E_D-predicted-9 controls and eight unfiltered distractors.
    entries_for("old", "AA", 9, [9] * 12, ["unreadable"] * 12, old_rows, old_labels)
    for label in (4, 8):
        entries_for("old", "AA", label, [label] * 4,
                    ["1", "1", "unreadable", "unreadable"], old_rows, old_labels)
    for cell, ratings in (("AA", ["9"] * 7 + ["4"] + ["unreadable"] * 12),
                          ("BB", ["9"] * 8 + ["4"] + ["unreadable"] * 11)):
        entries_for("new", cell, 9, [4, 8] * 10, ratings, new_rows, new_labels)
    entries_for("new", "AA", 9, [9] * 8, ["unreadable"] * 8, new_rows, new_labels)
    for label in (4, 8):
        entries_for("new", "BB", label, [label] * 4,
                    ["unreadable", "unreadable", "unreadable", "unreadable"], new_rows, new_labels)
    parent = {"task_id": base.TASK_ID, "review_id": "old-review", "entries": old_rows}
    child = {"task_id": "LIGHTGEN_SMALL_CLASS9_BOUNDED_REVIEW_EXTENSION_V1",
             "review_id": "new-review", "parent_review_id": "old-review", "entries": new_rows}
    return parent, child, old_labels, new_labels


def _valid_extension_report(parent: dict, child: dict, old_labels: dict[str, str], new_labels: dict[str, str]) -> dict:
    per_cell = {cell: {category: 0 for category in audit.RATING_CATEGORIES} for cell in ("AA", "BB")}
    for manifest, labels in ((parent, old_labels), (child, new_labels)):
        for row in manifest["entries"]:
            if row["cell"] in per_cell and row["requested_label"] == 9 and row["ed_prediction"] in (4, 8):
                per_cell[row["cell"]][audit._rating_category(labels[row["blind_id"]])] += 1
    counts = {category: sum(per_cell[cell][category] for cell in per_cell) for category in audit.RATING_CATEGORIES}
    return {"status": "COMPLETE_DIAGNOSTIC", "measurement_complete": True,
            "decision": {"code": "MIXED_OR_UNRESOLVED"},
            "human_diagnosis": {"counts": counts, "per_cell": per_cell,
                                "wilson_95": {key: audit._wilson(value, 60) for key, value in counts.items()}}}


def test_rating_consumer_recounts_fixed_116_and_rejects_misaligned_report() -> None:
    parent, child, old_labels, new_labels = _review_fixture()
    report = _valid_extension_report(parent, child, old_labels, new_labels)
    result = audit._independently_verify_ratings(parent, old_labels, child, new_labels, report)
    assert result["status"] == "PASSED"
    assert result["primary60"]["counts"] == {"readable9": 25, "readable_other": 2, "unreadable": 33}
    assert result["strata"] == audit.EXPECTED_STRATA
    assert result["combined_human_labels"]["readable_numeric"] == 51
    assert result["combined_human_labels"]["unreadable"] == 65
    report["human_diagnosis"]["counts"]["unreadable"] += 1
    with pytest.raises(base.EvidenceError, match="independent standard-library recount"):
        audit._independently_verify_ratings(parent, old_labels, child, new_labels, report)


def test_rating_consumer_rejects_duplicate_or_missing_cards() -> None:
    parent, child, old_labels, new_labels = _review_fixture()
    report = _valid_extension_report(parent, child, old_labels, new_labels)
    child["entries"][0]["proposal_id"] = child["entries"][1]["proposal_id"]
    with pytest.raises(base.EvidenceError, match="repeat class/proposal"):
        audit._independently_verify_ratings(parent, old_labels, child, new_labels, report)
    _, child, old_labels, new_labels = _review_fixture()
    del new_labels[child["entries"][0]["blind_id"]]
    with pytest.raises(base.EvidenceError, match="rating is missing"):
        audit._independently_verify_ratings(parent, old_labels, child, new_labels, report)


def test_embedded_review_png_is_decoded_and_bound_to_manifest(tmp_path: Path) -> None:
    pixels = np.zeros((28, 28), dtype=np.float32)
    pixels[5:20, 9:15] = 0.6
    uri = base._png_data_uri(pixels)
    manifest = {"entries": [{"blind_id": "card_001"}]}
    path = tmp_path / "blind_review.html"
    path.write_text(base._review_html("review", [{"blind_id": "card_001", "image": uri}]), encoding="utf-8")
    cards = audit._load_review_cards(tmp_path, manifest)
    decoded, _ = audit._decode_review_png(cards["card_001"], pixels, "card_001")
    expected = np.rint(pixels * 255).astype(np.uint8).astype(np.float32) / 255.0
    np.testing.assert_array_equal(decoded, expected)
    changed = pixels.copy()
    changed[6, 10] = 0.25
    with pytest.raises(base.EvidenceError, match="quantization differs"):
        audit._decode_review_png(cards["card_001"], changed, "card_001")
    manifest["entries"][0]["blind_id"] = "wrong_id"
    with pytest.raises(base.EvidenceError, match="do not align"):
        audit._load_review_cards(tmp_path, manifest)


def test_logits_are_independently_scored_and_unreadable_rows_have_no_accuracy() -> None:
    parent, child, old_labels, new_labels = _review_fixture()
    manifests = [("original60", parent, old_labels), ("extension56", child, new_labels)]
    rows = []
    float_logits = []
    png_logits = []
    for wave, manifest, ratings in manifests:
        for entry in manifest["entries"]:
            human = ratings[entry["blind_id"]]
            target = int(human) if human != "unreadable" else entry["proposal_id"] % 10
            ed = target if entry["proposal_id"] % 2 == 0 else (target + 1) % 10
            rows.append({"review_wave": wave, "blind_id": entry["blind_id"],
                         "stratum": audit._stratum(entry), "human_rating": human,
                         "ed_prediction": ed, "ed_confidence": 0.7, "ed_logit_margin": 2.0,
                         "ed_target_logit_margin": 2.0 if human != "unreadable" and ed == int(human) else -2.0})
            logits = np.zeros(10, dtype=np.float32)
            logits[target] = 3.0
            float_logits.append(logits.copy())
            if len(rows) == 1:
                logits[target] = 0.0
                logits[(target + 1) % 10] = 3.0
            png_logits.append(logits.copy())
    metrics = audit._complete_metrics(rows, np.stack(float_logits), np.stack(png_logits))
    assert metrics["readable_numeric_identity_comparison"]["denominator"] == 51
    assert sum(metrics["readable_numeric_identity_comparison"]["agreement_groups"].values()) == 51
    assert metrics["unreadable_classifier_outputs"]["denominator"] == 65
    assert metrics["unreadable_classifier_outputs"]["has_digit_ground_truth"] is False
    assert "accuracy" not in metrics["unreadable_classifier_outputs"]
    assert metrics["E_C_float_vs_review_PNG"]["prediction_flips"] >= 1
    assert metrics["readable_human_nines"]["cards"] > 0


def test_unfiltered_distractor_uses_its_own_saved_logits_not_class9_selection_field() -> None:
    distractor = {"blind_id": "card", "requested_label": 4, "stratum": "requested4_unfiltered",
                  "ed_prediction": 8}
    audit._validate_saved_ed_prediction(distractor, actual_prediction=4)
    selected_nine = {"blind_id": "class9", "requested_label": 9, "stratum": "requested9_ed8",
                     "ed_prediction": 8}
    with pytest.raises(base.EvidenceError, match="saved E_D prediction"):
        audit._validate_saved_ed_prediction(selected_nine, actual_prediction=4)


def _valid_checkpoint(model: ThirdConfirmationEvaluator) -> dict:
    return {
        "schema": audit.EC_CHECKPOINT_SCHEMA,
        "architecture_kind": ThirdConfirmationEvaluator.architecture_kind,
        "architecture": copy.deepcopy(audit.EC_ARCHITECTURE),
        "preprocessing": copy.deepcopy(audit.EC_PREPROCESSING),
        "class_order": list(range(10)),
        "prediction_rule": "argmax(logits,dim=1)",
        "qualification": {"schema": "third_confirmation_evaluator_qualification.v1",
                          "policy": {"min_accuracy": 0.985, "min_macro_recall": 0.985, "min_class_recall": 0.95},
                          "split_role": "E_C_QUALIFICATION", "evaluation_calls": 1, "count": 5500,
                          "accuracy": 0.9898, "macro_recall": 0.9898,
                          "per_class_recall": [0.98] * 10, "passed": True},
        "independence": {"role": "INDEPENDENT_CONFIRMATION_ONLY", "used_in_generator_loss": False,
                         "used_in_gradient_attribution": False, "used_in_hyperparameter_selection": False},
        "state_dict": model.state_dict(),
    }


def test_cnn_loader_is_strict_frozen_and_uses_logits(tmp_path: Path) -> None:
    original = ThirdConfirmationEvaluator()
    path = tmp_path / "ec.pt"
    torch.save(_valid_checkpoint(original), path)
    model, metadata = audit._load_ec(path)
    before = audit._snapshot(model)
    logits, calls = audit._forward_ec(model, np.zeros((2, 1, 28, 28), dtype=np.float32))
    assert metadata["qualification"]["passed"] is True
    assert logits.shape == (2, 10) and calls == 1
    assert not model.training and all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    assert audit._snapshots_equal(before, audit._snapshot(model))
    payload = _valid_checkpoint(original)
    payload["class_order"] = list(reversed(range(10)))
    torch.save(payload, path)
    with pytest.raises(base.EvidenceError, match="class order"):
        audit._load_ec(path)
    payload = _valid_checkpoint(original)
    payload["state_dict"].pop(next(iter(payload["state_dict"])))
    torch.save(payload, path)
    with pytest.raises(base.EvidenceError, match="strictly match"):
        audit._load_ec(path)


def test_incomplete_resume_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(base.EvidenceError, match="incomplete audit output"):
        audit._validate_output(tmp_path, SimpleNamespace(attribution_dir=tmp_path), {})


def test_saved_logits_csv_and_report_share_one_complete_consumer(tmp_path: Path) -> None:
    parent, child, old_labels, new_labels = _review_fixture()
    rating_report = _valid_extension_report(parent, child, old_labels, new_labels)
    rating_verification = audit._independently_verify_ratings(parent, old_labels, child, new_labels, rating_report)
    rows = []
    ed_logits = []
    ec_float_logits = []
    ec_png_logits = []
    for wave, manifest, ratings in (("original60", parent, old_labels), ("extension56", child, new_labels)):
        for entry in manifest["entries"]:
            rating = ratings[entry["blind_id"]]
            ed_prediction = entry["ed_prediction"] if entry["requested_label"] == 9 else entry["requested_label"]
            ed = np.zeros(10, dtype=np.float32)
            ed[ed_prediction] = 2.0
            ed_logits.append(ed)
            ec_prediction = int(rating) if rating != "unreadable" else entry["proposal_id"] % 10
            ec_png = np.zeros(10, dtype=np.float32)
            ec_png[ec_prediction] = 2.0
            ec_png_logits.append(ec_png.copy())
            ec_float = ec_png.copy()
            if not ec_float_logits:
                ec_float[ec_prediction] = 0.0
                ec_float[(ec_prediction + 1) % 10] = 2.0
            ec_float_logits.append(ec_float)
            ed_pred, ed_conf, ed_margin, ed_target_margin = audit._logit_metrics(ed, rating)
            rows.append({"review_wave": wave, "blind_id": entry["blind_id"], "stratum": audit._stratum(entry),
                         "cell": entry["cell"], "condition_id": f"{entry['cell']}_g1p00",
                         "requested_label": entry["requested_label"], "proposal_id": entry["proposal_id"],
                         "human_rating": rating, "human_numeric_readable": rating != "unreadable",
                         "ed_prediction": ed_pred, "ed_confidence": ed_conf, "ed_logit_margin": ed_margin,
                         "ed_target_logit_margin": ed_target_margin,
                         "ed_matches_human": None if rating == "unreadable" else ed_pred == int(rating)})
    ec_float_logits = np.stack(ec_float_logits)
    ec_png_logits = np.stack(ec_png_logits)
    results = audit._complete_metrics(rows, ec_float_logits, ec_png_logits)
    output = tmp_path / "complete"
    output.mkdir()
    (output / "per_image.csv").write_bytes(audit._csv_bytes(rows))
    bundle = io.BytesIO()
    np.savez_compressed(bundle, sample_ids=np.asarray([f"{r['review_wave']}:{r['blind_id']}" for r in rows]),
                        ed_saved_logits=np.stack(ed_logits), ec_float_logits=ec_float_logits,
                        ec_review_png_logits=ec_png_logits)
    (output / "evaluator_logits.npz").write_bytes(bundle.getvalue())
    report = {"task_id": audit.TASK_ID, "status": "COMPLETE_DIAGNOSTIC", "measurement_complete": True,
              "inputs": {"attribution_dir": "C:\\fixed\\attribution", "ab_evidence_dir": "C:\\fixed\\ab",
                         "original_review_dir": "C:\\fixed\\old", "extension_review_dir": "C:\\fixed\\new",
                         "original_review_id": parent["review_id"], "extension_review_id": child["review_id"]},
              "rating_verification": rating_verification, "results": results}
    (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (output / "validation_receipt.json").write_text(json.dumps({"status": "PASSED", "formal_ec_samples": 232,
                                                                  "new_ed_forwards": 0,
                                                                  "parameter_buffer_unchanged": True}), encoding="utf-8")
    for name in audit.OUTPUT_FILES:
        path = output / name
        if not path.exists():
            path.write_text("fixture", encoding="utf-8")
    inputs = SimpleNamespace(attribution_dir=Path("C:\\fixed\\attribution"), ab_evidence_dir=Path("C:\\fixed\\ab"))
    verified = audit._validate_output(output, inputs, rating_verification,
                                      original_review_dir=Path("C:\\fixed\\old"),
                                      extension_review_dir=Path("C:\\fixed\\new"),
                                      original_review_id=parent["review_id"], extension_review_id=child["review_id"])
    assert verified["results"]["readable_numeric_identity_comparison"]["denominator"] == 51
    with (output / "per_image.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    saved[0]["ec_png_prediction"] = str((int(saved[0]["ec_png_prediction"]) + 1) % 10)
    with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=saved[0].keys())
        writer.writeheader()
        writer.writerows(saved)
    with pytest.raises(base.EvidenceError, match="CSV ec_png_prediction"):
        audit._validate_output(output, inputs, rating_verification,
                               original_review_dir=Path("C:\\fixed\\old"),
                               extension_review_dir=Path("C:\\fixed\\new"),
                               original_review_id=parent["review_id"], extension_review_id=child["review_id"])
