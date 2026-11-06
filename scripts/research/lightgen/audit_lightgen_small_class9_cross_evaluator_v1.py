"""Compare saved class-9 human reviews with a frozen canonical CNN."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    EC_CHECKPOINT_SCHEMA,
    ThirdConfirmationEvaluator,
)
from scripts.research.lightgen import audit_lightgen_small_class9_bounded_review_extension_v1 as extension
from scripts.research.lightgen import audit_lightgen_small_class9_saved_defect_diagnosis_v1 as base


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_CLASS9_CROSS_EVALUATOR_AUDIT_V1"
RATING_CATEGORIES = ("readable9", "readable_other", "unreadable")
EXPECTED_STRATA = {"primary60": 60, "cross_cell20": 20, "ed_prediction9_controls20": 20, "distractors16": 16}
PNG_UPSCALE = 12
EC_ARCHITECTURE = {
    "schema": "third_confirmation_evaluator_architecture.v1",
    "architecture_kind": "third_confirmation_canonical_cnn_v1",
    "input": {"shape": [1, 28, 28], "range": "[0,1]"},
    "layers": [
        {"type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel": 5, "padding": 2},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel": 2},
        {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel": 5, "padding": 2},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel": 2},
        {"type": "Flatten"},
        {"type": "Linear", "in_features": 3136, "out_features": 128},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 128, "out_features": 10},
    ],
    "forbidden": ["dropout", "batch_norm", "residual_blocks", "pretrained_weights"],
    "class_order": list(range(10)),
    "prediction_rule": "argmax(logits,dim=1)",
}
EC_PREPROCESSING = {
    "schema": "third_confirmation_evaluator_preprocessing.v1",
    "input_domain": "canonical MNIST train images",
    "value_contract": "float32 values in [0,1] from uint8/255",
    "shape": [1, 28, 28],
    "normalization": "none beyond canonical uint8/255 conversion",
    "augmentation": "none",
    "synthetic_cmos": False,
    "optical_model": False,
}
OUTPUT_FILES = (
    "rating_independent_verification.json",
    "per_image.csv",
    "evaluator_logits.npz",
    "report.json",
    "report.html",
    "fixed_cases.html",
    "validation_receipt.json",
    "closeout_report.md",
)
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    base._require(condition, message)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    safe = base._json_safe(value)
    payload = json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def _rating_category(rating: str) -> str:
    return "unreadable" if rating == "unreadable" else "readable9" if rating == "9" else "readable_other"


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / total + z2 / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _independently_verify_ratings(
    parent_manifest: dict[str, Any], parent_ratings: dict[str, str],
    child_manifest: dict[str, Any], child_ratings: dict[str, str], extension_report: dict[str, Any],
) -> dict[str, Any]:
    all_entries = [("original60", row, parent_ratings) for row in parent_manifest["entries"]]
    all_entries.extend(("extension56", row, child_ratings) for row in child_manifest["entries"])
    _require(len(all_entries) == 116, "two review waves must contain exactly 116 cards")
    identities: set[tuple[int, int]] = set()
    blind_ids: set[tuple[str, str]] = set()
    strata = {name: 0 for name in EXPECTED_STRATA}
    primary_by_cell = {"AA": {key: 0 for key in RATING_CATEGORIES},
                       "BB": {key: 0 for key in RATING_CATEGORIES}}
    primary_by_wave = {"original20": {key: 0 for key in RATING_CATEGORIES},
                       "extension40": {key: 0 for key in RATING_CATEGORIES}}
    readable_labels: dict[str, int] = {}
    readable = unreadable = 0
    for wave, entry, ratings in all_entries:
        blind_id = entry.get("blind_id")
        _require(isinstance(blind_id, str) and (wave, blind_id) not in blind_ids,
                 "review card ID is missing or duplicated")
        blind_ids.add((wave, blind_id))
        label, proposal = entry.get("requested_label"), entry.get("proposal_id")
        _require(isinstance(label, int) and label in (4, 8, 9)
                 and isinstance(proposal, int) and proposal >= 0,
                 "review manifest has invalid requested label or proposal ID")
        identity = (label, proposal)
        _require(identity not in identities, f"the two review waves repeat class/proposal identity {identity}")
        identities.add(identity)
        _require(blind_id in ratings, f"review rating is missing for {wave}/{blind_id}")
        category = _rating_category(ratings[blind_id])
        is_primary = label == 9 and entry.get("cell") in ("AA", "BB") and entry.get("ed_prediction") in (4, 8)
        is_cross = label == 9 and entry.get("cell") in ("BA", "AB") and entry.get("ed_prediction") in (4, 8)
        is_ed9 = label == 9 and entry.get("ed_prediction") == 9
        if is_primary:
            strata["primary60"] += 1
            primary_by_cell[entry["cell"]][category] += 1
            wave_name = "original20" if wave == "original60" else "extension40"
            primary_by_wave[wave_name][category] += 1
        elif is_cross:
            strata["cross_cell20"] += 1
        elif is_ed9:
            strata["ed_prediction9_controls20"] += 1
        elif label in (4, 8):
            strata["distractors16"] += 1
        else:
            raise base.EvidenceError(f"manifest row has no declared stratum: {wave}/{blind_id}")
        if ratings[blind_id] == "unreadable":
            unreadable += 1
        else:
            readable += 1
            readable_labels[ratings[blind_id]] = readable_labels.get(ratings[blind_id], 0) + 1
    _require(strata == EXPECTED_STRATA, f"review stratum counts differ: {strata}")
    _require(readable == 51 and unreadable == 65, "combined human readability counts must be 51/65")
    _require(sum(primary_by_wave["original20"].values()) == 20
             and sum(primary_by_wave["extension40"].values()) == 40,
             "fixed primary review waves are incomplete")
    _require(all(sum(primary_by_cell[cell].values()) == 30 for cell in ("AA", "BB")),
             "primary AA/BB cell populations must each contain 30 cards")
    pooled = {category: sum(primary_by_cell[cell][category] for cell in primary_by_cell)
              for category in RATING_CATEGORIES}
    intervals = {category: _wilson(count, 60) for category, count in pooled.items()}
    _require(extension_report.get("status") == "COMPLETE_DIAGNOSTIC"
             and extension_report.get("measurement_complete") is True
             and extension_report.get("decision", {}).get("code") == "MIXED_OR_UNRESOLVED",
             "extension report is not formally finalized as mixed")
    saved_diagnosis = extension_report.get("human_diagnosis", {})
    _require(saved_diagnosis.get("counts") == pooled
             and saved_diagnosis.get("per_cell") == primary_by_cell
             and saved_diagnosis.get("wilson_95") == intervals,
             "extension report differs from independent standard-library recount")
    return {
        "status": "PASSED",
        "source_ratings_preserved_in_review_directories": True,
        "original_cards": 60,
        "extension_cards": 56,
        "unique_class_proposal_pairs": len(identities),
        "combined_human_labels": {"readable_numeric": readable, "unreadable": unreadable,
                                  "readable_numeric_by_digit": {str(k): readable_labels[k] for k in sorted(readable_labels)}},
        "strata": strata,
        "primary60": {"counts": pooled, "wilson_95": intervals,
                      "per_cell": primary_by_cell, "per_wave": primary_by_wave,
                      "decision": "MIXED_OR_UNRESOLVED"},
        "extension_report_matches_independent_recount": True,
        "method": "independent Python counters and Wilson calculation; original/extension validators separately checked task, review ID, card completeness, duplicates and label vocabulary",
    }


def _load_review_cards(review_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    path = review_dir / "blind_review.html"
    _require(path.is_file(), f"saved blind review page is missing: {path}")
    text = path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", text, flags=re.DOTALL)
    _require(len(scripts) == 1, "blind review page must contain one embedded card payload")
    script = scripts[0]
    start_marker = "cards="
    end_marker = ",labels="
    _require(start_marker in script and end_marker in script, "blind review card payload is not in the expected format")
    payload_text = script.split(start_marker, 1)[1].split(end_marker, 1)[0]
    cards = json.loads(payload_text)
    _require(isinstance(cards, list) and len(cards) == len(manifest["entries"]),
             "blind review embedded image population is incomplete")
    result: dict[str, str] = {}
    for card in cards:
        _require(isinstance(card, dict) and set(card) == {"blind_id", "image"}
                 and isinstance(card["blind_id"], str) and card["blind_id"] not in result,
                 "blind review embedded card IDs are invalid or duplicated")
        uri = card["image"]
        _require(isinstance(uri, str) and uri.startswith("data:image/png;base64,"),
                 "blind review image is not the saved PNG data URI")
        result[card["blind_id"]] = uri
    expected_ids = {entry["blind_id"] for entry in manifest["entries"]}
    _require(set(result) == expected_ids, "blind review PNG IDs do not align with its manifest")
    return result


def _decode_review_png(uri: str, source_view: np.ndarray, context: str) -> tuple[np.ndarray, np.ndarray]:
    encoded = uri.partition(",")[2]
    try:
        png_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(png_bytes)) as image:
            _require(image.mode == "L" and image.size == (28 * PNG_UPSCALE, 28 * PNG_UPSCALE),
                     f"review PNG geometry or mode differs at {context}")
            pixels = np.asarray(image, dtype=np.uint8).copy()
    except (ValueError, OSError) as error:
        raise base.EvidenceError(f"review PNG cannot be decoded at {context}") from error
    collapsed = pixels[::PNG_UPSCALE, ::PNG_UPSCALE]
    _require(collapsed.shape == (28, 28)
             and np.array_equal(pixels, np.repeat(np.repeat(collapsed, PNG_UPSCALE, axis=0), PNG_UPSCALE, axis=1)),
             f"review PNG is not the exact nearest-neighbor display of a 28x28 view at {context}")
    view = np.asarray(source_view, dtype=np.float32).reshape(28, 28)
    _require(np.isfinite(view).all() and float(view.min()) >= -1e-7 and float(view.max()) <= 1.0 + 1e-7,
             f"saved floating view is nonfinite or outside [0,1] at {context}")
    expected = np.rint(np.clip(view, 0.0, 1.0) * 255.0).astype(np.uint8)
    _require(np.array_equal(collapsed, expected), f"review PNG quantization differs from saved view at {context}")
    return collapsed.astype(np.float32) / 255.0, pixels


def _stratum(entry: dict[str, Any]) -> str:
    label, cell, prediction = entry["requested_label"], entry["cell"], entry["ed_prediction"]
    if label in (4, 8):
        return "distractors16"
    if prediction == 9:
        return "ed_prediction9_controls20"
    return "primary60" if cell in ("AA", "BB") else "cross_cell20"


def _validate_saved_ed_prediction(entry: dict[str, Any], actual_prediction: int) -> None:
    if int(entry["requested_label"]) == 9:
        _require(actual_prediction == int(entry["ed_prediction"]),
                 f"saved E_D prediction differs from its logits at {entry['blind_id']}")
    else:
        _require(int(entry["requested_label"]) in (4, 8)
                 and entry.get("stratum") in ("requested4_unfiltered", "requested8_unfiltered"),
                 f"non-class-9 review sample is not an unfiltered distractor at {entry['blind_id']}")


def _collect_rows(
    inputs: base.Inputs, parent_manifest: dict[str, Any], parent_ratings: dict[str, str], parent_pngs: dict[str, str],
    child_manifest: dict[str, Any], child_ratings: dict[str, str], child_pngs: dict[str, str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    float_images: list[np.ndarray] = []
    png_images: list[np.ndarray] = []
    ed_logits: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for wave, manifest, ratings, pngs in (
        ("original60", parent_manifest, parent_ratings, parent_pngs),
        ("extension56", child_manifest, child_ratings, child_pngs),
    ):
        for entry in manifest["entries"]:
            cell, label = entry["cell"], int(entry["requested_label"])
            proposal, index = int(entry["proposal_id"]), int(entry["cohort_index"])
            condition_id = entry.get("condition_id")
            context = f"{wave}/{entry['blind_id']}"
            _require(cell in base.CELLS and condition_id == f"{cell}_g1p00",
                     f"review row points to an unexpected production condition at {context}")
            identity = (label, proposal)
            _require(identity not in seen, f"review row repeats a class/proposal identity at {context}")
            seen.add(identity)
            block = inputs.blocks.get((cell, label))
            _require(block is not None and index < len(block["proposal_ids"])
                     and int(block["proposal_ids"][index]) == proposal
                     and int(block["requested_labels"][index]) == label,
                     f"review sample row does not match its saved attribution block at {context}")
            condition = block["conditions"].get(condition_id)
            _require(isinstance(condition, dict), f"saved production condition is missing at {context}")
            image = condition["images"][index].detach().cpu().numpy()
            logits = condition["logits"][index].detach().cpu().numpy().astype(np.float64, copy=True)
            _require(tuple(image.shape) == (1, 28, 28) and logits.shape == (10,)
                     and np.isfinite(logits).all(), f"saved view/logits have invalid shape or values at {context}")
            ed_prediction, ed_confidence, ed_margin, ed_target_margin = _logit_metrics(logits, ratings[entry["blind_id"]])
            _validate_saved_ed_prediction(entry, ed_prediction)
            png_image, _ = _decode_review_png(pngs[entry["blind_id"]], image, context)
            rating = ratings[entry["blind_id"]]
            rows.append({
                "review_wave": wave,
                "blind_id": entry["blind_id"],
                "stratum": _stratum(entry),
                "cell": cell,
                "condition_id": condition_id,
                "requested_label": label,
                "proposal_id": proposal,
                "human_rating": rating,
                "human_numeric_readable": rating != "unreadable",
                "ed_prediction": ed_prediction,
                "ed_confidence": ed_confidence,
                "ed_logit_margin": ed_margin,
                "ed_target_logit_margin": ed_target_margin,
                "ed_matches_human": None if rating == "unreadable" else ed_prediction == int(rating),
            })
            float_images.append(np.asarray(image, dtype=np.float32))
            png_images.append(png_image[None, ...])
            ed_logits.append(logits.astype(np.float32))
    _require(len(rows) == 116 and len(seen) == 116, "joined review sample population is incomplete")
    return rows, np.stack(float_images), np.stack(png_images), np.stack(ed_logits)


def _logit_metrics(logits: np.ndarray, rating: str) -> tuple[int, float, float, float | None]:
    values = np.asarray(logits, dtype=np.float64)
    _require(values.shape == (10,) and np.isfinite(values).all(), "classifier logits must be ten finite values")
    prediction = int(np.argmax(values))
    ordered = np.sort(values)
    margin = float(ordered[-1] - ordered[-2])
    exponentials = np.exp(values - np.max(values))
    probabilities = exponentials / np.sum(exponentials)
    target_margin = None
    if rating != "unreadable":
        target = int(rating)
        competitors = values.copy()
        competitors[target] = -np.inf
        target_margin = float(values[target] - np.max(competitors))
    return prediction, float(np.max(probabilities)), margin, target_margin


def _load_ec(checkpoint_path: Path) -> tuple[ThirdConfirmationEvaluator, dict[str, Any]]:
    _require(checkpoint_path.is_file(), f"E_C checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and payload.get("schema") == EC_CHECKPOINT_SCHEMA
             and payload.get("architecture_kind") == ThirdConfirmationEvaluator.architecture_kind,
             "E_C schema or architecture identity differs")
    _require(payload.get("architecture") == EC_ARCHITECTURE
             and payload.get("preprocessing") == EC_PREPROCESSING
             and payload.get("class_order") == list(range(10))
             and payload.get("prediction_rule") == "argmax(logits,dim=1)",
             "E_C architecture, class order, prediction rule or preprocessing differs")
    qualification = payload.get("qualification", {})
    policy = qualification.get("policy", {})
    recalls = qualification.get("per_class_recall")
    _require(qualification.get("schema") == "third_confirmation_evaluator_qualification.v1"
             and qualification.get("split_role") == "E_C_QUALIFICATION"
             and qualification.get("evaluation_calls") == 1 and qualification.get("count") == 5500
             and qualification.get("passed") is True and isinstance(recalls, list) and len(recalls) == 10
             and isinstance(policy.get("min_accuracy"), (float, int))
             and isinstance(policy.get("min_macro_recall"), (float, int))
             and isinstance(policy.get("min_class_recall"), (float, int))
             and float(qualification.get("accuracy", 0.0)) >= float(policy["min_accuracy"])
             and float(qualification.get("macro_recall", 0.0)) >= float(policy["min_macro_recall"])
             and min(map(float, recalls)) >= float(policy["min_class_recall"]),
             "E_C historical canonical-MNIST qualification record is incomplete or failed")
    independence = payload.get("independence", {})
    _require(independence.get("role") == "INDEPENDENT_CONFIRMATION_ONLY"
             and independence.get("used_in_generator_loss") is False
             and independence.get("used_in_gradient_attribution") is False
             and independence.get("used_in_hyperparameter_selection") is False,
             "E_C checkpoint is not an independent frozen confirmation evaluator")
    evaluator = ThirdConfirmationEvaluator()
    try:
        evaluator.load_state_dict(payload["state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise base.EvidenceError("E_C state_dict does not strictly match the canonical CNN") from error
    evaluator.freeze()
    _require(not evaluator.training and all(not parameter.requires_grad for parameter in evaluator.parameters()),
             "E_C must be frozen in eval mode")
    metadata = {"schema": payload["schema"], "architecture_kind": payload["architecture_kind"],
                "qualification": {"split_role": qualification["split_role"], "count": qualification["count"],
                                  "accuracy": qualification["accuracy"], "macro_recall": qualification["macro_recall"],
                                  "minimum_class_recall": min(map(float, recalls)), "passed": qualification["passed"]},
                "preprocessing": payload["preprocessing"], "class_order": payload["class_order"],
                "historical_scope": "canonical MNIST qualification only; not human readability or CMOS qualification"}
    return evaluator, metadata


def _snapshot(model: torch.nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    values = [(f"parameter:{name}", tensor.detach().cpu().clone()) for name, tensor in model.named_parameters()]
    values.extend((f"buffer:{name}", tensor.detach().cpu().clone()) for name, tensor in model.named_buffers())
    return tuple(values)


def _snapshots_equal(left: tuple[tuple[str, torch.Tensor], ...],
                     right: tuple[tuple[str, torch.Tensor], ...]) -> bool:
    return len(left) == len(right) and all(name_a == name_b and torch.equal(value_a, value_b)
                                            for (name_a, value_a), (name_b, value_b) in zip(left, right))


def _forward_ec(model: ThirdConfirmationEvaluator, images: np.ndarray, batch_size: int = 32) -> tuple[np.ndarray, int]:
    values = np.asarray(images, dtype=np.float32)
    _require(values.ndim == 4 and tuple(values.shape[1:]) == (1, 28, 28)
             and np.isfinite(values).all() and float(values.min()) >= 0.0 and float(values.max()) <= 1.0,
             "E_C inputs must be finite float32 [N,1,28,28] values in [0,1]")
    results = []
    calls = 0
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(values[start:start + batch_size]))
            output = model(batch)
            _require(isinstance(output, tuple) and len(output) == 2,
                     "E_C forward must return features and logits")
            features, logits = output
            _require(tuple(features.shape) == (len(batch), 128) and tuple(logits.shape) == (len(batch), 10)
                     and torch.isfinite(features).all().item() and torch.isfinite(logits).all().item(),
                     "E_C returned invalid feature or logits tensors")
            results.append(logits.detach().cpu().numpy().astype(np.float32, copy=True))
            calls += 1
    return np.concatenate(results, axis=0), calls


def _complete_metrics(rows: list[dict[str, Any]], ec_float_logits: np.ndarray,
                      ec_png_logits: np.ndarray) -> dict[str, Any]:
    _require(len(rows) == ec_float_logits.shape[0] == ec_png_logits.shape[0] == 116,
             "report consumer requires all 116 rows and both E_C views")
    groups = {name: {"cards": 0, "readable": 0, "unreadable": 0, "both_agree": 0,
                     "ec_only_agrees": 0, "ed_only_agrees": 0, "neither_agrees": 0,
                     "png_float_prediction_flips": 0, "ed_prediction9": 0,
                     "ec_png_prediction9": 0, "ec_png_confidence_sum": 0.0,
                     "ec_png_margin_sum": 0.0, "ed_prediction_counts": {str(i): 0 for i in range(10)},
                     "ec_float_prediction_counts": {str(i): 0 for i in range(10)},
                     "ec_png_prediction_counts": {str(i): 0 for i in range(10)}}
              for name in EXPECTED_STRATA}
    all_numeric = {name: 0 for name in ("both_agree", "ec_only_agrees", "ed_only_agrees", "neither_agrees")}
    by_human_digit: dict[str, dict[str, int]] = {}
    readable_nine = {"cards": 0, "ed_misses": 0, "ec_png_recovers_ed_miss": 0,
                     "both_miss_human_nine": 0}
    unreadable_predictions = {name: {str(i): 0 for i in range(10)}
                             for name in ("ed", "ec_float", "ec_png")}
    unreadable_scores = {name: {"confidence_sum": 0.0, "margin_sum": 0.0}
                         for name in ("ed", "ec_float", "ec_png")}
    for index, row in enumerate(rows):
        ed_prediction = int(row["ed_prediction"])
        ed_confidence = float(row["ed_confidence"])
        ed_margin = float(row["ed_logit_margin"])
        float_prediction, float_confidence, float_margin, float_target_margin = _logit_metrics(
            ec_float_logits[index], row["human_rating"])
        png_prediction, png_confidence, png_margin, png_target_margin = _logit_metrics(
            ec_png_logits[index], row["human_rating"])
        row.update({
            "ec_float_prediction": float_prediction, "ec_float_confidence": float_confidence,
            "ec_float_logit_margin": float_margin, "ec_float_target_logit_margin": float_target_margin,
            "ec_png_prediction": png_prediction, "ec_png_confidence": png_confidence,
            "ec_png_logit_margin": png_margin, "ec_png_target_logit_margin": png_target_margin,
            "ec_float_matches_human": None if row["human_rating"] == "unreadable" else float_prediction == int(row["human_rating"]),
            "ec_png_matches_human": None if row["human_rating"] == "unreadable" else png_prediction == int(row["human_rating"]),
            "ec_float_png_prediction_flip": float_prediction != png_prediction,
        })
        group = groups[row["stratum"]]
        group["cards"] += 1
        group["ed_prediction_counts"][str(ed_prediction)] += 1
        group["ec_float_prediction_counts"][str(float_prediction)] += 1
        group["ec_png_prediction_counts"][str(png_prediction)] += 1
        group["ed_prediction9"] += int(ed_prediction == 9)
        group["ec_png_prediction9"] += int(png_prediction == 9)
        group["png_float_prediction_flips"] += int(float_prediction != png_prediction)
        group["ec_png_confidence_sum"] += png_confidence
        group["ec_png_margin_sum"] += png_margin
        if row["human_rating"] == "unreadable":
            group["unreadable"] += 1
            unreadable_predictions["ed"][str(ed_prediction)] += 1
            unreadable_predictions["ec_float"][str(float_prediction)] += 1
            unreadable_predictions["ec_png"][str(png_prediction)] += 1
            for name, confidence, margin in (("ed", ed_confidence, ed_margin),
                                             ("ec_float", float_confidence, float_margin),
                                             ("ec_png", png_confidence, png_margin)):
                unreadable_scores[name]["confidence_sum"] += confidence
                unreadable_scores[name]["margin_sum"] += margin
            continue
        group["readable"] += 1
        ed_agrees = ed_prediction == int(row["human_rating"])
        ec_agrees = png_prediction == int(row["human_rating"])
        if ed_agrees and ec_agrees:
            label = "both_agree"
        elif ec_agrees:
            label = "ec_only_agrees"
        elif ed_agrees:
            label = "ed_only_agrees"
        else:
            label = "neither_agrees"
        group[label] += 1
        all_numeric[label] += 1
        digit = str(row["human_rating"])
        digit_row = by_human_digit.setdefault(digit, {"cards": 0, "ed_agrees": 0, "ec_png_agrees": 0,
                                                      "both_agree": 0, "ec_only_agrees": 0,
                                                      "ed_only_agrees": 0, "neither_agrees": 0})
        digit_row["cards"] += 1
        digit_row["ed_agrees"] += int(ed_agrees)
        digit_row["ec_png_agrees"] += int(ec_agrees)
        digit_row[label] += 1
        if digit == "9":
            readable_nine["cards"] += 1
            readable_nine["ed_misses"] += int(not ed_agrees)
            readable_nine["ec_png_recovers_ed_miss"] += int(not ed_agrees and ec_agrees)
            readable_nine["both_miss_human_nine"] += int(not ed_agrees and not ec_agrees)
    for name, expected in EXPECTED_STRATA.items():
        _require(groups[name]["cards"] == expected, f"report consumer saw incomplete stratum {name}")
    readable_total = sum(group["readable"] for group in groups.values())
    unreadable_total = sum(group["unreadable"] for group in groups.values())
    _require(readable_total == 51 and unreadable_total == 65
             and sum(all_numeric.values()) == 51,
             "unreadable rows entered the identity comparison or human-readable population is incomplete")
    for group in groups.values():
        count = group["cards"]
        group["mean_ec_png_confidence"] = group["ec_png_confidence_sum"] / count
        group["mean_ec_png_logit_margin"] = group["ec_png_margin_sum"] / count
        del group["ec_png_confidence_sum"], group["ec_png_margin_sum"]
    unreadable_score_means = {
        name: {"mean_confidence": values["confidence_sum"] / unreadable_total,
               "mean_top1_minus_top2_logit_margin": values["margin_sum"] / unreadable_total}
        for name, values in unreadable_scores.items()
    }
    return {
        "rows": 116,
        "human_reference": "numeric human rating on the displayed image; a perceptual reference for these selected cards, not a generator ground-truth label",
        "readable_numeric_identity_comparison": {"denominator": 51, "agreement_groups": all_numeric,
                                                   "by_human_digit": by_human_digit,
                                                   "E_D_agreement": sum(v["ed_agrees"] for v in by_human_digit.values()),
                                                   "E_C_PNG_agreement": sum(v["ec_png_agrees"] for v in by_human_digit.values())},
        "readable_human_nines": readable_nine,
        "unreadable_classifier_outputs": {"denominator": 65, "has_digit_ground_truth": False,
                                          "prediction_histograms": unreadable_predictions,
                                          "mean_confidence_and_margin": unreadable_score_means},
        "E_C_float_vs_review_PNG": {"prediction_flips": sum(g["png_float_prediction_flips"] for g in groups.values()),
                                    "denominator": 116,
                                    "float_view_is_the_comparison_to_saved_E_D_logits": True},
        "by_stratum": groups,
        "sample_selection_boundary": "evaluation population is deliberately E_D-stratified; these counts do not rank evaluator accuracy on the generated population",
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = ("review_wave", "blind_id", "stratum", "cell", "condition_id", "requested_label", "proposal_id",
              "human_rating", "human_numeric_readable", "ed_prediction", "ed_confidence", "ed_logit_margin",
              "ed_target_logit_margin", "ed_matches_human", "ec_float_prediction", "ec_float_confidence",
              "ec_float_logit_margin", "ec_float_target_logit_margin", "ec_float_matches_human",
              "ec_png_prediction", "ec_png_confidence", "ec_png_logit_margin", "ec_png_target_logit_margin",
              "ec_png_matches_human", "ec_float_png_prediction_flip")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _html_report(report: dict[str, Any]) -> str:
    metrics = report["results"]
    groups = metrics["readable_numeric_identity_comparison"]["agreement_groups"]
    unreadable = metrics["unreadable_classifier_outputs"]
    sections = []
    for title, key in (("主要60例", "primary60"), ("跨组合20例", "cross_cell20"),
                       ("E_D判9对照20例", "ed_prediction9_controls20"), ("请求4/8混入16例", "distractors16")):
        group = metrics["by_stratum"][key]
        counts = ", ".join(f"{digit}={count}" for digit, count in group["ec_png_prediction_counts"].items() if count)
        sections.append(f"<tr><td>{title}</td><td>{group['cards']}</td><td>{group['readable']}</td><td>{group['unreadable']}</td><td>{group['png_float_prediction_flips']}</td><td>{html.escape(counts)}</td></tr>")
    return """<!doctype html><html lang='zh'><meta charset='utf-8'><title>冻结CNN交叉评价核验</title>
<style>body{max-width:1100px;margin:24px auto;font-family:system-ui,'Microsoft YaHei',sans-serif;line-height:1.6;color:#20242a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:7px;text-align:center}section{margin:22px 0}</style>
<h1>Gaussian小架构：冻结CNN交叉评价核验</h1>
""" + f"<p>状态：{report['status']}；这是固定人工评阅卡片的分类器一致性审计，不是系统质量Gate。</p>" + f"<p>51张人工给出数字：两者一致 {groups['both_agree']}，仅E_C一致 {groups['ec_only_agrees']}，仅E_D一致 {groups['ed_only_agrees']}，两者均不一致 {groups['neither_agrees']}。65张不可辨认图没有数字真值；本页仅展示分类器输出分布，不能当作可辨性准确率。类别9中E_C_PNG纠正E_D漏判 {metrics['readable_human_nines']['ec_png_recovers_ed_miss']}/{metrics['readable_human_nines']['ed_misses']}。</p>" + f"<p>E_C在浮点图与实际评阅PNG之间预测翻转 {metrics['E_C_float_vs_review_PNG']['prediction_flips']}/116。E_C历史98.98%资格来自5,500张标准MNIST，不能视为CMOS可读性资格。</p>" + "<table><thead><tr><th>分层</th><th>卡片</th><th>人工可辨数字</th><th>不可辨认</th><th>量化翻转</th><th>E_C PNG预测</th></tr></thead><tbody>" + "".join(sections) + "</tbody></table>" + "<p>不可辨认样本E_D预测分布：" + html.escape(json.dumps(unreadable['prediction_histograms']['ed'], ensure_ascii=False)) + "；E_C浮点预测：" + html.escape(json.dumps(unreadable['prediction_histograms']['ec_float'], ensure_ascii=False)) + "；E_C评阅PNG预测：" + html.escape(json.dumps(unreadable['prediction_histograms']['ec_png'], ensure_ascii=False)) + "。</p>" + "<p>样本按E_D分层；不得据此排列总体分类器准确率。人眼数字仅是本批卡片的感知参照，不是逐图生成真值。</p><p><a href='fixed_cases.html'>查看116张逐图对照</a></p></html>"


def _fixed_cases_html(rows: list[dict[str, Any]], float_images: np.ndarray,
                      review_png_uris: list[str]) -> str:
    body = []
    for index, row in enumerate(rows):
        float_uri = base._png_data_uri(float_images[index, 0], upscale=6)
        png_uri = review_png_uris[index]
        human = "不可辨认" if row["human_rating"] == "unreadable" else row["human_rating"]
        body.append(
            f"<tr><td>{html.escape(row['review_wave'])}</td><td>{html.escape(row['blind_id'])}</td>"
            f"<td>{html.escape(row['stratum'])}</td><td>{html.escape(str(row['cell']))}</td>"
            f"<td><img src='{float_uri}' alt='保存浮点图'></td><td><img src='{png_uri}' alt='实际评阅PNG'></td>"
            f"<td>{html.escape(human)}</td><td>{row['ed_prediction']}</td>"
            f"<td>{row['ec_float_prediction']}</td><td>{row['ec_png_prediction']}</td>"
            f"<td>{row['ec_png_confidence']:.4f}/{row['ec_png_logit_margin']:.4f}</td></tr>")
    return ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>固定116图逐图对照</title>"
            "<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:20px}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:5px;text-align:center}img{width:168px;height:168px;object-fit:contain;image-rendering:pixelated}</style>"
            "<h1>固定116张：浮点视图、评阅PNG与冻结分类器</h1>"
            "<p>图像未逐张归一化。人眼数字为该卡片人工评阅结果；‘不可辨认’没有数字真值。E_D logits来自原归因保存结果；E_C分别作用于浮点图与实际嵌入PNG还原图。</p>"
            "<table><thead><tr><th>轮次</th><th>盲评ID</th><th>分层</th><th>组合</th><th>保存浮点图</th><th>评阅PNG</th><th>人工</th><th>E_D</th><th>E_C浮点</th><th>E_C PNG</th><th>E_C PNG置信/间隔</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></html>")


def _load_inputs(args: argparse.Namespace) -> tuple[
    base.Inputs, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
    dict[str, str], dict[str, str], dict[str, Any],
]:
    attribution_dir = _resolve(args.attribution_dir)
    ab_dir = _resolve(args.ab_evidence_dir)
    inputs = base._load_inputs(attribution_dir, ab_dir)
    parent_dir, child_dir = _resolve(args.original_review_dir), _resolve(args.extension_review_dir)
    parent_report = base._read_json(parent_dir / "report.json")
    parent_manifest = base._read_json(parent_dir / "blind_review_manifest.json")
    parent_payload = base._read_json(parent_dir / "human_ratings.json")
    parent_receipt = base._read_json(parent_dir / "human_finalization_verification.json")
    _require(parent_report.get("task_id") == base.TASK_ID and parent_report.get("status") == "COMPLETE_DIAGNOSTIC"
             and parent_report.get("measurement_complete") is True
             and parent_report.get("decision", {}).get("code") == "MIXED_OR_UNRESOLVED"
             and parent_receipt.get("status") == "PASSED",
             "original 60-card review is incomplete or its mixed result changed")
    parent_ratings = base._validate_ratings(parent_payload, parent_manifest)
    child_report = base._read_json(child_dir / "report.json")
    child_manifest = base._read_json(child_dir / "blind_review_manifest.json")
    child_payload = base._read_json(child_dir / "human_ratings.json")
    child_validation = base._read_json(child_dir / "validation_receipt.json")
    _require(child_manifest.get("task_id") == extension.TASK_ID
             and child_manifest.get("parent_review_id") == parent_manifest.get("review_id")
             and child_validation.get("status") == "PASSED"
             and child_validation.get("human_ratings_complete") is True,
             "extension manifest or human-rating completion does not match the original task")
    child_ratings = extension._validate_ratings(child_payload, child_manifest)
    _require(child_report.get("decision", {}).get("code") == "MIXED_OR_UNRESOLVED",
             "extension report decision differs from the preserved mixed evidence")
    rating_verification = _independently_verify_ratings(
        parent_manifest, parent_ratings, child_manifest, child_ratings, child_report)
    return inputs, parent_manifest, child_manifest, parent_report, child_report, parent_ratings, child_ratings, rating_verification


def _float_matches(actual: str, expected: float | None) -> bool:
    if expected is None:
        return actual == ""
    try:
        parsed = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and math.isclose(parsed, expected, rel_tol=1e-7, abs_tol=1e-7)


def _validate_output(output_dir: Path, inputs: base.Inputs, rating_verification: dict[str, Any],
                     *, original_review_dir: Path | None = None, extension_review_dir: Path | None = None,
                     original_review_id: str | None = None, extension_review_id: str | None = None) -> dict[str, Any]:
    missing = [name for name in OUTPUT_FILES if not (output_dir / name).is_file()]
    _require(not missing, f"cannot resume incomplete audit output; missing {missing}")
    report = base._read_json(output_dir / "report.json")
    _require(report.get("task_id") == TASK_ID and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("measurement_complete") is True
             and report.get("inputs", {}).get("attribution_dir") == str(inputs.attribution_dir)
             and report.get("inputs", {}).get("ab_evidence_dir") == str(inputs.ab_evidence_dir)
             and report.get("rating_verification") == rating_verification,
             "saved audit report is incomplete or belongs to different fixed inputs")
    report_inputs = report["inputs"]
    if original_review_dir is not None:
        _require(report_inputs.get("original_review_dir") == str(original_review_dir),
                 "saved report points to a different original review directory")
    if extension_review_dir is not None:
        _require(report_inputs.get("extension_review_dir") == str(extension_review_dir),
                 "saved report points to a different extension review directory")
    if original_review_id is not None:
        _require(report_inputs.get("original_review_id") == original_review_id,
                 "saved report points to a different original review ID")
    if extension_review_id is not None:
        _require(report_inputs.get("extension_review_id") == extension_review_id,
                 "saved report points to a different extension review ID")
    with np.load(output_dir / "evaluator_logits.npz", allow_pickle=False) as bundle:
        required = {"sample_ids", "ed_saved_logits", "ec_float_logits", "ec_review_png_logits"}
        _require(required.issubset(bundle.files), "saved evaluator logits are incomplete")
        _require(bundle["sample_ids"].shape == (116,) and bundle["ed_saved_logits"].shape == (116, 10)
                 and bundle["ec_float_logits"].shape == (116, 10)
                 and bundle["ec_review_png_logits"].shape == (116, 10),
                 "saved evaluator logits have invalid population or shape")
        _require(all(np.isfinite(bundle[name]).all() for name in required - {"sample_ids"}),
                 "saved evaluator logits contain nonfinite values")
        sample_ids = bundle["sample_ids"].astype(str).tolist()
        ed_logits = bundle["ed_saved_logits"].copy()
        ec_float_logits = bundle["ec_float_logits"].copy()
        ec_png_logits = bundle["ec_review_png_logits"].copy()
    with (output_dir / "per_image.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    _require(len(saved_rows) == 116 and len({row["blind_id"] for row in saved_rows}) == 116,
             "saved output CSV is incomplete or repeats review IDs")
    normalized_rows = []
    for index, row in enumerate(saved_rows):
        _require(sample_ids[index] == f"{row['review_wave']}:{row['blind_id']}",
                 "saved logits and CSV sample identities are misaligned")
        rating = row["human_rating"]
        _require(rating == "unreadable" or rating in {str(label) for label in range(10)},
                 "saved CSV contains an invalid human label")
        ed_prediction, ed_confidence, ed_margin, ed_target_margin = _logit_metrics(ed_logits[index], rating)
        _require(int(row["ed_prediction"]) == ed_prediction
                 and _float_matches(row["ed_confidence"], ed_confidence)
                 and _float_matches(row["ed_logit_margin"], ed_margin)
                 and _float_matches(row["ed_target_logit_margin"], ed_target_margin),
                 "saved E_D CSV metrics differ from independent NumPy recomputation")
        normalized_rows.append({
            "review_wave": row["review_wave"], "blind_id": row["blind_id"], "stratum": row["stratum"],
            "human_rating": rating, "human_numeric_readable": rating != "unreadable",
            "ed_prediction": ed_prediction, "ed_confidence": ed_confidence,
            "ed_logit_margin": ed_margin, "ed_target_logit_margin": ed_target_margin,
        })
    recomputed_results = _complete_metrics(normalized_rows, ec_float_logits, ec_png_logits)
    for index, row in enumerate(saved_rows):
        computed = normalized_rows[index]
        for field in ("ec_float_prediction", "ec_png_prediction"):
            _require(int(row[field]) == computed[field], f"saved CSV {field} differs from logits")
        for field in ("ec_float_confidence", "ec_float_logit_margin", "ec_float_target_logit_margin",
                      "ec_png_confidence", "ec_png_logit_margin", "ec_png_target_logit_margin"):
            _require(_float_matches(row[field], computed[field]), f"saved CSV {field} differs from logits")
        for field in ("ec_float_matches_human", "ec_png_matches_human", "ec_float_png_prediction_flip"):
            expected = computed[field]
            _require(row[field].strip().lower() == ("" if expected is None else str(expected).lower()),
                     f"saved CSV {field} differs from independent row comparison")
        expected_ed_match = computed["ed_prediction"] == int(computed["human_rating"]) if computed["human_rating"] != "unreadable" else None
        _require(row["ed_matches_human"].strip().lower() == ("" if expected_ed_match is None else str(expected_ed_match).lower()),
                 "saved CSV E_D human-match flag differs from independent comparison")
    _require(recomputed_results == report.get("results"),
             "saved report summary differs from the production CSV/logits consumer")
    receipt = base._read_json(output_dir / "validation_receipt.json")
    _require(receipt.get("status") == "PASSED" and receipt.get("formal_ec_samples") == 232
             and receipt.get("new_ed_forwards") == 0 and receipt.get("parameter_buffer_unchanged") is True,
             "saved output validation receipt is incomplete")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    (inputs, parent_manifest, child_manifest, parent_report, child_report,
     parent_ratings, child_ratings, rating_verification) = _load_inputs(args)
    parent_pngs = _load_review_cards(_resolve(args.original_review_dir), parent_manifest)
    child_pngs = _load_review_cards(_resolve(args.extension_review_dir), child_manifest)
    rows, float_images, png_images, ed_logits = _collect_rows(
        inputs, parent_manifest, parent_ratings, parent_pngs,
        child_manifest, child_ratings, child_pngs)
    review_png_uris = [parent_pngs[row["blind_id"]] if row["review_wave"] == "original60"
                       else child_pngs[row["blind_id"]] for row in rows]
    load_seconds = time.perf_counter() - load_started
    ec_path = _resolve(args.evaluator_checkpoint)
    model_started = time.perf_counter()
    evaluator, evaluator_metadata = _load_ec(ec_path)
    evaluator_load_seconds = time.perf_counter() - model_started
    before = _snapshot(evaluator)
    preflight_started = time.perf_counter()
    preflight_logits, preflight_calls = _forward_ec(evaluator, float_images[:1])
    _require(preflight_logits.shape == (1, 10), "representative E_C preflight did not return one logits row")
    preflight_seconds = time.perf_counter() - preflight_started
    if args.preflight_only:
        _require(_snapshots_equal(before, _snapshot(evaluator)), "E_C parameters or buffers changed during preflight")
        result = {"status": "PREFLIGHT_READY", "task_id": TASK_ID, "ratings": rating_verification,
                  "rows": len(rows), "evaluator": evaluator_metadata,
                  "execution": {"preflight_ec_samples": 1, "preflight_ec_batch_calls": preflight_calls,
                                "formal_ec_samples": 0, "new_ed_forwards": 0,
                                "optimizer_calls": 0, "optical_forwards": 0,
                                "load_seconds": load_seconds, "evaluator_load_seconds": evaluator_load_seconds,
                                "preflight_forward_seconds": preflight_seconds,
                                "elapsed_seconds": time.perf_counter() - started}}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    output_dir = _resolve(args.output_dir)
    expected_review_inputs = {
        "original_review_dir": _resolve(args.original_review_dir),
        "extension_review_dir": _resolve(args.extension_review_dir),
        "original_review_id": parent_manifest["review_id"],
        "extension_review_id": child_manifest["review_id"],
    }
    existing = output_dir.exists()
    _require(existing == bool(args.resume), "existing output requires --resume; --resume requires existing output")
    if existing:
        report = _validate_output(output_dir, inputs, rating_verification, **expected_review_inputs)
        print(json.dumps({"status": "RESUMED_COMPLETE", "output_dir": str(output_dir),
                          "results": report["results"]}, ensure_ascii=False), flush=True)
        return report

    formal_started = time.perf_counter()
    ec_float_logits, float_calls = _forward_ec(evaluator, float_images)
    ec_png_logits, png_calls = _forward_ec(evaluator, png_images)
    _require(ec_float_logits.shape == ec_png_logits.shape == (116, 10), "formal E_C output population is incomplete")
    formal_seconds = time.perf_counter() - formal_started
    metrics = _complete_metrics(rows, ec_float_logits, ec_png_logits)
    _require(_snapshots_equal(before, _snapshot(evaluator))
             and all(parameter.grad is None for parameter in evaluator.parameters()),
             "E_C parameters, buffers or gradients changed during inference")

    output_dir.mkdir(parents=True, exist_ok=False)
    sample_ids = np.asarray([f"{row['review_wave']}:{row['blind_id']}" for row in rows], dtype="U40")
    logits_buffer = io.BytesIO()
    np.savez_compressed(logits_buffer, sample_ids=sample_ids, ed_saved_logits=ed_logits,
                        ec_float_logits=ec_float_logits, ec_review_png_logits=ec_png_logits)
    outputs = {
        "rating_independent_verification.json": json.dumps(
            base._json_safe(rating_verification), ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        "per_image.csv": _csv_bytes(rows),
        "evaluator_logits.npz": logits_buffer.getvalue(),
    }
    for name, payload in outputs.items():
        _atomic_bytes(output_dir / name, payload)
    report = {
        "task_id": TASK_ID,
        "status": "COMPLETE_DIAGNOSTIC",
        "measurement_complete": True,
        "question": "Compare frozen E_D saved logits and frozen E_C predictions to the 116 human-review cards, retaining readable identity and unreadability as separate outcomes.",
        "inputs": {"original_review_dir": str(_resolve(args.original_review_dir)),
                   "extension_review_dir": str(_resolve(args.extension_review_dir)),
                   "attribution_dir": str(inputs.attribution_dir),
                   "ab_evidence_dir": str(inputs.ab_evidence_dir),
                   "evaluator_checkpoint": str(ec_path),
                   "original_task_id": parent_report["task_id"],
                   "extension_task_id": child_report["task_id"],
                   "original_review_id": parent_manifest["review_id"],
                   "extension_review_id": child_manifest["review_id"]},
        "rating_verification": rating_verification,
        "evaluator": evaluator_metadata,
        "results": metrics,
        "interpretation": {
            "identity_reference": "the human numeric rating for the 51 readable cards; no identity denominator for 65 unreadable cards",
            "selection_bias": "the cohort is selected by E_D strata and cannot rank overall evaluator accuracy or establish E_C superiority",
            "ec_scope": "E_C has historical qualification on canonical MNIST only; current CMOS-card comparison is diagnostic and does not qualify human readability",
            "ed_scope": "E_D scores are consumed from saved logits; no E_D CPU replay is claimed",
            "sample_truth": "generated cards do not have per-image generator ground truth; human labels are perceptual references for this review",
            "system_quality": "not re-evaluated; Gaussian A/B VALID_FAILED is preserved",
        },
        "execution": {"device": "cpu", "evaluator_eval_mode": True, "gradients_enabled": False,
                      "formal_ec_samples": 232, "formal_ec_batch_calls": float_calls + png_calls,
                      "formal_ec_float_samples": 116, "formal_ec_png_samples": 116,
                      "preflight_ec_samples": 1, "preflight_ec_batch_calls": preflight_calls,
                      "saved_ed_logit_rows": 116, "new_ed_forwards": 0,
                      "optical_model_constructions": 0, "optical_forwards": 0,
                      "encoder_forwards": 0, "decoder_forwards": 0, "sampling": 0,
                      "gradients": 0, "optimizer_constructions": 0, "optimizer_calls": 0,
                      "new_train_dev_test_reads": 0, "parameter_buffer_unchanged": True,
                      "elapsed_seconds": time.perf_counter() - started,
                      "input_load_seconds": load_seconds, "evaluator_load_seconds": evaluator_load_seconds,
                      "preflight_forward_seconds": preflight_seconds, "formal_forward_seconds": formal_seconds},
        "successor_advice": "Keep the human readability question separate from classifier identity metrics; use no classifier as a training objective based on this selected audit population.",
    }
    receipt = {
        "task_id": TASK_ID, "status": "PASSED",
        "checks": ["both review tasks, IDs, card sets and all 116 class/proposal pairs validated",
                   "all 116 embedded review PNGs decoded and matched fixed uint8 quantization of saved views",
                   "E_D predictions independently recomputed from saved logits; no E_D forward executed",
                   "E_C strict checkpoint load, architecture, class order and preprocessing validated",
                   "E_C output is tuple(features, logits) and both views produced finite 116x10 logits",
                   "unreadable cards excluded from identity agreement denominator",
                   "CSV, NPZ and report consumer retained all 116 rows and both E_C views",
                   "E_C parameters and buffers unchanged; no gradients, optical or optimizer operations"],
        "rating_independent_verification": "PASSED",
        "review_pngs_validated": 116,
        "formal_ec_samples": 232,
        "formal_ec_batch_calls": float_calls + png_calls,
        "preflight_ec_samples": 1,
        "new_ed_forwards": 0,
        "parameter_buffer_unchanged": True,
        "optimizer_calls": 0,
        "new_dataset_reads": 0,
    }
    report["validation_receipt"] = receipt
    _atomic_json(output_dir / "report.json", report)
    _atomic_bytes(output_dir / "report.html", _html_report(report).encode("utf-8"))
    _atomic_bytes(output_dir / "fixed_cases.html", _fixed_cases_html(rows, float_images, review_png_uris).encode("utf-8"))
    _atomic_json(output_dir / "validation_receipt.json", receipt)
    readable = metrics["readable_numeric_identity_comparison"]["agreement_groups"]
    rec_nine = metrics["readable_human_nines"]
    closeout = (
        f"结果与状态：`{report['status']}`；56条追加评分已正式收尾，固定116卡交叉评价完整。主要60例为9={rating_verification['primary60']['counts']['readable9']}、其他数字={rating_verification['primary60']['counts']['readable_other']}、不可辨认={rating_verification['primary60']['counts']['unreadable']}，仍为`MIXED_OR_UNRESOLVED`。\n\n"
        f"科学认识：51张可辨图中，两分类器都与人工数字一致{readable['both_agree']}张，仅E_C一致{readable['ec_only_agrees']}张，仅E_D一致{readable['ed_only_agrees']}张，两者均不一致{readable['neither_agrees']}张；在人眼判9且E_D漏判的{rec_nine['ed_misses']}张中，E_C PNG恢复{rec_nine['ec_png_recovers_ed_miss']}张。65张不可辨认图不设身份真值。\n\n"
        "目标缺口：E_C历史标准MNIST资格不证明CMOS人眼可读；E_D分层选样不能给总体分类准确率排序，生成图没有逐图真值；真实硬件与A/B系统质量未重新评价。\n\n"
        f"工程完成与验证：Windows CPU完成232个E_C样本前向（预检另计1个），E_D使用保存logits；116张网页PNG逐张解码并和固定量化视图一致。聚焦pytest、Ruff、编译与完整输出复读见验证收据。参数和buffer未变，光学、训练、优化器和新数据读取为0；本次总耗时{report['execution']['elapsed_seconds']:.2f}秒。\n\n"
        "决策影响与下一步：分类器读数仅作为身份互补诊断，不用E_C或E_D作为训练目标。建议后继质量工作聚焦固定增益下的可视结构改进，并以人眼可辨性作为独立结果；该建议不授权训练、调增益或扩大评阅。Gaussian A/B `VALID_FAILED`、大架构历史参照和硬件缺口均保持原义。\n"
    )
    _atomic_bytes(output_dir / "closeout_report.md", closeout.encode("utf-8"))
    size = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    _require(size <= MAX_OUTPUT_BYTES, f"audit output exceeds 64 MiB: {size}")
    _validate_output(output_dir, inputs, rating_verification, **expected_review_inputs)
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir),
                      "output_bytes": size, "results": metrics,
                      "elapsed_seconds": report["execution"]["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-review-dir", type=Path, required=True)
    parser.add_argument("--extension-review-dir", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--ab-evidence-dir", type=Path, required=True)
    parser.add_argument("--evaluator-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.resume:
        parser.error("--preflight-only cannot be combined with --resume")
    try:
        run(args)
    except base.EvidenceError as error:
        raise SystemExit(f"INVALID_HARNESS: {error}") from error


if __name__ == "__main__":
    main()
