"""Finalize the fixed D review, train one joint-update endpoint, and audit a small auxiliary reviewer."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
import scripts.research.lightgen.execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder_ref
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.evidence import atomic_torch_save, atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from ldm_project.v2.evidence.io import cpu_copy


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_JOINT_UPDATE_READABILITY_PILOT_V1"
AUTHORITY = "USER_APPROVED_SMALL_JOINT_UPDATE_READABILITY_PILOT_20261023"
SOURCE_STEP = 45_000
UPDATES = 5_000
ENDPOINT_STEP = SOURCE_STEP + UPDATES
BATCH_SIZE = 10
SEED = 2026101501
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
RECOVERY_RTOL = 2e-6
RECOVERY_ATOL = 5e-8
WINDOW_TOLERANCE = 1e-4
RECON_BOOTSTRAP_SEED = 2026102301
GEN_BOOTSTRAP_SEED = 2026102302
BOOTSTRAP_RESAMPLES = 2_000
TASK_SCHEMA = "lightgen_small_joint_update_readability_pilot.v1"
CHECKPOINT_SCHEMA = "lightgen_small_joint_update_checkpoint.v1"
PROTOTYPE_SCHEMA = "lightgen_cmos_readability_auxiliary_prototype.v1"
ED_SCHEMA = "pixel_mlp_confirmation_evaluator_checkpoint.v1"


def _require(ok: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not ok:
        raise error_type(message)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("schema") == TASK_SCHEMA
             and value.get("task_id") == TASK_ID and value.get("semantic_authority_id") == AUTHORITY,
             "joint-update configuration identity changed")
    training = value["training"]
    _require(training == {
        "seed": SEED, "source_global_step": SOURCE_STEP, "continuation_updates": UPDATES,
        "endpoint_global_step": ENDPOINT_STEP, "batch_size": BATCH_SIZE,
        "samples_per_class_per_batch": 1, "optimizer": "Adam", "learning_rate": 1e-3,
        "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0,
        "trainable_groups": ["encoder_phase", "decoder_phase"],
        "encoder_updates": UPDATES, "decoder_updates": UPDATES,
        "checkpoint_interval": CHECKPOINT_INTERVAL, "progress_interval": PROGRESS_INTERVAL,
        "checkpoints_at": [0, 2500, UPDATES],
    }, "joint-update budget, trainable scope, optimizer or checkpoint plan changed")
    data = value["data"]
    _require(data == {
        "training_file": "data/MNIST/processed/training.pt", "split_seed": 1234,
        "train_count": 55_000, "dev_count": 5_000, "dev_per_class": 500,
        "official_test_reads": 0,
    }, "MNIST data roles or split changed")
    optics = value["optics"]
    _require(optics["geometry_status"] == "PROJECT_GEOMETRY_UNMEASURED"
             and optics["ols_basis"] == "gaussian_s12_v1" and optics["phase_planes"] == 4
             and optics["phase_parameters"] == 40_000 and optics["propagation_grid"] == 2048
             and optics["reference_grid"] == 4096 and optics["common_complex_roi_m"] == 1.6e-4,
             "registered Gaussian geometry or numerical reference changed")
    _require(value["evaluation"]["evaluator_role"] == "frozen_confirmation_only"
             and value["evaluation"]["reconstruction_bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and value["evaluation"]["reconstruction_bootstrap_seed"] == RECON_BOOTSTRAP_SEED
             and value["evaluation"]["generation_bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and value["evaluation"]["generation_bootstrap_seed"] == GEN_BOOTSTRAP_SEED,
             "registered evaluation population or bootstrap changed")
    _require(value["quality_gate"] == {
        "reconstruction_psnr_db_min": 20.0, "reconstruction_ssim_min": 0.80,
        "generation_macro_recall_min": 0.80, "generation_recall_per_class_min": 0.60,
        "proxy_clipping_fraction_max": 0.01, "generated_to_dev_tv_ratio_min": 0.25,
    }, "historical system quality checklist changed")
    _require(value["output"]["experiment_root"] ==
             "outputs/20261022_lightgen_small_joint_update_readability_pilot_v1/runs/attempt_01"
             and value["output"]["formal_result_budget_bytes"] == 536_870_912,
             "task output or result budget changed")
    return value


def _resolve(value: Path, root: Path = ROOT) -> Path:
    return (value if value.is_absolute() else root / value).resolve()


def _same_json(left: Any, right: Any) -> bool:
    return json.loads(json.dumps(left, sort_keys=True, ensure_ascii=False)) == json.loads(
        json.dumps(right, sort_keys=True, ensure_ascii=False))


def _finalize_previous_review(
    control_dir: Path, ratings_path: Path, review_config_path: Path,
) -> dict[str, Any]:
    review_config = decoder_ref._read_config(review_config_path.resolve())
    ratings_path = ratings_path.resolve()
    saved_path = control_dir / "human_ratings_submitted.json"
    final_path = control_dir / "final_report.json"
    if final_path.is_file():
        _require(saved_path.is_file(), "finalized D review is missing its archived original ratings", FileNotFoundError)
        _require(_same_json(json.loads(saved_path.read_text(encoding="utf-8")),
                            json.loads(ratings_path.read_text(encoding="utf-8-sig"))),
                 "a different rating payload is already finalized; refusing to overwrite")
    else:
        _require(not saved_path.exists(), "D review has an orphaned ratings archive without a final report")
        decoder_ref._finalize_ratings(control_dir, ratings_path, review_config)
    report = json.loads(final_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == decoder_ref.TASK_ID
             and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("measurement_complete") is True
             and report.get("human_review", {}).get("conclusion") == "HUMAN_INDEPENDENT_EFFECT_UNCLEAR"
             and report.get("human_review", {}).get("quality_checks", {}).get(
                 "duplicate_consistency", {}).get("count") == 18
             and report.get("human_review", {}).get("quality_checks", {}).get(
                 "real_anchor_label_match", {}).get("count") == 20,
             "D ratings did not reach the registered diagnostic terminal state")
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(bool(rows), f"CSV is empty: {path}", FileNotFoundError)
    return rows


def _validate_finished_review(
    review_dir: Path, ratings_path: Path, expected_task: str, expected_review_id: str,
    expected_population: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = review_dir / "final_report.json"
    key_path = review_dir / "blind_review_key.json"
    manifest_path = review_dir / "blind_review_public_manifest.json"
    saved_ratings_path = review_dir / "human_ratings_submitted.json"
    for path in (report_path, key_path, manifest_path, saved_ratings_path, ratings_path):
        _require(path.is_file(), f"completed review evidence is missing: {path}", FileNotFoundError)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    submitted = json.loads(saved_ratings_path.read_text(encoding="utf-8-sig"))
    incoming = json.loads(ratings_path.read_text(encoding="utf-8-sig"))
    _require(_same_json(submitted, incoming), "archived human ratings differ from the submitted payload")
    _require(report.get("task_id") == expected_task and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("measurement_complete") is True
             and report.get("human_review", {}).get("review_id") == expected_review_id
             and key.get("task_id") == expected_task and key.get("review_id") == expected_review_id
             and manifest.get("task_id") == expected_task and manifest.get("review_id") == expected_review_id,
             "review package task, status, or review identity changed")
    cards = key.get("cards")
    rating_cards = submitted.get("cards")
    _require(isinstance(cards, list) and len(cards) == 240 and isinstance(rating_cards, list)
             and len(rating_cards) == 240 and len({item.get("card_id") for item in cards}) == 240
             and len({item.get("card_id") for item in rating_cards}) == 240
             and {item.get("card_id") for item in cards} == {item.get("card_id") for item in rating_cards},
             "review cards or ratings have missing, duplicate, or foreign IDs")
    rating_by_id = {item["card_id"]: item.get("rating") for item in rating_cards}
    _require(all(value == "unreadable" or (type(value) is int and 0 <= value <= 9)
                 for value in rating_by_id.values()), "ratings contain an invalid human label")
    generated = [card for card in cards if card.get("kind") == "generated"]
    _require(len(generated) == expected_population, "generated review population changed")
    qc = report.get("human_review", {}).get("quality_checks", {})
    dup = qc.get("duplicate_consistency", {}).get("count")
    anchor = qc.get("real_anchor_label_match", qc.get("anchor_label_match", {})).get("count")
    _require(isinstance(dup, int) and dup >= 18 and isinstance(anchor, int) and anchor >= 18,
             "review quality checks do not meet the fixed minimum")
    return report, key, rating_by_id


def _records_from_reviews(
    reference_dir: Path, control_dir: Path, reference_ratings_path: Path,
    control_ratings_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r_report, r_key, r_rating = _validate_finished_review(
        reference_dir, reference_ratings_path, "LIGHTGEN_SMALL_G1_G4_NATIVE_READABILITY_V1",
        "99d8041812f54d44b98031876fa21519", 200)
    d_report, d_key, d_rating = _validate_finished_review(
        control_dir, control_ratings_path, "LIGHTGEN_SMALL_DECODER_NATIVE_GRADIENT_AB_V1",
        "6346e07062f541cb904d65c016813b5c", 200)
    _require(r_report.get("human_review", {}).get("interpretation") == "HUMAN_READABILITY_IMPROVEMENT_SUPPORTED"
             and d_report.get("human_review", {}).get("conclusion") == "HUMAN_INDEPENDENT_EFFECT_UNCLEAR",
             "the registered R/D review conclusions changed")

    r_rows = _read_csv(reference_dir / "selected_per_image.csv")
    r_package = torch.load(reference_dir / "selected_saved_classifiers.pt", map_location="cpu", weights_only=False)
    _require(r_package.get("task_id") == r_report.get("task_id"), "R saved logits belong to another review task")
    r_feature_by_id: dict[tuple[str, int, int, int], tuple[np.ndarray, dict[str, float]]] = {}
    for method in ("G1", "G4"):
        payload = r_package.get(method)
        method_rows = [row for row in r_rows if row.get("method") == method]
        _require(isinstance(payload, dict) and len(method_rows) == 100
                 and tuple(payload.get("E_D_logits_saved", torch.empty(0)).shape) == (100, 10)
                 and tuple(payload.get("E_C_logits_saved", torch.empty(0)).shape) == (100, 10),
                 f"R {method} saved logits or fixed population is incomplete")
        for index, row in enumerate(method_rows):
            label, stream, proposal = int(row["requested_label"]), int(row["stream"]), int(row["proposal_id"])
            _require(int(payload["requested_labels"][index]) == label
                     and int(payload["proposal_ids"][index]) == proposal,
                     f"R {method} logit row identity differs from its saved feature row")
            key = (method, stream, label, proposal)
            features = np.concatenate((payload["E_D_logits_saved"][index].double().numpy(),
                                       payload["E_C_logits_saved"][index].double().numpy(),
                                       np.asarray([float(row["view_mean_intensity"]),
                                                   float(row["raw_peak_w_per_pixel"]),
                                                   float(row["proxy_clipping_area_fraction"])], dtype=np.float64)))
            r_feature_by_id[key] = (features, row)
    r_records: list[dict[str, Any]] = []
    for card in r_key["cards"]:
        if card.get("kind") != "generated":
            continue
        method, label, stream, proposal = (card.get("method"), int(card["requested_label"]),
                                           int(card["stream"]), int(card["proposal_id"]))
        key = (str(method), stream, label, proposal)
        _require(key in r_feature_by_id, f"R generated card does not map to saved classifier row: {key}")
        features, metric_row = r_feature_by_id[key]
        _require(int(metric_row["cohort_row"]) == int(card["cohort_row"])
                 and int(metric_row["component_id"]) == int(card["component_id"]),
                 "R generated card row or component identity differs")
        r_records.append({
            "package": "R", "card_id": card["card_id"], "branch": method, "method": method,
            "requested_label": label, "human_label": r_rating[card["card_id"]],
            "source_identity": [method, stream, label, proposal], "features": features,
            "ed_prediction": int(np.argmax(features[:10])), "ec_prediction": int(np.argmax(features[10:20])),
        })

    d_rows_by_branch = {branch: _read_csv(control_dir / branch / "generation_metrics_per_image.csv")
                        for branch in ("D0", "Dgrad")}
    d_generations = {branch: torch.load(control_dir / branch / "generated_cohort.pt", map_location="cpu",
                                        weights_only=False) for branch in ("D0", "Dgrad")}
    d_records: list[dict[str, Any]] = []
    d_identity_branches: dict[tuple[str, int, int, int], set[str]] = {}
    for card in d_key["cards"]:
        if card.get("kind") != "generated":
            continue
        branch = str(card.get("branch"))
        _require(branch in ("D0", "Dgrad"), "D generated card has an invalid branch")
        row_index = int(card["cohort_row"])
        _require(0 <= row_index < 10_000, "D card cohort row is out of range")
        metric_row = d_rows_by_branch[branch][row_index]
        gen = d_generations[branch]
        label, proposal = int(card["requested_label"]), int(card["proposal_id"])
        _require(int(metric_row["cohort_row"]) == row_index and int(metric_row["requested_label"]) == label
                 and int(metric_row["proposal_id"]) == proposal
                 and int(gen["requested_labels"][row_index]) == label
                 and int(gen["proposal_ids"][row_index]) == proposal
                 and int(gen["component_ids"][row_index]) == int(metric_row["component_id"])
                 and ("component_id" not in card
                      or int(gen["component_ids"][row_index]) == int(card["component_id"])),
                 "D generated card identity differs from its saved production row")
        ed_logits = gen["ed_logits"][row_index].double().numpy()
        ec_logits = gen["ec_logits"][row_index].double().numpy()
        features = np.concatenate((ed_logits, ec_logits,
                                   np.asarray([float(metric_row["view_mean_intensity"]),
                                               float(metric_row["raw_peak_w_per_pixel"]),
                                               float(metric_row["proxy_clipping_area_fraction"])], dtype=np.float64)))
        source_identity = ["G4", 0, label, proposal]
        d_identity_branches.setdefault(tuple(source_identity), set()).add(branch)
        d_records.append({
            "package": "D", "card_id": card["card_id"], "branch": branch, "method": "G4",
            "requested_label": label, "human_label": d_rating[card["card_id"]],
            "source_identity": source_identity, "features": features,
            "ed_prediction": int(np.argmax(ed_logits)), "ec_prediction": int(np.argmax(ec_logits)),
        })
    _require(len(r_records) == len(d_records) == 200, "R/D independent generated populations must each contain 200 cards")
    _require(all(branches == {"D0", "Dgrad"} for branches in d_identity_branches.values())
             and len(d_identity_branches) == 100,
             "D0/Dgrad paired G4 coefficients are missing or split across package identities")
    r_identities = {tuple(row["source_identity"]) for row in r_records if row["method"] == "G4"}
    d_identities = {tuple(row["source_identity"]) for row in d_records}
    _require(not (r_identities & d_identities), "same G4 proposal identity crosses the two review-package folds")
    records = r_records + d_records
    _require(len(records) == 400 and all(len(row["features"]) == 23
                                        and np.isfinite(row["features"]).all() for row in records),
             "automatic-review feature matrix is incomplete or nonfinite")
    summary = {
        "source_reviews": {
            "R": {"task_id": r_report["task_id"], "review_id": r_report["human_review"]["review_id"],
                  "duplicate_consistency": r_report["human_review"]["quality_checks"]["duplicate_consistency"]["count"],
                  "anchors_correct": r_report["human_review"]["quality_checks"]["real_anchor_label_match"]["count"]},
            "D": {"task_id": d_report["task_id"], "review_id": d_report["human_review"]["review_id"],
                  "duplicate_consistency": d_report["human_review"]["quality_checks"]["duplicate_consistency"]["count"],
                  "anchors_correct": d_report["human_review"]["quality_checks"]["real_anchor_label_match"]["count"]},
        },
        "records": 400, "numeric_human_labels": sum(row["human_label"] != "unreadable" for row in records),
        "unreadable_human_labels": sum(row["human_label"] == "unreadable" for row in records),
        "D0_Dgrad_same_coefficients_kept_in_one_group": True,
        "cross_package_shared_G4_identity_count": 0,
    }
    return records, summary


def _fit_auxiliary_model(
    train_features: np.ndarray, train_labels: list[int | str], steps: int = 500,
) -> dict[str, Any]:
    """Fit the fixed two-head linear helper using only explicitly supplied training-fold labels."""
    x = torch.as_tensor(train_features, dtype=torch.float64, device="cpu")
    _require(x.ndim == 2 and x.shape[1] == 23 and x.shape[0] == len(train_labels) > 0
             and torch.isfinite(x).all().item(), "auxiliary training features are malformed")
    y = torch.as_tensor([0 if label == "unreadable" else 1 for label in train_labels], dtype=torch.float64)
    numeric_mask = torch.as_tensor([label != "unreadable" for label in train_labels], dtype=torch.bool)
    numeric = torch.as_tensor([0 if label == "unreadable" else int(label) for label in train_labels],
                              dtype=torch.long)
    _require(bool(numeric_mask.any()), "training fold contains no numeric human labels")
    mean = x.mean(0)
    std = x.std(0, unbiased=False)
    divisor = torch.where(std > 0.0, std, torch.ones_like(std))
    standardized = (x - mean) / divisor
    digit = torch.nn.Linear(23, 10, dtype=torch.float64)
    readable = torch.nn.Linear(23, 1, dtype=torch.float64)
    with torch.no_grad():
        digit.weight.zero_()
        digit.bias.zero_()
        readable.weight.zero_()
        readable.bias.zero_()
    optimizer = torch.optim.Adam(list(digit.parameters()) + list(readable.parameters()), lr=0.01,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        digit_logits = digit(standardized[numeric_mask])
        readable_logits = readable(standardized).squeeze(1)
        weights = torch.cat((digit.weight.reshape(-1), readable.weight.reshape(-1)))
        loss = (F.cross_entropy(digit_logits, numeric[numeric_mask])
                + F.binary_cross_entropy_with_logits(readable_logits, y)
                + 0.01 * weights.square().mean())
        _require(torch.isfinite(loss).item(), "auxiliary review objective became nonfinite")
        loss.backward()
        _require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                     for parameter in list(digit.parameters()) + list(readable.parameters())),
                 "auxiliary review model received invalid gradients")
        optimizer.step()
    return {"mean": mean, "scale": divisor, "digit_weight": digit.weight.detach().clone(),
            "digit_bias": digit.bias.detach().clone(), "readable_weight": readable.weight.detach().clone(),
            "readable_bias": readable.bias.detach().clone(), "training_loss": float(loss.detach()),
            "optimizer_steps": steps}


def _predict_auxiliary(model: dict[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    _require(x.ndim == 2 and x.shape[1] == 23 and torch.isfinite(x).all().item(),
             "auxiliary prediction features are malformed")
    z = (x - model["mean"]) / model["scale"]
    digit_logits = F.linear(z, model["digit_weight"], model["digit_bias"])
    readable_logits = F.linear(z, model["readable_weight"], model["readable_bias"]).squeeze(1)
    probabilities = readable_logits.sigmoid().numpy()
    digits = digit_logits.argmax(1).numpy().astype(np.int64)
    predictions = np.where(probabilities >= 0.5, digits, 10).astype(np.int64)
    return predictions, probabilities, torch.softmax(digit_logits, dim=1).numpy()


def _review_metrics(records: list[dict[str, Any]], predictions: np.ndarray,
                    probabilities: np.ndarray) -> dict[str, Any]:
    _require(len(records) == len(predictions) == len(probabilities), "review prediction population is incomplete")
    y = np.asarray([10 if row["human_label"] == "unreadable" else int(row["human_label"])
                    for row in records], dtype=np.int64)
    confusion = np.zeros((11, 11), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)
    numeric = y < 10
    unreadable = ~numeric
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    brier = float(np.mean((p - numeric.astype(np.float64)) ** 2))
    logloss = float(-np.mean(numeric * np.log(p) + unreadable * np.log(1 - p)))
    per_class = []
    for label in range(10):
        mask = y == label
        per_class.append({"label": label, "count": int(mask.sum()),
                          "human_digit_agreement": float(np.mean(predictions[mask] == label)) if mask.any() else None,
                          "unreadable_rate": float(np.mean(predictions[mask] == 10)) if mask.any() else None})
    return {
        "population": len(records), "numeric_human_count": int(numeric.sum()),
        "human_unreadable_count": int(unreadable.sum()),
        "digit_agreement_on_numeric_human_labels": float(np.mean(predictions[numeric] == y[numeric])),
        "unreadable_recall": float(np.mean(predictions[unreadable] == 10)) if unreadable.any() else None,
        "unreadable_false_accept_rate": float(np.mean(predictions[unreadable] < 10)) if unreadable.any() else None,
        "readable_false_reject_rate": float(np.mean(predictions[numeric] == 10)),
        "readability_brier_score": brier, "readability_log_loss": logloss,
        "confusion_rows_human_0_to_9_then_unreadable": confusion.tolist(),
        "per_human_digit": per_class,
    }


def _run_auxiliary_prototype(
    records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path,
) -> dict[str, Any]:
    _require(len(records) == 400, "auxiliary prototype requires exactly the two fixed 200-card packages")
    card_keys = {(row["package"], row["card_id"]) for row in records}
    _require(len(card_keys) == 400, "auxiliary prototype package/card identities are duplicated")
    fold_specs = (("train_R_test_D", "R", "D"), ("train_D_test_R", "D", "R"))
    predictions_by_id: dict[str, dict[str, Any]] = {}
    fold_reports = []
    for name, training_package, validation_package in fold_specs:
        training = [row for row in records if row["package"] == training_package]
        validation = [row for row in records if row["package"] == validation_package]
        _require(len(training) == len(validation) == 200, "package-held-out fold population changed")
        model = _fit_auxiliary_model(np.stack([row["features"] for row in training]),
                                     [row["human_label"] for row in training])
        predicted, readable_probability, digit_probability = _predict_auxiliary(
            model, np.stack([row["features"] for row in validation]))
        metrics = _review_metrics(validation, predicted, readable_probability)
        fold_reports.append({"fold": name, "training_package": training_package,
                             "validation_package": validation_package,
                             "training_rows": len(training), "validation_rows": len(validation),
                             "numeric_training_rows": sum(row["human_label"] != "unreadable" for row in training),
                             "optimizer_steps": 500, "metrics": metrics})
        for row, output, probability, digit_prob in zip(validation, predicted, readable_probability,
                                                        digit_probability, strict=True):
            card_key = f"{row['package']}:{row['card_id']}"
            _require(card_key not in predictions_by_id, "a generated review card appears in multiple folds")
            predictions_by_id[card_key] = {
                "prototype_prediction": int(output), "readable_probability": float(probability),
                "digit_probability": digit_prob.tolist(),
            }
    _require(len(predictions_by_id) == 400, "out-of-fold predictions are missing review cards")
    oof_rows = []
    for row in records:
        card_key = f"{row['package']}:{row['card_id']}"
        oof = predictions_by_id[card_key]
        oof_rows.append({
            "package": row["package"], "card_id": row["card_id"], "branch": row["branch"],
            "method": row["method"], "requested_label": row["requested_label"],
            "human_label": row["human_label"], "source_identity": json.dumps(row["source_identity"]),
            "ed_prediction": row["ed_prediction"], "ec_prediction": row["ec_prediction"],
            **oof,
        })
    oof_path = output_dir / "out_of_fold_predictions.csv"
    decoder_ref._write_rows_csv(oof_path, oof_rows)
    oof_predictions = np.asarray([predictions_by_id[f"{row['package']}:{row['card_id']}"]["prototype_prediction"]
                                  for row in records])
    oof_readable_prob = np.asarray([predictions_by_id[f"{row['package']}:{row['card_id']}"]["readable_probability"]
                                    for row in records])
    prototype_metrics = _review_metrics(records, oof_predictions, oof_readable_prob)
    baseline = {}
    y = np.asarray([10 if row["human_label"] == "unreadable" else int(row["human_label"])
                    for row in records], dtype=np.int64)
    for name, key in (("E_D", "ed_prediction"), ("E_C", "ec_prediction")):
        p = np.asarray([row[key] for row in records], dtype=np.int64)
        numeric = y < 10
        baseline[name] = {
            "digit_agreement_on_numeric_human_labels": float(np.mean(p[numeric] == y[numeric])),
            "human_unreadable_false_accept_rate": float(np.mean(p[y == 10] < 10)) if np.any(y == 10) else None,
            "unreadable_output_supported": False,
            "interpretation": "forced digit predictions; cannot identify human unreadability",
        }
    full_model = _fit_auxiliary_model(np.stack([row["features"] for row in records]),
                                     [row["human_label"] for row in records])
    atomic_torch_save_new(output_dir / "development_model.pt", {
        "schema": PROTOTYPE_SCHEMA, "task_id": TASK_ID, "features": 23,
        "folds": ["train_R_test_D", "train_D_test_R"],
        "readability_threshold": 0.5, "model": full_model,
    })
    report = {
        "schema": PROTOTYPE_SCHEMA, "status": "AUXILIARY_PILOT_COMPLETE",
        "records": summary, "folds": fold_reports,
        "out_of_fold": prototype_metrics, "frozen_classifier_comparison": baseline,
        "development_fit": {"optimizer_steps": 500, "training_rows": 400,
                             "purpose": "development artifact only; not independent validation"},
        "optimizer_calls": 1500,
        "limitations": [
            "two package-held-out folds from one reviewer and one model family",
            "the model can assist or triage review but does not replace blind human review",
            "neither E_D nor E_C is qualified for human readability",
            "no requested class, branch, method, proposal ID, or rating prompt is a model feature",
            "the prototype is excluded from optical training and scientific acceptance",
        ],
    }
    atomic_write_json(output_dir / "report.json", report)
    _write_auxiliary_html(output_dir, report)
    return report


def _write_auxiliary_html(output_dir: Path, report: dict[str, Any]) -> None:
    sections = []
    for fold in report["folds"]:
        metrics = fold["metrics"]
        sections.append(
            f"<h2>{html.escape(fold['fold'])}</h2><p>折外数字吻合率 "
            f"{metrics['digit_agreement_on_numeric_human_labels']:.1%}；不可辨认召回 "
            f"{metrics['unreadable_recall']:.1%}；误接收率 {metrics['unreadable_false_accept_rate']:.1%}。</p>"
        )
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>辅助人工评阅原型</title>"
            "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.55}"
            "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.35rem}</style>"
            "<h1>400条既有盲评的辅助模型试点</h1>"
            "<p>整包留出两折；数值预测和可辨认度分开。该模型只用于辅助分流，不替代盲评。</p>"
            + "".join(sections)
            + "<p>折外总体数字吻合率 "
            + f"{report['out_of_fold']['digit_agreement_on_numeric_human_labels']:.1%}；"
            + f"不可辨认召回 {report['out_of_fold']['unreadable_recall']:.1%}；"
            + f"不可辨认误接收率 {report['out_of_fold']['unreadable_false_accept_rate']:.1%}。</p>"
            + "<p>与E_D/E_C的逐样本读数见 report.json；冻结分类器只输出数字，不能识别不可辨认。</p>"
            + "</html>")
    (output_dir / "report.html").write_text(page, encoding="utf-8")


def _write_paper_scope(path: Path) -> None:
    paper = {
        "title": "All-optical synthesis chip for large-scale intelligent semantic vision generation",
        "journal": "Science", "volume": 390, "pages": "1259-1265", "year": 2025,
        "doi": "10.1126/science.adv7434",
        "task_metrics": [
            {"task": "style transfer Fig. 3F", "lightgen_psnr_db_mean": 10.14,
             "range": [8.36, 12.48], "comparison_note": "style target differs from pixel reconstruction"},
            {"task": "3D view reconstruction Fig. 4D", "lightgen_psnr_db_mean": 13.79,
             "range": [8.07, 23.45], "comparison_note": "data, target and imaging conditions differ"},
            {"task": "misalignment calibration Fig. S19", "example_psnr_db_before": 13.80,
             "example_psnr_db_after": 16.05, "comparison_note": "single calibration case, not a system threshold"},
            {"task": "generated animal classification Fig. 3B", "species_accuracy": 0.95,
             "fur_color_accuracy": 0.90, "afhq_reference_accuracy": 0.967,
             "comparison_note": "binary/three-class animal proxies, not MNIST or human readability"},
        ],
        "ssim_reported_in_checked_main_and_supplement": False,
        "project_quality_checklist": {"psnr_db_min": 20.0, "ssim_min": 0.80,
                                       "provenance": "existing LightGen project standard, not a Science paper requirement"},
        "sources": ["https://doi.org/10.1126/science.adv7434",
                    "https://zenodo.org/records/17385609",
                    "https://bdt.semi.ac.cn/library/contents/757/24096.html"],
        "interpretation": "The paper's task-specific values do not justify lowering this project's historical checklist."
    }
    atomic_write_json(path, paper)


def _run_ratings_and_prototype(
    run_dir: Path, config: dict[str, Any], ratings_path: Path,
    control_dir: Path, reference_dir: Path,
) -> dict[str, Any]:
    _require(control_dir.resolve() == (ROOT / config["control"]["evidence_dir"]).resolve()
             and reference_dir.resolve() == (ROOT / config["readability_reference"]["directory"]).resolve(),
             "rating finalization inputs differ from the registered D/R review evidence")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_paper_scope(run_dir / "paper_metric_scope.json")
    d_report = _finalize_previous_review(
        control_dir, ratings_path, decoder_ref.ROOT / config["control"]["previous_config"])
    final_path = control_dir / "final_report.json"
    r_ratings = reference_dir / "human_ratings_submitted.json"
    records, record_summary = _records_from_reviews(reference_dir, control_dir, r_ratings, ratings_path)
    review_dir = run_dir / "human_review_closeout"
    review_dir.mkdir(exist_ok=True)
    closeout = {
        "status": d_report["human_review"]["status"],
        "conclusion": d_report["human_review"]["conclusion"],
        "ratings_file": str(ratings_path.resolve()),
        "formal_report": str(final_path.resolve()),
        "duplicate_consistency": d_report["human_review"]["quality_checks"]["duplicate_consistency"],
        "real_anchor_label_match": d_report["human_review"]["quality_checks"]["real_anchor_label_match"],
        "D0_requested_match": d_report["human_review"]["D0"]["equal_class_requested_match_rate"],
        "Dgrad_requested_match": d_report["human_review"]["Dgrad"]["equal_class_requested_match_rate"],
        "Dgrad_minus_D0": d_report["human_review"]["Dgrad_minus_D0"],
        "overwrite_or_new_cards": False,
    }
    atomic_write_json(review_dir / "closeout.json", closeout)
    prototype_dir = run_dir / "auto_review_prototype"
    prototype_dir.mkdir(exist_ok=True)
    prototype = _run_auxiliary_prototype(records, record_summary, prototype_dir)
    atomic_write_json(review_dir / "prototype_source_receipt.json", {
        "schema": "lightgen_joint_update_human_source_receipt.v1",
        "R_review_id": record_summary["source_reviews"]["R"]["review_id"],
        "D_review_id": record_summary["source_reviews"]["D"]["review_id"],
        "source_records": 400, "prototype_records": 400,
        "R_quality_passed": True, "D_quality_passed": True,
        "held_out_by_review_package": True,
    })
    atomic_write_json(run_dir / "task_config.json", config)
    return {"human_review": closeout, "auto_review_prototype": prototype}


def _load_fixed_inputs(
    config: dict[str, Any], source_path: Path, control_dir: Path,
    cohort_path: Path, reference_dir: Path,
) -> dict[str, Any]:
    expected_paths = {
        "source": ROOT / config["source"]["checkpoint"],
        "control": ROOT / config["control"]["evidence_dir"],
        "cohort": ROOT / config["generation"]["cohort"],
        "readability": ROOT / config["readability_reference"]["directory"],
    }
    _require(source_path.resolve() == expected_paths["source"].resolve()
             and control_dir.resolve() == expected_paths["control"].resolve()
             and cohort_path.resolve() == expected_paths["cohort"].resolve()
             and reference_dir.resolve() == expected_paths["readability"].resolve(),
             "explicit source/control/G4/review paths differ from the registered fixed inputs")
    previous_config_path = (ROOT / config["control"]["previous_config"]).resolve()
    previous_config = decoder_ref._read_config(previous_config_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("schema") == "lightgen_small_gaussian_foreground_balance_checkpoint.v1"
             and checkpoint.get("task_id") == decoder_ref.SOURCE_TASK_ID
             and checkpoint.get("branch") == "C" and checkpoint.get("seed") == SEED
             and checkpoint.get("completed_global_step") == SOURCE_STEP
             and checkpoint.get("completed_updates") == 10_000
             and checkpoint.get("source_checkpoint_step") == 35_000
             and checkpoint.get("evaluator_role") == "frozen_confirmation_only",
             "loaded source is not the registered C 45,000-step endpoint")
    source_identity = checkpoint.get("model_identity", {})
    _require(source_identity == previous_config["source"]["model_identity"]
             and tuple(checkpoint["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(checkpoint["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100)
             and float(checkpoint.get("structural_weight", math.nan)) == previous_config["loss"]["structural_weight"],
             "C source model identity, phase geometry or frozen SSIM coefficient changed")
    source_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    source_states = checkpoint.get("optimizer", {}).get("state", {})
    _require(len(source_groups) == 1 and len(source_groups[0].get("params", [])) == 2
             and source_groups[0].get("lr") == 1e-3
             and tuple(source_groups[0].get("betas", ())) == (0.9, 0.999)
             and source_groups[0].get("eps") == 1e-8 and source_groups[0].get("weight_decay") == 0.0
             and len(source_states) == 2
             and all(int(state["step"].item()) == SOURCE_STEP for state in source_states.values()),
             "C source Adam group or 45,000-step phase state changed")
    for name in ("scale", "power_reference", "camera_gain"):
        _require(torch.isfinite(torch.as_tensor(checkpoint[name])).all().item(),
                 f"C source {name} is nonfinite")
    source_dir = source_path.parent
    source_endpoint_path = source_dir / "endpoint_summary.json"
    source_report_path = source_dir.parent / "final_report.json"
    for path in (source_endpoint_path, source_report_path):
        _require(path.is_file(), f"C source endpoint evidence is missing: {path}", FileNotFoundError)
    source_endpoint = json.loads(source_endpoint_path.read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    baseline_config_path = (ROOT / previous_config["source"]["baseline_config"]).resolve()
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    _require(checkpoint.get("config_identity") == baseline_config
             and source_endpoint.get("branch") == "C" and source_endpoint.get("global_step") == SOURCE_STEP
             and source_endpoint.get("measurement_complete") is True
             and source_report.get("task_id") == decoder_ref.SOURCE_TASK_ID
             and source_report.get("measurement_complete") is True
             and source_report.get("run_id") == checkpoint.get("run_id"),
             "C endpoint report, checkpoint and source configuration are not aligned")
    terminal_path = source_dir / "terminal_model.pt"
    if terminal_path.is_file():
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(terminal.get("branch") == "C" and terminal.get("global_step") == SOURCE_STEP
                 and base._assert_close(checkpoint["model"], terminal.get("model", {}),
                                        "source_terminal_model") == 0.0,
                 "C source checkpoint differs from its saved terminal model")

    cohort = torch.load(cohort_path, map_location="cpu", weights_only=False)
    _require(cohort.get("schema") == "lightgen_small_decoder_native_gradient_fixed_g4_inputs.v1"
             and cohort.get("task_id") == decoder_ref.TASK_ID
             and cohort.get("source_task_id") == decoder_ref.GENERATION_TASK_ID
             and cohort.get("model") == "C_seed_2026101501"
             and cohort.get("method") == "G4" and cohort.get("stream") == 0
             and cohort.get("global_step") == SOURCE_STEP,
             "fixed D G4 input wrapper task, source, stream or endpoint changed")
    coefficients = cohort.get("ols_coefficients")
    requested = cohort.get("requested_labels")
    components = cohort.get("component_ids")
    proposals = cohort.get("proposal_ids")
    _require(isinstance(coefficients, torch.Tensor) and tuple(coefficients.shape) == (10_000, 100)
             and coefficients.is_complex() and torch.isfinite(coefficients).all().item()
             and isinstance(requested, torch.Tensor) and tuple(requested.shape) == (10_000,)
             and torch.equal(torch.bincount(requested.long(), minlength=10), torch.full((10,), 1000))
             and isinstance(components, torch.Tensor) and tuple(components.shape) == (10_000,)
             and bool(((components >= 0) & (components < 4)).all())
             and isinstance(proposals, torch.Tensor) and tuple(proposals.shape) == (10_000,),
             "fixed D G4 coefficients or class/component/proposal population is incomplete")
    for label in range(10):
        mask = requested.long() == label
        _require(proposals[mask].unique().numel() == 1000,
                 f"fixed D G4 class {label} has duplicate proposal IDs")
    source = {"checkpoint": checkpoint, "endpoint_summary": source_endpoint,
              "task_report": source_report, "cohort": cohort,
              "source_config": baseline_config, "source_dir": source_dir}

    review_key_path = reference_dir / "blind_review_key.json"
    review_report_path = reference_dir / "final_report.json"
    for path in (review_key_path, review_report_path):
        _require(path.is_file(), f"fixed source readability review is missing: {path}", FileNotFoundError)
    review_key = json.loads(review_key_path.read_text(encoding="utf-8"))
    review_report = json.loads(review_report_path.read_text(encoding="utf-8"))
    _require(review_key.get("task_id") == "LIGHTGEN_SMALL_G1_G4_NATIVE_READABILITY_V1"
             and review_key.get("review_id") == config["readability_reference"]["review_id"]
             and review_report.get("human_review", {}).get("review_id") == review_key.get("review_id")
             and review_report.get("measurement_complete") is True
             and review_report.get("human_review", {}).get("quality_checks", {}).get(
                 "duplicate_consistency", {}).get("count", 0) >= 18,
             "fixed source readability review identity or quality checks changed")
    reviewed: dict[str, list[int]] = {str(label): [] for label in range(10)}
    for card in review_key.get("cards", []):
        if card.get("kind") == "generated" and card.get("method") == "G4" and card.get("stream") == 0:
            reviewed[str(int(card["requested_label"]))].append(int(card["proposal_id"]))
    _require(all(len(reviewed[str(label)]) == 10
                 and len(set(reviewed[str(label)])) == 10 for label in range(10)),
             "source G4 review does not identify its fixed ten proposals per class")
    reviewed = {label: sorted(proposals) for label, proposals in reviewed.items()}
    control_machine_path = control_dir / "machine_report.json"
    control_checkpoint_path = control_dir / config["control"]["checkpoint"]
    control_endpoint_path = control_dir / "D0" / "endpoint_summary.json"
    control_rows_path = control_dir / "D0" / "reconstruction_metrics_dev.csv"
    control_gen_rows_path = control_dir / "D0" / "generation_metrics_per_image.csv"
    control_gen_path = control_dir / "D0" / "generated_cohort.pt"
    for path in (control_machine_path, control_checkpoint_path, control_endpoint_path,
                 control_rows_path, control_gen_rows_path, control_gen_path):
        _require(path.is_file(), f"fixed D0 evidence is missing: {path}", FileNotFoundError)
    machine = json.loads(control_machine_path.read_text(encoding="utf-8"))
    endpoint = json.loads(control_endpoint_path.read_text(encoding="utf-8"))
    control_cp = torch.load(control_checkpoint_path, map_location="cpu", weights_only=False)
    _require(machine.get("task_id") == decoder_ref.TASK_ID and machine.get("measurement_complete") is True
             and endpoint.get("branch") == "D0" and endpoint.get("global_step") == 50_000
             and endpoint.get("measurement_complete") is True
             and control_cp.get("schema") == "lightgen_small_decoder_native_gradient_checkpoint.v1"
             and control_cp.get("task_id") == decoder_ref.TASK_ID and control_cp.get("branch") == "D0"
             and control_cp.get("source_run_id") == source["checkpoint"]["run_id"]
             and control_cp.get("source_checkpoint_step") == SOURCE_STEP
             and control_cp.get("completed_updates") == 5_000
             and control_cp.get("completed_global_step") == ENDPOINT_STEP
             and control_cp.get("optimizer_steps") == {"encoder_adam_step": SOURCE_STEP,
                                                        "decoder_adam_step": ENDPOINT_STEP},
             "D0 endpoint identity, completion, or optimizer history changed")
    _require(control_cp.get("model_identity") == source["checkpoint"].get("model_identity")
             and tuple(control_cp["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(control_cp["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100),
             "D0 and C model identities or phase geometry differ")
    c_model = source["checkpoint"]["model"]
    d_model = control_cp["model"]
    _require(set(c_model) == set(d_model), "D0/C state dictionaries have different model keys")
    for name in c_model:
        if name != "runtime.generator.phase":
            _require(torch.equal(c_model[name], d_model[name]),
                     f"D0 changed a non-Decoder source value: {name}")
    _require(control_cp.get("schedule_start_global_step") == SOURCE_STEP
             and tuple(control_cp.get("schedule", torch.empty(0)).shape) == (UPDATES, BATCH_SIZE),
             "D0 fixed 45,000-50,000 schedule segment is missing")
    dev_rows = _read_csv(control_rows_path)
    gen_rows = _read_csv(control_gen_rows_path)
    d0_generation = torch.load(control_gen_path, map_location="cpu", weights_only=False)
    _require(len(dev_rows) == 5_000 and len({int(row["row"]) for row in dev_rows}) == 5_000
             and len(gen_rows) == 10_000 and len({int(row["cohort_row"]) for row in gen_rows}) == 10_000
             and tuple(d0_generation.get("images", torch.empty(0)).shape) == (10_000, 1, 28, 28)
             and tuple(d0_generation.get("ed_logits", torch.empty(0)).shape) == (10_000, 10)
             and tuple(d0_generation.get("ec_logits", torch.empty(0)).shape) == (10_000, 10)
             and torch.equal(d0_generation.get("ols_coefficients"), coefficients)
             and torch.equal(d0_generation.get("requested_labels"), requested)
             and torch.equal(d0_generation.get("component_ids"), components)
             and torch.equal(d0_generation.get("proposal_ids"), proposals),
             "D0 saved DEV or G4 evidence is incomplete")
    selection_path = control_dir / "g4_review_selection.json"
    _require(selection_path.is_file(), "D0 fixed native G4 selection is missing", FileNotFoundError)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection = selection_payload.get("selection")
    _require(selection_payload.get("task_id") == decoder_ref.TASK_ID
             and isinstance(selection, list) and len(selection) == 100
             and len({int(item["cohort_row"]) for item in selection}) == 100
             and selection_payload.get("previous_review_excluded") == reviewed,
             "D0 native G4 display selection changed")
    for item in selection:
        row = int(item["cohort_row"])
        label = int(item["requested_label"])
        _require(0 <= row < 10_000 and int(requested[row]) == label
                 and int(proposals[row]) == int(item["proposal_id"])
                 and int(proposals[row]) not in reviewed[str(label)],
                 "fixed D0 native G4 selection is misaligned or overlaps prior reviewed proposals")
    _require(torch.equal(d0_generation.get("native_capture_indices"),
                         torch.tensor(sorted(int(item["cohort_row"]) for item in selection), dtype=torch.long)),
             "D0 saved native display samples do not match the fixed G4 selection")
    for index, row in enumerate(gen_rows):
        _require(int(row["cohort_row"]) == index and int(row["requested_label"]) == int(requested[index])
                 and int(row["component_id"]) == int(components[index])
                 and int(row["proposal_id"]) == int(proposals[index]),
                 f"D0 G4 metrics identity is misaligned at cohort row {index}")
    return {
        "source": source, "previous_config": previous_config,
        "control_checkpoint": control_cp, "control_endpoint": endpoint,
        "control_generation": d0_generation, "control_dev_rows": dev_rows,
        "control_gen_rows": gen_rows, "selection": selection,
        "control_machine": machine, "control_checkpoint_path": control_checkpoint_path,
    }


def _optimizer_steps(optimizer: torch.optim.Adam, encoder: torch.nn.Parameter,
                     decoder: torch.nn.Parameter) -> tuple[int, int]:
    result = []
    for parameter in (encoder, decoder):
        state = optimizer.state.get(parameter)
        _require(isinstance(state, dict) and "step" in state, "Adam phase state is missing")
        result.append(int(torch.as_tensor(state["step"]).item()))
    return result[0], result[1]


def _optimizer_steps_by_phase(optimizer: torch.optim.Adam, encoder: torch.nn.Parameter,
                              decoder: torch.nn.Parameter) -> dict[str, int]:
    encoder_step, decoder_step = _optimizer_steps(optimizer, encoder, decoder)
    return {"encoder_adam_step": encoder_step, "decoder_adam_step": decoder_step}


def _preflight_recovery_input(
    config: dict[str, Any], source_checkpoint: dict[str, Any], model: torch.nn.Module,
    optimizer: torch.optim.Adam, encoder: torch.nn.Parameter, decoder: torch.nn.Parameter,
    rows: torch.Tensor, cpu_rng: torch.Tensor, cuda_rng: list[torch.Tensor],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID, "completed_updates": 3, "config_identity": config,
        "source_checkpoint": source_checkpoint, "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": _optimizer_steps_by_phase(optimizer, encoder, decoder),
        "rows": rows.detach().cpu(), "cuda_rng": [state.cpu() for state in cuda_rng],
        "cpu_rng": cpu_rng.cpu(),
    }


def _write_preflight_update_log(staging_dir: Path, updates: list[dict[str, Any]]) -> None:
    path = staging_dir / "engineering_update_log.json"
    payload = {"schema": "lightgen_joint_preflight_updates.v1", "updates": updates}
    atomic_write_json(path, payload)
    _require(json.loads(path.read_text(encoding="utf-8")) == payload,
             "joint preflight update log failed atomic readback")


def _record_preflight_call(staging_dir: Path, updates: list[dict[str, Any]],
                           entry: dict[str, Any]) -> None:
    updates.append(entry)
    _write_preflight_update_log(staging_dir, updates)


def _new_joint_model_optimizer(
    source: dict[str, Any], previous_config: dict[str, Any], device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Adam]:
    checkpoint = source["checkpoint"]
    model = base._model("small", 2048, device, previous_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder, decoder = base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is encoder or parameter is decoder)
    _require(encoder.requires_grad and decoder.requires_grad
             and all(parameter.requires_grad == (parameter is encoder or parameter is decoder)
                     for parameter in model.parameters()),
             "joint update must train exactly both phase groups")
    optimizer = torch.optim.Adam((encoder, decoder), lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    _require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == 2
             and optimizer.param_groups[0]["params"][0] is encoder
             and optimizer.param_groups[0]["params"][1] is decoder
             and _optimizer_steps(optimizer, encoder, decoder) == (SOURCE_STEP, SOURCE_STEP),
             "source Adam mapping or 45,000-step state does not match the two phase groups")
    return model, optimizer


def _joint_update(
    model: torch.nn.Module, optimizer: torch.optim.Adam, pixels: torch.Tensor,
    rows: torch.Tensor, device: torch.device, source: dict[str, Any],
) -> dict[str, float]:
    model.train()
    encoder, decoder = base._phase_groups(model)
    _require(encoder.requires_grad and decoder.requires_grad and rows.shape == (BATCH_SIZE,),
             "joint phase groups or logical batch changed")
    before_encoder, before_decoder = encoder.detach().clone(), decoder.detach().clone()
    steps_before = _optimizer_steps(optimizer, encoder, decoder)
    optimizer.zero_grad(set_to_none=True)
    cp = source["checkpoint"]
    parts, ssim, _native_gradient, result = decoder_ref._loss_parts(
        model, pixels, rows, device, cp["scale"].to(device),
        cp["power_reference"].to(device), cp["camera_gain"].to(device))
    path_witness = _validate_fullpath_training_forward(
        model, pixels, rows, source, result, parts, cp, ssim,
    )
    canonical_rec, canonical_ssim = _expected_cmos_terms(result, pixels, rows, cp)
    total = decoder_ref._objective(parts, ssim, None, cp["weights"],
                                   float(cp["structural_weight"]), None)
    _require(torch.isfinite(total).item(), "joint BOGT+SSIM objective is nonfinite")
    for name, actual_term, expected_term in (
        ("reconstruction", parts["rec"], canonical_rec),
        ("SSIM", ssim, canonical_ssim),
    ):
        _require(actual_term.requires_grad and expected_term.requires_grad,
                 f"{name} CMOS loss is detached before the optical phase update")
        actual_gradients = torch.autograd.grad(
            actual_term, (encoder, decoder), retain_graph=True, allow_unused=True,
        )
        expected_gradients = torch.autograd.grad(
            expected_term, (encoder, decoder), retain_graph=True, allow_unused=True,
        )
        for group_name, actual_gradient, expected_gradient in zip(
            ("Encoder", "Generator"), actual_gradients, expected_gradients, strict=True,
        ):
            if expected_gradient is None:
                _require(actual_gradient is None or not bool((actual_gradient != 0).any()),
                         f"{name} CMOS gradient unexpectedly reaches {group_name}")
            else:
                _require(actual_gradient is not None
                         and torch.allclose(actual_gradient, expected_gradient,
                                            rtol=1.0e-5, atol=1.0e-8),
                         f"{name} CMOS loss gradient differs from the same-forward reference for {group_name}")
    canonical_cmos = (
        float(cp["weights"]["rec"]) * canonical_rec
        + float(cp["structural_weight"]) * canonical_ssim
    )
    canonical_gradients = torch.autograd.grad(
        canonical_cmos, (encoder, decoder), retain_graph=True, allow_unused=True,
    )
    _require(all(gradient is not None and torch.isfinite(gradient).all().item()
                 and bool((gradient.abs().sum() > 0)) for gradient in canonical_gradients),
             "same-forward CMOS objective must reach both optical phase groups")
    total.backward()
    for name, parameter in (("Encoder", encoder), ("Decoder", decoder)):
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                 and bool((parameter.grad.abs().sum() > 0)), f"{name} joint gradient is missing or zero")
    for parameter in model.parameters():
        if parameter is not encoder and parameter is not decoder:
            _require(parameter.grad is None, "a non-phase model parameter received a training gradient")
    norms = [float(torch.linalg.vector_norm(parameter.grad.detach().double()).cpu())
             for parameter in (encoder, decoder)]
    optimizer.step()
    _require(not torch.equal(before_encoder, encoder) and not torch.equal(before_decoder, decoder),
             "both declared phase groups must update on every science step")
    steps_after = _optimizer_steps(optimizer, encoder, decoder)
    _require(steps_after == (steps_before[0] + 1, steps_before[1] + 1),
             "joint Adam steps did not advance together")
    _save_training_path_witness_if_selected(
        source, path_witness, steps_before=steps_before, steps_after=steps_after,
    )
    return {
        "dist": float(parts["dist"].detach()), "rec": float(parts["rec"].detach()),
        "op": float(parts["op"].detach()), "ssim_loss": float(ssim.detach()),
        "total": float(total.detach()), "encoder_gradient_norm": norms[0],
        "decoder_gradient_norm": norms[1],
        "mean_plane_power_w": float(result["plane_powers_w"].detach().double().mean()),
        "encoder_adam_step": steps_after[0], "decoder_adam_step": steps_after[1],
    }


def _expected_cmos_terms(
    result: dict[str, Any], pixels: torch.Tensor, rows: torch.Tensor,
    checkpoint: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = result["physical_w_per_pixel"]
    gain = checkpoint["camera_gain"].to(device=raw.device, dtype=raw.dtype)
    target = image_batch(pixels, rows, "small", raw.device)
    rec = (raw * gain - target).square().mean()
    target28 = pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(
        device=raw.device, dtype=torch.float32,
    ) / 255.0
    ssim = (1.0 - ssim_batch(resize_view(raw * gain), target28)).mean()
    return rec, ssim


def _validate_fullpath_training_forward(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    source: dict[str, Any],
    result: dict[str, Any],
    parts: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    ssim: torch.Tensor,
) -> dict[str, Any]:
    """Reject loss batches that do not come from the bound image-to-CMOS path."""
    labels = source.get("training_labels")
    train_indices = source.get("training_indices")
    training_pixels = source.get("training_pixels")
    _require(torch.is_tensor(labels) and torch.is_tensor(train_indices)
             and torch.is_tensor(training_pixels)
             and rows.ndim == 1 and rows.numel() == BATCH_SIZE
             and rows.unique().numel() == BATCH_SIZE,
             "full-path training requires unique source TRAIN rows and their bound labels")
    row_cpu = rows.detach().long().cpu()
    labels_cpu = labels.detach().long().cpu()
    train_cpu = train_indices.detach().long().cpu()
    _require(bool(torch.isin(row_cpu, train_cpu).all())
             and torch.equal(labels_cpu.index_select(0, row_cpu), torch.arange(BATCH_SIZE)),
             "full-path training batch rows or labels are not the registered class-balanced TRAIN batch")
    target = image_batch(pixels, rows, "small", next(model.parameters()).device)
    source_target = image_batch(
        training_pixels, rows, "small", next(model.parameters()).device,
    )
    _require(torch.equal(target, source_target),
             "training image tensor is not bound to the registered source TRAIN rows")
    image_input = result.get("image_input")
    _require(image_input is not None
             and torch.is_tensor(getattr(image_input, "image", None))
             and torch.is_tensor(getattr(image_input, "slm_intensity", None))
             and torch.is_tensor(getattr(image_input, "slm_amplitude", None))
             and getattr(image_input, "slm_field", None) is not None,
             "training loss did not consume an actual amplitude-SLM input record")
    _require(torch.equal(image_input.image, target)
             and image_input.slm_intensity.ndim == 4
             and image_input.slm_amplitude.shape == image_input.slm_intensity.shape
             and torch.is_complex(image_input.slm_field.tensor)
             and image_input.slm_field.tensor.shape == image_input.slm_amplitude.shape
             and torch.isfinite(image_input.slm_intensity).all().item()
             and torch.isfinite(image_input.slm_amplitude).all().item()
             and bool(((image_input.slm_amplitude >= 0.0)
                       & (image_input.slm_amplitude <= 1.0)).all())
             and torch.allclose(image_input.slm_amplitude.square(),
                                image_input.slm_intensity, rtol=1.0e-6, atol=1.0e-7),
             "training SLM command is missing, illegal, or not the square root of the prepared intensity")
    expected_intensity = F.interpolate(
        target, size=model.runtime.slm_plane.shape, mode="bilinear", align_corners=False,
    ).clamp(0.0, 1.0)
    expected_slm_field = torch.complex(
        image_input.slm_amplitude, torch.zeros_like(image_input.slm_amplitude),
    )
    physical_optics = getattr(model.runtime, "physical_optics", None)
    if physical_optics is not None:
        dy, dx = model.runtime.slm_plane.pixel_pitch_m
        pixel_area = float(dy) * float(dx)
        input_amplitude = math.sqrt(
            physical_optics.input_power_w
            / (model.runtime.slm_plane.shape[0] * model.runtime.slm_plane.shape[1] * pixel_area)
        )
        expected_slm_field = expected_slm_field * input_amplitude
    _require(torch.allclose(image_input.slm_intensity, expected_intensity,
                            rtol=0.0, atol=1.0e-12),
             "training SLM intensity is not the registered native image mapping")
    _require(torch.allclose(image_input.slm_field.tensor, expected_slm_field,
                            rtol=1.0e-6, atol=1.0e-7),
             "training SLM field does not carry the actual amplitude command and source illumination")
    encoder_fields = result.get("encoder_fields")
    generator_fields = result.get("decoder_fields")
    ols_input = result.get("ols_input_field")
    generator_entry = result.get("generator_entry_field")
    generator_exit = result.get("generator_exit_field")
    coefficients = result.get("ols_coefficients")
    plane_powers = result.get("plane_powers_w")
    raw_w_m2 = result.get("raw_intensity_w_per_m2")
    w_per_pixel = result.get("physical_w_per_pixel")
    _require(isinstance(encoder_fields, tuple) and len(encoder_fields) == 2
             and isinstance(generator_fields, tuple) and len(generator_fields) == 2
             and ols_input is not None and generator_entry is not None and generator_exit is not None
             and torch.is_complex(getattr(ols_input, "tensor", torch.empty(0)))
             and torch.is_tensor(coefficients) and torch.is_complex(coefficients)
             and tuple(coefficients.shape) == (BATCH_SIZE, 100)
             and tuple(plane_powers.shape) == (BATCH_SIZE, 5)
             and tuple(raw_w_m2.shape) == tuple(w_per_pixel.shape) == (BATCH_SIZE, 1, 200, 200)
             and all(torch.isfinite(field.tensor.real).all().item()
                     and torch.isfinite(field.tensor.imag).all().item()
                     for field in (*encoder_fields, *generator_fields))
             and torch.isfinite(coefficients.real).all().item()
             and torch.isfinite(coefficients.imag).all().item()
             and torch.isfinite(plane_powers).all().item()
             and torch.isfinite(raw_w_m2).all().item()
             and torch.isfinite(w_per_pixel).all().item(),
             "training loss is missing Encoder, actual OLS, Generator, plane-power, or CMOS path records")
    _require(torch.equal(encoder_fields[-1].tensor, ols_input.tensor)
             and torch.equal(generator_fields[-1].tensor, generator_exit.tensor),
             "saved layer traces are disconnected from the actual OLS or Generator boundary")
    actual_state = model.runtime.ols.project_state(ols_input)
    _require(actual_state.basis_id == model.runtime.ols.basis_id
             and torch.allclose(actual_state.coefficients, coefficients,
                                rtol=1.0e-12, atol=1.0e-14),
             "training OLS coefficients do not come from the actual Encoder output field")
    camera_pitch = model.runtime.camera_plane.pixel_pitch_m
    pixel_area = float(camera_pitch[0]) * float(camera_pitch[1])
    _require(torch.allclose(raw_w_m2 * pixel_area, w_per_pixel,
                            rtol=1.0e-12, atol=1.0e-14),
             "training CMOS output units do not match raw intensity times physical pixel area")
    rec_from_same_forward = (
        w_per_pixel * checkpoint["camera_gain"].to(device=w_per_pixel.device,
                                                   dtype=w_per_pixel.dtype)
        - target
    ).square().mean()
    _require(torch.allclose(parts["rec"], rec_from_same_forward, rtol=1.0e-7, atol=1.0e-10),
             "training reconstruction loss is disconnected from its full-path CMOS output")
    expected_parts = base._loss_parts_from_result(
        result, target,
        checkpoint["scale"].to(device=target.device),
        checkpoint["power_reference"].to(device=target.device),
        checkpoint["camera_gain"].to(device=target.device),
    )
    _require(all(torch.allclose(parts[name], expected_parts[name], rtol=1.0e-7, atol=1.0e-10)
                 for name in ("dist", "rec", "op")),
             "training BOGT terms do not consume the same full-path OLS/CMOS forward")
    target28 = pixels.index_select(0, row_cpu).unsqueeze(1).to(
        device=w_per_pixel.device, dtype=torch.float32,
    ) / 255.0
    view = resize_view(w_per_pixel * checkpoint["camera_gain"].to(
        device=w_per_pixel.device, dtype=w_per_pixel.dtype,
    ))
    expected_ssim = (1.0 - ssim_batch(view, target28)).mean()
    _require(torch.allclose(ssim, expected_ssim, rtol=1.0e-7, atol=1.0e-10),
             "training SSIM term does not consume the same full-path CMOS output")
    return {
        "source_rows": row_cpu,
        "source_labels": labels_cpu.index_select(0, row_cpu),
        "slm_intensity": image_input.slm_intensity.detach().cpu().clone(),
        "slm_amplitude": image_input.slm_amplitude.detach().cpu().clone(),
        "ols_coefficients": coefficients.detach().cpu().clone(),
        "plane_powers_w": plane_powers.detach().double().cpu().clone(),
        "raw_cmos_w_per_m2": raw_w_m2.detach().double().cpu().clone(),
        "path": "TRAIN image -> native amplitude SLM -> 2-layer Encoder -> measured Gaussian OLS -> 2-layer Generator -> raw CMOS",
    }


def _save_training_path_witness_if_selected(
    source: dict[str, Any], witness: dict[str, Any], *,
    steps_before: tuple[int, int], steps_after: tuple[int, int],
) -> None:
    witness_dir = source.get("training_witness_dir")
    update = source.get("active_training_update")
    selected = source.get("training_capture_updates", ())
    if witness_dir is None or update not in selected:
        return
    target = Path(witness_dir) / f"training_path_witness_update_{int(update):05d}.pt"
    source_step = int(source["checkpoint"].get("completed_global_step", SOURCE_STEP))
    payload = {
        "schema": "lightgen_fullpath_training_witness.v1",
        "source_training_run_id": source["checkpoint"].get("run_id"),
        "forward_global_step": source_step + int(update) - 1,
        "completed_global_step": source_step + int(update),
        "completed_update": int(update),
        "optimizer_steps_before": {
            "encoder": steps_before[0], "generator": steps_before[1],
        },
        "optimizer_steps_after": {
            "encoder": steps_after[0], "generator": steps_after[1],
        },
        **witness,
    }
    if not target.exists():
        atomic_torch_save_new(target, payload)
    else:
        existing = torch.load(target, map_location="cpu", weights_only=False)
        def witness_tensor_matches(key: str) -> bool:
            left, right = existing.get(key), payload[key]
            if not torch.is_tensor(left) or not torch.is_tensor(right):
                return False
            if left.is_floating_point() or left.is_complex():
                return left.shape == right.shape and left.dtype == right.dtype and torch.allclose(
                    left, right, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL,
                )
            return torch.equal(left, right)

        _require(existing.get("schema") == payload["schema"]
                 and existing.get("source_training_run_id") == payload["source_training_run_id"]
                 and existing.get("forward_global_step") == payload["forward_global_step"]
                 and existing.get("completed_global_step") == payload["completed_global_step"]
                 and existing.get("completed_update") == payload["completed_update"]
                 and existing.get("optimizer_steps_before") == payload["optimizer_steps_before"]
                 and existing.get("optimizer_steps_after") == payload["optimizer_steps_after"]
                 and all(witness_tensor_matches(key) for key in (
                     "source_rows", "source_labels", "slm_intensity", "slm_amplitude",
                     "ols_coefficients", "plane_powers_w", "raw_cmos_w_per_m2",
                 )),
                 "restored training witness conflicts with the actual resumed batch")


def _joint_direction_check(
    model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor,
    device: torch.device, source: dict[str, Any], group_name: str,
    epsilon: float = 1e-2,
) -> dict[str, float]:
    encoder, decoder = base._phase_groups(model)
    parameter = encoder if group_name == "encoder" else decoder

    def objective() -> torch.Tensor:
        cp = source["checkpoint"]
        parts, ssim, _gradient, _result = decoder_ref._loss_parts(
            model, pixels, rows, device, cp["scale"].to(device),
            cp["power_reference"].to(device), cp["camera_gain"].to(device))
        return decoder_ref._objective(parts, ssim, None, cp["weights"],
                                      float(cp["structural_weight"]), None)

    value = objective()
    gradient = torch.autograd.grad(value, parameter)[0]
    _require(torch.isfinite(gradient).all().item() and float(torch.linalg.vector_norm(gradient)) > 0.0,
             f"{group_name} direction gradient is invalid")
    norm = torch.linalg.vector_norm(gradient.detach().double())
    direction = (gradient.detach().double() / norm).to(parameter.dtype)
    original = parameter.detach().clone()
    plus_state, minus_state = original + epsilon * direction, original - epsilon * direction
    representable = float(((((plus_state.double() - original.double())
                            + (original.double() - minus_state.double())) * direction.double()).sum()) * 0.5)
    try:
        with torch.no_grad():
            parameter.copy_(plus_state)
        plus = float(objective().detach())
        with torch.no_grad():
            parameter.copy_(minus_state)
        minus = float(objective().detach())
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    analytic = float((gradient.detach().double() * direction.double()).sum())
    finite = (plus - minus) / (2 * representable)
    relative = abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-7)
    _require(representable > 0.0 and math.isfinite(analytic) and math.isfinite(finite)
             and analytic * finite > 0.0 and relative <= 0.05,
             f"{group_name} finite difference failed: autograd={analytic}, central={finite}, rel={relative}")
    return {"group": group_name, "analytic": analytic, "central_difference": finite,
            "relative_error": relative, "requested_rad": epsilon,
            "representable_half_step_rad": representable}


def _restore_joint_state(
    payload: dict[str, Any], previous_config: dict[str, Any], device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = base._model("small", 2048, device, previous_config)
    model.load_state_dict(payload["model"], strict=True)
    encoder, decoder = base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is encoder or parameter is decoder)
    optimizer = torch.optim.Adam((encoder, decoder), lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    recorded_steps = payload.get("optimizer_steps")
    _require(isinstance(recorded_steps, dict)
             and set(recorded_steps) == {"encoder_adam_step", "decoder_adam_step"}
             and all(type(recorded_steps[name]) is int for name in recorded_steps)
             and _optimizer_steps(optimizer, encoder, decoder) == (
                 recorded_steps["encoder_adam_step"], recorded_steps["decoder_adam_step"]),
             "recovery Adam step-to-phase mapping changed")
    return model, optimizer


def _recovery_worker(staging_dir: Path, config_path: Path, previous_config_path: Path) -> None:
    config = _read_config(config_path)
    previous_config = decoder_ref._read_config(previous_config_path)
    payload = torch.load(staging_dir / "recovery_input.pt", map_location="cpu", weights_only=False)
    _require(payload.get("task_id") == TASK_ID and payload.get("completed_updates") == 3
             and payload.get("config_identity") == config,
             "joint-update recovery worker received a different task or segment")
    _require(torch.cuda.is_available(), "joint-update recovery worker requires CUDA")
    device = torch.device("cuda")
    model, optimizer = _restore_joint_state(payload, previous_config, device)
    torch.cuda.set_rng_state_all([item.cpu() for item in payload["cuda_rng"]])
    torch.random.set_rng_state(payload["cpu_rng"].cpu())
    pixels, labels, train_indices, _dev = base._load_data(ROOT / config["data"]["training_file"])
    source = {"checkpoint": payload["source_checkpoint"],
              "training_labels": labels, "training_indices": train_indices,
              "training_pixels": pixels}
    log_path = staging_dir / "engineering_update_log.json"
    log_payload = json.loads(log_path.read_text(encoding="utf-8"))
    updates = log_payload["updates"]
    _require(len(updates) == 4 and all(item.get("status") == "COMPLETED" for item in updates),
             "joint recovery worker requires four durably completed preflight calls")
    call = {"call_number": 5, "branch": "independent_process_fourth", "status": "CALL_STARTED",
            "schedule_cursor": SOURCE_STEP + 3, "global_step_before": SOURCE_STEP + 3,
            "sample_rows": [int(value) for value in payload["rows"][3].tolist()],
            "device": torch.cuda.get_device_name(device)}
    _record_preflight_call(staging_dir, updates, call)
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    loss = _joint_update(model, optimizer, pixels, payload["rows"][3], device, source)
    end_event.record()
    torch.cuda.synchronize(device)
    encoder, decoder = base._phase_groups(model)
    result = {
        "task_id": TASK_ID, "completed_updates": 4,
        "origin_checkpoint_run_id": payload["source_checkpoint"].get("run_id"),
        "completed_global_step": SOURCE_STEP + 4, "schedule_cursor": SOURCE_STEP + 3,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": _optimizer_steps_by_phase(optimizer, encoder, decoder), "loss": loss,
        "rows": payload["rows"][3].detach().cpu().clone(),
        "cpu_rng": torch.random.get_rng_state().cpu().clone(),
        "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "cuda": {"device": torch.cuda.get_device_name(device),
                 "elapsed_event_ms": float(start_event.elapsed_time(end_event)),
                 "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved())},
    }
    atomic_torch_save_new(staging_dir / "recovery_fourth_update.pt", result)
    call.update({"status": "COMPLETED", "global_step_after": SOURCE_STEP + 4,
                 "optimizer_steps": result["optimizer_steps"], "loss": loss,
                 "elapsed_event_ms": result["cuda"]["elapsed_event_ms"],
                 "peak_reserved_vram_bytes": result["cuda"]["peak_reserved_vram_bytes"]})
    _write_preflight_update_log(staging_dir, updates)
    print(json.dumps({"status": "PASSED", "worker": "independent_process_recovery"}), flush=True)


def _run_preflight(
    args: argparse.Namespace, config: dict[str, Any], config_path: Path,
    inputs: dict[str, Any], pixels: torch.Tensor, labels: torch.Tensor,
    train_indices: torch.Tensor, dev_indices: torch.Tensor, schedule: torch.Tensor,
    device: torch.device, evaluator: torch.nn.Module, ec_evaluator: torch.nn.Module,
    staging_dir: Path,
) -> tuple[dict[str, Any], ResearchRunControl, PreflightReceipt, ResearchRunIdentity]:
    started = time.perf_counter()
    source = inputs["source"]
    previous_config = inputs["previous_config"]
    d0 = inputs["control_checkpoint"]
    _require(len(train_indices) == 55_000 and len(dev_indices) == 5_000
             and int(config["data"]["official_test_reads"]) == 0,
             "registered TRAIN/DEV roles changed or TEST was requested")
    _require(torch.equal(schedule[35_000:45_000], source["checkpoint"]["schedule"])
             and torch.equal(schedule[45_000:50_000], d0["schedule"]),
             "C and D0 sample fragments do not match the continuous 35k-50k schedule")
    _require(torch.equal(labels[schedule[45_000:50_000]], torch.arange(10).expand(UPDATES, 10)),
             "joint continuation rows do not contain one TRAIN example per class")

    d0_model = base._model("small", 2048, device, previous_config)
    d0_model.load_state_dict(d0["model"], strict=True)
    d0_model.eval()
    d0_dev = torch.load(inputs["control_checkpoint_path"].parent / "dev_display_examples.pt",
                        map_location="cpu", weights_only=False)
    _require(d0_dev.get("task_id") == decoder_ref.TASK_ID and d0_dev.get("branch") == "D0"
             and tuple(d0_dev.get("rows", torch.empty(0)).shape) == (20,),
             "D0 fixed DEV display evidence is incomplete")
    dev_by_label = {}
    for index, label in enumerate(d0_dev["labels"].tolist()):
        dev_by_label.setdefault(int(label), index)
    _require(set(dev_by_label) == set(range(10)), "D0 DEV replay does not cover all ten classes")
    d0_gen = inputs["control_generation"]
    requested = d0_gen["requested_labels"].long()
    first_gen = [int(torch.nonzero(requested == label, as_tuple=False)[0]) for label in range(10)]
    ed_state, ec_state = base._state_snapshot(evaluator), base._state_snapshot(ec_evaluator)
    d0_replay: dict[str, Any] = {"dev": [], "generation": []}
    with torch.inference_mode():
        dev_entries = [(label, dev_by_label[label]) for label in range(10)]
        dev_rows = torch.tensor([int(d0_dev["rows"][index]) for _label, index in dev_entries],
                                dtype=torch.long)
        dev_labels = torch.tensor([label for label, _index in dev_entries], dtype=torch.long, device=device)
        image = image_batch(pixels, dev_rows, "small", device)
        result = base._forward(d0_model, image)
        gained = result["physical_w_per_pixel"] * inputs["source"]["checkpoint"]["camera_gain"].to(
            device=device, dtype=torch.float32)
        view = resize_view(gained)
        saved = d0_dev["view28"][[index for _label, index in dev_entries]].to(device)
        _require(torch.allclose(view, saved, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL),
                 "D0 saved DEV view replay failed in production batch10")
        ed_logits = evaluator(view)
        ec_logits = decoder_ref._ec_logits(ec_evaluator, view)
        saved_ed_logits = evaluator(saved)
        saved_ec_logits = decoder_ref._ec_logits(ec_evaluator, saved)
        _require(torch.allclose(ed_logits, saved_ed_logits, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
                 and torch.allclose(ec_logits, saved_ec_logits, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
                 and torch.equal(ed_logits.argmax(1), saved_ed_logits.argmax(1))
                 and torch.equal(ec_logits.argmax(1), saved_ec_logits.argmax(1)),
                 "D0 DEV replay logits from saved views exceed the fixed numerical tolerance")
        ed_pred, ed_margin, _ed_top, _ = decoder_ref._logit_summary(ed_logits, dev_labels)
        ec_pred, ec_margin, _ec_top, _ = decoder_ref._logit_summary(ec_logits, dev_labels)
        for offset, (label, index) in enumerate(dev_entries):
            row = int(d0_dev["rows"][index])
            row_summary = next(item for item in inputs["control_dev_rows"] if int(item["row"]) == row)
            _require(int(ed_pred[offset]) == int(row_summary["ed_prediction"])
                     and int(ec_pred[offset]) == int(row_summary["ec_prediction"])
                     and math.isclose(float(ed_margin[offset]), float(row_summary["ed_requested_margin"]),
                                      rel_tol=RECOVERY_RTOL, abs_tol=RECOVERY_ATOL)
                     and math.isclose(float(ec_margin[offset]), float(row_summary["ec_requested_margin"]),
                                      rel_tol=RECOVERY_RTOL, abs_tol=RECOVERY_ATOL),
                     f"D0 batch10 DEV prediction or identity margin replay failed for row {row}: "
                     f"E_D={(int(ed_pred[offset]), float(ed_margin[offset]))} vs "
                     f"{(int(row_summary['ed_prediction']), float(row_summary['ed_requested_margin']))}; "
                     f"E_C={(int(ec_pred[offset]), float(ec_margin[offset]))} vs "
                     f"{(int(row_summary['ec_prediction']), float(row_summary['ec_requested_margin']))}")
            d0_replay["dev"].append({"class": label, "row": row,
                                     "view_max_abs_error": float((view[offset:offset + 1]
                                                                   - saved[offset:offset + 1]).abs().max()),
                                     "ed_logit_max_abs_error": float((ed_logits[offset:offset + 1]
                                                                       - saved_ed_logits[offset:offset + 1]).abs().max()),
                                     "ec_logit_max_abs_error": float((ec_logits[offset:offset + 1]
                                                                       - saved_ec_logits[offset:offset + 1]).abs().max()),
                                     "ed_prediction": int(ed_pred[offset]), "ec_prediction": int(ec_pred[offset]),
                                     "historical_full_logits_available": False,
                                     "logit_reference": "frozen evaluators recomputed on the saved D0 view"})
        gen_indices = torch.tensor(first_gen, dtype=torch.long)
        coefficient = inputs["source"]["cohort"]["ols_coefficients"].index_select(0, gen_indices).to(device)
        result = d0_model.forward_from_ols(coefficient)
        gained = result["physical_w_per_pixel"] * inputs["source"]["checkpoint"]["camera_gain"].to(
            device=device, dtype=torch.float32)
        view = resize_view(gained)
        saved_view = d0_gen["images"].index_select(0, gen_indices).to(device)
        ed_logits, ec_logits = evaluator(view), decoder_ref._ec_logits(ec_evaluator, view)
        saved_ed = d0_gen["ed_logits"].index_select(0, gen_indices).to(device)
        saved_ec = d0_gen["ec_logits"].index_select(0, gen_indices).to(device)
        _require(torch.allclose(view, saved_view, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
                 and torch.allclose(ed_logits, saved_ed, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
                 and torch.allclose(ec_logits, saved_ec, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
                 and torch.equal(ed_logits.argmax(1), saved_ed.argmax(1))
                 and torch.equal(ec_logits.argmax(1), saved_ec.argmax(1)),
                 "D0 saved G4 view/logit replay failed in production batch10")
        for offset, index in enumerate(first_gen):
            d0_replay["generation"].append({
                "cohort_row": index,
                "view_max_abs_error": float((view[offset:offset + 1]
                                              - saved_view[offset:offset + 1]).abs().max()),
                "ed_logit_max_abs_error": float((ed_logits[offset:offset + 1]
                                                  - saved_ed[offset:offset + 1]).abs().max()),
                "ec_logit_max_abs_error": float((ec_logits[offset:offset + 1]
                                                  - saved_ec[offset:offset + 1]).abs().max()),
                "prediction_identical": True,
            })
    del d0_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_D0_replay") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_D0_replay") == 0.0,
             "D0 replay changed a frozen evaluator")

    model, optimizer = _new_joint_model_optimizer(source, previous_config, device)
    encoder, decoder = base._phase_groups(model)
    load_model_error = base._assert_close(
        source["checkpoint"]["model"], base._state_snapshot(model), "joint_zero_update_loaded_model",
    )
    load_optimizer_error = base._assert_close(
        source["checkpoint"]["optimizer"], cpu_copy(optimizer.state_dict()),
        "joint_zero_update_loaded_optimizer",
    )
    _require(load_model_error == 0.0 and load_optimizer_error == 0.0,
             "C zero-update source model or Adam state differs immediately after restore")
    first_batch = schedule[SOURCE_STEP].clone()
    first_target = image_batch(pixels, first_batch, "small", device)
    independent_readout = base._small_producer_consumer_check(model, first_target[:1])
    _require(independent_readout.get("status") == "PASSED", "C source independent OLS/camera readout failed")
    fd = [_joint_direction_check(model, pixels, first_batch, device, source, group)
          for group in ("encoder", "decoder")]
    model_initial = base._state_snapshot(model)
    optimizer_initial = cpu_copy(optimizer.state_dict())
    cpu_rng_initial = torch.random.get_rng_state().clone()
    cuda_rng_initial = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    torch.random.set_rng_state(source["checkpoint"]["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in source["checkpoint"]["cuda_rng"]])
    update_log: list[dict[str, Any]] = []
    _write_preflight_update_log(staging_dir, update_log)
    trials = []
    for index in range(3):
        call = {"call_number": index + 1, "branch": "original_process", "status": "CALL_STARTED",
                "schedule_cursor": SOURCE_STEP + index, "global_step_before": SOURCE_STEP + index,
                "sample_rows": [int(value) for value in schedule[SOURCE_STEP + index].tolist()],
                "device": torch.cuda.get_device_name(device)}
        _record_preflight_call(staging_dir, update_log, call)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss = _joint_update(model, optimizer, pixels, schedule[SOURCE_STEP + index], device, source)
        end_event.record()
        torch.cuda.synchronize(device)
        steps = _optimizer_steps_by_phase(optimizer, encoder, decoder)
        call.update({"status": "COMPLETED", "global_step_after": SOURCE_STEP + index + 1,
                     "optimizer_steps": steps, "loss": loss,
                     "elapsed_event_ms": float(start_event.elapsed_time(end_event)),
                     "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device))})
        _write_preflight_update_log(staging_dir, update_log)
        trials.append(loss)
    _require(_optimizer_steps(optimizer, encoder, decoder) == (SOURCE_STEP + 3, SOURCE_STEP + 3),
             "joint preflight did not advance both Adam phase groups")
    recovery_input = _preflight_recovery_input(
        config, source["checkpoint"], model, optimizer, encoder, decoder,
        schedule[SOURCE_STEP:SOURCE_STEP + 4], torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    atomic_torch_save_new(staging_dir / "recovery_input.pt", recovery_input)
    call = {"call_number": 4, "branch": "original_process_fourth", "status": "CALL_STARTED",
            "schedule_cursor": SOURCE_STEP + 3, "global_step_before": SOURCE_STEP + 3,
            "sample_rows": [int(value) for value in recovery_input["rows"][3].tolist()],
            "device": torch.cuda.get_device_name(device)}
    _record_preflight_call(staging_dir, update_log, call)
    direct_start_event, direct_end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    direct_start_event.record()
    direct_loss = _joint_update(model, optimizer, pixels, schedule[SOURCE_STEP + 3], device, source)
    direct_end_event.record()
    torch.cuda.synchronize(device)
    direct_model, direct_optimizer = base._state_snapshot(model), cpu_copy(optimizer.state_dict())
    direct_steps = _optimizer_steps(optimizer, encoder, decoder)
    direct_cuda = {"device": torch.cuda.get_device_name(device),
                   "elapsed_event_ms": float(direct_start_event.elapsed_time(direct_end_event)),
                   "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device))}
    direct_result = {
        "task_id": TASK_ID, "completed_updates": 4,
        "origin_checkpoint_run_id": source["checkpoint"].get("run_id"),
        "completed_global_step": SOURCE_STEP + 4, "schedule_cursor": SOURCE_STEP + 3,
        "model": direct_model,
        "optimizer": direct_optimizer,
        "optimizer_steps": _optimizer_steps_by_phase(optimizer, encoder, decoder),
        "loss": direct_loss, "rows": recovery_input["rows"][3].detach().cpu().clone(),
        "cpu_rng": torch.random.get_rng_state().cpu().clone(),
        "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "cuda": direct_cuda,
    }
    atomic_torch_save_new(staging_dir / "direct_fourth_update.pt", direct_result)
    call.update({"status": "COMPLETED", "global_step_after": SOURCE_STEP + 4,
                 "optimizer_steps": direct_result["optimizer_steps"], "loss": direct_loss,
                 "elapsed_event_ms": direct_cuda["elapsed_event_ms"],
                 "peak_reserved_vram_bytes": direct_cuda["peak_reserved_vram_bytes"]})
    _write_preflight_update_log(staging_dir, update_log)
    worker = subprocess.run(
        [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_small_joint_update_readability_pilot_v1",
         "--recovery-worker-dir", str(staging_dir), "--config", str(config_path),
         "--previous-config", str(ROOT / config["control"]["previous_config"]),
         "--output-dir", str(ROOT / config["output"]["experiment_root"])],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    _require(worker.returncode == 0,
             "independent joint-update recovery failed: " + (worker.stderr[-2500:] or worker.stdout[-2500:]))
    recovered = torch.load(staging_dir / "recovery_fourth_update.pt", map_location="cpu", weights_only=False)
    model_error = base._assert_close(direct_model, recovered["model"], "joint_recovery_model")
    optimizer_error = base._assert_close(direct_optimizer, recovered["optimizer"], "joint_recovery_optimizer")
    base._assert_close(direct_loss, recovered["loss"], "joint_recovery_loss")
    _require(recovered["optimizer_steps"] == direct_result["optimizer_steps"]
             and model_error <= RECOVERY_ATOL and optimizer_error <= RECOVERY_ATOL
             and base._assert_close(direct_loss, recovered["loss"], "joint_recovery_loss") <= RECOVERY_ATOL
             and torch.equal(direct_result["rows"], recovered["rows"])
             and torch.equal(direct_result["cpu_rng"], recovered["cpu_rng"])
             and len(direct_result["cuda_rng"]) == len(recovered["cuda_rng"])
             and all(torch.equal(left, right) for left, right in zip(
                 direct_result["cuda_rng"], recovered["cuda_rng"], strict=True))
             and _optimizer_steps(optimizer, encoder, decoder) == direct_steps,
             "independent joint-update fourth-step recovery exceeded the fixed tolerance")
    model.load_state_dict(model_initial, strict=True)
    optimizer.load_state_dict(optimizer_initial)
    torch.random.set_rng_state(cpu_rng_initial)
    torch.cuda.set_rng_state_all(cuda_rng_initial)
    _require(base._assert_close(model_initial, base._state_snapshot(model), "joint_preflight_rollback") == 0.0
             and base._assert_close(optimizer_initial, cpu_copy(optimizer.state_dict()),
                                    "joint_optimizer_rollback") == 0.0,
             "preflight did not roll back all model and Adam changes")
    _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_after_joint_preflight") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_after_joint_preflight") == 0.0,
             "frozen evaluator changed during joint preflight")
    preflight = {
        "status": "READY", "task_id": TASK_ID, "source_global_step": SOURCE_STEP,
        "control_global_step": ENDPOINT_STEP, "fixed_data_rows": {"train": len(train_indices), "dev": len(dev_indices),
                                                                    "official_test_reads": 0},
        "zero_update_restore": {"model_max_abs_error": load_model_error,
                                 "optimizer_max_abs_error": load_optimizer_error,
                                 "optimizer_steps": _optimizer_steps_by_phase(optimizer, encoder, decoder)},
        "continuous_schedule": {"C_35000_45000_matches": True, "D0_45000_50000_matches": True},
        "D0_saved_replay": d0_replay, "source_independent_readout": independent_readout,
        "joint_phase_direction_checks": fd, "engineering_updates": trials,
        "engineering_update_log": str((staging_dir / "engineering_update_log.json").name),
        "engineering_adam_calls": 5, "independent_process_recovery": {
            "passed": True, "direct_vs_restored_model_max_abs": model_error,
            "direct_vs_restored_optimizer_max_abs": optimizer_error,
            "rtol": RECOVERY_RTOL, "atol": RECOVERY_ATOL,
        },
        "preflight_rolled_back": True, "frozen_evaluators_unchanged": True,
        "official_test_reads": 0, "scientific_updates_before_execution": 0,
        "device": torch.cuda.get_device_name(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=uuid.uuid4().hex,
        trajectory_id=f"{TASK_ID}:seed:{SEED}:source_step:{SOURCE_STEP}",
        semantic_authority_id=AUTHORITY,
        data_role="MNIST_TRAIN_55000_DEV_5000_OFFICIAL_TEST_UNREAD",
        config_identity=config,
        model_identity=source["checkpoint"]["model_identity"],
        optimizer_identity={"name": "Adam", "lr": 1e-3, "betas": [0.9, 0.999],
                            "epsilon": 1e-8, "weight_decay": 0.0,
                            "source_encoder_step": SOURCE_STEP, "source_decoder_step": SOURCE_STEP,
                            "trainable_groups": ["encoder_phase", "decoder_phase"]},
        sampler_identity={"seed": SEED, "logical_batch_size": BATCH_SIZE,
                          "sample_start_global_step": SOURCE_STEP, "continuation_updates": UPDATES},
        execution_source_semantics_id=config["execution_source_semantics_id"],
    )
    control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:seed:{SEED}:preflight_and_execution")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASSED", "checks": [
            "explicit_C_45000_source_and_D0_50000_control_identity",
            "C_35000_45000_and_D0_45000_50000_continuous_sample_fragments",
            "D0_per_class_DEV_and_G4_saved_view_replay",
            "D0_G4_logits_rtol_2e-6_atol_5e-8_and_prediction_identity",
            "joint_Encoder_and_Decoder_real_updates",
            "BOGT_plus_SSIM_direction_finite_difference_both_phase_groups",
            "independent_process_fourth_update_recovery_and_rollback",
            "E_D_E_C_and_nonphase_parameters_frozen",
            "official_TEST_read_count_zero",
        ], "recovery_passed": True, "all_declared_trainable_groups_updated": True},
        resource={"passed": True, "status": "PASSED", "device": torch.cuda.get_device_name(device),
                  "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
        implementation={"passed": True, "status": "PASSED", "engine": "cuda_gaussian_s12_grid_2048",
                        "selected_engine": "cuda_gaussian_s12_grid_2048"},
        selected_engine="cuda_gaussian_s12_grid_2048",
        resource_envelope={"device": torch.cuda.get_device_name(device),
                           "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
    )
    control.mark_preflight_ready(receipt)
    return preflight, control, receipt, identity


def _checkpoint_payload(
    model: torch.nn.Module, optimizer: torch.optim.Adam, completed: int, run_id: str,
    source: dict[str, Any], config: dict[str, Any], schedule: torch.Tensor,
    history: list[dict[str, Any]], elapsed: float,
) -> dict[str, Any]:
    encoder, decoder = base._phase_groups(model)
    return {
        "schema": CHECKPOINT_SCHEMA, "task_id": TASK_ID, "run_id": run_id,
        "source_run_id": source["checkpoint"]["run_id"],
        "source_checkpoint_step": SOURCE_STEP, "trainable_groups": ["encoder_phase", "decoder_phase"],
        "seed": SEED, "model_identity": source["checkpoint"]["model_identity"],
        "config_identity": config, "propagation_grid_size": 2048,
        "completed_updates": completed, "completed_global_step": SOURCE_STEP + completed,
        "model": base._state_snapshot(model), "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": {"encoder_adam_step": _optimizer_steps(optimizer, encoder, decoder)[0],
                            "decoder_adam_step": _optimizer_steps(optimizer, encoder, decoder)[1]},
        "schedule": schedule.detach().cpu(), "schedule_start_global_step": SOURCE_STEP,
        "scale": source["checkpoint"]["scale"].detach().cpu(),
        "power_reference": source["checkpoint"]["power_reference"].detach().cpu(),
        "camera_gain": source["checkpoint"]["camera_gain"].detach().cpu(),
        "weights": source["checkpoint"]["weights"],
        "structural_weight": source["checkpoint"]["structural_weight"],
        "history": history,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
        "evaluator_role": "frozen_confirmation_only", "elapsed_seconds_continuation": elapsed,
    }


def _validate_checkpoint(
    payload: dict[str, Any], run_id: str, source: dict[str, Any], config: dict[str, Any],
    schedule: torch.Tensor,
) -> int:
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("task_id") == TASK_ID
             and payload.get("run_id") == run_id
             and payload.get("source_run_id") == source["checkpoint"]["run_id"]
             and payload.get("source_checkpoint_step") == SOURCE_STEP
             and payload.get("trainable_groups") == ["encoder_phase", "decoder_phase"]
             and payload.get("seed") == SEED,
             "joint checkpoint task, source or trainable-scope identity changed")
    _require(payload.get("model_identity") == source["checkpoint"]["model_identity"]
             and payload.get("config_identity") == config
             and payload.get("propagation_grid_size") == 2048
             and payload.get("schedule_start_global_step") == SOURCE_STEP
             and torch.equal(payload.get("schedule"), schedule),
             "joint checkpoint model, config, geometry or sample rows changed")
    completed = payload.get("completed_updates")
    _require(type(completed) is int and 0 <= completed <= UPDATES
             and payload.get("completed_global_step") == SOURCE_STEP + completed,
             "joint checkpoint update counter is invalid")
    _require(payload.get("optimizer_steps") == {
        "encoder_adam_step": SOURCE_STEP + completed,
        "decoder_adam_step": SOURCE_STEP + completed,
    }, "joint checkpoint phase-specific Adam counters changed")
    _require(payload.get("structural_weight") == source["checkpoint"]["structural_weight"]
             and payload.get("weights") == source["checkpoint"]["weights"]
             and payload.get("evaluator_role") == "frozen_confirmation_only",
             "joint checkpoint loss references or evaluator role changed")
    _require(all(torch.isfinite(torch.as_tensor(payload[name])).all().item()
                 for name in ("scale", "power_reference", "camera_gain")),
             "joint checkpoint frozen physical references are nonfinite")
    return completed


def _save_checkpoint(optical_dir: Path, payload: dict[str, Any], completed: int,
                     run_id: str, source: dict[str, Any], config: dict[str, Any],
                     schedule: torch.Tensor) -> None:
    latest = optical_dir / "checkpoint_latest.pt"
    previous = optical_dir / "checkpoint_previous.pt"
    if latest.is_file():
        old = torch.load(latest, map_location="cpu", weights_only=False)
        _validate_checkpoint(old, run_id, source, config, schedule)
        atomic_torch_save(previous, old)
    atomic_torch_save(latest, payload)
    reread = torch.load(latest, map_location="cpu", weights_only=False)
    _require(_validate_checkpoint(reread, run_id, source, config, schedule) == completed,
             "joint rolling checkpoint did not pass full readback validation")
    if completed in (0, 2500, UPDATES):
        endpoint = optical_dir / f"checkpoint_step_{completed:06d}.pt"
        atomic_torch_save(endpoint, payload)
        saved = torch.load(endpoint, map_location="cpu", weights_only=False)
        _require(_validate_checkpoint(saved, run_id, source, config, schedule) == completed,
                 "joint milestone checkpoint did not pass full readback validation")


def _paired_effect(
    control_rows: list[dict[str, str]], joint_rows: list[dict[str, Any]],
    metric: str, seed: int, resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    control = {int(row["row"]): row for row in control_rows}
    joint = {int(row["row"]): row for row in joint_rows}
    _require(len(control) == len(joint) == 5_000 and control.keys() == joint.keys(),
             f"paired DEV population is incomplete or misaligned for {metric}")
    difference_by_class = []
    for label in range(10):
        rows = [row for row in sorted(control) if int(control[row]["label"]) == label]
        _require(len(rows) == 500 and all(int(joint[row]["label"]) == label for row in rows),
                 f"DEV class {label} is incomplete or misaligned")
        difference_by_class.append(np.asarray([float(joint[row][metric]) - float(control[row][metric])
                                               for row in rows], dtype=np.float64))
    point = float(np.mean([values.mean() for values in difference_by_class]))
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = np.mean([values[rng.integers(0, values.size, size=values.size)].mean()
                                for values in difference_by_class])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"metric": metric, "effect": "J_minus_D0", "point_estimate": point,
            "ci_95": [float(low), float(high)], "bootstrap_resamples": resamples,
            "bootstrap_seed": seed, "equal_class_paired": True}


def _paired_generation_effects(
    control_rows: list[dict[str, str]], joint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    a = {int(row["cohort_row"]): row for row in control_rows}
    b = {int(row["cohort_row"]): row for row in joint_rows}
    _require(len(a) == len(b) == 10_000 and a.keys() == b.keys(),
             "paired G4 generated population is incomplete")
    metrics: dict[str, dict[int, np.ndarray]] = {}
    pair_columns = {
        "ed_correct": ("ed_prediction",), "ec_correct": ("ec_prediction",),
        "view_mean_intensity": ("view_mean_intensity",),
        "proxy_clipping_area_fraction": ("proxy_clipping_area_fraction",),
        "capture_power_w": ("capture_power_w",),
    }
    for metric, columns in pair_columns.items():
        class_diff = {}
        for label in range(10):
            rows = [index for index in range(10_000) if int(a[index]["requested_label"]) == label]
            _require(len(rows) == 1000, f"G4 class {label} has an incomplete paired cohort")
            differences = []
            for index in rows:
                _require(int(b[index]["requested_label"]) == label
                         and int(b[index]["proposal_id"]) == int(a[index]["proposal_id"])
                         and int(b[index]["component_id"]) == int(a[index]["component_id"]),
                         f"G4 proposal/component identity changed for row {index}")
                if metric in {"ed_correct", "ec_correct"}:
                    suffix = metric[:2]
                    left = float(int(a[index][f"{suffix}_prediction"]) == label)
                    right = float(int(b[index][f"{suffix}_prediction"]) == label)
                else:
                    column = columns[0]
                    left, right = float(a[index][column]), float(b[index][column])
                differences.append(right - left)
            class_diff[label] = np.asarray(differences, dtype=np.float64)
        metrics[metric] = class_diff
    rng = np.random.default_rng(GEN_BOOTSTRAP_SEED)
    draws_by_metric = {metric: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for metric in metrics}
    ordered_labels = sorted(metrics["ed_correct"])
    for sample in range(BOOTSTRAP_RESAMPLES):
        class_indices = {
            label: rng.integers(0, metrics["ed_correct"][label].size,
                                size=metrics["ed_correct"][label].size)
            for label in ordered_labels
        }
        for metric, classes in metrics.items():
            draws_by_metric[metric][sample] = np.mean([
                classes[label][class_indices[label]].mean() for label in ordered_labels
            ])
    result = {}
    for metric, classes in metrics.items():
        point = float(np.mean([values.mean() for values in classes.values()]))
        draws = draws_by_metric[metric]
        low, high = np.quantile(draws, [0.025, 0.975])
        result[metric] = {"effect": "J_minus_D0", "point_estimate": point,
                          "ci_95": [float(low), float(high)], "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                          "bootstrap_seed": GEN_BOOTSTRAP_SEED, "equal_class_paired": True,
                          "shared_resampling_indices_across_metrics": True}
    return result


def _write_comparison_images(
    optical_dir: Path, d0_dev: dict[str, Any], j_dev: dict[str, Any],
    d0_generation: dict[str, Any], j_generation: dict[str, Any],
) -> None:
    _require(torch.equal(d0_dev["rows"], j_dev["rows"])
             and torch.equal(d0_dev["labels"], j_dev["labels"]),
             "D0/J fixed DEV display sample identities differ")
    dev_rows = d0_dev["rows"].tolist()
    labels = d0_dev["labels"].tolist()
    tile = 120
    canvas = Image.new("L", (tile * 3, (tile + 22) * len(dev_rows)), 255)
    draw = ImageDraw.Draw(canvas)
    values = (d0_dev["target28"], d0_dev["view28"], j_dev["view28"])
    headings = ("Target", "D0", "J")
    for index, (row, label) in enumerate(zip(dev_rows, labels, strict=True)):
        y = index * (tile + 22)
        for column, (name, tensor) in enumerate(zip(headings, values, strict=True)):
            image = tensor[index].detach().cpu().float().squeeze().clamp(0, 1).numpy()
            gray = np.rint(image * 255).astype(np.uint8)
            preview = Image.fromarray(gray, mode="L").resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(preview, (column * tile, y + 22))
            draw.text((column * tile + 4, y + 3), f"{name} · class {label} · row {row}", fill=0)
    canvas.save(optical_dir / "dev_comparison.png", optimize=True)

    d0_indices, j_indices = d0_generation["native_capture_indices"], j_generation["native_capture_indices"]
    _require(torch.equal(d0_indices, j_indices) and tuple(d0_indices.shape) == (100,),
             "D0/J fixed native G4 coefficient identities differ")
    d0_native, j_native = d0_generation["native_capture_images"], j_generation["native_capture_images"]
    _require(tuple(d0_native.shape) == tuple(j_native.shape) == (100, 1, 200, 200),
             "D0/J native G4 display population changed")
    for name, tensor in (("d0_native_g4.png", d0_native), ("j_native_g4.png", j_native)):
        sheet = Image.new("L", (10 * 124, 10 * 124), 255)
        painter = ImageDraw.Draw(sheet)
        for index, image_tensor in enumerate(tensor):
            image = image_tensor.detach().cpu().float().squeeze().clamp(0, 1).numpy()
            pixels = np.rint(image * 255).astype(np.uint8)
            preview = Image.fromarray(pixels, mode="L").resize((120, 120), Image.Resampling.NEAREST)
            x, y = (index % 10) * 124, (index // 10) * 124
            sheet.paste(preview, (x, y + 4))
            painter.text((x + 2, y - 1), f"{index + 1}", fill=0)
        sheet.save(optical_dir / name, optimize=True)


def _evaluate_joint_endpoint(
    optical_dir: Path, run_id: str, model: torch.nn.Module, inputs: dict[str, Any],
    pixels: torch.Tensor, labels: torch.Tensor, dev_indices: torch.Tensor,
    evaluator: torch.nn.Module, ec_evaluator: torch.nn.Module, device: torch.device,
) -> dict[str, Any]:
    branch_dir = optical_dir / "J"
    branch_dir.mkdir(exist_ok=True)
    source = inputs["source"]
    dev_summary, dev_examples = decoder_ref._evaluate_reconstruction(
        model, evaluator, ec_evaluator, pixels, labels, dev_indices, device, source)
    dev_examples.update({"task_id": TASK_ID, "run_id": run_id, "branch": "J", "global_step": ENDPOINT_STEP})
    atomic_torch_save(branch_dir / "dev_display_examples.pt", dev_examples)
    decoder_ref._write_rows_csv(branch_dir / "reconstruction_metrics_dev.csv", dev_summary["per_image"])
    cohort = source["cohort"]
    generation_summary, generated = decoder_ref._evaluate_generation(
        model, evaluator, ec_evaluator, source, cohort, inputs["selection"],
        labels, pixels, dev_indices, device)
    generated.update({"task_id": TASK_ID, "run_id": run_id, "branch": "J", "global_step": ENDPOINT_STEP})
    atomic_torch_save(branch_dir / "generated_cohort.pt", generated)
    saved_generation = torch.load(branch_dir / "generated_cohort.pt", map_location="cpu", weights_only=False)
    _require(saved_generation.get("task_id") == TASK_ID and saved_generation.get("branch") == "J"
             and saved_generation.get("global_step") == ENDPOINT_STEP
             and tuple(saved_generation["images"].shape) == (10_000, 1, 28, 28)
             and tuple(saved_generation["ed_logits"].shape) == (10_000, 10)
             and tuple(saved_generation["ec_logits"].shape) == (10_000, 10)
             and torch.equal(saved_generation["proposal_ids"], cohort["proposal_ids"]),
             "J generated cohort failed the saved production-consumer readback")
    decoder_ref._write_rows_csv(branch_dir / "generation_metrics_per_image.csv", generation_summary["per_image"])
    base._write_contact_sheet(branch_dir / "generation_preview.png", saved_generation["images"],
                              saved_generation["requested_labels"], 10)
    d0_dev = torch.load(inputs["control_checkpoint_path"].parent / "dev_display_examples.pt",
                        map_location="cpu", weights_only=False)
    _write_comparison_images(optical_dir, d0_dev, dev_examples, inputs["control_generation"], saved_generation)
    d0_reconstruction = inputs["control_endpoint"]["reconstruction"]
    primary = {metric: _paired_effect(inputs["control_dev_rows"], dev_summary["per_image"], metric,
                                       RECON_BOOTSTRAP_SEED)
               for metric in ("ssim", "psnr_db", "mse")}
    generation_effects = _paired_generation_effects(inputs["control_gen_rows"], generation_summary["per_image"])
    d0_generation_summary = inputs["control_endpoint"]["generation"]
    report = {
        "branch": "J", "global_step": ENDPOINT_STEP, "measurement_complete": True,
        "reconstruction": {key: value for key, value in dev_summary.items() if key != "per_image"},
        "generation": {key: value for key, value in generation_summary.items() if key != "per_image"},
        "primary_effects": primary, "fixed_G4_generation_effects": generation_effects,
        "control_D0": {
            "global_step": 50_000,
            "reconstruction": {key: d0_reconstruction.get(key) for key in
                               ("population", "mse_mean", "psnr_db_mean", "ssim_mean",
                                "native_gradient_mse_mean", "proxy_clipping_area_fraction_macro")},
            "generation": {key: d0_generation_summary.get(key) for key in
                           ("population", "ed_macro_recall", "ec_macro_recall",
                            "proxy_clipping_area_fraction_macro", "generated_to_dev_tv_ratio_by_class",
                            "capture_power_w_mean", "view_mean_intensity_mean")},
            "evidence_reused": True,
        },
    }
    atomic_write_json(branch_dir / "endpoint_summary.json", report)
    return {"summary": report, "reconstruction_rows": dev_summary["per_image"],
            "generation_rows": generation_summary["per_image"], "dev_examples": dev_examples,
            "generated": generated}


def _write_optical_html(optical_dir: Path, report: dict[str, Any]) -> None:
    # The HTML consumer receives the machine report, whose registered decision
    # object is primary_gate. endpoint_summary.json uses primary_effects instead.
    effect = report.get("primary_gate", {})
    if not effect:
        effect = report.get("primary_effects", {}).get("ssim", {})
    ci = effect.get("ci_95")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        ci = [None, None]

    def display_number(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, (int, float)) and math.isfinite(value) else "N/A"

    status = html.escape(report.get("scientific_decision", effect.get("decision", "PENDING")))
    point = effect.get("point_estimate")
    page = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>联合更新与可读性试点</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;line-height:1.55}"
        "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.45rem}img{max-width:100%}</style>"
        "<h1>Gaussian小架构：联合更新与自动评阅原型</h1>"
        f"<p>机器阶段：{html.escape(report.get('status',''))}；主要科学判断：{status}。</p>"
        f"<p>J−D0平均DEV SSIM：{display_number(point)}，95%区间"
        f"[{display_number(ci[0])}, {display_number(ci[1])}]。</p>"
        "<p>20 dB / 0.80仍作为原系统检查表；本轮只判断固定预算下的配对增量，"
        "Gaussian几何仍标记PROJECT_GEOMETRY_UNMEASURED。</p>"
        "<h2>固定DEV样本</h2><p>每类前两张，目标、D0、J并列：</p><img src='dev_comparison.png'>"
        "<h2>固定G4原生CMOS示例</h2><p>100个既定系数，D0和J分别按固定顺序展示：</p>"
        "<img src='d0_native_g4.png'><img src='j_native_g4.png'>"
        "<h2>机器报告</h2><p><a href='machine_report.json'>JSON</a> · "
        "<a href='J/reconstruction_metrics_dev.csv'>DEV逐图表</a> · "
        "<a href='J/generation_metrics_per_image.csv'>G4逐图表</a> · "
        "<a href='../auto_review_prototype/report.html'>自动评阅原型</a></p>"
        "<p>分类器只作辅助读数；未对J增加人工评分。此结果不改变历史VALID_FAILED。</p></html>"
    )
    (optical_dir / "report.html").write_text(page, encoding="utf-8")


def _validate_saved_machine_report(optical_dir: Path, progress: dict[str, Any]) -> dict[str, Any]:
    """Admit report-only closeout only for the already complete fixed measurement."""
    _require(progress.get("stage") == "EVALUATING"
             and progress.get("completed_updates") == UPDATES
             and progress.get("global_step") == ENDPOINT_STEP
             and progress.get("status") in {"RUNNING", "INTERRUPTED"}
             and progress.get("last_complete_save_unix") is not None,
             "saved-report closeout requires the complete endpoint in EVALUATING state")
    required = (
        "checkpoint_step_005000.pt", "terminal_model.pt", "J/endpoint_summary.json",
        "machine_report.json", "report.json", "dev_comparison.png", "d0_native_g4.png",
        "j_native_g4.png", "J/reconstruction_metrics_dev.csv", "J/generation_metrics_per_image.csv",
        "J/generated_cohort.pt", "J/dev_display_examples.pt",
    )
    missing = [name for name in required if not (optical_dir / name).is_file()]
    _require(not missing, f"saved-report closeout is missing required artifacts: {missing}")
    with (optical_dir / "J/reconstruction_metrics_dev.csv").open(encoding="utf-8-sig", newline="") as stream:
        reconstruction_rows = sum(1 for _row in csv.DictReader(stream))
    with (optical_dir / "J/generation_metrics_per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
        generation_rows = sum(1 for _row in csv.DictReader(stream))
    _require(reconstruction_rows == 5_000 and generation_rows == 10_000,
             "saved-report closeout found incomplete fixed-population CSVs")
    report = json.loads((optical_dir / "machine_report.json").read_text(encoding="utf-8"))
    report_copy = json.loads((optical_dir / "report.json").read_text(encoding="utf-8"))
    run_id = progress.get("run_id")
    _require(report == report_copy
             and report.get("schema") == "lightgen_small_joint_update_readability_report.v1"
             and report.get("task_id") == TASK_ID and report.get("run_id") == run_id
             and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("measurement_complete") is True
             and report.get("training", {}).get("new_scientific_updates") == UPDATES
             and report.get("training", {}).get("global_step") == ENDPOINT_STEP
             and report.get("primary_gate", {}).get("metric") in {
                 "mean_DEV_SSIM_J_minus_D0", "ssim",
             }
             and len(report.get("primary_gate", {}).get("ci_95", [])) == 2,
             "saved machine report identity, decision or completion state is invalid")
    density_probe = report.get("plane_power_density_training_probe")
    if density_probe is not None:
        coverage = base._validate_gaussian_plane_density_coverage(
            optical_dir, "J", [SOURCE_STEP + step for step in range(0, UPDATES + 1, CHECKPOINT_INTERVAL)], 2048,
        )
        _require(density_probe.get("coverage", {}).get("status") == "VERIFIED"
                 and density_probe.get("coverage", {}).get("record_count") == coverage["record_count"],
                 "saved J report did not consume complete plane-density probe coverage")
    endpoint = json.loads((optical_dir / "J/endpoint_summary.json").read_text(encoding="utf-8"))
    endpoint_effect = endpoint.get("primary_effects", {}).get("ssim", {})
    report_effect = report["primary_gate"]
    _require(endpoint.get("measurement_complete") is True
             and endpoint.get("global_step") == ENDPOINT_STEP
             and endpoint_effect.get("point_estimate") == report_effect.get("point_estimate")
             and endpoint_effect.get("ci_95") == report_effect.get("ci_95"),
             "saved endpoint summary differs from the machine decision")
    return report


def _finalize_saved_machine_report(
    run_dir: Path, optical_dir: Path, progress_path: Path, progress: dict[str, Any],
    report: dict[str, Any], control: ResearchRunControl,
) -> None:
    """Finish presentation and lifecycle state without repeating scientific work."""
    closeout_started = time.perf_counter()
    timing = report.setdefault("timing", {})
    # The prior successful resume measured endpoint evaluation and export after
    # model restore. The checkpoint's training duration is cumulative and must
    # not be subtracted from or added to that process wall time.
    successful_process_seconds = timing.get("total_cloud_process")
    _require(isinstance(successful_process_seconds, (int, float))
             and math.isfinite(successful_process_seconds) and successful_process_seconds >= 0,
             "saved successful evaluation process duration is invalid")
    timing["evaluation_and_export"] = float(successful_process_seconds)
    timing["closeout_repair_seconds"] = time.perf_counter() - closeout_started
    timing["interpretation"] = (
        "training is cumulative checkpoint time; evaluation_and_export and total_cloud_process "
        "are the successful post-restore evaluation process and are not additive with training; "
        "failed attempts and queue/wait time are excluded"
    )
    gate = report["primary_gate"]
    gate["metric"] = "mean_DEV_SSIM_J_minus_D0"
    gate["component_metric"] = "ssim"
    gate["estimand"] = "J_minus_D0"
    _write_optical_html(optical_dir, report)
    report["output"]["bytes"] = base._directory_bytes(run_dir)
    if report["output"]["bytes"] > report["output"]["budget_bytes"]:
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    atomic_write_json(optical_dir / "machine_report.json", report)
    atomic_write_json(optical_dir / "report.json", report)
    _write_optical_html(optical_dir, report)
    # One final readback verifies that the report and presentation consumed the
    # same decision and that the byte count includes all formal result files.
    report["output"]["bytes"] = base._directory_bytes(run_dir)
    atomic_write_json(optical_dir / "machine_report.json", report)
    atomic_write_json(optical_dir / "report.json", report)
    html_text = (optical_dir / "report.html").read_text(encoding="utf-8")
    expected_ssim = report["primary_gate"]["point_estimate"]
    _require(f"J−D0平均DEV SSIM：{expected_ssim:.6f}" in html_text
             and "STRUCTURAL_INCREMENT_SUPPORTED" in html_text,
             "saved-report HTML does not reflect the fixed machine decision")
    if report["measurement_complete"]:
        control.complete()
        state = "COMPLETE_DIAGNOSTIC"
    else:
        control.invalid_harness()
        state = "INVALID_HARNESS"
    progress.update({"stage": "COMPLETE" if report["measurement_complete"] else "OUTPUT_BUDGET_FAILED",
                     "status": state, "completed_updates": UPDATES, "global_step": ENDPOINT_STEP,
                     "checkpoint": str(optical_dir / "checkpoint_step_005000.pt"),
                     "last_complete_save_unix": time.time()})
    _write_progress(progress_path, progress)
    atomic_write_json(optical_dir / "run_state.json", {
        "status": state, "scientific_status": report["scientific_decision"],
        "completed_updates": UPDATES, "global_step": ENDPOINT_STEP,
        "measurement_complete": report["measurement_complete"], "official_test_reads": 0,
    })


def _write_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["last_heartbeat_unix"] = time.time()
    atomic_write_json(path, progress)


def _machine_report(
    run_dir: Path, optical_dir: Path, run_id: str, config: dict[str, Any],
    preflight: dict[str, Any], source: dict[str, Any], summary: dict[str, Any],
    window: dict[str, Any], readout: dict[str, Any], elapsed: dict[str, float],
) -> dict[str, Any]:
    effect = summary["primary_effects"]["ssim"]
    low, high = effect["ci_95"]
    if low > 0.0:
        decision = "STRUCTURAL_INCREMENT_SUPPORTED"
    elif high < 0.0:
        decision = "STRUCTURAL_OUTCOME_WORSE"
    else:
        decision = "STRUCTURAL_INCREMENT_UNCLEAR"
    checkpoint = torch.load(optical_dir / "checkpoint_step_005000.pt", map_location="cpu", weights_only=False)
    result = {
        "schema": "lightgen_small_joint_update_readability_report.v1",
        "task_id": TASK_ID, "run_id": run_id, "status": "COMPLETE_DIAGNOSTIC",
        "measurement_complete": True, "scientific_decision": decision,
        "objective": "J joint Encoder+Decoder 5000 updates versus existing D0 Decoder-only 5000 updates from same C45k source",
        "source": {"task_id": source["checkpoint"]["task_id"], "run_id": source["checkpoint"]["run_id"],
                   "global_step": SOURCE_STEP, "quality_status_preserved": "VALID_FAILED",
                   "geometry_status": "PROJECT_GEOMETRY_UNMEASURED"},
        "training": {"new_scientific_updates": UPDATES, "global_step": ENDPOINT_STEP,
                     "optimizer_steps": checkpoint["optimizer_steps"],
                     "trainable_groups": ["encoder_phase", "decoder_phase"],
                     "phase_parameter_count": 40_000, "elapsed_seconds": elapsed["training"],
                     "seconds_per_update": elapsed["training"] / UPDATES,
                     "device": preflight["device"],
                     "preflight_engineering_adam_calls_rolled_back": 5},
        "preflight": preflight,
        "numerical": {"endpoint_window": window, "independent_ols_camera_readout": readout},
        "endpoints": {"D0": summary["control_D0"], "J": summary},
        "primary_gate": {**effect, "metric": "mean_DEV_SSIM_J_minus_D0",
                         "component_metric": "ssim", "estimand": "J_minus_D0",
                         "decision": decision,
                         "meaning": {"lower_bound_gt_zero": "supports this local joint-update structural increment",
                                     "interval_spans_zero": "independent structural gain is not established",
                                     "upper_bound_lt_zero": "structure is worse under this budget"}},
        "generation_fixed_G4_pairing": summary["fixed_G4_generation_effects"],
        "historical_system_quality_checklist": {
            "status": "VALID_FAILED", "status_changed": False, "thresholds": config["quality_gate"],
            "interpretation": "checklist retained as system target; not an entry prerequisite for this paired research",
        },
        "auto_review_prototype": {"status": "SEE_LOCAL_AUXILIARY_REPORT",
                                  "decision_role": "none; does not enter optical loss or this Gate"},
        "human_review_of_J": "NOT_RUN_BY_PLAN",
        "timing": elapsed,
        "artifacts": {"endpoint_checkpoint": "checkpoint_step_005000.pt",
                      "dev_metrics": "J/reconstruction_metrics_dev.csv",
                      "generated_metrics": "J/generation_metrics_per_image.csv",
                      "paired_report": "J/endpoint_summary.json"},
        "limitations": [
            "one source model, one fixed continuation population and one seed",
            "G4 prior was not refit; generation result is compatibility with fixed old coefficients",
            "E_D/E_C are auxiliary classifiers, not human or hardware readability qualification",
            "no new human ratings were collected for J",
            "PROJECT_GEOMETRY_UNMEASURED; no physical CMOS or independent TEST claim",
            "the 20 dB/0.80 checklist and historical VALID_FAILED status are unchanged",
        ],
        "output": {"directory": str(run_dir), "optical_directory": str(optical_dir),
                   "bytes": base._directory_bytes(run_dir),
                   "budget_bytes": config["output"]["formal_result_budget_bytes"]},
    }
    return result


def _validate_resume_identity(
    identity_record: dict[str, Any], progress: dict[str, Any], config: dict[str, Any],
    source_run_id: str, model_identity: dict[str, Any],
) -> tuple[ResearchRunIdentity, str]:
    identity = ResearchRunIdentity.from_mapping(identity_record.get("identity", {}))
    progress_run_id = progress.get("run_id")
    expected_authorization = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
    _require(
        identity.task_id == TASK_ID
        and identity.semantic_authority_id == AUTHORITY
        and identity.trajectory_id == f"{TASK_ID}:seed:{SEED}:source_step:{SOURCE_STEP}"
        and identity.scientific_attempt_id == f"joint_seed_{SEED}_from_C{SOURCE_STEP}"
        and identity.data_role == "MNIST_TRAIN_55000_DEV_5000_OFFICIAL_TEST_UNREAD"
        and identity.config_identity == config
        and identity.model_identity == model_identity
        and identity.execution_source_semantics_id == config["execution_source_semantics_id"]
        and identity_record.get("authorization_id") == expected_authorization
        and identity_record.get("run_id") == progress_run_id
        and identity_record.get("source_run_id") == source_run_id,
        "joint resume run identity changed",
    )
    return identity, str(progress_run_id)


def _run_optical(
    args: argparse.Namespace, config: dict[str, Any], config_path: Path,
    source_path: Path, control_dir: Path, cohort_path: Path, reference_dir: Path,
) -> None:
    _require(torch.cuda.is_available(), "CUDA is required for the registered joint optical update")
    device = torch.device("cuda")
    run_dir = _resolve(args.output_dir)
    expected_run_dir = (ROOT / config["output"]["experiment_root"]).resolve()
    _require(run_dir == expected_run_dir, "output directory differs from the registered task path")
    optical_dir = run_dir / "optical"
    progress_path = run_dir / "progress.json"
    resuming = args.resume_state is not None
    _require(not (args.preflight_only and resuming), "preflight-only cannot be combined with resume")
    if resuming:
        _require(run_dir.is_dir() and optical_dir.is_dir()
                 and args.resume_state.resolve() == progress_path.resolve(),
                 "resume requires the explicit incomplete optical progress file")
    else:
        _require(not optical_dir.exists(), "an optical attempt already exists; explicit resume is required")

    inputs = _load_fixed_inputs(config, source_path, control_dir, cohort_path, reference_dir)
    pixels, labels, train_indices, dev_indices = base._load_data(ROOT / config["data"]["training_file"])
    inputs["source"]["training_labels"] = labels
    inputs["source"]["training_indices"] = train_indices
    inputs["source"]["training_pixels"] = pixels
    schedule = decoder_ref._schedule(labels, train_indices)
    _require(tuple(schedule.shape) == (50_000, 10), "fixed 45,000-50,000 schedule has the wrong shape")
    evaluator_path = (ROOT / config["evaluation"]["evaluator_checkpoint"]).resolve()
    ec_path = (ROOT / config["evaluation"]["ec_checkpoint"]).resolve()
    evaluator = base._load_evaluator(evaluator_path, device)
    ec_evaluator, ec_qualification = decoder_ref._load_ec_evaluator(ec_path, device)
    _require(evaluator.training is False and ec_evaluator.training is False
             and all(not parameter.requires_grad for parameter in evaluator.parameters())
             and all(not parameter.requires_grad for parameter in ec_evaluator.parameters()),
             "E_D/E_C must remain frozen in evaluation mode")
    ed_state, ec_state = base._state_snapshot(evaluator), base._state_snapshot(ec_evaluator)

    if resuming:
        progress_path = args.resume_state.resolve()
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _require(progress.get("schema") == "lightgen_small_joint_update_progress.v1"
                 and progress.get("task_id") == TASK_ID and progress.get("status") in {"RUNNING", "INTERRUPTED"}
                 and progress.get("config_identity") == config,
                 "joint resume progress identity or state is not resumable")
        identity_record = json.loads((optical_dir / "run_identity.json").read_text(encoding="utf-8"))
        identity, run_id = _validate_resume_identity(
            identity_record, progress, config, inputs["source"]["checkpoint"]["run_id"],
            inputs["source"]["checkpoint"]["model_identity"],
        )
        preflight = json.loads((optical_dir / "preflight.json").read_text(encoding="utf-8"))
        receipt = PreflightReceipt.from_mapping(
            json.loads((optical_dir / "preflight_ready.json").read_text(encoding="utf-8")))
        authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
        control = ResearchRunControl(identity, authorization_id=authorization_id)
        control.mark_preflight_ready(receipt)
        control.resume_scientific_attempt(optical_dir, identity.scientific_attempt_id or TASK_ID, receipt)
        if (optical_dir / "machine_report.json").exists():
            saved_report = _validate_saved_machine_report(optical_dir, progress)
            _finalize_saved_machine_report(
                run_dir, optical_dir, progress_path, progress, saved_report, control,
            )
            print(json.dumps({"status": saved_report["status"],
                              "scientific_decision": saved_report["scientific_decision"],
                              "report": str(optical_dir / "machine_report.json"),
                              "report_only_closeout": True}, ensure_ascii=False), flush=True)
            return
        registered = progress.get("checkpoint")
        _require(isinstance(registered, str), "resume progress lacks the explicit checkpoint path")
        checkpoint_path = Path(registered).resolve()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (ROOT / registered).resolve()
        _require(checkpoint_path.is_file() and checkpoint_path.parent.resolve() == optical_dir.resolve(),
                 "registered resume checkpoint is missing or outside the optical attempt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start = _validate_checkpoint(checkpoint, run_id, inputs["source"], config, schedule[SOURCE_STEP:])
        _require(progress.get("completed_updates") == start,
                 "resume progress counter differs from the loaded complete checkpoint")
        _require(progress.get("last_complete_save_unix") is not None,
                 "resume progress lacks a complete-save marker")
        model, optimizer = _restore_joint_state(checkpoint, inputs["previous_config"], device)
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        torch.random.set_rng_state(checkpoint["cpu_rng"].cpu())
        history = checkpoint.get("history")
        _require(isinstance(history, list), "resume checkpoint training history is missing")
        elapsed_base = float(checkpoint.get("elapsed_seconds_continuation", 0.0))
        _require(math.isfinite(elapsed_base) and elapsed_base >= 0, "resume elapsed training time is invalid")
    else:
        run_id = uuid.uuid4().hex
        stage = run_dir.parent / f"{run_dir.name}_preflight_{run_id[:8]}"
        stage.mkdir(parents=True, exist_ok=False)
        preserve_stage = False
        try:
            preflight, control, receipt, identity = _run_preflight(
                args, config, config_path, inputs, pixels, labels, train_indices, dev_indices,
                schedule, device, evaluator, ec_evaluator, stage)
            preflight["evaluator_qualification"] = ec_qualification
            if args.preflight_only:
                preflight_dir = run_dir.parent / f"{run_dir.name}_preflight_receipt_{run_id[:8]}"
                preflight_dir.mkdir(parents=False, exist_ok=False)
                atomic_write_new_json(preflight_dir / "config.json", config)
                atomic_write_new_json(preflight_dir / "preflight.json", preflight)
                atomic_write_new_json(preflight_dir / "preflight_ready.json", receipt.to_dict())
                atomic_write_new_json(preflight_dir / "run_identity.json", {
                    "identity": identity.to_dict(), "run_id": identity.run_id,
                    "authorization_id": f"{TASK_ID}:seed:{SEED}:preflight_and_execution",
                })
                atomic_write_new_json(preflight_dir / "preflight_status.json", {
                    "status": "READY", "scientific_updates": 0,
                    "engineering_adam_calls_rolled_back": 5,
                })
                print(json.dumps({"status": "PREFLIGHT_READY", "directory": str(preflight_dir)},
                                 ensure_ascii=False), flush=True)
                return
            _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_before_attempt") == 0.0
                     and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_before_attempt") == 0.0,
                     "preflight changed a frozen evaluator")
            run_dir.mkdir(parents=True, exist_ok=True)
            control.open_scientific_attempt(optical_dir, f"joint_seed_{SEED}_from_C{SOURCE_STEP}", receipt)
            identity = control.identity
            atomic_write_new_json(optical_dir / "config.json", config)
            atomic_write_new_json(optical_dir / "preflight.json", preflight)
            atomic_write_new_json(optical_dir / "preflight_ready.json", receipt.to_dict())
            atomic_write_new_json(optical_dir / "run_identity.json", {
                "identity": identity.to_dict(), "run_id": run_id,
                "source_run_id": inputs["source"]["checkpoint"]["run_id"],
                "authorization_id": f"{TASK_ID}:seed:{SEED}:preflight_and_execution",
            })
            atomic_torch_save_new(optical_dir / "source_references.pt", {
                "source_global_step": SOURCE_STEP,
                "source_run_id": inputs["source"]["checkpoint"]["run_id"],
                "scale": inputs["source"]["checkpoint"]["scale"],
                "power_reference": inputs["source"]["checkpoint"]["power_reference"],
                "camera_gain": inputs["source"]["checkpoint"]["camera_gain"],
                "weights": inputs["source"]["checkpoint"]["weights"],
                "structural_weight": inputs["source"]["checkpoint"]["structural_weight"],
            })
            atomic_torch_save_new(optical_dir / "fixed_training_rows.pt", {
                "schema": "lightgen_small_joint_update_fixed_rows.v1", "seed": SEED,
                "start_global_step": SOURCE_STEP, "rows": schedule[SOURCE_STEP:].cpu(),
            })
            cohort = inputs["source"]["cohort"]
            atomic_torch_save_new(optical_dir / "g4_fixed_inputs.pt", {
                "schema": "lightgen_small_joint_update_fixed_g4_inputs.v1",
                "task_id": TASK_ID, "source_task_id": cohort["task_id"],
                "source_model": cohort["model"], "method": "G4", "stream": 0,
                "source_global_step": SOURCE_STEP,
                "ols_coefficients": cohort["ols_coefficients"].cpu(),
                "requested_labels": cohort["requested_labels"].cpu(),
                "component_ids": cohort["component_ids"].cpu(),
                "proposal_ids": cohort["proposal_ids"].cpu(),
            })
            progress = {
                "schema": "lightgen_small_joint_update_progress.v1", "task_id": TASK_ID,
                "run_id": run_id, "source_run_id": inputs["source"]["checkpoint"]["run_id"],
                "config_identity": config, "status": "RUNNING", "stage": "READY_PREFLIGHT",
                "completed_updates": 0, "global_step": SOURCE_STEP,
                "checkpoint": str(optical_dir / "checkpoint_latest.pt"),
                "official_test_reads": 0,
                "monitoring": {"cloud_telemetry_seconds": 60,
                               "active_session_check_seconds": 300},
                "last_complete_save_unix": time.time(), "last_heartbeat_unix": time.time(),
            }
            _write_progress(progress_path, progress)
            model, optimizer = _new_joint_model_optimizer(inputs["source"], inputs["previous_config"], device)
            torch.random.set_rng_state(inputs["source"]["checkpoint"]["cpu_rng"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in inputs["source"]["checkpoint"]["cuda_rng"]])
            history = []
            start = 0
            elapsed_base = 0.0
            initial = _checkpoint_payload(model, optimizer, 0, run_id, inputs["source"],
                                          config, schedule[SOURCE_STEP:], history, 0.0)
            _save_checkpoint(optical_dir, initial, 0, run_id, inputs["source"], config, schedule[SOURCE_STEP:])
            progress.update({"stage": "TRAINING", "checkpoint": str(optical_dir / "checkpoint_latest.pt")})
            _write_progress(progress_path, progress)
        except Exception as error:
            preserve_stage = True
            atomic_write_json(stage / "preflight_failure.json", {
                "task_id": TASK_ID, "run_id": run_id, "error_type": type(error).__name__,
                "error": str(error),
            })
            raise
        finally:
            if not preserve_stage:
                import shutil
                shutil.rmtree(stage, ignore_errors=True)

    _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_before_training") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_before_training") == 0.0,
             "frozen evaluator changed before scientific training")
    inputs["source"]["training_witness_dir"] = optical_dir / "training_path_witnesses"
    inputs["source"]["training_capture_updates"] = sorted({
        1, start + 1, *[step for step in range(1, UPDATES + 1)
                        if step % CHECKPOINT_INTERVAL == 0],
    })
    encoder, decoder = base._phase_groups(model)
    # Record the frozen DEV probe at an explicit restored checkpoint before any
    # further optimizer update. The probe itself snapshots and verifies RNG,
    # modes, gradients, parameters, and buffers.
    density_probe = base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_indices, optical_dir,
        branch="J", global_step=SOURCE_STEP + start, grid_size=2048,
    )
    progress["plane_power_density_probe"] = density_probe
    _write_progress(progress_path, progress)
    nonphase = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                if name not in {"runtime.encoder.phase", "runtime.generator.phase"}}
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    started = time.perf_counter()
    training_start = started
    pending_signal: list[int] = []
    old_handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def _signal_handler(number: int, frame: Any) -> None:
        del frame
        pending_signal.append(number)

    for number in old_handlers:
        signal.signal(number, _signal_handler)
    completed = start
    try:
        for cursor in range(start, UPDATES):
            step = cursor + 1
            inputs["source"]["active_training_update"] = step
            values = _joint_update(model, optimizer, pixels, schedule[SOURCE_STEP + cursor], device,
                                   inputs["source"])
            completed = step
            if step % PROGRESS_INTERVAL == 0:
                history.append({"segment_step": step, **values})
                atomic_write_json(optical_dir / "training_history_live.json", history)
                progress.update({"stage": "TRAINING", "completed_updates": step,
                                 "global_step": SOURCE_STEP + step,
                                 "latest_loss": values, "training_elapsed_seconds": elapsed_base + time.perf_counter() - training_start,
                                 "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)})
                _write_progress(progress_path, progress)
            if step % CHECKPOINT_INTERVAL == 0 or pending_signal:
                checkpoint = _checkpoint_payload(model, optimizer, step, run_id, inputs["source"],
                                                 config, schedule[SOURCE_STEP:], history,
                                                 elapsed_base + time.perf_counter() - training_start)
                _save_checkpoint(optical_dir, checkpoint, step, run_id, inputs["source"],
                                 config, schedule[SOURCE_STEP:])
                progress.update({"completed_updates": step, "global_step": SOURCE_STEP + step,
                                 "checkpoint": str(optical_dir / "checkpoint_latest.pt"),
                                 "stage": "INTERRUPTED" if pending_signal else "TRAINING",
                                 "last_complete_save_unix": time.time()})
                _write_progress(progress_path, progress)
                density_probe = base._record_gaussian_plane_density_probe(
                    model, pixels, labels, dev_indices, optical_dir,
                    branch="J", global_step=SOURCE_STEP + step, grid_size=2048,
                )
                progress["plane_power_density_probe"] = density_probe
                _write_progress(progress_path, progress)
            if pending_signal:
                control.interrupted()
                atomic_write_json(optical_dir / "run_state.json", {
                    "status": "INTERRUPTED", "completed_updates": step,
                    "global_step": SOURCE_STEP + step, "scientific_updates": step,
                })
                return
        elapsed_training = elapsed_base + time.perf_counter() - training_start
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    _require(completed == UPDATES, "fixed joint scientific update budget was not completed")
    _require(_optimizer_steps(optimizer, encoder, decoder) == (ENDPOINT_STEP, ENDPOINT_STEP),
             "joint Encoder/Decoder Adam counters did not reach 50,000")
    _require(all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in nonphase.items())
             and base._assert_close(buffers, cpu_copy(dict(model.named_buffers())),
                                    "joint_frozen_buffers") == 0.0,
             "a frozen nonphase parameter or model buffer changed during joint training")
    _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_after_joint_training") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_after_joint_training") == 0.0,
             "a frozen confirmation evaluator changed during training")
    final_checkpoint = _checkpoint_payload(model, optimizer, UPDATES, run_id, inputs["source"],
                                           config, schedule[SOURCE_STEP:], history, elapsed_training)
    _save_checkpoint(optical_dir, final_checkpoint, UPDATES, run_id, inputs["source"],
                     config, schedule[SOURCE_STEP:])
    density_probe = base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_indices, optical_dir,
        branch="J", global_step=ENDPOINT_STEP, grid_size=2048,
    )
    plane_density_coverage = base._validate_gaussian_plane_density_coverage(
        optical_dir, "J", [SOURCE_STEP + step for step in range(0, UPDATES + 1, CHECKPOINT_INTERVAL)], 2048,
    )
    window_grid, window = base._window_check("small", pixels, labels, schedule,
                                             inputs["previous_config"], device, model=model)
    _require(window_grid == 2048 and window["complex_relative_error_max"] <= WINDOW_TOLERANCE
             and window["ols_coefficient_relative_error_by_grid"][2048] <= WINDOW_TOLERANCE
             and window["camera_w_per_pixel_relative_error_by_grid"][2048] <= WINDOW_TOLERANCE,
             "J endpoint failed the fixed 2048/4096 common-ROI, OLS or camera check")
    first_batch = image_batch(pixels, schedule[SOURCE_STEP], "small", device)
    readout = base._small_producer_consumer_check(model, first_batch[:1])
    _require(readout.get("status") == "PASSED", "J endpoint independent OLS/camera readout failed")
    terminal = {
        "schema": "lightgen_small_joint_update_terminal.v1", "task_id": TASK_ID,
        "run_id": run_id, "global_step": ENDPOINT_STEP,
        "model_identity": inputs["source"]["checkpoint"]["model_identity"],
        "model": base._state_snapshot(model), "optimizer_steps": final_checkpoint["optimizer_steps"],
    }
    atomic_torch_save(optical_dir / "terminal_model.pt", terminal)
    reread_terminal = torch.load(optical_dir / "terminal_model.pt", map_location="cpu", weights_only=False)
    _require(reread_terminal.get("task_id") == TASK_ID and reread_terminal.get("global_step") == ENDPOINT_STEP
             and base._assert_close(reread_terminal["model"], terminal["model"], "joint_terminal_model") == 0.0,
             "J terminal model failed saved readback")

    progress.update({"stage": "EVALUATING", "completed_updates": UPDATES,
                     "global_step": ENDPOINT_STEP, "checkpoint": str(optical_dir / "checkpoint_step_005000.pt"),
                     "last_complete_save_unix": time.time()})
    _write_progress(progress_path, progress)
    endpoint = _evaluate_joint_endpoint(optical_dir, run_id, model, inputs, pixels, labels,
                                        dev_indices, evaluator, ec_evaluator, device)
    elapsed_total = time.perf_counter() - started
    elapsed = {"preflight": float(preflight.get("elapsed_seconds", 0.0)),
               "training": elapsed_training, "evaluation_and_export": elapsed_total - elapsed_training,
               "total_cloud_process": elapsed_total}
    report = _machine_report(run_dir, optical_dir, run_id, config, preflight,
                             inputs["source"], endpoint["summary"], window, readout, elapsed)
    report["plane_power_density_training_probe"] = {
        "coverage": plane_density_coverage,
        "probe": density_probe,
        "measurement_scope": "fixed 20-image DEV probe at source, resume, and every 500-update checkpoint",
        "historical_logs_rewritten": False,
    }
    report["output"]["bytes"] = base._directory_bytes(run_dir)
    if report["output"]["bytes"] > report["output"]["budget_bytes"]:
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    atomic_write_json(optical_dir / "machine_report.json", report)
    atomic_write_json(optical_dir / "report.json", report)
    _write_optical_html(optical_dir, report)
    if report["measurement_complete"]:
        control.complete()
        state = "COMPLETE_DIAGNOSTIC"
    else:
        control.invalid_harness()
        state = "INVALID_HARNESS"
    progress.update({"stage": "COMPLETE" if report["measurement_complete"] else "OUTPUT_BUDGET_FAILED",
                     "status": state, "completed_updates": UPDATES, "global_step": ENDPOINT_STEP,
                     "checkpoint": str(optical_dir / "checkpoint_step_005000.pt"),
                     "last_complete_save_unix": time.time()})
    _write_progress(progress_path, progress)
    atomic_write_json(optical_dir / "run_state.json", {
        "status": state, "scientific_status": report["scientific_decision"],
        "completed_updates": UPDATES, "global_step": ENDPOINT_STEP,
        "measurement_complete": report["measurement_complete"], "official_test_reads": 0,
    })
    print(json.dumps({"status": report["status"], "scientific_decision": report["scientific_decision"],
                      "report": str(optical_dir / "machine_report.json"),
                      "elapsed_seconds": elapsed_total}, ensure_ascii=False), flush=True)


def _execute(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _read_config(config_path)
    run_dir = _resolve(args.output_dir)
    _require(run_dir == (ROOT / config["output"]["experiment_root"]).resolve(),
             "output directory differs from the registered task path")
    if args.recovery_worker_dir is not None:
        _require(args.previous_config is not None, "recovery worker requires the explicit prior config")
        _recovery_worker(_resolve(args.recovery_worker_dir), config_path, _resolve(args.previous_config))
        return
    _require(not (args.preflight_only and args.ratings_file is not None)
             and not (args.resume_state is not None and args.ratings_file is not None),
             "ratings finalization cannot be combined with preflight or resume")
    if args.ratings_file is not None:
        _require(args.control_evidence_dir is not None and args.readability_reference_dir is not None,
                 "rating finalization requires explicit R and D review directories")
        control_dir = _resolve(args.control_evidence_dir)
        reference_dir = _resolve(args.readability_reference_dir)
        _run_ratings_and_prototype(run_dir, config, _resolve(args.ratings_file), control_dir, reference_dir)
        print(json.dumps({"status": "REVIEW_AND_AUXILIARY_PILOT_COMPLETE",
                          "human_review": str(run_dir / "human_review_closeout" / "closeout.json"),
                          "prototype": str(run_dir / "auto_review_prototype" / "report.json")},
                         ensure_ascii=False), flush=True)
        return
    required = (args.source_checkpoint, args.control_evidence_dir, args.g4_cohort,
                args.readability_reference_dir)
    _require(all(value is not None for value in required),
             "training requires explicit source, control, G4 cohort and prior review paths")
    _run_optical(args, config, config_path, _resolve(args.source_checkpoint),
                 _resolve(args.control_evidence_dir), _resolve(args.g4_cohort),
                 _resolve(args.readability_reference_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--control-evidence-dir", type=Path)
    parser.add_argument("--g4-cohort", type=Path)
    parser.add_argument("--readability-reference-dir", type=Path)
    parser.add_argument("--ratings-file", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--recovery-worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--previous-config", type=Path, help=argparse.SUPPRESS)
    _execute(parser.parse_args())


if __name__ == "__main__":
    main()
