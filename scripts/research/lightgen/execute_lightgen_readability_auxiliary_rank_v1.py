"""Fit fixed packet-held-out readability ranking models from saved human reviews."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_READABILITY_AUXILIARY_RANK_V1"
SCHEMA = "lightgen_readability_auxiliary_rank.v1"
PACKAGES = ("R", "D", "P")
LABELS = tuple(range(10))
FIT_STEPS = 500
PREFLIGHT_STEPS = 3
BOOTSTRAP_SEED = 2026102316
BOOTSTRAP_RESAMPLES = 10_000
REVIEW_FRACTIONS = (0.1, 0.2, 0.3)
PRIMARY_FRACTION = 0.2
FEATURE_NAMES = tuple([f"ed_logit_{i}" for i in LABELS]
                      + [f"ec_logit_{i}" for i in LABELS]
                      + ["view_mean_intensity", "raw_peak_w_per_pixel", "proxy_clipping_area_fraction"])

OUTPUT_RELATIVE = Path("outputs/20261023_lightgen_readability_auxiliary_rank_v1/runs/attempt_01")
P_REVIEW_RELATIVE = Path("outputs/20261023_lightgen_c45k_phase_chart_readability_spec_v1/runs/attempt_01")
R_REVIEW_RELATIVE = Path("outputs/20261022_lightgen_small_g1_g4_native_readability_v1/runs/autodl_backup_attempt_01")
D_REVIEW_RELATIVE = Path("outputs/20261022_lightgen_small_decoder_native_gradient_ab_v1/runs/autodl_backup_attempt_01")
R_COHORT_RELATIVE = Path("outputs/20261021_lightgen_small_gaussian_replication_prior_confirmation_v1/runs/autodl_backup_attempt_01/generation/C_seed_2026101501")
PHASE_CONFIG_RELATIVE = Path("configs/research/lightgen/c45k_phase_chart_readability_spec_v1.yaml")


class AuxiliaryRankError(RuntimeError):
    """An input, fit, or measurement-consumer contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuxiliaryRankError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    torch.save(payload, pending)
    os.replace(pending, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None, f"CSV has no header: {path}")
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(pending, path)


def _feature_array(ed_logits: Any, ec_logits: Any, metrics: dict[str, Any]) -> np.ndarray:
    ed = torch.as_tensor(ed_logits).detach().cpu().double().numpy()
    ec = torch.as_tensor(ec_logits).detach().cpu().double().numpy()
    _require(ed.shape == ec.shape == (10,) and np.isfinite(ed).all() and np.isfinite(ec).all(),
             "saved evaluator logits are malformed")
    tail = np.asarray([float(metrics["view_mean_intensity"]),
                       float(metrics["raw_peak_w_per_pixel"]),
                       float(metrics["proxy_clipping_area_fraction"])], dtype=np.float64)
    features = np.concatenate((ed, ec, tail))
    _require(features.shape == (23,) and np.isfinite(features).all(),
             "fixed 23-feature input is incomplete or nonfinite")
    return features


def _exact_coefficient_key(value: Any) -> bytes:
    vector = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    _require(vector.shape == (100,) and np.iscomplexobj(vector) and np.isfinite(vector).all(),
             "saved complex coefficient vector is malformed")
    return vector.tobytes()


def _finalized_package_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.research.lightgen import execute_lightgen_c45k_phase_chart_readability_spec_v1 as review_task
    from scripts.research.lightgen import execute_lightgen_small_joint_update_readability_pilot_v1 as prior_task

    p_dir = ROOT / P_REVIEW_RELATIVE
    p_key = _read_json(p_dir / "blind_review_key.json")
    p_result = _read_json(p_dir / "human_review_results.json")
    p_report = _read_json(p_dir / "report.json")
    p_independent = _read_json(p_dir / "human_review_independent_count_check.json")
    _require(p_result.get("status") == "COMPLETE" and p_result.get("quality_control", {}).get("passed") is True
             and p_independent.get("status") == "PASSED"
             and p_key.get("review_id") == p_report.get("review_id")
             and p_report.get("primary_comparison") == p_result.get("primary_comparison"),
             "the fixed P human review is not finalized with valid quality control")
    p_ratings_path = p_dir / "human_ratings_submitted.json"
    p_ratings = review_task._read_ratings(p_ratings_path, p_key)

    r_dir = ROOT / R_REVIEW_RELATIVE
    d_dir = ROOT / D_REVIEW_RELATIVE
    r_ratings_path = r_dir / "human_ratings_submitted.json"
    d_ratings_path = d_dir / "human_ratings_submitted.json"
    legacy_records, legacy_receipt = prior_task._records_from_reviews(
        r_dir, d_dir, r_ratings_path, d_ratings_path)
    _require(len(legacy_records) == 400 and legacy_receipt["records"] == 400,
             "the fixed R/D review populations are incomplete")

    # R rows are checked against their explicit saved C45k cohort so feature
    # logits, coefficient groups, and blinded image identities share one row.
    r_key = _read_json(r_dir / "blind_review_key.json")
    r_cards = {card["card_id"]: card for card in r_key["cards"] if card.get("kind") == "generated"}
    r_payloads: dict[str, dict[str, Any]] = {}
    for method in ("G1", "G4"):
        path = ROOT / R_COHORT_RELATIVE / f"{method}_stream_0.pt"
        _require(path.is_file(), f"fixed R source cohort is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1"
                 and payload.get("method") == method and int(payload.get("global_step", -1)) == 45_000,
                 f"R {method} cohort identity or global step is wrong")
        r_payloads[method] = payload

    d_key = _read_json(d_dir / "blind_review_key.json")
    d_cards = {card["card_id"]: card for card in d_key["cards"] if card.get("kind") == "generated"}
    d_payloads = {
        branch: torch.load(d_dir / branch / "generated_cohort.pt", map_location="cpu", weights_only=False)
        for branch in ("D0", "Dgrad")
    }
    d_coefficients: dict[bytes, dict[str, Any]] = {}
    for branch, payload in d_payloads.items():
        _require(payload.get("task_id") == "LIGHTGEN_SMALL_DECODER_NATIVE_GRADIENT_AB_V1"
                 and payload.get("branch") == branch and int(payload.get("global_step", -1)) == 50_000,
                 f"D {branch} cohort identity or global step is wrong")
    for record in legacy_records:
        if record["package"] != "D":
            continue
        card = d_cards[record["card_id"]]
        branch = str(card["branch"])
        index = int(card["cohort_row"])
        coefficient_key = _exact_coefficient_key(d_payloads[branch]["ols_coefficients"][index])
        state = d_coefficients.setdefault(coefficient_key, {"branches": set(), "labels": set()})
        state["branches"].add(branch)
        state["labels"].add(int(record["requested_label"]))
    _require(len(d_coefficients) == 100
             and all(state["branches"] == {"D0", "Dgrad"} and len(state["labels"]) == 1
                     for state in d_coefficients.values()),
             "D0/Dgrad selected images do not form 100 exact same-coefficient pairs")

    # The P source loader rechecks C endpoint, saved logits, CSV identities,
    # and all 60,000 source metric rows before selecting the blind cards.
    config_path = ROOT / PHASE_CONFIG_RELATIVE
    p_config, p_paths = review_task._load_config(config_path)
    p_sources = review_task._load_sources(p_paths)
    _require(p_config["task_id"] == review_task.TASK_ID,
             "fixed P source configuration is for another review task")

    records: list[dict[str, Any]] = []
    coefficient_groups: dict[tuple[str, bytes], str] = {}

    def add_record(
        *, package: str, card: dict[str, Any], human_label: int | str, features: np.ndarray,
        source_identity: str, coefficient_key: bytes | None, method: str, branch: str,
        stream: int, cohort_row: int, component_id: int, proposal_id: int,
    ) -> None:
        _require(human_label == "unreadable" or (type(human_label) is int and human_label in LABELS),
                 f"invalid human label for {package}/{card['card_id']}")
        _require(features.shape == (23,) and np.isfinite(features).all(),
                 f"invalid fixed feature vector for {package}/{card['card_id']}")
        image_relative = str(({
            "R": R_REVIEW_RELATIVE,
            "D": D_REVIEW_RELATIVE,
            "P": P_REVIEW_RELATIVE,
        }[package] / str(card["image"])).as_posix())
        _require((ROOT / image_relative).is_file(), f"review image is missing: {image_relative}")
        if coefficient_key is None:
            group_id = f"{package}:singleton:{card['card_id']}"
        else:
            group_key = (package, coefficient_key)
            group_id = coefficient_groups.setdefault(group_key, f"{package}:coef_{len(coefficient_groups):04d}")
        records.append({
            "package": package, "card_id": str(card["card_id"]), "review_id": str(card.get("review_id", "")),
            "kind": "generated", "method": method, "branch": branch,
            "requested_label": int(card["requested_label"]), "human_label": human_label,
            "human_failure": bool(human_label == "unreadable" or human_label != int(card["requested_label"])),
            "stream": int(stream), "cohort_row": int(cohort_row), "component_id": int(component_id),
            "proposal_id": int(proposal_id), "source_identity": source_identity,
            "coefficient_group_id": group_id, "image_path": image_relative,
            "features": np.asarray(features, dtype=np.float64),
            "ed_logits": np.asarray(features[:10], dtype=np.float64),
            "ec_logits": np.asarray(features[10:20], dtype=np.float64),
        })

    for record in legacy_records:
        package = str(record["package"])
        if package == "R":
            card = r_cards[record["card_id"]]
            method = str(card["method"])
            index = int(card["cohort_row"])
            payload = r_payloads[method]
            _require(0 <= index < 10_000 and int(payload["requested_labels"][index]) == int(card["requested_label"])
                     and int(payload["proposal_ids"][index]) == int(card["proposal_id"])
                     and int(payload["component_ids"][index]) == int(card["component_id"]),
                     f"R card identity disagrees with its saved cohort: {card['card_id']}")
            features = record["features"]
            _require(np.array_equal(features[:20], _feature_array(
                payload["ed_logits"][index], payload["ec_logits"][index],
                {"view_mean_intensity": features[20], "raw_peak_w_per_pixel": features[21],
                 "proxy_clipping_area_fraction": features[22]})[:20]),
                f"R selected logits disagree with the explicit saved cohort: {card['card_id']}")
            coefficient_key = _exact_coefficient_key(payload["ols_coefficients"][index])
            add_record(package="R", card=card, human_label=record["human_label"], features=features,
                       source_identity=f"R:{method}:stream0:{card['requested_label']}:{card['proposal_id']}",
                       coefficient_key=coefficient_key, method=method, branch=method, stream=0,
                       cohort_row=index, component_id=int(card["component_id"]), proposal_id=int(card["proposal_id"]))
        else:
            card = d_cards[record["card_id"]]
            branch = str(card["branch"])
            index = int(card["cohort_row"])
            payload = d_payloads[branch]
            _require(int(payload["requested_labels"][index]) == int(card["requested_label"])
                     and int(payload["proposal_ids"][index]) == int(card["proposal_id"])
                     and int(payload["component_ids"][index]) == int(card.get("component_id", payload["component_ids"][index])),
                     f"D card identity disagrees with its saved cohort: {card['card_id']}")
            features = record["features"]
            coefficient_key = _exact_coefficient_key(payload["ols_coefficients"][index])
            add_record(package="D", card=card, human_label=record["human_label"], features=features,
                       source_identity=f"D:G4:stream0:{card['requested_label']}:{card['proposal_id']}",
                       coefficient_key=coefficient_key, method="G4", branch=branch, stream=0,
                       cohort_row=index, component_id=int(payload["component_ids"][index]),
                       proposal_id=int(card["proposal_id"]))

    p_cards = {card["card_id"]: card for card in p_key["cards"] if card.get("kind") == "generated"}
    for card_id, card in p_cards.items():
        method = str(card["method"])
        stream = int(card["stream"])
        label = int(card["requested_label"])
        index = int(card["cohort_row"])
        proposal = int(card["proposal_id"])
        component = int(card["component_id"])
        cohort = p_sources["cohorts"][(method, stream)]
        identity = (method, stream, label, proposal)
        metric = p_sources["metric_rows"].get(identity)
        _require(metric is not None and p_sources["identity_to_row"].get(identity) == index
                 and int(cohort["requested_labels"][index]) == label
                 and int(cohort["proposal_ids"][index]) == proposal
                 and int(cohort["component_ids"][index]) == component,
                 f"P card identity disagrees with its saved cohort: {card_id}")
        features = _feature_array(cohort["ed_logits"][index], cohort["ec_logits"][index], metric)
        _require(int(np.argmax(features[:10])) == int(metric["E_D_prediction"])
                 and int(np.argmax(features[10:20])) == int(metric["E_C_prediction"]),
                 f"P saved evaluator logits disagree with the registered metric row: {card_id}")
        coefficient_key = _exact_coefficient_key(cohort["ols_coefficients"][index])
        add_record(package="P", card=card, human_label=p_ratings[card_id], features=features,
                   source_identity=f"P:{method}:G4:{stream}:{label}:{proposal}",
                   coefficient_key=coefficient_key, method=method, branch="G4", stream=stream,
                   cohort_row=index, component_id=component, proposal_id=proposal)

    _require(len(records) == 1_600 and len({(row["package"], row["card_id"]) for row in records}) == 1_600,
             "generated human-review link table does not contain exactly 1,600 unique cards")
    counts = {package: sum(row["package"] == package for row in records) for package in PACKAGES}
    _require(counts == {"R": 200, "D": 200, "P": 1_200},
             f"fixed three-package populations changed: {counts}")
    for package in PACKAGES:
        for label in LABELS:
            group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in records:
                if row["package"] == package and row["requested_label"] == label:
                    group_rows[row["coefficient_group_id"]].append(row)
            _require(bool(group_rows) and all(all(item["requested_label"] == label for item in group)
                                              for group in group_rows.values()),
                     f"coefficient groups cross requested classes in {package}/{label}")
    d_groups: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["package"] == "D":
            d_groups[row["coefficient_group_id"]].add(row["branch"])
    _require(len(d_groups) == 100 and all(branches == {"D0", "Dgrad"} for branches in d_groups.values()),
             "D0/Dgrad same-coefficient rows were split across bootstrap groups")

    for row in records:
        row["review_id"] = str({"R": r_key["review_id"], "D": d_key["review_id"],
                                "P": p_key["review_id"]}[row["package"]])
    receipt = {
        "task_id": TASK_ID, "status": "PREPARED", "records": len(records), "package_counts": counts,
        "numeric_human_labels": sum(row["human_label"] != "unreadable" for row in records),
        "unreadable_human_labels": sum(row["human_label"] == "unreadable" for row in records),
        "human_failures": sum(row["human_failure"] for row in records),
        "D0_Dgrad_exact_coefficient_groups": len(d_groups),
        "hidden_duplicates_or_real_anchors_included": False,
        "feature_count": 23, "requested_label_is_feature": False,
        "cross_representation_proposal_ids_used_for_pairing": False,
        "P_template_review_decision": p_result["decision"],
        "legacy_package_source": legacy_receipt,
    }
    return records, receipt


def _fit_auxiliary_model(
    features: np.ndarray, human_labels: list[int | str], *, steps: int, device: torch.device,
) -> dict[str, Any]:
    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    _require(x.ndim == 2 and tuple(x.shape) == (len(human_labels), 23) and x.shape[0] > 0
             and torch.isfinite(x).all().item(), "auxiliary fit features are malformed")
    readable_target = torch.as_tensor([float(label != "unreadable") for label in human_labels],
                                      dtype=torch.float64, device=device)
    numeric_mask = readable_target.bool()
    digit_target = torch.as_tensor([0 if label == "unreadable" else int(label) for label in human_labels],
                                   dtype=torch.long, device=device)
    _require(bool(numeric_mask.any().item()), "auxiliary fit fold has no numeric human labels")
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    scale = torch.where(std > 0.0, std, torch.ones_like(std))
    standardized = (x - mean) / scale
    digit = torch.nn.Linear(23, 10, dtype=torch.float64, device=device)
    readable = torch.nn.Linear(23, 1, dtype=torch.float64, device=device)
    with torch.no_grad():
        digit.weight.zero_()
        digit.bias.zero_()
        readable.weight.zero_()
        readable.bias.zero_()
    initial_parameter_norm = 0.0
    optimizer = torch.optim.Adam(list(digit.parameters()) + list(readable.parameters()), lr=0.01,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    loss_value = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        digit_logits = digit(standardized[numeric_mask])
        readable_logits = readable(standardized).squeeze(1)
        weights = torch.cat((digit.weight.reshape(-1), readable.weight.reshape(-1)))
        loss = (F.cross_entropy(digit_logits, digit_target[numeric_mask])
                + F.binary_cross_entropy_with_logits(readable_logits, readable_target)
                + 0.01 * weights.square().mean())
        _require(torch.isfinite(loss).item(), "auxiliary fit objective became nonfinite")
        loss.backward()
        parameters = list(digit.parameters()) + list(readable.parameters())
        _require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                     for parameter in parameters), "auxiliary model received invalid gradients")
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    final_parameter_norm = math.sqrt(sum(float(parameter.detach().square().sum().cpu())
                                         for parameter in list(digit.parameters()) + list(readable.parameters())))
    model = {
        "mean": mean.detach(), "scale": scale.detach(),
        "digit_weight": digit.weight.detach().clone(), "digit_bias": digit.bias.detach().clone(),
        "readable_weight": readable.weight.detach().clone(), "readable_bias": readable.bias.detach().clone(),
        "training_loss": loss_value, "optimizer_steps": int(steps),
        "feature_count": 23, "dtype": "float64", "device": str(device),
        "initialization": "all_zero", "parameter_norm_after_fit": final_parameter_norm,
        "parameter_norm_before_fit": initial_parameter_norm,
    }
    _require(final_parameter_norm > initial_parameter_norm and math.isfinite(final_parameter_norm),
             "auxiliary parameters did not update during the authorized fit")
    return model


def _predict_auxiliary(
    model: dict[str, Any], features: np.ndarray, *, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    mean = model["mean"].to(device=device, dtype=torch.float64)
    scale = model["scale"].to(device=device, dtype=torch.float64)
    z = (x - mean) / scale
    digit_logits = F.linear(z, model["digit_weight"].to(device), model["digit_bias"].to(device))
    readable_logits = F.linear(z, model["readable_weight"].to(device), model["readable_bias"].to(device)).squeeze(1)
    digit_probability = torch.softmax(digit_logits, dim=1)
    readable_probability = torch.sigmoid(readable_logits)
    _require(torch.isfinite(digit_probability).all().item()
             and torch.isfinite(readable_probability).all().item(), "auxiliary prediction is nonfinite")
    return (digit_probability.detach().cpu().numpy(),
            readable_probability.detach().cpu().numpy())


def _model_for_save(model: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.detach().cpu() if torch.is_tensor(value) else value) for key, value in model.items()}


def _fit_preflight(records: list[dict[str, Any]], features: np.ndarray, output_dir: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is unavailable; auxiliary fitting cannot fall back to CPU")
    device = torch.device("cuda:0")
    probe = features[:80]
    labels = [row["human_label"] for row in records[:80]]
    _require(any(label != "unreadable" for label in labels) and any(label == "unreadable" for label in labels),
             "three-step preflight probe does not contain both readability outcomes")
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    model = _fit_auxiliary_model(probe, labels, steps=PREFLIGHT_STEPS, device=device)
    end_event.record()
    end_event.synchronize()
    _require(model["optimizer_steps"] == PREFLIGHT_STEPS and model["parameter_norm_after_fit"] > 0,
             "three-step CUDA preflight did not execute real auxiliary parameter updates")
    predicted, readable = _predict_auxiliary(model, probe, device=device)
    _require(predicted.shape == (80, 10) and readable.shape == (80,)
             and np.isfinite(predicted).all() and np.isfinite(readable).all(),
             "three-step CUDA preflight prediction shape is wrong")

    # Counterexample: high scores mean review risk.  The consumer must capture
    # none of the known failures at the high-risk end and all at the low-risk end.
    synthetic_failure = np.asarray([True, True] + [False] * 8)
    synthetic_risk = np.asarray([0.99, 0.95] + [0.2 - index * 0.01 for index in range(8)])
    top = np.argsort(-synthetic_risk, kind="stable")[:2]
    inverted_top = np.argsort(synthetic_risk, kind="stable")[:2]
    _require(synthetic_failure[top].all() and not synthetic_failure[inverted_top].any(),
             "risk-direction negative control failed")
    return {
        "task_id": TASK_ID, "status": "PASSED", "device": torch.cuda.get_device_name(device),
        "major_computation_device": "cuda", "dtype": "float64", "probe_rows": len(probe),
        "optimizer_steps": PREFLIGHT_STEPS, "parameter_norm_after_fit": model["parameter_norm_after_fit"],
        "cuda_event_ms": float(start_event.elapsed_time(end_event)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "prediction_rows": len(predicted), "risk_direction_negative_control": "PASSED",
        "label_leakage": False, "cpu_fallback": False,
    }


def _load_fit_input(output_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    path = output_dir / "auxiliary_fit_input.pt"
    _require(path.is_file(), f"fit input is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == SCHEMA and payload.get("task_id") == TASK_ID,
             "fit input schema or task identity is invalid")
    records = payload.get("records")
    features = torch.as_tensor(payload.get("features"), dtype=torch.float64).numpy()
    _require(isinstance(records, list) and features.shape == (1_600, 23)
             and len(records) == 1_600 and np.isfinite(features).all(),
             "fixed auxiliary population has a wrong shape or nonfinite features")
    _require([row["card_id"] for row in records] == payload.get("card_ids")
             and np.array_equal(np.stack([row["features"] for row in records]), features),
             "serialized feature rows and linked identities disagree")
    _validate_population(records)
    table = _read_csv(output_dir / "linked_population.csv")
    _require(len(table) == 1_600, "linked image/label/logit table is incomplete")
    for row, linked in zip(records, table, strict=True):
        _require(linked["package"] == row["package"] and linked["card_id"] == row["card_id"]
                 and linked["human_label"] == str(row["human_label"])
                 and linked["requested_label"] == str(row["requested_label"])
                 and linked["source_identity"] == row["source_identity"]
                 and linked["coefficient_group_id"] == row["coefficient_group_id"]
                 and linked["image_path"] == row["image_path"],
                 f"linked population row identity disagrees at {row['package']}/{row['card_id']}")
        linked_features = np.asarray([float(linked[name]) for name in FEATURE_NAMES], dtype=np.float64)
        _require(np.array_equal(linked_features, row["features"]),
                 f"linked saved logits/features disagree at {row['package']}/{row['card_id']}")
    return records, features


def _validate_population(records: list[dict[str, Any]]) -> None:
    _require(len(records) == 1_600 and all(row.get("package") in PACKAGES for row in records),
             "auxiliary population is incomplete or includes an unknown package")
    expected = {"R": 200, "D": 200, "P": 1_200}
    ids = [(row["package"], row["card_id"]) for row in records]
    counts = {package: sum(row["package"] == package for row in records) for package in PACKAGES}
    _require(len(set(ids)) == 1_600 and counts == expected,
             "auxiliary card identities or package counts changed")
    group_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in records:
        group_labels[(row["package"], row["coefficient_group_id"])].add(int(row["requested_label"]))
    _require(all(len(labels) == 1 for labels in group_labels.values()),
             "one coefficient group crosses requested-class bootstrap strata")
    for package in PACKAGES:
        for label in LABELS:
            subset = [row for row in records if row["package"] == package and row["requested_label"] == label]
            _require(subset and all(row["human_label"] == "unreadable"
                                    or (type(row["human_label"]) is int and row["human_label"] in LABELS)
                                    for row in subset), f"invalid labels in {package}/class_{label}")
            _require(all(row["human_failure"] == (row["human_label"] == "unreadable"
                                                    or row["human_label"] != label) for row in subset),
                     f"human failure derivation is wrong in {package}/class_{label}")
    d_groups: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["package"] == "D":
            d_groups[row["coefficient_group_id"]].add(row["branch"])
    _require(len(d_groups) == 100 and all(branches == {"D0", "Dgrad"} for branches in d_groups.values()),
             "D0/Dgrad coefficient-paired cards are not grouped")


def _prepare(output_dir: Path) -> None:
    records, receipt = _finalized_package_records()
    _validate_population(records)
    allowed_partial = {"auxiliary_fit_input.pt", "linked_population.csv.pending"}
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        _require((output_dir / "population_link_receipt.json").is_file() is False
                 and existing <= allowed_partial
                 and all((output_dir / name).is_file() for name in existing),
                 f"auxiliary output is already prepared or contains unrelated files: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    features = np.stack([row["features"] for row in records]).astype(np.float64, copy=False)
    input_path = output_dir / "auxiliary_fit_input.pt"
    if input_path.exists():
        previous = torch.load(input_path, map_location="cpu", weights_only=False)
        _require(previous.get("schema") == SCHEMA and previous.get("task_id") == TASK_ID
                 and previous.get("card_ids") == [row["card_id"] for row in records]
                 and np.array_equal(torch.as_tensor(previous.get("features"), dtype=torch.float64).numpy(), features),
                 "partial auxiliary fit input differs from the fixed linked population")
    else:
        _atomic_torch_save(input_path, {
            "schema": SCHEMA, "task_id": TASK_ID, "records": records,
            "features": torch.from_numpy(features.copy()),
            "card_ids": [row["card_id"] for row in records],
            "input_population": {"R": 200, "D": 200, "P": 1_200},
        })
    columns = ["package", "card_id", "review_id", "kind", "method", "branch", "requested_label",
               "human_label", "human_failure", "stream", "cohort_row", "component_id", "proposal_id",
               "source_identity", "coefficient_group_id", "image_path", *FEATURE_NAMES]
    _write_csv(output_dir / "linked_population.csv", [
        {**{key: value for key, value in row.items() if key not in {"features", "ed_logits", "ec_logits"}},
         **dict(zip(FEATURE_NAMES, row["features"].tolist(), strict=True))}
        for row in records
    ], columns)
    _write_json(output_dir / "run_config.json", {
        "schema": SCHEMA, "task_id": TASK_ID, "output_directory": str(OUTPUT_RELATIVE),
        "packages": {"R": {"records": 200, "review_id": "99d8041812f54d44b98031876fa21519"},
                     "D": {"records": 200, "review_id": "6346e07062f541cb904d65c016813b5c"},
                     "P": {"records": 1_200, "review_id": "5bdafe21b4cb475c8bd7a310bd481ce8"}},
        "features": list(FEATURE_NAMES), "optimizer_steps_per_fit": FIT_STEPS,
        "fit_plan": "leave one review package out for R, D, P; then one all-data development fit",
        "preflight_steps": PREFLIGHT_STEPS, "risk_score": "1 - p_readable * p_digit[requested_label]",
        "baseline_risk": "1 - softmax(saved_E_C_logits)[requested_label]",
        "primary": {"metric": "equal-package-weighted human-failure capture-rate difference at 20% budget",
                    "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
        "secondary": "E_D and 10%/30% review budgets are descriptive",
        "decision_rule": "candidate only if the 95% CI lower bound is greater than zero",
        "scientific_updates": 0, "optical_training_updates": 0, "new_sampling_rows": 0,
        "new_evaluator_forward_rows": 0, "official_test_reads": 0, "lpips_evaluations": 0,
        "full_development_fit_is_evaluation_evidence": False,
        "preparation_receipt": receipt,
    })
    _write_json(output_dir / "population_link_receipt.json", receipt)


def _read_model(path: Path) -> dict[str, Any]:
    model = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(model, dict) and model.get("feature_count") == 23
             and model.get("optimizer_steps") in {PREFLIGHT_STEPS, FIT_STEPS}
             and all(torch.is_tensor(model.get(key)) for key in
                     ("mean", "scale", "digit_weight", "digit_bias", "readable_weight", "readable_bias")),
             f"saved auxiliary model is malformed: {path}")
    return model


def _fit_one(
    records: list[dict[str, Any]], features: np.ndarray, train_indices: np.ndarray,
    held_indices: np.ndarray, output_dir: Path, *, fold: str, train_packages: list[str], resume: bool,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    fold_dir = output_dir / "folds" / f"heldout_{fold}"
    model_path = fold_dir / "model.pt"
    prediction_path = fold_dir / "predictions.csv"
    receipt_path = fold_dir / "fit_receipt.json"
    if receipt_path.is_file():
        _require(resume, f"completed fold already exists; use --resume to reuse it: {fold}")
        receipt = _read_json(receipt_path)
        _require(receipt.get("status") == "PASSED" and receipt.get("held_out_package") == fold
                 and receipt.get("training_packages") == train_packages
                 and receipt.get("training_rows") == len(train_indices)
                 and receipt.get("held_out_rows") == len(held_indices)
                 and model_path.is_file() and prediction_path.is_file(),
                 f"partial fold artifacts do not match the fixed fold: {fold}")
        return receipt

    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    model = _fit_auxiliary_model(
        features[train_indices], [records[index]["human_label"] for index in train_indices],
        steps=FIT_STEPS, device=device)
    digit_probability, readable_probability = _predict_auxiliary(model, features[held_indices], device=device)
    end_event.record()
    end_event.synchronize()
    _require(model["optimizer_steps"] == FIT_STEPS, f"fold did not complete 500 optimizer steps: {fold}")
    predictions: list[dict[str, Any]] = []
    for local, index in enumerate(held_indices.tolist()):
        row = records[index]
        predictions.append({
            "package": row["package"], "card_id": row["card_id"], "requested_label": row["requested_label"],
            "human_label": row["human_label"], "human_failure": int(row["human_failure"]),
            "fold": fold, "training_packages": "|".join(train_packages),
            "p_readable": float(readable_probability[local]),
            **{f"p_digit_{digit}": float(digit_probability[local, digit]) for digit in LABELS},
        })
    model_payload = _model_for_save(model)
    _atomic_torch_save(model_path, model_payload)
    prediction_fields = list(predictions[0]) if predictions else []
    _write_csv(prediction_path, predictions, prediction_fields)
    receipt = {
        "task_id": TASK_ID, "status": "PASSED", "fold": fold,
        "training_packages": train_packages, "held_out_package": fold,
        "training_rows": len(train_indices), "held_out_rows": len(held_indices),
        "numeric_training_rows": sum(records[int(i)]["human_label"] != "unreadable" for i in train_indices),
        "optimizer_steps": FIT_STEPS, "device": torch.cuda.get_device_name(device),
        "major_computation_device": "cuda", "dtype": "float64",
        "cuda_event_ms": float(start_event.elapsed_time(end_event)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameter_norm_after_fit": float(model["parameter_norm_after_fit"]),
        "prediction_rows": len(predictions), "training_normalization_rows": len(train_indices),
        "held_out_normalization_rows": 0,
        "normalization_mean": model_payload["mean"].tolist(),
        "normalization_scale": model_payload["scale"].tolist(),
    }
    _write_json(receipt_path, receipt)
    return receipt


def _fit_all(output_dir: Path, *, resume: bool) -> None:
    _require(torch.cuda.is_available(), "CUDA is unavailable; auxiliary fitting cannot fall back to CPU")
    records, features = _load_fit_input(output_dir)
    preflight_path = output_dir / "fit_preflight_receipt.json"
    _require(preflight_path.is_file(), "run the separate CUDA preflight before any 500-step fit")
    preflight = _read_json(preflight_path)
    _require(preflight.get("status") == "PASSED" and preflight.get("major_computation_device") == "cuda",
             "saved auxiliary preflight receipt is invalid")
    indices = np.arange(len(records), dtype=np.int64)
    fold_receipts = []
    for heldout in PACKAGES:
        held_indices = indices[np.asarray([row["package"] == heldout for row in records], dtype=bool)]
        train_indices = indices[np.asarray([row["package"] != heldout for row in records], dtype=bool)]
        training_packages = [package for package in PACKAGES if package != heldout]
        fold_receipts.append(_fit_one(records, features, train_indices, held_indices, output_dir,
                                      fold=heldout, train_packages=training_packages, resume=resume))

    full_dir = output_dir / "full_development"
    full_model_path = full_dir / "model.pt"
    full_receipt_path = full_dir / "fit_receipt.json"
    if full_receipt_path.is_file():
        _require(resume, "full development fit already exists; use --resume to reuse it")
        full_receipt = _read_json(full_receipt_path)
        _require(full_receipt.get("status") == "PASSED" and full_receipt.get("training_packages") == list(PACKAGES)
                 and full_receipt.get("training_rows") == len(records) and full_model_path.is_file(),
                 "saved full development fit does not match the fixed population")
    else:
        if resume:
            _require(not full_dir.exists() or not any(full_dir.iterdir()),
                     "partial full-development fit is present without its completed receipt")
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        full_model = _fit_auxiliary_model(
            features, [row["human_label"] for row in records], steps=FIT_STEPS, device=device)
        end_event.record()
        end_event.synchronize()
        full_payload = _model_for_save(full_model)
        _atomic_torch_save(full_model_path, full_payload)
        full_receipt = {
            "task_id": TASK_ID, "status": "PASSED", "fit_role": "development_model_only",
            "training_packages": list(PACKAGES), "training_rows": len(records),
            "optimizer_steps": FIT_STEPS, "device": torch.cuda.get_device_name(device),
            "major_computation_device": "cuda", "dtype": "float64",
            "cuda_event_ms": float(start_event.elapsed_time(end_event)),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "parameter_norm_after_fit": float(full_model["parameter_norm_after_fit"]),
            "in_sample_metrics_recorded": False, "acceptance_evidence": False,
        }
        _write_json(full_receipt_path, full_receipt)

    oof_rows: list[dict[str, Any]] = []
    for heldout in PACKAGES:
        fold_dir = output_dir / "folds" / f"heldout_{heldout}"
        receipt = _read_json(fold_dir / "fit_receipt.json")
        _require(receipt.get("status") == "PASSED", f"completed OOF fold has no valid receipt: {heldout}")
        oof_rows.extend(_read_csv(fold_dir / "predictions.csv"))
    _require(len(oof_rows) == 1_600 and len({(row["package"], row["card_id"]) for row in oof_rows}) == 1_600,
             "out-of-fold prediction population is incomplete or duplicated")
    fields = list(oof_rows[0])
    _write_csv(output_dir / "out_of_fold_predictions.csv", oof_rows, fields)
    receipt = {
        "task_id": TASK_ID, "status": "PASSED", "device": torch.cuda.get_device_name(0),
        "major_computation_device": "cuda", "cuda_available": True,
        "fit_count": 4, "fit_optimizer_steps": 4 * FIT_STEPS,
        "preflight_optimizer_steps": PREFLIGHT_STEPS,
        "oof_rows": len(oof_rows), "oof_folds": list(PACKAGES),
        "held_out_normalization_used": False,
        "full_development_model_is_evaluation_evidence": False,
        "fold_receipts": fold_receipts, "full_development_receipt": full_receipt,
    }
    _write_json(output_dir / "cuda_execution_receipt.json", receipt)


def _run_preflight(output_dir: Path, *, resume: bool) -> None:
    records, features = _load_fit_input(output_dir)
    receipt_path = output_dir / "fit_preflight_receipt.json"
    if receipt_path.exists():
        _require(resume, "CUDA preflight already exists; use --resume to verify and reuse it")
        receipt = _read_json(receipt_path)
        _require(receipt.get("status") == "PASSED" and receipt.get("major_computation_device") == "cuda"
                 and receipt.get("optimizer_steps") == PREFLIGHT_STEPS
                 and receipt.get("risk_direction_negative_control") == "PASSED",
                 "saved CUDA preflight is incomplete or invalid")
    else:
        receipt = _fit_preflight(records, features, output_dir)
        _write_json(receipt_path, receipt)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / exponents.sum(axis=1, keepdims=True)


def _capture_rate(indices: np.ndarray, risk: np.ndarray, failures: np.ndarray, fraction: float) -> dict[str, Any]:
    _require(indices.size > 0 and np.isfinite(risk[indices]).all(), "ranking input is empty or nonfinite")
    failure_count = int(failures[indices].sum())
    _require(failure_count > 0, "a review-package bootstrap sample contains no human failures")
    review_count = max(1, int(round(fraction * indices.size)))
    order = indices[np.argsort(-risk[indices], kind="stable")]
    captured = int(failures[order[:review_count]].sum())
    return {"review_count": review_count, "population_count": int(indices.size),
            "human_failure_count": failure_count, "captured_failure_count": captured,
            "failure_capture_rate": captured / failure_count}


def _scores(records: list[dict[str, Any]], predictions: list[dict[str, str]]) -> dict[str, np.ndarray]:
    record_lookup = {(row["package"], row["card_id"]): index for index, row in enumerate(records)}
    _require(len(record_lookup) == len(records), "input card identity is duplicated")
    prediction_lookup = {(row["package"], row["card_id"]): row for row in predictions}
    _require(len(prediction_lookup) == len(predictions) == len(records)
             and set(prediction_lookup) == set(record_lookup),
             "OOF predictions do not exactly cover the linked human-rating population")
    human_failure = np.asarray([row["human_failure"] for row in records], dtype=bool)
    requested = np.asarray([int(row["requested_label"]) for row in records], dtype=np.int64)
    ec_logits = np.stack([row["ec_logits"] for row in records]).astype(np.float64, copy=False)
    ed_logits = np.stack([row["ed_logits"] for row in records]).astype(np.float64, copy=False)
    ec_probability = _softmax(ec_logits)
    ed_probability = _softmax(ed_logits)
    aux_probability = np.empty((len(records), 10), dtype=np.float64)
    aux_readable = np.empty(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        row = prediction_lookup[(record["package"], record["card_id"])]
        expected_human_label = ("unreadable" if record["human_label"] == "unreadable"
                                else str(int(record["human_label"])))
        _require(row["human_label"] == expected_human_label and int(row["requested_label"]) == requested[index]
                 and int(row["human_failure"]) == int(human_failure[index])
                 and row["fold"] == record["package"],
                 f"OOF prediction target or fold identity differs: {record['package']}/{record['card_id']}")
        aux_readable[index] = float(row["p_readable"])
        aux_probability[index] = [float(row[f"p_digit_{label}"]) for label in LABELS]
        _require(np.isfinite(aux_probability[index]).all() and np.isfinite(aux_readable[index])
                 and abs(aux_probability[index].sum() - 1.0) <= 1e-10
                 and 0.0 <= aux_readable[index] <= 1.0,
                 f"OOF probability vector is invalid: {record['package']}/{record['card_id']}")
    rows = np.arange(len(records), dtype=np.int64)
    return {
        "human_failure": human_failure,
        "auxiliary": 1.0 - aux_readable * aux_probability[rows, requested],
        "E_C": 1.0 - ec_probability[rows, requested],
        "E_D": 1.0 - ed_probability[rows, requested],
    }


def _point_summary(records: list[dict[str, Any]], score: dict[str, np.ndarray]) -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for package in PACKAGES:
        indices = np.asarray([i for i, row in enumerate(records) if row["package"] == package], dtype=np.int64)
        packages[package] = {
            "population_count": int(indices.size),
            "human_failure_count": int(score["human_failure"][indices].sum()),
            "human_failure_rate": float(score["human_failure"][indices].mean()),
            "budgets": {},
        }
        for fraction in REVIEW_FRACTIONS:
            packages[package]["budgets"][f"{int(fraction * 100)}%"] = {
                method: _capture_rate(indices, score[method], score["human_failure"], fraction)
                for method in ("auxiliary", "E_C", "E_D")
            }
    primary = float(np.mean([
        packages[package]["budgets"]["20%"]["auxiliary"]["failure_capture_rate"]
        - packages[package]["budgets"]["20%"]["E_C"]["failure_capture_rate"]
        for package in PACKAGES
    ]))
    return {"package_metrics": packages,
            "primary_difference_auxiliary_minus_E_C": primary,
            "package_equal_weighting": True,
            "primary_review_fraction": PRIMARY_FRACTION}


def _grouped_paired_bootstrap(
    records: list[dict[str, Any]], score: dict[str, np.ndarray], *, resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(records):
        groups[(row["package"], int(row["requested_label"]))][row["coefficient_group_id"]].append(index)
    draws = np.empty(resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for iteration in range(resamples):
        package_indices: dict[str, list[int]] = {package: [] for package in PACKAGES}
        for package in PACKAGES:
            for label in LABELS:
                strata = groups.get((package, label), {})
                group_rows = list(strata.values())
                _require(bool(group_rows), f"bootstrap coefficient stratum is empty: {package}/{label}")
                if len({len(group) for group in group_rows}) != 1:
                    raise AuxiliaryRankError(
                        f"coefficient group sizes vary within bootstrap stratum {package}/{label}")
                picked = rng.integers(0, len(group_rows), size=len(group_rows))
                for index in picked:
                    package_indices[package].extend(group_rows[int(index)])
        differences = []
        for package in PACKAGES:
            sample = np.asarray(package_indices[package], dtype=np.int64)
            auxiliary_rate = _capture_rate(sample, score["auxiliary"], score["human_failure"], PRIMARY_FRACTION)
            ec_rate = _capture_rate(sample, score["E_C"], score["human_failure"], PRIMARY_FRACTION)
            differences.append(auxiliary_rate["failure_capture_rate"] - ec_rate["failure_capture_rate"])
        draws[iteration] = float(np.mean(differences))
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear").tolist()
    return {"bootstrap_seed": seed, "bootstrap_resamples": resamples,
            "strata": "review package x requested class; exact coefficient groups resampled together",
            "paired_methods": ["auxiliary", "E_C"], "confidence_interval_method": "percentile 95%",
            "ci_95": [float(lower), float(upper)],
            "bootstrap_mean": float(draws.mean()),
            "bootstrap_draws_min_max": [float(draws.min()), float(draws.max())]}


def _independent_bootstrap_check(
    records: list[dict[str, Any]], score: dict[str, np.ndarray], expected: dict[str, Any],
    *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    # Independent reference implementation constructs one resampled index
    # vector per package/class before ranking either method on that same draw.
    grouped: dict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    key_to_indices: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        key_to_indices[(row["package"], int(row["requested_label"]), row["coefficient_group_id"])].append(index)
    for (package, label, _group), index_list in key_to_indices.items():
        grouped[(package, label)].append(tuple(index_list))
    generator = np.random.default_rng(seed)
    reference = np.zeros(resamples, dtype=np.float64)
    for replicate in range(resamples):
        by_package: dict[str, list[int]] = {package: [] for package in PACKAGES}
        for package in PACKAGES:
            for label in LABELS:
                units = grouped[(package, label)]
                selected_units = generator.integers(0, len(units), len(units))
                for unit_index in selected_units:
                    by_package[package].extend(units[int(unit_index)])
        package_effects = []
        for package in PACKAGES:
            idx = np.asarray(by_package[package], dtype=np.int64)
            failures = score["human_failure"][idx]
            if int(failures.sum()) == 0:
                raise AuxiliaryRankError("independent bootstrap encountered a zero-failure package")
            budget = int(round(PRIMARY_FRACTION * len(idx)))
            aux_order = idx[np.argsort(-score["auxiliary"][idx], kind="stable")]
            ec_order = idx[np.argsort(-score["E_C"][idx], kind="stable")]
            aux_caught = int(score["human_failure"][aux_order[:budget]].sum())
            ec_caught = int(score["human_failure"][ec_order[:budget]].sum())
            package_effects.append((aux_caught - ec_caught) / int(failures.sum()))
        reference[replicate] = float(sum(package_effects) / len(package_effects))
    lower, upper = np.quantile(reference, [0.025, 0.975], method="linear").tolist()
    expected_ci = expected["ci_95"]
    _require(np.allclose([lower, upper], expected_ci, rtol=0.0, atol=1e-15),
             "independent bootstrap interval disagrees with primary result")
    return {"task_id": TASK_ID, "status": "PASSED", "resamples": len(reference),
            "seed": seed, "independent_ci_95": [float(lower), float(upper)],
            "primary_ci_match": True, "independent_algorithm": "group-first paired resampling and direct rank census"}


def _finalize(output_dir: Path) -> None:
    records, features = _load_fit_input(output_dir)
    cuda = _read_json(output_dir / "cuda_execution_receipt.json")
    _require(cuda.get("status") == "PASSED" and cuda.get("major_computation_device") == "cuda"
             and cuda.get("fit_count") == 4 and cuda.get("fit_optimizer_steps") == 2_000
             and cuda.get("oof_rows") == 1_600,
             "final consumer requires complete CUDA evidence for exactly four fixed fits")
    for heldout in PACKAGES:
        fold_dir = output_dir / "folds" / f"heldout_{heldout}"
        receipt = _read_json(fold_dir / "fit_receipt.json")
        model = _read_model(fold_dir / "model.pt")
        train_mask = np.asarray([row["package"] != heldout for row in records], dtype=bool)
        held_mask = ~train_mask
        train_packages = [package for package in PACKAGES if package != heldout]
        expected_mean = features[train_mask].mean(axis=0)
        expected_scale = features[train_mask].std(axis=0)
        expected_scale[expected_scale == 0.0] = 1.0
        _require(receipt.get("held_out_package") == heldout
                 and receipt.get("training_packages") == train_packages
                 and receipt.get("major_computation_device") == "cuda"
                 and receipt.get("optimizer_steps") == FIT_STEPS
                 and receipt.get("training_rows") == int(train_mask.sum())
                 and receipt.get("held_out_rows") == int(held_mask.sum())
                 and receipt.get("held_out_normalization_rows") == 0
                 and model.get("optimizer_steps") == FIT_STEPS
                 and np.allclose(model["mean"].numpy(), expected_mean, rtol=1e-12, atol=1e-12)
                 and np.allclose(model["scale"].numpy(), expected_scale, rtol=1e-12, atol=1e-12),
                 f"fold model normalization or package boundary is invalid: {heldout}")
    full_receipt = _read_json(output_dir / "full_development" / "fit_receipt.json")
    full_model = _read_model(output_dir / "full_development" / "model.pt")
    _require(full_receipt.get("training_packages") == list(PACKAGES)
             and full_receipt.get("training_rows") == len(records)
             and full_receipt.get("optimizer_steps") == FIT_STEPS
             and full_receipt.get("acceptance_evidence") is False
             and full_model.get("optimizer_steps") == FIT_STEPS,
             "full-data development model is incomplete or marked as evaluation evidence")
    prediction_rows = _read_csv(output_dir / "out_of_fold_predictions.csv")
    for row in prediction_rows:
        fold = row.get("fold")
        expected_training_packages = "|".join(package for package in PACKAGES if package != fold)
        _require(fold in PACKAGES and row.get("package") == fold
                 and row.get("training_packages") == expected_training_packages,
                 f"OOF row is not produced by the model that held its package out: {row}")
    score = _scores(records, prediction_rows)
    points = _point_summary(records, score)
    bootstrap = _grouped_paired_bootstrap(records, score)
    independent = _independent_bootstrap_check(records, score, bootstrap)
    lower, upper = bootstrap["ci_95"]
    if lower > 0.0:
        decision = "AUXILIARY_RANKING_CANDIDATE_SUPPORTED"
        interpretation = "折外辅助排序在固定20%预算下的失败捕获率提高，保留为后续独立验证候选。"
    else:
        decision = "SIMPLE_RANKING_BASELINE_RETAINED"
        interpretation = "本轮未确认辅助排序优于E_C；保留E_C简单排序基线，不追加拟合或评分。"
    result = {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC", "decision": decision,
        "interpretation": interpretation, "primary_metric": points,
        "primary_bootstrap": bootstrap,
        "gate": {"rule": "lower 95% CI > 0", "passed": lower > 0.0,
                 "candidate_only": True, "model_adoption": False},
        "secondary_descriptive_only": {
            "E_D": {package: points["package_metrics"][package]["budgets"] for package in PACKAGES},
            "budgets_10_20_30_percent": True,
            "no_secondary_acceptance_decision": True,
        },
        "population": {"records": len(records), "packages": {p: sum(r["package"] == p for r in records)
                                                               for p in PACKAGES},
                       "human_failures": int(score["human_failure"].sum()),
                       "features": 23, "hidden_duplicates_or_real_anchors_in_fit": 0},
        "fit_scope": {"packet_held_out_fits": 3, "development_fit": 1,
                      "optimizer_steps_each": FIT_STEPS, "total_authorized_fit_steps": 2_000,
                      "all_fits_cuda": True, "development_fit_evaluation_evidence": False},
        "independent_validation": independent,
        "scientific_updates": 0, "optical_training_updates": 0, "new_sampling_rows": 0,
        "encoder_forward_rows": 0, "evaluator_forward_rows": 0,
        "official_test_reads": 0, "lpips_evaluations": 0,
        "system_status_preserved": "VALID_FAILED",
    }
    _write_json(output_dir / "ranking_result.json", result)
    _write_json(output_dir / "independent_ranking_recalculation.json", independent)
    _write_report(output_dir, result)


def _write_report(output_dir: Path, result: dict[str, Any]) -> None:
    primary = result["primary_metric"]
    ci = result["primary_bootstrap"]["ci_95"]
    package = primary["package_metrics"]
    text = f"""结果与状态：{result['status']}；主要判定为`{result['decision']}`。辅助模型四次CUDA拟合及折外消费者完成。固定20%人工复核预算下，三个评阅包等权的失败捕获率差（辅助排序减E_C）为{primary['primary_difference_auxiliary_minus_E_C']:.4f}，95%区间[{ci[0]:.4f}, {ci[1]:.4f}]。模板G4仍按原标准判为收益未明确。

科学认识：{result['interpretation']} 结果仅适用于R、D、P三个已完成评阅包和固定C端点。D0/Dgrad的完全相同系数样本作为一组重采样；跨表示proposal编号不用于配对。折外预测只检验排序开发价值，不构成生成质量、模型采用或自动评阅器验收证据。

目标缺口：语义生成仍缺少跨训练轨迹与跨评阅者的独立可读性确认，以及真实光路/器件输入的验证。本轮辅助排序不能替代盲评或语义生成质量验收。

工程完成与验证：人口为R={package['R']['population_count']}、D={package['D']['population_count']}、P={package['P']['population_count']}，共1,600张生成图；隐藏重复和真实参照未进入拟合。完成三次留一评阅包500步拟合、一次全量开发500步拟合、CUDA三步更新预检、风险方向负控、折外逐行回读、训练折标准化核对和独立配对bootstrap复算。无光学训练、新生成、评价器新前向、TEST读取或LPIPS。包级失败捕获率：R辅助={package['R']['budgets']['20%']['auxiliary']['failure_capture_rate']:.4f}/E_C={package['R']['budgets']['20%']['E_C']['failure_capture_rate']:.4f}；D辅助={package['D']['budgets']['20%']['auxiliary']['failure_capture_rate']:.4f}/E_C={package['D']['budgets']['20%']['E_C']['failure_capture_rate']:.4f}；P辅助={package['P']['budgets']['20%']['auxiliary']['failure_capture_rate']:.4f}/E_C={package['P']['budgets']['20%']['E_C']['failure_capture_rate']:.4f}。独立统计回读状态：{result['independent_validation']['status']}。

决策影响与下一步：只有主要95%区间下界大于0时，建议保留轻量辅助排序作为候选，并另行设计独立的人眼确认；若区间包含0或低于0，保留E_C简单排序，不宣称人眼标签带来排序增量。该模型不进入光学训练损失，不自动扩样、追加人工评分或采用模型。本轮授权止于评分收尾、折外排序验证和结果归档。
"""
    path = output_dir / "closeout_report.md"
    pending = path.with_name(path.name + ".pending")
    pending.write_text(text, encoding="utf-8")
    os.replace(pending, path)


def _validate_final_package(output_dir: Path) -> None:
    records, features = _load_fit_input(output_dir)
    cuda = _read_json(output_dir / "cuda_execution_receipt.json")
    preflight = _read_json(output_dir / "fit_preflight_receipt.json")
    transfer = _read_json(output_dir / "transfer_receipt.json")
    result = _read_json(output_dir / "ranking_result.json")
    independent = _read_json(output_dir / "independent_ranking_recalculation.json")
    _require(preflight.get("status") == "PASSED" and preflight.get("major_computation_device") == "cuda"
             and preflight.get("optimizer_steps") == PREFLIGHT_STEPS
             and preflight.get("risk_direction_negative_control") == "PASSED",
             "Windows readback requires the completed CUDA preflight receipt")
    _require(cuda.get("status") == "PASSED" and cuda.get("major_computation_device") == "cuda"
             and cuda.get("fit_count") == 4 and cuda.get("fit_optimizer_steps") == 2_000
             and cuda.get("oof_rows") == 1_600,
             "Windows readback requires all four CUDA fit receipts")
    _require(transfer.get("status") == "COMPLETE" and len(transfer.get("files", [])) == 20
             and all(row.get("status") == "COPIED"
                     and row.get("remote_size_bytes") == row.get("local_size_bytes")
                     for row in transfer["files"]),
             "Windows readback requires the complete exact-file transfer receipt")
    prediction_rows = _read_csv(output_dir / "out_of_fold_predictions.csv")
    for row in prediction_rows:
        fold = row.get("fold")
        training = "|".join(package for package in PACKAGES if package != fold)
        _require(fold in PACKAGES and row.get("package") == fold and row.get("training_packages") == training,
                 "Windows readback found a prediction outside its held-out package")
    score = _scores(records, prediction_rows)
    recomputed_point = _point_summary(records, score)
    _require(result.get("status") == "COMPLETE_DIAGNOSTIC"
             and result.get("primary_metric") == recomputed_point
             and np.allclose(result.get("primary_bootstrap", {}).get("ci_95", []),
                             independent.get("independent_ci_95", []), rtol=0.0, atol=1e-15)
             and independent.get("status") == "PASSED" and independent.get("primary_ci_match") is True,
             "Windows readback did not reproduce the saved primary statistic or independent interval")
    report = (output_dir / "closeout_report.md").read_text(encoding="utf-8")
    headings = ("结果与状态：", "科学认识：", "目标缺口：", "工程完成与验证：", "决策影响与下一步：")
    positions = [report.find(heading) for heading in headings]
    _require(all(position >= 0 for position in positions) and positions == sorted(positions),
             "closeout report does not follow the required five-part feedback order")
    for fold in PACKAGES:
        fold_dir = output_dir / "folds" / f"heldout_{fold}"
        receipt = _read_json(fold_dir / "fit_receipt.json")
        model = _read_model(fold_dir / "model.pt")
        train_mask = np.asarray([row["package"] != fold for row in records], dtype=bool)
        expected_mean = features[train_mask].mean(axis=0)
        expected_scale = features[train_mask].std(axis=0)
        expected_scale[expected_scale == 0.0] = 1.0
        _require(receipt.get("held_out_package") == fold
                 and receipt.get("training_packages") == [package for package in PACKAGES if package != fold]
                 and receipt.get("major_computation_device") == "cuda"
                 and receipt.get("optimizer_steps") == FIT_STEPS
                 and receipt.get("held_out_normalization_rows") == 0
                 and model.get("optimizer_steps") == FIT_STEPS
                 and np.allclose(model["mean"].numpy(), expected_mean, rtol=1e-12, atol=1e-12)
                 and np.allclose(model["scale"].numpy(), expected_scale, rtol=1e-12, atol=1e-12)
                 and len(_read_csv(fold_dir / "predictions.csv")) == receipt.get("held_out_rows"),
                 f"Windows readback failed the held-out model or normalization check: {fold}")
    full_receipt = _read_json(output_dir / "full_development" / "fit_receipt.json")
    full_model = _read_model(output_dir / "full_development" / "model.pt")
    _require(full_receipt.get("training_packages") == list(PACKAGES)
             and full_receipt.get("training_rows") == len(records)
             and full_receipt.get("optimizer_steps") == FIT_STEPS
             and full_receipt.get("acceptance_evidence") is False
             and full_model.get("optimizer_steps") == FIT_STEPS
             and (output_dir / "resource_telemetry.csv").is_file()
             and features.shape == (1_600, 23),
             "Windows readback is missing a development model, telemetry, or fixed population")
    _write_json(output_dir / "windows_local_readback_receipt.json", {
        "task_id": TASK_ID, "status": "PASSED", "consumer_host": "Windows",
        "linked_population_rows": len(records), "feature_count": int(features.shape[1]),
        "package_counts": {package: sum(row["package"] == package for row in records) for package in PACKAGES},
        "out_of_fold_rows_recomputed": len(prediction_rows),
        "four_cuda_fits_and_2_000_steps_verified": True,
        "held_out_normalization_and_package_boundaries_verified": True,
        "primary_statistic_and_independent_bootstrap_verified": True,
        "transfer_receipt_status": transfer["status"],
        "decision": result["decision"],
        "system_status_preserved": "VALID_FAILED",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("prepare", "preflight", "fit", "finalize", "validate"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_RELATIVE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.stage == "prepare":
        _prepare(output_dir)
    elif args.stage == "preflight":
        _run_preflight(output_dir, resume=args.resume)
    elif args.stage == "fit":
        _fit_all(output_dir, resume=args.resume)
    elif args.stage == "finalize":
        _finalize(output_dir)
    else:
        _validate_final_package(output_dir)


if __name__ == "__main__":
    main()
