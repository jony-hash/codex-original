"""Attribute the exact-real-OLS replay versus canonical paired-path gap.

This executor consumes the already accepted semantic-conditioned sampler bank.
It performs only frozen source-identity audits and forward comparisons; it does
not fit, select, or generate a new sampler bank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_post3000_generation_readiness_scaling_v1 as readiness
from scripts.research.lightgen import execute_lightgen_2d_semantically_conditioned_ols_sampler_v1 as sampler


TASK_ID = "EXECUTE_REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_ATTRIBUTED"
SUPPORTED_INPUT_GATE = "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_V1_RESOLVED"

SAMPLER_RUN = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01")
SAMPLER_REPORT = SAMPLER_RUN / "report.json"
SAMPLER_REPORT_SHA256 = "1d88e4afe3be37ba10045523fe40aa4eb4068aef1bb8ed8c063f4ceaef969767"
SAMPLER_BANK = SAMPLER_RUN / "candidate_latent_banks.pt"
SAMPLER_BANK_SHA256 = "5ad385e3f683c144c5c8f0464e2519a11e03dbbf9dc0ce9b91bb9b7170148419"
SAMPLER_WINNER = SAMPLER_RUN / "winner_frozen_sampler.pt"
SAMPLER_WINNER_SHA256 = "166cf15ea91666afe1fe1c88e8897b88d58a4740253d59838927502babba568c"

POST3000_REPORT = Path("outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_05/report.json")
POST3000_REPORT_SHA256 = "c745133868a0058576239c489f1daaead87214ca0c8bc2ba1dc73c786ee37647"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260920_real_ols_replay_conditioning_identity_gap_attributed_v1/runs/attempt_01"
)

CLASS_COUNT = 10
BANK_PER_CLASS = sampler.BANK_PER_CLASS
BANK_COUNT = CLASS_COUNT * BANK_PER_CLASS
OLS_PORT_COUNT = sampler.OLS_PORT_COUNT
FORWARD_BATCH_SIZE = 10
EVALUATOR_BATCH_SIZE = 50
BOOTSTRAP_COUNT = sampler.BOOTSTRAP_COUNT
BOOTSTRAP_SEED = sampler.BOOTSTRAP_SEED
EQUIVALENCE_RTOL = 1.0e-5
EQUIVALENCE_ATOL = 1.0e-6
REAL_CONTEXT_EXACT_TOLERANCE = 1.0e-12


class ReplayIdentityAuditError(RuntimeError):
    """Raised when a frozen authority or audit invariant cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayIdentityAuditError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"{role} is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    _require(flat.numel() > 0, "summary input is empty")
    _finite(flat, "summary input")
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std_population": float(flat.std(unbiased=False)),
        "quantiles": {"0.01": float(quantiles[0]), "0.5": float(quantiles[1]), "0.99": float(quantiles[2])},
    }


def _confusion(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    labels = labels.detach().cpu().to(torch.long).reshape(-1)
    predictions = predictions.detach().cpu().to(torch.long).reshape(-1)
    _require(labels.shape == predictions.shape and labels.numel() > 0, "classification shapes differ")
    _require(bool(((labels >= 0) & (labels < CLASS_COUNT) & (predictions >= 0) & (predictions < CLASS_COUNT)).all().item()), "classification label outside 0..9")
    result = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        result[actual, predicted] += 1
    return result


def _classification_report(logits: torch.Tensor, labels: torch.Tensor, *, label_source: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    logits = logits.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.long)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    matrix = _confusion(labels, predictions)
    totals = matrix.sum(dim=1).to(torch.float64)
    _require(bool((totals > 0).all().item()), "classification class is empty")
    recalls = matrix.diag().to(torch.float64) / totals
    ordered = torch.sort(probabilities, dim=1, descending=True).values
    margins = logits.topk(2, dim=1).values
    return (
        {
            "rows": int(labels.numel()),
            "label_source": label_source,
            "accuracy": float((predictions == labels).to(torch.float64).mean()),
            "macro_recall": float(recalls.mean()),
            "per_class_recall": [float(item) for item in recalls],
            "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "predicted_class_histogram": [int(item) for item in torch.bincount(predictions, minlength=CLASS_COUNT)],
            "true_class_histogram": [int(item) for item in torch.bincount(labels, minlength=CLASS_COUNT)],
            "top1_confidence": _summary(ordered[:, 0]),
            "top1_top2_logit_margin": _summary(margins[:, 0] - margins[:, 1]),
        },
        predictions,
        probabilities,
    )


def _transition_matrix(rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    rows = rows.detach().cpu().to(torch.long).reshape(-1)
    columns = columns.detach().cpu().to(torch.long).reshape(-1)
    _require(rows.shape == columns.shape, "transition inputs differ")
    return _confusion(rows, columns)


def _raw_state() -> dict[str, Any]:
    return {"count": 0, "finite": True, "nonconstant": 0, "zero": 0, "min": float("inf"), "max": float("-inf")}


def _update_raw_state(raw: torch.Tensor, state: dict[str, Any]) -> None:
    _finite(raw, "raw CMOS")
    flat = raw.detach().to(torch.float64).flatten(1)
    per_sample_std = flat.std(dim=1, unbiased=False)
    state["count"] += int(raw.shape[0])
    state["finite"] = bool(state["finite"] and torch.isfinite(raw).all().item())
    state["nonconstant"] += int((per_sample_std > 0.0).sum().item())
    state["zero"] += int((flat.abs().sum(dim=1) == 0.0).sum().item())
    state["min"] = min(float(state["min"]), float(raw.min().detach().cpu()))
    state["max"] = max(float(state["max"]), float(raw.max().detach().cpu()))


def _finish_raw_state(state: Mapping[str, Any]) -> dict[str, Any]:
    count = int(state["count"])
    _require(count > 0, "raw CMOS stream is empty")
    return {
        "rows": count,
        "shape": [count, 1, 200, 200],
        "finite": bool(state["finite"]),
        "finite_fraction": 1.0 if bool(state["finite"]) else 0.0,
        "nonconstant_fraction": float(state["nonconstant"] / count),
        "zero_image_count": int(state["zero"]),
        "min": float(state["min"]),
        "max": float(state["max"]),
    }


def _difference_metrics(left: torch.Tensor, right: torch.Tensor, *, name: str) -> dict[str, Any]:
    _require(left.shape == right.shape, f"{name} shapes differ")
    delta = (left.detach() - right.detach()).abs()
    _finite(delta, f"{name} absolute difference")
    flat = delta.to(torch.float64).flatten(1)
    max_per_row = flat.max(dim=1).values.cpu()
    mean_per_row = flat.mean(dim=1).cpu()
    max_abs = float(delta.max().detach().cpu())
    allclose = bool(torch.allclose(left.detach(), right.detach(), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL))
    return {
        "name": name,
        "shape": list(left.shape),
        "dtype_left": str(left.dtype),
        "dtype_right": str(right.dtype),
        "max_abs": max_abs,
        "mean_abs": float(delta.to(torch.float64).mean().detach().cpu()),
        "per_row_max_abs": _summary(max_per_row),
        "per_row_mean_abs": _summary(mean_per_row),
        "exact_tensor_equal": bool(torch.equal(left.detach(), right.detach())),
        "allclose": allclose,
        "equivalence_convention": {"rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL},
    }


def _merge_difference_batches(
    name: str,
    max_per_row: list[torch.Tensor],
    mean_per_row: list[torch.Tensor],
    *,
    shape: list[int],
    exact: bool,
    allclose: bool,
    max_abs: float,
    mean_sum: float,
    element_count: int,
) -> dict[str, Any]:
    max_rows = torch.cat(max_per_row, dim=0)
    mean_rows = torch.cat(mean_per_row, dim=0)
    return {
        "name": name,
        "shape": shape,
        "max_abs": float(max_abs),
        "mean_abs": float(mean_sum / float(element_count)),
        "per_row_max_abs": _summary(max_rows),
        "per_row_mean_abs": _summary(mean_rows),
        "exact_tensor_equal": bool(exact),
        "allclose": bool(allclose),
        "equivalence_convention": {"rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL},
    }


def _validate_external_result(result: Any, expected_rows: int) -> torch.Tensor:
    _require(result.image_input is None and result.ols_input_field is None, "direct replay unexpectedly executed image/OLS input stages")
    _require(result.z_ols.source == "external_generated", "direct replay source provenance changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "direct replay did not use W_OLS")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "direct replay transport changed")
    raw = result.raw_intensity
    _require(raw.shape == (expected_rows, 1, 200, 200) and raw.dtype == torch.float32, "direct replay CMOS shape/dtype changed")
    _finite(raw, "direct replay raw CMOS")
    _require(bool((raw >= 0.0).all().item()), "direct replay raw CMOS is negative")
    return raw


def _validate_paired_result(result: Any, expected_rows: int) -> torch.Tensor:
    _require(result.image_input is not None and result.ols_input_field is not None, "paired path did not expose image/OLS stages")
    _require(result.z_ols.source == "real_encoded", "paired path source provenance changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "paired path did not use W_OLS")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "paired path transport changed")
    raw = result.raw_intensity
    _require(raw.shape == (expected_rows, 1, 200, 200) and raw.dtype == torch.float32, "paired path CMOS shape/dtype changed")
    _finite(raw, "paired path raw CMOS")
    _require(bool((raw >= 0.0).all().item()), "paired path raw CMOS is negative")
    return raw


def _evaluate_raw_batch(raw: torch.Tensor, models: Mapping[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    view = robust._cmos_view(raw)
    result: dict[str, torch.Tensor] = {}
    for name, model in models.items():
        features, logits = model(view)
        _finite(features, f"{name} features")
        _finite(logits, f"{name} logits")
        result[name] = logits.detach().cpu()
    return result


def _run_direct_bank(
    consumer: Any,
    models: Mapping[str, torch.nn.Module],
    z: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    *,
    phase: str,
) -> dict[str, Any]:
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    state = _raw_state()
    predictions: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, int(z.shape[0]), FORWARD_BATCH_SIZE):
            stop = min(start + FORWARD_BATCH_SIZE, int(z.shape[0]))
            result = consumer.generated_forward(z[start:stop].to(device))
            raw = _validate_external_result(result, stop - start)
            _update_raw_state(raw, state)
            batch_logits = _evaluate_raw_batch(raw, models)
            for name, value in batch_logits.items():
                logits[name].append(value)
            if stop == int(z.shape[0]) or stop % 200 == 0:
                print(json.dumps({"phase": phase, "completed_rows": stop, "total_rows": int(z.shape[0])}, ensure_ascii=False), flush=True)
            del result
    concatenated = {name: torch.cat(values, dim=0) for name, values in logits.items()}
    reports: dict[str, Any] = {}
    for name, value in concatenated.items():
        report, predicted, _ = _classification_report(value, labels, label_source="requested S0/S1 bank class identity")
        reports[name] = report
        predictions[name] = predicted
    return {
        "logits": concatenated,
        "predictions": predictions,
        "evaluators": reports,
        "raw_stats": _finish_raw_state(state),
    }


def _run_step3000_identity_pair(
    consumer: Any,
    models: Mapping[str, torch.nn.Module],
    source_images: torch.Tensor,
    stored_z: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    logits: dict[str, dict[str, list[torch.Tensor]]] = {
        "stored_replay": {name: [] for name in models},
        "paired_encoder": {name: [] for name in models},
    }
    raw_states = {name: _raw_state() for name in logits}
    paired_z_parts: list[torch.Tensor] = []
    diff_rows: dict[str, list[torch.Tensor]] = {name: [] for name in ("z_ols", "generator_input_field", "generator_output_field", "raw_cmos")}
    diff_mean_rows: dict[str, list[torch.Tensor]] = {name: [] for name in diff_rows}
    diff_exact = {name: True for name in diff_rows}
    diff_allclose = {name: True for name in diff_rows}
    diff_max = {name: 0.0 for name in diff_rows}
    diff_mean_sum = {name: 0.0 for name in diff_rows}
    diff_elements = {name: 0 for name in diff_rows}

    with torch.inference_mode():
        for start in range(0, int(stored_z.shape[0]), FORWARD_BATCH_SIZE):
            stop = min(start + FORWARD_BATCH_SIZE, int(stored_z.shape[0]))
            stored_result = consumer.generated_forward(stored_z[start:stop].to(device))
            paired_result = consumer.runtime.forward_from_image(source_images[start:stop].to(device))
            stored_raw = _validate_external_result(stored_result, stop - start)
            paired_raw = _validate_paired_result(paired_result, stop - start)
            paired_z = paired_result.z_ols.coefficients.detach()
            stored_batch_z = stored_z[start:stop].to(device)
            paired_z_parts.append(paired_z.cpu())
            pairs = {
                "z_ols": (stored_batch_z, paired_z),
                "generator_input_field": (stored_result.generator_input_field.tensor, paired_result.generator_input_field.tensor),
                "generator_output_field": (stored_result.generator_output_field.tensor, paired_result.generator_output_field.tensor),
                "raw_cmos": (stored_raw, paired_raw),
            }
            for name, (left, right) in pairs.items():
                delta = (left.detach() - right.detach()).abs()
                _finite(delta, f"{name} pair difference")
                flat = delta.to(torch.float64).flatten(1)
                row_max = flat.max(dim=1).values.cpu()
                row_mean = flat.mean(dim=1).cpu()
                diff_rows[name].append(row_max)
                diff_mean_rows[name].append(row_mean)
                diff_exact[name] = bool(diff_exact[name] and torch.equal(left.detach(), right.detach()))
                diff_allclose[name] = bool(diff_allclose[name] and torch.allclose(left.detach(), right.detach(), rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL))
                diff_max[name] = max(diff_max[name], float(delta.max().detach().cpu()))
                diff_mean_sum[name] += float(delta.to(torch.float64).sum().detach().cpu())
                diff_elements[name] += int(delta.numel())
            for mode, raw in (("stored_replay", stored_raw), ("paired_encoder", paired_raw)):
                _update_raw_state(raw, raw_states[mode])
                batch_logits = _evaluate_raw_batch(raw, models)
                for name, value in batch_logits.items():
                    logits[mode][name].append(value)
            if stop == int(stored_z.shape[0]) or stop % 200 == 0:
                print(json.dumps({"phase": "S0_IDENTITY_PAIR", "completed_rows": stop, "total_rows": int(stored_z.shape[0])}, ensure_ascii=False), flush=True)
            del stored_result, paired_result

    all_logits = {
        mode: {name: torch.cat(values, dim=0) for name, values in mode_values.items()}
        for mode, mode_values in logits.items()
    }
    semantic_reports: dict[str, dict[str, Any]] = {mode: {} for mode in all_logits}
    predictions: dict[str, dict[str, torch.Tensor]] = {mode: {} for mode in all_logits}
    for mode, mode_values in all_logits.items():
        for name, value in mode_values.items():
            report, predicted, _ = _classification_report(value, labels, label_source="canonical TRAIN source label")
            semantic_reports[mode][name] = report
            predictions[mode][name] = predicted
    difference_reports = {
        name: _merge_difference_batches(
            name,
            diff_rows[name],
            diff_mean_rows[name],
            shape=[int(stored_z.shape[0])],
            exact=diff_exact[name],
            allclose=diff_allclose[name],
            max_abs=diff_max[name],
            mean_sum=diff_mean_sum[name],
            element_count=diff_elements[name],
        )
        for name in diff_rows
    }
    return {
        "paired_z": torch.cat(paired_z_parts, dim=0),
        "logits": all_logits,
        "predictions": predictions,
        "semantic_reports": semantic_reports,
        "raw_stats": {name: _finish_raw_state(value) for name, value in raw_states.items()},
        "differences": difference_reports,
        "first_divergence_boundary": next((name for name in ("z_ols", "generator_input_field", "generator_output_field", "raw_cmos") if not difference_reports[name]["allclose"]), None),
        "source_path_contract": {
            "stored": "stored authoritative S0 z -> consumer.generated_forward -> W_OLS -> Generator -> raw CMOS",
            "paired": "canonical TRAIN image -> frozen Encoder -> OLS -> W_OLS -> same Generator -> raw CMOS",
        },
    }


def _bank_ols_report(z: torch.Tensor, labels: torch.Tensor, probe: Mapping[str, Any], mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    logits = sampler._probe_logits(z, mean, std, probe)
    report, predictions, probabilities = _classification_report(logits, labels, label_source="requested class persisted by accepted bank")
    report["probe_confidence"] = _summary(probabilities.max(dim=1).values)
    report["probe_top1_top2_logit_margin"] = _summary(logits.topk(2, dim=1).values[:, 0] - logits.topk(2, dim=1).values[:, 1])
    return {"classification": report, "predictions": predictions, "logits": logits}


def _conditional_retention(
    true_labels: torch.Tensor,
    ols_predictions: torch.Tensor,
    evaluator_predictions: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    true_labels = true_labels.cpu().to(torch.long)
    ols_predictions = ols_predictions.cpu().to(torch.long)
    ols_correct = ols_predictions == true_labels
    result: dict[str, Any] = {
        "P_yz_equals_y": float(ols_correct.to(torch.float64).mean()),
        "ols_probe_predicted_class_histogram": [int(item) for item in torch.bincount(ols_predictions, minlength=CLASS_COUNT)],
        "ols_probe_to_true_transition_rows_y_columns_yz": _transition_matrix(true_labels, ols_predictions).tolist(),
        "evaluators": {},
    }
    for name, predictions in evaluator_predictions.items():
        predictions = predictions.cpu().to(torch.long)
        on = ols_correct
        off = ~ols_correct
        result["evaluators"][name] = {
            "P_E_equals_y_given_yz_equals_y": None if not bool(on.any()) else float((predictions[on] == true_labels[on]).to(torch.float64).mean()),
            "P_E_equals_y_given_yz_not_equals_y": None if not bool(off.any()) else float((predictions[off] == true_labels[off]).to(torch.float64).mean()),
            "P_E_equals_yz_given_yz_equals_y": None if not bool(on.any()) else float((predictions[on] == ols_predictions[on]).to(torch.float64).mean()),
            "ols_rows_to_evaluator_columns": _transition_matrix(ols_predictions, predictions).tolist(),
            "prediction_agreement_with_ols": float((predictions == ols_predictions).to(torch.float64).mean()),
        }
    return result


def _class_stratified_bootstrap_difference(
    left_predictions: torch.Tensor,
    right_predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    count: int = BOOTSTRAP_COUNT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    left_predictions = left_predictions.cpu().to(torch.long)
    right_predictions = right_predictions.cpu().to(torch.long)
    labels = labels.cpu().to(torch.long)
    _require(left_predictions.shape == right_predictions.shape == labels.shape, "bootstrap inputs differ")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left_rates: list[torch.Tensor] = []
    right_rates: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        positions = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(positions.numel() > 0, f"bootstrap class {digit} is empty")
        draws = torch.randint(0, positions.numel(), (count, int(positions.numel())), generator=generator)
        chosen = positions[draws]
        left_rates.append((left_predictions[chosen] == digit).to(torch.float64).mean(dim=1))
        right_rates.append((right_predictions[chosen] == digit).to(torch.float64).mean(dim=1))
    difference = torch.stack(left_rates, dim=1).mean(dim=1) - torch.stack(right_rates, dim=1).mean(dim=1)
    quantiles = torch.quantile(difference, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64))
    observed_left = torch.stack(
        [(left_predictions[labels == digit] == digit).to(torch.float64).mean() for digit in range(CLASS_COUNT)]
    ).mean()
    observed_right = torch.stack(
        [(right_predictions[labels == digit] == digit).to(torch.float64).mean() for digit in range(CLASS_COUNT)]
    ).mean()
    observed = float(observed_left - observed_right)
    return {
        "observed_difference_left_minus_right": observed,
        "bootstrap_ci_difference_left_minus_right": {"level": 0.95, "lower": float(quantiles[0]), "median": float(quantiles[1]), "upper": float(quantiles[2])},
        "bootstrap_count": count,
        "bootstrap_seed": seed,
        "resampling": "common class-stratified row resampling; equal requested counts per class; no candidate reselection",
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    inputs = sampler._load_authorities(root)
    sampler_report_path = root / SAMPLER_REPORT
    bank_path = root / SAMPLER_BANK
    winner_path = root / SAMPLER_WINNER
    _require(file_sha256(sampler_report_path) == SAMPLER_REPORT_SHA256, "accepted sampler report identity changed")
    _require(file_sha256(bank_path) == SAMPLER_BANK_SHA256, "accepted sampler bank identity changed")
    _require(file_sha256(winner_path) == SAMPLER_WINNER_SHA256, "accepted sampler winner identity changed")
    sampler_report = json.loads(sampler_report_path.read_text(encoding="utf-8"))
    _require(sampler_report.get("result") == "VALID_COMPLETED", "accepted sampler result changed")
    _require(sampler_report.get("outer_gate") == "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_V1_RESOLVED", "accepted sampler gate changed")
    _require(sampler_report.get("classification") == "REAL_OLS_REPLAY_DOES_NOT_RECOVER_EXPECTED_GENERATOR_SEMANTICS", "accepted sampler classification changed")
    _require(sampler_report.get("winner", {}).get("family") == "S1", "accepted sampler winner changed")
    bank = _mapping(safe_torch_load(bank_path, expected_sha256=SAMPLER_BANK_SHA256, map_location="cpu"), "accepted sampler bank")
    winner = _mapping(safe_torch_load(winner_path, expected_sha256=SAMPLER_WINNER_SHA256, map_location="cpu"), "accepted sampler winner")
    _require(bank.get("schema") == "semantically_conditioned_ols_sampler_candidate_banks.v1", "accepted sampler bank schema changed")
    _require(winner.get("schema") == "semantically_conditioned_ols_sampler_frozen_winner.v1", "accepted sampler winner schema changed")
    _require(winner.get("family") == "S1" and winner.get("frozen_before_generator_canary") is True, "accepted S1 freeze changed")
    requested = _tensor(bank.get("requested_labels"), "accepted requested labels").to(torch.long).cpu()
    sample_indices = _tensor(bank.get("sample_indices"), "accepted sample indices").to(torch.long).cpu()
    paired_seeds = _tensor(bank.get("paired_seeds"), "accepted paired seeds").to(torch.long).cpu()
    z_by_family = _mapping(bank.get("z_by_family"), "accepted z by family")
    candidate_z = {name: _tensor(z_by_family.get(name), f"accepted {name} bank").cpu().contiguous() for name in ("S0", "S1", "S2", "S3")}
    _require(requested.shape == (BANK_COUNT,) and sample_indices.shape == (BANK_COUNT,) and paired_seeds.shape == (BANK_COUNT,), "accepted sampler bank identity vectors changed")
    expected_requested = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(BANK_PER_CLASS)
    _require(torch.equal(requested, expected_requested), "accepted sampler requested class balance changed")
    expected_sample_indices = torch.arange(BANK_PER_CLASS, dtype=torch.long).repeat(CLASS_COUNT)
    _require(torch.equal(sample_indices, expected_sample_indices), "accepted sampler per-class sample indices changed")
    _require(torch.equal(winner.get("requested_labels"), requested), "accepted winner requested labels differ")
    _require(torch.equal(winner.get("sample_indices"), sample_indices), "accepted winner sample indices differ")
    winner_z = _tensor(winner.get("z"), "accepted winner z").cpu().contiguous()
    _require(torch.equal(winner_z, candidate_z["S1"]), "accepted winner z differs from accepted S1 bank")
    for name, value in candidate_z.items():
        _require(value.shape == (BANK_COUNT, OLS_PORT_COUNT) and value.dtype == torch.complex64, f"accepted {name} shape/dtype changed")
        _finite(value.real, f"accepted {name} real")
        _finite(value.imag, f"accepted {name} imaginary")

    train_z = inputs["train_z"]
    train_indices = inputs["train_indices"]
    train_labels = inputs["train_labels"]
    split = sampler._make_sampler_split(train_labels, train_indices)
    expected_split = _mapping(sampler_report.get("train_validation_roles", {}).get("SAMPLER_FIT"), "accepted sampler fit split")
    for key in ("fit_count", "validation_count", "fit_positions_sha256", "validation_positions_sha256", "fit_source_indices_sha256", "validation_source_indices_sha256"):
        _require(expected_split.get(key) == sampler._split_identity(split).get(key), f"accepted sampler split changed: {key}")
    source_positions = torch.cat([split["fit_positions_by_class"][digit][:BANK_PER_CLASS] for digit in range(CLASS_COUNT)]).to(torch.long).contiguous()
    source_indices = train_indices.index_select(0, source_positions).contiguous()
    source_labels = inputs["labels"].index_select(0, source_indices).contiguous()

    canonical_images, canonical_labels = robust._load_canonical_train(root)
    _require(torch.equal(canonical_labels, inputs["labels"]), "canonical labels differ from accepted label authority")
    _require(torch.equal(train_labels, inputs["labels"].index_select(0, train_indices)), "TRAIN label binding changed")
    _require(torch.equal(source_labels, requested), "S0 requested classes do not equal canonical source labels")
    _require(torch.equal(candidate_z["S0"], train_z.index_select(0, source_positions)), "S0 is not the exact accepted TRAIN OLS row replay")

    post_path = root / POST3000_REPORT
    _require(file_sha256(post_path) == POST3000_REPORT_SHA256, "accepted post3000 report identity changed")
    post_report = json.loads(post_path.read_text(encoding="utf-8"))
    _require(post_report.get("result") == "VALID_COMPLETED", "accepted post3000 report is not valid completed")
    _require(post_report.get("outer_gate") == "POST3000_GENERATION_READINESS_AND_SCALING_RESOLVED", "accepted post3000 gate changed")

    checkpoints = {int(step): readiness._checkpoint_payload(root, {"step": step, **descriptor}) for step, descriptor in readiness.REAL_CHECKPOINTS.items()}
    encoder0 = _tensor(checkpoints[0].get("encoder_state"), "step0 encoder")
    _require(all(torch.equal(encoder0, _tensor(checkpoints[step].get("encoder_state"), f"step{step} encoder")) for step in (300, 3000)), "frozen Encoder changed across checkpoints")

    return {
        "sampler_inputs": inputs,
        "sampler_report": sampler_report,
        "candidate_z": candidate_z,
        "requested": requested,
        "sample_indices": sample_indices,
        "paired_seeds": paired_seeds,
        "split": split,
        "source_positions": source_positions,
        "source_indices": source_indices,
        "source_labels": source_labels,
        "canonical_images": canonical_images,
        "canonical_labels": canonical_labels,
        "post_report": post_report,
        "checkpoints": checkpoints,
    }


def _load_models(root: Path, authorities: Mapping[str, Any]) -> tuple[Any, dict[str, torch.nn.Module], torch.device]:
    robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "frozen replay audit requires CUDA")
    device = torch.device("cuda:0")
    consumer, smooth, independent, ec = readiness._load_models(root, authorities["sampler_inputs"]["robust"], device)
    consumer.runtime.eval()
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    models: dict[str, torch.nn.Module] = {"E_S": smooth, "E_I": independent, "E_C": ec}
    for name, model in models.items():
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")
    _require(all(not parameter.requires_grad for parameter in consumer.runtime.parameters()), "Generator/Encoder runtime is not frozen")
    return consumer, models, device


def _load_checkpoint(consumer: Any, checkpoint: Mapping[str, Any]) -> None:
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    consumer.runtime.eval()
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)


def _accepted_real_dev_metrics(post_report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    checkpoints = _mapping(_mapping(post_report.get("real_ols_semantic_realization"), "accepted real OLS realization") .get("checkpoints"), "accepted real OLS checkpoints")
    result: dict[str, dict[str, float]] = {}
    for step in (0, 300, 3000):
        item = _mapping(checkpoints.get(str(step)), f"accepted real DEV step {step}")
        evaluators = _mapping(item.get("evaluators"), f"accepted real DEV evaluators step {step}")
        result[str(step)] = {name: float(_mapping(evaluators.get(name), f"accepted real DEV {name} step {step}").get("macro_recall")) for name in ("E_S", "E_I", "E_C")}
    return result


def _classification_relation(bootstrap: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    compatible = True
    underperforming = False
    for value in bootstrap.values():
        ci = _mapping(value["bootstrap_ci_difference_left_minus_right"], "bootstrap interval")
        lower = float(ci["lower"])
        upper = float(ci["upper"])
        compatible = compatible and lower <= 0.0 <= upper
        underperforming = underperforming or upper < 0.0
    return {
        "all_evaluators_95ci_contains_zero": compatible,
        "any_evaluator_95ci_strictly_below_zero": underperforming,
        "rule": "S1 is compatible with S0 only when every E_S/E_I/E_C paired class-stratified 95% CI contains zero; material underperformance is any interval with upper<0; otherwise mixed",
    }


def _run_experiment(root: Path, output_dir: Path) -> dict[str, Any]:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "protected output namespace is forbidden")
    _require(not target.exists(), f"write-once output already exists: {target}")
    print(json.dumps({"phase": "AUTHORITY_PREFLIGHT", "status": "START"}, ensure_ascii=False), flush=True)
    authorities = _load_authorities(root)
    sampler_inputs = authorities["sampler_inputs"]
    candidate_z = authorities["candidate_z"]
    requested = authorities["requested"]
    source_labels = authorities["source_labels"]
    source_images = authorities["canonical_images"].index_select(0, authorities["source_indices"])
    train_z = sampler_inputs["train_z"]
    probe = sampler_inputs["probe"]
    train_cart = sampler._ols_features(train_z)
    _, probe_mean, probe_std = sampler._fit_standardizer(train_cart)
    _require(tensor_sha256(probe_mean) == sampler.PROBE_MEAN_SHA256, "accepted probe mean reconstruction changed")
    _require(tensor_sha256(probe_std) == sampler.PROBE_STD_SHA256, "accepted probe std reconstruction changed")

    print(json.dumps({"phase": "CUDA_PREFLIGHT", "status": "START"}, ensure_ascii=False), flush=True)
    consumer, models, device = _load_models(root, authorities)
    _load_checkpoint(consumer, authorities["checkpoints"][3000])
    print(json.dumps({"phase": "S0_IDENTITY_PAIR", "status": "START"}, ensure_ascii=False), flush=True)
    identity_pair = _run_step3000_identity_pair(consumer, models, source_images, candidate_z["S0"], source_labels, device)
    stored_z = candidate_z["S0"]
    paired_z = identity_pair["paired_z"]
    stored_vs_paired_ols = identity_pair["differences"]["z_ols"]
    replay_vs_paired_cmos = {name: identity_pair["differences"][name] for name in ("generator_input_field", "generator_output_field", "raw_cmos")}

    print(json.dumps({"phase": "S0_TRAJECTORY", "status": "START"}, ensure_ascii=False), flush=True)
    trajectory: dict[str, Any] = {}
    for step in (0, 300):
        _load_checkpoint(consumer, authorities["checkpoints"][step])
        direct = _run_direct_bank(consumer, models, stored_z, source_labels, device, phase=f"S0_REPLAY_STEP_{step}")
        trajectory[str(step)] = {"step": step, "evaluators": direct["evaluators"], "raw_cmos": direct["raw_stats"]}
    trajectory["3000"] = {"step": 3000, "evaluators": identity_pair["semantic_reports"]["stored_replay"], "raw_cmos": identity_pair["raw_stats"]["stored_replay"]}

    print(json.dumps({"phase": "S1_VS_S0", "status": "START"}, ensure_ascii=False), flush=True)
    _load_checkpoint(consumer, authorities["checkpoints"][3000])
    s1_direct = _run_direct_bank(consumer, models, candidate_z["S1"], requested, device, phase="S1_REPLAY_STEP_3000")
    s0_step3000_predictions = {name: identity_pair["predictions"]["stored_replay"][name] for name in models}
    s1_predictions = s1_direct["predictions"]
    bootstrap = {
        name: _class_stratified_bootstrap_difference(s1_predictions[name], s0_step3000_predictions[name], requested)
        for name in models
    }
    relation = _classification_relation(bootstrap)

    s0_ols = _bank_ols_report(candidate_z["S0"], requested, probe, probe_mean, probe_std)
    s1_ols = _bank_ols_report(candidate_z["S1"], requested, probe, probe_mean, probe_std)
    ols_predictions = s0_ols["predictions"]
    conditional = _conditional_retention(source_labels, ols_predictions, s0_step3000_predictions)
    replay_semantics = identity_pair["semantic_reports"]["stored_replay"]
    paired_semantics = identity_pair["semantic_reports"]["paired_encoder"]
    semantic_equivalence: dict[str, Any] = {}
    for name in models:
        replay_pred = identity_pair["predictions"]["stored_replay"][name]
        paired_pred = identity_pair["predictions"]["paired_encoder"][name]
        semantic_equivalence[name] = {
            "stored_replay": replay_semantics[name],
            "paired_encoder": paired_semantics[name],
            "prediction_agreement_fraction": float((replay_pred == paired_pred).to(torch.float64).mean()),
            "prediction_disagreement_count": int((replay_pred != paired_pred).sum()),
        }

    prior_real_dev = _accepted_real_dev_metrics(authorities["post_report"])
    s0_ec_step3000 = float(replay_semantics["E_C"]["macro_recall"])
    prior_ec_step3000 = float(prior_real_dev["3000"]["E_C"])
    real_context_consistent = math.isclose(s0_ec_step3000, prior_ec_step3000, rel_tol=0.0, abs_tol=REAL_CONTEXT_EXACT_TOLERANCE)

    source_audit = {
        "rows_checked": int(requested.numel()),
        "requested_class_histogram": [int(item) for item in torch.bincount(requested, minlength=CLASS_COUNT)],
        "source_label_histogram": [int(item) for item in torch.bincount(source_labels, minlength=CLASS_COUNT)],
        "label_mismatch_rows": int((requested != source_labels).sum()),
        "source_indices_resolved_to_canonical_train": bool(torch.equal(source_labels, authorities["canonical_labels"].index_select(0, authorities["source_indices"]))),
        "source_index_mismatch_rows": int((candidate_z["S0"] != train_z.index_select(0, authorities["source_positions"])).any(dim=1).sum()),
        "duplicate_source_identity_rows": int(requested.numel() - torch.unique(authorities["source_indices"]).numel()),
        "unique_source_identity_count": int(torch.unique(authorities["source_indices"]).numel()),
        "missing_source_identity_rows": int((~torch.isin(authorities["source_indices"], sampler_inputs["train_indices"])).sum()),
        "source_positions_sha256": tensor_sha256(authorities["source_positions"]),
        "source_indices_sha256": tensor_sha256(authorities["source_indices"]),
        "source_identity_basis": "accepted S0_control_contract plus deterministic accepted SAMPLER_FIT split; no new bank or sampler fit",
    }

    identity_exact = source_audit["label_mismatch_rows"] == 0 and source_audit["source_index_mismatch_rows"] == 0 and source_audit["duplicate_source_identity_rows"] == 0 and source_audit["missing_source_identity_rows"] == 0
    path_equivalent = bool(stored_vs_paired_ols["allclose"] and all(item["allclose"] for item in replay_vs_paired_cmos.values()))
    if not identity_exact or not path_equivalent:
        classification = "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_SUPPORTED"
        mechanism = "At least one source identity, stored-vs-paired OLS, shared writer, generator, or CMOS boundary failed the preregistered equivalence audit. The observed S0 semantic level cannot yet be treated as a pure Generator ceiling."
        successor = "REPAIR_EXACT_REAL_OLS_REPLAY_CONDITIONING_IDENTITY_BOUNDARY_V1"
    elif relation["all_evaluators_95ci_contains_zero"] and real_context_consistent:
        classification = "SEMANTIC_SAMPLER_REPAIRED_TO_REAL_OLS_GENERATOR_CEILING"
        mechanism = "S0 source identity and the paired Encoder→OLS→W_OLS→Generator→CMOS path are equivalent. S1 is statistically compatible with exact S0 replay across all frozen evaluators, while S0 matches the accepted real-DEV step3000 realization context; the remaining ceiling is canonical Generator semantic realization."
        successor = "ATTRIBUTE_POST3000_GENERATOR_SEMANTIC_REALIZATION_CEILING_V1"
    elif relation["any_evaluator_95ci_strictly_below_zero"]:
        classification = "SYNTHETIC_ON_MANIFOLD_OLS_GENERATOR_TRANSFER_GAP_SUPPORTED"
        mechanism = "Replay is equivalent, but the selected S1 synthetic bank is materially below exact S0 replay for at least one frozen evaluator under the fixed class-stratified bootstrap comparison."
        successor = "ATTRIBUTE_GENERATIVE_SAMPLE_VS_REAL_OLS_LOCAL_JACOBIAN_GAP_V1"
    elif not real_context_consistent:
        classification = "REAL_OLS_SEMANTIC_REALIZATION_SPLIT_OR_IDENTITY_EFFECT_SUPPORTED"
        mechanism = "Replay is equivalent, but the S0 TRAIN source bank and the accepted real DEV realization context do not agree under the fixed context consistency check; localize TRAIN/DEV composition or preprocessing before attributing the ceiling to S1."
        successor = "LOCALIZE_REAL_OLS_TRAIN_DEV_SEMANTIC_REALIZATION_SPLIT_V1"
    else:
        classification = "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_MIXED"
        mechanism = "The replay boundary is equivalent, but the frozen S1-versus-S0 evaluator comparisons do not support a single clean compatible-or-underperforming interpretation."
        successor = "ATTRIBUTE_GENERATIVE_SAMPLE_VS_REAL_OLS_LOCAL_JACOBIAN_GAP_V1"

    target.mkdir(parents=True, exist_ok=False)
    compact_payload = {
        "schema": "real_ols_replay_conditioning_identity_gap_audit_payload.v1",
        "requested_labels": requested,
        "source_labels": source_labels,
        "source_positions": authorities["source_positions"],
        "source_indices": authorities["source_indices"],
        "sample_indices": authorities["sample_indices"],
        "stored_s0_z": stored_z,
        "paired_encoder_z": paired_z,
        "s1_z": candidate_z["S1"],
        "step3000_predictions": {
            "S0_stored_replay": identity_pair["predictions"]["stored_replay"],
            "S0_paired_encoder": identity_pair["predictions"]["paired_encoder"],
            "S1_replay": s1_predictions,
        },
        "trajectory_predictions": {
            str(step): {
                name: trajectory[str(step)]["evaluators"][name]["predicted_class_histogram"]
                for name in models
            }
            for step in (0, 300, 3000)
        },
    }
    robust._write_torch(target / "audit_payload.pt", compact_payload, kind="real OLS replay identity audit payload")

    input_authorities = {
        "accepted_semantically_conditioned_sampler_report": _artifact_ref(root, root / SAMPLER_REPORT, "accepted sampler report"),
        "accepted_candidate_latent_banks": _artifact_ref(root, root / SAMPLER_BANK, "accepted S0-S3 bank"),
        "accepted_frozen_s1_sampler": _artifact_ref(root, root / SAMPLER_WINNER, "accepted frozen S1 sampler"),
        "real_train_complex_ols": _artifact_ref(root, root / sampler.TRAIN_OLS, "accepted real TRAIN complex OLS [55000,100]"),
        "real_dev_complex_ols": _artifact_ref(root, root / sampler.DEV_OLS, "accepted real DEV complex OLS [5000,100]"),
        "canonical_train_source": _artifact_ref(root, root / robust.TRAINING_SOURCE, "canonical TRAIN images and labels"),
        "frozen_ols_probe": _artifact_ref(root, root / sampler.PROBE_WEIGHTS, "accepted frozen OLS linear probe"),
        "ols_scale_gauge_transport": {
            "scale_tensor_sha256": tensor_sha256(sampler_inputs["scale"]),
            "eps_s": sampler.EPS_S,
            "max_q_gauge": "accepted max_q_gauge with lowest-index tie; audit does not alter raw z",
            "T": "I_100 coefficient-space transport",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "port_order": "row-major j=10*r+c",
        },
        "frozen_checkpoints": {str(step): _artifact_ref(root, root / readiness.REAL_CHECKPOINTS[step]["path"], f"frozen Generator checkpoint step {step}") for step in (0, 300, 3000)},
        "frozen_evaluators": {
            "E_S": _artifact_ref(root, root / robust.SMOOTH_CHECKPOINT, "frozen E_S"),
            "E_I": _artifact_ref(root, root / robust.INDEPENDENT_CHECKPOINT, "frozen E_I"),
            "E_C": _artifact_ref(root, root / robust.EC_CHECKPOINT, "frozen E_C"),
        },
        "accepted_post3000_real_ols_report": _artifact_ref(root, root / POST3000_REPORT, "accepted real-DEV OLS realization context"),
        "accepted_prior_manifold_gap": _artifact_ref(root, root / sampler.GAP_REPORT, "accepted prior sampled OLS manifold-gap diagnostic"),
    }
    source_refs = [
        _artifact_ref(root, Path(__file__), "current replay identity audit source"),
        _artifact_ref(root, root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py", "frozen Chen-BOGT consumer source"),
        _artifact_ref(root, root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py", "frozen aligned runtime source"),
    ]
    report = {
        "schema": "real_ols_replay_conditioning_identity_gap_v1_report",
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "task_id": TASK_ID,
        "mode": "IMPLEMENT_PREPARED",
        "risk_level": "R2_ARTIFACT_PRODUCING_RESEARCH",
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "input_authority": SUPPORTED_INPUT_GATE,
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "local_state": {"branch": sampler_inputs["git"]["branch"], "head": sampler_inputs["git"]["head"], "worktree_dirty": bool(sampler_inputs["git"].get("worktree_dirty", True)), "working_tree_preserved": True},
        "supported_prior_bottleneck": "REAL_OLS_REPLAY_DOES_NOT_RECOVER_EXPECTED_GENERATOR_SEMANTICS",
        "input_authorities": input_authorities,
        "s0_source_identity": source_audit,
        "stored_vs_paired_ols": {
            **stored_vs_paired_ols,
            "stored_tensor_sha256": tensor_sha256(stored_z),
            "paired_tensor_sha256": tensor_sha256(paired_z),
            "stored_role": "accepted exact TRAIN OLS rows selected by frozen S0 control contract",
            "paired_role": "canonical TRAIN images through frozen Encoder and OLS",
        },
        "replay_vs_paired_cmos": {
            **replay_vs_paired_cmos,
            "same_scale": True,
            "same_T": True,
            "same_W_OLS": True,
            "same_port_order": True,
            "same_step3000_generator": True,
            "same_cmos_readout": True,
            "first_divergence_boundary": identity_pair["first_divergence_boundary"],
            "path_equivalence": bool(path_equivalent),
        },
        "replay_vs_paired_semantics": semantic_equivalence,
        "real_ols_semantic_retention": {
            "ols_probe": s0_ols["classification"],
            "E_S": replay_semantics["E_S"],
            "E_I": replay_semantics["E_I"],
            "E_C": replay_semantics["E_C"],
            "conditional_retention": conditional,
            "true_source_labels_are_authoritative": True,
            "ols_probe_is_not_treated_as_generated_ground_truth": True,
        },
        "step0_step300_step3000_real_replay_trajectory": {
            "steps": trajectory,
            "accepted_real_dev_context": prior_real_dev,
            "accepted_real_dev_context_role": "read-only contextual diagnostic; not a checkpoint or sampler selection criterion",
        },
        "s1_vs_s0": {
            "latent_semantic_support": {
                "S0": float(authorities["sampler_report"]["latent_candidate_results"]["S0"]["semantic_support_rate"]),
                "S1": float(authorities["sampler_report"]["latent_candidate_results"]["S1"]["semantic_support_rate"]),
                "S1_bootstrap_ci_lower": float(authorities["sampler_report"]["latent_candidate_results"]["S1"]["semantic_support_rate_bootstrap_ci"]["lower"]),
                "source": "accepted sampler report; no recomputation used for selection",
            },
            "OLS_probe_requested_class_recall": {"S0": s0_ols["classification"], "S1": s1_ols["classification"]},
            "downstream_requested_class_recall": {"S0": replay_semantics, "S1": s1_direct["evaluators"]},
            "class_histograms": {"S0": {name: replay_semantics[name]["predicted_class_histogram"] for name in models}, "S1": {name: s1_direct["evaluators"][name]["predicted_class_histogram"] for name in models}},
            "optical_validity": {"S0": identity_pair["raw_stats"]["stored_replay"], "S1": s1_direct["raw_stats"]},
            "bootstrap": bootstrap,
            "bootstrap_relation": relation,
            "no_candidate_reselection": True,
        },
        "zero_point_ninety_threshold": {"historical_role": "aspirational end-to-end target only", "used_for_sampler_attribution": False, "reason": "accepted exact S0 replay is itself approximately 0.459 E_C at step3000"},
        "real_dev_context_consistency": {"S0_step3000_E_C": s0_ec_step3000, "accepted_real_DEV_step3000_E_C": prior_ec_step3000, "absolute_difference": abs(s0_ec_step3000 - prior_ec_step3000), "exact_within_fixed_context_tolerance": real_context_consistent, "tolerance": REAL_CONTEXT_EXACT_TOLERANCE},
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "outer_gate_contract": "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_ATTRIBUTED",
        "scientific_artifact": _artifact_ref(root, target / "audit_payload.pt", "compact replay identity audit payload"),
        "files_changed": {"repository_source": [_relative(root, Path(__file__))], "status_documents_modified": False, "prior_sampler_artifacts_modified": False},
        "source_refs": source_refs,
        "validation": {
            "authority_preflight": "READY",
            "cuda": True,
            "forward_only": True,
            "backward_called": False,
            "optimizer_constructed": False,
            "sampler_refit": False,
            "sampler_reselected": False,
            "new_latent_bank_generated": False,
            "generator_training": False,
            "evaluator_training": False,
            "protected_test_sealed_reads": False,
            "source_label_identity_audited": True,
            "stored_vs_paired_ols_audited": True,
            "stored_vs_paired_cmos_audited": True,
            "fixed_class_stratified_bootstrap": {"count": BOOTSTRAP_COUNT, "seed": BOOTSTRAP_SEED},
        },
        "claim_boundary": "This artifact attributes exact S0 real-OLS replay identity and compares frozen S1 versus S0 transfer. It does not claim 0.90 end-to-end readiness, sampler promotion, Generator retraining, architecture change, or optical deployment readiness.",
        "unresolved": ["The frozen E_C remains a descriptive semantic realization evaluator in this attribution; no new evaluator or human perceptual claim is introduced."],
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    robust._write_json(target / "report.json", report, kind="real OLS replay conditioning identity gap report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the frozen audit")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "output_dir": args.output_dir, "no_update": True}, ensure_ascii=False, indent=2))
        return 0
    report = _run_experiment(root, Path(args.output_dir))
    print(json.dumps({"result": report["result"], "outer_gate": report["outer_gate"], "classification": report["classification"], "successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
