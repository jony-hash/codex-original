"""CPU-only review of saved LightGen semantic-target comparison evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


TASK_ID = "M0_SEMANTIC_TARGET_EVIDENCE_REVIEW_V1"
PARENT_TASK_ID = "M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1"
DEFAULT_ARTIFACT_DIR = Path(
    "D:/codex-original/outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01"
)
DEFAULT_OUTPUT_DIR = Path(
    "D:/codex-original/outputs/20261004_m0_semantic_target_evidence_review_v1/runs/attempt_01"
)
ENDPOINTS = ("step2200", "A_feature", "B_class_ce")
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
ROLES = ("fit", "historic_holdout", "confirmation")
FOCUS_CLASSES = (1, 7)
COMPARISON_CLASSES = (0, 4, 8)
DISPLAY_CLASSES = FOCUS_CLASSES + COMPARISON_CLASSES
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-7


class EvidenceAnomalyError(ValueError):
    """Raised when saved evidence cannot be reconciled with its parent report."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceAnomalyError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict) or any(not torch.is_tensor(item) for item in value.values()):
        raise EvidenceAnomalyError(f"expected a tensor-only mapping in {path}")
    return value


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value.real) & torch.isfinite(value.imag) if torch.is_complex(value) else torch.isfinite(value)
    if not bool(finite.all()):
        raise EvidenceAnomalyError(f"nonfinite values in {name}")


def _roles(population: Mapping[str, Any]) -> list[str]:
    values = population.get("roles")
    if not isinstance(values, list) or len(values) != 600:
        raise EvidenceAnomalyError("population roles must contain 600 rows")
    return [str(value) for value in values]


def _verify_population(population: Mapping[str, Any]) -> tuple[list[int], list[int], list[str]]:
    if population.get("task_id") != PARENT_TASK_ID:
        raise EvidenceAnomalyError("population belongs to a different parent task")
    labels = population.get("labels")
    source_indices = population.get("source_row_indices")
    roles = _roles(population)
    if not isinstance(labels, list) or len(labels) != 600 or not isinstance(source_indices, list) or len(source_indices) != 600:
        raise EvidenceAnomalyError("population labels/source row indices must each contain 600 entries")
    labels = [int(value) for value in labels]
    source_indices = [int(value) for value in source_indices]
    if len(set(source_indices)) != 600:
        raise EvidenceAnomalyError("population source row indices are not unique")
    if roles != ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300:
        raise EvidenceAnomalyError("population roles do not match the declared 200/100/300 split")
    if Counter(labels[:200]) != Counter({label: 20 for label in range(10)}):
        raise EvidenceAnomalyError("fit population is not balanced at 20 rows per class")
    if Counter(labels[200:300]) != Counter({label: 10 for label in range(10)}):
        raise EvidenceAnomalyError("historic comparison population is not balanced at 10 rows per class")
    if Counter(labels[300:]) != Counter({label: 30 for label in range(10)}):
        raise EvidenceAnomalyError("confirmation population is not balanced at 30 rows per class")
    confirmation = population.get("confirmation_indices")
    if not isinstance(confirmation, list) or [int(value) for value in confirmation] != source_indices[300:]:
        raise EvidenceAnomalyError("confirmation_indices do not match the saved population order")
    if int(population.get("original_fit_rows", -1)) != 200 or int(population.get("historic_holdout_rows", -1)) != 100:
        raise EvidenceAnomalyError("saved population counts do not match the declared roles")
    if int(population.get("new_confirmation_rows", -1)) != 300:
        raise EvidenceAnomalyError("saved confirmation count is not 300")
    return labels, source_indices, roles


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for actual, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[actual, predicted] += 1
    correct = confusion.diag()
    predicted_counts = confusion.sum(dim=0)
    recall = [float(correct[i]) / int((labels == i).sum()) for i in range(10)]
    return {
        "rows": int(labels.numel()),
        "macro_recall": float(sum(recall) / 10),
        "per_class_correct": [int(value) for value in correct.tolist()],
        "per_class_recall": recall,
        "predicted_class_counts": [int(value) for value in predicted_counts.tolist()],
        "confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
    }


def _one_sided_exact_p(wins: int, discordant: int) -> float:
    if discordant == 0:
        return 1.0
    return min(1.0, sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2**discordant))


def _holm(p_values: Mapping[int, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[str(label)] = running
    return adjusted


def _verify_parent_metrics(
    report: Mapping[str, Any],
    readouts: Mapping[str, Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    roles: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    recalculated: dict[str, dict[str, dict[str, Any]]] = {}
    parent_metrics = report.get("evaluator_metrics")
    if not isinstance(parent_metrics, dict):
        raise EvidenceAnomalyError("parent report is missing evaluator_metrics")
    if report.get("task_id") != PARENT_TASK_ID:
        raise EvidenceAnomalyError("parent report belongs to a different task")
    for endpoint in ENDPOINTS:
        recalculated[endpoint] = {}
        for evaluator in EVALUATORS:
            logits = readouts[endpoint][f"{evaluator.lower()}_logits"]
            recalculated[endpoint][evaluator] = {}
            for role in ROLES:
                positions = torch.tensor([i for i, value in enumerate(roles) if value == role], dtype=torch.long)
                result = _classification_metrics(logits.index_select(0, positions), labels.index_select(0, positions))
                saved = parent_metrics[endpoint][evaluator]["by_role"][role]
                for field in ("per_class_correct", "predicted_class_counts", "confusion_matrix_actual_rows_predicted_columns"):
                    if result[field] != saved[field]:
                        raise EvidenceAnomalyError(f"parent report mismatch at {endpoint}/{evaluator}/{role}/{field}")
                if not math.isclose(result["macro_recall"], float(saved["macro_recall"]), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
                    raise EvidenceAnomalyError(f"parent report macro recall mismatch at {endpoint}/{evaluator}/{role}")
                recalculated[endpoint][evaluator][role] = result
    if report.get("status") != "VALID_FAILED" or report.get("classification") != "NOT_SUPPORTED_WITHIN_SCOPE":
        raise EvidenceAnomalyError("parent scientific terminal state differs from the declared source result")
    return recalculated


def _verify_paired_tests(
    report: Mapping[str, Any],
    readouts: Mapping[str, Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    roles: Sequence[str],
) -> dict[str, Any]:
    confirmation = torch.tensor([i for i, role in enumerate(roles) if role == "confirmation"], dtype=torch.long)
    a_pred = readouts["A_feature"]["e_d_logits"].index_select(0, confirmation).argmax(dim=1)
    b_pred = readouts["B_class_ce"]["e_d_logits"].index_select(0, confirmation).argmax(dim=1)
    actual = labels.index_select(0, confirmation)
    raw: dict[str, Any] = {}
    p_values: dict[int, float] = {}
    saved_raw = report["paired_statistics"]["raw"]
    for label in FOCUS_CLASSES:
        class_mask = actual == label
        a_correct = a_pred[class_mask] == actual[class_mask]
        b_correct = b_pred[class_mask] == actual[class_mask]
        a_only = int((a_correct & ~b_correct).sum())
        b_only = int((b_correct & ~a_correct).sum())
        p_value = _one_sided_exact_p(b_only, a_only + b_only)
        current = {"a_only_correct": a_only, "b_only_correct": b_only, "discordant_pairs": a_only + b_only, "one_sided_exact_p": p_value}
        if current["a_only_correct"] != saved_raw[str(label)]["a_only_correct"] or current["b_only_correct"] != saved_raw[str(label)]["b_only_correct"]:
            raise EvidenceAnomalyError(f"parent paired counts mismatch for class {label}")
        if not math.isclose(p_value, float(saved_raw[str(label)]["one_sided_exact_p"]), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
            raise EvidenceAnomalyError(f"parent exact paired p-value mismatch for class {label}")
        raw[str(label)] = current
        p_values[label] = p_value
    adjusted = _holm(p_values)
    saved_adjusted = report["paired_statistics"]["holm_adjusted_p"]
    for label, value in adjusted.items():
        if not math.isclose(value, float(saved_adjusted[label]), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
            raise EvidenceAnomalyError(f"parent Holm adjusted p-value mismatch for class {label}")
    return {"raw": raw, "holm_adjusted_p": adjusted}


def _load_parent_artifacts(artifact_dir: Path) -> dict[str, Any]:
    report = _read_json(artifact_dir / "report.json")
    population = _read_json(artifact_dir / "input_population.json")
    labels_list, source_indices_list, roles = _verify_population(population)
    labels = torch.tensor(labels_list, dtype=torch.int64)
    readouts = {endpoint: _load_tensor_file(artifact_dir / f"readouts_{endpoint}.pt") for endpoint in ENDPOINTS}
    row_metrics = {endpoint: _load_tensor_file(artifact_dir / f"row_metrics_{endpoint}.pt") for endpoint in ENDPOINTS}
    histories = {
        arm: _read_json(artifact_dir / f"training_history_{arm}.json")["records"]
        for arm in ("A_feature", "B_class_ce")
    }
    return {
        "report": report,
        "population": population,
        "labels": labels,
        "source_indices": torch.tensor(source_indices_list, dtype=torch.int64),
        "roles": roles,
        "readouts": readouts,
        "row_metrics": row_metrics,
        "histories": histories,
    }


def _verify_readouts(bundle: Mapping[str, Any]) -> dict[str, Any]:
    labels: torch.Tensor = bundle["labels"]
    source_indices: torch.Tensor = bundle["source_indices"]
    readouts: Mapping[str, Mapping[str, torch.Tensor]] = bundle["readouts"]
    row_metrics: Mapping[str, Mapping[str, torch.Tensor]] = bundle["row_metrics"]
    for endpoint in ENDPOINTS:
        values = readouts[endpoint]
        _verify_row_identity(endpoint, values, row_metrics[endpoint], labels, source_indices)
        expected_shapes = {
            "z": (600, 100),
            "raw_cmos": (600, 1, 200, 200),
            "target_slm_intensity": (600, 1, 200, 200),
            "power_by_plane": (600, 5),
        }
        for name, shape in expected_shapes.items():
            if tuple(values[name].shape) != shape:
                raise EvidenceAnomalyError(f"unexpected {endpoint}/{name} shape: {tuple(values[name].shape)}")
            _finite_tensor(f"{endpoint}/{name}", values[name])
        for evaluator in EVALUATORS:
            logits = values[f"{evaluator.lower()}_logits"]
            if tuple(logits.shape) != (600, 10):
                raise EvidenceAnomalyError(f"unexpected {endpoint}/{evaluator} logits shape")
            _finite_tensor(f"{endpoint}/{evaluator} logits", logits)
        metrics = row_metrics[endpoint]
        for name, metric in metrics.items():
            if torch.is_tensor(metric) and metric.is_floating_point():
                _finite_tensor(f"{endpoint}/{name}", metric)
            if torch.is_tensor(metric) and metric.ndim > 0 and int(metric.shape[0]) != 600:
                raise EvidenceAnomalyError(f"row metric {endpoint}/{name} does not have 600 rows")
    reference = readouts[ENDPOINTS[0]]
    differences = {}
    for endpoint in ENDPOINTS[1:]:
        current = readouts[endpoint]
        for field in ("target_slm_intensity", "z"):
            max_difference = float((current[field] - reference[field]).abs().max())
            if not torch.allclose(current[field], reference[field], rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
                raise EvidenceAnomalyError(f"frozen {field} differs between step2200 and {endpoint}")
            differences[f"{endpoint}_{field}_max_abs_difference"] = max_difference
    classifications = _verify_parent_metrics(bundle["report"], readouts, labels, bundle["roles"])
    paired = _verify_paired_tests(bundle["report"], readouts, labels, bundle["roles"])
    return {"parent_report_classification_metrics_match": True, "parent_pairwise_tests_match": True, "frozen_input_readout_differences": differences, "classification_metrics": classifications, "paired_tests": paired}


def _verify_row_identity(
    endpoint: str,
    readout: Mapping[str, torch.Tensor],
    row_metrics: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> None:
    if not torch.equal(readout.get("labels", torch.empty(0, dtype=torch.int64)), labels):
        raise EvidenceAnomalyError(f"row ordering/labels changed in {endpoint}")
    if not torch.equal(readout.get("source_indices", torch.empty(0, dtype=torch.int64)), source_indices):
        raise EvidenceAnomalyError(f"row ordering/labels changed in {endpoint}")
    if not torch.equal(row_metrics.get("labels", torch.empty(0, dtype=torch.int64)), labels):
        raise EvidenceAnomalyError(f"row metric labels do not match population in {endpoint}")
    if not torch.equal(row_metrics.get("source_indices", torch.empty(0, dtype=torch.int64)), source_indices):
        raise EvidenceAnomalyError(f"row metric source indices do not match population in {endpoint}")


def _logit_scores(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = logits.to(torch.float64)
    true_logits = values.gather(1, labels[:, None]).squeeze(1)
    masked = values.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    best_wrong = masked.max(dim=1).values
    margins = true_logits - best_wrong
    logsumexp = torch.logsumexp(values, dim=1)
    cross_entropy = logsumexp - true_logits
    true_probabilities = torch.exp(-cross_entropy)
    predictions = values.argmax(dim=1)
    return cross_entropy, true_probabilities, margins, predictions


def _row_score_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels: torch.Tensor = bundle["labels"]
    source_indices: torch.Tensor = bundle["source_indices"]
    roles: Sequence[str] = bundle["roles"]
    result: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        values = bundle["readouts"][endpoint]
        for evaluator in EVALUATORS:
            logits = values[f"{evaluator.lower()}_logits"]
            ce, probability, margin, predictions = _logit_scores(logits, labels)
            for position in range(600):
                result.append({
                    "endpoint": endpoint,
                    "evaluator": evaluator,
                    "population_position": position,
                    "source_row_index": int(source_indices[position]),
                    "role": roles[position],
                    "label": int(labels[position]),
                    "prediction": int(predictions[position]),
                    "correct": int(predictions[position]) == int(labels[position]),
                    "true_class_logit": float(logits[position, labels[position]]),
                    "best_wrong_logit": float(logits[position].masked_fill(F.one_hot(labels[position], 10).bool(), float("-inf")).max()),
                    "true_class_probability": float(probability[position]),
                    "true_class_cross_entropy": float(ce[position]),
                    "true_minus_best_wrong_logit": float(margin[position]),
                })
    return result


def _row_transition_rows(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels: torch.Tensor = bundle["labels"]
    source_indices: torch.Tensor = bundle["source_indices"]
    roles: Sequence[str] = bundle["roles"]
    scored: dict[tuple[str, str], tuple[torch.Tensor, ...]] = {}
    for endpoint in ENDPOINTS:
        for evaluator in EVALUATORS:
            scored[(endpoint, evaluator)] = _logit_scores(bundle["readouts"][endpoint][f"{evaluator.lower()}_logits"], labels)
    rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    evaluator_changes: dict[str, Any] = {}
    for role in ROLES:
        for label in range(10):
            positions = [i for i in range(600) if roles[i] == role and int(labels[i]) == label]
            for evaluator in EVALUATORS:
                per_endpoint = {endpoint: scored[(endpoint, evaluator)] for endpoint in ENDPOINTS}
                for i in positions:
                    start_ce, start_prob, start_margin, start_pred = per_endpoint["step2200"]
                    a_ce, a_prob, a_margin, a_pred = per_endpoint["A_feature"]
                    b_ce, b_prob, b_margin, b_pred = per_endpoint["B_class_ce"]
                    rows.append({
                        "role": role,
                        "label": label,
                        "evaluator": evaluator,
                        "population_position": i,
                        "source_row_index": int(source_indices[i]),
                        "step2200_prediction": int(start_pred[i]),
                        "A_prediction": int(a_pred[i]),
                        "B_prediction": int(b_pred[i]),
                        "step2200_correct": bool(start_pred[i] == label),
                        "A_correct": bool(a_pred[i] == label),
                        "B_correct": bool(b_pred[i] == label),
                        "B_minus_A_true_class_CE": float(b_ce[i] - a_ce[i]),
                        "B_minus_A_true_class_probability": float(b_prob[i] - a_prob[i]),
                        "B_minus_A_true_minus_best_wrong_margin": float(b_margin[i] - a_margin[i]),
                        "A_minus_step2200_true_class_CE": float(a_ce[i] - start_ce[i]),
                        "A_minus_step2200_margin": float(a_margin[i] - start_margin[i]),
                        "B_minus_step2200_true_class_CE": float(b_ce[i] - start_ce[i]),
                        "B_minus_step2200_margin": float(b_margin[i] - start_margin[i]),
                    })
                a_correct = [bool(per_endpoint["A_feature"][3][i] == label) for i in positions]
                b_correct = [bool(per_endpoint["B_class_ce"][3][i] == label) for i in positions]
                start_correct = [bool(per_endpoint["step2200"][3][i] == label) for i in positions]
                key = f"{role}/class_{label}/{evaluator}"
                split_summary[key] = {
                    "rows": len(positions),
                    "step2200_correct": sum(start_correct),
                    "A_correct": sum(a_correct),
                    "B_correct": sum(b_correct),
                    "B_gain_vs_A": sum(b_correct) - sum(a_correct),
                    "B_gain_vs_step2200": sum(b_correct) - sum(start_correct),
                    "A_to_B_became_correct": sum((not a) and b for a, b in zip(a_correct, b_correct, strict=True)),
                    "A_to_B_became_wrong": sum(a and (not b) for a, b in zip(a_correct, b_correct, strict=True)),
                    "A_to_B_prediction_changed": int(sum(per_endpoint["A_feature"][3][i] != per_endpoint["B_class_ce"][3][i] for i in positions)),
                    "median_B_minus_A_CE": _median([float(per_endpoint["B_class_ce"][0][i] - per_endpoint["A_feature"][0][i]) for i in positions]),
                    "median_B_minus_A_margin": _median([float(per_endpoint["B_class_ce"][2][i] - per_endpoint["A_feature"][2][i]) for i in positions]),
                }
        for focus_label in FOCUS_CLASSES:
            positions = [i for i in range(600) if roles[i] == role and int(labels[i]) == focus_label]
            e_d_ce_a, _, e_d_margin_a, _ = scored[("A_feature", "E_D")]
            e_d_ce_b, _, e_d_margin_b, _ = scored[("B_class_ce", "E_D")]
            e_d_correct = scored[("B_class_ce", "E_D")][3] == labels
            candidates: dict[str, Any] = {}
            for evaluator in ("E_S", "E_I", "E_C"):
                ce_a = scored[("A_feature", evaluator)][0]
                ce_b = scored[("B_class_ce", evaluator)][0]
                overlap = [i for i in positions if ce_b[i] < ce_a[i] and e_d_ce_b[i] >= e_d_ce_a[i]]
                candidates[evaluator] = {"rows": len(overlap), "source_row_indices": [int(source_indices[i]) for i in overlap], "E_D_correct_among_rows": sum(bool(e_d_correct[i]) for i in overlap)}
            evaluator_changes[f"{role}/class_{focus_label}"] = candidates
    return rows, {"by_role_class_evaluator": split_summary, "training_evaluator_CE_down_E_D_CE_not_down_for_focus_classes": evaluator_changes}


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2)


def _verify_and_map_training_history(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    population = bundle["population"]
    source_indices = [int(value) for value in population["source_row_indices"]]
    labels = [int(value) for value in population["labels"]]
    roles = _roles(population)
    mapping_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for arm in ("A_feature", "B_class_ce"):
        records = bundle["histories"][arm]
        if len(records) != 300 or [int(record["update"]) for record in records] != list(range(1, 301)):
            raise EvidenceAnomalyError(f"{arm} does not contain exactly 300 ordered updates")
        usage: Counter[int] = Counter()
        for record in records:
            positions = [int(value) for value in record["fit_positions"]]
            recorded_labels = [int(value) for value in record["labels"]]
            optimizer_labels = [int(value) for value in record["optimizer_labels"]]
            if positions != list(range(int(record["batch_index"]) * 10, int(record["batch_index"]) * 10 + 10)):
                raise EvidenceAnomalyError(f"{arm} update {record['update']} has an unexpected fit batch")
            expected_labels = [labels[position] for position in positions]
            if recorded_labels != expected_labels or optimizer_labels != expected_labels:
                raise EvidenceAnomalyError(f"{arm} update {record['update']} changed its fit labels")
            if any(roles[position] != "fit" for position in positions):
                raise EvidenceAnomalyError(f"{arm} update {record['update']} includes a non-fit row")
            logged_rows = [int(value) for value in record["source_row_indices"]]
            for within_batch, position in enumerate(positions):
                usage[position] += 1
                actual_source = source_indices[position]
                logged = logged_rows[within_batch]
                mapping_rows.append({
                    "arm": arm,
                    "update": int(record["update"]),
                    "batch_index": int(record["batch_index"]),
                    "within_batch": within_batch,
                    "fit_position": position,
                    "recorded_source_row_index": logged,
                    "mapped_source_row_index": actual_source,
                    "source_index_field_was_population_position": logged == position and logged != actual_source,
                    "label": labels[position],
                    "role": roles[position],
                })
            update_rows.append({
                "arm": arm,
                "update": int(record["update"]),
                "cycle": (int(record["update"]) - 1) // 20 + 1,
                "batch_index": int(record["batch_index"]),
                "fit_positions": " ".join(map(str, positions)),
                "source_row_indices_mapped": " ".join(map(str, [source_indices[position] for position in positions])),
                "labels": " ".join(map(str, recorded_labels)),
                "semantic_objective": record["semantic_objective"],
                "semantic_objective_loss": float(record["semantic_objective_loss"]),
                "semantic_feature_L_S": float(record["semantic_feature_losses"]["L_S"]),
                "semantic_feature_L_I": float(record["semantic_feature_losses"]["L_I"]),
                "semantic_feature_L_C": float(record["semantic_feature_losses"]["L_C"]),
                "true_label_CE_L_S": float(record["semantic_class_cross_entropies_true_labels"]["L_S"]),
                "true_label_CE_L_I": float(record["semantic_class_cross_entropies_true_labels"]["L_I"]),
                "true_label_CE_L_C": float(record["semantic_class_cross_entropies_true_labels"]["L_C"]),
                "raw_reconstruction_MSE": float(record["raw_reconstruction_mse"]),
                "L_op": float(record["l_op"]),
                "generator_gradient_norm": float(record["generator_gradient_norm"]),
                "encoder_gradient_norm_read_only": float(record["encoder_gradient_norm_read_only"]),
                "L_dist_generator_gradient_norm": float(record["l_dist_generator_gradient_norm"]),
                "encoder_adam_step": int(record["encoder_adam_step"]),
                "generator_adam_step": int(record["generator_adam_step"]),
            })
        if any(count != 15 for count in usage.values()) or set(usage) != set(range(200)):
            raise EvidenceAnomalyError(f"{arm} fit rows were not each used exactly 15 times")
        coverage[arm] = {"updates": len(records), "fit_rows": len(usage), "updates_per_fit_row": 15, "logged_source_row_field_is_population_position": all(row["recorded_source_row_index"] == row["fit_position"] for row in mapping_rows if row["arm"] == arm)}
    return mapping_rows, update_rows, coverage


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_cycles(update_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "semantic_objective_loss", "semantic_feature_L_S", "semantic_feature_L_I", "semantic_feature_L_C",
        "true_label_CE_L_S", "true_label_CE_L_I", "true_label_CE_L_C", "raw_reconstruction_MSE", "L_op",
        "generator_gradient_norm", "encoder_gradient_norm_read_only",
    )
    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in update_rows:
        grouped[(str(row["arm"]), int(row["cycle"]))].append(row)
    for (arm, cycle), rows in sorted(grouped.items()):
        if len(rows) != 20 or [int(row["batch_index"]) for row in rows] != list(range(20)):
            raise EvidenceAnomalyError(f"{arm} cycle {cycle} does not contain the fixed 20 ordered batches")
        summary: dict[str, Any] = {"arm": arm, "cycle": cycle, "updates": len(rows), "semantic_objective_definition": rows[0]["semantic_objective"]}
        for metric in metrics:
            summary[f"mean_{metric}"] = sum(float(row[metric]) for row in rows) / len(rows)
        result.append(summary)
    if len(result) != 30:
        raise EvidenceAnomalyError("expected 15 fixed-batch cycles per arm")
    return result


def _training_batch_first_last(update_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "semantic_feature_L_S", "semantic_feature_L_I", "semantic_feature_L_C",
        "true_label_CE_L_S", "true_label_CE_L_I", "true_label_CE_L_C",
        "raw_reconstruction_MSE", "L_op", "generator_gradient_norm", "encoder_gradient_norm_read_only",
    )
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in update_rows:
        grouped[(str(row["arm"]), int(row["batch_index"]))].append(row)
    result = []
    for (arm, batch_index), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["cycle"]))
        if len(rows) != 15 or int(rows[0]["cycle"]) != 1 or int(rows[-1]["cycle"]) != 15:
            raise EvidenceAnomalyError(f"{arm} batch {batch_index} does not have first and last records across all 15 cycles")
        output: dict[str, Any] = {"arm": arm, "batch_index": batch_index, "fit_positions": rows[0]["fit_positions"], "mapped_source_row_indices": rows[0]["source_row_indices_mapped"]}
        for metric in metrics:
            first = float(rows[0][metric])
            last = float(rows[-1][metric])
            output[f"first_{metric}"] = first
            output[f"last_{metric}"] = last
            output[f"last_minus_first_{metric}"] = last - first
        result.append(output)
    if len(result) != 40:
        raise EvidenceAnomalyError("expected first/last comparison for 20 fixed batches per arm")
    return result


def _plot_pages(bundle: Mapping[str, Any], output_dir: Path) -> list[str]:
    readouts = bundle["readouts"]
    labels: torch.Tensor = bundle["labels"]
    source_indices: torch.Tensor = bundle["source_indices"]
    roles: Sequence[str] = bundle["roles"]
    target = F.adaptive_avg_pool2d(readouts["step2200"]["target_slm_intensity"].clamp(0, 1), (28, 28))[:, 0].numpy()
    views: dict[str, Any] = {"target_200x200_pooled": target}
    for endpoint in ENDPOINTS:
        views[endpoint] = F.adaptive_avg_pool2d(readouts[endpoint]["raw_cmos"].clamp(0, 1), (28, 28))[:, 0].numpy()
    pred = {endpoint: {evaluator: readouts[endpoint][f"{evaluator.lower()}_logits"].argmax(dim=1).tolist() for evaluator in EVALUATORS} for endpoint in ENDPOINTS}
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    saved: list[str] = []
    for label in DISPLAY_CLASSES:
        fit_and_historic, confirmation = _figure_page_positions(roles, labels, label)
        for page_name, positions in (("fit_historic", fit_and_historic), ("confirmation", confirmation)):
            if len(positions) != 30:
                raise EvidenceAnomalyError(f"class {label} {page_name} page is missing rows")
            fig, axes = plt.subplots(30, 5, figsize=(9, 20), squeeze=False, gridspec_kw={"width_ratios": [1.55, 1, 1, 1, 1]})
            column_titles = ["row / role / E_S,E_I,E_C,E_D", "target pooled", "step2200", "A feature", "B class CE"]
            for column, title in enumerate(column_titles):
                axes[0, column].set_title(title, fontsize=8)
            for row_index, position in enumerate(positions):
                role_label = {"fit": "fit", "historic_holdout": "historic", "confirmation": "confirmation"}[roles[position]]
                ed_labels = "/".join(str(pred[endpoint][evaluator][position]) for evaluator in EVALUATORS)
                axes[row_index, 0].text(0.98, 0.5, f"{int(source_indices[position])}  {role_label}  {ed_labels}", ha="right", va="center", fontsize=5.3)
                axes[row_index, 0].axis("off")
                for column, key in enumerate(("target_200x200_pooled", "step2200", "A_feature", "B_class_ce"), start=1):
                    axes[row_index, column].imshow(views[key][position], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    axes[row_index, column].axis("off")
            fig.suptitle(f"Class {label}: {page_name}; fixed [0,1], no per-image rescaling", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.992))
            path = figures_dir / f"class_{label}_{page_name}.png"
            fig.savefig(path, dpi=115)
            plt.close(fig)
            saved.append(path.relative_to(output_dir).as_posix())
    if len(saved) != 10:
        raise EvidenceAnomalyError("expected ten complete image review pages")
    return saved


def _figure_page_positions(roles: Sequence[str], labels: torch.Tensor, label: int) -> tuple[list[int], list[int]]:
    fit_and_historic = [i for i, role in enumerate(roles) if role in ("fit", "historic_holdout") and int(labels[i]) == label]
    confirmation = [i for i, role in enumerate(roles) if role == "confirmation" and int(labels[i]) == label]
    if len(fit_and_historic) != 30 or len(confirmation) != 30:
        raise EvidenceAnomalyError(f"class {label} does not have 30 rows in both figure pages")
    return fit_and_historic, confirmation


def _image_metric_deltas(bundle: Mapping[str, Any]) -> dict[str, Any]:
    metrics = bundle["row_metrics"]
    result: dict[str, Any] = {}
    roles = bundle["roles"]
    labels: torch.Tensor = bundle["labels"]
    for role in ROLES:
        for label in range(10):
            positions = [i for i in range(600) if roles[i] == role and int(labels[i]) == label]
            entry: dict[str, Any] = {}
            for metric in ("raw_mse", "raw_foreground_mse", "raw_background_mse", "evaluator_view_foreground_mse", "evaluator_view_background_mse", "raw_fraction_above_one", "raw_zero_image", "raw_constant_image"):
                entry[metric] = {endpoint: float(metrics[endpoint][metric].index_select(0, torch.tensor(positions)).to(torch.float64).mean()) for endpoint in ENDPOINTS}
                entry[f"{metric}_B_minus_A"] = entry[metric]["B_class_ce"] - entry[metric]["A_feature"]
            result[f"{role}/class_{label}"] = entry
    return result


def _build_report(bundle: Mapping[str, Any], verification: Mapping[str, Any], coverage: Mapping[str, Any], score_summary: Mapping[str, Any], image_deltas: Mapping[str, Any], figure_paths: Sequence[str]) -> dict[str, Any]:
    report = bundle["report"]
    endpoint = {arm: report["evaluator_metrics"][arm]["E_D"]["by_role"]["confirmation"] for arm in ENDPOINTS}
    focus_changes = {}
    for label in FOCUS_CLASSES:
        focus_changes[str(label)] = {
            arm: {evaluator: {"correct": report["evaluator_metrics"][arm][evaluator]["by_role"]["confirmation"]["per_class_correct"][label], "recall": report["evaluator_metrics"][arm][evaluator]["by_role"]["confirmation"]["per_class_recall"][label]} for evaluator in EVALUATORS}
            for arm in ENDPOINTS
        }
    ed_start = float(endpoint["step2200"]["macro_recall"])
    ed_a = float(endpoint["A_feature"]["macro_recall"])
    ed_b = float(endpoint["B_class_ce"]["macro_recall"])
    findings = [
        "B changed evaluator scores during the 300-step fit, including lower true-label CE on the fixed training-batch cycle averages; this describes the logged objective response and is not itself semantic image improvement.",
        "Neither class 1 nor class 7 gained E_D-correct confirmation rows: start, A, and B are all 0/30. The parent paired McNemar/Holm results remain p=1 for both classes.",
        f"Confirmation E_D macro recall is {ed_start:.4f} at step2200, {ed_a:.4f} for A, and {ed_b:.4f} for B; B minus A is {ed_b-ed_a:.4f}, below the parent preregistered +0.05 threshold.",
        "Evaluator responses diverged by class: B raises E_I and E_C class-7 correct counts relative to A, while E_D remains 0/30; for class 1 E_C correct count decreases from A to B. Evaluator score improvements cannot be treated as E_D transfer.",
        "The saved evidence does not distinguish semantic-gradient conflict, inherited Adam-state direction, or limitations in the generated optical presentation. No cause or architecture-capacity conclusion is assigned.",
    ]
    macro_deltas = {
        "step2200": ed_start,
        "A_feature": ed_a,
        "B_class_ce": ed_b,
        "A_minus_step2200": ed_a - ed_start,
        "B_minus_step2200": ed_b - ed_start,
        "B_minus_A": ed_b - ed_a,
    }
    return {
        "schema": "m0_semantic_target_evidence_review.v1",
        "task_id": TASK_ID,
        "parent_task_id": PARENT_TASK_ID,
        "status": "EVIDENCE_REVIEW_COMPLETED",
        "parent_status_preserved": report["status"],
        "parent_classification_preserved": report["classification"],
        "scientific_gate_rejudged": False,
        "training_updates": 0,
        "optical_forwards": 0,
        "evaluator_forwards": 0,
        "gradient_computations": 0,
        "probe_fits": 0,
        "resampling": 0,
        "scope": {"rows": 600, "roles": {role: bundle["roles"].count(role) for role in ROLES}, "endpoints": list(ENDPOINTS), "evaluators": list(EVALUATORS), "parent_training_updates_per_arm": 300, "input_paths": report["research_run_identity"].get("inputs", {})},
        "reconciliation": verification,
        "training_log_mapping": coverage,
        "confirmation_E_D": {arm: {"macro_recall": endpoint[arm]["macro_recall"], "correct_per_class": endpoint[arm]["per_class_correct"], "predicted_class_counts": endpoint[arm]["predicted_class_counts"]} for arm in ENDPOINTS},
        "confirmation_E_D_macro_recall": macro_deltas,
        "confirmation_focus_class_by_evaluator": focus_changes,
        "paired_tests_recomputed_from_logits": verification["paired_tests"],
        "training_scores": {"definition_note": "A and B semantic_objective_loss have different definitions and scales; compare shared monitored metrics and same batches, not cross-objective scalar magnitude.", "cycles": 15, "updates_per_cycle": 20, "first_and_last_cycle_means": score_summary["first_last_cycle_means"], "all_cycle_artifact": "training_cycle_summary.csv"},
        "row_score_artifact": "row_evaluator_scores.csv",
        "row_transition_artifact": "row_prediction_score_transitions.csv",
        "training_log_row_mapping_artifact": "training_log_source_row_mapping.csv",
        "training_update_artifact": "training_updates_with_mapped_rows.csv",
        "same_batch_first_last_artifact": "training_batch_first_last.csv",
        "image_error_summary": image_deltas,
        "figure_artifacts": list(figure_paths),
        "findings": findings,
        "unresolved_causes": ["semantic-gradient conflict", "inherited Adam-state direction", "optical image presentation"],
        "one_next_measurement_question": "On the saved fit population, do per-class E_D true-minus-best-wrong gradients align or conflict with each training evaluator's class-CE gradients after the inherited Adam transform?",
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "stop": True,
    }


def run_review(artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"parent artifact directory does not exist: {artifact_dir}")
    bundle = _load_parent_artifacts(artifact_dir)
    verification = _verify_readouts(bundle)
    row_scores = _row_score_rows(bundle)
    transition_rows, score_summary = _row_transition_rows(bundle)
    mapping_rows, update_rows, coverage = _verify_and_map_training_history(bundle)
    cycle_rows = _training_cycles(update_rows)
    batch_comparison_rows = _training_batch_first_last(update_rows)
    image_deltas = _image_metric_deltas(bundle)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "row_evaluator_scores.csv", row_scores)
    _write_csv(output_dir / "row_prediction_score_transitions.csv", transition_rows)
    _write_csv(output_dir / "training_log_source_row_mapping.csv", mapping_rows)
    _write_csv(output_dir / "training_updates_with_mapped_rows.csv", update_rows)
    _write_csv(output_dir / "training_cycle_summary.csv", cycle_rows)
    _write_csv(output_dir / "training_batch_first_last.csv", batch_comparison_rows)
    figure_paths = _plot_pages(bundle, output_dir)
    score_summary["first_last_cycle_means"] = {}
    for arm in ("A_feature", "B_class_ce"):
        arm_cycles = [row for row in cycle_rows if row["arm"] == arm]
        score_summary["first_last_cycle_means"][arm] = {"first": arm_cycles[0], "last": arm_cycles[-1]}
    result = _build_report(bundle, verification, coverage, score_summary, image_deltas, figure_paths)
    _write_json(output_dir / "score_transition_summary.json", score_summary)
    _write_json(output_dir / "report.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review saved LightGen semantic-target evidence without model or evaluator execution.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_review(args.artifact_dir, args.output_dir)
    print(json.dumps({"task_id": result["task_id"], "status": result["status"], "output_dir": str(args.output_dir.resolve()), "next_action": result["next_action"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
