from __future__ import annotations

from collections import Counter
import json

import pytest
import torch

from scripts.research.lightgen import review_lightgen_2d_semantic_target_evidence_v1 as review


def _population() -> dict[str, object]:
    labels = torch.cat((torch.arange(10).repeat(20), torch.arange(10).repeat(10), torch.arange(10).repeat_interleave(30))).tolist()
    indices = list(range(10000, 10600))
    return {
        "task_id": review.PARENT_TASK_ID,
        "labels": labels,
        "source_row_indices": indices,
        "confirmation_indices": indices[300:],
        "roles": ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300,
        "original_fit_rows": 200,
        "historic_holdout_rows": 100,
        "new_confirmation_rows": 300,
    }


def _logits(labels: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((labels.numel(), 10), dtype=torch.float32)
    result.scatter_(1, labels[:, None], 2.0)
    return result


def _make_readouts(labels: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    endpoints = {}
    for endpoint in review.ENDPOINTS:
        endpoints[endpoint] = {f"{evaluator.lower()}_logits": _logits(labels) for evaluator in review.EVALUATORS}
    # A held-out class-7 row is improved by the three training evaluators in B,
    # while E_D's saved prediction and score stay unchanged.
    position = 300 + 7 * 30
    for evaluator in ("E_S", "E_I", "E_C"):
        for endpoint in ("step2200", "A_feature"):
            logits = endpoints[endpoint][f"{evaluator.lower()}_logits"]
            logits[position, 7] = 0.0
            logits[position, 8] = 1.0
        logits = endpoints["B_class_ce"][f"{evaluator.lower()}_logits"]
        logits[position, 7] = 3.0
        logits[position, 8] = 1.0
    for endpoint in review.ENDPOINTS:
        logits = endpoints[endpoint]["e_d_logits"]
        logits[position, 7] = 0.0
        logits[position, 8] = 1.0
    return endpoints


def _history(population: dict[str, object], *, wrong_logged_rows: bool = True) -> dict[str, list[dict[str, object]]]:
    labels = [int(value) for value in population["labels"]]
    histories = {}
    for arm in ("A_feature", "B_class_ce"):
        records = []
        for update in range(1, 301):
            batch_index = (update - 1) % 20
            positions = list(range(batch_index * 10, batch_index * 10 + 10))
            records.append({
                "update": update,
                "batch_index": batch_index,
                "fit_positions": positions,
                "source_row_indices": positions if wrong_logged_rows else [10000 + value for value in positions],
                "labels": [labels[position] for position in positions],
                "optimizer_labels": [labels[position] for position in positions],
                "semantic_objective": "feature" if arm == "A_feature" else "class_ce",
                "semantic_objective_loss": float(update),
                "semantic_feature_losses": {"L_S": 1.0, "L_I": 2.0, "L_C": 3.0},
                "semantic_class_cross_entropies_true_labels": {"L_S": 4.0, "L_I": 5.0, "L_C": 6.0},
                "raw_reconstruction_mse": 0.1,
                "l_op": 0.2,
                "generator_gradient_norm": 0.3,
                "encoder_gradient_norm_read_only": 0.4,
                "l_dist_generator_gradient_norm": 0.0,
                "encoder_adam_step": 2200,
                "generator_adam_step": 2200 + update,
            })
        histories[arm] = records
    return histories


def test_population_roles_labels_and_order_are_checked() -> None:
    population = _population()
    labels, indices, roles = review._verify_population(population)
    assert len(labels) == len(indices) == len(roles) == 600
    assert Counter(roles) == {"fit": 200, "historic_holdout": 100, "confirmation": 300}

    wrong_label = dict(population, labels=[1, *population["labels"][1:]])
    with pytest.raises(review.EvidenceAnomalyError, match="fit population"):
        review._verify_population(wrong_label)

    wrong_confirmation_order = dict(population, confirmation_indices=list(reversed(population["confirmation_indices"])))
    with pytest.raises(review.EvidenceAnomalyError, match="confirmation_indices"):
        review._verify_population(wrong_confirmation_order)


def test_endpoint_row_order_and_labels_must_match_population() -> None:
    population = _population()
    labels, indices, _ = review._verify_population(population)
    labels_tensor = torch.tensor(labels)
    indices_tensor = torch.tensor(indices)
    readout = {"labels": labels_tensor.clone(), "source_indices": indices_tensor.clone()}
    row_metrics = {"labels": labels_tensor.clone(), "source_indices": indices_tensor.clone()}
    review._verify_row_identity("A_feature", readout, row_metrics, labels_tensor, indices_tensor)

    readout["source_indices"][[0, 1]] = readout["source_indices"][[1, 0]]
    with pytest.raises(review.EvidenceAnomalyError, match="row ordering/labels"):
        review._verify_row_identity("A_feature", readout, row_metrics, labels_tensor, indices_tensor)

    readout["source_indices"] = indices_tensor.clone()
    row_metrics["labels"][0] = 1
    with pytest.raises(review.EvidenceAnomalyError, match="row metric labels"):
        review._verify_row_identity("A_feature", readout, row_metrics, labels_tensor, indices_tensor)


def test_true_class_ce_probability_margin_and_prediction_are_recomputed() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 3.0]], dtype=torch.float32)
    labels = torch.tensor([1, 1])
    ce, probability, margin, predictions = review._logit_scores(logits, labels)
    assert predictions.tolist() == [1, 2]
    assert probability[0] > probability[1]
    assert margin.tolist() == pytest.approx([1.0, -2.0])
    assert ce[0] < ce[1]


def test_role_specific_changes_surface_evaluator_disagreement_without_claiming_ed_gain() -> None:
    population = _population()
    labels, indices, roles = review._verify_population(population)
    bundle = {"labels": torch.tensor(labels), "source_indices": torch.tensor(indices), "roles": roles, "readouts": _make_readouts(torch.tensor(labels))}
    rows, summary = review._row_transition_rows(bundle)
    key = "confirmation/class_7/E_S"
    assert summary["by_role_class_evaluator"][key]["A_correct"] == 29
    assert summary["by_role_class_evaluator"][key]["B_correct"] == 30
    disagreement = summary["training_evaluator_CE_down_E_D_CE_not_down_for_focus_classes"]["confirmation/class_7"]
    assert disagreement["E_S"]["rows"] == 1
    assert disagreement["E_S"]["E_D_correct_among_rows"] == 0
    assert len(rows) == 600 * 4
    assert all(row["role"] == "confirmation" for row in rows if row["label"] == 7 and row["evaluator"] == "E_D" and row["population_position"] >= 300)
    json.dumps(summary)


def test_logged_population_positions_are_mapped_to_real_source_rows_and_each_fit_row_used_15_times() -> None:
    population = _population()
    bundle = {"population": population, "histories": _history(population)}
    mapping, updates, coverage = review._verify_and_map_training_history(bundle)
    assert len(mapping) == 6000
    assert len(updates) == 600
    assert mapping[0]["recorded_source_row_index"] == 0
    assert mapping[0]["mapped_source_row_index"] == 10000
    assert mapping[0]["source_index_field_was_population_position"] is True
    assert coverage["A_feature"]["fit_rows"] == 200
    assert coverage["B_class_ce"]["updates_per_fit_row"] == 15

    bad = _history(population)
    bad["A_feature"][0]["labels"][0] = 9
    with pytest.raises(review.EvidenceAnomalyError, match="changed its fit labels"):
        review._verify_and_map_training_history({"population": population, "histories": bad})


def test_training_cycle_summary_keeps_fixed_batch_order_and_reports_same_objective_metrics() -> None:
    updates = []
    for arm in ("A_feature", "B_class_ce"):
        for update in range(1, 301):
            updates.append({"arm": arm, "update": update, "cycle": (update - 1) // 20 + 1, "batch_index": (update - 1) % 20, "semantic_objective": arm, "semantic_objective_loss": float(update), "semantic_feature_L_S": 1.0, "semantic_feature_L_I": 2.0, "semantic_feature_L_C": 3.0, "true_label_CE_L_S": 4.0, "true_label_CE_L_I": 5.0, "true_label_CE_L_C": 6.0, "raw_reconstruction_MSE": 0.1, "L_op": 0.2, "generator_gradient_norm": 0.3, "encoder_gradient_norm_read_only": 0.4})
    result = review._training_cycles(updates)
    assert len(result) == 30
    assert result[0]["mean_semantic_objective_loss"] == pytest.approx(10.5)
    assert result[0]["mean_true_label_CE_L_S"] == pytest.approx(4.0)
    updates[1]["batch_index"] = 19
    with pytest.raises(review.EvidenceAnomalyError, match="fixed 20 ordered batches"):
        review._training_cycles(updates)


def test_batch_first_last_summary_pairs_the_same_fixed_batch_across_all_cycles() -> None:
    updates = []
    for arm in ("A_feature", "B_class_ce"):
        for update in range(1, 301):
            updates.append({"arm": arm, "update": update, "cycle": (update - 1) // 20 + 1, "batch_index": (update - 1) % 20, "fit_positions": "0 1 2", "source_row_indices_mapped": "10000 10001 10002", "semantic_feature_L_S": 1.0, "semantic_feature_L_I": 2.0, "semantic_feature_L_C": 3.0, "true_label_CE_L_S": 4.0, "true_label_CE_L_I": 5.0, "true_label_CE_L_C": 6.0, "raw_reconstruction_MSE": float(update), "L_op": 0.2, "generator_gradient_norm": 0.3, "encoder_gradient_norm_read_only": 0.4})
    rows = review._training_batch_first_last(updates)
    batch_zero = rows[0]
    assert batch_zero["arm"] == "A_feature"
    assert batch_zero["batch_index"] == 0
    assert batch_zero["first_raw_reconstruction_MSE"] == pytest.approx(1.0)
    assert batch_zero["last_raw_reconstruction_MSE"] == pytest.approx(281.0)
    assert batch_zero["last_minus_first_raw_reconstruction_MSE"] == pytest.approx(280.0)


def test_five_class_figure_pages_cover_all_rows_by_role_without_overlap() -> None:
    population = _population()
    labels = torch.tensor(population["labels"])
    roles = population["roles"]
    for label in review.DISPLAY_CLASSES:
        first, confirmation = review._figure_page_positions(roles, labels, label)
        assert len(first) == len(confirmation) == 30
        assert Counter(roles[position] for position in first) == {"fit": 20, "historic_holdout": 10}
        assert all(roles[position] == "confirmation" for position in confirmation)
        assert set(first).isdisjoint(confirmation)


def test_parent_classification_recalculation_matches_and_rejects_changed_counts() -> None:
    population = _population()
    labels = torch.tensor(population["labels"], dtype=torch.int64)
    roles = population["roles"]
    readouts = _make_readouts(labels)
    evaluator_metrics = {}
    for endpoint in review.ENDPOINTS:
        evaluator_metrics[endpoint] = {}
        for evaluator in review.EVALUATORS:
            logits = readouts[endpoint][f"{evaluator.lower()}_logits"]
            evaluator_metrics[endpoint][evaluator] = {"by_role": {}}
            for role in review.ROLES:
                positions = torch.tensor([i for i, value in enumerate(roles) if value == role])
                result = review._classification_metrics(logits[positions], labels[positions])
                evaluator_metrics[endpoint][evaluator]["by_role"][role] = result
    report = {"task_id": review.PARENT_TASK_ID, "status": "VALID_FAILED", "classification": "NOT_SUPPORTED_WITHIN_SCOPE", "evaluator_metrics": evaluator_metrics}
    result = review._verify_parent_metrics(report, readouts, labels, roles)
    assert result["B_class_ce"]["E_D"]["confirmation"]["per_class_correct"][7] == 29

    report["evaluator_metrics"]["step2200"]["E_D"]["by_role"]["fit"]["per_class_correct"][1] += 1
    with pytest.raises(review.EvidenceAnomalyError, match="per_class_correct"):
        review._verify_parent_metrics(report, readouts, labels, roles)
