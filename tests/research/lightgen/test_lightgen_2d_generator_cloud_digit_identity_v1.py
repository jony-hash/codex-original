from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import audit_lightgen_2d_generator_cloud_digit_identity_v1 as task


def _dev_rows() -> list[dict[str, str]]:
    return [
        {"role": "dev", "position": str(position), "source_row_index": str(10_000 + position), "label": str(position % 10)}
        for position in range(task.source.DEV_ROWS)
    ]


def test_dev_index_rejects_missing_duplicate_rows_wrong_roles_and_class_population() -> None:
    rows = _dev_rows()
    assert len(task._index_dev_rows(rows)) == task.source.DEV_ROWS

    duplicate = [*rows[:-1], {**rows[-1], "source_row_index": rows[0]["source_row_index"]}]
    with pytest.raises(task.SavedEvidenceError, match="source rows repeat"):
        task._index_dev_rows(duplicate)

    wrong_role = [*rows[:-1], {**rows[-1], "role": "train"}]
    with pytest.raises(task.SavedEvidenceError, match="role or row count"):
        task._index_dev_rows(wrong_role)

    wrong_position = [*rows[:-1], {**rows[-1], "position": rows[0]["position"]}]
    with pytest.raises(task.SavedEvidenceError, match="positions are missing or repeated"):
        task._index_dev_rows(wrong_position)

    wrong_class = [*rows[:-2], {**rows[-2], "label": "9"}, rows[-1]]
    with pytest.raises(task.SavedEvidenceError, match="class population differs"):
        task._index_dev_rows(wrong_class)


def test_conditional_index_rejects_cross_class_draw_collision_and_incomplete_keys() -> None:
    rows = [
        {"requested_class": str(digit), "draw_index": str(draw)}
        for digit in range(10)
        for draw in range(2)
    ]
    indexed = task._index_sample_rows(rows, expected_per_class=2)
    assert len(indexed) == 20

    collision = [*rows[:-1], {**rows[-1], "requested_class": rows[0]["requested_class"], "draw_index": rows[0]["draw_index"]}]
    with pytest.raises(task.SavedEvidenceError, match="keys collide or are incomplete"):
        task._index_sample_rows(collision, expected_per_class=2)


def test_saved_logits_must_match_the_csv_predictions() -> None:
    logits = torch.zeros((2, 10))
    logits[0, 2] = 3
    logits[1, 4] = 3
    task._verify_logits_predictions(logits, [2, 4], "E_D")
    with pytest.raises(task.SavedEvidenceError, match="disagree with the per-row CSV"):
        task._verify_logits_predictions(logits, [2, 5], "E_D")


def test_numpy_qfg_uses_whole_image_mass_normalization_and_all_components() -> None:
    target = torch.zeros((1, 200, 200), dtype=torch.float64)
    target[:, 70:130, 70:130] = 1.0
    raw = target * 0.6
    raw[:, 50:55, 50:55] = 0.2

    observed = task._numpy_qfg(raw, target)
    scaled = task._numpy_qfg(raw * 7.0, target)

    assert observed == pytest.approx(scaled, rel=1e-12, abs=1e-14)
    assert observed[0] > 0
    assert observed[1] > 0
    assert observed[2] == pytest.approx(observed[0] + observed[1], rel=1e-12, abs=1e-14)
    assert observed[3] > 0
    assert observed[4] == 0
    assert observed[5] > 0


def test_dev_shape_energy_decomposition_includes_both_within_terms() -> None:
    generated = torch.tensor([[0.0, 1.0], [0.1, 0.9]], dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    result = task._energy_distance(generated, target, task._mean_distance(target, target))
    cross = np.mean([[np.linalg.norm(x - y) for y in target.numpy()] for x in generated.numpy()])
    within_output = np.mean([[np.linalg.norm(x - y) for y in generated.numpy()] for x in generated.numpy()])
    within_target = np.mean([[np.linalg.norm(x - y) for y in target.numpy()] for x in target.numpy()])
    assert result["cross_distance"] == pytest.approx(cross)
    assert result["within_output_distance"] == pytest.approx(within_output)
    assert result["within_target_distance"] == pytest.approx(within_target)
    assert result["energy_distance"] == pytest.approx(max(0.0, 2 * cross - within_output - within_target))


def test_analysis_csv_writer_round_trips_union_of_empirical_and_sample_columns(tmp_path: Path) -> None:
    destination = tmp_path / "summary.csv"
    task._write_csv(destination, [{"population": "dev", "q_fg": 0.1}, {"population": "sample", "generator_input_power": 0.2}])
    content = destination.read_text(encoding="utf-8-sig")
    assert "q_fg" in content and "generator_input_power" in content
    assert content.count("\n") == 3


def test_shape_gain_without_identity_gain_routes_to_confusion_diagnosis() -> None:
    summary = [
        {"population": population, "training_seed": seed, "focus_delta_B_minus_A_e_d_correct": -0.01, "focus_delta_B_minus_A_e_d_identity_margin": -0.4}
        for population in ("empirical_dev", "conditional_sample")
        for seed in task.TRAIN_SEEDS
    ]
    shape = []
    for seed in task.TRAIN_SEEDS:
        for arm, distance in (("A", 0.014), ("B", 0.012)):
            for digit in task.FOCUS_CLASSES:
                shape.append({"scenario": f"seed_{seed}_{arm}_step6700", "requested_class": digit, "own_class_energy_distance": distance, "class_separation_margin": 0.004})

    recommendation = task._next_step_from_descriptive_summary(summary, shape)

    assert recommendation.startswith("DEV_shape_distance_improves_while_focus_identity_proxies_do_not")


def test_focus_identity_pairs_keep_the_source_key_and_prediction_transition() -> None:
    rows = [
        {"population": "empirical_dev", "scenario": f"seed_{task.TRAIN_SEEDS[0]}_{arm}_step6700", "training_seed": task.TRAIN_SEEDS[0], "source_row_index": 12_345, "sample_seed": "", "requested_class": 2, "draw_index": "", "e_d_prediction": prediction, "e_d_identity_margin": margin, "raw_cmos_integral": 1.0e-8, "q_fg": q_fg}
        for arm, prediction, margin, q_fg in (("A", 2, 0.5, 0.4), ("B", 8, -0.1, 0.3))
    ]

    paired = task._focus_identity_pairs(rows)

    assert len(paired) == 1
    assert paired[0]["source_row_index"] == 12_345
    assert paired[0]["paired_outcome"] == "A_correct_B_wrong"
    assert paired[0]["delta_B_minus_A_e_d_identity_margin"] == pytest.approx(-0.6)
    assert paired[0]["delta_B_minus_A_q_fg"] == pytest.approx(-0.1)


def test_saved_report_must_consume_readback_summary_and_raw_parent_decisions() -> None:
    summary = [{"population": "empirical_dev", "training_seed": task.TRAIN_SEEDS[0]}]
    decisions = {str(seed): {"passes_both_conditions": True} for seed in task.TRAIN_SEEDS}
    report = {
        "paired_identity_summary_from_saved_csv": summary,
        "q_fg_seed_decisions_replayed": decisions,
        "parent_q_fg_decision_replayed_from_raw": True,
        "parent_classification_preserved": "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS",
    }

    assert task._verify_saved_report_consumers(report, summary, decisions)["passed"] is True

    with pytest.raises(task.SavedEvidenceError, match="identity summary read from CSV"):
        task._verify_saved_report_consumers({**report, "paired_identity_summary_from_saved_csv": []}, summary, decisions)
