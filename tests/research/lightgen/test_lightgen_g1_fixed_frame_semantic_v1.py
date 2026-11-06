from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_semantic_v1 as study


def test_per_sample_gauge_rotation_can_make_the_raw_mean_unreachable() -> None:
    response = torch.tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex128)
    commands = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    raw = commands.to(torch.complex128) @ response.T
    phases = torch.angle(raw[torch.arange(2), raw.abs().argmax(dim=1)])
    gauged = raw * torch.exp(-1j * phases[:, None])
    raw_mean = raw.mean(0)
    gauged_mean = gauged.mean(0)

    assert torch.allclose(raw_mean, (commands.mean(0).to(torch.complex128) @ response.T))
    assert torch.allclose(gauged_mean, torch.tensor([0.5, 0.5], dtype=torch.complex128))
    # The second output of B a is always imaginary for real nonnegative a, so this gauge mean is unreachable.
    assert torch.linalg.vector_norm(response @ torch.tensor([0.5, 0.5], dtype=torch.complex128)
                                    - gauged_mean) > 0.1


def test_raw_phase_features_use_frozen_scale_and_interleaved_real_imag() -> None:
    coefficients = torch.zeros((1, 100), dtype=torch.complex128)
    coefficients[0, 0] = 1.0 + 2.0j
    coefficients[0, 1] = 6.0 + 8.0j
    scale = torch.ones(100, dtype=torch.float64)
    scale[1] = 16.0
    features = study._raw_features(coefficients, scale)
    assert torch.allclose(features[0, :4], torch.tensor([1.0, 2.0, 1.5, 2.0], dtype=torch.float64))


def test_registered_schedule_provides_exactly_750_exposures_per_target() -> None:
    config = study._load_config(study.DEFAULT_CONFIG)
    schedule = study._schedule(config)
    assert tuple(schedule.shape) == (15_000, 10)
    assert torch.equal(torch.bincount(schedule.flatten(), minlength=study.TRAIN_ROWS),
                       torch.full((study.TRAIN_ROWS,), 750))
    for batch in schedule:
        assert torch.equal(batch.div(20, rounding_mode="floor"), torch.arange(10))


def test_source_checkpoint_binding_accepts_cloud_path_for_same_registered_input() -> None:
    configured = study.ROOT / (
        "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/"
        "autodl_backup_attempt_01/terminal_checkpoint.pt")
    saved_cloud_path = (
        "/root/autodl-tmp/codex-original/outputs/20261027_lightgen_g1_slm_projected_mapping_v2/"
        "runs/autodl_backup_attempt_01/terminal_checkpoint.pt")

    assert saved_cloud_path != str(configured)
    assert study._registered_source_path_matches(saved_cloud_path, configured) is True
    assert study._registered_source_path_matches(
        saved_cloud_path.replace("autodl_backup_attempt_01", "autodl_backup_other"),
        configured) is False


def test_box_loss_matches_independent_numpy_and_descends_on_both_bounds() -> None:
    checked = study._check_box_loss(torch.device("cpu"))
    assert checked["passed"] is True
    assert checked["production_value"] == pytest.approx(checked["independent_numpy_value"], abs=1.0e-18)
    assert checked["gradient_descent_direction_passed"] is True


def test_ready_consumers_require_executed_freeze_and_optimizer_invariants() -> None:
    group_only_receipt = SimpleNamespace(
        status="READY",
        authorization_id=study.AUTHORITY,
        selected_engine="response_matrix",
        checks={
            "CORRECTNESS_INVARIANTS": {"passed": True},
            "RESOURCE_ENVELOPE": {"passed": True},
            "IMPLEMENTATION_SELECTION": {"passed": True},
        },
    )
    assert all(group_only_receipt.checks[key]["passed"] for key in group_only_receipt.checks)
    assert study._ready_receipt_checks_pass(group_only_receipt) is False

    receipt = SimpleNamespace(
        status="READY",
        authorization_id=study.AUTHORITY,
        selected_engine="response_matrix",
        checks={
            "CORRECTNESS_INVARIANTS": {
                "passed": True,
                "frozen_optics_projection_gain_prior_and_evaluators_unchanged": True,
                "optimizer_contains_only_mapper_parameters": True,
            },
            "RESOURCE_ENVELOPE": {"passed": True},
            "IMPLEMENTATION_SELECTION": {"passed": True, "frozen_inputs_unchanged": True},
        },
    )
    assert study._ready_receipt_checks_pass(receipt) is True

    receipt.checks["CORRECTNESS_INVARIANTS"][
        "frozen_optics_projection_gain_prior_and_evaluators_unchanged"] = False
    assert study._ready_receipt_checks_pass(receipt) is False
    receipt.checks["CORRECTNESS_INVARIANTS"][
        "frozen_optics_projection_gain_prior_and_evaluators_unchanged"] = True
    receipt.checks["IMPLEMENTATION_SELECTION"]["frozen_inputs_unchanged"] = False
    assert study._ready_receipt_checks_pass(receipt) is False
    receipt.checks["IMPLEMENTATION_SELECTION"]["frozen_inputs_unchanged"] = True
    receipt.checks["CORRECTNESS_INVARIANTS"]["optimizer_contains_only_mapper_parameters"] = False
    assert study._ready_receipt_checks_pass(receipt) is False


def test_fixed_frame_semantic_gate_checks_both_accuracy_and_each_class_diversity() -> None:
    diversity = {str(label): {"mean_pairwise_normalized_l2": 0.5} for label in range(10)}
    reference = {"evaluators": {family: {"accuracy": 0.75, "per_class_recall": [0.75] * 10}
                                for family in ("ED", "EC")}}
    candidate = {"evaluators": {family: {"accuracy": 0.74, "per_class_recall": [0.74] * 10}
                                for family in ("ED", "EC")}}
    summary = {"conditions": {"old_g1_direct_decoder": reference,
                              "fixed_frame_mapped": candidate},
               "same_class_diversity": {
                   "old_g1_direct_decoder": {"by_class": diversity},
                   "fixed_frame_mapped": {"by_class": {
                       str(label): {"mean_pairwise_normalized_l2": 0.4}
                       for label in range(10)}}}}
    config = {"measurement": {"semantic_accuracy_drop_max_absolute": 0.02,
                               "class_diversity_retention_min": 0.8}}

    assert study._semantic_final_gate(summary, config)["passed"] is True
    summary["same_class_diversity"]["fixed_frame_mapped"]["by_class"]["7"][
        "mean_pairwise_normalized_l2"] = 0.399
    assert study._semantic_final_gate(summary, config)["passed"] is False


def test_saved_readout_semantics_match_the_frozen_raw_cmos_consumer() -> None:
    torch.manual_seed(41)
    labels = torch.arange(10).repeat_interleave(2)
    cohort = {"labels": labels}
    readouts = torch.rand((20, 1, 28, 28), dtype=torch.float32)
    raw_cmos = readouts.double() / 6.25e-12
    def make_evaluator() -> torch.nn.Module:
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 10))

    inputs = {"camera_gain": 1.0,
              "evaluator_ed": make_evaluator(),
              "evaluator_ec": make_evaluator()}
    reference, _ = study.exposure._semantic_summary(
        inputs, cohort, {"direct_decoder": raw_cmos})
    consumed = study._candidate_semantics_from_readouts(
        inputs, cohort, {"direct_decoder": readouts})

    assert consumed["conditions"]["direct_decoder"]["evaluators"] == \
        reference["conditions"]["direct_decoder"]["evaluators"]
    assert consumed["same_class_diversity"] == reference["same_class_diversity"]


def test_summary_uses_rms_without_squaring_in_place() -> None:
    values = [0.0, 0.3]
    rows = [{"label": label, "row_passed": True,
             **{key: value for key in study.METRIC_KEYS}}
            for label in range(10) for value in values]
    summary = study._summarize_rows(rows)
    assert summary["metrics"][study.METRIC_KEYS[0]]["rms"] == pytest.approx(
        math.sqrt(0.045))
    assert values == [0.0, 0.3]


def test_full_optics_error_row_index_is_local_to_the_single_sample_batch() -> None:
    target = torch.ones((1, study.PORT_COUNT), dtype=torch.complex128)
    raw = torch.ones((1, 1, 2, 2), dtype=torch.float64)
    outputs = {grid: {"coefficients": target.clone(), "raw_cmos": raw.clone()}
               for grid in study.GRIDS}
    direct = {grid: raw.clone() for grid in study.GRIDS}

    with pytest.raises(IndexError):
        study.projected._error_row(outputs, target, direct, 9)
    metrics = study.projected._error_row(outputs, target, direct, 0)
    assert all(value == 0.0 for value in metrics.values())
    # Saved population indices are metadata; every consumer slices its row to a one-sample batch.
    assert all(value == 0.0 for value in study._metric_row(outputs, target, direct).values())


def test_measure_command_consumes_a_singleton_output_independent_of_source_index(monkeypatch) -> None:
    target = torch.ones((1, study.PORT_COUNT), dtype=torch.complex128)
    raw = torch.ones((1, 1, 2, 2), dtype=torch.float64)
    amplitude = torch.full((1, 1, 200, 200), 0.5, dtype=torch.float32)

    def forward_amplitude(_model, _command):
        return {"coefficients": target.clone(), "raw_cmos": raw.clone()}

    def forward_target(_model, _target):
        return {"raw_cmos": raw.clone()}

    monkeypatch.setattr(study.projected, "_forward_amplitude", forward_amplitude)
    monkeypatch.setattr(study.projected, "_forward_target", forward_target)
    models = {grid: torch.nn.Linear(1, 1) for grid in study.GRIDS}

    metrics, outputs = study._measure_command({}, amplitude, target, models)

    assert all(value == 0.0 for value in metrics.values())
    assert all(outputs[grid]["coefficients"].shape == (1, study.PORT_COUNT)
               for grid in study.GRIDS)
