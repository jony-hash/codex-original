from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from scripts.research.lightgen import measure_lightgen_train_generation_ols_support_v1 as support


def test_torch_projective_distance_matches_independent_numpy_and_phase_scale_invariance() -> None:
    rng = np.random.default_rng(913)
    reference = rng.normal(size=(7, 100)) + 1j * rng.normal(size=(7, 100))
    query = rng.normal(size=(5, 100)) + 1j * rng.normal(size=(5, 100))
    scale = np.linspace(0.2, 2.4, 100)
    rows = np.arange(7, dtype=np.int64) + 900

    production = support._nearest_torch(
        torch.from_numpy(query), torch.from_numpy(reference), torch.from_numpy(scale),
        torch.from_numpy(rows),
    )
    independent = support._nearest_numpy(query, reference, scale, rows)
    for key in ("distance", "amplitude_distance", "relative_phase_penalty"):
        assert np.allclose(production[key].numpy(), independent[key], rtol=1e-12, atol=1e-12)
    assert np.array_equal(production["nearest_reference_row"].numpy(),
                          independent["nearest_reference_row"])

    transformed = query * (3.7 * np.exp(0.91j))
    invariant = support._nearest_numpy(transformed, reference, scale, rows)
    assert np.allclose(invariant["distance"], independent["distance"], rtol=1e-12, atol=1e-12)


def test_equal_modal_amplitudes_with_changed_relative_phase_are_detected() -> None:
    reference = np.ones((1, 100), dtype=np.complex128)
    query = np.ones((1, 100), dtype=np.complex128)
    query[:, 50:] *= -1.0
    result = support._nearest_numpy(query, reference, np.ones(100), np.array([42]))

    assert result["amplitude_distance"][0] == pytest.approx(0.0, abs=1e-12)
    assert result["distance"][0] > 1.0
    assert result["relative_phase_penalty"][0] > 1.0


def test_raw_cmos_summary_accepts_gpu_intensity_and_cpu_coefficients() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raw = torch.empty((2, 1, 200, 200), dtype=torch.float32, device=device)
    raw[0].fill_(1.0)
    raw[1].fill_(2.0)
    sensor_w_pixel = raw * 0.5
    coefficients = torch.ones((2, 100), dtype=torch.complex64, device="cpu")

    summary = support._raw_output_summary(raw, sensor_w_pixel, coefficients)

    assert summary.device.type == "cpu"
    assert torch.allclose(summary[:, 0], torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(summary[:, 1], torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(summary[:, 2], torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(summary[:, 3], torch.tensor([20_000.0, 40_000.0], dtype=torch.float64))
    assert torch.allclose(summary[:, 4], torch.tensor([100.0, 100.0], dtype=torch.float64))


def test_generation_raw_summaries_return_numpy_arrays_for_numpy_analysis() -> None:
    raw0 = torch.ones((2, 1, 2, 2), dtype=torch.float64)
    raw1 = torch.full((2, 1, 2, 2), 3.0, dtype=torch.float64)
    generation = {"bundle": {"conditions": {
        "E0G0": {"raw_cmos": raw0},
        "E1G1": {"raw_cmos": raw1},
    }}}

    summaries = support._generation_raw_summaries(generation)

    assert isinstance(summaries[0]["raw_cmos_summary"], np.ndarray)
    assert np.allclose(summaries[0]["raw_cmos_summary"], [[1.0, 1.0, 1.0]] * 2)
    assert np.allclose(summaries[2_000]["raw_cmos_summary"], [[3.0, 3.0, 3.0]] * 2)
    assert np.allclose(summaries[0]["terminal_raw_cmos_delta_rms"], [2.0, 2.0])


def test_sample_selection_uses_first_distinct_schedule_rows_and_disjoint_roles() -> None:
    schedule = torch.empty((220, 10), dtype=torch.long)
    for step in range(220):
        for label in range(10):
            schedule[step, label] = label * 200 + step % 200
    labels = torch.arange(10).repeat_interleave(200)
    train = torch.arange(2_000)

    records = support._sample_manifest(schedule, labels, train)

    assert len(records) == 2_000
    for label in range(10):
        selected = [row for row in records if row["label"] == label]
        assert [row["source_train_row"] for row in selected] == list(
            range(label * 200, (label + 1) * 200)
        )
        assert [row["schedule_step"] for row in selected] == list(range(200))
        assert sum(row["role"] == "reference" for row in selected) == 100
        assert sum(row["role"] == "calibration" for row in selected) == 100
        assert {row["source_train_row"] for row in selected if row["role"] == "reference"}.isdisjoint(
            row["source_train_row"] for row in selected if row["role"] == "calibration"
        )


def test_sample_selection_rejects_schedule_label_mismatch() -> None:
    schedule = torch.arange(2_000, dtype=torch.long).reshape(200, 10)
    labels = torch.arange(10).repeat_interleave(200)
    labels[1] = 9

    with pytest.raises(RuntimeError, match="wrong class"):
        support._sample_manifest(schedule, labels, torch.arange(2_000))


def test_saved_manifest_consumer_rejects_missing_or_reordered_rows(tmp_path) -> None:
    records = [
        {"population_index": 0, "label": 0, "class_position": 0,
         "source_train_row": 18, "schedule_step": 4,
         "schedule_flat_position": 40, "role": "reference", "positive_control": True},
        {"population_index": 1, "label": 0, "class_position": 1,
         "source_train_row": 21, "schedule_step": 8,
         "schedule_flat_position": 80, "role": "calibration", "positive_control": True},
    ]
    path = tmp_path / "manifest.csv"
    support._write_csv(path, records, list(records[0]))
    support._validate_sample_manifest(path, records)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.reverse()
    support._write_csv(path, rows, list(records[0]))
    with pytest.raises(RuntimeError, match="missing, changed, or reordered"):
        support._validate_sample_manifest(path, records)


def test_command_identity_rejects_wrong_or_incomplete_slm_commands() -> None:
    saved = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    identical = saved.clone()
    support._validate_generation_commands(
        saved, identical, expected_rows=2, command_shape=(1, 2, 2)
    )

    wrong = identical.clone()
    wrong[1, 0, 1, 1] = 0.5
    with pytest.raises(RuntimeError, match="command rows differ"):
        support._validate_generation_commands(
            saved, wrong, expected_rows=2, command_shape=(1, 2, 2)
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        support._validate_generation_commands(
            saved[:1], identical[:1], expected_rows=2, command_shape=(1, 2, 2)
        )


def test_phase_provenance_consumer_rejects_wrong_source_tensor() -> None:
    model0 = {"runtime.encoder.phase": torch.zeros(1),
              "runtime.generator.phase": torch.zeros(1)}
    model2k = {"runtime.encoder.phase": torch.ones(1),
               "runtime.generator.phase": torch.full((1,), 2.0)}
    checkpoints = {0: {"model": model0}, 2_000: {"model": model2k}}
    provenance = {"encoder_step": 0, "generator_step": 2_000,
                  "source_training_run_id": support.TRAINING_RUN_ID}
    states = {"encoder": model0["runtime.encoder.phase"].clone(),
              "generator": model2k["runtime.generator.phase"].clone()}
    support._validate_condition_phase("E0G1", provenance, states, checkpoints)

    states["encoder"].fill_(1.0)
    with pytest.raises(RuntimeError, match="phase tensors do not match"):
        support._validate_condition_phase("E0G1", provenance, states, checkpoints)


@pytest.mark.parametrize(
    ("terminal", "delta", "expected"),
    [
        ((0.01, 0.03), (0.002, 0.01), "DIFFERENCE_EXPANDED"),
        ((0.01, 0.03), (-0.01, 0.001), "DIFFERENCE_PRESENT_NOT_CONFIRMED_EXPANDED"),
        ((-0.01, 0.01), (0.002, 0.01), "EVIDENCE_INSUFFICIENT"),
    ],
)
def test_decision_route_uses_both_predeclared_interval_signs(terminal, delta, expected) -> None:
    route = support._route_decision({
        "G_2000": {"lower": terminal[0], "upper": terminal[1]},
        "DeltaG": {"lower": delta[0], "upper": delta[1]},
    })

    assert route == expected


def test_saved_adam_steps_are_read_without_constructing_optimizer() -> None:
    state = {
        "param_groups": [{"params": [0, 1]}],
        "state": {
            0: {"step": torch.tensor(45_000), "exp_avg": torch.zeros(2, 1, 100, 100),
                "exp_avg_sq": torch.zeros(2, 1, 100, 100)},
            1: {"step": torch.tensor(50_000), "exp_avg": torch.zeros(2, 1, 100, 100),
                "exp_avg_sq": torch.zeros(2, 1, 100, 100)},
        },
    }

    assert support._optimizer_steps_without_instantiation(state) == (45_000, 50_000)
    del state["state"][1]["exp_avg_sq"]
    with pytest.raises(RuntimeError, match="moment state is incomplete"):
        support._optimizer_steps_without_instantiation(state)


def test_attempt02_preflight_source_accepts_only_the_known_cache_mismatch_failure() -> None:
    source_status = {
        "task_id": support.TASK_ID, "status": "INVALID_HARNESS",
        "stage": "representative_preflight",
        "error": "step-0 cache preflight failed: diagnostic mismatch",
        "production_forwards_attempted": 40, "production_forwards_saved": 40,
        "cross_grid_reference_forwards_attempted": 40,
        "optimizer_updates": 0, "official_test_reads": 0,
    }
    support._validate_preflight_source_status(source_status)

    wrong_failure = dict(source_status, error="cross-grid preflight failed")
    with pytest.raises(RuntimeError, match="cache-mismatch-only"):
        support._validate_preflight_source_status(wrong_failure)


def test_reused_cross_grid_report_is_rejected_when_any_saved_row_changes() -> None:
    rows = [{
        "endpoint_step": 0 if index < 20 else 2_000,
        "population_index": index,
        "source_train_row": index + 100,
        "label": index % 10,
        "coefficient_relative_error": 1.0e-5,
        "raw_cmos_relative_error": 2.0e-5,
        "passed_1e-4": True,
    } for index in range(40)]
    report = {
        "task_id": support.TASK_ID, "status": "PASSED",
        "production_rows_included": 40, "independent_reference_rows": 40,
        "rows_per_class_per_endpoint": 2, "production_grid": 2048,
        "reference_grid": 4096, "tolerance": 1.0e-4,
        "semantic_evidence": False, "hardware_evidence": False,
        "rows": rows,
        "maximum_coefficient_relative_error": 1.0e-5,
        "maximum_raw_cmos_relative_error": 2.0e-5,
    }
    support._validate_reused_cross_grid_report(report, rows)

    changed = [dict(row) for row in rows]
    changed[17]["population_index"] = 19
    with pytest.raises(RuntimeError, match="do not independently reproduce"):
        support._validate_reused_cross_grid_report(report, changed)


def test_old_cache_mismatch_is_retained_as_nonblocking_diagnostic() -> None:
    context = {
        "cache": {
            "rows": torch.tensor([14, 27]),
            "coefficients": torch.tensor([[1.0 + 0.0j], [1.0 + 0.0j]], dtype=torch.complex128),
        },
        "cache_replay_max_relative_error": 7.7e-7,
    }
    manifest = [{"source_train_row": 14}, {"source_train_row": 27}]
    train_outputs = {0: {"coefficients": torch.tensor(
        [[2.0 + 0.0j], [2.0 + 0.0j]], dtype=torch.complex128
    )}}

    summary = support._cache_crosscheck(context, train_outputs, manifest)

    assert summary["status"] == "DIAGNOSTIC_MISMATCH"
    assert summary["passed"] is False
    assert summary["blocking_gate"] is False
    assert summary["tolerance"] == support.CACHE_TOLERANCE
    assert summary["maximum_relative_error"] == pytest.approx(0.5)

    errors = torch.tensor([0.5, 0.25], dtype=torch.float64)
    record = {
        "task_id": support.TASK_ID, "status": "DIAGNOSTIC_MISMATCH",
        "source_attempt": "attempt_02_saved_controls_reused", "rows_compared": 2,
        "tolerance": support.CACHE_TOLERANCE, "passed": False,
        "blocking_gate": False, "mismatch_policy": "report_as_diagnostic_limitation",
        "user_approved_nonblocking_treatment": True,
        "step_2000_cache_used_as_reference": False,
        "maximum_relative_error": 0.5,
        "p95_relative_error": float(torch.quantile(errors, 0.95)),
    }
    assert support._validate_cache_diagnostic_record(
        record, errors, rows_compared=2,
        source_attempt="attempt_02_saved_controls_reused",
    ) is False

    blocked = dict(record, blocking_gate=True)
    with pytest.raises(RuntimeError, match="not the measured diagnostic record"):
        support._validate_cache_diagnostic_record(
            blocked, errors, rows_compared=2,
            source_attempt="attempt_02_saved_controls_reused",
        )


def test_forward_input_binds_source_row_before_native_slm_and_optical_forward(monkeypatch) -> None:
    pixels = torch.stack([torch.full((28, 28), index, dtype=torch.uint8) for index in range(6)])
    runtime = {
        "pixels": pixels, "labels": torch.tensor([9, 0, 1, 2, 3, 4]),
        "train_indices": torch.tensor([1, 3, 4, 5]), "device": torch.device("cpu"),
    }
    consumed = []

    def fake_forward(model, commands):
        consumed.append(commands.detach().cpu().clone())
        count = len(commands)
        raw = torch.ones((count, 1, 200, 200), dtype=torch.float32)
        return {
            "ols_coefficients": torch.ones((count, 100), dtype=torch.complex64),
            "plane_powers_w": torch.ones((count, 5)),
            "raw_intensity_w_per_m2": raw,
            "physical_w_per_pixel": raw,
        }

    monkeypatch.setattr(support.base, "_forward", fake_forward)
    measured = support._measure_rows(
        torch.nn.Linear(1, 1), torch.tensor([1, 4]), runtime,
        expected_source_rows=torch.tensor([1, 4]), expected_labels=torch.tensor([0, 3]),
        keep_raw=True,
    )

    assert torch.equal(measured["source_rows"], torch.tensor([1, 4]))
    assert torch.equal(measured["labels"], torch.tensor([0, 3]))
    assert torch.equal(measured["source_pixels"], pixels[[1, 4]])
    assert torch.equal(measured["slm_commands"], consumed[0])
    assert torch.equal(consumed[0], support.image_batch(pixels, torch.tensor([1, 4]), "small",
                                                        torch.device("cpu")))


def test_forward_input_rejects_population_position_passed_as_source_image_row() -> None:
    runtime = {
        "pixels": torch.zeros((2, 28, 28), dtype=torch.uint8),
        "labels": torch.tensor([0, 0]), "train_indices": torch.tensor([0, 1]),
        "device": torch.device("cpu"),
    }

    with pytest.raises(RuntimeError, match="differ from the manifest source_train_row"):
        support._forward_input_identity(
            torch.tensor([0]), runtime,
            expected_source_rows=torch.tensor([30340]), expected_labels=torch.tensor([0]),
        )


def test_forward_input_rejects_same_class_wrong_image_dev_row_and_wrong_label() -> None:
    runtime = {
        "pixels": torch.zeros((3, 28, 28), dtype=torch.uint8),
        "labels": torch.tensor([0, 0, 0]), "train_indices": torch.tensor([0, 1]),
        "device": torch.device("cpu"),
    }
    with pytest.raises(RuntimeError, match="differ from the manifest source_train_row"):
        support._forward_input_identity(
            torch.tensor([0]), runtime,
            expected_source_rows=torch.tensor([1]), expected_labels=torch.tensor([0]),
        )
    with pytest.raises(RuntimeError, match="outside TRAIN"):
        support._forward_input_identity(
            torch.tensor([2]), runtime,
            expected_source_rows=torch.tensor([2]), expected_labels=torch.tensor([0]),
        )
    with pytest.raises(RuntimeError, match="labels differ"):
        support._forward_input_identity(
            torch.tensor([1]), runtime,
            expected_source_rows=torch.tensor([1]), expected_labels=torch.tensor([1]),
        )


def _synthetic_control_evidence():
    pixels = torch.stack([torch.full((28, 28), index, dtype=torch.uint8) for index in range(21)])
    labels = torch.tensor([label for label in range(10) for _ in range(2)] + [0])
    runtime = {
        "pixels": pixels, "labels": labels, "train_indices": torch.arange(20),
        "device": torch.device("cpu"),
    }
    control_source_rows = {
        label * 200 + offset: label * 2 + offset
        for label in range(10) for offset in (0, 1)
    }
    manifest = [{
        "population_index": index,
        "source_train_row": control_source_rows.get(index, 21 + index),
        "label": index // 200,
        "positive_control": index in control_source_rows,
    } for index in range(2_000)]
    rows = torch.tensor(list(control_source_rows.values()))
    population_indices = torch.tensor(list(control_source_rows))
    commands = support.image_batch(pixels, rows, "small", torch.device("cpu"))
    controls = {}
    for step in support.PHASE_STEPS:
        item = {"population_indices": population_indices.clone()}
        for grid in ("2048", "4096"):
            item[f"source_rows_{grid}"] = rows.clone()
            item[f"labels_{grid}"] = labels.index_select(0, rows).clone()
            item[f"source_pixels_{grid}"] = pixels.index_select(0, rows).clone()
            item[f"slm_commands_{grid}"] = commands.clone()
            item[f"coefficients_{grid}"] = torch.ones((20, 100), dtype=torch.complex128)
            item[f"plane_powers_w_{grid}"] = torch.ones((20, 5))
            item[f"raw_summaries_{grid}"] = torch.ones((20, 5))
            item[f"raw_cmos_{grid}"] = torch.ones((20, 1, 200, 200))
        controls[step] = item
    return controls, manifest, runtime


def test_actual_control_input_consumer_rejects_wrong_same_class_source_row() -> None:
    controls, manifest, runtime = _synthetic_control_evidence()
    controls[0]["source_rows_2048"][0] = 1
    controls[0]["source_pixels_2048"][0].copy_(runtime["pixels"][1])

    with pytest.raises(RuntimeError, match="wrong source row or label"):
        support._validate_control_input_evidence(controls, manifest, runtime)


def test_actual_control_input_consumer_rejects_dev_row_wrong_label_and_missing_row() -> None:
    controls, manifest, runtime = _synthetic_control_evidence()
    dev_manifest = [dict(row) for row in manifest]
    dev_manifest[0]["source_train_row"] = 20
    with pytest.raises(RuntimeError, match="outside TRAIN"):
        support._validate_control_input_evidence(controls, dev_manifest, runtime)

    controls, manifest, runtime = _synthetic_control_evidence()
    controls[2_000]["labels_4096"][0] = 9
    with pytest.raises(RuntimeError, match="wrong source row or label"):
        support._validate_control_input_evidence(controls, manifest, runtime)

    controls, manifest, runtime = _synthetic_control_evidence()
    controls[0]["population_indices"][0] = controls[0]["population_indices"][1]
    with pytest.raises(RuntimeError, match="missing, duplicate, or reordered"):
        support._validate_control_input_evidence(controls, manifest, runtime)
