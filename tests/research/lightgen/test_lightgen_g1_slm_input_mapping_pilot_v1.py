from __future__ import annotations

import math
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot


def test_engineering_worker_launches_project_module_from_repository_root() -> None:
    pilot_dir = Path(pilot.__file__).resolve().parent
    command, project_root = pilot._engineering_worker_command(
        pilot_dir / "unused.yaml", pilot_dir / "unused-preflight",
        pilot_dir / "unused-step3.pt", pilot_dir / "unused-step4.pt",
    )
    assert command[:4] == [
        sys.executable, "-B", "-m",
        "scripts.research.lightgen.execute_lightgen_g1_slm_input_mapping_pilot_v1",
    ]
    assert project_root == Path(pilot.__file__).resolve().parents[3]
    help_result = subprocess.run(
        [*command, "--help"], cwd=project_root, capture_output=True, text=True, check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "engineering-worker" in help_result.stdout
    assert "closeout-repair" in help_result.stdout


def test_mapper_has_the_fixed_one_forward_amplitude_interface() -> None:
    mapper = pilot.AmplitudeMapper()
    values = mapper(torch.zeros((2, pilot.REAL_DIM), dtype=torch.float32))
    assert tuple(values.shape) == (2, 1, 200, 200)
    assert values.dtype == torch.float32
    assert torch.isfinite(values).all()
    assert 0.0 <= float(values.min()) <= float(values.max()) <= 1.0
    assert [tuple(layer.weight.shape) for layer in mapper.network if isinstance(layer, torch.nn.Linear)] == [
        (256, 200), (40_000, 256),
    ]


def test_global_phase_objective_matches_independent_numpy_and_rejects_portwise_phase() -> None:
    target = np.asarray([0.4 + 0.3j, -0.2 + 0.7j, 0.11 - 0.23j], dtype=np.complex128)
    actual = np.asarray([0.39 + 0.28j, -0.18 + 0.71j, 0.10 - 0.21j], dtype=np.complex128)
    torch_loss = pilot._phase_invariant_loss(torch.from_numpy(actual[None]), torch.from_numpy(target[None]))
    numpy_error = pilot._numpy_phase_error(actual, target)[1]
    assert float(torch_loss[0]) == pytest.approx(numpy_error**2, abs=1.0e-12)

    common_phase = actual * np.exp(1j * 0.71)
    assert pilot._numpy_phase_error(common_phase, actual)[1] < 1.0e-14
    portwise_phase = actual.copy()
    portwise_phase[1] *= np.exp(0.4j)
    assert pilot._numpy_phase_error(portwise_phase, actual)[1] > 1.0e-3

    checks = pilot._numeric_math_checks()
    assert checks["executed"] and checks["passed"]
    assert checks["derivative_relative_error"] < 1.0e-5
    assert checks["wrong_port_phase_error"] > 1.0e-3


def test_g1_sampler_is_deterministic_and_rejects_only_by_total_modal_power() -> None:
    scale = torch.ones(100, dtype=torch.float64)
    covariance = torch.zeros((200, 200), dtype=torch.float64)
    small_factor = torch.eye(200, dtype=torch.float64) * 1.0e-7
    fit = {
        "mean": torch.zeros(200, dtype=torch.float64),
        "covariance": covariance,
        "cholesky": small_factor,
        "ridge": torch.tensor(1.0e-14, dtype=torch.float64),
    }
    first = pilot._sample_g1_class(fit, scale, label=3, seed=2026102613, count=8,
                                   proposal_limit=32, proposal_batch_size=8)
    second = pilot._sample_g1_class(fit, scale, label=3, seed=2026102613, count=8,
                                    proposal_limit=32, proposal_batch_size=8)
    assert first[0].dtype == torch.complex64
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.all(first[0].abs().square().sum(dim=1) <= pilot.base.MAX_MODAL_POWER_W)
    assert first[2]["selection_rule"] == "first accepted candidates in proposal order; total modal power only"

    over_power_mean = torch.zeros(200, dtype=torch.float64)
    over_power_mean[0] = 0.02
    rejected_fit = {**fit, "mean": over_power_mean}
    with pytest.raises(RuntimeError, match="proposal cap"):
        pilot._sample_g1_class(rejected_fit, scale, label=3, seed=1, count=1,
                               proposal_limit=8, proposal_batch_size=4)


def test_population_producer_schema_is_accepted_by_attempt_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "population": {
            "train_per_class": 1000,
            "validation_per_class": 200,
            "holdout_per_class": 200,
            "train_seed_base": 2026102610,
            "validation_seed_base": 2026102630,
            "holdout_seed_base": 2026102650,
            "max_proposals_per_class_per_population": 1000,
            "proposal_batch_size": 256,
            "schedule_seed": 2026102671,
        },
    }

    def fake_sampler(_fit, _scale, *, label, seed, count, proposal_limit, proposal_batch_size):
        value = (seed % 100 + 1) * 1.0e-7
        coefficients = torch.full(
            (count, pilot.PORT_COUNT), complex(value, value), dtype=torch.complex64,
        )
        proposal_ids = torch.arange(count, dtype=torch.long)
        return coefficients, proposal_ids, {
            "label": label, "seed": seed, "proposal_limit": proposal_limit,
            "proposal_batch_size": proposal_batch_size,
        }

    monkeypatch.setattr(pilot, "_sample_g1_class", fake_sampler)
    population = pilot._sample_population(
        config,
        {"prior": {label: {} for label in range(pilot.CLASS_COUNT)},
         "scale": torch.ones(pilot.PORT_COUNT, dtype=torch.float64)},
    )
    population["training_schedule"] = pilot._build_schedule(population, config)
    saved_payload = dict(population)

    monkeypatch.setattr(Path, "is_file", lambda path: path.name == "fixed_population_and_schedule.pt")
    monkeypatch.setattr(pilot.torch, "load", lambda *_args, **_kwargs: dict(saved_payload))
    loaded, schedule = pilot._load_attempt_population(Path("unused-attempt"))

    assert loaded["schema"] == pilot.POPULATION_SCHEMA
    assert loaded["task_id"] == pilot.TASK_ID
    assert tuple(schedule.shape) == (5, 1000, pilot.CLASS_COUNT)
    assert tuple(loaded["holdout"]["coefficients"].shape) == (2000, pilot.PORT_COUNT)


def test_schedule_has_one_row_per_class_in_each_balanced_batch_and_epoch() -> None:
    train = {
        "coefficients": torch.zeros((10_000, 100), dtype=torch.complex64),
        "labels": torch.arange(10).repeat_interleave(1000),
    }
    population = {"train": train}
    config = {"population": {"train_per_class": 1000, "schedule_seed": 2026102671}}
    schedule = pilot._build_schedule(population, config)
    assert tuple(schedule.shape) == (5, 1000, 10)
    for step in (0, 999, 1000, 2500, 4999):
        batch = pilot._balanced_batch(population, schedule, step)
        epoch, within_epoch = divmod(step, 1000)
        indices = schedule[epoch, within_epoch]
        labels = train["labels"][indices]
        assert torch.equal(labels, torch.arange(10))
        assert torch.equal(batch, train["coefficients"][indices])
    for epoch in range(5):
        for label in range(10):
            ids = schedule[epoch, :, label]
            assert torch.equal(ids.sort().values, torch.arange(label * 1000, (label + 1) * 1000))


def test_a8_amplitude_and_i8_intensity_use_distinct_rounding_domains() -> None:
    amplitude = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).reshape(1, 1, 1, 3)
    a8_codes, a8_intensity, i8_codes, i8_intensity = pilot._quantize_amplitude(amplitude)
    assert a8_codes.flatten().tolist() == [0, 128, 255]
    assert i8_codes.flatten().tolist() == [0, 64, 255]
    assert a8_intensity.flatten().tolist() == pytest.approx([0.0, (128 / 255) ** 2, 1.0])
    assert i8_intensity.flatten().tolist() == pytest.approx([0.0, 64 / 255, 1.0])
    assert not math.isclose(float(a8_intensity[0, 0, 0, 1]), float(i8_intensity[0, 0, 0, 1]))


def test_final_row_consumer_rejects_wrong_target_and_wrong_command_encoding() -> None:
    target = np.asarray([1.0 + 0.2j, -0.3 + 0.8j, 0.11 - 0.23j], dtype=np.complex128)
    native_cmos = np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float64)
    accepted, _ = pilot._single_record_passes(target, target, native_cmos, native_cmos)
    assert accepted

    wrong_target = target.copy()
    wrong_target[1] *= np.exp(0.4j)
    accepted_wrong_target, error = pilot._single_record_passes(target, wrong_target, native_cmos, native_cmos)
    assert not accepted_wrong_target
    assert error["coefficient_relative_error"] > pilot.MEASUREMENT_LIMIT

    wrong_encoded_output = target * 0.25
    accepted_wrong_encoding, error = pilot._single_record_passes(
        wrong_encoded_output, target, native_cmos * 0.25, native_cmos,
    )
    assert not accepted_wrong_encoding
    assert error["coefficient_relative_error"] > pilot.MEASUREMENT_LIMIT


def test_measurement_record_initializes_both_quantized_grid_accumulators() -> None:
    row = pilot._new_measurement_record(population_index=423, label=2, proposal_id=71)
    for condition in pilot.QUANTIZED_CONDITION_NAMES:
        for grid in pilot.GRIDS:
            row["quantized"][condition][str(grid)] = {"coefficient_relative_error": 0.0}
    assert row["sample_index_within_class"] == 23
    assert set(row["quantized"]) == {"A8_amplitude_linear", "I8_intensity_linear"}
    assert all(set(row["quantized"][condition]) == {"2048", "4096"}
               for condition in pilot.QUANTIZED_CONDITION_NAMES)


def test_primary_gate_comparison_ignores_corrected_auxiliary_metadata_only() -> None:
    original = {
        "threshold": 1.0e-4,
        "coefficient_alignment": "one global phase only",
        "raw_cmos": "native W/m^2; no normalization, power match, or gain adjustment",
        "cross_grid_metrics": ["coefficient", "CMOS"],
        "passed_rows": 0,
        "denominator": 2000,
        "pass_fraction": 0.0,
        "all_rows_passed": False,
        "maxima": {
            "coefficient_2048": 0.6165998805934169,
            "coefficient_4096": 0.6165976009860961,
            "native_cmos_2048": 0.8171431469068353,
            "native_cmos_4096": 0.8171416743622418,
            "cross_grid_coefficient": 4.797231225860619e-05,
            "cross_grid_native_cmos": 3.4887104699665726e-05,
        },
    }
    corrected = {
        **original,
        "raw_cmos": "native raw W/m^2; no normalization, power match, or gain adjustment",
        "auxiliary_camera_readout": {"pixel_area_m2": 6.25e-12, "camera_gain_unit": "W^-1"},
    }

    assert not pilot._metric_tree_matches(original, corrected)
    assert pilot._continuous_primary_gate_matches(original, corrected)

    changed_primary_error = copy.deepcopy(corrected)
    changed_primary_error["maxima"]["coefficient_2048"] += 2.0e-14
    assert not pilot._continuous_primary_gate_matches(original, changed_primary_error)


def test_closeout_report_reads_camera_area_from_primary_gate_metadata(tmp_path: Path) -> None:
    maxima = {
        "coefficient_2048": 0.1,
        "coefficient_4096": 0.1,
        "native_cmos_2048": 0.2,
        "native_cmos_4096": 0.2,
        "cross_grid_coefficient": 1.0e-5,
        "cross_grid_native_cmos": 2.0e-5,
    }
    decision = {
        "run_id": "fixed-run",
        "continuous_final_gate": {
            "maxima": maxima,
            "auxiliary_camera_readout": {"pixel_area_m2": 6.25e-12},
        },
        "slm_power_and_clipping": {
            "fixed_camera_gain_clipping_fraction_2048_continuous": 0.01140315,
        },
        "auxiliary_ED_EC_2048_continuous_only": {
            "E_D": {"correct": 1526}, "E_C": {"correct": 1590},
        },
        "quantized_terminal_measurements_not_gates": {
            condition: {"cross_grid": {
                "coefficient_relative_error_max": 3.0e-5,
                "native_cmos_relative_error_max": 4.0e-5,
            }}
            for condition in pilot.QUANTIZED_CONDITION_NAMES
        },
    }
    pilot._write_closeout_revision_report(
        tmp_path, decision, {}, {"passed": True},
        {"passed": True, "model_max_absolute_difference": 0.0,
         "optimizer_max_absolute_difference": 0.0, "loss_absolute_difference": 0.0},
    )

    report = (tmp_path / "closeout_report.md").read_text(encoding="utf-8")
    assert "W/m² × 6.25e-12 m²/像素" in report


def test_checkpoint_payload_deepcopies_adam_state(monkeypatch: pytest.MonkeyPatch) -> None:
    mapper = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    mapper(torch.ones((1, 2))).sum().backward()
    optimizer.step()
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    schedule = torch.zeros((5, 1000, 10), dtype=torch.long)
    payload = pilot._checkpoint_payload(
        mapper, optimizer, run_id="test-run", completed_updates=1,
        loss_history=[{"loss": 0.1}], schedule=schedule,
    )
    parameter_id = next(iter(payload["optimizer_state"]["state"]))
    saved_average = payload["optimizer_state"]["state"][parameter_id]["exp_avg"].clone()
    optimizer.state[next(iter(mapper.parameters()))]["exp_avg"].add_(1.0)
    assert torch.equal(payload["optimizer_state"]["state"][parameter_id]["exp_avg"], saved_average)
    assert payload["optimizer_state_deepcopied"] is True


def test_camera_density_is_integrated_before_frozen_gain() -> None:
    model = SimpleNamespace(runtime=SimpleNamespace(
        camera_plane=SimpleNamespace(pixel_pitch_m=(2.5e-6, 2.5e-6)),
    ))
    area = pilot._camera_pixel_area_m2(model)
    raw = torch.tensor([[[[4.0e10]]]], dtype=torch.float64)
    w_per_pixel = pilot._camera_w_per_pixel(raw, area)
    readout = pilot._readout_for_evaluator(raw, area, camera_gain_w_inverse=2.0)
    independent = np.asarray([[[[4.0e10]]]], dtype=np.float64) * area

    assert area == pytest.approx(6.25e-12, rel=1.0e-12)
    assert w_per_pixel.item() == pytest.approx(0.25, abs=1.0e-15)
    assert independent.item() == pytest.approx(0.25, abs=1.0e-15)
    assert readout[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1.0e-7)
    old_wrong_unit_value = min(1.0, raw.item() * 2.0)
    assert old_wrong_unit_value == 1.0
    assert readout[0, 0, 0, 0].item() != pytest.approx(old_wrong_unit_value)


def test_recovery_comparison_rejects_corrupted_serialized_adam_state() -> None:
    schedule = torch.zeros((5, 1000, 10), dtype=torch.long)
    history = [{"completed_update": index, "loss": 0.1} for index in range(1, 5)]
    live_state = {
        "mapper": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.25])}}, "param_groups": []},
    }
    worker = {
        "task_id": pilot.TASK_ID,
        "run_id": "fixed-run",
        "completed_updates": 4,
        "checkpoint_step": 4,
        "next_schedule_cursor": 4,
        "engineering_adam_calls": 1,
        "step4_schedule_indices": schedule[0, 3].clone(),
        "mapper_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {"state": {0: {"exp_avg": torch.tensor([0.25])}}, "param_groups": []},
        "step4_loss": 0.1,
        "loss_history": copy.deepcopy(history),
        "torch_cpu_rng": torch.tensor([1, 2], dtype=torch.uint8),
        "torch_cuda_rng": [torch.tensor([3, 4], dtype=torch.uint8)],
    }
    live_rng = {"cpu": worker["torch_cpu_rng"], "cuda": worker["torch_cuda_rng"]}
    checked = pilot._assert_engineering_recovery_pair(
        live_state, 0.1, live_rng, history, worker, run_id="fixed-run", schedule=schedule,
    )
    assert checked["optimizer_max_absolute_difference"] == 0.0
    assert checked["rng_equal"] and checked["schedule_equal"]

    corrupted = copy.deepcopy(worker)
    corrupted["optimizer_state"]["state"][0]["exp_avg"].add_(0.01)
    old_restored_a = copy.deepcopy(corrupted["optimizer_state"])
    old_restored_b = copy.deepcopy(corrupted["optimizer_state"])
    old_two_reloaded_check, _ = pilot._allclose_tree(old_restored_a, old_restored_b, atol=0.0)
    assert old_two_reloaded_check
    with pytest.raises(RuntimeError, match="differs from the independently restored"):
        pilot._assert_engineering_recovery_pair(
            live_state, 0.1, live_rng, history, corrupted, run_id="fixed-run", schedule=schedule,
        )


def test_saved_recovery_evidence_consumer_reloads_live_and_worker_checkpoints(tmp_path: Path) -> None:
    engineering = tmp_path / "engineering_recovery"
    checkpoints = engineering / "checkpoints"
    checkpoints.mkdir(parents=True)
    schedule = torch.zeros((5, 1000, 10), dtype=torch.long)
    history = [{"completed_update": index, "loss": 0.1 / index} for index in range(1, 4)]
    rng = {"cpu": torch.tensor([1, 2], dtype=torch.uint8),
           "cuda": [torch.tensor([3, 4], dtype=torch.uint8)]}

    def optimizer_state(step: int, average: float) -> dict[str, object]:
        return {"state": {index: {"step": torch.tensor(float(step)), "exp_avg": torch.tensor([average])}
                           for index in range(4)}, "param_groups": []}

    step3 = {
        "task_id": pilot.TASK_ID, "run_id": "fixed-run", "completed_updates": 3,
        "checkpoint_step": 3, "next_schedule_cursor": 3,
        "next_batch_indices": schedule[0, 3].clone(),
        "optimizer_state": optimizer_state(3, 0.1), "optimizer_state_deepcopied": True,
    }
    live = {
        "task_id": pilot.TASK_ID, "run_id": "fixed-run", "completed_updates": 4,
        "checkpoint_step": 4, "next_schedule_cursor": 4,
        "next_batch_indices": schedule[0, 4].clone(), "last_loss": 0.025,
        "loss_history_count": 4, "mapper_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": optimizer_state(4, 0.2), "optimizer_state_deepcopied": True,
        "torch_cpu_rng": rng["cpu"], "torch_cuda_rng": rng["cuda"],
    }
    resumed = {
        **live,
        "mapper_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": optimizer_state(4, 0.2),
        "step4_loss": 0.025,
        "step4_schedule_indices": schedule[0, 3].clone(),
        "loss_history": [*history, {"completed_update": 4, "loss": 0.025}],
    }
    receipt = {
        "executed": True, "passed": True, "engineering_adam_calls": 5,
        "formal_restore_function": "_restore_training_state", "save_preserved_rng": True,
        "rng_equal": True, "step3_history_count": 3, "step4_history_count": 4,
    }
    torch.save(step3, checkpoints / "checkpoint_step_000003.pt")
    torch.save(live, checkpoints / "checkpoint_step_000004_live.pt")
    torch.save(resumed, engineering / "worker_step4_restored.pt")
    (engineering / "training_history.json").write_text(
        json.dumps({"updates": history}), encoding="utf-8",
    )
    (engineering / "engineering_recovery.json").write_text(json.dumps(receipt), encoding="utf-8")

    checked = pilot._verify_saved_engineering_recovery(
        engineering, run_id="fixed-run", schedule=schedule,
    )
    assert checked["saved_recovery_evidence_reconsumed"]
    assert checked["additional_engineering_adam_calls_during_reconsumption"] == 0
    assert checked["rng_equal_recomputed"] and checked["schedule_equal_recomputed"]

    resumed["optimizer_state"]["state"][0]["exp_avg"].add_(0.01)
    torch.save(resumed, engineering / "worker_step4_restored.pt")
    with pytest.raises(RuntimeError, match="live-versus-restored recovery"):
        pilot._verify_saved_engineering_recovery(
            engineering, run_id="fixed-run", schedule=schedule,
        )


def test_quantized_cross_grid_statistics_use_each_condition_outputs() -> None:
    records = []
    for label in range(10):
        row = pilot._new_measurement_record(population_index=label * 200, label=label, proposal_id=label)
        for grid in pilot.GRIDS:
            row["continuous"][str(grid)] = {
                "coefficient_relative_error": 0.1,
                "native_cmos_relative_error": 0.2,
            }
            for condition in pilot.QUANTIZED_CONDITION_NAMES:
                row["quantized"][condition][str(grid)] = {
                    "coefficient_relative_error": 0.3,
                    "native_cmos_relative_error": 0.4,
                }
        row["cross_grid"] = {
            "coefficient_relative_error_one_global_phase": 0.001,
            "native_cmos_relative_error": 0.002,
            "direct_reference_native_cmos_relative_error": 0.003,
        }
        row["quantized_cross_grid"]["A8_amplitude_linear"] = {
            "coefficient_relative_error_one_global_phase": 0.011,
            "native_cmos_relative_error": 0.012,
            "direct_reference_native_cmos_relative_error": 0.003,
        }
        row["quantized_cross_grid"]["I8_intensity_linear"] = {
            "coefficient_relative_error_one_global_phase": 0.021,
            "native_cmos_relative_error": 0.022,
            "direct_reference_native_cmos_relative_error": 0.003,
        }
        row["continuous_pass"] = False
        records.append(row)

    continuous = pilot._summarize_condition(records, "continuous")
    a8 = pilot._summarize_condition(records, "A8_amplitude_linear")
    i8 = pilot._summarize_condition(records, "I8_intensity_linear")
    assert continuous["cross_grid"]["coefficient_relative_error_max"] == pytest.approx(0.001)
    assert a8["cross_grid"]["coefficient_relative_error_max"] == pytest.approx(0.011)
    assert a8["cross_grid"]["native_cmos_relative_error_max"] == pytest.approx(0.012)
    assert i8["cross_grid"]["coefficient_relative_error_max"] == pytest.approx(0.021)
    assert i8["cross_grid"]["native_cmos_relative_error_max"] == pytest.approx(0.022)


def test_transfer_receipt_checks_actual_file_size_after_copy(tmp_path: Path) -> None:
    evidence = tmp_path / "receipt"
    evidence.mkdir()
    payload = evidence / "evidence.bin"
    payload.write_bytes(b"1234567890")
    receipt = {
        "status": "COMPLETE",
        "destination": str(evidence),
        "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/closeout_repair_01",
        "files": [{
            "path": "evidence.bin", "remote_size_bytes": 10, "local_size_bytes": 10,
            "status": "COPIED",
        }],
    }
    (evidence / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    verified = pilot._verify_transfer_receipt(
        evidence,
        remote_run_suffix="/outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/closeout_repair_01",
        required_files={"evidence.bin"},
    )
    assert verified["files_verified_against_actual_size"] == 1

    payload.write_bytes(b"short")
    stale_receipt = receipt["files"][0]
    assert stale_receipt["status"] == "COPIED"
    assert stale_receipt["remote_size_bytes"] == stale_receipt["local_size_bytes"]
    assert payload.is_file()
    with pytest.raises(RuntimeError, match="actual file"):
        pilot._verify_transfer_receipt(
            evidence,
            remote_run_suffix="/outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/closeout_repair_01",
            required_files={"evidence.bin"},
        )


def test_fixed_display_arrays_and_contact_sheet_preserve_all_four_output_columns(tmp_path: Path) -> None:
    records = []
    for label in range(10):
        for sample in range(2):
            records.append({
                "label": label,
                "sample_index_within_class": sample,
                "proposal_id": label * 2 + sample,
                "raw_cmos": np.full((200, 200), 1.0e9 + label + sample, dtype=np.float64),
                "direct_raw_cmos": np.full((200, 200), 2.0e9 + label + sample, dtype=np.float64),
            })
    area, gain = 6.25e-12, 50.0
    display = pilot._display_arrays_from_records(records, area, gain)
    pilot._save_first_two_contact_sheet(tmp_path, records, area, gain)
    sheet = np.asarray(Image.open(tmp_path / "figures" / "continuous_2048_first_two_per_class.png").convert("RGB"))

    assert sheet.shape == (2268, 1048, 3)
    assert np.array_equal(display["mapped_w_per_pixel"], display["mapped_raw_cmos_w_per_m2"] * area)
    for label in range(10):
        row_y = 8 + label * 226 + 18
        for sample in range(2):
            sample_index = label * 2 + sample
            mapped_x = 8 + (sample * 2 + 1) * 208
            direct_x = mapped_x + 208
            assert np.array_equal(
                sheet[row_y:row_y + 200, mapped_x:mapped_x + 200],
                np.repeat(display["mapped_display_u8"][sample_index, ..., None], 3, axis=2),
            )
            assert np.array_equal(
                sheet[row_y:row_y + 200, direct_x:direct_x + 200],
                np.repeat(display["direct_display_u8"][sample_index, ..., None], 3, axis=2),
            )
