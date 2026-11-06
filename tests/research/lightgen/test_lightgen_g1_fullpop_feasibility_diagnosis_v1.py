from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_g1_fullpop_feasibility_diagnosis_v1 as diagnosis


def test_fixed_and_free_phase_controls_cover_box_and_shared_phase_cases() -> None:
    result = diagnosis._analytic_controls()

    assert result["known_box_command_error"] <= diagnosis.SOLVER_STOP
    assert result["fixed_phase_rotated_target_error"] > 1.0e-4
    assert result["free_phase_rotated_target_error"] <= diagnosis.SOLVER_STOP
    assert result["out_of_box_target_error"] > 1.0e-4
    assert result["complex128_directional_derivative_abs_error"] <= 1.0e-5
    assert result["complex128_boundary_one_sided_derivative_abs_error"] <= 1.0e-5
    assert result["descent_direction_decreases_objective"] is True
    assert result["all_commands_bounded"] is True


def test_cloud_positive_control_reads_cloud_witness_path(tmp_path, monkeypatch) -> None:
    cloud_path = tmp_path / "cloud_witness.pt"
    windows_path = tmp_path / "windows_witness.pt"
    targets = torch.arange(20 * 100, dtype=torch.float64).reshape(20, 100).to(torch.complex128)
    labels = torch.arange(10).repeat_interleave(2)
    indices = torch.arange(20)
    amplitudes = torch.full((20, 1, 200, 200), 0.25, dtype=torch.float32)
    for path, command in ((cloud_path, amplitudes),
                          (windows_path, torch.full_like(amplitudes, 0.75))):
        torch.save({"population_role": "known_reachable_fit20",
                    "target_coefficients": targets,
                    "source_validation_indices": indices,
                    "amplitudes": command}, path)
    captured: dict[str, torch.Tensor] = {}

    def measure(_targets, commands, _inputs, _device):
        captured["amplitudes"] = commands["known_reachable_witness"].clone()
        rows = [{"target_index": index, "row_passed": True} for index in range(10)]
        return {}, {}, rows

    monkeypatch.setattr(diagnosis, "_measure_commands", measure)
    source = {
        "host": "cloud",
        "fullpop_config": {"source": {
            "fit20_cloud_witness_measurements": str(cloud_path),
            "fit20_witness_measurements": str(windows_path),
        }},
        "population": {"known_reachable": {
            "coefficients": targets,
            "labels": labels,
            "source_validation_indices": indices,
        }},
        "inputs": {},
    }

    result = diagnosis._positive_controls(source, torch.device("cpu"))

    assert result["status"] == "TEN_POSITIVE_CONTROLS_PASSED"
    assert result["count"] == result["passed"] == 10
    assert torch.equal(captured["amplitudes"], amplitudes[::2])


def test_zero_update_baseline_preserves_production_amplitude_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(diagnosis, "TARGET_COUNT", 1)
    command = torch.full((1, 1, 200, 200), 0.4, dtype=torch.float32)
    preclip = torch.full((40_000,), 0.4, dtype=torch.float64)
    measured_shapes: dict[str, tuple[int, ...]] = {}

    def mapper_command(*_args):
        return command.clone(), 0.0, preclip.clone()

    def measure(targets, commands, _inputs, _device, **_kwargs):
        measured_shapes["target"] = tuple(targets.shape)
        measured_shapes.update({key: tuple(value.shape) for key, value in commands.items()})
        coefficients = torch.zeros((1, 100), dtype=torch.complex128)
        outputs = {"box_b5000": {2048: {"coefficients": coefficients}}}
        return outputs, {}, []

    monkeypatch.setattr(diagnosis, "_one_mapper_command", mapper_command)
    monkeypatch.setattr(diagnosis, "_measure_commands", measure)
    monkeypatch.setattr(diagnosis, "_residual_decomposition",
                        lambda *_args: {"identity_vector_max_abs_error": 0.0})
    source = {"inputs": {"response_matrix": torch.ones((1, 1), dtype=torch.complex128)},
              "source_states": {"frozen_v2_baseline": {}, "box_b_candidate": {}},
              "identity": SimpleNamespace(run_id="source-run")}
    targets = {"coefficients": torch.ones((1, 100), dtype=torch.complex128),
               "source_validation_indices": torch.tensor([2]),
               "labels": torch.tensor([0]), "proposal_ids": torch.tensor([17])}

    result = diagnosis._baseline_measurement(source, targets, tmp_path, torch.device("cpu"))

    assert result["status"] == "ZERO_UPDATE_BASELINE_COMPLETE"
    assert measured_shapes == {"target": (1, 100),
                               "frozen_v2": (1, 1, 200, 200),
                               "box_b5000": (1, 1, 200, 200)}


def test_measure_commands_preserves_global_target_index_offset(monkeypatch) -> None:
    monkeypatch.setattr(diagnosis.pilot, "_new_optical_model",
                        lambda _checkpoint, grid, _device: grid)
    monkeypatch.setattr(diagnosis.projected, "_forward_target",
                        lambda _model, _target: {"raw_cmos": torch.zeros((1, 1, 2, 2))})
    monkeypatch.setattr(diagnosis.projected, "_forward_amplitude",
                        lambda _model, _amplitude: {
                            "coefficients": torch.zeros((1, 100), dtype=torch.complex128),
                            "raw_cmos": torch.zeros((1, 1, 2, 2)),
                        })
    monkeypatch.setattr(diagnosis.projected, "_error_row",
                        lambda *_args: {key: 0.0 for key in diagnosis.METRIC_KEYS})
    targets = torch.zeros((2, 100), dtype=torch.complex128)
    commands = {"example": torch.zeros((2, 1, 200, 200), dtype=torch.float32)}

    _, _, rows = diagnosis._measure_commands(
        targets, commands, {"checkpoint": object()}, torch.device("cpu"), target_index_offset=7)

    assert [row["target_index"] for row in rows] == [7, 8]


def test_measurement_row_identity_repair_is_idempotent_and_preserves_raw_arrays() -> None:
    commands = {"example": torch.arange(4)}
    payload = {"target_index": 4, "label": 2, "source_validation_index": 222,
               "commands": commands,
               "rows": {condition: {"condition": condition, "target_index": 0,
                                    "label": 2, "source_validation_index": 222}
                        for condition in diagnosis.CONDITIONS}}
    original_commands = payload["commands"]["example"].clone()
    repairs: list[dict[str, object]] = []

    changed = diagnosis._repair_measurement_row_identity(
        payload, section="conditions", target_index=4, conditions=diagnosis.CONDITIONS,
        repair_rows=repairs)
    repeated = diagnosis._repair_measurement_row_identity(
        payload, section="conditions", target_index=4, conditions=diagnosis.CONDITIONS,
        repair_rows=repairs)

    assert changed is True
    assert repeated is False
    assert len(repairs) == 4
    assert all(row["target_index"] == 4 and row["label"] == 2
               and row["source_validation_index"] == 222 for row in payload["rows"].values())
    assert torch.equal(payload["commands"]["example"], original_commands)


def test_measurement_row_identity_repair_rejects_unexpected_old_target() -> None:
    payload = {"target_index": 4, "label": 2, "source_validation_index": 222,
               "rows": {"frozen_v2": {"condition": "frozen_v2", "target_index": 3,
                                        "label": 2, "source_validation_index": 222}}}

    with pytest.raises(RuntimeError, match="cannot be safely repaired"):
        diagnosis._repair_measurement_row_identity(
            payload, section="conditions", target_index=4, conditions=("frozen_v2",),
            repair_rows=[])


def test_fixed_phase_witness_does_not_realign_a_second_global_phase() -> None:
    target = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    phase_shifted = torch.tensor([0.0 + 1.0j], dtype=torch.complex128)

    assert diagnosis.reachability._phase_error(phase_shifted, target)[1] < 1.0e-15
    assert abs(diagnosis._fixed_phase_relative_error(phase_shifted, target) - 2.0**0.5) < 1.0e-15


def test_phase_shifted_target_uses_complex128_before_applying_phase() -> None:
    target = torch.tensor([0.12345678 + 0.23456789j, -0.34567891 + 0.45678912j],
                          dtype=torch.complex64)
    phase = 0.731
    shifted = diagnosis._phase_shifted_target(target, phase)
    independent = target.numpy().astype(np.complex128) * np.exp(1j * phase)
    old_order = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))

    assert shifted.dtype == torch.complex128
    assert np.allclose(shifted.numpy(), independent, atol=1.0e-16, rtol=1.0e-15)
    assert old_order.dtype == torch.complex64
    assert float(torch.max(torch.abs(shifted - old_order.to(torch.complex128)))) > 1.0e-9


def test_condition_summaries_include_per_class_metrics_and_pass_counts() -> None:
    rows = []
    for condition in diagnosis.CONDITIONS:
        for label in range(diagnosis.CLASS_COUNT):
            value = 2.0e-4 if label == 9 else label * 1.0e-5
            rows.append({"condition": condition, "label": label,
                         "row_passed": value <= diagnosis.GATE_LIMIT,
                         **{key: value for key in diagnosis.METRIC_KEYS}})

    summary = diagnosis._summarize_conditions(rows)

    for condition in diagnosis.CONDITIONS:
        assert summary[condition]["denominator"] == 10
        assert summary[condition]["per_class"]["9"]["denominator"] == 1
        assert summary[condition]["per_class"]["9"]["passed_rows"] == 0
        assert summary[condition]["per_class"]["9"]["metrics"]["coefficient_2048"]["mean"] == 2.0e-4


def test_report_uses_fixed_phase_repair_counts_from_saved_audit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(diagnosis, "ATTEMPT_DIR", tmp_path)
    diagnosis._atomic_json(tmp_path / "measurement_row_identity_repair.json", {
        "rows": [{"target_index": index} for index in range(116)],
        "fixed_phase_metric_repair": {
            "status": "COMPLETE", "targets_recomputed": 20,
            "updated_targets": 20, "decision_changes": 0,
        },
    })

    report = diagnosis._write_report(
        {}, "run", {"source_validation_indices": torch.arange(20)},
        {"total_start_iterations": 24_072}, [], {},
        {"any_condition_witness_rows": 20, "fixed_phase_witness_rows": 20,
         "classification": "CURRENT_PHASE_WITNESS_20_OF_20",
         "interpretation": "20/20 witnessed."},
    )

    assert "固定相位误差在20条保存输出上重新计算" in report
    assert "其中20条数值修订、0条门槛判定变化" in report


def test_phase_metric_review_transfer_reconciles_actual_sizes(tmp_path) -> None:
    (tmp_path / "review_identity.json").write_text("review", encoding="utf-8")
    (tmp_path / "conditions.pt").write_bytes(b"review-conditions")
    manifest = diagnosis._build_review_transfer_manifest(tmp_path, "run")
    files = list(manifest["files"]) + [{
        "path": "transfer_manifest.json",
        "bytes": (tmp_path / "transfer_manifest.json").stat().st_size,
    }]
    diagnosis._atomic_json(tmp_path / "transfer_receipt.json", {
        "status": "COMPLETE", "remote_alias": "autodl-monthly",
        "remote_run_dir": manifest["remote_run_dir"],
        "files": [{"path": item["path"], "status": "COPIED", "scp_exit_code": 0,
                   "remote_size_bytes": item["bytes"], "local_size_bytes": item["bytes"]}
                  for item in files],
    })

    result = diagnosis._verify_review_transfer(tmp_path, "run")

    assert result["status"] == "COMPLETE"
    assert result["file_count"] == 3
    assert result["bytes_total"] == sum(item["bytes"] for item in files)
    (tmp_path / "conditions.pt").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="size/status mismatch"):
        diagnosis._verify_review_transfer(tmp_path, "run")


def test_exact_mapper_command_replay_rejects_small_command_substitution() -> None:
    saved = torch.zeros((1, 1, 200, 200), dtype=torch.float32)

    diagnosis._require_exact_command_replay(saved.clone(), saved, 4, "box_b5000")
    substituted = saved.clone()
    substituted[0, 0, 0, 0] = 1.0e-8
    with pytest.raises(RuntimeError, match="differs from exact CUDA replay"):
        diagnosis._require_exact_command_replay(substituted, saved, 4, "box_b5000")


def test_windows_display_conversion_uses_area_and_fixed_gain_pixelwise() -> None:
    raw = torch.tensor([[0.0, 1.0e8], [2.0e10, 5.0e10]], dtype=torch.float64).numpy()

    display = diagnosis._display_uint8(raw, 6.25e-12, 1.0)

    assert display.tolist() == [[0, 0], [32, 80]]


def test_complex_residual_decomposition_retains_all_cross_terms() -> None:
    response = torch.tensor([[0.8 + 0.1j, 0.2 - 0.3j, 0.4 + 0.2j],
                             [0.1 - 0.2j, 0.7 + 0.1j, 0.3 - 0.4j]], dtype=torch.complex128)
    target = torch.tensor([0.4 + 0.2j, 0.3 - 0.1j], dtype=torch.complex128)
    phase = 0.23
    preclip = torch.tensor([-0.05, 0.46, 1.08], dtype=torch.float64)
    amplitude = preclip.clamp(0.0, 1.0).to(torch.float32)
    actual = response @ amplitude.to(torch.complex128)

    result = diagnosis._residual_decomposition(response, target, phase, preclip, amplitude, actual)

    assert result["identity_vector_max_abs_error"] <= 1.0e-14
    assert result["identity_norm_sq_abs_error"] <= 1.0e-14
    assert len(result["cross_terms_twice_real_inner_product"]) == 6


def test_fixed_phase_checkpoint_restores_same_fourth_iteration() -> None:
    matrix = torch.tensor([[1.0 + 0.2j, 0.3 - 0.1j, 0.2 + 0.3j],
                           [0.1 + 0.4j, 0.8 + 0.2j, 0.4 - 0.2j]], dtype=torch.complex128)
    target = torch.tensor([0.43 + 0.1j, 0.24 - 0.07j], dtype=torch.complex128)
    phase = 0.37

    uninterrupted = diagnosis._fixed_phase_solve(matrix, target, phase, max_steps=4)
    checkpoint = diagnosis._fixed_phase_solve(matrix, target, phase, max_steps=3)
    restored = diagnosis._fixed_phase_solve(matrix, target, phase, max_steps=4,
                                             initial_state=checkpoint["state"])

    assert uninterrupted["iterations"] == restored["iterations"] == 4
    assert torch.equal(uninterrupted["state"]["amplitude"], restored["state"]["amplitude"])
    assert torch.equal(uninterrupted["state"]["best_amplitude"], restored["state"]["best_amplitude"])
    assert torch.equal(uninterrupted["state"]["best_error"], restored["state"]["best_error"])
    assert uninterrupted["state"]["loss"] == restored["state"]["loss"]


def test_solver_checkpoint_counts_every_free_phase_start() -> None:
    targets = {"source_run_id": "fullpop-run"}
    results = [{"fixed_phase": {"iterations": 4}, "free_phase": {"iterations": 6}}]

    payload = diagnosis._solver_checkpoint_payload(
        SimpleNamespace(run_id="diagnosis-run"), targets, results, 1, "free_phase",
        {"iteration": 2}, 2,
    )

    assert payload["run_id"] == "diagnosis-run"
    assert payload["source_run_id"] == payload["source_fullpop_run_id"] == "fullpop-run"
    assert payload["committed_start_iterations"] == 4 + (6 * 8) + (2 * 8)


def test_solver_checkpoint_accepts_fixed_phase_only_route_transition() -> None:
    targets = {"source_run_id": "fullpop-run"}
    fixed_only_row = {"target_index": 0, "fixed_phase": {"iterations": 17}}

    payload = diagnosis._solver_checkpoint_payload(
        SimpleNamespace(run_id="diagnosis-run"), targets, [fixed_only_row], 0,
        "free_phase", None, 0,
    )

    assert payload["branch"] == "free_phase"
    assert payload["committed_start_iterations"] == 17


def test_attempt_identity_reconsumes_its_embedded_preflight(tmp_path, monkeypatch) -> None:
    config = diagnosis._load_config(diagnosis.DEFAULT_CONFIG)
    ready = diagnosis._run_identity(config, "source-run", "diagnosis-run")
    attempt = ready.with_scientific_attempt("diagnosis-run:attempt_01")
    evidence_dir = tmp_path / "attempt_01"
    preflight_dir = evidence_dir / "preflight_evidence"
    preflight_dir.mkdir(parents=True)
    diagnosis._atomic_json(preflight_dir / "preflight_identity.json", {
        "task_id": diagnosis.TASK_ID, "run_id": ready.run_id,
        "run_identity": ready.to_dict(),
    })
    diagnosis._atomic_json(evidence_dir / "run_identity.json", attempt.to_dict())
    monkeypatch.setattr(diagnosis, "PREFLIGHT_DIR", tmp_path / "cloud_only_preflight")

    loaded = diagnosis._assert_attempt(evidence_dir)

    assert loaded.execution_semantics() == attempt.execution_semantics()


def test_initial_solver_checkpoint_is_a_valid_zero_iteration_resume_point() -> None:
    targets = {"source_run_id": "fullpop-run"}

    payload = diagnosis._solver_checkpoint_payload(
        SimpleNamespace(run_id="diagnosis-run"), targets, [], 0, "fixed_phase", None, 0,
    )

    assert payload["target_cursor"] == 0
    assert payload["branch"] == "fixed_phase"
    assert payload["iteration"] == 0
    assert payload["solver_state"] is None
    assert payload["committed_start_iterations"] == 0
