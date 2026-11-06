from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reach
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_transfer_diagnosis_v1 as task


def test_diagnostic_population_indices_bind_two_saved_rows_per_class() -> None:
    assert task._target_indices() == [
        index for label in range(10) for index in (20 * label, 20 * label + 1)
    ]


def _synthetic_source_and_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 command: torch.Tensor | None = None):
    source_dir = tmp_path / "source"
    review_dir = tmp_path / "review"
    source_dir.mkdir()
    review_dir.mkdir()
    command = torch.zeros((1, 1, 200, 200), dtype=torch.float32) if command is None else command
    state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    scale = torch.ones(100, dtype=torch.float64)
    response = torch.zeros((100, 40000), dtype=torch.complex128)
    inverse = torch.zeros((40000, 200), dtype=torch.float64)
    populations = {}
    terminal_roles = {}
    for role in ("train", "development", "holdout"):
        coeff = torch.ones((1, 100), dtype=torch.complex128)
        populations[role] = {"coefficients": coeff, "source_indices": torch.tensor([0]),
                             "labels": torch.tensor([0]), "proposal_ids": torch.tensor([10])}
        terminal_roles[role] = {"indices": torch.tensor([0]), "labels": torch.tensor([0]),
                                "proposal_ids": torch.tensor([10]), "target_coefficients": coeff.clone(),
                                "commands": command.clone()}
    original_payloads = {
        role: {"task_id": task.fixed.TASK_ID, "cohort_role": role, "start": 0, "stop": 1,
               "indices": torch.tensor([0]), "commands": command.clone(),
               "frozen_v2_commands": command.clone()}
        for role in ("train", "development", "holdout")
    }
    source = {
        "source_dir": source_dir,
        "checkpoint": {"run_id": "source-run", "mapper_state": state},
        "inputs": {"source_mapper": {"mapper_state": state}, "response_matrix": response,
                   "inverse": inverse, "scale": scale},
        "population": populations,
    }
    binding = {"schema": f"{task.TASK_ID}.model_binding.v1", "source_run_id": "source-run",
               "terminal_step": 15000, "terminal_loaded_state": state,
               "v2_loaded_state": state, "response_matrix": response,
               "right_inverse": inverse, "scale": scale}
    terminal_payload = {"task_id": task.TASK_ID, "source_run_id": "source-run",
                        "roles": terminal_roles}
    v2_payload = {"task_id": task.TASK_ID, "role": terminal_roles["holdout"]}
    summary = {"schema": f"{task.TASK_ID}.source_review.v1", "task_id": task.TASK_ID,
               "status": "SOURCE_REPLAY_PRODUCED", "source_run_id": "source-run"}
    torch.save(binding, review_dir / "model_binding.pt")
    torch.save(terminal_payload, review_dir / "terminal_command_replay.pt")
    torch.save(v2_payload, review_dir / "v2_holdout_command_replay.pt")
    (review_dir / "source_review_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    task._write_manifest(review_dir)
    _write_synthetic_transfer_receipt(review_dir)

    monkeypatch.setattr(task, "_load_source", lambda _source_dir, _config: source)
    monkeypatch.setattr(task, "_load_shard_row",
                        lambda _source_dir, role, _index: (original_payloads[role], 0))
    monkeypatch.setattr(task, "_cpu_command_rows",
                        lambda _state, targets, _inputs: command.expand(targets.shape[0], -1, -1, -1).clone())
    config = {"source": {"v2_terminal_checkpoint": "unused.pt"}}
    return source_dir, review_dir, source, config, original_payloads, command


def _write_synthetic_transfer_receipt(review_dir: Path) -> None:
    paths = task._manifest_transfer_paths(review_dir)
    files = []
    for relative in sorted(paths):
        size = (review_dir / relative).stat().st_size
        files.append({"path": relative, "status": "COPIED", "scp_exit_code": 0,
                      "remote_size_bytes": size, "local_size_bytes": size})
    receipt = {"status": "COMPLETE", "destination": str(review_dir.resolve()),
               "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/"
                                "20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/"
                                "runs/source_review_01", "files": files}
    (review_dir / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_manifest_allows_only_autodl_root_telemetry_sidecar(tmp_path: Path) -> None:
    evidence = tmp_path / "preflight"
    evidence.mkdir()
    (evidence / "preflight_summary.json").write_text("{}", encoding="utf-8")
    task._write_manifest(evidence)
    (evidence / "resource_telemetry.csv").write_text("timestamp_utc,run_id\n", encoding="utf-8")
    assert task._verify_manifest_files(evidence)["files"] == [
        {"path": "preflight_summary.json", "bytes": 2}]

    (evidence / "unregistered_evidence.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="does not enumerate the current evidence files exactly"):
        task._verify_manifest_files(evidence)


def test_transfer_receipt_accepts_registered_runs_root_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "attempt_02" / "result.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"result")
    receipt = {
        "status": "COMPLETE",
        "destination": str(bundle.resolve()),
        "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/"
                          "20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs",
        "files": [{"path": "attempt_02/result.bin", "status": "COPIED", "scp_exit_code": 0,
                   "remote_size_bytes": artifact.stat().st_size,
                   "local_size_bytes": artifact.stat().st_size}],
    }
    task._write_json(bundle / "transfer_receipt.json", receipt)
    assert task._verify_transfer_receipt(bundle, {"attempt_02/result.bin"})["status"] == "COMPLETE"

    receipt["remote_run_dir"] = "/root/autodl-tmp/codex-original/outputs/other/runs"
    task._write_json(bundle / "transfer_receipt.json", receipt)
    with pytest.raises(RuntimeError, match="points at a different run"):
        task._verify_transfer_receipt(bundle, {"attempt_02/result.bin"})


def test_residual_verification_scales_roundoff_sized_terms_to_target() -> None:
    response = np.eye(2, dtype=np.complex128)
    target = np.asarray([1.0 + 1.0e-15, 0.0], dtype=np.complex128)
    amplitude = np.asarray([1.0, 0.0], dtype=np.float64)
    expected = task._np_residual_decomposition(response, target, 0.0, amplitude, amplitude,
                                               response @ amplitude)
    saved = {**expected, "terms": {
        name: torch.as_tensor(value, dtype=torch.complex128).clone()
        for name, value in expected["terms"].items()
    }}
    saved["terms"]["unclipped_matrix_residual"] = torch.zeros(2, dtype=torch.complex128)
    delta = task._verify_numpy_decomposition(saved, expected, 0, "test", 2048,
                                             float(np.linalg.norm(target)))
    assert delta <= 1.0e-8


def test_source_consumer_accepts_bound_cuda_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir, review_dir, source, config, _, _ = _synthetic_source_and_review(tmp_path, monkeypatch)
    result = task._verify_source(config, source_dir, review_dir)
    assert result["status"] == "SOURCE_REPLAY_VERIFIED"
    assert result["terminal_rows_verified"] == 3


def test_source_consumer_rejects_transfer_receipt_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir, review_dir, _, config, _, _ = _synthetic_source_and_review(tmp_path, monkeypatch)
    receipt_path = review_dir / "transfer_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"][0]["remote_size_bytes"] += 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="current size differs from the receipt"):
        task._verify_source(config, source_dir, review_dir)
    monkeypatch.setattr(task, "_verify_transfer_receipt", lambda *_args, **_kwargs: {})
    assert task._verify_source(config, source_dir, review_dir)["status"] == "SOURCE_REPLAY_VERIFIED"


def test_source_consumer_rejects_checkpoint_or_target_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir, review_dir, source, config, _, _ = _synthetic_source_and_review(tmp_path, monkeypatch)
    source["checkpoint"]["mapper_state"] = {"weight": torch.tensor([1.01], dtype=torch.float32)}
    with pytest.raises(RuntimeError, match="source binding"):
        task._verify_source(config, source_dir, review_dir)

    source["checkpoint"]["mapper_state"] = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    terminal_path = review_dir / "terminal_command_replay.pt"
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    terminal["roles"]["development"]["target_coefficients"][0, 0] += 0.25
    torch.save(terminal, terminal_path)
    task._write_manifest(review_dir)
    _write_synthetic_transfer_receipt(review_dir)
    with pytest.raises(RuntimeError, match="population identity"):
        task._verify_source(config, source_dir, review_dir)


def test_source_consumer_rejects_paired_measurement_and_replay_command_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    source_dir, review_dir, _, config, original_payloads, _ = _synthetic_source_and_review(
        tmp_path, monkeypatch, original)
    changed = original.clone()
    changed[..., 0, 0] = 0.05
    terminal_path = review_dir / "terminal_command_replay.pt"
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    terminal["roles"]["development"]["commands"] = changed.clone()
    torch.save(terminal, terminal_path)
    original_payloads["development"]["commands"] = changed.clone()
    task._write_manifest(review_dir)
    _write_synthetic_transfer_receipt(review_dir)
    with pytest.raises(RuntimeError, match="CPU checkpoint replay"):
        task._verify_source(config, source_dir, review_dir)


def test_diagnostic_consumer_rejects_nullspace_instruction_replacement() -> None:
    original = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    changed = original.clone()
    changed[..., 0, 0] = 0.05
    assert task._consume_source_bound_command(original, original, require_bitwise=True) == 0.0
    with pytest.raises(RuntimeError, match="differs from its bound source"):
        task._consume_source_bound_command(changed, original, require_bitwise=True)


def test_reused_full_optics_rows_bind_target_command_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.ones(100, dtype=torch.complex128)
    command = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    command.reshape(-1)[:100] = 1.0
    response = torch.zeros((100, 40000), dtype=torch.complex128)
    response[:, :100] = torch.eye(100, dtype=torch.complex128)
    source_shard = {
        "task_id": task.fixed.TASK_ID,
        "cohort_role": "development",
        "start": 0,
        "stop": 1,
        "indices": torch.tensor([0]),
        "labels": torch.tensor([0]),
        "proposal_ids": torch.tensor([10]),
        "target_coefficients": target.unsqueeze(0),
        "commands": command.clone(),
        "candidate_outputs": {
            grid: {"coefficients": target.unsqueeze(0),
                   "raw_cmos": torch.full((1, 1, 200, 200), 2.0, dtype=torch.float64)}
            for grid in task.GRIDS
        },
        "direct_raw_cmos": {
            grid: torch.ones((1, 1, 200, 200), dtype=torch.float64) for grid in task.GRIDS
        },
    }
    source_shard["rows"] = [task._np_metric_row(
        {grid: {"coefficients": source_shard["candidate_outputs"][grid]["coefficients"][0].numpy(),
                "raw_cmos": source_shard["candidate_outputs"][grid]["raw_cmos"][0].numpy()}
         for grid in task.GRIDS},
        target.numpy(),
        {grid: source_shard["direct_raw_cmos"][grid][0].numpy() for grid in task.GRIDS})]
    source = {
        "source_dir": Path("unused"),
        "population": {"development": {
            "coefficients": target.unsqueeze(0),
            "labels": torch.tensor([0]),
            "proposal_ids": torch.tensor([10]),
        }},
        "inputs": {"response_matrix": response},
    }
    measurement = {
        "source_population_index": 0,
        "label": 0,
        "proposal_id": 10,
        "target_coefficients": target,
        "source_measurement_reference": task._saved_source_measurement_reference(0),
    }
    monkeypatch.setattr(task, "_load_shard_row", lambda *_args: (source_shard, 0))

    direct, terminal, reference = task._reused_source_measurements(
        source, measurement, command, validate_source_measurement=True)
    assert reference == measurement["source_measurement_reference"]
    assert direct[2048].shape == terminal[2048]["raw_cmos"].shape
    assert torch.equal(direct[2048], source_shard["direct_raw_cmos"][2048])
    assert torch.equal(terminal[4096]["raw_cmos"], source_shard["candidate_outputs"][4096]["raw_cmos"])

    wrong_command = command.clone()
    wrong_command[..., 0, 0] = 0.25
    with pytest.raises(RuntimeError, match="command differs from its saved source row"):
        task._reused_source_measurements(source, measurement, wrong_command)
    wrong_target = {**measurement, "target_coefficients": target.clone()}
    wrong_target["target_coefficients"][0] += 0.5
    with pytest.raises(RuntimeError, match="not bound to fixed development target"):
        task._reused_source_measurements(source, wrong_target, command)


def test_recovery_state_comparator_uses_tolerance_for_numeric_scalars() -> None:
    assert task._nested_max_delta({"loss": 0.25}, {"loss": 0.25 + 5.0e-17}) == pytest.approx(5.0e-17)
    assert task._nested_max_delta({"cursor": 3}, {"cursor": 4}) == float("inf")


def test_zero_phase_projection_matches_independent_numpy() -> None:
    torch.manual_seed(45)
    matrix_np = np.random.default_rng(45).normal(size=(4, 9))
    matrix = torch.tensor(matrix_np, dtype=torch.float64)
    inverse = torch.tensor(np.linalg.pinv(matrix_np), dtype=torch.float64)
    scale_np = np.asarray([0.7, 1.4], dtype=np.float64)
    scale = torch.tensor(scale_np)
    raw_np = np.random.default_rng(46).uniform(0.2, 0.8, size=(1, 9)).astype(np.float32)
    target_np = np.asarray([[0.3 + 0.1j, 0.5 + 0.2j]], dtype=np.complex128)
    raw = torch.tensor(raw_np)
    target = torch.tensor(target_np)
    command, _ = task._fixed_phase_projection(raw, target, matrix, inverse, scale, 0.0)

    predicted_ri = raw_np.astype(np.float64) @ matrix_np.T
    normalized_target = target_np / np.sqrt(scale_np)[None, :]
    target_ri = np.stack((normalized_target.real, normalized_target.imag), axis=-1).reshape(1, -1)
    expected = np.clip(raw_np.astype(np.float64) + (target_ri - predicted_ri) @ inverse.numpy().T,
                       0.0, 1.0).astype(np.float32)
    np.testing.assert_array_equal(command.reshape(1, -1).numpy(), expected)


def test_numpy_metric_consumer_recomputes_six_metrics() -> None:
    target = np.asarray([1.0 + 0j, 0.3 - 0.2j], dtype=np.complex128)
    actual2048 = np.exp(0.4j) * target
    actual4096 = np.exp(-0.2j) * target
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    output = {
        2048: {"coefficients": actual2048, "raw_cmos": raw},
        4096: {"coefficients": actual4096, "raw_cmos": raw * (1.0 + 1.0e-5)},
    }
    reference = {2048: raw * (1.0 + 2.0e-5), 4096: raw * (1.0 + 3.0e-5)}
    result = task._np_metric_row(output, target, reference)
    assert result["coefficient_2048"] < 1.0e-15
    assert result["coefficient_4096"] < 1.0e-15
    assert result["native_cmos_2048"] == pytest.approx(2.0e-5 / 1.00002, rel=1.0e-9)
    assert result["cross_grid_native_cmos"] == pytest.approx(1.0e-5 / 1.00001, rel=1.0e-9)


def test_numpy_residual_decomposition_keeps_cross_terms_and_rejects_tampering() -> None:
    rng = np.random.default_rng(28)
    response = rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    preclip = rng.uniform(-0.2, 1.2, size=9)
    command = np.clip(preclip, 0.0, 1.0).astype(np.float32).astype(np.float64)
    actual = response @ command + 1.0e-9 * (rng.normal(size=4) + 1j * rng.normal(size=4))
    expected = task._np_residual_decomposition(response, target, 0.37, preclip, command, actual)
    saved = {**expected,
             "terms": {key: torch.from_numpy(value.copy()) for key, value in expected["terms"].items()},
             "term_squared_norms": dict(expected["term_squared_norms"]),
             "pairwise_cross_terms": dict(expected["pairwise_cross_terms"])}
    assert task._verify_numpy_decomposition(saved, expected, 0, "synthetic", 2048,
                                            float(np.linalg.norm(target))) < 1.0e-14
    saved["pairwise_cross_terms"][next(iter(saved["pairwise_cross_terms"]))] += 1.0
    with pytest.raises(RuntimeError, match="residual decomposition scalar differs"):
        task._verify_numpy_decomposition(saved, expected, 0, "synthetic", 2048,
                                        float(np.linalg.norm(target)))


def test_independent_numpy_fixed_phase_bounds_match_solver_reference() -> None:
    rng = np.random.default_rng(31)
    response_np = rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))
    target_np = rng.normal(size=4) + 1j * rng.normal(size=4)
    amplitude_np = rng.uniform(0.0, 1.0, size=9)
    phase = -0.63
    expected = task._np_fixed_phase_box_bounds(response_np, target_np, phase, amplitude_np)
    actual = task.exposure._dual_box_error_bounds(
        torch.tensor(response_np, dtype=torch.complex128),
        torch.tensor(np.exp(1j * phase) * target_np, dtype=torch.complex128),
        torch.tensor(amplitude_np, dtype=torch.float64))
    for name in expected:
        assert expected[name] == pytest.approx(actual[name], abs=1.0e-14)


def test_nonzero_fixed_phase_solver_initialization_and_loss_use_rotated_target(tmp_path: Path) -> None:
    phase = 0.37
    response = torch.zeros((1, 40_000), dtype=torch.complex128)
    response[0, 0] = 1.0
    target = torch.tensor([0.5 * np.exp(-1j * phase)], dtype=torch.complex128)
    seed_amplitude = torch.zeros(40_000, dtype=torch.float64)
    seed_amplitude[0] = 0.5
    base_row = {
        "index": 0,
        "target_coefficients": target,
        "class_mean_amplitude": seed_amplitude,
        "fixed_phase_rad": {"fista_fixed_terminal_phase": phase},
    }

    result = task._solve_target(base_row, None, 1, tmp_path, 0, "phase-test", 0,
                                response, 1.0, torch.device("cpu"))
    state = {"phase_rad": phase, "best_amplitude": result["amplitude_float64"]}
    assert result["relative_error"] <= 1.0e-12
    assert task._solver_state_loss("fixed", response, target, state) <= 1.0e-12


def test_cached_response_spectral_constant_avoids_recomputation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reach, "_projected_spectral_norm",
                        lambda _matrix: pytest.fail("cached spectral constant was ignored"))
    response = torch.eye(2, dtype=torch.complex128)
    target = torch.tensor([0.5 + 0j, 0.5 + 0j], dtype=torch.complex128)
    result = reach._bounded_phase_solve(response, target, lipschitz=1.0)
    assert result["relative_error"] <= 1.0e-6


_CUDA_RECOVERY_SOURCE = task.ROOT / "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01"


@pytest.mark.parametrize("kind", ["fixed", "free"])
@pytest.mark.skipif(not torch.cuda.is_available()
                    or not (_CUDA_RECOVERY_SOURCE / "checkpoints" / "terminal_checkpoint.pt").is_file(),
                    reason="CUDA and the explicitly fixed-frame source checkpoint are required")
def test_solver_recovery_subprocess_consumes_registered_output_dir(
    kind: str, tmp_path: Path,
) -> None:
    config = task._config(task.DEFAULT_CONFIG)
    source = task._load_source(_CUDA_RECOVERY_SOURCE, config)
    task.fixed._set_numerics()
    device = torch.device("cuda:0")
    response = source["inputs"]["response_matrix"].to(device=device, dtype=torch.complex128)
    target = source["population"]["development"]["coefficients"][0].to(
        device=device, dtype=torch.complex128)
    lipschitz = reach._projected_spectral_norm(response)
    directory = tmp_path / kind
    directory.mkdir()
    result = task._solver_recovery_case(
        kind, response, target, 0.0, lipschitz, directory, f"test-{kind}",
        _CUDA_RECOVERY_SOURCE, source["checkpoint"]["run_id"])
    assert result["state_max_abs_delta"] <= task.SOLVER_RECOVERY_TOLERANCE
    assert result["loss_abs_delta"] <= task.SOLVER_RECOVERY_TOLERANCE
