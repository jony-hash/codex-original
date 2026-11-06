from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen.execute_lightgen_g1_slm_projected_mapping_v2 import (
    _consume_production_row,
    _error_row,
    _load_matrix,
    _numpy_minimum_norm_correction,
    _phase_error,
    _project_double_from_raw,
    _project_from_raw,
    _raw_error,
    _quantized_amplitudes,
    _verify_transfer_sizes,
    _write_transfer_manifest,
)


def _identity_projection() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = torch.zeros((200, 40_000), dtype=torch.float64)
    matrix[:, :200] = torch.eye(200, dtype=torch.float64)
    inverse = torch.zeros((40_000, 200), dtype=torch.float64)
    inverse[:200, :] = torch.eye(200, dtype=torch.float64)
    scale = torch.ones(100, dtype=torch.float64)
    return matrix, inverse, scale


def test_camera_pixel_area_is_read_from_the_frozen_camera_plane() -> None:
    model = SimpleNamespace(
        runtime=SimpleNamespace(camera_plane=SimpleNamespace(pixel_pitch_m=(2.5e-6, 2.5e-6)))
    )
    assert projected._camera_pixel_area(model) == pytest.approx(6.25e-12, rel=1.0e-15)


def test_numpy_metrics_align_one_global_phase_and_compare_whole_intensity_arrays() -> None:
    target = torch.arange(1, 101, dtype=torch.float64).to(torch.complex128)
    target = target + 1j * torch.flip(target.real, dims=(0,))
    shifted = target * torch.exp(torch.tensor(0.37j, dtype=torch.complex128))

    assert _phase_error(shifted, target) < 1.0e-14
    assert _raw_error(torch.full((1, 1, 2, 2), 2.0, dtype=torch.float64),
                      torch.full((1, 1, 2, 2), 2.0, dtype=torch.float64)) == 0.0
    with pytest.raises(ValueError, match="complex error inputs"):
        _phase_error(shifted, target[0])


def test_numpy_gram_solve_matches_full_moore_penrose_action() -> None:
    generator = np.random.default_rng(2026102711)
    matrix = generator.normal(size=(4, 12))
    residual = generator.normal(size=(3, 4))
    expected = residual @ np.linalg.pinv(matrix).T

    actual = _numpy_minimum_norm_correction(matrix, residual)

    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_fixed_projection_uses_interleaved_real_imaginary_order_and_clamps_only_after_correction() -> None:
    matrix, inverse, scale = _identity_projection()
    desired = torch.linspace(0.2, 0.8, 100, dtype=torch.float64)
    target = torch.complex(desired, desired).reshape(1, 100)
    raw = torch.full((1, 1, 200, 200), 0.5, dtype=torch.float64)

    amplitude, detail = _project_from_raw(raw, target, matrix, inverse, scale)
    continuous, _ = _project_double_from_raw(raw, target, matrix, inverse, scale)
    recovered = amplitude.double().reshape(1, 40_000) @ matrix.T

    assert amplitude.dtype == torch.float32
    assert continuous.dtype == torch.float64
    assert torch.equal(amplitude, continuous.float())
    assert torch.isfinite(amplitude).all()
    assert float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0
    assert torch.allclose(recovered[:, 0::2], desired.reshape(1, 100), atol=3e-8, rtol=0.0)
    assert torch.allclose(recovered[:, 1::2], desired.reshape(1, 100), atol=3e-8, rtol=0.0)
    assert float(detail["clipped_fraction"][0]) == 0.0


def test_matrix_loader_rejects_a_right_inverse_with_nullspace_contamination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(projected, "PORT_COUNT", 2)
    monkeypatch.setattr(projected, "REAL_DIM", 4)
    monkeypatch.setattr(projected, "PIXEL_COUNT", 5)
    response = torch.zeros((2, 5), dtype=torch.complex128)
    response[0, 0] = 1.0
    response[0, 2] = 1.0j
    response[1, 1] = 1.0
    response[1, 3] = 1.0j
    scale = torch.tensor([4.0, 9.0], dtype=torch.float64)
    matrix = projected._normalized_response(response, scale)
    expected_matrix = torch.zeros((4, 5), dtype=torch.float64)
    expected_matrix[0, 0] = 0.5
    expected_matrix[1, 2] = 0.5
    expected_matrix[2, 1] = 1.0 / 3.0
    expected_matrix[3, 3] = 1.0 / 3.0
    assert torch.equal(matrix, expected_matrix)
    inverse = torch.linalg.pinv(matrix)
    payload_path = tmp_path / "fixed_response_and_projector.pt"

    def save(candidate_inverse: torch.Tensor) -> None:
        torch.save({
            "schema": "lightgen_g1_slm_fixed_response_projector.v2",
            "task_id": projected.TASK_ID,
            "response_matrix": response,
            "normalized_real_matrix": matrix,
            "full_right_inverse": candidate_inverse,
            "scale": scale,
            "real_imag_order": "per_port_interleaved_real_imag",
        }, payload_path)

    save(inverse)
    loaded = _load_matrix(tmp_path, torch.device("cpu"),
                          expected_response=response, expected_scale=scale)
    assert torch.allclose(loaded[1] @ loaded[2], torch.eye(4, dtype=torch.float64), atol=1e-12)

    wrong_inverse = inverse.clone()
    wrong_inverse[4, 0] += 1.0
    assert torch.allclose(matrix @ wrong_inverse, torch.eye(4, dtype=torch.float64), atol=1e-12)
    save(wrong_inverse)
    with pytest.raises(RuntimeError, match="Moore-Penrose"):
        _load_matrix(tmp_path, torch.device("cpu"),
                     expected_response=response, expected_scale=scale)


def test_final_row_consumer_rejects_wrong_target_and_wrong_amplitude_intensity_encoding() -> None:
    response = torch.zeros((100, 40_000), dtype=torch.complex128)
    response[:, :100] = torch.eye(100, dtype=torch.complex128)
    amplitude = torch.full((1, 1, 200, 200), 0.4, dtype=torch.float32)
    actual = (amplitude.reshape(1, -1).double().to(torch.complex128) @ response.T)
    target = actual[0].clone()
    population = {
        "coefficients": torch.stack([target] + [target * (1.0 + i / 10) for i in range(1, 10)]),
        "labels": torch.arange(10),
        "proposal_ids": torch.arange(10),
    }
    raw = torch.ones((1, 1, 200, 200), dtype=torch.float64)
    outputs = {grid: {"coefficients": actual.clone(), "raw_cmos": raw.clone()}
               for grid in (2048, 4096)}
    direct = {grid: raw.clone() for grid in (2048, 4096)}
    consumed = _consume_production_row(
        population, 0, target, amplitude, outputs, direct, response,
        "continuous_amplitude_float32",
    )
    metrics = {key: float(consumed[key]) for key in (
        "coefficient_2048", "coefficient_4096", "native_cmos_2048", "native_cmos_4096",
        "cross_grid_coefficient", "cross_grid_native_cmos",
    )}

    with pytest.raises(ValueError, match="target-population identity"):
        _consume_production_row(
            population, 0, population["coefficients"][1], amplitude, outputs, direct,
            response, "continuous_amplitude_float32", reported_metrics=metrics,
        )
    with pytest.raises(ValueError, match="command encoding"):
        _consume_production_row(
            population, 0, target, amplitude, outputs, direct, response, "intensity_float32",
        )
    with pytest.raises(ValueError, match="amplitude-to-coefficient response"):
        _consume_production_row(
            population, 0, target, amplitude, {
                grid: {"coefficients": actual * 2, "raw_cmos": raw.clone()}
                for grid in (2048, 4096)
            }, direct, response, "continuous_amplitude_float32",
        )

    assert consumed["row_passed"]


def test_a8_and_i8_codes_follow_distinct_command_semantics() -> None:
    amplitude = torch.tensor([[[[0.5, 0.25]]]], dtype=torch.float32)
    a8_codes, a8_amplitude, i8_codes, i8_amplitude = _quantized_amplitudes(amplitude)

    assert a8_codes.tolist() == [[[[128, 64]]]]
    assert i8_codes.tolist() == [[[[64, 16]]]]
    assert torch.equal(a8_amplitude, a8_codes.double() / 255.0)
    assert torch.equal(i8_amplitude, torch.sqrt(i8_codes.double() / 255.0))
    assert not torch.equal(a8_amplitude, i8_amplitude)


def test_six_error_rows_keep_grid_and_cross_grid_metrics_separate() -> None:
    target = torch.ones((1, 100), dtype=torch.complex128)
    outputs = {
        2048: {"coefficients": target.clone(), "raw_cmos": torch.full((1, 1, 200, 200), 1.1, dtype=torch.float64)},
        4096: {"coefficients": target * 1.01, "raw_cmos": torch.full((1, 1, 200, 200), 1.2, dtype=torch.float64)},
    }
    direct = {grid: torch.ones((1, 1, 200, 200), dtype=torch.float64) for grid in (2048, 4096)}

    errors = _error_row(outputs, target, direct, 0)

    assert errors["coefficient_2048"] == pytest.approx(0.0, abs=1e-14)
    assert errors["coefficient_4096"] == pytest.approx(0.01, abs=1e-12)
    assert errors["native_cmos_2048"] == pytest.approx(0.1, abs=1e-12)
    assert errors["native_cmos_4096"] == pytest.approx(0.2, abs=1e-12)
    assert errors["cross_grid_coefficient"] == pytest.approx(0.01 / 1.01, abs=1e-12)
    assert errors["cross_grid_native_cmos"] == pytest.approx(1 / 12, abs=1e-12)


def test_transfer_consumer_rejects_a_file_truncated_after_manifest_creation(tmp_path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    payload = evidence / "measurement.bin"
    payload.write_bytes(b"complete-evidence")
    _write_transfer_manifest(evidence)
    payload.write_bytes(b"short")

    with pytest.raises(RuntimeError, match="size mismatch"):
        _verify_transfer_sizes(evidence)


def test_windows_transfer_consumer_checks_remote_receipt_and_actual_local_size(tmp_path) -> None:
    evidence = tmp_path / "backup"
    evidence.mkdir()
    payload = evidence / "measurement.bin"
    payload.write_bytes(b"complete-evidence")
    _write_transfer_manifest(evidence)
    manifest_path = evidence / "cloud_transfer_manifest.json"
    receipt = {
        "status": "COMPLETE",
        "remote_alias": "autodl-monthly",
        "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/20261027_"
        "lightgen_g1_slm_projected_mapping_v2/runs/attempt_01",
        "destination": str(evidence.resolve()),
        "files": [
            {"path": "measurement.bin", "remote_size_bytes": payload.stat().st_size,
             "local_size_bytes": payload.stat().st_size, "status": "COPIED"},
            {"path": "cloud_transfer_manifest.json", "remote_size_bytes": manifest_path.stat().st_size,
             "local_size_bytes": manifest_path.stat().st_size, "status": "COPIED"},
        ],
    }
    (evidence / "transfer_receipt.json").write_bytes(json.dumps(receipt).encode("utf-8-sig"))

    checked = _verify_transfer_sizes(evidence, require_transfer_receipt=True)

    assert checked["transfer_receipt"]["checked"]
    assert all(row["matched"] for row in checked["transfer_receipt"]["files"])


def test_windows_transfer_consumer_rejects_changed_file_after_copy_receipt(tmp_path) -> None:
    evidence = tmp_path / "backup"
    evidence.mkdir()
    payload = evidence / "measurement.bin"
    payload.write_bytes(b"complete-evidence")
    _write_transfer_manifest(evidence)
    manifest_path = evidence / "cloud_transfer_manifest.json"
    size = payload.stat().st_size
    receipt = {
        "status": "COMPLETE",
        "remote_alias": "autodl-monthly",
        "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/20261027_"
        "lightgen_g1_slm_projected_mapping_v2/runs/attempt_01",
        "destination": str(evidence.resolve()),
        "files": [
            {"path": "measurement.bin", "remote_size_bytes": size,
             "local_size_bytes": size, "status": "COPIED"},
            {"path": "cloud_transfer_manifest.json", "remote_size_bytes": manifest_path.stat().st_size,
             "local_size_bytes": manifest_path.stat().st_size, "status": "COPIED"},
        ],
    }
    payload.write_bytes(b"truncated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["remote_file_bytes"] = payload.stat().st_size
    manifest["total_bytes"] = payload.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt["files"][1]["remote_size_bytes"] = manifest_path.stat().st_size
    receipt["files"][1]["local_size_bytes"] = manifest_path.stat().st_size
    (evidence / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(RuntimeError, match="three-way transfer size mismatch"):
        _verify_transfer_sizes(evidence, require_transfer_receipt=True)


def test_saved_result_consumer_loads_binary_terminal_checkpoint(tmp_path, monkeypatch) -> None:
    (tmp_path / "run_identity.json").write_text(json.dumps({"run_id": "fixed-run"}), encoding="utf-8")
    preflight = tmp_path / "preflight_evidence"
    preflight.mkdir()
    (preflight / "preflight_summary.json").write_text(json.dumps({
        "branch": "ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL",
    }), encoding="utf-8")
    torch.save({"completed_updates": 5000}, tmp_path / "terminal_checkpoint.pt")
    summary = {
        "all_rows_passed": False,
        "denominator": 2000,
        "passed_rows": 0,
        "threshold": 1.0e-4,
        "maxima": {"coefficient_2048": 1.0},
        "per_class": {},
    }
    monkeypatch.setattr(projected, "_load_attempt_population", lambda _: ({"holdout": {}}, torch.empty(0)))
    monkeypatch.setattr(projected, "_reconsume_continuous", lambda *_: ([], summary))

    result = projected._consume_saved(
        tmp_path, {"response_matrix": torch.eye(2, dtype=torch.complex128)},
        finalize=False, write_rows=False,
    )

    assert result["new_mapper_updates"] == 5000
    assert result["branch"] == "ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL"
    assert result["lifecycle_state"] == projected.RUN_VALID_FAILED
