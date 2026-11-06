from __future__ import annotations

import sys

import pytest
import torch

from scripts.research.lightgen import diagnose_ols_four_nine_window_closure_v8 as v8


def test_v8_config_freezes_windows_primary_grid_and_two_candidates() -> None:
    config = v8._load_config(v8.DEFAULT_CONFIG)
    assert tuple(config["optics"]["propagation_grid_sizes"]) == (4096, 6144, 8192)
    assert config["optics"]["primary_response_grid"] == 4096
    assert {item["endpoint_key"] for item in config["candidates"]["optimized"]} == set(v8.TARGET_NAMES)
    assert config["candidates"]["zero_phase_controls"] == []
    assert config["scope"]["formal_training_updates"] == 0
    assert config["scope"]["phase_optimization_updates"] == 0


def test_primary_response_grid_rejects_mixed_matrix_and_jacobian() -> None:
    v8._validate_matrix_jacobian_grid(4096, 4096, 4096)
    with pytest.raises(ValueError, match="configured primary grid"):
        v8._validate_matrix_jacobian_grid(4096, 3072, 4096)


def test_row_selection_ignores_scalar_archive_provenance() -> None:
    coordinates = v8._coordinate_pool([2], torch.device("cpu"))
    matrix = torch.arange(4 * 3, dtype=torch.float64).reshape(4, 3).to(torch.complex128)
    pool = {
        "coordinates_yx_m": torch.tensor(coordinates, dtype=torch.float64),
        "K": matrix,
        "H": matrix * (1.0 + 1.0j),
        "source_archive": torch.tensor(1),
    }
    selected = v8._row_selection(pool, 2, torch.device("cpu"))
    assert set(selected) == {"K", "H"}
    assert selected["K"].shape == (4, 3)
    assert torch.equal(selected["K"], matrix)


def test_all_channel_controls_cover_real_imaginary_targets_at_all_scales() -> None:
    config = v8._load_config(v8.DEFAULT_CONFIG)
    h = torch.zeros((4, 10000), dtype=torch.complex128)
    h[0, 0] = 1.0e-4 + 1.0e-5j
    h[1, 1] = 1.2e-4 - 2.0e-5j
    h[2, 2] = -1.1e-4 + 3.0e-5j
    h[3, 3] = 0.9e-4 + 4.0e-5j
    coefficients = h @ torch.full((10000,), 0.5, dtype=torch.complex128)
    records, direct_patterns, commands = v8._control_targets_all_channels(h, coefficients, 2, config)
    target_records = [record for record in records if record["target_family"] == "single_channel_basis"]
    assert len(target_records) == 4 * 2 * 3
    assert {record["target_channel_index"] for record in target_records} == {0, 1, 2, 3}
    assert {record["target_component"] for record in target_records} == {"real", "imag"}
    assert {record["target_magnitude_scale_index"] for record in target_records} == {0, 1, 2}
    assert len(commands) == len(records)
    assert len(direct_patterns) == 8
    for _, delta in direct_patterns:
        assert float((0.5 + delta).min()) >= 0.25 - 1.0e-12
        assert float((0.5 + delta).max()) <= 0.75 + 1.0e-12


def test_window_metrics_align_only_one_common_phase() -> None:
    reference = torch.tensor([1.0 + 0.5j, -0.25 + 2.0j, 0.75 - 0.125j], dtype=torch.complex128)
    phase = 0.37
    test = reference * torch.polar(torch.ones((), dtype=torch.float64), torch.tensor(phase, dtype=torch.float64))
    metrics = v8._window_pair_metrics(reference, test)
    assert metrics["raw_complex_relative_error"] > 0.1
    assert metrics["common_global_phase_aligned_complex_relative_error"] < 1.0e-14
    assert abs(metrics["common_global_phase_rad"] - phase) < 1.0e-14
    assert metrics["intensity_relative_l2_error"] < 1.0e-14
    assert metrics["roi_power_relative_error"] < 1.0e-14


def test_analytic_common_phase_control_accepts_global_piston_and_rejects_wrong_sign() -> None:
    base = torch.tensor([1.0 + 0.5j, -0.25 + 2.0j, 0.75 - 0.125j], dtype=torch.complex128)
    phase_rad = 1.0e-4
    common = base * torch.exp(torch.tensor(1j * phase_rad, dtype=torch.complex128))
    errors = v8._analytic_common_phase_errors(base, common, common, base, phase_rad)
    assert max(errors.values()) < 1.0e-14

    wrong_sign = base * torch.exp(torch.tensor(-1j * phase_rad, dtype=torch.complex128))
    rejected = v8._analytic_common_phase_errors(base, wrong_sign, common, base, phase_rad)
    assert rejected["first_screen_common_phase_relative_error"] > 1.0e-4


def test_pattern_acceptance_requires_every_window_matrix_and_field_gate() -> None:
    evidence = {
        "window_qualified": True,
        "matrix_prediction_relative_error_by_grid": {"4096": 2.0e-5, "6144": 4.0e-5, "8192": 3.0e-5},
    }
    assert v8._pattern_matrix_and_window_pass(evidence, 1.0e-4)
    evidence["matrix_prediction_relative_error_by_grid"]["8192"] = 1.1e-4
    assert not v8._pattern_matrix_and_window_pass(evidence, 1.0e-4)
    evidence["matrix_prediction_relative_error_by_grid"].pop("8192")
    assert not v8._pattern_matrix_and_window_pass(evidence, 1.0e-4)
    evidence["matrix_prediction_relative_error_by_grid"]["8192"] = 3.0e-5
    evidence["window_qualified"] = False
    assert not v8._pattern_matrix_and_window_pass(evidence, 1.0e-4)


def test_saved_v7_window_error_parser_ignores_boolean_summary_fields() -> None:
    summary = {
        "3072_vs_4096": {"common_global_phase_aligned_complex_relative_error": 2.0e-5},
        "3072_vs_6144": {"common_global_phase_aligned_complex_relative_error": 8.0e-5},
        "4096_vs_6144": {"common_global_phase_aligned_complex_relative_error": 3.0e-5},
        "all_three_window_comparisons_pass": True,
    }
    assert v8._saved_pair_aligned_errors(summary) == {
        "3072_vs_4096": 2.0e-5,
        "3072_vs_6144": 8.0e-5,
        "4096_vs_6144": 3.0e-5,
    }
    with pytest.raises(ValueError, match="missing a numeric pair record"):
        v8._saved_pair_aligned_errors({"all_three_window_comparisons_pass": False})


def test_failed_preflight_returns_nonzero_cli_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["v8", "--preflight-only"])
    monkeypatch.setattr(v8, "run", lambda *_args, **_kwargs: {"status": "FAIL_ENGINEERING_PREFLIGHT", "passed": False})
    with pytest.raises(SystemExit) as error:
        v8.main()
    assert error.value.code == 2


def test_repaired_preflight_appends_source_snapshots_without_digest_lock(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(v8, "_preflight", lambda *_args: {"status": "PASS", "passed": True})
    v8.run(v8.DEFAULT_CONFIG, tmp_path, preflight_only=True)
    v8.run(v8.DEFAULT_CONFIG, tmp_path, preflight_only=True)
    invocations = sorted((tmp_path / "source_snapshot").glob("invocation_*"))
    assert len(invocations) == 2
    assert all((item / "configuration.yaml").is_file() for item in invocations)
    assert all((item / "diagnose_ols_four_nine_window_closure_v8.py").is_file() for item in invocations)


