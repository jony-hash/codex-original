from __future__ import annotations

import csv
import gzip
import json

import numpy as np
import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import audit_lightgen_recent_plane_density_v1 as audit


def test_recent_density_config_fixes_population_and_zero_update_boundary() -> None:
    cfg = audit.read_config(audit.ROOT / "configs/research/lightgen/gaussian_recent_plane_density_audit_v1.yaml")
    assert [item["key"] for item in cfg["endpoints"]] == list(audit.ENDPOINTS)
    assert cfg["training_updates"] == cfg["optimizer_creation"] == cfg["prior_sampling"] == 0
    assert cfg["execution"]["official_test_reads"] == 0
    assert cfg["statistics"]["bootstrap_seed"] == 2026102101


def test_replication_grid_identity_comes_from_explicit_runtime_config() -> None:
    cfg = audit.read_config(audit.ROOT / "configs/research/lightgen/gaussian_recent_plane_density_audit_v1.yaml")
    spec = next(item for item in cfg["endpoints"] if item["key"] == "R51245000")
    checkpoint = {
        "task_id": spec["task_id"], "schema": "lightgen_small_gaussian_replication_checkpoint.v1",
        "model_identity": {
            "encoder": [2, 100, 100], "ols": "gaussian_s12_v1",
            "ols_geometry": {"port_pitch_m": 2e-5, "field_1e_radius_m": 3.5e-6, "transport": "identity"},
            "decoder": [2, 100, 100], "phase_shape": [2, 1, 100, 100],
            "trainable_phase_parameters": 40000, "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        },
    }
    proof = audit._identity_config(spec, checkpoint, cfg)
    assert proof["production_grid"] == 2048
    assert proof["checkpoint_embeds_config"] is False
    assert "runtime_config" in proof["grid_identity_basis"]

    checkpoint["propagation_grid_size"] = 4096
    with pytest.raises(RuntimeError, match="conflicts"):
        audit._identity_config(spec, checkpoint, cfg)


def test_generation_proposal_ids_are_unique_per_class_not_globally() -> None:
    valid_ids = torch.arange(1000).repeat(10)
    valid_labels = torch.arange(10).repeat_interleave(1000)
    assert audit._validate_proposal_population(valid_ids, valid_labels)
    invalid_ids = valid_ids.clone()
    invalid_ids[1] = invalid_ids[0]
    assert not audit._validate_proposal_population(invalid_ids, valid_labels)


def test_npz_readback_supports_string_and_nan_arrays(tmp_path) -> None:
    path = tmp_path / "measurement.npz"
    audit.write_npz(path, identity=np.asarray("fixed"), values=np.asarray([1.0, np.nan]))
    with np.load(path, allow_pickle=False) as saved:
        assert saved["identity"].item() == "fixed"
        assert np.array_equal(saved["values"], np.asarray([1.0, np.nan]), equal_nan=True)


def test_synthesized_plane_uses_persisted_physical_geometry() -> None:
    geometry = {
        "plane_key": "fiber_output_and_decoder_entry", "plane_id": "ols_exit",
        "coordinate_frame": "generator", "shape": [6, 8], "pixel_pitch_m": [2e-6, 3e-6],
        "origin_m": [-6e-6, -12e-6], "z_m": 0.012, "wavelength_m": 532e-9,
    }
    plane = audit._field_plane_from_geometry(torch.nn.Identity(), geometry)
    assert plane.plane_id == "ols_exit"
    assert plane.shape == (6, 8)
    assert plane.pixel_pitch_m == (2e-6, 3e-6)
    assert plane.origin_m == (-6e-6, -12e-6)


def test_camera_equal_power_rescale_changes_readout_but_restores_integrated_power() -> None:
    plane = audit.PlaneSpec("camera", (28, 28), (2e-6, 2e-6), 532e-9, 0.0, "camera")
    field = torch.ones((2, 1, 28, 28), dtype=torch.complex64)
    field[1] *= 0.5
    complex_field = ComplexField(field, plane, OperationProvenance("test", plane.plane_id, None))
    original, _ = audit._reconstruction_metrics(complex_field, torch.zeros((2, 28, 28), dtype=torch.uint8),
                                               np.asarray([0, 1]), 1e9)
    target_power = torch.as_tensor([original[0, 4], original[0, 4]], dtype=torch.float64)
    scale = target_power / torch.as_tensor(original[:, 4], dtype=torch.float64)
    equal, _ = audit._reconstruction_metrics(complex_field, torch.zeros((2, 28, 28), dtype=torch.uint8),
                                             np.asarray([0, 1]), 1e9, scale)
    assert original[1, 4] < original[0, 4]
    assert equal[1, 4] == pytest.approx(original[0, 4], rel=1e-6)
    assert equal[1, 0] != pytest.approx(original[1, 0], rel=1e-6)


def test_class_paired_bootstrap_is_deterministic_and_preserves_zero_effect() -> None:
    labels = np.repeat(np.arange(10), 3)
    first = audit._bootstrap_paired(np.zeros(30), labels, 100, 2026102101)
    second = audit._bootstrap_paired(np.zeros(30), labels, 100, 2026102101)
    assert first == second
    assert first["mean"] == 0.0
    assert first["ci95_low"] == first["ci95_high"] == 0.0


def test_local_direction_recheck_keeps_coefficients_across_exposures(tmp_path, monkeypatch) -> None:
    cfg = audit.read_config(audit.ROOT / "configs/research/lightgen/gaussian_recent_plane_density_audit_v1.yaml")
    cfg["statistics"]["finite_difference_relative_radii"] = [0.01]
    cfg["statistics"]["finite_difference_primary_radius"] = 0.01
    cfg["statistics"]["finite_difference_endpoints"] = [
        "small_step_030000", "small_control_step_035000", "small_treatment_step_035000", "large_step_030000",
    ]

    previous_root = (
        tmp_path / "outputs" / "20261022_gaussian_ols_effective_channel_dual_grid_v1" / "runs" / "attempt_01"
    )
    payload_path = previous_root / "engineering" / "local_direction_repair_inputs.npz"
    payload_path.parent.mkdir(parents=True)
    payload: dict[str, np.ndarray] = {}
    for endpoint in cfg["statistics"]["finite_difference_endpoints"]:
        payload[f"{endpoint}__rows"] = np.arange(20, dtype=np.int64) + 100
        payload[f"{endpoint}__labels"] = np.repeat(np.arange(10, dtype=np.int64), 2)
        coefficients = np.zeros((20, 100), dtype=np.complex128)
        coefficients[:, 0] = 1.0 + 0.25j
        payload[f"{endpoint}__coefficients"] = coefficients
    np.savez(payload_path, **payload)

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    basis_root = previous_root / "bases"
    basis_root.mkdir(parents=True)
    for endpoint in cfg["statistics"]["finite_difference_endpoints"]:
        kind = "small" if endpoint.startswith("small") else "large"
        grids = (2048, 4096) if kind == "small" else (4096, 6144)
        for role, grid in zip(("production", "reference"), grids):
            np.savez(
                basis_root / f"{endpoint}_{role}_{grid}.npz",
                camera_field_per_unit_coefficient=np.zeros((1, 2, 2), dtype=np.complex128),
            )

    jacobian_calls = 0

    def load_basis(_root, _endpoint, _role, _grid, _device):
        return torch.zeros((100, 4), dtype=torch.complex128), 1.0

    def local_jacobian(_z, _basis, _geometry, _camera, _exposure):
        nonlocal jacobian_calls
        jacobian_calls += 1
        return (
            torch.full((4, 200), 0.01, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
        )

    camera = {"exposure_sweep_s": [1e-3, 2e-3], "dark_current_e_per_pixel_s": 0.0}
    monkeypatch.setattr(audit.dual_grid, "load_basis", load_basis)
    monkeypatch.setattr(audit.dual_grid, "read_config", lambda _path: {})
    monkeypatch.setattr(audit.dual_grid, "read_noise_proxy", lambda _cfg: camera)
    monkeypatch.setattr(audit.dual_grid, "_local_jacobian", local_jacobian)
    monkeypatch.setattr(audit.dual_grid, "noise_scale", lambda _camera, _exposure: 1.0)
    monkeypatch.setattr(
        audit.dual_grid, "clipped_camera_moments", lambda rate, _camera: (rate, rate + 1.0)
    )
    monkeypatch.setattr(audit.dual_grid, "_full_well_integer", lambda _camera: 100)
    monkeypatch.setattr(
        audit.dual_grid,
        "_finite_difference_linear_prediction",
        lambda _jacobians, _role, _direction, _radius: torch.zeros(4, dtype=torch.float64),
    )

    receipt = audit._finite_difference_recheck(cfg, tmp_path, torch.device("cpu"))

    assert receipt["row_count"] == 4 * 20 * 2 * 1 * 2
    assert jacobian_calls == 4 * 20 * 2 * 2


def test_report_reads_comparisons_from_verified_population_payload(tmp_path) -> None:
    metric = {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    comparisons = {
        "C45000_to_V45000/production": {
            "plane_metrics": {"camera_physical_roi": {"power_w": metric}},
        },
        "D050000_to_Dgrad50000/production": {
            "native_reconstruction": {"mse": metric},
            "Dgrad_equal_camera_power_vs_D0_native": {"mse": metric},
        },
    }
    report = {
        "finite_difference_recheck": {"row_count": 1920, "primary_radius_non_linear_count": 0},
        "population_tables": {
            "per_image_plane_rows": 1_240_000,
            "unique_keys": 1_240_000,
            "table": "tables/plane_density_per_image.csv.gz",
            "paired_comparisons": comparisons,
        },
        "preflight": {
            "max_same_grid_complex_relative_error": 0.0,
            "max_direct_vs_synthesized_mse_absolute_error": 0.0,
            "max_power_trace_relative_error": 0.0,
            "max_independent_numpy_area_integral_relative_error": 0.0,
        },
    }

    audit._write_report(tmp_path, report)

    output = (tmp_path / "中文诊断报告.md").read_text(encoding="utf-8")
    assert "1,240,000条" in output
    assert "V减C" in output


def test_finalize_saved_measurement_requires_and_consumes_population_and_fd_receipts(tmp_path) -> None:
    cfg = audit.read_config(audit.ROOT / "configs/research/lightgen/gaussian_recent_plane_density_audit_v1.yaml")
    metric = {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    comparisons = {
        "C45000_to_V45000/production": {
            "plane_metrics": {"camera_physical_roi": {"power_w": metric}},
        },
        "D050000_to_Dgrad50000/production": {
            "native_reconstruction": {"mse": metric},
            "Dgrad_equal_camera_power_vs_D0_native": {"mse": metric},
        },
    }
    expected_rows = 1_240_000
    finite_rows = 1920
    finite = {
        "status": "COMPLETED",
        "endpoints": ["small_step_030000", "small_control_step_035000",
                      "small_treatment_step_035000", "large_step_030000"],
        "fixed_images_per_endpoint": 20,
        "exposures_s": [1e-5, 1e-4, 1e-3, 1e-2],
        "relative_radii": [0.001, 0.01, 0.1],
        "primary_radius": 0.01,
        "row_count": finite_rows,
        "expected_row_count": finite_rows,
        "primary_radius_non_linear_count": 0,
    }
    population = {
        "status": "VERIFIED",
        "per_image_plane_rows": expected_rows,
        "expected_per_image_plane_rows": expected_rows,
        "unique_keys": expected_rows,
        "summary_rows": 2112,
        "table": "tables/plane_density_per_image.csv.gz",
        "paired_comparisons": comparisons,
    }
    report = {
        "task_id": audit.TASK_ID,
        "status": "VALID_FROZEN_DIAGNOSTIC_COMPLETE",
        "identity_checks": {"fixed": True},
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "finite_difference_recheck": finite,
        "population_tables": population,
        "generation_population": [{}] * 16,
        "preflight": {
            "max_same_grid_complex_relative_error": 0.0,
            "max_direct_vs_synthesized_mse_absolute_error": 0.0,
            "max_power_trace_relative_error": 0.0,
            "max_independent_numpy_area_integral_relative_error": 0.0,
        },
    }
    preflight_dir = tmp_path / "preflight"
    local_direction_dir = tmp_path / "local_direction_recheck"
    tables_dir = tmp_path / "tables"
    preflight_dir.mkdir()
    local_direction_dir.mkdir()
    tables_dir.mkdir()
    audit.write_json(preflight_dir / "receipt.json", {
        "task_id": audit.TASK_ID, "status": "READY", "identity_checks": {"fixed": True},
    })
    audit.write_json(tmp_path / "machine_report.json", report)
    audit.write_json(local_direction_dir / "receipt.json", finite)
    audit.write_json(tables_dir / "paired_endpoint_comparisons.json", comparisons)
    for name in ("plane_density_per_image.csv.gz", "plane_density_summary.csv",
                 "power_area_log_decomposition.csv"):
        (tables_dir / name).touch()
    with gzip.open(local_direction_dir / "finite_difference_controls.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row"])
        writer.writerows((index,) for index in range(finite_rows))

    completion = audit.finalize_saved_measurement(cfg, tmp_path)

    assert completion["per_image_plane_rows"] == expected_rows
    assert completion["finite_difference_rows"] == finite_rows
    assert completion["actual_training_probe_run_in_this_attempt"] is False
    assert (tmp_path / "中文诊断报告.md").is_file()
    assert json.loads((tmp_path / "measurement_complete.json").read_text(encoding="utf-8")) == completion

