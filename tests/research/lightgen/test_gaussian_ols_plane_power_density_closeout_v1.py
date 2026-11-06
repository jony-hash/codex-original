from __future__ import annotations

import math
import json
import csv

import numpy as np
import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.lightgen.plane_power_density import (
    field_intensity_distribution,
)
from scripts.research.lightgen.closeout_gaussian_ols_plane_power_density_v1 import (
    ChenLightGen400Baseline,
    LARGE_PLANES,
    LightGen400Config,
    SMALL_PLANES,
    _large_model,
    create_power_chain_figures,
    field_plane_metrics,
    mark_remote_sources_restored,
    validate_unique_keys,
    validate_density_core,
)


def test_complex64_area_integral_matches_independent_float64_component_square() -> None:
    plane = PlaneSpec("complex64-integral", (2, 2), (2.0e-6, 3.0e-6), 532.0e-9, 0.0, "test")
    values = torch.tensor([1 + 1j, 3 + 4j, 5 + 12j, 7 + 24j], dtype=torch.complex64).reshape(1, 1, 2, 2)
    field = ComplexField(values, plane, OperationProvenance("test", plane.plane_id, None))
    measured, _ = field_plane_metrics(field)
    components = values.numpy().astype(np.complex128)
    expected = float((components.real ** 2 + components.imag ** 2).sum() * 2.0e-6 * 3.0e-6)
    assert measured[0, 0] == pytest.approx(expected, rel=1.0e-12, abs=0.0)


def _field(values: torch.Tensor) -> ComplexField:
    plane = PlaneSpec("test_plane", (values.shape[-2], values.shape[-1]), (2.0e-6, 3.0e-6), 532.0e-9, 0.0, "test")
    return ComplexField(values, plane, OperationProvenance("test", plane.plane_id, None))


def test_uniform_intensity_has_equal_geometric_and_effective_area() -> None:
    values = torch.full((1, 1, 8, 10), 2.0 + 0.0j, dtype=torch.complex128)
    metrics, _ = field_plane_metrics(_field(values))
    assert metrics[0, 2] == pytest.approx(metrics[0, 1])
    assert metrics[0, 3] == pytest.approx(metrics[0, 4])
    assert metrics[0, 3] == pytest.approx(4.0)
    assert metrics[0, 3] / 1.0e4 == pytest.approx(4.0e-4)


def test_power_scaling_changes_density_but_not_effective_area() -> None:
    values = torch.ones((1, 1, 8, 10), dtype=torch.complex128)
    before, _ = field_plane_metrics(_field(values))
    after, _ = field_plane_metrics(_field(values * math.sqrt(2.0)))
    assert after[0, 0] == pytest.approx(2.0 * before[0, 0])
    assert after[0, 2] == pytest.approx(before[0, 2])
    assert after[0, 3] == pytest.approx(2.0 * before[0, 3])
    assert after[0, 4] == pytest.approx(2.0 * before[0, 4])


def test_zero_power_does_not_fabricate_effective_area_or_density() -> None:
    metrics, _ = field_plane_metrics(_field(torch.zeros((1, 1, 8, 10), dtype=torch.complex128)))
    assert metrics[0, 0] == 0.0
    assert metrics[0, 3] == 0.0
    assert math.isnan(metrics[0, 2])
    assert math.isnan(metrics[0, 4])


def test_coherent_plane_integral_keeps_interference_cross_term() -> None:
    mode = torch.ones((1, 1, 8, 10), dtype=torch.complex128)
    constructive, _ = field_plane_metrics(_field(mode + mode))
    one_mode, _ = field_plane_metrics(_field(mode))
    destructive, _ = field_plane_metrics(_field(mode - mode))
    assert constructive[0, 0] == pytest.approx(4.0 * one_mode[0, 0])
    assert destructive[0, 0] == 0.0


def test_intensity_quantiles_include_area_and_power_weighted_views() -> None:
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.complex128)
    result = field_intensity_distribution(_field(values))[0]
    assert result["area_weighted_q50_w_m2"] == pytest.approx(4.0)
    assert result["power_weighted_q50_w_m2"] == pytest.approx(16.0)
    assert result["maximum_intensity_w_m2"] == pytest.approx(16.0)


def test_zero_power_intensity_power_weighted_quantiles_are_undefined() -> None:
    result = field_intensity_distribution(
        _field(torch.zeros((1, 1, 8, 10), dtype=torch.complex128))
    )[0]
    assert result["area_weighted_q99_w_m2"] == 0.0
    assert result["power_weighted_q99_w_m2"] is None


def test_export_key_consumer_rejects_missing_and_duplicate_rows() -> None:
    expected = {("ep", 0), ("ep", 1)}
    validate_unique_keys([("ep", 0), ("ep", 1)], expected, "valid table")
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_unique_keys([("ep", 0), ("ep", 0)], expected, "bad duplicate table")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_unique_keys([("ep", 0)], expected, "bad missing table")


def test_independent_density_and_coherent_interference_controls() -> None:
    result = validate_density_core()
    assert result["uniform_area_error"] == pytest.approx(0.0)
    assert result["independent_numpy_power_relative_error"] == pytest.approx(0.0)
    assert result["independent_numpy_effective_area_relative_error"] == pytest.approx(0.0)
    assert result["coherent_constructive_to_incoherent_ratio"] == pytest.approx(2.0)


def test_final_status_consumes_the_current_remote_restoration_receipt(monkeypatch, tmp_path) -> None:
    from scripts.research.lightgen import closeout_gaussian_ols_plane_power_density_v1 as closeout

    root = tmp_path
    engineering = root / "engineering"
    engineering.mkdir()
    monkeypatch.setattr(closeout, "_output_root", lambda _cfg: root)
    with pytest.raises(RuntimeError, match="receipt is missing"):
        mark_remote_sources_restored({})

    receipt = {
        "status": "RESTORED_TO_ORIGINAL_ABSENT_STATE",
        "all_staged_paths_absent": True,
        "staged_path_count": 12,
        "originally_absent_path_count": 12,
        "paths": [{"originally_existed": False, "exists_after_restore": False} for _ in range(12)],
    }
    (engineering / "remote_source_restoration_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    pending = {
        "status": "CLOSEOUT_ARTIFACTS_FINALIZED_RESTORATION_PENDING",
        "remote_source_restoration": "RESTORATION_PENDING",
        "repaired_export": {
            "unique_power_summary_rows": 4400,
            "unique_noise_rows": 3200,
            "raw_ablation_rows_by_grid": {"production": 3060000, "reference": 3060000},
        },
        "plane_density": {"per_image_rows": 604200, "readback_rows": 604200, "coverage": [{} for _ in range(14)]},
        "fiber_modal_field_audit": {
            "status": "FIBER_MODAL_FIELD_POWER_AUDIT_COMPLETE",
            "small_rows": 60000,
            "large_rows": 600,
            "total_rows": 60600,
            "summary": [{} for _ in range(14)],
        },
        "representative_plane_intensity_figures": ["map.png" for _ in range(14)],
        "power_chain_figures": ["chain.png" for _ in range(3)],
        "local_direction": {
            "complex128_recompute": "VALIDATED_FLOAT64_LOCAL_DIRECTION_RECOMPUTE",
            "local_direction_count_rows": 28800,
        },
    }
    for name in ("measurement_complete.json", "machine_report.json"):
        (root / name).write_text(json.dumps(pending), encoding="utf-8")
    (root / "中文收尾报告.md").write_text("临时远端源文件恢复状态待核验。", encoding="utf-8")

    result = mark_remote_sources_restored({})
    assert result["status"] == "CLOSEOUT_COMPLETION_RECEIPT_CONSUMED"
    completed = json.loads((root / "measurement_complete.json").read_text(encoding="utf-8"))
    assert completed["status"] == "DUAL_GRID_CONDITIONAL_DIAGNOSTIC_AND_PLANE_DENSITY_COMPLETE_REAL_NOISE_CALIBRATION_PENDING"
    assert completed["remote_source_restoration"] == "RESTORED_TO_ORIGINAL_ABSENT_STATE"


def test_large_decoder_recomputes_only_grid_sampled_mode_buffers() -> None:
    source = ChenLightGen400Baseline(LightGen400Config(2048, True))
    source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}
    checkpoint = {"model": source_state}

    target = _large_model(checkpoint, 4096, torch.device("cpu"))
    assert target.ols.modes_y.shape == (10, 4096)
    assert target.ols.modes_x.shape == (10, 4096)
    assert torch.equal(target.ols.centers_m.cpu(), source_state["ols.centers_m"])
    assert torch.equal(target.ols.transport.cpu(), source_state["ols.transport"])
    assert all(torch.equal(target.state_dict()[key].cpu(), value) for key, value in source_state.items()
               if key not in {"ols.modes_y", "ols.modes_x"})

    invalid = {key: value.clone() for key, value in source_state.items()}
    invalid["encoder_phase"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected grid-dependent state"):
        _large_model({"model": invalid}, 4096, torch.device("cpu"))


def test_power_chain_figures_cover_all_small_states_and_both_large_grids(tmp_path) -> None:
    root = tmp_path
    output = root / "power_density"
    output.mkdir()
    states = ("step_000000", "step_005000", "step_010000", "step_020000", "step_030000",
              "control_step_032500", "control_step_035000", "treatment_step_032500", "treatment_step_035000")
    metrics = ("power_w", "mean_intensity_geometric_w_m2", "mean_intensity_effective_w_m2", "effective_area_m2")
    rows = []
    for state in states:
        for plane in SMALL_PLANES:
            for metric in metrics:
                rows.append((f"small_{state}", "production", plane, "all", metric, "2.0"))
    for role in ("decoder_production", "decoder_reference"):
        for plane in LARGE_PLANES:
            for metric in metrics:
                rows.append(("large_step_030000", role, plane, "all", metric, "2.0"))
    path = output / "per_class_plane_power_density_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint", "grid_role", "plane_key", "class", "metric", "mean"))
        writer.writerows(rows)

    figures = create_power_chain_figures(root)
    assert len(figures) == 3
    assert all((root / figure).is_file() for figure in figures)
