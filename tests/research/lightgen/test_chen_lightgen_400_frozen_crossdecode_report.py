from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import diagnose_chen_lightgen_400_frozen_crossdecode_v1 as crossdecode


def test_report_writer_reads_gate_decision_from_analysis(tmp_path) -> None:
    combos = {
        combo: {
            "g0_metrics": {
                "fit": {"mse400_unclipped_mean": 0.4},
                "confirmation": {
                    "mse400_unclipped_mean": 0.5,
                    "ssim28_mean": 0.6,
                    "saturation_fraction": 0.0,
                },
            }
        }
        for combo in ("R00", "R01", "R10", "R11")
    }
    gate = {
        combo: {"mse_improvement_fraction_vs_R00": 0.0, "passed": False}
        for combo in ("R01", "R10", "R11")
    }
    report = {
        "status": "VALID_COMPLETED",
        "decision": "NO_FIXED_COMBINATION_PASSES",
        "common_gain_w_inverse": 3.0,
        "preflight_status": "READY",
        "formal_cross_decoder_forwards": 0,
        "source_cached_cross_decoder_forwards": 600,
        "window_checks": [{} for _ in range(20)],
        "analysis": {
            "combos": combos,
            "local_gates_vs_R00": gate,
            "passed_combinations": [],
        },
    }
    path = tmp_path / "results_analysis.md"

    crossdecode._write_report(path, report)

    content = path.read_text(encoding="utf-8")
    assert "统一读出下达到预设局部门槛的组合：无" in content
    assert "VALID_COMPLETED" in content


def _curve_row(population: str, count: int, gain_name: str, combo: str) -> dict[str, object]:
    offset = 0.02 if combo in {"R10", "R11"} else 0.0
    return {
        "population": population,
        "sample_count": count,
        "gain_name": gain_name,
        "gain_w_inverse": 1.0,
        "combo": combo,
        "mse400_unclipped_mean": 1.0 - offset,
        "ssim28_mean": 0.8 + offset,
        "saturation_fraction": 0.005,
    }


def test_gain_curve_consumer_rejects_mixed_population_as_confirmation() -> None:
    gains = {"R00_shared_constrained": 1.0, "curve_00_x0.250": 0.25}
    mixed_only = [
        _curve_row("all300", 300, gain_name, combo)
        for gain_name in gains
        for combo in crossdecode.COMBOS
    ]

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        crossdecode._finalize_gain_curve_records(mixed_only, gains)

    complete = [
        _curve_row(population, count, gain_name, combo)
        for population, count in (("fit100", 100), ("confirmation200", 200), ("all300", 300))
        for gain_name in gains
        for combo in crossdecode.COMBOS
    ]
    finalized = crossdecode._finalize_gain_curve_records(complete, gains)
    confirmation_r10 = next(
        row for row in finalized
        if row["population"] == "confirmation200"
        and row["gain_name"] == "curve_00_x0.250"
        and row["combo"] == "R10"
    )
    assert confirmation_r10["mse_improvement_vs_same_gain_R00"] == pytest.approx(0.02)
    assert confirmation_r10["ssim_delta_vs_same_gain_R00"] == pytest.approx(0.02)
    assert confirmation_r10["same_gain_gate_shaped"] is False


def test_cross_power_reader_replaces_historical_diagonal_decoder_metadata(tmp_path) -> None:
    rows = torch.tensor([0], dtype=torch.long)
    target = torch.zeros((1, 1, 400, 400), dtype=torch.float32)
    coefficients = torch.ones((1, 100), dtype=torch.complex64)
    modal = torch.ones((1, 100), dtype=torch.float64) * 1.0e-9
    encoder_planes = torch.arange(1, 8, dtype=torch.float64)[None, :]
    decoder_planes = torch.tensor([[11.0, 12.0, 13.0, 14.0, 15.0]], dtype=torch.float64)
    source_state = {
        "rows": rows,
        "labels": torch.tensor([7]),
        "target": target,
        "ols_coefficients": coefficients,
        "physical_w_per_pixel": target.clone(),
        "modal_power_w": modal,
        "slm_after_modulation_power_w": torch.tensor([1.0e-4], dtype=torch.float64),
        "ols_incident_power_w": torch.tensor([2.0e-5], dtype=torch.float64),
        "ols_modal_sum_w": modal.sum(1),
        "camera_roi_power_w": torch.tensor([22.0], dtype=torch.float64),
        "seven_plane_power_w": encoder_planes,
    }
    payload = {
        "schema": "chen_lightgen_400_frozen_crossdecode_block.v1",
        "combo": "R01",
        "encoder": "E0",
        "decoder": "D1",
        "grid": crossdecode.GRID,
        "rows": rows,
        "labels": source_state["labels"],
        "target": target,
        "coefficients": coefficients,
        "physical_w_per_pixel": torch.ones_like(target),
        "modal_power_w": modal,
        "slm_after_modulation_power_w": source_state["slm_after_modulation_power_w"],
        "ols_incident_power_w": source_state["ols_incident_power_w"],
        "ols_modal_sum_w": source_state["ols_modal_sum_w"],
        # This reproduces the historical bug: use source E0D0 camera/seven-plane metadata.
        "camera_roi_power_w": source_state["camera_roi_power_w"],
        "seven_plane_power_w": encoder_planes,
        "decoder_plane_power_w": decoder_planes,
        "forward_calls": 1,
    }
    destination = tmp_path / "R01"
    destination.mkdir()
    torch.save(payload, destination / "block_000_001.pt")
    loaded = {
        "population": {"all_rows": rows},
        "labels": torch.tensor([7]),
        "states": {crossdecode.STATE0: source_state},
    }

    measured = crossdecode._read_cross_combo("R01", loaded, tmp_path)

    assert measured["camera_roi_power_w"].item() == pytest.approx(160000.0)
    assert torch.equal(measured["seven_plane_power_w"], torch.tensor([[1, 2, 11, 12, 13, 14, 15]], dtype=torch.float64))
    assert loaded["power_overlay"][0]["row"] == 0
    assert loaded["power_overlay"][0]["camera_power_w_original"] == pytest.approx(22.0)

    payload["camera_roi_power_w"] = torch.tensor([23.0], dtype=torch.float64)
    torch.save(payload, destination / "block_000_001.pt")
    with pytest.raises(RuntimeError, match="camera power"):
        crossdecode._validate_cross_block(destination / "block_000_001.pt", "R01", loaded)


def test_symmetric_ssim_decomposition_matches_same_brightness_and_blur_cases() -> None:
    target = torch.rand((1, 1, 28, 28), dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    cases = {
        "same": target,
        "brightness": target * 0.8,
        "blur": torch.nn.functional.avg_pool2d(target, 3, stride=1, padding=1),
    }
    for image in cases.values():
        l0, c0, _ = crossdecode._ssim_parts(target, target)
        l1, c1, _ = crossdecode._ssim_parts(image, target)
        result = crossdecode._ssim_symmetric_decomposition(l0, c0, l1, c1)
        assert result["absolute_error"] <= 1.0e-9
        assert result["delta_ssim"] == pytest.approx(float((l1 * c1 - l0 * c0).mean()), abs=1.0e-12)
