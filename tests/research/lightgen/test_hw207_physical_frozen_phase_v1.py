import numpy as np
import torch

from scripts.research.lightgen import (
    execute_hw207_physical_frozen_phase_v1 as execute,
    hw207_physical_frozen_phase_v1 as model,
)


def test_c0_validation_inventory_preserves_original_rows_and_never_reads_official_test():
    config = model.read_json(model.DEFAULT_CONFIG)
    data = model.load_c0_validation(config)
    assert len(data["indices"]) == 2475
    assert data["class_counts"] == {"0": 592, "1": 674, "2": 596, "3": 613}
    assert np.unique(data["indices"]).size == len(data["indices"])
    assert data["official_test_opened"] is False


def test_candidate_inventory_keeps_seven_original_assets_and_height_separate():
    records = model.candidate_metadata(model.read_json(model.DEFAULT_CONFIG))
    assert len(records) == 7
    assert records[0]["candidate_id"] == "C3.5_P1_seed11"
    processed = next(row for row in records if row["candidate_id"] == "C3.5_P2_process_aware")
    assert processed["original_height"]["representation"] == "height_um"
    assert processed["representation"] == "phase_rad"
    assert processed["shape_rows_columns"] == [100, 100]
    assert all(row["phase_sha256"] for row in records)


def test_native_dmd_command_and_level1_fill_factor_power_are_explicit():
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:18, 11:16] = 255
    native = model.native_dmd_pattern(image, mode="static_binary")
    assert native.shape == (1600, 2560)
    assert set(np.unique(native)) == {0, 255}
    rows, cols = np.where(native != 0)
    assert rows.min() == (1600 - 28) // 2 + 10
    assert cols.min() == (2560 - 28) // 2 + 11

    all_on = torch.full((1, 28, 28), 255, dtype=torch.uint8)
    level0, pitch0 = model.dmd_field(all_on, mode="static_binary", model_level=0, subpixels=16)
    level1, pitch1 = model.dmd_field(all_on, mode="static_binary", model_level=1, subpixels=16)
    energy0 = float(level0.abs().square().sum() * pitch0**2)
    energy1 = float(level1.abs().square().sum() * pitch1**2)
    assert np.isclose(energy1 / energy0, 0.92, rtol=0, atol=1e-6)


def test_fixed_cmos_roi_physical_area_is_exact_and_argmax_abstains_on_ties():
    config = model.read_json(model.DEFAULT_CONFIG)
    weights = model.fixed_sensor_roi_weights(config)
    assert len(weights) == 4
    for item in weights:
        assert np.isclose(item["fractions"].sum() * item["pixel_area_um2"], 2500.0,
                          rtol=1e-10, atol=1e-7)
    assert model.cmos.predict_from_roi_energies([0, 0, 0, 0]) is None
    assert model.cmos.predict_from_roi_energies([1, 2, 2, 0]) is None
    assert model.cmos.predict_from_roi_energies([1, 3, 2, 0]) == 1


def test_cmos_subpixel_irradiance_is_area_averaged_per_physical_pixel():
    subpixels = torch.tensor([[[1.0, 3.0, 5.0, 7.0],
                               [3.0, 5.0, 7.0, 9.0],
                               [9.0, 11.0, 13.0, 15.0],
                               [11.0, 13.0, 15.0, 17.0]]])
    actual = model.area_average_cmos_pixels(subpixels, (2, 2), 2)
    assert torch.equal(actual, torch.tensor([[[3.0, 7.0], [11.0, 15.0]]]))
    try:
        model.area_average_cmos_pixels(subpixels, (2, 2), 1)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("mismatched physical pixel sampling must fail closed")


def test_cmos_affine_handles_axis_swap_flips_and_subpixel_origin_without_moving_rois():
    config = model.read_json(model.DEFAULT_CONFIG)
    config["cmos"]["pixel_to_window_affine"] = {
        "matrix_um_per_pixel": [[0.0, -6.5], [-6.5, 0.0]],
        "offset_um": [6756.3, 6755.75],
        "status": "synthetic test transform",
    }
    matrix, offset = model.cmos_pixel_to_window_affine(config)
    weights = model.fixed_sensor_roi_weights(config)
    assert np.isclose(abs(np.linalg.det(matrix)), 6.5**2, rtol=0, atol=1e-12)
    for item in weights:
        assert np.isclose(item["fractions"].sum() * item["pixel_area_um2"], 2500.0,
                          rtol=1e-10, atol=1e-7)
    center = matrix @ np.asarray([1024.0, 1024.0]) + offset
    assert np.allclose(center, [100.3, 99.75], atol=1e-12)


def test_cmos_calibration_interface_is_identity_without_data_and_uses_only_supplied_maps():
    raw = np.full((2, 2), 100.0)
    unchanged, status = model.apply_cmos_calibration(raw)
    assert np.array_equal(unchanged, raw)
    assert status["status"] == "UNCALIBRATED_IDENTITY"
    calibrated, status = model.apply_cmos_calibration(
        raw, dark_frame=np.full((2, 2), 10.0), exposure_s=2.0, dark_exposure_s=1.0,
        flat_field=np.asarray([[1.0, 2.0], [1.0, 2.0]]), gain_e_per_adu=2.0,
    )
    assert np.allclose(calibrated, [[120.0, 60.0], [120.0, 60.0]])
    assert status["status"] == "CALIBRATED_WITH_EXPLICIT_INPUTS"
    assert status["absolute_photon_calibration"] is False


def test_independent_direct_fresnel_reference_and_wrong_sign_counterexample():
    result = execute._preflight_small_reference(torch.device("cpu"))
    assert result["status"] == "PASS"
    assert result["direct_fresnel_relative_l2"] <= result["direct_reference_tolerance"]
    assert result["wrong_sign_counterexample_rejected"] is True
    assert result["axis_swap_relative_l2"] <= result["tolerances"]["axis_swap_relative_l2"]
    assert result["analytic_tilted_gaussian_relative_l2"] <= result["tolerances"]["analytic_gaussian_relative_l2"]
    assert result["fp32_vs_fp64_complex_field_relative_l2"] <= result["tolerances"]["fp32_vs_fp64_relative_l2"]
    assert result["fp32_vs_fp64_energy_relative_difference"] <= result["tolerances"]["fp32_vs_fp64_energy_relative_difference"]


def test_completed_partial_sampling_allows_only_labeled_diagnostic_consumers():
    assert execute._sampling_admission(
        {"status": "PREFLIGHT_READY", "acceptance_status": "P2_PARTIAL_PASS"}, "test"
    ) is True
    assert execute._sampling_admission(
        {"status": "PREFLIGHT_READY", "acceptance_status": "P2_PASS"}, "test"
    ) is False
    for p2 in (
        {"status": "PREFLIGHT_FAILED", "acceptance_status": "P2_PARTIAL_PASS"},
        {"status": "PREFLIGHT_READY", "acceptance_status": "P2_INVALID_HARNESS"},
    ):
        try:
            execute._sampling_admission(p2, "test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed or invalid P2 evidence must block candidate consumers")


def test_photo_estimate_abcd_does_not_claim_conjugacy_or_one_to_one():
    config = model.read_json(model.DEFAULT_CONFIG)
    result = model.focal_geometry_summary(config)
    assert np.isclose(result["input_relay_B_mm"], -80.0, atol=1e-10)
    photo = result["output_scenarios"]["PHOTO_ESTIMATE_AND_C37_C38_REFERENCE"]
    ideal = result["output_scenarios"]["IDEAL_1X_CONJUGATE"]
    assert photo["is_conjugate"] is False
    assert np.isclose(photo["B_mm"], -86.66666666666667, atol=1e-10)
    assert ideal["is_conjugate"] is True
    assert ideal["signed_magnification_if_conjugate"] == -1.0


def test_small_grid_production_forward_reaches_cmos_roi_consumer():
    config = model.read_json(model.DEFAULT_CONFIG)
    data = model.load_c0_validation(config)
    row = int(np.flatnonzero(data["labels"] == 0)[0])
    image = torch.as_tensor(data["images"][row:row + 1], dtype=torch.uint8)
    phase = np.zeros((100, 100), dtype=np.float32)
    scenario = config["output_relay"]["scenarios"]["PHOTO_ESTIMATE_AND_C37_C38_REFERENCE"]
    result = model.forward_field(image, phase, config, scenario, device="cpu",
                                 model_level=0, mode="static_binary",
                                 input_lens_grid=32, output_lens_grid=32, plan_cache={})
    assert result["cmos_intensity"].shape == (1, 2048, 2048)
    intensity = result["cmos_intensity"][0].detach().numpy()
    measure = model.measure_cmos(intensity, 0,
                                 float(result["dmd_input_energy"][0]),
                                 float(result["phase_plane_energy"][0]),
                                 config, model.fixed_sensor_roi_weights(config))
    assert np.isfinite(intensity).all()
    assert len(measure["roi_scores"]) == 4
    assert "dmd_to_cmos_efficiency" in measure
    assert "phase_to_cmos_efficiency" in measure
