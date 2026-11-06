import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/research/lightgen"))
import hw207_mnist4_c2_v1 as c2
import hw207_mnist4_c21_c3_spatial_v1 as c21
from ldm_project.v2.application.research.lightgen import hw207_spatial_geometry as geo


def _active_length_inside_target(pitch_um: float, target_width_um: float, fill_factor: float) -> float:
    half_active = pitch_um * math.sqrt(fill_factor) / 2
    half_target = target_width_um / 2
    centers = [(column - (geo.DMD_COLUMNS - 1) / 2) * pitch_um
               for column in range(geo.DMD_COLUMNS)]
    return sum(max(0.0, min(center + half_active, half_target)
                   - max(center - half_active, -half_target)) for center in centers)


def test_dmd_zero_degree_grid_coordinates():
    x0, y0 = geo.dmd_mirror_center(0, 0)
    x1, y1 = geo.dmd_mirror_center(1, 0)
    x2, y2 = geo.dmd_mirror_center(0, 1)
    assert math.isclose(x1 - x0, geo.DMD_PITCH_UM, abs_tol=1e-12)
    assert math.isclose(y2 - y0, -geo.DMD_PITCH_UM, abs_tol=1e-12)
    assert math.isclose(x0, -(geo.DMD_COLUMNS - 1) * geo.DMD_PITCH_UM / 2)
    assert math.isclose(y0, (geo.DMD_ROWS - 1) * geo.DMD_PITCH_UM / 2)


def test_dmd_45_degree_rotation_coordinates():
    matrix = geo.scaled_rotation(1.0, 45.0)
    x, y, _ = geo._matrix_and_offset(matrix, (0.0, 0.0))
    px, py = geo._transform(geo.DMD_PITCH_UM, 0.0, x, y)
    assert math.isclose(px, geo.DMD_PITCH_UM / math.sqrt(2), abs_tol=1e-12)
    assert math.isclose(py, geo.DMD_PITCH_UM / math.sqrt(2), abs_tol=1e-12)
    bounds = geo.dmd_array_bounds_um(45.0)
    expected_w = (geo.DMD_COLUMNS + geo.DMD_ROWS) * geo.DMD_PITCH_UM / math.sqrt(2)
    assert math.isclose(bounds["width_um"], expected_w, rel_tol=1e-12)


def test_dmd_boundary_crop_counts_full_and_partial_cells():
    affine = geo.scaled_rotation(-1.0, 0.0)
    lattice = geo.local_mirror_window(affine)
    counts = geo.dmd_coverage_counts(lattice, affine, fill_factor=0.92)
    assert counts["intersecting_mirrors"] == counts["fully_inside_mirrors"] + counts["partially_clipped_mirrors"]
    assert counts["partially_clipped_mirrors"] > 0
    assert counts["active_intersecting_mirrors"] == (
        counts["fully_inside_active_mirrors"] + counts["partially_clipped_active_mirrors"]
    )
    assert 0 < counts["active_area_coverage_fraction"] < 1


def test_mirror_active_area_uses_explicit_fill_factor():
    pitch, fill = 7.56, 0.92
    active_width = pitch * math.sqrt(fill)
    assert math.isclose(active_width**2, pitch**2 * fill, rel_tol=1e-15)
    affine = geo.scaled_rotation(1.0, 0.0)
    lattice = geo.local_mirror_window(affine, width_um=20.0)
    counts = geo.dmd_coverage_counts(lattice, affine, target_width_um=20.0, fill_factor=fill)
    assert counts["active_area_coverage_fraction"] <= counts["full_pitch_area_coverage_fraction"]


def test_historical_26_28_mapping_is_compared_in_physical_units():
    old_26 = c2.geometry(26, 1.0)
    matched_28_magnification = 200.0 / (28 * geo.DMD_PITCH_UM)
    old_28 = c2.geometry(28, matched_28_magnification)
    new_a_28_span = 28 * geo.DMD_PITCH_UM
    assert math.isclose(old_26["array_width_um"], 196.56, abs_tol=1e-12)
    assert math.isclose(old_28["array_width_um"], 200.0, abs_tol=1e-12)
    assert math.isclose(new_a_28_span, 211.68, abs_tol=1e-12)
    assert not math.isclose(matched_28_magnification, 1.0)


def test_nearest_and_area_source_sampling_have_distinct_binary_case():
    affine = geo.scaled_rotation(-0.2, 0.0)
    nearest, _ = geo.build_source_sampling_matrix(affine, method="nearest")
    area, _ = geo.build_source_sampling_matrix(affine, method="area")
    image = torch.zeros((1, 28, 28), dtype=torch.float32)
    image[0, 9:19, 13] = 1
    nearest_commands = torch.sparse.mm(nearest, image.reshape(1, -1).double().T).T >= 0.5
    area_commands = torch.sparse.mm(area, image.reshape(1, -1).double().T).T >= 0.5
    assert nearest_commands.shape == area_commands.shape
    assert torch.any(nearest_commands != area_commands)


def test_physical_area_integration_matches_independent_axis_interval_sum():
    affine = geo.scaled_rotation(-1.0, 0.0)
    lattice = geo.local_mirror_window(affine)
    fill = 0.92
    area = geo.build_target_area_matrix(lattice, affine, fill_factor=fill)
    all_on = torch.ones((area.shape[1], 1), dtype=torch.float64)
    integrated_target_area = float(torch.sparse.mm(area, all_on).sum()) * geo.TARGET_PIXEL_UM**2
    one_axis = _active_length_inside_target(geo.DMD_PITCH_UM, geo.TARGET_WIDTH_UM, fill)
    expected = one_axis**2
    assert math.isclose(integrated_target_area, expected, rel_tol=1e-10, abs_tol=1e-8)


def test_ideal_4f_relay_has_expected_magnification_and_conjugate_distance():
    result = geo.ideal_4f_segments(150.0, 30.0)
    assert math.isclose(result["signed_magnification"], -0.2)
    assert result["object_to_H1_mm"] == 150.0
    assert result["H1_to_H2_mm"] == 180.0
    assert result["H2_to_image_mm"] == 30.0
    assert result["total_length_mm"] == 360.0
    assert abs(result["conjugate_B_mm"]) < 1e-10


def test_photo_estimate_distance_mismatch_is_not_ideal_conjugacy():
    matrix = geo.ray_matrix(230.0, 350.0, 150.0, 150.0, 150.0)
    assert math.isclose(matrix[0][1], -80.0, abs_tol=1e-10)
    assert abs(matrix[0][1]) > 0


def test_cmos_pixel_area_integral_conserves_uniform_field_energy():
    camera_map = geo.scaled_rotation(2.0, 0.0)
    area, info = geo.build_cmos_area_matrix(camera_map, sensor_shape=(128, 128))
    result = geo.integrate_cmos_pixels(torch.ones((100, 100)), area)
    assert torch.isclose(result.sum(), torch.tensor(40000.0, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(info["source_area_capture_fraction"], torch.ones((100, 100), dtype=torch.float64))


def test_equal_quadrant_roi_scores_and_four_class_metrics():
    image = torch.zeros((8, 8), dtype=torch.float64)
    image[:4, :4] = 1
    image[:4, 4:] = 2
    image[4:, :4] = 3
    image[4:, 4:] = 4
    rois = geo.equal_quadrant_rois(image.shape)
    result = geo.roi_scores(image, rois)
    assert result["scores"].tolist() == [16.0, 32.0, 48.0, 64.0]
    assert result["prediction"] == 3
    assert result["outside_energy"] == 0
    metrics = geo.roi_classification_metrics(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
    assert metrics["confusion_4x5"] == [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                        [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]


def test_roi_outside_energy_is_reported_separately():
    image = torch.zeros((4, 4), dtype=torch.float64)
    image[0, 0] = 7
    result = geo.roi_scores(image, geo.equal_quadrant_rois(image.shape))
    assert result["outside_energy"] == 0
    off_roi = torch.zeros((8, 8), dtype=torch.float64)
    off_roi[3, 3] = 7
    rectangles = ((0, 2, 0, 2), (0, 2, 6, 8), (6, 8, 0, 2), (6, 8, 6, 8))
    result = geo.roi_scores(off_roi, rectangles)
    assert result["outside_energy"] == 7
    assert result["prediction"] == -1


def test_all_zero_roi_image_abstains():
    image = torch.zeros((8, 8), dtype=torch.float32)
    result = geo.roi_scores(image, geo.equal_quadrant_rois(image.shape))
    assert result["prediction"] == -1
    assert result["scores"].sum() == 0


def test_tied_roi_scores_abstain():
    image = torch.ones((8, 8), dtype=torch.float32)
    result = geo.roi_scores(image, geo.equal_quadrant_rois(image.shape))
    assert result["prediction"] == -1


def _energy_ledger_for_transform(magnification: float, offset_um=(0.0, 0.0)):
    affine = geo.scaled_rotation(magnification, 0.0)
    source, lattice = geo.build_source_sampling_matrix(
        affine, offset_um=offset_um, method="area")
    target = geo.build_target_area_matrix(
        lattice, affine, offset_um=offset_um, fill_factor=0.92)
    result = geo.map_binary_images(torch.ones((1, 28, 28)), source, target, affine)
    ledger = geo.target_energy_account(result["commands"], result["irradiance"], 0.92, affine)
    return ledger


def test_binary_mapping_energy_ledger_uses_one_reference_plane():
    for magnification in (-1.0, -0.2, -2.0):
        ledger = _energy_ledger_for_transform(magnification)
        assert torch.allclose(ledger["target_energy_relative_um2"]
                              + ledger["outside_energy_relative_um2"],
                              ledger["on_active_area_um2"], atol=1e-7, rtol=1e-7)
        determinant = magnification**2
        assert torch.allclose(ledger["projected_on_active_area_um2"],
                              ledger["on_active_area_um2"] * determinant,
                              atol=1e-7, rtol=1e-7)
        assert 0 <= float(ledger["target_capture_fraction"][0]) <= 1


def test_binary_mapping_energy_ledger_reports_nonunit_relay_crop():
    ledger = _energy_ledger_for_transform(-0.2, (80.0, 0.0))
    assert torch.allclose(ledger["target_energy_relative_um2"]
                          + ledger["outside_energy_relative_um2"],
                          ledger["on_active_area_um2"], atol=1e-7, rtol=1e-7)
    assert 0 < float(ledger["target_capture_fraction"][0]) < 1


def test_production_mapper_fp32_and_fp64_inputs_agree():
    affine = geo.scaled_rotation(-1.0, 0.0)
    lattice = geo.local_mirror_window(affine)
    source, _ = geo.build_source_sampling_matrix(affine, method="nearest")
    target = geo.build_target_area_matrix(lattice, affine, fill_factor=0.92)
    image = torch.zeros((1, 28, 28), dtype=torch.float32)
    image[0, 5:23, 12:16] = 1
    fp32 = geo.map_binary_images(image, source, target, affine)
    fp64 = geo.map_binary_images(image.double(), source, target, affine)
    assert torch.equal(fp32["commands"], fp64["commands"])
    assert torch.allclose(fp32["relative_pattern"], fp64["relative_pattern"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(fp32["irradiance"], fp64["irradiance"], atol=1e-7, rtol=1e-7)


def test_relay_pupil_grating_periods_and_passband_are_reported_correctly():
    candidates = [
        {"id": "C3-A", "input_magnification_signed": -1.0},
        {"id": "C3-B", "input_magnification_signed": -0.2},
        {"id": "C3-C", "input_magnification_signed": -0.26666666666666666},
    ]
    config = {
        "relay_diagnostic_scenarios": {
            "pupil_cutoff_cycles_per_um": [0.05, 0.1, 0.2],
            "defocus_um": [0.0],
            "dmd_grating_periods_in_mirrors": [1, 2, 4],
        },
        "dmd": {"fill_factor": 0.92},
        "lightgen_target": {"wavelength_nm": 532},
    }
    result = c21._grating_sensitivity(candidates, config, torch.device("cpu"))
    assert result["status"] == "UNPHYSICAL_DIAGNOSTIC_ONLY"
    assert len(result["cases"]) == 27
    cases = {(item["candidate"], item["command_period_mirrors"], item["pupil_cutoff_cycles_per_um"]): item
             for item in result["cases"]}
    for candidate in candidates:
        pitch = abs(candidate["input_magnification_signed"]) * geo.DMD_PITCH_UM
        for period_mirrors in (1, 2, 4):
            item = cases[(candidate["id"], period_mirrors, 0.2)]
            assert math.isclose(item["spatial_period_um"], period_mirrors * pitch)
            assert math.isclose(item["fundamental_cycles_per_um"], 1 / (period_mirrors * pitch))
            assert item["fundamental_semantics"] == "resultant field supercell, including active-mirror aperture"
            assert math.isfinite(item["modulation_depth"])
            assert item["model_status"] == "UNPHYSICAL_DIAGNOSTIC_ONLY"
    assert cases[("C3-A", 1, 0.2)]["command_profile"] == "all_on"
    assert cases[("C3-A", 1, 0.2)]["command_off_mirrors_per_period"] == 0
    assert cases[("C3-A", 1, 0.2)]["modulation_origin"] == "finite active-mirror aperture lattice"
    assert cases[("C3-A", 2, 0.2)]["command_profile"] == "1-on/1-off"
    assert cases[("C3-A", 4, 0.2)]["command_profile"] == "2-on/2-off"
    # The production pupil must suppress only spatial components outside its
    # cutoff: C3-A's 1/2/4-pitch frequencies are 0.132/0.066/0.033 cyc/um;
    # C3-B/C3-C need the 4-pitch component to enter the 0.2 cyc/um passband.
    assert cases[("C3-A", 1, 0.05)]["modulation_depth"] == 0.0
    assert cases[("C3-A", 1, 0.1)]["modulation_depth"] == 0.0
    assert cases[("C3-A", 1, 0.2)]["modulation_depth"] > 0.0
    assert cases[("C3-A", 2, 0.05)]["modulation_depth"] == 0.0
    assert cases[("C3-A", 2, 0.1)]["modulation_depth"] > 0.0
    assert cases[("C3-A", 2, 0.2)]["modulation_depth"] > 0.0
    assert cases[("C3-A", 4, 0.05)]["modulation_depth"] > 0.0
    assert cases[("C3-A", 4, 0.1)]["modulation_depth"] > 0.0
    assert cases[("C3-B", 1, 0.2)]["modulation_depth"] == 0.0
    assert cases[("C3-B", 2, 0.2)]["modulation_depth"] == 0.0
    assert cases[("C3-B", 4, 0.1)]["modulation_depth"] == 0.0
    assert cases[("C3-B", 4, 0.2)]["modulation_depth"] > 0.0
    assert cases[("C3-C", 1, 0.2)]["modulation_depth"] == 0.0
    assert cases[("C3-C", 2, 0.2)]["modulation_depth"] == 0.0
    assert cases[("C3-C", 4, 0.1)]["modulation_depth"] == 0.0
    assert cases[("C3-C", 4, 0.2)]["modulation_depth"] > 0.0


def test_output_cmos_sampling_audit_uses_relay_area_integral_and_fixed_roi_consumer():
    candidates = [
        {"id": "C3-A", "output_magnification_signed": -1.0},
        {"id": "C3-B", "output_magnification_signed": -5.0},
        {"id": "C3-C", "output_magnification_signed": -200.0 / 30.0},
    ]
    config = {
        "lightgen_target": {"rows": 100, "columns": 100, "neuron_pitch_um": 2.0},
        "cmos": {"rows": 2048, "columns": 2048, "pixel_um": 6.5},
        "coordinate_model": {
            "output_rotation_deg_diagnostic": 0.0,
            "output_offset_um_diagnostic": [0.0, 0.0],
            "output_transform_status": "TEST_ONLY_CENTERED_AFFINE",
        },
    }

    audit = c21._output_cmos_sampling_audit(candidates, config)

    assert audit["status"] == "COMPONENT_DIAGNOSTIC_ONLY_NO_LIGHTGEN_FIELD"
    assert audit["sensor_shape"] == [2048, 2048]
    assert audit["roi_order"] == ["top_left", "top_right", "bottom_left", "bottom_right"]
    assert set(audit["conditions"]) == {"C3-A", "C3-B", "C3-C"}
    expected_pixels_per_neuron = {"C3-A": 2.0 / 6.5, "C3-B": 10.0 / 6.5,
                                  "C3-C": (2.0 * 200.0 / 30.0) / 6.5}
    for candidate_id, result in audit["conditions"].items():
        assert math.isclose(result["source_relative_energy_um2"], 40000.0)
        assert math.isclose(result["camera_integrated_relative_energy_um2"], 40000.0,
                            rel_tol=1e-12, abs_tol=1e-10)
        assert result["energy_closure_relative_error"] <= 1e-12
        assert math.isclose(result["minimum_cell_capture_fraction"], 1.0, abs_tol=1e-12)
        assert result["roi_prediction_for_uniform_field"] == -1
        assert result["uniform_roi_score_tie"]
        assert result["outside_roi_relative_energy_um2"] == 0.0
        assert all(math.isclose(value, expected_pixels_per_neuron[candidate_id])
                   for value in result["cmos_pixels_per_lightgen_neuron_by_source_axis"])


def test_cmos_area_integral_preserves_signed_rotation_anisotropic_scale_and_offset():
    # A 180-degree relay inversion maps a top-left source spot to the bottom-right ROI.
    inverted = geo.scaled_rotation(-1.0, 0.0)
    area, info = geo.build_cmos_area_matrix(
        inverted, object_pixels=4, object_pixel_um=2.0,
        sensor_shape=(16, 16), sensor_pixel_um=1.0)
    source = torch.zeros((4, 4), dtype=torch.float64)
    source[0, 0] = 1.0
    camera = geo.integrate_cmos_pixels(source, area).reshape(16, 16)
    result = geo.roi_scores(camera, geo.equal_quadrant_rois(camera.shape))
    assert result["prediction"] == 3
    assert result["scores"].tolist() == [0.0, 0.0, 0.0, 4.0]
    assert torch.allclose(info["source_area_capture_fraction"], torch.ones((4, 4), dtype=torch.float64))

    # This independent affine includes rotation, unequal axis scales, and translation.
    affine = ((0.0, -2.0), (3.0, 0.0))
    offset = (1.0, -2.0)
    area, info = geo.build_cmos_area_matrix(
        affine, offset, object_pixels=2, object_pixel_um=1.0,
        sensor_shape=(16, 16), sensor_pixel_um=1.0)
    camera = geo.integrate_cmos_pixels(torch.ones((2, 2), dtype=torch.float64), area)
    assert math.isclose(float(camera.sum()), 4.0, rel_tol=1e-12, abs_tol=1e-12)
    assert torch.allclose(info["source_area_capture_fraction"], torch.ones((2, 2), dtype=torch.float64))
