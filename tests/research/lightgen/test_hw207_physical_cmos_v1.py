import math

import numpy as np
import pytest

from scripts.research.lightgen import hw207_physical_cmos_v1 as model


def _centered_weights(*, magnification=1.0, rotation_deg=0.0):
    matrix, offset = model.pixel_to_window_affine(
        (2048, 2048), 6.5, signed_magnification=magnification,
        rotation_deg=rotation_deg,
    )
    return model.build_roi_pixel_weights((2048, 2048), matrix, offset), matrix


def test_fixed_roi_coordinates_and_areas_are_exact():
    rois = model._validated_rois(model.DEFAULT_ROIS_UM, (200.0, 200.0))
    assert rois == model.DEFAULT_ROIS_UM
    assert [((x1 - x0) * (y1 - y0)) for _, x0, x1, y0, y1 in rois] == [2500.0] * 4
    assert sum((x1 - x0) * (y1 - y0) for _, x0, x1, y0, y1 in rois) / 40000 == 0.25


def test_cmos_roi_pixel_width_is_fixed_while_object_feature_projection_changes():
    config = model.load_config()
    nominal = model.nominal_geometry_summary(config)
    sensitivity = model.geometry_sensitivity_summary(config)
    assert math.isclose(
        nominal["fixed_cmos_pixel_equivalents_from_reference_pitch"]["roi_side_px"],
        50.0 / 6.5,
        rel_tol=1e-14,
    )
    object_plane = nominal["nominal_0p6x_if_phase_layer_is_object"]
    assert math.isclose(object_plane["fixed_cmos_roi_side_px"], 50.0 / 6.5, rel_tol=1e-14)
    assert math.isclose(object_plane["projected_50um_object_feature_side_px"], 30.0 / 6.5, rel_tol=1e-14)
    cases = {round(item["object_distance_from_pi_cls_mm"], 3): item
             for item in sensitivity["f30_thin_lens_scenarios"]}
    assert math.isclose(cases[60.0]["image_distance_from_principal_plane_mm"], 60.0, rel_tol=1e-14)
    assert math.isclose(cases[80.0]["image_distance_from_principal_plane_mm"], 48.0, rel_tol=1e-14)
    assert math.isclose(cases[42.857]["image_distance_from_principal_plane_mm"], 100.0, rel_tol=1e-14)


def test_dmd_calibration_patterns_are_native_shape_and_binary():
    patterns = model.make_dmd_calibration_patterns()
    assert len(patterns) == 6
    for pixels in patterns.values():
        assert pixels.shape == (1600, 2560)
        assert pixels.dtype == np.uint8
        assert set(np.unique(pixels).tolist()) <= {0, 255}


def test_area_weight_conservation_at_nominal_one_to_one_mapping():
    weights, matrix = _centered_weights()
    pixel_area = abs(np.linalg.det(matrix))
    for item in weights:
        assert math.isclose(
            float(item["fractions"].sum()) * pixel_area,
            2500.0,
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        assert math.isclose(item["effective_pixel_count"], 2500.0 / pixel_area, rel_tol=1e-10)


def test_axis_aligned_pixel_overlap_matches_independent_rectangle_reference():
    pitch_um = 6.5
    matrix = np.diag((pitch_um, pitch_um))
    offset = np.asarray((0.0, 0.0))
    weights = model.build_roi_pixel_weights((32, 32), matrix, offset)
    image = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
    actual = model.integrate_roi_energies(image, weights)
    expected = []
    for _, x0, x1, y0, y1 in model.DEFAULT_ROIS_UM:
        total = 0.0
        for row in range(32):
            overlap_y = max(0.0, min(y1, pitch_um * (row + 1)) - max(y0, pitch_um * row))
            for column in range(32):
                overlap_x = max(0.0, min(x1, pitch_um * (column + 1)) - max(x0, pitch_um * column))
                total += image[row, column] * overlap_x * overlap_y / (pitch_um**2)
        expected.append(total)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("roi_id", range(4))
def test_four_ideal_roi_spots_reproduce_fixed_argmax(roi_id):
    weights, _ = _centered_weights()
    image = np.zeros((2048, 2048), dtype=np.float64)
    x_center = (60.0, 140.0, 60.0, 140.0)[roi_id]
    y_center = (60.0, 60.0, 140.0, 140.0)[roi_id]
    matrix, offset = model.pixel_to_window_affine((2048, 2048), 6.5, signed_magnification=1.0)
    edge_xy = np.linalg.solve(matrix, np.asarray((x_center, y_center)) - offset)
    column, row = (int(math.floor(value)) for value in edge_xy)
    image[row, column] = 100.0
    energies = model.integrate_roi_energies(image, weights)
    assert model.predict_from_roi_energies(energies) == roi_id
    assert energies[roi_id] == 100.0


@pytest.mark.parametrize("magnification,rotation", ((1.0, 45.0), (-1.0, 0.0), (0.6, -90.0), (-0.6, 13.0)))
@pytest.mark.parametrize("roi_id", range(4))
def test_fixed_roi_argmax_survives_explicit_affine_orientation(magnification, rotation, roi_id):
    shape = (2048, 2048)
    matrix, offset = model.pixel_to_window_affine(
        shape, 6.5, signed_magnification=magnification, rotation_deg=rotation
    )
    weights = model.build_roi_pixel_weights(shape, matrix, offset)
    x_center = (60.0, 140.0, 60.0, 140.0)[roi_id]
    y_center = (60.0, 60.0, 140.0, 140.0)[roi_id]
    column, row = np.floor(np.linalg.solve(matrix, np.asarray((x_center, y_center)) - offset)).astype(int)
    image = np.zeros(shape, dtype=np.float64)
    image[row, column] = 1.0
    energies = model.integrate_roi_energies(image, weights)
    assert model.predict_from_roi_energies(energies) == roi_id


def test_rotation_and_inversion_preserve_roi_area_weights():
    for magnification, rotation in ((1.0, 45.0), (-1.0, 0.0), (-0.6, 13.0)):
        weights, matrix = _centered_weights(magnification=magnification, rotation_deg=rotation)
        pixel_area = abs(np.linalg.det(matrix))
        for item in weights:
            assert math.isclose(float(item["fractions"].sum()) * pixel_area, 2500.0,
                                rel_tol=1e-9, abs_tol=1e-7)


def test_exact_ties_and_zero_energy_abstain():
    assert model.predict_from_roi_energies((0.0, 0.0, 0.0, 0.0)) is None
    assert model.predict_from_roi_energies((1.0, 2.0, 2.0, 0.0)) is None
    assert model.predict_from_roi_energies((1.0, 2.0, 2.0 + 1e-16, 0.0)) is None


@pytest.mark.parametrize("scale", (1e-20, 1.0, 1e20))
def test_argmax_decision_is_invariant_to_positive_energy_scale(scale):
    assert model.predict_from_roi_energies((scale, 2.0 * scale, 0.0, 0.0)) == 1
    assert model.predict_from_roi_energies((scale, 2.0 * scale, 2.0 * scale, 0.0)) is None


def test_negative_background_subtracted_signal_fails_closed():
    weights, _ = _centered_weights()
    image = np.zeros((2048, 2048), dtype=np.float64)
    image[1000, 1000] = -0.1
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model.integrate_roi_energies(image, weights)


def test_overlapping_or_out_of_window_roi_definitions_are_rejected():
    overlapping = ((0, 35, 90, 35, 85), (1, 80, 130, 35, 85),
                   (2, 35, 85, 115, 165), (3, 115, 165, 115, 165))
    with pytest.raises(ValueError, match="must not overlap"):
        model._validated_rois(overlapping, (200, 200))
    outside = ((0, -1, 49, 35, 85), (1, 115, 165, 35, 85),
               (2, 35, 85, 115, 165), (3, 115, 165, 115, 165))
    with pytest.raises(ValueError, match="outside"):
        model._validated_rois(outside, (200, 200))


def test_sensor_missing_roi_area_is_rejected():
    matrix = np.eye(2) * 6.5
    offset = np.asarray((10000.0, 10000.0))
    with pytest.raises(ValueError, match="does not fully cover ROI"):
        model.build_roi_pixel_weights((16, 16), matrix, offset)


def test_abcd_diagnostic_matches_independent_matrix_product():
    actual = np.asarray(model.geometry.ray_matrix(230.0, 350.0, 150.0, 150.0, 150.0))
    p_obj = np.asarray(((1.0, 230.0), (0.0, 1.0)))
    lens = np.asarray(((1.0, 0.0), (-1.0 / 150.0, 1.0)))
    p_gap = np.asarray(((1.0, 350.0), (0.0, 1.0)))
    expected = np.asarray(((1.0, 150.0), (0.0, 1.0))) @ lens @ p_gap @ lens @ p_obj
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_fixed_roi_classification_metrics_match_independent_confusion_reference():
    labels = np.asarray((0, 0, 1, 1, 2, 2, 3, 3), dtype=np.int64)
    predictions = (0, 1, 1, None, 2, 3, 3, 3)
    actual = model.fixed_roi_classification_metrics(labels, predictions)

    expected_confusion = np.asarray(
        ((1, 1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 1), (0, 0, 0, 2)),
        dtype=np.int64,
    )
    expected_abstentions = np.asarray((0, 1, 0, 0), dtype=np.int64)
    np.testing.assert_array_equal(
        actual["confusion_matrix_true_rows_predicted_columns"], expected_confusion
    )
    np.testing.assert_array_equal(actual["abstentions_by_true_class"], expected_abstentions)
    assert actual["accuracy"] == 5 / 8
    assert actual["unresolved_count"] == 1

    # Independent classwise reference includes abstentions as false negatives.
    expected_f1 = []
    for class_id in range(4):
        tp = int(expected_confusion[class_id, class_id])
        support = int(expected_confusion[class_id].sum() + expected_abstentions[class_id])
        predicted = int(expected_confusion[:, class_id].sum())
        expected_f1.append(2 * tp / (support + predicted))
    assert math.isclose(actual["macro_f1"], sum(expected_f1) / 4, rel_tol=0, abs_tol=1e-15)


def test_fixed_roi_classification_metrics_reject_invalid_decision_values():
    with pytest.raises(ValueError, match="four-class range"):
        model.fixed_roi_classification_metrics((0, 4), (0, 1))
    with pytest.raises(ValueError, match="same one-dimensional shape"):
        model.fixed_roi_classification_metrics((0, 1), (0,))
    with pytest.raises(ValueError, match="class ids"):
        model.fixed_roi_classification_metrics((0,), (1.5,))


def test_intensity_concentration_metrics_match_uniform_and_single_pixel_references():
    uniform = np.ones((2, 2), dtype=np.float64)
    uniform_metrics = model.intensity_concentration_metrics(uniform, pixel_pitch_um=2.0)
    assert math.isclose(uniform_metrics["intensity_rms_radius_um"], math.sqrt(2.0), rel_tol=1e-14)
    assert math.isclose(uniform_metrics["effective_pixel_count"], 4.0, rel_tol=1e-14)
    assert math.isclose(uniform_metrics["effective_area_um2"], 16.0, rel_tol=1e-14)
    assert math.isclose(uniform_metrics["effective_area_fraction_of_map"], 1.0, rel_tol=1e-14)

    single_pixel = np.zeros((2, 2), dtype=np.float64)
    single_pixel[0, 1] = 7.0
    concentrated = model.intensity_concentration_metrics(single_pixel, pixel_pitch_um=2.0)
    assert concentrated["intensity_rms_radius_um"] == 0.0
    assert concentrated["effective_pixel_count"] == 1.0
    assert concentrated["effective_area_um2"] == 4.0
    assert concentrated["effective_area_fraction_of_map"] == 0.25


def test_intensity_concentration_metrics_reject_invalid_intensity():
    with pytest.raises(ValueError, match="positive total energy"):
        model.intensity_concentration_metrics(np.zeros((2, 2)), pixel_pitch_um=1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model.intensity_concentration_metrics(np.asarray([[1.0, -0.1]]), pixel_pitch_um=1.0)

