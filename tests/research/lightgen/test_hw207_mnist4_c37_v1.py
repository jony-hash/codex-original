"""C3.7 targeted regressions: frozen geometry, spot statistics, losses, fabrication.

The suite is CPU only.  It exercises the production helpers in
``hw207_mnist4_c37_v1`` and the P5 export contract; it does not replace the
classification or focus experiments, which run on the cloud GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import hw207_mnist4_c37_v1 as c37
from scripts.research.lightgen import visualize_hw207_mnist4_c37_v1 as c37viz


PITCH = c37.CMOS_PITCH_UM
EXPECTED_ROI = ((921, 1024, 921, 1024), (921, 1024, 1024, 1127),
                (1024, 1127, 921, 1024), (1024, 1127, 1024, 1127))


def _cpu_forward_context() -> dict:
    return {
        "device": torch.device("cpu"),
        "lens_pitch_um": 4.0,
        "screen_to_lens_um": 80_000.0,
        "lens_to_cmos_um": 100_000.0,
        "aperture_mm": 2.0,
        "propagation_cache": {},
    }


@pytest.fixture(scope="module")
def propagation():
    """Two-sample forward result plus the single-sample recomputations."""
    torch.manual_seed(20261105)
    context = _cpu_forward_context()
    grid = torch.arange(c37.GRID, dtype=torch.float32)
    phase = (0.6 * torch.sin(grid[None, :] / 7.0) + 0.4 * torch.cos(grid[:, None] / 5.0))
    fields = torch.ones((2, c37.GRID, c37.GRID), dtype=torch.complex64)
    fields[1] = 0.5
    with torch.no_grad():
        batch = c37.forward_batch(fields, phase, context)
        single = [c37.forward_batch(fields[index:index + 1], phase, context) for index in range(2)]
        shifted = c37.forward_batch(fields[:1], phase + 0.7, context)
    return {"context": context, "phase": phase, "fields": fields, "batch": batch,
            "single": single, "shifted": shifted}


def test_frozen_roi_unchanged():
    assert c37.roi_bounds() == EXPECTED_ROI
    assert c37.ROI_SIDE_PX == 103
    for r0, r1, c0, c1 in c37.roi_bounds():
        assert (r1 - r0) == (c1 - c0) == c37.ROI_SIDE_PX
        assert 0 <= r0 < r1 <= c37.CMOS and 0 <= c0 < c1 <= c37.CMOS
    centres = c37.roi_center_px(c37.roi_bounds())
    assert np.allclose(centres.mean(0), [c37.FRAME_CENTER_PX, c37.FRAME_CENTER_PX])


def test_cores_inside_own_roi():
    for fraction in (0.3, 0.4, 0.5):
        cores = c37.core_bounds(fraction)
        for roi, core in zip(c37.roi_bounds(), cores):
            assert roi[0] <= core[0] < core[1] <= roi[1]
            assert roi[2] <= core[2] < core[3] <= roi[3]
        assert all((core[1] - core[0]) == (core[3] - core[2]) for core in cores)


def test_cores_do_not_overlap():
    for fraction in (0.3, 0.4, 0.5):
        masks = c37.core_masks(fraction, "cpu")
        assert float((masks.sum(0) > 1).sum()) == 0.0
        side = c37.core_side_px(fraction)
        assert side % 2 == 1
        expected_area = side * side
        assert all(int(masks[index].sum()) == expected_area for index in range(4))
        spec = c37.core_spec(fraction)
        assert spec["core_side_px"] == side
        assert all(record["core_edge_margin_to_roi_px"] > 0 for record in spec["records"])


def test_peak_coordinate_correct():
    intensity = torch.zeros(c37.CMOS, c37.CMOS)
    intensity[1003, 1007] = 5.0
    intensity[500, 500] = 1.0
    scores = torch.zeros(4, dtype=torch.float64)
    scores[0] = 1.0
    detected = float(intensity.sum()) * PITCH ** 2
    record = c37.row_statistics(intensity, scores, torch.tensor(detected),
                                torch.tensor([0.0, 0.0, 0.0, 0.0]), 0)
    assert record["global_peak_row"] == 1003
    assert record["global_peak_col"] == 1007
    assert record["global_peak_in_target_roi"] == 1.0
    assert record["global_peak_intensity"] == pytest.approx(5.0)
    distance = math.hypot(1003 - 972.0, 1007 - 972.0) * PITCH
    assert record["peak_distance_to_target_centre_um"] == pytest.approx(distance, rel=1e-9)


def test_centroid_correct():
    intensity = torch.zeros(c37.CMOS, c37.CMOS)
    intensity[900:1100, 1000:1010] = 1.0
    scores = torch.zeros(4, dtype=torch.float64)
    detected = float(intensity.sum()) * PITCH ** 2
    record = c37.row_statistics(intensity, scores, torch.tensor(detected),
                                torch.tensor([0.0, 0.0, 0.0, 0.0]), 0)
    assert record["centroid_row"] == pytest.approx(999.5, abs=1e-9)
    assert record["centroid_col"] == pytest.approx(1004.5, abs=1e-9)
    assert record["centroid_in_target_roi"] == 1.0


def test_area_weighted_energy_integral():
    intensity = torch.zeros(c37.CMOS, c37.CMOS)
    intensity[100:140, 100:160] = 2.0
    expected = 40 * 60 * 2.0 * PITCH ** 2
    record = c37.row_statistics(intensity, torch.tensor([0.0, 0.0, 0.0, 0.0]),
                                torch.tensor(expected), torch.tensor([0.0, 0.0, 0.0, 0.0]), 0)
    assert record["full_cmos_energy_um2"] == pytest.approx(expected, rel=1e-12)
    assert record["outside_target_roi_energy_fraction"] == pytest.approx(1.0, rel=1e-12)
    assert record["outside_target_roi_energy_fraction"] * expected == pytest.approx(expected, rel=1e-12)


def test_classification_argmax_and_tie():
    scores = torch.tensor([[1.0, 2.0, 0.5, 0.2]])
    assert int(scores.argmax(1)) == 1
    tied = np.array([[1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 3.0, 3.0]])
    assert tied.argmax(1).tolist() == [0, 2]
    predictions = np.array([0, 2])
    aggregates = c37.aggregate_statistics(
        [{"global_peak_in_target_roi": 1.0, "centroid_in_target_roi": 1.0,
          "target_roi_share_of_four_roi": 0.5, "target_roi_share_of_cmos": 0.4,
          "outside_target_roi_energy_fraction": 0.6, "peak_distance_to_target_centre_um": 1.0}
         for _ in range(2)],
        np.array([0, 2]), predictions, np.array([0.3, 0.3]), np.array([0.4, 0.4]))
    assert aggregates["accuracy"] == 1.0
    assert aggregates["confusion_matrix"] == [[1, 0, 0, 0], [0, 0, 0, 0],
                                              [0, 0, 1, 0], [0, 0, 0, 0]]


def _synthetic_intensity(seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    intensity = torch.zeros((2, c37.CMOS, c37.CMOS), dtype=torch.float32)
    for index in range(2):
        rows = torch.randint(900, 1150, (12,), generator=generator)
        columns = torch.randint(900, 1150, (12,), generator=generator)
        values = torch.rand(12, generator=generator) + 0.1
        intensity[index, rows, columns] = values
    return intensity


def test_phase_gradient_nonzero():
    intensity = _synthetic_intensity().requires_grad_(True)
    masks = c37.core_masks(0.4, "cpu")
    core = torch.stack([(intensity * masks[index]).sum((1, 2)) for index in range(4)], dim=1)
    energy = core.sum() * PITCH ** 2
    energy.backward()
    assert intensity.grad is not None
    assert torch.isfinite(intensity.grad).all()
    assert float(intensity.grad.abs().sum()) > 0
    expected_mask = masks.sum(0) * PITCH ** 2
    assert torch.allclose(intensity.grad[0], expected_mask, rtol=1e-5, atol=1e-8)


def test_focus_loss_gradient_matches_finite_difference():
    intensity = _synthetic_intensity().requires_grad_(True)
    masks = c37.core_masks(0.4, "cpu")
    labels = torch.tensor([0, 3])
    pixel_area = PITCH ** 2

    def loss_of(values: torch.Tensor) -> torch.Tensor:
        result = {
            "camera_intensity": values,
            "roi_scores": torch.ones((2, 4), dtype=torch.float32),
            "detected_energy": values.sum((1, 2)) * pixel_area,
            "outside_roi_full_cmos_energy": torch.zeros(2, dtype=torch.float32),
            "input_energy": torch.full((2,), 1.0, dtype=torch.float32),
            "comparison_crop_energy": torch.ones(2, dtype=torch.float32),
        }
        loss, _ = _objective_with_config(result, labels, masks)
        return loss

    loss = loss_of(intensity)
    loss.backward()
    gradient = intensity.grad.detach().clone()
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    step = 1e-4
    probes = [(0, 905, 902), (0, 1000, 1100), (1, 940, 1120)]
    for sample, row, column in probes:
        values = []
        for sign in (1.0, -1.0):
            probe = intensity.detach().clone()
            probe[sample, row, column] += sign * step
            values.append(float(loss_of(probe)))
        finite_difference = (values[0] - values[1]) / (2 * step)
        assert finite_difference == pytest.approx(float(gradient[sample, row, column]), rel=0.05,
                                                  abs=1e-6)


def _objective_with_config(result, labels, masks):
    config = {
        "training": {"classification_margin_weight": 0.0, "roi_leakage_weight": 0.0,
                     "focus_weight": 1.0, "phase_smoothness_weight": 0.0},
        "core": {"spread_normalisation_um": 669.5},
        "energy": {"adaptive_lagrange_rho": 200.0},
    }
    phase = torch.zeros((c37.GRID, c37.GRID), dtype=torch.float32)
    return c37.objective(result, labels, phase, {"device": torch.device("cpu")}, config,
                         groups=["focus"], eta_target=0.0, multiplier=0.0, masks=masks)


def test_focus_reduction_modes_match_their_definitions():
    """The worst-class reduction must pick the class with the least core energy."""
    base = _synthetic_intensity()
    intensity = torch.cat([base, base.flip(0)]).requires_grad_(True)
    masks = c37.core_masks(0.4, "cpu")
    labels = torch.tensor([0, 1, 2, 3])
    pixel_area = PITCH ** 2
    result = {
        "camera_intensity": intensity,
        "roi_scores": torch.ones((4, 4), dtype=torch.float32),
        "detected_energy": intensity.sum((1, 2)) * pixel_area,
        "outside_roi_full_cmos_energy": torch.zeros(4, dtype=torch.float32),
        "input_energy": torch.full((4,), 1.0, dtype=torch.float32),
        "comparison_crop_energy": torch.ones(4, dtype=torch.float32),
    }
    terms = c37.spot_terms(result, labels, masks, {"device": torch.device("cpu")})
    log_core = torch.log(terms["core_fraction"] + 1e-6)
    expected_mean = -log_core.mean()
    expected_worst = -log_core.min()

    def focus_of(mode: str) -> float:
        config = {
            "training": {"classification_margin_weight": 0.0, "roi_leakage_weight": 0.0,
                         "focus_weight": 1.0, "phase_smoothness_weight": 0.0,
                         "focus_mode": mode},
            "core": {"spread_normalisation_um": 669.5},
            "energy": {"adaptive_lagrange_rho": 200.0},
        }
        phase = torch.zeros((c37.GRID, c37.GRID), dtype=torch.float32)
        _, components = c37.objective(result, labels, phase, {"device": torch.device("cpu")},
                                      config, groups=["focus"], eta_target=0.0, multiplier=0.0,
                                      masks=masks)
        return float(components["focus"])

    assert focus_of("mean") == pytest.approx(float(expected_mean), rel=1e-6)
    assert focus_of("worst_class") == pytest.approx(float(expected_worst), rel=1e-6)
    assert float(expected_worst) > float(expected_mean), (
        "the worst class must be at least as penalised as the average")


def test_global_constant_phase_preserves_intensity(propagation):
    reference = propagation["batch"]["camera_intensity"][0]
    shifted = propagation["shifted"]["camera_intensity"][0]
    relative = float((reference - shifted).abs().max()) / float(reference.max())
    assert relative < 5e-3
    assert float(propagation["shifted"]["roi_scores"][0, 0]) == pytest.approx(
        float(propagation["batch"]["roi_scores"][0, 0]), rel=5e-3)


def test_batch_size_invariance(propagation):
    batched = propagation["batch"]["camera_intensity"]
    for index in range(2):
        single = propagation["single"][index]["camera_intensity"][0]
        relative = float((batched[index] - single).abs().max()) / max(float(single.max()), 1e-12)
        assert relative < 5e-3
        assert float(propagation["batch"]["roi_scores"][index].sum()) == pytest.approx(
            float(propagation["single"][index]["roi_scores"][0].sum()), rel=5e-3)


def test_propagation_numerically_stable(propagation):
    result = propagation["batch"]
    intensity = result["camera_intensity"]
    assert torch.isfinite(intensity).all()
    assert float(intensity.min()) >= 0.0
    for key in ("roi_scores", "input_energy", "detected_energy", "lens_energy"):
        assert torch.isfinite(result[key]).all()
        assert float(result[key].min()) > 0.0
    assert float(result["lens_energy"][0]) <= float(result["input_energy"][0]) * 1.0001


def test_energy_accounting_no_double_normalisation(propagation):
    result = propagation["batch"]
    four = result["roi_scores"].sum(1)
    outside = result["outside_roi_full_cmos_energy"]
    detected = result["detected_energy"]
    relative = ((four + outside - detected).abs() / detected).max()
    assert float(relative) < 1e-4
    eta = c37.eta_phase_to_cmos(detected, result["input_energy"])
    manual = (detected / result["input_energy"])
    assert torch.allclose(eta, manual, rtol=1e-6, atol=0.0)
    assert float(eta.max()) <= 1.0


def test_outside_target_roi_statistics():
    intensity = torch.zeros(c37.CMOS, c37.CMOS)
    intensity[921:1024, 921:1024] = 1.0
    intensity[100:200, 100:200] = 3.0
    inside = 103 * 103 * 1.0 * PITCH ** 2
    stray = 100 * 100 * 3.0 * PITCH ** 2
    total = inside + stray
    record = c37.row_statistics(intensity, torch.tensor([inside, 0.0, 0.0, 0.0], dtype=torch.float64),
                                torch.tensor(total), torch.tensor([inside, 0.0, 0.0, 0.0]), 0)
    assert record["outside_target_roi_energy_fraction"] == pytest.approx(stray / total, rel=1e-9)
    assert record["outside_target_roi_peak_intensity"] == pytest.approx(3.0)
    assert record["target_roi_peak_intensity"] == pytest.approx(1.0)


def test_fabrication_operator_is_height_domain():
    """The print operator must act on physical height, not on wrapped phase.

    A height domain pipeline cannot depend on which 2*pi representative of the
    phase is supplied, because ``height_from_phase`` wraps first.  A naive blur
    of the wrapped phase does depend on it, which is exactly the failure mode
    this regression rules out.
    """
    config = c37.load_config()
    grid = torch.arange(c37.GRID, dtype=torch.float32)
    phase = (2 * math.pi * torch.sin(grid[None, :] / 4.0)).expand(c37.GRID, c37.GRID).clone()
    wrapped = torch.remainder(phase, 2 * math.pi)
    recovered = c37.processed_phase(phase, config, delta_n=0.5, psf_sigma_px=0.0,
                                    minimum_feature_radius_px=0, height_levels=64,
                                    height_noise_nm=0.0, noise_seed=0)
    difference = torch.remainder(recovered - wrapped + math.pi, 2 * math.pi) - math.pi
    assert float(difference.abs().max()) < 0.15

    kwargs = {"psf_sigma_px": 1.2, "minimum_feature_radius_px": 1, "height_levels": 64,
              "height_noise_nm": 0.0, "noise_seed": 0}
    printed = c37.processed_phase(phase, config, delta_n=0.5, **kwargs)
    shifted = c37.processed_phase(phase + 2 * math.pi, config, delta_n=0.5, **kwargs)
    assert torch.allclose(printed, shifted, rtol=0.0, atol=1e-6)

    naive = torch.nn.functional.avg_pool2d(phase[None, None], 3, stride=1, padding=1)[0, 0]
    naive_shifted = torch.nn.functional.avg_pool2d((phase + 2 * math.pi)[None, None], 3,
                                                  stride=1, padding=1)[0, 0]
    assert not torch.allclose(naive, naive_shifted, rtol=0.0, atol=1e-3)

    height = c37.height_map_from_phase(phase, 0.5)
    assert float(height.max()) <= float(config["fabrication_assumptions"]["maximum_height_um"])


def test_phase_height_unit_conversion():
    phase = torch.tensor([[0.0, math.pi / 2], [math.pi, 2 * math.pi - 1e-3]])
    for delta_n in (0.5, 0.6):
        height = c37.height_map_from_phase(phase, delta_n)
        expected = torch.remainder(phase, 2 * math.pi) * (c37.c34.WAVELENGTH_UM / (2 * math.pi * delta_n))
        assert torch.allclose(height, expected, rtol=1e-6, atol=1e-9)
        round_trip = c37.c34.phase_from_height(height, delta_n)
        difference = torch.remainder(round_trip - torch.remainder(phase, 2 * math.pi) + math.pi,
                                     2 * math.pi) - math.pi
        assert float(difference.abs().max()) < 1e-4


def test_saved_intensity_map_matches_export(tmp_path: Path):
    rng = np.random.default_rng(7)
    intensity = np.zeros((c37.CMOS, c37.CMOS), dtype=np.float32)
    intensity[921:1024, 921:1024] = rng.random((103, 103)).astype(np.float32)
    path = tmp_path / "cmos.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, intensity=intensity)
    loaded = c37viz.load_intensity(path)
    assert loaded.shape == (c37.CMOS, c37.CMOS)
    assert np.array_equal(loaded, intensity.astype(np.float64))
    average = intensity[640:640 + 768, 640:640 + 768]
    average_path = tmp_path / "average.npy"
    np.save(average_path, average, allow_pickle=False)
    assert np.array_equal(c37viz.load_average(average_path), average.astype(np.float64))


def test_figure_roi_bounds_match_frozen_config():
    assert c37viz.FRAME_EXTENT_UM[0] == pytest.approx(-c37.CMOS / 2 * PITCH)
    assert c37viz.FRAME_EXTENT_UM[1] == pytest.approx(c37.CMOS / 2 * PITCH)
    for roi in c37.roi_bounds():
        x, y, width, height = c37viz.roi_pixel_extent(roi)
        assert width == pytest.approx(c37.ROI_SIDE_PX * PITCH)
        assert height == pytest.approx(c37.ROI_SIDE_PX * PITCH)
        centre_x = x + width / 2
        centre_y = y + height / 2
        expected = c37.roi_center_px(c37.roi_bounds())
        index = c37.roi_bounds().index(roi)
        assert centre_x == pytest.approx((expected[index][1] - c37.FRAME_CENTER_PX) * PITCH)
        assert centre_y == pytest.approx((expected[index][0] - c37.FRAME_CENTER_PX) * PITCH)
    assert c37viz.AVERAGE_EXTENT_UM[1] - c37viz.AVERAGE_EXTENT_UM[0] == pytest.approx(
        c37viz.AVERAGE_CROP[1] * PITCH)