"""C3.3 single-layer phase screen: physics, export and fabrication contracts."""

import math

import numpy as np
import pytest
import torch

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as core


@pytest.fixture(scope="module")
def configuration():
    return core.build_configuration(0.0, "cpu")


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(0)
    image = torch.zeros((2, 28, 28), dtype=torch.uint8)
    image[0, 6:22, 8:12] = 255
    image[1, 8:20, 6:22] = 255
    return image


@pytest.fixture(scope="module")
def random_phase():
    generator = torch.Generator().manual_seed(17)
    return torch.rand((1, 1, core.GRID, core.GRID), generator=generator) * (2 * math.pi)


def test_phase_screen_geometry(configuration):
    screen = configuration.base.screen
    assert screen.layers == 1
    assert screen.active_shape == (core.GRID, core.GRID)
    assert screen.phase.shape == (1, 1, core.GRID, core.GRID)
    assert screen.spacing_m == 0.0
    assert core.PITCH_M == 2e-6
    assert core.GRID * core.PITCH_M == pytest.approx(200e-6)


def test_phase_wrapping_range(random_phase):
    wrapped = core.wrap_phase(random_phase + 7.5)
    assert float(wrapped.min()) >= 0.0
    assert float(wrapped.max()) < 2 * math.pi
    assert torch.allclose(core.wrap_phase(wrapped), wrapped, atol=1e-6)


def test_zero_phase_zero_distance_is_identity(configuration, images):
    zeros = torch.zeros(1, 1, core.GRID, core.GRID)
    result = core.forward_field(images, configuration, zeros)
    assert torch.allclose(result["lightgen_output_field"], result["lightgen_input_field"], atol=1e-6)


def test_phase_changes_intensity_under_finite_pupil(configuration, images, random_phase):
    zeros = torch.zeros_like(random_phase)
    with torch.no_grad():
        plain = core.forward_field(images, configuration, zeros)
        modulated = core.forward_field(images, configuration, random_phase)
    change = (modulated["camera_ideal_intensity"] - plain["camera_ideal_intensity"]).abs().mean()
    relative = change / plain["camera_ideal_intensity"].mean()
    assert float(relative) > 1e-3


def test_global_phase_shift_is_invisible(configuration, images, random_phase):
    with torch.no_grad():
        base = core.forward_field(images, configuration, random_phase)
        shifted = core.forward_field(images, configuration, random_phase + 1.234)
    relative = (shifted["camera_ideal_intensity"] - base["camera_ideal_intensity"]).abs().max()
    relative = relative / base["camera_ideal_intensity"].max()
    assert float(relative) < 1e-4


def test_phase_gradient_present_and_sparse_fd_matches(configuration, images):
    generator = torch.Generator().manual_seed(5)
    phase = (torch.rand((1, 1, core.GRID, core.GRID), generator=generator)
             * (0.2 * math.pi)).requires_grad_(True)
    labels = torch.tensor([0, 1])

    def loss_value() -> torch.Tensor:
        result = core.forward_field(images, configuration, phase)
        return core.loss_from_result({**result, "phase": phase}, labels)

    loss = loss_value()
    gradient = torch.autograd.grad(loss, phase)[0]
    assert float(gradient.norm()) > 1e-4
    entries = torch.topk(gradient.detach().abs().flatten(), 2).indices
    step = 5e-3
    numeric = []
    with torch.no_grad():
        original = phase.flatten().clone()
        for index in entries.tolist():
            phase.flatten()[index] = original[index] + step
            plus = float(loss_value())
            phase.flatten()[index] = original[index] - step
            minus = float(loss_value())
            phase.flatten()[index] = original[index]
            numeric.append((plus - minus) / (2 * step))
    analytic = gradient.flatten()[entries]
    scale = analytic.abs().max().clamp_min(1e-12)
    assert float(((analytic - torch.tensor(numeric)).abs() / scale).max()) < 0.05


def test_energy_ledger_and_roi_conservation(configuration, images, random_phase):
    with torch.no_grad():
        result = core.forward_field(images, configuration, random_phase)
    scores = result["roi_scores"].sum(1)
    reconstructed = scores + result["outside_roi_energy"]
    assert torch.allclose(reconstructed, result["detected_energy"], rtol=1e-6, atol=1e-6)
    detected = result["detected_energy"]
    input_energy = result["energy"]["lightgen_input_energy_um2"]
    assert torch.all((detected > 0) & (detected <= input_energy * 1.001))


def test_classification_decision_rules():
    scores = torch.tensor([[3.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    total = scores.sum(1)
    predictions = core.predictions_from_scores(scores, total)
    assert predictions.tolist() == [0, -1, -1]


def test_height_conversion_round_trip(random_phase):
    delta_n = 0.6
    height = core.phase_to_height_um(random_phase, delta_n)
    assert core.height_two_pi_um(delta_n) == pytest.approx(core.WAVELENGTH_M / delta_n * 1e6)
    rebuilt = core.height_to_phase(height, delta_n)
    assert torch.allclose(core.wrap_phase(rebuilt), core.wrap_phase(random_phase), atol=1e-6)
    assert float(height.max()) <= core.height_two_pi_um(delta_n) + 1e-9


def test_height_quantization_bounds(random_phase):
    height = core.phase_to_height_um(random_phase, 0.55)
    quantized = core.quantize_height(height, 32, 1.0)
    step = 1.0 / 31
    residual = (quantized / step - (quantized / step).round()).abs()
    assert float(residual.max()) < 1e-4
    assert float(quantized.min()) >= 0.0
    assert float(quantized.max()) <= 1.0 + 1e-6


def test_fabrication_operators_are_bounded(random_phase):
    height = core.phase_to_height_um(random_phase, 0.5)
    smoothed = core.smooth_height(height, 0.8)
    opened = core.min_feature_open(smoothed, 1)
    perturbed = core.perturb_height(opened, 0.02, torch.Generator().manual_seed(2))
    assert smoothed.shape == height.shape
    assert float(opened.min()) >= float(smoothed.min()) - 1e-6
    assert float(perturbed.min()) >= 0.0
    assert float((perturbed - opened).abs().mean()) > 0.0


def test_stl_mesh_closed_and_sized():
    grid = 6
    height = np.zeros((grid, grid), dtype=np.float64)
    height[2:4, 2:4] = 0.8
    triangles, info = core.heightmap_triangles(height, substrate_um=0.5)
    assert triangles.shape[1:] == (3, 3)
    assert info["vertices"] == 2 * grid * grid
    check = core.check_mesh(triangles)
    assert check["closed"] is True
    assert check["nonmanifold_edges"] == 0
    assert check["degenerate_faces"] == 0
    span = (grid - 1) * core.PITCH_M * 1e6
    assert check["bbox_um"][0] == pytest.approx([0.0, span])
    assert check["bbox_um"][2][1] == pytest.approx(0.8)


def test_stl_round_trip_units(tmp_path):
    grid = 5
    height = np.full((grid, grid), 0.3)
    triangles, _ = core.heightmap_triangles(height, substrate_um=0.2)
    path = tmp_path / "probe.stl"
    core.write_binary_stl(path, triangles)
    reread, _ = core.read_binary_stl(path)
    assert reread.shape == triangles.shape
    assert np.allclose(reread, triangles, atol=1e-5)
    span = (grid - 1) * core.PITCH_M * 1e6
    assert reread[:, :, 0].max() == pytest.approx(span, abs=1e-4)
    assert reread[:, :, 2].min() == pytest.approx(-0.2, abs=1e-5)


def test_fabrication_resimulation_preserves_shapes(configuration, images, random_phase):
    delta_n = 0.6
    height = core.phase_to_height_um(random_phase, delta_n)
    constrained = core.quantize_height(height, 16, 1.2)
    constrained = core.smooth_height(constrained, 0.8)
    constrained = core.min_feature_open(constrained, 1)
    constrained = core.perturb_height(constrained, 0.02, torch.Generator().manual_seed(3))
    rebuilt = core.height_to_phase(constrained, delta_n)
    assert rebuilt.shape == random_phase.shape
    with torch.no_grad():
        result = core.forward_field(images, configuration, rebuilt)
    assert torch.isfinite(result["roi_scores"]).all()
    assert torch.isfinite(result["camera_ideal_intensity"]).all()
    assert float(result["detected_energy"].min()) > 0.0
    error = (core.wrap_phase(rebuilt) - core.wrap_phase(random_phase)).abs().mean()
    assert float(error) > 0.0


def test_stratified_split_is_disjoint_and_balanced():
    labels = torch.randint(0, 4, (4000,))
    indices = torch.arange(4000)
    train, validation, holdout = core.stratified_split(indices, labels, seed=1,
                                                       n_holdout=400, n_validation=400)
    union = torch.cat((train, validation, holdout))
    assert len(torch.unique(union)) == len(union) == 4000
    counts = torch.bincount(labels[holdout], minlength=4)
    assert counts.min().item() == counts.max().item() == 100
    assert set(train.tolist()).isdisjoint(holdout.tolist())