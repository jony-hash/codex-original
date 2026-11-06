from __future__ import annotations

import math

import numpy as np
import torch

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    PIXEL_PITCH_M,
    SOURCE_POWER_W,
    WAVELENGTH_M,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_finite_pixel import (
    FinitePixelAngularSpectrum,
    FinitePixelChenLightGen400,
    FinitePixelConfig,
    FinitePixelRSLinear,
    _UnvalidatedFinitePixelRSLinear,
    _relative_l2,
    _legendre_rule,
)


def _independent_rs_sum(values: torch.Tensor, order: int, output_shape: tuple[int, int], *,
                        centered: bool = False, periodic: bool = False) -> torch.Tensor:
    """Direct double-precision quadrature, independent of the production FFT path."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    offsets = [(float(y * PIXEL_PITCH_M / 2), float(x * PIXEL_PITCH_M / 2))
               for y in nodes for x in nodes]
    areas = [(float(wy * wx / 4) * PIXEL_PITCH_M**2)
             for wy in weights for wx in weights]
    batch, _, in_h, in_w = values.shape
    out_h, out_w = output_shape
    source_y, source_x = np.meshgrid(
        (np.arange(in_h) + 0.5 - in_h / 2) * PIXEL_PITCH_M,
        (np.arange(in_w) + 0.5 - in_w / 2) * PIXEL_PITCH_M, indexing="ij")
    target_y, target_x = np.meshgrid(
        (np.arange(out_h) + 0.5 - out_h / 2) * PIXEL_PITCH_M,
        (np.arange(out_w) + 0.5 - out_w / 2) * PIXEL_PITCH_M, indexing="ij")
    source_points = np.stack((source_y.reshape(-1), source_x.reshape(-1)), axis=1)
    target_points = np.stack((target_y.reshape(-1), target_x.reshape(-1)), axis=1)
    source_values = values.detach().cpu().numpy().astype(np.complex128)
    result = np.zeros((batch, order**2 if not centered else 1, out_h * out_w),
                      dtype=np.complex128)
    period_y, period_x = in_h * PIXEL_PITCH_M, in_w * PIXEL_PITCH_M
    for beta in (range(order**2) if not centered else (None,)):
        bdy, bdx = (offsets[beta] if beta is not None else (0.0, 0.0))
        for alpha, (ady, adx) in enumerate(offsets):
            dy = target_points[:, None, 0] - source_points[None, :, 0] + bdy - ady
            dx = target_points[:, None, 1] - source_points[None, :, 1] + bdx - adx
            if periodic:
                dy = np.remainder(dy + period_y / 2, period_y) - period_y / 2
                dx = np.remainder(dx + period_x / 2, period_x) - period_x / 2
            radius = np.sqrt(3.0e-3**2 + dy**2 + dx**2)
            kernel = (3.0e-3 / radius**2) * (
                1.0 / (2.0 * math.pi * radius) - 1j / WAVELENGTH_M
            ) * np.exp(2j * math.pi * radius / WAVELENGTH_M)
            value = source_values[:, alpha].reshape(batch, -1)
            result[:, 0 if centered else beta] += (value[:, None, :] * kernel[None] * areas[alpha]).sum(-1)
    return torch.from_numpy(result.reshape(batch, -1, out_h, out_w))


def test_gauss_legendre_pixel_rule_integrates_polynomials_to_declared_order() -> None:
    for order in (2, 4, 8, 12):
        nodes, weights = _legendre_rule(order)
        for degree in range(2 * order):
            actual = (weights * nodes.pow(degree)).sum()
            expected = 0.0 if degree % 2 else 1.0 / (degree + 1)
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64),
                                       rtol=1.0e-13, atol=1.0e-14)


def test_propagation_grid_zero_pads_source_plus_observation_support() -> None:
    config = FinitePixelConfig(quadrature_order=2, window_cells=64,
                               distance_m=3.0e-3, checkpoint_propagation=False)
    propagator = FinitePixelAngularSpectrum(2, (32, 32), config)
    fft_size = propagator.fy.numel()
    required = (64 + 32 - 1) * config.subgrid_factor
    assert fft_size >= required
    assert fft_size > config.window_cells * config.subgrid_factor
    assert config.fft_size_for((32, 32)) == fft_size


def test_rs_probe_consumer_reads_propagator_center_field_layout() -> None:
    from scripts.research.lightgen.execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 import (
        _select_rs_center_probes,
    )

    config = FinitePixelConfig(quadrature_order=2, window_cells=32,
                               distance_m=3.0e-3, checkpoint_propagation=False)
    propagator = FinitePixelAngularSpectrum(2, (32, 32), config)
    source = torch.ones((1, 4, 32, 32), dtype=torch.complex64)
    _, _, center_field = propagator(source, capture_center=True)
    assert center_field is not None and center_field.shape == (1, 32, 32)
    selected = _select_rs_center_probes(center_field)
    expected = center_field[0, 2:32:6, 2:32:6].reshape(-1)
    assert selected.shape == (25,)
    torch.testing.assert_close(selected, expected)


def test_independent_rs_derivative_gate_uses_strict_operator_tolerance(monkeypatch, tmp_path) -> None:
    from scripts.research.lightgen import execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 as runner

    monkeypatch.setattr(runner, "ORDERS", (12,))
    monkeypatch.setattr(runner, "DIRECT_RS_WINDOW_CELLS", 32)
    gaussian = runner._make_rs_patterns()["gaussian"]
    monkeypatch.setattr(runner, "_make_rs_patterns", lambda: {"gaussian": gaussian})
    observed_tolerances = []
    original_status = runner._numerical_qualification_status

    def record_status(measurements, required_keys, tolerance):
        observed_tolerances.append((required_keys, tolerance))
        return original_status(measurements, required_keys, tolerance)

    monkeypatch.setattr(runner, "_numerical_qualification_status", record_status)
    result = runner._direct_rs_path_check(torch.device("cpu"), tmp_path)

    assert result["derivative_tolerance"] == runner.WINDOW_TOLERANCE
    assert any(
        "gaussian:reference_derivative_12_vs_16" in required_keys
        and tolerance == runner.WINDOW_TOLERANCE
        for required_keys, tolerance in observed_tolerances
    )


def test_zero_padded_propagator_has_analytic_plane_wave_transfer_and_preserves_power() -> None:
    order = 12
    cells = 32
    pitch = PIXEL_PITCH_M
    distance = 3.0e-3
    config = FinitePixelConfig(quadrature_order=order, window_cells=cells,
                               distance_m=distance, checkpoint_propagation=False)
    propagator = FinitePixelAngularSpectrum(order, (cells, cells), config)
    nodes, _ = _legendre_rule(order)
    centers = (torch.arange(cells, dtype=torch.float64) + 0.5 - cells / 2.0) * pitch
    offsets = nodes * (pitch / 2.0)
    f_x = 1.0 / (cells * pitch)
    x_nodes = centers[None, :] + offsets[:, None]
    xx = x_nodes[None, :, None, :].expand(order, -1, cells, -1)
    yy = x_nodes[:, None, :, None].expand(-1, order, -1, cells)
    phase = 2.0 * math.pi * f_x * xx
    values = torch.exp(1j * (phase + torch.zeros_like(yy)))
    values = values.reshape(1, order**2, cells, cells).to(torch.complex64)
    fft_size = propagator.fft_size
    fx = torch.fft.fftfreq(fft_size, d=config.sample_pitch_m, dtype=torch.float64)
    selected_bin = 4
    selected_frequency = fx[selected_bin]
    spectrum = torch.zeros((1, fft_size, fft_size), dtype=torch.complex64)
    spectrum[0, 0, selected_bin] = 1.0
    power_spectrum = spectrum.clone()
    spectral_power = propagator._transfer_and_power(power_spectrum)
    f_z = torch.sqrt(torch.tensor(WAVELENGTH_M**-2, dtype=torch.float64) - selected_frequency.square())
    expected_transfer = torch.exp(2j * math.pi * distance * f_z).to(torch.complex64)
    torch.testing.assert_close(power_spectrum[0, 0, selected_bin], expected_transfer, rtol=2.0e-5, atol=2.0e-5)
    expected_spectral_power = config.sample_pitch_m**2 / fft_size**2
    assert abs(float(spectral_power[0]) - expected_spectral_power) / expected_spectral_power < 2.0e-5

    propagated, power, center = propagator(values, capture_center=True)
    assert center is not None and torch.isfinite(propagated).all()
    assert torch.isfinite(power).all() and bool((power > 0.0).all())


def test_physical_model_units_ols_reemission_cmos_and_batch_gradients() -> None:
    torch.manual_seed(17)
    model = FinitePixelChenLightGen400(
        FinitePixelConfig(quadrature_order=2, window_cells=512, checkpoint_propagation=True)
    )
    with torch.no_grad():
        model.encoder_phase.uniform_(-0.2, 0.2)
        model.decoder_phase.uniform_(-0.2, 0.2)
    images = torch.stack((torch.full((1, 400, 400), 0.25),
                          torch.linspace(0.0, 1.0, 400 * 400).reshape(1, 400, 400)))
    result = model.forward_from_image(images)
    torch.testing.assert_close(result["input_power_w"],
                               torch.tensor([SOURCE_POWER_W * 0.25,
                                             SOURCE_POWER_W * images[1].mean().item()],
                                            dtype=result["input_power_w"].dtype),
                               rtol=2.0e-6, atol=1.0e-10)
    assert result["physical_w_per_pixel"].shape == (2, 1, 400, 400)
    assert result["plane_powers_w"].shape == (2, 7)
    assert torch.isfinite(result["ols_coefficients"]).all()

    mode_y = model.modes_y[4]
    mode_x = model.modes_x[4]
    matching_mode = torch.einsum("yi,xj->ijyx", mode_y, mode_x).reshape(1, 4, 400, 400)
    coefficient = model._project_coefficients(matching_mode.to(torch.complex64))
    assert abs(complex(coefficient[0, 44]) - 1.0) < 1.0e-5
    other = coefficient.clone()
    other[0, 44] = 0.0
    assert float(other.abs().max()) < 1.0e-7

    arbitrary = torch.arange(100, dtype=torch.float32).to(torch.complex64).unsqueeze(0)
    reemitted = model._reemit(arbitrary)
    modal_power = arbitrary.abs().square().sum(1)
    output_power = model._weighted_power(reemitted)
    quadratic = torch.einsum("i,ij,j->", arbitrary[0].conj(), model.gram.to(torch.complex64), arbitrary[0])
    torch.testing.assert_close(output_power, quadratic.real.unsqueeze(0), rtol=2.0e-5, atol=1.0e-6)
    assert bool((output_power <= modal_power * (1.0 + 1.0e-6)).all())

    anti_phase_nodes = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.complex64).reshape(1, 4, 1, 1)
    node_weights = model.node_weight_2d.float().reshape(1, 4, 1, 1)
    coherent_average_intensity = (anti_phase_nodes * node_weights).sum(1).abs().square()
    detector_average_intensity = (anti_phase_nodes.abs().square() * node_weights).sum(1)
    assert float(coherent_average_intensity.max()) < 1.0e-7
    torch.testing.assert_close(detector_average_intensity, torch.ones_like(detector_average_intensity))

    target = images * 0.7
    loss = (result["physical_w_per_pixel"] - target).square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in (model.encoder_phase, model.decoder_phase))


def test_numerical_decision_consumer_computes_status_from_measurements() -> None:
    from scripts.research.lightgen.execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 import (
        _numerical_qualification_status,
    )

    order = 2
    values = torch.randn((1, order**2, 8, 8), dtype=torch.complex64)
    values = values * torch.exp(1j * torch.linspace(0.0, 0.3, 64).reshape(1, 1, 8, 8))
    config = FinitePixelConfig(quadrature_order=order, window_cells=32,
                               distance_m=3.0e-3, checkpoint_propagation=False)
    propagator = FinitePixelAngularSpectrum(order, (8, 8), config)
    reference, _, _ = propagator(values)
    perturbed_source = values.clone()
    perturbed_source[:, :, 3, 4] *= 1j
    altered, _, _ = propagator(perturbed_source)
    measurements = {"propagated_field": _relative_l2(reference, reference.clone()),
                    "counterexample_field": _relative_l2(altered, reference)}
    assert _numerical_qualification_status(
        {"propagated_field": measurements["propagated_field"]}, {"propagated_field"}, 1.0e-4
    ) == "NUMERICALLY_QUALIFIED"
    assert _numerical_qualification_status(
        {"counterexample_field": measurements["counterexample_field"]}, {"counterexample_field"}, 1.0e-4
    ) == "NUMERICAL_NOT_QUALIFIED"
    assert _numerical_qualification_status({}, {"propagated_field"}, 1.0e-4) == "INVALID_HARNESS"


def test_production_rs_linear_matches_direct_sum_at_edges_and_pixel_centres() -> None:
    torch.manual_seed(37)
    order = 2
    config = FinitePixelConfig(quadrature_order=order, window_cells=16,
                               checkpoint_propagation=False, propagation_backend="rs_linear",
                               window_power_tile_cells=5, window_power_source_chunk=2,
                               window_power_target_chunk=2)
    operator = FinitePixelRSLinear(order, (8, 8), config)
    source = torch.randn((1, order**2, 8, 8), dtype=torch.complex128)
    node_fields, _, center = operator(source, output_shape=(6, 6), capture_center=True)
    direct_nodes = _independent_rs_sum(source, order, (6, 6))
    direct_center = _independent_rs_sum(source, order, (6, 6), centered=True)
    torch.testing.assert_close(node_fields, direct_nodes, rtol=2.0e-10, atol=2.0e-12)
    assert center is not None
    torch.testing.assert_close(center[:, None], direct_center, rtol=2.0e-10, atol=2.0e-12)
    old_operator = _UnvalidatedFinitePixelRSLinear(order, (8, 8), config)
    old_nodes, _, _ = old_operator(source, output_shape=(6, 6))
    old_error = torch.linalg.vector_norm(old_nodes.to(torch.complex128) - direct_nodes) \
        / torch.linalg.vector_norm(direct_nodes)
    assert float(old_error) > 1.0e-4

    # An edge source is far from the opposite edge in the physical plane.
    edge_source = torch.zeros((1, order**2, 8, 8), dtype=torch.complex128)
    edge_source[:, :, -1, -1] = 1.0
    edge_output, _, _ = operator(edge_source, output_shape=(8, 8))
    direct_edge = _independent_rs_sum(edge_source, order, (8, 8))
    torch.testing.assert_close(edge_output, direct_edge, rtol=2.0e-10, atol=2.0e-12)
    cyclic_edge = _independent_rs_sum(edge_source, order, (8, 8), periodic=True)
    assert float((cyclic_edge[:, 0, 0, 0] - edge_output[:, 0, 0, 0]).abs().max()) > 1.0e-7


def test_production_rs_linear_autograd_matches_direct_directional_difference() -> None:
    torch.manual_seed(43)
    order = 2
    config = FinitePixelConfig(quadrature_order=order, window_cells=12,
                               checkpoint_propagation=False, propagation_backend="rs_linear")
    operator = FinitePixelRSLinear(order, (8, 8), config)
    amplitude = torch.randn((1, 1, 8, 8), dtype=torch.float64)
    phase = torch.randn((8, 8), dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(phase)
    direction = direction / torch.linalg.vector_norm(direction)
    cotangent = torch.randn((1, order**2, 8, 8), dtype=torch.complex128)

    def objective(value: torch.Tensor) -> torch.Tensor:
        field = (amplitude * torch.exp(1j * value)[None, None]).expand(1, order**2, 8, 8)
        propagated, _, _ = operator(field)
        return (propagated * cotangent.conj()).real.sum()

    loss = objective(phase)
    gradient, = torch.autograd.grad(loss, phase)
    analytic = float((gradient * direction).sum())
    step = 1.25e-3
    finite = float((objective(phase.detach() + step * direction)
                    - objective(phase.detach() - step * direction)) / (2 * step))
    assert analytic * finite > 0.0
    assert abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-14) < 1e-5

    # The independent direct-space producer gives the same directional derivative.
    plus = amplitude * torch.exp(1j * (phase.detach() + step * direction))[None, None]
    minus = amplitude * torch.exp(1j * (phase.detach() - step * direction))[None, None]
    plus = plus.expand(1, order**2, 8, 8)
    minus = minus.expand(1, order**2, 8, 8)
    direct_plus = _independent_rs_sum(plus, order, (8, 8))
    direct_minus = _independent_rs_sum(minus, order, (8, 8))
    direct_finite = float((((direct_plus * cotangent.conj()).real.sum()
                            - (direct_minus * cotangent.conj()).real.sum()) / (2 * step)))
    assert abs(analytic - direct_finite) / max(abs(analytic), abs(direct_finite), 1e-14) < 1e-5


def test_weak_checkerboard_field_cancellation_matches_direct_reference_in_complex64_path() -> None:
    cells = 32
    order = 2
    coords = (torch.arange(cells, dtype=torch.float64) + 0.5 - cells / 2) * PIXEL_PITCH_M
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    amplitude = torch.exp(-(xx.square() + yy.square()) / (2 * (18.0e-6) ** 2))
    parity = (torch.arange(cells)[:, None] + torch.arange(cells)[None, :]) % 2
    pattern = (amplitude * torch.where(parity == 0, 1.0, -1.0)).to(torch.complex64)
    source = pattern[None, None].expand(1, order**2, cells, cells)
    operator = FinitePixelRSLinear(order, (cells, cells), FinitePixelConfig(
        quadrature_order=order, window_cells=cells, checkpoint_propagation=False,
        propagation_backend="rs_linear"))
    node_fields, _, _ = operator(source, output_shape=(cells, cells))
    direct = _independent_rs_sum(source, order, (cells, cells))
    error = torch.linalg.vector_norm(node_fields.to(torch.complex128) - direct) \
        / torch.linalg.vector_norm(direct)
    assert float(torch.linalg.vector_norm(direct)) > 0.0
    assert float(error) < 1.0e-4


def test_weak_checkerboard_passes_the_production_candidate_gate(monkeypatch, tmp_path) -> None:
    from scripts.research.lightgen import execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 as runner

    cells = 32
    coords = (np.arange(cells, dtype=np.float64) + 0.5 - cells / 2) * PIXEL_PITCH_M
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    amplitude = np.exp(-(xx**2 + yy**2) / (2 * (18.0e-6) ** 2))
    parity = (np.arange(cells)[:, None] + np.arange(cells)[None, :]) % 2
    checkerboard = amplitude * np.where(parity == 0, 1.0, -1.0)
    monkeypatch.setattr(runner, "_make_rs_patterns", lambda: {"checkerboard": checkerboard})
    monkeypatch.setattr(runner, "DIRECT_RS_ORDERS", (2, 4, 8, 16))
    monkeypatch.setattr(runner, "CANDIDATE_ORDERS", (2, 4, 8))
    monkeypatch.setattr(runner, "DIRECT_RS_WINDOW_CELLS", cells)
    result = runner._direct_rs_path_check(torch.device("cpu"), tmp_path)
    assert result["reference_status"] == "NUMERICALLY_QUALIFIED"
    assert result["reference_derivative_status"] == "NUMERICALLY_QUALIFIED"
    assert result["candidate_status"] == {
        "2": "NUMERICAL_NOT_QUALIFIED",
        "4": "NUMERICALLY_QUALIFIED",
        "8": "NUMERICALLY_QUALIFIED",
    }, result["measurements"]


def test_research_model_selects_the_directly_checked_rs_backend() -> None:
    config = FinitePixelConfig(quadrature_order=2, window_cells=512,
                               checkpoint_propagation=False, propagation_backend="rs_linear")
    model = FinitePixelChenLightGen400(config)
    assert isinstance(model.propagator, FinitePixelRSLinear)
    assert model.propagator.operator.__class__.__name__ == "FinitePixelRayleighSommerfeld"


def test_rs_full_window_power_is_area_integral_of_observation_nodes() -> None:
    order = 2
    config = FinitePixelConfig(quadrature_order=order, window_cells=16,
                               checkpoint_propagation=False, propagation_backend="rs_linear")
    operator = FinitePixelRSLinear(order, (8, 8), config)
    source = torch.ones((1, order**2, 8, 8), dtype=torch.complex128)
    with torch.no_grad():
        _, power, _ = operator(source, output_shape=(6, 6), full_window_power=True)
    full_nodes = _independent_rs_sum(source, order, (16, 16))
    _, observer_weights = _legendre_rule(order)
    wy, wx = torch.meshgrid(observer_weights, observer_weights, indexing="ij")
    expected = (full_nodes.abs().square() * (wy * wx).reshape(1, -1, 1, 1)
                ).sum() * PIXEL_PITCH_M**2
    torch.testing.assert_close(power[0], expected, rtol=2.0e-10, atol=2.0e-20)


def test_production_far_axis_tile_matches_independent_coordinate_sum() -> None:
    from scripts.research.lightgen.execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 import (
        _far_axis_tile_check,
    )

    result = _far_axis_tile_check(2, torch.device("cpu"))
    assert result["status"] == "passed"
    assert result["target_offset_cells"] == [-2047.5, -2047.5]
    assert result["relative_error"] <= result["tolerance"]


def test_rs_model_returns_camera_physical_pixels_and_seven_plane_units() -> None:
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=2, window_cells=512, checkpoint_propagation=False,
        propagation_backend="rs_linear"))
    image = torch.zeros((1, 1, 400, 400), dtype=torch.float32)
    image[:, :, 100:300, 145:255] = 1.0
    with torch.no_grad():
        result = model.forward_from_image(image, capture_fields=True)
    assert result["physical_w_per_pixel"].shape == (1, 1, 400, 400)
    assert result["camera_roi_w_per_pixel"].shape == (1, 1, 500, 500)
    assert result["ols_coefficients"].shape == (1, 100)
    assert result["plane_powers_w"].shape == (1, 7)
    assert result["plane_power_scope"] == "camera_observation_500x500"
    assert bool(torch.isfinite(result["physical_w_per_pixel"]).all())
    assert bool(torch.isfinite(result["plane_powers_w"]).all())
    torch.testing.assert_close(result["physical_w_per_pixel"],
                               result["camera_roi_w_per_pixel"][..., 50:450, 50:450])
    assert float(result["input_power_w"][0]) > 0.0
    assert float(result["camera_roi_power_w"][0]) > 0.0


def test_rs_input_keeps_the_400_physical_pixels_piecewise_constant() -> None:
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=4, window_cells=512, checkpoint_propagation=False,
        propagation_backend="rs_linear"))
    checkerboard = (torch.arange(400)[:, None] + torch.arange(400)[None, :]).remainder(2).float()
    isolated = torch.zeros_like(checkerboard)
    isolated[200, 200] = 1.0
    inputs = torch.stack((checkerboard, isolated)).unsqueeze(1)
    actual = model._initial_node_intensities(inputs)
    expected = inputs.expand(-1, model.order**2, -1, -1)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    # This reproduces the removed second interpolation: it bleeds light into
    # formerly black pixels and therefore must disagree with the SLM map.
    nodes, _ = _legendre_rule(model.order)
    from torch.nn import functional as functional

    y = torch.arange(400, dtype=torch.float32)
    x = torch.arange(400, dtype=torch.float32)
    yy, xx = torch.meshgrid(y + nodes[-1].float() / 2, x + nodes[-1].float() / 2, indexing="ij")
    grid = torch.stack((2 * xx / 399 - 1, 2 * yy / 399 - 1), dim=-1).unsqueeze(0)
    old_sample = functional.grid_sample(inputs[1:2], grid, mode="bilinear",
                                        padding_mode="border", align_corners=True)
    assert float(old_sample[0, 0, 199, 199]) > 0.0
    assert float(actual[1, 0, 200, 199]) == 0.0

