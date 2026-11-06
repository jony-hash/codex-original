from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    PIXEL_PITCH_M,
    WAVELENGTH_M,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_finite_pixel import (
    FinitePixelAngularSpectrum,
    FinitePixelChenLightGen400,
    FinitePixelConfig,
    FinitePixelRSLinear,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_rs_convolution import (
    FinitePixelRayleighSommerfeld,
    FinitePixelRayleighSommerfeldCached,
)
from scripts.research.lightgen.execute_chen_lightgen_400_fast_rs_compact_v1 import (
    HarnessError,
    _compact_result,
    _compact_task_metrics,
    _estimate_remaining_work,
    _finite_difference_record_passes,
    _total_loss,
    _validate_reference_bundle,
    _verify_qualification_input_snapshot,
    _cost_probe,
    _parse_args,
)
from scripts.research.lightgen import execute_chen_lightgen_400_fast_rs_compact_v1 as fast_rs
from scripts.research.lightgen import execute_chen_lightgen_400_q12_trainability_v1 as q12


def direct_node_integral(
    values, order, shape, centered, output_offset_cells=(0.0, 0.0), target_order=None,
):
    """Independent NumPy sum over source coordinates, without FFT/embedding."""
    target_order = order if target_order is None else target_order
    nodes, weights = np.polynomial.legendre.leggauss(order)
    offsets = [(y * PIXEL_PITCH_M / 2, x * PIXEL_PITCH_M / 2) for y in nodes for x in nodes]
    areas = np.outer(weights, weights).reshape(-1) * PIXEL_PITCH_M**2 / 4
    target_nodes, _ = np.polynomial.legendre.leggauss(target_order)
    target_offsets = [(y * PIXEL_PITCH_M / 2, x * PIXEL_PITCH_M / 2)
                      for y in target_nodes for x in target_nodes]
    input_shape = values.shape[-2:]
    sy, sx = np.meshgrid((np.arange(input_shape[0]) + .5 - input_shape[0] / 2) * PIXEL_PITCH_M,
                         (np.arange(input_shape[1]) + .5 - input_shape[1] / 2) * PIXEL_PITCH_M,
                         indexing="ij")
    ty, tx = np.meshgrid((np.arange(shape[0]) + .5 - shape[0] / 2) * PIXEL_PITCH_M,
                         (np.arange(shape[1]) + .5 - shape[1] / 2) * PIXEL_PITCH_M,
                         indexing="ij")
    ty += output_offset_cells[0] * PIXEL_PITCH_M
    tx += output_offset_cells[1] * PIXEL_PITCH_M
    targets = [(0, 0)] if centered else target_offsets
    out = np.zeros((values.shape[0], len(targets), *shape), dtype=np.complex128)
    for target, (dy, dx) in enumerate(targets):
        for iy in range(shape[0]):
            for ix in range(shape[1]):
                for source, (oy, ox) in enumerate(offsets):
                    radius = np.sqrt(.003**2 + (ty[iy, ix] + dy - sy - oy)**2
                                     + (tx[iy, ix] + dx - sx - ox)**2)
                    kernel = .003 * np.exp(2j * np.pi * radius / WAVELENGTH_M) / (2 * np.pi * radius**3) \
                        * (1 - 2j * np.pi * radius / WAVELENGTH_M)
                    out[:, target, iy, ix] += (values[:, source] * kernel * areas[source]).sum((-2, -1))
    return out


def direct_node_adjoint(values, source_order, target_order, output_offset_cells=(0.0, 0.0),
                        input_shape=(4, 3)):
    """Independent coordinate sum for the conjugate transpose of the node map."""
    batch, _, output_height, output_width = values.shape
    source_nodes, source_weights = np.polynomial.legendre.leggauss(source_order)
    target_nodes, _ = np.polynomial.legendre.leggauss(target_order)
    source_offsets = [(y * PIXEL_PITCH_M / 2, x * PIXEL_PITCH_M / 2)
                      for y in source_nodes for x in source_nodes]
    target_offsets = [(y * PIXEL_PITCH_M / 2, x * PIXEL_PITCH_M / 2)
                      for y in target_nodes for x in target_nodes]
    source_areas = np.outer(source_weights, source_weights).reshape(-1) * PIXEL_PITCH_M**2 / 4
    input_height, input_width = input_shape
    sy, sx = np.meshgrid((np.arange(input_height) + .5 - input_height / 2) * PIXEL_PITCH_M,
                         (np.arange(input_width) + .5 - input_width / 2) * PIXEL_PITCH_M,
                         indexing="ij")
    ty, tx = np.meshgrid((np.arange(output_height) + .5 - output_height / 2) * PIXEL_PITCH_M,
                         (np.arange(output_width) + .5 - output_width / 2) * PIXEL_PITCH_M,
                         indexing="ij")
    ty += output_offset_cells[0] * PIXEL_PITCH_M
    tx += output_offset_cells[1] * PIXEL_PITCH_M
    output = np.zeros((batch, source_order**2, input_height, input_width), dtype=np.complex128)
    for source_index, (source_dy, source_dx) in enumerate(source_offsets):
        area = source_areas[source_index]
        for target_index, (target_dy, target_dx) in enumerate(target_offsets):
            dy = ty[:, :, None, None] + target_dy - sy[None, None, :, :] - source_dy
            dx = tx[:, :, None, None] + target_dx - sx[None, None, :, :] - source_dx
            radius = np.sqrt(.003**2 + dy**2 + dx**2)
            kernel = .003 * np.exp(2j * np.pi * radius / WAVELENGTH_M) / (2 * np.pi * radius**3) \
                * (1 - 2j * np.pi * radius / WAVELENGTH_M)
            output[:, source_index] += (
                values[:, target_index, :, :, None, None] * kernel.conj()[None] * area
            ).sum(axis=(1, 2))
    return output


@pytest.mark.parametrize("centered", [True, False])
def test_rs_toeplitz_producer_matches_independent_node_integral(centered):
    torch.manual_seed(9)
    source = torch.randn(1, 4, 4, 5, dtype=torch.complex128)
    operator = FinitePixelRayleighSommerfeld(2, pair_block=3)
    actual = operator(source, output_shape=(3, 4), centered=centered)
    reference = direct_node_integral(source.numpy(), 2, (3, 4), centered)
    torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=2e-10, atol=1e-12)
    # Changing the circulant embedding must not change a fixed observation.
    larger = operator(source, output_shape=(3, 4), centered=centered, fft_shape=(16, 20))
    torch.testing.assert_close(larger, actual, rtol=2e-10, atol=1e-12)


def test_rs_high_precision_accumulator_reaches_inverse_fft(monkeypatch):
    source = torch.randn(1, 4, 4, 5, dtype=torch.complex64)
    operator = FinitePixelRayleighSommerfeld(2, pair_block=2)
    observed_ifft_dtypes = []
    original_ifft2 = torch.fft.ifft2

    def observe_ifft2(values, *args, **kwargs):
        observed_ifft_dtypes.append(values.dtype)
        return original_ifft2(values, *args, **kwargs)

    monkeypatch.setattr(torch.fft, "ifft2", observe_ifft2)
    result = operator(source, output_shape=(3, 4))
    assert observed_ifft_dtypes and set(observed_ifft_dtypes) == {torch.complex128}
    assert result.dtype == torch.complex64


def test_rs_observer_tile_uses_physical_offset_and_source_chunks():
    torch.manual_seed(23)
    source = torch.randn(1, 4, 400, 400, dtype=torch.complex128)
    offset = (-2047.5, -2047.5)  # first pixel centre of a 4096-cell window
    operator = FinitePixelRayleighSommerfeld(2, pair_block=2)
    actual = operator(source, output_shape=(1, 1), output_offset_cells=offset,
                      source_chunk_size=2)
    reference = direct_node_integral(source.numpy(), 2, (1, 1), False, offset)
    torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=3e-10, atol=1e-14)


def test_extra_zero_padding_changes_embedding_without_changing_observation(monkeypatch):
    torch.manual_seed(29)
    source = torch.randn(1, 4, 8, 8, dtype=torch.complex128)
    base = FinitePixelRSLinear(2, (8, 8), FinitePixelConfig(
        quadrature_order=2, window_cells=32, checkpoint_propagation=False,
        propagation_backend="rs_linear"))
    padded = FinitePixelRSLinear(2, (8, 8), FinitePixelConfig(
        quadrature_order=2, window_cells=32, checkpoint_propagation=False,
        propagation_backend="rs_linear", fft_embedding_extra_cells=11))
    embeddings = []
    original_apply = FinitePixelRayleighSommerfeld._apply_convolution

    def record_embedding(self, values, output_shape, centered, fft_shape, **kwargs):
        embeddings.append(tuple(fft_shape))
        return original_apply(self, values, output_shape, centered, fft_shape, **kwargs)

    monkeypatch.setattr(FinitePixelRayleighSommerfeld, "_apply_convolution", record_embedding)
    actual, power, _ = base(source, output_shape=(6, 6))
    wider, wider_power, _ = padded(source, output_shape=(6, 6))
    assert embeddings[0] != embeddings[1]
    torch.testing.assert_close(wider, actual, rtol=2e-10, atol=1e-12)
    torch.testing.assert_close(wider_power, power, rtol=2e-10, atol=1e-12)


def test_extra_zero_padding_is_passed_to_center_observer(monkeypatch):
    model = FinitePixelRSLinear(2, (8, 8), FinitePixelConfig(
        quadrature_order=2, window_cells=32, checkpoint_propagation=False,
        propagation_backend="rs_linear", fft_embedding_extra_cells=11))
    observed_calls = []

    def observe(values, *, output_shape=None, centered=False, fft_shape=None, **kwargs):
        observed_calls.append({"output_shape": output_shape, "centered": centered,
                               "fft_shape": fft_shape})
        channels = 1 if centered else 4
        return torch.zeros((values.shape[0], channels, *output_shape), dtype=values.dtype)

    monkeypatch.setattr(model.operator, "forward", observe)
    model(torch.ones((1, 4, 8, 8), dtype=torch.complex128), output_shape=(6, 6), capture_center=True)

    assert len(observed_calls) == 2
    assert observed_calls[0]["centered"] is False
    assert observed_calls[1]["centered"] is True
    assert observed_calls[0]["fft_shape"] == observed_calls[1]["fft_shape"]
    assert observed_calls[0]["fft_shape"] != FinitePixelRSLinear._fft_shape((8, 8), (6, 6))


def test_mixed_source_and_observer_quadrature_matches_direct_sum_and_adjoint():
    torch.manual_seed(31)
    source = torch.randn(1, 16, 4, 3, dtype=torch.complex128, requires_grad=True)
    operator = FinitePixelRayleighSommerfeld(4, pair_block=3)
    actual = operator(source, output_shape=(2, 2), target_order=2)
    reference = direct_node_integral(source.detach().numpy(), 4, (2, 2), False,
                                     target_order=2)
    torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=3e-10, atol=1e-12)

    cotangent = torch.randn_like(actual)
    objective = (actual.conj() * cotangent).real.sum()
    gradient, = torch.autograd.grad(objective, source)
    lhs = torch.vdot(actual.detach().reshape(-1), cotangent.reshape(-1))
    rhs = torch.vdot(source.detach().reshape(-1), gradient.reshape(-1))
    torch.testing.assert_close(lhs, rhs, rtol=3e-10, atol=1e-12)
    direct_adjoint = torch.from_numpy(
        direct_node_adjoint(cotangent.detach().numpy(), source_order=4, target_order=2)
    )
    torch.testing.assert_close(gradient, direct_adjoint, rtol=3e-10, atol=1e-12)


def test_real_wraparound_counterexample_and_production_rejection():
    # A source at the far edge must not be observed with the near-neighbour
    # kernel across the boundary, as an unpadded periodic FFT would do.
    operator = FinitePixelRayleighSommerfeld(2)
    source = torch.zeros(1, 4, 8, 8, dtype=torch.complex128)
    source[:, :, -1, -1] = 1
    actual = operator(source, centered=True)
    independent = torch.from_numpy(direct_node_integral(source.numpy(), 2, (8, 8), True))
    torch.testing.assert_close(actual, independent, rtol=2e-10, atol=1e-12)
    # The faulty circular behavior is assembled from the same physical kernel.
    folded = sum(torch.fft.ifft2(torch.fft.fft2(source[:, i], s=(8, 8))
                                * operator._kernel_spectrum((8, 8), (8, 8), (8, 8),
                                    slice(0, 1), slice(i, i + 1), True,
                                    source.device, source.dtype)[0, 0]) for i in range(4))
    assert float((folded[:, 0, 0] - actual[:, 0, 0, 0]).abs().max()) > 1e-6
    with pytest.raises(ValueError, match="all source/target"):
        operator(source, fft_shape=(8, 8))


def test_rs_discrete_adjoint_and_physical_phase_derivative():
    torch.manual_seed(15)
    operator = FinitePixelRayleighSommerfeld(2, pair_block=3)
    amplitude = torch.randn(1, 4, 4, 5, dtype=torch.complex128)
    phase = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(1, 4, 3, 4, dtype=torch.complex128)
    offset = (2.25, -1.5)

    def objective(p):
        field = amplitude * torch.exp(1j * p)
        return (operator(field, output_shape=(3, 4), output_offset_cells=offset)
                * weight.conj()).real.sum()

    value = objective(phase)
    gradient, = torch.autograd.grad(value, phase)
    direction = torch.randn_like(phase)
    direction /= torch.linalg.vector_norm(direction)
    analytic = float((gradient * direction).sum())
    for step in (2.5e-3, 1.25e-3):
        finite = float((objective(phase.detach() + step * direction)
                        - objective(phase.detach() - step * direction)) / (2 * step))
        assert finite * analytic > 0
        assert abs(finite - analytic) / abs(finite) < 1e-5
    # Direct coordinate sum checks the derivative independently of our adjoint.
    derivative_field = 1j * amplitude.numpy() * np.exp(1j * phase.detach().numpy()) * direction.numpy()
    independent = float((direct_node_integral(derivative_field, 2, (3, 4), False, offset)
                         * weight.numpy().conj()).real.sum())
    assert abs(independent - analytic) / abs(independent) < 1e-9


def test_regression_old_angular_spectrum_fails_direct_rs_consumer():
    from scripts.research.lightgen import execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 as old
    from scripts.research.lightgen import diagnose_chen_lightgen_400_sampling_closure_v1 as sampling

    pattern = old._make_rs_patterns()["checkerboard"]
    reference, _ = sampling._rayleigh_sommerfeld_pixel_integral(pattern, 12)
    # A small periodic domain reproduces the old invalid linear-padding claim.
    source = torch.from_numpy(pattern)[None, None].expand(1, 16, -1, -1)
    former = FinitePixelAngularSpectrum(4, (32, 32), FinitePixelConfig(
        quadrature_order=4, window_cells=32, checkpoint_propagation=False))
    _, _, previous = former(source, capture_center=True)
    corrected = FinitePixelRayleighSommerfeld(4)(source, centered=True)[:, 0]

    def consume(field):
        actual = field[0, 2:32:6, 2:32:6].numpy().reshape(-1)
        phase = np.exp(-1j * np.angle(np.vdot(reference, actual)))
        error = float(np.linalg.norm(actual * phase - reference) / np.linalg.norm(reference))
        return old._numerical_qualification_status({"value": error}, {"value"}, 1e-4)

    assert consume(previous) == "NUMERICAL_NOT_QUALIFIED"
    assert consume(corrected) == "NUMERICALLY_QUALIFIED"


@pytest.mark.parametrize("order,source_shape,output_shape,fft_shape", [
    (2, (4, 4), (3, 3), (8, 8)),
    (4, (4, 4), (5, 5), (10, 10)),
])
def test_cached_rs_symmetry_matches_direct_integral_and_adjoint(order, source_shape,
                                                                 output_shape, fft_shape):
    torch.manual_seed(101 + order)
    source = torch.randn(1, order**2, *source_shape, dtype=torch.complex128, requires_grad=True)
    cached = FinitePixelRayleighSommerfeldCached(order, pair_block=2, cache_build_batch=3)
    actual = cached(source, output_shape=output_shape, fft_shape=fft_shape)
    reference = direct_node_integral(source.detach().numpy(), order, output_shape, False)
    torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=2e-10, atol=1e-12)

    cotangent = torch.randn_like(actual)
    objective = (actual.conj() * cotangent).real.sum()
    gradient, = torch.autograd.grad(objective, source)
    direct = torch.from_numpy(direct_node_adjoint(
        cotangent.detach().numpy(), source_order=order, target_order=order,
        input_shape=source_shape,
    ))
    torch.testing.assert_close(gradient, direct, rtol=3e-10, atol=1e-12)
    assert cached.cache_info["cache_bytes"] > 0
    assert "_kernel_cache" not in cached.state_dict()


def test_cached_rs_q12_uses_703_exact_node_difference_kernels():
    operator = FinitePixelRayleighSommerfeldCached(12)
    assert operator.cache_info["distinct_absolute_node_differences"] == 37
    assert operator.cache_info["canonical_kernel_count"] == 703


def test_cached_q12_matches_original_operator_for_all_node_pairs():
    torch.manual_seed(116)
    source = torch.randn(1, 144, 3, 3, dtype=torch.complex128, requires_grad=True)
    reference_op = FinitePixelRayleighSommerfeld(12, pair_block=8)
    cached_op = FinitePixelRayleighSommerfeldCached(12, pair_block=8, cache_build_batch=8)
    reference = reference_op(source, output_shape=(4, 4), fft_shape=(8, 8))
    candidate = cached_op(source, output_shape=(4, 4), fft_shape=(8, 8))
    relative_error = torch.linalg.vector_norm(candidate - reference) / torch.linalg.vector_norm(reference)
    assert relative_error < 2.0e-12
    cotangent = torch.randn_like(reference)
    reference_gradient, = torch.autograd.grad((reference.conj() * cotangent).real.sum(), source,
                                             retain_graph=True)
    candidate_gradient, = torch.autograd.grad((candidate.conj() * cotangent).real.sum(), source)
    gradient_error = torch.linalg.vector_norm(candidate_gradient - reference_gradient) \
        / torch.linalg.vector_norm(reference_gradient)
    assert gradient_error < 2.0e-12


def test_cached_rs_complex64_candidate_is_an_explicit_low_precision_operator():
    torch.manual_seed(117)
    source = torch.randn(1, 4, 4, 4, dtype=torch.complex64)
    high = FinitePixelRayleighSommerfeldCached(2, compute_precision="complex128")
    low = FinitePixelRayleighSommerfeldCached(2, compute_precision="complex64")
    high_value = high(source)
    low_value = low(source)
    assert low_value.dtype == torch.complex64
    relative_error = torch.linalg.vector_norm(
        (low_value - high_value).to(torch.complex128)
    ) / torch.linalg.vector_norm(high_value.to(torch.complex128))
    assert relative_error < 1.0e-5


def test_cached_rs_400_to_500_style_embedding_and_extra_padding():
    torch.manual_seed(118)
    source = torch.randn(1, 16, 4, 4, dtype=torch.complex128)
    cached = FinitePixelRayleighSommerfeldCached(4, pair_block=2, cache_build_batch=3)
    minimum_embedding = cached(source, output_shape=(5, 5), fft_shape=(10, 10))
    wider_embedding = cached(source, output_shape=(5, 5), fft_shape=(14, 14))
    reference = torch.from_numpy(direct_node_integral(source.numpy(), 4, (5, 5), False))
    torch.testing.assert_close(minimum_embedding, reference, rtol=2e-10, atol=1e-12)
    torch.testing.assert_close(wider_embedding, reference, rtol=2e-10, atol=1e-12)


def test_cached_rs_rectangular_observation_uses_exact_fallback():
    torch.manual_seed(119)
    source = torch.randn(1, 4, 4, 5, dtype=torch.complex128)
    cached = FinitePixelRayleighSommerfeldCached(2)
    actual = cached(source, output_shape=(3, 4), fft_shape=(8, 10))
    reference = direct_node_integral(source.numpy(), 2, (3, 4), False)
    torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=2e-10, atol=1e-12)
    assert cached.cache_info["cache_bytes"] == 0


def test_production_adapter_selects_cached_operator_and_keeps_cache_out_of_state_dict():
    config = FinitePixelConfig(quadrature_order=2, window_cells=8,
                               checkpoint_propagation=False,
                               propagation_backend="rs_linear_cached")
    adapter = FinitePixelRSLinear(2, (4, 4), config)
    assert isinstance(adapter.operator, FinitePixelRayleighSommerfeldCached)
    assert "operator._kernel_cache" not in adapter.state_dict()


def test_compact_observation_stores_one_physical_image_not_redundant_views():
    result = {
        "physical_w_per_pixel": torch.zeros((1, 1, 400, 400), dtype=torch.float64),
        "ols_coefficients": torch.zeros((1, 100), dtype=torch.complex128),
        "input_power_w": torch.zeros((1,), dtype=torch.float64),
        "ols_input_power_w": torch.zeros((1,), dtype=torch.float64),
        "ols_output_power_w": torch.zeros((1,), dtype=torch.float64),
        "camera_roi_power_w": torch.zeros((1,), dtype=torch.float64),
        "plane_powers_w": torch.zeros((1, 7), dtype=torch.float64),
    }
    compact = _compact_result(result, 51301, 0, "source_step_30000", "complex128")
    image_keys = [key for key, value in compact.items()
                  if isinstance(value, torch.Tensor) and value.ndim >= 3]
    assert image_keys == ["physical_w_per_pixel"]
    assert compact["physical_w_per_pixel"].dtype == torch.float32
    assert compact["physical_w_per_pixel"].numel() * compact["physical_w_per_pixel"].element_size() == 640_000


def test_cached_complex128_precision_is_preserved_between_production_planes():
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=2, window_cells=400, checkpoint_propagation=False,
        propagation_backend="rs_linear_cached", rs_compute_precision="complex128",
    ))
    observed_dtypes = []

    def identity_propagation(values, *, output_shape=None, capture_center=False,
                             full_window_power=False):
        observed_dtypes.append(values.dtype)
        power = torch.zeros(values.shape[0], dtype=torch.float64, device=values.device)
        return values, power, None

    model.propagator.forward = identity_propagation
    with torch.no_grad():
        result = model.forward_from_image(torch.zeros(1, 1, 400, 400))
    assert observed_dtypes == [torch.complex128] * 7
    assert result["ols_coefficients"].dtype == torch.complex128


def test_fast_runner_worker_cli_and_explicit_gradient_reference_path():
    with pytest.raises(HarnessError, match="worker mode is disabled"):
        _parse_args(["--worker-payload", "worker.json"])

    required_flags = [
        "--qualification-only", "--source-checkpoint", "baseline.pt", "--source-report", "report.json",
        "--source-start-state", "start.pt", "--source-endpoint-checkpoint", "end.pt",
        "--fixed-population", "population.pt", "--config", "config.json",
        "--data", "training.pt", "--evaluator", "evaluator.pt",
        "--qualification-report", "qualification.md", "--reference-run", "prior-run",
        "--structure-coefficient", "eta.json", "--output-dir", "run",
    ]
    with pytest.raises(HarnessError, match="gradient_reference_dir"):
        _parse_args(["--mode", "preflight", *required_flags])

    complete_paths = [*required_flags, "--gradient-reference-dir", "gradients"]
    parsed = _parse_args(["--mode", "preflight", *complete_paths, "--target-hours", "1"])
    assert parsed.qualification_only and parsed.target_hours == 1.0

    with pytest.raises(HarnessError, match="worker mode is disabled"):
        _parse_args(["--worker-payload", "worker.json", "--mode", "resume"])


def test_fast_runner_estimate_counts_the_predeclared_validation_work():
    operator = {"records": {
        "cached_complex128": {
            "elapsed_seconds_including_cold_build": 10.0,
            "cache": {"cache_build_seconds": 2.0},
        },
        "cached_complex64": {
            "elapsed_seconds_including_cold_build": 6.0,
            "cache": {"cache_build_seconds": 2.0},
        },
    }}
    estimate = _estimate_remaining_work(operator)
    assert estimate["map_equivalents"]["complex128"]["finite_difference_perturbed_forwards"] == 308
    assert estimate["map_equivalents"]["complex64"]["compact_full_chain"] == 280
    assert estimate["map_equivalents"]["complex64"]["batch10_cost_cases"] == 560
    assert all("recovery" not in key for group in estimate["map_equivalents"].values() for key in group)
    assert any("estimated duration is advisory" in assumption
               for assumption in estimate["assumptions"])
    assert estimate["map_seconds_from_one_hop_after_cache_build"] == {
        "complex128": 4.0, "complex64": 2.0,
    }


def test_fast_runner_rejects_missing_or_duplicate_reference_population(tmp_path):
    args = type("Args", (), {
        "reference_run": tmp_path / "reference",
        "gradient_reference_dir": tmp_path / "gradients",
    })()
    rows = torch.arange(200).reshape(10, 20)
    context = {"population": {"confirm_rows_by_class": rows}}
    with pytest.raises(HarnessError, match="reference bundle raw files differ"):
        _validate_reference_bundle(args, context)

    duplicate_rows = rows.clone()
    duplicate_rows[1, 0] = duplicate_rows[0, 0]
    duplicate_context = {"population": {"confirm_rows_by_class": duplicate_rows}}
    with pytest.raises(HarnessError, match="reference bundle raw files differ"):
        _validate_reference_bundle(args, duplicate_context)


def test_complex64_resume_rejects_a_partial_compact_measurement_pair(monkeypatch, tmp_path):
    output = tmp_path / "run"
    compact_dir = output / "precision_c64"
    compact_dir.mkdir(parents=True)
    row = 12345
    (compact_dir / f"{fast_rs.STATES[0]}_row{row}_label0.pt").write_bytes(b"partial")

    class DummyModel:
        pass

    monkeypatch.setattr(fast_rs, "_make_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(fast_rs, "_set_phases", lambda *args, **kwargs: None)
    context = {
        "state_phases": {fast_rs.STATES[0]: {}},
        "population": {"confirm_rows_by_class": torch.tensor([[row], [row + 1]])},
    }
    args = type("Args", (), {"reference_run": tmp_path / "reference"})()

    with pytest.raises(HarnessError, match="partial complex64 compact evidence"):
        fast_rs._measure_complex64_full_chain(
            context, args, output, torch.device("cpu"), fast_rs.Heartbeat(output, {}), {}
        )


def test_fast_runner_resume_rejects_a_different_loaded_phase_state():
    saved = {
        "metadata": {"source_checkpoint_step": 30_000, "rows": [1, 2]},
        "state_phases": {
            fast_rs.STATES[0]: {"encoder_phase": torch.zeros(1, 2)},
            fast_rs.STATES[1]: {"encoder_phase": torch.ones(1, 2)},
        },
    }
    current = {
        "metadata": dict(saved["metadata"]),
        "state_phases": {
            fast_rs.STATES[0]: {"encoder_phase": torch.zeros(1, 2)},
            fast_rs.STATES[1]: {"encoder_phase": torch.full((1, 2), 2.0)},
        },
    }
    with pytest.raises(HarnessError, match="resume model phases differ"):
        _verify_qualification_input_snapshot(saved, current)


def test_b_total_loss_gradient_matches_the_production_microbatch_and_finite_difference(monkeypatch):
    class SyntheticOptics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.full((2, 1, 1, 1), 0.08))
            self.decoder_phase = torch.nn.Parameter(torch.full((4, 1, 1, 1), -0.03))

        def forward_from_image(self, target):
            level = (self.encoder_phase.sum() + self.decoder_phase.sum()) / 6.0
            rows = torch.arange(target.shape[0], dtype=target.dtype, device=target.device)
            modulation = torch.sin(rows[:, None, None, None] * 0.2 + level)
            base = 0.65 * target + 0.12
            delta = 0.07 + 0.03 * torch.cos(target * 8.0)
            prediction = base + modulation * delta
            return {"physical_w_per_pixel": prediction / q12.FIXED_GAIN}

    pixels = torch.rand((10, 1, 28, 28), dtype=torch.float32)

    def image_batch(data, rows, device):
        return data.index_select(0, rows.cpu()).to(device), None

    monkeypatch.setattr(fast_rs.baseline, "_image_batch", image_batch)
    model = SyntheticOptics()
    beta, eta = 0.8, 0.35
    loss, gradients, _elapsed, _peak = q12._run_microbatch(
        model, pixels, torch.arange(10), beta, eta, "B", 1, torch.device("cpu")
    )
    analytic = float(gradients["encoder_phase"][0, 0, 0, 0])
    original = model.encoder_phase.detach().clone()
    epsilon = 1.0e-3
    losses = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            model.encoder_phase.copy_(original)
            model.encoder_phase[0, 0, 0, 0] += sign * epsilon
        candidate, _, _, _ = q12._run_microbatch(
            model, pixels, torch.arange(10), beta, eta, "B", 1, torch.device("cpu")
        )
        losses.append(candidate)
    derivative = (losses[0] - losses[1]) / (2.0 * epsilon)
    assert loss > 0
    assert abs(analytic - derivative) / max(abs(analytic), abs(derivative)) < 1.0e-2

    one_target = pixels[:1]
    with torch.no_grad():
        model.encoder_phase.copy_(original)
    output = model.forward_from_image(one_target)
    b_loss = _total_loss(output, one_target, beta, eta, "B")
    a_loss = _total_loss(output, one_target, beta, eta, "A")
    assert not torch.isclose(b_loss, a_loss)


def test_finite_difference_rejects_an_inverted_derivative_sign():
    checks = [{
        "degenerate": False, "same_sign": False, "relative_error": 1.0e-6,
    } for _ in q12.FD_STEPS]
    assert not _finite_difference_record_passes(checks)


def test_batch10_cost_saves_one_gradient_per_case_and_never_calls_adam(monkeypatch, tmp_path):
    class FakeOperator:
        def __init__(self):
            self.cache_info = {"cached_geometry": None}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.zeros((2, 1, 2, 2)))
            self.decoder_phase = torch.nn.Parameter(torch.zeros((4, 1, 2, 2)))
            self.propagator = type("Propagator", (), {"operator": FakeOperator()})()

    model = FakeModel()
    calls = []

    def run_once(current, pixels, rows, beta, eta, branch, microbatch, device):
        calls.append((branch, microbatch))
        current.propagator.operator.cache_info["cached_geometry"] = "400x400"
        gradients = {"encoder_phase": torch.ones_like(current.encoder_phase),
                     "decoder_phase": torch.full_like(current.decoder_phase, 2.0)}
        return 1.25, gradients, 4.0 + microbatch, 1024

    monkeypatch.setattr(fast_rs, "_make_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(fast_rs.prior, "_set_trainable", lambda value: None)
    monkeypatch.setattr(fast_rs.q12, "_run_microbatch", run_once)
    eta_path = tmp_path / "eta.json"
    eta_path.write_text(json.dumps({"eta": 0.25}), encoding="utf-8")
    context = {
        "state_phases": {fast_rs.STATES[0]: {}}, "beta": 1.0,
        "population": {"fit_rows_by_class": torch.arange(100).reshape(10, 10)},
        "pixels": torch.empty(0),
    }
    result = _cost_probe(context, tmp_path / "run", torch.device("cpu"),
                         fast_rs.Heartbeat(tmp_path / "run", {}), eta_path, "complex128")
    assert len(calls) == 4
    assert set(calls) == {("A", 1), ("A", 2), ("B", 1), ("B", 2)}
    assert result["status"] == "NUMERICALLY_QUALIFIED"
    assert result["adam_calls"] == result["optimizer_calls"] == result["optimizer_updates"] == 0
    assert all(item["adam_calls"] == 0 for item in result["records"].values())


def test_heartbeat_uses_task_id_from_current_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    progress = {
        "task_id": fast_rs.QUALIFICATION_TASK_ID,
        "stage": "complex64_finite_difference",
    }
    heartbeat_path = tmp_path / "heartbeat.json"
    with fast_rs.Heartbeat(tmp_path, progress, interval_s=0.01):
        deadline = time.monotonic() + 2.0
        while not heartbeat_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.005)
    assert heartbeat_path.is_file()
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["task_id"] == fast_rs.QUALIFICATION_TASK_ID
    assert heartbeat["stage"] == progress["stage"]
    assert heartbeat["last_progress"]["task_id"] == fast_rs.QUALIFICATION_TASK_ID


def test_compact_float32_readback_reproduces_gain_output_and_mse():
    torch.manual_seed(120)
    original = torch.rand(1, 1, 400, 400, dtype=torch.float64) * 1.0e-11
    target = torch.rand(1, 1, 400, 400, dtype=torch.float32)
    compact = original.to(torch.float32)
    replay = _compact_task_metrics(compact, target, original)
    assert replay["task_output_relative_l2_from_original_float64"] < 1.0e-6
    assert replay["mse_relative_error_from_original_float64"] < 1.0e-6
    assert replay["pixel_count"] == 160_000


@pytest.mark.parametrize("compute_precision", ["complex128", "complex64"])
def test_cached_rs_static_transpose_schedule_matches_integral_and_reordered_adjoint(
    compute_precision,
):
    torch.manual_seed(1007)
    source_dtype = torch.complex128 if compute_precision == "complex128" else torch.complex64
    source = torch.randn(1, 4, 4, 4, dtype=source_dtype)
    target_indices = (3, 0, 2)
    reference = FinitePixelRayleighSommerfeld(2, pair_block=2)
    cached = FinitePixelRayleighSommerfeldCached(
        2, pair_block=2, compute_precision=compute_precision
    )

    actual = cached(source, output_shape=(4, 4), fft_shape=(8, 8),
                    target_indices=target_indices, target_order=2, source_chunk_size=3)
    direct = reference(source, output_shape=(4, 4), fft_shape=(8, 8),
                       target_indices=target_indices, target_order=2, source_chunk_size=3)
    tolerance = 2e-10 if compute_precision == "complex128" else 2e-4
    torch.testing.assert_close(actual, direct, rtol=tolerance, atol=1e-12)

    cotangent = torch.randn_like(actual)
    cached_adjoint = cached._apply_convolution(
        cotangent, (4, 4), False, (8, 8), adjoint=True,
        input_shape=(1, 4, 4, 4), target_indices=target_indices,
        source_chunk_size=2, target_order=2,
    )
    direct_adjoint = reference._apply_convolution(
        cotangent, (4, 4), False, (8, 8), adjoint=True,
        input_shape=(1, 4, 4, 4), target_indices=target_indices,
        source_chunk_size=2, target_order=2,
    )
    torch.testing.assert_close(cached_adjoint, direct_adjoint, rtol=tolerance, atol=1e-12)
    assert cached._transpose_schedule_cache
    assert "_pair_transposed_host" not in cached.state_dict()


def test_cached_rs_transpose_schedule_removes_device_boolean_reduction(monkeypatch):
    operator = FinitePixelRayleighSommerfeldCached(2, pair_block=2)
    source = torch.randn(1, 4, 4, 4, dtype=torch.complex128)
    operator(source, output_shape=(4, 4), fft_shape=(8, 8))
    cache = operator._get_kernel_cache((4, 4), (4, 4), (8, 8), torch.device("cpu"))
    target_ids = torch.tensor([0, 1])
    source_ids = torch.tensor([0, 1])
    original_any = torch.Tensor.any

    def reject_pair_block_reduction(tensor, *args, **kwargs):
        if tensor.dtype == torch.bool and tuple(tensor.shape) == (2, 2):
            raise AssertionError("pair transpose predicate reduced a device tensor")
        return original_any(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "any", reject_pair_block_reduction)
    with pytest.raises(AssertionError, match="pair transpose predicate"):
        bool(torch.ones((2, 2), dtype=torch.bool).any())
    operator._spectra_for_pairs(
        cache, target_ids, source_ids, (8, 8), (4, 4), (4, 4),
        has_transposed_pair=any(operator._pair_transposed_host[index][source]
                                for index in (0, 1) for source in (0, 1)),
    )


def test_cached_rs_rejects_a_wrong_precomputed_transpose_schedule():
    torch.manual_seed(1013)
    source = torch.randn(1, 4, 4, 4, dtype=torch.complex128)
    reference = FinitePixelRayleighSommerfeld(2, pair_block=2)
    cached = FinitePixelRayleighSommerfeldCached(2, pair_block=2)
    assert any(any(row) for row in cached._pair_transposed_host)
    cached._pair_transposed_host = tuple(tuple(False for _ in row)
                                         for row in cached._pair_transposed_host)

    expected = reference(source, output_shape=(4, 4), fft_shape=(8, 8))
    wrong = cached(source, output_shape=(4, 4), fft_shape=(8, 8))
    assert float(torch.linalg.vector_norm(wrong - expected) / torch.linalg.vector_norm(expected)) > 1e-6
