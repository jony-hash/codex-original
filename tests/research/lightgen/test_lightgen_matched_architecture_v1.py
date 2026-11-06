from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import scripts.research.lightgen.execute_chen_lightgen_400_mnist_baseline_v1 as chen_baseline
import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as matched_runner
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ChenLightGen400Baseline,
    LightGen400Config,
)
from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import (
    _PRODUCTION_TRANSFER_CACHE,
    angular_spectrum_propagate,
    clear_production_transfer_cache,
)
from scripts.research.lightgen.execute_lightgen_matched_architecture_v1 import (
    SmallMatchedAdapter,
    ROOT,
    _evaluate_reconstruction,
    _gradient_norms,
    _load_evaluator,
    _independent_psnr,
    _json_window_diagnostics,
    _model,
    _read_config,
    _reconstruction_rows,
    _window_check,
)


@pytest.mark.parametrize(("arm", "output_size"), [("small", 200), ("chen400", 400)])
def test_native_image_batch_matches_independent_half_pixel_reference(arm, output_size) -> None:
    pixels = torch.zeros((6, 28, 28), dtype=torch.uint8)
    pixels[1].fill_(255)
    pixels[2] = torch.arange(28, dtype=torch.int64).mul(255).div(27, rounding_mode="floor").to(torch.uint8)
    pixels[3] = torch.arange(28, dtype=torch.int64).view(28, 1).mul(255).div(27, rounding_mode="floor").to(torch.uint8)
    pixels[4] = ((torch.arange(28).view(28, 1) + torch.arange(28)) % 2).to(torch.uint8) * 255
    pixels[5, 14, 14] = 255
    original = pixels.clone()
    rows = torch.tensor([5, 0, 3, 5, 1, 2, 4], dtype=torch.long)

    actual = matched_runner._image_batch(pixels, rows, arm, torch.device("cpu"))

    selected = pixels.numpy()[rows.numpy()].astype(np.float64) / 255.0
    coordinates = (np.arange(output_size, dtype=np.float64) + 0.5) * 28.0 / output_size - 0.5
    coordinates = np.clip(coordinates, 0.0, 27.0)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, 27)
    fraction = coordinates - lower
    top = (selected[:, lower[:, None], lower[None, :]] * (1.0 - fraction[None, None, :])
           + selected[:, lower[:, None], upper[None, :]] * fraction[None, None, :])
    bottom = (selected[:, upper[:, None], lower[None, :]] * (1.0 - fraction[None, None, :])
              + selected[:, upper[:, None], upper[None, :]] * fraction[None, None, :])
    expected = (top * (1.0 - fraction[None, :, None])
                + bottom * fraction[None, :, None])[:, None]

    assert actual.shape == (7, 1, output_size, output_size)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual.min().item() >= 0.0 and actual.max().item() <= 1.0
    assert torch.equal(actual[0], actual[3])
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.all(actual[4] == 1.0)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=2e-6)
    assert torch.equal(pixels, original)


def test_row_mismatch_is_rejected_at_production_measurement_boundary() -> None:
    rows = torch.tensor([12, 25, 38], dtype=torch.long)
    _reconstruction_rows(rows, rows.clone())

    try:
        _reconstruction_rows(rows, rows.roll(1))
    except ValueError as error:
        assert "MNIST rows do not match" in str(error)
    else:
        raise AssertionError("a mispaired prediction and target row was accepted")


def test_numpy_psnr_reference_matches_direct_torch_mean() -> None:
    target = torch.linspace(0.0, 1.0, 2 * 28 * 28, dtype=torch.float32).reshape(2, 1, 28, 28)
    image = (target * 0.93 + 0.025).clamp(0.0, 1.0)
    mse = (image.double() - target.double()).square().mean((1, 2, 3)).clamp_min(1e-12)
    expected = float((-10.0 * torch.log10(mse)).double().mean())
    assert abs(_independent_psnr(image, target) - expected) < 1e-10


def test_component_gradient_calibration_recomputes_each_loss_and_matches_analytic_norms() -> None:
    class TwoStageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.decoder_phase = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    model = TwoStageModel()
    factory_calls = 0

    def make_parts():
        nonlocal factory_calls
        factory_calls += 1
        return {
            "dist": model.encoder_phase.square().sum(),
            "rec": (model.encoder_phase * model.decoder_phase).square().sum(),
            "op": model.decoder_phase.square().sum(),
        }

    norms = _gradient_norms(model, make_parts)
    encoder = model.encoder_phase.detach().double()
    decoder = model.decoder_phase.detach().double()
    expected_rec = torch.cat(
        (2.0 * encoder * decoder.square(), 2.0 * decoder * encoder.square())
    ).norm()

    assert set(norms) == {"dist", "rec", "op"}
    assert factory_calls == 3
    assert abs(norms["dist"] - (2.0 * encoder).norm().item()) < 1e-12
    assert abs(norms["rec"] - expected_rec.item()) < 1e-12
    assert abs(norms["op"] - (2.0 * decoder).norm().item()) < 1e-12
    assert all(torch.isfinite(torch.tensor(value)) and value > 0.0 for value in norms.values())


def test_gradient_calibration_rejects_a_phase_group_disconnected_from_all_losses() -> None:
    class TwoStageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.decoder_phase = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    model = TwoStageModel()
    def encoder_only_parts():
        return {
            "dist": model.encoder_phase.square().sum(),
            "rec": model.encoder_phase.pow(3).sum(),
            "op": model.encoder_phase.pow(4).sum(),
        }

    try:
        _gradient_norms(model, encoder_only_parts)
    except RuntimeError as error:
        assert "did not reach every trainable phase group" in str(error)
    else:
        raise AssertionError("an objective disconnected from the decoder phase was accepted")


def test_reconstruction_measurement_reports_full_population_psnr_by_class(monkeypatch) -> None:
    class IdentityIntensityModel(torch.nn.Module):
        def forward_from_image(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"physical_w_per_pixel": images * 0.8}

    class FixedEvaluator(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros((images.shape[0], 10), dtype=images.dtype, device=images.device)

    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    class_levels = (25 + 20 * torch.arange(10, dtype=torch.uint8)).repeat_interleave(2)
    pixels = class_levels[:, None, None].expand(-1, 28, 28).contiguous()
    dev_indices = torch.arange(labels.numel(), dtype=torch.long)

    def direct_view_batch(
        source_pixels: torch.Tensor,
        rows: torch.Tensor,
        arm: str,
        device: torch.device,
    ) -> torch.Tensor:
        del arm
        return source_pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0

    monkeypatch.setattr(matched_runner, "_image_batch", direct_view_batch)

    metrics, _ = _evaluate_reconstruction(
        IdentityIntensityModel(),
        FixedEvaluator(),
        pixels,
        labels,
        dev_indices,
        "small",
        torch.device("cpu"),
        torch.tensor(1.0),
    )

    target = pixels.unsqueeze(1).float() / 255.0
    expected_by_class = []
    for label in range(10):
        selected = labels == label
        mse = ((target[selected] * 0.8 - target[selected]) ** 2).mean()
        expected_by_class.append(float(-10.0 * torch.log10(mse)))
    assert metrics["psnr_count_by_class"] == [2] * 10
    assert torch.allclose(torch.tensor(metrics["psnr_db_by_class"]), torch.tensor(expected_by_class), atol=1e-5)


def test_small_physical_forward_and_generated_ols_path_preserve_si_output() -> None:
    model = SmallMatchedAdapter(propagation_grid_size=512).eval()
    image = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    image[:, :, 30:170, 50:150] = 0.6
    with torch.no_grad():
        encoded = model.forward_from_image(image)
        generated = model.forward_from_ols(encoded["ols_coefficients"])

        state = ComplexOLSState(
            coefficients=encoded["ols_coefficients"],
            source="test",
            basis_id=model.runtime.ols.basis_id,
            mode_geometry=model.runtime.ols.mode_identity(),
        )
        generator_input = model.runtime.ols.write_to_generator_input(
            state, model.runtime.generator_entrance_plane
        )
        generator_output, _, _ = model.runtime.generator(generator_input, "test_generated")
        generator_output = type(generator_output)(
            generator_output.tensor,
            model.runtime.generator_exit_plane,
            generator_output.provenance,
        )
        camera_field = model.runtime.output_relay(generator_output)
        pixel_area = model.runtime.camera_plane.pixel_pitch_m[0] * model.runtime.camera_plane.pixel_pitch_m[1]
        reference_w_per_pixel = camera_field.tensor.abs().square() * pixel_area

    assert encoded["physical_w_per_pixel"].shape == (1, 1, 200, 200)
    assert encoded["plane_powers_w"].shape == (1, 5)
    assert torch.isfinite(encoded["physical_w_per_pixel"]).all()
    assert torch.allclose(encoded["physical_w_per_pixel"], encoded["raw_intensity_w_per_m2"] * pixel_area)
    assert torch.allclose(generated["physical_w_per_pixel"], reference_w_per_pixel, rtol=0.0, atol=0.0)


def test_gaussian_small_config_selects_the_fixed_system_identity() -> None:
    config = matched_runner._read_config(
        ROOT / "configs/research/lightgen/small_gaussian_system_baseline_v1.yaml"
    )
    model = matched_runner._model("small", 512, torch.device("cpu"), config)

    assert isinstance(model, SmallMatchedAdapter)
    assert model.runtime.ols.basis_id == "gaussian_s12_v1"
    assert model.runtime.ols.mode_identity()["transport"] == "T=I_100 coefficient space"
    assert model.runtime.ols.mode_identity()["port_center_pitch_m"] == 20.0e-6
    assert model.runtime.ols.mode_identity()["gaussian_field_1e_radius_m"] == 3.5e-6
    assert matched_runner._model_identity(config, "small")["ols"] == "gaussian_s12_v1"


def test_gaussian_production_readouts_match_independent_numpy_sums() -> None:
    model = SmallMatchedAdapter(512, "gaussian_s12_v1").eval()
    image = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    image[:, :, 30:170, 50:150] = 0.6

    result = matched_runner._small_producer_consumer_check(model, image)

    assert result["status"] == "PASSED"
    assert result["ols_coefficient_relative_l2_error"] <= 1.0e-5
    assert result["camera_w_per_pixel_relative_l2_error"] <= 1.0e-6


def test_gaussian_ols_rejects_an_identity_state_from_the_wrong_basis() -> None:
    model = SmallMatchedAdapter(512, "gaussian_s12_v1")
    try:
        ComplexOLSState(
            coefficients=torch.zeros((1, 100), dtype=torch.complex64),
            basis_id=matched_runner.LP01_OLS_MODEL,
            mode_geometry=model.runtime.ols.mode_identity(),
        )
    except ValueError as error:
        assert "does not match its basis_id" in str(error)
    else:
        raise AssertionError("Gaussian geometry was accepted under an LP01 basis identity")


def test_gaussian_sampler_fails_closed_at_the_fixed_proposal_limit() -> None:
    fitted = {
        label: {
            "mean": torch.ones(200, dtype=torch.float64),
            "cholesky": torch.zeros((200, 200), dtype=torch.float64),
            "ridge": torch.tensor(0.0, dtype=torch.float64),
        }
        for label in range(10)
    }
    try:
        matched_runner._sample_gaussians(fitted, torch.ones(100), max_proposals_per_class=256)
    except matched_runner.SamplingPopulationIncomplete as error:
        assert "class 0 retained 0/1000" in str(error)
        assert "limit=256" in str(error)
    else:
        raise AssertionError("a deficient generated cohort passed the fixed population boundary")


def test_gaussian_quality_gate_consumes_every_registered_metric() -> None:
    config = matched_runner._read_config(
        ROOT / "configs/research/lightgen/small_gaussian_system_baseline_v1.yaml"
    )
    reconstruction = {
        "psnr_db_mean": 20.0,
        "ssim_mean": 0.80,
        "saturation_fraction": 0.01,
    }
    generation = {
        "macro_recall": 0.80,
        "recall_by_class": [0.60] * 10,
        "proxy_clipping_fraction_by_class": [0.01] * 10,
        "generated_to_dev_tv_ratio": [0.25] * 10,
    }

    assert matched_runner._quality_decision(config, reconstruction, generation)["status"] == (
        "VALID_COMPLETED_QUALITY_PASSED"
    )
    generation["recall_by_class"][0] = 0.59
    failed = matched_runner._quality_decision(config, reconstruction, generation)
    assert failed["status"] == "VALID_FAILED"
    assert failed["checks"]["generation_recall_by_class"]["passed"] is False


def test_phase_parameter_counts_match_the_registered_architectures() -> None:
    small = SmallMatchedAdapter(propagation_grid_size=512)
    large = ChenLightGen400Baseline()
    small_count = sum(parameter.numel() for parameter in (small.encoder_phase, small.decoder_phase))
    large_count = sum(parameter.numel() for parameter in (large.encoder_phase, large.decoder_phase))
    assert small_count == 40_000
    assert large_count == 960_000


def test_400_window_gate_uses_selected_grid_not_rejected_coarse_grid(monkeypatch) -> None:
    diagnostics = {
        2048: {"complex_relative_error_vs_largest_window": {"plane": 2.0e-2}},
        3072: {"complex_relative_error_vs_largest_window": {"plane": 5.0e-5}},
        4096: {"complex_relative_error_vs_largest_window": {"plane": 0.0}},
        "half_pixel_sampling_reference": {},
    }
    monkeypatch.setattr(chen_baseline, "_window_reference", lambda *args, **kwargs: (3072, diagnostics))
    config = {
        "optics": {
            "chen400": {
                "propagation_grid_candidates": [2048, 3072, 4096],
                "common_complex_roi_m": 1.5e-3,
            }
        }
    }

    selected, result = _window_check(
        "chen400",
        torch.zeros((1, 28, 28), dtype=torch.uint8),
        torch.zeros((1,), dtype=torch.long),
        torch.tensor([[0]], dtype=torch.long),
        config,
        torch.device("cpu"),
        model=torch.nn.Identity(),
    )

    assert selected == 3072
    assert result["complex_relative_error_max"] == 5.0e-5
    assert result["complex_relative_error_by_grid"][2048] == 2.0e-2
    assert result["minimum_converged_grid"] == 3072


def test_400_window_gate_uses_predeclared_production_grid(monkeypatch) -> None:
    diagnostics = {
        2048: {"complex_relative_error_vs_largest_window": {"camera": 2.0e-5}},
        3072: {"complex_relative_error_vs_largest_window": {"camera": 5.0e-5}},
        4096: {"complex_relative_error_vs_largest_window": {"camera": 1.0e-5}},
        6144: {"complex_relative_error_vs_largest_window": {"camera": 0.0}},
    }
    monkeypatch.setattr(chen_baseline, "_window_reference", lambda *args, **kwargs: (2048, diagnostics))
    config = {
        "optics": {
            "chen400": {
                "propagation_grid_candidates": [2048, 3072, 4096, 6144],
                "production_propagation_grid": 6144,
                "common_complex_roi_m": 1.5e-3,
            }
        }
    }

    selected, result = _window_check(
        "chen400",
        torch.zeros((1, 28, 28), dtype=torch.uint8),
        torch.zeros((1,), dtype=torch.long),
        torch.tensor([[0]], dtype=torch.long),
        config,
        torch.device("cpu"),
        model=torch.nn.Identity(),
    )

    assert selected == 6144
    assert result["minimum_converged_grid"] == 2048
    assert result["complex_relative_error_max"] == 0.0


def test_400_window_gate_rejects_predeclared_grid_above_tolerance(monkeypatch) -> None:
    diagnostics = {
        2048: {"complex_relative_error_vs_largest_window": {"camera": 2.0e-5}},
        3072: {"complex_relative_error_vs_largest_window": {"camera": 1.1e-4}},
        4096: {"complex_relative_error_vs_largest_window": {"camera": 0.0}},
    }
    monkeypatch.setattr(chen_baseline, "_window_reference", lambda *args, **kwargs: (2048, diagnostics))
    config = {
        "optics": {
            "chen400": {
                "propagation_grid_candidates": [2048, 3072, 4096],
                "production_propagation_grid": 3072,
                "common_complex_roi_m": 1.5e-3,
            }
        }
    }

    try:
        _window_check(
            "chen400",
            torch.zeros((1, 28, 28), dtype=torch.uint8),
            torch.zeros((1,), dtype=torch.long),
            torch.tensor([[0]], dtype=torch.long),
            config,
            torch.device("cpu"),
            model=torch.nn.Identity(),
        )
    except RuntimeError as error:
        assert "configured 400 production propagation grid failed" in str(error)
    else:
        raise AssertionError("an out-of-tolerance production grid was accepted")


def test_window_diagnostics_are_json_serializable_with_sorted_string_keys() -> None:
    diagnostics = {
        2048: {"complex_relative_error": 0.02},
        "half_pixel_sampling_reference": {"complex_relative_error": 0.0},
    }
    normalized = _json_window_diagnostics({"diagnostics": diagnostics})

    assert set(normalized["diagnostics"]) == {"2048", "half_pixel_sampling_reference"}
    json.dumps(normalized, sort_keys=True)


def test_400_model_uses_the_diagnostic_gaussian_s12_ols_contract() -> None:
    model = _model("chen400", 512, torch.device("cpu"))

    assert isinstance(model, ChenLightGen400Baseline)
    assert isinstance(model.ols, GaussianFiberArrayOLS)
    assert model.ols.diagnostic_only is True
    assert model.ols.geometry_diagnostics["normalization"] == "analytic_infinite_plane_S12"


def test_frozen_confirmation_evaluator_uses_qualified_checkpoint_state_dict() -> None:
    config = _read_config(ROOT / "configs/research/lightgen/matched_small_vs_chen400_v1.yaml")
    checkpoint = ROOT / config["evaluation"]["evaluator_checkpoint"]
    evaluator = _load_evaluator(checkpoint, torch.device("cpu"))

    assert not evaluator.training
    assert all(not parameter.requires_grad for parameter in evaluator.parameters())


def test_inplace_spectral_product_preserves_asm_forward_and_input_gradient() -> None:
    torch.manual_seed(21)
    plane = PlaneSpec(
        plane_id="matched_test_input",
        shape=(32, 32),
        pixel_pitch_m=(3.5e-6, 3.5e-6),
        wavelength_m=532e-9,
        z_m=0.0,
        coordinate_frame="matched_test_grid",
    )
    real = torch.randn((2, 1, 32, 32), dtype=torch.float32)
    imag = torch.randn((2, 1, 32, 32), dtype=torch.float32)
    source = torch.complex(real, imag)
    weights = torch.linspace(0.1, 1.0, 32 * 32).view(1, 1, 32, 32)

    clear_production_transfer_cache()
    try:
        direct_input = source.clone().requires_grad_()
        direct_field = ComplexField(
            direct_input,
            plane,
            OperationProvenance("test_field", plane.plane_id, None),
        )
        direct = angular_spectrum_propagate(direct_field, 1.5e-3, "direct")
        direct_loss = (direct.tensor.real * weights).sum() + (
            direct.tensor.imag * weights.flip(-1)
        ).sum()
        direct_gradient = torch.autograd.grad(direct_loss, direct_input)[0]

        optimized_input = source.clone().requires_grad_()
        optimized_field = ComplexField(
            optimized_input,
            plane,
            OperationProvenance("test_field", plane.plane_id, None),
        )
        optimized = matched_runner._inplace_spectral_angular_spectrum_propagate(
            optimized_field, 1.5e-3, "optimized"
        )
        optimized_loss = (optimized.tensor.real * weights).sum() + (
            optimized.tensor.imag * weights.flip(-1)
        ).sum()
        optimized_gradient = torch.autograd.grad(optimized_loss, optimized_input)[0]

        assert torch.equal(optimized.tensor, direct.tensor)
        assert torch.equal(optimized_gradient, direct_gradient)
        assert optimized.plane.z_m == direct.plane.z_m
        assert optimized.provenance.operation == direct.provenance.operation

        key, transfer = next(iter(_PRODUCTION_TRANSFER_CACHE.items()))
        assert key[0] == (32, 32)
        wrong = torch.fft.ifft2(
            torch.fft.fft2(source, norm="ortho") * transfer.conj(), norm="ortho"
        )
        assert not torch.equal(wrong, direct.tensor.detach())
    finally:
        clear_production_transfer_cache()


def test_single_per_example_checkpoint_matches_nested_layer_checkpoint_gradients() -> None:
    torch.manual_seed(43)
    nested_checkpoint_model = ChenLightGen400Baseline(
        LightGen400Config(propagation_grid_size=512, checkpoint_propagation=True)
    )
    per_example_checkpoint_model = _model("chen400", 512, torch.device("cpu"))
    assert isinstance(per_example_checkpoint_model, ChenLightGen400Baseline)
    assert per_example_checkpoint_model.config.checkpoint_propagation is False
    per_example_checkpoint_model.load_state_dict(nested_checkpoint_model.state_dict())
    nested_checkpoint_model.eval()
    per_example_checkpoint_model.eval()
    target = 0.05 + 0.9 * torch.rand((10, 1, 400, 400), dtype=torch.float32)
    scale = torch.ones(100, dtype=torch.float64)
    power_reference = torch.ones(7, dtype=torch.float64)
    camera_gain = torch.ones((), dtype=torch.float64)

    nested_parts, nested_output = matched_runner._loss_components(
        nested_checkpoint_model, target, scale, power_reference, camera_gain
    )
    nested_total = sum(nested_parts.values())
    nested_gradients = torch.autograd.grad(
        nested_total, tuple(matched_runner._phase_groups(nested_checkpoint_model))
    )
    single_parts, single_output = matched_runner._loss_components(
        per_example_checkpoint_model, target, scale, power_reference, camera_gain
    )
    single_total = sum(single_parts.values())
    single_gradients = torch.autograd.grad(
        single_total, tuple(matched_runner._phase_groups(per_example_checkpoint_model))
    )

    for name in ("dist", "rec", "op"):
        torch.testing.assert_close(single_parts[name], nested_parts[name], rtol=1e-6, atol=1e-9)
    for name in ("physical_w_per_pixel", "ols_coefficients", "plane_powers_w"):
        torch.testing.assert_close(single_output[name], nested_output[name], rtol=1e-6, atol=1e-9)
    for expected, actual in zip(nested_gradients, single_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)


def test_chen400_microbatch_preserves_logical_batch_losses_and_phase_gradients(
    monkeypatch,
) -> None:
    torch.manual_seed(37)
    model = ChenLightGen400Baseline(
        LightGen400Config(propagation_grid_size=512, checkpoint_propagation=False)
    )
    target = 0.05 + 0.9 * torch.rand((10, 1, 400, 400), dtype=torch.float32)
    direct = model.forward_from_image(target)
    direct_outputs = {
        key: direct[key].detach().clone()
        for key in ("physical_w_per_pixel", "ols_coefficients", "plane_powers_w")
    }
    direct_power = (
        direct["ols_coefficients"].real.double().square()
        + direct["ols_coefficients"].imag.double().square()
    )
    scale = direct_power.detach().mean(0).clamp_min(matched_runner.EPS_S)
    power_reference = direct["plane_powers_w"].detach().double().mean(0).clamp_min(
        matched_runner.EPS_OP
    )
    camera_gain = torch.tensor(1.0, dtype=torch.float64)

    def reference_parts(result):
        raw = result["physical_w_per_pixel"]
        reconstruction = (raw * camera_gain.to(raw) - target).square().mean()
        coefficients = result["ols_coefficients"]
        power = coefficients.real.double().square() + coefficients.imag.double().square()
        normalized = power / (scale.unsqueeze(0) + matched_runner.EPS_S)
        distribution = matched_runner.truncated_normal_positive_kl(
            matched_runner.q_from_batch(normalized),
            matched_runner.fixed_prior(device=coefficients.device),
        ).mean()
        planes = result["plane_powers_w"].double()
        optical_power = torch.log(
            (planes + matched_runner.EPS_OP)
            / (power_reference.unsqueeze(0) + matched_runner.EPS_OP)
        ).square().mean()
        return {"dist": distribution, "rec": reconstruction, "op": optical_power}

    weights = {"dist": 0.2, "rec": 0.5, "op": 0.3}
    direct_parts = reference_parts(direct)
    direct_total = sum(weights[name] * direct_parts[name] for name in weights)
    parameters = tuple(matched_runner._phase_groups(model))
    direct_gradients = torch.autograd.grad(direct_total, parameters, retain_graph=True)
    reference_norms = {}
    for index, name in enumerate(("dist", "rec", "op")):
        component_gradients = torch.autograd.grad(
            direct_parts[name],
            parameters,
            retain_graph=index < 2,
            allow_unused=True,
        )
        squared_norm = torch.zeros((), dtype=torch.float64)
        for gradient in component_gradients:
            if gradient is not None:
                squared_norm += gradient.detach().double().square().sum()
        reference_norms[name] = float(squared_norm.sqrt())
    del direct, direct_total

    seen_batch_sizes: list[int] = []
    original_forward = model.forward_from_image

    def observe_forward(images):
        seen_batch_sizes.append(int(images.shape[0]))
        return original_forward(images)

    monkeypatch.setattr(model, "forward_from_image", observe_forward)
    actual_parts, actual = matched_runner._loss_components(
        model, target, scale, power_reference, camera_gain
    )
    assert seen_batch_sizes == [1] * 10
    for key, expected in direct_outputs.items():
        torch.testing.assert_close(actual[key], expected, rtol=1e-6, atol=1e-9)
    for key, expected in direct_parts.items():
        torch.testing.assert_close(actual_parts[key], expected, rtol=1e-6, atol=1e-9)
    del direct_parts
    actual_total = sum(weights[name] * actual_parts[name] for name in weights)
    actual_gradients = torch.autograd.grad(actual_total, parameters)
    assert seen_batch_sizes == [1] * 20
    for expected, actual_gradient in zip(direct_gradients, actual_gradients, strict=True):
        assert torch.isfinite(actual_gradient).all()
        assert bool((actual_gradient.abs().sum() > 0))
        torch.testing.assert_close(actual_gradient, expected, rtol=2e-5, atol=1e-8)
    del actual_parts, actual, actual_total, actual_gradients
    actual_norms = matched_runner._chen400_streamed_gradient_norms(
        model, target, scale, power_reference, camera_gain
    )
    assert seen_batch_sizes == [1] * 90
    for name in ("dist", "rec", "op"):
        assert abs(actual_norms[name] - reference_norms[name]) <= max(
            1e-8, abs(reference_norms[name]) * 2e-5
        )
    finite_difference = matched_runner._finite_difference_check(
        model, target, scale, power_reference, camera_gain, weights
    )
    assert set(finite_difference) == {"encoder", "decoder"}
    for diagnostics in finite_difference.values():
        assert diagnostics["direction"] == "unit_l2_autograd_group_gradient"
        assert diagnostics["parameter_count"] > 0
        assert abs(diagnostics["direction_l2_norm"] - 1.0) <= 1e-6
        assert diagnostics["epsilon_radians"] == 1e-2
        assert diagnostics["autograd"] * diagnostics["central_difference"] > 0.0
        assert diagnostics["relative_error"] <= 0.05
