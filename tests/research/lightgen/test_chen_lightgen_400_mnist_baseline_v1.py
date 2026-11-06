from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    PHASE_LEVELS,
    SOURCE_POWER_W,
    ChenLightGen400Baseline,
    LightGen400Config,
)
from scripts.research.lightgen.execute_chen_lightgen_400_mnist_baseline_v1 import (
    TRAIN_SEED,
    UPDATES,
    _balanced_schedule,
    _calibrated_weights,
    _assert_state_close,
    _classification_summary,
    _complex_relative_errors,
    _gradient_norms,
    _independent_physics_checks,
    _read_config,
    _reconstruction_batch_metrics,
    _report_thresholds,
    _fit_class_gaussians,
    _sample_latents,
    _terminal_checkpoint_replay,
    _validate_modal_scale,
    _validate_adam_state,
)
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as runner
from ldm_project.v2.application.research.shared.research_run_control import (
    ResearchRunControl,
    ResearchRunIdentity,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_fixed_config_and_balanced_schedule_cover_each_train_row() -> None:
    root = Path(__file__).resolve().parents[3]
    config = _read_config(root / "configs/research/lightgen/chen_lightgen_400_mnist_baseline_v1.yaml")
    assert config["task_id"] == "CHEN_LIGHTGEN_400_MNIST_BASELINE_V1"
    assert config["data"]["official_test_reads"] == 0
    assert config["output"]["output_test_reads"] == 0
    assert config["output"]["run_dir"].endswith("/runs/attempt_09")

    labels = torch.arange(10).repeat_interleave(5_500)
    train_rows = torch.arange(labels.numel())
    rows = _balanced_schedule(labels, train_rows)
    assert rows.shape == (UPDATES, 10)
    assert torch.equal(labels[rows[0]], torch.arange(10))
    for label in range(10):
        scheduled = rows[:5_500, label]
        assert torch.equal(torch.sort(scheduled).values, torch.arange(label * 5_500, (label + 1) * 5_500))
    assert TRAIN_SEED == 2026101401


def test_six_layer_production_path_preserves_power_and_shared_ols_decoder() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, checkpoint_propagation=False)).eval()
    image = torch.zeros((1, 1, 400, 400), dtype=torch.float32)
    image[:, :, 130:270, 150:250] = 0.75
    with torch.no_grad():
        reconstruction = model.forward_from_image(image)
        generated = model.forward_from_ols(reconstruction["ols_coefficients"])

    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 960_000
    assert tuple(model.encoder_phase.shape) == (2, 1, 400, 400)
    assert tuple(model.decoder_phase.shape) == (4, 1, 400, 400)
    assert tuple(reconstruction["plane_powers_w"].shape) == (1, 7)
    assert math.isclose(float(reconstruction["input_power_w"][0]),
                        SOURCE_POWER_W * float(image.mean()), rel_tol=2.0e-5)
    assert reconstruction["ols_input_field"].plane == model.ols.input_plane
    assert reconstruction["ols_input_field"].plane.z_m == pytest.approx(6.0e-3)
    assert reconstruction["ols_output_field"].plane.z_m == pytest.approx(6.0e-3)
    assert reconstruction["generator_entry_field"].plane.z_m == pytest.approx(9.0e-3)
    assert [field.plane.z_m for field in reconstruction["decoder_fields"]] == pytest.approx(
        [12.0e-3, 15.0e-3, 18.0e-3, 21.0e-3]
    )
    assert reconstruction["camera_plane"].z_m == pytest.approx(21.0e-3)
    assert bool((reconstruction["ols_output_power_w"] <= reconstruction["ols_input_power_w"] * 1.00001).all())
    torch.testing.assert_close(generated["raw_intensity_w_per_m2"],
                               reconstruction["raw_intensity_w_per_m2"], rtol=2e-5, atol=1e-8)


def test_independent_gaussian_and_propagation_references_and_fine_sampling_keep_960k_phases() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, checkpoint_propagation=False))
    checks = _independent_physics_checks(model)
    assert checks["gaussian_unit_coefficient_absolute_error"] < 5e-6
    assert checks["independent_angular_spectrum_complex64_relative_l2"] < runner.WINDOW_TOLERANCE
    assert checks["independent_angular_spectrum_complex128_relative_l2"] < 1e-10
    fine = ChenLightGen400Baseline(LightGen400Config(1024, False, pixel_sampling_factor=2))
    assert sum(parameter.numel() for parameter in fine.parameters() if parameter.requires_grad) == 960_000
    assert tuple(fine.encoder_phase.shape) == (2, 1, 400, 400)
    quantized = model.quantized_phase_copy()
    for phase in (quantized.encoder_phase, quantized.decoder_phase):
        levels = PHASE_LEVELS.to(device=phase.device, dtype=phase.dtype)
        nearest = (phase[..., None] - levels).abs().argmin(-1)
        torch.testing.assert_close(phase, levels[nearest])


def test_latent_rejection_rate_counts_all_valid_draws_in_overshoot_batch() -> None:
    gaussian_models = {
        label: {"mean": torch.zeros(200, dtype=torch.float64),
                "cholesky": torch.zeros((200, 200), dtype=torch.float64)}
        for label in range(10)
    }
    samples, requested, rejection = _sample_latents(
        gaussian_models, torch.ones(100, dtype=torch.float64), samples_per_class=1
    )

    assert samples.shape == (10, 100)
    assert torch.equal(requested, torch.arange(10))
    for per_class in rejection.values():
        assert per_class["trials"] == 256
        assert per_class["valid_draws"] == 256
        assert per_class["retained_samples"] == 1
        assert per_class["rejection_fraction"] == 0.0


def test_latent_fit_uses_observed_uneven_train_class_populations(monkeypatch) -> None:
    counts = [5, *([4] * 9)]
    labels = torch.cat([torch.full((count,), label, dtype=torch.long) for label, count in enumerate(counts)])
    train_indices = torch.arange(labels.numel())

    class FakeModel:
        def eval(self):
            return self

        def forward_from_image(self, images):
            row = images[:, 0, 0, 0].double()
            coefficients = torch.stack(
                (
                    torch.complex(row + 1.0, row.square() + 0.25),
                    torch.complex(row.square() + 1.0, row.pow(3) + 0.5),
                ),
                dim=1,
            ).to(torch.complex64)
            return {"ols_coefficients": coefficients}

    def image_batch(_pixels, rows, device):
        images = rows.to(device=device, dtype=torch.float32).reshape(-1, 1, 1, 1)
        return images, images

    monkeypatch.setattr(runner, "TRAIN_COUNT", labels.numel())
    monkeypatch.setattr(runner, "BATCH_SIZE", 5)
    monkeypatch.setattr(runner, "OLS_PORT_COUNT", 2)
    monkeypatch.setattr(runner, "_image_batch", image_batch)
    scale = torch.ones(2, dtype=torch.float64)
    models = _fit_class_gaussians(
        FakeModel(), torch.empty(0), labels, train_indices, scale, torch.device("cpu")
    )

    all_rows = train_indices.to(torch.float32).reshape(-1, 1, 1, 1)
    expected_coefficients = FakeModel().forward_from_image(all_rows)["ols_coefficients"]
    expected_features = torch.view_as_real(
        runner._gauge_and_scale(expected_coefficients.to(torch.complex128), scale)[labels == 0]
    ).reshape(counts[0], -1)
    torch.testing.assert_close(models[0]["mean"], expected_features.mean(0))
    assert models[0]["covariance"].shape == (4, 4)
    assert int(torch.bincount(labels, minlength=10)[0]) == 5


def test_terminal_checkpoint_replay_reads_existing_exact_checkpoint(monkeypatch, tmp_path) -> None:
    class TinyReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor(0.1))
            self.decoder_phase = torch.nn.Parameter(torch.tensor(-0.2))

        def forward_from_image(self, _images):
            value = self.encoder_phase + self.decoder_phase
            field = torch.ones((1, 1, 2, 2), dtype=torch.complex64) * value.to(torch.complex64)
            return {
                "field": field,
                "ols_coefficients": torch.ones((1, 100), dtype=torch.complex64) * value.to(torch.complex64),
                "plane_powers_w": torch.ones((1, 7)) * value.square(),
                "physical_w_per_pixel": torch.ones((1, 1, 2, 2)) * value.square(),
            }

    monkeypatch.setattr(runner, "UPDATES", 1)
    monkeypatch.setattr(runner, "_field_set", lambda result: {"field": result["field"]})
    monkeypatch.setattr(
        runner,
        "_image_batch",
        lambda _pixels, _rows, _device: (torch.zeros((1, 1, 400, 400)), torch.zeros((1, 1, 400, 400))),
    )
    def save_new(path, payload):
        with Path(path).open("xb") as handle:
            torch.save(payload, handle)

    monkeypatch.setattr(runner, "atomic_torch_save_new", save_new)
    model = TinyReplayModel()
    optimizer = torch.optim.Adam(
        [model.encoder_phase, model.decoder_phase], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8
    )
    (model.encoder_phase.square() + model.decoder_phase.square()).backward()
    optimizer.step()
    checkpoint = tmp_path / "terminal_continuous_phase.pt"

    first = _terminal_checkpoint_replay(
        model, optimizer, torch.empty(0), torch.tensor([0]), torch.device("cpu"), checkpoint
    )
    saved_bytes = checkpoint.read_bytes()
    second = _terminal_checkpoint_replay(
        model, optimizer, torch.empty(0), torch.tensor([0]), torch.device("cpu"), checkpoint
    )

    assert first["passed"] and second["passed"]
    assert checkpoint.read_bytes() == saved_bytes


def test_metric_consumers_reject_mismatched_rows_and_shuffled_labels() -> None:
    target = torch.rand((4, 1, 400, 400))
    rows = torch.tensor([4, 7, 10, 12])
    view, psnr, ssim = _reconstruction_batch_metrics(target, target, rows, rows)
    assert view.shape == (4, 1, 28, 28)
    assert bool((psnr > 100).all())
    assert bool((ssim > 0.999).all())
    with pytest.raises(ValueError, match="rows do not match"):
        _reconstruction_batch_metrics(target, target.flip(0), rows, rows.flip(0))

    requested = torch.arange(10).repeat(10)
    correctly_predicted = requested.clone()
    correct = _classification_summary(requested, correctly_predicted)
    shuffled = _classification_summary(torch.roll(requested, 1), correctly_predicted)
    assert correct["macro_recall"] == 1.0
    assert shuffled["macro_recall"] == 0.0


def test_window_comparison_uses_one_global_phase_for_all_planes() -> None:
    reference = {"encoder": torch.ones(1, 1, 2, 2, dtype=torch.complex128),
                 "decoder": torch.ones(1, 1, 2, 2, dtype=torch.complex128)}
    candidate = {"encoder": reference["encoder"].clone(),
                 "decoder": reference["decoder"] * 1j}
    errors, phase = _complex_relative_errors(candidate, reference)
    assert phase == pytest.approx(math.pi / 4.0)
    assert errors["encoder"] > 0.5
    assert errors["decoder"] > 0.5


def test_production_window_reference_consumes_tensor_field_set_and_aligns_fine_samples(monkeypatch) -> None:
    class TinyWindowModel:
        def __init__(self, config):
            self.config = config
            self.encoder_phase = torch.zeros(2, 1, 1, 1)
            self.decoder_phase = torch.zeros(4, 1, 1, 1)
            self.ols = SimpleNamespace(geometry_diagnostics={"mode": "test"})

        def to(self, _device):
            return self

        def forward_from_image(self, _sample):
            side = 8 * self.config.pixel_sampling_factor
            coordinate = (
                torch.arange(side, dtype=torch.float64) + 0.5 - side / 2
            ) * self.config.pixel_pitch_m
            y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
            real = 1.0 + 1.0e4 * x + 2.0e4 * y
            imaginary = 0.2 + 5.0e3 * x - 5.0e3 * y
            plane = torch.complex(real, imaginary).to(torch.complex64)[None, None]
            return {
                "encoder_fields": [SimpleNamespace(tensor=plane), SimpleNamespace(tensor=plane)],
                "ols_output_field": SimpleNamespace(tensor=plane),
                "decoder_fields": [SimpleNamespace(tensor=plane) for _ in range(4)],
                "input_power_w": torch.tensor([1.0]),
                "ols_input_power_w": torch.tensor([1.0]),
                "ols_output_power_w": torch.tensor([1.0]),
                "plane_powers_w": torch.ones((1, 7)),
            }

    monkeypatch.setattr(runner, "ChenLightGen400Baseline", TinyWindowModel)
    model = TinyWindowModel(LightGen400Config(512, checkpoint_propagation=False))
    coarse_plane = model.forward_from_image(torch.zeros((1, 1, 8, 8)))["encoder_fields"][0].tensor
    fine_model = TinyWindowModel(LightGen400Config(1024, checkpoint_propagation=False, pixel_sampling_factor=2))
    fine_plane = fine_model.forward_from_image(torch.zeros((1, 1, 8, 8)))["encoder_fields"][0].tensor
    aligned = runner._average_fine_samples_at_coarse_pixel_centers(fine_plane)
    correct_error, _ = _complex_relative_errors({"field": aligned}, {"field": coarse_plane})
    decimated_error, _ = _complex_relative_errors({"field": fine_plane[..., ::2, ::2]}, {"field": coarse_plane})
    assert correct_error["field"] < 1.0e-6
    assert decimated_error["field"] > runner.WINDOW_TOLERANCE

    selected, diagnostics = runner._window_reference(
        model,
        torch.zeros((1, 1, 8, 8)),
        [512, 1024],
        roi_m=24.0e-6,
    )
    assert selected == 512
    assert diagnostics[512]["complex_relative_error_vs_largest_window"]["ols_output"] == 0.0
    assert diagnostics[512]["complex_relative_error_vs_half_pixel_sampling"]["ols_output"] < 1.0e-6
    assert diagnostics[512]["spatial_sampling_reference"]["acceptance_threshold"] is None


def test_recovery_adam_state_requires_fixed_groups_hyperparameters_and_step() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, checkpoint_propagation=False))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8)
    objective = model.encoder_phase.square().sum() + model.decoder_phase.square().sum()
    objective.backward()
    optimizer.step()
    _validate_adam_state(optimizer, completed_updates=1)
    optimizer.param_groups[0]["lr"] = 2.0e-3
    with pytest.raises(RuntimeError, match="hyperparameters changed"):
        _validate_adam_state(optimizer, completed_updates=1)


def test_modal_power_scale_accepts_positive_physical_values_below_epsilon() -> None:
    _validate_modal_scale(torch.tensor([6.8e-19, 2.5e-8], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="strictly positive"):
        _validate_modal_scale(torch.tensor([1.0e-9, 0.0], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="strictly positive"):
        _validate_modal_scale(torch.tensor([1.0e-9, float("nan")], dtype=torch.float64))


def test_gradient_norm_calibration_counts_unconnected_group_as_zero() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, checkpoint_propagation=False))
    with torch.no_grad():
        model.encoder_phase.fill_(0.01)
    components = {"dist": model.encoder_phase.sin().sum()}
    total_norms, group_norms = _gradient_norms(model, components)
    assert total_norms["dist"] > 0.0
    assert group_norms["dist"]["encoder_phase"] > 0.0
    assert group_norms["dist"]["decoder_phase"] == 0.0
    assert _calibrated_weights(total_norms)["dist"] == pytest.approx(3.0)

    disconnected = {"dist": torch.ones((), requires_grad=True)}
    with pytest.raises(RuntimeError, match="across all trainable groups"):
        _gradient_norms(model, disconnected)


def test_preflight_receipt_factory_and_consumer_use_shared_contract() -> None:
    identity = ResearchRunIdentity(
        task_id="RECEIPT_TEST",
        run_id="run-1",
        trajectory_id="trajectory-1",
        semantic_authority_id="authority-1",
        data_role="TRAIN",
        config_identity={},
        model_identity={},
        optimizer_identity={},
        sampler_identity={},
    )
    control = ResearchRunControl(identity, authorization_id="receipt-test-authorization")
    receipt = control.make_ready_receipt(
        correctness={"passed": True},
        resource={"passed": True},
        implementation={"passed": True, "engine": "tiny", "selected_engine": "tiny"},
        selected_engine="tiny",
    )
    control.mark_preflight_ready(receipt)
    assert control.receipt == receipt
    assert control.preflight_status == "READY"


def test_recovery_maximum_difference_includes_complex_imaginary_component() -> None:
    before = {"field": torch.zeros((1,), dtype=torch.complex64)}
    after = {"field": torch.tensor([5.0e-8j], dtype=torch.complex64)}
    assert _assert_state_close(before, after, "complex_recovery") == pytest.approx(5.0e-8)


def test_single_report_gate_consumes_all_fixed_final_thresholds() -> None:
    config = {
        "evaluation": {
            "reconstruction_psnr_db_min": 20.0,
            "reconstruction_ssim_min": 0.8,
            "generated_macro_recall_min": 0.8,
            "generated_per_class_recall_min": 0.6,
            "saturation_fraction_max": 0.01,
            "minimum_generated_to_dev_within_class_shape_distance_ratio": 0.25,
        }
    }
    reconstruction = {"psnr_db_mean": 21.0, "ssim_mean": 0.81, "saturation_fraction": 0.005}
    generated = {
        "ed_macro_recall": 0.82,
        "ed_recall_by_class": [0.7] * 10,
        "saturation_fraction_by_class": [0.005] * 10,
        "shape_distance_generated_to_dev_ratio_by_class": {str(label): 0.3 for label in range(10)},
    }
    assert _report_thresholds(config, reconstruction, generated)["passed"] is True
    generated["shape_distance_generated_to_dev_ratio_by_class"]["0"] = 0.24
    decision = _report_thresholds(config, reconstruction, generated)
    assert decision["passed"] is False
    assert decision["within_class_shape_check_passed"] is False
