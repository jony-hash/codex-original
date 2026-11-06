from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ChenLightGen400Baseline,
    LightGen400Config,
)
from scripts.research.lightgen import diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_fixed_quickcheck_population_is_canonical_balanced_and_disjoint() -> None:
    labels = torch.arange(10).repeat_interleave(100)
    train_rows = torch.arange(labels.numel())
    first = quickcheck._fixed_rows(labels, train_rows)
    second = quickcheck._fixed_rows(labels, train_rows)

    assert torch.equal(first["fit_rows_by_class"], second["fit_rows_by_class"])
    assert torch.equal(first["confirm_rows_by_class"], second["confirm_rows_by_class"])
    assert first["fit_rows_by_class"].shape == (10, 10)
    assert first["confirm_rows_by_class"].shape == (10, 20)
    assert not bool(torch.isin(first["fit_rows_by_class"].flatten(),
                               first["confirm_rows_by_class"].flatten()).any())
    assert torch.equal(labels[first["fit_rows_by_class"]], torch.arange(10).unsqueeze(1).expand(-1, 10))
    assert torch.equal(labels[first["confirm_rows_by_class"]], torch.arange(10).unsqueeze(1).expand(-1, 20))


def test_saved_ols_forward_keeps_default_output_and_exposes_matching_seven_plane_powers() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, checkpoint_propagation=False)).eval()
    image = torch.zeros((1, 1, 400, 400), dtype=torch.float32)
    image[:, :, 120:280, 145:255] = 0.7
    with torch.no_grad():
        image_result = model.forward_from_image(image)
        default = model.forward_from_ols(image_result["ols_coefficients"])
        diagnostic = model.forward_from_ols(image_result["ols_coefficients"], return_diagnostics=True)

    assert set(default) == {"raw_intensity_w_per_m2", "physical_w_per_pixel"}
    torch.testing.assert_close(default["raw_intensity_w_per_m2"], diagnostic["raw_intensity_w_per_m2"])
    torch.testing.assert_close(default["physical_w_per_pixel"], diagnostic["physical_w_per_pixel"])
    assert len(diagnostic["decoder_fields"]) == 4
    assert tuple(diagnostic["plane_powers_w"].shape) == (1, 5)
    complete_powers = torch.cat((image_result["plane_powers_w"][:, :2], diagnostic["plane_powers_w"]), dim=1)
    assert tuple(complete_powers.shape) == (1, 7)
    assert tuple(diagnostic["generator_entry_field"].tensor.shape[-2:]) == (512, 512)
    torch.testing.assert_close(image_result["raw_intensity_w_per_m2"],
                               diagnostic["raw_intensity_w_per_m2"], rtol=2.0e-5, atol=1.0e-8)
    torch.testing.assert_close(image_result["plane_powers_w"], complete_powers, rtol=2.0e-5, atol=1.0e-10)


def test_frozen_gain_counterfactual_reads_large_confirm_population_in_bounded_no_grad_batches() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self.grad_enabled: list[bool] = []

        def forward_from_ols(self, coefficients: torch.Tensor, *, return_diagnostics: bool = False):
            assert not return_diagnostics
            self.batch_sizes.append(coefficients.shape[0])
            self.grad_enabled.append(torch.is_grad_enabled())
            physical = coefficients[:, 0].real.reshape(-1, 1, 1, 1)
            return {"physical_w_per_pixel": physical, "raw_intensity_w_per_m2": physical}

    model = RecordingModel()
    coefficients = torch.zeros((200, 100), dtype=torch.complex64)
    coefficients[:, 0] = torch.arange(200)
    powers = torch.ones((200, 2), dtype=torch.float32)
    result = quickcheck._physical_from_ols_in_batches(model, coefficients, powers, batch_size=10)

    assert model.batch_sizes == [10] * 20
    assert model.grad_enabled == [False] * 20
    torch.testing.assert_close(result.flatten(), torch.arange(200, dtype=torch.float32))


def test_unit_views_keep_physical_task_and_display_values_separate() -> None:
    physical = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float64)
    task, display = quickcheck._unit_outputs(physical, torch.tensor(4.0, dtype=torch.float64))

    torch.testing.assert_close(task, torch.tensor([[[[1.0, 3.0]]]], dtype=torch.float64))
    torch.testing.assert_close(display, torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float64))
    assert math.isclose(float(physical[0, 0, 0, 1]), 0.75)


def test_finite_difference_objective_values_disable_autograd(monkeypatch) -> None:
    def losses(_model, _z, _target, _encoder_powers, _references):
        assert not torch.is_grad_enabled()
        return torch.tensor(0.0), {"rec": torch.tensor(2.0), "op": torch.tensor(3.0)}, {}

    monkeypatch.setattr(quickcheck, "_losses", losses)
    value = quickcheck._weighted_objective_value_without_grad(
        object(), torch.empty(0), torch.empty(0), torch.empty(0), {},
        {"rec": 2.0, "op": 4.0}, ("rec", "op"),
    )

    assert value == 16.0


def test_adam_direction_probe_uses_verified_gradient_without_extra_forward() -> None:
    class DecoderOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_phase = torch.nn.Parameter(torch.zeros((4, 1, 1, 1)))

    model = DecoderOnly()
    gradient = torch.tensor([1.0, -1.0, 2.0, -2.0]).reshape(4, 1, 1, 1)
    before, after = quickcheck._adam_first_step_from_gradient(model, gradient)
    delta = after - before

    assert torch.all(delta != 0)
    assert torch.all(delta * gradient < 0)


def test_initial_frozen_latent_kl_uses_all_fit_rows_in_logical_batches(monkeypatch) -> None:
    population, _, _ = _toy_fixed_population()
    original_q = quickcheck.baseline.q_from_batch
    batch_shapes: list[tuple[int, ...]] = []

    def recording_q(values: torch.Tensor):
        batch_shapes.append(tuple(values.shape))
        return original_q(values)

    monkeypatch.setattr(quickcheck.baseline, "q_from_batch", recording_q)
    scale = torch.ones(100, dtype=torch.float64)
    measured = quickcheck._initial_latent_kl(population, scale, torch.device("cpu"))

    expected = []
    for batch_index in range(10):
        positions = quickcheck._batch_indices(population, population["fit_rows_by_class"][:, batch_index])
        z = population["coefficients"].index_select(0, positions)
        power = z.real.double().square() + z.imag.double().square()
        u = power / (scale.unsqueeze(0) + quickcheck.baseline.EPS_S)
        expected.append(quickcheck.baseline.truncated_normal_positive_kl(
            original_q(u), quickcheck.baseline.fixed_prior(device=torch.device("cpu"))
        ).mean())

    assert batch_shapes == [(10, 100)] * 10
    assert math.isclose(measured, float(torch.stack(expected).mean()), rel_tol=1.0e-12, abs_tol=1.0e-12)


def _toy_fixed_population() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    labels = torch.arange(10).repeat_interleave(100)
    train_rows = torch.arange(labels.numel())
    fit_by_class = torch.stack([label * 100 + torch.arange(10) for label in range(10)])
    confirm_by_class = torch.stack([label * 100 + 10 + torch.arange(20) for label in range(10)])
    fit_rows = fit_by_class.flatten()
    confirm_rows = confirm_by_class.flatten()
    all_rows = torch.cat((fit_rows, confirm_rows))
    generator = torch.Generator().manual_seed(88)
    coefficients = torch.complex(torch.randn((300, 100), generator=generator),
                                 torch.randn((300, 100), generator=generator))
    generated = torch.complex(torch.randn((10, 100), generator=generator),
                              torch.randn((10, 100), generator=generator))
    population = {
        "fit_rows_by_class": fit_by_class, "confirm_rows_by_class": confirm_by_class,
        "fit_rows": fit_rows, "confirm_rows": confirm_rows, "all_rows": all_rows,
        "all_labels": labels[all_rows], "coefficients": coefficients,
        "encoder_power_planes_w": torch.ones((300, 2), dtype=torch.float32),
        "generated_indices": torch.arange(10) * 1000, "generated_labels": torch.arange(10),
        "generated_coefficients": generated,
    }
    return population, labels, train_rows


def test_saved_fixed_population_is_loaded_without_resampling_or_refitting(tmp_path, monkeypatch) -> None:
    population, labels, train_rows = _toy_fixed_population()
    saved = tmp_path / "fixed_population.pt"
    torch.save(population, saved)
    monkeypatch.setattr(quickcheck, "_fixed_rows", lambda *_: pytest.fail("must not resample fixed rows"))
    loaded = quickcheck._load_fixed_population(saved, labels, train_rows)
    assert loaded.keys() == population.keys()
    for key in population:
        assert torch.equal(loaded[key], population[key])


def test_fixed_generation_coefficients_match_explicit_source_artifact() -> None:
    population, _, _ = _toy_fixed_population()
    source_coefficients = torch.zeros((10_000, 100), dtype=torch.complex64)
    source_labels = torch.arange(10).repeat_interleave(1000)
    source_coefficients.index_copy_(0, population["generated_indices"], population["generated_coefficients"])
    source = {
        "schema": "chen_lightgen_400_empirical_complex_gaussian_samples.v1",
        "requested_labels": source_labels,
        "coefficients": source_coefficients,
    }

    quickcheck._validate_saved_generation_coefficients(population, source)
    source["coefficients"][0, 0] += 1.0
    with pytest.raises(RuntimeError, match="differ from the saved source generation artifact"):
        quickcheck._validate_saved_generation_coefficients(population, source)


def test_decoder_grid_model_rebuilds_geometry_and_copies_only_phase_parameters() -> None:
    source = ChenLightGen400Baseline(LightGen400Config(512, False))
    with torch.no_grad():
        source.encoder_phase.fill_(0.25)
        source.decoder_phase.fill_(0.75)
    diagnostic = quickcheck._diagnostic_model(source, 1024)

    assert diagnostic.config.propagation_grid_size == 1024
    assert diagnostic.encoder_phase.shape == source.encoder_phase.shape
    assert diagnostic.decoder_phase.shape == source.decoder_phase.shape
    torch.testing.assert_close(diagnostic.encoder_phase, source.encoder_phase)
    torch.testing.assert_close(diagnostic.decoder_phase, source.decoder_phase)
    assert diagnostic.ols.modes_x.shape[-1] == 1024
    assert source.ols.modes_x.shape[-1] == 512


def test_fixed_coefficient_window_check_never_reencodes_images(monkeypatch) -> None:
    population, labels, _ = _toy_fixed_population()
    model = ChenLightGen400Baseline(LightGen400Config(512, False))
    pixels = torch.zeros((labels.numel(), 1, 28, 28), dtype=torch.float32)
    calls: list[torch.Tensor] = []

    def fixed_z_window(_model, coefficients, candidates, _roi, *, include_sampling_reference):
        del include_sampling_reference
        calls.append(coefficients.detach().cpu().clone())
        plane_errors = {name: 5.0e-5 for name in
                        ("generator_entry", "decoder_1", "decoder_2", "decoder_3", "decoder_4")}
        return {
            "complex_relative_error_vs_largest_window": {
                candidates[0]: plane_errors, candidates[1]: {name: 0.0 for name in plane_errors}
            },
            "aligned_global_phase_vs_largest_window_rad": {candidates[0]: 0.0, candidates[1]: 0.0},
            "grid_diagnostics": {}, "spatial_sampling_reference": {"status": "MEASURED_DIAGNOSTIC_ONLY"},
        }

    monkeypatch.setattr(quickcheck, "_window_from_ols", fixed_z_window)
    monkeypatch.setattr(quickcheck, "DECODER_GRID", 4096)
    monkeypatch.setattr(quickcheck.baseline, "_window_reference",
                        lambda *_args, **_kwargs: pytest.fail("window diagnostic must not reencode images"))
    config = {"optics": {"propagation_grid_candidates": [2048, 3072, 4096, 6144],
                         "common_complex_comparison_roi_m": 1.5e-3}}
    result = quickcheck._window_checks(model, population, pixels, labels, config, torch.device("cpu"))

    assert result["passed"]
    assert result["fixed_coefficients_consumed_without_reencoding"]
    assert len(calls) == 20
    row_positions = {int(row): index for index, row in enumerate(population["all_rows"].tolist())}
    expected = []
    for label, row in enumerate(population["confirm_rows_by_class"][:, 0].tolist()):
        expected.append(population["coefficients"][row_positions[int(row)]])
        expected.append(population["generated_coefficients"][label])
    for observed, target in zip(calls, expected, strict=True):
        torch.testing.assert_close(observed[0], target)


def test_production_reconstruction_consumer_detects_shuffled_target_content() -> None:
    target = torch.rand((10, 1, 28, 28), generator=torch.Generator().manual_seed(99))
    rows = torch.arange(10)
    result = quickcheck._validate_pairing_negative_control(target, target, rows, rows)
    assert result["passed"]
    assert result["consumer_shuffled_psnr28_db"] < result["consumer_clean_psnr28_db"]
    assert result["consumer_shuffled_ssim28"] < result["consumer_clean_ssim28"]


def test_unit_artifacts_roundtrip_and_reject_wrong_saved_units(tmp_path) -> None:
    physical = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)

    def readout(task):
        return {"physical_w_per_pixel": physical, "task_intensity_unclipped": task,
                "display_intensity_clipped": task.clamp(0.0, 1.0), "camera_gain_w_inverse": 4.0}

    payloads = {"source_2048": {"unit_readout": readout(physical * 4.0)},
                "start_4096": {"unit_readout": readout(physical * 4.0)}}
    generated = {"A": {"unit_readout": readout(physical * 4.0),
                        "display_views": physical * 4.0}}
    quickcheck._write_units(tmp_path, payloads, generated)
    saved = torch.load(tmp_path / "unit_readout_reconstruction.pt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(saved["by_state"]["source_2048"]["task_intensity_unclipped"], physical * 4.0)
    saved_generation = torch.load(tmp_path / "unit_readout_generation.pt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(saved_generation["by_state"]["A"]["physical_w_per_pixel"], physical)

    bad_payloads = {"source_2048": {"unit_readout": readout(physical * 3.0)}}
    (tmp_path / "bad").mkdir()
    with pytest.raises(RuntimeError, match="does not equal physical W/pixel times gain"):
        quickcheck._write_units(tmp_path / "bad", bad_payloads,
                                {"A": {"unit_readout": readout(physical * 4.0)}})


def test_resume_routing_covers_a_b_and_evaluation_interruptions() -> None:
    assert quickcheck._resume_scenario_checks()["a_interrupted"]["passed"]
    assert quickcheck._resume_decision("a_reconstruction_only", 500, set()) == (
        "train_arm", "b_reconstruction_plus_power"
    )
    assert quickcheck._resume_scenario_checks()["a_complete_b_interrupted"]["passed"]
    assert quickcheck._resume_scenario_checks()["b_complete_evaluation_interrupted"]["passed"]
    with pytest.raises(RuntimeError, match="before completed A endpoint"):
        quickcheck._resume_decision("b_reconstruction_plus_power", 100, set())


def test_resume_cli_requires_an_explicit_checkpoint_and_accepts_4096_profile(monkeypatch) -> None:
    base = ["diagnose", "--mode", "resume", "--decoder-grid", "4096", "--checkpoint", "checkpoint.pt",
            "--source-report", "report.json", "--fixed-population", "fixed_population.pt",
            "--generated-latents", "generated_ols_samples.pt",
            "--preflight-dir", "preflight", "--output-dir", "attempt"]
    try:
        monkeypatch.setattr(sys, "argv", base)
        with pytest.raises(RuntimeError, match="explicit --resume-checkpoint"):
            quickcheck._parse_args()

        monkeypatch.setattr(sys, "argv", base + ["--resume-checkpoint", "attempt/A/checkpoint_signal_step_000143.pt"])
        parsed = quickcheck._parse_args()
        assert parsed.mode == "resume"
        assert parsed.decoder_grid == 4096
        assert parsed.resume_checkpoint.name == "checkpoint_signal_step_000143.pt"
        assert quickcheck.TASK_ID == quickcheck.TASK_ID_4096
    finally:
        quickcheck._activate_profile(2048)


def test_recovery_payload_with_wrong_decoder_grid_is_rejected_before_state_load(monkeypatch) -> None:
    quickcheck._activate_profile(4096)
    try:
        with pytest.raises(RuntimeError, match="different task, run, arm, or decoder grid"):
            quickcheck._validate_training_payload(
                {"schema": "chen_lightgen_400_quickcheck_recovery.v2", "task_id": quickcheck.TASK_ID,
                 "run_id": "expected", "arm": "a_reconstruction_only", "decoder_grid": 2048},
                "a_reconstruction_only", "expected", Path("run"), Path("preflight"), {}, {}, {}, {},
            )
    finally:
        quickcheck._activate_profile(2048)


def test_training_update_changes_all_decoder_masks_and_preserves_frozen_encoder() -> None:
    model = ChenLightGen400Baseline(LightGen400Config(512, checkpoint_propagation=False)).eval()
    intensity = torch.zeros((1, 1, 400, 400), dtype=torch.float32)
    intensity[:, :, 130:270, 155:245] = 0.6
    with torch.no_grad():
        encoded = model.forward_from_image(intensity)
    reference_powers = encoded["plane_powers_w"][0].detach().clone()
    references = {
        "scale": torch.ones(100, dtype=torch.float64),
        "power_reference": reference_powers.double(),
        "camera_gain": torch.tensor(1.0, dtype=torch.float64),
    }
    before_encoder = model.encoder_phase.detach().clone()
    before_decoder = model.decoder_phase.detach().clone()
    optimizer = quickcheck._arm_optimizer(model)
    values = quickcheck._train_update(
        model, optimizer, encoded["ols_coefficients"], intensity,
        encoded["plane_powers_w"][:, :2], references,
        {"rec": 2.806518986937987, "op": 0.1933624529189577},
        "b_reconstruction_plus_power",
    )

    assert all(math.isfinite(values[key]) for key in ("total", "l_rec", "l_op", "decoder_gradient_norm"))
    assert all(value > 0.0 for value in values["decoder_layer_max_update"])
    torch.testing.assert_close(model.encoder_phase, before_encoder)
    assert not torch.equal(model.decoder_phase, before_decoder)
