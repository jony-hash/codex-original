from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
import scripts.research.lightgen.execute_lightgen_small_foreground_balance_ab_v1 as foreground
import scripts.research.lightgen.execute_lightgen_small_structural_continuation_ab_v1 as continuation


def _numpy_ssim(image: np.ndarray, target: np.ndarray) -> float:
    coordinate = np.arange(-3, 4, dtype=np.float64)
    kernel_1d = np.exp(-(coordinate**2) / (2.0 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]

    def moments(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded = np.pad(value, ((3, 3), (3, 3)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (7, 7))
        mean = np.sum(windows * kernel, axis=(-2, -1))
        second = np.sum(windows**2 * kernel, axis=(-2, -1))
        return mean, second - mean**2

    score = []
    for x, y in zip(image, target, strict=True):
        mean_x, var_x = moments(x)
        mean_y, var_y = moments(y)
        padded_x = np.pad(x, ((3, 3), (3, 3)), mode="constant")
        padded_y = np.pad(y, ((3, 3), (3, 3)), mode="constant")
        windows_x = np.lib.stride_tricks.sliding_window_view(padded_x, (7, 7))
        windows_y = np.lib.stride_tricks.sliding_window_view(padded_y, (7, 7))
        cov = np.sum(windows_x * windows_y * kernel, axis=(-2, -1)) - mean_x * mean_y
        value = ((2 * mean_x * mean_y + 0.01**2) * (2 * cov + 0.03**2)) / (
            (mean_x**2 + mean_y**2 + 0.01**2) * (var_x + var_y + 0.03**2)
        )
        score.append(value.mean())
    return float(np.mean(score))


def test_structural_loss_matches_independent_numpy_and_production_operation_order() -> None:
    generator = np.random.default_rng(44)
    raw = generator.uniform(0.0, 0.9, size=(2, 1, 56, 56)).astype(np.float32)
    target = generator.uniform(0.0, 1.0, size=(2, 1, 28, 28)).astype(np.float32)
    gain = 1.8
    actual = continuation._structural_loss_from_raw(
        torch.from_numpy(raw), torch.tensor(gain), torch.from_numpy(target)
    )

    clamped = np.clip(raw * gain, 0.0, 1.0)
    pooled = clamped.reshape(2, 1, 28, 2, 28, 2).mean(axis=(3, 5))[:, 0]
    expected = 1.0 - _numpy_ssim(pooled, target[:, 0])
    wrong_order = np.clip((raw * gain).reshape(2, 1, 28, 2, 28, 2).mean(axis=(3, 5)), 0, 1)[:, 0]
    wrong_score = 1.0 - _numpy_ssim(wrong_order, target[:, 0])

    assert abs(float(actual) - expected) < 2.0e-6
    assert abs(float(actual) - wrong_score) > 1.0e-4


def test_original_bogt_is_preserved_and_treatment_adds_only_frozen_ssim_term() -> None:
    parts = {"dist": torch.tensor(2.0), "rec": torch.tensor(3.0), "op": torch.tensor(5.0)}
    weights = {"dist": 0.5, "rec": 0.25, "op": 0.75}
    original = continuation._total_loss(parts, None, weights, None)
    treatment = continuation._total_loss(parts, torch.tensor(0.4), weights, 2.0)

    assert float(original) == 5.5
    assert float(treatment) == pytest.approx(6.3, abs=1.0e-6)


@pytest.mark.parametrize(
    ("route", "branch"),
    (("foreground", "C"), ("foreground", "V"), ("structural", "control"),
     ("structural", "treatment")),
)
def test_readout_structural_objective_matches_numpy_value_and_gradient(route: str, branch: str) -> None:
    generator = torch.Generator(device="cpu").manual_seed(2026102408)
    pixels = torch.randint(16, 241, (12, 28, 28), dtype=torch.uint8, generator=generator)
    rows = torch.tensor([11, 0, 5, 2, 9, 1, 6, 10, 3, 7], dtype=torch.long)
    gain = torch.tensor(1.8, dtype=torch.float32)

    class TinyCamera(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.factor = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
            self.received: list[torch.Tensor] = []

        def forward_from_image(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            self.received.append(images.detach().clone())
            response = torch.where(images > 0.5, 1.2, 0.2)
            return {
                "physical_w_per_pixel": self.factor * response,
                "ols_coefficients": torch.ones((len(images), 100), dtype=torch.complex64),
                "plane_powers_w": torch.ones((len(images), 5), dtype=torch.float64),
            }

    model = TinyCamera()
    scale = torch.ones(100, dtype=torch.float32)
    power_reference = torch.ones(5, dtype=torch.float64)
    weights = {"dist": 0.2, "rec": 0.3, "op": 0.4}
    if route == "foreground":
        parts, structure, _result = foreground._loss_terms(
            model, pixels, rows, scale, power_reference, gain, branch,
        )
        objective = foreground._total_loss(
            parts, structure, weights, 0.25, branch, 0.6 if branch == "V" else None,
        )
    else:
        parts, structure, _result = continuation._loss_terms(
            model, pixels, rows, scale, power_reference, gain, branch,
        )
        objective = continuation._total_loss(
            parts, structure, weights, 0.25 if branch == "treatment" else None,
        )
    gradient = torch.autograd.grad(objective, model.factor)[0]

    assert len(model.received) == 1
    native = model.received[0].double().numpy()
    expected_rows = rows.numpy()
    native_targets = native
    target28 = pixels.index_select(0, rows).unsqueeze(1).float().div(255.0).numpy().astype(np.float64)
    response = np.where(native_targets > 0.5, 1.2, 0.2)
    dist_value = float(parts["dist"].detach())
    op_value = float(parts["op"].detach())
    use_structure = route == "foreground" or branch == "treatment"

    def adaptive_pool(value: np.ndarray) -> np.ndarray:
        batch, channels, height, width = value.shape
        output = np.empty((batch, channels, 28, 28), dtype=np.float64)
        for out_y in range(28):
            y0 = int(np.floor(out_y * height / 28))
            y1 = int(np.ceil((out_y + 1) * height / 28))
            for out_x in range(28):
                x0 = int(np.floor(out_x * width / 28))
                x1 = int(np.ceil((out_x + 1) * width / 28))
                output[:, :, out_y, out_x] = value[:, :, y0:y1, x0:x1].mean(axis=(2, 3))
        return output

    def reference_total(
        factor: float, *, pool_before_clamp: bool = False, detach_structure: bool = False,
    ) -> float:
        gained = factor * response * 1.8
        rec = float(np.mean((gained - native_targets) ** 2))
        if route == "foreground" and branch == "V":
            foreground_mass = native_targets.sum(axis=(1, 2, 3))
            background = 1.0 - native_targets
            background_mass = background.sum(axis=(1, 2, 3))
            squared = (gained - native_targets) ** 2
            stroke = (native_targets * squared).sum(axis=(1, 2, 3)) / foreground_mass
            background_error = (background * squared).sum(axis=(1, 2, 3)) / background_mass
            rec = 0.6 * float(0.5 * (stroke + background_error).mean())
        structural = 0.0
        if use_structure:
            structural_input = 0.7 * response * 1.8 if detach_structure else gained
            if pool_before_clamp:
                pooled = np.clip(adaptive_pool(structural_input), 0.0, 1.0)[:, 0]
            else:
                pooled = adaptive_pool(np.clip(structural_input, 0.0, 1.0))[:, 0]
            structural = 1.0 - _numpy_ssim(pooled, target28[:, 0])
        value = 0.2 * dist_value + 0.4 * op_value
        if route == "foreground" and branch == "V":
            value += 0.3 * rec
        else:
            value += 0.3 * rec
        if use_structure:
            value += 0.25 * structural
        return value

    expected = reference_total(0.7)
    epsilon = 1.0e-4
    finite_difference = (reference_total(0.7 + epsilon) - reference_total(0.7 - epsilon)) / (2.0 * epsilon)
    wrong_pool_gradient = (
        reference_total(0.7 + epsilon, pool_before_clamp=True)
        - reference_total(0.7 - epsilon, pool_before_clamp=True)
    ) / (2.0 * epsilon)
    detached_structure_gradient = (
        reference_total(0.7 + epsilon, detach_structure=True)
        - reference_total(0.7 - epsilon, detach_structure=True)
    ) / (2.0 * epsilon)

    assert torch.equal(model.received[0], base._image_batch(pixels, rows, "small", torch.device("cpu")))
    assert expected_rows.tolist() == [11, 0, 5, 2, 9, 1, 6, 10, 3, 7]
    assert float(objective.detach()) == pytest.approx(expected, rel=0.0, abs=2.0e-6)
    assert float(gradient) == pytest.approx(finite_difference, rel=1.0e-3, abs=2.0e-5)
    assert np.sign(float(gradient)) == np.sign(finite_difference)
    if use_structure:
        wrong_pool_value = reference_total(0.7, pool_before_clamp=True)
        assert abs(wrong_pool_value - expected) > 2.0e-6
        assert not np.isclose(wrong_pool_gradient, finite_difference, rtol=1.0e-3, atol=2.0e-5)
        assert not np.isclose(detached_structure_gradient, finite_difference, rtol=1.0e-3, atol=2.0e-5)
    else:
        assert wrong_pool_gradient == pytest.approx(finite_difference, rel=0.0, abs=1.0e-12)
        assert detached_structure_gradient == pytest.approx(finite_difference, rel=0.0, abs=1.0e-12)


def test_structural_lambda_is_the_one_time_gradient_norm_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    class TwoPhaseToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor(0.2))
            self.decoder_phase = torch.nn.Parameter(torch.tensor(0.4))

    model = TwoPhaseToy()

    def components(model_arg, target, scale, power_reference, camera_gain):
        del scale, power_reference, camera_gain
        value = model_arg.encoder_phase * 0.7 + model_arg.decoder_phase * 0.4
        raw = torch.ones_like(target) * (0.2 + 0.1 * value)
        parts = {"dist": value.square() + 1.0, "rec": (value - 0.8).square(), "op": value.square() + 0.5}
        return parts, {"physical_w_per_pixel": raw}

    monkeypatch.setattr(base, "_loss_components", components)
    pixels = torch.zeros((20, 28, 28), dtype=torch.uint8)
    rows = torch.arange(10)
    config = {"loss": {"lambda_multiplier": 1.0}}
    weight, details = continuation._calibrate_structural_weight(
        model, pixels, rows, torch.ones(100), torch.ones(5), torch.tensor(1.0),
        {"dist": 1.0, "rec": 2.0, "op": 1.0}, config,
    )

    assert weight == pytest.approx(details["weighted_rec_gradient_norm"] /
                                   details["unweighted_ssim_gradient_norm"])
    assert details["lambda_ssim"] == weight


def test_finite_difference_consumer_rejects_a_reversed_gradient_sign() -> None:
    with pytest.raises(RuntimeError, match="finite difference failed"):
        continuation._validate_direction_values("structural", "encoder", 1.0, -1.0)
    relative = continuation._validate_direction_values("structural", "encoder", 1.0, 1.02)
    assert relative < 0.05


def test_continuation_schedule_preserves_the_original_30000_rows_and_wraps_per_class() -> None:
    labels = torch.arange(10).repeat_interleave(6)
    train_indices = torch.arange(labels.numel())
    expected = base._schedule(labels, train_indices, continuation.SEED)
    actual = continuation._continued_schedule(labels, train_indices, continuation.SEED)

    assert torch.equal(actual[: continuation.SOURCE_UPDATES], expected)
    assert actual.shape == (continuation.TOTAL_UPDATES, 10)
    assert torch.equal(labels[actual], torch.arange(10).expand(continuation.TOTAL_UPDATES, 10))
    assert torch.equal(actual[continuation.SOURCE_UPDATES], expected[0])


def test_source_metadata_accepts_only_the_named_gaussian_step_and_basis() -> None:
    source_config = {"optics": {"small": {
        "ols_basis": "gaussian_s12_v1", "port_pitch_m": 20.0e-6,
        "field_1e_radius_m": 3.5e-6, "ols_transport": "identity",
    }}}
    identity = base._model_identity(source_config, "small")
    config = {"source": {"model_identity": identity}}
    valid = {
        "schema": "lightgen_matched_architecture_checkpoint.v1",
        "task_id": continuation.SOURCE_TASK_ID,
        "arm": "small",
        "seed": continuation.SEED,
        "completed_updates": continuation.SOURCE_UPDATES,
        "config_identity": source_config,
        "model_identity": identity,
        "evaluator_role": "frozen_confirmation_only",
        "optimizer": {"param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999),
                                         "eps": 1.0e-8, "weight_decay": 0.0}]},
    }
    continuation._validate_source_metadata(valid, config, source_config)

    wrong_basis = copy.deepcopy(valid)
    wrong_basis["model_identity"]["ols"] = "lp01_v1"
    with pytest.raises(RuntimeError, match="model identity or OLS basis"):
        continuation._validate_source_metadata(wrong_basis, config, source_config)
    wrong_step = copy.deepcopy(valid)
    wrong_step["completed_updates"] += 1
    with pytest.raises(RuntimeError, match="task, arm, seed, or step"):
        continuation._validate_source_metadata(wrong_step, config, source_config)


def test_preflight_updates_do_not_mutate_shared_source_adam_step(monkeypatch: pytest.MonkeyPatch) -> None:
    class TwoPhaseToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor(0.2))
            self.decoder_phase = torch.nn.Parameter(torch.tensor(0.4))

    model = TwoPhaseToy()
    source_optimizer = torch.optim.Adam([model.encoder_phase, model.decoder_phase])
    for parameter in (model.encoder_phase, model.decoder_phase):
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    for state in source_optimizer.state.values():
        state["step"].fill_(continuation.SOURCE_UPDATES)
    source = {
        "model": base._state_snapshot(model),
        "optimizer": source_optimizer.state_dict(),
    }
    original_source_optimizer = copy.deepcopy(source["optimizer"])
    monkeypatch.setattr(base, "_model", lambda arm, grid, device, config: TwoPhaseToy().to(device))
    monkeypatch.setattr(base, "_phase_groups", lambda model_arg: [model_arg.encoder_phase, model_arg.decoder_phase])

    faulty_model = TwoPhaseToy()
    faulty_optimizer = torch.optim.Adam([faulty_model.encoder_phase, faulty_model.decoder_phase])
    faulty_optimizer.load_state_dict(source["optimizer"])
    for parameter in base._phase_groups(faulty_model):
        parameter.grad = torch.ones_like(parameter)
    faulty_optimizer.step()
    assert {int(state["step"]) for state in source["optimizer"]["state"].values()} == {
        continuation.SOURCE_UPDATES + 1
    }
    source["optimizer"] = original_source_optimizer

    first, first_optimizer = continuation._new_branch(source, {}, torch.device("cpu"), 2048)
    for parameter in base._phase_groups(first):
        parameter.grad = torch.ones_like(parameter)
    first_optimizer.step()

    assert base._optimizer_update_count(first_optimizer, first) == continuation.SOURCE_UPDATES + 1
    assert {int(state["step"]) for state in source["optimizer"]["state"].values()} == {
        continuation.SOURCE_UPDATES
    }
    assert base._assert_close(original_source_optimizer, source["optimizer"],
                              "source_optimizer_after_preflight") == 0.0
    second, second_optimizer = continuation._new_branch(source, {}, torch.device("cpu"), 2048)
    assert base._optimizer_update_count(second_optimizer, second) == continuation.SOURCE_UPDATES


def test_resume_checkpoint_consumes_the_exact_continuation_schedule_rows() -> None:
    schedule = torch.arange(continuation.CONTINUATION_UPDATES * continuation.BATCH_SIZE).reshape(
        continuation.CONTINUATION_UPDATES, continuation.BATCH_SIZE
    )
    references = {
        "scale": torch.ones(100),
        "power_reference": torch.ones(5),
        "camera_gain": torch.tensor(1.0),
        "weights": {"dist": 1.0, "rec": 1.0, "op": 1.0},
        "structural_weight": None,
    }
    source = {"run_id": "source-run", "model_identity": {"ols": "gaussian_s12_v1"}}
    config = {"task_id": continuation.TASK_ID}
    checkpoint = {
        "schema": "lightgen_small_gaussian_structural_continuation_checkpoint.v1",
        "task_id": continuation.TASK_ID,
        "run_id": "continuation-run",
        "source_run_id": source["run_id"],
        "source_checkpoint_step": continuation.SOURCE_UPDATES,
        "branch": "control",
        "seed": continuation.SEED,
        "model_identity": source["model_identity"],
        "config_identity": config,
        "propagation_grid_size": 2048,
        "completed_updates": continuation.CONTINUATION_UPDATES,
        "completed_global_step": continuation.TOTAL_UPDATES,
        "schedule": schedule.clone(),
        "schedule_start_global_step": continuation.SOURCE_UPDATES,
        "scale": references["scale"].clone(),
        "power_reference": references["power_reference"].clone(),
        "camera_gain": references["camera_gain"].clone(),
        "weights": copy.deepcopy(references["weights"]),
        "structural_weight": None,
    }

    assert continuation._validate_continuation_checkpoint(
        checkpoint, "control", "continuation-run", source, config, schedule, references
    ) == continuation.CONTINUATION_UPDATES
    checkpoint["schedule"] = checkpoint["schedule"].clone()
    checkpoint["schedule"][0, 0] += 1
    with pytest.raises(RuntimeError, match="sample rows or schedule cursor"):
        continuation._validate_continuation_checkpoint(
            checkpoint, "control", "continuation-run", source, config, schedule, references
        )


def test_paired_bootstrap_rejects_a_missing_or_mislabelled_dev_row() -> None:
    control = [{"row": row, "label": row // 500, "ssim": 0.5} for row in range(5000)]
    treatment = [{"row": row, "label": row // 500, "ssim": 0.51} for row in range(5000)]
    result = continuation._stratified_paired_bootstrap(control, treatment, "ssim")
    assert result["observed_difference"] == pytest.approx(0.01)
    assert result["ci_95"][0] > 0.0
    with pytest.raises(RuntimeError, match="complete paired DEV population"):
        continuation._stratified_paired_bootstrap(control[:-1], treatment, "ssim")
