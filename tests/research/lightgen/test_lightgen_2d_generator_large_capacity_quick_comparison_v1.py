from __future__ import annotations

import torch
import pytest
from torch import nn

from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import LightGen2DForwardResult
from ldm_project.v2.contracts import ComplexField, IntensityFrame, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.propagation import angular_spectrum_propagate
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as task


def _field(tensor: torch.Tensor, *, z_m: float = 0.0, plane_id: str = "generator") -> ComplexField:
    plane = PlaneSpec(
        plane_id=plane_id,
        shape=tuple(tensor.shape[-2:]),
        pixel_pitch_m=(2.0e-6, 2.0e-6),
        wavelength_m=532.0e-9,
        z_m=z_m,
        coordinate_frame="generator",
    )
    return ComplexField(tensor, plane, OperationProvenance("test_field", plane_id, None))


def _small_generator_pair(dtype: torch.dtype = torch.float32):
    torch.manual_seed(71)
    old = torch.randn((2, 1, 4, 4), dtype=dtype) * 0.2
    small = FiniteCropPhaseStack(layers=2, active_shape=(4, 4), propagation_shape=(64, 64), spacing_m=1.5e-3)
    with torch.no_grad():
        small.phase.copy_(old)
    large = task.ExpandedGenerator(
        old,
        propagation_grid_size=64,
        active_size=4,
        expanded_size=12,
        half_spacing_m=0.75e-3,
    )
    return small, large


def _baseline(small: nn.Module, field: ComplexField) -> torch.Tensor:
    output, _records, _stats = small(field, "small")
    return output.tensor


def _expanded(large: task.ExpandedGenerator, field: ComplexField) -> tuple[torch.Tensor, tuple[ComplexField, ...]]:
    output, records, _stats = large(field, "large")
    return output.tensor, records


def _faulty_intermediate_crop(field: ComplexField, phase: torch.Tensor, *, grid: int = 64) -> torch.Tensor:
    """Independent counterexample that inserts the forbidden z=0.75 mm crop."""
    active = 4
    padding = (grid - active) // 2
    flat_phase = phase.reshape(-1)
    phase0 = flat_phase[: active * active].view(1, active, active)
    transfer0 = torch.polar(torch.ones_like(phase0), phase0).to(field.tensor.dtype)
    first = field.tensor * transfer0
    plane = field.plane.with_plane("fault_pad_1", shape=(grid, grid), coordinate_frame="fault_grid")
    padded = ComplexField(torch.nn.functional.pad(first, (padding,) * 4), plane, OperationProvenance("finite_crop_center_pad", field.plane.plane_id, None))
    propagated = angular_spectrum_propagate(padded, 0.75e-3, "fault_075mm")
    intermediate_crop = propagated.tensor[..., padding : padding + active, padding : padding + active]
    recropped = ComplexField(intermediate_crop, field.plane.with_plane("fault_crop_075mm", z_m=propagated.plane.z_m), OperationProvenance("finite_crop_hard_aperture", propagated.plane.plane_id, 0.75e-3))
    padded2 = ComplexField(torch.nn.functional.pad(recropped.tensor, (padding,) * 4), recropped.plane.with_plane("fault_pad_2", shape=(grid, grid), coordinate_frame="fault_grid"), OperationProvenance("finite_crop_center_pad", recropped.plane.plane_id, None))
    propagated2 = angular_spectrum_propagate(padded2, 0.75e-3, "fault_150mm")
    active2 = propagated2.tensor[..., padding : padding + active, padding : padding + active]
    phase1 = flat_phase[active * active : 2 * active * active].view(1, active, active)
    second = active2 * torch.polar(torch.ones_like(phase1), phase1).to(active2.dtype)
    padded3 = ComplexField(torch.nn.functional.pad(second, (padding,) * 4), propagated2.plane.with_plane("fault_pad_3", shape=(grid, grid), coordinate_frame="fault_grid"), OperationProvenance("finite_crop_center_pad", propagated2.plane.plane_id, None))
    propagated3 = angular_spectrum_propagate(padded3, 1.5e-3, "fault_300mm")
    return propagated3.tensor[..., padding : padding + active, padding : padding + active]


def test_zero_added_phase_matches_original_path_and_rejects_midplane_crop_counterexample() -> None:
    small, large = _small_generator_pair()
    field = _field(torch.complex(torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)))
    expected = _baseline(small, field)
    actual, records = _expanded(large, field)
    assert len(records) == 2
    assert records[0].plane.z_m == 1.5e-3
    assert records[1].plane.z_m == 3.0e-3
    assert torch.equal(actual, expected)
    assert torch.allclose(actual, expected, rtol=task.RTOL, atol=task.ATOL)

    faulty = _faulty_intermediate_crop(field, small.phase.detach())
    assert not torch.allclose(faulty, expected, rtol=task.RTOL, atol=task.ATOL)


def test_new_phase_autograd_matches_independent_central_difference() -> None:
    small, large = _small_generator_pair(torch.float64)
    del small
    field = _field(torch.complex(torch.randn(1, 1, 4, 4, dtype=torch.float64), torch.randn(1, 1, 4, 4, dtype=torch.float64)))
    weights = torch.linspace(0.2, 1.4, 16, dtype=torch.float64).view(1, 1, 4, 4)
    output, _records, _stats = large(field, "large")
    loss = (output.tensor.abs().square() * weights).sum()
    gradient = torch.autograd.grad(loss, large.phase)[0]
    parameter_index = large.old_phase_count + 5
    analytic = float(gradient[parameter_index].item())
    epsilon = 1.0e-4
    original = float(large.phase[parameter_index].item())
    with torch.no_grad():
        large.phase[parameter_index] = original + epsilon
        plus = float((large(field, "plus")[0].tensor.abs().square() * weights).sum().item())
        large.phase[parameter_index] = original - epsilon
        minus = float((large(field, "minus")[0].tensor.abs().square() * weights).sum().item())
        large.phase[parameter_index] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert abs(analytic) > 1.0e-8
    assert abs(analytic - finite_difference) <= 2.0e-5 + 2.0e-4 * abs(finite_difference)


def test_large_layout_has_exact_17x_generator_parameter_count_and_zero_added_start() -> None:
    old = torch.randn((2, 1, 100, 100), dtype=torch.float32)
    model = task.ExpandedGenerator(old)
    assert model.phase.shape == (340_000,)
    assert model.phase[:20_000].reshape_as(old).equal(old)
    assert torch.count_nonzero(model.phase[20_000:]).item() == 0
    assert model.phase_layout["new_phase_support_fov_m"] == pytest.approx([800e-6, 800e-6])


def test_training_batches_reject_holdout_roles_and_preserve_fixed_order() -> None:
    labels = torch.cat(
        (
            torch.arange(10).repeat(20),
            torch.arange(10).repeat(10),
            torch.arange(10).repeat_interleave(30),
        )
    ).to(torch.long)
    images = torch.zeros((600, 1, 28, 28))
    first, first_labels, positions = task.training_batch(images, labels, 0)
    assert first.shape == (10, 1, 28, 28)
    assert first_labels.tolist() == list(range(10))
    assert positions.tolist() == list(range(10))
    with pytest.raises(task.LargeCapacityRunError, match="only canonical TRAIN"):
        task.training_batch(images, labels, 0, data_role="confirmation")
    with pytest.raises(task.LargeCapacityRunError, match="outside the fixed 20"):
        task.training_batch(images, labels, 20)


def test_generator_adam_is_fresh_and_scoped_to_one_parameter() -> None:
    phase = nn.Parameter(torch.zeros(340_000))
    optimizer = task._new_generator_optimizer(phase)
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["params"] == [phase]
    assert optimizer.param_groups[0]["lr"] == task.GENERATOR_LEARNING_RATE
    assert tuple(optimizer.param_groups[0]["betas"]) == task.GENERATOR_BETAS
    assert optimizer.param_groups[0]["eps"] == task.GENERATOR_EPS


def test_joint_semantic_normalization_includes_all_generator_coordinates() -> None:
    encoder = torch.ones((2, 1, 2, 2), dtype=torch.float64)
    generator = torch.full((340_000,), 2.0, dtype=torch.float64)
    gradients = {
        name: {"encoder.phase": encoder.clone(), "generator.phase": generator.clone()}
        for name in ("L_S", "L_I", "L_C")
    }
    actual, unavailable = diagnostic._safe_canonical_gradient(gradients)
    assert unavailable is None
    assert actual is not None
    vectors = []
    for name in ("L_S", "L_I", "L_C"):
        flat = torch.cat((gradients[name]["encoder.phase"].reshape(-1), gradients[name]["generator.phase"].reshape(-1)))
        vectors.append(flat / flat.norm())
    joint = sum(vectors)
    expected = joint / joint.square().mean().sqrt() * diagnostic.DELTA_SIC
    actual_flat = torch.cat((actual["encoder.phase"].reshape(-1), actual["generator.phase"].reshape(-1)))
    assert torch.allclose(actual_flat, expected, rtol=1e-12, atol=1e-12)
    without_large_tail = joint[: encoder.numel()]
    assert not torch.allclose(actual_flat[: encoder.numel()], without_large_tail / without_large_tail.square().mean().sqrt() * diagnostic.DELTA_SIC)


def test_microbatch_proxy_preserves_global_batch_coupled_bogt_gradient() -> None:
    torch.manual_seed(1337)
    theta = torch.tensor([0.18, -0.11], dtype=torch.float64, requires_grad=True)
    features = torch.randn((10, 100), dtype=torch.float64)
    offset = torch.linspace(-0.2, 0.3, 100, dtype=torch.float64).view(1, -1)
    scale = torch.linspace(0.4, 1.6, 100, dtype=torch.float64)

    def coefficients(value: torch.Tensor) -> torch.Tensor:
        real = value[0] * features + offset
        imag = value[1] * features.flip(0) - 0.3 * offset
        return torch.complex(real, imag)

    def full_batch_loss(value: torch.Tensor) -> torch.Tensor:
        u = bogt.statistical_variable(coefficients(value), scale)
        q = bogt.q_from_batch(u)
        prior = bogt.fixed_prior(device=value.device, dtype=torch.float64)
        return bogt.truncated_normal_positive_kl(q, prior).mean()

    full_loss = full_batch_loss(theta)
    full_gradient = torch.autograd.grad(full_loss, theta)[0]
    detached_coefficients = coefficients(theta.detach())
    proxy_loss, d_loss_d_ols = task._dist_loss_and_ols_gradient(detached_coefficients, scale)
    assert proxy_loss.item() == pytest.approx(full_loss.item(), rel=1e-13, abs=1e-13)

    microbatch_gradient = torch.zeros_like(theta)
    for start in range(0, 10, task.LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(10, start + task.LARGE_GRADIENT_MICROBATCH_SIZE)
        z_chunk = coefficients(theta)[start:stop]
        proxy = torch.real(torch.conj(d_loss_d_ols[start:stop]) * z_chunk).sum()
        microbatch_gradient += torch.autograd.grad(proxy, theta, retain_graph=stop < 10)[0]
    assert torch.allclose(microbatch_gradient, full_gradient, rtol=2e-11, atol=2e-11)

    epsilon = 1e-5
    with torch.no_grad():
        plus = theta.detach().clone()
        minus = theta.detach().clone()
        plus[0] += epsilon
        minus[0] -= epsilon
    finite_difference = float((full_batch_loss(plus) - full_batch_loss(minus)).item() / (2.0 * epsilon))
    assert full_gradient[0].item() == pytest.approx(finite_difference, rel=2e-5, abs=2e-7)

    faulty_gradient = torch.zeros_like(theta)
    for start in range(0, 10, task.LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(10, start + task.LARGE_GRADIENT_MICROBATCH_SIZE)
        z_chunk = coefficients(theta)[start:stop]
        wrong_batch_gradient = d_loss_d_ols.mean(dim=0, keepdim=True).expand_as(z_chunk)
        wrong_proxy = torch.real(torch.conj(wrong_batch_gradient) * z_chunk).sum()
        faulty_gradient += torch.autograd.grad(wrong_proxy, theta, retain_graph=stop < 10)[0]
    assert not torch.allclose(faulty_gradient, full_gradient, rtol=1e-4, atol=1e-7)


def test_five_plane_power_contract_keeps_original_bogt_plane_order() -> None:
    old, expanded = _small_generator_pair()
    source = _field(torch.complex(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)))
    _out, generator_records, _stats = expanded(source, "generator")
    encoder_records = (
        _field(torch.ones((1, 1, 4, 4), dtype=torch.complex64), z_m=-1e-3, plane_id="encoder_1"),
        _field(torch.full((1, 1, 4, 4), 2.0, dtype=torch.complex64), z_m=0.0, plane_id="encoder_2"),
    )
    result = LightGen2DForwardResult(
        raw_camera=IntensityFrame(torch.ones((1, 1, 4, 4)), source.plane, source.provenance),
        z_ols=type("OLS", (), {"coefficients": torch.zeros((1, 100), dtype=torch.complex64)})(),
        generator_input_field=_field(torch.full((1, 1, 4, 4), 3.0, dtype=torch.complex64)),
        generator_output_field=generator_records[-1],
        image_input=type("ImageInput", (), {"slm_intensity": torch.ones((1, 1, 4, 4))})(),
        encoder_output_field=encoder_records[-1],
        ols_input_field=encoder_records[-1],
        encoder_records=encoder_records,
        generator_records=generator_records,
        crop_statistics={},
    )
    actual = bogt.contracted_plane_powers(result)
    expected_fields = (*encoder_records, result.generator_input_field, *generator_records)
    expected = torch.stack([bogt.physical_plane_power(value) for value in expected_fields], dim=1)
    assert actual.shape == (1, 5)
    assert torch.equal(actual, expected)
    assert generator_records[0].plane.z_m == 1.5e-3
    assert generator_records[1].plane.z_m == 3.0e-3
    del old


def _role_labels() -> torch.Tensor:
    return torch.cat(
        (
            torch.arange(10).repeat(20),
            torch.arange(10).repeat(10),
            torch.arange(10).repeat_interleave(30),
        )
    ).to(torch.long)


def _predictions(labels: torch.Tensor, counts: list[int]) -> torch.Tensor:
    prediction = labels.clone()
    start = task.FIT_ROWS + task.HISTORIC_ROWS
    for class_id in range(10):
        class_rows = torch.nonzero(labels[start:] == class_id, as_tuple=False).flatten() + start
        prediction[class_rows[counts[class_id] :]] = (class_id + 1) % 10
    return prediction


def _summary(labels: torch.Tensor, predictions: torch.Tensor, mse: float = 0.1) -> dict[str, object]:
    logits = torch.nn.functional.one_hot(predictions, num_classes=10).to(torch.float32)
    metrics = task.semantic._class_metrics(labels, logits)
    by_role = {}
    for role in ("fit", "historic_holdout", "confirmation"):
        by_role[role] = {
            "overall": {"raw_mse": mse, "raw_fraction_above_one": 0.0, "raw_zero_image_count": 0, "raw_constant_image_count": 0},
            **{
                str(class_id): {"raw_mse": mse, "raw_fraction_above_one": 0.0, "raw_zero_image_count": 0, "raw_constant_image_count": 0}
                for class_id in range(10)
            },
        }
    return {
        "evaluators": {"E_D": metrics},
        "image_metrics_by_role": by_role,
        "all_finite": True,
        "has_zero_image": False,
        "has_constant_image": False,
    }


def test_final_decision_consumes_confirmation_correct_counts_and_image_risks() -> None:
    labels = _role_labels()
    start_pred = _predictions(labels, [10] * 10)
    improved_pred = _predictions(labels, [12, 13, 12, 12, 12, 12, 12, 13, 12, 12])
    summaries = {
        "A_start": _summary(labels, start_pred),
        "B_start": _summary(labels, start_pred),
        "A_step0300": _summary(labels, start_pred),
        "B_step0300": _summary(labels, improved_pred),
    }
    decision = task._final_decision(labels, summaries, {}, frozen_valid=True)
    assert decision["status"] == "VALID_COMPLETED"
    assert decision["classification"] == "QUICK_CAPACITY_GAIN_SUPPORTED"
    assert decision["next_action"] == "NONE"

    summaries["B_step0300"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_fraction_above_one"] = 0.02
    decision = task._final_decision(labels, summaries, {}, frozen_valid=True)
    assert decision["status"] == "VALID_FAILED"
    assert decision["training_recommendation"] == "NONE"


def test_paired_correctness_uses_the_production_readout_key_and_confirmation_rows() -> None:
    labels = _role_labels()
    base_predictions = labels.clone()
    changed_predictions = labels.clone()
    start = task.FIT_ROWS + task.HISTORIC_ROWS
    class_one_rows = torch.nonzero(labels[start:] == 1, as_tuple=False).flatten() + start
    changed_predictions[class_one_rows[:2]] = 2
    base = {"e_d_logits": torch.nn.functional.one_hot(base_predictions, num_classes=10).float()}
    changed = {"e_d_logits": torch.nn.functional.one_hot(changed_predictions, num_classes=10).float()}
    output = task._paired_correctness(base, changed, labels)
    assert output["1"] == {
        "rows": 30,
        "correct_to_correct": 28,
        "wrong_to_correct": 0,
        "correct_to_wrong": 2,
        "wrong_to_wrong": 0,
        "predicted_class_transitions": {"1->1": 28, "1->2": 2},
    }
