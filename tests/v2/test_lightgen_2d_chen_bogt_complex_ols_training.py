"""Focused contract, numerical, and bounded-resume checks for Chen BOGT."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import lightgen_2d_chen_bogt_complex_ols_training as chen


def _images() -> torch.Tensor:
    return torch.linspace(0.05, 0.95, 10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28)


def _theta0_identity() -> dict[str, str]:
    return {"state_identity_sha256": "fixture-theta0"}


def _fixture_references(consumer: chen.LightGen2DChenBOGTComplexOLSTrainingConsumer) -> chen.FrozenBogtReferences:
    return chen.FrozenBogtReferences(
        torch.ones(100, dtype=torch.float64),
        torch.ones(5, dtype=torch.float64),
        chen.SYNTHETIC_ROLE,
        10,
        True,
        consumer.initialization_fingerprint(),
        {"source": "unit fixture"},
    )


def _prepared_consumer() -> tuple[
    chen.LightGen2DChenBOGTComplexOLSTrainingConsumer,
    chen.FrozenBogtReferences,
    chen.BogtLossWeights,
    torch.Tensor,
    chen.DataReadLedger,
]:
    images = _images()
    ledger = chen.DataReadLedger()
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    references = chen.collect_bogt_train_references(
        consumer,
        images,
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
        source_indices=torch.arange(10, dtype=torch.long),
        data_identity={"source": "unit fixture"},
        ledger=ledger,
    )
    weights = chen.calibrate_initial_gradient_weights(
        consumer,
        images,
        references,
        data_role=chen.SYNTHETIC_ROLE,
        calibration_batch_identity={
            "source": "unit fixture",
            "seed": chen.SCHEDULE_SEED,
            "batch_size": chen.LOGICAL_BATCH_SIZE,
            "class_balanced": True,
        },
        engineering_only=True,
        ledger=ledger,
    )
    consumer.bind_weights(weights)
    return consumer, references, weights, images, ledger


def _phase_fixture() -> tuple[torch.Tensor, chen.FrozenBogtReferences, torch.Tensor]:
    references = chen.FrozenBogtReferences(
        torch.ones(100, dtype=torch.float64),
        torch.ones(5, dtype=torch.float64),
        chen.SYNTHETIC_ROLE,
        24,
        True,
        _theta0_identity(),
        {"source": "phase unit fixture"},
    )
    rows = 24
    u = torch.full((rows, 100), 0.25, dtype=torch.float64)
    u[:, 0] = torch.linspace(2.0, 3.5, rows, dtype=torch.float64)
    u[:, 1] = torch.linspace(0.7, 1.7, rows, dtype=torch.float64)
    phase = torch.zeros((rows, 100), dtype=torch.float64)
    phase[:, 1] = 0.1 * torch.log(u[:, 0])
    phase[:, 2] = -0.2 * torch.log(u[:, 1])
    phase[:, 3] = 0.05 * torch.log(u[:, 0] * u[:, 1])
    amplitude = u.sqrt()
    z = torch.complex(amplitude * phase.cos(), amplitude * phase.sin()).to(torch.complex64)
    return z, references, u


def test_parameter_firewall_and_optimizer_contract() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    named = list(consumer.runtime.named_parameters())
    assert [name for name, _ in named] == list(chen.ALLOWED_TRAINABLE_PARAMETER_NAMES)
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for _, parameter in named)
    assert all(not buffer.requires_grad for buffer in consumer.runtime.buffers())
    manifest = consumer.parameter_manifest()
    assert manifest["optimizer_parameter_names"] == list(chen.ALLOWED_TRAINABLE_PARAMETER_NAMES)
    assert manifest["parameter_shapes"] == {name: list(shape) for name, shape in chen.EXPECTED_PARAMETER_SHAPES.items()}

    optimizer = consumer.create_optimizer()
    optimizer_manifest = consumer.optimizer_manifest(optimizer)
    assert optimizer_manifest["name"] == "Adam"
    assert optimizer_manifest["learning_rate"] == chen.LEARNING_RATE
    assert tuple(optimizer_manifest["betas"]) == chen.ADAM_BETAS
    assert optimizer_manifest["weight_decay"] == 0.0
    assert optimizer_manifest["amsgrad"] is False

    with pytest.raises(chen.TrainingContractError):
        consumer.create_optimizer([parameter for _, parameter in named] + [torch.nn.Parameter(torch.ones(1))])


def test_statistical_variable_is_normalized_modal_power_and_gauge_invariant() -> None:
    scale = torch.arange(1, 101, dtype=torch.float64)
    references = chen.FrozenBogtReferences(
        scale,
        torch.ones(5, dtype=torch.float64),
        chen.SYNTHETIC_ROLE,
        2,
        True,
        _theta0_identity(),
        {"source": "u unit fixture"},
    )
    z = torch.zeros((2, 100), dtype=torch.complex64)
    z[0, 0] = torch.complex(torch.tensor(2.0), torch.tensor(0.0))
    z[0, 1] = torch.complex(torch.tensor(0.0), torch.tensor(3.0))
    z[1, 0] = torch.complex(torch.tensor(4.0), torch.tensor(0.0))
    z[1, 1] = torch.complex(torch.tensor(0.0), torch.tensor(5.0))
    u = chen.statistical_variable(z, references)
    assert u[0, 0].item() == pytest.approx(4.0)
    assert u[0, 1].item() == pytest.approx(9.0 / 2.0)
    angle = torch.tensor(0.63)
    rotation = torch.complex(angle.cos(), angle.sin())
    assert torch.allclose(u, chen.statistical_variable(z * rotation, references))


def _numeric_tn_positive_kl(q_loc: float, q_variance: float, p_loc: float, p_variance: float) -> float:
    dtype = torch.float64
    q_sigma = math.sqrt(q_variance)
    p_sigma = math.sqrt(p_variance)
    upper = max(50.0, q_loc + 14.0 * q_sigma, p_loc + 14.0 * p_sigma)
    x = torch.linspace(0.0, upper, 120_001, dtype=dtype)
    sqrt_two = math.sqrt(2.0)
    q_z = 0.5 * (1.0 + math.erf(q_loc / (q_sigma * sqrt_two)))
    p_z = 0.5 * (1.0 + math.erf(p_loc / (p_sigma * sqrt_two)))
    q_log_density = (
        -0.5 * ((x - q_loc) / q_sigma).square()
        - math.log(q_sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(q_z)
    )
    p_log_density = (
        -0.5 * ((x - p_loc) / p_sigma).square()
        - math.log(p_sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(p_z)
    )
    return float(torch.trapezoid(torch.exp(q_log_density) * (q_log_density - p_log_density), x))


@pytest.mark.parametrize(
    ("q_loc", "q_variance", "p_loc", "p_variance"),
    (
        (0.0, 0.25, 1.0, 0.25),
        (0.5, 0.5, 1.0, 0.25),
        (1.0, 0.01, 1.0, 0.25),
        (1.3, 0.5, 1.0, 0.25),
        (1.0, 0.25, 1.0, 0.25),
        (3.0, 4.0, 1.0, 0.25),
    ),
)
def test_positive_truncated_normal_kl_matches_independent_quadrature(
    q_loc: float,
    q_variance: float,
    p_loc: float,
    p_variance: float,
) -> None:
    analytic = chen.tn_positive_kl(
        torch.tensor(q_loc),
        torch.tensor(q_variance),
        torch.tensor(p_loc),
        torch.tensor(p_variance),
    )
    numeric = _numeric_tn_positive_kl(q_loc, q_variance, p_loc, p_variance)
    assert torch.isfinite(analytic)
    assert float(analytic) >= -1.0e-9
    assert float(analytic) == pytest.approx(numeric, abs=3.0e-4)


def test_q_uses_population_moments_and_variance_floor() -> None:
    constant = torch.full((10, 100), 2.0, dtype=torch.float64)
    moments = chen.q_batch_moments(constant)
    assert torch.equal(moments.mean, torch.full((100,), 2.0, dtype=torch.float64))
    assert torch.equal(moments.variance, torch.full((100,), chen.VARIANCE_FLOOR, dtype=torch.float64))

    varying = constant.clone()
    varying[:, 0] = torch.arange(10, dtype=torch.float64)
    varying_moments = chen.q_from_batch(varying)
    assert varying_moments.loc[0].item() == pytest.approx(4.5)
    assert varying_moments.variance[0].item() == pytest.approx(8.25)


def test_optical_power_loss_uses_five_plane_relative_log_power() -> None:
    reference = torch.ones(5, dtype=torch.float64)
    equal = torch.ones((2, 5), dtype=torch.float64)
    assert chen.optical_power_loss_from_powers(equal, reference).item() == pytest.approx(0.0, abs=1.0e-12)
    high = torch.full((1, 5), 2.0, dtype=torch.float64)
    low = torch.full((1, 5), 0.5, dtype=torch.float64)
    assert chen.optical_power_loss_from_powers(high, reference).item() > 0.0
    assert chen.optical_power_loss_from_powers(low, reference).item() > 0.0
    near_zero = torch.zeros((1, 5), dtype=torch.float64)
    assert torch.isfinite(chen.optical_power_loss_from_powers(near_zero, reference))


def test_reference_collector_is_engineering_only_and_float64() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    ledger = chen.DataReadLedger()
    references = chen.collect_bogt_train_references(
        consumer,
        _images(),
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
        source_indices=torch.arange(10, dtype=torch.long),
        data_identity={"source": "collector unit fixture"},
        ledger=ledger,
    )
    assert references.engineering_only is True
    assert references.count == 10
    assert references.scale.dtype == torch.float64
    assert references.power_reference.dtype == torch.float64
    assert torch.isfinite(references.scale).all()
    assert torch.isfinite(references.power_reference).all()
    assert ledger.counts[chen.SYNTHETIC_ROLE] == 10


@pytest.mark.parametrize("bad_value", ("zero", "nan", "inf"))
def test_weight_calibration_rejects_invalid_initial_gradients(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    images = _images()
    references = _fixture_references(consumer)

    def invalid_gradients(outputs: object, inputs: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        del outputs, kwargs
        parameters = list(inputs)  # type: ignore[arg-type]
        fill = {"zero": 0.0, "nan": float("nan"), "inf": float("inf")}[bad_value]
        return tuple(torch.full_like(parameter, fill) for parameter in parameters)

    monkeypatch.setattr(torch.autograd, "grad", invalid_gradients)
    with pytest.raises(chen.TrainingContractError):
        chen.calibrate_initial_gradient_weights(
            consumer,
            images,
            references,
            data_role=chen.SYNTHETIC_ROLE,
            calibration_batch_identity={
                "seed": chen.SCHEDULE_SEED,
                "batch_size": chen.LOGICAL_BATCH_SIZE,
                "class_balanced": True,
            },
            engineering_only=True,
        )


def test_loss_components_are_finite_and_real_update_changes_only_allowed_phases() -> None:
    consumer, references, weights, images, ledger = _prepared_consumer()
    before_frozen = consumer.frozen_state_sha256
    breakdown = consumer.forward_loss(images, data_role=chen.SYNTHETIC_ROLE)
    assert all(torch.isfinite(value) for value in (breakdown.L_dist, breakdown.L_rec, breakdown.L_op, breakdown.L_total))
    expected_total = weights.alpha * breakdown.L_dist + weights.beta * breakdown.L_rec + weights.gamma * breakdown.L_op
    assert torch.allclose(breakdown.L_total, expected_total)

    optimizer = consumer.create_optimizer()
    update = consumer.bogt_training_step(
        images,
        optimizer,
        global_step=0,
        data_role=chen.SYNTHETIC_ROLE,
        ledger=ledger,
    )
    assert update.autograd.projection_active is True
    assert update.autograd.writer_active is True
    assert update.autograd.forbidden_gradients_absent is True
    assert all(value > 0.0 for value in update.parameter_update_norms.values())
    assert update.frozen_state_unchanged is True
    assert consumer.frozen_state_sha256 == before_frozen
    assert ledger.counts[chen.SYNTHETIC_ROLE] == 30


def _sampler() -> chen.DeterministicTrainRoleSampler:
    return chen.DeterministicTrainRoleSampler(
        torch.arange(chen.TRAIN_COUNT, dtype=torch.long),
        torch.arange(60_000, dtype=torch.long).remainder(10),
    )


def test_checkpoint_resume_matches_uninterrupted_engineering_trajectory() -> None:
    consumer_a, references, weights, images, _ = _prepared_consumer()
    optimizer_a = consumer_a.create_optimizer()
    sampler_a = _sampler()
    trace_a: list[dict[str, object]] = []
    for step in range(3):
        sampler_a.next_batch()
        trace_a.append(
            consumer_a.bogt_training_step(
                images,
                optimizer_a,
                global_step=step,
                data_role=chen.SYNTHETIC_ROLE,
            ).metrics()
        )

    consumer_b = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    consumer_b.bind_references(references)
    consumer_b.bind_weights(weights)
    optimizer_b = consumer_b.create_optimizer()
    sampler_b = _sampler()
    sampler_b.next_batch()
    first = consumer_b.bogt_training_step(
        images,
        optimizer_b,
        global_step=0,
        data_role=chen.SYNTHETIC_ROLE,
    )

    with tempfile.TemporaryDirectory(prefix="chen-bogt-test-") as temporary:
        checkpoint = Path(temporary) / "step_000001.engineering.pt"
        chen.save_checkpoint(
            checkpoint,
            chen.checkpoint_payload(
                consumer_b,
                optimizer_b,
                completed_updates=1,
                sampler_state=sampler_b.state_dict(),
                training_history=[first.metrics()],
                engineering_only=True,
            ),
        )
        consumer_c = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
            device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
        )
        optimizer_c = consumer_c.create_optimizer()
        loaded = chen.load_checkpoint(checkpoint, consumer_c, optimizer_c, expected_engineering_only=True)
        sampler_c = _sampler()
        sampler_c.load_state_dict(loaded.sampler_state)
        trace_c = [first.metrics()]
        for step in range(1, 3):
            sampler_c.next_batch()
            trace_c.append(
                consumer_c.bogt_training_step(
                    images,
                    optimizer_c,
                    global_step=step,
                    data_role=chen.SYNTHETIC_ROLE,
                ).metrics()
            )

        assert sampler_c.cursor == sampler_a.cursor == 3
        for left, right in zip(trace_a, trace_c, strict=True):
            assert left["completed_updates"] == right["completed_updates"]
            assert left["loss"] == pytest.approx(right["loss"], abs=1.0e-7)
        for name, value in consumer_a.trainable_state_dict().items():
            assert torch.allclose(value, consumer_c.trainable_state_dict()[name], rtol=1.0e-5, atol=1.0e-6)

        bad_payload = chen.checkpoint_payload(
            consumer_b,
            optimizer_b,
            completed_updates=1,
            sampler_state=sampler_b.state_dict(),
            training_history=[first.metrics()],
            engineering_only=True,
        )
        bad_payload["selection_rule"] = "changed"
        bad_checkpoint = Path(temporary) / "tampered.engineering.pt"
        torch.save(bad_payload, bad_checkpoint)
        with pytest.raises(chen.CheckpointError):
            chen.load_checkpoint(bad_checkpoint, consumer_c, optimizer_c, expected_engineering_only=True)


def test_data_firewall_rejects_non_train_roles_before_read() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    ledger = chen.DataReadLedger()
    for role in (chen.DEV_ROLE, chen.PROTECTED_ROLE, chen.TEST_ROLE, chen.SEALED_ROLE):
        with pytest.raises(chen.DataRoleError):
            chen.collect_bogt_train_references(
                consumer,
                _images(),
                data_role=role,
                engineering_only=True,
                ledger=ledger,
            )
    with pytest.raises(chen.DataRoleError):
        chen.collect_bogt_train_references(
            consumer,
            _images(),
            data_role=chen.SYNTHETIC_ROLE,
            engineering_only=False,
            ledger=ledger,
        )
    ledger.assert_forbidden_zero()
    assert all(ledger.counts[role] == 0 for role in (chen.DEV_ROLE, chen.PROTECTED_ROLE, chen.TEST_ROLE, chen.SEALED_ROLE))


def test_phase_fit_is_gauged_wrapped_and_deterministic() -> None:
    z, references, u = _phase_fixture()
    model_a = chen.fit_wrapped_conditional_phase_model(
        z,
        references,
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
    )
    model_b = chen.fit_wrapped_conditional_phase_model(
        z,
        references,
        data_role=chen.SYNTHETIC_ROLE,
        engineering_only=True,
    )
    assert model_a.coefficients.shape == (101, 100)
    assert torch.allclose(model_a.coefficients, model_b.coefficients)
    assert torch.allclose(model_a.residual_mean, model_b.residual_mean)
    assert torch.allclose(model_a.covariance, model_a.covariance.T, atol=1.0e-12)
    assert torch.isfinite(model_a.cholesky).all()
    assert torch.allclose(model_a.cholesky, torch.tril(model_a.cholesky))
    assert torch.isfinite(model_a.conditional_mean_phase(u[:3])).all()
    wrapped = chen.wrap_phase(torch.tensor([math.pi - 1.0e-5, -math.pi + 1.0e-5], dtype=torch.float64))
    circular_gap = torch.atan2(torch.sin(wrapped[0] - wrapped[1]), torch.cos(wrapped[0] - wrapped[1]))
    assert abs(circular_gap.item()) < 3.0e-5

    gauged, _, indices = chen.max_q_gauge(z, references)
    angle = torch.tensor(0.71)
    rotated, _, rotated_indices = chen.max_q_gauge(
        z * torch.complex(angle.cos(), angle.sin()), references
    )
    assert torch.equal(indices, rotated_indices)
    assert torch.allclose(gauged, rotated, atol=2.0e-6, rtol=2.0e-6)

    tie = torch.ones((1, 100), dtype=torch.complex64)
    _, _, tie_indices = chen.max_q_gauge(tie, references)
    assert tie_indices.item() == 0

    sampled = chen.construct_sampled_ols(
        u[:3],
        references,
        model_a,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    assert sampled.coefficients.shape == (3, 100)
    assert sampled.coefficients.dtype == torch.complex64
    assert torch.isfinite(sampled.coefficients).all()
    assert torch.allclose(chen.statistical_variable(sampled.coefficients, references), u[:3], atol=2.0e-5, rtol=2.0e-5)


def test_real_and_generated_paths_share_the_same_w_ols_writer() -> None:
    consumer = chen.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device="cpu", propagation_grid_size=chen.ENGINEERING_PROPAGATION_GRID
    )
    real = consumer.runtime.forward_from_image(_images())
    generated = consumer.generated_forward(real.z_ols)
    assert torch.allclose(
        real.generator_input_field.tensor,
        generated.generator_input_field.tensor,
        atol=1.0e-6,
        rtol=1.0e-5,
    )
    assert real.generator_input_field.provenance.operation == generated.generator_input_field.provenance.operation


def test_scientific_graph_is_static_and_requires_explicit_execution() -> None:
    from scripts.research import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as execution

    graph = execution.build_scientific_execution_graph()
    assert graph["task_id"] == execution.NEXT_TASK
    assert graph["scientific_execution_started"] is False
    assert graph["implementation_only"] is False
    assert graph["cli_execution"] == "--execute"
    assert graph["updates"] == chen.SCIENTIFIC_UPDATES
    assert graph["checkpoint_steps"] == list(chen.CHECKPOINT_STEPS)
    assert graph["forbidden_roles"] == ["dev", "protected", "test", "sealed"]
    assert graph["steps"][-1] == "publish write-once generation bundle"
