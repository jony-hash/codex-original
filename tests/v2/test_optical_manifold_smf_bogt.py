from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_review_v1.json"


def test_scaled_contract_and_review_are_explicit() -> None:
    spec = bogt._load_spec(CONFIG)
    bogt._assert_contract(spec)
    formal, _ = bogt._load_formal(spec)
    assert formal.encoder_layers == formal.decoder_layers == 2
    assert bogt._load_review(REVIEW, spec)["bounded_diagnostic_training_authorized"] is True
    review = __import__("json").loads(REVIEW.read_text(encoding="utf-8"))
    assert review["schema"] == bogt.REVIEW_SCHEMA
    assert review["training_authorized"] is False
    assert review["formal_promotion"] is False


def test_field_transmission_is_sqrt_889_per_layer() -> None:
    assert bogt.FIELD_TRANSMISSION == pytest.approx(0.889**0.5)
    assert bogt.FIELD_TRANSMISSION**2 == pytest.approx(0.889)
    assert bogt.FIELD_TRANSMISSION**4 == pytest.approx(0.889**2)


def test_three_mm_relay_geometry_is_b0_and_minus_two_point_five() -> None:
    total = 3.0e-3
    u = total / 3.5
    v = total - u
    f = u * v / total
    matrix = bogt.path_abcd((u, v), (f,))
    assert matrix.a == pytest.approx(-2.5)
    assert matrix.b_m == pytest.approx(0.0, abs=1.0e-12)
    assert matrix.determinant == pytest.approx(1.0)
    assert u == pytest.approx(0.8571428571428571e-3)
    assert v == pytest.approx(2.142857142857143e-3)
    assert f == pytest.approx(0.6122448979591837e-3)


def test_ols_aggregate_kl_closed_form_and_zero() -> None:
    mean = torch.full((bogt.PORT_COUNT,), 1.0)
    variance = torch.full_like(mean, 0.25)
    assert bogt.bogt_kl_surrogate(mean, variance) == pytest.approx(0.0, abs=1.0e-7)
    changed = bogt.bogt_kl_surrogate(mean + 0.5, variance)
    assert float(changed) > 0.0


def test_two_pass_surrogate_matches_full_batch_gradient() -> None:
    z = (torch.rand(10, bogt.PORT_COUNT, dtype=torch.float64) + 0.2).requires_grad_()
    mean, variance = bogt.aggregate_moments(z)
    full = bogt.bogt_kl_surrogate(mean, variance)
    gradient_full = torch.autograd.grad(full, z, retain_graph=True)[0]
    surrogate = bogt.two_pass_bogt_surrogate(z)
    gradient_two_pass = torch.autograd.grad(surrogate, z)[0]
    assert torch.allclose(gradient_two_pass, gradient_full, atol=1.0e-10, rtol=1.0e-8)


def test_positive_prior_sampling_is_fixed_phase_and_reproducible() -> None:
    scale = torch.ones(bogt.PORT_COUNT)
    phase = torch.full((bogt.PORT_COUNT,), 0.37)
    left = bogt.sample_prior_coefficients((3, bogt.PORT_COUNT), scale, phase=phase, generator=torch.Generator().manual_seed(7))
    right = bogt.sample_prior_coefficients((3, bogt.PORT_COUNT), scale, phase=phase, generator=torch.Generator().manual_seed(7))
    assert torch.equal(left, right)
    assert torch.isfinite(left).all()
    assert torch.all(left.abs() > 0)
    assert torch.angle(left).mean().item() == pytest.approx(0.37, abs=1.0e-5)
    assert torch.equal(bogt.fixed_calibrated_phase(), torch.zeros(bogt.PORT_COUNT))


def test_contract_rejects_electronic_vae_fields() -> None:
    spec = bogt._load_spec(CONFIG)
    payload = {**spec.payload, "optical_contract": {**spec.payload["optical_contract"], "logvar": True}}
    invalid = bogt.Spec(spec.path, payload, spec.config_hash)
    with pytest.raises(ValueError, match="electronic VAE"):
        bogt._assert_contract(invalid)


def test_research_schedule_is_balanced_and_disjoint() -> None:
    labels = torch.arange(55000, dtype=torch.long) % 10
    calibration, schedule = bogt._research_schedule(labels)
    assert calibration.shape == (1000,)
    assert schedule.shape == (100, 10)
    assert not torch.isin(calibration, schedule).any()
    assert torch.equal(labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(100))


def test_independent_kl_reference_matches_surrogate() -> None:
    mean = torch.linspace(0.8, 1.2, bogt.PORT_COUNT, dtype=torch.float64)
    variance = torch.linspace(0.15, 0.35, bogt.PORT_COUNT, dtype=torch.float64)
    assert torch.allclose(bogt.bogt_kl_surrogate(mean, variance), bogt.independent_bogt_kl_reference(mean, variance), atol=1.0e-12, rtol=1.0e-12)


def test_pilot_gate_results_are_fail_closed() -> None:
    metrics = {
        "mse_kl_op_heldout_mse": 1.0,
        "mse_op_heldout_mse": 1.0,
        "mse_only_kl": 1.0,
        "mse_op_kl": 1.0,
        "mse_kl_kl": 0.9,
        "mse_kl_op_kl": 0.9,
        "mse_only_op": 1.0,
        "mse_op_op": 0.8,
        "ols_effective_rank_retention": 0.95,
        "paired_mse": 0.5,
        "template_control_mse": 0.6,
        "cyclic_control_mse": 0.7,
        "four_phase_updated": True,
        "ols_exposure_calibration_evaluator_frozen": True,
    }
    gates = bogt.pilot_gate_results(metrics)
    assert all(gate.passed for gate in gates)
    metrics["mse_kl_op_heldout_mse"] = 1.2
    assert not bogt.pilot_gate_results(metrics)[0].passed


def test_pilot_gate_checks_both_factorial_kl_pairs() -> None:
    metrics = {
        "mse_kl_op_heldout_mse": 1.0,
        "mse_op_heldout_mse": 1.0,
        "mse_only_kl": 1.0,
        "mse_kl_kl": 0.9,
        "mse_op_kl": 1.0,
        "mse_kl_op_kl": 0.96,
        "mse_only_op": 1.0,
        "mse_op_op": 0.8,
        "ols_effective_rank_retention": 0.95,
        "paired_mse": 0.5,
        "template_control_mse": 0.6,
        "cyclic_control_mse": 0.7,
    }
    gates = bogt.pilot_gate_results(metrics)
    assert gates[1].passed
    assert not gates[2].passed


def test_gradient_weight_writer_separates_local_and_final_replay() -> None:
    panels = (
        {"raw_mse_grad_norm": 1.0, "raw_kl_grad_norm": 2.0, "raw_op_grad_norm": 2.0},
        {"raw_mse_grad_norm": 2.0, "raw_kl_grad_norm": 2.0, "raw_op_grad_norm": 1.0},
        {"raw_mse_grad_norm": 4.0, "raw_kl_grad_norm": 2.0, "raw_op_grad_norm": 2.0},
    )
    weights = bogt.calibrate_objective_weights(panels)
    assert all(record["local_target_kl_ratio"] == pytest.approx(0.10) for record in weights["panel_local_calibration"])
    final = weights["final_fixed_weight_replay"]
    assert final[0]["effective_kl_ratio_final_weight"] != pytest.approx(0.10)
    assert len({record["final_alpha"] for record in final}) == 1
    assert len({record["final_gamma"] for record in final}) == 1
    assert bogt.final_fixed_weight_ratio_gate(final)


def test_final_fixed_replay_changes_all_panels_and_rejects_stale_local_values() -> None:
    panels = ({"mse": 1.0, "kl": 1.0, "op": 1.0},) * 3
    left = bogt.replay_fixed_weight_ratios(panels, alpha=0.1, gamma=0.1)
    right = bogt.replay_fixed_weight_ratios(panels, alpha=0.2, gamma=0.2)
    assert all(a["effective_kl_ratio_final_weight"] != b["effective_kl_ratio_final_weight"] for a, b in zip(left, right, strict=True))
    assert all(a["effective_op_ratio_final_weight"] != b["effective_op_ratio_final_weight"] for a, b in zip(left, right, strict=True))
    stale = [{"kl": 0.10, "op": 0.10}] * 3
    assert not bogt.final_fixed_weight_ratio_gate(stale)


def test_pilot_classification_preserves_multiple_science_failures() -> None:
    science = bogt.pilot_gate_results({
        "mse_kl_op_heldout_mse": 2.0,
        "mse_op_heldout_mse": 1.0,
        "mse_only_kl": 1.0,
        "mse_kl_kl": 1.0,
        "mse_op_kl": 1.0,
        "mse_kl_op_kl": 1.0,
        "mse_only_op": 1.0,
        "mse_op_op": 1.0,
        "ols_effective_rank_retention": 0.5,
        "paired_mse": 2.0,
        "template_control_mse": 1.0,
        "cyclic_control_mse": 1.0,
    })
    status, failed = bogt.classify_pilot_result(science, (bogt.GateResult("resource", True, {}),))
    assert status == "valid_failed_multiple_science_gates"
    assert "combined_heldout_mse" in failed
    assert "ols_effective_rank_retention" in failed


def test_phase_update_metrics_exposes_each_of_four_layers() -> None:
    initial = {
        "system.encoder.phase": torch.zeros(2, 1, 4, 4),
        "system.decoder.phase": torch.zeros(2, 1, 4, 4),
    }
    final = {name: value.clone() for name, value in initial.items()}
    final["system.encoder.phase"][0, 0, 0, 0] = 0.25
    final["system.encoder.phase"][1, 0, 0, 0] = 0.5
    final["system.decoder.phase"][0, 0, 0, 0] = 0.75
    final["system.decoder.phase"][1, 0, 0, 0] = 1.0
    updates = bogt._phase_update_metrics(initial, final)
    assert set(updates) == {
        "system.encoder.phase[0]",
        "system.encoder.phase[1]",
        "system.decoder.phase[0]",
        "system.decoder.phase[1]",
    }
    assert updates["system.decoder.phase[1]"] == pytest.approx(1.0)


def test_phase_quantization_uses_registered_eight_levels() -> None:
    phase = torch.tensor([-0.1, 0.4, 2.5, 5.9])
    quantized = bogt._phase_quantize(phase)
    assert torch.all(torch.isin(quantized, torch.tensor(bogt.PHASE_LEVELS)))


def test_loss_components_route_kl_to_coefficients_and_power_loss_to_planes() -> None:
    camera = torch.ones(1, 1, 2, 2, requires_grad=True)
    coefficients = (torch.ones(2, bogt.PORT_COUNT, dtype=torch.complex64) * (1.0 + 0.1j)).requires_grad_()
    powers = tuple((torch.ones(2) * 1.2).requires_grad_() for _ in range(5))
    output = bogt.ForwardOutput(camera, coefficients, powers, (), (), None)
    calibration = bogt.Calibration(torch.ones(bogt.PORT_COUNT), torch.ones(5), 1000)
    components = bogt.loss_components(output, torch.zeros_like(camera), calibration)
    components["kl"].backward(retain_graph=True)
    assert coefficients.grad is not None and float(coefficients.grad.abs().sum()) > 0.0
    coefficients.grad = None
    components["op"].backward()
    assert coefficients.grad is None
    assert all(float(value.grad.abs().sum()) > 0.0 for value in powers)


def test_lazy_router_contains_new_actions() -> None:
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    schema = (ROOT / "src/ldm_project/v2/application/_cli_schema_research.py").read_text(encoding="utf-8")
    for action in ("smf-bogt-validate", "smf-bogt-preflight", "smf-bogt-pilot-train"):
        assert action in router or action in schema
    assert "optical_manifold_smf_bogt" in router
    assert "--pilot-review-contract" in schema
