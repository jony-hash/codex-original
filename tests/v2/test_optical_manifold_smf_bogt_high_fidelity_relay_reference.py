from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import cli
from ldm_project.v2.application.research import optical_manifold_smf_bogt_high_fidelity_relay_reference as audit
from ldm_project.v2.config import load_config
from ldm_project.v2.contracts import ComplexField, OperationProvenance


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_review_v1.json"
FORMAL = load_config(ROOT / "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml").system


def test_contract_and_review_are_locked() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    assert audit._load_review(REVIEW, spec).name.endswith("high_fidelity_relay_reference_review_v1.json")
    assert spec.payload["hypothesis"] == audit.HYPOTHESIS
    assert spec.payload["reference"]["dtype"] == "torch.complex128"


def test_abcd_and_pupil_occupancy_are_recomputed() -> None:
    details, gates = audit._backend_validation(FORMAL)
    assert all(gate.passed for gate in gates)
    assert details["backends"]["identity_backend"]["abcd"]["A"] == pytest.approx(-2.5)
    assert details["backends"]["identity_backend"]["abcd"]["C_per_m"] == pytest.approx(-14.0)
    assert details["backends"]["scaled_chen_backend"]["abcd"]["C_per_m"] == pytest.approx(-1633.3333333333)
    assert details["backends"]["identity_backend"]["source_pupil_bins"] == [7843, 10000]
    assert details["backends"]["scaled_chen_backend"]["target_pupil_bins"] == [40000, 40000]


def test_task_subset_is_balanced_and_registered() -> None:
    labels = torch.full((10000,), -1, dtype=torch.long)
    for label, values in audit.TASK_INDICES.items():
        labels[torch.tensor(values, dtype=torch.long)] = label
    data = type("Data", (), {"test_labels": labels})()
    values, labels = audit._task_indices(data)
    assert values.numel() == 100
    assert torch.equal(labels, torch.arange(10).repeat_interleave(10))
    assert audit.canonical_sha256(values.tolist()) == audit.TASK_SUBSET_SHA256


def test_synthetic_probe_family_and_reference_convergence() -> None:
    for name in audit.SYNTHETIC_PROBES:
        field = audit._synthetic_probe(name)
        assert field.shape == (1, 1, 100, 100)
        assert field.dtype == torch.complex128
        assert torch.isfinite(field).all()
    audit._FORMAL_CACHE = FORMAL
    convergence = audit._reference_convergence(audit._synthetic_probe("P4_multifrequency"), "identity_backend")
    assert convergence["passed"] is True
    assert convergence["complex_relative_l2"] <= 1e-11
    assert convergence["power_relative_difference"] <= 1e-12


def test_reference_is_independent_and_metrics_are_locked() -> None:
    source = inspect.getsource(audit._reference_core)
    assert "_fourier_resample" not in source
    assert "_quadratic_phase" not in source
    assert "reference_b0_imaging" not in source
    assert audit._reference_relay(audit._synthetic_probe("P0_dc"), "scaled_chen_backend").dtype == torch.complex128
    assert audit._numerical_pass({
        "finite": True,
        "complex_relative_l2": 0.002,
        "normalized_complex_correlation": 0.999999,
        "amplitude_relative_l2": 0.002,
        "phase_rms_rad": 0.002,
        "field_power_relative_difference": 1e-4,
        "intensity_relative_l2": 0.002,
        "integrated_intensity_relative_difference": 1e-4,
    }) is True


def test_production_identity_matches_reference_on_dc() -> None:
    audit._FORMAL_CACHE = FORMAL
    _, relays = audit._build_backend_planes(FORMAL)
    field = audit._synthetic_probe("P0_dc")
    production = relays["identity_backend"](ComplexField(field, relays["identity_backend"].source, OperationProvenance("test", "chip", None))).tensor
    reference = audit._reference_relay(field, "identity_backend")
    metrics = audit._compare_one(production, reference)
    assert metrics["complex_relative_l2"] < 2e-3
    assert metrics["normalized_complex_correlation"] > 0.999999


def test_canonicalization_is_analytic_only_and_non_mutating() -> None:
    audit._FORMAL_CACHE = FORMAL
    _, relays = audit._build_backend_planes(FORMAL)
    field = audit._synthetic_probe("P1_low_mode")
    production = relays["scaled_chen_backend"](ComplexField(field, relays["scaled_chen_backend"].source, OperationProvenance("test", "chip", None))).tensor
    before = production.clone()
    canonical = audit._canonicalize_field(production, "scaled_chen_backend")
    assert canonical.shape == field.shape
    assert torch.equal(production, before)
    assert audit.CANONICALIZATION["fit"] is False
    assert audit.CANONICALIZATION["physical_forward_modified"] is False


def test_classification_boundaries() -> None:
    passing = {"finite": True, "complex_relative_l2": 1e-5, "normalized_complex_correlation": 1.0, "amplitude_relative_l2": 1e-5, "phase_rms_rad": 1e-5, "field_power_relative_difference": 1e-6, "intensity_relative_l2": 1e-5, "integrated_intensity_relative_difference": 1e-6}
    failing = {**passing, "complex_relative_l2": 0.1, "normalized_complex_correlation": 0.8}
    synthetic = {"aggregate": {"identity_backend": passing, "scaled_chen_backend": passing}}
    task = {"aggregate": {"I_to_I": passing, "S_to_S": passing}}
    assert audit._classify_numeric(synthetic, task)[0] == "relay_numerical_reference_consistent"
    synthetic["aggregate"]["identity_backend"] = failing
    task["aggregate"]["I_to_I"] = failing
    assert audit._classify_numeric(synthetic, task)[0] == "relay_numerical_implementation_mismatch_supported"


def test_cli_actions_are_lazy() -> None:
    parser = cli._parser()
    validate = parser.parse_args(["research", "optical-manifold", "scaled-chen-high-fidelity-relay-reference-validate", "--config", "config.yaml", "--review-contract", "review.json", "--dataset-root", "data", "--output", "validation.json"])
    run = parser.parse_args(["research", "optical-manifold", "scaled-chen-high-fidelity-relay-reference-run", "--config", "config.yaml", "--review-contract", "review.json", "--validation-evidence", "validation.json", "--dataset-root", "data", "--output", "audit.json"])
    assert validate.research_action.endswith("-validate")
    assert run.device == "cuda"
    router = (ROOT / "src/ldm_project/v2/application/_cli_research_optical_manifold.py").read_text(encoding="utf-8")
    assert "optical_manifold_smf_bogt_high_fidelity_relay_reference" in router
    assert "import optical_manifold_smf_bogt_high_fidelity_relay_reference" not in router


def test_no_update_source_contract() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "backward(" not in source
    assert "evaluator_calls" in source
    assert "prior_sampled" in source
    assert "canonicalization_applied_to_physical_forward" not in source
