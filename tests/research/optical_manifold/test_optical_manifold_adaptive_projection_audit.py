from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold_adaptive_projection_audit as adaptive
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_audit as regularization
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_pilot as pilot
from ldm_project.v2.config import load_config


CONFIG = "configs/research/optical_manifold/lightgen_optical_manifold_identity_adaptive_projection_audit_v1.yaml"
REVIEW = "configs/research/optical_manifold/lightgen_optical_manifold_identity_adaptive_projection_audit_review_v1.json"


def test_contract_and_lazy_cli_are_registered() -> None:
    spec = adaptive._load_spec(CONFIG)
    adaptive._assert_contract(spec)
    review, _ = adaptive._load_review(REVIEW, spec)
    assert review["no_update_audit_authorized"] is True
    assert review["training_authorized"] is False
    assert spec.payload["projection_stability"]["candidate_components"] == [16, 24, 32]

    parser = _parser()
    validate = parser.parse_args(
        [
            "research", "optical-manifold", "adaptive-projection-audit-validate",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--dataset-root", "data", "--output", "validation.json",
        ]
    )
    run = parser.parse_args(
        [
            "research", "optical-manifold", "adaptive-projection-audit-run",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--validation-evidence", "validation.json", "--dataset-root", "data",
            "--output", "audit.json",
        ]
    )
    assert validate.research_action == "adaptive-projection-audit-validate"
    assert run.device == "cuda"


def test_adaptive_projection_does_not_import_optimizer_or_evaluator() -> None:
    source = inspect.getsource(adaptive)
    assert "torch.optim" not in source
    assert "SmoothMNISTEvaluator" not in source
    assert "load_frozen_evaluator" not in source


def test_fresh_panels_exclude_prior_panels_and_are_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = torch.arange(56000, dtype=torch.long).remainder(10)
    pilot_data = SimpleNamespace(
        train_indices=torch.arange(55000, dtype=torch.long),
        heldout_indices=torch.arange(55000, 56000, dtype=torch.long),
        labels=labels,
    )
    full_data = SimpleNamespace(pilot_data=pilot_data)
    regularization_data = SimpleNamespace(
        full_data=full_data,
        calibration_indices=torch.arange(0, 1000, dtype=torch.long),
        gradient_batches=tuple(
            torch.arange(1000 + i * 100, 1100 + i * 100, dtype=torch.long)
            for i in range(3)
        ),
    )
    objective_data = SimpleNamespace(
        full_data=full_data,
        regularization_data=regularization_data,
        projection_indices=torch.arange(1300, 2300, dtype=torch.long),
        gradient_batches=tuple(
            torch.arange(2300 + i * 100, 2400 + i * 100, dtype=torch.long)
            for i in range(3)
        ),
        manifest={"manifest_hash": "objective"},
    )
    monkeypatch.setattr(adaptive.objective, "_load_data", lambda _root: objective_data)
    monkeypatch.setattr(
        adaptive.pilot,
        "_pilot_schedule",
        lambda _indices, _labels: torch.arange(3000, 4000, dtype=torch.long).reshape(100, 10),
    )
    data = adaptive._load_data("unused")
    selected = torch.cat((*data.calibration_panels, data.assessment_indices, *data.gradient_batches))
    assert selected.numel() == 4300
    assert torch.unique(selected).numel() == selected.numel()
    excluded = torch.cat(
        (
            regularization_data.calibration_indices,
            *regularization_data.gradient_batches,
            objective_data.projection_indices,
            *objective_data.gradient_batches,
            pilot._pilot_schedule(pilot_data.train_indices, pilot_data.labels).reshape(-1),
            pilot_data.heldout_indices,
        )
    )
    assert not torch.isin(selected, excluded).any()
    expected_calibration = torch.arange(10).repeat_interleave(100)
    assert all(torch.equal(pilot_data.labels.index_select(0, panel), expected_calibration) for panel in data.calibration_panels)
    assert torch.equal(
        pilot_data.labels.index_select(0, data.assessment_indices), expected_calibration
    )
    assert data.manifest["no_replacement"] is True


def test_consensus_projection_and_rank_controls_are_finite() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    fields = tuple(
        torch.complex(
            torch.randn(40, 1, 100, 100, generator=generator),
            torch.randn(40, 1, 100, 100, generator=generator),
        )
        for _ in range(3)
    )
    fits = tuple(adaptive._fit_projection_rank(panel, 16) for panel in fields)
    consensus = adaptive._consensus_projection(fits, torch.cat(fields), 16)
    directions = adaptive._directions(16, torch.device("cpu"))
    scores = adaptive._project_fields(fields[0], consensus)
    _, ranks = adaptive._rank_coefficients(scores, directions)
    _, control_ranks = adaptive._rank_coefficients(scores, directions, rank_shift=1)
    expected = torch.arange(40).reshape(-1, 1).expand_as(ranks)
    assert consensus.basis.shape == (10000, 16)
    assert torch.isfinite(consensus.basis).all()
    assert torch.equal(torch.sort(ranks, dim=0).values, expected)
    assert torch.equal(torch.sort(control_ranks, dim=0).values, expected)
    assert not torch.equal(ranks, control_ranks)


def test_candidate_report_keeps_assessment_out_of_consensus_fit() -> None:
    generator = torch.Generator(device="cpu").manual_seed(42)
    calibration = tuple(
        torch.complex(
            torch.randn(40, 1, 100, 100, generator=generator),
            torch.randn(40, 1, 100, 100, generator=generator),
        )
        for _ in range(3)
    )
    assessment = torch.complex(
        torch.randn(40, 1, 100, 100, generator=generator),
        torch.randn(40, 1, 100, 100, generator=generator),
    )
    original = regularization.Projection(
        mean_field=torch.zeros(10000, dtype=torch.complex64),
        basis=torch.eye(10000, 32, dtype=torch.complex64),
        whitening=torch.ones(32),
        eigenvalues=torch.ones(32),
        explained_variance=torch.ones(32) / 32,
    )
    consensus, report = adaptive._candidate_report(16, calibration, assessment, original)
    assert consensus.basis.shape == (10000, 16)
    assert report["assessment_refit_overlap"] >= 0.0
    assert report["assessment_used_for_fit"] is False
    assert report["assessment_refit_diagnostic_only"] is True


def test_parameter_contract_has_no_optimizer_and_only_phase_parameters() -> None:
    config = load_config("configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml")
    model = base.IdentityOpticalManifoldAutoencoder(config.system)
    contract = adaptive._parameter_contract(model)
    assert tuple(contract["trainable_parameter_names"]) == adaptive.EXPECTED_TRAINABLE
    assert contract["optimizer_created"] is False
    assert contract["optimizer_parameter_names"] == []
