from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_audit as regularization_audit
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_objective_audit as objective
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_pilot as pilot
from ldm_project.v2.config import load_config


CONFIG = "configs/research/optical_manifold/lightgen_optical_manifold_identity_regularization_objective_audit_v1.yaml"
REVIEW = "configs/research/optical_manifold/lightgen_optical_manifold_identity_regularization_objective_audit_review_v1.json"


def test_objective_audit_contract_and_lazy_cli_are_registered() -> None:
    spec = objective._load_spec(CONFIG)
    objective._assert_contract(spec)
    review, _ = objective._load_review(REVIEW, spec)
    assert review["no_update_audit_authorized"] is True
    assert review["training_authorized"] is False
    assert spec.payload["regularization_gradient"]["tested_multipliers"] == [1.0, 2.0, 4.0]

    parser = _parser()
    validate = parser.parse_args(
        [
            "research", "optical-manifold", "manifold-regularization-objective-audit-validate",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--dataset-root", "data", "--output", "validation.json",
        ]
    )
    run = parser.parse_args(
        [
            "research", "optical-manifold", "manifold-regularization-objective-audit-run",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--validation-evidence", "validation.json", "--dataset-root", "data",
            "--output", "audit.json",
        ]
    )
    assert validate.research_action == "manifold-regularization-objective-audit-validate"
    assert run.device == "cuda"


def test_fresh_panels_exclude_all_prior_panels_and_are_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = torch.arange(60000, dtype=torch.long).remainder(10)
    pilot_data = SimpleNamespace(
        train_indices=torch.arange(55000, dtype=torch.long),
        heldout_indices=torch.arange(55000, 56000, dtype=torch.long),
        labels=labels,
        manifest={"manifest_hash": pilot.SOURCE_MANIFEST_HASH},
    )
    full_data = SimpleNamespace(
        pilot_data=pilot_data,
        manifest={"manifest_hash": "full-manifest"},
    )
    calibration = torch.arange(0, 1000, dtype=torch.long)
    gradient_batches = tuple(
        torch.arange(1000 + batch * 100, 1100 + batch * 100, dtype=torch.long)
        for batch in range(3)
    )
    regularization_data = SimpleNamespace(
        full_data=full_data,
        calibration_indices=calibration,
        gradient_batches=gradient_batches,
        manifest={"manifest_hash": "regularization-manifest"},
    )
    monkeypatch.setattr(objective.full_epoch, "_load_data", lambda _root: full_data)
    monkeypatch.setattr(objective.audit, "_load_data", lambda _root: regularization_data)
    monkeypatch.setattr(
        objective.pilot,
        "_load_data",
        lambda _root: SimpleNamespace(manifest={"manifest_hash": "pilot-training-manifest"}),
    )

    data = objective._load_data("unused")
    excluded = torch.cat((calibration, *gradient_batches, pilot._pilot_schedule(pilot_data.train_indices, labels).reshape(-1), pilot_data.heldout_indices))
    selected = torch.cat((data.projection_indices, *data.gradient_batches))
    assert data.projection_indices.shape == (1000,)
    assert len(data.gradient_batches) == 3
    assert all(batch.shape == (100,) for batch in data.gradient_batches)
    assert torch.unique(selected).numel() == selected.numel()
    assert not torch.isin(selected, excluded).any()
    assert data.manifest["no_replacement"] is True


def test_rank_coefficients_cover_all_ranks_and_shift_is_a_control() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    scores = torch.randn(100, objective.PCA_COMPONENTS, generator=generator)
    directions = torch.randn(objective.FIXED_DIRECTIONS, objective.PCA_COMPONENTS, generator=generator)
    directions = torch.nn.functional.normalize(directions, dim=1)
    coefficients, ranks = objective._rank_coefficients(scores, directions)
    control_coefficients, control_ranks = objective._rank_coefficients(scores, directions, rank_shift=1)
    expected = torch.arange(100).reshape(-1, 1).expand(100, objective.FIXED_DIRECTIONS)
    assert coefficients.shape == (100, objective.FIXED_DIRECTIONS)
    assert objective.audit._rank_assignment_is_permutation(ranks)
    assert objective.audit._rank_assignment_is_permutation(control_ranks)
    assert torch.equal(torch.sort(ranks, dim=0).values, expected)
    assert torch.equal(torch.sort(control_ranks, dim=0).values, expected)
    assert not torch.equal(coefficients, control_coefficients)


def test_parameter_contract_has_no_optimizer_and_only_phase_parameters() -> None:
    config = load_config("configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml")
    model = base.IdentityOpticalManifoldAutoencoder(config.system)
    contract = objective._parameter_contract(model)
    assert tuple(contract["trainable_parameter_names"]) == objective.EXPECTED_TRAINABLE
    assert contract["optimizer_created"] is False
    assert contract["optimizer_parameter_names"] == []


def test_wrong_arm_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"schema": pilot.RESUME_SCHEMA, "arm": "mse_only", "step": 100}, path)
    with pytest.raises(ValueError, match="checkpoint lineage"):
        objective._load_arm_checkpoint(path, objective.file_sha256(path), "regularized", "config", "manifest")


def test_objective_audit_does_not_import_torch_optimizer_or_evaluator() -> None:
    source = inspect.getsource(objective)
    assert "torch.optim" not in source
    assert "SmoothMNISTEvaluator" not in source
    assert "load_frozen_evaluator" not in source


def test_projection_metric_control_is_finite() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    scores = torch.randn(100, objective.PCA_COMPONENTS, generator=generator)
    directions = regularization_audit._fixed_directions(torch.device("cpu"))
    learned = regularization_audit._sliced_wasserstein(scores, directions)
    control = regularization_audit._sliced_wasserstein(torch.zeros_like(scores), directions)
    assert torch.isfinite(learned)
    assert torch.isfinite(control)
