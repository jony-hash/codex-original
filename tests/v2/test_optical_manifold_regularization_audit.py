from __future__ import annotations

import inspect
import json

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import optical_manifold_regularization_audit as audit


CONFIG = "configs/research/lightgen_optical_manifold_identity_regularization_audit_v1.yaml"
REVIEW = "configs/research/lightgen_optical_manifold_identity_regularization_audit_review_v1.json"


def test_regularization_contract_and_cli_are_registered() -> None:
    spec = audit._load_spec(CONFIG)
    audit._assert_contract(spec)
    review, _ = audit._load_review(REVIEW, spec)
    assert review["no_update_audit_authorized"] is True
    assert review["training_authorized"] is False
    parser = _parser()
    validate = parser.parse_args(
        [
            "research", "optical-manifold", "manifold-regularization-audit-validate",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--dataset-root", "data", "--output", "validation.json",
        ]
    )
    assert validate.research_action == "manifold-regularization-audit-validate"
    run = parser.parse_args(
        [
            "research", "optical-manifold", "manifold-regularization-audit-run",
            "--config", CONFIG, "--review-contract", REVIEW,
            "--validation-evidence", "validation.json", "--dataset-root", "data",
            "--output", "audit.json",
        ]
    )
    assert run.device == "cuda"


def test_source_review_rejects_tampering(tmp_path) -> None:
    spec = audit._load_spec(CONFIG)
    payload = json.loads((audit._resolve(REVIEW)).read_text(encoding="utf-8"))
    payload["source_checkpoint_sha256"] = "0" * 64
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="review contract"):
        audit._load_review(path, spec)


def test_train_panel_contract_is_disjoint_and_deterministic(monkeypatch) -> None:
    class _Pilot:
        pass

    pilot = _Pilot()
    pilot.train_indices = torch.arange(55000, dtype=torch.long)
    pilot.heldout_indices = torch.arange(55000, 56000, dtype=torch.long)
    pilot.labels = torch.arange(60000, dtype=torch.long).remainder(10)
    pilot.manifest = {"manifest_hash": audit.SOURCE_MANIFEST_HASH}
    full = _Pilot()
    full.pilot_data = pilot
    full.manifest = {"manifest_hash": "full"}

    monkeypatch.setattr(audit.full_epoch, "_load_data", lambda _root: full)
    data = audit._load_data("unused")
    assert data.calibration_indices.shape == (1000,)
    assert len(data.gradient_batches) == 3
    assert all(batch.shape == (100,) for batch in data.gradient_batches)
    selected = torch.cat((data.calibration_indices, *data.gradient_batches))
    assert torch.unique(selected).numel() == selected.numel()
    assert not torch.isin(selected, pilot.heldout_indices).any()
    assert torch.equal(
        pilot.labels.index_select(0, data.gradient_batches[0]),
        torch.arange(10).repeat_interleave(10),
    )


def test_complex_projection_and_swd_are_finite() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    fields = torch.randn(64, 1, 100, 100, generator=generator) + 1j * torch.randn(
        64, 1, 100, 100, generator=generator
    )
    projection, scores = audit._fit_projection(fields.to(torch.complex64))
    assert projection.basis.dtype == torch.complex64
    assert projection.basis.shape == (10000, audit.PCA_COMPONENTS)
    assert scores.shape == (64, audit.PCA_COMPONENTS)
    directions = audit._fixed_directions(torch.device("cpu"))
    learned = audit._sliced_wasserstein(scores, directions)
    collapsed = audit._sliced_wasserstein(torch.zeros_like(scores), directions)
    assert torch.isfinite(learned)
    assert torch.isfinite(collapsed)
    assert float(collapsed) > 0.0


def test_projection_does_not_import_optimizer_or_evaluator() -> None:
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "SmoothMNISTEvaluator" not in source
    assert "load_frozen_evaluator" not in source


def test_gradient_report_has_expected_scope_and_alignment() -> None:
    names = audit.EXPECTED_TRAINABLE
    reconstruction = {name: torch.ones(2, 1, 2, 2) for name in names}
    regularizer = {
        names[0]: torch.full((2, 1, 2, 2), 0.5),
        names[1]: torch.zeros(2, 1, 2, 2),
    }
    report = audit._gradient_report(reconstruction, regularizer, 0.1)
    assert report["regularizer_decoder_norm"] == 0.0
    assert report["combined_encoder_norm"] > 0.0
    assert report["combined_reconstruction_dot_encoder"] > 0.0
    assert report["combined_regularizer_dot_encoder"] > 0.0
    assert report["finite"] is True


def test_validation_payload_rejects_wrong_source_hash(tmp_path) -> None:
    spec = audit._load_spec(CONFIG)
    review = json.loads(audit._resolve(REVIEW).read_text(encoding="utf-8"))
    review["source_pre_ols_field_sha256"] = "0" * 64
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="review contract"):
        audit._load_review(path, spec)


def test_no_update_contract_is_explicit() -> None:
    spec = audit._load_spec(CONFIG)
    assert spec.payload["authorization"]["no_update_audit_authorized"] is True
    assert spec.payload["authorization"]["training_authorized"] is False
    assert spec.payload["regularization"]["approximation_role"] == "project_approximation"
