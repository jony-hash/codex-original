from __future__ import annotations

import argparse
import inspect
import json
from types import SimpleNamespace

import torch
import pytest

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base, optical_manifold_reconstruction_audit as audit


CONFIG = "configs/research/optical_manifold/lightgen_optical_manifold_identity_reconstruction_audit_v1.yaml"
REVIEW = "configs/research/optical_manifold/lightgen_optical_manifold_identity_reconstruction_audit_review_v1.json"
SOURCE_TRAINING = "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/epoch_training.json"
SOURCE_CHECKPOINT = "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/full_epoch_final_step_005500.pt"


@pytest.mark.historical_artifact
def test_audit_config_review_and_source_are_bound() -> None:
    spec = audit._load_spec(CONFIG)
    review, review_path = audit._load_review(REVIEW, spec)
    payload, checkpoint, training_path, checkpoint_path = audit._load_source(
        spec,
        review_path,
        SOURCE_TRAINING,
        SOURCE_CHECKPOINT,
        audit._git_state()["commit"],
    )
    assert review["no_update_audit_authorized"] is True
    assert payload["evidence"]["status"] == "passed"
    assert checkpoint["global_step"] == 5500
    assert len(checkpoint["loss_history"]) == 5500
    assert training_path.name == "epoch_training.json"
    assert checkpoint_path.name == "full_epoch_final_step_005500.pt"


def test_source_checkpoint_rejects_wrong_path() -> None:
    spec = audit._load_spec(CONFIG)
    _, review_path = audit._load_review(REVIEW, spec)
    try:
        audit._load_source(spec, review_path, SOURCE_TRAINING, "missing.pt", audit._git_state()["commit"])
    except ValueError as exc:
        assert "registered source evidence paths" in str(exc)
    else:
        raise AssertionError("wrong source checkpoint path was accepted")


def test_phase_controls_are_deterministic_and_restore_learned_state() -> None:
    spec = audit._load_spec(CONFIG)
    formal, _ = audit._load_formal(spec)
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    with torch.no_grad():
        model.system.encoder.phase.fill_(0.25)
        model.system.decoder.phase.fill_(-0.5)
    encoder = model.system.encoder.phase.detach().clone()
    decoder = model.system.decoder.phase.detach().clone()
    audit._phase_variant(model, "zero_phase")
    assert not torch.equal(model.system.encoder.phase, encoder)
    assert not torch.equal(model.system.decoder.phase, decoder)
    audit._restore_phases(model, encoder, decoder)
    audit._phase_variant(model, "random_phase")
    random_encoder = model.system.encoder.phase.detach().clone()
    random_decoder = model.system.decoder.phase.detach().clone()
    audit._restore_phases(model, encoder, decoder)
    audit._phase_variant(model, "random_phase")
    assert torch.equal(model.system.encoder.phase, random_encoder)
    assert torch.equal(model.system.decoder.phase, random_decoder)
    audit._restore_phases(model, encoder, decoder)
    assert torch.equal(model.system.encoder.phase, encoder)
    assert torch.equal(model.system.decoder.phase, decoder)


def test_metric_controls_distinguish_instance_and_template_predictions() -> None:
    target = torch.arange(400, dtype=torch.float32).reshape(100, 1, 2, 2)
    prediction = target.clone()
    labels = torch.arange(10).repeat_interleave(10)
    metrics = audit._metrics(prediction, target, labels)
    assert metrics["paired_mse"] == 0.0
    assert metrics["exact_target_top1"] == 1.0
    assert metrics["centered_pearson_mean"] == 1.0


def test_source_metric_gate_consumes_authoritative_replay_not_cpu_statistic(monkeypatch) -> None:
    calls = []

    def fake_evaluate(model, pilot_data, indices, device):
        calls.append((model, pilot_data, indices, device))
        return audit.SOURCE_PAIRED_MSE

    monkeypatch.setattr(audit.full_epoch.pilot, "_evaluate", fake_evaluate)
    model = SimpleNamespace(eval=lambda: None)
    pilot_data = object()
    indices = torch.arange(2)
    data = SimpleNamespace(pilot_data=pilot_data, heldout_indices=indices)
    measured = audit._authoritative_source_replay_mse(model, data, torch.device("cpu"))

    assert measured == audit.SOURCE_PAIRED_MSE
    assert calls == [(model, pilot_data, indices, torch.device("cpu"))]

    passed = audit._source_metric_reproduction_gate(
        audit.SOURCE_PAIRED_MSE,
        audit.SOURCE_PAIRED_MSE - 3.7e-9,
    )
    failed = audit._source_metric_reproduction_gate(
        audit.SOURCE_PAIRED_MSE + 3.7e-9,
        audit.SOURCE_PAIRED_MSE,
    )
    assert passed.passed is True
    assert failed.passed is False


def test_audit_cli_actions_are_lazy_and_registered() -> None:
    parser = _parser()
    common = [
        "research",
        "optical-manifold",
        "reconstruction-audit-validate",
        "--config",
        CONFIG,
        "--review-contract",
        REVIEW,
        "--source-training-evidence",
        SOURCE_TRAINING,
        "--source-checkpoint",
        SOURCE_CHECKPOINT,
        "--dataset-root",
        "data",
        "--output",
        "validation.json",
    ]
    args = parser.parse_args(common)
    assert args.research_action == "reconstruction-audit-validate"
    run_args = parser.parse_args(
        [
            "research",
            "optical-manifold",
            "reconstruction-audit-run",
            "--config",
            CONFIG,
            "--review-contract",
            REVIEW,
            "--validation-evidence",
            "validation.json",
            "--source-training-evidence",
            SOURCE_TRAINING,
            "--source-checkpoint",
            SOURCE_CHECKPOINT,
            "--dataset-root",
            "data",
            "--output",
            "audit.json",
        ]
    )
    assert run_args.research_action == "reconstruction-audit-run"
    source = inspect.getsource(audit)
    assert "torch.optim" not in source


@pytest.mark.historical_artifact
def test_validation_artifact_has_no_update_boundary(tmp_path, monkeypatch) -> None:
    state = audit._git_state()
    monkeypatch.setattr(
        audit,
        "_git_state",
        lambda: {"commit": state["commit"], "branch": audit.BRANCH, "clean": True, "status": []},
    )
    args = argparse.Namespace(
        config=CONFIG,
        review_contract=REVIEW,
        source_training_evidence=SOURCE_TRAINING,
        source_checkpoint=SOURCE_CHECKPOINT,
        dataset_root="data",
        output=str(tmp_path / "validation.json"),
    )
    assert audit.validate(args) == 0
    payload = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert payload["schema"] == audit.VALIDATION_SCHEMA
    assert payload["no_update_audit_authorized"] is True
    assert payload["training_authorized"] is False
    assert payload["formal_promotion"] is False
