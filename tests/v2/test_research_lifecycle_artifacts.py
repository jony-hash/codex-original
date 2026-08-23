from __future__ import annotations

import json
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from ldm_project.v2.application.research import electronic_o0_latent_transport as latent_transport
from ldm_project.v2.application.research import electronic_o0_mixture_transport as mixture_transport

from ldm_project.v2.application.research.lifecycle_artifacts import (
    POSTERIOR_CONTRACT,
    load_manifest,
    load_posterior,
    reuse_decision,
    stage_action_decision,
    validate_evidence_inputs,
    write_failure_record,
    write_posterior,
)
from ldm_project.v2.application.research.transport_canary import run_flow_canary


def test_posterior_is_content_bound_and_reusable(tmp_path):
    artifact = tmp_path / "encoded_posterior.v1.pt"
    metadata = {
        "producer_contract": "electronic_o0_latent_transport.posterior.v1",
        "posterior_contract_revision": POSTERIOR_CONTRACT,
        "encoder_sha256": "encoder",
        "config_sha256": "config",
        "dataset_root": str(tmp_path),
        "dataset_split_sha256": "split",
        "code_dim": 8,
        "dtype": "torch.float32",
        "train_count": 2,
        "validation_count": 1,
        "row_order": "stratified_mnist_indices.v1",
    }
    tensors = {
        "train_images": torch.zeros(2, 1, 28, 28),
        "train_labels": torch.tensor([0, 1]),
        "validation_images": torch.zeros(1, 1, 28, 28),
        "validation_labels": torch.tensor([2]),
        "train_codes": torch.zeros(2, 8),
        "validation_codes": torch.zeros(1, 8),
    }
    digest = write_posterior(artifact, tensors, metadata)
    assert digest
    assert set(load_posterior(artifact, expected=metadata)) == set(tensors)
    assert reuse_decision(
        "posterior",
        producer_contract=metadata["producer_contract"],
        expected_inputs=metadata,
        artifact=artifact,
    )["decision"] == "reuse"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        load_posterior(artifact, expected=metadata)
    artifact.unlink()
    with pytest.raises(ValueError):
        load_posterior(artifact, expected=metadata)


def test_posterior_orphan_is_recovered_only_for_exact_content(tmp_path):
    artifact = tmp_path / "encoded_posterior.v1.pt"
    tensors = {"train_codes": torch.zeros(2, 8)}
    metadata = {"producer_contract": "posterior.v1", "code_dim": 8}
    write_posterior(artifact, tensors, metadata)
    artifact.with_suffix(".json").unlink()
    write_posterior(artifact, tensors, metadata)
    assert load_posterior(artifact, expected=metadata)["train_codes"].shape == (2, 8)
    artifact.with_suffix(".json").unlink()
    with pytest.raises(FileExistsError):
        write_posterior(artifact, {"train_codes": torch.ones(2, 8)}, metadata)


def test_stage_reuse_decision_reuses_exact_contract_and_rejects_stale(tmp_path):
    evidence = tmp_path / "fit.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "fit.v1",
                "status": "valid",
                "producer_contract_revision": "fit.v1",
                "input_artifact_hashes": {"posterior": "posterior-hash"},
            }
        ),
        encoding="utf-8",
    )
    decision = stage_action_decision(
        "fit",
        producer_contract="fit.v1",
        expected_inputs={"posterior": "posterior-hash"},
        evidence=evidence,
    )
    assert decision["action"] == "REUSE"
    with pytest.raises(ValueError, match="earliest invalid stage=fit"):
        stage_action_decision(
            "fit",
            producer_contract="fit.v1",
            expected_inputs={"posterior": "changed"},
            evidence=evidence,
        )
def test_failed_run_records_are_immutable_attempts(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema: research_run_manifest.v1\n", encoding="utf-8")
    manifest = {"output_root": str(tmp_path / "runs"), "_manifest_path": str(manifest_path)}
    first = write_failure_record(
        manifest,
        stage="screen",
        exc=RuntimeError("first"),
        completed={"posterior": {"sha256": "abc", "reusable": True, "reuse_reason": "exact_contract_match"}},
    )
    second = write_failure_record(manifest, stage="screen", exc=RuntimeError("second"))
    assert first != second
    assert (first / "failure.json").is_file()
    assert (second / "completed-artifacts.json").is_file()
    failure = json.loads((first / "failure.json").read_text(encoding="utf-8"))
    assert failure["scientific_classification"] is None
    assert failure["permitted_next_actions"] == []
    assert failure["completed_artifacts"]["posterior"]["reuse_reason"] == "exact_contract_match"
    assert (first / "safe-error.txt").read_text(encoding="utf-8") == "RuntimeError: first\n"


def _write_measurement(path: Path, *, schema: str, producer: str, consumer: str = "closeout", permitted: list[str] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "status": "valid",
                "producer_contract_revision": producer,
                "consumer_contract": consumer,
                "invocation_mode": "manifest",
                "legacy_diagnostic_only": False,
                "permitted_consumers": permitted or ["closeout"],
                "passed": False,
            }
        ),
        encoding="utf-8",
    )


def test_manifest_closeout_dispatch_is_pure_consumer(tmp_path, monkeypatch):
    latent_evidence = tmp_path / "latent-measurement.json"
    mixture_evidence = tmp_path / "mixture-measurement.json"
    _write_measurement(latent_evidence, schema="electronic_o0_latent_transport_measure-flow.v1", producer="latent-measurement.v1")
    _write_measurement(mixture_evidence, schema="electronic_o0_mixture_transport_screen-measure.v1", producer="mixture-measurement.v1")
    config_latent = "configs/research/electronic_o0_latent_transport_v1.yaml"
    config_mixture = "configs/research/electronic_o0_mixture_transport_v1.yaml"
    monkeypatch.setattr(latent_transport, "_INVOCATION_MODE", "manifest")
    monkeypatch.setattr(mixture_transport, "_INVOCATION_MODE", "manifest")
    for module in (latent_transport, mixture_transport):
        monkeypatch.setattr(module, "_start_cuda", lambda args: pytest.fail("closeout initialized CUDA"))
        monkeypatch.setattr(module, "_load_wae", lambda *args, **kwargs: pytest.fail("closeout loaded a model"))
    latent_args = Namespace(
        research_action="closeout",
        config=config_latent,
        measurement_evidence=str(latent_evidence),
        locked_evidence=None,
        output=str(tmp_path / "latent-closeout.json"),
        _run_manifest={"manifest": True},
    )
    mixture_args = Namespace(
        research_action="closeout",
        config=config_mixture,
        measurement_evidence=str(mixture_evidence),
        locked_evidence=None,
        screen_evidence=None,
        output=str(tmp_path / "mixture-closeout.json"),
        _run_manifest={"manifest": True},
    )
    assert latent_transport._dispatch(latent_args) == 0
    assert mixture_transport._dispatch(mixture_args) == 0


def test_manifest_consumer_rejects_legacy_and_wrong_consumer_before_runtime(tmp_path):
    evidence = tmp_path / "measurement.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_latent_transport_measure-flow.v1",
                "status": "valid",
                "producer_contract_revision": "latent-measurement.v1",
                "consumer_contract": "latent-closeout.v1",
                "invocation_mode": "legacy",
                "legacy_diagnostic_only": True,
                "permitted_consumers": ["report"],
            }
        ),
        encoding="utf-8",
    )
    namespace = Namespace(_run_manifest={"manifest": True}, measurement_evidence=str(evidence), posterior_artifact=None)
    with pytest.raises(ValueError, match="legacy"):
        validate_evidence_inputs(namespace, program="electronic-o0-latent-transport", action="closeout")


def test_manifest_fit_reuse_returns_before_cuda_or_fitting(tmp_path, monkeypatch):
    config = "configs/research/electronic_o0_latent_transport_v1.yaml"
    training = tmp_path / "training.pt"
    decoder = tmp_path / "decoder.pt"
    posterior = tmp_path / "posterior.pt"
    preflight = tmp_path / "preflight.json"
    output = tmp_path / "fit.json"
    for path in (training, decoder, posterior):
        path.write_bytes(path.name.encode("utf-8"))
    preflight.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_latent_transport_preflight.v1",
                "status": "valid",
                "producer_contract_revision": "latent-preflight.v1",
                "consumer_contract": "latent-fit-flow.v1",
                "invocation_mode": "manifest",
                "legacy_diagnostic_only": False,
                "permitted_consumers": ["fit-flow"],
                "input_artifact_hashes": {"posterior": hashlib.sha256(posterior.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    posterior_hash = hashlib.sha256(posterior.read_bytes()).hexdigest()
    preflight_hash = hashlib.sha256(preflight.read_bytes()).hexdigest()
    output.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_latent_transport_fit-flow.v1",
                "status": "valid",
                "producer_contract_revision": "latent-fit-flow.v1",
                "consumer_contract": "latent-flow-fit.v1",
                "input_artifact_hashes": {"posterior": posterior_hash, "preflight": preflight_hash},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(latent_transport, "_start_cuda", lambda args: pytest.fail("reuse initialized CUDA"))
    monkeypatch.setattr(latent_transport, "_flow_train", lambda *args, **kwargs: pytest.fail("reuse fitted flow"))
    monkeypatch.setattr(latent_transport, "_posterior_for_args", lambda *args, **kwargs: {})
    args = Namespace(
        research_action="fit-flow",
        config=config,
        dataset_root="data",
        device="cuda",
        output=str(output),
        posterior_artifact=str(posterior),
        wae_training_checkpoint=str(training),
        wae_decoder_checkpoint=str(decoder),
        preflight_evidence=str(preflight),
        resume_checkpoint=None,
        seed=8401,
        wae_confirm_evidence=None,
        _run_manifest={"manifest": True},
        _validated_posterior={},
    )
    assert latent_transport._dispatch(args) == 0


@pytest.mark.scheduled_diagnostic
def test_bounded_cuda_canary_replays_and_rejects_tamper(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the scheduled canary")
    labels = torch.arange(10).repeat_interleave(5500)
    posterior = {
        "train_images": torch.zeros(55000, 1, 28, 28),
        "train_labels": labels,
        "validation_images": torch.zeros(5000, 1, 28, 28),
        "validation_labels": torch.arange(10).repeat_interleave(500),
        "train_codes": torch.zeros(55000, 8),
        "validation_codes": torch.zeros(5000, 8),
    }
    result = run_flow_canary(posterior, device=torch.device("cuda"), temporary_root=tmp_path)
    assert result["status"] == "valid"
    assert result["replay_exact"] is True
    assert result["tamper_rejected"] is True


def test_manifest_rejects_missing_inputs_before_runtime(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "run.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema: research_run_manifest.v1",
                "program: electronic-o0-latent-transport",
                "action: encode-posterior",
                f"config: {config}",
                f"dataset_root: {dataset}",
                "device: cuda",
                "deterministic:",
                "  cublas_workspace_config: ':4096:8'",
                "inputs:",
                "  checkpoints: {}",
                "  evaluator_evidence: {}",
                "  upstream_evidence: {}",
                "artifacts:",
                "  posterior: {}",
                "  fit: {}",
                "  calibration: {}",
                "  measurement: {}",
                "seeds: {}",
                "output_root: " + str(tmp_path / "20260823_harness"),
            ]
        ),
        encoding="utf-8",
    )
    assert load_manifest(manifest, program="electronic-o0-latent-transport", action="encode-posterior")["schema"] == "research_run_manifest.v1"
    missing = tmp_path / "missing.pt"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("  checkpoints: {}", f"  checkpoints:\n    encoder:\n      path: {missing}\n      role: encoder\n      sha256: {'0' * 64}"), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_manifest(manifest, program="electronic-o0-latent-transport", action="encode-posterior")
