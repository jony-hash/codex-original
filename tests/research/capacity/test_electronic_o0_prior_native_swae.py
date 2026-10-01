from __future__ import annotations

import json
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.capacity.electronic_o0_prior_native_swae import CODE_DIM, PROJECTION_COUNT, _apply_complete_o0_gate, _closeout, _objective_hash, _optimizer, _prior_native_terms, _schedule_hash, _validate_anchor, balanced_epoch_schedule, classwise_sliced_wasserstein_quantile, fixed_projections, gaussian_quantiles, _stage_decision
from ldm_project.v2.application.research.capacity.electronic_o0_wae import ConditionalWAEEncoder, ConditionalWAEDecoder


def _balanced_codes(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((100, CODE_DIM), generator=generator), torch.arange(
        10
    ).repeat_interleave(10)


def test_fixed_projection_and_quantile_contract_is_deterministic() -> None:
    first = fixed_projections(device=torch.device("cpu"))
    second = fixed_projections(device=torch.device("cpu"))
    assert first.shape == (PROJECTION_COUNT, CODE_DIM)
    assert torch.equal(first, second)
    assert torch.allclose(
        first.square().sum(1), torch.ones(PROJECTION_COUNT), atol=1e-6
    )
    assert torch.isfinite(
        gaussian_quantiles(10, device=torch.device("cpu"), dtype=torch.float32)
    ).all()


def test_classwise_swd_detects_shift_and_scale() -> None:
    codes, labels = _balanced_codes()
    projections = fixed_projections(device=torch.device("cpu"))
    baseline = classwise_sliced_wasserstein_quantile(
        codes, labels, projections=projections
    )
    shifted = classwise_sliced_wasserstein_quantile(
        codes + 2.0, labels, projections=projections
    )
    scaled = classwise_sliced_wasserstein_quantile(
        codes * 2.0, labels, projections=projections
    )
    assert torch.isfinite(baseline)
    assert shifted > baseline
    assert scaled > baseline


def test_prior_native_terms_have_no_mmd_and_have_finite_gradients() -> None:
    torch.manual_seed(12)
    encoder = ConditionalWAEEncoder()
    decoder = ConditionalWAEDecoder("O0-I")
    images = torch.rand(100, 1, 28, 28)
    labels = torch.arange(10).repeat_interleave(10)
    prior = torch.randn(100, CODE_DIM)
    total, terms = _prior_native_terms(
        encoder,
        decoder,
        images,
        labels,
        prior,
        w_swd=0.5,
        w_cycle=0.25,
        projections=fixed_projections(device=torch.device("cpu")),
    )
    assert set(terms) == {"reconstruction", "swd", "cycle", "encoded"}
    assert torch.isfinite(total)
    total.backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_new_optimizer_is_fresh_and_not_source_optimizer() -> None:
    encoder = ConditionalWAEEncoder()
    decoder = ConditionalWAEDecoder("O0-I")
    optimizer = _optimizer(torch.device("cpu"), encoder, decoder)
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["lr"] == 2.0e-4


def test_closeout_is_a_pure_consumer(tmp_path) -> None:
    measurement = tmp_path / "measurement.json"
    measurement.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_prior_native_swae_locked-final.v1",
                "status": "valid",
                "passed": False,
                "classification": "prior_alignment_semantic_or_novelty_failed",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "closeout.json"
    args = Namespace(output=str(output))
    assert _closeout(args, str(measurement), output) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == "prior_alignment_semantic_or_novelty_failed"
    assert payload["formal_promotion"] is False
    assert payload["optical_training_authorized"] is False
    assert payload["fiber_training_authorized"] is False


def test_dispatch_rejects_legacy_invocation() -> None:
    from ldm_project.v2.application.research.capacity import electronic_o0_prior_native_swae

    with pytest.raises(ValueError, match="manifest-only"):
        electronic_o0_prior_native_swae.dispatch(
            Namespace(research_action="validate", run_manifest=None)
        )


def test_manifest_resolves_new_program_and_action(tmp_path: Path) -> None:
    from ldm_project.v2.application.research.shared.lifecycle_artifacts import load_manifest, manifest_args

    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    dataset = tmp_path / "data"
    dataset.mkdir()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "run.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema: research_run_manifest.v1",
                "program: electronic-o0-prior-native-swae",
                "action: validate",
                f"config: {config}",
                f"dataset_root: {dataset}",
                "device: cuda",
                "deterministic:",
                "  cublas_workspace_config: ':4096:8'",
                "inputs:",
                "  checkpoints:",
                f"    source:\n      path: {source}\n      role: source\n      sha256: {digest}",
                "  evaluator_evidence: {}",
                "  upstream_evidence: {}",
                "artifacts:",
                "  posterior: {}",
                "  fit: {}",
                "  calibration: {}",
                "  measurement: {}",
                "  validation:",
                f"    path: {tmp_path / 'validation.json'}",
                "seeds:",
                "  swae: 8601",
                f"output_root: {tmp_path / 'run'}",
            ]
        ),
        encoding="utf-8",
    )
    namespace = Namespace(
        run_manifest=str(manifest),
        config=None,
        dataset_root=None,
        device=None,
        output=None,
    )
    loaded = load_manifest(
        manifest, program="electronic-o0-prior-native-swae", action="validate"
    )
    expanded = manifest_args(
        namespace,
        program="electronic-o0-prior-native-swae",
        action="validate",
        field_names=("config", "dataset_root", "device", "output"),
    )
    assert loaded["program"] == "electronic-o0-prior-native-swae"
    assert expanded.seed == 8601


def test_stage_reuse_and_invalid_existing_evidence_are_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit.json"
    output.write_text(
        json.dumps(
            {
                "status": "valid",
                "producer_contract_revision": "contract.v1",
                "input_artifact_hashes": {"source": "source-hash"},
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(_run_manifest={"manifest": True}, output=str(output))
    decision = _stage_decision(
        args,
        "audit",
        "contract.v1",
        {"source": "source-hash"},
    )
    assert decision is not None
    assert decision["action"] == "REUSE"

    output.write_text(json.dumps({"status": "invalid_harness"}), encoding="utf-8")
    with pytest.raises(ValueError, match="existing evidence is not valid"):
        _stage_decision(
            args,
            "audit",
            "contract.v1",
            {"source": "source-hash"},
        )


def test_manifest_resolves_checkpoint_and_peer_list(tmp_path: Path) -> None:
    from ldm_project.v2.application.research.shared.lifecycle_artifacts import load_manifest, manifest_args

    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    dataset = tmp_path / "data"
    dataset.mkdir()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "run.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema: research_run_manifest.v1",
                "program: electronic-o0-prior-native-swae",
                "action: locked-final",
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
                f"  checkpoint:\n    path: {source}\n    role: checkpoint\n    sha256: {digest}",
                f"  peer_confirm_evidence:\n    - path: {source}\n      role: peer_confirm_evidence[0]\n      sha256: {digest}",
                "seeds:",
                "  swae: 8601",
                f"output_root: {tmp_path / 'run'}",
            ]
        ),
        encoding="utf-8",
    )
    namespace = Namespace(
        run_manifest=str(manifest),
        config=None,
        dataset_root=None,
        device=None,
        output=None,
        checkpoint=None,
        resume_checkpoint=None,
        anchor_confirm_evidence=None,
        peer_confirm_evidence=[],
    )
    loaded = load_manifest(
        manifest, program="electronic-o0-prior-native-swae", action="locked-final"
    )
    expanded = manifest_args(
        namespace,
        program="electronic-o0-prior-native-swae",
        action="locked-final",
        field_names=(
            "config",
            "dataset_root",
            "device",
            "output",
            "checkpoint",
            "resume_checkpoint",
            "anchor_confirm_evidence",
            "peer_confirm_evidence",
        ),
    )
    assert loaded["program"] == "electronic-o0-prior-native-swae"
    assert expanded.checkpoint == str(source)
    assert expanded.peer_confirm_evidence == [str(source)]


def test_schedule_and_objective_hashes_are_deterministic() -> None:
    labels = torch.arange(10).repeat_interleave(10)
    first = balanced_epoch_schedule(labels, seed=8601)
    second = balanced_epoch_schedule(labels, seed=8601)
    assert _schedule_hash(first) == _schedule_hash(second)
    assert _objective_hash({"w_swd": 1.0, "w_cycle": 2.0}) == _objective_hash(
        {"w_swd": 1.0, "w_cycle": 2.0}
    )
    assert _objective_hash({"w_swd": 1.0, "w_cycle": 2.0}) != _objective_hash(
        {"w_swd": 1.1, "w_cycle": 2.0}
    )


def test_anchor_requires_admitted_seed_8601(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_prior_native_swae_confirm.v1",
                "status": "valid",
                "classification": "electronic_prior_native_swae_admitted",
                "seed": 8601,
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(seed=8602, anchor_confirm_evidence=str(anchor))
    _validate_anchor(args)
    args.anchor_confirm_evidence = None
    with pytest.raises(ValueError, match="anchor confirmation"):
        _validate_anchor(args)


def test_complete_gate_rejects_missing_jacobian_evidence() -> None:
    measurement = {
        "gates": {"panel": True},
        "official_test": {"count": 10_000},
        "jacobian": {"rank_1e-3_pass_fraction": 0.0},
        "memorization": {"pass_fraction": 1.0, "exact_duplicate_count": 0},
    }
    result = _apply_complete_o0_gate(measurement)
    assert result["passed"] is False
    assert result["gates"]["jacobian_rank"] is False
