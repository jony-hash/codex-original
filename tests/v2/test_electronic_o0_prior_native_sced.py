from __future__ import annotations

import json
import math
from argparse import Namespace

import pytest
import torch

from ldm_project.v2.application.research import (
    electronic_o0_prior_native_sced as sced,
)
from ldm_project.v2.evidence import file_sha256

from ldm_project.v2.application.research.electronic_o0_prior_native_sced import (
    CODE_DIM,
    PROJECTION_COUNT,
    REFERENCE_PROJECTION_SEED,
    _ordered_pairwise_absolute_mean,
    _objective_hash,
    _synthetic_scale_losses,
    classwise_sliced_energy_gaussian,
    fixed_projections,
    gaussian_absolute_moment,
    _terms,
)
from ldm_project.v2.application.research.electronic_o0_prior_native_swae import (
    classwise_sliced_wasserstein_quantile,
)
from ldm_project.v2.application.research.wae_models import (
    ConditionalWAEDecoder,
    ConditionalWAEEncoder,
)


def _balanced_codes(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((100, CODE_DIM), generator=generator), torch.arange(10).repeat_interleave(10)


def test_sced_projection_bank_is_deterministic_and_normalized() -> None:
    first = fixed_projections(device=torch.device("cpu"))
    second = fixed_projections(device=torch.device("cpu"))
    assert first.shape == (PROJECTION_COUNT, CODE_DIM)
    assert torch.equal(first, second)
    assert torch.allclose(first.square().sum(1), torch.ones(PROJECTION_COUNT), atol=1e-6)
    reference = fixed_projections(seed=REFERENCE_PROJECTION_SEED, device=torch.device("cpu"))
    assert not torch.equal(first, reference)


def test_gaussian_absolute_moment_matches_standard_normal_at_zero() -> None:
    value = gaussian_absolute_moment(torch.zeros((), dtype=torch.float64))
    assert float(value) == pytest.approx(math.sqrt(2.0 / math.pi), abs=1e-12)


def test_sorted_u_statistic_matches_naive_values_and_gradients() -> None:
    generator = torch.Generator(device="cpu").manual_seed(19)
    projected = torch.randn((7, 13), generator=generator, dtype=torch.float64)
    projected.requires_grad_()
    efficient = _ordered_pairwise_absolute_mean(projected, dim=0)
    naive = (
        (projected[:, None, :] - projected[None, :, :]).abs().sum(dim=(0, 1))
        / (projected.shape[0] * (projected.shape[0] - 1))
    )
    assert torch.allclose(efficient, naive, atol=1.0e-12, rtol=0.0)
    efficient_gradient = torch.autograd.grad(efficient.sum(), projected, retain_graph=True)[0]
    naive_gradient = torch.autograd.grad(naive.sum(), projected)[0]
    assert torch.allclose(
        efficient_gradient, naive_gradient, atol=1.0e-12, rtol=0.0
    )


def test_vectorized_n10_audit_has_registered_scale_directions() -> None:
    generator = torch.Generator(device="cpu").manual_seed(271829)
    base = torch.randn((64, 10, 10, CODE_DIM), generator=generator)
    projected = base @ fixed_projections(device=torch.device("cpu")).t()
    sced_09, _ = _synthetic_scale_losses(projected, 0.9)
    sced_10, swd_10 = _synthetic_scale_losses(projected, 1.0)
    sced_11, _ = _synthetic_scale_losses(projected, 1.1)
    _, swd_shrunk = _synthetic_scale_losses(projected, 0.8343666503)
    assert sced_10 < sced_09
    assert sced_10 < sced_11
    assert float(swd_shrunk) + 0.02 < float(swd_10)


def test_sced_detects_shift_and_scale_with_finite_gradient() -> None:
    codes, labels = _balanced_codes()
    projections = fixed_projections(device=torch.device("cpu"))
    baseline = classwise_sliced_energy_gaussian(codes, labels, projections=projections)
    shifted = classwise_sliced_energy_gaussian(codes + 2.0, labels, projections=projections)
    scaled = classwise_sliced_energy_gaussian(codes * 2.0, labels, projections=projections)
    assert torch.isfinite(baseline)
    assert shifted > baseline
    assert scaled > baseline
    codes.requires_grad_()
    classwise_sliced_energy_gaussian(codes, labels, projections=projections).backward()
    assert codes.grad is not None
    assert torch.isfinite(codes.grad).all()
    assert float(codes.grad.norm()) > 0.0


def test_sced_keeps_the_finite_u_statistic_without_clamping() -> None:
    # The repeated sample has a finite, non-negative raw estimate.  Checking
    # the exact un-clamped expression guards against silently adding relu/max.
    codes = torch.zeros((100, CODE_DIM))
    labels = torch.arange(10).repeat_interleave(10)
    value = classwise_sliced_energy_gaussian(
        codes, labels, projections=fixed_projections(device=torch.device("cpu"))
    )
    assert torch.isfinite(value)
    expected = 2.0 * math.sqrt(2.0 / math.pi) - 2.0 / math.sqrt(math.pi)
    assert float(value) == pytest.approx(expected, abs=1e-6)


def test_reference_swd_retains_the_finite_batch_shrinkage_negative_control() -> None:
    generator = torch.Generator(device="cpu").manual_seed(271829)
    base = torch.randn((100, CODE_DIM), generator=generator)
    labels = torch.arange(10).repeat_interleave(10)
    projections = fixed_projections(device=torch.device("cpu"))
    low = classwise_sliced_wasserstein_quantile(
        base * 0.83436665, labels, projections=projections
    )
    unit = classwise_sliced_wasserstein_quantile(base, labels, projections=projections)
    assert float(low) + 0.02 < float(unit)


def test_sced_objective_has_finite_encoder_and_decoder_gradients() -> None:
    generator = torch.Generator(device="cpu").manual_seed(8601)
    images = torch.rand((100, 1, 28, 28), generator=generator)
    labels = torch.arange(10).repeat_interleave(10)
    encoder = ConditionalWAEEncoder()
    decoder = ConditionalWAEDecoder("O0-I")
    prior = torch.randn((100, CODE_DIM), generator=generator)
    total, terms = _terms(
        encoder,
        decoder,
        images,
        labels,
        prior,
        w_sced=0.5,
        w_cycle=0.25,
        projections=fixed_projections(device=torch.device("cpu")),
    )
    total.backward()
    assert all(torch.isfinite(value) for value in terms.values() if value.ndim == 0)
    encoder_gradients = [parameter.grad for parameter in encoder.parameters()]
    decoder_gradients = [parameter.grad for parameter in decoder.parameters()]
    assert any(gradient is not None for gradient in encoder_gradients)
    assert any(gradient is not None for gradient in decoder_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in (*encoder_gradients, *decoder_gradients)
        if gradient is not None
    )


def test_sced_rejects_invalid_class_batch_and_mismatched_projection_dtype() -> None:
    codes, labels = _balanced_codes()
    projections = fixed_projections(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="at least two samples"):
        classwise_sliced_energy_gaussian(
            codes[:91], labels[:91], projections=projections
        )
    with pytest.raises(ValueError, match="floating dtype"):
        classwise_sliced_energy_gaussian(
            codes.double(), labels, projections=projections
        )


def test_objective_hash_binds_objective_revision_and_weights() -> None:
    first = _objective_hash({"w_sced": 1.0, "w_cycle": 2.0})
    assert first == _objective_hash({"w_sced": 1.0, "w_cycle": 2.0})
    assert first != _objective_hash({"w_sced": 1.1, "w_cycle": 2.0})
    assert first != _objective_hash(
        {"w_sced": 1.0, "w_cycle": 2.0}, schedule_hash="different"
    )


def test_validate_reuse_returns_before_cuda(tmp_path, monkeypatch) -> None:
    training = tmp_path / "training.pt"
    decoder = tmp_path / "decoder.pt"
    closeout = tmp_path / "closeout.json"
    manifest = tmp_path / "manifest.yaml"
    for path in (training, decoder, closeout, manifest):
        path.write_bytes(path.name.encode("utf-8"))
    expected = {
        "source_training": file_sha256(training),
        "source_decoder": file_sha256(decoder),
        "h_swae_closeout": file_sha256(closeout),
        "lineage_sha256": sced.canonical_sha256({"git": {}}),
    }
    output = tmp_path / "validation.json"
    output.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_prior_native_sced_validation.v1",
                "status": "valid",
                "producer_contract_revision": "electronic_o0_prior_native_sced-validation.v1",
                "input_artifact_hashes": expected,
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        run_manifest=str(manifest),
        research_action="validate",
        output=str(output),
        config=str(tmp_path / "config.yaml"),
        wae_training_checkpoint=str(training),
        wae_decoder_checkpoint=str(decoder),
        h_swae_closeout=str(closeout),
    )
    monkeypatch.setattr(sced, "prepare_args", lambda namespace, **kwargs: namespace)
    monkeypatch.setattr(sced, "validate_manifest_namespace", lambda *args, **kwargs: None)
    monkeypatch.setattr(sced, "_config", lambda path: {})
    monkeypatch.setattr(sced, "_lineage", lambda namespace, config: {"git": {}})
    monkeypatch.setattr(
        sced, "_require_cuda", lambda namespace: pytest.fail("REUSE initialized CUDA")
    )
    assert sced.dispatch(args) == 0


def test_anchor_requires_admitted_seed_8601_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "anchor.pt"
    checkpoint.write_bytes(b"anchor")
    evidence = tmp_path / "anchor.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_prior_native_sced_confirm.v1",
                "status": "valid",
                "passed": True,
                "classification": "electronic_prior_native_sced_admitted",
                "seed": 8601,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "invocation_mode": "manifest",
                "legacy_diagnostic_only": False,
                "permitted_consumers": ["screen", "confirm", "locked-final"],
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(seed=8602, anchor_confirm_evidence=str(evidence))
    assert sced._anchor_confirm(args, action="screen")["seed"] == 8601
    checkpoint.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="missing or stale"):
        sced._anchor_confirm(args, action="screen")


def test_valid_failed_screen_cannot_be_confirmed(tmp_path) -> None:
    evidence = tmp_path / "screen.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_prior_native_sced_screen.v1",
                "status": "valid",
                "passed": False,
                "invocation_mode": "manifest",
                "legacy_diagnostic_only": False,
                "permitted_consumers": ["report", "closeout"],
            }
        ),
        encoding="utf-8",
    )
    assert sced._check_consumer(
        str(evidence),
        action="report",
        schemas=("electronic_o0_prior_native_sced_screen.v1",),
    )["passed"] is False
    with pytest.raises(ValueError, match="does not permit"):
        sced._check_consumer(
            str(evidence),
            action="confirm",
            schemas=("electronic_o0_prior_native_sced_screen.v1",),
        )


def test_snapshot_clones_optimizer_rng_sampler_and_partial_history() -> None:
    encoder = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *decoder.parameters()), lr=2.0e-4
    )
    loss = encoder(torch.ones(1, 2)).sum() + decoder(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    schedule = (torch.arange(10),)
    schedule_summary = {
        "steps": 1,
        "logical_samples": 10,
        "unique_samples": 10,
        "minimum_coverage": 1,
        "maximum_repeats": 1,
        "repeat_samples": 0,
    }
    partial = [{"total": 1.0, "reconstruction": 2.0, "sced": 3.0, "cycle": 4.0}]
    snapshot = sced._snapshot(
        encoder,
        decoder,
        optimizer,
        seed=8601,
        epoch=0,
        batch_cursor=1,
        step=1,
        weights={"w_sced": 0.5, "w_cycle": 0.25},
        schedule=schedule,
        schedule_summary=schedule_summary,
        history=[],
        history_window=partial,
        source_hash="source",
    )
    snapshot_encoder = {
        key: value.clone() for key, value in snapshot["encoder"].items()
    }
    optimizer.zero_grad(set_to_none=True)
    (encoder(torch.ones(1, 2)).sum() + decoder(torch.ones(1, 2)).sum()).backward()
    optimizer.step()
    assert sced._states_equal(snapshot["encoder"], snapshot_encoder)
    expected_random = torch.rand(4)
    restored_encoder = torch.nn.Linear(2, 2)
    restored_decoder = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(
        (*restored_encoder.parameters(), *restored_decoder.parameters()), lr=2.0e-4
    )
    sced._restore(
        snapshot,
        restored_encoder,
        restored_decoder,
        restored_optimizer,
        torch.device("cpu"),
    )
    assert sced._states_equal(snapshot["encoder"], restored_encoder.state_dict())
    assert sced._states_equal(snapshot["decoder"], restored_decoder.state_dict())
    assert sced._optimizer_state_equal(
        snapshot["optimizer"], restored_optimizer.state_dict()
    )
    assert torch.equal(expected_random, torch.rand(4))
    assert snapshot["sampler_state"]["cursor"] == 1
    assert snapshot["history_window"] == partial


def test_locked_final_requires_distinct_seeds_and_seed_8601_anchor(tmp_path) -> None:
    evidences = []
    checkpoints = []
    for seed in (8601, 8602, 8603):
        checkpoint = tmp_path / f"seed_{seed}.pt"
        checkpoint.write_bytes(str(seed).encode("utf-8"))
        evidence = tmp_path / f"seed_{seed}.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema": "electronic_o0_prior_native_sced_confirm.v1",
                    "status": "valid",
                    "passed": True,
                    "classification": "electronic_prior_native_sced_admitted",
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "invocation_mode": "manifest",
                    "legacy_diagnostic_only": False,
                    "permitted_consumers": ["locked-final"],
                }
            ),
            encoding="utf-8",
        )
        checkpoints.append(checkpoint)
        evidences.append(evidence)
    _, seeds = sced._locked_confirmations(
        str(evidences[0]),
        [str(evidences[1]), str(evidences[2])],
        str(checkpoints[0]),
    )
    assert set(seeds) == {8601, 8602, 8603}
    with pytest.raises(ValueError, match="seed-8601 anchor"):
        sced._locked_confirmations(
            str(evidences[0]),
            [str(evidences[1]), str(evidences[2])],
            str(checkpoints[1]),
        )
    duplicate = json.loads(evidences[2].read_text(encoding="utf-8"))
    duplicate["seed"] = 8602
    evidences[2].write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="three passed"):
        sced._locked_confirmations(
            str(evidences[0]),
            [str(evidences[1]), str(evidences[2])],
            str(checkpoints[0]),
        )
