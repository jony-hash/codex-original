from __future__ import annotations

import torch

from ldm_project.v2.application.research.capacity.electronic_o0_wae import BOOTSTRAP_SEED, LOCKED_SEEDS, TRAIN_SEEDS, classify_o0, balanced_epoch_schedule, balanced_schedule_summary, panel_codes
from ldm_project.v2.application.research.capacity.wae_models import FIXED_EXPOSURE_SCALE, ConditionalWAEEncoder, ConditionalWAEDecoder, parameter_counts, render_view, wae_loss


def test_wae_shapes_and_inference_budget() -> None:
    encoder = ConditionalWAEEncoder()
    image_decoder = ConditionalWAEDecoder("O0-I")
    command_decoder = ConditionalWAEDecoder("O0-G")
    codes = torch.randn(20, 8)
    labels = torch.arange(10).repeat_interleave(2)
    assert encoder(torch.randn(20, 1, 28, 28), labels).shape == (20, 8)
    images = image_decoder(codes, labels)
    commands = command_decoder(codes, labels)
    assert images.shape == (20, 1, 28, 28)
    assert commands.shape == (20, 1, 200, 200)
    assert render_view(commands, "O0-G").shape == (20, 1, 28, 28)
    counts = parameter_counts(encoder, command_decoder)
    assert counts["total"] <= 8_000_000
    assert counts["decoder"] <= 6_000_000
    assert FIXED_EXPOSURE_SCALE == 0.4913053810596466


def test_registered_seeds_and_panels_are_disjoint_and_deterministic() -> None:
    assert TRAIN_SEEDS == (7301, 7302, 7303)
    assert LOCKED_SEEDS == (4242, 104729, 161803)
    assert BOOTSTRAP_SEED == 90210
    codes_a, labels_a = panel_codes(4242, device=torch.device("cpu"))
    codes_b, labels_b = panel_codes(4242, device=torch.device("cpu"))
    assert torch.equal(codes_a, codes_b)
    assert torch.equal(labels_a, labels_b)
    assert labels_a.bincount().tolist() == [50] * 10


def test_o0_classification_is_fail_closed() -> None:
    assert classify_o0({"passed": True}, candidate="O0-I") == (
        "electronic_image_manifold_admitted",
        "run_wae_o0_g",
    )
    assert classify_o0({"passed": False}, candidate="O0-I")[0] == "wae_image_manifold_failed"
    assert classify_o0({"status": "invalid_harness"}, candidate="O0-G")[0] == "invalid_harness"
    assert classify_o0({"passed": True, "o0_i_baseline": None}, candidate="O0-G")[0] == "command_domain_bridge_failed"


def test_reconstruction_uses_encoded_latent_not_prior_code() -> None:
    torch.manual_seed(7)
    encoder = ConditionalWAEEncoder()
    decoder = ConditionalWAEDecoder("O0-I")
    images = torch.rand(20, 1, 28, 28)
    labels = torch.arange(10).repeat_interleave(2)
    prior_a = torch.zeros(20, 8)
    prior_b = torch.ones(20, 8)
    torch.manual_seed(101)
    loss_a, _ = wae_loss(
        decoder, encoder, images, labels, prior_a, w_mmd=0.0, w_cycle=0.0
    )
    torch.manual_seed(101)
    loss_b, _ = wae_loss(
        decoder, encoder, images, labels, prior_b, w_mmd=0.0, w_cycle=0.0
    )
    assert torch.equal(loss_a, loss_b)


def test_balanced_epoch_is_full_coverage_and_deterministic() -> None:
    labels = torch.tensor([label for label in range(10) for _ in range(11 + label)])
    first = balanced_epoch_schedule(labels, seed=73)
    second = balanced_epoch_schedule(labels, seed=73)
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    summary = balanced_schedule_summary(labels, first)
    assert summary["unique_samples"] == labels.numel()
    assert summary["minimum_coverage"] >= 1
    assert summary["steps"] == 2
