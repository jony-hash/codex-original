from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research import execute_lightgen_2d_semantically_conditioned_ols_sampler_v1 as sampler


def test_requested_bank_is_balanced_and_paired() -> None:
    requested, sample_indices, seeds = sampler._make_requested_bank()
    assert requested.shape == (sampler.BANK_COUNT,)
    assert torch.bincount(requested, minlength=sampler.CLASS_COUNT).tolist() == [sampler.BANK_PER_CLASS] * sampler.CLASS_COUNT
    assert sample_indices[: sampler.BANK_PER_CLASS].tolist() == list(range(sampler.BANK_PER_CLASS))
    assert torch.unique(seeds).numel() == sampler.BANK_COUNT


def test_joint_model_is_bounded_and_seed_reproducible() -> None:
    torch.manual_seed(7)
    labels = torch.arange(sampler.CLASS_COUNT, dtype=torch.long).repeat_interleave(40)
    values = torch.randn((labels.numel(), 200), dtype=torch.float64)
    model = sampler._fit_joint_model(values, labels, representation="synthetic")
    assert all(0 <= item["rank"] <= sampler.PCA_RANK_CAP for item in model["per_class"])
    requested = torch.arange(sampler.CLASS_COUNT, dtype=torch.long).repeat_interleave(2)
    seeds = torch.arange(requested.numel(), dtype=torch.int64) + 100
    first = sampler._sample_joint_model(model, requested, seeds)
    second = sampler._sample_joint_model(model, requested, seeds)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_s3_reconstruction_rejects_zero_phase_pairs() -> None:
    value = torch.zeros((1, 300), dtype=torch.float64)
    scale = torch.ones(100, dtype=torch.float64)
    with pytest.raises(sampler.SamplerExecutionError, match="zero circular phase pair"):
        sampler._reconstruct_s3(value, scale)


def test_bootstrap_uses_class_stratified_requested_bank() -> None:
    requested, _, _ = sampler._make_requested_bank()
    correct = torch.zeros((1, sampler.BANK_COUNT), dtype=torch.float64)
    correct[:, ::2] = 1.0
    observed, samples = sampler._bootstrap_rates(correct, requested)
    assert observed.shape == (1,)
    assert samples.shape == (1, sampler.BOOTSTRAP_COUNT)
    assert float(observed[0]) == pytest.approx(0.5)


def test_executor_has_no_neural_update_path() -> None:
    source = Path(sampler.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.autograd" not in source
    assert "torch.optim" not in source
    assert "GENERATOR_CANARY" in source
    assert source.index("WINNER_FROZEN") < source.index("readiness._load_models")
