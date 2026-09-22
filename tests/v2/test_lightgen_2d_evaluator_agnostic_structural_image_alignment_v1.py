from __future__ import annotations

import torch

from scripts.research.execute_lightgen_2d_evaluator_agnostic_structural_image_alignment_v1 import (
    STRUCTURAL_DIMENSION,
    StructuralContinuationSampler,
    _fit_surrogate_normalization,
    _structural_surrogate,
)


def test_structural_surrogate_is_finite_deterministic_and_dimensioned() -> None:
    images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
    images[1, 0, 7:21, 7:21] = 1.0
    first = _structural_surrogate(images)
    second = _structural_surrogate(images.clone())
    assert first.shape == (2, STRUCTURAL_DIMENSION)
    assert first.dtype == torch.float64
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_structural_surrogate_has_a_finite_image_gradient() -> None:
    images = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    images[0, 0, 8:20, 8:20] = 0.75
    images.requires_grad_()
    loss = _structural_surrogate(images).square().mean()
    gradient = torch.autograd.grad(loss, images)[0]
    assert gradient.shape == images.shape
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_surrogate_normalization_is_train_only_and_clamps_zero_variance() -> None:
    images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
    images[::2, 0, 8:20, 8:20] = 1.0
    mean, scale = _fit_surrogate_normalization(images)
    assert mean.shape == (STRUCTURAL_DIMENSION,)
    assert scale.shape == (STRUCTURAL_DIMENSION,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(scale).all()
    assert bool((scale >= 1.0e-6).all())


def test_continuation_sampler_preserves_balanced_class_contract() -> None:
    state = {
        "seed": 1234,
        "permutations": [torch.arange(6000, dtype=torch.long) + digit * 10000 for digit in range(10)],
    }
    sampler = StructuralContinuationSampler(state, cursor=6000)
    indices, labels = sampler.next_batch()
    assert labels.tolist() == list(range(10))
    assert indices.tolist() == [digit * 10000 for digit in range(10)]
    assert sampler.cursor == 6001
