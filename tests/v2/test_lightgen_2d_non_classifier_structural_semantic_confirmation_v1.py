from __future__ import annotations

import torch

from scripts.research.execute_lightgen_2d_non_classifier_structural_semantic_confirmation_v1 import (
    CLASS_COUNT,
    SIGNATURE_DIMENSION,
    _make_structural_split,
    _standardize,
    _topology,
    _topology_summary,
    _bootstrap_rate,
    structural_signature,
)


def test_structural_signature_is_deterministic_finite_and_dimensioned() -> None:
    images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
    images[1, 0, 7:21, 7:21] = 1.0
    first = structural_signature(images, device=torch.device("cpu"), batch_size=1)
    second = structural_signature(images, device=torch.device("cpu"), batch_size=2)
    assert first.shape == (2, SIGNATURE_DIMENSION)
    assert first.dtype == torch.float64
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_topology_uses_padded_background_and_counts_a_ring_hole() -> None:
    mask = torch.zeros((1, 28, 28), dtype=torch.bool)
    mask[:, 7:21, 7] = True
    mask[:, 7:21, 20] = True
    mask[:, 7, 7:21] = True
    mask[:, 20, 7:21] = True
    topology = _topology(mask)
    assert topology.shape == (1, 4)
    assert topology[0].tolist() == [1.0, 1.0, 0.0, 1.0]


def test_structural_split_is_class_stratified_and_disjoint() -> None:
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(20)
    split = _make_structural_split(labels)
    assert split["fit_positions"].numel() == 180
    assert split["validation_positions"].numel() == 20
    assert torch.unique(torch.cat([split["fit_positions"], split["validation_positions"]])).numel() == labels.numel()
    assert torch.bincount(split["fit_labels"], minlength=CLASS_COUNT).tolist() == [18] * CLASS_COUNT
    assert torch.bincount(split["validation_labels"], minlength=CLASS_COUNT).tolist() == [2] * CLASS_COUNT


def test_standardization_and_topology_summary_are_train_only_shapes() -> None:
    images = torch.zeros((10, 1, 28, 28), dtype=torch.float32)
    images[::2, 0, 8:20, 8:20] = 1.0
    signatures = structural_signature(images, device=torch.device("cpu"))
    fit, queries, mean, scale = _standardize(signatures[:8], {"query": signatures[8:]})
    assert fit.shape == (8, SIGNATURE_DIMENSION)
    assert queries["query"].shape == (2, SIGNATURE_DIMENSION)
    assert mean.shape == (SIGNATURE_DIMENSION,)
    assert scale.shape == (SIGNATURE_DIMENSION,)
    summary = _topology_summary(signatures, torch.arange(CLASS_COUNT, dtype=torch.long))
    assert set(summary) == {"threshold_0.25", "threshold_0.50", "threshold_0.75"}


def test_bootstrap_rate_is_reproducible_under_frozen_seed() -> None:
    values = torch.tensor([True, False, True, False] * CLASS_COUNT)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(4)
    first = _bootstrap_rate(values, labels, seed=271828, replicates=9)
    second = _bootstrap_rate(values, labels, seed=271828, replicates=9)
    assert first == second
