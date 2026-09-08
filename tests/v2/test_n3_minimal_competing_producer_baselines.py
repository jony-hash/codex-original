from pathlib import Path

import pytest
import torch

from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison


def test_development_fold_assignment_is_deterministic_and_class_balanced() -> None:
    source_indices = torch.arange(comparison.POOL_COUNT, dtype=torch.int64)
    labels = torch.arange(comparison.CLASS_COUNT, dtype=torch.int64).repeat_interleave(
        comparison.ROWS_PER_CLASS
    )
    first, first_manifest = comparison._assign_folds(source_indices, labels)
    second, second_manifest = comparison._assign_folds(source_indices, labels)

    assert torch.equal(first, second)
    assert first_manifest["assignment_sha256"] == second_manifest["assignment_sha256"]
    assert torch.equal(
        torch.bincount(first, minlength=comparison.FOLD_COUNT),
        torch.full((comparison.FOLD_COUNT,), comparison.HELDOUT_COUNT_PER_FOLD, dtype=torch.int64),
    )
    for fold in range(comparison.FOLD_COUNT):
        heldout = labels[first == fold]
        assert torch.equal(
            torch.bincount(heldout, minlength=comparison.CLASS_COUNT),
            torch.full(
                (comparison.CLASS_COUNT,),
                comparison.HELDOUT_ROWS_PER_CLASS_PER_FOLD,
                dtype=torch.int64,
            ),
        )


def test_analytic_shrinkage_is_fit_only_bounded_and_positive_definite() -> None:
    generator = torch.Generator(device="cpu").manual_seed(19)
    q_fit = torch.rand((comparison.FIT_COUNT_PER_FOLD, comparison.DIMENSION), generator=generator, dtype=torch.float64)
    q_fit = q_fit + 0.2

    fitted = comparison._fit_shrinkage(q_fit)

    assert 0.0 <= float(fitted["alpha"]) <= 1.0
    assert torch.allclose(fitted["sigma"], fitted["sigma"].transpose(0, 1))
    assert torch.linalg.eigvalsh(fitted["sigma"]).min() > 0.0
    assert fitted["fit_count"] == comparison.FIT_COUNT_PER_FOLD


def test_existing_development_lineage_is_consumable() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    q, source_indices, labels, identity = comparison._load_development_pool(repo_root)

    assert q.shape == (comparison.POOL_COUNT, comparison.DIMENSION)
    assert source_indices.shape == (comparison.POOL_COUNT,)
    assert labels.shape == (comparison.POOL_COUNT,)
    assert identity["fit_and_internal_validation_union"] is True


def test_stale_manifest_digest_is_rejected_before_consumption() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="hash changed"):
        comparison._read_json_exact(
            repo_root,
            comparison.SOURCE_MANIFEST,
            "0" * 64,
            "tampered manifest",
        )
