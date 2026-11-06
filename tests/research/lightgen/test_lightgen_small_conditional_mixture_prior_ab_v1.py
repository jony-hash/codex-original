from __future__ import annotations

import json

import pytest
import torch

import scripts.research.lightgen.execute_lightgen_small_conditional_mixture_prior_ab_v1 as research
import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base


def test_single_gaussian_and_k1_mixture_have_identical_sample_covariance() -> None:
    values = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [4.0, 3.0, 0.0],
                           [1.0, 5.0, 3.0], [3.0, 2.0, 5.0]], dtype=torch.float64)
    single = research._fit_single_class(values)
    mixture = research._fit_mixture_class(values, 1, seed=4, ridge=single["ridge"])
    assert torch.equal(mixture["weights"], torch.ones(1, dtype=torch.float64))
    assert torch.allclose(mixture["means"][0], single["mean"], rtol=0.0, atol=1e-14)
    assert torch.allclose(mixture["covariances"][0], single["covariance"], rtol=0.0, atol=1e-14)
    assert torch.allclose(mixture["choleskies"][0], single["cholesky"], rtol=0.0, atol=1e-14)
    reconstructed = mixture["choleskies"][0] @ mixture["choleskies"][0].T
    assert torch.allclose(reconstructed, mixture["regularized_covariances"][0], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(
        mixture["choleskies"][0].T @ mixture["choleskies"][0],
        mixture["regularized_covariances"][0], rtol=1e-8, atol=1e-10,
    )


def test_kmeans_plus_plus_is_seeded_and_rejects_empty_cluster_inputs() -> None:
    generator = torch.Generator().manual_seed(18)
    centers = torch.tensor([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]], dtype=torch.float64)
    values = torch.cat([center + 0.05 * torch.randn((40, 2), generator=generator, dtype=torch.float64)
                        for center in centers])
    labels_a, centers_a, _ = research._kmeans_labels(values, 4, seed=19)
    labels_b, centers_b, _ = research._kmeans_labels(values, 4, seed=19)
    assert torch.equal(labels_a, labels_b)
    assert torch.equal(centers_a, centers_b)
    assert bool((torch.bincount(labels_a, minlength=4) > 0).all())
    with pytest.raises(RuntimeError, match="zero remaining distance"):
        research._kmeans_labels(torch.ones((12, 3), dtype=torch.float64), 4, seed=3)


def test_fixed_power_rejection_is_applied_before_retaining_mixture_samples() -> None:
    scale = torch.ones(100, dtype=torch.float64)
    valid_prior = {
        "weights": torch.full((4,), 0.25, dtype=torch.float64),
        "means": torch.zeros((4, 200), dtype=torch.float64),
        "choleskies": torch.zeros((4, 200, 200), dtype=torch.float64),
    }
    coefficients, components, proposal_ids, records = research._sample_mixture_class(
        valid_prior, scale, label=2, sample_count=16, proposal_limit=16,
    )
    assert coefficients.shape == (16, 100)
    assert components.shape == proposal_ids.shape == (16,)
    assert records["power_rejected_candidates"] == 0
    assert bool((coefficients.abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all())

    invalid_prior = {**valid_prior, "means": torch.full((4, 200), 0.002, dtype=torch.float64)}
    with pytest.raises(RuntimeError, match="sampling incomplete"):
        research._sample_mixture_class(
            invalid_prior, scale, label=2, sample_count=1, proposal_limit=8,
        )


def test_numpy_independent_pool_matches_production_clamp_then_pool() -> None:
    generator = torch.Generator().manual_seed(32)
    native = torch.rand((10, 1, 200, 200), generator=generator)
    native[0, 0, 0, 0] = 1.4
    views = base._resize_view(native)
    cohort = {
        "requested_labels": torch.arange(10),
        "images": views,
        "native_capture_indices": torch.arange(10),
        "native_capture_images": native.clamp(0.0, 1.0),
    }
    result = research._validate_readout_numpy(cohort)
    assert result["sample_count"] == 10
    assert result["maximum_absolute_error"] <= 1e-6


def test_decode_cohort_returns_clipped_excess_from_production_readout() -> None:
    class Model(torch.nn.Module):
        def forward_from_ols(self, coefficients):
            values = torch.full((coefficients.shape[0], 1, 200, 200), 1.25,
                                dtype=torch.float32, device=coefficients.device)
            return {"physical_w_per_pixel": values}

    class EvaluatorD(torch.nn.Module):
        def forward(self, images):
            return torch.zeros((images.shape[0], 10), device=images.device)

    class EvaluatorC(torch.nn.Module):
        def forward(self, images):
            return torch.zeros((images.shape[0], 128), device=images.device), torch.zeros(
                (images.shape[0], 10), device=images.device
            )

    result = research._decode_cohort(
        "TEST", Model(), EvaluatorD(), EvaluatorC(), torch.zeros((10, 100), dtype=torch.complex64),
        torch.arange(10), torch.zeros(10, dtype=torch.long), torch.arange(10),
        torch.tensor(1.0), torch.device("cpu"), capture_count=1, expected_per_class=1,
    )

    assert result["proxy_clipped_excess_fraction"].shape == (10,)
    assert torch.allclose(result["proxy_clipping_area_fraction"], torch.ones(10, dtype=torch.float64))
    assert torch.allclose(result["proxy_clipped_excess_fraction"], torch.full((10,), 0.2, dtype=torch.float64))


def test_resume_and_feature_cache_reject_missing_or_misaligned_identity(tmp_path) -> None:
    identity = {"task_id": research.TASK_ID, "run_id": "run-1"}
    with pytest.raises(RuntimeError, match="resume requires"):
        research._validate_resume(tmp_path / "missing", identity)

    rows = torch.arange(55_000, dtype=torch.long)
    labels = torch.zeros(55_000, dtype=torch.long)
    bad_cache = {
        "task_id": research.TASK_ID,
        "source_run_id": "source-1",
        "rows": rows.flip(0),
        "labels": labels,
        "features_200": None,
    }
    with pytest.raises(RuntimeError, match="wrong source identity, rows, or labels"):
        research._validate_feature_cache(bad_cache, rows, labels, "source-1")

    run_dir = tmp_path / "ready-but-incomplete-preflight"
    run_dir.mkdir()
    complete_identity = {
        "run_id": "run-1",
        "config": {"data": {"train_label_counts": [1] * 10, "dev_per_class": 2}},
    }
    (run_dir / "run_identity.json").write_text(json.dumps(complete_identity))
    (run_dir / "preflight.json").write_text(json.dumps({
        "status": "READY", "run_id": "run-1",
        "data_roles": {"train_label_counts": [1] * 10, "dev_label_counts": [2] * 10},
    }))
    research._validate_resume(run_dir, complete_identity)
    (run_dir / "preflight.json").write_text(json.dumps({"status": "READY", "run_id": "run-1"}))
    with pytest.raises(RuntimeError, match="missing the fixed TRAIN/DEV role validation"):
        research._validate_resume(run_dir, complete_identity)


def test_data_role_consumer_accepts_fixed_unbalanced_train_and_rejects_overlap() -> None:
    train_counts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    train_labels = torch.tensor([label for label, count in enumerate(train_counts)
                                 for _ in range(count)], dtype=torch.long)
    dev_labels = torch.arange(10).repeat_interleave(2)
    train_rows = torch.arange(train_labels.numel())
    dev_rows = torch.arange(100, 120)
    contract = {
        "train_count": int(train_rows.numel()), "dev_count": int(dev_rows.numel()),
        "train_label_counts": train_counts, "dev_per_class": 2,
    }

    result = research._validate_data_roles(train_rows, dev_rows, train_labels, dev_labels, contract)

    assert result["train_label_counts"] == train_counts
    assert result["dev_label_counts"] == [2] * 10
    with pytest.raises(RuntimeError, match="duplicated or overlap"):
        research._validate_data_roles(train_rows, train_rows[:20], train_labels, dev_labels,
                                      {**contract, "dev_count": 20})

