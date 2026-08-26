from __future__ import annotations

import torch

from ldm_project.v1.application.teacher_failure_attribution import (
    _base,
    _config,
    _partition,
    _sample_seed,
)
from ldm_project.v1.application.teacher_distribution_admission import _noise


CONFIG = "configs/research/teacher_failure_attribution_v1.yaml"


def test_b2_config_locks_package_b_and_forbids_updates() -> None:
    config = _config(CONFIG)
    assert config["package_b_config_sha256"]
    assert config["latent_shape"] == [1, 25, 25]
    assert config["sampling_steps"] == 200
    assert config["microbatch_size"] == 16
    assert config["teacher_training_authorized"] is False
    assert config["student_training_authorized"] is False
    assert config["promotion_authorized"] is False


def test_b2_partition_is_strictly_disjoint_and_class_balanced() -> None:
    images = torch.arange(5000, dtype=torch.float32).reshape(5000, 1, 1, 1)
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(500)
    first, second = _partition(images, labels, seed=32452843, samples_per_class=250)

    assert first[0].shape == second[0].shape == (2500, 1, 1, 1)
    assert first[1].tolist() == second[1].tolist()
    assert all(int((first[1] == label).sum()) == 250 for label in range(10))
    assert all(int((second[1] == label).sum()) == 250 for label in range(10))
    assert set(first[0].flatten().tolist()).isdisjoint(set(second[0].flatten().tolist()))
    assert set(first[0].flatten().tolist()) | set(second[0].flatten().tolist()) == set(images.flatten().tolist())


def test_b2_noise_is_full_latent_gaussian_and_resume_deterministic() -> None:
    config = _config(CONFIG)
    seed_a = _sample_seed(config, bank=0, label=3, index=7)
    seed_b = _sample_seed(config, bank=1, label=3, index=7)
    assert seed_a == _sample_seed(config, bank=0, label=3, index=7)
    assert seed_a != seed_b
    first = _noise([seed_a, seed_b], (1, 25, 25))
    resumed = torch.cat((_noise([seed_a], (1, 25, 25)), _noise([seed_b], (1, 25, 25))))
    assert first.shape == (2, 1, 25, 25)
    assert torch.isfinite(first).all()
    assert torch.equal(first, resumed)


def test_b2_evidence_base_contains_required_fail_closed_fields() -> None:
    config = _config(CONFIG)
    payload = _base(
        "lineage",
        config,
        CONFIG,
        {"head": "test-head", "dirty_worktree": False, "status": []},
        status="valid",
        passed=True,
        classification="teacher_attribution_lineage_validated",
        first_failed_gate=None,
        permitted=["preflight"],
    )
    for key in (
        "classification",
        "first_failed_gate",
        "permitted_next_actions",
        "forbidden_next_actions",
        "input_artifact_hashes",
        "dataset_split_hash",
        "config_sha256",
        "git_head",
        "dirty_worktree",
        "seed",
        "evaluator_calls",
        "runtime_seconds",
        "cuda_peak_memory_bytes",
        "state_hash_before",
        "state_hash_after",
    ):
        assert key in payload
    assert payload["input_artifact_hashes"]["package_b_config"] == config["package_b_config_sha256"]
    assert payload["evaluator_calls"] == 0
    assert payload["optimizer_created"] is False
    assert payload["checkpoint_created"] is False
