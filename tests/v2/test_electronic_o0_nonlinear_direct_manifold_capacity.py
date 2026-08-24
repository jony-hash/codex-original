from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import electronic_o0_nonlinear_direct_manifold_capacity as ndmc


def _synthetic_canary() -> ndmc.ManifoldCanary:
    generator = torch.Generator(device="cpu").manual_seed(17)
    images = torch.rand((400, 1, 28, 28), generator=generator)
    labels = torch.arange(10).repeat_interleave(40)
    return ndmc._fit_canary(images, labels)


def _valid_result() -> dict:
    row = {
        "count": 2500,
        "primary": {"accuracy": 0.96, "minimum_recall": 0.92},
        "secondary": {"accuracy": 0.96, "minimum_recall": 0.92},
        "primary_counts": {"correct": 2400, "total": 2500, "recall_correct": [230] * 10, "recall_total": [250] * 10},
        "secondary_counts": {"correct": 2400, "total": 2500, "recall_correct": [230] * 10, "recall_total": [250] * 10},
        "agreement": 0.96,
        "agreement_counts": {"correct": 2400, "total": 2500, "recall_correct": [0] * 10, "recall_total": [0] * 10},
        "camera_code_fraction": 0.30,
        "saturation": 0.01,
        "coverage": {"primary": 1.0, "secondary": 1.0},
        "diversity_ratio": {"primary": [1.0] * 10, "secondary": [1.0] * 10},
        "label_shuffled_primary_counts": {"correct": 100, "total": 2500, "recall_correct": [0] * 10, "recall_total": [0] * 10},
        "zero_code": {"primary": {"correct": 10, "total": 10}, "secondary": {"correct": 10, "total": 10}},
        "memorization": {"passed": True, "exact_duplicate_count": 0},
        "semantic_pass": True,
        "safety_pass": True,
        "agreement_pass": True,
    }
    return {
        "banks": {"271828": dict(row), "314159": dict(row)},
        "controls": {"zero_code": {}, "label_shift": {}},
        "historical_replay": {"matches": True},
        "chunk_exact": True,
        "optimizer_created": False,
        "checkpoint_created": False,
        "parameter_hash_before": {"primary": "p", "secondary": "s"},
        "parameter_hash_after": {"primary": "p", "secondary": "s"},
        "observability": {"rank_pass": True, "curvature_pass": True},
        "cross_panel_stability": True,
    }


def test_registered_config_and_cli_contract() -> None:
    config = ndmc._config("configs/research/electronic_o0_nonlinear_direct_manifold_capacity_v1.yaml")
    assert config["name"] == ndmc.CONFIG_NAME
    assert config["gaussian_bank_seeds"] == [271828, 314159]
    assert config["synthetic_diagnostic_only"] is True


def test_canary_is_finite_deterministic_and_chunk_probe_exact() -> None:
    canary = _synthetic_canary()
    labels = torch.arange(10).repeat_interleave(3)
    codes = torch.randn((30, 8), generator=torch.Generator().manual_seed(31))
    first = canary.sample(labels, codes, chunk_size=64)
    second = canary.sample(labels, codes, chunk_size=257)
    assert first.shape == (30, 1, 28, 28)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.equal(first, canary.sample(labels, codes, chunk_size=64))


def test_canary_rejects_wrong_code_shape_and_chunk() -> None:
    canary = _synthetic_canary()
    labels = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"labels \[N\] and codes \[N,8\]"):
        canary.sample(labels, torch.zeros(2, 7))
    with pytest.raises(ValueError, match="64/257"):
        canary.sample(labels, torch.zeros(2, 8), chunk_size=128)


def test_classification_is_fail_closed_and_priority_ordered() -> None:
    admitted = _valid_result()
    assert ndmc._classify(admitted) == (
        "nonlinear_direct_manifold_capacity_admitted",
        "register_direct_generator_architecture_preflight",
    )

    invalid = _valid_result()
    invalid["historical_replay"]["matches"] = False
    assert ndmc._classify(invalid)[0] == "invalid_harness"

    safety = _valid_result()
    safety["banks"]["271828"]["safety_pass"] = False
    assert ndmc._classify(safety)[0] == "nonlinear_manifold_safety_failed"

    consensus = _valid_result()
    consensus["banks"]["271828"]["agreement_pass"] = False
    consensus["banks"]["271828"]["agreement"] = 0.90
    consensus["banks"]["271828"]["agreement_counts"]["correct"] = 2250
    assert ndmc._classify(consensus)[0] == "generated_evaluator_consensus_failed"

    semantic = _valid_result()
    semantic["banks"]["271828"]["semantic_pass"] = False
    semantic["banks"]["271828"]["primary"]["minimum_recall"] = 0.80
    assert ndmc._classify(semantic)[0] == "nonlinear_direct_semantic_capacity_failed"

    observability = _valid_result()
    observability["observability"]["rank_pass"] = False
    assert ndmc._classify(observability)[0] == "nonlinear_manifold_capacity_not_established"


def test_manifest_only_dispatch_rejects_legacy_invocation() -> None:
    args = Namespace(research_action="audit", run_manifest=None)
    with pytest.raises(ValueError, match="manifest-only"):
        ndmc.dispatch(args)


def test_consumer_rejects_wrong_stage_before_runtime(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "electronic_o0_nonlinear_direct_manifold_capacity_preflight.v1",
                "status": "valid",
                "invocation_mode": "manifest",
                "legacy_diagnostic_only": False,
                "permitted_consumers": ["audit"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid H-NDMC report evidence"):
        ndmc._check_consumer(str(evidence), "report")
