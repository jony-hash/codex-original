from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import _cli_research_optical_manifold as router
from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.current import frozen_mpcdar_rank16_float32_readout as runtime
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification as s10


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification_review_v1.json"
ARTIFACT = ROOT / s10.S9_ARTIFACT_DIR


def test_config_review_and_parent_identity_are_exactly_locked() -> None:
    spec = s10._load_spec(CONFIG)
    s10._assert_contract(spec)
    assert s10._load_review(REVIEW, spec) == REVIEW.resolve()
    parent = s10._verify_parent_evidence()
    assert parent["artifact"].manifest["rank"] == 16
    assert parent["artifact"].manifest["dtype"] == "torch.float32"


def test_registered_consumer_and_storage_contract() -> None:
    assert s10.RANK == 16
    assert s10.BATCH_SIZES == (1, 8, 32)
    assert s10.WARMUP_COUNT == 50
    assert s10.MEASURED_REPETITIONS == 200
    assert s10.SERIALIZED_BYTES == 2739448
    assert s10.RAW_FLOAT32_TENSOR_BYTES == 2737408
    assert s10._runtime_source_contract()["formula_tokens_present"]
    assert s10._runtime_source_contract()["pure_float32_frozen_runtime"]


def test_deterministic_dense_fixture_is_fixed_and_finite() -> None:
    first = s10._make_fixture()
    second = s10._make_fixture()
    assert first.shape == (32, 256)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.count_nonzero(first) == first.numel()


def test_quantile_method_and_statistics_are_not_best_only() -> None:
    summary = s10._summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["sample_count"] == 5
    assert summary["median_ms"] == 3.0
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["p99_ms"] == pytest.approx(4.96)
    assert summary["quantile_method"] == "linear: index=(n-1)*q"


def test_cli_contract_is_cpu_only_and_separates_validate_run() -> None:
    validate = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mpcdar-rank16-runtime-performance-qualification-validate",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--artifact-dir",
            "artifact",
            "--output",
            "validation.json",
        ]
    )
    run = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mpcdar-rank16-runtime-performance-qualification-run",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--validation-evidence",
            "validation.json",
            "--artifact-dir",
            "artifact",
            "--output",
            "audit.json",
        ]
    )
    assert validate.research_action.endswith("-validate")
    assert run.research_action.endswith("-run")
    assert run.device == "cpu"


def test_router_registers_only_explicit_s10_actions() -> None:
    source = inspect.getsource(router.dispatch)
    assert "scaled-chen-mpcdar-rank16-runtime-performance-qualification-validate" in source
    assert "scaled-chen-mpcdar-rank16-runtime-performance-qualification-run" in source
    assert "optical_manifold_smf_bogt_mpcdar_rank16_runtime_performance_qualification" in source


def test_optional_paths_and_readiness_boundary_are_locked() -> None:
    contract = s10._config_payload()
    assert contract["optional"]["gpu"] == "not_evaluated"
    assert contract["optional"]["full_w_runtime"] == "not_evaluated"
    assert contract["memory"]["process_memory_delta"] == "not_recorded"
    assert contract["authorization"]["training_authorized"] is False
    assert contract["authorization"]["formal_promotion"] is False


def test_environment_records_expected_and_observed_windows_versions() -> None:
    environment = s10._environment()
    assert environment["expected_windows_version"] == "10.0.26200.0"
    assert environment["observed_windows_version"] == environment["os_version"]
    assert isinstance(environment["windows_version_match"], bool)
    assert isinstance(environment["registered_runtime_match"], bool)


def test_frozen_s9_runtime_has_no_research_or_training_dependency() -> None:
    source = inspect.getsource(runtime)
    assert "application.research" not in source
    assert "torch.optim" not in source
    assert ".to(torch.float64)" not in source
    assert runtime.RANK == 16


def test_result_schema_constants_are_stable() -> None:
    assert s10.CLASSIFICATION == "frozen_rank16_float32_runtime_performance_characterized"
    assert s10.NEXT_ACTION_RESULT_REVIEW == "manual_review_MPCDAR_R16_RUNTIME_PQ_result"
    assert json.loads(REVIEW.read_text(encoding="utf-8"))["classification"] == s10.CLASSIFICATION
