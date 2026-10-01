from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import _cli_research_optical_manifold as router
from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.current import frozen_mpcdar_rank16_float32_readout as candidate
from ldm_project.v2.application.current import frozen_mpcdar_rank16_readout as reference
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification as s9
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_v1.yaml"
REVIEW = ROOT / "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_review_v1.json"


def _source_payload() -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(shape, dtype=torch.float64)
        for name, shape in reference.TENSOR_SHAPES.items()
    }


def _candidate_artifact(tmp_path: Path) -> Path:
    source = _source_payload()
    source["mu_fit"][0] = 1.0
    source["c_mean"][0] = 2.0
    source["T16"][0, 0] = 3.0
    source["V16"][0, 0] = 4.0
    payload = s9._build_payload(source)
    artifact_dir = tmp_path / "artifact"
    payload_path = artifact_dir / "payload.pt"
    atomic_torch_save_new(payload_path, payload, artifact_kind="test S9 payload")
    manifest = s9._manifest(
        payload,
        payload_path,
        {"commit": "0" * 40, "branch": s9.BRANCH, "clean": True},
    )
    atomic_write_new_json(artifact_dir / "manifest.json", manifest, artifact_kind="test S9 manifest")
    return artifact_dir


def test_config_review_and_contract_are_exactly_locked() -> None:
    spec = s9._load_spec(CONFIG)
    s9._assert_contract(spec)
    assert s9._load_review(REVIEW, spec) == REVIEW
    assert s9.RANK == 16
    assert s9.EVALUATION["candidate_family"] == [16]
    assert s9.EVALUATION["search"] is False


def test_registered_storage_contract() -> None:
    assert s9.SCALAR_COUNT_TOTAL == 684352
    assert s9.RAW_FLOAT64_TENSOR_BYTES == 5474816
    assert s9.RAW_FLOAT32_TENSOR_BYTES == 2737408
    assert sum(torch.Size(shape).numel() for shape in candidate.TENSOR_SHAPES.values()) == 684352
    assert sum(torch.Size(shape).numel() * 4 for shape in candidate.TENSOR_SHAPES.values()) == 2737408


def test_exact_cast_and_fresh_runtime_formula(tmp_path: Path) -> None:
    artifact_dir = _candidate_artifact(tmp_path)
    loaded = candidate.load_frozen_mpcdar_rank16_float32_readout(artifact_dir / "manifest.json")
    source = _source_payload()
    source["mu_fit"][0] = 1.0
    source["c_mean"][0] = 2.0
    source["T16"][0, 0] = 3.0
    source["V16"][0, 0] = 4.0
    identity = s9._exact_cast_identity(source, {name: getattr(loaded, name) for name in candidate.PAYLOAD_KEYS})
    assert identity["passed"]
    features = torch.zeros((2, 3, 256), dtype=torch.float32)
    features[..., 0] = 2.0
    prediction = loaded.predict(features)
    assert prediction.shape == (2, 3, 40000)
    assert prediction.dtype == torch.float32
    assert torch.all(prediction[..., 0] == 14.0)
    assert torch.count_nonzero(prediction[..., 1:]) == 0


def test_runtime_requires_float32_input(tmp_path: Path) -> None:
    loaded = candidate.load_frozen_mpcdar_rank16_float32_readout(_candidate_artifact(tmp_path) / "manifest.json")
    with pytest.raises(ValueError, match="torch.float32"):
        loaded.predict(torch.zeros((1, 256), dtype=torch.float64))


def test_runtime_is_narrow_and_has_no_explicit_upcast_or_research_dependency() -> None:
    source = inspect.getsource(candidate)
    assert ".to(torch.float64)" not in source
    assert ".double(" not in source
    assert "torch.linalg" not in source
    assert "torch.optim" not in source
    assert "application.research" not in source
    assert "safe_torch_load" in source
    assert s9._source_contract()["pure_float32_runtime_contract"]


def test_manifest_has_parent_identity_and_four_float32_entries(tmp_path: Path) -> None:
    manifest = json.loads((_candidate_artifact(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_artifact"]["manifest_sha256"] == s9.PARENT_ARTIFACT_MANIFEST_SHA256
    assert manifest["parent_artifact"]["payload_sha256"] == s9.PARENT_ARTIFACT_PAYLOAD_SHA256
    assert set(manifest["tensors"]) == {"mu_fit", "c_mean", "T16", "V16"}
    assert all(entry["dtype"] == "torch.float32" for entry in manifest["tensors"].values())
    assert manifest["tensor_count"] == 4
    assert manifest["scalar_count_total"] == 684352
    assert manifest["raw_float32_tensor_bytes"] == 2737408


def test_cli_separates_build_and_qualify_actions() -> None:
    build = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mpcdar-rank16-float32-numerical-qualification-build",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--parent-manifest",
            "parent.json",
            "--artifact-dir",
            "artifact",
        ]
    )
    qualify = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mpcdar-rank16-float32-numerical-qualification-qualify",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--artifact-dir",
            "artifact",
            "--output",
            "audit.json",
        ]
    )
    assert build.research_action.endswith("-build")
    assert qualify.research_action.endswith("-qualify")
    assert qualify.device == "cuda"


def test_router_contains_only_explicit_s9_actions() -> None:
    source = inspect.getsource(router.dispatch)
    assert "scaled-chen-mpcdar-rank16-float32-numerical-qualification-build" in source
    assert "scaled-chen-mpcdar-rank16-float32-numerical-qualification-qualify" in source
    assert "optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification" in source


def test_s8_reference_runtime_module_is_not_modified_by_s9_contract() -> None:
    source = inspect.getsource(reference)
    assert ".to(torch.float64)" in source
    assert reference.RANK == candidate.RANK == 16
    assert reference.PAYLOAD_KEYS == candidate.PAYLOAD_KEYS
