from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application import _cli_research_optical_manifold as router
from ldm_project.v2.application.cli_schema import _parser
from ldm_project.v2.application.current import frozen_mpcdar_rank16_readout as runtime
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification as s8,
)
from ldm_project.v2.evidence import atomic_torch_save, atomic_torch_save_new, atomic_write_new_json


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_v1.yaml"
REVIEW = ROOT / "configs/research/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_review_v1.json"


def _artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    payload = {
        name: torch.zeros(shape, dtype=torch.float64)
        for name, shape in runtime.TENSOR_SHAPES.items()
    }
    payload["mu_fit"][0] = 1.0
    payload["c_mean"][0] = 2.0
    payload["T16"][0, 0] = 3.0
    payload["V16"][0, 0] = 4.0
    payload_path = artifact_dir / "payload.pt"
    atomic_torch_save_new(payload_path, payload, artifact_kind="test payload")
    manifest = s8._manifest(
        payload,
        payload_path,
        {"commit": "0" * 40, "branch": s8.BRANCH, "clean": True},
    )
    atomic_write_new_json(artifact_dir / "manifest.json", manifest, artifact_kind="test manifest")
    return artifact_dir


def test_config_review_and_contract_are_exactly_locked() -> None:
    spec = s8._load_spec(CONFIG)
    s8._assert_contract(spec)
    assert s8._load_review(REVIEW, spec) == REVIEW
    assert s8.RANK == 16
    assert s8.EVALUATION["tested_ranks"] == [16]
    assert s8.EVALUATION["search"] is False


def test_payload_scalar_count_and_raw_bytes_are_registered() -> None:
    assert s8.SCALAR_COUNT_TOTAL == 684352
    assert s8.RAW_FLOAT64_TENSOR_BYTES == 5474816
    assert sum(torch.Size(shape).numel() for shape in runtime.TENSOR_SHAPES.values()) == 684352
    assert sum(torch.Size(shape).numel() * 8 for shape in runtime.TENSOR_SHAPES.values()) == 5474816


def test_runtime_loads_payload_and_preserves_registered_formula(tmp_path: Path) -> None:
    artifact = runtime.load_frozen_mpcdar_rank16_readout(_artifact(tmp_path) / "manifest.json")
    features = torch.zeros((2, 3, 256), dtype=torch.float32)
    features[..., 0] = 2.0
    prediction = artifact.predict(features)
    assert prediction.shape == (2, 3, 40000)
    assert prediction.dtype == torch.float64
    assert torch.all(prediction[..., 0] == 14.0)
    assert torch.count_nonzero(prediction[..., 1:]) == 0


def test_runtime_rejects_payload_tampering(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    payload_path = artifact / "payload.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload["c_mean"][0] = 99.0
    atomic_torch_save(payload_path, payload)
    with pytest.raises(runtime.FrozenMPCDARRank16ArtifactError, match="hash"):
        runtime.load_frozen_mpcdar_rank16_readout(artifact / "manifest.json")


def test_new_torch_writer_is_write_once_and_old_writer_remains_replaceable(tmp_path: Path) -> None:
    immutable = tmp_path / "immutable.pt"
    atomic_torch_save_new(immutable, {"value": torch.tensor(1)}, artifact_kind="test")
    with pytest.raises(FileExistsError):
        atomic_torch_save_new(immutable, {"value": torch.tensor(2)}, artifact_kind="test")
    replaceable = tmp_path / "replaceable.pt"
    atomic_torch_save(replaceable, {"value": torch.tensor(1)})
    atomic_torch_save(replaceable, {"value": torch.tensor(2)})
    assert torch.load(replaceable, map_location="cpu", weights_only=True)["value"].item() == 2


def test_runtime_boundary_has_no_research_or_reconstruction_dependencies() -> None:
    source = inspect.getsource(runtime)
    assert "application.research" not in source
    assert "from ..research" not in source
    assert "torch.linalg" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "safe_torch_load" in source


def test_runtime_contract_constants_and_manifest_identity() -> None:
    assert runtime.PAYLOAD_KEYS == frozenset({"mu_fit", "c_mean", "T16", "V16"})
    assert runtime.TENSOR_SHAPES["T16"] == (256, 16)
    assert runtime.TENSOR_SHAPES["V16"] == (40000, 16)
    assert runtime.PARENT_ID == s8.PARENT_ID
    assert runtime.SCALAR_COUNT_TOTAL == s8.SCALAR_COUNT_TOTAL


def test_cli_separates_build_and_qualify_actions() -> None:
    build = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mpcdar-rank16-integration-qualification-build",
            "--config",
            "config.yaml",
            "--review-contract",
            "review.json",
            "--artifact-dir",
            "artifact",
        ]
    )
    qualify = _parser().parse_args(
        [
            "research",
            "optical-manifold",
            "scaled-chen-mpcdar-rank16-integration-qualification-qualify",
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
    assert build.dataset_root == "data"
    assert qualify.research_action.endswith("-qualify")
    assert qualify.device == "cuda"


def test_router_contains_only_explicit_s8_actions() -> None:
    source = inspect.getsource(router.dispatch)
    assert "scaled-chen-mpcdar-rank16-integration-qualification-build" in source
    assert "scaled-chen-mpcdar-rank16-integration-qualification-qualify" in source
    assert "optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification" in source


def test_manifest_has_exact_four_payload_entries(tmp_path: Path) -> None:
    manifest = json.loads((_artifact(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["tensors"]) == {"mu_fit", "c_mean", "T16", "V16"}
    assert manifest["tensor_count"] == 4
    assert manifest["created_git"]["clean"] is True
