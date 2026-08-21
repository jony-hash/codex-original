from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.fiber_image_generation import (
    FUNCTION_IDS,
    FiberImageResearchSpec,
    _mapping,
    _ref,
    _resolve,
)
from ldm_project.v2.optics.fiber_pool import FiberTransportPool


def test_image_function_registry_is_fixed_and_disjoint() -> None:
    mapping = _mapping(FiberTransportPool())
    assert tuple(mapping) == FUNCTION_IDS
    assert mapping["S1a"] != mapping["S1b"] != mapping["S1c"]
    assert mapping["N0"].startswith("full_phase-")
    assert FiberImageResearchSpec().bootstrap_replicates == 10000


def test_image_artifact_reference_is_relative_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "tensor.pt"
    torch.save(torch.ones(2, 3), path)
    reference = _ref(path, tmp_path)
    assert reference["relative_path"] == "tensor.pt"
    assert _resolve(reference, tmp_path) == path
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash-mismatched"):
        _resolve(reference, tmp_path)


def test_image_artifact_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _resolve({"relative_path": "/tmp/escape", "sha256": "x"}, tmp_path)
    with pytest.raises(ValueError):
        _resolve({"relative_path": "../escape", "sha256": "x"}, tmp_path)
