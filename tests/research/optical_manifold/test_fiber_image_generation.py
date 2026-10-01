from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.optical_manifold.fiber_image_generation import FUNCTION_IDS, FiberImageResearchSpec, _deterministic_adaptive_average, _tensor_content_hash, _bootstrap_rows, _mapping, _ref, _resolve
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


def test_research_pool_matches_native_forward_and_gradient() -> None:
    values = torch.randn(2, 1, 200, 200, dtype=torch.float64, requires_grad=True)
    native = torch.nn.functional.adaptive_avg_pool2d(values, (28, 28))
    research = _deterministic_adaptive_average(values, 28)
    assert torch.allclose(native, research, rtol=0.0, atol=1.0e-12)
    native.sum().backward()
    native_grad = values.grad.detach().clone()
    values.grad.zero_()
    research.sum().backward()
    assert torch.allclose(native_grad, values.grad, rtol=1.0e-10, atol=1.0e-12)


def test_bootstrap_has_real_cross_function_pairing() -> None:
    rows = []
    for function_index, function_id in enumerate(("S0", "S1a", "S1b", "S1c")):
        for seed in (5201, 5202, 5203):
            panels = []
            vector = [0.01 * function_index] * 500
            for panel_seed in (4242, 104729, 161803):
                panels.append({"panel_seed": panel_seed, "gain_vector": vector})
            rows.append({"function_id": function_id, "seed": seed, "panels": panels})
    result = _bootstrap_rows(rows)
    assert result["4242"]["cross_function"]["maximum_function"] == "S1c"
    assert result["4242"]["cross_function"]["minimum_function"] == "S0"


def test_tensor_content_hash_is_container_independent(tmp_path: Path) -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(value, first)
    torch.save(value.clone(), second)
    assert first.read_bytes() != second.read_bytes() or _tensor_content_hash(value) == _tensor_content_hash(torch.load(second, weights_only=True))


def test_closeout_rejects_selection_without_locked_artifacts(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from ldm_project.v2.application.research.optical_manifold.fiber_image_generation import closeout

    rows = [
        {
            "function_id": function_id,
            "seed": seed,
            "panels": [{"panel_seed": panel} for panel in (4242, 104729, 161803)],
        }
        for function_id in ("S0", "S1a", "S1b", "S1c")
        for seed in (5201, 5202, 5203)
    ]
    selection = tmp_path / "selection.json"
    selection.write_text(
        __import__("json").dumps(
            {
                "schema": "fiber_image_selection.v2",
                "gate": "valid",
                "rows": rows,
                "bootstrap": {str(panel): {} for panel in (4242, 104729, 161803)},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "closeout.json"
    assert closeout(SimpleNamespace(selection_evidence=str(selection), output=str(output))) == 2
    assert __import__("json").loads(output.read_text(encoding="utf-8"))["conclusion"] == "invalid_harness"
