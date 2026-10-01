from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.lightgen import materialize_lightgen_2d_frozen_dev_ols_cmos_v1 as materializer


def test_execution_plan_is_read_only_and_fail_closed_for_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "run"
    target.parent.mkdir()
    plan = materializer.build_execution_plan(repo_root=tmp_path, output_dir=target)
    assert plan["outer_gate"] == materializer.OUTER_GATE
    assert plan["optimizer"] is False
    assert plan["backward"] is False
    assert plan["probe_fitting"] is False
    target.mkdir()
    with pytest.raises(materializer.MaterializationError, match="already exists"):
        materializer.build_execution_plan(repo_root=tmp_path, output_dir=target)


def test_materialized_tensor_validation_rejects_zero_or_constant_cmos() -> None:
    z = torch.ones((materializer.DEV_COUNT, 100), dtype=torch.complex64)
    indices = torch.arange(materializer.DEV_COUNT, dtype=torch.long)
    labels = torch.arange(materializer.DEV_COUNT, dtype=torch.long).remainder(10)
    raw = torch.ones((materializer.DEV_COUNT, 1, 200, 200), dtype=torch.float32)
    with pytest.raises(materializer.MaterializationError, match="zero images|constant images"):
        materializer._validate_materialized_tensors(z, raw, indices, labels)


def test_materialized_tensor_validation_accepts_finite_nonconstant_cmos() -> None:
    z = torch.ones((materializer.DEV_COUNT, 100), dtype=torch.complex64)
    indices = torch.arange(materializer.DEV_COUNT, dtype=torch.long)
    labels = torch.arange(materializer.DEV_COUNT, dtype=torch.long).remainder(10)
    raw = torch.zeros((materializer.DEV_COUNT, 1, 200, 200), dtype=torch.float32)
    raw[:, :, 0, 0] = 1.0
    diagnostics = materializer._validate_materialized_tensors(z, raw, indices, labels)
    assert diagnostics["cmos"]["finite_fraction"] == 1.0
    assert diagnostics["cmos"]["zero_image_count"] == 0
    assert diagnostics["cmos"]["constant_image_count"] == 0
