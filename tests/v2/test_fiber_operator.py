from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec, lp01_radial_profile
from ldm_project.v2.optics.fiber_pool import FiberTransportPool


def test_lp01_offline_fixture_overlap_and_normalization() -> None:
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "fiber_lp01_v2_2.json").read_text())
    rho = torch.tensor(fixture["rho"])
    expected = torch.tensor(fixture["normalized_amplitude"])
    actual = lp01_radial_profile(rho, fixture["v_number"])
    overlap = torch.dot(actual, expected) / (actual.norm() * expected.norm())
    assert float(overlap) >= 0.995
    assert float((actual.square().sum() - expected.square().sum()).abs()) <= 1e-4
    with pytest.raises(ValueError, match="V=2.2"):
        lp01_radial_profile(rho, 2.1)


def test_fiber_operator_plane_mapping_and_identity_transport() -> None:
    plane = PlaneSpec("fiber_input", (10, 10), (2e-6, 2e-6), 532e-9, 0.0, "ports")
    spec = FiberArraySpec(facet_magnification=1.5, lateral_offset_m=(-1e-6, 2e-6))
    operator = FiberArrayOLS(plane, spec)
    field = ComplexField(torch.ones(10, 10, dtype=torch.complex64), plane, OperationProvenance("fixture", None, None))
    output = operator(field)
    assert output.plane.pixel_pitch_m == (3e-6, 3e-6)
    assert output.plane.origin_m == (-1e-6, 2e-6)
    assert torch.equal(output.tensor, field.tensor)


def test_pool_is_deterministic_disjoint_and_passive() -> None:
    first, second = FiberTransportPool(), FiberTransportPool()
    assert first.evidence()["schedule_hash"] == second.evidence()["schedule_hash"]
    assert first.records["holdout"][0].content_hash == second.records["holdout"][0].content_hash
    ids = [record.matrix_id for records in first.records.values() for record in records]
    assert len(ids) == len(set(ids))
    assert all(float(torch.linalg.svdvals(first.matrix(pool, 0)).max()) <= 1.0 + 1e-5 for pool in first.records)


def test_nonpassive_transport_rejected() -> None:
    plane = PlaneSpec("fiber_input", (10, 10), (2e-6, 2e-6), 532e-9, 0.0, "ports")
    with pytest.raises(ValueError, match="non-passive"):
        FiberArrayOLS(plane, transport=torch.eye(100, dtype=torch.complex64) * 1.01)
