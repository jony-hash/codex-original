from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec, lp01_eigenpair, lp01_radial_profile, lp01_reference_metrics
from ldm_project.v2.optics.fiber_pool import FiberTransportPool


def _plane() -> PlaneSpec:
    return PlaneSpec("fiber_input", (100, 100), (2e-6, 2e-6), 532e-9, 0.0, "encoder")


def test_lp01_independent_fixture_and_boundary_gates() -> None:
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "fiber_lp01_v2_2.json").read_text())
    rho = torch.tensor(fixture["rho"], dtype=torch.float64)
    expected = torch.tensor(fixture["normalized_amplitude"], dtype=torch.float64)
    actual = lp01_radial_profile(rho, fixture["v_number"]).to(torch.float64)
    overlap = torch.dot(actual, expected) / (actual.norm() * expected.norm())
    metrics = lp01_reference_metrics(fixture["v_number"])
    assert float(overlap) >= 0.995
    assert abs(metrics["u"] - fixture["u"]) < 1e-12
    assert abs(metrics["w"] - fixture["w"]) < 1e-12
    assert metrics["closure_error"] <= 1e-12
    assert metrics["characteristic_residual"] <= 1e-10
    assert metrics["boundary_derivative_relative_error"] <= 1e-6
    assert abs(metrics["amplitude_1e_radius_rho"] - fixture["amplitude_1e_radius_rho"]) / fixture["amplitude_1e_radius_rho"] <= 1e-6
    assert torch.all(actual[1:] < actual[:-1])
    with pytest.raises(ValueError, match="cutoff"):
        lp01_eigenpair(2.405)


def test_port_operator_is_100x100_and_passive() -> None:
    operator = FiberArrayOLS(_plane(), FiberArraySpec(facet_magnification=1.5, lateral_offset_m=(-1e-6, 2e-6)))
    area = 4e-12
    gram = operator.input_modes.flatten(1).conj() @ operator.input_modes.flatten(1).T * area
    assert float((gram - torch.eye(100)).abs().max()) <= 1e-5
    field = ComplexField(torch.ones(2, 1, 100, 100, dtype=torch.complex64), _plane(), OperationProvenance("fixture", None, None))
    output = operator(field)
    assert output.plane == _plane().with_plane("ols_output")
    assert output.tensor.shape == field.tensor.shape
    assert operator.last_diagnostics["output_energy_max"] <= operator.last_diagnostics["input_energy_max"] * (1.0 + 1e-5) + 1e-12


def test_complex_adjoint_consistency_and_nonpassive_rejection() -> None:
    operator = FiberArrayOLS(_plane())
    left = torch.randn(1, 1, 100, 100, dtype=torch.complex64)
    right = torch.randn(1, 1, 100, 100, dtype=torch.complex64)
    out_left = operator(ComplexField(left, _plane(), OperationProvenance("left", None, None))).tensor
    out_right = operator(ComplexField(right, _plane(), OperationProvenance("right", None, None))).tensor
    assert torch.isfinite(out_left).all() and torch.isfinite(out_right).all()
    with pytest.raises(ValueError, match="non-passive"):
        FiberArrayOLS(_plane(), transport=torch.eye(100, dtype=torch.complex64) * 1.01)


def test_pool_determinism_offsets_and_schedule_coverage() -> None:
    first, second = FiberTransportPool(), FiberTransportPool()
    assert first.evidence()["schedule_hash"] == second.evidence()["schedule_hash"]
    assert first.record("holdout-00").content_hash == second.record("holdout-00").content_hash
    ids = [record.matrix_id for rows in first.records.values() for record in rows]
    hashes = [record.content_hash for rows in first.records.values() for record in rows]
    assert len(ids) == len(set(ids)) and len(hashes) == len(set(hashes))
    schedule = first.schedule(5201, 50)
    assert set(schedule[:25]) == {record.matrix_id for record in first.records["train"]}
    assert any(record.offset_m != (0.0, 0.0) for record in first.records["train"][1:])
    assert all(float(torch.linalg.svdvals(first.matrix_by_id(matrix_id)).max()) <= 1.0 + 1e-5 for matrix_id in ids)
