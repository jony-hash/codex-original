"""Independent numerical checks consumed by the Gaussian geometry decision."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import GaussianFiberArrayOLS, GaussianFiberGeometry
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import verify_lightgen_2d_gaussian_fiber_geometry_v1 as verification


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def measured_checks():
    geometry = GaussianFiberGeometry()
    op = GaussianFiberArrayOLS(verification.plane(), geometry)
    fine = GaussianFiberArrayOLS(verification.plane(1600), geometry)
    checks, _ = verification.modal_checks(op)
    checks["small_tilt"], _, _ = verification.tilt_checks(op, verification.COARSE_ANGLES, precision="small_FP32")
    checks["refined_tilt"], _, _ = verification.tilt_checks(fine, verification.FINE_ANGLES, precision="refined_FP64")
    checks["tilt_gradient"] = verification.gradient_check(op)
    checks["old_geometry_rejected"] = verification.negative_geometry_check(verification.plane())
    checks["small_runtime_writer"], runtime, frozen = verification.small_writer_check(geometry)
    assert all(torch.equal(tensor, runtime.state_dict()[name]) for name, tensor in frozen.items())
    assert all(parameter.grad is None for parameter in runtime.parameters())
    return checks


def test_complete_producer_measurement_reaches_supported_decision(measured_checks):
    assert verification.decision(measured_checks) == "GEOMETRY_NUMERICALLY_SUPPORTED", measured_checks


def test_phase_loss_in_actual_producer_fails_the_final_decision(measured_checks, monkeypatch):
    op = GaussianFiberArrayOLS(verification.plane(), GaussianFiberGeometry())
    original = op.project_coefficients
    monkeypatch.setattr(op, "project_coefficients", lambda field: original(field).abs().to(torch.complex64))
    faulty_modal, _ = verification.modal_checks(op)
    checks = copy.deepcopy(measured_checks)
    checks.update(faulty_modal)
    assert not checks["two_port_coherence"]["passed"]
    assert verification.decision(checks) == "GEOMETRY_NOT_SUPPORTED"


def test_missing_area_in_actual_producer_is_an_engineering_error(monkeypatch):
    op = GaussianFiberArrayOLS(verification.plane(), GaussianFiberGeometry())
    original = op.project_coefficients
    monkeypatch.setattr(op, "project_coefficients", lambda field: original(field) / math.prod(op.input_plane.pixel_pitch_m))
    with pytest.raises(ValueError, match="input/output power"):
        verification.modal_checks(op)


def test_reemission_is_full_gaussian_without_port_cropping_or_renormalization():
    p = verification.plane()
    op = GaussianFiberArrayOLS(p, GaussianFiberGeometry())
    coefficients = torch.zeros((1, 100), dtype=torch.complex128)
    coefficients[0, 44] = 1
    emitted = op.reemit_coefficients(coefficients).numpy()[0]
    expected = verification.reference_fields(p, op.geometry)[44]
    assert np.allclose(emitted, expected, rtol=1e-12, atol=1e-15)
    assert emitted[44, 54].real > 0  # Beyond this port's former rectangular cell.


def test_bad_geometry_is_rejected_by_complete_operator_bound():
    bad = GaussianFiberGeometry(port_pitch_m=160e-6 / 9, waist_m=7.111e-6)
    with pytest.raises(ValueError, match="non-passive"):
        GaussianFiberArrayOLS(verification.plane(), bad)


def test_refinement_uses_same_fov_and_port_coordinates():
    coarse = GaussianFiberArrayOLS(verification.plane(), GaussianFiberGeometry())
    fine = GaussianFiberArrayOLS(verification.plane(1600), GaussianFiberGeometry())
    assert torch.equal(coarse.centers_m, fine.centers_m)
    assert coarse.input_plane.fov_m == fine.input_plane.fov_m
    assert fine.input_plane.pixel_pitch_m == (0.125e-6, 0.125e-6)


def test_public_boundaries_reject_invalid_geometry_and_input():
    with pytest.raises(ValueError, match="finite and positive"):
        GaussianFiberGeometry(waist_m=float("nan"))
    op = GaussianFiberArrayOLS(verification.plane(), GaussianFiberGeometry())
    bad = torch.ones((1, 100, 100), dtype=torch.complex64)
    bad[0, 0, 0] = complex(float("nan"), 0)
    field = ComplexField(bad, op.input_plane, OperationProvenance("test_input", None, None))
    with pytest.raises(ValueError, match="finite"):
        op.project_coefficients(field)
    incompatible = op.input_plane.with_plane("different_physical_plane")
    field = ComplexField(torch.ones_like(bad), incompatible, OperationProvenance("test_input", None, None))
    with pytest.raises(ValueError, match="physical plane"):
        op.project_coefficients(field)


def test_decision_refuses_incomplete_measurement(measured_checks):
    checks = dict(measured_checks)
    del checks["refined_tilt"]
    with pytest.raises(ValueError, match="complete measurement"):
        verification.decision(checks)
