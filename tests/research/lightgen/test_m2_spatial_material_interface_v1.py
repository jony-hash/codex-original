"""Independent numerical controls for the production M2 producer and consumer."""
import math

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.material_nonlinearity import (
    MaterialNonlinearity, MaterialParameters,
)
from ldm_project.v2.application.research.lightgen.spatial_material_interface import (
    SpatialMaterialInterface, SyntheticResponse, WeightedModeProjector, diagnostic_decision,
)


def test_weighted_projection_nonorthogonal_independent_qr_and_negative_control():
    v = torch.tensor([[1, 0], [1, 1j], [0, 2], [1j, 1]], dtype=torch.complex128)
    field = torch.tensor([1+2j, 2, 4j, -1], dtype=torch.complex128).reshape(1, 1, 2, 2)
    p = WeightedModeProjector(v, 0.3)
    actual, residual = p.project(field)
    q, _ = np.linalg.qr(v.numpy()*math.sqrt(0.3))
    expected = q @ q.conj().T @ (field.numpy().ravel()*math.sqrt(0.3))/math.sqrt(0.3)
    np.testing.assert_allclose(actual.numpy().ravel(), expected, rtol=1e-12, atol=1e-12)
    faulty = v @ v.mH @ field.flatten() * 0.3
    assert not np.allclose(faulty.numpy(), expected, rtol=1e-3, atol=1e-3)
    assert torch.max((v.mH @ residual.flatten()*0.3).abs()) < 1e-12
    assert p.metrics(field)["closure_relative"].max() < 1e-12
    in_span = (v @ torch.tensor([1+1j, 3], dtype=torch.complex128)).reshape(1, 1, 2, 2)
    assert p.metrics(in_span)["eta_perp"].max() < 1e-25


def test_m1_exact_old_formula_and_m0_m2_limits():
    z = torch.tensor([[0.3+0.2j, 0.1j]], dtype=torch.complex128)
    reemit = lambda c: torch.stack((c[:, 0], c[:, 1], c[:, 0], c[:, 1]), 1).reshape(-1, 2, 2)
    field = reemit(z).unsqueeze(1)
    interface = SpatialMaterialInterface(7.0, 0.4, reemit)
    response = SyntheticResponse("saturation", 0.4)
    actual = interface.apply("M1", field, z, response)
    peak = 2*z.abs().square()/(math.pi*0.2**2)/1e4
    old = MaterialNonlinearity().forward(z, peak, MaterialParameters(
        "saturable_absorption", 0.6, 0.4, 7/1e4))
    assert torch.equal(actual, reemit(old).unsqueeze(1))
    for model in ("M0", "M2"):
        assert torch.equal(interface.apply(model, field, z, SyntheticResponse()), field)
        linear = interface.apply(model, field, z, SyntheticResponse("linear", proxy_linear_transmission=.8))
        torch.testing.assert_close(linear.abs().square().sum(), .8*field.abs().square().sum())
    for family in ("saturation", "suppression"):
        out = interface.apply("M2", field, z, SyntheticResponse(family, .4))
        assert (out.abs() <= field.abs()+1e-14).all()


def test_m2_derivative_and_production_decision():
    # Deliberately nonorthogonal and spatially overlapping: removes special-case orthogonality.
    v = torch.tensor([[1, 0], [1, 1], [.3, 1], [.2, -.4]], dtype=torch.complex128)
    z = torch.tensor([[1+1j, .4-.2j]], dtype=torch.complex128)
    reemit = lambda c: (c @ v.T).reshape(-1, 2, 2)
    interface = SpatialMaterialInterface(1, 2, reemit)
    field = reemit(z).unsqueeze(1).requires_grad_()
    response = SyntheticResponse("saturation", .8)
    assert torch.autograd.gradcheck(lambda f: interface.apply("M2", f, z, response), (field,))
    out = interface.apply("M2", field, z, response)
    projector = WeightedModeProjector(v, .2)
    eta = projector.metrics(out)["eta_perp"].item()
    parallel, _ = projector.project(out)
    delta = ((out.abs().square()-parallel.abs().square()).norm()/out.abs().square().norm()).item()
    field_diff = ((out-field).norm()/out.norm()).item()
    assert eta > 1e-8
    assert diagnostic_decision(eta, field_diff, delta, True)[0] == "C"
    assert diagnostic_decision(eta, field_diff, delta, False)[0] == "D"
    assert diagnostic_decision(eta, field_diff, 0, True)[0] == "B"
    identity = interface.apply("M2", field, z, SyntheticResponse())
    assert diagnostic_decision(projector.metrics(identity)["eta_perp"].item(), 0, 0, True)[0] == "A"


def test_reject_invalid_material_boundary():
    with pytest.raises(ValueError):
        SyntheticResponse("linear", proxy_linear_transmission=1.1)
    with pytest.raises(ValueError):
        WeightedModeProjector(torch.eye(2, dtype=torch.complex128), -1)


def test_saved_evaluator_batch_contract_regression():
    from scripts.research.lightgen.validate_lightgen_m2_spatial_material_interface_v1 import frozen_logits, frozen_reemit
    calls = []

    def shape_sensitive_evaluator(views):
        calls.append(len(views))
        return views + len(views)

    views = torch.zeros(40, 1)
    actual = frozen_logits(shape_sensitive_evaluator, views, 10)
    assert calls == [10, 10, 10, 10]
    assert torch.equal(actual, torch.full_like(views, 10))
    # The original faulty whole-cohort call fails the same expected result.
    assert not torch.equal(shape_sensitive_evaluator(views), actual)
    class Operator:
        reemit_coefficients = staticmethod(shape_sensitive_evaluator)

    calls.clear()
    assert torch.equal(frozen_reemit(Operator(), views, 10), actual)
    assert calls == [10, 10, 10, 10]


def test_independent_reference_promotes_before_pixel_area_weighting():
    from scripts.research.lightgen.validate_lightgen_m2_spatial_material_interface_v1 import independent_qr_projection
    v = torch.tensor([[1, 0], [1, 1j], [0, 2], [1j, 1]], dtype=torch.complex128)*1e5
    field = torch.tensor([1.1234567+2j, 2.314159, 4.271828j, -1.414214],
                         dtype=torch.complex64).reshape(1, 1, 2, 2)
    expected, _ = WeightedModeProjector(v, 4e-12).project(field)
    qr = independent_qr_projection(v, field, 4e-12)
    np.testing.assert_allclose(qr, expected.numpy().ravel(), rtol=1e-12, atol=1e-12)
    q, _ = np.linalg.qr(v.numpy()*math.sqrt(4e-12))
    faulty_rhs = (field.numpy().ravel()*math.sqrt(4e-12)).astype(np.complex64)
    faulty = q @ (q.conj().T @ faulty_rhs)/math.sqrt(4e-12)
    assert np.linalg.norm(faulty-qr)/np.linalg.norm(qr) > 1e-10
