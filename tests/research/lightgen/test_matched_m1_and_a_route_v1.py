"""Independent matched-M1 formula, boundary, and derivative tests."""
import math

import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.matched_m1_diagnostic import matched_mode_response
from ldm_project.v2.application.research.lightgen.spatial_material_interface import (
    SpatialMaterialInterface, SyntheticResponse, WeightedModeProjector,
)


@pytest.mark.parametrize("family", ["saturation", "suppression", "identity", "linear"])
def test_matched_m1_independent_formula_and_subspace(family):
    # Non-unit and overlapping columns expose omitted norm and wrong averaging.
    v = torch.tensor([[1, .5j], [2, .7], [.1j, 1.3], [.5, 2]], dtype=torch.complex128)
    z = torch.tensor([[1+.3j, .4-.5j], [0, .7j]], dtype=torch.complex128)
    interface = SpatialMaterialInterface(2.3, 3, lambda c: (c@v.T).reshape(-1, 2, 2))
    response = SyntheticResponse(family, .4, 1.7, .8)
    output, a = matched_mode_response(z, v, (2, 2), .3, interface, response, mode_chunk_size=1)
    density = np.abs(v.numpy())**2
    x = np.abs(z.numpy())[:, None, :]**2*density[None]/2.3/1.7
    if family == "saturation":
        t = np.sqrt(1-.4+.4*x/(1+x))
    elif family == "suppression":
        t = 1/np.sqrt(1+.4*x)
    elif family == "linear":
        t = np.ones_like(x)*math.sqrt(.8)
    else:
        t = np.ones_like(x)
    expected = np.sum(density[None]*t*.3, axis=1)/np.sum(density*.3, axis=0)
    np.testing.assert_allclose(a.numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(output.numpy(), z.numpy()*expected, rtol=1e-12, atol=1e-12)
    assert (a >= 0).all() and (a <= 1+1e-12).all()
    field = (output@v.T).reshape(2, 1, 2, 2)
    assert WeightedModeProjector(v, .3).metrics(field)["eta_perp"].max() < 1e-25
    if family == "saturation":
        faulty_power_average = np.sum(density[None]*t**2*.3, axis=1)/np.sum(density*.3, axis=0)
        assert np.max(np.abs(expected-faulty_power_average)) > .01
        faulty_missing_norm = np.sum(density[None]*t*.3, axis=1)
        assert not np.allclose(expected, faulty_missing_norm)


def test_matched_m1_derivatives_and_invalid_modes():
    v = torch.tensor([[1, .5], [2, .7j], [.1j, 1.3], [.5, 2]], dtype=torch.complex128)
    z = torch.tensor([[1+.3j, .4-.5j]], dtype=torch.complex128, requires_grad=True)
    interface = SpatialMaterialInterface(2.3, 3, lambda c: (c@v.T).reshape(-1, 2, 2))
    response = SyntheticResponse("saturation", .4, 1.7)
    assert torch.autograd.gradcheck(lambda values: matched_mode_response(
        values, v, (2, 2), .3, interface, response)[0], (z,))
    with pytest.raises(ValueError, match="mode norms"):
        matched_mode_response(z, v*0, (2, 2), .3, interface, response)


def test_a_gate_cannot_upgrade_frontend_presence_to_conditional_acceptance():
    from scripts.research.lightgen.validate_a_route_conditional_input_reachability_v1 import (
        conditional_gate, validate_a_trace,
    )
    from scripts.research.lightgen.validate_lightgen_m2_spatial_material_interface_v1 import DiagnosticStop
    assert conditional_gate(True, True, False, True) == "A_ROUTE_FORWARD_PATH_VALID_CONDITIONAL_CONTROL_UNPROVEN"
    assert conditional_gate(False, True, False, False) == "A_ROUTE_BLOCKED_GAMMA_NOT_SPECIFIED"
    assert conditional_gate(True, False, False, False) == "A_ROUTE_NUMERICALLY_INVALID"
    validate_a_trace({"SLM":320,"Encoder":320,"OLS_projection":320,"external_ols_injection":0}, 320)
    with pytest.raises(DiagnosticStop):
        validate_a_trace({"SLM":0,"Encoder":0,"OLS_projection":320}, 320)
    with pytest.raises(DiagnosticStop):
        validate_a_trace({"SLM":320,"Encoder":320,"OLS_projection":320,"external_ols_injection":320}, 320)
