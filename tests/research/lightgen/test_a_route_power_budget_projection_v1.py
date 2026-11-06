"""Area, conjugation, nonorthogonal counterexamples, and LS-only consumer."""
import numpy as np
import pytest
import torch

from ldm_project.v2.application.research.lightgen.power_budget_projection_diagnostics import (
    OLSProjectionPowerAudit, area_power, validate_production_handoff, validate_projection_accounting,
)


def test_nonorthogonal_nonunit_modes_against_independent_qr():
    v = torch.tensor([[1,1j], [2,.3], [.4j,2], [.2,.6j]], dtype=torch.complex128)
    e = torch.tensor([1+2j, .4, 3j, -.7], dtype=torch.complex128).reshape(1,1,2,2)
    area = .2
    b = (e.flatten(1)@v.conj())*area
    producer = OLSProjectionPowerAudit(v, area)
    result = producer.analyze(e, b)
    validate_projection_accounting(result)
    q,_ = np.linalg.qr(v.numpy()*np.sqrt(area))
    expected = q@(q.conj().T@e.numpy().ravel())
    np.testing.assert_allclose(result["field_LS"].numpy().ravel(),expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(producer.gram.numpy(),v.numpy().conj().T@v.numpy()*area,rtol=1e-12,atol=1e-12)
    assert not torch.allclose(result["P_coefficient_norm_metric"],result["P_match_w"])
    # Omitting G^+ gives the actual production overlap synthesis, not the LS projection.
    assert not np.allclose(result["field_match"].numpy().ravel(),expected,rtol=1e-3)
    assert producer.basis_diagnostics()["basis_status"] == "OLS_BASIS_NONORTHOGONAL"
    with pytest.raises(ValueError,match="cannot replace"):
        validate_production_handoff(result["field_LS"],result["field_match"])
    validate_production_handoff(result["field_match"],result["field_match"])


def test_power_area_derivative_and_consumer_negative_control():
    e = torch.tensor([1+2j,.4,3j,-.7],dtype=torch.complex128).reshape(1,1,2,2).requires_grad_()
    v = torch.tensor([[1,0],[0,1],[1,1j],[.1j,.3]],dtype=torch.complex128)
    p=OLSProjectionPowerAudit(v,.3)
    def captured_power(field):
        overlaps=field.flatten(1)@v.conj()*.3
        return p.analyze(field,overlaps)["P_LS_w"]
    assert torch.autograd.gradcheck(captured_power,(e,))
    result=p.analyze(e,e.flatten(1)@v.conj()*.3)
    assert area_power(e,.6).item()==pytest.approx(2*area_power(e,.3).item())
    faulty={**result,"P_LS_w":result["P_encoder_w"]*2}
    with pytest.raises(ValueError,match="captured power"):
        validate_projection_accounting(faulty)
    with pytest.raises(ValueError):
        area_power(e,-.3)


def test_rank_deficiency_and_near_orthogonal_label_are_distinct():
    v=torch.tensor([[1,1],[2,2],[3,3],[4,4]],dtype=torch.complex128)
    p=OLSProjectionPowerAudit(v,.1)
    assert p.basis_diagnostics()["numerical_rank"]==1
    assert p.basis_diagnostics()["basis_status"]=="OLS_BASIS_ILL_CONDITIONED"
    identity=OLSProjectionPowerAudit(torch.eye(4,dtype=torch.complex128)/.5,.25)
    assert identity.basis_diagnostics()["basis_status"]=="OLS_BASIS_NEAR_ORTHOGONAL"
