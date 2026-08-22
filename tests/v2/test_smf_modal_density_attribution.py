from __future__ import annotations

import torch

from ldm_project.v2.application.research.smf_modal_density_attribution import (
    MODES,
    build_nested_modal_banks,
    gram_summary,
    project_reemit,
    require_smf_v2_implementation,
)
from ldm_project.v2.contracts import PlaneSpec
from ldm_project.v2.optics.fiber_operator import FiberArraySpec


def _plane() -> PlaneSpec:
    return PlaneSpec("encoder_input", (100, 100), (2e-6, 2e-6), 532e-9, 0.0, "encoder")


def test_nested_synthetic_modal_banks_and_gram_gates() -> None:
    plane = _plane()
    banks = build_nested_modal_banks(plane, spec=FiberArraySpec())
    assert tuple(banks) == MODES
    assert [banks[count].shape[0] for count in MODES] == [100, 196, 400]
    area = 4e-12
    for count in MODES:
        summary = gram_summary(banks[count], area)
        assert summary["rank"] == count
        assert summary["max_identity_error"] <= 1e-5
    assert torch.allclose(banks[100], banks[196][:100], atol=1e-5, rtol=0)
    assert torch.allclose(banks[196], banks[400][:196], atol=1e-5, rtol=0)


def test_projection_is_finite_and_energy_nonincreasing() -> None:
    plane = _plane()
    banks = build_nested_modal_banks(plane, spec=FiberArraySpec())
    field = torch.randn(2, 100, 100, dtype=torch.complex64)
    input_energy = field.abs().square().sum((-2, -1))
    retentions = []
    for count in MODES:
        coefficients, output = project_reemit(field, banks[count], 4e-12)
        assert coefficients.shape == (2, count)
        assert torch.isfinite(output).all()
        retentions.append(float(output.abs().square().sum((-2, -1)).mean() / input_energy.mean()))
    assert retentions[0] <= retentions[1] + 1e-5 <= retentions[2] + 1e-5


def test_placeholder_smf_lifecycle_fails_closed() -> None:
    try:
        require_smf_v2_implementation()
    except RuntimeError as error:
        assert "non-consumable" in str(error)
    else:
        raise AssertionError("incomplete SMF lifecycle must not issue evidence")
