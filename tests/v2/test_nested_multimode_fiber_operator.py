from __future__ import annotations

import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.optics.few_mode_fiber_operator import FewModeFiberArrayOLS, FewModeFiberArraySpec
from ldm_project.v2.optics.nested_multimode_fiber_operator import (
    NestedMultimodeFiberArrayOLS,
    NestedMultimodeFiberArraySpec,
    _orthonormalize_residual_block,
)


def _plane() -> PlaneSpec:
    return PlaneSpec("fiber_input", (100, 100), (2.0e-6, 2.0e-6), 532.0e-9, 0.0, "encoder")


def _build() -> NestedMultimodeFiberArrayOLS:
    return NestedMultimodeFiberArrayOLS(_plane())


def test_m6_construction_contract_and_no_training_parameters() -> None:
    operator = _build()
    assert operator.diagnostic_only is True
    assert operator.input_modes.shape == (600, 100, 100)
    assert operator.output_modes.shape == (600, 100, 100)
    assert operator.transport.shape == (600, 600)
    assert operator.input_modes.dtype == torch.complex64
    assert operator.output_modes.dtype == torch.complex64
    assert operator.transport.dtype == torch.complex64
    assert torch.isfinite(operator.input_modes).all()
    assert torch.isfinite(operator.output_modes).all()
    assert torch.isfinite(operator.transport).all()
    assert tuple(operator.parameters()) == ()


def test_m6_preserves_smf_and_fmf_prefixes_exactly() -> None:
    plane = _plane()
    smf = FiberArrayOLS(plane, FiberArraySpec())
    fmf = FewModeFiberArrayOLS(plane, FewModeFiberArraySpec())
    mmf = _build()
    assert torch.equal(fmf.input_modes[0::2], smf.input_modes)
    assert torch.equal(mmf.input_modes[:200], fmf.input_modes)
    assert torch.equal(mmf.output_modes[:200], fmf.output_modes)
    assert torch.equal(mmf.input_modes[:200:2], smf.input_modes)
    assert torch.equal(mmf.output_modes[:200:2], smf.output_modes)


def test_m6_mode_metadata_matches_block_preserving_order() -> None:
    metadata = _build().metadata
    assert len(metadata["mode_family_names"]) == 600
    assert len(metadata["mode_degree"]) == 600
    assert len(metadata["port_index"]) == 600
    assert len(metadata["port_x"]) == 600
    assert len(metadata["port_y"]) == 600
    assert metadata["mode_family_names"][:4] == ["LP01", "LP11a", "LP01", "LP11a"]
    assert metadata["mode_family_names"][200:300] == ["ETA1"] * 100
    assert metadata["mode_family_names"][300:400] == ["XI2"] * 100
    assert metadata["mode_family_names"][400:500] == ["XIETA"] * 100
    assert metadata["mode_family_names"][500:600] == ["ETA2"] * 100
    assert metadata["port_index"][200:300] == list(range(100))
    assert metadata["ordering_contract"] == (
        "exact_fmf_prefix_interleaved_per_port_then_additional_candidate_blocks_y_then_x"
    )


def test_m6_bank_is_orthonormal_and_full_rank() -> None:
    operator = _build()
    area = 4.0e-12
    gram = operator.input_modes.flatten(1).conj() @ operator.input_modes.flatten(1).T * area
    diagonal = torch.diagonal(gram).real
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
    assert torch.isfinite(gram).all()
    assert float((diagonal - 1.0).abs().max()) <= 2.0e-4
    assert float(off_diagonal.abs().max()) <= 2.0e-4
    assert int(torch.linalg.matrix_rank(gram)) == 600


def test_m6_added_modes_have_deterministic_positive_sign_pivots() -> None:
    operator = _build()
    flattened = operator.input_modes[200:].real.reshape(400, -1)
    for row in flattened:
        pivot = torch.nonzero(row.abs() > 1.0e-12, as_tuple=False)[0, 0]
        assert float(row[pivot]) > 0.0


def test_m6_repeated_construction_is_identical() -> None:
    first = _build()
    second = _build()
    assert torch.equal(first.input_modes, second.input_modes)
    assert torch.equal(first.output_modes, second.output_modes)
    assert torch.equal(first.transport, second.transport)
    assert first.metadata == second.metadata


def test_m6_identity_transport_and_forward_contract() -> None:
    operator = _build()
    identity = torch.eye(600, dtype=torch.complex64)
    assert torch.equal(operator.transport, identity)
    singular_values = torch.linalg.svdvals(operator.transport)
    assert torch.allclose(singular_values, torch.ones(600), atol=0.0, rtol=0.0)

    torch.manual_seed(17)
    values = torch.randn(2, 1, 100, 100, dtype=torch.complex64)
    output = operator(ComplexField(values, _plane(), OperationProvenance("fixture", None, None)))
    assert output.tensor.shape == values.shape
    assert output.tensor.dtype == values.dtype
    assert torch.isfinite(output.tensor).all()
    assert output.provenance.operation == "synthetic_nested_mmf_m6_second_order_identity_transport"
    assert operator.last_diagnostics["coefficients"].shape == (2, 600)
    assert operator.last_diagnostics["mode_coefficients"].shape == (2, 6, 100)
    assert operator.last_diagnostics["mode_power"].shape == (2, 6)


def test_m6_rejects_unsupported_order_and_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="fixed to the M6"):
        NestedMultimodeFiberArraySpec(modes=("LP01", "LP11a", "ETA1"))
    with pytest.raises(ValueError, match="10x10"):
        NestedMultimodeFiberArraySpec(array_shape=(5, 5))
    with pytest.raises(ValueError, match="pitch"):
        NestedMultimodeFiberArrayOLS(
            PlaneSpec("fiber_input", (100, 100), (1.0e-6, 1.0e-6), 532.0e-9, 0.0, "encoder")
        )


def test_m6_rank_failure_is_fail_closed() -> None:
    accepted = torch.zeros(1, 2, 2, dtype=torch.float64)
    accepted[0, 0, 0] = 1.0
    candidates = torch.zeros(2, 2, 2, dtype=torch.float64)
    candidates[:, 0, 1] = 1.0
    with pytest.raises(RuntimeError, match="synthetic_mmf_contract_rank_failure"):
        _orthonormalize_residual_block(accepted, candidates, 1.0, "fixture")
