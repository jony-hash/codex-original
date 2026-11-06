from __future__ import annotations

import math

import pytest
import torch

from scripts.research.lightgen import diagnose_ols_phase_dimension_power_v6 as v6
from scripts.research.lightgen import diagnose_ols_gaussian_dimension_reaudit_v1 as baseline


def test_absolute_uniform_objective_penalizes_zero_delivery() -> None:
    target = 0.25
    uniform = torch.tensor([math.sqrt(target), math.sqrt(target)], dtype=torch.complex128)
    zero = torch.zeros(2, dtype=torch.complex128)

    uniform_values = v6._objective_values(uniform, 1.0, 2, target)
    zero_values = v6._objective_values(zero, 1.0, 2, target)

    assert float(uniform_values[v6.OBJECTIVES[2]]) == 0.0
    assert float(zero_values[v6.OBJECTIVES[2]]) == 1.0
    assert float(uniform_values[v6.OBJECTIVES[0]]) < float(zero_values[v6.OBJECTIVES[0]])


def test_single_channel_neff_is_constant_one() -> None:
    coefficients = torch.tensor([0.02 + 0.03j], dtype=torch.complex128, requires_grad=True)

    neff = v6._objective_values(coefficients, 249.3e-6, 1, None)["N_eff"]
    gradient = torch.autograd.grad(neff, coefficients)[0]

    assert math.isclose(float(neff), 1.0, rel_tol=0.0, abs_tol=1e-14)
    assert float(gradient.abs().max()) <= 1e-12


def test_initialization_seeds_are_fixed_and_zero_matches_v5_state() -> None:
    zero = torch.zeros((2, 1, 100, 100), dtype=torch.float64)

    first = v6._initializations(zero)
    second = v6._initializations(zero)

    assert tuple(first) == v6.INITIALIZATIONS
    assert torch.equal(first["v5_zero"], zero.to(torch.float32))
    assert torch.equal(first[v6.INITIALIZATIONS[1]], second[v6.INITIALIZATIONS[1]])
    assert torch.equal(first[v6.INITIALIZATIONS[2]], second[v6.INITIALIZATIONS[2]])
    for name in v6.INITIALIZATIONS[1:]:
        assert float(first[name].min()) >= -math.pi
        assert float(first[name].max()) <= math.pi


def test_array_coordinate_order_matches_nested_v5_coordinate_contract() -> None:
    for size in v6.SIZES:
        axis = baseline._array_coordinates(size, v6.PITCH_M, torch.device("cpu")).tolist()
        coords = [[x, y] for y in axis for x in axis]
        assert len(coords) == size * size
        assert coords == sorted(coords, key=lambda xy: (xy[1], xy[0]))
        if size % 2:
            assert [0.0, 0.0] in coords
        else:
            assert [0.0, 0.0] not in coords


def test_adjoint_row_selection_uses_pool_device_and_v5_nested_coordinates() -> None:
    pools = {
        7: torch.arange(49, dtype=torch.float64).reshape(49, 1),
        10: torch.arange(100, dtype=torch.float64).reshape(100, 1),
    }

    selected_3 = v6._selected_adjoint_rows(pools, 3, torch.device("cpu"))
    selected_4 = v6._selected_adjoint_rows(pools, 4, torch.device("cpu"))

    assert selected_3[:, 0].tolist() == [16, 17, 18, 23, 24, 25, 30, 31, 32]
    assert selected_4[:, 0].tolist() == [33, 34, 35, 36, 43, 44, 45, 46, 53, 54, 55, 56, 63, 64, 65, 66]


def test_common_global_phase_alignment_does_not_change_intensity() -> None:
    reference = torch.tensor([1.0 + 0.2j, -0.3 + 0.7j], dtype=torch.complex128)
    test = reference * torch.polar(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.8, dtype=torch.float64),
    )

    aligned, phase = v6._align_phase(reference, test)

    assert math.isclose(phase, 0.8, rel_tol=0.0, abs_tol=1e-6)
    assert v6._relative_error(reference, aligned) <= 1e-14
    assert torch.allclose(reference.abs().square(), test.abs().square())


def test_pareto_report_consumes_v5_fixed_source_efficiency_field() -> None:
    coefficients = torch.tensor(
        [0.02 + 0.03j, 0.01 - 0.02j, -0.01 + 0.04j, 0.03 - 0.01j],
        dtype=torch.complex128,
    )
    produced = v6.v5._power_metrics(coefficients, before_power=1.0e-5, input_power=249.3e-6, size=2)

    assert "eta_system_fixed_source" in produced
    assert math.isclose(
        v6._eta_system_from_power_metrics(produced),
        produced["eta_total"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_representative_path_reader_accepts_analytic_control_endpoint_schema() -> None:
    summary = {
        "phase_path": "trajectories/n01_B_seed/phase_analytic_constant_neff.pt",
        "endpoints": {
            "best": {"phase_path": "trajectories/n01_B_seed/phase_analytic_constant_neff.pt"},
            "terminal": {"phase_path": "trajectories/n01_B_seed/phase_analytic_constant_neff.pt"},
        },
    }

    assert "best_phase_path" not in summary
    assert v6._endpoint_phase_path(summary, "best") == summary["endpoints"]["best"]["phase_path"]
    assert v6._endpoint_phase_path(summary, "terminal") == summary["endpoints"]["terminal"]["phase_path"]


def test_saved_representative_field_reuse_validates_metadata_shape_and_dtype(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v6, "ROI_SIZE", 2)
    field = torch.tensor([[1.0 + 0.5j, 0.2 - 0.4j], [-0.1j, 0.8 + 0.1j]], dtype=torch.complex64)
    metadata = {
        "schema": "ols_phase_dimension_power_v6_representative_field.v1",
        "array_size": 1,
        "objective": v6.OBJECTIVES[0],
        "initialization": "v5_zero",
        "trajectory_id": "n01_A_v5_zero",
        "endpoint": "best",
        "grid": 3072,
        "precision": "complex64",
        "dx_m": v6.DX_M,
        "common_roi_pixels": 2,
        "common_roi_extent_m": 2 * v6.DX_M,
        "global_phase_fitting_for_G3_only": "one common global phase per compared field",
    }
    path = tmp_path / "representative.npz"
    v6._save_roi_field(path, field, metadata)

    loaded = v6._load_existing_roi_field(path, metadata)

    assert loaded is not None
    assert loaded.dtype == torch.complex64
    assert torch.equal(loaded, field)
    assert v6._load_existing_roi_field(tmp_path / "missing.npz", metadata) is None
    with pytest.raises(ValueError, match="metadata differs"):
        v6._load_existing_roi_field(path, {**metadata, "trajectory_id": "wrong"})
