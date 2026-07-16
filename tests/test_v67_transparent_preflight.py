from __future__ import annotations

import torch

from tools.diagnose_v67_transparent_boundary_convergence import (
    choose_grid,
    correlation,
    guard_band_energy,
    phase_circular_error,
    relative_l2,
    total_energy,
)
from tools.diagnose_v67_transparent_boundary_convergence import run


def test_v67_metrics_are_deterministic() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert relative_l2(value, value) == 0.0
    assert correlation(value, value) > 0.999
    assert guard_band_energy(torch.ones(4, 4), guard_fraction=0.25) > 0.0


def test_v67_energy_is_sum_and_phase_error_is_wrapped() -> None:
    field = torch.ones(1, 1, 2, 2, dtype=torch.complex64)
    assert torch.equal(total_energy(field), torch.tensor([4.0]))
    shifted = torch.polar(torch.ones_like(field.real), torch.full_like(field.real, 2.0 * torch.pi - 0.1))
    assert phase_circular_error(shifted, field) < 0.11


def test_v67_selects_smallest_grid_only_when_all_gates_pass() -> None:
    result = choose_grid(
        {
            1024: {
                "camera_intensity_corr": 0.9995,
                "central_field_relative_l2": 5e-4,
                "energy_relative_difference": 5e-4,
                "guard_band_energy_ratio": 5e-5,
            },
            1536: {},
        }
    )
    assert result["status"] == "converged"
    assert result["selected_grid"] == 1024


def test_v67_rejects_nonconverged_grid() -> None:
    result = choose_grid({1024: {"camera_intensity_corr": 0.9}, 1536: {}})
    assert result == {
        "status": "grid-not-converged",
        "selected_grid": None,
        "reference_grid": 2048,
    }


def test_v67_report_uses_versioned_schema_without_inputs(tmp_path) -> None:
    report = run(output=tmp_path / "preflight.json", device="cpu")
    assert report["schema"] == "v67_transparent_convergence_v2"
    assert report["status"] == "preflight-input-required"
