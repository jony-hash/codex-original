from __future__ import annotations

import torch

from tools.diagnose_v67_transparent_boundary_convergence import (
    choose_grid,
    correlation,
    guard_band_energy,
    relative_l2,
)


def test_v67_metrics_are_deterministic() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert relative_l2(value, value) == 0.0
    assert correlation(value, value) > 0.999
    assert guard_band_energy(torch.ones(4, 4), guard_fraction=0.25) > 0.0


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
