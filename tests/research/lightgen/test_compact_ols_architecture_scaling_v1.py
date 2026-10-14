from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/research/lightgen/diagnose_compact_ols_architecture_scaling_v1.py"
SPEC = importlib.util.spec_from_file_location("compact_ols_architecture_scaling_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_requested_grid_and_fixed_z_comparators_are_explicit() -> None:
    config = MODULE._load_config(MODULE.DEFAULT_CONFIG)
    grid = MODULE._architecture_grid(config)
    assert len(grid) == 284
    assert sum(item["scan_role"] == "requested_grid" for item in grid) == 280
    for aperture in (100, 200, 300, 400):
        assert sum(item["aperture_pixels"] == aperture and item["last_distance_m"] == 1.5e-3
                   and item["fiber_pitch_m"] == 125e-6 for item in grid) == 1


def test_kinematic_map_has_100_explicit_ports_and_pitch_dependent_geometry() -> None:
    config = MODULE._load_config(MODULE.DEFAULT_CONFIG)
    architecture = {"aperture_pixels": 100, "aperture_m": 200e-6,
                    "last_distance_m": 1.5e-3, "fiber_pitch_m": 125e-6}
    result = MODULE._reachability(architecture, config)
    assert len(result["fiber_rows"]) == 100
    assert len(result["map_10x10"]) == 10
    assert math.isclose(result["fiber_rows"][0]["center_x_m"], -562.5e-6, rel_tol=0.0, abs_tol=1e-18)
    assert math.isclose(result["fiber_array_outer_extent_m"], 1.25e-3, rel_tol=0.0, abs_tol=1e-18)
    assert 0 <= result["pass_count"] <= 100


def test_distribution_metrics_preserve_raw_channel_power_and_known_uniform_case() -> None:
    powers = torch.full((100,), 1e-6, dtype=torch.float64)
    result = MODULE._distribution_metrics(powers, torch.tensor(1e-3, dtype=torch.float64), input_power_w=2e-3)
    assert math.isclose(result["N_eff"], 100.0, rel_tol=1e-14)
    assert math.isclose(result["eta_total"], 0.1, rel_tol=1e-14)
    assert math.isclose(result["P_fiber_total_w"], 1e-4, rel_tol=1e-14)
    assert math.isclose(result["system_input_to_fiber_efficiency"], 0.05, rel_tol=1e-14)
    assert math.isclose(result["edge_retention_corner_mean_over_center_four_mean"], 1.0, rel_tol=1e-14)
    assert math.isclose(result["uniformity_P10_over_P90"], 1.0, rel_tol=1e-14)
    assert math.isclose(result["coefficient_of_variation"], 0.0, abs_tol=1e-14)
    assert result["utilization_fraction_by_relative_dB"] == {"-40": 1.0, "-30": 1.0, "-20": 1.0, "-10": 1.0}


def test_zero_coupling_metrics_are_finite_and_do_not_invent_coverage() -> None:
    result = MODULE._distribution_metrics(torch.zeros(100), torch.tensor(1e-3), input_power_w=1e-3)
    assert result["N_eff"] == 0.0
    assert result["eta_total"] == 0.0
    assert result["P_fiber_total_w"] == 0.0
    assert all(value == 0.0 for value in result["utilization_fraction_by_relative_dB"].values())


def test_uniform_distribution_objective_uses_configured_target() -> None:
    measured = {"eta_each_tensor": torch.tensor([0.8, 0.2], dtype=torch.float64)}
    configured = MODULE._objective("maximize_uniform_distribution", measured, uniform_target=0.75)
    legacy_default = MODULE._objective("maximize_uniform_distribution", measured, uniform_target=0.01)
    assert not torch.isclose(configured, legacy_default)
