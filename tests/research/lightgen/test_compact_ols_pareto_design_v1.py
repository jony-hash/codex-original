from __future__ import annotations

import math

import torch

from scripts.research.lightgen import diagnose_compact_ols_pareto_design_v1 as study


def test_sweep_grid_and_derived_propagation_support_match_plan() -> None:
    config = study._load_config(study.DEFAULT_CONFIG)
    grid = study._architecture_grid(config)

    assert len(grid) == 280
    assert study._required_grid_size(config) == 4096
    assert config["optics"]["propagation_grid_size"] == 4096
    assert {item["aperture_pixels"]: round(item["aperture_um"], 6) for item in grid if item["fiber_pitch_m"] == 125e-6 and item["last_distance_m"] == 0.5e-3} == {
        100: 200.0, 150: 300.0, 200: 400.0, 250: 500.0,
        300: 600.0, 400: 800.0, 500: 1000.0, 600: 1200.0,
    }
    assert all(item["scan_role"] == "STANDARD_SM_FIBER_REFERENCE" for item in grid if item["fiber_pitch_m"] == 125e-6)
    assert all(item["scan_role"] == "NON_STANDARD_GEOMETRY" for item in grid if item["fiber_pitch_m"] != 125e-6)


def test_power_metrics_use_input_denominator_and_preserve_uniform_100_channel_case() -> None:
    power_each = torch.full((100,), 1.0e-6, dtype=torch.float64)
    metrics = study._distribution_metrics(power_each, torch.tensor(2.0e-4), 2.0e-4, 2.0e-4, 2.0e-6)

    assert math.isclose(metrics["eta_total"], 0.5, rel_tol=1e-12)
    assert math.isclose(metrics["fiber_capture_fraction_after_encoder"], 0.5, rel_tol=1e-7)
    assert math.isclose(metrics["N_eff"], 100.0, rel_tol=1e-12)
    assert metrics["P10_over_P90"] == 1.0
    assert metrics["coefficient_of_variation"] == 0.0
    assert metrics["utilization_fraction_by_relative_threshold"] == {"-40_dB": 1.0, "-30_dB": 1.0, "-20_dB": 1.0, "-10_dB": 1.0}


def test_concentrated_channel_metrics_do_not_invent_uniform_coverage() -> None:
    power_each = torch.zeros(100, dtype=torch.float64)
    power_each[55] = 4.0e-6
    metrics = study._distribution_metrics(power_each, torch.tensor(1.0e-4), 1.0e-4, 2.0e-4, 2.0e-6)

    assert math.isclose(metrics["eta_total"], 0.04, rel_tol=1e-12)
    assert metrics["N_eff"] == 1.0
    assert metrics["utilization_fraction_by_relative_threshold"]["-10_dB"] == 0.01
    assert metrics["fiber_power_map_w"][5][5] == 4.0e-6


def test_three_objectives_have_distinct_declared_consumers() -> None:
    config = study._load_config(study.DEFAULT_CONFIG)
    power = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    measurement = {"power_each_w": power}

    throughput = study._objective("maximize_total_coupling", measurement, 0.25, 10.0)
    dimension = study._objective("maximize_effective_dimension", measurement, 0.25, 10.0)
    uniform = study._objective("maximize_uniform_distribution", measurement, 0.25, 10.0)

    assert math.isclose(float(throughput), 0.1)
    assert math.isclose(float(dimension), 1.0)
    assert math.isclose(float(uniform), -0.75)
    assert config["optimization"]["objective_names"] == list(study.OBJECTIVES)


def test_level_a_requires_all_three_simulation_screen_conditions() -> None:
    config = study._load_config(study.DEFAULT_CONFIG)
    metrics = {"N_eff": 95.0, "eta_total": 0.31, "edge_retention_corner_over_center": 0.002}
    assert study._level(metrics, config) == "LEVEL_A"
    metrics["eta_total"] = 0.29
    assert study._level(metrics, config) == "LEVEL_A_DIMENSION_ONLY_SCREEN_INCOMPLETE"
    assert study._level({"N_eff": 70.0}, config) == "LEVEL_B"
    assert study._level({"N_eff": 49.9}, config) == "LEVEL_C"


def test_pareto_frontier_removes_a_point_dominated_in_all_three_axes() -> None:
    points = [
        {"aperture_m": 2.0, "N_eff": 70.0, "eta_total": 0.40, "name": "dominated"},
        {"aperture_m": 3.0, "N_eff": 85.0, "eta_total": 0.45, "name": "wide_high_dim"},
        {"aperture_m": 1.0, "N_eff": 55.0, "eta_total": 0.30, "name": "compact"},
        {"aperture_m": 1.5, "N_eff": 71.0, "eta_total": 0.41, "name": "dominates_middle"},
    ]

    frontier = study._pareto_frontier(points)

    assert {point["name"] for point in frontier} == {"wide_high_dim", "compact", "dominates_middle"}


def test_minimum_aperture_summary_chooses_best_measured_point_at_that_size() -> None:
    points = [
        {"aperture_pixels": 500, "N_eff": 82.0, "eta_total": 0.02, "last_distance_m": 0.005},
        {"aperture_pixels": 500, "N_eff": 95.0, "eta_total": 0.05, "last_distance_m": 0.010},
        {"aperture_pixels": 600, "N_eff": 99.0, "eta_total": 0.35, "last_distance_m": 0.001},
    ]

    selected = study._minimum_aperture_over_dimension_threshold(points, 80.0)

    assert selected is not None
    assert selected["aperture_pixels"] == 500
    assert selected["N_eff"] == 95.0


def test_manufacturability_score_exposes_relative_components_and_custom_pitch() -> None:
    config = study._load_config(study.DEFAULT_CONFIG)
    standard = study._manufacturability({"aperture_pixels": 100, "fiber_pitch_m": 125e-6}, config)
    custom = study._manufacturability({"aperture_pixels": 600, "fiber_pitch_m": 50e-6}, config)

    assert standard["manufacturability_score"] == 100.0
    assert custom["fiber_pitch_below_75um_custom_requirement"] is True
    assert custom["printer_aperture_feature_and_area_limits"] == "UNRATED_NOT_SUPPLIED"
    assert custom["manufacturability_score"] < standard["manufacturability_score"]
