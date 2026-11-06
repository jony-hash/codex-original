from __future__ import annotations

import torch

from scripts.research.lightgen import diagnose_ols_frozen_phase_controllability_v7 as v7


def test_fixed_config_and_selected_candidate_set_are_preserved() -> None:
    config = v7._load_config(v7.DEFAULT_CONFIG)
    assert tuple(config["optics"]["propagation_grid_sizes"]) == (3072, 4096, 6144)
    assert {item["endpoint_key"] for item in config["candidates"]["optimized"]} == set(v7.TARGET_NAMES)
    assert config["scope"]["formal_training_updates"] == 0
    assert config["scope"]["phase_optimization_updates"] == 0


def test_zero_phase_array_union_has_125_coordinates_and_nested_selection() -> None:
    coordinates = v7._coordinate_pool([2, 3, 5, 10], torch.device("cpu"))
    assert len(coordinates) == 125
    assert len(v7._pool_indices(coordinates, 2)) == 4
    assert len(v7._pool_indices(coordinates, 3)) == 9
    assert len(v7._pool_indices(coordinates, 5)) == 25
    assert len(v7._pool_indices(coordinates, 10)) == 100


def test_v6_candidate_qualification_consumes_nested_all_pairwise_window_evidence() -> None:
    pairs = {
        "3072_vs_4096": {"complex_relative_error_after_common_global_phase": 2.0e-6},
        "3072_vs_6144": {"complex_relative_error_after_common_global_phase": 4.0e-6},
        "4096_vs_6144": {"complex_relative_error_after_common_global_phase": 3.0e-6},
    }
    endpoint = {
        "status": "WINDOW_QUALIFIED_REPRESENTATIVE",
        "metrics": {
            "status": "MEASURED",
            "window_comparisons": {
                "complex64": {
                    **pairs,
                },
                "production_threshold": 1.0e-4,
                "production_complex64_all_three_pairs_pass": True,
            },
        },
    }
    assert v7._v6_endpoint_window_qualified(endpoint)
    invalid_pairs = {**pairs, "3072_vs_6144": {"complex_relative_error_after_common_global_phase": 1.1e-4}}
    too_large = {
        **endpoint,
        "metrics": {
            **endpoint["metrics"],
            "window_comparisons": {
                "complex64": invalid_pairs,
                "production_threshold": 1.0e-4,
                "production_complex64_all_three_pairs_pass": True,
            },
        },
    }
    assert not v7._v6_endpoint_window_qualified(too_large)


def test_amplitude_probe_population_is_deterministic_and_unnormalized() -> None:
    config = v7._load_config(v7.DEFAULT_CONFIG)
    names_a, patterns_a = v7._amp_probes(config)
    names_b, patterns_b = v7._amp_probes(config)
    assert len(names_a) == 34
    assert names_a == names_b
    assert torch.equal(patterns_a, patterns_b)
    assert torch.equal(torch.unique(patterns_a[:32]), torch.tensor([0.25, 0.75]))
    assert torch.all(patterns_a[32] == 0.5)
    assert torch.all(patterns_a[33] == 1.0)


def test_response_at_command_moves_commands_to_response_device_and_dtype() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    response = torch.tensor([[1.0 + 2.0j, -0.5 + 0.25j]], dtype=torch.complex128, device=device)
    command = torch.tensor([0.25, 0.75], dtype=torch.float32, device="cpu")
    result = v7._response_at_command(response, command)
    assert result.device == response.device
    assert result.dtype == response.dtype
    expected = response @ command.to(device=device, dtype=torch.complex128)
    torch.testing.assert_close(result, expected)


def test_forward_validation_pattern_moves_device_resident_controls_to_cpu() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    command = torch.tensor([0.1, -0.2], dtype=torch.float64, device=device)
    pattern = v7._compose_forward_validation_pattern(command)
    assert pattern.device.type == "cpu"
    torch.testing.assert_close(pattern, torch.tensor([0.6, 0.3], dtype=torch.float64))


def test_random_phase_directions_match_two_screen_tensor_shape() -> None:
    directions = v7._phase_direction_probes(4, 2026101408, torch.device("cpu"))
    assert directions.shape == (4, 2, 1, 100, 100)
    phase = torch.zeros((2, 1, 100, 100), dtype=torch.float64)
    assert (phase + directions[0]).shape == phase.shape
    rms = directions.square().mean(dim=(1, 2, 3, 4)).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms))


def test_common_roi_batch_preserves_single_field_batch_and_spatial_axes() -> None:
    field = torch.zeros((3072, 3072), dtype=torch.float32)
    rois = v7._stack_common_roi_batch([field], 3072)
    assert rois.shape == (1, 2048, 2048)


def test_power_trace_archive_keeps_scalar_and_layer_pair_columns_rectangular() -> None:
    records = {
        0: {
            "source_input_power_w": 2.0,
            "layer_1_power_before_and_after_transmission_w": [1.5, 1.0],
            "full_global_output_power_w": 0.8,
        },
        1: {
            "source_input_power_w": 2.0,
            "layer_1_power_before_and_after_transmission_w": [1.4, 0.9],
            "full_global_output_power_w": 0.7,
        },
    }
    arrays = v7._power_trace_arrays(records, 3072)
    assert arrays["power_trace_3072_source_input_power_w"].shape == (2,)
    assert arrays["power_trace_3072_layer_1_power_before_and_after_transmission_w"].shape == (2, 2)
    assert arrays["power_trace_3072_full_global_output_power_w"].tolist() == [0.8, 0.7]


def test_bounded_real_control_solves_target_and_respects_command_interval() -> None:
    h = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex128)
    coefficients = h @ torch.tensor([0.5, 0.5], dtype=torch.complex128)
    m = torch.cat((h.real, h.imag), dim=0)
    u, singular, vh = torch.linalg.svd(m, full_matrices=False)
    target = torch.tensor([0.1, 0.0], dtype=torch.float64)
    record, command = v7._bounded_control(
        m, u, singular, vh, coefficients, 1, [0.1, 0.01, 0.001, 0.0001],
        0.001, target, (0.25, 0.75), 0.5,
    )
    assert record["target_fully_within_command_bounds"]
    assert record["residual_relative_to_original_target"] < 1.0e-12
    assert float((0.5 + command).min()) >= 0.25
    assert float((0.5 + command).max()) <= 0.75


def test_shared_phase_projection_is_reported_separately_from_raw_spectrum() -> None:
    coefficients = torch.tensor([0.2 + 0.1j, -0.1 + 0.3j], dtype=torch.complex128)
    common = 1j * coefficients
    independent = torch.tensor([0.1 + 0.0j, 0.0 + 0.2j], dtype=torch.complex128)
    j = torch.stack((common, independent), dim=1)
    report = v7._phase_jacobian_shape_report(j, coefficients, [0.1, 0.01])
    assert report["raw_real_control_spectrum"]["singular_values"]
    assert report["global_phase_projected_real_control_spectrum"]["singular_values"]
    assert "one shared output global-phase direction" in report["global_phase_direction"]
