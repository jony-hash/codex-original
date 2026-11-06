from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.research.lightgen import diagnose_ols_decoder_camera_interface_v9 as v9


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/research/lightgen/ols_decoder_camera_interface_v9.yaml"


def test_v9_config_freezes_routes_and_numerical_consumers(tmp_path: Path) -> None:
    config = v9._load_config(CONFIG)
    assert config["optics"]["grids"] == [4096, 6144, 8192]
    assert config["gates"]["aligned_common_roi_complex64_max"] == 1.0e-4
    assert config["scope"]["formal_training_updates"] == 0
    config["gates"]["aligned_common_roi_complex64_max"] = 1.1e-4
    config_path = tmp_path / "ols_decoder_camera_interface_v9_invalid_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed numerical gates"):
        v9._load_config(config_path)


def test_camera_power_factorization_matches_independent_central_difference() -> None:
    device = torch.device("cpu")
    _nodes, weights = v9._gl_axis(4, device, torch.float64)
    x = torch.linspace(-1.0, 1.0, 800, dtype=torch.float64)
    phase = 0.31 * x[None, :] + 0.17 * x[:, None]
    response = torch.stack(
        (
            torch.ones((800, 800), dtype=torch.complex128),
            torch.polar(torch.ones((800, 800), dtype=torch.float64), phase),
        )
    )
    rng = np.random.default_rng(2026101609)
    h_np = rng.normal(size=(2, 10000)) + 1j * rng.normal(size=(2, 10000))
    h = torch.from_numpy(h_np.astype(np.complex128))
    command = torch.full((10000,), 0.5, dtype=torch.float64)
    coefficients = h @ command.to(torch.complex128)
    a, b, _field = v9._camera_linear_factors(response, weights, h, coefficients)
    direction = torch.linspace(-1.0, 1.0, 10000, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    analytic = a @ (b @ direction)
    epsilon = 1.0e-5
    plus = v9._pixel_powers_from_response(response, h @ (command + epsilon * direction).to(torch.complex128), 4, weights).reshape(-1)
    minus = v9._pixel_powers_from_response(response, h @ (command - epsilon * direction).to(torch.complex128), 4, weights).reshape(-1)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    error = torch.linalg.vector_norm(finite_difference - analytic)
    scale = torch.linalg.vector_norm(analytic).clamp_min(v9.EPS)
    assert float(error) <= 1.0e-4 * float(scale) + 1.0e-10


def test_gl_pixel_integral_preserves_constant_field_area() -> None:
    field = torch.ones((200, 200), dtype=torch.complex128)
    _samples, pixels, _weights = v9._camera_quadrature_field(field, 4)
    expected = (2.5e-6) ** 2
    assert pixels.shape == (200, 200)
    assert float(torch.max(torch.abs(pixels - expected)) / expected) <= 1.0e-12


@pytest.mark.parametrize(
    ("candidate_name", "pattern_name", "pattern_index"),
    [("candidate_2x2_C", "binary_probe_22", 22), ("candidate_3x3_B", "binary_probe_15", 15)],
)
def test_v8_window_pilot_resolves_exact_binary_pattern_despite_boolean_summary(
    candidate_name: str, pattern_name: str, pattern_index: int
) -> None:
    commands = np.zeros((34, 10000), dtype=np.float32)
    commands[pattern_index] = 0.25
    commands[pattern_index, 0] = 0.75
    commands[32] = 0.5
    candidate = {
        "pattern_names": np.asarray([f"binary_probe_{i}" for i in range(32)] + ["uniform_t_0.5", "uniform_t_1.0"]),
        "commands": commands,
    }
    source = {
        "report": {
            "v7_worst_probe_replay": True,
            "window_pilot": {
                candidate_name: {
                    "patterns": {
                        pattern_name: {
                            "input_identity": {
                                "source_v7_pattern": pattern_name,
                                "v7_probe_index": pattern_index,
                            }
                        },
                        "uniform_t_0.5": {
                            "input_identity": {"source_v7_pattern": "uniform_t_0.5", "v7_probe_index": None}
                        },
                    }
                }
            },
        }
    }
    assert v9._pilot_pattern_indices(source, candidate_name, candidate) == [pattern_index, 32]


def test_camera_noise_weighting_flattens_pixel_rows_in_production_jacobian_report(tmp_path: Path) -> None:
    device = torch.device("cpu")
    config = v9._load_config(CONFIG)
    axis = torch.linspace(-1.0, 1.0, 800, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    envelope = torch.exp(-2.0 * (xx.square() + yy.square()))
    response = torch.stack(
        (
            envelope,
            envelope * torch.exp(1j * 0.2 * xx),
            envelope * torch.exp(1j * 0.2 * yy),
            envelope * torch.exp(1j * 0.2 * (xx + yy)),
        )
    ).to(torch.complex128)
    h = np.zeros((4, 10000), dtype=np.complex128)
    h[:, :4] = np.asarray(
        [[1.0, 0.2j, 0.1, 0.0], [0.1j, 1.0, 0.2, 0.0], [0.0, 0.1, 1.0, 0.2j], [0.2, 0.0, 0.1j, 1.0]],
        dtype=np.complex128,
    ) * 1.0e-3
    _nodes, weights = v9._gl_axis(4, device, torch.float64)
    candidate = {
        "commands": np.concatenate((np.full((49, 10000), 0.5, dtype=np.float32), np.full((1, 10000), 0.25, dtype=np.float32))),
        "pattern_names": np.asarray([f"probe_{index}" for index in range(49)] + ["uniform_t_0.5"]),
    }
    # Place the required uniform command at the final row; all earlier rows
    # deliberately differ so the production helper selects this exact base.
    candidate["commands"][:49, :1] = 0.25
    candidate["commands"][49] = 0.5
    report = v9._camera_jacobian_report(
        config,
        tmp_path,
        candidate,
        "candidate_test",
        "gaussian_s12",
        "direct",
        "zero_phase",
        4096,
        response,
        weights,
        h,
        np.full((50,), 249.3e-6, dtype=np.float64),
    )
    assert report["factorization"]["A_shape"][0] == 40000
    assert report["complex128_central_difference"]["passed"] is True


@pytest.mark.parametrize("mode", ["gaussian_s12", "lp01"])
def test_static_fiber_mode_gram_is_cpu_backed_and_normalized(mode: str) -> None:
    gram = v9._fiber_mode_gram(mode, 1)
    assert gram.device.type == "cpu"
    assert gram.shape == (1, 1)
    assert float(gram[0, 0].real) == pytest.approx(1.0, abs=1.0e-4)

