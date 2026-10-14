from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.research.lightgen.diagnose_ols_g3_reference_closure_v4 import (
    _analytic_controls,
    _bandlimited_integral,
    _frequency_quadrature,
    _validate_saved_raw_level,
)
from scripts.research.lightgen import diagnose_ols_g3_second_hop_error_isolation_v3 as v3


@pytest.mark.parametrize("order", [4, 8])
def test_frequency_quadrature_has_exact_domain_measure_and_sorted_nodes(order: int) -> None:
    node_count = order * 32
    frequencies, weights = _frequency_quadrature(node_count, order)
    nyquist = 1.0 / (2.0 * v3.DX_M)
    assert frequencies.shape == weights.shape == (node_count,)
    assert np.all(np.diff(frequencies) > 0.0)
    assert frequencies[0] > -nyquist
    assert frequencies[-1] < nyquist
    assert np.sum(weights) == pytest.approx(2.0 * nyquist, rel=1e-14)


@pytest.mark.parametrize("row,column", [(50, 50), (50, 99), (0, 0)])
def test_zero_distance_single_pixel_matches_analytic_sinc(row: int, column: int) -> None:
    axis = (np.arange(100, dtype=np.float64) + 0.5) * v3.DX_M - 100 * v3.DX_M / 2
    offset_x, offset_y = 0.37 * v3.DX_M, -0.21 * v3.DX_M
    field = np.zeros((100, 100), dtype=np.complex128)
    field[row, column] = np.exp(0.31j)
    position = np.asarray([[axis[column] + offset_x, axis[row] + offset_y]])
    observed = _bandlimited_integral(field, 0.0, position, 2048)[0]
    expected = np.exp(0.31j) * np.sinc(offset_x / v3.DX_M) * np.sinc(offset_y / v3.DX_M)
    assert abs(observed - expected) / (np.sqrt(v3.SOURCE_POWER_W) / 200e-6) <= 1e-10


def test_far_position_and_two_pixel_coherent_phase_are_preserved() -> None:
    axis = (np.arange(100, dtype=np.float64) + 0.5) * v3.DX_M - 100 * v3.DX_M / 2
    field = np.zeros((100, 100), dtype=np.complex128)
    field[50, 50] = 1.0
    far = np.asarray([[axis[50] + 1.71e-3, axis[50] - 1.83e-3]])
    actual = _bandlimited_integral(field, 0.0, far, 32768)[0]
    dx, dy = far[0] - np.asarray([axis[50], axis[50]])
    expected = np.sinc(dx / v3.DX_M) * np.sinc(dy / v3.DX_M)
    assert abs(actual - expected) / (np.sqrt(v3.SOURCE_POWER_W) / 200e-6) <= 1e-10

    coherent = np.zeros((100, 100), dtype=np.complex128)
    coherent[50, 50] = 1.0
    coherent[50, 51] = 1.0j
    point = np.asarray([[axis[50] + 0.5 * v3.DX_M, axis[50]]])
    actual_coherent = _bandlimited_integral(coherent, 0.0, point, 2048)[0]
    expected_coherent = np.sinc(0.5) + 1j * np.sinc(-0.5)
    assert abs(actual_coherent - expected_coherent) / (np.sqrt(v3.SOURCE_POWER_W) / 200e-6) <= 1e-10


def test_analytic_control_marks_dropped_complex_phase_as_rejected() -> None:
    controls = _analytic_controls()
    coherent = controls["coherent_two_pixel_known_phase"]
    negative_controls = controls["dropped_imaginary_phase_negative_control_error_over_E0"]

    assert coherent["lost_phase_rejected"] is True
    assert negative_controls > 1e-10


def test_raw_reference_consumer_rejects_missing_complex_values(tmp_path) -> None:
    path = tmp_path / "missing_value.npz"
    metadata = json.dumps({
        "schema": "ols_g3_reference_level.v1",
        "phase": "zero",
        "quadrature_order": 4,
        "frequency_nodes_per_axis": 4,
    })
    np.savez_compressed(
        path,
        metadata_json=np.asarray(metadata),
        frequency_nodes_hz=np.zeros(4),
        frequency_weights_per_hz=np.ones(4),
        positions_xy_m=np.zeros((125, 2)),
        common_middle_complex128=np.zeros((100, 100), dtype=np.complex128),
        last_phase_rad=np.zeros((100, 100)),
    )
    with pytest.raises(ValueError, match="missing required arrays"):
        _validate_saved_raw_level(
            path,
            phase_name="zero",
            order=4,
            node_count=4,
            expected_positions=np.zeros((125, 2)),
            expected_input=np.zeros((100, 100), dtype=np.complex128),
            expected_phase=np.zeros((100, 100)),
        )
