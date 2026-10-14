from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.research.lightgen import diagnose_ols_window_dimension_power_v5 as v5


def test_odd_and_even_array_coordinates_are_nested_in_two_unique_pools() -> None:
    for size in v5.SIZES:
        pool_size = 7 if size % 2 else 10
        indices = v5._array_pool_indices(size, pool_size)
        assert len(indices) == size**2
        assert len(set(indices)) == size**2


def test_common_global_phase_alignment_does_not_align_ports_independently() -> None:
    reference = torch.tensor([1 + 0j, 1 + 0j], dtype=torch.complex128)
    common = reference * torch.exp(torch.tensor(0.4j, dtype=torch.complex128))
    phi, aligned = v5.baseline._phase_alignment(reference, common)
    assert abs(phi - 0.4) < 1e-12
    assert v5.baseline._rel_l2(reference, aligned) < 1e-12

    port_shifted = torch.tensor([1 + 0j, 1j], dtype=torch.complex128)
    _, port_aligned = v5.baseline._phase_alignment(reference, port_shifted)
    assert v5.baseline._rel_l2(reference, port_aligned) > 0.5


def test_real_phase_dtype_preparation_preserves_complex_buffers() -> None:
    runtime = SimpleNamespace(
        encoder=SimpleNamespace(phase=nn.Parameter(torch.zeros(2, dtype=torch.float32))),
        generator=SimpleNamespace(phase=nn.Parameter(torch.zeros(2, dtype=torch.float32))),
    )
    complex_buffer = torch.tensor([0.375 + 0.625j], dtype=torch.complex64)
    before = complex_buffer.clone()
    v5._cast_real_phase_parameters(runtime, torch.float64)
    assert runtime.encoder.phase.dtype == torch.float64
    assert runtime.generator.phase.dtype == torch.float64
    assert torch.equal(complex_buffer, before)
    assert complex_buffer.imag.item() == pytest.approx(0.625)


def test_even_array_center_and_uniform_power_metrics_are_explicit() -> None:
    coefficients = torch.ones(4, dtype=torch.complex128)
    metrics = v5._power_metrics(coefficients, before_power=8.0, input_power=16.0, size=2)
    assert metrics["N_eff_single_field"] == pytest.approx(4.0)
    assert metrics["eta_fiber_face"] == pytest.approx(0.5)
    assert metrics["eta_system_fixed_source"] == pytest.approx(0.25)
    assert metrics["edge_to_center_power_ratio"] == pytest.approx(1.0)
    assert metrics["corner_to_center_power_ratio"] == pytest.approx(1.0)
    assert metrics["center_channel_count"] == 4
    assert metrics["edge_channel_count"] == 4


def test_fixed_common_roi_crop_uses_physical_window_center() -> None:
    for grid in v5.GRIDS:
        field = torch.arange(grid * grid, dtype=torch.float64).reshape(grid, grid)
        roi = v5._roi_tensor(field, grid)
        offset = (grid - v5.ROI_SIZE) // 2
        assert roi.shape == (v5.ROI_SIZE, v5.ROI_SIZE)
        assert roi[0, 0] == field[offset, offset]
        assert roi[-1, -1] == field[offset + v5.ROI_SIZE - 1, offset + v5.ROI_SIZE - 1]
