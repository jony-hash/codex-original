from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from scripts.research.lightgen.execute_ols_decoder_mnist_reconstruction_v12 import (
    DX_M,
    _assert_row_pair,
    _roi_power_w,
    _set_frozen_encoder_phase,
    _tree_max_error,
    _validate_front_cache,
)


def test_v12_roi_power_integrates_pixel_area() -> None:
    field = torch.ones((2, 3), dtype=torch.complex64)
    assert _roi_power_w(field) == pytest.approx(6 * DX_M**2, rel=0.0, abs=1e-30)


def test_v12_frontend_cache_rejects_complex64_as_complex128() -> None:
    bundle = {
        "candidate": "candidate_2x2_C",
        "mode": "gaussian_s12",
        "grid": 4096,
        "row_id": 7,
        "precision": "complex128",
        "decoder_entrance_field": torch.zeros((100, 100), dtype=torch.complex64),
        "coefficients_sqrt_w": torch.zeros(4, dtype=torch.complex64),
    }
    with pytest.raises(ValueError, match="native dtype"):
        _validate_front_cache(bundle, candidate="candidate_2x2_C", mode="gaussian_s12",
                              grid=4096, row_id=7, precision="complex128")


def test_v12_rejects_mismatched_input_target_row() -> None:
    with pytest.raises(ValueError, match="row identity"):
        _assert_row_pair(11, 12)


def test_v12_frozen_encoder_assignment_disables_autograd_in_place_mutation() -> None:
    phase = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float32))
    runtime = SimpleNamespace(encoder=SimpleNamespace(phase=phase))
    source = torch.ones((2, 2), dtype=torch.float64)

    with pytest.raises(RuntimeError, match="in-place operation"):
        phase.copy_(source.to(dtype=phase.dtype))

    _set_frozen_encoder_phase(runtime, source, torch.device("cpu"))

    assert torch.equal(phase, torch.ones_like(phase))
    assert phase.requires_grad


def test_v12_complex_response_requires_complex_command_dtype() -> None:
    response = torch.ones((2, 3), dtype=torch.complex64)
    command = torch.full((3,), 0.5, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="scalar type ComplexFloat"):
        response @ command

    coefficients = response @ command.to(dtype=response.dtype)
    assert torch.equal(coefficients, torch.full((2,), 1.5 + 0j, dtype=torch.complex64))


def test_v12_checkpoint_comparator_accepts_equal_empty_nested_history() -> None:
    valid, error, paths = _tree_max_error(
        {"history": [], "empty": {}}, {"history": [], "empty": {}}, atol=0.0, rtol=0.0)
    assert valid
    assert error == 0.0
    assert paths == []

    invalid, mismatch_error, mismatch_paths = _tree_max_error(
        {"history": []}, {"history": [{"step": 1}]}, atol=0.0, rtol=0.0)
    assert not invalid
    assert mismatch_error == float("inf")
    assert mismatch_paths == ["root.history"]
