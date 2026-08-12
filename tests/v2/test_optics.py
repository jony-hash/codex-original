from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from ldm_project.v2.config import FormalSystemConfig
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics import (
    CoherentImagingRelay,
    FiniteCropPhaseStack,
    FormalAmplitudeFiniteCropIdentitySystem,
    angular_spectrum_propagate,
    identity_ols,
    reference_angular_spectrum_propagate,
    reference_b0_imaging,
)
from ldm_project.v2.optics.propagation import _PRODUCTION_TRANSFER_CACHE, clear_production_transfer_cache


def _field(shape: tuple[int, int] = (8, 8)) -> ComplexField:
    plane = PlaneSpec("source", shape, (2e-6, 2e-6), 532e-9, 0.0, "unit")
    tensor = torch.randn(1, 1, *shape, dtype=torch.complex64)
    return ComplexField(tensor, plane, OperationProvenance("source", None, None))


def test_asm_reference_agrees_for_small_field() -> None:
    field = _field()
    production = angular_spectrum_propagate(field, 1e-3, "production")
    reference = reference_angular_spectrum_propagate(field, 1e-3, "reference")
    assert torch.allclose(production.tensor, reference.tensor, rtol=1e-3, atol=1e-3)
    assert production.provenance.propagation_distance_m == 1e-3


def test_production_asm_cache_is_physics_and_dtype_scoped() -> None:
    clear_production_transfer_cache()
    field = _field()
    first = angular_spectrum_propagate(field, 1e-3, "first")
    second = angular_spectrum_propagate(field, 1e-3, "second")
    assert torch.equal(first.tensor, second.tensor)
    assert len(_PRODUCTION_TRANSFER_CACHE) == 1
    angular_spectrum_propagate(field, 2e-3, "other-distance")
    assert len(_PRODUCTION_TRANSFER_CACHE) == 2


def test_identity_ols_is_complex_linear() -> None:
    left, right = _field(), _field()
    scalar = torch.tensor(0.25 + 0.75j)
    combined = ComplexField(left.tensor + scalar * right.tensor, left.plane, left.provenance)
    assert torch.allclose(identity_ols(combined).tensor, identity_ols(left).tensor + scalar * identity_ols(right).tensor)


def test_named_relay_preserves_constant_field_and_provenance() -> None:
    field = _field((8, 8))
    field = ComplexField(torch.ones_like(field.tensor), field.plane, field.provenance)
    target = PlaneSpec("target", (4, 4), (4e-6, 4e-6), 532e-9, 0.4, "target")
    relay = CoherentImagingRelay(field.plane, target, -1.0, name="unit_relay", segments_m=(0.2, 0.2), focal_lengths_m=(0.1,), clear_aperture_m=1e-3)
    output = relay(field)
    reference = reference_b0_imaging(
        field,
        target,
        relay.abcd,
        name="unit_relay",
        clear_aperture_m=relay.clear_aperture_m,
        source_distance_m=relay.matrix_segments_m[0],
        target_distance_m=relay.matrix_segments_m[-1],
    )
    assert output.plane == target
    assert torch.allclose(output.tensor.abs(), torch.ones_like(output.tensor.abs()))
    assert torch.allclose(output.tensor, reference.tensor)
    assert output.provenance.parameters["relay"] == "unit_relay"
    assert output.provenance.parameters["abcd"]["B_m"] == pytest.approx(0.0)
    assert output.provenance.parameters["pupil_effective"] is True
    assert output.provenance.parameters["pupil_model"] == "circular_coherent_source_target_v1"


def test_coherent_relay_pupil_is_active_and_retention_is_bounded() -> None:
    field = _field((32, 32))
    target = PlaneSpec("target", (16, 16), (4e-6, 4e-6), 532e-9, 0.4, "target")
    wide = CoherentImagingRelay(field.plane, target, -1.0, name="wide", segments_m=(0.2, 0.2), focal_lengths_m=(0.1,), clear_aperture_m=0.02)
    narrow = CoherentImagingRelay(field.plane, target, -1.0, name="narrow", segments_m=(0.2, 0.2), focal_lengths_m=(0.1,), clear_aperture_m=0.001)
    wide_out, narrow_out = wide(field), narrow(field)
    assert not torch.equal(wide_out.tensor, narrow_out.tensor)
    for relay in (wide, narrow):
        for key in ("source_pupil_retention", "target_pupil_retention"):
            value = relay.last_diagnostic[key]
            assert torch.isfinite(value).all()
            assert (value >= 0).all() and (value <= 1.0 + 1.0e-4).all()
    assert float(narrow.last_diagnostic["source_pupil_retention"].mean()) <= float(wide.last_diagnostic["source_pupil_retention"].mean())


def test_coherent_relay_pupil_preserves_complex_gradients() -> None:
    field = _field((16, 16))
    field.tensor.requires_grad_(True)
    target = PlaneSpec("target", (8, 8), (4e-6, 4e-6), 532e-9, 0.4, "target")
    relay = CoherentImagingRelay(field.plane, target, -1.0, name="grad", segments_m=(0.2, 0.2), focal_lengths_m=(0.1,), clear_aperture_m=0.01)
    relay(field).tensor.abs().square().mean().backward()
    assert field.tensor.grad is not None
    assert torch.isfinite(field.tensor.grad.real).all() and torch.isfinite(field.tensor.grad.imag).all()


def test_finite_crop_phase_stack_preserves_shape_and_unit_mask() -> None:
    field = _field((8, 8))
    stack = FiniteCropPhaseStack(layers=2, active_shape=(8, 8), propagation_shape=(16, 16), spacing_m=1e-3)
    output, records, stats = stack(field, "unit")
    assert output.tensor.shape[-2:] == (8, 8)
    assert len(records) == 2
    assert torch.allclose(torch.polar(torch.ones_like(stack.phase), stack.phase).abs(), torch.ones_like(stack.phase))
    assert stats["crop_retention"].shape == (1, 2)


def test_formal_system_is_fixed_to_the_formal_contract() -> None:
    system = FormalAmplitudeFiniteCropIdentitySystem(FormalSystemConfig(kind="formal_amplitude_finite_crop_identity"))
    assert system.encoder.layers == 2
    assert system.decoder.layers == 2
    assert system.front.magnification == -0.4
    assert system.back.magnification == -2.5
    assert system.camera_plane.z_m - system.chip_plane.z_m == pytest.approx(0.350)
    assert system.front.abcd.b_m == pytest.approx(0.0, abs=1e-9)
    assert system.back.abcd.b_m == pytest.approx(0.0, abs=1e-9)


def test_formal_trace_has_one_root_and_connected_provenance() -> None:
    config = FormalSystemConfig(kind="diagnostic_amplitude_finite_crop_identity", slm_shape=(8, 8), slm_pixel_pitch_m=62.5e-6, metasurface_shape=(4, 4), metasurface_pixel_pitch_m=50e-6, propagation_grid_size=8)
    system = FormalAmplitudeFiniteCropIdentitySystem(config)
    result = system.forward(system.command(torch.full((1, 1, 8, 8), 0.5)), return_optical_trace=True)
    assert result.optical_trace is not None
    roots = [field for field in result.optical_trace if field.provenance.upstream_plane is None]
    assert [field.provenance.operation for field in roots] == ["amplitude_slm"]
    assert all(field.provenance.upstream_plane is not None or field.provenance.operation in {"ideal_amplitude_slm", "amplitude_slm"} for field in result.optical_trace)


def test_optics_does_not_import_upper_layers() -> None:
    optics_root = Path("src/ldm_project/v2/optics")
    forbidden = {"student", "teacher", "application", "diagnostics"}
    for path in optics_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(part in forbidden for part in node.module.split(".")), path
            if isinstance(node, ast.Import):
                assert not any(part in forbidden for alias in node.names for part in alias.name.split(".")), path
