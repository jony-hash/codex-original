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
    reference = reference_b0_imaging(field, target, relay.abcd, name="unit_relay")
    assert output.plane == target
    assert torch.allclose(output.tensor.abs(), torch.ones_like(output.tensor.abs()))
    assert torch.allclose(output.tensor, reference.tensor)
    assert output.provenance.parameters["relay"] == "unit_relay"
    assert output.provenance.parameters["abcd"]["B_m"] == pytest.approx(0.0)


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


def test_v2_optics_does_not_import_upper_layers() -> None:
    optics_root = Path("src/ldm_project/v2/optics")
    forbidden = {"student", "teacher", "application", "diagnostics"}
    for path in optics_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(part in forbidden for part in node.module.split(".")), path
            if isinstance(node, ast.Import):
                assert not any(part in forbidden for alias in node.names for part in alias.name.split(".")), path
