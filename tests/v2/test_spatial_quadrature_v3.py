from __future__ import annotations

import torch

from ldm_project.v2.application.cli_handlers import (
    SPATIAL_QUADRATURE_ARCHITECTURE,
    _SPATIAL_QUADRATURE_HANDLERS,
)
from ldm_project.v2.application.current.spatial_quadrature_runtime import (
    seeded_spatial_bundle,
    spatial_codes,
)
from ldm_project.v2.config import load_config
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.quadrature_receiver import BalancedFourPortReceiver
from ldm_project.v2.student.spatial_quadrature_generator import (
    HierarchicalSpatialAmplitudeGenerator,
)


def test_v3_config_and_dispatch_are_independent_from_identity() -> None:
    config = load_config("configs/v2/current/formal_spatial_quadrature_v3.yaml")
    assert config.architecture == SPATIAL_QUADRATURE_ARCHITECTURE
    assert config.system.kind == "formal_amplitude_finite_crop_quadrature_v1"
    assert config.spatial_quadrature_generation is not None
    assert config.spatial_quadrature_generation.code_dim == 32
    assert config.conditional_generation is None
    assert ("student", "train") in _SPATIAL_QUADRATURE_HANDLERS
    assert ("student", "sample") in _SPATIAL_QUADRATURE_HANDLERS


def test_hierarchical_generator_is_deterministic_and_uses_all_code_blocks() -> None:
    torch.manual_seed(9241)
    generator = HierarchicalSpatialAmplitudeGenerator()
    labels = torch.tensor([0, 1], dtype=torch.long)
    code = torch.randn(2, 32)
    first = generator(code, labels)
    second = generator(code, labels)
    assert first.shape == (2, 1, 200, 200)
    assert torch.equal(first, second)
    assert bool(torch.isfinite(first).all())
    assert float(generator(code + torch.nn.functional.pad(torch.ones(2, 8), (0, 24)), labels).sub(first).abs().max()) > 0.0
    assert float(generator(code + torch.nn.functional.pad(torch.ones(2, 24), (8, 0)), labels).sub(first).abs().max()) > 0.0


def test_four_port_receiver_is_single_exposure_and_exactly_recovers_quadratures() -> None:
    plane = PlaneSpec("camera", (200, 200), (2.5e-6, 2.5e-6), 532e-9, 1.0, "camera")
    field = ComplexField(
        torch.full((1, 1, 200, 200), complex(1.0, 2.0)),
        plane,
        OperationProvenance("test_camera_field", "camera", None),
    )
    receiver = BalancedFourPortReceiver(exposure=2.0, lo_amplitude=0.7, r_calibration_scale=1.0)
    readout = receiver(field)
    assert readout.packed.tensor.shape == (1, 1, 400, 400)
    assert torch.allclose(readout.r_difference, torch.full_like(readout.r_difference, 1.4), atol=1e-6)
    assert torch.allclose(readout.j_difference, torch.full_like(readout.j_difference, 2.8), atol=1e-6)
    residuals = receiver.energy_identity(field, readout)
    assert max(float(value.abs().max()) for value in residuals.values()) < 1e-6
    assert receiver.formal_view(readout).shape == (1, 1, 28, 28)


def test_spatial_bundle_keeps_receiver_fixed_and_returns_four_port_view() -> None:
    config = load_config("configs/v2/current/formal_spatial_quadrature_v3.yaml")
    bundle = seeded_spatial_bundle(config)
    codes = spatial_codes(3, device=torch.device("cpu"), seed=17)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    commands = bundle.generator(codes, labels)
    assert commands.shape == (3, 1, 200, 200)
    assert not any(parameter.requires_grad for parameter in bundle.receiver.parameters())
