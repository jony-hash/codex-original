"""Runtime primitives for the independent v3 spatial-quadrature architecture."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...config import V2Config
from ...optics import FormalAmplitudeFiniteCropIdentitySystem
from ...optics.quadrature_receiver import BalancedFourPortReceiver
from ...student.spatial_quadrature_generator import HierarchicalSpatialAmplitudeGenerator


SPATIAL_ARCHITECTURE = "conditional_spatial_quadrature_v3"
SPATIAL_GENERATOR_CONTRACT = "spatial_quadrature_generator_v3|label32+global8+local24|200x200|single_pass"
SPATIAL_RECEIVER_CONTRACT = "single_exposure_balanced_four_port_v1|R_difference_formal_view|no_software_decoder"


@dataclass(frozen=True)
class SpatialCalibration:
    receiver_exposure: float
    lo_amplitude: float
    r_calibration_scale: float
    calibration_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "receiver_exposure": self.receiver_exposure,
            "lo_amplitude": self.lo_amplitude,
            "r_calibration_scale": self.r_calibration_scale,
            "calibration_seed": self.calibration_seed,
        }


class SpatialQuadratureBundle(nn.Module):
    """Trainable generator + phase-only optical system + fixed receiver."""

    architecture_kind = SPATIAL_ARCHITECTURE

    def __init__(self, config: V2Config, *, calibration: SpatialCalibration | None = None) -> None:
        super().__init__()
        if config.architecture != SPATIAL_ARCHITECTURE or config.spatial_quadrature_generation is None:
            raise ValueError("spatial quadrature bundle requires conditional_spatial_quadrature_v3")
        settings = config.spatial_quadrature_generation
        self.system = FormalAmplitudeFiniteCropIdentitySystem(config.system)
        self.generator = HierarchicalSpatialAmplitudeGenerator()
        selected = calibration or SpatialCalibration(
            settings.receiver_exposure,
            settings.lo_amplitude,
            settings.r_calibration_scale,
            settings.calibration_seed,
        )
        self.receiver = BalancedFourPortReceiver(
            exposure=selected.receiver_exposure,
            lo_amplitude=selected.lo_amplitude,
            r_calibration_scale=selected.r_calibration_scale,
            beamsplitter_amplitude_factor=settings.beamsplitter_amplitude_factor,
        )
        self.calibration = selected


def seeded_spatial_bundle(config: V2Config, *, calibration: SpatialCalibration | None = None) -> SpatialQuadratureBundle:
    if config.spatial_quadrature_generation is None:
        raise ValueError("spatial quadrature config is missing")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.spatial_quadrature_generation.initialization_seed)
        return SpatialQuadratureBundle(config, calibration=calibration)


def spatial_codes(count: int, *, device: torch.device, seed: int) -> torch.Tensor:
    if count < 1:
        raise ValueError("spatial code count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(count, 32, generator=generator, dtype=torch.float32).to(device)


def spatial_forward(
    bundle: SpatialQuadratureBundle,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    return_packed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    command = bundle.generator(codes, labels)
    result = bundle.system(
        bundle.system.command(command),
        return_optical_trace=True,
    )
    if result.optical_trace is None:
        raise RuntimeError("spatial quadrature forward did not return an optical trace")
    camera_field = result.optical_trace[-1]
    readout = bundle.receiver(camera_field)
    view = bundle.receiver.formal_view(readout)
    packed = readout.packed.tensor if return_packed else None
    return view, command, packed


def spatial_forward_batched(
    bundle: SpatialQuadratureBundle,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    microbatch_size: int = 1,
    return_packed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if microbatch_size < 1:
        raise ValueError("spatial optical microbatch_size must be positive")
    views, commands, packed = [], [], []
    for start in range(0, codes.shape[0], microbatch_size):
        view, command, sensor = spatial_forward(
            bundle,
            codes[start : start + microbatch_size],
            labels[start : start + microbatch_size],
            return_packed=return_packed,
        )
        views.append(view)
        commands.append(command)
        if sensor is not None:
            packed.append(sensor)
    return (
        torch.cat(views, dim=0),
        torch.cat(commands, dim=0),
        torch.cat(packed, dim=0) if packed else None,
    )


def spatial_parameter_groups(bundle: SpatialQuadratureBundle) -> dict[str, tuple[nn.Parameter, ...]]:
    return {
        "generator": tuple(bundle.generator.parameters()),
        "encoder_phase": tuple(bundle.system.encoder.parameters()),
        "decoder_phase": tuple(bundle.system.decoder.parameters()),
    }


def freeze_spatial_generator(bundle: SpatialQuadratureBundle) -> None:
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(False)


def unfreeze_spatial_generator(bundle: SpatialQuadratureBundle) -> None:
    for parameter in bundle.generator.parameters():
        parameter.requires_grad_(True)
