"""LightGen-aligned 2-D semantic-generation input and complex-OLS runtime.

This module is an implementation of the definition-only alignment contract.
It is a research runtime boundary, not a training entry point or a scientific
generation-acceptance path.  The authoritative latent is the complex 100-port
state.  Both an optically encoded state and an externally supplied state pass
through the same coefficient-space transport and modal re-emission writer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.contracts import AmplitudeCommand, ComplexField, IntensityFrame, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.devices import IdealAmplitudeSLM
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.optics.relays import CoherentImagingRelay


OLS_GRID_SHAPE = (10, 10)
OLS_PORT_COUNT = 100
ROW_MAJOR_PORT_ORDER = tuple(
    (row, column)
    for row in range(OLS_GRID_SHAPE[0])
    for column in range(OLS_GRID_SHAPE[1])
)
OLS_INTERFACE_ID = "lightgen_10x10_complex_ols_v1"

WAVELENGTH_M = 532.0e-9
SLM_SHAPE = (200, 200)
SLM_PIXEL_PITCH_M = 2.5e-6
OLS_SHAPE = (100, 100)
OLS_PIXEL_PITCH_M = 2.0e-6
LAYER_SPACING_M = 1.50e-3
ENCODER_LAYERS = 2
GENERATOR_LAYERS = 2

# These are the existing project relay values.  They are project conventions,
# not unresolved LightGen paper implementation details.
FRONT_SEGMENTS_M = (0.025, 0.125, 0.0732142857142857, 0.3125, 0.0892857142857143)
FRONT_MATRIX_SEGMENTS_M = (
    sum(FRONT_SEGMENTS_M[:3]),
    FRONT_SEGMENTS_M[3],
    FRONT_SEGMENTS_M[4],
)
FRONT_MAGNIFICATION = -0.4
FRONT_FOCAL_LENGTHS_M = (223.2142857142857e-3, 89.2857142857143e-3)
FRONT_CLEAR_APERTURE_M = 23.75e-3
BACK_SEGMENTS_M = (0.100, 0.250)
BACK_MAGNIFICATION = -2.5
BACK_FOCAL_LENGTH_M = 71.4285714285714e-3
BACK_CLEAR_APERTURE_M = 26.60e-3


ALIGNMENT_METADATA: dict[str, tuple[str, ...]] = {
    "CHEN_FACT": (
        "amplitude-modulation SLM and coherent 4f input relay",
        "10x10 OLS dimension and complex single-mode modal coupling",
        "post-OLS generation uses the photonic generator",
    ),
    "CHEN_UNRESOLVED": (
        "digital image-to-SLM normalization, transfer curve, and illumination phase",
        "machine-readable port order and exact modal-basis normalization",
        "complete relay, layer, aperture, and generator entrance numerical details",
    ),
    "PROJECT_DECISION": (
        "normalized monochrome image, bilinear resize, and sqrt amplitude mapping",
        "200x200 SLM, 100x100 OLS plane, 532 nm, and row-major 100-port order",
        "scaled LP01 mode bank, coefficient-space T=I_100, and two-plus-two phase stacks",
        "the generator exit is coupled to the raw CMOS plane through the existing project B=0 relay",
    ),
}


@dataclass(frozen=True)
class LightGen2DAlignmentRuntimeConfig:
    """Numerical runtime knobs frozen by the aligned project baseline."""

    propagation_grid_size: int = 2048

    def __post_init__(self) -> None:
        if self.propagation_grid_size < max(OLS_SHAPE):
            raise ValueError("propagation_grid_size must cover the 100x100 active plane")


@dataclass(frozen=True)
class ComplexOLSState:
    """Canonical batch complex state at the 10x10 single-mode OLS interface."""

    coefficients: torch.Tensor
    port_order: tuple[tuple[int, int], ...] = ROW_MAJOR_PORT_ORDER
    interface_id: str = OLS_INTERFACE_ID
    source: str = "external"

    def __post_init__(self) -> None:
        if not torch.is_complex(self.coefficients):
            raise TypeError("ComplexOLSState requires a complex [batch,100] tensor")
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != OLS_PORT_COUNT:
            raise ValueError("ComplexOLSState requires shape [batch,100]")
        if not torch.isfinite(self.coefficients).all():
            raise ValueError("ComplexOLSState must be finite")
        if tuple(self.port_order) != ROW_MAJOR_PORT_ORDER:
            raise ValueError("ComplexOLSState requires the frozen row-major 10x10 port order")
        if self.interface_id != OLS_INTERFACE_ID:
            raise ValueError("ComplexOLSState belongs to another OLS interface")
        if not self.source:
            raise ValueError("ComplexOLSState source is required")

    @classmethod
    def from_external(cls, coefficients: torch.Tensor) -> "ComplexOLSState":
        """Bind a raw generated tensor to the fixed external OLS interface."""
        return cls(coefficients=coefficients, source="external_generated")

    @property
    def amplitude(self) -> torch.Tensor:
        return self.coefficients.abs()

    @property
    def phase(self) -> torch.Tensor:
        return torch.angle(self.coefficients)


@dataclass(frozen=True)
class LightGenImageOpticalInput:
    """Explicit image-to-SLM optical-input trace."""

    image: torch.Tensor
    slm_intensity: torch.Tensor
    slm_amplitude: torch.Tensor
    slm_field: ComplexField


class LightGenImageInputAdapter:
    """Map an image to an explicit amplitude-SLM command and complex field."""

    def __init__(self, slm_plane: PlaneSpec) -> None:
        self.slm_plane = slm_plane
        self.slm = IdealAmplitudeSLM()

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("LightGen image input requires shape [batch,1,H,W]")
        if torch.is_complex(image) or not torch.is_floating_point(image):
            raise TypeError("LightGen image input must be a real floating tensor")
        if not torch.isfinite(image).all():
            raise ValueError("LightGen image input must be finite")
        if image.numel() and (image.min() < 0.0 or image.max() > 1.0):
            raise ValueError("LightGen image input must be normalized to [0,1]")

    def __call__(self, image: torch.Tensor) -> LightGenImageOpticalInput:
        self._validate_image(image)
        slm_intensity = F.interpolate(
            image,
            size=self.slm_plane.shape,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        slm_amplitude = slm_intensity.sqrt()
        command = AmplitudeCommand(
            slm_amplitude,
            self.slm_plane,
            IdealAmplitudeSLM.calibration_id,
        )
        slm_field = self.slm(command)
        slm_field = ComplexField(
            slm_field.tensor,
            slm_field.plane,
            OperationProvenance(
                "lightgen_explicit_image_to_slm_field",
                slm_field.plane.plane_id,
                None,
                {
                    "image_domain": "real_normalized_nonnegative",
                    "slm_intensity_mapping": "clamp_bilinear_resize",
                    "slm_amplitude_mapping": "sqrt",
                    "illumination_phase_rad": 0.0,
                    "mapping_status": "PROJECT_DECISION",
                },
            ),
        )
        return LightGenImageOpticalInput(
            image=image,
            slm_intensity=slm_intensity,
            slm_amplitude=slm_amplitude,
            slm_field=slm_field,
        )


class LightGenAlignedComplexOLS(nn.Module):
    """The aligned 100-port complex OLS interface and modal writer."""

    diagnostic_only = False

    def __init__(self, ols_plane: PlaneSpec) -> None:
        super().__init__()
        identity_transport = torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)
        self.operator = FiberArrayOLS(
            ols_plane,
            FiberArraySpec(
                grid_shape=OLS_GRID_SHAPE,
                pitch_m=OLS_PIXEL_PITCH_M,
                wavelength_m=WAVELENGTH_M,
                waist_m=7.111e-6,
                fill=0.8,
                v_number=2.2,
                mode_model="step_index_lp01_ports",
                expected_fov_m=(200.0e-6, 200.0e-6),
            ),
            transport=identity_transport,
        )

    @property
    def transport(self) -> torch.Tensor:
        return self.operator.transport

    def project_state(self, field: ComplexField) -> ComplexOLSState:
        coefficients = self.operator.project_coefficients(field)
        return ComplexOLSState(coefficients, source="real_encoded")

    @staticmethod
    def _validate_generator_plane(target_plane: PlaneSpec, reference: PlaneSpec) -> None:
        if target_plane.shape != reference.shape:
            raise ValueError("OLS generator plane shape mismatch")
        if target_plane.pixel_pitch_m != reference.pixel_pitch_m:
            raise ValueError("OLS generator plane sampling mismatch")
        if abs(target_plane.wavelength_m - reference.wavelength_m) > 1.0e-15:
            raise ValueError("OLS generator plane wavelength mismatch")
        if target_plane.fov_m != reference.fov_m:
            raise ValueError("OLS generator plane FOV mismatch")

    def write_to_generator_input(
        self,
        state: ComplexOLSState,
        generator_plane: PlaneSpec,
    ) -> ComplexField:
        """Implement ``W_OLS(z)=sum_j (Tz)_j v_j`` at one shared boundary."""
        if not isinstance(state, ComplexOLSState):
            raise TypeError("W_OLS requires a ComplexOLSState")
        self._validate_generator_plane(generator_plane, self.operator.input_plane)
        transmitted = self.operator.transmit_coefficients(state.coefficients)
        reemitted = self.operator.reemit_coefficients(transmitted).unsqueeze(1)
        return ComplexField(
            reemitted,
            generator_plane,
            OperationProvenance(
                "lightgen_ols_coefficient_transmission_and_modal_reemission",
                self.operator.input_plane.plane_id,
                None,
                {
                    "interface_id": OLS_INTERFACE_ID,
                    "port_count": OLS_PORT_COUNT,
                    "grid_shape": list(OLS_GRID_SHAPE),
                    "port_order": "row_major_j=10*r+c",
                    "projection": "z_j=<E_encoder,u_j>",
                    "transport": "T=I_100_coefficient_space",
                    "re_emission": "sum_j_(Tz)_j_v_j",
                    "whole_field_identity_bypass": False,
                    "ols_semantics": "PROJECT_DECISION",
                },
            ),
        )


@dataclass(frozen=True)
class LightGen2DForwardResult:
    """Typed result shared by image and direct complex-OLS execution paths."""

    raw_camera: IntensityFrame
    z_ols: ComplexOLSState
    generator_input_field: ComplexField
    generator_output_field: ComplexField
    image_input: LightGenImageOpticalInput | None
    encoder_output_field: ComplexField | None
    ols_input_field: ComplexField | None
    encoder_records: tuple[ComplexField, ...]
    generator_records: tuple[ComplexField, ...]
    crop_statistics: dict[str, torch.Tensor]

    @property
    def raw_intensity(self) -> torch.Tensor:
        return self.raw_camera.tensor


class LightGen2DSemanticGenerationInputOLSAlignment(nn.Module):
    """Canonical 2-D LightGen input/OLS/generator runtime.

    The module intentionally has no label/controller input and no producer,
    prior sampler, phase repair, or electronic decoder.  It only defines the
    two paths that converge at the same complex OLS writer and generator.
    """

    def __init__(
        self,
        config: LightGen2DAlignmentRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LightGen2DAlignmentRuntimeConfig()
        wavelength = WAVELENGTH_M
        self.slm_plane = PlaneSpec(
            "lightgen_slm_input",
            SLM_SHAPE,
            (SLM_PIXEL_PITCH_M, SLM_PIXEL_PITCH_M),
            wavelength,
            0.0,
            "slm",
        )
        encoder_z = sum(FRONT_SEGMENTS_M)
        self.encoder_entrance_plane = PlaneSpec(
            "lightgen_encoder_entrance",
            OLS_SHAPE,
            (OLS_PIXEL_PITCH_M, OLS_PIXEL_PITCH_M),
            wavelength,
            encoder_z,
            "encoder",
        )
        self.ols_plane = self.encoder_entrance_plane.with_plane(
            "lightgen_ols_input",
            z_m=encoder_z + ENCODER_LAYERS * LAYER_SPACING_M,
            coordinate_frame="ols",
        )
        self.generator_entrance_plane = self.ols_plane.with_plane(
            "lightgen_generator_entrance",
            coordinate_frame="generator",
        )
        generator_exit_z = encoder_z + (ENCODER_LAYERS + GENERATOR_LAYERS) * LAYER_SPACING_M
        self.generator_exit_plane = self.generator_entrance_plane.with_plane(
            "lightgen_generator_exit",
            z_m=generator_exit_z,
        )
        self.camera_plane = PlaneSpec(
            "lightgen_raw_cmos",
            SLM_SHAPE,
            (SLM_PIXEL_PITCH_M, SLM_PIXEL_PITCH_M),
            wavelength,
            generator_exit_z + sum(BACK_SEGMENTS_M),
            "camera",
        )

        self.input_adapter = LightGenImageInputAdapter(self.slm_plane)
        self.input_relay = CoherentImagingRelay(
            self.slm_plane,
            self.encoder_entrance_plane,
            FRONT_MAGNIFICATION,
            name="lightgen_input_4f",
            segments_m=FRONT_SEGMENTS_M,
            matrix_segments_m=FRONT_MATRIX_SEGMENTS_M,
            focal_lengths_m=FRONT_FOCAL_LENGTHS_M,
            clear_aperture_m=FRONT_CLEAR_APERTURE_M,
        )
        self.encoder = FiniteCropPhaseStack(
            layers=ENCODER_LAYERS,
            active_shape=OLS_SHAPE,
            propagation_shape=(self.config.propagation_grid_size,) * 2,
            spacing_m=LAYER_SPACING_M,
        )
        self.ols = LightGenAlignedComplexOLS(self.ols_plane)
        self.generator = FiniteCropPhaseStack(
            layers=GENERATOR_LAYERS,
            active_shape=OLS_SHAPE,
            propagation_shape=(self.config.propagation_grid_size,) * 2,
            spacing_m=LAYER_SPACING_M,
        )
        self.output_relay = CoherentImagingRelay(
            self.generator_exit_plane,
            self.camera_plane,
            BACK_MAGNIFICATION,
            name="lightgen_output_b0",
            segments_m=BACK_SEGMENTS_M,
            focal_lengths_m=(BACK_FOCAL_LENGTH_M,),
            clear_aperture_m=BACK_CLEAR_APERTURE_M,
        )

    @property
    def alignment_metadata(self) -> dict[str, tuple[str, ...]]:
        return {key: tuple(value) for key, value in ALIGNMENT_METADATA.items()}

    @staticmethod
    def _coerce_external_state(z: ComplexOLSState | torch.Tensor) -> ComplexOLSState:
        if isinstance(z, ComplexOLSState):
            return z
        if isinstance(z, torch.Tensor):
            return ComplexOLSState.from_external(z)
        raise TypeError("forward_from_ols requires a ComplexOLSState or complex tensor")

    def w_ols(self, state: ComplexOLSState) -> ComplexField:
        """The single writer shared by real and future generated OLS states."""
        return self.ols.write_to_generator_input(state, self.generator_entrance_plane)

    @staticmethod
    def _validate_relay_diagnostics(relay: CoherentImagingRelay) -> None:
        for name in ("source_pupil_retention", "target_pupil_retention"):
            value = relay.last_diagnostic.get(name)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1.0 + 1.0e-4).any():
                raise ValueError(f"LightGen relay diagnostic is invalid: {name}")

    def _run_generator(
        self,
        state: ComplexOLSState,
        *,
        image_input: LightGenImageOpticalInput | None,
        encoder_output: ComplexField | None,
        ols_input: ComplexField | None,
        encoder_records: tuple[ComplexField, ...],
    ) -> LightGen2DForwardResult:
        generator_input = self.w_ols(state)
        generator_output, generator_records, generator_stats = self.generator(
            generator_input,
            "lightgen_generator",
        )
        generator_output = ComplexField(
            generator_output.tensor,
            self.generator_exit_plane,
            OperationProvenance(
                "lightgen_generator_exit",
                generator_output.plane.plane_id,
                None,
                {"generator_layers": GENERATOR_LAYERS},
            ),
        )
        camera_field = self.output_relay(generator_output)
        self._validate_relay_diagnostics(self.output_relay)
        if not torch.isfinite(camera_field.tensor).all():
            raise ValueError("LightGen raw CMOS field must be finite")
        raw_camera = IntensityFrame(
            camera_field.tensor.abs().square(),
            self.camera_plane,
            camera_field.provenance,
        )
        return LightGen2DForwardResult(
            raw_camera=raw_camera,
            z_ols=state,
            generator_input_field=generator_input,
            generator_output_field=generator_output,
            image_input=image_input,
            encoder_output_field=encoder_output,
            ols_input_field=ols_input,
            encoder_records=encoder_records,
            generator_records=generator_records,
            crop_statistics={
                "generator_crop_retention": generator_stats["crop_retention"],
                "generator_phase": generator_stats["phase"],
            },
        )

    def forward_from_image(self, image: torch.Tensor) -> LightGen2DForwardResult:
        """Execute ``image -> optical encoder -> z_real -> W_OLS -> generator``."""
        image_input = self.input_adapter(image)
        encoder_entrance = self.input_relay(image_input.slm_field)
        self._validate_relay_diagnostics(self.input_relay)
        encoder_output, encoder_records, _ = self.encoder(
            encoder_entrance,
            "lightgen_encoder",
        )
        ols_input = ComplexField(
            encoder_output.tensor,
            self.ols_plane,
            OperationProvenance(
                "lightgen_encoder_to_ols_input_plane",
                encoder_output.plane.plane_id,
                None,
                {"modal_projection_boundary": True},
            ),
        )
        state = self.ols.project_state(ols_input)
        return self._run_generator(
            state,
            image_input=image_input,
            encoder_output=encoder_output,
            ols_input=ols_input,
            encoder_records=encoder_records,
        )

    def forward_from_ols(
        self,
        z: ComplexOLSState | torch.Tensor,
    ) -> LightGen2DForwardResult:
        """Execute ``z_generated -> W_OLS -> same generator -> raw CMOS``."""
        state = self._coerce_external_state(z)
        return self._run_generator(
            state,
            image_input=None,
            encoder_output=None,
            ols_input=None,
            encoder_records=(),
        )

    def forward(self, image: torch.Tensor) -> LightGen2DForwardResult:
        return self.forward_from_image(image)


__all__ = [
    "ALIGNMENT_METADATA",
    "ComplexOLSState",
    "LightGen2DAlignmentRuntimeConfig",
    "LightGen2DForwardResult",
    "LightGen2DSemanticGenerationInputOLSAlignment",
    "LightGenAlignedComplexOLS",
    "LightGenImageInputAdapter",
    "LightGenImageOpticalInput",
    "OLS_GRID_SHAPE",
    "OLS_INTERFACE_ID",
    "OLS_PORT_COUNT",
    "ROW_MAJOR_PORT_ORDER",
]
