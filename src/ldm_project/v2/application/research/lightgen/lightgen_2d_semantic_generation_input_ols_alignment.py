"""LightGen-aligned 2-D semantic-generation input and complex-OLS runtime.

This module is an implementation of the definition-only alignment contract.
It is a research runtime boundary, not a training entry point or a scientific
generation-acceptance path.  The authoritative latent is the complex 100-port
state.  Both an optically encoded state and an externally supplied state pass
through the same coefficient-space transport and modal re-emission writer.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import GaussianFiberArrayOLS, GaussianFiberGeometry
from ldm_project.v2.contracts import AmplitudeCommand, ComplexField, IntensityFrame, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.devices import IdealAmplitudeSLM
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.optics.relays import CoherentImagingRelay
from ldm_project.v2.application.research.lightgen.material_nonlinearity import MaterialNonlinearity, MaterialParameters


OLS_GRID_SHAPE = (10, 10)
OLS_PORT_COUNT = 100
ROW_MAJOR_PORT_ORDER = tuple(
    (row, column)
    for row in range(OLS_GRID_SHAPE[0])
    for column in range(OLS_GRID_SHAPE[1])
)
OLS_INTERFACE_ID = "lightgen_10x10_complex_ols_v1"
LP01_OLS_MODEL = "lp01_ports_v1"
GAUSSIAN_OLS_MODEL = "gaussian_s12_v1"
OLS_MODELS = (LP01_OLS_MODEL, GAUSSIAN_OLS_MODEL)

WAVELENGTH_M = 532.0e-9
SLM_SHAPE = (200, 200)
SLM_PIXEL_PITCH_M = 2.5e-6
OLS_SHAPE = (100, 100)
OLS_PIXEL_PITCH_M = 2.0e-6
LAYER_SPACING_M = 1.50e-3
ENCODER_LAYERS = 2
GENERATOR_LAYERS = 2


def _field_power_w(field: ComplexField) -> torch.Tensor:
    """Integrate a calibrated field over its declared physical pixel area."""
    if not torch.is_complex(field.tensor) or not torch.isfinite(field.tensor).all():
        raise ValueError("optical power requires a finite complex field")
    area = field.plane.pixel_pitch_m[0] * field.plane.pixel_pitch_m[1]
    return field.tensor.abs().square().reshape(field.tensor.shape[0], -1).sum(dim=1) * area

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
        "selected LP01 or Gaussian S12 mode bank, coefficient-space T=I_100, and two-plus-two phase stacks",
        "the generator exit is coupled to the raw CMOS plane through the existing project B=0 relay",
    ),
}


@dataclass(frozen=True)
class LightGenPhysicalOpticsConfig:
    """Explicit, simulation-only SI configuration for the LightGen research path."""

    input_power_w: float
    layer_power_transmission: float = 1.0
    global_fiber_geometry: bool = False
    fiber_pitch_m: float = 125.0e-6
    fiber_cladding_diameter_m: float = 125.0e-6
    fiber_mfd_m: float = 14.222e-6
    fiber_length_m: float = 8.5e-3
    fiber_attenuation_db_per_km: float = 0.0
    material: MaterialParameters | None = None
    camera_gain_w_inverse: float | None = None

    def __post_init__(self) -> None:
        positive = (self.input_power_w, self.fiber_pitch_m, self.fiber_cladding_diameter_m,
                    self.fiber_mfd_m, self.fiber_length_m)
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("physical optical powers and geometry must be finite positive SI values")
        if not math.isfinite(self.layer_power_transmission) or not 0.0 <= self.layer_power_transmission <= 1.0:
            raise ValueError("layer_power_transmission must be within [0,1]")
        if not math.isfinite(self.fiber_attenuation_db_per_km) or self.fiber_attenuation_db_per_km < 0.0:
            raise ValueError("fiber attenuation must be finite nonnegative dB/km")
        if self.camera_gain_w_inverse is not None and (
            not math.isfinite(self.camera_gain_w_inverse) or self.camera_gain_w_inverse <= 0.0
        ):
            raise ValueError("camera_gain_w_inverse must be finite positive when configured")
        if self.material is not None and self.material.model not in {"identity", "saturable_absorption"}:
            raise ValueError("the integrated material supports identity or saturable_absorption")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LightGen2DAlignmentRuntimeConfig:
    """Numerical runtime knobs frozen by the aligned project baseline."""

    propagation_grid_size: int = 2048
    ols_model: str = LP01_OLS_MODEL
    physical_optics: LightGenPhysicalOpticsConfig | None = None

    def __post_init__(self) -> None:
        if self.propagation_grid_size < max(OLS_SHAPE):
            raise ValueError("propagation_grid_size must cover the 100x100 active plane")
        if self.ols_model not in OLS_MODELS:
            raise ValueError(f"ols_model must be one of {OLS_MODELS}")


@dataclass(frozen=True)
class ComplexOLSState:
    """Canonical batch complex state at the 10x10 single-mode OLS interface."""

    coefficients: torch.Tensor
    port_order: tuple[tuple[int, int], ...] = ROW_MAJOR_PORT_ORDER
    interface_id: str = OLS_INTERFACE_ID
    source: str = "external"
    basis_id: str = LP01_OLS_MODEL
    mode_geometry: Mapping[str, Any] | None = None

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
        if self.basis_id not in OLS_MODELS:
            raise ValueError("ComplexOLSState basis_id is unsupported")
        if self.basis_id == GAUSSIAN_OLS_MODEL and self.mode_geometry is None:
            raise ValueError("Gaussian ComplexOLSState requires explicit mode geometry")
        if self.mode_geometry is not None:
            if not isinstance(self.mode_geometry, Mapping) or self.mode_geometry.get("basis_id") != self.basis_id:
                raise ValueError("ComplexOLSState mode geometry does not match its basis_id")
            object.__setattr__(self, "mode_geometry", copy.deepcopy(dict(self.mode_geometry)))

    @classmethod
    def from_external(
        cls,
        coefficients: torch.Tensor,
        *,
        basis_id: str = LP01_OLS_MODEL,
        mode_geometry: Mapping[str, Any] | None = None,
    ) -> "ComplexOLSState":
        """Bind generated coefficients to an explicit modal basis."""
        return cls(coefficients=coefficients, source="external_generated", basis_id=basis_id, mode_geometry=mode_geometry)

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

    def __init__(
        self,
        ols_plane: PlaneSpec,
        *,
        ols_model: str = LP01_OLS_MODEL,
        output_plane: PlaneSpec | None = None,
        physical_optics: LightGenPhysicalOpticsConfig | None = None,
    ) -> None:
        super().__init__()
        if ols_model not in OLS_MODELS:
            raise ValueError(f"ols_model must be one of {OLS_MODELS}")
        self.ols_model = ols_model
        self.physical_optics = physical_optics
        if ols_model == GAUSSIAN_OLS_MODEL:
            if physical_optics is not None and physical_optics.global_fiber_geometry:
                raise ValueError("global array integration currently uses the existing LP01 overlap model")
            self.operator = GaussianFiberArrayOLS(ols_plane, GaussianFiberGeometry())
            self.output_plane = ols_plane
        else:
            identity_transport = torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)
            if physical_optics is not None and physical_optics.global_fiber_geometry:
                if output_plane is None:
                    raise ValueError("global fiber geometry requires an explicit decoder output plane")
                input_spec = FiberArraySpec(
                    grid_shape=OLS_GRID_SHAPE,
                    pitch_m=OLS_PIXEL_PITCH_M,
                    wavelength_m=WAVELENGTH_M,
                    waist_m=physical_optics.fiber_mfd_m / 2.0,
                    fill=0.8,
                    v_number=2.2,
                    mode_model="step_index_lp01_ports",
                    expected_fov_m=ols_plane.fov_m,
                    array_center_pitch_m=physical_optics.fiber_pitch_m,
                )
                output_spec = FiberArraySpec(
                    grid_shape=OLS_GRID_SHAPE,
                    pitch_m=OLS_PIXEL_PITCH_M,
                    wavelength_m=WAVELENGTH_M,
                    waist_m=physical_optics.fiber_mfd_m / 2.0,
                    fill=0.8,
                    v_number=2.2,
                    mode_model="step_index_lp01_ports",
                    expected_fov_m=output_plane.fov_m,
                )
                self.operator = FiberArrayOLS(
                    ols_plane,
                    input_spec,
                    transport=identity_transport,
                    output_plane=output_plane,
                    output_spec=output_spec,
                )
                self.output_plane = output_plane
            else:
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
                self.output_plane = ols_plane

    @property
    def basis_id(self) -> str:
        return self.ols_model

    def mode_identity(self) -> dict[str, Any]:
        plane = self.operator.input_plane
        identity: dict[str, Any] = {
            "basis_id": self.basis_id,
            "wavelength_m": plane.wavelength_m,
            "array_shape": list(OLS_GRID_SHAPE),
            "port_count": OLS_PORT_COUNT,
            "port_order": "row_major_j=10*r+c",
            "plane_shape": list(plane.shape),
            "pixel_pitch_m": list(plane.pixel_pitch_m),
            "fov_m": list(plane.fov_m),
            "plane_origin_m": list(plane.origin_m),
            "plane_z_m": plane.z_m,
            "transport": "T=I_100 coefficient space",
            "parameter_status": "PROJECT_GEOMETRY_UNMEASURED",
            "output_mode_plane": self.output_plane.to_dict(),
        }
        if self.ols_model == GAUSSIAN_OLS_MODEL:
            geometry = self.operator.geometry
            identity.update({
                "mode_model": "normalized_gaussian_fundamental_mode_S12",
                "port_center_pitch_m": geometry.port_pitch_m,
                "gaussian_field_1e_radius_m": geometry.waist_m,
                "gaussian_mode_field_diameter_m": 2.0 * geometry.waist_m,
                "normalization": "sqrt(2/pi)/w * exp(-r^2/w^2); analytic infinite-plane unit power",
                "per_port_aperture": False,
                "discrete_renormalization": False,
                "orthogonalization": False,
            })
        else:
            spec = self.operator.spec
            identity.update({
                "mode_model": spec.mode_model,
                "port_center_pitch_m": spec.array_center_pitch_m or spec.expected_fov_m[0] * spec.fill / (OLS_GRID_SHAPE[0] - 1),
                "lp01_waist_m": spec.waist_m,
                "fill": spec.fill,
                "v_number": spec.v_number,
                "normalization": "discrete unit-power LP01 profile with per-port rectangular cells",
                "per_port_aperture": True,
                "discrete_renormalization": True,
                "orthogonalization": spec.mode_model == "step_index_lp01_lowdin",
            })
            if spec.array_center_pitch_m is not None:
                center_y = [
                    float((index - 4.5) * spec.array_center_pitch_m + plane.origin_m[0])
                    for index in range(10)
                ]
                center_x = [
                    float((index - 4.5) * spec.array_center_pitch_m + plane.origin_m[1])
                    for index in range(10)
                ]
                identity["fiber_center_x_m_row_major"] = [value for _row in range(10) for value in center_x]
                identity["fiber_center_y_m_row_major"] = [value for value in center_y for _column in range(10)]
                identity["fiber_center_span_m"] = 9.0 * spec.array_center_pitch_m
                identity["array_support_definition"] = "fiber pitch Voronoi cells; cladding diameter is an independent geometry field"
        if self.physical_optics is not None:
            identity["physical_fiber_geometry"] = {
                "pitch_m": self.physical_optics.fiber_pitch_m,
                "cladding_diameter_m": self.physical_optics.fiber_cladding_diameter_m,
                "mfd_m": self.physical_optics.fiber_mfd_m,
                "length_m": self.physical_optics.fiber_length_m,
                "attenuation_db_per_km": self.physical_optics.fiber_attenuation_db_per_km,
                "parameter_status": "reference_scenario_not_current_device_measurement",
            }
        return identity

    @property
    def transport(self) -> torch.Tensor:
        return self.operator.transport.to(torch.complex64)

    def project_state(self, field: ComplexField) -> ComplexOLSState:
        coefficients = self.operator.project_coefficients(field)
        return ComplexOLSState(coefficients, source="real_encoded", basis_id=self.basis_id, mode_geometry=self.mode_identity())

    def validate_state_basis(self, state: ComplexOLSState) -> None:
        if state.basis_id != self.basis_id:
            raise ValueError(f"OLS state basis {state.basis_id} cannot be used by {self.basis_id} writer")
        if state.mode_geometry is not None and state.mode_geometry != self.mode_identity():
            raise ValueError("OLS state geometry differs from the selected mode geometry")
        if self.basis_id == GAUSSIAN_OLS_MODEL and state.mode_geometry is None:
            raise ValueError("Gaussian OLS state must carry the exact selected mode geometry")

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
        self.validate_state_basis(state)
        self._validate_generator_plane(generator_plane, self.output_plane)
        transmitted = self.operator.transmit_coefficients(state.coefficients)
        return self.write_transmitted_coefficients_to_generator_input(transmitted, generator_plane)

    def write_transmitted_coefficients_to_generator_input(
        self, coefficients: torch.Tensor, generator_plane: PlaneSpec
    ) -> ComplexField:
        """Write post-transport coefficients through the selected output modes."""
        if coefficients.ndim != 2 or coefficients.shape[1] != OLS_PORT_COUNT or not torch.is_complex(coefficients):
            raise ValueError("post-transport OLS coefficients must be complex [batch,100]")
        if not torch.isfinite(coefficients).all():
            raise ValueError("post-transport OLS coefficients must be finite")
        self._validate_generator_plane(generator_plane, self.output_plane)
        reemitted = self.operator.reemit_coefficients(coefficients).unsqueeze(1)
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
                    "basis_id": self.basis_id,
                    "mode_geometry": self.mode_identity(),
                    "physical_fiber_attenuation_db_per_km": (
                        None if self.physical_optics is None else self.physical_optics.fiber_attenuation_db_per_km
                    ),
                    "material_model": (
                        None if self.physical_optics is None or self.physical_optics.material is None
                        else self.physical_optics.material.model
                    ),
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
    physical_power_trace: dict[str, torch.Tensor] | None = None
    physical_sensor_output_w_per_pixel: torch.Tensor | None = None
    task_sensor_output: torch.Tensor | None = None
    task_sensor_camera_gain_w_inverse: float | None = None

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
        self.physical_optics = self.config.physical_optics
        self.global_fiber_geometry = bool(
            self.physical_optics is not None and self.physical_optics.global_fiber_geometry
        )
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
        ols_z = encoder_z + ENCODER_LAYERS * LAYER_SPACING_M
        self.ols_plane = self.encoder_entrance_plane.with_plane(
            "lightgen_ols_input",
            shape=(self.config.propagation_grid_size, self.config.propagation_grid_size)
            if self.global_fiber_geometry
            else OLS_SHAPE,
            z_m=ols_z,
            coordinate_frame="fiber_global" if self.global_fiber_geometry else "ols",
        )
        self.generator_entrance_plane = self.encoder_entrance_plane.with_plane(
            "lightgen_generator_entrance",
            z_m=ols_z,
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
            layer_power_transmission=(self.physical_optics.layer_power_transmission if self.physical_optics else 1.0),
            preserve_global_output=self.global_fiber_geometry,
        )
        self.ols = LightGenAlignedComplexOLS(
            self.ols_plane,
            ols_model=self.config.ols_model,
            output_plane=self.generator_entrance_plane,
            physical_optics=self.physical_optics,
        )
        self.generator = FiniteCropPhaseStack(
            layers=GENERATOR_LAYERS,
            active_shape=OLS_SHAPE,
            propagation_shape=(self.config.propagation_grid_size,) * 2,
            spacing_m=LAYER_SPACING_M,
            layer_power_transmission=(self.physical_optics.layer_power_transmission if self.physical_optics else 1.0),
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
        metadata = {key: tuple(value) for key, value in ALIGNMENT_METADATA.items()}
        metadata["PROJECT_DECISION"] += (f"selected OLS model: {self.ols.basis_id}",)
        return metadata

    def _coerce_external_state(self, z: ComplexOLSState | torch.Tensor) -> ComplexOLSState:
        if isinstance(z, ComplexOLSState):
            return z
        if isinstance(z, torch.Tensor):
            if self.ols.basis_id == GAUSSIAN_OLS_MODEL:
                raise ValueError("Gaussian OLS requires an external ComplexOLSState with explicit basis and geometry")
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
        encoder_statistics: dict[str, torch.Tensor] | None = None,
        physical_power_trace: dict[str, torch.Tensor] | None = None,
    ) -> LightGen2DForwardResult:
        if self.physical_optics is None:
            generator_input = self.w_ols(state)
        else:
            if ols_input is None or physical_power_trace is None:
                raise ValueError("absolute-power execution requires the image-to-fiber physical path")
            config = self.physical_optics
            transported = self.ols.operator.transmit_coefficients(state.coefficients)
            coupled_power = transported.abs().square()
            before_fiber = physical_power_trace["P_before_fiber_W"]
            if torch.any(coupled_power.sum(dim=1) > before_fiber * (1.0 + 1.0e-5) + 1.0e-12):
                raise ValueError("complex fiber overlaps exceed incident power")
            safe_before = before_fiber.clamp_min(1.0e-30)
            coupling_each = coupled_power / safe_before[:, None]
            attenuation_db = config.fiber_attenuation_db_per_km * config.fiber_length_m / 1000.0
            fiber_power_transmission = 10.0 ** (-attenuation_db / 10.0)
            fiber_output_coefficients = transported * math.sqrt(fiber_power_transmission)
            fiber_output_power = fiber_output_coefficients.abs().square()
            waist_m = config.fiber_mfd_m / 2.0
            intensity_peak_w_m2 = 2.0 * fiber_output_power / (math.pi * waist_m**2)
            intensity_peak_w_cm2 = intensity_peak_w_m2 / 1.0e4
            material_coefficients = (
                fiber_output_coefficients
                if config.material is None
                else MaterialNonlinearity().forward(
                    fiber_output_coefficients,
                    intensity_peak_w_cm2,
                    config.material,
                )
            )
            material_power = material_coefficients.abs().square()
            generator_input = self.ols.write_transmitted_coefficients_to_generator_input(
                material_coefficients, self.generator_entrance_plane
            )
            reemitted_power = _field_power_w(generator_input)
            if torch.any(reemitted_power > material_power.sum(dim=1) * (1.0 + 1.0e-5) + 1.0e-12):
                raise ValueError("fiber modal re-emission exceeds post-material channel power")
            physical_power_trace.update({
                "fiber_coupled_power_W": coupled_power,
                "fiber_coupling_efficiency_each": coupling_each,
                "fiber_coupling_efficiency_total": coupling_each.sum(dim=1),
                "P_fiber_total_W": coupled_power.sum(dim=1),
                "fiber_attenuation_dB": torch.full_like(before_fiber, attenuation_db),
                "fiber_power_transmission": torch.full_like(before_fiber, fiber_power_transmission),
                "fiber_output_power_W": fiber_output_power,
                "P_fiber_output_total_W": fiber_output_power.sum(dim=1),
                "I_peak_W_m2": intensity_peak_w_m2,
                "I_peak_W_cm2": intensity_peak_w_cm2,
                "nonlinear_activation_coverage_input_W_cm2": intensity_peak_w_cm2,
                "P_after_material_each_W": material_power,
                "P_after_material_W": material_power.sum(dim=1),
                "P_generator_entry_W": reemitted_power,
            })
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
        physical_sensor_output_w_per_pixel: torch.Tensor | None = None
        task_sensor_output: torch.Tensor | None = None
        camera_gain_w_inverse: float | None = None
        if physical_power_trace is not None:
            decoder_layer_power = generator_stats["layer_power"]
            camera_area = self.camera_plane.pixel_pitch_m[0] * self.camera_plane.pixel_pitch_m[1]
            physical_sensor_output_w_per_pixel = raw_camera.tensor * camera_area
            camera_gain_w_inverse = self.physical_optics.camera_gain_w_inverse
            if camera_gain_w_inverse is not None:
                task_sensor_output = (physical_sensor_output_w_per_pixel * camera_gain_w_inverse).clamp(0.0, 1.0)
            physical_power_trace.update({
                "P_decoder_1_W": decoder_layer_power[:, 0, 3],
                "P_decoder_2_W": decoder_layer_power[:, 1, 3],
                "P_camera_W": raw_camera.tensor.reshape(raw_camera.tensor.shape[0], -1).sum(dim=1) * camera_area,
                "P_camera_from_sensor_pixels_W": physical_sensor_output_w_per_pixel.reshape(
                    physical_sensor_output_w_per_pixel.shape[0], -1
                ).sum(dim=1),
                "layer_power_transmission": torch.full_like(
                    physical_power_trace["P_before_fiber_W"], self.physical_optics.layer_power_transmission
                ),
                "fiber_mfd_m": torch.full_like(physical_power_trace["P_before_fiber_W"], self.physical_optics.fiber_mfd_m),
                "fiber_pitch_m": torch.full_like(physical_power_trace["P_before_fiber_W"], self.physical_optics.fiber_pitch_m),
            })
        crop_statistics = {
            "generator_crop_retention": generator_stats["crop_retention"],
            "generator_phase": generator_stats["phase"],
            "generator_layer_power": generator_stats["layer_power"],
        }
        if encoder_statistics is not None:
            crop_statistics["encoder_layer_power"] = encoder_statistics["layer_power"]
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
            crop_statistics=crop_statistics,
            physical_power_trace=physical_power_trace,
            physical_sensor_output_w_per_pixel=physical_sensor_output_w_per_pixel,
            task_sensor_output=task_sensor_output,
            task_sensor_camera_gain_w_inverse=camera_gain_w_inverse,
        )

    def forward_from_image(self, image: torch.Tensor) -> LightGen2DForwardResult:
        """Execute ``image -> optical encoder -> z_real -> W_OLS -> generator``."""
        image_input = self.input_adapter(image)
        physical_power_trace: dict[str, torch.Tensor] | None = None
        if self.physical_optics is not None:
            pixel_area = self.slm_plane.pixel_pitch_m[0] * self.slm_plane.pixel_pitch_m[1]
            amplitude = math.sqrt(
                self.physical_optics.input_power_w
                / (self.slm_plane.shape[0] * self.slm_plane.shape[1] * pixel_area)
            )
            illumination_real = torch.full_like(image_input.slm_field.tensor.real, amplitude)
            illumination = torch.complex(illumination_real, torch.zeros_like(illumination_real))
            source_field = ComplexField(
                illumination,
                self.slm_plane,
                OperationProvenance(
                    "lightgen_absolute_input_power_calibration",
                    self.slm_plane.plane_id,
                    None,
                    {
                        "input_power_w": self.physical_optics.input_power_w,
                        "source_profile": "uniform_plane_wave_over_declared_SLM_FOV",
                        "calibration_boundary": "single_optical_path_inlet_before_amplitude_SLM",
                        "field_unit": "sqrt(W)/m",
                    },
                ),
            )
            calibrated_slm_field = ComplexField(
                illumination * image_input.slm_field.tensor,
                self.slm_plane,
                OperationProvenance(
                    "lightgen_absolute_source_times_amplitude_slm_command",
                    self.slm_plane.plane_id,
                    None,
                    {
                        "input_power_w": self.physical_optics.input_power_w,
                        "downstream_renormalization": False,
                    },
                ),
            )
            physical_power_trace = {
                "P_input_W": _field_power_w(source_field),
                "P_after_slm_W": _field_power_w(calibrated_slm_field),
            }
            tolerance = max(1.0e-12, self.physical_optics.input_power_w * 1.0e-6)
            if torch.any((physical_power_trace["P_input_W"] - self.physical_optics.input_power_w).abs() > tolerance):
                raise ValueError("single inlet calibration did not produce configured input_power_w")
            image_input = replace(image_input, slm_field=calibrated_slm_field)
        encoder_entrance = self.input_relay(image_input.slm_field)
        self._validate_relay_diagnostics(self.input_relay)
        encoder_output, encoder_records, encoder_statistics = self.encoder(
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
        if physical_power_trace is not None:
            encoder_layer_power = encoder_statistics["layer_power"]
            physical_power_trace.update({
                "P_encoder_entrance_W": _field_power_w(encoder_entrance),
                "P_encoder_1_W": encoder_layer_power[:, 0, 3],
                "P_encoder_2_W": encoder_layer_power[:, 1, 3],
                "P_before_fiber_W": _field_power_w(ols_input),
                "encoder_layer_input_power_W": encoder_layer_power[:, :, 0],
                "encoder_after_device_power_W": encoder_layer_power[:, :, 1],
                "encoder_after_propagation_power_W": encoder_layer_power[:, :, 2],
                "encoder_after_aperture_power_W": encoder_layer_power[:, :, 3],
            })
        return self._run_generator(
            state,
            image_input=image_input,
            encoder_output=encoder_output,
            ols_input=ols_input,
            encoder_records=encoder_records,
            encoder_statistics=encoder_statistics,
            physical_power_trace=physical_power_trace,
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
    "LightGenPhysicalOpticsConfig",
    "LightGen2DForwardResult",
    "LightGen2DSemanticGenerationInputOLSAlignment",
    "LightGenAlignedComplexOLS",
    "LightGenImageInputAdapter",
    "LightGenImageOpticalInput",
    "OLS_GRID_SHAPE",
    "OLS_INTERFACE_ID",
    "LP01_OLS_MODEL",
    "GAUSSIAN_OLS_MODEL",
    "OLS_MODELS",
    "OLS_PORT_COUNT",
    "ROW_MAJOR_PORT_ORDER",
]
