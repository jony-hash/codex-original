"""Strict v2 YAML configuration models."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormalSystemConfig(StrictModel):
    kind: Literal["formal_amplitude_finite_crop_identity", "diagnostic_amplitude_finite_crop_identity"]
    wavelength_m: float = 532.0e-9
    slm_shape: tuple[int, int] = (200, 200)
    slm_pixel_pitch_m: float = 2.5e-6
    metasurface_shape: tuple[int, int] = (100, 100)
    metasurface_pixel_pitch_m: float = 2.0e-6
    propagation_grid_size: int = 2048
    encoder_layers: int = 2
    decoder_layers: int = 2
    layer_spacing_m: float = 1.50e-3
    front_magnification: float = -0.4
    back_magnification: float = -2.5
    front_lens_focal_lengths_m: tuple[float, float] = (223.2142857142857e-3, 89.2857142857143e-3)
    back_lens_focal_length_m: float = 71.4285714285714e-3
    front_clear_aperture_m: float = 23.75e-3
    back_clear_aperture_m: float = 26.60e-3
    ols_kind: Literal["identity"] = "identity"
    boundary: Literal["finite_crop"] = "finite_crop"

    @model_validator(mode="after")
    def formal_geometry_is_locked(self) -> "FormalSystemConfig":
        if self.kind == "diagnostic_amplitude_finite_crop_identity":
            if self.propagation_grid_size < max(self.metasurface_shape):
                raise ValueError("diagnostic propagation grid must cover the active metasurface")
            return self
        expected = {
            "wavelength_m": 532.0e-9,
            "slm_shape": (200, 200),
            "slm_pixel_pitch_m": 2.5e-6,
            "metasurface_shape": (100, 100),
            "metasurface_pixel_pitch_m": 2.0e-6,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "layer_spacing_m": 1.50e-3,
            "front_magnification": -0.4,
            "back_magnification": -2.5,
            "front_lens_focal_lengths_m": (223.2142857142857e-3, 89.2857142857143e-3),
            "back_lens_focal_length_m": 71.4285714285714e-3,
            "front_clear_aperture_m": 23.75e-3,
            "back_clear_aperture_m": 26.60e-3,
        }
        for key, value in expected.items():
            actual = getattr(self, key)
            if isinstance(value, float):
                if abs(actual - value) > max(abs(value) * 1.0e-12, 1.0e-15):
                    raise ValueError(f"formal v2 system locks {key}={value!r}")
            elif actual != value:
                raise ValueError(f"formal v2 system locks {key}={value!r}")
        if self.propagation_grid_size < 100:
            raise ValueError("propagation_grid_size must cover the active metasurface")
        return self

    @property
    def is_formal(self) -> bool:
        return self.kind == "formal_amplitude_finite_crop_identity"


class TeacherConfig(StrictModel):
    first_stage_checkpoint: str
    diffusion_checkpoint: str
    target_plane: Literal["teacher_camera"] = "teacher_camera"
    guidance_scale: float = 4.0
    seed: int = 1234
    latent_geometry: str = "matched_first_stage"


class PrototypeCalibrationConfig(StrictModel):
    epochs: int = Field(default=5, ge=2)
    learning_rate: float = Field(default=5.0e-4, gt=0.0)
    fixed_exposure_scale: float = Field(gt=0.0)
    clean_corr_min: float = 0.70
    clean_dice_min: float = 0.68
    high_saturation_max: float = 0.08


class CapacityAuditConfig(StrictModel):
    dct_modes: int = Field(default=64, ge=8)
    selected_basis_dim: Literal[8] = 8
    max_cycles_per_mm: float = Field(default=20.0, gt=0.0, le=20.0)
    codes_per_label: int = Field(default=100, ge=2)
    seed: int = 1234
    finite_difference_epsilon: float = Field(default=1.0e-3, gt=0.0)
    slm_code_fraction_min: float = 0.25
    camera_code_fraction_min: float = 0.25
    support_energy_min: float = 0.95
    clipping_max: float = 0.08
    command_gain: float = 0.1


class StudentTrainingConfig(StrictModel):
    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    code_dim: Literal[8] = 8
    batch_size: int = Field(default=20, ge=20)
    prototype_fidelity_weight: float = 1.0
    paired_teacher_residual_weight: float = 1.0
    code_transport_weight: float = 0.01
    clipping_weight: float = 0.1
    saturation_weight: float = 0.1
    phase_tv_weight: float = 1.0e-5
    paired_teacher_corr_min: float = 0.70
    paired_teacher_dice_min: float = 0.68
    bootstrap_seed: Literal[1234] = 1234
    bootstrap_resamples: Literal[10000] = 10000

    @model_validator(mode="after")
    def training_contract_is_explicit(self) -> "StudentTrainingConfig":
        if self.batch_size % 10:
            raise ValueError("student batch_size must be divisible by ten")
        expected = {
            "prototype_fidelity_weight": 1.0,
            "paired_teacher_residual_weight": 1.0,
            "code_transport_weight": 0.01,
            "clipping_weight": 0.1,
            "saturation_weight": 0.1,
            "phase_tv_weight": 1.0e-5,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"v2 training locks {name}={value}")
        return self


class ExecutionConfig(StrictModel):
    """Memory-safe physical execution controls; they never alter the optics."""

    prototype_microbatch_size: int = Field(default=1, ge=1)
    capacity_microbatch_size: int = Field(default=1, ge=1)
    training_microbatch_size: int = Field(default=1, ge=1)
    evaluation_microbatch_size: int = Field(default=1, ge=1)
    activation_checkpoint: bool = True


class RunConfig(StrictModel):
    run_name: str
    output_root: str = "outputs"
    evidence_tier: Literal["diagnostic", "acceptance"] = "diagnostic"
    device: str = "cuda"


class V2Config(StrictModel):
    schema_version: Literal[2]
    system: FormalSystemConfig
    teacher: TeacherConfig | None = None
    prototype_calibration: PrototypeCalibrationConfig | None = None
    capacity_audit: CapacityAuditConfig | None = None
    student_training: StudentTrainingConfig | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    run: RunConfig

    @model_validator(mode="after")
    def formal_audit_contract_is_locked(self) -> "V2Config":
        if not self.system.is_formal or self.capacity_audit is None:
            return self
        expected = {
            "dct_modes": 64,
            "selected_basis_dim": 8,
            "max_cycles_per_mm": 20.0,
            "codes_per_label": 100,
            "seed": 1234,
            "finite_difference_epsilon": 1.0e-3,
            "command_gain": 0.1,
        }
        for name, value in expected.items():
            if getattr(self.capacity_audit, name) != value:
                raise ValueError(f"formal capacity locks {name}={value}")
        execution = self.execution
        if (
            execution.prototype_microbatch_size,
            execution.capacity_microbatch_size,
            execution.training_microbatch_size,
            execution.evaluation_microbatch_size,
            execution.activation_checkpoint,
        ) != (1, 1, 1, 1, True):
            raise ValueError("formal execution locks microbatches to one with activation checkpointing")
        return self


def load_config(path: str | Path) -> V2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("v2 configuration must be a YAML mapping")
    return V2Config.model_validate(payload)
