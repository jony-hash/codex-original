"""Strict v2 YAML configuration models."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormalSystemConfig(StrictModel):
    kind: Literal["formal_amplitude_finite_crop_identity"]
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
    ols_kind: Literal["identity"] = "identity"
    boundary: Literal["finite_crop"] = "finite_crop"

    @model_validator(mode="after")
    def formal_geometry_is_locked(self) -> "FormalSystemConfig":
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


class TeacherConfig(StrictModel):
    first_stage_checkpoint: str
    diffusion_checkpoint: str
    target_plane: Literal["teacher_camera"] = "teacher_camera"
    guidance_scale: float = 4.0
    seed: int = 1234
    latent_geometry: str = "matched_first_stage"


class PrototypeCalibrationConfig(StrictModel):
    epochs: int = Field(default=5, ge=1)
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


class StudentTrainingConfig(StrictModel):
    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    code_dim: int = Field(default=8, ge=1)
    paired_teacher_corr_min: float = 0.70
    paired_teacher_dice_min: float = 0.68
    bootstrap_seed: Literal[1234] = 1234
    bootstrap_resamples: Literal[10000] = 10000


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
    run: RunConfig


def load_config(path: str | Path) -> V2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("v2 configuration must be a YAML mapping")
    return V2Config.model_validate(payload)
