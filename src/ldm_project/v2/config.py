"""Strict v2 YAML configuration models."""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import warnings

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormalSystemConfig(StrictModel):
    kind: Literal["formal_amplitude_finite_crop_identity", "diagnostic_amplitude_finite_crop_identity", "diagnostic_amplitude_finite_crop_identity_short_spacing_v1"]
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
        if self.kind in {"diagnostic_amplitude_finite_crop_identity", "diagnostic_amplitude_finite_crop_identity_short_spacing_v1"}:
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


class LightGenSystemConfig(StrictModel):
    """A deliberately isolated architecture proxy, not a formal hardware claim."""

    kind: Literal["diagnostic_lightgen_amplitude_fiber_ols_400"]
    slm_shape: tuple[int, int] = (400, 400)
    slm_pixel_pitch_m: float = 3.0e-6
    metasurface_shape: tuple[int, int] = (400, 400)
    metasurface_pixel_pitch_m: float = 3.0e-6
    propagation_grid_size: int = 512
    encoder_layers: Literal[2] = 2
    decoder_layers: Literal[4] = 4
    fiber_grid_shape: tuple[int, int] = (10, 10)
    fiber_mode_waist_m: float = 30.0e-6
    # These are numerical proxy values only.  The reference contract records them as unknown.
    proxy_wavelength_m: float = 532.0e-9
    proxy_layer_spacing_m: float = 1.0e-3
    ols_kind: Literal["gaussian_single_mode_fiber"] = "gaussian_single_mode_fiber"
    boundary: Literal["finite_crop"] = "finite_crop"
    physical_parameter_status: Literal["unknown"] = "unknown"

    @model_validator(mode="after")
    def diagnostic_contract_is_locked(self) -> "LightGenSystemConfig":
        expected = {
            "slm_shape": (400, 400), "slm_pixel_pitch_m": 3.0e-6,
            "metasurface_shape": (400, 400), "metasurface_pixel_pitch_m": 3.0e-6,
            "encoder_layers": 2, "decoder_layers": 4, "fiber_grid_shape": (10, 10),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"LightGen diagnostic contract locks {name}={value}")
        if self.propagation_grid_size < 400:
            raise ValueError("LightGen propagation grid must cover the 400x400 active plane")
        return self

    @property
    def is_formal(self) -> bool:
        return False


SystemConfig = FormalSystemConfig | LightGenSystemConfig


class TeacherConfig(StrictModel):
    first_stage_checkpoint: str
    diffusion_checkpoint: str
    target_plane: Literal["teacher_camera"] = "teacher_camera"
    guidance_scale: float = 4.0
    seed: int = 1234
    latent_geometry: str = "matched_first_stage"


class PrototypeCalibrationConfig(StrictModel):
    prototype_contract_kind: Literal["formal_prototype_reconstruction_v1", "formal_prototype_shape_guard_v1"] = "formal_prototype_reconstruction_v1"
    epochs: int = Field(default=5, ge=2)
    learning_rate: float = Field(default=5.0e-4, gt=0.0)
    fixed_exposure_scale: float = Field(gt=0.0)
    initialization: Literal["uniform_half", "clean_intensity_to_amplitude"] = "uniform_half"
    optimizer_steps_per_epoch: int = Field(default=1, ge=1)
    prototype_correlation_weight: float = Field(default=0.25, ge=0.0)
    prototype_dice_weight: float = Field(default=0.30, ge=0.0)
    prototype_ce_weight: float = Field(default=0.10, ge=0.0)
    prototype_ce_temperature: float = Field(default=0.07, gt=0.0)
    foreground_weight: float = Field(default=1.0, gt=0.0)
    background_weight: float = Field(default=1.0, gt=0.0)
    roi_threshold: float = Field(default=0.20, gt=0.0, lt=1.0)
    background_energy_weight: float = Field(default=0.0, ge=0.0)
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


class ConditionalGenerationConfig(StrictModel):
    """Teacher-free, full-optical-path conditional generation contract."""

    code_dim: Literal[8] = 8
    hidden_dim: Literal[64] = 64
    command_gain: float = Field(default=0.1, gt=0.0, le=0.5)
    fixed_exposure_scale: float = Field(default=0.4913053810596466, gt=0.0)
    logical_batch_size: int = Field(default=50, ge=50)
    epochs: int = Field(default=5, ge=1)
    generator_learning_rate: float = Field(default=2.0e-4, gt=0.0)
    phase_learning_rate: float = Field(default=5.0e-5, gt=0.0)
    sinkhorn_epsilon: float = Field(default=0.05, gt=0.0)
    sinkhorn_iterations: int = Field(default=20, ge=1)
    samples_per_label: int = Field(default=6000, ge=1)
    shard_size: int = Field(default=1000, ge=1)
    # The identity-anchored v2 line is deliberately reproducible.  Keeping the
    # seed in the scoped config makes it part of the configuration hash rather
    # than an implicit CLI default.
    initialization_seed: Literal[1234] = 1234

    @model_validator(mode="after")
    def conditional_contract_is_explicit(self) -> "ConditionalGenerationConfig":
        if self.logical_batch_size % 10:
            raise ValueError("conditional logical_batch_size must be divisible by ten")
        if self.logical_batch_size // 10 < 5:
            raise ValueError("conditional batches require at least five samples per label")
        if self.samples_per_label % self.shard_size:
            raise ValueError("samples_per_label must divide exactly into shards")
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
    architecture: Literal[
        "legacy_v2_distillation",
        "conditional_global_amplitude_v1",
        "conditional_identity_logit_amplitude_v2",
        "conditional_class_conditioned_modes_v3",
        "conditional_parity_stabilized_modes_v4",
        "conditional_multi_anchor_local_modes_v5",
        "conditional_pca_transport_codesign_v6",
        "conditional_pca_self_imaging_transport_v6_2",
    ] = "legacy_v2_distillation"
    system: SystemConfig
    teacher: TeacherConfig | None = None
    prototype_calibration: PrototypeCalibrationConfig | None = None
    capacity_audit: CapacityAuditConfig | None = None
    student_training: StudentTrainingConfig | None = None
    conditional_generation: ConditionalGenerationConfig | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    run: RunConfig

    @model_validator(mode="after")
    def formal_audit_contract_is_locked(self) -> "V2Config":
        if self.architecture in {
            "conditional_global_amplitude_v1",
            "conditional_identity_logit_amplitude_v2",
            "conditional_class_conditioned_modes_v3",
            "conditional_parity_stabilized_modes_v4",
            "conditional_multi_anchor_local_modes_v5",
            "conditional_pca_transport_codesign_v6",
        }:
            if self.conditional_generation is None:
                raise ValueError("conditional architecture requires conditional_generation")
            forbidden = {
                "teacher": self.teacher,
                "prototype_calibration": self.prototype_calibration,
                "capacity_audit": self.capacity_audit,
                "student_training": self.student_training,
            }
            supplied = [name for name, value in forbidden.items() if value is not None]
            if supplied:
                raise ValueError(f"conditional architecture rejects legacy fields: {', '.join(supplied)}")
            if not self.system.is_formal and self.architecture != "conditional_pca_self_imaging_transport_v6_2":
                raise ValueError("conditional architecture requires the formal identity-OLS system")
            if self.architecture in {"conditional_identity_logit_amplitude_v2", "conditional_class_conditioned_modes_v3", "conditional_parity_stabilized_modes_v4"}:
                if self.conditional_generation.command_gain != 0.5:
                    raise ValueError("identity conditional architectures lock conditional command_gain=0.5")
                if self.conditional_generation.initialization_seed != 1234:
                    raise ValueError("identity conditional architectures lock initialization_seed=1234")
            if self.architecture == "conditional_multi_anchor_local_modes_v5":
                if self.conditional_generation.command_gain != 0.25:
                    raise ValueError("multi-anchor architecture locks conditional command_gain=0.25")
                if self.conditional_generation.initialization_seed != 1234:
                    raise ValueError("multi-anchor architecture locks initialization_seed=1234")
            if self.architecture == "conditional_pca_transport_codesign_v6":
                if self.conditional_generation.command_gain != 0.05:
                    raise ValueError("PCA transport architecture locks conditional command_gain=0.05")
                if self.conditional_generation.initialization_seed != 1234:
                    raise ValueError("PCA transport architecture locks initialization_seed=1234")
            if self.architecture == "conditional_pca_self_imaging_transport_v6_2":
                if self.system.kind != "diagnostic_amplitude_finite_crop_identity_short_spacing_v1":
                    raise ValueError("v6.2 requires the explicit short-spacing diagnostic system")
                if abs(self.system.layer_spacing_m - 0.25e-3) > 1e-15:
                    raise ValueError("v6.2 locks layer_spacing_m=0.25e-3")
                if self.conditional_generation.command_gain != 0.05:
                    raise ValueError("v6.2 locks initial command gain=0.05")
            execution = self.execution
            if (execution.training_microbatch_size, execution.evaluation_microbatch_size, execution.activation_checkpoint) != (1, 1, True):
                raise ValueError("conditional formal execution locks training/evaluation microbatches to one with activation checkpointing")
            return self
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
        if self.prototype_calibration is not None:
            expected_prototype = {
                "epochs": 5,
                "learning_rate": 5.0e-4,
                "initialization": "clean_intensity_to_amplitude",
                "optimizer_steps_per_epoch": 40,
                "prototype_correlation_weight": 0.25,
                "prototype_dice_weight": 0.30,
                "prototype_ce_weight": 0.10,
                "prototype_ce_temperature": 0.07,
            }
            if self.prototype_calibration.prototype_contract_kind == "formal_prototype_shape_guard_v1":
                expected_prototype |= {
                    "foreground_weight": 3.0,
                    "background_weight": 1.0,
                    "roi_threshold": 0.20,
                    "background_energy_weight": 0.10,
                }
            for name, value in expected_prototype.items():
                if getattr(self.prototype_calibration, name) != value:
                    raise ValueError(f"formal prototype locks {name}={value}")
        return self


def _resolve_config_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.is_file():
        return requested
    manifest_candidates = [
        Path("configs/v2/path_migration_manifest.yaml"),
        Path(__file__).resolve().parents[3] / "configs/v2/path_migration_manifest.yaml",
    ]
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for entry in payload.get("entries", []):
            if entry.get("old_path") == requested.as_posix():
                target = Path(str(entry["new_path"]))
                if not target.is_file():
                    raise FileNotFoundError(f"migrated v2 config target is missing: {target}")
                warnings.warn(
                    f"deprecated v2 config path {requested}; use {target}",
                    UserWarning,
                    stacklevel=3,
                )
                return target
        break
    raise FileNotFoundError(f"v2 config does not exist: {requested}")


def load_config(path: str | Path) -> V2Config:
    resolved = _resolve_config_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("v2 configuration must be a YAML mapping")
    return V2Config.model_validate(payload)
