from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from ..config import (
    OneShotConfig,
    PhysicsConfig,
    _ONE_SHOT_COMMAND_TRANSFORMS,
    _ONE_SHOT_DECODER_DIAGNOSTIC_MODES,
    _ONE_SHOT_DECODER_MASK_COMPONENTS,
    _ONE_SHOT_DECODER_MODULATION_OVERRIDES,
    _ONE_SHOT_GENERATOR_ARCHES,
    _ONE_SHOT_OLS_CALIBRATIONS,
    _ONE_SHOT_SLM_OVERRIDE_MODES,
    _ONE_SHOT_TARGET_CODE_BRIDGE_ARCHES,
    _ONE_SHOT_TARGET_CODE_BRIDGE_SOURCES,
    _ONE_SHOT_TARGET_CODE_MODES,
    _ONE_SHOT_TEMPLATE_INITIALIZERS,
    _expected_d2nn_modulation,
    _target_code_bridge_mode_is_compatible,
)
from .layers import (
    ElectronicGeneratorE,
    IdealAmplitudeSLMModel,
    PhaseAmplitudeDiffractiveStack,
    PhaseOnlyDiffractiveStack,
    _apply_complex_ols,
    _angular_spectrum_propagate_center_padded_with_diagnostics,
    _flatten_complex_field,
    _resize_complex_field,
    _resize_real_intensity,
    _vector_to_complex_field,
)
from ..optics.sampling import PlaneSampling, sample_center_aperture
from ..optics import BackOpticalPath, FrontOpticalPath, normalize_real_intensity
from ..optics.fields import build_optical_field_record


ONE_SHOT_MODEL_VERSION = "ldm_distilled_single_shot_optical_student_v3"


class _BridgeResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _MultiScaleBridgeAdapter(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
        )
        self.wide = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.project = nn.Conv2d(3 * hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(
            torch.cat([self.local(x), self.dilated(x), self.wide(x)], dim=1)
        )


class OneShotOpticalGenerator(nn.Module):
    def __init__(
        self,
        physics: PhysicsConfig,
        config: OneShotConfig | None = None,
        image_size: int = 200,
    ) -> None:
        super().__init__()
        self.physics = physics
        self.config = config or OneShotConfig()
        self.image_size = image_size
        self.uses_diffractive_core = self.config.optical_core_arch in {
            "diffractive_phase_only",
            "phase_amplitude_d2nn",
        }
        self.uses_direct_camera_core = (
            self.config.optical_core_arch == "camera_direct_clean_oracle"
        )
        self._validate_config()
        self.generator_e = ElectronicGeneratorE(
            image_size=image_size,
            grid_size=self.config.metasurface_grid_size,
            code_dim=self.config.code_dim,
            num_classes=self.config.num_classes,
            label_embed_dim=self.config.label_embed_dim,
            hidden_dim=self.config.hidden_dim,
            arch=self.config.generator_e_arch,
            code_residual_scale=self.config.code_residual_scale,
            command_transform=self.config.generator_command_transform,
            background_floor=self.config.slm_background_floor,
            template_init=self.config.generator_template_init,
        )
        self.slm = IdealAmplitudeSLMModel(
            self.config.slm_background_floor,
            self.config.slm_residual_phase_scale_rad,
        )
        d2nn = self._resolve_d2nn_params()
        if self.uses_diffractive_core:
            encoder_modulation = d2nn["modulation"]
            decoder_modulation = self._decoder_modulation(d2nn)
            encoder_kwargs = (
                {"amplitude_logit_init": d2nn["amplitude_logit_init"]}
                if encoder_modulation == "phase_amplitude"
                else {}
            )
            decoder_kwargs = (
                {"amplitude_logit_init": d2nn["amplitude_logit_init"]}
                if decoder_modulation == "phase_amplitude"
                else {}
            )
            encoder_stack_cls = (
                PhaseAmplitudeDiffractiveStack
                if encoder_modulation == "phase_amplitude"
                else PhaseOnlyDiffractiveStack
            )
            decoder_stack_cls = (
                PhaseAmplitudeDiffractiveStack
                if decoder_modulation == "phase_amplitude"
                else PhaseOnlyDiffractiveStack
            )
            self.encoder = encoder_stack_cls(
                d2nn["encoder_layers"],
                d2nn["grid_size"],
                d2nn["layer_spacing"],
                d2nn["wavelength"],
                d2nn["pixel_size"],
                d2nn["propagation_grid_size"],
                d2nn["phase_init"],
                **encoder_kwargs,
            )
            self.decoder = decoder_stack_cls(
                d2nn["decoder_layers"],
                d2nn["grid_size"],
                d2nn["layer_spacing"],
                d2nn["wavelength"],
                d2nn["pixel_size"],
                d2nn["propagation_grid_size"],
                d2nn["phase_init"],
                **decoder_kwargs,
            )
        elif self.uses_direct_camera_core:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        else:
            grid_features = (
                2
                * self.config.metasurface_grid_size
                * self.config.metasurface_grid_size
            )
            self.encoder = nn.Linear(grid_features, grid_features, bias=False)
            self.decoder = nn.Linear(grid_features, grid_features, bias=False)
        self._d2nn = d2nn
        self.front_path = FrontOpticalPath(physics)
        self.back_path = BackOpticalPath(physics, propagation_method="angular_spectrum")
        calibration_grid = (
            d2nn["grid_size"]
            if self.uses_diffractive_core
            else self.config.metasurface_grid_size
        )
        if self.config.ols_transfer_calibration in {
            "phase_screen",
            "amplitude_phase_screen",
        }:
            self.ols_phase_screen = nn.Parameter(
                torch.zeros(1, 1, calibration_grid, calibration_grid)
            )
        else:
            self.register_parameter("ols_phase_screen", None)
        if self.config.ols_transfer_calibration == "amplitude_phase_screen":
            self.ols_amplitude_log = nn.Parameter(
                torch.zeros(1, 1, calibration_grid, calibration_grid)
            )
        else:
            self.register_parameter("ols_amplitude_log", None)
        if self.config.target_code_bridge_source != "none":
            hidden_channels = int(self.config.target_code_bridge_hidden_channels)
            self.target_code_bridge_adapter = self._build_target_code_bridge_adapter(
                self.config.target_code_bridge_arch,
                hidden_channels,
            )
        else:
            self.target_code_bridge_adapter = None
        self._initialize_layers()

    def _build_target_code_bridge_adapter(
        self, arch: str, hidden_channels: int
    ) -> nn.Module:
        if arch == "shallow":
            return nn.Sequential(
                nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            )
        if arch == "deep_residual":
            return nn.Sequential(
                nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                _BridgeResidualBlock(hidden_channels),
                nn.SiLU(),
                _BridgeResidualBlock(hidden_channels),
                nn.SiLU(),
                _BridgeResidualBlock(hidden_channels),
                nn.SiLU(),
                _BridgeResidualBlock(hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            )
        if arch == "multiscale":
            return _MultiScaleBridgeAdapter(hidden_channels)
        raise ValueError(f"Unsupported target_code_bridge_arch={arch!r}")

    def _decoder_modulation(self, d2nn: Mapping[str, Any]) -> str:
        override = self.config.decoder_diagnostic_modulation_override
        if override == "none":
            return str(d2nn["modulation"])
        return override

    def _decoder_phase_layer_mask(self) -> tuple[bool, ...]:
        requested = tuple(self.config.decoder_diagnostic_phase_layer_mask)
        if requested:
            return requested
        return (True,) * int(self._d2nn["decoder_layers"])

    def _resolve_d2nn_params(self) -> dict:
        """从 config.d2nn 解析所有 D2NN 参数（优先），否则回退到散落字段."""
        d2nn_cfg = self.config.d2nn
        if d2nn_cfg is not None:
            return {
                "wavelength": d2nn_cfg.wavelength,
                "pixel_size": d2nn_cfg.pixel_size,
                "grid_size": d2nn_cfg.grid_size,
                "propagation_grid_size": d2nn_cfg.propagation_grid_size,
                "encoder_layers": d2nn_cfg.encoder_layers,
                "decoder_layers": d2nn_cfg.decoder_layers,
                "layer_spacing": d2nn_cfg.layer_spacing,
                "modulation": d2nn_cfg.modulation,
                "phase_init": d2nn_cfg.phase_init,
                "amplitude_logit_init": d2nn_cfg.amplitude_logit_init,
            }
        return {
            "wavelength": self.physics.optics.wavelength,
            "pixel_size": self.physics.optics.pixel_size,
            "grid_size": self.config.diffractive_grid_size,
            "propagation_grid_size": self.config.diffractive_propagation_grid_size,
            "encoder_layers": self.config.encoder_diffractive_layers,
            "decoder_layers": self.config.decoder_diffractive_layers,
            "layer_spacing": self.config.diffractive_layer_spacing,
            "modulation": (
                "phase_amplitude"
                if self.config.optical_core_arch == "phase_amplitude_d2nn"
                else "phase_only"
            ),
            "phase_init": self.config.diffractive_phase_init,
            "amplitude_logit_init": 4.0,
        }

    def _validate_dimensions(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.config.code_dim <= 0:
            raise ValueError(
                f"one_shot.code_dim must be positive, got {self.config.code_dim}"
            )
        if self.config.label_embed_dim <= 0:
            raise ValueError(
                f"one_shot.label_embed_dim must be positive, got {self.config.label_embed_dim}"
            )
        if self.config.hidden_dim <= 0:
            raise ValueError(
                f"one_shot.hidden_dim must be positive, got {self.config.hidden_dim}"
            )
        if self.config.num_classes <= 0:
            raise ValueError(
                f"one_shot.num_classes must be positive, got {self.config.num_classes}"
            )

    def _validate_generator_config(self) -> None:
        if self.config.generator_e_arch not in _ONE_SHOT_GENERATOR_ARCHES:
            raise ValueError(
                "Unsupported one_shot.generator_e_arch: "
                f"{self.config.generator_e_arch}; expected label_template, "
                "label_template_code_residual, mlp, or template_residual_mlp."
            )
        if self.config.generator_command_transform not in _ONE_SHOT_COMMAND_TRANSFORMS:
            raise ValueError(
                "Unsupported one_shot.generator_command_transform: "
                f"{self.config.generator_command_transform}"
            )
        if self.config.generator_template_init not in _ONE_SHOT_TEMPLATE_INITIALIZERS:
            raise ValueError(
                "Unsupported one_shot.generator_template_init: "
                f"{self.config.generator_template_init}"
            )
        if self.config.generator_e_arch == "label_template" and (
            self.config.generator_command_transform != "intensity_to_amplitude"
            or self.config.generator_template_init != "mnist_prototype"
            or self.config.distill_target_source != "mnist_prototype"
        ):
            raise ValueError(
                "label_template requires mnist_prototype initialization and targets plus "
                "the intensity_to_amplitude command transform"
            )
        if self.config.generator_e_arch == "label_template_code_residual" and (
            self.config.generator_command_transform != "intensity_to_amplitude"
            or self.config.generator_template_init != "mnist_prototype"
        ):
            raise ValueError(
                "label_template_code_residual requires mnist_prototype initialization plus "
                "the intensity_to_amplitude command transform"
            )
        if self.config.code_residual_scale < 0.0:
            raise ValueError(
                "one_shot.code_residual_scale must be non-negative, "
                f"got {self.config.code_residual_scale}"
            )
        if self.config.diagnostic_slm_override_mode not in _ONE_SHOT_SLM_OVERRIDE_MODES:
            raise ValueError(
                "Unsupported one_shot.diagnostic_slm_override_mode: "
                f"{self.config.diagnostic_slm_override_mode}; expected none or lowfreq_additive."
            )
        if float(self.config.diagnostic_slm_override_amplitude) < 0.0:
            raise ValueError(
                "one_shot.diagnostic_slm_override_amplitude must be non-negative, "
                f"got {self.config.diagnostic_slm_override_amplitude}"
            )
        if (
            self.config.diagnostic_slm_override_mode == "none"
            and float(self.config.diagnostic_slm_override_amplitude) != 0.0
        ):
            raise ValueError(
                "one_shot.diagnostic_slm_override_amplitude must be 0.0 when "
                "one_shot.diagnostic_slm_override_mode='none'."
            )

    def _validate_diagnostic_config(self) -> None:
        if self.config.ols_kind not in {
            "pointwise_tanh",
            "tanh",
            "gelu",
            "silu",
            "relu",
            "identity",
        }:
            raise ValueError(
                "Unsupported one_shot.ols_kind: "
                f"{self.config.ols_kind}; expected gelu, tanh, silu, relu, or identity."
            )
        if (
            self.config.decoder_diagnostic_mode
            not in _ONE_SHOT_DECODER_DIAGNOSTIC_MODES
        ):
            raise ValueError(
                "Unsupported one_shot.decoder_diagnostic_mode: "
                f"{self.config.decoder_diagnostic_mode}; expected normal, bypass_decoder, "
                "or propagation_only_decoder."
            )
        if (
            self.config.decoder_diagnostic_modulation_override
            not in _ONE_SHOT_DECODER_MODULATION_OVERRIDES
        ):
            raise ValueError(
                "Unsupported one_shot.decoder_diagnostic_modulation_override: "
                f"{self.config.decoder_diagnostic_modulation_override}; expected none or phase_amplitude."
            )
        if (
            self.config.decoder_diagnostic_mask_component
            not in _ONE_SHOT_DECODER_MASK_COMPONENTS
        ):
            raise ValueError(
                "Unsupported one_shot.decoder_diagnostic_mask_component: "
                f"{self.config.decoder_diagnostic_mask_component}; expected trained, phase_only, "
                "or amplitude_only."
            )
        if (
            self.config.decoder_diagnostic_mask_component != "trained"
            and self.config.decoder_diagnostic_mode != "normal"
        ):
            raise ValueError(
                "one_shot.decoder_diagnostic_mask_component phase_only/amplitude_only "
                "requires decoder_diagnostic_mode=normal"
            )
        if (
            self.config.decoder_diagnostic_mask_component != "trained"
            and self.config.decoder_diagnostic_modulation_override != "phase_amplitude"
        ):
            raise ValueError(
                "one_shot.decoder_diagnostic_mask_component phase_only/amplitude_only "
                "requires decoder_diagnostic_modulation_override=phase_amplitude"
            )
        phase_layer_mask = tuple(self.config.decoder_diagnostic_phase_layer_mask)
        if phase_layer_mask:
            decoder_layers = int(self._resolve_d2nn_params()["decoder_layers"])
            if len(phase_layer_mask) != decoder_layers:
                raise ValueError(
                    "one_shot.decoder_diagnostic_phase_layer_mask length must equal "
                    f"decoder layer count {decoder_layers}, got {len(phase_layer_mask)}"
                )
            if not all(type(enabled) is bool for enabled in phase_layer_mask):
                raise ValueError(
                    "one_shot.decoder_diagnostic_phase_layer_mask entries must be booleans"
                )
            if not any(phase_layer_mask):
                raise ValueError(
                    "one_shot.decoder_diagnostic_phase_layer_mask must enable at least one layer"
                )
            if self.config.decoder_diagnostic_mode != "normal":
                raise ValueError(
                    "one_shot.decoder_diagnostic_phase_layer_mask requires decoder_diagnostic_mode=normal"
                )
            if self.config.decoder_diagnostic_mask_component != "phase_only":
                raise ValueError(
                    "one_shot.decoder_diagnostic_phase_layer_mask requires "
                    "decoder_diagnostic_mask_component=phase_only"
                )
        if self.config.target_code_diagnostic_mode not in _ONE_SHOT_TARGET_CODE_MODES:
            raise ValueError(
                "Unsupported one_shot.target_code_diagnostic_mode: "
                f"{self.config.target_code_diagnostic_mode}; expected none, spatial_identity_code, "
                "lowfreq_digit_code, thick_lowfreq_digit_code, sharp_stroke_digit_code, "
                "small_sharp_stroke_digit_code, or class_codebook_grid."
            )
        if (
            self.config.target_code_bridge_source
            not in _ONE_SHOT_TARGET_CODE_BRIDGE_SOURCES
        ):
            raise ValueError(
                "Unsupported one_shot.target_code_bridge_source: "
                f"{self.config.target_code_bridge_source}; expected none, encoded_amplitude, or ols_output_amplitude."
            )
        if int(self.config.target_code_bridge_hidden_channels) <= 0:
            raise ValueError(
                "one_shot.target_code_bridge_hidden_channels must be positive, "
                f"got {self.config.target_code_bridge_hidden_channels}"
            )
        if (
            self.config.target_code_bridge_arch
            not in _ONE_SHOT_TARGET_CODE_BRIDGE_ARCHES
        ):
            raise ValueError(
                "Unsupported one_shot.target_code_bridge_arch: "
                f"{self.config.target_code_bridge_arch}; expected shallow, deep_residual, or multiscale."
            )
        if not _target_code_bridge_mode_is_compatible(self.config):
            raise ValueError(
                "one_shot.target_code_bridge_source requires target_code_diagnostic_mode "
                "to be spatial_identity_code, lowfreq_digit_code, thick_lowfreq_digit_code, "
                "sharp_stroke_digit_code, or small_sharp_stroke_digit_code."
            )
        if self.config.ols_transfer_calibration not in _ONE_SHOT_OLS_CALIBRATIONS:
            raise ValueError(
                "Unsupported one_shot.ols_transfer_calibration: "
                f"{self.config.ols_transfer_calibration}; expected none, phase_screen, or amplitude_phase_screen."
            )

    def _validate_d2nn_modulation(self, d2nn_cfg) -> None:
        if d2nn_cfg.modulation not in {"phase_only", "phase_amplitude"}:
            raise ValueError(
                "Unsupported one_shot.d2nn.modulation: "
                f"{d2nn_cfg.modulation}; expected phase_only or phase_amplitude."
            )
        if not torch.isfinite(torch.tensor(float(d2nn_cfg.amplitude_logit_init))):
            raise ValueError(
                "one_shot.d2nn.amplitude_logit_init must be finite, "
                f"got {d2nn_cfg.amplitude_logit_init}"
            )
        expected_modulation = _expected_d2nn_modulation(self.config.optical_core_arch)
        if d2nn_cfg.modulation != expected_modulation:
            raise ValueError(
                "one_shot.optical_core_arch and one_shot.d2nn.modulation disagree: "
                f"{self.config.optical_core_arch} requires {expected_modulation}, "
                f"got {d2nn_cfg.modulation}."
            )

    @staticmethod
    def _validate_d2nn_geometry(d2nn_cfg) -> None:
        if d2nn_cfg.grid_size <= 0:
            raise ValueError(
                f"one_shot.d2nn.grid_size must be positive, got {d2nn_cfg.grid_size}"
            )
        if d2nn_cfg.propagation_grid_size < d2nn_cfg.grid_size:
            raise ValueError(
                "one_shot.d2nn.propagation_grid_size must be at least "
                "one_shot.d2nn.grid_size, got "
                f"{d2nn_cfg.propagation_grid_size} and {d2nn_cfg.grid_size}"
            )
        if d2nn_cfg.encoder_layers <= 0:
            raise ValueError(
                "one_shot.d2nn.encoder_layers must be positive, "
                f"got {d2nn_cfg.encoder_layers}"
            )
        if d2nn_cfg.decoder_layers <= 0:
            raise ValueError(
                "one_shot.d2nn.decoder_layers must be positive, "
                f"got {d2nn_cfg.decoder_layers}"
            )
        if d2nn_cfg.layer_spacing <= 0.0:
            raise ValueError(
                "one_shot.d2nn.layer_spacing must be positive, "
                f"got {d2nn_cfg.layer_spacing}"
            )
        if d2nn_cfg.phase_init not in {"uniform_pi", "uniform_2pi", "small"}:
            raise ValueError(
                "Unsupported one_shot.d2nn.phase_init: "
                f"{d2nn_cfg.phase_init}; expected uniform_pi, uniform_2pi, or small."
            )

    def _validate_explicit_d2nn_config(self) -> None:
        d2nn_cfg = self.config.d2nn
        if d2nn_cfg is None:
            return
        self._validate_d2nn_modulation(d2nn_cfg)
        self._validate_d2nn_geometry(d2nn_cfg)

    def _validate_legacy_diffractive_config(self) -> None:
        if self.config.diffractive_grid_size <= 0:
            raise ValueError(
                "one_shot.diffractive_grid_size must be positive, "
                f"got {self.config.diffractive_grid_size}"
            )
        if (
            self.config.diffractive_propagation_grid_size
            < self.config.diffractive_grid_size
        ):
            raise ValueError(
                "one_shot.diffractive_propagation_grid_size must be at least "
                "one_shot.diffractive_grid_size, got "
                f"{self.config.diffractive_propagation_grid_size} and {self.config.diffractive_grid_size}"
            )
        if self.config.encoder_diffractive_layers <= 0:
            raise ValueError(
                "one_shot.encoder_diffractive_layers must be positive, "
                f"got {self.config.encoder_diffractive_layers}"
            )
        if self.config.decoder_diffractive_layers <= 0:
            raise ValueError(
                "one_shot.decoder_diffractive_layers must be positive, "
                f"got {self.config.decoder_diffractive_layers}"
            )
        if self.config.diffractive_layer_spacing <= 0.0:
            raise ValueError(
                "one_shot.diffractive_layer_spacing must be positive, "
                f"got {self.config.diffractive_layer_spacing}"
            )
        if self.config.diffractive_phase_init not in {
            "uniform_pi",
            "uniform_2pi",
            "small",
        }:
            raise ValueError(
                "Unsupported one_shot.diffractive_phase_init: "
                f"{self.config.diffractive_phase_init}; expected uniform_pi, uniform_2pi, or small."
            )

    def _validate_optical_config(self) -> None:
        if self.config.optical_core_arch not in {
            "dense_linear",
            "diffractive_phase_only",
            "phase_amplitude_d2nn",
            "camera_direct_clean_oracle",
        }:
            raise ValueError(
                "Unsupported one_shot.optical_core_arch: "
                f"{self.config.optical_core_arch}; expected dense_linear, "
                "diffractive_phase_only, phase_amplitude_d2nn, or camera_direct_clean_oracle."
            )
        if self.config.metasurface_grid_size <= 0:
            raise ValueError(
                "one_shot.metasurface_grid_size must be positive, "
                f"got {self.config.metasurface_grid_size}"
            )
        if self.config.layer_spacing <= 0.0:
            raise ValueError(
                f"one_shot.layer_spacing must be positive, got {self.config.layer_spacing}"
            )
        if not self.uses_diffractive_core:
            return
        if self.config.d2nn is not None:
            self._validate_explicit_d2nn_config()
        else:
            self._validate_legacy_diffractive_config()

    def _validate_config(self) -> None:
        self._validate_dimensions()
        self._validate_generator_config()
        self._validate_diagnostic_config()
        self._validate_optical_config()

    def _initialize_layers(self) -> None:
        if not self.uses_diffractive_core and not self.uses_direct_camera_core:
            nn.init.xavier_uniform_(self.encoder.weight, gain=0.5)
            nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)

    def _apply_ols_transfer_calibration(self, field: torch.Tensor) -> torch.Tensor:
        if self.config.ols_transfer_calibration == "none":
            return field
        if self.ols_phase_screen is None:
            raise RuntimeError(
                "OLS phase screen parameter is missing while calibration is enabled."
            )
        phase = self.ols_phase_screen.to(device=field.device, dtype=field.real.dtype)
        phase_transfer = torch.polar(torch.ones_like(phase), phase).to(field.dtype)
        calibrated = field * phase_transfer
        if self.config.ols_transfer_calibration == "amplitude_phase_screen":
            if self.ols_amplitude_log is None:
                raise RuntimeError(
                    "OLS amplitude screen parameter is missing while calibration is enabled."
                )
            amplitude = torch.exp(
                self.ols_amplitude_log.to(device=field.device, dtype=field.real.dtype)
            )
            calibrated = calibrated * amplitude.to(field.dtype)
        return calibrated

    def _bypass_decoder_layers(
        self,
        field: torch.Tensor,
        *,
        num_layers: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        layer_fields = field.unsqueeze(1).expand(-1, num_layers, -1, -1, -1)
        batch = field.shape[0]
        zeros = field.real.new_zeros(batch, num_layers)
        diagnostics = {
            "input_energy": zeros,
            "padded_input_energy": zeros,
            "propagated_full_energy": zeros,
            "cropped_energy": zeros,
            "crop_retention": zeros,
            "propagation_energy_ratio": zeros,
        }
        return field, layer_fields, diagnostics

    def _propagation_only_decoder_layers(
        self,
        field: torch.Tensor,
        *,
        num_layers: int,
        wavelength: float,
        layer_spacing: float,
        pixel_size: float,
        propagation_grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        layer_fields = []
        diagnostics: dict[str, list[torch.Tensor]] = {
            "input_energy": [],
            "padded_input_energy": [],
            "propagated_full_energy": [],
            "cropped_energy": [],
            "crop_retention": [],
            "propagation_energy_ratio": [],
        }
        current = field
        for _ in range(num_layers):
            current, layer_diagnostics = (
                _angular_spectrum_propagate_center_padded_with_diagnostics(
                    current,
                    layer_spacing,
                    wavelength,
                    pixel_size,
                    propagation_grid_size,
                )
            )
            layer_fields.append(current)
            for key in diagnostics:
                diagnostics[key].append(layer_diagnostics[key])
        stacked_diagnostics = {
            key: torch.stack(values, dim=1) for key, values in diagnostics.items()
        }
        return current, torch.stack(layer_fields, dim=1), stacked_diagnostics

    def _diagnostic_slm_override(
        self,
        slm_command: torch.Tensor,
        code: torch.Tensor,
        generator_diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mode = self.config.diagnostic_slm_override_mode
        amplitude = float(self.config.diagnostic_slm_override_amplitude)
        if mode == "none":
            return slm_command, {
                "diagnostic_slm_override_active": slm_command.new_tensor(
                    False, dtype=torch.bool
                ),
                "diagnostic_slm_override_amplitude": slm_command.new_tensor(0.0),
            }
        if mode != "lowfreq_additive":
            raise RuntimeError(f"Unhandled diagnostic_slm_override_mode={mode!r}")
        base_command = generator_diagnostics.get("generator_base_command", slm_command)
        if not torch.is_tensor(base_command):
            base_command = slm_command
        coords = torch.linspace(
            -torch.pi,
            torch.pi,
            slm_command.shape[-1],
            device=slm_command.device,
            dtype=slm_command.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        features = torch.tanh(
            code.to(device=slm_command.device, dtype=slm_command.dtype)
        )
        if features.shape[1] < 8:
            features = F.pad(features, (0, 8 - features.shape[1]))
        coeff = features[:, :8].view(-1, 8, 1, 1)
        basis = torch.stack(
            [
                torch.sin(xx),
                torch.cos(xx),
                torch.sin(yy),
                torch.cos(yy),
                torch.sin(xx + yy),
                torch.cos(xx - yy),
                torch.sin(2.0 * xx + yy),
                torch.cos(xx + 2.0 * yy),
            ],
            dim=0,
        ).unsqueeze(0)
        perturbation = (coeff * basis).sum(dim=1, keepdim=True)
        perturbation = F.avg_pool2d(perturbation, kernel_size=7, stride=1, padding=3)
        max_abs = (
            perturbation.flatten(1)
            .abs()
            .amax(dim=1)
            .view(-1, 1, 1, 1)
            .clamp_min(1.0e-6)
        )
        perturbation = perturbation / max_abs
        overridden = (
            base_command.to(dtype=slm_command.dtype) + amplitude * perturbation
        ).clamp(0.0, 1.0)
        return overridden, {
            "diagnostic_slm_override_active": slm_command.new_tensor(
                True, dtype=torch.bool
            ),
            "diagnostic_slm_override_amplitude": slm_command.new_tensor(amplitude),
            "diagnostic_slm_override_pattern": perturbation,
            "diagnostic_slm_override_base_command": base_command,
        }

    def forward(
        self,
        code: torch.Tensor,
        class_labels: torch.Tensor,
        *,
        return_optical_fields: bool = False,
    ) -> dict[str, torch.Tensor]:
        slm_command, generator_diagnostics = self.generator_e.forward_with_diagnostics(
            code, class_labels
        )
        slm_command, override_diagnostics = self._diagnostic_slm_override(
            slm_command,
            code,
            generator_diagnostics,
        )
        generator_diagnostics = {**generator_diagnostics, **override_diagnostics}
        slm_outputs = self.slm(slm_command)
        if self.uses_direct_camera_core:
            outputs = {
                **self._forward_direct_camera(slm_outputs),
                **generator_diagnostics,
            }
            if return_optical_fields:
                outputs["optical_fields"] = {}
            return outputs
        if self.uses_diffractive_core:
            return {
                **self._forward_diffractive(
                    slm_outputs,
                    class_labels,
                    return_optical_fields=return_optical_fields,
                ),
                **generator_diagnostics,
            }
        slm_transfer = slm_outputs["slm_transfer"]
        encoder_input_field = self.front_path.forward_field(
            slm_transfer,
            propagation_method="angular_spectrum",
        )
        encoder_grid = _resize_complex_field(
            encoder_input_field,
            (self.config.metasurface_grid_size, self.config.metasurface_grid_size),
        )
        encoded_vector = self.encoder(_flatten_complex_field(encoder_grid))
        encoded_field = _vector_to_complex_field(
            encoded_vector, self.config.metasurface_grid_size
        )
        ols_field = self._apply_ols_transfer_calibration(
            _apply_complex_ols(encoded_field, self.config.ols_kind)
        )
        decoder_input_field, target_code_amplitude = self._target_code_field(
            encoded_field=encoded_field,
            ols_field=ols_field,
            slm_command=slm_outputs["slm_command"],
            class_labels=class_labels,
            grid_size=self.config.metasurface_grid_size,
        )
        if self.config.decoder_diagnostic_mode == "bypass_decoder":
            chip_grid = decoder_input_field
        else:
            decoded_vector = self.decoder(_flatten_complex_field(decoder_input_field))
            chip_grid = _vector_to_complex_field(
                decoded_vector, self.config.metasurface_grid_size
            )
        chip_field = _resize_complex_field(
            chip_grid, (self.image_size, self.image_size)
        )
        optical_readout = self.back_path(chip_field)
        outputs = {
            "x_hat": optical_readout["camera_plane"],
            "camera_plane": optical_readout["camera_plane"],
            "camera_plane_loss": optical_readout["camera_plane_loss"],
            "slm_command": slm_outputs["slm_command"],
            "slm_amplitude": slm_outputs["slm_amplitude"],
            "slm_phase": slm_outputs["slm_phase"],
            "encoder_input_amplitude": encoder_input_field.abs(),
            "encoder_input_phase": torch.angle(encoder_input_field),
            "encoder_input_grid_amplitude": encoder_grid.abs(),
            "encoder_input_grid_phase": torch.angle(encoder_grid),
            "encoded_amplitude": encoded_field.abs(),
            "encoded_phase": torch.angle(encoded_field),
            "ols_input_amplitude": encoded_field.abs(),
            "ols_input_phase": torch.angle(encoded_field),
            "ols_output_amplitude": ols_field.abs(),
            "ols_output_phase": torch.angle(ols_field),
            "ols_amplitude": ols_field.abs(),
            "ols_phase": torch.angle(ols_field),
            "decoder_input_amplitude": decoder_input_field.abs(),
            "decoder_input_phase": torch.angle(decoder_input_field),
            "target_code_amplitude": target_code_amplitude,
            "target_code_diagnostic_mode": self.config.target_code_diagnostic_mode,
            "target_code_spatial_scale": self._effective_target_code_spatial_scale(),
            "target_code_diagnostic_active": torch.tensor(
                self.config.target_code_diagnostic_mode != "none",
                device=chip_grid.device,
            ),
            "target_code_bridge_source": self.config.target_code_bridge_source,
            "target_code_bridge_arch": self.config.target_code_bridge_arch,
            "target_code_bridge_active": torch.tensor(
                self.config.target_code_bridge_source != "none",
                device=chip_grid.device,
            ),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "decoder_diagnostic_mode": self.config.decoder_diagnostic_mode,
            "decoder_diagnostic_modulation_override": self.config.decoder_diagnostic_modulation_override,
            "decoder_diagnostic_mask_component": self.config.decoder_diagnostic_mask_component,
            "decoder_diagnostic_phase_layer_mask": tuple(
                self.config.decoder_diagnostic_phase_layer_mask
            ),
            "decoder_diagnostic_phase_layer_mask_applied": self._decoder_phase_layer_mask(),
            "decoder_bypass_active": torch.tensor(
                self.config.decoder_diagnostic_mode == "bypass_decoder",
                device=chip_grid.device,
            ),
            "ols_transfer_calibration": self.config.ols_transfer_calibration,
            "chip_real": chip_field.real,
            "chip_imag": chip_field.imag,
            **optical_readout,
            **generator_diagnostics,
        }
        if return_optical_fields:
            outputs["optical_fields"] = {}
        return outputs

    def _forward_direct_camera(
        self, slm_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        camera_plane_loss = slm_outputs["slm_command"]
        camera_phase = torch.zeros_like(camera_plane_loss)
        camera_complex = torch.complex(camera_plane_loss, camera_phase)
        return {
            "x_hat": camera_plane_loss.clamp(0.0, 1.0),
            "camera_plane": camera_plane_loss.clamp(0.0, 1.0),
            "camera_plane_loss": camera_plane_loss,
            "slm_command": slm_outputs["slm_command"],
            "slm_amplitude": slm_outputs["slm_amplitude"],
            "slm_phase": slm_outputs["slm_phase"],
            "encoder_input_amplitude": slm_outputs["slm_amplitude"],
            "encoder_input_phase": slm_outputs["slm_phase"],
            "encoder_input_grid_amplitude": slm_outputs["slm_amplitude"],
            "encoder_input_grid_phase": slm_outputs["slm_phase"],
            "encoded_amplitude": camera_plane_loss,
            "encoded_phase": camera_phase,
            "ols_input_amplitude": camera_plane_loss,
            "ols_input_phase": camera_phase,
            "ols_output_amplitude": camera_plane_loss,
            "ols_output_phase": camera_phase,
            "ols_amplitude": camera_plane_loss,
            "ols_phase": camera_phase,
            "decoder_input_amplitude": camera_plane_loss,
            "decoder_input_phase": camera_phase,
            "decoder_output_amplitude": camera_plane_loss,
            "decoder_output_phase": camera_phase,
            "decoder_diagnostic_mode": self.config.decoder_diagnostic_mode,
            "decoder_diagnostic_modulation_override": self.config.decoder_diagnostic_modulation_override,
            "decoder_diagnostic_mask_component": self.config.decoder_diagnostic_mask_component,
            "decoder_diagnostic_phase_layer_mask": tuple(
                self.config.decoder_diagnostic_phase_layer_mask
            ),
            "decoder_diagnostic_phase_layer_mask_applied": self._decoder_phase_layer_mask(),
            "target_code_diagnostic_mode": self.config.target_code_diagnostic_mode,
            "target_code_spatial_scale": self._effective_target_code_spatial_scale(),
            "target_code_diagnostic_active": torch.tensor(
                self.config.target_code_diagnostic_mode != "none",
                device=camera_plane_loss.device,
            ),
            "target_code_bridge_source": self.config.target_code_bridge_source,
            "target_code_bridge_arch": self.config.target_code_bridge_arch,
            "target_code_bridge_active": torch.tensor(
                self.config.target_code_bridge_source != "none",
                device=camera_plane_loss.device,
            ),
            "target_code_amplitude": camera_plane_loss,
            "decoder_bypass_active": torch.tensor(
                self.config.decoder_diagnostic_mode == "bypass_decoder",
                device=camera_plane_loss.device,
            ),
            "ols_transfer_calibration": self.config.ols_transfer_calibration,
            "chip_real": camera_complex.real,
            "chip_imag": camera_complex.imag,
            "chip_amplitude": camera_plane_loss,
            "chip_phase": camera_phase,
            "camera_intensity_raw": camera_plane_loss,
            "probe_plane": camera_plane_loss,
            "camera_phase": camera_phase,
        }

    @staticmethod
    def _normalize_per_sample(tensor: torch.Tensor) -> torch.Tensor:
        min_value = tensor.amin(dim=(-2, -1), keepdim=True)
        max_value = tensor.amax(dim=(-2, -1), keepdim=True)
        return (tensor - min_value) / (max_value - min_value).clamp_min(1.0e-6)

    def _effective_target_code_spatial_scale(self) -> float | None:
        if self.config.target_code_diagnostic_mode != "small_sharp_stroke_digit_code":
            return None
        if self.config.target_code_spatial_scale is not None:
            return float(self.config.target_code_spatial_scale)
        return 0.72

    def _target_code_field(
        self,
        *,
        encoded_field: torch.Tensor | None = None,
        ols_field: torch.Tensor,
        slm_command: torch.Tensor,
        class_labels: torch.Tensor,
        grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.config.target_code_diagnostic_mode
        bridge_source = self.config.target_code_bridge_source

        def _apply_target_code_shape(amplitude: torch.Tensor) -> torch.Tensor:
            if mode == "lowfreq_digit_code":
                amplitude = F.avg_pool2d(amplitude, kernel_size=7, stride=1, padding=3)
                amplitude = F.avg_pool2d(amplitude, kernel_size=7, stride=1, padding=3)
                return self._normalize_per_sample(amplitude)
            if mode == "thick_lowfreq_digit_code":
                amplitude = F.max_pool2d(amplitude, kernel_size=5, stride=1, padding=2)
                amplitude = F.avg_pool2d(amplitude, kernel_size=7, stride=1, padding=3)
                amplitude = F.avg_pool2d(amplitude, kernel_size=7, stride=1, padding=3)
                return self._normalize_per_sample(amplitude)
            if mode == "sharp_stroke_digit_code":
                normalized = self._normalize_per_sample(amplitude)
                threshold = torch.maximum(
                    normalized.new_full((normalized.shape[0], 1, 1, 1), 0.30),
                    normalized.mean(dim=(-2, -1), keepdim=True)
                    + 0.35 * normalized.std(dim=(-2, -1), keepdim=True, unbiased=False),
                )
                strokes = torch.sigmoid((normalized - threshold) * 18.0)
                strokes = F.max_pool2d(strokes, kernel_size=3, stride=1, padding=1)
                strokes = F.avg_pool2d(strokes, kernel_size=3, stride=1, padding=1)
                return self._normalize_per_sample(strokes)
            if mode == "small_sharp_stroke_digit_code":
                normalized = self._normalize_per_sample(amplitude)
                threshold = torch.maximum(
                    normalized.new_full((normalized.shape[0], 1, 1, 1), 0.42),
                    normalized.mean(dim=(-2, -1), keepdim=True)
                    + 0.55 * normalized.std(dim=(-2, -1), keepdim=True, unbiased=False),
                )
                strokes = torch.sigmoid((normalized - threshold) * 22.0)
                strokes = F.avg_pool2d(strokes, kernel_size=3, stride=1, padding=1)
                scale = self._effective_target_code_spatial_scale()
                assert scale is not None
                small_size = max(4, int(round(strokes.shape[-1] * scale)))
                shrunk = F.interpolate(
                    strokes,
                    size=(small_size, small_size),
                    mode="bilinear",
                    align_corners=False,
                )
                canvas = torch.zeros_like(strokes)
                top = (strokes.shape[-2] - small_size) // 2
                left = (strokes.shape[-1] - small_size) // 2
                canvas[..., top : top + small_size, left : left + small_size] = shrunk
                return self._normalize_per_sample(canvas)
            return amplitude

        if bridge_source != "none":
            if self.target_code_bridge_adapter is None:
                raise RuntimeError(
                    "target_code_bridge_adapter is missing despite active bridge source"
                )
            source_field = (
                encoded_field if bridge_source == "encoded_amplitude" else ols_field
            )
            if source_field is None:
                raise ValueError(
                    f"target_code_bridge_source={bridge_source!r} is not available for this model path"
                )
            amplitude = self._normalize_per_sample(source_field.abs().float())
            if amplitude.shape[-2:] != (grid_size, grid_size):
                amplitude = F.interpolate(
                    amplitude,
                    size=(grid_size, grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
            amplitude = torch.sigmoid(
                self.target_code_bridge_adapter(amplitude.to(device=ols_field.device))
            )
            amplitude = _apply_target_code_shape(
                self._normalize_per_sample(amplitude)
            ).to(dtype=ols_field.real.dtype)
            return torch.complex(amplitude, torch.zeros_like(amplitude)).to(
                ols_field.dtype
            ), amplitude
        if mode == "none":
            return ols_field, ols_field.abs()
        dtype = ols_field.real.dtype
        device = ols_field.device
        if mode in {
            "spatial_identity_code",
            "lowfreq_digit_code",
            "thick_lowfreq_digit_code",
            "sharp_stroke_digit_code",
            "small_sharp_stroke_digit_code",
        }:
            amplitude = F.interpolate(
                slm_command.to(device=device, dtype=dtype),
                size=(grid_size, grid_size),
                mode="bilinear",
                align_corners=False,
            )
            amplitude = self._normalize_per_sample(amplitude)
            amplitude = _apply_target_code_shape(amplitude)
        elif mode == "class_codebook_grid":
            coords = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            labels = class_labels.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
            freq_x = labels.remainder(5.0) + 1.0
            freq_y = torch.floor(labels / 5.0) + 1.0
            phase = (labels + 1.0) * 0.37
            pattern = torch.sin(freq_x * torch.pi * xx + phase) + torch.cos(
                freq_y * torch.pi * yy - phase
            )
            checker = torch.sin((freq_x + freq_y) * torch.pi * (xx + yy) * 0.5)
            amplitude = self._normalize_per_sample(pattern + 0.35 * checker)
        else:
            raise RuntimeError(f"Unhandled target code diagnostic mode: {mode!r}")
        return torch.complex(amplitude, torch.zeros_like(amplitude)).to(
            ols_field.dtype
        ), amplitude

    def _forward_diffractive(
        self,
        slm_outputs: dict[str, torch.Tensor],
        class_labels: torch.Tensor,
        *,
        return_optical_fields: bool = False,
    ) -> dict[str, torch.Tensor]:
        d2nn = self._d2nn
        slm_transfer = slm_outputs["slm_transfer"]
        front_field, front_path_diagnostics = (
            self.front_path.forward_field_with_diagnostics(
                slm_transfer,
                propagation_method="angular_spectrum",
            )
        )
        front_path_diagnostics = dict(front_path_diagnostics)
        slm_field = front_path_diagnostics.pop("_slm_field", slm_transfer)
        if self.physics.front_path_model == "coherent_imaging_4f":
            encoder_input_field = front_field
        elif self.physics.physical_forward_contract:
            source_sampling = PlaneSampling(
                "lightgen_propagated_field",
                height=int(front_field.shape[-2]),
                width=int(front_field.shape[-1]),
                pixel_size_y=self.physics.optics.pixel_size,
                pixel_size_x=self.physics.optics.pixel_size,
            )
            encoder_input_field = sample_center_aperture(
                front_field,
                source_sampling,
                target_size=(d2nn["grid_size"], d2nn["grid_size"]),
                target_pixel_size=d2nn["pixel_size"],
            )
        else:
            encoder_input_field = _resize_complex_field(
                front_field,
                (d2nn["grid_size"], d2nn["grid_size"]),
            )
        encoder_grid = encoder_input_field
        encoded_field, encoder_layer_fields, encoder_crop_diagnostics = self.encoder(
            encoder_grid
        )
        ols_field = self._apply_ols_transfer_calibration(
            _apply_complex_ols(encoded_field, self.config.ols_kind)
        )
        decoder_input_field, target_code_amplitude = self._target_code_field(
            encoded_field=encoded_field,
            ols_field=ols_field,
            slm_command=slm_outputs["slm_command"],
            class_labels=class_labels,
            grid_size=d2nn["grid_size"],
        )
        if self.config.decoder_diagnostic_mode == "bypass_decoder":
            chip_grid, decoder_layer_fields, decoder_crop_diagnostics = (
                self._bypass_decoder_layers(
                    decoder_input_field,
                    num_layers=d2nn["decoder_layers"],
                )
            )
        elif self.config.decoder_diagnostic_mode == "propagation_only_decoder":
            chip_grid, decoder_layer_fields, decoder_crop_diagnostics = (
                self._propagation_only_decoder_layers(
                    decoder_input_field,
                    num_layers=d2nn["decoder_layers"],
                    wavelength=d2nn["wavelength"],
                    layer_spacing=d2nn["layer_spacing"],
                    pixel_size=d2nn["pixel_size"],
                    propagation_grid_size=d2nn["propagation_grid_size"],
                )
            )
        else:
            chip_grid, decoder_layer_fields, decoder_crop_diagnostics = self.decoder(
                decoder_input_field,
                mask_component=self.config.decoder_diagnostic_mask_component,
                phase_layer_mask=tuple(self.config.decoder_diagnostic_phase_layer_mask),
            )
        raw_optical_readout = self.back_path(
            chip_grid,
            input_pixel_size=d2nn["pixel_size"],
            output_size=(self.image_size, self.image_size),
            output_pixel_size=self.physics.optics.pixel_size,
        )
        camera_intensity = raw_optical_readout["camera_intensity_raw"]
        if tuple(camera_intensity.shape[-2:]) != (self.image_size, self.image_size):
            camera_intensity = _resize_real_intensity(
                camera_intensity, (self.image_size, self.image_size)
            )
        camera_plane_loss = normalize_real_intensity(
            camera_intensity,
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        physical_camera_plane = camera_plane_loss.clamp(0.0, 1.0)
        chip_amplitude = chip_grid.abs()
        chip_intensity_loss = normalize_real_intensity(
            chip_amplitude.pow(2),
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        chip_amplitude_loss = normalize_real_intensity(
            chip_amplitude,
            self.physics.readout.normalize_percentile,
            clamp=False,
        )
        probe_plane_loss = raw_optical_readout["probe_plane_loss"]
        primary_plane = self.physics.readout.primary_plane
        if primary_plane == "camera":
            selected_plane_loss = camera_plane_loss
        elif primary_plane == "probe":
            selected_plane_loss = probe_plane_loss
        elif primary_plane == "chip_amplitude":
            selected_plane_loss = chip_amplitude_loss
        elif primary_plane == "chip_intensity":
            selected_plane_loss = chip_intensity_loss
        else:
            raise ValueError(
                "Unsupported physics.readout.primary_plane: "
                f"{primary_plane}; expected camera, probe, chip_amplitude, or chip_intensity."
            )
        selected_native_plane_loss = selected_plane_loss
        selected_native_plane = selected_native_plane_loss.clamp(0.0, 1.0)
        selected_output_size = (self.image_size, self.image_size)
        selected_plane_loss = _resize_real_intensity(
            selected_native_plane_loss, selected_output_size
        )
        selected_plane = selected_plane_loss.clamp(0.0, 1.0)
        optical_readout = {
            **raw_optical_readout,
            "camera_plane": selected_plane,
            "camera_plane_loss": selected_plane_loss,
        }
        encoder_phase_transfer = self.encoder.phase_transfers()
        decoder_phase_transfer = self.decoder.phase_transfers(
            self.config.decoder_diagnostic_mask_component,
            tuple(self.config.decoder_diagnostic_phase_layer_mask),
        )
        outputs = {
            "x_hat": selected_plane,
            "camera_plane": selected_plane,
            "camera_plane_loss": selected_plane_loss,
            "primary_readout_plane": primary_plane,
            "primary_readout_plane_native": selected_native_plane,
            "primary_readout_plane_loss_native": selected_native_plane_loss,
            "slm_command": slm_outputs["slm_command"],
            "slm_amplitude": slm_outputs["slm_amplitude"],
            "slm_phase": slm_outputs["slm_phase"],
            "encoder_input_amplitude": encoder_input_field.abs(),
            "encoder_input_phase": torch.angle(encoder_input_field),
            "encoder_input_grid_amplitude": encoder_grid.abs(),
            "encoder_input_grid_phase": torch.angle(encoder_grid),
            "encoded_amplitude": encoded_field.abs(),
            "encoded_phase": torch.angle(encoded_field),
            "ols_input_amplitude": encoded_field.abs(),
            "ols_input_phase": torch.angle(encoded_field),
            "ols_output_amplitude": ols_field.abs(),
            "ols_output_phase": torch.angle(ols_field),
            "ols_amplitude": ols_field.abs(),
            "ols_phase": torch.angle(ols_field),
            "decoder_input_amplitude": decoder_input_field.abs(),
            "decoder_input_phase": torch.angle(decoder_input_field),
            "target_code_amplitude": target_code_amplitude,
            "target_code_diagnostic_mode": self.config.target_code_diagnostic_mode,
            "target_code_spatial_scale": self._effective_target_code_spatial_scale(),
            "target_code_diagnostic_active": torch.tensor(
                self.config.target_code_diagnostic_mode != "none",
                device=chip_grid.device,
            ),
            "target_code_bridge_source": self.config.target_code_bridge_source,
            "target_code_bridge_arch": self.config.target_code_bridge_arch,
            "target_code_bridge_active": torch.tensor(
                self.config.target_code_bridge_source != "none",
                device=chip_grid.device,
            ),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "decoder_diagnostic_mode": self.config.decoder_diagnostic_mode,
            "decoder_diagnostic_modulation_override": self.config.decoder_diagnostic_modulation_override,
            "decoder_diagnostic_mask_component": self.config.decoder_diagnostic_mask_component,
            "decoder_diagnostic_phase_layer_mask": tuple(
                self.config.decoder_diagnostic_phase_layer_mask
            ),
            "decoder_diagnostic_phase_layer_mask_applied": self._decoder_phase_layer_mask(),
            "decoder_bypass_active": torch.tensor(
                self.config.decoder_diagnostic_mode == "bypass_decoder",
                device=chip_grid.device,
            ),
            "decoder_propagation_only_active": torch.tensor(
                self.config.decoder_diagnostic_mode == "propagation_only_decoder",
                device=chip_grid.device,
            ),
            "ols_transfer_calibration": self.config.ols_transfer_calibration,
            "chip_real": chip_grid.real,
            "chip_imag": chip_grid.imag,
            "chip_intensity_primary_loss": chip_intensity_loss,
            "chip_intensity_primary": chip_intensity_loss.clamp(0.0, 1.0),
            "chip_amplitude_primary_loss": chip_amplitude_loss,
            "chip_amplitude_primary": chip_amplitude_loss.clamp(0.0, 1.0),
            "physical_camera_plane": physical_camera_plane,
            "physical_camera_plane_loss": camera_plane_loss,
            "physical_camera_intensity_raw": raw_optical_readout[
                "camera_intensity_raw"
            ],
            "front_path_diagnostics": front_path_diagnostics,
            "diffractive_camera_intensity_raw": raw_optical_readout[
                "camera_intensity_raw"
            ],
            "diffractive_camera_plane_raw": raw_optical_readout["camera_plane"],
            "diffractive_camera_plane_loss_raw": raw_optical_readout[
                "camera_plane_loss"
            ],
            "diffractive_encoder_layer_amplitude": encoder_layer_fields.abs(),
            "diffractive_encoder_layer_phase": torch.angle(encoder_layer_fields),
            "diffractive_decoder_layer_amplitude": decoder_layer_fields.abs(),
            "diffractive_decoder_layer_phase": torch.angle(decoder_layer_fields),
            "diffractive_encoder_mask_amplitude": encoder_phase_transfer.abs(),
            "diffractive_encoder_mask_phase": torch.angle(encoder_phase_transfer),
            "diffractive_decoder_mask_amplitude": decoder_phase_transfer.abs(),
            "diffractive_decoder_mask_phase": torch.angle(decoder_phase_transfer),
            **{
                f"diffractive_encoder_crop_{key}": value
                for key, value in encoder_crop_diagnostics.items()
            },
            **{
                f"diffractive_decoder_crop_{key}": value
                for key, value in decoder_crop_diagnostics.items()
            },
            **optical_readout,
        }
        if return_optical_fields:
            outputs["optical_fields"] = self._build_diffractive_optical_fields(
                slm_transfer=slm_transfer,
                slm_field=slm_field,
                front_field=front_field,
                front_path_diagnostics=front_path_diagnostics,
                encoder_input_field=encoder_input_field,
                encoder_grid=encoder_grid,
                encoder_layer_fields=encoder_layer_fields,
                encoded_field=encoded_field,
                ols_field=ols_field,
                decoder_input_field=decoder_input_field,
                decoder_layer_fields=decoder_layer_fields,
                chip_grid=chip_grid,
                raw_optical_readout=raw_optical_readout,
            )
        return outputs

    def _build_diffractive_optical_fields(
        self,
        *,
        slm_transfer: torch.Tensor,
        slm_field: torch.Tensor,
        front_field: torch.Tensor,
        front_path_diagnostics: Mapping[str, object],
        encoder_input_field: torch.Tensor,
        encoder_grid: torch.Tensor,
        encoder_layer_fields: torch.Tensor,
        encoded_field: torch.Tensor,
        ols_field: torch.Tensor,
        decoder_input_field: torch.Tensor,
        decoder_layer_fields: torch.Tensor,
        chip_grid: torch.Tensor,
        raw_optical_readout: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, object]]:
        """Capture model-internal complex fields with provenance, never display proxies."""
        front_phase_available = self.physics.front_path_model != "relay_intensity_psf"
        target_active = self.config.target_code_diagnostic_mode != "none"
        bridge_active = self.config.target_code_bridge_source != "none"
        decoder_phase_available = (
            front_phase_available and not target_active and not bridge_active
        )
        slm_sampling = PlaneSampling(
            "slm",
            height=int(slm_transfer.shape[-2]),
            width=int(slm_transfer.shape[-1]),
            pixel_size_y=self.physics.optics.pixel_size,
            pixel_size_x=self.physics.optics.pixel_size,
        ).to_dict()
        front_sampling = PlaneSampling(
            "pre_encoder_relay_image"
            if self.physics.front_path_model == "coherent_imaging_4f"
            else "lightgen_propagated_field",
            height=int(front_field.shape[-2]),
            width=int(front_field.shape[-1]),
            pixel_size_y=self._d2nn["pixel_size"]
            if self.physics.front_path_model == "coherent_imaging_4f"
            else self.physics.optics.pixel_size,
            pixel_size_x=self._d2nn["pixel_size"]
            if self.physics.front_path_model == "coherent_imaging_4f"
            else self.physics.optics.pixel_size,
        ).to_dict()
        d2nn_sampling = PlaneSampling(
            "lightgen_d2nn",
            height=int(encoder_input_field.shape[-2]),
            width=int(encoder_input_field.shape[-1]),
            pixel_size_y=self._d2nn["pixel_size"],
            pixel_size_x=self._d2nn["pixel_size"],
        ).to_dict()

        def add(
            fields: dict[str, dict[str, object]],
            name: str,
            field: torch.Tensor,
            *,
            upstream: str | None,
            operation: str,
            phase_available: bool,
            source_kind: str = "propagated_complex_field",
            sampling: Mapping[str, object] | None = None,
        ) -> None:
            fields[name] = build_optical_field_record(
                field,
                upstream=upstream,
                operation=operation,
                phase_available=phase_available,
                source_kind=source_kind,
                sampling=sampling,
            )

        fields: dict[str, dict[str, object]] = {}
        front_source_kind = (
            "propagated_complex_field"
            if front_phase_available
            else "intensity_relay_embedding"
        )
        add(
            fields,
            "slm_output",
            slm_field,
            upstream=None,
            operation="slm_output_field",
            phase_available=True,
            sampling=slm_sampling,
        )
        front_operation = str(
            front_path_diagnostics.get("operation", self.physics.front_path_model)
        )
        front_name = (
            "pre_encoder_relay_image"
            if self.physics.front_path_model == "coherent_imaging_4f"
            else "lightgen_input"
        )
        add(
            fields,
            front_name,
            front_field,
            upstream="slm_output",
            operation=front_operation,
            phase_available=front_phase_available,
            source_kind=front_source_kind,
            sampling=front_sampling,
        )
        if self.physics.front_path_model == "coherent_imaging_4f":
            fields[front_name]["relay_trace"] = {
                "front_relay": front_path_diagnostics.get("front_relay"),
                "segments_m": front_path_diagnostics.get("segments_m"),
                "components": front_path_diagnostics.get("relay_trace"),
            }
        add(
            fields,
            "encoder_input",
            encoder_input_field,
            upstream=front_name,
            operation="coherent_imaging_4f"
            if self.physics.front_path_model == "coherent_imaging_4f"
            else (
                "physical_aperture_sampling"
                if self.physics.physical_forward_contract
                else self.physics.front_path_model
            ),
            phase_available=front_phase_available,
            source_kind=front_source_kind,
            sampling=d2nn_sampling,
        )
        add(
            fields,
            "encoder_grid",
            encoder_grid,
            upstream="encoder_input",
            operation=(
                "coherent_relay_alias"
                if self.physics.front_path_model == "coherent_imaging_4f"
                else "physical_aperture_sampling"
            ),
            phase_available=front_phase_available,
            source_kind=front_source_kind,
            sampling=d2nn_sampling,
        )
        encoder_source_kind = (
            "propagated_complex_field"
            if front_phase_available
            else "downstream_of_intensity_relay"
        )
        previous = "encoder_grid"
        for layer_index in range(int(encoder_layer_fields.shape[1])):
            name = f"encoder_layer_{layer_index + 1}"
            add(
                fields,
                name,
                encoder_layer_fields[:, layer_index],
                upstream=previous,
                operation="encoder_diffractive_layer",
                phase_available=front_phase_available,
                source_kind=encoder_source_kind,
                sampling=d2nn_sampling,
            )
            previous = name
        add(
            fields,
            "encoder_output",
            encoded_field,
            upstream=previous,
            operation="encoder_output",
            phase_available=front_phase_available,
            source_kind=encoder_source_kind,
            sampling=d2nn_sampling,
        )
        add(
            fields,
            "ols_input",
            encoded_field,
            upstream="encoder_output",
            operation="ols_input_alias",
            phase_available=front_phase_available,
            source_kind=encoder_source_kind,
            sampling=d2nn_sampling,
        )
        add(
            fields,
            "ols_output",
            ols_field,
            upstream="ols_input",
            operation=f"complex_ols:{self.config.ols_kind}",
            phase_available=front_phase_available,
            source_kind=encoder_source_kind,
            sampling=d2nn_sampling,
        )
        if target_active:
            decoder_source_kind = "target_code_intervention"
        elif bridge_active:
            decoder_source_kind = "target_code_bridge_intervention"
        elif front_phase_available:
            decoder_source_kind = "propagated_complex_field"
        else:
            decoder_source_kind = "downstream_of_intensity_relay"
        add(
            fields,
            "decoder_input",
            decoder_input_field,
            upstream="ols_output",
            operation="decoder_input",
            phase_available=decoder_phase_available,
            source_kind=decoder_source_kind,
            sampling=d2nn_sampling,
        )
        previous = "decoder_input"
        for layer_index in range(int(decoder_layer_fields.shape[1])):
            name = f"decoder_layer_{layer_index + 1}"
            add(
                fields,
                name,
                decoder_layer_fields[:, layer_index],
                upstream=previous,
                operation="decoder_diffractive_layer",
                phase_available=decoder_phase_available,
                source_kind=decoder_source_kind,
                sampling=d2nn_sampling,
            )
            previous = name
        add(
            fields,
            "decoder_output",
            chip_grid,
            upstream=previous,
            operation="decoder_output",
            phase_available=decoder_phase_available,
            source_kind=decoder_source_kind,
            sampling=d2nn_sampling,
        )
        add(
            fields,
            "chip",
            chip_grid,
            upstream="decoder_output",
            operation="chip_plane",
            phase_available=decoder_phase_available,
            source_kind=decoder_source_kind,
            sampling=d2nn_sampling,
        )
        if bool(raw_optical_readout.get("camera_field_available", False)):
            real = raw_optical_readout.get("camera_field_real")
            imag = raw_optical_readout.get("camera_field_imag")
            if torch.is_tensor(real) and torch.is_tensor(imag):
                add(
                    fields,
                    "camera_field",
                    torch.complex(real, imag),
                    upstream="chip",
                    operation="coherent_camera_propagation",
                    phase_available=decoder_phase_available,
                    source_kind=decoder_source_kind,
                    sampling=raw_optical_readout.get("camera_sampling"),
                )
        return fields


def validate_one_shot_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    model_version = payload.get("model_version")
    if model_version != ONE_SHOT_MODEL_VERSION:
        raise ValueError("Incompatible one-shot checkpoint: unexpected model_version.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Incompatible one-shot checkpoint: missing model_state_dict.")


def _phase_slice_compatible_state(
    target: torch.Tensor, value: torch.Tensor
) -> torch.Tensor | None:
    if target.ndim < 1 or value.ndim != target.ndim:
        return None
    if tuple(target.shape[1:]) != tuple(value.shape[1:]):
        return None
    shared_layers = min(int(target.shape[0]), int(value.shape[0]))
    if shared_layers <= 0:
        return None
    sliced = target.detach().clone()
    sliced[:shared_layers].copy_(
        value[:shared_layers].to(device=target.device, dtype=target.dtype)
    )
    return sliced


def _collect_compatible_checkpoint_state(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    compatible_state = {}
    skipped_keys = []
    sliced_keys = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            compatible_state[key] = value
        elif target is not None and key.endswith(".phase"):
            sliced_value = _phase_slice_compatible_state(target, value)
            if sliced_value is not None:
                compatible_state[key] = sliced_value
                sliced_keys.append(key)
            else:
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)
    return compatible_state, skipped_keys, sliced_keys


def _load_compatible_checkpoint_state(
    model: OneShotOpticalGenerator,
    compatible_state: Mapping[str, torch.Tensor],
    *,
    description: str,
) -> tuple[list[str], list[str]]:
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            compatible_state, strict=False
        )
    except RuntimeError as exc:
        raise ValueError(
            f"Incompatible {description}: state dict does not match model config."
        ) from exc
    return missing_keys, unexpected_keys


def _checkpoint_load_summary(
    compatible_state: Mapping[str, torch.Tensor],
    sliced_keys: list[str],
    skipped_keys: list[str],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> dict[str, int]:
    return {
        "loaded": len(compatible_state),
        "sliced": len(sliced_keys),
        "skipped": len(skipped_keys),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
    }


def _print_partial_checkpoint_load(
    description: str, summary: Mapping[str, int]
) -> None:
    print(
        f"partially loaded {description}: "
        f"loaded={summary['loaded']} sliced={summary['sliced']} skipped={summary['skipped']} "
        f"missing={summary['missing']} unexpected={summary['unexpected']}"
    )


def _validate_optical_tolerant_full_checkpoint_load(
    description: str,
    skipped_keys: list[str],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    optical_prefixes = ("encoder.", "decoder.", "ols_phase_screen", "ols_amplitude_log")
    unexpected_missing = [
        key for key in missing_keys if not key.startswith(optical_prefixes)
    ]
    unexpected_skipped = [
        key for key in skipped_keys if not key.startswith(optical_prefixes)
    ]
    unexpected_loaded = [
        key for key in unexpected_keys if not key.startswith(optical_prefixes)
    ]
    if unexpected_missing or unexpected_skipped or unexpected_loaded:
        raise ValueError(
            f"Incompatible {description}: full checkpoint load requires an exact model match "
            f"(missing={len(unexpected_missing)} unexpected={len(unexpected_loaded)} "
            f"skipped={len(unexpected_skipped)})."
        )


def _validate_exact_checkpoint_load(
    description: str, summary: Mapping[str, int]
) -> None:
    if summary["skipped"] or summary["missing"] or summary["unexpected"]:
        raise ValueError(
            f"Incompatible {description}: full checkpoint load requires an exact model match "
            f"(loaded={summary['loaded']} skipped={summary['skipped']} "
            f"missing={summary['missing']} unexpected={summary['unexpected']})."
        )


def _validate_partial_checkpoint_load(
    description: str,
    include_prefixes: tuple[str, ...] | None,
    expected_missing_prefixes: tuple[str, ...],
    expected_skipped_prefixes: tuple[str, ...],
    skipped_keys: list[str],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    unexpected_missing = [
        key
        for key in missing_keys
        if include_prefixes is not None
        and key.startswith(include_prefixes)
        and not key.startswith(expected_missing_prefixes)
    ]
    unexpected_skipped = [
        key for key in skipped_keys if not key.startswith(expected_skipped_prefixes)
    ]
    if unexpected_missing or unexpected_skipped or unexpected_keys:
        raise ValueError(
            f"Incompatible {description}: partial checkpoint load found unexpected keys "
            f"(missing={unexpected_missing[:5]} skipped={unexpected_skipped[:5]} "
            f"unexpected={unexpected_keys[:5]})."
        )


def _load_one_shot_checkpoint_state(
    model: OneShotOpticalGenerator,
    checkpoint_state: Mapping[str, torch.Tensor],
    *,
    include_prefixes: tuple[str, ...] | None = None,
    description: str = "one-shot checkpoint",
    allow_partial: bool = False,
    expected_missing_prefixes: tuple[str, ...] = (),
    expected_skipped_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    model_state = model.state_dict()
    if include_prefixes is None and not allow_partial:
        compatible_state, skipped_keys, sliced_keys = (
            _collect_compatible_checkpoint_state(model_state, checkpoint_state)
        )
        missing_keys, unexpected_keys = _load_compatible_checkpoint_state(
            model, compatible_state, description=description
        )
        _validate_optical_tolerant_full_checkpoint_load(
            description, skipped_keys, missing_keys, unexpected_keys
        )
        summary = _checkpoint_load_summary(
            compatible_state, sliced_keys, skipped_keys, missing_keys, unexpected_keys
        )
        if skipped_keys or missing_keys or unexpected_keys:
            _print_partial_checkpoint_load(description, summary)
        return summary

    filtered_state = {
        key: value
        for key, value in checkpoint_state.items()
        if include_prefixes is None or key.startswith(include_prefixes)
    }
    if not filtered_state:
        raise ValueError(
            f"Incompatible {description}: no checkpoint state entries match prefixes {include_prefixes}."
        )
    compatible_state, skipped_keys, sliced_keys = _collect_compatible_checkpoint_state(
        model_state, filtered_state
    )
    if not compatible_state:
        raise ValueError(
            f"Incompatible {description}: no state dict entries match the current model config."
        )
    missing_keys, unexpected_keys = _load_compatible_checkpoint_state(
        model, compatible_state, description=description
    )
    summary = _checkpoint_load_summary(
        compatible_state, sliced_keys, skipped_keys, missing_keys, unexpected_keys
    )
    if not allow_partial:
        _validate_exact_checkpoint_load(description, summary)
    else:
        _validate_partial_checkpoint_load(
            description,
            include_prefixes,
            expected_missing_prefixes,
            expected_skipped_prefixes,
            skipped_keys,
            missing_keys,
            unexpected_keys,
        )
    if skipped_keys or missing_keys or unexpected_keys:
        _print_partial_checkpoint_load(description, summary)
    return summary


def load_one_shot_checkpoint(
    model: OneShotOpticalGenerator, payload: Mapping[str, Any]
) -> None:
    validate_one_shot_checkpoint_payload(payload)
    checkpoint_state = payload["model_state_dict"]
    _load_one_shot_checkpoint_state(
        model, checkpoint_state, description="one-shot checkpoint"
    )


def load_one_shot_checkpoint_subset(
    model: OneShotOpticalGenerator,
    payload: Mapping[str, Any],
    *,
    include_prefixes: tuple[str, ...],
    description: str,
    expected_missing_prefixes: tuple[str, ...] = (),
    expected_skipped_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    validate_one_shot_checkpoint_payload(payload)
    checkpoint_state = payload["model_state_dict"]
    return _load_one_shot_checkpoint_state(
        model,
        checkpoint_state,
        include_prefixes=include_prefixes,
        description=description,
        allow_partial=True,
        expected_missing_prefixes=expected_missing_prefixes,
        expected_skipped_prefixes=expected_skipped_prefixes,
    )
