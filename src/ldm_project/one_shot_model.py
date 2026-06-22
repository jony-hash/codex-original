from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .config import OneShotConfig, PhysicsConfig
from .one_shot_optical_layers import (
    ElectronicGeneratorE,
    IdealAmplitudeSLMModel,
    PhaseAmplitudeDiffractiveStack,
    PhaseOnlyDiffractiveStack,
    _apply_complex_ols,
    _flatten_complex_field,
    _resize_complex_field,
    _resize_real_intensity,
    _vector_to_complex_field,
)
from .optics import BackOpticalPath, FrontOpticalPath, normalize_real_intensity


ONE_SHOT_MODEL_VERSION = "ldm_distilled_single_shot_optical_student_v3"


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
        self.uses_direct_camera_core = self.config.optical_core_arch == "camera_direct_clean_oracle"
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
            amplitude_kwargs = (
                {"amplitude_logit_init": d2nn["amplitude_logit_init"]}
                if d2nn["modulation"] == "phase_amplitude"
                else {}
            )
            stack_cls = (
                PhaseAmplitudeDiffractiveStack
                if d2nn["modulation"] == "phase_amplitude"
                else PhaseOnlyDiffractiveStack
            )
            self.encoder = stack_cls(
                d2nn["encoder_layers"],
                d2nn["grid_size"],
                d2nn["layer_spacing"],
                d2nn["wavelength"],
                d2nn["pixel_size"],
                d2nn["propagation_grid_size"],
                d2nn["phase_init"],
                **amplitude_kwargs,
            )
            self.decoder = stack_cls(
                d2nn["decoder_layers"],
                d2nn["grid_size"],
                d2nn["layer_spacing"],
                d2nn["wavelength"],
                d2nn["pixel_size"],
                d2nn["propagation_grid_size"],
                d2nn["phase_init"],
                **amplitude_kwargs,
            )
        elif self.uses_direct_camera_core:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        else:
            grid_features = 2 * self.config.metasurface_grid_size * self.config.metasurface_grid_size
            self.encoder = nn.Linear(grid_features, grid_features, bias=False)
            self.decoder = nn.Linear(grid_features, grid_features, bias=False)
        self._d2nn = d2nn
        self.front_path = FrontOpticalPath(physics)
        self.back_path = BackOpticalPath(physics, propagation_method="angular_spectrum")
        self._initialize_layers()

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

    def _validate_config(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.config.code_dim <= 0:
            raise ValueError(f"one_shot.code_dim must be positive, got {self.config.code_dim}")
        if self.config.label_embed_dim <= 0:
            raise ValueError(
                f"one_shot.label_embed_dim must be positive, got {self.config.label_embed_dim}"
            )
        if self.config.hidden_dim <= 0:
            raise ValueError(f"one_shot.hidden_dim must be positive, got {self.config.hidden_dim}")
        if self.config.num_classes <= 0:
            raise ValueError(f"one_shot.num_classes must be positive, got {self.config.num_classes}")
        if self.config.generator_e_arch not in {"label_template", "mlp", "template_residual_mlp"}:
            raise ValueError(
                "Unsupported one_shot.generator_e_arch: "
                f"{self.config.generator_e_arch}; expected label_template, mlp, or template_residual_mlp."
            )
        if self.config.generator_command_transform not in {
            "direct_amplitude",
            "intensity_to_amplitude",
        }:
            raise ValueError(
                "Unsupported one_shot.generator_command_transform: "
                f"{self.config.generator_command_transform}"
            )
        if self.config.generator_template_init not in {"none", "mnist_prototype"}:
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
        if self.config.code_residual_scale < 0.0:
            raise ValueError(
                "one_shot.code_residual_scale must be non-negative, "
                f"got {self.config.code_residual_scale}"
            )
        if self.config.ols_kind not in {"pointwise_tanh", "tanh", "gelu", "silu", "relu", "identity"}:
            raise ValueError(
                "Unsupported one_shot.ols_kind: "
                f"{self.config.ols_kind}; expected gelu, tanh, silu, relu, or identity."
            )
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
            raise ValueError(f"one_shot.layer_spacing must be positive, got {self.config.layer_spacing}")
        if self.uses_diffractive_core:
            if self.config.d2nn is not None:
                d2nn_cfg = self.config.d2nn
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
                expected_modulation = (
                    "phase_amplitude"
                    if self.config.optical_core_arch == "phase_amplitude_d2nn"
                    else "phase_only"
                )
                if d2nn_cfg.modulation != expected_modulation:
                    raise ValueError(
                        "one_shot.optical_core_arch and one_shot.d2nn.modulation disagree: "
                        f"{self.config.optical_core_arch} requires {expected_modulation}, "
                        f"got {d2nn_cfg.modulation}."
                    )
                if d2nn_cfg.grid_size <= 0:
                    raise ValueError(
                        "one_shot.d2nn.grid_size must be positive, "
                        f"got {d2nn_cfg.grid_size}"
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
            else:
                if self.config.diffractive_grid_size <= 0:
                    raise ValueError(
                        "one_shot.diffractive_grid_size must be positive, "
                        f"got {self.config.diffractive_grid_size}"
                    )
                if self.config.diffractive_propagation_grid_size < self.config.diffractive_grid_size:
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
                if self.config.diffractive_phase_init not in {"uniform_pi", "uniform_2pi", "small"}:
                    raise ValueError(
                        "Unsupported one_shot.diffractive_phase_init: "
                        f"{self.config.diffractive_phase_init}; expected uniform_pi, uniform_2pi, or small."
                    )

    def _initialize_layers(self) -> None:
        if not self.uses_diffractive_core and not self.uses_direct_camera_core:
            nn.init.xavier_uniform_(self.encoder.weight, gain=0.5)
            nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)

    def forward(self, code: torch.Tensor, class_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        slm_command, generator_diagnostics = self.generator_e.forward_with_diagnostics(code, class_labels)
        slm_outputs = self.slm(slm_command)
        if self.uses_direct_camera_core:
            return {
                **self._forward_direct_camera(slm_outputs),
                **generator_diagnostics,
            }
        if self.uses_diffractive_core:
            return {
                **self._forward_diffractive(slm_outputs),
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
        encoded_field = _vector_to_complex_field(encoded_vector, self.config.metasurface_grid_size)
        ols_field = _apply_complex_ols(encoded_field, self.config.ols_kind)
        decoded_vector = self.decoder(_flatten_complex_field(ols_field))
        chip_grid = _vector_to_complex_field(decoded_vector, self.config.metasurface_grid_size)
        chip_field = _resize_complex_field(chip_grid, (self.image_size, self.image_size))
        optical_readout = self.back_path(chip_field)
        return {
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
            "decoder_input_amplitude": ols_field.abs(),
            "decoder_input_phase": torch.angle(ols_field),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "chip_real": chip_field.real,
            "chip_imag": chip_field.imag,
            **optical_readout,
            **generator_diagnostics,
        }

    def _forward_direct_camera(self, slm_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
            "chip_real": camera_complex.real,
            "chip_imag": camera_complex.imag,
            "chip_amplitude": camera_plane_loss,
            "chip_phase": camera_phase,
            "camera_intensity_raw": camera_plane_loss,
            "probe_plane": camera_plane_loss,
            "camera_phase": camera_phase,
        }

    def _forward_diffractive(self, slm_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d2nn = self._d2nn
        slm_transfer = slm_outputs["slm_transfer"]
        encoder_input_field = self.front_path.forward_field(
            slm_transfer,
            propagation_method="angular_spectrum",
        )
        encoder_grid = _resize_complex_field(
            encoder_input_field,
            (d2nn["grid_size"], d2nn["grid_size"]),
        )
        encoded_field, encoder_layer_fields, encoder_crop_diagnostics = self.encoder(encoder_grid)
        ols_field = _apply_complex_ols(encoded_field, self.config.ols_kind)
        chip_grid, decoder_layer_fields, decoder_crop_diagnostics = self.decoder(ols_field)
        raw_optical_readout = self.back_path(chip_grid)
        camera_intensity = _resize_real_intensity(
            raw_optical_readout["camera_intensity_raw"],
            (self.image_size, self.image_size),
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
        selected_plane = selected_plane_loss.clamp(0.0, 1.0)
        optical_readout = {
            **raw_optical_readout,
            "camera_plane": selected_plane,
            "camera_plane_loss": selected_plane_loss,
        }
        encoder_phase_transfer = self.encoder.phase_transfers()
        decoder_phase_transfer = self.decoder.phase_transfers()
        return {
            "x_hat": selected_plane,
            "camera_plane": selected_plane,
            "camera_plane_loss": selected_plane_loss,
            "primary_readout_plane": primary_plane,
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
            "decoder_input_amplitude": ols_field.abs(),
            "decoder_input_phase": torch.angle(ols_field),
            "decoder_output_amplitude": chip_grid.abs(),
            "decoder_output_phase": torch.angle(chip_grid),
            "chip_real": chip_grid.real,
            "chip_imag": chip_grid.imag,
            "chip_intensity_primary_loss": chip_intensity_loss,
            "chip_intensity_primary": chip_intensity_loss.clamp(0.0, 1.0),
            "chip_amplitude_primary_loss": chip_amplitude_loss,
            "chip_amplitude_primary": chip_amplitude_loss.clamp(0.0, 1.0),
            "physical_camera_plane": physical_camera_plane,
            "physical_camera_plane_loss": camera_plane_loss,
            "physical_camera_intensity_raw": raw_optical_readout["camera_intensity_raw"],
            "diffractive_camera_intensity_raw": raw_optical_readout["camera_intensity_raw"],
            "diffractive_camera_plane_raw": raw_optical_readout["camera_plane"],
            "diffractive_camera_plane_loss_raw": raw_optical_readout["camera_plane_loss"],
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


def validate_one_shot_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    model_version = payload.get("model_version")
    if model_version != ONE_SHOT_MODEL_VERSION:
        raise ValueError("Incompatible one-shot checkpoint: unexpected model_version.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Incompatible one-shot checkpoint: missing model_state_dict.")


def _phase_slice_compatible_state(target: torch.Tensor, value: torch.Tensor) -> torch.Tensor | None:
    if target.ndim < 1 or value.ndim != target.ndim:
        return None
    if tuple(target.shape[1:]) != tuple(value.shape[1:]):
        return None
    if value.shape[0] < target.shape[0]:
        return None
    sliced = target.detach().clone()
    sliced[: target.shape[0]].copy_(value[: target.shape[0]].to(device=target.device, dtype=target.dtype))
    return sliced


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
        try:
            missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
        except RuntimeError as exc:
            raise ValueError(f"Incompatible {description}: state dict does not match model config.") from exc
        optical_prefixes = ("encoder.", "decoder.")
        unexpected_missing = [key for key in missing_keys if not key.startswith(optical_prefixes)]
        unexpected_skipped = [key for key in skipped_keys if not key.startswith(optical_prefixes)]
        unexpected_loaded = [key for key in unexpected_keys if not key.startswith(optical_prefixes)]
        if unexpected_missing or unexpected_skipped or unexpected_loaded:
            raise ValueError(
                f"Incompatible {description}: full checkpoint load requires an exact model match "
                f"(missing={len(unexpected_missing)} unexpected={len(unexpected_loaded)} "
                f"skipped={len(unexpected_skipped)})."
            )
        if skipped_keys or missing_keys or unexpected_keys:
            print(
                f"partially loaded {description}: "
                f"loaded={len(compatible_state)} sliced={len(sliced_keys)} skipped={len(skipped_keys)} "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
            )
        return {
            "loaded": len(compatible_state),
            "sliced": len(sliced_keys),
            "skipped": len(skipped_keys),
            "missing": len(missing_keys),
            "unexpected": len(unexpected_keys),
        }

    filtered_state = {
        key: value
        for key, value in checkpoint_state.items()
        if include_prefixes is None or key.startswith(include_prefixes)
    }
    if not filtered_state:
        raise ValueError(
            f"Incompatible {description}: no checkpoint state entries match prefixes {include_prefixes}."
        )
    compatible_state = {}
    skipped_keys = []
    sliced_keys = []
    for key, value in filtered_state.items():
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
    if not compatible_state:
        raise ValueError(
            f"Incompatible {description}: no state dict entries match the current model config."
        )
    try:
        missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    except RuntimeError as exc:
        raise ValueError(f"Incompatible {description}: state dict does not match model config.") from exc
    summary = {
        "loaded": len(compatible_state),
        "sliced": len(sliced_keys),
        "skipped": len(skipped_keys),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
    }
    if not allow_partial:
        if skipped_keys or missing_keys or unexpected_keys:
            raise ValueError(
                f"Incompatible {description}: full checkpoint load requires an exact model match "
                f"(loaded={summary['loaded']} skipped={summary['skipped']} "
                f"missing={summary['missing']} unexpected={summary['unexpected']})."
            )
    else:
        unexpected_missing = [
            key
            for key in missing_keys
            if include_prefixes is not None
            and key.startswith(include_prefixes)
            and not key.startswith(expected_missing_prefixes)
        ]
        unexpected_skipped = [
            key
            for key in skipped_keys
            if not key.startswith(expected_skipped_prefixes)
        ]
        if unexpected_missing or unexpected_skipped or unexpected_keys:
            raise ValueError(
                f"Incompatible {description}: partial checkpoint load found unexpected keys "
                f"(missing={unexpected_missing[:5]} skipped={unexpected_skipped[:5]} "
                f"unexpected={unexpected_keys[:5]})."
            )
    if skipped_keys or missing_keys or unexpected_keys:
        print(
            f"partially loaded {description}: "
            f"loaded={summary['loaded']} sliced={summary['sliced']} skipped={summary['skipped']} "
            f"missing={summary['missing']} unexpected={summary['unexpected']}"
        )
    return summary


def load_one_shot_checkpoint(model: OneShotOpticalGenerator, payload: Mapping[str, Any]) -> None:
    validate_one_shot_checkpoint_payload(payload)
    checkpoint_state = payload["model_state_dict"]
    _load_one_shot_checkpoint_state(model, checkpoint_state, description="one-shot checkpoint")


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
