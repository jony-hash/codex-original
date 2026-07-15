"""Canonical electronic code/label to amplitude-command generator."""
from __future__ import annotations

import torch
from torch import nn


class ElectronicGeneratorE(nn.Module):
    """Generate a normalized electronic amplitude command from code and label.

    The output is an electronic command in ``[0, 1]``.  Conversion to the
    physical complex SLM transfer remains the responsibility of the optics
    device layer; keeping that boundary explicit prevents command values from
    being confused with optical amplitude.
    """

    def __init__(
        self,
        image_size: int,
        grid_size: int,
        code_dim: int,
        num_classes: int,
        label_embed_dim: int,
        hidden_dim: int,
        arch: str = "mlp",
        code_residual_scale: float = 0.0,
        command_transform: str = "direct_amplitude",
        background_floor: float = 0.05,
        template_init: str = "none",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.num_classes = num_classes
        self.arch = arch
        self.code_residual_scale = float(code_residual_scale)
        self.command_transform = command_transform
        self.background_floor = float(background_floor)
        self.template_init = template_init
        if self.arch not in {
            "label_template",
            "label_template_code_residual",
            "mlp",
            "template_residual_mlp",
        }:
            raise ValueError(
                "Unsupported one_shot.generator_e_arch: "
                f"{self.arch}; expected label_template, label_template_code_residual, "
                "mlp, or template_residual_mlp."
            )
        if self.command_transform not in {"direct_amplitude", "intensity_to_amplitude"}:
            raise ValueError(
                f"Unsupported generator command transform: {self.command_transform}"
            )
        if (
            self.command_transform == "intensity_to_amplitude"
            and self.background_floor >= 1.0
        ):
            raise ValueError("intensity_to_amplitude requires background_floor < 1.0")
        if self.arch in {"label_template", "label_template_code_residual"}:
            if self.template_init != "mnist_prototype":
                raise ValueError(
                    f"{self.arch} requires template_init='mnist_prototype'"
                )
            self.template_logits = nn.Parameter(
                torch.zeros(num_classes, 1, image_size, image_size)
            )
            self.label_embedding = None
            self.net = None
            self.residual_net = None
        else:
            self.register_parameter("template_logits", None)
            self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
            self.net = nn.Sequential(
                nn.Linear(code_dim + label_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, image_size * image_size),
            )
            self.residual_net = None
        if self.arch == "label_template_code_residual":
            self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
            self.residual_net = nn.Sequential(
                nn.Linear(code_dim + label_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, image_size * image_size),
            )
        if self.arch == "template_residual_mlp":
            self.residual_net = nn.Sequential(
                nn.Linear(code_dim + label_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, image_size * image_size),
            )

    def initialize_label_templates(self, intensity_bank: torch.Tensor) -> None:
        if (
            self.arch not in {"label_template", "label_template_code_residual"}
            or self.template_logits is None
        ):
            raise ValueError(
                "Label templates can only initialize generator_e_arch='label_template' "
                "or 'label_template_code_residual'."
            )
        expected = (self.num_classes, 1, self.image_size, self.image_size)
        if tuple(intensity_bank.shape) != expected:
            raise ValueError(
                f"label template bank must have shape {expected}, got {tuple(intensity_bank.shape)}"
            )
        bank = intensity_bank.detach().to(
            device=self.template_logits.device,
            dtype=self.template_logits.dtype,
        )
        if not torch.isfinite(bank).all():
            raise ValueError("label template bank must contain only finite values")
        eps = torch.finfo(bank.dtype).eps
        stable = bank.clamp(eps, 1.0 - eps)
        with torch.no_grad():
            self.template_logits.copy_(torch.logit(stable))

    def _validate_inputs(
        self, code: torch.Tensor, class_labels: torch.Tensor
    ) -> torch.Tensor:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError(
                f"code must have shape [batch, {self.code_dim}], got {tuple(code.shape)}"
            )
        labels = class_labels.long()
        if labels.ndim != 1 or labels.shape[0] != code.shape[0]:
            raise ValueError(
                "class_labels must be a 1D tensor with the same batch size as code."
            )
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(f"class_labels must be in [0, {self.num_classes - 1}].")
        return labels

    def _features(self, code: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        labels = self._validate_inputs(code, class_labels)
        if self.label_embedding is None:
            raise ValueError("label_template does not use MLP features")
        return torch.cat([code, self.label_embedding(labels)], dim=1)

    def _to_amplitude_command(self, desired_intensity: torch.Tensor) -> torch.Tensor:
        if self.command_transform == "direct_amplitude":
            return desired_intensity
        amplitude = desired_intensity.clamp_min(0.0).sqrt()
        return ((amplitude - self.background_floor) / (1.0 - self.background_floor)).clamp(0.0, 1.0)

    def _diagnostic_metadata(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "generator_command_transform": reference.new_tensor(
                1 if self.command_transform == "intensity_to_amplitude" else 0,
                dtype=torch.int64,
            ),
            "generator_code_conditioning_active": reference.new_tensor(
                self.arch != "label_template", dtype=torch.bool
            ),
        }

    def forward_with_diagnostics(
        self, code: torch.Tensor, class_labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.arch in {"label_template", "label_template_code_residual"}:
            labels = self._validate_inputs(code, class_labels)
            if self.template_logits is None:
                raise RuntimeError(f"{self.arch} generator is missing template logits")
            template_logits = self.template_logits.index_select(0, labels)
            base_intensity = torch.sigmoid(template_logits)
            if self.arch == "label_template":
                desired_intensity = base_intensity
                residual_map = torch.zeros_like(desired_intensity)
            else:
                if self.residual_net is None:
                    raise RuntimeError(
                        "label_template_code_residual generator is missing residual_net"
                    )
                residual_logits = self.residual_net(self._features(code, class_labels))
                residual_map = torch.tanh(
                    residual_logits.reshape(code.shape[0], 1, self.image_size, self.image_size)
                )
                desired_intensity = torch.sigmoid(
                    template_logits + self.code_residual_scale * residual_map
                )
            command = self._to_amplitude_command(desired_intensity)
            base_command = self._to_amplitude_command(base_intensity)
            background_mask = (base_intensity.detach() < 0.2).float()
            bg_abs = (residual_map.abs() * background_mask).flatten(1).sum(dim=1) / background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
            return command, {
                "generator_base_command": base_command,
                "generator_desired_intensity": desired_intensity,
                "generator_residual_map": residual_map,
                "generator_residual_abs_mean": residual_map.abs().flatten(1).mean(dim=1),
                "generator_residual_bg_abs_mean": bg_abs,
                **self._diagnostic_metadata(command),
            }
        features = self._features(code, class_labels)
        if self.net is None:
            raise RuntimeError("MLP generator is missing its network")
        base_logits = self.net(features)
        base_grid = torch.sigmoid(
            base_logits.reshape(code.shape[0], 1, self.image_size, self.image_size)
        )
        if self.residual_net is None:
            zero_residual = torch.zeros_like(base_grid)
            command = self._to_amplitude_command(base_grid)
            diagnostics = {
                "generator_base_command": command,
                "generator_residual_map": zero_residual,
                "generator_residual_abs_mean": zero_residual.flatten(1).mean(dim=1),
                "generator_residual_bg_abs_mean": zero_residual.flatten(1).mean(dim=1),
                **self._diagnostic_metadata(command),
            }
            if self.command_transform == "intensity_to_amplitude":
                diagnostics["generator_desired_intensity"] = base_grid
            return command, diagnostics
        residual_logits = self.residual_net(features)
        residual_map = torch.tanh(
            residual_logits.reshape(code.shape[0], 1, self.image_size, self.image_size)
        )
        slm_logits = base_logits.reshape(
            code.shape[0], 1, self.image_size, self.image_size
        )
        desired_intensity = torch.sigmoid(
            slm_logits + self.code_residual_scale * residual_map
        )
        command = self._to_amplitude_command(desired_intensity)
        background_mask = (base_grid.detach() < 0.2).float()
        bg_abs = (residual_map.abs() * background_mask).flatten(1).sum(dim=1) / background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
        diagnostics = {
            "generator_base_command": self._to_amplitude_command(base_grid),
            "generator_residual_map": residual_map,
            "generator_residual_abs_mean": residual_map.abs().flatten(1).mean(dim=1),
            "generator_residual_bg_abs_mean": bg_abs,
            **self._diagnostic_metadata(command),
        }
        if self.command_transform == "intensity_to_amplitude":
            diagnostics["generator_desired_intensity"] = desired_intensity
        return command, diagnostics

    def forward(self, code: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        command, _ = self.forward_with_diagnostics(code, class_labels)
        return command


__all__ = ["ElectronicGeneratorE"]
