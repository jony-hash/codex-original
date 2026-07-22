"""Neutral implementation of the diagnostic semantic-command equation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

V73_SEMANTIC_AMPLITUDE = 0.4027685085311532
V73_BACKGROUND_FLOOR = 0.05


def semantic_basis(size: int = 200, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.linspace(-torch.pi, torch.pi, size, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    basis = torch.stack((torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy), torch.sin(xx + yy), torch.cos(xx - yy), torch.sin(2.0 * xx + yy), torch.cos(xx + 2.0 * yy)), dim=0)
    basis = F.avg_pool2d(basis[:, None], kernel_size=7, stride=1, padding=3)[:, 0]
    return basis / basis.flatten(1).abs().amax(1).clamp_min(1.0e-6)[:, None, None]


def project_residual_to_basis(residual_logits: torch.Tensor, basis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if residual_logits.ndim != 4 or residual_logits.shape[1] != 1:
        raise ValueError("residual_logits must have shape [B,1,H,W]")
    basis = semantic_basis(residual_logits.shape[-1], dtype=residual_logits.dtype, device=residual_logits.device) if basis is None else basis.to(device=residual_logits.device, dtype=residual_logits.dtype)
    if tuple(basis.shape) != (8, residual_logits.shape[-2], residual_logits.shape[-1]):
        raise ValueError("basis must have shape [8,H,W] matching residual_logits")
    flat_basis = basis.flatten(1)
    coefficients = residual_logits[:, 0].flatten(1) @ flat_basis.T / flat_basis.square().sum(1).clamp_min(1.0e-12)
    return torch.einsum("bk,khw->bhw", coefficients, basis)[:, None], coefficients


def semantic_command_residual(template_logits: torch.Tensor, residual_logits: torch.Tensor, *, amplitude: float = V73_SEMANTIC_AMPLITUDE, background_floor: float = V73_BACKGROUND_FLOOR, basis: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if template_logits.shape != residual_logits.shape:
        raise ValueError("template_logits and residual_logits must have equal shapes")
    base_intensity = torch.sigmoid(template_logits)
    base_command = ((base_intensity.sqrt() - background_floor) / (1.0 - background_floor)).clamp(0.0, 1.0)
    pattern, coefficients = project_residual_to_basis(residual_logits, basis)
    support = base_intensity.detach()
    weighted_mean = (support * pattern).flatten(1).sum(1, keepdim=True) / support.flatten(1).sum(1, keepdim=True).clamp_min(1.0e-8)
    centered = support * (pattern - weighted_mean[:, :, None, None])
    max_abs = centered.flatten(1).abs().amax(1, keepdim=True).clamp_min(1.0e-8)
    semantic_pattern = centered / max_abs[:, :, None, None]
    command_delta = float(amplitude) * semantic_pattern
    command = (base_command.detach() + command_delta).clamp(0.0, 1.0)
    return command, {
        "generator_basis_coefficients": coefficients,
        "generator_semantic_pattern": semantic_pattern,
        "generator_command_delta": command_delta,
        "generator_support_energy": (support * command_delta.square()).flatten(1).sum(1),
        "generator_frequency_energy": coefficients.square(),
        "generator_command_clipping": ((command <= 0.0) | (command >= 1.0)).float().flatten(1).mean(1),
        "generator_weighted_mean": weighted_mean,
        "generator_base_command": base_command,
        "generator_desired_intensity": base_intensity,
        "generator_residual_map": semantic_pattern,
        "generator_code_delta_logits": residual_logits,
    }
