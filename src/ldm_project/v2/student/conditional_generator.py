"""The v2 formal generator: label prototype plus a frozen capacity-approved basis."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .capacity import CodeBasisArtifactV2


@dataclass(frozen=True)
class GeneratorDiagnostics:
    unclamped_command: torch.Tensor
    coefficients: torch.Tensor
    clipping_fraction: torch.Tensor
    clipping_penalty: torch.Tensor


class LabelPrototypeBasisGenerator(nn.Module):
    """No sample-dependent normalization and no template multiplication are permitted."""

    def __init__(self, *, code_dim: int, basis_artifact: CodeBasisArtifactV2, hidden_dim: int = 64, command_gain: float = 0.1) -> None:
        super().__init__()
        if basis_artifact.basis.shape[:2] != (10, 8):
            raise ValueError("formal v2 generator requires ten labels and an eight-dimensional basis")
        if code_dim <= 0 or hidden_dim <= 0 or command_gain <= 0.0:
            raise ValueError("invalid v2 generator dimensions")
        self.code_dim = code_dim
        self.command_gain = command_gain
        self.basis_artifact_hash = basis_artifact.artifact_hash
        self.template_logits = nn.Parameter(torch.zeros(10, *basis_artifact.basis.shape[-2:]))
        # Bias-free layers make the semantic zero code exactly the label prototype.
        self.coefficient_net = nn.Sequential(nn.Linear(code_dim, hidden_dim, bias=False), nn.SiLU(), nn.Linear(hidden_dim, 8, bias=False))
        self.register_buffer("basis", basis_artifact.basis.clone())
        self.register_buffer("label_normalizer", basis_artifact.normalizer.clone())

    def forward_with_diagnostics(self, code: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, GeneratorDiagnostics]:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("code must have shape [batch, code_dim]")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be a batch of integers in [0, 9]")
        template = torch.sigmoid(self.template_logits.index_select(0, labels))
        coefficients = torch.tanh(self.coefficient_net(code))
        selected_basis = self.basis.index_select(0, labels)
        delta = torch.einsum("bk,bkhw->bhw", coefficients, selected_basis)
        normalizer = self.label_normalizer.index_select(0, labels)
        unclamped = template + self.command_gain * delta / normalizer
        command = unclamped.clamp(0.0, 1.0).unsqueeze(1)
        clipping = ((unclamped < 0.0) | (unclamped > 1.0)).to(code.dtype).mean()
        clipping_penalty = (torch.relu(-unclamped).square() + torch.relu(unclamped - 1.0).square()).mean()
        return command, GeneratorDiagnostics(unclamped.unsqueeze(1), coefficients, clipping, clipping_penalty)

    def prototype_command(self, labels: torch.Tensor) -> torch.Tensor:
        """The immutable semantic baseline, independent of coefficient-net parameters."""
        if labels.ndim != 1 or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")
        return torch.sigmoid(self.template_logits.index_select(0, labels)).unsqueeze(1)

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.forward_with_diagnostics(code, labels)[0]


class ConditionalGlobalAmplitudeGenerator(nn.Module):
    """The teacher-free electronic generator that supplies the physical SLM.

    All stochasticity enters as a global amplitude command *before* the optical
    encoder.  The module deliberately has no decoder-side latent, sample-wise
    normalization, or template multiplication path.
    """

    architecture_kind = "conditional_global_amplitude_v1"

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (200, 200),
        code_dim: int = 8,
        hidden_dim: int = 64,
        command_gain: float = 0.1,
        class_carrier_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if shape[0] <= 0 or shape[1] <= 0 or code_dim != 8 or hidden_dim != 64 or not 0.0 < command_gain <= 0.5:
            raise ValueError("conditional generator requires 8D code, 64 hidden units, and a valid amplitude gain")
        self.shape = shape
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.command_gain = command_gain
        if class_carrier_logits is None:
            carrier = torch.zeros(10, *shape)
        else:
            if tuple(class_carrier_logits.shape) != (10, *shape):
                raise ValueError("class_carrier_logits must have shape [10, H, W]")
            carrier = class_carrier_logits.detach().clone()
        self.class_carrier_logits = nn.Parameter(carrier)
        self.global_modes = nn.Parameter(self._initial_global_modes(shape))
        # Both maps are bias-free, so z=0 maps exactly to the class carrier.
        self.code_projection = nn.Linear(code_dim, hidden_dim, bias=False)
        self.class_heads = nn.Parameter(torch.empty(10, code_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.code_projection.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.class_heads, a=5**0.5)

    @staticmethod
    def _initial_global_modes(shape: tuple[int, int]) -> torch.Tensor:
        """Eight full-aperture low-frequency Fourier directions, unit RMS."""
        height, width = shape
        yy = torch.linspace(-1.0, 1.0, height).view(height, 1)
        xx = torch.linspace(-1.0, 1.0, width).view(1, width)
        modes = torch.stack((
            torch.cos(torch.pi * xx).expand(height, width),
            torch.sin(torch.pi * xx).expand(height, width),
            torch.cos(torch.pi * yy).expand(height, width),
            torch.sin(torch.pi * yy).expand(height, width),
            torch.cos(torch.pi * (xx + yy)),
            torch.sin(torch.pi * (xx + yy)),
            torch.cos(torch.pi * (xx - yy)),
            torch.sin(torch.pi * (xx - yy)),
        ))
        return ConditionalGlobalAmplitudeGenerator.normalized_modes(modes)

    @staticmethod
    def normalized_modes(modes: torch.Tensor) -> torch.Tensor:
        if modes.ndim != 3:
            raise ValueError("global modes must have shape [code_dim, H, W]")
        centered = modes - modes.mean(dim=(-2, -1), keepdim=True)
        rms = centered.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8).sqrt()
        return centered / rms

    def coefficients(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(code, labels)
        hidden = torch.nn.functional.silu(self.code_projection(code))
        selected_heads = self.class_heads.index_select(0, labels)
        return torch.tanh(torch.einsum("bkh,bh->bk", selected_heads, hidden))

    def _validate_inputs(self, code: torch.Tensor, labels: torch.Tensor) -> None:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("code must have shape [batch, 8]")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.numel() == 0:
            raise ValueError("labels must have shape [batch]")
        if labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(code, labels)
        modes = self.normalized_modes(self.global_modes)
        residual = torch.einsum("bk,khw->bhw", coefficients, modes)
        logits = self.class_carrier_logits.index_select(0, labels) + self.command_gain * residual
        return torch.sigmoid(logits).unsqueeze(1)

    def mode_orthogonality_penalty(self) -> torch.Tensor:
        modes = self.normalized_modes(self.global_modes).flatten(1)
        gram = modes @ modes.T / modes.shape[1]
        return (gram - torch.eye(self.code_dim, device=modes.device, dtype=modes.dtype)).square().mean()

    def class_carrier_command(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1 or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")
        return torch.sigmoid(self.class_carrier_logits.index_select(0, labels)).unsqueeze(1)

class IdentityAnchoredConditionalAmplitudeGenerator(nn.Module):
    """Identity-anchored conditional generator for the v2 recovery path.

    The code enters the optical path only through the amplitude SLM command.
    The residual coefficient map is deliberately initialized as an identity
    anchor: ``Wy`` is zero at initialization, so ``c(z, y) == tanh(z)``.
    Both linear maps are bias-free and the command is never normalized per
    sample.  The class carrier and the global modes remain trainable, while
    the gain is fixed by the architecture contract at ``0.5``.
    """

    architecture_kind = "conditional_identity_logit_amplitude_v2"
    fixed_command_gain = 0.5

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (200, 200),
        code_dim: int = 8,
        hidden_dim: int = 64,
        command_gain: float = fixed_command_gain,
        class_carrier_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if shape[0] <= 0 or shape[1] <= 0 or code_dim != 8 or hidden_dim != 64 or command_gain != self.fixed_command_gain:
            raise ValueError("identity-anchored generator requires 8D code, 64 hidden units, and fixed gain 0.5")
        self.shape = shape
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.command_gain = self.fixed_command_gain
        if class_carrier_logits is None:
            carrier = torch.zeros(10, *shape)
        else:
            if tuple(class_carrier_logits.shape) != (10, *shape):
                raise ValueError("class_carrier_logits must have shape [10, H, W]")
            carrier = class_carrier_logits.detach().clone()
        self.class_carrier_logits = nn.Parameter(carrier)
        self.global_modes = nn.Parameter(self._initial_global_modes(shape))

        # Wz is bias-free by contract.  Wy starts at zero, making the initial
        # coefficient map exactly tanh(z) for every label.
        self.code_projection = nn.Linear(code_dim, hidden_dim, bias=False)
        self.class_heads = nn.Parameter(torch.zeros(10, code_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.code_projection.weight, a=5**0.5)

    @staticmethod
    def _initial_global_modes(shape: tuple[int, int]) -> torch.Tensor:
        """Eight full-aperture, low-frequency directions with unit RMS."""
        height, width = shape
        yy = torch.linspace(-1.0, 1.0, height).view(height, 1)
        xx = torch.linspace(-1.0, 1.0, width).view(1, width)
        modes = torch.stack((
            torch.cos(torch.pi * xx).expand(height, width),
            torch.sin(torch.pi * xx).expand(height, width),
            torch.cos(torch.pi * yy).expand(height, width),
            torch.sin(torch.pi * yy).expand(height, width),
            torch.cos(torch.pi * (xx + yy)),
            torch.sin(torch.pi * (xx + yy)),
            torch.cos(torch.pi * (xx - yy)),
            torch.sin(torch.pi * (xx - yy)),
        ))
        return IdentityAnchoredConditionalAmplitudeGenerator.normalized_modes(modes)

    @staticmethod
    def normalized_modes(modes: torch.Tensor) -> torch.Tensor:
        if modes.ndim != 3:
            raise ValueError("global modes must have shape [code_dim, H, W]")
        centered = modes - modes.mean(dim=(-2, -1), keepdim=True)
        rms = centered.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8).sqrt()
        return centered / rms

    @property
    def Wz(self) -> nn.Linear:
        """Formula-level alias for the bias-free code map."""
        return self.code_projection

    @property
    def Wy(self) -> torch.Tensor:
        """Formula-level alias for the per-label residual map."""
        return self.class_heads

    @property
    def By(self) -> torch.Tensor:
        """Formula-level alias for the class-carrier logits."""
        return self.class_carrier_logits

    @property
    def P(self) -> torch.Tensor:
        """Formula-level alias for the trainable global modes."""
        return self.global_modes

    def _validate_inputs(self, code: torch.Tensor, labels: torch.Tensor) -> None:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("code must have shape [batch, 8]")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.numel() == 0:
            raise ValueError("labels must have shape [batch]")
        if labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")

    def coefficients(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(code, labels)
        h = torch.nn.functional.silu(self.code_projection(code))
        residual = torch.einsum("bkh,bh->bk", self.class_heads.index_select(0, labels), h)
        # The direct z term is the identity anchor.  It is not a decoder-side
        # latent: coefficients are converted immediately into the SLM command.
        return torch.tanh(code + residual)

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(code, labels)
        modes = self.normalized_modes(self.global_modes)
        residual = torch.einsum("bk,khw->bhw", coefficients, modes)
        logits = self.class_carrier_logits.index_select(0, labels) + self.command_gain * residual
        return torch.sigmoid(logits).unsqueeze(1)

    def mode_orthogonality_penalty(self) -> torch.Tensor:
        modes = self.normalized_modes(self.global_modes).flatten(1)
        gram = modes @ modes.T / modes.shape[1]
        return (gram - torch.eye(self.code_dim, device=modes.device, dtype=modes.dtype)).square().mean()

    def class_carrier_command(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1 or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")
        return torch.sigmoid(self.class_carrier_logits.index_select(0, labels)).unsqueeze(1)


# Descriptive compatibility alias used by the architecture/config contract.
ConditionalIdentityLogitAmplitudeGenerator = IdentityAnchoredConditionalAmplitudeGenerator


class ConditionalClassConditionedModesGenerator(nn.Module):
    """Identity-anchored generator with an independent physical mode frame per label."""

    architecture_kind = "conditional_class_conditioned_modes_v3"
    fixed_command_gain = 0.5

    def __init__(self, *, shape: tuple[int, int] = (200, 200), code_dim: int = 8, hidden_dim: int = 64,
                 command_gain: float = fixed_command_gain,
                 class_carrier_logits: torch.Tensor | None = None,
                 class_modes: torch.Tensor | None = None) -> None:
        super().__init__()
        if shape[0] <= 0 or shape[1] <= 0 or code_dim != 8 or hidden_dim != 64 or command_gain != self.fixed_command_gain:
            raise ValueError("class-conditioned modes require 8D code and fixed gain 0.5")
        self.shape, self.code_dim = shape, code_dim
        self.command_gain = self.fixed_command_gain
        carrier = torch.zeros(10, *shape) if class_carrier_logits is None else class_carrier_logits.detach().clone()
        if tuple(carrier.shape) != (10, *shape):
            raise ValueError("class_carrier_logits must have shape [10, H, W]")
        if class_modes is None:
            base = IdentityAnchoredConditionalAmplitudeGenerator._initial_global_modes(shape)
            modes = base.unsqueeze(0).expand(10, -1, -1, -1).clone()
        else:
            modes = class_modes.detach().clone()
        if tuple(modes.shape) != (10, code_dim, *shape):
            raise ValueError("class_modes must have shape [10, 8, H, W]")
        self.class_carrier_logits = nn.Parameter(carrier)
        self.class_modes = nn.Parameter(modes)

    @staticmethod
    def normalized_modes(modes: torch.Tensor) -> torch.Tensor:
        if modes.ndim != 4:
            raise ValueError("class modes must have shape [10, 8, H, W]")
        centered = modes - modes.mean(dim=(-2, -1), keepdim=True)
        rms = centered.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8).sqrt()
        return centered / rms

    def _validate_inputs(self, code: torch.Tensor, labels: torch.Tensor) -> None:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("code must have shape [batch, 8]")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")

    def coefficients(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(code, labels)
        return torch.tanh(code)

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(code, labels)
        modes = self.normalized_modes(self.class_modes).index_select(0, labels)
        residual = torch.einsum("bk,bkhw->bhw", coefficients, modes)
        logits = self.class_carrier_logits.index_select(0, labels) + self.command_gain * residual
        return torch.sigmoid(logits).unsqueeze(1)

    def mode_orthogonality_penalty(self) -> torch.Tensor:
        modes = self.normalized_modes(self.class_modes).flatten(2)
        gram = torch.matmul(modes, modes.transpose(1, 2)) / modes.shape[-1]
        eye = torch.eye(self.code_dim, device=modes.device, dtype=modes.dtype).expand(10, -1, -1)
        return (gram - eye).square().mean()

    def class_carrier_command(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1 or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")
        return torch.sigmoid(self.class_carrier_logits.index_select(0, labels)).unsqueeze(1)


class ConditionalParityStabilizedModesGenerator(nn.Module):
    """Class-conditioned odd diversity modes plus even semantic stabilizers."""

    architecture_kind = "conditional_parity_stabilized_modes_v4"
    fixed_command_gain = 0.5

    def __init__(self, *, shape: tuple[int, int] = (200, 200), code_dim: int = 8,
                 hidden_dim: int = 64, command_gain: float = fixed_command_gain,
                 class_carrier_logits: torch.Tensor | None = None,
                 class_modes: torch.Tensor | None = None,
                 semantic_stabilizers: torch.Tensor | None = None) -> None:
        super().__init__()
        if shape[0] <= 0 or shape[1] <= 0 or code_dim != 8 or hidden_dim != 64 or command_gain != self.fixed_command_gain:
            raise ValueError("parity-stabilized modes require 8D code and fixed gain 0.5")
        self.shape, self.code_dim = shape, code_dim
        carrier = torch.zeros(10, *shape) if class_carrier_logits is None else class_carrier_logits.detach().clone()
        modes = (IdentityAnchoredConditionalAmplitudeGenerator._initial_global_modes(shape).unsqueeze(0).expand(10, -1, -1, -1).clone()
                 if class_modes is None else class_modes.detach().clone())
        stabilizers = torch.zeros(10, code_dim, *shape) if semantic_stabilizers is None else semantic_stabilizers.detach().clone()
        if tuple(carrier.shape) != (10, *shape) or tuple(modes.shape) != (10, code_dim, *shape) or tuple(stabilizers.shape) != (10, code_dim, *shape):
            raise ValueError("parity generator tensors have invalid shapes")
        self.command_gain = self.fixed_command_gain
        self.class_carrier_logits = nn.Parameter(carrier)
        self.class_modes = nn.Parameter(modes)
        self.semantic_stabilizers = nn.Parameter(stabilizers)

    @staticmethod
    def normalized_modes(modes: torch.Tensor) -> torch.Tensor:
        return ConditionalClassConditionedModesGenerator.normalized_modes(modes)

    def _validate_inputs(self, code: torch.Tensor, labels: torch.Tensor) -> None:
        if code.ndim != 2 or code.shape[1] != self.code_dim or labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("code must be [batch,8] and labels must be in [0,9]")

    def coefficients(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(code, labels)
        return torch.tanh(code)

    def parity_terms(self, code: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(code, labels)
        t = torch.tanh(code)
        energy = t.square().sum(dim=1)
        rho = 1.0 - torch.exp(-energy)
        weights = t.square() / energy.clamp_min(1.0e-8).unsqueeze(1)
        modes = self.normalized_modes(self.class_modes).index_select(0, labels)
        odd = torch.einsum("bk,bkhw->bhw", t, modes)
        even_modes = torch.tanh(self.semantic_stabilizers).index_select(0, labels)
        even = torch.einsum("bk,bkhw->bhw", weights, even_modes)
        return odd, rho, even

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        odd, rho, even = self.parity_terms(code, labels)
        logits = self.class_carrier_logits.index_select(0, labels) + self.fixed_command_gain * (odd + rho[:, None, None] * even)
        return torch.sigmoid(logits).unsqueeze(1)

    def mode_orthogonality_penalty(self) -> torch.Tensor:
        modes = self.normalized_modes(self.class_modes).flatten(2)
        gram = torch.matmul(modes, modes.transpose(1, 2)) / modes.shape[-1]
        eye = torch.eye(self.code_dim, device=modes.device, dtype=modes.dtype).expand(10, -1, -1)
        return (gram - eye).square().mean()

    def class_carrier_command(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1 or labels.numel() == 0 or labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")
        return torch.sigmoid(self.class_carrier_logits.index_select(0, labels)).unsqueeze(1)
