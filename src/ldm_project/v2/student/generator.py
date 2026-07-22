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
