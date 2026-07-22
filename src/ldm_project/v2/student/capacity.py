"""Frozen DCT/SVD command-capacity audit; it never creates an optimizer."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from ..evidence import atomic_torch_save
from .admission import balanced_code_fraction


@dataclass(frozen=True)
class CodeBasisArtifactV2:
    basis: torch.Tensor  # [10, 8, H, W]
    normalizer: torch.Tensor  # [10, 1, 1]
    candidate_count: int
    seed: int
    source_hash: str

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.basis, self.normalizer):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str((self.candidate_count, self.seed, self.source_hash)).encode("utf-8"))
        return digest.hexdigest()

    def save(self, path: str) -> None:
        atomic_torch_save(path, {"basis": self.basis, "normalizer": self.normalizer, "candidate_count": self.candidate_count, "seed": self.seed, "source_hash": self.source_hash, "artifact_hash": self.artifact_hash})

    @classmethod
    def load(cls, path: str) -> "CodeBasisArtifactV2":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("invalid code basis artifact")
        artifact = cls(payload["basis"], payload["normalizer"], int(payload["candidate_count"]), int(payload["seed"]), str(payload["source_hash"]))
        if payload.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("code basis artifact hash mismatch")
        return artifact


def supported_dct_candidates(support: torch.Tensor, *, candidate_count: int) -> torch.Tensor:
    """Return weighted-zero-mean low-frequency command directions for one label."""
    if support.ndim != 2 or candidate_count < 8:
        raise ValueError("support must be [H, W] and at least eight candidates are required")
    height, width = support.shape
    y = (torch.arange(height, device=support.device, dtype=support.dtype) + 0.5) / height
    x = (torch.arange(width, device=support.device, dtype=support.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    candidates = []
    max_frequency = max(1, int(candidate_count**0.5) + 1)
    for fy in range(max_frequency):
        for fx in range(max_frequency):
            if fy == 0 and fx == 0:
                continue
            direction = torch.cos(torch.pi * fy * yy) * torch.cos(torch.pi * fx * xx) * support
            direction = direction - (direction * support).sum() / support.sum().clamp_min(1.0e-12) * support
            norm = direction.square().sum().sqrt()
            if norm > 1.0e-12:
                candidates.append(direction / norm)
            if len(candidates) == candidate_count:
                return torch.stack(candidates)
    raise ValueError("requested DCT candidate count exceeds the selected support basis")


def derive_basis(response_matrix: torch.Tensor, candidates: torch.Tensor, *, selected_basis_dim: int = 8) -> torch.Tensor:
    """Use the right response singular vectors to choose input-space command directions."""
    if response_matrix.ndim != 2 or candidates.ndim != 3 or response_matrix.shape[0] != candidates.shape[0]:
        raise ValueError("response_matrix and candidates must agree on candidate axis")
    if selected_basis_dim != 8 or response_matrix.shape[0] < selected_basis_dim:
        raise ValueError("v2 formal capacity selects exactly eight basis directions")
    left, _, _ = torch.linalg.svd(response_matrix, full_matrices=False)
    coefficients = left[:, :selected_basis_dim].transpose(0, 1)
    return torch.einsum("kc,chw->khw", coefficients, candidates)


def build_code_basis(response_matrices: torch.Tensor, supports: torch.Tensor, *, candidate_count: int, seed: int, source_hash: str) -> CodeBasisArtifactV2:
    """Build a fixed [label, 8, H, W] artifact from frozen response measurements."""
    if response_matrices.shape[0] != 10 or supports.shape[0] != 10:
        raise ValueError("formal capacity artifact requires all ten labels")
    bases = []
    for label in range(10):
        candidates = supported_dct_candidates(supports[label], candidate_count=candidate_count)
        bases.append(derive_basis(response_matrices[label], candidates))
    basis = torch.stack(bases)
    normalizer = basis.abs().sum(dim=1).amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
    return CodeBasisArtifactV2(basis=basis, normalizer=normalizer, candidate_count=candidate_count, seed=seed, source_hash=source_hash)


def capacity_metrics(commands_unclamped: torch.Tensor, physical_slm_intensity: torch.Tensor, raw_camera: torch.Tensor, supports: torch.Tensor) -> tuple[float, float, float, float]:
    """Return command/camera code factors, support energy, and one-clipping fraction."""
    if commands_unclamped.shape[:2] != (10, physical_slm_intensity.shape[1]) or raw_camera.shape[:2] != commands_unclamped.shape[:2]:
        raise ValueError("capacity tensors must share [10, code] axes")
    command_delta = commands_unclamped - commands_unclamped.mean(dim=1, keepdim=True)
    support = supports[:, None]
    support_energy = (command_delta.square() * support).sum() / command_delta.square().sum().clamp_min(1.0e-12)
    clipping = ((commands_unclamped < 0.0) | (commands_unclamped > 1.0)).to(torch.float32).mean()
    return balanced_code_fraction(physical_slm_intensity), balanced_code_fraction(raw_camera), float(support_energy.item()), float(clipping.item())
