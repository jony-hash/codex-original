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
    singular_values: torch.Tensor | None = None
    mode_indices: torch.Tensor | None = None
    frequencies_cycles_per_mm: torch.Tensor | None = None

    @property
    def artifact_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.basis, self.normalizer, self.singular_values if self.singular_values is not None else torch.empty(0), self.mode_indices if self.mode_indices is not None else torch.empty(0, dtype=torch.int64), self.frequencies_cycles_per_mm if self.frequencies_cycles_per_mm is not None else torch.empty(0)):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str((self.candidate_count, self.seed, self.source_hash)).encode("utf-8"))
        return digest.hexdigest()

    def save(self, path: str) -> None:
        atomic_torch_save(path, {"basis": self.basis, "normalizer": self.normalizer, "candidate_count": self.candidate_count, "seed": self.seed, "source_hash": self.source_hash, "singular_values": self.singular_values, "mode_indices": self.mode_indices, "frequencies_cycles_per_mm": self.frequencies_cycles_per_mm, "artifact_hash": self.artifact_hash})

    @classmethod
    def load(cls, path: str) -> "CodeBasisArtifactV2":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("invalid code basis artifact")
        artifact = cls(payload["basis"], payload["normalizer"], int(payload["candidate_count"]), int(payload["seed"]), str(payload["source_hash"]), payload.get("singular_values"), payload.get("mode_indices"), payload.get("frequencies_cycles_per_mm"))
        if payload.get("artifact_hash") != artifact.artifact_hash:
            raise ValueError("code basis artifact hash mismatch")
        return artifact


def supported_dct_candidates(support: torch.Tensor, *, candidate_count: int, pixel_pitch_m: tuple[float, float], max_cycles_per_mm: float) -> DCTCandidateSet:
    """Return weighted-zero-mean low-frequency command directions for one label."""
    if support.ndim != 2 or candidate_count < 8 or max_cycles_per_mm <= 0.0:
        raise ValueError("support must be [H, W] and at least eight candidates are required")
    height, width = support.shape
    y = (torch.arange(height, device=support.device, dtype=support.dtype) + 0.5) / height
    x = (torch.arange(width, device=support.device, dtype=support.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    candidates: list[tuple[float, int, int, torch.Tensor]] = []
    fov_y = height * pixel_pitch_m[0]
    fov_x = width * pixel_pitch_m[1]
    # DCT mode k contains k/2 cycles across the sampled FOV.
    max_mode_y = int(2.0 * fov_y * max_cycles_per_mm * 1.0e3)
    max_mode_x = int(2.0 * fov_x * max_cycles_per_mm * 1.0e3)
    for fy in range(max_mode_y + 1):
        for fx in range(max_mode_x + 1):
            if fy == 0 and fx == 0:
                continue
            frequency = ((fy / (2.0 * fov_y)) ** 2 + (fx / (2.0 * fov_x)) ** 2) ** 0.5 / 1.0e3
            if frequency > max_cycles_per_mm + 1.0e-12:
                continue
            direction = torch.cos(torch.pi * fy * yy) * torch.cos(torch.pi * fx * xx) * support
            direction = direction - (direction * support).sum() / support.sum().clamp_min(1.0e-12) * support
            norm = direction.square().sum().sqrt()
            if norm > 1.0e-12:
                candidates.append((frequency, fy, fx, direction / norm))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(candidates) < candidate_count:
        raise ValueError("requested DCT candidate count exceeds the physical <=20 cycles/mm support basis")
    selected = candidates[:candidate_count]
    return DCTCandidateSet(
        directions=torch.stack([item[3] for item in selected]),
        mode_indices=torch.tensor([[item[1], item[2]] for item in selected], device=support.device, dtype=torch.int64),
        frequencies_cycles_per_mm=torch.tensor([item[0] for item in selected], device=support.device, dtype=support.dtype),
    )


def derive_basis(response_matrix: torch.Tensor, candidates: torch.Tensor, *, selected_basis_dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the right response singular vectors to choose input-space command directions."""
    if response_matrix.ndim != 2 or candidates.ndim != 3 or response_matrix.shape[0] != candidates.shape[0]:
        raise ValueError("response_matrix and candidates must agree on candidate axis")
    if selected_basis_dim != 8 or response_matrix.shape[0] < selected_basis_dim:
        raise ValueError("v2 formal capacity selects exactly eight basis directions")
    left, singular_values, _ = torch.linalg.svd(response_matrix, full_matrices=False)
    coefficients = left[:, :selected_basis_dim].transpose(0, 1)
    return torch.einsum("kc,chw->khw", coefficients, candidates), singular_values


def build_code_basis(response_matrices: torch.Tensor, supports: torch.Tensor, *, candidate_count: int, seed: int, source_hash: str, pixel_pitch_m: tuple[float, float], max_cycles_per_mm: float) -> CodeBasisArtifactV2:
    """Build a fixed [label, 8, H, W] artifact from frozen response measurements."""
    if response_matrices.shape[0] != 10 or supports.shape[0] != 10:
        raise ValueError("formal capacity artifact requires all ten labels")
    bases, spectra, modes, frequencies = [], [], [], []
    for label in range(10):
        candidates = supported_dct_candidates(supports[label], candidate_count=candidate_count, pixel_pitch_m=pixel_pitch_m, max_cycles_per_mm=max_cycles_per_mm)
        basis, singular_values = derive_basis(response_matrices[label], candidates.directions)
        bases.append(basis)
        spectra.append(singular_values)
        modes.append(candidates.mode_indices)
        frequencies.append(candidates.frequencies_cycles_per_mm)
    basis = torch.stack(bases)
    normalizer = basis.abs().sum(dim=1).amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
    return CodeBasisArtifactV2(basis=basis, normalizer=normalizer, candidate_count=candidate_count, seed=seed, source_hash=source_hash, singular_values=torch.stack(spectra), mode_indices=torch.stack(modes), frequencies_cycles_per_mm=torch.stack(frequencies))


@torch.no_grad()
def finite_difference_response_matrices(system, prototype_commands: torch.Tensor, supports: torch.Tensor, *, candidate_count: int, epsilon: float, pixel_pitch_m: tuple[float, float], max_cycles_per_mm: float, microbatch_size: int = 1) -> tuple[torch.Tensor, bool, str, str]:
    """Measure frozen command-to-raw-camera responses without creating an optimizer.

    The caller deliberately chooses when to run this full-geometry CUDA audit;
    v2 itself neither updates parameters nor saves full optical fields here.
    """
    if prototype_commands.ndim != 4 or prototype_commands.shape[:2] != (10, 1):
        raise ValueError("prototype_commands must have shape [10, 1, H, W]")
    if supports.shape != (10, *prototype_commands.shape[-2:]):
        raise ValueError("supports must have shape [10, H, W]")
    if epsilon <= 0.0:
        raise ValueError("finite-difference epsilon must be positive")
    if microbatch_size != 1:
        raise ValueError("finite-difference response directions are physically measured one command at a time")
    before = {name: value.detach().clone() for name, value in system.state_dict().items()}
    before_hash = _state_hash(before)
    rows = []
    for label in range(10):
        candidates = supported_dct_candidates(supports[label], candidate_count=candidate_count, pixel_pitch_m=pixel_pitch_m, max_cycles_per_mm=max_cycles_per_mm)
        responses = []
        base = prototype_commands[label : label + 1]
        for direction in candidates.directions:
            delta = direction.unsqueeze(0).unsqueeze(0) * epsilon
            plus = system.forward(system.command((base + delta).clamp(0.0, 1.0))).raw_camera.tensor
            minus = system.forward(system.command((base - delta).clamp(0.0, 1.0))).raw_camera.tensor
            responses.append(((plus - minus) / (2.0 * epsilon)).flatten().detach().cpu())
        rows.append(torch.stack(responses))
    after = system.state_dict()
    after_hash = _state_hash(after)
    unchanged = before_hash == after_hash and all(torch.equal(before[name], value) for name, value in after.items())
    return torch.stack(rows), unchanged, before_hash, after_hash


def _state_hash(state: dict[str, torch.Tensor] | torch.nn.modules.module.OrderedDict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capacity_metrics(commands_unclamped: torch.Tensor, physical_slm_intensity: torch.Tensor, raw_camera: torch.Tensor, supports: torch.Tensor) -> tuple[float, float, float, float]:
    """Return command/camera code factors, support energy, and one-clipping fraction."""
    if commands_unclamped.shape[:2] != (10, physical_slm_intensity.shape[1]) or raw_camera.shape[:2] != commands_unclamped.shape[:2]:
        raise ValueError("capacity tensors must share [10, code] axes")
    command_delta = commands_unclamped - commands_unclamped.mean(dim=1, keepdim=True)
    support = supports[:, None]
    support_energy = (command_delta.square() * support).sum() / command_delta.square().sum().clamp_min(1.0e-12)
    clipping = ((commands_unclamped < 0.0) | (commands_unclamped > 1.0)).to(torch.float32).mean()
    return balanced_code_fraction(physical_slm_intensity), balanced_code_fraction(raw_camera), float(support_energy.item()), float(clipping.item())


@torch.no_grad()
def replay_capacity(system, prototype_commands: torch.Tensor, artifact: CodeBasisArtifactV2, supports: torch.Tensor, *, codes_per_label: int, command_gain: float = 0.1, microbatch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay one fixed 100×10 basis experiment inside the frozen formal system."""
    if prototype_commands.shape[0] != 10 or prototype_commands.shape[1] != 1:
        raise ValueError("prototype commands must be [10, 1, H, W]")
    if microbatch_size < 1:
        raise ValueError("capacity replay microbatch must be positive")
    generator = torch.Generator(device="cpu").manual_seed(artifact.seed)
    coefficients = torch.tanh(torch.randn(codes_per_label, 8, generator=generator, dtype=prototype_commands.dtype))
    commands = []
    cameras = []
    for label in range(10):
        delta = torch.einsum("ck,khw->chw", coefficients, artifact.basis[label].cpu()) / artifact.normalizer[label].cpu()
        unclamped = prototype_commands[label, 0].detach().cpu().unsqueeze(0) + command_gain * delta
        label_cameras = []
        for start in range(0, codes_per_label, microbatch_size):
            command = unclamped[start : start + microbatch_size].clamp(0.0, 1.0).to(prototype_commands.device).unsqueeze(1)
            label_cameras.append(system.forward(system.command(command)).raw_camera.tensor.squeeze(1).detach().cpu())
        cameras.append(torch.cat(label_cameras))
        commands.append(unclamped)
    unclamped_commands = torch.stack(commands)
    return unclamped_commands, unclamped_commands.clamp(0.0, 1.0).square(), torch.stack(cameras), coefficients.cpu()
@dataclass(frozen=True)
class DCTCandidateSet:
    directions: torch.Tensor  # [candidate, H, W]
    mode_indices: torch.Tensor  # [candidate, 2], (fy, fx)
    frequencies_cycles_per_mm: torch.Tensor  # [candidate]
