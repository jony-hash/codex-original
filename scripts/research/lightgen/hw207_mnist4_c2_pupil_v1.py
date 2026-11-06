"""Padded, pixel-integrated coherent input for C2 pupil convergence studies."""
import math

import torch

from hw207_mnist4_c2_v1 import geometry


def padded_axis_field(pattern, magnification, fill_factor, pitch_um, width_um,
                      offset_um=(0.0, 0.0)):
    """Average the continuous square-mirror field over each padded grid cell.

    This preserves contributions outside the central 200 um before pupil
    filtering. It does not claim to model the actual tilted-mirror field.
    """
    if pattern.ndim != 3 or pattern.shape[-2] != pattern.shape[-1]:
        raise ValueError("Expected a batch of square mirror patterns")
    if not ((pattern == 0) | (pattern == 1)).all():
        raise ValueError("Mirror commands must be binary")
    if (not all(math.isfinite(v) for v in (fill_factor, pitch_um, width_um, *offset_um))
            or len(offset_um) != 2 or not 0 < fill_factor <= 1
            or pitch_um <= 0 or width_um < 200):
        raise ValueError("Invalid grid or optical parameters")
    n = round(width_um / pitch_um)
    if n <= 0 or not math.isclose(n * pitch_um, width_um, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Window width must be an integer number of grid cells")
    count = pattern.shape[-1]
    g = geometry(count, magnification)
    mirror_pitch = g["pitch_um"]
    half = mirror_pitch * math.sqrt(fill_factor) / 2
    starts = torch.arange(n, device=pattern.device, dtype=torch.float64) * pitch_um - width_um / 2
    centers = (torch.arange(count, device=pattern.device, dtype=torch.float64) + .5 - count / 2) * mirror_pitch
    matrices = []
    for offset in offset_um:
        left, right = centers + offset - half, centers + offset + half
        overlap = (torch.minimum(starts[:, None] + pitch_um, right[None, :])
                   - torch.maximum(starts[:, None], left[None, :])).clamp_min(0)
        matrices.append(overlap / pitch_um)
    commands = pattern.double()
    if magnification < 0:
        commands = commands.flip((-2, -1))
    x, y = matrices
    return (y @ commands @ x.T) / abs(magnification)
