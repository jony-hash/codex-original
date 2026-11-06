"""Exact square-aperture overlap with target pixels under assumed rotation."""
import math

import torch

from hw207_mnist4_c2_v1 import geometry


def _clip(polygon, axis, edge, keep_above):
    result = []
    if not polygon:
        return result
    previous = polygon[-1]
    previous_inside = previous[axis] >= edge if keep_above else previous[axis] <= edge
    for current in polygon:
        inside = current[axis] >= edge if keep_above else current[axis] <= edge
        if inside != previous_inside:
            fraction = (edge - previous[axis]) / (current[axis] - previous[axis])
            result.append((previous[0] + fraction * (current[0] - previous[0]),
                           previous[1] + fraction * (current[1] - previous[1])))
        if inside:
            result.append(current)
        previous, previous_inside = current, inside
    return result


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    return abs(sum(x * polygon[(i + 1) % len(polygon)][1] -
                   y * polygon[(i + 1) % len(polygon)][0]
                   for i, (x, y) in enumerate(polygon))) / 2


def rotated_coverage_matrix(count, magnification, fill_factor, rotation_rad, offset_um=(0., 0.)):
    """Sparse pixel-by-mirror area fractions in the assumed target coordinate frame."""
    if (not 0 < fill_factor <= 1 or len(offset_um) != 2 or
            not all(math.isfinite(v) for v in (rotation_rad, *offset_um))):
        raise ValueError("Invalid optical parameters")
    pitch = geometry(count, magnification)["pitch_um"]
    half = pitch * math.sqrt(fill_factor) / 2
    c, s = math.cos(rotation_rad), math.sin(rotation_rad)
    indices, values = [], []
    for row in range(count):
        for col in range(count):
            cx, cy = (col + .5 - count / 2) * pitch, (row + .5 - count / 2) * pitch
            polygon = [(c*x - s*y + offset_um[0], s*x + c*y + offset_um[1])
                       for x, y in ((cx-half, cy-half), (cx+half, cy-half),
                                    (cx+half, cy+half), (cx-half, cy+half))]
            minx = max(0, math.floor((min(x for x, y in polygon) + 100) / 2))
            maxx = min(99, math.floor((max(x for x, y in polygon) + 100) / 2))
            miny = max(0, math.floor((min(y for x, y in polygon) + 100) / 2))
            maxy = min(99, math.floor((max(y for x, y in polygon) + 100) / 2))
            for py in range(miny, maxy + 1):
                for px in range(minx, maxx + 1):
                    clipped = polygon
                    for axis, edge, above in ((0, 2*px-100, True), (0, 2*px-98, False),
                                              (1, 2*py-100, True), (1, 2*py-98, False)):
                        clipped = _clip(clipped, axis, edge, above)
                    area = _area(clipped)
                    if area > 1e-12:
                        indices.append((py * 100 + px, row * count + col))
                        values.append(area / 4)
    if indices:
        ij = torch.tensor(indices, dtype=torch.int64).T
    else:
        ij = torch.empty((2, 0), dtype=torch.int64)
    return torch.sparse_coo_tensor(ij, torch.tensor(values, dtype=torch.float64),
                                   (10000, count * count)).coalesce()


def rotated_intensity(pattern, magnification, matrix):
    if pattern.ndim != 3 or pattern.shape[-1] != pattern.shape[-2]:
        raise ValueError("Expected a batch of square patterns")
    if not ((pattern == 0) | (pattern == 1)).all():
        raise ValueError("Mirror commands must be binary")
    image = pattern.flip((-2, -1)) if magnification < 0 else pattern
    out = torch.sparse.mm(matrix.to(image.device), image.reshape(len(image), -1).double().T)
    return out.T.reshape(-1, 100, 100) / magnification ** 2
