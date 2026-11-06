"""Physical-coordinate utilities for HW207 diagnostic-only spatial mapping.

This module models geometry and finite-area integration.  It does not model a
calibrated DMD reflection port, a fabricated LightGen layer, or measured camera
response.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


DMD_ROWS = 1600
DMD_COLUMNS = 2560
DMD_PITCH_UM = 7.56
TARGET_WIDTH_UM = 200.0
TARGET_PIXELS = 100
TARGET_PIXEL_UM = TARGET_WIDTH_UM / TARGET_PIXELS
CMOS_ROWS = 2048
CMOS_COLUMNS = 2048
CMOS_PIXEL_UM = 6.5


def _matrix_and_offset(
    matrix: Sequence[Sequence[float]], offset_um: Sequence[float]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix) or len(offset_um) != 2:
        raise ValueError("Expected a 2x2 affine matrix and a 2-vector offset")
    a = tuple(tuple(float(value) for value in row) for row in matrix)
    b = (float(offset_um[0]), float(offset_um[1]))
    values = (*a[0], *a[1], *b)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Affine parameters must be finite")
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(determinant) < 1e-15:
        raise ValueError("Affine matrix must be invertible")
    return a, b, (determinant, abs(determinant))


def rotation_matrix(degrees: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if not math.isfinite(degrees):
        raise ValueError("Rotation must be finite")
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, -sine), (sine, cosine))


def scaled_rotation(magnification: float, degrees: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if not math.isfinite(magnification) or magnification == 0:
        raise ValueError("Magnification must be finite and nonzero")
    rotation = rotation_matrix(degrees)
    return tuple(tuple(magnification * value for value in row) for row in rotation)


def _transform(
    x: float,
    y: float,
    matrix: tuple[tuple[float, float], tuple[float, float]],
    offset: tuple[float, float],
) -> tuple[float, float]:
    return (
        matrix[0][0] * x + matrix[0][1] * y + offset[0],
        matrix[1][0] * x + matrix[1][1] * y + offset[1],
    )


def _inverse_point(
    x: float,
    y: float,
    matrix: tuple[tuple[float, float], tuple[float, float]],
    offset: tuple[float, float],
) -> tuple[float, float]:
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    x, y = x - offset[0], y - offset[1]
    return (
        (matrix[1][1] * x - matrix[0][1] * y) / determinant,
        (-matrix[1][0] * x + matrix[0][0] * y) / determinant,
    )


def _clip_polygon(
    polygon: list[tuple[float, float]], axis: int, edge: float, keep_greater: bool
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= edge if keep_greater else previous[axis] <= edge
    for current in polygon:
        current_inside = current[axis] >= edge if keep_greater else current[axis] <= edge
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = 0.0 if denominator == 0 else (edge - previous[axis]) / denominator
            result.append((
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            ))
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return result


def _clip_rectangle(
    polygon: list[tuple[float, float]], x0: float, x1: float, y0: float, y1: float
) -> list[tuple[float, float]]:
    polygon = _clip_polygon(polygon, 0, x0, True)
    polygon = _clip_polygon(polygon, 0, x1, False)
    polygon = _clip_polygon(polygon, 1, y0, True)
    return _clip_polygon(polygon, 1, y1, False)


def polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(sum(
        x * polygon[(index + 1) % len(polygon)][1]
        - y * polygon[(index + 1) % len(polygon)][0]
        for index, (x, y) in enumerate(polygon)
    )) / 2.0


def _cell_corners(center_x: float, center_y: float, half_x: float, half_y: float):
    return [
        (center_x - half_x, center_y - half_y),
        (center_x + half_x, center_y - half_y),
        (center_x + half_x, center_y + half_y),
        (center_x - half_x, center_y + half_y),
    ]


def dmd_mirror_center(column: float, row: float) -> tuple[float, float]:
    """Return a mirror center in centered DMD-package coordinates (µm)."""
    if not math.isfinite(column) or not math.isfinite(row):
        raise ValueError("DMD indices must be finite")
    return (
        (column - (DMD_COLUMNS - 1) / 2) * DMD_PITCH_UM,
        ((DMD_ROWS - 1) / 2 - row) * DMD_PITCH_UM,
    )


def dmd_array_bounds_um(angle_deg: float = 0.0) -> dict[str, float]:
    """Return the rotated physical bounding box of the complete active array."""
    rotation = rotation_matrix(angle_deg)
    width = DMD_COLUMNS * DMD_PITCH_UM
    height = DMD_ROWS * DMD_PITCH_UM
    corners = [
        _transform(x, y, rotation, (0.0, 0.0))
        for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2))
    ]
    xs, ys = [point[0] for point in corners], [point[1] for point in corners]
    return {"width_um": max(xs) - min(xs), "height_um": max(ys) - min(ys),
            "x_min_um": min(xs), "x_max_um": max(xs),
            "y_min_um": min(ys), "y_max_um": max(ys)}


def local_mirror_window(
    matrix: Sequence[Sequence[float]],
    offset_um: Sequence[float] = (0.0, 0.0),
    width_um: float = TARGET_WIDTH_UM,
    pitch_um: float = DMD_PITCH_UM,
) -> dict[str, object]:
    """Select the DMD sub-array whose full cells can intersect a target square."""
    affine, offset, _ = _matrix_and_offset(matrix, offset_um)
    if (not math.isfinite(width_um) or width_um <= 0
            or not math.isfinite(pitch_um) or pitch_um <= 0):
        raise ValueError("Target width and DMD pitch must be positive")
    half = width_um / 2
    dmd_corners = [_inverse_point(x, y, affine, offset)
                   for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))]
    xs, ys = [point[0] for point in dmd_corners], [point[1] for point in dmd_corners]
    col_center = (DMD_COLUMNS - 1) / 2
    row_center = (DMD_ROWS - 1) / 2
    col_lo = max(0, math.floor(min(xs) / pitch_um + col_center - 0.5) - 1)
    col_hi = min(DMD_COLUMNS, math.ceil(max(xs) / pitch_um + col_center + 0.5) + 2)
    row_values = [row_center - y / pitch_um for y in ys]
    row_lo = max(0, math.floor(min(row_values) - 0.5) - 1)
    row_hi = min(DMD_ROWS, math.ceil(max(row_values) + 0.5) + 2)
    if col_hi <= col_lo or row_hi <= row_lo:
        raise ValueError("Target square does not intersect the DMD array")
    rows = torch.arange(row_lo, row_hi, dtype=torch.int64)
    columns = torch.arange(col_lo, col_hi, dtype=torch.int64)
    center_x = (columns.to(torch.float64) - col_center) * pitch_um
    center_y = (row_center - rows.to(torch.float64)) * pitch_um
    return {"row_start": row_lo, "column_start": col_lo, "rows": rows, "columns": columns,
            "center_x_um": center_x, "center_y_um": center_y,
            "shape": (len(rows), len(columns)), "full_array_shape": (DMD_ROWS, DMD_COLUMNS)}


def _source_coordinates(x_um: float, y_um: float, width_um: float, size: int):
    pixel_um = width_um / size
    return ((x_um + width_um / 2) / pixel_um,
            (width_um / 2 - y_um) / pixel_um)


def build_source_sampling_matrix(
    matrix: Sequence[Sequence[float]],
    offset_um: Sequence[float] = (0.0, 0.0),
    method: str = "nearest",
    *,
    source_size: int = 28,
    source_width_um: float = TARGET_WIDTH_UM,
    pitch_um: float = DMD_PITCH_UM,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Map MNIST source cells to local DMD mirror commands by nearest or area."""
    if (method not in ("nearest", "area") or source_size <= 0
            or not math.isfinite(source_width_um) or source_width_um <= 0
            or not math.isfinite(pitch_um) or pitch_um <= 0):
        raise ValueError("Unsupported mapping method or source grid")
    affine, offset, _ = _matrix_and_offset(matrix, offset_um)
    lattice = local_mirror_window(affine, offset, source_width_um, pitch_um)
    rows, columns = lattice["rows"], lattice["columns"]
    source_indices: list[tuple[int, int]] = []
    weights: list[float] = []
    mirror_index = 0
    for row in rows.tolist():
        center_y = ((DMD_ROWS - 1) / 2 - row) * pitch_um
        for column in columns.tolist():
            center_x = (column - (DMD_COLUMNS - 1) / 2) * pitch_um
            center = _transform(center_x, center_y, affine, offset)
            if method == "nearest":
                source_col, source_row = _source_coordinates(*center, source_width_um, source_size)
                ix, iy = math.floor(source_col), math.floor(source_row)
                if 0 <= ix < source_size and 0 <= iy < source_size:
                    source_indices.append((mirror_index, iy * source_size + ix))
                    weights.append(1.0)
            else:
                source_polygon = [
                    _source_coordinates(*_transform(x, y, affine, offset), source_width_um, source_size)
                    for x, y in _cell_corners(center_x, center_y, pitch_um / 2, pitch_um / 2)
                ]
                full_area = polygon_area(source_polygon)
                if full_area > 0:
                    min_col = max(0, math.floor(min(point[0] for point in source_polygon)))
                    max_col = min(source_size - 1, math.floor(max(point[0] for point in source_polygon)))
                    min_row = max(0, math.floor(min(point[1] for point in source_polygon)))
                    max_row = min(source_size - 1, math.floor(max(point[1] for point in source_polygon)))
                    for source_row in range(min_row, max_row + 1):
                        for source_col in range(min_col, max_col + 1):
                            area = polygon_area(_clip_rectangle(
                                source_polygon, source_col, source_col + 1,
                                source_row, source_row + 1,
                            ))
                            if area > 1e-12:
                                source_indices.append((mirror_index, source_row * source_size + source_col))
                                weights.append(area / full_area)
            mirror_index += 1
    if source_indices:
        indices = torch.tensor(source_indices, dtype=torch.int64).T.contiguous()
        values = torch.tensor(weights, dtype=torch.float64)
    else:
        indices = torch.empty((2, 0), dtype=torch.int64)
        values = torch.empty((0,), dtype=torch.float64)
    matrix_out = torch.sparse_coo_tensor(
        indices, values, (rows.numel() * columns.numel(), source_size * source_size)
    ).coalesce()
    return matrix_out, lattice


def build_target_area_matrix(
    lattice: dict[str, object],
    matrix: Sequence[Sequence[float]],
    offset_um: Sequence[float] = (0.0, 0.0),
    fill_factor: float = 0.92,
    *,
    pitch_um: float = DMD_PITCH_UM,
    target_width_um: float = TARGET_WIDTH_UM,
    target_pixels: int = TARGET_PIXELS,
) -> torch.Tensor:
    """Integrate active DMD mirror areas over target-plane pixels exactly."""
    if (not math.isfinite(fill_factor) or not 0 < fill_factor <= 1 or target_pixels <= 0
            or not math.isfinite(target_width_um) or target_width_um <= 0
            or not math.isfinite(pitch_um) or pitch_um <= 0):
        raise ValueError("Invalid fill factor or target grid")
    affine, offset, _ = _matrix_and_offset(matrix, offset_um)
    rows, columns = lattice["rows"], lattice["columns"]
    center_xs, center_ys = lattice["center_x_um"], lattice["center_y_um"]
    active_half = pitch_um * math.sqrt(fill_factor) / 2
    target_pixel = target_width_um / target_pixels
    target_half = target_width_um / 2
    indices: list[tuple[int, int]] = []
    values: list[float] = []
    mirror_index = 0
    for ri in range(len(rows)):
        for ci in range(len(columns)):
            cx, cy = float(center_xs[ci]), float(center_ys[ri])
            polygon = [_transform(x, y, affine, offset) for x, y in
                       _cell_corners(cx, cy, active_half, active_half)]
            min_x = max(0, math.floor((min(p[0] for p in polygon) + target_half) / target_pixel))
            max_x = min(target_pixels - 1,
                        math.floor((max(p[0] for p in polygon) + target_half) / target_pixel))
            min_y = max(0, math.floor((target_half - max(p[1] for p in polygon)) / target_pixel))
            max_y = min(target_pixels - 1,
                        math.floor((target_half - min(p[1] for p in polygon)) / target_pixel))
            for py in range(min_y, max_y + 1):
                y0 = target_half - (py + 1) * target_pixel
                y1 = target_half - py * target_pixel
                for px in range(min_x, max_x + 1):
                    x0 = -target_half + px * target_pixel
                    x1 = x0 + target_pixel
                    area = polygon_area(_clip_rectangle(polygon, x0, x1, y0, y1))
                    if area > 1e-12:
                        indices.append((py * target_pixels + px, mirror_index))
                        values.append(area / (target_pixel * target_pixel))
            mirror_index += 1
    shape = (target_pixels * target_pixels, len(rows) * len(columns))
    if indices:
        ij = torch.tensor(indices, dtype=torch.int64).T.contiguous()
        vals = torch.tensor(values, dtype=torch.float64)
    else:
        ij = torch.empty((2, 0), dtype=torch.int64)
        vals = torch.empty((0,), dtype=torch.float64)
    return torch.sparse_coo_tensor(ij, vals, shape).coalesce()


def map_binary_images(
    images: torch.Tensor,
    source_matrix: torch.Tensor,
    target_area_matrix: torch.Tensor,
    affine: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
    target_pixels: int = TARGET_PIXELS,
) -> dict[str, torch.Tensor]:
    """Create binary mirror commands and area-integrated target-plane maps."""
    if images.ndim != 3 or images.shape[-2:] != (28, 28):
        raise ValueError("Expected a batch of 28x28 MNIST images")
    if not torch.isfinite(images).all() or not ((images == 0) | (images == 1)).all():
        raise ValueError("DMD input must be finite binary MNIST")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    _, _, det = _matrix_and_offset(affine, (0.0, 0.0))
    device = images.device
    sample_matrix = source_matrix.to(device=device, dtype=torch.float64)
    area_matrix = target_area_matrix.to(device=device, dtype=torch.float64)
    sampled = torch.sparse.mm(sample_matrix, images.reshape(len(images), -1).double().T).T
    commands = (sampled >= threshold).to(torch.float32)
    coverage = torch.sparse.mm(area_matrix, commands.double().T).T.reshape(
        len(images), target_pixels, target_pixels
    )
    return {"commands": commands, "relative_pattern": coverage.to(torch.float32),
            "irradiance": (coverage / det[1]).to(torch.float32),
            "determinant_abs": torch.full((len(images),), det[1], dtype=torch.float64, device=device)}


def target_energy_account(
    commands: torch.Tensor,
    irradiance: torch.Tensor,
    fill_factor: float,
    affine: Sequence[Sequence[float]],
    *,
    pitch_um: float = DMD_PITCH_UM,
    target_pixel_um: float = TARGET_PIXEL_UM,
) -> dict[str, torch.Tensor]:
    """Account geometric energy at a single DMD-reference plane.

    ``irradiance`` is divided by the affine area Jacobian by
    :func:`map_binary_images`, so its target-plane integral is an input-plane
    equivalent area.  Compare that quantity with the DMD active area before
    magnification; also return the separately projected active area for the
    target plane.  No measured optical power is implied.
    """
    if commands.ndim != 2 or irradiance.ndim != 3 or len(commands) != len(irradiance):
        raise ValueError("Expected matching command and irradiance batches")
    if (not math.isfinite(fill_factor) or not 0 < fill_factor <= 1
            or not math.isfinite(target_pixel_um) or target_pixel_um <= 0
            or not math.isfinite(pitch_um) or pitch_um <= 0):
        raise ValueError("Invalid fill factor or target pixel")
    if (not torch.isfinite(commands).all() or not ((commands == 0) | (commands == 1)).all()
            or not torch.isfinite(irradiance).all() or (irradiance < 0).any()):
        raise ValueError("Commands must be binary and irradiance finite and nonnegative")
    _, _, det = _matrix_and_offset(affine, (0.0, 0.0))
    input_active_area = commands.double().sum(1) * pitch_um**2 * fill_factor
    projected_active_area = input_active_area * det[1]
    target_energy = irradiance.double().sum((1, 2)) * target_pixel_um**2
    outside = (input_active_area - target_energy).clamp_min(0)
    fraction = torch.where(input_active_area > 0, outside / input_active_area,
                           torch.zeros_like(input_active_area))
    return {"on_active_area_um2": input_active_area,
            "projected_on_active_area_um2": projected_active_area,
            "target_energy_relative_um2": target_energy,
            "outside_energy_relative_um2": outside, "outside_fraction": fraction,
            "target_capture_fraction": 1 - fraction}


def image_quality_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, torch.Tensor]:
    """C2-compatible fixed-threshold shape proxies plus relative intensity error."""
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("Expected matching batches of two-dimensional images")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all() or (candidate < 0).any():
        raise ValueError("Image values must be finite and candidate intensity nonnegative")
    ref = reference.bool()
    ref_vector, out_vector = ref.flatten(1).double(), candidate.flatten(1).double()
    denominator = torch.linalg.vector_norm(ref_vector, dim=1)
    nrmse = torch.linalg.vector_norm(ref_vector - out_vector, dim=1) / denominator.clamp_min(1)
    maximum = candidate.amax((1, 2), keepdim=True)
    predicted = (candidate >= 0.5 * maximum) & (maximum > 0)
    intersection = (ref & predicted).sum((1, 2)).double()
    ref_count = ref.sum((1, 2)).double()
    pred_count = predicted.sum((1, 2)).double()
    dice = 2 * intersection / (ref_count + pred_count).clamp_min(1)
    iou = intersection / (ref_count + pred_count - intersection).clamp_min(1)
    padded = torch.nn.functional.pad(ref.to(torch.float32)[:, None], (1, 1, 1, 1))
    neighbors = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1]
                 + padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:])[:, 0]
    thin = ref & (neighbors <= 2)
    thin_count = thin.sum((1, 2)).double()
    thin_kept = (thin & predicted).sum((1, 2)).double()
    thin_retention = torch.where(thin_count > 0, thin_kept / thin_count, torch.full_like(thin_count, float("nan")))
    foreground_coverage = torch.where(ref_count > 0, pred_count / ref_count, torch.full_like(ref_count, float("nan")))
    illuminated_fraction = (candidate > 0).float().mean((1, 2))
    return {"nrmse": nrmse, "dice": dice, "iou": iou, "thin_pixel_retention": thin_retention,
            "foreground_coverage_ratio": foreground_coverage,
            "illuminated_pixel_fraction": illuminated_fraction}


def ray_matrix(object_distance_mm: float, lens_gap_mm: float, image_distance_mm: float,
               focal1_mm: float, focal2_mm: float | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Paraxial ABCD matrix for one or two thin lenses, distances in millimetres."""
    values = [object_distance_mm, lens_gap_mm, image_distance_mm, focal1_mm]
    if focal2_mm is not None:
        values.append(focal2_mm)
    if not all(math.isfinite(value) for value in values) or min(values[:3]) < 0:
        raise ValueError("Ray-matrix distances must be finite and nonnegative")
    if focal1_mm == 0 or (focal2_mm is not None and focal2_mm == 0):
        raise ValueError("Focal lengths must be nonzero")

    def multiply(left, right):
        return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(2)) for j in range(2))
                     for i in range(2))

    def prop(distance):
        return ((1.0, distance), (0.0, 1.0))

    def lens(focal):
        return ((1.0, 0.0), (-1.0 / focal, 1.0))

    result = multiply(lens(focal1_mm), prop(object_distance_mm))
    if focal2_mm is None:
        return multiply(prop(image_distance_mm), result)
    result = multiply(prop(lens_gap_mm), result)
    result = multiply(lens(focal2_mm), result)
    return multiply(prop(image_distance_mm), result)


def ideal_4f_segments(focal1_mm: float, focal2_mm: float) -> dict[str, object]:
    if any(not math.isfinite(x) or x <= 0 for x in (focal1_mm, focal2_mm)):
        raise ValueError("4f focal lengths must be positive and finite")
    matrix = ray_matrix(focal1_mm, focal1_mm + focal2_mm, focal2_mm, focal1_mm, focal2_mm)
    return {"object_to_H1_mm": focal1_mm, "H1_to_H2_mm": focal1_mm + focal2_mm,
            "H2_to_image_mm": focal2_mm, "signed_magnification": -focal2_mm / focal1_mm,
            "total_length_mm": 2 * (focal1_mm + focal2_mm),
            "ABCD": [list(matrix[0]), list(matrix[1])], "conjugate_B_mm": matrix[0][1]}


def dmd_coverage_counts(
    lattice: dict[str, object], matrix: Sequence[Sequence[float]],
    offset_um: Sequence[float] = (0.0, 0.0), *, pitch_um: float = DMD_PITCH_UM,
    target_width_um: float = TARGET_WIDTH_UM, fill_factor: float = 0.92,
) -> dict[str, float | int]:
    """Count full and clipped mirror footprints intersecting a square target."""
    if (not math.isfinite(pitch_um) or pitch_um <= 0
            or not math.isfinite(target_width_um) or target_width_um <= 0
            or not math.isfinite(fill_factor) or not 0 < fill_factor <= 1):
        raise ValueError("Invalid DMD pitch, target width, or fill factor")
    affine, offset, det = _matrix_and_offset(matrix, offset_um)
    rows, columns = lattice["rows"], lattice["columns"]
    centers_x, centers_y = lattice["center_x_um"], lattice["center_y_um"]
    target_half = target_width_um / 2
    full_area = det[1] * pitch_um**2
    active_full_area = full_area * fill_factor
    active_half = pitch_um * math.sqrt(fill_factor) / 2
    intersecting = full = partial = 0
    active_intersecting = active_full = active_partial = 0
    covered_area = active_covered_area = 0.0
    for ri in range(len(rows)):
        for ci in range(len(columns)):
            cx, cy = float(centers_x[ci]), float(centers_y[ri])
            polygon = [_transform(x, y, affine, offset)
                       for x, y in _cell_corners(cx, cy, pitch_um / 2, pitch_um / 2)]
            overlap = polygon_area(_clip_rectangle(polygon, -target_half, target_half,
                                                   -target_half, target_half))
            if overlap > 1e-10:
                intersecting += 1
                covered_area += overlap
                if abs(overlap - full_area) <= max(1e-8, full_area * 1e-10):
                    full += 1
                else:
                    partial += 1
            active_polygon = [_transform(x, y, affine, offset)
                              for x, y in _cell_corners(cx, cy, active_half, active_half)]
            active_overlap = polygon_area(_clip_rectangle(
                active_polygon, -target_half, target_half, -target_half, target_half
            ))
            if active_overlap > 1e-10:
                active_intersecting += 1
                active_covered_area += active_overlap
                if abs(active_overlap - active_full_area) <= max(1e-8, active_full_area * 1e-10):
                    active_full += 1
                else:
                    active_partial += 1
    return {"intersecting_mirrors": intersecting, "fully_inside_mirrors": full,
            "partially_clipped_mirrors": partial,
            "full_pitch_area_coverage_fraction": covered_area / (target_width_um**2),
            "active_intersecting_mirrors": active_intersecting,
            "fully_inside_active_mirrors": active_full,
            "partially_clipped_active_mirrors": active_partial,
            "active_area_coverage_fraction": active_covered_area / (target_width_um**2)}


def build_cmos_area_matrix(
    matrix: Sequence[Sequence[float]],
    offset_um: Sequence[float] = (0.0, 0.0),
    *,
    object_pixels: int = TARGET_PIXELS,
    object_pixel_um: float = TARGET_PIXEL_UM,
    sensor_shape: tuple[int, int] = (CMOS_ROWS, CMOS_COLUMNS),
    sensor_pixel_um: float = CMOS_PIXEL_UM,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Integrate output-plane pixel energy into finite CMOS pixel areas."""
    affine, offset, det = _matrix_and_offset(matrix, offset_um)
    sensor_rows, sensor_cols = sensor_shape
    if (len(sensor_shape) != 2 or min(sensor_rows, sensor_cols, object_pixels) <= 0
            or not math.isfinite(object_pixel_um) or not math.isfinite(sensor_pixel_um)
            or min(object_pixel_um, sensor_pixel_um) <= 0):
        raise ValueError("Pixel dimensions must be positive")
    fov_x, fov_y = sensor_cols * sensor_pixel_um, sensor_rows * sensor_pixel_um
    centers_x = (torch.arange(object_pixels, dtype=torch.float64) + 0.5 - object_pixels / 2) * object_pixel_um
    centers_y = (object_pixels / 2 - torch.arange(object_pixels, dtype=torch.float64) - 0.5) * object_pixel_um
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    captured_by_source = torch.zeros(object_pixels * object_pixels, dtype=torch.float64)
    for row in range(object_pixels):
        for col in range(object_pixels):
            cx, cy = float(centers_x[col]), float(centers_y[row])
            polygon = [_transform(x, y, affine, offset)
                       for x, y in _cell_corners(cx, cy, object_pixel_um / 2, object_pixel_um / 2)]
            min_col = max(0, math.floor((min(p[0] for p in polygon) + fov_x / 2) / sensor_pixel_um))
            max_col = min(sensor_cols - 1,
                          math.floor((max(p[0] for p in polygon) + fov_x / 2) / sensor_pixel_um))
            min_row = max(0, math.floor((fov_y / 2 - max(p[1] for p in polygon)) / sensor_pixel_um))
            max_row = min(sensor_rows - 1,
                          math.floor((fov_y / 2 - min(p[1] for p in polygon)) / sensor_pixel_um))
            source_index = row * object_pixels + col
            for camera_row in range(min_row, max_row + 1):
                y1 = fov_y / 2 - camera_row * sensor_pixel_um
                y0 = y1 - sensor_pixel_um
                for camera_col in range(min_col, max_col + 1):
                    x0 = -fov_x / 2 + camera_col * sensor_pixel_um
                    area = polygon_area(_clip_rectangle(polygon, x0, x0 + sensor_pixel_um, y0, y1))
                    if area > 1e-12:
                        # Divide by the image Jacobian so the matrix maps irradiance
                        # to energy and an ideal relay conserves total power.
                        object_area = area / det[1]
                        row_indices.append(camera_row * sensor_cols + camera_col)
                        col_indices.append(source_index)
                        values.append(object_area)
                        captured_by_source[source_index] += object_area / (object_pixel_um**2)
    if row_indices:
        indices = torch.tensor((row_indices, col_indices), dtype=torch.int64)
        vals = torch.tensor(values, dtype=torch.float64)
    else:
        indices = torch.empty((2, 0), dtype=torch.int64)
        vals = torch.empty((0,), dtype=torch.float64)
    matrix_out = torch.sparse_coo_tensor(
        indices, vals, (sensor_rows * sensor_cols, object_pixels * object_pixels)
    ).coalesce()
    return matrix_out, {"sensor_shape": sensor_shape, "sensor_fov_um": (fov_x, fov_y),
                        "sensor_pixel_um": sensor_pixel_um, "jacobian_abs": det[1],
                        "source_area_capture_fraction": captured_by_source.reshape(object_pixels, object_pixels)}


def integrate_cmos_pixels(intensity: torch.Tensor, area_matrix: torch.Tensor) -> torch.Tensor:
    """Return ideal pixel-integrated energy; no QE, noise, or gain is applied."""
    if intensity.ndim != 2 or not torch.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("Expected a finite nonnegative 2-D intensity image")
    if area_matrix.shape[1] != intensity.numel():
        raise ValueError("CMOS integration matrix does not match the object grid")
    integrated = torch.sparse.mm(area_matrix.to(intensity.device), intensity.reshape(-1, 1).double())
    return integrated.reshape(-1)


def cmos_noise_model(
    incident_photons: torch.Tensor,
    *,
    qe_532: float,
    exposure_s: float,
    dark_current_e_per_s: float,
    read_noise_e: float,
    full_well_e: float,
    gain_e_per_dn: float,
    adc_bits: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Parametric diagnostic camera response; every physical parameter is explicit."""
    values = (qe_532, exposure_s, dark_current_e_per_s, read_noise_e,
              full_well_e, gain_e_per_dn)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("CMOS noise parameters must be finite and explicit")
    if not 0 <= qe_532 <= 1 or min(exposure_s, full_well_e, gain_e_per_dn) <= 0:
        raise ValueError("Invalid CMOS QE, exposure, full well, or conversion gain")
    if dark_current_e_per_s < 0 or read_noise_e < 0 or not 1 <= adc_bits <= 16:
        raise ValueError("Invalid CMOS noise or ADC parameter")
    if not torch.isfinite(incident_photons).all() or (incident_photons < 0).any():
        raise ValueError("Incident photon count must be finite and nonnegative")
    photo_mean = incident_photons.double() * qe_532
    dark_mean = torch.full_like(photo_mean, dark_current_e_per_s * exposure_s)
    electrons = (torch.poisson(photo_mean, generator=generator)
                 + torch.poisson(dark_mean, generator=generator)
                 + torch.randn(photo_mean.shape, generator=generator, dtype=torch.float64,
                               device=photo_mean.device) * read_noise_e)
    saturated = electrons.clamp(0, full_well_e)
    max_dn = (1 << adc_bits) - 1
    digital = torch.round(saturated / gain_e_per_dn).clamp(0, max_dn).to(torch.int32)
    return {"electrons_before_clipping": electrons, "dn": digital,
            "saturated_fraction": (electrons >= full_well_e).double().mean()}


def reflect_wavevector(k_in: Sequence[float], mirror_normal: Sequence[float]) -> tuple[float, float, float]:
    """Apply k_out = k_in - 2(k_in·n)n to normalized 3-D vectors."""
    if len(k_in) != 3 or len(mirror_normal) != 3:
        raise ValueError("Expected 3-D wavevector and mirror normal")
    k = tuple(float(value) for value in k_in)
    n = tuple(float(value) for value in mirror_normal)
    if not all(math.isfinite(value) for value in (*k, *n)):
        raise ValueError("Vectors must be finite")
    knorm = math.sqrt(sum(value * value for value in k))
    nnorm = math.sqrt(sum(value * value for value in n))
    if abs(knorm - 1) > 1e-8 or abs(nnorm - 1) > 1e-8:
        raise ValueError("Wavevector and normal must be unit vectors")
    projection = sum(k[index] * n[index] for index in range(3))
    return tuple(k[index] - 2 * projection * n[index] for index in range(3))


def roi_scores(intensity: torch.Tensor, rectangles: Sequence[Sequence[int]]) -> dict[str, object]:
    """Integrate four nonoverlapping half-open ROIs; ties and zero maps abstain."""
    if intensity.ndim != 2 or len(rectangles) != 4:
        raise ValueError("Expected a 2-D CMOS image and four ROI rectangles")
    if not torch.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("CMOS intensity must be finite and nonnegative")
    height, width = intensity.shape
    normalized = []
    areas = []
    for rectangle in rectangles:
        if len(rectangle) != 4:
            raise ValueError("ROI must be (row_start,row_end,col_start,col_end)")
        r0, r1, c0, c1 = (int(value) for value in rectangle)
        if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
            raise ValueError("ROI is outside the sensor image")
        for pr0, pr1, pc0, pc1 in normalized:
            if max(r0, pr0) < min(r1, pr1) and max(c0, pc0) < min(c1, pc1):
                raise ValueError("ROIs must not overlap")
        normalized.append((r0, r1, c0, c1))
        areas.append((r1 - r0) * (c1 - c0))
    if len(set(areas)) != 1:
        raise ValueError("Four-class ROIs must have equal pixel area")
    scores = torch.stack([
        intensity[r0:r1, c0:c1].double().sum()
        for r0, r1, c0, c1 in normalized
    ])
    maximum = scores.max()
    tied = int(torch.isclose(scores, maximum, rtol=1e-12, atol=0.0).sum()) != 1
    prediction = -1 if not torch.isfinite(scores).all() or scores.sum() <= 0 or tied else int(scores.argmax())
    total = intensity.double().sum()
    outside = (total - scores.sum()).clamp_min(0)
    roundoff = torch.finfo(torch.float64).eps * max(float(total), 1.0) * 16
    outside = outside.masked_fill(outside <= roundoff, 0.0)
    return {"scores": scores, "prediction": prediction, "outside_energy": outside,
            "correct_roi_fraction": None if prediction < 0 else scores[prediction] / scores.sum()}


def equal_quadrant_rois(image_shape: Sequence[int]) -> tuple[tuple[int, int, int, int], ...]:
    """Return equal top-left, top-right, bottom-left, bottom-right ROIs."""
    if len(image_shape) != 2:
        raise ValueError("Expected a two-dimensional image shape")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("A 2x2 equal ROI layout requires positive even dimensions")
    return (
        (0, height // 2, 0, width // 2),
        (0, height // 2, width // 2, width),
        (height // 2, height, 0, width // 2),
        (height // 2, height, width // 2, width),
    )


def roi_classification_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, object]:
    """Return a 4x5 confusion matrix; the fifth column is abstention."""
    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("Expected matching label and prediction vectors")
    if len(labels) == 0 or ((labels < 0) | (labels > 3)).any() or ((predictions < -1) | (predictions > 3)).any():
        raise ValueError("ROI metrics require nonempty 0-3 labels and predictions")
    cm = torch.bincount(labels.to(torch.int64) * 5 + (predictions.to(torch.int64) + 1), minlength=20).reshape(4, 5)
    true_positive = cm[:, 1:5].diagonal()
    support = cm.sum(1)
    predicted = cm[:, 1:5].sum(0)
    f1 = 2 * true_positive.double() / (support + predicted).clamp_min(1)
    correct = int(true_positive.sum())
    decided = int((predictions >= 0).sum())
    return {"confusion_4x5": cm.tolist(), "accuracy_all": correct / len(labels),
            "macro_f1": float(f1.mean()), "coverage": decided / len(labels),
            "accuracy_decided": correct / decided if decided else None,
            "per_class_recall": (true_positive.double() / support.clamp_min(1)).tolist()}
