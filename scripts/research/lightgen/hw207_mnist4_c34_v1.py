"""C3.4 scalar-wave model and artifact audit for the HW207 MNIST4 diagnostic.

All Model-B distances and aperture values in this module are geometric
assumptions derived from tape estimates.  This is not a calibrated hardware
model, a DLP9000 micromirror model, or a material-specific printed-device model.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as c33
from scripts.research.lightgen.hw207_mnist4_c2_v1 import coherent_pupil
from ldm_project.v2.application.research.lightgen import hw207_spatial_geometry as geo

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/research/lightgen/hw207_mnist4_c34_v1.json"
BACKUP_PATH = ROOT / "outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c33_source_01"
SESSION_SOURCE_ID = "01a18b9f-206d-7c61-96a3-0da3a49efb8c"
WAVELENGTH_UM = 0.532
PITCH_UM = 2.0
GRID = 100
CMOS_SHAPE = (2048, 2048)
CMOS_PITCH_UM = 6.5
CMOS_CROP_OFFSET = 640
MODEL_B_D1_UM = 80_000.0
MODEL_B_F_UM = 30_000.0
MODEL_B_D2_UM = 100_000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("identity") != "HW207_MNIST4_C34_NUMERICAL_DIAGNOSTIC":
        raise ValueError("unexpected C3.4 configuration identity")
    if data.get("wavelength_nm") != 532.0:
        raise ValueError("C3.4 is frozen to the 532 nm scenario")
    return data


def abcd_one_lens(d1_um: float, focal_um: float, d2_um: float) -> dict[str, float]:
    if min(d1_um, focal_um, d2_um) <= 0:
        raise ValueError("distances and focal length must be positive")
    d1, focal, d2 = d1_um, focal_um, d2_um
    a = 1.0 - d2 / focal
    b = d1 + d2 - d1 * d2 / focal
    c = -1.0 / focal
    d = 1.0 - d1 / focal
    conjugate_after_lens = 1.0 / (1.0 / focal - 1.0 / d1)
    return {
        "A": a,
        "B_um": b,
        "C_per_um": c,
        "D": d,
        "determinant": a * d - b * c,
        "conjugate_image_distance_after_lens_um": conjugate_after_lens,
        "magnification_at_conjugate": -conjugate_after_lens / d1,
        "camera_defocus_after_conjugate_um": d2 - conjugate_after_lens,
    }


@dataclass(frozen=True)
class CZTPlan:
    input_count: int
    output_count: int
    fft_count: int
    input_chirp: torch.Tensor
    kernel_fft: torch.Tensor
    output_chirp: torch.Tensor


def to_complex128(field: torch.Tensor) -> torch.Tensor:
    """Promote a complex field without discarding its imaginary component."""
    if not torch.is_complex(field):
        raise ValueError("field must be complex before precision promotion")
    return field.to(dtype=torch.complex128)


def build_czt_plan(input_count: int, output_count: int, input_pitch_um: float,
                   output_pitch_um: float, wavelength_um: float, distance_um: float,
                   *, device: torch.device, dtype: torch.dtype,
                   phase_dtype: torch.dtype | None = None) -> CZTPlan:
    """Plan a centered, scaled DFT sum using Bluestein convolution.

    ``phase_dtype`` controls the precision used to construct chirp angles;
    the returned chirps and convolution remain in ``dtype``.
    """
    if min(input_count, output_count, input_pitch_um, output_pitch_um,
           wavelength_um, distance_um) <= 0:
        raise ValueError("CZT dimensions and optical scales must be positive")
    phase_dtype = dtype if phase_dtype is None else phase_dtype
    if phase_dtype not in (torch.complex64, torch.complex128):
        raise ValueError("phase_dtype must be complex64 or complex128")
    real_dtype = torch.float64 if phase_dtype == torch.complex128 else torch.float32
    alpha = input_pitch_um * output_pitch_um / (wavelength_um * distance_um)
    n_center = (input_count - 1) / 2.0
    k_center = (output_count - 1) / 2.0
    n = torch.arange(input_count, device=device, dtype=real_dtype)
    k = torch.arange(output_count, device=device, dtype=real_dtype)
    q = torch.arange(-(input_count - 1), output_count, device=device, dtype=real_dtype)
    input_chirp = torch.polar(torch.ones_like(n), -math.pi * alpha * n.square()
                              + 2.0 * math.pi * alpha * k_center * n).to(dtype)
    kernel = torch.polar(torch.ones_like(q), math.pi * alpha * q.square()).to(dtype)
    output_chirp = torch.polar(torch.ones_like(k), -math.pi * alpha * k.square()
                               + 2.0 * math.pi * alpha * n_center * k
                               - 2.0 * math.pi * alpha * n_center * k_center).to(dtype)
    fft_count = 1 << (input_count + output_count - 2).bit_length()
    kernel_fft = torch.fft.fft(F.pad(kernel.to(dtype), (0, fft_count - kernel.numel())))
    return CZTPlan(input_count, output_count, fft_count,
                   input_chirp.to(dtype), kernel_fft.to(dtype), output_chirp.to(dtype))


def czt_sum_last(x: torch.Tensor, plan: CZTPlan) -> torch.Tensor:
    """Compute sum_n x[n] exp(-i 2 pi alpha (n-nc)(k-kc))."""
    if x.shape[-1] != plan.input_count:
        raise ValueError("CZT input length does not match its plan")
    transformed = torch.fft.fft(x * plan.input_chirp, n=plan.fft_count, dim=-1)
    convolution = torch.fft.ifft(transformed * plan.kernel_fft, dim=-1)
    selected = convolution[..., plan.input_count - 1:
                           plan.input_count - 1 + plan.output_count]
    return selected * plan.output_chirp


def _coordinate_grid(count: int, pitch_um: float, *, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    return (torch.arange(count, device=device, dtype=dtype) - (count - 1) / 2.0) * pitch_um


def scaled_fresnel(field: torch.Tensor, input_pitch_um: float,
                   output_shape: tuple[int, int], output_pitch_um: float,
                   distance_um: float, *, wavelength_um: float = WAVELENGTH_UM,
                   plan_cache: dict[tuple, CZTPlan] | None = None,
                   phase_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Paraxial scalar propagation on independent centered input/output grids.

    The field is represented at sample centers.  The integral uses the input
    sample area, while the output is sampled at the requested physical pitch.
    A global exp(i*k*z) factor is omitted because it cannot affect intensity or
    the subsequent phase-to-intensity calculations. ``phase_dtype`` can raise
    only phase/chirp construction precision without promoting the complex field
    or FFT work dtype.
    """
    if field.ndim != 3 or not torch.is_complex(field):
        raise ValueError("field must be complex [batch, y, x]")
    in_y, in_x = field.shape[-2:]
    out_y, out_x = output_shape
    device, dtype = field.device, field.dtype
    phase_dtype = dtype if phase_dtype is None else phase_dtype
    if phase_dtype not in (torch.complex64, torch.complex128):
        raise ValueError("phase_dtype must be complex64 or complex128")
    real_dtype = torch.float64 if phase_dtype == torch.complex128 else torch.float32
    y_in = _coordinate_grid(in_y, input_pitch_um, device=device, dtype=real_dtype)
    x_in = _coordinate_grid(in_x, input_pitch_um, device=device, dtype=real_dtype)
    y_out = _coordinate_grid(out_y, output_pitch_um, device=device, dtype=real_dtype)
    x_out = _coordinate_grid(out_x, output_pitch_um, device=device, dtype=real_dtype)
    pre = torch.exp((1j * math.pi / (wavelength_um * distance_um))
                    * (y_in[:, None].square() + x_in[None, :].square())).to(dtype)
    work = field * pre
    if plan_cache is None:
        plan_cache = {}

    def plan(n: int, m: int) -> CZTPlan:
        key = (n, m, float(input_pitch_um), float(output_pitch_um), float(wavelength_um),
               float(distance_um), str(device), dtype, phase_dtype)
        if key not in plan_cache:
            plan_cache[key] = build_czt_plan(n, m, input_pitch_um, output_pitch_um,
                                             wavelength_um, distance_um,
                                             device=device, dtype=dtype,
                                             phase_dtype=phase_dtype)
        return plan_cache[key]

    py = plan(y_in.numel(), y_out.numel())
    px = plan(x_in.numel(), x_out.numel())
    work = czt_sum_last(work.transpose(-2, -1), py).transpose(-2, -1)
    work = czt_sum_last(work, px)
    post = torch.exp((1j * math.pi / (wavelength_um * distance_um))
                     * (y_out[:, None].square() + x_out[None, :].square())).to(dtype)
    scale = -1j * (input_pitch_um ** 2) / (wavelength_um * distance_um)
    return work * post * scale


@lru_cache(maxsize=2)
def fixed_roi_bounds(device: str = "cpu") -> tuple[tuple[int, int, int, int], ...]:
    configuration = c33.build_configuration(52_000.0, device)
    result = tuple(tuple(int(v) + CMOS_CROP_OFFSET for v in roi)
                   for roi in configuration.base.rois)
    if len(result) != 4 or any(r0 < 0 or c0 < 0 or r1 > CMOS_SHAPE[0] or c1 > CMOS_SHAPE[1]
                               for r0, r1, c0, c1 in result):
        raise ValueError("fixed C3.3 ROI does not fit in the candidate CMOS frame")
    return result


def input_field(images: torch.Tensor, configuration, *, cutoff_cycles_um: float = 0.1) -> torch.Tensor:
    """C33 binary DMD/scalar input-relay proxy, with no labels in the function."""
    if images.ndim != 3 or images.shape[-2:] != (28, 28) or images.dtype != torch.uint8:
        raise ValueError("images must be uint8 [batch,28,28]")
    c = configuration.base
    binary = (images.to(c.device) >= 128).float()
    mapped = geo.map_binary_images(binary, c.source_matrix, c.target_matrix, c.affine)
    amplitude = mapped["irradiance"].clamp_min(0).sqrt()
    field = torch.complex(amplitude, torch.zeros_like(amplitude))
    padded = F.pad(field, (206, 206, 206, 206))
    filtered = coherent_pupil(padded, 2.0, cutoff_cycles_um, 0.0, 0.532)
    return filtered[:, 206:306, 206:306]


def model_b_forward(field: torch.Tensor, phase_rad: torch.Tensor,
                    aperture_diameter_mm: float, *, lens_pitch_um: float = 4.0,
                    plan_cache: dict[tuple, CZTPlan] | None = None,
                    return_camera: bool = True,
                    screen_to_lens_um: float = MODEL_B_D1_UM,
                    lens_to_cmos_um: float = MODEL_B_D2_UM,
                    lens_to_cmos_chirp_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor | None]:
    """P80 -> finite thin lens -> P100 scalar Fresnel -> fixed candidate CMOS.

    The aperture diameter and all geometry here are assumption scenarios.  The
    returned predictions are diagnostic numerical results only. The optional
    CMOS chirp dtype affects phase construction only; the propagated field and
    FFT retain the input field dtype.
    """
    if field.shape[-2:] != (GRID, GRID):
        raise ValueError("C34 phase screen input must remain 100x100")
    if phase_rad.shape not in {(GRID, GRID), (1, 1, GRID, GRID)}:
        raise ValueError("phase screen must be 100x100")
    if aperture_diameter_mm not in (0.5, 1.0, 2.0):
        raise ValueError("use one of the predeclared aperture scenarios")
    if lens_pitch_um not in (1.0, 2.0, 4.0):
        raise ValueError("lens-plane sampling must be 1, 2 or 4 um for the convergence check")
    if min(screen_to_lens_um, lens_to_cmos_um) <= 0:
        raise ValueError("candidate propagation distances must be positive")
    phase = phase_rad.reshape(GRID, GRID).to(device=field.device, dtype=field.real.dtype)
    screen = field * torch.polar(torch.ones_like(phase), phase)
    diameter_um = aperture_diameter_mm * 1000.0
    side = int(round(diameter_um / lens_pitch_um))
    if side < 2 or not math.isclose(side * lens_pitch_um, diameter_um, abs_tol=1e-8):
        raise ValueError("aperture diameter must align to the lens sample pitch")
    lens_field = scaled_fresnel(screen, PITCH_UM, (side, side), lens_pitch_um,
                                screen_to_lens_um, plan_cache=plan_cache)
    coord = _coordinate_grid(side, lens_pitch_um, device=field.device, dtype=field.real.dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    pupil = ((xx.square() + yy.square()) <= (diameter_um / 2.0) ** 2).to(field.real.dtype)
    lens_phase = torch.exp((-1j * math.pi / (WAVELENGTH_UM * MODEL_B_F_UM))
                           * (xx.square() + yy.square())).to(field.dtype)
    at_lens = lens_field * pupil * lens_phase
    camera_field = scaled_fresnel(at_lens, lens_pitch_um, CMOS_SHAPE,
                                  CMOS_PITCH_UM, lens_to_cmos_um,
                                  plan_cache=plan_cache,
                                  phase_dtype=lens_to_cmos_chirp_dtype)
    intensity = camera_field.abs().square()
    roi = fixed_roi_bounds("cpu")
    scores = torch.stack([intensity[:, r0:r1, c0:c1].sum((1, 2))
                          * CMOS_PITCH_UM ** 2 for r0, r1, c0, c1 in roi], dim=1)
    detected = intensity.sum((1, 2)) * CMOS_PITCH_UM ** 2
    crop = intensity[:, CMOS_CROP_OFFSET:CMOS_CROP_OFFSET + 768,
                     CMOS_CROP_OFFSET:CMOS_CROP_OFFSET + 768].sum((1, 2)) * CMOS_PITCH_UM ** 2
    input_energy = field.abs().square().sum((1, 2)) * PITCH_UM ** 2
    lens_energy = at_lens.abs().square().sum((1, 2)) * lens_pitch_um ** 2
    return {
        "camera_complex_field": camera_field if return_camera else None,
        "camera_intensity": intensity if return_camera else None,
        "roi_scores": scores,
        "detected_energy": detected,
        "comparison_crop_energy": crop,
        "outside_roi_energy": (crop - scores.sum(1)).clamp_min(0),
        "outside_roi_full_cmos_energy": (detected - scores.sum(1)).clamp_min(0),
        "input_energy": input_energy,
        "lens_energy": lens_energy,
    }


def height_from_phase(phase_rad: torch.Tensor | np.ndarray, delta_n: float) -> torch.Tensor | np.ndarray:
    if delta_n <= 0:
        raise ValueError("delta_n must be positive")
    if isinstance(phase_rad, torch.Tensor):
        wrapped = torch.remainder(phase_rad, 2 * math.pi)
        return wrapped * (WAVELENGTH_UM / (2 * math.pi * delta_n))
    wrapped = np.remainder(np.asarray(phase_rad, dtype=np.float64), 2 * math.pi)
    return wrapped * (WAVELENGTH_UM / (2 * math.pi * delta_n))


def phase_from_height(height_um: torch.Tensor | np.ndarray, delta_n: float) -> torch.Tensor | np.ndarray:
    if delta_n <= 0:
        raise ValueError("delta_n must be positive")
    if isinstance(height_um, torch.Tensor):
        return 2 * math.pi * delta_n * height_um / WAVELENGTH_UM
    return 2 * math.pi * delta_n * np.asarray(height_um, dtype=np.float64) / WAVELENGTH_UM


def fabricate_height(phase_rad: torch.Tensor, delta_n: float, *, psf_sigma_px: float,
                     minimum_feature_radius_px: int, height_levels: int,
                     height_noise_nm: float = 0.0, noise_seed: int = 0,
                     edge_error_nm: float = 0.0, maximum_height_um: float = 1.2) -> torch.Tensor:
    """Apply matched, explicitly assumed height-domain print operators.

    ``edge_error_nm`` is a vertical height bias on the outermost sample row and
    column.  It is not a lateral device-edge displacement; the latter is below
    this model's 2 um sampling resolution and must remain unmodelled here.
    """
    if (not math.isfinite(delta_n) or delta_n <= 0 or psf_sigma_px < 0
            or minimum_feature_radius_px < 0 or height_noise_nm < 0
            or not math.isfinite(edge_error_nm) or maximum_height_um <= 0):
        raise ValueError("fabrication parameters must be finite and nonnegative")
    height = height_from_phase(phase_rad, delta_n)
    if psf_sigma_px > 0:
        height = c33.smooth_height(height, psf_sigma_px)
    if minimum_feature_radius_px > 0:
        height = c33.min_feature_open(height, minimum_feature_radius_px)
    if height_levels < 0 or height_levels == 1:
        raise ValueError("height_levels must be 0 (no quantization) or >=2")
    if height_levels >= 2:
        height = c33.quantize_height(height, height_levels, maximum_height_um)
    if height_noise_nm > 0:
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        noise = torch.randn(height.shape, generator=generator, dtype=torch.float32).to(height.device)
        height = height + noise * (height_noise_nm / 1000.0)
    if edge_error_nm != 0:
        delta = edge_error_nm / 1000.0
        height = height.clone()
        height[..., 0, :] += delta
        height[..., -1, :] += delta
        height[..., :, 0] += delta
        height[..., :, -1] += delta
    return height.clamp(0.0, maximum_height_um)


def mesh_200um(height_um: np.ndarray, *, substrate_um: float = 1.2) -> tuple[np.ndarray, dict[str, Any]]:
    """Make a closed heightfield mesh whose outer boundary is exactly 200 um.

    The 100 phase samples are located at the centers 1,3,...,199 um.  Boundary
    vertices at 0 and 200 um extend the nearest edge sample by half a cell.
    This preserves all 100x100 samples without claiming a 198 um device.
    """
    height = np.asarray(height_um, dtype=np.float64)
    if height.shape != (GRID, GRID) or not np.isfinite(height).all():
        raise ValueError("height map must be a finite 100x100 array")
    if substrate_um <= 0 or float(height.min()) < 0:
        raise ValueError("height must be nonnegative and substrate thickness positive")
    values = np.pad(height, ((1, 1), (1, 1)), mode="edge")
    centers = np.arange(GRID, dtype=np.float64) * PITCH_UM + PITCH_UM / 2
    coords = np.concatenate(([0.0], centers, [GRID * PITCH_UM]))
    xx, yy = np.meshgrid(coords, coords)
    top = np.stack([xx, yy, values], axis=-1)
    bottom = np.stack([xx, yy, np.full_like(xx, -substrate_um)], axis=-1)
    ny, nx = values.shape
    n = nx * ny
    top_idx = np.arange(n).reshape(ny, nx)
    bottom_idx = n + top_idx
    vertices = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3)], axis=0)
    triangles: list[np.ndarray] = []

    def quad(a: int, b: int, c: int, d: int, outward: tuple[float, float, float]) -> None:
        points = vertices[[a, b, c, d]]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(normal, np.asarray(outward, dtype=np.float64))) < 0:
            points = points[::-1]
        triangles.append(points[[0, 1, 2]])
        triangles.append(points[[0, 2, 3]])

    for j in range(ny - 1):
        for i in range(nx - 1):
            quad(top_idx[j, i], top_idx[j, i + 1], top_idx[j + 1, i + 1],
                 top_idx[j + 1, i], (0, 0, 1))
            quad(bottom_idx[j, i], bottom_idx[j + 1, i], bottom_idx[j + 1, i + 1],
                 bottom_idx[j, i + 1], (0, 0, -1))
    for i in range(nx - 1):
        quad(top_idx[0, i], top_idx[0, i + 1], bottom_idx[0, i + 1], bottom_idx[0, i], (0, -1, 0))
        quad(top_idx[ny - 1, i + 1], top_idx[ny - 1, i], bottom_idx[ny - 1, i],
             bottom_idx[ny - 1, i + 1], (0, 1, 0))
    for j in range(ny - 1):
        quad(top_idx[j + 1, 0], top_idx[j, 0], bottom_idx[j, 0], bottom_idx[j + 1, 0], (-1, 0, 0))
        quad(top_idx[j, nx - 1], top_idx[j + 1, nx - 1], bottom_idx[j + 1, nx - 1],
             bottom_idx[j, nx - 1], (1, 0, 0))
    result = np.stack(triangles).reshape(-1, 3, 3)
    return result, {
        "unit": "um",
        "footprint_um": [float(GRID * PITCH_UM), float(GRID * PITCH_UM)],
        "sample_count": [GRID, GRID],
        "sample_pitch_um": PITCH_UM,
        "sample_center_first_um": float(centers[0]),
        "sample_center_last_um": float(centers[-1]),
        "boundary_vertices_um": [0.0, float(GRID * PITCH_UM)],
        "substrate_thickness_um": float(substrate_um),
        "height_range_um": [float(height.min()), float(height.max())],
        "vertices": int(len(vertices)),
        "triangles": int(len(result)),
        "top_surface_is_nonoverlapping_heightfield": True,
        "self_intersection_independent_triangle_triangle_test": "NOT_RUN",
    }


def write_binary_stl(path: Path, triangles: np.ndarray, *, header: str = "hw207-c34-parameterized") -> None:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii")[:80].ljust(80, b" "))
        handle.write(struct.pack("<I", len(triangles)))
        for normal, triangle in zip(normals, triangles, strict=True):
            handle.write(struct.pack("<12fH", *normal, *triangle.reshape(-1), 0))


def recover_height_from_mesh_vertices(path: Path) -> np.ndarray:
    triangles, _ = _read_binary_stl_independent(path)
    flat = triangles.reshape(-1, 3)
    verts = np.unique(np.round(flat * 1e5).astype(np.int64), axis=0).astype(np.float64) / 1e5
    top = verts[verts[:, 2] >= -1e-8]
    centers = np.arange(GRID, dtype=np.float64) * PITCH_UM + PITCH_UM / 2
    rows = []
    for y in centers:
        row = []
        for x in centers:
            candidates = top[(np.abs(top[:, 0] - x) <= 1e-4) & (np.abs(top[:, 1] - y) <= 1e-4)]
            if len(candidates) != 1:
                raise ValueError("could not uniquely recover a center height from the STL")
            row.append(float(candidates[0, 2]))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def predictions_from_scores(scores: torch.Tensor, detected_energy: torch.Tensor) -> torch.Tensor:
    maximum = scores.max(1, keepdim=True).values
    tied = torch.isclose(scores, maximum, rtol=1e-12, atol=0).sum(1) != 1
    return torch.where((detected_energy > 0) & ~tied, scores.argmax(1), -1)


def classification_metrics(labels: torch.Tensor, predictions: torch.Tensor,
                           scores: torch.Tensor, detected: torch.Tensor,
                           *, fraction_denominator: torch.Tensor | None = None) -> dict[str, Any]:
    y = labels.detach().cpu().long()
    pred = predictions.detach().cpu().long()
    score_cpu = scores.detach().cpu().double()
    detected_cpu = detected.detach().cpu().double()
    confusion = torch.zeros((4, 5), dtype=torch.int64)
    for truth, decision in zip(y.tolist(), pred.tolist(), strict=True):
        confusion[truth, decision if decision >= 0 else 4] += 1
    tp = confusion.diag()[:4].double()
    predicted_count = confusion[:, :4].sum(0).double()
    truth_count = confusion.sum(1).double()
    precision = tp / predicted_count.clamp_min(1)
    recall = tp / truth_count.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    correct = (pred == y) & (pred >= 0)
    denominator = detected_cpu if fraction_denominator is None else fraction_denominator.detach().cpu().double()
    denominator = denominator.clamp_min(1e-30)
    correct_energy_fraction = score_cpu[torch.arange(len(y)), y] / denominator
    wrong = score_cpu.clone()
    wrong[torch.arange(len(y)), y] = -float("inf")
    top_wrong = wrong.max(1).values / denominator
    return {
        "sample_count": int(len(y)),
        "accuracy_all_samples": float(correct.double().mean()),
        "macro_f1_classes_0_to_3": float(f1.mean()),
        "coverage": float((pred >= 0).double().mean()),
        "accuracy_when_decided": float(correct[pred >= 0].double().mean()) if (pred >= 0).any() else 0.0,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "confusion_4x5_with_abstention_column": confusion.tolist(),
        "correct_roi_energy_fraction_mean": float(correct_energy_fraction.mean()),
        "strongest_wrong_roi_energy_fraction_mean": float(top_wrong.mean()),
        "correct_minus_strongest_wrong_energy_fraction_mean": float((correct_energy_fraction - top_wrong).mean()),
        "mean_detected_energy_um2": float(detected_cpu.mean()),
        "mean_roi_energy_um2": score_cpu.mean(0).tolist(),
    }


def stratified_c34_split(c0_train_indices: torch.Tensor, labels: torch.Tensor,
                         seed: int = 20261105) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    parts = {"train": [], "validation": [], "holdout": []}
    for cls in range(4):
        rows = c0_train_indices[labels[c0_train_indices] == cls]
        order = torch.randperm(len(rows), generator=generator)
        rows = rows[order]
        parts["holdout"].append(rows[:500])
        parts["validation"].append(rows[500:1000])
        parts["train"].append(rows[1000:])
    result = {key: torch.sort(torch.cat(value).long()).values for key, value in parts.items()}
    all_parts = torch.cat(list(result.values()))
    if len(all_parts) != 22_279 or len(torch.unique(all_parts)) != 22_279:
        raise ValueError("C3.4 split is incomplete or overlapping")
    return result


def _read_binary_stl_independent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError("not a binary STL")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError("STL byte count does not match its triangle count")
    records = np.frombuffer(raw, dtype=np.uint8, offset=84).reshape(count, 50)
    values = records[:, :48].copy().view("<f4").reshape(count, 12)
    return values[:, 3:].reshape(count, 3, 3).astype(np.float64), values[:, :3].astype(np.float64)


def audit_stl_independent(path: Path) -> dict[str, Any]:
    triangles, declared_normals = _read_binary_stl_independent(path)
    flat = triangles.reshape(-1, 3)
    scale = 1e5
    vertex_keys, inverse = np.unique(np.round(flat * scale).astype(np.int64), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edge_a = faces[:, [0, 1, 2]].reshape(-1)
    edge_b = faces[:, [1, 2, 0]].reshape(-1)
    edges = np.stack([edge_a, edge_b], axis=1)
    undirected, undirected_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    directed, directed_counts = np.unique(edges, axis=0, return_counts=True)
    directed_map = {tuple(edge.tolist()): int(n) for edge, n in zip(directed, directed_counts, strict=True)}
    opposite_ok = all(directed_map.get((int(b), int(a)), 0) == n
                      for (a, b), n in directed_map.items())
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    calculated = np.divide(cross, lengths[:, None], out=np.zeros_like(cross), where=lengths[:, None] > 0)
    normal_agreement = np.einsum("ij,ij->i", calculated, declared_normals)
    bbox = [[float(flat[:, axis].min()), float(flat[:, axis].max())] for axis in range(3)]
    sidecar_candidates = [path.parent / "stl_metadata.json", path.parent / "fabrication_design.json",
                          path.parent / "height_design.json", path.parent / "phase_metadata.json",
                          path.parent / "export_summary.json"]
    sidecar = next((candidate for candidate in sidecar_candidates if candidate.exists()), None)
    unit_status = "NOT_ENCODED_IN_STL"
    unit_source = None
    if sidecar is not None:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        serialized = json.dumps(metadata, ensure_ascii=False).lower()
        if ("micrometre" in serialized or "micrometer" in serialized or '"unit": "um"' in serialized
                or '"units": "um"' in serialized):
            unit_status = "SIDECAR_DECLARES_UM"
            try:
                unit_source = str(sidecar.resolve().relative_to(ROOT))
            except ValueError:
                unit_source = str(sidecar.resolve())
    try:
        path_label = str(path.resolve().relative_to(ROOT))
    except ValueError:
        path_label = str(path.resolve())
    return {
        "path": path_label,
        "sha256": sha256_file(path),
        "bytes": len(path.read_bytes()),
        "triangles": int(len(triangles)),
        "bbox_numeric_without_intrinsic_unit": bbox,
        "declared_unit_status": unit_status,
        "unit_evidence_path": unit_source,
        "closed_two_manifold_by_edge_incidence": bool(np.all(undirected_counts == 2)),
        "nonmanifold_or_boundary_edges": int(np.sum(undirected_counts != 2)),
        "opposite_directed_edge_pairing": bool(opposite_ok),
        "degenerate_triangles": int(np.sum(lengths <= 1e-12)),
        "declared_normal_agreement_fraction": float(np.mean(normal_agreement > 0.99)),
        "heightfield_graph_non_self_intersection_by_construction": True,
        "self_intersection_independent_triangle_triangle_test": "NOT_RUN",
    }


def audit_c33_artifacts(backup: Path = BACKUP_PATH) -> dict[str, Any]:
    run_dirs = sorted(p for p in backup.glob("train_*") if p.is_dir())
    rows = []
    for run in run_dirs:
        checkpoint_path = run / "phase.pt"
        summary_path = run / "training_summary.json"
        record: dict[str, Any] = {"run": run.name, "files": sorted(p.name for p in run.iterdir() if p.is_file())}
        record["required_files_present"] = checkpoint_path.exists() and summary_path.exists()
        if record["required_files_present"]:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            phase = checkpoint["phase"]
            fingerprint = c33.tensor_sha256(phase)
            validation_row = next((entry for entry in summary["history"]
                                   if int(entry["epoch"]) == int(summary["best_epoch"])), None)
            if validation_row is None:
                raise ValueError(f"best epoch missing from history for {run.name}")
            record.update({
                "phase_shape": list(phase.shape),
                "phase_finite": bool(torch.isfinite(phase).all()),
                "phase_sha256_tensor": fingerprint,
                "summary_phase_fingerprint_matches": fingerprint == summary["phase"]["fingerprint"],
                "best_epoch_checkpoint_matches_summary": int(checkpoint["best_epoch"]) == int(summary["best_epoch"]),
                "checkpoint_seed_matches_summary": int(checkpoint["seed"]) == int(summary["seed"]),
                "checkpoint_geometry_matches_summary": float(checkpoint["defocus_um"]) == float(summary["defocus_um"]),
                "validation_accuracy": float(summary["validation"]["accuracy"]),
                "validation_loss_at_best_epoch": float(validation_row["val_loss"]),
                "holdout_accuracy_historical": float(summary["holdout"]["accuracy"]),
                "seed": int(summary["seed"]),
                "best_epoch": int(summary["best_epoch"]),
                "train_rows": int(summary["rows"]["train"]),
                "validation_rows": int(summary["rows"]["validation"]),
                "holdout_rows": int(summary["rows"]["holdout"]),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "summary_sha256": sha256_file(summary_path),
                "optimizer_state_saved": "optimizer" in checkpoint,
                "per_sample_predictions_saved": "predictions" in checkpoint,
            })
        rows.append(record)
    valid = [r for r in rows if r.get("required_files_present") and
             r.get("summary_phase_fingerprint_matches") and r.get("phase_finite") and
             r.get("best_epoch_checkpoint_matches_summary") and r.get("checkpoint_seed_matches_summary")]
    ranked = sorted(valid, key=lambda r: (-r["validation_accuracy"],
                                          r["validation_loss_at_best_epoch"], r["run"]))
    selected = ranked[0] if ranked else None
    ties = [] if selected is None else [r["run"] for r in valid
                                       if r["validation_accuracy"] == selected["validation_accuracy"]]
    stls = sorted(backup.rglob("*.stl"))
    all_files = sorted(p for p in backup.rglob("*") if p.is_file())
    return {
        "identity": "C3.4_C33_ARTIFACT_AUDIT",
        "session_source_id": SESSION_SOURCE_ID,
        "backup_root": str(backup.relative_to(ROOT)),
        "artifact_inventory": {"files": len(all_files),
                               "bytes": int(sum(p.stat().st_size for p in all_files)),
                               "all_file_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in all_files}},
        "training_runs": {"expected": 19, "found": len(rows), "records": rows,
                          "complete_for_inference": len(valid),
                          "optimizer_replay_possible": False,
                          "reason_optimizer_replay": "phase.pt has the selected phase and split rows but no optimizer state or source-side per-step checkpoints",
                          "validation_only_selection": {"unique_best_run": selected["run"] if selected else None,
                                                        "validation_accuracy": selected["validation_accuracy"] if selected else None,
                                                        "validation_loss_at_best_epoch": selected["validation_loss_at_best_epoch"] if selected else None,
                                                        "validation_accuracy_ties": ties,
                                                        "selection_rule": "highest saved validation accuracy; then lower validation loss; then run name; holdout not used"}},
        "stl_files": {"expected_by_transfer_receipt": 24, "found": len(stls),
                      "records": [audit_stl_independent(path) for path in stls]},
        "unit_limit": "STL carries no intrinsic length unit; use only explicit sidecar metadata, never infer micrometres from numeric bounds",
    }


def model_a_forward(images: torch.Tensor, phase: torch.Tensor, configuration) -> dict[str, torch.Tensor]:
    if phase.requires_grad:
        raise ValueError("frozen C3.3 inference phase must not require gradients")
    with torch.no_grad():
        return c33.forward_field(images, configuration, phase)

