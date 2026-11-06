"""Assumption-labelled scalar model for frozen HW207 MNIST4 phase screens.

This module is intentionally diagnostic. It never touches hardware, changes a
phase map, or trains a parameter. Geometry and detector registration are read
from the dedicated v1 config and remain assumptions until calibrated.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34
from scripts.research.lightgen import hw207_physical_cmos_v1 as cmos


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/hw207_physical_frozen_phase_v1.json"
SOURCE_CONFIG = ROOT / "configs/research/lightgen/hw207_physical_cmos_v1.json"
WAVELENGTH_UM = 0.532
DMD_PITCH_UM = 7.56
PHASE_PITCH_UM = 2.0
CMOS_SHAPE = (2048, 2048)
CMOS_PITCH_UM = 6.5
CLASSES = (0, 1, 2, 3)


def read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path | str, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"project input escapes repository root: {value}") from exc
    return path


def _idx_images(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) != 16:
        raise ValueError(f"truncated IDX image header: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", header)
    if magic != 2051 or (rows, columns) != (28, 28):
        raise ValueError(f"unexpected IDX image header {magic}/{count}/{rows}/{columns}: {path}")
    expected_size = 16 + count * rows * columns
    if path.stat().st_size != expected_size:
        raise ValueError(f"IDX image byte count mismatch: {path}")
    return np.memmap(path, mode="r", dtype=np.uint8, offset=16, shape=(count, rows, columns))


def _idx_labels(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"truncated IDX label header: {path}")
    magic, count = struct.unpack(">II", header)
    if magic != 2049:
        raise ValueError(f"unexpected IDX label magic {magic}: {path}")
    if path.stat().st_size != 8 + count:
        raise ValueError(f"IDX label byte count mismatch: {path}")
    return np.memmap(path, mode="r", dtype=np.uint8, offset=8, shape=(count,))


def load_c0_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Load only C0 validation rows, retaining their exact original order."""
    data_cfg = config["data"]
    index_path = repo_path(config["source_configs"]["c0_index_manifest"])
    index_doc = read_json(index_path)
    validation = index_doc["validation"]
    raw_indices = np.asarray(validation["original_indices"], dtype=np.int64)
    if raw_indices.ndim != 1 or raw_indices.size == 0 or np.unique(raw_indices).size != raw_indices.size:
        raise ValueError("C0 validation indices must be a nonempty unique sequence")
    source_name = str(validation.get("source", ""))
    if "training.pt" not in source_name:
        raise ValueError(f"C0 validation provenance is not MNIST training population: {source_name}")
    images_path = repo_path(data_cfg["images_train_idx"])
    labels_path = repo_path(data_cfg["labels_train_idx"])
    images = _idx_images(images_path)
    labels = _idx_labels(labels_path)
    if images.shape[0] != labels.shape[0] or np.any(raw_indices < 0) or np.any(raw_indices >= images.shape[0]):
        raise ValueError("C0 validation rows do not fit the raw MNIST training IDX population")
    selected_labels = np.asarray(labels[raw_indices], dtype=np.uint8)
    allowed = np.isin(selected_labels, np.asarray(data_cfg["allowed_classes"], dtype=np.uint8))
    positions = np.flatnonzero(allowed)
    indices = raw_indices[positions]
    selected_labels = selected_labels[positions]
    selected_images = np.asarray(images[indices], dtype=np.uint8)
    if not np.isin(selected_labels, CLASSES).all():
        raise ValueError("validation filter returned a label outside 0..3")
    return {
        "images": selected_images,
        "labels": selected_labels.astype(np.int64),
        "indices": indices,
        "source": source_name,
        "index_manifest_path": str(index_path.relative_to(ROOT)),
        "index_manifest_sha256": sha256_file(index_path),
        "images_path": str(images_path.relative_to(ROOT)),
        "images_sha256": sha256_file(images_path),
        "labels_path": str(labels_path.relative_to(ROOT)),
        "labels_sha256": sha256_file(labels_path),
        "class_counts": {str(k): int(np.sum(selected_labels == k)) for k in CLASSES},
        "official_test_opened": False,
    }


def candidate_records(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    source = read_json(SOURCE_CONFIG)
    raw = source["phase_candidates_to_audit_when_model_is_calibrated"]
    records: list[dict[str, Any]] = []
    for row in raw:
        phase_path = repo_path(row["path"])
        if not phase_path.is_file():
            raise FileNotFoundError(phase_path)
        phase = np.load(phase_path, mmap_mode="r", allow_pickle=False)
        if phase.shape != (100, 100) or phase.dtype.kind not in "fiu" or not np.isfinite(phase).all():
            raise ValueError(f"invalid phase asset: {phase_path}")
        rec: dict[str, Any] = {
            "candidate_id": row["name"],
            "source_experiment": row.get("source_report"),
            "source_type": row["source_type"],
            "phase_path": row["path"],
            "phase_sha256": sha256_file(phase_path),
            "representation": "phase_rad",
            "shape_rows_columns": list(phase.shape),
            "dtype": str(phase.dtype),
            "pixel_pitch_um": PHASE_PITCH_UM,
            "wavelength_nm": 532.0,
            "reference_plane": "historical numerical LightGen phase-plane input; physical device binding UNKNOWN",
            "phase_min_raw_rad": float(np.min(phase)),
            "phase_max_raw_rad": float(np.max(phase)),
            "phase_min_wrapped_rad": float(np.min(np.remainder(phase, 2 * math.pi))),
            "phase_max_wrapped_rad": float(np.max(np.remainder(phase, 2 * math.pi))),
            "coordinate_convention": "saved row-major array; source-to-hardware affine/rotation UNKNOWN; no rotation or relabeling applied",
            "physical_complex_transmission": "UNKNOWN; numerical forward uses unit-amplitude exp(i*phase)",
        }
        if row.get("source_report"):
            report_path = repo_path(row["source_report"])
            rec["source_report_status"] = "PRESENT" if report_path.is_file() else "MISSING"
            if report_path.is_file():
                rec["source_report_sha256_provenance_only"] = sha256_file(report_path)
        if row.get("height_path"):
            height_path = repo_path(row["height_path"])
            height = np.load(height_path, mmap_mode="r", allow_pickle=False)
            if height.shape != (100, 100) or height.dtype.kind not in "fiu" or not np.isfinite(height).all():
                raise ValueError(f"invalid height asset: {height_path}")
            rec["original_height"] = {
                "path": row["height_path"],
                "sha256": sha256_file(height_path),
                "representation": "height_um",
                "shape_rows_columns": list(height.shape),
                "dtype": str(height.dtype),
                "minimum_um": float(np.min(height)),
                "maximum_um": float(np.max(height)),
                "phase_height_relation": "source processing assumptions only; material delta-n and measured complex transmission UNKNOWN",
            }
        records.append(rec)
    if len(records) != 7 or len({row["candidate_id"] for row in records}) != 7:
        raise ValueError("the frozen inventory must contain exactly seven unique historical candidates")
    return records


def _aperture_subpixel_weights(subpixels: int, fill_factor: float, *, device: torch.device) -> torch.Tensor:
    if subpixels <= 0 or not 0 < fill_factor <= 1:
        raise ValueError("invalid DMD subpixel geometry")
    edges = torch.linspace(-0.5, 0.5, subpixels + 1, dtype=torch.float64, device=device)
    active_half = math.sqrt(fill_factor) / 2.0
    overlap = (torch.minimum(edges[1:], torch.tensor(active_half, device=device))
               - torch.maximum(edges[:-1], torch.tensor(-active_half, device=device))).clamp_min(0)
    # Each subcell stores one amplitude whose squared magnitude represents
    # the exact covered area fraction; this preserves mirror fill-factor power
    # while resolving the aperture edges on the chosen diagnostic grid.
    return (overlap / (edges[1:] - edges[:-1])).clamp_min(0).sqrt().to(torch.float32)


def dmd_field(images: torch.Tensor, *, mode: str, model_level: int,
              subpixels: int = 8, fill_factor: float = 0.92) -> tuple[torch.Tensor, float]:
    """Create a centered native-pitch DMD patch; no labels enter this function."""
    if images.ndim != 3 or images.shape[-2:] != (28, 28) or images.dtype != torch.uint8:
        raise ValueError("DMD input requires uint8 [batch,28,28] MNIST rows")
    if model_level not in (0, 1):
        raise ValueError("DMD model_level must be 0 or 1")
    if mode == "static_binary":
        amplitude = (images >= 128).to(torch.float32)
    elif mode == "gray_effective_amplitude":
        amplitude = (images.to(torch.float32) / 255.0).clamp(0, 1).sqrt()
    else:
        raise ValueError("unsupported DMD input encoding")
    if model_level == 0:
        subpixels = 1
        cell_weights = torch.ones(1, dtype=torch.float32, device=images.device)
    else:
        cell_weights = _aperture_subpixel_weights(subpixels, fill_factor, device=images.device)
    expanded = amplitude.repeat_interleave(subpixels, dim=-2).repeat_interleave(subpixels, dim=-1)
    axis_weights = cell_weights.repeat(28)
    expanded = expanded * axis_weights[None, :, None] * axis_weights[None, None, :]
    pitch_um = DMD_PITCH_UM / subpixels
    return torch.complex(expanded, torch.zeros_like(expanded)), pitch_um


def native_dmd_pattern(image: np.ndarray, *, mode: str) -> np.ndarray:
    """Materialize one 1600x2560 DMD command with the MNIST patch centered."""
    values = np.asarray(image)
    if values.shape != (28, 28) or values.dtype != np.uint8:
        raise ValueError("native DMD command requires one uint8 28x28 MNIST image")
    result = np.zeros((1600, 2560), dtype=np.uint8)
    row0, col0 = (1600 - 28) // 2, (2560 - 28) // 2
    if mode == "static_binary":
        patch = np.where(values >= 128, 255, 0).astype(np.uint8)
    elif mode == "gray_effective_amplitude":
        patch = values.copy()
    else:
        raise ValueError("unsupported DMD input encoding")
    result[row0:row0 + 28, col0:col0 + 28] = patch
    return result


def _plane_coordinates(count: int, pitch_um: float, *, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
    return (torch.arange(count, device=device, dtype=dtype) - (count - 1) / 2.0) * pitch_um


def _lens_multiply(field: torch.Tensor, pitch_um: float, focal_um: float,
                   aperture_um: float, *, plan_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count_y, count_x = field.shape[-2:]
    real_dtype = torch.float64 if plan_dtype == torch.complex128 else torch.float32
    y = _plane_coordinates(count_y, pitch_um, device=field.device, dtype=real_dtype)
    x = _plane_coordinates(count_x, pitch_um, device=field.device, dtype=real_dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    aperture = ((xx.square() + yy.square()) <= (aperture_um / 2.0) ** 2)
    lens_phase = torch.exp((-1j * math.pi / (WAVELENGTH_UM * focal_um))
                           * (xx.square() + yy.square())).to(field.dtype)
    before = field
    after = before * aperture.to(field.real.dtype) * lens_phase
    return after, before, aperture


def _energy(field: torch.Tensor, pitch_um: float) -> torch.Tensor:
    return field.abs().square().sum(dim=(-2, -1)) * pitch_um**2


def area_average_cmos_pixels(subpixel_intensity: torch.Tensor, sensor_shape: tuple[int, int],
                              oversample: int) -> torch.Tensor:
    """Integrate a regular subpixel irradiance grid over each physical CMOS pixel."""
    if subpixel_intensity.ndim != 3 or not subpixel_intensity.is_floating_point():
        raise ValueError("subpixel_intensity must be floating [batch, rows, columns]")
    rows, columns = (int(value) for value in sensor_shape)
    factor = int(oversample)
    if rows <= 0 or columns <= 0 or factor <= 0:
        raise ValueError("sensor dimensions and oversample must be positive")
    if subpixel_intensity.shape[-2:] != (rows * factor, columns * factor):
        raise ValueError("subpixel map shape must equal sensor shape times oversample")
    return subpixel_intensity.reshape(
        subpixel_intensity.shape[0], rows, factor, columns, factor,
    ).mean(dim=(2, 4))


def propagate_dmd_to_phase(images: torch.Tensor, config: dict[str, Any], *, device: torch.device,
                           model_level: int = 1, mode: str = "static_binary",
                           plan_cache: dict | None = None,
                           override_lens_grid: int | None = None) -> dict[str, torch.Tensor]:
    """DMD → L2 → L1 → phase-plane propagation with finite lens pupils."""
    optics = config["input_relay"]
    grid = int(override_lens_grid or config["numerics"]["lens_plane_grid"][0])
    lens_pitch = float(config["numerics"]["input_lens_plane_pitch_um"])
    if override_lens_grid is not None:
        aperture_um = float(optics["lens_clear_aperture_mm"]) * 1000.0
        lens_pitch = aperture_um / grid
    if plan_cache is None:
        plan_cache = {}
    dmd, dmd_pitch = dmd_field(
        images.to(device), mode=mode, model_level=model_level,
        subpixels=int(config["dmd"]["subpixels_per_micromirror_axis"]),
        fill_factor=float(config["dmd"]["fill_factor"]),
    )
    lens_f = float(optics["lens_focal_lengths_mm"][0]) * 1000.0
    aperture = float(optics["lens_clear_aperture_mm"]) * 1000.0
    l2 = c34.scaled_fresnel(
        dmd, dmd_pitch, (grid, grid), lens_pitch,
        float(optics["distances_mm"]["DMD_to_L2"]) * 1000.0,
        wavelength_um=WAVELENGTH_UM, plan_cache=plan_cache,
    )
    l2_lens, l2_before, _ = _lens_multiply(l2, lens_pitch, lens_f, aperture)
    l1 = c34.scaled_fresnel(
        l2_lens, lens_pitch, (grid, grid), lens_pitch,
        float(optics["distances_mm"]["L2_to_L1"]) * 1000.0,
        wavelength_um=WAVELENGTH_UM, plan_cache=plan_cache,
    )
    l1_lens, l1_before, _ = _lens_multiply(l1, lens_pitch, lens_f, aperture)
    phase_field = c34.scaled_fresnel(
        l1_lens, lens_pitch, (100, 100), PHASE_PITCH_UM,
        float(optics["distances_mm"]["L1_to_phase_layer"]) * 1000.0,
        wavelength_um=WAVELENGTH_UM, plan_cache=plan_cache,
    )
    return {
        "dmd_field": dmd,
        "dmd_input_energy": _energy(dmd, dmd_pitch),
        "l2_field": l2,
        "l2_field_after_lens": l2_lens,
        "l2_before_aperture_energy": _energy(l2_before, lens_pitch),
        "l2_after_aperture_energy": _energy(l2_lens, lens_pitch),
        "l1_field": l1,
        "l1_field_after_lens": l1_lens,
        "l1_before_aperture_energy": _energy(l1_before, lens_pitch),
        "l1_after_aperture_energy": _energy(l1_lens, lens_pitch),
        "phase_field": phase_field,
        "phase_plane_energy": _energy(phase_field, PHASE_PITCH_UM),
    }


def output_cmos_field(phase_field: torch.Tensor, phase_rad: torch.Tensor | np.ndarray,
                      config: dict[str, Any], scenario: dict[str, Any], *,
                      plan_cache: dict | None = None, output_lens_grid: int | None = None,
                      output_aperture_mm: float | None = None,
                      cmos_oversample: int | None = None) -> dict[str, torch.Tensor]:
    if phase_field.ndim != 3 or phase_field.shape[-2:] != (100, 100) or not torch.is_complex(phase_field):
        raise ValueError("phase_field must be complex [batch,100,100]")
    phase = torch.as_tensor(phase_rad, device=phase_field.device, dtype=phase_field.real.dtype)
    if phase.shape != (100, 100) or not torch.isfinite(phase).all():
        raise ValueError("candidate phase must be finite 100x100")
    if plan_cache is None:
        plan_cache = {}
    screen = phase_field * torch.polar(torch.ones_like(phase), phase)
    phase_energy = _energy(phase_field, PHASE_PITCH_UM)
    lens_grid = int(output_lens_grid or config["numerics"]["output_lens_plane_grid"][0])
    aperture_mm = float(output_aperture_mm or config["output_relay"]["clear_aperture_mm"])
    aperture_um = aperture_mm * 1000.0
    lens_extent_um = float(config["output_relay"]["clear_aperture_mm"]) * 1000.0
    lens_pitch = lens_extent_um / lens_grid
    phase_to_lens_um = float(scenario["phase_to_lens_mm"]) * 1000.0
    lens_to_cmos_um = float(scenario["lens_to_cmos_mm"]) * 1000.0
    focal_um = float(config["output_relay"]["focal_length_mm"]) * 1000.0
    cmos_rows, cmos_columns = (
        int(value) for value in config["cmos"]["sensor_shape_rows_columns"]
    )
    cmos_pitch_um = float(config["cmos"]["pixel_pitch_um"])
    oversample = int(cmos_oversample or config["numerics"]["cmos_oversample_default"])
    if oversample < 1:
        raise ValueError("CMOS oversample factor must be a positive integer")
    at_lens = c34.scaled_fresnel(
        screen, PHASE_PITCH_UM, (lens_grid, lens_grid), lens_pitch, phase_to_lens_um,
        wavelength_um=WAVELENGTH_UM, plan_cache=plan_cache,
    )
    lens, before, pupil = _lens_multiply(at_lens, lens_pitch, focal_um, aperture_um)
    cmos_field = c34.scaled_fresnel(
        lens, lens_pitch, (cmos_rows * oversample, cmos_columns * oversample),
        cmos_pitch_um / oversample, lens_to_cmos_um,
        wavelength_um=WAVELENGTH_UM, plan_cache=plan_cache,
    )
    subpixel_intensity = cmos_field.abs().square()
    pixel_intensity = area_average_cmos_pixels(subpixel_intensity, (cmos_rows, cmos_columns), oversample)
    return {
        "phase_field_after_screen": screen,
        "phase_energy": phase_energy,
        "output_lens_field_before_pupil": before,
        "output_lens_field_after_pupil": lens,
        "output_lens_energy_before_pupil": _energy(before, lens_pitch),
        "output_lens_energy_after_pupil": _energy(lens, lens_pitch),
        "cmos_field": cmos_field,
        "cmos_subpixel_intensity": subpixel_intensity,
        "cmos_intensity": pixel_intensity,
        "cmos_energy": _energy(pixel_intensity, cmos_pitch_um),
        "cmos_oversample_factor": torch.as_tensor(oversample, device=phase_field.device),
        "output_aperture_mask": pupil,
    }


def forward_field(images: torch.Tensor, phase_rad: torch.Tensor | np.ndarray,
                  config: dict[str, Any], scenario: dict[str, Any], *,
                  device: torch.device | str, model_level: int = 1,
                  mode: str = "static_binary", plan_cache: dict | None = None,
                  input_relay_cache: dict[str, torch.Tensor] | None = None,
                  input_lens_grid: int | None = None,
                  output_lens_grid: int | None = None,
                  output_aperture_mm: float | None = None,
                  cmos_oversample: int | None = None) -> dict[str, torch.Tensor]:
    """Single production call path used by preflight, inference and replay."""
    torch_device = torch.device(device)
    if plan_cache is None:
        plan_cache = {}
    if input_relay_cache is None:
        inp = propagate_dmd_to_phase(
            images, config, device=torch_device, model_level=model_level, mode=mode,
            plan_cache=plan_cache, override_lens_grid=input_lens_grid,
        )
    else:
        inp = input_relay_cache
        if not torch.is_complex(inp["phase_field"]) or inp["phase_field"].shape != (images.shape[0], 100, 100):
            raise ValueError("cached input relay field shape does not match the input rows")
    output = output_cmos_field(
        inp["phase_field"], phase_rad, config, scenario, plan_cache=plan_cache,
        output_lens_grid=output_lens_grid, output_aperture_mm=output_aperture_mm,
        cmos_oversample=cmos_oversample,
    )
    return {**inp, **output}


def cmos_pixel_to_window_affine(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cmos_cfg = config["cmos"]
    affine = cmos_cfg.get("pixel_to_window_affine")
    if affine is not None:
        matrix = np.asarray(affine["matrix_um_per_pixel"], dtype=np.float64)
        offset = np.asarray(affine["offset_um"], dtype=np.float64)
    else:
        rows, columns = (int(value) for value in cmos_cfg["sensor_shape_rows_columns"])
        pitch = float(cmos_cfg["pixel_pitch_um"])
        width, height = (float(v) for v in config["fixed_roi_geometry"]["physical_window_um"])
        matrix = np.asarray(((pitch, 0.0), (0.0, pitch)), dtype=np.float64)
        offset = np.asarray((width / 2.0 - columns * pitch / 2.0,
                             height / 2.0 - rows * pitch / 2.0), dtype=np.float64)
    if matrix.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("CMOS pixel-to-window affine must be a finite 2x2 matrix and 2-vector")
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise ValueError("CMOS pixel-to-window affine must be invertible")
    return matrix, offset


def apply_cmos_calibration(
    raw_counts: np.ndarray,
    *,
    dark_frame: np.ndarray | None = None,
    dark_exposure_s: float | None = None,
    exposure_s: float | None = None,
    flat_field: np.ndarray | None = None,
    gain_e_per_adu: float | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only supplied detector calibrations; absent calibration is explicit identity."""
    raw = np.asarray(raw_counts)
    if raw.ndim != 2 or raw.dtype.kind not in "buif" or not np.isfinite(raw).all():
        raise ValueError("raw CMOS frame must be a finite numeric two-dimensional array")
    corrected = raw.astype(np.float64, copy=True)
    applied: list[str] = []
    if dark_frame is not None:
        dark = np.asarray(dark_frame, dtype=np.float64)
        if dark.shape != raw.shape or not np.isfinite(dark).all():
            raise ValueError("dark frame must be finite and match the CMOS frame shape")
        if exposure_s is not None or dark_exposure_s is not None:
            if exposure_s is None or dark_exposure_s is None or not math.isfinite(exposure_s) or not math.isfinite(dark_exposure_s) or exposure_s <= 0 or dark_exposure_s <= 0:
                raise ValueError("sample and dark exposure times must both be positive finite seconds")
            dark = dark * (exposure_s / dark_exposure_s)
        corrected -= dark
        applied.append("dark_subtraction")
    corrected = np.maximum(corrected, 0.0)
    if flat_field is not None:
        flat = np.asarray(flat_field, dtype=np.float64)
        if flat.shape != raw.shape or not np.isfinite(flat).all() or np.any(flat < 0):
            raise ValueError("flat field must be finite, nonnegative and match the CMOS frame shape")
        positive = flat[flat > 0]
        if positive.size == 0:
            raise ValueError("flat field must contain positive response values")
        normalized_flat = flat / float(np.mean(positive))
        if np.any(normalized_flat <= 0):
            raise ValueError("flat field contains a zero-response pixel")
        corrected /= normalized_flat
        applied.append("flat_field_division")
    if gain_e_per_adu is not None:
        gain = np.asarray(gain_e_per_adu, dtype=np.float64)
        if not np.isfinite(gain).all() or np.any(gain <= 0) or (gain.ndim > 0 and gain.shape != raw.shape):
            raise ValueError("gain must be a positive finite scalar or a full-frame map")
        corrected *= gain
        applied.append("gain_conversion")
    if exposure_s is not None:
        if not math.isfinite(exposure_s) or exposure_s <= 0:
            raise ValueError("exposure_s must be positive and finite")
        corrected /= exposure_s
        applied.append("exposure_rate_normalization")
    return corrected, {
        "status": "CALIBRATED_WITH_EXPLICIT_INPUTS" if applied else "UNCALIBRATED_IDENTITY",
        "applied_corrections": applied,
        "absolute_photon_calibration": False,
    }


def fixed_sensor_roi_weights(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    cmos_cfg = config["cmos"]
    rows, columns = (int(value) for value in cmos_cfg["sensor_shape_rows_columns"])
    window = config["fixed_roi_geometry"]
    width, height = (float(v) for v in window["physical_window_um"])
    matrix, offset = cmos_pixel_to_window_affine(config)
    rois = []
    for key in ("0", "1", "2", "3"):
        item = window["class_to_roi"][key]
        x0, x1 = (float(v) for v in item["x_um"])
        y0, y1 = (float(v) for v in item["y_um"])
        rois.append((int(key), x0, x1, y0, y1))
    return cmos.build_roi_pixel_weights(
        (rows, columns), matrix, offset, roi_rectangles_um=rois, window_um=(width, height)
    )


def _point_in_roi(x_um: float, y_um: float, label: int, config: dict[str, Any]) -> bool:
    roi = config["fixed_roi_geometry"]["class_to_roi"][str(int(label))]
    x0, x1 = roi["x_um"]
    y0, y1 = roi["y_um"]
    return float(x0) <= x_um < float(x1) and float(y0) <= y_um < float(y1)


def measure_cmos(intensity: np.ndarray, label: int, dmd_input_energy: float,
                 phase_plane_energy: float,
                 config: dict[str, Any], roi_weights: Sequence[dict[str, Any]]) -> dict[str, Any]:
    image = np.asarray(intensity)
    if image.shape != tuple(config["cmos"]["sensor_shape_rows_columns"]):
        raise ValueError("CMOS intensity shape differs from configured physical sensor")
    if image.dtype.kind not in "fiu" or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("CMOS intensity must be finite, numeric and nonnegative")
    matrix, offset = cmos_pixel_to_window_affine(config)
    pixel_area_um2 = abs(float(np.linalg.det(matrix)))
    pixel_energy = image.astype(np.float64, copy=False) * pixel_area_um2
    total = float(pixel_energy.sum(dtype=np.float64))
    scores = cmos.integrate_roi_energies(pixel_energy, roi_weights)
    pred = cmos.predict_from_roi_energies(scores)
    strongest_wrong = max(float(scores[k]) for k in range(4) if k != label)
    flat = int(np.argmax(image))
    peak_row, peak_col = np.unravel_index(flat, image.shape)
    row_profile = pixel_energy.sum(axis=1, dtype=np.float64)
    col_profile = pixel_energy.sum(axis=0, dtype=np.float64)
    row_centroid = float(np.dot(row_profile, np.arange(image.shape[0], dtype=np.float64) + 0.5) / total) if total else None
    col_centroid = float(np.dot(col_profile, np.arange(image.shape[1], dtype=np.float64) + 0.5) / total) if total else None
    centroid_xy = matrix @ np.asarray((col_centroid, row_centroid), dtype=np.float64) + offset if total else None
    peak_xy = matrix @ np.asarray((peak_col + 0.5, peak_row + 0.5), dtype=np.float64) + offset
    cx_um, cy_um = (float(centroid_xy[0]), float(centroid_xy[1])) if total else (None, None)
    peak_x_um, peak_y_um = float(peak_xy[0]), float(peak_xy[1])
    return {
        "true_label": int(label),
        "prediction": -1 if pred is None else int(pred),
        "abstained": pred is None,
        "roi_scores": scores.tolist(),
        "roi_fractions_of_cmos": (scores / total).tolist() if total > 0 else [0.0] * 4,
        "roi_fractions_of_four_roi_energy": (scores / scores.sum()).tolist() if scores.sum() > 0 else [0.0] * 4,
        "cmos_energy_model_units_um2": total,
        "correct_roi_energy_model_units_um2": float(scores[label]),
        "strongest_wrong_roi_energy_model_units_um2": strongest_wrong,
        "correct_to_strongest_wrong_roi_ratio": float(scores[label] / strongest_wrong) if strongest_wrong > 0 else None,
        "outside_four_roi_energy_model_units_um2": max(total - float(scores.sum()), 0.0),
        "outside_four_roi_fraction_of_cmos": max(total - float(scores.sum()), 0.0) / total if total > 0 else 0.0,
        "input_dmd_energy_model_units_um2": float(dmd_input_energy),
        "phase_plane_input_energy_model_units_um2": float(phase_plane_energy),
        "cmos_pixel_area_um2_from_affine": pixel_area_um2,
        "dmd_to_cmos_efficiency": total / float(dmd_input_energy) if dmd_input_energy > 0 else None,
        "phase_to_cmos_efficiency": total / float(phase_plane_energy) if phase_plane_energy > 0 else None,
        "peak_row_col": [int(peak_row), int(peak_col)],
        "peak_x_y_in_window_um": [peak_x_um, peak_y_um],
        "peak_in_correct_roi": _point_in_roi(peak_x_um, peak_y_um, label, config),
        "centroid_row_col": [row_centroid, col_centroid] if total else [None, None],
        "centroid_x_y_in_window_um": [cx_um, cy_um],
        "centroid_in_correct_roi": bool(total and _point_in_roi(cx_um, cy_um, label, config)),
    }


def phase_array(candidate: dict[str, Any]) -> np.ndarray:
    values = np.load(repo_path(candidate["phase_path"]), allow_pickle=False)
    if values.shape != (100, 100) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"invalid frozen phase candidate: {candidate['candidate_id']}")
    return np.asarray(values, dtype=np.float32)


def candidate_metadata(config: dict[str, Any]) -> list[dict[str, Any]]:
    return candidate_records(config)


def focal_geometry_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Recompute paraxial ABCD matrices from the configured distances."""
    def p(distance: float) -> np.ndarray:
        return np.asarray(((1.0, distance), (0.0, 1.0)), dtype=np.float64)

    def lens(focal: float) -> np.ndarray:
        return np.asarray(((1.0, 0.0), (-1.0 / focal, 1.0)), dtype=np.float64)

    inp = config["input_relay"]
    dmd_to_l2 = float(inp["distances_mm"]["DMD_to_L2"])
    l2_to_l1 = float(inp["distances_mm"]["L2_to_L1"])
    l1_to_phase = float(inp["distances_mm"]["L1_to_phase_layer"])
    f1, f2 = (float(x) for x in inp["lens_focal_lengths_mm"])
    input_matrix = p(l1_to_phase) @ lens(f2) @ p(l2_to_l1) @ lens(f1) @ p(dmd_to_l2)
    out = {}
    f = float(config["output_relay"]["focal_length_mm"])
    for name, scenario in config["output_relay"]["scenarios"].items():
        matrix = p(float(scenario["lens_to_cmos_mm"])) @ lens(f) @ p(float(scenario["phase_to_lens_mm"]))
        out[name] = {"ABCD": matrix.tolist(), "B_mm": float(matrix[0, 1]),
                     "is_conjugate": bool(abs(matrix[0, 1]) < 1e-9),
                     "signed_magnification_if_conjugate": float(matrix[0, 0]) if abs(matrix[0, 1]) < 1e-9 else None}
    return {
        "input_relay_ABCD": input_matrix.tolist(),
        "input_relay_B_mm": float(input_matrix[0, 1]),
        "input_relay_is_conjugate": bool(abs(input_matrix[0, 1]) < 1e-9),
        "output_scenarios": out,
        "status": "THIN_LENS_ABCD_ON_PHOTO_ESTIMATE_AND_THEORETICAL_SCENARIOS_ONLY",
    }

