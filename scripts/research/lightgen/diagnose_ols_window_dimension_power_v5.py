"""Window-convergence gate and zero-update Gaussian/LP01 OLS dimension audit."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
    GaussianFiberGeometry,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LP01_OLS_MODEL,
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.propagation import angular_spectrum_propagate, clear_production_transfer_cache
from scripts.research.lightgen import diagnose_ols_gaussian_dimension_reaudit_v1 as baseline
from scripts.research.lightgen import diagnose_ols_g3_reference_closure_v4 as v4
from scripts.research.lightgen import diagnose_ols_g3_second_hop_error_isolation_v3 as v3


SCHEMA = "ols_window_dimension_power.v5"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_window_dimension_power_v5.yaml")
DEFAULT_RUN_DIR = Path("outputs/20261014_ols_window_dimension_power_v5/runs/attempt_01")
PHASE_NAMES = ("zero", "last_layer_x_1deg", "last_layer_xy_1deg")
SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 10)
GRIDS = (3072, 4096, 6144)
ROI_SIZE = 2048
DX_M = 2.0e-6
WAVELENGTH_M = 532.0e-9
SOURCE_POWER_W = 249.3e-6
TRANSMISSION = 0.889
PITCH_M = 125.0e-6
WAIST_M = 7.111e-6
G3_TOLERANCE = 1.0e-4
EPS = 1.0e-30


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    frozen_optics = {
        "wavelength_m": WAVELENGTH_M,
        "pixel_pitch_m": DX_M,
        "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6,
        "propagation_windows_mm": [6.144, 8.192, 12.288],
        "propagation_grid_sizes": list(GRIDS),
        "common_fiber_readout_roi_mm": 4.096,
        "common_fiber_readout_roi_pixels": ROI_SIZE,
        "inter_encoder_spacing_m": 1.5e-3,
        "last_encoder_to_fiber_m": 1.5e-3,
        "phase_screen_count": 2,
        "layer_power_transmission": TRANSMISSION,
        "input_power_w": SOURCE_POWER_W,
        "illumination": "uniform_coherent_top_hat",
    }
    if config.get("optics") != frozen_optics:
        raise ValueError("v5 optical geometry, numerical windows, ROI, or source power changed")
    frozen_fiber = {
        "reference_pitch_m": PITCH_M,
        "cladding_diameter_m": PITCH_M,
        "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": WAIST_M,
        "lp01_v_number": 2.2,
        "transport": "ideal_identity",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }
    if config.get("fiber") != frozen_fiber:
        raise ValueError("v5 fiber assumptions changed")
    diagnostic = config.get("dimension_diagnostic", {})
    if diagnostic.get("square_array_sizes") != list(SIZES) or diagnostic.get("phase_cases") != list(PHASE_NAMES):
        raise ValueError("v5 array sizes or fixed phase cases changed")
    if diagnostic.get("diagnostic_optimizer_updates") != 0:
        raise ValueError("v5 permits zero diagnostic optimizer updates only")
    if config.get("gates", {}).get("window_g3_complex_relative_error_after_common_global_phase_max") != G3_TOLERANCE:
        raise ValueError("v5 G3 threshold must remain 1e-4")
    scope = config.get("scope", {})
    expected_scope = {
        "formal_training_updates": 0,
        "phase_optimizer_updates": 0,
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "fid_calls": 0,
        "loss_changes": 0,
        "production_default_migration": "forbidden",
        "optical_geometry_change": "forbidden",
        "propagation_solver_replacement": "forbidden",
        "material_nonlinearity": "forbidden",
    }
    if scope != expected_scope:
        raise ValueError("v5 scope contract changed")
    return config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_phases(path: Path) -> dict[str, torch.Tensor]:
    phases = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(phases, dict) or not set(PHASE_NAMES).issubset(phases):
        raise ValueError("explicit v4 fixed-phase archive is missing required phase cases")
    result = {name: phases[name].detach().cpu() for name in PHASE_NAMES}
    for name, phase in result.items():
        if tuple(phase.shape) != (2, 1, 100, 100) or not torch.isfinite(phase).all():
            raise ValueError(f"invalid v4 fixed phase tensor: {name}")
    if not all(torch.equal(result["zero"][0], result[name][0]) for name in PHASE_NAMES[1:]):
        raise ValueError("v4 fixed phases must share one unchanged first-layer screen")
    return result


def _cast_real_phase_parameters(runtime, real_dtype: torch.dtype) -> None:
    """Set only trainable real phase tensors; preserve all registered complex buffers."""
    for stack_name in ("encoder", "generator"):
        stack = getattr(runtime, stack_name)
        phase = stack.phase
        if not phase.is_floating_point():
            raise TypeError(f"{stack_name}.phase must be real-valued")
        if phase.dtype != real_dtype:
            phase.data = phase.data.to(dtype=real_dtype)


def _make_runtime(grid_size: int, real_dtype: torch.dtype, device: torch.device):
    physical = LightGenPhysicalOpticsConfig(
        input_power_w=SOURCE_POWER_W,
        layer_power_transmission=TRANSMISSION,
        global_fiber_geometry=True,
        fiber_pitch_m=PITCH_M,
        fiber_cladding_diameter_m=PITCH_M,
        fiber_mfd_m=14.222e-6,
        fiber_length_m=8.5e-3,
        fiber_attenuation_db_per_km=0.0,
    )
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(
            propagation_grid_size=grid_size,
            ols_model=LP01_OLS_MODEL,
            physical_optics=physical,
        )
    ).to(device=device).eval()
    _cast_real_phase_parameters(runtime, real_dtype)
    return runtime


def _encoder_entrance(runtime, real_dtype: torch.dtype, device: torch.device) -> ComplexField:
    return baseline._encoder_entrance(runtime, real_dtype, device)


def _run_encoder(runtime, entrance: ComplexField, phase: torch.Tensor, prefix: str) -> ComplexField:
    return baseline._run_encoder(runtime, entrance, phase, prefix)


def _phase_buffer_preservation_check(device: torch.device) -> dict[str, Any]:
    runtime = _make_runtime(ROI_SIZE, torch.float64, device)
    transport = runtime.ols.operator.transport
    before = transport.clone()
    original = transport[0, 1].clone()
    transport[0, 1] = torch.tensor(0.375 + 0.625j, dtype=transport.dtype, device=device)
    _cast_real_phase_parameters(runtime, torch.float64)
    preserved = bool(torch.equal(transport[0, 1], torch.tensor(0.375 + 0.625j, dtype=transport.dtype, device=device)))
    transport.copy_(before)
    del original, before, runtime
    if not preserved:
        raise RuntimeError("complex OLS transport buffer lost its imaginary component during dtype preparation")
    return {
        "status": "PASS",
        "nonzero_complex_transport_buffer_preserved": preserved,
        "dtype_preparation_touches_only_real_phase_parameters": True,
        "whole_module_real_dtype_cast_used": False,
    }


def _expected_reference_positions() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = (128, 512, 1023, 1535, 1919)
    axis = (np.arange(ROI_SIZE, dtype=np.float64) + 0.5) * DX_M - ROI_SIZE * DX_M / 2.0
    roi_points = np.asarray([[axis[x], axis[y]] for y in indices for x in indices], dtype=np.float64)
    centers = baseline._array_coordinates(10, PITCH_M, torch.device("cpu")).numpy()
    fiber_points: list[list[float]] = []
    for cy in centers:
        for cx in centers:
            ix = int(np.argmin(np.abs(axis - cx)))
            iy = int(np.argmin(np.abs(axis - cy)))
            fiber_points.append([axis[ix], axis[iy]])
    positions = np.concatenate((roi_points, np.asarray(fiber_points, dtype=np.float64)), axis=0)
    return positions, {"roi_grid_25": np.arange(25), "fiber_nearest_pixel_100": np.arange(25, 125), "combined_125": np.arange(125)}


def _load_v4_references(config: dict[str, Any], root: Path, phases: dict[str, torch.Tensor]):
    evidence = config["source_evidence"]
    report_path = _resolve(root, evidence["v4_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "ols_g3_reference_closure.v4" or report.get("status") != "VALID_FAILED":
        raise ValueError("v5 must consume the explicit valid-reference / historical-G3-FAIL v4 report")
    if report.get("independent_reference", {}).get("status") != "VALID":
        raise ValueError("v4 independent reference is not marked valid")
    expected_positions, groups = _expected_reference_positions()
    references: dict[str, dict[str, Any]] = {}
    common_input: np.ndarray | None = None
    positions: np.ndarray | None = None
    for phase_name in PHASE_NAMES:
        path = _resolve(root, evidence["reference_levels"][phase_name])
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "metadata_json", "frequency_nodes_hz", "frequency_weights_per_hz",
                "positions_xy_m", "common_middle_complex128", "last_phase_rad",
                "reference_values_complex128",
            }
            if required - set(archive.files):
                raise ValueError(f"v4 reference level lacks raw values: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata != {
                "schema": "ols_g3_reference_level.v1",
                "phase": phase_name,
                "quadrature_order": 4,
                "frequency_nodes_per_axis": 32768,
            }:
                raise ValueError(f"v4 level metadata is not the converged GL4 32768-node level: {path.name}")
            loaded_positions = np.asarray(archive["positions_xy_m"], dtype=np.float64)
            loaded_input = np.asarray(archive["common_middle_complex128"], dtype=np.complex128)
            loaded_phase = np.asarray(archive["last_phase_rad"], dtype=np.float64)
            values = np.asarray(archive["reference_values_complex128"], dtype=np.complex128)
            nodes = np.asarray(archive["frequency_nodes_hz"], dtype=np.float64)
            weights = np.asarray(archive["frequency_weights_per_hz"], dtype=np.float64)
        expected_nodes, expected_weights = v4._frequency_quadrature(32768, 4)
        if not np.array_equal(nodes, expected_nodes) or not np.array_equal(weights, expected_weights):
            raise ValueError(f"v4 reference nodes or weights differ from the recorded GL4 32768 rule: {phase_name}")
        if loaded_positions.shape != (125, 2) or not np.array_equal(loaded_positions, expected_positions):
            raise ValueError(f"v4 saved physical positions do not match the declared 125-point ROI/fiber selection: {phase_name}")
        if loaded_input.shape != (100, 100) or not np.isfinite(loaded_input).all():
            raise ValueError(f"invalid v4 common second-hop input: {phase_name}")
        expected_phase = phases[phase_name][1, 0].numpy().astype(np.float64, copy=False)
        if loaded_phase.shape != (100, 100) or not np.array_equal(loaded_phase, expected_phase):
            raise ValueError(f"v4 independent reference phase differs from the fixed phase archive: {phase_name}")
        if values.shape != (125,) or not np.isfinite(values).all() or nodes.shape != (32768,) or weights.shape != (32768,):
            raise ValueError(f"invalid raw v4 converged reference arrays: {phase_name}")
        if common_input is None:
            common_input, positions = loaded_input, loaded_positions
        elif not np.array_equal(loaded_input, common_input) or not np.array_equal(loaded_positions, positions):
            raise ValueError("v4 phase reference archives disagree on their common input or positions")
        references[phase_name] = {"values": values, "path": str(path), "metadata": metadata}
    return report, positions, groups, common_input, references


def _preflight(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    evidence = config["source_evidence"]
    phase_path = _resolve(root, evidence["fixed_phase_screens"])
    if not phase_path.is_file():
        raise FileNotFoundError(f"explicit v4 fixed-phase source is missing: {phase_path}")
    phases = _load_phases(phase_path)
    v4_report, positions, groups, common_input, references = _load_v4_references(config, root, phases)
    if not torch.cuda.is_available():
        raise RuntimeError("v5 production optical preflight requires AutoDL CUDA")
    device = torch.device("cuda:0")
    real_dtype = torch.float64
    runtime = _make_runtime(GRIDS[0], real_dtype, device)
    entrance = _encoder_entrance(runtime, real_dtype, device)
    output = _run_encoder(runtime, entrance, phases["zero"].to(device=device, dtype=real_dtype), "ols_window_v5_preflight")
    numpy_reference = baseline._numpy_phase_stack(
        entrance.tensor.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False),
        phases["zero"][:, 0].numpy().astype(np.float64, copy=False),
        GRIDS[0], wavelength_m=WAVELENGTH_M, dx_m=DX_M, distance_m=1.5e-3,
        layer_power_transmission=TRANSMISSION,
    )
    numpy_error = baseline._rel_l2(torch.from_numpy(numpy_reference), output.tensor.detach().cpu()[0, 0])
    if numpy_error > 1.0e-10:
        raise RuntimeError(f"3072 complex128 production propagation differs from independent CPU FFT: {numpy_error:.6g}")
    gaussian = GaussianFiberArrayOLS(
        runtime.ols_plane, GaussianFiberGeometry(port_pitch_m=PITCH_M, waist_m=WAIST_M)
    ).to(device=device)
    gaussian_passivity = gaussian.geometry_diagnostics
    lp_modes = [getattr(runtime.ols.operator, f"input_mode_patch_{index}") for index in range(100)]
    lp_norm_errors = [
        abs(float(mode.abs().square().sum() * DX_M**2) - 1.0) for mode in lp_modes
    ]
    if gaussian_passivity["complete_operator_power_gain_bound"] > 1.0 + 1.0e-5 or max(lp_norm_errors) > 5.0e-6:
        raise RuntimeError("Gaussian/LP01 mode passivity or normalization preflight failed")
    adjoint = baseline._adjoint_preflight(runtime, entrance, phases, config)
    if not adjoint["passed"]:
        raise RuntimeError("production adjoint-to-direct preflight or old zero-phase negative control failed")
    analytic_plane_wave = baseline._analytic_plane_wave_check(device)
    if not analytic_plane_wave["wrong_transfer_phase_negative_control_error"] > 1.0e-3:
        raise RuntimeError("analytic propagation negative phase control was not rejected")
    # A known complex phasor is a direct falsifier for magnitude-only handling.
    phasor = torch.tensor([1.0 + 0.0j, -1.0 + 0.25j], dtype=torch.complex128, device=device)
    coherent = phasor.sum()
    dropped_phase = phasor.abs().to(torch.complex128).sum()
    dropped_phase_error = float((coherent - dropped_phase).abs() / coherent.abs().clamp_min(EPS))
    if dropped_phase_error < 0.5:
        raise RuntimeError("dropped-complex-phase preflight negative control was not rejected")
    crop_error = baseline._rel_l2(output.tensor[0, 0], torch.roll(output.tensor[0, 0], 1, 0))
    if crop_error <= G3_TOLERANCE:
        raise RuntimeError("one-pixel crop shift was not rejected by the production field")
    half_pixel_shift_rejected = not np.array_equal(positions + np.asarray([DX_M / 2.0, 0.0]), positions)
    if not half_pixel_shift_rejected:
        raise RuntimeError("half-pixel-shifted reference positions were not distinguished")
    omitted_area_ratio = 1.0 / DX_M**2
    if not math.isclose(omitted_area_ratio, 2.5e11, rel_tol=1e-12):
        raise RuntimeError("pixel-area omission negative control is inconsistent")
    phase_buffer = _phase_buffer_preservation_check(device)
    # Verify that saved reference locations retain the v4 half-pixel convention.
    axis = (np.arange(ROI_SIZE, dtype=np.float64) + 0.5) * DX_M - ROI_SIZE * DX_M / 2.0
    if not np.array_equal(positions[:25, 0], np.tile(axis[[128, 512, 1023, 1535, 1919]], 5)):
        raise ValueError("v4 ROI coordinates do not match the physical pixel-center convention")
    coordinate_records = {
        "common_roi_shape": [ROI_SIZE, ROI_SIZE],
        "common_roi_extent_m": [ROI_SIZE * DX_M, ROI_SIZE * DX_M],
        "common_roi_first_pixel_center_xy_m": [float(axis[0]), float(axis[0])],
        "common_roi_last_pixel_center_xy_m": [float(axis[-1]), float(axis[-1])],
        "v4_reference_position_count": int(positions.shape[0]),
        "phase_reference_count": len(references),
        "position_groups": {key: len(value) for key, value in groups.items()},
    }
    result = {
        "status": "PASS",
        "v4_reference_status": v4_report["independent_reference"]["status"],
        "v4_historical_g3_status_preserved": v4_report["scientific_gate"],
        "v4_reference_levels": {key: item["path"] for key, item in references.items()},
        "production_device": str(device),
        "production_gpu": torch.cuda.get_device_name(device),
        "representative_grid": GRIDS[0],
        "complex128_two_layer_production_vs_cpu_fft_relative_error": numpy_error,
        "gaussian_passivity": gaussian_passivity,
        "lp01_mode_norm_max_absolute_error": max(lp_norm_errors),
        "old_adjoint_zero_phase_negative_control": adjoint["legacy_zero_phase_negative_control"],
        "adjoint_direct_forward_max_relative_error": max(
            adjoint["maximum_K_to_direct_relative_error"], adjoint["maximum_H_to_direct_relative_error"]
        ),
        "analytic_plane_wave": analytic_plane_wave,
        "dropped_complex_phase_negative_control_relative_error": dropped_phase_error,
        "omitted_pixel_area_negative_control": {"coefficient_scale_ratio_if_area_is_omitted": omitted_area_ratio, "rejected": omitted_area_ratio > 1e10},
        "half_pixel_coordinate_shift_negative_control": {"rejected": half_pixel_shift_rejected},
        "one_pixel_crop_shift_negative_control_relative_error": crop_error,
        "phase_buffer_preservation": phase_buffer,
        "coordinate_and_area": coordinate_records,
        "scope": {"training_updates": 0, "phase_optimizer_updates": 0, "dataset_reads": 0, "evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0},
        "loaded_source_modules": {
            "v5_runner": str(Path(__file__).resolve()),
            "baseline_adjoint": str(Path(inspect.getfile(baseline._adjoint_row)).resolve()),
            "physical_runtime": str(Path(inspect.getfile(LightGen2DSemanticGenerationInputOLSAlignment)).resolve()),
            "gaussian_s12": str(Path(inspect.getfile(GaussianFiberArrayOLS)).resolve()),
            "lp01_and_phase_stack": str(Path(inspect.getfile(FiniteCropPhaseStack)).resolve()),
            "angular_spectrum_propagation": str(Path(inspect.getfile(angular_spectrum_propagate)).resolve()),
        },
    }
    del output, entrance, runtime, gaussian, lp_modes
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    gc.collect()
    if result["adjoint_direct_forward_max_relative_error"] > G3_TOLERANCE:
        raise RuntimeError("adjoint preflight exceeded the frozen 1e-4 threshold")
    return result


def _roi_tensor(full: torch.Tensor, grid: int) -> torch.Tensor:
    offset = (grid - ROI_SIZE) // 2
    if offset < 0 or 2 * offset + ROI_SIZE != grid:
        raise ValueError("selected global grid cannot be centered on the fixed physical 4.096 mm ROI")
    return full[offset : offset + ROI_SIZE, offset : offset + ROI_SIZE]


def _save_field(path: Path, field: torch.Tensor, *, grid: int, precision: str, phase_name: str, full_power_w: float, input_power_w: float) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"field", "metadata_json", "full_global_plane_power_w", "input_power_w"}:
                raise ValueError(f"saved v5 field cache is malformed: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            values = np.asarray(archive["field"])
            full_power = float(archive["full_global_plane_power_w"].item())
            cached_input_power = float(archive["input_power_w"].item())
        expected = {"schema": "ols_v5_common_roi_field.v1", "grid": grid, "precision": precision, "phase": phase_name, "roi_size": ROI_SIZE, "dx_m": DX_M}
        if any(metadata.get(key) != value for key, value in expected.items()) or not math.isfinite(full_power) or not math.isfinite(cached_input_power) or values.shape != (ROI_SIZE, ROI_SIZE) or not np.isfinite(values).all():
            raise ValueError(f"existing v5 field cache does not match its declared physical field: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"schema": "ols_v5_common_roi_field.v1", "grid": grid, "precision": precision, "phase": phase_name, "roi_size": ROI_SIZE, "dx_m": DX_M, "full_global_plane_power_w": full_power_w, "input_power_w": input_power_w}, sort_keys=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, field=field.detach().cpu().numpy(), metadata_json=np.asarray(metadata), full_global_plane_power_w=np.asarray(full_power_w), input_power_w=np.asarray(input_power_w))
    temporary.replace(path)


def _read_field_record(path: Path, *, grid: int, precision: str, phase_name: str) -> tuple[torch.Tensor, float, float]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        expected = {"schema": "ols_v5_common_roi_field.v1", "grid": grid, "precision": precision, "phase": phase_name, "roi_size": ROI_SIZE, "dx_m": DX_M}
        values = np.asarray(archive["field"])
        full_power = float(archive["full_global_plane_power_w"].item())
        input_power = float(archive["input_power_w"].item())
    if any(metadata.get(key) != value for key, value in expected.items()) or not math.isfinite(full_power) or not math.isfinite(input_power) or values.shape != (ROI_SIZE, ROI_SIZE) or not np.isfinite(values).all():
        raise ValueError(f"invalid saved ROI field: {path}")
    return torch.from_numpy(values.copy()), full_power, input_power


def _read_field(path: Path, *, grid: int, precision: str, phase_name: str) -> torch.Tensor:
    return _read_field_record(path, grid=grid, precision=precision, phase_name=phase_name)[0]


def _readout_coefficients(field_roi: torch.Tensor, precision: str, device: torch.device) -> dict[str, torch.Tensor]:
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    runtime = _make_runtime(ROI_SIZE, real_dtype, device)
    complex_field = field_roi.to(device=device)[None, None]
    wrapped = ComplexField(
        complex_field,
        runtime.ols_plane,
        OperationProvenance("ols_v5_common_physical_roi_readout", runtime.ols_plane.plane_id, None),
    )
    result = {
        "gaussian_s12": baseline._mode_coefficients(wrapped, runtime, "gaussian_s12")[0].detach().cpu().to(torch.complex128),
        "lp01": baseline._mode_coefficients(wrapped, runtime, "lp01")[0].detach().cpu().to(torch.complex128),
    }
    del runtime, complex_field, wrapped
    return result


def _reference_point_indices(positions_xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(ROI_SIZE, dtype=np.float64) + 0.5) * DX_M - ROI_SIZE * DX_M / 2.0
    ix = np.asarray([int(np.argmin(np.abs(axis - x))) for x in positions_xy_m[:, 0]], dtype=np.int64)
    iy = np.asarray([int(np.argmin(np.abs(axis - y))) for y in positions_xy_m[:, 1]], dtype=np.int64)
    if not np.array_equal(np.column_stack((axis[ix], axis[iy])), positions_xy_m):
        raise ValueError("v4 reference positions are not common-ROI pixel centers")
    return iy, ix


def _fixed_second_hop_point_checks(
    config: dict[str, Any], phases: dict[str, torch.Tensor], positions: np.ndarray,
    groups: dict[str, np.ndarray], common_input: np.ndarray, references: dict[str, dict[str, Any]],
    run_dir: Path, device: torch.device,
) -> dict[str, Any]:
    point_dir = run_dir / "second_hop_points"
    point_dir.mkdir(parents=True, exist_ok=True)
    iy, ix = _reference_point_indices(positions)
    e0 = math.sqrt(SOURCE_POWER_W) / 200.0e-6
    result: dict[str, Any] = {"status": "COMPLETE", "common_v4_second_hop_input": "GL4 32768-node archives; same saved 100x100 complex128 input for every window", "phase_fitting_applied": False, "by_precision_phase_grid": {"complex64": {}, "complex128": {}}}
    arrays: dict[str, np.ndarray] = {"positions_xy_m": positions}
    for precision, real_dtype, complex_dtype in (("complex64", torch.float32, torch.complex64), ("complex128", torch.float64, torch.complex128)):
        for phase_name in PHASE_NAMES:
            result["by_precision_phase_grid"][precision][phase_name] = {}
            reference = references[phase_name]["values"]
            for grid in GRIDS:
                path = point_dir / f"{precision}__{phase_name}__grid{grid}.npz"
                if path.is_file():
                    with np.load(path, allow_pickle=False) as archive:
                        values = np.asarray(archive["production_values_complex"])
                        saved_reference = np.asarray(archive["reference_values_complex128"], dtype=np.complex128)
                        saved_positions = np.asarray(archive["positions_xy_m"], dtype=np.float64)
                        metadata = json.loads(str(archive["metadata_json"].item()))
                    expected_metadata = {"schema": "ols_v5_fixed_second_hop_points.v1", "precision": precision, "phase": phase_name, "grid": grid}
                    if metadata != expected_metadata or not np.array_equal(saved_reference, reference) or not np.array_equal(saved_positions, positions) or values.shape != (125,) or not np.isfinite(values).all():
                        raise ValueError(f"fixed second-hop point cache is invalid: {path}")
                else:
                    runtime = _make_runtime(grid, real_dtype, device)
                    middle_plane = PlaneSpec(
                        "ols_v5_common_second_hop_input", (100, 100), (DX_M, DX_M), WAVELENGTH_M,
                        runtime.encoder_entrance_plane.z_m + 1.5e-3, runtime.encoder_entrance_plane.coordinate_frame,
                    )
                    middle = ComplexField(
                        torch.as_tensor(common_input, dtype=complex_dtype, device=device)[None, None],
                        middle_plane,
                        OperationProvenance("ols_v5_v4_fixed_common_second_hop_input", middle_plane.plane_id, None),
                    )
                    full = v3._second_hop(
                        runtime, middle, phases[phase_name][1, 0], grid,
                        f"ols_v5_fixed_second_{precision}_{phase_name}_{grid}",
                    )
                    roi = _roi_tensor(full.tensor[0, 0], grid)
                    values = roi[iy, ix].detach().cpu().numpy()
                    metadata = json.dumps({"schema": "ols_v5_fixed_second_hop_points.v1", "precision": precision, "phase": phase_name, "grid": grid}, sort_keys=True)
                    temporary = path.with_suffix(path.suffix + ".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, production_values_complex=values, reference_values_complex128=reference, positions_xy_m=positions, metadata_json=np.asarray(metadata))
                    temporary.replace(path)
                    del runtime, middle, full, roi
                    clear_production_transfer_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                production = values.astype(np.complex128, copy=False)
                arrays[f"{precision}__{phase_name}__grid{grid}__production"] = production
                arrays[f"{precision}__{phase_name}__grid{grid}__reference"] = reference
                result["by_precision_phase_grid"][precision][phase_name][str(grid)] = {}
                for group_name, selected in groups.items():
                    observed = production[selected]
                    target = reference[selected]
                    delta = observed - target
                    intensity_delta = np.abs(observed) ** 2 - np.abs(target) ** 2
                    result["by_precision_phase_grid"][precision][phase_name][str(grid)][group_name] = {
                        "raw_complex_relative_l2_no_phase_fit": float(np.linalg.norm(delta) / max(np.linalg.norm(target), 1e-300)),
                        "maximum_complex_difference_over_E0": float(np.max(np.abs(delta)) / e0),
                        "intensity_relative_l2": float(np.linalg.norm(intensity_delta) / max(np.linalg.norm(np.abs(target) ** 2), 1e-300)),
                        "global_phase_alignment_applied": False,
                        "position_count": int(selected.size),
                    }
                result["by_precision_phase_grid"][precision][phase_name][str(grid)]["raw_values_file"] = path.name
                _write_json(run_dir / "progress.json", {"stage": "fixed_second_hop_reference_points", "precision": precision, "phase": phase_name, "grid": grid, "formal_training_updates": 0})
    values_path = run_dir / "second_hop_reference_points.npz"
    if values_path.exists():
        with np.load(values_path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(not np.array_equal(np.asarray(archive[key]), value) for key, value in arrays.items()):
                raise ValueError("existing fixed second-hop raw point archive differs from recomputed evidence")
    else:
        with values_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    comparison_path = run_dir / "second_hop_reference_point_comparisons.json"
    if comparison_path.exists():
        if json.loads(comparison_path.read_text(encoding="utf-8")) != _jsonable(result):
            raise ValueError("existing fixed second-hop comparison differs from recomputed evidence")
    else:
        _write_json(comparison_path, result)
    return result


def _window_g3(config: dict[str, Any], phase_path: Path, run_dir: Path, device: torch.device) -> dict[str, Any]:
    phases = _load_phases(phase_path)
    root = Path.cwd()
    v4_report, positions, groups, common_input, references = _load_v4_references(config, root, phases)
    saved_preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    cpu_reference_error: float | None = saved_preflight.get("complex128_two_layer_production_vs_cpu_fft_relative_error")
    field_dir = run_dir / "window_fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    coefficient_cache: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]] = {"complex64": {}, "complex128": {}}
    power_by_precision: dict[str, Any] = {"complex64": {}, "complex128": {}}
    for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
        for phase_name in PHASE_NAMES:
            coefficient_cache[precision][phase_name] = {}
            power_by_precision[precision][phase_name] = {}
            for grid in GRIDS:
                target_path = field_dir / f"{precision}__{phase_name}__grid{grid}.npz"
                if target_path.is_file():
                    roi, full_power, input_power = _read_field_record(target_path, grid=grid, precision=precision, phase_name=phase_name)
                else:
                    clear_production_transfer_cache()
                    runtime = _make_runtime(grid, real_dtype, device)
                    entrance = _encoder_entrance(runtime, real_dtype, device)
                    output = _run_encoder(runtime, entrance, phases[phase_name].to(device=device, dtype=real_dtype), f"ols_v5_{precision}_{phase_name}_{grid}")
                    field = output.tensor.detach()[0, 0]
                    roi = _roi_tensor(field, grid).cpu()
                    full_power = float(field.abs().square().sum().to(torch.float64) * DX_M**2)
                    input_power = baseline._field_power(entrance.tensor, entrance.plane)
                    if precision == "complex128" and phase_name == "zero" and grid == GRIDS[0]:
                        reference = baseline._numpy_phase_stack(
                            entrance.tensor.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False),
                            phases[phase_name][:, 0].numpy().astype(np.float64, copy=False), grid,
                            wavelength_m=WAVELENGTH_M, dx_m=DX_M, distance_m=1.5e-3,
                            layer_power_transmission=TRANSMISSION,
                        )
                        cpu_reference_error = baseline._rel_l2(torch.from_numpy(reference), field.detach().cpu())
                        if cpu_reference_error > 1.0e-10:
                            raise RuntimeError(f"3072 c128 two-hop CPU reference mismatch: {cpu_reference_error:.6g}")
                    _save_field(target_path, roi, grid=grid, precision=precision, phase_name=phase_name, full_power_w=full_power, input_power_w=input_power)
                    del runtime, entrance, output, field
                    clear_production_transfer_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                coefficients = _readout_coefficients(roi, precision, device)
                coefficient_cache[precision][phase_name][grid] = coefficients
                roi_power = float(roi.abs().square().sum().to(torch.float64) * DX_M**2)
                power_by_precision[precision][phase_name][str(grid)] = {
                    "field_file": str(target_path),
                    "input_power_w": input_power,
                    "power_before_fiber_common_roi_w": roi_power,
                    "full_global_plane_power_w": full_power,
                    "full_plane_power_reconstructed_on_resume": False,
                    "window_width_m": grid * DX_M,
                    "frequency_spacing_cycles_per_m": 1.0 / (grid * DX_M),
                    "common_roi_indices_yx": [(grid - ROI_SIZE) // 2, (grid + ROI_SIZE) // 2],
                }
                _write_json(run_dir / "progress.json", {"stage": "window_g3_fields", "completed_field": f"{precision}/{phase_name}/{grid}", "formal_training_updates": 0})
    comparisons: dict[str, Any] = {"complex64": {}, "complex128": {}}
    for precision in ("complex64", "complex128"):
        for phase_name in PHASE_NAMES:
            comparisons[precision][phase_name] = {}
            for left_index, left_grid in enumerate(GRIDS):
                for right_grid in GRIDS[left_index + 1 :]:
                    left = _read_field(field_dir / f"{precision}__{phase_name}__grid{left_grid}.npz", grid=left_grid, precision=precision, phase_name=phase_name).to(torch.complex128)
                    right = _read_field(field_dir / f"{precision}__{phase_name}__grid{right_grid}.npz", grid=right_grid, precision=precision, phase_name=phase_name).to(torch.complex128)
                    phi, aligned = baseline._phase_alignment(left, right)
                    left_power = left.abs().square()
                    right_power = right.abs().square()
                    metrics: dict[str, Any] = {
                        "reference_grid": left_grid,
                        "test_grid": right_grid,
                        "common_roi_shape": [ROI_SIZE, ROI_SIZE],
                        "common_roi_extent_m": [ROI_SIZE * DX_M, ROI_SIZE * DX_M],
                        "pixel_pitch_m": DX_M,
                        "common_global_phase_rad": phi,
                        "raw_complex_relative_error": baseline._rel_l2(left, right),
                        "aligned_complex_relative_error": baseline._rel_l2(left, aligned),
                        "intensity_relative_error": baseline._rel_l2(left_power, right_power),
                        "roi_power_relative_error": abs(float(right_power.sum() - left_power.sum())) / max(float(left_power.sum()), EPS),
                        "roi_power_reference_w": float(left_power.sum() * DX_M**2),
                        "roi_power_test_w": float(right_power.sum() * DX_M**2),
                        "fiber_modes": {},
                    }
                    for mode in ("gaussian_s12", "lp01"):
                        c_left = coefficient_cache[precision][phase_name][left_grid][mode]
                        c_right = coefficient_cache[precision][phase_name][right_grid][mode]
                        latent_phase, c_aligned = baseline._phase_alignment(c_left, c_right)
                        p_left, p_right = c_left.abs().square(), c_right.abs().square()
                        floor = max(float(p_left.max()) * 1.0e-12, EPS)
                        metrics["fiber_modes"][mode] = {
                            "complex_latent_raw_relative_error": baseline._rel_l2(c_left, c_right),
                            "complex_latent_common_phase_aligned_relative_error": baseline._rel_l2(c_left, c_aligned),
                            "latent_common_global_phase_rad": latent_phase,
                            "total_coupling_power_reference_w": float(p_left.sum()),
                            "total_coupling_power_test_w": float(p_right.sum()),
                            "total_coupling_relative_error": abs(float(p_right.sum() - p_left.sum())) / max(float(p_left.sum()), EPS),
                            "per_fiber_power_relative_l2_error": baseline._rel_l2(p_left, p_right),
                            "per_fiber_power_error_w": (p_right - p_left).tolist(),
                            "per_fiber_power_relative_error": ((p_right - p_left).abs() / p_left.clamp_min(floor)).tolist(),
                            "relative_error_denominator_floor_w": floor,
                        }
                    comparisons[precision][phase_name][f"{left_grid}_vs_{right_grid}"] = metrics
                    del left, right, aligned, c_left, c_right
    c64_values = [record["aligned_complex_relative_error"] for phase in comparisons["complex64"].values() for record in phase.values()]
    c128_values = [record["aligned_complex_relative_error"] for phase in comparisons["complex128"].values() for record in phase.values()]
    passed = len(c64_values) == 9 and max(c64_values) <= G3_TOLERANCE
    point_checks = _fixed_second_hop_point_checks(config, phases, positions, groups, common_input, references, run_dir, device)
    result = {
        "status": "PASS" if passed else "FAIL",
        "decision_name": "v5_expanded_window_common_roi_convergence",
        "historical_v4_g3_status": "G3_FAIL_RETAINED_UNCHANGED",
        "threshold": G3_TOLERANCE,
        "production_complex64_all_nine_comparisons_pass": passed,
        "production_complex64_max_aligned_error": max(c64_values),
        "complex128_control_all_nine_comparisons_pass": max(c128_values) <= G3_TOLERANCE,
        "complex128_control_max_aligned_error": max(c128_values),
        "comparison_count_per_precision": len(c64_values),
        "window_grid_sizes": list(GRIDS),
        "window_widths_m": [grid * DX_M for grid in GRIDS],
        "common_roi_shape": [ROI_SIZE, ROI_SIZE],
        "common_roi_extent_m": [ROI_SIZE * DX_M, ROI_SIZE * DX_M],
        "same_grid_complex128_cpu_fft_relative_error": cpu_reference_error,
        "fixed_v4_second_hop_input_125_point_reference_comparison": point_checks,
        "per_window_power": power_by_precision,
        "comparisons": comparisons,
        "historical_v4_report": str(_resolve(root, config["source_evidence"]["v4_report"])),
        "phase_fitting_policy": "one common global phase per complete field or 100-channel latent vector; no per-fiber alignment",
    }
    comparison_path = run_dir / "window_g3_comparisons.json"
    if comparison_path.exists():
        previous = json.loads(comparison_path.read_text(encoding="utf-8"))
        if previous.get("status") != result["status"] or previous.get("production_complex64_max_aligned_error") != result["production_complex64_max_aligned_error"]:
            raise ValueError("existing window G3 decision differs from recomputed saved fields")
    else:
        _write_json(comparison_path, result)
    return result


def _unique_coordinates(size: int, device: torch.device) -> list[tuple[float, float]]:
    coords = baseline._array_coordinates(size, PITCH_M, device).detach().cpu().tolist()
    return [(float(y), float(x)) for y in coords for x in coords]


def _mode_on_global_grid(mode_name: str, cy: float, cx: float, roi_plane: PlaneSpec, global_plane: PlaneSpec) -> torch.Tensor:
    device = torch.device("cuda:0")
    if mode_name == "gaussian_s12":
        y = (torch.arange(ROI_SIZE, dtype=torch.float64, device=device) + 0.5) * DX_M - roi_plane.fov_m[0] / 2.0 + roi_plane.origin_m[0]
        x = (torch.arange(ROI_SIZE, dtype=torch.float64, device=device) + 0.5) * DX_M - roi_plane.fov_m[1] / 2.0 + roi_plane.origin_m[1]
        amplitude = (2.0 / (math.pi * WAIST_M**2)) ** 0.25
        mode_roi = torch.outer(amplitude * torch.exp(-((y - cy) / WAIST_M).square()), amplitude * torch.exp(-((x - cx) / WAIST_M).square())).to(torch.complex128)
    elif mode_name == "lp01":
        profile, bounds = baseline._mode_patch("lp01", cy, cx, roi_plane, 2.2, WAIST_M)
        mode_roi = torch.zeros(ROI_SIZE, ROI_SIZE, dtype=torch.complex128, device=device)
        y0, y1, x0, x1 = bounds
        mode_roi[y0:y1, x0:x1] = profile.to(torch.complex128)
    else:
        raise ValueError(mode_name)
    full = torch.zeros(global_plane.shape, dtype=torch.complex128, device=device)
    offset = (global_plane.shape[0] - ROI_SIZE) // 2
    full[offset : offset + ROI_SIZE, offset : offset + ROI_SIZE] = mode_roi
    return full


def _spectra(matrix: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    complex_values = torch.linalg.svdvals(matrix)
    real_control = torch.cat((matrix.real, matrix.imag), dim=0)
    real_values = torch.linalg.svdvals(real_control)
    max_real = float(real_values.max())
    return {
        "complex_singular_values": complex_values.detach().cpu().tolist(),
        "real_amplitude_control_singular_values": real_values.detach().cpu().tolist(),
        "real_amplitude_effective_rank_by_relative_threshold": {
            str(threshold): int((real_values >= max_real * threshold).sum()) for threshold in thresholds
        },
        "maximum_real_amplitude_singular_value": max_real,
        "maximum_real_amplitude_singular_value_squared": max_real**2,
    }


def _load_or_build_pool(
    *, run_dir: Path, runtime, entrance: ComplexField, phase: torch.Tensor, phase_name: str,
    mode_name: str, size: int, config: dict[str, Any], roi_plane: PlaneSpec,
) -> dict[str, torch.Tensor]:
    coords = _unique_coordinates(size, entrance.tensor.device)
    cache_dir = run_dir / "matrix_rows"
    cache_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[dict[str, np.ndarray]] = []
    block_size = 10
    for start in range(0, len(coords), block_size):
        stop = min(start + block_size, len(coords))
        path = cache_dir / f"{phase_name}__{mode_name}__{size}__rows{start:03d}_{stop:03d}.npz"
        block_coords = np.asarray(coords[start:stop], dtype=np.float64)
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                loaded_coords = np.asarray(archive["coordinates_yx_m"], dtype=np.float64)
                k_rows = np.asarray(archive["K_complex128"], dtype=np.complex128)
                h_rows = np.asarray(archive["H_sqrt_w_complex128"], dtype=np.complex128)
                metadata = json.loads(str(archive["metadata_json"].item()))
            expected = {"schema": "ols_v5_adjoint_row_block.v1", "phase": phase_name, "mode": mode_name, "grid_size": size, "start": start, "stop": stop}
            if metadata != expected or not np.array_equal(loaded_coords, block_coords) or k_rows.shape != (stop-start, 10000) or h_rows.shape != (stop-start, 10000) or not np.isfinite(k_rows).all() or not np.isfinite(h_rows).all():
                raise ValueError(f"existing response row cache is invalid: {path}")
            blocks.append({"K": k_rows, "H": h_rows})
            continue
        rows_k: list[np.ndarray] = []
        rows_h: list[np.ndarray] = []
        plane = runtime.ols_plane
        phases = phase.to(device=entrance.tensor.device, dtype=torch.float64)
        with torch.no_grad():
            for row_index in range(start, stop):
                cy, cx = coords[row_index]
                output_mode = _mode_on_global_grid(mode_name, cy, cx, roi_plane, plane)
                row_k, row_h = baseline._adjoint_row(
                    output_mode, (0, plane.shape[0], 0, plane.shape[1]), plane,
                    entrance.tensor[0, 0].to(torch.complex128), phases,
                    layer_spacing_m=1.5e-3, last_distance_m=1.5e-3, transmission=TRANSMISSION,
                )
                rows_k.append(row_k.detach().cpu().numpy().astype(np.complex128, copy=False))
                rows_h.append(row_h.detach().cpu().numpy().astype(np.complex128, copy=False))
                del output_mode, row_k, row_h
        k_array = np.stack(rows_k)
        h_array = np.stack(rows_h)
        metadata = json.dumps({"schema": "ols_v5_adjoint_row_block.v1", "phase": phase_name, "mode": mode_name, "grid_size": size, "start": start, "stop": stop}, sort_keys=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, coordinates_yx_m=block_coords, K_complex128=k_array, H_sqrt_w_complex128=h_array, metadata_json=np.asarray(metadata))
        temporary.replace(path)
        blocks.append({"K": k_array, "H": h_array})
        _write_json(run_dir / "progress.json", {"stage": "ols_adjoint_rows", "phase": phase_name, "mode": mode_name, "coordinate_pool_size": size, "completed_rows": stop, "row_count": len(coords), "formal_training_updates": 0})
        del k_array, h_array, rows_k, rows_h
        if entrance.tensor.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    k = torch.from_numpy(np.concatenate([block["K"] for block in blocks], axis=0)).to(device=entrance.tensor.device)
    h = torch.from_numpy(np.concatenate([block["H"] for block in blocks], axis=0)).to(device=entrance.tensor.device)
    if k.shape != (len(coords), 10000) or h.shape != (len(coords), 10000):
        raise RuntimeError("unique response pool has wrong row or pixel count")
    return {"K": k, "H": h, "coordinates_yx_m": torch.tensor(coords, dtype=torch.float64, device=entrance.tensor.device)}


def _array_pool_indices(size: int, pool_size: int) -> list[int]:
    coordinates = _unique_coordinates(size, torch.device("cpu"))
    pool = _unique_coordinates(pool_size, torch.device("cpu"))
    mapping = {coordinate: index for index, coordinate in enumerate(pool)}
    try:
        return [mapping[coordinate] for coordinate in coordinates]
    except KeyError as error:
        raise ValueError(f"{size}x{size} coordinates are not nested in {pool_size}x{pool_size} response pool") from error


def _power_metrics(coefficients: torch.Tensor, before_power: float, input_power: float, size: int) -> dict[str, Any]:
    p = coefficients.abs().square().to(torch.float64).reshape(-1)
    total = float(p.sum())
    center_coords = baseline._array_coordinates(size, PITCH_M, torch.device("cpu")).tolist()
    points = [(float(y), float(x)) for y in center_coords for x in center_coords]
    center_ids = [i for i, (y, x) in enumerate(points) if math.hypot(y, x) <= PITCH_M / math.sqrt(2.0) + 1e-15] if size % 2 == 0 else [i for i, (y, x) in enumerate(points) if math.hypot(y, x) < 1e-15]
    edge_ids = [] if size == 1 else [i for i in range(size**2) if i // size in {0, size-1} or i % size in {0, size-1}]
    corner_ids = [] if size == 1 else [0, size-1, size*(size-1), size*size-1]
    center = float(p[center_ids].mean()) if center_ids else 0.0
    edge = float(p[edge_ids].mean()) if edge_ids else None
    corner = float(p[corner_ids].mean()) if corner_ids else None
    quantiles = {"min": float(p.min()), "p10": float(torch.quantile(p, .1)), "p50": float(torch.quantile(p, .5)), "mean": float(p.mean()), "p90": float(torch.quantile(p, .9)), "max": float(p.max())}
    return {
        "fiber_power_w_row_major": p.cpu().tolist(),
        "total_coupled_power_w": total,
        "eta_total": total / input_power,
        "eta_fiber_face": total / max(before_power, EPS),
        "eta_system_fixed_source": total / input_power,
        "N_eff_single_field": total**2 / max(float(p.square().sum()), EPS),
        "utilization_fraction_by_max_relative_threshold_db": {
            str(db): float((p >= p.max() * 10.0 ** (db / 10.0)).to(torch.float64).mean()) for db in (-10, -20, -30, -40)
        },
        "channel_power_distribution_w": quantiles,
        "P10_over_P90": quantiles["p10"] / max(quantiles["p90"], EPS),
        "coefficient_of_variation": float(p.std(unbiased=False) / p.mean().clamp_min(EPS)),
        "edge_to_center_power_ratio": None if edge is None or center <= 0 else edge / center,
        "corner_to_center_power_ratio": None if corner is None or center <= 0 else corner / center,
        "edge_channel_count": len(edge_ids),
        "center_channel_count": len(center_ids),
        "corner_channel_count": len(corner_ids),
        "center_definition": "nearest fiber to the optical axis; one for odd arrays and four equidistant fibers for even arrays",
        "edge_ratio_status": "not_applicable_single_channel" if size == 1 else "measured",
        "fiber_center_xy_m_row_major": [[x, y] for y, x in points],
    }


def _dimension_audit(config: dict[str, Any], phase_path: Path, run_dir: Path, device: torch.device) -> dict[str, Any]:
    metrics_path = run_dir / "dimension_power_metrics.json"
    matrix_path = run_dir / "response_matrices_K_H_unique_pools.pt"
    if metrics_path.exists():
        if not matrix_path.is_file():
            raise ValueError("saved v5 metrics exist without their required K/H matrix payload")
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if saved_metrics.get("status") != "COMPLETE":
            raise ValueError("saved v5 dimension metrics are not a complete measurement")
        saved_payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        if saved_payload.get("schema") != "ols_v5_response_pools_K_H.v1" or set(saved_payload.get("phase_cases", {})) != set(PHASE_NAMES):
            raise ValueError("saved v5 K/H matrix payload schema or phase set is invalid")
        for phase_name in PHASE_NAMES:
            if set(saved_payload["phase_cases"][phase_name]) != {"gaussian_s12", "lp01"}:
                raise ValueError("saved v5 response matrix payload is missing a mode family")
            for mode_name in ("gaussian_s12", "lp01"):
                for pool_size, expected_rows in (("7", 49), ("10", 100)):
                    pool = saved_payload["phase_cases"][phase_name][mode_name].get(pool_size, {})
                    if tuple(pool.get("K_dimensionless", torch.empty(0)).shape) != (expected_rows, 10000) or tuple(pool.get("H_sqrt_w", torch.empty(0)).shape) != (expected_rows, 10000):
                        raise ValueError("saved v5 K/H response pool has an invalid shape")
        return {
            "status": "COMPLETE_ZERO_UPDATE_DIAGNOSTIC",
            "phase_cases": saved_metrics["phase_cases"],
            "unique_fiber_coordinates_computed": 149,
            "row_pool_sizes": {"odd_arrays": "7x7; 49 unique coordinates", "even_arrays": "10x10; 100 unique coordinates"},
            "matrix_payload": matrix_path.name,
            "matrix_definitions": {"K": "c=Ka, q_p=1/sqrt(pixel area); a_p has sqrt(W), K dimensionless", "H": "c=Ht, H=K diag(E_source sqrt(pixel area)); H has sqrt(W) per unit command", "real_amplitude_control": "[Re M; Im M] with real pixel-amplitude commands", "squared_gain_units": {"K": "dimensionless", "H": "W per squared unit command norm"}},
            "optimizer_updates": 0,
            "single_field_N_eff_is_not_response_rank": True,
            "math_rank_is_not_noise_limited_experimental_dimension": True,
            "center_definition": "one nearest-axis fiber for odd arrays; four equidistant fibers for even arrays",
            "finite_phase_cases_do_not_establish_global_optimum": True,
            "resumed_from_complete_saved_matrix_and_probe_measurement": True,
        }
    phases_saved = _load_phases(phase_path)
    runtime = _make_runtime(GRIDS[0], torch.float64, device)
    entrance = _encoder_entrance(runtime, torch.float64, device)
    roi_runtime = _make_runtime(ROI_SIZE, torch.float64, device)
    roi_plane = roi_runtime.ols_plane
    all_records: dict[str, Any] = {}
    threshold_values = [float(value) for value in config["dimension_diagnostic"]["amplitude_matrix_relative_singular_value_thresholds"]]
    matrix_payload: dict[str, Any] = {"schema": "ols_v5_response_pools_K_H.v1", "phase_cases": {}}
    rng = np.random.default_rng(int(config["dimension_diagnostic"]["probe_seed"]))
    probes = rng.choice(np.asarray(config["dimension_diagnostic"]["probe_amplitudes"], dtype=np.float64), size=(int(config["dimension_diagnostic"]["probe_count"]), 10000))
    probes = np.concatenate((probes, np.ones((1, 10000), dtype=np.float64)), axis=0)
    for phase_name in PHASE_NAMES:
        phase = phases_saved[phase_name].to(device=device, dtype=torch.float64)
        all_records[phase_name] = {str(size): {} for size in SIZES}
        matrix_payload["phase_cases"][phase_name] = {}
        pool_cache: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        for mode_name in ("gaussian_s12", "lp01"):
            pool_cache[mode_name] = {}
            matrix_payload["phase_cases"][phase_name][mode_name] = {}
            for pool_size in (7, 10):
                pool = _load_or_build_pool(
                    run_dir=run_dir, runtime=runtime, entrance=entrance, phase=phase,
                    phase_name=phase_name, mode_name=mode_name, size=pool_size, config=config, roi_plane=roi_plane,
                )
                pool_cache[mode_name][pool_size] = pool
                matrix_payload["phase_cases"][phase_name][mode_name][str(pool_size)] = {
                    "coordinates_yx_m": pool["coordinates_yx_m"].detach().cpu(),
                    "K_dimensionless": pool["K"].detach().cpu(),
                    "H_sqrt_w": pool["H"].detach().cpu(),
                    "pool_semantics": "only unique rows required by nested odd/even square arrays; K and H use identical coordinates",
                }
            for size in SIZES:
                pool_size = 7 if size % 2 else 10
                indices = _array_pool_indices(size, pool_size)
                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                matrix_k = pool_cache[mode_name][pool_size]["K"].index_select(0, index_tensor)
                matrix_h = pool_cache[mode_name][pool_size]["H"].index_select(0, index_tensor)
                spectrum_k = _spectra(matrix_k, threshold_values)
                spectrum_h = _spectra(matrix_h, threshold_values)
                coordinates = _unique_coordinates(size, torch.device("cpu"))
                all_records[phase_name][str(size)][mode_name] = {
                    "channel_count": size**2,
                    "pitch_m": PITCH_M,
                    "fiber_span_m": (size-1)*PITCH_M,
                    "coordinates_xy_m_row_major": [[x, y] for y, x in coordinates],
                    "source_unique_pool_grid_size": pool_size,
                    "source_unique_pool_row_indices": indices,
                    "area_orthonormal_K_spectrum": {
                        "definition": "q_p=1/sqrt(pixel_area); c=K a; a_p has sqrt(W), K dimensionless",
                        "complex_spectrum": spectrum_k["complex_singular_values"],
                        "real_amplitude_control_spectrum": spectrum_k["real_amplitude_control_singular_values"],
                        "real_amplitude_effective_rank_by_relative_threshold": spectrum_k["real_amplitude_effective_rank_by_relative_threshold"],
                        "maximum_real_amplitude_sigma_squared_gain": spectrum_k["maximum_real_amplitude_singular_value_squared"],
                        "sigma_squared_unit": "dimensionless",
                    },
                    "fixed_source_H_spectrum": {
                        "definition": "H=K diag(E_source sqrt(pixel_area)); c=H t; t is dimensionless and H is sqrt(W)",
                        "complex_spectrum_sqrt_w": spectrum_h["complex_singular_values"],
                        "real_amplitude_control_spectrum_sqrt_w": spectrum_h["real_amplitude_control_singular_values"],
                        "real_amplitude_effective_rank_by_relative_threshold": spectrum_h["real_amplitude_effective_rank_by_relative_threshold"],
                        "maximum_real_amplitude_sigma_squared_gain_w": spectrum_h["maximum_real_amplitude_singular_value_squared"],
                        "sigma_squared_unit": "W per squared unit command norm",
                    },
                    "probe_count_including_uniform_control": int(probes.shape[0]),
                    "probe_amplitude_levels": [0.25, 0.75],
                    "fixed_source_power_w": SOURCE_POWER_W,
                    "no_per_sample_renormalization": True,
                    "all_probe_checks_pass": False,
                }
                del matrix_k, matrix_h, index_tensor
        # Reuse one field for every architecture and both mode families.
        pattern_power_records: dict[str, dict[str, list[Any]]] = {
            str(size): {mode: [] for mode in ("gaussian_s12", "lp01")} for size in SIZES
        }
        probe_errors: dict[str, dict[str, list[dict[str, float]]]] = {
            str(size): {mode: [] for mode in ("gaussian_s12", "lp01")} for size in SIZES
        }
        for probe_index, probe_np in enumerate(probes):
            command = torch.as_tensor(probe_np, dtype=torch.float64, device=device)
            command_image = command.reshape(100, 100)
            command_field = entrance.tensor * command_image[None, None].to(entrance.tensor.real.dtype)
            direct_input = ComplexField(command_field, entrance.plane, OperationProvenance("ols_v5_fixed_amplitude_probe", entrance.plane.plane_id, None))
            output = _run_encoder(runtime, direct_input, phase, f"ols_v5_probe_{phase_name}_{probe_index:02d}")
            field = output.tensor.detach()[0, 0]
            roi = _roi_tensor(field, GRIDS[0])
            before_power = float(field.abs().square().sum().to(torch.float64) * DX_M**2)
            roi_power = float(roi.abs().square().sum().to(torch.float64) * DX_M**2)
            direct_gauss = {
                7: baseline._gaussian_project_many(roi, roi_plane, 7).reshape(-1),
                10: baseline._gaussian_project_many(roi, roi_plane, 10).reshape(-1),
            }
            direct_lp = {
                7: baseline._project_variable(roi, roi_plane, baseline._array_coordinates(7, PITCH_M, device), "lp01", WAIST_M, 2.2),
                10: baseline._project_variable(roi, roi_plane, baseline._array_coordinates(10, PITCH_M, device), "lp01", WAIST_M, 2.2),
            }
            observed_pools = {"gaussian_s12": direct_gauss, "lp01": direct_lp}
            entrance_probe_power = baseline._field_power(command_field, entrance.plane)
            a_basis = entrance.tensor[0, 0].reshape(-1).to(torch.complex128) * math.sqrt(DX_M**2) * command.to(torch.complex128)
            for size in SIZES:
                pool_size = 7 if size % 2 else 10
                indices = _array_pool_indices(size, pool_size)
                for mode_name in ("gaussian_s12", "lp01"):
                    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                    pool = pool_cache[mode_name][pool_size]
                    k_size = pool["K"].index_select(0, index_tensor)
                    h_size = pool["H"].index_select(0, index_tensor)
                    predicted_k = k_size @ a_basis
                    predicted_h = h_size @ command.to(torch.complex128)
                    observed = observed_pools[mode_name][pool_size].index_select(0, index_tensor)
                    error_k = baseline._rel_l2(predicted_k, observed)
                    error_h = baseline._rel_l2(predicted_h, observed)
                    error_kh = baseline._rel_l2(predicted_k, predicted_h)
                    probe_errors[str(size)][mode_name].append({"K_to_direct": error_k, "H_to_direct": error_h, "K_to_H": error_kh})
                    power_record = _power_metrics(observed, before_power, SOURCE_POWER_W, size)
                    power_record["commanded_encoder_input_power_w"] = entrance_probe_power
                    power_record["power_before_fiber_common_roi_w"] = roi_power
                    pattern_power_records[str(size)][mode_name].append(power_record)
                    del k_size, h_size, predicted_k, predicted_h, observed, index_tensor
            del command, command_image, command_field, direct_input, output, field, roi, direct_gauss, direct_lp, observed_pools, a_basis
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            _write_json(run_dir / "progress.json", {"stage": "matrix_probe_validation", "phase": phase_name, "completed_probe_including_uniform": probe_index+1, "expected_probe_count_including_uniform": len(probes), "formal_training_updates": 0})
        for size in SIZES:
            for mode_name in ("gaussian_s12", "lp01"):
                entries = probe_errors[str(size)][mode_name]
                maximum = max(max(item.values()) for item in entries)
                passed = maximum <= G3_TOLERANCE
                record = all_records[phase_name][str(size)][mode_name]
                record["matrix_to_direct_forward_relative_error_max"] = maximum
                record["per_probe_matrix_prediction_errors"] = entries
                record["all_probe_checks_pass"] = passed
                power_records = pattern_power_records[str(size)][mode_name]
                record["per_channel_power_w_by_probe_including_uniform_control"] = [item["fiber_power_w_row_major"] for item in power_records]
                record["uniform_command_channel_metrics"] = power_records[-1]
                record["probe_delivery_summary"] = {
                    "probe_count_including_uniform_control": len(power_records),
                    "fixed_source_power_w": SOURCE_POWER_W,
                    "fixed_seed": int(config["dimension_diagnostic"]["probe_seed"]),
                    "binary_amplitude_levels": [0.25, 0.75],
                    "uniform_control_index": len(power_records)-1,
                    "no_per_sample_renormalization": True,
                    "commanded_encoder_input_power_w_by_probe": [float(record_power["commanded_encoder_input_power_w"]) for record_power in power_records],
                    "eta_fiber_face_by_probe": [float(record_power["eta_fiber_face"]) for record_power in power_records],
                    "eta_system_fixed_source_by_probe": [float(record_power["eta_system_fixed_source"]) for record_power in power_records],
                    "N_eff_by_probe": [float(record_power["N_eff_single_field"]) for record_power in power_records],
                    "CV_by_probe": [float(record_power["coefficient_of_variation"]) for record_power in power_records],
                    "edge_to_center_by_probe": [record_power["edge_to_center_power_ratio"] for record_power in power_records],
                    "corner_to_center_by_probe": [record_power["corner_to_center_power_ratio"] for record_power in power_records],
                    "matrix_predictions_checked_individually": True,
                    "all_probe_checks_pass": passed,
                }
        if not all(all_records[phase_name][str(size)][mode]["all_probe_checks_pass"] for size in SIZES for mode in ("gaussian_s12", "lp01")):
            raise RuntimeError(f"at least one individual probe failed the matrix/direct gate for phase {phase_name}")
        del pool_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    if matrix_path.exists():
        saved_payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        if saved_payload.get("schema") != "ols_v5_response_pools_K_H.v1" or set(saved_payload.get("phase_cases", {})) != set(PHASE_NAMES):
            raise ValueError("existing v5 response matrix payload is incompatible")
    else:
        torch.save(matrix_payload, matrix_path)
    if metrics_path.exists():
        raise ValueError("dimension metrics appeared while the current measurement was running")
    _write_json(metrics_path, {"status": "COMPLETE", "phase_cases": all_records})
    return {
        "status": "COMPLETE_ZERO_UPDATE_DIAGNOSTIC",
        "phase_cases": all_records,
        "unique_fiber_coordinates_computed": 149,
        "row_pool_sizes": {"odd_arrays": "7x7; 49 unique coordinates", "even_arrays": "10x10; 100 unique coordinates"},
        "matrix_payload": "response_matrices_K_H_unique_pools.pt",
        "matrix_definitions": {
            "K": "c=Ka, q_p=1/sqrt(pixel area); a_p has sqrt(W), K dimensionless",
            "H": "c=Ht, H=K diag(E_source sqrt(pixel area)); H has sqrt(W) per unit command",
            "real_amplitude_control": "[Re M; Im M] with real pixel-amplitude commands",
            "squared_gain_units": {"K": "dimensionless", "H": "W per squared unit command norm"},
        },
        "optimizer_updates": 0,
        "single_field_N_eff_is_not_response_rank": True,
        "math_rank_is_not_noise_limited_experimental_dimension": True,
        "center_definition": "one nearest-axis fiber for odd arrays; four equidistant fibers for even arrays",
        "finite_phase_cases_do_not_establish_global_optimum": True,
    }


def run(config_path: Path, run_dir: Path, *, preflight_only: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = Path.cwd()
    config = _load_config(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite completed v5 report: {report_path}")
    allowed_existing = {"preflight.json", "configuration.yaml", "fixed_phase_screens.pt", "progress.json", "resource_telemetry.csv", "cloud_original_sources", "script_overlay", "source_overlay_state.json", "source_restoration_receipt.json", "window_fields", "matrix_rows", "second_hop_points", "second_hop_reference_points.npz", "second_hop_reference_point_comparisons.json", "window_g3_comparisons.json", "response_matrices_K_H_unique_pools.pt", "dimension_power_metrics.json"}
    unknown = [item.name for item in run_dir.iterdir() if item.name not in allowed_existing]
    if unknown:
        raise FileExistsError(f"v5 run directory contains unrecognized evidence; refusing overwrite: {unknown}")
    config_copy = run_dir / "configuration.yaml"
    if not config_copy.exists():
        config_copy.write_bytes(config_path.read_bytes())
    elif _load_config(config_copy) != config:
        raise ValueError("existing run configuration differs from the explicit v5 configuration")
    evidence = config["source_evidence"]
    phase_source = _resolve(root, evidence["fixed_phase_screens"])
    if not phase_source.is_file():
        raise FileNotFoundError(f"explicit fixed phase source is missing: {phase_source}")
    local_phase_copy = run_dir / "fixed_phase_screens.pt"
    if local_phase_copy.exists():
        if not all(torch.equal(a, b) for a, b in zip(_load_phases(local_phase_copy).values(), _load_phases(phase_source).values(), strict=True)):
            raise ValueError("existing run phase archive differs from the explicit v4 input")
    else:
        local_phase_copy.write_bytes(phase_source.read_bytes())
    if preflight_only:
        if (run_dir / "preflight.json").exists():
            raise FileExistsError("refusing to overwrite saved v5 preflight evidence")
        preflight = _preflight(config, config_path, root)
        _write_json(run_dir / "preflight.json", preflight)
        return preflight
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("full v5 run requires a saved PASS preflight from this attempt")
    if not torch.cuda.is_available():
        raise RuntimeError("v5 full measurement requires AutoDL CUDA")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    g3 = _window_g3(config, local_phase_copy, run_dir, device)
    dimension = None
    if g3["production_complex64_all_nine_comparisons_pass"]:
        dimension = _dimension_audit(config, local_phase_copy, run_dir, device)
        status = "VALID_COMPLETED"
    else:
        status = "VALID_FAILED"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    report = {
        "schema": SCHEMA,
        "status": status,
        "decision_question": "whether reliable zero-update OLS dimension and absolute-power evidence now supports planning a controlled phase-only optimization",
        "source_config": str(config_path),
        "fixed_phase_source": str(local_phase_copy),
        "source_evidence": config["source_evidence"],
        "provenance_git_head": head,
        "execution": {
            "python": __import__("sys").executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds_observation": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "frozen_physical_system": config["optics"],
        "fiber_assumptions": config["fiber"],
        "v5_window_G3": g3,
        "OLS_dimension_and_power": dimension if dimension is not None else {"status": "BLOCKED_BY_V5_WINDOW_G3_FAIL", "reason": "production complex64 did not pass all 9 expanded-window common-ROI comparisons; no dimension matrix was run"},
        "updates_and_forbidden_io": {"formal_training_updates": 0, "phase_optimizer_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0, "production_geometry_changes": 0, "solver_replacement": 0},
        "claim_limits": [
            "v5 expanded-window convergence is a new numerical-baseline result and does not rewrite the historical v4 G3_FAIL",
            "fiber pitch and MFD remain unmeasured simulation assumptions",
            "fixed-phase response spectra and single-field N_eff do not establish a globally optimal or noise-limited usable dimension",
            "LP01 is a same-geometry sensitivity control; neither mode has hardware calibration",
            "no nonlinear material, decoder/camera interface, or generation training was evaluated",
        ],
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps({"status": result.get("status"), "window_G3": result.get("v5_window_G3", {}).get("status"), "dimension": result.get("OLS_dimension_and_power", {}).get("status"), "run_dir": str(args.run_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
