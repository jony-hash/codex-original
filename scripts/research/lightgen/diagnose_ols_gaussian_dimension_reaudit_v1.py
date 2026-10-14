"""Zero-update G3 convergence and gated OLS response-dimension audit."""

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
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS
from ldm_project.v2.optics.fiber_operator import lp01_eigenpair, lp01_radial_profile
from ldm_project.v2.optics.propagation import (
    _transfer_function,
    angular_spectrum_propagate,
    clear_production_transfer_cache,
)


SCHEMA = "ols_gaussian_dimension_reaudit.v1"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_gaussian_dimension_reaudit_v1.yaml")
DEFAULT_PHASES = Path(
    "outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/"
    "autodl_backup_attempt_05/fixed_phase_screens.pt"
)
DEFAULT_RUN_DIR = Path(
    "outputs/20261013_ols_gaussian_dimension_reaudit_v1/runs/attempt_01"
)
PHASE_NAMES = ("zero", "last_layer_x_1deg", "last_layer_xy_1deg")
SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 10)
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


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    expected = {
        "wavelength_m": 532.0e-9,
        "pixel_pitch_m": 2.0e-6,
        "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6,
        "propagation_windows_mm": [4.096, 6.144, 8.192],
        "propagation_grid_sizes": [2048, 3072, 4096],
        "inter_encoder_spacing_m": 1.5e-3,
        "last_encoder_to_fiber_m": 1.5e-3,
        "phase_screen_count": 2,
    }
    for key, expected_value in expected.items():
        if value["optics"].get(key) != expected_value:
            raise ValueError(f"frozen optics.{key} changed")
    expected_fiber = {
        "reference_pitch_m": 125.0e-6,
        "cladding_diameter_m": 125.0e-6,
        "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": 7.111e-6,
        "lp01_v_number": 2.2,
        "transport": "ideal_identity",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }
    for key, expected_value in expected_fiber.items():
        if value["fiber"].get(key) != expected_value:
            raise ValueError(f"frozen fiber.{key} changed")
    if value["scope"] != {
        "formal_training": "forbidden",
        "dataset_read": "forbidden",
        "task_evaluator": "forbidden",
        "fid": "forbidden",
        "task_loss_change": "forbidden",
        "production_default_migration": "forbidden",
        "optical_geometry_change": "forbidden",
        "material_nonlinearity": "forbidden",
        "phase_optimizer_updates": "forbidden",
    }:
        raise ValueError("scope contract changed")
    if value["gates"]["g3_complex_relative_error_after_common_global_phase_max"] != 1.0e-4:
        raise ValueError("G3 tolerance must remain 1e-4")
    return value


def _phase_alignment(reference: torch.Tensor, test: torch.Tensor) -> tuple[float, torch.Tensor]:
    phi = torch.angle((reference.conj() * test).sum())
    return float(phi), test * torch.exp(torch.complex(torch.zeros_like(phi), -phi))


def _rel_l2(reference: torch.Tensor, test: torch.Tensor) -> float:
    ref = reference.to(torch.complex128 if torch.is_complex(reference) else torch.float64)
    candidate = test.to(torch.complex128 if torch.is_complex(test) else torch.float64)
    return float(torch.linalg.vector_norm(ref - candidate) / torch.linalg.vector_norm(ref).clamp_min(EPS))


def _field_power(field: torch.Tensor, plane: PlaneSpec) -> float:
    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    return float(field.abs().square().sum().to(torch.float64) * area)


def _make_runtime(grid_size: int, real_dtype: torch.dtype, device: torch.device):
    config = LightGenPhysicalOpticsConfig(
        input_power_w=249.3e-6,
        layer_power_transmission=0.889,
        global_fiber_geometry=True,
        fiber_pitch_m=125.0e-6,
        fiber_cladding_diameter_m=125.0e-6,
        fiber_mfd_m=14.222e-6,
        fiber_length_m=8.5e-3,
        fiber_attenuation_db_per_km=0.0,
    )
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(
            propagation_grid_size=grid_size,
            ols_model=LP01_OLS_MODEL,
            physical_optics=config,
        )
    ).to(device=device, dtype=real_dtype).eval()
    return runtime


def _encoder_entrance(runtime, real_dtype: torch.dtype, device: torch.device) -> ComplexField:
    image = torch.ones((1, 1, 28, 28), dtype=real_dtype, device=device)
    image_input = runtime.input_adapter(image)
    slm_area = math.prod(runtime.slm_plane.pixel_pitch_m)
    amplitude = math.sqrt(
        runtime.physical_optics.input_power_w
        / (runtime.slm_plane.shape[0] * runtime.slm_plane.shape[1] * slm_area)
    )
    illumination_real = torch.full_like(image_input.slm_field.tensor.real, amplitude)
    illumination = torch.complex(illumination_real, torch.zeros_like(illumination_real))
    calibrated = ComplexField(
        illumination * image_input.slm_field.tensor,
        runtime.slm_plane,
        OperationProvenance(
            "ols_gaussian_dimension_audit_fixed_absolute_input",
            runtime.slm_plane.plane_id,
            None,
            {"input_power_w": runtime.physical_optics.input_power_w, "downstream_renormalization": False},
        ),
    )
    return runtime.input_relay(calibrated)


def _run_encoder(runtime, entrance: ComplexField, phase: torch.Tensor, prefix: str) -> ComplexField:
    with torch.no_grad():
        runtime.encoder.phase.copy_(phase.to(device=runtime.encoder.phase.device, dtype=runtime.encoder.phase.dtype))
        output, _records, _statistics = runtime.encoder(entrance, prefix)
    return ComplexField(
        output.tensor,
        runtime.ols_plane,
        OperationProvenance("ols_gaussian_dimension_audit_encoder_output", output.plane.plane_id, None),
    )


def _mode_coefficients(
    field: ComplexField,
    runtime,
    basis: str,
    *,
    pitch_m: float = 125.0e-6,
    mode_field_diameter_m: float = 14.222e-6,
) -> torch.Tensor:
    if basis == "lp01":
        return runtime.ols.operator.project_coefficients(field)
    if basis == "gaussian_s12":
        gaussian = GaussianFiberArrayOLS(
            field.plane,
            GaussianFiberGeometry(port_pitch_m=pitch_m, waist_m=mode_field_diameter_m / 2.0),
        ).to(device=field.tensor.device)
        return gaussian.project_coefficients(field)
    raise ValueError(f"unsupported mode basis: {basis}")


def _numpy_asm(field: np.ndarray, wavelength_m: float, dx_m: float, distance_m: float) -> np.ndarray:
    ny, nx = field.shape
    fy = np.fft.fftfreq(ny, d=dx_m)[:, None]
    fx = np.fft.fftfreq(nx, d=dx_m)[None, :]
    radicand = 1.0 - (wavelength_m * fx) ** 2 - (wavelength_m * fy) ** 2
    transfer = np.exp(
        2j * np.pi * distance_m / wavelength_m * np.sqrt(np.maximum(radicand, 0.0))
    ) * (radicand >= 0.0)
    return np.fft.ifft2(np.fft.fft2(field, norm="ortho") * transfer, norm="ortho")


def _numpy_phase_stack(
    entrance: np.ndarray,
    phases: np.ndarray,
    grid_size: int,
    *,
    wavelength_m: float,
    dx_m: float,
    distance_m: float,
    layer_power_transmission: float,
) -> np.ndarray:
    side = entrance.shape[0]
    start = (grid_size - side) // 2
    current = entrance
    for layer_index in range(2):
        modulated = current * np.sqrt(layer_power_transmission) * np.exp(1j * phases[layer_index])
        padded = np.zeros((grid_size, grid_size), dtype=np.complex128)
        padded[start : start + side, start : start + side] = modulated
        propagated = _numpy_asm(padded, wavelength_m, dx_m, distance_m)
        if layer_index == 0:
            current = propagated[start : start + side, start : start + side]
        else:
            return propagated
    raise RuntimeError("two-layer propagation did not produce an output")


def _analytic_plane_wave_check(device: torch.device) -> dict[str, Any]:
    size, dx, wavelength, distance = 256, 2.0e-6, 532.0e-9, 1.5e-3
    m, n = 3, -5
    y = (torch.arange(size, device=device, dtype=torch.float64) + 0.5) * dx - size * dx / 2.0
    x = (torch.arange(size, device=device, dtype=torch.float64) + 0.5) * dx - size * dx / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    source = torch.exp(2j * math.pi * (n * xx + m * yy) / (size * dx)).to(torch.complex128)
    plane = PlaneSpec("audit_plane_wave", (size, size), (dx, dx), wavelength, 0.0, "analytic_probe")
    result = angular_spectrum_propagate(
        ComplexField(source[None, None], plane, OperationProvenance("audit_plane_wave", plane.plane_id, None)),
        distance,
        "audit_plane_wave_output",
    ).tensor[0, 0]
    fx, fy = n / (size * dx), m / (size * dx)
    radicand = 1.0 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2
    expected_phase = 2.0 * math.pi * distance * math.sqrt(radicand) / wavelength
    expected = source * complex(math.cos(expected_phase), math.sin(expected_phase))
    wrong_sign = source * complex(math.cos(expected_phase), -math.sin(expected_phase))
    error = _rel_l2(expected, result)
    wrong_sign_error = _rel_l2(wrong_sign, result)
    intensity_error = _rel_l2(source.abs().square(), result.abs().square())
    if error > 1.0e-11 or intensity_error > 1.0e-11 or wrong_sign_error < 1.0e-3:
        raise RuntimeError("analytic plane-wave propagation preflight failed")
    return {
        "production_complex128_relative_error": error,
        "intensity_relative_error": intensity_error,
        "wrong_transfer_phase_negative_control_error": wrong_sign_error,
        "negative_control_rejected": wrong_sign_error > 1.0e-3,
        "grid_size": size,
        "frequency_bins": [m, n],
    }


def _run_preflight(config: dict[str, Any], phase_path: Path, device: torch.device) -> dict[str, Any]:
    phases = torch.load(phase_path, map_location="cpu", weights_only=True)
    missing = sorted(set(PHASE_NAMES) - set(phases))
    if missing:
        raise ValueError(f"saved phase file is missing: {missing}")
    phase = phases["zero"].to(device=device, dtype=torch.float64)
    grid = 2048
    runtime = _make_runtime(grid, torch.float64, device)
    entrance = _encoder_entrance(runtime, torch.float64, device)
    output = _run_encoder(runtime, entrance, phase, "ols_g3_preflight")
    reference = _numpy_phase_stack(
        entrance.tensor.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False),
        phase[:, 0].detach().cpu().numpy().astype(np.float64, copy=False),
        grid,
        wavelength_m=5.32e-7,
        dx_m=2.0e-6,
        distance_m=1.5e-3,
        layer_power_transmission=0.889,
    )
    cpu_error = _rel_l2(torch.from_numpy(reference), output.tensor.detach().cpu()[0, 0])
    if cpu_error > 1.0e-10:
        raise RuntimeError(f"independent CPU reference failed: relative error={cpu_error:.6g}")
    lp = _mode_coefficients(output, runtime, "lp01")
    gaussian = _mode_coefficients(output, runtime, "gaussian_s12")
    if not torch.isfinite(lp).all() or not torch.isfinite(gaussian).all():
        raise RuntimeError("fiber projection preflight returned nonfinite complex values")
    if lp.numel() != 100 or gaussian.numel() != 100:
        raise RuntimeError("10x10 mode preflight did not return 100 channels")
    gaussian_operator = GaussianFiberArrayOLS(
        output.plane, GaussianFiberGeometry(port_pitch_m=125.0e-6, waist_m=7.111e-6)
    ).to(device=device)
    gaussian_passivity = gaussian_operator.geometry_diagnostics
    if float(gaussian_passivity["complete_operator_power_gain_bound"]) > 1.0 + 1.0e-5:
        raise RuntimeError("Gaussian S12 array projection/re-emission is non-passive")
    lp_operator = runtime.ols.operator
    lp_modes = [getattr(lp_operator, f"input_mode_patch_{i}") for i in range(100)]
    lp_area = math.prod(output.plane.pixel_pitch_m)
    lp_norm_errors = [
        abs(float(mode.abs().square().sum() * lp_area) - 1.0) for mode in lp_modes
    ]
    if max(lp_norm_errors) > 5.0e-6:
        raise RuntimeError("LP01 global-plane mode norm preflight failed")
    phase_x = phases["last_layer_x_1deg"].to(device=device, dtype=torch.float64)
    output_x = _run_encoder(runtime, entrance, phase_x, "ols_g3_preflight_nonzero_phase")
    cpu_x = _numpy_phase_stack(
        entrance.tensor.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False),
        phase_x[:, 0].detach().cpu().numpy().astype(np.float64, copy=False),
        grid,
        wavelength_m=5.32e-7,
        dx_m=2.0e-6,
        distance_m=1.5e-3,
        layer_power_transmission=0.889,
    )
    nonzero_phase_reference_error = _rel_l2(torch.from_numpy(cpu_x), output_x.tensor.detach().cpu()[0, 0])
    if nonzero_phase_reference_error > 1.0e-10:
        raise RuntimeError(
            f"independent nonzero-phase CPU reference failed: relative error={nonzero_phase_reference_error:.6g}"
        )
    wrong_phase_error = _rel_l2(output.tensor[0, 0], output_x.tensor[0, 0])
    if wrong_phase_error <= 1.0e-4:
        raise RuntimeError("ignored-phase negative control was not rejected")
    c64_runtime = _make_runtime(grid, torch.float32, device)
    c64_entrance = _encoder_entrance(c64_runtime, torch.float32, device)
    c64_output = _run_encoder(c64_runtime, c64_entrance, phases["zero"].to(device=device, dtype=torch.float32), "ols_g3_preflight_c64")
    c64_lp = _mode_coefficients(c64_output, c64_runtime, "lp01")
    c64_gaussian = _mode_coefficients(c64_output, c64_runtime, "gaussian_s12")
    if not torch.isfinite(c64_lp).all() or not torch.isfinite(c64_gaussian).all():
        raise RuntimeError("production complex64 projection preflight returned nonfinite values")
    wrong_roi = torch.roll(output.tensor[0, 0], shifts=1, dims=0)
    crop_error = _rel_l2(output.tensor[0, 0], wrong_roi)
    if crop_error <= 1.0e-4:
        raise RuntimeError("one-pixel crop-shift negative control was not rejected")
    area = 2.0e-6**2
    area_omission_ratio = 1.0 / area
    if not math.isclose(area_omission_ratio, 2.5e11, rel_tol=1.0e-12):
        raise RuntimeError("pixel-area omission negative control is inconsistent")
    adjoint_check = _adjoint_preflight(runtime, entrance, phases, config)
    if not adjoint_check["passed"]:
        raise RuntimeError("complex adjoint and direct forward preflight failed")
    return {
        "status": "PASS",
        "production_forward": "two real finite-crop encoder layers, complex128",
        "input_power_w": _field_power(entrance.tensor, entrance.plane),
        "cpu_numpy_reference_relative_error": cpu_error,
        "nonzero_phase_cpu_numpy_reference_relative_error": nonzero_phase_reference_error,
        "ignored_phase_negative_control_relative_error": wrong_phase_error,
        "ignored_phase_rejected": wrong_phase_error > 1.0e-4,
        "lp01_projection_finite_100_channels": True,
        "gaussian_s12_projection_finite_100_channels": True,
        "production_complex64_projection_finite_100_channels": True,
        "gaussian_complete_projection_reemission_power_gain_bound": gaussian_passivity["complete_operator_power_gain_bound"],
        "gaussian_mode_norm_max_absolute_error": gaussian_passivity["mode_norm_max_absolute_error"],
        "lp01_disjoint_cell_mode_norm_max_absolute_error": max(lp_norm_errors),
        "mode_passivity_checks_pass": True,
        "analytic_plane_wave": _analytic_plane_wave_check(device),
        "wrong_common_roi_crop_negative_control_relative_error": crop_error,
        "wrong_crop_rejected": crop_error > 1.0e-4,
        "omitted_pixel_area_power_ratio": area_omission_ratio,
        "complex_adjoint_direct_forward": adjoint_check,
        "saved_phase_source": str(phase_path),
        "loaded_source_modules": {
            "diagnostic_runner": str(Path(__file__).resolve()),
            "lightgen_physical_runtime": str(Path(inspect.getfile(LightGen2DSemanticGenerationInputOLSAlignment)).resolve()),
            "gaussian_s12_ols": str(Path(inspect.getfile(GaussianFiberArrayOLS)).resolve()),
            "lp01_fiber_operator": str(Path(inspect.getfile(FiberArrayOLS)).resolve()),
            "finite_crop_phase_stack": str(Path(inspect.getfile(FiniteCropPhaseStack)).resolve()),
            "angular_spectrum_propagation": str(Path(inspect.getfile(angular_spectrum_propagate)).resolve()),
        },
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "phase_optimizer_updates": 0,
    }


def _run_g3(config: dict[str, Any], phase_path: Path, device: torch.device, run_dir: Path) -> dict[str, Any]:
    saved_phases = torch.load(phase_path, map_location="cpu", weights_only=True)
    grid_sizes = tuple(int(value) for value in config["optics"]["propagation_grid_sizes"])
    roi_size = grid_sizes[0]
    fields: dict[str, dict[str, dict[int, torch.Tensor]]] = {"complex64": {}, "complex128": {}}
    coefficients: dict[str, dict[str, dict[int, torch.Tensor]]] = {"complex64": {}, "complex128": {}}
    power_records: dict[str, dict[str, dict[int, dict[str, Any]]]] = {"complex64": {}, "complex128": {}}
    boundary_records: dict[str, dict[int, float]] = {"complex64": {}, "complex128": {}}
    numpy_reference: dict[str, float] = {}
    raw_payload: dict[str, Any] = {"physical_roi_extent_m": [4.096e-3, 4.096e-3], "pixel_pitch_m": 2.0e-6}
    for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
        for name in PHASE_NAMES:
            fields[precision][name] = {}
            coefficients[precision][name] = {}
            power_records[precision][name] = {}
            boundary_records[precision][name] = {}
            for grid in grid_sizes:
                clear_production_transfer_cache()
                runtime = _make_runtime(grid, real_dtype, device)
                entrance = _encoder_entrance(runtime, real_dtype, device)
                phase = saved_phases[name].to(device=device, dtype=real_dtype)
                output = _run_encoder(runtime, entrance, phase, f"ols_g3_{precision}_{name}_{grid}")
                field = output.tensor.detach()[0, 0]
                offset = (grid - roi_size) // 2
                roi_native = field[offset : offset + roi_size, offset : offset + roi_size].cpu()
                roi = roi_native.to(torch.complex128)
                fields[precision][name][grid] = roi_native
                roi_field = ComplexField(
                    roi[None, None],
                    PlaneSpec(
                        f"common_roi_{grid}", (roi_size, roi_size), (2.0e-6, 2.0e-6), 532.0e-9,
                        runtime.ols_plane.z_m, "g3_common_physical_roi",
                    ),
                    OperationProvenance("g3_common_roi_readout", runtime.ols_plane.plane_id, None),
                )
                # Rebuild the LP01 global-plane projector for the common ROI so its
                # physical 125 um centers are invariant to the enclosing window.
                roi_real_dtype = torch.float32 if precision == "complex64" else torch.float64
                roi_runtime = _make_runtime(roi_size, roi_real_dtype, device)
                roi_project_field = ComplexField(roi_native[None, None].to(device), roi_runtime.ols_plane,
                    OperationProvenance("g3_fixed_geometry_projection", roi_runtime.ols_plane.plane_id, None))
                lp = _mode_coefficients(roi_project_field, roi_runtime, "lp01")[0].detach().cpu().to(torch.complex128)
                gaussian = _mode_coefficients(roi_project_field, roi_runtime, "gaussian_s12")[0].detach().cpu().to(torch.complex128)
                coefficients[precision][name][grid] = torch.stack((gaussian, lp), dim=0)
                full_power = _field_power(field, output.plane)
                roi_power = _field_power(roi, roi_field.plane)
                edge = 10
                edge_power = (
                    field[:edge].abs().square().sum() + field[-edge:].abs().square().sum()
                    + field[edge:-edge, :edge].abs().square().sum()
                    + field[edge:-edge, -edge:].abs().square().sum()
                ) * math.prod(output.plane.pixel_pitch_m)
                boundary_records[precision][name][grid] = float(edge_power.to(torch.float64) / max(full_power, EPS))
                power_records[precision][name][grid] = {
                    "input_power_at_encoder_entrance_w": _field_power(entrance.tensor, entrance.plane),
                    "power_before_fiber_w": full_power,
                    "common_roi_power_w": roi_power,
                    "common_roi_retained_fraction": roi_power / max(full_power, EPS),
                    "window_width_m": float(grid * 2.0e-6),
                    "frequency_spacing_cycles_per_m": 1.0 / (grid * 2.0e-6),
                    "common_roi_crop_indices": [offset, offset + roi_size, offset, offset + roi_size],
                }
                if precision == "complex128" and grid == grid_sizes[0]:
                    cpu_field = _numpy_phase_stack(
                        entrance.tensor.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False),
                        phase.detach().cpu().numpy()[:, 0].astype(np.float64, copy=False),
                        grid,
                        wavelength_m=5.32e-7,
                        dx_m=2.0e-6,
                        distance_m=1.5e-3,
                        layer_power_transmission=0.889,
                    )
                    numpy_reference[name] = _rel_l2(torch.from_numpy(cpu_field), field.cpu())
                    if numpy_reference[name] > 1.0e-10:
                        raise RuntimeError(f"CPU reference mismatch for {name}: {numpy_reference[name]:.6g}")
                del runtime, roi_runtime, output, field, entrance, roi_project_field, roi_field
                clear_production_transfer_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    for name in PHASE_NAMES:
        raw_payload[name] = {
            precision: {str(grid): fields[precision][name][grid] for grid in grid_sizes}
            for precision in ("complex64", "complex128")
        }
    torch.save(raw_payload, run_dir / "g3_common_roi_fields.pt")

    error_attribution = _g3_error_attribution(config, saved_phases, fields, device, roi_size)

    comparisons: dict[str, Any] = {}
    for precision in ("complex64", "complex128"):
        comparisons[precision] = {}
        for name in PHASE_NAMES:
            comparisons[precision][name] = {}
            for left_index in range(len(grid_sizes)):
                for right_index in range(left_index + 1, len(grid_sizes)):
                    small, large = grid_sizes[left_index], grid_sizes[right_index]
                    reference = fields[precision][name][small]
                    test = fields[precision][name][large]
                    phi, aligned = _phase_alignment(reference, test)
                    int_ref, int_test = reference.abs().square(), test.abs().square()
                    p_ref = float(int_ref.sum() * 2.0e-6**2)
                    p_test = float(int_test.sum() * 2.0e-6**2)
                    record: dict[str, Any] = {
                        "reference_grid": small,
                        "test_grid": large,
                        "common_roi_shape": [roi_size, roi_size],
                        "common_roi_extent_m": [4.096e-3, 4.096e-3],
                        "pixel_pitch_m": 2.0e-6,
                        "global_phase_rad": phi,
                        "raw_complex_relative_error": _rel_l2(reference, test),
                        "complex_relative_error_after_phase_alignment": _rel_l2(reference, aligned),
                        "intensity_relative_error": _rel_l2(int_ref, int_test),
                        "common_roi_power_relative_error": abs(p_test - p_ref) / max(abs(p_ref), EPS),
                        "power_reference_w": p_ref,
                        "power_test_w": p_test,
                    }
                    fiber_errors: dict[str, Any] = {}
                    for mode_index, mode_name in enumerate(("gaussian_s12", "lp01")):
                        coeff_ref = coefficients[precision][name][small][mode_index]
                        coeff_test = coefficients[precision][name][large][mode_index]
                        latent_phi, aligned_coeff_test = _phase_alignment(coeff_ref, coeff_test)
                        power_ref, power_test = coeff_ref.abs().square(), coeff_test.abs().square()
                        max_ref_power = float(power_ref.max())
                        floor = max(max_ref_power * 1.0e-12, EPS)
                        fiber_errors[mode_name] = {
                            "complex_latent_raw_relative_error": _rel_l2(coeff_ref, coeff_test),
                            "complex_latent_common_phase_aligned_relative_error": _rel_l2(coeff_ref, aligned_coeff_test),
                            "complex_latent_common_global_phase_rad": latent_phi,
                            "total_coupling_power_reference_w": float(power_ref.sum()),
                            "total_coupling_power_test_w": float(power_test.sum()),
                            "total_coupling_relative_error": abs(float(power_test.sum() - power_ref.sum())) / max(float(power_ref.sum()), EPS),
                            "per_fiber_power_relative_l2_error": _rel_l2(power_ref, power_test),
                            "per_fiber_power_absolute_error_w": (power_test - power_ref).tolist(),
                            "per_fiber_power_relative_error": ((power_test - power_ref).abs() / power_ref.clamp_min(floor)).tolist(),
                            "per_fiber_relative_denominator_floor_w": floor,
                        }
                    record["fiber_modes"] = fiber_errors
                    comparisons[precision][name][f"{small}_vs_{large}"] = record
    c64_aligned = [
        item["complex_relative_error_after_phase_alignment"]
        for phase in comparisons["complex64"].values()
        for item in phase.values()
    ]
    c128_aligned = [
        item["complex_relative_error_after_phase_alignment"]
        for phase in comparisons["complex128"].values()
        for item in phase.values()
    ]
    tol = float(config["gates"]["g3_complex_relative_error_after_common_global_phase_max"])
    production_pass = max(c64_aligned, default=float("inf")) <= tol
    high_precision_pass = max(c128_aligned, default=float("inf")) <= tol
    return {
        "status": "PASS" if production_pass else "FAIL",
        "production_dtype": "complex64",
        "high_precision_diagnostic_dtype": "complex128",
        "tolerance": tol,
        "production_dtype_pass": production_pass,
        "high_precision_diagnostic_pass": high_precision_pass,
        "window_grid_sizes": list(grid_sizes),
        "window_widths_m": [grid * 2.0e-6 for grid in grid_sizes],
        "common_roi_shape": [roi_size, roi_size],
        "complex128_cpu_numpy_reference_relative_error": numpy_reference,
        "boundary_20um_power_fraction": boundary_records,
        "per_window_power": power_records,
        "comparisons": comparisons,
        "error_attribution": error_attribution,
        "boundary_wraparound_is_not_inferred_from_small_edge_energy": True,
    }


def _g3_error_attribution(
    config: dict[str, Any],
    saved_phases: dict[str, torch.Tensor],
    final_fields: dict[str, dict[str, dict[int, torch.Tensor]]],
    device: torch.device,
    roi_size: int,
) -> dict[str, Any]:
    grid_sizes = tuple(int(value) for value in config["optics"]["propagation_grid_sizes"])
    dx = float(config["optics"]["pixel_pitch_m"])
    wavelength = float(config["optics"]["wavelength_m"])
    spacing = float(config["optics"]["inter_encoder_spacing_m"])
    transmission = float(config["optics"]["layer_power_transmission"])
    stage_fields: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]] = {
        "complex64": {},
        "complex128": {},
    }
    kernel_records: dict[str, Any] = {}
    fft_records: dict[str, Any] = {"complex64": {}}
    coordinate_records: dict[str, Any] = {}

    for grid in grid_sizes:
        transfer = _transfer_function(
            (grid, grid),
            pixel_pitch_m=(dx, dx),
            wavelength_m=wavelength,
            distance_m=spacing,
            device=device,
            real_dtype=torch.float32,
        )
        transfer_cpu = transfer.detach().cpu().numpy()
        fx64 = np.fft.fftfreq(grid, d=dx)
        fy64 = np.fft.fftfreq(grid, d=dx)
        fx32 = fx64.astype(np.float32)
        fy32 = fy64.astype(np.float32)
        reference_error_sq = 0.0
        legacy_error_sq = 0.0
        reference_norm_sq = 0.0
        for row_start in range(0, grid, 128):
            row_stop = min(row_start + 128, grid)
            argument64 = 1.0 - (wavelength * fx64[None, :]) ** 2 - (wavelength * fy64[row_start:row_stop, None]) ** 2
            phase64 = (2.0 * math.pi * spacing / wavelength) * np.sqrt(np.maximum(argument64, 0.0))
            reference64 = (np.exp(1j * phase64) * (argument64 >= 0.0)).astype(np.complex64)
            actual_block = transfer_cpu[row_start:row_stop]
            reference_error_sq += float(np.square(np.abs(actual_block - reference64), dtype=np.float64).sum())
            reference_norm_sq += float(np.square(np.abs(reference64), dtype=np.float64).sum())

            argument32 = (
                np.float32(1.0)
                - (np.float32(wavelength) * fx32[None, :]) ** 2
                - (np.float32(wavelength) * fy32[row_start:row_stop, None]) ** 2
            )
            phase32 = np.float32(2.0 * math.pi * spacing / wavelength) * np.sqrt(
                np.maximum(argument32, np.float32(0.0))
            )
            legacy32 = (np.exp(1j * phase32.astype(np.float64)) * (argument32 >= 0.0)).astype(np.complex64)
            legacy_error_sq += float(np.square(np.abs(actual_block - legacy32), dtype=np.float64).sum())
        kernel_records[str(grid)] = {
            "production_complex64_vs_independent_numpy_complex128_cast_relative_error": math.sqrt(
                reference_error_sq / max(reference_norm_sq, EPS)
            ),
            "legacy_float32_phase_kernel_vs_independent_numpy_complex128_cast_relative_error": math.sqrt(
                legacy_error_sq / max(reference_norm_sq, EPS)
            ),
            "legacy_kernel_is_negative_control": True,
            "frequency_spacing_cycles_per_m": 1.0 / (grid * dx),
            "sampled_nyquist_cycles_per_m": 1.0 / (2.0 * dx),
            "maximum_propagating_radial_frequency_cycles_per_m": 1.0 / wavelength,
        }
        offset = (grid - roi_size) // 2
        coordinates = {
            "grid_size": grid,
            "grid_extent_m": grid * dx,
            "roi_indices_yx": [offset, offset + roi_size, offset, offset + roi_size],
            "roi_first_pixel_center_xy_m": [
                (offset + 0.5) * dx - grid * dx / 2.0,
                (offset + 0.5) * dx - grid * dx / 2.0,
            ],
            "roi_last_pixel_center_xy_m": [
                (offset + roi_size - 0.5) * dx - grid * dx / 2.0,
                (offset + roi_size - 0.5) * dx - grid * dx / 2.0,
            ],
        }
        coordinate_records[str(grid)] = coordinates

        for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
            for phase_name in PHASE_NAMES:
                runtime = _make_runtime(grid, real_dtype, device)
                entrance = _encoder_entrance(runtime, real_dtype, device)
                phase = saved_phases[phase_name].to(device=device, dtype=real_dtype)
                first_mask = torch.polar(
                    torch.full_like(phase[0, 0], math.sqrt(transmission)), phase[0, 0]
                ).to(entrance.tensor.dtype)
                first_layer_active = entrance.tensor[0, 0] * first_mask
                padding = (grid - 100) // 2
                padded_tensor = torch.nn.functional.pad(first_layer_active, (padding, padding, padding, padding))
                padded_plane = entrance.plane.with_plane(
                    f"g3_stage1_pad_{precision}_{phase_name}_{grid}",
                    shape=(grid, grid),
                    coordinate_frame=f"g3_stage1_window_{grid}",
                )
                padded = ComplexField(
                    padded_tensor[None, None],
                    padded_plane,
                    OperationProvenance("g3_stage1_production_pad", entrance.plane.plane_id, None),
                )
                clear_production_transfer_cache()
                first_propagated = angular_spectrum_propagate(
                    padded, spacing, f"g3_stage1_propagated_{precision}_{phase_name}_{grid}"
                )
                roi_offset = (grid - roi_size) // 2
                global_roi = first_propagated.tensor[
                    0, 0, roi_offset : roi_offset + roi_size, roi_offset : roi_offset + roi_size
                ].detach().cpu()
                aperture_offset = (grid - 100) // 2
                aperture_crop = first_propagated.tensor[
                    0, 0, aperture_offset : aperture_offset + 100, aperture_offset : aperture_offset + 100
                ].detach().cpu()
                stage_fields[precision].setdefault(phase_name, {})[grid] = {
                    "first_propagation_common_roi": global_roi,
                    "intermediate_200um_aperture_crop": aperture_crop,
                    "second_propagation_common_roi": final_fields[precision][phase_name][grid],
                }
                if precision == "complex64":
                    high_precision_same_kernel = torch.fft.ifft2(
                        torch.fft.fft2(padded_tensor.to(torch.complex128), norm="ortho")
                        * _transfer_function(
                            (grid, grid),
                            pixel_pitch_m=(dx, dx),
                            wavelength_m=wavelength,
                            distance_m=spacing,
                            device=device,
                            real_dtype=torch.float32,
                        ).to(torch.complex128),
                        norm="ortho",
                    )
                    low_precision = first_propagated.tensor[0, 0]
                    high_precision_same_kernel = high_precision_same_kernel.detach()
                    fft_records[precision].setdefault(phase_name, {})[str(grid)] = {
                        "same_complex64_kernel_cast_to_both_dtypes_relative_field_error": _rel_l2(
                            low_precision, high_precision_same_kernel
                        ),
                        "same_kernel_fft_accumulation_control": True,
                    }
                    del high_precision_same_kernel, low_precision
                del runtime, entrance, first_mask, first_layer_active, padded_tensor, padded, first_propagated
                clear_production_transfer_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    stage_comparisons: dict[str, Any] = {"complex64": {}, "complex128": {}}
    for precision in ("complex64", "complex128"):
        for phase_name in PHASE_NAMES:
            stage_comparisons[precision][phase_name] = {}
            for left_index, reference_grid in enumerate(grid_sizes):
                for test_grid in grid_sizes[left_index + 1 :]:
                    pair = f"{reference_grid}_vs_{test_grid}"
                    stage_comparisons[precision][phase_name][pair] = {}
                    for stage_name in (
                        "first_propagation_common_roi",
                        "intermediate_200um_aperture_crop",
                        "second_propagation_common_roi",
                    ):
                        reference = stage_fields[precision][phase_name][reference_grid][stage_name]
                        test = stage_fields[precision][phase_name][test_grid][stage_name]
                        phase_shift, aligned = _phase_alignment(reference, test)
                        reference_power = float(reference.abs().square().sum() * dx**2)
                        test_power = float(test.abs().square().sum() * dx**2)
                        stage_comparisons[precision][phase_name][pair][stage_name] = {
                            "raw_complex_relative_error": _rel_l2(reference, test),
                            "global_phase_aligned_complex_relative_error": _rel_l2(reference, aligned),
                            "shared_global_phase_rad": phase_shift,
                            "intensity_relative_error": _rel_l2(reference.abs().square(), test.abs().square()),
                            "power_relative_error": abs(test_power - reference_power) / max(abs(reference_power), EPS),
                            "power_reference_w": reference_power,
                            "power_test_w": test_power,
                        }
    roi_coordinates = [coordinate_records[str(grid)] for grid in grid_sizes]
    first_coordinate = roi_coordinates[0]["roi_first_pixel_center_xy_m"]
    coordinate_error = max(
        abs(item["roi_first_pixel_center_xy_m"][axis] - first_coordinate[axis])
        for item in roi_coordinates
        for axis in (0, 1)
    )
    return {
        "kernel_precision": kernel_records,
        "same_kernel_fft_precision": fft_records,
        "stagewise_window_comparisons": stage_comparisons,
        "coordinate_and_crop_audit": {
            "windows": coordinate_records,
            "maximum_common_roi_first_pixel_coordinate_difference_m": coordinate_error,
            "common_roi_coordinates_match": coordinate_error == 0.0,
            "crop_is_indexed_in_physical_pixel_coordinates": True,
        },
        "interpretation_limits": {
            "cpu_kernel_reference_is_independent_numpy_same_sampled_frequency_grid": True,
            "small_edge_energy_does_not_prove_absence_of_periodic_boundary_effects": True,
            "frequency_spacing_changes_with_window_at_fixed_dx": True,
        },
    }


def _array_coordinates(size: int, pitch_m: float, device: torch.device) -> torch.Tensor:
    return (torch.arange(size, dtype=torch.float64, device=device) - (size - 1.0) / 2.0) * pitch_m


def _mode_patch(
    mode: str,
    center_y: float,
    center_x: float,
    plane: PlaneSpec,
    v_number: float,
    waist_m: float,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    device = torch.device("cuda")
    height, width = plane.shape
    dy, dx = plane.pixel_pitch_m
    y = (torch.arange(height, dtype=torch.float64, device=device) + 0.5) * dy - plane.fov_m[0] / 2.0 + plane.origin_m[0]
    x = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) * dx - plane.fov_m[1] / 2.0 + plane.origin_m[1]
    cell = 125.0e-6
    y0 = int(torch.searchsorted(y, torch.tensor(center_y - cell / 2.0, device=device, dtype=torch.float64)))
    y1 = int(torch.searchsorted(y, torch.tensor(center_y + cell / 2.0, device=device, dtype=torch.float64)))
    x0 = int(torch.searchsorted(x, torch.tensor(center_x - cell / 2.0, device=device, dtype=torch.float64)))
    x1 = int(torch.searchsorted(x, torch.tensor(center_x + cell / 2.0, device=device, dtype=torch.float64)))
    if not (y0 < y1 and x0 < x1):
        raise ValueError("fiber support is unresolved on the global plane")
    yy, xx = torch.meshgrid(y[y0:y1], x[x0:x1], indexing="ij")
    radius = torch.sqrt((yy - center_y).square() + (xx - center_x).square())
    if mode != "lp01":
        raise ValueError(mode)
    _, _, rho_e = lp01_eigenpair(v_number)
    core_radius = waist_m / rho_e
    profile = lp01_radial_profile(radius / core_radius, v_number).to(torch.float64)
    norm = torch.sqrt(profile.square().sum() * dy * dx)
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise ValueError("LP01 mode normalization failed")
    return profile / norm, (y0, y1, x0, x1)


def _project_variable(field: torch.Tensor, plane: PlaneSpec, coords: torch.Tensor, mode: str, waist_m: float, v_number: float) -> torch.Tensor:
    if mode == "gaussian_s12":
        return _gaussian_project_many(field, plane, int(coords.numel()))
    values: list[torch.Tensor] = []
    area = math.prod(plane.pixel_pitch_m)
    for cy in coords:
        for cx in coords:
            profile, (y0, y1, x0, x1) = _mode_patch(mode, float(cy), float(cx), plane, v_number, waist_m)
            coefficient = (profile.conj() * field[y0:y1, x0:x1]).sum() * area
            values.append(coefficient)
    return torch.stack(values)


def _adjoint_transfer(field: torch.Tensor, plane: PlaneSpec, distance_m: float) -> torch.Tensor:
    transfer = _transfer_function(
        plane.shape,
        pixel_pitch_m=plane.pixel_pitch_m,
        wavelength_m=plane.wavelength_m,
        distance_m=distance_m,
        device=field.device,
        real_dtype=field.real.dtype,
    )
    return torch.fft.ifft2(torch.fft.fft2(field, norm="ortho") * transfer.conj(), norm="ortho")


def _adjoint_row(
    output_mode: torch.Tensor,
    bounds: tuple[int, int, int, int],
    plane: PlaneSpec,
    entrance: torch.Tensor,
    phases: torch.Tensor,
    *,
    layer_spacing_m: float,
    last_distance_m: float,
    transmission: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = plane.shape
    del bounds
    active_start_y = (height - 100) // 2
    active_start_x = (width - 100) // 2
    back2 = _adjoint_transfer(output_mode, plane, last_distance_m)
    crop2 = back2[
        active_start_y : active_start_y + 100,
        active_start_x : active_start_x + 100,
    ] * torch.polar(torch.full_like(phases[1, 0], math.sqrt(transmission)), phases[1, 0]).conj()
    padded = torch.zeros((height, width), dtype=output_mode.dtype, device=output_mode.device)
    padded[
        active_start_y : active_start_y + 100,
        active_start_x : active_start_x + 100,
    ] = crop2
    back1 = _adjoint_transfer(padded, plane, layer_spacing_m)
    active = back1[
        active_start_y : active_start_y + 100,
        active_start_x : active_start_x + 100,
    ] * torch.polar(torch.full_like(phases[0, 0], math.sqrt(transmission)), phases[0, 0]).conj()
    pixel_area = math.prod(plane.pixel_pitch_m)
    root_pixel_area = math.sqrt(pixel_area)
    matrix_k = active.conj().reshape(-1) * root_pixel_area
    matrix_h = matrix_k * entrance.reshape(-1) * root_pixel_area
    return matrix_k, matrix_h


def _adjoint_preflight(runtime, entrance: ComplexField, saved_phases: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, Any]:
    plane = runtime.ols_plane
    device = entrance.tensor.device
    n = plane.shape[0]
    axis = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * 2.0e-6 - n * 2.0e-6 / 2.0
    amplitude_1d = (2.0 / (math.pi * 7.111e-6**2)) ** 0.25
    mode_1d = amplitude_1d * torch.exp(-(axis / 7.111e-6).square())
    output_mode = torch.outer(mode_1d, mode_1d).to(torch.complex128)
    zero_probe = torch.zeros((10000,), dtype=torch.float64, device=device)
    zero_probe[5050] = 1.0
    random_values = np.random.default_rng(20261014).choice(np.asarray([0.25, 0.75]), size=10000)
    random_probe = torch.as_tensor(random_values, dtype=torch.float64, device=device)
    patterns = {"center_pixel_impulse": zero_probe, "fixed_binary_amplitude_probe": random_probe}
    pixel_area = math.prod(plane.pixel_pitch_m)
    max_k_error = 0.0
    max_h_error = 0.0
    comparisons: dict[str, Any] = {}
    for phase_name in ("zero", "last_layer_x_1deg"):
        phase = saved_phases[phase_name].to(device=device, dtype=torch.float64)
        matrix_k, matrix_h = _adjoint_row(
            output_mode,
            (0, n, 0, n),
            plane,
            entrance.tensor[0, 0].to(torch.complex128),
            phase,
            layer_spacing_m=float(config["optics"]["inter_encoder_spacing_m"]),
            last_distance_m=float(config["optics"]["last_encoder_to_fiber_m"]),
            transmission=float(config["optics"]["layer_power_transmission"]),
        )
        comparisons[phase_name] = {}
        for probe_name, command in patterns.items():
            input_tensor = entrance.tensor * command.reshape(100, 100).to(entrance.tensor.real.dtype)[None, None]
            direct_input = ComplexField(
                input_tensor,
                entrance.plane,
                OperationProvenance("adjoint_preflight_command", entrance.plane.plane_id, None),
            )
            direct_output = _run_encoder(runtime, direct_input, phase, f"ols_adjoint_{phase_name}_{probe_name}")
            observed = _gaussian_project_many(direct_output.tensor[0, 0], plane, 1).reshape(-1)[0]
            area_orthonormal_coefficients = (
                entrance.tensor[0, 0].reshape(-1).to(torch.complex128)
                * math.sqrt(pixel_area)
                * command.to(torch.complex128)
            )
            predicted_k = matrix_k @ area_orthonormal_coefficients
            predicted_h = matrix_h @ command.to(torch.complex128)
            denominator = observed.abs().clamp_min(EPS)
            error_k = float((predicted_k - observed).abs() / denominator)
            error_h = float((predicted_h - observed).abs() / denominator)
            max_k_error = max(max_k_error, error_k)
            max_h_error = max(max_h_error, error_h)
            comparisons[phase_name][probe_name] = {
                "K_to_direct_relative_error": error_k,
                "H_to_direct_relative_error": error_h,
                "direct_coefficient_sqrt_w": [float(observed.real), float(observed.imag)],
                "predicted_K_coefficient_sqrt_w": [float(predicted_k.real), float(predicted_k.imag)],
                "predicted_H_coefficient_sqrt_w": [float(predicted_h.real), float(predicted_h.imag)],
            }
    zero_observed = complex(*comparisons["zero"]["center_pixel_impulse"]["direct_coefficient_sqrt_w"])
    legacy_zero_response = 0.0  # The old implementation multiplied by both zero phase angles.
    legacy_error = abs(zero_observed - legacy_zero_response) / max(abs(zero_observed), EPS)
    return {
        "passed": max(max_k_error, max_h_error) <= 1.0e-4 and legacy_error > 0.99,
        "maximum_K_to_direct_relative_error": max_k_error,
        "maximum_H_to_direct_relative_error": max_h_error,
        "legacy_zero_phase_negative_control": {
            "direct_response_magnitude_sqrt_w": abs(zero_observed),
            "old_angle_product_response_magnitude_sqrt_w": legacy_zero_response,
            "relative_error": legacy_error,
            "rejected": legacy_error > 0.99,
        },
        "fixed_phase_cases": list(comparisons),
        "probes_per_phase": list(patterns),
        "input_basis": "area-orthonormal pixel basis; q=1/sqrt(pixel_area)",
        "matrix_definitions": "K dimensionless maps sqrt(W) orthonormal amplitudes to sqrt(W) fiber coefficients; H=K diag(E_source sqrt(pixel_area)) maps dimensionless commands to sqrt(W)",
        "comparisons": comparisons,
    }


def _response_matrix(
    runtime,
    entrance: ComplexField,
    phases: torch.Tensor,
    size: int,
    mode: str,
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    coords = _array_coordinates(size, 125.0e-6, entrance.tensor.device)
    rows_k: list[torch.Tensor] = []
    rows_h: list[torch.Tensor] = []
    plane = runtime.ols_plane
    n = plane.shape[0]
    axis = (torch.arange(n, dtype=torch.float64, device=entrance.tensor.device) + 0.5) * 2.0e-6 - n * 2.0e-6 / 2.0
    gaussian_amplitude_1d = (2.0 / (math.pi * (7.111e-6**2))) ** 0.25
    for cy in coords:
        for cx in coords:
            if mode == "gaussian_s12":
                mode_y = gaussian_amplitude_1d * torch.exp(-((axis - cy) / 7.111e-6).square())
                mode_x = gaussian_amplitude_1d * torch.exp(-((axis - cx) / 7.111e-6).square())
                output_mode = torch.outer(mode_y, mode_x).to(torch.complex128)
                bounds = (0, n, 0, n)
            else:
                profile, bounds = _mode_patch(mode, float(cy), float(cx), plane, 2.2, 7.111e-6)
                output_mode = torch.zeros((n, n), dtype=torch.complex128, device=entrance.tensor.device)
                y0, y1, x0, x1 = bounds
                output_mode[y0:y1, x0:x1] = profile.to(torch.complex128)
            row_k, row_h = _adjoint_row(
                output_mode,
                bounds,
                plane,
                entrance.tensor[0, 0].to(torch.complex128),
                phases.to(device=entrance.tensor.device, dtype=torch.float64),
                layer_spacing_m=float(config["optics"]["inter_encoder_spacing_m"]),
                last_distance_m=float(config["optics"]["last_encoder_to_fiber_m"]),
                transmission=float(config["optics"]["layer_power_transmission"]),
            )
            rows_k.append(row_k)
            rows_h.append(row_h)
            del output_mode, row_k, row_h
            if mode == "lp01":
                del profile
    matrix_k = torch.stack(rows_k)
    matrix_h = torch.stack(rows_h)
    thresholds = config["dimension_diagnostic"]["amplitude_matrix_relative_singular_value_thresholds"]

    def spectra(matrix: torch.Tensor) -> dict[str, Any]:
        complex_singular = torch.linalg.svdvals(matrix)
        real_amplitude_matrix = torch.cat((matrix.real, matrix.imag), dim=0)
        amplitude_singular = torch.linalg.svdvals(real_amplitude_matrix)
        max_sigma = float(amplitude_singular.max())
        return {
            "complex_singular_values": complex_singular.detach().cpu().tolist(),
            "real_amplitude_control_singular_values": amplitude_singular.detach().cpu().tolist(),
            "real_amplitude_effective_rank_by_relative_threshold": {
                str(threshold): int((amplitude_singular >= max_sigma * float(threshold)).sum())
                for threshold in thresholds
            },
            "max_real_amplitude_sigma_squared": max_sigma**2,
        }

    spectrum_k = spectra(matrix_k)
    spectrum_h = spectra(matrix_h)
    edge_indices = [index for index in range(size * size) if index // size in {0, size - 1} or index % size in {0, size - 1}]
    nominal_center = bool((coords.abs() <= torch.finfo(coords.dtype).eps).any())
    center_indices = [
        index
        for index, (cy, cx) in enumerate(
            ( (float(y), float(x)) for y in coords for x in coords )
        )
        if math.hypot(cy, cx) <= 125.0e-6
    ]
    if size == 1:
        edge_indices = []
    record = {
        "channel_count": size * size,
        "grid_size": size,
        "pitch_m": 125.0e-6,
        "mode_basis": mode,
        "fiber_centers_xy_m_row_major": [
            [float(x), float(y)] for y in coords.detach().cpu() for x in coords.detach().cpu()
        ],
        "fiber_span_m": float((size - 1) * 125.0e-6),
        "edge_to_center_power_ratio": None,
        "area_orthonormal_input_basis": {
            "basis_field_per_pixel_per_unit_amplitude_m_inverse": 1.0 / math.sqrt(math.prod(plane.pixel_pitch_m)),
            "response_matrix": "K, c = K a; a has sqrt(W) coordinates and K is dimensionless",
            "complex_sigma_squared_power_gain_unit": "dimensionless",
            "max_real_amplitude_sigma_squared_power_gain": spectrum_k["max_real_amplitude_sigma_squared"],
            **spectrum_k,
        },
        "fixed_illumination_amplitude_command": {
            "response_matrix": "H = K diag(E_source sqrt(pixel_area)); c = H t",
            "matrix_units": "sqrt(W) per unit dimensionless command",
            "complex_sigma_squared_power_gain_unit": "W per squared unit command norm",
            "max_real_amplitude_sigma_squared_power_gain_w": spectrum_h["max_real_amplitude_sigma_squared"],
            **spectrum_h,
        },
        "relative_rank_error_floor_fraction": float(config["dimension_diagnostic"]["numerical_rank_error_floor_fraction"]),
        "probe_count": int(config["dimension_diagnostic"]["probe_count"]),
        "probe_amplitudes": config["dimension_diagnostic"]["probe_amplitudes"],
        "matrix_to_direct_forward_relative_error_max": None,
        "matrix_to_direct_forward_relative_error_all": [],
        "all_probe_checks_pass": False,
        "phase_optimizer_updates": 0,
        "pixel_area_m2": math.prod(plane.pixel_pitch_m),
        "input_is_encoder_entrance_illumination_times_amplitude_command": True,
        "input_field_power_before_encoder_w": _field_power(entrance.tensor, entrance.plane),
        "edge_center_index_counts": {"edge": len(edge_indices), "center": len(center_indices), "has_center_axis_fiber": bool(nominal_center)},
    }
    return {"K_dimensionless": matrix_k.detach().cpu(), "H_sqrt_w": matrix_h.detach().cpu()}, record


def _gaussian_project_many(field: torch.Tensor, plane: PlaneSpec, size: int) -> torch.Tensor:
    device = field.device
    coords = _array_coordinates(size, 125.0e-6, device)
    dy, dx = plane.pixel_pitch_m
    y = (torch.arange(plane.shape[0], dtype=torch.float64, device=device) + 0.5) * dy - plane.fov_m[0] / 2.0 + plane.origin_m[0]
    x = (torch.arange(plane.shape[1], dtype=torch.float64, device=device) + 0.5) * dx - plane.fov_m[1] / 2.0 + plane.origin_m[1]
    amplitude_1d = (2.0 / (math.pi * (7.111e-6**2))) ** 0.25
    modes_y = amplitude_1d * torch.exp(-((y[None] - coords[:, None]) / 7.111e-6).square())
    modes_x = amplitude_1d * torch.exp(-((x[None] - coords[:, None]) / 7.111e-6).square())
    modes_y_complex = modes_y.to(torch.complex128)
    modes_x_complex = modes_x.to(torch.complex128)
    left = modes_y_complex.conj() @ field.to(torch.complex128) * dy
    return left @ modes_x_complex.conj().T * dx


def _direct_probe_fields(
    runtime,
    entrance: ComplexField,
    phase: torch.Tensor,
    patterns: torch.Tensor,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for index, pattern in enumerate(patterns):
        active = pattern.reshape(100, 100).to(device=entrance.tensor.device, dtype=entrance.tensor.real.dtype)
        direct = entrance.tensor * active[None, None]
        field = ComplexField(direct, entrance.plane, OperationProvenance("fixed_amplitude_probe", entrance.plane.plane_id, None))
        output = _run_encoder(runtime, field, phase, f"ols_probe_{index:02d}")
        outputs.append(output.tensor[0, 0].to(torch.complex128).detach())
    return outputs


def _dimension_probe_validation(
    runtime,
    entrance: ComplexField,
    phases_by_name: dict[str, torch.Tensor],
    patterns: torch.Tensor,
    matrices: dict[str, Any],
    records: dict[str, Any],
    input_power_w: float,
) -> None:
    matrix_probes = torch.cat((patterns, torch.ones((1, 10000), dtype=torch.float64)), dim=0)
    for phase_name, phase in phases_by_name.items():
        direct_fields = _direct_probe_fields(runtime, entrance, phase, matrix_probes.to(entrance.tensor.device))
        before_power = torch.tensor(
            [_field_power(field, runtime.ols_plane) for field in direct_fields],
            dtype=torch.float64,
            device=entrance.tensor.device,
        )
        command_input_power = torch.tensor(
            [
                _field_power(
                    entrance.tensor[0, 0] * pattern.reshape(100, 100).to(entrance.tensor.device),
                    entrance.plane,
                )
                for pattern in matrix_probes
            ],
            dtype=torch.float64,
            device=entrance.tensor.device,
        )
        for size in SIZES:
            coords = _array_coordinates(size, 125.0e-6, entrance.tensor.device)
            for basis in ("gaussian_s12", "lp01"):
                matrix_k = matrices[phase_name][str(size)][basis]["K_dimensionless"].to(entrance.tensor.device)
                matrix_h = matrices[phase_name][str(size)][basis]["H_sqrt_w"].to(entrance.tensor.device)
                predicted_values: list[torch.Tensor] = []
                predicted_k_values: list[torch.Tensor] = []
                observed_values: list[torch.Tensor] = []
                for pattern, direct_field in zip(matrix_probes.to(entrance.tensor.device), direct_fields, strict=True):
                    command = pattern.to(torch.complex128)
                    area_orthonormal_coefficients = (
                        entrance.tensor[0, 0].reshape(-1).to(torch.complex128)
                        * math.sqrt(math.prod(entrance.plane.pixel_pitch_m))
                        * command
                    )
                    predicted_k_values.append(matrix_k @ area_orthonormal_coefficients)
                    predicted_values.append(matrix_h @ command)
                    if basis == "gaussian_s12":
                        observed_values.append(_gaussian_project_many(direct_field, runtime.ols_plane, size).reshape(-1))
                    else:
                        observed_values.append(
                            _project_variable(direct_field, runtime.ols_plane, coords, "lp01", 7.111e-6, 2.2)
                        )
                predicted = torch.stack(predicted_values)
                predicted_k = torch.stack(predicted_k_values)
                observed = torch.stack(observed_values)
                error = _rel_l2(predicted, observed)
                error_k = _rel_l2(predicted_k, observed)
                error_k_h = _rel_l2(predicted_k, predicted)
                channel_power = predicted.abs().square().to(torch.float64)
                total_channel_power = channel_power.sum(dim=1)
                probe_cv = channel_power.std(dim=1, unbiased=False) / channel_power.mean(dim=1).clamp_min(EPS)
                neff = total_channel_power.square() / channel_power.square().sum(dim=1).clamp_min(EPS)
                mode_coords = _array_coordinates(size, 125.0e-6, entrance.tensor.device)
                centers = [(float(y), float(x)) for y in mode_coords for x in mode_coords]
                center_ids = [i for i, (y, x) in enumerate(centers) if math.hypot(y, x) <= 125.0e-6]
                edge_ids = [] if size == 1 else [i for i in range(size**2) if i // size in {0, size - 1} or i % size in {0, size - 1}]
                center_each = channel_power[:, center_ids].mean(dim=1) if center_ids else None
                edge_each = channel_power[:, edge_ids].mean(dim=1) if edge_ids else None
                utilization: dict[str, list[float]] = {}
                max_each = channel_power.max(dim=1).values
                for threshold_db in (-10, -20, -30, -40):
                    cutoff = max_each * (10.0 ** (threshold_db / 10.0))
                    utilization[str(threshold_db)] = (channel_power >= cutoff[:, None]).to(torch.float64).mean(dim=1).detach().cpu().tolist()

                def distribution(values: torch.Tensor) -> dict[str, float]:
                    return {
                        "min": float(values.min()),
                        "p10": float(torch.quantile(values, 0.1)),
                        "p50": float(torch.quantile(values, 0.5)),
                        "mean": float(values.mean()),
                        "p90": float(torch.quantile(values, 0.9)),
                        "max": float(values.max()),
                    }

                target = records[phase_name][str(size)][basis]
                target["matrix_to_direct_forward_relative_error_max"] = error
                target["K_to_direct_forward_relative_error_max"] = error_k
                target["K_to_H_relative_error_max"] = error_k_h
                target["K_to_direct_forward_relative_error_all"] = [
                    _rel_l2(p, o) for p, o in zip(predicted_k, observed, strict=True)
                ]
                target["H_to_direct_forward_relative_error_all"] = [
                    _rel_l2(p, o) for p, o in zip(predicted, observed, strict=True)
                ]
                target["K_to_H_relative_error_all"] = [
                    _rel_l2(p, h) for p, h in zip(predicted_k, predicted, strict=True)
                ]
                target["matrix_to_direct_forward_relative_error_all"] = [
                    _rel_l2(p, o) for p, o in zip(predicted, observed, strict=True)
                ]
                target["all_probe_checks_pass"] = max(error, error_k, error_k_h) <= 1.0e-4
                target["probe_power_delivery"] = {
                    "probe_count_including_one_uniform_control": int(channel_power.shape[0]),
                    "fixed_illumination_source_power_w": input_power_w,
                    "amplitude_command_levels": [0.25, 0.75],
                    "no_sample_renormalization": True,
                    "commanded_encoder_entrance_power_w": distribution(command_input_power),
                    "field_power_before_fiber_w": distribution(before_power),
                    "total_coupled_power_w": distribution(total_channel_power),
                    "eta_face": distribution(total_channel_power / before_power.clamp_min(EPS)),
                    "eta_system": distribution(total_channel_power / input_power_w),
                    "N_eff_single_field": distribution(neff),
                    "channel_coefficient_of_variation": distribution(probe_cv),
                    "per_channel_power_w_by_probe": channel_power.detach().cpu().tolist(),
                    "utilization_fraction_by_relative_threshold_db": utilization,
                    "edge_to_center_power_ratio": None if edge_each is None or center_each is None else distribution(edge_each / center_each.clamp_min(EPS)),
                    "fiber_centers_xy_m_row_major": [[float(x), float(y)] for y, x in centers],
                }
                if max(error, error_k, error_k_h) > 1.0e-4:
                    raise RuntimeError(f"response matrix failed direct forward check: {phase_name}/{size}/{basis}: {max(error, error_k, error_k_h):.6g}")
                del matrix_k, matrix_h, predicted, predicted_k, observed, predicted_values, predicted_k_values, observed_values, channel_power
        del direct_fields, before_power, command_input_power
        if entrance.tensor.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def _channel_metrics(coefficients: torch.Tensor, before_power_w: float, input_power_w: float, grid_size: int) -> dict[str, Any]:
    channel_power = coefficients.abs().square().to(torch.float64)
    total = float(channel_power.sum())
    positive = channel_power[channel_power > 0]
    maximum = float(channel_power.max()) if channel_power.numel() else 0.0
    minimum_nonzero = float(positive.min()) if positive.numel() else 0.0
    centered = (grid_size - 1.0) / 2.0
    centers = [((i - centered) * 125.0e-6, (j - centered) * 125.0e-6) for i in range(grid_size) for j in range(grid_size)]
    center_indices = [k for k, (y, x) in enumerate(centers) if math.hypot(x, y) <= 125.0e-6]
    edge_indices = [] if grid_size == 1 else [k for k in range(grid_size**2) if k // grid_size in {0, grid_size - 1} or k % grid_size in {0, grid_size - 1}]
    center_power = float(channel_power[center_indices].mean()) if center_indices else None
    edge_power = float(channel_power[edge_indices].mean()) if edge_indices else None
    values = {
        "channel_count": int(channel_power.numel()),
        "fiber_power_w_row_major": channel_power.tolist(),
        "eta_face_total": total / max(before_power_w, EPS),
        "eta_system_total": total / max(input_power_w, EPS),
        "total_coupled_power_w": total,
        "N_eff_single_field": total**2 / max(float(channel_power.square().sum()), EPS),
        "utilization_fraction_by_relative_threshold_db": {},
        "power_p10_w": float(torch.quantile(channel_power, 0.1)),
        "power_p90_w": float(torch.quantile(channel_power, 0.9)),
        "P10_over_P90": float(torch.quantile(channel_power, 0.1) / torch.quantile(channel_power, 0.9).clamp_min(EPS)),
        "coefficient_of_variation": float(channel_power.std(unbiased=False) / channel_power.mean().clamp_min(EPS)),
        "edge_to_center_power_ratio": None if center_power is None or edge_power is None or center_power <= 0.0 else edge_power / center_power,
        "minimum_channel_power_w": float(channel_power.min()),
        "minimum_nonzero_channel_power_w": minimum_nonzero,
        "maximum_channel_power_w": maximum,
        "edge_channel_count": len(edge_indices),
        "center_group_channel_count": len(center_indices),
        "edge_center_groups": "outermost row/column mean divided by fibers within one pitch of array center; absent groups reported null",
    }
    for db in (-10, -20, -30, -40):
        threshold = maximum * 10.0 ** (db / 10.0)
        values["utilization_fraction_by_relative_threshold_db"][str(db)] = float((channel_power >= threshold).sum() / channel_power.numel())
    return values


def _dimension_audit(config: dict[str, Any], phase_path: Path, device: torch.device, run_dir: Path) -> dict[str, Any]:
    saved = torch.load(phase_path, map_location="cpu", weights_only=True)
    runtime = _make_runtime(2048, torch.float64, device)
    entrance = _encoder_entrance(runtime, torch.float64, device)
    diagnostics: dict[str, Any] = {}
    all_matrices: dict[str, Any] = {}
    probe_generator = np.random.default_rng(int(config["dimension_diagnostic"]["probe_seed"]))
    pattern_numpy = probe_generator.choice(
        np.asarray(config["dimension_diagnostic"]["probe_amplitudes"], dtype=np.float64),
        size=(int(config["dimension_diagnostic"]["probe_count"]), 10000),
    )
    patterns = torch.from_numpy(pattern_numpy)
    phases_by_name = {name: saved[name].to(device=device, dtype=torch.float64) for name in PHASE_NAMES}
    flat_power_by_phase: dict[str, float] = {}
    for phase_name, phase in phases_by_name.items():
        diagnostics[phase_name] = {}
        all_matrices[phase_name] = {}
        for size in SIZES:
            diagnostics[phase_name][str(size)] = {}
            all_matrices[phase_name][str(size)] = {}
            for basis in ("gaussian_s12", "lp01"):
                matrix, record = _response_matrix(runtime, entrance, phase, size, basis, config)
                diagnostics[phase_name][str(size)][basis] = record
                all_matrices[phase_name][str(size)][basis] = matrix
                del matrix
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        active = torch.ones((1, 1, 100, 100), dtype=torch.float64, device=device)
        flat_input = ComplexField(
            entrance.tensor * active,
            entrance.plane,
            OperationProvenance("uniform_amplitude_command", entrance.plane.plane_id, None),
        )
        flat_field = _run_encoder(runtime, flat_input, phase, "ols_dimension_uniform")
        flat_power_by_phase[phase_name] = _field_power(flat_field.tensor, flat_field.plane)
        for size in SIZES:
            coords = _array_coordinates(size, 125.0e-6, device)
            gaussian_coeff = _gaussian_project_many(flat_field.tensor[0, 0], flat_field.plane, size).reshape(-1).detach().cpu()
            lp_coeff = _project_variable(flat_field.tensor[0, 0], flat_field.plane, coords, "lp01", 7.111e-6, 2.2).detach().cpu()
            diagnostics[phase_name][str(size)]["gaussian_s12"]["uniform_command_channel_metrics"] = _channel_metrics(
                gaussian_coeff, flat_power_by_phase[phase_name], float(config["optics"]["input_power_w"]), size
            )
            diagnostics[phase_name][str(size)]["lp01"]["uniform_command_channel_metrics"] = _channel_metrics(
                lp_coeff, flat_power_by_phase[phase_name], float(config["optics"]["input_power_w"]), size
            )
            for basis in ("gaussian_s12", "lp01"):
                diagnostics[phase_name][str(size)][basis]["edge_to_center_power_ratio"] = diagnostics[
                    phase_name
                ][str(size)][basis]["uniform_command_channel_metrics"]["edge_to_center_power_ratio"]
        del active, flat_input, flat_field
    _dimension_probe_validation(
        runtime,
        entrance,
        phases_by_name,
        patterns,
        all_matrices,
        diagnostics,
        float(config["optics"]["input_power_w"]),
    )
    if not all(
        record["all_probe_checks_pass"]
        for phase_record in diagnostics.values()
        for size_record in phase_record.values()
        for record in size_record.values()
    ):
        raise RuntimeError("at least one full response matrix failed direct forward validation")
    torch.save({"schema": SCHEMA, "phase_cases": all_matrices}, run_dir / "response_matrices_K_H_complex128.pt")
    return {
        "status": "COMPLETE_ZERO_UPDATE_DIAGNOSTIC",
        "phase_cases": diagnostics,
        "mode_definitions": {
            "gaussian_s12": "analytic full-plane Gaussian, w=7.111 um, no crop, discrete renormalization, or orthogonalization",
            "lp01": "V=2.2 step-index LP01, fixed MFD-derived core, disjoint 125 um Voronoi cells, discrete unit-power normalization",
            "transport": "T=I for both bases",
        },
        "historical_optimized_phase_replay": "UNAVAILABLE: prior compact dimension run did not save optimized phase screens; no replay or substitute is claimed",
        "optimizer_updates": 0,
        "matrix_payload": "response_matrices_K_H_complex128.pt",
        "complex_rank_is_not_noise_limited_experimental_dimension": True,
    }


def run(config_path: Path, phase_path: Path, run_dir: Path, *, preflight_only: bool) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    if not phase_path.is_file():
        raise FileNotFoundError(f"explicit fixed-phase source is missing: {phase_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("this measurement requires the configured AutoDL CUDA device")
    device = torch.device("cuda")
    started = time.perf_counter()
    preflight = _run_preflight(config, phase_path, device)
    _write_json(run_dir / "preflight.json", preflight)
    if preflight_only:
        return preflight
    g3 = _run_g3(config, phase_path, device, run_dir)
    dimension = None
    if g3["production_dtype_pass"]:
        dimension = _dimension_audit(config, phase_path, device, run_dir)
        status = "VALID_COMPLETED"
    else:
        status = "VALID_FAILED"
    power_interface = {
        "decision": "future training retains an explicitly calibrated absolute-power forward and records power throughout training and final results; no new loss term is authorized by this plan",
        "existing_L_op": "five-plane log power-ratio squared penalty against a frozen TRAIN-derived reference; it does not maximize fiber coupling",
        "physical_plane_power_source": "physical_plane_power computes pixel area times sum(abs(E)^2); physical watts require the field amplitude calibration to remain valid",
        "camera_interface_source": "physical_sensor_output_w_per_pixel and fixed camera_gain_w_inverse/task_sensor_output exist in the physical forward result",
        "current_loss_consumer": "BOGT reconstruction still consumes result.raw_intensity; task_sensor_output is not the current reconstruction consumer",
        "pre_training_requirement": "before future Gaussian training, validate W/pixel and task-sensor interfaces, fixed gain behavior, reference-power compatibility, and gradient flow; do not reuse LP01 TRAIN references automatically",
        "no_loss_change": True,
        "no_training_updates": 0,
    }
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    report = {
        "schema": SCHEMA,
        "status": status,
        "decision_question": "whether current evidence supports planning the next controlled OLS dimension optimization",
        "source_config": str(config_path),
        "fixed_phase_source": str(phase_path),
        "source_evidence": config["source_evidence"],
        "loaded_source_modules": {
            "diagnostic_runner": str(Path(__file__).resolve()),
            "lightgen_physical_runtime": str(Path(inspect.getfile(LightGen2DSemanticGenerationInputOLSAlignment)).resolve()),
            "gaussian_s12_ols": str(Path(inspect.getfile(GaussianFiberArrayOLS)).resolve()),
            "lp01_fiber_operator": str(Path(inspect.getfile(FiberArrayOLS)).resolve()),
            "finite_crop_phase_stack": str(Path(inspect.getfile(FiniteCropPhaseStack)).resolve()),
            "angular_spectrum_propagation": str(Path(inspect.getfile(angular_spectrum_propagate)).resolve()),
        },
        "provenance_git_head": head,
        "execution": {
            "python": __import__("sys").executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds_observation": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
        "frozen_physical_system": config["optics"],
        "fiber_assumptions": config["fiber"],
        "updates_and_forbidden_io": {
            "formal_training_updates": 0,
            "phase_optimizer_updates": 0,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "fid_calls": 0,
            "loss_changes": 0,
            "production_geometry_changes": 0,
            "material_forward_calls": 0,
        },
        "preflight": preflight,
        "G3": g3,
        "OLS_dimension": dimension if dimension is not None else {
            "status": "BLOCKED_BY_G3_FAIL",
            "reason": "the production complex64 common-ROI field convergence gate did not pass; no dimension matrix or ranking was generated",
            "historical_results_preserved": True,
        },
        "power_interface": power_interface,
        "historical_results": {
            "previous_G3": "FAIL, retained unchanged",
            "compact_dimension_phase_optimization": "historical finite-step measurements retained; optimized phase screens were not saved and cannot be replayed",
            "gaussian_adoption": "user selected Gaussian S12 as future simulation main reference with same-geometry LP01 sensitivity control",
        },
        "claim_limits": [
            "fiber coordinates and MFD remain unmeasured simulation assumptions",
            "a pass in the high-precision diagnostic alone does not establish convergence of the production complex64 path",
            "single-field N_eff is power participation, not independent response rank",
            "a linear response rank without detector noise and measurement calibration is not experimental usable dimension",
            "finite fixed-phase diagnostics do not prove a global phase-only optimum",
        ],
    }
    _write_json(run_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase-source", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.phase_source, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
