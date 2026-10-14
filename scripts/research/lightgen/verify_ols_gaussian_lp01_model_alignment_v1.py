"""Cloud-only zero-update validation of same-plane Gaussian and LP01 OLS models."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import torch
import yaml

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.fiber_operator import (
    FiberArraySpec,
    _local_lp01_mode_patches,
    lp01_eigenpair,
)
from ldm_project.v2.optics.propagation import clear_production_transfer_cache
from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
    GaussianFiberGeometry,
    PASSIVITY_TOLERANCE,
)


SCHEMA = "ols_gaussian_lp01_model_alignment.v1"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_gaussian_lp01_model_alignment_v1.yaml")
EPS = 1.0e-30


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (np.bool_, np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    expected = {
        "wavelength_m": 532.0e-9,
        "pixel_pitch_m": 2.0e-6,
        "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6,
        "global_grid_size": 2048,
        "inter_encoder_spacing_m": 1.5e-3,
        "last_encoder_to_fiber_m": 1.5e-3,
        "phase_screen_count": 2,
        "layer_power_transmission": 0.889,
    }
    for key, value in expected.items():
        if config["optics"].get(key) != value:
            raise ValueError(f"frozen optical setting changed: optics.{key}")
    expected_fiber = {
        "array_shape": [10, 10],
        "pitch_m": 125.0e-6,
        "cladding_diameter_m": 125.0e-6,
        "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": 7.111e-6,
        "lp01_v_number": 2.2,
        "transport": "ideal_identity",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }
    for key, value in expected_fiber.items():
        if config["fiber"].get(key) != value:
            raise ValueError(f"frozen fiber setting changed: fiber.{key}")
    if config["scope"] != {
        "purpose": "synthetic_physical_interface_validation",
        "task_training": "forbidden",
        "dataset_read": "forbidden",
        "task_evaluator": "forbidden",
        "fid": "forbidden",
        "task_loss_change": "forbidden",
        "formal_model_default_change": "forbidden",
        "architecture_change": "forbidden",
        "material_nonlinearity": "forbidden",
    }:
        raise ValueError("research scope contract differs from this task")
    if any(int(config["execution"][key]) != 0 for key in (
        "task_training_updates", "phase_optimizer_updates", "dataset_reads", "task_evaluator_calls"
    )):
        raise ValueError("all task and optimizer update/read counts must remain zero")
    return config


def _plane(config: dict[str, Any]) -> PlaneSpec:
    optics = config["optics"]
    return PlaneSpec(
        "gaussian_lp01_encoder_plane",
        tuple(optics["encoder_shape"]),
        (float(optics["pixel_pitch_m"]),) * 2,
        float(optics["wavelength_m"]),
        0.0,
        "encoder",
    )


def _make_phases(config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    optics = config["optics"]
    n = int(optics["encoder_shape"][0])
    dx = float(optics["pixel_pitch_m"])
    wavelength = float(optics["wavelength_m"])
    axis = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * dx - n * dx / 2.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    zeros = torch.zeros((n, n), dtype=torch.float32, device=device)
    radians = math.radians(1.0)
    k = 2.0 * math.pi / wavelength
    x_ramp = (k * math.sin(radians) * xx).to(torch.float32)
    xy_ramp = (k * math.sin(radians) * (xx + yy)).to(torch.float32)
    return {
        "zero": torch.stack((zeros, zeros)).unsqueeze(1),
        "last_layer_x_1deg": torch.stack((zeros, x_ramp)).unsqueeze(1),
        "last_layer_xy_1deg": torch.stack((zeros, xy_ramp)).unsqueeze(1),
    }


def _source_field(config: dict[str, Any], plane: PlaneSpec, device: torch.device) -> ComplexField:
    side = float(config["optics"]["encoder_aperture_m"])
    count = plane.shape[0] * plane.shape[1]
    area = math.prod(plane.pixel_pitch_m)
    amplitude = math.sqrt(float(config["optics"]["input_power_w"]) / (count * area))
    values = torch.full((1, 1, *plane.shape), amplitude, dtype=torch.complex64, device=device)
    if abs(plane.fov_m[0] - side) > 1.0e-12 or abs(plane.fov_m[1] - side) > 1.0e-12:
        raise ValueError("synthetic square illumination must match the 200 um trainable aperture")
    return ComplexField(values, plane, OperationProvenance("synthetic_coherent_uniform_200um_square", plane.plane_id, None))


def _run_phase_stack(
    source: ComplexField,
    phase: torch.Tensor,
    config: dict[str, Any],
    *,
    prefix: str,
) -> tuple[ComplexField, tuple[ComplexField, ...], dict[str, torch.Tensor], FiniteCropPhaseStack]:
    optics = config["optics"]
    stack = FiniteCropPhaseStack(
        layers=2,
        active_shape=tuple(optics["encoder_shape"]),
        propagation_shape=(int(optics["global_grid_size"]),) * 2,
        spacing_m=float(optics["inter_encoder_spacing_m"]),
        layer_power_transmission=float(optics["layer_power_transmission"]),
        preserve_global_output=True,
    ).to(source.tensor.device)
    with torch.no_grad():
        stack.phase.copy_(phase)
    before = stack.phase.detach().clone()
    output, records, diagnostics = stack(source, prefix)
    if not torch.equal(stack.phase.detach(), before):
        raise RuntimeError("phase-only optical forward changed a phase screen")
    return output, records, diagnostics, stack


def _power(values: torch.Tensor, plane: PlaneSpec) -> torch.Tensor:
    return values.abs().square().sum(dim=(-2, -1)) * math.prod(plane.pixel_pitch_m)


def _canonical_field(field: ComplexField, plane: PlaneSpec) -> ComplexField:
    physical = (
        field.plane.shape == plane.shape
        and field.plane.pixel_pitch_m == plane.pixel_pitch_m
        and field.plane.fov_m == plane.fov_m
        and field.plane.origin_m == plane.origin_m
        and field.plane.wavelength_m == plane.wavelength_m
        and field.plane.z_m == plane.z_m
        and field.plane.coordinate_frame.endswith("_propagation_grid")
    )
    if not physical:
        raise ValueError("phase cases did not produce the same physical fiber reference plane")
    return ComplexField(field.tensor, plane, field.provenance)


class SamePlaneLP01OLS:
    """LP01 input projection and re-emission using the production Voronoi cells."""

    def __init__(self, plane: PlaneSpec, config: dict[str, Any], device: torch.device) -> None:
        fiber = config["fiber"]
        spec = FiberArraySpec(
            grid_shape=(10, 10),
            pitch_m=plane.pixel_pitch_m[1],
            wavelength_m=plane.wavelength_m,
            waist_m=float(fiber["mode_field_diameter_m"]) / 2.0,
            fill=0.8,
            v_number=float(fiber["lp01_v_number"]),
            mode_model="step_index_lp01_ports",
            expected_fov_m=plane.fov_m,
            array_center_pitch_m=float(fiber["pitch_m"]),
        )
        modes, bounds = _local_lp01_mode_patches(plane, spec)
        self.plane = plane
        self.bounds = bounds
        self.modes = tuple(mode.to(device) for mode in modes)
        self.area = math.prod(plane.pixel_pitch_m)
        self.device = device

    def project(self, values: torch.Tensor) -> torch.Tensor:
        rows = []
        for mode, (y0, y1, x0, x1) in zip(self.modes, self.bounds, strict=True):
            rows.append((mode.conj() * values[:, 0, y0:y1, x0:x1]).sum(dim=(-2, -1)) * self.area)
        return torch.stack(rows, dim=1)

    def reemit(self, coefficients: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(
            (coefficients.shape[0], 1, *self.plane.shape), dtype=torch.complex64, device=self.device
        )
        for index, (mode, (y0, y1, x0, x1)) in enumerate(zip(self.modes, self.bounds, strict=True)):
            values[:, 0, y0:y1, x0:x1] += coefficients[:, index, None, None] * mode
        return values

    def gram(self) -> torch.Tensor:
        gram = torch.zeros((100, 100), dtype=torch.complex128, device=self.device)
        for i, (mi, bi) in enumerate(zip(self.modes, self.bounds, strict=True)):
            iy0, iy1, ix0, ix1 = bi
            for j in range(i, 100):
                mj = self.modes[j]
                jy0, jy1, jx0, jx1 = self.bounds[j]
                y0, y1 = max(iy0, jy0), min(iy1, jy1)
                x0, x1 = max(ix0, jx0), min(ix1, jx1)
                if y0 >= y1 or x0 >= x1:
                    continue
                left = mi[y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0].to(torch.complex128)
                right = mj[y0 - jy0:y1 - jy0, x0 - jx0:x1 - jx0].to(torch.complex128)
                value = (left.conj() * right).sum() * self.area
                gram[i, j] = value
                gram[j, i] = value.conj()
        return gram


def _fiber_centers(config: dict[str, Any]) -> list[tuple[float, float]]:
    pitch = float(config["fiber"]["pitch_m"])
    axis = [(index - 4.5) * pitch for index in range(10)]
    return [(x, y) for y in axis for x in axis]


def _gaussian_field_mode(operator: GaussianFiberArrayOLS, index: int, device: torch.device) -> torch.Tensor:
    row, col = divmod(index, 10)
    y = operator.modes_y[row].to(device=device, dtype=torch.float64)
    x = operator.modes_x[col].to(device=device, dtype=torch.float64)
    return torch.outer(y, x).to(torch.complex64)[None, None]


def _lp01_field_mode(lp: SamePlaneLP01OLS, index: int) -> torch.Tensor:
    mode = lp.modes[index]
    y0, y1, x0, x1 = lp.bounds[index]
    values = torch.zeros((1, 1, *lp.plane.shape), dtype=torch.complex64, device=lp.device)
    values[0, 0, y0:y1, x0:x1] = mode
    return values


def _channel_metrics(
    coefficients: torch.Tensor,
    p_before: float,
    p_input: float,
    *,
    output_power: float,
) -> dict[str, Any]:
    powers = coefficients[0].abs().square().to(torch.float64)
    total = float(powers.sum())
    sorted_power = powers.sort().values
    n_eff = total * total / max(float(powers.square().sum()), EPS)
    boundary = [r * 10 + c for r in range(10) for c in range(10) if r in {0, 9} or c in {0, 9}]
    central = [r * 10 + c for r in (4, 5) for c in (4, 5)]
    edge_mean = float(powers[boundary].mean())
    center_mean = float(powers[central].mean())
    return {
        "fiber_power_W_row_major": powers.cpu().tolist(),
        "total_coupled_power_W": total,
        "re_emitted_field_power_W": output_power,
        "eta_fiber_face": total / max(p_before, EPS),
        "eta_system": total / max(p_input, EPS),
        "eta_field_reemitted_over_input": output_power / max(p_input, EPS),
        "N_eff_instantaneous": n_eff,
        "N_eff_claim_limit": "single-field power participation ratio, not independent latent rank",
        "central_nearest_four_channel_indices": central,
        "edge_channel_indices": boundary,
        "edge_to_center_mean_power_ratio": edge_mean / max(center_mean, EPS),
        "center_four_channel_power_mean_W": center_mean,
        "edge_36_channel_power_mean_W": edge_mean,
        "channel_power_min_W": float(sorted_power[0]),
        "channel_power_p10_W": float(torch.quantile(powers, 0.1)),
        "channel_power_p50_W": float(torch.quantile(powers, 0.5)),
        "channel_power_p90_W": float(torch.quantile(powers, 0.9)),
        "channel_power_max_W": float(sorted_power[-1]),
    }


def _cpu_reference_gaussian(
    values: torch.Tensor,
    plane: PlaneSpec,
    centers: list[tuple[float, float]],
    waist_m: float,
) -> np.ndarray:
    source = values.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False)
    dy, dx = plane.pixel_pitch_m
    y_axis = (np.arange(plane.shape[0], dtype=np.float64) + 0.5) * dy - plane.fov_m[0] / 2 + plane.origin_m[0]
    x_axis = (np.arange(plane.shape[1], dtype=np.float64) + 0.5) * dx - plane.fov_m[1] / 2 + plane.origin_m[1]
    amp = math.sqrt(2.0 / (math.pi * waist_m**2))
    outputs = []
    radius = 12.0 * waist_m
    for cx, cy in centers:
        yi = np.flatnonzero(np.abs(y_axis - cy) <= radius)
        xi = np.flatnonzero(np.abs(x_axis - cx) <= radius)
        yy = y_axis[yi, None] - cy
        xx = x_axis[None, xi] - cx
        mode = amp * np.exp(-(yy * yy + xx * xx) / waist_m**2)
        outputs.append(np.sum(np.conj(mode) * source[np.ix_(yi, xi)], dtype=np.complex128) * dy * dx)
    return np.asarray(outputs, dtype=np.complex128)


def _cpu_reference_lp01(values: torch.Tensor, lp: SamePlaneLP01OLS) -> np.ndarray:
    source = values.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False)
    outputs = []
    for mode, (y0, y1, x0, x1) in zip(lp.modes, lp.bounds, strict=True):
        patch = mode.detach().cpu().numpy().astype(np.complex128, copy=False)
        outputs.append(np.sum(np.conj(patch) * source[y0:y1, x0:x1], dtype=np.complex128) * lp.area)
    return np.asarray(outputs, dtype=np.complex128)


def _relative_vector_error(actual: np.ndarray, reference: np.ndarray, denominator: float) -> float:
    return float(np.linalg.norm(actual - reference) / max(float(denominator), EPS))


def _tilted_square_field(
    config: dict[str, Any], plane: PlaneSpec, device: torch.device, theta_x_deg: float, theta_y_deg: float
) -> torch.Tensor:
    n = int(round(float(config["optics"]["encoder_aperture_m"]) / plane.pixel_pitch_m[0]))
    side = float(config["optics"]["encoder_aperture_m"])
    pin = float(config["optics"]["input_power_w"])
    amplitude = math.sqrt(pin) / side
    y = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * plane.pixel_pitch_m[0] - side / 2.0
    x = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * plane.pixel_pitch_m[1] - side / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    k = 2.0 * math.pi / plane.wavelength_m
    phase = k * (math.sin(math.radians(theta_x_deg)) * xx + math.sin(math.radians(theta_y_deg)) * yy)
    patch = torch.polar(torch.full_like(phase, amplitude), phase).to(torch.complex64)
    h, w = plane.shape
    y0, x0 = (h - n) // 2, (w - n) // 2
    values = torch.zeros((1, 1, h, w), dtype=torch.complex64, device=device)
    values[0, 0, y0:y0 + n, x0:x0 + n] = patch
    return values


def _mp_gaussian_tilt_reference(
    config: dict[str, Any], centers: list[tuple[float, float]], theta_x_deg: float, theta_y_deg: float
) -> np.ndarray:
    mp.mp.dps = int(config["diagnostics"]["mp_dps"])
    fiber = config["fiber"]
    optics = config["optics"]
    w = mp.mpf(str(fiber["gaussian_1e_amplitude_radius_m"]))
    side = mp.mpf(str(optics["encoder_aperture_m"]))
    pin = mp.mpf(str(optics["input_power_w"]))
    wavelength = mp.mpf(str(optics["wavelength_m"]))
    amp_mode = mp.sqrt(2 / (mp.pi * w * w))
    e0 = mp.sqrt(pin) / side
    qx = 2 * mp.pi / wavelength * mp.sin(mp.mpf(str(theta_x_deg)) * mp.pi / 180)
    qy = 2 * mp.pi / wavelength * mp.sin(mp.mpf(str(theta_y_deg)) * mp.pi / 180)
    half = side / 2

    def axis_integral(center: float, q: mp.mpf) -> mp.mpc:
        c = mp.mpf(str(center))
        qw2 = q * w / 2
        return (
            mp.sqrt(mp.pi) * w / 2
            * mp.exp(1j * q * c - qw2 * qw2)
            * (mp.erf((half - c) / w - 1j * qw2) - mp.erf((-half - c) / w - 1j * qw2))
        )

    return np.asarray([
        complex(amp_mode * e0 * axis_integral(cx, qx) * axis_integral(cy, qy))
        for cx, cy in centers
    ], dtype=np.complex128)


def _independent_lp01_constants(config: dict[str, Any]) -> dict[str, float]:
    mp.mp.dps = int(config["diagnostics"]["mp_dps"])
    v = mp.mpf(str(config["fiber"]["lp01_v_number"]))

    def characteristic(u: mp.mpf) -> mp.mpf:
        w = mp.sqrt(v * v - u * u)
        return u * mp.besselj(1, u) / mp.besselj(0, u) - w * mp.besselk(1, w) / mp.besselk(0, w)

    u = mp.findroot(characteristic, (mp.mpf("1.4"), mp.mpf("1.8")))
    w = mp.sqrt(v * v - u * u)
    boundary = mp.besselj(0, u)
    denom = mp.besselk(0, w)

    def profile(rho: mp.mpf) -> mp.mpf:
        if rho <= 1:
            return mp.besselj(0, u * rho)
        return boundary * mp.besselk(0, w * rho) / denom

    rho_e = mp.findroot(lambda rho: profile(rho) - mp.e ** -1, (mp.mpf("1.05"), mp.mpf("1.3")))
    prod_u, prod_w, prod_rho = lp01_eigenpair(float(v))
    return {
        "u_mpmath": float(u),
        "w_mpmath": float(w),
        "rho_1e_mpmath": float(rho_e),
        "u_production": prod_u,
        "w_production": prod_w,
        "rho_1e_production": prod_rho,
        "max_abs_difference": max(abs(float(u) - prod_u), abs(float(w) - prod_w), abs(float(rho_e) - prod_rho)),
    }


def _gaussian_gram(operator: GaussianFiberArrayOLS, device: torch.device) -> torch.Tensor:
    y = operator.modes_y.to(device=device, dtype=torch.float64)
    x = operator.modes_x.to(device=device, dtype=torch.float64)
    dy, dx = operator.input_plane.pixel_pitch_m
    gy = y @ y.T * dy
    gx = x @ x.T * dx
    return torch.kron(gy, gx).to(torch.complex128)


def _mode_gram_summary(gram: torch.Tensor, tol: float) -> dict[str, Any]:
    real_gram = gram.real.to(torch.float64)
    eig = torch.linalg.eigvalsh(real_gram)
    max_offdiag = (gram - torch.diag(gram.diagonal())).abs().max()
    gain = float(eig[-1]) ** 2
    return {
        "mode_norm_max_abs_error": float((gram.diagonal().real - 1.0).abs().max()),
        "max_offdiagonal_overlap": float(max_offdiag),
        "gram_min_eigenvalue": float(eig[0]),
        "gram_max_eigenvalue": float(eig[-1]),
        "complete_project_reemit_power_gain_bound": gain,
        "passive_at_relative_tolerance": gain <= 1.0 + tol,
    }


def _single_mode_checks(
    config: dict[str, Any],
    plane: PlaneSpec,
    gaussian: GaussianFiberArrayOLS,
    lp: SamePlaneLP01OLS,
    centers: list[tuple[float, float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    pin = float(config["optics"]["input_power_w"])
    selected: list[tuple[str, int]] = []
    # A 10x10 array has no port exactly at (0,0); all four nearest ports are retained.
    selected.extend(("center_nearest", r * 10 + c) for r in (4, 5) for c in (4, 5))
    selected.extend(("edge_center", index) for index in (4, 40, 49, 94))
    selected.extend(("corner", index) for index in (0, 9, 90, 99))
    results = []
    for region, index in selected:
        for name, mode_tensor, projector, emitter, cpu_fn in (
            ("gaussian", _gaussian_field_mode(gaussian, index, device), gaussian.project_coefficients,
             gaussian.reemit_coefficients, lambda values: _cpu_reference_gaussian(values, plane, centers, gaussian.geometry.waist_m)),
            ("lp01", _lp01_field_mode(lp, index), lp.project, lp.reemit, lambda values: _cpu_reference_lp01(values, lp)),
        ):
            power = _power(mode_tensor[:, 0], plane)
            scaled = mode_tensor * math.sqrt(pin / float(power[0]))
            field = ComplexField(scaled, plane, OperationProvenance(f"single_mode_{name}_{index}", plane.plane_id, 0.0))
            coeff = projector(field) if name == "gaussian" else projector(field.tensor)
            out = emitter(coeff)
            p_in = float(_power(field.tensor[:, 0], plane)[0])
            out_field = out.unsqueeze(1) if out.ndim == 3 else out
            p_out = float(_power(out_field[:, 0], plane)[0])
            p_chan = coeff[0].abs().square().to(torch.float64)
            other = float(p_chan.sum() - p_chan[index])
            reference = cpu_fn(field.tensor)
            error = _relative_vector_error(coeff[0].detach().cpu().numpy().astype(np.complex128), reference, math.sqrt(pin))
            results.append({
                "mode_model": name,
                "target_region": region,
                "target_index_row_major": index,
                "target_center_xy_m": list(centers[index]),
                "input_power_W": p_in,
                "target_coupling_efficiency_over_input": float(p_chan[index]) / max(p_in, EPS),
                "other_99_coupling_efficiency_over_input": other / max(p_in, EPS),
                "target_to_other_power_ratio": float(p_chan[index]) / max(other, EPS),
                "complex_coefficients": coeff[0].detach().cpu().tolist(),
                "re_emitted_power_W": p_out,
                "cpu_complex_coefficient_error_over_sqrt_input_power": error,
            })
    return results


def _coherent_two_port_checks(
    config: dict[str, Any], plane: PlaneSpec, gaussian: GaussianFiberArrayOLS, lp: SamePlaneLP01OLS,
    device: torch.device,
) -> list[dict[str, Any]]:
    pin = float(config["optics"]["input_power_w"])
    i, j = 44, 45
    modes_g = [_gaussian_field_mode(gaussian, index, device)[0, 0] for index in (i, j)]
    modes_l = [_lp01_field_mode(lp, index)[0, 0] for index in (i, j)]
    results = []
    for model_name, modes, project, emit, cpu_ref in (
        ("gaussian", modes_g, gaussian.project_coefficients, gaussian.reemit_coefficients,
         lambda field: _cpu_reference_gaussian(field, plane, _fiber_centers_from_plane_and_geometry(gaussian), gaussian.geometry.waist_m)),
        ("lp01", modes_l, lp.project, lp.reemit, lambda field: _cpu_reference_lp01(field, lp)),
    ):
        for delta in (0.0, math.pi / 2.0, math.pi):
            values = (modes[0] + torch.exp(torch.tensor(1j * delta, device=device)) * modes[1])[None, None]
            power = float(_power(values[:, 0], plane)[0])
            values = values * math.sqrt(pin / power)
            field = ComplexField(values, plane, OperationProvenance("coherent_two_port_superposition", plane.plane_id, 0.0))
            coeff = project(field) if model_name == "gaussian" else project(field.tensor)
            output = emit(coeff)
            output_field = output.unsqueeze(1) if output.ndim == 3 else output
            pair_phase = float(torch.angle(coeff[0, j] * coeff[0, i].conj()))
            phase_error = math.atan2(math.sin(pair_phase - delta), math.cos(pair_phase - delta))
            reference = cpu_ref(field.tensor)
            error = _relative_vector_error(coeff[0].detach().cpu().numpy().astype(np.complex128), reference, math.sqrt(pin))
            results.append({
                "mode_model": model_name,
                "ports_row_major": [i, j],
                "relative_phase_rad_requested": delta,
                "relative_phase_rad_measured": pair_phase,
                "relative_phase_error_rad": phase_error,
                "coefficient_error_over_sqrt_input_power": error,
                "input_power_W": float(_power(field.tensor[:, 0], plane)[0]),
                "re_emitted_power_W": float(_power(output_field[:, 0], plane)[0]),
                "complex_coefficients": [complex(value) for value in coeff[0].detach().cpu().tolist()],
            })
    return results


def _fiber_centers_from_plane_and_geometry(operator: GaussianFiberArrayOLS) -> list[tuple[float, float]]:
    p = operator.geometry.port_pitch_m
    axis = [(index - 4.5) * p for index in range(10)]
    return [(x, y) for y in axis for x in axis]


def _tilt_checks(
    config: dict[str, Any], plane: PlaneSpec, gaussian: GaussianFiberArrayOLS, lp: SamePlaneLP01OLS,
    centers: list[tuple[float, float]], device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pin = float(config["optics"]["input_power_w"])
    checks: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    max_gaussian_error = 0.0
    for axis in config["diagnostics"]["input_tilt_axes"]:
        for angle in config["diagnostics"]["input_tilt_angles_deg"]:
            tx = float(angle) if axis == "x" else 0.0
            ty = float(angle) if axis == "y" else 0.0
            values = _tilted_square_field(config, plane, device, tx, ty)
            field = ComplexField(values, plane, OperationProvenance("finite_200um_square_tilt", plane.plane_id, None))
            cg = gaussian.project_coefficients(field)[0]
            cl = lp.project(values)[0]
            p_in = float(_power(values[:, 0], plane)[0])
            ref_g = _mp_gaussian_tilt_reference(config, centers, tx, ty)
            err_g = _relative_vector_error(cg.detach().cpu().numpy().astype(np.complex128), ref_g, math.sqrt(pin))
            max_gaussian_error = max(max_gaussian_error, err_g)
            ref_l = _cpu_reference_lp01(values, lp)
            err_l = _relative_vector_error(cl.detach().cpu().numpy().astype(np.complex128), ref_l, math.sqrt(pin))
            power_cpu = float(np.sum(np.abs(values.detach().cpu().numpy()[0, 0].astype(np.complex128)) ** 2, dtype=np.float64) * math.prod(plane.pixel_pitch_m))
            power_error = abs(p_in - power_cpu) / pin
            checks.append({
                "axis": axis,
                "angle_deg": float(angle),
                "input_power_W": p_in,
                "gaussian_complex_coefficient_error_over_sqrt_input_power": err_g,
                "lp01_complex_coefficient_cpu_sum_error_over_sqrt_input_power": err_l,
                "independent_cpu_power_error_over_input_power": power_error,
                "passed": err_g <= float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"])
                and err_l <= float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"])
                and power_error <= float(config["diagnostics"]["power_error_over_input_power_max"]),
            })
            coefficient_rows.append({
                "axis": axis,
                "angle_deg": float(angle),
                "gaussian_complex_coefficients": cg.detach().cpu(),
                "lp01_complex_coefficients": cl.detach().cpu(),
            })
    negative_values = _tilted_square_field(config, plane, device, 1.0, 0.0)
    negative_field = ComplexField(negative_values, plane, OperationProvenance("phase_loss_negative_control", plane.plane_id, None))
    correct = gaussian.project_coefficients(negative_field)[0]
    phase_lost = gaussian.project_coefficients(ComplexField(
        negative_values.abs().to(torch.complex64), plane, OperationProvenance("phase_discarded", plane.plane_id, None)
    ))[0]
    phase_error = float(torch.linalg.vector_norm(correct - phase_lost) / math.sqrt(pin))
    central = 44
    missing_area = _cpu_reference_gaussian(negative_values, plane, centers, gaussian.geometry.waist_m)[central] / math.prod(plane.pixel_pitch_m)
    correct_cpu = _cpu_reference_gaussian(negative_values, plane, centers, gaussian.geometry.waist_m)[central]
    area_error = abs(missing_area - correct_cpu) / math.sqrt(pin)
    negative_controls = {
        "complex_phase_discarded": {
            "expected_rejection": True,
            "observed_error_over_sqrt_input_power": phase_error,
            "rejected_at_1e-4": phase_error > float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"]),
        },
        "pixel_area_omitted": {
            "expected_rejection": True,
            "observed_error_over_sqrt_input_power": float(area_error),
            "rejected_at_1e-4": area_error > float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"]),
        },
    }
    return checks, coefficient_rows, {
        "passed": all(item["passed"] for item in checks) and all(item["rejected_at_1e-4"] for item in negative_controls.values()),
        "maximum_gaussian_error_over_sqrt_input_power": max_gaussian_error,
        "negative_controls": negative_controls,
    }


def _overlap_mode_negative(config: dict[str, Any], plane: PlaneSpec, device: torch.device) -> dict[str, Any]:
    w = float(config["fiber"]["gaussian_1e_amplitude_radius_m"])
    dy, dx = plane.pixel_pitch_m
    yy = (torch.arange(plane.shape[0], dtype=torch.float64, device=device) + 0.5) * dy - plane.fov_m[0] / 2
    xx = (torch.arange(plane.shape[1], dtype=torch.float64, device=device) + 0.5) * dx - plane.fov_m[1] / 2
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    amplitude = math.sqrt(2.0 / (math.pi * w * w))
    u0 = amplitude * torch.exp(-(x.square() + y.square()) / w**2)
    u1 = amplitude * torch.exp(-((x - 1.0e-6).square() + y.square()) / w**2)
    area = dx * dy
    modes = torch.stack((u0, u1))
    gram = modes.reshape(2, -1) @ modes.reshape(2, -1).T * area
    eig, vectors = torch.linalg.eigh(gram)
    lam = eig[-1]
    q = vectors[:, -1]
    incident = (q[0] * u0 + q[1] * u1) / torch.sqrt(lam)
    coeff = torch.stack(((u0 * incident).sum() * area, (u1 * incident).sum() * area))
    emitted = coeff[0] * u0 + coeff[1] * u1
    p_in = float(incident.abs().square().sum() * area)
    p_out = float(emitted.abs().square().sum() * area)
    gain_bound = float(lam.square())
    return {
        "two_mode_center_separation_m": 1.0e-6,
        "gram_max_eigenvalue": float(lam),
        "complete_project_reemit_gain_bound": gain_bound,
        "measured_constructed_eigenfield_power_gain": p_out / max(p_in, EPS),
        "expected_rejection": True,
        "rejected_by_passivity_gate": gain_bound > 1.0 + float(config["diagnostics"]["passivity_relative_tolerance"])
        and p_out / max(p_in, EPS) > 1.0 + float(config["diagnostics"]["passivity_relative_tolerance"]),
    }


def _gradient_check(
    config: dict[str, Any], source: ComplexField, phase: torch.Tensor, plane: PlaneSpec,
    gaussian: GaussianFiberArrayOLS, device: torch.device,
) -> dict[str, Any]:
    pin = float(config["optics"]["input_power_w"])
    optics = config["optics"]
    stack = FiniteCropPhaseStack(
        layers=2,
        active_shape=tuple(optics["encoder_shape"]),
        propagation_shape=(int(optics["global_grid_size"]),) * 2,
        spacing_m=float(optics["inter_encoder_spacing_m"]),
        layer_power_transmission=float(optics["layer_power_transmission"]),
        preserve_global_output=True,
    ).to(device)
    with torch.no_grad():
        stack.phase.copy_(phase)
    output, _, _, = stack(source, "gradient_audit")
    output = _canonical_field(output, plane)
    coeff = gaussian.project_coefficients(output)
    objective = coeff.abs().square().sum() / pin
    gradient = torch.autograd.grad(objective, stack.phase)[0]
    epsilon = float(config["diagnostics"]["gradient_finite_difference_step_rad"])
    rows = []
    for layer in (0, 1):
        flat = gradient[layer, 0].abs().reshape(-1)
        pixel = int(torch.argmax(flat))
        iy, ix = divmod(pixel, gradient.shape[-1])
        ad = float(gradient[layer, 0, iy, ix])
        values = []
        for sign in (-1.0, 1.0):
            perturbed = phase.clone()
            perturbed[layer, 0, iy, ix] += sign * epsilon
            trial, _, _, _ = _run_phase_stack(source, perturbed, config, prefix=f"gradient_fd_{layer}_{sign:+.0f}")
            trial = _canonical_field(trial, plane)
            trial_coeff = gaussian.project_coefficients(trial)
            values.append(float(trial_coeff.abs().square().sum() / pin))
        fd = (values[1] - values[0]) / (2.0 * epsilon)
        relative = abs(ad - fd) / max(abs(ad), abs(fd), 1.0e-10)
        rows.append({
            "layer_index_zero_based": layer,
            "selected_pixel_yx": [iy, ix],
            "autograd_derivative_per_rad": ad,
            "central_finite_difference_per_rad": fd,
            "relative_error": relative,
            "sign_agrees": math.copysign(1.0, ad) == math.copysign(1.0, fd) if ad and fd else ad == fd,
        })
    return {
        "objective": "sum(abs(Gaussian_complex_coefficients)^2)/declared_input_power",
        "finite_difference_step_rad": epsilon,
        "optimizer_steps": 0,
        "both_layers_have_finite_nonzero_gradient": all(math.isfinite(item["autograd_derivative_per_rad"]) for item in rows)
        and all(abs(item["autograd_derivative_per_rad"]) > 0.0 for item in rows),
        "relative_error_max": float(config["diagnostics"]["gradient_relative_error_max"]),
        "passed": all(item["relative_error"] <= float(config["diagnostics"]["gradient_relative_error_max"]) and item["sign_agrees"] for item in rows),
        "layer_checks": rows,
    }


def _phase_model_records(
    config: dict[str, Any], phase_inputs: dict[str, torch.Tensor], device: torch.device
) -> tuple[PlaneSpec, dict[str, ComplexField], dict[str, dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_plane = _plane(config)
    source = _source_field(config, base_plane, device)
    phase_outputs: dict[str, ComplexField] = {}
    propagation: dict[str, dict[str, Any]] = {}
    layer1_fields: dict[str, torch.Tensor] = {}
    for name, phase in phase_inputs.items():
        output, records, diagnostics, stack = _run_phase_stack(source, phase, config, prefix=f"alignment_{name}")
        phase_outputs[name] = output
        layer1_fields[name] = records[0].tensor.detach().cpu()
        propagation[name] = {
            "input_power_W": float(_power(source.tensor[:, 0], source.plane)[0]),
            "fiber_plane_power_before_OLS_W": float(_power(output.tensor[:, 0], output.plane)[0]),
            "layer_power_diagnostics": diagnostics["layer_power"].detach().cpu().tolist(),
            "crop_retention": diagnostics["crop_retention"].detach().cpu().tolist(),
            "phase_screen_unchanged": True,
            "phase_optimizer_updates": 0,
        }
        del stack
    reference = next(iter(phase_outputs.values())).plane
    global_plane = PlaneSpec(
        "common_4p096mm_fiber_reference",
        (int(config["optics"]["global_grid_size"]),) * 2,
        (float(config["optics"]["pixel_pitch_m"]),) * 2,
        float(config["optics"]["wavelength_m"]),
        float(reference.z_m),
        "encoder_global",
    )
    canonical = {name: _canonical_field(field, global_plane) for name, field in phase_outputs.items()}
    return global_plane, canonical, propagation, {"source": source.tensor.detach().cpu(), **phase_inputs}, layer1_fields


def _main_measurements(
    config: dict[str, Any], plane: PlaneSpec, fields: dict[str, ComplexField],
    gaussian: GaussianFiberArrayOLS, lp: SamePlaneLP01OLS,
    propagation: dict[str, dict[str, Any]], centers: list[tuple[float, float]], device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pin = float(config["optics"]["input_power_w"])
    model_results: dict[str, Any] = {}
    readouts: dict[str, Any] = {"plane": plane.to_dict(), "fields": {}, "outputs": {}}
    coefficient_payload: dict[str, Any] = {"phase_cases": {}, "input_power_W": pin}
    cpu_validation: dict[str, Any] = {}
    for name, field in fields.items():
        values = field.tensor
        cg = gaussian.project_coefficients(field)
        cl = lp.project(values)
        gout = gaussian.reemit_coefficients(cg).unsqueeze(1)
        lout = lp.reemit(cl)
        p_before = float(_power(values[:, 0], plane)[0])
        pg = float(_power(gout[:, 0], plane)[0])
        pl = float(_power(lout[:, 0], plane)[0])
        rg = _channel_metrics(cg, p_before, pin, output_power=pg)
        rl = _channel_metrics(cl, p_before, pin, output_power=pl)
        model_results[name] = {"gaussian": rg, "lp01": rl, "same_field_input_power_W": p_before}
        cpu_g = _cpu_reference_gaussian(values, plane, centers, gaussian.geometry.waist_m)
        cpu_l = _cpu_reference_lp01(values, lp)
        err_g = _relative_vector_error(cg[0].detach().cpu().numpy().astype(np.complex128), cpu_g, math.sqrt(pin))
        err_l = _relative_vector_error(cl[0].detach().cpu().numpy().astype(np.complex128), cpu_l, math.sqrt(pin))
        cpu_power = float(np.sum(np.abs(values.detach().cpu().numpy()[0, 0].astype(np.complex128)) ** 2, dtype=np.float64) * math.prod(plane.pixel_pitch_m))
        cpu_validation[name] = {
            "gaussian_coefficient_error_over_sqrt_input_power": err_g,
            "lp01_coefficient_error_over_sqrt_input_power": err_l,
            "input_power_error_over_input_power": abs(p_before - cpu_power) / pin,
            "gaussian_power_error_over_input_power": abs(pg - float(np.sum(np.abs(gout.detach().cpu().numpy()[0, 0].astype(np.complex128)) ** 2, dtype=np.float64) * math.prod(plane.pixel_pitch_m)))/pin,
            "lp01_power_error_over_input_power": abs(pl - float(np.sum(np.abs(lout.detach().cpu().numpy()[0, 0].astype(np.complex128)) ** 2, dtype=np.float64) * math.prod(plane.pixel_pitch_m)))/pin,
        }
        # One common phase is applied to the LP01 vector; no per-port phase fitting is allowed.
        g_np = cg[0].detach().cpu().numpy().astype(np.complex128)
        l_np = cl[0].detach().cpu().numpy().astype(np.complex128)
        phase = float(np.angle(np.vdot(g_np, l_np))) if np.linalg.norm(g_np) and np.linalg.norm(l_np) else 0.0
        l_aligned = l_np * np.exp(-1j * phase)
        gp = np.abs(g_np) ** 2
        lpow = np.abs(l_np) ** 2
        emitted_intensity_g = gout.abs().square()
        emitted_intensity_l = lout.abs().square()
        emitted_intensity_relative_error = float(
            torch.linalg.vector_norm(emitted_intensity_l - emitted_intensity_g)
            / torch.linalg.vector_norm(emitted_intensity_g).clamp_min(EPS)
        )
        coefficient_payload["phase_cases"][name] = {
            "gaussian_complex_coefficients": cg[0].detach().cpu(),
            "lp01_complex_coefficients": cl[0].detach().cpu(),
            "gaussian_channel_power_W": cg[0].abs().square().detach().cpu(),
            "lp01_channel_power_W": cl[0].abs().square().detach().cpu(),
            "input_power_W": p_before,
            "gaussian_reemitted_power_W": pg,
            "lp01_reemitted_power_W": pl,
        }
        model_results[name]["complex_latent_comparison"] = {
            "common_global_phase_rad": phase,
            "raw_complex_relative_error": float(np.linalg.norm(l_np - g_np) / max(np.linalg.norm(g_np), EPS)),
            "phase_aligned_complex_relative_error": float(np.linalg.norm(l_aligned - g_np) / max(np.linalg.norm(g_np), EPS)),
            "raw_complex_error_over_sqrt_input_power": float(np.linalg.norm(l_np - g_np) / math.sqrt(pin)),
            "phase_aligned_complex_error_over_sqrt_input_power": float(np.linalg.norm(l_aligned - g_np) / math.sqrt(pin)),
            "emitted_field_intensity_relative_error": emitted_intensity_relative_error,
            "per_fiber_power_distribution_relative_error": float(np.linalg.norm(lpow - gp) / max(np.linalg.norm(gp), EPS)),
            "fiber_total_coupling_absolute_error": abs(rl["eta_fiber_face"] - rg["eta_fiber_face"]),
            "fiber_total_coupling_system_absolute_error": abs(rl["eta_system"] - rg["eta_system"]),
            "per_fiber_power_max_abs_error_over_input_power": float(np.max(np.abs(lpow - gp)) / pin),
            "per_fiber_power_l2_error_over_input_power": float(np.linalg.norm(lpow - gp) / pin),
            "per_fiber_raw_complex_error_over_sqrt_input_power": (np.abs(l_np - g_np) / math.sqrt(pin)).tolist(),
            "per_fiber_complex_error_over_sqrt_input_power": (np.abs(l_aligned - g_np) / math.sqrt(pin)).tolist(),
            "gaussian_edge_to_center_ratio": rg["edge_to_center_mean_power_ratio"],
            "lp01_edge_to_center_ratio": rl["edge_to_center_mean_power_ratio"],
        }
        readouts["fields"][name] = values.detach().cpu()
        readouts["outputs"][name] = {"gaussian": gout.detach().cpu(), "lp01": lout.detach().cpu()}
        propagation[name]["fiber_plane_power_before_OLS_W"] = p_before
    return model_results, readouts, {"coefficients": coefficient_payload, "cpu_validation": cpu_validation}


def _write_report_md(report: dict[str, Any]) -> str:
    decision = report["final_decision"]
    verdict = report["checks"]
    return (
        "# Gaussian OLS 仿真基线与 LP01 对照验证\n\n"
        "## 结果与状态\n\n"
        f"唯一判定：**{decision}**。Gaussian公式、共同全局坐标、复数投影/重发射和独立数值核验的内部检查状态为 `{'PASS' if verdict['all_numerical_checks_passed'] else 'FAIL'}`。"
        "本轮正式训练、phase optimizer更新、数据读取及任务评价器调用均为0。\n\n"
        "## 科学认识\n\n"
        "Gaussian主模型使用论文S12完整解析归一化复Gaussian基模；LP01对照使用仓库的V=2.2阶跃折射率基模和原Voronoi端口截断。"
        "两者在同一4.096 mm参考面、同一125 μm阵列坐标、同一14.222 μm MFD定义下消费相同复场。观察到的差值属于模型敏感性；"
        "单场N_eff只是功率参与率，不代表独立复latent秩。参数均为未测量的仿真假设。\n\n"
        "## 目标缺口\n\n"
        "本结果不包含器件实测MFD、NA、真实端口坐标、复传输矩阵、相位稳定性或硬件验证，也不验证正式训练、语义生成质量或Gaussian相对LP01的任务收益。"
        "历史G3传播窗口收敛问题本轮未复测，仍保持未解决。\n\n"
        "## 工程完成与验证\n\n"
        f"单模匹配检查：{len(report['single_mode_checks'])}项；相干双端口检查：{len(report['coherent_two_port_checks'])}项；有限200 μm方形倾角检查：{len(report['tilt_checks'])}项；"
        f"解析/数值复系数误差门槛为1e-4，完整算子被动性相对门槛为1e-5。阶段前向采用生产FiniteCropPhaseStack和ASM。\n\n"
        "## 决策影响与下一步\n\n"
        "若所有数值检查通过，Gaussian可作为未标定125 μm参考几何下的后续仿真统一参考模型，LP01结果保留为模式假设敏感性对照；"
        "这不授权切换正式默认模型或开始训练。维度分析和非线性分析留待单独决策与授权。\n"
    )


def run(config_path: Path, run_dir: Path, *, preflight_only: bool) -> dict[str, Any]:
    config = _load_config(config_path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path.resolve(), run_dir / "executed_config.yaml")
    shutil.copyfile(Path(__file__).resolve(), run_dir / "executed_runner.py")
    if not torch.cuda.is_available():
        raise RuntimeError("approved optical validation requires CUDA on AutoDL")
    device = torch.device("cuda:0")
    torch.manual_seed(20261013)
    # This command runs in a fresh process, so CUDA peak counters already start
    # at zero; resetting with a device object is not supported on every build.
    clear_production_transfer_cache()
    started = time.perf_counter()
    plan = {
        "schema": SCHEMA,
        "question": "Is the complete Gaussian S12 model numerically self-consistent and suitable as an uncalibrated simulation reference against same-plane LP01?",
        "input_power_W": config["optics"]["input_power_w"],
        "synthetic_only": True,
        "formal_training_updates": 0,
        "phase_optimizer_updates": 0,
        "dataset_reads": 0,
        "evaluator_calls": 0,
        "g3_status": "HISTORICAL_FAIL_REMAINS_UNRESOLVED_NOT_RETESTED",
        "frozen_config": config,
        "stop": "complete this optical model validation and return evidence; do not begin dimension, nonlinearity, or generation work",
    }
    _write_json(run_dir / "plan.json", plan)
    active_plane = _plane(config)
    source = _source_field(config, active_plane, device)
    phases = _make_phases(config, device)
    preflight_phase = phases["last_layer_x_1deg"]
    preflight_field, _, preflight_diag, _ = _run_phase_stack(source, preflight_phase, config, prefix="preflight")
    global_plane = PlaneSpec(
        "common_4p096mm_fiber_reference",
        (int(config["optics"]["global_grid_size"]),) * 2,
        (float(config["optics"]["pixel_pitch_m"]),) * 2,
        float(config["optics"]["wavelength_m"]),
        float(preflight_field.plane.z_m),
        "encoder_global",
    )
    preflight_field = _canonical_field(preflight_field, global_plane)
    geometry = GaussianFiberGeometry(
        port_pitch_m=float(config["fiber"]["pitch_m"]),
        waist_m=float(config["fiber"]["gaussian_1e_amplitude_radius_m"]),
    )
    gaussian = GaussianFiberArrayOLS(global_plane, geometry).to(device)
    lp = SamePlaneLP01OLS(global_plane, config, device)
    preflight_g = gaussian.project_coefficients(preflight_field)
    preflight_l = lp.project(preflight_field.tensor)
    preflight_g_out = gaussian.reemit_coefficients(preflight_g)
    preflight_l_out = lp.reemit(preflight_l)
    p_input = float(_power(source.tensor[:, 0], source.plane)[0])
    p_before = float(_power(preflight_field.tensor[:, 0], global_plane)[0])
    preflight = {
        "status": "PASS" if all(torch.isfinite(x).all() for x in (preflight_field.tensor, preflight_g, preflight_l, preflight_g_out, preflight_l_out))
        and float(_power(preflight_g_out, global_plane)[0]) <= p_before * (1.0 + PASSIVITY_TOLERANCE)
        and float(_power(preflight_l_out, global_plane)[0]) <= p_before * (1.0 + PASSIVITY_TOLERANCE) else "FAIL",
        "device": torch.cuda.get_device_name(device),
        "input_power_W": p_input,
        "power_before_OLS_W": p_before,
        "global_plane": global_plane.to_dict(),
        "gaussian_coefficients_finite": bool(torch.isfinite(preflight_g).all()),
        "lp01_coefficients_finite": bool(torch.isfinite(preflight_l).all()),
        "gaussian_output_power_W": float(_power(preflight_g_out, global_plane)[0]),
        "lp01_output_power_W": float(_power(preflight_l_out, global_plane)[0]),
        "complex_fields_not_resized": True,
        "representative_production_phase_stack_and_OLS_forward": True,
        "layer_power_diagnostics": preflight_diag["layer_power"].detach().cpu().tolist(),
        "updates_dataset_evaluators": {"task_training": 0, "phase_optimizer": 0, "dataset_reads": 0, "task_evaluator_calls": 0},
    }
    _write_json(run_dir / "preflight.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError("representative Gaussian/LP01 production preflight failed")
    if preflight_only:
        return {"status": "PREFLIGHT_PASS", "preflight": preflight}

    centers = _fiber_centers(config)
    phase_plane, fields, propagation, phase_payload, layer1_fields = _phase_model_records(config, phases, device)
    if phase_plane != global_plane:
        raise ValueError("preflight and full measurement physical planes differ")
    gaussian_gram = _gaussian_gram(gaussian, device)
    lp_gram = lp.gram()
    tolerance = float(config["diagnostics"]["passivity_relative_tolerance"])
    gram_summary = {
        "gaussian": _mode_gram_summary(gaussian_gram, tolerance),
        "lp01": _mode_gram_summary(lp_gram, tolerance),
    }
    main_results, readouts, extras = _main_measurements(config, phase_plane, fields, gaussian, lp, propagation, centers, device)
    single_mode = _single_mode_checks(config, phase_plane, gaussian, lp, centers, device)
    coherent = _coherent_two_port_checks(config, phase_plane, gaussian, lp, device)
    tilt_checks, tilt_coeffs, tilt_summary = _tilt_checks(config, phase_plane, gaussian, lp, centers, device)
    mp_modes = _independent_lp01_constants(config)
    overlap_negative = _overlap_mode_negative(config, phase_plane, device)
    gradient = _gradient_check(config, source, phases["last_layer_xy_1deg"], phase_plane, gaussian, device)

    input_tilt_coeffs = []
    for record in tilt_coeffs:
        input_tilt_coeffs.append({
            **{key: record[key] for key in ("axis", "angle_deg")},
            "gaussian_complex_coefficients": record["gaussian_complex_coefficients"],
            "lp01_complex_coefficients": record["lp01_complex_coefficients"],
        })
    extras["coefficients"]["input_tilts"] = input_tilt_coeffs
    extras["coefficients"]["single_mode_checks"] = single_mode
    extras["coefficients"]["coherent_two_port_checks"] = coherent
    all_cpu_coeff_ok = all(
        row["gaussian_coefficient_error_over_sqrt_input_power"] <= float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"])
        and row["lp01_coefficient_error_over_sqrt_input_power"] <= float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"])
        and row["input_power_error_over_input_power"] <= float(config["diagnostics"]["power_error_over_input_power_max"])
        for row in extras["cpu_validation"].values()
    )
    checks = {
        "representative_preflight": preflight["status"] == "PASS",
        "gaussian_and_lp01_complete_operator_passive": gram_summary["gaussian"]["passive_at_relative_tolerance"]
        and gram_summary["lp01"]["passive_at_relative_tolerance"],
        "sampled_mode_norms_within_1e-4": gram_summary["gaussian"]["mode_norm_max_abs_error"] <= 1.0e-4
        and gram_summary["lp01"]["mode_norm_max_abs_error"] <= 1.0e-4,
        "all_phase_case_cpu_complex_and_power_references": all_cpu_coeff_ok,
        "single_mode_match_cpu_reference": all(row["cpu_complex_coefficient_error_over_sqrt_input_power"] <= float(config["diagnostics"]["coefficient_error_over_sqrt_input_power_max"]) for row in single_mode),
        "coherent_phase_retained": all(abs(row["relative_phase_error_rad"]) <= 1.0e-4 for row in coherent),
        "finite_square_tilt_and_negative_controls": tilt_summary["passed"],
        "overlap_mode_energy_creation_rejected": overlap_negative["rejected_by_passivity_gate"],
        "autograd_vs_finite_difference": gradient["passed"],
        "independent_lp01_eigen_solution": mp_modes["max_abs_difference"]
        <= float(config["diagnostics"]["lp01_independent_eigen_solution_abs_tolerance"]),
        "re_emitted_fields_remain_passive": all(
            main_results[name][model]["re_emitted_field_power_W"]
            <= main_results[name]["same_field_input_power_W"] * (1.0 + tolerance)
            for name in main_results for model in ("gaussian", "lp01")
        ),
    }
    decision = "GAUSSIAN_S12_REFERENCE_VALIDATED_LP01_SENSITIVITY_RECORDED" if all(checks.values()) else "INVALID_HARNESS"
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE" if all(checks.values()) else "INVALID_HARNESS",
        "final_decision": decision,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds_observed": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "frozen_system": {
            "wavelength_m": config["optics"]["wavelength_m"],
            "pixel_pitch_m": config["optics"]["pixel_pitch_m"],
            "encoder_shape": config["optics"]["encoder_shape"],
            "encoder_aperture_m": config["optics"]["encoder_aperture_m"],
            "inter_encoder_spacing_m": config["optics"]["inter_encoder_spacing_m"],
            "encoder_layer_spacing_sequence_m": [config["optics"]["inter_encoder_spacing_m"]],
            "last_encoder_to_fiber_m": config["optics"]["last_encoder_to_fiber_m"],
            "global_plane": phase_plane.to_dict(),
            "global_plane_extent_m": list(phase_plane.fov_m),
            "fiber_pitch_m": config["fiber"]["pitch_m"],
            "fiber_mfd_m": config["fiber"]["mode_field_diameter_m"],
            "gaussian_1e_radius_m": config["fiber"]["gaussian_1e_amplitude_radius_m"],
            "fiber_centers_xy_m_row_major": centers,
            "outermost_fiber_center_xy_m": [centers[0], centers[9], centers[90], centers[99]],
            "array_outer_extent_including_assumed_cladding_m": 9.0 * float(config["fiber"]["pitch_m"]) + float(config["fiber"]["cladding_diameter_m"]),
            "array_outer_extent_assumption": "9*pitch + 125um cladding diameter; cladding is a simulation geometry assumption",
            "geometry_status": config["fiber"]["geometry_status"],
            "layer_power_transmission": config["optics"]["layer_power_transmission"],
            "ols_transport": "T=I_100",
            "field_resize": False,
        },
        "model_definitions": {
            "gaussian": {
                "formula": "sqrt(2/(pi*w^2))*exp(-((x-xj)^2+(y-yj)^2)/w^2)",
                "analytic_infinite_plane_normalization": True,
                "per_port_crop": False,
                "discrete_renormalization": False,
                "orthogonalization": False,
            },
            "lp01": {
                "V": config["fiber"]["lp01_v_number"],
                "radial_solution": "step-index LP01 scalar eigenmode from production fiber_operator",
                "per_port_support": "production Voronoi cell truncation at pitch/2",
                "per_port_discrete_unit_power_normalization": True,
                "output_geometry": "same global coordinates and same LP01 mode patches as input projection",
                "legacy_compact_decoder_mapping_used": False,
            },
        },
        "checks": {**checks, "all_numerical_checks_passed": all(checks.values())},
        "gram_summary": gram_summary,
        "phase_propagation": propagation,
        "same_field_model_comparison": main_results,
        "single_mode_checks": single_mode,
        "coherent_two_port_checks": coherent,
        "tilt_checks": tilt_checks,
        "negative_controls": {
            "complex_phase_and_area": tilt_summary["negative_controls"],
            "overlapping_modes_passivity": overlap_negative,
        },
        "gradient_check": gradient,
        "independent_lp01_eigen_solution": mp_modes,
        "cpu_reference_summary": extras["cpu_validation"],
        "updates_and_external_calls": {"task_training_updates": 0, "phase_optimizer_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0, "FID_calls": 0},
        "single_mode_complex_coefficients": single_mode,
        "coherent_two_port_complex_coefficients": coherent,
        "historical_g3": "previous FAIL remains unresolved; not retested or reclassified by this task",
        "limitations": [
            "All fiber geometry remains an unmeasured simulation assumption.",
            "T=I ignores measured loss, crosstalk, phase drift, polarization, and device variation.",
            "Single-field N_eff is instantaneous power participation, not independent complex latent dimension.",
            "No camera, task loss, data, evaluator, training, material, or semantic generation was evaluated.",
        ],
        "source_outputs": [
            "plan.json", "preflight.json", "report.json", "report.md", "fixed_phase_screens.pt",
            "complex_field_readouts.pt", "coefficients_and_power.pt", "gram_matrices.pt",
            "cpu_reference_validation.json", "gradient_validation.json", "module_sources.json",
            "executed_config.yaml", "executed_runner.py", "resource_telemetry.csv",
        ],
    }
    _write_json(run_dir / "cpu_reference_validation.json", {
        "mpmath_precision_digits": int(config["diagnostics"]["mp_dps"]),
        "gaussian_tilt_checks": tilt_checks,
        "phase_case_references": extras["cpu_validation"],
        "lp01_independent_eigen_solution": mp_modes,
        "negative_controls": tilt_summary["negative_controls"],
    })
    _write_json(run_dir / "gradient_validation.json", gradient)
    torch.save({name: value.detach().cpu() for name, value in phase_payload.items() if name != "source"}, run_dir / "fixed_phase_screens.pt")
    readouts["encoder_input_field"] = phase_payload["source"]
    readouts["first_encoder_output_fields"] = layer1_fields
    torch.save(readouts, run_dir / "complex_field_readouts.pt")
    torch.save(extras["coefficients"], run_dir / "coefficients_and_power.pt")
    torch.save({"gaussian": gaussian_gram.detach().cpu(), "lp01": lp_gram.detach().cpu(), "fiber_centers_xy_m_row_major": centers}, run_dir / "gram_matrices.pt")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    module_sources = {
        "project_root": str(Path.cwd()),
        "git_head_provenance_only": commit,
        "modules": [
            {"name": name, "path": str(Path(module.__file__).resolve())}
            for name, module in (
                ("finite_crop_phase_stack", __import__("ldm_project.v2.optics.d2nn", fromlist=["FiniteCropPhaseStack"])),
                ("angular_spectrum", __import__("ldm_project.v2.optics.propagation", fromlist=["angular_spectrum_propagate"])),
                ("fiber_operator", __import__("ldm_project.v2.optics.fiber_operator", fromlist=["FiberArrayOLS"])),
                ("gaussian_baseline", __import__("ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline", fromlist=["GaussianFiberArrayOLS"])),
            )
        ],
        "source_digests_are_not_admission_or_success_gates": True,
    }
    _write_json(run_dir / "module_sources.json", module_sources)
    _write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(_write_report_md(report), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError("one or more predeclared numerical checks failed; see report.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps({"status": result.get("status"), "decision": result.get("final_decision"), "preflight": result.get("preflight")}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
