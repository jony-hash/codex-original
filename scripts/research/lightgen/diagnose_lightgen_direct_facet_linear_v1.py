"""Diagnose a literature-only glass -> Te -> air facet at normal incidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTER_COLUMNS = (
    "record_id", "material_or_device", "sample_identity", "parameter", "value",
    "original_unit", "converted_value", "converted_unit", "wavelength",
    "condition", "source_id", "uncertainty_or_limit", "model_use",
)
POWER_TOL = 1e-9
FIELD_TOL = 1e-10


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    _write_new(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema") != "lightgen_direct_facet_linear.v1":
        raise ValueError("Unsupported or malformed direct-facet config")
    return config


def parse_parameter_register(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.reader(stream, strict=True))
    if not raw_rows or tuple(raw_rows[0]) != REGISTER_COLUMNS:
        raise ValueError(f"{csv_path}: parameter-register header does not match the 13-column contract")
    parsed: list[dict[str, str]] = []
    for line_number, values in enumerate(raw_rows[1:], start=2):
        if len(values) != len(REGISTER_COLUMNS):
            raise ValueError(f"{csv_path}:{line_number}: {len(values)} fields, expected 13")
        row = dict(zip(REGISTER_COLUMNS, values, strict=True))
        if not row["model_use"].strip():
            raise ValueError(f"{csv_path}:{line_number}: model_use is empty")
        parsed.append(row)
    return parsed


def _interpolate_te(config: dict[str, Any]) -> tuple[float, float, float]:
    material = config["material"]
    wavelength_um = float(config["model"]["wavelength_nm"]) / 1000.0
    lower, upper = material["te_nk_lower"], material["te_nk_upper"]
    lam0, lam1 = float(lower["wavelength_um"]), float(upper["wavelength_um"])
    if not lam0 <= wavelength_um <= lam1 or lam0 >= lam1:
        raise ValueError("Target wavelength is outside the configured adjacent Te data points")
    fraction = (wavelength_um - lam0) / (lam1 - lam0)
    n = float(lower["n"]) + fraction * (float(upper["n"]) - float(lower["n"]))
    k = float(lower["k"]) + fraction * (float(upper["k"]) - float(lower["k"]))
    return wavelength_um, n, k


def _malitson_n(wavelength_um: float, coefficients: dict[str, float]) -> float:
    n_squared = 1.0
    for b_name, c_name in (("b1", "c1_um"), ("b2", "c2_um"), ("b3", "c3_um")):
        b, c = float(coefficients[b_name]), float(coefficients[c_name])
        n_squared += b * wavelength_um**2 / (wavelength_um**2 - c**2)
    return math.sqrt(n_squared)


def load_model_inputs(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    if (model["time_factor"] != "exp(-i omega t)" or model["incidence"] != "normal_plane_wave"
            or model["polarization"] != "scalar_isotropic"
            or model["propagation_direction"] != "glass -> Te -> air"):
        raise ValueError("The direct-facet diagnostic accepts only its declared normal-incidence convention")
    planes = config["reference_planes"]
    if planes["input_plane"] != "z=0 at glass/Te boundary" or planes["output_plane"] != "z=d at Te/air boundary":
        raise ValueError("The direct-facet diagnostic requires the declared common reference planes")
    wavelength_um, te_n, te_k = _interpolate_te(config)
    wavelength_m = float(config["model"]["wavelength_nm"]) * 1e-9
    thickness_m = float(config["material"]["te_thickness_nm"]) * 1e-9
    n_air = float(config["media"]["air_refractive_index"])
    glass_k = float(config["media"]["glass_extinction_coefficient_assumption"])
    if wavelength_m <= 0 or thickness_m < 0 or n_air <= 0 or glass_k != 0:
        raise ValueError("Wavelength, thickness, and media assumptions are invalid")
    if te_n <= 0 or te_k < 0:
        raise ValueError("Te index is not passive under the configured exp(-i omega t) convention")
    glass = config["media"]["glass_model"]
    n_glass = _malitson_n(wavelength_um, glass["malitson_sellmeier"])
    checks = config["checks"]
    field_tolerance = float(checks["complex_field_absolute_tolerance"])
    power_tolerance = float(checks["power_and_dissipation_absolute_tolerance"])
    quadrature_orders = tuple(int(value) for value in checks["absorption_quadrature_orders"])
    profile_samples = int(config["output"]["profile_samples"])
    if field_tolerance <= 0 or power_tolerance <= 0 or len(quadrature_orders) != 2 or profile_samples < 3:
        raise ValueError("Configured tolerances, quadrature orders, or profile sample count are invalid")
    if tuple(sorted(quadrature_orders)) != quadrature_orders or min(quadrature_orders) < 2:
        raise ValueError("Absorption quadrature orders must be two increasing positive orders")
    if quadrature_orders != (64, 128):
        raise ValueError("This diagnostic uses the fixed 64/128-point absorption comparison")
    return {
        "wavelength_m": wavelength_m,
        "wavelength_um": wavelength_um,
        "te_thickness_m": thickness_m,
        "n_glass": n_glass,
        "k_glass_assumption": glass_k,
        "n_te": te_n,
        "k_te": te_k,
        "n_air": n_air,
        "te_source_id": str(config["material"]["te_source_id"]),
        "glass_source_id": str(glass["source_id"]),
        "sample_identity": str(config["material"]["sample_identity"]),
        "field_tolerance": field_tolerance,
        "power_tolerance": power_tolerance,
        "absorption_quadrature_orders": list(quadrature_orders),
        "profile_samples": profile_samples,
    }


def _complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def single_layer_coefficients(
    n_in: float, n_layer: complex, n_out: float, thickness_m: float, wavelength_m: float,
) -> dict[str, complex]:
    """Return r, t at z=d, and layer forward/backward amplitudes at z=0."""
    if n_in <= 0 or n_out <= 0 or thickness_m < 0 or wavelength_m <= 0:
        raise ValueError("Refractive indices and wavelength must be positive; thickness cannot be negative")
    if n_layer.real <= 0 or n_layer.imag < 0:
        raise ValueError("The layer must have positive real index and nonnegative extinction coefficient")
    r01 = (n_in - n_layer) / (n_in + n_layer)
    r12 = (n_layer - n_out) / (n_layer + n_out)
    t01 = 2.0 * n_in / (n_in + n_layer)
    t12 = 2.0 * n_layer / (n_layer + n_out)
    delta = 2.0 * math.pi * n_layer * thickness_m / wavelength_m
    round_trip = np.exp(2j * delta)
    denominator = 1.0 + r01 * r12 * round_trip
    r = (r01 + r12 * round_trip) / denominator
    t = t01 * t12 * np.exp(1j * delta) / denominator
    forward_0 = t01 / denominator
    backward_0 = r12 * round_trip * forward_0
    return {"r": complex(r), "t": complex(t), "forward_0": complex(forward_0),
            "backward_0": complex(backward_0)}


def solve_boundary_continuity_independently(
    n_in: float, n_layer: complex, n_out: float, thickness_m: float, wavelength_m: float,
) -> tuple[complex, complex]:
    """Solve four E/H boundary conditions without the closed-form stack expression."""
    phase = np.exp(2j * math.pi * n_layer * thickness_m / wavelength_m)
    matrix = np.array([
        [1.0, -1.0, -1.0, 0.0],
        [n_in, n_layer, -n_layer, 0.0],
        [0.0, phase, 1.0 / phase, -1.0],
        [0.0, n_layer * phase, -n_layer / phase, -n_out],
    ], dtype=np.complex128)
    rhs = np.array([-1.0, n_in, 0.0, 0.0], dtype=np.complex128)
    solution = np.linalg.solve(matrix, rhs)
    return complex(solution[0]), complex(solution[3])


def power_metrics(r: complex, t: complex, n_in: float, n_out: float) -> dict[str, float]:
    reflectance = float(abs(r) ** 2)
    transmittance = float((n_out / n_in) * abs(t) ** 2)
    return {"R": reflectance, "T": transmittance, "A_flux": 1.0 - reflectance - transmittance}


def absorption_density_per_m(
    z_m: np.ndarray | float, forward_0: complex, backward_0: complex, n_layer: complex,
    wavelength_m: float, n_in: float,
) -> np.ndarray:
    z = np.asarray(z_m, dtype=np.float64)
    k0 = 2.0 * math.pi / wavelength_m
    field = forward_0 * np.exp(1j * k0 * n_layer * z) + backward_0 * np.exp(-1j * k0 * n_layer * z)
    return (k0 * float((n_layer**2).imag) / n_in) * np.abs(field) ** 2


def integrate_absorption(
    thickness_m: float, forward_0: complex, backward_0: complex, n_layer: complex,
    wavelength_m: float, n_in: float, order: int,
) -> float:
    if thickness_m == 0 or n_layer.imag == 0:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(order)
    z = 0.5 * thickness_m * (nodes + 1.0)
    values = absorption_density_per_m(z, forward_0, backward_0, n_layer, wavelength_m, n_in)
    return float(0.5 * thickness_m * np.dot(weights, values))


def validate_power_balance(metrics: dict[str, float], integrated_absorption: float,
                           tolerance: float = POWER_TOL) -> float:
    values = (metrics["R"], metrics["T"], metrics["A_flux"], integrated_absorption)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite optical power quantity")
    if metrics["R"] < -tolerance or metrics["T"] < -tolerance or metrics["A_flux"] < -tolerance:
        raise ValueError("Unphysical negative reflected, transmitted, or absorbed power")
    error = abs(metrics["A_flux"] - integrated_absorption)
    if error > tolerance:
        raise ValueError(f"Flux absorption differs from integrated dissipation by {error:.6g}")
    return error


def validate_absorption_balance(r: complex, t: complex, n_in: float, n_out: float,
                                integrated_absorption: float, tolerance: float = POWER_TOL) -> float:
    return validate_power_balance(power_metrics(r, t, n_in, n_out), integrated_absorption, tolerance)


def _identity(inputs: dict[str, Any]) -> dict[str, Any]:
    keys = ("wavelength_m", "te_thickness_m", "n_glass", "k_glass_assumption", "n_te", "k_te",
            "n_air", "te_source_id", "glass_source_id", "sample_identity", "field_tolerance",
            "power_tolerance", "absorption_quadrature_orders", "profile_samples")
    return {key: inputs[key] for key in keys}


def _validate_register_values(rows: list[dict[str, str]], inputs: dict[str, Any]) -> None:
    by_id = {row["record_id"]: row for row in rows}
    for record_id, expected in (("TE-LIN-N", inputs["n_te"]), ("TE-LIN-K", inputs["k_te"])):
        row = by_id.get(record_id)
        if row is None:
            raise ValueError(f"Parameter register is missing {record_id}")
        if row["material_or_device"] != inputs["sample_identity"] or row["source_id"] != inputs["te_source_id"]:
            raise ValueError(f"Parameter register identity does not match {record_id}")
        if abs(float(row["value"]) - expected) > 5e-5:
            raise ValueError(f"Parameter register value disagrees with interpolation for {record_id}")


def _load_inputs(config_path: str | Path):
    config = load_config(config_path)
    inputs = load_model_inputs(config)
    register_path = _resolve_repo_path(config["sources"]["parameter_register_csv"])
    register_rows = parse_parameter_register(register_path)
    _validate_register_values(register_rows, inputs)
    return config, inputs, register_path, register_rows


def run_preflight(config_path: str | Path, output_dir: str | Path) -> int:
    config, inputs, register_path, register_rows = _load_inputs(config_path)
    n_in, n_out = inputs["n_glass"], inputs["n_air"]
    n_te = complex(inputs["n_te"], inputs["k_te"])
    thickness, wavelength = inputs["te_thickness_m"], inputs["wavelength_m"]
    primary = single_layer_coefficients(n_in, n_te, n_out, thickness, wavelength)
    independent_r, independent_t = solve_boundary_continuity_independently(n_in, n_te, n_out, thickness, wavelength)
    zero = single_layer_coefficients(n_in, n_te, n_out, 0.0, wavelength)
    bare_r, bare_t = (n_in - n_out) / (n_in + n_out), 2.0 * n_in / (n_in + n_out)
    lossless = single_layer_coefficients(n_in, complex(inputs["n_te"], 0.0), n_out, thickness, wavelength)
    lossless_power = power_metrics(lossless["r"], lossless["t"], n_in, n_out)
    order_64, order_128 = inputs["absorption_quadrature_orders"]
    power_tolerance, field_tolerance = inputs["power_tolerance"], inputs["field_tolerance"]
    a64 = integrate_absorption(thickness, primary["forward_0"], primary["backward_0"], n_te, wavelength, n_in, order_64)
    a128 = integrate_absorption(thickness, primary["forward_0"], primary["backward_0"], n_te, wavelength, n_in, order_128)
    absorption_error = validate_absorption_balance(primary["r"], primary["t"], n_in, n_out, a128, power_tolerance)
    negative_rejected = False
    wrong_t = complex(bare_t)
    wrong_r = complex(bare_r)
    wrong_t_power = abs(wrong_t) ** 2
    wrong_metrics = {"R": abs(wrong_r) ** 2, "T": wrong_t_power,
                     "A_flux": 1.0 - abs(wrong_r) ** 2 - wrong_t_power}
    try:
        validate_power_balance(wrong_metrics, 0.0)
    except ValueError:
        negative_rejected = True
    checks = [
        {"check": "register_schema_and_model_use", "passed": True, "rows": len(register_rows),
         "columns": len(REGISTER_COLUMNS), "register_path": str(register_path)},
        {"check": "independent_boundary_continuity",
         "passed": abs(primary["r"] - independent_r) <= field_tolerance and abs(primary["t"] - independent_t) <= field_tolerance,
         "delta_r": abs(primary["r"] - independent_r), "delta_t": abs(primary["t"] - independent_t)},
        {"check": "zero_thickness_equals_bare_interface",
         "passed": abs(zero["r"] - bare_r) <= field_tolerance and abs(zero["t"] - bare_t) <= field_tolerance,
         "delta_r": abs(zero["r"] - bare_r), "delta_t": abs(zero["t"] - bare_t)},
        {"check": "lossless_power_conservation",
         "passed": abs(lossless_power["A_flux"]) <= power_tolerance and abs(lossless_power["R"] + lossless_power["T"] - 1.0) <= power_tolerance,
         **lossless_power},
        {"check": "passive_absorption_and_integrated_dissipation",
         "passed": a128 > 0 and absorption_error <= power_tolerance,
         "A_flux": power_metrics(primary["r"], primary["t"], n_in, n_out)["A_flux"],
         "A_integral_64": a64, "A_integral_128": a128, "energy_error": absorption_error},
        {"check": "quadrature_convergence", "passed": abs(a64 - a128) <= power_tolerance,
         "order_64": a64, "order_128": a128, "absolute_difference": abs(a64 - a128)},
        {"check": "incorrect_power_normalization_rejected", "passed": negative_rejected,
         "incorrect_transmittance": wrong_t_power,
         "correct_transmittance": power_metrics(wrong_r, wrong_t, n_in, n_out)["T"]},
    ]
    passed = all(check["passed"] for check in checks)
    payload = {"schema": "lightgen_direct_facet_linear_preflight.v1",
               "status": "passed" if passed else "invalid_harness", "model_identity": _identity(inputs),
               "config_path": str(Path(config_path).resolve()), "checks": checks, "check_count": len(checks)}
    out = Path(output_dir)
    _write_json_new(out / "preflight.json", payload)
    if not passed:
        return 2
    report = ["# Direct-facet linear diagnostic preflight", "", "Status: **passed**", "",
              f"Model identity: `{inputs['sample_identity']}`, {wavelength:.12g} m, thickness {thickness:.12g} m.",
              "The candidate passed source-table, independent-field, power, absorption, quadrature, and negative-normalization checks."]
    _write_new(out / "preflight_report.md", "\n".join(report) + "\n")
    return 0


def _bare_coefficients(n_in: float, n_out: float, thickness_m: float, wavelength_m: float) -> dict[str, complex]:
    r = (n_in - n_out) / (n_in + n_out)
    t = 2.0 * n_in / (n_in + n_out) * np.exp(2j * math.pi * n_out * thickness_m / wavelength_m)
    return {"r": complex(r), "t": complex(t)}


def _save_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ("case", "rE_real", "rE_imag", "tE_real", "tE_imag", "tauE_real", "tauE_imag",
               "R", "T", "A_flux", "A_integral_128", "reference_plane_m")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_profile(path: Path, inputs: dict[str, Any], solution: dict[str, complex], count: int) -> None:
    columns = ("z_m", "z_nm", "field_real_over_incident", "field_imag_over_incident",
               "field_magnitude_over_incident", "absorbed_power_fraction_per_m")
    z_values = np.linspace(0.0, inputs["te_thickness_m"], count, dtype=np.float64)
    n_te = complex(inputs["n_te"], inputs["k_te"])
    k0 = 2.0 * math.pi / inputs["wavelength_m"]
    field = solution["forward_0"] * np.exp(1j * k0 * n_te * z_values) + \
        solution["backward_0"] * np.exp(-1j * k0 * n_te * z_values)
    density = absorption_density_per_m(z_values, solution["forward_0"], solution["backward_0"],
                                       n_te, inputs["wavelength_m"], inputs["n_glass"])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for z, e, q in zip(z_values, field, density, strict=True):
            writer.writerow({"z_m": float(z), "z_nm": float(z * 1e9),
                             "field_real_over_incident": float(e.real), "field_imag_over_incident": float(e.imag),
                             "field_magnitude_over_incident": float(abs(e)),
                             "absorbed_power_fraction_per_m": float(q)})


def _measurement_payload(inputs: dict[str, Any], config_path: Path, register_path: Path,
                         register_rows: list[dict[str, str]], profile_path: Path) -> dict[str, Any]:
    n_in, n_out = inputs["n_glass"], inputs["n_air"]
    n_te = complex(inputs["n_te"], inputs["k_te"])
    wavelength, thickness = inputs["wavelength_m"], inputs["te_thickness_m"]
    coated = single_layer_coefficients(n_in, n_te, n_out, thickness, wavelength)
    bare = _bare_coefficients(n_in, n_out, thickness, wavelength)
    coat_power = power_metrics(coated["r"], coated["t"], n_in, n_out)
    bare_power = power_metrics(bare["r"], bare["t"], n_in, n_out)
    order_64, order_128 = inputs["absorption_quadrature_orders"]
    a64 = integrate_absorption(thickness, coated["forward_0"], coated["backward_0"], n_te, wavelength, n_in, order_64)
    a128 = integrate_absorption(thickness, coated["forward_0"], coated["backward_0"], n_te, wavelength, n_in, order_128)
    energy_error = validate_absorption_balance(coated["r"], coated["t"], n_in, n_out, a128,
                                               inputs["power_tolerance"])
    independent_r, independent_t = solve_boundary_continuity_independently(n_in, n_te, n_out, thickness, wavelength)
    dr, dt = abs(coated["r"] - independent_r), abs(coated["t"] - independent_t)
    checks = {
        "independent_boundary_continuity": {"passed": dr <= inputs["field_tolerance"] and dt <= inputs["field_tolerance"],
                                            "delta_r": dr, "delta_t": dt},
        "absorption_balance": {"passed": energy_error <= inputs["power_tolerance"], "A_flux": coat_power["A_flux"],
                               "A_integral_128": a128, "absolute_error": energy_error},
        "quadrature_convergence": {"passed": abs(a64 - a128) <= inputs["power_tolerance"], "A_integral_64": a64,
                                   "A_integral_128": a128, "absolute_difference": abs(a64 - a128)},
    }
    tau_coated, tau_bare = math.sqrt(n_out / n_in) * coated["t"], math.sqrt(n_out / n_in) * bare["t"]
    phase_difference = float(np.angle(coated["t"] * np.conj(bare["t"])))
    return {
        "schema": "lightgen_direct_facet_linear_measurement.v1",
        "status": "valid" if all(item["passed"] for item in checks.values()) else "invalid_harness",
        "evidence_tier": "diagnostic_only",
        "model_scope": "literature-only planar local glass-bearing region; not a measured SMF100 device",
        "config_path": str(config_path.resolve()), "parameter_register_path": str(register_path.resolve()),
        "parameter_register_row_count": len(register_rows), "model_inputs": inputs,
        "conventions": {"time_factor": "exp(-i omega t)", "layer_index": "N=n+ik, k>=0",
                        "propagation_direction": "glass -> Te -> air", "incidence": "normal, scalar, isotropic",
                        "reference_planes": {"input": "glass/Te plane z=0", "output": "z=d at Te/air boundary"},
                        "bare_comparator": "glass/air interface with air propagation from z=0 to z=d",
                        "power_normalized_transmission": "tauE=sqrt(n_air/n_glass)*tE; |tauE|^2=T",
                        "unmodeled": ["adhesion layer", "oxide", "roughness", "facet gap", "scattering", "fiber modal profile"]},
        "coated": {"rE": _complex_pair(coated["r"]), "tE": _complex_pair(coated["t"]),
                   "tauE": _complex_pair(tau_coated), **coat_power, "A_integral_64": a64, "A_integral_128": a128},
        "bare": {"rE": _complex_pair(bare["r"]), "tE": _complex_pair(bare["t"]),
                 "tauE": _complex_pair(tau_bare), **bare_power, "A_integral_128": 0.0},
        "coated_to_bare": {"transmitted_power_ratio": coat_power["T"] / bare_power["T"],
                            "phase_difference_rad": phase_difference,
                            "phase_difference_deg": math.degrees(phase_difference), "same_reference_plane": True},
        "independent_boundary_solution": {"rE": _complex_pair(independent_r), "tE": _complex_pair(independent_t),
                                          "delta_r": dr, "delta_t": dt},
        "checks": checks, "field_profile_csv": str(profile_path),
        "interpretation": "The result is a local linear literature reference only; it is not a target-fiber prediction or CW nonlinear model.",
    }


def render_saved_report(payload: dict[str, Any]) -> str:
    coated, bare, delta = payload["coated"], payload["bare"], payload["coated_to_bare"]
    inputs, checks = payload["model_inputs"], payload["checks"]
    decision = "valid local linear reference" if payload["status"] == "valid" else "invalid measurement harness"
    scope_statement = (
        "This valid calculation supports a quantitative *local linear literature reference* only. It does not establish a deposited SMF100 film, its cross-gap continuity, support, actual complex transmission, or any nonlinear response. Do not impose a minimum transmittance from this calculation."
        if payload["status"] == "valid" else
        "This measurement is invalid as a scientific result. Do not use its optical values to support a material or device conclusion; repair the calculation harness and repeat the check."
    )
    lines = [
        "# G1 direct-facet Te local linear diagnostic", "",
        f"Decision: **{decision}** under the declared literature-only assumptions.", "",
        "## Fixed model", "",
        f"- Direction: glass → Te → air; vacuum wavelength {inputs['wavelength_m'] * 1e9:.12g} nm.",
        f"- Te: {inputs['sample_identity']}, {inputs['te_thickness_m'] * 1e9:.12g} nm; interpolated `n={inputs['n_te']:.12g}`, `k={inputs['k_te']:.12g}`.",
        f"- Glass approximation: fused-silica Sellmeier `n={inputs['n_glass']:.12g}`, `k=0` by explicit assumption.",
        "- The literature data associate the 30 nm Te film with a planar SiO₂ substrate; this calculation uses the target-direction local boundary order, glass → Te → air.",
        "- The local plane-wave result omits guided-mode profile, fiber-array gaps/support, oxide, adhesion layers, roughness and scattering.", "",
        "## Bare facet and coated facet", "",
        "| Case | R | T | A (flux) | `rE` | `tE` at z=d |", "|---|---:|---:|---:|---|---|",
        f"| Bare glass/air | {bare['R']:.12g} | {bare['T']:.12g} | {bare['A_flux']:.12g} | {bare['rE']['real']:.12g}{bare['rE']['imag']:+.12g}i | {bare['tE']['real']:.12g}{bare['tE']['imag']:+.12g}i |",
        f"| 30 nm Te | {coated['R']:.12g} | {coated['T']:.12g} | {coated['A_flux']:.12g} | {coated['rE']['real']:.12g}{coated['rE']['imag']:+.12g}i | {coated['tE']['real']:.12g}{coated['tE']['imag']:+.12g}i |", "",
        f"Coated/bare transmitted-power ratio: **{delta['transmitted_power_ratio']:.12g}**. Same-reference-plane phase difference: **{delta['phase_difference_rad']:.12g} rad** ({delta['phase_difference_deg']:.12g}°).", "",
        f"Integrated Te dissipation: {coated['A_integral_128']:.12g}; difference from `1-R-T`: {checks['absorption_balance']['absolute_error']:.3g}. Gauss-Legendre 64/128 order difference: {checks['quadrature_convergence']['absolute_difference']:.3g}.", "",
        "## Verification consumed from the saved calculation", "",
        f"- Independent four-boundary continuity solve: `Δr={payload['independent_boundary_solution']['delta_r']:.3g}`, `Δt={payload['independent_boundary_solution']['delta_t']:.3g}`.",
        f"- Result status: `{payload['status']}`; checks: " + ", ".join(f"{name}={'passed' if value['passed'] else 'failed'}" for name, value in checks.items()) + ".",
        "- Power-normalized complex transmission is stored separately from electric-field `tE`; no absolute physical field intensity is inferred.", "",
        "## Scope and next step", "",
        scope_statement, "",
        "Next: close the target array's full facet/support layer sequence and continuous-film evidence; then plan same-sample 532 nm CW linear calibration. Measure nonlinear response only after the target sample and operating conditions are identified. G1 remains the fixed deployment endpoint; this diagnostic authorizes no forward-chain integration or training.", "",
        f"Absorption profile: `{payload['field_profile_csv']}`.",
    ]
    return "\n".join(lines) + "\n"


def run_measurement(config_path: str | Path, output_dir: str | Path) -> int:
    _, inputs, register_path, register_rows = _load_inputs(config_path)
    out = Path(output_dir)
    preflight_path = out / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("A successful preflight from this entrypoint is required before measurement")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("model_identity") != _identity(inputs):
        raise ValueError("Preflight did not pass for the current explicit model inputs")
    artifact_names = ("linear_diagnostic.json", "bare_vs_coated.csv", "te_field_absorption_profile.csv", "report.md")
    for name in artifact_names:
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnostic artifact: {out / name}")
    profile_path = out / "te_field_absorption_profile.csv"
    payload = _measurement_payload(inputs, Path(config_path), register_path, register_rows, profile_path)
    solution = single_layer_coefficients(inputs["n_glass"], complex(inputs["n_te"], inputs["k_te"]),
                                         inputs["n_air"], inputs["te_thickness_m"], inputs["wavelength_m"])
    _save_profile(profile_path, inputs, solution, inputs["profile_samples"])
    _write_json_new(out / "linear_diagnostic.json", payload)
    coated, bare = payload["coated"], payload["bare"]
    comparison_rows = []
    for case, result, integral in (("bare_glass_air", bare, 0.0),
                                   ("glass_te_air_30nm", coated, coated["A_integral_128"])):
        comparison_rows.append({
            "case": case, "rE_real": result["rE"]["real"], "rE_imag": result["rE"]["imag"],
            "tE_real": result["tE"]["real"], "tE_imag": result["tE"]["imag"],
            "tauE_real": result["tauE"]["real"], "tauE_imag": result["tauE"]["imag"],
            "R": result["R"], "T": result["T"], "A_flux": result["A_flux"],
            "A_integral_128": integral, "reference_plane_m": inputs["te_thickness_m"],
        })
    _save_comparison(out / "bare_vs_coated.csv", comparison_rows)
    saved_payload = json.loads((out / "linear_diagnostic.json").read_text(encoding="utf-8"))
    _write_new(out / "report.md", render_saved_report(saved_payload))
    return 0 if payload["status"] == "valid" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "measure"))
    parser.add_argument("--config", required=True, help="Explicit YAML research config")
    parser.add_argument("--output-dir", required=True, help="Explicit diagnostic-only output directory")
    args = parser.parse_args(argv)
    try:
        if args.mode == "preflight":
            return run_preflight(args.config, args.output_dir)
        return run_measurement(args.config, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"diagnostic error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
