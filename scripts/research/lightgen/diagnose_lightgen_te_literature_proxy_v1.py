"""Compute a literature-only 532 nm glass -> Te -> air thin-film proxy."""

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

import diagnose_lightgen_direct_facet_linear_v1 as linear


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_STATES = ("solid_25C", "quasi_liquid_80C")
SWEEP_COLUMNS = (
    "thickness_nm", "state", "temperature_c", "phase", "n", "n_digitization_half_width",
    "k", "k_digitization_half_width", "rE_real", "rE_imag", "tE_exit_real", "tE_exit_imag",
    "tE_z50_real", "tE_z50_imag", "tauE_real", "tauE_imag", "R", "T", "A_flux",
    "A_integral_64", "A_integral_128", "T_over_bare", "phase_vs_bare_rad",
    "phase_vs_bare_deg", "R_digitization_min", "R_digitization_max", "T_digitization_min",
    "T_digitization_max", "A_digitization_min", "A_digitization_max",
    "phase_vs_bare_digitization_min_rad", "phase_vs_bare_digitization_max_rad",
)
STATE_COLUMNS = (
    "thickness_nm", "quantity", "solid_25C", "quasi_liquid_80C", "delta_80C_minus_25C",
    "digitization_min", "digitization_max", "unit",
)
PROFILE_COLUMNS = (
    "state", "temperature_c", "phase", "z_nm", "field_real_over_incident",
    "field_imag_over_incident", "field_magnitude_over_incident", "absorbed_power_fraction_per_m",
)


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    _write_new(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_digitized_csv(path: Path) -> list[dict[str, Any]]:
    expected = {
        "state", "temperature_c", "phase", "wavelength_nm", "n", "n_digitization_half_width",
        "k", "k_digitization_half_width", "source_n", "source_k", "extraction_basis",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError("Digitized n/k CSV does not match the expected 11-column contract")
        rows = list(reader)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append({
            **row,
            "temperature_c": int(row["temperature_c"]),
            "wavelength_nm": float(row["wavelength_nm"]),
            "n": float(row["n"]),
            "n_digitization_half_width": float(row["n_digitization_half_width"]),
            "k": float(row["k"]),
            "k_digitization_half_width": float(row["k_digitization_half_width"]),
        })
    return parsed


def load_inputs(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema") != "lightgen_te_literature_proxy.v1":
        raise ValueError("Unsupported or malformed Te literature-proxy config")
    model, material, media, sources, checks = (config[key] for key in ("model", "material", "media", "sources", "checks"))
    if (model["time_factor"] != "exp(-i omega t)" or model["incidence"] != "normal_plane_wave"
            or model["polarization"] != "scalar_isotropic"
            or model["propagation_direction"] != "glass -> Te -> air"):
        raise ValueError("The proxy accepts only the declared normal-incidence convention")
    wavelength_nm = float(model["wavelength_nm"])
    reference_nm = float(model["output_reference_plane_nm"])
    nominal_nm = float(material["nominal_thickness_nm"])
    sweep_start, sweep_stop, sweep_step = (float(value) for value in material["thickness_sweep_nm"])
    if wavelength_nm != 532.0 or reference_nm != 50.0 or nominal_nm != 10.0:
        raise ValueError("This diagnostic is fixed at 532 nm, d=10 nm nominal, and z=50 nm output reference")
    if (sweep_start != 1.0 or sweep_stop != 50.0 or sweep_step != 1.0
            or sweep_stop > reference_nm or sweep_step <= 0):
        raise ValueError("Thickness sweep must be 1–50 nm in 1 nm steps and fit before z=50 nm")
    if media["glass_extinction_coefficient_assumption"] != 0.0 or media["air_refractive_index"] != 1.0:
        raise ValueError("This proxy requires its declared glass-loss and air-index assumptions")
    if sources["paper_doi"] != "10.1002/adma.202511261":
        raise ValueError("The Te parameter identity must remain bound to the declared 2025 paper")
    if tuple(state["id"] for state in material["states"]) != EXPECTED_STATES:
        raise ValueError("The configuration must declare the paper's solid 25 C and quasi-liquid 80 C states")
    digitization_path = _resolve(material["digitization_csv"])
    rows = _read_digitized_csv(digitization_path)
    if tuple(row["state"] for row in rows) != EXPECTED_STATES:
        raise ValueError("Digitization CSV must contain each expected state exactly once and in declared order")
    for expected_state, row in zip(material["states"], rows, strict=True):
        if (expected_state["id"] != row["state"] or expected_state["temperature_c"] != row["temperature_c"]
                or expected_state["phase"] != row["phase"] or row["wavelength_nm"] != wavelength_nm
                or row["source_n"] != expected_state["source_n_figure"]
                or row["source_k"] != expected_state["source_k_figure"]):
            raise ValueError(f"Digitized row identity does not match config for {expected_state['id']}")
        values = (row["n"], row["k"], row["n_digitization_half_width"], row["k_digitization_half_width"])
        if not all(math.isfinite(value) for value in values) or row["n"] <= 0 or row["k"] < 0:
            raise ValueError(f"Nonphysical or nonfinite digitized n/k values for {expected_state['id']}")
        if min(row["n_digitization_half_width"], row["k_digitization_half_width"]) < 0:
            raise ValueError("Digitization half-widths cannot be negative")
    audit_path = digitization_path.with_name("digitization_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "lightgen_te_paper_figure_digitization.v1":
        raise ValueError("Digitization audit is missing or has an unsupported schema")
    audit_records = audit.get("records")
    if not isinstance(audit_records, list) or len(audit_records) != 4:
        raise ValueError("Digitization audit must contain independent extraction records for n and k in each state")
    agreement_tolerance = float(checks["curve_extraction_agreement_tolerance"])
    for record in audit_records:
        for key in ("value", "value_primary", "value_secondary", "digitization_half_width"):
            if not math.isfinite(float(record[key])):
                raise ValueError("Digitization audit contains nonfinite parameter evidence")
        if abs(float(record["value_primary"]) - float(record["value_secondary"])) > agreement_tolerance:
            raise ValueError(f"Independent extraction methods disagree for {record['state']} {record['parameter']}")
        csv_row = next(row for row in rows if row["state"] == record["state"])
        if abs(float(csv_row[record["parameter"]]) - float(record["value"])) > 1e-12:
            raise ValueError("Digitized parameter CSV differs from the image-extraction audit")
    source_paths = {
        "fig1c": _resolve(sources["figure_1c_local_path"]),
        "fig_s2": _resolve(sources["figure_s2_local_path"]),
    }
    if not all(path.is_file() for path in source_paths.values()):
        raise ValueError("Required original paper figure images are missing from the evidence package")
    glass_model = media["glass_model"]
    wavelength_um = wavelength_nm / 1000.0
    n_glass = linear._malitson_n(wavelength_um, glass_model["malitson_sellmeier"])
    quadrature_orders = tuple(int(value) for value in checks["absorption_quadrature_orders"])
    if quadrature_orders != (64, 128):
        raise ValueError("The proxy uses the fixed 64/128-point absorption comparison")
    field_tolerance = float(checks["complex_field_absolute_tolerance"])
    power_tolerance = float(checks["power_and_dissipation_absolute_tolerance"])
    profile_samples = int(config["output"]["profile_samples"])
    if field_tolerance <= 0 or power_tolerance <= 0 or profile_samples < 3:
        raise ValueError("Configured tolerances or profile sample count are invalid")
    return {
        "config": config,
        "config_path": config_file,
        "digitization_path": digitization_path,
        "audit_path": audit_path,
        "source_paths": source_paths,
        "wavelength_nm": wavelength_nm,
        "wavelength_m": wavelength_nm * 1e-9,
        "n_glass": n_glass,
        "n_air": float(media["air_refractive_index"]),
        "thickness_nm": np.arange(sweep_start, sweep_stop + sweep_step / 2.0, sweep_step),
        "nominal_thickness_nm": nominal_nm,
        "reference_plane_nm": reference_nm,
        "sample_identity": str(material["sample_identity"]),
        "states": rows,
        "field_tolerance": field_tolerance,
        "power_tolerance": power_tolerance,
        "quadrature_orders": list(quadrature_orders),
        "profile_samples": profile_samples,
        "source_doi": str(sources["paper_doi"]),
        "glass_source_id": str(glass_model["source_id"]),
    }


def _complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _identity(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_identity": inputs["sample_identity"],
        "paper_doi": inputs["source_doi"],
        "wavelength_nm": inputs["wavelength_nm"],
        "reference_plane_nm": inputs["reference_plane_nm"],
        "nominal_thickness_nm": inputs["nominal_thickness_nm"],
        "thickness_sweep_nm": [float(value) for value in inputs["thickness_nm"]],
        "n_glass": inputs["n_glass"],
        "n_air": inputs["n_air"],
        "states": [
            {key: row[key] for key in ("state", "temperature_c", "phase", "n", "n_digitization_half_width", "k", "k_digitization_half_width")}
            for row in inputs["states"]
        ],
    }


def _bare_at_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    n_in, n_out = inputs["n_glass"], inputs["n_air"]
    wavelength = inputs["wavelength_m"]
    z_ref = inputs["reference_plane_nm"] * 1e-9
    r = complex((n_in - n_out) / (n_in + n_out))
    t_exit = complex(2.0 * n_in / (n_in + n_out))
    t_ref = t_exit * np.exp(2j * math.pi * n_out * z_ref / wavelength)
    metrics = linear.power_metrics(r, t_ref, n_in, n_out)
    return {"r": r, "t_exit": t_exit, "t_ref": complex(t_ref), "metrics": metrics}


def _stack_at_reference(inputs: dict[str, Any], n_te: complex, thickness_nm: float) -> dict[str, Any]:
    n_in, n_out = inputs["n_glass"], inputs["n_air"]
    wavelength = inputs["wavelength_m"]
    thickness_m = thickness_nm * 1e-9
    z_ref = inputs["reference_plane_nm"] * 1e-9
    if thickness_m > z_ref:
        raise ValueError("Te thickness extends beyond the shared output reference plane")
    exit_solution = linear.single_layer_coefficients(n_in, n_te, n_out, thickness_m, wavelength)
    air_propagation = np.exp(2j * math.pi * n_out * (z_ref - thickness_m) / wavelength)
    t_ref = exit_solution["t"] * air_propagation
    metrics = linear.power_metrics(exit_solution["r"], t_ref, n_in, n_out)
    return {
        **exit_solution,
        "t_exit": exit_solution["t"],
        "t_ref": complex(t_ref),
        "metrics": metrics,
    }


def _phase_difference(first: complex, second: complex) -> float:
    return float(np.angle(first / second))


def _align_angle(angle: float, reference: float) -> float:
    return float(reference + np.angle(np.exp(1j * (angle - reference))))


def _integrated_absorption(inputs: dict[str, Any], solution: dict[str, Any], n_te: complex, thickness_nm: float, order: int) -> float:
    return linear.integrate_absorption(
        thickness_nm * 1e-9, solution["forward_0"], solution["backward_0"], n_te,
        inputs["wavelength_m"], inputs["n_glass"], order,
    )


def run_preflight(config_path: str | Path, output_dir: str | Path) -> int:
    inputs = load_inputs(config_path)
    n_in, n_out = inputs["n_glass"], inputs["n_air"]
    wavelength = inputs["wavelength_m"]
    bare = _bare_at_reference(inputs)
    order_64, order_128 = inputs["quadrature_orders"]
    checks: list[dict[str, Any]] = []
    max_delta_r = 0.0
    max_delta_t = 0.0
    max_energy_error = 0.0
    max_quadrature_delta = 0.0
    for state in inputs["states"]:
        n_te = complex(state["n"], state["k"])
        stack = _stack_at_reference(inputs, n_te, inputs["nominal_thickness_nm"])
        closed = linear.single_layer_coefficients(
            n_in, n_te, n_out, inputs["nominal_thickness_nm"] * 1e-9, wavelength
        )
        independent_r, independent_t = linear.solve_boundary_continuity_independently(
            n_in, n_te, n_out, inputs["nominal_thickness_nm"] * 1e-9, wavelength
        )
        delta_r, delta_t_exit = abs(closed["r"] - independent_r), abs(closed["t"] - independent_t)
        max_delta_r = max(max_delta_r, delta_r)
        max_delta_t = max(max_delta_t, delta_t_exit)
        a64 = _integrated_absorption(inputs, stack, n_te, inputs["nominal_thickness_nm"], order_64)
        a128 = _integrated_absorption(inputs, stack, n_te, inputs["nominal_thickness_nm"], order_128)
        energy_error = linear.validate_absorption_balance(
            stack["r"], stack["t_ref"], n_in, n_out, a128, inputs["power_tolerance"]
        )
        max_energy_error = max(max_energy_error, energy_error)
        q_delta = abs(a64 - a128)
        max_quadrature_delta = max(max_quadrature_delta, q_delta)
        checks.append({
            "check": f"independent_boundary_continuity_{state['state']}",
            "passed": delta_r <= inputs["field_tolerance"] and delta_t_exit <= inputs["field_tolerance"],
            "delta_r": delta_r, "delta_t_at_film_exit": delta_t_exit,
        })
        checks.append({
            "check": f"passive_absorption_and_quadrature_{state['state']}",
            "passed": a128 > 0 and energy_error <= inputs["power_tolerance"] and q_delta <= inputs["power_tolerance"],
            "A_flux": stack["metrics"]["A_flux"], "A_integral_64": a64,
            "A_integral_128": a128, "energy_error": energy_error, "quadrature_delta": q_delta,
        })
        lossless = _stack_at_reference(
            inputs, complex(state["n"], 0.0), inputs["nominal_thickness_nm"]
        )
        checks.append({
            "check": f"lossless_control_{state['state']}",
            "passed": abs(lossless["metrics"]["A_flux"]) <= inputs["power_tolerance"]
            and abs(lossless["metrics"]["R"] + lossless["metrics"]["T"] - 1.0) <= inputs["power_tolerance"],
            **lossless["metrics"],
        })
    zero = _stack_at_reference(inputs, complex(inputs["states"][0]["n"], inputs["states"][0]["k"]), 0.0)
    zero_delta_r = abs(zero["r"] - bare["r"])
    zero_delta_t = abs(zero["t_ref"] - bare["t_ref"])
    zero_phase = abs(_phase_difference(zero["t_ref"], bare["t_ref"]))
    checks.append({
        "check": "zero_thickness_is_bare_interface_at_z50",
        "passed": zero_delta_r <= inputs["field_tolerance"] and zero_delta_t <= inputs["field_tolerance"]
        and zero_phase <= inputs["field_tolerance"],
        "delta_r": zero_delta_r, "delta_t_z50": zero_delta_t, "phase_delta_rad": zero_phase,
    })
    wrong_t_power = abs(bare["t_ref"]) ** 2
    wrong_metrics = {"R": bare["metrics"]["R"], "T": wrong_t_power,
                     "A_flux": 1.0 - bare["metrics"]["R"] - wrong_t_power}
    wrong_normalization_rejected = False
    try:
        linear.validate_power_balance(wrong_metrics, 0.0, inputs["power_tolerance"])
    except ValueError:
        wrong_normalization_rejected = True
    checks.append({
        "check": "missing_flux_index_factor_is_rejected",
        "passed": wrong_normalization_rejected,
        "incorrect_T": wrong_t_power, "correct_T": bare["metrics"]["T"],
    })
    checks.append({
        "check": "digitization_methods_agree_and_source_images_exist",
        "passed": len(inputs["states"]) == 2 and all(path.is_file() for path in inputs["source_paths"].values()),
        "state_count": len(inputs["states"]), "source_image_count": len(inputs["source_paths"]),
        "digitization_csv": str(inputs["digitization_path"]), "audit_json": str(inputs["audit_path"]),
    })
    passed = all(item["passed"] for item in checks)
    payload = {
        "schema": "lightgen_te_literature_proxy_preflight.v1",
        "status": "passed" if passed else "invalid_harness",
        "evidence_tier": "diagnostic_only",
        "model_identity": _identity(inputs),
        "config_path": str(Path(config_path).resolve()),
        "checks": checks,
        "check_count": len(checks),
        "max_independent_field_delta": max(max_delta_r, max_delta_t),
        "max_energy_balance_error": max_energy_error,
        "max_quadrature_difference": max_quadrature_delta,
    }
    out = Path(output_dir)
    _write_json_new(out / "preflight.json", payload)
    report = [
        "# Te literature-proxy preflight", "", f"Status: **{payload['status']}**", "",
        f"Model: `{inputs['sample_identity']}`, 532 nm, 10 nm nominal, z=50 nm.",
        "Checks exercise the same thin-film producer and reject an incorrect power normalization.",
    ]
    _write_new(out / "preflight_report.md", "\n".join(report) + "\n")
    return 0 if passed else 2


def _corner_values(state: dict[str, Any]) -> list[tuple[float, float]]:
    dn, dk = state["n_digitization_half_width"], state["k_digitization_half_width"]
    candidates = []
    for n_sign in (-1.0, 1.0):
        for k_sign in (-1.0, 1.0):
            n, k = state["n"] + n_sign * dn, state["k"] + k_sign * dk
            if n <= 0 or k < 0:
                raise ValueError("Digitization interval includes a nonphysical Te index")
            candidates.append((n, k))
    return candidates


def _sweep_row(inputs: dict[str, Any], state: dict[str, Any], thickness_nm: float,
               bare: dict[str, Any]) -> dict[str, Any]:
    n_te = complex(state["n"], state["k"])
    solution = _stack_at_reference(inputs, n_te, thickness_nm)
    metrics = solution["metrics"]
    a64 = _integrated_absorption(inputs, solution, n_te, thickness_nm, 64)
    a128 = _integrated_absorption(inputs, solution, n_te, thickness_nm, 128)
    energy_error = linear.validate_absorption_balance(
        solution["r"], solution["t_ref"], inputs["n_glass"], inputs["n_air"], a128, inputs["power_tolerance"]
    )
    if abs(a64 - a128) > inputs["power_tolerance"]:
        raise ValueError("64/128-point absorption quadrature does not converge")
    corners: list[dict[str, float]] = []
    central_phase = _phase_difference(solution["t_ref"], bare["t_ref"])
    for n, k in _corner_values(state):
        corner = _stack_at_reference(inputs, complex(n, k), thickness_nm)
        corner_phase = _align_angle(_phase_difference(corner["t_ref"], bare["t_ref"]), central_phase)
        corners.append({
            "R": corner["metrics"]["R"], "T": corner["metrics"]["T"],
            "A": corner["metrics"]["A_flux"], "phase": corner_phase,
        })
    tau = math.sqrt(inputs["n_air"] / inputs["n_glass"]) * solution["t_ref"]
    return {
        "thickness_nm": thickness_nm, "state": state["state"],
        "temperature_c": state["temperature_c"], "phase": state["phase"],
        "n": state["n"], "n_digitization_half_width": state["n_digitization_half_width"],
        "k": state["k"], "k_digitization_half_width": state["k_digitization_half_width"],
        "rE_real": solution["r"].real, "rE_imag": solution["r"].imag,
        "tE_exit_real": solution["t_exit"].real, "tE_exit_imag": solution["t_exit"].imag,
        "tE_z50_real": solution["t_ref"].real, "tE_z50_imag": solution["t_ref"].imag,
        "tauE_real": tau.real, "tauE_imag": tau.imag,
        "R": metrics["R"], "T": metrics["T"], "A_flux": metrics["A_flux"],
        "A_integral_64": a64, "A_integral_128": a128,
        "T_over_bare": metrics["T"] / bare["metrics"]["T"],
        "phase_vs_bare_rad": central_phase,
        "phase_vs_bare_deg": math.degrees(central_phase),
        "R_digitization_min": min(item["R"] for item in corners),
        "R_digitization_max": max(item["R"] for item in corners),
        "T_digitization_min": min(item["T"] for item in corners),
        "T_digitization_max": max(item["T"] for item in corners),
        "A_digitization_min": min(item["A"] for item in corners),
        "A_digitization_max": max(item["A"] for item in corners),
        "phase_vs_bare_digitization_min_rad": min(item["phase"] for item in corners),
        "phase_vs_bare_digitization_max_rad": max(item["phase"] for item in corners),
        "_energy_error": energy_error,
        "_solution": solution,
    }


def _comparison_rows(rows: list[dict[str, Any]], thickness_nm: float) -> list[dict[str, Any]]:
    at_thickness = {row["state"]: row for row in rows if row["thickness_nm"] == thickness_nm}
    solid, liquid = (at_thickness[state] for state in EXPECTED_STATES)
    quantities = (
        ("R", "R", "fraction"), ("T", "T", "fraction"), ("A", "A_flux", "fraction"),
        ("phase_vs_bare_rad", "phase_vs_bare_rad", "rad"),
    )
    result = []
    for label, key, unit in quantities:
        delta = liquid[key] - solid[key]
        if key == "phase_vs_bare_rad":
            delta = _phase_difference(
                complex(liquid["tE_z50_real"], liquid["tE_z50_imag"]),
                complex(solid["tE_z50_real"], solid["tE_z50_imag"]),
            )
            liquid_angles = (liquid["phase_vs_bare_digitization_min_rad"],
                             liquid["phase_vs_bare_digitization_max_rad"])
            solid_angles = (solid["phase_vs_bare_digitization_min_rad"],
                            solid["phase_vs_bare_digitization_max_rad"])
            corners = [
                _align_angle(liquid_angle - solid_angle, delta)
                for liquid_angle in liquid_angles for solid_angle in solid_angles
            ]
            low, high = min(corners), max(corners)
        else:
            low = liquid[f"{key.split('_')[0]}_digitization_min"] - solid[f"{key.split('_')[0]}_digitization_max"]
            high = liquid[f"{key.split('_')[0]}_digitization_max"] - solid[f"{key.split('_')[0]}_digitization_min"]
        result.append({
            "thickness_nm": thickness_nm, "quantity": label,
            "solid_25C": solid[key], "quasi_liquid_80C": liquid[key],
            "delta_80C_minus_25C": delta, "digitization_min": low,
            "digitization_max": high, "unit": unit,
        })
    return result


def _state_delta_sweep(rows: list[dict[str, Any]], thicknesses: list[float]) -> list[dict[str, Any]]:
    result = []
    for thickness_nm in thicknesses:
        at_thickness = {row["state"]: row for row in rows if row["thickness_nm"] == thickness_nm}
        solid, liquid = (at_thickness[state] for state in EXPECTED_STATES)
        for quantity, key, low_key, high_key in (
            ("delta_T", "T", "T_digitization_min", "T_digitization_max"),
            ("delta_R", "R", "R_digitization_min", "R_digitization_max"),
            ("delta_A", "A_flux", "A_digitization_min", "A_digitization_max"),
        ):
            result.append({
                "thickness_nm": thickness_nm, "quantity": quantity,
                "solid_25C": solid[key], "quasi_liquid_80C": liquid[key],
                "delta_80C_minus_25C": liquid[key] - solid[key],
                "digitization_min": liquid[low_key] - solid[high_key],
                "digitization_max": liquid[high_key] - solid[low_key], "unit": "fraction",
            })
        solid_phase = complex(solid["tE_z50_real"], solid["tE_z50_imag"])
        liquid_phase = complex(liquid["tE_z50_real"], liquid["tE_z50_imag"])
        delta_phase = _phase_difference(liquid_phase, solid_phase)
        # Propagate each state corner interval around its saved central phase.
        solid_phase_corners = (solid["phase_vs_bare_digitization_min_rad"], solid["phase_vs_bare_digitization_max_rad"])
        liquid_phase_corners = (liquid["phase_vs_bare_digitization_min_rad"], liquid["phase_vs_bare_digitization_max_rad"])
        corner_deltas = [
            _align_angle(liquid_value - solid_value, delta_phase)
            for liquid_value in liquid_phase_corners for solid_value in solid_phase_corners
        ]
        result.append({
            "thickness_nm": thickness_nm, "quantity": "delta_phase_80C_minus_25C",
            "solid_25C": solid["phase_vs_bare_rad"], "quasi_liquid_80C": liquid["phase_vs_bare_rad"],
            "delta_80C_minus_25C": delta_phase, "digitization_min": min(corner_deltas),
            "digitization_max": max(corner_deltas), "unit": "rad",
        })
    return result


def _save_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_profiles(path: Path, inputs: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    nominal_rows = {row["state"]: row for row in rows if row["thickness_nm"] == inputs["nominal_thickness_nm"]}
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROFILE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for state in inputs["states"]:
            result = nominal_rows[state["state"]]["_solution"]
            n_te = complex(state["n"], state["k"])
            thickness_m = inputs["nominal_thickness_nm"] * 1e-9
            z_values = np.linspace(0.0, thickness_m, inputs["profile_samples"], dtype=np.float64)
            k0 = 2.0 * math.pi / inputs["wavelength_m"]
            field = result["forward_0"] * np.exp(1j * k0 * n_te * z_values) + \
                result["backward_0"] * np.exp(-1j * k0 * n_te * z_values)
            density = linear.absorption_density_per_m(
                z_values, result["forward_0"], result["backward_0"], n_te,
                inputs["wavelength_m"], inputs["n_glass"],
            )
            for z_nm, e, q in zip(z_values * 1e9, field, density, strict=True):
                writer.writerow({
                    "state": state["state"], "temperature_c": state["temperature_c"], "phase": state["phase"],
                    "z_nm": float(z_nm), "field_real_over_incident": float(e.real),
                    "field_imag_over_incident": float(e.imag), "field_magnitude_over_incident": float(abs(e)),
                    "absorbed_power_fraction_per_m": float(q),
                })


def _measurement_payload(inputs: dict[str, Any], config_path: Path, rows: list[dict[str, Any]],
                         state_deltas: list[dict[str, Any]], profile_path: Path) -> dict[str, Any]:
    bare = _bare_at_reference(inputs)
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    nominal = [row for row in public_rows if row["thickness_nm"] == inputs["nominal_thickness_nm"]]
    comparison = _comparison_rows(public_rows, inputs["nominal_thickness_nm"])
    status = "valid" if all(
        row["R"] >= 0 and row["T"] >= 0 and row["A_flux"] >= 0
        and abs(row["R"] + row["T"] + row["A_flux"] - 1.0) <= inputs["power_tolerance"]
        for row in public_rows
    ) else "invalid_measurement"
    return {
        "schema": "lightgen_te_literature_proxy_measurement.v1",
        "status": status,
        "evidence_tier": "diagnostic_only",
        "scientific_question": "What local linear R/T/A and thermally switched optical-state modulation follow from this paper's digitized 532 nm n/k proxy?",
        "model_scope": "literature-only normal-incidence local planar glass -> Te -> air model; not a measured sputtered film, SMF100 array, or nonlinear response",
        "config_path": str(config_path.resolve()),
        "digitization_csv": str(inputs["digitization_path"]),
        "digitization_audit_json": str(inputs["audit_path"]),
        "model_identity": _identity(inputs),
        "conventions": {
            "time_factor": "exp(-i omega t)",
            "Te_index": "N=n+ik, k>=0",
            "propagation_direction": "glass -> Te -> air",
            "approximation": "normal plane wave; scalar isotropic; smooth uniform layer; no scattering",
            "input_reference_plane": "z=0 at glass/Te interface",
            "output_reference_plane": "z=50 nm in air for every thickness and bare comparator",
            "power_normalized_transmission": "tauE=sqrt(n_air/n_glass)*tE; |tauE|^2=T",
            "glass": "Malitson fused silica n at 532 nm; zero absorption is a research assumption, not actual SMF characterization",
            "unmodeled": ["fiber modal profile", "array support/gaps", "adhesion layer", "oxide", "roughness", "scattering", "angle/polarization spread"],
        },
        "paper_sample_identity_and_transfer_limits": {
            "paper_sample": "elemental Te thermal-chromic film; paper reports solid 25 C and quasi-liquid 80 C optical states",
            "paper_process": "thermal evaporation for the reported 10 nm resonator film; not magnetron sputtering",
            "paper_resonator_stack": "15 nm Ti / 150 nm Au / 10 nm Te / 50 nm SiO2 in the reported Au-mirror device",
            "ellipsometry_sample_stack": "complete substrate and layer sequence for the n/k ellipsometry specimen is not identified in the retrieved article text",
            "transfer": "paper optical constants are applied as an explicitly labeled proxy to a different glass/Te/air local stack; 1–50 nm sweep is not a claim that the paper measured these direct-facet samples",
            "optical_state_change": "temperature-driven state comparison only; does not establish optical-intensity-induced nonlinearity, CW self-action, a switching threshold, or response time",
        },
        "bare_reference": {
            "rE": _complex_pair(bare["r"]), "tE_z50": _complex_pair(bare["t_ref"]),
            "R": bare["metrics"]["R"], "T": bare["metrics"]["T"],
            "A_flux": bare["metrics"]["A_flux"], "reference_plane_nm": inputs["reference_plane_nm"],
        },
        "nominal_10nm_rows": nominal,
        "nominal_state_comparison": comparison,
        "sweep_row_count": len(public_rows),
        "state_delta_sweep": state_deltas,
        "field_absorption_profile_csv": str(profile_path.resolve()),
        "result_note": "Digitization half-widths are propagated corner envelopes; they are not experimental error bars.",
        "stop_point": "No G1 forward-chain integration, training, fabrication, purchase, or physical measurement is authorized by this local proxy.",
    }


def render_saved_report(payload: dict[str, Any]) -> str:
    state_rows = payload["nominal_10nm_rows"]
    comparison = {row["quantity"]: row for row in payload["nominal_state_comparison"]}
    rows = {row["state"]: row for row in state_rows}
    delta_t_rows = [row for row in payload["state_delta_sweep"] if row["quantity"] == "delta_T"]
    delta_phase_rows = [row for row in payload["state_delta_sweep"]
                        if row["quantity"] == "delta_phase_80C_minus_25C"]
    max_delta_t = max(delta_t_rows, key=lambda row: abs(row["delta_80C_minus_25C"]))
    max_delta_phase = max(delta_phase_rows, key=lambda row: abs(row["delta_80C_minus_25C"]))
    lines = [
        "# 532 nm Te literature-proxy thin-film calculation", "",
        f"Status: **{payload['status']}** · evidence tier: `{payload['evidence_tier']}`", "",
        "## Fixed model and source boundary", "",
        f"- Paper DOI: [{payload['model_identity']['paper_doi']}](https://doi.org/{payload['model_identity']['paper_doi']}); n from Fig. 1c and k from Fig. S2.",
        f"- Sample label: `{payload['model_identity']['sample_identity']}`; 532 nm; local glass → Te → air; smooth uniform film; normal incidence.",
        "- Paper states are solid at 25 °C and quasi-liquid at 80 °C. The paper used thermal evaporation; its 10 nm resonator layer sits in a Ti/Au/Te/SiO₂ structure. This calculation applies the paper's figure values to a different glass/Te/air stack.",
        "- The ellipsometry specimen's complete substrate/layer stack is not specified in the retrieved article text. Digitized values are raster-derived, and the paper does not report experimental uncertainty with these plotted curves.",
        "- The thickness sweep is a direct-facet model design sweep using those optical constants. It is not a set of samples measured by the paper or a prediction for the project's sputtered Te.",
        "- The 25/80 °C comparison is thermally driven optical-state contrast, not optical-intensity nonlinearity, a ReLU transfer curve, or a CW switching threshold.", "",
        "## Figure-digitized 532 nm inputs", "",
        "| Paper state | n | Raster/axis half-width | k | Raster/axis half-width |",
        "|---|---:|---:|---:|---:|",
    ]
    for state in payload["model_identity"]["states"]:
        lines.append(f"| {state['phase']} ({state['temperature_c']} °C) | {state['n']:.9g} | ±{state['n_digitization_half_width']:.4g} | {state['k']:.9g} | ±{state['k_digitization_half_width']:.4g} |")
    lines.extend([
        "", "## Nominal 10 nm result at shared z = 50 nm reference", "",
        "| State | R | T | A = 1−R−T | T / bare | phase vs bare |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for state_id in EXPECTED_STATES:
        row = rows[state_id]
        lines.append(f"| {row['phase']} ({row['temperature_c']} °C) | {row['R']:.8g} | {row['T']:.8g} | {row['A_flux']:.8g} | {row['T_over_bare']:.8g} | {row['phase_vs_bare_deg']:.5g}° |")
    lines.extend([
        "", f"- 80 °C minus 25 °C transmission: **{comparison['T']['delta_80C_minus_25C']:.8g}** (corner envelope {comparison['T']['digitization_min']:.8g} to {comparison['T']['digitization_max']:.8g}).",
        f"- 80 °C minus 25 °C phase at z=50 nm: **{math.degrees(comparison['phase_vs_bare_rad']['delta_80C_minus_25C']):.5g}°** (digitization corner envelope {math.degrees(comparison['phase_vs_bare_rad']['digitization_min']):.5g}° to {math.degrees(comparison['phase_vs_bare_rad']['digitization_max']):.5g}°).",
        f"- 10 nm state changes in reflection and absorption: ΔR={comparison['R']['delta_80C_minus_25C']:.8g}; ΔA={comparison['A']['delta_80C_minus_25C']:.8g}.",
        f"- Across the sampled 1–50 nm grid, the largest absolute central transmission change is ΔT={max_delta_t['delta_80C_minus_25C']:.8g} at {max_delta_t['thickness_nm']:.0f} nm; its digitization-corner envelope is {max_delta_t['digitization_min']:.8g} to {max_delta_t['digitization_max']:.8g}.",
        f"- The largest absolute central phase-state difference is {math.degrees(max_delta_phase['delta_80C_minus_25C']):.5g}° at {max_delta_phase['thickness_nm']:.0f} nm. The 2 nm grid point is a numerical extremum, not a deposition recommendation; continuity at that thickness is unproven.",
        "- All sweep rows store `rE`, exit-plane `tE`, z=50 nm `tE`, power-normalized `tauE`, R/T/A, absorption-integral checks, bare-normalized transmission, and digitization envelopes.",
        "- The 10 nm profile stores complex field and absorbed-power density relative to incident field/power; no unknown physical intensity is inferred.", "",
        "## Interpretation and next action", "",
        "This result quantifies the local linear optical effect implied by the 2025 paper's two temperature-defined states under a declared planar interface approximation. It supports a design-space comparison only. It does not establish a photoinduced nonlinearity at 532 nm or determine whether the target sputtered film will reproduce these constants.",
        "Next: use the existing G1 endpoint as the fixed deployment reference; obtain the target sputtered film's same-sample 532 nm n,k and thickness, then measure the target facet's R/T/complex field before planning intensity-dependent response tests. No G1 integration, training, fabrication, procurement, or physical measurement is included here.", "",
        "The calculation artifacts are linked in `measurement.json`, `thickness_sweep.csv`, `state_comparison_at_10nm.csv`, and `te_field_absorption_profiles_10nm.csv`.",
    ])
    return "\n".join(lines) + "\n"


def run_measurement(config_path: str | Path, output_dir: str | Path) -> int:
    inputs = load_inputs(config_path)
    out = Path(output_dir)
    preflight_path = out / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("A successful preflight from this entrypoint is required before measure")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("model_identity") != _identity(inputs):
        raise ValueError("Preflight does not pass for the current explicit literature-proxy inputs")
    artifact_names = (
        "measurement.json", "thickness_sweep.csv", "state_comparison_at_10nm.csv",
        "state_delta_vs_thickness.csv", "te_field_absorption_profiles_10nm.csv", "report.md",
        "transmission_vs_thickness.png", "transmission_vs_thickness.svg",
        "transmission_vs_thickness.pdf", "transmission_vs_thickness.tiff",
    )
    existing = [name for name in artifact_names if (out / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite diagnostic artifacts: {', '.join(existing)}")
    bare = _bare_at_reference(inputs)
    rows = [
        _sweep_row(inputs, state, float(thickness_nm), bare)
        for thickness_nm in inputs["thickness_nm"] for state in inputs["states"]
    ]
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    thicknesses = [float(value) for value in inputs["thickness_nm"]]
    state_deltas = _state_delta_sweep(public_rows, thicknesses)
    profile_path = out / "te_field_absorption_profiles_10nm.csv"
    _save_profiles(profile_path, inputs, rows)
    _save_csv(out / "thickness_sweep.csv", SWEEP_COLUMNS, public_rows)
    _save_csv(out / "state_comparison_at_10nm.csv", STATE_COLUMNS,
              _comparison_rows(public_rows, inputs["nominal_thickness_nm"]))
    _save_csv(out / "state_delta_vs_thickness.csv", STATE_COLUMNS, state_deltas)
    payload = _measurement_payload(inputs, Path(config_path), rows, state_deltas, profile_path)
    _write_json_new(out / "measurement.json", payload)
    saved_payload = json.loads((out / "measurement.json").read_text(encoding="utf-8"))
    _write_new(out / "report.md", render_saved_report(saved_payload))
    render_plot_from_saved_results(saved_payload, out / "thickness_sweep.csv", out)
    return 0 if payload["status"] == "valid" else 2


def render_plot_from_saved_results(payload: dict[str, Any], sweep_csv: Path, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with sweep_csv.open("r", encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    colors = {"solid_25C": "#0072B2", "quasi_liquid_80C": "#D55E00"}
    labels = {"solid_25C": "Solid, 25 °C", "quasi_liquid_80C": "Quasi-liquid, 80 °C"}
    fig, ax = plt.subplots(figsize=(7.1, 2.75), constrained_layout=True)
    for state_id in EXPECTED_STATES:
        selected = [row for row in data if row["state"] == state_id]
        x = np.asarray([float(row["thickness_nm"]) for row in selected])
        y = np.asarray([float(row["T"]) for row in selected])
        low = np.asarray([float(row["T_digitization_min"]) for row in selected])
        high = np.asarray([float(row["T_digitization_max"]) for row in selected])
        linestyle = "-" if state_id == "solid_25C" else "--"
        label = f"{labels[state_id]} (band: digitization bounds only)"
        ax.fill_between(x, low, high, color=colors[state_id], alpha=0.16, linewidth=0)
        ax.plot(x, y, color=colors[state_id], linestyle=linestyle, linewidth=1.7, label=label)
    ax.set_title("532 nm local glass–Te–air transmission proxy")
    ax.set_xlabel("Te thickness (nm)")
    ax.set_ylabel("Transmitted power fraction, T")
    ax.set_xlim(1.0, 50.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks((1, 10, 20, 30, 40, 50))
    ax.grid(axis="y", color="#D8DDE2", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    # Reflectance and absorptance remain fully saved in the sweep CSV and tables.
    fig.savefig(output_dir / "transmission_vs_thickness.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "transmission_vs_thickness.svg", bbox_inches="tight")
    fig.savefig(output_dir / "transmission_vs_thickness.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "transmission_vs_thickness.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "measure"))
    parser.add_argument("--config", required=True, help="Explicit YAML research-proxy config")
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
