"""Conditional power/coverage sweeps using saved, measured component overlaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from ldm_project.v2.application.research.lightgen.material_nonlinearity_diagnostic import (
    MATERIAL_MODELS, MaterialNonlinearity, MaterialParameters, nonlinear_coverage, validate_material_scenarios,
)
from ldm_project.v2.application.research.lightgen.physical_power_diagnostics import no_energy_creation


LASER_POWERS_MW = (0.2493, 1.0, 10.0, 100.0, 300.0)
LAYER_POWER_TRANSMISSIONS = (1.0, 0.95, 0.889, 0.80, 0.70)
COUPLING_SENSITIVITIES = (0.1, 0.3, 0.5, 0.7, 0.9)
I_SAT_W_CM2 = (1e1, 1e2, 520.0, 1e3, 1e4, 1e5, 1e6, 1e7, 3e7)


def load_evidence(diagnostic_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    diagnostic = json.loads((diagnostic_dir / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostic["schema"] != "physical_power_diagnostics.v1" or not diagnostic["numeric_component_checks_passed"]:
        raise ValueError("budget requires executed passing component diagnostics")
    report = json.loads((diagnostic_dir / "absolute_power.json").read_text(encoding="utf-8"))
    payload = torch.load(diagnostic_dir / "component_power_evidence.pt", map_location="cpu", weights_only=True)
    if payload["schema"] != "physical_power_component_evidence.v1" or report["power_unit"] != "W" or report["field_unit"] != "sqrt(W)/m":
        raise ValueError("budget input units or schema are invalid")
    source_power = float(payload["source_power_w"])
    if not math.isfinite(source_power) or source_power <= 0 or source_power != report["source_power_w"]:
        raise ValueError("source calibration is invalid")
    for basis, values in payload["basis_results"].items():
        powers = values["channel_power_w"]
        if powers.shape != (len(payload["profile_ids"]), 100) or powers.tolist() != report["basis_results"][basis]["channel_power_w_each"]:
            raise ValueError("saved channel-power identity differs from diagnostic JSON")
        if not torch.allclose(values["coefficients"].abs().square(), powers, rtol=1e-12, atol=1e-18):
            raise ValueError("channel powers do not come from the saved complex coefficients")
        if not no_energy_creation(values["before_fiber_w"], powers)["passed"]:
            raise ValueError("non-passive channel evidence cannot enter budget")
        if not torch.isfinite(values["peak_mode_density_per_m2"]).all() or (values["peak_mode_density_per_m2"] <= 0).any():
            raise ValueError("invalid mode peak density")
    return report, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_report, payload = load_evidence(args.diagnostic_dir)
    material_validation = validate_material_scenarios()
    if not material_validation["passed"]:
        raise ValueError("material equation diagnostic failed; sweep cannot start")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    material = MaterialNonlinearity()
    with (args.output_dir / "material_validation.json").open("x", encoding="utf-8") as stream:
        json.dump(material_validation, stream, indent=2, allow_nan=False)
    budget_rows = []
    coverage_rows = []
    channel_rows = []
    material_rows = []
    # Toy generic transfer coefficients, separate from every material reference.
    material_assumptions = {"t_ns": 0.70, "delta_t": 0.20, "i_rsa_over_i_sat": 10.0, "phi_max_rad": 0.5,
                            "status": "ARBITRARY_GENERIC_SCENARIO_NOT_MEASURED_MATERIAL"}
    coverage_file = args.output_dir / "coverage_distributions.jsonl"
    with coverage_file.open("x", encoding="utf-8") as distributions:
        for basis, values in payload["basis_results"].items():
            source_coefficients = values["coefficients"].to(torch.complex128)
            for power_mw in LASER_POWERS_MW:
                for transmission in LAYER_POWER_TRANSMISSIONS:
                    scenario_id = f"{basis}/P{power_mw:g}mW/T{transmission:g}"
                    source_scale = (power_mw * 1e-3 / payload["source_power_w"])**0.5
                    # Two encoder layers: each amplitude gets sqrt(T); no rescale after fiber.
                    coefficients = source_coefficients * source_scale * transmission
                    channel_powers = coefficients.abs().square()
                    intensities = channel_powers * values["peak_mode_density_per_m2"][None] / 1e4
                    before_fiber = values["before_fiber_w"] * source_scale**2 * transmission**2
                    passive = no_energy_creation(before_fiber, channel_powers)
                    if not passive["passed"]:
                        raise ValueError("power sweep created modal energy")
                    for sample, profile in enumerate(payload["profile_ids"]):
                        def stage(name: str, count: int) -> float:
                            return float(payload["stage_power_w"][name][sample] * source_scale**2 * transmission**count)
                        budget_rows.append({"scenario_id": scenario_id, "profile": profile, "basis": basis,
                                            "P_laser_mW": power_mw, "T_layer_power": transmission,
                                            "P_input_w": power_mw*1e-3,
                                            "P_after_encoder_1_w": stage("P_after_encoder_1", 1),
                                            "P_after_encoder_2_w": stage("P_after_encoder_2", 2),
                                            "P_before_fiber_w": float(before_fiber[sample]),
                                            "eta_c_model": float(channel_powers[sample].sum()/before_fiber[sample]),
                                            "P_fiber_total_w": float(channel_powers[sample].sum()),
                                            "P_before_nonlinearity_w": float(values["reemitted_power_w"][sample]*source_scale**2*transmission**2),
                                            "P_camera_no_material_lp01_w": stage("P_camera", 4) if basis == "lp01_ports_v1" else "",
                                            "no_energy_creation": passive["status"]})
                        for channel in range(100):
                            channel_rows.append({"scenario_id": scenario_id, "profile": profile, "channel": channel,
                                                 "row": channel//10, "column": channel%10,
                                                 "power_w": float(channel_powers[sample, channel]),
                                                 "peak_intensity_w_cm2": float(intensities[sample, channel])})
                    for i_sat in I_SAT_W_CM2:
                        coverage = nonlinear_coverage(intensities, i_sat)
                        label = ("TaP_other_wavelength_tapered_QD_benchmark_NOT_532nm" if i_sat == 520 else
                                 "user_requested_Te_532nm_high_threshold_reference_UNVERIFIED" if i_sat == 3e7 else "generic_threshold_sweep")
                        distributions.write(json.dumps({"scenario_id": scenario_id, "threshold_label": label, **coverage}, allow_nan=False) + "\n")
                        coverage_rows.append({"scenario_id": scenario_id, "I_sat_w_cm2": i_sat, "threshold_label": label,
                                              **coverage["batch_statistics"], "fraction_linear": coverage["fraction_linear"],
                                              "fraction_nonlinear": coverage["fraction_nonlinear"], "fraction_saturated": coverage["fraction_saturated"],
                                              "nonlinear_activation_coverage": coverage["nonlinear_activation_coverage"]})
                        for model in MATERIAL_MODELS:
                            params = MaterialParameters(model, t_ns=0.70, delta_t=0.20, i_sat_w_cm2=i_sat,
                                                        i_rsa_w_cm2=10*i_sat, phi_max_rad=0.5)
                            output = material.forward(coefficients, intensities, params)
                            each_out = output.abs().square()
                            if (each_out > channel_powers*(1+1e-12)+1e-18).any():
                                raise ValueError("passive material scenario created energy")
                            for sample, profile in enumerate(payload["profile_ids"]):
                                phase_change = torch.angle(output[sample] * coefficients[sample].conj())
                                material_rows.append({"scenario_id": scenario_id, "profile": profile, "model": model,
                                                      "I_sat_w_cm2": i_sat, "threshold_label": label,
                                                      "power_transmission_lumped": float(each_out[sample].sum()/channel_powers[sample].sum()),
                                                      "P_facet_out_lumped_w": float(each_out[sample].sum()),
                                                      "max_abs_phase_change_rad": float(phase_change.abs().max()),
                                                      "spatial_approximation": "per_channel_peak_lumped; no mode-reshaping or local-sheet integration"})
    # Scalar sensitivity is a separate hypothetical budget, never overriding actual overlap.
    sensitivity_rows = [{"P_laser_mW": power, "T_layer_power": t, "eta_c_sensitivity": eta,
                         "P_before_fiber_w": power*1e-3*t**2, "P_fiber_total_w": power*1e-3*t**2*eta,
                         "P_output_four_layers_no_other_loss_w": power*1e-3*t**4*eta,
                         "scope": "HYPOTHETICAL_UNCALIBRATED_FACET_TOTAL_ONLY_NO_EQUAL_CHANNEL_ASSUMPTION"}
                        for power in LASER_POWERS_MW for t in LAYER_POWER_TRANSMISSIONS for eta in COUPLING_SENSITIVITIES]
    tables = {"power_budget.csv": budget_rows, "channel_power.csv": channel_rows, "coverage.csv": coverage_rows,
              "material_response.csv": material_rows, "coupling_sensitivity.csv": sensitivity_rows}
    reread_counts = {}
    for filename, rows in tables.items():
        with (args.output_dir / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (args.output_dir / filename).open(encoding="utf-8", newline="") as stream:
            reread_counts[filename] = len(list(csv.DictReader(stream)))
        if reread_counts[filename] != len(rows):
            raise ValueError("saved budget row count differs")
    coverage_count = 0
    with coverage_file.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if len(item["r_each"]) != len(payload["profile_ids"]) or any(len(row) != 100 for row in item["r_each"]):
                raise ValueError("saved coverage lost channels")
            coverage_count += 1
    if coverage_count != len(coverage_rows):
        raise ValueError("saved coverage scenario count differs")
    report = {"schema": "physical_power_budget.v1", "status": "CONDITIONAL_SWEEP_COMPLETED", "diagnostic_only": True,
              "source_diagnostic_dir": str(args.diagnostic_dir.resolve()), "source_entrypoint_probe": source_report["production_entrypoint_probe"],
              "input_powers_mW": LASER_POWERS_MW, "layer_power_transmissions": LAYER_POWER_TRANSMISSIONS,
              "i_sat_w_cm2": I_SAT_W_CM2, "coupling": "saved component-overlap coefficients; scalar sensitivity separate",
              "material_parameters": material_assumptions, "material_models": MATERIAL_MODELS,
              "material_equations_verified": material_validation["passed"],
              "material_power_approximation": "per-channel peak lumped sensitivity, not integrated physical sheet",
              "mode_and_material_reference_plane": source_report["material_plane_status"],
              "fiber_transport": "I_100; length/attenuation/phase uncalibrated; no additional fiber-loss factor assumed",
              "row_counts_verified": reread_counts, "coverage_distributions_verified": coverage_count,
              "scientific_updates": 0, "evaluator_calls": 0, "training_loss_modified": False,
              "claim_limit": "untrained synthetic illumination and numerical port basis, not saved-model or hardware feasibility"}
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
