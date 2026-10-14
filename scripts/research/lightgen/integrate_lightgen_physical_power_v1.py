"""Run the bounded physical-power integration gates for the LightGen 2+2 path."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
    GaussianFiberGeometry,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)
from ldm_project.v2.application.research.lightgen.material_nonlinearity import (
    MaterialNonlinearity,
    MaterialParameters,
)
from ldm_project.v2.application.research.lightgen.material_nonlinearity_diagnostic import (
    nonlinear_coverage,
    validate_material_scenarios,
)
from ldm_project.v2.application.research.lightgen.physical_power_diagnostics import (
    calibrate_source,
    encoder_power_radius_diagnostic,
    fiber_reference_plane_diagnostic,
    gaussian,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import clear_production_transfer_cache


GATE_TOLERANCES = {
    "absolute_power_relative": 1.0e-6,
    "free_space_power_relative": 1.0e-4,
    "window_relative_change": 1.0e-4,
    "fiber_passivity_relative": 1.0e-5,
    "identity_material_relative": 1.0e-6,
    "matched_gaussian_eta_absolute": 1.0e-4,
}
INPUT_IMAGE_NAMES = ("uniform", "vertical_bar", "diagonal", "center_disk")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize evidence value {type(value).__name__}")


def _power_stats(values: torch.Tensor) -> dict[str, Any]:
    data = values.detach().to(torch.float64)
    quantile = torch.quantile(data.flatten(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=data.device))
    return {
        "min": float(data.min()),
        "mean": float(data.mean()),
        "median": float(quantile[1]),
        "p10": float(quantile[0]),
        "p50": float(quantile[1]),
        "p90": float(quantile[2]),
        "max": float(data.max()),
        "per_sample": data.tolist(),
    }


def _controls() -> tuple[torch.Tensor, list[str]]:
    axis = torch.linspace(-1.0, 1.0, 28, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    images = torch.stack(
        (
            torch.ones_like(xx),
            ((xx >= -0.18) & (xx <= 0.18)).float(),
            (torch.abs(xx - yy) <= 0.16).float(),
            ((xx.square() + yy.square()) <= 0.32).float(),
        ),
        dim=0,
    )[:, None]
    return images, list(INPUT_IMAGE_NAMES)


def _physical_config(config: dict[str, Any], *, input_power_w: float | None = None,
                     layer_transmission: float | None = None,
                     global_fiber_geometry: bool = True,
                     material: MaterialParameters | None = None) -> LightGenPhysicalOpticsConfig:
    optics = config["optics"]
    fiber = config["fiber"]
    return LightGenPhysicalOpticsConfig(
        input_power_w=float(optics["input_power_w"] if input_power_w is None else input_power_w),
        layer_power_transmission=float(
            optics["layer_power_transmission"] if layer_transmission is None else layer_transmission
        ),
        global_fiber_geometry=global_fiber_geometry,
        fiber_pitch_m=float(fiber["pitch_m"]),
        fiber_cladding_diameter_m=float(fiber["cladding_diameter_m"]),
        fiber_mfd_m=float(fiber["mfd_m"]),
        fiber_length_m=float(fiber["length_m"]),
        fiber_attenuation_db_per_km=float(fiber["attenuation_db_per_km"]),
        material=material,
    )


def _run_forward(grid_size: int, physical: LightGenPhysicalOpticsConfig | None,
                 images: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=grid_size, physical_optics=physical)
    ).to(device).eval()
    with torch.no_grad():
        result = runtime.forward_from_image(images.to(device))
    if result.ols_input_field is None:
        raise RuntimeError("image forward did not return the OLS reference-plane field")
    trace = result.physical_power_trace
    if physical is not None and trace is None:
        raise RuntimeError("physical forward omitted its power trace")
    tensors: dict[str, torch.Tensor] = {
        "ols_field": result.ols_input_field.tensor.detach(),
        "generator_input": result.generator_input_field.tensor.detach(),
        "camera": result.raw_camera.tensor.detach(),
        "coefficients": result.z_ols.coefficients.detach(),
        "encoder_layer_power": result.crop_statistics["encoder_layer_power"].detach(),
        "decoder_layer_power": result.crop_statistics["generator_layer_power"].detach(),
    }
    if trace is not None:
        tensors["power_trace"] = {key: value.detach() for key, value in trace.items()}
    metadata = {
        "grid_size": grid_size,
        "phase_initialization": "canonical_zero_phase",
        "encoder_phase_shape": list(runtime.encoder.phase.shape),
        "decoder_phase_shape": list(runtime.generator.phase.shape),
        "ols_geometry": runtime.ols.mode_identity(),
        "physical_trace_available": trace is not None,
    }
    del result, runtime
    return metadata, tensors


def _center_crop(values: torch.Tensor, size: int) -> torch.Tensor:
    height, width = values.shape[-2:]
    if height < size or width < size:
        raise ValueError("comparison ROI exceeds the field plane")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return values[..., y0:y0 + size, x0:x0 + size]


def _relative_l1(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.to(torch.float64)
    b = right.to(torch.float64)
    return float((a - b).abs().sum() / b.abs().sum().clamp_min(1.0e-30))


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.to(torch.complex128 if torch.is_complex(left) else torch.float64)
    b = right.to(torch.complex128 if torch.is_complex(right) else torch.float64)
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1.0e-30))


def _window_convergence(config: dict[str, Any], image: torch.Tensor, device: torch.device) -> dict[str, Any]:
    sizes = (2048, 3072, 4096)
    runs: dict[int, dict[str, Any]] = {}
    for size in sizes:
        physical = _physical_config(config)
        metadata, values = _run_forward(size, physical, image, device)
        field = values["ols_field"][0, 0]
        trace = values["power_trace"]
        roi = _center_crop(field, 2048).cpu()
        runs[size] = {
            "metadata": metadata,
            "roi_complex_field": roi,
            "roi_intensity": roi.abs().square(),
            "p_before_fiber_w": float(trace["P_before_fiber_W"][0]),
            "eta_each": trace["fiber_coupling_efficiency_each"][0].cpu(),
            "eta_total": float(trace["fiber_coupling_efficiency_total"][0]),
            "generator_input": values["generator_input"][0].cpu(),
        }
        del values
        del field, trace, metadata
        clear_production_transfer_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    comparisons: dict[str, Any] = {}
    for smaller, larger in ((2048, 3072), (2048, 4096), (3072, 4096)):
        a, b = runs[smaller], runs[larger]
        comparisons[f"{smaller}_vs_{larger}"] = {
            "relative_change_vs_larger_window": {
                "fiber_plane_intensity_common_roi_l1": _relative_l1(a["roi_intensity"], b["roi_intensity"]),
                "fiber_plane_complex_field_common_roi_l2": _relative_l2(a["roi_complex_field"], b["roi_complex_field"]),
                "P_before_fiber_relative": abs(a["p_before_fiber_w"] - b["p_before_fiber_w"])
                / max(abs(b["p_before_fiber_w"]), 1.0e-30),
                "each_fiber_coupling_eta_l2": _relative_l2(a["eta_each"], b["eta_each"]),
                "total_coupling_relative": abs(a["eta_total"] - b["eta_total"])
                / max(abs(b["eta_total"]), 1.0e-30),
                "output_field_l2": _relative_l2(a["generator_input"], b["generator_input"]),
            },
            "P_before_fiber_w": {"smaller": a["p_before_fiber_w"], "larger": b["p_before_fiber_w"]},
            "eta_total": {"smaller": a["eta_total"], "larger": b["eta_total"]},
        }
    maxima = {
        key: max(pair["relative_change_vs_larger_window"][key] for pair in comparisons.values())
        for key in comparisons["2048_vs_4096"]["relative_change_vs_larger_window"]
    }
    tolerance = GATE_TOLERANCES["window_relative_change"]
    return {
        "grids": list(sizes),
        "pixel_pitch_m": 2.0e-6,
        "physical_width_m": [size * 2.0e-6 for size in sizes],
        "comparisons": comparisons,
        "maximum_relative_change": maxima,
        "predeclared_tolerance": tolerance,
        "status": "PASS" if all(value <= tolerance for value in maxima.values()) else "FAIL",
        "claim_limit": "canonical zero-phase, one synthetic uniform input; no trained-checkpoint phase sampling claim",
    }


def _power_and_material_sweeps(config: dict[str, Any], trace: dict[str, torch.Tensor],
                               coefficients: torch.Tensor, sample_names: list[str]) -> dict[str, Any]:
    optics = config["optics"]
    fiber = config["fiber"]
    material_spec = config["material"]["saturable_absorption"]
    input_scan = [float(item) for item in optics["input_power_scan_w"]]
    transmission_scan = [float(item) for item in optics["layer_power_transmission_scan"]]
    i_sat_scan = [float(item) for item in material_spec["i_sat_w_cm2_scan"]]
    base_input = float(optics["input_power_w"])
    base_transmission = float(optics["layer_power_transmission"])
    base_coeff = coefficients.detach().to(torch.complex128)
    base_power = trace["fiber_coupled_power_W"].detach().to(torch.float64)
    fiber_loss_db = float(fiber["attenuation_db_per_km"]) * float(fiber["length_m"]) / 1000.0
    fiber_power_transmission = 10.0 ** (-fiber_loss_db / 10.0)
    waist = float(fiber["mfd_m"]) / 2.0
    material = MaterialNonlinearity()
    power_scenarios: list[dict[str, Any]] = []
    material_scenarios: list[dict[str, Any]] = []
    for input_power in input_scan:
        for layer_t in transmission_scan:
            scale_power = (input_power / base_input) * (layer_t / base_transmission) ** 2
            each_coupled = base_power * scale_power
            each_fiber_output = each_coupled * fiber_power_transmission
            intensity_m2 = 2.0 * each_fiber_output / (math.pi * waist**2)
            intensity_cm2 = intensity_m2 / 1.0e4
            power_scenarios.append({
                "input_power_w": input_power,
                "layer_power_transmission": layer_t,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "P_input_W_per_sample": trace["P_input_W"].detach().cpu().tolist()
                if input_power == base_input else (trace["P_input_W"].detach().cpu() * (input_power / base_input)).tolist(),
                "P_after_slm_W_per_sample": (trace["P_after_slm_W"].detach().cpu() * (input_power / base_input)).tolist(),
                "P_before_fiber_W_per_sample": (
                    trace["P_before_fiber_W"].detach().cpu()
                    * (input_power / base_input)
                    * (layer_t / base_transmission) ** 2
                ).tolist(),
                "fiber_power_transmission": fiber_power_transmission,
                "fiber_coupled_power_W_per_sample_per_channel": each_coupled.tolist(),
                "P_fiber_total_W_per_sample": each_coupled.sum(dim=1).tolist(),
                "eta_total_per_sample": (
                    each_coupled.sum(dim=1)
                    / (trace["P_before_fiber_W"].detach().cpu().to(torch.float64)
                       * (input_power / base_input) * (layer_t / base_transmission) ** 2)
                ).tolist(),
                "fiber_output_power_W_per_sample_per_channel": each_fiber_output.tolist(),
                "I_peak_W_m2_per_sample_per_channel": intensity_m2.tolist(),
                "I_peak_W_cm2_per_sample_per_channel": intensity_cm2.tolist(),
                "I_peak_W_cm2_distribution": _power_stats(intensity_cm2),
                "decoder_exit_power_linear_scaling_W_per_sample": (
                    trace["P_camera_W"].detach().cpu().to(torch.float64)
                    * (input_power / base_input)
                    * (layer_t / base_transmission) ** 4
                ).tolist(),
                "sample_names": sample_names,
                "sweep_status": "reference_scenario_not_current_device_measurement",
            })
            for i_sat in i_sat_scan:
                ratios = nonlinear_coverage(intensity_cm2.to(torch.float64), i_sat)
                parameters = MaterialParameters(
                    "saturable_absorption",
                    t_ns=float(material_spec["T_ns"]),
                    delta_t=float(material_spec["Delta_T"]),
                    i_sat_w_cm2=i_sat,
                )
                coefficient_scale = math.sqrt(input_power / base_input) * (layer_t / base_transmission)
                fiber_coefficients = base_coeff * coefficient_scale * math.sqrt(fiber_power_transmission)
                material_output = material.forward(
                    fiber_coefficients,
                    intensity_cm2.to(fiber_coefficients.device, dtype=torch.float64),
                    parameters,
                )
                after_each = material_output.abs().square().detach().cpu().to(torch.float64)
                label = material_spec.get("benchmark_labels", {}).get(str(i_sat), None)
                material_scenarios.append({
                    "input_power_w": input_power,
                    "layer_power_transmission": layer_t,
                    "i_sat_w_cm2": i_sat,
                    "material_model": "saturable_absorption",
                    "T_ns_low_intensity_power_transmission": parameters.t_ns,
                    "Delta_T": parameters.delta_t,
                    "benchmark_label": label,
                    "material_parameter_status": config["material"]["saturable_absorption"]["material_parameter_status"],
                    "r_i_per_sample_per_channel": ratios["r_each"],
                    "nonlinear_activation_coverage_per_sample": ratios["per_sample_fraction_nonlinear"],
                    "fraction_linear": ratios["fraction_linear"],
                    "fraction_active": ratios["nonlinear_activation_coverage"],
                    "fraction_saturated": ratios["fraction_saturated"],
                    "ratio_distribution": ratios["batch_statistics"],
                    "ratio_per_sample_statistics": ratios["per_sample_statistics"],
                    "ratio_per_channel_statistics": ratios["per_channel_statistics"],
                    "P_after_material_W_per_sample_per_channel": after_each.tolist(),
                    "P_after_material_total_W_per_sample": after_each.sum(dim=1).tolist(),
                    "passive": bool((after_each <= each_fiber_output.detach().cpu().to(torch.float64) * (1.0 + 1e-10) + 1e-30).all()),
                    "sample_names": sample_names,
                })
    return {
        "power_scenarios": power_scenarios,
        "material_scenarios": material_scenarios,
        "power_scan_w": input_scan,
        "layer_transmission_scan": transmission_scan,
        "i_sat_scan_w_cm2": i_sat_scan,
        "i_sat_special_labels": material_spec.get("benchmark_labels", {}),
        "attenuation_formula": "P_out/P_in = 10^(-alpha_dB_per_km * L_m / 1000 / 10)",
        "SA_formula": "T(I)=T_ns+Delta_T*I/(I+I_sat); E_out=sqrt(T(I))*E_in; phase response disabled",
        "SA_parameter_status": "generic sweep only; no current-device measurement or material identification",
    }


def run(config_path: Path, output_dir: Path, artifact_path: Path, device_name: str) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output_dir}")
    if artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite final artifact: {artifact_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "lightgen_physical_power_proxy.v1":
        raise ValueError("unsupported physical power reference config schema")
    output_dir.mkdir(parents=True, exist_ok=False)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly selected but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)
    images, sample_names = _controls()
    base_input = float(config["optics"]["input_power_w"])
    physical = _physical_config(config)

    # The production 2048 grid carries a small fixed control batch for per-sample distributions.
    main_metadata, main_values = _run_forward(2048, physical, images, device)
    trace = main_values["power_trace"]
    ols = main_values["ols_field"]
    runtime_geometry = main_metadata["ols_geometry"]
    plane = PlaneSpec(
        "integration_global_fiber_plane",
        (2048, 2048),
        (2.0e-6, 2.0e-6),
        532.0e-9,
        float(runtime_geometry["plane_z_m"]),
        "fiber_global",
    )
    geometry = fiber_reference_plane_diagnostic(
        plane,
        trainable_width_m=200.0e-6,
        fiber_pitch_m=physical.fiber_pitch_m,
        cladding_diameter_m=physical.fiber_cladding_diameter_m,
    )
    radii = encoder_power_radius_diagnostic(
        ComplexField(ols, plane, OperationProvenance("integration_encoder_to_fiber", None, None)),
        fiber_pitch_m=physical.fiber_pitch_m,
        cladding_diameter_m=physical.fiber_cladding_diameter_m,
    )
    power_summary = {
        key: _power_stats(value)
        for key, value in (
            ("P_input_W", trace["P_input_W"]),
            ("P_after_slm_W", trace["P_after_slm_W"]),
            ("P_encoder_entrance_W", trace["P_encoder_entrance_W"]),
            ("P_encoder_1_W", trace["P_encoder_1_W"]),
            ("P_encoder_2_W", trace["P_encoder_2_W"]),
            ("P_before_fiber_W", trace["P_before_fiber_W"]),
            ("P_fiber_total_W", trace["P_fiber_total_W"]),
            ("P_fiber_output_total_W", trace["P_fiber_output_total_W"]),
            ("P_decoder_1_W", trace["P_decoder_1_W"]),
            ("P_decoder_2_W", trace["P_decoder_2_W"]),
            ("P_camera_W", trace["P_camera_W"]),
        )
    }
    channel_power = trace["fiber_coupled_power_W"].detach().cpu()
    intensity_cm2 = trace["I_peak_W_cm2"].detach().cpu()
    per_sample_channels = {
        "sample_names": sample_names,
        "complex_latent_real": main_values["coefficients"].real.detach().cpu().tolist(),
        "complex_latent_imag": main_values["coefficients"].imag.detach().cpu().tolist(),
        "fiber_power_W_per_channel": channel_power.tolist(),
        "fiber_coupling_efficiency_per_channel": trace["fiber_coupling_efficiency_each"].detach().cpu().tolist(),
        "I_peak_W_m2_per_channel": trace["I_peak_W_m2"].detach().cpu().tolist(),
        "I_peak_W_cm2_per_channel": intensity_cm2.tolist(),
        "channel_power_distribution_W": _power_stats(channel_power),
        "channel_intensity_distribution_W_cm2": _power_stats(intensity_cm2),
        "sum_channel_power_le_incident": bool((trace["P_fiber_total_W"] <= trace["P_before_fiber_W"] * (1.0 + 1e-5) + 1e-12).all()),
    }

    # Small engineering grid checks production input-power linearity without an optimizer/evaluator.
    single = images[:1]
    scale_meta, scale_base = _run_forward(1024, physical, single, device)
    scale_physical = _physical_config(config, input_power_w=2.0 * base_input)
    _, scale_double = _run_forward(1024, scale_physical, single, device)
    base_trace = scale_base["power_trace"]
    double_trace = scale_double["power_trace"]
    scale_keys = (
        "P_input_W", "P_after_slm_W", "P_encoder_entrance_W", "P_encoder_1_W",
        "P_encoder_2_W", "P_before_fiber_W", "P_fiber_total_W", "P_fiber_output_total_W",
        "P_decoder_1_W", "P_decoder_2_W", "P_camera_W",
    )
    scale_errors = {
        key: _relative_l2(double_trace[key], 2.0 * base_trace[key]) for key in scale_keys
    }
    input_power_error = float(
        (trace["P_input_W"] - base_input).abs().max() / base_input
    )

    # G8 reuses the existing matched Gaussian OLS control on the 4.096 mm physical plane.
    gauss_geometry = GaussianFiberGeometry(
        port_pitch_m=physical.fiber_pitch_m,
        waist_m=physical.fiber_mfd_m / 2.0,
    )
    gauss_operator = GaussianFiberArrayOLS(plane, gauss_geometry).to(device)
    center_y, center_x = gauss_operator.centers_m[44].tolist()
    matched_values = gaussian(
        plane, gauss_geometry.waist_m, x_m=center_x, y_m=center_y
    ).to(device)
    matched_source = calibrate_source(matched_values, plane, 1.0e-3)
    matched_source = ComplexField(
        matched_source.tensor.to(device), plane, matched_source.provenance
    )
    with torch.no_grad():
        matched_coeff = gauss_operator.project_coefficients(matched_source)
    matched_each = matched_coeff.abs().square()
    matched_input_power = float(matched_source.tensor.abs().square().sum() * math.prod(plane.pixel_pitch_m))
    matched_eta = float(matched_each[0, 44] / matched_input_power)
    matched_sum_eta = float(matched_each.sum() / matched_input_power)

    # A3 to A4 identity comparison; A0 remains the pre-calibration legacy baseline.
    a3_meta, a3_values = _run_forward(1024, physical, single, device)
    identity_config = _physical_config(config, material=MaterialParameters("identity"))
    a4_identity_meta, a4_identity_values = _run_forward(1024, identity_config, single, device)
    identity_equal = all(
        torch.equal(a3_values[key].cpu(), a4_identity_values[key].cpu())
        for key in ("generator_input", "camera", "coefficients")
    ) and all(
        torch.equal(a3_values["power_trace"][key].cpu(), a4_identity_values["power_trace"][key].cpu())
        for key in a3_values["power_trace"]
    )
    legacy_meta, legacy_values = _run_forward(1024, None, single, device)
    legacy_camera = legacy_values["camera"].to(torch.float64)
    physical_camera = a3_values["camera"].to(torch.float64)
    legacy_normalized = legacy_camera / legacy_camera.sum().clamp_min(1.0e-30)
    physical_normalized = physical_camera / physical_camera.sum().clamp_min(1.0e-30)
    a0_shape_l1 = _relative_l1(legacy_normalized, physical_normalized)
    a0_raw_relative = _relative_l2(legacy_camera, physical_camera)
    camera_units_comparable = False

    # Material activation is sampled as a generic SA scenario, then the full declared sweep follows.
    sweep = _power_and_material_sweeps(
        config, trace, main_values["coefficients"], sample_names
    )
    material_validation = validate_material_scenarios()
    sa_parameters = MaterialParameters(
        "saturable_absorption",
        t_ns=float(config["material"]["saturable_absorption"]["T_ns"]),
        delta_t=float(config["material"]["saturable_absorption"]["Delta_T"]),
        i_sat_w_cm2=520.0,
    )
    sa_meta, sa_values = _run_forward(
        1024,
        _physical_config(config, material=sa_parameters),
        single,
        device,
    )
    sa_trace = sa_values["power_trace"]
    sa_intensity = sa_trace["I_peak_W_cm2"]
    sa_expected_transmission = sa_parameters.t_ns + sa_parameters.delta_t * sa_intensity / (
        sa_intensity + sa_parameters.i_sat_w_cm2
    )
    sa_expected_power = sa_trace["fiber_output_power_W"] * sa_expected_transmission
    sa_material_relative_error = _relative_l2(
        sa_trace["P_after_material_each_W"], sa_expected_power
    )

    convergence = _window_convergence(config, single, device)
    layer_power = {
        "encoder": main_values["encoder_layer_power"].detach().cpu(),
        "decoder": main_values["decoder_layer_power"].detach().cpu(),
    }
    propagation_relative_errors: dict[str, float] = {}
    transmission_errors: dict[str, float] = {}
    for stack_name, stack_power in layer_power.items():
        input_power = stack_power[..., 0]
        device_power = stack_power[..., 1]
        propagated_power = stack_power[..., 2]
        propagation_relative_errors[stack_name] = float(
            ((propagated_power - device_power).abs() / device_power.abs().clamp_min(1e-30)).max()
        )
        expected_device_power = input_power * physical.layer_power_transmission
        transmission_errors[stack_name] = float(
            ((device_power - expected_device_power).abs() / expected_device_power.abs().clamp_min(1e-30)).max()
        )
    all_centers = geometry["fiber_center_x_m_row_major"] + geometry["fiber_center_y_m_row_major"]
    centers_inside = geometry["all_fiber_centers_inside_global_plane"] and len(all_centers) == 200
    g1 = geometry["status"] == "PASS" and runtime_geometry["plane_shape"] == [2048, 2048]
    g2 = centers_inside
    g3 = convergence["status"] == "PASS"
    g4 = input_power_error <= GATE_TOLERANCES["absolute_power_relative"]
    g5 = max(propagation_relative_errors.values()) <= GATE_TOLERANCES["free_space_power_relative"]
    g6 = max(transmission_errors.values()) <= 1.0e-6
    g7 = per_sample_channels["sum_channel_power_le_incident"]
    g8 = (
        abs(matched_eta - 1.0) <= GATE_TOLERANCES["matched_gaussian_eta_absolute"]
        and matched_sum_eta <= 1.0 + 1.0e-4
    )
    g9 = identity_equal and camera_units_comparable and a0_shape_l1 <= GATE_TOLERANCES["identity_material_relative"]
    g10 = max(scale_errors.values()) <= GATE_TOLERANCES["absolute_power_relative"]
    if convergence["status"] != "PASS":
        sweep["gate_authority"] = "EXPLORATORY_ONLY_OUT_OF_ORDER_G3_FAILED"
        sweep["execution_order_note"] = (
            "This composite attempt calculated the generic SA sweep before publishing G3; "
            "retain the values as exploratory, not as gate-authorized A4.2 acceptance."
        )
        material_validation["gate_authority"] = "unit_equation_check_only_not_A4_2_acceptance"
    gates = {
        "G1_global_fiber_geometry": {"status": "PASS" if g1 else "FAIL", "evidence": "geometry"},
        "G2_all_100_centers_inside_global_plane": {"status": "PASS" if g2 else "FAIL", "centers_inside": centers_inside},
        "G3_window_convergence": {"status": "PASS" if g3 else "FAIL", "evidence": "window_convergence"},
        "G4_absolute_input_power": {"status": "PASS" if g4 else "FAIL", "max_relative_error": input_power_error},
        "G5_free_space_power_conservation": {"status": "PASS" if g5 else "FAIL", "max_relative_errors": propagation_relative_errors},
        "G6_sqrt_layer_power_transmission": {"status": "PASS" if g6 else "FAIL", "relative_errors": transmission_errors},
        "G7_fiber_overlap_passivity": {"status": "PASS" if g7 else "FAIL", "max_total_eta": float(trace["fiber_coupling_efficiency_total"].max())},
        "G8_matched_Gaussian_overlap": {
            "status": "PASS" if g8 else "FAIL", "matched_eta_channel_44": matched_eta,
            "sum_eta": matched_sum_eta, "input_power_w": matched_input_power,
        },
        "G9_identity_material_vs_original_linear_baseline": {
            "status": "PASS" if g9 else "FAIL",
            "A4_identity_equals_A3_physical_exactly": identity_equal,
            "A0_legacy_vs_A3_normalized_camera_l1": a0_shape_l1,
            "A0_legacy_vs_A3_raw_camera_relative_l2": a0_raw_relative,
            "A0_A3_camera_units_comparable": camera_units_comparable,
            "reason": "A4 identity exactly reproduces A3, but original A0 has no W/m^2 camera calibration and uses the compact 200um coupling plane; therefore numerical equivalence to A0 is not established.",
        },
        "G10_no_hidden_renormalization": {
            "status": "PASS" if g10 else "FAIL",
            "relative_errors_under_2x_input_power": scale_errors,
            "source_calibration_boundary": "once before amplitude SLM",
            "downstream_renormalization_provenance": False,
        },
    }
    status = "PASS" if all(item["status"] == "PASS" for item in gates.values()) else "VALID_FAILED"
    report = {
        "schema": "physical_power_integration.v1",
        "date": "2026-10-13",
        "status": status,
        "scope": "small_aligned_2plus2_lightgen_research_runtime_only",
        "device": str(device),
        "device_selection": device_name,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_loaded": False,
        "training_updates": 0,
        "evaluator_calls": 0,
        "dataset_reads": 0,
        "runtime": main_metadata,
        "reference_config": str(config_path),
        "reference_status": config["status"],
        "measurement_evidence_path": str(output_dir / "physical_power_integration.json"),
        "final_artifact_path": str(artifact_path),
        "geometry": geometry,
        "field_power_coverage": radii,
        "absolute_power": power_summary,
        "fiber_channels": per_sample_channels,
        "window_convergence": convergence,
        "free_space_energy_relative_errors": propagation_relative_errors,
        "layer_transmission_relative_errors": transmission_errors,
        "power_and_nonlinearity_sweep": sweep,
        "execution_order_compliance": {
            "status": "DEVIATION_EXPLORATORY_ONLY" if convergence["status"] != "PASS" else "IN_ORDER",
            "note": "The composite diagnostic computed the material sweep before final G3 evaluation; no training ran, and material results do not authorize A4.2 acceptance.",
        },
        "material_formula_validation": material_validation,
        "integrated_material_control": {
            "runtime_metadata": sa_meta,
            "scenario": "generic SA sweep point at 520 W/cm^2; other-wavelength/structure TaP benchmark label only",
            "parameters": {
                "T_ns": sa_parameters.t_ns,
                "Delta_T": sa_parameters.delta_t,
                "I_sat_W_cm2": sa_parameters.i_sat_w_cm2,
                "phase_response": "disabled",
            },
            "actual_post_material_power_each_W": sa_trace["P_after_material_each_W"],
            "expected_power_from_SA_formula_each_W": sa_expected_power,
            "relative_error": sa_material_relative_error,
            "passive": bool((sa_trace["P_after_material_each_W"] <= sa_trace["fiber_output_power_W"] * (1.0 + 1.0e-10) + 1.0e-30).all()),
        },
        "input_power_scaling_control": {
            "grid_size": scale_meta["grid_size"],
            "input_power_ratio": 2.0,
            "relative_error_by_power_stage": scale_errors,
        },
        "matched_gaussian_control": {
            "geometry": {"pitch_m": gauss_geometry.port_pitch_m, "waist_m": gauss_geometry.waist_m},
            "matched_channel_index": 44,
            "matched_channel_eta": matched_eta,
            "total_eta": matched_sum_eta,
            "input_power_w": matched_input_power,
            "model": "existing analytic Gaussian S12 overlap control; not a hardware MFD calibration",
        },
        "identity_and_baseline_control": {
            "A3_metadata": a3_meta,
            "A4_identity_metadata": a4_identity_meta,
            "A0_metadata": legacy_meta,
            "A4_identity_exactly_matches_A3": identity_equal,
            "A0_vs_A3_normalized_camera_l1": a0_shape_l1,
            "A0_vs_A3_raw_camera_relative_l2": a0_raw_relative,
            "camera_units_comparable": camera_units_comparable,
            "interpretation": "the nonlinear identity control passes relative to A3; strict G9 against original A0 cannot pass because A0 has no calibrated camera irradiance units and uses different fiber geometry",
        },
        "gates": gates,
        "conditional_A0_A4_training_smoke": {
            "status": "NOT_RUN_G3_AND_G9_FAILED",
            "reason": "the user-defined all-gates-pass precondition was not met",
            "training_stability": None,
            "reconstruction_or_generation_loss": None,
            "scientific_metrics": None,
        },
        "physical_claim_boundary": [
            "all values are a declared reference scenario and synthetic zero-phase control, not current-device measurement",
            "encoder output is measured at the global 2048x2048 plane after the existing 1.5mm final propagation",
            "fiber output-to-decoder mode basis remains the existing compact 100x100 project basis and is not a measured relay",
            "material response is a static lumped Gaussian peak-intensity approximation; it does not resolve a film over x/y",
            "no trained checkpoint, real MNIST sample, independent generalization, readability or hardware device was measured",
        ],
        "permitted_next_actions": [
            "define a calibrated mapping from the physical A3/A4 irradiance field to the existing reconstruction/evaluator target while preserving the existing loss",
            "specify or measure the fiber-output-to-decoder relay/mode plane before training on the new coordinates",
        ],
        "blocked_actions": [
            "A0-A4 training smoke while G3/G9 is failed",
            "formal long training, evaluator changes, dataset changes, new power/coupling/coverage loss, device/material adoption, or hardware deployment",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "physical_power_integration.json").write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    report = run(args.config, args.output_dir, args.artifact_output, args.device)
    print(json.dumps({"status": report["status"], "gates": report["gates"], "artifact": str(args.artifact_output)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
