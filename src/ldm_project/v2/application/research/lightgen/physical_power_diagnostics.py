"""Zero-update physical-power audit of existing LightGen optical components.

Only synthetic inputs; no evaluator, dataset, checkpoint or optimizer. Explicit
source calibration is a diagnostic boundary, never a training-model change.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import angular_spectrum_propagate, clear_production_transfer_cache
from . import lightgen_2d_semantic_generation_input_ols_alignment as alignment
from .gaussian_fiber_baseline import GaussianFiberArrayOLS, GaussianFiberGeometry
from .lightgen_2d_chen_bogt_complex_ols_training import physical_plane_power


POWER_RTOL = 1e-5
POWER_ATOL_W = 1e-12


def field(values: torch.Tensor, plane: PlaneSpec, label: str) -> ComplexField:
    return ComplexField(values, plane, OperationProvenance(label, None, None))


def gaussian(plane: PlaneSpec, waist_m: float, *, x_m: float = 0, y_m: float = 0, angle_rad: float = 0) -> torch.Tensor:
    if not all(math.isfinite(x) for x in (waist_m, x_m, y_m, angle_rad)) or waist_m <= 0:
        raise ValueError("Gaussian geometry must be finite with positive waist")
    yy = (torch.arange(plane.shape[0], dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[0] - plane.fov_m[0] / 2 + plane.origin_m[0]
    xx = (torch.arange(plane.shape[1], dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[1] - plane.fov_m[1] / 2 + plane.origin_m[1]
    radius = (yy[:, None] - y_m).square() + (xx[None, :] - x_m).square()
    amplitude = torch.exp(-radius / waist_m**2)
    phase = (2 * math.pi / plane.wavelength_m) * math.sin(angle_rad) * xx[None, :]
    return (amplitude * torch.exp(1j * phase)).unsqueeze(0).unsqueeze(0)


def calibrate_source(values: torch.Tensor, plane: PlaneSpec, power_w: float) -> ComplexField:
    """Set an incident source profile once to sqrt(W)/m on the declared plane."""
    if not math.isfinite(power_w) or power_w <= 0:
        raise ValueError("source power must be finite positive W")
    incoming = field(values, plane, "diagnostic_uncalibrated_source_profile")
    norm = physical_plane_power(incoming)
    if (norm <= 0).any():
        raise ValueError("source profile must have positive integral")
    multiplier = (power_w / norm).sqrt().reshape(-1, *([1] * (values.ndim - 1)))
    return field(values * multiplier, plane, "diagnostic_source_calibration_sqrt_W_per_m")


def no_energy_creation(before_w: torch.Tensor, each_w: torch.Tensor) -> dict[str, object]:
    """Acceptance uses actual overlap-produced channel powers, never renormalizes."""
    if before_w.ndim != 1 or each_w.shape != (before_w.shape[0], 100):
        raise ValueError("no-energy check requires [B] input and [B,100] output powers")
    if not torch.isfinite(before_w).all() or not torch.isfinite(each_w).all() or (before_w <= 0).any() or (each_w < 0).any():
        raise ValueError("invalid physical channel power measurement")
    total = each_w.sum(1)
    passed = bool((total <= before_w * (1 + POWER_RTOL) + POWER_ATOL_W).all())
    return {"status": "PASS" if passed else "FAIL", "passed": passed,
            "max_total_over_input": float((total / before_w).max()),
            "excess_w_each_sample": (total - before_w).tolist(),
            "rtol": POWER_RTOL, "atol_w": POWER_ATOL_W}


def array_support(width_m: float, pitch_m: float, diameter_m: float) -> dict[str, object]:
    if not all(math.isfinite(x) and x > 0 for x in (width_m, pitch_m, diameter_m)):
        raise ValueError("array support dimensions must be finite positive metres")
    extent = 9 * pitch_m + diameter_m
    margin = (width_m - extent) / 2
    return {"window_width_m": width_m, "array_center_span_m": 9 * pitch_m,
            "array_outer_width_m": extent, "margin_m": {name: margin for name in ("left", "right", "top", "bottom")},
            "status": "FAIL" if margin < 0 else "PASS", "small_margin_warning": 0 <= margin < diameter_m / 2}


def fiber_reference_plane_diagnostic(
    plane: PlaneSpec,
    *,
    trainable_width_m: float,
    fiber_pitch_m: float,
    cladding_diameter_m: float,
) -> dict[str, object]:
    """Check a centered physical 10x10 array against the declared global plane."""
    if any(not math.isfinite(value) or value <= 0.0 for value in (
        trainable_width_m, fiber_pitch_m, cladding_diameter_m
    )):
        raise ValueError("fiber reference-plane dimensions must be finite positive metres")
    y_axis = [(index - 4.5) * fiber_pitch_m + plane.origin_m[0] for index in range(10)]
    x_axis = [(index - 4.5) * fiber_pitch_m + plane.origin_m[1] for index in range(10)]
    fiber_center_x = [x for _row in range(10) for x in x_axis]
    fiber_center_y = [y for y in y_axis for _column in range(10)]
    y_min = plane.origin_m[0] - plane.fov_m[0] / 2.0
    y_max = plane.origin_m[0] + plane.fov_m[0] / 2.0
    x_min = plane.origin_m[1] - plane.fov_m[1] / 2.0
    x_max = plane.origin_m[1] + plane.fov_m[1] / 2.0
    outer_y_min = min(y_axis) - cladding_diameter_m / 2.0
    outer_y_max = max(y_axis) + cladding_diameter_m / 2.0
    outer_x_min = min(x_axis) - cladding_diameter_m / 2.0
    outer_x_max = max(x_axis) + cladding_diameter_m / 2.0
    margins = {
        "left": outer_x_min - x_min,
        "right": x_max - outer_x_max,
        "top": outer_y_min - y_min,
        "bottom": y_max - outer_y_max,
    }
    centers_inside = all(x_min <= value <= x_max for value in fiber_center_x) and all(
        y_min <= value <= y_max for value in fiber_center_y
    )
    outer_extent = 9.0 * fiber_pitch_m + cladding_diameter_m
    geometry_passed = centers_inside and min(margins.values()) >= 0.0
    return {
        "trainable_width_m": trainable_width_m,
        "simulation_width_m": list(plane.fov_m),
        "simulation_physical_boundary_m": {"x": [x_min, x_max], "y": [y_min, y_max]},
        "fiber_rows": 10,
        "fiber_columns": 10,
        "fiber_pitch_m": fiber_pitch_m,
        "cladding_diameter_m": cladding_diameter_m,
        "fiber_center_span_m": 9.0 * fiber_pitch_m,
        "fiber_outer_extent_m": outer_extent,
        "fiber_center_x_m_row_major": fiber_center_x,
        "fiber_center_y_m_row_major": fiber_center_y,
        "fiber_center_min_max_m": {
            "x": [min(fiber_center_x), max(fiber_center_x)],
            "y": [min(fiber_center_y), max(fiber_center_y)],
        },
        "left_margin_m": margins["left"],
        "right_margin_m": margins["right"],
        "top_margin_m": margins["top"],
        "bottom_margin_m": margins["bottom"],
        "all_fiber_centers_inside_global_plane": centers_inside,
        "status": "PASS" if geometry_passed else "FAIL",
        "cladding_pitch_independent": True,
    }


def encoder_power_radius_diagnostic(
    field_at_fiber: ComplexField,
    *,
    fiber_pitch_m: float,
    cladding_diameter_m: float,
) -> dict[str, object]:
    """Measure radial energy quantiles and energy inside the array outer square."""
    values = field_at_fiber.tensor
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3 or not torch.is_complex(values) or not torch.isfinite(values).all():
        raise ValueError("fiber-plane field must be finite complex [B,H,W] or [B,1,H,W]")
    dy, dx = field_at_fiber.plane.pixel_pitch_m
    height, width = field_at_fiber.plane.shape
    y = (torch.arange(height, dtype=values.real.dtype, device=values.device) + 0.5) * dy - field_at_fiber.plane.fov_m[0] / 2.0 + field_at_fiber.plane.origin_m[0]
    x = (torch.arange(width, dtype=values.real.dtype, device=values.device) + 0.5) * dx - field_at_fiber.plane.fov_m[1] / 2.0 + field_at_fiber.plane.origin_m[1]
    radius = torch.sqrt(y[:, None].square() + x[None, :].square()).reshape(-1)
    pixel_power = values.abs().square().reshape(values.shape[0], -1) * (dy * dx)
    order = torch.argsort(radius)
    ordered_radius = radius.index_select(0, order)
    cumulative_power = pixel_power.index_select(1, order).cumsum(dim=1)
    total_power = pixel_power.sum(dim=1)
    if torch.any(total_power <= 0.0):
        raise ValueError("fiber-plane radial power requires positive total optical power")
    radii = {}
    for fraction in (0.90, 0.95, 0.99):
        indices = (cumulative_power >= total_power[:, None] * fraction).to(torch.int64).argmax(dim=1)
        radii[f"r{int(fraction * 100)}_m"] = ordered_radius.index_select(0, indices).tolist()
    half_width = (9.0 * fiber_pitch_m + cladding_diameter_m) / 2.0
    inside = (y[:, None].abs() <= half_width) & (x[None, :].abs() <= half_width)
    inside_power = (pixel_power.reshape(values.shape[0], height, width) * inside).sum(dim=(-2, -1))
    fraction_inside = inside_power / total_power
    return {
        "total_power_W": total_power.tolist(),
        "power_radii_m": radii,
        "fiber_array_half_width_m": half_width,
        "power_fraction_inside_fiber_array_extent": fraction_inside.tolist(),
        "power_fraction_outside_fiber_array_extent": (1.0 - fraction_inside).tolist(),
        "array_extent_definition": "axis-aligned outer square including independently configured cladding diameter",
    }


@torch.no_grad()
def geometry_diagnostic() -> dict[str, object]:
    runtime = alignment.LightGen2DSemanticGenerationInputOLSAlignment()
    identity = runtime.ols.mode_identity()
    active_width = runtime.ols_plane.fov_m[1]
    padded_width = runtime.encoder.propagation_shape[1] * runtime.ols_plane.pixel_pitch_m[1]
    return {
        "wavelength_m": runtime.ols_plane.wavelength_m,
        "trainable_mask_shapes": {"encoder": list(runtime.encoder.phase.shape), "decoder": list(runtime.generator.phase.shape)},
        "trainable_mask_width_m": active_width,
        "propagation_grid": list(runtime.encoder.propagation_shape), "propagation_grid_width_m": padded_width,
        "coupling_grid_width_m": active_width, "actual_numerical_mode_identity": identity,
        "lp01_center_margin_m": (active_width - 9 * identity["port_center_pitch_m"]) / 2,
        "gaussian_center_margin_m": (active_width - 9 * GaussianFiberGeometry().port_pitch_m) / 2,
        "nominal_125um_array_at_coupling_plane": array_support(active_width, 125e-6, 125e-6),
        "nominal_125um_array_at_transient_pad": array_support(padded_width, 125e-6, 125e-6),
        "status": "FAIL", "reason": "nominal facet array cannot fit the actual coupling plane; pad is not that plane",
        "parameter_status": "PROJECT_GEOMETRY_UNMEASURED",
    }


@torch.no_grad()
def power_conservation_diagnostic() -> dict[str, object]:
    pitch = alignment.OLS_PIXEL_PITCH_M
    size = alignment.LightGen2DAlignmentRuntimeConfig().propagation_grid_size
    plane = PlaneSpec("audit_free_space", (size, size), (pitch, pitch), alignment.WAVELENGTH_M, 0, "diagnostic")
    source = calibrate_source(gaussian(plane, 40e-6), plane, 1e-3)
    powers = [float(physical_plane_power(source)[0])]
    current = angular_spectrum_propagate(source, alignment.LAYER_SPACING_M, "audit_free_1")
    powers.append(float(physical_plane_power(current)[0]))
    coordinates = (torch.arange(size, dtype=torch.float64) - (size-1)/2) * pitch
    mask_phase = 0.2 * (coordinates[:, None].square() + coordinates[None, :].square()) / (40e-6)**2
    modulated = field(current.tensor * torch.exp(1j * mask_phase), current.plane, "diagnostic_unit_phase_mask")
    powers.append(float(physical_plane_power(modulated)[0]))
    for index in range(3):
        current = angular_spectrum_propagate(modulated if index == 0 else current, alignment.LAYER_SPACING_M, f"audit_free_{index + 2}")
        powers.append(float(physical_plane_power(current)[0]))
    relative_error = abs(powers[-1] / powers[0] - 1)
    # Independent Parseval/power summation on the actual production result.
    numpy_power = float(np.sum(np.abs(current.tensor.numpy())**2, dtype=np.float64) * pitch**2)
    independent_power_error = abs(numpy_power - powers[-1]) / powers[0]
    del source, current, modulated, mask_phase
    clear_production_transfer_cache()

    # Same physical source and ROI at two padded sizes, not a surrogate phase stack.
    roi_results = []
    convergence_powers = []
    for grid_size in (size, size + size // 2):
        convergence_plane = PlaneSpec(f"convergence_{grid_size}", (grid_size, grid_size), (pitch, pitch), alignment.WAVELENGTH_M, 0, "diagnostic")
        beam = gaussian(convergence_plane, 20e-6)
        propagated = angular_spectrum_propagate(field(beam, convergence_plane, "same_gaussian_profile"), alignment.LAYER_SPACING_M, f"convergence_out_{grid_size}")
        start = (grid_size - 100) // 2
        roi_results.append(propagated.tensor[..., start:start+100, start:start+100].clone())
        convergence_powers.append(float(physical_plane_power(propagated)[0]))
        del beam, propagated
        clear_production_transfer_cache()
    convergence_error = float(torch.linalg.vector_norm(roi_results[1] - roi_results[0]) / torch.linalg.vector_norm(roi_results[1]))

    # Deliberately inadequate window: energy conservation can hide cyclic wrap.
    wrap = []
    for grid_size in (256, size):
        test_plane = PlaneSpec(f"wrap_{grid_size}", (grid_size, grid_size), (pitch, pitch), alignment.WAVELENGTH_M, 0, "diagnostic")
        beam = calibrate_source(gaussian(test_plane, 20e-6, x_m=200e-6, angle_rad=math.radians(3)), test_plane, 1e-3)
        output = angular_spectrum_propagate(beam, 2e-3, f"wrap_output_{grid_size}")
        intensity = output.tensor.abs().square()
        wrap.append({"grid_size": grid_size, "width_m": grid_size * pitch,
                     "relative_power_error": abs(float(physical_plane_power(output)[0]) / 1e-3 - 1),
                     "left_half_power_fraction": float(intensity[..., :grid_size//2].sum() / intensity.sum())})
        del beam, output, intensity
        clear_production_transfer_cache()
    wrap_detected = wrap[0]["left_half_power_fraction"] > 0.8 and wrap[1]["left_half_power_fraction"] < 0.01
    return {"status": "PASS" if relative_error < POWER_RTOL and convergence_error < 1e-4 and wrap_detected else "FAIL",
            "power_unit": "W", "input_assumption": "synthetic 1mW calibrated Gaussian, no aperture or material loss",
            "stage_power_w": powers, "relative_power_error": relative_error,
            "independent_numpy_power_relative_error": independent_power_error,
            "pad_convergence_grids": [size, size + size//2], "pad_convergence_roi_relative_complex_error": convergence_error,
            "pad_convergence_power_integrals": convergence_powers,
            "wrap_negative_control": wrap, "wrap_detected_despite_power_conservation": wrap_detected,
            "claim_limit": "Gaussian zero-update checks only; arbitrary trained-phase sampling accuracy is not established"}


@torch.no_grad()
def fiber_overlap_diagnostic() -> dict[str, object]:
    runtime = alignment.LightGen2DSemanticGenerationInputOLSAlignment()
    plane = runtime.ols_plane
    lp01 = runtime.ols.operator
    mode_index = 44
    lp01_input = calibrate_source(lp01.input_modes[mode_index].to(torch.complex128)[None, None], plane, 1e-3)
    lp01_each = lp01.transmit_coefficients(lp01.project_coefficients(lp01_input)).abs().square()
    lp01_eta = float(lp01_each[0, mode_index] / physical_plane_power(lp01_input)[0])
    geometry = GaussianFiberGeometry()
    operator = GaussianFiberArrayOLS(plane, geometry)
    center_y, center_x = operator.centers_m[mode_index].tolist()
    wf = geometry.waist_m
    cases = [
        ("matched", 0, 0, 0, wf), ("x_offset", wf, 0, 0, wf),
        ("y_offset", 0, wf, 0, wf), ("angle_offset", 0, 0, math.radians(1), wf),
        ("wrong_waist", 0, 0, 0, 2*wf),
    ]
    rows = []
    maximum_reference_error = 0.0
    for name, offset_x, offset_y, angle, wb in cases:
        incident = calibrate_source(gaussian(plane, wb, x_m=center_x+offset_x, y_m=center_y+offset_y, angle_rad=angle), plane, 1e-3)
        coefficients = operator.project_coefficients(incident)
        each = operator.transmit_coefficients(coefficients).abs().square()
        eta = float(each[0, mode_index] / physical_plane_power(incident)[0])
        q = 2 * math.pi / plane.wavelength_m * math.sin(angle)
        analytic_eta = (2*wf*wb / (wf**2+wb**2))**2 * math.exp(-2*(offset_x**2+offset_y**2)/(wf**2+wb**2) - q*q*wf**2*wb**2/(2*(wf**2+wb**2)))
        error = abs(eta - analytic_eta)
        maximum_reference_error = max(maximum_reference_error, error)
        check = no_energy_creation(physical_plane_power(incident), each)
        rows.append({"case": name, "eta_target_channel": eta, "eta_analytic_infinite_plane": analytic_eta,
                     "absolute_reference_error": error, "no_energy_creation": check,
                     "channel_power_w_each": each[0].tolist()})
    matched = rows[0]["eta_target_channel"]
    passed = abs(lp01_eta-1) < POWER_RTOL and maximum_reference_error < 2e-4 and all(row["eta_target_channel"] < matched for row in rows[1:]) and all(row["no_energy_creation"]["passed"] for row in rows)
    return {"status": "PASS" if passed else "FAIL", "lp01_matched_eta": lp01_eta,
            "gaussian_geometry": operator.geometry_diagnostics, "cases": rows,
            "maximum_analytic_reference_error": maximum_reference_error,
            "parameter_status": "PROJECT_GEOMETRY_UNMEASURED"}


@torch.no_grad()
def absolute_power_diagnostic() -> tuple[dict[str, object], dict[str, object]]:
    """Existing component trace with a newly calibrated, synthetic source only.

    No ComplexOLSState patch or alternative full-runtime acceptance is used.
    The independent trace explicitly reports production entrypoint blockage.
    """
    runtime = alignment.LightGen2DSemanticGenerationInputOLSAlignment().double().eval()
    runtime.requires_grad_(False)
    state_before = {key: value.clone() for key, value in runtime.state_dict().items()}
    slm_plane = runtime.slm_plane
    profiles = [("center", 60e-6, 0, 0), ("x_offset", 60e-6, 35e-6, 0),
                ("y_offset", 60e-6, 0, -40e-6), ("wide", 120e-6, 0, 0)]
    inputs = torch.cat([gaussian(slm_plane, waist, x_m=x, y_m=y) for _, waist, x, y in profiles])
    source = calibrate_source(inputs, slm_plane, 1e-3)
    # Unit SLM command: external Gaussian illumination is incident on its plane.
    unit_slm = runtime.input_adapter(torch.ones(len(profiles), 1, 28, 28, dtype=torch.float64)).slm_field
    after_slm = field(source.tensor * unit_slm.tensor, slm_plane, "unit_command_slm_with_calibrated_illumination")
    entrance = runtime.input_relay(after_slm)
    encoder_output, encoder_records, encoder_stats = runtime.encoder(entrance, "audit_encoder")
    ols_input = field(encoder_output.tensor, runtime.ols_plane, "existing_encoder_to_ols_component")
    stages = {"P_input": physical_plane_power(source), "P_after_slm": physical_plane_power(after_slm),
              "P_encoder_entrance": physical_plane_power(entrance),
              "P_after_encoder_1": physical_plane_power(encoder_records[0]),
              "P_after_encoder_2": physical_plane_power(encoder_records[1]),
              "P_before_fiber": physical_plane_power(ols_input)}
    payloads = {}
    basis_reports = {}
    operators = {"lp01_ports_v1": runtime.ols.operator,
                 "gaussian_s12_v1": GaussianFiberArrayOLS(runtime.ols_plane, GaussianFiberGeometry())}
    for basis_id, operator in operators.items():
        coefficients = operator.project_coefficients(ols_input)
        transmitted = operator.transmit_coefficients(coefficients)
        each = transmitted.abs().square()
        emitted = field(operator.reemit_coefficients(transmitted).unsqueeze(1), runtime.generator_entrance_plane, "existing_modal_reemission_component")
        before = stages["P_before_fiber"]
        check = no_energy_creation(before, each)
        emitted_power = physical_plane_power(emitted)
        if basis_id == "lp01_ports_v1":
            peak_mode_density = operator.output_modes.abs().square().flatten(1).amax(1).double()
            peak_kind = "sampled rectangular-cell LP01 peak, numerical plane only"
            decoder_output, decoder_records, _ = runtime.generator(emitted, "audit_generator")
            decoder_output = field(decoder_output.tensor, runtime.generator_exit_plane, "existing_generator_exit_plane_binding")
            camera = runtime.output_relay(decoder_output)
            stages.update({"P_fiber_total": each.sum(1), "P_before_nonlinearity": emitted_power,
                           "P_after_decoder_1": physical_plane_power(decoder_records[0]),
                           "P_after_decoder_2": physical_plane_power(decoder_records[1]), "P_camera": physical_plane_power(camera)})
        else:
            peak_mode_density = torch.full((100,), 2/(math.pi*operator.geometry.waist_m**2), dtype=torch.float64)
            peak_kind = "analytic Gaussian peak, numerical plane assumed material plane"
        peak_intensity = each * peak_mode_density[None] / 1e4
        payloads[basis_id] = {"coefficients": transmitted.cpu(), "channel_power_w": each.cpu(),
                              "before_fiber_w": before.cpu(), "reemitted_power_w": emitted_power.cpu(),
                              "peak_mode_density_per_m2": peak_mode_density.cpu(), "peak_intensity_w_cm2": peak_intensity.cpu()}
        basis_reports[basis_id] = {"channel_power_w_each": each.tolist(), "P_fiber_total_w": each.sum(1).tolist(),
                                  "P_before_nonlinearity_w": emitted_power.tolist(), "coupling_each_sample": (each.sum(1)/before).tolist(),
                                  "no_energy_creation": check, "peak_intensity_w_cm2_each": peak_intensity.tolist(), "peak_intensity_definition": peak_kind,
                                  "reemission_no_energy_creation": bool((emitted_power <= before*(1+POWER_RTOL)+POWER_ATOL_W).all())}
    # Check the same source scaling against one unscaled optical component pass.
    unit_incident = calibrate_source(inputs[:1], slm_plane, 1.0)
    unit_entrance = runtime.input_relay(unit_incident)
    unit_out, _, _ = runtime.encoder(unit_entrance, "audit_linearity_encoder")
    unit_coefficients = runtime.ols.operator.project_coefficients(field(unit_out.tensor, runtime.ols_plane, "linearity_ols"))
    scaling_error = float((unit_coefficients.abs().square()*1e-3 - payloads["lp01_ports_v1"]["channel_power_w"][:1]).abs().max())
    # Exercise the unmodified full API and preserve its actual engineering result.
    try:
        runtime.forward_from_image(torch.ones(1, 1, 28, 28, dtype=torch.float64))
    except NameError as error:
        if "Mapping" not in str(error):
            raise
        entrypoint = {"status": "BLOCKED_ENGINEERING", "exception_type": "NameError", "message": str(error),
                      "location": "ComplexOLSState.__post_init__:128", "production_source_modified": False}
    else:
        entrypoint = {"status": "PASS", "input": "synthetic all-ones image; zero phase; no checkpoint"}
    unchanged = all(torch.equal(value, runtime.state_dict()[key]) for key, value in state_before.items())
    stage_values = list(stages.values())
    passive_stages = all(bool((right <= left*(1+POWER_RTOL)+POWER_ATOL_W).all()) for left, right in zip(stage_values, stage_values[1:]))
    passed = unchanged and passive_stages and scaling_error < POWER_ATOL_W and all(x["no_energy_creation"]["passed"] and x["reemission_no_energy_creation"] for x in basis_reports.values())
    clear_production_transfer_cache()
    return {
        "status": "PASS" if passed else "FAIL", "scope": "component_diagnostic_only_not_full_entrypoint_acceptance",
        "power_unit": "W", "field_unit": "sqrt(W)/m", "source_power_w": 1e-3,
        "source_plane": slm_plane.to_dict(), "source_assumption": "four Gaussian illumination scenarios; unit ideal SLM command; source calibration only at input",
        "profiles": [{"id": name, "waist_m": waist, "x_m": x, "y_m": y} for name, waist, x, y in profiles],
        "checkpoint": None, "phase_state": "all_zero_untrained_2plus2",
        "stage_power_w": {key: value.tolist() for key, value in stages.items()},
        "encoder_crop_retention": encoder_stats["crop_retention"].tolist(), "basis_results": basis_reports,
        "source_scaling_max_channel_error_w": scaling_error, "passive_stage_powers": passive_stages,
        "model_state_unchanged": unchanged, "production_entrypoint_probe": entrypoint,
        "existing_absolute_power_model": False, "material_added_to_training": False,
        "material_plane_status": "NUMERICAL_OLS_PLANE_ASSUMED_MATERIAL_PLANE_UNCALIBRATED",
        "scientific_updates": 0, "optimizer_updates": 0, "evaluator_calls": 0,
    }, {"schema": "physical_power_component_evidence.v1", "source_power_w": 1e-3,
        "profile_ids": [x[0] for x in profiles], "basis_results": payloads, "stage_power_w": {key: value.cpu() for key, value in stages.items()}}
