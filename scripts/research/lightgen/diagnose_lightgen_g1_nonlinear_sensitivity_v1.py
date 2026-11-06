"""Run frozen G1 propagation for explicitly hypothetical local Te responses."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import diagnose_lightgen_g1_te_inserted_path_v1 as linear_ref
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as mapper_ref
from scripts.research.lightgen import execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder_ref


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_nonlinear_sensitivity_v1.yaml"
PRIMARY_TOLERANCE = 1.0e-6
GRID_TOLERANCE = 1.0e-3


class DiagnosticStop(RuntimeError):
    """Invalid research input or numerical preflight."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticStop(message)


def _resolve(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), f"missing or out-of-root input: {relative_path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    require(path.resolve() == DEFAULT_CONFIG.resolve(), "use the explicit fixed sensitivity configuration")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict)
            and config.get("schema") == "lightgen_g1_nonlinear_sensitivity.v1"
            and config.get("task_id") == "LIGHTGEN_G1_NONLINEAR_SENSITIVITY_V1"
            and config.get("evidence_tier") == "diagnostic_only"
            and config["response"]["updates"] == 0,
            "configuration identity or zero-update scope changed")
    require(config["base_config"] == "configs/research/lightgen/g1_te_inserted_path_v1.yaml"
            and config["source"] == {
                "attempt04_fields": "outputs/20261029_lightgen_g1_te_inserted_path_v1/runs/autodl_backup_attempt_04/complex_fields.npz",
                "attempt04_metadata": "outputs/20261029_lightgen_g1_te_inserted_path_v1/runs/autodl_backup_attempt_04/complex_field_metadata.json",
            }, "frozen base configuration or attempt04 source changed")
    require(config["model"] == {
        "primary_grid": 2048, "reference_grid": 4096, "dtype": "complex128",
        "frozen_optical_branch": "C", "frozen_optical_step": 45000,
        "frozen_mapper_step": 5000, "frozen_cohort": "C_seed_2026101501/G1_stream_0.pt",
    }, "frozen G1 model identity changed")
    require(config["population"]["primary_rows"] == 100
            and config["population"]["reference_rows"] == 10
            and config["population"]["reference_positions"] == list(range(0, 100, 10))
            and config["population"]["arms"] == ["direct", "mapped"],
            "fixed population or optical paths changed")
    response = config["response"]
    require(response == {
        "baseline_state": "solid_25C", "baseline_thickness_nm": 10.0,
        "reference_scale_definition": "median_direct_effective_mean_irradiance_2048",
        "saturating_shape": "u_over_one_plus_u", "amplitude_fraction": 0.5,
        "amplitude_scales": [0.1, 1.0, 10.0], "amplitude_signs": [-1, 1],
        "phase_amplitude_scale": 1.0, "phase_values_rad": [-0.7853981633974483, 0.7853981633974483],
        "phase_amplitude_signs": [-1, 1], "phase_only_amplitude_fraction": 0.0,
        "phase_only_scale": 1.0, "phase_only_values_rad": [-0.7853981633974483, 0.7853981633974483],
        "updates": 0,
    }, "fixed hypothetical response design changed")
    require(config["controls"] == {
        "bare_reference": True, "constant_q0_linear_reference": True,
        "identity": True, "global_phase_only": True,
        "equal_power_uniform_complex_control": True,
        "wrong_power_as_field_amplitude_negative_control": True,
    }, "required control set changed")
    return config


def _conditions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = config["response"]
    rows: list[dict[str, Any]] = []
    for sign in spec["amplitude_signs"]:
        for scale in spec["amplitude_scales"]:
            rows.append({
                "condition_id": f"amp_{'inc' if sign > 0 else 'dec'}_scale_{scale:g}",
                "kind": "amplitude", "sigma": int(sign), "m": float(spec["amplitude_fraction"]),
                "a": float(scale), "psi_rad": 0.0,
            })
    for sign in spec["phase_amplitude_signs"]:
        for phase in spec["phase_values_rad"]:
            rows.append({
                "condition_id": f"amp_{'inc' if sign > 0 else 'dec'}_phase_{'pos' if phase > 0 else 'neg'}",
                "kind": "amplitude_and_phase", "sigma": int(sign),
                "m": float(spec["amplitude_fraction"]),
                "a": float(spec["phase_amplitude_scale"]), "psi_rad": float(phase),
            })
    for phase in spec["phase_only_values_rad"]:
        rows.append({
            "condition_id": f"phase_only_{'pos' if phase > 0 else 'neg'}",
            "kind": "phase_only", "sigma": 0,
            "m": float(spec["phase_only_amplitude_fraction"]),
            "a": float(spec["phase_only_scale"]), "psi_rad": float(phase),
        })
    require(len(rows) == 12 and len({row["condition_id"] for row in rows}) == 12,
            "the fixed twelve-condition design is incomplete")
    return rows


def _load_base(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    base_path = _resolve(config["base_config"])
    base_config = linear_ref._load_config(base_path)
    context = linear_ref._load_inputs(base_config, device)
    fields_path = _resolve(config["source"]["attempt04_fields"])
    metadata_path = _resolve(config["source"]["attempt04_metadata"])
    metadata = _read_json(metadata_path)
    require(metadata.get("schema") == "lightgen_g1_te_inserted_path_complex_fields.v1"
            and metadata.get("complex_dtype") == "complex128"
            and set(metadata.get("grids", {})) == {"2048", "4096"},
            "attempt 04 complex-field metadata identity is invalid")
    arrays = np.load(fields_path, allow_pickle=False)
    require(np.array_equal(arrays["2048_direct_native_capture_indices"], context["indices"].cpu().numpy())
            and np.array_equal(arrays["2048_mapped_native_capture_indices"], context["indices"].cpu().numpy()),
            "attempt 04 population indices do not match the frozen cohort")
    require(np.array_equal(arrays["2048_direct_labels"], context["labels"].cpu().numpy())
            and np.array_equal(arrays["2048_mapped_labels"], context["labels"].cpu().numpy()),
            "attempt 04 labels do not match the frozen cohort")
    for grid, count in ((2048, 100), (4096, 10)):
        for arm in ("direct", "mapped"):
            prefix = f"{grid}_{arm}"
            entry = arrays[f"{prefix}_bare_generator_entry"]
            camera = arrays[f"{prefix}_bare_camera"]
            require(entry.shape == (count, 100, 100) and camera.shape == (count, 200, 200)
                    and entry.dtype == np.complex128 and camera.dtype == np.complex128
                    and np.isfinite(entry).all() and np.isfinite(camera).all(),
                    f"saved bare fields are invalid for {prefix}")
            require(np.array_equal(arrays[f"{prefix}_native_capture_indices"],
                                   arrays[f"{grid}_direct_native_capture_indices"]),
                    f"sample index order differs for {prefix}")
    return {"base_config": base_config, "context": context, "metadata": metadata,
            "metadata_path": metadata_path,
            "fields_path": fields_path, "arrays": arrays}


def _reference_intensity(fields: Mapping[str, Any], pixel_pitch_m: list[float]) -> float:
    vals: list[float] = []
    require(len(pixel_pitch_m) == 2 and all(math.isfinite(float(x)) and float(x) > 0.0
                                            for x in pixel_pitch_m),
            "generator-entry pixel pitch is invalid")
    dx, dy = (float(x) for x in pixel_pitch_m)
    area = dx * dy
    for index in range(100):
        intensity = np.abs(fields[f"2048_direct_bare_generator_entry"][index]) ** 2
        power = float(intensity.sum() * area)
        squared_power = float(np.square(intensity).sum() * area)
        require(power > 0.0 and squared_power > 0.0, "zero direct reference field")
        vals.append(squared_power / power)
    value = float(np.median(np.asarray(vals, dtype=np.float64)))
    require(math.isfinite(value) and value > 0.0, "invalid frozen I_ref")
    return value


def _plane_matches(plane: Any, saved: Mapping[str, Any], name: str) -> None:
    current = plane.to_dict()
    for key in ("plane_id", "shape", "pixel_pitch_m", "fov_m", "wavelength_m", "z_m",
                "coordinate_frame", "origin_m"):
        expected = saved.get(key)
        actual = current.get(key)
        if isinstance(expected, list) and isinstance(actual, list):
            require(len(expected) == len(actual)
                    and all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-15)
                            for a, b in zip(expected, actual, strict=True)),
                    f"saved coordinate metadata differs at {name}.{key}")
        elif isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
            require(math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1.0e-15),
                    f"saved coordinate metadata differs at {name}.{key}")
        else:
            require(actual == expected, f"saved coordinate metadata differs at {name}.{key}")


def _q0(context: Mapping[str, Any]) -> complex:
    value = context["transfers"][10.0, "solid_25C"]["q"]
    q = complex(value)
    require(math.isfinite(q.real) and math.isfinite(q.imag), "nonfinite ADMA2025 q0")
    return q


def _identity(config: Mapping[str, Any], base: Mapping[str, Any], conditions: list[dict[str, Any]],
              i_ref: float) -> dict[str, Any]:
    context = base["context"]
    q0 = _q0(context)
    return {
        "task_id": config["task_id"], "evidence_tier": config["evidence_tier"],
        "cohort_model": context["cohort"].get("model"),
        "optical_branch": "C", "optical_step": 45000, "mapper_step": 5000,
        "cohort": config["model"]["frozen_cohort"],
        "native_capture_indices": context["indices"].cpu().tolist(),
        "labels": context["labels"].cpu().tolist(), "arms": config["population"]["arms"],
        "grids": [2048, 4096], "reference_population_positions": config["population"]["reference_positions"],
        "input_fields": config["source"],
        "generator_entry_plane": base["metadata"]["grids"]["2048"]["arms"]["direct"]["planes"]["generator_entry"]["plane"],
        "q0_real": q0.real, "q0_imag": q0.imag,
        "q0_intensity_multiplier": abs(q0) ** 2, "i_ref_w_m2": i_ref,
        "conditions": conditions, "camera_gain_w_inverse": context["camera_gain"],
        "updates": 0,
    }


def _complex_field(value: torch.Tensor, plane: Any, op: str) -> ComplexField:
    return ComplexField(value, plane, OperationProvenance(op, plane.plane_id, None,
                        {"evidence_tier": "diagnostic_only"}))


def _response_field(entry: ComplexField, q0: complex, condition: Mapping[str, Any],
                    i_ref: float, bare_transmission: float) -> tuple[ComplexField, dict[str, float]]:
    intensity = entry.tensor.abs().square().to(torch.float64)
    u = intensity / (float(condition["a"]) * i_ref)
    h = u / (1.0 + u)
    gain = 1.0 + float(condition["sigma"]) * float(condition["m"]) * h
    require(torch.isfinite(gain).all().item() and float(gain.min()) >= 0.5 - 1.0e-12
            and float(gain.max()) <= 1.5 + 1.0e-12,
            f"nonlinear amplitude power factor is invalid for {condition['condition_id']}")
    qmag = abs(q0) * torch.sqrt(gain)
    phase = math.atan2(q0.imag, q0.real) + float(condition["psi_rad"]) * h
    qmap = torch.polar(qmag, phase).to(torch.complex128)
    absolute_transmission = bare_transmission * qmap.abs().square()
    require(torch.isfinite(qmap.real).all().item() and torch.isfinite(qmap.imag).all().item()
            and torch.isfinite(absolute_transmission).all().item()
            and float(absolute_transmission.max()) <= 1.0 + 1.0e-12,
            f"hypothetical response violates passive power bounds for {condition['condition_id']}")
    result = _complex_field(entry.tensor.to(torch.complex128) * qmap, entry.plane,
                            "hypothetical_local_nonlinear_response")
    zero_input = intensity == 0.0
    if bool(zero_input.any().item()):
        require(torch.count_nonzero(result.tensor[zero_input]).item() == 0,
                f"zero input pixels became nonzero for {condition['condition_id']}")
    metrics = {"h_min": float(h.min().item()), "h_max": float(h.max().item()),
               "gain_min": float(gain.min().item()), "gain_max": float(gain.max().item()),
               "absolute_transmission_min": float(absolute_transmission.min().item()),
               "absolute_transmission_max": float(absolute_transmission.max().item())}
    return result, metrics


def _camera_from_entry(runtime: Any, entry: ComplexField) -> ComplexField:
    generated, _, _ = runtime.generator(entry, "lightgen_generator")
    output = ComplexField(
        generated.tensor, runtime.generator_exit_plane,
        OperationProvenance("lightgen_generator_exit", generated.plane.plane_id, None,
                            {"generator_layers": 2, "nonlinear_sensitivity": True}),
    )
    camera = runtime.output_relay(output)
    runtime._validate_relay_diagnostics(runtime.output_relay)
    require(torch.isfinite(camera.tensor.real).all().item()
            and torch.isfinite(camera.tensor.imag).all().item(), "nonfinite CMOS complex field")
    return camera


def _power(value: torch.Tensor, area: float) -> float:
    return float(value.abs().square().sum().item() * area)


def _shape_error(actual: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    a = a / max(float(a.sum()), 1.0e-300)
    b = b / max(float(b.sum()), 1.0e-300)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1.0e-300))


def _field_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return linear_ref._relative_error(actual, expected)


def _readout_metrics(raw: torch.Tensor, context: Mapping[str, Any], pixel_area: float) -> dict[str, Any]:
    view = mapper_ref._readout_for_evaluator(raw, pixel_area,
                                             context["camera_gain"])
    fixed_native = (raw * pixel_area * context["camera_gain"]).clamp(0.0, 1.0)
    with torch.inference_mode():
        ed = context["evaluator_ed"](view)
        ec = context["evaluator_ec"](view)[1]
    return {"fixed_view28": view[0, 0].detach().cpu().numpy(),
            "fixed_native_u8": torch.round(fixed_native[0, 0] * 255.0).to(torch.uint8).cpu().numpy(),
            "ed_logits": ed[0].detach().cpu().numpy(),
            "ec_logits": ec[0].detach().cpu().numpy(),
            "ed_prediction": int(ed.argmax(1).item()),
            "ec_prediction": int(ec.argmax(1).item()),
            "clipped_fraction": float(((raw * pixel_area * context["camera_gain"]) >= 1.0)
                                       .double().mean().item())}


def _run_sample(adapter: Any, entry: ComplexField, q0: complex, condition: Mapping[str, Any],
                i_ref: float, bare_transmission: float, pixel_area: float) -> tuple[ComplexField, ComplexField, dict[str, float]]:
    modified_entry, response_metrics = _response_field(entry, q0, condition, i_ref, bare_transmission)
    camera = _camera_from_entry(adapter.runtime, modified_entry)
    require(torch.isfinite(camera.tensor.abs().square()).all().item(), "nonfinite CMOS intensity")
    return modified_entry, camera, response_metrics


def _arrays_for_grid(base: Mapping[str, Any], grid: int, indices: list[int],
                     conditions: list[dict[str, Any]], config: Mapping[str, Any],
                     i_ref: float, output: Path, *, preflight: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = base["context"]
    source = base["arrays"]
    adapter = mapper_ref._new_optical_model(context["optical_checkpoint"], grid, context["device"])
    adapter.eval().requires_grad_(False)
    runtime = adapter.runtime
    model_before = linear_ref._snapshot(adapter)
    module_before = {name: linear_ref._snapshot(context[key]) for name, key in (
        ("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec"))}
    pixel_area = math.prod(runtime.camera_plane.pixel_pitch_m)
    q0 = _q0(context)
    bare_transmission = float(context["transfers"][0.0, "bare"]["T"])
    rows: list[dict[str, Any]] = []
    fields: dict[str, np.ndarray] = {}
    count = len(indices)
    for arm in config["population"]["arms"]:
        prefix = f"{grid}_{arm}"
        arm_planes = base["metadata"]["grids"][str(grid)]["arms"][arm]["planes"]
        _plane_matches(runtime.generator_entrance_plane,
                       arm_planes["generator_entry"]["plane"], f"{arm}.generator_entry")
        _plane_matches(runtime.camera_plane, arm_planes["camera"]["plane"], f"{arm}.camera")
        saved_indices = np.asarray(source[f"{prefix}_native_capture_indices"], dtype=np.int64)
        saved_labels = np.asarray(source[f"{prefix}_labels"], dtype=np.int64)
        saved_positions = np.asarray(source[f"{prefix}_population_positions"], dtype=np.int64)
        require(np.array_equal(saved_indices, context["indices"].cpu().numpy()[indices]),
                f"attempt 04 native indices differ for grid{grid}/{arm}")
        require(np.array_equal(saved_positions, np.asarray(indices, dtype=np.int64)),
                f"attempt 04 population positions differ for grid{grid}/{arm}")
        require(np.array_equal(saved_labels, context["labels"].cpu().numpy()[indices]),
                f"attempt 04 label order differs for grid{grid}/{arm}")
        storage_rows = indices if grid == 2048 else list(range(count))
        bare_entry_np = np.asarray(source[f"{prefix}_bare_generator_entry"][storage_rows], dtype=np.complex128)
        bare_camera_np = np.asarray(source[f"{prefix}_bare_camera"][storage_rows], dtype=np.complex128)
        require(np.isfinite(bare_entry_np).all() and np.isfinite(bare_camera_np).all(),
                f"nonfinite saved bare input for grid{grid}/{arm}")
        result_shape = (count, len(conditions))
        out_entry = np.empty(result_shape + (100, 100), dtype=np.complex128)
        out_camera = np.empty(result_shape + (200, 200), dtype=np.complex128)
        out_raw = np.empty(result_shape + (200, 200), dtype=np.float64)
        out_fixed_native = np.empty(result_shape + (200, 200), dtype=np.uint8)
        out_view = np.empty(result_shape + (28, 28), dtype=np.float32)
        for local_pos, population_pos in enumerate(indices):
            native_index = int(saved_indices[local_pos])
            label = int(saved_labels[local_pos])
            entry_np = bare_entry_np[local_pos]
            camera_np = bare_camera_np[local_pos]
            entry = _complex_field(torch.as_tensor(entry_np[None, None], device=context["device"]),
                                   runtime.generator_entrance_plane, "attempt04_saved_bare_entry")
            bare_raw = torch.as_tensor(np.abs(camera_np[None, None]) ** 2, device=context["device"], dtype=torch.float64)
            bare_entry_power = _power(entry.tensor, math.prod(entry.plane.pixel_pitch_m))
            bare_camera_power = float(bare_raw.sum().item() * pixel_area)
            # The linear reference is independently reconstructed from the saved complex field.
            q0_field = _complex_field(entry.tensor * q0, entry.plane, "constant_q0_linear_reference")
            q0_camera = _camera_from_entry(runtime, q0_field)
            previous_q0_camera_np = np.asarray(source[f"{prefix}_te_10nm_solid_25C_camera"][local_pos],
                                               dtype=np.complex128)
            q0_reconstruction_error = _field_relative(
                q0_camera.tensor, torch.as_tensor(previous_q0_camera_np[None, None], device=context["device"]))
            require(q0_reconstruction_error <= PRIMARY_TOLERANCE,
                    f"q0 forward replay differs from attempt 04 for {grid}/{arm}/{native_index}")
            for cond_index, condition in enumerate(conditions):
                modified_entry, camera, response = _run_sample(
                    adapter, entry, q0, condition, i_ref, bare_transmission, pixel_area)
                raw = camera.tensor.abs().square().to(torch.float64)
                raw_np = raw[0, 0].detach().cpu().numpy()
                entry_np_out = modified_entry.tensor[0, 0].detach().cpu().numpy()
                camera_np_out = camera.tensor[0, 0].detach().cpu().numpy()
                control_overlap = torch.sum(torch.conj(entry.tensor.to(torch.complex128))
                                            * modified_entry.tensor.to(torch.complex128))
                require(abs(complex(control_overlap.item())) > 1.0e-30,
                        f"uniform complex control phase is undefined for {condition['condition_id']}")
                control_phase = float(torch.angle(control_overlap).item())
                input_ratio = _power(modified_entry.tensor, math.prod(entry.plane.pixel_pitch_m)) / max(bare_entry_power, 1.0e-300)
                control_scalar = math.sqrt(max(input_ratio, 0.0)) * complex(math.cos(control_phase), math.sin(control_phase))
                predicted_control_camera = camera_np * control_scalar
                predicted_control_raw = np.abs(predicted_control_camera) ** 2
                control_camera_power = float(predicted_control_raw.sum() * pixel_area)
                raw_power = float(raw_np.sum() * pixel_area)
                camera_phase = linear_ref._overlap_phase(
                    camera.tensor, torch.as_tensor(camera_np[None, None], device=context["device"]))
                uniform_alpha = np.vdot(camera_np, camera_np_out) / max(float(np.vdot(camera_np, camera_np).real), 1.0e-300)
                uniform_residual = float(np.linalg.norm(camera_np_out - uniform_alpha * camera_np)
                                         / max(np.linalg.norm(camera_np_out), 1.0e-300))
                readout = _readout_metrics(raw, context, pixel_area)
                baseline_readout = _readout_metrics(bare_raw, context, pixel_area)
                row = {
                    "grid": grid, "arm": arm, "population_position": int(population_pos),
                    "native_capture_index": native_index, "label": label,
                    **condition, **response,
                    "i_ref_w_m2": i_ref,
                    "input_power_before_w": bare_entry_power,
                    "input_power_after_w": _power(modified_entry.tensor, math.prod(entry.plane.pixel_pitch_m)),
                    "membrane_power_ratio": input_ratio,
                    "cmos_power_before_w": bare_camera_power,
                    "cmos_power_after_w": raw_power,
                    "cmos_power_ratio": raw_power / max(bare_camera_power, 1.0e-300),
                    "equal_power_control_cmos_power_w": control_camera_power,
                    "equal_power_control_cmos_power_ratio": control_camera_power / max(bare_camera_power, 1.0e-300),
                    "cmos_phase_vs_bare_rad": camera_phase,
                    "normalized_shape_error_vs_bare": _shape_error(raw_np, np.asarray(bare_raw[0, 0].cpu())),
                    "normalized_shape_error_vs_equal_power": _shape_error(raw_np, predicted_control_raw),
                    "best_uniform_complex_fit_residual": uniform_residual,
                    "equal_power_control_scalar_real": control_scalar.real,
                    "equal_power_control_scalar_imag": control_scalar.imag,
                    "equal_power_control_field_amplitude": abs(control_scalar),
                    "equal_power_control_phase_rad": math.atan2(control_scalar.imag, control_scalar.real),
                    "equal_power_control_input_power_residual": abs(
                        _power(modified_entry.tensor, math.prod(entry.plane.pixel_pitch_m))
                        - _power(entry.tensor * control_scalar, math.prod(entry.plane.pixel_pitch_m)))
                        / max(_power(modified_entry.tensor, math.prod(entry.plane.pixel_pitch_m)), 1.0e-300),
                    "fixed_exposure_clipped_fraction": readout["clipped_fraction"],
                    "ED_prediction": readout["ed_prediction"],
                    "EC_prediction": readout["ec_prediction"],
                    "ED_prediction_changed": readout["ed_prediction"] != baseline_readout["ed_prediction"],
                    "EC_prediction_changed": readout["ec_prediction"] != baseline_readout["ec_prediction"],
                    "q0_reconstruction_error_vs_attempt04": q0_reconstruction_error,
                }
                rows.append(row)
                out_entry[local_pos, cond_index] = entry_np_out
                out_camera[local_pos, cond_index] = camera_np_out
                out_raw[local_pos, cond_index] = raw_np
                out_fixed_native[local_pos, cond_index] = readout["fixed_native_u8"]
                out_view[local_pos, cond_index] = readout["fixed_view28"].astype(np.float32)
            fields[f"{prefix}_native_capture_indices"] = saved_indices
            fields[f"{prefix}_labels"] = saved_labels
            fields[f"{prefix}_population_positions"] = saved_positions
        fields[f"{grid}_{arm}_hypothetical_entry"] = out_entry
        fields[f"{grid}_{arm}_hypothetical_camera"] = out_camera
        fields[f"{grid}_{arm}_hypothetical_raw_w_m2"] = out_raw
        fields[f"{grid}_{arm}_hypothetical_fixed_native_u8"] = out_fixed_native
        fields[f"{grid}_{arm}_hypothetical_fixed_view28"] = out_view
    linear_ref._assert_unchanged(adapter, model_before, f"G1 optical model grid {grid}")
    for name, key in (("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec")):
        linear_ref._assert_unchanged(context[key], module_before[name], name)
    np.savez_compressed(output / f"fields_grid_{grid}.npz", **fields)
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, {"grid": grid, "saved_sample_count_per_arm": count,
                  "saved_conditions": len(conditions), "field_archive": f"fields_grid_{grid}.npz",
                  "frozen_model_state_unchanged": True}


def _preflight_controls(config: Mapping[str, Any], base: Mapping[str, Any],
                        conditions: list[dict[str, Any]], i_ref: float) -> list[dict[str, Any]]:
    context = base["context"]
    fields = base["arrays"]
    q0 = _q0(context)
    bare_t = float(context["transfers"][0.0, "bare"]["T"])
    checks: list[dict[str, Any]] = []
    selected = [0, 90]
    for grid, positions in ((2048, selected), (4096, [0, 9])):
        adapter = mapper_ref._new_optical_model(context["optical_checkpoint"], grid, context["device"])
        adapter.eval().requires_grad_(False)
        runtime = adapter.runtime
        area = math.prod(runtime.camera_plane.pixel_pitch_m)
        model_before = linear_ref._snapshot(adapter)
        module_before = {name: linear_ref._snapshot(context[key]) for name, key in (
            ("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec"))}
        for arm in config["population"]["arms"]:
            prefix = f"{grid}_{arm}"
            for pos in positions:
                saved_population_position = int(fields[f"{prefix}_population_positions"][pos])
                entry_np = fields[f"{prefix}_bare_generator_entry"][pos]
                bare_camera_np = fields[f"{prefix}_bare_camera"][pos]
                entry = _complex_field(torch.as_tensor(entry_np[None, None], device=context["device"]),
                                       runtime.generator_entrance_plane, "attempt04_saved_bare_entry")
                bare_camera = torch.as_tensor(bare_camera_np[None, None], device=context["device"])
                baseline_raw = bare_camera.abs().square()
                # Control 1: identity injection reproduces the saved bare camera field.
                identity_camera = _camera_from_entry(runtime, entry)
                identity_error = _field_relative(identity_camera.tensor, bare_camera)
                require(identity_error <= PRIMARY_TOLERANCE,
                        f"saved bare field does not replay in the frozen generator: {grid}/{arm}/{pos}")
                # Control 2: the saved literature q0 recreates the saved 10 nm solid proxy.
                q0_entry = _complex_field(entry.tensor * q0, entry.plane, "constant_q0_negative_control")
                q0_camera = _camera_from_entry(runtime, q0_entry)
                prior_camera = torch.as_tensor(
                    fields[f"{prefix}_te_10nm_solid_25C_camera"][pos][None, None],
                    device=context["device"], dtype=torch.complex128)
                q0_error = _field_relative(q0_camera.tensor, prior_camera)
                require(q0_error <= PRIMARY_TOLERANCE,
                        f"constant q0 does not reproduce attempt 04: {grid}/{arm}/{pos}")
                phase = 0.37
                phase_entry = _complex_field(entry.tensor * complex(math.cos(phase), math.sin(phase)),
                                             entry.plane, "global_phase_negative_control")
                phase_camera = _camera_from_entry(runtime, phase_entry)
                phase_error = _field_relative(phase_camera.tensor,
                                              bare_camera * complex(math.cos(phase), math.sin(phase)))
                phase_intensity_error = linear_ref._relative_error(
                    phase_camera.tensor.abs().square(), baseline_raw)
                require(phase_error <= PRIMARY_TOLERANCE and phase_intensity_error <= PRIMARY_TOLERANCE,
                        f"global phase control failed for {grid}/{arm}/{pos}")
                for condition in conditions:
                    mod_entry, mod_camera, _ = _run_sample(adapter, entry, q0, condition, i_ref, bare_t, area)
                    overlap = torch.sum(torch.conj(entry.tensor.to(torch.complex128))
                                        * mod_entry.tensor.to(torch.complex128))
                    mod_power = _power(mod_entry.tensor, math.prod(entry.plane.pixel_pitch_m))
                    bare_power = _power(entry.tensor, math.prod(entry.plane.pixel_pitch_m))
                    require(abs(complex(overlap.item())) > 1.0e-30 and mod_power > 0.0,
                            f"equal power control is undefined: {condition['condition_id']}")
                    phase_control = float(torch.angle(overlap).item())
                    c = math.sqrt(mod_power / bare_power) * complex(math.cos(phase_control), math.sin(phase_control))
                    control_entry = _complex_field(entry.tensor * c, entry.plane,
                                                   "equal_power_uniform_complex_control")
                    control_camera = _camera_from_entry(runtime, control_entry)
                    expected = bare_camera * c
                    control_field_error = _field_relative(control_camera.tensor, expected)
                    control_power_error = abs(_power(control_entry.tensor, math.prod(entry.plane.pixel_pitch_m))
                                              - mod_power) / mod_power
                    te_raw = mod_camera.tensor.abs().square()
                    control_raw = control_camera.tensor.abs().square()
                    shape_error = linear_ref._shape_error(te_raw, control_raw)
                    control_shape_residual = linear_ref._shape_error(
                        control_raw, baseline_raw * (abs(c) ** 2))
                    require(control_field_error <= PRIMARY_TOLERANCE
                            and control_power_error <= PRIMARY_TOLERANCE
                            and control_shape_residual <= PRIMARY_TOLERANCE,
                            f"uniform equal power control failed: {grid}/{arm}/{pos}/{condition['condition_id']}")
                    # The negative case deliberately uses power gain as a field multiplier.
                    wrong_discrepancy = None
                    if condition["condition_id"] == "amp_inc_scale_1":
                        i = entry.tensor.abs().square().to(torch.float64)
                        u = i / (float(condition["a"]) * i_ref)
                        h = u / (1.0 + u)
                        gain = 1.0 + float(condition["sigma"]) * float(condition["m"]) * h
                        phase_map = math.atan2(q0.imag, q0.real) + float(condition["psi_rad"]) * h
                        wrong_qmap = torch.polar(abs(q0) * gain, phase_map).to(torch.complex128)
                        wrong_entry = _complex_field(entry.tensor * wrong_qmap, entry.plane,
                                                     "negative_control_power_used_as_field_amplitude")
                        wrong_camera = _camera_from_entry(runtime, wrong_entry)
                        wrong_raw = wrong_camera.tensor.abs().square().to(torch.float64)
                        wrong_discrepancy = linear_ref._relative_error(wrong_raw, mod_camera.tensor.abs().square())
                        require(wrong_discrepancy > 1.0e-2,
                                "production negative control did not distinguish power from field amplitude")
                    checks.append({
                        "grid": grid, "arm": arm,
                        "native_capture_index": int(fields[f"{prefix}_native_capture_indices"][pos]),
                        "population_position": saved_population_position, "condition_id": condition["condition_id"],
                        "identity_replay_relative_error": identity_error,
                        "q0_replay_relative_error_vs_attempt04": q0_error,
                        "phase_control_field_error": phase_error,
                        "phase_control_intensity_error": phase_intensity_error,
                        "equal_power_control_field_error": control_field_error,
                        "equal_power_control_power_error": control_power_error,
                        "equal_power_control_shape_residual": control_shape_residual,
                        "equal_power_control_shape_error": shape_error,
                        "wrong_power_as_field_raw_relative_discrepancy": wrong_discrepancy,
                    })
        linear_ref._assert_unchanged(adapter, model_before, f"preflight G1 optical model grid {grid}")
        for name, key in (("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec")):
            linear_ref._assert_unchanged(context[key], module_before[name], f"preflight {name}")
    return checks


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows to write to {path.name}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = ("normalized_shape_error_vs_bare", "normalized_shape_error_vs_equal_power",
            "best_uniform_complex_fit_residual", "membrane_power_ratio", "cmos_power_ratio",
            "fixed_exposure_clipped_fraction", "cmos_phase_vs_bare_rad")
    grouping = sorted({(int(row["grid"]), row["arm"], row["condition_id"])
                       for row in rows})
    for grid, arm, condition_id in grouping:
        for label in ["all", *range(10)]:
            selected = [row for row in rows if row["grid"] == grid and row["arm"] == arm
                        and row["condition_id"] == condition_id
                        and (label == "all" or int(row["label"]) == label)]
            if not selected:
                continue
            item: dict[str, Any] = {"grid": grid, "arm": arm, "condition_id": condition_id,
                                "class_label": label,
                                "sample_count": len(selected),
                                "ED_prediction_changed_count": sum(bool(row["ED_prediction_changed"]) for row in selected),
                                "EC_prediction_changed_count": sum(bool(row["EC_prediction_changed"]) for row in selected)}
            for key in keys:
                values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
                item[f"{key}_median"] = float(np.median(values))
                item[f"{key}_p10"] = float(np.quantile(values, 0.1))
                item[f"{key}_p90"] = float(np.quantile(values, 0.9))
            result.append(item)
    return result


def _field_metadata(base: Mapping[str, Any], conditions: list[dict[str, Any]],
                    i_ref: float) -> dict[str, Any]:
    grids: dict[str, Any] = {}
    for grid in (2048, 4096):
        arms: dict[str, Any] = {}
        for arm in ("direct", "mapped"):
            planes = base["metadata"]["grids"][str(grid)]["arms"][arm]["planes"]
            arms[arm] = {"generator_entry": planes["generator_entry"]["plane"],
                         "camera": planes["camera"]["plane"]}
        grids[str(grid)] = {"arms": arms}
    return {
        "schema": "lightgen_g1_nonlinear_sensitivity_field_metadata.v1",
        "complex_dtype": "complex128", "intensity_dtype": "float64",
        "source_attempt04_metadata": base["metadata_path"].relative_to(ROOT).as_posix(),
        "source_population": "saved native_capture_indices; 100 rows, 10 per class",
        "bare_field_storage": base["fields_path"].relative_to(ROOT).as_posix(),
        "array_axis_order": {
            "bare_generator_entry": ["sample", "y", "x"],
            "bare_camera": ["sample", "y", "x"],
            "hypothetical_generator_entry": ["sample", "condition", "y", "x"],
            "hypothetical_camera": ["sample", "condition", "y", "x"],
            "hypothetical_raw_w_m2": ["sample", "condition", "y", "x"],
            "hypothetical_fixed_native_u8": ["sample", "condition", "y", "x"],
            "hypothetical_fixed_view28": ["sample", "condition", "y", "x"],
        },
        "intensity_convention": "squared scalar field interpreted using the saved attempt04 numerical optical power convention; not a calibrated physical SMF facet irradiance",
        "generator_entry_i_ref_w_m2": i_ref,
        "conditions": conditions, "grids": grids,
        "equal_power_control": "uniform complex scalar applied to the saved bare entry field; scalar is in sample_metrics.csv and output field is reconstructible from saved bare camera field",
    }


def _save_gallery(rows: list[dict[str, Any]], fields_path: Path, conditions: list[dict[str, Any]],
                  grid: int) -> int:
    archive = np.load(fields_path, allow_pickle=False)
    count = 0
    gallery_dir = fields_path.parent / "galleries"
    gallery_dir.mkdir(exist_ok=True)
    for arm in ("direct", "mapped"):
        arm_rows = [row for row in rows if row["grid"] == grid and row["arm"] == arm]
        labels = [int(row) for row in archive[f"{grid}_{arm}_labels"]]
        for condition_index, condition in enumerate(conditions):
            views = archive[f"{grid}_{arm}_hypothetical_fixed_native_u8"][:, condition_index]
            per_condition = [row for row in arm_rows if row["condition_id"] == condition["condition_id"]]
            clipped = max(float(row["fixed_exposure_clipped_fraction"]) for row in per_condition)
            ordered = sorted(range(len(labels)), key=lambda i: (labels[i], i))
            side = 10 if len(labels) == 100 else 2
            columns = 10 if len(labels) == 100 else 5
            fig, axes = plt.subplots(side, columns, figsize=(10, 10 if side == 10 else 2.4),
                                     squeeze=False, constrained_layout=True)
            for axis, idx in zip(axes.flat, ordered, strict=True):
                axis.imshow(views[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                axis.set_title(str(labels[idx]), fontsize=5, pad=1)
                axis.axis("off")
            fig.suptitle(f"grid {grid} / {arm} / {condition['condition_id']} / fixed exposure; max clip={clipped:.4g}")
            path = gallery_dir / f"grid{grid}_{arm}_{condition['condition_id']}_fixed.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            count += 1
            raw = archive[f"{grid}_{arm}_hypothetical_raw_w_m2"][:, condition_index]
            normalized = raw / np.maximum(raw.sum(axis=(-2, -1), keepdims=True), 1.0e-300)
            common_max = float(normalized.max())
            fig, axes = plt.subplots(side, columns, figsize=(10, 10 if side == 10 else 2.4),
                                     squeeze=False, constrained_layout=True)
            for axis, idx in zip(axes.flat, ordered, strict=True):
                axis.imshow(normalized[idx], cmap="gray", vmin=0.0, vmax=common_max,
                            interpolation="nearest")
                axis.set_title(str(labels[idx]), fontsize=5, pad=1)
                axis.axis("off")
            fig.suptitle(f"grid {grid} / {arm} / {condition['condition_id']} / total-power normalized; common vmax={common_max:.4g}")
            path = gallery_dir / f"grid{grid}_{arm}_{condition['condition_id']}_normalized.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            count += 1
    return count


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen G1 hypothetical nonlinear sensitivity diagnostic",
        "",
        f"Status: `{payload['status']}`; evidence tier: `{payload['evidence_tier']}`; optimizer updates: 0.",
        "",
        "All twelve response conditions are explicit mathematical sensitivity assumptions, not measured Te properties.",
        "The material basis is the saved ADMA2025 10 nm, 25°C solid linear proxy. The fields are the saved attempt 04 bare fields at the numerical generator entrance.",
        "",
        "## Frozen input and response contract",
        "",
        f"- Frozen I_ref: `{payload['identity']['i_ref_w_m2']:.9g} W/m²`, derived as the median effective mean irradiance of the 100 saved direct fields on the 2048 plane.",
        f"- Relative linear q0: `{payload['identity']['q0_real']:.9g} + {payload['identity']['q0_imag']:.9g}i`; material baseline intensity multiplier: `{payload['identity']['q0_intensity_multiplier']:.9g}`.",
        "- Spatial response: `q(I)=q0*sqrt(1+sigma*m*h(I/(a*I_ref)))*exp(i*psi*h(...))`, `h(u)=u/(1+u)`.",
        "- SLM, OLS, saved fields, mapper, evaluator, and 100-sample population remained frozen; no training occurred.",
        "",
        "## Condition summary",
        "",
        "See `condition_summary.csv` for overall and per-class median and 10th–90th percentile shape, power, phase, clipping and evaluator-change summaries by grid and arm. `sample_metrics.csv` retains every sample row; `cross_grid_checks.csv` reports raw-intensity and normalized-shape grid differences for the ten shared rows.",
        "",
        "## Interpretation boundary",
        "",
        "Changes that exceed the equal-power control and numerical negative-control envelope describe only these declared response assumptions in the current numerical optical path. They are not a fitted Te response, target-film prediction, hardware result, or training authorization.",
        "",
        "The fixed-exposure gallery uses 200×200 numerical CMOS samples quantized with the frozen gain and common 8-bit [0,255] display scale; the second gallery uses per-frame total-power normalization with one shared scale per condition. Neither auto-stretches individual tiles. The 28×28 pooled `fixed_view28` is saved separately for the frozen evaluators.",
        "",
    ]
    return "\n".join(lines)


def _preflight(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is required for production-path preflight")
    mapper_ref._set_numerics()
    device = torch.device("cuda")
    base = _load_base(config, device)
    conditions = _conditions(config)
    entry_meta = base["metadata"]["grids"]["2048"]["arms"]["direct"]["planes"]["generator_entry"]["plane"]
    i_ref = _reference_intensity(base["arrays"], entry_meta["pixel_pitch_m"])
    identity = _identity(config, base, conditions, i_ref)
    checks = _preflight_controls(config, base, conditions, i_ref)
    wrong_rows = [row for row in checks
                  if row["condition_id"] == "amp_inc_scale_1"
                  and row["wrong_power_as_field_raw_relative_discrepancy"] is not None]
    require(len(wrong_rows) == 8
            and all(float(row["wrong_power_as_field_raw_relative_discrepancy"]) > 1.0e-2
                    for row in wrong_rows),
            "production power-as-field negative control coverage is incomplete")
    for grid in (2048, 4096):
        for arm in config["population"]["arms"]:
            for condition in conditions:
                check_rows = [row for row in checks if row["grid"] == grid and row["arm"] == arm
                              and row["condition_id"] == condition["condition_id"]]
                require(len(check_rows) == 2, f"preflight condition coverage incomplete: {grid}/{arm}/{condition['condition_id']}")
    return {"schema": "lightgen_g1_nonlinear_sensitivity_preflight.v1",
            "status": "PREFLIGHT_PASSED", "evidence_tier": "diagnostic_only",
            "optimizer_updates": 0, "identity": identity,
            "checks": [{"name": "saved_attempt04_fields_and_coordinate_identity", "passed": True},
                       {"name": "frozen_linear_q0_reproduces_attempt04", "passed": True},
                       {"name": "identity_global_phase_equal_power_controls", "passed": True},
                       {"name": "wrong_power_as_field_rejected", "passed": True,
                        "representative_rows": len(wrong_rows),
                        "minimum_raw_relative_discrepancy": min(
                            float(row["wrong_power_as_field_raw_relative_discrepancy"]) for row in wrong_rows)},
                       {"name": "hypotheses_finite_and_passive_in_absolute_transmission", "passed": True},
                       {"name": "model_mapper_evaluator_state_unchanged", "passed": True}],
            "production_control_rows": checks,
            "stop": "Measure is admitted only with this matching PREFLIGHT_PASSED receipt."}


def _measure(config: Mapping[str, Any], output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    require(preflight.get("status") == "PREFLIGHT_PASSED"
            and preflight.get("evidence_tier") == "diagnostic_only",
            "current production preflight is not passed")
    require(len(preflight.get("checks", [])) == 6
            and all(isinstance(check, dict) and check.get("passed") is True
                    for check in preflight.get("checks", [])),
            "preflight receipt is incomplete or contains a failed producer check")
    mapper_ref._set_numerics()
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "CUDA is required for the full frozen measurement")
    base = _load_base(config, device)
    conditions = _conditions(config)
    entry_meta = base["metadata"]["grids"]["2048"]["arms"]["direct"]["planes"]["generator_entry"]["plane"]
    i_ref = _reference_intensity(base["arrays"], entry_meta["pixel_pitch_m"])
    identity = _identity(config, base, conditions, i_ref)
    require(preflight.get("identity") == identity, "preflight identity does not match current loaded inputs")
    _write_json_new(output / "field_metadata.json", _field_metadata(base, conditions, i_ref))
    all_rows: list[dict[str, Any]] = []
    grid_receipts: list[dict[str, Any]] = []
    for grid, positions in ((2048, list(range(100))),
                            (4096, list(config["population"]["reference_positions"]))):
        rows, receipt = _arrays_for_grid(base, grid, positions, conditions, config, i_ref, output,
                                         preflight=False)
        all_rows.extend(rows)
        grid_receipts.append(receipt)
    summaries = _summary(all_rows)
    _write_csv_new(output / "sample_metrics.csv", all_rows)
    _write_csv_new(output / "condition_summary.csv", summaries)
    gallery_count = sum(_save_gallery(all_rows, output / f"fields_grid_{grid}.npz", conditions, grid)
                        for grid in (2048, 4096))
    cross_grid: list[dict[str, Any]] = []
    cross_shape_by_sample: dict[tuple[str, str, int], float] = {}
    loaded = {grid: np.load(output / f"fields_grid_{grid}.npz", allow_pickle=False) for grid in (2048, 4096)}
    for arm in config["population"]["arms"]:
        idx2048 = loaded[2048][f"2048_{arm}_native_capture_indices"]
        idx4096 = loaded[4096][f"4096_{arm}_native_capture_indices"]
        pos2048 = loaded[2048][f"2048_{arm}_population_positions"]
        pos4096 = loaded[4096][f"4096_{arm}_population_positions"]
        require(np.array_equal(idx2048[pos4096], idx4096), f"cross-grid sample identity differs for {arm}")
        require(np.array_equal(pos2048[pos4096], pos4096), f"cross-grid population position differs for {arm}")
        for condition_index, condition in enumerate(conditions):
            a = loaded[2048][f"2048_{arm}_hypothetical_raw_w_m2"][pos4096, condition_index]
            b = loaded[4096][f"4096_{arm}_hypothetical_raw_w_m2"][:, condition_index]
            for j, (left, right) in enumerate(zip(a, b, strict=True)):
                error = float(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1.0e-300))
                shape_error = _shape_error(left, right)
                position = int(pos4096[j])
                cross_shape_by_sample[(arm, condition["condition_id"], position)] = shape_error
                cross_grid.append({"arm": arm, "condition_id": condition["condition_id"],
                                   "native_capture_index": int(idx4096[j]),
                                   "population_position": position,
                                   "raw_intensity_relative_error": error,
                                   "normalized_shape_relative_error": shape_error})
    for row in all_rows:
        key = (str(row["arm"]), str(row["condition_id"]), int(row["population_position"]))
        row["cross_grid_normalized_shape_relative_error"] = cross_shape_by_sample.get(key)
    max_grid_error = max(row["raw_intensity_relative_error"] for row in cross_grid)
    require(max_grid_error <= GRID_TOLERANCE,
            f"2048/4096 raw outputs exceed frozen tolerance: {max_grid_error}")
    _write_csv_new(output / "cross_grid_checks.csv", cross_grid)
    checks = [
        {"name": "all_12_conditions_all_100_rows_two_arms_2048", "passed":
         len([r for r in all_rows if r["grid"] == 2048]) == 2400},
        {"name": "all_12_conditions_reference_rows_two_arms_4096", "passed":
         len([r for r in all_rows if r["grid"] == 4096]) == 240},
        {"name": "saved_field_archives_and_galleries", "passed": gallery_count == 96,
         "gallery_count": gallery_count},
        {"name": "2048_4096_raw_intensity_consistency", "passed": max_grid_error <= GRID_TOLERANCE,
         "maximum_relative_error": max_grid_error, "tolerance": GRID_TOLERANCE},
        {"name": "frozen_model_mapper_evaluators", "passed": all(x["frozen_model_state_unchanged"]
          for x in grid_receipts)},
    ]
    require(all(check["passed"] for check in checks), "one or more production consumers failed")
    payload = {"schema": "lightgen_g1_nonlinear_sensitivity_measurement.v1",
               "status": "COMPLETE_DIAGNOSTIC_ONLY", "evidence_tier": "diagnostic_only",
               "optimizer_updates": 0, "scientific_updates": 0, "identity": identity,
               "conditions": conditions, "grid_receipts": grid_receipts,
               "checks": checks,
               "files": ["preflight.json", "field_metadata.json", "sample_metrics.csv", "condition_summary.csv",
                         "cross_grid_checks.csv", "fields_grid_2048.npz", "fields_grid_4096.npz",
                         "galleries/"],
               "interpretation": "hypothesis sensitivity only; not a fitted Te response or target-device prediction"}
    _write_json_new(output / "measurement.json", payload)
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_render_report({**payload, "identity": identity}))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "measure"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output = output.resolve()
    require(output.is_relative_to(ROOT / "outputs") and "runs" in output.parts
            and any(part.startswith("diagnostic-only") for part in output.parts),
            "output must be an explicit diagnostic-only directory under outputs/.../runs/")
    configured_output = (ROOT / config["output"]["run_dir"]).resolve()
    require(output == configured_output,
            "output directory differs from the explicit research configuration")
    if args.mode == "preflight":
        require(not output.exists(), "preflight output already exists; select a new attempt directory")
        output.mkdir(parents=True, exist_ok=False)
        try:
            result = _preflight(config, output)
        except Exception as exc:
            result = {"schema": "lightgen_g1_nonlinear_sensitivity_preflight.v1",
                      "status": "INVALID_HARNESS", "evidence_tier": "diagnostic_only",
                      "optimizer_updates": 0, "failure_type": type(exc).__name__,
                      "failure": str(exc), "stop": "Full measurement was not started."}
            _write_json_new(output / "preflight.json", result)
            print(json.dumps({"status": result["status"], "path": str(output / "preflight.json")}, ensure_ascii=False))
            return 2
        _write_json_new(output / "preflight.json", result)
        print(json.dumps({"status": result["status"], "path": str(output / "preflight.json")}, ensure_ascii=False))
        return 0
    receipt_path = output / "preflight.json"
    require(output.is_dir() and receipt_path.is_file() and not (output / "measurement.json").exists(),
            "measure requires the current preflight receipt and an unused measurement destination")
    preflight = _read_json(receipt_path)
    try:
        result = _measure(config, output, preflight)
    except Exception as exc:
        if not (output / "measurement.json").exists():
            _write_json_new(output / "measurement.json", {
                "schema": "lightgen_g1_nonlinear_sensitivity_measurement.v1",
                "status": "INVALID_HARNESS", "evidence_tier": "diagnostic_only",
                "optimizer_updates": 0, "failure_type": type(exc).__name__, "failure": str(exc),
            })
        raise
    print(json.dumps({"status": result["status"], "path": str(output / "measurement.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
