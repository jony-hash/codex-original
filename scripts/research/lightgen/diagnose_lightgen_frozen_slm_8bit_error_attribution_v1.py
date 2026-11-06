"""Attribute frozen 8-bit SLM replay errors from saved evidence only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_FROZEN_SLM_8BIT_ERROR_ATTRIBUTION_V1"
REPLAY_TASK_ID = "LIGHTGEN_FROZEN_SLM_8BIT_REPLAY_V1"
REACHABILITY_TASK_ID = "LIGHTGEN_FROZEN_SLM_OLS_INPUT_REACHABILITY_V1"
METHODS = ("G1", "G4")
CONDITIONS = ("A8_amplitude_linear", "I8_intensity_linear")
ALL_CONDITIONS = ("continuous", *CONDITIONS)
GRIDS = (2048, 4096)
IMAGE_SHAPE = (200, 200)
PIXEL_COUNT = 40_000
POWER_FIELDS = (
    "P_source_before_SLM_W", "P_after_SLM_W", "P_encoder_entrance_W",
    "P_encoder_layer_1_after_aperture_W", "P_encoder_layer_2_after_aperture_W",
    "P_before_OLS_incident_W", "P_after_OLS_reemitted_decoder_entry_W",
    "P_decoder_layer_1_after_aperture_W", "P_decoder_layer_2_after_aperture_W",
    "P_CMOS_W",
)
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/frozen_slm_8bit_error_attribution_v1.yaml"
EXPECTED_ATTEMPT = ROOT / "outputs/20261025_lightgen_frozen_slm_8bit_error_attribution_v1/runs/attempt_01"


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}", ValueError)
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)
    dtype = torch.complex128 if actual.is_complex() or expected.is_complex() else torch.float64
    actual_flat = actual.reshape(-1).to(dtype)
    expected_flat = expected.reshape(-1).to(dtype)
    denominator = torch.linalg.vector_norm(expected_flat)
    _require(torch.isfinite(denominator).item() and float(denominator) > 0.0,
             "relative-error reference is zero or nonfinite")
    return float(torch.linalg.vector_norm(actual_flat - expected_flat) / denominator)


def _relative_norm(vector: torch.Tensor, reference: torch.Tensor) -> float:
    vector = torch.as_tensor(vector)
    reference = torch.as_tensor(reference)
    dtype = torch.complex128 if vector.is_complex() or reference.is_complex() else torch.float64
    vector_norm = torch.linalg.vector_norm(vector.reshape(-1).to(dtype))
    denominator = torch.linalg.vector_norm(reference.reshape(-1).to(dtype))
    _require(torch.isfinite(vector_norm).item() and torch.isfinite(denominator).item()
             and float(denominator) > 0.0,
             "relative-norm reference is zero or nonfinite")
    return float(vector_norm / denominator)


def _relative_residual(actual: torch.Tensor, expected: torch.Tensor, normalization: torch.Tensor) -> float:
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)
    normalization = torch.as_tensor(normalization)
    dtype = torch.complex128 if actual.is_complex() or expected.is_complex() else torch.float64
    denominator = torch.linalg.vector_norm(normalization.reshape(-1).to(dtype))
    _require(torch.isfinite(denominator).item() and float(denominator) > 0.0,
             "residual normalization is zero or nonfinite")
    difference = actual.reshape(-1).to(dtype) - expected.reshape(-1).to(dtype)
    return float(torch.linalg.vector_norm(difference) / denominator)


def _phase_aligned_error(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    actual = torch.as_tensor(actual, dtype=torch.complex128).reshape(-1)
    reference = torch.as_tensor(reference, dtype=torch.complex128).reshape(-1)
    cross = torch.vdot(reference, actual)
    _require(torch.isfinite(cross.real).item() and torch.isfinite(cross.imag).item()
             and abs(cross) > 0.0, "phase-alignment reference is zero or nonfinite")
    phase = float(torch.angle(cross))
    aligned = actual * torch.exp(torch.tensor(-1j * phase, dtype=torch.complex128))
    return phase, _relative(aligned, reference)


def _validate_population(expected: list[dict[str, Any]], observed: list[dict[str, Any]]) -> None:
    expected_keys = [row.get("sample_key") for row in expected]
    observed_keys = [row.get("sample_key") for row in observed]
    _require(len(expected) == len(observed) == 200
             and len(set(expected_keys)) == 200 and len(set(observed_keys)) == 200
             and set(expected_keys) == set(observed_keys),
             "fixed population has a missing or duplicate sample identity")
    expected_by_key = {row["sample_key"]: row for row in expected}
    _require(all(expected_by_key[row["sample_key"]] == row for row in observed),
             "saved sample identity fields differ from the fixed population")
    for method in METHODS:
        rows = [row for row in expected if row.get("method") == method]
        _require(len(rows) == 100 and all(sum(item.get("label") == label for item in rows) == 10
                                          for label in range(10)),
                 f"fixed population is incomplete for {method}")


def _validate_config(config: dict[str, Any], path: Path) -> None:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(), "attribution must use the fixed config", ValueError)
    _require(config.get("schema") == "lightgen_frozen_slm_8bit_error_attribution.v1"
             and config.get("task_id") == TASK_ID and config.get("evidence_tier") == "diagnostic_only",
             "attribution config identity changed", ValueError)
    source = config["source"]
    _require(source.get("replay_attempt_dir") ==
             "outputs/20261025_lightgen_frozen_slm_8bit_replay_v1/runs/attempt_01"
             and source.get("expected_replay_run_id") == "01a34fe66f3743b2a8eb7701c7a9e9fd"
             and source.get("reachability_attempt_dir") ==
             "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01"
             and source.get("expected_reachability_run_id") == "04390141af8c48278cfd3578f3688e0c",
             "fixed upstream attempt path or identity changed", ValueError)
    population = config["population"]
    _require(population.get("methods") == list(METHODS)
             and population.get("samples_per_method") == 100
             and population.get("samples_per_class_per_method") == 10
             and population.get("attribution_rows") == 400
             and population.get("preflight_samples") == 20,
             "fixed population or preflight changed", ValueError)
    optics = config["optics"]
    _require(optics.get("checkpoint_step") == 45_000
             and math.isclose(float(optics.get("source_power_w", 0.0)), 2.493e-4,
                              rel_tol=0.0, abs_tol=1e-12)
             and optics.get("response_matrix_grid") == 2048
             and optics.get("response_matrix_shape") == [100, PIXEL_COUNT]
             and optics.get("propagation_grids") == list(GRIDS)
             and optics.get("encoder_layers") == 2 and optics.get("decoder_layers") == 2
             and optics.get("phase_masks") == "frozen"
             and optics.get("global_gain") == "frozen_C_checkpoint",
             "frozen model or optical configuration changed", ValueError)
    measurement = config["measurement"]
    _require(measurement.get("precision_limit") == 1.0e-4
             and measurement.get("matrix_prediction_relative_error_max") == 1.0e-6
             and measurement.get("matrix_delta_closure_relative_error_max") == 1.0e-6
             and measurement.get("cross_grid_relative_error_max") == 1.0e-4
             and measurement.get("power_relative_error_max") == 1.0e-10
             and measurement.get("power_absolute_error_w") == 1.0e-15
             and all(measurement.get(key) == 0 for key in
                     ("model_updates", "adam_calls", "prior_fit", "input_resolve",
                      "new_sampling", "new_human_review")),
             "analysis thresholds or zero-update boundary changed", ValueError)
    _require(config["output"].get("experiment_root") ==
             "outputs/20261025_lightgen_frozen_slm_8bit_error_attribution_v1/runs/attempt_01"
             and config["output"].get("final_consumer") ==
             "recompute_from_saved_gray_response_matrix_and_sample_outputs",
             "fixed output path or consumer changed", ValueError)


def _load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    replay_dir = (ROOT / config["source"]["replay_attempt_dir"]).resolve()
    reach_dir = (ROOT / config["source"]["reachability_attempt_dir"]).resolve()
    _require(replay_dir.is_dir() and reach_dir.is_dir(),
             "one of the two fixed completed attempts is missing", FileNotFoundError)
    replay_identity = _read_json(replay_dir / "run_identity.json")
    replay_final = _read_json(replay_dir / "final_decision.json")
    replay_consumed = _read_json(replay_dir / "windows_reconsumption.json")
    reach_identity = _read_json(reach_dir / "run_identity.json")
    reach_final = _read_json(reach_dir / "final_decision.json")
    _require(replay_identity.get("task_id") == REPLAY_TASK_ID
             and replay_identity.get("run_id") == config["source"]["expected_replay_run_id"]
             and replay_identity.get("status") == "CLOSED"
             and replay_final.get("run_id") == replay_identity["run_id"]
             and replay_final.get("sample_denominator_per_condition") == 200
             and replay_consumed.get("run_id") == replay_identity["run_id"]
             and replay_consumed.get("recomputed_records") == 600
             and replay_consumed.get("output_sample_rows_recomputed") == 1200,
             "8-bit replay attempt is not the fixed closed evidence")
    _require(reach_identity.get("task_id") == REACHABILITY_TASK_ID
             and reach_identity.get("run_id") == config["source"]["expected_reachability_run_id"]
             and reach_identity.get("status") == "CLOSED"
             and reach_final.get("run_id") == reach_identity["run_id"]
             and reach_final.get("decision") == "ALL_200_NUMERICAL_COMMAND_WITNESSES_VERIFIED",
             "continuous reachability attempt is not the fixed closed evidence")

    authority = _read_json(replay_dir / "input_authority.json")
    identities = authority.get("sample_identities")
    _require(authority.get("run_id") == replay_identity["run_id"]
             and authority.get("sample_count") == 200
             and authority.get("checkpoint_step") == 45_000
             and authority.get("methods") == list(METHODS)
             and isinstance(identities, list),
             "saved input authority is incomplete or changed")
    _validate_population(identities, identities)
    commands = torch.load(reach_dir / "amplitude_and_intensity_commands.pt",
                          map_location="cpu", weights_only=False)
    _require(commands.get("schema") == "lightgen_frozen_slm_ols_amplitude_commands.v1"
             and commands.get("sample_identities") == identities
             and tuple(commands["amplitude_command"].shape) == (200, 1, *IMAGE_SHAPE)
             and tuple(commands["intensity_command"].shape) == (200, 1, *IMAGE_SHAPE),
             "upstream continuous instruction identity or shape changed")
    amplitude = commands["amplitude_command"].to(torch.float64).reshape(200, PIXEL_COUNT)
    _require(torch.isfinite(amplitude).all().item()
             and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0
             and torch.allclose(commands["intensity_command"].double().reshape(200, PIXEL_COUNT),
                                amplitude.square(), rtol=0.0, atol=2e-7),
             "upstream continuous amplitude is nonfinite or out of range")

    encoded = torch.load(replay_dir / "encoded_commands.pt", map_location="cpu", weights_only=False)
    _require(encoded.get("task_id") == REPLAY_TASK_ID
             and encoded.get("source_run_id") == config["source"]["expected_reachability_run_id"]
             and encoded.get("sample_identities") == identities
             and encoded.get("encoding") == {
                 "rounding": "floor(255*x+0.5)",
                 "A8": "Q8(a) amplitude gray code; production intensity input is (gray/255)^2",
                 "I8": "Q8(a^2) intensity gray code; production intensity input is gray/255",
                 "decoder": "gray/255 in float64; frozen production adapter applies sqrt to intensity input",
             }, "saved gray command identity or decoding rule changed")
    gray_a, gray_i = encoded.get("A8_gray_u8"), encoded.get("I8_gray_u8")
    _require(isinstance(gray_a, torch.Tensor) and gray_a.dtype == torch.uint8
             and tuple(gray_a.shape) == (200, 1, *IMAGE_SHAPE)
             and isinstance(gray_i, torch.Tensor) and gray_i.dtype == torch.uint8
             and tuple(gray_i.shape) == (200, 1, *IMAGE_SHAPE),
             "saved uint8 gray arrays are malformed")
    gray_a = gray_a.reshape(200, PIXEL_COUNT)
    gray_i = gray_i.reshape(200, PIXEL_COUNT)
    _require(torch.equal(gray_a, torch.floor(amplitude * 255.0 + 0.5).to(torch.uint8))
             and torch.equal(gray_i, torch.floor(amplitude.square() * 255.0 + 0.5).to(torch.uint8)),
             "saved uint8 gray values do not match the fixed half-up encoding")
    amplitude_a8 = gray_a.to(torch.float64) / 255.0
    intensity_i8 = gray_i.to(torch.float64) / 255.0
    amplitude_i8 = intensity_i8.sqrt()
    _require(torch.allclose(encoded["A8_decoded_amplitude"].reshape(200, PIXEL_COUNT).double(),
                            amplitude_a8, rtol=0.0, atol=1e-7)
             and torch.allclose(encoded["I8_decoded_intensity"].reshape(200, PIXEL_COUNT).double(),
                                intensity_i8, rtol=0.0, atol=1e-7)
             and torch.allclose(encoded["I8_effective_amplitude_after_production_sqrt"]
                                .reshape(200, PIXEL_COUNT).double(), amplitude_i8,
                                rtol=0.0, atol=1e-7),
             "saved decoded arrays disagree with uint8 commands")

    matrix_payload = torch.load(reach_dir / "response_matrix.pt", map_location="cpu", weights_only=False)
    matrix = matrix_payload.get("matrix")
    _require(matrix_payload.get("task_id") == REACHABILITY_TASK_ID
             and matrix_payload.get("dtype") == "complex128"
             and isinstance(matrix, torch.Tensor) and matrix.dtype == torch.complex128
             and tuple(matrix.shape) == (100, PIXEL_COUNT)
             and torch.isfinite(matrix.real).all().item() and torch.isfinite(matrix.imag).all().item(),
             "saved 2048 response matrix is malformed or has the wrong grid")

    target_payload = torch.load(replay_dir / "target_actual_coefficients.pt",
                                map_location="cpu", weights_only=False)
    ref_payload = torch.load(replay_dir / "reference_coefficients_and_cmos.pt",
                             map_location="cpu", weights_only=False)
    expected_schemas = (
        (target_payload, "lightgen_frozen_slm_8bit_target_actual_coefficients.v1"),
        (ref_payload, "lightgen_frozen_slm_8bit_direct_reference_outputs.v1"),
    )
    for payload, schema in expected_schemas:
        _require(payload.get("schema") == schema and payload.get("task_id") == REPLAY_TASK_ID
                 and payload.get("run_id") == replay_identity["run_id"]
                 and payload.get("sample_identities") == identities,
                 f"saved reference identity changed: {schema}")
    target = target_payload["target_coefficients"].to(torch.complex128)
    direct_coeffs, direct_cmos = ref_payload["direct_coefficients"], ref_payload[
        "direct_raw_cmos_intensity_w_per_m2"
    ]
    _require(tuple(target.shape) == (200, 100), "target coefficient shape changed")
    for grid in GRIDS:
        _require(tuple(direct_coeffs[str(grid)].shape) == (200, 100)
                 and direct_coeffs[str(grid)].dtype == torch.complex128
                 and torch.isfinite(direct_coeffs[str(grid)].real).all().item()
                 and torch.isfinite(direct_coeffs[str(grid)].imag).all().item()
                 and tuple(direct_cmos[str(grid)].shape) == (200, 1, *IMAGE_SHAPE)
                 and torch.isfinite(direct_cmos[str(grid)]).all().item()
                 and float(direct_cmos[str(grid)].min()) >= 0.0,
                 f"direct reference payload is malformed for grid {grid}")

    compact = torch.load(reach_dir / "compact_replay_evidence.pt", map_location="cpu", weights_only=False)
    _require(compact.get("schema") == "lightgen_frozen_slm_ols_compact_replay_evidence.v1"
             and compact.get("run_id") == reach_identity["run_id"]
             and compact.get("sample_identities") == identities,
             "continuous source replay evidence is malformed")
    _require(torch.equal(target, compact["target_coefficients"].to(torch.complex128)),
             "fixed target coefficients differ between the two saved attempts")
    for grid in GRIDS:
        payload = compact.get("grids", {}).get(str(grid), {})
        _require(tuple(payload.get("actual_coefficients").shape) == (200, 100)
                 and tuple(payload.get("actual_raw_cmos_intensity_w_per_m2").shape)
                 == (200, 1, *IMAGE_SHAPE),
                 f"continuous source outputs are missing for grid {grid}")
    camera_gain = float(authority["camera_gain_w_inverse"])
    _require(math.isfinite(camera_gain) and camera_gain > 0.0,
             "frozen camera gain is not finite and positive")
    return {
        "replay_dir": replay_dir, "reach_dir": reach_dir,
        "replay_identity": replay_identity, "reach_identity": reach_identity,
        "identities": identities, "amplitude": amplitude,
        "amplitude_a8": amplitude_a8, "amplitude_i8": amplitude_i8,
        "intensity_continuous": amplitude.square(),
        "intensity_a8": amplitude_a8.square(), "intensity_i8": intensity_i8,
        "gray_a": gray_a, "gray_i": gray_i, "matrix": matrix,
        "target": target, "direct_coeffs": direct_coeffs, "direct_cmos": direct_cmos,
        "source_compact": compact,
        "camera_gain": camera_gain,
        "source_power": float(config["optics"]["source_power_w"]),
    }


def _display_view(raw: torch.Tensor, pixel_area: float, gain: float) -> torch.Tensor:
    sensor = (raw.to(torch.float64) * pixel_area * gain).clamp(0.0, 1.0)
    return F.adaptive_avg_pool2d(sensor.reshape(1, 1, *IMAGE_SHAPE), (28, 28))[0, 0]


def _response_energy_decomposition(
    zero_component: torch.Tensor, rest_component: torch.Tensor, observed_delta: torch.Tensor,
) -> dict[str, float]:
    total = float(torch.vdot(observed_delta, observed_delta).real)
    zero = float(torch.vdot(zero_component, zero_component).real)
    rest = float(torch.vdot(rest_component, rest_component).real)
    cross = float(2.0 * torch.vdot(zero_component, rest_component).real)
    denominator = total if total > 0.0 else 1.0
    return {
        "zero_energy": zero, "rest_energy": rest, "cross_energy": cross, "total_energy": total,
        "zero_fraction": zero / denominator, "rest_fraction": rest / denominator,
        "cross_fraction": cross / denominator,
        "closure_relative_error": abs(total - zero - rest - cross) / denominator,
    }


def _scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"saved scalar is malformed: {name}") from exc
    _require(math.isfinite(result), f"saved scalar is nonfinite: {name}")
    return result


def _load_output(inputs: dict[str, Any], identity: dict[str, Any],
                 condition: str, grid: int) -> dict[str, Any]:
    path = inputs["replay_dir"] / "samples" / condition / str(grid) / f"{identity['sample_key']}.pt"
    _require(path.is_file(), f"saved production output is missing: {path}", FileNotFoundError)
    output = torch.load(path, map_location="cpu", weights_only=False)
    _require(output.get("schema") == "lightgen_frozen_slm_8bit_sample_output.v1"
             and output.get("task_id") == REPLAY_TASK_ID
             and output.get("run_id") == inputs["replay_identity"]["run_id"]
             and output.get("condition") == condition and output.get("grid") == grid
             and output.get("sample_identity") == identity,
             f"saved sample identity changed: {condition}/{grid}/{identity['sample_key']}")
    coefficients = output.get("actual_coefficients")
    raw = output.get("raw_cmos_intensity_w_per_m2")
    _require(isinstance(coefficients, torch.Tensor) and coefficients.dtype == torch.complex128
             and tuple(coefficients.shape) == (100,)
             and isinstance(raw, torch.Tensor) and tuple(raw.shape) == IMAGE_SHAPE
             and torch.isfinite(coefficients.real).all().item()
             and torch.isfinite(coefficients.imag).all().item()
             and torch.isfinite(raw).all().item() and float(raw.min()) >= 0.0,
             f"saved output is malformed or nonfinite: {condition}/{grid}/{identity['sample_key']}")
    power = output.get("power_trace_w")
    _require(isinstance(power, dict) and set(power) == set(POWER_FIELDS)
             and all(_scalar(value, key) >= 0.0 for key, value in power.items()),
             f"saved power trace changed: {condition}/{grid}/{identity['sample_key']}")
    return output


def _power_checks(inputs: dict[str, Any], output: dict[str, Any], intensity: torch.Tensor,
                  grid: int, sample_key: str) -> dict[str, float]:
    trace = output["power_trace_w"]
    expected_slm = inputs["source_power"] * float(intensity.mean())
    slm = float(trace["P_after_SLM_W"])
    slm_abs = abs(slm - expected_slm)
    slm_rel = slm_abs / max(abs(expected_slm), 1.0e-30)
    _require(slm_rel <= 1.0e-10 or slm_abs <= 1.0e-15,
             f"analytic SLM input power disagrees: {sample_key}/{grid}")
    area = _scalar(output["camera_pixel_area_m2"], "camera_pixel_area_m2")
    _require(area > 0.0, f"camera pixel area is not positive: {sample_key}/{grid}")
    raw_integral = float(output["raw_cmos_intensity_w_per_m2"].double().sum()) * area
    camera_saved = _scalar(output["camera_total_power_w"], "camera_total_power_w")
    camera_trace = float(trace["P_CMOS_W"])
    camera_abs = max(abs(raw_integral - camera_saved), abs(raw_integral - camera_trace))
    camera_rel = camera_abs / max(abs(raw_integral), 1.0e-30)
    _require(camera_rel <= 1.0e-10 or camera_abs <= 1.0e-15,
             f"CMOS intensity integral disagrees with saved power: {sample_key}/{grid}")
    _require(math.isclose(float(trace["P_source_before_SLM_W"]), inputs["source_power"],
                          rel_tol=0.0, abs_tol=1.0e-12),
             f"frozen source power changed: {sample_key}/{grid}")
    return {
        "analytic_slm_power_relative_error": slm_rel,
        "camera_integral_power_relative_error": camera_rel,
        "P_after_SLM_W": slm,
        "P_after_SLM_expected_W": expected_slm,
        "P_CMOS_integrated_W": raw_integral,
        "P_CMOS_trace_W": camera_trace,
    }


def _matrix_terms(inputs: dict[str, Any], condition: str, indices: list[int],
                  continuous_prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    if condition == CONDITIONS[0]:
        quantized = inputs["amplitude_a8"][indices]
        codes = inputs["gray_a"][indices]
        quantized_intensity = inputs["intensity_a8"]
    else:
        quantized = inputs["amplitude_i8"][indices]
        codes = inputs["gray_i"][indices]
        quantized_intensity = inputs["intensity_i8"]
    amplitude = inputs["amplitude"][indices]
    delta = quantized - amplitude
    delta_zero = delta * (codes == 0)
    delta_rest = delta * (codes != 0)
    intensity_delta = quantized_intensity[indices] - inputs["intensity_continuous"][indices]
    vectors = torch.cat((delta, delta_zero, delta_rest, intensity_delta), dim=0).T.contiguous()
    predicted = inputs["matrix"] @ vectors.to(torch.complex128)
    n = amplitude.shape[0]
    pred_delta = predicted[:, :n].T.contiguous()
    pred_zero = predicted[:, n:2 * n].T.contiguous()
    pred_rest = predicted[:, 2 * n:3 * n].T.contiguous()
    pred_wrong_intensity = predicted[:, 3 * n:4 * n].T.contiguous()
    return {
        "quantized": quantized,
        "codes": codes,
        "pred_delta": pred_delta,
        "pred_zero": pred_zero,
        "pred_rest": pred_rest,
        "pred_wrong_intensity": pred_wrong_intensity,
        "pred_quantized": continuous_prediction + pred_delta,
    }


def _continuous_record(inputs: dict[str, Any], index: int,
                       outputs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    identity = inputs["identities"][index]
    source_grids = inputs["source_compact"]["grids"]
    coeff_errors: list[float] = []
    cmos_errors: list[float] = []
    source_coeff_errors: list[float] = []
    source_cmos_errors: list[float] = []
    display_errors: list[float] = []
    for grid in GRIDS:
        output = outputs[grid]
        coeff_errors.append(_phase_aligned_error(output["actual_coefficients"],
                                                 inputs["target"][index])[1])
        cmos_errors.append(_relative(output["raw_cmos_intensity_w_per_m2"],
                                     inputs["direct_cmos"][str(grid)][index, 0].double()))
        source_coeff_errors.append(_relative(
            output["actual_coefficients"], source_grids[str(grid)]["actual_coefficients"][index]
        ))
        source_cmos_errors.append(_relative(
            output["raw_cmos_intensity_w_per_m2"],
            source_grids[str(grid)]["actual_raw_cmos_intensity_w_per_m2"][index, 0].double(),
        ))
        area = float(output["camera_pixel_area_m2"])
        view = _display_view(output["raw_cmos_intensity_w_per_m2"], area, inputs["camera_gain"])
        display_errors.append(_relative(view, view))
        power = _power_checks(inputs, output, inputs["intensity_continuous"][index],
                              grid, identity["sample_key"])
    cross_coeff = _relative(outputs[2048]["actual_coefficients"], outputs[4096]["actual_coefficients"])
    cross_cmos = _relative(outputs[2048]["raw_cmos_intensity_w_per_m2"],
                           outputs[4096]["raw_cmos_intensity_w_per_m2"])
    _require(cross_coeff <= 1.0e-4 and cross_cmos <= 1.0e-4,
             f"continuous cross-grid window failed: {identity['sample_key']}")
    _require(max(source_coeff_errors + source_cmos_errors) <= 1.0e-6,
             f"continuous source replay failed: {identity['sample_key']}")
    witness = (
        max(coeff_errors) <= 1.0e-4 and max(cmos_errors) <= 1.0e-4
        and cross_coeff <= 1.0e-4 and cross_cmos <= 1.0e-4
        and max(source_coeff_errors + source_cmos_errors) <= 1.0e-6
    )
    return {
        **identity, "condition": "continuous",
        "raw_coefficient_delta_relative_to_continuous": 0.0,
        "phase_aligned_coefficient_delta_relative_to_continuous": 0.0,
        "target_coefficient_relative_error_worst_grid": max(coeff_errors),
        "direct_cmos_relative_error_worst_grid": max(cmos_errors),
        "increment_cmos_relative_error_worst_grid": 0.0,
        "display_view28_increment_relative_error_worst_grid": 0.0,
        "display_view28_increment_linf_worst_grid": 0.0,
        "response_zero_energy_fraction": 0.0,
        "response_rest_energy_fraction": 0.0,
        "response_coherent_cross_energy_fraction": 0.0,
        "cross_grid_coefficient_relative_error": cross_coeff,
        "cross_grid_cmos_intensity_relative_error": cross_cmos,
        "source_continuous_coefficient_replay_relative_error_2048": source_coeff_errors[0],
        "source_continuous_cmos_replay_relative_error_2048": source_cmos_errors[0],
        "source_continuous_coefficient_replay_relative_error_4096": source_coeff_errors[1],
        "source_continuous_cmos_replay_relative_error_4096": source_cmos_errors[1],
        "original_precision_witness": bool(witness),
    }


def _quantized_record(inputs: dict[str, Any], condition: str, index: int, term_index: int,
                     terms: dict[str, torch.Tensor], continuous_prediction: torch.Tensor,
                     outputs: dict[int, dict[str, Any]],
                     continuous_outputs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    identity = inputs["identities"][index]
    amplitude = inputs["amplitude"][index]
    quantized = terms["quantized"][term_index]
    delta = quantized - amplitude
    quantized_intensity = inputs["intensity_a8"] if condition == CONDITIONS[0] \
        else inputs["intensity_i8"]
    intensity_delta = quantized_intensity[index] - inputs["intensity_continuous"][index]
    codes = terms["codes"][term_index]
    zc = continuous_outputs[2048]["actual_coefficients"].to(torch.complex128)
    zq = outputs[2048]["actual_coefficients"].to(torch.complex128)
    observed_delta = zq - zc
    d0 = terms["pred_zero"][term_index]
    drest = terms["pred_rest"][term_index]
    predicted_delta = terms["pred_delta"][term_index]
    phase, phase_error = _phase_aligned_error(zq, zc)
    energy = _response_energy_decomposition(d0, drest, observed_delta)
    positive = amplitude > 0.0
    positive_count = int(positive.sum())
    zeroed_count = int((positive & (codes == 0)).sum())
    predicted_cont_error = _relative(continuous_prediction[term_index], zc)
    predicted_quant_error = _relative(terms["pred_quantized"][term_index], zq)
    predicted_delta_error = _relative_residual(predicted_delta, observed_delta, zc)
    closure_error = _relative_residual(d0 + drest, observed_delta, zc)
    wrong_intensity_error = _relative_residual(terms["pred_wrong_intensity"][term_index],
                                               observed_delta, zc)
    _require(max(predicted_cont_error, predicted_quant_error, predicted_delta_error, closure_error)
             <= 1.0e-6, f"2048 response matrix fails for {identity['sample_key']} {condition}")

    row: dict[str, Any] = {
        **identity, "condition": condition,
        "zeroed_positive_pixel_count": zeroed_count,
        "original_positive_pixel_count": positive_count,
        "zeroed_positive_fraction_all_pixels": zeroed_count / PIXEL_COUNT,
        "zeroed_positive_fraction_original_positive": zeroed_count / positive_count
        if positive_count else 0.0,
        "amplitude_delta_mean": float(delta.mean()),
        "amplitude_delta_l2": float(torch.linalg.vector_norm(delta)),
        "amplitude_delta_l2_relative_to_original": _relative_norm(delta, amplitude),
        "amplitude_delta_linf": float(delta.abs().max()),
        "intensity_delta_mean": float(intensity_delta.mean()),
        "intensity_delta_l2": float(torch.linalg.vector_norm(intensity_delta)),
        "intensity_delta_l2_relative_to_original": _relative_norm(intensity_delta, amplitude.square()),
        "intensity_delta_linf": float(intensity_delta.abs().max()),
        "response_zero_component_l2": float(torch.linalg.vector_norm(d0)),
        "response_rest_component_l2": float(torch.linalg.vector_norm(drest)),
        "response_total_delta_l2": float(torch.linalg.vector_norm(observed_delta)),
        "response_zero_energy": energy["zero_energy"],
        "response_rest_energy": energy["rest_energy"],
        "response_coherent_cross_energy": energy["cross_energy"],
        "response_total_energy": energy["total_energy"],
        "response_zero_energy_fraction": energy["zero_fraction"],
        "response_rest_energy_fraction": energy["rest_fraction"],
        "response_coherent_cross_energy_fraction": energy["cross_fraction"],
        "response_energy_closure_relative_error": energy["closure_relative_error"],
        "raw_coefficient_delta_relative_to_continuous": _relative_norm(observed_delta, zc),
        "phase_aligned_coefficient_delta_relative_to_continuous": phase_error,
        "continuous_increment_phase_rad_2048": phase,
        "matrix_continuous_prediction_relative_error": predicted_cont_error,
        "matrix_quantized_prediction_relative_error": predicted_quant_error,
        "matrix_delta_prediction_relative_error": predicted_delta_error,
        "matrix_delta_closure_relative_error": closure_error,
        "wrong_intensity_as_amplitude_prediction_relative_error": wrong_intensity_error,
    }
    target_errors: list[float] = []
    direct_cmos_errors: list[float] = []
    cmos_increment_errors: list[float] = []
    display_errors: list[float] = []
    display_linf: list[float] = []
    for grid in GRIDS:
        output = outputs[grid]
        baseline = continuous_outputs[grid]
        raw = output["raw_cmos_intensity_w_per_m2"].double()
        base_raw = baseline["raw_cmos_intensity_w_per_m2"].double()
        target_error = _phase_aligned_error(output["actual_coefficients"],
                                            inputs["target"][index])[1]
        direct_cmos_error = _relative(raw, inputs["direct_cmos"][str(grid)][index, 0].double())
        cmos_increment = _relative(raw, base_raw)
        view = _display_view(raw, float(output["camera_pixel_area_m2"]), inputs["camera_gain"])
        base_view = _display_view(base_raw, float(baseline["camera_pixel_area_m2"]),
                                  inputs["camera_gain"])
        display_error = _relative(view, base_view)
        display_max = float((view - base_view).abs().max())
        area = float(output["camera_pixel_area_m2"])
        wrong_squared_power = float(raw.square().sum()) * area
        saved_power = float(output["camera_total_power_w"])
        wrong_squared_power_error = abs(wrong_squared_power - saved_power) / max(abs(saved_power), 1e-30)
        target_errors.append(target_error)
        direct_cmos_errors.append(direct_cmos_error)
        cmos_increment_errors.append(cmos_increment)
        display_errors.append(display_error)
        display_linf.append(display_max)
        intensity = inputs["intensity_a8"][index] if condition == CONDITIONS[0] \
            else inputs["intensity_i8"][index]
        power = _power_checks(inputs, output, intensity, grid, identity["sample_key"])
        for name, value in power.items():
            row[f"{name}_{grid}"] = value
        row[f"target_coefficient_error_{grid}"] = target_error
        row[f"direct_cmos_error_{grid}"] = direct_cmos_error
        row[f"increment_cmos_error_{grid}"] = cmos_increment
        row[f"display_view28_increment_relative_error_{grid}"] = display_error
        row[f"display_view28_increment_linf_{grid}"] = display_max
        row[f"wrong_squared_cmos_integral_relative_error_{grid}"] = wrong_squared_power_error
        for field in POWER_FIELDS:
            row[f"{field}_{grid}"] = float(output["power_trace_w"][field])
            row[f"{field}_continuous_{grid}"] = float(baseline["power_trace_w"][field])
    cross_coeff = _relative(outputs[2048]["actual_coefficients"],
                            outputs[4096]["actual_coefficients"])
    cross_cmos = _relative(outputs[2048]["raw_cmos_intensity_w_per_m2"],
                           outputs[4096]["raw_cmos_intensity_w_per_m2"])
    _require(cross_coeff <= 1.0e-4 and cross_cmos <= 1.0e-4,
             f"quantized cross-grid window failed: {identity['sample_key']} {condition}")
    row["cross_grid_coefficient_relative_error"] = cross_coeff
    row["cross_grid_cmos_intensity_relative_error"] = cross_cmos
    row["target_coefficient_relative_error_worst_grid"] = max(target_errors)
    row["direct_cmos_relative_error_worst_grid"] = max(direct_cmos_errors)
    row["increment_cmos_relative_error_worst_grid"] = max(cmos_increment_errors)
    row["display_view28_increment_relative_error_worst_grid"] = max(display_errors)
    row["display_view28_increment_linf_worst_grid"] = max(display_linf)
    row["original_precision_witness"] = bool(
        max(target_errors) <= 1.0e-4 and max(direct_cmos_errors) <= 1.0e-4
        and cross_coeff <= 1.0e-4 and cross_cmos <= 1.0e-4
    )
    return row


def _run_analysis(
    inputs: dict[str, Any], selected: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identities = inputs["identities"]
    selected_keys = None if selected is None else {row["sample_key"] for row in selected}
    indices = [i for i, row in enumerate(identities)
               if selected_keys is None or row["sample_key"] in selected_keys]
    continuous_prediction = (
        inputs["matrix"] @ inputs["amplitude"][indices].T.to(torch.complex128)
    ).T.contiguous()
    matrix_terms = {condition: _matrix_terms(inputs, condition, indices, continuous_prediction)
                    for condition in CONDITIONS}
    quantized_records: list[dict[str, Any]] = []
    continuous_records: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    for term_index, index in enumerate(indices):
        identity = identities[index]
        continuous_outputs = {grid: _load_output(inputs, identity, "continuous", grid)
                              for grid in GRIDS}
        continuous_records.append(_continuous_record(inputs, index, continuous_outputs))
        for condition in CONDITIONS:
            outputs = {grid: _load_output(inputs, identity, condition, grid) for grid in GRIDS}
            row = _quantized_record(inputs, condition, index, term_index, matrix_terms[condition],
                                    continuous_prediction, outputs, continuous_outputs)
            quantized_records.append(row)
            for grid in GRIDS:
                current_trace = outputs[grid]["power_trace_w"]
                base_trace = continuous_outputs[grid]["power_trace_w"]
                power_row: dict[str, Any] = {**identity, "condition": condition, "grid": grid}
                for field in POWER_FIELDS:
                    value = float(current_trace[field])
                    baseline = float(base_trace[field])
                    delta = value - baseline
                    power_row[f"{field}_current"] = value
                    power_row[f"{field}_continuous"] = baseline
                    power_row[f"{field}_delta_W"] = delta
                    power_row[f"{field}_delta_relative_to_continuous"] = (
                        delta / abs(baseline) if abs(baseline) > 1e-30 else 0.0
                    )
                power_rows.append(power_row)
    expected = len(indices)
    _require(len(quantized_records) == expected * 2
             and len(continuous_records) == expected
             and len(power_rows) == expected * 2 * len(GRIDS),
             "saved-evidence analysis population changed")
    return quantized_records, continuous_records, power_rows


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), "summary values are empty or nonfinite")
    return {
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _summarize(quantized: list[dict[str, Any]],
               continuous: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in ALL_CONDITIONS:
        rows = continuous if condition == "continuous" else [
            row for row in quantized if row["condition"] == condition
        ]
        _require(len(rows) == 200, f"summary denominator changed: {condition}")
        result[condition] = {
            "denominator": len(rows),
            "passed": sum(bool(row["original_precision_witness"]) for row in rows),
            "coefficient_error_failures": sum(
                row["target_coefficient_relative_error_worst_grid"] > 1e-4 for row in rows
            ),
            "cmos_error_failures": sum(
                row["direct_cmos_relative_error_worst_grid"] > 1e-4 for row in rows
            ),
            "both_coefficient_and_cmos_failures": sum(
                row["target_coefficient_relative_error_worst_grid"] > 1e-4
                and row["direct_cmos_relative_error_worst_grid"] > 1e-4 for row in rows
            ),
            "raw_coefficient_delta_relative_to_continuous": _stats([
                row["raw_coefficient_delta_relative_to_continuous"] for row in rows
            ]),
            "phase_aligned_coefficient_delta_relative_to_continuous": _stats([
                row["phase_aligned_coefficient_delta_relative_to_continuous"] for row in rows
            ]),
            "target_coefficient_error": _stats([
                row["target_coefficient_relative_error_worst_grid"] for row in rows
            ]),
            "increment_cmos_error": _stats([
                row["increment_cmos_relative_error_worst_grid"] for row in rows
            ]),
            "direct_cmos_error": _stats([
                row["direct_cmos_relative_error_worst_grid"] for row in rows
            ]),
            "display_view28_increment_error": _stats([
                row["display_view28_increment_relative_error_worst_grid"] for row in rows
            ]),
            "zero_energy_fraction": _stats([row["response_zero_energy_fraction"] for row in rows]),
            "rest_energy_fraction": _stats([row["response_rest_energy_fraction"] for row in rows]),
            "coherent_cross_energy_fraction": _stats([
                row["response_coherent_cross_energy_fraction"] for row in rows
            ]),
            "by_method_class": {},
        }
        for method in METHODS:
            for label in range(10):
                group = [row for row in rows if row["method"] == method and row["label"] == label]
                _require(len(group) == 10, f"class denominator changed for {condition} {method}:{label}")
                result[condition]["by_method_class"][f"{method}:{label}"] = {
                    "denominator": len(group),
                    "passed": sum(bool(row["original_precision_witness"]) for row in group),
                    "raw_coefficient_delta_relative_to_continuous": _stats([
                        row["raw_coefficient_delta_relative_to_continuous"] for row in group
                    ]),
                    "phase_aligned_coefficient_delta_relative_to_continuous": _stats([
                        row["phase_aligned_coefficient_delta_relative_to_continuous"] for row in group
                    ]),
                    "target_coefficient_error": _stats([
                        row["target_coefficient_relative_error_worst_grid"] for row in group
                    ]),
                    "increment_cmos_error": _stats([
                        row["increment_cmos_relative_error_worst_grid"] for row in group
                    ]),
                    "direct_cmos_error": _stats([
                        row["direct_cmos_relative_error_worst_grid"] for row in group
                    ]),
                }
    _require([result[condition]["passed"] for condition in ALL_CONDITIONS] == [200, 2, 0],
             "recomputed original precision witnesses differ from 200/2/0")
    _require(result[CONDITIONS[0]]["coefficient_error_failures"] == 198
             and result[CONDITIONS[0]]["both_coefficient_and_cmos_failures"] == 190,
             "recomputed A8 failure pattern differs from the prior fixed result")
    return result


def _input_perturbation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        _require(len(rows) == 200, f"input summary denominator changed: {condition}")
        result[condition] = {
            "amplitude_bias_mean": float(np.mean([row["amplitude_delta_mean"] for row in rows])),
            "amplitude_l2_relative": _stats([
                row["amplitude_delta_l2_relative_to_original"] for row in rows
            ]),
            "amplitude_linf": _stats([row["amplitude_delta_linf"] for row in rows]),
            "intensity_bias_mean": float(np.mean([row["intensity_delta_mean"] for row in rows])),
            "intensity_l2_relative": _stats([
                row["intensity_delta_l2_relative_to_original"] for row in rows
            ]),
            "intensity_linf": _stats([row["intensity_delta_linf"] for row in rows]),
            "zeroed_original_positive_fraction": _stats([
                row["zeroed_positive_fraction_original_positive"] for row in rows
            ]),
        }
    return result


def _negative_controls(records: list[dict[str, Any]]) -> dict[str, Any]:
    _require(records, "negative controls require saved attribution rows")
    correct_errors = [float(row["matrix_delta_prediction_relative_error"]) for row in records]
    wrong_errors = [float(row["wrong_intensity_as_amplitude_prediction_relative_error"])
                    for row in records]
    wrong_index = max(range(len(records)), key=lambda i: wrong_errors[i] - correct_errors[i])
    square_errors = [float(row[f"wrong_squared_cmos_integral_relative_error_{grid}"])
                     for row in records for grid in GRIDS]
    square_index = int(np.argmax(square_errors))
    synthetic_zero = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    synthetic_rest = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    total_energy = float(torch.vdot(synthetic_zero + synthetic_rest,
                                    synthetic_zero + synthetic_rest).real)
    without_cross = float(torch.vdot(synthetic_zero, synthetic_zero).real
                          + torch.vdot(synthetic_rest, synthetic_rest).real)
    raw = torch.tensor([[0.25, 0.5], [0.75, 0.9]], dtype=torch.float64)
    correct_integral = float(raw.sum())
    wrong_squared_integral = float(raw.square().sum())
    reference = torch.tensor([0.2 + 0.3j, -0.5 + 0.1j], dtype=torch.complex128)
    rotated = reference * torch.exp(torch.tensor(0.731j, dtype=torch.complex128))
    _, phase_error = _phase_aligned_error(rotated, reference)
    _require(wrong_errors[wrong_index] > correct_errors[wrong_index] + 1e-8
             and square_errors[square_index] > 1e-3
             and abs(total_energy - without_cross) > 0.0
             and abs(correct_integral - wrong_squared_integral) > 0.0
             and phase_error <= 1e-12,
             "analytic negative or common-phase control failed")
    return {
        "intensity_delta_misused_as_amplitude_delta": {
            "rejected": True, "sample_key": records[wrong_index]["sample_key"],
            "condition": records[wrong_index]["condition"],
            "correct_error": correct_errors[wrong_index], "wrong_error": wrong_errors[wrong_index],
        },
        "omitted_coherent_cross_term": {
            "rejected": True, "synthetic_complete_energy": total_energy,
            "synthetic_without_cross_energy": without_cross,
            "maximum_observed_absolute_cross_fraction": max(
                abs(float(row["response_coherent_cross_energy_fraction"])) for row in records
            ),
        },
        "cmos_intensity_squared_again": {
            "rejected": True, "sample_key": records[square_index // 2]["sample_key"],
            "synthetic_integral": correct_integral,
            "synthetic_wrong_squared_integral": wrong_squared_integral,
            "max_saved_power_mismatch": square_errors[square_index],
        },
        "common_phase_equivalent_positive_control": {
            "passed": phase_error <= 1e-12, "phase_aligned_relative_error": phase_error,
        },
    }


def _power_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for grid in GRIDS:
            group = [row for row in rows if row["condition"] == condition and row["grid"] == grid]
            summary[condition][str(grid)] = {
                field: {
                    "delta_W": _stats([row[f"{field}_delta_W"] for row in group]),
                    "relative_delta": _stats([
                        row[f"{field}_delta_relative_to_continuous"] for row in group
                    ]),
                }
                for field in POWER_FIELDS
            }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(rows, f"cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_preflight(identities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for row in identities:
        selected.setdefault((row["method"], int(row["label"])), row)
    _require(len(selected) == 20, "preflight did not select one row per method and class")
    return [selected[(method, label)] for method in METHODS for label in range(10)]


def _save_atlas(path: Path, inputs: dict[str, Any], selected: list[dict[str, Any]]) -> None:
    tile, label_height, gap, columns = 128, 26, 6, 6
    canvas = Image.new("RGB", (columns * tile + (columns + 1) * gap,
                               len(selected) * (tile + label_height) + (len(selected) + 1) * gap),
                       "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    headers = ("original a", "A8 gray", "I8 gray", "continuous CMOS", "A8 CMOS", "I8 CMOS")
    for col, label in enumerate(headers):
        draw.text((gap + col * (tile + gap), 2), label, fill="black", font=font)
    index_by_key = {row["sample_key"]: i for i, row in enumerate(inputs["identities"])}
    for row_number, identity in enumerate(selected):
        index = index_by_key[identity["sample_key"]]
        y = gap + row_number * (tile + label_height)
        images = [
            Image.fromarray(inputs["amplitude"][index].reshape(*IMAGE_SHAPE)
                            .clamp(0, 1).mul(255).round().byte().numpy()),
            Image.fromarray(inputs["gray_a"][index].reshape(*IMAGE_SHAPE).numpy()),
            Image.fromarray(inputs["gray_i"][index].reshape(*IMAGE_SHAPE).numpy()),
        ]
        for condition in ALL_CONDITIONS:
            output = _load_output(inputs, identity, condition, 2048)
            view = _display_view(output["raw_cmos_intensity_w_per_m2"],
                                 float(output["camera_pixel_area_m2"]), inputs["camera_gain"])
            images.append(Image.fromarray(view.clamp(0, 1).mul(255).round().byte().numpy())
                          .resize((tile, tile), Image.Resampling.NEAREST))
        images = [image.resize((tile, tile), Image.Resampling.NEAREST) for image in images[:3]] + images[3:]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (gap + col * (tile + gap), y + label_height))
        draw.text((gap, y), f"{identity['sample_key']} label={identity['label']}",
                  fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_report(path: Path, summary: dict[str, Any], powers: dict[str, Any],
                  controls: dict[str, Any], preflight: dict[str, Any],
                  records: list[dict[str, Any]]) -> None:
    input_summary = _input_perturbation_summary(records)
    lines = [
        "# 冻结 8 位回放的保存证据误差归因", "",
        "## 结果与状态", "",
        f"归因有效。固定人口复算通过数：连续参考 {summary['continuous']['passed']}/200，"
        f"A8 {summary[CONDITIONS[0]]['passed']}/200，I8 {summary[CONDITIONS[1]]['passed']}/200。"
        f"A8 有 {summary[CONDITIONS[0]]['coefficient_error_failures']} 条系数误差超限，"
        f"其中 {summary[CONDITIONS[0]]['both_coefficient_and_cmos_failures']} 条 CMOS 误差也超限。",
        "",
        "## 科学认识", "",
        "2048 网格响应矩阵逐条预测连续与量化后的复系数，并闭合输入振幅变化到观测 OLS 系数差。"
        "置零区域、其余取整区域与相干交叉项分开计量；功率依据保存的完整物理 trace，CMOS 变化依据原生强度和"
        "固定增益裁剪视图。因为没有相机复场，不能识别 Decoder 内部相位贡献。",
        "",
        f"I8 的理论置零条件为 a < sqrt(0.5/255) = {math.sqrt(0.5 / 255.0):.8f}；"
        "逐样本像素数与比例见归因表，不预设其主导误差。",
        "",
        "| 条件 | 通过 | 系数超限 | CMOS超限 | OLS系数变化 median/P95/max | CMOS增量 median/P95/max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ALL_CONDITIONS:
        row = summary[condition]
        coeff = row["raw_coefficient_delta_relative_to_continuous"]
        cmos = row["increment_cmos_error"]
        lines.append(
            f"| {condition} | {row['passed']}/200 | {row['coefficient_error_failures']} | "
            f"{row['cmos_error_failures']} | {coeff['median']:.3g}/{coeff['p95']:.3g}/{coeff['max']:.3g} | "
            f"{cmos['median']:.3g}/{cmos['p95']:.3g}/{cmos['max']:.3g} |"
        )
    lines.extend([
        "", "### 输入扰动与相干响应分解", "",
        "偏差为逐样本空间均值再对样本取平均；L2 与 L∞ 列为 median/P95/max。置零比例以原振幅大于零的像素为分母。",
        "", "| 条件 | 振幅偏差均值 | 振幅相对L2 | 振幅L∞ | 强度偏差均值 | 强度相对L2 | 强度L∞ | 原正振幅置零比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        row = input_summary[condition]
        amplitude_l2, amplitude_linf = row["amplitude_l2_relative"], row["amplitude_linf"]
        intensity_l2, intensity_linf = row["intensity_l2_relative"], row["intensity_linf"]
        zeroed = row["zeroed_original_positive_fraction"]
        lines.append(
            f"| {condition} | {row['amplitude_bias_mean']:.3g} | "
            f"{amplitude_l2['median']:.3g}/{amplitude_l2['p95']:.3g}/{amplitude_l2['max']:.3g} | "
            f"{amplitude_linf['median']:.3g}/{amplitude_linf['p95']:.3g}/{amplitude_linf['max']:.3g} | "
            f"{row['intensity_bias_mean']:.3g} | "
            f"{intensity_l2['median']:.3g}/{intensity_l2['p95']:.3g}/{intensity_l2['max']:.3g} | "
            f"{intensity_linf['median']:.3g}/{intensity_linf['p95']:.3g}/{intensity_linf['max']:.3g} | "
            f"{zeroed['median']:.3g}/{zeroed['p95']:.3g}/{zeroed['max']:.3g} |"
        )
    lines.extend([
        "", "响应能量列按 `||d0||²`、`||d+||²` 和 `2 Re(d0†d+)` 除以总差分能量；交叉项可正负，分量比例可超出 0–100%，三项逐样本相加为 100%。",
        "", "| 条件 | 置零响应能量比例 median/P95/max | 其余像素响应能量比例 median/P95/max | 相干交叉项比例 median/P95/max |",
        "|---|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        row = summary[condition]
        zero, rest, cross = (row["zero_energy_fraction"], row["rest_energy_fraction"],
                             row["coherent_cross_energy_fraction"])
        lines.append(
            f"| {condition} | {zero['median']:.3g}/{zero['p95']:.3g}/{zero['max']:.3g} | "
            f"{rest['median']:.3g}/{rest['p95']:.3g}/{rest['max']:.3g} | "
            f"{cross['median']:.3g}/{cross['p95']:.3g}/{cross['max']:.3g} |"
        )
    lines.extend(["", "### 方法与类别残差", "",
                  "| 条件 | 方法 | 类别 | 分母 | 通过 | OLS系数变化 median/P95/max | 目标系数误差 median/P95/max | CMOS增量 median/P95/max | 直接参考CMOS误差 median/P95/max |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for condition in ALL_CONDITIONS:
        for method in METHODS:
            for label in range(10):
                row = summary[condition]["by_method_class"][f"{method}:{label}"]
                coeff = row["raw_coefficient_delta_relative_to_continuous"]
                target = row["target_coefficient_error"]
                cmos = row["increment_cmos_error"]
                direct = row["direct_cmos_error"]
                lines.append(
                    f"| {condition} | {method} | {label} | {row['denominator']} | {row['passed']} | "
                    f"{coeff['median']:.3g}/{coeff['p95']:.3g}/{coeff['max']:.3g} | "
                    f"{target['median']:.3g}/{target['p95']:.3g}/{target['max']:.3g} | "
                    f"{cmos['median']:.3g}/{cmos['p95']:.3g}/{cmos['max']:.3g} | "
                    f"{direct['median']:.3g}/{direct['p95']:.3g}/{direct['max']:.3g} |"
                )
    lines.extend([
        "", "### 各面的功率差", "",
        "逐样本 W 与相对差见 power_attribution.csv。P_before_OLS_incident_W 对应 Gaussian OLS 入射面；"
        "P_after_OLS_reemitted_decoder_entry_W 对应 OLS 再发射后的 Decoder 入口。",
        "", "| 条件 | 网格 | 面 | 功率差 W median/P95/max | 相对差 median/P95/max |",
        "|---|---:|---|---:|---:|",
    ])
    for condition in CONDITIONS:
        for grid in GRIDS:
            for field in POWER_FIELDS:
                item = powers[condition][str(grid)][field]
                d, r = item["delta_W"], item["relative_delta"]
                lines.append(
                    f"| {condition} | {grid} | {field} | {d['median']:.3g}/{d['p95']:.3g}/{d['max']:.3g} | "
                    f"{r['median']:.3g}/{r['p95']:.3g}/{r['max']:.3g} |"
                )
    lines.extend([
        "", "## 目标缺口", "",
        "结论限定于固定 200 条指令、两种理想量化校准和已保存模拟数据。器件 LUT、残余相位、有限消光、"
        "反射损耗、相机复场及实际光路未测量；这不构成硬件验收、语义可读性或离散命令可达性判断。",
        "", "## 工程完成与验证", "",
        f"固定 20 条预检、两种编码共 {preflight['condition_rows']} 行通过。"
        "复系数预测与振幅扰动闭合阈值为 1e-6，功率积分核对阈值为 1e-10，两网格沿用 1e-4。"
        "最终消费者从保存灰阶、响应矩阵、目标及 1,200 份输出重算 400 条归因、分母和通过数。",
        "",
        "负控识别强度扰动误作振幅、遗漏相干交叉项及 CMOS 强度再次平方；公共相位等效正控通过。"
        "固定 20 条输入与 CMOS 对照图为 fixed_20_comparison.png。",
        "", "## 决策影响与下一步", "",
        "唯一判断：保存证据把本批量化扰动到 OLS 系数的变化数值闭合，并量化其对应的功率、原生 CMOS 强度"
        "和固定增益裁剪视图变化；相机复场缺失使 Decoder 内部相位贡献无法识别。原 200/2/0 结论不变。"
        "本轮到此停止；实物 LUT、离散命令优化、量化感知训练和一次前向映射另行设计。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_identity(run_dir: Path, config: dict[str, Any], allow_closed: bool) -> dict[str, Any]:
    _require(run_dir.resolve() == EXPECTED_ATTEMPT.resolve(),
             "output path differs from the fixed attempt", ValueError)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "run_identity.json"
    snapshot_path = run_dir / "run_config_snapshot.json"
    if identity_path.is_file():
        identity = _read_json(identity_path)
        _require(identity.get("task_id") == TASK_ID
                 and identity.get("status") in {"ACTIVE", "CLOSED"}
                 and (allow_closed or identity.get("status") != "CLOSED"),
                 "output directory belongs to a different or completed attempt")
        _require(snapshot_path.is_file()
                 and _read_json(snapshot_path) == {
                     "config_path": "configs/research/lightgen/frozen_slm_8bit_error_attribution_v1.yaml",
                     "config": config,
                 }, "saved attribution configuration changed")
        return identity
    _require(not allow_closed, "verify-only needs an existing completed attempt", FileNotFoundError)
    identity = {
        "schema": "lightgen_frozen_slm_8bit_error_attribution_run.v1",
        "task_id": TASK_ID, "run_id": uuid.uuid4().hex, "status": "ACTIVE",
        "source_replay_run_id": config["source"]["expected_replay_run_id"],
        "source_reachability_run_id": config["source"]["expected_reachability_run_id"],
        "zero_update_scope": {
            "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
            "input_resolve": 0, "new_sampling": 0, "new_human_review": 0,
        },
    }
    _atomic_json(identity_path, identity)
    _atomic_json(snapshot_path, {
        "config_path": "configs/research/lightgen/frozen_slm_8bit_error_attribution_v1.yaml",
        "config": config,
    })
    return identity


def _write_preflight(run_dir: Path, identity: dict[str, Any],
                     inputs: dict[str, Any]) -> dict[str, Any]:
    selected = _select_preflight(inputs["identities"])
    quantized, continuous, power_rows = _run_analysis(inputs, selected)
    controls = _negative_controls(quantized)
    errors = [
        value for row in quantized for value in (
            row["matrix_continuous_prediction_relative_error"],
            row["matrix_quantized_prediction_relative_error"],
            row["matrix_delta_prediction_relative_error"],
            row["matrix_delta_closure_relative_error"],
        )
    ]
    checks = {
        "fixed_20_sample_identities": len(selected) == 20
            and all(row["sample_key"] in {item["sample_key"] for item in inputs["identities"]}
                    for row in selected),
        "two_encoding_conditions_40_rows": len(quantized) == 40,
        "continuous_matrix_predictions_le_1e-6": max(
            row["matrix_continuous_prediction_relative_error"] for row in quantized
        ) <= 1e-6,
        "quantized_matrix_predictions_and_delta_closure_le_1e-6": max(errors) <= 1e-6,
        "2048_response_only_and_2048_4096_cross_grid_le_1e-4": max(
            max(row["cross_grid_coefficient_relative_error"],
                row["cross_grid_cmos_intensity_relative_error"]) for row in quantized
        ) <= 1e-4,
        "power_and_camera_integrals_recomputed": len(power_rows) == 80,
        "analytic_negative_and_common_phase_controls_pass": all(
            value.get("rejected", value.get("passed", False)) for value in controls.values()
        ),
        "continuous_source_replay": max(
            max(row["source_continuous_coefficient_replay_relative_error_2048"],
                row["source_continuous_cmos_replay_relative_error_2048"],
                row["source_continuous_coefficient_replay_relative_error_4096"],
                row["source_continuous_cmos_replay_relative_error_4096"]) for row in continuous
        ) <= 1e-6,
    }
    _require(all(checks.values()), "saved-evidence 20-row preflight failed")
    value = {
        "schema": "lightgen_frozen_slm_8bit_error_attribution_preflight.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"], "status": "PREFLIGHT_PASSED",
        "selected_samples": selected, "condition_rows": len(quantized),
        "maximum_matrix_prediction_or_delta_closure_error": max(errors),
        "checks": checks, "negative_controls": controls,
        "quantized_records": quantized, "continuous_records": continuous,
        "zero_update_scope": identity["zero_update_scope"],
    }
    _atomic_json(run_dir / "preflight.json", value)
    return value


def _assert_records_equal(saved: list[dict[str, Any]], current: list[dict[str, Any]],
                          expected_count: int) -> None:
    _require(len(saved) == len(current) == expected_count,
             f"record denominator changed: expected {expected_count}")
    saved_by_id = {(row["condition"], row["sample_key"]): row for row in saved}
    _require(len(saved_by_id) == expected_count, "saved record identities are duplicated")
    for row in current:
        key = (row["condition"], row["sample_key"])
        _require(key in saved_by_id, f"saved evidence row is missing: {key}")
        other = saved_by_id[key]
        _require(set(row) == set(other), f"saved record fields changed: {key}")
        for field, value in row.items():
            old = other[field]
            if isinstance(value, bool):
                _require(old is value, f"saved evidence differs after recomputation: {key} {field}")
            elif isinstance(value, (int, float)):
                _require(isinstance(old, (int, float))
                         and abs(float(value) - float(old)) <= max(1e-12, abs(float(value)) * 1e-9),
                         f"saved evidence differs after recomputation: {key} {field}")
            else:
                _require(value == old, f"saved evidence identity differs: {key} {field}")


def _verify_csv(path: Path, recomputed: list[dict[str, Any]]) -> None:
    _require(path.is_file(), f"saved table is missing: {path}", FileNotFoundError)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        saved = list(csv.DictReader(stream))
    _require(len(saved) == len(recomputed), f"saved table denominator changed: {path.name}")
    key_fields = ("condition", "sample_key", "grid") if "grid" in saved[0] \
        else ("condition", "sample_key")
    saved_map = {tuple(row[field] for field in key_fields): row for row in saved}
    _require(len(saved_map) == len(saved), f"saved table contains duplicate rows: {path.name}")
    for row in recomputed:
        key = tuple(str(row[field]) for field in key_fields)
        _require(key in saved_map, f"saved table row is missing: {path.name} {key}")
        for field, value in row.items():
            _require(field in saved_map[key], f"saved table column is missing: {path.name} {field}")
            saved_value = saved_map[key][field]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _require(abs(float(saved_value) - float(value)) <= max(1e-12, abs(float(value)) * 1e-9),
                         f"saved table differs after recomputation: {path.name} {key} {field}")
            else:
                _require(str(value) == saved_value,
                         f"saved table identity differs: {path.name} {key} {field}")


def _summarize_power(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for grid in GRIDS:
            group = [row for row in rows if row["condition"] == condition and row["grid"] == grid]
            summary[condition][str(grid)] = {
                field: {
                    "delta_W": _stats([row[f"{field}_delta_W"] for row in group]),
                    "relative_delta": _stats([
                        row[f"{field}_delta_relative_to_continuous"] for row in group
                    ]),
                }
                for field in POWER_FIELDS
            }
    return summary


def _execute(run_dir: Path, identity: dict[str, Any],
             inputs: dict[str, Any]) -> dict[str, Any]:
    preflight = _read_json(run_dir / "preflight.json")
    _require(preflight.get("run_id") == identity["run_id"]
             and preflight.get("status") == "PREFLIGHT_PASSED"
             and all(preflight.get("checks", {}).values()),
             "full analysis requires the passing fixed 20-sample preflight")
    _require(not (run_dir / "final_decision.json").exists(),
             "attempt is already finalized; use verify-only")
    records, continuous, power_rows = _run_analysis(inputs)
    _require(len(records) == 400 and len(continuous) == 200 and len(power_rows) == 800,
             "full saved-evidence population is incomplete")
    preflight_keys = {row["sample_key"] for row in preflight["selected_samples"]}
    _assert_records_equal(preflight["quantized_records"],
                          [row for row in records if row["sample_key"] in preflight_keys], 40)
    _assert_records_equal(preflight["continuous_records"],
                          [row for row in continuous if row["sample_key"] in preflight_keys], 20)
    summary = _summarize(records, continuous)
    controls = _negative_controls(records)
    power_summary = _summarize_power(power_rows)
    _write_csv(run_dir / "input_ols_cmos_attribution.csv", records)
    _write_csv(run_dir / "power_attribution.csv", power_rows)
    _save_atlas(run_dir / "fixed_20_comparison.png", inputs,
                _select_preflight(inputs["identities"]))
    final = {
        "schema": "lightgen_frozen_slm_8bit_error_attribution_final.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "status": "VALID_SAVED_EVIDENCE_ATTRIBUTION",
        "decision": "INPUT_TO_OLS_COEFFICIENT_CHANGE_CLOSED; CMOS_INTERNAL_PHASE_UNIDENTIFIABLE",
        "sample_count": 200, "attribution_record_count": len(records),
        "power_record_count": len(power_rows),
        "original_precision_counts": {
            condition: summary[condition]["passed"] for condition in ALL_CONDITIONS
        },
        "conditions": summary, "power_summary": power_summary,
        "negative_controls": controls,
        "max_matrix_error": max(
            value for row in records for value in (
                row["matrix_continuous_prediction_relative_error"],
                row["matrix_quantized_prediction_relative_error"],
                row["matrix_delta_prediction_relative_error"],
                row["matrix_delta_closure_relative_error"],
            )
        ),
        "max_cross_grid_coefficient_error": max(
            row["cross_grid_coefficient_relative_error"] for row in records
        ),
        "max_cross_grid_cmos_error": max(
            row["cross_grid_cmos_intensity_relative_error"] for row in records
        ),
        "records": records, "continuous_reference_checks": continuous,
        "zero_update_scope": identity["zero_update_scope"],
        "limitations": [
            "The response matrix is used only for the saved 2048 grid.",
            "No saved CMOS complex field exists; Decoder internal phase contributions are unidentified.",
            "This result concerns saved ideal-encoding commands, not measured SLM response or hardware acceptance.",
        ],
    }
    _atomic_json(run_dir / "final_decision.json", final)
    _atomic_json(run_dir / "negative_control_results.json", controls)
    _write_report(run_dir / "report.md", summary, power_summary, controls, preflight, records)
    identity["status"] = "CLOSED"
    identity["final_decision"] = final["decision"]
    _atomic_json(run_dir / "run_identity.json", identity)
    return final


def _verify_only(run_dir: Path, identity: dict[str, Any],
                 inputs: dict[str, Any]) -> dict[str, Any]:
    final = _read_json(run_dir / "final_decision.json")
    _require(identity.get("status") == "CLOSED"
             and final.get("run_id") == identity["run_id"]
             and final.get("status") == "VALID_SAVED_EVIDENCE_ATTRIBUTION",
             "verify-only requires a completed valid attempt")
    records, continuous, power_rows = _run_analysis(inputs)
    _assert_records_equal(final.get("records", []), records, 400)
    _assert_records_equal(final.get("continuous_reference_checks", []), continuous, 200)
    summary = _summarize(records, continuous)
    controls = _negative_controls(records)
    power_summary = _summarize_power(power_rows)
    _require(final.get("conditions") == summary
             and final.get("negative_controls") == controls
             and final.get("power_summary") == power_summary,
             "final summaries differ from recomputed saved evidence")
    _require(final.get("original_precision_counts") ==
             {condition: summary[condition]["passed"] for condition in ALL_CONDITIONS},
             "original precision counts differ from recomputed evidence")
    _verify_csv(run_dir / "input_ols_cmos_attribution.csv", records)
    _verify_csv(run_dir / "power_attribution.csv", power_rows)
    _write_report(run_dir / "report.md", summary, power_summary, controls,
                  _read_json(run_dir / "preflight.json"), records)
    _atomic_json(run_dir / "windows_reconsumption.json", {
        "schema": "lightgen_frozen_slm_8bit_error_attribution_windows_reconsumption.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "status": "RECOMPUTED_AND_MATCHED",
        "attribution_records_recomputed": len(records),
        "continuous_reference_rows_recomputed": len(continuous),
        "saved_output_rows_revalidated": 1200,
        "power_rows_recomputed": len(power_rows),
        "original_precision_counts_recomputed": {
            condition: summary[condition]["passed"] for condition in ALL_CONDITIONS
        },
        "matrix_max_error_recomputed": max(
            value for row in records for value in (
                row["matrix_continuous_prediction_relative_error"],
                row["matrix_quantized_prediction_relative_error"],
                row["matrix_delta_prediction_relative_error"],
                row["matrix_delta_closure_relative_error"],
            )
        ),
        "consumer_recomputed_from_saved_gray_matrix_outputs_and_power_traces": True,
        "zero_update_scope": identity["zero_update_scope"],
    })
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=EXPECTED_ATTEMPT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "attribution config must be a mapping", ValueError)
    _validate_config(config, config_path)
    run_dir = args.output_dir.resolve()
    identity = _run_identity(run_dir, config, allow_closed=args.verify_only)
    inputs = _load_inputs(config)
    if args.preflight_only:
        result = _write_preflight(run_dir, identity, inputs)
        print(json.dumps({"status": result["status"], "run_id": identity["run_id"],
                          "condition_rows": result["condition_rows"]}, ensure_ascii=False))
    elif args.execute:
        result = _execute(run_dir, identity, inputs)
        print(json.dumps({"status": result["status"], "run_id": identity["run_id"],
                          "counts": result["original_precision_counts"]}, ensure_ascii=False))
    else:
        result = _verify_only(run_dir, identity, inputs)
        print(json.dumps({"status": "RECOMPUTED_AND_MATCHED", "run_id": identity["run_id"],
                          "counts": result["original_precision_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
