"""Replay frozen G1/G4 SLM commands under two ideal 8-bit encodings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as source_run


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_FROZEN_SLM_8BIT_REPLAY_V1"
METHODS = ("G1", "G4")
CONDITIONS = ("continuous", "A8_amplitude_linear", "I8_intensity_linear")
GRIDS = (2048, 4096)
IMAGE_SHAPE = (200, 200)
PIXEL_COUNT = 40_000
LEVELS = 255
PRECISION_LIMIT = 1.0e-4
SOURCE_REPLAY_LIMIT = 1.0e-6
POWER_TRACE_FIELDS = frozenset({
    "P_source_before_SLM_W", "P_after_SLM_W", "P_encoder_entrance_W",
    "P_encoder_layer_1_after_aperture_W", "P_encoder_layer_2_after_aperture_W",
    "P_before_OLS_incident_W", "P_after_OLS_reemitted_decoder_entry_W",
    "P_decoder_layer_1_after_aperture_W", "P_decoder_layer_2_after_aperture_W", "P_CMOS_W",
})
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/frozen_slm_8bit_replay_v1.yaml"
EXPECTED_ATTEMPT = ROOT / "outputs/20261025_lightgen_frozen_slm_8bit_replay_v1/runs/attempt_01"


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON mapping: {path}", ValueError)
    return value


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def quantize_u8(values: torch.Tensor) -> torch.Tensor:
    """Return uint8 codes using float64 round-half-up on the closed unit interval."""
    values = torch.as_tensor(values, dtype=torch.float64)
    _require(torch.isfinite(values).all().item(), "SLM command contains nonfinite values", ValueError)
    _require((values >= 0.0).all().item() and (values <= 1.0).all().item(),
             "SLM command is outside [0,1]", ValueError)
    codes = torch.floor(values * LEVELS + 0.5)
    _require((codes >= 0.0).all().item() and (codes <= LEVELS).all().item(),
             "rounded SLM code is outside uint8 range", ValueError)
    return codes.to(torch.uint8)


def decode_u8(codes: torch.Tensor) -> torch.Tensor:
    codes = torch.as_tensor(codes)
    _require(codes.dtype == torch.uint8, "saved SLM gray codes must be uint8", ValueError)
    return codes.to(torch.float64) / LEVELS


def _sample_identity(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_key": source_run._item_key(item),
        "method": item["method"],
        "cohort_row": int(item["cohort_row"]),
        "label": int(item["requested_label"]),
        "proposal_id": int(item["proposal_id"]),
        "component_id": int(item["component_id"]),
    }


def _validate_identity_population(expected: list[dict[str, Any]], observed: list[dict[str, Any]]) -> None:
    expected_keys = [row.get("sample_key") for row in expected]
    observed_keys = [row.get("sample_key") for row in observed]
    _require(len(expected_keys) == len(observed_keys) and len(set(expected_keys)) == len(expected_keys)
             and len(set(observed_keys)) == len(observed_keys)
             and set(expected_keys) == set(observed_keys),
             "saved population has a missing or duplicate sample identity")
    expected_by_key = {row["sample_key"]: row for row in expected}
    _require(all(expected_by_key[row["sample_key"]] == row for row in observed),
             "saved population identity fields differ from the fixed selection")


def _expected_config(config: dict[str, Any], path: Path) -> None:
    _require(config.get("schema") == "lightgen_frozen_slm_8bit_replay.v1"
             and config.get("task_id") == TASK_ID
             and config.get("evidence_tier") == "diagnostic_only",
             "8-bit replay configuration identity changed", ValueError)
    expected_path = (ROOT / "configs/research/lightgen/frozen_slm_8bit_replay_v1.yaml").resolve()
    _require(path.resolve() == expected_path, "8-bit replay must use the fixed config", ValueError)
    source = config["source"]
    _require(source.get("reachability_config") ==
             "configs/research/lightgen/frozen_slm_ols_input_reachability_v1.yaml"
             and source.get("reachability_attempt_dir") ==
             "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01"
             and source.get("expected_source_task_id") == source_run.TASK_ID
             and source.get("expected_source_run_id") == "04390141af8c48278cfd3578f3688e0c"
             and source.get("expected_methods") == list(METHODS)
             and source.get("expected_samples_per_method") == 100
             and source.get("expected_samples_per_class_per_method") == 10,
             "fixed reachability input or population changed", ValueError)
    slm = config["slm"]
    _require(slm.get("paper_model") == "HES6001" and slm.get("resolution") == [1920, 1080]
             and slm.get("pixel_pitch_um") == 8.0 and slm.get("wavelength_nm") == 532
             and slm.get("input_frame_rate_hz") == 60 and slm.get("code_levels") == 256
             and slm.get("encoding_rounding") == "floor_255x_plus_0_5"
             and slm.get("amplitude_calibration_hypothesis") == "gray_equals_quantized_amplitude"
             and slm.get("intensity_calibration_hypothesis") == "gray_equals_quantized_intensity"
             and slm.get("measured_gray_response_curve_available") is False,
             "device evidence or fixed 8-bit calibration hypotheses changed", ValueError)
    _require(config.get("device_sources") == {
        "paper_supplement": "research_assets/lightgen/science_adv7434/paper/sciencesm.pdf",
        "manufacturer_product_page": "https://holoeye.com/products/spatial-light-modulators/amplitude-slms/",
    }, "device provenance sources changed", ValueError)
    optics = config["optics"]
    _require(optics.get("ols_basis") == "gaussian_s12_v1"
             and optics.get("propagation_grids") == list(GRIDS)
             and optics.get("encoder_layers") == 2 and optics.get("decoder_layers") == 2
             and optics.get("phase_quantization") == "none_frozen_checkpoint_values"
             and optics.get("input_shape") == list(IMAGE_SHAPE)
             and math.isclose(float(optics.get("source_power_w", 0.0)), 2.493e-4, rel_tol=0.0, abs_tol=1e-12)
             and optics.get("global_gain") == "frozen_C_checkpoint"
             and optics.get("runtime_config") == "configs/research/lightgen/small_gaussian_system_baseline_v1.yaml"
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
             "fixed propagation, gain, phase or geometry assumptions changed", ValueError)
    measurement = config["measurement"]
    _require(config.get("conditions") == list(CONDITIONS)
             and measurement.get("precision_limit") == PRECISION_LIMIT
             and measurement.get("source_output_replay_limit") == SOURCE_REPLAY_LIMIT
             and measurement.get("reference_replay_rtol") == 2.0e-6
             and measurement.get("reference_replay_atol") == 5.0e-8
             and measurement.get("preflight_rows") == "first_per_method_per_class"
             and measurement.get("expected_preflight_rows") == 20
             and all(measurement.get(key) == 0 for key in (
                 "model_updates", "adam_calls", "prior_fit", "new_sampling", "new_human_review", "input_resolve"
             )),
             "measurement scope, gates or zero-update boundary changed", ValueError)
    _require(config["output"].get("experiment_root") ==
             "outputs/20261025_lightgen_frozen_slm_8bit_replay_v1/runs/attempt_01"
             and config["output"].get("final_decision_consumer") ==
             "windows_recompute_from_saved_gray_and_production_outputs",
             "fixed output path or final consumer changed", ValueError)


def _source_inputs(config: dict[str, Any]) -> dict[str, Any]:
    reach_path = _resolve(config["source"]["reachability_config"])
    reach_config = source_run._load_yaml(reach_path)
    source_run._validate_config(reach_config, reach_path)
    inputs = source_run._load_inputs(reach_config)
    source_dir = _resolve(config["source"]["reachability_attempt_dir"])
    _require(source_dir.resolve() == _resolve(reach_config["output"]["experiment_root"]).resolve()
             and source_dir.is_dir(), "fixed upstream attempt directory is missing or changed", FileNotFoundError)
    identity = _load_json(source_dir / "run_identity.json")
    final = _load_json(source_dir / "final_decision.json")
    preflight = _load_json(source_dir / "preflight.json")
    ref_json = _load_json(source_dir / "reference_replay_final.json")
    _require(identity.get("task_id") == source_run.TASK_ID
             and identity.get("run_id") == config["source"]["expected_source_run_id"]
             and identity.get("status") == "CLOSED"
             and final.get("task_id") == source_run.TASK_ID
             and final.get("run_id") == identity["run_id"]
             and final.get("decision") == "ALL_200_NUMERICAL_COMMAND_WITNESSES_VERIFIED"
             and preflight.get("status") == "PREFLIGHT_PASSED"
             and ref_json.get("rows") == 200 and ref_json.get("all_passed") is True,
             "upstream frozen reachability run is not the specified completed evidence")
    commands = torch.load(source_dir / "amplitude_and_intensity_commands.pt", map_location="cpu", weights_only=False)
    compact = torch.load(source_dir / "compact_replay_evidence.pt", map_location="cpu", weights_only=False)
    ref_outputs = torch.load(source_dir / "reference_replay_final.pt", map_location="cpu", weights_only=False)
    expected_identities = [_sample_identity(item) for item in inputs["entries"]]
    _validate_identity_population(expected_identities, expected_identities)
    _require(len(expected_identities) == 200
             and commands.get("schema") == "lightgen_frozen_slm_ols_amplitude_commands.v1"
             and commands.get("sample_identities") == expected_identities
             and compact.get("sample_identities") == expected_identities
             and compact.get("schema") == "lightgen_frozen_slm_ols_compact_replay_evidence.v1"
             and compact.get("run_id") == identity["run_id"]
             and tuple(commands["amplitude_command"].shape) == (200, 1, 200, 200)
             and tuple(commands["intensity_command"].shape) == (200, 1, 200, 200)
             and tuple(compact["target_coefficients"].shape) == (200, 100),
             "upstream instruction, identity, or target evidence is incomplete")
    amplitude = commands["amplitude_command"].to(torch.float64)
    _require(torch.isfinite(amplitude).all().item() and float(amplitude.min()) >= 0.0
             and float(amplitude.max()) <= 1.0
             and torch.allclose(commands["intensity_command"].double(), amplitude.square(), rtol=0.0, atol=2e-7),
             "saved continuous command is nonfinite, out of range, or inconsistent")
    target = torch.stack([source_run._target_for(inputs, item).cpu().to(torch.complex128)
                          for item in inputs["entries"]])
    _require(torch.equal(compact["target_coefficients"].to(torch.complex128), target),
             "saved upstream targets differ from the fixed selection")
    _verify_source_witness(inputs, amplitude, target, compact)
    _verify_saved_reference(inputs, ref_outputs, ref_json)
    inputs.update({
        "source_dir": source_dir,
        "source_identity": identity,
        "source_final": final,
        "source_preflight": preflight,
        "source_commands": commands,
        "source_compact": compact,
        "source_reference_outputs": ref_outputs,
        "amplitude": amplitude,
        "targets": target,
        "sample_identities": expected_identities,
    })
    return inputs


def _verify_source_witness(
    inputs: dict[str, Any], amplitude: torch.Tensor, targets: torch.Tensor, compact: dict[str, Any],
) -> None:
    _require(all(grid in compact.get("grids", {}) for grid in ("2048", "4096")),
             "upstream output grids are incomplete")
    pass_count = 0
    for index, item in enumerate(inputs["entries"]):
        row_ok = True
        for grid in ("2048", "4096"):
            saved = compact["grids"][str(grid)]
            actual_coeff = saved["actual_coefficients"][index].to(torch.complex128)
            direct_coeff = saved["direct_coefficients"][index].to(torch.complex128)
            actual_raw = saved["actual_raw_cmos_intensity_w_per_m2"][index].double()
            direct_raw = saved["direct_raw_cmos_intensity_w_per_m2"][index].double()
            source_actual_coeff = saved["source_saved_actual_coefficients"][index].to(torch.complex128)
            source_direct_coeff = saved["source_saved_direct_coefficients"][index].to(torch.complex128)
            source_actual_raw = saved["source_saved_actual_raw_cmos_intensity_w_per_m2"][index].double()
            source_direct_raw = saved["source_saved_direct_raw_cmos_intensity_w_per_m2"][index].double()
            row_ok &= source_run._phase_error(actual_coeff, targets[index])[1] <= PRECISION_LIMIT
            row_ok &= source_run._rel_l2(actual_raw, direct_raw) <= PRECISION_LIMIT
            _require(source_run._rel_l2(direct_coeff, targets[index]) <= 1.0e-12,
                     f"upstream direct Decoder reference does not replay the fixed coefficient target: {source_run._item_key(item)}")
            _require(source_run._rel_l2(actual_coeff, source_actual_coeff) <= SOURCE_REPLAY_LIMIT
                     and source_run._rel_l2(direct_coeff, source_direct_coeff) <= SOURCE_REPLAY_LIMIT
                     and source_run._rel_l2(actual_raw, source_actual_raw) <= SOURCE_REPLAY_LIMIT
                     and source_run._rel_l2(direct_raw, source_direct_raw) <= SOURCE_REPLAY_LIMIT,
                     f"upstream production output replay changed: {source_run._item_key(item)} grid={grid}")
        cross_coeff = source_run._rel_l2(
            compact["grids"]["2048"]["actual_coefficients"][index].to(torch.complex128),
            compact["grids"]["4096"]["actual_coefficients"][index].to(torch.complex128),
        )
        cross_cmos = source_run._rel_l2(
            compact["grids"]["2048"]["actual_raw_cmos_intensity_w_per_m2"][index].double(),
            compact["grids"]["4096"]["actual_raw_cmos_intensity_w_per_m2"][index].double(),
        )
        row_ok &= cross_coeff <= PRECISION_LIMIT and cross_cmos <= PRECISION_LIMIT
        pass_count += int(row_ok)
    _require(pass_count == 200, "upstream source witness fails independent saved-output recomputation")


def _verify_saved_reference(
    inputs: dict[str, Any], outputs: dict[str, Any], reference: dict[str, Any],
) -> None:
    _require(outputs.get("schema") == "lightgen_frozen_slm_ols_reference_replay_outputs.v1"
             and outputs.get("task_id") == source_run.TASK_ID
             and set(outputs.get("methods", {})) == set(METHODS)
             and reference.get("rows") == 200,
             "upstream saved reference replay is malformed")
    total = 0
    for method in METHODS:
        records = outputs["methods"][method]
        items = [item for item in inputs["entries"] if item["method"] == method]
        _require(tuple(records["cohort_rows"].shape) == (100,)
                 and tuple(records["native200"].shape) == (100, 1, 200, 200)
                 and tuple(records["view28"].shape) == (100, 1, 28, 28)
                 and records["cohort_rows"].tolist() == [int(item["cohort_row"]) for item in items],
                 f"upstream saved view rows changed for {method}")
        for offset, item in enumerate(items):
            replay_index = inputs["replay_indices"][method][int(item["cohort_row"])]
            expected_native = inputs["replay"][method]["native200"][replay_index].float()
            expected_view = inputs["replay"][method]["view28"][replay_index].float()
            native = records["native200"][offset].float()
            view = records["view28"][offset].float()
            _require(torch.allclose(native, expected_native, rtol=2.0e-6, atol=5.0e-8)
                     and torch.allclose(view, expected_view, rtol=2.0e-6, atol=5.0e-8),
                     f"upstream historical saved view changed: {source_run._item_key(item)}")
            total += 1
    _require(total == 200, "upstream reference denominator changed")


def _build_encoded_bundle(amplitude: torch.Tensor, identities: list[dict[str, Any]], source_run_id: str) -> dict[str, Any]:
    amplitude = torch.as_tensor(amplitude, dtype=torch.float64, device="cpu")
    _require(tuple(amplitude.shape) == (len(identities), 1, *IMAGE_SHAPE)
             and torch.isfinite(amplitude).all().item()
             and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0,
             "continuous amplitude commands are invalid for gray encoding", ValueError)
    amplitude_codes = quantize_u8(amplitude)
    intensity_codes = quantize_u8(amplitude.square())
    decoded_amplitude = decode_u8(amplitude_codes)
    decoded_intensity = decode_u8(intensity_codes)
    return {
        "schema": "lightgen_frozen_slm_8bit_encoded_commands.v1",
        "task_id": TASK_ID,
        "source_run_id": source_run_id,
        "sample_identities": identities,
        "encoding": {
            "rounding": "floor(255*x+0.5)",
            "A8": "Q8(a) amplitude gray code; production intensity input is (gray/255)^2",
            "I8": "Q8(a^2) intensity gray code; production intensity input is gray/255",
            "decoder": "gray/255 in float64; frozen production adapter applies sqrt to intensity input",
        },
        "A8_gray_u8": amplitude_codes,
        "I8_gray_u8": intensity_codes,
        "A8_decoded_amplitude": decoded_amplitude.float(),
        "A8_production_intensity": decoded_amplitude.square().float(),
        "I8_decoded_intensity": decoded_intensity.float(),
        "I8_effective_amplitude_after_production_sqrt": decoded_intensity.sqrt().float(),
    }


def _validate_encoded_bundle(bundle: dict[str, Any], amplitude: torch.Tensor,
                             identities: list[dict[str, Any]], source_run_id: str) -> None:
    expected = _build_encoded_bundle(amplitude, identities, source_run_id)
    _require(bundle.get("schema") == expected["schema"] and bundle.get("task_id") == TASK_ID
             and bundle.get("source_run_id") == source_run_id
             and bundle.get("sample_identities") == identities
             and bundle.get("encoding") == expected["encoding"],
             "saved gray encoding identity or decoding rule changed")
    for field in ("A8_gray_u8", "I8_gray_u8", "A8_decoded_amplitude", "A8_production_intensity",
                  "I8_decoded_intensity", "I8_effective_amplitude_after_production_sqrt"):
        value = bundle.get(field)
        _require(isinstance(value, torch.Tensor) and torch.equal(value, expected[field]),
                 f"saved gray command or decoded input changed: {field}")


def _ensure_encoded_bundle(run_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "encoded_commands.pt"
    if not path.is_file():
        _atomic_torch(path, _build_encoded_bundle(
            inputs["amplitude"], inputs["sample_identities"], inputs["source_identity"]["run_id"]
        ))
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    _validate_encoded_bundle(bundle, inputs["amplitude"], inputs["sample_identities"],
                             inputs["source_identity"]["run_id"])
    return bundle


def _save_input_authority(run_dir: Path, config: dict[str, Any], inputs: dict[str, Any], run_id: str) -> None:
    value = {
        "schema": "lightgen_frozen_slm_8bit_input_authority.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "source_config": config["source"]["reachability_config"],
        "source_attempt_dir": config["source"]["reachability_attempt_dir"],
        "source_run_id": inputs["source_identity"]["run_id"],
        "checkpoint_step": 45_000,
        "camera_gain_w_inverse": float(inputs["checkpoint"]["camera_gain"]),
        "methods": list(METHODS),
        "sample_count": 200,
        "sample_identities": inputs["sample_identities"],
    }
    path = run_dir / "input_authority.json"
    config_snapshot = {
        "config_path": "configs/research/lightgen/frozen_slm_8bit_replay_v1.yaml",
        "config": config,
    }
    config_path = run_dir / "run_config_snapshot.json"
    if config_path.is_file():
        _require(_load_json(config_path) == config_snapshot, "saved run configuration changed")
    else:
        _atomic_json(config_path, config_snapshot)
    if path.is_file():
        _require(_load_json(path) == value, "saved 8-bit input authority changed")
    else:
        _atomic_json(path, value)


def _prepare_output(run_dir: Path, allow_closed: bool) -> dict[str, Any]:
    _require(run_dir.resolve() == EXPECTED_ATTEMPT.resolve(), "output path differs from the fixed attempt", ValueError)
    path = run_dir / "run_identity.json"
    if path.is_file():
        identity = _load_json(path)
        _require(identity.get("task_id") == TASK_ID
                 and identity.get("status") in {"ACTIVE", "POPULATION_COMPLETE_PENDING_WINDOWS_RECONSUMPTION", "CLOSED"}
                 and (allow_closed or identity.get("status") != "CLOSED"),
                 "output path belongs to another run or is already closed")
        return identity
    _require(not run_dir.exists() or not any(run_dir.iterdir()),
             "fixed output directory already contains files without this run identity")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "lightgen_frozen_slm_8bit_replay_run.v1",
        "task_id": TASK_ID,
        "run_id": uuid.uuid4().hex,
        "status": "ACTIVE",
        "source_run_id": "04390141af8c48278cfd3578f3688e0c",
        "checkpoint_step": 45_000,
        "methods": list(METHODS),
        "sample_count": 200,
        "conditions": list(CONDITIONS),
        "grids": list(GRIDS),
        "model_updates": 0,
        "adam_calls": 0,
        "prior_fit": 0,
        "new_sampling": 0,
        "new_human_review": 0,
        "input_resolve": 0,
    }
    _atomic_json(path, identity)
    return identity


def _output_path(run_dir: Path, condition: str, grid: int, sample_key: str) -> Path:
    return run_dir / "samples" / condition / str(grid) / f"{sample_key}.pt"


def _intensity_for(condition: str, index: int, amplitude: torch.Tensor, bundle: dict[str, Any],
                   device: torch.device) -> torch.Tensor:
    if condition == "continuous":
        intensity = amplitude[index, 0].to(device=device, dtype=torch.float64).square()
    elif condition == "A8_amplitude_linear":
        gray = bundle["A8_gray_u8"][index, 0]
        quantized_amplitude = decode_u8(gray).to(device=device)
        intensity = quantized_amplitude.square()
    elif condition == "I8_intensity_linear":
        gray = bundle["I8_gray_u8"][index, 0]
        intensity = decode_u8(gray).to(device=device)
    else:
        raise ValueError(f"unknown SLM encoding condition: {condition}")
    _require(tuple(intensity.shape) == IMAGE_SHAPE and torch.isfinite(intensity).all().item()
             and float(intensity.min()) >= 0.0 and float(intensity.max()) <= 1.0,
             f"decoded production intensity is invalid: {condition}")
    return intensity.reshape(1, 1, *IMAGE_SHAPE)


def _verify_saved_gray_decoding(amplitude: torch.Tensor, bundle: dict[str, Any]) -> None:
    for index in range(amplitude.shape[0]):
        a8_input = _intensity_for("A8_amplitude_linear", index, amplitude, bundle, torch.device("cpu"))[0, 0]
        i8_input = _intensity_for("I8_intensity_linear", index, amplitude, bundle, torch.device("cpu"))[0, 0]
        a8_saved = bundle["A8_production_intensity"][index, 0].double()
        i8_saved = bundle["I8_decoded_intensity"][index, 0].double()
        _require(torch.allclose(a8_input, a8_saved, rtol=0.0, atol=1.0e-7)
                 and torch.allclose(i8_input, i8_saved, rtol=0.0, atol=1.0e-7),
                 f"saved gray codes do not decode to the measured production input: row {index}")


def _runtime_forward(model: torch.nn.Module, intensity: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        result = model.runtime.forward_from_image(intensity)
    trace = result.physical_power_trace
    _require(trace is not None and result.physical_sensor_output_w_per_pixel is not None,
             "production image path did not return absolute power evidence")
    coefficients = result.z_ols.coefficients[0].detach().to(torch.complex128).cpu()
    raw = result.raw_camera.tensor[0, 0].detach().to(torch.float64).cpu()
    _require(tuple(coefficients.shape) == (100,) and tuple(raw.shape) == IMAGE_SHAPE
             and torch.isfinite(coefficients.real).all().item() and torch.isfinite(coefficients.imag).all().item()
             and torch.isfinite(raw).all().item() and float(raw.min()) >= 0.0,
             "production output is malformed or nonfinite")
    power_sources = {
        "P_source_before_SLM_W": "P_input_W",
        "P_after_SLM_W": "P_after_slm_W",
        "P_encoder_entrance_W": "P_encoder_entrance_W",
        "P_encoder_layer_1_after_aperture_W": "P_encoder_1_W",
        "P_encoder_layer_2_after_aperture_W": "P_encoder_2_W",
        "P_before_OLS_incident_W": "P_before_fiber_W",
        "P_after_OLS_reemitted_decoder_entry_W": "P_generator_entry_W",
        "P_decoder_layer_1_after_aperture_W": "P_decoder_1_W",
        "P_decoder_layer_2_after_aperture_W": "P_decoder_2_W",
        "P_CMOS_W": "P_camera_from_sensor_pixels_W",
    }
    powers: dict[str, float] = {}
    for result_name, trace_name in power_sources.items():
        value = trace.get(trace_name)
        _require(isinstance(value, torch.Tensor) and value.numel() >= 1 and torch.isfinite(value).all().item(),
                 f"production power trace is missing {trace_name}")
        powers[result_name] = float(value.reshape(-1)[0].detach().double().cpu())
    camera_total = float(result.physical_sensor_output_w_per_pixel[0, 0].detach().double().sum().cpu())
    pixel_area = float(model.runtime.camera_plane.pixel_pitch_m[0] * model.runtime.camera_plane.pixel_pitch_m[1])
    _require(math.isclose(powers["P_CMOS_W"], camera_total, rel_tol=1.0e-10, abs_tol=1.0e-15)
             and math.isclose(float(raw.sum()) * pixel_area, camera_total, rel_tol=1.0e-10, abs_tol=1.0e-15),
             "CMOS intensity and integrated absolute power disagree")
    return {
        "actual_coefficients": coefficients,
        "raw_cmos_intensity_w_per_m2": raw,
        "power_trace_w": powers,
        "camera_pixel_area_m2": pixel_area,
        "camera_total_power_w": camera_total,
    }


def _sample_file_identity(run_id: str, condition: str, grid: int, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lightgen_frozen_slm_8bit_sample_output.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "condition": condition,
        "grid": grid,
        "sample_identity": identity,
    }


def _validate_sample_output(output: dict[str, Any], expected: dict[str, Any], run_id: str,
                            condition: str, grid: int) -> None:
    _require(isinstance(output, dict) and output.get("schema") == "lightgen_frozen_slm_8bit_sample_output.v1"
             and output.get("task_id") == TASK_ID and output.get("run_id") == run_id
             and output.get("condition") == condition and output.get("grid") == grid
             and output.get("sample_identity") == expected,
             f"saved sample output identity changed: {condition} {grid} {expected['sample_key']}")
    coefficients = output.get("actual_coefficients")
    raw = output.get("raw_cmos_intensity_w_per_m2")
    _require(isinstance(coefficients, torch.Tensor) and coefficients.dtype == torch.complex128
             and tuple(coefficients.shape) == (100,)
             and isinstance(raw, torch.Tensor) and tuple(raw.shape) == IMAGE_SHAPE
             and torch.isfinite(coefficients.real).all().item() and torch.isfinite(coefficients.imag).all().item()
             and torch.isfinite(raw).all().item() and float(raw.min()) >= 0.0,
             f"saved production output is malformed: {condition} {grid} {expected['sample_key']}")
    powers = output.get("power_trace_w")
    _require(isinstance(powers, dict) and set(powers) == POWER_TRACE_FIELDS
             and all(math.isfinite(float(value)) and float(value) >= 0.0 for value in powers.values()),
             f"saved absolute-power table is malformed: {condition} {grid} {expected['sample_key']}")


def _preflight_items(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for item in inputs["entries"]:
        key = (item["method"], int(item["requested_label"]))
        chosen.setdefault(key, item)
    _require(len(chosen) == 20 and all((method, label) in chosen for method in METHODS for label in range(10)),
             "fixed preflight does not contain one row per method/class")
    return [chosen[(method, label)] for method in METHODS for label in range(10)]


def _run_sample_set(run_dir: Path, run_id: str, items: list[dict[str, Any]], inputs: dict[str, Any],
                    bundle: dict[str, Any], models: dict[int, Any]) -> None:
    total = len(CONDITIONS) * len(GRIDS) * len(items)
    completed = 0
    identity_indices = {identity["sample_key"]: index
                        for index, identity in enumerate(inputs["sample_identities"])}
    for condition in CONDITIONS:
        for grid in GRIDS:
            model = models[grid]
            for item in items:
                identity = _sample_identity(item)
                path = _output_path(run_dir, condition, grid, identity["sample_key"])
                if path.is_file():
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    _validate_sample_output(saved, identity, run_id, condition, grid)
                else:
                    index = identity_indices[identity["sample_key"]]
                    intensity = _intensity_for(condition, index, inputs["amplitude"], bundle,
                                               next(model.parameters()).device)
                    result = _runtime_forward(model, intensity)
                    saved = {**_sample_file_identity(run_id, condition, grid, identity), **result}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_torch(path, saved)
                completed += 1
                if completed % 20 == 0 or completed == total:
                    _atomic_json(run_dir / "progress.json", {
                        "task_id": TASK_ID, "run_id": run_id, "phase": "frozen_8bit_full_path",
                        "completed_this_stage": completed, "stage_total": total,
                        "condition": condition, "grid": grid,
                        "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
                        "new_sampling": 0, "new_human_review": 0,
                    })


def _load_all_outputs(run_dir: Path, run_id: str, inputs: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    expected_keys = [identity["sample_key"] for identity in inputs["sample_identities"]]
    _require(len(expected_keys) == 200 and len(set(expected_keys)) == 200,
             "fixed population contains missing or duplicate sample identities")
    identity_map = {identity["sample_key"]: identity for identity in inputs["sample_identities"]}
    for condition in CONDITIONS:
        for grid in GRIDS:
            observed_identities: list[dict[str, Any]] = []
            for sample_key in expected_keys:
                path = _output_path(run_dir, condition, grid, sample_key)
                _require(path.is_file(), f"saved production output is missing: {condition}/{grid}/{sample_key}",
                         FileNotFoundError)
                output = torch.load(path, map_location="cpu", weights_only=False)
                _validate_sample_output(output, identity_map[sample_key], run_id, condition, grid)
                key = (condition, grid, sample_key)
                _require(key not in result, f"duplicate saved output identity: {key}")
                result[key] = output
                observed_identities.append(output["sample_identity"])
            _validate_identity_population(inputs["sample_identities"], observed_identities)
    _require(len(result) == len(CONDITIONS) * len(GRIDS) * 200,
             "saved output population denominator changed")
    return result


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected.reshape(-1).double())
    _require(torch.isfinite(denominator).item() and denominator.item() > 0.0,
             "relative-error reference is zero or nonfinite")
    return float(torch.linalg.vector_norm((actual - expected).reshape(-1).double()) / denominator)


def _metric_records(inputs: dict[str, Any], outputs: dict[tuple[str, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    compact = inputs["source_compact"]["grids"]
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for index, item in enumerate(inputs["entries"]):
            identity = inputs["sample_identities"][index]
            sample_key = identity["sample_key"]
            target = inputs["targets"][index]
            metrics: dict[str, float] = {}
            phase_values: dict[str, float] = {}
            for grid in GRIDS:
                output = outputs[(condition, grid, sample_key)]
                direct = compact[str(grid)]
                actual_coeff = output["actual_coefficients"].to(torch.complex128)
                actual_raw = output["raw_cmos_intensity_w_per_m2"].double()
                direct_raw = direct["direct_raw_cmos_intensity_w_per_m2"][index, 0].double()
                reference_phase, target_coeff_error = source_run._phase_error(actual_coeff, target)
                increment_phase, continuous_coeff_error = source_run._phase_error(
                    actual_coeff, outputs[("continuous", grid, sample_key)]["actual_coefficients"].to(torch.complex128)
                )
                metrics[f"target_coefficient_relative_error_{grid}"] = target_coeff_error
                metrics[f"direct_cmos_intensity_relative_error_{grid}"] = _relative(actual_raw, direct_raw)
                metrics[f"continuous_increment_coefficient_relative_error_{grid}"] = continuous_coeff_error
                metrics[f"continuous_increment_cmos_intensity_relative_error_{grid}"] = _relative(
                    actual_raw, outputs[("continuous", grid, sample_key)]["raw_cmos_intensity_w_per_m2"].double()
                )
                metrics[f"source_continuous_coefficient_replay_relative_error_{grid}"] = source_run._rel_l2(
                    outputs[("continuous", grid, sample_key)]["actual_coefficients"].to(torch.complex128),
                    compact[str(grid)]["actual_coefficients"][index].to(torch.complex128),
                )
                metrics[f"source_continuous_cmos_replay_relative_error_{grid}"] = source_run._rel_l2(
                    outputs[("continuous", grid, sample_key)]["raw_cmos_intensity_w_per_m2"].double(),
                    compact[str(grid)]["actual_raw_cmos_intensity_w_per_m2"][index, 0].double(),
                )
                metrics[f"camera_power_W_{grid}"] = float(output["camera_total_power_w"])
                metrics[f"camera_pixel_area_m2_{grid}"] = float(output["camera_pixel_area_m2"])
                for power_name, power_value in output["power_trace_w"].items():
                    metrics[f"{power_name}_{grid}"] = float(power_value)
                phase_values[f"target_phase_rad_{grid}"] = reference_phase
                phase_values[f"continuous_increment_phase_rad_{grid}"] = increment_phase
            metrics["cross_grid_coefficient_relative_error"] = source_run._rel_l2(
                outputs[(condition, 2048, sample_key)]["actual_coefficients"].to(torch.complex128),
                outputs[(condition, 4096, sample_key)]["actual_coefficients"].to(torch.complex128),
            )
            metrics["cross_grid_cmos_intensity_relative_error"] = source_run._rel_l2(
                outputs[(condition, 2048, sample_key)]["raw_cmos_intensity_w_per_m2"].double(),
                outputs[(condition, 4096, sample_key)]["raw_cmos_intensity_w_per_m2"].double(),
            )
            worst_target = max(metrics[f"target_coefficient_relative_error_{grid}"] for grid in GRIDS)
            worst_direct = max(metrics[f"direct_cmos_intensity_relative_error_{grid}"] for grid in GRIDS)
            source_replay_ok = condition != "continuous" or all(
                metrics[f"source_continuous_{kind}_replay_relative_error_{grid}"] <= SOURCE_REPLAY_LIMIT
                for kind in ("coefficient", "cmos") for grid in GRIDS
            )
            witness = bool(
                worst_target <= PRECISION_LIMIT and worst_direct <= PRECISION_LIMIT
                and metrics["cross_grid_coefficient_relative_error"] <= PRECISION_LIMIT
                and metrics["cross_grid_cmos_intensity_relative_error"] <= PRECISION_LIMIT
                and source_replay_ok
            )
            records.append({
                **identity, "condition": condition, **metrics, **phase_values,
                "worst_target_coefficient_relative_error": worst_target,
                "worst_direct_cmos_intensity_relative_error": worst_direct,
                "witness": witness,
            })
    _require(len(records) == 600, "metric consumer did not preserve the 600 condition-sample records")
    return records


def _metrics_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "lightgen_frozen_slm_8bit_producer_metrics.v1",
        "task_id": TASK_ID,
        "metric_count": len(records),
        "metrics": [{key: value for key, value in record.items() if key not in {"witness"}}
                    for record in records],
    }


def _assert_metrics_match(recomputed: dict[str, Any], producer: dict[str, Any]) -> None:
    _require(set(recomputed) == set(producer), "producer metric fields differ from consumer fields")
    for key, value in recomputed.items():
        saved = producer[key]
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            _require(isinstance(saved, (float, int))
                     and abs(float(saved) - float(value)) <= max(1e-12, abs(float(value)) * 1e-9),
                     f"saved production output differs from its independently recomputed metric: {key}")
        else:
            _require(saved == value, f"saved production metric identity changed: {key}")


def _check_producer_metrics(path: Path, records: list[dict[str, Any]], run_id: str) -> None:
    payload = _load_json(path)
    _require(payload.get("schema") == "lightgen_frozen_slm_8bit_producer_metrics.v1"
             and payload.get("task_id") == TASK_ID and payload.get("run_id") == run_id
             and payload.get("metric_count") == 600
             and isinstance(payload.get("metrics"), list) and len(payload["metrics"]) == 600,
             "saved producer metrics are incomplete or belong to another run")
    indexed = {(row.get("condition"), row.get("sample_key")): row for row in payload["metrics"]}
    _require(len(indexed) == 600, "saved producer metric identities are duplicated")
    for record in records:
        key = (record["condition"], record["sample_key"])
        _require(key in indexed, f"saved producer metric row is missing: {key}")
        _assert_metrics_match({k: v for k, v in record.items() if k != "witness"}, indexed[key])


def _preflight_controls(models: dict[int, Any], source_power: float) -> dict[str, Any]:
    controls: dict[str, Any] = {"grid_controls": {}, "power_endpoints": {}}
    codes = (torch.arange(PIXEL_COUNT, dtype=torch.int64).reshape(1, 1, *IMAGE_SHAPE) % 256).to(torch.uint8)
    a8_amplitude = decode_u8(codes)
    a8_intensity_before_storage = a8_amplitude.square()
    a8_intensity_after_storage = decode_u8(quantize_u8(a8_amplitude)).square()
    i8_intensity = decode_u8(codes)
    _require(torch.equal(quantize_u8(a8_amplitude), codes) and torch.equal(quantize_u8(i8_intensity), codes),
             "known gray-grid positive control did not round-trip")
    _require(torch.equal(a8_intensity_before_storage, a8_intensity_after_storage),
             "A8 pre/post-storage production intensity changed")
    for grid, model in models.items():
        cases = (
            ("A8", a8_intensity_before_storage, a8_intensity_after_storage),
            ("I8", i8_intensity, decode_u8(quantize_u8(i8_intensity))),
        )
        for name, before_storage, after_storage in cases:
            decoded = _runtime_forward(model, before_storage.to(next(model.parameters()).device))
            direct = _runtime_forward(model, after_storage.to(next(model.parameters()).device))
            coeff_error = source_run._rel_l2(decoded["actual_coefficients"], direct["actual_coefficients"])
            cmos_error = source_run._rel_l2(decoded["raw_cmos_intensity_w_per_m2"],
                                            direct["raw_cmos_intensity_w_per_m2"])
            _require(coeff_error <= 1.0e-12 and cmos_error <= 1.0e-12,
                     f"saved-gray production positive control changed output: {name}, grid={grid}")
            controls["grid_controls"][f"{name}_{grid}"] = {
                "coefficient_relative_error": coeff_error, "raw_cmos_relative_error": cmos_error,
                "gray_round_trip": True,
            }
        endpoint_results: dict[str, Any] = {}
        for name, level, expected_after in (("full_off", 0.0, 0.0), ("full_on", 1.0, source_power)):
            intensity = torch.full((1, 1, *IMAGE_SHAPE), level, dtype=torch.float64,
                                   device=next(model.parameters()).device)
            with torch.no_grad():
                result = model.runtime.forward_from_image(intensity)
            trace = result.physical_power_trace
            _require(trace is not None, "power endpoint test did not return physical trace")
            p_input = float(trace["P_input_W"].reshape(-1)[0].detach().double().cpu())
            p_after_slm = float(trace["P_after_slm_W"].reshape(-1)[0].detach().double().cpu())
            _require(math.isclose(p_input, source_power, rel_tol=0.0, abs_tol=1e-12)
                     and math.isclose(p_after_slm, expected_after, rel_tol=0.0, abs_tol=1e-12),
                     f"input-power meaning changed for {name}, grid={grid}")
            endpoint_results[name] = {"P_source_before_SLM_W": p_input, "P_after_SLM_W": p_after_slm}
        controls["power_endpoints"][str(grid)] = endpoint_results
    return controls


def _make_preflight(run_dir: Path, identity: dict[str, Any], inputs: dict[str, Any], config: dict[str, Any],
                    bundle: dict[str, Any], models: dict[int, Any], state_before: dict[str, torch.Tensor]) -> dict[str, Any]:
    reference_model = source_run._new_reference_model(inputs["checkpoint"], torch.device("cuda"))
    historical = source_run._reference_replay_all(
        reference_model, inputs, float(inputs["checkpoint"]["camera_gain"]), run_dir,
        result_name="historical_reference_replay.json", run_id=identity["run_id"],
    )
    _require(historical["rows"] == 200 and historical["all_passed"],
             "original saved native/view reference did not replay; stop before 8-bit results")
    controls = _preflight_controls(models, float(config["optics"]["source_power_w"]))
    preflight_items = _preflight_items(inputs)
    _run_sample_set(run_dir, identity["run_id"], preflight_items, inputs, bundle, models)
    output_map = _load_preflight_outputs(run_dir, identity["run_id"], preflight_items)
    limited = _metric_records_for_items(inputs, output_map, preflight_items)
    cross_grid_max = {
        condition: {
            "coefficient": max(row["cross_grid_coefficient_relative_error"] for row in limited
                               if row["condition"] == condition),
            "raw_cmos_intensity": max(row["cross_grid_cmos_intensity_relative_error"] for row in limited
                                       if row["condition"] == condition),
        }
        for condition in CONDITIONS
    }
    for condition, values in cross_grid_max.items():
        _require(values["coefficient"] <= PRECISION_LIMIT and values["raw_cmos_intensity"] <= PRECISION_LIMIT,
                 f"preflight numerical window failed for {condition}; stop without expanding the population")
    source_replay_max = max(
        value for row in limited if row["condition"] == "continuous"
        for key, value in row.items() if key.startswith("source_continuous_")
    )
    _require(source_replay_max <= SOURCE_REPLAY_LIMIT,
             "continuous 20-row production replay differs from the saved upstream witness")
    state_unchanged = True
    for grid, model in models.items():
        source_run._assert_state_unchanged(model, state_before[str(grid)])
    _require(state_unchanged, "frozen optical model state changed in preflight")
    result = {
        "schema": "lightgen_frozen_slm_8bit_preflight.v1",
        "task_id": TASK_ID,
        "run_id": identity["run_id"],
        "status": "PREFLIGHT_PASSED",
        "preflight_population": {
            "rows": len(preflight_items), "G1": 10, "G4": 10, "per_class_per_method": 1,
            "conditions": list(CONDITIONS), "grids": list(GRIDS),
        },
        "reference_replay": {
            "rows": historical["rows"], "all_passed": historical["all_passed"],
            "max_relative_error": historical["max_relative_error"],
            "max_view28_relative_error": historical["max_view28_relative_error"],
            "rtol": 2.0e-6, "atol": 5.0e-8,
        },
        "positive_controls": controls,
        "cross_grid": cross_grid_max,
        "continuous_source_output_replay_max": source_replay_max,
        "frozen_model_state_unchanged": state_unchanged,
        "model_updates": 0, "optical_parameter_updates": 0, "adam_calls": 0,
        "prior_fit": 0, "new_sampling": 0, "new_human_review": 0, "input_resolve": 0,
    }
    _atomic_json(run_dir / "preflight.json", result)
    _atomic_json(run_dir / "preflight_execution.json", {
        "status": result["status"], "run_id": identity["run_id"],
        "device": "cuda", "sample_rows": len(preflight_items),
        "condition_grid_outputs": len(preflight_items) * len(CONDITIONS) * len(GRIDS),
        "model_updates": 0, "adam_calls": 0,
    })
    return result


def _load_preflight_outputs(run_dir: Path, run_id: str, items: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for condition in CONDITIONS:
        for grid in GRIDS:
            for item in items:
                identity = _sample_identity(item)
                path = _output_path(run_dir, condition, grid, identity["sample_key"])
                _require(path.is_file(), f"preflight sample output missing: {path.name}", FileNotFoundError)
                output = torch.load(path, map_location="cpu", weights_only=False)
                _validate_sample_output(output, identity, run_id, condition, grid)
                result[(condition, grid, identity["sample_key"])] = output
    return result


def _metric_records_for_items(inputs: dict[str, Any], outputs: dict[tuple[str, int, str], dict[str, Any]],
                              items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Preflight rows use the same complete-population metric consumer with its denominator restricted to 20.
    compact = inputs["source_compact"]["grids"]
    by_key = {identity["sample_key"]: index for index, identity in enumerate(inputs["sample_identities"])}
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for item in items:
            index = by_key[source_run._item_key(item)]
            identity = inputs["sample_identities"][index]
            target = inputs["targets"][index]
            sample_key = identity["sample_key"]
            metrics: dict[str, float] = {}
            for grid in GRIDS:
                current = outputs[(condition, grid, sample_key)]
                continuous = outputs[("continuous", grid, sample_key)]
                base = compact[str(grid)]
                metrics[f"target_coefficient_relative_error_{grid}"] = source_run._phase_error(
                    current["actual_coefficients"].to(torch.complex128), target
                )[1]
                metrics[f"direct_cmos_intensity_relative_error_{grid}"] = _relative(
                    current["raw_cmos_intensity_w_per_m2"].double(),
                    base["direct_raw_cmos_intensity_w_per_m2"][index, 0].double(),
                )
                metrics[f"continuous_increment_coefficient_relative_error_{grid}"] = source_run._phase_error(
                    current["actual_coefficients"].to(torch.complex128),
                    continuous["actual_coefficients"].to(torch.complex128),
                )[1]
                metrics[f"continuous_increment_cmos_intensity_relative_error_{grid}"] = _relative(
                    current["raw_cmos_intensity_w_per_m2"].double(),
                    continuous["raw_cmos_intensity_w_per_m2"].double(),
                )
                if condition == "continuous":
                    metrics[f"source_continuous_coefficient_replay_relative_error_{grid}"] = source_run._rel_l2(
                        continuous["actual_coefficients"].to(torch.complex128),
                        base["actual_coefficients"][index].to(torch.complex128),
                    )
                    metrics[f"source_continuous_cmos_replay_relative_error_{grid}"] = source_run._rel_l2(
                        continuous["raw_cmos_intensity_w_per_m2"].double(),
                        base["actual_raw_cmos_intensity_w_per_m2"][index, 0].double(),
                    )
            metrics["cross_grid_coefficient_relative_error"] = source_run._rel_l2(
                outputs[(condition, 2048, sample_key)]["actual_coefficients"].to(torch.complex128),
                outputs[(condition, 4096, sample_key)]["actual_coefficients"].to(torch.complex128),
            )
            metrics["cross_grid_cmos_intensity_relative_error"] = source_run._rel_l2(
                outputs[(condition, 2048, sample_key)]["raw_cmos_intensity_w_per_m2"].double(),
                outputs[(condition, 4096, sample_key)]["raw_cmos_intensity_w_per_m2"].double(),
            )
            records.append({**identity, "condition": condition, **metrics})
    return records


def _execute_full(run_dir: Path, identity: dict[str, Any], inputs: dict[str, Any],
                  bundle: dict[str, Any], models: dict[int, Any]) -> None:
    preflight = _load_json(run_dir / "preflight.json")
    _require(preflight.get("task_id") == TASK_ID and preflight.get("run_id") == identity["run_id"]
             and preflight.get("status") == "PREFLIGHT_PASSED"
             and preflight.get("reference_replay", {}).get("rows") == 200
             and preflight.get("reference_replay", {}).get("all_passed") is True,
             "full population is blocked by a missing or failed same-run preflight")
    state_before = {str(grid): source_run._snapshot_state(model) for grid, model in models.items()}
    _run_sample_set(run_dir, identity["run_id"], inputs["entries"], inputs, bundle, models)
    outputs = _load_all_outputs(run_dir, identity["run_id"], inputs)
    records = _metric_records(inputs, outputs)
    _atomic_json(run_dir / "producer_metrics.json", {
        **_metrics_payload(records), "run_id": identity["run_id"],
        "metric_definition": "numeric errors recomputed from saved coefficient, CMOS and power outputs",
    })
    for grid, model in models.items():
        source_run._assert_state_unchanged(model, state_before[str(grid)])
    identity["status"] = "POPULATION_COMPLETE_PENDING_WINDOWS_RECONSUMPTION"
    _atomic_json(run_dir / "run_identity.json", identity)
    _atomic_json(run_dir / "measurement_completion.json", {
        "schema": "lightgen_frozen_slm_8bit_measurement_completion.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "sample_denominator": 200, "condition_grid_outputs": 1200,
        "frozen_model_state_unchanged": True,
        "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
        "new_sampling": 0, "new_human_review": 0,
    })


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in records if row["condition"] == condition]
        by_method: dict[str, Any] = {}
        for method in METHODS:
            method_rows = [row for row in condition_rows if row["method"] == method]
            by_class = {}
            for label in range(10):
                class_rows = [row for row in method_rows if row["label"] == label]
                _require(len(class_rows) == 10, f"fixed {condition}/{method}/class-{label} denominator changed")
                by_class[str(label)] = {
                    "denominator": 10,
                    "passed": sum(bool(row["witness"]) for row in class_rows),
                    **_stats(class_rows),
                }
            _require(len(method_rows) == 100, f"fixed {condition}/{method} denominator changed")
            by_method[method] = {"denominator": 100, "passed": sum(bool(row["witness"]) for row in method_rows),
                                 **_stats(method_rows), "by_class": by_class}
        result[condition] = {
            "denominator": len(condition_rows), "passed": sum(bool(row["witness"]) for row in condition_rows),
            "by_method": by_method,
        }
    return result


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [float(row["worst_target_coefficient_relative_error"]) for row in rows]
    cmos = [float(row["worst_direct_cmos_intensity_relative_error"]) for row in rows]
    return {
        "target_coefficient_error_median": float(np.median(target)),
        "target_coefficient_error_p95": float(np.percentile(target, 95)),
        "target_coefficient_error_max": max(target),
        "direct_cmos_error_median": float(np.median(cmos)),
        "direct_cmos_error_p95": float(np.percentile(cmos, 95)),
        "direct_cmos_error_max": max(cmos),
    }


def _decision(summary: dict[str, Any]) -> tuple[str, str]:
    a8 = summary["A8_amplitude_linear"]["passed"]
    i8 = summary["I8_intensity_linear"]["passed"]
    if a8 == 200 and i8 == 200:
        return "BOTH_IDEAL_CALIBRATION_HYPOTHESES_PRESERVE_200_OF_200", (
            "两个理想校准假设下均为 200/200：本批连续振幅指令的直接 8 位编码保留指定数值复现精度。"
        )
    if a8 == 200 or i8 == 200:
        return "RESULT_DEPENDS_ON_IDEAL_CALIBRATION_HYPOTHESIS", (
            f"结果依赖编码假设：A8 为 {a8}/200，I8 为 {i8}/200；未通过样本表示原指令直接编码未保留指定精度。"
        )
    return "PARTIAL_OR_NO_8BIT_NUMERICAL_PRECISION_WITNESSES", (
        f"指定精度通过数为 A8 {a8}/200、I8 {i8}/200；未通过样本不能据此宣称 8 位器件无法制备目标。"
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    preferred = ["condition", "method", "label", "sample_key", "cohort_row", "proposal_id", "component_id",
                 "witness", "worst_target_coefficient_relative_error", "worst_direct_cmos_intensity_relative_error",
                 "cross_grid_coefficient_relative_error", "cross_grid_cmos_intensity_relative_error"]
    fields = preferred + sorted({key for record in records for key in record} - set(preferred))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_atlas(path: Path, inputs: dict[str, Any], outputs: dict[tuple[str, int, str], dict[str, Any]]) -> None:
    cell = 104
    label_height = 22
    margin = 6
    for method in METHODS:
        items = [item for item in inputs["entries"] if item["method"] == method]
        image = Image.new("RGB", (3 * cell + 2 * margin, len(items) * (cell + label_height) + margin), "white")
        draw = ImageDraw.Draw(image)
        for row, item in enumerate(items):
            identity = _sample_identity(item)
            sample_key = identity["sample_key"]
            top = margin + row * (cell + label_height)
            draw.text((margin, top), f"{method} class={identity['label']} row={identity['cohort_row']}", fill="black")
            for column, condition in enumerate(CONDITIONS):
                raw = outputs[(condition, 2048, sample_key)]["raw_cmos_intensity_w_per_m2"].double()
                area = float(outputs[(condition, 2048, sample_key)]["camera_pixel_area_m2"])
                gain = float(inputs["checkpoint"]["camera_gain"])
                display = (raw * area * gain).clamp(0.0, 1.0)
                pixels = display.mul(255).round().to(torch.uint8).numpy()
                tile = Image.fromarray(pixels).resize((cell - 4, cell - label_height - 2))
                x = margin + column * cell
                image.paste(tile.convert("RGB"), (x, top + label_height))
                draw.text((x + 2, top + 1), condition.replace("_", " "), fill="black")
        image.save(path.with_name(f"{path.stem}_{method}{path.suffix}"))


def _write_report(path: Path, decision: str, decision_text: str, summary: dict[str, Any],
                  preflight: dict[str, Any], source_identity: dict[str, Any]) -> None:
    lines = [
        "# 冻结 SLM 8 位灰阶回放报告",
        "",
        "## 1．结果与状态",
        "",
        f"最终判断：{decision_text}",
        f"A8 通过数：{summary['A8_amplitude_linear']['passed']}/200；I8 通过数：{summary['I8_intensity_linear']['passed']}/200。连续参考为 {summary['continuous']['passed']}/200。",
        "通过标准：2048/4096 两网格的目标公共相位系数误差、直接注入原生 CMOS 强度误差，以及跨网格误差均不超过 1e-4。该数值标准不是硬件容差或语义质量门槛。",
        "",
        "## 2．科学认识",
        "",
        "A8 将 8 位码解释为线性振幅，输入生产光路的强度是 Q8(a)^2；I8 将 8 位码解释为线性强度，输入是 Q8(a^2)。两种理想校准假设分别报告，不根据较优结果推断器件真实响应。",
        "论文补充材料写明 HES6001、1920×1080、60 Hz 与 532 nm；HOLOEYE 当前 HES 7020-1 6001 同系列产品页列出 8 位（256 灰阶）寻址、8 μm 像素及 60 Hz。论文未给出本器件灰阶响应曲线，因此这里是器件参数支持下的编码情景诊断。分辨率、像素间距和刷新率仅记录为器件依据，没有替换项目 200×200 采样几何，也没有据此宣称整屏并行显示或实际帧率。来源：[论文补充材料](../../../../research_assets/lightgen/science_adv7434/paper/sciencesm.pdf)；[HOLOEYE 产品页](https://holoeye.com/products/spatial-light-modulators/amplitude-slms/)。",
        "本次只量化输入振幅；Encoder/Decoder 相位保持冻结，不做 8 级相位量化。每次测量仍经过完整生产路径，绝对功率、全局增益和曝光均未重标定。",
        "",
        "## 3．目标缺口",
        "",
        "当前证据没有 HES6001 实物的灰阶 LUT、逐像素振幅/相位响应、有限消光、反射损耗或稳定时间标定；A8 与 I8 均不能被当作器件实测模型。项目几何仍为 PROJECT_GEOMETRY_UNMEASURED，历史 VALID_FAILED 保持。",
        "",
        "## 4．工程完成与验证",
        "",
        f"固定上轮 run_id：{source_identity['run_id']}；累计 C 45,000 步；G1/G4 各 100 条，每类各 10 条。灰阶、原系数、目标输出、源参考和样本身份均保存在本 attempt。",
        f"预检：{preflight['status']}；原 200 条历史显示视图复放通过={preflight['reference_replay']['all_passed']}；模型更新、Adam、先验拟合、新采样、新人工评阅和重新求解均为 0。",
        "功率表分别记录 OLS 前入射功率 P_before_OLS_incident_W 和 OLS 再发射后的 Decoder 入口 P_after_OLS_reemitted_decoder_entry_W；CMOS 误差使用增益与截断前的原生 W/m² 强度。Windows 消费者从 uint8 灰阶解码输入，并由保存输出独立重算误差与分母。",
        "",
        "## 5．决策影响与下一步",
        "",
        "无论通过、部分通过或未通过，本次到此停止。需要确定实物响应时，下一项证据应是同一 HES6001 器件在 532 nm 下的灰阶—复振幅/强度 LUT 与重复性标定；只有基于实测曲线，才能设计离散指令优化或量化感知训练。此建议不构成本轮后续训练、优化或部署授权。",
        "",
        "分组摘要（目标系数/原生 CMOS 的中位数、P95、最大值）：",
        "",
        "| 条件 | 方法 | 通过数/分母 | 系数中位数 / P95 / 最大 | CMOS 中位数 / P95 / 最大 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for method in METHODS:
            row = summary[condition]["by_method"][method]
            lines.append(
                f"| {condition} | {method} | {row['passed']}/100 | "
                f"{row['target_coefficient_error_median']:.3g} / {row['target_coefficient_error_p95']:.3g} / {row['target_coefficient_error_max']:.3g} | "
                f"{row['direct_cmos_error_median']:.3g} / {row['direct_cmos_error_p95']:.3g} / {row['direct_cmos_error_max']:.3g} |"
            )
    lines.extend([
        "",
        "按类别的通过数与残差分布：",
        "",
        "| 条件 | 方法 | 类别 | 通过数/分母 | 系数中位数 / P95 / 最大 | CMOS 中位数 / P95 / 最大 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        for method in METHODS:
            for label in range(10):
                row = summary[condition]["by_method"][method]["by_class"][str(label)]
                lines.append(
                    f"| {condition} | {method} | {label} | {row['passed']}/10 | "
                    f"{row['target_coefficient_error_median']:.3g} / {row['target_coefficient_error_p95']:.3g} / {row['target_coefficient_error_max']:.3g} | "
                    f"{row['direct_cmos_error_median']:.3g} / {row['direct_cmos_error_p95']:.3g} / {row['direct_cmos_error_max']:.3g} |"
                )
    lines.extend([
        "",
        "逐图功率及误差见 per_image_power_and_error.csv；全部 200 条样本的 2048 网格三条件对照分别见 comparison_atlas_G1.png 与 comparison_atlas_G4.png。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final_consumer(run_dir: Path, identity: dict[str, Any], inputs: dict[str, Any],
                    bundle: dict[str, Any]) -> dict[str, Any]:
    preflight = _load_json(run_dir / "preflight.json")
    completion = _load_json(run_dir / "measurement_completion.json")
    _require(preflight.get("run_id") == identity["run_id"] and preflight.get("status") == "PREFLIGHT_PASSED"
             and completion.get("run_id") == identity["run_id"]
             and completion.get("condition_grid_outputs") == 1200,
             "saved engineering preflight or population completion is missing")
    outputs = _load_all_outputs(run_dir, identity["run_id"], inputs)
    records = _metric_records(inputs, outputs)
    _check_producer_metrics(run_dir / "producer_metrics.json", records, identity["run_id"])
    for record in records:
        _require(record["cross_grid_coefficient_relative_error"] <= PRECISION_LIMIT
                 and record["cross_grid_cmos_intensity_relative_error"] <= PRECISION_LIMIT,
                 f"cross-grid numerical window failed; no quantization decision is valid: {record['sample_key']} {record['condition']}")
        if record["condition"] == "continuous":
            _require(all(record[f"source_continuous_{kind}_replay_relative_error_{grid}"] <= SOURCE_REPLAY_LIMIT
                         for kind in ("coefficient", "cmos") for grid in GRIDS),
                     f"continuous upstream replay failed; no quantization decision is valid: {record['sample_key']}")
    # Validate gray codes again at the final consumer boundary, then restore the encoded values.
    _validate_encoded_bundle(bundle, inputs["amplitude"], inputs["sample_identities"],
                             inputs["source_identity"]["run_id"])
    _verify_saved_gray_decoding(inputs["amplitude"], bundle)
    historical_outputs = torch.load(run_dir / "historical_reference_replay.pt",
                                    map_location="cpu", weights_only=False)
    _require(historical_outputs.get("run_id") == identity["run_id"],
             "historical reference replay belongs to another 8-bit attempt")
    _verify_saved_reference(inputs, historical_outputs, _load_json(run_dir / "historical_reference_replay.json"))
    summary = _summary(records)
    decision, decision_text = _decision(summary)
    coefficients = {
        "schema": "lightgen_frozen_slm_8bit_target_actual_coefficients.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "sample_identities": inputs["sample_identities"],
        "target_coefficients": inputs["targets"].to(torch.complex128),
        "actual_coefficients": {
            condition: {
                str(grid): torch.stack([
                    outputs[(condition, grid, row["sample_key"])]["actual_coefficients"]
                    for row in inputs["sample_identities"]
                ])
                for grid in GRIDS
            }
            for condition in CONDITIONS
        },
    }
    reference_outputs = {
        "schema": "lightgen_frozen_slm_8bit_direct_reference_outputs.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "sample_identities": inputs["sample_identities"],
        "direct_coefficients": {
            str(grid): inputs["source_compact"]["grids"][str(grid)]["direct_coefficients"].to(torch.complex128)
            for grid in GRIDS
        },
        "direct_raw_cmos_intensity_w_per_m2": {
            str(grid): inputs["source_compact"]["grids"][str(grid)][
                "direct_raw_cmos_intensity_w_per_m2"
            ].to(torch.float64)
            for grid in GRIDS
        },
    }
    _atomic_torch(run_dir / "target_actual_coefficients.pt", coefficients)
    _atomic_torch(run_dir / "reference_coefficients_and_cmos.pt", reference_outputs)
    _atomic_json(run_dir / "power_table_definition.json", {
        "schema": "lightgen_frozen_slm_8bit_power_table_definition.v1",
        "unit": "W",
        "source_power": "P_source_before_SLM_W is the calibrated 249.3 uW inlet before the amplitude command",
        "pre_OLS_plane": "P_before_OLS_incident_W is the encoder output field incident on the Gaussian OLS modal projection; runtime trace key P_before_fiber_W is retained only as the internal legacy field name",
        "post_OLS_plane": "P_after_OLS_reemitted_decoder_entry_W is the re-emitted field at the Decoder entrance after the Gaussian OLS projection",
        "camera_power": "P_CMOS_W is the integral of raw CMOS W/m^2 intensity over camera pixel area before gain or clipping",
        "other_planes": "SLM, encoder entrance and each layer aperture power are recorded at the stated reference planes",
    })
    final = {
        "schema": "lightgen_frozen_slm_8bit_final_decision.v1",
        "task_id": TASK_ID,
        "run_id": identity["run_id"],
        "status": "FINALIZED",
        "decision": decision,
        "decision_text": decision_text,
        "acceptance_limit": PRECISION_LIMIT,
        "acceptance_is_hardware_tolerance": False,
        "sample_denominator_per_condition": 200,
        "conditions": summary,
        "records": records,
        "zero_update_scope": {
            "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
            "new_sampling": 0, "new_human_review": 0, "input_resolve": 0,
        },
        "evidence_recomputed_from_saved_gray_and_outputs": True,
        "delivered_arrays": [
            "encoded_commands.pt", "samples/{condition}/{grid}/{sample_key}.pt",
            "target_actual_coefficients.pt", "reference_coefficients_and_cmos.pt",
            "power_table_definition.json", "comparison_atlas_G1.png", "comparison_atlas_G4.png",
        ],
    }
    _atomic_json(run_dir / "final_decision.json", final)
    _write_csv(run_dir / "per_image_power_and_error.csv", records)
    _write_atlas(run_dir / "comparison_atlas.png", inputs, outputs)
    _write_report(run_dir / "report.md", decision, decision_text, summary, preflight, inputs["source_identity"])
    _atomic_json(run_dir / "windows_reconsumption.json", {
        "schema": "lightgen_frozen_slm_8bit_windows_reconsumption.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "decision": decision, "conditions": summary,
        "recomputed_records": len(records), "gray_rows": 200,
        "producer_numeric_rows_independently_checked": 600,
        "output_sample_rows_recomputed": 1200,
        "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
        "new_sampling": 0, "new_human_review": 0,
    })
    identity["status"] = "CLOSED"
    identity["final_decision"] = decision
    identity["passed"] = {
        condition: summary[condition]["passed"] for condition in CONDITIONS
    }
    _atomic_json(run_dir / "run_identity.json", identity)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=EXPECTED_ATTEMPT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run_dir: Path | None = None
    try:
        config_path = args.config.resolve()
        config = source_run._load_yaml(config_path)
        _expected_config(config, config_path)
        run_dir = args.output_dir.resolve()
        identity = _prepare_output(run_dir, allow_closed=args.verify_only)
        inputs = _source_inputs(config)
        _save_input_authority(run_dir, config, inputs, identity["run_id"])
        bundle = _ensure_encoded_bundle(run_dir, inputs)
        if args.prepare_only:
            _atomic_json(run_dir / "preparation.json", {
                "schema": "lightgen_frozen_slm_8bit_preparation.v1",
                "task_id": TASK_ID, "run_id": identity["run_id"],
                "source_run_id": inputs["source_identity"]["run_id"],
                "source_witnesses_recomputed": 200,
                "gray_code_rows": 200,
                "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
                "new_sampling": 0, "new_human_review": 0,
            })
            print(json.dumps({"status": "INPUTS_PREPARED", "run_id": identity["run_id"],
                              "gray_codes": str(run_dir / "encoded_commands.pt")}, ensure_ascii=False))
            return
        if args.verify_only:
            result = _final_consumer(run_dir, identity, inputs, bundle)
            print(json.dumps({
                "status": result["status"], "decision": result["decision"],
                "passed": {condition: result["conditions"][condition]["passed"] for condition in CONDITIONS},
                "total_per_condition": 200,
            }, ensure_ascii=False))
            return
        _require(torch.cuda.is_available(), "AutoDL CUDA is required for the frozen optical measurement")
        device = torch.device("cuda")
        models = {grid: source_run._new_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
        state_before = {str(grid): source_run._snapshot_state(model) for grid, model in models.items()}
        if args.preflight_only:
            result = _make_preflight(run_dir, identity, inputs, config, bundle, models, state_before)
            print(json.dumps({"status": result["status"], "run_id": identity["run_id"],
                              "preflight": str(run_dir / "preflight.json")}, ensure_ascii=False))
            return
        receipt = _load_json(run_dir / "preflight.json")
        _require(receipt.get("status") == "PREFLIGHT_PASSED" and receipt.get("run_id") == identity["run_id"],
                 "--execute requires this attempt's passing 20-row production preflight")
        _execute_full(run_dir, identity, inputs, bundle, models)
        print(json.dumps({"status": "POPULATION_COMPLETE_PENDING_WINDOWS_RECONSUMPTION",
                          "run_id": identity["run_id"], "condition_grid_outputs": 1200}, ensure_ascii=False))
    except Exception as exc:
        if run_dir is not None and run_dir.is_dir():
            _atomic_json(run_dir / "engineering_failure.json", {
                "task_id": TASK_ID, "status": "ENGINEERING_INVALID",
                "exception_type": type(exc).__name__, "message": str(exc),
            })
        print(f"ENGINEERING_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
