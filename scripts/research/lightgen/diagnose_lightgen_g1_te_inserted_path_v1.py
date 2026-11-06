"""Frozen G1 full-path diagnostic with a uniform Te film after OLS.

This entry point consumes the saved 2025 paper proxy and existing LightGen
checkpoints. It does not fit parameters or modify the production runtime.
"""
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

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as mapper_ref
from scripts.research.lightgen import execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder_ref


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_te_inserted_path_v1.yaml"
TOLERANCES = {
    "field_and_raw_intensity_scaling_relative": 1.0e-6,
    "normalized_shape_relative": 1.0e-6,
    "thin_film_power_ratio_absolute": 1.0e-9,
    "grid_2048_vs_4096_raw_intensity_relative": 1.0e-3,
}
STATES = ("solid_25C", "quasi_liquid_80C")
CONDITIONS = ("bare", "te_10nm_solid_25C", "te_10nm_quasi_liquid_80C")


class DiagnosticStop(RuntimeError):
    """An invalid diagnostic input or failed numerical preflight."""


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
    require(path.resolve() == DEFAULT_CONFIG.resolve(), "this diagnostic uses its explicit fixed configuration")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict)
            and config.get("schema") == "lightgen_g1_te_inserted_path.v1"
            and config.get("task_id") == "LIGHTGEN_G1_TE_INSERTED_PATH_V1"
            and config.get("evidence_tier") == "diagnostic_only",
            "configuration identity or evidence tier changed")
    require(config["population"] == {
        "selection": "saved_native_capture_indices", "rows": 100,
        "per_class": 10, "classes": 10, "sample_order": "saved_index_order",
    }, "fixed G1 cohort selection changed")
    require(config["model"]["training_updates"] == 0
            and config["model"]["checkpoint_branch"] == "C"
            and config["model"]["checkpoint_seed"] == 2026101501
            and config["model"]["cumulative_step"] == 45000
            and config["model"]["mapper_checkpoint_step"] == 5000
            and config["model"]["primary_grid"] == 2048
            and config["model"]["reference_grid"] == 4096,
            "frozen model identity, grid, or update count changed")
    require(config["material"]["sample_identity"] ==
            "LIT-TE-ADMA2025-THERMOCHROMIC-FIGURE-PROXY"
            and config["material"]["insertion_plane"] == "OLS_output_generator_entrance"
            and config["material"]["relative_transfer"] ==
            "tE_film_at_z50 / tE_bare_at_z50",
            "film sample identity or insertion definition changed")
    require(config["controls"] == {
        "no_film": True, "identity_transfer": True, "global_phase_only": True,
        "equal_loss_amplitude_only": True,
        "wrong_power_as_field_amplitude_negative_control": True,
    }, "the fixed controls changed")
    require(config["tolerances"] == TOLERANCES, "predeclared numerical tolerances changed")
    return config


def _complex(real: float, imag: float) -> complex:
    value = complex(float(real), float(imag))
    require(math.isfinite(value.real) and math.isfinite(value.imag), "nonfinite complex film coefficient")
    return value


def _load_transfer_table(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    source = config["source"]
    measurement = _read_json(_resolve(source["te_measurement"]))
    preflight = _read_json(_resolve(source["te_preflight"]))
    require(measurement.get("status") == "valid"
            and measurement.get("evidence_tier") == "diagnostic_only"
            and measurement.get("model_identity", {}).get("sample_identity") ==
            config["material"]["sample_identity"]
            and measurement.get("model_identity", {}).get("paper_doi") == "10.1002/adma.202511261"
            and measurement.get("model_identity", {}).get("wavelength_nm") == 532.0
            and measurement.get("model_identity", {}).get("reference_plane_nm") == 50.0,
            "saved Te measurement does not match the fixed paper proxy")
    require(preflight.get("status") == "passed"
            and preflight.get("evidence_tier") == "diagnostic_only",
            "saved Te thin-film preflight did not pass")
    bare = measurement["bare_reference"]
    bare_t = _complex(bare["tE_z50"]["real"], bare["tE_z50"]["imag"])
    bare_power = float(bare["T"])
    require(abs(abs(bare_t) ** 2 / float(measurement["model_identity"]["n_glass"])
                - bare_power) <= 1.0e-9,
            "bare reference field and power normalization disagree")
    nominal_rows = measurement["nominal_10nm_rows"]
    require(len(nominal_rows) == 2 and {row["state"] for row in nominal_rows} == set(STATES),
            "saved 10 nm state rows are incomplete")
    records: dict[float, dict[str, Any]] = {}
    for row in nominal_rows:
        require(float(row["thickness_nm"]) == 10.0, "nominal Te film thickness changed")
        t = _complex(row["tE_z50_real"], row["tE_z50_imag"])
        q = t / bare_t
        expected_power_ratio = float(row["T"]) / bare_power
        require(abs(abs(q) ** 2 - expected_power_ratio) <= TOLERANCES["thin_film_power_ratio_absolute"],
                f"complex field transfer and saved power ratio disagree for {row['state']}")
        require(abs(math.atan2(q.imag, q.real) - float(row["phase_vs_bare_rad"])) <= 1.0e-10,
                f"saved phase and common-reference transfer disagree for {row['state']}")
        records[10.0, row["state"]] = {
            **row, "q": q, "power_ratio": expected_power_ratio,
            "power_ratio_min": float(row["T_digitization_min"]) / bare_power,
            "power_ratio_max": float(row["T_digitization_max"]) / bare_power,
        }
    sweep_path = _resolve(source["te_thickness_sweep"])
    with sweep_path.open(encoding="utf-8-sig", newline="") as stream:
        sweep_rows = list(csv.DictReader(stream))
    for raw in sweep_rows:
        thickness = float(raw["thickness_nm"])
        if thickness != 50.0:
            continue
        state = raw["state"]
        t = _complex(raw["tE_z50_real"], raw["tE_z50_imag"])
        q = t / bare_t
        ratio = float(raw["T"]) / bare_power
        require(abs(abs(q) ** 2 - ratio) <= TOLERANCES["thin_film_power_ratio_absolute"],
                f"50 nm transfer and power ratio disagree for {state}")
        records[50.0, state] = {
            "thickness_nm": thickness, "state": state, "temperature_c": int(raw["temperature_c"]),
            "phase": raw["phase"], "n": float(raw["n"]), "k": float(raw["k"]),
            "R": float(raw["R"]), "T": float(raw["T"]), "A_flux": float(raw["A_flux"]),
            "T_digitization_min": float(raw["T_digitization_min"]),
            "T_digitization_max": float(raw["T_digitization_max"]),
            "phase_vs_bare_rad": math.atan2(q.imag, q.real),
            "q": q, "power_ratio": ratio,
            "power_ratio_min": float(raw["T_digitization_min"]) / bare_power,
            "power_ratio_max": float(raw["T_digitization_max"]) / bare_power,
        }
    records[0.0, "bare"] = {
        "thickness_nm": 0.0, "state": "bare", "temperature_c": "", "phase": "bare",
        "R": float(bare["R"]), "T": bare_power, "A_flux": float(bare["A_flux"]),
        "q": 1.0 + 0.0j, "power_ratio": 1.0, "power_ratio_min": 1.0, "power_ratio_max": 1.0,
    }
    require({(d, s) for d, s in records if d != 0.0} ==
            {(d, s) for d in (10.0, 50.0) for s in STATES},
            "saved Te data do not contain both states at 10 and 50 nm")
    return {"measurement": measurement, "preflight": preflight, "bare_t": bare_t,
            "bare_power": bare_power, "records": records}, records


def _load_inputs(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    source = config["source"]
    optical_checkpoint = torch.load(_resolve(source["optical_checkpoint"]), map_location="cpu", weights_only=False)
    mapper_checkpoint = torch.load(_resolve(source["mapper_checkpoint"]), map_location="cpu", weights_only=False)
    cohort = torch.load(_resolve(source["g1_cohort"]), map_location="cpu", weights_only=False)
    pilot_config = yaml.safe_load(_resolve(source["mapper_config"]).read_text(encoding="utf-8"))
    prior_path = _resolve(source["fitted_gaussian_prior"])
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    require(set(prior) == set(range(10)), "frozen C prior no longer contains one Gaussian per G1 class")
    for label in range(10):
        item = prior[label]
        require(tuple(item["mean"].shape) == (200,)
                and tuple(item["covariance"].shape) == (200, 200)
                and tuple(item["cholesky"].shape) == (200, 200)
                and torch.isfinite(item["mean"]).all().item()
                and torch.isfinite(item["covariance"]).all().item()
                and torch.isfinite(item["cholesky"]).all().item(),
                f"frozen class-{label} G1 Gaussian prior is incomplete")
    require(optical_checkpoint.get("branch") == "C"
            and optical_checkpoint.get("seed") == 2026101501
            and optical_checkpoint.get("completed_global_step") == 45000
            and optical_checkpoint.get("evaluator_role") == "frozen_confirmation_only"
            and optical_checkpoint.get("model_identity", {}).get("ols") == "gaussian_s12_v1"
            and optical_checkpoint.get("model_identity", {}).get("ols_geometry", {}).get("transport") == "identity",
            "loaded optical checkpoint does not match frozen C45k G1 reference")
    optimizer_steps = sorted({int(torch.as_tensor(v["step"]).item())
                              for v in optical_checkpoint["optimizer"]["state"].values()
                              if isinstance(v, dict) and "step" in v})
    require(optimizer_steps == [45000], "C45k optimizer state differs from its saved endpoint")
    require(mapper_checkpoint.get("task_id") == "LIGHTGEN_G1_SLM_INPUT_MAPPING_PILOT_V1"
            and mapper_checkpoint.get("checkpoint_step") == 5000
            and mapper_checkpoint.get("completed_updates") == 5000,
            "loaded SLM mapper is not the frozen v1 step-5000 checkpoint")
    mapper_steps = sorted({int(torch.as_tensor(v["step"]).item())
                           for v in mapper_checkpoint["optimizer_state"]["state"].values()
                           if isinstance(v, dict) and "step" in v})
    require(mapper_steps == [5000], "SLM mapper optimizer state differs from step 5000")
    require(pilot_config.get("schema") == "lightgen_g1_slm_input_mapping_pilot.v1"
            and pilot_config["source"]["fitted_gaussian_prior"] == source["fitted_gaussian_prior"],
            "G1 prior reference differs from the mapping configuration")
    require(cohort.get("schema") == "lightgen_small_conditional_mixture_generated_cohort.v1"
            and cohort.get("method") == "G1" and cohort.get("stream") == 0
            and cohort.get("global_step") == 45000
            and cohort.get("model") == "C_seed_2026101501"
            and cohort.get("classifier_based_sample_selection") is False
            and cohort.get("per_image_normalization") is False,
            "saved G1 cohort identity or selection policy changed")
    indices = torch.as_tensor(cohort["native_capture_indices"], dtype=torch.long)
    labels = torch.as_tensor(cohort["requested_labels"], dtype=torch.long)[indices]
    coefficients = torch.as_tensor(cohort["ols_coefficients"])[indices]
    proposal_ids = torch.as_tensor(cohort["proposal_ids"], dtype=torch.long)[indices]
    require(indices.shape == (100,) and coefficients.shape == (100, 100)
            and coefficients.dtype == torch.complex64
            and torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 10)),
            "saved native capture rows are not 10 fixed samples per class")
    expected_indices = torch.tensor([label * 1000 + j for label in range(10) for j in range(10)])
    require(torch.equal(indices, expected_indices), "native capture indices no longer follow the fixed saved order")
    transfer_source, records = _load_transfer_table(config)

    mapper = mapper_ref._new_mapper(device, int(pilot_config["population"]["initialization_seed"]))
    mapper.load_state_dict(mapper_checkpoint["mapper_state"], strict=True)
    mapper.eval().requires_grad_(False)
    scale = torch.as_tensor(optical_checkpoint["scale"], dtype=torch.float64)
    features = mapper_ref._feature_batch(coefficients, scale, device)
    with torch.inference_mode():
        amplitude = mapper(features)
    require(amplitude.shape == (100, 1, 200, 200)
            and torch.isfinite(amplitude).all().item()
            and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0,
            "frozen mapper output is invalid as an amplitude-SLM command")

    ed_path = _resolve(pilot_config["source"]["evaluator_ed_checkpoint"])
    ec_path = _resolve(pilot_config["source"]["evaluator_ec_checkpoint"])
    evaluator_ed = mapper_ref._load_evaluator(ed_path, "ED").to(device).eval()
    evaluator_ec, ec_meta = decoder_ref._load_ec_evaluator(ec_path, device)
    evaluator_ec = evaluator_ec.to(device).eval()
    for module in (mapper, evaluator_ed, evaluator_ec):
        module.requires_grad_(False)
    return {
        "optical_checkpoint": optical_checkpoint, "mapper_checkpoint": mapper_checkpoint,
        "cohort": cohort, "indices": indices, "labels": labels, "coefficients": coefficients,
        "proposal_ids": proposal_ids, "amplitude": amplitude,
        "mapper": mapper, "evaluator_ed": evaluator_ed, "evaluator_ec": evaluator_ec,
        "ec_meta": ec_meta, "camera_gain": float(torch.as_tensor(optical_checkpoint["camera_gain"]).item()),
        "transfers": records, "transfer_source": transfer_source,
        "device": device,
    }


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _assert_unchanged(module: torch.nn.Module, before: Mapping[str, torch.Tensor], name: str) -> None:
    after = module.state_dict()
    require(set(after) == set(before)
            and all(torch.equal(after[k].detach().cpu(), before[k]) for k in before),
            f"frozen {name} changed during the diagnostic")


def _field_power(field: ComplexField) -> torch.Tensor:
    pixel_area = math.prod(field.plane.pixel_pitch_m)
    return field.tensor.abs().square().sum(dim=(-3, -2, -1)) * pixel_area


def _camera_field_from_entry(runtime: Any, entry: ComplexField) -> ComplexField:
    generated, _, _ = runtime.generator(entry, "lightgen_generator")
    output = ComplexField(
        generated.tensor, runtime.generator_exit_plane,
        OperationProvenance("lightgen_generator_exit", generated.plane.plane_id, None,
                            {"generator_layers": 2, "te_diagnostic": True}),
    )
    camera = runtime.output_relay(output)
    runtime._validate_relay_diagnostics(runtime.output_relay)
    require(torch.isfinite(camera.tensor.real).all().item()
            and torch.isfinite(camera.tensor.imag).all().item(), "CMOS field is nonfinite")
    return camera


def _transfer_field(field: ComplexField, q: complex, tag: str) -> ComplexField:
    factor = torch.as_tensor(q, dtype=field.tensor.dtype, device=field.tensor.device)
    return ComplexField(
        field.tensor * factor, field.plane,
        OperationProvenance("te_literature_proxy_transfer", field.plane.plane_id, None,
                            {"case": tag, "q_real": q.real, "q_imag": q.imag,
                             "reference_plane_nm": 50.0}),
    )


def _numpy_complex(field: ComplexField) -> np.ndarray:
    array = field.tensor.detach().cpu().numpy().astype(np.complex128, copy=True)
    if array.ndim == 4 and array.shape[:2] == (1, 1):
        return array[0, 0]
    return array


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    dtype = torch.complex128 if torch.is_complex(actual) or torch.is_complex(expected) else torch.float64
    a = actual.detach().to(dtype=dtype)
    b = expected.detach().to(dtype=dtype)
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1.0e-30))


def _shape_error(actual: torch.Tensor, baseline: torch.Tensor) -> float:
    a = actual.detach().double()
    b = baseline.detach().double()
    a = a / a.sum().clamp_min(1.0e-30)
    b = b / b.sum().clamp_min(1.0e-30)
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1.0e-30))


def _overlap_phase(actual: torch.Tensor, reference: torch.Tensor) -> float:
    overlap = torch.sum(torch.conj(reference.to(torch.complex128)) * actual.to(torch.complex128))
    return float(torch.angle(overlap).item())


def _wrapped_phase_error(observed: float, predicted: float) -> float:
    delta = observed - predicted
    return math.atan2(math.sin(delta), math.cos(delta))


def _transfer_observation(baseline: Mapping[str, Any], actual: Mapping[str, Any],
                          expected_q: complex, expected_power_ratio: float) -> dict[str, float | bool]:
    entry_error = _relative_error(actual["entry"].tensor,
                                  baseline["entry"].tensor * torch.as_tensor(
                                      expected_q, dtype=baseline["entry"].tensor.dtype,
                                      device=baseline["entry"].tensor.device))
    camera_error = _relative_error(actual["camera"].tensor,
                                   baseline["camera"].tensor * torch.as_tensor(
                                       expected_q, dtype=baseline["camera"].tensor.dtype,
                                       device=baseline["camera"].tensor.device))
    intensity_error = _relative_error(actual["raw"], baseline["raw"] * expected_power_ratio)
    power_ratio = float(actual["raw"].sum().item()
                        / baseline["raw"].sum().clamp_min(1.0e-30).item())
    entry_phase = _overlap_phase(actual["entry"].tensor, baseline["entry"].tensor)
    camera_phase = _overlap_phase(actual["camera"].tensor, baseline["camera"].tensor)
    predicted_phase = math.atan2(expected_q.imag, expected_q.real)
    shape_error = _shape_error(actual["raw"], baseline["raw"])
    passed = (
        max(entry_error, camera_error, intensity_error)
        <= TOLERANCES["field_and_raw_intensity_scaling_relative"]
        and abs(power_ratio - expected_power_ratio)
        <= TOLERANCES["field_and_raw_intensity_scaling_relative"]
        and shape_error <= TOLERANCES["normalized_shape_relative"]
        and abs(_wrapped_phase_error(entry_phase, predicted_phase)) <= 1.0e-6
        and abs(_wrapped_phase_error(camera_phase, predicted_phase)) <= 1.0e-6
    )
    return {
        "generator_entry_field_relative_error": entry_error,
        "cmos_field_relative_error": camera_error,
        "raw_intensity_relative_error": intensity_error,
        "observed_power_ratio": power_ratio,
        "expected_power_ratio": expected_power_ratio,
        "normalized_shape_relative_error": shape_error,
        "generator_entry_phase_observed_rad": entry_phase,
        "cmos_phase_observed_rad": camera_phase,
        "phase_predicted_rad": predicted_phase,
        "generator_entry_phase_residual_rad": _wrapped_phase_error(entry_phase, predicted_phase),
        "cmos_phase_residual_rad": _wrapped_phase_error(camera_phase, predicted_phase),
        "passed": passed,
    }


def _require_transfer_observation(observation: Mapping[str, Any], name: str) -> None:
    require(bool(observation["passed"]), f"transfer acceptance rejected {name}: {dict(observation)}")


def _run_branch(runtime: Any, adapter: Any, arm: str, coefficients: torch.Tensor,
                amplitude: torch.Tensor | None) -> dict[str, Any]:
    if arm == "mapped":
        require(amplitude is not None, "mapped arm requires frozen mapper output")
        result = runtime.forward_from_image(amplitude.square().to(dtype=torch.float64))
        entry = result.generator_input_field
        camera = runtime.output_relay(result.generator_output_field)
        runtime._validate_relay_diagnostics(runtime.output_relay)
        raw = result.raw_camera.tensor
        require(_relative_error(camera.tensor.abs().square(), raw) <= 1.0e-12,
                "replayed mapped camera field disagrees with production CMOS intensity")
        return {"entry": entry, "camera": camera, "raw": raw,
                "state": result.z_ols, "production_result": result}
    require(arm == "direct", "unknown input arm")
    state = ComplexOLSState(
        coefficients=coefficients.to(device=runtime.ols.operator.transport.device,
                                      dtype=torch.complex128),
        source="matched_empirical_prior", basis_id=runtime.ols.basis_id,
        mode_geometry=runtime.ols.mode_identity(),
    )
    entry = runtime.ols.write_to_generator_input(state, runtime.generator_entrance_plane)
    camera = _camera_field_from_entry(runtime, entry)
    raw = camera.tensor.abs().square()
    # The direct adapter is an independent consumer of the same frozen OLS coefficients.
    reference = adapter.forward_from_ols(state.coefficients)["raw_intensity_w_per_m2"]
    require(_relative_error(raw, reference) <= 1.0e-12,
            "direct OLS replay disagrees with the saved production adapter consumer")
    return {"entry": entry, "camera": camera, "raw": raw,
            "state": state, "production_result": None}


def _apply_transfer(runtime: Any, baseline: Mapping[str, Any], q: complex,
                    tag: str) -> dict[str, Any]:
    entry = _transfer_field(baseline["entry"], q, tag)
    camera = _camera_field_from_entry(runtime, entry)
    return {"entry": entry, "camera": camera, "raw": camera.tensor.abs().square()}


def _readout(raw: torch.Tensor, pixel_area: float, gain: float) -> torch.Tensor:
    return mapper_ref._readout_for_evaluator(raw, pixel_area, gain)


def _measure_one(runtime: Any, adapter: Any, arm: str, coefficients: torch.Tensor,
                 amplitude: torch.Tensor | None, transfer_specs: Mapping[tuple[float, str], Mapping[str, Any]],
                 pixel_area: float, camera_gain: float, ed: torch.nn.Module, ec: torch.nn.Module,
                 *, controls: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = _run_branch(runtime, adapter, arm, coefficients, amplitude)
    raw_by_condition = {"bare": baseline["raw"].detach()}
    camera_by_condition = {"bare": baseline["camera"]}
    entry_by_condition = {"bare": baseline["entry"]}
    transfer_by_condition: dict[str, complex] = {"bare": 1.0 + 0.0j}
    control_rows: list[dict[str, Any]] = []
    for state, condition in zip(STATES, CONDITIONS[1:]):
        spec = transfer_specs[10.0, state]
        film = _apply_transfer(runtime, baseline, spec["q"], condition)
        raw_by_condition[condition] = film["raw"].detach()
        camera_by_condition[condition] = film["camera"]
        entry_by_condition[condition] = film["entry"]
        transfer_by_condition[condition] = spec["q"]

    if controls:
        spec = transfer_specs[10.0, "solid_25C"]
        q = spec["q"]
        phase = math.atan2(q.imag, q.real)
        phase_q = complex(math.cos(phase), math.sin(phase))
        wrong_q = complex(abs(q) ** 2 * math.cos(phase), abs(q) ** 2 * math.sin(phase))
        for tag, control_q, expected_power in (
            ("identity", 1.0 + 0.0j, 1.0),
            ("global_phase_only", phase_q, 1.0),
            ("equal_loss_amplitude_only", complex(abs(q), 0.0), abs(q) ** 2),
            ("wrong_power_used_as_field_amplitude", wrong_q, abs(q) ** 4),
        ):
            control = _apply_transfer(runtime, baseline, control_q, tag)
            expected_q = q if tag == "wrong_power_used_as_field_amplitude" else control_q
            expected_ratio = abs(q) ** 2 if tag == "wrong_power_used_as_field_amplitude" else expected_power
            observation = _transfer_observation(baseline, control, expected_q, expected_ratio)
            rejected_as_expected = not bool(observation["passed"])
            if tag == "wrong_power_used_as_field_amplitude":
                require(rejected_as_expected,
                        "production transfer acceptance did not reject power transmission used as field amplitude")
            else:
                _require_transfer_observation(observation, tag)
            control_rows.append({
                "arm": arm, "control": tag, "q_real": control_q.real, "q_imag": control_q.imag,
                **observation, "rejected_as_expected": rejected_as_expected,
            })

    records: list[dict[str, Any]] = []
    views: dict[str, torch.Tensor] = {}
    logits: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for condition, raw in raw_by_condition.items():
            view = _readout(raw, pixel_area, camera_gain)
            views[condition] = view.detach()
            logits[condition] = {
                "ed": ed(view),
                "ec": ec(view)[1],
            }
    for condition, raw in raw_by_condition.items():
        spec = transfer_specs[0.0, "bare"] if condition == "bare" else transfer_specs[
            10.0, STATES[CONDITIONS.index(condition) - 1]
        ]
        q = transfer_by_condition[condition]
        camera = camera_by_condition[condition]
        entry = entry_by_condition[condition]
        transfer_observation = _transfer_observation(baseline, {
            "entry": entry, "camera": camera, "raw": raw,
        }, q, abs(q) ** 2)
        _require_transfer_observation(transfer_observation, f"{arm}/{condition}")
        field_error = float(transfer_observation["cmos_field_relative_error"])
        raw_error = _relative_error(raw, baseline["raw"] * abs(q) ** 2)
        shape_error = _shape_error(raw, baseline["raw"])
        p_after = float(raw.sum().item() * pixel_area)
        p_before = float(baseline["raw"].sum().item() * pixel_area)
        unclipped = raw * (pixel_area * camera_gain)
        clip_fraction = float((unclipped >= 1.0).double().mean().item())
        ed_pred = int(logits[condition]["ed"].argmax(1).item())
        ec_pred = int(logits[condition]["ec"].argmax(1).item())
        state = "bare" if spec is None else spec["state"]
        records.append({
            "arm": arm, "condition": condition, "state": state,
            "thickness_nm": float(spec["thickness_nm"]),
            "temperature_c": spec["temperature_c"] if condition == "bare" else int(spec["temperature_c"]),
            "R": float(spec["R"]), "T": float(spec["T"]), "A": float(spec["A_flux"]),
            "q_real": q.real, "q_imag": q.imag, "q_abs": abs(q),
            "q_phase_rad": math.atan2(q.imag, q.real),
            "material_power_ratio_expected": abs(q) ** 2,
            "generator_entry_power_before_W": float(_field_power(baseline["entry"])[0].item()),
            "generator_entry_power_after_W": float(_field_power(entry)[0].item()),
            "generator_entry_field_scaling_relative_error": _relative_error(
                entry.tensor, baseline["entry"].tensor * torch.as_tensor(
                    q, dtype=entry.tensor.dtype, device=entry.tensor.device
                )
            ),
            "generator_entry_phase_predicted_rad": float(transfer_observation["phase_predicted_rad"]),
            "generator_entry_phase_observed_rad": float(transfer_observation["generator_entry_phase_observed_rad"]),
            "generator_entry_phase_residual_rad": float(transfer_observation["generator_entry_phase_residual_rad"]),
            "cmos_power_before_W": p_before, "cmos_power_after_W": p_after,
            "cmos_power_ratio_observed": p_after / max(p_before, 1.0e-30),
            "cmos_power_ratio_relative_error": abs(p_after / max(p_before, 1.0e-30) - abs(q) ** 2)
                                                / max(abs(q) ** 2, 1.0e-30),
            "cmos_field_scaling_relative_error": field_error,
            "cmos_field_phase_predicted_rad": float(transfer_observation["phase_predicted_rad"]),
            "cmos_field_phase_observed_rad": float(transfer_observation["cmos_phase_observed_rad"]),
            "cmos_field_phase_residual_rad": float(transfer_observation["cmos_phase_residual_rad"]),
            "cmos_intensity_scaling_relative_error": raw_error,
            "normalized_shape_relative_error": shape_error,
            "fixed_exposure_clipped_fraction": clip_fraction,
            "ED_prediction": ed_pred, "EC_prediction": ec_pred,
            "ED_logits": logits[condition]["ed"].squeeze(0).detach().cpu().tolist(),
            "EC_logits": logits[condition]["ec"].squeeze(0).detach().cpu().tolist(),
            "view28": views[condition][0, 0].detach().cpu().numpy(),
            "raw_cmos_w_m2": raw[0, 0].detach().cpu().numpy(),
            "camera_field": _numpy_complex(camera),
            "generator_entry_field": _numpy_complex(entry),
        })
    extras = {
        "baseline": baseline, "raw_by_condition": raw_by_condition,
        "camera_by_condition": camera_by_condition, "entry_by_condition": entry_by_condition,
        "views": views, "control_rows": control_rows,
    }
    return {"records": records, "extras": extras}, extras


def _run_preflight(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is required for the production-path preflight")
    mapper_ref._set_numerics()
    device = torch.device("cuda")
    context = _load_inputs(config, device)
    q10 = context["transfers"]
    labels = context["labels"]
    selected = [0, 90]
    checks: list[dict[str, Any]] = []
    models = {grid: mapper_ref._new_optical_model(context["optical_checkpoint"], grid, device)
              for grid in (2048, 4096)}
    model_snapshots = {grid: _snapshot(model) for grid, model in models.items()}
    module_snapshots = {name: _snapshot(context[key]) for name, key in (
        ("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec"))}
    measurements = []
    cross_raw: dict[tuple[int, int, str, str], np.ndarray] = {}
    for grid in (2048, 4096):
        adapter = models[grid]
        runtime = adapter.runtime
        pixel_area = math.prod(runtime.camera_plane.pixel_pitch_m)
        for idx in selected:
            native_index = int(context["indices"][idx])
            coeff = context["coefficients"][idx:idx + 1].to(device)
            amp = context["amplitude"][idx:idx + 1]
            for arm in ("direct", "mapped"):
                result, _ = _measure_one(
                    runtime, adapter, arm, coeff, None if arm == "direct" else amp,
                    q10, pixel_area, context["camera_gain"], context["evaluator_ed"],
                    context["evaluator_ec"], controls=True,
                )
                for row in result["records"]:
                    require(row["generator_entry_field_scaling_relative_error"]
                            <= TOLERANCES["field_and_raw_intensity_scaling_relative"],
                            f"preflight generator-entry field scaling failed for {arm}/{row['condition']}/grid{grid}")
                    require(row["cmos_power_ratio_relative_error"]
                            <= TOLERANCES["field_and_raw_intensity_scaling_relative"],
                            f"preflight CMOS power ratio failed for {arm}/{row['condition']}/grid{grid}")
                    require(row["cmos_field_scaling_relative_error"] <= TOLERANCES["field_and_raw_intensity_scaling_relative"],
                            f"preflight field scaling failed for {arm}/{row['condition']}/grid{grid}")
                    require(row["cmos_intensity_scaling_relative_error"] <= TOLERANCES["field_and_raw_intensity_scaling_relative"],
                            f"preflight intensity scaling failed for {arm}/{row['condition']}/grid{grid}")
                    require(row["normalized_shape_relative_error"] <= TOLERANCES["normalized_shape_relative"],
                            f"preflight normalized shape changed for {arm}/{row['condition']}/grid{grid}")
                    require(max(abs(row["generator_entry_phase_residual_rad"]),
                                abs(row["cmos_field_phase_residual_rad"])) <= 1.0e-6,
                            f"preflight observed phase disagrees with relative transfer for {arm}/{row['condition']}/grid{grid}")
                for control in result["extras"]["control_rows"]:
                    if control["control"] == "wrong_power_used_as_field_amplitude":
                        require(not control["passed"] and control["rejected_as_expected"]
                                and abs(control["observed_power_ratio"] - control["expected_power_ratio"]) > 1.0e-2,
                                "production acceptance did not reject power transmission used as field amplitude")
                    else:
                        require(control["passed"]
                                and abs(control["observed_power_ratio"] - control["expected_power_ratio"]) <= 1.0e-6,
                                f"production control acceptance failed: {control['control']}")
                for condition, raw in result["extras"]["raw_by_condition"].items():
                    cross_raw[grid, idx, arm, condition] = raw.detach().cpu().numpy()
                if grid == 2048:
                    baseline = result["extras"]["baseline"]
                    for state in STATES:
                        spec50 = context["transfers"][50.0, state]
                        film50 = _apply_transfer(runtime, baseline, spec50["q"], f"te_50nm_{state}")
                        observation50 = _transfer_observation(baseline, film50, spec50["q"], spec50["power_ratio"])
                        _require_transfer_observation(observation50, f"50 nm {arm}/{state}")
                        controls = result["extras"]["control_rows"]
                        controls.append({"arm": arm, "control": f"te_50nm_{state}",
                                         **observation50, "rejected_as_expected": False})
                measurements.append({"grid": grid, "sample_index": native_index,
                                     "native_capture_index": native_index,
                                     "population_position": idx, "label": int(labels[idx]),
                                     "arm": arm, "checks": [_metric_row({}, row) for row in result["records"]],
                                     "controls": result["extras"]["control_rows"]})
    # Independent pixel-level cross-grid check on the two fixed rows and each condition.
    grid_errors: list[float] = []
    grid_checks: list[dict[str, Any]] = []
    for idx in selected:
        for arm in ("direct", "mapped"):
            for condition in CONDITIONS:
                a = cross_raw[2048, idx, arm, condition]
                b = cross_raw[4096, idx, arm, condition]
                error = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))
                grid_errors.append(error)
                grid_checks.append({"sample_index": int(context["indices"][idx]),
                                    "native_capture_index": int(context["indices"][idx]),
                                    "population_position": idx, "arm": arm, "condition": condition,
                                    "raw_intensity_relative_error": error})
    for grid, model in models.items():
        _assert_unchanged(model, model_snapshots[grid], f"G1 C45k optical model grid {grid}")
    for name, key in (("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec")):
        _assert_unchanged(context[key], module_snapshots[name], name)
    require(max(grid_errors) <= TOLERANCES["grid_2048_vs_4096_raw_intensity_relative"],
            "2048/4096 pixel-level raw CMOS outputs exceed the predeclared tolerance")
    checks.extend([
        {"name": "frozen_input_identity_and_population", "passed": True, "rows": 100},
        {"name": "saved_te_complex_transfer_matches_power_ratio", "passed": True},
        {"name": "production_path_two_rows_two_grids_two_arms", "passed": True,
         "sample_indices": [int(context["indices"][index]) for index in selected],
         "population_positions": selected, "grids": [2048, 4096]},
        {"name": "identity_phase_and_equal_loss_controls", "passed": True},
        {"name": "wrong_power_as_field_amplitude_rejected", "passed": True},
        {"name": "frozen_model_mapper_and_evaluators", "passed": True},
        {"name": "cross_grid_raw_intensity_relative_error", "passed": True,
         "max_relative_error": max(grid_errors),
         "tolerance": TOLERANCES["grid_2048_vs_4096_raw_intensity_relative"]},
    ])
    payload = {
        "schema": "lightgen_g1_te_inserted_path_preflight.v1", "status": "PREFLIGHT_PASSED",
        "evidence_tier": "diagnostic_only", "optimizer_updates": 0,
        "model_identity": _model_identity(config, context), "checks": checks,
        "production_rows": measurements,
        "cross_grid_checks": grid_checks,
        "stop": "Run the frozen 100-row measurement only when this receipt is PREFLIGHT_PASSED.",
    }
    _write_json_new(output / "preflight.json", payload)
    return payload


def _model_identity(config: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    pack = context["cohort"]
    transfer_rows = {
        f"{thickness:g}nm_{state}": _json_transfer_record(record)
        for (thickness, state), record in sorted(context["transfers"].items(), key=lambda item: item[0])
    }
    return {
        "optical_branch": "C", "optical_seed": 2026101501, "optical_step": 45000,
        "mapper_step": 5000,
        "source_paths": {key: value for key, value in config["source"].items()},
        "cohort": "C_seed_2026101501/G1_stream_0.pt", "cohort_stream": 0,
        "sample_count": 100, "per_class": 10,
        "saved_indices": context["indices"].tolist(),
        "native_capture_indices": context["indices"].tolist(),
        "population_positions": list(range(100)),
        "labels": context["labels"].tolist(), "proposal_ids": context["proposal_ids"].tolist(),
        "te_sample_identity": config["material"]["sample_identity"],
        "paper_doi": "10.1002/adma.202511261", "wavelength_nm": 532.0,
        "reference_plane_nm": 50.0, "insertion_plane": config["material"]["insertion_plane"],
        "nominal_thickness_nm": config["material"]["nominal_thickness_nm"],
        "control_thickness_nm": config["material"]["control_thickness_nm"],
        "transfer_parameters": transfer_rows,
        "primary_grid": config["model"]["primary_grid"],
        "reference_grid": config["model"]["reference_grid"],
        "camera_gain_w_inverse": context["camera_gain"],
        "cohort_schema": pack["schema"], "optimizer_updates": 0,
    }


def _json_transfer_record(record: Mapping[str, Any]) -> dict[str, Any]:
    q = record["q"]
    return {
        **{key: value for key, value in record.items() if key != "q"},
        "q_real": float(q.real),
        "q_imag": float(q.imag),
    }


def _metric_row(sample: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items()
            if k not in {"view28", "raw_cmos_w_m2", "camera_field", "generator_entry_field"}}


def _run_grid(config: Mapping[str, Any], context: dict[str, Any], grid: int,
              sample_indices: list[int], *, controls: bool) -> dict[str, Any]:
    device = context["device"]
    adapter = mapper_ref._new_optical_model(context["optical_checkpoint"], grid, device)
    adapter.eval().requires_grad_(False)
    runtime = adapter.runtime
    snapshot = _snapshot(adapter)
    pixel_area = math.prod(runtime.camera_plane.pixel_pitch_m)
    arm_data: dict[str, dict[str, Any]] = {}
    for arm in ("direct", "mapped"):
        sample_records: list[dict[str, Any]] = []
        sample_raw: dict[str, list[np.ndarray]] = {cond: [] for cond in CONDITIONS}
        sample_views: dict[str, list[np.ndarray]] = {cond: [] for cond in CONDITIONS}
        field_samples: dict[str, Any] = {
            "bare_generator_entry": [], "bare_camera": [],
            "te": {cond: {"generator_entry": [], "camera": []} for cond in CONDITIONS[1:]},
        }
        field_metadata: dict[str, Any] | None = None
        control_rows: list[dict[str, Any]] = []
        for index in sample_indices:
            native_index = int(context["indices"][index])
            coeff = context["coefficients"][index:index + 1].to(device)
            amp = context["amplitude"][index:index + 1]
            result, _ = _measure_one(
                runtime, adapter, arm, coeff, None if arm == "direct" else amp,
                context["transfers"], pixel_area, context["camera_gain"],
                context["evaluator_ed"], context["evaluator_ec"],
                controls=(controls and index % 10 == 0),
            )
            baseline = result["extras"]["baseline"]
            if field_metadata is None:
                field_metadata = {
                    "generator_entry": {
                        "plane": baseline["entry"].plane.to_dict(),
                        "provenance": baseline["entry"].provenance.to_dict(),
                    },
                    "camera": {
                        "plane": baseline["camera"].plane.to_dict(),
                        "provenance": baseline["camera"].provenance.to_dict(),
                    },
                    "conditions": {
                        condition: {
                            "generator_entry": {
                                "plane": entry.plane.to_dict(),
                                "provenance": entry.provenance.to_dict(),
                            },
                            "camera": {
                                "plane": result["extras"]["camera_by_condition"][condition].plane.to_dict(),
                                "provenance": result["extras"]["camera_by_condition"][condition].provenance.to_dict(),
                            },
                        }
                        for condition, entry in result["extras"]["entry_by_condition"].items()
                    },
                }
            field_samples["bare_generator_entry"].append(_numpy_complex(baseline["entry"]))
            field_samples["bare_camera"].append(_numpy_complex(baseline["camera"]))
            control_rows.extend({"sample_index": native_index, "native_capture_index": native_index,
                                 "population_position": index,
                                 "label": int(context["labels"][index]), **row}
                                for row in result["extras"]["control_rows"])
            for row in result["records"]:
                row.update({"sample_index": native_index, "native_capture_index": native_index,
                            "population_position": index,
                            "label": int(context["labels"][index]),
                            "proposal_id": int(context["proposal_ids"][index])})
                row["ED_correct"] = int(row["ED_prediction"] == row["label"])
                row["EC_correct"] = int(row["EC_prediction"] == row["label"])
                sample_records.append(row)
                condition = row["condition"]
                sample_raw[condition].append(row["raw_cmos_w_m2"])
                sample_views[condition].append(row["view28"])
                if condition != "bare":
                    field_samples["te"][condition]["generator_entry"].append(row["generator_entry_field"])
                    field_samples["te"][condition]["camera"].append(row["camera_field"])
            if controls and index % 10 == 0:
                for state in STATES:
                    spec50 = context["transfers"][50.0, state]
                    film50 = _apply_transfer(runtime, baseline, spec50["q"], f"te_50nm_{state}")
                    observation50 = _transfer_observation(baseline, film50, spec50["q"], spec50["power_ratio"])
                    _require_transfer_observation(observation50, f"50 nm {arm}/{index}/{state}")
                    control_rows.append({"sample_index": native_index, "native_capture_index": native_index,
                                         "population_position": index,
                                         "label": int(context["labels"][index]),
                                         "arm": arm, "control": f"te_50nm_{state}",
                                         **observation50, "rejected_as_expected": False})
        _assert_unchanged(adapter, snapshot, f"G1 model grid {grid}")
        arm_data[arm] = {
            "records": sample_records,
            "raw": {k: np.stack(v).astype(np.float64, copy=False) for k, v in sample_raw.items()},
            "views": {k: np.stack(v).astype(np.float32, copy=False) for k, v in sample_views.items()},
            "selected_fields": field_samples, "field_metadata": field_metadata,
            "sample_indices": [int(context["indices"][index]) for index in sample_indices],
            "native_capture_indices": [int(context["indices"][index]) for index in sample_indices],
            "population_positions": list(sample_indices),
            "labels": [int(context["labels"][index]) for index in sample_indices],
            "proposal_ids": [int(context["proposal_ids"][index]) for index in sample_indices],
            "controls": control_rows,
        }
    return {
        "grid": grid, "pixel_area_m2": pixel_area, "arms": arm_data,
        "state_checks": [{"name": f"c45k_optical_model_grid_{grid}_unchanged", "passed": True}],
    }


def _write_json_new(path: Path, value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.write("\n")


def _write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty result table: {path.name}")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                             for k, v in row.items()})


def _write_power_sweep(output: Path, config: Mapping[str, Any], context: Mapping[str, Any],
                       primary: Mapping[str, Any], transfer_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    sweep_path = _resolve(config["source"]["te_thickness_sweep"])
    with sweep_path.open(encoding="utf-8-sig", newline="") as stream:
        te_rows = list(csv.DictReader(stream))
    bare_power = float(transfer_source["bare_power"])
    rows: list[dict[str, Any]] = []
    for arm, arm_data in primary["arms"].items():
        base_power = np.asarray([
            float(r["cmos_power_before_W"]) for r in arm_data["records"] if r["condition"] == "bare"
        ], dtype=np.float64)
        sample_labels = [int(r["label"]) for r in arm_data["records"] if r["condition"] == "bare"]
        sample_ids = [int(r["sample_index"]) for r in arm_data["records"] if r["condition"] == "bare"]
        for te in te_rows:
            thickness = float(te["thickness_nm"])
            state = te["state"]
            ratio = float(te["T"]) / bare_power
            ratio_min = float(te["T_digitization_min"]) / bare_power
            ratio_max = float(te["T_digitization_max"]) / bare_power
            for i, incident_output in enumerate(base_power):
                rows.append({
                    "arm": arm, "sample_index": sample_ids[i], "label": sample_labels[i],
                    "thickness_nm": thickness, "state": state, "temperature_c": int(te["temperature_c"]),
                    "local_R": float(te["R"]), "local_T": float(te["T"]), "local_A": float(te["A_flux"]),
                    "T_over_bare": ratio, "T_over_bare_digitization_min": ratio_min,
                    "T_over_bare_digitization_max": ratio_max,
                    "predicted_cmos_power_W": float(incident_output * ratio),
                    "predicted_cmos_power_digitization_min_W": float(incident_output * ratio_min),
                    "predicted_cmos_power_digitization_max_W": float(incident_output * ratio_max),
                    "scope": "analytic transfer scaling of the saved bare full-path output; no per-thickness propagation",
                })
    _write_csv_new(output / "thickness_power_sweep.csv", rows)
    return rows


def _compose_labeled_gallery(frames: np.ndarray, labels: list[int], *, columns: int,
                             sample_indices: list[int] | None = None) -> tuple[np.ndarray, list[dict[str, int]], float]:
    require(frames.ndim == 3 and len(frames) == len(labels),
            "gallery expects one 2D frame and label per saved sample")
    indices = list(range(len(frames))) if sample_indices is None else sample_indices
    require(len(indices) == len(frames), "gallery sample identities do not match the saved frames")
    height, width = frames.shape[-2:]
    rows = math.ceil(len(frames) / columns)
    canvas = np.zeros((rows * height, columns * width), dtype=frames.dtype)
    positions: list[dict[str, int]] = []
    class_counts: dict[int, int] = {}
    max_error = 0.0
    for sample_position, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        row = int(label)
        col = class_counts.get(row, 0)
        require(0 <= row < rows and col < columns, "gallery class layout exceeds its declared rows/columns")
        class_counts[row] = col + 1
        tile = canvas[row * height:(row + 1) * height, col * width:(col + 1) * width]
        tile[...] = frame
        max_error = max(max_error, float(np.max(np.abs(tile - frame), initial=0.0)))
        positions.append({"sample_position": sample_position, "sample_index": int(indices[sample_position]),
                          "native_capture_index": int(indices[sample_position]),
                          "label": row, "row": row, "column": col})
    return canvas, positions, max_error


def _save_arrays(output: Path, primary: Mapping[str, Any], reference: Mapping[str, Any],
                 camera_gain: float) -> dict[str, Any]:
    camera_arrays: dict[str, np.ndarray] = {}
    pixel_power_arrays: dict[str, np.ndarray] = {}
    native_arrays: dict[str, np.ndarray] = {}
    field_arrays: dict[str, np.ndarray] = {}
    field_metadata: dict[str, Any] = {
        "schema": "lightgen_g1_te_inserted_path_complex_fields.v1",
        "complex_dtype": "complex128",
        "sample_index_semantics": "native_capture_indices from the saved cohort; population_positions index the selected 100-row subset",
        "camera_model_scope": "current numerical camera plane; not a calibrated GENSE2020 sensor model",
        "grids": {},
    }
    image_metric_rows: list[dict[str, Any]] = []
    pixel_power_errors: list[float] = []
    fixed_native_errors: list[float] = []
    view_errors: list[float] = []
    for grid_name, result in (("2048", primary), ("4096", reference)):
        grid_metadata: dict[str, Any] = {"pixel_area_m2": result["pixel_area_m2"], "arms": {}}
        for arm, data in result["arms"].items():
            labels = [int(v) for v in data["labels"]]
            indices = [int(v) for v in data["sample_indices"]]
            grid_metadata["arms"][arm] = {
                "native_capture_indices": indices, "population_positions": data["population_positions"],
                "labels": labels, "proposal_ids": data["proposal_ids"],
                "planes": data["field_metadata"],
            }
            field_prefix = f"{grid_name}_{arm}"
            fields = data["selected_fields"]
            field_arrays[f"{field_prefix}_native_capture_indices"] = np.asarray(indices, dtype=np.int64)
            field_arrays[f"{field_prefix}_population_positions"] = np.asarray(
                data["population_positions"], dtype=np.int64
            )
            field_arrays[f"{field_prefix}_labels"] = np.asarray(labels, dtype=np.int64)
            field_arrays[f"{field_prefix}_proposal_ids"] = np.asarray(data["proposal_ids"], dtype=np.int64)
            field_arrays[f"{field_prefix}_bare_generator_entry"] = np.stack(fields["bare_generator_entry"])
            field_arrays[f"{field_prefix}_bare_camera"] = np.stack(fields["bare_camera"])
            for condition in CONDITIONS[1:]:
                field_arrays[f"{field_prefix}_{condition}_generator_entry"] = np.stack(
                    fields["te"][condition]["generator_entry"]
                )
                field_arrays[f"{field_prefix}_{condition}_camera"] = np.stack(
                    fields["te"][condition]["camera"]
                )
            raw_conditions = data["raw"]
            bare_raw = raw_conditions["bare"]
            area = float(result["pixel_area_m2"])
            for condition, raw in raw_conditions.items():
                key = f"{grid_name}_{arm}_{condition}"
                views = data["views"][condition]
                pixel_power = raw * area
                unclipped_native = raw * (area * camera_gain)
                native = np.clip(unclipped_native, 0.0, 1.0)
                pixel_power_arrays[key + "_pixel_power_w"] = pixel_power
                native_arrays[key + "_native_fixed_exposure"] = native
                camera_arrays[key + "_raw_cmos_w_m2"] = raw
                camera_arrays[key + "_fixed_exposure_view28"] = views
                power_error = abs(float(pixel_power.sum()) - float(raw.sum()) * area) / max(
                    abs(float(raw.sum()) * area), 1.0e-30
                )
                pixel_power_errors.append(power_error)
                native_error = float(np.max(np.abs(native - np.clip(unclipped_native, 0.0, 1.0)), initial=0.0))
                fixed_native_errors.append(native_error)
                pooled = _readout(torch.from_numpy(raw).unsqueeze(1), area, camera_gain).squeeze(1).numpy()
                view_error = float(np.linalg.norm(pooled - views) / max(float(np.linalg.norm(views)), 1.0e-30))
                view_errors.append(view_error)
                if condition == "bare":
                    continue
                for local, sample_index in enumerate(indices):
                    bare_frame = bare_raw[local]
                    raw_frame = raw[local]
                    bare_native = np.clip(bare_frame * (area * camera_gain), 0.0, 1.0)
                    normalized_bare = bare_frame / max(float(bare_frame.sum()), 1.0e-30)
                    normalized_raw = raw_frame / max(float(raw_frame.sum()), 1.0e-30)
                    image_metric_rows.append({
                        "grid": int(grid_name), "arm": arm, "sample_index": sample_index,
                        "native_capture_index": sample_index,
                        "population_position": data["population_positions"][local],
                        "proposal_id": data["proposal_ids"][local],
                        "label": labels[local], "condition": condition,
                        "cmos_power_ratio": float(raw_frame.sum() / max(float(bare_frame.sum()), 1.0e-30)),
                        "raw_normalized_shape_relative_error": float(
                            np.linalg.norm(normalized_raw - normalized_bare)
                            / max(float(np.linalg.norm(normalized_bare)), 1.0e-30)
                        ),
                        "native_fixed_exposure_relative_error": float(
                            np.linalg.norm(native[local] - bare_native)
                            / max(float(np.linalg.norm(bare_native)), 1.0e-30)
                        ),
                        "bare_native_clipped_fraction": float(np.mean(bare_frame * (area * camera_gain) >= 1.0)),
                        "te_native_clipped_fraction": float(np.mean(unclipped_native[local] >= 1.0)),
                        "view28_relative_error": float(
                            np.linalg.norm(views[local] - data["views"]["bare"][local])
                            / max(float(np.linalg.norm(data["views"]["bare"][local])), 1.0e-30)
                        ),
                    })
        field_metadata["grids"][grid_name] = grid_metadata
    require(len(image_metric_rows) == 440, f"image-difference consumer expected 440 rows; got {len(image_metric_rows)}")
    require(max(pixel_power_errors) <= 1.0e-12, "saved pixel powers do not sum to raw CMOS power")
    require(max(fixed_native_errors) <= 1.0e-12, "native fixed-exposure image does not match the declared clamp")
    require(max(view_errors) <= 1.0e-6, "saved 28x28 views do not reproduce the fixed-exposure readout")
    for filename, values in (("cmos_outputs.npz", camera_arrays),
                             ("cmos_pixel_power_w.npz", pixel_power_arrays),
                             ("cmos_fixed_exposure_native.npz", native_arrays),
                             ("complex_fields.npz", field_arrays)):
        with (output / filename).open("xb") as stream:
            np.savez_compressed(stream, **values)
    _write_json_new(output / "complex_field_metadata.json", field_metadata)
    _write_csv_new(output / "image_delta_metrics.csv", image_metric_rows)
    return {
        "pixel_power_max_relative_error": max(pixel_power_errors),
        "native_fixed_exposure_max_absolute_error": max(fixed_native_errors),
        "fixed_exposure_28x28_reproduction_max_relative_error": max(view_errors),
        "image_delta_rows": len(image_metric_rows),
        "complex_fields": "complex_fields.npz",
        "field_metadata": "complex_field_metadata.json",
        "camera_sampling": field_metadata["grids"]["2048"]["arms"]["mapped"]["planes"]["camera"]["plane"],
    }


def _save_galleries(output: Path, result: Mapping[str, Any], camera_gain: float) -> dict[str, Any]:
    gallery_records: list[dict[str, Any]] = []
    maximum_tile_error = 0.0
    for arm, data in result["arms"].items():
        labels = [int(v) for v in data["labels"]]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
        for axis, condition, title in zip(axes, CONDITIONS, ("Bare", "Te 10 nm · solid 25 °C", "Te 10 nm · quasi-liquid 80 °C"), strict=True):
            frames = data["views"][condition]
            gallery, positions, tile_error = _compose_labeled_gallery(
                frames, labels, columns=10, sample_indices=data["sample_indices"]
            )
            require(tile_error == 0.0, f"fixed-exposure gallery tile mismatch for {arm}/{condition}")
            maximum_tile_error = max(maximum_tile_error, tile_error)
            gallery_records.append({"arm": arm, "condition": condition, "kind": "fixed_exposure_28x28",
                                    "sample_count": len(frames), "positions": positions,
                                    "max_tile_reconstruction_error": tile_error})
            axis.imshow(gallery, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([14 + 28 * i for i in range(10)], labels=list(range(10)))
            axis.set_ylabel("requested class")
        figure.suptitle(f"G1 {arm} path · fixed gain and exposure · {len(labels)} saved samples")
        figure.savefig(output / f"gallery_{arm}_triptych.png", dpi=240)
        figure.savefig(output / f"gallery_{arm}_triptych.pdf")
        plt.close(figure)

        normalized_figure, normalized_axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
        for axis, condition, title in zip(normalized_axes, CONDITIONS, ("Bare", "Te 10 nm · solid 25 °C", "Te 10 nm · quasi-liquid 80 °C"), strict=True):
            raw = data["raw"][condition]
            normalized = raw / np.maximum(raw.sum(axis=(1, 2), keepdims=True), 1.0e-30)
            gallery, positions, tile_error = _compose_labeled_gallery(
                normalized.astype(np.float64), labels, columns=10, sample_indices=data["sample_indices"]
            )
            require(tile_error == 0.0, f"normalized-shape gallery tile mismatch for {arm}/{condition}")
            maximum_tile_error = max(maximum_tile_error, tile_error)
            gallery_records.append({"arm": arm, "condition": condition, "kind": "raw_total_power_normalized",
                                    "sample_count": len(normalized), "positions": positions,
                                    "max_tile_reconstruction_error": tile_error})
            axis.imshow(gallery, cmap="gray", vmin=0.0, vmax=float(np.max(gallery)), interpolation="nearest")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([14 + 28 * i for i in range(10)], labels=list(range(10)))
            axis.set_ylabel("requested class")
        normalized_figure.suptitle(f"G1 {arm} path · each raw CMOS frame normalized by its own total power")
        normalized_figure.savefig(output / f"gallery_{arm}_total_power_normalized_shape.png", dpi=240)
        plt.close(normalized_figure)

        first_positions = [labels.index(label) for label in range(10)]
        native_figure, native_axes = plt.subplots(10, 3, figsize=(8, 22), constrained_layout=True)
        pixel_area = float(result["pixel_area_m2"])
        with np.load(output / "cmos_fixed_exposure_native.npz", allow_pickle=False) as saved_native:
            for condition_column, condition in enumerate(CONDITIONS):
                native_frames = saved_native[f"2048_{arm}_{condition}_native_fixed_exposure"]
                for label, local in enumerate(first_positions):
                    axis = native_axes[label, condition_column]
                    frame = native_frames[local]
                    expected_frame = np.clip(
                        data["raw"][condition][local] * (pixel_area * camera_gain), 0.0, 1.0
                    )
                    require(np.array_equal(frame, expected_frame),
                            f"native gallery frame does not match saved raw CMOS for {arm}/{condition}/{label}")
                    axis.imshow(frame, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                    axis.set_title(f"class {label} · sample {data['sample_indices'][local]}" if condition_column == 0
                                   else f"class {label}", fontsize=7)
                    axis.set_axis_off()
                    gallery_records.append({
                        "arm": arm, "condition": condition, "kind": "native_fixed_exposure_first_per_class",
                        "sample_count": 1, "sample_index": int(data["sample_indices"][local]),
                        "native_capture_index": int(data["sample_indices"][local]),
                        "label": label, "source_frame_exact": bool(np.array_equal(frame, expected_frame)),
                    })
        for axis, title in zip(native_axes[0], ("Bare", "Te solid 25 °C", "Te quasi-liquid 80 °C"), strict=True):
            axis.set_title(title, fontsize=9)
        native_figure.suptitle(f"G1 {arm} path · native numerical-camera pixels · fixed exposure")
        native_figure.savefig(output / f"gallery_{arm}_fixed_exposure_native_first_per_class.png", dpi=180)
        plt.close(native_figure)
    payload = {
        "schema": "lightgen_g1_te_inserted_path_gallery_mapping.v1",
        "layout": "row=requested class; column=saved order within class",
        "maximum_tile_reconstruction_error": maximum_tile_error,
        "gallery_checks": gallery_records,
    }
    _write_json_new(output / "gallery_mapping.json", payload)
    require(maximum_tile_error == 0.0 and len(gallery_records) == 72,
            "saved gallery tiles do not map exactly to the reported source frames")
    return {"maximum_tile_reconstruction_error": maximum_tile_error,
            "gallery_checks": len(gallery_records), "mapping": "gallery_mapping.json"}


def _save_thickness_plot(output: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for state, color, style in (("solid", "#1874B4", "-"), ("quasi_liquid", "#C5561A", "--")):
        selected = [r for r in rows if r["state"].startswith(state)]
        by_d: dict[float, list[dict[str, Any]]] = {}
        for row in selected:
            by_d.setdefault(float(row["thickness_nm"]), []).append(row)
        x = sorted(by_d)
        y = [np.mean([r["T_over_bare"] for r in by_d[d]]) for d in x]
        lo = [np.mean([r["T_over_bare_digitization_min"] for r in by_d[d]]) for d in x]
        hi = [np.mean([r["T_over_bare_digitization_max"] for r in by_d[d]]) for d in x]
        ax.plot(x, y, color=color, linestyle=style, label=state.replace("_", " "))
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Te thickness (nm)")
    ax.set_ylabel("CMOS power relative to bare-path output")
    ax.set_xlim(1, 50)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D7DEE5", linewidth=0.7)
    ax.legend(frameon=False)
    ax.set_title("532 nm literature-proxy Te · analytic transfer scaling")
    fig.savefig(output / "thickness_power_ratio.png", dpi=300)
    fig.savefig(output / "thickness_power_ratio.pdf")
    plt.close(fig)


def _summarize(result: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm, data in result["arms"].items():
        conditions: dict[str, Any] = {}
        for condition in CONDITIONS:
            rows = [r for r in data["records"] if r["condition"] == condition]
            conditions[condition] = {
                "n": len(rows),
                "mean_cmos_power_W": float(np.mean([r["cmos_power_after_W"] for r in rows])),
                "mean_power_ratio_vs_bare": float(np.mean([r["cmos_power_ratio_observed"] for r in rows])),
                "max_complex_field_scaling_error": float(max(r["cmos_field_scaling_relative_error"] for r in rows)),
                "max_raw_intensity_scaling_error": float(max(r["cmos_intensity_scaling_relative_error"] for r in rows)),
                "max_normalized_shape_error": float(max(r["normalized_shape_relative_error"] for r in rows)),
                "max_cmos_phase_residual_rad": float(max(abs(r["cmos_field_phase_residual_rad"]) for r in rows)),
                "mean_cmos_phase_predicted_rad": float(np.mean([r["cmos_field_phase_predicted_rad"] for r in rows])),
                "mean_cmos_phase_observed_rad": float(np.mean([r["cmos_field_phase_observed_rad"] for r in rows])),
                "mean_fixed_exposure_clipped_fraction": float(np.mean([r["fixed_exposure_clipped_fraction"] for r in rows])),
                "ED_correct": int(sum(r["ED_prediction"] == r["label"] for r in rows)),
                "EC_correct": int(sum(r["EC_prediction"] == r["label"] for r in rows)),
            }
        summary[arm] = conditions
    return summary


def _render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# G1 full-path Te literature-proxy diagnostic",
        "",
        f"Status: **{payload['status']}** · evidence tier: `diagnostic_only` · updates: 0",
        "",
        "## Fixed optical model and insertion point",
        "",
        f"The saved C45k G1 optical model and the frozen v1 step-5000 SLM mapper were run on {payload['model_identity']['sample_count']} `native_capture_indices` from `G1_stream_0.pt` ({payload['model_identity']['per_class']} per class). The film is inserted after OLS re-emission and before the first generator layer. The source is the 2025 Advanced Materials figure-digitized 532 nm proxy, not a target sputtered sample.",
        "",
        "The field multiplier is `q=tE_film(z=50 nm)/tE_bare(z=50 nm)`. The existing OLS field is interpreted as the bare-facet transmitted field. This relative transfer avoids adding the bare Fresnel loss twice. Reflection is included in local R/T/A accounting; backward feedback into the fiber or encoder is outside this forward-only diagnostic.",
        "",
        "## CMOS and auxiliary evaluator results",
        "",
        f"All displayed CMOS views use the saved C45k camera gain ({payload['fixed_camera']['gain_w_inverse']:.10g} W^-1) and the current numerical camera plane ({payload['fixed_camera']['numerical_camera_plane']['shape']} pixels; pitch {payload['fixed_camera']['numerical_camera_plane']['pixel_pitch_m']} m). This is not a calibrated GENSE2020 sensor model. Normalized-shape values are computed from raw, unclipped intensities. E_D/E_C are frozen auxiliary readouts and do not establish human readability or a new model gate.",
        "",
        "| Path | Condition | Mean CMOS power ratio | Max raw-shape error | Predicted / observed phase (rad) | Max phase residual | Mean clipped fraction | E_D correct | E_C correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("mapped", "direct"):
        for condition in CONDITIONS:
            row = summary[arm][condition]
            lines.append(f"| {arm} | {condition} | {row['mean_power_ratio_vs_bare']:.8g} | {row['max_normalized_shape_error']:.3g} | {row['mean_cmos_phase_predicted_rad']:.6g} / {row['mean_cmos_phase_observed_rad']:.6g} | {row['max_cmos_phase_residual_rad']:.3g} | {row['mean_fixed_exposure_clipped_fraction']:.4%} | {row['ED_correct']}/{row['n']} | {row['EC_correct']}/{row['n']} |")
    lines.extend([
        "",
        "## Local thin-film states and thickness range",
        "",
        "The 10 nm state and the 1–50 nm power curves consume the saved thin-film calculation. Full G1 propagation was performed at 10 nm for 100 rows and at 50 nm for the one-per-class reference subset; intermediate thicknesses are analytic power scaling of the saved bare-path output.",
        "",
        "The 25 °C solid and 80 °C quasi-liquid settings are thermally defined paper states. They do not supply an optical-intensity threshold, CW nonlinear coefficient, or ReLU response. Digitization envelopes are not experimental error bars.",
        "",
        "Observed global phase is independently measured from `arg(sum(conj(E_bare) * E_Te))` at generator entrance and CMOS; predicted phase and wrapped residual are separate columns in `sample_metrics.csv`.",
        "",
        "## Numerical checks",
        "",
    ])
    for check in payload["checks"]:
        lines.append(f"- `{check['name']}`: **{'PASS' if check['passed'] else 'FAIL'}**")
    lines.extend([
        "",
        "`complex_fields.npz` stores the full primary cohort and all ten reference-grid fields; `complex_field_metadata.json` records sample identities, PlaneSpec values, and propagation provenance. Pixel power, native fixed-exposure images, image-difference metrics, and gallery mapping are regenerated by this same entrypoint.",
        "",
    ])
    lines.extend([
        "",
        "The full result is limited to this frozen G1 numerical path, saved 100-row cohort, scalar uniform-film transfer, and fixed camera gain. Any change beyond uniform field scaling or global phase indicates a numerical or implementation discrepancy; it is not evidence of nonlinear gain.",
        "",
        "Next simulation step: investigate source-matched nonlinear Te models under their published excitation conditions, then decide whether any parameter set supports a distinct 532 nm intensity-dependent study. The optical thin-film and full-path results remain proxies pending target-device characterization; physical preparation is not required for this simulation stage.",
        "",
    ])
    return "\n".join(lines)


def _write_measurement(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is required for the 100-row full-path measurement")
    mapper_ref._set_numerics()
    device = torch.device("cuda")
    context = _load_inputs(config, device)
    frozen_modules = {name: _snapshot(context[key]) for name, key in (
        ("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec"))}
    output.mkdir(parents=True, exist_ok=True)
    preflight_path = output / "preflight.json"
    require(preflight_path.is_file(), "measurement requires the saved production preflight")
    preflight = _read_json(preflight_path)
    current_identity = _model_identity(config, context)
    _validate_preflight_receipt(preflight, current_identity)
    expected_files = {"measurement.json", "sample_metrics.csv", "control_metrics.csv",
                      "thickness_power_sweep.csv", "cmos_outputs.npz", "cmos_pixel_power_w.npz",
                      "cmos_fixed_exposure_native.npz", "complex_fields.npz", "complex_field_metadata.json",
                      "image_delta_metrics.csv", "gallery_mapping.json",
                      "gallery_mapped_triptych.png", "gallery_mapped_triptych.pdf",
                      "gallery_direct_triptych.png", "gallery_direct_triptych.pdf",
                      "gallery_mapped_total_power_normalized_shape.png",
                      "gallery_direct_total_power_normalized_shape.png",
                      "gallery_mapped_fixed_exposure_native_first_per_class.png",
                      "gallery_direct_fixed_exposure_native_first_per_class.png",
                      "thickness_power_ratio.png", "thickness_power_ratio.pdf", "report.md"}
    require(not any((output / name).exists() for name in expected_files),
            "measurement output path already contains result files")
    indices = list(range(100))
    primary = _run_grid(config, context, 2048, indices, controls=True)
    reference_indices = list(range(0, 100, 10))
    reference = _run_grid(config, context, 4096, reference_indices, controls=False)
    frozen_state_checks = primary["state_checks"] + reference["state_checks"]
    for name, key in (("mapper", "mapper"), ("E_D", "evaluator_ed"), ("E_C", "evaluator_ec")):
        _assert_unchanged(context[key], frozen_modules[name], f"{name} during full measurement")
        frozen_state_checks.append({"name": f"{name}_unchanged_during_full_measurement", "passed": True})
    cross_grid: list[dict[str, Any]] = []
    for arm in ("direct", "mapped"):
        low = primary["arms"][arm]
        high = reference["arms"][arm]
        for local, population_position in enumerate(reference_indices):
            sample_index = int(context["indices"][population_position])
            for condition in CONDITIONS:
                error = float(np.linalg.norm(
                    low["raw"][condition][population_position] - high["raw"][condition][local]
                ) / max(np.linalg.norm(high["raw"][condition][local]), 1.0e-30))
                require(error <= TOLERANCES["grid_2048_vs_4096_raw_intensity_relative"],
                        f"2048/4096 full-path CMOS output differs by {error:.6g} for {arm}/{sample_index}/{condition}")
                cross_grid.append({"arm": arm, "sample_index": sample_index,
                                   "population_position": population_position,
                                   "condition": condition, "raw_cmos_relative_error": error})
    sample_rows = [row for arm in ("mapped", "direct") for row in primary["arms"][arm]["records"]]
    control_rows = [row for arm in ("mapped", "direct") for row in primary["arms"][arm]["controls"]]
    summary = _summarize(primary)
    transfer_source = context["transfer_source"]
    metric_checks = [
        {"name": "full_population_has_100_rows_per_arm_and_condition",
         "passed": all(len([r for r in primary["arms"][arm]["records"] if r["condition"] == cond]) == 100
                        for arm in ("direct", "mapped") for cond in CONDITIONS)},
        {"name": "full_path_field_and_intensity_scaling", "passed": max(
            max(r["cmos_field_scaling_relative_error"], r["generator_entry_field_scaling_relative_error"],
                r["cmos_power_ratio_relative_error"]) for r in sample_rows)
            <= TOLERANCES["field_and_raw_intensity_scaling_relative"]
            and max(r["cmos_intensity_scaling_relative_error"] for r in sample_rows)
            <= TOLERANCES["field_and_raw_intensity_scaling_relative"]},
        {"name": "normalized_shape_invariance", "passed": max(
            r["normalized_shape_relative_error"] for r in sample_rows)
            <= TOLERANCES["normalized_shape_relative"]},
        {"name": "observed_global_phase_matches_transfer", "passed": max(
            max(abs(r["generator_entry_phase_residual_rad"]), abs(r["cmos_field_phase_residual_rad"]))
            for r in sample_rows) <= 1.0e-6},
        {"name": "controls_use_production_transfer_acceptance", "passed": (
            all(row["passed"] for row in control_rows
                if row["control"] != "wrong_power_used_as_field_amplitude")
            and all((not row["passed"]) and row["rejected_as_expected"] for row in control_rows
                    if row["control"] == "wrong_power_used_as_field_amplitude")
        )},
        {"name": "2048_4096_grid_consistency", "passed": max(
            r["raw_cmos_relative_error"] for r in cross_grid)
            <= TOLERANCES["grid_2048_vs_4096_raw_intensity_relative"]},
        {"name": "model_mapper_and_evaluators_frozen",
         "passed": all(check["passed"] for check in frozen_state_checks),
         "state_checks": frozen_state_checks},
    ]
    require(all(check["passed"] for check in metric_checks),
            "one or more full-path numerical acceptance checks failed")
    payload = {
        "schema": "lightgen_g1_te_inserted_path_measurement.v1", "status": "valid",
        "evidence_tier": "diagnostic_only", "optimizer_updates": 0,
        "scientific_question": config["scientific_question"],
        "model_identity": _model_identity(config, context),
        "film_identity": {
            "sample": config["material"]["sample_identity"], "paper_doi": "10.1002/adma.202511261",
            "paper_states": {s: _json_transfer_record(context["transfers"][10.0, s]) for s in STATES},
            "bare_reference_t_z50": {"real": transfer_source["bare_t"].real,
                                      "imag": transfer_source["bare_t"].imag},
            "bare_reference_T": transfer_source["bare_power"],
            "relative_complex_transfers": {
                s: {"real": context["transfers"][10.0, s]["q"].real,
                    "imag": context["transfers"][10.0, s]["q"].imag}
                for s in STATES
            },
            "reference_plane_nm": 50.0,
            "model_scope": "uniform scalar local film after OLS re-emission; reflected field does not return upstream",
        },
        "fixed_camera": {"gain_w_inverse": context["camera_gain"],
                         "pixel_area_m2": primary["pixel_area_m2"],
                         "numerical_camera_plane": primary["arms"]["mapped"]["field_metadata"]["camera"]["plane"],
                         "readout": "raw W/m2 integrated over camera pixel, fixed gain, clamp [0,1], adaptive average pool 28x28"},
        "summary": summary,
        "checks": metric_checks,
        "cross_grid_checks": cross_grid,
        "sample_metrics": "sample_metrics.csv", "controls": "control_metrics.csv",
        "camera_outputs": "cmos_outputs.npz", "complex_fields": "complex_fields.npz",
        "thickness_power_sweep": "thickness_power_sweep.csv",
        "stop_point": "No G1 training, material fitting, thermal dynamics, physical preparation, or measurement is included.",
    }
    _write_csv_new(output / "sample_metrics.csv", [_metric_row({}, row) for row in sample_rows])
    _write_csv_new(output / "control_metrics.csv", control_rows)
    sweep_rows = _write_power_sweep(output, config, context, primary, transfer_source)
    derived_checks = _save_arrays(output, primary, reference, context["camera_gain"])
    gallery_checks = _save_galleries(output, primary, context["camera_gain"])
    payload["checks"].extend([
        {"name": "saved_pixel_power_and_fixed_exposure_reconsumption", "passed": (
            derived_checks["pixel_power_max_relative_error"] <= 1.0e-12
            and derived_checks["native_fixed_exposure_max_absolute_error"] <= 1.0e-12
            and derived_checks["fixed_exposure_28x28_reproduction_max_relative_error"] <= 1.0e-6
            and derived_checks["image_delta_rows"] == 440
        ), "details": derived_checks},
        {"name": "gallery_tiles_match_saved_frames", "passed": (
            gallery_checks["maximum_tile_reconstruction_error"] == 0.0
            and gallery_checks["gallery_checks"] == 72
        ), "details": gallery_checks},
    ])
    require(all(check["passed"] for check in payload["checks"]),
            "one or more saved-output or gallery consumers failed")
    _save_thickness_plot(output, sweep_rows)
    _write_json_new(output / "measurement.json", payload)
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_render_report(_read_json(output / "measurement.json")))
    return payload


def _validate_preflight_receipt(preflight: Mapping[str, Any], current_identity: Mapping[str, Any]) -> None:
    require(preflight.get("status") == "PREFLIGHT_PASSED"
            and preflight.get("evidence_tier") == "diagnostic_only",
            "the current run does not have a passing no-update production preflight")
    require(preflight.get("model_identity") == current_identity,
            "preflight receipt does not match the loaded model, cohort, film parameters, grids, or camera gain")


def _preflight_command(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    require(not output.exists(), "preflight output directory already exists; use a new explicit attempt path")
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run_preflight(config, output)
    except Exception as exc:
        payload = {
            "schema": "lightgen_g1_te_inserted_path_preflight.v1",
            "status": "INVALID_HARNESS", "evidence_tier": "diagnostic_only",
            "optimizer_updates": 0, "failure_type": type(exc).__name__, "failure": str(exc),
            "stop": "No full measurement was started.",
        }
        if not (output / "preflight.json").exists():
            _write_json_new(output / "preflight.json", payload)
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
    require(output.is_relative_to(ROOT / "outputs")
            and "runs" in output.parts and any(part.startswith("diagnostic-only") for part in output.parts),
            "output directory must be an explicit diagnostic-only run under outputs/.../runs/")
    if args.mode == "preflight":
        result = _preflight_command(config, output)
        print(json.dumps({"status": result["status"], "path": str(output / "preflight.json")}, ensure_ascii=False))
        return 0 if result["status"] == "PREFLIGHT_PASSED" else 2
    result = _write_measurement(config, output)
    print(json.dumps({"status": result["status"], "path": str(output / "measurement.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
