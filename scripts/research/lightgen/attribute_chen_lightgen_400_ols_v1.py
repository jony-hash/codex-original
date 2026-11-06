"""Attribute LightGen400 decoder response to brightness and OLS power allocation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ChenLightGen400Baseline,
    LightGen400Config,
    OLS_PORT_COUNT,
)
from ldm_project.v2.optics.propagation import clear_production_transfer_cache
from scripts.research.lightgen import (
    diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck,
    execute_chen_lightgen_400_mnist_baseline_v1 as baseline,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_OLS_ATTRIBUTION_V1"
DECODER_GRID = 4096
REFERENCE_GRID = 6144
WINDOW_TOLERANCE = 1.0e-4
GAIN_SCALE = 7.65732439919
BATCH_SIZE = 10
SCENARIOS = {"R0": 0.0, "R10": 0.1, "R50": 0.5}
READOUTS = ("original_gain", "fit_calibrated_gain")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite completed shard: {path}")
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _save_json_new(path: Path, value: Any) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    _atomic_json(path, value)


def _require_finite(value: torch.Tensor, name: str) -> None:
    _require(torch.isfinite(value).all().item(), f"non-finite value in {name}")


def _load_inputs(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], dict[str, Any],
                                                                          ChenLightGen400Baseline,
                                                                          dict[str, torch.Tensor],
                                                                          dict[str, Any], dict[str, torch.Tensor],
                                                                          torch.Tensor, torch.Tensor, torch.Tensor,
                                                                          dict[str, Any], dict[str, torch.Tensor]]:
    config, checkpoint, source_report, source_model, references = quickcheck._load_inputs(
        _resolve(args.checkpoint), _resolve(args.source_report), _resolve(args.config), device
    )
    _require(checkpoint["completed_updates"] == 30_000, "checkpoint is not the explicit 30000-step source")
    _require(source_report.get("status") == "VALID_FAILED", "historical baseline status changed")
    population_pixels, labels, train_indices, _ = baseline._load_data(_resolve(args.data))
    population = quickcheck._load_fixed_population(_resolve(args.fixed_population), labels, train_indices)
    generated_payload = quickcheck._load_generated(_resolve(args.generated_latents))
    selected_indices = torch.tensor(
        [1000 * label + offset for label in range(10) for offset in range(10)], dtype=torch.long
    )
    selected_labels = generated_payload["requested_labels"].index_select(0, selected_indices).long()
    expected_labels = torch.arange(10).repeat_interleave(10)
    _require(torch.equal(selected_labels, expected_labels), "fixed 100 generated rows are not 10 per requested class")
    generated = {
        "indices": selected_indices,
        "labels": selected_labels,
        "coefficients": generated_payload["coefficients"].index_select(0, selected_indices).cpu(),
    }
    _require(tuple(generated["coefficients"].shape) == (100, OLS_PORT_COUNT), "selected generation coefficient shape changed")
    _require(generated["coefficients"].is_complex(), "selected generation coefficients are not complex")
    return (config, checkpoint, source_model, references, population, generated,
            population_pixels, labels, train_indices, source_report, generated_payload)


def _make_decoder(source_model: ChenLightGen400Baseline, grid: int, device: torch.device) -> ChenLightGen400Baseline:
    candidate = ChenLightGen400Baseline(LightGen400Config(grid, True)).to(device)
    with torch.no_grad():
        candidate.encoder_phase.copy_(source_model.encoder_phase)
        candidate.decoder_phase.copy_(source_model.decoder_phase)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    candidate.eval()
    return candidate


def _transform_coefficients(coefficients: torch.Tensor, fraction: float) -> torch.Tensor:
    _require(coefficients.ndim == 2 and coefficients.shape[1] == OLS_PORT_COUNT and coefficients.is_complex(),
             "OLS coefficient input must be complex [N,100]")
    magnitude = coefficients.abs()
    _require(bool((magnitude > 0).all()), "zero OLS coefficient has undefined phase; refusing to invent one")
    power = magnitude.double().square()
    total = power.sum(dim=1, keepdim=True)
    moved = (1.0 - fraction) * power + fraction * total / OLS_PORT_COUNT
    phase = coefficients / magnitude
    transformed = moved.sqrt().to(coefficients.real.dtype) * phase
    return transformed.to(coefficients.dtype)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    dtype = torch.complex128 if actual.is_complex() else torch.float64
    actual64 = actual.to(dtype)
    expected64 = expected.to(dtype)
    denominator = torch.linalg.vector_norm(expected64)
    _require(float(denominator) > 0.0, "relative error reference is zero")
    return float(torch.linalg.vector_norm(actual64 - expected64) / denominator)


def _scenario_power_rows(
    name: str, coefficients: torch.Tensor, ids: torch.Tensor, labels: torch.Tensor, population_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    transformed = _transform_coefficients(coefficients, SCENARIOS[name])
    original_power = coefficients.abs().double().square()
    power = transformed.abs().double().square()
    total = power.sum(1)
    neff = total.square() / power.square().sum(1).clamp_min(torch.finfo(torch.float64).tiny)
    central_indices = torch.tensor([r * 10 + c for r in range(2, 8) for c in range(2, 8)])
    outer_indices = torch.tensor([i for i in range(100) if i not in set(central_indices.tolist())])
    central_fraction = power.index_select(1, central_indices).sum(1) / total
    outer_fraction = power.index_select(1, outer_indices).sum(1) / total
    _require(torch.allclose(power.sum(1), original_power.sum(1), rtol=2.0e-7, atol=1.0e-18),
             f"{name} failed per-sample modal-power conservation")
    phase_error = _relative_l2(transformed / transformed.abs(), coefficients / coefficients.abs())
    summary_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    for row in range(coefficients.shape[0]):
        sample_id = int(ids[row])
        label = int(labels[row])
        root_power = power[row]
        summary_rows.append({
            "scenario": name, "population": population_name, "sample_id": sample_id, "label": label,
            "total_modal_power_w": float(total[row]), "minimum_port_power_w": float(root_power.min()),
            "ports_at_least_1nw": int((root_power >= 1.0e-9).sum()), "n_eff": float(neff[row]),
            "central_6x6_power_fraction": float(central_fraction[row]),
            "outer_36_power_fraction": float(outer_fraction[row]),
        })
        for port, value in enumerate(root_power.tolist()):
            port_rows.append({
                "scenario": name, "population": population_name, "sample_id": sample_id,
                "label": label, "port_index_zero_based": port, "port_power_w": value,
                "total_modal_power_w": float(total[row]), "n_eff": float(neff[row]),
            })
    metrics = {
        "max_total_power_relative_error": max(
            abs(float(a - b)) / max(abs(float(b)), 1.0e-30)
            for a, b in zip(power.sum(1), original_power.sum(1), strict=True)
        ),
        "phase_relative_l2_error": phase_error,
        "minimum_port_power_w": float(power.min()),
        "minimum_total_modal_power_w": float(total.min()),
        "mean_n_eff": float(neff.mean()),
        "mean_ports_at_least_1nw": float((power >= 1.0e-9).sum(1).double().mean()),
    }
    return summary_rows, port_rows, metrics


def _window_fields(result: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "decoder_entrance": result["generator_entry_field"].tensor,
        **{f"decoder_{index + 1}": field.tensor for index, field in enumerate(result["decoder_fields"])},
    }


def _window_preflight(
    model_4096: ChenLightGen400Baseline, source_model: ChenLightGen400Baseline,
    population: dict[str, Any], generated: dict[str, torch.Tensor], device: torch.device,
    roi_m: float,
) -> dict[str, Any]:
    clear_production_transfer_cache()
    model_6144 = _make_decoder(source_model, REFERENCE_GRID, device)
    roi_pixels = int(round(roi_m / model_4096.config.pixel_pitch_m))
    records: list[dict[str, Any]] = []
    confirm_positions = [100 + label * 20 for label in range(10)]
    selections = [
        ("reconstruction", int(population["confirm_rows"][position - 100]),
         population["coefficients"][position : position + 1])
        for position in confirm_positions
    ]
    selections += [
        ("generation", int(generated["indices"][label * 10]),
         generated["coefficients"][label * 10 : label * 10 + 1])
        for label in range(10)
    ]
    for scenario, fraction in SCENARIOS.items():
        for sample_type, sample_id, coefficient in selections:
            z = _transform_coefficients(coefficient, fraction).to(device)
            with torch.no_grad():
                current = model_4096.forward_from_ols(z, return_diagnostics=True)
                reference = model_6144.forward_from_ols(z, return_diagnostics=True)
            current_fields = {key: baseline._common_roi(value, roi_pixels).detach().cpu()
                              for key, value in _window_fields(current).items()}
            reference_fields = {key: baseline._common_roi(value, roi_pixels).detach().cpu()
                                for key, value in _window_fields(reference).items()}
            errors, phase = baseline._complex_relative_errors(current_fields, reference_fields)
            maximum = max(errors.values())
            records.append({"scenario": scenario, "sample_type": sample_type, "sample_id": sample_id,
                            "decoder_grid": DECODER_GRID, "reference_grid": REFERENCE_GRID,
                            "roi_m": roi_m, "roi_pixels": roi_pixels,
                            "complex_relative_error_by_plane": errors,
                            "global_phase_alignment_rad": phase,
                            "max_complex_relative_error": maximum, "passed": maximum <= WINDOW_TOLERANCE})
            del current, reference
    del model_6144
    clear_production_transfer_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"records": records, "tolerance": WINDOW_TOLERANCE,
            "passed": all(record["passed"] for record in records),
            "fixed_coefficients_consumed_without_reencoding": True}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _scenario_inputs(population: dict[str, Any], generated: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "reconstruction": {"coefficients": population["coefficients"].cpu(),
                            "ids": population["all_rows"].long().cpu(),
                            "labels": population["all_labels"].long().cpu()},
        "generation": {"coefficients": generated["coefficients"].cpu(),
                       "ids": generated["indices"].long().cpu(),
                       "labels": generated["labels"].long().cpu()},
    }


def _raw_path(output_dir: Path, scenario: str, population: str, shard: int) -> Path:
    return output_dir / "raw" / scenario / f"{population}_{shard:04d}.npz"


def _load_raw_shard(path: Path, ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as shard:
        actual_ids = torch.from_numpy(shard["sample_ids"].copy()).long()
        actual_labels = torch.from_numpy(shard["labels"].copy()).long()
        physical = torch.from_numpy(shard["physical_w_per_pixel"].copy()).float()
        decoder_powers = torch.from_numpy(shard["decoder_plane_powers_w"].copy()).double()
    _require(torch.equal(actual_ids, ids.cpu()) and torch.equal(actual_labels, labels.cpu()),
             f"saved shard keys differ from fixed batch: {path}")
    _require(tuple(physical.shape) == (ids.numel(), 1, 400, 400), f"physical readout shape changed: {path}")
    _require(tuple(decoder_powers.shape) == (ids.numel(), 5), f"Decoder power shape changed: {path}")
    _require_finite(physical, "physical W/pixel readout")
    _require_finite(decoder_powers, "five Decoder plane powers")
    return physical, decoder_powers


def _collect_arm(
    model: ChenLightGen400Baseline, scenario: str, data: dict[str, dict[str, torch.Tensor]],
    output_dir: Path, device: torch.device,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for population_name, content in data.items():
            coeffs, ids, labels = content["coefficients"], content["ids"], content["labels"]
            for start in range(0, ids.numel(), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, ids.numel())
                path = _raw_path(output_dir, scenario, population_name, start // BATCH_SIZE)
                if path.exists():
                    _load_raw_shard(path, ids[start:stop], labels[start:stop])
                    continue
                z = _transform_coefficients(coeffs[start:stop], SCENARIOS[scenario]).to(device)
                result = model.forward_from_ols(z, return_diagnostics=True)
                physical = result["physical_w_per_pixel"].detach().float().cpu()
                planes = result["plane_powers_w"].detach().double().cpu()
                _require(tuple(planes.shape) == (stop - start, 5), "Decoder power vector must include entry and four screens")
                _require_finite(physical, "forward physical W/pixel")
                _save_npz_new(
                    path,
                    sample_ids=ids[start:stop].numpy(), labels=labels[start:stop].numpy(),
                    physical_w_per_pixel=physical.numpy(), decoder_plane_powers_w=planes.numpy(),
                )
                _append_event(output_dir / "progress.jsonl", {
                    "status": "READOUT_SAVED", "scenario": scenario, "population": population_name,
                    "start": start, "stop": stop, "sample_count": stop - start,
                    "timestamp_unix": time.time(),
                })
                _atomic_json(output_dir / "progress.json", {
                    "status": "RUNNING", "scenario": scenario, "population": population_name,
                    "completed_samples_in_shard": stop, "updated_at_unix": time.time(),
                })
            counts[population_name] = ids.numel()
    return counts


def _old_readout_regression(
    output_dir: Path, previous_dir: Path, population: dict[str, Any], generated: dict[str, torch.Tensor],
    original_gain: float, current_metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    old_recon_metrics = json.loads((previous_dir / "per_sample_metrics.json").read_text(encoding="utf-8"))
    old_gen_metrics = json.loads((previous_dir / "per_sample_generation_metrics.json").read_text(encoding="utf-8"))
    old_recon = sorted(old_recon_metrics["start_4096"], key=lambda row: row["row"])
    old_gen = sorted(old_gen_metrics["start_4096"], key=lambda row: row["saved_generation_index"])
    _require(len(old_recon) == 300 and len(old_gen) == 10, "previous same-window R0 metric populations changed")
    max_metric_error = {"mse400_unclipped": 0.0, "psnr28_db": 0.0, "ssim28": 0.0,
                        "saturation_fraction": 0.0}
    current_reconstruction = sorted(
        (row for row in current_metrics["reconstruction"] if row["readout"] == "original_gain"),
        key=lambda row: row["sample_id"],
    )
    current_generation = sorted(
        (row for row in current_metrics["generation"] if row["readout"] == "original_gain"),
        key=lambda row: row["sample_id"],
    )
    _require(len(current_reconstruction) == 300 and len(current_generation) == 100,
             "R0 regression readout counts changed")
    for old, new in zip(old_recon, current_reconstruction, strict=True):
        _require(old["row"] == new["sample_id"] and old["label"] == new["label"],
                 "R0 rows do not match the prior same-window measurement")
        for metric in max_metric_error:
            max_metric_error[metric] = max(max_metric_error[metric], abs(float(old[metric]) - float(new[metric])))
    _require(max(max_metric_error.values()) <= 2.0e-5,
             f"R0 same-window reconstruction readout regression exceeded tolerance: {max_metric_error}")
    for class_index, label in enumerate(range(10)):
        new = current_generation[class_index * 10]
        old = old_gen[label]
        _require(int(old["saved_generation_index"]) == int(new["sample_id"])
                 and int(old["requested_label"]) == label,
                 "R0 generated sample key differs from prior fixed readout")
        _require(int(old["ed_prediction"]) == int(new["ed_prediction"]),
                 "R0 generated E_D prediction differs from the prior same-window readout")
    old_units = torch.load(previous_dir / "unit_readout_reconstruction.pt", map_location="cpu", weights_only=False)
    old_gen_units = torch.load(previous_dir / "unit_readout_generation.pt", map_location="cpu", weights_only=False)
    _require(old_units.get("schema") == "chen_lightgen_400_quickcheck_units.v1"
             and old_gen_units.get("schema") == "chen_lightgen_400_quickcheck_units.v1",
             "previous R0 physical unit artifacts have an unknown schema")
    new_unit_checks: dict[str, float] = {}
    for kind, saved, unit, ids in (
        ("reconstruction", old_units["by_state"]["start_4096"], "unit_readout_reconstruction.pt",
         population["confirm_rows_by_class"][:, 0].long()),
        ("generation", old_gen_units["by_state"]["start_4096"], "unit_readout_generation.pt",
         generated["indices"][::10].long()),
    ):
        _require(saved["camera_gain_w_inverse"] > 0.0, f"previous {kind} gain is invalid")
        _require(abs(float(saved["camera_gain_w_inverse"]) - original_gain) / original_gain <= 1.0e-12,
                 f"previous {kind} readout used a different frozen gain")
        _require(torch.equal(torch.tensor(saved["row_indices"] if kind == "reconstruction"
                                          else saved["saved_generation_indices"]), ids),
                 f"previous {kind} selected readout keys changed")
        positions = (100 + torch.arange(10) * 20) if kind == "reconstruction" else (torch.arange(10) * 10)
        current_physical = []
        current_task = []
        for position in positions.tolist():
            shard_index = position // BATCH_SIZE
            in_shard = position % BATCH_SIZE
            raw_path = _raw_path(output_dir, "R0", kind, shard_index)
            with np.load(raw_path, allow_pickle=False) as shard:
                physical = torch.from_numpy(shard["physical_w_per_pixel"].copy()).float()[in_shard]
            current_physical.append(physical)
            current_task.append(physical * original_gain)
        physical_error = _relative_l2(torch.stack(current_physical), saved["physical_w_per_pixel"])
        task_error = _relative_l2(torch.stack(current_task), saved["task_intensity_unclipped"])
        _require(physical_error <= 1.0e-5 and task_error <= 1.0e-5,
                 f"R0 raw physical unit regression failed for {kind}: {physical_error}, {task_error}")
        new_unit_checks[kind] = max(physical_error, task_error)
    return {"passed": True, "source_metric_max_absolute_error": max_metric_error,
            "source_raw_unit_max_relative_l2_error": new_unit_checks,
            "generated_prediction_match": True, "reconstruction_samples_checked": 300,
            "generated_samples_checked": 10, "old_reference_state": "start_4096",
            "prior_gain_w_inverse": original_gain}


def _power_scaling_check(model: ChenLightGen400Baseline, coefficients: torch.Tensor,
                         camera_gain: float, device: torch.device) -> dict[str, float]:
    z = coefficients[:1].to(device)
    with torch.no_grad():
        baseline_result = model.forward_from_ols(z, return_diagnostics=True)
        scaled_result = model.forward_from_ols(z * math.sqrt(GAIN_SCALE), return_diagnostics=True)
    base_field = baseline_result["decoder_fields"][-1].tensor
    scaled_field = scaled_result["decoder_fields"][-1].tensor
    field_error = _relative_l2(scaled_field, base_field * math.sqrt(GAIN_SCALE))
    intensity_error = _relative_l2(
        scaled_result["physical_w_per_pixel"], baseline_result["physical_w_per_pixel"] * GAIN_SCALE
    )
    task = baseline_result["physical_w_per_pixel"] * camera_gain
    restored = scaled_result["physical_w_per_pixel"] * (camera_gain / GAIN_SCALE)
    gain_restoration_error = _relative_l2(restored, task)
    _require(field_error <= 1.0e-5 and intensity_error <= 1.0e-5 and gain_restoration_error <= 1.0e-5,
             "global coefficient power scaling failed linear field/intensity/gain checks")
    return {"coefficient_amplitude_multiplier": math.sqrt(GAIN_SCALE),
            "physical_power_multiplier": GAIN_SCALE, "complex_field_relative_l2_error": field_error,
            "physical_intensity_relative_l2_error": intensity_error,
            "inverse_gain_restoration_relative_l2_error": gain_restoration_error,
            "photon_noise_or_exposure_model": "NOT_INCLUDED"}


def _negative_controls(model: ChenLightGen400Baseline, population: dict[str, Any], pixels: torch.Tensor,
                       gain: float, device: torch.device) -> dict[str, Any]:
    rows = population["confirm_rows_by_class"][:, 0].long()
    positions = 100 + torch.arange(10) * 20
    coeffs = population["coefficients"].index_select(0, positions)
    with torch.no_grad():
        physical = model.forward_from_ols(coeffs.to(device))["physical_w_per_pixel"]
    task = physical * gain
    target, _ = baseline._image_batch(pixels, rows, device)
    try:
        baseline._reconstruction_batch_metrics(task, target, rows, rows.flip(0))
    except ValueError:
        row_mismatch_rejected = True
    else:
        raise RuntimeError("production metric consumer accepted mismatched target row identities")
    _, clean_psnr, clean_ssim = baseline._reconstruction_batch_metrics(task, target, rows, rows)
    shuffled_target = target.roll(1, dims=0)
    _, shuffled_psnr, shuffled_ssim = baseline._reconstruction_batch_metrics(task, shuffled_target, rows, rows)
    clean_mse = float((task - target).square().mean())
    shuffled_mse = float((task - shuffled_target).square().mean())
    _require(shuffled_mse > clean_mse + 1.0e-6,
             "content-shuffled target control did not worsen production reconstruction MSE")
    return {"row_identity_mismatch_rejected": row_mismatch_rejected,
            "correct_content_mse400": clean_mse, "shuffled_content_mse400": shuffled_mse,
            "shuffled_mse_increase_fraction": shuffled_mse / max(clean_mse, 1.0e-12) - 1.0,
            "correct_psnr28_mean": float(clean_psnr.mean()),
            "shuffled_psnr28_mean": float(shuffled_psnr.mean()),
            "correct_ssim28_mean": float(clean_ssim.mean()),
            "shuffled_ssim28_mean": float(shuffled_ssim.mean())}


def _save_run_state(output_dir: Path, source_model: ChenLightGen400Baseline,
                    references: dict[str, torch.Tensor], population: dict[str, Any],
                    generated: dict[str, torch.Tensor], evaluator_state: dict[str, torch.Tensor]) -> None:
    path = output_dir / "run_state.pt"
    if path.exists():
        _verify_run_state(output_dir, source_model, references, population, generated, evaluator_state)
        return
    _atomic_torch_save(path, {
        "schema": "chen_lightgen_400_ols_attribution_run_state.v1",
        "source_encoder_phase": source_model.encoder_phase.detach().cpu().clone(),
        "source_decoder_phase": source_model.decoder_phase.detach().cpu().clone(),
        "references": {name: value.detach().cpu().clone() for name, value in references.items()},
        "evaluator_state": {name: value.detach().cpu().clone() for name, value in evaluator_state.items()},
        "fit_rows": population["fit_rows"].clone(), "confirm_rows": population["confirm_rows"].clone(),
        "all_rows": population["all_rows"].clone(), "all_labels": population["all_labels"].clone(),
        "coefficients": population["coefficients"].clone(),
        "encoder_power_planes_w": population["encoder_power_planes_w"].clone(),
        "generated_indices": generated["indices"].clone(), "generated_labels": generated["labels"].clone(),
        "generated_coefficients": generated["coefficients"].clone(),
    })


def _verify_run_state(output_dir: Path, source_model: ChenLightGen400Baseline,
                      references: dict[str, torch.Tensor], population: dict[str, Any],
                      generated: dict[str, torch.Tensor], evaluator_state: dict[str, torch.Tensor]) -> None:
    path = output_dir / "run_state.pt"
    _require(path.is_file(), "explicit run directory is missing saved model and population state")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == "chen_lightgen_400_ols_attribution_run_state.v1",
             "run state schema changed")
    current = {
        "source_encoder_phase": source_model.encoder_phase.detach().cpu(),
        "source_decoder_phase": source_model.decoder_phase.detach().cpu(),
        "fit_rows": population["fit_rows"], "confirm_rows": population["confirm_rows"],
        "all_rows": population["all_rows"], "all_labels": population["all_labels"],
        "coefficients": population["coefficients"],
        "encoder_power_planes_w": population["encoder_power_planes_w"],
        "generated_indices": generated["indices"], "generated_labels": generated["labels"],
        "generated_coefficients": generated["coefficients"],
    }
    for name, value in current.items():
        _require(torch.equal(saved[name], value.cpu()), f"explicit resume inputs differ from saved {name}")
    _require(saved["references"].keys() == references.keys(), "frozen reference names changed")
    for name, value in references.items():
        _require(torch.equal(saved["references"][name], value.detach().cpu()),
                 f"frozen reference {name} changed since preflight")
    _require(saved["evaluator_state"].keys() == evaluator_state.keys(), "frozen evaluator state names changed")
    for name, value in evaluator_state.items():
        _require(torch.equal(saved["evaluator_state"][name], value.detach().cpu()),
                 f"frozen evaluator parameter {name} changed since preflight")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the fixed-window optical measurement")
    device = torch.device("cuda")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done_path = output_dir / "preflight.json"
    started = time.perf_counter()
    (config, checkpoint, source_model, references, population, generated, pixels, labels,
     train_indices, source_report, full_generated) = _load_inputs(args, device)
    if done_path.is_file():
        existing = _read_json(done_path)
        _require(existing.get("status") == "READY" and existing.get("passed") is True
                 and existing.get("source_checkpoint_step") == checkpoint.get("completed_updates")
                 and existing.get("fixed_population", {}).get("fit_rows") == population["fit_rows"].tolist()
                 and existing.get("fixed_population", {}).get("confirm_rows") == population["confirm_rows"].tolist(),
                 "saved preflight does not match the explicitly loaded checkpoint and population")
        manifest = _read_json(output_dir / "run_manifest.json")
        _require(manifest.get("source_checkpoint_original_path") == str(args.checkpoint)
                 and manifest.get("source_report_original_path") == str(args.source_report)
                 and manifest.get("generated_latents_original_path") == str(args.generated_latents)
                 and manifest.get("fixed_population_original_path") == str(args.fixed_population)
                 and manifest.get("previous_same_window_readouts_original_path") == str(args.previous_evaluation_dir),
                 "preflight run manifest differs from explicit input paths")
        evaluator = baseline._load_evaluator(_resolve(args.evaluator), device)
        evaluator_state = {key: value.detach().cpu().clone() for key, value in evaluator.state_dict().items()}
        _verify_run_state(output_dir, source_model, references, population, generated, evaluator_state)
        return existing
    _require(checkpoint.get("task_id") == baseline.TASK_ID, "checkpoint task id changed")
    _require(checkpoint.get("model_identity") == {
        "encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400], "ols": "gaussian_s12_v1"
    }, "checkpoint model identity changed")
    _require(checkpoint.get("data_roles") == {"train": 55_000, "dev": 5_000, "official_test_reads": 0},
             "checkpoint data roles changed")
    _require(checkpoint["config"]["optics"]["selected_production_grid"] == 2048,
             "source Encoder and checkpoint grid must remain 2048")
    _require(config["evaluation"]["evaluator_role"] == "frozen_independent_confirmation_only",
             "E_D role is not frozen independent confirmation")
    _require(tuple(population["coefficients"].shape) == (300, 100)
             and tuple(population["all_rows"].shape) == (300,), "fixed TRAIN population changed")
    _require(not bool(torch.isin(population["all_rows"], train_indices).logical_not().any()),
             "fixed population includes non-TRAIN rows")
    _require(tuple(full_generated["coefficients"].shape) == (10_000, 100),
             "saved generated coefficient population changed")
    _require(population["coefficients"].abs().min().item() > 0.0
             and generated["coefficients"].abs().min().item() > 0.0,
             "zero coefficient prevents phase-preserving power redistribution")

    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    source_state = {key: value.detach().cpu().clone() for key, value in source_model.state_dict().items()}
    model = quickcheck._diagnostic_model(source_model, DECODER_GRID)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    evaluator_path = _resolve(args.evaluator)
    evaluator = baseline._load_evaluator(evaluator_path, device)
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    evaluator.eval()
    evaluator_state = {key: value.detach().cpu().clone() for key, value in evaluator.state_dict().items()}
    _require(all(not parameter.requires_grad for parameter in model.parameters()), "Decoder is not frozen")
    _require(all(not parameter.requires_grad for parameter in source_model.parameters()), "source model is not frozen")
    _require(all(not parameter.requires_grad for parameter in evaluator.parameters()), "E_D is not frozen")

    data = _scenario_inputs(population, generated)
    power_summary: list[dict[str, Any]] = []
    power_ports: list[dict[str, Any]] = []
    power_checks: dict[str, Any] = {}
    for scenario in SCENARIOS:
        for population_name, content in data.items():
            summaries, ports, checks = _scenario_power_rows(
                scenario, content["coefficients"], content["ids"], content["labels"], population_name
            )
            power_summary.extend(summaries)
            power_ports.extend(ports)
            power_checks[f"{scenario}_{population_name}"] = checks
    _require(power_checks["R50_reconstruction"]["minimum_port_power_w"] >= 1.0e-9,
             "R50 failed to place every fixed reconstruction OLS channel above 1 nW")
    _write_csv(output_dir / "ols_power_per_sample.csv", power_summary)
    _write_csv(output_dir / "ols_power_per_port.csv", power_ports)

    window = _window_preflight(model, source_model, population, generated, device,
                               float(config["optics"]["common_complex_comparison_roi_m"]))
    _require(window["passed"], "4096 vs 6144 fixed-coefficient convergence failed 1e-4")
    old_recon_state = torch.load(_resolve(args.previous_evaluation_dir) / "unit_readout_reconstruction.pt",
                                 map_location="cpu", weights_only=False)["by_state"]["start_4096"]
    original_gain = float(references["camera_gain"])
    _require(abs(float(old_recon_state["camera_gain_w_inverse"]) - original_gain) / original_gain <= 1.0e-12,
             "previous R0 readout and source checkpoint use different frozen camera gain")
    scaling = _power_scaling_check(model, population["coefficients"][:1], original_gain, device)
    controls = _negative_controls(model, population, pixels, original_gain, device)

    output_dir_manifest = {
        "task_id": TASK_ID, "source_checkpoint_original_path": str(args.checkpoint),
        "source_report_original_path": str(args.source_report),
        "generated_latents_original_path": str(args.generated_latents),
        "fixed_population_original_path": str(args.fixed_population),
        "previous_same_window_readouts_original_path": str(args.previous_evaluation_dir),
        "config_original_path": str(args.config), "data_original_path": str(args.data),
        "evaluator_original_path": str(args.evaluator),
        "source_step": 30_000, "source_encoder_grid": 2048, "decoder_grid": DECODER_GRID,
        "comparison_grid": REFERENCE_GRID, "data_roles": {"fit": 100, "confirmation": 200,
        "canonical_train": True, "official_test_reads": 0},
        "generated_indices": generated["indices"].tolist(), "generated_labels": generated["labels"].tolist(),
        "scenario_fraction_to_uniform": SCENARIOS, "readouts": list(READOUTS),
        "camera_gain_w_inverse_source": original_gain,
        "gain_scale_counterfactual_C": GAIN_SCALE,
        "decoder_state_source": "checkpoint phase parameters copied; 4096 geometry rebuilt from physical config",
        "model_update_count": 0, "started_at_unix": time.time() - (time.perf_counter() - started),
    }
    _save_run_state(output_dir, source_model, references, population, generated, evaluator_state)
    _atomic_json(output_dir / "run_manifest.json", output_dir_manifest)

    r0_data = _collect_arm(model, "R0", data, output_dir, device)
    _require(r0_data == {"reconstruction": 300, "generation": 100}, "R0 fixed readout count changed")
    r0_records, r0_gains, _ = _finalize_arm(
        "R0", data, population, pixels, labels, evaluator, evaluator_state, references,
        output_dir, device, generate_units=True,
    )
    r0_regression = _old_readout_regression(
        output_dir, _resolve(args.previous_evaluation_dir), population, generated, original_gain,
        r0_records,
    )
    _require(all(torch.equal(value, source_model.state_dict()[key].detach().cpu())
                 for key, value in source_state.items()), "source model changed during zero-update preflight")
    _require(all(torch.equal(value, model.state_dict()[key].detach().cpu())
                 for key, value in model_state.items()), "Decoder changed during zero-update preflight")
    _require(all(torch.equal(value, evaluator.state_dict()[key].detach().cpu())
                 for key, value in evaluator_state.items()), "E_D state changed during zero-update preflight")
    result = {
        "schema": "chen_lightgen_400_ols_attribution_preflight.v1", "task_id": TASK_ID,
        "status": "READY", "passed": True, "source_checkpoint_step": 30_000,
        "source_baseline_status_preserved": source_report.get("status"),
        "source_encoder_grid": 2048, "decoder_grid": DECODER_GRID, "comparison_grid": REFERENCE_GRID,
        "window_checks": window, "power_distribution_checks": power_checks,
        "R0_same_window_regression": r0_regression, "power_scaling": scaling,
        "negative_controls": controls,
        "fixed_population": {"fit_rows": population["fit_rows"].tolist(),
                             "confirm_rows": population["confirm_rows"].tolist(),
                             "all_labels": population["all_labels"].tolist(), "source": "canonical TRAIN"},
        "fixed_generated_indices": generated["indices"].tolist(),
        "camera_gains_w_inverse": r0_gains, "frozen_state_verified": True,
        "updates": 0, "official_test_reads": 0,
        "started_at_unix": time.time() - (time.perf_counter() - started),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(done_path, result)
    _append_event(output_dir / "execution_events.jsonl", {"event": "PREFLIGHT_READY", "time_unix": time.time()})
    return result


def _read_raw_population(output_dir: Path, scenario: str, population_name: str,
                         content: dict[str, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, content["ids"].numel(), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, content["ids"].numel())
        path = _raw_path(output_dir, scenario, population_name, start // BATCH_SIZE)
        _require(path.is_file(), f"missing explicit completed raw shard: {path}")
        parts.append(_load_raw_shard(path, content["ids"][start:stop], content["labels"][start:stop]))
    return parts


def _compute_gain(raw_parts: list[tuple[torch.Tensor, torch.Tensor]], content: dict[str, torch.Tensor],
                  pixels: torch.Tensor, device: torch.device) -> float:
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for start in range(0, 100, BATCH_SIZE):
        physical = raw_parts[start // BATCH_SIZE][0].to(device).double()
        rows = content["ids"][start : start + BATCH_SIZE]
        target, _ = baseline._image_batch(pixels, rows, device)
        target = target.double()
        numerator += (physical * target).sum().cpu()
        denominator += physical.square().sum().cpu()
    _require(float(denominator) > 0.0, "fit-only camera gain denominator is zero")
    gain = float(numerator / denominator)
    _require(math.isfinite(gain) and gain > 0.0, "fit-only camera gain is invalid")
    return gain


def _unit_path(output_dir: Path, scenario: str, population_name: str, shard: int) -> Path:
    return output_dir / "unit_readouts" / scenario / f"{population_name}_{shard:04d}.npz"


def _finalize_arm(
    scenario: str, data: dict[str, dict[str, torch.Tensor]], population: dict[str, Any],
    pixels: torch.Tensor, labels: torch.Tensor, evaluator: torch.nn.Module,
    evaluator_state: dict[str, torch.Tensor], references: dict[str, torch.Tensor],
    output_dir: Path, device: torch.device, *, generate_units: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], dict[str, Any]]:
    reconstruction = data["reconstruction"]
    generated = data["generation"]
    raw_recon = _read_raw_population(output_dir, scenario, "reconstruction", reconstruction)
    raw_gen = _read_raw_population(output_dir, scenario, "generation", generated)
    fit_gain = _compute_gain(raw_recon, reconstruction, pixels, device)
    gains = {"original_gain": float(references["camera_gain"]), "fit_calibrated_gain": fit_gain}
    records: dict[str, list[dict[str, Any]]] = {"reconstruction": [], "generation": []}
    generation_predictions: dict[str, list[int]] = {key: [] for key in READOUTS}
    generation_saturations: dict[str, list[float]] = {key: [] for key in READOUTS}
    evaluator.eval()
    for population_name, content, raw_parts in (
        ("reconstruction", reconstruction, raw_recon), ("generation", generated, raw_gen)
    ):
        for shard_index, start in enumerate(range(0, content["ids"].numel(), BATCH_SIZE)):
            stop = min(start + BATCH_SIZE, content["ids"].numel())
            physical = raw_parts[shard_index][0].to(device)
            target = None
            if population_name == "reconstruction":
                rows = content["ids"][start:stop]
                target, _ = baseline._image_batch(pixels, rows, device)
                target = target.to(device)
            stored: dict[str, np.ndarray] = {
                "sample_ids": content["ids"][start:stop].numpy(),
                "labels": content["labels"][start:stop].numpy(),
                "physical_w_per_pixel": physical.detach().cpu().numpy(),
                "decoder_plane_powers_w": raw_parts[shard_index][1].numpy(),
                "original_gain_w_inverse": np.asarray(gains["original_gain"], dtype=np.float64),
                "fit_calibrated_gain_w_inverse": np.asarray(gains["fit_calibrated_gain"], dtype=np.float64),
            }
            for readout_name, gain in gains.items():
                task = physical * gain
                display = task.clamp(0.0, 1.0)
                stored[f"task_intensity_unclipped_{readout_name}"] = task.detach().cpu().numpy()
                stored[f"display_intensity_clipped_{readout_name}"] = display.detach().cpu().numpy()
                saturation = ((task < 0.0) | (task > 1.0)).float().mean((1, 2, 3))
                camera_power = physical.double().sum((1, 2, 3))
                if population_name == "reconstruction":
                    view, psnr, ssim = baseline._reconstruction_batch_metrics(
                        task, target, content["ids"][start:stop], content["ids"][start:stop]
                    )
                    mse400 = (task - target).square().mean((1, 2, 3))
                    for i in range(stop - start):
                        population_role = "fit" if start + i < 100 else "confirmation"
                        records[population_name].append({
                            "scenario": scenario, "readout": readout_name,
                            "sample_id": int(content["ids"][start + i]), "label": int(content["labels"][start + i]),
                            "population": population_role, "mse400_unclipped": float(mse400[i].cpu()),
                            "psnr28_db": float(psnr[i].cpu()), "ssim28": float(ssim[i].cpu()),
                            "saturation_fraction": float(saturation[i].cpu()),
                            "camera_power_w": float(camera_power[i].cpu()),
                        })
                else:
                    view = baseline._resize_view(task)
                    with torch.no_grad():
                        logits = evaluator(view)
                    predicted = logits.argmax(1).detach().cpu().long()
                    generation_predictions[readout_name].extend(predicted.tolist())
                    generation_saturations[readout_name].extend(saturation.detach().cpu().tolist())
                    for i in range(stop - start):
                        records[population_name].append({
                            "scenario": scenario, "readout": readout_name,
                            "sample_id": int(content["ids"][start + i]), "label": int(content["labels"][start + i]),
                            "ed_prediction": int(predicted[i]), "saturation_fraction": float(saturation[i].cpu()),
                            "camera_power_w": float(camera_power[i].cpu()),
                        })
            if generate_units:
                unit_path = _unit_path(output_dir, scenario, population_name, shard_index)
                if not unit_path.exists():
                    _save_npz_new(unit_path, **stored)
                else:
                    with np.load(unit_path, allow_pickle=False) as old:
                        _require(np.array_equal(old["sample_ids"], stored["sample_ids"]),
                                 f"resumed unit readout sample keys changed: {unit_path}")
                        _require(np.allclose(old["physical_w_per_pixel"], stored["physical_w_per_pixel"],
                                             rtol=0.0, atol=0.0),
                                 f"resumed unit readout differs from raw shard: {unit_path}")
            if target is not None:
                del target
    _require(all(torch.equal(value, evaluator.state_dict()[key].detach().cpu())
                 for key, value in evaluator_state.items()), "E_D changed during frozen readout")
    summaries: dict[str, Any] = {"camera_gains_w_inverse": gains, "readouts": {}}
    recon_rows = records["reconstruction"]
    generation_rows = records["generation"]
    for readout_name in READOUTS:
        recon_readout = [row for row in recon_rows if row["readout"] == readout_name]
        generation_readout = [row for row in generation_rows if row["readout"] == readout_name]
        summaries["readouts"][readout_name] = {}
        for role in ("fit", "confirmation", "all_train"):
            rows = recon_readout if role == "all_train" else [row for row in recon_readout if row["population"] == role]
            summaries["readouts"][readout_name][role] = _metric_summary(rows)
        labels_t = torch.tensor([row["label"] for row in generation_readout])
        preds_t = torch.tensor([row["ed_prediction"] for row in generation_readout])
        ed = baseline._classification_summary(labels_t, preds_t)
        summaries["readouts"][readout_name]["generation"] = {
            "ed_macro_recall_descriptive": ed["macro_recall"],
            "ed_recall_by_class_descriptive": ed["recall_by_class"].tolist(),
            "ed_confusion_matrix_descriptive": ed["confusion_matrix"].tolist(),
            "saturation_fraction": _overall_saturation(generation_readout),
            "saturation_fraction_by_class": {
                str(label): _overall_saturation([row for row in generation_readout if row["label"] == label])
                for label in range(10)
            },
            "mean_camera_power_w": float(np.mean([row["camera_power_w"] for row in generation_readout])),
        }
    return records, gains, summaries


def _overall_saturation(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([row["saturation_fraction"] for row in rows])) if rows else 0.0


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot summarize empty reconstruction population")
    return {metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("mse400_unclipped", "psnr28_db", "ssim28", "saturation_fraction", "camera_power_w")}


def _load_unit_readout(path: Path, position: int) -> dict[str, np.ndarray]:
    shard_path = path.parent / f"{path.name}_{position // BATCH_SIZE:04d}.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        local = position % BATCH_SIZE
        loaded = {}
        for key in shard.files:
            value = shard[key]
            loaded[key] = (value[local] if value.ndim > 0 else value).copy()
        return loaded


def _save_atlases(output_dir: Path, population: dict[str, Any], generated: dict[str, torch.Tensor],
                  pixels: torch.Tensor, labels: torch.Tensor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir
    arms = ("R0", "R10", "R50")
    readouts = ("original_gain", "fit_calibrated_gain")
    cell, margin, label_width, header = 28, 4, 28, 16
    recon_rows = population["confirm_rows_by_class"][:, 0].long().tolist()
    columns = ["target"] + [f"{arm}_{gain}" for arm in arms for gain in readouts]
    image = Image.new("L", (label_width + margin + len(columns) * (cell + margin),
                             header + margin + 10 * (cell + margin)), color=255)
    draw = ImageDraw.Draw(image)
    for col, title in enumerate(columns):
        draw.text((label_width + margin + col * (cell + margin), 1), title[:14], fill=0)
    for label, row in enumerate(recon_rows):
        y = header + margin + label * (cell + margin)
        draw.text((1, y + 9), str(label), fill=0)
        target, _ = baseline._image_batch(pixels, torch.tensor([row]), torch.device("cpu"))
        target_view = F.adaptive_avg_pool2d(target, (28, 28))[0, 0]
        values = [target_view.numpy()]
        position = 100 + label * 20
        for arm in arms:
            item = _load_unit_readout(run_dir / "unit_readouts" / arm / "reconstruction", position)
            for gain in readouts:
                task = torch.from_numpy(item[f"task_intensity_unclipped_{gain}"]).float().unsqueeze(0)
                values.append(baseline._resize_view(task)[0, 0].numpy())
        for col, value in enumerate(values):
            tile = Image.fromarray((np.clip(value, 0, 1) * 255.0).round().astype(np.uint8), mode="L")
            image.paste(tile, (label_width + margin + col * (cell + margin), y))
    image.save(run_dir / "reconstruction_atlas.png")

    columns = [f"{arm}_{gain}" for arm in arms for gain in readouts]
    image = Image.new("L", (label_width + margin + len(columns) * (cell + margin),
                             header + margin + 10 * (cell + margin)), color=255)
    draw = ImageDraw.Draw(image)
    for col, title in enumerate(columns):
        draw.text((label_width + margin + col * (cell + margin), 1), title[:14], fill=0)
    for label in range(10):
        y = header + margin + label * (cell + margin)
        draw.text((1, y + 9), str(label), fill=0)
        position = label * 10
        for col, arm_gain in enumerate((f"{arm}_{gain}" for arm in arms for gain in readouts)):
            arm, gain = arm_gain.split("_", maxsplit=1)
            item = _load_unit_readout(run_dir / "unit_readouts" / arm / "generation", position)
            task = torch.from_numpy(item[f"task_intensity_unclipped_{gain}"]).float().unsqueeze(0)
            value = baseline._resize_view(task)[0, 0].numpy()
            tile = Image.fromarray((np.clip(value, 0, 1) * 255.0).round().astype(np.uint8), mode="L")
            image.paste(tile, (label_width + margin + col * (cell + margin), y))
    image.save(run_dir / "generation_atlas.png")


def _zip_unit_readouts(output_dir: Path) -> None:
    archive_path = output_dir / "unit_readouts.zip"
    expected = sorted(path.relative_to(output_dir).as_posix()
                      for path in (output_dir / "unit_readouts").rglob("*.npz"))
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            _require(archive.testzip() is None and sorted(archive.namelist()) == expected,
                     "existing unit readout archive is incomplete")
        return
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted((output_dir / "unit_readouts").rglob("*.npz")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    os.replace(temporary, archive_path)


def _decision(summaries: dict[str, Any]) -> dict[str, Any]:
    arms = {key: value["readouts"] for key, value in summaries.items()}
    r0_original = arms["R0"]["original_gain"]["confirmation"]
    r0_calibrated = arms["R0"]["fit_calibrated_gain"]["confirmation"]
    brightness_improvement = 1.0 - r0_calibrated["mse400_unclipped"] / r0_original["mse400_unclipped"]
    brightness_supported = (brightness_improvement >= 0.10
                            and r0_calibrated["ssim28"] >= r0_original["ssim28"]
                            and r0_calibrated["saturation_fraction"] <= 0.01)
    distribution: dict[str, Any] = {}
    for arm in ("R10", "R50"):
        recon = arms[arm]["fit_calibrated_gain"]["confirmation"]
        gen = arms[arm]["fit_calibrated_gain"]["generation"]
        r0_gen = arms["R0"]["fit_calibrated_gain"]["generation"]
        improvement = 1.0 - recon["mse400_unclipped"] / r0_calibrated["mse400_unclipped"]
        supported = (improvement >= 0.10 and recon["ssim28"] >= r0_calibrated["ssim28"]
                     and recon["saturation_fraction"] <= 0.01
                     and gen["ed_macro_recall_descriptive"] >= r0_gen["ed_macro_recall_descriptive"]
                     and gen["saturation_fraction"] <= 0.01)
        distribution[arm] = {"confirmation_mse_improvement_vs_calibrated_R0": improvement,
                             "confirmation_ssim_delta_vs_calibrated_R0":
                                 recon["ssim28"] - r0_calibrated["ssim28"],
                             "generation_macro_recall_delta_vs_calibrated_R0":
                                 gen["ed_macro_recall_descriptive"] - r0_gen["ed_macro_recall_descriptive"],
                             "supported": supported}
    if brightness_supported and any(row["supported"] for row in distribution.values()):
        decision = "BOTH_BRIGHTNESS_AND_CHANNEL_ALLOCATION_SUPPORTED"
    elif brightness_supported:
        decision = "BRIGHTNESS_INTERFACE_PRIORITY_SUPPORTED"
    elif any(row["supported"] for row in distribution.values()):
        decision = "PHYSICAL_ENCODER_COUPLING_COMPARISON_WORTH_PLANNING"
    else:
        decision = "FROZEN_DECODER_COUNTERFACTUALS_DID_NOT_SUPPORT_EITHER_EXPLANATION"
    return {"decision": decision,
            "brightness_calibration": {"confirmation_mse_improvement_vs_original_gain": brightness_improvement,
                                       "confirmation_ssim_delta": r0_calibrated["ssim28"] - r0_original["ssim28"],
                                       "confirmation_saturation_fraction": r0_calibrated["saturation_fraction"],
                                       "supported": brightness_supported},
            "channel_allocation": distribution}


def _write_closeout(path: Path, report: dict[str, Any]) -> None:
    decision = report["scientific_decision"]
    summaries = report["summary_by_scenario"]
    lines = [
        "# LightGen400 OLS 功率与通道分配快速归因",
        "",
        "## 结果与状态",
        "",
        f"状态：`{report['status']}`。最终判断：`{decision['decision']}`。原 30,000 步基线 `VALID_FAILED` 身份保留。",
        "",
        "确认集重建指标（全部使用 200 张固定 TRAIN confirmation）：",
        "",
        "| 光学对照 | 读出增益 | MSE400 | PSNR28 (dB) | SSIM28 | 超范围比例 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for readout in READOUTS:
            metrics = summaries[scenario]["readouts"][readout]["confirmation"]
            lines.append(
                f"| {scenario} | {readout} | {metrics['mse400_unclipped']:.6g} | "
                f"{metrics['psnr28_db']:.4f} | {metrics['ssim28']:.5f} | "
                f"{metrics['saturation_fraction']:.5%} |"
            )
    lines.extend(["", "生成读数使用冻结 E_D，仅作描述性定位：", "",
                  "| 光学对照 | 读出增益 | 宏召回 | 超范围比例 |", "|---|---|---:|---:|"])
    for scenario in SCENARIOS:
        for readout in READOUTS:
            metrics = summaries[scenario]["readouts"][readout]["generation"]
            lines.append(f"| {scenario} | {readout} | {metrics['ed_macro_recall_descriptive']:.4f} | "
                         f"{metrics['saturation_fraction']:.5%} |")
    lines.extend([
        "",
        "## 科学认识",
        "",
        f"亮度校准相对原 gain 的确认集 MSE 改善为 "
        f"{decision['brightness_calibration']['confirmation_mse_improvement_vs_original_gain']:.2%}，"
        f"SSIM 变化 {decision['brightness_calibration']['confirmation_ssim_delta']:+.5f}；"
        f"按预设条件{'获得支持' if decision['brightness_calibration']['supported'] else '未获支持'}。",
        "",
        "R10、R50 均逐样本保留 OLS 总模态功率和各端口相位；它们只改变功率分配，是数学反事实。",
    ])
    for arm, values in decision["channel_allocation"].items():
        lines.append(
            f"{arm} 相对校准 R0 的确认集 MSE 改善 {values['confirmation_mse_improvement_vs_calibrated_R0']:.2%}，"
            f"SSIM 变化 {values['confirmation_ssim_delta_vs_calibrated_R0']:+.5f}，"
            f"生成宏召回变化 {values['generation_macro_recall_delta_vs_calibrated_R0']:+.4f}；"
            f"预设分配收益条件{'通过' if values['supported'] else '未通过'}。"
        )
    lines.extend([
        "",
        "通道增亮不能说明两层 Encoder 可实现该系数变换；本轮也没有光子噪声、曝光、实测几何或硬件阈值。",
        "",
        "## 目标缺口",
        "",
        "本轮只用了固定 300 张 TRAIN 系数及 100 条已保存生成系数。尚未完成全量 DEV、完整 10,000 条生成验收、"
        "盲法人眼识读或物理 OLS 标定；因此不支持正式基线达标、硬件信噪比收益或模型采用结论。",
        "",
        "## 工程完成与验证",
        "",
        f"实际测量 3 个光学对照×(300 张重建+100 条生成)，模型更新 {report['model_updates']} 次，新潜变量抽样 "
        f"{report['new_latent_samples']} 条；Decoder、Encoder、OLS 参考和 E_D 均冻结。4096/6144 窗口、R0 历史回归、"
        "功率缩放律、错配行拒绝、内容打乱负控和读出单位检查详见 `preflight.json`。",
        "",
        f"运行耗时 {report['elapsed_seconds']:.1f} 秒；CUDA 峰值显存 "
        f"{report['peak_cuda_memory_bytes'] / (1024 ** 3):.2f} GiB。",
        "",
        "## 决策影响与下一步",
        "",
        f"当前证据指向：`{decision['decision']}`。该判断只约束冻结 Decoder、给定 OLS 系数和 4096 离散窗口。"
        "如果亮度读出获支持，建议单独规划接口标定；如果 R10/R50 获支持，建议另行设计 Encoder 可达性与任务兼容性测量。"
        "如果两类均未获支持，本结果只说明本轮固定 Decoder 反事实未解决问题，不能据此判断耦合优化无用或模型容量不足。",
        "",
        "本轮停止于该归因报告；上述建议不自动授权训练、几何修改、增益修改或模型采用。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    preflight = _read_json(output_dir / "preflight.json")
    _require(preflight.get("status") == "READY" and preflight.get("passed") is True,
             "explicit output directory has no passing READY preflight")
    if args.mode == "run":
        _require(not any((output_dir / "raw" / arm).exists() for arm in ("R10", "R50")),
                 "partial readouts exist; use explicit --mode resume")
        _require(not (output_dir / "report.json").exists(), "completed report already exists")
    device = torch.device("cuda")
    (config, checkpoint, source_model, references, population, generated, pixels, labels,
     train_indices, source_report, full_generated) = _load_inputs(args, device)
    del train_indices, source_report, full_generated
    evaluator = baseline._load_evaluator(_resolve(args.evaluator), device)
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    evaluator.eval()
    evaluator_state = {key: value.detach().cpu().clone() for key, value in evaluator.state_dict().items()}
    run_manifest = _read_json(output_dir / "run_manifest.json")
    _require(run_manifest.get("source_step") == checkpoint.get("completed_updates") == 30_000
             and run_manifest.get("source_checkpoint_original_path") == str(args.checkpoint)
             and run_manifest.get("fixed_population_original_path") == str(args.fixed_population)
             and run_manifest.get("generated_latents_original_path") == str(args.generated_latents)
             and run_manifest.get("decoder_grid") == DECODER_GRID
             and run_manifest.get("data_roles") == {"fit": 100, "confirmation": 200,
                 "canonical_train": True, "official_test_reads": 0},
             "run manifest does not match the explicitly supplied checkpoint, population, or data roles")
    _verify_run_state(output_dir, source_model, references, population, generated, evaluator_state)
    _require(run_manifest.get("source_report_original_path") == str(args.source_report)
             and run_manifest.get("config_original_path") == str(args.config)
             and run_manifest.get("data_original_path") == str(args.data)
             and run_manifest.get("evaluator_original_path") == str(args.evaluator)
             and run_manifest.get("previous_same_window_readouts_original_path") == str(args.previous_evaluation_dir),
             "run manifest differs from explicitly supplied report, config, data, evaluator, or reference paths")
    if (output_dir / "report.json").is_file():
        return _read_json(output_dir / "report.json")
    model = quickcheck._diagnostic_model(source_model, DECODER_GRID)
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_model.eval()
    model.eval()
    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    source_state = {key: value.detach().cpu().clone() for key, value in source_model.state_dict().items()}
    data = _scenario_inputs(population, generated)
    torch.cuda.reset_peak_memory_stats(device)
    count_by_arm: dict[str, Any] = {}
    for scenario in ("R10", "R50"):
        count_by_arm[scenario] = _collect_arm(model, scenario, data, output_dir, device)
    summaries: dict[str, Any] = {}
    all_reconstruction: list[dict[str, Any]] = []
    all_generation: list[dict[str, Any]] = []
    gains: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        records, arm_gains, arm_summary = _finalize_arm(
            scenario, data, population, pixels, labels, evaluator, evaluator_state,
            references, output_dir, device, generate_units=True,
        )
        summaries[scenario] = arm_summary
        all_reconstruction.extend(records["reconstruction"])
        all_generation.extend(records["generation"])
        gains[scenario] = arm_gains
    _require(all(torch.equal(value, model.state_dict()[key].detach().cpu())
                 for key, value in model_state.items()), "Decoder weights changed during forward-only measurement")
    _require(all(torch.equal(value, source_model.state_dict()[key].detach().cpu())
                 for key, value in source_state.items()), "source Encoder weights changed during measurement")
    _require(all(torch.equal(value, evaluator.state_dict()[key].detach().cpu())
                 for key, value in evaluator_state.items()), "frozen E_D changed during measurement")
    _write_csv(output_dir / "per_sample_reconstruction_metrics.csv", all_reconstruction)
    _write_csv(output_dir / "per_sample_generation_metrics.csv", all_generation)
    _save_atlases(output_dir, population, generated, pixels, labels)
    decision = _decision(summaries)
    report = {
        "schema": "chen_lightgen_400_ols_attribution_report.v1", "task_id": TASK_ID,
        "status": "VALID_COMPLETED", "scientific_decision": decision,
        "source_baseline_status": "VALID_FAILED", "source_checkpoint_step": 30_000,
        "source_encoder_grid": 2048, "decoder_grid": DECODER_GRID,
        "fixed_data_roles": {"fit": 100, "confirmation": 200, "canonical_train": True,
                             "official_test_reads": 0},
        "fixed_generation_samples": 100, "fixed_generation_indices": generated["indices"].tolist(),
        "optical_counterfactuals": {arm: {"fraction_toward_uniform": fraction,
            "total_modal_power_preserved_per_sample": True, "phase_preserved_per_port": True}
            for arm, fraction in SCENARIOS.items()},
        "readouts": READOUTS, "camera_gains_w_inverse": gains,
        "summary_by_scenario": summaries,
        "counts_by_scenario": {"R0": {"reconstruction": 300, "generation": 100}, **count_by_arm},
        "model_updates": 0, "new_latent_samples": 0,
        "hardware_noise_or_photon_budget": "NOT_MEASURED",
        "OLS_hardware_geometry": "PROJECT_GEOMETRY_UNMEASURED",
        "evaluator_role": "frozen independent descriptive E_D only",
        "elapsed_seconds": float(time.time() - run_manifest.get("started_at_unix", time.time())),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "artifacts": {"power_per_port": "ols_power_per_port.csv", "power_per_sample": "ols_power_per_sample.csv",
                       "reconstruction_metrics": "per_sample_reconstruction_metrics.csv",
                       "generation_metrics": "per_sample_generation_metrics.csv",
                       "unit_readouts": "unit_readouts.zip", "reconstruction_atlas": "reconstruction_atlas.png",
                       "generation_atlas": "generation_atlas.png", "closeout": "closeout.md"},
    }
    report["evaluator_frozen"] = True
    report["decision_checks"] = decision
    _atomic_json(output_dir / "summary.json", {"decision": decision, "summary_by_scenario": summaries})
    _zip_unit_readouts(output_dir)
    _atomic_json(output_dir / "progress.json", {"status": "MEASUREMENTS_COMPLETE", "updated_at_unix": time.time()})
    _write_closeout(output_dir / "closeout.md", report)
    _atomic_json(output_dir / "report.json", report)
    _append_event(output_dir / "execution_events.jsonl", {"event": "MEASUREMENTS_COMPLETE", "time_unix": time.time()})
    return report


def _prepare(args: argparse.Namespace) -> argparse.Namespace:
    _require(args.mode in ("preflight", "run", "resume"), "--mode is required")
    required = ("checkpoint", "source_report", "generated_latents", "fixed_population",
                "previous_evaluation_dir", "config", "data", "evaluator", "output_dir")
    for name in required:
        _require(getattr(args, name) is not None, f"explicit --{name.replace('_', '-')} is required")
    _require(args.batch_size == BATCH_SIZE, "scientific readout batch is fixed at 10")
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--generated-latents", type=Path, required=True)
    parser.add_argument("--fixed-population", type=Path, required=True)
    parser.add_argument("--previous-evaluation-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return _prepare(parser.parse_args())


def main() -> None:
    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "preflight":
            result = _preflight(args)
        else:
            result = _run_measurement(args)
        print(json.dumps({"status": result.get("status"), "mode": args.mode,
                          "output_dir": str(output_dir)}, ensure_ascii=False))
    except Exception as error:
        _append_event(output_dir / "execution_events.jsonl", {
            "event": "EXECUTION_ERROR", "mode": args.mode, "error_type": type(error).__name__,
            "error": str(error), "completed_scientific_updates": 0, "time_unix": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
