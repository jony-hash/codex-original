"""Attribute fixed-cohort CMOS visibility changes across frozen Gaussian endpoints."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import html
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
from ldm_project.v2.application.research.lightgen.image_readout import resize_view


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_CMOS_VISIBILITY_ATTRIBUTION_V1"
EVIDENCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_STRUCTURAL_CONTINUATION_AB_V1"
EVIDENCE_SCHEMA = "lightgen_small_gaussian_structural_continuation_checkpoint.v1"
COHORT_SIZE = 10_000
SAMPLES_PER_CLASS = 1_000
BATCH_SIZE = 10
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 2026102101
GAIN_MULTIPLIERS = (0.85, 1.0, 1.15)
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
RESULT_BUDGET_BYTES = 512 * 1024 * 1024
BRANCHES = {"A": "control", "B": "treatment"}
CELL_INPUT = {"AA": "A", "BA": "A", "AB": "B", "BB": "B"}
CELL_DECODER = {"AA": "A", "BA": "B", "AB": "A", "BB": "B"}


def _require(condition: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error_type(message)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "continuation config must be a mapping")
    _require(value.get("schema") == "lightgen_small_gaussian_structural_continuation_ab.v1"
             and value.get("task_id") == EVIDENCE_TASK_ID,
             "explicit configuration is not the approved Gaussian A/B continuation")
    _require(value["source"]["checkpoint_step"] == 30_000
             and value["training"]["source_completed_updates"] == 30_000
             and value["training"]["continuation_updates"] == 5_000
             and value["training"]["seed"] == 2026101501,
             "source endpoint or continuation identity changed")
    _require(value["optics"]["small"]["ols_basis"] == "gaussian_s12_v1"
             and value["optics"]["small"]["production_propagation_grid"] == 2048
             and value["optics"]["small"]["phase_parameters"] == 40_000
             and value["optics"]["small"]["geometry_status"] == "PROJECT_GEOMETRY_UNMEASURED",
             "Gaussian model geometry or production grid changed")
    _require(value["evaluation"]["evaluator_role"] == "frozen_confirmation_only"
             and value["evaluation"]["generated_samples_per_class"] == SAMPLES_PER_CLASS
             and value["evaluation"]["classifier_sample_selection"] is False
             and value["evaluation"]["max_proposals_per_class"] == 1_000_000
             and value["evaluation"]["power_relative_covariance_ridge"] == 1.0e-6
             and value["evaluation"]["paired_bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and value["evaluation"]["paired_bootstrap_seed"] == 2026102004
             and value["output"]["formal_result_budget_bytes"] == RESULT_BUDGET_BYTES,
             "fixed cohort, source evaluator role or continuation output budget changed")
    return value


def _expected_model_identity(config: dict[str, Any]) -> dict[str, Any]:
    identity = config["source"]["model_identity"]
    _require(identity == {
        "encoder": [2, 100, 100],
        "ols": "gaussian_s12_v1",
        "ols_geometry": {
            "port_pitch_m": 20.0e-6,
            "field_1e_radius_m": 3.5e-6,
            "transport": "identity",
        },
        "decoder": [2, 100, 100],
    }, "configured Gaussian endpoint identity changed")
    return identity


def _load_endpoint(evidence_dir: Path, branch: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    branch_dir = evidence_dir / BRANCHES[branch]
    checkpoint_path = branch_dir / "checkpoint_step_005000.pt"
    cohort_path = branch_dir / "generated_cohort.pt"
    summary_path = branch_dir / "endpoint_summary.json"
    for path in (checkpoint_path, cohort_path, summary_path):
        _require(path.is_file(), f"required fixed endpoint evidence is missing: {path}", FileNotFoundError)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cohort = torch.load(cohort_path, map_location="cpu", weights_only=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(isinstance(checkpoint, dict) and isinstance(cohort, dict) and isinstance(summary, dict),
             f"{branch} endpoint payloads must be mappings", TypeError)
    expected_model = _expected_model_identity(config)
    _require(checkpoint.get("schema") == EVIDENCE_SCHEMA
             and checkpoint.get("task_id") == EVIDENCE_TASK_ID
             and checkpoint.get("branch") == BRANCHES[branch]
             and checkpoint.get("source_checkpoint_step") == 30_000
             and checkpoint.get("completed_updates") == 5_000
             and checkpoint.get("completed_global_step") == 35_000
             and checkpoint.get("seed") == 2026101501
             and checkpoint.get("propagation_grid_size") == 2048,
             f"{branch} checkpoint task, branch, step, seed or grid differs")
    _require(checkpoint.get("model_identity") == expected_model,
             f"{branch} checkpoint is not the configured Gaussian model")
    _require(checkpoint.get("config_identity") == config,
             f"{branch} checkpoint configuration differs from the explicit config")
    _require(checkpoint.get("evaluator_role") == "frozen_confirmation_only",
             f"{branch} checkpoint evaluator role changed")
    _require(isinstance(checkpoint.get("model"), dict)
             and isinstance(checkpoint.get("optimizer"), dict)
             and isinstance(checkpoint.get("schedule"), torch.Tensor)
             and tuple(checkpoint["schedule"].shape) == (5_000, 10),
             f"{branch} endpoint is missing model, optimizer provenance or fixed schedule")
    _require(summary.get("branch") == BRANCHES[branch]
             and summary.get("global_step") == 35_000
             and summary.get("measurement_complete") is True
             and summary.get("status") == "VALID_FAILED"
             and summary.get("generation", {}).get("population") == COHORT_SIZE,
             f"{branch} prior endpoint summary is incomplete or has a different status")
    _require(cohort.get("task_id") == EVIDENCE_TASK_ID
             and cohort.get("branch") == BRANCHES[branch]
             and cohort.get("run_id") == checkpoint.get("run_id")
             and cohort.get("global_step") == 35_000,
             f"{branch} generation cohort identity differs from its endpoint")
    _validate_cohort(cohort, branch)
    for name in ("scale", "power_reference", "camera_gain"):
        tensor = checkpoint.get(name)
        _require(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all().item(),
                 f"{branch} frozen {name} is missing or nonfinite")
    _require(tuple(checkpoint["scale"].shape) == (100,)
             and tuple(checkpoint["camera_gain"].shape) == ()
             and float(checkpoint["camera_gain"]) > 0.0,
             f"{branch} frozen scale or global gain has an unexpected shape/value")
    weights = checkpoint.get("weights")
    _require(isinstance(weights, dict) and set(weights) == {"dist", "rec", "op"}
             and all(math.isfinite(float(value)) for value in weights.values()),
             f"{branch} original BOGT weights are incomplete")
    structural_weight = checkpoint.get("structural_weight")
    if branch == "A":
        _require(structural_weight is None, "A control endpoint unexpectedly contains an SSIM weight")
    else:
        _require(isinstance(structural_weight, (int, float)) and math.isfinite(structural_weight)
                 and structural_weight > 0.0,
                 "B endpoint has an invalid frozen SSIM calibration")
    return checkpoint, cohort, summary


def _validate_cohort(cohort: dict[str, Any], branch: str) -> None:
    expected = {
        "images": ((COHORT_SIZE, 1, 28, 28), False),
        "requested_labels": ((COHORT_SIZE,), False),
        "predicted_labels": ((COHORT_SIZE,), False),
        "proposal_ids": ((COHORT_SIZE,), False),
        "ols_coefficients": ((COHORT_SIZE, 100), True),
        "proxy_clipping_fraction": ((COHORT_SIZE,), False),
    }
    for name, (shape, complex_expected) in expected.items():
        tensor = cohort.get(name)
        _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape,
                 f"{branch} cohort field {name} has the wrong shape")
        _require(tensor.is_complex() is complex_expected,
                 f"{branch} cohort field {name} has the wrong numeric kind")
        finite = torch.isfinite(tensor.real).all() and torch.isfinite(tensor.imag).all() if tensor.is_complex() else torch.isfinite(tensor).all()
        _require(bool(finite), f"{branch} cohort field {name} contains nonfinite values")
    labels = cohort["requested_labels"].long()
    proposals = cohort["proposal_ids"].long()
    _require(torch.equal(labels, torch.arange(10).repeat_interleave(SAMPLES_PER_CLASS)),
             f"{branch} cohort labels are missing, duplicated or not class-major")
    expected_proposals = torch.arange(SAMPLES_PER_CLASS).repeat(10)
    _require(torch.equal(proposals, expected_proposals),
             f"{branch} cohort proposal IDs are incomplete, duplicated or not class-major")
    _require(cohort["predicted_labels"].long().min() >= 0
             and cohort["predicted_labels"].long().max() < 10,
             f"{branch} saved predictions are outside the ten classes")
    _require(cohort["ols_coefficients"].dtype == torch.complex64,
             f"{branch} proposal coefficients must retain the saved complex64 representation")
    _require(cohort["images"].dtype == torch.float32,
             f"{branch} saved production views must be float32")
    _require((cohort["images"] >= 0).all().item() and (cohort["images"] <= 1).all().item()
             and (cohort["proxy_clipping_fraction"] >= 0).all().item()
             and (cohort["proxy_clipping_fraction"] <= 1).all().item(),
             f"{branch} saved view or clipping fractions fall outside their declared ranges")


def _validate_pairing(inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> None:
    a_ck, a_cohort, _ = inputs["A"]
    b_ck, b_cohort, _ = inputs["B"]
    _require(a_ck["source_run_id"] == b_ck["source_run_id"]
             and a_ck["source_checkpoint_step"] == b_ck["source_checkpoint_step"] == 30_000,
             "A/B checkpoints do not share the fixed 30,000-step source endpoint")
    _require(a_ck["config_identity"] == b_ck["config_identity"]
             and a_ck["model_identity"] == b_ck["model_identity"],
             "A/B endpoint configuration or model identity differs")
    _require(a_ck["structural_weight"] is None and b_ck["structural_weight"] > 0.0,
             "A/B loss branch identities are invalid")
    for name in ("scale", "power_reference", "camera_gain", "schedule"):
        _require(torch.equal(a_ck[name], b_ck[name]), f"A/B frozen {name} or sample schedule differs")
    _require(a_ck["weights"] == b_ck["weights"], "A/B original BOGT loss weights differ")
    for field in ("requested_labels", "proposal_ids"):
        _require(torch.equal(a_cohort[field], b_cohort[field]),
                 f"A/B proposal {field} are not paired")


def _gain_key(cell: str, multiplier: float) -> str:
    return f"{cell}_g{multiplier:.2f}".replace(".", "p")


def _condition_ids(cell: str) -> list[str]:
    if cell in {"AA", "BB"}:
        return [_gain_key(cell, gain) for gain in GAIN_MULTIPLIERS]
    return [_gain_key(cell, 1.0)]


def _production_view(raw_w_per_pixel: torch.Tensor, gain: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    _require(raw_w_per_pixel.ndim == 4 and tuple(raw_w_per_pixel.shape[1:]) == (1, 200, 200),
             "production camera output must be [B,1,200,200]")
    scaled = raw_w_per_pixel * (gain.to(device=raw_w_per_pixel.device, dtype=torch.float32) * multiplier)
    return resize_view(scaled)


def _readout_statistics(raw: torch.Tensor, gain: torch.Tensor, multiplier: float) -> dict[str, torch.Tensor]:
    scaled = raw * (gain.to(device=raw.device, dtype=torch.float32) * multiplier)
    mask = scaled > 1.0
    total = raw.double().sum((1, 2, 3))
    scaled_total = scaled.double().sum((1, 2, 3))
    clip_raw = raw.double().masked_fill(~mask, 0.0).sum((1, 2, 3))
    excess = (scaled.double() - 1.0).clamp_min(0.0).sum((1, 2, 3))
    nan = torch.full_like(total, float("nan"))
    return {
        "view_mean": resize_view(scaled).double().mean((1, 2, 3)),
        "camera_total_power_w": total,
        "peak_w_per_pixel": raw.double().amax((1, 2, 3)),
        "scaled_peak": scaled.double().amax((1, 2, 3)),
        "clipping_area_fraction": mask.double().mean((1, 2, 3)),
        "clipping_region_power_share": torch.where(total > 0, clip_raw / total, nan),
        "excess_removed_power_share": torch.where(scaled_total > 0, excess / scaled_total, nan),
    }


def _identity_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _require(tuple(logits.shape) == (labels.numel(), 10), "E_D logits must be [B,10]")
    target = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    return target - competitors.max(1).values


def _assert_native_replay(cell: str, condition: dict[str, torch.Tensor], cohort: dict[str, Any],
                          indices: torch.Tensor, phase: str) -> dict[str, float]:
    expected_images = cohort["images"].index_select(0, indices.long())
    expected_predictions = cohort["predicted_labels"].index_select(0, indices.long())
    _require(tuple(condition["images"].shape) == tuple(expected_images.shape),
             f"{phase} {cell} saved view shape changed")
    _require(torch.allclose(condition["images"], expected_images, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
             f"{phase} {cell} generated view differs from saved endpoint output")
    _require(torch.equal(condition["predicted_labels"], expected_predictions),
             f"{phase} {cell} prediction differs from saved endpoint output")
    delta = (condition["images"].double() - expected_images.double()).abs()
    expected_clipping = cohort["proxy_clipping_fraction"].index_select(0, indices.long()).double()
    clipping_delta = (condition["clipping_area_fraction"].double() - expected_clipping).abs()
    _require(torch.allclose(condition["clipping_area_fraction"].double(), expected_clipping,
                            rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
             f"{phase} {cell} proxy clipping area differs from saved endpoint output")
    return {"max_abs_view_difference": float(delta.max()),
            "max_abs_view_scale": float(expected_images.double().abs().max()),
            "max_abs_clipping_fraction_difference": float(clipping_delta.max())}


def _device_model(
    checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, int], Any]:
    model = base._model("small", 2048, device, config).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "frozen endpoint contains a trainable parameter")
    calls = {"encoder_samples": 0, "encoder_forward_calls": 0,
             "decoder_samples": 0, "decoder_forward_calls": 0}
    def count_encoder(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        calls["encoder_forward_calls"] += 1
        calls["encoder_samples"] += int(inputs[0].tensor.shape[0])

    def count_decoder(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        calls["decoder_forward_calls"] += 1
        calls["decoder_samples"] += int(inputs[0].tensor.shape[0])

    handle = model.runtime.encoder.register_forward_hook(
        count_encoder
    )
    decoder_handle = model.runtime.generator.register_forward_hook(count_decoder)
    return model, base._state_snapshot(model), calls, (handle, decoder_handle)


def _count_evaluator_forward(
    _module: torch.nn.Module,
    inputs: tuple[Any, ...],
    _output: Any,
    *,
    calls: dict[str, int],
) -> None:
    calls["samples"] += int(inputs[0].shape[0])
    calls["forward_calls"] += 1


def _preflight(inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], config: dict[str, Any],
               evaluator_path: Path, device: torch.device) -> dict[str, Any]:
    _require(torch.cuda.is_available() and device.type == "cuda", "this diagnostic requires the selected CUDA host")
    _require(evaluator_path.is_file(), f"explicit frozen evaluator checkpoint is missing: {evaluator_path}", FileNotFoundError)
    evaluator = base._load_evaluator(evaluator_path, device)
    evaluator_state = base._state_snapshot(evaluator)
    evaluator_calls = {"samples": 0, "forward_calls": 0}
    evaluator_hook = evaluator.register_forward_hook(
        partial(_count_evaluator_forward, calls=evaluator_calls)
    )
    models: dict[str, torch.nn.Module] = {}
    model_snapshots: dict[str, dict[str, torch.Tensor]] = {}
    calls: dict[str, dict[str, int]] = {}
    hooks: list[Any] = []
    for letter in ("A", "B"):
        model, snapshot, counter, hook = _device_model(inputs[letter][0], config, device)
        models[letter], model_snapshots[letter], calls[letter] = model, snapshot, counter
        hooks.append(hook)
    start = time.perf_counter()
    replay: dict[str, Any] = {}
    optical_seconds = 0.0
    evaluator_seconds = 0.0
    preflight_indices = torch.arange(10, dtype=torch.long) * SAMPLES_PER_CLASS
    try:
        with torch.inference_mode():
            for cell in ("AA", "BA", "AB", "BB"):
                input_letter = CELL_INPUT[cell]
                decoder_letter = CELL_DECODER[cell]
                _, cohort, _ = inputs[input_letter]
                coefficients = cohort["ols_coefficients"].index_select(0, preflight_indices)
                labels = cohort["requested_labels"].index_select(0, preflight_indices)
                started = time.perf_counter()
                output = models[decoder_letter].forward_from_ols(coefficients.to(device))
                raw = output["physical_w_per_pixel"]
                torch.cuda.synchronize(device)
                optical_seconds += time.perf_counter() - started
                multipliers = GAIN_MULTIPLIERS if cell in {"AA", "BB"} else (1.0,)
                gain = inputs[decoder_letter][0]["camera_gain"]
                result: dict[str, dict[str, torch.Tensor]] = {}
                for multiplier in multipliers:
                    started = time.perf_counter()
                    view = _production_view(raw, gain, multiplier)
                    logits = evaluator(view)
                    torch.cuda.synchronize(device)
                    evaluator_seconds += time.perf_counter() - started
                    condition = {
                        "images": view.cpu(),
                        "logits": logits.cpu(),
                        "predicted_labels": logits.argmax(1).cpu(),
                        **{key: value.cpu() for key, value in _readout_statistics(raw, gain, multiplier).items()},
                    }
                    result[_gain_key(cell, multiplier)] = condition
                native_diff: dict[str, float] | None = None
                if cell == "AA":
                    native_diff = _assert_native_replay(
                        cell, result[_gain_key(cell, 1.0)], cohort, preflight_indices, "preflight"
                    )
                elif cell == "BB":
                    native_diff = _assert_native_replay(
                        cell, result[_gain_key(cell, 1.0)], cohort, preflight_indices, "preflight"
                    )
                replay[cell] = {
                    "input_source": input_letter,
                    "decoder": decoder_letter,
                    "class_ids": labels.tolist(),
                    "proposal_ids": cohort["proposal_ids"].index_select(0, preflight_indices).tolist(),
                    "native_saved_view_match": native_diff,
                    "multipliers": list(multipliers),
                }
    finally:
        for model_hooks in hooks:
            for hook in model_hooks:
                hook.remove()
        evaluator_hook.remove()
    for letter in ("A", "B"):
        _require(base._assert_close(model_snapshots[letter], base._state_snapshot(models[letter]),
                                    f"{letter}_frozen_model_after_preflight") == 0.0,
                 f"{letter} endpoint changed during preflight")
        _require(calls[letter]["encoder_samples"] == 0,
                 f"Encoder {letter} was called during Decoder-only preflight")
    _require(sum(value["decoder_samples"] for value in calls.values()) == 40
             and sum(value["decoder_forward_calls"] for value in calls.values()) == 4,
             "preflight did not execute the four fixed decoder combinations on ten samples each")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), "evaluator_after_preflight") == 0.0,
             "frozen E_D changed during preflight")
    _require(evaluator_calls["samples"] == 80 and evaluator_calls["forward_calls"] == 8,
             f"preflight E_D call/sample counts changed: {evaluator_calls}")
    _require(all(not parameter.requires_grad for parameter in evaluator.parameters()), "E_D was unfrozen")
    return {
        "status": "PASSED",
        "device": torch.cuda.get_device_name(device),
        "cuda_available": True,
        "cells": replay,
        "encoder_calls": calls,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "gradient_computation": False,
        "generation_sampling": False,
        "evaluator_samples": evaluator_calls["samples"],
        "evaluator_forward_calls": evaluator_calls["forward_calls"],
        "official_test_reads": 0,
        "optical_seconds": optical_seconds,
        "evaluator_seconds": evaluator_seconds,
        "wall_seconds": time.perf_counter() - start,
        "replay_tolerance": {"rtol": REPLAY_RTOL, "atol": REPLAY_ATOL},
    }


def _expected_block_identity(cell: str, label: int,
                             inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    input_letter = CELL_INPUT[cell]
    decoder_letter = CELL_DECODER[cell]
    return {
        "schema": "lightgen_small_cmos_visibility_block.v1",
        "task_id": TASK_ID,
        "cell": cell,
        "input_source": input_letter,
        "decoder": decoder_letter,
        "class_id": label,
        "sample_count": SAMPLES_PER_CLASS,
        "input_run_id": inputs[input_letter][0]["run_id"],
        "decoder_run_id": inputs[decoder_letter][0]["run_id"],
        "conditions": _condition_ids(cell),
    }


def _endpoint_identities(inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    return {
        letter: {
            "branch": inputs[letter][0]["branch"],
            "run_id": inputs[letter][0]["run_id"],
            "global_step": inputs[letter][0]["completed_global_step"],
            "model_identity": inputs[letter][0]["model_identity"],
        }
        for letter in ("A", "B")
    }


def _validate_block(payload: Any, expected: dict[str, Any], expected_labels: torch.Tensor,
                    expected_proposals: torch.Tensor) -> dict[str, Any]:
    _require(isinstance(payload, dict) and payload.get("identity") == expected,
             "saved result block identity does not match the requested decoder/input/class")
    labels = payload.get("requested_labels")
    proposals = payload.get("proposal_ids")
    _require(isinstance(labels, torch.Tensor) and torch.equal(labels, expected_labels),
             "saved result block has missing, duplicate or wrong class rows")
    _require(isinstance(proposals, torch.Tensor) and torch.equal(proposals, expected_proposals),
             "saved result block proposal IDs are missing, duplicated or misaligned")
    variants = payload.get("conditions")
    _require(isinstance(variants, dict) and set(variants) == set(expected["conditions"]),
             "saved result block is missing a readout condition")
    required_shapes = {
        "images": (SAMPLES_PER_CLASS, 1, 28, 28),
        "logits": (SAMPLES_PER_CLASS, 10),
        "predicted_labels": (SAMPLES_PER_CLASS,),
        "identity_margin": (SAMPLES_PER_CLASS,),
    }
    stats = ("view_mean", "camera_total_power_w", "peak_w_per_pixel", "scaled_peak",
             "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share")
    for name, condition in variants.items():
        _require(isinstance(condition, dict), f"saved condition {name} is malformed")
        for field, shape in required_shapes.items():
            tensor = condition.get(field)
            _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape
                     and torch.isfinite(tensor).all().item(),
                     f"saved condition {name} field {field} is missing or invalid")
        _require(condition["predicted_labels"].min() >= 0 and condition["predicted_labels"].max() < 10,
                 f"saved condition {name} prediction is out of range")
        for field in stats:
            tensor = condition.get(field)
            _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (SAMPLES_PER_CLASS,),
                     f"saved condition {name} statistic {field} is missing")
            if field in {"clipping_region_power_share", "excess_removed_power_share"}:
                positive_power = condition["camera_total_power_w"] > 0
                _require(not torch.isinf(tensor).any().item()
                         and torch.isfinite(tensor[positive_power]).all().item()
                         and torch.isnan(tensor[~positive_power]).all().item(),
                         f"saved condition {name} zero-power ratio handling is invalid")
            else:
                _require(torch.isfinite(tensor).all().item(),
                         f"saved condition {name} statistic {field} is nonfinite")
        _require((condition["camera_total_power_w"] >= 0).all().item()
                 and (condition["peak_w_per_pixel"] >= 0).all().item()
                 and (condition["scaled_peak"] >= 0).all().item()
                 and (condition["view_mean"] >= 0).all().item()
                 and (condition["view_mean"] <= 1).all().item()
                 and (condition["clipping_area_fraction"] >= 0).all().item()
                 and (condition["clipping_area_fraction"] <= 1).all().item(),
                 f"saved condition {name} camera power statistic is negative")
    native = payload.get("native_raw_examples")
    _require(isinstance(native, torch.Tensor) and tuple(native.shape) == (10, 1, 200, 200)
             and torch.isfinite(native).all().item() and (native >= 0).all().item(),
             "saved native-resolution examples are missing or invalid")
    if expected["cell"] in {"AA", "BB"}:
        replay = payload.get("native_replay_check")
        _require(isinstance(replay, dict) and replay.get("passed") is True
                 and math.isfinite(float(replay.get("max_abs_view_difference", float("nan"))))
                 and math.isfinite(float(replay.get("max_abs_view_scale", float("nan"))))
                 and math.isfinite(float(replay.get("max_abs_clipping_fraction_difference", float("nan")))),
                 "native original-combination replay evidence is missing")
    return payload


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temp)
        with temp.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _run_block(cell: str, label: int, inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
               models: dict[str, torch.nn.Module], evaluator: torch.nn.Module, device: torch.device,
               output_dir: Path, resume: bool) -> tuple[dict[str, Any], dict[str, float]]:
    input_letter = CELL_INPUT[cell]
    decoder_letter = CELL_DECODER[cell]
    _, cohort, _ = inputs[input_letter]
    decoder_checkpoint = inputs[decoder_letter][0]
    offset = label * SAMPLES_PER_CLASS
    expected_labels = cohort["requested_labels"][offset : offset + SAMPLES_PER_CLASS]
    expected_proposals = cohort["proposal_ids"][offset : offset + SAMPLES_PER_CLASS]
    identity = _expected_block_identity(cell, label, inputs)
    path = output_dir / "blocks" / cell / f"class_{label:02d}.pt"
    if path.exists():
        _require(resume, f"result block already exists; pass --resume to validate and consume it: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload = _validate_block(payload, identity, expected_labels, expected_proposals)
        if cell in {"AA", "BB"}:
            indices = torch.arange(offset, offset + SAMPLES_PER_CLASS, dtype=torch.long)
            replay = _assert_native_replay(cell, payload["conditions"][_gain_key(cell, 1.0)], cohort,
                                           indices, "resume")
            _require(torch.equal(payload["conditions"][_gain_key(cell, 1.0)]["predicted_labels"],
                                 cohort["predicted_labels"][offset : offset + SAMPLES_PER_CLASS]),
                     f"resumed {cell} predictions differ from the saved endpoint")
            _require(math.isclose(replay["max_abs_view_difference"],
                                  float(payload["native_replay_check"]["max_abs_view_difference"]),
                                  rel_tol=0.0, abs_tol=1.0e-12),
                     f"resumed {cell} replay error differs from the saved measurement")
            _require(math.isclose(replay["max_abs_clipping_fraction_difference"],
                                  float(payload["native_replay_check"]["max_abs_clipping_fraction_difference"]),
                                  rel_tol=0.0, abs_tol=1.0e-12),
                     f"resumed {cell} clipping replay error differs from the saved measurement")
        timing = payload.get("timing_seconds", {})
        _require(isinstance(timing, dict)
                 and math.isfinite(float(timing.get("optical_seconds", float("nan"))))
                 and math.isfinite(float(timing.get("evaluator_seconds", float("nan")))),
                 "resumed block is missing measured timing evidence")
        return payload, {"optical_seconds": float(timing["optical_seconds"]),
                         "evaluator_seconds": float(timing["evaluator_seconds"]), "reused": 1.0}
    results_by_condition: dict[str, list[dict[str, torch.Tensor]]] = {key: [] for key in identity["conditions"]}
    raw_examples: torch.Tensor | None = None
    optical_seconds = 0.0
    evaluator_seconds = 0.0
    for start in range(offset, offset + SAMPLES_PER_CLASS, BATCH_SIZE):
        coefficients = cohort["ols_coefficients"][start : start + BATCH_SIZE]
        labels = cohort["requested_labels"][start : start + BATCH_SIZE]
        started = time.perf_counter()
        output = models[decoder_letter].forward_from_ols(coefficients.to(device))
        raw = output["physical_w_per_pixel"]
        torch.cuda.synchronize(device)
        optical_seconds += time.perf_counter() - started
        if raw_examples is None:
            raw_examples = raw.detach().cpu()
        multipliers = GAIN_MULTIPLIERS if cell in {"AA", "BB"} else (1.0,)
        for multiplier in multipliers:
            started = time.perf_counter()
            scaled_view = _production_view(raw, decoder_checkpoint["camera_gain"], multiplier)
            logits = evaluator(scaled_view)
            torch.cuda.synchronize(device)
            evaluator_seconds += time.perf_counter() - started
            label_device = labels.to(device=device, dtype=torch.long)
            stats = _readout_statistics(raw, decoder_checkpoint["camera_gain"], multiplier)
            condition = {
                "images": scaled_view.detach().cpu(),
                "logits": logits.detach().cpu(),
                "predicted_labels": logits.argmax(1).detach().cpu(),
                "identity_margin": _identity_margin(logits, label_device).detach().cpu(),
                **{key: value.detach().cpu() for key, value in stats.items()},
            }
            results_by_condition[_gain_key(cell, multiplier)].append(condition)
    _require(raw_examples is not None, "fixed class block produced no optical output")
    _require(raw_examples is not None and tuple(raw_examples.shape) == (BATCH_SIZE, 1, 200, 200),
             "first ten native camera examples were not retained")
    conditions: dict[str, dict[str, torch.Tensor]] = {}
    for condition_id, pieces in results_by_condition.items():
        conditions[condition_id] = {
            key: torch.cat([piece[key] for piece in pieces], dim=0)
            for key in pieces[0]
        }
    payload = {
        "identity": identity,
        "requested_labels": expected_labels.clone(),
        "proposal_ids": expected_proposals.clone(),
        "conditions": conditions,
        "native_raw_examples": raw_examples,
        "timing_seconds": {
            "optical_seconds": optical_seconds,
            "evaluator_seconds": evaluator_seconds,
        },
    }
    if cell in {"AA", "BB"}:
        indices = torch.arange(offset, offset + SAMPLES_PER_CLASS, dtype=torch.long)
        replay = _assert_native_replay(cell, conditions[_gain_key(cell, 1.0)], cohort, indices, "full")
        payload["native_replay_check"] = {"passed": True, **replay}
    payload = _validate_block(payload, identity, expected_labels, expected_proposals)
    if cell in {"AA", "BB"}:
        _require(torch.equal(conditions[_gain_key(cell, 1.0)]["predicted_labels"],
                             cohort["predicted_labels"][offset : offset + SAMPLES_PER_CLASS]),
                 f"full {cell} predictions differ from the saved endpoint")
    _atomic_torch_save(path, payload)
    readback = torch.load(path, map_location="cpu", weights_only=False)
    readback = _validate_block(readback, identity, expected_labels, expected_proposals)
    return readback, {**payload["timing_seconds"], "reused": 0.0}


def _condition_arrays(output_dir: Path, cell: str, condition_id: str,
                      inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, torch.Tensor]:
    pieces: dict[str, list[torch.Tensor]] = {}
    for label in range(10):
        path = output_dir / "blocks" / cell / f"class_{label:02d}.pt"
        expected = _expected_block_identity(cell, label, inputs)
        block = _validate_block(
            torch.load(path, map_location="cpu", weights_only=False), expected,
            torch.full((SAMPLES_PER_CLASS,), label, dtype=torch.long),
            torch.arange(SAMPLES_PER_CLASS, dtype=torch.long),
        )
        condition = block["conditions"].get(condition_id)
        _require(isinstance(condition, dict), f"complete block is missing {condition_id}")
        for key, value in condition.items():
            pieces.setdefault(key, []).append(value)
    return {key: torch.cat(parts, dim=0) for key, parts in pieces.items()}


def _summarize_condition(cell: str, condition_id: str, values: dict[str, torch.Tensor],
                         labels: torch.Tensor) -> dict[str, Any]:
    predictions = values["predicted_labels"].long()
    correct = (predictions == labels).reshape(10, SAMPLES_PER_CLASS)
    result: dict[str, Any] = {
        "cell": cell,
        "condition_id": condition_id,
        "samples": int(labels.numel()),
        "recall_by_class": correct.double().mean(1).tolist(),
        "macro_recall": float(correct.double().mean(1).mean()),
        "identity_margin_mean": float(values["identity_margin"].double().mean()),
        "identity_margin_by_class": values["identity_margin"].double().reshape(10, SAMPLES_PER_CLASS).mean(1).tolist(),
        "confusion_matrix": torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).tolist(),
    }
    for key in ("view_mean", "camera_total_power_w", "peak_w_per_pixel", "scaled_peak",
                "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share"):
        matrix = values[key].double().reshape(10, SAMPLES_PER_CLASS)
        finite = torch.isfinite(matrix)
        per_class = [float(matrix[i][finite[i]].mean()) if finite[i].any() else None for i in range(10)]
        class_means = [value for value in per_class if value is not None]
        result[key + "_by_class"] = per_class
        result[key + "_equal_class_mean"] = float(np.mean(class_means)) if class_means else None
    return result


def _paired_bootstrap(values: dict[str, dict[str, torch.Tensor]], labels: torch.Tensor) -> dict[str, Any]:
    contrasts = {
        "decoder_B_minus_A_fixed_zA": ("BA_g1p00", "AA_g1p00"),
        "decoder_B_minus_A_fixed_zB": ("BB_g1p00", "AB_g1p00"),
        "decoder_interaction": ("BB_g1p00", "AB_g1p00", "BA_g1p00", "AA_g1p00"),
        "input_B_minus_A_fixed_decoderA": ("AB_g1p00", "AA_g1p00"),
        "input_B_minus_A_fixed_decoderB": ("BB_g1p00", "BA_g1p00"),
        "gain_AA_0p85_minus_1p00": ("AA_g0p85", "AA_g1p00"),
        "gain_AA_1p15_minus_1p00": ("AA_g1p15", "AA_g1p00"),
        "gain_BB_0p85_minus_1p00": ("BB_g0p85", "BB_g1p00"),
        "gain_BB_1p15_minus_1p00": ("BB_g1p15", "BB_g1p00"),
    }
    names = set(values)
    _require(all(all(name in names for name in contrast) for contrast in contrasts.values()),
             "paired bootstrap is missing a required fixed condition")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    correct = {name: ((entry["predicted_labels"].long() == labels).reshape(10, SAMPLES_PER_CLASS).numpy())
               for name, entry in values.items()}
    margins = {name: entry["identity_margin"].double().reshape(10, SAMPLES_PER_CLASS).numpy()
               for name, entry in values.items()}
    clipping = {name: entry["clipping_area_fraction"].double().reshape(10, SAMPLES_PER_CLASS).numpy()
                for name, entry in values.items()}
    sampled: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in ("macro_recall", "class9_recall", "macro_margin", "class9_margin", "macro_clipping_area")}
        for name in contrasts
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, SAMPLES_PER_CLASS, size=(10, SAMPLES_PER_CLASS))
        recall_by_condition: dict[str, np.ndarray] = {}
        margin_by_condition: dict[str, np.ndarray] = {}
        clip_by_condition: dict[str, np.ndarray] = {}
        for name in names:
            recall_by_condition[name] = np.take_along_axis(correct[name], indices, axis=1).mean(axis=1)
            margin_by_condition[name] = np.take_along_axis(margins[name], indices, axis=1).mean(axis=1)
            clip_by_condition[name] = np.take_along_axis(clipping[name], indices, axis=1).mean(axis=1)
        for contrast_name, operands in contrasts.items():
            if len(operands) == 2:
                left, right = operands
                deltas = {
                    "macro_recall": float((recall_by_condition[left] - recall_by_condition[right]).mean()),
                    "class9_recall": float(recall_by_condition[left][9] - recall_by_condition[right][9]),
                    "macro_margin": float((margin_by_condition[left] - margin_by_condition[right]).mean()),
                    "class9_margin": float(margin_by_condition[left][9] - margin_by_condition[right][9]),
                    "macro_clipping_area": float((clip_by_condition[left] - clip_by_condition[right]).mean()),
                }
            else:
                bb, ab, ba, aa = operands
                deltas = {
                    "macro_recall": float(((recall_by_condition[bb] - recall_by_condition[ab]) -
                                            (recall_by_condition[ba] - recall_by_condition[aa])).mean()),
                    "class9_recall": float((recall_by_condition[bb][9] - recall_by_condition[ab][9]) -
                                            (recall_by_condition[ba][9] - recall_by_condition[aa][9])),
                    "macro_margin": float(((margin_by_condition[bb] - margin_by_condition[ab]) -
                                            (margin_by_condition[ba] - margin_by_condition[aa])).mean()),
                    "class9_margin": float((margin_by_condition[bb][9] - margin_by_condition[ab][9]) -
                                            (margin_by_condition[ba][9] - margin_by_condition[aa][9])),
                    "macro_clipping_area": float(((clip_by_condition[bb] - clip_by_condition[ab]) -
                                                  (clip_by_condition[ba] - clip_by_condition[aa])).mean()),
                }
            for metric, delta in deltas.items():
                sampled[contrast_name][metric].append(delta)
    result: dict[str, Any] = {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                              "method": "equal-class paired bootstrap; one shared class/proposal resample for all cells and gains",
                              "contrasts": {}}
    for name, metrics in sampled.items():
        result["contrasts"][name] = {}
        for metric, samples in metrics.items():
            array = np.asarray(samples, dtype=np.float64)
            result["contrasts"][name][metric] = {
                "estimate": float(array.mean()),
                "ci_95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            }
    return result


def _correctness_transitions(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], labels: torch.Tensor,
                             left_name: str, right_name: str) -> dict[str, Any]:
    left_pred = left["predicted_labels"].long()
    right_pred = right["predicted_labels"].long()
    left_ok = left_pred == labels
    right_ok = right_pred == labels
    rows: list[dict[str, int]] = []
    for label in range(10):
        selected = labels == label
        rows.append({
            "class": label,
            "both_correct": int((left_ok & right_ok & selected).sum()),
            "left_only_correct": int((left_ok & ~right_ok & selected).sum()),
            "right_only_correct": int((~left_ok & right_ok & selected).sum()),
            "both_wrong": int((~left_ok & ~right_ok & selected).sum()),
        })
    return {
        "left": left_name,
        "right": right_name,
        "per_class": rows,
        "overall": {key: sum(row[key] for row in rows) for key in rows[0] if key != "class"},
    }


def _recommendation(summaries: dict[str, dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    contrasts = bootstrap["contrasts"]
    decoder_names = ("decoder_B_minus_A_fixed_zA", "decoder_B_minus_A_fixed_zB")
    if all(contrasts[name]["class9_recall"]["ci_95"][1] < 0.0 for name in decoder_names):
        return {
            "code": "DECODER_IDENTITY_TRADEOFF_SUPPORTED",
            "recommendation": "下一轮只做一次固定预算的身份相关笔画训练对照。",
            "basis": "B Decoder 在固定 zA 与固定 zB 时都使类别9召回下降，且两个95%区间上界均低于0。",
        }
    input_names = ("input_B_minus_A_fixed_decoderA", "input_B_minus_A_fixed_decoderB")
    if all(contrasts[name]["class9_recall"]["ci_95"][1] < 0.0 for name in input_names):
        return {
            "code": "ENCODER_PRIOR_INPUT_DISTRIBUTION_PRIORITY",
            "recommendation": "下一轮优先针对Encoder与拟合先验共同形成的输入分布做定点诊断。",
            "basis": "B输入分布在固定A、B Decoder时都使类别9召回下降；本轮不能分开归因Encoder与先验拟合。",
        }
    for gain_name in ("gain_AA_0p85_minus_1p00", "gain_AA_1p15_minus_1p00",
                      "gain_BB_0p85_minus_1p00", "gain_BB_1p15_minus_1p00"):
        gain_contrast = contrasts[gain_name]
        clipping_delta = gain_contrast["macro_clipping_area"]["estimate"]
        if (gain_contrast["macro_recall"]["ci_95"][0] > 0.0
                and gain_contrast["class9_recall"]["ci_95"][0] > 0.0
                and clipping_delta <= 0.0):
            paired_native = "AA" if gain_name.startswith("gain_AA") else "BB"
            related = "gain_AA" if paired_native == "AA" else "gain_BB"
            other_name = gain_name.replace(related, "gain_BB" if paired_native == "AA" else "gain_AA")
            other = contrasts[other_name]
            if (other["macro_recall"]["ci_95"][0] > 0.0
                    and other["class9_recall"]["ci_95"][0] > 0.0
                    and other["macro_clipping_area"]["estimate"] <= 0.0):
                return {
                    "code": "READOUT_GAIN_DIRECTION_SUPPORTED",
                    "recommendation": "另行用TRAIN确定一个唯一全局增益，再对完整确认人口做一次重构与生成复核。",
                    "basis": f"同一增益方向在AA与BB均提高宏召回和类别9召回的95%区间下界，且等类截断面积均未增加。{gain_name} 与 {other_name}。",
                }
    return {
        "code": "MIXED_OR_UNRESOLVED",
        "recommendation": "下一步只补足本报告指出的Decoder与输入分布交互或类别9身份缺陷测量。",
        "basis": "预注册规则没有得到单一且跨原组合一致的Decoder、输入分布或增益方向证据。",
    }


def _write_csv(output_dir: Path, conditions: dict[str, dict[str, Any]], labels: torch.Tensor,
               proposals: torch.Tensor, cells: dict[str, str]) -> None:
    path = output_dir / "per_image_metrics.csv"
    fields = ["condition_id", "cell", "input_source", "decoder", "class", "proposal_id", "predicted_label",
              "correct", "identity_margin", "view_mean", "camera_total_power_w", "peak_w_per_pixel",
              "scaled_peak", "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition_id, value in conditions.items():
            cell = cells[condition_id]
            input_source = CELL_INPUT[cell]
            decoder = CELL_DECODER[cell]
            for index in range(COHORT_SIZE):
                row = {
                    "condition_id": condition_id,
                    "cell": cell,
                    "input_source": input_source,
                    "decoder": decoder,
                    "class": int(labels[index]),
                    "proposal_id": int(proposals[index]),
                    "predicted_label": int(value["predicted_labels"][index]),
                    "correct": int(value["predicted_labels"][index]) == int(labels[index]),
                }
                for field in fields[8:]:
                    statistic = value[field]
                    number = float(statistic[index])
                    row[field] = number if math.isfinite(number) else ""
                writer.writerow(row)


def _write_previews(output_dir: Path, condition_values: dict[str, dict[str, Any]],
                    raw_examples: dict[str, torch.Tensor], camera_gain: float) -> list[str]:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for condition_id, value in condition_values.items():
        images = value["images"].reshape(10, SAMPLES_PER_CLASS, 28, 28)[:, :10]
        canvas = Image.new("L", (10 * 36, 10 * 36), 255)
        draw = ImageDraw.Draw(canvas)
        for label in range(10):
            for proposal in range(10):
                x, y = proposal * 36 + 4, label * 36 + 4
                patch = (images[label, proposal].clamp(0, 1).mul(255).round().byte().numpy())
                canvas.paste(Image.fromarray(patch, mode="L"), (x, y + 8))
                draw.text((x, y), f"{label}:{proposal}", fill=0)
        name = f"{condition_id}_28x28.png"
        canvas.save(preview_dir / name)
        names.append(f"previews/{name}")
    # Native raw output is independent of the gain probe; display the first ten IDs per class for all four cells.
    for cell, samples in raw_examples.items():
        canvas = Image.new("L", (10 * 208, 10 * 208), 255)
        draw = ImageDraw.Draw(canvas)
        for label in range(10):
            raw = samples[label].reshape(10, 1, 200, 200)[:, 0]
            for proposal in range(10):
                patch = (raw[proposal] * camera_gain).clamp(0, 1).mul(255).round().byte().numpy()
                x, y = proposal * 208, label * 208 + 8
                canvas.paste(Image.fromarray(patch, mode="L"), (x, y))
                draw.text((x, y - 8), f"{label}:{proposal}", fill=0)
        name = f"{cell}_native_200x200.png"
        canvas.save(preview_dir / name)
        names.append(f"previews/{name}")
    return names


def _write_html(output_dir: Path, report: dict[str, Any], preview_paths: list[str]) -> None:
    def format_number(value: Any) -> str:
        return "NULL" if value is None else f"{float(value):.4f}"

    cells = []
    for path in preview_paths:
        label = Path(path).stem.replace("_", " ")
        cells.append(f"<figure><figcaption>{html.escape(label)}</figcaption><img src='{html.escape(path)}' loading='lazy'></figure>")
    condition_rows = []
    recall_rows = []
    for name, result in report["results"]["conditions"].items():
        condition_rows.append(
            "<tr><th>" + html.escape(name) + "</th>"
            + "".join(f"<td>{format_number(result[key])}</td>" for key in
                     ("macro_recall", "identity_margin_mean", "clipping_area_fraction_equal_class_mean",
                      "clipping_region_power_share_equal_class_mean", "excess_removed_power_share_equal_class_mean",
                      "view_mean_equal_class_mean"))
            + "</tr>"
        )
        recalls = result["recall_by_class"]
        recall_rows.append(
            "<tr><th>" + html.escape(name) + "</th>" + "".join(f"<td>{value:.3f}</td>" for value in recalls) + "</tr>"
        )
    contrast_rows = []
    for name, result in report["results"]["paired_bootstrap"]["contrasts"].items():
        macro = result["macro_recall"]
        class9 = result["class9_recall"]
        margin9 = result["class9_margin"]
        contrast_rows.append(
            "<tr><th>" + html.escape(name) + "</th>"
            + f"<td>{macro['estimate']:+.4f} [{macro['ci_95'][0]:+.4f}, {macro['ci_95'][1]:+.4f}]</td>"
            + f"<td>{class9['estimate']:+.4f} [{class9['ci_95'][0]:+.4f}, {class9['ci_95'][1]:+.4f}]</td>"
            + f"<td>{margin9['estimate']:+.4f} [{margin9['ci_95'][0]:+.4f}, {margin9['ci_95'][1]:+.4f}]</td></tr>"
        )
    decision = report["decision"]
    content = [
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>Gaussian CMOS可视性归因</title>",
        "<style>body{font:15px sans-serif;margin:24px;color:#222}figure{display:inline-block;vertical-align:top;margin:8px}img{max-width:min(92vw,980px);image-rendering:pixelated;border:1px solid #aaa}pre{white-space:pre-wrap;background:#f4f4f4;padding:14px}table{border-collapse:collapse;margin:12px 0;display:block;overflow-x:auto}th,td{border:1px solid #bbb;padding:6px 9px;white-space:nowrap}th{background:#eee}.decision{background:#eef7ed;padding:14px;border-left:5px solid #4a8}</style>",
        "<h1>Gaussian小架构：冻结端点交叉重放与CMOS可视性归因</h1>",
        "<p>28×28灰度图统一使用[0,1]；原生200×200预览按冻结全局增益显示，同样使用[0,1]，不按单图归一化。生成没有逐图真值，PSNR/SSIM不适用。</p>",
        "<h2>判读说明</h2><p>AA/BB为旧端点原组合重放；BA/AB固定输入系数更换Decoder；增益0.85/1.15只作方向探针。全人口推断只适用于本固定proposal队列。</p>",
        f"<div class='decision'><h2>本轮建议：{html.escape(decision['recommendation'])}</h2><p>{html.escape(decision['basis'])}</p><p>报告状态：{html.escape(report['status'])}；系统质量Gate未重评，历史A/B仍为VALID_FAILED。</p></div>",
        "<h2>各组合可读性和截断汇总</h2><table><thead><tr><th>组合</th><th>宏召回</th><th>平均身份余量</th><th>截断面积</th><th>截断区功率份额</th><th>超范围功率份额</th><th>平均28×28强度</th></tr></thead><tbody>" + "".join(condition_rows) + "</tbody></table>",
        "<h2>各类别召回</h2><table><thead><tr><th>组合</th>" + "".join(f"<th>{label}</th>" for label in range(10)) + "</tr></thead><tbody>" + "".join(recall_rows) + "</tbody></table>",
        "<h2>配对Bootstrap：差值及95%区间</h2><p>同一次等类proposal重采样同步用于四组合及全部增益条件，2000次。</p><table><thead><tr><th>比较</th><th>宏召回差</th><th>类别9召回差</th><th>类别9身份余量差</th></tr></thead><tbody>" + "".join(contrast_rows) + "</tbody></table>",
        "<h2>机器报告</h2><details><summary>展开完整JSON</summary><pre>" + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + "</pre></details>",
        "<h2>固定样本图</h2>" + "".join(cells),
        "</html>",
    ]
    (output_dir / "report.html").write_text("\n".join(content), encoding="utf-8")


def _build_report(inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
                  condition_values: dict[str, dict[str, Any]], preflight: dict[str, Any],
                  execution: dict[str, Any], output_dir: Path, config_path: Path,
                  evaluator_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    labels = inputs["A"][1]["requested_labels"].long()
    summaries: dict[str, dict[str, Any]] = {}
    cells: dict[str, str] = {}
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for cell in ("AA", "BA", "AB", "BB"):
        for condition_id in _condition_ids(cell):
            tensors[condition_id] = condition_values[condition_id]
            summaries[condition_id] = _summarize_condition(cell, condition_id, condition_values[condition_id], labels)
            cells[condition_id] = cell
    bootstrap = _paired_bootstrap(tensors, labels)
    transition_pairs = [
        ("AA_g1p00", "BA_g1p00"), ("AB_g1p00", "BB_g1p00"),
        ("AA_g1p00", "AB_g1p00"), ("BA_g1p00", "BB_g1p00"),
        ("AA_g1p00", "AA_g0p85"), ("AA_g1p00", "AA_g1p15"),
        ("BB_g1p00", "BB_g0p85"), ("BB_g1p00", "BB_g1p15"),
    ]
    transitions = {
        f"{left}_to_{right}": _correctness_transitions(tensors[left], tensors[right], labels, left, right)
        for left, right in transition_pairs
    }
    replay_checks: dict[str, Any] = {}
    for cell in ("AA", "BB"):
        blocks = [torch.load(output_dir / "blocks" / cell / f"class_{label:02d}.pt",
                             map_location="cpu", weights_only=False) for label in range(10)]
        replay_checks[cell] = {
            "passed": all(block["native_replay_check"]["passed"] for block in blocks),
            "max_abs_view_difference": max(block["native_replay_check"]["max_abs_view_difference"] for block in blocks),
            "max_abs_view_scale": max(block["native_replay_check"]["max_abs_view_scale"] for block in blocks),
            "max_abs_clipping_fraction_difference": max(
                block["native_replay_check"]["max_abs_clipping_fraction_difference"] for block in blocks
            ),
            "tolerance": {"rtol": REPLAY_RTOL, "atol": REPLAY_ATOL},
        }
    recommendation = _recommendation(summaries, bootstrap)
    report = {
        "schema": "lightgen_small_cmos_visibility_attribution_report.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE_DIAGNOSTIC",
        "measurement_complete": True,
        "system_quality_gate": "NOT_REEVALUATED; original A/B status VALID_FAILED remains unchanged",
        "evidence_tier": "diagnostic_only",
        "scientific_question": "Do generator visibility, class-9 confusion, and proxy clipping changes follow the frozen Decoder, the saved Encoder-plus-prior coefficient population, or the global readout gain?",
        "inputs": {
            "ab_evidence_dir": str(inputs["A"][0]["evidence_dir"]),
            "config": str(config_path),
            "evaluator_checkpoint": str(evaluator_path),
            "A": {"branch": "control", "run_id": inputs["A"][0]["run_id"], "global_step": 35_000,
                  "source_run_id": inputs["A"][0]["source_run_id"], "source_step": 30_000},
            "B": {"branch": "treatment", "run_id": inputs["B"][0]["run_id"], "global_step": 35_000,
                  "source_run_id": inputs["B"][0]["source_run_id"], "source_step": 30_000,
                  "frozen_ssim_weight": float(inputs["B"][0]["structural_weight"])},
            "model_identity": inputs["A"][0]["model_identity"],
            "grid": 2048,
            "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
            "camera_gain": float(inputs["A"][0]["camera_gain"]),
            "proposal_count_per_cell": COHORT_SIZE,
            "proposal_ids_paired": True,
        },
        "preflight": preflight,
        "results": {"conditions": summaries, "paired_bootstrap": bootstrap,
                    "correctness_transitions": transitions, "native_replay": replay_checks},
        "decision": recommendation,
        "limitations": [
            "系数输入人口是各自Encoder与拟合Gaussian先验的联合产物，本轮不能分开归因这两部分。",
            "同一proposal ID在A/B中使用相同随机提案编号，但因拟合均值与协方差不同，并非相同物理潜变量。",
            "增益探针只覆盖0.85、1.00、1.15倍；不据此直接选择或采用真实相机增益。",
            "生成样本没有逐图真值，本轮不报告生成PSNR或SSIM，也不重新判定整套系统质量。",
            "统计区间只描述固定10,000条类别平衡proposal队列，不能外推为全人口发生率或硬件可读性。",
            "类别9与增益的关联是读出敏感性诊断，不单独证明结构因果。",
        ],
        "execution": execution,
        "output_bytes": base._directory_bytes(output_dir),
        "output_budget_bytes": RESULT_BUDGET_BYTES,
    }
    _require(report["output_bytes"] <= RESULT_BUDGET_BYTES,
             "formal task outputs exceed the fixed 512 MiB result budget")
    return report, cells


def _write_final_artifacts(output_dir: Path, inputs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
                           condition_values: dict[str, dict[str, torch.Tensor]], preflight: dict[str, Any],
                           execution: dict[str, Any], config_path: Path, evaluator_path: Path) -> dict[str, Any]:
    labels = inputs["A"][1]["requested_labels"].long()
    proposals = inputs["A"][1]["proposal_ids"].long()
    _write_csv(output_dir, condition_values, labels, proposals,
               {condition: condition[:2] for condition in condition_values})
    report, _ = _build_report(inputs, condition_values, preflight, execution, output_dir,
                              config_path, evaluator_path)
    # Include the CSV and all block files in the budget calculation, then save the machine report.
    report["output_bytes"] = base._directory_bytes(output_dir)
    _atomic_json(output_dir / "final_report.json", report)
    previews = _write_previews(
        output_dir,
        condition_values,
        {cell: torch.stack([
            torch.load(output_dir / "blocks" / cell / f"class_{label:02d}.pt", map_location="cpu", weights_only=False)["native_raw_examples"]
            for label in range(10)
        ]) for cell in ("AA", "BA", "AB", "BB")},
        float(inputs["A"][0]["camera_gain"]),
    )
    report["preview_files"] = previews
    report["output_bytes"] = base._directory_bytes(output_dir)
    _atomic_json(output_dir / "final_report.json", report)
    _write_html(output_dir, report, previews)
    _atomic_json(output_dir / "validation_receipt.json", {
        "status": "PASSED",
        "task_id": TASK_ID,
        "measurement_complete": True,
        "checked": [
            "explicit A/B endpoint identities and common source step",
            "fixed paired proposal labels and IDs with finite complex64 coefficients",
            "AA/BB full native-cohort replay against saved production views and predictions",
            "four frozen Decoder/input combinations and three native gain settings",
            "parameter and buffer immutability; encoder calls zero; optimizer calls zero",
            "complete per-class block readback and 2000 synchronized paired bootstrap resamples",
            "independent production readout regression and negative controls",
        ],
        "output_bytes_including_receipt": base._directory_bytes(output_dir),
    })
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _resolve(args.config)
    evidence_dir = _resolve(args.ab_evidence_dir)
    evaluator_path = _resolve(args.evaluator_checkpoint)
    output_dir = _resolve(args.output_dir)
    config = _load_config(config_path)
    inputs = {letter: _load_endpoint(evidence_dir, letter, config) for letter in ("A", "B")}
    for letter in inputs:
        inputs[letter][0]["evidence_dir"] = str(evidence_dir)
    _validate_pairing(inputs)
    expected_evaluator = _resolve(ROOT / config["evaluation"]["evaluator_checkpoint"])
    _require(evaluator_path == expected_evaluator,
             "explicit evaluator path differs from the continuation's fixed E_D checkpoint")
    _require(torch.cuda.is_available(), "CUDA is unavailable on the required execution host")
    device = torch.device("cuda")
    existed = output_dir.exists()
    if existed:
        _require(args.resume, "output directory already exists; pass --resume to validate and continue")
        _require((output_dir / "preflight.json").is_file(), "resume output has no completed preflight evidence")
    preflight_started = time.perf_counter()
    preflight = _preflight(inputs, config, evaluator_path, device)
    preflight_seconds = time.perf_counter() - preflight_started
    if args.resume:
        saved_preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
        _require(saved_preflight.get("status") == "PASSED"
                 and saved_preflight.get("task_id") == TASK_ID
                 and saved_preflight.get("endpoint_identities") == _endpoint_identities(inputs)
                 and saved_preflight.get("evaluator_checkpoint") == str(evaluator_path)
                 and saved_preflight.get("config") == str(config_path),
                 "saved preflight does not match the explicit fixed endpoints")
        preflight = saved_preflight
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(output_dir / "preflight.json", {
            "schema": "lightgen_small_cmos_visibility_preflight.v1",
            "task_id": TASK_ID,
            "status": "PASSED",
            "endpoint_identities": _endpoint_identities(inputs),
            "endpoint_source_run_id": inputs["A"][0]["source_run_id"],
            "evaluator_checkpoint": str(evaluator_path),
            "config": str(config_path),
            **preflight,
        })
    if args.preflight_only:
        result = {"status": "PREFLIGHT_READY", "output_dir": str(output_dir),
                  "preflight": preflight, "elapsed_seconds": time.perf_counter() - started}
        _atomic_json(output_dir / "preflight_result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    execution = {
        "preflight_seconds_this_invocation": preflight_seconds,
        "optical_seconds": 0.0,
        "evaluator_seconds": 0.0,
        "block_reuse_count": 0,
        "block_compute_count": 0,
        "started_epoch_seconds": time.time(),
    }
    models: dict[str, torch.nn.Module] = {}
    snapshots: dict[str, dict[str, torch.Tensor]] = {}
    calls: dict[str, dict[str, int]] = {}
    hooks: list[Any] = []
    for letter in ("A", "B"):
        model, snapshot, counter, hook = _device_model(inputs[letter][0], config, device)
        models[letter], snapshots[letter], calls[letter] = model, snapshot, counter
        hooks.append(hook)
    evaluator = base._load_evaluator(evaluator_path, device)
    evaluator_snapshot = base._state_snapshot(evaluator)
    evaluator_calls = {"samples": 0, "forward_calls": 0}
    evaluator_hook = evaluator.register_forward_hook(
        partial(_count_evaluator_forward, calls=evaluator_calls)
    )
    full_expected_evaluator_samples = 0
    full_expected_evaluator_calls = 0
    try:
        with torch.inference_mode():
            total_blocks = 0
            for cell in ("AA", "BA", "AB", "BB"):
                for label in range(10):
                    block, timing = _run_block(
                        cell, label, inputs, models, evaluator, device, output_dir, args.resume
                    )
                    execution["optical_seconds"] += timing["optical_seconds"]
                    execution["evaluator_seconds"] += timing["evaluator_seconds"]
                    execution["block_reuse_count"] += int(timing["reused"])
                    execution["block_compute_count"] += int(not timing["reused"])
                    if not timing["reused"]:
                        full_expected_evaluator_samples += SAMPLES_PER_CLASS * len(_condition_ids(cell))
                        full_expected_evaluator_calls += (
                            SAMPLES_PER_CLASS // BATCH_SIZE
                        ) * len(_condition_ids(cell))
                    total_blocks += 1
                    _atomic_json(output_dir / "progress.json", {
                        "task_id": TASK_ID,
                        "completed_blocks": total_blocks,
                        "total_blocks": 40,
                        "last_block": {"cell": cell, "class_id": label},
                    })
    finally:
        for model_hooks in hooks:
            for hook in model_hooks:
                hook.remove()
        evaluator_hook.remove()
    for letter in ("A", "B"):
        _require(base._assert_close(snapshots[letter], base._state_snapshot(models[letter]),
                                    f"{letter}_endpoint_after_full_replay") == 0.0,
                 f"{letter} model parameter or buffer changed during replay")
        _require(calls[letter]["encoder_samples"] == 0,
                 f"Encoder {letter} was called in full measurement")
    optical_samples = sum(value["decoder_samples"] for value in calls.values())
    full_expected_optical_samples = execution["block_compute_count"] * SAMPLES_PER_CLASS
    optical_calls = sum(value["decoder_forward_calls"] for value in calls.values())
    full_expected_optical_calls = execution["block_compute_count"] * (SAMPLES_PER_CLASS // BATCH_SIZE)
    _require(optical_samples == full_expected_optical_samples,
             f"full replay optical sample count changed: {optical_samples} expected {full_expected_optical_samples}")
    _require(optical_calls == full_expected_optical_calls,
             f"full replay optical forward call count changed: {optical_calls} expected {full_expected_optical_calls}")
    _require(base._assert_close(evaluator_snapshot, base._state_snapshot(evaluator), "evaluator_after_full_replay") == 0.0,
             "E_D state changed during full replay")
    _require(evaluator_calls["samples"] == full_expected_evaluator_samples
             and evaluator_calls["forward_calls"] == full_expected_evaluator_calls,
             f"full replay E_D call/sample counts changed: {evaluator_calls}")
    execution["encoder_calls"] = calls
    execution["optical_forward_samples_this_invocation"] = optical_samples
    execution["optical_forward_calls_this_invocation"] = optical_calls
    execution["stored_optical_sample_count"] = COHORT_SIZE * 4
    execution["stored_evaluator_sample_count"] = 80_000
    execution["optimizer_created"] = False
    execution["optimizer_steps"] = 0
    execution["gradient_computation"] = False
    execution["generation_sampling"] = False
    execution["evaluator_samples_this_invocation"] = evaluator_calls["samples"]
    execution["evaluator_forward_calls_this_invocation"] = evaluator_calls["forward_calls"]
    execution["wall_seconds_before_export"] = time.perf_counter() - started
    condition_values: dict[str, dict[str, torch.Tensor]] = {}
    for cell in ("AA", "BA", "AB", "BB"):
        for condition_id in _condition_ids(cell):
            condition_values[condition_id] = _condition_arrays(output_dir, cell, condition_id, inputs)
    # E_D forwards are tracked by a pre-forward hook; this is a concrete consumer count.
    execution["evaluator_samples_expected_for_new_blocks"] = full_expected_evaluator_samples
    execution["evaluator_forward_calls_expected_for_new_blocks"] = full_expected_evaluator_calls
    execution["bootstrap_resamples"] = BOOTSTRAP_RESAMPLES
    execution["bootstrap_seed"] = BOOTSTRAP_SEED
    export_started = time.perf_counter()
    report = _write_final_artifacts(output_dir, inputs, condition_values, preflight, execution,
                                    config_path, evaluator_path)
    execution["export_seconds"] = time.perf_counter() - export_started
    execution["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    report["execution"].update(execution)
    checks = [
        "explicit A/B endpoint identities and common source step",
        "fixed paired proposal labels and IDs with finite complex64 coefficients",
        "AA/BB full native-cohort replay against saved production views and predictions",
        "four frozen Decoder/input combinations and three native gain settings",
        "parameter and buffer immutability; encoder calls zero; optimizer calls zero",
        "complete per-class block readback and 2000 synchronized paired bootstrap resamples",
        "independent production readout regression and negative controls",
    ]
    for _ in range(3):
        report["output_bytes"] = base._directory_bytes(output_dir)
        _atomic_json(output_dir / "final_report.json", report)
        _write_html(output_dir, report, report["preview_files"])
        _atomic_json(output_dir / "validation_receipt.json", {
            "status": "PASSED",
            "task_id": TASK_ID,
            "measurement_complete": True,
            "output_bytes_before_receipt_write": base._directory_bytes(output_dir),
            "checked": checks,
        })
    final_bytes = base._directory_bytes(output_dir)
    _require(final_bytes <= RESULT_BUDGET_BYTES,
             "final research artifacts exceed the approved 512 MiB result budget")
    report["output_bytes"] = final_bytes
    _atomic_json(output_dir / "final_report.json", report)
    _write_html(output_dir, report, report["preview_files"])
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir),
                      "elapsed_seconds": execution["elapsed_seconds_this_invocation"],
                      "output_bytes": report["output_bytes"], "decision": report["decision"]},
                     ensure_ascii=False), flush=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-evidence-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluator-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.resume:
        parser.error("--preflight-only and --resume cannot be combined")
    return args


def main() -> None:
    torch.set_grad_enabled(False)
    run(_parse_args())


if __name__ == "__main__":
    main()
