"""Run the fixed-population LightGen400 decoder diagnostic and A/B fit."""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    OLS_PORT_COUNT,
    ChenLightGen400Baseline,
    LightGen400Config,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from ldm_project.v2.optics.propagation import clear_production_transfer_cache

from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_QUICKCHECK_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_QUICKCHECK_20261015"
TASK_ID_4096 = "CHEN_LIGHTGEN_400_QUICKCHECK_4096_V2"
AUTHORITY_4096 = "USER_APPROVED_CHEN_LIGHTGEN_400_QUICKCHECK_4096_20261015"
DECODER_GRID = 2048
SELECTION_SEED = 2026101501
FIT_PER_CLASS = 10
CONFIRM_PER_CLASS = 20
POPULATION_PER_CLASS = FIT_PER_CLASS + CONFIRM_PER_CLASS
UPDATES = 500
BATCH_SIZE = 10
WINDOW_TOLERANCE = 1.0e-4
LOCAL_MSE_IMPROVEMENT_MIN = 0.10
LOCAL_SATURATION_MAX = 0.01
LOCAL_TASK_SIZE = 28
ARM_WEIGHTS = {"a_reconstruction_only": ("rec",), "b_reconstruction_plus_power": ("rec", "op")}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _activate_profile(decoder_grid: int) -> None:
    global TASK_ID, AUTHORITY, DECODER_GRID
    _require(decoder_grid in (2048, 4096), "decoder diagnostic grid must be 2048 or 4096")
    DECODER_GRID = decoder_grid
    if decoder_grid == 4096:
        TASK_ID, AUTHORITY = TASK_ID_4096, AUTHORITY_4096
    else:
        TASK_ID, AUTHORITY = "CHEN_LIGHTGEN_400_QUICKCHECK_V1", "USER_APPROVED_CHEN_LIGHTGEN_400_QUICKCHECK_20261015"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _load_inputs(
    checkpoint_path: Path, report_path: Path, config_path: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], ChenLightGen400Baseline, dict[str, torch.Tensor]]:
    _require(checkpoint_path.is_file(), f"explicit baseline checkpoint is missing: {checkpoint_path}")
    _require(report_path.is_file(), f"explicit baseline report is missing: {report_path}")
    _require(config_path.is_file(), f"explicit baseline config is missing: {config_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = _load_json(report_path)
    config = baseline._read_config(config_path)
    _require(checkpoint.get("schema") == "chen_lightgen_400_mnist_baseline_checkpoint.v1",
             "source checkpoint schema is not the Chen LightGen400 baseline")
    _require(checkpoint.get("completed_updates") == 30_000, "source checkpoint is not exactly step 30000")
    _require(checkpoint.get("task_id") == baseline.TASK_ID, "source checkpoint task identity changed")
    _require(report.get("status") == "VALID_FAILED", "source report does not preserve the valid-failed baseline result")
    _require(report.get("task_id") == baseline.TASK_ID, "source report belongs to another baseline")
    _require(checkpoint.get("data_roles") == {"train": 55_000, "dev": 5_000, "official_test_reads": 0},
             "source checkpoint data roles changed")
    _require(report.get("data_roles") == {"train_count": 55_000, "dev_count": 5_000,
                                           "dev_is_official_train_source_subset": True,
                                           "official_test_reads": 0},
             "source report data roles changed")
    expected_checkpoint_config = copy.deepcopy(config)
    expected_checkpoint_config["optics"]["selected_production_grid"] = 2048
    _require(checkpoint.get("config") == expected_checkpoint_config,
             "source checkpoint configuration differs from the explicit baseline config")
    _require(checkpoint.get("model_identity") == {
        "encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400], "ols": "gaussian_s12_v1"
    }, "source model identity changed")
    _require(config["optics"]["propagation_grid_candidates"] == [2048, 3072, 4096, 6144],
             "baseline propagation-window candidates changed")
    _require(checkpoint.get("config", {}).get("optics", {}).get("selected_production_grid") == 2048,
             "source checkpoint production grid is not 2048")
    references = checkpoint.get("references")
    _require(isinstance(references, dict) and set(references) == {"scale", "power_reference", "camera_gain"},
             "source frozen references are incomplete")
    _require(tuple(references["scale"].shape) == (OLS_PORT_COUNT,), "source modal scale has the wrong shape")
    _require(tuple(references["power_reference"].shape) == (7,), "source power reference must retain seven planes")
    _require(torch.isfinite(references["scale"]).all().item() and bool((references["scale"] > 0).all()),
             "source modal scales are invalid")
    _require(torch.isfinite(references["power_reference"]).all().item()
             and bool((references["power_reference"] > 0).all()), "source seven-plane power reference is invalid")
    gain = references["camera_gain"].reshape(())
    _require(torch.isfinite(gain).item() and float(gain) > 0.0, "source camera gain is invalid")
    weights = checkpoint.get("bogt_weights")
    _require(isinstance(weights, dict) and set(weights) == {"dist", "rec", "op"}
             and all(math.isfinite(float(value)) and float(value) > 0 for value in weights.values()),
             "source frozen BOGT weights are invalid")
    model = ChenLightGen400Baseline(LightGen400Config(2048, True)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    source_optimizer = torch.optim.Adam(
        [model.encoder_phase, model.decoder_phase],
        lr=1.0e-3,
        betas=baseline.TRAIN_BETAS,
        eps=baseline.TRAIN_EPSILON,
    )
    source_optimizer.load_state_dict(checkpoint["optimizer"])
    baseline._validate_adam_state(source_optimizer, completed_updates=30_000)
    reference_tensors = {name: tensor.detach().clone().to(device) for name, tensor in references.items()}
    model.eval()
    return config, checkpoint, report, model, reference_tensors


def _diagnostic_model(source_model: ChenLightGen400Baseline, decoder_grid: int) -> ChenLightGen400Baseline:
    """Rebuild grid-dependent geometry and copy only the learned phase masks."""
    candidate = ChenLightGen400Baseline(LightGen400Config(decoder_grid, True)).to(
        source_model.encoder_phase.device
    )
    with torch.no_grad():
        candidate.encoder_phase.copy_(source_model.encoder_phase)
        candidate.decoder_phase.copy_(source_model.decoder_phase)
    candidate.eval()
    return candidate


def _load_fixed_population(
    path: Path, labels: torch.Tensor, train_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Load the already selected population without resampling or refitting it."""
    _require(path.is_file(), f"explicit fixed population is missing: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "fit_rows_by_class", "confirm_rows_by_class", "fit_rows", "confirm_rows", "all_rows",
        "all_labels", "coefficients", "encoder_power_planes_w", "generated_indices",
        "generated_labels", "generated_coefficients",
    }
    _require(isinstance(saved, dict) and set(saved) == expected,
             "fixed population tensor contract changed")
    population = {key: value.detach().cpu() for key, value in saved.items()}
    _require(tuple(population["fit_rows_by_class"].shape) == (10, 10)
             and tuple(population["confirm_rows_by_class"].shape) == (10, 20),
             "fixed population row group shapes changed")
    _require(tuple(population["fit_rows"].shape) == (100,)
             and tuple(population["confirm_rows"].shape) == (200,)
             and tuple(population["all_rows"].shape) == (300,),
             "fixed population flattened row shapes changed")
    _require(torch.equal(population["fit_rows"], population["fit_rows_by_class"].flatten())
             and torch.equal(population["confirm_rows"], population["confirm_rows_by_class"].flatten())
             and torch.equal(population["all_rows"], torch.cat((population["fit_rows"], population["confirm_rows"]))),
             "fixed population flattened rows do not match their saved class groups")
    fit, confirm, all_rows = population["fit_rows"], population["confirm_rows"], population["all_rows"]
    _require(torch.unique(all_rows).numel() == all_rows.numel(), "fixed population rows are not unique")
    _require(bool(torch.isin(all_rows, train_indices).all()), "fixed population includes a non-TRAIN row")
    _require(torch.equal(labels[fit], torch.arange(10).repeat_interleave(10))
             and torch.equal(labels[confirm], torch.arange(10).repeat_interleave(20)),
             "fixed population labels do not preserve their recorded class order")
    _require(torch.equal(population["all_labels"], labels[all_rows].cpu()),
             "fixed population labels differ from canonical MNIST TRAIN")
    _require(tuple(population["coefficients"].shape) == (300, OLS_PORT_COUNT)
             and population["coefficients"].is_complex()
             and torch.isfinite(population["coefficients"]).all().item(),
             "fixed TRAIN OLS coefficients are malformed")
    powers = population["encoder_power_planes_w"]
    _require(tuple(powers.shape) == (300, 2) and torch.isfinite(powers).all().item()
             and bool((powers > 0).all()), "fixed Encoder power planes are malformed")
    expected_indices = torch.arange(10, dtype=torch.long) * 1000
    _require(torch.equal(population["generated_indices"], expected_indices)
             and torch.equal(population["generated_labels"], torch.arange(10)),
             "fixed generated latent selection indices or labels changed")
    generated = population["generated_coefficients"]
    _require(tuple(generated.shape) == (10, OLS_PORT_COUNT) and generated.is_complex()
             and torch.isfinite(generated).all().item(), "fixed generated OLS coefficients are malformed")
    return population


def _verify_population_against_source_encoder(
    source_model: ChenLightGen400Baseline,
    population: dict[str, torch.Tensor],
    pixels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Independently confirm saved rows came from the explicit 2048 source model."""
    max_coefficient_error = 0.0
    max_encoder_power_error = 0.0
    max_shared_path_error = 0.0
    gain_numerator = torch.zeros((), dtype=torch.float64)
    gain_denominator = torch.zeros((), dtype=torch.float64)
    fit_count = 0
    rows = population["all_rows"]
    source_model.eval()
    with torch.no_grad():
        for offset in range(0, rows.numel(), BATCH_SIZE):
            row_batch = rows[offset : offset + BATCH_SIZE]
            target, _ = baseline._image_batch(pixels, row_batch, device)
            encoded = source_model.forward_from_image(target)
            saved_z = population["coefficients"][offset : offset + row_batch.numel()].to(device)
            saved_encoder_power = population["encoder_power_planes_w"][offset : offset + row_batch.numel()].to(device)
            max_coefficient_error = max(
                max_coefficient_error,
                float((encoded["ols_coefficients"] - saved_z).abs().max().cpu()),
            )
            max_encoder_power_error = max(
                max_encoder_power_error,
                float((encoded["plane_powers_w"][:, :2] - saved_encoder_power).abs().max().cpu()),
            )
            reemitted = source_model.forward_from_ols(saved_z)
            error = (encoded["raw_intensity_w_per_m2"] - reemitted["raw_intensity_w_per_m2"]).abs().max()
            max_shared_path_error = max(max_shared_path_error, float(error.cpu()))
            if offset < population["fit_rows"].numel():
                count = min(row_batch.numel(), population["fit_rows"].numel() - offset)
                physical = encoded["physical_w_per_pixel"][:count].double()
                targets = target[:count].double()
                gain_numerator += (physical * targets).sum().cpu()
                gain_denominator += physical.square().sum().cpu()
                fit_count += count
    _require(max_coefficient_error <= 1.0e-6,
             "saved OLS coefficients do not match their source 2048 image encodings")
    _require(max_encoder_power_error <= 1.0e-10,
             "saved Encoder power planes do not match their source image paths")
    _require(max_shared_path_error <= 1.0e-8,
             "image and saved-OLS source readouts disagree")
    _require(fit_count == 100 and bool(gain_denominator > 0), "fit-only gain counterfactual is incomplete")
    fit_gain = float(gain_numerator / gain_denominator)
    _require(math.isfinite(fit_gain) and fit_gain > 0.0, "fit-only gain counterfactual is invalid")
    return {
        "max_abs_saved_coefficient_error": max_coefficient_error,
        "max_abs_saved_encoder_power_error_w": max_encoder_power_error,
        "max_abs_image_vs_saved_ols_readout_error_w_per_m2": max_shared_path_error,
        "fit_only_gain_w_inverse": fit_gain,
        "fit_gain_numerator": float(gain_numerator),
        "fit_gain_denominator": float(gain_denominator),
    }


def _fixed_rows(labels: torch.Tensor, train_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    fit_rows: list[torch.Tensor] = []
    confirm_rows: list[torch.Tensor] = []
    for label in range(10):
        rows = train_indices[labels[train_indices] == label]
        _require(rows.numel() >= POPULATION_PER_CLASS, f"canonical TRAIN class {label} is too small")
        chosen = rows[torch.randperm(rows.numel(), generator=generator)[:POPULATION_PER_CLASS]]
        fit_rows.append(chosen[:FIT_PER_CLASS])
        confirm_rows.append(chosen[FIT_PER_CLASS:])
    fit = torch.stack(fit_rows)
    confirm = torch.stack(confirm_rows)
    _require(not bool(torch.isin(fit.flatten(), confirm.flatten()).any()), "fit and confirmation rows overlap")
    _require(torch.equal(labels[fit.flatten()], torch.arange(10).repeat_interleave(FIT_PER_CLASS)),
             "fit rows do not preserve class order")
    _require(torch.equal(labels[confirm.flatten()], torch.arange(10).repeat_interleave(CONFIRM_PER_CLASS)),
             "confirmation rows do not preserve class order")
    return {"fit_rows_by_class": fit, "confirm_rows_by_class": confirm}


def _extract_fixed_population(
    model: ChenLightGen400Baseline,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    rows: dict[str, torch.Tensor],
    generated_payload: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float]]:
    fit_rows = rows["fit_rows_by_class"].flatten()
    confirm_rows = rows["confirm_rows_by_class"].flatten()
    all_rows = torch.cat((fit_rows, confirm_rows))
    coefficients: list[torch.Tensor] = []
    encoder_powers: list[torch.Tensor] = []
    gain_num = torch.zeros((), dtype=torch.float64)
    gain_den = torch.zeros((), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, all_rows.numel(), BATCH_SIZE):
            batch_rows = all_rows[offset : offset + BATCH_SIZE]
            target, _ = baseline._image_batch(pixels, batch_rows, device)
            image_result = model.forward_from_image(target)
            z = image_result["ols_coefficients"]
            reemitted = model.forward_from_ols(z, return_diagnostics=True)
            image_raw = image_result["raw_intensity_w_per_m2"]
            generated_raw = reemitted["raw_intensity_w_per_m2"]
            _require(torch.allclose(image_raw, generated_raw, rtol=2.0e-5, atol=1.0e-8),
                     "image reconstruction and saved-OLS decoder paths disagree")
            decoder_powers = reemitted["plane_powers_w"]
            complete_powers = torch.cat((image_result["plane_powers_w"][:, :2], decoder_powers), dim=1)
            _require(tuple(complete_powers.shape) == (batch_rows.numel(), 7),
                     "saved-latent loss path does not expose the original seven power planes")
            _require(torch.allclose(image_result["plane_powers_w"], complete_powers,
                                    rtol=2.0e-5, atol=1.0e-10),
                     "saved-OLS seven-plane powers differ from the image reconstruction path")
            coefficients.append(z.detach().cpu())
            encoder_powers.append(image_result["plane_powers_w"][:, :2].detach().cpu())
            if offset < FIT_PER_CLASS * 10:
                camera = image_result["physical_w_per_pixel"].double()
                target64 = target.double()
                gain_num += (camera * target64).sum().cpu()
                gain_den += camera.square().sum().cpu()
    z_all = torch.cat(coefficients)
    power_all = torch.cat(encoder_powers)
    _require(tuple(z_all.shape) == (300, OLS_PORT_COUNT), "fixed TRAIN population latent shape changed")
    gen_coeffs = generated_payload["coefficients"]
    gen_labels = generated_payload["requested_labels"].long()
    _require(generated_payload.get("schema") == "chen_lightgen_400_empirical_complex_gaussian_samples.v1",
             "saved baseline generated latent schema changed")
    gen_indices = torch.arange(10, dtype=torch.long) * 1000
    gen_requested = gen_labels.index_select(0, gen_indices)
    _require(torch.equal(gen_requested, torch.arange(10)), "saved generated latents are not grouped by requested class")
    population = {
        **rows,
        "fit_rows": fit_rows,
        "confirm_rows": confirm_rows,
        "all_rows": all_rows,
        "all_labels": labels[all_rows].cpu(),
        "coefficients": z_all,
        "encoder_power_planes_w": power_all,
        "generated_indices": gen_indices,
        "generated_labels": gen_requested,
        "generated_coefficients": gen_coeffs.index_select(0, gen_indices).cpu(),
    }
    _require(bool(gain_den > 0), "fit-only camera gain has a zero denominator")
    gain_fit = float(gain_num / gain_den)
    _require(math.isfinite(gain_fit) and gain_fit > 0, "fit-only camera gain is invalid")
    return population, {"fit_only_gain_w_inverse": gain_fit,
                        "gain_numerator": float(gain_num), "gain_denominator": float(gain_den)}


def _batch_indices(population: dict[str, Any], rows: torch.Tensor) -> torch.Tensor:
    map_rows = {int(row): index for index, row in enumerate(population["all_rows"].tolist())}
    return torch.tensor([map_rows[int(row)] for row in rows.tolist()], dtype=torch.long)


def _latent_result(
    model: ChenLightGen400Baseline,
    z: torch.Tensor,
    encoder_powers: torch.Tensor,
    *,
    diagnostics: bool,
) -> dict[str, torch.Tensor]:
    result = model.forward_from_ols(z, return_diagnostics=diagnostics)
    powers = torch.cat((encoder_powers, result["plane_powers_w"]), dim=1) if diagnostics else torch.empty(0)
    if diagnostics:
        _require(tuple(powers.shape) == (z.shape[0], 7), "complete BOGT power loss must contain seven planes")
    return {
        "raw_intensity_w_per_m2": result["raw_intensity_w_per_m2"],
        "physical_w_per_pixel": result["physical_w_per_pixel"],
        "plane_powers_w": powers,
    }


def _physical_from_ols_in_batches(
    model: ChenLightGen400Baseline,
    coefficients: torch.Tensor,
    encoder_powers: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    _require(batch_size > 0 and coefficients.shape[0] == encoder_powers.shape[0],
             "batched OLS readout requires a positive batch size and matching rows")
    parts = []
    with torch.no_grad():
        for start in range(0, coefficients.shape[0], batch_size):
            stop = min(start + batch_size, coefficients.shape[0])
            result = _latent_result(
                model, coefficients[start:stop], encoder_powers[start:stop], diagnostics=False
            )
            parts.append(result["physical_w_per_pixel"])
    return torch.cat(parts, dim=0)


def _unit_outputs(physical_w_per_pixel: torch.Tensor, camera_gain_w_inverse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    task_unclipped = physical_w_per_pixel * camera_gain_w_inverse.to(
        device=physical_w_per_pixel.device, dtype=physical_w_per_pixel.dtype
    )
    return task_unclipped, task_unclipped.clamp(0.0, 1.0)


def _losses(
    model: ChenLightGen400Baseline,
    z: torch.Tensor,
    target: torch.Tensor,
    encoder_powers: torch.Tensor,
    references: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    result = _latent_result(model, z, encoder_powers, diagnostics=True)
    task_raw = result["physical_w_per_pixel"] * references["camera_gain"].to(
        device=z.device, dtype=result["physical_w_per_pixel"].dtype
    )
    l_rec = (task_raw - target).square().mean()
    powers = result["plane_powers_w"].double()
    l_op = torch.log((powers + baseline.EPS_OP) /
                     (references["power_reference"].double().unsqueeze(0) + baseline.EPS_OP)).square().mean()
    return l_rec, {"rec": l_rec, "op": l_op}, result


def _weighted_objective_value_without_grad(
    model: ChenLightGen400Baseline,
    z: torch.Tensor,
    target: torch.Tensor,
    encoder_powers: torch.Tensor,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    terms: tuple[str, ...],
) -> float:
    with torch.no_grad():
        _, components, _ = _losses(model, z, target, encoder_powers, references)
        value = sum(weights[name] * components[name] for name in terms)
    return float(value.detach().cpu())


def _set_decoder_trainable(model: ChenLightGen400Baseline) -> None:
    model.requires_grad_(False)
    model.decoder_phase.requires_grad_(True)


def _train_update(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    z: torch.Tensor,
    target: torch.Tensor,
    encoder_powers: torch.Tensor,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    arm: str,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    _, components, _ = _losses(model, z, target, encoder_powers, references)
    terms = ARM_WEIGHTS[arm]
    total = sum(weights[name] * components[name] for name in terms)
    _require(torch.isfinite(total).item(), f"{arm} total objective is nonfinite")
    total.backward()
    gradient = model.decoder_phase.grad
    _require(gradient is not None and torch.isfinite(gradient).all().item(), f"{arm} Decoder gradient is invalid")
    _require(bool(gradient.abs().sum() > 0), f"{arm} Decoder did not receive a real update direction")
    layer_norms = [float(torch.linalg.vector_norm(gradient[index].double()).detach().cpu()) for index in range(4)]
    _require(all(math.isfinite(value) and value > 0 for value in layer_norms),
             f"{arm} one or more Decoder phase planes have no gradient")
    previous = model.decoder_phase.detach().clone()
    optimizer.step()
    layer_deltas = [float((model.decoder_phase[index] - previous[index]).abs().max().detach().cpu()) for index in range(4)]
    _require(all(math.isfinite(value) and value > 0 for value in layer_deltas),
             f"{arm} one or more Decoder phase planes did not change")
    return {
        "total": float(total.detach().cpu()),
        "l_rec": float(components["rec"].detach().cpu()),
        "l_op": float(components["op"].detach().cpu()),
        "decoder_gradient_norm": float(torch.linalg.vector_norm(gradient.detach().double()).cpu()),
        "decoder_layer_gradient_norms": layer_norms,
        "decoder_layer_max_update": layer_deltas,
    }


def _objective_gradient_diagnostics(
    model: ChenLightGen400Baseline,
    z: torch.Tensor,
    target: torch.Tensor,
    encoder_powers: torch.Tensor,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    arm: str,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    _, components, _ = _losses(model, z, target, encoder_powers, references)
    rec_gradient = torch.autograd.grad(weights["rec"] * components["rec"], model.decoder_phase,
                                       retain_graph=True)[0]
    op_gradient = torch.autograd.grad(weights["op"] * components["op"], model.decoder_phase,
                                      retain_graph=True, allow_unused=True)[0]
    _require(op_gradient is not None,
             "weighted power loss is disconnected from the trainable Decoder parameters")
    rec64, op64 = rec_gradient.double(), op_gradient.double()
    rec_norm = torch.linalg.vector_norm(rec64)
    op_norm = torch.linalg.vector_norm(op64)
    _require(torch.isfinite(rec_norm).item() and torch.isfinite(op_norm).item()
             and bool(rec_norm > 0) and bool(op_norm > 0),
             "weighted reconstruction or power gradient is zero or nonfinite")
    cosine = float((rec64.flatten() @ op64.flatten() /
                    (rec_norm * op_norm).clamp_min(1.0e-300)).detach().cpu()) if bool(op_norm > 0) else None
    finite_difference: list[dict[str, Any]] = []
    terms = ARM_WEIGHTS[arm]
    epsilon = 1.0e-2
    total = sum(weights[name] * components[name] for name in terms)
    total_gradient = torch.autograd.grad(total, model.decoder_phase)[0]
    for layer in range(4):
        flat_index = int(total_gradient[layer].detach().abs().flatten().argmax())
        y, x = divmod(flat_index, total_gradient.shape[-1])
        autodiff = float(total_gradient[layer, 0, y, x].detach().cpu())
        with torch.no_grad():
            original = model.decoder_phase[layer, 0, y, x].clone()
            model.decoder_phase[layer, 0, y, x] = original + epsilon
        plus = _weighted_objective_value_without_grad(
            model, z, target, encoder_powers, references, weights, terms
        )
        with torch.no_grad():
            model.decoder_phase[layer, 0, y, x] = original - epsilon
        minus = _weighted_objective_value_without_grad(
            model, z, target, encoder_powers, references, weights, terms
        )
        with torch.no_grad():
            model.decoder_phase[layer, 0, y, x] = original
        finite = (plus - minus) / (2.0 * epsilon)
        relative_error = abs(finite - autodiff) / max(abs(finite), abs(autodiff), 1.0e-12)
        finite_difference.append({"layer": layer + 1, "index_yx": [y, x], "autodiff": autodiff,
                                  "central_difference": finite, "relative_error": relative_error,
                                  "same_direction": finite * autodiff > 0.0})
    _require(all(math.isfinite(item["autodiff"]) and math.isfinite(item["central_difference"])
                 and item["same_direction"] and item["relative_error"] <= 0.25 for item in finite_difference),
             f"{arm} Decoder finite-difference check did not confirm gradient values and directions")
    before, after = _adam_first_step_from_gradient(model, total_gradient)
    descent: list[dict[str, Any]] = []
    for item in finite_difference:
        layer = item["layer"] - 1
        y, x = item["index_yx"]
        delta = float((after[layer, 0, y, x] - before[layer, 0, y, x]).detach().cpu())
        agrees = delta * item["central_difference"] < 0.0
        descent.append({"layer": layer + 1, "index_yx": [y, x], "adam_delta": delta,
                        "descends_finite_difference": agrees})
    _require(all(item["descends_finite_difference"] for item in descent),
             f"{arm} first Adam update did not descend the finite-difference direction")
    return {
        "weighted_l_rec_gradient_norm": float(rec_norm.cpu()),
        "weighted_l_op_gradient_norm": float(op_norm.cpu()),
        "rec_op_cosine": cosine,
        "l_op_decoder_gradient_zero": bool(op_norm == 0),
        "finite_difference": finite_difference,
        "finite_difference_epsilon_rad": epsilon,
        "first_adam_update_direction": descent,
    }


def _validate_pairing_negative_control(
    prediction: torch.Tensor, target: torch.Tensor, rows: torch.Tensor, labels: torch.Tensor
) -> dict[str, Any]:
    _, clean_psnr, clean_ssim = baseline._reconstruction_batch_metrics(prediction, target, rows, rows)
    permutation = torch.roll(torch.arange(rows.numel()), shifts=1)
    shuffled_target = target.index_select(0, permutation.to(target.device))
    _, shuffled_psnr, shuffled_ssim = baseline._reconstruction_batch_metrics(
        prediction, shuffled_target, rows, rows
    )
    clean_mse = float((prediction - target).square().mean().cpu())
    shuffled_mse = float((prediction - shuffled_target).square().mean().cpu())
    _require(math.isfinite(clean_mse) and math.isfinite(shuffled_mse) and shuffled_mse > clean_mse
             and float(shuffled_psnr.mean()) < float(clean_psnr.mean())
             and float(shuffled_ssim.mean()) < float(clean_ssim.mean()),
             "production reconstruction metric did not detect shuffled target content")
    try:
        baseline._reconstruction_batch_metrics(prediction, target, rows, rows.flip(0))
    except ValueError as error:
        _require("rows do not match" in str(error), "pairing consumer rejected rows for an unexpected reason")
        return {"passed": True, "rejection": "row_identity_mismatch", "rows": rows.tolist(),
                "shuffled_rows": rows.flip(0).tolist(), "labels": labels.tolist(),
                "clean_mse": clean_mse, "shuffled_content_mse": shuffled_mse,
                "consumer_clean_psnr28_db": float(clean_psnr.mean().cpu()),
                "consumer_shuffled_psnr28_db": float(shuffled_psnr.mean().cpu()),
                "consumer_clean_ssim28": float(clean_ssim.mean().cpu()),
                "consumer_shuffled_ssim28": float(shuffled_ssim.mean().cpu())}
    raise RuntimeError("reconstruction metric consumer accepted shuffled target rows")


def _window_from_ols(
    source_model: ChenLightGen400Baseline,
    coefficients: torch.Tensor,
    candidates: list[int],
    roi_m: float,
    *,
    include_sampling_reference: bool,
) -> dict[str, Any]:
    diagnostics: dict[int | str, dict[str, Any]] = {}

    def evaluate(grid: int, *, factor: int = 1, key: int | str | None = None) -> dict[str, torch.Tensor]:
        clear_production_transfer_cache()
        candidate = ChenLightGen400Baseline(
            LightGen400Config(grid, True, pixel_sampling_factor=factor)
        ).to(coefficients.device)
        candidate.encoder_phase.data.copy_(source_model.encoder_phase.detach())
        candidate.decoder_phase.data.copy_(source_model.decoder_phase.detach())
        candidate.eval()
        with torch.no_grad():
            result = candidate.forward_from_ols(coefficients, return_diagnostics=True)
        fields = {"generator_entry": result["generator_entry_field"].tensor,
                  **{f"decoder_{index + 1}": field.tensor
                     for index, field in enumerate(result["decoder_fields"])} }
        roi_pixels = int(round(roi_m / candidate.config.pixel_pitch_m))
        cropped = {name: baseline._common_roi(value, roi_pixels).detach().cpu()
                   for name, value in fields.items()}
        diagnostics[grid if key is None else key] = {
            "plane_powers_w": result["plane_powers_w"][0].detach().cpu().tolist(),
            "roi_pixels": roi_pixels,
        }
        del candidate, result
        if coefficients.device.type == "cuda":
            torch.cuda.empty_cache()
        return cropped

    largest = candidates[-1]
    reference = evaluate(largest)
    errors: dict[int, dict[str, float]] = {largest: {name: 0.0 for name in reference}}
    phases = {largest: 0.0}
    for grid in candidates[:-1]:
        current = evaluate(grid)
        errors[grid], phases[grid] = baseline._complex_relative_errors(current, reference)
    if include_sampling_reference:
        factor = 2
        fine = evaluate(candidates[0] * factor, factor=factor, key="half_pixel_sampling_reference")
        fine = {name: baseline._average_fine_samples_at_coarse_pixel_centers(value)
                for name, value in fine.items()}
        coarse = evaluate(candidates[0])
        fine_errors, fine_phase = baseline._complex_relative_errors(coarse, fine)
    else:
        fine_errors, fine_phase = {}, None
    return {
        "complex_relative_error_vs_largest_window": errors,
        "aligned_global_phase_vs_largest_window_rad": phases,
        "grid_diagnostics": diagnostics,
        "max_selected_grid_error": max(errors[candidates[0]].values()),
        "spatial_sampling_reference": {
            "status": "MEASURED_DIAGNOSTIC_ONLY" if include_sampling_reference else "NOT_SELECTED",
            "reference_sampling_factor": 2 if include_sampling_reference else None,
            "complex_relative_error_by_plane": fine_errors,
            "aligned_global_phase_rad": fine_phase,
            "acceptance_threshold": None,
        },
    }


def _window_checks(
    model: ChenLightGen400Baseline,
    population: dict[str, Any],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    *,
    include_generated: bool = True,
) -> dict[str, Any]:
    available = [int(value) for value in config["optics"]["propagation_grid_candidates"]]
    _require(DECODER_GRID in available and available[-1] > DECODER_GRID,
             "diagnostic decoder grid and independent reference grid are not configured")
    candidates = [DECODER_GRID, available[-1]]
    roi = float(config["optics"]["common_complex_comparison_roi_m"])
    real: list[dict[str, Any]] = []
    gen: list[dict[str, Any]] = []
    confirm_by_class = population["confirm_rows_by_class"]
    model.eval()
    row_to_position = {int(row): position for position, row in enumerate(population["all_rows"].tolist())}
    for label in range(10):
        row = int(confirm_by_class[label, 0])
        position = row_to_position[row]
        coefficients = population["coefficients"][position : position + 1].to(device)
        details = _window_from_ols(
            model, coefficients, candidates, roi, include_sampling_reference=label in (0, 1)
        )
        grid_errors = details["complex_relative_error_vs_largest_window"]
        record = {"label": label, "row": row, "saved_population_position": position, "grids": details,
                  "max_decoder_grid_error": max(grid_errors[DECODER_GRID].values()),
                  "all_selected_planes_within_tolerance": max(grid_errors[DECODER_GRID].values())
                  <= WINDOW_TOLERANCE}
        real.append(record)
        if include_generated:
            z = population["generated_coefficients"][label : label + 1].to(device)
            generated = _window_from_ols(
                model, z, candidates, roi, include_sampling_reference=label in (0, 1)
            )
            gen_max = max(generated["complex_relative_error_vs_largest_window"][DECODER_GRID].values())
            gen.append({"requested_label": label,
                        "saved_generation_index": int(population["generated_indices"][label]),
                        "max_decoder_grid_error": gen_max,
                        "all_selected_planes_within_tolerance": gen_max <= WINDOW_TOLERANCE,
                        **generated})
    return {
        "candidate_grids": candidates,
        "reference_grid": candidates[-1],
        "source_encoder_grid": 2048,
        "required_decoder_grid": DECODER_GRID,
        "tolerance": WINDOW_TOLERANCE,
        "reconstruction": real,
        "generated": gen,
        "fixed_coefficients_consumed_without_reencoding": True,
        "passed": all(record["all_selected_planes_within_tolerance"] for record in real)
                  and all(record["all_selected_planes_within_tolerance"] for record in gen),
    }


def _fixed_population_for_task(
    args: argparse.Namespace,
    source_model: ChenLightGen400Baseline,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if args.fixed_population is not None:
        population = _load_fixed_population(_resolve(args.fixed_population), labels, train_indices)
        if args.generated_latents is not None:
            _validate_saved_generation_coefficients(population, _load_generated(_resolve(args.generated_latents)))
        validation = _verify_population_against_source_encoder(source_model, population, pixels, device)
        return population, validation
    _require(DECODER_GRID == 2048,
             "4096 diagnostic requires the explicit saved --fixed-population artifact")
    rows = _fixed_rows(labels, train_indices)
    population, gain = _extract_fixed_population(
        source_model, pixels, labels, rows, _load_generated(_resolve(args.generated_latents)), device
    )
    return population, {"legacy_2048_population_extraction": True, **gain}


def _validate_saved_generation_coefficients(
    population: dict[str, torch.Tensor], generated_payload: dict[str, Any]
) -> None:
    _require(generated_payload.get("schema") == "chen_lightgen_400_empirical_complex_gaussian_samples.v1",
             "saved source generation artifact schema changed")
    indices = population["generated_indices"]
    expected_labels = torch.arange(10, dtype=torch.long)
    _require(torch.equal(generated_payload["requested_labels"].index_select(0, indices), expected_labels),
             "fixed generation selection indices no longer identify one sample per class")
    expected = generated_payload["coefficients"].index_select(0, indices)
    _require(torch.equal(population["generated_coefficients"], expected),
             "fixed generation coefficients differ from the saved source generation artifact")


def _evaluate(
    model: ChenLightGen400Baseline,
    population: dict[str, Any],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    evaluator: torch.nn.Module,
    references: dict[str, torch.Tensor],
    device: torch.device,
    *,
    include_unit_readout: bool,
) -> dict[str, Any]:
    all_rows = population["all_rows"]
    coeffs = population["coefficients"].to(device)
    enc_power = population["encoder_power_planes_w"].to(device)
    unit_rows = population["confirm_rows_by_class"][:, 0].tolist()
    records: list[dict[str, Any]] = []
    unit: dict[str, list[torch.Tensor]] = {"physical_w_per_pixel": [], "task_intensity_unclipped": [],
                                         "display_intensity_clipped": [], "target": []}
    unit_row_ids: list[int] = []
    views_for_atlas: dict[int, torch.Tensor] = {}
    predictions: list[torch.Tensor] = []
    model.eval()
    evaluator.eval()
    with torch.no_grad():
        for offset in range(0, all_rows.numel(), BATCH_SIZE):
            rows = all_rows[offset : offset + BATCH_SIZE]
            pos = torch.arange(offset, offset + rows.numel(), dtype=torch.long)
            target, _ = baseline._image_batch(pixels, rows, device)
            result = _latent_result(model, coeffs[pos.to(device)], enc_power[pos.to(device)], diagnostics=True)
            physical = result["physical_w_per_pixel"]
            task_raw, display = _unit_outputs(physical, references["camera_gain"])
            view, psnr, ssim = baseline._reconstruction_batch_metrics(task_raw, target, rows, rows)
            mse400 = (task_raw - target).square().mean((1, 2, 3))
            saturation = ((task_raw < 0.0) | (task_raw > 1.0)).float().mean((1, 2, 3))
            logits = evaluator(view)
            predicted = logits.argmax(1).detach().cpu()
            predictions.append(predicted)
            label_batch = labels[rows].cpu()
            for index in range(rows.numel()):
                position = offset + index
                records.append({"row": int(rows[index]), "label": int(label_batch[index]),
                                "population": "fit" if position < 100 else "confirmation",
                                "mse400_unclipped": float(mse400[index].cpu()),
                                "psnr28_db": float(psnr[index].cpu()), "ssim28": float(ssim[index].cpu()),
                                "saturation_fraction": float(saturation[index].cpu()),
                                "ed_prediction": int(predicted[index])})
                if int(rows[index]) in unit_rows:
                    unit_row_ids.append(int(rows[index]))
                    unit["physical_w_per_pixel"].append(physical[index].detach().cpu())
                    unit["task_intensity_unclipped"].append(task_raw[index].detach().cpu())
                    unit["display_intensity_clipped"].append(display[index].detach().cpu())
                    unit["target"].append(target[index].detach().cpu())
                if int(rows[index]) in set(population["confirm_rows_by_class"][:, :3].flatten().tolist()):
                    views_for_atlas[int(rows[index])] = view[index].detach().cpu()
    predicted = torch.cat(predictions)
    requested = labels[all_rows].cpu()
    summaries: dict[str, Any] = {}
    for population_name, chosen in (("fit", torch.arange(0, 100)), ("confirmation", torch.arange(100, 300))):
        local_rows = chosen.tolist()
        classification = baseline._classification_summary(requested[local_rows], predicted[local_rows])
        local_records = [records[index] for index in local_rows]
        summaries[population_name] = {
            "count": len(local_rows),
            "mse400_unclipped_mean": sum(record["mse400_unclipped"] for record in local_records) / len(local_records),
            "psnr28_db_mean": sum(record["psnr28_db"] for record in local_records) / len(local_records),
            "ssim28_mean": sum(record["ssim28"] for record in local_records) / len(local_records),
            "saturation_fraction": sum(record["saturation_fraction"] for record in local_records) / len(local_records),
            "ed_macro_recall_descriptive": classification["macro_recall"],
            "ed_recall_by_class_descriptive": classification["recall_by_class"].tolist(),
            "ed_confusion_matrix_descriptive": classification["confusion_matrix"].tolist(),
        }
    result = {"per_sample": records, "summary": summaries, "unit_readout": None, "preview_views": views_for_atlas}
    if include_unit_readout:
        result["unit_readout"] = {
            "schema": "chen_lightgen_400_quickcheck_units.v1",
            "units": {"physical_w_per_pixel": "W/pixel",
                      "task_intensity_unclipped": "dimensionless target scale after frozen gain",
                      "display_intensity_clipped": "dimensionless, clipped to [0,1]"},
            "camera_gain_w_inverse": float(references["camera_gain"].detach().cpu()),
            "row_indices": unit_row_ids,
            **{name: torch.stack(values) for name, values in unit.items()},
        }
    return result


def _evaluate_generated(
    model: ChenLightGen400Baseline,
    population: dict[str, Any],
    evaluator: torch.nn.Module,
    references: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    z = population["generated_coefficients"].to(device)
    labels = population["generated_labels"].long()
    model.eval()
    with torch.no_grad():
        result = model.forward_from_ols(z, return_diagnostics=True)
        physical = result["physical_w_per_pixel"]
        task_raw, display = _unit_outputs(physical, references["camera_gain"])
        views = baseline._resize_view(task_raw)
        predicted = evaluator(views).argmax(1).cpu()
        classification = baseline._classification_summary(labels, predicted)
        saturation = ((task_raw < 0.0) | (task_raw > 1.0)).float().mean((1, 2, 3)).cpu()
        camera_power = physical.double().sum((1, 2, 3)).cpu()
        return {
            "indices": population["generated_indices"].tolist(),
            "requested_labels": labels.tolist(),
            "predictions": predicted.tolist(),
            "ed_macro_recall_descriptive": classification["macro_recall"],
            "ed_confusion_matrix_descriptive": classification["confusion_matrix"].tolist(),
            "saturation_fraction": float(saturation.mean()),
            "per_sample": [{"saved_generation_index": int(index), "requested_label": int(label),
                             "ed_prediction": int(pred), "saturation_fraction": float(saturation[position]),
                             "camera_power_w": float(camera_power[position])}
                            for position, (index, label, pred) in enumerate(zip(
                                population["generated_indices"].tolist(), labels.tolist(), predicted.tolist(), strict=True
                            ))],
            "unit_readout": {
                "schema": "chen_lightgen_400_quickcheck_units.v1",
                "units": {"physical_w_per_pixel": "W/pixel",
                          "task_intensity_unclipped": "dimensionless target scale after frozen gain",
                          "display_intensity_clipped": "dimensionless, clipped to [0,1]"},
                "camera_gain_w_inverse": float(references["camera_gain"].detach().cpu()),
                "saved_generation_indices": population["generated_indices"].tolist(),
                "requested_labels": labels,
                "physical_w_per_pixel": physical.detach().cpu(),
                "task_intensity_unclipped": task_raw.detach().cpu(),
                "display_intensity_clipped": display.detach().cpu(),
            },
            "display_views": views.detach().cpu(),
        }


def _initial_latent_kl(population: dict[str, Any], scale: torch.Tensor, device: torch.device) -> float:
    batch_losses = []
    for batch_index in range(FIT_PER_CLASS):
        rows = population["fit_rows_by_class"][:, batch_index]
        positions = _batch_indices(population, rows)
        z = population["coefficients"].index_select(0, positions).to(device)
        power = z.real.double().square() + z.imag.double().square()
        u = power / (scale.to(device).double().unsqueeze(0) + baseline.EPS_S)
        batch_losses.append(baseline.truncated_normal_positive_kl(
            baseline.q_from_batch(u), baseline.fixed_prior(device=device)
        ).mean())
    return float(torch.stack(batch_losses).mean().detach().cpu())


def _save_atlas(path: Path, pixels: torch.Tensor, labels: torch.Tensor, population: dict[str, Any], views: dict[str, dict[int, torch.Tensor]]) -> None:
    selected = population["confirm_rows_by_class"][:, :3].flatten().tolist()
    columns = ["target", "source_2048", "start_4096", "A", "B"]
    cell, margin, label_width = 28, 4, 22
    image = Image.new("L", (label_width + len(columns) * (cell + margin) + margin,
                            margin + len(selected) * (cell + margin)), color=255)
    draw = ImageDraw.Draw(image)
    for row_index, row in enumerate(selected):
        target, _ = baseline._image_batch(pixels, torch.tensor([row]), torch.device("cpu"))
        target_view = F.adaptive_avg_pool2d(target, (28, 28))[0, 0]
        values = [target_view] + [views[name][row][0] for name in ("source_2048", "start_4096", "A", "B")]
        y = margin + row_index * (cell + margin)
        draw.text((1, y + 8), str(int(labels[row])), fill=0)
        for column, value in enumerate(values):
            pixels_u8 = (value.clamp(0, 1).numpy() * 255.0).round().astype("uint8")
            tile = Image.fromarray(pixels_u8, mode="L")
            x = label_width + margin + column * (cell + margin)
            image.paste(tile, (x, y))
    image.save(path)


def _save_generation_atlas(path: Path, generated: dict[str, dict[str, Any]]) -> None:
    columns = ["source_2048", "start_4096", "A", "B"]
    cell, margin, label_width = 28, 4, 22
    image = Image.new("L", (label_width + len(columns) * (cell + margin) + margin,
                            margin + 10 * (cell + margin)), color=255)
    draw = ImageDraw.Draw(image)
    for label in range(10):
        y = margin + label * (cell + margin)
        draw.text((1, y + 8), str(label), fill=0)
        for column, state in enumerate(columns):
            value = generated[state]["display_views"][label, 0]
            pixels_u8 = (value.clamp(0, 1).numpy() * 255.0).round().astype("uint8")
            tile = Image.fromarray(pixels_u8, mode="L")
            x = label_width + margin + column * (cell + margin)
            image.paste(tile, (x, y))
    image.save(path)


def _arm_optimizer(model: ChenLightGen400Baseline) -> torch.optim.Adam:
    _set_decoder_trainable(model)
    return torch.optim.Adam([model.decoder_phase], lr=1.0e-3, betas=baseline.TRAIN_BETAS,
                            eps=baseline.TRAIN_EPSILON, weight_decay=0.0)


def _adam_first_step_from_gradient(
    model: ChenLightGen400Baseline, gradient: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _require(gradient.shape == model.decoder_phase.shape,
             "Adam direction probe gradient shape does not match Decoder phases")
    optimizer = _arm_optimizer(model)
    before = model.decoder_phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    model.decoder_phase.grad = gradient.detach().clone()
    optimizer.step()
    return before, model.decoder_phase.detach().clone()


def _batch_for_step(population: dict[str, Any], pixels: torch.Tensor, labels: torch.Tensor,
                    step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fit_by_class = population["fit_rows_by_class"]
    rows = fit_by_class[:, step % FIT_PER_CLASS]
    target, _ = baseline._image_batch(pixels, rows, device)
    positions = _batch_indices(population, rows)
    z = population["coefficients"].index_select(0, positions).to(device)
    encoder_power = population["encoder_power_planes_w"].index_select(0, positions).to(device)
    return rows, z, target, encoder_power


def _snapshot_training(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    step: int,
    arm: str,
    latest: dict[str, Any],
    run_id: str,
    *,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    population: dict[str, torch.Tensor],
    preflight_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "chen_lightgen_400_quickcheck_recovery.v2", "task_id": TASK_ID,
        "run_id": run_id, "arm": arm, "completed_updates": step,
        "decoder_grid": DECODER_GRID,
        "diagnostic_config": {"source_encoder_grid": 2048, "decoder_grid": DECODER_GRID,
                              "phase_shape": [400, 400], "pixel_pitch_m": 3.0e-6,
                              "logical_batch_size": BATCH_SIZE, "updates_per_arm": UPDATES,
                              "training_arm_weights": ARM_WEIGHTS[arm]},
        "preflight_dir": str(preflight_dir.resolve()), "run_dir": str(run_dir.resolve()),
        "model": baseline._state_snapshot(model), "optimizer": baseline._cpu_snapshot(optimizer.state_dict()),
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "references": {name: value.detach().cpu().clone() for name, value in references.items()},
        "bogt_weights": dict(weights),
        "fixed_population": baseline._cpu_snapshot(population),
        "data_roles": {"fit": 100, "confirmation": 200, "canonical_train": True,
                       "official_test_reads": 0},
        "latest_losses": latest,
        "sample_schedule": {"seed": SELECTION_SEED, "next_step": step,
                            "next_position_in_class_cycle": step % FIT_PER_CLASS},
    }


def _recovery_worker(input_path: Path, output_path: Path) -> None:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_quickcheck_recovery_probe.v1",
             "recovery probe input schema changed")
    device = torch.device("cuda")
    decoder_grid = int(payload["decoder_grid"])
    _require(decoder_grid == DECODER_GRID, "recovery worker profile and saved decoder grid differ")
    model = ChenLightGen400Baseline(LightGen400Config(decoder_grid, True)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = _arm_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    refs = {name: value.to(device) for name, value in payload["references"].items()}
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    _train_update(model, optimizer, payload["z"].to(device), payload["target"].to(device),
                  payload["encoder_powers"].to(device), refs, payload["weights"], payload["arm"])
    atomic_torch_save_new(output_path, {
        "model": baseline._state_snapshot(model), "optimizer": baseline._cpu_snapshot(optimizer.state_dict()),
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "next_step": payload["step"] + 1,
        "next_position_in_class_cycle": (payload["step"] + 1) % FIT_PER_CLASS,
    })


def _interrupt_probe_worker(input_path: Path, output_path: Path, ready_path: Path) -> None:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_quickcheck_signal_probe.v1",
             "signal probe input schema changed")
    device = torch.device("cuda")
    model = ChenLightGen400Baseline(LightGen400Config(int(payload["decoder_grid"]), True)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = _arm_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    references = {name: value.to(device) for name, value in payload["references"].items()}
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    requested: list[int] = []

    def handler(signum: int, frame: Any) -> None:
        del frame
        requested.append(signum)

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_handlers:
        signal.signal(sig, handler)
    try:
        atomic_write_new_json(ready_path, {"status": "ARMED", "starting_step": payload["step"]})
        while not requested:
            time.sleep(0.01)
        step = int(payload["step"])
        latest = _train_update(model, optimizer, payload["z"].to(device), payload["target"].to(device),
                               payload["encoder_powers"].to(device), references, payload["weights"],
                               payload["arm"])
        step += 1
        atomic_torch_save_new(output_path, {
            "model": baseline._state_snapshot(model),
            "optimizer": baseline._cpu_snapshot(optimizer.state_dict()),
            "rng_cpu": torch.random.get_rng_state().cpu(),
            "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
            "requested_signal": requested[-1], "completed_updates": step,
            "next_position_in_class_cycle": step % FIT_PER_CLASS, "latest_losses": latest,
        })
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)


def _resume_decision(checkpoint_arm: str, completed_updates: int, completed_arms: set[str]) -> tuple[str, str | None]:
    _require(checkpoint_arm in ARM_WEIGHTS and 0 <= completed_updates <= UPDATES,
             "explicit resume checkpoint branch or update count is invalid")
    _require(completed_arms <= set(ARM_WEIGHTS), "completed branch set contains an unknown arm")
    if checkpoint_arm == "a_reconstruction_only":
        if completed_updates < UPDATES:
            return "resume_arm", checkpoint_arm
        return "train_arm", "b_reconstruction_plus_power"
    _require("a_reconstruction_only" in completed_arms,
             "B checkpoint cannot resume before completed A endpoint is recorded")
    if completed_updates < UPDATES:
        return "resume_arm", checkpoint_arm
    return "evaluate", None


def _resume_scenario_checks() -> dict[str, Any]:
    scenarios = {
        "a_interrupted": _resume_decision("a_reconstruction_only", 143, set()),
        "a_complete_b_interrupted": _resume_decision("b_reconstruction_plus_power", 221,
                                                        {"a_reconstruction_only"}),
        "b_complete_evaluation_interrupted": _resume_decision("b_reconstruction_plus_power", UPDATES,
                                                                {"a_reconstruction_only"}),
    }
    expected = {
        "a_interrupted": ("resume_arm", "a_reconstruction_only"),
        "a_complete_b_interrupted": ("resume_arm", "b_reconstruction_plus_power"),
        "b_complete_evaluation_interrupted": ("evaluate", None),
    }
    _require(scenarios == expected, "resume routing failed an interruption boundary scenario")
    return {key: {"decision": value[0], "arm": value[1], "passed": value == expected[key]}
            for key, value in scenarios.items()}


def _preflight_signal_probe(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    population: dict[str, Any],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    arm: str,
    step: int,
    signum: int,
    output_dir: Path,
) -> dict[str, Any]:
    _, z, target, encoder_powers = _batch_for_step(population, pixels, labels, step, torch.device("cuda"))
    start = _snapshot_training(model, optimizer, step, arm, {}, "signal-probe",
                               references=references, weights=weights, population=population,
                               preflight_dir=output_dir, run_dir=output_dir)
    stem = "sigint" if signum == signal.SIGINT else "sigterm"
    input_path = output_dir / f"signal_probe_{stem}_input.pt"
    output_path = output_dir / f"signal_probe_{stem}_output.pt"
    ready_path = output_dir / f"signal_probe_{stem}_ready.json"
    atomic_torch_save_new(input_path, {
        "schema": "chen_lightgen_400_quickcheck_signal_probe.v1", "decoder_grid": DECODER_GRID,
        "model": start["model"], "optimizer": start["optimizer"], "rng_cpu": start["rng_cpu"],
        "rng_cuda": start["rng_cuda"], "references": start["references"], "weights": weights,
        "arm": arm, "step": step, "z": z.detach().cpu(), "target": target.detach().cpu(),
        "encoder_powers": encoder_powers.detach().cpu(),
    })
    command = [sys.executable, "-m", "scripts.research.lightgen.diagnose_chen_lightgen_400_quickcheck_v1",
               "--decoder-grid", str(DECODER_GRID), "--signal-worker", str(input_path),
               "--signal-worker-output", str(output_path), "--signal-ready", str(ready_path)]
    _offload_model_for_cuda_worker(model, optimizer)
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while not ready_path.is_file():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(f"signal probe exited before arming: {stdout[-1000:]} {stderr[-2000:]}")
            time.sleep(0.05)
        process.send_signal(signum)
        stdout, stderr = process.communicate()
    finally:
        _restore_model_after_cuda_worker(model, optimizer, start["optimizer"],
                                         start["rng_cpu"], start["rng_cuda"])
    baseline._assert_state_close(start["model"], baseline._state_snapshot(model),
                                 f"{stem}_parent_model_after_worker")
    baseline._assert_state_close(start["optimizer"], baseline._cpu_snapshot(optimizer.state_dict()),
                                 f"{stem}_parent_optimizer_after_worker")
    _require(process.returncode == 0, f"signal probe process failed: {stderr[-3000:]}")
    observed = torch.load(output_path, map_location="cpu", weights_only=False)
    _require(observed.get("requested_signal") == signum and observed.get("completed_updates") == step + 1
             and observed.get("next_position_in_class_cycle") == (step + 1) % FIT_PER_CLASS,
             f"{stem} did not save at the first completed update boundary")
    return {"passed": True, "signal": stem, "requested_signal_number": signum,
            "starting_updates": step, "completed_updates": observed["completed_updates"],
            "next_batch_position": observed["next_position_in_class_cycle"],
            "independent_process_exit_code": process.returncode}


def _preflight_recovery_probe(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    population: dict[str, Any],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    arm: str,
    step: int,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    rows, z, target, encoder_powers = _batch_for_step(population, pixels, labels, step, torch.device("cuda"))
    start = _snapshot_training(model, optimizer, step, arm, {}, run_id,
                               references=references, weights=weights, population=population,
                               preflight_dir=output_dir, run_dir=output_dir)
    input_path = output_dir / f"recovery_probe_{arm}.pt"
    output_path = output_dir / f"recovery_probe_{arm}_worker.pt"
    atomic_torch_save_new(input_path, {
        "schema": "chen_lightgen_400_quickcheck_recovery_probe.v1", "model": start["model"],
        "optimizer": start["optimizer"], "rng_cpu": start["rng_cpu"], "rng_cuda": start["rng_cuda"],
        "decoder_grid": DECODER_GRID,
        "references": {name: value.detach().cpu() for name, value in references.items()},
        "weights": weights, "arm": arm, "step": step,
        "z": z.detach().cpu(), "target": target.detach().cpu(),
        "encoder_powers": encoder_powers.detach().cpu(),
    })
    torch.random.set_rng_state(start["rng_cpu"])
    torch.cuda.set_rng_state_all(start["rng_cuda"])
    _train_update(model, optimizer, z, target, encoder_powers, references, weights, arm)
    expected_model = baseline._state_snapshot(model)
    expected_optimizer = baseline._cpu_snapshot(optimizer.state_dict())
    expected_rng = {"cpu": torch.random.get_rng_state().cpu(),
                    "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]}
    model.load_state_dict(start["model"], strict=True)
    optimizer.load_state_dict(start["optimizer"])
    torch.random.set_rng_state(start["rng_cpu"])
    torch.cuda.set_rng_state_all(start["rng_cuda"])
    command = [sys.executable, "-m", "scripts.research.lightgen.diagnose_chen_lightgen_400_quickcheck_v1",
               "--decoder-grid", str(DECODER_GRID), "--recovery-worker", str(input_path),
               "--worker-output", str(output_path)]
    _offload_model_for_cuda_worker(model, optimizer)
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    finally:
        _restore_model_after_cuda_worker(model, optimizer, start["optimizer"],
                                         expected_rng["cpu"], expected_rng["cuda"])
    baseline._assert_state_close(start["model"], baseline._state_snapshot(model),
                                 f"{arm}_parent_model_after_worker")
    baseline._assert_state_close(start["optimizer"], baseline._cpu_snapshot(optimizer.state_dict()),
                                 f"{arm}_parent_optimizer_after_worker")
    _require(completed.returncode == 0, f"independent recovery worker failed: {completed.stderr[-3000:]}")
    observed = torch.load(output_path, map_location="cpu", weights_only=False)
    model_error = baseline._assert_state_close(expected_model, observed["model"],
                                               f"{arm}_independent_model_recovery")
    optimizer_error = baseline._assert_state_close(expected_optimizer,
                                                   observed["optimizer"], f"{arm}_independent_optimizer_recovery")
    rng_error = baseline._assert_state_close(
        expected_rng,
        {"cpu": observed["rng_cpu"], "cuda": observed["rng_cuda"]},
        f"{arm}_independent_rng_recovery",
    )
    _require(observed.get("next_step") == step + 1, "independent recovery used the wrong next update")
    _require(observed.get("next_position_in_class_cycle") == (step + 1) % FIT_PER_CLASS,
             "independent recovery used the wrong next batch position")
    return {"passed": True, "starting_step": step, "next_step": step + 1,
            "sample_rows": rows.tolist(), "model_max_abs_error": model_error,
            "optimizer_max_abs_error": optimizer_error, "rng_max_abs_error": rng_error,
            "independent_process_exit_code": completed.returncode,
            "parent_cuda_state_offloaded_during_worker": True}


def _offload_model_for_cuda_worker(
    model: ChenLightGen400Baseline, optimizer: torch.optim.Optimizer
) -> None:
    optimizer.zero_grad(set_to_none=True)
    model.to("cpu")
    optimizer.state.clear()
    clear_production_transfer_cache()
    torch.cuda.empty_cache()


def _restore_model_after_cuda_worker(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
    rng_cpu: torch.Tensor,
    rng_cuda: list[torch.Tensor],
) -> None:
    model.to("cuda")
    optimizer.load_state_dict(optimizer_state)
    torch.random.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_cuda)


def _prepare_arm_preflight(
    source_model: ChenLightGen400Baseline,
    arm: str,
    population: dict[str, Any],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    model = copy.deepcopy(source_model).to("cuda")
    optimizer = _arm_optimizer(model)
    start_state = baseline._state_snapshot(model)
    evaluator_placeholder = None
    del evaluator_placeholder
    model.train()
    history: list[dict[str, float]] = []
    update_started = time.perf_counter()
    for step in range(3):
        _, z, target, encoder_powers = _batch_for_step(population, pixels, labels, step, torch.device("cuda"))
        history.append(_train_update(model, optimizer, z, target, encoder_powers, references, weights, arm))
    three_step_seconds = time.perf_counter() - update_started
    _require(baseline._assert_state_close(start_state, baseline._state_snapshot(source_model),
                                          f"{arm}_rolled_back_source") == 0.0,
             "preflight changed the source checkpoint model")
    gradient_model = copy.deepcopy(source_model).to("cuda")
    _arm_optimizer(gradient_model)
    _, z, target, encoder_powers = _batch_for_step(population, pixels, labels, 0, torch.device("cuda"))
    gradient_diagnostics = _objective_gradient_diagnostics(
        gradient_model, z, target, encoder_powers, references, weights, arm
    )
    del gradient_model
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    model_after = baseline._state_snapshot(source_model)
    recovery = _preflight_recovery_probe(model, optimizer, population, pixels, labels,
                                         references, weights, arm, 3, output_dir, run_id)
    signum = signal.SIGINT if arm == "a_reconstruction_only" else signal.SIGTERM
    signal_probe = _preflight_signal_probe(
        model, optimizer, population, pixels, labels, references, weights, arm, 3, signum, output_dir
    )
    _require(baseline._assert_state_close(model_after, baseline._state_snapshot(source_model),
                                          f"{arm}_preflight_source_immutable") == 0.0,
             "preflight modified the loaded source model")
    return {"three_cuda_updates": history, "all_decoder_masks_updated": True,
            "optimizer_reset_from_same_source": True, "gradient_diagnostics": gradient_diagnostics,
            "three_step_seconds": three_step_seconds,
            "seconds_per_production_update": three_step_seconds / 3.0,
            "estimated_500_step_seconds": three_step_seconds / 3.0 * UPDATES,
            "independent_recovery": recovery, "signal_safe_boundary": signal_probe}


def _check_evaluator_frozen(evaluator: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = evaluator.state_dict()
    _require(current.keys() == state.keys() and all(torch.equal(current[key], state[key]) for key in state),
             "frozen E_D parameters changed during decoder optimization")


def _load_generated(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"explicit saved generation coefficients are missing: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, dict) and value.get("schema") == "chen_lightgen_400_empirical_complex_gaussian_samples.v1",
             "saved generation coefficient artifact schema changed")
    _require(tuple(value["coefficients"].shape) == (10_000, OLS_PORT_COUNT)
             and value["coefficients"].is_complex(), "saved generation coefficient population changed")
    _require(tuple(value["requested_labels"].shape) == (10_000,), "saved generation labels changed")
    return value


def _make_identity(config: dict[str, Any], model_grid: int, run_id: str, manifest: Any) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{TASK_ID}:source_step_30000:selection_seed_{SELECTION_SEED}",
        semantic_authority_id=f"{TASK_ID}:SOURCE_STEP_30000:DECODER_GRID_{model_grid}",
        data_role="canonical_MNIST_TRAIN_fit_100_confirm_200_per_confirmation_population",
        config_identity={"source_config": config, "source_encoder_grid": 2048,
                         "diagnostic_decoder_grid": model_grid, "fit_per_class": FIT_PER_CLASS,
                         "confirm_per_class": CONFIRM_PER_CLASS, "updates_per_arm": UPDATES,
                         "local_gate": {"mse_improvement_min": LOCAL_MSE_IMPROVEMENT_MIN,
                                        "ssim_not_lower": True,
                                        "saturation_max": LOCAL_SATURATION_MAX}},
        model_identity={"source_step": 30_000, "encoder": [2, 400, 400], "ols": "gaussian_s12_v1_100_ports",
                        "decoder": [4, 400, 400], "grid": model_grid},
        optimizer_identity={"name": "Adam", "groups": ["decoder_phase"], "learning_rate": 1.0e-3,
                            "betas": baseline.TRAIN_BETAS, "epsilon": baseline.TRAIN_EPSILON,
                            "reset_at_start": True},
        sampler_identity={"seed": SELECTION_SEED, "batch_size": BATCH_SIZE,
                          "fixed_fit_rows_per_class": FIT_PER_CLASS, "updates_per_arm": UPDATES,
                          "arm_order": list(ARM_WEIGHTS)},
        execution_source_semantics_id=manifest.execution_source_semantics_id,
        source_manifest_id=manifest.source_manifest_id,
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for this fixed decoder diagnostic")
    device = torch.device("cuda")
    started = time.perf_counter()
    output_dir = _resolve(args.output_dir)
    _require(not output_dir.exists(), f"refusing to overwrite preflight output: {output_dir}")
    config_path, checkpoint_path, report_path = map(_resolve, (args.config, args.checkpoint, args.source_report))
    data_path = _resolve(args.data)
    config, checkpoint, source_report, source_model, references = _load_inputs(
        checkpoint_path, report_path, config_path, device
    )
    _require(DECODER_GRID == 2048 or args.fixed_population is not None,
             "4096 diagnostic requires the explicit saved fixed-population path")
    evaluator_path = (_resolve(args.evaluator) if args.evaluator else
                      _resolve(Path(config["evaluation"]["evaluator_checkpoint"])))
    pixels, labels, train_indices, _ = baseline._load_data(data_path)
    population, population_validation = _fixed_population_for_task(
        args, source_model, pixels, labels, train_indices, device
    )
    model = _diagnostic_model(source_model, DECODER_GRID)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_torch_save_new(output_dir / "fixed_population.pt", population)
    evaluator = baseline._load_evaluator(evaluator_path, device)
    evaluator_state = {key: value.detach().clone() for key, value in evaluator.state_dict().items()}
    torch.cuda.reset_peak_memory_stats(device)
    source_2048_metrics = _evaluate(source_model, population, pixels, labels, evaluator, references, device,
                                    include_unit_readout=True)
    source_2048_generated = _evaluate_generated(source_model, population, evaluator, references, device)
    baseline_metrics = _evaluate(model, population, pixels, labels, evaluator, references, device,
                                 include_unit_readout=True)
    generated_baseline = _evaluate_generated(model, population, evaluator, references, device)
    windows = _window_checks(model, population, pixels, labels, config, device)
    atomic_write_new_json(output_dir / "window_checks.json", baseline._json_safe(windows))
    if not windows["passed"]:
        failure = {
            "status": "INVALID_HARNESS", "task_id": TASK_ID,
            "stage": "fixed_coefficient_decoder_window", "source_checkpoint_step": 30_000,
            "error": "4096 Decoder window failed 1e-4 convergence against 6144",
            "completed_scientific_updates": 0, "official_test_reads": 0,
            "window_checks": windows,
            "decision": "repair_numerical_path_and_repeat_preflight_without_automatic_grid_migration_or_tolerance_relaxation",
        }
        atomic_write_new_json(output_dir / "preflight_failure.json", baseline._json_safe(failure))
        raise RuntimeError(failure["error"])
    negative_rows = population["fit_rows_by_class"][:, 0]
    negative_positions = _batch_indices(population, negative_rows)
    negative_target, _ = baseline._image_batch(pixels, negative_rows, device)
    negative_result = _latent_result(
        model, population["coefficients"].index_select(0, negative_positions).to(device),
        population["encoder_power_planes_w"].index_select(0, negative_positions).to(device), diagnostics=False
    )
    negative_prediction, _ = _unit_outputs(negative_result["physical_w_per_pixel"], references["camera_gain"])
    negative = _validate_pairing_negative_control(
        negative_prediction, negative_target, negative_rows, labels[negative_rows]
    )
    fit_only_gain = population_validation["fit_only_gain_w_inverse"]
    fixed_gain = references["camera_gain"].detach().cpu()
    confirm_target, _ = baseline._image_batch(pixels, population["confirm_rows"], device)
    confirm_source = _physical_from_ols_in_batches(
        model, population["coefficients"][100:].to(device),
        population["encoder_power_planes_w"][100:].to(device), batch_size=BATCH_SIZE
    )
    alternative_raw = confirm_source * fit_only_gain
    fixed_raw = confirm_source * fixed_gain.to(device=device, dtype=confirm_source.dtype)
    gain_counterfactual = {
        **population_validation,
        "source_confirm_mse400_with_frozen_gain": float((fixed_raw - confirm_target).square().mean().cpu()),
        "source_confirm_mse400_with_fit_only_gain": float((alternative_raw - confirm_target).square().mean().cpu()),
        "used_for_training": False,
        "fit_population": 100,
        "confirmation_population": 200,
    }
    weights = {key: float(value) for key, value in checkpoint["bogt_weights"].items()}
    preflight_model_state = baseline._state_snapshot(model)
    run_id = uuid.uuid4().hex
    relevant_paths = [config_path, Path(__file__),
                      Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_baseline.py"),
                      Path("scripts/research/lightgen/execute_chen_lightgen_400_mnist_baseline_v1.py"),
                      Path("src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py"),
                      Path("src/ldm_project/v2/optics/propagation.py")]
    if args.fixed_population is not None:
        relevant_paths.append(_resolve(args.fixed_population))
    if args.generated_latents is not None:
        relevant_paths.append(_resolve(args.generated_latents))
    manifest = capture_execution_source_manifest(
        ROOT, task_id=TASK_ID, semantic_authority_id=f"{TASK_ID}:SOURCE_STEP_30000:DECODER_GRID_{DECODER_GRID}",
        relevant_paths=tuple(relevant_paths),
        config_identity={"task": TASK_ID, "selection_seed": SELECTION_SEED, "updates": UPDATES,
                         "source_encoder_grid": 2048, "decoder_grid": DECODER_GRID},
    )
    identity = _make_identity(config, DECODER_GRID, run_id, manifest)
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    arm_preflight: dict[str, Any] = {}
    for arm in ARM_WEIGHTS:
        arm_preflight[arm] = _prepare_arm_preflight(
            model, arm, population, pixels, labels, references, weights, output_dir, run_id
        )
    _check_evaluator_frozen(evaluator, evaluator_state)
    _require(baseline._assert_state_close(preflight_model_state, baseline._state_snapshot(model),
                                          "preflight_source_model_immutable") == 0.0,
             "preflight changed the source model")
    independent_physics = baseline._independent_physics_checks(model)
    resume_scenarios = _resume_scenario_checks()
    signal_checks_passed = all(arm_preflight[arm]["signal_safe_boundary"]["passed"] for arm in ARM_WEIGHTS)
    recovery_checks_passed = all(arm_preflight[arm]["independent_recovery"]["passed"] for arm in ARM_WEIGHTS)
    peak_vram = torch.cuda.max_memory_reserved(device)
    elapsed = time.perf_counter() - started
    preflight_update_counts = {
        arm: {"three_step_cuda_update": 3, "gradient_direction_probe": 1,
              "independent_recovery_continuous_and_worker": 2, "signal_boundary_probe": 1,
              "total_preflight_updates_not_science": 7}
        for arm in ARM_WEIGHTS
    }
    estimated_training_seconds = sum(arm_preflight[arm]["estimated_500_step_seconds"] for arm in ARM_WEIGHTS)
    selected_engine = f"cuda_400x400_decoder_grid_{DECODER_GRID}_ols_latent_batch10"
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "source_checkpoint_step": 30_000,
                     "selected_rows": {"fit": population["fit_rows"].tolist(),
                                       "confirmation": population["confirm_rows"].tolist()},
                     "fixed_population_consumed_directly": args.fixed_population is not None,
                     "source_encoder_grid": 2048, "decoder_grid": DECODER_GRID,
                     "production_window_decoder_grid_passed": windows["passed"],
                     "window_tolerance": WINDOW_TOLERANCE,
                     "shared_image_ols_path_passed": population_validation.get(
                         "max_abs_image_vs_saved_ols_readout_error_w_per_m2", 0.0
                     ) <= 1.0e-8,
                     "pairing_negative_control_passed": negative["passed"],
                     "three_cuda_updates_and_four_decoder_masks_passed": True,
                     "independent_recovery_passed": recovery_checks_passed,
                     "independent_signal_safe_boundary_checks_passed": signal_checks_passed,
                     "independent_gaussian_and_propagation_checks_passed": bool(independent_physics),
                     "all_resume_scenarios_passed": all(value["passed"] for value in resume_scenarios.values()),
                     "evaluator_frozen": True},
        resource={"passed": True, "cuda_device": torch.cuda.get_device_name(device),
                  "peak_reserved_vram_bytes": peak_vram, "preflight_seconds": elapsed},
        implementation={"passed": True, "engine": selected_engine, "selected_engine": selected_engine,
                        "arms": {"a": list(ARM_WEIGHTS["a_reconstruction_only"]),
                                 "b": list(ARM_WEIGHTS["b_reconstruction_plus_power"])}},
        selected_engine=selected_engine,
        resource_envelope={"device": torch.cuda.get_device_name(device),
                           "peak_reserved_vram_bytes": peak_vram},
    )
    control.mark_preflight_ready(receipt)
    preflight = {
        "schema": "chen_lightgen_400_quickcheck_preflight.v2", "status": "READY",
        "task_id": TASK_ID, "authorization_id": AUTHORITY, "run_identity": identity.to_dict(),
        "receipt": receipt.to_dict(), "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_step": checkpoint["completed_updates"], "source_baseline_status": source_report["status"],
        "source_report": str(report_path), "config": str(config_path), "data": str(data_path),
        "generated_latents": str(_resolve(args.generated_latents)) if args.generated_latents is not None else None,
        "fixed_population": str(_resolve(args.fixed_population)) if args.fixed_population is not None else None,
        "evaluator": str(evaluator_path),
        "source_encoder_grid": 2048, "decoder_diagnostic_grid": DECODER_GRID,
        "data_roles": {"fit": 100, "confirmation": 200, "both_canonical_train": True,
                        "official_test_reads": 0},
        "selection_seed": SELECTION_SEED, "population": {
            "fit_rows_by_class": population["fit_rows_by_class"].tolist(),
            "confirmation_rows_by_class": population["confirm_rows_by_class"].tolist()},
        "frozen_references": {"scale": references["scale"].cpu().tolist(),
                              "power_reference_w": references["power_reference"].cpu().tolist(),
                              "camera_gain_w_inverse": float(references["camera_gain"].cpu())},
        "bogt_weights": weights, "fixed_population_gain_counterfactual": gain_counterfactual,
        "source_2048_metrics": source_2048_metrics["summary"],
        "decoder_4096_start_metrics": baseline_metrics["summary"],
        "saved_generation_metrics": {key: value for key, value in generated_baseline.items()
                                     if key not in {"unit_readout", "display_views"}},
        "source_2048_generated_metrics": {key: value for key, value in source_2048_generated.items()
                                          if key not in {"unit_readout", "display_views"}},
        "unit_contract": baseline_metrics["unit_readout"]["units"],
        "unit_example_rows": baseline_metrics["unit_readout"]["row_indices"],
        "window_checks": windows, "negative_controls": {"shuffled_reconstruction_target_rows": negative},
        "population_source_validation": population_validation,
        "independent_physics_checks": independent_physics,
        "gradient_diagnostics": {arm: value["gradient_diagnostics"] for arm, value in arm_preflight.items()},
        "three_step_and_recovery": arm_preflight,
        "preflight_update_counts_by_arm": preflight_update_counts,
        "estimated_A_plus_B_training_seconds_from_three_step_speed": estimated_training_seconds,
        "resume_scenarios": resume_scenarios,
        "constant_l_dist_on_fit_population": _initial_latent_kl(population, references["scale"], device),
        "peak_reserved_vram_bytes": peak_vram, "elapsed_seconds": elapsed,
        "official_test_reads": 0,
    }
    atomic_write_new_json(output_dir / "preflight.json", baseline._json_safe(preflight))
    atomic_write_new_json(output_dir / "preflight_ready.json", baseline._json_safe(receipt.to_dict()))
    atomic_torch_save_new(output_dir / "source_2048_model_state.pt", baseline._state_snapshot(source_model))
    atomic_torch_save_new(output_dir / "decoder_4096_start_model_state.pt", baseline._state_snapshot(model))
    atomic_torch_save_new(output_dir / "source_frozen_references.pt",
                          {name: value.detach().cpu() for name, value in references.items()})
    atomic_torch_save_new(output_dir / "source_2048_unit_readout.pt", source_2048_metrics["unit_readout"])
    atomic_torch_save_new(output_dir / "decoder_4096_start_unit_readout.pt", baseline_metrics["unit_readout"])
    atomic_torch_save_new(output_dir / "source_2048_generated_unit_readout.pt", source_2048_generated["unit_readout"])
    atomic_torch_save_new(output_dir / "decoder_4096_start_generated_unit_readout.pt",
                          generated_baseline["unit_readout"])
    return preflight


def _read_preflight(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], ResearchRunIdentity, dict[str, Any]]:
    preflight_dir = _resolve(path)
    preflight = _load_json(preflight_dir / "preflight.json")
    receipt = _load_json(preflight_dir / "preflight_ready.json")
    _require(preflight.get("schema") == "chen_lightgen_400_quickcheck_preflight.v2"
             and preflight.get("status") == "READY" and preflight.get("task_id") == TASK_ID,
             "an explicit READY quickcheck preflight is required")
    _require(preflight.get("source_encoder_grid") == 2048
             and preflight.get("decoder_diagnostic_grid") == DECODER_GRID,
             "READY preflight uses a different encoder or diagnostic decoder grid")
    _require(receipt == preflight.get("receipt"), "preflight READY receipt differs from its report")
    _require(receipt.get("authorization_id") == AUTHORITY and receipt.get("status") == "READY",
             "preflight receipt belongs to another authorization or is not READY")
    identity = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    for key, argument in (("source_checkpoint", args.checkpoint), ("source_report", args.source_report),
                          ("config", args.config), ("data", args.data)):
        _require(Path(preflight[key]).resolve() == _resolve(argument).resolve(),
                 f"run input does not match READY preflight: {key}")
    if preflight.get("generated_latents") is not None:
        _require(args.generated_latents is not None
                 and Path(preflight["generated_latents"]).resolve() == _resolve(args.generated_latents).resolve(),
                 "run generated coefficient input does not match READY preflight")
    if preflight.get("fixed_population") is not None:
        _require(args.fixed_population is not None
                 and Path(preflight["fixed_population"]).resolve() == _resolve(args.fixed_population).resolve(),
                 "run fixed population does not match READY preflight")
    _require(identity.model_identity.get("grid") == DECODER_GRID,
             "READY run identity has a different diagnostic decoder grid")
    if args.evaluator is not None:
        _require(Path(preflight["evaluator"]).resolve() == _resolve(args.evaluator).resolve(),
                 "run evaluator does not match READY preflight")
    return preflight, identity, receipt


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(baseline._json_safe(record), ensure_ascii=False) + "\n")
        stream.flush()


def _validate_arm_optimizer(optimizer: torch.optim.Optimizer, completed_updates: int) -> None:
    _require(isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1,
             "resume optimizer is not a single-group Adam")
    group = optimizer.param_groups[0]
    _require(len(group["params"]) == 1 and group["lr"] == 1.0e-3
             and tuple(group["betas"]) == baseline.TRAIN_BETAS and group["eps"] == baseline.TRAIN_EPSILON
             and group["weight_decay"] == 0.0 and group["amsgrad"] is False,
             "resume Adam configuration differs from the approved Decoder diagnostic")
    _require(len(optimizer.state) == (0 if completed_updates == 0 else 1),
             "resume Adam moment state count differs from the update cursor")
    for parameter, state in optimizer.state.items():
        step = state.get("step")
        step_value = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        _require(step_value == completed_updates, "resume Adam counter differs from the branch update cursor")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            _require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                     and torch.isfinite(value).all().item(), f"resume Adam {key} state is invalid")


def _validate_training_payload(
    payload: dict[str, Any],
    arm: str,
    run_id: str,
    run_dir: Path,
    preflight_dir: Path,
    source_state: dict[str, torch.Tensor],
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    population: dict[str, torch.Tensor],
) -> int:
    _require(payload.get("schema") == "chen_lightgen_400_quickcheck_recovery.v2"
             and payload.get("task_id") == TASK_ID and payload.get("run_id") == run_id
             and payload.get("arm") == arm and payload.get("decoder_grid") == DECODER_GRID,
             "explicit recovery checkpoint belongs to a different task, run, arm, or decoder grid")
    _require(Path(payload.get("run_dir", "")).resolve() == run_dir.resolve()
             and Path(payload.get("preflight_dir", "")).resolve() == preflight_dir.resolve(),
             "explicit recovery checkpoint belongs to different output or READY preflight paths")
    expected_config = {"source_encoder_grid": 2048, "decoder_grid": DECODER_GRID,
                       "phase_shape": [400, 400], "pixel_pitch_m": 3.0e-6,
                       "logical_batch_size": BATCH_SIZE, "updates_per_arm": UPDATES,
                       "training_arm_weights": ARM_WEIGHTS[arm]}
    _require(payload.get("diagnostic_config") == expected_config,
             "explicit recovery checkpoint diagnostic configuration changed")
    _require(payload.get("data_roles") == {"fit": 100, "confirmation": 200, "canonical_train": True,
                                           "official_test_reads": 0},
             "explicit recovery checkpoint data roles changed")
    _require(payload.get("bogt_weights") == weights, "explicit recovery checkpoint BOGT weights changed")
    saved_references = payload.get("references")
    _require(isinstance(saved_references, dict) and saved_references.keys() == references.keys()
             and all(torch.equal(saved_references[key], references[key].detach().cpu()) for key in references),
             "explicit recovery checkpoint frozen references changed")
    saved_population = payload.get("fixed_population")
    _require(isinstance(saved_population, dict) and saved_population.keys() == population.keys(),
             "explicit recovery checkpoint fixed-population keys changed")
    _require(baseline._assert_state_close(population, saved_population,
                                          "explicit_recovery_fixed_population") == 0.0,
             "explicit recovery checkpoint fixed population changed")
    step = payload.get("completed_updates")
    _require(isinstance(step, int) and 0 <= step <= UPDATES, "explicit recovery update cursor is invalid")
    schedule = payload.get("sample_schedule", {})
    _require(schedule == {"seed": SELECTION_SEED, "next_step": step,
                          "next_position_in_class_cycle": step % FIT_PER_CLASS},
             "explicit recovery next-batch cursor changed")
    model_state = payload.get("model")
    _require(isinstance(model_state, dict) and model_state.keys() == source_state.keys(),
             "explicit recovery model state keys changed")
    frozen = {key: value for key, value in source_state.items() if key != "decoder_phase"}
    frozen_now = {key: value for key, value in model_state.items() if key != "decoder_phase"}
    _require(baseline._assert_state_close(frozen, frozen_now, "explicit_recovery_frozen_model") == 0.0,
             "explicit recovery checkpoint changed frozen Encoder or OLS geometry")
    return step


def _retain_arm_checkpoints(arm_dir: Path) -> None:
    candidates = list(arm_dir.glob("checkpoint_step_*.pt")) + list(arm_dir.glob("checkpoint_signal_step_*.pt"))
    if len(candidates) <= 3:
        return
    def step_of(path: Path) -> int:
        match = __import__("re").search(r"step_(\d{6})\.pt$", path.name)
        _require(match is not None, f"unrecognized checkpoint name: {path.name}")
        return int(match.group(1))
    ordered = sorted(candidates, key=step_of)
    keep = set(ordered[-2:])
    endpoint = arm_dir / "checkpoint_step_000500.pt"
    if endpoint.is_file():
        keep.add(endpoint)
    for path in ordered:
        if path not in keep:
            path.unlink()


def _save_arm_checkpoint(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    step: int,
    arm: str,
    latest: dict[str, Any],
    run_id: str,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    population: dict[str, torch.Tensor],
    preflight_dir: Path,
    run_dir: Path,
    *,
    signaled: bool = False,
) -> Path:
    arm_dir = run_dir / arm
    label = "checkpoint_signal" if signaled else "checkpoint_step"
    path = arm_dir / f"{label}_{step:06d}.pt"
    atomic_torch_save_new(path, _snapshot_training(
        model, optimizer, step, arm, latest, run_id, references=references, weights=weights,
        population=population, preflight_dir=preflight_dir, run_dir=run_dir,
    ))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("completed_updates") == step and payload.get("arm") == arm
             and payload.get("decoder_grid") == DECODER_GRID,
             "saved recovery checkpoint failed its step/branch/grid readback")
    _require(baseline._assert_state_close(baseline._state_snapshot(model), payload["model"],
                                          f"{arm}_saved_checkpoint_model") == 0.0,
             "saved recovery checkpoint model differs from the live branch")
    _require(baseline._assert_state_close(baseline._cpu_snapshot(optimizer.state_dict()), payload["optimizer"],
                                          f"{arm}_saved_checkpoint_optimizer") == 0.0,
             "saved recovery checkpoint Adam differs from the live branch")
    _require(torch.equal(torch.random.get_rng_state(), payload["rng_cpu"])
             and all(torch.equal(current.cpu(), saved) for current, saved in
                     zip(torch.cuda.get_rng_state_all(), payload["rng_cuda"], strict=True)),
             "saved recovery checkpoint RNG differs from the live branch")
    _retain_arm_checkpoints(arm_dir)
    return path


def _load_arm_checkpoint(
    path: Path,
    arm: str,
    source_model: ChenLightGen400Baseline,
    run_id: str,
    run_dir: Path,
    preflight_dir: Path,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    population: dict[str, torch.Tensor],
) -> tuple[ChenLightGen400Baseline, torch.optim.Optimizer, dict[str, Any], int]:
    _require(path.is_file() and path.resolve().parent == (run_dir / arm).resolve(),
             "explicit resume checkpoint is missing or outside its branch directory")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state = baseline._state_snapshot(source_model)
    step = _validate_training_payload(payload, arm, run_id, run_dir, preflight_dir,
                                      source_state, references, weights, population)
    model = copy.deepcopy(source_model).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    optimizer = _arm_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    _validate_arm_optimizer(optimizer, step)
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return model, optimizer, payload, step


def _train_arm(
    arm: str,
    source_model: ChenLightGen400Baseline,
    population: dict[str, torch.Tensor],
    pixels: torch.Tensor,
    labels: torch.Tensor,
    evaluator: torch.nn.Module,
    references: dict[str, torch.Tensor],
    weights: dict[str, float],
    run_dir: Path,
    run_id: str,
    preflight_dir: Path,
    resume_checkpoint: Path | None = None,
) -> tuple[ChenLightGen400Baseline | None, dict[str, Any]]:
    arm_dir = run_dir / arm
    if resume_checkpoint is None:
        arm_dir.mkdir(parents=False, exist_ok=False)
        torch.manual_seed(SELECTION_SEED)
        torch.cuda.manual_seed_all(SELECTION_SEED)
        model = copy.deepcopy(source_model).to("cuda")
        optimizer = _arm_optimizer(model)
        start_step = 0
        latest: dict[str, Any] = {}
    else:
        _require(arm_dir.is_dir(), f"resume branch directory is missing: {arm_dir}")
        model, optimizer, payload, start_step = _load_arm_checkpoint(
            resume_checkpoint, arm, source_model, run_id, run_dir, preflight_dir,
            references, weights, population,
        )
        latest = payload.get("latest_losses", {})
    _validate_arm_optimizer(optimizer, start_step)
    model.train()
    evaluator_state = {key: value.detach().clone() for key, value in evaluator.state_dict().items()}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_heartbeat = started
    requested_signal: list[int] = []
    previous_handlers = {sig: signal.getsignal(sig)
                         for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT))}

    def handler(signum: int, frame: Any) -> None:
        del frame
        requested_signal.append(signum)

    for sig in previous_handlers:
        signal.signal(sig, handler)
    _append_jsonl(run_dir / "segments.jsonl", {
        "event": "training_segment_started", "arm": arm, "start_step": start_step,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "timestamp_unix": time.time(),
    })
    try:
        for cursor in range(start_step, UPDATES):
            rows, z, target, encoder_powers = _batch_for_step(
                population, pixels, labels, cursor, torch.device("cuda")
            )
            latest = _train_update(model, optimizer, z, target, encoder_powers, references, weights, arm)
            step = cursor + 1
            if step % 100 == 0:
                record = {"step": step, "rows": rows.tolist(), **latest}
                history.append(record)
                _append_jsonl(arm_dir / "training_history.jsonl", record)
            elapsed = time.perf_counter() - started
            if step % 100 == 0 or elapsed - (last_heartbeat - started) >= 60.0:
                checkpoint = None
                if step % 100 == 0:
                    checkpoint = _save_arm_checkpoint(
                        model, optimizer, step, arm, latest, run_id, references, weights, population,
                        preflight_dir, run_dir,
                    )
                now = time.perf_counter()
                atomic_write_json(arm_dir / "progress.json", {
                    "arm": arm, "completed_updates": step, "total_updates": UPDATES,
                    "latest_losses": latest, "segment_seconds": now - started,
                    "seconds_per_update_in_segment": (now - started) / max(1, step - start_step),
                    "estimated_remaining_seconds": (UPDATES - step)
                    * (now - started) / max(1, step - start_step),
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved("cuda"),
                    "checkpoint": str(checkpoint.relative_to(run_dir)) if checkpoint else None,
                })
                atomic_write_json(run_dir / "progress.json", {
                    "status": "RUNNING", "current_arm": arm,
                    "current_task_local_update": step, "current_optimizer_step": step,
                    "current_seed": SELECTION_SEED,
                    "completed_updates_by_arm": {
                        name: UPDATES if (run_dir / name / "arm_completion.json").is_file()
                        else (step if name == arm else 0) for name in ARM_WEIGHTS
                    },
                    "explicit_resume_checkpoint": str(checkpoint) if checkpoint else None,
                })
                last_heartbeat = now
            if requested_signal:
                path = _save_arm_checkpoint(
                    model, optimizer, step, arm, latest, run_id, references, weights, population,
                    preflight_dir, run_dir, signaled=True,
                )
                event = {"event": "training_interrupted", "status": "INTERRUPTED", "signal": requested_signal[-1],
                         "arm": arm, "completed_updates": step,
                         "checkpoint": str(path.relative_to(run_dir)), "timestamp_unix": time.time()}
                _append_jsonl(run_dir / "interruption_events.jsonl", event)
                atomic_write_json(run_dir / "progress.json", {
                    "status": "INTERRUPTED", "current_arm": arm,
                    "current_task_local_update": step, "current_optimizer_step": step,
                    "current_seed": SELECTION_SEED,
                    "resume_checkpoint": str(path.relative_to(run_dir)),
                })
                _append_jsonl(run_dir / "segments.jsonl", {"event": "training_segment_interrupted", **event})
                return None, {"status": "INTERRUPTED", "completed_updates": step,
                              "checkpoint": str(path.relative_to(run_dir)), "history": history,
                              "signal": requested_signal[-1]}
    finally:
        for sig, old in previous_handlers.items():
            signal.signal(sig, old)
    _check_evaluator_frozen(evaluator, evaluator_state)
    endpoint = arm_dir / "checkpoint_step_000500.pt"
    _require(endpoint.is_file(), "500-step branch endpoint checkpoint was not saved")
    terminal, _, payload, terminal_step = _load_arm_checkpoint(
        endpoint, arm, source_model, run_id, run_dir, preflight_dir, references, weights, population
    )
    _require(terminal_step == UPDATES and payload.get("completed_updates") == UPDATES,
             "500-step endpoint checkpoint failed step and optimizer readback")
    completion = {"status": "COMPLETED", "arm": arm, "completed_updates": UPDATES,
                  "endpoint_checkpoint": str(endpoint.relative_to(run_dir)),
                  "next_batch_position": UPDATES % FIT_PER_CLASS}
    if not (arm_dir / "arm_completion.json").is_file():
        atomic_write_new_json(arm_dir / "arm_completion.json", completion)
    _append_jsonl(run_dir / "segments.jsonl", {
        "event": "training_segment_completed", "arm": arm, "start_step": start_step,
        "completed_updates": UPDATES, "timestamp_unix": time.time(),
    })
    return terminal, {"status": "COMPLETED", "completed_updates": UPDATES,
                      "history": history, "endpoint_checkpoint": str(endpoint.relative_to(run_dir)),
                      "latest_losses": payload.get("latest_losses", latest),
                      "elapsed_seconds": time.perf_counter() - started}


def _write_units(path: Path, payloads: dict[str, dict[str, Any]], generation: dict[str, dict[str, Any]]) -> None:
    recon = {name: value["unit_readout"] for name, value in payloads.items()}
    gen = {}
    for name, value in generation.items():
        _require(isinstance(value.get("unit_readout"), dict),
                 f"generation result has no unit readout for {name}")
        gen[name] = value["unit_readout"]
    reconstruction_path = path / "unit_readout_reconstruction.pt"
    generation_path = path / "unit_readout_generation.pt"
    atomic_torch_save_new(reconstruction_path, {
        "schema": "chen_lightgen_400_quickcheck_units.v1", "by_state": recon,
    })
    atomic_torch_save_new(generation_path, {
        "schema": "chen_lightgen_400_quickcheck_units.v1", "by_state": gen,
    })
    saved_reconstruction = torch.load(reconstruction_path, map_location="cpu", weights_only=False)
    saved_generation = torch.load(generation_path, map_location="cpu", weights_only=False)
    _require(saved_reconstruction.get("schema") == "chen_lightgen_400_quickcheck_units.v1"
             and saved_generation.get("schema") == "chen_lightgen_400_quickcheck_units.v1",
             "saved unit outputs failed schema readback")
    for collection in (saved_reconstruction["by_state"], saved_generation["by_state"]):
        for name, values in collection.items():
            physical = values["physical_w_per_pixel"]
            task = values["task_intensity_unclipped"]
            display = values["display_intensity_clipped"]
            gain = float(values["camera_gain_w_inverse"])
            _require(torch.isfinite(physical).all().item() and torch.isfinite(task).all().item()
                     and torch.isfinite(display).all().item() and math.isfinite(gain) and gain > 0,
                     f"saved unit readout is nonfinite for {name}")
            _require(torch.allclose(task, physical * gain, rtol=1.0e-6, atol=1.0e-12),
                     f"saved task intensity does not equal physical W/pixel times gain for {name}")
            _require(torch.equal(display, task.clamp(0.0, 1.0)),
                     f"saved display intensity is not the clipped task intensity for {name}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the fixed decoder diagnostic")
    device = torch.device("cuda")
    run_dir = _resolve(args.output_dir)
    _require(args.mode in ("run", "resume"), "run executor received an unsupported mode")
    _require(not (run_dir / "report.json").exists(), "completed diagnostic report already exists")
    if args.mode == "run":
        _require(not run_dir.exists(), f"refusing to overwrite quickcheck output: {run_dir}")
        _require(args.resume_checkpoint is None, "fresh run cannot consume a recovery checkpoint")
    else:
        _require(run_dir.is_dir(), f"resume run directory is missing: {run_dir}")
        _require(args.resume_checkpoint is not None, "resume mode requires explicit --resume-checkpoint")
    preflight, identity, receipt = _read_preflight(args.preflight_dir, args)
    checkpoint_path, report_path, config_path, data_path = map(
        _resolve, (args.checkpoint, args.source_report, args.config, args.data)
    )
    config, checkpoint, source_report, source_model, references = _load_inputs(
        checkpoint_path, report_path, config_path, device
    )
    evaluator_path = (_resolve(args.evaluator) if args.evaluator else
                      _resolve(Path(config["evaluation"]["evaluator_checkpoint"])))
    _require(evaluator_path.resolve() == Path(preflight["evaluator"]).resolve(),
             "run evaluator does not match READY preflight")
    pixels, labels, train_indices, _ = baseline._load_data(data_path)
    if args.fixed_population is not None:
        population = _load_fixed_population(_resolve(args.fixed_population), labels, train_indices)
    else:
        _require(DECODER_GRID == 2048 and args.generated_latents is not None,
                 "this run requires its explicit fixed-population artifact")
        population, _ = _fixed_population_for_task(args, source_model, pixels, labels, train_indices, device)
    saved_population = torch.load(_resolve(args.preflight_dir) / "fixed_population.pt",
                                  map_location="cpu", weights_only=False)
    _require(isinstance(saved_population, dict) and saved_population.keys() == population.keys(),
             "READY preflight fixed population keys differ from the explicit population")
    _require(baseline._assert_state_close(population, saved_population,
                                          "source_fixed_population_matches_ready_preflight") == 0.0,
             "explicit fixed population differs from the READY preflight population")
    source_state = torch.load(_resolve(args.preflight_dir) / "source_2048_model_state.pt",
                              map_location="cpu", weights_only=False)
    baseline._assert_state_close(baseline._state_snapshot(source_model), source_state,
                                 "source_checkpoint_matches_ready_preflight")
    diagnostic_start = _diagnostic_model(source_model, DECODER_GRID)
    start_state = torch.load(_resolve(args.preflight_dir) / "decoder_4096_start_model_state.pt",
                             map_location="cpu", weights_only=False)
    baseline._assert_state_close(baseline._state_snapshot(diagnostic_start), start_state,
                                 "decoder_grid_start_matches_ready_preflight")
    saved_references = torch.load(_resolve(args.preflight_dir) / "source_frozen_references.pt",
                                  map_location="cpu", weights_only=False)
    _require(saved_references.keys() == checkpoint["references"].keys()
             and all(torch.equal(saved_references[name], checkpoint["references"][name])
                     for name in saved_references),
             "frozen references differ from READY preflight")
    _require(torch.equal(population["fit_rows"], saved_population["fit_rows"])
             and torch.equal(population["confirm_rows"], saved_population["confirm_rows"]),
             "run row population differs from READY preflight")
    _require(torch.allclose(population["coefficients"], saved_population["coefficients"], rtol=1.0e-6, atol=1.0e-8)
             and torch.allclose(population["encoder_power_planes_w"], saved_population["encoder_power_planes_w"],
                                rtol=1.0e-6, atol=1.0e-10),
             "run latent inputs differ from READY preflight")
    _require(torch.equal(population["generated_coefficients"], saved_population["generated_coefficients"]),
             "saved generated OLS coefficients differ from READY preflight")
    population = saved_population
    references = {name: value.to(device) for name, value in checkpoint["references"].items()}
    weights = {name: float(value) for name, value in checkpoint["bogt_weights"].items()}
    evaluator = baseline._load_evaluator(evaluator_path, device)
    evaluator_state = {key: value.detach().clone() for key, value in evaluator.state_dict().items()}
    run_id = identity.run_id
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    if args.mode == "run":
        identity = control.open_scientific_attempt(run_dir, attempt_id="attempt_01", receipt=receipt)
    else:
        identity = control.resume_scientific_attempt(run_dir, attempt_id="attempt_01", receipt=receipt)
    manifest_path = run_dir / "run_manifest.json"
    expected_manifest = {
        "schema": "chen_lightgen_400_quickcheck_run_manifest.v2", "task_id": TASK_ID,
        "run_id": run_id, "authorization_id": AUTHORITY, "source_encoder_grid": 2048,
        "decoder_diagnostic_grid": DECODER_GRID, "preflight_dir": str(_resolve(args.preflight_dir)),
        "source_checkpoint": str(checkpoint_path), "source_report": str(report_path),
        "config": str(config_path), "data": str(data_path),
        "fixed_population": str(_resolve(args.fixed_population)) if args.fixed_population is not None else None,
        "generated_latents": str(_resolve(args.generated_latents)) if args.generated_latents is not None else None,
    }
    if args.mode == "run":
        atomic_write_new_json(manifest_path, expected_manifest)
        atomic_torch_save_new(run_dir / "fixed_population.pt", population)
        _append_jsonl(run_dir / "segments.jsonl", {"event": "run_started", "timestamp_unix": time.time()})
    else:
        _require(manifest_path.is_file() and _load_json(manifest_path) == expected_manifest,
                 "existing run manifest does not match this explicit resume request")
    resume_path = _resolve(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    if resume_path is not None:
        _require(resume_path.is_file(), f"explicit recovery checkpoint is missing: {resume_path}")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    arms: dict[str, Any] = {}
    terminal_models: dict[str, ChenLightGen400Baseline] = {}

    def load_completed_arm(arm: str) -> ChenLightGen400Baseline:
        completion_path = run_dir / arm / "arm_completion.json"
        _require(completion_path.is_file(), f"completed branch endpoint record is missing for {arm}")
        completion = _load_json(completion_path)
        _require(completion.get("status") == "COMPLETED" and completion.get("arm") == arm
                 and completion.get("completed_updates") == UPDATES,
                 f"completed branch endpoint record is invalid for {arm}")
        endpoint = run_dir / completion["endpoint_checkpoint"]
        model, _, payload, step = _load_arm_checkpoint(
            endpoint, arm, diagnostic_start, run_id, run_dir, _resolve(args.preflight_dir),
            references, weights, population,
        )
        _require(step == UPDATES and payload.get("completed_updates") == UPDATES,
                 f"completed branch endpoint state is not the fixed terminal step for {arm}")
        return model

    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        resume_arm = resume_payload.get("arm")
        completed_arms = {arm for arm in ARM_WEIGHTS
                          if (run_dir / arm / "arm_completion.json").is_file()}
        decision, next_arm = _resume_decision(
            resume_arm, int(resume_payload.get("completed_updates", -1)), completed_arms
        )
        _append_jsonl(run_dir / "segments.jsonl", {
            "event": "resume_started", "resume_checkpoint": str(resume_path),
            "resume_decision": decision, "arm": resume_arm,
            "completed_updates": resume_payload.get("completed_updates"), "timestamp_unix": time.time(),
        })
        if resume_arm == "b_reconstruction_plus_power":
            terminal_models["a_reconstruction_only"] = load_completed_arm("a_reconstruction_only")
        current, status = _train_arm(
            resume_arm, diagnostic_start, population, pixels, labels, evaluator, references, weights,
            run_dir, run_id, _resolve(args.preflight_dir), resume_checkpoint=resume_path,
        )
        arms[resume_arm] = status
        if current is None:
            control.interrupted()
            return {"status": "INTERRUPTED", "resume_checkpoint": status["checkpoint"]}
        terminal_models[resume_arm] = current
        if resume_arm == "a_reconstruction_only":
            next_arm = "b_reconstruction_plus_power"
        if next_arm == "b_reconstruction_plus_power":
            current, status = _train_arm(
                next_arm, diagnostic_start, population, pixels, labels, evaluator, references, weights,
                run_dir, run_id, _resolve(args.preflight_dir),
            )
            arms[next_arm] = status
            if current is None:
                control.interrupted()
                return {"status": "INTERRUPTED", "resume_checkpoint": status["checkpoint"]}
            terminal_models[next_arm] = current
        elif decision == "evaluate":
            terminal_models["b_reconstruction_plus_power"] = current
    else:
        for arm in ARM_WEIGHTS:
            current, status = _train_arm(
                arm, diagnostic_start, population, pixels, labels, evaluator, references, weights,
                run_dir, run_id, _resolve(args.preflight_dir),
            )
            arms[arm] = status
            if current is None:
                control.interrupted()
                return {"status": "INTERRUPTED", "resume_checkpoint": status["checkpoint"]}
            terminal_models[arm] = current

    if "a_reconstruction_only" not in terminal_models:
        terminal_models["a_reconstruction_only"] = load_completed_arm("a_reconstruction_only")
    if "b_reconstruction_plus_power" not in terminal_models:
        terminal_models["b_reconstruction_plus_power"] = load_completed_arm("b_reconstruction_plus_power")
    for arm in ARM_WEIGHTS:
        if arm not in arms:
            completion = _load_json(run_dir / arm / "arm_completion.json")
            arms[arm] = {"status": completion["status"],
                         "completed_updates": completion["completed_updates"],
                         "endpoint_checkpoint": completion["endpoint_checkpoint"],
                         "resumed_without_duplicate_updates": True}
    _check_evaluator_frozen(evaluator, evaluator_state)
    arm_checkpoint_paths = {
        arm: str((run_dir / arm / "checkpoint_step_000500.pt").relative_to(run_dir)) for arm in ARM_WEIGHTS
    }
    atomic_write_json(run_dir / "progress.json", {
        "status": "EVALUATING", "current_arm": "evaluation", "current_task_local_update": UPDATES,
        "completed_updates_by_arm": {arm: UPDATES for arm in ARM_WEIGHTS},
        "resume_checkpoint": arm_checkpoint_paths["b_reconstruction_plus_power"],
        "evaluation_updates": 0,
    })
    evaluation_index = 1
    while (run_dir / f"evaluation_{evaluation_index:02d}").exists():
        evaluation_index += 1
    evaluation_dir = run_dir / f"evaluation_{evaluation_index:02d}"
    evaluation_dir.mkdir(parents=False, exist_ok=False)
    _append_jsonl(run_dir / "segments.jsonl", {
        "event": "evaluation_started", "evaluation_segment": evaluation_dir.name,
        "resume_checkpoint": arm_checkpoint_paths["b_reconstruction_plus_power"],
        "timestamp_unix": time.time(),
    })
    requested_evaluation_signal: list[int] = []
    previous_evaluation_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def evaluation_handler(signum: int, frame: Any) -> None:
        del frame
        requested_evaluation_signal.append(signum)

    for sig in previous_evaluation_handlers:
        signal.signal(sig, evaluation_handler)

    def evaluation_interrupted(stage: str) -> bool:
        if not requested_evaluation_signal:
            return False
        event = {"event": "evaluation_interrupted", "stage": stage,
                 "signal": requested_evaluation_signal[-1], "evaluation_segment": evaluation_dir.name,
                 "resume_checkpoint": arm_checkpoint_paths["b_reconstruction_plus_power"],
                 "optimizer_updates_after_endpoint": 0, "timestamp_unix": time.time()}
        _append_jsonl(run_dir / "interruption_events.jsonl", event)
        _append_jsonl(run_dir / "segments.jsonl", event)
        atomic_write_json(run_dir / "progress.json", {
            "status": "INTERRUPTED_EVALUATION", "current_arm": "evaluation",
            "resume_checkpoint": arm_checkpoint_paths["b_reconstruction_plus_power"],
            "optimizer_updates_after_endpoint": 0,
        })
        control.interrupted()
        return True

    try:
        source_result = _evaluate(source_model, population, pixels, labels, evaluator, references, device,
                                  include_unit_readout=True)
        if evaluation_interrupted("source_2048"):
            return {"status": "INTERRUPTED_EVALUATION", "optimizer_updates_after_endpoint": 0}
        source_generated = _evaluate_generated(source_model, population, evaluator, references, device)
        start_result = _evaluate(diagnostic_start, population, pixels, labels, evaluator, references, device,
                                 include_unit_readout=True)
        if evaluation_interrupted("decoder_4096_start"):
            return {"status": "INTERRUPTED_EVALUATION", "optimizer_updates_after_endpoint": 0}
        start_generated = _evaluate_generated(diagnostic_start, population, evaluator, references, device)
        output_results: dict[str, dict[str, Any]] = {
            "source_2048": source_result, "start_4096": start_result,
        }
        generated_results: dict[str, dict[str, Any]] = {
            "source_2048": source_generated, "start_4096": start_generated,
        }
        preview_views: dict[str, dict[int, torch.Tensor]] = {
            "source_2048": source_result["preview_views"], "start_4096": start_result["preview_views"],
        }
        for arm, state_name in (("a_reconstruction_only", "A"), ("b_reconstruction_plus_power", "B")):
            output_results[state_name] = _evaluate(
                terminal_models[arm], population, pixels, labels, evaluator, references, device,
                include_unit_readout=True,
            )
            if evaluation_interrupted(f"{state_name}_reconstruction"):
                return {"status": "INTERRUPTED_EVALUATION", "optimizer_updates_after_endpoint": 0}
            generated_results[state_name] = _evaluate_generated(
                terminal_models[arm], population, evaluator, references, device
            )
            preview_views[state_name] = output_results[state_name]["preview_views"]
        endpoint_windows = {
            state_name: _window_checks(terminal_models[arm], population, pixels, labels, config, device)
            for arm, state_name in (("a_reconstruction_only", "A"),
                                    ("b_reconstruction_plus_power", "B"))
        }
        if evaluation_interrupted("terminal_window_checks"):
            return {"status": "INTERRUPTED_EVALUATION", "optimizer_updates_after_endpoint": 0}
        window_pass = all(value["passed"] for value in endpoint_windows.values())
        _check_evaluator_frozen(evaluator, evaluator_state)
        local_gate: dict[str, Any] = {}
        start_confirm = output_results["start_4096"]["summary"]["confirmation"]
        for state_name in ("A", "B"):
            metrics = output_results[state_name]["summary"]["confirmation"]
            relative_improvement = 1.0 - metrics["mse400_unclipped_mean"] / start_confirm["mse400_unclipped_mean"]
            local_gate[state_name] = {
                "confirm_mse_improvement_fraction_vs_4096_start": relative_improvement,
                "ssim_not_lower": metrics["ssim28_mean"] >= start_confirm["ssim28_mean"],
                "saturation_within_limit": metrics["saturation_fraction"] <= LOCAL_SATURATION_MAX,
                "passed": relative_improvement >= LOCAL_MSE_IMPROVEMENT_MIN
                          and metrics["ssim28_mean"] >= start_confirm["ssim28_mean"]
                          and metrics["saturation_fraction"] <= LOCAL_SATURATION_MAX,
            }
        gate_passed = any(value["passed"] for value in local_gate.values())
        if not window_pass:
            control.invalid_harness()
            status = "INVALID_HARNESS"
        elif gate_passed:
            control.complete()
            status = "LOCAL_RECONSTRUCTION_IMPROVEMENT_SUPPORTED"
        else:
            control.valid_failed()
            status = "VALID_FAILED"
        per_sample = {name: value["per_sample"] for name, value in output_results.items()}
        atomic_write_new_json(evaluation_dir / "per_sample_metrics.json", baseline._json_safe(per_sample))
        atomic_write_new_json(evaluation_dir / "per_sample_generation_metrics.json", baseline._json_safe({
            name: value["per_sample"] for name, value in generated_results.items()
        }))
        _write_units(evaluation_dir, output_results, generated_results)
        atomic_torch_save_new(evaluation_dir / "fixed_population.pt", population)
        _save_atlas(evaluation_dir / "comparison_atlas.png", pixels, labels, population, preview_views)
        _save_generation_atlas(evaluation_dir / "generation_atlas.png", generated_results)
        atomic_torch_save_new(evaluation_dir / "generated_display_views.pt", {
            name: value["display_views"] for name, value in generated_results.items()
        })
        if evaluation_interrupted("artifact_writeback"):
            return {"status": "INTERRUPTED_EVALUATION", "optimizer_updates_after_endpoint": 0}
        atomic_write_new_json(evaluation_dir / "terminal_window_checks.json",
                              baseline._json_safe(endpoint_windows))
        report = {
            "schema": "chen_lightgen_400_quickcheck_report.v2", "status": status,
            "task_id": TASK_ID, "authorization_id": AUTHORITY, "run_identity": identity.to_dict(),
            "source_baseline_status": source_report["status"], "source_checkpoint_step": 30_000,
            "source_encoder_grid": 2048, "decoder_diagnostic_grid": DECODER_GRID,
            "preflight": str(_resolve(args.preflight_dir)), "evaluation_segment": evaluation_dir.name,
            "data_roles": {"fit": 100, "confirmation": 200, "both_canonical_train": True,
                           "official_test_reads": 0},
            "source_and_arm_metrics": {name: value["summary"] for name, value in output_results.items()},
            "generated_saved_latent_metrics": {name: {key: val for key, val in value.items()
                                                       if key not in {"unit_readout", "display_views", "per_sample"}}
                                                for name, value in generated_results.items()},
            "local_gate": {"criteria": {"confirm_mse_improvement_fraction_min": LOCAL_MSE_IMPROVEMENT_MIN,
                                          "ssim_not_lower_than_4096_start": True,
                                          "saturation_fraction_max": LOCAL_SATURATION_MAX},
                           "by_arm": local_gate, "any_arm_passed": gate_passed,
                           "power_term_interpretation": (
                               "A_ONLY_SUPPORTS_LOCAL_POWER_TERM_OBSTRUCTION" if local_gate["A"]["passed"]
                               and not local_gate["B"]["passed"] else
                               "B_ONLY_SUPPORTS_POWER_TERM_ASSISTANCE" if local_gate["B"]["passed"]
                               and not local_gate["A"]["passed"] else
                               "BOTH_ARMS_PASS_POWER_TERM_EFFECT_UNRESOLVED" if all(
                                   value["passed"] for value in local_gate.values()
                               ) else "NO_BRANCH_MET_LOCAL_IMPROVEMENT_GATE")},
            "window_checks": {"preflight": preflight["window_checks"],
                              "terminal_by_arm": endpoint_windows, "passed": window_pass},
            "training": arms,
            "endpoint_checkpoints": arm_checkpoint_paths,
            "constant_l_dist_on_fit_population": _initial_latent_kl(population, references["scale"], device),
            "frozen": {"encoder_and_ols": True, "camera_gain_w_inverse": float(references["camera_gain"].cpu()),
                       "scale_and_seven_plane_power_reference": True, "evaluator": True},
            "units": {"physical": "W/pixel", "task_unclipped": "dimensionless after frozen gain",
                      "display": "dimensionless clipped [0,1]"},
            "elapsed_seconds": time.perf_counter() - started,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "official_test_reads": 0,
            "report_sections": ["结果与状态", "科学认识", "目标缺口", "工程完成与验证", "决策影响与下一步"],
        }
        atomic_write_new_json(run_dir / "report.json", baseline._json_safe(report))
        atomic_write_json(run_dir / "latest_status.json", {
            "status": status, "completed_updates_by_arm": {key: value["completed_updates"] for key, value in arms.items()},
            "report": "report.json",
        })
        atomic_write_json(run_dir / "progress.json", {
            "status": status, "current_arm": "complete", "current_task_local_update": UPDATES,
            "current_optimizer_step": UPDATES, "current_seed": SELECTION_SEED,
            "completed_updates_by_arm": {arm: UPDATES for arm in ARM_WEIGHTS},
            "evaluation_updates": 0,
        })
        atomic_write_json(manifest_path, {**expected_manifest, "final_status": status,
                                          "report": "report.json"})
        _append_jsonl(run_dir / "segments.jsonl", {
            "event": "evaluation_completed", "evaluation_segment": evaluation_dir.name,
            "status": status, "timestamp_unix": time.time(),
        })
        return report
    finally:
        for sig, old in previous_evaluation_handlers.items():
            signal.signal(sig, old)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume"))
    parser.add_argument("--decoder-grid", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--generated-latents", type=Path)
    parser.add_argument("--fixed-population", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen/chen_lightgen_400_mnist_baseline_v1.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/MNIST/processed/training.pt"))
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--recovery-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--signal-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--signal-worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--signal-ready", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    _activate_profile(args.decoder_grid)
    if args.recovery_worker:
        _require(args.worker_output is not None, "recovery worker output path is required")
        return args
    if args.signal_worker:
        _require(args.signal_worker_output is not None and args.signal_ready is not None,
                 "signal worker requires output and ready paths")
        return args
    _require(args.mode is not None, "--mode is required")
    _require(args.checkpoint is not None and args.source_report is not None,
             "explicit --checkpoint and --source-report are required")
    _require(args.fixed_population is not None or args.generated_latents is not None,
             "an explicit --fixed-population or --generated-latents input is required")
    if args.decoder_grid == 4096:
        _require(args.fixed_population is not None,
                 "4096 diagnostic requires explicit --fixed-population")
        _require(args.generated_latents is not None,
                 "4096 diagnostic requires explicit --generated-latents for source coefficient validation")
    _require(args.output_dir is not None, "explicit --output-dir is required")
    if args.mode in ("run", "resume"):
        _require(args.preflight_dir is not None, "run mode requires explicit --preflight-dir")
    if args.mode == "resume":
        _require(args.resume_checkpoint is not None, "resume mode requires explicit --resume-checkpoint")
    else:
        _require(args.resume_checkpoint is None, "--resume-checkpoint is only valid in resume mode")
    return args


def main() -> None:
    args = _parse_args()
    if args.recovery_worker:
        _recovery_worker(_resolve(args.recovery_worker), _resolve(args.worker_output))
        return
    if args.signal_worker:
        _interrupt_probe_worker(_resolve(args.signal_worker), _resolve(args.signal_worker_output),
                                _resolve(args.signal_ready))
        return
    try:
        result = _preflight(args) if args.mode == "preflight" else _run(args)
        print(json.dumps({"status": result.get("status"), "mode": args.mode,
                          "output_dir": str(_resolve(args.output_dir))}, ensure_ascii=False))
    except BaseException as error:
        output_dir = _resolve(args.output_dir)
        if output_dir.is_dir():
            if args.mode == "preflight" and not (output_dir / "preflight_failure.json").exists():
                atomic_write_new_json(output_dir / "preflight_failure.json", {
                    "status": "INVALID_HARNESS", "task_id": TASK_ID, "error_type": type(error).__name__,
                    "error": str(error), "completed_scientific_updates": 0, "official_test_reads": 0,
                })
            elif args.mode in ("run", "resume") and (output_dir / "report.json").exists() is False:
                completed = {}
                for arm in ARM_WEIGHTS:
                    progress = output_dir / arm / "progress.json"
                    completed[arm] = _load_json(progress).get("completed_updates", 0) if progress.is_file() else 0
                _append_jsonl(output_dir / "execution_errors.jsonl", {
                    "status": "INVALID_HARNESS", "task_id": TASK_ID,
                    "error_type": type(error).__name__, "error": str(error),
                    "completed_scientific_updates_by_arm": completed,
                    "resume_checkpoint_required": True, "official_test_reads": 0,
                })
        raise


if __name__ == "__main__":
    main()
