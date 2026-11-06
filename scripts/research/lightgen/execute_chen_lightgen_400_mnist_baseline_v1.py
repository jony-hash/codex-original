"""Train the fixed-budget, six-layer 400x400 LightGen MNIST baseline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ACTIVE_SHAPE,
    FIBER_LENGTH_M,
    LAYER_POWER_TRANSMISSION,
    OLS_PORT_COUNT,
    PHASE_LEVELS,
    PIXEL_PITCH_M,
    SOURCE_POWER_W,
    ChenLightGen400Baseline,
    LightGen400Config,
    integrated_power,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    ED_CHECKPOINT_SCHEMA,
    PixelMLPConfirmationEvaluatorV1,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import (
    EPS_OP,
    EPS_S,
    fixed_prior,
    q_from_batch,
    truncated_normal_positive_kl,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    CORRECTNESS_INVARIANTS,
    IMPLEMENTATION_SELECTION,
    RESOURCE_ENVELOPE,
    RUN_INTERRUPTED,
    HeartbeatWriter,
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import (
    atomic_torch_save,
    atomic_torch_save_new,
    atomic_write_json,
    atomic_write_new_json,
    safe_torch_load,
)
from ldm_project.v2.optics.propagation import (
    angular_spectrum_propagate,
    clear_production_transfer_cache,
    reference_angular_spectrum_propagate,
)
from ldm_project.v2.student.evaluator import stratified_mnist_indices


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DEFAULT = Path("configs/research/lightgen/chen_lightgen_400_mnist_baseline_v1.yaml")
TASK_ID = "CHEN_LIGHTGEN_400_MNIST_BASELINE_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_BASELINE_20261014"
ED_SCHEMA = ED_CHECKPOINT_SCHEMA
UPDATES = 30_000
BATCH_SIZE = 10
TRAIN_COUNT = 55_000
DEV_COUNT = 5_000
TRAIN_SEED = 2026101401
SAMPLE_SEED = 2026101402
TRAIN_EPSILON = 1.0e-8
TRAIN_BETAS = (0.9, 0.999)
RADIUS_RIDGE = 1.0e-6
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
WINDOW_TOLERANCE = 1.0e-4
RECOVERY_RTOL = 2.0e-6
RECOVERY_ATOL = 5.0e-8
MAX_MODAL_POWER_W = SOURCE_POWER_W * LAYER_POWER_TRANSMISSION**2
TASK_SIZE = 28


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_modal_scale(scale: torch.Tensor) -> None:
    _require(
        torch.isfinite(scale).all().item() and bool((scale > 0).all()),
        "TRAIN modal power reference must be finite and strictly positive",
    )


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "config must contain a mapping")
    _require(config.get("schema") == "chen_lightgen_400_mnist_baseline.v1", "baseline config schema changed")
    expected = {
        "task_id": TASK_ID,
        "semantic_authority_id": "CHEN_LIGHTGEN_400_PAPER_REFERENCED_DIAGNOSTIC_V1",
        "data": {
            "training_file": "data/MNIST/processed/training.pt",
            "train_count": TRAIN_COUNT,
            "dev_count": DEV_COUNT,
            "split_seed": 1234,
            "examples_per_label_in_dev": 500,
            "official_test_reads": 0,
        },
        "optics": {
            "encoder_layers": 2,
            "decoder_layers": 4,
            "active_shape": [400, 400],
            "pixel_pitch_m": PIXEL_PITCH_M,
            "wavelength_m": 5.32e-7,
            "adjacent_layer_spacing_m": 3.0e-3,
            "encoder_to_ols_distance_m": 3.0e-3,
            "ols_to_decoder_distance_m": 3.0e-3,
            "decoder_to_camera_distance_m": 3.0e-3,
            "propagation_grid_candidates": [2048, 3072, 4096, 6144],
            "common_complex_comparison_roi_m": 1.5e-3,
            "window_complex_relative_error_max": WINDOW_TOLERANCE,
            "pixel_sampling_reference_factor": 2,
            "layer_power_transmission": LAYER_POWER_TRANSMISSION,
            "input_power_w": SOURCE_POWER_W,
            "ols": {
                "basis_id": "gaussian_s12_v1",
                "ports": OLS_PORT_COUNT,
                "rows": 10,
                "columns": 10,
                "port_pitch_m": 125.0e-6,
                "cladding_diameter_m": 125.0e-6,
                "mode_field_diameter_m": 14.222e-6,
                "fiber_length_m": FIBER_LENGTH_M,
                "transport": "ideal_identity",
                "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
                "phase_screen_to_ols_distances_are_project_approximations": True,
            },
        },
        "loss": {
            "family": "truncated_gaussian_bogt",
            "trainable_groups": ["encoder_phase", "decoder_phase"],
            "reconstruction": "fixed_camera_gain_unclipped_physical_w_per_pixel_mse",
            "optical_power_planes": 7,
            "optical_power_loss": "relative_log_power_squared_mean",
            "initial_weight_calibration": "inverse_gradient_norm_once_on_first_balanced_train_batch",
            "weights_frozen_after_calibration": True,
            "camera_gain": "global_least_squares_scalar_fit_from_initial_full_train",
            "camera_gain_frozen": True,
            "eps_s": EPS_S,
            "eps_op": EPS_OP,
            "prior_mean": 1.0,
            "prior_variance": 0.25,
        },
        "training": {
            "updates": UPDATES,
            "logical_batch_size": BATCH_SIZE,
            "samples_per_label_per_batch": 1,
            "schedule_seed": TRAIN_SEED,
            "initialization": "zero_phase_six_layers",
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "betas": list(TRAIN_BETAS),
            "epsilon": TRAIN_EPSILON,
            "weight_decay": 0.0,
            "early_stopping": False,
            "checkpoint_every_updates": CHECKPOINT_INTERVAL,
            "progress_every_updates": PROGRESS_INTERVAL,
            "retained_recovery_checkpoints": 2,
            "checkpoint_activation_recomputation": True,
            "display_quantization_levels": 8,
        },
        "generation": {
            "sampler": "class_conditional_empirical_complex_gaussian",
            "fit_role": "TRAIN",
            "fit_count": TRAIN_COUNT,
            "global_phase_gauge": "maximum_magnitude_port",
            "feature_scale": "frozen_initial_train_mean_modal_power",
            "covariance": "joint_real_imag_full_covariance",
            "covariance_relative_ridge": RADIUS_RIDGE,
            "samples_per_label": 1000,
            "sampling_seed": SAMPLE_SEED,
            "maximum_modal_power_w": MAX_MODAL_POWER_W,
            "rejection": "reject_draws_above_fixed_maximum_modal_power",
            "classifier_based_sample_selection": False,
        },
        "evaluation": {
            "evaluator_checkpoint": "outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt",
            "evaluator_role": "frozen_independent_confirmation_only",
            "reconstruction_psnr_db_min": 20.0,
            "reconstruction_ssim_min": 0.80,
            "generated_macro_recall_min": 0.80,
            "generated_per_class_recall_min": 0.60,
            "saturation_fraction_max": 0.01,
            "shape_distance": "unit_sum_intensity_total_variation",
            "minimum_generated_to_dev_within_class_shape_distance_ratio": 0.25,
            "shape_pairs_per_class": 10_000,
            "nearest_train_reference_per_class": 50,
        },
        "output": {"run_dir": "outputs/20261014_chen_lightgen_400_mnist_baseline_v1/runs/attempt_09",
                   "evidence_tier": "diagnostic-only", "formal_default_migration": "forbidden",
                   "output_test_reads": 0},
    }

    def verify_section(actual: dict[str, Any], expected_values: dict[str, Any], path_prefix: str) -> None:
        for key, expected_value in expected_values.items():
            value = actual.get(key)
            field = f"{path_prefix}.{key}"
            if isinstance(expected_value, dict):
                _require(isinstance(value, dict), f"fixed {field} section is missing")
                verify_section(value, expected_value, field)
            else:
                _require(value == expected_value, f"fixed {field} contract changed")

    for section, values in expected.items():
        if isinstance(values, dict):
            actual = config.get(section)
            _require(isinstance(actual, dict), f"config section {section} is missing")
            verify_section(actual, values, section)
        else:
            _require(config.get(section) == values, f"fixed {section} contract changed")
    return config


def _load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(path.is_file(), f"local MNIST TRAIN source is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "MNIST source schema changed")
    pixels, labels = payload
    _require(isinstance(pixels, torch.Tensor) and isinstance(labels, torch.Tensor), "MNIST tensors are missing")
    _require(pixels.dtype == torch.uint8 and tuple(pixels.shape) == (60_000, 28, 28), "MNIST images changed")
    _require(labels.dtype == torch.long and tuple(labels.shape) == (60_000,), "MNIST labels changed")
    _require(int(labels.min()) == 0 and int(labels.max()) == 9, "MNIST labels are outside 0..9")
    train, dev = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    train, dev = train.sort().values, dev.sort().values
    _require(train.numel() == TRAIN_COUNT and dev.numel() == DEV_COUNT, "MNIST split counts changed")
    _require(not bool(torch.isin(train, dev).any()), "MNIST TRAIN and DEV overlap")
    _require(torch.equal(torch.bincount(labels[dev], minlength=10), torch.full((10,), 500)),
             "MNIST DEV class counts changed")
    return pixels, labels, train, dev


def _balanced_schedule(labels: torch.Tensor, train_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(TRAIN_SEED)
    by_class = []
    for label in range(10):
        candidates = train_indices[labels[train_indices] == label]
        _require(candidates.numel() > 0, f"TRAIN class {label} is empty")
        by_class.append(candidates[torch.randperm(candidates.numel(), generator=generator)])
    rows = torch.empty((UPDATES, BATCH_SIZE), dtype=torch.long)
    cursors = [0] * 10
    for step in range(UPDATES):
        for label in range(10):
            if cursors[label] == by_class[label].numel():
                cursors[label] = 0
            rows[step, label] = by_class[label][cursors[label]]
            cursors[label] += 1
    _require(torch.equal(labels[rows[0]], torch.arange(10)), "first TRAIN batch is not label-balanced")
    for label, candidates in enumerate(by_class):
        scheduled_prefix = rows[: candidates.numel(), label]
        _require(torch.equal(torch.sort(scheduled_prefix).values, torch.sort(candidates).values),
                 f"fixed schedule does not cover every TRAIN row for label {label}")
    return rows


def _image_batch(pixels: torch.Tensor, rows: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.interpolate(
        pixels.index_select(0, rows.cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0,
        size=ACTIVE_SHAPE,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    return target, target


def _loss_components(
    model: ChenLightGen400Baseline,
    target: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    result = model.forward_from_image(target)
    task_raw = result["physical_w_per_pixel"] * camera_gain.to(
        device=result["physical_w_per_pixel"].device, dtype=result["physical_w_per_pixel"].dtype
    )
    l_rec = (task_raw - target).square().mean()
    z = result["ols_coefficients"]
    power = z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()
    u = power / (scale.to(device=z.device).unsqueeze(0) + EPS_S)
    l_dist = truncated_normal_positive_kl(q_from_batch(u), fixed_prior(device=z.device)).mean()
    powers = result["plane_powers_w"].to(torch.float64)
    l_op = torch.log((powers + EPS_OP) / (power_reference.to(device=z.device).unsqueeze(0) + EPS_OP)).square().mean()
    values = {"dist": l_dist, "rec": l_rec, "op": l_op}
    return torch.stack([l_dist, l_rec, l_op]), values, result


def _gradient_norms(
    model: ChenLightGen400Baseline, components: dict[str, torch.Tensor]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    parameters = (model.encoder_phase, model.decoder_phase)
    norms: dict[str, float] = {}
    group_norms: dict[str, dict[str, float]] = {}
    for index, (name, loss) in enumerate(components.items()):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(components) - 1,
            allow_unused=True,
        )
        norm_squared = torch.zeros((), dtype=torch.float64, device=loss.device)
        per_group: dict[str, float] = {}
        for group, gradient in zip(("encoder_phase", "decoder_phase"), gradients, strict=True):
            if gradient is None:
                per_group[group] = 0.0
                continue
            _require(torch.isfinite(gradient).all().item(), f"{name} gradient contains nonfinite values")
            group_norm_squared = gradient.detach().to(torch.float64).square().sum()
            norm_squared += group_norm_squared
            per_group[group] = float(group_norm_squared.sqrt().cpu())
        norm = float(norm_squared.sqrt().cpu())
        _require(
            math.isfinite(norm) and norm > 0.0,
            f"initial {name} gradient is zero or nonfinite across all trainable groups",
        )
        norms[name] = norm
        group_norms[name] = per_group
    return norms, group_norms


def _calibrated_weights(norms: dict[str, float]) -> dict[str, float]:
    inverse = {key: 1.0 / value for key, value in norms.items()}
    total = sum(inverse.values())
    weights = {key: 3.0 * value / total for key, value in inverse.items()}
    _require(all(math.isfinite(value) and value > 0 for value in weights.values()), "BOGT weights are invalid")
    return weights


def _train_update(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, float]:
    target, _ = _image_batch(pixels, rows, device)
    optimizer.zero_grad(set_to_none=True)
    components, values, result = _loss_components(model, target, scale, power_reference, camera_gain)
    total = sum(weights[name] * values[name] for name in ("dist", "rec", "op"))
    _require(torch.isfinite(total).item(), "BOGT total loss is nonfinite")
    total.backward()
    for name, parameter in (("encoder", model.encoder_phase), ("decoder", model.decoder_phase)):
        gradient = parameter.grad
        _require(gradient is not None and torch.isfinite(gradient).all().item(), f"{name} phase gradient is missing or nonfinite")
        _require(bool(gradient.abs().sum() > 0), f"{name} phase masks did not receive a real gradient")
        gradient_norm = float(torch.linalg.vector_norm(gradient.detach().to(torch.float64)).cpu())
        _require(math.isfinite(gradient_norm) and gradient_norm > 0.0, f"{name} phase gradient norm is invalid")
        if name == "encoder":
            encoder_gradient_norm = gradient_norm
        else:
            decoder_gradient_norm = gradient_norm
    _require(result["plane_powers_w"].shape == (BATCH_SIZE, 7), "BOGT must consume all seven physical power planes")
    before_encoder = model.encoder_phase.detach().clone()
    before_decoder = model.decoder_phase.detach().clone()
    optimizer.step()
    _require(not torch.equal(before_encoder, model.encoder_phase), "encoder phase did not update")
    _require(not torch.equal(before_decoder, model.decoder_phase), "decoder phase did not update")
    plane_powers = result["plane_powers_w"].detach().to(torch.float64).mean(0).cpu().tolist()
    return {
        **{name: float(value.detach()) for name, value in values.items()},
        "total": float(total.detach()),
        "encoder_gradient_norm": encoder_gradient_norm,
        "decoder_gradient_norm": decoder_gradient_norm,
        **{f"plane_{index + 1}_power_w": value for index, value in enumerate(plane_powers)},
    }


def _calibrate_initial_references(
    model: ChenLightGen400Baseline,
    pixels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    scale_sum = torch.zeros(OLS_PORT_COUNT, dtype=torch.float64)
    power_sum = torch.zeros(7, dtype=torch.float64)
    gain_numerator = torch.zeros((), dtype=torch.float64)
    gain_denominator = torch.zeros((), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for rows in train_indices.split(BATCH_SIZE):
            target, _ = _image_batch(pixels, rows, device)
            result = model.forward_from_image(target)
            z = result["ols_coefficients"]
            scale_sum += (z.real.double().square() + z.imag.double().square()).sum(0).cpu()
            power_sum += result["plane_powers_w"].double().sum(0).cpu()
            camera = result["physical_w_per_pixel"].double()
            target64 = target.double()
            gain_numerator += (camera * target64).sum().cpu()
            gain_denominator += camera.square().sum().cpu()
    scale = scale_sum / float(train_indices.numel())
    power_reference = power_sum / float(train_indices.numel())
    _validate_modal_scale(scale)
    _require(torch.isfinite(power_reference).all().item() and bool((power_reference > 0).all()), "TRAIN plane power reference is invalid")
    _require(bool(gain_denominator > 0), "initial camera output has zero energy")
    gain = (gain_numerator / gain_denominator).to(torch.float64)
    _require(torch.isfinite(gain).item() and gain > 0, "fixed camera gain is invalid")
    return scale, power_reference, gain, {"numerator": float(gain_numerator), "denominator": float(gain_denominator)}


def _common_roi(tensor: torch.Tensor, pixels: int) -> torch.Tensor:
    y = (tensor.shape[-2] - pixels) // 2
    x = (tensor.shape[-1] - pixels) // 2
    return tensor[..., y : y + pixels, x : x + pixels]


def _average_fine_samples_at_coarse_pixel_centers(tensor: torch.Tensor) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    _require(height % 2 == 0 and width % 2 == 0, "fine propagation ROI must split into complete 2x2 pixel blocks")
    leading = tensor.shape[:-2]
    blocks = tensor.reshape(*leading, height // 2, 2, width // 2, 2)
    return blocks.mean(dim=(-1, -3))


def _complex_relative_errors(
    candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> tuple[dict[str, float], float]:
    _require(candidate.keys() == reference.keys(), "common-ROI propagation planes changed")
    overlap = sum(
        (reference[name].to(torch.complex128).conj() * candidate[name].to(torch.complex128)).sum()
        for name in candidate
    )
    phase = torch.angle(overlap)
    rotation = torch.exp(-1j * phase)
    errors: dict[str, float] = {}
    for name in candidate:
        current = candidate[name].to(torch.complex128) * rotation
        expected = reference[name].to(torch.complex128)
        denominator = torch.linalg.vector_norm(expected)
        _require(bool(denominator > 0), f"common-ROI reference field is zero at {name}")
        errors[name] = float((torch.linalg.vector_norm(current - expected) / denominator).cpu())
    return errors, float(phase.cpu())


def _field_set(result: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        "encoder_1": result["encoder_fields"][0].tensor,
        "encoder_2_ols_input": result["encoder_fields"][1].tensor,
        "ols_output": result["ols_output_field"].tensor,
        "decoder_1": result["decoder_fields"][0].tensor,
        "decoder_2": result["decoder_fields"][1].tensor,
        "decoder_3": result["decoder_fields"][2].tensor,
        "decoder_4_camera_input": result["decoder_fields"][3].tensor,
    }


def _window_reference(
    model: ChenLightGen400Baseline,
    sample: torch.Tensor,
    candidates: list[int],
    roi_m: float,
    *,
    reference_sampling_factor: int = 2,
    include_sampling_reference: bool = True,
) -> tuple[int, dict[str, Any]]:
    diagnostics: dict[int | str, dict[str, Any]] = {}

    def evaluate(
        grid: int, *, pixel_sampling_factor: int = 1, diagnostic_key: int | str | None = None
    ) -> dict[str, torch.Tensor]:
        clear_production_transfer_cache()
        candidate = ChenLightGen400Baseline(
            LightGen400Config(grid, True, pixel_sampling_factor=pixel_sampling_factor)
        ).to(sample.device)
        with torch.no_grad():
            candidate.encoder_phase.copy_(model.encoder_phase)
            candidate.decoder_phase.copy_(model.decoder_phase)
        with torch.no_grad():
            result = candidate.forward_from_image(sample)
        roi_pixels = int(round(roi_m / candidate.config.pixel_pitch_m))
        cropped = {
            name: _common_roi(field, roi_pixels).detach().cpu()
            for name, field in _field_set(result).items()
        }
        diagnostics[grid if diagnostic_key is None else diagnostic_key] = {
            "gaussian_mode": candidate.ols.geometry_diagnostics,
            "input_power_w": float(result["input_power_w"][0]),
            "ols_input_power_w": float(result["ols_input_power_w"][0]),
            "ols_output_power_w": float(result["ols_output_power_w"][0]),
            "plane_powers_w": result["plane_powers_w"][0].cpu().tolist(),
            "max_fiber_power_ratio": float((result["ols_output_power_w"] / result["input_power_w"]).max()),
        }
        del result, candidate
        if sample.device.type == "cuda":
            torch.cuda.empty_cache()
        return cropped
    reference_grid = candidates[-1]
    reference_fields = evaluate(reference_grid)
    comparisons: dict[int, dict[str, float]] = {
        reference_grid: {name: 0.0 for name in reference_fields}
    }
    global_phases: dict[int, float] = {reference_grid: 0.0}
    for grid in candidates[:-1]:
        grid_fields = evaluate(grid)
        comparisons[grid], global_phases[grid] = _complex_relative_errors(grid_fields, reference_fields)
        del grid_fields
    if include_sampling_reference:
        fine_fields = evaluate(
            candidates[0] * reference_sampling_factor,
            pixel_sampling_factor=reference_sampling_factor,
            diagnostic_key="half_pixel_sampling_reference",
        )
        # Coarse samples sit at pixel centers; average the four surrounding fine samples.
        fine_fields = {
            name: _average_fine_samples_at_coarse_pixel_centers(field)
            for name, field in fine_fields.items()
        }
        coarse_fields = evaluate(candidates[0])
        fine_comparison, fine_phase = _complex_relative_errors(coarse_fields, fine_fields)
        diagnostics[candidates[0]]["complex_relative_error_vs_half_pixel_sampling"] = fine_comparison
        diagnostics[candidates[0]]["aligned_global_phase_vs_half_pixel_sampling_rad"] = fine_phase
        diagnostics[candidates[0]]["spatial_sampling_reference"] = {
            "status": "MEASURED_DIAGNOSTIC_ONLY",
            "reference_sampling_factor": reference_sampling_factor,
            "complex_relative_error_by_plane": fine_comparison,
            "aligned_global_phase_rad": fine_phase,
            "acceptance_threshold": None,
            "decision_use": "report_sampling_difference_without_adding_a_gate",
        }
    for grid in candidates:
        diagnostics[grid]["complex_relative_error_vs_largest_window"] = comparisons[grid]
        diagnostics[grid]["aligned_global_phase_vs_largest_window_rad"] = global_phases[grid]
    accepted = [grid for grid in candidates if max(comparisons[grid].values()) <= WINDOW_TOLERANCE]
    _require(bool(accepted), "no fixed propagation window meets the complex-field tolerance")
    selected = min(accepted)
    del reference_fields
    if include_sampling_reference:
        del fine_fields, coarse_fields
    clear_production_transfer_cache()
    return selected, diagnostics


def _independent_physics_checks(model: ChenLightGen400Baseline) -> dict[str, float | bool]:
    operator = model.ols
    row = 4
    column = 4
    gaussian_mode = operator.modes_y[row, :, None] * operator.modes_x[column, None, :]
    gaussian_field = ComplexField(
        gaussian_mode.to(torch.complex128)[None],
        operator.input_plane,
        OperationProvenance("analytic_gaussian_mode_reference", None, None),
    )
    coefficients = operator.project_coefficients(gaussian_field)
    coefficient_error = float((coefficients[0, row * 10 + column] - 1.0).abs().cpu())
    _require(coefficient_error <= 5.0e-6,
             "Gaussian OLS projection failed the unit-integral analytic mode reference")
    emitted = operator(gaussian_field)
    input_power = integrated_power(gaussian_field)
    output_power = integrated_power(emitted)
    _require(bool((output_power <= input_power * (1.0 + 1.0e-5) + 1.0e-12).all()),
             "Gaussian OLS exceeded the incident power in its independent mode check")

    plane = PlaneSpec("independent_asm_source", (64, 64), (3.0e-6, 3.0e-6), 532.0e-9, 0.0, "independent")
    generator = torch.Generator(device="cpu").manual_seed(2026101403)
    real32 = torch.randn((1, 1, 64, 64), generator=generator, dtype=torch.float32)
    imag32 = torch.randn((1, 1, 64, 64), generator=generator, dtype=torch.float32)
    field32 = ComplexField(
        torch.complex(real32, imag32), plane, OperationProvenance("reference_fixture_complex64", None, None)
    )
    production32 = angular_spectrum_propagate(field32, 3.0e-3, "production_complex64_reference_hop")
    field128 = ComplexField(
        torch.complex(real32.double(), imag32.double()), plane,
        OperationProvenance("independent_double_reference_fixture", None, None),
    )
    independent128 = reference_angular_spectrum_propagate(
        field128, 3.0e-3, "independent_reference_hop"
    )
    complex64_reference_error = float(
        (torch.linalg.vector_norm(production32.tensor.to(torch.complex128) - independent128.tensor)
         / torch.linalg.vector_norm(independent128.tensor)).cpu()
    )
    _require(complex64_reference_error <= WINDOW_TOLERANCE,
             "complex64 production propagation disagrees with its independent double reference")

    generator = torch.Generator(device="cpu").manual_seed(2026101403)
    real64 = torch.randn((1, 1, 64, 64), generator=generator, dtype=torch.float64)
    imag64 = torch.randn((1, 1, 64, 64), generator=generator, dtype=torch.float64)
    field64 = ComplexField(
        torch.complex(real64, imag64), plane, OperationProvenance("reference_fixture_complex128", None, None)
    )
    production128 = angular_spectrum_propagate(field64, 3.0e-3, "production_complex128_reference_hop")
    independent128 = reference_angular_spectrum_propagate(
        field64, 3.0e-3, "independent_complex128_reference_hop"
    )
    complex128_reference_error = float(
        (torch.linalg.vector_norm(production128.tensor - independent128.tensor)
         / torch.linalg.vector_norm(independent128.tensor)).cpu()
    )
    _require(complex128_reference_error <= 1.0e-10,
             "complex128 production angular-spectrum propagation disagrees with its independent reference")
    return {
        "gaussian_unit_coefficient_absolute_error": coefficient_error,
        "ols_mode_output_to_input_power_ratio": float((output_power / input_power).max().cpu()),
        "independent_angular_spectrum_complex64_relative_l2": complex64_reference_error,
        "independent_angular_spectrum_complex128_relative_l2": complex128_reference_error,
    }


def _phase_direction_check(
    model: ChenLightGen400Baseline,
    optimizer_factory: Any,
    target: torch.Tensor,
    rows: torch.Tensor,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, Any]:
    _, components, _ = _loss_components(model, target, scale, power_reference, camera_gain)
    total = sum(weights[name] * components[name] for name in ("dist", "rec", "op"))
    gradient, = torch.autograd.grad(total, (model.encoder_phase,))
    center_region = gradient[0, 0, 190:210, 190:210].detach().abs()
    _require(torch.isfinite(center_region).all().item(), "central encoder phase gradient is nonfinite")
    local_index = int(center_region.argmax())
    y = 190 + local_index // center_region.shape[1]
    x = 190 + local_index % center_region.shape[1]
    parameter = model.encoder_phase
    analytic = float(gradient[0, 0, y, x].detach().cpu())
    epsilon = 1.0e-2
    original = parameter[0, 0, y, x].detach().clone()

    def value() -> float:
        with torch.no_grad():
            _, parts, _ = _loss_components(model, target, scale, power_reference, camera_gain)
            return float(sum(weights[name] * parts[name] for name in ("dist", "rec", "op")).cpu())

    with torch.no_grad():
        parameter[0, 0, y, x] = original + epsilon
    plus = value()
    with torch.no_grad():
        parameter[0, 0, y, x] = original - epsilon
    minus = value()
    with torch.no_grad():
        parameter[0, 0, y, x] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    _require(math.isfinite(finite_difference) and abs(finite_difference) > 0.0,
             "central phase finite difference is zero or nonfinite")
    _require(analytic * finite_difference > 0.0,
             "central phase autograd derivative has the wrong sign")
    _require(abs(analytic - finite_difference) <= max(1.0e-8, 0.05 * abs(finite_difference)),
             "central phase autograd derivative disagrees with finite differences")

    update_model = copy.deepcopy(model)
    update_optimizer = optimizer_factory(update_model.parameters())
    before = float(update_model.encoder_phase[0, 0, y, x].detach().cpu())
    _train_update(update_model, update_optimizer, pixels, labels, rows, device,
                  scale, power_reference, camera_gain, weights)
    delta = float(update_model.encoder_phase[0, 0, y, x].detach().cpu()) - before
    _require(delta * finite_difference < 0.0,
             "first Adam update direction disagrees with the independent central difference")
    return {
        "parameter": [0, 0, y, x],
        "epsilon_rad": epsilon,
        "autograd_derivative": analytic,
        "central_finite_difference": finite_difference,
        "first_adam_parameter_delta": delta,
        "update_direction_agrees": True,
    }


def _activation_recomputation_check(
    model: ChenLightGen400Baseline,
    target: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, float | bool]:
    reference = ChenLightGen400Baseline(
        LightGen400Config(model.config.propagation_grid_size, checkpoint_propagation=False)
    ).to(target.device)
    reference.load_state_dict(model.state_dict())

    def gradients(candidate: ChenLightGen400Baseline) -> tuple[torch.Tensor, torch.Tensor]:
        _, components, _ = _loss_components(candidate, target, scale, power_reference, camera_gain)
        objective = sum(weights[name] * components[name] for name in ("dist", "rec", "op"))
        return torch.autograd.grad(objective, (candidate.encoder_phase, candidate.decoder_phase))

    checkpointed = gradients(model)
    direct = gradients(reference)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for name, expected, actual in zip(("encoder", "decoder"), checkpointed, direct, strict=True):
        _require(torch.isfinite(expected).all().item() and torch.isfinite(actual).all().item(),
                 f"{name} activation-recomputation gradient is nonfinite")
        maximum_absolute_error = max(maximum_absolute_error,
                                     float((expected - actual).abs().max().cpu()))
        denominator = torch.linalg.vector_norm(actual).clamp_min(1.0e-30)
        maximum_relative_error = max(
            maximum_relative_error,
            float((torch.linalg.vector_norm(expected - actual) / denominator).cpu()),
        )
        _require(torch.allclose(expected, actual, rtol=2.0e-5, atol=2.0e-7),
                 f"{name} activation recomputation changed the production gradient")
    del reference, checkpointed, direct
    if target.device.type == "cuda":
        torch.cuda.empty_cache()
    return {"passed": True, "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_l2_error": maximum_relative_error}


def _negative_control_checks(
    evaluator: PixelMLPConfirmationEvaluatorV1,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    first_result: dict[str, object],
    first_target: torch.Tensor,
    camera_gain: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    output = first_result["physical_w_per_pixel"] * camera_gain.to(
        device=first_result["physical_w_per_pixel"].device,
        dtype=first_result["physical_w_per_pixel"].dtype,
    )
    _reconstruction_batch_metrics(output, first_target, rows[0], rows[0])
    permutation = torch.roll(torch.arange(BATCH_SIZE), shifts=1)
    rejected_pairing = False
    try:
        _reconstruction_batch_metrics(
            output,
            first_target.index_select(0, permutation.to(device)),
            rows[0],
            rows[0].index_select(0, permutation),
        )
    except ValueError as error:
        rejected_pairing = "rows do not match" in str(error)
    _require(rejected_pairing, "reconstruction evaluator accepted shuffled image-target row pairing")

    control_rows = rows[:10].reshape(-1)
    control_target, _ = _image_batch(pixels, control_rows, device)
    control_view = F.adaptive_avg_pool2d(control_target, (TASK_SIZE, TASK_SIZE))
    with torch.no_grad():
        predicted = evaluator(control_view).argmax(1).cpu()
    true_labels = labels[control_rows].cpu()
    paired = _classification_summary(true_labels, predicted)
    shuffled = _classification_summary(torch.roll(true_labels, shifts=1), predicted)
    _require(paired["macro_recall"] > shuffled["macro_recall"],
             "frozen E_D metric did not reject the shuffled generation-label negative control")
    return {
        "reconstruction_pairing_rejected": rejected_pairing,
        "clean_train_evaluator_rows": int(control_rows.numel()),
        "correct_request_macro_recall": paired["macro_recall"],
        "shuffled_request_macro_recall": shuffled["macro_recall"],
        "shuffled_request_confusion_matrix": shuffled["confusion_matrix"].tolist(),
    }


def _resize_view(raw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (TASK_SIZE, TASK_SIZE))


def _ssim_batch(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """7x7 Gaussian SSIM, sigma=1.5, data range 1, K1=.01, K2=.03."""
    coordinate = torch.arange(-3, 4, dtype=torch.float64, device=image.device)
    kernel_1d = torch.exp(-coordinate.square() / (2.0 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).to(image.dtype)[None, None]
    x = F.conv2d(image, kernel, padding=3)
    y = F.conv2d(target, kernel, padding=3)
    x2 = F.conv2d(image.square(), kernel, padding=3) - x.square()
    y2 = F.conv2d(target.square(), kernel, padding=3) - y.square()
    xy = F.conv2d(image * target, kernel, padding=3) - x * y
    return (((2.0 * x * y + 0.01**2) * (2.0 * xy + 0.03**2)) /
            ((x.square() + y.square() + 0.01**2) * (x2 + y2 + 0.03**2))).mean((1, 2, 3))


def _reconstruction_batch_metrics(
    unclipped: torch.Tensor,
    target: torch.Tensor,
    prediction_rows: torch.Tensor,
    target_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.equal(prediction_rows.detach().cpu(), target_rows.detach().cpu()):
        raise ValueError("reconstruction prediction and target MNIST rows do not match")
    if unclipped.shape != target.shape:
        raise ValueError("reconstruction prediction and target shapes differ")
    view = _resize_view(unclipped)
    target_view = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))
    mse = (view - target_view).square().mean((1, 2, 3)).clamp_min(1.0e-12)
    psnr = -10.0 * torch.log10(mse)
    return view, psnr, _ssim_batch(view, target_view)


def _classification_summary(requested: torch.Tensor, predicted: torch.Tensor) -> dict[str, Any]:
    requested = requested.detach().cpu().long().reshape(-1)
    predicted = predicted.detach().cpu().long().reshape(-1)
    if requested.shape != predicted.shape or requested.numel() == 0:
        raise ValueError("confirmation labels and predictions must have the same nonempty shape")
    if bool(((requested < 0) | (requested > 9) | (predicted < 0) | (predicted > 9)).any()):
        raise ValueError("confirmation labels and predictions must be in 0..9")
    confusion = torch.bincount(requested * 10 + predicted, minlength=100).reshape(10, 10)
    recall = confusion.diag().double() / confusion.sum(1).clamp_min(1)
    return {
        "confusion_matrix": confusion,
        "recall_by_class": recall,
        "macro_recall": float(recall.mean()),
    }


def _evaluate_reconstruction(
    model: ChenLightGen400Baseline,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    camera_gain: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    predictions: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    psnr_parts: list[torch.Tensor] = []
    ssim_parts: list[torch.Tensor] = []
    saturation = 0
    seen = 0
    examples: dict[str, list[torch.Tensor]] = {"images": [], "targets": [], "labels": [], "indices": []}
    fixed_rows = torch.cat([
        dev_indices[labels[dev_indices] == label][:10]
        for label in range(10)
    ])
    model.eval()
    with torch.no_grad():
        for start in range(0, dev_indices.numel(), BATCH_SIZE):
            rows = dev_indices[start : start + BATCH_SIZE]
            target, _ = _image_batch(pixels, rows, device)
            result = model.forward_from_image(target)
            unclipped = result["physical_w_per_pixel"] * camera_gain.to(
                device=result["physical_w_per_pixel"].device,
                dtype=result["physical_w_per_pixel"].dtype,
            )
            view, psnr, ssim = _reconstruction_batch_metrics(unclipped, target, rows, rows)
            target_view = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))
            psnr_parts.append(psnr.cpu())
            ssim_parts.append(ssim.cpu())
            logits = evaluator(view)
            logit_parts.append(logits.cpu())
            predictions.append(logits.argmax(1).cpu())
            saturation += int(((unclipped < 0.0) | (unclipped > 1.0)).sum())
            seen += unclipped.numel()
            selected = torch.isin(rows, fixed_rows)
            if bool(selected.any()):
                selected_device = selected.to(device=view.device)
                examples["images"].append(view[selected_device].cpu())
                examples["targets"].append(target_view[selected_device].cpu())
                examples["labels"].append(labels[rows[selected]].cpu())
                examples["indices"].append(rows[selected].cpu())
    logits = torch.cat(logit_parts)
    predicted = torch.cat(predictions)
    classification = _classification_summary(labels[dev_indices], predicted)
    return {
        "population": DEV_COUNT,
        "psnr_db_mean": float(torch.cat(psnr_parts).mean()),
        "ssim_mean": float(torch.cat(ssim_parts).mean()),
        "ed_macro_recall": classification["macro_recall"],
        "ed_recall_by_class": classification["recall_by_class"].tolist(),
        "confusion_matrix": classification["confusion_matrix"].tolist(),
        "saturation_fraction": saturation / seen,
    }, {key: torch.cat(value) for key, value in examples.items()}


def _gauge_and_scale(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pivot_index = z.abs().argmax(dim=1, keepdim=True)
    pivot = z.gather(1, pivot_index)
    unit_gauge = pivot / pivot.abs().clamp_min(1.0e-30)
    return z * unit_gauge.conj() / scale.to(z.device).sqrt().unsqueeze(0)


def _fit_class_gaussians(
    model: ChenLightGen400Baseline,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    chunks: dict[int, list[torch.Tensor]] = {label: [] for label in range(10)}
    train_counts = torch.bincount(labels[train_indices], minlength=10)
    model.eval()
    with torch.no_grad():
        for start in range(0, TRAIN_COUNT, BATCH_SIZE):
            rows = train_indices[start : start + BATCH_SIZE]
            target, _ = _image_batch(pixels, rows, device)
            z = model.forward_from_image(target)["ols_coefficients"]
            normalized = _gauge_and_scale(z.to(torch.complex128), scale)
            for label in range(10):
                selected = labels[rows] == label
                if selected.any():
                    chunks[label].append(normalized[selected.to(device=normalized.device)].cpu())
    result: dict[int, dict[str, torch.Tensor]] = {}
    for label in range(10):
        coefficients = torch.cat(chunks[label])
        expected_count = int(train_counts[label])
        _require(coefficients.shape == (expected_count, OLS_PORT_COUNT),
                 f"TRAIN class {label} latent fit population changed: expected {expected_count}, "
                 f"got {coefficients.shape[0]}")
        features = torch.view_as_real(coefficients).reshape(coefficients.shape[0], 2 * OLS_PORT_COUNT)
        mean = features.mean(0)
        centered = features - mean
        covariance = centered.T @ centered / float(features.shape[0] - 1)
        ridge = covariance.diag().mean() * RADIUS_RIDGE
        regularized = covariance + ridge * torch.eye(covariance.shape[0], dtype=torch.float64)
        cholesky, info = torch.linalg.cholesky_ex(regularized)
        _require(int(info) == 0, f"class {label} complex latent covariance is not positive definite")
        result[label] = {"mean": mean, "covariance": covariance, "cholesky": cholesky, "ridge": ridge}
    return result


def _sample_latents(
    gaussian_models: dict[int, dict[str, torch.Tensor]],
    scale: torch.Tensor,
    *,
    samples_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(SAMPLE_SEED)
    outputs: list[torch.Tensor] = []
    requested: list[torch.Tensor] = []
    rejection: dict[str, dict[str, float | int]] = {}
    feature_count = 2 * OLS_PORT_COUNT
    for label in range(10):
        gaussian = gaussian_models[label]
        accepted: list[torch.Tensor] = []
        accepted_count = 0
        valid_draw_count = 0
        trials = 0
        while accepted_count < samples_per_class:
            count = min(max(2 * (samples_per_class - accepted_count), 256), 4096)
            noise = torch.randn((count, feature_count), generator=generator, dtype=torch.float64)
            draws = gaussian["mean"] + noise @ gaussian["cholesky"].T
            complex_scaled = torch.view_as_complex(draws.reshape(count, OLS_PORT_COUNT, 2).contiguous())
            complex_draws = complex_scaled * scale.sqrt().unsqueeze(0)
            modal_power = complex_draws.abs().square().sum(1)
            valid = modal_power <= MAX_MODAL_POWER_W
            valid_draw_count += int(valid.sum())
            chosen = complex_draws[valid][: samples_per_class - accepted_count]
            if chosen.numel():
                accepted.append(chosen)
                accepted_count += chosen.shape[0]
            trials += count
        samples = torch.cat(accepted)
        outputs.append(samples.to(torch.complex64))
        requested.append(torch.full((samples_per_class,), label, dtype=torch.long))
        rejection[str(label)] = {
            "trials": trials,
            "valid_draws": valid_draw_count,
            "retained_samples": accepted_count,
            "rejection_fraction": 1.0 - valid_draw_count / trials,
            "max_modal_power_w": float(samples.abs().square().sum(1).max()),
        }
    return torch.cat(outputs), torch.cat(requested), rejection


def _shape_distance(image: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    a = image.flatten(1)
    b = other.flatten(1)
    a = a / a.sum(1, keepdim=True).clamp_min(1.0e-12)
    b = b / b.sum(1, keepdim=True).clamp_min(1.0e-12)
    return 0.5 * (a - b).abs().sum(1)


def _task_views_for_rows(pixels: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    views = []
    for batch_rows in rows.split(128):
        resized = F.interpolate(
            pixels.index_select(0, batch_rows.cpu()).unsqueeze(1).float() / 255.0,
            size=ACTIVE_SHAPE,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        views.append(F.adaptive_avg_pool2d(resized, (TASK_SIZE, TASK_SIZE)))
    return torch.cat(views)


def _pair_distances(values: torch.Tensor, pair_count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(values.shape[0], (pair_count,), generator=generator)
    offset = torch.randint(1, values.shape[0], (pair_count,), generator=generator)
    second = (first + offset) % values.shape[0]
    return _shape_distance(values[first], values[second])


def _evaluate_generation(
    model: ChenLightGen400Baseline,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    samples: torch.Tensor,
    requested: torch.Tensor,
    labels: torch.Tensor,
    pixels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    camera_gain: torch.Tensor,
    examples_per_class: int = 10,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    logits: list[torch.Tensor] = []
    views: list[torch.Tensor] = []
    raw_images: list[torch.Tensor] = []
    saturation_by_class = torch.zeros(10, dtype=torch.long)
    seen_by_class = torch.zeros(10, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for start in range(0, samples.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            result = model.forward_from_ols(samples[start:end].to(device))
            unclipped = result["physical_w_per_pixel"] * camera_gain.to(
                device=result["physical_w_per_pixel"].device,
                dtype=result["physical_w_per_pixel"].dtype,
            )
            view = _resize_view(unclipped)
            logits.append(evaluator(view).cpu())
            views.append(view.cpu())
            raw_images.append(unclipped.cpu())
            classes = requested[start:end]
            saturated = ((unclipped < 0) | (unclipped > 1)).flatten(1).sum(1).cpu()
            for index, label in enumerate(classes.tolist()):
                saturation_by_class[label] += saturated[index]
                seen_by_class[label] += unclipped[index].numel()
    all_logits = torch.cat(logits)
    all_views = torch.cat(views)
    all_raw = torch.cat(raw_images)
    predicted = all_logits.argmax(1)
    classification = _classification_summary(requested, predicted)
    shape_ratio: dict[str, float] = {}
    nearest_train_distance: list[float] = []
    for label in range(10):
        generated_class = all_views[requested == label]
        dev_class_ids = dev_indices[labels[dev_indices] == label]
        dev_images = _task_views_for_rows(pixels, dev_class_ids)
        train_class_ids = train_indices[labels[train_indices] == label]
        fixed_train_ids = train_class_ids[:50]
        train_images = _task_views_for_rows(pixels, fixed_train_ids)
        generated_pairs = _pair_distances(generated_class, 10_000, 2026101403 + label)
        dev_pairs = _pair_distances(dev_images, 10_000, 2026101503 + label)
        shape_ratio[str(label)] = float(generated_pairs.median() / dev_pairs.median().clamp_min(1.0e-12))
        minima = []
        for start in range(0, generated_class.shape[0], 32):
            generated = generated_class[start : start + 32]
            left = generated.flatten(1)
            right = train_images.flatten(1)
            left = left / left.sum(1, keepdim=True).clamp_min(1.0e-12)
            right = right / right.sum(1, keepdim=True).clamp_min(1.0e-12)
            minima.append((0.5 * (left[:, None] - right[None, :]).abs().sum(2)).min(1).values)
        nearest_train_distance.append(float(torch.cat(minima).mean()))
    selected = torch.cat([
        torch.nonzero(requested == label, as_tuple=False).flatten()[:examples_per_class]
        for label in range(10)
    ])
    return {
        "population": samples.shape[0],
        "samples_per_class": samples.shape[0] // 10,
        "ed_macro_recall": classification["macro_recall"],
        "ed_recall_by_class": classification["recall_by_class"].tolist(),
        "confusion_matrix": classification["confusion_matrix"].tolist(),
        "saturation_fraction_by_class": (saturation_by_class.double() / seen_by_class.clamp_min(1)).tolist(),
        "shape_distance_generated_to_dev_ratio_by_class": shape_ratio,
        "nearest_fixed_train_reference_tv_mean_by_class": nearest_train_distance,
    }, {
        "requested_labels": requested[selected],
        "evaluator_views_28x28": all_views[selected],
        "raw_cmos_w_per_pixel_400x400": all_raw[selected],
        "logits": all_logits[selected],
    }


def _image_uint8(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().squeeze().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array)


def _save_reconstruction_atlas(examples: dict[str, torch.Tensor], path: Path) -> None:
    canvas = Image.new("L", (20 + 10 * 58, 10 * 32), color=255)
    draw = ImageDraw.Draw(canvas)
    for label in range(10):
        indices = torch.nonzero(examples["labels"] == label, as_tuple=False).flatten()[:10]
        _require(indices.numel() == 10, f"reconstruction atlas lacks fixed rows for label {label}")
        y = label * 32 + 2
        draw.text((2, y + 8), str(label), fill=0)
        for column, index in enumerate(indices.tolist()):
            x = 20 + column * 58
            canvas.paste(_image_uint8(examples["targets"][index]), (x, y))
            canvas.paste(_image_uint8(examples["images"][index]), (x + 29, y))
    canvas.save(path, format="PNG")


def _save_generation_atlas(examples: dict[str, torch.Tensor], path: Path) -> None:
    canvas = Image.new("L", (20 + 10 * 30, 10 * 32), color=255)
    draw = ImageDraw.Draw(canvas)
    for label in range(10):
        indices = torch.nonzero(examples["requested_labels"] == label, as_tuple=False).flatten()[:10]
        _require(indices.numel() == 10, f"generation atlas lacks fixed samples for label {label}")
        y = label * 32 + 2
        draw.text((2, y + 8), str(label), fill=0)
        for column, index in enumerate(indices.tolist()):
            canvas.paste(_image_uint8(examples["evaluator_views_28x28"][index]),
                         (20 + column * 30, y))
    canvas.save(path, format="PNG")


def _load_evaluator(path: Path, device: torch.device) -> PixelMLPConfirmationEvaluatorV1:
    _require(path.is_file(), f"independent E_D checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and payload.get("schema") == ED_SCHEMA, "E_D checkpoint schema changed")
    _require(payload.get("authority_status") == "QUALIFIED", "E_D is not qualified")
    _require(payload.get("independence", {}).get("role") == "INDEPENDENT_CONFIRMATION_ONLY", "E_D role changed")
    evaluator = PixelMLPConfirmationEvaluatorV1().to(device)
    evaluator.load_state_dict(payload["state_dict"])
    evaluator.freeze()
    return evaluator


def _source_manifest(config_path: Path, config: dict[str, Any]):
    return capture_execution_source_manifest(
        ROOT,
        task_id=TASK_ID,
        semantic_authority_id=AUTHORITY,
        relevant_paths=(
            config_path,
            Path(__file__),
            Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_baseline.py"),
            Path("src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py"),
            Path("src/ldm_project/v2/optics/propagation.py"),
            Path("src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py"),
            Path("src/ldm_project/v2/application/research/lightgen/confirmation_evaluators.py"),
        ),
        config_identity=config,
    )


def _run_identity(config: dict[str, Any], manifest: Any, run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{TASK_ID}:seed:{TRAIN_SEED}",
        semantic_authority_id=AUTHORITY,
        data_role="canonical_MNIST_TRAIN_55000_DEV_5000_TEST_SEALED",
        config_identity=config,
        model_identity={"encoder": [2, 400, 400], "ols": "gaussian_s12_v1_100_ports_identity", "decoder": [4, 400, 400]},
        optimizer_identity={"name": "Adam", "learning_rate": 1.0e-3, "betas": TRAIN_BETAS, "epsilon": TRAIN_EPSILON},
        sampler_identity={"balanced_rows_seed": TRAIN_SEED, "batch_size": BATCH_SIZE, "updates": UPDATES},
        execution_source_semantics_id=manifest.execution_source_semantics_id,
        source_manifest_id=manifest.source_manifest_id,
    )


def _state_snapshot(model: ChenLightGen400Baseline) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _assert_state_close(left: dict[str, Any], right: dict[str, Any], name: str) -> float:
    _require(left.keys() == right.keys(), f"{name} state keys changed across recovery")
    maximum = 0.0
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            _require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), f"{name} state is nonfinite")
            _require(a.is_complex() == b.is_complex(), f"{name} tensor type changed at {key}")
            if a.numel():
                if a.is_complex():
                    error = float((a.to(torch.complex128) - b.to(torch.complex128)).abs().max())
                else:
                    error = float((a.double() - b.double()).abs().max())
            else:
                error = 0.0
            maximum = max(maximum, error)
            _require(torch.allclose(a, b, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL),
                     f"{name} recovery differs at {key}")
        elif isinstance(a, dict):
            nested = _assert_state_close(a, b, f"{name}.{key}")
            maximum = max(maximum, nested)
        elif isinstance(a, (list, tuple)):
            _require(type(a) is type(b) and len(a) == len(b), f"{name} sequence state differs at {key}")
            nested = _assert_state_close(
                {str(index): value for index, value in enumerate(a)},
                {str(index): value for index, value in enumerate(b)},
                f"{name}.{key}",
            )
            maximum = max(maximum, nested)
        elif isinstance(a, (int, float)):
            _require(a == b or math.isclose(float(a), float(b), rel_tol=1.0e-10, abs_tol=1.0e-12),
                     f"{name} scalar state differs at {key}")
        else:
            _require(a == b, f"{name} state differs at {key}")
    return maximum


def _validate_adam_state(optimizer: torch.optim.Optimizer, completed_updates: int) -> None:
    _require(isinstance(optimizer, torch.optim.Adam), "recovery optimizer is not Adam")
    _require(len(optimizer.param_groups) == 1, "recovery Adam parameter groups changed")
    group = optimizer.param_groups[0]
    _require(group["lr"] == 1.0e-3 and tuple(group["betas"]) == TRAIN_BETAS
             and group["eps"] == TRAIN_EPSILON and group["weight_decay"] == 0.0
             and group["amsgrad"] is False, "recovery Adam hyperparameters changed")
    parameters = list(group["params"])
    _require(len(parameters) == 2 and len(optimizer.state) == 2,
             "recovery Adam trainable parameter population changed")
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        _require(isinstance(state, dict), "recovery Adam parameter state is missing")
        step = state.get("step")
        step_value = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        _require(step_value == completed_updates, "recovery Adam step counters differ from model step")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            _require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                     and torch.isfinite(value).all().item(),
                     f"recovery Adam {name} state is missing, malformed, or nonfinite")


def _preflight_recovery(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    staging: Path,
) -> dict[str, Any]:
    initial_model = _state_snapshot(model)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng_cpu = torch.random.get_rng_state().clone()
    initial_rng_cuda = torch.cuda.get_rng_state_all()
    update_started = time.perf_counter()
    losses = []
    for index in range(3):
        losses.append(_train_update(model, optimizer, images, labels, rows[index], device,
                                    scale, power_reference, gain, weights))
        if index == 0:
            atomic_torch_save_new(staging / "recovery_step_000001.pt", {
                "schema": "chen_lightgen_400_recovery_preflight.v1",
                "completed_updates": 1,
                "model": _state_snapshot(model),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng_cpu": torch.random.get_rng_state().cpu(),
                "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
            })
    parent_update_seconds = time.perf_counter() - update_started
    expected_final = _state_snapshot(model)
    expected_optimizer = _cpu_snapshot(optimizer.state_dict())
    checkpoint_path = staging / "recovery_step_000001.pt"
    command = [sys.executable, str(Path(__file__).resolve()), "--recovery-worker",
               "--checkpoint", str(checkpoint_path), "--result", str(staging / "recovery_worker_result.pt"),
               "--config", str(staging / "recovery_config.json"), "--rows", str(staging / "recovery_rows.pt"),
               "--images", str(staging / "recovery_images.pt"), "--labels", str(staging / "recovery_labels.pt"),
               "--scale", str(staging / "recovery_scale.pt"), "--power-reference", str(staging / "recovery_power_reference.pt"),
               "--gain", str(staging / "recovery_gain.pt"), "--weights", str(staging / "recovery_weights.json")]
    atomic_write_new_json(staging / "recovery_config.json", {"propagation_grid_size": model.config.propagation_grid_size})
    atomic_torch_save_new(staging / "recovery_rows.pt", rows[:3].cpu())
    atomic_torch_save_new(staging / "recovery_images.pt", images.cpu())
    atomic_torch_save_new(staging / "recovery_labels.pt", labels.cpu())
    atomic_torch_save_new(staging / "recovery_scale.pt", scale.cpu())
    atomic_torch_save_new(staging / "recovery_power_reference.pt", power_reference.cpu())
    atomic_torch_save_new(staging / "recovery_gain.pt", gain.cpu())
    atomic_write_new_json(staging / "recovery_weights.json", weights)
    replay_started = time.perf_counter()
    worker = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    _require(worker.returncode == 0, f"independent-process recovery preflight failed: {worker.stderr[-2000:]}")
    recovered = torch.load(staging / "recovery_worker_result.pt", map_location="cpu", weights_only=False)
    _require(recovered.get("completed_updates") == 3, "recovery worker stopped at the wrong update")
    model_error = _assert_state_close(expected_final, recovered["model"], "model")
    optimizer_error = _assert_state_close(expected_optimizer, recovered["optimizer"], "optimizer")
    model.load_state_dict(initial_model)
    optimizer.load_state_dict(initial_optimizer)
    torch.random.set_rng_state(initial_rng_cpu)
    torch.cuda.set_rng_state_all(initial_rng_cuda)
    return {"passed": True, "updates": 3, "checkpoint_update": 1,
            "independent_process": True, "sigint_and_sigterm_safe_boundaries": recovered["signals"],
            "signal_checkpoint_readbacks": recovered["signal_checkpoint_readbacks"],
            "model_max_abs_recovery_error": model_error, "optimizer_max_abs_recovery_error": optimizer_error,
            "losses": losses, "independent_process_replay_seconds": time.perf_counter() - replay_started,
            "parent_three_update_seconds": parent_update_seconds,
            "seconds_per_scientific_update": parent_update_seconds / 3.0,
            "model_restored_after_preflight": True}


def _recovery_worker(args: argparse.Namespace) -> None:
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = torch.load(args.rows, map_location="cpu", weights_only=True)
    images = torch.load(args.images, map_location="cpu", weights_only=True)
    labels = torch.load(args.labels, map_location="cpu", weights_only=True)
    scale = torch.load(args.scale, map_location="cpu", weights_only=True)
    power_reference = torch.load(args.power_reference, map_location="cpu", weights_only=True)
    gain = torch.load(args.gain, map_location="cpu", weights_only=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    device = torch.device("cuda")
    model = ChenLightGen400Baseline(LightGen400Config(config["propagation_grid_size"], True)).to(device)
    model.load_state_dict(checkpoint_payload["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, betas=TRAIN_BETAS, eps=TRAIN_EPSILON)
    optimizer.load_state_dict(checkpoint_payload["optimizer"])
    torch.random.set_rng_state(checkpoint_payload["rng_cpu"])
    torch.cuda.set_rng_state_all(checkpoint_payload["rng_cuda"])
    observed: list[int] = []
    old_handlers: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        del frame
        observed.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    try:
        signal_readbacks: dict[str, dict[str, float]] = {}
        for index in (1, 2):
            _train_update(model, optimizer, images, labels, rows[index], device,
                          scale, power_reference, gain, weights)
            signal.raise_signal(signal.SIGINT if index == 1 else signal.SIGTERM)
            _require(observed[-1] == (signal.SIGINT if index == 1 else signal.SIGTERM),
                     "safe-point signal handler did not receive the expected signal")
            signal_name = "sigint" if index == 1 else "sigterm"
            signal_checkpoint = Path(args.result).with_name(f"signal_{signal_name}_checkpoint.pt")
            expected_model = _state_snapshot(model)
            expected_optimizer = _cpu_snapshot(optimizer.state_dict())
            atomic_torch_save_new(signal_checkpoint, {
                "completed_updates": index + 1,
                "model": expected_model,
                "optimizer": expected_optimizer,
                "rng_cpu": torch.random.get_rng_state().cpu(),
                "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
            })
            signal_state = torch.load(signal_checkpoint, map_location="cpu", weights_only=False)
            _require(signal_state["completed_updates"] == index + 1,
                     f"{signal_name} checkpoint update count changed")
            model_error = _assert_state_close(expected_model, signal_state["model"], f"{signal_name}.model")
            optimizer_error = _assert_state_close(expected_optimizer, signal_state["optimizer"], f"{signal_name}.optimizer")
            model.load_state_dict(signal_state["model"])
            optimizer.load_state_dict(signal_state["optimizer"])
            torch.random.set_rng_state(signal_state["rng_cpu"])
            torch.cuda.set_rng_state_all(signal_state["rng_cuda"])
            if index == 2:
                signal_readbacks = {
                    "sigint": signal_readbacks["sigint"],
                    "sigterm": {"model_max_abs_error": model_error,
                                "optimizer_max_abs_error": optimizer_error},
                }
            else:
                signal_readbacks = {"sigint": {"model_max_abs_error": model_error,
                                                "optimizer_max_abs_error": optimizer_error}}
        atomic_torch_save_new(Path(args.result), {
            "completed_updates": 3,
            "model": _state_snapshot(model),
            "optimizer": _cpu_snapshot(optimizer.state_dict()),
            "signals": [int(value) for value in observed],
            "signal_checkpoint_readbacks": signal_readbacks,
        })
    finally:
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, int, bool, str)) or value is None:
        return value
    raise TypeError(f"report value is not JSON serializable: {type(value)!r}")


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    atomic_torch_save(temporary, payload)
    readback = torch.load(temporary, map_location="cpu", weights_only=False)
    _require(readback.get("schema") == payload["schema"] and readback.get("completed_updates") == payload["completed_updates"],
             "new checkpoint failed its file readback")
    _assert_state_close(payload["model"], readback["model"], "checkpoint_model_readback")
    _assert_state_close(payload["optimizer"], readback["optimizer"], "checkpoint_optimizer_readback")
    _require(readback.get("config") == payload.get("config")
             and readback.get("data_roles") == payload.get("data_roles")
             and readback.get("sampler", {}).get("seed") == TRAIN_SEED
             and readback.get("sampler", {}).get("next_step") == payload["completed_updates"] + 1,
             "new checkpoint failed config, role, or sampler-cursor readback")
    _require(readback.get("references", {}).keys() == payload.get("references", {}).keys()
             and readback.get("history_every_100_updates") == payload.get("history_every_100_updates")
             and readback.get("bogt_weights") == payload.get("bogt_weights"),
             "new checkpoint failed frozen-reference or history readback")
    os.replace(temporary, path)
    verified = torch.load(path, map_location="cpu", weights_only=False)
    _require(verified.get("completed_updates") == payload["completed_updates"]
             and verified.get("schema") == payload["schema"], "published checkpoint failed reload")


def _terminal_checkpoint_replay(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
    path: Path,
) -> dict[str, Any]:
    model.eval()
    target, _ = _image_batch(pixels, rows[0], device)
    with torch.no_grad():
        before = model.forward_from_image(target)
    payload = {
        "schema": "chen_lightgen_400_terminal_model.v1",
        "completed_updates": UPDATES,
        "model": _state_snapshot(model),
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
    else:
        atomic_torch_save_new(path, payload)
        saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == payload["schema"] and saved.get("completed_updates") == UPDATES,
             "terminal model checkpoint failed its readback")
    _assert_state_close(payload["model"], saved["model"], "terminal_model_checkpoint")
    _assert_state_close(payload["optimizer"], saved["optimizer"], "terminal_optimizer_checkpoint")
    _assert_state_close(
        {"rng_cpu": payload["rng_cpu"], "rng_cuda": payload["rng_cuda"]},
        {"rng_cpu": saved["rng_cpu"], "rng_cuda": saved["rng_cuda"]},
        "terminal_rng_checkpoint",
    )
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    _validate_adam_state(optimizer, UPDATES)
    torch.random.set_rng_state(saved["rng_cpu"])
    torch.cuda.set_rng_state_all(saved["rng_cuda"])
    with torch.no_grad():
        after = model.forward_from_image(target)
    field_errors: dict[str, float] = {}
    for name, expected in _field_set(before).items():
        actual = _field_set(after)[name]
        _require(torch.allclose(expected, actual, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL),
                 f"terminal field {name} changed after checkpoint reload")
        field_errors[name] = float((expected - actual).abs().max().cpu())
    for name in ("ols_coefficients", "plane_powers_w", "physical_w_per_pixel"):
        _require(torch.allclose(before[name], after[name], rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL),
                 f"terminal saved-model measurement {name} changed after checkpoint reload")
    power_error = float((before["plane_powers_w"] - after["plane_powers_w"]).abs().max().cpu())
    return {"passed": True, "completed_updates": UPDATES,
            "field_max_abs_error_by_plane": field_errors,
            "seven_plane_power_max_abs_error_w": power_error,
            "checkpoint_reloaded_before_terminal_measurements": True}


def _training_checkpoint(
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_id: str,
    config: dict[str, Any],
    references: dict[str, Any],
    rows: torch.Tensor,
    preflight: dict[str, Any],
    weights: dict[str, float],
    history: list[dict[str, float | int]],
) -> dict[str, Any]:
    return {
        "schema": "chen_lightgen_400_mnist_baseline_checkpoint.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "completed_updates": step,
        "model_identity": {"encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400], "ols": "gaussian_s12_v1"},
        "config": config,
        "data_roles": {"train": TRAIN_COUNT, "dev": DEV_COUNT, "official_test_reads": 0},
        "model": _state_snapshot(model),
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "sampler": {"seed": TRAIN_SEED, "next_row": rows[step].cpu() if step < UPDATES else None,
                    "next_step": step + 1},
        "references": references,
        "bogt_weights": weights,
        "history_every_100_updates": history,
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "preflight_status": "READY",
        "preflight": preflight,
    }


def _restore_checkpoint(
    path: Path,
    model: ChenLightGen400Baseline,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    run_id: str,
    rows: torch.Tensor,
    references: dict[str, Any],
    recalibrated_weights: dict[str, float],
) -> tuple[int, dict[str, float], list[dict[str, float | int]]]:
    _require(path.is_file(), f"explicit recovery checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_mnist_baseline_checkpoint.v1", "recovery checkpoint schema changed")
    _require(payload.get("task_id") == TASK_ID and payload.get("run_id") == run_id, "recovery run identity changed")
    _require(payload.get("config") == config, "recovery training configuration changed")
    _require(payload.get("model_identity") == {"encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400], "ols": "gaussian_s12_v1"},
             "recovery model definition changed")
    _require(payload.get("data_roles") == {"train": TRAIN_COUNT, "dev": DEV_COUNT, "official_test_reads": 0},
             "recovery data roles changed")
    _require(payload.get("preflight_status") == "READY" and isinstance(payload.get("preflight"), dict),
             "recovery checkpoint did not carry a verified preflight")
    step = payload.get("completed_updates")
    _require(isinstance(step, int) and 1 <= step <= UPDATES, "recovery checkpoint step is invalid")
    _require(payload.get("references", {}).keys() == references.keys(), "recovery frozen reference keys changed")
    if step < UPDATES:
        _require(torch.equal(payload["sampler"]["next_row"], rows[step]), "recovery next TRAIN rows changed")
    else:
        _require(payload["sampler"].get("next_row") is None, "terminal recovery sampler cursor changed")
    _require(payload["sampler"].get("seed") == TRAIN_SEED, "recovery sampler seed changed")
    _require(payload["sampler"].get("next_step") == step + 1, "recovery schedule cursor changed")
    for name in references:
        _require(torch.allclose(payload["references"][name].detach().cpu(), references[name].detach().cpu(),
                                rtol=1.0e-8, atol=1.0e-12),
                 f"recovery frozen {name} changed")
    checkpoint_weights = payload.get("bogt_weights")
    _require(isinstance(checkpoint_weights, dict) and checkpoint_weights.keys() == recalibrated_weights.keys(),
             "recovery BOGT weight state is invalid")
    for name, value in recalibrated_weights.items():
        _require(math.isclose(float(checkpoint_weights[name]), value, rel_tol=1.0e-10, abs_tol=1.0e-12),
                 f"recovery frozen BOGT weight {name} changed")
    history = payload.get("history_every_100_updates")
    _require(isinstance(history, list) and all(isinstance(item, dict) for item in history),
             "recovery training history is invalid")
    _require(not history or history[-1].get("step") <= step, "recovery history exceeds checkpoint step")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    _validate_adam_state(optimizer, step)
    torch.random.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state_all(payload["rng_cuda"])
    history_steps = [item.get("step") for item in history]
    expected_history_steps = list(range(PROGRESS_INTERVAL, (step // PROGRESS_INTERVAL) * PROGRESS_INTERVAL + 1,
                                        PROGRESS_INTERVAL))
    _require(history_steps == expected_history_steps, "recovery training history cursor changed")
    return step, {key: float(value) for key, value in checkpoint_weights.items()}, history


def _retain_recent_recovery_checkpoints(run_dir: Path) -> None:
    checkpoints = sorted(run_dir.glob("checkpoint_step_[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    for path in checkpoints[:-2]:
        path.unlink()


def _report_thresholds(config: dict[str, Any], reconstruction: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    criteria = config["evaluation"]
    reconstruction_pass = (
        reconstruction["psnr_db_mean"] >= criteria["reconstruction_psnr_db_min"]
        and reconstruction["ssim_mean"] >= criteria["reconstruction_ssim_min"]
    )
    generated_saturation = generated["saturation_fraction_by_class"]
    generated_recall_pass = (
        generated["ed_macro_recall"] >= criteria["generated_macro_recall_min"]
        and min(generated["ed_recall_by_class"]) >= criteria["generated_per_class_recall_min"]
    )
    saturation_pass = (
        reconstruction["saturation_fraction"] <= criteria["saturation_fraction_max"]
        and max(generated_saturation) <= criteria["saturation_fraction_max"]
    )
    shape_ratio = [float(generated["shape_distance_generated_to_dev_ratio_by_class"][str(i)]) for i in range(10)]
    shape_pass = min(shape_ratio) >= criteria["minimum_generated_to_dev_within_class_shape_distance_ratio"]
    return {
        "passed": reconstruction_pass and generated_recall_pass and saturation_pass and shape_pass,
        "reconstruction_passed": reconstruction_pass,
        "generated_recall_passed": generated_recall_pass,
        "saturation_passed": saturation_pass,
        "within_class_shape_check_passed": shape_pass,
        "criteria": criteria,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recovery-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--images", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--labels", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scale", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--power-reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gain", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--weights", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.recovery_worker:
        _recovery_worker(args)
        return
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _read_config(config_path)
    data_path = ROOT / config["data"]["training_file"]
    evaluator_path = ROOT / config["evaluation"]["evaluator_checkpoint"]
    run_dir = ROOT / config["output"]["run_dir"]
    resume_payload = None
    if args.resume is None:
        _require(not run_dir.exists(), f"protected run directory already exists: {run_dir}")
    else:
        _require(run_dir.is_dir(), f"recovery run directory is missing: {run_dir}")
        _require(args.resume.resolve().is_file(), "explicit recovery checkpoint is missing")
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        _require(resume_payload.get("schema") == "chen_lightgen_400_mnist_baseline_checkpoint.v1",
                 "explicit recovery checkpoint schema changed")
        _require(resume_payload.get("task_id") == TASK_ID, "explicit recovery checkpoint belongs to another task")
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for this fixed-budget training path")
    device = torch.device("cuda")
    started = time.perf_counter()
    pixels, labels, train_indices, dev_indices = _load_data(data_path)
    rows = _balanced_schedule(labels, train_indices)
    evaluator = _load_evaluator(evaluator_path, device)
    torch.cuda.reset_peak_memory_stats(device)
    reference_model = ChenLightGen400Baseline().to(device)
    sample = _image_batch(pixels, rows[0], device)[0][:1]
    try:
        selected_grid, window_diagnostics = _window_reference(
            reference_model,
            sample,
            config["optics"]["propagation_grid_candidates"],
            config["optics"]["common_complex_comparison_roi_m"],
            reference_sampling_factor=config["optics"]["pixel_sampling_reference_factor"],
        )
    except Exception as error:
        atomic_write_new_json(run_dir / "preflight_failure.json", {
            "status": "INVALID_HARNESS",
            "task_id": TASK_ID,
            "attempt_id": Path(config["output"]["run_dir"]).name,
            "stage": "propagation_window_preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_scientific_updates": 0,
        })
        raise
    del reference_model
    clear_production_transfer_cache()
    config["optics"]["selected_production_grid"] = selected_grid
    model = ChenLightGen400Baseline(LightGen400Config(selected_grid, True)).to(device)
    initial_model_state = _state_snapshot(model)
    manifest = _source_manifest(config_path, config)
    run_id = uuid.uuid4().hex if resume_payload is None else resume_payload["run_id"]
    _require(isinstance(run_id, str) and run_id, "recovery run identity is missing")
    identity = _run_identity(config, manifest, run_id)
    control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:preflight_and_training:01")

    def optimizer_factory(parameters: Any) -> torch.optim.Adam:
        return torch.optim.Adam(parameters, lr=1.0e-3, betas=TRAIN_BETAS, eps=TRAIN_EPSILON)

    optimizer = optimizer_factory(model.parameters())
    staging_dir = run_dir.parent / f"preflight_attempt_{uuid.uuid4().hex[:8]}"
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=False)

    try:
        independent_physics = _independent_physics_checks(model)
        scale, power_reference, camera_gain, camera_fit = _calibrate_initial_references(
            model, pixels, train_indices, device
        )
        first_target, _ = _image_batch(pixels, rows[0], device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, first_components, first_result = _loss_components(model, first_target, scale, power_reference, camera_gain)
        _require(first_result["plane_powers_w"].shape == (BATCH_SIZE, 7),
                 "representative path does not expose all seven BOGT power planes")
        _require(bool((first_result["ols_output_power_w"] <= first_result["ols_input_power_w"] * (1 + 1e-5) + 1e-12).all()),
                 "representative OLS output exceeds its input power")
        gradient_norms, gradient_norms_by_group = _gradient_norms(model, first_components)
        weights = _calibrated_weights(gradient_norms)
        optimizer.zero_grad(set_to_none=True)
        activation_recomputation = _activation_recomputation_check(
            model, first_target, scale, power_reference, camera_gain, weights
        )
        phase_direction = _phase_direction_check(
            model, optimizer_factory, first_target, rows[0], pixels, labels, device,
            scale, power_reference, camera_gain, weights,
        )
        negative_controls = _negative_control_checks(
            evaluator, pixels, labels, rows, first_result, first_target, camera_gain, device
        )
        _require(_assert_state_close(initial_model_state, _state_snapshot(model), "no_update_model") == 0.0,
                 "no-update preflight changed the initialized model")
        preflight_recovery = _preflight_recovery(
            model, optimizer, pixels, labels, rows, device, scale, power_reference, camera_gain,
            weights, staging_dir,
        )
        _require(_assert_state_close(initial_model_state, _state_snapshot(model), "rolled_back_model") == 0.0,
                 "engineering preflight updates were not rolled back")
        peak_vram_bytes = torch.cuda.max_memory_reserved(device)
        seconds_per_update = preflight_recovery["seconds_per_scientific_update"]
        preflight = {
            "status": "CHECKS_PASSED_PENDING_RECEIPT",
            "selected_production_grid": selected_grid,
            "window_diagnostics": window_diagnostics,
            "spatial_sampling_reference_decision_use": "diagnostic_only_no_acceptance_threshold_defined",
            "zero_update_train_reference_rows": TRAIN_COUNT,
            "initial_camera_gain_w_inverse": float(camera_gain),
            "camera_gain_fit_sums": camera_fit,
            "modal_scale_epsilon_diagnostics": {
                "eps_s_w": EPS_S,
                "min_train_mean_modal_power_w": float(scale.min()),
                "max_train_mean_modal_power_w": float(scale.max()),
                "channels_at_or_below_eps_s": int((scale <= EPS_S).sum()),
                "minimum_scale_over_eps_s": float((scale / EPS_S).min()),
            },
            "initial_modal_power_scale_w": scale.tolist(),
            "initial_seven_plane_power_reference_w": power_reference.tolist(),
            "initial_gradient_norms": gradient_norms,
            "initial_gradient_norms_by_trainable_group": gradient_norms_by_group,
            "frozen_bogt_weights": weights,
            "positive_power_prior": {"mean": 1.0, "variance": 0.25},
            "independent_physics_checks": independent_physics,
            "activation_recomputation_equivalence": activation_recomputation,
            "phase_central_difference_and_update_direction": phase_direction,
            "negative_controls": negative_controls,
            "representative_three_update_and_independent_recovery": preflight_recovery,
            "engineering_updates_rolled_back": True,
            "peak_reserved_vram_bytes": peak_vram_bytes,
            "preflight_seconds": time.perf_counter() - started,
            "measured_update_rate_seconds_per_step": seconds_per_update,
            "observed_30000_update_seconds": seconds_per_update * UPDATES,
            "official_test_reads": 0,
        }
    except BaseException as error:
        control.invalid_harness()
        atomic_write_new_json(staging_dir / "preflight_failure.json", {
            "status": control.state,
            "error_type": type(error).__name__,
            "error": str(error),
            "selected_production_grid": selected_grid,
            "completed_scientific_updates": 0,
        })
        raise
    atomic_write_new_json(staging_dir / "preflight.json", _json_safe(preflight))
    selected_engine = f"cuda_400x400_six_phase_masks_grid_{selected_grid}"
    try:
        receipt = control.make_ready_receipt(
            correctness={
                "passed": True,
                "checks": ["400x400_two_plus_four_layers", "gaussian_s12_power_integrals", "shared_camera_gain",
                           "seven_power_planes", "complex_window_convergence_1e-4",
                           "spatial_sampling_reference_recorded", "trainable_encoder_decoder",
                           "independent_gaussian_and_propagation_references", "activation_recomputation_gradient_equivalence",
                           "phase_finite_difference_and_adam_direction",
                           "paired_reconstruction_and_generation_label_negative_controls", "independent_recovery",
                           "sigint_sigterm_safe_boundary"],
                "window_max_complex_relative_error": max(
                    max(window_diagnostics[grid]["complex_relative_error_vs_largest_window"].values())
                    for grid in config["optics"]["propagation_grid_candidates"]
                ),
                "three_step_recovery_passed": True,
            },
            resource={"passed": True, "peak_reserved_vram_bytes": peak_vram_bytes,
                      "estimated_training_seconds_from_observed_rate": seconds_per_update * UPDATES},
            implementation={"passed": True, "engine": selected_engine, "selected_engine": selected_engine},
            selected_engine=selected_engine,
            resource_envelope={"device": torch.cuda.get_device_name(device), "peak_reserved_vram_bytes": peak_vram_bytes},
        )
        control.mark_preflight_ready(receipt)
        atomic_write_new_json(staging_dir / "preflight_ready.json", _json_safe(receipt.to_dict()))
    except Exception as error:
        control.invalid_harness()
        atomic_write_new_json(staging_dir / "preflight_failure.json", {
            "status": control.state,
            "stage": "preflight_receipt",
            "error_type": type(error).__name__,
            "error": str(error),
            "selected_production_grid": selected_grid,
            "completed_scientific_updates": 0,
            "completed_preflight_updates": 3,
        })
        raise
    if args.resume is not None:
        reference_tensors = {"scale": scale.detach().cpu(), "power_reference": power_reference.detach().cpu(),
                             "camera_gain": camera_gain.detach().cpu()}
        start_step, weights, history = _restore_checkpoint(
            args.resume, model, optimizer, config, run_id, rows, reference_tensors, weights
        )
        control.state = RUN_INTERRUPTED
        control.preflight_status = "READY"
        control.receipt = receipt
        control.open_scientific_attempt(run_dir, attempt_id="attempt_09", receipt=receipt, resume=True)
        references = reference_tensors
    else:
        control.open_scientific_attempt(run_dir, attempt_id="attempt_09", receipt=receipt)
        start_step = 0
        references = {"scale": scale.detach().cpu(), "power_reference": power_reference.detach().cpu(),
                      "camera_gain": camera_gain.detach().cpu()}
        history: list[dict[str, float | int]] = []

    heartbeat = HeartbeatWriter(run_dir, run_id, total_steps=UPDATES)
    heartbeat.update(state=control.state, phase="TRAINING", step=start_step, force=True)
    heartbeat.start_liveness()
    live_history_path = run_dir / f"training_history_live_{uuid.uuid4().hex[:8]}.jsonl"
    evaluator_state = {name: value.detach().clone() for name, value in evaluator.state_dict().items()}
    rate_start = time.perf_counter()
    rate_start_step = start_step
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    requested_signal: list[int] = []

    def signal_handler(signum: int, frame: Any) -> None:
        del frame
        requested_signal.append(signum)

    for signum in handlers:
        signal.signal(signum, signal_handler)
    try:
        for cursor in range(start_step, UPDATES):
            step = cursor + 1
            losses = _train_update(model, optimizer, pixels, labels, rows[cursor], device,
                                   scale, power_reference, camera_gain, weights)
            if step % PROGRESS_INTERVAL == 0:
                history_record: dict[str, float | int] = {"step": step, **losses}
                history.append(history_record)
                with live_history_path.open("a", encoding="utf-8", newline="\n") as history_file:
                    history_file.write(json.dumps(history_record, ensure_ascii=False) + "\n")
                    history_file.flush()
                elapsed = time.perf_counter() - rate_start
                rate = elapsed / max(1, step - rate_start_step)
                heartbeat.update(state=control.state, phase="TRAINING", step=step, latest_losses=losses)
                atomic_write_json(run_dir / "progress_summary.json", {
                    "completed_updates": step,
                    "total_updates": UPDATES,
                    "latest_loss": losses,
                    "seconds_per_step_since_last_report": rate,
                    "estimated_seconds_remaining": rate * (UPDATES - step),
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
                })
                rate_start = time.perf_counter()
                rate_start_step = step
            checkpoint_path = None
            if step % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = run_dir / f"checkpoint_step_{step:06d}.pt"
                _atomic_checkpoint(checkpoint_path, _training_checkpoint(
                    model, optimizer, step, run_id, config, references, rows, preflight, weights, history
                ))
                _retain_recent_recovery_checkpoints(run_dir)
            if requested_signal:
                checkpoint_path = run_dir / f"checkpoint_signal_step_{step:06d}.pt"
                _atomic_checkpoint(checkpoint_path, _training_checkpoint(
                    model, optimizer, step, run_id, config, references, rows, preflight, weights, history
                ))
                control.interrupted()
                heartbeat.update(state=control.state, phase=control.state, step=step,
                                 last_recovery_checkpoint=checkpoint_path.name, force=True)
                atomic_write_json(run_dir / "interrupted.json", {
                    "state": control.state, "signal": requested_signal[-1],
                    "completed_updates": step, "checkpoint": checkpoint_path.name,
                })
                heartbeat.stop_liveness()
                return
            if step % CHECKPOINT_INTERVAL == 0:
                heartbeat.update(state=control.state, phase="TRAINING", step=step,
                                 latest_losses=losses, last_recovery_checkpoint=checkpoint_path.name, force=True)

        _require(all(torch.equal(value, evaluator_state[name]) for name, value in evaluator.state_dict().items()),
                 "independent confirmation evaluator changed during training")
        terminal_replay = _terminal_checkpoint_replay(
            model, optimizer, pixels, rows, device, run_dir / "terminal_continuous_phase.pt"
        )
        reconstruction, reconstruction_examples = _evaluate_reconstruction(
            model, evaluator, pixels, labels, dev_indices, device, camera_gain
        )
        gaussian_models = _fit_class_gaussians(model, pixels, labels, train_indices, scale, device)
        samples, requested, rejection = _sample_latents(
            gaussian_models, scale, samples_per_class=config["generation"]["samples_per_label"]
        )
        generated, generation_examples = _evaluate_generation(
            model, evaluator, samples, requested, labels, pixels, train_indices, dev_indices,
            device, camera_gain,
        )
        thresholds = _report_thresholds(config, reconstruction, generated)
        quantized = model.quantized_phase_copy().to(device).eval()
        quantized_eval, _ = _evaluate_reconstruction(
            quantized, evaluator, pixels, labels, dev_indices, device, camera_gain
        )
        atomic_torch_save_new(run_dir / "terminal_quantized_model.pt", {
            "schema": "chen_lightgen_400_terminal_model.v1",
            "levels_rad": PHASE_LEVELS.tolist(),
            "model": _state_snapshot(quantized),
        })
        quantized_payload = torch.load(run_dir / "terminal_quantized_model.pt", map_location="cpu", weights_only=False)
        _require(quantized_payload.get("levels_rad") == PHASE_LEVELS.tolist(),
                 "eight-level quantized terminal checkpoint failed readback")
        _assert_state_close(_state_snapshot(quantized), quantized_payload["model"], "quantized_terminal_checkpoint")
        atomic_torch_save_new(run_dir / "reconstruction_examples.pt", reconstruction_examples)
        atomic_torch_save_new(run_dir / "generated_examples.pt", generation_examples)
        atomic_torch_save_new(run_dir / "generated_ols_samples.pt", {
            "schema": "chen_lightgen_400_empirical_complex_gaussian_samples.v1",
            "coefficients": samples.cpu(), "requested_labels": requested,
            "class_models": gaussian_models, "rejection_accounting": rejection,
        })
        _save_reconstruction_atlas(reconstruction_examples, run_dir / "reconstruction_atlas.png")
        _save_generation_atlas(generation_examples, run_dir / "generation_atlas.png")
        with Image.open(run_dir / "reconstruction_atlas.png") as atlas:
            _require(atlas.size == (600, 320), "reconstruction atlas dimensions changed")
            atlas.verify()
        with Image.open(run_dir / "generation_atlas.png") as atlas:
            _require(atlas.size == (320, 320), "generation atlas dimensions changed")
            atlas.verify()
        atomic_write_json(run_dir / "training_history.json", _json_safe(history))
        if thresholds["passed"]:
            final_status = "COMPLETED"
        else:
            final_status = "VALID_FAILED"
        final = {
            "schema": "chen_lightgen_400_mnist_baseline_report.v1",
            "task_id": TASK_ID,
            "evidence_tier": "diagnostic-only",
            "paper_reproduction": False,
            "status": final_status,
            "authorization": AUTHORITY,
            "preflight": preflight,
            "model": {"encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400],
                      "phase_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "selected_production_grid": selected_grid,
                      "continuous_phase_updates": UPDATES,
                      "quantized_readout_updates": 0},
            "optical_assumptions": {"wavelength_m": 5.32e-7, "pixel_pitch_m": PIXEL_PITCH_M,
                                    "layer_spacing_m": 3.0e-3, "phase_to_ols_gaps_project_approximation_m": [3.0e-3, 3.0e-3],
                                    "layer_power_transmission": LAYER_POWER_TRANSMISSION,
                                    "fiber_length_m_metadata_only": FIBER_LENGTH_M,
                                    "fiber_geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
                                    "ols_basis": "gaussian_s12_v1", "ols_transport": "ideal_identity"},
            "data_roles": {"train_count": TRAIN_COUNT, "dev_count": DEV_COUNT,
                           "dev_is_official_train_source_subset": True, "official_test_reads": 0},
            "training": {"updates": UPDATES,
                         "loss_metrics_at_first_interval": history[0] if history else None,
                         "loss_metrics_at_terminal_interval": history[-1] if history else None,
                         "history_every_100_updates": history,
                         "initial_gradient_norms": gradient_norms, "frozen_weights": weights,
                         "frozen_camera_gain_w_inverse": float(camera_gain),
                         "camera_gain_fit_sums": camera_fit},
            "reconstruction": reconstruction,
            "terminal_checkpoint_replay": terminal_replay,
            "generated": generated,
            "generator": {"sampler": config["generation"]["sampler"], "fit_rows": TRAIN_COUNT,
                          "fit_rows_by_class": torch.bincount(labels[train_indices], minlength=10).tolist(),
                          "covariance_relative_ridge": RADIUS_RIDGE, "samples_per_class": 1000,
                          "power_support_max_w": MAX_MODAL_POWER_W, "rejection_accounting": rejection},
            "quantized_8_level_readout_diagnostic": quantized_eval,
            "decision": thresholds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "source_manifest": manifest.to_dict(),
            "artifacts": {"continuous_model": "terminal_continuous_phase.pt",
                          "quantized_model": "terminal_quantized_model.pt",
                          "reconstruction_examples": "reconstruction_examples.pt",
                          "generated_examples": "generated_examples.pt",
                          "generated_latents": "generated_ols_samples.pt",
                          "reconstruction_atlas": "reconstruction_atlas.png",
                          "generation_atlas": "generation_atlas.png",
                          "training_history": "training_history.json"},
        }
        atomic_write_new_json(run_dir / "report.json", _json_safe(final))
        if thresholds["passed"]:
            control.complete()
        else:
            control.valid_failed()
        heartbeat.update(state=control.state, phase=control.state, step=UPDATES, force=True)
        print(json.dumps({"status": control.state, "decision": thresholds, "report": str(run_dir / "report.json")},
                         ensure_ascii=False), flush=True)
    except BaseException as error:
        heartbeat.stop_liveness()
        if control.state in {"SCIENTIFIC_ACTIVE", "RUNNING"}:
            control.invalid_harness()
        atomic_write_json(run_dir / "engineering_error.json", {
            "status": control.state, "error_type": type(error).__name__, "error": str(error),
            "completed_updates": heartbeat._step, "recovery_checkpoint": heartbeat._last_recovery_checkpoint,
        })
        raise
    finally:
        heartbeat.stop_liveness()
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


if __name__ == "__main__":
    main()
