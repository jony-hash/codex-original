"""Run one arm of a paired 2+2 versus Chen-referenced 400x400 comparison."""

from __future__ import annotations

import argparse
import csv
import copy
import gzip
import html
import json
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.checkpoint import checkpoint

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ChenLightGen400Baseline,
    LightGen400Config,
)
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    ED_CHECKPOINT_SCHEMA,
    PixelMLPConfirmationEvaluatorV1,
)
from ldm_project.v2.application.research.lightgen.plane_power_density import (
    field_intensity_distribution,
    field_plane_metrics,
    small_image_result_fields,
)
from ldm_project.v2.application.research.lightgen.native_image_inputs import (
    image_batch as _image_batch,
    native_image_size as _native_size,
)
from ldm_project.v2.application.research.lightgen.image_readout import (
    resize_view as _resize_view,
    ssim_batch as _ssim_batch,
)
from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import (
    EPS_OP,
    EPS_S,
    fixed_prior,
    q_from_batch,
    truncated_normal_positive_kl,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    GAUSSIAN_OLS_MODEL,
    LP01_OLS_MODEL,
    ComplexOLSState,
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.optics.propagation import clear_production_transfer_cache
from ldm_project.v2.evidence import (
    atomic_torch_save,
    atomic_torch_save_new,
    atomic_write_json,
    atomic_write_new_json,
)
from ldm_project.v2.evidence.io import cpu_copy as _cpu_copy
from ldm_project.v2.student.evaluator import stratified_mnist_indices


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/research/lightgen/matched_small_vs_chen400_v1.yaml")
TASK_ID = "LIGHTGEN_MATCHED_SMALL_VS_CHEN400_V1"
AUTHORITY = "USER_APPROVED_MATCHED_ARCHITECTURE_COMPARISON_20261016"
ED_SCHEMA = ED_CHECKPOINT_SCHEMA
UPDATES = 30_000
BATCH_SIZE = 10
CHEN400_FORWARD_MICROBATCH_SIZE = 1
TRAIN_COUNT = 55_000
DEV_COUNT = 5_000
SAMPLE_COUNT = 1_000
SAMPLE_BASE_SEED = 2026101601
SAMPLE_MAX_PROPOSALS_PER_CLASS = 1_000_000
SHAPE_PAIR_COUNT = 10_000
POWER_RIDGE = 1.0e-6
WINDOW_TOLERANCE = 1.0e-4
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
EVALUATION_STEPS = (0, 5_000, 10_000, 20_000, 30_000)
SOURCE_POWER_W = 2.493e-4
LAYER_POWER_TRANSMISSION = 0.889
MAX_MODAL_POWER_W = SOURCE_POWER_W * LAYER_POWER_TRANSMISSION**2
TRAIN_BETAS = (0.9, 0.999)
TRAIN_EPSILON = 1.0e-8
RECOVERY_RTOL = 2.0e-6
RECOVERY_ATOL = 5.0e-8
BOOTSTRAP_SEED = 2026102003
BOOTSTRAP_RESAMPLES = 2_000
GAUSSIAN_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_SYSTEM_BASELINE_V1"
GAUSSIAN_AUTHORITY = "USER_APPROVED_SMALL_GAUSSIAN_SYSTEM_BASELINE_20261020"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "comparison config must be a mapping")
    _require(config.get("schema") == "lightgen_matched_small_vs_chen400.v1", "comparison config schema changed")
    _require(config["training"]["updates"] == UPDATES, "fixed update budget changed")
    _require(config["training"]["evaluation_steps"] == list(EVALUATION_STEPS), "fixed evaluation steps changed")
    _require(config["data"]["official_test_reads"] == 0, "official TEST reads are forbidden")
    _require(config["optics"]["window_complex_relative_error_max"] == WINDOW_TOLERANCE, "window gate changed")
    _require(config["optics"]["small"]["phase_parameters"] == 40_000, "small architecture parameter count changed")
    _require(config["optics"]["chen400"]["phase_parameters"] == 960_000, "400 architecture parameter count changed")
    small = config["optics"]["small"]
    if small.get("ols_basis") == GAUSSIAN_OLS_MODEL:
        _require(config.get("task_id") == GAUSSIAN_TASK_ID, "Gaussian baseline task identity changed")
        _require(config.get("semantic_authority_id") == GAUSSIAN_AUTHORITY, "Gaussian baseline authority changed")
        _require(config["training"]["paired_schedule_seeds"] == [2026101501], "Gaussian baseline must use its fixed seed")
        _require(config["training"]["logical_batch_size"] == BATCH_SIZE
                 and config["training"]["samples_per_class_per_batch"] == 1,
                 "Gaussian baseline logical batch population changed")
        _require(config["training"]["initialization"] == "zero_phase"
                 and config["training"]["optimizer"] == "Adam"
                 and config["training"]["learning_rate"] == 1.0e-3
                 and config["training"]["betas"] == [0.9, 0.999]
                 and config["training"]["epsilon"] == TRAIN_EPSILON
                 and config["training"]["weight_decay"] == 0.0
                 and config["training"]["early_stopping"] is False,
                 "Gaussian baseline optimizer or initialization changed")
        _require(config["data"]["train_count"] == TRAIN_COUNT
                 and config["data"]["dev_count"] == DEV_COUNT
                 and config["data"]["dev_per_class"] == 500,
                 "Gaussian baseline data roles changed")
        _require(config["data"]["training_file"] == "data/MNIST/processed/training.pt"
                 and config["data"]["split_seed"] == 1234,
                 "Gaussian baseline data source or split changed")
        _require(config["optics"]["wavelength_m"] == 5.32e-7
                 and config["optics"]["input_power_w"] == SOURCE_POWER_W
                 and config["optics"]["layer_power_transmission"] == LAYER_POWER_TRANSMISSION,
                 "Gaussian baseline optical power or wavelength changed")
        _require(small.get("phase_pixel_pitch_m") == 2.0e-6
                 and small.get("slm_shape") == [200, 200]
                 and small.get("ols_ports") == 100,
                 "Gaussian baseline phase geometry or OLS population changed")
        _require(small.get("ols_transport") == "identity"
                 and small.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
                 "Gaussian baseline OLS identity or geometry status changed")
        _require(small.get("production_propagation_grid") == 2048, "Gaussian production grid must remain 2048")
        _require(small.get("propagation_grid_candidates") == [2048, 4096], "Gaussian reference grid must remain 4096")
        _require(small.get("port_pitch_m") == 20.0e-6, "Gaussian port pitch changed")
        _require(small.get("field_1e_radius_m") == 3.5e-6, "Gaussian mode radius changed")
        _require(config["evaluation"].get("max_proposals_per_class") == SAMPLE_MAX_PROPOSALS_PER_CLASS,
                 "Gaussian sampling proposal cap changed")
        _require(config["evaluation"].get("generated_samples_per_class") == SAMPLE_COUNT
                 and config["evaluation"].get("generated_and_dev_pair_count_per_class") == SHAPE_PAIR_COUNT
                 and config["evaluation"].get("paired_bootstrap_resamples") == BOOTSTRAP_RESAMPLES
                 and config["evaluation"].get("paired_bootstrap_seed") == BOOTSTRAP_SEED
                 and config["evaluation"].get("classifier_sample_selection") is False,
                 "Gaussian baseline generation population or comparison settings changed")
        _require(config["quality_gate"] == {
            "reconstruction_psnr_db_min": 20.0,
            "reconstruction_ssim_min": 0.80,
            "generation_macro_recall_min": 0.80,
            "generation_recall_per_class_min": 0.60,
            "proxy_clipping_fraction_max": 0.01,
            "generated_to_dev_tv_ratio_min": 0.25,
        }, "Gaussian system quality thresholds changed")
        _require(config["output"].get("experiment_root") ==
                 "outputs/20261020_lightgen_small_gaussian_system_baseline_v1/runs/attempt_01"
                 and config["output"].get("save_evaluation_checkpoints") is True
                 and config["output"].get("formal_result_budget_bytes") == 512 * 1024 * 1024,
                 "Gaussian baseline output path, checkpoint plan, or budget changed")
    return config


class SmallMatchedAdapter(torch.nn.Module):
    """Expose the existing compact optical path through the matched-runner contract."""

    def __init__(self, propagation_grid_size: int, ols_model: str = LP01_OLS_MODEL) -> None:
        super().__init__()
        if ols_model not in {LP01_OLS_MODEL, GAUSSIAN_OLS_MODEL}:
            raise ValueError(f"unsupported small OLS basis: {ols_model}")
        physical = LightGenPhysicalOpticsConfig(
            input_power_w=SOURCE_POWER_W,
            layer_power_transmission=LAYER_POWER_TRANSMISSION,
            global_fiber_geometry=False,
            fiber_attenuation_db_per_km=0.0,
        )
        self.runtime = LightGen2DSemanticGenerationInputOLSAlignment(
            LightGen2DAlignmentRuntimeConfig(
                propagation_grid_size=propagation_grid_size,
                ols_model=ols_model,
                physical_optics=physical,
            )
        )
        _require(self.runtime.ols.basis_id == ols_model, "small OLS basis changed")
        if ols_model == GAUSSIAN_OLS_MODEL:
            identity = self.runtime.ols.mode_identity()
            _require(identity.get("port_center_pitch_m") == 20.0e-6, "Gaussian OLS port pitch changed")
            _require(identity.get("gaussian_field_1e_radius_m") == 3.5e-6, "Gaussian OLS field radius changed")
            _require(identity.get("transport") == "T=I_100 coefficient space", "Gaussian OLS transport changed")

    @property
    def encoder_phase(self) -> torch.nn.Parameter:
        return self.runtime.encoder.phase

    @property
    def decoder_phase(self) -> torch.nn.Parameter:
        return self.runtime.generator.phase

    @property
    def config(self) -> LightGen2DAlignmentRuntimeConfig:
        return self.runtime.config

    def forward_from_image(self, intensity: torch.Tensor) -> dict[str, Any]:
        result = self.runtime.forward_from_image(intensity)
        trace = result.physical_power_trace
        _require(trace is not None, "small architecture did not return SI power trace")
        raw = result.physical_sensor_output_w_per_pixel
        _require(raw is not None, "small architecture did not return physical W/pixel")
        coefficients = result.z_ols.coefficients
        powers = torch.stack(
            (
                trace["P_encoder_1_W"],
                trace["P_encoder_2_W"],
                trace["P_generator_entry_W"],
                trace["P_decoder_1_W"],
                trace["P_decoder_2_W"],
            ),
            dim=1,
        )
        return {
            "physical_w_per_pixel": raw,
            "ols_coefficients": coefficients,
            "plane_powers_w": powers,
            "image_input": result.image_input,
            "encoder_fields": result.encoder_records,
            "decoder_fields": result.generator_records,
            "ols_input_field": result.ols_input_field,
            "generator_entry_field": result.generator_input_field,
            "generator_exit_field": result.generator_output_field,
            "raw_intensity_w_per_m2": result.raw_camera.tensor,
        }

    def forward_from_ols(self, coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
        if coefficients.ndim != 2 or coefficients.shape[1] != 100 or not torch.is_complex(coefficients):
            raise ValueError("generated OLS coefficients must be complex [batch,100]")
        state = ComplexOLSState(
            coefficients=coefficients,
            source="matched_empirical_prior",
            basis_id=self.runtime.ols.basis_id,
            mode_geometry=self.runtime.ols.mode_identity(),
        )
        generator_input = self.runtime.ols.write_to_generator_input(
            state, self.runtime.generator_entrance_plane
        )
        generator_output, _, _ = self.runtime.generator(generator_input, "matched_generated_generator")
        generator_output = ComplexField(
            generator_output.tensor,
            self.runtime.generator_exit_plane,
            OperationProvenance(
                "lightgen_generator_exit",
                generator_output.plane.plane_id,
                None,
                {"generator_layers": 2},
            ),
        )
        camera_field = self.runtime.output_relay(generator_output)
        area = self.runtime.camera_plane.pixel_pitch_m[0] * self.runtime.camera_plane.pixel_pitch_m[1]
        raw = camera_field.tensor.abs().square()
        return {
            "raw_intensity_w_per_m2": raw,
            "physical_w_per_pixel": raw * area,
        }


class SamplingPopulationIncomplete(RuntimeError):
    """The fixed candidate budget could not produce the registered cohort."""


def _model(
    arm: str,
    grid: int,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> torch.nn.Module:
    if arm == "small":
        small = {} if config is None else config["optics"]["small"]
        model: torch.nn.Module = SmallMatchedAdapter(
            grid, small.get("ols_basis", LP01_OLS_MODEL)
        )
    elif arm == "chen400":
        # The matched runner checkpoints one Chen sample at a time in _forward.
        # Disable nested per-hop checkpoints so only one saved-tensor hook stack
        # retains the large propagation-grid inputs during gradient calibration.
        model = ChenLightGen400Baseline(
            LightGen400Config(grid, checkpoint_propagation=False)
        )
        _require(isinstance(model.ols, GaussianFiberArrayOLS), "400 architecture OLS basis changed")
        _require(model.ols.diagnostic_only is True, "400 Gaussian OLS must remain diagnostic-only")
        _require(
            model.ols.geometry_diagnostics.get("normalization") == "analytic_infinite_plane_S12",
            "400 Gaussian OLS normalization changed",
        )
    else:
        raise ValueError("arm must be small or chen400")
    return model.to(device)


def _phase_groups(model: torch.nn.Module) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    encoder = getattr(model, "encoder_phase")
    decoder = getattr(model, "decoder_phase")
    return encoder, decoder


def _load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(path.is_file(), f"MNIST source is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "MNIST source schema changed")
    pixels, labels = payload
    _require(isinstance(pixels, torch.Tensor) and pixels.dtype == torch.uint8 and tuple(pixels.shape) == (60_000, 28, 28),
             "MNIST image source must remain uint8 [60000,28,28]")
    _require(isinstance(labels, torch.Tensor) and labels.dtype == torch.long and tuple(labels.shape) == (60_000,),
             "MNIST labels must remain long [60000]")
    train, dev = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    train, dev = train.sort().values, dev.sort().values
    _require(train.numel() == TRAIN_COUNT and dev.numel() == DEV_COUNT, "MNIST role counts changed")
    _require(not bool(torch.isin(train, dev).any()), "MNIST TRAIN and DEV overlap")
    _require(torch.equal(torch.bincount(labels[dev], minlength=10), torch.full((10,), 500)),
             "MNIST DEV must contain 500 rows per class")
    return pixels, labels, train, dev


def _schedule(labels: torch.Tensor, train_indices: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_class: list[torch.Tensor] = []
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
    _require(torch.equal(labels[rows[0]], torch.arange(10)), "first batch is not label-balanced")
    return rows


def _forward(model: torch.nn.Module, images: torch.Tensor) -> dict[str, Any]:
    if isinstance(model, SmallMatchedAdapter):
        return model.forward_from_image(images)
    if not isinstance(model, ChenLightGen400Baseline):
        return model.forward_from_image(images)

    selected_keys = ("physical_w_per_pixel", "ols_coefficients", "plane_powers_w")
    pieces: list[tuple[torch.Tensor, ...]] = []
    for image_batch in images.split(CHEN400_FORWARD_MICROBATCH_SIZE, dim=0):
        def selected_outputs(image: torch.Tensor) -> tuple[torch.Tensor, ...]:
            result = model.forward_from_image(image)
            return tuple(result[key] for key in selected_keys)

        if torch.is_grad_enabled() and any(parameter.requires_grad for parameter in model.parameters()):
            piece = checkpoint(selected_outputs, image_batch, use_reentrant=False)
        else:
            piece = selected_outputs(image_batch)
        pieces.append(piece)

    return {
        key: torch.cat([piece[index] for piece in pieces], dim=0)
        for index, key in enumerate(selected_keys)
    }


def _record_gaussian_plane_density_probe(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    dev_indices: torch.Tensor,
    run_dir: Path,
    *,
    branch: str,
    global_step: int,
    grid_size: int,
) -> dict[str, Any]:
    """Record one fixed 20-image DEV probe without changing training state."""
    _require(isinstance(model, SmallMatchedAdapter)
             and model.runtime.ols.basis_id == GAUSSIAN_OLS_MODEL,
             "plane-density training probes require the small Gaussian OLS adapter")
    _require(global_step >= 0 and grid_size == model.runtime.config.propagation_grid_size,
             "plane-density probe step or propagation grid is inconsistent")
    rows: list[int] = []
    for label in range(10):
        selected = dev_indices[labels.index_select(0, dev_indices) == label][:2]
        _require(selected.numel() == 2, f"fixed DEV probe class {label} must contain two rows")
        rows.extend(int(value) for value in selected.tolist())
    _require(len(set(rows)) == 20, "fixed DEV probe rows are duplicated")
    selected_rows = torch.tensor(rows, dtype=torch.long)
    selected_labels = labels.index_select(0, selected_rows).to(dtype=torch.long)
    branch_root = run_dir / "plane_power_density" / branch
    branch_root.mkdir(parents=True, exist_ok=True)
    contract_path = branch_root / "recording_contract.json"
    contract = {
        "schema": "lightgen_plane_power_density_training_probe.v1",
        "branch": branch,
        "data_role": "fixed_DEV_probe",
        "samples_per_class": 2,
        "class_order": list(range(10)),
        "rows": selected_rows.tolist(),
        "labels": selected_labels.tolist(),
        "grid_size": int(grid_size),
        "checkpoint_interval": 500,
        "start_and_resume_checkpoint_are_captured": True,
        "planes": [
            "source_before_slm", "slm_output", "encoder_entrance", "encoder_layer_1_aperture",
            "encoder_layer_2_aperture", "ols_incident", "fiber_output_and_decoder_entry",
            "decoder_layer_1_aperture", "decoder_layer_2_aperture_and_exit", "camera_physical_roi",
        ],
        "density_units": ["W/m2", "W/cm2"],
        "material_response_threshold": None,
    }
    if contract_path.exists():
        prior_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        _require(prior_contract == contract, "plane-density training probe contract changed on resume")
    else:
        atomic_write_new_json(contract_path, contract)
    shard_path = branch_root / f"step_{global_step:06d}.csv.gz"
    expected_keys = {(row, label, plane_index) for row, label in zip(rows, selected_labels.tolist(), strict=True)
                     for plane_index in range(10)}

    def write_coverage() -> None:
        coverage_path = branch_root / "coverage.json"
        shard_steps = sorted(int(path.name.removeprefix("step_").removesuffix(".csv.gz"))
                             for path in branch_root.glob("step_*.csv.gz"))
        coverage = {
            "schema": "lightgen_plane_power_density_training_probe_coverage.v1",
            "branch": branch,
            "grid_size": grid_size,
            "rows_per_shard": 200,
            "recorded_global_steps": shard_steps,
            "shard_count": len(shard_steps),
            "record_count": 200 * len(shard_steps),
            "last_shard_complete": True,
        }
        atomic_write_json(coverage_path, coverage)

    if shard_path.is_file():
        with gzip.open(shard_path, "rt", newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        actual_keys = {(int(record["sample_row"]), int(record["label"]), int(record["plane_index"]))
                       for record in existing}
        _require(len(existing) == 200 and actual_keys == expected_keys
                 and all(int(record["global_step"]) == global_step and record["branch"] == branch
                         and int(record["grid_size"]) == grid_size for record in existing),
                 f"existing plane-density shard is incomplete or belongs to another probe: {shard_path}")
        write_coverage()
        return {"status": "REUSED_VERIFIED", "global_step": global_step, "rows": len(existing),
                "path": str(shard_path)}

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    modules = list(model.modules())
    training_modes = [module.training for module in modules]
    parameters = list(model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    gradients = [None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                 for parameter in parameters]
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    records: list[dict[str, Any]] = []
    try:
        model.eval()
        images = _image_batch(pixels, selected_rows, "small", next(model.parameters()).device)
        with torch.inference_mode():
            result = model.runtime.forward_from_image(images)
            fields, _ = small_image_result_fields(model, result)
            _require(len(fields) == 10, "plane-density probe did not expose all ten physical planes")
            for plane_index, (plane_key, field) in enumerate(fields):
                metrics, geometry = field_plane_metrics(field)
                distributions = field_intensity_distribution(field)
                _require(metrics.shape == (20, 7) and len(distributions) == 20,
                         f"plane-density result shape changed at {plane_key}")
                for sample_index, (sample_row, label) in enumerate(zip(rows, selected_labels.tolist(), strict=True)):
                    values = metrics[sample_index]
                    distribution = distributions[sample_index]
                    record: dict[str, Any] = {
                        "schema": "lightgen_plane_power_density_training_probe.v1",
                        "branch": branch,
                        "global_step": global_step,
                        "grid_role": "training_production",
                        "grid_size": grid_size,
                        "data_role": "fixed_DEV_probe",
                        "sample_row": sample_row,
                        "label": label,
                        "plane_index": plane_index,
                        "plane_key": plane_key,
                        "plane_id": geometry["plane_id"],
                        "coordinate_frame": geometry["coordinate_frame"],
                        "height": geometry["shape"][0],
                        "width": geometry["shape"][1],
                        "pixel_pitch_y_m": geometry["pixel_pitch_m"][0],
                        "pixel_pitch_x_m": geometry["pixel_pitch_m"][1],
                        "origin_y_m": geometry["origin_m"][0],
                        "origin_x_m": geometry["origin_m"][1],
                        "z_m": geometry["z_m"],
                        "measurement_boundary": geometry["measurement_boundary"],
                        "power_w": values[0],
                        "geometric_area_m2": values[1],
                        "effective_area_m2": values[2],
                        "mean_intensity_geometric_w_m2": values[3],
                        "mean_intensity_geometric_w_cm2": values[3] / 1.0e4,
                        "mean_intensity_effective_w_m2": values[4],
                        "mean_intensity_effective_w_cm2": values[4] / 1.0e4,
                        "effective_to_geometric_area_ratio": values[5],
                        "integral_intensity_squared_w2_m2": values[6],
                        **distribution,
                    }
                    records.append(record)
        _require(len(records) == 200 and {
            (int(item["sample_row"]), int(item["label"]), int(item["plane_index"])) for item in records
        } == expected_keys, "plane-density probe table keys are incomplete")
    finally:
        for module, training in zip(modules, training_modes, strict=True):
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    after_state = model.state_dict()
    _require(after_state.keys() == state.keys()
             and all(torch.equal(after_state[key].detach().cpu(), value) for key, value in state.items()),
             "plane-density probe changed a model parameter or buffer")
    _require([parameter.requires_grad for parameter in parameters] == requires_grad,
             "plane-density probe changed trainable parameter flags")
    _require(all((parameter.grad is None) == (gradient is None)
                 and (gradient is None or torch.equal(parameter.grad.detach().cpu(), gradient))
                 for parameter, gradient in zip(parameters, gradients, strict=True)),
             "plane-density probe changed a training gradient")
    _require(random.getstate() == python_rng
             and np.array_equal(np.random.get_state()[1], numpy_rng[1])
             and torch.equal(torch.random.get_rng_state(), cpu_rng)
             and (not cuda_rng or all(torch.equal(now, before) for now, before in zip(
                 torch.cuda.get_rng_state_all(), cuda_rng, strict=True))),
             "plane-density probe changed a random-number generator state")
    _require([module.training for module in modules] == training_modes,
             "plane-density probe changed model train/eval modes")
    fieldnames = list(records[0])
    temporary = shard_path.with_name(shard_path.name + ".tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, shard_path)
    write_coverage()
    return {"status": "RECORDED", "global_step": global_step, "rows": 200, "path": str(shard_path)}


def _validate_gaussian_plane_density_coverage(
    run_dir: Path, branch: str, expected_steps: list[int], grid_size: int,
) -> dict[str, Any]:
    branch_root = run_dir / "plane_power_density" / branch
    contract = json.loads((branch_root / "recording_contract.json").read_text(encoding="utf-8"))
    coverage = json.loads((branch_root / "coverage.json").read_text(encoding="utf-8"))
    recorded_steps = coverage.get("recorded_global_steps")
    _require(contract.get("grid_size") == grid_size and coverage.get("grid_size") == grid_size
             and isinstance(recorded_steps, list) and set(expected_steps).issubset(set(recorded_steps))
             and coverage.get("record_count") == 200 * len(recorded_steps),
             f"Gaussian plane-density probe coverage is incomplete for {branch}")
    for step in expected_steps:
        path = branch_root / f"step_{step:06d}.csv.gz"
        _require(path.is_file(), f"Gaussian plane-density checkpoint shard is missing: {path}")
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        keys = {(int(row["sample_row"]), int(row["label"]), int(row["plane_index"])) for row in rows}
        expected = {(int(row), int(label), plane) for row, label in zip(contract["rows"], contract["labels"], strict=True)
                    for plane in range(10)}
        _require(len(rows) == 200 and keys == expected
                 and all(int(row["global_step"]) == step for row in rows),
                 f"Gaussian plane-density shard is incomplete: {path}")
    return {"status": "VERIFIED", "branch": branch, "steps": sorted(expected_steps),
            "record_count": 200 * len(expected_steps), "grid_size": grid_size}


def _inplace_spectral_angular_spectrum_propagate(
    field: ComplexField, distance_m: float, plane_id: str
) -> ComplexField:
    """Use the registered ASM kernel while avoiding a second full FFT-sized product tensor."""
    from ldm_project.v2.optics import propagation

    if distance_m < 0.0:
        raise ValueError("formal v2 propagation distance must be non-negative")
    key = (
        field.plane.shape,
        field.plane.pixel_pitch_m,
        field.plane.wavelength_m,
        distance_m,
        str(field.tensor.device),
        str(field.tensor.real.dtype),
    )
    transfer = propagation._PRODUCTION_TRANSFER_CACHE.get(key)
    if transfer is None:
        with torch.inference_mode(False):
            transfer = propagation._transfer_function(
                field.plane.shape,
                pixel_pitch_m=field.plane.pixel_pitch_m,
                wavelength_m=field.plane.wavelength_m,
                distance_m=distance_m,
                device=field.tensor.device,
                real_dtype=field.tensor.real.dtype,
            )
        propagation._PRODUCTION_TRANSFER_CACHE[key] = transfer
    spectrum = torch.fft.fft2(field.tensor, norm="ortho")
    spectrum.mul_(transfer)
    propagated = torch.fft.ifft2(spectrum, norm="ortho")
    plane = field.plane.with_plane(plane_id, z_m=field.plane.z_m + distance_m)
    provenance = OperationProvenance(
        "propagate_angular_spectrum",
        field.plane.plane_id,
        distance_m,
        {"wavelength_m": field.plane.wavelength_m, "sampling": plane.to_dict()},
    )
    return ComplexField(propagated, plane, provenance)


def _install_chen400_memory_optimization() -> None:
    """Scope in-place spectral multiplication to this matched-comparison process."""
    import ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline as chen400_runtime

    chen400_runtime.angular_spectrum_propagate = _inplace_spectral_angular_spectrum_propagate


def _loss_components(
    model: torch.nn.Module,
    target: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    result = _forward(model, target)
    return _loss_parts_from_result(result, target, scale, power_reference, camera_gain), result


def _loss_parts_from_result(
    result: dict[str, Any],
    target: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw = result["physical_w_per_pixel"]
    gain = camera_gain.to(device=raw.device, dtype=raw.dtype)
    rec = (raw * gain - target).square().mean()
    z = result["ols_coefficients"]
    power = z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()
    normalized = power / (scale.to(device=z.device).unsqueeze(0) + EPS_S)
    dist = truncated_normal_positive_kl(q_from_batch(normalized), fixed_prior(device=z.device)).mean()
    planes = result["plane_powers_w"].to(torch.float64)
    _require(planes.ndim == 2 and planes.shape[0] == target.shape[0], "power-plane producer returned malformed values")
    op = torch.log(
        (planes + EPS_OP) / (power_reference.to(device=planes.device).unsqueeze(0) + EPS_OP)
    ).square().mean()
    parts = {"dist": dist, "rec": rec, "op": op}
    _require(torch.isfinite(torch.stack(list(parts.values()))).all().item(), "BOGT component is nonfinite")
    return parts


def _gradient_norms(
    model: torch.nn.Module,
    loss_parts_factory: Callable[[], dict[str, torch.Tensor]],
) -> dict[str, float]:
    groups = _phase_groups(model)
    norms: dict[str, float] = {}
    reached_groups = [False for _ in groups]
    loss_names = ("dist", "rec", "op")
    for name in loss_names:
        parts = loss_parts_factory()
        _require(set(parts) == set(loss_names), "gradient calibration received incomplete BOGT loss components")
        loss = parts[name]
        gradients = torch.autograd.grad(loss, groups, retain_graph=False, allow_unused=True)
        squared = torch.zeros((), dtype=torch.float64, device=loss.device)
        for group_index, (group, gradient) in enumerate(zip(groups, gradients, strict=True)):
            if gradient is None:
                continue
            _require(torch.isfinite(gradient).all().item(),
                     f"{name} loss produced a nonfinite gradient")
            reached_groups[group_index] = True
            squared += gradient.detach().double().square().sum()
        norm = float(squared.sqrt().cpu())
        _require(math.isfinite(norm) and norm > 0.0, f"{name} gradient norm is invalid or disconnected")
        norms[name] = norm
        del gradients, loss, parts, squared
    _require(all(reached_groups), "the combined BOGT objective did not reach every trainable phase group")
    return norms


def _chen400_streamed_gradient_norms(
    model: ChenLightGen400Baseline,
    target: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
) -> dict[str, float]:
    """Compute exact logical-batch gradients while holding one sample graph at a time."""
    groups = _phase_groups(model)
    loss_names = ("dist", "rec", "op")
    output_keys = ("physical_w_per_pixel", "ols_coefficients", "plane_powers_w")
    with torch.no_grad():
        initial_result = _forward(model, target)
        detached_outputs = {
            key: initial_result[key].detach().requires_grad_(True)
            for key in output_keys
        }
    del initial_result

    norms: dict[str, float] = {}
    reached_groups = [False for _ in groups]
    for name in loss_names:
        parts = _loss_parts_from_result(
            detached_outputs, target, scale, power_reference, camera_gain
        )
        output_gradients = torch.autograd.grad(
            parts[name],
            tuple(detached_outputs[key] for key in output_keys),
            allow_unused=True,
        )
        gradients_by_key = dict(zip(output_keys, output_gradients, strict=True))
        del parts, output_gradients
        active_keys = tuple(key for key in output_keys if gradients_by_key[key] is not None)
        _require(bool(active_keys), f"{name} loss is disconnected from every model output")
        accumulated = [torch.zeros_like(group) for group in groups]

        for sample_index in range(target.shape[0]):
            sample_result = _forward(model, target[sample_index : sample_index + 1])
            sample_outputs = tuple(sample_result[key] for key in active_keys)
            sample_output_gradients = tuple(
                gradients_by_key[key][sample_index : sample_index + 1]
                for key in active_keys
            )
            sample_gradients = torch.autograd.grad(
                sample_outputs,
                groups,
                grad_outputs=sample_output_gradients,
                allow_unused=True,
            )
            for group_index, gradient in enumerate(sample_gradients):
                if gradient is None:
                    continue
                _require(torch.isfinite(gradient).all().item(),
                         f"{name} loss produced a nonfinite streamed gradient")
                accumulated[group_index].add_(gradient.detach())
                reached_groups[group_index] = True
            del sample_result, sample_outputs, sample_output_gradients, sample_gradients

        squared = torch.zeros((), dtype=torch.float64, device=target.device)
        for gradient_sum in accumulated:
            squared += gradient_sum.double().square().sum()
        norm = float(squared.sqrt().cpu())
        _require(math.isfinite(norm) and norm > 0.0,
                 f"{name} gradient norm is invalid or disconnected")
        norms[name] = norm
        del gradients_by_key, active_keys, accumulated, squared

    _require(all(reached_groups), "the combined BOGT objective did not reach every trainable phase group")
    del detached_outputs
    return norms


def _weights(norms: dict[str, float]) -> dict[str, float]:
    inverse = {key: 1.0 / value for key, value in norms.items()}
    total = sum(inverse.values())
    result = {key: 3.0 * value / total for key, value in inverse.items()}
    _require(all(math.isfinite(value) and value > 0.0 for value in result.values()), "calibrated loss weights are invalid")
    return result


def _calibrate(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    train_indices: torch.Tensor,
    arm: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    scale_sum = torch.zeros(100, dtype=torch.float64)
    powers_sum: torch.Tensor | None = None
    gain_numerator = torch.zeros((), dtype=torch.float64)
    gain_denominator = torch.zeros((), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        calibration_batches = math.ceil(train_indices.numel() / BATCH_SIZE)
        for batch_number, rows in enumerate(train_indices.split(BATCH_SIZE), start=1):
            target = _image_batch(pixels, rows, arm, device)
            result = _forward(model, target)
            z = result["ols_coefficients"]
            scale_sum += (z.real.double().square() + z.imag.double().square()).sum(0).cpu()
            planes = result["plane_powers_w"].double().sum(0).cpu()
            if powers_sum is None:
                powers_sum = torch.zeros_like(planes)
            _require(planes.shape == powers_sum.shape, "power-plane count changed during TRAIN calibration")
            powers_sum += planes
            camera = result["physical_w_per_pixel"].double()
            gain_numerator += (camera * target.double()).sum().cpu()
            gain_denominator += camera.square().sum().cpu()
            if batch_number == 1 or batch_number % 100 == 0 or batch_number == calibration_batches:
                calibrated = min(batch_number * BATCH_SIZE, train_indices.numel())
                print(
                    f"TRAIN_REFERENCE_CALIBRATION {calibrated}/{train_indices.numel()} rows",
                    flush=True,
                )
    scale = scale_sum / float(train_indices.numel())
    _require(powers_sum is not None, "TRAIN plane power reference is empty")
    power_reference = powers_sum / float(train_indices.numel())
    _require(torch.isfinite(scale).all().item() and bool((scale > 0.0).all()), "TRAIN modal scale is invalid")
    _require(torch.isfinite(power_reference).all().item() and bool((power_reference > 0.0).all()),
             "TRAIN power-plane reference is invalid")
    _require(bool(gain_denominator > 0.0), "initial camera has zero power")
    camera_gain = gain_numerator / gain_denominator
    _require(torch.isfinite(camera_gain).item() and camera_gain > 0.0, "global camera gain is invalid")
    return scale, power_reference, camera_gain, {
        "numerator": float(gain_numerator),
        "denominator": float(gain_denominator),
    }


def _train_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    arm: str,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, float]:
    model.train()
    target = _image_batch(pixels, rows, arm, device)
    optimizer.zero_grad(set_to_none=True)
    parts, result = _loss_components(model, target, scale, power_reference, camera_gain)
    total = sum(weights[name] * parts[name] for name in ("dist", "rec", "op"))
    _require(torch.isfinite(total).item(), "total BOGT loss is nonfinite")
    total.backward()
    parameters = _phase_groups(model)
    before = [parameter.detach().clone() for parameter in parameters]
    gradient_norms: list[float] = []
    for name, parameter in zip(("encoder", "decoder"), parameters, strict=True):
        gradient = parameter.grad
        _require(gradient is not None and torch.isfinite(gradient).all().item() and bool((gradient.abs().sum() > 0)),
                 f"{name} phase group has no finite nonzero gradient")
        gradient_norms.append(float(torch.linalg.vector_norm(gradient.detach().double()).cpu()))
    optimizer.step()
    for name, old, parameter in zip(("encoder", "decoder"), before, parameters, strict=True):
        _require(not torch.equal(old, parameter), f"{name} phase group did not update")
    return {
        **{name: float(value.detach()) for name, value in parts.items()},
        "total": float(total.detach()),
        "encoder_gradient_norm": gradient_norms[0],
        "decoder_gradient_norm": gradient_norms[1],
        "mean_plane_power_w": float(result["plane_powers_w"].detach().double().mean()),
    }


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _assert_close(left: Any, right: Any, path: str = "state") -> float:
    if isinstance(left, torch.Tensor):
        _require(isinstance(right, torch.Tensor) and left.shape == right.shape, f"{path} shape changed")
        _require(torch.isfinite(left).all().item() and torch.isfinite(right).all().item(), f"{path} is nonfinite")
        error = float((left.to(torch.complex128) - right.to(torch.complex128)).abs().max()) if left.is_complex() else float((left.double() - right.double()).abs().max())
        _require(torch.allclose(left, right, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL), f"{path} differs after recovery")
        return error
    if isinstance(left, dict):
        _require(isinstance(right, dict) and left.keys() == right.keys(), f"{path} keys differ")
        return max((_assert_close(left[key], right[key], f"{path}.{key}") for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        _require(type(left) is type(right) and len(left) == len(right), f"{path} sequence differs")
        return max((_assert_close(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right, strict=True))), default=0.0)
    _require(left == right or (isinstance(left, (int, float)) and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)),
             f"{path} scalar differs")
    return 0.0


def _reconstruction_rows(
    prediction_rows: torch.Tensor,
    target_rows: torch.Tensor,
) -> None:
    if not torch.equal(prediction_rows.detach().cpu(), target_rows.detach().cpu()):
        raise ValueError("reconstruction prediction and target MNIST rows do not match")


def _independent_psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    x = image.detach().cpu().double().numpy()
    y = target.detach().cpu().double().numpy()
    mse = ((x - y) ** 2).mean(axis=(1, 2, 3))
    return float((-10.0 * torch.log10(torch.from_numpy(mse).clamp_min(1e-12))).mean())


def _classification(requested: torch.Tensor, predicted: torch.Tensor) -> dict[str, Any]:
    requested = requested.detach().cpu().long().reshape(-1)
    predicted = predicted.detach().cpu().long().reshape(-1)
    if requested.shape != predicted.shape or not requested.numel():
        raise ValueError("requested and predicted labels must have equal nonempty shapes")
    confusion = torch.bincount(requested * 10 + predicted, minlength=100).reshape(10, 10)
    recall = confusion.diag().double() / confusion.sum(1).clamp_min(1)
    return {
        "confusion_matrix": confusion,
        "recall_by_class": recall,
        "macro_recall": float(recall.mean()),
    }


def _evaluate_reconstruction(
    model: torch.nn.Module,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    dev_indices: torch.Tensor,
    arm: str,
    device: torch.device,
    gain: torch.Tensor,
    collect_per_image: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    mse_parts: list[torch.Tensor] = []
    psnr_parts: list[torch.Tensor] = []
    ssim_parts: list[torch.Tensor] = []
    predicted_parts: list[torch.Tensor] = []
    example_psnr_parts: list[torch.Tensor] = []
    class_labels_parts: list[torch.Tensor] = []
    row_parts: list[torch.Tensor] = []
    clipping_parts: list[torch.Tensor] = []
    saturation = 0
    seen = 0
    fixed_rows = torch.stack([
        dev_indices[labels[dev_indices] == label][:2] for label in range(10)
    ]).reshape(-1)
    examples: dict[str, list[torch.Tensor]] = {"rows": [], "labels": [], "targets": [], "images": []}
    model.eval()
    evaluator.eval()
    with torch.no_grad():
        for rows in dev_indices.split(BATCH_SIZE):
            target_native = _image_batch(pixels, rows, arm, device)
            result = _forward(model, target_native)
            unclipped = result["physical_w_per_pixel"] * gain.to(device=device, dtype=torch.float32)
            view = _resize_view(unclipped)
            target_view = pixels.index_select(0, rows).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
            _reconstruction_rows(rows, rows)
            if view.shape != target_view.shape:
                raise ValueError("common reconstruction view and original MNIST target shapes differ")
            mse = (view - target_view).square().mean((1, 2, 3)).clamp_min(1e-12)
            psnr = -10.0 * torch.log10(mse)
            ssim = _ssim_batch(view, target_view)
            mse_parts.append(mse.cpu())
            psnr_parts.append(psnr.cpu())
            ssim_parts.append(ssim.cpu())
            batch_labels = labels.index_select(0, rows.detach().cpu()).cpu()
            class_labels_parts.append(batch_labels)
            row_parts.append(rows.detach().cpu())
            clipping_parts.append((((unclipped < 0.0) | (unclipped > 1.0)).double().mean((1, 2, 3))).cpu())
            logits = evaluator(view)
            batch_predictions = logits.argmax(1).cpu()
            predicted_parts.append(batch_predictions)
            saturation += int(((unclipped < 0.0) | (unclipped > 1.0)).sum())
            seen += unclipped.numel()
            mask = torch.isin(rows, fixed_rows)
            if bool(mask.any()):
                examples["rows"].append(rows[mask].cpu())
                examples["labels"].append(labels[rows[mask]].cpu())
                examples["targets"].append(target_view[mask.to(device)].cpu())
                examples["images"].append(view[mask.to(device)].cpu())
                example_psnr_parts.append(psnr[mask.to(device)].cpu())
    predicted = torch.cat(predicted_parts)
    summary = _classification(labels[dev_indices], predicted)
    psnr_all = torch.cat(psnr_parts)
    mse_all = torch.cat(mse_parts)
    ssim_all = torch.cat(ssim_parts)
    class_labels = torch.cat(class_labels_parts)
    rows_all = torch.cat(row_parts)
    clipping_all = torch.cat(clipping_parts)
    ed_all = predicted
    mse_by_class: list[float] = []
    psnr_by_class: list[float] = []
    ssim_by_class: list[float] = []
    psnr_count_by_class: list[int] = []
    ed_recall_by_class: list[float] = []
    for label in range(10):
        class_mask = class_labels == label
        class_mse = mse_all[class_mask]
        class_ssim = ssim_all[class_mask]
        class_psnr = psnr_all[class_labels == label]
        _require(class_psnr.numel() > 0, f"DEV reconstruction population is empty for class {label}")
        mse_by_class.append(float(class_mse.double().mean()))
        psnr_by_class.append(float(class_psnr.double().mean()))
        ssim_by_class.append(float(class_ssim.double().mean()))
        psnr_count_by_class.append(int(class_psnr.numel()))
        class_predictions = ed_all[class_mask]
        ed_recall_by_class.append(float((class_predictions == label).double().mean()))
    target_check = torch.cat(examples["targets"])
    example_check = torch.cat(examples["images"])
    independent = _independent_psnr(example_check, target_check)
    torch_reference = float(torch.cat(example_psnr_parts).double().mean())
    _require(math.isclose(independent, torch_reference, rel_tol=1e-6, abs_tol=1e-6),
             "independent NumPy PSNR reference differs from the production metric")
    summary: dict[str, Any] = {
        "population": int(dev_indices.numel()),
        "mse_mean": float(mse_all.mean()),
        "mse_by_class": mse_by_class,
        "psnr_db_mean": float(psnr_all.mean()),
        "psnr_db_by_class": psnr_by_class,
        "psnr_count_by_class": psnr_count_by_class,
        "ssim_mean": float(ssim_all.mean()),
        "ssim_by_class": ssim_by_class,
        "ed_macro_recall": summary["macro_recall"],
        "ed_recall_by_class": summary["recall_by_class"].tolist(),
        "confusion_matrix": summary["confusion_matrix"].tolist(),
        "saturation_fraction": saturation / seen,
        "common_view": "fixed gain, clamp to [0,1], adaptive average pool to 28x28",
        "common_target": "original MNIST uint8 28x28 / 255",
    }
    if collect_per_image:
        summary["per_image"] = [
            {
                "row": int(row),
                "label": int(label),
                "mse": float(mse),
                "psnr_db": float(psnr),
                "ssim": float(ssim_value),
                "ed_prediction": int(prediction),
                "ed_correct": bool(prediction == label),
                "proxy_clipping_fraction": float(clipping),
            }
            for row, label, mse, psnr, ssim_value, prediction, clipping in zip(
                rows_all.tolist(), class_labels.tolist(), mse_all.tolist(), psnr_all.tolist(),
                ssim_all.tolist(), ed_all.tolist(), clipping_all.tolist(), strict=True
            )
        ]
    return summary, {key: torch.cat(value) for key, value in examples.items()}


def _gauge_and_scale(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pivot = z.gather(1, z.abs().argmax(dim=1, keepdim=True))
    gauge = pivot / pivot.abs().clamp_min(1e-30)
    return z * gauge.conj() / scale.to(z.device).sqrt().unsqueeze(0)


def _fit_gaussians(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    arm: str,
    scale: torch.Tensor,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    chunks: dict[int, list[torch.Tensor]] = {label: [] for label in range(10)}
    model.eval()
    with torch.no_grad():
        for rows in train_indices.split(BATCH_SIZE):
            target = _image_batch(pixels, rows, arm, device)
            z = _forward(model, target)["ols_coefficients"]
            normalized = _gauge_and_scale(z.to(torch.complex128), scale)
            for label in range(10):
                mask = labels[rows] == label
                if bool(mask.any()):
                    chunks[label].append(normalized[mask.to(device=normalized.device)].cpu())
    fitted: dict[int, dict[str, torch.Tensor]] = {}
    for label in range(10):
        values = torch.cat(chunks[label])
        expected = int((labels[train_indices] == label).sum())
        _require(values.shape == (expected, 100), f"TRAIN prior fit population changed for class {label}")
        features = torch.view_as_real(values).reshape(expected, 200)
        mean = features.mean(0)
        centered = features - mean
        covariance = centered.T @ centered / float(expected - 1)
        ridge = covariance.diag().mean() * POWER_RIDGE
        regularized = covariance + ridge * torch.eye(200, dtype=torch.float64)
        cholesky, info = torch.linalg.cholesky_ex(regularized)
        _require(int(info) == 0, f"TRAIN class {label} covariance is not positive definite")
        fitted[label] = {"mean": mean, "covariance": covariance, "cholesky": cholesky, "ridge": ridge}
    return fitted


def _sample_gaussians(
    fitted: dict[int, dict[str, torch.Tensor]],
    scale: torch.Tensor,
    max_proposals_per_class: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if max_proposals_per_class is not None:
        _require(isinstance(max_proposals_per_class, int) and not isinstance(max_proposals_per_class, bool)
                 and max_proposals_per_class > 0,
                 "proposal limit must be a positive integer")
    output: list[torch.Tensor] = []
    requested: list[torch.Tensor] = []
    proposal_ids: list[torch.Tensor] = []
    accounting: dict[str, Any] = {}
    for label in range(10):
        generator = torch.Generator(device="cpu").manual_seed(SAMPLE_BASE_SEED + label)
        accepted_values: list[torch.Tensor] = []
        accepted_ids: list[torch.Tensor] = []
        accepted_count = 0
        valid_count = 0
        proposal_count = 0
        gaussian = fitted[label]
        while accepted_count < SAMPLE_COUNT:
            if max_proposals_per_class is not None:
                remaining_budget = max_proposals_per_class - proposal_count
                if remaining_budget <= 0:
                    raise SamplingPopulationIncomplete(
                        f"sampling population incomplete: class {label} retained {accepted_count}/"
                        f"{SAMPLE_COUNT} after {proposal_count} proposals; limit={max_proposals_per_class}"
                    )
                count = min(256, remaining_budget)
            else:
                count = 256
            noise = torch.randn((count, 200), generator=generator, dtype=torch.float64)
            draws = gaussian["mean"] + noise @ gaussian["cholesky"].T
            normalized = torch.view_as_complex(draws.reshape(count, 100, 2).contiguous())
            coefficients = normalized * scale.double().sqrt().unsqueeze(0)
            valid = coefficients.abs().square().sum(1) <= MAX_MODAL_POWER_W
            valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
            valid_count += int(valid.sum())
            remaining = SAMPLE_COUNT - accepted_count
            chosen_positions = valid_positions[:remaining]
            if chosen_positions.numel():
                accepted_values.append(coefficients[chosen_positions].to(torch.complex64))
                accepted_ids.append(chosen_positions + proposal_count)
                accepted_count += int(chosen_positions.numel())
            proposal_count += count
        values = torch.cat(accepted_values)
        ids = torch.cat(accepted_ids).long()
        _require(values.shape == (SAMPLE_COUNT, 100) and ids.shape == (SAMPLE_COUNT,),
                 f"sampler retained the wrong cohort for class {label}")
        output.append(values)
        requested.append(torch.full((SAMPLE_COUNT,), label, dtype=torch.long))
        proposal_ids.append(ids)
        accounting[str(label)] = {
            "trials": proposal_count,
            "proposal_limit": max_proposals_per_class,
            "valid_draws": valid_count,
            "retained_samples": accepted_count,
            "rejection_fraction": 1.0 - valid_count / proposal_count,
            "max_modal_power_w": float(values.abs().square().sum(1).max()),
            "proposal_ids": ids.tolist(),
            "noise_seed": SAMPLE_BASE_SEED + label,
        }
    return torch.cat(output), torch.cat(requested), torch.cat(proposal_ids), accounting


def _shape_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    a = first.flatten(1)
    b = second.flatten(1)
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    b = b / b.sum(1, keepdim=True).clamp_min(1e-12)
    return 0.5 * (a - b).abs().sum(1)


def _pair_distances(values: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(values.shape[0], (SHAPE_PAIR_COUNT,), generator=generator)
    offset = torch.randint(1, values.shape[0], (SHAPE_PAIR_COUNT,), generator=generator)
    second = (first + offset) % values.shape[0]
    return _shape_distance(values[first], values[second])


def _evaluate_generation(
    model: torch.nn.Module,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    samples: torch.Tensor,
    requested: torch.Tensor,
    proposal_ids: torch.Tensor,
    labels: torch.Tensor,
    pixels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    arm: str,
    device: torch.device,
    gain: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    images: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    clipping_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, samples.shape[0], BATCH_SIZE):
            result = model.forward_from_ols(samples[start : start + BATCH_SIZE].to(device))
            raw = result["physical_w_per_pixel"] * gain.to(device=device, dtype=torch.float32)
            view = _resize_view(raw)
            images.append(view.cpu())
            predictions.append(evaluator(view).argmax(1).cpu())
            clipping_parts.append((((raw < 0.0) | (raw > 1.0)).double().mean((1, 2, 3))).cpu())
    generated_images = torch.cat(images)
    predicted = torch.cat(predictions)
    clipping_fraction = torch.cat(clipping_parts)
    requested_cpu = requested.detach().cpu().long()
    classification = _classification(requested, predicted)
    generated_tv: list[float] = []
    dev_tv: list[float] = []
    ratios: list[float] = []
    dev_views = pixels.index_select(0, dev_indices).unsqueeze(1).float() / 255.0
    dev_labels = labels[dev_indices]
    for label in range(10):
        generated_class = generated_images[requested == label]
        dev_class = dev_views[dev_labels == label]
        generated_mean = float(_pair_distances(generated_class, 2026101603 + label).mean())
        dev_mean = float(_pair_distances(dev_class, 2026101703 + label).mean())
        generated_tv.append(generated_mean)
        dev_tv.append(dev_mean)
        ratios.append(generated_mean / max(dev_mean, 1e-12))
    # Preserve real, unenhanced 28x28 model views and their sampler identities.
    examples = {
        "images": generated_images,
        "requested_labels": requested.cpu(),
        "predicted_labels": predicted,
        "proposal_ids": proposal_ids.cpu(),
        "ols_coefficients": samples.cpu(),
        "proxy_clipping_fraction": clipping_fraction,
    }
    return {
        "population": int(requested.numel()),
        "samples_per_class": SAMPLE_COUNT,
        "macro_recall": classification["macro_recall"],
        "recall_by_class": classification["recall_by_class"].tolist(),
        "confusion_matrix": classification["confusion_matrix"].tolist(),
        "proxy_clipping_fraction_by_class": [
            float(clipping_fraction[requested_cpu == label].mean()) for label in range(10)
        ],
        "proxy_clipping_fraction_max_class": max(
            float(clipping_fraction[requested_cpu == label].mean()) for label in range(10)
        ),
        "generated_within_class_tv_mean": generated_tv,
        "dev_within_class_tv_mean": dev_tv,
        "generated_to_dev_tv_ratio": ratios,
        "generated_shape_pairs_per_class": SHAPE_PAIR_COUNT,
        "dev_shape_pairs_per_class": SHAPE_PAIR_COUNT,
        "sampling_power_bound_w": MAX_MODAL_POWER_W,
        "classifier_based_sample_selection": False,
    }, examples


def _window_error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    a = left.detach().cpu().to(torch.complex128).reshape(-1)
    b = right.detach().cpu().to(torch.complex128).reshape(-1)
    phase = torch.angle((a.conj() * b).sum())
    aligned = a * torch.exp(1j * phase)
    return float(torch.linalg.vector_norm(aligned - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)), float(phase)


def _small_producer_consumer_check(
    model: SmallMatchedAdapter,
    sample: torch.Tensor,
) -> dict[str, Any]:
    """Check the production OLS and camera readouts against direct NumPy sums."""
    if model.runtime.ols.basis_id != GAUSSIAN_OLS_MODEL:
        return {"status": "NOT_APPLICABLE_LP01"}
    with torch.no_grad():
        result = model.forward_from_image(sample)
    field = result["ols_input_field"].tensor.detach().cpu().numpy()[0, 0]
    operator = model.runtime.ols.operator
    modes_y = operator.modes_y.detach().cpu().numpy()
    modes_x = operator.modes_x.detach().cpu().numpy()
    dy, dx = operator.input_plane.pixel_pitch_m
    intermediate = np.einsum("rh,hw->rw", modes_y, field) * dy
    independent_coefficients = np.einsum("rw,cw->rc", intermediate, modes_x) * dx
    production_coefficients = result["ols_coefficients"].detach().cpu().numpy()[0].reshape(10, 10)
    coefficient_error = float(
        np.linalg.norm(independent_coefficients - production_coefficients)
        / max(np.linalg.norm(independent_coefficients), 1.0e-30)
    )
    camera_intensity = result["raw_intensity_w_per_m2"].detach().cpu().numpy()
    camera_dy, camera_dx = model.runtime.camera_plane.pixel_pitch_m
    independent_camera = camera_intensity * camera_dy * camera_dx
    production_camera = result["physical_w_per_pixel"].detach().cpu().numpy()
    camera_error = float(
        np.linalg.norm(independent_camera - production_camera)
        / max(np.linalg.norm(independent_camera), 1.0e-30)
    )
    _require(math.isfinite(coefficient_error) and coefficient_error <= 1.0e-5,
             "independent NumPy Gaussian OLS coefficient projection disagrees with production")
    _require(math.isfinite(camera_error) and camera_error <= 1.0e-6,
             "independent full-camera W/pixel readout disagrees with production")
    return {
        "status": "PASSED",
        "ols_coefficient_relative_l2_error": coefficient_error,
        "camera_w_per_pixel_relative_l2_error": camera_error,
        "port_pitch_m": 20.0e-6,
        "field_1e_radius_m": 3.5e-6,
        "camera_pixel_area_m2": camera_dy * camera_dx,
    }


def _small_window_check(
    sample: torch.Tensor,
    grids: list[int],
    device: torch.device,
    phase_state: dict[str, torch.Tensor] | None = None,
    ols_model: str = LP01_OLS_MODEL,
    production_grid: int | None = None,
) -> tuple[int, dict[str, Any]]:
    all_fields: dict[int, dict[str, torch.Tensor]] = {}
    for grid in grids:
        clear_production_transfer_cache()
        candidate = SmallMatchedAdapter(grid, ols_model).to(device).eval()
        if phase_state is not None:
            candidate.load_state_dict(phase_state)
        with torch.no_grad():
            out = candidate.forward_from_image(sample)
        fields: dict[str, torch.Tensor] = {}
        common_roi_pixels = int(round(1.6e-4 / 2.0e-6))

        def common_roi(field: torch.Tensor) -> torch.Tensor:
            y = (field.shape[-2] - common_roi_pixels) // 2
            x = (field.shape[-1] - common_roi_pixels) // 2
            return field[..., y : y + common_roi_pixels, x : x + common_roi_pixels]

        for index, field in enumerate(out["encoder_fields"]):
            fields[f"encoder_{index + 1}"] = common_roi(field.tensor)
        for index, field in enumerate(out["decoder_fields"]):
            fields[f"generator_{index + 1}"] = common_roi(field.tensor)
        fields["ols_input"] = common_roi(out["ols_input_field"].tensor)
        fields["generator_entry"] = common_roi(out["generator_entry_field"].tensor)
        fields["generator_exit"] = common_roi(out["generator_exit_field"].tensor)
        fields["ols_coefficients"] = out["ols_coefficients"]
        fields["camera_w_per_pixel"] = out["physical_w_per_pixel"]
        all_fields[grid] = {name: value.detach().cpu() for name, value in fields.items()}
        del candidate, out
        torch.cuda.empty_cache()
    reference = all_fields[grids[-1]]
    comparisons: dict[int, dict[str, float]] = {grids[-1]: {name: 0.0 for name in reference}}
    phase_offsets: dict[int, dict[str, float]] = {grids[-1]: {name: 0.0 for name in reference}}
    for grid in grids[:-1]:
        comparisons[grid] = {}
        phase_offsets[grid] = {}
        for name in reference:
            error, phase = _window_error(all_fields[grid][name], reference[name])
            comparisons[grid][name] = error
            phase_offsets[grid][name] = phase
    accepted = [grid for grid in grids if max(comparisons[grid].values()) <= WINDOW_TOLERANCE]
    _require(bool(accepted), "small architecture has no propagation grid within the 1e-4 window tolerance")
    field_names = (
        "encoder_1", "encoder_2", "generator_1", "generator_2",
        "ols_input", "generator_entry", "generator_exit",
    )
    field_error_by_grid = {
        grid: max(comparisons[grid][name] for name in field_names) for grid in comparisons
    }
    coefficient_error_by_grid = {
        grid: comparisons[grid]["ols_coefficients"] for grid in comparisons
    }
    camera_error_by_grid = {
        grid: comparisons[grid]["camera_w_per_pixel"] for grid in comparisons
    }
    if production_grid is None:
        selected = min(grid for grid in grids if field_error_by_grid[grid] <= WINDOW_TOLERANCE)
    else:
        _require(production_grid in grids and production_grid != grids[-1],
                 "configured Gaussian production grid must precede its reference grid")
        _require(field_error_by_grid[production_grid] <= WINDOW_TOLERANCE,
                 "configured Gaussian production grid failed the complex-window tolerance")
        _require(coefficient_error_by_grid[production_grid] <= WINDOW_TOLERANCE,
                 "configured Gaussian production grid failed the OLS coefficient comparison")
        _require(camera_error_by_grid[production_grid] <= WINDOW_TOLERANCE,
                 "configured Gaussian production grid failed the full-camera W/pixel comparison")
        selected = production_grid
    return selected, {
        "status": "PASSED",
        "reference_grid": grids[-1],
        "selected_grid": selected,
        "common_roi_m": 1.6e-4,
        "complex_relative_error_max": field_error_by_grid[selected],
        "common_roi_pixels": common_roi_pixels,
        "complex_relative_error_vs_reference": comparisons,
        "aligned_global_phase_rad": phase_offsets,
        "grid_candidates": grids,
        "production_grid_fixed": production_grid is not None,
        "ols_coefficient_relative_error_by_grid": coefficient_error_by_grid,
        "camera_w_per_pixel_relative_error_by_grid": camera_error_by_grid,
    }


def _assert_row_negative_control(rows: torch.Tensor) -> dict[str, Any]:
    try:
        _reconstruction_rows(rows, rows.roll(1))
    except ValueError as error:
        return {"passed": True, "rejected_error": str(error), "mismatch_rows": True}
    raise RuntimeError("mismatched MNIST row negative control was not rejected")


def _finite_difference_check(
    model: torch.nn.Module,
    target: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, dict[str, Any]]:
    parts, _ = _loss_components(model, target, scale, power_reference, camera_gain)
    objective = sum(weights[name] * parts[name] for name in ("dist", "rec", "op"))
    groups = _phase_groups(model)
    gradients = torch.autograd.grad(objective, groups)
    del parts, objective
    epsilon = 1.0e-2
    diagnostics: dict[str, dict[str, float | int]] = {}
    for name, parameter, gradient in zip(("encoder", "decoder"), groups, gradients, strict=True):
        gradient64 = gradient.detach().to(torch.float64)
        gradient_norm = torch.linalg.vector_norm(gradient64)
        _require(torch.isfinite(gradient_norm).item() and gradient_norm > 0.0,
                 f"{name} finite-difference direction has an invalid gradient norm")
        direction = (gradient64 / gradient_norm).to(dtype=parameter.dtype)
        analytic = float((gradient64 * direction.to(torch.float64)).sum())
        original = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter.add_(direction, alpha=epsilon)
            plus_parts, _ = _loss_components(model, target, scale, power_reference, camera_gain)
            plus = float(sum(weights[key] * plus_parts[key] for key in ("dist", "rec", "op")).detach())
            del plus_parts
            with torch.no_grad():
                parameter.add_(direction, alpha=-2.0 * epsilon)
            minus_parts, _ = _loss_components(model, target, scale, power_reference, camera_gain)
            minus = float(sum(weights[key] * minus_parts[key] for key in ("dist", "rec", "op")).detach())
            del minus_parts
        finally:
            with torch.no_grad():
                parameter.copy_(original)
        del original
        finite = (plus - minus) / (2.0 * epsilon)
        relative = abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-7)
        _require(math.isfinite(finite) and analytic * finite > 0.0 and relative <= 0.05,
                 f"{name} normalized group-direction finite difference did not agree: "
                 f"autograd={analytic:.12g}, central={finite:.12g}, relative_error={relative:.6g}, "
                 f"plus={plus:.12g}, minus={minus:.12g}")
        diagnostics[name] = {
            "direction": "unit_l2_autograd_group_gradient",
            "parameter_count": int(parameter.numel()),
            "direction_l2_norm": float(torch.linalg.vector_norm(direction.to(torch.float64))),
            "epsilon_radians": epsilon,
            "autograd": analytic,
            "central_difference": finite,
            "relative_error": relative,
        }
    return diagnostics


def _preflight_recovery(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    arm: str,
    seed: int,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
    run_id: str,
    staging_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    model_identity: dict[str, Any],
) -> dict[str, Any]:
    task_id = config.get("task_id", TASK_ID)
    model_initial = _state_snapshot(model)
    optimizer_initial = copy.deepcopy(optimizer.state_dict())
    rng_initial = torch.cuda.get_rng_state_all()
    cpu_rng_initial = torch.random.get_rng_state()
    trial_losses: list[dict[str, float]] = []
    for index in range(3):
        trial_losses.append(_train_update(
            model, optimizer, pixels, rows[index], arm, device,
            scale, power_reference, camera_gain, weights,
        ))
    trial_model = _state_snapshot(model)
    trial_optimizer = _cpu_copy(optimizer.state_dict())
    checkpoint = {
        "schema": "lightgen_matched_architecture_preflight_recovery.v1",
        "task_id": task_id,
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "model_identity": model_identity,
        "config_identity": config,
        "completed_updates": 3,
        "propagation_grid_size": model.config.propagation_grid_size,
        "model": trial_model,
        "optimizer": trial_optimizer,
        "schedule": rows.detach().cpu(),
        "scale": scale.detach().cpu(),
        "power_reference": power_reference.detach().cpu(),
        "camera_gain": camera_gain.detach().cpu(),
        "weights": weights,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
    }
    recovery_path = staging_dir / "preflight_step_000003.pt"
    atomic_torch_save_new(recovery_path, checkpoint)
    checkpoint = torch.load(recovery_path, map_location="cpu", weights_only=False)
    _require(
        checkpoint.get("schema") == "lightgen_matched_architecture_preflight_recovery.v1"
        and checkpoint.get("task_id") == task_id
        and checkpoint.get("run_id") == run_id
        and checkpoint.get("arm") == arm
        and checkpoint.get("seed") == seed
        and checkpoint.get("model_identity") == model_identity
        and checkpoint.get("config_identity") == config
        and checkpoint.get("completed_updates") == 3,
        "saved three-step preflight checkpoint identity changed",
    )
    # Compare the uninterrupted fourth update with a new operating-system process.
    uninterrupted_loss = _train_update(
        model, optimizer, pixels, rows[3], arm, device,
        scale, power_reference, camera_gain, weights,
    )
    uninterrupted_model = _state_snapshot(model)
    uninterrupted_optimizer = _cpu_copy(optimizer.state_dict())
    torch.cuda.empty_cache()
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--recovery-worker-dir",
        str(staging_dir),
        "--config",
        str(config_path),
        "--arm",
        arm,
        "--seed",
        str(seed),
    ]
    worker = subprocess.run(worker_command, capture_output=True, text=True, check=False)
    _require(worker.returncode == 0,
             "independent-process recovery worker failed: " + (worker.stderr[-3000:] or worker.stdout[-3000:]))
    worker_path = staging_dir / "preflight_fourth_update_independent_process.pt"
    _require(worker_path.is_file(), "independent-process recovery output is missing")
    recovered = torch.load(worker_path, map_location="cpu", weights_only=False)
    _require(
        recovered.get("task_id") == task_id
        and recovered.get("run_id") == run_id
        and recovered.get("arm") == arm
        and recovered.get("completed_updates") == 4,
        "independent-process recovery output identity changed",
    )
    model_error = _assert_close(uninterrupted_model, recovered["model"], "recovery_model")
    optimizer_error = _assert_close(uninterrupted_optimizer, recovered["optimizer"], "recovery_optimizer")
    _assert_close(uninterrupted_loss, recovered["loss"], "recovery_loss")
    model.load_state_dict(model_initial)
    optimizer.load_state_dict(optimizer_initial)
    torch.cuda.set_rng_state_all(rng_initial)
    torch.random.set_rng_state(cpu_rng_initial)
    _require(_assert_close(model_initial, _state_snapshot(model), "rolled_back_model") == 0.0,
             "preflight model updates were not rolled back")
    _assert_close(optimizer_initial, optimizer.state_dict(), "rolled_back_optimizer")
    _assert_close(rng_initial, torch.cuda.get_rng_state_all(), "rolled_back_cuda_rng")
    _assert_close(cpu_rng_initial, torch.random.get_rng_state(), "rolled_back_cpu_rng")
    return {
        "status": "PASSED",
        "trial_updates": 3,
        "engineering_adam_calls": 5,
        "trial_losses": trial_losses,
        "independent_process_checkpoint_reload_next_update": True,
        "independent_worker_stdout": worker.stdout[-2000:],
        "saved_checkpoint": recovery_path.name,
        "max_model_recovery_error": model_error,
        "max_optimizer_recovery_error": optimizer_error,
        "preflight_updates_rolled_back": True,
    }


def _run_recovery_worker(staging_dir: Path, config_path: Path, arm: str, seed: int) -> None:
    checkpoint_path = staging_dir / "preflight_step_000003.pt"
    _require(checkpoint_path.is_file(), "independent recovery worker cannot find the saved checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _read_config(config_path)
    task_id = config.get("task_id", TASK_ID)
    _require(checkpoint.get("task_id") == task_id, "recovery worker checkpoint task identity changed")
    _require(checkpoint.get("arm") == arm and checkpoint.get("seed") == seed,
             "recovery worker checkpoint arm or seed changed")
    _require(checkpoint.get("config_identity") == config, "recovery worker checkpoint config changed")
    _require(checkpoint.get("model_identity") == _model_identity(config, arm),
             "recovery worker checkpoint model identity changed")
    _require(checkpoint.get("completed_updates") == 3, "recovery worker requires the saved three-update state")
    _require(torch.cuda.is_available(), "independent recovery worker requires CUDA")
    device = torch.device("cuda")
    pixels, labels, train_indices, _ = _load_data(ROOT / config["data"]["training_file"])
    expected_schedule = _schedule(labels, train_indices, seed)
    schedule = checkpoint.get("schedule")
    _require(isinstance(schedule, torch.Tensor) and torch.equal(schedule, expected_schedule),
             "recovery worker schedule differs from the fixed TRAIN population")
    grid = int(checkpoint["propagation_grid_size"])
    model = _model(arm, grid, device, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.Adam(
        _phase_groups(model), lr=1e-3, betas=TRAIN_BETAS,
        eps=TRAIN_EPSILON, weight_decay=0.0,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.cuda.set_rng_state_all([value.to(device="cpu") for value in checkpoint["cuda_rng"]])
    torch.random.set_rng_state(checkpoint["cpu_rng"])
    loss = _train_update(
        model, optimizer, pixels, schedule[3], arm, device,
        checkpoint["scale"], checkpoint["power_reference"],
        checkpoint["camera_gain"], checkpoint["weights"],
    )
    worker_path = staging_dir / "preflight_fourth_update_independent_process.pt"
    atomic_torch_save_new(worker_path, {
        "task_id": task_id,
        "run_id": checkpoint["run_id"],
        "arm": arm,
        "seed": seed,
        "completed_updates": 4,
        "model": _state_snapshot(model),
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "loss": loss,
    })
    print(json.dumps({"status": "PASSED", "worker": "independent_process", "output": str(worker_path)}), flush=True)


def _model_identity(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == "small":
        basis = config["optics"]["small"].get("ols_basis", LP01_OLS_MODEL)
        return {
            "encoder": [2, 100, 100],
            "ols": basis,
            "ols_geometry": {
                "port_pitch_m": config["optics"]["small"].get("port_pitch_m", 7.111e-6),
                "field_1e_radius_m": config["optics"]["small"].get("field_1e_radius_m", 7.111e-6),
                "transport": config["optics"]["small"].get("ols_transport", "identity"),
            },
            "decoder": [2, 100, 100],
        }
    return {
        "encoder": [2, 400, 400],
        "ols": "gaussian_s12_global_100_ports_identity",
        "decoder": [4, 400, 400],
    }


def _make_identity(config: dict[str, Any], arm: str, seed: int, run_id: str) -> ResearchRunIdentity:
    task_id = config.get("task_id", TASK_ID)
    authority = config.get("semantic_authority_id", AUTHORITY)
    return ResearchRunIdentity(
        task_id=task_id,
        run_id=run_id,
        trajectory_id=f"{task_id}:{arm}:schedule:{seed}",
        semantic_authority_id=authority,
        data_role="MNIST_TRAIN_55000_DEV_5000_OFFICIAL_TEST_UNREAD",
        config_identity=config,
        model_identity=_model_identity(config, arm),
        optimizer_identity={"name": "Adam", "lr": 1e-3, "betas": TRAIN_BETAS, "epsilon": TRAIN_EPSILON, "weight_decay": 0.0},
        sampler_identity={"paired_schedule_seed": seed, "batch_size": BATCH_SIZE, "updates": UPDATES},
        execution_source_semantics_id=config.get("execution_source_semantics_id", "matched_architecture_v1"),
        source_manifest_id=None,
    )


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    step: int,
    arm: str,
    seed: int,
    run_id: str,
    schedule: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    camera_gain: torch.Tensor,
    weights: dict[str, float],
    history: list[dict[str, Any]],
    dev_metrics: list[dict[str, Any]],
    task_id: str,
    model_identity: dict[str, Any],
    config_identity: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_matched_architecture_checkpoint.v1",
        "task_id": task_id,
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "model_identity": model_identity,
        "config_identity": config_identity,
        "propagation_grid_size": model.config.propagation_grid_size,
        "completed_updates": step,
        "model": _state_snapshot(model),
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "schedule": schedule,
        "scale": scale.cpu(),
        "power_reference": power_reference.cpu(),
        "camera_gain": camera_gain.cpu(),
        "weights": weights,
        "history": history,
        "dev_metrics": dev_metrics,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
        "evaluator_role": "frozen_confirmation_only",
        "elapsed_seconds": elapsed_seconds,
    }


def _optimizer_update_count(optimizer: torch.optim.Adam, model: torch.nn.Module) -> int:
    steps: list[int] = []
    for parameter in _phase_groups(model):
        state = optimizer.state.get(parameter)
        _require(isinstance(state, dict) and "step" in state,
                 "Adam checkpoint is missing state for a trainable phase group")
        step = state["step"]
        steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    _require(len(set(steps)) == 1, "Adam phase groups have different update counts")
    return steps[0]


def _save_checkpoint(run_dir: Path, payload: dict[str, Any]) -> None:
    latest = run_dir / "checkpoint_latest.pt"
    previous = run_dir / "checkpoint_previous.pt"
    if latest.exists():
        atomic_torch_save(previous, torch.load(latest, map_location="cpu", weights_only=False))
    atomic_torch_save(latest, payload)


def _evaluate_and_save(
    model: torch.nn.Module,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    dev_indices: torch.Tensor,
    arm: str,
    device: torch.device,
    camera_gain: torch.Tensor,
    step: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    result, examples = _evaluate_reconstruction(
        model, evaluator, pixels, labels, dev_indices, arm, device, camera_gain,
        collect_per_image=step == UPDATES,
    )
    result["step"] = step
    return result, examples


def _score_saved_views(
    views: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    device: torch.device,
    *,
    source_rows: torch.Tensor | None = None,
    clipping_fraction: torch.Tensor | None = None,
) -> dict[str, Any]:
    if views.shape != targets.shape or views.ndim != 4 or views.shape[1:] != (1, 28, 28):
        raise ValueError("saved comparison views must match [N,1,28,28] targets")
    if labels.shape != (views.shape[0],) or rows.shape != labels.shape:
        raise ValueError("saved comparison rows and labels do not match the view population")
    if source_rows is not None and source_rows.shape != rows.shape:
        raise ValueError("mismatched-input source rows do not match the target population")
    mse_parts: list[torch.Tensor] = []
    psnr_parts: list[torch.Tensor] = []
    ssim_parts: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    evaluator.eval()
    with torch.no_grad():
        for start in range(0, views.shape[0], BATCH_SIZE):
            view = views[start : start + BATCH_SIZE].to(device=device, dtype=torch.float32)
            target = targets[start : start + BATCH_SIZE].to(device=device, dtype=torch.float32)
            mse = (view - target).square().mean((1, 2, 3)).clamp_min(1.0e-12)
            mse_parts.append(mse.cpu())
            psnr_parts.append((-10.0 * torch.log10(mse)).cpu())
            ssim_parts.append(_ssim_batch(view, target).cpu())
            predictions.append(evaluator(view).argmax(1).cpu())
    mse_all = torch.cat(mse_parts)
    psnr_all = torch.cat(psnr_parts)
    ssim_all = torch.cat(ssim_parts)
    prediction_all = torch.cat(predictions)
    labels_cpu = labels.detach().cpu().long()
    classification = _classification(labels_cpu, prediction_all)
    per_image: list[dict[str, Any]] = []
    for index in range(rows.numel()):
        label = int(labels_cpu[index])
        item: dict[str, Any] = {
            "row": int(rows[index]),
            "label": label,
            "mse": float(mse_all[index]),
            "psnr_db": float(psnr_all[index]),
            "ssim": float(ssim_all[index]),
            "ed_prediction": int(prediction_all[index]),
            "ed_correct": bool(prediction_all[index] == label),
        }
        if source_rows is not None:
            item["source_row"] = int(source_rows[index])
        if clipping_fraction is not None:
            item["proxy_clipping_fraction"] = float(clipping_fraction[index])
        per_image.append(item)
    return {
        "population": int(rows.numel()),
        "mse_mean": float(mse_all.mean()),
        "mse_by_class": [
            float(mse_all[labels_cpu == label].mean()) for label in range(10)
        ],
        "psnr_db_mean": float(psnr_all.mean()),
        "psnr_db_by_class": [
            float(psnr_all[labels_cpu == label].mean()) for label in range(10)
        ],
        "ssim_mean": float(ssim_all.mean()),
        "ssim_by_class": [
            float(ssim_all[labels_cpu == label].mean()) for label in range(10)
        ],
        "ed_macro_recall": classification["macro_recall"],
        "ed_recall_by_class": classification["recall_by_class"].tolist(),
        "confusion_matrix": classification["confusion_matrix"].tolist(),
        "per_image": per_image,
    }


def _stratified_paired_bootstrap(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    metric: str,
    direction: str,
) -> dict[str, Any]:
    first_by_row = {int(row["row"]): row for row in first}
    second_by_row = {int(row["row"]): row for row in second}
    _require(first_by_row.keys() == second_by_row.keys(), "paired bootstrap row populations differ")
    differences: list[np.ndarray] = []
    for label in range(10):
        rows = sorted(row_id for row_id in first_by_row if int(first_by_row[row_id]["label"]) == label)
        _require(len(rows) == 500, f"paired bootstrap class {label} population changed")
        left = np.asarray([float(first_by_row[row][metric]) for row in rows], dtype=np.float64)
        right = np.asarray([float(second_by_row[row][metric]) for row in rows], dtype=np.float64)
        delta = left - right if direction == "first_minus_second" else right - left
        differences.append(delta)
    observed = float(np.mean([values.mean() for values in differences]))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        class_means = [values[generator.integers(0, values.size, size=values.size)].mean() for values in differences]
        boot[index] = np.mean(class_means)
    return {
        "estimand": f"equal-class mean {metric} difference ({direction})",
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "strata": "10 MNIST classes, 500 paired rows sampled with replacement within each class",
    }


def _evaluate_reconstruction_comparisons(
    model: SmallMatchedAdapter,
    evaluator: PixelMLPConfirmationEvaluatorV1,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    gain: torch.Tensor,
    primary_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    targets = pixels.index_select(0, dev_indices).unsqueeze(1).float() / 255.0
    dev_labels = labels.index_select(0, dev_indices).cpu()
    global_mean = pixels.index_select(0, train_indices).double().mean(0).float() / 255.0
    mean_views = global_mean[None, None].expand(dev_indices.numel(), 1, 28, 28).contiguous()
    mean_rows = _score_saved_views(
        mean_views, targets, dev_labels, dev_indices.cpu(), evaluator, device
    )

    source_rows = dev_indices.clone()
    for label in range(10):
        class_rows = dev_indices[labels[dev_indices] == label]
        _require(class_rows.numel() == 500, f"DEV class {label} mismatch-control population changed")
        source_rows[labels[source_rows] == label] = class_rows.roll(1)
    mismatch_views: list[torch.Tensor] = []
    mismatch_clipping: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dev_indices.numel(), BATCH_SIZE):
            inputs = source_rows[start : start + BATCH_SIZE]
            native = _image_batch(pixels, inputs, "small", device)
            output = _forward(model, native)
            raw = output["physical_w_per_pixel"] * gain.to(device=device, dtype=torch.float32)
            mismatch_views.append(_resize_view(raw).cpu())
            mismatch_clipping.append((((raw < 0.0) | (raw > 1.0)).double().mean((1, 2, 3))).cpu())
    mismatch_rows = _score_saved_views(
        torch.cat(mismatch_views), targets, dev_labels, dev_indices.cpu(), evaluator, device,
        source_rows=source_rows.cpu(), clipping_fraction=torch.cat(mismatch_clipping),
    )
    _require(all(int(row["row"]) != int(row["source_row"]) for row in mismatch_rows["per_image"]),
             "fixed within-class mismatch control retained an identity input")
    _require(all(int(row["label"]) == int(labels[row["source_row"]]) for row in mismatch_rows["per_image"]),
             "fixed mismatch control crossed a class boundary")
    bootstrap = {
        "global_mean_minus_primary_mse": _stratified_paired_bootstrap(
            mean_rows["per_image"], primary_rows, "mse", "first_minus_second"
        ),
        "primary_minus_global_mean_psnr_db": _stratified_paired_bootstrap(
            primary_rows, mean_rows["per_image"], "psnr_db", "first_minus_second"
        ),
        "primary_minus_global_mean_ssim": _stratified_paired_bootstrap(
            primary_rows, mean_rows["per_image"], "ssim", "first_minus_second"
        ),
        "same_class_mismatch_minus_primary_mse": _stratified_paired_bootstrap(
            mismatch_rows["per_image"], primary_rows, "mse", "first_minus_second"
        ),
        "primary_minus_same_class_mismatch_psnr_db": _stratified_paired_bootstrap(
            primary_rows, mismatch_rows["per_image"], "psnr_db", "first_minus_second"
        ),
        "primary_minus_same_class_mismatch_ssim": _stratified_paired_bootstrap(
            primary_rows, mismatch_rows["per_image"], "ssim", "first_minus_second"
        ),
    }
    return {
        "global_mean_image_definition": "one pixelwise mean image from all 55,000 TRAIN targets at native 28x28 / 255",
        "global_mean_image_metrics": mean_rows,
        "same_class_mismatched_input_metrics": mismatch_rows,
        "paired_bootstrap": bootstrap,
    }, {"global_mean_image_28x28": global_mean, "mismatched_source_rows": source_rows.cpu()}


def _quality_decision(
    config: dict[str, Any],
    reconstruction: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["quality_gate"]
    checks = {
        "reconstruction_psnr": {
            "value": reconstruction["psnr_db_mean"],
            "operator": ">=",
            "threshold": thresholds["reconstruction_psnr_db_min"],
            "passed": reconstruction["psnr_db_mean"] >= thresholds["reconstruction_psnr_db_min"],
        },
        "reconstruction_ssim": {
            "value": reconstruction["ssim_mean"],
            "operator": ">=",
            "threshold": thresholds["reconstruction_ssim_min"],
            "passed": reconstruction["ssim_mean"] >= thresholds["reconstruction_ssim_min"],
        },
        "generation_macro_recall": {
            "value": generation["macro_recall"],
            "operator": ">=",
            "threshold": thresholds["generation_macro_recall_min"],
            "passed": generation["macro_recall"] >= thresholds["generation_macro_recall_min"],
        },
        "generation_recall_by_class": {
            "values": generation["recall_by_class"],
            "operator": ">= all",
            "threshold": thresholds["generation_recall_per_class_min"],
            "passed": min(generation["recall_by_class"]) >= thresholds["generation_recall_per_class_min"],
        },
        "reconstruction_proxy_clipping": {
            "value": reconstruction["saturation_fraction"],
            "operator": "<=",
            "threshold": thresholds["proxy_clipping_fraction_max"],
            "passed": reconstruction["saturation_fraction"] <= thresholds["proxy_clipping_fraction_max"],
        },
        "generation_proxy_clipping_by_class": {
            "values": generation["proxy_clipping_fraction_by_class"],
            "operator": "<= all",
            "threshold": thresholds["proxy_clipping_fraction_max"],
            "passed": max(generation["proxy_clipping_fraction_by_class"])
            <= thresholds["proxy_clipping_fraction_max"],
        },
        "generated_to_dev_shape_ratio": {
            "values": generation["generated_to_dev_tv_ratio"],
            "operator": ">= all",
            "threshold": thresholds["generated_to_dev_tv_ratio_min"],
            "passed": min(generation["generated_to_dev_tv_ratio"])
            >= thresholds["generated_to_dev_tv_ratio_min"],
        },
    }
    passed = all(value["passed"] for value in checks.values())
    return {
        "status": "VALID_COMPLETED_QUALITY_PASSED" if passed else "VALID_FAILED",
        "passed": passed,
        "thresholds": thresholds,
        "checks": checks,
    }


def _write_reconstruction_csv(
    path: Path,
    primary: list[dict[str, Any]],
    mean_baseline: list[dict[str, Any]],
    mismatch: list[dict[str, Any]],
) -> None:
    by_row = [
        {int(row["row"]): row for row in records}
        for records in (primary, mean_baseline, mismatch)
    ]
    _require(all(mapping.keys() == by_row[0].keys() for mapping in by_row[1:]),
             "reconstruction CSV populations differ across fixed comparisons")
    _require(len(by_row[0]) == DEV_COUNT, "reconstruction CSV does not contain all DEV rows")
    fields = [
        "row", "label", "mse", "psnr_db", "ssim", "ed_prediction", "ed_correct",
        "proxy_clipping_fraction", "global_mean_mse", "global_mean_psnr_db",
        "global_mean_ssim", "global_mean_ed_prediction", "global_mean_ed_correct",
        "same_class_mismatch_source_row", "same_class_mismatch_mse",
        "same_class_mismatch_psnr_db", "same_class_mismatch_ssim",
        "same_class_mismatch_ed_prediction", "same_class_mismatch_ed_correct",
        "same_class_mismatch_proxy_clipping_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_id in sorted(by_row[0]):
            current, mean, mismatched = (mapping[row_id] for mapping in by_row)
            _require(current["label"] == mean["label"] == mismatched["label"],
                     f"reconstruction CSV label mismatch at row {row_id}")
            writer.writerow({
                "row": row_id,
                "label": current["label"],
                "mse": current["mse"],
                "psnr_db": current["psnr_db"],
                "ssim": current["ssim"],
                "ed_prediction": current["ed_prediction"],
                "ed_correct": current["ed_correct"],
                "proxy_clipping_fraction": current["proxy_clipping_fraction"],
                "global_mean_mse": mean["mse"],
                "global_mean_psnr_db": mean["psnr_db"],
                "global_mean_ssim": mean["ssim"],
                "global_mean_ed_prediction": mean["ed_prediction"],
                "global_mean_ed_correct": mean["ed_correct"],
                "same_class_mismatch_source_row": mismatched["source_row"],
                "same_class_mismatch_mse": mismatched["mse"],
                "same_class_mismatch_psnr_db": mismatched["psnr_db"],
                "same_class_mismatch_ssim": mismatched["ssim"],
                "same_class_mismatch_ed_prediction": mismatched["ed_prediction"],
                "same_class_mismatch_ed_correct": mismatched["ed_correct"],
                "same_class_mismatch_proxy_clipping_fraction": mismatched["proxy_clipping_fraction"],
            })


def _write_contact_sheet(
    path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    images_per_class: int,
) -> None:
    values = images.detach().cpu().float()
    class_ids = labels.detach().cpu().long()
    cell = 112
    margin = 8
    sheet = Image.new("L", (10 * cell, images_per_class * cell), color=255)
    draw = ImageDraw.Draw(sheet)
    for label in range(10):
        selected = values[class_ids == label][:images_per_class]
        _require(selected.shape[0] == images_per_class, f"preview class {label} has too few fixed samples")
        for image_index in range(images_per_class):
            x = label * cell
            y = image_index * cell
            image = (selected[image_index, 0].clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
            tile = Image.fromarray(image, mode="L").resize((cell - 2 * margin, cell - 2 * margin), Image.Resampling.NEAREST)
            sheet.paste(tile, (x + margin, y + margin))
            if image_index == 0:
                draw.text((x + margin, y + 1), str(label), fill=0)
    sheet.save(path, format="PNG", optimize=True)


def _write_results_html(
    path: Path,
    report: dict[str, Any],
    generation: dict[str, Any] | None,
) -> None:
    status = html.escape(str(report["status"]))
    recon = report.get("reconstruction", [])
    endpoint = recon[-1] if recon else {}
    comparison = report.get("reconstruction_comparisons", {})
    mean = comparison.get("global_mean_image_metrics", {})
    mismatch = comparison.get("same_class_mismatched_input_metrics", {})
    quality = report.get("quality_gate", {})
    lines = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>LightGen Gaussian小架构系统基线</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}"
        "table{border-collapse:collapse}td,th{padding:.45rem .7rem;border:1px solid #ccd1d1;text-align:left}"
        "img{max-width:100%;height:auto;border:1px solid #ccd1d1}.status{font-weight:700;color:#154360}</style>",
        f"<h1>Gaussian小架构系统基线</h1><p class='status'>状态：{status}</p>",
        f"<p>步数：{report.get('training', {}).get('updates', '未完成')}；总耗时：{report.get('elapsed_seconds', 0):.1f}秒</p>",
        "<h2>重构端点（step 30000）</h2>",
        f"<p>PSNR {endpoint.get('psnr_db_mean', '—')} dB；SSIM {endpoint.get('ssim_mean', '—')}；"
        f"E_D macro recall {endpoint.get('ed_macro_recall', '—')}；代理截断 {endpoint.get('saturation_fraction', '—')}</p>",
        "<table><tr><th>对照</th><th>MSE</th><th>PSNR dB</th><th>SSIM</th><th>E_D宏召回</th></tr>",
        f"<tr><td>当前输入重构</td><td>{endpoint.get('mse_mean', '—')}</td><td>{endpoint.get('psnr_db_mean', '—')}</td><td>{endpoint.get('ssim_mean', '—')}</td><td>{endpoint.get('ed_macro_recall', '—')}</td></tr>",
        f"<tr><td>55k TRAIN全局均值图</td><td>{mean.get('mse_mean', '—')}</td><td>{mean.get('psnr_db_mean', '—')}</td><td>{mean.get('ssim_mean', '—')}</td><td>{mean.get('ed_macro_recall', '—')}</td></tr>",
        f"<tr><td>类内错配输入</td><td>{mismatch.get('mse_mean', '—')}</td><td>{mismatch.get('psnr_db_mean', '—')}</td><td>{mismatch.get('ssim_mean', '—')}</td><td>{mismatch.get('ed_macro_recall', '—')}</td></tr></table>",
        "<h2>固定展示样本</h2><p>横轴按类别0至9排列；每类固定展示前两张DEV目标与重构。</p>",
        "<p>DEV目标</p><img src='dev_target_preview.png' alt='每类前两张DEV目标'>",
        "<p>模型重构</p><img src='dev_preview.png' alt='每类前两张DEV重构'>",
    ]
    if generation is not None:
        lines.extend([
            "<h2>条件生成</h2>",
            f"<p>样本 {generation.get('population', 0)}；E_D宏召回 {generation.get('macro_recall', '—')}；"
            f"各类形态比最小值 {min(generation.get('generated_to_dev_tv_ratio', [0]))}</p>",
            "<p>每类按采样序号展示前十张，无分类器筛选。</p><img src='generation_preview.png' alt='每类前十张生成图'>",
        ])
    lines.append("<h2>质量判定</h2><pre>" + html.escape(json.dumps(quality, ensure_ascii=False, indent=2)) + "</pre>")
    lines.append("</html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_evaluator(path: Path, device: torch.device) -> PixelMLPConfirmationEvaluatorV1:
    _require(path.is_file(), f"frozen evaluator checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and payload.get("schema") == ED_SCHEMA, "frozen evaluator schema changed")
    _require(payload.get("authority_status") == "QUALIFIED", "frozen confirmation evaluator is not qualified")
    _require(
        payload.get("independence", {}).get("role") == "INDEPENDENT_CONFIRMATION_ONLY",
        "frozen confirmation evaluator role changed",
    )
    evaluator = PixelMLPConfirmationEvaluatorV1().to(device)
    evaluator.load_state_dict(payload["state_dict"], strict=True)
    evaluator.freeze()
    _require(not any(parameter.requires_grad for parameter in evaluator.parameters()), "confirmation evaluator is not frozen")
    return evaluator


def _window_check(
    arm: str,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module | None = None,
) -> tuple[int, dict[str, Any]]:
    sample = _image_batch(pixels, rows[0], arm, device)[:1]
    if arm == "small":
        small = config["optics"]["small"]
        return _small_window_check(
            sample,
            small["propagation_grid_candidates"],
            device,
            None if model is None else _state_snapshot(model),
            ols_model=small.get("ols_basis", LP01_OLS_MODEL),
            production_grid=small.get("production_propagation_grid"),
        )
    import scripts.research.lightgen.execute_chen_lightgen_400_mnist_baseline_v1 as baseline

    reference = model if model is not None else ChenLightGen400Baseline().to(device)
    minimum_converged_grid, diagnostics = baseline._window_reference(
        reference,
        sample,
        config["optics"]["chen400"]["propagation_grid_candidates"],
        config["optics"]["chen400"]["common_complex_roi_m"],
        reference_sampling_factor=2,
        include_sampling_reference=True,
    )
    error_by_grid = {
        int(grid): max(values["complex_relative_error_vs_largest_window"].values())
        for grid, values in diagnostics.items()
        if isinstance(grid, int)
    }
    configured_grid = config["optics"]["chen400"].get("production_propagation_grid")
    selected = minimum_converged_grid if configured_grid is None else configured_grid
    _require(
        isinstance(selected, int) and not isinstance(selected, bool) and selected in error_by_grid,
        "configured production propagation grid is absent from the window diagnostics",
    )
    selected_error = error_by_grid[selected]
    _require(
        selected_error <= WINDOW_TOLERANCE,
        "configured 400 production propagation grid failed the complex-window tolerance",
    )
    return selected, {
        "status": "PASSED",
        "selected_grid": selected,
        "minimum_converged_grid": minimum_converged_grid,
        "complex_relative_error_max": selected_error,
        "complex_relative_error_by_grid": error_by_grid,
        "diagnostics": diagnostics,
    }


def _json_window_diagnostics(value: Any) -> Any:
    """Normalize window diagnostic mapping keys only at JSON serialization boundaries."""
    if isinstance(value, dict):
        return {str(key): _json_window_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_window_diagnostics(item) for item in value]
    return value


def _run(args: argparse.Namespace) -> None:
    if args.recovery_worker_dir is not None:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        _run_recovery_worker(args.recovery_worker_dir, config_path, args.arm, args.seed)
        return
    _require(args.run_dir is not None, "explicit run directory is required")
    _require(not (args.preflight_only and args.resume_checkpoint is not None),
             "preflight-only cannot be combined with an explicit resume checkpoint")
    config_path = (args.config if args.config.is_absolute() else ROOT / args.config).resolve()
    config = _read_config(config_path)
    run_dir = (args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir).resolve()
    _require(args.arm in {"small", "chen400"}, "unsupported arm")
    allowed_seeds = config["training"]["paired_schedule_seeds"]
    _require(args.seed in allowed_seeds, "seed is not one of the registered schedule seeds")
    resuming = args.resume_checkpoint is not None
    if resuming:
        _require(run_dir.is_dir(), f"resume run directory is missing: {run_dir}")
        _require(not (run_dir / "report.json").exists(), "completed run reports cannot be resumed")
        resume_path = (args.resume_checkpoint if args.resume_checkpoint.is_absolute()
                       else ROOT / args.resume_checkpoint).resolve()
        _require(resume_path.is_file(), f"explicit resume checkpoint is missing: {resume_path}")
        try:
            resume_path.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError("resume checkpoint must be inside the explicitly selected run directory") from error
    else:
        _require(not run_dir.exists(), f"refusing to overwrite existing run directory: {run_dir}")
    _require(torch.cuda.is_available(), "CUDA is required for this optical training comparison")
    device = torch.device("cuda")
    if args.arm == "chen400":
        _install_chen400_memory_optimization()
    process_started = time.perf_counter()
    pixels, labels, train_indices, dev_indices = _load_data(ROOT / config["data"]["training_file"])
    schedule = _schedule(labels, train_indices, args.seed)
    evaluator_path = ROOT / config["evaluation"]["evaluator_checkpoint"]
    evaluator = _load_evaluator(evaluator_path, device)
    evaluator_state = _state_snapshot(evaluator)
    task_id = config.get("task_id", TASK_ID)
    auth_id = f"{task_id}:{args.arm}:{args.seed}:preflight_and_execution"
    model_identity = _model_identity(config, args.arm)
    recovery_staging: Path | None = None
    preflight_only_dir: Path | None = None
    preflight: dict[str, Any]
    scale: torch.Tensor
    power_reference: torch.Tensor
    camera_gain: torch.Tensor
    weights: dict[str, float]
    history: list[dict[str, Any]]
    dev_metrics: list[dict[str, Any]]
    elapsed_base_seconds = 0.0

    if resuming:
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _require(saved_config == config, "resume config differs from the original run config")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        _require(checkpoint.get("schema") == "lightgen_matched_architecture_checkpoint.v1",
                 "resume checkpoint schema changed")
        run_id = checkpoint.get("run_id")
        _require(isinstance(run_id, str) and bool(run_id), "resume checkpoint run identity is missing")
        _require(checkpoint.get("task_id") == task_id and checkpoint.get("arm") == args.arm
                 and checkpoint.get("seed") == args.seed,
                 "resume checkpoint task, arm, or seed identity changed")
        _require(checkpoint.get("model_identity") == model_identity
                 and checkpoint.get("config_identity") == config,
                 "resume checkpoint model or configuration identity changed")
        start_step = checkpoint.get("completed_updates")
        _require(isinstance(start_step, int) and 0 <= start_step <= UPDATES,
                 "resume checkpoint update count is invalid")
        saved_identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
        identity = _make_identity(config, args.arm, args.seed, run_id)
        expected_identity = json.loads(json.dumps(identity.__dict__))
        _require(saved_identity.get("identity") == expected_identity
                 and saved_identity.get("authorization_id") == auth_id,
                 "resume run identity or authorization changed")
        receipt = PreflightReceipt.from_mapping(json.loads(
            (run_dir / "preflight_ready.json").read_text(encoding="utf-8")
        ))
        control = ResearchRunControl(identity, authorization_id=auth_id)
        control.mark_preflight_ready(receipt)
        preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
        selected_grid = preflight.get("selected_propagation_grid")
        _require(isinstance(selected_grid, int), "resume preflight does not contain a fixed production grid")
        _require(checkpoint.get("propagation_grid_size") == selected_grid,
                 "resume checkpoint propagation grid differs from the original preflight")
        configured_grid = config["optics"][args.arm].get("production_propagation_grid")
        if configured_grid is not None:
            _require(selected_grid == configured_grid, "resume selected grid differs from the fixed production grid")
        clear_production_transfer_cache()
        model = _model(args.arm, selected_grid, device, config)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = torch.optim.Adam(
            _phase_groups(model), lr=1e-3, betas=TRAIN_BETAS,
            eps=TRAIN_EPSILON, weight_decay=0.0,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        _require(torch.equal(checkpoint["schedule"], schedule), "resume sample rows differ from the fixed TRAIN schedule")
        references = torch.load(run_dir / "initial_references.pt", map_location="cpu", weights_only=False)
        _assert_close(checkpoint["scale"], references["scale"], "resume_scale_reference")
        _assert_close(checkpoint["power_reference"], references["power_reference"], "resume_power_reference")
        _assert_close(checkpoint["camera_gain"], references["camera_gain"], "resume_camera_gain")
        _assert_close(checkpoint["weights"], references["weights"], "resume_frozen_weights")
        _require(torch.equal(references["paired_schedule"], schedule), "saved reference schedule changed")
        scale = checkpoint["scale"].to(device=device)
        power_reference = checkpoint["power_reference"].to(device=device)
        camera_gain = checkpoint["camera_gain"].to(device=device)
        weights = checkpoint["weights"]
        history = checkpoint.get("history")
        dev_metrics = checkpoint.get("dev_metrics")
        _require(isinstance(history, list) and isinstance(dev_metrics, list),
                 "resume checkpoint history or DEV metrics are incomplete")
        if start_step:
            _require(_optimizer_update_count(optimizer, model) == start_step,
                     "resume Adam update count differs from the checkpoint step")
        else:
            _require(not optimizer.state, "step-zero resume checkpoint has nonempty Adam state")
        _require(checkpoint.get("evaluator_role") == "frozen_confirmation_only",
                 "resume evaluator role changed")
        elapsed_base_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        _require(math.isfinite(elapsed_base_seconds) and elapsed_base_seconds >= 0.0,
                 "resume checkpoint elapsed time is invalid")
        torch.cuda.set_rng_state_all([value.to(device="cpu") for value in checkpoint["cuda_rng"]])
        torch.random.set_rng_state(checkpoint["cpu_rng"])
        control.resume_scientific_attempt(run_dir, attempt_id=f"{args.arm}_seed_{args.seed}", receipt=receipt)
    else:
        run_id = uuid.uuid4().hex
        identity = _make_identity(config, args.arm, args.seed, run_id)
        control = ResearchRunControl(identity, authorization_id=auth_id)
        preflight_started = time.perf_counter()
        selected_grid, window_diagnostics = _window_check(
            args.arm, pixels, labels, schedule, config, device
        )
        clear_production_transfer_cache()
        model = _model(args.arm, selected_grid, device, config)
        phase_count = sum(parameter.numel() for parameter in _phase_groups(model))
        expected_phase_count = 40_000 if args.arm == "small" else 960_000
        _require(phase_count == expected_phase_count, "trainable phase parameter count changed")
        optimizer = torch.optim.Adam(
            _phase_groups(model), lr=1e-3, betas=TRAIN_BETAS,
            eps=TRAIN_EPSILON, weight_decay=0.0,
        )
        first_target = _image_batch(pixels, schedule[0], args.arm, device)
        initial_readout = (
            _small_producer_consumer_check(model, first_target[:1])
            if isinstance(model, SmallMatchedAdapter) else {"status": "NOT_APPLICABLE"}
        )
        calibration_started = time.perf_counter()
        scale, power_reference, camera_gain, gain_fit = _calibrate(
            model, pixels, train_indices, args.arm, device
        )
        calibration_seconds = time.perf_counter() - calibration_started
        with torch.no_grad():
            _, initial_result = _loss_components(model, first_target, scale, power_reference, camera_gain)
        _require(initial_result["plane_powers_w"].shape == (BATCH_SIZE, 5 if args.arm == "small" else 7),
                 "model power-plane count differs from its declared physical layers")
        del initial_result
        if isinstance(model, ChenLightGen400Baseline):
            norms = _chen400_streamed_gradient_norms(
                model, first_target, scale, power_reference, camera_gain
            )
        else:
            norms = _gradient_norms(
                model,
                lambda: _loss_components(model, first_target, scale, power_reference, camera_gain)[0],
            )
        weights = _weights(norms)
        finite_difference = _finite_difference_check(
            model, first_target, scale, power_reference, camera_gain, weights
        )
        negative_control = _assert_row_negative_control(schedule[0])
        memory_execution = {
            "logical_batch_size": BATCH_SIZE,
            "forward_microbatch_size": CHEN400_FORWARD_MICROBATCH_SIZE if args.arm == "chen400" else BATCH_SIZE,
            "per_example_gradient_checkpointing": args.arm == "chen400",
            "chen400_inner_propagation_checkpointing": False if args.arm == "chen400" else None,
            "gradient_norms_recomputed_per_component": True,
            "gradient_norms_streamed_per_example": args.arm == "chen400",
            "bogt_loss_aggregated_over_full_logical_batch": True,
        }
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        recovery_staging = run_dir.parent / f"preflight_{args.arm}_seed_{args.seed}_{uuid.uuid4().hex[:8]}"
        recovery_staging.mkdir(parents=False, exist_ok=False)
        recovery_seconds_started = time.perf_counter()
        preflight_recovery = _preflight_recovery(
            model, optimizer, pixels, schedule, args.arm, args.seed, device,
            scale, power_reference, camera_gain, weights, run_id, recovery_staging,
            config, config_path, model_identity,
        )
        preflight = {
            "status": "READY",
            "arm": args.arm,
            "seed": args.seed,
            "selected_propagation_grid": selected_grid,
            "window": _json_window_diagnostics(window_diagnostics),
            "production_ols_coefficient_relative_error": window_diagnostics[
                "ols_coefficient_relative_error_by_grid"][selected_grid],
            "production_camera_w_per_pixel_relative_error": window_diagnostics[
                "camera_w_per_pixel_relative_error_by_grid"][selected_grid],
            "phase_parameters": phase_count,
            "train_role": "55000 MNIST TRAIN rows; no official TEST",
            "dev_role": "5000 MNIST DEV rows; 500 per class",
            "initial_camera_gain_w_inverse": float(camera_gain),
            "camera_gain_fit_sums": gain_fit,
            "initial_modal_scale_w": scale.tolist(),
            "initial_plane_power_reference_w": power_reference.tolist(),
            "initial_component_gradient_norms": norms,
            "frozen_loss_weights": weights,
            "central_difference": finite_difference,
            "mismatched_row_negative_control": negative_control,
            "three_update_recovery_and_rollback": preflight_recovery,
            "initial_independent_readout": initial_readout,
            "memory_execution": memory_execution,
            "calibration_seconds": calibration_seconds,
            "preflight_validation_seconds": time.perf_counter() - preflight_started,
            "recovery_roundtrip_seconds": time.perf_counter() - recovery_seconds_started,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "official_test_reads": 0,
            "evaluator_frozen_and_excluded_from_backward": True,
            "engineering_adam_calls": 5,
            "scientific_updates_before_execution": 0,
        }
        receipt = control.make_ready_receipt(
            correctness={
                "passed": True,
                "checks": [
                    "matched_mnist_roles",
                    "fixed_balanced_schedule",
                    "physical_coordinates_and_si_power",
                    "fixed_production_grid_vs_larger_reference_1e-4",
                    "independent_numpy_ols_and_full_camera_readout",
                    "unclipped_bogt_loss_and_independent_finite_difference",
                    "both_phase_groups_update",
                    "evaluator_frozen_and_excluded",
                    "mismatched_mnist_rows_rejected",
                    "independent_process_checkpoint_reload_and_rollback",
                ],
                "max_window_error": window_diagnostics["complex_relative_error_max"],
                "max_ols_coefficient_relative_error": window_diagnostics[
                    "ols_coefficient_relative_error_by_grid"][selected_grid],
                "max_camera_w_per_pixel_relative_error": window_diagnostics[
                    "camera_w_per_pixel_relative_error_by_grid"][selected_grid],
                "independent_process_recovery_passed": True,
            },
            resource={
                "passed": True,
                "device": torch.cuda.get_device_name(device),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            },
            implementation={
                "passed": True,
                "engine": f"cuda_{args.arm}_grid_{selected_grid}",
                "selected_engine": f"cuda_{args.arm}_grid_{selected_grid}",
            },
            selected_engine=f"cuda_{args.arm}_grid_{selected_grid}",
            resource_envelope={
                "device": torch.cuda.get_device_name(device),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            },
        )
        control.mark_preflight_ready(receipt)
        if args.preflight_only:
            preflight_only_dir = run_dir.parent / f"{run_dir.name}_preflight_{run_id[:8]}"
            preflight_only_dir.mkdir(parents=False, exist_ok=False)
            atomic_write_new_json(preflight_only_dir / "config.json", config)
            atomic_write_new_json(preflight_only_dir / "preflight.json", preflight)
            atomic_write_new_json(preflight_only_dir / "preflight_ready.json", receipt.to_dict())
            atomic_write_new_json(preflight_only_dir / "run_identity.json", {
                "identity": identity.__dict__, "authorization_id": auth_id,
            })
            atomic_torch_save_new(preflight_only_dir / "initial_references.pt", {
                "scale": scale.cpu(), "power_reference": power_reference.cpu(),
                "camera_gain": camera_gain.cpu(), "weights": weights, "paired_schedule": schedule,
            })
            _require(recovery_staging.parent.resolve() == run_dir.parent.resolve()
                     and recovery_staging.name.startswith(f"preflight_{args.arm}_seed_{args.seed}_"),
                     "refusing to clean an unexpected recovery staging path")
            shutil.rmtree(recovery_staging)
            print(json.dumps({"status": "PREFLIGHT_READY", "preflight": str(preflight_only_dir / "preflight.json")}), flush=True)
            return
        control.open_scientific_attempt(run_dir, attempt_id=f"{args.arm}_seed_{args.seed}", receipt=receipt)
        atomic_write_new_json(run_dir / "config.json", config)
        atomic_write_new_json(run_dir / "preflight.json", preflight)
        atomic_write_new_json(run_dir / "preflight_ready.json", receipt.to_dict())
        atomic_write_new_json(run_dir / "run_identity.json", {
            "identity": identity.__dict__,
            "authorization_id": auth_id,
            "receipt": receipt.to_dict(),
        })
        atomic_torch_save_new(run_dir / "initial_references.pt", {
            "scale": scale.cpu(),
            "power_reference": power_reference.cpu(),
            "camera_gain": camera_gain.cpu(),
            "weights": weights,
            "paired_schedule": schedule,
        })
        history = []
        dev_metrics = []
        start_step = 0
        elapsed_base_seconds = 0.0
        _require(recovery_staging.parent.resolve() == run_dir.parent.resolve()
                 and recovery_staging.name.startswith(f"preflight_{args.arm}_seed_{args.seed}_"),
                 "refusing to clean an unexpected recovery staging path")
        shutil.rmtree(recovery_staging)

    phase_count = sum(parameter.numel() for parameter in _phase_groups(model))
    _require(phase_count == (40_000 if args.arm == "small" else 960_000),
             "trainable phase parameter count changed")
    phase_parameter_ids = {id(parameter) for parameter in _phase_groups(model)}
    _require(all(parameter.requires_grad == (id(parameter) in phase_parameter_ids)
                 for parameter in model.parameters()),
             "a frozen optical component became trainable or a phase group became frozen")
    _require(_assert_close(evaluator_state, _state_snapshot(evaluator), "frozen_evaluator_before_training") == 0.0,
             "frozen evaluator changed before training")
    if not resuming:
        metric0, examples0 = _evaluate_and_save(
            model, evaluator, pixels, labels, dev_indices, args.arm, device, camera_gain, 0
        )
        dev_metrics.append(metric0)
        atomic_torch_save(run_dir / "dev_examples_step_000000.pt", examples0)
        if config["output"].get("save_evaluation_checkpoints", False):
            checkpoint0 = _checkpoint_payload(
                model, optimizer, 0, args.arm, args.seed, run_id, schedule, scale,
                power_reference, camera_gain, weights, history, dev_metrics,
                task_id, model_identity, config, time.perf_counter() - process_started,
            )
            _save_checkpoint(run_dir, checkpoint0)
            atomic_torch_save(run_dir / "checkpoint_step_000000.pt", checkpoint0)
    else:
        _require(control.state == "SCIENTIFIC_ACTIVE", "explicit resume failed to reopen the active research attempt")

    gaussian_density_enabled = (isinstance(model, SmallMatchedAdapter)
                                and model.runtime.ols.basis_id == GAUSSIAN_OLS_MODEL)
    if gaussian_density_enabled:
        _record_gaussian_plane_density_probe(
            model, pixels, labels, dev_indices, run_dir, branch="main", global_step=start_step,
            grid_size=selected_grid,
        )

    existing_eval_steps = {int(metric["step"]) for metric in dev_metrics}
    for evaluation_step in EVALUATION_STEPS:
        if evaluation_step <= start_step and evaluation_step not in existing_eval_steps:
            metric, examples = _evaluate_and_save(
                model, evaluator, pixels, labels, dev_indices, args.arm, device, camera_gain, evaluation_step
            )
            dev_metrics.append(metric)
            atomic_torch_save(run_dir / f"dev_examples_step_{evaluation_step:06d}.pt", examples)
            existing_eval_steps.add(evaluation_step)
    dev_metrics.sort(key=lambda item: int(item["step"]))
    atomic_write_json(run_dir / "dev_metrics_live.json", dev_metrics)

    stopped_by_signal = False
    pending_signal: list[int] = []
    old_handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    completed_step = start_step

    def handler(number: int, frame: Any) -> None:
        del frame
        pending_signal.append(number)

    for number in old_handlers:
        signal.signal(number, handler)
    training_started = time.perf_counter()
    rate_start = training_started
    try:
        for cursor in range(start_step, UPDATES):
            step = cursor + 1
            losses = _train_update(
                model, optimizer, pixels, schedule[cursor], args.arm, device,
                scale, power_reference, camera_gain, weights,
            )
            completed_step = step
            if step % PROGRESS_INTERVAL == 0:
                history.append({"step": step, **losses})
                atomic_write_json(run_dir / "training_history_live.json", history)
                atomic_write_json(run_dir / "progress_summary.json", {
                    "completed_updates": step,
                    "total_updates": UPDATES,
                    "latest_loss": losses,
                    "seconds_per_step_since_previous_progress": (time.perf_counter() - rate_start) / PROGRESS_INTERVAL,
                    "elapsed_seconds_this_process": time.perf_counter() - process_started,
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
                })
                rate_start = time.perf_counter()
            if step in EVALUATION_STEPS:
                metric, examples = _evaluate_and_save(
                    model, evaluator, pixels, labels, dev_indices, args.arm, device, camera_gain, step
                )
                dev_metrics = [item for item in dev_metrics if int(item["step"]) != step]
                dev_metrics.append(metric)
                dev_metrics.sort(key=lambda item: int(item["step"]))
                atomic_torch_save(run_dir / f"dev_examples_step_{step:06d}.pt", examples)
                atomic_write_json(run_dir / "dev_metrics_live.json", dev_metrics)
            if step % CHECKPOINT_INTERVAL == 0 or pending_signal:
                checkpoint_payload = _checkpoint_payload(
                    model, optimizer, step, args.arm, args.seed, run_id, schedule,
                    scale, power_reference, camera_gain, weights, history, dev_metrics,
                    task_id, model_identity, config,
                    elapsed_base_seconds + time.perf_counter() - process_started,
                )
                _save_checkpoint(run_dir, checkpoint_payload)
                if step in EVALUATION_STEPS and config["output"].get("save_evaluation_checkpoints", False):
                    atomic_torch_save(run_dir / f"checkpoint_step_{step:06d}.pt", checkpoint_payload)
                if gaussian_density_enabled:
                    _record_gaussian_plane_density_probe(
                        model, pixels, labels, dev_indices, run_dir, branch="main", global_step=step,
                        grid_size=selected_grid,
                    )
            if pending_signal:
                control.interrupted()
                atomic_write_json(run_dir / "interrupted.json", {
                    "status": control.state,
                    "signal": pending_signal[-1],
                    "completed_updates": step,
                    "checkpoint": "checkpoint_latest.pt",
                })
                atomic_write_json(run_dir / "run_state.json", {
                    "status": control.state,
                    "completed_updates": step,
                    "scientific_updates": step,
                })
                stopped_by_signal = True
                break
        training_seconds_this_process = time.perf_counter() - training_started
        if stopped_by_signal:
            return
        _require(completed_step == UPDATES, "fixed scientific update budget was not completed")
        _require(_optimizer_update_count(optimizer, model) == UPDATES,
                 "loaded Adam state does not contain exactly the fixed scientific updates")
        plane_density_logging = None
        if gaussian_density_enabled:
            expected_density_steps = sorted({start_step} | {
                step for step in range(start_step + 1, UPDATES + 1) if step % CHECKPOINT_INTERVAL == 0
            })
            plane_density_logging = _validate_gaussian_plane_density_coverage(
                run_dir, "main", expected_density_steps, selected_grid,
            )
        _require(sorted(int(item["step"]) for item in dev_metrics) == list(EVALUATION_STEPS),
                 "one or more registered DEV checkpoints are missing or duplicated")
        _require(_assert_close(evaluator_state, _state_snapshot(evaluator), "frozen_evaluator_after_training") == 0.0,
                 "independent confirmation evaluator changed during training")
        if isinstance(model, SmallMatchedAdapter):
            final_readout = _small_producer_consumer_check(
                model, _image_batch(pixels, schedule[0], args.arm, device)[:1]
            )
        else:
            final_readout = {"status": "NOT_APPLICABLE"}
        endpoint_grid, endpoint_window = _window_check(
            args.arm, pixels, labels, schedule, config, device, model=model
        )
        endpoint_error = endpoint_window["complex_relative_error_max"]
        _require(endpoint_grid == selected_grid and endpoint_error <= WINDOW_TOLERANCE,
                 "trained endpoint failed the fixed production-grid complex-window tolerance")
        _require(
            endpoint_window["ols_coefficient_relative_error_by_grid"][selected_grid] <= WINDOW_TOLERANCE
            and endpoint_window["camera_w_per_pixel_relative_error_by_grid"][selected_grid] <= WINDOW_TOLERANCE,
            "trained endpoint OLS coefficient or full-camera readout comparison failed",
        )
        terminal_model = _state_snapshot(model)
        atomic_torch_save(run_dir / "terminal_continuous_phase.pt", {
            "schema": "lightgen_matched_architecture_terminal_model.v1",
            "task_id": task_id,
            "run_id": run_id,
            "arm": args.arm,
            "seed": args.seed,
            "model_identity": model_identity,
            "completed_updates": UPDATES,
            "model": terminal_model,
        })
        primary_endpoint = next(metric for metric in dev_metrics if int(metric["step"]) == UPDATES)
        reconstruction_comparisons: dict[str, Any] = {}
        comparison_references: dict[str, torch.Tensor] = {}
        if isinstance(model, SmallMatchedAdapter) and model.runtime.ols.basis_id == GAUSSIAN_OLS_MODEL:
            _require(len(primary_endpoint.get("per_image", [])) == DEV_COUNT,
                     "terminal Gaussian reconstruction is missing per-image DEV metrics")
            reconstruction_comparisons, comparison_references = _evaluate_reconstruction_comparisons(
                model, evaluator, pixels, labels, train_indices, dev_indices, device,
                camera_gain, primary_endpoint["per_image"],
            )
            atomic_torch_save(run_dir / "global_mean_baseline.pt", {
                "task_id": task_id,
                "run_id": run_id,
                **comparison_references,
            })
            atomic_write_json(run_dir / "reconstruction_comparisons.json", reconstruction_comparisons)
            _write_reconstruction_csv(
                run_dir / "reconstruction_metrics_step_030000.csv",
                primary_endpoint["per_image"],
                reconstruction_comparisons["global_mean_image_metrics"]["per_image"],
                reconstruction_comparisons["same_class_mismatched_input_metrics"]["per_image"],
            )
            dev_display = torch.load(run_dir / "dev_examples_step_030000.pt", map_location="cpu", weights_only=False)
            _write_contact_sheet(run_dir / "dev_preview.png", dev_display["images"], dev_display["labels"], 2)
            _write_contact_sheet(run_dir / "dev_target_preview.png", dev_display["targets"], dev_display["labels"], 2)

        fitted = _fit_gaussians(model, pixels, labels, train_indices, args.arm, scale, device)
        atomic_torch_save(run_dir / "fitted_gaussian_prior.pt", {
            "task_id": task_id,
            "run_id": run_id,
            "ols_basis": model.runtime.ols.basis_id if isinstance(model, SmallMatchedAdapter) else "global_gaussian_s12_v1",
            "scale": scale.cpu(),
            "power_ridge_relative": POWER_RIDGE,
            "per_class": {
                str(label): {key: value for key, value in fitted[label].items() if key in {"mean", "cholesky", "ridge"}}
                for label in range(10)
            },
        })
        max_proposals = config["evaluation"].get("max_proposals_per_class")
        try:
            samples, requested, proposal_ids, rejection = _sample_gaussians(
                fitted, scale, max_proposals_per_class=max_proposals
            )
        except SamplingPopulationIncomplete as error:
            generated = {
                "status": "INCOMPLETE",
                "reason": str(error),
                "registered_population": SAMPLE_COUNT * 10,
                "max_proposals_per_class": max_proposals,
            }
            generation_examples = None
            measurement_status = "MEASUREMENT_INCOMPLETE"
            quality_gate = None
        else:
            generated, generation_examples = _evaluate_generation(
                model, evaluator, samples, requested, proposal_ids, labels, pixels,
                train_indices, dev_indices, args.arm, device, camera_gain,
            )
            generated["proposal_rejection"] = rejection
            generated["status"] = "COMPLETE"
            generation_examples.update({"task_id": task_id, "run_id": run_id})
            atomic_torch_save(run_dir / "generated_cohort.pt", generation_examples)
            if config.get("quality_gate"):
                quality_gate = _quality_decision(config, primary_endpoint, generated)
                measurement_status = quality_gate["status"]
            else:
                quality_gate = None
                measurement_status = "VALID_COMPLETED"
        atomic_write_json(run_dir / "training_history.json", history)
        atomic_write_json(run_dir / "dev_metrics.json", dev_metrics)
        atomic_write_json(run_dir / "generation_metrics.json", generated)
        if generation_examples is not None:
            _write_contact_sheet(
                run_dir / "generation_preview.png",
                generation_examples["images"],
                generation_examples["requested_labels"],
                10,
            )
        _require(_assert_close(evaluator_state, _state_snapshot(evaluator), "frozen_evaluator_after_all_measurements") == 0.0,
                 "independent confirmation evaluator changed during final measurements")
        report = {
            "schema": "lightgen_matched_architecture_run_report.v1",
            "task_id": task_id,
            "status": measurement_status,
            "measurement_complete": generation_examples is not None,
            "evidence_tier": "diagnostic_only",
            "arm": args.arm,
            "seed": args.seed,
            "run_id": run_id,
            "architecture": model_identity,
            "trainable_phase_parameters": phase_count,
            "optics": {
                "wavelength_m": 5.32e-7,
                "input_power_w": SOURCE_POWER_W,
                "layer_power_transmission": LAYER_POWER_TRANSMISSION,
                "selected_propagation_grid": selected_grid,
                "window_reference_grid": config["optics"][args.arm].get("propagation_grid_candidates", [])[-1],
                "small_ols_mode_geometry": (model.runtime.ols.mode_identity()
                                             if isinstance(model, SmallMatchedAdapter) else None),
                "project_geometry_measured": False,
            },
            "data_roles": {
                "train_count": TRAIN_COUNT,
                "dev_count": DEV_COUNT,
                "dev_per_class": 500,
                "official_test_reads": 0,
                "paired_schedule_seed": args.seed,
                "evaluator_role": "frozen E_D confirmation instrument; DEV is development data, not blind",
            },
            "training": {
                "updates": UPDATES,
                "logical_batch_size": BATCH_SIZE,
                "samples_per_class_per_batch": 1,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "betas": TRAIN_BETAS,
                "epsilon": TRAIN_EPSILON,
                "weight_decay": 0.0,
                "initialization": "zero_phase",
                "trained_groups": ["encoder_phase", "decoder_phase"],
                "frozen_camera_gain_w_inverse": float(camera_gain),
                "initial_gradient_norms": preflight.get("initial_component_gradient_norms"),
                "frozen_loss_weights": weights,
                "loss_definition": "native unclipped physical W/pixel MSE + truncated Gaussian modal KL + mean squared log ratio of each configuration's recorded physical plane powers",
                "history_every_100_updates": history,
                "engineering_adam_calls": 5,
                "scientific_updates": UPDATES,
            },
            "reconstruction": dev_metrics,
            "reconstruction_comparisons": reconstruction_comparisons,
            "generation": generated,
            "quality_gate": quality_gate,
            "preflight": preflight,
            "initial_independent_readout": preflight.get("initial_independent_readout"),
            "endpoint_independent_readout": final_readout,
            "endpoint_window": {
                "selected_production_grid": selected_grid,
                "selected_again_at_endpoint": endpoint_grid,
                "selected_grid_complex_relative_error_max": endpoint_error,
                "selected_grid_passed": True,
                "diagnostics": _json_window_diagnostics(endpoint_window),
            },
            "elapsed_seconds": elapsed_base_seconds + time.perf_counter() - process_started,
            "elapsed_seconds_this_process": time.perf_counter() - process_started,
            "training_seconds_this_process": training_seconds_this_process,
            "plane_power_density_logging": plane_density_logging or {"status": "NOT_APPLICABLE"},
            "seconds_per_scientific_update_this_process": (
                training_seconds_this_process / (completed_step - start_step)
                if completed_step > start_step else None
            ),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "device": torch.cuda.get_device_name(device),
            "claim_boundary": {
                "whole_configuration_comparison_only": True,
                "resolution_only_causality": False,
                "parameter_count_only_causality": False,
                "human_readability": False,
                "novelty_or_coverage": False,
                "generalization": False,
                "hardware_validation": False,
                "model_adoption": False,
                "not_blind": True,
                "proxy_clipping_is_not_physical_full_well_saturation": True,
            },
            "artifacts": {
                "terminal_model": "terminal_continuous_phase.pt",
                "endpoint_checkpoints": [
                    f"checkpoint_step_{step:06d}.pt" for step in EVALUATION_STEPS
                ] if config["output"].get("save_evaluation_checkpoints", False) else [],
                "rolling_recovery_checkpoints": ["checkpoint_latest.pt", "checkpoint_previous.pt"],
                "frozen_initial_references": "initial_references.pt",
                "fitted_gaussian_prior": "fitted_gaussian_prior.pt",
                "generation_examples": "generated_cohort.pt" if generation_examples is not None else None,
                "training_history": "training_history.json",
                "dev_metrics": "dev_metrics.json",
                "reconstruction_rows": "reconstruction_metrics_step_030000.csv" if reconstruction_comparisons else None,
                "report_html": "results.html",
                "dev_target_preview": "dev_target_preview.png" if reconstruction_comparisons else None,
                "dev_reconstruction_preview": "dev_preview.png" if reconstruction_comparisons else None,
                "generation_preview": "generation_preview.png" if generation_examples is not None else None,
            },
        }
        if generation_examples is not None and args.arm == "small":
            _write_results_html(run_dir / "results.html", report, generated)
        else:
            _write_results_html(run_dir / "results.html", report, None)
        atomic_write_json(run_dir / "report.json", report)
        output_budget = int(config["output"].get("formal_result_budget_bytes", 512 * 1024 * 1024))
        output_bytes = _directory_bytes(run_dir)
        report["delivery"] = {
            "formal_result_budget_bytes": output_budget,
            "output_bytes": output_bytes,
            "within_budget": output_bytes <= output_budget,
        }
        atomic_write_json(run_dir / "report.json", report)
        if measurement_status == "MEASUREMENT_INCOMPLETE" or output_bytes > output_budget:
            control.invalid_harness()
        elif measurement_status == "VALID_FAILED":
            control.valid_failed()
        else:
            control.complete()
        atomic_write_json(run_dir / "run_state.json", {
            "status": control.state,
            "scientific_status": measurement_status,
            "delivery_within_budget": output_bytes <= output_budget,
            "completed_updates": completed_step,
            "measurement_complete": generation_examples is not None,
        })
        print(json.dumps({"status": measurement_status, "run_control_state": control.state,
                          "arm": args.arm, "seed": args.seed, "report": str(run_dir / "report.json")}), flush=True)
    except BaseException as error:
        if control.state in {"SCIENTIFIC_ACTIVE", "RUNNING"}:
            control.invalid_harness()
        atomic_write_json(run_dir / "engineering_error.json", {
            "status": control.state,
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_updates": completed_step,
            "checkpoint": "checkpoint_latest.pt" if (run_dir / "checkpoint_latest.pt").is_file() else None,
        })
        atomic_write_json(run_dir / "run_state.json", {
            "status": control.state,
            "scientific_status": "ENGINEERING_INVALID",
            "completed_updates": completed_step,
        })
        raise
    finally:
        for number, old in old_handlers.items():
            signal.signal(number, old)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--arm", required=True, choices=("small", "chen400"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--recovery-worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
