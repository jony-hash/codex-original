"""Compare recent LightGen optical endpoints and conditionally continue optics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.evidence import atomic_torch_save, atomic_torch_save_new
from ldm_project.v2.evidence.io import cpu_copy
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_semantic_v1 as frame
from scripts.research.lightgen import execute_lightgen_g1_semantic_exposure200_v1 as exposure
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base
from scripts.research.lightgen import execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder
from scripts.research.lightgen import execute_lightgen_small_joint_update_readability_pilot_v1 as joint


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_RECENT_OPTICAL_FULLPATH_SEMANTIC_V1"
AUTHORITY = "USER_APPROVED_LIGHTGEN_RECENT_OPTICAL_FULLPATH_SEMANTIC_20261030"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/recent_optical_fullpath_semantic_v1.yaml"
OUTPUT_DIR = ROOT / "outputs/20261030_lightgen_recent_optical_fullpath_semantic_v1/runs/attempt_01"
ATTEMPT_DIR = OUTPUT_DIR / "scientific_attempt"
ENDPOINT_ORDER = ("C45k", "J", "D0", "Dgrad")
TRAIN_UPDATES = 2_000
BATCH_SIZE = 10
CLASS_COUNT = 10
PORT_COUNT = 100
COMMAND_SHAPE = (1, 200, 200)
GRID_PRODUCTION = 2048
GRID_REFERENCE = 4096
RECOVERY_RTOL = 2.0e-6
RECOVERY_ATOL = 5.0e-8
COMMAND_REPLAY_ATOL = 5.0e-6
GATE_LIMIT = 1.0e-4
CAMERA_AREA_EXPECTED = 6.25e-12
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _write_task_contract(output_dir: Path) -> None:
    _write_json(output_dir / "implementation_contract.json", {
        "task_id": TASK_ID,
        "risk": "high",
        "behavioral_consequence": (
            "The conditional 2,000-step path changes only the selected Encoder and Generator phases."
        ),
        "decision_consumer": "final_reconsumption.json -> final_decision.json",
        "scope": {
            "repository_changes": [
                "configs/research/lightgen/recent_optical_fullpath_semantic_v1.yaml",
                "scripts/research/lightgen/execute_lightgen_recent_optical_fullpath_semantic_v1.py",
                "tests/research/lightgen/test_lightgen_recent_optical_fullpath_semantic_v1.py",
            ],
            "run_output": OUTPUT_DIR.relative_to(ROOT).as_posix(),
            "existing_source_artifacts_modified": False,
        },
        "invariants": [
            "C45k, D0, Dgrad, and J retain their explicit checkpoint and Adam identities.",
            "Mapper, prior, fixed projection, camera gain, and E_D/E_C remain frozen.",
            "Training is either zero updates or exactly 2,000 joint optical updates.",
            "The 2,000-row final population is not sampled until endpoint routing and training finish.",
            "The final consumer reloads saved commands and checkpoints and replays the full optical path.",
            "No hardware deployment or human readability acceptance is inferred.",
        ],
        "verification": [
            "Fixed mapper saved-command replay.",
            "Three production-entrypoint optical updates, CMOS gradient checks, and wrong-target negative control.",
            "Independent-process fourth-update recovery and frozen-group rollback.",
            "Saved-command full-population reconsumption and the single predeclared final gate.",
        ],
        "stop": [
            "A valid failed final gate ends this candidate screen.",
            "An invalid harness is repaired and rechecked under this task before scientific inference.",
            "No automatic extra updates, resampling, threshold change, human review, or hardware action.",
        ],
    })
    recovery = """# Checkpoint recovery\n\nThe entrypoint saves complete model, Adam, schedule, history, and CPU/CUDA RNG state at updates 0, 500, 1,000, 1,500, and 2,000. It verifies every checkpoint by reading it back. `scientific_attempt/task_state.json` records the exact last complete checkpoint path.\n\nIf the AutoDL command exits before the final decision:\n\n1. Confirm with the AutoDL Status command that the previous `screen` process has exited. Do not start a second process while the first is active.\n2. Preserve the existing run directory and its saved population. Do not use the ordinary `execute` command again.\n3. Resume only from the explicit state file with:\n\n   ```bash\n   /root/miniconda3/bin/python -m scripts.research.lightgen.execute_lightgen_recent_optical_fullpath_semantic_v1 \\\n     --config configs/research/lightgen/recent_optical_fullpath_semantic_v1.yaml \\\n     --resume-state outputs/20261030_lightgen_recent_optical_fullpath_semantic_v1/runs/attempt_01/scientific_attempt/task_state.json\n   ```\n\nResume revalidates the selected endpoint, saved route, READY receipt, schedule, and exact checkpoint recorded in `task_state.json`. It does not discover a latest checkpoint or resample an already saved final population.\n"""
    (output_dir / "recovery_instructions.md").write_text(recovery, encoding="utf-8")


def _save_torch(path: Path, value: Any, *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new:
        atomic_torch_save_new(path, value)
    else:
        atomic_torch_save(path, value)


def _path(config: Mapping[str, Any], value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(),
             "this research route uses its registered configuration", ValueError)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict)
             and value.get("schema") == "lightgen_recent_optical_fullpath_semantic.v1"
             and value.get("task_id") == TASK_ID
             and value.get("semantic_authority_id") == AUTHORITY
             and value.get("evidence_tier") == "diagnostic_only",
             "task identity or diagnostic evidence tier changed", ValueError)
    population = value["population"]
    _require(population["classes"] == CLASS_COUNT
             and population["development_per_class"] == 20
             and population["final_per_class"] == 200
             and population["development_seed_base"] == 202610300910
             and population["final_seed_base"] == 202610300930
             and population["prior_alpha"] == 1.0
             and population["rejection"] == "original_total_modal_power_only"
             and population["classifier_based_selection"] is False,
             "fixed prior population or sampling rule changed", ValueError)
    training = value["training"]
    _require(training["maximum_new_updates"] == TRAIN_UPDATES
             and training["batch_size"] == BATCH_SIZE
             and training["samples_per_class_per_batch"] == 1
             and training["schedule_seed"] == 202610300950
             and training["optimizer"] == "Adam"
             and training["learning_rate"] == "inherit_from_selected_endpoint"
             and training["loss"] == "inherited_BOGT_plus_fixed_SSIM"
             and training["trainable_groups"] == ["encoder_phase", "generator_phase"]
             and training["frozen_groups"] == [
                 "mapper", "fixed_projection", "prior", "camera_gain", "evaluators"
             ]
             and training["checkpoint_interval"] == 500
             and training["checkpoints_at"] == list(CHECKPOINT_STEPS)
             and training["official_test_reads"] == 0
             and training["official_test_reads"] == 0,
             "short optical-update scope or data roles changed", ValueError)
    measurement = value["measurement"]
    _require(measurement["production_grid"] == GRID_PRODUCTION
             and measurement["independent_reference_grid"] == GRID_REFERENCE
             and measurement["existing_solver_positive_controls_per_class"] == 2
             and measurement["per_image_normalization"] is False
             and measurement["power_matching"] is False
             and measurement["gain_adjustment"] is False
             and measurement["fixed_camera_conversion"] == (
                 "raw_cmos_to_w_per_pixel_then_fixed_gain_then_clamp_then_adaptive_average_pool_28"
             )
             and measurement["endpoint_skip_average_macro_recall_gain"] == 0.02
             and measurement["endpoint_skip_each_evaluator_minimum_delta"] == 0.0
             and measurement["final_ed_paired_macro_recall_lower_bound_min"] == 0.0
             and measurement["final_ec_macro_recall_drop_max"] == 0.02
             and measurement["final_class_diversity_retention_min"] == 0.8
             and measurement["bootstrap_resamples"] == 2_000
             and measurement["bootstrap_seed"] == 202610300970,
             "fixed optical measurement or final decision changed", ValueError)
    _require(value["output"]["experiment_root"]
             == OUTPUT_DIR.relative_to(ROOT).as_posix()
             and value["output"]["preserve_source_artifacts"] is True,
             "output location or source-preservation rule changed", ValueError)
    _require(set(value["sources"]["endpoints"]) == set(ENDPOINT_ORDER),
             "the four registered optical endpoints are incomplete", ValueError)
    expected_sources = {
        "mapper_checkpoint": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01/checkpoints/terminal_checkpoint.pt",
        "physical_phase_prior": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01/prior_evidence/physical_phase_gaussian_prior.pt",
        "fixed_response_projector": "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/preflight_evidence/fixed_response_and_projector.pt",
        "mapper_replay_shard": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01/measurements/holdout/shard_0000.pt",
        "solver_rows": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_attempt_02/solver_rows",
        "frame_config": "configs/research/lightgen/g1_fixed_frame_semantic_v1.yaml",
        "optical_source_config": "configs/research/lightgen/small_decoder_native_gradient_ab_v1.yaml",
        "mnist_training_file": "data/MNIST/processed/training.pt",
    }
    _require(all(value["sources"].get(name) == path
                 for name, path in expected_sources.items())
             and value["sources"].get("mapper_checkpoint_run_id")
             == "3521afc652f2427eac984485971340c4",
             "explicit recent mapper, prior, projection, witness, config, or data source changed",
             ValueError)
    expected_endpoint_paths = {
        "C45k": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01/C/checkpoint_step_010000.pt",
        "D0": "outputs/20261022_lightgen_small_decoder_native_gradient_ab_v1/runs/autodl_backup_attempt_01/D0/checkpoint_step_005000.pt",
        "Dgrad": "outputs/20261022_lightgen_small_decoder_native_gradient_ab_v1/runs/autodl_backup_attempt_01/Dgrad/checkpoint_step_005000.pt",
        "J": "outputs/20261022_lightgen_small_joint_update_readability_pilot_v1/runs/autodl_backup_attempt_01/optical/checkpoint_step_005000.pt",
    }
    _require(all(value["sources"]["endpoints"][name].get("checkpoint") == path
                 for name, path in expected_endpoint_paths.items()),
             "an explicit C45k, D0, Dgrad, or J endpoint path changed", ValueError)
    return value


def _tree_max_delta(left: Any, right: Any) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        _require(left.shape == right.shape and left.dtype == right.dtype,
                 "state tensors differ in shape or dtype")
        if left.is_floating_point() or left.is_complex():
            return float((left.detach().cpu() - right.detach().cpu()).abs().max()) if left.numel() else 0.0
        return 0.0 if torch.equal(left.detach().cpu(), right.detach().cpu()) else math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        _require(set(left) == set(right), "state mappings have different keys")
        return max((_tree_max_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        _require(len(left) == len(right), "state sequences have different lengths")
        return max((_tree_max_delta(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float, str, bool, type(None))) and isinstance(
            right, (int, float, str, bool, type(None))):
        if isinstance(left, float) or isinstance(right, float):
            return 0.0 if math.isclose(float(left), float(right), rel_tol=RECOVERY_RTOL,
                                       abs_tol=RECOVERY_ATOL) else math.inf
        return 0.0 if left == right else math.inf
    return 0.0 if left == right else math.inf


def _states_close(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.allclose(
            left.detach().cpu(), right.detach().cpu(), rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_states_close(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _states_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=RECOVERY_RTOL, abs_tol=RECOVERY_ATOL)
    return left == right


def _optimizer_phase_steps(optimizer: torch.optim.Optimizer,
                           encoder: torch.nn.Parameter,
                           generator: torch.nn.Parameter) -> tuple[int, int]:
    result = []
    for parameter in (encoder, generator):
        state = optimizer.state.get(parameter)
        _require(isinstance(state, Mapping) and "step" in state,
                 "an endpoint Adam phase state is missing")
        result.append(int(torch.as_tensor(state["step"]).item()))
    return result[0], result[1]


def _assert_optical_optimizer(model: torch.nn.Module,
                              optimizer_parameters: Sequence[torch.nn.Parameter]) -> None:
    encoder, generator = base._phase_groups(model)
    expected = {id(encoder), id(generator)}
    _require(len(optimizer_parameters) == 2
             and {id(parameter) for parameter in optimizer_parameters} == expected,
             "optimizer must contain exactly Encoder and Generator phase parameters")


def _validate_dataset_roles(labels: torch.Tensor, train_indices: torch.Tensor,
                            dev_indices: torch.Tensor) -> None:
    total_by_class = torch.bincount(labels, minlength=CLASS_COUNT)
    train_by_class = torch.bincount(labels[train_indices], minlength=CLASS_COUNT)
    dev_by_class = torch.bincount(labels[dev_indices], minlength=CLASS_COUNT)
    expected_train_by_class = total_by_class - dev_by_class
    _require(train_indices.numel() == 55_000 and dev_indices.numel() == 5_000
             and not bool(torch.isin(train_indices, dev_indices).any())
             and torch.equal(dev_by_class, torch.full((CLASS_COUNT,), 500))
             and bool((expected_train_by_class > 0).all())
             and torch.equal(train_by_class, expected_train_by_class),
             "MNIST TRAIN/DEV roles changed or overlap")


def _validate_endpoint_identity(name: str, spec: Mapping[str, Any],
                                checkpoint: Mapping[str, Any]) -> None:
    expected_branch = spec["checkpoint_branch"]
    branch_matches = (
        "branch" not in checkpoint if expected_branch is None
        else checkpoint.get("branch") == expected_branch
    )
    model_identity = checkpoint.get("model_identity", {})
    model = checkpoint.get("model", {})
    _require(checkpoint.get("task_id") == spec["task_id"]
             and checkpoint.get("run_id") == spec["checkpoint_run_id"]
             and checkpoint.get("source_run_id") == spec["source_run_id"]
             and branch_matches
             and checkpoint.get("completed_global_step") == spec["global_step"]
             and model_identity.get("encoder") == [2, 100, 100]
             and model_identity.get("decoder") == [2, 100, 100]
             and model_identity.get("ols") == "gaussian_s12_v1"
             and tuple(model.get("runtime.encoder.phase", torch.empty(0)).shape) == (2, 1, 100, 100)
             and tuple(model.get("runtime.generator.phase", torch.empty(0)).shape) == (2, 1, 100, 100),
             f"{name} model identity, run lineage, endpoint step, or 2+2 phase shapes changed")


def _load_inputs(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    sources = config["sources"]
    frame_config = frame._load_config(_path(config, sources["frame_config"]))
    inputs = frame._load_inputs(frame_config)
    projector = torch.load(
        _path(config, sources["fixed_response_projector"]),
        map_location="cpu", weights_only=False,
    )
    _require(torch.equal(projector.get("response_matrix"), inputs["response_matrix"])
             and torch.equal(projector.get("scale"), inputs["scale"])
             and torch.equal(projector.get("full_right_inverse"), inputs["inverse"])
             and projector.get("real_imag_order") == "per_port_interleaved_real_imag",
             "fixed SLM projection differs from its registered response and scale")
    prior = torch.load(_path(config, sources["physical_phase_prior"]),
                       map_location="cpu", weights_only=False)
    _require(prior.get("schema") == f"{frame.TASK_ID}.prior.v1"
             and prior.get("task_id") == frame.TASK_ID
             and set(prior.get("fit", {})) == set(range(CLASS_COUNT))
             and torch.equal(prior.get("scale"), inputs["scale"]),
             "physical-phase prior identity, classes, or modal scale changed")
    mapper_checkpoint = torch.load(_path(config, sources["mapper_checkpoint"]),
                                   map_location="cpu", weights_only=False)
    _require(mapper_checkpoint.get("schema") == f"{frame.TASK_ID}.checkpoint.v1"
             and mapper_checkpoint.get("task_id") == frame.TASK_ID
             and mapper_checkpoint.get("run_id")
             == sources["mapper_checkpoint_run_id"]
             and mapper_checkpoint.get("completed_updates") == 15_000
             and mapper_checkpoint.get("checkpoint_step") == 15_000
             and mapper_checkpoint.get("source_mapper_updates") == 10_000
             and mapper_checkpoint.get("lineage_mapper_updates") == 25_000
             and float(mapper_checkpoint.get("selected_alpha", math.nan)) == 1.0
             and mapper_checkpoint.get("engine") == "response_matrix"
             and set(mapper_checkpoint.get("mapper_state", {})) == {
                 "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"
             },
             "frozen mapper is not the registered 15,000-update alpha=1 endpoint")
    replay_path = _path(config, sources["mapper_replay_shard"])
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    _require(replay.get("task_id") == frame.TASK_ID
             and replay.get("cohort_role") == "holdout"
             and tuple(replay.get("commands", torch.empty(0)).shape) == (10, 1, 200, 200)
             and tuple(replay.get("target_coefficients", torch.empty(0)).shape) == (10, PORT_COUNT),
             "saved mapper replay shard is missing or malformed")
    data_path = _path(config, sources["mnist_training_file"])
    pixels, labels, train_indices, dev_indices = base._load_data(data_path)
    _validate_dataset_roles(labels, train_indices, dev_indices)
    previous_config = decoder._read_config(_path(config, sources["optical_source_config"]))
    camera_area = frame._camera_area(inputs, device)
    _require(math.isclose(camera_area, CAMERA_AREA_EXPECTED, rel_tol=1.0e-12, abs_tol=0.0),
             "fixed camera pixel area changed")
    return {
        "frame_config": frame_config,
        "prior": prior,
        "projector": projector,
        "mapper_checkpoint": mapper_checkpoint,
        "mapper_replay": replay,
        "mapper_replay_path": replay_path,
        "pixels": pixels,
        "labels": labels,
        "train_indices": train_indices,
        "dev_indices": dev_indices,
        "previous_config": previous_config,
        "camera_area": camera_area,
        "device": device,
        "input_bundle": inputs,
        "camera_gain": inputs["camera_gain"],
        "matrix": inputs["matrix"].to(device=device, dtype=torch.float64),
        "inverse": inputs["inverse"].to(device=device, dtype=torch.float64),
        "scale": inputs["scale"].to(device=device, dtype=torch.float64),
    }


def _load_endpoints(config: Mapping[str, Any], runtime: Mapping[str, Any]
                    ) -> dict[str, dict[str, Any]]:
    endpoints: dict[str, dict[str, Any]] = {}
    c_reference: dict[str, Any] | None = None
    for name in ENDPOINT_ORDER:
        spec = config["sources"]["endpoints"][name]
        path = _path(config, spec["checkpoint"])
        _require(path.is_file(), f"explicit {name} checkpoint is missing: {path}", FileNotFoundError)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _validate_endpoint_identity(name, spec, checkpoint)
        _require(checkpoint.get("evaluator_role") == "frozen_confirmation_only"
                 and set(checkpoint["model"]) == set(runtime["input_bundle"]["checkpoint"]["model"]),
                 f"{name} model state or evaluator role differs from the registered C-series model")
        if c_reference is None:
            c_reference = checkpoint
        else:
            for key in ("scale", "power_reference", "camera_gain", "weights", "structural_weight",
                        "model_identity"):
                left, right = c_reference.get(key), checkpoint.get(key)
                if torch.is_tensor(left):
                    _require(torch.equal(left, right), f"{name} changed C-series fixed reference {key}")
                else:
                    _require(left == right, f"{name} changed C-series fixed reference {key}")
            for key, value in c_reference["model"].items():
                if key not in {"runtime.encoder.phase", "runtime.generator.phase"}:
                    _require(torch.equal(value, checkpoint["model"][key]),
                             f"{name} changed frozen optical component {key}")
        expected_steps = (spec["encoder_adam_step"], spec["generator_adam_step"])
        _validate_saved_optical_adam(checkpoint.get("optimizer"), expected_steps)
        steps = expected_steps
        endpoints[name] = {"path": path, "checkpoint": checkpoint, "optimizer_steps": steps}
    c = endpoints["C45k"]["checkpoint"]["model"]
    for name in ("D0", "Dgrad"):
        _require(torch.equal(c["runtime.encoder.phase"],
                             endpoints[name]["checkpoint"]["model"]["runtime.encoder.phase"]),
                 f"{name} is not the registered Generator-only C45k continuation")
    j_model = endpoints["J"]["checkpoint"]["model"]
    _require(not torch.equal(c["runtime.encoder.phase"], j_model["runtime.encoder.phase"])
             and not torch.equal(c["runtime.generator.phase"], j_model["runtime.generator.phase"]),
             "J endpoint no longer contains joint Encoder/Generator phase updates")
    return endpoints


def _validate_saved_optical_adam(
    payload: Mapping[str, Any] | None, expected_steps: tuple[int, int],
) -> None:
    """Validate the saved Encoder/Generator Adam payload without constructing an optimizer."""
    _require(isinstance(payload, Mapping)
             and isinstance(payload.get("param_groups"), list)
             and len(payload["param_groups"]) == 1
             and isinstance(payload.get("state"), Mapping),
             "saved optical Adam payload is malformed")
    group = payload["param_groups"][0]
    parameter_ids = group.get("params")
    _require(isinstance(parameter_ids, list) and len(parameter_ids) == 2
             and parameter_ids[0] != parameter_ids[1]
             and float(group.get("lr", math.nan)) == 1.0e-3
             and tuple(group.get("betas", ())) == (0.9, 0.999)
             and float(group.get("eps", math.nan)) == 1.0e-8
             and float(group.get("weight_decay", math.nan)) == 0.0
             and group.get("amsgrad") is False,
             "saved optical Adam parameter order or settings changed")
    _require(set(payload["state"]) == set(parameter_ids),
             "saved optical Adam states do not cover exactly the two phase parameters")
    for parameter_id, expected_step in zip(parameter_ids, expected_steps, strict=True):
        state = payload["state"][parameter_id]
        _require(isinstance(state, Mapping)
                 and float(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
                 == float(expected_step)
                 and torch.is_tensor(state.get("exp_avg"))
                 and tuple(state["exp_avg"].shape) == (2, 1, 100, 100)
                 and torch.is_tensor(state.get("exp_avg_sq"))
                 and tuple(state["exp_avg_sq"].shape) == (2, 1, 100, 100)
                 and torch.isfinite(state["exp_avg"]).all().item()
                 and torch.isfinite(state["exp_avg_sq"]).all().item(),
                 "saved Adam step, moment shape, or moment values do not match its optical phase")


def _new_mapper(runtime: Mapping[str, Any]) -> torch.nn.Module:
    mapper = frame._new_mapper(runtime["input_bundle"], runtime["device"],
                               runtime["mapper_checkpoint"]["mapper_state"])
    mapper.eval()
    for parameter in mapper.parameters():
        parameter.requires_grad_(False)
    return mapper


def _assert_legal_commands(commands: torch.Tensor) -> None:
    _require(torch.is_tensor(commands) and commands.is_floating_point()
             and commands.ndim == 4 and tuple(commands.shape[1:]) == COMMAND_SHAPE
             and torch.isfinite(commands).all().item()
             and float(commands.min()) >= 0.0 and float(commands.max()) <= 1.0,
             "SLM commands must be finite float amplitudes in [0,1]")


def _make_commands(mapper: torch.nn.Module, coefficients: torch.Tensor,
                   runtime: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    device = runtime["device"]
    targets_cpu = coefficients.detach().cpu().to(torch.complex128)
    _require(targets_cpu.ndim == 2 and targets_cpu.shape[1] == PORT_COUNT
             and torch.isfinite(targets_cpu.real).all().item()
             and torch.isfinite(targets_cpu.imag).all().item(),
             "prior samples must be finite complex OLS targets")
    features = projected._normalized_features(targets_cpu, runtime["scale"], device)
    commands, clipped = [], []
    with torch.inference_mode():
        for start in range(0, len(targets_cpu), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(targets_cpu))
            target = targets_cpu[start:stop].to(device=device, dtype=torch.complex128)
            command, detail = projected._map_amplitude(
                mapper, features[start:stop], target, runtime["matrix"],
                runtime["inverse"], runtime["scale"],
            )
            commands.append(command.detach().to(torch.float32).cpu())
            clipped.append(detail["clipped_fraction"].detach().double().cpu())
    result = torch.cat(commands)
    _assert_legal_commands(result)
    return result, {
        "minimum": float(result.min()),
        "maximum": float(result.max()),
        "mean_clipped_fraction": float(torch.cat(clipped).mean()),
    }


def _assert_mapper_replay(commands: torch.Tensor, expected: torch.Tensor) -> float:
    _assert_legal_commands(commands)
    _require(torch.is_tensor(expected) and expected.is_floating_point()
             and expected.shape == commands.shape and torch.isfinite(expected).all().item()
             and float(expected.min()) >= 0.0 and float(expected.max()) <= 1.0,
             "saved mapper replay commands are malformed")
    max_error = float((commands - expected).abs().max())
    _require(max_error <= COMMAND_REPLAY_ATOL,
             f"frozen mapper failed saved-command replay: max_abs={max_error:.6g}")
    return max_error


def _verify_mapper_replay(mapper: torch.nn.Module,
                          runtime: Mapping[str, Any]) -> dict[str, Any]:
    replay = runtime["mapper_replay"]
    commands, details = _make_commands(
        mapper, replay["target_coefficients"].to(torch.complex128), runtime
    )
    expected = replay["commands"].float()
    max_error = _assert_mapper_replay(commands, expected)
    return {
        "task_id": TASK_ID,
        "saved_shard": str(runtime["mapper_replay_path"]),
        "rows": int(commands.shape[0]),
        "max_abs_command_error": max_error,
        "tolerance": COMMAND_REPLAY_ATOL,
        "within_slm_bounds": True,
        "command_summary": details,
    }


def _sample_cohort(runtime: Mapping[str, Any], *, count: int, seed_base: int,
                   role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return frame._cohort(
        runtime["prior"]["fit"], runtime["scale"].cpu(), alpha=1.0,
        role=role, count=count, seed_base=seed_base, config=runtime["frame_config"],
    )


def _build_train_schedule(labels: torch.Tensor, train_indices: torch.Tensor,
                          seed: int, updates: int = TRAIN_UPDATES) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = torch.empty((updates, CLASS_COUNT), dtype=torch.long)
    for label in range(CLASS_COUNT):
        candidates = train_indices[labels[train_indices] == label]
        _require(candidates.numel() >= updates, f"TRAIN class {label} lacks distinct rows")
        permutation = torch.randperm(candidates.numel(), generator=generator)
        schedule[:, label] = candidates[permutation[:updates]]
    _require(torch.equal(labels[schedule], torch.arange(CLASS_COUNT).expand(updates, CLASS_COUNT))
             and all(torch.unique(schedule[:, label]).numel() == updates
                     for label in range(CLASS_COUNT)),
             "two-thousand-step schedule is not balanced or contains repeated class rows")
    return schedule


def _choose_endpoint(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    def metrics(name: str) -> tuple[float, float]:
        evaluators = summaries[name]["conditions"]["direct_decoder"]["evaluators"]
        return (float(evaluators["ED"]["macro_recall"]),
                float(evaluators["EC"]["macro_recall"]))

    _require(set(summaries) == set(ENDPOINT_ORDER), "endpoint comparison is incomplete")
    return max(ENDPOINT_ORDER, key=lambda name: (
        min(metrics(name)), sum(metrics(name)) / 2.0, -ENDPOINT_ORDER.index(name)
    ))


def _should_skip_training(selected: str, summaries: Mapping[str, Mapping[str, Any]],
                          average_gain: float = 0.02,
                          individual_minimum: float = 0.0) -> tuple[bool, dict[str, Any]]:
    baseline = summaries["C45k"]["conditions"]["direct_decoder"]["evaluators"]
    candidate = summaries[selected]["conditions"]["direct_decoder"]["evaluators"]
    deltas = {
        family: float(candidate[family]["macro_recall"] - baseline[family]["macro_recall"])
        for family in ("ED", "EC")
    }
    average_delta = sum(deltas.values()) / 2.0
    skip = average_delta + 1.0e-12 >= average_gain and all(
        value >= individual_minimum for value in deltas.values()
    )
    return skip, {
        "selected_endpoint": selected,
        "baseline": "C45k",
        "ED_macro_recall_delta": deltas["ED"],
        "EC_macro_recall_delta": deltas["EC"],
        "average_macro_recall_delta": round(average_delta, 12),
        "average_gain_minimum": average_gain,
        "individual_delta_minimum": individual_minimum,
        "skip_short_training": skip,
    }


def _new_endpoint_model(checkpoint: Mapping[str, Any], grid: int,
                        runtime: Mapping[str, Any]) -> torch.nn.Module:
    model = exposure._make_runtime_model(runtime["input_bundle"], grid, runtime["device"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _measure_commands(checkpoint: Mapping[str, Any], commands: torch.Tensor,
                      runtime: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    _assert_legal_commands(commands)
    model = _new_endpoint_model(checkpoint, GRID_PRODUCTION, runtime)
    coefficients, raw_images, readouts = [], [], []
    gain = float(runtime["camera_gain"])
    with torch.inference_mode():
        for start in range(0, commands.shape[0], BATCH_SIZE):
            command = commands[start:start + BATCH_SIZE].to(runtime["device"])
            output = projected._forward_amplitude(model, command)
            coefficients.append(output["coefficients"].detach().cpu())
            raw = output["raw_cmos"].detach().cpu()
            raw_images.append(raw)
            readouts.append(pilot._readout_for_evaluator(raw, runtime["camera_area"], gain).cpu())
    del model
    torch.cuda.empty_cache()
    return {
        "coefficients": torch.cat(coefficients),
        "raw_cmos": torch.cat(raw_images),
        "readouts": torch.cat(readouts).float(),
    }


def _measure_endpoint_cohort(name: str, endpoint: Mapping[str, Any],
                             cohort: Mapping[str, Any], commands: torch.Tensor,
                             runtime: Mapping[str, Any], output_dir: Path
                             ) -> dict[str, Any]:
    measured = _measure_commands(endpoint["checkpoint"], commands, runtime)
    summary = frame._summary_from_readouts(runtime["input_bundle"], cohort, measured["readouts"])
    payload = {
        "schema": f"{TASK_ID}.endpoint_measurement.v1",
        "task_id": TASK_ID,
        "condition": name,
        "checkpoint_path": str(endpoint["path"]),
        "checkpoint_run_id": endpoint["checkpoint"]["run_id"],
        "global_step": endpoint["checkpoint"]["completed_global_step"],
        "cohort_role": cohort["role"],
        "labels": cohort["labels"].clone(),
        "proposal_ids": cohort["proposal_ids"].clone(),
        "commands": commands.clone(),
        **measured,
        "semantic_summary": summary,
    }
    _save_torch(output_dir / f"development_{name}.pt", payload, new=True)
    return summary


def _load_saved_route_evidence(config: Mapping[str, Any],
                               runtime: Mapping[str, Any],
                               endpoints: Mapping[str, Mapping[str, Any]],
                               mapper: torch.nn.Module,
                               output_dir: Path) -> dict[str, Any]:
    saved_config = output_dir / "run_config.yaml"
    _require(saved_config.is_file()
             and yaml.safe_load(saved_config.read_text(encoding="utf-8")) == config,
             "partial attempt configuration differs from the registered task")
    receipt_path = output_dir / "asset_reuse_receipt.json"
    _require(receipt_path.is_file(), "partial attempt asset receipt is missing",
             FileNotFoundError)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("task_id") == TASK_ID
             and receipt.get("mapper_completed_updates") == 15_000
             and receipt.get("training_rows") == 55_000
             and receipt.get("development_rows") == 5_000
             and receipt.get("official_test_reads") == 0,
             "partial attempt asset receipt has the wrong identity or data roles")
    for name, endpoint in endpoints.items():
        saved = receipt.get("sources", {}).get(name, {})
        checkpoint = endpoint["checkpoint"]
        _require(saved.get("run_id") == checkpoint["run_id"]
                 and saved.get("source_run_id") == checkpoint.get("source_run_id")
                 and saved.get("task_id") == checkpoint["task_id"]
                 and saved.get("global_step") == checkpoint["completed_global_step"]
                 and saved.get("optimizer_steps") == list(endpoint["optimizer_steps"]),
                 f"partial attempt asset receipt changed the {name} endpoint")

    replay = _verify_mapper_replay(mapper, runtime)
    replay_path = output_dir / "mapper_replay.json"
    _require(replay_path.is_file(), "partial attempt mapper replay receipt is missing",
             FileNotFoundError)
    replay_saved = json.loads(replay_path.read_text(encoding="utf-8"))
    _require(replay_saved.get("task_id") == TASK_ID
             and replay_saved.get("rows") == replay["rows"]
             and replay_saved.get("max_abs_command_error", math.inf) <= COMMAND_REPLAY_ATOL,
             "partial attempt mapper replay receipt is invalid")

    population_path = output_dir / "development_population_and_commands.pt"
    _require(population_path.is_file(), "saved development population is missing",
             FileNotFoundError)
    population = torch.load(population_path, map_location="cpu", weights_only=False)
    cohort = population.get("cohort")
    commands = population.get("commands")
    _require(population.get("schema") == f"{TASK_ID}.development_population.v1"
             and population.get("task_id") == TASK_ID
             and population.get("seed_base") == 202610300910
             and population.get("mapper_checkpoint_run_id")
             == runtime["mapper_checkpoint"]["run_id"]
             and isinstance(cohort, Mapping)
             and cohort.get("role") == "new_development_prior_200_not_final_holdout",
             "saved development population has the wrong identity or role")
    _validate_cohort(cohort, 20)
    _assert_legal_commands(commands)
    regenerated, _details = _make_commands(mapper, cohort["coefficients"], runtime)
    _require(regenerated.shape == commands.shape
             and float((regenerated - commands).abs().max()) <= COMMAND_REPLAY_ATOL,
             "frozen input adapter no longer reproduces saved development commands")

    summaries: dict[str, Any] = {}
    for name, endpoint in endpoints.items():
        path = output_dir / f"development_{name}.pt"
        _require(path.is_file(), f"saved {name} development measurement is missing",
                 FileNotFoundError)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        _require(artifact.get("schema") == f"{TASK_ID}.endpoint_measurement.v1"
                 and artifact.get("task_id") == TASK_ID
                 and artifact.get("condition") == name
                 and _path(config, artifact.get("checkpoint_path", "")) == endpoint["path"]
                 and artifact.get("checkpoint_run_id") == endpoint["checkpoint"]["run_id"]
                 and artifact.get("global_step") == endpoint["checkpoint"]["completed_global_step"]
                 and artifact.get("cohort_role") == cohort["role"]
                 and torch.equal(artifact.get("labels"), cohort["labels"])
                 and torch.equal(artifact.get("proposal_ids"), cohort["proposal_ids"])
                 and torch.equal(artifact.get("commands"), commands),
                 f"saved {name} development measurement is not bound to the fixed commands")
        produced = {key: artifact.get(key) for key in ("coefficients", "raw_cmos", "readouts")}
        _require(torch.is_tensor(produced["coefficients"])
                 and produced["coefficients"].shape == (200, PORT_COUNT)
                 and produced["coefficients"].is_complex()
                 and all(torch.is_tensor(value) and value.shape[0] == 200
                         and torch.isfinite(value).all().item() for value in produced.values()),
                 f"saved {name} development optical outputs are incomplete")
        summary = frame._summary_from_readouts(
            runtime["input_bundle"], cohort, produced["readouts"]
        )
        _require(_states_close(summary, artifact.get("semantic_summary")),
                 f"saved {name} evaluator summary differs from its raw CMOS readout")
        summaries[name] = summary

    selected_name = _choose_endpoint(summaries)
    skip_training, skip_decision = _should_skip_training(
        selected_name, summaries,
        float(config["measurement"]["endpoint_skip_average_macro_recall_gain"]),
        float(config["measurement"]["endpoint_skip_each_evaluator_minimum_delta"]),
    )
    route_path = output_dir / "endpoint_route_decision.json"
    _require(route_path.is_file(), "saved endpoint route decision is missing", FileNotFoundError)
    route = json.loads(route_path.read_text(encoding="utf-8"))
    _require(route.get("task_id") == TASK_ID
             and route.get("selected_endpoint") == selected_name
             and route.get("skip_decision", {}).get("skip_short_training") is skip_training
             and route.get("conditional_training")
             == ("skip" if skip_training else "exactly_2000_joint_optical_updates")
             and route.get("final_population_consumed") is False,
             "saved endpoint route decision differs from the registered selection rule")
    saved_skip = route["skip_decision"]
    _require(saved_skip.get("baseline") == "C45k"
             and all(math.isclose(float(saved_skip.get(key, math.nan)),
                                  float(skip_decision[key]), abs_tol=1.0e-12)
                     for key in ("ED_macro_recall_delta", "EC_macro_recall_delta",
                                 "average_macro_recall_delta")),
             "saved endpoint skip evidence differs from the recomputed development metrics")

    positive_path = output_dir / "solver_positive_controls.json"
    _require(positive_path.is_file(), "saved solver positive controls are missing",
             FileNotFoundError)
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    witness_rows = _load_solver_positive_controls(config)
    expected_witnesses = [
        (row["label"], row["proposal_id"], row["condition"]) for row in witness_rows
    ]
    saved_witnesses = [
        (row.get("label"), row.get("proposal_id"), row.get("condition"))
        for row in positive.get("rows", [])
    ]
    _require(positive.get("task_id") == TASK_ID
             and positive.get("source_role") == "historical_engineering_positive_controls_only"
             and positive.get("passed") is True
             and positive.get("rows_per_class") == {str(label): 2 for label in range(CLASS_COUNT)}
             and saved_witnesses == expected_witnesses
             and all(row.get("all_metric_values_within_tolerance") is True
                     for row in positive.get("rows", [])),
             "saved 4,096-grid positive controls are incomplete or not bound to the witnesses")
    return {
        "cohort": cohort,
        "commands": commands,
        "endpoint_summaries": summaries,
        "route": route,
        "positive": positive,
        "selected_name": selected_name,
        "selected": {"name": selected_name, **endpoints[selected_name]},
        "skip_training": skip_training,
        "skip_decision": skip_decision,
    }


def _load_solver_positive_controls(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    directory = _path(config, config["sources"]["solver_rows"])
    records = []
    for index in range(20):
        path = directory / f"row_{index:04d}.pt"
        _require(path.is_file(), f"saved solver positive control is missing: {path}",
                 FileNotFoundError)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("task_id") == "LIGHTGEN_G1_FIXED_FRAME_TRANSFER_DIAGNOSIS_V1"
                 and payload.get("scientific_updates") == 0
                 and payload.get("mapper_updates") == 0
                 and payload.get("optical_updates") == 0,
                 "solver positive-control lineage changed")
        label = int(payload["base_row"]["label"])
        result = payload["solver_results"]["fista_fixed_terminal_phase"]
        command = result["command"].to(torch.float32)
        _assert_legal_commands(command)
        records.append({
            "path": path, "label": label,
            "proposal_id": int(payload["base_row"]["proposal_id"]),
            "target": payload["base_row"]["target_coefficients"].reshape(1, PORT_COUNT),
            "command": command.reshape(1, 1, 200, 200),
            "condition": result["condition"],
        })
    selected = []
    counts = {label: 0 for label in range(CLASS_COUNT)}
    for row in records:
        if counts[row["label"]] < 2:
            selected.append(row)
            counts[row["label"]] += 1
    _require(counts == {label: 2 for label in range(CLASS_COUNT)}
             and len({(row["label"], row["proposal_id"]) for row in selected}) == 20,
             "saved engineering positives do not provide two distinct rows per class")
    return selected


def _measure_positive_controls(config: Mapping[str, Any],
                               runtime: Mapping[str, Any],
                               c_checkpoint: Mapping[str, Any],
                               output_dir: Path) -> dict[str, Any]:
    rows = _load_solver_positive_controls(config)
    models = {
        grid: _new_endpoint_model(c_checkpoint, grid, runtime)
        for grid in (GRID_PRODUCTION, GRID_REFERENCE)
    }
    results = []
    with torch.inference_mode():
        for row in rows:
            target = row["target"].to(runtime["device"], dtype=torch.complex128)
            outputs, direct = {}, {}
            for grid in (GRID_PRODUCTION, GRID_REFERENCE):
                outputs[grid] = projected._forward_amplitude(
                    models[grid], row["command"].to(runtime["device"])
                )
                direct[grid] = projected._forward_target(models[grid], target)["raw_cmos"]
            metrics = projected._error_row(outputs, target, direct, 0)
            errors = [float(value) for key, value in metrics.items()
                      if key not in {"grid_2048", "grid_4096"}]
            _require(bool(errors) and max(errors) <= GATE_LIMIT,
                     f"saved solver positive failed full-path grid check: {metrics}")
            results.append({
                "label": row["label"], "proposal_id": row["proposal_id"],
                "condition": row["condition"], "metrics": metrics,
                "all_metric_values_within_tolerance": True,
            })
    del models
    torch.cuda.empty_cache()
    summary = {
        "task_id": TASK_ID,
        "source_role": "historical_engineering_positive_controls_only",
        "rows": results,
        "rows_per_class": {str(label): 2 for label in range(CLASS_COUNT)},
        "independent_reference_grid": GRID_REFERENCE,
        "existing_numerical_tolerance": GATE_LIMIT,
        "passed": len(results) == 20,
        "semantic_evidence": False,
        "hardware_evidence": False,
    }
    _write_json(output_dir / "solver_positive_controls.json", summary)
    return summary


def _real_dev_rows(labels: torch.Tensor, dev_indices: torch.Tensor) -> torch.Tensor:
    rows = torch.cat([
        dev_indices[labels[dev_indices] == label][:20]
        for label in range(CLASS_COUNT)
    ])
    _require(rows.numel() == 200
             and torch.equal(torch.bincount(labels[rows], minlength=CLASS_COUNT),
                             torch.full((CLASS_COUNT,), 20)),
             "real-image DEV diagnostic must contain 20 rows per class")
    return rows


def _measure_reconstruction(checkpoint: Mapping[str, Any], rows: torch.Tensor,
                            runtime: Mapping[str, Any]) -> dict[str, Any]:
    model = base._model("small", GRID_PRODUCTION, runtime["device"], runtime["previous_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pixels = runtime["pixels"]
    device = runtime["device"]
    gain = checkpoint["camera_gain"].to(device=device, dtype=torch.float32)
    native_mse, view_mse, ssim_values = [], [], []
    with torch.inference_mode():
        for start in range(0, rows.numel(), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            target_native = image_batch(pixels, batch_rows, "small", device)
            result = base._forward(model, target_native)
            gained = result["physical_w_per_pixel"] * gain
            target28 = pixels.index_select(0, batch_rows).unsqueeze(1).to(
                device=device, dtype=torch.float32
            ) / 255.0
            view = resize_view(gained)
            native_mse.append((gained - target_native).square().mean().cpu())
            view_mse.append((view - target28).square().mean().cpu())
            ssim_values.append(ssim_batch(view, target28).mean().cpu())
    del model
    torch.cuda.empty_cache()
    mse_native, mse_view, ssim = (
        float(torch.stack(values).mean()) for values in (native_mse, view_mse, ssim_values)
    )
    return {
        "rows": int(rows.numel()),
        "role": "MNIST_DEV_reconstruction_diagnostic_only",
        "native_camera_mse_fixed_gain": mse_native,
        "view28_mse_fixed_gain": mse_view,
        "view28_psnr_db": 10.0 * math.log10(1.0 / max(mse_view, 1.0e-30)),
        "view28_ssim": ssim,
        "official_test_reads": 0,
    }


def _optimizer_and_model(checkpoint: Mapping[str, Any],
                         runtime: Mapping[str, Any],
                         expected_steps: tuple[int, int] | None = None
                         ) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = base._model("small", GRID_PRODUCTION, runtime["device"], runtime["previous_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder, generator = base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is encoder or parameter is generator)
    group = checkpoint["optimizer"]["param_groups"][0]
    optimizer = torch.optim.Adam(
        (encoder, generator), lr=float(group["lr"]),
        betas=tuple(float(value) for value in group["betas"]),
        eps=float(group["eps"]), weight_decay=float(group["weight_decay"]),
    )
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    _assert_optical_optimizer(
        model, [parameter for item in optimizer.param_groups for parameter in item["params"]]
    )
    actual_steps = _optimizer_phase_steps(optimizer, encoder, generator)
    if expected_steps is None:
        recorded = checkpoint.get("optimizer_steps")
        if recorded is not None:
            expected_steps = (
                int(recorded["encoder_adam_step"]),
                int(recorded.get("generator_adam_step", recorded.get("decoder_adam_step"))),
            )
        else:
            expected_steps = tuple(
                int(torch.as_tensor(state["step"]).item())
                for state in optimizer.state.values()
            )
    _require(actual_steps == expected_steps,
             f"selected endpoint Adam state differs: expected {expected_steps}, got {actual_steps}")
    return model, optimizer


def _snapshot_frozen(runtime: Mapping[str, Any], mapper: torch.nn.Module) -> dict[str, Any]:
    inputs = runtime["input_bundle"]
    return {
        "mapper": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "prior_scale": runtime["prior"]["scale"].clone(),
        "prior_fit": copy.deepcopy(runtime["prior"]["fit"]),
        "projection_scale": runtime["scale"].detach().cpu().clone(),
        "projection_matrix": runtime["matrix"].detach().cpu().clone(),
        "projection_inverse": runtime["inverse"].detach().cpu().clone(),
        "response_matrix": inputs["response_matrix"].clone(),
        "inverse": inputs["inverse"].clone(),
        "camera_gain": torch.as_tensor(runtime["camera_gain"]).detach().cpu().clone(),
        "evaluators": {
            family: {key: value.detach().cpu().clone()
                     for key, value in inputs[name].state_dict().items()}
            for family, name in (("ED", "evaluator_ed"), ("EC", "evaluator_ec"))
        },
    }


def _assert_frozen_unchanged(snapshot: Mapping[str, Any], runtime: Mapping[str, Any],
                             mapper: torch.nn.Module) -> None:
    inputs = runtime["input_bundle"]
    _require(_tree_max_delta(snapshot["mapper"], mapper.state_dict()) == 0.0,
             "frozen SLM mapper changed")
    _require(torch.equal(snapshot["prior_scale"], runtime["prior"]["scale"])
             and _tree_max_delta(snapshot["prior_fit"], runtime["prior"]["fit"]) == 0.0
             and torch.equal(snapshot["projection_scale"], runtime["scale"].detach().cpu())
             and torch.equal(snapshot["projection_matrix"], runtime["matrix"].detach().cpu())
             and torch.equal(snapshot["projection_inverse"], runtime["inverse"].detach().cpu())
             and torch.equal(snapshot["response_matrix"], inputs["response_matrix"])
             and torch.equal(snapshot["inverse"], inputs["inverse"])
             and torch.equal(snapshot["camera_gain"],
                             torch.as_tensor(runtime["camera_gain"]).detach().cpu()),
             "frozen prior, projection, or camera gain changed")
    for family, name in (("ED", "evaluator_ed"), ("EC", "evaluator_ec")):
        _require(_tree_max_delta(snapshot["evaluators"][family],
                                 inputs[name].state_dict()) == 0.0,
                 f"frozen evaluator {family} changed")


def _verify_wrong_target_negative(model: torch.nn.Module, pixels: torch.Tensor,
                                  rows: torch.Tensor, runtime: Mapping[str, Any],
                                  source: Mapping[str, Any]) -> dict[str, float]:
    device = runtime["device"]
    cp = source["checkpoint"]
    target = image_batch(pixels, rows, "small", device)
    wrong_rows = rows.roll(1)
    wrong_target = image_batch(pixels, wrong_rows, "small", device)
    result = base._forward(model, target)
    scale = cp["scale"].to(device)
    power = cp["power_reference"].to(device)
    gain = cp["camera_gain"].to(device)
    correct_parts = base._loss_parts_from_result(result, target, scale, power, gain)
    wrong_parts = base._loss_parts_from_result(result, wrong_target, scale, power, gain)
    view = resize_view(result["physical_w_per_pixel"] * gain.to(torch.float32))
    target28 = pixels.index_select(0, rows).unsqueeze(1).to(
        device=device, dtype=torch.float32
    ) / 255.0
    wrong28 = pixels.index_select(0, wrong_rows).unsqueeze(1).to(
        device=device, dtype=torch.float32
    ) / 255.0
    correct_ssim = (1.0 - ssim_batch(view, target28)).mean()
    wrong_ssim = (1.0 - ssim_batch(view, wrong28)).mean()
    weights = cp["weights"]
    structural = float(cp["structural_weight"])
    correct = sum(float(weights[key]) * correct_parts[key]
                  for key in ("dist", "rec", "op")) + structural * correct_ssim
    wrong = sum(float(weights[key]) * wrong_parts[key]
                for key in ("dist", "rec", "op")) + structural * wrong_ssim
    delta = float((wrong - correct).detach().abs())
    _require(math.isfinite(delta) and delta > 1.0e-7,
             "wrong-target negative control was not distinguishable from the aligned loss")
    return {
        "aligned_total": float(correct.detach()),
        "wrong_target_total": float(wrong.detach()),
        "absolute_delta": delta,
        "rejected": True,
    }


def _verify_cmos_gradients(model: torch.nn.Module, pixels: torch.Tensor,
                           rows: torch.Tensor, runtime: Mapping[str, Any],
                           source: Mapping[str, Any]) -> dict[str, float]:
    cp = source["checkpoint"]
    device = runtime["device"]
    parts, ssim_loss, _gradient, _result = decoder._loss_parts(
        model, pixels, rows, device, cp["scale"].to(device),
        cp["power_reference"].to(device), cp["camera_gain"].to(device),
    )
    image_term = float(cp["weights"]["rec"]) * parts["rec"] \
        + float(cp["structural_weight"]) * ssim_loss
    encoder, generator = base._phase_groups(model)
    gradients = torch.autograd.grad(image_term, (encoder, generator))
    result = {}
    for name, gradient in zip(
            ("encoder_cmos_gradient_norm", "generator_cmos_gradient_norm"),
            gradients, strict=True):
        norm = float(torch.linalg.vector_norm(gradient.detach().double()).cpu())
        _require(math.isfinite(norm) and norm > 0.0,
                 f"CMOS image term is disconnected from {name}")
        result[name] = norm
    return result


def _recovery_worker(config_path: Path, recovery_input: Path, recovery_output: Path) -> None:
    config = _read_config(config_path)
    payload = torch.load(recovery_input, map_location="cpu", weights_only=False)
    _require(payload.get("task_id") == TASK_ID and payload.get("completed_updates") == 3,
             "recovery worker received a different preflight prefix")
    previous_config = decoder._read_config(_path(config, config["sources"]["optical_source_config"]))
    pixels, labels, train_indices, _dev = base._load_data(
        _path(config, config["sources"]["mnist_training_file"])
    )
    device = torch.device("cuda:0")
    model = base._model("small", GRID_PRODUCTION, device, previous_config)
    model.load_state_dict(payload["model"], strict=True)
    encoder, generator = base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is encoder or parameter is generator)
    optimizer = torch.optim.Adam((encoder, generator), lr=1.0e-3, betas=(0.9, 0.999),
                                 eps=1.0e-8, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizer"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    torch.random.set_rng_state(payload["cpu_rng"])
    source = {"checkpoint": payload["source_checkpoint"],
              "training_labels": labels, "training_indices": train_indices,
              "training_pixels": pixels}
    row = payload["schedule"][3]
    loss = joint._joint_update(model, optimizer, pixels, row, device, source)
    torch.cuda.synchronize(device)
    steps = _optimizer_phase_steps(optimizer, encoder, generator)
    _save_torch(recovery_output, {
        "task_id": TASK_ID,
        "completed_updates": 4,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": list(steps),
        "loss": loss,
        "rows": row.clone(),
        "cpu_rng": torch.random.get_rng_state().cpu().clone(),
        "cuda_rng": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()],
    }, new=True)


def _validate_preflight_prefix(prefix: Mapping[str, Any],
                               selected: Mapping[str, Any],
                               runtime: Mapping[str, Any],
                               schedule: torch.Tensor) -> tuple[int, int]:
    source_checkpoint = selected["checkpoint"]
    source_steps = selected["optimizer_steps"]
    _require(prefix.get("task_id") == TASK_ID and prefix.get("completed_updates") == 3
             and torch.equal(prefix.get("schedule"), schedule)
             and _states_close(prefix.get("source_checkpoint"), source_checkpoint),
             "saved three-step preflight prefix is not bound to this endpoint and schedule")
    model_state = prefix.get("model")
    _require(isinstance(model_state, Mapping)
             and set(model_state) == set(source_checkpoint["model"]),
             "saved preflight prefix model state is incomplete")
    for key, value in source_checkpoint["model"].items():
        if key not in {"runtime.encoder.phase", "runtime.generator.phase"}:
            _require(torch.equal(value, model_state[key]),
                     f"saved preflight prefix changed frozen optical component {key}")
    for key in ("runtime.encoder.phase", "runtime.generator.phase"):
        _require(not torch.equal(source_checkpoint["model"][key], model_state[key]),
                 f"saved preflight prefix did not update {key}")
    _require(torch.is_tensor(prefix.get("cpu_rng"))
             and isinstance(prefix.get("cuda_rng"), list)
             and len(prefix["cuda_rng"]) == torch.cuda.device_count()
             and all(torch.is_tensor(value) for value in prefix["cuda_rng"]),
             "saved preflight prefix lacks complete CPU/CUDA RNG state")
    check_model, check_optimizer = _optimizer_and_model(
        source_checkpoint, runtime, source_steps
    )
    check_model.load_state_dict(model_state, strict=True)
    check_optimizer.load_state_dict(copy.deepcopy(prefix["optimizer"]))
    encoder, generator = base._phase_groups(check_model)
    steps = _optimizer_phase_steps(check_optimizer, encoder, generator)
    expected = (source_steps[0] + 3, source_steps[1] + 3)
    _require(steps == expected
             and all(torch.isfinite(value).all().item()
                     for state in check_optimizer.state.values()
                     for value in state.values() if torch.is_tensor(value)),
             "saved preflight prefix Adam state is incomplete or non-finite")
    del check_model, check_optimizer
    torch.cuda.empty_cache()
    return steps


def _run_recovery_worker(config_path: Path, recovery_input: Path,
                         recovery_output: Path) -> dict[str, Any]:
    if not recovery_output.is_file():
        completed = subprocess.run([
            sys.executable, "-m",
            "scripts.research.lightgen.execute_lightgen_recent_optical_fullpath_semantic_v1",
            "--mode", "recovery-worker", "--config", str(config_path),
            "--recovery-input", str(recovery_input),
            "--recovery-output", str(recovery_output),
        ], cwd=ROOT, capture_output=True, text=True)
        detail = (completed.stderr or completed.stdout)[-4_000:]
        _require(completed.returncode == 0,
                 f"independent recovery worker failed ({completed.returncode}): {detail}")
    _require(recovery_output.is_file(), "independent recovery worker did not save its result")
    result = torch.load(recovery_output, map_location="cpu", weights_only=False)
    _require(result.get("task_id") == TASK_ID and result.get("completed_updates") == 4,
             "independent recovery worker returned a different task prefix")
    return result


def _preflight_training(config: Mapping[str, Any], selected: Mapping[str, Any],
                        runtime: Mapping[str, Any], mapper: torch.nn.Module,
                        schedule: torch.Tensor, snapshot: Mapping[str, Any],
                        output_dir: Path) -> dict[str, Any]:
    device = runtime["device"]
    source = {"checkpoint": selected["checkpoint"],
              "training_labels": runtime["labels"],
              "training_indices": runtime["train_indices"],
              "training_pixels": runtime["pixels"]}
    model, optimizer = _optimizer_and_model(
        selected["checkpoint"], runtime, selected["optimizer_steps"]
    )
    encoder, generator = base._phase_groups(model)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    _assert_optical_optimizer(model, parameters)
    _require(torch.cuda.is_available(), "optical update preflight requires AutoDL CUDA")
    mapper_ids = {id(parameter) for parameter in mapper.parameters()}
    _require(not any(id(parameter) in mapper_ids for parameter in parameters),
             "mapper-only training path was incorrectly placed in the optical optimizer")
    mapper_only_rejected = False
    try:
        _assert_optical_optimizer(model, list(mapper.parameters()))
    except RuntimeError:
        mapper_only_rejected = True
    _require(mapper_only_rejected,
             "negative control failed to reject a mapper-only optimizer")
    wrong_target = _verify_wrong_target_negative(
        model, runtime["pixels"], schedule[0], runtime, source
    )
    cmos_gradients = _verify_cmos_gradients(
        model, runtime["pixels"], schedule[0], runtime, source
    )
    directions = [
        joint._joint_direction_check(model, runtime["pixels"], schedule[0], device,
                                     source, name)
        for name in ("encoder", "generator")
    ]
    initial_phases = [parameter.detach().clone() for parameter in (encoder, generator)]
    source_steps = _optimizer_phase_steps(optimizer, encoder, generator)
    recovery_input = output_dir / "preflight_recovery_input.pt"
    recovery_output = output_dir / "preflight_recovery_worker_result.pt"
    prefix_reused = recovery_input.is_file()
    update_losses = []
    if prefix_reused:
        prefix = torch.load(recovery_input, map_location="cpu", weights_only=False)
        steps_after_three = _validate_preflight_prefix(
            prefix, selected, runtime, schedule
        )
    else:
        for index in range(3):
            update_losses.append(joint._joint_update(
                model, optimizer, runtime["pixels"], schedule[index], device, source
            ))
        steps_after_three = _optimizer_phase_steps(optimizer, encoder, generator)
        _require(steps_after_three == (source_steps[0] + 3, source_steps[1] + 3)
                 and all(not torch.equal(old, new.detach())
                         for old, new in zip(initial_phases, (encoder, generator), strict=True)),
                 "three real preflight updates did not advance both optical phase groups")
        prefix = {
            "task_id": TASK_ID,
            "completed_updates": 3,
            "model": base._state_snapshot(model),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "source_checkpoint": cpu_copy(selected["checkpoint"]),
            "schedule": schedule.clone(),
            "cpu_rng": torch.random.get_rng_state().cpu().clone(),
            "cuda_rng": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()],
        }
        _save_torch(recovery_input, prefix, new=True)
    _assert_frozen_unchanged(snapshot, runtime, mapper)
    direct_model, direct_optimizer = _optimizer_and_model(
        selected["checkpoint"], runtime, selected["optimizer_steps"]
    )
    direct_model.load_state_dict(prefix["model"], strict=True)
    direct_optimizer.load_state_dict(prefix["optimizer"])
    torch.random.set_rng_state(prefix["cpu_rng"])
    torch.cuda.set_rng_state_all(prefix["cuda_rng"])
    direct_loss = joint._joint_update(
        direct_model, direct_optimizer, runtime["pixels"], schedule[3], device, source
    )
    torch.cuda.synchronize(device)
    recovered = _run_recovery_worker(DEFAULT_CONFIG, recovery_input, recovery_output)
    direct_model_state = base._state_snapshot(direct_model)
    direct_optimizer_state = cpu_copy(direct_optimizer.state_dict())
    model_delta = _tree_max_delta(direct_model_state, recovered["model"])
    optimizer_delta = _tree_max_delta(direct_optimizer_state, recovered["optimizer"])
    _require(_states_close(direct_model_state, recovered["model"])
             and _states_close(direct_optimizer_state, recovered["optimizer"])
             and _states_close(direct_loss, recovered["loss"])
             and torch.equal(schedule[3], recovered["rows"]),
             "independent-process recovery differs from the uninterrupted fourth update")
    _assert_frozen_unchanged(snapshot, runtime, mapper)
    result = {
        "status": "READY",
        "selected_endpoint": selected["name"],
        "selected_checkpoint": str(selected["path"]),
        "selected_global_step": selected["checkpoint"]["completed_global_step"],
        "initial_optimizer_steps": list(source_steps),
        "optimizer_steps_after_three": list(steps_after_three),
        "engineering_update_calls": 5 + int(prefix_reused),
        "three_step_prefix_reused_from_failed_preflight": prefix_reused,
        "scientific_updates": 0,
        "cmos_only_phase_gradient_norms": cmos_gradients,
        "joint_loss_direction_checks": directions,
        "wrong_target_negative_control": wrong_target,
        "mapper_only_optimizer_rejected": True,
        "recovery": {
            "passed": True,
            "direct_vs_recovered_model_max_abs": model_delta,
            "direct_vs_recovered_optimizer_max_abs": optimizer_delta,
            "direct_fourth_update_loss": direct_loss,
        },
        "three_real_update_observations": update_losses or "validated_saved_three_step_prefix",
        "frozen_adapter_projection_prior_camera_and_evaluators_unchanged": True,
        "preflight_rolled_back": True,
        "device": torch.cuda.get_device_name(device),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        "official_test_reads": 0,
    }
    _write_json(output_dir / "training_preflight.json", result)
    return result


def _make_ready_control(config: Mapping[str, Any], selected: Mapping[str, Any],
                        runtime: Mapping[str, Any], replay: Mapping[str, Any],
                        positive: Mapping[str, Any],
                        training_preflight: Mapping[str, Any] | None
                        ) -> tuple[ResearchRunControl, PreflightReceipt, ResearchRunIdentity]:
    checkpoint = selected["checkpoint"]
    device = runtime["device"]
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=uuid.uuid4().hex,
        trajectory_id=f"{TASK_ID}:selected:{selected['name']}:updates:{TRAIN_UPDATES}",
        semantic_authority_id=AUTHORITY,
        data_role="MNIST_TRAIN_55000_DEV_5000_NEW_PRIOR_DEV200_FINAL2000_OFFICIAL_TEST_UNREAD",
        config_identity=config,
        model_identity=checkpoint["model_identity"],
        optimizer_identity={
            "name": "Adam",
            "learning_rate": checkpoint["optimizer"]["param_groups"][0]["lr"],
            "betas": checkpoint["optimizer"]["param_groups"][0]["betas"],
            "epsilon": checkpoint["optimizer"]["param_groups"][0]["eps"],
            "weight_decay": checkpoint["optimizer"]["param_groups"][0]["weight_decay"],
            "initial_encoder_step": selected["optimizer_steps"][0],
            "initial_generator_step": selected["optimizer_steps"][1],
            "trainable_groups": ["encoder_phase", "generator_phase"],
        },
        sampler_identity={
            "development_seed_base": 202610300910,
            "final_seed_base": 202610300930,
            "schedule_seed": 202610300950,
            "maximum_new_updates": TRAIN_UPDATES,
            "batch_size": BATCH_SIZE,
        },
        execution_source_semantics_id=config["execution_source_semantics_id"],
    )
    control = ResearchRunControl(
        identity,
        authorization_id=f"{TASK_ID}:selected:{selected['name']}:single-final-decision",
    )
    checks = [
        "four_explicit_C45k_D0_Dgrad_J_checkpoint_and_Adam_phase_identities",
        "latest_fixed_frame_mapper_and_alpha1_raw_physical_phase_prior",
        "fixed_response_projector_identity_and_saved_command_replay",
        "new_development_population_original_modal_power_rejection_only",
        "same_saved_legal_SLM_commands_for_all_old_optical_endpoints",
        "2048_production_full_SLM_to_raw_CMOS_path",
        "4096_independent_reference_two_saved_solver_positives_per_class",
        "TRAIN_55000_DEV_5000_roles_and_official_TEST_unread",
    ]
    if training_preflight is not None:
        checks.extend([
            "three_real_BOGT_plus_SSIM_updates_through_production_image_to_CMOS_entrypoint",
            "CMOS_reconstruction_and_SSIM_gradients_reach_both_optical_phase_groups",
            "mapper_only_optimizer_negative_control_rejected",
            "wrong_target_negative_control_detected",
            "independent_process_fourth_update_recovery_and_rollback",
            "frozen_adapter_projection_prior_camera_and_evaluators_isolated",
        ])
    receipt = control.make_ready_receipt(
        correctness={
            "passed": True, "status": "PASSED", "checks": checks,
            "recovery_passed": training_preflight is None
            or training_preflight["recovery"]["passed"],
            "all_declared_trainable_groups_updated": training_preflight is None
            or (
                training_preflight["optimizer_steps_after_three"][0]
                > training_preflight["initial_optimizer_steps"][0]
                and training_preflight["optimizer_steps_after_three"][1]
                > training_preflight["initial_optimizer_steps"][1]
            ),
        },
        resource={
            "passed": True, "status": "PASSED",
            "device": torch.cuda.get_device_name(device),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        implementation={
            "passed": True, "status": "PASSED",
            "engine": "cuda_gaussian_s12_grid_2048",
            "selected_engine": "cuda_gaussian_s12_grid_2048",
        },
        selected_engine="cuda_gaussian_s12_grid_2048",
        resource_envelope={
            "device": torch.cuda.get_device_name(device),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    )
    control.mark_preflight_ready(receipt)
    return control, receipt, identity


def _checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        completed: int, selected: Mapping[str, Any],
                        schedule: torch.Tensor, config: Mapping[str, Any],
                        run_id: str, history: Sequence[Mapping[str, Any]],
                        training_elapsed_seconds: float = 0.0) -> dict[str, Any]:
    encoder, generator = base._phase_groups(model)
    steps = _optimizer_phase_steps(optimizer, encoder, generator)
    return {
        "schema": f"{TASK_ID}.optical_checkpoint.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "selected_endpoint": selected["name"],
        "source_checkpoint": str(selected["path"]),
        "source_run_id": selected["checkpoint"]["run_id"],
        "source_global_step": selected["checkpoint"]["completed_global_step"],
        "completed_updates": completed,
        "completed_global_step": selected["checkpoint"]["completed_global_step"] + completed,
        "trainable_groups": ["encoder_phase", "generator_phase"],
        "optimizer_steps": {
            "encoder_adam_step": steps[0],
            "generator_adam_step": steps[1],
        },
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "schedule": schedule.clone(),
        "schedule_seed": config["training"]["schedule_seed"],
        "history": copy.deepcopy(list(history)),
        "history_count": len(history),
        "training_elapsed_seconds": training_elapsed_seconds,
        "config_identity": config,
        "weights": copy.deepcopy(selected["checkpoint"]["weights"]),
        "structural_weight": selected["checkpoint"]["structural_weight"],
        "scale": selected["checkpoint"]["scale"].clone(),
        "power_reference": selected["checkpoint"]["power_reference"].clone(),
        "camera_gain": selected["checkpoint"]["camera_gain"].clone(),
        "model_identity": selected["checkpoint"]["model_identity"],
        "rng_state": {
            "cpu": torch.random.get_rng_state().cpu().clone(),
            "cuda": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()],
        },
    }


def _validate_training_checkpoint(checkpoint: Mapping[str, Any], selected: Mapping[str, Any],
                                 schedule: torch.Tensor, config: Mapping[str, Any],
                                 run_id: str) -> int:
    completed = checkpoint.get("completed_updates")
    source = selected["checkpoint"]
    _require(checkpoint.get("schema") == f"{TASK_ID}.optical_checkpoint.v1"
             and checkpoint.get("task_id") == TASK_ID
             and checkpoint.get("run_id") == run_id
             and checkpoint.get("selected_endpoint") == selected["name"]
             and checkpoint.get("source_checkpoint") == str(selected["path"])
             and checkpoint.get("source_run_id") == source["run_id"]
             and checkpoint.get("source_global_step") == source["completed_global_step"]
             and checkpoint.get("trainable_groups") == ["encoder_phase", "generator_phase"]
             and checkpoint.get("model_identity") == source["model_identity"]
             and checkpoint.get("config_identity") == config
             and checkpoint.get("schedule_seed") == config["training"]["schedule_seed"]
             and torch.equal(checkpoint.get("schedule"), schedule),
             "resume checkpoint task, source, config, model, or sample schedule changed")
    _require(type(completed) is int and completed in CHECKPOINT_STEPS
             and checkpoint.get("completed_global_step")
             == source["completed_global_step"] + completed
             and checkpoint.get("history_count") == completed
             and isinstance(checkpoint.get("history"), list)
             and len(checkpoint["history"]) == completed,
             "resume checkpoint does not represent a complete authorized save point")
    _require(all(
        item.get("completed_update") == index + 1
        and item.get("sample_rows") == [int(value) for value in schedule[index].tolist()]
        and item.get("labels") == list(range(CLASS_COUNT))
        for index, item in enumerate(checkpoint["history"])
    ), "resume checkpoint history does not match the registered sample schedule")
    expected_steps = {
        "encoder_adam_step": selected["optimizer_steps"][0] + completed,
        "generator_adam_step": selected["optimizer_steps"][1] + completed,
    }
    _require(checkpoint.get("optimizer_steps") == expected_steps,
             "resume checkpoint Encoder/Generator Adam counters changed")
    _require(set(checkpoint.get("model", {})) == set(source["model"])
             and all(torch.equal(value, checkpoint["model"][key])
                     for key, value in source["model"].items()
                     if key not in {"runtime.encoder.phase", "runtime.generator.phase"}),
             "resume checkpoint changed a frozen model component")
    for key in ("weights", "structural_weight", "scale", "power_reference", "camera_gain"):
        _require(_states_close(checkpoint.get(key), source.get(key)),
                 f"resume checkpoint changed the fixed C/J reference {key}")
    rng = checkpoint.get("rng_state")
    _require(isinstance(rng, Mapping) and torch.is_tensor(rng.get("cpu"))
             and isinstance(rng.get("cuda"), list)
             and all(torch.is_tensor(value) for value in rng["cuda"]),
             "resume checkpoint lacks complete CPU/CUDA RNG state")
    return completed


def _save_training_checkpoint(path: Path, payload: Mapping[str, Any],
                              completed: int, run_id: str) -> None:
    _save_torch(path, payload, new=True)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id
             and saved.get("completed_updates") == completed
             and saved.get("history_count") == completed
             and saved.get("optimizer_steps") == payload.get("optimizer_steps")
             and _states_close(saved.get("model"), payload.get("model"))
             and _states_close(saved.get("optimizer"), payload.get("optimizer"))
             and _states_close(saved.get("rng_state"), payload.get("rng_state"))
             and _states_close(saved.get("history"), payload.get("history")),
             f"checkpoint readback failed at update {completed}")


def _train_optics(config: Mapping[str, Any], selected: Mapping[str, Any],
                  runtime: Mapping[str, Any], schedule: torch.Tensor,
                  run_id: str, control: ResearchRunControl,
                  receipt: PreflightReceipt,
                  resume_checkpoint_path: Path | None = None
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    model, optimizer = _optimizer_and_model(
        selected["checkpoint"], runtime, selected["optimizer_steps"]
    )
    source = {"checkpoint": selected["checkpoint"],
              "training_labels": runtime["labels"],
              "training_indices": runtime["train_indices"],
              "training_pixels": runtime["pixels"],
              "training_witness_dir": ATTEMPT_DIR / "training_path_witnesses",
              "training_capture_updates": sorted({1, *CHECKPOINT_STEPS[1:]})}
    history: list[dict[str, Any]] = []
    start_update = 0
    elapsed_base = 0.0
    if resume_checkpoint_path is None:
        control.open_scientific_attempt(ATTEMPT_DIR, f"{run_id}:attempt_01", receipt)
        _save_torch(ATTEMPT_DIR / "training_schedule.pt", schedule.clone(), new=True)
        initial_path = ATTEMPT_DIR / "checkpoint_step_00000.pt"
        initial = _checkpoint_payload(model, optimizer, 0, selected, schedule,
                                      config, run_id, [])
        _save_training_checkpoint(initial_path, initial, 0, run_id)
        _write_json(ATTEMPT_DIR / "task_state.json", {
            "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
            "completed_updates": 0, "authorized_updates": TRAIN_UPDATES,
            "selected_endpoint": selected["name"],
            "checkpoint_path": str(initial_path),
            "last_complete_save_unix": time.time(),
        })
        last_checkpoint_path = initial_path
    else:
        checkpoint_path = resume_checkpoint_path.resolve()
        _require(checkpoint_path.is_file()
                 and checkpoint_path.parent.resolve() == ATTEMPT_DIR.resolve(),
                 "resume requires an explicit checkpoint inside this scientific attempt")
        expected_schedule_path = ATTEMPT_DIR / "training_schedule.pt"
        saved_schedule = torch.load(expected_schedule_path, map_location="cpu", weights_only=False)
        _require(torch.equal(saved_schedule, schedule),
                 "saved recovery schedule differs from the registered 2,000-step schedule")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_update = _validate_training_checkpoint(checkpoint, selected, schedule, config, run_id)
        _require(checkpoint_path.name == f"checkpoint_step_{start_update:05d}.pt"
                 or (start_update == TRAIN_UPDATES
                     and checkpoint_path.name == "terminal_checkpoint.pt"),
                 "resume checkpoint filename does not match its complete update count")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        encoder, generator = base._phase_groups(model)
        _require(_optimizer_phase_steps(optimizer, encoder, generator)
                 == (selected["optimizer_steps"][0] + start_update,
                     selected["optimizer_steps"][1] + start_update),
                 "restored optimizer state differs from the checkpoint update count")
        history = copy.deepcopy(checkpoint["history"])
        rng = checkpoint["rng_state"]
        torch.random.set_rng_state(rng["cpu"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        elapsed_base = float(checkpoint.get("training_elapsed_seconds", 0.0))
        _require(math.isfinite(elapsed_base) and elapsed_base >= 0.0,
                 "resume checkpoint accumulated training duration is invalid")
        last_checkpoint_path = checkpoint_path
        control.resume_scientific_attempt(
            ATTEMPT_DIR, f"{run_id}:attempt_01", receipt
        )
        _write_json(ATTEMPT_DIR / "task_state.json", {
            "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
            "completed_updates": start_update, "authorized_updates": TRAIN_UPDATES,
            "selected_endpoint": selected["name"],
            "checkpoint_path": str(checkpoint_path),
            "last_complete_save_unix": time.time(),
        })
    source["training_capture_updates"] = sorted({
        1, start_update + 1, *CHECKPOINT_STEPS[1:],
    })
    started = time.perf_counter()
    for index in range(start_update, TRAIN_UPDATES):
        source["active_training_update"] = index + 1
        loss = joint._joint_update(
            model, optimizer, runtime["pixels"], schedule[index],
            runtime["device"], source,
        )
        history.append({
            "completed_update": index + 1,
            "sample_rows": [int(value) for value in schedule[index].tolist()],
            "labels": list(range(CLASS_COUNT)),
            "loss_is_pre_update": True,
            **loss,
        })
        completed = index + 1
        if completed in CHECKPOINT_STEPS[1:]:
            torch.cuda.synchronize(runtime["device"])
            checkpoint_path = ATTEMPT_DIR / f"checkpoint_step_{completed:05d}.pt"
            checkpoint_payload = _checkpoint_payload(
                model, optimizer, completed, selected, schedule, config, run_id,
                history, elapsed_base + time.perf_counter() - started,
            )
            _save_training_checkpoint(checkpoint_path, checkpoint_payload, completed, run_id)
            last_checkpoint_path = checkpoint_path
            encoder, generator = base._phase_groups(model)
            steps = _optimizer_phase_steps(optimizer, encoder, generator)
            _write_json(ATTEMPT_DIR / "task_state.json", {
                "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
                "completed_updates": completed, "authorized_updates": TRAIN_UPDATES,
                "selected_endpoint": selected["name"],
                "checkpoint_path": str(checkpoint_path),
                "last_complete_save_unix": time.time(),
                "optimizer_steps": {"encoder": steps[0], "generator": steps[1]},
            })
        if completed % 100 == 0:
            _write_json(ATTEMPT_DIR / "progress.json", {
                "task_id": TASK_ID, "run_id": run_id,
                "completed_updates": completed, "authorized_updates": TRAIN_UPDATES,
                "checkpoint_path": str(last_checkpoint_path),
                "last_complete_save_unix": time.time(),
                "updated_at_unix": time.time(),
            })
    encoder, generator = base._phase_groups(model)
    final_steps = _optimizer_phase_steps(optimizer, encoder, generator)
    _require(len(history) == TRAIN_UPDATES
             and final_steps == (selected["optimizer_steps"][0] + TRAIN_UPDATES,
                                 selected["optimizer_steps"][1] + TRAIN_UPDATES),
             "training did not finish exactly 2,000 joint optical updates")
    model.eval()
    training_elapsed = elapsed_base + time.perf_counter() - started
    terminal = _checkpoint_payload(
        model, optimizer, TRAIN_UPDATES, selected, schedule, config, run_id,
        history, training_elapsed,
    )
    terminal_path = ATTEMPT_DIR / "terminal_checkpoint.pt"
    if terminal_path.is_file():
        existing_terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(_validate_training_checkpoint(existing_terminal, selected, schedule,
                                               config, run_id) == TRAIN_UPDATES
                 and _states_close(existing_terminal["model"], terminal["model"])
                 and _states_close(existing_terminal["optimizer"], terminal["optimizer"])
                 and _states_close(existing_terminal["rng_state"], terminal["rng_state"])
                 and _states_close(existing_terminal["history"], terminal["history"]),
                 "existing terminal checkpoint differs from the restored 2,000-step state")
        terminal = existing_terminal
        training_elapsed = float(terminal["training_elapsed_seconds"])
    else:
        _save_training_checkpoint(terminal_path, terminal, TRAIN_UPDATES, run_id)
    _write_json(ATTEMPT_DIR / "task_state.json", {
        "task_id": TASK_ID, "run_id": run_id, "state": "TERMINAL_MEASUREMENT_PENDING",
        "completed_updates": TRAIN_UPDATES, "authorized_updates": TRAIN_UPDATES,
        "selected_endpoint": selected["name"],
        "checkpoint_path": str(terminal_path),
        "last_complete_save_unix": time.time(),
        "training_elapsed_seconds": training_elapsed,
        "optimizer_steps": terminal["optimizer_steps"],
    })
    summary = {
        "task_id": TASK_ID,
        "selected_endpoint": selected["name"],
        "source_global_step": selected["checkpoint"]["completed_global_step"],
        "completed_updates": TRAIN_UPDATES,
        "completed_global_step": terminal["completed_global_step"],
        "optimizer_steps": terminal["optimizer_steps"],
        "training_elapsed_seconds": training_elapsed,
        "training_loss_start": history[0],
        "training_loss_terminal": history[-1],
        "loss_contract": "inherited C/J BOGT plus fixed SSIM; no classifier or coefficient-fit term",
        "full_training_forward": "real image to amplitude SLM to Encoder to identity OLS to Generator to raw CMOS",
        "lineage": "project extension; not a pure reproduction of the paper",
    }
    _write_json(ATTEMPT_DIR / "training_summary.json", summary)
    return terminal, {"training_updates": TRAIN_UPDATES, "history": history}


def _validate_cohort(cohort: Mapping[str, Any], expected_per_class: int) -> None:
    total = CLASS_COUNT * expected_per_class
    labels = cohort.get("labels")
    coefficients = cohort.get("coefficients")
    proposals = cohort.get("proposal_ids")
    _require(torch.is_tensor(labels) and labels.dtype == torch.long and labels.shape == (total,)
             and torch.is_tensor(coefficients) and coefficients.shape == (total, PORT_COUNT)
             and coefficients.is_complex() and torch.isfinite(coefficients.real).all().item()
             and torch.isfinite(coefficients.imag).all().item()
             and torch.is_tensor(proposals) and proposals.shape == (total,),
             "prior cohort is incomplete or malformed")
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT),
                         torch.full((CLASS_COUNT,), expected_per_class))
             and torch.equal(labels, torch.arange(CLASS_COUNT).repeat_interleave(expected_per_class))
             and torch.all(coefficients.abs().square().sum(1) <= pilot.base.MAX_MODAL_POWER_W),
             "prior cohort class roles or original power rejection changed")


def _paired_class_bootstrap(labels: torch.Tensor, baseline: torch.Tensor,
                            candidate: torch.Tensor, seed: int,
                            resamples: int) -> dict[str, Any]:
    labels = labels.detach().cpu().long()
    baseline = baseline.detach().cpu().long()
    candidate = candidate.detach().cpu().long()
    _require(labels.shape == baseline.shape == candidate.shape
             and torch.equal(torch.bincount(labels, minlength=CLASS_COUNT),
                             torch.full((CLASS_COUNT,), 200)),
             "paired class bootstrap requires 200 final rows in every class")
    base_correct = baseline == labels
    candidate_correct = candidate == labels
    rng = np.random.default_rng(seed)
    differences = np.zeros(resamples, dtype=np.float64)
    point, per_class, draws_by_class = [], {}, []
    for label in range(CLASS_COUNT):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten().numpy()
        base_values = base_correct[indices].numpy().astype(np.float64)
        candidate_values = candidate_correct[indices].numpy().astype(np.float64)
        delta = float(candidate_values.mean() - base_values.mean())
        point.append(delta)
        per_class[str(label)] = {
            "baseline_recall": float(base_values.mean()),
            "candidate_recall": float(candidate_values.mean()),
            "delta": delta,
        }
        draws_by_class.append(rng.integers(0, len(indices), size=(resamples, len(indices))))
    for label in range(CLASS_COUNT):
        draws = draws_by_class[label]
        indices = torch.nonzero(labels == label, as_tuple=False).flatten().numpy()
        base_values = base_correct[indices].numpy().astype(np.float64)
        candidate_values = candidate_correct[indices].numpy().astype(np.float64)
        differences += (
            candidate_values[draws].mean(axis=1) - base_values[draws].mean(axis=1)
        ) / CLASS_COUNT
    lower, upper = np.quantile(differences, [0.025, 0.975], method="linear")
    return {
        "point_macro_recall_delta": float(np.mean(point)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "paired_class_stratified_ci95": [float(lower), float(upper)],
        "per_class": per_class,
        "lower_bound_positive": bool(lower > 0.0),
    }


def _final_gate(labels: torch.Tensor, baseline: Mapping[str, Any],
                candidate: Mapping[str, Any], config: Mapping[str, Any]
                ) -> dict[str, Any]:
    measurement = config["measurement"]
    base_eval = baseline["semantic_summary"]["conditions"]["direct_decoder"]["evaluators"]
    cand_eval = candidate["semantic_summary"]["conditions"]["direct_decoder"]["evaluators"]
    ed_ci = _paired_class_bootstrap(
        labels, torch.tensor(base_eval["ED"]["predicted_labels"]),
        torch.tensor(cand_eval["ED"]["predicted_labels"]),
        int(measurement["bootstrap_seed"]), int(measurement["bootstrap_resamples"]),
    )
    ec_delta = float(cand_eval["EC"]["macro_recall"] - base_eval["EC"]["macro_recall"])
    base_div = baseline["semantic_summary"]["same_class_diversity"]["direct_decoder"]["by_class"]
    cand_div = candidate["semantic_summary"]["same_class_diversity"]["direct_decoder"]["by_class"]
    ratios = {
        str(label): float(cand_div[str(label)]["mean_pairwise_normalized_l2"])
        / max(float(base_div[str(label)]["mean_pairwise_normalized_l2"]), 1.0e-30)
        for label in range(CLASS_COUNT)
    }
    ed_pass = ed_ci["paired_class_stratified_ci95"][0] > float(
        measurement["final_ed_paired_macro_recall_lower_bound_min"]
    )
    ec_pass = ec_delta >= -float(measurement["final_ec_macro_recall_drop_max"])
    diversity_pass = all(
        value >= float(measurement["final_class_diversity_retention_min"])
        for value in ratios.values()
    )
    return {
        "scientific_question": (
            "Does the reused adapter and selected optical endpoint yield a better numerical "
            "semantic-generation candidate through the complete SLM path?"
        ),
        "baseline": "C45k with the same frozen mapper and projection",
        "candidate": candidate["condition"],
        "E_D_paired_macro_recall": ed_ci,
        "E_C_macro_recall_delta": ec_delta,
        "E_C_loss_limit": -float(measurement["final_ec_macro_recall_drop_max"]),
        "per_class_diversity_retention": ratios,
        "class_diversity_minimum": float(measurement["final_class_diversity_retention_min"]),
        "criteria": {
            "E_D_paired_bootstrap_lower_bound_gt_zero": ed_pass,
            "E_C_macro_recall_drop_at_most_2pp": ec_pass,
            "all_classes_retain_at_least_80pct_diversity": diversity_pass,
        },
        "passed": bool(ed_pass and ec_pass and diversity_pass),
        "human_readability": "not_measured",
        "full_G1_distribution": "not_established",
        "hardware_deployment": "not_measured",
    }


def _nearest_same_class_training_rows(readouts: torch.Tensor, labels: torch.Tensor,
                                     pixels: torch.Tensor, train_indices: torch.Tensor,
                                     data_labels: torch.Tensor, device: torch.device
                                     ) -> list[dict[str, Any]]:
    images = readouts.detach().cpu().float().flatten(1)
    labels = labels.detach().cpu().long()
    output: list[dict[str, Any] | None] = [None] * len(labels)
    for label in range(CLASS_COUNT):
        generated_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        train_rows = train_indices[data_labels[train_indices] == label]
        train_images = pixels.index_select(0, train_rows).reshape(-1, 784).to(
            device=device, dtype=torch.float32
        ) / 255.0
        train_norms = train_images.square().sum(dim=1)
        for start in range(0, generated_indices.numel(), 32):
            indices = generated_indices[start:start + 32]
            generated = images.index_select(0, indices).to(device)
            distance_sq = (
                generated.square().sum(1, keepdim=True) + train_norms.unsqueeze(0)
                - 2.0 * generated @ train_images.T
            ).clamp_min(0.0)
            minima, positions = distance_sq.min(dim=1)
            for offset, row_index in enumerate(indices.tolist()):
                train_position = int(positions[offset])
                output[row_index] = {
                    "population_index": row_index,
                    "label": label,
                    "nearest_same_class_train_row": int(train_rows[train_position]),
                    "nearest_same_class_l2_over_sqrt784": float(minima[offset].sqrt().cpu() / 28.0),
                }
        del train_images, train_norms
    _require(all(item is not None for item in output),
             "training-neighbor diagnostic did not consume every generation row")
    return [item for item in output if item is not None]


def _save_contact_sheet(path: Path, labels: torch.Tensor,
                        conditions: Mapping[str, Mapping[str, Any]]) -> None:
    names = list(conditions)
    examples_per_class = 10
    cell = 30
    margin = 12
    header = 34
    columns = len(names) * examples_per_class
    image = Image.new("RGB", (margin * 2 + columns * cell, header + margin + 10 * cell), "white")
    draw = ImageDraw.Draw(image)
    for condition_index, name in enumerate(names):
        draw.text((margin + condition_index * examples_per_class * cell, 8), name, fill="black")
    for label in range(CLASS_COUNT):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()[:examples_per_class]
        draw.text((2, header + margin + label * cell + 8), str(label), fill="black")
        for condition_index, name in enumerate(names):
            readouts = conditions[name]["readouts"]
            for local, row in enumerate(rows.tolist()):
                values = (readouts[row, 0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                crop = Image.fromarray(values, mode="L").resize((cell - 2, cell - 2))
                x = margin + (condition_index * examples_per_class + local) * cell
                y = header + margin + label * cell
                image.paste(crop.convert("RGB"), (x + 1, y + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_sample_csv(path: Path, cohort: Mapping[str, Any],
                      conditions: Mapping[str, Mapping[str, Any]],
                      neighbors: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    names = list(conditions)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["population_index", "label", "proposal_id"]
        for name in names:
            header.extend([
                f"{name}_E_D_prediction", f"{name}_E_C_prediction",
                f"{name}_nearest_same_class_train_row",
                f"{name}_nearest_same_class_l2_over_sqrt784",
            ])
        writer.writerow(header)
        for index, label in enumerate(cohort["labels"].tolist()):
            values: list[Any] = [index, label, int(cohort["proposal_ids"][index])]
            for name in names:
                evaluators = conditions[name]["semantic_summary"]["conditions"][
                    "direct_decoder"
                ]["evaluators"]
                values.extend([
                    evaluators["ED"]["predicted_labels"][index],
                    evaluators["EC"]["predicted_labels"][index],
                    neighbors[name][index]["nearest_same_class_train_row"],
                    neighbors[name][index]["nearest_same_class_l2_over_sqrt784"],
                ])
            writer.writerow(values)


def _consume_final_measurements(config: Mapping[str, Any], runtime: Mapping[str, Any],
                                output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    population_path = output_dir / "final_population_and_commands.pt"
    _require(population_path.is_file(), "saved final population is missing", FileNotFoundError)
    population = torch.load(population_path, map_location="cpu", weights_only=False)
    _require(population.get("schema") == f"{TASK_ID}.final_population.v1"
             and population.get("task_id") == TASK_ID
             and population.get("seed_base") == 202610300930,
             "saved final population identity changed")
    cohort = population.get("cohort")
    commands = population.get("commands")
    _require(isinstance(cohort, Mapping), "saved final cohort is missing")
    _validate_cohort(cohort, 200)
    _assert_legal_commands(commands)
    _require(commands.shape[0] == 2_000
             and torch.equal(cohort["labels"], torch.arange(CLASS_COUNT).repeat_interleave(200))
             and cohort.get("role")
             == "new_final_population_not_consumed_during_route_or_training",
             "final consumer rejects an incomplete or misaligned population")

    registry_path = output_dir / "final_condition_registry.json"
    _require(registry_path.is_file(), "saved final condition registry is missing",
             FileNotFoundError)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    condition_registry = registry.get("conditions")
    required_conditions = {"C45k", "selected_initial", "final_candidate"}
    _require(registry.get("task_id") == TASK_ID
             and isinstance(condition_registry, Mapping)
             and set(condition_registry) == required_conditions,
             "saved final condition registry is incomplete")
    reconsumed: dict[str, Any] = {}
    for name in ("C45k", "selected_initial", "final_candidate"):
        record = condition_registry[name]
        artifact_name = record.get("artifact")
        _require(isinstance(artifact_name, str)
                 and Path(artifact_name).name == artifact_name
                 and artifact_name.startswith("final_")
                 and artifact_name.endswith(".pt"),
                 f"final {name} artifact path is invalid")
        artifact_path = output_dir / artifact_name
        _require(artifact_path.is_file(), f"saved final {name} outputs are missing",
                 FileNotFoundError)
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        _require(artifact.get("task_id") == TASK_ID
                 and artifact.get("artifact_condition") == Path(artifact_name).stem.removeprefix("final_")
                 and torch.equal(artifact.get("labels"), cohort["labels"])
                 and torch.equal(artifact.get("proposal_ids"), cohort["proposal_ids"])
                 and torch.equal(artifact.get("commands"), commands),
                 f"saved final {name} artifact is not bound to the shared population and commands")
        checkpoint_path_value = record.get("checkpoint_path")
        _require(isinstance(checkpoint_path_value, str) and checkpoint_path_value,
                 f"saved final {name} checkpoint path is missing")
        checkpoint_path = _path(config, checkpoint_path_value)
        artifact_checkpoint_path_value = artifact.get("checkpoint_path")
        _require(isinstance(artifact_checkpoint_path_value, str)
                 and artifact_checkpoint_path_value,
                 f"saved final {name} output lacks its checkpoint binding")
        _require(checkpoint_path.is_file(), f"saved final {name} checkpoint is missing",
                 FileNotFoundError)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(checkpoint.get("run_id") == artifact.get("checkpoint_run_id")
                 and checkpoint.get("completed_global_step") == artifact.get("checkpoint_step")
                 and record.get("checkpoint_run_id") == checkpoint.get("run_id")
                 and record.get("checkpoint_step") == checkpoint.get("completed_global_step")
                 and _path(config, artifact_checkpoint_path_value) == checkpoint_path,
                 f"saved final {name} checkpoint identity differs from its output artifact")
        produced = {
            key: artifact.get(key)
            for key in ("raw_cmos", "coefficients", "readouts")
        }
        _require(all(torch.is_tensor(value) and value.shape[0] == 2_000
                     and torch.isfinite(value).all().item() for value in produced.values()),
                 f"saved final {name} optical outputs are incomplete or non-finite")
        measured = _measure_commands(checkpoint, commands, runtime)
        _require(all(measured[key].shape == produced[key].shape
                     for key in ("raw_cmos", "coefficients", "readouts")),
                 f"saved final {name} output tensor shapes changed on replay")
        max_cmos_error = float((measured["raw_cmos"] - produced["raw_cmos"]).abs().max())
        max_coeff_error = float((measured["coefficients"] - produced["coefficients"]).abs().max())
        max_readout_error = float((measured["readouts"] - produced["readouts"]).abs().max())
        _require(max_cmos_error <= 1.0e-12 and max_coeff_error <= 1.0e-12
                 and max_readout_error <= 1.0e-12,
                 f"saved {name} full-path output failed independent production replay")
        summary = frame._summary_from_readouts(
            runtime["input_bundle"], cohort, measured["readouts"]
        )
        reconsumed[name] = {
            "condition": name,
            "checkpoint_run_id": checkpoint["run_id"],
            "checkpoint_step": checkpoint["completed_global_step"],
            "commands": commands.clone(),
            "raw_cmos": measured["raw_cmos"],
            "coefficients": measured["coefficients"],
            "readouts": measured["readouts"],
            "semantic_summary": summary,
            "producer_replay_max_abs_cmos": max_cmos_error,
            "producer_replay_max_abs_coefficients": max_coeff_error,
            "producer_replay_max_abs_readout": max_readout_error,
            "artifact": artifact_name,
        }
    gate = _final_gate(cohort["labels"], reconsumed["C45k"],
                       reconsumed["final_candidate"], config)
    result = {
        "schema": f"{TASK_ID}.final_reconsumption.v1",
        "task_id": TASK_ID,
        "population_role": cohort["role"],
        "population_rows": 2_000,
        "rows_per_class": {str(label): 200 for label in range(CLASS_COUNT)},
        "same_saved_slm_commands_for_all_conditions": True,
        "replayed_full_production_path_from_saved_commands": True,
        "conditions": {
            name: {
                "checkpoint_run_id": item["checkpoint_run_id"],
                "checkpoint_step": item["checkpoint_step"],
                "max_abs_cmos_replay_error": item["producer_replay_max_abs_cmos"],
                "max_abs_coefficients_replay_error": item["producer_replay_max_abs_coefficients"],
                "max_abs_readout_replay_error": item["producer_replay_max_abs_readout"],
                "ED": item["semantic_summary"]["conditions"]["direct_decoder"]["evaluators"]["ED"],
                "EC": item["semantic_summary"]["conditions"]["direct_decoder"]["evaluators"]["EC"],
                "diversity": item["semantic_summary"]["same_class_diversity"]["direct_decoder"],
            }
            for name, item in reconsumed.items()
        },
        "final_gate": gate,
    }
    _write_json(output_dir / "final_reconsumption.json", result)
    _save_torch(output_dir / "final_reconsumed_outputs.pt", {
        "task_id": TASK_ID,
        "conditions": {
            name: {key: item[key] for key in ("raw_cmos", "coefficients", "readouts")}
            for name, item in reconsumed.items()
        },
    })
    return result, reconsumed


def _save_source_receipt(config: Mapping[str, Any], output_dir: Path,
                         endpoints: Mapping[str, Mapping[str, Any]],
                         runtime: Mapping[str, Any]) -> None:
    result = {
        "task_id": TASK_ID,
        "sources": {
            name: {
                "endpoint_label": name,
                "path": str(value["path"]),
                "run_id": value["checkpoint"]["run_id"],
                "source_run_id": value["checkpoint"].get("source_run_id"),
                "task_id": value["checkpoint"]["task_id"],
                "checkpoint_branch_field_present": "branch" in value["checkpoint"],
                "checkpoint_branch": value["checkpoint"].get("branch"),
                "global_step": value["checkpoint"]["completed_global_step"],
                "optimizer_steps": list(value["optimizer_steps"]),
            }
            for name, value in endpoints.items()
        },
        "mapper_checkpoint": str(_path(config, config["sources"]["mapper_checkpoint"])),
        "mapper_completed_updates": runtime["mapper_checkpoint"]["completed_updates"],
        "mapper_lineage_updates": runtime["mapper_checkpoint"]["lineage_mapper_updates"],
        "physical_phase_prior": str(_path(config, config["sources"]["physical_phase_prior"])),
        "fixed_response_projector": str(_path(config, config["sources"]["fixed_response_projector"])),
        "training_data": str(_path(config, config["sources"]["mnist_training_file"])),
        "training_rows": int(runtime["train_indices"].numel()),
        "development_rows": int(runtime["dev_indices"].numel()),
        "official_test_reads": 0,
    }
    _write_json(output_dir / "asset_reuse_receipt.json", result)


def _write_closeout(report_path: Path, route: Mapping[str, Any],
                    final_consumer: Mapping[str, Any],
                    training_summary: Mapping[str, Any] | None,
                    preflight: Mapping[str, Any] | None,
                    positive: Mapping[str, Any],
                    endpoint_summaries: Mapping[str, Any],
                    reconstruction: Mapping[str, Any]) -> None:
    state = ("PASSED_CANDIDATE_SCREEN" if final_consumer["final_gate"]["passed"]
             else "VALID_FAILED")
    endpoint_metrics = {
        name: summary["conditions"]["direct_decoder"]["evaluators"]
        for name, summary in endpoint_summaries.items()
    }
    lines = [
        "# LightGen 近期光学端点复用与完整 SLM 路径语义生成收尾",
        "",
        "## 结果与状态",
        "",
        f"- 状态：{state}。",
        f"- 开发端点选择：{route['selected_endpoint']}；短训跳过：{route['skip_decision']['skip_short_training']}。",
        "- 终态人口为 2,000 条新先验样本，每类 200；评价使用冻结 mapper、固定投影和完整 2,048 SLM→CMOS 路径。",
        f"- 唯一终态科学判据通过：{final_consumer['final_gate']['passed']}。",
        "",
        "## 科学认识",
        "",
        "- 开发端点比较只决定复用旧光学权重或继续一次短训；终态留出只在路线与训练完成后生成。",
        "- 冻结 mapper 只准备 SLM 振幅命令。光学训练从真实图像进入 SLM，并通过 CMOS 的 BOGT＋固定 SSIM 损失更新 Encoder 与 Generator。",
        "- 4,096 网格求解见证仅用于数值路径正控，不构成语义生成或硬件证据。",
        "- 该 2+2 Gaussian OLS 结果是项目数值候选，不代表论文原规模复现、人工可读性或实物部署。",
        "",
        "## 目标缺口",
        "",
        "- 人眼可读性、G1 全分布泛化、真实器件标定和实物 SLM→CMOS 部署均未测量。",
        "",
        "## 工程完成与验证",
        "",
        f"- 旧端点 E_D/E_C：{json.dumps(endpoint_metrics, ensure_ascii=False)}",
        f"- 4,096 正控：{len(positive['rows'])}/20；沿用既有数值容差 {GATE_LIMIT:g}。",
        f"- 光学预检：{json.dumps(preflight, ensure_ascii=False) if preflight else '条件式跳训，未执行训练更新预检；端点复用路径的数值前向和输入回放均完成。'}",
        f"- 短训：{json.dumps(training_summary, ensure_ascii=False) if training_summary else '0 次科学更新；复用所选旧端点。'}",
        f"- 真实 DEV 重构诊断：{json.dumps(reconstruction, ensure_ascii=False)}",
        "",
        "## 决策影响与下一步",
        "",
        f"- 最终判据：{json.dumps(final_consumer['final_gate'], ensure_ascii=False)}",
        "- 若通过，只进入正式人工复审准备；不等于通过人工可读性或硬件验收。",
        "- 若未通过，保留全部检查点与逐样本证据并停止；不自动追加更新、重采样或改阈值。",
        "- 本任务建议不构成对新训练、量化、材料接入或硬件操作的额外授权。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _load_resume_context(config: Mapping[str, Any], runtime: Mapping[str, Any],
                         endpoints: Mapping[str, Mapping[str, Any]],
                         mapper: torch.nn.Module, resume_state_path: Path
                         ) -> dict[str, Any]:
    expected_state_path = (ATTEMPT_DIR / "task_state.json").resolve()
    _require(resume_state_path.resolve() == expected_state_path,
             "resume requires the explicit task_state.json for this attempt", ValueError)
    _require((OUTPUT_DIR / "run_config.yaml").is_file()
             and yaml.safe_load((OUTPUT_DIR / "run_config.yaml").read_text(encoding="utf-8"))
             == config,
             "saved run configuration differs from the registered task")
    task_state = json.loads(resume_state_path.read_text(encoding="utf-8"))
    _require(task_state.get("task_id") == TASK_ID
             and task_state.get("state") in {"SCIENTIFIC_ACTIVE", "TERMINAL_MEASUREMENT_PENDING"}
             and task_state.get("authorized_updates") in {0, TRAIN_UPDATES},
             "task state is not an incomplete resumable attempt")

    dev_population = torch.load(
        OUTPUT_DIR / "development_population_and_commands.pt",
        map_location="cpu", weights_only=False,
    )
    dev_cohort = dev_population.get("cohort")
    dev_commands = dev_population.get("commands")
    _require(dev_population.get("schema") == f"{TASK_ID}.development_population.v1"
             and dev_population.get("task_id") == TASK_ID
             and dev_population.get("seed_base") == 202610300910
             and isinstance(dev_cohort, Mapping)
             and dev_cohort.get("role") == "new_development_prior_200_not_final_holdout",
             "saved development input population is missing or has the wrong identity")
    _validate_cohort(dev_cohort, 20)
    _assert_legal_commands(dev_commands)
    regenerated, _details = _make_commands(mapper, dev_cohort["coefficients"], runtime)
    _require(regenerated.shape == dev_commands.shape
             and float((regenerated - dev_commands).abs().max()) <= COMMAND_REPLAY_ATOL,
             "frozen mapper no longer reproduces the saved development SLM commands")
    replay = _verify_mapper_replay(mapper, runtime)
    replay_saved = json.loads((OUTPUT_DIR / "mapper_replay.json").read_text(encoding="utf-8"))
    _require(replay_saved.get("task_id") == TASK_ID
             and replay["rows"] == replay_saved.get("rows")
             and replay["max_abs_command_error"] <= COMMAND_REPLAY_ATOL,
             "saved mapper replay evidence is missing or invalid")

    endpoint_summaries: dict[str, Any] = {}
    for name in ENDPOINT_ORDER:
        path = OUTPUT_DIR / f"development_{name}.pt"
        _require(path.is_file(), f"saved {name} development measurement is missing",
                 FileNotFoundError)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        endpoint = endpoints[name]
        _require(artifact.get("task_id") == TASK_ID
                 and artifact.get("condition") == name
                 and artifact.get("checkpoint_run_id") == endpoint["checkpoint"]["run_id"]
                 and artifact.get("global_step") == endpoint["checkpoint"]["completed_global_step"]
                 and artifact.get("cohort_role") == dev_cohort["role"]
                 and torch.equal(artifact.get("labels"), dev_cohort["labels"])
                 and torch.equal(artifact.get("proposal_ids"), dev_cohort["proposal_ids"])
                 and torch.equal(artifact.get("commands"), dev_commands),
                 f"saved {name} endpoint measurement is not bound to the fixed development commands")
        endpoint_summaries[name] = artifact.get("semantic_summary")
        _require(isinstance(endpoint_summaries[name], Mapping),
                 f"saved {name} semantic summary is missing")
    selected_name = _choose_endpoint(endpoint_summaries)
    skip_training, skip_decision = _should_skip_training(
        selected_name, endpoint_summaries,
        float(config["measurement"]["endpoint_skip_average_macro_recall_gain"]),
        float(config["measurement"]["endpoint_skip_each_evaluator_minimum_delta"]),
    )
    route = json.loads((OUTPUT_DIR / "endpoint_route_decision.json").read_text(encoding="utf-8"))
    _require(route.get("task_id") == TASK_ID
             and route.get("selected_endpoint") == selected_name
             and route.get("skip_decision", {}).get("skip_short_training") is skip_training
             and route.get("conditional_training")
             == ("skip" if skip_training else "exactly_2000_joint_optical_updates"),
             "saved endpoint route decision differs from the recomputed registered rule")
    saved_skip = route["skip_decision"]
    _require(saved_skip.get("baseline") == "C45k"
             and saved_skip.get("selected_endpoint") == selected_name
             and math.isclose(float(saved_skip.get("ED_macro_recall_delta", math.nan)),
                              float(skip_decision["ED_macro_recall_delta"]), abs_tol=1.0e-12)
             and math.isclose(float(saved_skip.get("EC_macro_recall_delta", math.nan)),
                              float(skip_decision["EC_macro_recall_delta"]), abs_tol=1.0e-12)
             and math.isclose(float(saved_skip.get("average_macro_recall_delta", math.nan)),
                              float(skip_decision["average_macro_recall_delta"]), abs_tol=1.0e-12),
             "saved endpoint skip evidence differs from recomputed evaluation metrics")
    positive = json.loads((OUTPUT_DIR / "solver_positive_controls.json").read_text(encoding="utf-8"))
    _require(positive.get("task_id") == TASK_ID and positive.get("passed") is True
             and len(positive.get("rows", [])) == 20,
             "saved solver positive controls are incomplete or failed")

    selected = {"name": selected_name, **endpoints[selected_name]}
    schedule = _build_train_schedule(
        runtime["labels"], runtime["train_indices"],
        int(config["training"]["schedule_seed"]),
    )
    training_preflight = None
    resume_checkpoint_path = None
    if not skip_training:
        preflight_path = OUTPUT_DIR / "training_preflight.json"
        _require(preflight_path.is_file(), "training preflight evidence is missing",
                 FileNotFoundError)
        training_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        _require(training_preflight.get("status") == "READY"
                 and training_preflight.get("scientific_updates") == 0
                 and training_preflight.get("recovery", {}).get("passed") is True
                 and training_preflight.get("mapper_only_optimizer_rejected") is True,
                 "saved optical training preflight is not READY")
        schedule_path = ATTEMPT_DIR / "training_schedule.pt"
        _require(schedule_path.is_file(), "saved training schedule is missing", FileNotFoundError)
        saved_schedule = torch.load(schedule_path, map_location="cpu", weights_only=False)
        _require(torch.equal(saved_schedule, schedule),
                 "saved training schedule differs from the registered per-class schedule")

    identity_record = json.loads((OUTPUT_DIR / "run_identity.json").read_text(encoding="utf-8"))
    identity = ResearchRunIdentity.from_mapping(identity_record.get("run_identity", {}))
    authorization_id = f"{TASK_ID}:selected:{selected_name}:single-final-decision"
    _require(identity.task_id == TASK_ID and identity.run_id == task_state.get("run_id")
             and identity_record.get("run_id") == identity.run_id
             and identity.trajectory_id == f"{TASK_ID}:selected:{selected_name}:updates:{TRAIN_UPDATES}"
             and identity.semantic_authority_id == AUTHORITY
             and identity.config_identity == config
             and identity.model_identity == selected["checkpoint"]["model_identity"]
             and identity.execution_source_semantics_id == config["execution_source_semantics_id"],
             "saved run identity differs from the selected optical endpoint")
    receipt = PreflightReceipt.from_mapping(
        json.loads((OUTPUT_DIR / "preflight_ready.json").read_text(encoding="utf-8"))
    )
    _require(receipt.status == "READY" and receipt.authorization_id == authorization_id,
             "saved READY receipt is not bound to the selected endpoint authorization")
    control = ResearchRunControl(identity, authorization_id=authorization_id)
    control.mark_preflight_ready(receipt)

    if skip_training:
        _require(task_state.get("training_skipped_by_endpoint_rule") is True
                 and task_state.get("completed_updates") == 0,
                 "saved task state does not match the zero-update endpoint route")
    else:
        _require(task_state.get("authorized_updates") == TRAIN_UPDATES
                 and type(task_state.get("completed_updates")) is int
                 and task_state.get("completed_updates") in CHECKPOINT_STEPS,
                 "saved training state is not at a complete checkpoint boundary")
        checkpoint_path_value = task_state.get("checkpoint_path")
        _require(isinstance(checkpoint_path_value, str) and checkpoint_path_value,
                 "task state lacks the explicit last-complete checkpoint")
        resume_checkpoint_path = Path(checkpoint_path_value).resolve()
        _require(resume_checkpoint_path.parent == ATTEMPT_DIR.resolve()
                 and resume_checkpoint_path.is_file(),
                 "task state checkpoint is missing or outside this attempt")
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_updates = _validate_training_checkpoint(
            checkpoint, selected, schedule, config, identity.run_id
        )
        _require(checkpoint_updates == task_state["completed_updates"],
                 "task state update count differs from its registered checkpoint")
    return {
        "endpoint_summaries": endpoint_summaries,
        "route": route,
        "positive": positive,
        "selected": selected,
        "selected_name": selected_name,
        "skip_training": skip_training,
        "skip_decision": skip_decision,
        "schedule": schedule,
        "training_preflight": training_preflight,
        "control": control,
        "receipt": receipt,
        "identity": identity,
        "resume_checkpoint_path": resume_checkpoint_path,
        "task_state": task_state,
    }


def _run(config_path: Path, resume_state_path: Path | None = None,
        resume_preflight: bool = False) -> dict[str, Any]:
    config = _read_config(config_path)
    _require(torch.cuda.is_available(), "formal optical endpoint comparison requires AutoDL CUDA")
    resuming = resume_state_path is not None
    repairing_preflight = bool(resume_preflight)
    _require(not (resuming and repairing_preflight),
             "choose either scientific resume or preflight repair, not both", ValueError)
    if resuming or repairing_preflight:
        _require(OUTPUT_DIR.is_dir(),
                 f"registered output directory is missing: {OUTPUT_DIR}", FileNotFoundError)
    if repairing_preflight:
        _require((OUTPUT_DIR / "run_config.yaml").is_file()
                 and yaml.safe_load((OUTPUT_DIR / "run_config.yaml").read_text(encoding="utf-8"))
                 == config
                 and not (OUTPUT_DIR / "training_preflight.json").exists()
                 and not (OUTPUT_DIR / "preflight_ready.json").exists()
                 and not (OUTPUT_DIR / "run_identity.json").exists()
                 and not ATTEMPT_DIR.exists()
                 and ((OUTPUT_DIR / "preflight_recovery_input.pt").is_file()
                      or (OUTPUT_DIR / "training_preflight.json").is_file())
                 and not (OUTPUT_DIR / "final_population_and_commands.pt").exists()
                 and not (OUTPUT_DIR / "final_decision.json").exists(),
                 "preflight repair is allowed only for this saved pre-training partial attempt",
                 ValueError)
        repair_number = 1
        repair_note = OUTPUT_DIR / f"preflight_repair_{repair_number:02d}.json"
        while repair_note.exists():
            repair_number += 1
            repair_note = OUTPUT_DIR / f"preflight_repair_{repair_number:02d}.json"
        repair_text = OUTPUT_DIR / f"runner_source_preflight_repair_{repair_number:02d}.py"
        repair_text.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        telemetry_path = OUTPUT_DIR / "resource_telemetry.csv"
        prior_task_run_id = None
        if telemetry_path.is_file():
            with telemetry_path.open(newline="", encoding="utf-8") as stream:
                prior_task_run_id = next(csv.DictReader(stream), {}).get("run_id")
        _write_json(repair_note, {
            "task_id": TASK_ID,
            "repair_number": repair_number,
            "state": "REPAIR_STARTED",
            "prior_task_run_id": prior_task_run_id,
            "prior_exit": "COMMAND_EXITED_NONZERO before scientific updates",
            "cause": "recovery worker was launched by script path, so the project root was absent from sys.path",
            "repair": "launch the worker as a module and surface captured stderr; reuse saved development, endpoint, positive-control, and three-step recovery-prefix evidence",
            "development_population_resampled": False,
            "endpoint_measurements_repeated": False,
            "final_population_created": False,
            "scientific_updates": 0,
            "repair_runner_source": str(repair_text),
        })
    elif not resuming:
        _require(not OUTPUT_DIR.exists(),
                 f"new dated output already exists; refusing to overwrite: {OUTPUT_DIR}",
                 FileExistsError)
        OUTPUT_DIR.mkdir(parents=True)
        (OUTPUT_DIR / "run_config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (OUTPUT_DIR / "runner_source.py").write_text(
            Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
        )
        _write_task_contract(OUTPUT_DIR)
    frame._set_numerics()
    device = torch.device("cuda:0")
    runtime = _load_inputs(config, device)
    runtime["sources"] = config["sources"]["endpoints"]
    endpoints = _load_endpoints(config, runtime)
    mapper = _new_mapper(runtime)
    _require(all(not parameter.requires_grad for parameter in mapper.parameters()),
             "the inference mapper must remain frozen")
    replay = _verify_mapper_replay(mapper, runtime)
    if not resuming and not repairing_preflight:
        _write_json(OUTPUT_DIR / "mapper_replay.json", replay)
        _save_source_receipt(config, OUTPUT_DIR, endpoints, runtime)
    frozen_snapshot = _snapshot_frozen(runtime, mapper)

    if resuming:
        resumed = _load_resume_context(
            config, runtime, endpoints, mapper, resume_state_path
        )
        endpoint_summaries = resumed["endpoint_summaries"]
        route = resumed["route"]
        positive = resumed["positive"]
        selected = resumed["selected"]
        selected_name = resumed["selected_name"]
        skip_training = resumed["skip_training"]
        skip_decision = resumed["skip_decision"]
        schedule = resumed["schedule"]
        training_preflight = resumed["training_preflight"]
        control = resumed["control"]
        receipt = resumed["receipt"]
        identity = resumed["identity"]
        resume_checkpoint_path = resumed["resume_checkpoint_path"]
    elif repairing_preflight:
        saved = _load_saved_route_evidence(
            config, runtime, endpoints, mapper, OUTPUT_DIR
        )
        endpoint_summaries = saved["endpoint_summaries"]
        route = saved["route"]
        positive = saved["positive"]
        selected = saved["selected"]
        selected_name = saved["selected_name"]
        skip_training = saved["skip_training"]
        skip_decision = saved["skip_decision"]
        schedule = _build_train_schedule(
            runtime["labels"], runtime["train_indices"],
            int(config["training"]["schedule_seed"]),
        )
        training_preflight = None
        resume_checkpoint_path = None
    else:
        dev_cohort, dev_accounting = _sample_cohort(
            runtime, count=20, seed_base=202610300910,
            role="new_development_prior_200_not_final_holdout",
        )
        _validate_cohort(dev_cohort, 20)
        dev_commands, dev_command_summary = _make_commands(
            mapper, dev_cohort["coefficients"], runtime
        )
        _save_torch(OUTPUT_DIR / "development_population_and_commands.pt", {
            "schema": f"{TASK_ID}.development_population.v1",
            "task_id": TASK_ID, "seed_base": 202610300910,
            "cohort": dev_cohort, "sampling_accounting": dev_accounting,
            "commands": dev_commands, "command_summary": dev_command_summary,
            "mapper_checkpoint_run_id": runtime["mapper_checkpoint"]["run_id"],
            "fixed_projection_path": str(_path(config, config["sources"]["fixed_response_projector"])),
        }, new=True)
        endpoint_summaries = {}
        for name in ENDPOINT_ORDER:
            endpoint_summaries[name] = _measure_endpoint_cohort(
                name, endpoints[name], dev_cohort, dev_commands, runtime, OUTPUT_DIR
            )
        selected_name = _choose_endpoint(endpoint_summaries)
        skip_training, skip_decision = _should_skip_training(
            selected_name, endpoint_summaries,
            float(config["measurement"]["endpoint_skip_average_macro_recall_gain"]),
            float(config["measurement"]["endpoint_skip_each_evaluator_minimum_delta"]),
        )
        route = {
            "task_id": TASK_ID,
            "selected_endpoint": selected_name,
            "selection_rule": [
                "maximize min(E_D_macro_recall,E_C_macro_recall)",
                "then maximize their average",
                "then C45k,J,D0,Dgrad",
            ],
            "endpoint_metrics": {
                name: {
                    "E_D": summary["conditions"]["direct_decoder"]["evaluators"]["ED"]["macro_recall"],
                    "E_C": summary["conditions"]["direct_decoder"]["evaluators"]["EC"]["macro_recall"],
                }
                for name, summary in endpoint_summaries.items()
            },
            "skip_decision": skip_decision,
            "conditional_training": "skip" if skip_training else "exactly_2000_joint_optical_updates",
            "final_population_consumed": False,
        }
        _write_json(OUTPUT_DIR / "endpoint_route_decision.json", route)
        positive = _measure_positive_controls(
            config, runtime, endpoints["C45k"]["checkpoint"], OUTPUT_DIR
        )
        selected = {"name": selected_name, **endpoints[selected_name]}
        schedule = _build_train_schedule(
            runtime["labels"], runtime["train_indices"],
            int(config["training"]["schedule_seed"]),
        )
    if not resuming:
        training_preflight = None
        if not skip_training:
            saved_preflight_path = OUTPUT_DIR / "training_preflight.json"
            if repairing_preflight and saved_preflight_path.is_file():
                training_preflight = json.loads(
                    saved_preflight_path.read_text(encoding="utf-8")
                )
                _require(training_preflight.get("status") == "READY"
                         and training_preflight.get("selected_endpoint") == selected_name
                         and training_preflight.get("scientific_updates") == 0
                         and training_preflight.get("recovery", {}).get("passed") is True
                         and training_preflight.get("mapper_only_optimizer_rejected") is True,
                         "saved preflight result cannot authorize this selected endpoint")
            else:
                training_preflight = _preflight_training(
                    config, selected, runtime, mapper, schedule,
                    _snapshot_frozen(runtime, mapper), OUTPUT_DIR,
                )
            if repairing_preflight:
                repair_note_path = max(
                    OUTPUT_DIR.glob("preflight_repair_[0-9][0-9].json"),
                    key=lambda path: path.name,
                )
                repair_note = json.loads(repair_note_path.read_text(encoding="utf-8"))
                repair_note["state"] = "REPAIR_COMPLETED"
                repair_note["training_preflight_status"] = training_preflight["status"]
                repair_note["reused_saved_development_and_endpoint_measurements"] = True
                repair_note["reused_saved_three_step_recovery_prefix"] = training_preflight[
                    "three_step_prefix_reused_from_failed_preflight"
                ]
                _write_json(repair_note_path, repair_note)
        control, receipt, identity = _make_ready_control(
            config, selected, runtime, replay, positive, training_preflight
        )
        _write_json(OUTPUT_DIR / "preflight_ready.json", receipt.to_dict())
        _write_json(OUTPUT_DIR / "run_identity.json", {
            "task_id": TASK_ID, "run_id": identity.run_id,
            "trajectory_id": identity.trajectory_id, "run_identity": identity.to_dict(),
        })
        resume_checkpoint_path = None
    training_summary = None
    final_checkpoint = selected["checkpoint"]
    if not skip_training:
        final_checkpoint, training_summary = _train_optics(
            config, selected, runtime, schedule, identity.run_id, control, receipt,
            resume_checkpoint_path,
        )
    else:
        if resuming:
            control.resume_scientific_attempt(
                ATTEMPT_DIR, f"{identity.run_id}:attempt_01", receipt
            )
        else:
            control.open_scientific_attempt(
                ATTEMPT_DIR, f"{identity.run_id}:attempt_01", receipt
            )
            _write_json(ATTEMPT_DIR / "task_state.json", {
                "task_id": TASK_ID, "run_id": identity.run_id,
                "state": "TERMINAL_MEASUREMENT_PENDING",
                "completed_updates": 0, "authorized_updates": 0,
                "selected_endpoint": selected_name,
                "training_skipped_by_endpoint_rule": True,
                "checkpoint_path": str(selected["path"]),
                "last_complete_save_unix": time.time(),
            })

    final_population_path = OUTPUT_DIR / "final_population_and_commands.pt"
    if final_population_path.is_file():
        _require(resuming, "final population already exists; refusing to consume it a second time")
        saved_population = torch.load(final_population_path, map_location="cpu", weights_only=False)
        _require(saved_population.get("schema") == f"{TASK_ID}.final_population.v1"
                 and saved_population.get("task_id") == TASK_ID
                 and saved_population.get("seed_base") == 202610300930
                 and saved_population.get("mapper_checkpoint_run_id")
                 == runtime["mapper_checkpoint"]["run_id"],
                 "saved final population identity differs from this resumed attempt")
        final_cohort = saved_population.get("cohort")
        final_commands = saved_population.get("commands")
        final_accounting = saved_population.get("sampling_accounting")
        _require(isinstance(final_cohort, Mapping)
                 and final_cohort.get("role")
                 == "new_final_population_not_consumed_during_route_or_training",
                 "saved final population role changed")
        _validate_cohort(final_cohort, 200)
        _assert_legal_commands(final_commands)
        regenerated_commands, final_command_summary = _make_commands(
            mapper, final_cohort["coefficients"], runtime
        )
        _require(regenerated_commands.shape == final_commands.shape
                 and float((regenerated_commands - final_commands).abs().max())
                 <= COMMAND_REPLAY_ATOL,
                 "frozen mapper no longer reproduces the saved final SLM commands")
    else:
        final_cohort, final_accounting = _sample_cohort(
            runtime, count=200, seed_base=202610300930,
            role="new_final_population_not_consumed_during_route_or_training",
        )
        _validate_cohort(final_cohort, 200)
        final_commands, final_command_summary = _make_commands(
            mapper, final_cohort["coefficients"], runtime
        )
        _save_torch(final_population_path, {
            "schema": f"{TASK_ID}.final_population.v1",
            "task_id": TASK_ID, "seed_base": 202610300930,
            "cohort": final_cohort,
            "sampling_accounting": final_accounting, "commands": final_commands,
            "command_summary": final_command_summary,
            "mapper_checkpoint_run_id": runtime["mapper_checkpoint"]["run_id"],
            "fixed_projection_path": str(_path(config, config["sources"]["fixed_response_projector"])),
        }, new=True)

    unique_conditions: dict[str, dict[str, Any]] = {}
    conditions_to_checkpoint = {
        "C45k": endpoints["C45k"]["checkpoint"],
        "selected_initial": selected["checkpoint"],
        "final_candidate": final_checkpoint,
    }
    condition_checkpoint_paths = {
        "C45k": endpoints["C45k"]["path"],
        "selected_initial": selected["path"],
        "final_candidate": (
            ATTEMPT_DIR / "terminal_checkpoint.pt"
            if training_summary is not None else selected["path"]
        ),
    }
    state_signatures: dict[tuple[str, int], str] = {}
    for name, checkpoint in conditions_to_checkpoint.items():
        signature = (checkpoint["run_id"], int(checkpoint["completed_global_step"]))
        if signature in state_signatures:
            unique_conditions[name] = unique_conditions[state_signatures[signature]]
            continue
        state_signatures[signature] = name
        artifact_path = OUTPUT_DIR / f"final_{name}.pt"
        if artifact_path.is_file():
            _require(resuming,
                     f"unexpected pre-existing final measurement artifact: {artifact_path}")
            saved = torch.load(artifact_path, map_location="cpu", weights_only=False)
            _require(saved.get("task_id") == TASK_ID
                     and saved.get("artifact_condition") == name
                     and saved.get("checkpoint_run_id") == checkpoint["run_id"]
                     and saved.get("checkpoint_step") == checkpoint["completed_global_step"]
                     and _path(config, saved.get("checkpoint_path", ""))
                     == condition_checkpoint_paths[name]
                     and torch.equal(saved.get("labels"), final_cohort["labels"])
                     and torch.equal(saved.get("proposal_ids"), final_cohort["proposal_ids"])
                     and torch.equal(saved.get("commands"), final_commands),
                     f"saved final {name} measurement is not bound to this terminal population")
            produced = {
                key: saved.get(key) for key in ("raw_cmos", "coefficients", "readouts")
            }
            _require(all(torch.is_tensor(value) and value.shape[0] == 2_000
                         and torch.isfinite(value).all().item()
                         for value in produced.values()),
                     f"saved final {name} optical outputs are incomplete")
        else:
            produced = _measure_commands(checkpoint, final_commands, runtime)
        semantic = frame._summary_from_readouts(
            runtime["input_bundle"], final_cohort, produced["readouts"]
        )
        payload = {
            "condition": name,
            "artifact_name": name,
            "checkpoint": checkpoint,
            "commands": final_commands,
            "produced": produced,
            "semantic_summary": semantic,
        }
        unique_conditions[name] = payload
        if not artifact_path.is_file():
            _save_torch(artifact_path, {
                "task_id": TASK_ID, "artifact_condition": name,
                "checkpoint_run_id": checkpoint["run_id"],
                "checkpoint_step": checkpoint["completed_global_step"],
                "checkpoint_path": str(condition_checkpoint_paths[name]),
                "cohort_role": final_cohort["role"],
                "labels": final_cohort["labels"],
                "proposal_ids": final_cohort["proposal_ids"],
                "commands": final_commands,
                **produced,
                "semantic_summary": semantic,
            }, new=True)

    _write_json(OUTPUT_DIR / "final_condition_registry.json", {
        "task_id": TASK_ID,
        "conditions": {
            name: {
                "artifact": f"final_{unique_conditions[name]['artifact_name']}.pt",
                "checkpoint_path": str(condition_checkpoint_paths[name]),
                "checkpoint_run_id": conditions_to_checkpoint[name]["run_id"],
                "checkpoint_step": conditions_to_checkpoint[name]["completed_global_step"],
            }
            for name in ("C45k", "selected_initial", "final_candidate")
        },
    })

    real_rows = _real_dev_rows(runtime["labels"], runtime["dev_indices"])
    reconstruction = {}
    reconstruction_conditions = {
        "C45k": endpoints["C45k"]["checkpoint"],
        "selected_initial": selected["checkpoint"],
        "final_candidate": final_checkpoint,
    }
    reconstructed_signatures: dict[tuple[str, int], str] = {}
    for name, checkpoint in reconstruction_conditions.items():
        signature = (checkpoint["run_id"], int(checkpoint["completed_global_step"]))
        if signature in reconstructed_signatures:
            reconstruction[name] = reconstruction[reconstructed_signatures[signature]]
            continue
        reconstructed_signatures[signature] = name
        reconstruction[name] = _measure_reconstruction(checkpoint, real_rows, runtime)

    neighbors = {}
    for name, payload in unique_conditions.items():
        neighbors[name] = _nearest_same_class_training_rows(
            payload["produced"]["readouts"], final_cohort["labels"],
            runtime["pixels"], runtime["train_indices"], runtime["labels"], runtime["device"],
        )
        _write_json(OUTPUT_DIR / f"nearest_train_{name}.json", {
            "task_id": TASK_ID,
            "definition": (
                "nearest original MNIST TRAIN image within the requested class in fixed "
                "28x28 readout space; L2 divided by sqrt(784)"
            ),
            "rows": neighbors[name],
        })
    _save_contact_sheet(
        OUTPUT_DIR / "final_condition_comparison.png", final_cohort["labels"],
        {name: payload["produced"] for name, payload in unique_conditions.items()},
    )
    _write_sample_csv(
        OUTPUT_DIR / "final_per_sample_results.csv", final_cohort,
        unique_conditions, neighbors,
    )
    final_consumer, reconsumed = _consume_final_measurements(config, runtime, OUTPUT_DIR)
    terminal_vs_start = None
    if training_summary is not None:
        selected_semantic = reconsumed["selected_initial"]["semantic_summary"]
        final_semantic = reconsumed["final_candidate"]["semantic_summary"]
        terminal_vs_start = {}
        for family, key in (("E_D", "ED"), ("E_C", "EC")):
            terminal_vs_start[family] = _paired_class_bootstrap(
                final_cohort["labels"],
                torch.tensor(selected_semantic["conditions"]["direct_decoder"]["evaluators"][key]["predicted_labels"]),
                torch.tensor(final_semantic["conditions"]["direct_decoder"]["evaluators"][key]["predicted_labels"]),
                int(config["measurement"]["bootstrap_seed"]),
                int(config["measurement"]["bootstrap_resamples"]),
            )
    final_result = {
        "task_id": TASK_ID,
        "status": "PASSED_CANDIDATE_SCREEN"
        if final_consumer["final_gate"]["passed"] else "VALID_FAILED",
        "scientific_updates": TRAIN_UPDATES if training_summary else 0,
        "route": route,
        "selected_start_vs_C45k_development_delta": {
            "E_D": (
                endpoint_summaries[selected_name]["conditions"]["direct_decoder"]["evaluators"]["ED"]["macro_recall"]
                - endpoint_summaries["C45k"]["conditions"]["direct_decoder"]["evaluators"]["ED"]["macro_recall"]
            ),
            "E_C": (
                endpoint_summaries[selected_name]["conditions"]["direct_decoder"]["evaluators"]["EC"]["macro_recall"]
                - endpoint_summaries["C45k"]["conditions"]["direct_decoder"]["evaluators"]["EC"]["macro_recall"]
            ),
        },
        "terminal_vs_selected_start_final_population_paired_delta": terminal_vs_start,
        "final_gate": final_consumer["final_gate"],
        "final_population_rows": 2_000,
        "solver_positive_rows": 20,
        "human_readability": "not_measured",
        "hardware_deployment": "not_measured",
    }
    _write_json(OUTPUT_DIR / "final_decision.json", final_result)
    _write_json(OUTPUT_DIR / "reconstruction_diagnostics.json", reconstruction)
    if final_consumer["final_gate"]["passed"]:
        control.complete()
    else:
        control.valid_failed()
    training_report = None if training_summary is None else {
        "updates": training_summary["training_updates"],
        "selected_endpoint": selected_name,
        "terminal_global_step": final_checkpoint["completed_global_step"],
        "loss": "BOGT plus fixed SSIM",
    }
    _write_closeout(
        OUTPUT_DIR / "closeout_report.md", route, final_consumer, training_report,
        training_preflight, positive, endpoint_summaries, reconstruction,
    )
    _write_json(ATTEMPT_DIR / "task_state.json", {
        "task_id": TASK_ID, "run_id": identity.run_id,
        "state": final_result["status"],
        "completed_updates": final_result["scientific_updates"],
        "authorized_updates": final_result["scientific_updates"],
        "final_decision_path": str(OUTPUT_DIR / "final_decision.json"),
    })
    _assert_frozen_unchanged(frozen_snapshot, runtime, mapper)
    return final_result


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("execute", "recovery-worker"), default="execute")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume-state", type=Path,
                        help="resume this incomplete attempt from its explicit task_state.json")
    parser.add_argument("--resume-preflight", action="store_true",
                        help="repair this registered attempt before READY using saved evidence")
    parser.add_argument("--recovery-input", type=Path)
    parser.add_argument("--recovery-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    config_path = args.config.resolve()
    _read_config(config_path)
    if args.mode == "recovery-worker":
        _require(args.resume_state is None and not args.resume_preflight
                 and args.recovery_input is not None and args.recovery_output is not None,
                 "recovery worker needs explicit input and output paths", ValueError)
        _recovery_worker(
            config_path, args.recovery_input.resolve(), args.recovery_output.resolve()
        )
        print(json.dumps({"status": "RECOVERY_STEP4_SAVED"}), flush=True)
        return 0
    _require(args.recovery_input is None and args.recovery_output is None,
             "recovery input/output are available only to the internal recovery worker", ValueError)
    result = _run(
        config_path, None if args.resume_state is None else args.resume_state.resolve(),
        resume_preflight=args.resume_preflight,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
