"""Train one continuous G1-to-amplitude-SLM mapper through the frozen C optics."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    PixelMLPConfirmationEvaluatorV1,
    ThirdConfirmationEvaluator,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PREFLIGHT_READY,
    RUN_COMPLETED,
    RUN_SCIENTIFIC_ACTIVE,
    RUN_VALID_FAILED,
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    check_resume_compatibility,
)
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_SLM_INPUT_MAPPING_PILOT_V1"
POPULATION_SCHEMA = "lightgen_g1_slm_mapping_population.v1"
CLASS_COUNT = 10
PORT_COUNT = 100
REAL_DIM = 200
QUANTIZED_CONDITION_NAMES = ("A8_amplitude_linear", "I8_intensity_linear")
PIXEL_SHAPE = (200, 200)
PIXEL_COUNT = 40_000
GRIDS = (2048, 4096)
MEASUREMENT_LIMIT = 1.0e-4
RECOVERY_ABSOLUTE_TOLERANCE = 5.0e-8
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_slm_input_mapping_pilot_v1.yaml"
PREFLIGHT_DIR = ROOT / "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/preflight_01"
ATTEMPT_DIR = ROOT / "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/attempt_01"
WINDOWS_BACKUP_DIR = ROOT / "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/autodl_backup_attempt_01"
CLOSEOUT_REPAIR_DIR = ROOT / "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/closeout_repair_01"
CLOSEOUT_REPAIR_BACKUP_DIR = ROOT / "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/autodl_backup_closeout_repair_01"
MAPPER_ID = "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid"
POWER_BOUND_W = 0.0001970270253
MODULE_PATH = f"scripts.research.lightgen.{Path(__file__).stem}"


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_config(config_path: Path) -> dict[str, Any]:
    _require(config_path.resolve() == DEFAULT_CONFIG.resolve(),
             "this task uses its fixed explicit configuration", ValueError)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "configuration must be a mapping", ValueError)
    _validate_config(value)
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == "lightgen_g1_slm_input_mapping_pilot.v1"
             and config.get("task_id") == TASK_ID
             and config.get("evidence_tier") == "diagnostic_only",
             "task identity or evidence tier changed", ValueError)
    _require(config.get("semantic_authority_id") ==
             "USER_AUTHORIZED_G1_SLM_INPUT_MAPPING_PILOT_20261026",
             "this runner requires the current user's task authority", ValueError)
    optics = config["optics"]
    _require(optics.get("checkpoint_branch") == "C"
             and optics.get("checkpoint_seed") == 2026101501
             and optics.get("cumulative_step") == 45_000
             and optics.get("source_power_w") == 2.493e-4
             and optics.get("layer_power_transmission") == 0.889
             and optics.get("slm_shape") == [200, 200]
             and optics.get("encoder_layers") == 2
             and optics.get("decoder_layers") == 2
             and optics.get("ols_basis") == "gaussian_s12_v1"
             and optics.get("ols_ports") == 100
             and optics.get("ols_transport") == "identity"
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED"
             and optics.get("training_grid") == 2048
             and optics.get("measurement_grids") == list(GRIDS),
             "fixed C optical source or grid contract changed", ValueError)
    population = config["population"]
    _require(population == {
        "classes": 10,
        "train_per_class": 1000,
        "validation_per_class": 200,
        "holdout_per_class": 200,
        "train_seed_base": 2026102610,
        "validation_seed_base": 2026102630,
        "holdout_seed_base": 2026102650,
        "initialization_seed": 2026102670,
        "schedule_seed": 2026102671,
        "max_proposals_per_class_per_population": 1_000_000,
        "proposal_batch_size": 256,
        "rejection": "original_total_modal_power_only",
        "max_modal_power_w": POWER_BOUND_W,
    }, "fixed G1 populations or seeds changed", ValueError)
    training = config["training"]
    _require(training == {
        "steps": 5000,
        "epochs": 5,
        "logical_batch_size": 10,
        "samples_per_class_per_batch": 1,
        "mapper": MAPPER_ID,
        "optimizer": "Adam",
        "learning_rate": 1.0e-3,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "mapper_precision": "float32",
        "optical_precision": "float64_complex128",
        "amp": False,
        "tf32": False,
        "checkpoint_interval_steps": 500,
        "progress_interval_steps": 100,
        "validation_steps": [0, 1000, 2500, 5000],
        "select_best_validation_checkpoint": False,
    }, "mapper training budget or optimizer changed", ValueError)
    measurement = config["measurement"]
    _require(measurement.get("relative_error_max") == MEASUREMENT_LIMIT
             and measurement.get("coefficient_phase_alignment") == "one_global_phase"
             and measurement.get("cmos_normalization") == "none"
             and measurement.get("cmos_power_matching") is False
             and measurement.get("cmos_gain_adjustment") is False
             and measurement.get("continuous_is_primary") is True
             and measurement.get("quantized_conditions") == [
                 "A8_amplitude_linear", "I8_intensity_linear"
             ]
             and measurement.get("quantization_aware_training") is False
             and measurement.get("evaluator_scope") == "frozen_ED_EC_auxiliary_at_2048_only",
             "final measurement or quantization contract changed", ValueError)
    output = config["output"]
    _require(output.get("preflight_root") == PREFLIGHT_DIR.relative_to(ROOT).as_posix()
             and output.get("experiment_root") == ATTEMPT_DIR.relative_to(ROOT).as_posix()
             and output.get("windows_backup_root") == WINDOWS_BACKUP_DIR.relative_to(ROOT).as_posix()
             and output.get("final_consumer") == "recompute_saved_population_commands_and_raw_outputs",
             "fixed output paths or consumer changed", ValueError)


class AmplitudeMapper(torch.nn.Module):
    """The sole trainable component: normalized complex coefficients to real SLM amplitude."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(REAL_DIM, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, PIXEL_COUNT),
            torch.nn.Sigmoid(),
        )

    def forward(self, normalized_coefficients: torch.Tensor) -> torch.Tensor:
        if normalized_coefficients.ndim != 2 or normalized_coefficients.shape[1] != REAL_DIM:
            raise ValueError("mapper input must have shape [batch,200]")
        return self.network(normalized_coefficients).reshape(-1, 1, *PIXEL_SHAPE)


def _new_mapper(device: torch.device, initialization_seed: int) -> AmplitudeMapper:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        mapper = AmplitudeMapper().to(device=device, dtype=torch.float32)
    return mapper


def _numpy_phase_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    a = np.asarray(actual, dtype=np.complex128).reshape(-1)
    t = np.asarray(target, dtype=np.complex128).reshape(-1)
    _require(a.shape == t.shape and np.isfinite(a.real).all() and np.isfinite(a.imag).all()
             and np.isfinite(t.real).all() and np.isfinite(t.imag).all(),
             "complex error inputs are malformed", ValueError)
    target_norm = float(np.linalg.norm(t))
    _require(math.isfinite(target_norm) and target_norm > 0.0,
             "complex error reference has zero or nonfinite norm", ValueError)
    phase = float(np.angle(np.vdot(t, a)))
    return phase, float(np.linalg.norm(a - np.exp(1j * phase) * t) / target_norm)


def _numpy_relative_error(actual: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    denominator = float(np.linalg.norm(t.reshape(-1)))
    _require(a.shape == t.shape and math.isfinite(denominator) and denominator > 0.0
             and np.isfinite(a).all() and np.isfinite(t).all(),
             "native intensity error inputs are malformed", ValueError)
    return float(np.linalg.norm((a - t).reshape(-1)) / denominator)


def _phase_invariant_loss(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared relative complex error minimized over one phase per sample."""
    if actual.shape != target.shape or actual.ndim != 2 or not torch.is_complex(actual) or not torch.is_complex(target):
        raise ValueError("phase-invariant loss requires equal complex [batch,ports] tensors")
    target_power = target.abs().square().sum(dim=1)
    if not torch.isfinite(target_power).all() or torch.any(target_power <= 0.0):
        raise ValueError("phase-invariant loss target has zero or nonfinite power")
    actual_power = actual.abs().square().sum(dim=1)
    correlation = (target.conj() * actual).sum(dim=1)
    residual_power = (actual_power + target_power - 2.0 * correlation.abs()).clamp_min(0.0)
    return residual_power / target_power


def _numeric_math_checks() -> dict[str, Any]:
    target = np.asarray([0.4 + 0.3j, -0.2 + 0.7j, 0.11 - 0.23j], dtype=np.complex128)
    actual = np.asarray([0.39 + 0.28j, -0.18 + 0.71j, 0.10 - 0.21j], dtype=np.complex128)
    torch_target = torch.from_numpy(target[None, :]).requires_grad_(False)
    torch_actual = torch.from_numpy(actual[None, :]).requires_grad_(True)
    torch_value = _phase_invariant_loss(torch_actual, torch_target).sum()
    torch_value.backward()
    numpy_phase, numpy_error = _numpy_phase_error(actual, target)
    numpy_loss = numpy_error**2
    _require(abs(float(torch_value.detach()) - numpy_loss) <= 1e-12,
             "complex128 loss disagrees with independent NumPy global-phase reference")
    direction = np.asarray([0.2 - 0.3j, -0.5 + 0.1j, 0.4 + 0.2j], dtype=np.complex128)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-6
    plus = _numpy_phase_error(actual + epsilon * direction, target)[1] ** 2
    minus = _numpy_phase_error(actual - epsilon * direction, target)[1] ** 2
    fd = (plus - minus) / (2.0 * epsilon)
    actual_grad = float(torch.real((torch_actual.grad.conj() * torch.from_numpy(direction[None, :])).sum()))
    derivative_relative_error = abs(fd - actual_grad) / max(abs(fd), abs(actual_grad), 1.0e-12)
    _require(derivative_relative_error <= 1.0e-5,
             "complex128 objective derivative disagrees with independent central difference")
    wrong_target = target.copy()
    wrong_target[1] *= np.exp(0.4j)
    _, wrong_error = _numpy_phase_error(actual, wrong_target)
    _require(wrong_error > numpy_error and wrong_error > 1.0e-3,
             "global-phase counterexample did not distinguish a portwise phase change")
    return {
        "executed": True,
        "passed": True,
        "numpy_phase_rad": numpy_phase,
        "torch_loss": float(torch_value.detach()),
        "numpy_loss": numpy_loss,
        "central_difference_directional_derivative": fd,
        "autograd_directional_derivative": actual_grad,
        "derivative_relative_error": derivative_relative_error,
        "wrong_port_phase_error": wrong_error,
    }


def _validate_source_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    checkpoint_path = _resolve(source["checkpoint"])
    prior_path = _resolve(source["fitted_gaussian_prior"])
    reference_path = _resolve(source["source_references"])
    baseline_dir = _resolve(source["baseline_evidence_dir"])
    replication_dir = _resolve(source["replication_evidence_dir"])
    reachability_dir = _resolve(source["reachability_attempt_dir"])
    evaluator_ed_path = _resolve(source["evaluator_ed_checkpoint"])
    evaluator_ec_path = _resolve(source["evaluator_ec_checkpoint"])
    explicit_files = (
        checkpoint_path, prior_path, reference_path,
        baseline_dir / "machine_report.json", baseline_dir / "config.json",
        baseline_dir / "C" / "endpoint_summary.json", baseline_dir / "C" / "terminal_model.pt",
        baseline_dir / "final_report.json", replication_dir / "report.json",
        reachability_dir / "run_identity.json", reachability_dir / "windows_reconsumption.json",
        reachability_dir / "amplitude_and_intensity_commands.pt",
        reachability_dir / "compact_replay_evidence.pt",
        evaluator_ed_path, evaluator_ec_path,
    )
    for path in explicit_files:
        _require(path.is_file(), f"explicit fixed input is missing: {path}", FileNotFoundError)

    source_report = _json(baseline_dir / "machine_report.json")
    source_config = _json(baseline_dir / "config.json")
    endpoint = _json(baseline_dir / "C" / "endpoint_summary.json")
    _require(source_report.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
             and source_report.get("measurement_complete") is True
             and endpoint.get("branch") == "C"
             and endpoint.get("global_step") == 45_000
             and endpoint.get("status") == "VALID_FAILED"
             and endpoint.get("measurement_complete") is True,
             "fixed C source task or its historical status changed")
    source_optics = source_config["optics"]
    source_small = source_optics["small"]
    planned_optics = config["optics"]
    _require(source_optics.get("input_power_w") == planned_optics["source_power_w"]
             and source_optics.get("layer_power_transmission") == planned_optics["layer_power_transmission"]
             and source_small.get("slm_shape") == planned_optics["slm_shape"]
             and source_small.get("encoder_layers") == planned_optics["encoder_layers"]
             and source_small.get("generator_layers") == planned_optics["decoder_layers"]
             and source_small.get("ols_basis") == planned_optics["ols_basis"]
             and source_small.get("ols_ports") == planned_optics["ols_ports"]
             and source_small.get("ols_transport") == planned_optics["ols_transport"]
             and source_small.get("production_propagation_grid") == planned_optics["training_grid"]
             and source_small.get("geometry_status") == planned_optics["geometry_status"],
             "loaded C optical setup does not match the frozen plan")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("task_id") == source_report["task_id"]
             and checkpoint.get("branch") == "C"
             and checkpoint.get("seed") == 2026101501
             and checkpoint.get("completed_global_step") == 45_000
             and checkpoint.get("config_identity") == source_config
             and checkpoint.get("evaluator_role") == "frozen_confirmation_only",
             "loaded checkpoint is not the explicit frozen C endpoint")
    _require(tuple(checkpoint["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(checkpoint["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100),
             "C checkpoint phase stack is not the frozen two-layer 2+2 system")
    identity = checkpoint.get("model_identity", {})
    _require(identity.get("ols") == "gaussian_s12_v1"
             and identity.get("ols_geometry", {}).get("transport") == "identity",
             "C checkpoint OLS identity or transport changed")
    optimizer_states = checkpoint.get("optimizer", {}).get("state", {})
    optimizer_steps = [int(torch.as_tensor(value["step"]).item()) for value in optimizer_states.values()
                       if isinstance(value, dict) and "step" in value]
    _require(len(optimizer_steps) == 2 and all(step == 45_000 for step in optimizer_steps),
             "source C optimizer state does not match cumulative step 45,000")
    references = torch.load(reference_path, map_location="cpu", weights_only=False)
    for field in ("scale", "power_reference", "camera_gain"):
        _require(torch.equal(torch.as_tensor(checkpoint[field]).cpu(),
                             torch.as_tensor(references[field]).cpu()),
                 f"source reference does not match the C checkpoint field: {field}")
    _require(checkpoint["weights"] == references["weights"],
             "source objective weights differ from the C reference")
    terminal = torch.load(baseline_dir / "C" / "terminal_model.pt", map_location="cpu", weights_only=False)
    _require(terminal.get("branch") == "C" and terminal.get("global_step") == 45_000
             and terminal.get("run_id") == source_report.get("run_id")
             and set(terminal["model"]) == set(checkpoint["model"])
             and all(torch.equal(terminal["model"][key], checkpoint["model"][key])
                     for key in checkpoint["model"]),
             "explicit C checkpoint differs from its saved terminal model")
    _require(checkpoint["scale"].shape == (PORT_COUNT,)
             and checkpoint["scale"].dtype == torch.float64
             and torch.isfinite(checkpoint["scale"]).all().item()
             and torch.all(checkpoint["scale"] > 0.0),
             "frozen modal scale is invalid")
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    _require(set(prior) == set(range(CLASS_COUNT)),
             "fixed C G1 prior must contain one full-covariance Gaussian for each class")
    for label in range(CLASS_COUNT):
        fit = prior[label]
        _require(tuple(fit["mean"].shape) == (REAL_DIM,)
                 and tuple(fit["covariance"].shape) == (REAL_DIM, REAL_DIM)
                 and tuple(fit["cholesky"].shape) == (REAL_DIM, REAL_DIM)
                 and torch.isfinite(fit["mean"]).all().item()
                 and torch.isfinite(fit["covariance"]).all().item()
                 and torch.isfinite(fit["cholesky"]).all().item(),
                 f"fixed G1 class-{label} Gaussian parameters are incomplete")
        regularized = fit["covariance"].double() + fit["ridge"].double() * torch.eye(REAL_DIM, dtype=torch.float64)
        _require(torch.allclose(fit["cholesky"].double() @ fit["cholesky"].double().T,
                                regularized, rtol=1.0e-7, atol=1.0e-10),
                 f"fixed G1 class-{label} Cholesky does not describe its saved covariance")

    replication_report = _json(replication_dir / "report.json")
    _require(replication_report.get("task_id") ==
             "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1"
             and replication_report.get("status") == "COMPLETE_DIAGNOSTIC",
             "fixed G1 replication source is incomplete")
    reachability_identity = _json(reachability_dir / "run_identity.json")
    reachability_reconsumption = _json(reachability_dir / "windows_reconsumption.json")
    _require(reachability_identity.get("task_id") == "LIGHTGEN_FROZEN_SLM_OLS_INPUT_REACHABILITY_V1"
             and reachability_identity.get("status") == "CLOSED"
             and reachability_identity.get("checkpoint_step") == 45_000
             and reachability_reconsumption.get("status") == "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED"
             and reachability_reconsumption.get("decision") == "ALL_200_NUMERICAL_COMMAND_WITNESSES_VERIFIED",
             "fixed saved G1 command-replay preparation evidence is not closed and consumed")
    commands = torch.load(reachability_dir / "amplitude_and_intensity_commands.pt",
                          map_location="cpu", weights_only=False)
    compact = torch.load(reachability_dir / "compact_replay_evidence.pt",
                         map_location="cpu", weights_only=False)
    _require(commands.get("schema") == "lightgen_frozen_slm_ols_amplitude_commands.v1"
             and compact.get("schema") == "lightgen_frozen_slm_ols_compact_replay_evidence.v1"
             and compact.get("run_id") == reachability_identity.get("run_id")
             and tuple(commands["amplitude_command"].shape) == (200, 1, 200, 200)
             and tuple(commands["intensity_command"].shape) == (200, 1, 200, 200)
             and tuple(compact["target_coefficients"].shape) == (200, PORT_COUNT),
             "saved dual-grid command replay arrays have changed shape or identity")

    evaluator_ed = _load_evaluator(evaluator_ed_path, "ED")
    evaluator_ec = _load_evaluator(evaluator_ec_path, "EC")
    source_power = float(checkpoint["config_identity"]["optics"]["input_power_w"])
    _require(source_power == config["optics"]["source_power_w"]
             and math.isclose(base.MAX_MODAL_POWER_W, config["population"]["max_modal_power_w"],
                              rel_tol=0.0, abs_tol=1.0e-15),
             "source power or original total-modal-power rejection bound changed")
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "prior": prior,
        "references": references,
        "scale": checkpoint["scale"].detach().cpu().double(),
        "camera_gain": float(torch.as_tensor(checkpoint["camera_gain"]).item()),
        "power_bound_w": base.MAX_MODAL_POWER_W,
        "source_report": source_report,
        "source_config": source_config,
        "commands": commands,
        "compact": compact,
        "reachability_identity": reachability_identity,
        "evaluator_ed": evaluator_ed,
        "evaluator_ec": evaluator_ec,
        "evaluator_ed_path": evaluator_ed_path,
        "evaluator_ec_path": evaluator_ec_path,
    }


def _load_evaluator(path: Path, family: str) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "ED": ("sic_confirmation_evaluator_d_checkpoint.v1", "pixel_mlp_confirmation_evaluator_v1",
               "E_D_QUALIFICATION"),
        "EC": ("third_confirmation_evaluator_checkpoint.v1", "third_confirmation_canonical_cnn_v1",
               "E_C_QUALIFICATION"),
    }
    schema, architecture, split = expected[family]
    _require(payload.get("schema") == schema and payload.get("architecture_kind") == architecture
             and payload.get("authority_status", "QUALIFIED") == "QUALIFIED"
             and payload.get("qualification", {}).get("split_role") == split
             and payload.get("qualification", {}).get("passed") is True
             and payload.get("independence", {}).get("role") == "INDEPENDENT_CONFIRMATION_ONLY"
             and payload.get("independence", {}).get("used_on_generated_cmos_in_this_task") is False,
             f"frozen {family} evaluator checkpoint is not qualified for independent confirmation")
    evaluator: torch.nn.Module = (PixelMLPConfirmationEvaluatorV1() if family == "ED"
                                  else ThirdConfirmationEvaluator())
    evaluator.load_state_dict(payload["state_dict"], strict=True)
    evaluator.eval()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in evaluator.parameters()),
             f"{family} evaluator must remain frozen")
    return evaluator


def _sample_g1_class(
    fit: Mapping[str, torch.Tensor], scale: torch.Tensor, *, label: int, seed: int,
    count: int, proposal_limit: int, proposal_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted: list[torch.Tensor] = []
    proposal_ids: list[torch.Tensor] = []
    evaluated = 0
    retained = 0
    rejected = 0
    while retained < count:
        remaining = proposal_limit - evaluated
        _require(remaining > 0,
                 f"G1 class {label} reached its fixed proposal cap before population completion")
        batch = min(proposal_batch_size, remaining)
        noise = torch.randn((batch, REAL_DIM), generator=generator, dtype=torch.float64)
        normalized_real = fit["mean"].double() + noise @ fit["cholesky"].double().T
        normalized_complex = torch.view_as_complex(normalized_real.reshape(batch, PORT_COUNT, 2).contiguous())
        coefficients = normalized_complex * scale.double().sqrt().unsqueeze(0)
        valid = coefficients.abs().square().sum(dim=1) <= base.MAX_MODAL_POWER_W
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        take = positions[: count - retained]
        if take.numel():
            accepted.append(coefficients[take].to(torch.complex64))
            proposal_ids.append(take + evaluated)
            retained += int(take.numel())
        rejected += int((~valid).sum())
        evaluated += batch
    samples = torch.cat(accepted)
    proposals = torch.cat(proposal_ids).long()
    _require(tuple(samples.shape) == (count, PORT_COUNT)
             and torch.unique(proposals).numel() == count
             and torch.all(samples.abs().square().sum(dim=1) <= base.MAX_MODAL_POWER_W),
             "G1 fixed-power population failed sample identity validation")
    return samples, proposals, {
        "label": label,
        "normal_seed": seed,
        "proposal_limit": proposal_limit,
        "proposals_evaluated": evaluated,
        "retained_samples": retained,
        "power_rejected_candidates": rejected,
        "rejection_fraction": rejected / evaluated,
        "power_bound_w": base.MAX_MODAL_POWER_W,
        "selection_rule": "first accepted candidates in proposal order; total modal power only",
    }


def _sample_population(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    population_config = config["population"]
    counts = {
        "train": int(population_config["train_per_class"]),
        "validation": int(population_config["validation_per_class"]),
        "holdout": int(population_config["holdout_per_class"]),
    }
    seed_bases = {
        "train": int(population_config["train_seed_base"]),
        "validation": int(population_config["validation_seed_base"]),
        "holdout": int(population_config["holdout_seed_base"]),
    }
    result: dict[str, Any] = {"schema": POPULATION_SCHEMA, "task_id": TASK_ID}
    all_ids: set[tuple[str, int, int]] = set()
    all_coefficients: dict[str, torch.Tensor] = {}
    accounting: dict[str, Any] = {}
    for cohort_name, count in counts.items():
        coefficients_by_class: list[torch.Tensor] = []
        labels_by_class: list[torch.Tensor] = []
        proposals_by_class: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        group_coefficients: list[torch.Tensor] = []
        for label in range(CLASS_COUNT):
            seed = seed_bases[cohort_name] + label
            sampled, proposal_ids, record = _sample_g1_class(
                inputs["prior"][label], inputs["scale"], label=label, seed=seed,
                count=count,
                proposal_limit=int(population_config["max_proposals_per_class_per_population"]),
                proposal_batch_size=int(population_config["proposal_batch_size"]),
            )
            coefficients_by_class.append(sampled)
            labels_by_class.append(torch.full((count,), label, dtype=torch.long))
            proposals_by_class.append(proposal_ids)
            rows.extend({"label": label, "sample_index_within_class": index,
                         "proposal_id": int(proposal_ids[index]), "normal_seed": seed}
                        for index in range(count))
            group_coefficients.append(sampled)
            accounting[f"{cohort_name}_{label}"] = record
            ids = {(cohort_name, label, int(value)) for value in proposal_ids.tolist()}
            _require(not (all_ids & ids), "population groups reused a proposal identity")
            all_ids.update(ids)
        coefficients = torch.cat(coefficients_by_class).contiguous()
        labels = torch.cat(labels_by_class)
        proposal_ids = torch.cat(proposals_by_class)
        _require(tuple(coefficients.shape) == (count * CLASS_COUNT, PORT_COUNT)
                 and tuple(labels.shape) == (count * CLASS_COUNT,)
                 and all(int((labels == label).sum()) == count for label in range(CLASS_COUNT)),
                 f"{cohort_name} G1 population is incomplete or imbalanced")
        group_coefficients_tensor = torch.stack(group_coefficients)
        all_coefficients[cohort_name] = coefficients
        result[cohort_name] = {
            "coefficients": coefficients,
            "labels": labels,
            "proposal_ids": proposal_ids,
            "rows": rows,
            "coefficients_by_class": group_coefficients_tensor,
            "seeds": [seed_bases[cohort_name] + label for label in range(CLASS_COUNT)],
        }
    _require(all(not torch.equal(all_coefficients[left], all_coefficients[right])
                 for left, right in (("train", "validation"), ("train", "holdout"),
                                     ("validation", "holdout"))),
             "train, validation, and holdout coefficient cohorts are not independent")
    result["proposal_accounting"] = accounting
    result["rejection_rule"] = "one saved class Gaussian; reject only total modal power over fixed bound"
    result["power_bound_w"] = base.MAX_MODAL_POWER_W
    return result


def _build_schedule(population: Mapping[str, Any], config: Mapping[str, Any]) -> torch.Tensor:
    train = population["train"]
    per_class = int(config["population"]["train_per_class"])
    generator = torch.Generator(device="cpu").manual_seed(int(config["population"]["schedule_seed"]))
    schedule = torch.empty((5, per_class, CLASS_COUNT), dtype=torch.long)
    for epoch in range(5):
        for label in range(CLASS_COUNT):
            start = label * per_class
            order = torch.randperm(per_class, generator=generator) + start
            schedule[epoch, :, label] = order
    _require(tuple(schedule.shape) == (5, 1000, 10), "balanced five-epoch schedule shape changed")
    for epoch in range(5):
        for label in range(CLASS_COUNT):
            expected = torch.arange(label * per_class, (label + 1) * per_class)
            _require(torch.equal(schedule[epoch, :, label].sort().values, expected),
                     f"schedule omitted or repeated class-{label} training rows in epoch {epoch}")
    return schedule


def _save_population(path: Path, population: Mapping[str, Any], schedule: torch.Tensor) -> None:
    payload = dict(population)
    payload["training_schedule"] = schedule
    _atomic_torch(path, payload)


def _feature_batch(coefficients: torch.Tensor, scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    normalized = coefficients.to(device=device, dtype=torch.complex128) / scale.to(
        device=device, dtype=torch.float64
    ).sqrt().unsqueeze(0)
    features = torch.view_as_real(normalized).reshape(coefficients.shape[0], REAL_DIM).float()
    _require(torch.isfinite(features).all().item(), "normalized complex G1 mapper features are nonfinite")
    return features


def _new_optical_model(checkpoint: Mapping[str, Any], grid: int, device: torch.device) -> base.SmallMatchedAdapter:
    model = base.SmallMatchedAdapter(grid, base.GAUSSIAN_OLS_MODEL)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys,
             f"C model weights did not load exactly at grid {grid}")
    model.to(device=device)
    for parameter in model.parameters():
        _require(parameter.is_floating_point(), "optical parameters must be real phase values")
        if parameter.dtype != torch.float64:
            parameter.data = parameter.detach().to(dtype=torch.float64)
        parameter.requires_grad_(False)
    _require(all(parameter.dtype == torch.float64 and not parameter.requires_grad
                 for parameter in model.parameters()),
             "frozen optics are not float64 or contain a trainable parameter")
    _require(all(not buffer.is_floating_point() or buffer.dtype == torch.float64
                 for buffer in model.buffers())
             and all(not torch.is_complex(buffer) or buffer.dtype == torch.complex128
                     for buffer in model.buffers()),
             "frozen optical buffers are not float64/complex128")
    _require(model.runtime.ols.basis_id == base.GAUSSIAN_OLS_MODEL
             and model.runtime.ols.mode_identity().get("transport") == "T=I_100 coefficient space",
             "runtime Gaussian OLS identity or transport changed")
    model.eval()
    return model


def _camera_pixel_area_m2(model: base.SmallMatchedAdapter) -> float:
    plane = model.runtime.camera_plane
    pitch = tuple(float(value) for value in plane.pixel_pitch_m)
    _require(len(pitch) == 2 and all(math.isfinite(value) and value > 0.0 for value in pitch),
             "frozen C camera plane has invalid pixel pitch")
    area = pitch[0] * pitch[1]
    _require(math.isclose(area, 6.25e-12, rel_tol=1.0e-12, abs_tol=0.0),
             f"frozen C camera pixel area changed: {area:.17g} m^2")
    return area


def _camera_w_per_pixel(raw_w_per_m2: torch.Tensor | np.ndarray, pixel_area_m2: float
                        ) -> torch.Tensor | np.ndarray:
    _require(math.isfinite(pixel_area_m2) and pixel_area_m2 > 0.0,
             "camera pixel area must be finite and positive", ValueError)
    if torch.is_tensor(raw_w_per_m2):
        _require(raw_w_per_m2.is_floating_point() and torch.isfinite(raw_w_per_m2).all().item()
                 and float(raw_w_per_m2.min()) >= 0.0,
                 "raw CMOS W/m^2 tensor must be finite, real and nonnegative", ValueError)
        return raw_w_per_m2 * pixel_area_m2
    values = np.asarray(raw_w_per_m2)
    _require(np.issubdtype(values.dtype, np.floating) and np.isfinite(values).all()
             and float(values.min()) >= 0.0,
             "raw CMOS W/m^2 array must be finite, real and nonnegative", ValueError)
    return values * pixel_area_m2


def _snapshot_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_state_unchanged(module: torch.nn.Module, before: Mapping[str, torch.Tensor], name: str) -> None:
    after = module.state_dict()
    _require(set(after) == set(before), f"{name} state keys changed")
    _require(all(torch.equal(after[key].detach().cpu(), before[key]) for key in before),
             f"{name} frozen parameters or buffers changed")


def _forward_actual(model: base.SmallMatchedAdapter, intensity: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(intensity.ndim == 4 and tuple(intensity.shape[1:]) == (1, 200, 200)
             and not torch.is_complex(intensity) and torch.isfinite(intensity).all().item()
             and float(intensity.min()) >= 0.0 and float(intensity.max()) <= 1.0,
             "production amplitude-SLM intensity input is malformed")
    result = model.forward_from_image(intensity.to(dtype=torch.float64))
    coefficients = result["ols_coefficients"]
    raw = result["raw_intensity_w_per_m2"]
    _require(tuple(coefficients.shape) == (intensity.shape[0], PORT_COUNT)
             and coefficients.dtype == torch.complex128
             and tuple(raw.shape) == (intensity.shape[0], 1, *PIXEL_SHAPE)
             and raw.dtype == torch.float64
             and torch.isfinite(coefficients.real).all().item()
             and torch.isfinite(coefficients.imag).all().item()
             and torch.isfinite(raw).all().item(),
             "actual production image-to-OLS-to-raw-CMOS forward is incomplete")
    return {"coefficients": coefficients, "raw_cmos": raw}


def _forward_direct(model: base.SmallMatchedAdapter, coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
    result = model.forward_from_ols(coefficients.to(dtype=torch.complex128))
    raw = result["raw_intensity_w_per_m2"]
    _require(tuple(raw.shape) == (coefficients.shape[0], 1, *PIXEL_SHAPE)
             and raw.dtype == torch.float64 and torch.isfinite(raw).all().item(),
             "direct target-OLS-to-frozen-Decoder raw-CMOS reference is incomplete")
    return {"raw_cmos": raw}


def _set_numerics() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _run_id(preflight_dir: Path) -> str:
    identity_path = preflight_dir / "preflight_identity.json"
    _require(identity_path.is_file(), "same-task preflight identity is missing")
    identity = _json(identity_path)
    _require(identity.get("task_id") == TASK_ID and isinstance(identity.get("run_id"), str),
             "preflight identity belongs to another task")
    return str(identity["run_id"])


def _source_semantics(config: Mapping[str, Any]) -> str:
    """Stable, human-readable execution semantics; no digest is an admission lock."""
    semantics = {
        "task_id": TASK_ID,
        "semantic_authority_id": config["semantic_authority_id"],
        "checkpoint": config["source"]["checkpoint"],
        "prior": config["source"]["fitted_gaussian_prior"],
        "fixed_C_step": config["optics"]["cumulative_step"],
        "populations": config["population"],
        "mapper": config["training"],
        "measurement": config["measurement"],
    }
    return json.dumps(semantics, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _research_identity(config: Mapping[str, Any], run_id: str, attempt_id: str | None = None) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{run_id}:continuous_G1_mapper",
        semantic_authority_id=str(config["semantic_authority_id"]),
        data_role="SYNTHETIC_G1_TRAIN_VALIDATION_HOLDOUT",
        config_identity=dict(config),
        model_identity={
            "frozen_optics": "C_seed_2026101501_cumulative_step_45000",
            "ols": "gaussian_s12_v1_T_equals_I_100",
            "mapper": MAPPER_ID,
            "evidence_tier": "diagnostic_only",
        },
        optimizer_identity={
            "kind": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999],
            "epsilon": 1.0e-8, "weight_decay": 0.0, "updates": 5000,
        },
        sampler_identity={
            "distribution": "one full-covariance Gaussian per class with fixed total-power rejection",
            "seed_bases": [2026102610, 2026102630, 2026102650],
            "per_class_counts": [1000, 200, 200],
            "proposal_limit_per_class_per_population": 1_000_000,
        },
        execution_source_semantics_id=_source_semantics(config),
        scientific_attempt_id=attempt_id,
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _allclose_tree(actual: Any, expected: Any, *, atol: float) -> tuple[bool, float]:
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        a = actual.detach().cpu()
        e = expected.detach().cpu()
        if a.shape != e.shape:
            return False, float("inf")
        if a.is_complex() or e.is_complex():
            delta = (a.to(torch.complex128) - e.to(torch.complex128)).abs()
        else:
            delta = (a.to(torch.float64) - e.to(torch.float64)).abs()
        maximum = float(delta.max()) if delta.numel() else 0.0
        return bool(torch.isfinite(delta).all() and maximum <= atol), maximum
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            return False, float("inf")
        results = [_allclose_tree(actual[key], expected[key], atol=atol) for key in actual]
        return all(passed for passed, _ in results), max((value for _, value in results), default=0.0)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False, float("inf")
        results = [_allclose_tree(left, right, atol=atol) for left, right in zip(actual, expected, strict=True)]
        return all(passed for passed, _ in results), max((value for _, value in results), default=0.0)
    if isinstance(actual, (int, float, np.number)) and isinstance(expected, (int, float, np.number)):
        delta = abs(float(actual) - float(expected))
        return math.isfinite(delta) and delta <= atol, delta
    return actual == expected, 0.0 if actual == expected else float("inf")


def _make_optimizer(mapper: AmplitudeMapper) -> torch.optim.Adam:
    return torch.optim.Adam(
        mapper.parameters(), lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0,
    )


def _train_step(
    mapper: AmplitudeMapper,
    optimizer: torch.optim.Adam,
    optical_model: base.SmallMatchedAdapter,
    coefficients: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    *,
    audit_parameter_updates: bool = False,
) -> dict[str, Any]:
    features = _feature_batch(coefficients, scale, device)
    target = coefficients.to(device=device, dtype=torch.complex128)
    before = ([parameter.detach().clone() for parameter in mapper.parameters()]
              if audit_parameter_updates else None)
    optimizer.zero_grad(set_to_none=True)
    amplitude = mapper(features)
    _require(amplitude.dtype == torch.float32 and torch.isfinite(amplitude).all().item()
             and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0,
             "mapper must emit finite continuous float32 amplitude in [0,1]")
    intensity = amplitude.square().to(dtype=torch.float64)
    production = _forward_actual(optical_model, intensity)
    predicted = production["coefficients"]
    del production
    losses = _phase_invariant_loss(predicted, target)
    loss = losses.mean()
    _require(torch.isfinite(loss).item(), "production coefficient objective is nonfinite")
    loss.backward()
    grad_norms: list[float] = []
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"trainable mapper parameter has no finite gradient: {name}")
        norm = float(torch.linalg.vector_norm(parameter.grad.detach().double()))
        _require(math.isfinite(norm) and norm > 0.0, f"mapper parameter did not receive a real gradient: {name}")
        grad_norms.append(norm)
    optimizer.step()
    if before is not None:
        _require(all(not torch.equal(old, new.detach())
                     for old, new in zip(before, mapper.parameters(), strict=True)),
                 "at least one mapper parameter tensor did not change during the preflight updates")
    return {
        "loss": float(loss.detach()),
        "mean_relative_coefficient_error": float(losses.detach().mean().sqrt()),
        "gradient_norms_by_parameter": grad_norms,
    }


def _balanced_batch(population: Mapping[str, Any], schedule: torch.Tensor, step_index: int) -> torch.Tensor:
    epoch, within_epoch = divmod(step_index, 1000)
    _require(0 <= epoch < 5 and 0 <= within_epoch < 1000, "training schedule cursor is outside five epochs")
    indices = schedule[epoch, within_epoch]
    coefficients = population["train"]["coefficients"][indices]
    labels = population["train"]["labels"][indices]
    _require(tuple(coefficients.shape) == (10, PORT_COUNT)
             and torch.equal(labels, torch.arange(CLASS_COUNT)),
             "logical training batch must contain exactly one G1 row per class")
    return coefficients


def _make_mapper_optimizer_state(mapper: AmplitudeMapper, optimizer: torch.optim.Adam) -> dict[str, Any]:
    return {
        "mapper": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


def _checkpoint_payload(
    mapper: AmplitudeMapper, optimizer: torch.optim.Adam, *, run_id: str, completed_updates: int,
    loss_history: Sequence[Mapping[str, Any]], schedule: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_g1_slm_input_mapping_checkpoint.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "checkpoint_step": completed_updates,
        "completed_updates": completed_updates,
        "next_schedule_cursor": completed_updates,
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "mapper_state": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
        "schedule_prefix_length": completed_updates,
        "schedule_identity": "saved_training_schedule.pt:5x1000x10",
        "next_batch_indices": (schedule[completed_updates // 1000, completed_updates % 1000].clone()
                                if completed_updates < 5000 else None),
        "last_loss": None if not loss_history else float(loss_history[-1]["loss"]),
        "loss_history_count": len(loss_history),
        "optimizer_state_deepcopied": True,
    }


def _save_and_reopen_checkpoint(
    path: Path, mapper: AmplitudeMapper, optimizer: torch.optim.Adam, *, run_id: str,
    completed_updates: int, loss_history: Sequence[Mapping[str, Any]], schedule: torch.Tensor,
) -> dict[str, Any]:
    payload = _checkpoint_payload(mapper, optimizer, run_id=run_id, completed_updates=completed_updates,
                                  loss_history=loss_history, schedule=schedule)
    _atomic_torch(path, payload)
    reopened = torch.load(path, map_location="cpu", weights_only=False)
    _require(reopened.get("task_id") == TASK_ID and reopened.get("run_id") == run_id
             and reopened.get("completed_updates") == completed_updates
             and reopened.get("next_schedule_cursor") == completed_updates
             and reopened.get("optimizer_state_deepcopied") is True,
             f"checkpoint readback identity or cursor mismatch: {path}")
    cpu_mapper = _new_mapper(torch.device("cpu"), 2026102670)
    cpu_mapper.load_state_dict(reopened["mapper_state"], strict=True)
    cpu_optimizer = _make_optimizer(cpu_mapper)
    cpu_optimizer.load_state_dict(copy.deepcopy(reopened["optimizer_state"]))
    _require(all(torch.equal(cpu_mapper.state_dict()[key], reopened["mapper_state"][key])
                 for key in reopened["mapper_state"]), "checkpoint mapper weights failed a fresh-process load")
    return reopened


def _restore_training_state(
    checkpoint_path: Path, mapper: AmplitudeMapper, optimizer: torch.optim.Adam, *, run_id: str,
    schedule: torch.Tensor,
) -> tuple[int, list[dict[str, Any]]]:
    _require(checkpoint_path.is_file(), f"explicit resume checkpoint is missing: {checkpoint_path}", FileNotFoundError)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == "lightgen_g1_slm_input_mapping_checkpoint.v1"
             and saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id,
             "explicit recovery checkpoint identity differs from this attempt")
    _require(saved.get("optimizer_state_deepcopied") is True,
             "explicit recovery checkpoint does not prove a deep-copied Adam state")
    completed = int(saved.get("completed_updates", -1))
    expected_next = (schedule[completed // 1000, completed % 1000]
                     if completed < 5000 else None)
    saved_next = saved.get("next_batch_indices")
    next_matches = ((expected_next is None and saved_next is None)
                    or (expected_next is not None and torch.is_tensor(saved_next)
                        and torch.equal(expected_next, saved_next)))
    _require(0 <= completed <= 5000 and saved.get("checkpoint_step") == completed
             and saved.get("next_schedule_cursor") == completed
             and saved.get("schedule_identity") == "saved_training_schedule.pt:5x1000x10"
             and saved.get("schedule_prefix_length") == completed and next_matches,
             "explicit checkpoint update count or sampler cursor is invalid")
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    torch.random.set_rng_state(saved["torch_cpu_rng"].cpu())
    torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    history_path = checkpoint_path.parent.parent / "training_history.json"
    history = _json(history_path).get("updates", []) if history_path.is_file() else []
    _require(len(history) >= completed,
             "training history is incomplete relative to the explicit recovery checkpoint")
    history = [dict(item) for item in history[:completed]]
    _require(all(item.get("completed_update") == index + 1
                 for index, item in enumerate(history)),
             "training history does not form the completed checkpoint prefix")
    optimizer_steps = [int(torch.as_tensor(value["step"]).item()) for value in optimizer.state.values()
                       if isinstance(value, Mapping) and "step" in value]
    if completed:
        _require(len(optimizer_steps) == len(list(mapper.parameters()))
                 and all(step == completed for step in optimizer_steps),
                 "explicit recovery Adam state does not match the mapper checkpoint step")
    return completed, history


def _training_update_with_index(
    mapper: AmplitudeMapper, optimizer: torch.optim.Adam, optical_model: base.SmallMatchedAdapter,
    population: Mapping[str, Any], schedule: torch.Tensor, scale: torch.Tensor, device: torch.device,
    step_index: int, *, audit_parameter_updates: bool = False,
) -> dict[str, Any]:
    batch = _balanced_batch(population, schedule, step_index)
    return _train_step(mapper, optimizer, optical_model, batch, scale, device,
                       audit_parameter_updates=audit_parameter_updates)


def _preflight_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["output"]["preflight_root"])


def _attempt_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["output"]["experiment_root"])


def _load_saved_population(preflight_dir: Path) -> tuple[dict[str, Any], torch.Tensor]:
    path = preflight_dir / "fixed_population_and_schedule.pt"
    _require(path.is_file(), "fixed G1 populations and sample schedule are missing; rerun the preflight")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == POPULATION_SCHEMA
             and payload.get("task_id") == TASK_ID,
             "saved fixed G1 population belongs to another task")
    schedule = payload.pop("training_schedule")
    _require(tuple(schedule.shape) == (5, 1000, 10), "saved G1 training schedule is malformed")
    return payload, schedule


def _save_population_and_schedule(preflight_dir: Path, population: Mapping[str, Any], schedule: torch.Tensor) -> None:
    payload = dict(population)
    payload["training_schedule"] = schedule
    _atomic_torch(preflight_dir / "fixed_population_and_schedule.pt", payload)


def _baseline_replay(config: Mapping[str, Any], inputs: Mapping[str, Any],
                     preflight_dir: Path, device: torch.device) -> dict[str, Any]:
    commands = inputs["commands"]
    identities = commands["sample_identities"]
    first_by_class: dict[int, int] = {}
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping) or identity.get("method") != "G1":
            continue
        label = identity.get("requested_label", identity.get("label"))
        _require(isinstance(label, int) and label in range(CLASS_COUNT),
                 "saved G1 replay row is missing its class identity")
        first_by_class.setdefault(int(label), index)
    _require(set(first_by_class) == set(range(CLASS_COUNT)),
             "saved G1 command replay does not contain the first row for all ten classes")
    models = {grid: _new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
    state_before = {grid: _snapshot_state(model) for grid, model in models.items()}
    rows: list[dict[str, Any]] = []
    for label in range(CLASS_COUNT):
        index = first_by_class[label]
        amplitude = commands["amplitude_command"][index:index + 1].to(device=device, dtype=torch.float64)
        intensity = commands["intensity_command"][index:index + 1].to(device=device, dtype=torch.float64)
        _require(torch.allclose(amplitude.square(), intensity, rtol=0.0, atol=1.0e-12),
                 f"saved G1 replay command does not preserve amplitude/intensity semantics for class {label}")
        target = inputs["compact"]["target_coefficients"][index:index + 1].to(device=device,
                                                                                dtype=torch.complex128)
        row: dict[str, Any] = {"method": "G1", "label": label,
                               "proposal_id": int(identities[index].get("proposal_id", -1)),
                               "source_index": index, "grids": {}}
        for grid in GRIDS:
            produced = _forward_actual(models[grid], intensity)
            direct = _forward_direct(models[grid], target)
            saved_grid = inputs["compact"]["grids"][str(grid)]
            saved_actual_coefficients = saved_grid["actual_coefficients"][index:index + 1].to(device)
            saved_actual_raw = saved_grid["actual_raw_cmos_intensity_w_per_m2"][index:index + 1].to(device)
            saved_direct_raw = saved_grid["direct_raw_cmos_intensity_w_per_m2"][index:index + 1].to(device)
            saved_direct_coefficients = saved_grid["direct_coefficients"][index:index + 1].to(device)
            actual_replay = {
                "coefficient_relative_error": _numpy_phase_error(
                    produced["coefficients"][0].detach().cpu().numpy(),
                    saved_actual_coefficients[0].detach().cpu().numpy(),
                )[1],
                "raw_cmos_relative_error": _numpy_relative_error(
                    produced["raw_cmos"][0].detach().cpu().numpy(), saved_actual_raw[0].detach().cpu().numpy()
                ),
                "direct_coefficient_relative_error": _numpy_phase_error(
                    target[0].detach().cpu().numpy(), saved_direct_coefficients[0].detach().cpu().numpy()
                )[1],
                "direct_raw_cmos_relative_error": _numpy_relative_error(
                    direct["raw_cmos"][0].detach().cpu().numpy(), saved_direct_raw[0].detach().cpu().numpy()
                ),
            }
            _require(actual_replay["coefficient_relative_error"] <= 2.0e-6
                     and actual_replay["raw_cmos_relative_error"] <= 2.0e-6
                     and actual_replay["direct_coefficient_relative_error"] <= 2.0e-6
                     and actual_replay["direct_raw_cmos_relative_error"] <= 2.0e-6,
                     f"zero-update C production replay differs from its saved reference at {grid}/{label}")
            row["grids"][str(grid)] = actual_replay
        _assert_state_unchanged(models[2048], state_before[2048], "2048 frozen C optics")
        _assert_state_unchanged(models[4096], state_before[4096], "4096 frozen C optics")
        rows.append(row)
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = {
        "schema": "lightgen_g1_slm_zero_update_baseline.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "source_C_run_id": inputs["checkpoint"]["run_id"],
        "population": "first saved G1 command for each class; no new sampling",
        "rows": rows,
        "all_rows_replayed_both_grids": len(rows) == 10,
        "trainable_model_updates": 0,
        "optimizer_calls": 0,
    }
    _atomic_json(preflight_dir / "zero_update_baseline.json", summary)
    return summary


def _production_gradient_check(
    mapper: AmplitudeMapper, optical_model: base.SmallMatchedAdapter,
    coefficients: torch.Tensor, scale: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    feature = _feature_batch(coefficients[:1], scale, device)
    amplitude = mapper(feature).detach().double().requires_grad_(True)
    target = coefficients[:1].to(device=device, dtype=torch.complex128)
    measured = _forward_actual(optical_model, amplitude.square())
    objective = _phase_invariant_loss(measured["coefficients"], target).mean()
    derivative = torch.autograd.grad(objective, amplitude)[0]
    generator = torch.Generator(device="cpu").manual_seed(2026102671)
    flat_indices = torch.randperm(PIXEL_COUNT, generator=generator)[:64]
    signs = torch.where(torch.arange(64) % 2 == 0, 1.0, -1.0).to(torch.float64)
    direction = torch.zeros(PIXEL_COUNT, dtype=torch.float64, device=device)
    direction[flat_indices.to(device)] = signs.to(device) / 8.0
    epsilon = 1.0e-4
    amp0 = amplitude.detach()
    plus = _forward_actual(optical_model, (amp0 + epsilon * direction.reshape(1, 1, *PIXEL_SHAPE)).square())
    minus = _forward_actual(optical_model, (amp0 - epsilon * direction.reshape(1, 1, *PIXEL_SHAPE)).square())
    plus_loss = _phase_invariant_loss(plus["coefficients"], target).mean()
    minus_loss = _phase_invariant_loss(minus["coefficients"], target).mean()
    finite_difference = float((plus_loss - minus_loss) / (2.0 * epsilon))
    autograd_value = float((derivative * direction.reshape(1, 1, *PIXEL_SHAPE)).sum())
    relative_error = abs(finite_difference - autograd_value) / max(
        abs(finite_difference), abs(autograd_value), 1.0e-12
    )
    _require(math.isfinite(relative_error) and relative_error <= 5.0e-3,
             "actual production optical derivative disagrees with the complex128 central difference")
    return {
        "executed": True,
        "passed": True,
        "producer": "mapper_amplitude_squared_to_frozen_C_encoder_OLS_coefficients",
        "direction_support_pixels": 64,
        "epsilon": epsilon,
        "finite_difference": finite_difference,
        "autograd": autograd_value,
        "relative_error": relative_error,
    }


def _single_record_passes(actual_coeff: np.ndarray, target_coeff: np.ndarray,
                          actual_cmos: np.ndarray, target_cmos: np.ndarray) -> tuple[bool, dict[str, float]]:
    coefficient_error = _numpy_phase_error(actual_coeff, target_coeff)[1]
    cmos_error = _numpy_relative_error(actual_cmos, target_cmos)
    return coefficient_error <= MEASUREMENT_LIMIT and cmos_error <= MEASUREMENT_LIMIT, {
        "coefficient_relative_error": coefficient_error,
        "native_cmos_relative_error": cmos_error,
    }


def _production_negative_controls(
    mapper: AmplitudeMapper, optical_model: base.SmallMatchedAdapter,
    inputs: Mapping[str, Any], device: torch.device,
) -> dict[str, Any]:
    del mapper
    commands = inputs["commands"]
    identities = commands["sample_identities"]
    candidate = next((index for index, identity in enumerate(identities)
                      if identity.get("method") == "G1"
                      and identity.get("requested_label", identity.get("label")) == 0), None)
    _require(candidate is not None, "saved positive-control G1 command for class 0 is missing")
    amplitude = commands["amplitude_command"][candidate:candidate + 1].to(device=device, dtype=torch.float64)
    intensity = commands["intensity_command"][candidate:candidate + 1].to(device=device, dtype=torch.float64)
    target = inputs["compact"]["target_coefficients"][candidate:candidate + 1].to(
        device=device, dtype=torch.complex128
    )
    direct = _forward_direct(optical_model, target)
    correct = _forward_actual(optical_model, intensity)
    correct_pass, correct_errors = _single_record_passes(
        correct["coefficients"][0].detach().cpu().numpy(), target[0].cpu().numpy(),
        correct["raw_cmos"][0].detach().cpu().numpy(), direct["raw_cmos"][0].detach().cpu().numpy(),
    )
    _require(correct_pass, "positive production control failed its decision consumer")

    wrong_target = target.clone()
    wrong_target[:, 1] *= torch.exp(torch.tensor(0.4j, dtype=torch.complex128, device=device))
    wrong_target_direct = _forward_direct(optical_model, wrong_target)
    wrong_target_pass, wrong_target_errors = _single_record_passes(
        correct["coefficients"][0].detach().cpu().numpy(), wrong_target[0].cpu().numpy(),
        correct["raw_cmos"][0].detach().cpu().numpy(), wrong_target_direct["raw_cmos"][0].detach().cpu().numpy(),
    )
    _require(not wrong_target_pass,
             "final consumer accepted a real production output paired with an incorrect target")

    wrong_encoded = _forward_actual(optical_model, amplitude)
    wrong_encoding_pass, wrong_encoding_errors = _single_record_passes(
        wrong_encoded["coefficients"][0].detach().cpu().numpy(), target[0].cpu().numpy(),
        wrong_encoded["raw_cmos"][0].detach().cpu().numpy(), direct["raw_cmos"][0].detach().cpu().numpy(),
    )
    _require(not wrong_encoding_pass,
             "final consumer accepted amplitude sent as intensity instead of its square")
    return {
        "executed": True,
        "passed": True,
        "decision_consumer": "_single_record_passes (also used by saved-evidence consumer)",
        "positive_production_control": correct_errors,
        "wrong_target_rejected": {"rejected": not wrong_target_pass, **wrong_target_errors},
        "amplitude_instead_of_intensity_rejected": {
            "rejected": not wrong_encoding_pass, **wrong_encoding_errors,
        },
    }


def _engineering_worker(config_path: Path, preflight_dir: Path, checkpoint_path: Path,
                        output_path: Path) -> None:
    config = _load_config(config_path)
    _set_numerics()
    _require(torch.cuda.is_available(), "engineering recovery worker requires CUDA")
    device = torch.device("cuda:0")
    inputs = _validate_source_inputs(config)
    population, schedule = _load_saved_population(preflight_dir)
    run_id = _run_id(preflight_dir)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(saved.get("completed_updates") == 3 and saved.get("run_id") == run_id,
             "engineering recovery worker requires the explicit three-update checkpoint")
    mapper = _new_mapper(device, int(config["population"]["initialization_seed"]))
    optimizer = _make_optimizer(mapper)
    model = _new_optical_model(inputs["checkpoint"], 2048, device)
    completed, history = _restore_training_state(
        checkpoint_path, mapper, optimizer, run_id=run_id, schedule=schedule,
    )
    _require(completed == 3 and len(history) == 3,
             "engineering recovery worker did not restore the complete three-step prefix")
    step4_indices = schedule[0, 3].detach().cpu().clone()
    result = _training_update_with_index(mapper, optimizer, model, population, schedule,
                                         inputs["scale"], device, 3)
    result["completed_update"] = 4
    history.append(result)
    result_payload = _checkpoint_payload(mapper, optimizer, run_id=run_id, completed_updates=4,
                                         loss_history=history, schedule=schedule)
    result_payload["step4_loss"] = result["loss"]
    result_payload["loss_history"] = history
    result_payload["step4_schedule_indices"] = step4_indices
    result_payload["engineering_adam_calls"] = 1
    _atomic_torch(output_path, result_payload)
    del model, mapper, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def _engineering_worker_command(config_path: Path, preflight_dir: Path,
                                checkpoint_path: Path, output_path: Path) -> tuple[list[str], Path]:
    project_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable, "-B", "-m", MODULE_PATH,
        "--mode", "engineering-worker",
        "--config", str(config_path.resolve()), "--preflight-dir", str(preflight_dir.resolve()),
        "--checkpoint", str(checkpoint_path.resolve()), "--worker-output", str(output_path.resolve()),
    ]
    return command, project_root


def _assert_engineering_recovery_pair(
    live_state: Mapping[str, Any], live_loss: float, live_rng: Mapping[str, Any],
    live_history: Sequence[Mapping[str, Any]], worker: Mapping[str, Any], *,
    run_id: str, schedule: torch.Tensor,
) -> dict[str, Any]:
    _require(worker.get("task_id") == TASK_ID and worker.get("run_id") == run_id
             and worker.get("completed_updates") == 4 and worker.get("checkpoint_step") == 4
             and worker.get("next_schedule_cursor") == 4
             and worker.get("engineering_adam_calls") == 1,
             "independent recovery worker output identity or cursor is invalid")
    expected_indices = schedule[0, 3].detach().cpu()
    _require(torch.is_tensor(worker.get("step4_schedule_indices"))
             and torch.equal(worker["step4_schedule_indices"], expected_indices),
             "recovery worker did not execute the exact saved fourth-step batch")
    worker_history = worker.get("loss_history")
    _require(isinstance(worker_history, list) and len(worker_history) == len(live_history) == 4
             and all(item.get("completed_update") == index + 1
                     for index, item in enumerate(worker_history))
             and all(item.get("completed_update") == index + 1
                     for index, item in enumerate(live_history)),
             "live and recovered update histories are incomplete or misaligned")
    live_model_optimizer = live_state
    resumed_model_optimizer = {
        "mapper": worker["mapper_state"],
        "optimizer": worker["optimizer_state"],
    }
    model_pass, max_model_delta = _allclose_tree(
        live_model_optimizer["mapper"], resumed_model_optimizer["mapper"],
        atol=RECOVERY_ABSOLUTE_TOLERANCE,
    )
    optimizer_pass, max_optimizer_delta = _allclose_tree(
        live_model_optimizer["optimizer"], resumed_model_optimizer["optimizer"],
        atol=RECOVERY_ABSOLUTE_TOLERANCE,
    )
    history_pass, max_history_delta = _allclose_tree(
        list(live_history), worker_history, atol=RECOVERY_ABSOLUTE_TOLERANCE,
    )
    loss_delta = abs(float(live_loss) - float(worker["step4_loss"]))
    rng_pass, rng_delta = _allclose_tree(
        live_rng, {"cpu": worker["torch_cpu_rng"], "cuda": worker["torch_cuda_rng"]}, atol=0.0,
    )
    _require(model_pass and optimizer_pass and history_pass and rng_pass
             and loss_delta <= RECOVERY_ABSOLUTE_TOLERANCE,
             "live fourth update differs from the independently restored production step")
    return {
        "model_max_absolute_difference": max_model_delta,
        "optimizer_max_absolute_difference": max_optimizer_delta,
        "history_max_absolute_difference": max_history_delta,
        "loss_absolute_difference": loss_delta,
        "rng_equal": rng_pass,
        "rng_max_absolute_difference": rng_delta,
        "schedule_equal": True,
        "schedule_step4_indices": expected_indices.tolist(),
    }


def _engineering_recovery(
    config_path: Path, config: Mapping[str, Any], inputs: Mapping[str, Any], population: Mapping[str, Any],
    schedule: torch.Tensor, preflight_dir: Path, engineering_dir: Path, device: torch.device,
) -> dict[str, Any]:
    _require(not engineering_dir.exists(),
             f"engineering recovery output already exists: {engineering_dir}", FileExistsError)
    engineering_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = engineering_dir / "checkpoints"
    checkpoint_dir.mkdir()
    run_id = _run_id(preflight_dir)
    model = _new_optical_model(inputs["checkpoint"], 2048, device)
    optical_before = _snapshot_state(model)
    mapper = _new_mapper(device, int(config["population"]["initialization_seed"]))
    optimizer = _make_optimizer(mapper)
    before_parameters = [parameter.detach().clone() for parameter in mapper.parameters()]
    preflight_history: list[dict[str, Any]] = []
    losses: list[float] = []
    for step_index in range(3):
        record = _training_update_with_index(
            mapper, optimizer, model, population, schedule, inputs["scale"], device,
            step_index, audit_parameter_updates=True,
        )
        record["completed_update"] = step_index + 1
        preflight_history.append(record)
        losses.append(float(record["loss"]))
    _require(all(not torch.equal(before, after.detach())
                 for before, after in zip(before_parameters, mapper.parameters(), strict=True)),
             "three-step preflight did not update every mapper parameter tensor")
    history_path = engineering_dir / "training_history.json"
    _atomic_json(history_path, {"task_id": TASK_ID, "run_id": run_id, "updates": preflight_history})
    rng_before_save = {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }
    state_after_three = _make_mapper_optimizer_state(mapper, optimizer)
    saved_after_three = _checkpoint_payload(mapper, optimizer, run_id=run_id, completed_updates=3,
                                            loss_history=preflight_history, schedule=schedule)
    engineering_checkpoint = checkpoint_dir / "checkpoint_step_000003.pt"
    _atomic_torch(engineering_checkpoint, saved_after_three)
    reopened = torch.load(engineering_checkpoint, map_location="cpu", weights_only=False)
    _require(reopened.get("completed_updates") == 3 and reopened.get("next_schedule_cursor") == 3,
             "engineering three-step checkpoint did not read back with its cursor")
    save_rng_after = {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }
    rng_save_pass, _ = _allclose_tree(rng_before_save, save_rng_after, atol=0.0)
    model_checkpoint_pass, model_checkpoint_delta = _allclose_tree(
        state_after_three["mapper"], reopened["mapper_state"], atol=0.0,
    )
    optimizer_checkpoint_pass, optimizer_checkpoint_delta = _allclose_tree(
        state_after_three["optimizer"], reopened["optimizer_state"], atol=0.0,
    )
    _require(rng_save_pass and model_checkpoint_pass and optimizer_checkpoint_pass,
             "serialization changed the live RNG, mapper or deep-copied Adam state")

    step4_indices = schedule[0, 3].detach().cpu().clone()
    live_result = _training_update_with_index(
        mapper, optimizer, model, population, schedule, inputs["scale"], device, 3,
    )
    live_result["completed_update"] = 4
    preflight_history.append(live_result)
    live_state = _make_mapper_optimizer_state(mapper, optimizer)
    live_rng = {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }
    direct_checkpoint = _checkpoint_payload(mapper, optimizer, run_id=run_id, completed_updates=4,
                                            loss_history=preflight_history, schedule=schedule)
    direct_path = checkpoint_dir / "checkpoint_step_000004_live.pt"
    _atomic_torch(direct_path, direct_checkpoint)
    direct_rng_after_save = {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }
    direct_save_rng_pass, _ = _allclose_tree(live_rng, direct_rng_after_save, atol=0.0)
    _require(direct_save_rng_pass,
             "writing the live fourth-step engineering checkpoint changed the process RNG")

    worker_output = engineering_dir / "worker_step4_restored.pt"
    command, project_root = _engineering_worker_command(
        config_path, preflight_dir, engineering_checkpoint, worker_output,
    )
    worker = subprocess.run(command, cwd=project_root, capture_output=True, text=True, check=False)
    (engineering_dir / "engineering_recovery_worker.stdout.log").write_text(worker.stdout, encoding="utf-8")
    (engineering_dir / "engineering_recovery_worker.stderr.log").write_text(worker.stderr, encoding="utf-8")
    _require(worker.returncode == 0 and worker_output.is_file(),
             "independent-process recovery step failed; inspect preserved worker logs")
    resumed = torch.load(worker_output, map_location="cpu", weights_only=False)
    _require(torch.equal(resumed.get("step4_schedule_indices"), step4_indices),
             "recovered fourth step used a different saved sample schedule")
    pair = _assert_engineering_recovery_pair(
        live_state, float(live_result["loss"]), live_rng, preflight_history,
        resumed, run_id=run_id, schedule=schedule,
    )
    _assert_state_unchanged(model, optical_before, "preflight frozen C optics")
    result = {
        "executed": True,
        "passed": True,
        "independent_process": True,
        "engineering_adam_calls": 5,
        "first_process_updates": 4,
        "recovered_process_updates": 1,
        "frozen_model_updates": 0,
        "completed_updates_after_resume": int(resumed["completed_updates"]),
        **pair,
        "save_preserved_rng": rng_save_pass and direct_save_rng_pass,
        "step3_checkpoint_model_max_absolute_difference": model_checkpoint_delta,
        "step3_checkpoint_optimizer_max_absolute_difference": optimizer_checkpoint_delta,
        "step3_history_count": 3,
        "step4_history_count": len(preflight_history),
        "formal_restore_function": "_restore_training_state",
        "tolerance": RECOVERY_ABSOLUTE_TOLERANCE,
        "three_update_losses": losses,
        "live_step4_loss": float(live_result["loss"]),
        "restored_step4_loss": float(resumed["step4_loss"]),
        "step4_adam_call_count": 2,
    }
    _atomic_json(engineering_dir / "engineering_recovery.json", result)
    del model, mapper, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _record_preflight_failure(preflight_dir: Path, run_id: str, error: Exception) -> None:
    failures = sorted(preflight_dir.glob("preflight_failure_*.json"))
    record = {
        "schema": "lightgen_g1_slm_input_mapping_preflight_failure.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "status": "BLOCKED_ENGINEERING_OR_INPUT",
        "error_type": type(error).__name__,
        "error": str(error),
        "failure_sequence": len(failures) + 1,
        "utc_recorded_at": time.time(),
        "scientific_updates": 0,
    }
    _atomic_json(preflight_dir / f"preflight_failure_{len(failures) + 1:02d}.json", record)


def _run_preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    preflight_dir = _preflight_dir(config)
    preflight_dir.mkdir(parents=True, exist_ok=True)
    identity_path = preflight_dir / "preflight_identity.json"
    review_path = preflight_dir / "implementation_review.json"
    _require(review_path.is_file(), "separate implementation review record is required before preflight",
             FileNotFoundError)
    review = _json(review_path)
    _require(review.get("schema") == "lightgen_g1_slm_input_mapping_implementation_review.v1"
             and review.get("task_id") == TASK_ID
             and review.get("status") == "REVIEWED"
             and review.get("boundary", {}).get("preflight_and_training_authority") is False,
             "implementation review record is incomplete or incorrectly claims execution authority")
    if identity_path.is_file():
        identity = _json(identity_path)
        _require(identity.get("task_id") == TASK_ID,
                 "fixed preflight directory belongs to a different task")
        _require(not (preflight_dir / "preflight_receipt.json").is_file(),
                 "this task already has a READY preflight receipt")
        run_id = str(identity["run_id"])
    else:
        existing_names = {path.name for path in preflight_dir.iterdir()}
        _require(existing_names <= {"implementation_review.json"},
                 "fixed preflight directory contains unowned files without a task identity")
        run_id = uuid.uuid4().hex
        _atomic_json(identity_path, {
            "schema": "lightgen_g1_slm_input_mapping_preflight_identity.v1",
            "task_id": TASK_ID,
            "run_id": run_id,
            "status": "PREPARING",
            "scientific_attempt_opened": False,
        })
    try:
        _set_numerics()
        shutil.copy2(config_path, preflight_dir / "run_config.yaml")
        shutil.copy2(Path(__file__).resolve(), preflight_dir / "runner_source.py")
        inputs = _validate_source_inputs(config)
        _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized AutoDL preflight")
        device = torch.device("cuda:0")
        free_before, total_gpu = torch.cuda.mem_get_info(device)
        disk = shutil.disk_usage(preflight_dir)
        _atomic_json(preflight_dir / "input_lineage.json", {
            "task_id": TASK_ID,
            "source_checkpoint": str(inputs["checkpoint_path"]),
            "source_run_id": inputs["checkpoint"]["run_id"],
            "branch": inputs["checkpoint"]["branch"],
            "cumulative_step": inputs["checkpoint"]["completed_global_step"],
            "optimizer_steps": 45_000,
            "prior_path": str(_resolve(config["source"]["fitted_gaussian_prior"])),
            "prior_classes": list(range(CLASS_COUNT)),
            "prior_covariance_dimensions": [REAL_DIM, REAL_DIM],
            "prior_fit_this_task": False,
            "new_mnist_raw_reads": 0,
            "C_historical_status_preserved": "VALID_FAILED",
            "hardware_acceptance": False,
        })
        baseline = _baseline_replay(config, inputs, preflight_dir, device)
        population = _sample_population(config, inputs)
        schedule = _build_schedule(population, config)
        population_path = preflight_dir / "fixed_population_and_schedule.pt"
        if population_path.is_file():
            previous = torch.load(population_path, map_location="cpu", weights_only=False)
            _require(torch.equal(previous["train"]["coefficients"], population["train"]["coefficients"])
                     and torch.equal(previous["validation"]["coefficients"],
                                     population["validation"]["coefficients"])
                     and torch.equal(previous["holdout"]["coefficients"], population["holdout"]["coefficients"])
                     and torch.equal(previous["training_schedule"], schedule),
                     "fixed population or schedule differs from the earlier same-task preflight")
        else:
            _save_population_and_schedule(preflight_dir, population, schedule)

        source_checkpoint = inputs["checkpoint"]
        model = _new_optical_model(source_checkpoint, 2048, device)
        optical_before = _snapshot_state(model)
        mapper = _new_mapper(device, int(config["population"]["initialization_seed"]))
        mapper_before = {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()}
        _atomic_torch(preflight_dir / "mapper_initialization.pt", {
            "schema": "lightgen_g1_slm_mapper_initialization.v1",
            "task_id": TASK_ID,
            "run_id": run_id,
            "seed": int(config["population"]["initialization_seed"]),
            "mapper_id": MAPPER_ID,
            "state_dict": mapper_before,
        })
        optimizer = _make_optimizer(mapper)
        first_train_row = population["train"]["coefficients"][0:1]
        gradient_check = _production_gradient_check(mapper, model, first_train_row, inputs["scale"], device)
        negative_controls = _production_negative_controls(mapper, model, inputs, device)
        recovery = _engineering_recovery(
            config_path, config, inputs, population, schedule, preflight_dir,
            preflight_dir / "engineering_recovery_recheck", device,
        )
        _require(all(torch.equal(mapper_before[key], mapper.state_dict()[key].detach().cpu())
                     for key in mapper_before),
                 "preflight engineering calls changed the initial formal mapper instance")
        _assert_state_unchanged(model, optical_before, "2048 frozen C optics after preflight")
        _require(torch.cuda.is_available(), "CUDA became unavailable before completing representative preflight")
        free_after, _ = torch.cuda.mem_get_info(device)
        _require(baseline["all_rows_replayed_both_grids"] is True
                 and gradient_check["passed"] is True
                 and negative_controls["passed"] is True
                 and recovery["passed"] is True,
                 "one or more executed engineering preflight checks did not pass")
        correctness = {
            "status": "PASS", "passed": True,
            "source_lineage": True,
            "zero_update_first_G1_per_class_replay_both_grids": True,
            "independent_numpy_complex_metric_and_global_phase": _numeric_math_checks(),
            "actual_production_gradient_central_difference": gradient_check,
            "real_production_wrong_target_and_amplitude_intensity_negative_controls": negative_controls,
            "sample_population_and_balanced_schedule": True,
            "G1_only_single_gaussian_power_rejection": True,
            "training_rows_class_counts": {str(y): 1000 for y in range(CLASS_COUNT)},
            "validation_rows_class_counts": {str(y): 200 for y in range(CLASS_COUNT)},
            "holdout_rows_class_counts": {str(y): 200 for y in range(CLASS_COUNT)},
            "new_MNIST_reads": 0,
        }
        resource = {
            "status": "PASS", "passed": True,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device),
            "free_cuda_memory_before_bytes": int(free_before),
            "free_cuda_memory_after_bytes": int(free_after),
            "total_cuda_memory_bytes": int(total_gpu),
            "gpu_memory_values_are_observations": True,
            "preflight_directory_free_disk_bytes": int(disk.free),
            "preflight_directory_total_disk_bytes": int(disk.total),
            "production_cuda_three_step_and_recovery_completed": True,
            "model_updates_in_formal_trial": 0,
        }
        implementation = {
            "status": "PASS", "passed": True,
            "selected_engine": "frozen_C_2048_full_production_image_to_OLS_forward",
            "engine": "frozen_C_2048_full_production_image_to_OLS_forward",
            "mapper_architecture": MAPPER_ID,
            "mapper_parameters_updated_in_engineering_trial": True,
            "all_four_mapper_parameter_tensors_updated": True,
            "frozen_optics_and_evaluators": True,
            "checkpoint_save_reload_and_independent_process_recovery": recovery,
            "formal_adam_updates": 0,
        }
        control = ResearchRunControl(
            identity=_research_identity(config, run_id),
            authorization_id=str(config["semantic_authority_id"]),
        )
        receipt = control.make_ready_receipt(
            correctness=correctness,
            resource=resource,
            implementation=implementation,
            selected_engine="frozen_C_2048_full_production_image_to_OLS_forward",
            resource_envelope={"observed_disk_free_bytes": int(disk.free),
                               "observed_cuda_free_bytes_before": int(free_before)},
            created_at=time.time(),
        )
        control.mark_preflight_ready(receipt)
        _atomic_json(preflight_dir / "preflight_receipt.json", receipt.to_dict())
        _atomic_json(preflight_dir / "preflight_summary.json", {
            "schema": "lightgen_g1_slm_input_mapping_preflight.v1",
            "task_id": TASK_ID, "run_id": run_id, "status": PREFLIGHT_READY,
            "baseline": baseline, "gradient_check": gradient_check,
            "negative_controls": negative_controls, "engineering_recovery": recovery,
            "formal_scientific_updates": 0,
            "ready_receipt_path": str(preflight_dir / "preflight_receipt.json"),
            "scientific_attempt_opened": False,
        })
        return receipt.to_dict()
    except Exception as error:
        _record_preflight_failure(preflight_dir, run_id, error)
        _atomic_json(identity_path, {
            **_json(identity_path), "status": "PREFLIGHT_BLOCKED",
            "latest_error_type": type(error).__name__, "latest_error": str(error),
        })
        raise


def _load_ready_receipt(config: Mapping[str, Any], preflight_dir: Path) -> PreflightReceipt:
    path = preflight_dir / "preflight_receipt.json"
    _require(path.is_file(), "the same task has no executed READY preflight receipt", FileNotFoundError)
    receipt = PreflightReceipt.from_mapping(_json(path))
    _require(receipt.status == PREFLIGHT_READY
             and receipt.authorization_id == config["semantic_authority_id"]
             and receipt.source_semantics_id == _source_semantics(config),
             "READY receipt belongs to another authority or execution semantics")
    for key in ("CORRECTNESS_INVARIANTS", "RESOURCE_ENVELOPE", "IMPLEMENTATION_SELECTION"):
        _require(receipt.checks[key].get("passed") is True,
                 f"READY receipt lacks an executed passing check group: {key}")
    _require(receipt.checks["CORRECTNESS_INVARIANTS"].get(
        "real_production_wrong_target_and_amplitude_intensity_negative_controls", {}).get("passed") is True
        and receipt.checks["IMPLEMENTATION_SELECTION"].get("checkpoint_save_reload_and_independent_process_recovery", {}).get(
            "passed") is True,
        "READY receipt is missing production counterfactual or recovery evidence")
    return receipt


def _load_attempt_population(attempt_dir: Path) -> tuple[dict[str, Any], torch.Tensor]:
    path = attempt_dir / "fixed_population_and_schedule.pt"
    _require(path.is_file(), "fixed G1 population is missing from the active attempt", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == POPULATION_SCHEMA
             and payload.get("task_id") == TASK_ID,
             "fixed G1 population in the attempt has another identity")
    schedule = payload.pop("training_schedule")
    _require(tuple(schedule.shape) == (5, 1000, CLASS_COUNT),
             "fixed G1 sample schedule in the attempt has changed")
    return payload, schedule


def _save_training_trace(attempt_dir: Path, history: Sequence[Mapping[str, Any]],
                         validations: Sequence[Mapping[str, Any]]) -> None:
    _atomic_json(attempt_dir / "training_history.json", {
        "schema": "lightgen_g1_slm_input_mapping_training_history.v1",
        "task_id": TASK_ID,
        "updates": list(history),
    })
    _atomic_json(attempt_dir / "validation_trace.json", {
        "schema": "lightgen_g1_slm_input_mapping_validation_trace.v1",
        "task_id": TASK_ID,
        "measurements": list(validations),
        "model_selection": "terminal step 5000 only; validation observations did not select checkpoints",
    })


def _validate_coefficients(
    mapper: AmplitudeMapper, optical_model: base.SmallMatchedAdapter,
    population: Mapping[str, Any], scale: torch.Tensor, device: torch.device, step: int,
) -> dict[str, Any]:
    mapper.eval()
    losses_by_class: dict[int, list[float]] = {label: [] for label in range(CLASS_COUNT)}
    with torch.no_grad():
        coefficients = population["validation"]["coefficients"]
        labels = population["validation"]["labels"]
        for start in range(0, len(coefficients), 10):
            target_cpu = coefficients[start:start + 10]
            target = target_cpu.to(device=device, dtype=torch.complex128)
            feature = _feature_batch(target_cpu, scale, device)
            amplitude = mapper(feature)
            actual = _forward_actual(optical_model, amplitude.square().double())
            errors = _phase_invariant_loss(actual["coefficients"], target).sqrt().detach().cpu()
            for index, error in enumerate(errors.tolist()):
                label = int(labels[start + index])
                losses_by_class[label].append(float(error))
    _require(all(len(values) == 200 for values in losses_by_class.values()),
             "fixed validation cohort changed class denominator")
    all_errors = [value for values in losses_by_class.values() for value in values]
    return {
        "step": step,
        "population": "fixed G1 validation only; observation, no selection or retuning",
        "count": len(all_errors),
        "mean_relative_coefficient_error": float(np.mean(all_errors)),
        "median_relative_coefficient_error": float(np.median(all_errors)),
        "maximum_relative_coefficient_error": float(np.max(all_errors)),
        "per_class": {
            str(label): {
                "count": len(losses_by_class[label]),
                "mean_relative_coefficient_error": float(np.mean(losses_by_class[label])),
                "maximum_relative_coefficient_error": float(np.max(losses_by_class[label])),
            }
            for label in range(CLASS_COUNT)
        },
    }


def _update_progress(attempt_dir: Path, run_id: str, completed: int,
                     started: float, history: Sequence[Mapping[str, Any]], status: str) -> None:
    recent = [float(item["loss"]) for item in history[-100:]]
    _atomic_json(attempt_dir / "progress.json", {
        "schema": "lightgen_g1_slm_input_mapping_progress.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "status": status,
        "phase": "TRAINING" if completed < 5000 else "POST_TRAIN_COLLECTION",
        "completed_updates": completed,
        "registered_updates": 5000,
        "progress_interval_steps": 100,
        "recent_mean_loss": None if not recent else float(np.mean(recent)),
        "elapsed_seconds_observation": time.time() - started,
        "frozen_optical_updates": 0,
        "evaluator_updates": 0,
    })


def _train_to_terminal(
    config_path: Path, config: Mapping[str, Any], run_id: str,
    population: dict[str, Any], schedule: torch.Tensor,
    inputs: Mapping[str, Any], preflight_dir: Path, attempt_dir: Path,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "formal mapper training requires AutoDL CUDA")
    device = torch.device("cuda:0")
    checkpoint = inputs["checkpoint"]
    optical_model = _new_optical_model(checkpoint, 2048, device)
    optical_before = _snapshot_state(optical_model)
    mapper = _new_mapper(device, int(config["population"]["initialization_seed"]))
    init_payload = torch.load(preflight_dir / "mapper_initialization.pt", map_location="cpu", weights_only=False)
    _require(init_payload.get("run_id") == run_id and init_payload.get("seed") == 2026102670
             and init_payload.get("mapper_id") == MAPPER_ID,
             "fixed mapper initialization belongs to another preflight")
    mapper.load_state_dict(init_payload["state_dict"], strict=True)
    initial_state = {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()}
    _atomic_torch(attempt_dir / "mapper_initialization.pt", init_payload)
    optimizer = _make_optimizer(mapper)
    checkpoint_dir = attempt_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_path = attempt_dir / "validation_trace.json"
    if resume_checkpoint is None:
        completed = 0
        history: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        _save_training_trace(attempt_dir, history, validations)
    else:
        completed, history = _restore_training_state(
            resume_checkpoint, mapper, optimizer, run_id=run_id, schedule=schedule,
        )
        saved_validation = _json(validation_path).get("measurements", []) if validation_path.is_file() else []
        validations = [dict(item) for item in saved_validation]
    if completed == 0:
        validations.append(_validate_coefficients(mapper, optical_model, population, inputs["scale"], device, 0))
        _save_training_trace(attempt_dir, history, validations)
        _save_and_reopen_checkpoint(
            checkpoint_dir / "checkpoint_step_000000.pt", mapper, optimizer, run_id=run_id,
            completed_updates=0, loss_history=history, schedule=schedule,
        )
    _atomic_json(attempt_dir / "task_state.json", {
        "schema": "lightgen_g1_slm_input_mapping_task_state.v1",
        "task_id": TASK_ID, "run_id": run_id, "status": "SCIENTIFIC_ACTIVE",
        "phase": "TRAINING", "completed_updates": completed, "registered_updates": 5000,
        "evidence_tier": "diagnostic_only", "scientific_attempt_opened": True,
    })
    started = time.time()
    validation_steps = set(config["training"]["validation_steps"])
    requested_signal: dict[str, int | None] = {"number": None}
    previous_signal_handlers: dict[int, Any] = {}

    def mark_interruption(signum: int, frame: Any) -> None:
        del frame
        requested_signal["number"] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signum] = signal.signal(signum, mark_interruption)

    def save_interruption() -> None:
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{completed:06d}.pt"
        checkpoint_is_current = False
        if checkpoint_path.is_file():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_is_current = saved.get("completed_updates") == completed
        if not checkpoint_is_current:
            _save_and_reopen_checkpoint(
                checkpoint_path, mapper, optimizer, run_id=run_id,
                completed_updates=completed, loss_history=history, schedule=schedule,
            )
        _save_training_trace(attempt_dir, history, validations)
        _update_progress(attempt_dir, run_id, completed, started, history, "INTERRUPTED")
        recent = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda path: int(path.stem.removeprefix("checkpoint_step_")),
        )
        keep = {checkpoint_dir / "checkpoint_step_000000.pt", *recent[-2:]}
        for obsolete in recent:
            if obsolete not in keep:
                obsolete.unlink()
        _atomic_json(attempt_dir / "task_state.json", {
            "schema": "lightgen_g1_slm_input_mapping_task_state.v1",
            "task_id": TASK_ID, "run_id": run_id, "status": "INTERRUPTED",
            "phase": "TRAINING", "completed_updates": completed,
            "registered_updates": 5000, "explicit_recovery_required": True,
            "recovery_checkpoint": str(checkpoint_path.relative_to(attempt_dir)),
            "received_signal": requested_signal["number"],
        })
        run_identity = _json(attempt_dir / "run_identity.json")
        run_identity.update({"status": "INTERRUPTED", "lifecycle_state": "INTERRUPTED",
                             "completed_updates": completed})
        _atomic_json(attempt_dir / "run_identity.json", run_identity)

    try:
        while completed < 5000:
            if requested_signal["number"] is not None:
                save_interruption()
                break
            record = _training_update_with_index(
                mapper, optimizer, optical_model, population, schedule,
                inputs["scale"], device, completed,
            )
            completed += 1
            record["completed_update"] = completed
            record["elapsed_seconds_observation"] = time.time() - started
            history.append(record)
            if completed in validation_steps:
                validations.append(_validate_coefficients(
                    mapper, optical_model, population, inputs["scale"], device, completed,
                ))
            if completed % 100 == 0:
                _save_training_trace(attempt_dir, history, validations)
                _update_progress(attempt_dir, run_id, completed, started, history, "RUNNING")
            if completed % 500 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{completed:06d}.pt"
                _save_and_reopen_checkpoint(
                    checkpoint_path, mapper, optimizer, run_id=run_id,
                    completed_updates=completed, loss_history=history, schedule=schedule,
                )
                recent = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
                for obsolete in recent[:-2]:
                    if obsolete.name != "checkpoint_step_000000.pt":
                        obsolete.unlink()
            if requested_signal["number"] is not None:
                save_interruption()
                break
        if completed < 5000:
            _assert_state_unchanged(optical_model, optical_before,
                                    "frozen C optics after requested training interruption")
        else:
            _require(completed == 5000 and len(history) == 5000,
                     "formal mapper update count differs from the fixed 5,000-step budget")
            _require([item["step"] for item in validations] == [0, 1000, 2500, 5000],
                     "validation observations do not match the fixed four endpoints")
            _assert_state_unchanged(optical_model, optical_before, "frozen C optics after 5,000 mapper updates")
        if completed < 5000:
            return {
                "completed_updates": completed, "history": history, "validations": validations,
                "mapper": mapper, "optical_model": optical_model, "device": device,
                "initial_state": initial_state, "parameter_audit": {},
            }
        parameter_audit: dict[str, dict[str, float]] = {}
        terminal_state = mapper.state_dict()
        for name in initial_state:
            delta = (terminal_state[name].detach().cpu().double() - initial_state[name].double()).abs()
            maximum = float(delta.max())
            norm = float(torch.linalg.vector_norm(delta))
            _require(maximum > 0.0 and norm > 0.0,
                     f"terminal mapper parameter tensor did not change: {name}")
            parameter_audit[name] = {"max_absolute_change": maximum, "l2_change": norm}
        _atomic_json(attempt_dir / "mapper_parameter_update_audit.json", {
            "task_id": TASK_ID, "run_id": run_id,
            "all_trainable_parameter_tensors_changed": True,
            "parameter_changes": parameter_audit,
            "frozen_optics_updates": 0,
        })
        terminal_path = checkpoint_dir / "terminal_checkpoint.pt"
        _save_and_reopen_checkpoint(
            terminal_path, mapper, optimizer, run_id=run_id,
            completed_updates=completed, loss_history=history, schedule=schedule,
        )
        shutil.copy2(terminal_path, attempt_dir / "terminal_checkpoint.pt")
        _save_training_trace(attempt_dir, history, validations)
        _update_progress(attempt_dir, run_id, completed, started, history, "TRAINING_COMPLETE")
        _atomic_json(attempt_dir / "task_state.json", {
            "schema": "lightgen_g1_slm_input_mapping_task_state.v1",
            "task_id": TASK_ID, "run_id": run_id, "status": "POST_TRAIN_COLLECTION",
            "phase": "POST_TRAIN_COLLECTION", "completed_updates": completed,
            "registered_updates": 5000, "evidence_tier": "diagnostic_only",
            "scientific_attempt_opened": True,
        })
    except KeyboardInterrupt:
        save_interruption()
        raise
    finally:
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
    return {
        "completed_updates": completed,
        "history": history,
        "validations": validations,
        "mapper": mapper,
        "optical_model": optical_model,
        "device": device,
        "initial_state": initial_state,
        "parameter_audit": parameter_audit,
    }


def _open_attempt(config: Mapping[str, Any], receipt: PreflightReceipt, run_id: str,
                  *, resume: bool) -> dict[str, Any]:
    attempt_id = f"{run_id}:attempt_01"
    identity = _research_identity(config, run_id, attempt_id)
    control = ResearchRunControl(identity=identity,
                                 authorization_id=str(config["semantic_authority_id"]))
    control.mark_preflight_ready(receipt)
    opened = control.open_scientific_attempt(
        _attempt_dir(config), attempt_id, receipt, resume=resume,
    )
    result = opened.to_dict()
    result.update({"status": RUN_SCIENTIFIC_ACTIVE, "lifecycle_state": RUN_SCIENTIFIC_ACTIVE,
                   "evidence_tier": "diagnostic_only"})
    _atomic_json(_attempt_dir(config) / "run_identity.json", result)
    return result


def _execute_or_resume(config_path: Path, config: Mapping[str, Any],
                       resume_checkpoint: Path | None) -> dict[str, Any]:
    preflight_dir = _preflight_dir(config)
    attempt_dir = _attempt_dir(config)
    receipt = _load_ready_receipt(config, preflight_dir)
    run_id = _run_id(preflight_dir)
    inputs = _validate_source_inputs(config)
    population, schedule = _load_saved_population(preflight_dir)
    if resume_checkpoint is None:
        _open_attempt(config, receipt, run_id, resume=False)
        for name in (
            "implementation_review.json", "preflight_identity.json", "preflight_receipt.json",
            "preflight_summary.json", "input_lineage.json", "zero_update_baseline.json",
            "engineering_recovery.json", "fixed_population_and_schedule.pt", "mapper_initialization.pt",
            "run_config.yaml", "runner_source.py",
        ):
            source_path = preflight_dir / name
            _require(source_path.is_file(), f"required preflight evidence is missing: {source_path}",
                     FileNotFoundError)
            shutil.copy2(source_path, attempt_dir / ("preflight_" + name if name in {
                "fixed_population_and_schedule.pt", "mapper_initialization.pt", "run_config.yaml",
                "runner_source.py",
            } else name))
        shutil.copy2(preflight_dir / "fixed_population_and_schedule.pt",
                     attempt_dir / "fixed_population_and_schedule.pt")
        _atomic_torch(attempt_dir / "training_schedule.pt", schedule)
        shutil.copy2(preflight_dir / "mapper_initialization.pt", attempt_dir / "preflight_mapper_initialization.pt")
        shutil.copy2(config_path, attempt_dir / "run_config.yaml")
        shutil.copy2(Path(__file__).resolve(), attempt_dir / "runner_source.py")
    else:
        _require(attempt_dir.is_dir(), "explicit resume requires the original attempt directory")
        checkpoint_path = resume_checkpoint.resolve()
        _require(checkpoint_path.is_file()
                 and checkpoint_path.is_relative_to((attempt_dir / "checkpoints").resolve()),
                 "resume must name an explicit checkpoint inside this attempt")
        saved_identity = _json(attempt_dir / "run_identity.json")
        state_path = attempt_dir / "task_state.json"
        saved_state = _json(state_path) if state_path.is_file() else {}
        _require(saved_identity.get("status") != "CLOSED"
                 and saved_state.get("phase") != "CLOSED",
                 "a closed scientific attempt cannot be resumed; use evidence verification")
        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(0 <= int(saved_checkpoint.get("completed_updates", -1)) < 5000,
                 "resume checkpoint must precede the terminal step; use measure for terminal outputs")
        expected_identity = _research_identity(config, run_id, f"{run_id}:attempt_01")
        check_resume_compatibility(expected_identity, saved_identity)
        _open_attempt(config, receipt, run_id, resume=True)
        population, schedule = _load_attempt_population(attempt_dir)
    result = _train_to_terminal(
        config_path, config, run_id, population, schedule, inputs, preflight_dir, attempt_dir,
        resume_checkpoint,
    )
    return {
        "run_id": run_id,
        "attempt_dir": str(attempt_dir),
        "completed_updates": result["completed_updates"],
        "training_complete": result["completed_updates"] == 5000,
    }


def _quantize_amplitude(amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    value = amplitude.detach().to(dtype=torch.float64)
    _require(torch.isfinite(value).all().item() and float(value.min()) >= 0.0 and float(value.max()) <= 1.0,
             "continuous mapper amplitude is outside [0,1]")
    a8_codes = torch.floor(255.0 * value + 0.5).clamp(0, 255).to(torch.uint8)
    a8_amplitude = a8_codes.to(torch.float64) / 255.0
    a8_intensity = a8_amplitude.square()
    i8_codes = torch.floor(255.0 * value.square() + 0.5).clamp(0, 255).to(torch.uint8)
    i8_intensity = i8_codes.to(torch.float64) / 255.0
    return a8_codes, a8_intensity, i8_codes, i8_intensity


def _readout_for_evaluator(
    raw_cmos_w_per_m2: torch.Tensor, camera_pixel_area_m2: float, camera_gain_w_inverse: float,
) -> torch.Tensor:
    w_per_pixel = _camera_w_per_pixel(raw_cmos_w_per_m2, camera_pixel_area_m2)
    fixed = (w_per_pixel * camera_gain_w_inverse).clamp(0.0, 1.0)
    return torch.nn.functional.adaptive_avg_pool2d(fixed, (28, 28)).float()


def _display_arrays_from_records(
    records: Sequence[Mapping[str, Any]], camera_pixel_area_m2: float, camera_gain_w_inverse: float,
) -> dict[str, np.ndarray]:
    ordered = [item for label in range(CLASS_COUNT)
               for item in [record for record in records if int(record["label"]) == label][:2]]
    _require(len(ordered) == CLASS_COUNT * 2,
             "fixed display population must have the first two rows of every class")
    mapped_raw = np.stack([np.asarray(item["raw_cmos"], dtype=np.float64) for item in ordered])
    direct_raw = np.stack([np.asarray(item["direct_raw_cmos"], dtype=np.float64) for item in ordered])
    mapped_w_per_pixel = _camera_w_per_pixel(mapped_raw, camera_pixel_area_m2)
    direct_w_per_pixel = _camera_w_per_pixel(direct_raw, camera_pixel_area_m2)
    mapped_gain_scaled = mapped_w_per_pixel * camera_gain_w_inverse
    direct_gain_scaled = direct_w_per_pixel * camera_gain_w_inverse
    mapped_u8 = np.rint(np.clip(mapped_gain_scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    direct_u8 = np.rint(np.clip(direct_gain_scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    return {
        "labels": np.asarray([int(item["label"]) for item in ordered], dtype=np.int64),
        "sample_indices_within_class": np.asarray(
            [int(item["sample_index_within_class"]) for item in ordered], dtype=np.int64,
        ),
        "proposal_ids": np.asarray([int(item["proposal_id"]) for item in ordered], dtype=np.int64),
        "mapped_raw_cmos_w_per_m2": mapped_raw,
        "direct_raw_cmos_w_per_m2": direct_raw,
        "mapped_w_per_pixel": mapped_w_per_pixel,
        "direct_w_per_pixel": direct_w_per_pixel,
        "mapped_gain_scaled": mapped_gain_scaled,
        "direct_gain_scaled": direct_gain_scaled,
        "mapped_display_u8": mapped_u8,
        "direct_display_u8": direct_u8,
    }


def _save_first_two_contact_sheet(
    attempt_dir: Path, records: Sequence[Mapping[str, Any]],
    camera_pixel_area_m2: float, camera_gain_w_inverse: float,
) -> None:
    width, height = 200, 200
    margin, label_height, columns = 8, 18, 5
    rows = CLASS_COUNT
    canvas = Image.new("RGB", (margin + columns * (width + margin),
                                margin + rows * (height + label_height + margin)), "white")
    draw = ImageDraw.Draw(canvas)
    manifest: list[dict[str, Any]] = []
    display = _display_arrays_from_records(records, camera_pixel_area_m2, camera_gain_w_inverse)
    display_index = 0
    for label in range(CLASS_COUNT):
        class_records = [item for item in records if int(item["label"]) == label][:2]
        _require(len(class_records) == 2, f"first-two output examples are incomplete for class {label}")
        row_y = margin + label * (height + label_height + margin)
        draw.text((margin, row_y), f"class {label}", fill="black")
        for sample_slot, item in enumerate(class_records):
            image = display["mapped_display_u8"][display_index]
            preview = Image.fromarray(image, mode="L").convert("RGB")
            col = sample_slot * 2 + 1
            x = margin + col * (width + margin)
            canvas.paste(preview, (x, row_y + label_height))
            draw.text((x, row_y), f"mapped {sample_slot + 1}", fill="black")
            reference_image = Image.fromarray(
                display["direct_display_u8"][display_index], mode="L"
            ).convert("RGB")
            ref_col = col + 1
            ref_x = margin + ref_col * (width + margin)
            canvas.paste(reference_image, (ref_x, row_y + label_height))
            draw.text((ref_x, row_y), f"direct {sample_slot + 1}", fill="black")
            manifest.append({"label": label, "sample_index_within_class": int(item["sample_index_within_class"]),
                             "proposal_id": int(item["proposal_id"]),
                             "mapped_condition": "continuous", "grid": 2048})
            display_index += 1
    figure_dir = attempt_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(figure_dir / "continuous_2048_first_two_per_class.png")
    _atomic_npz(figure_dir / "continuous_2048_first_two_per_class_display_reference.npz", **display)
    _atomic_json(figure_dir / "continuous_2048_first_two_per_class_manifest.json", {
        "task_id": TASK_ID,
        "source": "fixed G1 holdout rows in saved population order",
        "display": "W/m^2 times frozen camera pixel area, frozen global gain, clamp, round to uint8; no per-image normalization",
        "camera_pixel_area_m2": camera_pixel_area_m2,
        "camera_gain_w_inverse": camera_gain_w_inverse,
        "items": manifest,
    })


def _run_final_measurement(
    config: Mapping[str, Any], attempt_dir: Path, inputs: Mapping[str, Any],
) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "terminal full-optics measurement requires AutoDL CUDA")
    device = torch.device("cuda:0")
    population, _ = _load_attempt_population(attempt_dir)
    terminal_path = attempt_dir / "terminal_checkpoint.pt"
    _require(terminal_path.is_file(), "terminal mapper checkpoint is missing", FileNotFoundError)
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    _require(terminal.get("task_id") == TASK_ID
             and terminal.get("run_id") == _json(attempt_dir / "run_identity.json")["run_id"]
             and terminal.get("completed_updates") == 5000,
             "terminal measurement must load the explicit 5,000-step mapper checkpoint")
    mapper = _new_mapper(device, int(config["population"]["initialization_seed"]))
    mapper.load_state_dict(terminal["mapper_state"], strict=True)
    mapper.eval()
    models = {grid: _new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
    model_states = {grid: _snapshot_state(model) for grid, model in models.items()}
    evaluator_ed = inputs["evaluator_ed"].to(device).eval()
    evaluator_ec = inputs["evaluator_ec"].to(device).eval()
    camera_pixel_area_m2 = _camera_pixel_area_m2(models[2048])
    for evaluator in (evaluator_ed, evaluator_ec):
        for parameter in evaluator.parameters():
            parameter.requires_grad_(False)
    evaluator_states = {
        "ED": _snapshot_state(evaluator_ed),
        "EC": _snapshot_state(evaluator_ec),
    }
    holdout = population["holdout"]
    shard_dir = attempt_dir / "measurements"
    shard_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 10
    for batch_index, start in enumerate(range(0, len(holdout["coefficients"]), batch_size)):
        shard_path = shard_dir / f"holdout_batch_{batch_index:04d}.pt"
        if shard_path.is_file():
            previous = torch.load(shard_path, map_location="cpu", weights_only=False)
            expected_indices = torch.arange(start, min(start + batch_size, len(holdout["coefficients"])))
            _require(previous.get("task_id") == TASK_ID
                     and previous.get("run_id") == terminal["run_id"]
                     and torch.equal(previous.get("population_indices"), expected_indices),
                     f"existing terminal measurement shard has another identity: {shard_path}")
            continue
        target_cpu = holdout["coefficients"][start:start + batch_size]
        target = target_cpu.to(device=device, dtype=torch.complex128)
        labels = holdout["labels"][start:start + batch_size]
        features = _feature_batch(target_cpu, inputs["scale"], device)
        with torch.no_grad():
            amplitude = mapper(features)
            a8_codes, a8_intensity, i8_codes, i8_intensity = _quantize_amplitude(amplitude)
            commands = {
                "continuous": amplitude.square().double(),
                "A8_amplitude_linear": a8_intensity,
                "I8_intensity_linear": i8_intensity,
            }
            grid_rows: dict[str, Any] = {}
            for grid in GRIDS:
                optical_chunk_size = 1 if grid == 4096 else batch_size
                direct_chunks: list[torch.Tensor] = []
                condition_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
                    condition: {"actual_coefficients": [], "actual_raw_cmos": []}
                    for condition in commands
                }
                for chunk_start in range(0, len(target), optical_chunk_size):
                    chunk_stop = min(chunk_start + optical_chunk_size, len(target))
                    direct = _forward_direct(models[grid], target[chunk_start:chunk_stop])
                    direct_chunks.append(direct["raw_cmos"].detach().cpu().clone())
                    for condition, intensity in commands.items():
                        actual = _forward_actual(models[grid], intensity[chunk_start:chunk_stop])
                        condition_chunks[condition]["actual_coefficients"].append(
                            actual["coefficients"].detach().cpu().clone()
                        )
                        condition_chunks[condition]["actual_raw_cmos"].append(
                            actual["raw_cmos"].detach().cpu().clone()
                        )
                condition_rows = {
                    condition: {
                        "actual_coefficients": torch.cat(values["actual_coefficients"]),
                        "actual_raw_cmos": torch.cat(values["actual_raw_cmos"]),
                        "mean_slm_output_power_w": (
                            inputs["checkpoint"]["config_identity"]["optics"]["input_power_w"]
                            * commands[condition].detach().mean(dim=(1, 2, 3)).cpu().double()
                        ),
                    }
                    for condition, values in condition_chunks.items()
                }
                grid_rows[str(grid)] = {
                    "conditions": condition_rows,
                    "direct_raw_cmos": torch.cat(direct_chunks),
                }
            payload = {
                "schema": "lightgen_g1_slm_input_mapping_measurement_shard.v1",
                "task_id": TASK_ID,
                "run_id": terminal["run_id"],
                "population_indices": torch.arange(start, start + len(target_cpu)),
                "labels": labels.clone(),
                "proposal_ids": holdout["proposal_ids"][start:start + len(target_cpu)].clone(),
                "target_coefficients": target_cpu.clone(),
                "continuous_amplitude": amplitude.detach().cpu().clone(),
                "A8_amplitude_gray_u8": a8_codes.detach().cpu().clone(),
                "I8_intensity_gray_u8": i8_codes.detach().cpu().clone(),
                "conditions": grid_rows,
            }
            _atomic_torch(shard_path, payload)
    consumer = _consume_saved_measurements(
        config, attempt_dir, inputs, camera_pixel_area_m2=camera_pixel_area_m2, write_outputs=True,
        output_dir=attempt_dir, close_attempt=True,
        auxiliary_evaluators=(evaluator_ed, evaluator_ec), auxiliary_device=device,
    )
    _assert_state_unchanged(models[2048], model_states[2048], "2048 frozen C optics after terminal measurement")
    _assert_state_unchanged(models[4096], model_states[4096], "4096 frozen C optics after terminal measurement")
    _assert_state_unchanged(evaluator_ed, evaluator_states["ED"], "frozen E_D evaluator")
    _assert_state_unchanged(evaluator_ec, evaluator_states["EC"], "frozen E_C evaluator")
    del models, mapper, evaluator_ed, evaluator_ec
    gc.collect()
    torch.cuda.empty_cache()
    return consumer


def _auxiliary_evaluator_summary(
    ed_predictions: Sequence[int], ec_predictions: Sequence[int], labels: Sequence[int],
) -> dict[str, Any]:
    _require(len(ed_predictions) == len(ec_predictions) == len(labels) == 2000,
             "auxiliary evaluator input count changed")
    result: dict[str, Any] = {}
    for name, predictions in (("E_D", ed_predictions), ("E_C", ec_predictions)):
        recalls = {}
        for label in range(CLASS_COUNT):
            indices = [index for index, target in enumerate(labels) if target == label]
            _require(len(indices) == 200, f"auxiliary holdout class-{label} denominator changed")
            recalls[str(label)] = sum(int(predictions[index] == label) for index in indices) / len(indices)
        result[name] = {
            "count": 2000,
            "correct": sum(int(prediction == target) for prediction, target in zip(predictions, labels, strict=True)),
            "macro_recall": float(np.mean(list(recalls.values()))),
            "recall_by_class": recalls,
            "gate_use": False,
        }
    return result


def _consume_saved_measurements(
    config: Mapping[str, Any], attempt_dir: Path, inputs: Mapping[str, Any], *,
    camera_pixel_area_m2: float, write_outputs: bool,
    output_dir: Path | None = None, close_attempt: bool = False, return_audit: bool = False,
    auxiliary_evaluators: tuple[torch.nn.Module, torch.nn.Module] | None = None,
    auxiliary_device: torch.device | None = None,
) -> dict[str, Any]:
    """Recompute all denominators, errors, quantization and the one outer gate from saved tensors."""
    population, _ = _load_attempt_population(attempt_dir)
    holdout = population["holdout"]
    run_id = _json(attempt_dir / "run_identity.json")["run_id"]
    shard_dir = attempt_dir / "measurements"
    expected_count = len(holdout["coefficients"])
    expected_batches = math.ceil(expected_count / 10)
    _require(expected_count == 2000 and expected_batches == 200,
             "fixed final holdout or batch denominator changed")
    if auxiliary_evaluators is None:
        evaluator_ed = inputs["evaluator_ed"]
        evaluator_ec = inputs["evaluator_ec"]
        device = torch.device("cpu")
    else:
        evaluator_ed, evaluator_ec = auxiliary_evaluators
        device = auxiliary_device or torch.device("cpu")
    evaluator_ed.eval()
    evaluator_ec.eval()
    evaluator_states = (_snapshot_state(evaluator_ed), _snapshot_state(evaluator_ec))

    records: list[dict[str, Any]] = []
    ed_predictions: list[int] = []
    ec_predictions: list[int] = []
    labels_all: list[int] = []
    display_records: list[dict[str, Any]] = []
    for batch_index in range(expected_batches):
        shard_path = shard_dir / f"holdout_batch_{batch_index:04d}.pt"
        _require(shard_path.is_file(), f"final saved measurement shard is missing: {shard_path}", FileNotFoundError)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        start = batch_index * 10
        stop = min(start + 10, expected_count)
        indices = torch.arange(start, stop)
        target = holdout["coefficients"][start:stop]
        labels = holdout["labels"][start:stop]
        proposal_ids = holdout["proposal_ids"][start:stop]
        _require(payload.get("schema") == "lightgen_g1_slm_input_mapping_measurement_shard.v1"
                 and payload.get("task_id") == TASK_ID and payload.get("run_id") == run_id
                 and torch.equal(payload.get("population_indices"), indices)
                 and torch.equal(payload.get("labels"), labels)
                 and torch.equal(payload.get("proposal_ids"), proposal_ids)
                 and torch.equal(payload.get("target_coefficients"), target),
                 f"saved measurement shard does not match the fixed holdout rows: {shard_path}")
        amplitude = payload.get("continuous_amplitude")
        _require(torch.is_tensor(amplitude) and tuple(amplitude.shape) == (stop - start, 1, *PIXEL_SHAPE)
                 and amplitude.dtype == torch.float32 and torch.isfinite(amplitude).all().item()
                 and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0,
                 f"saved continuous amplitude command is malformed: {shard_path}")
        expected_a8, _, expected_i8, _ = _quantize_amplitude(amplitude)
        _require(torch.equal(payload.get("A8_amplitude_gray_u8"), expected_a8)
                 and torch.equal(payload.get("I8_intensity_gray_u8"), expected_i8),
                 f"saved A8/I8 uint8 instructions do not match the fixed rounding rule: {shard_path}")
        saved_grids = payload.get("conditions")
        _require(isinstance(saved_grids, Mapping) and set(saved_grids) == {str(grid) for grid in GRIDS},
                 f"saved two-grid condition outputs are incomplete: {shard_path}")
        for grid in GRIDS:
            grid_payload = saved_grids[str(grid)]
            direct_raw = grid_payload.get("direct_raw_cmos")
            _require(torch.is_tensor(direct_raw)
                     and tuple(direct_raw.shape) == (stop - start, 1, *PIXEL_SHAPE)
                     and direct_raw.dtype == torch.float64,
                     f"direct target-Decoder reference is missing at grid {grid}")
            for condition in ("continuous", "A8_amplitude_linear", "I8_intensity_linear"):
                condition_payload = grid_payload["conditions"].get(condition)
                _require(isinstance(condition_payload, Mapping), f"missing {condition} output at grid {grid}")
                actual_coefficients = condition_payload.get("actual_coefficients")
                actual_raw = condition_payload.get("actual_raw_cmos")
                _require(torch.is_tensor(actual_coefficients)
                         and tuple(actual_coefficients.shape) == (stop - start, PORT_COUNT)
                         and torch.is_complex(actual_coefficients)
                         and actual_coefficients.dtype == torch.complex128
                         and torch.is_tensor(actual_raw)
                         and tuple(actual_raw.shape) == (stop - start, 1, *PIXEL_SHAPE)
                         and actual_raw.dtype == torch.float64,
                         f"saved actual {condition} production output is malformed at grid {grid}")
        continuous_2048 = saved_grids["2048"]["conditions"]["continuous"]["actual_raw_cmos"].to(device)
        readout = _readout_for_evaluator(
            continuous_2048, camera_pixel_area_m2, inputs["camera_gain"],
        )
        with torch.no_grad():
            ed_logits = evaluator_ed(readout)
            _, ec_logits = evaluator_ec(readout)
        ed_batch = ed_logits.argmax(dim=1).detach().cpu().tolist()
        ec_batch = ec_logits.argmax(dim=1).detach().cpu().tolist()
        ed_predictions.extend(int(value) for value in ed_batch)
        ec_predictions.extend(int(value) for value in ec_batch)
        labels_all.extend(int(value) for value in labels.tolist())
        for local, global_index in enumerate(range(start, stop)):
            label = int(labels[local])
            if sum(1 for item in display_records if item["label"] == label) < 2:
                display_records.append({
                    "label": label,
                    "sample_index_within_class": global_index - label * 200,
                    "proposal_id": int(proposal_ids[local]),
                    "raw_cmos": saved_grids["2048"]["conditions"]["continuous"][
                        "actual_raw_cmos"][local, 0].numpy(),
                    "direct_raw_cmos": saved_grids["2048"]["direct_raw_cmos"][local, 0].numpy(),
                })

            target_coeff = target[local].numpy()
            row = _new_measurement_record(
                global_index, label, int(proposal_ids[local]),
            )
            for grid in GRIDS:
                grid_payload = saved_grids[str(grid)]
                direct_cmos = grid_payload["direct_raw_cmos"][local].numpy()
                condition_errors: dict[str, dict[str, float]] = {}
                for condition in ("continuous", "A8_amplitude_linear", "I8_intensity_linear"):
                    condition_payload = grid_payload["conditions"][condition]
                    actual_coeff = condition_payload["actual_coefficients"][local].numpy()
                    actual_cmos = condition_payload["actual_raw_cmos"][local].numpy()
                    _, errors = _single_record_passes(actual_coeff, target_coeff, actual_cmos, direct_cmos)
                    phase = _numpy_phase_error(actual_coeff, target_coeff)[0]
                    condition_errors[condition] = {
                        "coefficient_phase_rad": phase,
                        **errors,
                    }
                row["continuous"][str(grid)] = condition_errors["continuous"]
                row["quantized"]["A8_amplitude_linear"][str(grid)] = condition_errors[
                    "A8_amplitude_linear"
                ]
                row["quantized"]["I8_intensity_linear"][str(grid)] = condition_errors[
                    "I8_intensity_linear"
                ]
                condition_payload = grid_payload["conditions"]
                for condition in ("continuous", "A8_amplitude_linear", "I8_intensity_linear"):
                    power = condition_payload[condition].get("mean_slm_output_power_w")
                    if torch.is_tensor(power):
                        _require(torch.isfinite(power).all().item()
                                 and tuple(power.shape) == (stop - start,),
                                 f"saved power observation is malformed for {condition}/{grid}")

            direct_cross_cmos = _numpy_relative_error(
                saved_grids["2048"]["direct_raw_cmos"][local].numpy(),
                saved_grids["4096"]["direct_raw_cmos"][local].numpy(),
            )
            for condition in ("continuous", "A8_amplitude_linear", "I8_intensity_linear"):
                cross_coeff = _numpy_phase_error(
                    saved_grids["2048"]["conditions"][condition]["actual_coefficients"][local].numpy(),
                    saved_grids["4096"]["conditions"][condition]["actual_coefficients"][local].numpy(),
                )[1]
                cross_cmos = _numpy_relative_error(
                    saved_grids["2048"]["conditions"][condition]["actual_raw_cmos"][local].numpy(),
                    saved_grids["4096"]["conditions"][condition]["actual_raw_cmos"][local].numpy(),
                )
                cross_values = {
                    "coefficient_relative_error_one_global_phase": cross_coeff,
                    "native_cmos_relative_error": cross_cmos,
                    "direct_reference_native_cmos_relative_error": direct_cross_cmos,
                }
                if condition == "continuous":
                    row["cross_grid"] = cross_values
                else:
                    row["quantized_cross_grid"][condition] = cross_values
            rows_for_gate = [
                row["continuous"][str(grid)]["coefficient_relative_error"] for grid in GRIDS
            ] + [row["continuous"][str(grid)]["native_cmos_relative_error"] for grid in GRIDS]
            rows_for_gate.extend((cross_coeff, cross_cmos))
            row["continuous_pass"] = all(value <= MEASUREMENT_LIMIT for value in rows_for_gate)
            records.append(row)

    _require(len(records) == expected_count and len(ed_predictions) == len(ec_predictions) ==
             len(labels_all) == expected_count,
             "saved evidence consumer did not recompute every fixed holdout row")
    counts = Counter(int(item["label"]) for item in records)
    _require(counts == Counter({label: 200 for label in range(CLASS_COUNT)}),
             "saved evidence consumer found changed per-class denominators")
    continuous_passed = sum(bool(item["continuous_pass"]) for item in records)
    overall_pass = continuous_passed == expected_count
    auxiliary = _auxiliary_evaluator_summary(ed_predictions, ec_predictions, labels_all)
    continuous_by_class = _summarize_condition(records, "continuous")
    quantized = {
        condition: _summarize_condition(records, condition)
        for condition in ("A8_amplitude_linear", "I8_intensity_linear")
    }
    command_power = _recompute_command_power(
        records, population, inputs, attempt_dir, camera_pixel_area_m2,
    )
    decision = {
        "schema": "lightgen_g1_slm_input_mapping_final_decision.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "status": "COMPLETE_DIAGNOSTIC" if overall_pass else "VALID_FAILED",
        "lifecycle_state": RUN_COMPLETED if overall_pass else RUN_VALID_FAILED,
        "decision": ("CONTINUOUS_G1_INPUT_MAPPING_NUMERICALLY_SUPPORTED_ON_FIXED_HOLDOUT"
                     if overall_pass else "MAPPING_NOT_SUPPORTED_WITHIN_FIXED_5000_UPDATE_BUDGET"),
        "evidence_tier": "diagnostic_only",
        "decision_scope": "2,000 fixed G1 holdout coefficients; C45k frozen model; ideal amplitude SLM; grids 2048 and 4096",
        "population": {"total": expected_count, "per_class": 200,
                       "class_count": CLASS_COUNT, "rows_recomputed": len(records)},
        "continuous_final_gate": {
            "threshold": MEASUREMENT_LIMIT,
            "coefficient_alignment": "one global phase only",
            "raw_cmos": "native raw W/m^2; no normalization, power match, or gain adjustment",
            "auxiliary_camera_readout": {
                "raw_cmos_unit": "W/m^2",
                "pixel_area_m2": camera_pixel_area_m2,
                "camera_gain_unit": "W^-1 after pixel integration",
                "evaluator_input": "clip(raw_cmos * pixel_area * fixed_camera_gain, 0, 1), then adaptive average pool to 28x28",
            },
            "cross_grid_metrics": ["mapped coefficient one-global-phase relative L2",
                                   "mapped native CMOS relative L2"],
            "passed_rows": continuous_passed,
            "denominator": expected_count,
            "pass_fraction": continuous_passed / expected_count,
            "all_rows_passed": overall_pass,
            "maxima": _maximum_continuous_errors(records),
        },
        "per_class_continuous": continuous_by_class,
        "quantized_terminal_measurements_not_gates": quantized,
        "slm_power_and_clipping": command_power,
        "auxiliary_ED_EC_2048_continuous_only": auxiliary,
        "updates": {"mapper": 5000, "frozen_optics": 0, "evaluators": 0},
        "new_MNIST_raw_reads": 0,
        "human_review": 0,
        "hardware_acceptance": False,
        "historical_C_status_preserved": "VALID_FAILED",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }
    if write_outputs:
        _require(output_dir is not None,
                 "consumer output requires an explicit derived-output directory", ValueError)
        _atomic_json(output_dir / "final_decision.json", decision)
        _write_sample_csv(output_dir / "per_sample_measurements.csv", records)
        _save_first_two_contact_sheet(
            output_dir, display_records, camera_pixel_area_m2, inputs["camera_gain"],
        )
        _atomic_json(output_dir / "auxiliary_ED_EC_2048.json", {
            "schema": "lightgen_g1_slm_mapping_auxiliary_evaluators.v1",
            "task_id": TASK_ID,
            "scope": "frozen E_D/E_C on 2048 continuous mapped holdout only",
            "not_in_loss_or_final_gate": True,
            "metrics": auxiliary,
            "recomputed_from_saved_raw_cmos": True,
        })
        if close_attempt:
            _require(output_dir.resolve() == attempt_dir.resolve(),
                     "only the original attempt output may close its lifecycle", ValueError)
            _write_closeout_report(output_dir, decision)
            state = _json(attempt_dir / "task_state.json")
            state.update({"status": decision["status"], "phase": "CLOSED",
                          "completed_updates": 5000, "final_decision": decision["decision"],
                          "stop_point": "final evidence consumer completed; no automatic follow-up"})
            _atomic_json(attempt_dir / "task_state.json", state)
            run_id = decision["run_id"]
            receipt = PreflightReceipt.from_mapping(_json(attempt_dir / "preflight_receipt.json"))
            lifecycle = ResearchRunControl(
                identity=_research_identity(config, run_id, f"{run_id}:attempt_01"),
                authorization_id=str(config["semantic_authority_id"]),
            )
            lifecycle.mark_preflight_ready(receipt)
            lifecycle.open_scientific_attempt(
                attempt_dir, f"{run_id}:attempt_01", receipt, resume=True,
            )
            lifecycle.complete() if overall_pass else lifecycle.valid_failed()
            run_identity = _json(attempt_dir / "run_identity.json")
            run_identity.update({"status": "CLOSED", "lifecycle_state": lifecycle.state,
                                 "scientific_status": decision["status"]})
            _atomic_json(attempt_dir / "run_identity.json", run_identity)
            state["lifecycle_state"] = lifecycle.state
            _atomic_json(attempt_dir / "task_state.json", state)
    _assert_state_unchanged(evaluator_ed, evaluator_states[0], "frozen E_D during evidence consumption")
    _assert_state_unchanged(evaluator_ec, evaluator_states[1], "frozen E_C during evidence consumption")
    if return_audit:
        return {"decision": decision, "records": records, "display_records": display_records}
    return decision


def _new_measurement_record(population_index: int, label: int, proposal_id: int) -> dict[str, Any]:
    return {
        "population_index": population_index,
        "label": label,
        "sample_index_within_class": population_index - label * 200,
        "proposal_id": proposal_id,
        "continuous": {},
        "quantized": {condition: {} for condition in QUANTIZED_CONDITION_NAMES},
        "quantized_cross_grid": {condition: {} for condition in QUANTIZED_CONDITION_NAMES},
    }


def _summarize_condition(records: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for grid in GRIDS:
        field = "continuous" if condition == "continuous" else "quantized"
        if condition == "continuous":
            values = [item[field][str(grid)] for item in records]
        else:
            values = [item[field][condition][str(grid)] for item in records]
        passes = [entry["coefficient_relative_error"] <= MEASUREMENT_LIMIT
                  and entry["native_cmos_relative_error"] <= MEASUREMENT_LIMIT for entry in values]
        result[str(grid)] = {
            "denominator": len(values),
            "rows_with_coefficient_and_cmos_error_le_1e-4": sum(passes),
            "pass_fraction": sum(passes) / len(values),
            "coefficient_relative_error_mean": float(np.mean([v["coefficient_relative_error"] for v in values])),
            "coefficient_relative_error_max": float(np.max([v["coefficient_relative_error"] for v in values])),
            "native_cmos_relative_error_mean": float(np.mean([v["native_cmos_relative_error"] for v in values])),
            "native_cmos_relative_error_max": float(np.max([v["native_cmos_relative_error"] for v in values])),
        }
    cross_values_by_row = [
        item["cross_grid"] if condition == "continuous"
        else item["quantized_cross_grid"][condition]
        for item in records
    ]
    result["cross_grid"] = {
        "denominator": len(records),
        "coefficient_relative_error_mean": float(np.mean([
            item["coefficient_relative_error_one_global_phase"] for item in cross_values_by_row
        ])),
        "coefficient_relative_error_max": float(np.max([
            item["coefficient_relative_error_one_global_phase"] for item in cross_values_by_row
        ])),
        "native_cmos_relative_error_mean": float(np.mean([
            item["native_cmos_relative_error"] for item in cross_values_by_row
        ])),
        "native_cmos_relative_error_max": float(np.max([
            item["native_cmos_relative_error"] for item in cross_values_by_row
        ])),
        "direct_reference_native_cmos_relative_error_max": float(np.max([
            item["direct_reference_native_cmos_relative_error"] for item in cross_values_by_row
        ])),
    }
    result["per_class"] = {}
    for label in range(CLASS_COUNT):
        class_rows = [item for item in records if int(item["label"]) == label]
        class_result: dict[str, Any] = {"denominator": len(class_rows)}
        for grid in GRIDS:
            entries = [item["continuous"][str(grid)] if condition == "continuous"
                       else item["quantized"][condition][str(grid)] for item in class_rows]
            class_result[str(grid)] = {
                "coefficient_error_max": float(np.max([entry["coefficient_relative_error"] for entry in entries])),
                "native_cmos_error_max": float(np.max([entry["native_cmos_relative_error"] for entry in entries])),
                "pass_fraction": sum(entry["coefficient_relative_error"] <= MEASUREMENT_LIMIT
                                     and entry["native_cmos_relative_error"] <= MEASUREMENT_LIMIT
                                     for entry in entries) / len(entries),
            }
        class_cross_values = [
            item["cross_grid"] if condition == "continuous"
            else item["quantized_cross_grid"][condition]
            for item in class_rows
        ]
        class_result["cross_grid"] = {
            "coefficient_error_max": float(np.max([
                item["coefficient_relative_error_one_global_phase"] for item in class_cross_values
            ])),
            "native_cmos_error_max": float(np.max([
                item["native_cmos_relative_error"] for item in class_cross_values
            ])),
        }
        if condition == "continuous":
            class_result["all_grid_and_cross_grid_pass_fraction"] = sum(
                bool(item["continuous_pass"]) for item in class_rows
            ) / len(class_rows)
        result["per_class"][str(label)] = class_result
    return result


def _maximum_continuous_errors(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "coefficient_2048": max(item["continuous"]["2048"]["coefficient_relative_error"] for item in records),
        "coefficient_4096": max(item["continuous"]["4096"]["coefficient_relative_error"] for item in records),
        "native_cmos_2048": max(item["continuous"]["2048"]["native_cmos_relative_error"] for item in records),
        "native_cmos_4096": max(item["continuous"]["4096"]["native_cmos_relative_error"] for item in records),
        "cross_grid_coefficient": max(item["cross_grid"]["coefficient_relative_error_one_global_phase"]
                                       for item in records),
        "cross_grid_native_cmos": max(item["cross_grid"]["native_cmos_relative_error"] for item in records),
    }


def _recompute_command_power(records: Sequence[Mapping[str, Any]], population: Mapping[str, Any],
                             inputs: Mapping[str, Any], attempt_dir: Path,
                             camera_pixel_area_m2: float) -> dict[str, Any]:
    source_power = float(inputs["checkpoint"]["config_identity"]["optics"]["input_power_w"])
    camera_gain = float(inputs["camera_gain"])
    condition_names = ("continuous", "A8_amplitude_linear", "I8_intensity_linear")
    per_class: dict[str, dict[int, list[tuple[float, float]]]] = {
        condition: {label: [] for label in range(CLASS_COUNT)} for condition in condition_names
    }
    total_pixels = 0
    clipped_pixels = 0
    for index in range(200):
        payload = torch.load(attempt_dir / "measurements" / f"holdout_batch_{index:04d}.pt",
                             map_location="cpu", weights_only=False)
        amplitude = payload["continuous_amplitude"]
        a8_codes = payload["A8_amplitude_gray_u8"]
        i8_codes = payload["I8_intensity_gray_u8"]
        labels = payload["labels"]
        _require(tuple(amplitude.shape) == tuple(a8_codes.shape) == tuple(i8_codes.shape)
                 and amplitude.shape[1:] == (1, *PIXEL_SHAPE)
                 and len(labels) == len(amplitude),
                 "saved amplitude/intensity commands changed shape")
        cmos = payload["conditions"]["2048"]["conditions"]["continuous"]["actual_raw_cmos"]
        _require(tuple(cmos.shape) == tuple(amplitude.shape) and torch.isfinite(cmos).all().item(),
                 "saved native CMOS readout is missing from command-power/clipping audit")
        fixed_readout = _camera_w_per_pixel(cmos.double(), camera_pixel_area_m2) * camera_gain
        clipped_pixels += int(((fixed_readout < 0.0) | (fixed_readout > 1.0)).sum())
        total_pixels += fixed_readout.numel()
        intensities = {
            "continuous": amplitude.double().square(),
            "A8_amplitude_linear": (a8_codes.double() / 255.0).square(),
            "I8_intensity_linear": i8_codes.double() / 255.0,
        }
        saturation = {
            "continuous": (amplitude == 0.0) | (amplitude == 1.0),
            "A8_amplitude_linear": (a8_codes == 0) | (a8_codes == 255),
            "I8_intensity_linear": (i8_codes == 0) | (i8_codes == 255),
        }
        for row, label_tensor in enumerate(labels):
            label = int(label_tensor)
            _require(label in range(CLASS_COUNT), "saved measurement contains an invalid class label")
            for condition in condition_names:
                per_class[condition][label].append((
                    source_power * float(intensities[condition][row].mean()),
                    float(saturation[condition][row].double().mean()),
                ))
    _require(len(records) == 2000 and len(population["holdout"]["labels"]) == 2000
             and total_pixels == 2000 * PIXEL_COUNT,
             "power/clipping audit denominator differs from the fixed holdout")
    conditions: dict[str, Any] = {}
    for condition in condition_names:
        rows = [item for label in range(CLASS_COUNT) for item in per_class[condition][label]]
        _require(len(rows) == 2000 and all(len(per_class[condition][label]) == 200 for label in range(CLASS_COUNT)),
                 f"power/clipping per-class denominator changed for {condition}")
        conditions[condition] = {
            "mean_slm_output_power_w": float(np.mean([row[0] for row in rows])),
            "saturated_pixel_fraction": float(np.mean([row[1] for row in rows])),
            "mean_slm_output_power_w_by_class": {
                str(label): float(np.mean([row[0] for row in per_class[condition][label]]))
                for label in range(CLASS_COUNT)
            },
            "saturated_pixel_fraction_by_class": {
                str(label): float(np.mean([row[1] for row in per_class[condition][label]]))
                for label in range(CLASS_COUNT)
            },
        }
    return {
        "source_power_w": source_power,
        "definition": "source power multiplied by mean SLM intensity; no gain or power correction",
        "conditions": conditions,
        "fixed_camera_gain_clipping_fraction_2048_continuous": clipped_pixels / total_pixels,
        "camera_readout_conversion": {
            "raw_cmos_unit": "W/m^2",
            "camera_pixel_area_m2": camera_pixel_area_m2,
            "gain_unit": "W^-1 applied after pixel-area integration",
            "fixed_signal": "clip(raw_cmos_w_per_m2 * camera_pixel_area_m2 * camera_gain_w_inverse, 0, 1)",
        },
        "fixed_camera_gain_clipping_definition": "fraction outside [0,1] after W/m^2 to W/pixel integration and fixed gain, before clamp",
        "new_MNIST_reads": 0,
    }


def _write_sample_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in records:
        row: dict[str, Any] = {
            "population_index": item["population_index"],
            "label": item["label"],
            "sample_index_within_class": item["sample_index_within_class"],
            "proposal_id": item["proposal_id"],
            "continuous_pass": item["continuous_pass"],
        }
        for grid in GRIDS:
            for name, values in (("continuous", item["continuous"][str(grid)]),
                                 ("A8", item["quantized"]["A8_amplitude_linear"][str(grid)]),
                                 ("I8", item["quantized"]["I8_intensity_linear"][str(grid)])):
                row[f"{name}_coefficient_error_{grid}"] = values["coefficient_relative_error"]
                row[f"{name}_native_cmos_error_{grid}"] = values["native_cmos_relative_error"]
        row["continuous_cross_grid_coefficient_error"] = item["cross_grid"][
            "coefficient_relative_error_one_global_phase"
        ]
        row["continuous_cross_grid_cmos_error"] = item["cross_grid"]["native_cmos_relative_error"]
        row["direct_cross_grid_cmos_error"] = item["cross_grid"][
            "direct_reference_native_cmos_relative_error"
        ]
        for name, condition in (("A8", "A8_amplitude_linear"), ("I8", "I8_intensity_linear")):
            row[f"{name}_cross_grid_coefficient_error"] = item["quantized_cross_grid"][condition][
                "coefficient_relative_error_one_global_phase"
            ]
            row[f"{name}_cross_grid_cmos_error"] = item["quantized_cross_grid"][condition][
                "native_cmos_relative_error"
            ]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_closeout_report(attempt_dir: Path, decision: Mapping[str, Any]) -> None:
    passed = bool(decision["continuous_final_gate"]["all_rows_passed"])
    maxima = decision["continuous_final_gate"]["maxima"]
    text = f"""# G1 一次前向 SLM 输入映射试训

## 结果与状态

{decision['status']}。连续映射留出分母为 {decision['population']['rows_recomputed']}；通过行数为 {decision['continuous_final_gate']['passed_rows']}。终态判据 {'全部通过' if passed else '未全部通过'}。本轮训练映射器 5,000 步，冻结光学更新 0；A8/I8 仅为终态辅助测量。

## 科学认识

结论只适用于 C45k、每类一个完整协方差 G1 高斯并按原总模态功率拒绝的 2,000 条固定留出系数、理想连续振幅 SLM 与 2048/4096 数值网格。{'该留出人口支持连续输入映射的数值候选。' if passed else '该留出人口未满足预注册连续数值判据；这不证明 G1 在物理上不可制备。'} E_D/E_C 是冻结的 2048 辅助读出，不进损失或最终门槛；本轮没有人工评分或实物验证。

## 目标缺口

系数误差最大值：2048={maxima['coefficient_2048']:.6g}，4096={maxima['coefficient_4096']:.6g}；原生 CMOS 强度误差最大值：2048={maxima['native_cmos_2048']:.6g}，4096={maxima['native_cmos_4096']:.6g}；跨网格系数/CMOS 最大值={maxima['cross_grid_coefficient']:.6g}/{maxima['cross_grid_native_cmos']:.6g}。仍缺真实 SLM 标定、实测光纤/器件复传输、物理链路读出和人眼语义质量证据。

## 工程完成与验证

固定 C 检查点、单高斯 G1 先验、系数人口、映射器终点、验证轨迹、连续/A8/I8 指令、两网格系数和原生 CMOS 输出、逐样本指标及首两样本图均保存于本 attempt。独立消费者从保存人口与实际前向输出重算分母、误差和最终判据；所有 optical/evaluator 参数保持冻结。历史 C `VALID_FAILED` 与 `PROJECT_GEOMETRY_UNMEASURED` 保持。

## 决策影响与下一步

{'可保留该连续映射候选；量化结果只决定是否另行设计量化训练。' if passed else '不续训、不扩网、不改先验、不降低阈值；本次停止。'} 本结果不授权 G4 训练、人工评分、物理实验或模型采用。
"""
    (attempt_dir / "closeout_report.md").write_text(text, encoding="utf-8")


def _verify_transfer_receipt(
    evidence_dir: Path, *, remote_run_suffix: str, required_files: set[str],
) -> dict[str, Any]:
    receipt_path = evidence_dir / "transfer_receipt.json"
    _require(receipt_path.is_file(), "AutoDL pull transfer receipt is missing", FileNotFoundError)
    receipt = _json(receipt_path)
    _require(receipt.get("status") == "COMPLETE"
             and Path(str(receipt.get("destination", ""))).resolve() == evidence_dir.resolve()
             and str(receipt.get("remote_run_dir", "")).endswith(remote_run_suffix),
             "AutoDL transfer receipt does not cover this exact G1 attempt")
    files = receipt.get("files")
    _require(isinstance(files, list) and files,
             "AutoDL transfer receipt has no per-file transfer results")
    copied: set[str] = set()
    for item in files:
        relative = str(item.get("path", ""))
        file_path = (evidence_dir / relative).resolve()
        _require(Path(relative).is_absolute() is False and ".." not in Path(relative).parts
                 and file_path.is_relative_to(evidence_dir.resolve())
                 and item.get("status") == "COPIED" and file_path.is_file(),
                 f"AutoDL transfer receipt contains an incomplete file: {relative}")
        actual_size = file_path.stat().st_size
        _require(item.get("remote_size_bytes") == item.get("local_size_bytes") == actual_size,
                 f"AutoDL transfer receipt size differs from the actual file: {relative}")
        normalized = relative.replace("\\", "/")
        _require(normalized not in copied, f"AutoDL transfer receipt repeats a file: {relative}")
        copied.add(normalized)
    missing = required_files - copied
    _require(not missing,
             f"AutoDL transfer omitted required evidence files: {sorted(missing)[:8]}")
    return {
        **receipt,
        "files_verified_against_actual_size": len(copied),
        "required_files_verified": len(required_files),
    }


def _metric_tree_matches(left: Any, right: Any, *, atol: float = 1.0e-14) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_metric_tree_matches(left[key], right[key], atol=atol)
                                               for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_metric_tree_matches(a, b, atol=atol)
                                               for a, b in zip(left, right, strict=True))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=atol,
        )
    return left == right


def _continuous_primary_gate_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    primary_fields = (
        "threshold", "coefficient_alignment", "cross_grid_metrics",
        "passed_rows", "denominator", "pass_fraction", "all_rows_passed", "maxima",
    )
    if any(key not in left or key not in right for key in primary_fields):
        return False
    camera_descriptions = (left.get("raw_cmos"), right.get("raw_cmos"))
    camera_terms = ("w/m^2", "no normalization", "power match", "gain adjustment")
    if any(not isinstance(description, str)
           or any(term not in description.lower() for term in camera_terms)
           for description in camera_descriptions):
        return False
    left_primary = {key: left[key] for key in primary_fields}
    right_primary = {key: right[key] for key in primary_fields}
    return _metric_tree_matches(left_primary, right_primary)


def _verify_saved_evidence(config: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    resolved = evidence_dir.resolve()
    _require(resolved == WINDOWS_BACKUP_DIR.resolve(),
             "Windows evidence consumption is restricted to the fixed backup path")
    attempt_required = {f"measurements/holdout_batch_{index:04d}.pt" for index in range(200)} | {
        "run_identity.json", "task_state.json", "fixed_population_and_schedule.pt", "terminal_checkpoint.pt",
        "training_history.json", "validation_trace.json", "final_decision.json",
        "per_sample_measurements.csv", "closeout_report.md", "auxiliary_ED_EC_2048.json",
        "preflight_receipt.json", "preflight_summary.json", "input_lineage.json",
        "zero_update_baseline.json", "engineering_recovery.json", "implementation_review.json",
        "mapper_initialization.pt", "mapper_parameter_update_audit.json", "progress.json",
        "preflight_identity.json", "run_config.yaml", "runner_source.py",
        "preflight_run_config.yaml", "preflight_runner_source.py",
        "checkpoints/checkpoint_step_000000.pt", "checkpoints/checkpoint_step_004500.pt",
        "checkpoints/checkpoint_step_005000.pt", "checkpoints/terminal_checkpoint.pt",
        "figures/continuous_2048_first_two_per_class.png",
        "figures/continuous_2048_first_two_per_class_manifest.json",
    }
    transfer = _verify_transfer_receipt(
        resolved,
        remote_run_suffix="/outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/attempt_01",
        required_files=attempt_required,
    )
    run_identity = _json(resolved / "run_identity.json")
    task_state = _json(resolved / "task_state.json")
    terminal = torch.load(resolved / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    history = _json(resolved / "training_history.json")
    validation_trace = _json(resolved / "validation_trace.json")
    parameter_audit = _json(resolved / "mapper_parameter_update_audit.json")
    receipt = PreflightReceipt.from_mapping(_json(resolved / "preflight_receipt.json"))
    review = _json(resolved / "implementation_review.json")
    _require(run_identity.get("task_id") == TASK_ID
             and run_identity.get("status") == "CLOSED"
             and task_state.get("task_id") == TASK_ID
             and task_state.get("phase") == "CLOSED"
             and task_state.get("completed_updates") == 5000
             and terminal.get("schema") == "lightgen_g1_slm_input_mapping_checkpoint.v1"
             and terminal.get("task_id") == TASK_ID
             and terminal.get("run_id") == run_identity.get("run_id")
             and terminal.get("completed_updates") == terminal.get("checkpoint_step") == 5000
             and len(history.get("updates", [])) == 5000
             and all(item.get("completed_update") == index + 1
                     for index, item in enumerate(history["updates"]))
             and [item.get("step") for item in validation_trace.get("measurements", [])]
             == [0, 1000, 2500, 5000]
             and parameter_audit.get("all_trainable_parameter_tensors_changed") is True
             and receipt.status == PREFLIGHT_READY
             and review.get("status") == "REVIEWED",
             "Windows verification found incomplete training, preflight, review or frozen-boundary evidence")
    inputs = _validate_source_inputs(config)
    camera_model = _new_optical_model(inputs["checkpoint"], 2048, torch.device("cpu"))
    camera_pixel_area_m2 = _camera_pixel_area_m2(camera_model)
    recorded = _json(resolved / "final_decision.json")
    recomputed = _consume_saved_measurements(
        config, resolved, inputs, camera_pixel_area_m2=camera_pixel_area_m2, write_outputs=False,
    )
    saved_gate = recorded["continuous_final_gate"]
    new_gate = recomputed["continuous_final_gate"]
    maxima_match = all(
        math.isclose(float(saved_gate["maxima"][key]), float(new_gate["maxima"][key]),
                     rel_tol=0.0, abs_tol=1.0e-14)
        for key in new_gate["maxima"]
    )
    quantized_match = _metric_tree_matches(
        recorded["quantized_terminal_measurements_not_gates"],
        recomputed["quantized_terminal_measurements_not_gates"],
    )
    power_match = _metric_tree_matches(recorded["slm_power_and_clipping"],
                                       recomputed["slm_power_and_clipping"])
    auxiliary_match = _metric_tree_matches(
        recorded["auxiliary_ED_EC_2048_continuous_only"],
        recomputed["auxiliary_ED_EC_2048_continuous_only"],
    )
    _require(recorded.get("task_id") == TASK_ID
             and recorded.get("run_id") == recomputed["run_id"]
             and recorded.get("status") == recomputed["status"]
             and recorded.get("lifecycle_state") == recomputed["lifecycle_state"]
             and run_identity.get("lifecycle_state") == recomputed["lifecycle_state"]
             and run_identity.get("scientific_status") == recomputed["status"]
             and task_state.get("lifecycle_state") == recomputed["lifecycle_state"]
             and recorded.get("decision") == recomputed["decision"]
             and recorded.get("population") == recomputed["population"]
             and saved_gate.get("passed_rows") == new_gate["passed_rows"]
             and saved_gate.get("denominator") == new_gate["denominator"]
             and saved_gate.get("all_rows_passed") == new_gate["all_rows_passed"]
             and maxima_match and quantized_match and power_match and auxiliary_match,
             "Windows reconsumption differs from the cloud's saved scientific decision")
    result = {
        "schema": "lightgen_g1_slm_input_mapping_windows_reconsumption.v1",
        "task_id": TASK_ID,
        "run_id": recomputed["run_id"],
        "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
        "decision": recomputed["decision"],
        "transfer_status": transfer["status"],
        "rows_recomputed": recomputed["population"]["rows_recomputed"],
        "continuous_final_gate": recomputed["continuous_final_gate"],
        "quantized_terminal_measurements_not_gates": recomputed["quantized_terminal_measurements_not_gates"],
        "slm_power_and_clipping": recomputed["slm_power_and_clipping"],
        "auxiliary_ED_EC_2048_continuous_only": recomputed["auxiliary_ED_EC_2048_continuous_only"],
        "all_file_sizes_verified_from_transfer_receipt": True,
        "not_a_hardware_acceptance": True,
    }
    del camera_model
    return result


def _validate_closed_attempt(evidence_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = _json(evidence_dir / "run_identity.json")
    state = _json(evidence_dir / "task_state.json")
    decision = _json(evidence_dir / "final_decision.json")
    terminal = torch.load(evidence_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    _require(identity.get("task_id") == state.get("task_id") == decision.get("task_id") == TASK_ID
             and identity.get("status") == "CLOSED" and state.get("phase") == "CLOSED"
             and state.get("completed_updates") == 5000
             and terminal.get("task_id") == TASK_ID
             and terminal.get("run_id") == identity.get("run_id") == decision.get("run_id")
             and terminal.get("completed_updates") == terminal.get("checkpoint_step") == 5000
             and decision.get("status") == "VALID_FAILED"
             and decision.get("decision") == "MAPPING_NOT_SUPPORTED_WITHIN_FIXED_5000_UPDATE_BUDGET",
             "closeout repair requires the exact closed 5,000-step VALID_FAILED attempt")
    history = _json(evidence_dir / "training_history.json").get("updates", [])
    _require(len(history) == 5000 and all(item.get("completed_update") == index + 1
                                           for index, item in enumerate(history)),
             "closed attempt training history is incomplete or misordered")
    return identity, decision, terminal


def _closeout_zero_update_check(
    source_dir: Path, inputs: Mapping[str, Any], camera_pixel_area_m2: float, device: torch.device,
) -> dict[str, Any]:
    shard = torch.load(source_dir / "measurements" / "holdout_batch_0000.pt",
                       map_location="cpu", weights_only=False)
    amplitude = shard["continuous_amplitude"][:1].to(device=device)
    target = shard["target_coefficients"][:1].to(device=device, dtype=torch.complex128)
    model = _new_optical_model(inputs["checkpoint"], 2048, device)
    model_before = _snapshot_state(model)
    evaluator_ed = copy.deepcopy(inputs["evaluator_ed"]).to(device).eval()
    evaluator_ec = copy.deepcopy(inputs["evaluator_ec"]).to(device).eval()
    ed_before, ec_before = _snapshot_state(evaluator_ed), _snapshot_state(evaluator_ec)
    with torch.no_grad():
        mapped = _forward_actual(model, amplitude.square().to(dtype=torch.float64))
        direct = _forward_direct(model, target)
        readout = _readout_for_evaluator(
            mapped["raw_cmos"], camera_pixel_area_m2, inputs["camera_gain"],
        )
        ed_logits = evaluator_ed(readout)
        _, ec_logits = evaluator_ec(readout)
    saved_actual = shard["conditions"]["2048"]["conditions"]["continuous"]
    saved_direct = shard["conditions"]["2048"]["direct_raw_cmos"][:1]
    coefficient_error = _numpy_phase_error(
        mapped["coefficients"][0].detach().cpu().numpy(),
        saved_actual["actual_coefficients"][0].numpy(),
    )[1]
    mapped_cmos_error = _numpy_relative_error(
        mapped["raw_cmos"][0].detach().cpu().numpy(),
        saved_actual["actual_raw_cmos"][0].numpy(),
    )
    direct_cmos_error = _numpy_relative_error(
        direct["raw_cmos"][0].detach().cpu().numpy(), saved_direct[0].numpy(),
    )
    raw_numpy = mapped["raw_cmos"].detach().cpu().numpy()
    independent_w_per_pixel = raw_numpy * camera_pixel_area_m2
    independent_gain_scaled = independent_w_per_pixel * float(inputs["camera_gain"])
    independent_fixed = np.clip(independent_gain_scaled, 0.0, 1.0)
    expected_readout = torch.nn.functional.adaptive_avg_pool2d(
        torch.from_numpy(independent_fixed).to(device=device, dtype=torch.float64), (28, 28),
    ).float()
    readout_delta = float((readout - expected_readout).abs().max().detach().cpu())
    _require(math.isfinite(float(ed_logits.max())) and math.isfinite(float(ec_logits.max()))
             and coefficient_error <= MEASUREMENT_LIMIT
             and mapped_cmos_error <= MEASUREMENT_LIMIT
             and direct_cmos_error <= MEASUREMENT_LIMIT
             and readout_delta <= 1.0e-7,
             "zero-update mapped/direct production replay or independent camera-unit check failed")
    _assert_state_unchanged(model, model_before, "C optics during closeout zero-update replay")
    _assert_state_unchanged(evaluator_ed, ed_before, "E_D during closeout zero-update replay")
    _assert_state_unchanged(evaluator_ec, ec_before, "E_C during closeout zero-update replay")
    result = {
        "executed": True,
        "passed": True,
        "scientific_updates": 0,
        "rows_replayed": 1,
        "path": "saved amplitude^2 -> frozen C encoder/OLS/decoder -> native raw CMOS -> W/pixel -> gain -> E_D/E_C",
        "camera_pixel_area_m2": camera_pixel_area_m2,
        "camera_gain_w_inverse": float(inputs["camera_gain"]),
        "coefficient_relative_error_one_global_phase": coefficient_error,
        "mapped_raw_cmos_relative_error": mapped_cmos_error,
        "direct_raw_cmos_relative_error": direct_cmos_error,
        "independent_numpy_readout_max_absolute_difference": readout_delta,
        "ED_prediction": int(ed_logits.argmax(dim=1)[0].detach().cpu()),
        "EC_prediction": int(ec_logits.argmax(dim=1)[0].detach().cpu()),
        "frozen_optics_and_evaluators": True,
    }
    del model, evaluator_ed, evaluator_ec
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _verify_saved_engineering_recovery(
    engineering_dir: Path, *, run_id: str, schedule: torch.Tensor,
) -> dict[str, Any]:
    receipt = _json(engineering_dir / "engineering_recovery.json")
    step3 = torch.load(
        engineering_dir / "checkpoints" / "checkpoint_step_000003.pt",
        map_location="cpu", weights_only=False,
    )
    live = torch.load(
        engineering_dir / "checkpoints" / "checkpoint_step_000004_live.pt",
        map_location="cpu", weights_only=False,
    )
    resumed = torch.load(engineering_dir / "worker_step4_restored.pt", map_location="cpu", weights_only=False)
    saved_history = _json(engineering_dir / "training_history.json").get("updates", [])
    worker_history = resumed.get("loss_history")
    step3_next = schedule[0, 3]
    step4_next = schedule[0, 4]
    identity_pass = all(
        payload.get("task_id") == TASK_ID and payload.get("run_id") == run_id
        for payload in (step3, live, resumed)
    )
    cursor_pass = (
        step3.get("completed_updates") == step3.get("checkpoint_step") == 3
        and step3.get("next_schedule_cursor") == 3
        and torch.is_tensor(step3.get("next_batch_indices"))
        and torch.equal(step3["next_batch_indices"], step3_next)
        and live.get("completed_updates") == live.get("checkpoint_step") == 4
        and live.get("next_schedule_cursor") == 4
        and torch.is_tensor(live.get("next_batch_indices"))
        and torch.equal(live["next_batch_indices"], step4_next)
        and resumed.get("completed_updates") == resumed.get("checkpoint_step") == 4
        and resumed.get("next_schedule_cursor") == 4
        and torch.is_tensor(resumed.get("next_batch_indices"))
        and torch.equal(resumed["next_batch_indices"], step4_next)
        and torch.is_tensor(resumed.get("step4_schedule_indices"))
        and torch.equal(resumed["step4_schedule_indices"], step3_next)
    )
    history_pass, history_delta = _allclose_tree(
        saved_history, worker_history[:3] if isinstance(worker_history, list) else None, atol=0.0,
    )
    step3_adam_steps = [
        int(torch.as_tensor(value["step"]).item())
        for value in step3.get("optimizer_state", {}).get("state", {}).values()
        if isinstance(value, Mapping) and "step" in value
    ]
    live_adam_steps = [
        int(torch.as_tensor(value["step"]).item())
        for value in live.get("optimizer_state", {}).get("state", {}).values()
        if isinstance(value, Mapping) and "step" in value
    ]
    resumed_adam_steps = [
        int(torch.as_tensor(value["step"]).item())
        for value in resumed.get("optimizer_state", {}).get("state", {}).values()
        if isinstance(value, Mapping) and "step" in value
    ]
    _require(isinstance(worker_history, list) and len(saved_history) == 3 and len(worker_history) == 4
             and all(item.get("completed_update") == index + 1
                     for index, item in enumerate(saved_history))
             and all(item.get("completed_update") == index + 1
                     for index, item in enumerate(worker_history))
             and identity_pass and cursor_pass and history_pass
             and len(step3_adam_steps) == len(live_adam_steps) == len(resumed_adam_steps) == 4
             and all(value == 3 for value in step3_adam_steps)
             and all(value == 4 for value in live_adam_steps + resumed_adam_steps)
             and live.get("loss_history_count") == resumed.get("loss_history_count") == 4
             and step3.get("optimizer_state_deepcopied") is True
             and live.get("optimizer_state_deepcopied") is True
             and resumed.get("optimizer_state_deepcopied") is True,
             "saved recovery checkpoints, schedule cursor, or complete update history are inconsistent")
    live_model_pass, model_delta = _allclose_tree(
        live["mapper_state"], resumed["mapper_state"], atol=RECOVERY_ABSOLUTE_TOLERANCE,
    )
    live_optimizer_pass, optimizer_delta = _allclose_tree(
        live["optimizer_state"], resumed["optimizer_state"], atol=RECOVERY_ABSOLUTE_TOLERANCE,
    )
    rng_pass, rng_delta = _allclose_tree(
        {"cpu": live["torch_cpu_rng"], "cuda": live["torch_cuda_rng"]},
        {"cpu": resumed["torch_cpu_rng"], "cuda": resumed["torch_cuda_rng"]}, atol=0.0,
    )
    loss_delta = abs(float(live["last_loss"]) - float(resumed["step4_loss"]))
    receipt_pass = (
        receipt.get("executed") is True and receipt.get("passed") is True
        and receipt.get("engineering_adam_calls") == 5
        and receipt.get("formal_restore_function") == "_restore_training_state"
        and receipt.get("save_preserved_rng") is True and receipt.get("rng_equal") is True
        and receipt.get("step3_history_count") == 3 and receipt.get("step4_history_count") == 4
    )
    _require(live_model_pass and live_optimizer_pass and rng_pass
             and loss_delta <= RECOVERY_ABSOLUTE_TOLERANCE and receipt_pass,
             "saved live-versus-restored recovery result fails independent evidence reconsumption")
    return {
        **receipt,
        "saved_recovery_evidence_reconsumed": True,
        "additional_engineering_adam_calls_during_reconsumption": 0,
        "model_max_absolute_difference_recomputed": model_delta,
        "optimizer_max_absolute_difference_recomputed": optimizer_delta,
        "history_prefix_max_absolute_difference": history_delta,
        "loss_absolute_difference_recomputed": loss_delta,
        "rng_equal_recomputed": rng_pass,
        "rng_max_absolute_difference_recomputed": rng_delta,
        "schedule_equal_recomputed": True,
    }


def _closeout_revision_required_files() -> set[str]:
    return {
        "implementation_review.json", "closeout_revision.json", "source_comparison.json", "final_decision.json",
        "per_sample_measurements.csv", "auxiliary_ED_EC_2048.json", "closeout_report.md",
        "run_config.yaml", "runner_source.py", "zero_update_camera_replay.json",
        "engineering_recovery_recheck.json", "engineering_recovery/training_history.json",
        "engineering_recovery/engineering_recovery.json",
        "engineering_recovery/checkpoints/checkpoint_step_000003.pt",
        "engineering_recovery/checkpoints/checkpoint_step_000004_live.pt",
        "engineering_recovery/worker_step4_restored.pt",
        "engineering_recovery/engineering_recovery_worker.stdout.log",
        "engineering_recovery/engineering_recovery_worker.stderr.log",
        "figures/continuous_2048_first_two_per_class.png",
        "figures/continuous_2048_first_two_per_class_manifest.json",
        "figures/continuous_2048_first_two_per_class_display_reference.npz",
    }


def _write_closeout_revision_report(
    output_dir: Path, decision: Mapping[str, Any], source_comparison: Mapping[str, Any],
    zero_update: Mapping[str, Any], recovery: Mapping[str, Any],
) -> None:
    maxima = decision["continuous_final_gate"]["maxima"]
    clip = decision["slm_power_and_clipping"]["fixed_camera_gain_clipping_fraction_2048_continuous"]
    ed = decision["auxiliary_ED_EC_2048_continuous_only"]["E_D"]["correct"]
    ec = decision["auxiliary_ED_EC_2048_continuous_only"]["E_C"]["correct"]
    quantized = decision["quantized_terminal_measurements_not_gates"]
    a8_cross = quantized["A8_amplitude_linear"]["cross_grid"]
    i8_cross = quantized["I8_intensity_linear"]["cross_grid"]
    text = f"""# G1 一次前向 SLM 输入映射试训收尾修订

## 结果与状态

工程收尾修订已完成。原运行 `{decision['run_id']}` 的科学状态保持 `VALID_FAILED / MAPPING_NOT_SUPPORTED_WITHIN_FIXED_5000_UPDATE_BUDGET`；本修订没有重新打开attempt，也没有追加正式训练更新。固定人口仍为2,000条，正式映射器终点仍为5,000步。

## 科学认识

连续输入映射主判据与原终点一致：2048/4096系数最大误差为{maxima['coefficient_2048']:.12g}/{maxima['coefficient_4096']:.12g}，原生CMOS最大误差为{maxima['native_cmos_2048']:.12g}/{maxima['native_cmos_4096']:.12g}；连续跨网格系数/CMOS最大差为{maxima['cross_grid_coefficient']:.12g}/{maxima['cross_grid_native_cmos']:.12g}。A8自身跨网格系数/CMOS最大差为{a8_cross['coefficient_relative_error_max']:.12g}/{a8_cross['native_cmos_relative_error_max']:.12g}；I8为{i8_cross['coefficient_relative_error_max']:.12g}/{i8_cross['native_cmos_relative_error_max']:.12g}，均由对应量化条件的两网格实际输出计算。相机辅助显示现统一按 `W/m² × {decision['continuous_final_gate']['auxiliary_camera_readout']['pixel_area_m2']:.12g} m²/像素 × 冻结增益` 计算；裁剪比例为{clip:.8%}，E_D/E_C在固定2,000行上分别正确{ed}/{ec}。辅助分类结果不进入损失或最终判据。原“裁剪100%、E_D/E_C均为200/2,000且全部判为类别3”的辅助解释已撤回。

## 目标缺口

结果仍只适用于固定G1留出人口、C45k数值光路、理想振幅SLM及本次5,000步预算。该修订不证明全部G1目标可制备，不证明真实光路、器件或相机已复现，也不提供人眼语义质量、硬件就绪或模型采用证据。

## 工程完成与验证

零更新生产重放及独立NumPy相机换算检查{'通过' if zero_update['passed'] else '未通过'}；恢复补验{'通过' if recovery['passed'] else '未通过'}，原进程连续执行三步后直接执行第四步，独立进程通过 `_restore_training_state` 从第三步检查点恢复后执行同一批次。恢复模型/Adam/损失绝对差分别为{recovery['model_max_absolute_difference']:.3g}/{recovery['optimizer_max_absolute_difference']:.3g}/{recovery['loss_absolute_difference']:.3g}，RNG与日程精确一致。工程调用为5次Adam，不计入正式5,000步。原200个测量分片被完整重算；A8/I8跨网格统计分别来自各自量化条件的两网格实际输出。原attempt和Windows备份未修改。Windows回传后从原始分片独立重算并逐像素核对样本图，记录在本修订的Windows复核收据中。

## 决策影响与下一步

本轮只修正测量工程和结果解释，连续映射候选仍未通过既定阈值，科学结论保持 `VALID_FAILED`。建议停止本试训；本授权不包含续训、调整阈值、推进G4、量化训练、人工评分或实物实验。任何后续科学工作需单独确定问题、人口和决策标准。
"""
    (output_dir / "closeout_report.md").write_text(text, encoding="utf-8")


def _run_closeout_repair(
    config_path: Path, config: Mapping[str, Any], source_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    _require(source_dir.resolve() == ATTEMPT_DIR.resolve(),
             "closeout repair must consume the explicit original AutoDL attempt directory")
    _require(output_dir.resolve() == CLOSEOUT_REPAIR_DIR.resolve(),
             "closeout repair output must be the named closeout_repair_01 directory")
    review_path = output_dir / "implementation_review.json"
    _require(output_dir.is_dir() and review_path.is_file(),
             "closeout output is missing its separate implementation review record", FileNotFoundError)
    present = {item.name for item in output_dir.iterdir()}
    base_outputs = {"implementation_review.json", "engineering_recovery", "resource_telemetry.csv"}
    allowed_outputs = base_outputs | {
        Path(relative).parts[0] for relative in _closeout_revision_required_files()
    }
    _require(present <= allowed_outputs,
             "closeout output contains unrecognized prior results; refusing to overwrite them")
    review = _json(review_path)
    _require(review.get("schema") == "lightgen_g1_slm_input_mapping_closeout_implementation_review.v1"
             and review.get("task_id") == TASK_ID and review.get("status") == "REVIEWED"
             and review.get("preflight_or_training_authority") is False,
             "separate closeout implementation review is missing or incomplete")
    identity, old_decision, terminal = _validate_closed_attempt(source_dir)
    inputs = _validate_source_inputs(config)
    preflight_dir = _preflight_dir(config)
    run_id = _run_id(preflight_dir)
    _require(identity.get("run_id") == run_id,
             "closed attempt and registered preflight identities differ")
    partial_outputs = present - base_outputs
    if partial_outputs:
        required_outputs = _closeout_revision_required_files()
        _require(not all((output_dir / relative).is_file() for relative in required_outputs),
                 "a complete closeout revision already exists; refusing to overwrite it")
        partial_decision_path = output_dir / "final_decision.json"
        _require(partial_decision_path.is_file(),
                 "partial closeout outputs lack their matching final-decision identity")
        partial_decision = _json(partial_decision_path)
        partial_revision = partial_decision.get("closeout_revision", {})
        partial_matches_source = (
            partial_decision.get("task_id") == TASK_ID
            and partial_decision.get("run_id") == run_id
            and partial_decision.get("status") == old_decision.get("status") == "VALID_FAILED"
            and partial_decision.get("decision") == old_decision.get("decision")
            and partial_decision.get("population") == old_decision.get("population")
            and partial_decision.get("updates") == old_decision.get("updates")
            and _continuous_primary_gate_matches(
                partial_decision.get("continuous_final_gate", {}), old_decision["continuous_final_gate"],
            )
            and _metric_tree_matches(
                partial_decision.get("per_class_continuous"), old_decision["per_class_continuous"],
            )
            and partial_revision.get("revision_id") == "closeout_repair_01"
            and partial_revision.get("parent_run_id") == run_id
            and partial_revision.get("formal_scientific_updates_added") == 0
            and partial_revision.get("scientific_attempt_reopened") is False
        )
        _require(partial_matches_source,
                 "partial closeout outputs do not match this fixed source; refusing to overwrite them")
    old_population, old_schedule = _load_attempt_population(source_dir)
    preflight_population, preflight_schedule = _load_saved_population(preflight_dir)
    for role in ("train", "validation", "holdout"):
        for key in ("coefficients", "labels", "proposal_ids"):
            _require(torch.equal(old_population[role][key], preflight_population[role][key]),
                     f"original {role} {key} differs from the fixed preflight population")
    _require(torch.equal(old_schedule, preflight_schedule),
             "original attempt training schedule differs from its fixed preflight schedule")
    _require(terminal.get("completed_updates") == 5000
             and old_decision["population"].get("rows_recomputed") == 2000,
             "closed scientific endpoint or fixed holdout population changed")
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "AutoDL CUDA is unavailable for the approved engineering recovery check")
    _set_numerics()
    camera_model = _new_optical_model(inputs["checkpoint"], 2048, torch.device("cpu"))
    camera_pixel_area_m2 = _camera_pixel_area_m2(camera_model)
    zero_update = _closeout_zero_update_check(source_dir, inputs, camera_pixel_area_m2, device)
    engineering_dir = output_dir / "engineering_recovery"
    if engineering_dir.exists():
        recovery = _verify_saved_engineering_recovery(
            engineering_dir, run_id=run_id, schedule=preflight_schedule,
        )
    else:
        recovery = _engineering_recovery(
            config_path, config, inputs, preflight_population, preflight_schedule, preflight_dir,
            engineering_dir, device,
        )
        recovery["saved_recovery_evidence_reconsumed"] = False
    audit = _consume_saved_measurements(
        config, source_dir, inputs, camera_pixel_area_m2=camera_pixel_area_m2,
        write_outputs=False, return_audit=True,
    )
    decision = audit["decision"]
    old_gate = old_decision["continuous_final_gate"]
    new_gate = decision["continuous_final_gate"]
    old_per_class = old_decision["per_class_continuous"]
    new_per_class = decision["per_class_continuous"]
    per_class_section_matches = {
        key: key in new_per_class and _metric_tree_matches(old_per_class[key], new_per_class[key])
        for key in old_per_class
    }
    primary_diagnostic = {
        "run_id_matches": decision.get("run_id") == identity.get("run_id"),
        "status_matches": decision.get("status") == old_decision.get("status") == "VALID_FAILED",
        "decision_matches": decision.get("decision") == old_decision.get("decision"),
        "population_matches": decision.get("population") == old_decision.get("population"),
        "updates_match": decision.get("updates") == old_decision.get("updates"),
        "continuous_primary_gate_matches": _continuous_primary_gate_matches(new_gate, old_gate),
        "per_class_sections_match": per_class_section_matches,
    }
    primary_matches = (
        all(value is True for key, value in primary_diagnostic.items()
            if key != "per_class_sections_match")
        and all(per_class_section_matches.values())
        and set(old_per_class) == set(new_per_class)
    )
    if not primary_matches:
        diagnostic_path = output_dir / "engineering_recovery" / "closeout_primary_comparison_diagnostic.json"
        _atomic_json(diagnostic_path, {
            "schema": "lightgen_g1_slm_closeout_primary_comparison_diagnostic.v1",
            "task_id": TASK_ID,
            "run_id": run_id,
            "matches": primary_diagnostic,
            "original_primary_gate": old_gate,
            "recomputed_primary_gate": new_gate,
            "per_class_section_differences": {
                key: {"original": old_per_class[key], "recomputed": new_per_class.get(key)}
                for key, matches in per_class_section_matches.items() if not matches
            },
            "original_primary_decision_metadata": {
                "status": old_decision.get("status"), "decision": old_decision.get("decision"),
                "population": old_decision.get("population"), "updates": old_decision.get("updates"),
            },
            "recomputed_primary_decision_metadata": {
                "status": decision.get("status"), "decision": decision.get("decision"),
                "population": decision.get("population"), "updates": decision.get("updates"),
            },
        })
    _require(primary_matches,
             "full shard reconsumption changed the original continuous scientific decision")
    auxiliary = decision["auxiliary_ED_EC_2048_continuous_only"]
    clip_fraction = decision["slm_power_and_clipping"][
        "fixed_camera_gain_clipping_fraction_2048_continuous"
    ]
    expected_aux_counts = (1526, 1590)
    observed_aux_counts = (auxiliary["E_D"]["correct"], auxiliary["E_C"]["correct"])
    review_expectations = {
        "clipping_fraction_matches_previous_independent_review": math.isclose(
            clip_fraction, 0.01140315, rel_tol=0.0, abs_tol=2.0e-8,
        ),
        "auxiliary_correct_counts_match_previous_independent_review": observed_aux_counts == expected_aux_counts,
        "these_are_engineering_crosschecks_not_scientific_gates": True,
    }
    _require(review_expectations["clipping_fraction_matches_previous_independent_review"]
             and review_expectations["auxiliary_correct_counts_match_previous_independent_review"],
             "corrected camera output differs from the independent unit/review cross-check")
    source_comparison = {
        "task_id": TASK_ID,
        "run_id": run_id,
        "old_scientific_status": old_decision["status"],
        "new_scientific_status": decision["status"],
        "old_decision": old_decision["decision"],
        "new_decision": decision["decision"],
        "original_primary_gate_matches_full_reconsumption": primary_matches,
        "original_continuous_maxima": old_decision["continuous_final_gate"]["maxima"],
        "recomputed_continuous_maxima": decision["continuous_final_gate"]["maxima"],
        "original_auxiliary_readout": old_decision["auxiliary_ED_EC_2048_continuous_only"],
        "corrected_auxiliary_readout": auxiliary,
        "review_expectations": review_expectations,
        "withdrawn_auxiliary_interpretation": "old 100% clipping and ED/EC 200/2000 all class 3 were caused by applying W^-1 directly to W/m^2",
        "original_attempt_files_modified": False,
    }
    decision["closeout_revision"] = {
        "revision_id": "closeout_repair_01",
        "parent_run_id": run_id,
        "source_attempt": str(source_dir),
        "formal_scientific_updates_added": 0,
        "engineering_adam_calls": recovery["engineering_adam_calls"],
        "scientific_attempt_reopened": False,
        "historical_status_preserved": "VALID_FAILED",
    }
    _atomic_json(output_dir / "final_decision.json", decision)
    _write_sample_csv(output_dir / "per_sample_measurements.csv", audit["records"])
    _save_first_two_contact_sheet(
        output_dir, audit["display_records"], camera_pixel_area_m2, inputs["camera_gain"],
    )
    _atomic_json(output_dir / "auxiliary_ED_EC_2048.json", {
        "schema": "lightgen_g1_slm_mapping_auxiliary_evaluators.v1",
        "task_id": TASK_ID,
        "scope": "frozen E_D/E_C on 2048 continuous mapped holdout only",
        "not_in_loss_or_final_gate": True,
        "metrics": auxiliary,
        "camera_readout_conversion": decision["continuous_final_gate"]["auxiliary_camera_readout"],
        "recomputed_from_saved_raw_cmos": True,
        "previous_wrong_unit_interpretation_withdrawn": True,
    })
    _atomic_json(output_dir / "zero_update_camera_replay.json", zero_update)
    _atomic_json(output_dir / "engineering_recovery_recheck.json", recovery)
    _atomic_json(output_dir / "source_comparison.json", source_comparison)
    _atomic_json(output_dir / "closeout_revision.json", {
        "schema": "lightgen_g1_slm_input_mapping_closeout_revision.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "status": "ENGINEERING_CLOSEOUT_REPAIRED",
        "source_attempt": str(source_dir),
        "source_run_identity": identity,
        "formal_scientific_updates_added": 0,
        "original_holdout_shards_recomputed": 200,
        "original_holdout_shards_copied": 0,
        "corrected_camera_pixel_area_m2": camera_pixel_area_m2,
        "camera_unit_check": zero_update,
        "engineering_recovery": recovery,
        "original_primary_gate_preserved": primary_matches,
        "quantized_cross_grid_metrics_use_each_condition_actual_output": True,
        "original_attempt_reopened": False,
        "original_attempt_modified": False,
        "required_return_files": sorted(_closeout_revision_required_files()),
    })
    _write_closeout_revision_report(output_dir, decision, source_comparison, zero_update, recovery)
    shutil.copy2(config_path, output_dir / "run_config.yaml")
    shutil.copy2(Path(__file__).resolve(), output_dir / "runner_source.py")
    del camera_model
    return {
        "status": "ENGINEERING_CLOSEOUT_REPAIRED",
        "task_id": TASK_ID,
        "run_id": run_id,
        "scientific_status": decision["status"],
        "decision": decision["decision"],
        "new_revision_dir": str(output_dir),
        "continuous_primary_gate_preserved": primary_matches,
        "engineering_adam_calls": recovery["engineering_adam_calls"],
        "formal_scientific_updates_added": 0,
    }


def _verify_closeout_revision(
    config: Mapping[str, Any], source_dir: Path, revision_dir: Path,
) -> dict[str, Any]:
    _require(source_dir.resolve() == WINDOWS_BACKUP_DIR.resolve(),
             "Windows closeout verification requires the exact preserved original backup")
    _require(revision_dir.resolve() == CLOSEOUT_REPAIR_BACKUP_DIR.resolve(),
             "Windows closeout verification requires the named new revision backup")
    attempt_required = {f"measurements/holdout_batch_{index:04d}.pt" for index in range(200)} | {
        "run_identity.json", "task_state.json", "fixed_population_and_schedule.pt", "terminal_checkpoint.pt",
        "training_history.json", "validation_trace.json", "final_decision.json",
        "per_sample_measurements.csv", "closeout_report.md", "auxiliary_ED_EC_2048.json",
        "preflight_receipt.json", "preflight_summary.json", "input_lineage.json",
        "zero_update_baseline.json", "engineering_recovery.json", "implementation_review.json",
        "mapper_initialization.pt", "mapper_parameter_update_audit.json", "progress.json",
        "preflight_identity.json", "run_config.yaml", "runner_source.py",
        "preflight_run_config.yaml", "preflight_runner_source.py",
        "checkpoints/checkpoint_step_000000.pt", "checkpoints/checkpoint_step_004500.pt",
        "checkpoints/checkpoint_step_005000.pt", "checkpoints/terminal_checkpoint.pt",
        "figures/continuous_2048_first_two_per_class.png",
        "figures/continuous_2048_first_two_per_class_manifest.json",
    }
    old_transfer = _verify_transfer_receipt(
        source_dir,
        remote_run_suffix="/outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/attempt_01",
        required_files=attempt_required,
    )
    revision_transfer = _verify_transfer_receipt(
        revision_dir,
        remote_run_suffix="/outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/closeout_repair_01",
        required_files=_closeout_revision_required_files(),
    )
    revision = _json(revision_dir / "closeout_revision.json")
    recovery = _json(revision_dir / "engineering_recovery_recheck.json")
    cloud_decision = _json(revision_dir / "final_decision.json")
    identity, old_decision, terminal = _validate_closed_attempt(source_dir)
    _require(revision.get("task_id") == TASK_ID and revision.get("run_id") == identity.get("run_id")
             and revision.get("status") == "ENGINEERING_CLOSEOUT_REPAIRED"
             and revision.get("original_attempt_modified") is False
             and recovery.get("passed") is True and recovery.get("engineering_adam_calls") == 5
             and cloud_decision.get("run_id") == identity.get("run_id")
             and cloud_decision.get("status") == "VALID_FAILED"
             and terminal.get("completed_updates") == 5000,
             "new closeout revision is incomplete or changes the original science status")
    inputs = _validate_source_inputs(config)
    camera_model = _new_optical_model(inputs["checkpoint"], 2048, torch.device("cpu"))
    camera_pixel_area_m2 = _camera_pixel_area_m2(camera_model)
    audit = _consume_saved_measurements(
        config, source_dir, inputs, camera_pixel_area_m2=camera_pixel_area_m2,
        write_outputs=False, return_audit=True,
    )
    local_decision = audit["decision"]
    expected_primary = (
        cloud_decision.get("status") == old_decision.get("status") == local_decision.get("status")
        and cloud_decision.get("decision") == old_decision.get("decision") == local_decision.get("decision")
        and cloud_decision.get("population") == old_decision.get("population") == local_decision.get("population")
        and _metric_tree_matches(cloud_decision["continuous_final_gate"], local_decision["continuous_final_gate"])
        and _continuous_primary_gate_matches(
            old_decision["continuous_final_gate"], local_decision["continuous_final_gate"],
        )
        and _metric_tree_matches(cloud_decision["per_class_continuous"], local_decision["per_class_continuous"])
        and _metric_tree_matches(cloud_decision["quantized_terminal_measurements_not_gates"],
                                 local_decision["quantized_terminal_measurements_not_gates"])
        and _metric_tree_matches(cloud_decision["slm_power_and_clipping"],
                                 local_decision["slm_power_and_clipping"])
        and _metric_tree_matches(cloud_decision["auxiliary_ED_EC_2048_continuous_only"],
                                 local_decision["auxiliary_ED_EC_2048_continuous_only"])
    )
    _require(expected_primary,
             "Windows full-population recomputation differs from cloud revision or original primary gate")

    windows_display_dir = revision_dir / "windows_recomputed"
    _require(not windows_display_dir.exists(),
             "Windows reconsumption output already exists; refusing to overwrite it", FileExistsError)
    _save_first_two_contact_sheet(
        windows_display_dir, audit["display_records"], camera_pixel_area_m2, inputs["camera_gain"],
    )
    cloud_display_path = revision_dir / "figures" / "continuous_2048_first_two_per_class_display_reference.npz"
    windows_display_path = windows_display_dir / "figures" / "continuous_2048_first_two_per_class_display_reference.npz"
    with np.load(cloud_display_path, allow_pickle=False) as cloud_display, \
            np.load(windows_display_path, allow_pickle=False) as windows_display:
        display_fields = tuple(cloud_display.files)
        _require(set(display_fields) == set(windows_display.files),
                 "Windows display pixel audit fields differ from the cloud reference")
        field_matches = {
            name: bool(np.array_equal(cloud_display[name], windows_display[name]))
            for name in display_fields
        }
    cloud_image = np.asarray(Image.open(
        revision_dir / "figures" / "continuous_2048_first_two_per_class.png",
    ).convert("RGB"))
    windows_image = np.asarray(Image.open(
        windows_display_dir / "figures" / "continuous_2048_first_two_per_class.png",
    ).convert("RGB"))
    pixel_crops_match = True
    for label in range(CLASS_COUNT):
        row_y = 8 + label * 226 + 18
        for sample_slot in range(2):
            mapped_index = label * 2 + sample_slot
            mapped_x = 8 + (sample_slot * 2 + 1) * 208
            direct_x = mapped_x + 208
            for x in (mapped_x, direct_x):
                pixel_crops_match = pixel_crops_match and np.array_equal(
                    cloud_image[row_y:row_y + 200, x:x + 200],
                    windows_image[row_y:row_y + 200, x:x + 200],
                )
            _require(mapped_index < 20, "fixed display sample index is outside the 20-row audit")
    display_pass = all(field_matches.values()) and pixel_crops_match
    _require(display_pass,
             "Windows raw CMOS, W/pixel, fixed-gain or rendered sample pixels differ from cloud output")
    display_result = {
        "executed": True,
        "passed": display_pass,
        "sample_rows": 20,
        "raw_cmos_pixel_arrays_equal": field_matches.get("mapped_raw_cmos_w_per_m2") is True
        and field_matches.get("direct_raw_cmos_w_per_m2") is True,
        "w_per_pixel_arrays_equal": field_matches.get("mapped_w_per_pixel") is True
        and field_matches.get("direct_w_per_pixel") is True,
        "fixed_gain_arrays_equal": field_matches.get("mapped_gain_scaled") is True
        and field_matches.get("direct_gain_scaled") is True,
        "uint8_display_arrays_equal": field_matches.get("mapped_display_u8") is True
        and field_matches.get("direct_display_u8") is True,
        "rendered_sample_crops_pixelwise_equal": pixel_crops_match,
        "field_matches": field_matches,
        "camera_pixel_area_m2": camera_pixel_area_m2,
        "camera_gain_w_inverse": float(inputs["camera_gain"]),
    }
    result = {
        "schema": "lightgen_g1_slm_input_mapping_windows_closeout_reconsumption.v1",
        "task_id": TASK_ID,
        "run_id": identity["run_id"],
        "status": "WINDOWS_CLOSEOUT_RECONSUMPTION_PASSED",
        "decision": local_decision["decision"],
        "original_transfer_status": old_transfer["status"],
        "revision_transfer_status": revision_transfer["status"],
        "original_transfer_files_actual_sizes_verified": old_transfer["files_verified_against_actual_size"],
        "revision_transfer_files_actual_sizes_verified": revision_transfer["files_verified_against_actual_size"],
        "rows_recomputed": local_decision["population"]["rows_recomputed"],
        "primary_and_all_auxiliary_metrics_match": expected_primary,
        "continuous_final_gate": local_decision["continuous_final_gate"],
        "quantized_terminal_measurements_not_gates": local_decision["quantized_terminal_measurements_not_gates"],
        "slm_power_and_clipping": local_decision["slm_power_and_clipping"],
        "auxiliary_ED_EC_2048_continuous_only": local_decision["auxiliary_ED_EC_2048_continuous_only"],
        "display_pixel_audit": display_result,
        "original_attempt_modified": False,
        "scientific_updates_added": 0,
        "not_a_hardware_acceptance": True,
    }
    _atomic_json(revision_dir / "windows_display_pixel_audit.json", display_result)
    _atomic_json(revision_dir / "windows_closeout_reconsumption.json", result)
    del camera_model
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "execute", "resume", "measure", "verify-evidence",
                                 "engineering-worker", "closeout-repair", "verify-closeout-revision"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--source-evidence-dir", type=Path)
    parser.add_argument("--revision-evidence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)

    if args.mode == "engineering-worker":
        _require(args.preflight_dir is not None and args.checkpoint is not None and args.worker_output is not None,
                 "engineering worker needs explicit preflight, checkpoint and output paths", ValueError)
        _engineering_worker(config_path, args.preflight_dir.resolve(), args.checkpoint.resolve(),
                            args.worker_output.resolve())
        return 0
    if args.mode == "preflight":
        print(json.dumps(_run_preflight(config_path, config), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode in {"execute", "resume"}:
        if args.mode == "resume":
            _require(args.checkpoint is not None,
                     "resume requires an explicit checkpoint path; automatic-latest is disabled", ValueError)
            checkpoint = args.checkpoint.resolve()
        else:
            _require(args.checkpoint is None,
                     "execute starts only from the registered initialization; use resume for a checkpoint",
                     ValueError)
            checkpoint = None
        result = _execute_or_resume(config_path, config, checkpoint)
        if not result["training_complete"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 130
        inputs = _validate_source_inputs(config)
        decision = _run_final_measurement(config, _attempt_dir(config), inputs)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "measure":
        _require(args.checkpoint is None,
                 "measure reloads only the fixed terminal checkpoint", ValueError)
        state_path = _attempt_dir(config) / "task_state.json"
        _require(state_path.is_file() and _json(state_path).get("phase") != "CLOSED",
                 "closed evidence is read-only; use verify-evidence", FileNotFoundError)
        inputs = _validate_source_inputs(config)
        decision = _run_final_measurement(config, _attempt_dir(config), inputs)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "verify-evidence":
        _require(args.evidence_dir is not None,
                 "verify-evidence requires the explicit Windows backup path", ValueError)
        result = _verify_saved_evidence(config, args.evidence_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "closeout-repair":
        _require(args.source_evidence_dir is not None and args.output_dir is not None,
                 "closeout-repair requires explicit source evidence and new output directories", ValueError)
        result = _run_closeout_repair(
            config_path, config, args.source_evidence_dir.resolve(), args.output_dir.resolve(),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode == "verify-closeout-revision":
        _require(args.source_evidence_dir is not None and args.revision_evidence_dir is not None,
                 "verify-closeout-revision requires explicit original and revised evidence directories", ValueError)
        result = _verify_closeout_revision(
            config, args.source_evidence_dir.resolve(), args.revision_evidence_dir.resolve(),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
