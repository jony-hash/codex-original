"""Run the frozen-C G1 projected SLM input-mapping diagnostic."""

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

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    RUN_COMPLETED,
    RUN_SCIENTIFIC_ACTIVE,
    RUN_VALID_FAILED,
)
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reach
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_SLM_PROJECTED_MAPPING_V2"
SCHEMA = "lightgen_g1_slm_projected_mapping.v2"
POPULATION_SCHEMA = "lightgen_g1_slm_projected_mapping_population.v2"
CHECKPOINT_SCHEMA = "lightgen_g1_slm_projected_mapping_checkpoint.v2"
MODULE_PATH = f"scripts.research.lightgen.{Path(__file__).stem}"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_slm_projected_mapping_v2.yaml"
PREFLIGHT_DIR = ROOT / "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/preflight_01"
ATTEMPT_DIR = ROOT / "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/attempt_01"
CLASS_COUNT = 10
PORT_COUNT = 100
REAL_DIM = 200
PIXEL_COUNT = 40_000
IMAGE_SHAPE = (200, 200)
GRIDS = (2048, 4096)
GATE_LIMIT = 1.0e-4
MATRIX_LIMIT = 1.0e-6
RECOVERY_TOLERANCE = 5.0e-8
AUTHORITY = "USER_AUTHORIZED_G1_SLM_PROJECTED_MAPPING_20261027"


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temp)
    os.replace(temp, path)


def _load_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(), "projected G1 task uses its fixed configuration", ValueError)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "configuration must be a mapping", ValueError)
    _validate_config(value)
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == SCHEMA and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY
             and config.get("evidence_tier") == "diagnostic_only",
             "projected mapping identity or authorization changed", ValueError)
    expected_sources = {
        "checkpoint": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01/C/checkpoint_step_010000.pt",
        "fitted_gaussian_prior": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01/C/fitted_gaussian_prior.pt",
        "source_references": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01/source_references.pt",
        "baseline_evidence_dir": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01",
        "replication_evidence_dir": "outputs/20261021_lightgen_small_gaussian_replication_prior_confirmation_v1/runs/autodl_backup_attempt_01",
        "reachability_attempt_dir": "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01",
        "source_population_evidence_dir": "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/autodl_backup_attempt_01",
        "source_population": "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/autodl_backup_attempt_01/fixed_population_and_schedule.pt",
        "source_mapper_checkpoint": "outputs/20261026_lightgen_g1_slm_input_mapping_pilot_v1/runs/autodl_backup_attempt_01/terminal_checkpoint.pt",
        "response_matrix": "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01/response_matrix.pt",
        "evaluator_ed_checkpoint": "outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt",
        "evaluator_ec_checkpoint": "outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_03/evaluator_checkpoint.pt",
    }
    _require(config["source"] == expected_sources, "explicit frozen source paths changed", ValueError)
    optics = config["optics"]
    _require(optics.get("checkpoint_branch") == "C" and optics.get("checkpoint_seed") == 2026101501
             and optics.get("cumulative_step") == 45000 and optics.get("source_power_w") == 2.493e-4
             and optics.get("layer_power_transmission") == 0.889 and optics.get("slm_shape") == [200, 200]
             and optics.get("encoder_layers") == 2 and optics.get("decoder_layers") == 2
             and optics.get("ols_basis") == "gaussian_s12_v1" and optics.get("ols_ports") == 100
             and optics.get("ols_transport") == "identity"
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED"
             and optics.get("encoder_shape") == [100, 100]
             and optics.get("decoder_shape") == [100, 100]
             and optics.get("training_grid") == 2048 and optics.get("measurement_grids") == list(GRIDS),
             "frozen C optical contract changed", ValueError)
    population = config["population"]
    _require(population.get("classes") == 10 and population.get("train_per_class") == 1000
             and population.get("validation_per_class") == 200
             and population.get("final_holdout_per_class") == 200
             and population.get("final_holdout_seed_base") == 2026102750
             and population.get("rng_seed") == 2026102770
             and population.get("schedule_seed") == 2026102771
             and population.get("max_proposals_per_class") == 1_000_000
             and population.get("proposal_batch_size") == 256
             and population.get("rejection") == "original_total_modal_power_only",
             "fixed G1 population or rejection rule changed", ValueError)
    training = config["training"]
    _require(training.get("maximum_updates") == 5000 and training.get("epochs") == 5
             and training.get("logical_batch_size") == 10 and training.get("samples_per_class_per_batch") == 1
             and training.get("mapper") == "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid"
             and training.get("optimizer") == "Adam" and training.get("learning_rate") == 1.0e-4
             and training.get("betas") == [0.9, 0.999] and training.get("epsilon") == 1.0e-8
             and training.get("weight_decay") == 0.0 and training.get("rng_seed") == 2026102770
             and training.get("schedule_seed") == 2026102771
             and population.get("rng_seed") == training.get("rng_seed")
             and population.get("schedule_seed") == training.get("schedule_seed")
             and training.get("mapper_precision") == "float32"
             and training.get("fixed_projection_precision") == "float64"
             and training.get("optical_precision") == "float64_complex128"
             and training.get("amp") is False and training.get("tf32") is False
             and training.get("checkpoint_interval_steps") == 500
             and training.get("progress_interval_steps") == 100
             and training.get("validation_steps") == [0, 1000, 2500, 5000]
             and training.get("select_best_validation_checkpoint") is False,
             "fixed projected mapper training contract changed", ValueError)
    measurement = config["measurement"]
    _require(measurement.get("relative_error_max") == GATE_LIMIT
             and measurement.get("coefficient_phase_alignment") == "one_global_phase"
             and measurement.get("cmos_normalization") == "none"
             and measurement.get("cmos_power_matching") is False
             and measurement.get("cmos_gain_adjustment") is False
             and measurement.get("measure_a8_i8_only_if_continuous_passes") is True,
             "the single terminal measurement rule changed", ValueError)
    _require(config["output"].get("experiment_root") == ATTEMPT_DIR.relative_to(ROOT).as_posix()
             and config["output"].get("preflight_root") == PREFLIGHT_DIR.relative_to(ROOT).as_posix(),
             "fixed output paths changed", ValueError)
    reachability = config["reachability"]
    _require(reachability == {
        "probe_per_class": 2,
        "phase_starts": 8,
        "max_steps_per_start": 5000,
        "coefficient_stop_relative_error": 1.0e-6,
    }, "fixed reachability witness budget changed", ValueError)
    _require(measurement.get("measure_a8_i8_only_if_continuous_passes") is True
             and measurement.get("quantization_aware_training") is False
             and measurement.get("continuous_is_primary") is True
             and measurement.get("evaluator_scope") == "frozen_ED_EC_auxiliary_at_2048_only",
             "fixed quantization and evaluator scope changed", ValueError)


def _legacy_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(config["source"])
    optics = config["optics"]
    pop = config["population"]
    return {
        "source": source,
        "optics": {
            "source_power_w": optics["source_power_w"],
            "layer_power_transmission": optics["layer_power_transmission"],
            "slm_shape": optics["slm_shape"],
            "encoder_layers": optics["encoder_layers"],
            "decoder_layers": optics["decoder_layers"],
            "ols_basis": optics["ols_basis"],
            "ols_ports": optics["ols_ports"],
            "ols_transport": optics["ols_transport"],
            "geometry_status": optics["geometry_status"],
            "training_grid": optics["training_grid"],
            "measurement_grids": optics["measurement_grids"],
        },
        "population": {
            "train_per_class": pop["train_per_class"],
            "validation_per_class": pop["validation_per_class"],
            "holdout_per_class": pop["final_holdout_per_class"],
            "train_seed_base": 2026102610,
            "validation_seed_base": 2026102630,
            "holdout_seed_base": pop["final_holdout_seed_base"],
            "initialization_seed": 2026102670,
            "schedule_seed": pop["schedule_seed"],
            "max_proposals_per_class_per_population": pop["max_proposals_per_class"],
            "proposal_batch_size": pop["proposal_batch_size"],
            "max_modal_power_w": base.MAX_MODAL_POWER_W,
        },
    }


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = pilot._validate_source_inputs(_legacy_validation_config(config))
    population_path = pilot._resolve(config["source"]["source_population"])
    mapper_path = pilot._resolve(config["source"]["source_mapper_checkpoint"])
    matrix_path = pilot._resolve(config["source"]["response_matrix"])
    for path in (population_path, mapper_path, matrix_path):
        _require(path.is_file(), f"explicit projected-mapping source is missing: {path}", FileNotFoundError)
    source_population = torch.load(population_path, map_location="cpu", weights_only=False)
    _require(source_population.get("schema") == "lightgen_g1_slm_mapping_population.v1"
             and source_population.get("task_id") == pilot.TASK_ID,
             "reused source coefficient population is not the closed original G1 pilot population")
    expected_seed_bases = {"train": 2026102610, "validation": 2026102630}
    for cohort, per_class in (("train", 1000), ("validation", 200)):
        item = source_population[cohort]
        _require(tuple(item["coefficients"].shape) == (10 * per_class, 100)
                 and tuple(item["labels"].shape) == (10 * per_class,)
                 and torch.equal(item["labels"], torch.arange(10).repeat_interleave(per_class))
                 and torch.is_complex(item["coefficients"])
                 and torch.isfinite(item["coefficients"].real).all().item()
                 and torch.isfinite(item["coefficients"].imag).all().item(),
                 f"reused {cohort} coefficients have wrong roles, labels, or values")
        _require(item.get("seeds") == [expected_seed_bases[cohort] + label for label in range(10)]
                 and tuple(item["proposal_ids"].shape) == (10 * per_class,)
                 and tuple(item["coefficients_by_class"].shape) == (10, per_class, 100),
                 f"reused {cohort} source seed or per-class identity changed")
        for label in range(10):
            ids = item["proposal_ids"][label * per_class:(label + 1) * per_class]
            _require(torch.unique(ids).numel() == per_class
                     and torch.all(item["coefficients"][label * per_class:(label + 1) * per_class]
                                   .abs().square().sum(dim=1) <= base.MAX_MODAL_POWER_W),
                     f"reused {cohort} class-{label} proposal IDs or power rule changed")
    source_mapper = torch.load(mapper_path, map_location="cpu", weights_only=False)
    _require(source_mapper.get("task_id") == pilot.TASK_ID
             and source_mapper.get("checkpoint_step") == 5000
             and source_mapper.get("completed_updates") == 5000
             and source_mapper.get("optimizer_state_deepcopied") is True
             and set(source_mapper["mapper_state"]) == {
                 "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias",
             }, "explicit warm-start mapper is not the original 5,000-step endpoint")
    source_optimizer_steps = [int(torch.as_tensor(value["step"]).item())
                              for value in source_mapper["optimizer_state"]["state"].values()]
    _require(len(source_optimizer_steps) == 4 and all(step == 5000 for step in source_optimizer_steps),
             "original mapper checkpoint Adam state is not at update 5,000")
    response_payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
    response = response_payload.get("matrix")
    _require(response_payload.get("task_id") == reach.TASK_ID and torch.is_tensor(response)
             and tuple(response.shape) == (100, 40000) and response.dtype == torch.complex128
             and torch.isfinite(response.real).all().item() and torch.isfinite(response.imag).all().item(),
             "saved frozen C amplitude response matrix is malformed or from another task")
    return {
        **inputs,
        "population_path": population_path,
        "source_population": source_population,
        "source_mapper_path": mapper_path,
        "source_mapper": source_mapper,
        "response_matrix_path": matrix_path,
        "response_matrix": response.contiguous(),
    }


class AmplitudeMapper(torch.nn.Module):
    """Original MLP; the fixed projection remains a non-trainable forward operation."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(REAL_DIM, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, PIXEL_COUNT),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != REAL_DIM:
            raise ValueError("projected mapper input must be [batch,200]")
        return self.network(features).reshape(-1, 1, *IMAGE_SHAPE)


def _new_mapper(device: torch.device, state: Mapping[str, torch.Tensor]) -> AmplitudeMapper:
    mapper = AmplitudeMapper().to(device=device, dtype=torch.float32)
    mapper.load_state_dict(state, strict=True)
    return mapper


def _normalized_features(coefficients: torch.Tensor, scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    normalized = coefficients.to(device=device, dtype=torch.complex128) / scale.to(
        device=device, dtype=torch.float64
    ).sqrt().unsqueeze(0)
    result = torch.view_as_real(normalized).reshape(coefficients.shape[0], REAL_DIM).float()
    _require(torch.isfinite(result).all().item(), "normalized G1 features contain nonfinite values")
    return result


def _normalized_response(response: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Arrange one port at a time, then its real and imaginary response rows;
    # this matches ``view_as_real(...).reshape(batch, REAL_DIM)`` for targets.
    paired = torch.view_as_real(response).permute(0, 2, 1).reshape(REAL_DIM, PIXEL_COUNT)
    row_scale = scale.double().sqrt().repeat_interleave(2)
    return (paired / row_scale[:, None]).contiguous()


def _pseudoinverse(matrix: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    _require(matrix.dtype == torch.float64 and tuple(matrix.shape) == (REAL_DIM, PIXEL_COUNT),
             "normalized C response matrix must be float64 [200,40000]")
    left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
    _require(tuple(singular.shape) == (REAL_DIM,) and torch.isfinite(singular).all().item()
             and float(singular.min()) > 0.0, "normalized response matrix is not full row rank")
    inverse = (right_h.mT / singular[None, :]) @ left.mT
    identity_error = float((matrix @ inverse - torch.eye(REAL_DIM, dtype=torch.float64,
                                                         device=matrix.device)).abs().max().item())
    _require(identity_error <= 1.0e-10,
             f"full-rank fixed right inverse residual {identity_error:.3g} exceeds 1e-10")
    return inverse.contiguous(), {
        "minimum_singular_value": float(singular.min().item()),
        "maximum_singular_value": float(singular.max().item()),
        "condition_number": float((singular.max() / singular.min()).item()),
        "max_abs_AP_minus_I": identity_error,
    }


def _project_double_from_raw(
    raw: torch.Tensor, target: torch.Tensor, matrix: torch.Tensor, inverse: torch.Tensor,
    scale: torch.Tensor, *, include_preclip: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = raw.shape[0]
    command = raw.reshape(batch, PIXEL_COUNT).double()
    predicted_ri = command @ matrix.T
    predicted = torch.view_as_complex(predicted_ri.reshape(batch, PORT_COUNT, 2).contiguous())
    predicted = predicted * scale.to(device=raw.device, dtype=torch.float64).sqrt().unsqueeze(0)
    correlation = (target.conj() * predicted).sum(dim=1)
    safe_correlation = torch.where(correlation.abs() > 0.0, correlation,
                                   torch.ones_like(correlation))
    phase = torch.angle(safe_correlation)
    aligned = target * torch.exp(1j * phase).unsqueeze(1)
    normalized_target = aligned / scale.to(device=raw.device, dtype=torch.float64).sqrt().unsqueeze(0)
    target_ri = torch.view_as_real(normalized_target).reshape(batch, REAL_DIM)
    residual = target_ri - predicted_ri
    corrected = command + residual @ inverse.T
    clipped = corrected.clamp(0.0, 1.0)
    detail = {
        "phase": phase,
        "preclip_min": corrected.amin(dim=1),
        "preclip_max": corrected.amax(dim=1),
        "clipped_fraction": ((corrected < 0.0) | (corrected > 1.0)).double().mean(dim=1),
        "correlation_abs": correlation.abs(),
    }
    if include_preclip:
        detail["preclip_amplitude"] = corrected.reshape(batch, 1, *IMAGE_SHAPE)
    return clipped.reshape(batch, 1, *IMAGE_SHAPE), detail


def _project_from_raw(
    raw: torch.Tensor, target: torch.Tensor, matrix: torch.Tensor, inverse: torch.Tensor,
    scale: torch.Tensor, *, include_preclip: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    amplitude, detail = _project_double_from_raw(
        raw, target, matrix, inverse, scale, include_preclip=include_preclip,
    )
    return amplitude.to(torch.float32), detail


def _map_amplitude(
    mapper: AmplitudeMapper, features: torch.Tensor, target: torch.Tensor,
    matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor, *, include_preclip: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw = mapper(features)
    return _project_from_raw(raw, target, matrix, inverse, scale, include_preclip=include_preclip)


def _build_population(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    source = inputs["source_population"]
    population: dict[str, Any] = {"schema": POPULATION_SCHEMA, "task_id": TASK_ID}
    for key in ("train", "validation"):
        population[key] = {name: source[key][name] for name in
                           ("coefficients", "labels", "proposal_ids", "rows", "coefficients_by_class", "seeds")}
        population[key]["source_task_id"] = pilot.TASK_ID
        population[key]["source_role"] = f"reused_fixed_{key}_coefficients_only"
    per_class = int(config["population"]["final_holdout_per_class"])
    holdout_coefficients: list[torch.Tensor] = []
    holdout_labels: list[torch.Tensor] = []
    holdout_proposals: list[torch.Tensor] = []
    accounting: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        values, proposal_ids, record = pilot._sample_g1_class(
            inputs["prior"][label], inputs["scale"], label=label,
            seed=int(config["population"]["final_holdout_seed_base"]) + label,
            count=per_class,
            proposal_limit=int(config["population"]["max_proposals_per_class"]),
            proposal_batch_size=int(config["population"]["proposal_batch_size"]),
        )
        holdout_coefficients.append(values)
        holdout_labels.append(torch.full((per_class,), label, dtype=torch.long))
        holdout_proposals.append(proposal_ids)
        accounting[str(label)] = record
    coefficients = torch.cat(holdout_coefficients).contiguous()
    labels = torch.cat(holdout_labels)
    proposals = torch.cat(holdout_proposals)
    _require(tuple(coefficients.shape) == (2000, PORT_COUNT)
             and torch.equal(labels, torch.arange(CLASS_COUNT).repeat_interleave(per_class))
             and torch.all(coefficients.abs().square().sum(dim=1) <= base.MAX_MODAL_POWER_W),
             "new final holdout is incomplete or violates the original power-only rejection")
    population["holdout"] = {
        "coefficients": coefficients,
        "labels": labels,
        "proposal_ids": proposals,
        "rows": [{"label": label, "sample_index_within_class": index,
                  "proposal_id": int(proposals[label * per_class + index]),
                  "normal_seed": int(config["population"]["final_holdout_seed_base"]) + label}
                 for label in range(CLASS_COUNT) for index in range(per_class)],
        "seeds": [int(config["population"]["final_holdout_seed_base"]) + label
                  for label in range(CLASS_COUNT)],
        "source_role": "new_fixed_final_holdout_not_consumed_during_route_selection",
    }
    _require(all(not torch.equal(population[left]["coefficients"], population[right]["coefficients"])
                 for left, right in (("train", "validation"), ("train", "holdout"),
                                     ("validation", "holdout"))),
             "training, validation and new final holdout are not independent")
    population["proposal_accounting"] = accounting
    population["rejection_rule"] = "one saved full-covariance G1 Gaussian per class; modal power only"
    population["power_bound_w"] = base.MAX_MODAL_POWER_W
    schedule_config = {"population": {
        "train_per_class": 1000,
        "schedule_seed": int(config["population"]["schedule_seed"]),
    }}
    schedule = pilot._build_schedule(population, schedule_config)
    return population, schedule


def _set_numerics() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _run_identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    semantics = json.dumps({
        "task_id": TASK_ID,
        "authority": AUTHORITY,
        "source_checkpoint": config["source"]["checkpoint"],
        "source_population": config["source"]["source_population"],
        "response_matrix": config["source"]["response_matrix"],
        "optics": config["optics"],
        "population": config["population"],
        "training": config["training"],
        "measurement": config["measurement"],
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{run_id}:G1_projected_mapper",
        semantic_authority_id=AUTHORITY,
        data_role="SYNTHETIC_G1_TRAIN_VALIDATION_NEW_HOLDOUT",
        config_identity=dict(config),
        model_identity={
            "frozen_optics": "C_seed_2026101501_step_45000",
            "response": "saved complex128 amplitude-to-coefficient B and normalized real A",
            "projector": "full_row_rank Moore-Penrose right inverse; no truncated directions",
            "mapper": "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid + fixed projection",
            "evidence_tier": "diagnostic_only",
        },
        optimizer_identity={
            "kind": "Adam",
            "learning_rate": 1.0e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "new_updates_max": 5000,
        },
        sampler_identity={
            "train": "reused fixed 10,000-row source population",
            "validation": "reused fixed 2,000-row source population",
            "holdout": "new fixed 2,000-row G1 population; seeds 2026102750+y",
        },
        execution_source_semantics_id=semantics,
    )


def _load_control(preflight_dir: Path) -> tuple[ResearchRunControl, PreflightReceipt]:
    identity_payload = _json(preflight_dir / "preflight_identity.json")
    _require(identity_payload.get("task_id") == TASK_ID, "preflight belongs to another task")
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    receipt = PreflightReceipt.from_mapping(_json(preflight_dir / "preflight_receipt.json"))
    _require(identity_payload.get("run_id") == identity.run_id
             and identity.task_id == TASK_ID
             and receipt.source_semantics_id == identity.execution_source_semantics_id,
             "preflight identity and READY receipt do not describe the same fixed task")
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    return control, receipt


def _assert_attempt_identity(attempt_dir: Path) -> ResearchRunIdentity:
    preflight = _json(attempt_dir / "preflight_evidence" / "preflight_identity.json")
    expected = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    attempt = ResearchRunIdentity.from_mapping(_json(attempt_dir / "run_identity.json"))
    attempt_id = f"{expected.run_id}:attempt_01"
    expected = expected.with_scientific_attempt(attempt_id)
    _require(preflight.get("task_id") == TASK_ID and preflight.get("run_id") == expected.run_id
             and attempt.execution_semantics() == expected.execution_semantics(),
             "attempt identity differs from its consumed READY preflight")
    state_path = attempt_dir / "task_state.json"
    if state_path.is_file():
        state = _json(state_path)
        _require(state.get("task_id") == TASK_ID and state.get("run_id") == expected.run_id,
                 "attempt state file belongs to another run")
    receipt = PreflightReceipt.from_mapping(
        _json(attempt_dir / "preflight_evidence" / "preflight_receipt.json")
    )
    _require(receipt.authorization_id == AUTHORITY and receipt.status == "READY"
             and receipt.source_semantics_id == expected.execution_source_semantics_id,
             "attempt has no matching READY receipt from the fixed preflight")
    return attempt


def _load_population(population_path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(population_path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == POPULATION_SCHEMA and payload.get("task_id") == TASK_ID,
             "saved projected-mapping population identity mismatch")
    schedule = payload.pop("training_schedule")
    expected = {
        "train": (10_000, 1000, [2026102610 + label for label in range(CLASS_COUNT)]),
        "validation": (2_000, 200, [2026102630 + label for label in range(CLASS_COUNT)]),
        "holdout": (2_000, 200, [2026102750 + label for label in range(CLASS_COUNT)]),
    }
    for role, (count, per_class, seeds) in expected.items():
        cohort = payload.get(role)
        _require(isinstance(cohort, Mapping), f"saved {role} population is missing")
        coefficients = cohort.get("coefficients")
        labels = cohort.get("labels")
        proposal_ids = cohort.get("proposal_ids")
        _require(torch.is_tensor(coefficients) and torch.is_complex(coefficients)
                 and tuple(coefficients.shape) == (count, PORT_COUNT)
                 and torch.isfinite(coefficients.real).all().item()
                 and torch.isfinite(coefficients.imag).all().item()
                 and torch.is_tensor(labels) and tuple(labels.shape) == (count,)
                 and torch.equal(labels, torch.arange(CLASS_COUNT).repeat_interleave(per_class))
                 and torch.is_tensor(proposal_ids) and tuple(proposal_ids.shape) == (count,)
                 and cohort.get("seeds") == seeds
                 and bool(torch.all(coefficients.abs().square().sum(dim=1) <= base.MAX_MODAL_POWER_W)),
                 f"saved {role} coefficients, labels, proposal IDs, seed roles, or power boundary changed")
        for label in range(CLASS_COUNT):
            ids = proposal_ids[label * per_class:(label + 1) * per_class]
            _require(torch.unique(ids).numel() == per_class,
                     f"saved {role} class-{label} proposal IDs are not unique")
    _require(payload["train"].get("source_role") == "reused_fixed_train_coefficients_only"
             and payload["validation"].get("source_role") == "reused_fixed_validation_coefficients_only"
             and payload["holdout"].get("source_role") == "new_fixed_final_holdout_not_consumed_during_route_selection",
             "saved coefficient population roles changed")
    _require(torch.is_tensor(schedule) and schedule.dtype == torch.long
             and tuple(schedule.shape) == (5, 1000, CLASS_COUNT),
             "saved fixed five-pass sample schedule is malformed")
    expected_schedule = pilot._build_schedule(
        payload, {"population": {"train_per_class": 1000, "schedule_seed": 2026102771}},
    )
    _require(torch.equal(schedule, expected_schedule),
             "saved five-pass schedule differs from the fixed seed and training population")
    return payload, schedule


def _camera_pixel_area(model: base.SmallMatchedAdapter) -> float:
    pitch = model.runtime.camera_plane.pixel_pitch_m
    _require(len(pitch) == 2 and all(math.isfinite(float(value)) and float(value) > 0.0 for value in pitch),
             "frozen camera plane has invalid pixel pitch")
    area = float(pitch[0]) * float(pitch[1])
    _require(math.isclose(area, 6.25e-12, rel_tol=1.0e-12, abs_tol=0.0),
             "frozen camera pixel area changed")
    return area


def _forward_amplitude(model: base.SmallMatchedAdapter, amplitude: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(amplitude.ndim == 4 and tuple(amplitude.shape[1:]) == (1, *IMAGE_SHAPE)
             and not torch.is_complex(amplitude) and torch.isfinite(amplitude).all().item()
             and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0,
             "SLM amplitude command is malformed or outside [0,1]")
    # The stored command remains float32; converting its exact values at the
    # optical boundary keeps SLM -> CMOS propagation in float64/complex128.
    result = model.runtime.forward_from_amplitude(amplitude.to(dtype=torch.float64))
    coefficients = result.z_ols.coefficients
    raw = result.raw_camera.tensor
    w_per_pixel = result.physical_sensor_output_w_per_pixel
    _require(tuple(coefficients.shape) == (amplitude.shape[0], PORT_COUNT)
             and coefficients.dtype == torch.complex128
             and tuple(raw.shape) == (amplitude.shape[0], 1, *IMAGE_SHAPE)
             and raw.dtype == torch.float64 and tuple(w_per_pixel.shape) == tuple(raw.shape)
             and w_per_pixel.dtype == torch.float64
             and torch.isfinite(coefficients.real).all().item()
             and torch.isfinite(coefficients.imag).all().item()
             and torch.isfinite(raw).all().item() and torch.isfinite(w_per_pixel).all().item(),
             "full amplitude-SLM -> Encoder -> OLS -> Decoder -> raw CMOS forward is incomplete")
    return {"coefficients": coefficients, "raw_cmos": raw, "w_per_pixel": w_per_pixel}


def _forward_target(model: base.SmallMatchedAdapter, coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
    result = model.forward_from_ols(coefficients.to(dtype=torch.complex128))
    raw = result["raw_intensity_w_per_m2"]
    w_per_pixel = result["physical_w_per_pixel"]
    _require(tuple(raw.shape) == tuple(w_per_pixel.shape)
             == (coefficients.shape[0], 1, *IMAGE_SHAPE)
             and raw.dtype == w_per_pixel.dtype == torch.float64
             and torch.isfinite(raw).all().item()
             and torch.isfinite(w_per_pixel).all().item(),
             "direct frozen Decoder raw CMOS reference is incomplete")
    return {"raw_cmos": raw, "w_per_pixel": w_per_pixel}


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _assert_frozen(module: torch.nn.Module, before: Mapping[str, torch.Tensor], name: str) -> None:
    after = module.state_dict()
    _require(set(after) == set(before)
             and all(torch.equal(after[key].detach().cpu(), before[key]) for key in before),
             f"frozen {name} changed")


def _phase_error(actual: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    t = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    _require(np.iscomplexobj(a) and np.iscomplexobj(t),
             "complex error inputs must be complex arrays", ValueError)
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    t = np.asarray(t, dtype=np.complex128).reshape(-1)
    _require(a.shape == t.shape and a.size > 0 and np.isfinite(a.real).all()
             and np.isfinite(a.imag).all() and np.isfinite(t.real).all()
             and np.isfinite(t.imag).all(), "complex error inputs are malformed", ValueError)
    denominator = float(np.linalg.norm(t))
    _require(math.isfinite(denominator) and denominator > 0.0,
             "complex error reference has zero or nonfinite norm", ValueError)
    phase = float(np.angle(np.vdot(t, a)))
    return float(np.linalg.norm(a - np.exp(1j * phase) * t) / denominator)


def _raw_error(actual: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    t = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    _require(not np.iscomplexobj(a) and not np.iscomplexobj(t),
             "native intensity error inputs must be real arrays", ValueError)
    a = np.asarray(a, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    denominator = float(np.linalg.norm(t.reshape(-1)))
    _require(a.shape == t.shape and math.isfinite(denominator) and denominator > 0.0
             and np.isfinite(a).all() and np.isfinite(t).all()
             and float(a.min()) >= 0.0 and float(t.min()) >= 0.0,
             "native intensity error inputs are malformed", ValueError)
    return float(np.linalg.norm((a - t).reshape(-1)) / denominator)


def _error_row(
    outputs: Mapping[int, Mapping[str, torch.Tensor]], targets: torch.Tensor,
    direct_raw: Mapping[int, torch.Tensor], index: int,
) -> dict[str, float]:
    e2048 = _phase_error(outputs[2048]["coefficients"][index], targets[index])
    e4096 = _phase_error(outputs[4096]["coefficients"][index], targets[index])
    c2048 = _raw_error(outputs[2048]["raw_cmos"][index], direct_raw[2048][index])
    c4096 = _raw_error(outputs[4096]["raw_cmos"][index], direct_raw[4096][index])
    cross_coeff = _phase_error(outputs[2048]["coefficients"][index],
                               outputs[4096]["coefficients"][index])
    cross_cmos = _raw_error(outputs[2048]["raw_cmos"][index], outputs[4096]["raw_cmos"][index])
    return {
        "coefficient_2048": e2048,
        "coefficient_4096": e4096,
        "native_cmos_2048": c2048,
        "native_cmos_4096": c4096,
        "cross_grid_coefficient": cross_coeff,
        "cross_grid_native_cmos": cross_cmos,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("coefficient_2048", "coefficient_4096", "native_cmos_2048", "native_cmos_4096",
            "cross_grid_coefficient", "cross_grid_native_cmos")
    values = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in keys}
    passed = [all(float(row[key]) <= GATE_LIMIT for key in keys) for row in rows]
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        selected = [row for row in rows if int(row["label"]) == label]
        per_class[str(label)] = {
            "denominator": len(selected),
            "all_six_pass_fraction": sum(bool(row["row_passed"]) for row in selected) / len(selected),
            **{key: {
                "mean": float(np.mean([float(row[key]) for row in selected])),
                "rms": float(np.sqrt(np.mean(np.square([float(row[key]) for row in selected])))),
                "median": float(np.median([float(row[key]) for row in selected])),
                "maximum": float(np.max([float(row[key]) for row in selected])),
            } for key in keys},
        }
    return {
        "denominator": len(rows),
        "threshold": GATE_LIMIT,
        "all_rows_passed": bool(rows) and all(passed),
        "passed_rows": int(sum(passed)),
        "pass_fraction": float(sum(passed) / len(passed)) if passed else 0.0,
        "maxima": {key: float(np.max(value)) for key, value in values.items()},
        "per_class": per_class,
    }


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["population_index", "label", "sample_index_within_class", "proposal_id",
              "coefficient_2048", "coefficient_4096", "native_cmos_2048", "native_cmos_4096",
              "cross_grid_coefficient", "cross_grid_native_cmos", "row_passed"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def _measure_cohort(
    mapper: AmplitudeMapper, models: Mapping[int, base.SmallMatchedAdapter],
    population: Mapping[str, Any], matrix: torch.Tensor, inverse: torch.Tensor,
    scale: torch.Tensor, device: torch.device, *, role: str, batch_size: int = 10,
    shard_dir: Path | None = None,
) -> list[dict[str, Any]]:
    coeff = population["coefficients"]
    labels = population["labels"]
    proposals = population["proposal_ids"]
    _require(tuple(coeff.shape[1:]) == (PORT_COUNT,) and len(labels) == len(coeff) == len(proposals),
             f"{role} population role or row arrays are inconsistent")
    rows: list[dict[str, Any]] = []
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    for batch_no, start in enumerate(range(0, len(coeff), batch_size)):
        stop = min(start + batch_size, len(coeff))
        targets = coeff[start:stop].to(device=device, dtype=torch.complex128)
        features = _normalized_features(coeff[start:stop], scale, device)
        with torch.no_grad():
            amplitude, projection = _map_amplitude(mapper, features, targets, matrix, inverse, scale)
            outputs = {grid: _forward_amplitude(models[grid], amplitude) for grid in GRIDS}
            references = {grid: _forward_target(models[grid], targets) for grid in GRIDS}
        for offset in range(stop - start):
            idx = start + offset
            row = {
                "population_index": idx,
                "label": int(labels[idx]),
                "sample_index_within_class": idx - int(labels[idx]) * (len(labels) // CLASS_COUNT),
                "proposal_id": int(proposals[idx]),
                **_error_row(outputs, targets, {grid: references[grid]["raw_cmos"] for grid in GRIDS}, offset),
            }
            row["row_passed"] = all(row[key] <= GATE_LIMIT for key in (
                "coefficient_2048", "coefficient_4096", "native_cmos_2048", "native_cmos_4096",
                "cross_grid_coefficient", "cross_grid_native_cmos",
            ))
            row["clipped_fraction"] = float(projection["clipped_fraction"][offset].detach().cpu())
            rows.append(row)
        if shard_dir is not None:
            _atomic_torch(shard_dir / f"{role}_batch_{batch_no:04d}.pt", {
                "schema": "lightgen_g1_slm_projected_measurement_shard.v2",
                "task_id": TASK_ID,
                "population_role": role,
                "population_indices": torch.arange(start, stop),
                "labels": labels[start:stop].clone(),
                "proposal_ids": proposals[start:stop].clone(),
                "target_coefficients": coeff[start:stop].clone().to(torch.complex128),
                "command_encoding": "continuous_amplitude_float32",
                "continuous_amplitude": amplitude.detach().cpu().clone(),
                "mapped_coefficients": {str(grid): outputs[grid]["coefficients"].detach().cpu().clone()
                                         for grid in GRIDS},
                "mapped_raw_cmos_w_per_m2": {str(grid): outputs[grid]["raw_cmos"].detach().cpu().clone()
                                              for grid in GRIDS},
                "direct_raw_cmos_w_per_m2": {str(grid): references[grid]["raw_cmos"].detach().cpu().clone()
                                              for grid in GRIDS},
            })
            attempt_dir = shard_dir.parents[1]
            _atomic_json(attempt_dir / "progress.json", {
                "task_id": TASK_ID,
                "run_id": _json(attempt_dir / "run_identity.json")["run_id"],
                "phase": "measurement",
                "population_role": role,
                "completed_rows": stop,
                "total_rows": len(coeff),
                "completed_batches": batch_no + 1,
                "updated_at_unix": time.time(),
            })
    return rows


def _write_population(preflight_dir: Path, population: Mapping[str, Any], schedule: torch.Tensor) -> None:
    payload = dict(population)
    payload["training_schedule"] = schedule
    _atomic_torch(preflight_dir / "fixed_population_and_schedule.pt", payload)


def _save_matrix(preflight_dir: Path, matrix: torch.Tensor, inverse: torch.Tensor,
                 response: torch.Tensor, scale: torch.Tensor) -> None:
    _atomic_torch(preflight_dir / "fixed_response_and_projector.pt", {
        "schema": "lightgen_g1_slm_fixed_response_projector.v2",
        "task_id": TASK_ID,
        "response_matrix": response.detach().cpu().contiguous(),
        "normalized_real_matrix": matrix.detach().cpu().contiguous(),
        "full_right_inverse": inverse.detach().cpu().contiguous(),
        "scale": scale.detach().cpu().double().contiguous(),
        "real_imag_order": "per_port_interleaved_real_imag",
    })


def _load_matrix(
    preflight_dir: Path, device: torch.device, *,
    expected_response: torch.Tensor, expected_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = preflight_dir / "fixed_response_and_projector.pt"
    _require(path.is_file(), "fixed response and right inverse are missing", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "lightgen_g1_slm_fixed_response_projector.v2"
             and payload.get("task_id") == TASK_ID
             and payload.get("real_imag_order") == "per_port_interleaved_real_imag",
             "fixed response/projector evidence identity changed")
    response = payload["response_matrix"].to(device=device, dtype=torch.complex128)
    matrix = payload["normalized_real_matrix"].to(device=device, dtype=torch.float64)
    inverse = payload["full_right_inverse"].to(device=device, dtype=torch.float64)
    scale = payload["scale"].to(device=device, dtype=torch.float64)
    _require(tuple(response.shape) == (PORT_COUNT, PIXEL_COUNT)
             and tuple(matrix.shape) == (REAL_DIM, PIXEL_COUNT)
             and tuple(inverse.shape) == (PIXEL_COUNT, REAL_DIM)
             and tuple(scale.shape) == (PORT_COUNT,),
             "saved fixed matrix or projector dimensions changed")
    expected_response = expected_response.to(device=device, dtype=torch.complex128)
    expected_scale = expected_scale.to(device=device, dtype=torch.float64)
    _require(torch.equal(response, expected_response) and torch.equal(scale, expected_scale)
             and torch.isfinite(scale).all().item() and torch.all(scale > 0.0).item(),
             "saved response or modal scale differs from the explicit frozen inputs")
    expected_matrix = _normalized_response(expected_response, expected_scale)
    matrix_error = float((matrix - expected_matrix).abs().max().item())
    identity_error = float((matrix @ inverse - torch.eye(REAL_DIM, device=device,
                                                          dtype=torch.float64)).abs().max().item())
    gram = matrix @ matrix.T
    moore_penrose = matrix.T @ torch.linalg.solve(
        gram, torch.eye(REAL_DIM, device=device, dtype=torch.float64),
    )
    inverse_error = float((inverse - moore_penrose).abs().max().item())
    _require(matrix_error <= 1.0e-14 and identity_error <= 1.0e-10 and inverse_error <= 1.0e-10,
             "saved normalized matrix or right inverse is inconsistent with the full Moore-Penrose solution")
    return response, matrix, inverse, scale


def _numpy_minimum_norm_correction(matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Apply the full-row-rank Moore-Penrose solution without materializing A+."""
    values = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(residual, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] <= values.shape[1]
             and targets.ndim == 2 and targets.shape[1] == values.shape[0]
             and np.isfinite(values).all() and np.isfinite(targets).all(),
             "independent NumPy projection inputs are malformed", ValueError)
    gram = values @ values.T
    weights = np.linalg.solve(gram, targets.T)
    correction = (values.T @ weights).T
    _require(np.isfinite(correction).all(), "independent NumPy correction is nonfinite", ValueError)
    return correction


def _matrix_checks(
    response: torch.Tensor, matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch_identity = matrix @ inverse
    max_identity_error = float((torch_identity - torch.eye(REAL_DIM, dtype=torch.float64,
                                                              device=device)).abs().max().item())
    response_np = response.detach().cpu().numpy()
    scale_np = scale.detach().cpu().numpy()
    numpy_matrix = np.stack((response_np.real, response_np.imag), axis=1).reshape(REAL_DIM, PIXEL_COUNT)
    numpy_matrix = numpy_matrix / np.repeat(np.sqrt(scale_np), 2)[:, None]
    matrix_np_error = float(np.max(np.abs(numpy_matrix - matrix.detach().cpu().numpy())))
    _require(max_identity_error <= 1.0e-10 and matrix_np_error <= 1.0e-14,
             "normalized matrix, real/imag ordering, or full right inverse failed independent verification")

    rng = np.random.default_rng(2026102772)
    raw = rng.uniform(0.2, 0.8, size=(2, PIXEL_COUNT)).astype(np.float64)
    targets = np.stack((response.detach().cpu().numpy() @ raw[0],
                        response.detach().cpu().numpy() @ raw[1]))
    test_targets = torch.from_numpy(targets).to(device=device, dtype=torch.complex128)
    raw_tensor = torch.from_numpy(raw).to(device=device, dtype=torch.float64).reshape(2, 1, 200, 200)
    with torch.no_grad():
        projected, _ = _project_from_raw(raw_tensor, test_targets, matrix, inverse, scale)
    residuals: list[np.ndarray] = []
    for idx in range(2):
        u = raw[idx]
        predicted = numpy_matrix @ u
        predicted_complex = (predicted.reshape(PORT_COUNT, 2)[:, 0]
                             + 1j * predicted.reshape(PORT_COUNT, 2)[:, 1]) * np.sqrt(scale_np)
        corr = np.vdot(targets[idx], predicted_complex)
        phi = 0.0 if corr == 0.0 else float(np.angle(corr))
        aligned = targets[idx] * np.exp(1j * phi) / np.sqrt(scale_np)
        goal = np.stack((aligned.real, aligned.imag), axis=1).reshape(REAL_DIM)
        residuals.append(goal - predicted)
    corrections = _numpy_minimum_norm_correction(numpy_matrix, np.stack(residuals))
    outputs = [np.clip(raw[idx] + corrections[idx], 0.0, 1.0) for idx in range(2)]
    numpy_output_error = float(np.max(np.abs(np.stack(outputs).reshape(2, 1, 200, 200)
                                             - projected.detach().cpu().numpy())))
    _require(numpy_output_error <= 2.0e-7,
             f"fixed projection output differs from independent NumPy reference: {numpy_output_error:.3g}")
    return {
        "matrix_shape": list(matrix.shape),
        "projector_shape": list(inverse.shape),
        "max_abs_AP_minus_I": max_identity_error,
        "independent_numpy_matrix_max_abs_error": matrix_np_error,
        "independent_numpy_projection_max_abs_error": numpy_output_error,
        "independent_numpy_solver": "A.T @ solve(A @ A.T, residual)",
        "full_rank_directions_retained": REAL_DIM,
    }


def _matrix_and_camera_checks(
    inputs: Mapping[str, Any], response: torch.Tensor, model: base.SmallMatchedAdapter,
    device: torch.device,
) -> dict[str, Any]:
    rng = torch.Generator(device="cpu").manual_seed(2026102773)
    random = (torch.rand((1, 1, 200, 200), generator=rng, dtype=torch.float64) * 0.8 + 0.1).to(device)
    existing = inputs["commands"]["amplitude_command"][:2].to(device=device, dtype=torch.float64)
    commands = torch.cat((torch.zeros_like(random), torch.ones_like(random), random, existing), dim=0)
    with torch.no_grad():
        produced = _forward_amplitude(model, commands)
        predicted = commands.reshape(len(commands), -1).to(torch.complex128) @ response.T
    errors = []
    for idx in range(len(commands)):
        actual = produced["coefficients"][idx]
        expected = predicted[idx]
        denom = max(float(torch.linalg.vector_norm(actual).item()), 1.0e-30)
        errors.append(float((torch.linalg.vector_norm(actual - expected) / denom).item()))
    maximum = max(errors)
    area = _camera_pixel_area(model)
    camera_unit_probe = torch.tensor([1.0, 2.0], dtype=torch.float64)
    converted = camera_unit_probe * area
    independent_probe = np.asarray([1.0, 2.0], dtype=np.float64) * np.float64(area)
    camera_conversion_error = float(np.max(np.abs(converted.cpu().numpy() - independent_probe)))
    camera_conversion_tolerance = float(
        2.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(independent_probe))), 1.0e-300)
    )
    raw_numpy = produced["raw_cmos"].detach().cpu().numpy()
    w_per_pixel_numpy = produced["w_per_pixel"].detach().cpu().numpy()
    independent_w_per_pixel = raw_numpy * np.float64(area)
    production_conversion_error = float(
        np.max(np.abs(w_per_pixel_numpy - independent_w_per_pixel))
        / max(float(np.max(np.abs(independent_w_per_pixel))), 1.0e-300)
    )
    _require(maximum <= MATRIX_LIMIT and camera_conversion_error <= camera_conversion_tolerance
             and production_conversion_error <= 1.0e-14,
             f"response/camera check failed: response={maximum:.3g}, "
             f"camera_probe={camera_conversion_error:.3g}/{camera_conversion_tolerance:.3g}, "
             f"production_W_per_pixel={production_conversion_error:.3g}")
    return {
        "production_response_relative_errors": errors,
        "maximum_production_response_relative_error": maximum,
        "response_bound": MATRIX_LIMIT,
        "camera_pixel_area_m2": area,
        "camera_unit_probe_max_abs_error": camera_conversion_error,
        "camera_unit_probe_tolerance_abs": camera_conversion_tolerance,
        "production_w_per_pixel_numpy_relative_error": production_conversion_error,
        "fixed_camera_gain_w_inverse": inputs["camera_gain"],
        "commands": ["zero", "all_one", "spatially_nonuniform", "two_saved_reachability_commands"],
    }


def _projection_gradient_checks(
    matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
    target: torch.Tensor, device: torch.device,
) -> dict[str, float]:
    base_command = torch.full((1, 1, 200, 200), 0.5, dtype=torch.float64, device=device)
    with torch.no_grad():
        target_for_command = torch.view_as_complex(
            (base_command.reshape(1, -1) @ matrix.T).reshape(1, 100, 2).contiguous()
        ) * scale.sqrt().unsqueeze(0)
    raw = base_command.detach().clone().requires_grad_(True)
    mapped, _ = _project_double_from_raw(raw, target_for_command, matrix, inverse, scale)
    direction = torch.linspace(-1.0, 1.0, PIXEL_COUNT, dtype=torch.float64, device=device).reshape_as(raw)
    direction = direction / torch.linalg.vector_norm(direction)
    probe = torch.linspace(0.25, 1.25, PIXEL_COUNT, dtype=torch.float64, device=device).reshape_as(raw)
    objective = (mapped.double() * probe).sum()
    grad, = torch.autograd.grad(objective, raw)
    analytic = float((grad * direction).sum().item())
    eps = 1.0e-3
    with torch.no_grad():
        plus = _project_double_from_raw(raw + eps * direction, target_for_command, matrix, inverse, scale)[0]
        minus = _project_double_from_raw(raw - eps * direction, target_for_command, matrix, inverse, scale)[0]
        numeric = float((((plus.double() * probe).sum() - (minus.double() * probe).sum())
                         / (2.0 * eps)).item())
    derivative_error = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1.0e-12)
    _require(math.isfinite(analytic) and math.isfinite(numeric) and derivative_error <= 1.0e-5,
             f"fixed correction directional derivative failed: {derivative_error:.3g}")
    with torch.no_grad():
        negative, _ = _project_double_from_raw(raw - 1.0e-3 * grad / grad.norm(),
                                               target_for_command, matrix, inverse, scale)
        lowered = float((negative * probe).sum().item())
    current = float(objective.detach().item())
    _require(lowered < current, "projected-correction negative direction did not lower the independent scalar objective")
    return {"analytic_directional_derivative": analytic,
            "complex128_central_difference": numeric,
            "relative_derivative_error": derivative_error,
            "negative_direction_objective_delta": lowered - current}


def _optical_gradient_checks(model: base.SmallMatchedAdapter, target: torch.Tensor,
                             device: torch.device) -> dict[str, float]:
    command = torch.full((1, 1, 200, 200), 0.5, dtype=torch.float64, device=device, requires_grad=True)
    result = _forward_amplitude(model, command)
    loss = pilot._phase_invariant_loss(result["coefficients"], target.reshape(1, -1)).sum()
    gradient, = torch.autograd.grad(loss, command)
    direction = gradient.detach() / torch.linalg.vector_norm(gradient.detach()).clamp_min(1.0e-30)
    analytic = float((gradient.detach() * direction).sum().item())
    eps = 2.0e-4
    with torch.no_grad():
        plus = _forward_amplitude(model, command.detach() + eps * direction)["coefficients"]
        minus = _forward_amplitude(model, command.detach() - eps * direction)["coefficients"]
        plus_loss = float(pilot._phase_invariant_loss(plus, target.reshape(1, -1)).sum().item())
        minus_loss = float(pilot._phase_invariant_loss(minus, target.reshape(1, -1)).sum().item())
    numeric = (plus_loss - minus_loss) / (2.0 * eps)
    error = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1.0e-12)
    _require(math.isfinite(analytic) and analytic > 0.0 and math.isfinite(numeric)
             and error <= 1.0e-5 and minus_loss < float(loss.detach().item()),
             f"full production optical derivative/value/direction check failed: {error:.3g}")
    return {"autograd_directional_derivative": analytic,
            "complex128_central_difference": numeric,
            "relative_derivative_error": error,
            "descent_probe_minus_loss": minus_loss,
            "center_loss": float(loss.detach().item())}


def _load_initial_mapper(inputs: Mapping[str, Any], device: torch.device) -> AmplitudeMapper:
    return _new_mapper(device, inputs["source_mapper"]["mapper_state"])


def _optimizer(mapper: AmplitudeMapper, config: Mapping[str, Any]) -> torch.optim.Adam:
    training = config["training"]
    return torch.optim.Adam(mapper.parameters(), lr=float(training["learning_rate"]),
                            betas=tuple(float(x) for x in training["betas"]),
                            eps=float(training["epsilon"]), weight_decay=float(training["weight_decay"]))


def _batch(population: Mapping[str, Any], schedule: torch.Tensor, step: int) -> torch.Tensor:
    epoch, within = divmod(step, 1000)
    _require(0 <= epoch < 5, "training schedule cursor is outside the fixed five passes")
    indices = schedule[epoch, within]
    values = population["train"]["coefficients"][indices]
    labels = population["train"]["labels"][indices]
    _require(tuple(values.shape) == (10, 100) and torch.equal(labels, torch.arange(10)),
             "scheduled batch is not exactly one fixed training coefficient per class")
    return values


def _training_step(
    mapper: AmplitudeMapper, optimizer: torch.optim.Adam, model: base.SmallMatchedAdapter,
    coefficients: torch.Tensor, matrix: torch.Tensor, inverse: torch.Tensor,
    scale: torch.Tensor, device: torch.device, *, audit: bool = False,
) -> dict[str, Any]:
    target = coefficients.to(device=device, dtype=torch.complex128)
    features = _normalized_features(coefficients, scale, device)
    before = [parameter.detach().clone() for parameter in mapper.parameters()] if audit else None
    optimizer.zero_grad(set_to_none=True)
    amplitude, projection = _map_amplitude(mapper, features, target, matrix, inverse, scale)
    output = _forward_amplitude(model, amplitude)
    losses = pilot._phase_invariant_loss(output["coefficients"], target)
    loss = losses.mean()
    _require(torch.isfinite(loss).item(), "full-optics projected training loss is nonfinite")
    loss.backward()
    grad_norms: dict[str, float] = {}
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"projected mapper parameter gradient is absent or nonfinite: {name}")
        grad_norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach().double()).item())
        _require(math.isfinite(grad_norms[name]), f"projected mapper gradient norm is nonfinite: {name}")
    optimizer.step()
    if before is not None:
        _require(all(not torch.equal(old, new.detach()) for old, new in
                     zip(before, mapper.parameters(), strict=True)),
                 "a declared projected mapper parameter tensor did not update")
    return {
        "loss": float(loss.detach().cpu()),
        "coefficient_relative_error_rms": float(torch.sqrt(losses.detach().mean()).cpu()),
        "coefficient_relative_error_mean": float(torch.sqrt(losses.detach()).mean().cpu()),
        "gradient_norms": grad_norms,
        "clipped_fraction_mean": float(projection["clipped_fraction"].mean().detach().cpu()),
    }


def _snapshot_rng() -> dict[str, Any]:
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]}


def _checkpoint_payload(mapper: AmplitudeMapper, optimizer: torch.optim.Adam, *, run_id: str,
                        completed: int, history: Sequence[Mapping[str, Any]],
                        schedule: torch.Tensor) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "run_id": run_id,
        "source_mapper_checkpoint": "original_G1_mapper_step_5000",
        "source_mapper_updates": 5000,
        "completed_updates": completed,
        "checkpoint_step": completed,
        "next_schedule_cursor": completed,
        "next_batch_indices": (schedule[completed // 1000, completed % 1000].clone()
                                if completed < 5000 else None),
        "mapper_state": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()],
        "history_count": len(history),
        "last_loss": None if not history else float(history[-1]["loss"]),
        "optimizer_deepcopied": True,
        "schedule_identity": "fixed_population_and_schedule.pt:5x1000x10",
        "schedule_prefix_length": completed,
    }


def _save_checkpoint(path: Path, mapper: AmplitudeMapper, optimizer: torch.optim.Adam, *,
                     run_id: str, completed: int, history: Sequence[Mapping[str, Any]],
                     schedule: torch.Tensor) -> dict[str, Any]:
    rng_before = _snapshot_rng()
    payload = _checkpoint_payload(mapper, optimizer, run_id=run_id, completed=completed,
                                  history=history, schedule=schedule)
    _atomic_torch(path, payload)
    rng_after = _snapshot_rng()
    _require(torch.equal(rng_before["cpu"], rng_after["cpu"])
             and all(torch.equal(a, b) for a, b in zip(rng_before["cuda"], rng_after["cuda"], strict=True)),
             "checkpoint serialization changed process RNG")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id
             and saved.get("completed_updates") == completed
             and saved.get("optimizer_deepcopied") is True
             and saved.get("history_count") == len(history),
             "checkpoint failed identity, optimizer-copy, history, or step readback")
    return saved


def _restore_checkpoint(path: Path, mapper: AmplitudeMapper, optimizer: torch.optim.Adam, *,
                        run_id: str, schedule: torch.Tensor) -> tuple[int, list[dict[str, Any]]]:
    _require(path.is_file(), f"explicit projected-mapper checkpoint is missing: {path}", FileNotFoundError)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == CHECKPOINT_SCHEMA and saved.get("task_id") == TASK_ID
             and saved.get("run_id") == run_id and saved.get("optimizer_deepcopied") is True,
             "explicit recovery checkpoint identity or Adam copy state is invalid")
    completed = int(saved.get("completed_updates", -1))
    _require(0 <= completed <= 5000 and saved.get("checkpoint_step") == completed
             and saved.get("next_schedule_cursor") == completed
             and saved.get("schedule_prefix_length") == completed
             and saved.get("schedule_identity") == "fixed_population_and_schedule.pt:5x1000x10",
             "recovery checkpoint step or sampler cursor is invalid")
    expected = schedule[completed // 1000, completed % 1000] if completed < 5000 else None
    recorded = saved.get("next_batch_indices")
    _require((expected is None and recorded is None) or
             (expected is not None and torch.is_tensor(recorded) and torch.equal(expected, recorded)),
             "recovery checkpoint sample schedule does not match fixed population")
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    torch.random.set_rng_state(saved["torch_cpu_rng"].cpu())
    torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    history_path = path.parent.parent / "training_history.json"
    history = _json(history_path).get("updates", [])
    _require(len(history) >= completed and all(
        item.get("completed_update") == i + 1
        and item.get("sample_indices") == schedule[i // 1000, i % 1000].tolist()
        for i, item in enumerate(history[:completed])
    ),
             "recovery history is not a complete prefix of the checkpoint")
    state_steps = [int(torch.as_tensor(item["step"]).item()) for item in optimizer.state.values()
                   if isinstance(item, Mapping) and "step" in item]
    if completed:
        _require(len(state_steps) == len(list(mapper.parameters())) and all(step == completed for step in state_steps),
                 "restored Adam state does not match the explicit checkpoint step")
    return completed, [dict(item) for item in history[:completed]]


def _tree_max_delta(left: Any, right: Any) -> tuple[bool, float]:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return False, math.inf
        if left.is_floating_point() or left.is_complex():
            delta = float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())
            return delta <= RECOVERY_TOLERANCE, delta
        return torch.equal(left.cpu(), right.cpu()), 0.0 if torch.equal(left.cpu(), right.cpu()) else math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, math.inf
        checks = [_tree_max_delta(left[key], right[key]) for key in left]
        return all(item[0] for item in checks), max((item[1] for item in checks), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, math.inf
        checks = [_tree_max_delta(a, b) for a, b in zip(left, right, strict=True)]
        return all(item[0] for item in checks), max((item[1] for item in checks), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = abs(float(left) - float(right))
        return delta <= RECOVERY_TOLERANCE, delta
    return left == right, 0.0 if left == right else math.inf


def _training_update(
    mapper: AmplitudeMapper, optimizer: torch.optim.Adam, model: base.SmallMatchedAdapter,
    population: Mapping[str, Any], schedule: torch.Tensor, matrix: torch.Tensor,
    inverse: torch.Tensor, scale: torch.Tensor, device: torch.device, index: int,
    *, audit: bool = False,
) -> dict[str, Any]:
    result = _training_step(mapper, optimizer, model, _batch(population, schedule, index),
                            matrix, inverse, scale, device, audit=audit)
    result["sample_indices"] = schedule[index // 1000, index % 1000].tolist()
    return result


def _engineering_worker(config_path: Path, preflight_dir: Path,
                         checkpoint_path: Path, output_path: Path) -> None:
    config = _load_config(config_path)
    _set_numerics()
    _require(torch.cuda.is_available(), "recovery worker requires CUDA")
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    population, schedule = _load_population(preflight_dir / "fixed_population_and_schedule.pt")
    response, matrix, inverse, scale = _load_matrix(
        preflight_dir, device, expected_response=inputs["response_matrix"], expected_scale=inputs["scale"],
    )
    run_id = _json(preflight_dir / "preflight_identity.json")["run_id"]
    mapper = _load_initial_mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    completed, history = _restore_checkpoint(checkpoint_path, mapper, optimizer,
                                             run_id=run_id, schedule=schedule)
    _require(completed == 3 and len(history) == 3,
             "independent recovery worker requires the exact explicit three-update checkpoint")
    indices = schedule[0, 3].clone()
    record = _training_update(mapper, optimizer, model, population, schedule,
                              matrix, inverse, scale, device, 3)
    record["completed_update"] = 4
    history.append(record)
    payload = _checkpoint_payload(mapper, optimizer, run_id=run_id, completed=4,
                                  history=history, schedule=schedule)
    payload.update({"step4_loss": record["loss"], "step4_schedule_indices": indices,
                    "history": history, "engineering_adam_calls": 1})
    _atomic_torch(output_path, payload)


def _engineering_recovery(
    config_path: Path, config: Mapping[str, Any], inputs: Mapping[str, Any],
    population: Mapping[str, Any], schedule: torch.Tensor,
    response: torch.Tensor, matrix: torch.Tensor, inverse: torch.Tensor,
    scale: torch.Tensor, preflight_dir: Path, device: torch.device,
) -> dict[str, Any]:
    work = preflight_dir / "engineering_recovery"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = work / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    run_id = _json(preflight_dir / "preflight_identity.json")["run_id"]
    mapper = _load_initial_mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    optical_before = _clone_state(model)
    mapper_before = [parameter.detach().clone() for parameter in mapper.parameters()]
    history: list[dict[str, Any]] = []
    for index in range(3):
        item = _training_update(mapper, optimizer, model, population, schedule,
                                matrix, inverse, scale, device, index, audit=True)
        item["completed_update"] = index + 1
        history.append(item)
    _require(all(not torch.equal(before, after.detach()) for before, after in
                 zip(mapper_before, mapper.parameters(), strict=True)),
             "the real three-step projected update did not change every mapper parameter tensor")
    _atomic_json(work / "training_history.json", {"task_id": TASK_ID, "run_id": run_id, "updates": history})
    rng_before_save = _snapshot_rng()
    state_before_step4 = {
        "mapper": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }
    cp3 = checkpoint_dir / "checkpoint_step_000003.pt"
    saved3 = _save_checkpoint(cp3, mapper, optimizer, run_id=run_id, completed=3,
                              history=history, schedule=schedule)
    same_mapper, mapper_delta = _tree_max_delta(state_before_step4["mapper"], saved3["mapper_state"])
    same_adam, adam_delta = _tree_max_delta(state_before_step4["optimizer"], saved3["optimizer_state"])
    _require(same_mapper and same_adam and torch.equal(rng_before_save["cpu"], torch.random.get_rng_state())
             and all(torch.equal(a, b) for a, b in zip(rng_before_save["cuda"],
                                                        torch.cuda.get_rng_state_all(), strict=True)),
             "checkpoint serialization changed mapper, deep-copied Adam state, or RNG")

    step4_indices = schedule[0, 3].clone()
    live = _training_update(mapper, optimizer, model, population, schedule,
                            matrix, inverse, scale, device, 3)
    live["completed_update"] = 4
    history.append(live)
    _atomic_json(work / "training_history.json", {"task_id": TASK_ID, "run_id": run_id, "updates": history})
    live_state = {
        "mapper": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }
    live_rng = _snapshot_rng()
    live4_path = checkpoint_dir / "checkpoint_step_000004_live.pt"
    _save_checkpoint(live4_path, mapper, optimizer, run_id=run_id, completed=4,
                     history=history, schedule=schedule)
    _require(torch.equal(live_rng["cpu"], torch.random.get_rng_state())
             and all(torch.equal(a, b) for a, b in zip(live_rng["cuda"],
                                                        torch.cuda.get_rng_state_all(), strict=True)),
             "live fourth-step checkpoint changed process RNG")
    worker_out = work / "worker_step4_restored.pt"
    command = [sys.executable, "-B", "-m", MODULE_PATH, "--mode", "recovery-worker",
               "--config", str(config_path.resolve()),
               "--checkpoint", str(cp3.resolve()), "--worker-output", str(worker_out.resolve())]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (work / "recovery_worker.stdout.log").write_text(result.stdout, encoding="utf-8")
    (work / "recovery_worker.stderr.log").write_text(result.stderr, encoding="utf-8")
    _require(result.returncode == 0 and worker_out.is_file(),
             "independent-process checkpoint restore step failed; preserve and inspect worker logs")
    worker = torch.load(worker_out, map_location="cpu", weights_only=False)
    _require(worker.get("completed_updates") == 4 and worker.get("engineering_adam_calls") == 1
             and torch.equal(worker["step4_schedule_indices"], step4_indices),
             "independent restored process used the wrong step or sample batch")
    mapper_ok, mapper_diff = _tree_max_delta(live_state["mapper"], worker["mapper_state"])
    adam_ok, adam_diff = _tree_max_delta(live_state["optimizer"], worker["optimizer_state"])
    hist_ok, hist_diff = _tree_max_delta(history, worker["history"])
    rng_ok = torch.equal(live_rng["cpu"], worker["torch_cpu_rng"]) and all(
        torch.equal(a, b) for a, b in zip(live_rng["cuda"], worker["torch_cuda_rng"], strict=True)
    )
    loss_delta = abs(float(live["loss"]) - float(worker["step4_loss"]))
    _assert_frozen(model, optical_before, "C optical model during recovery preflight")
    _require(mapper_ok and adam_ok and hist_ok and rng_ok and loss_delta <= RECOVERY_TOLERANCE,
             "live and independently restored fourth update differ beyond fixed recovery tolerance")
    record = {
        "status": "PASSED",
        "executed_after_zero_update_validation_failed": True,
        "first_process_updates": 4,
        "restored_process_updates": 1,
        "engineering_adam_calls": 5,
        "formal_scientific_updates": 0,
        "source_mapper_updates_before_this_task": 5000,
        "frozen_optical_updates": 0,
        "model_max_absolute_difference": mapper_diff,
        "adam_max_absolute_difference": adam_diff,
        "history_max_absolute_difference": hist_diff,
        "step4_loss_absolute_difference": loss_delta,
        "rng_exactly_equal": rng_ok,
        "saved_schedule_exactly_equal": True,
        "checkpoint_save_preserved_rng": True,
        "adam_state_deepcopied": True,
        "restore_function": "_restore_checkpoint",
        "absolute_tolerance": RECOVERY_TOLERANCE,
        "three_update_losses": [float(item["loss"]) for item in history[:3]],
        "live_step4_loss": float(live["loss"]),
        "restored_step4_loss": float(worker["step4_loss"]),
        "step3_checkpoint_model_max_abs_delta": mapper_delta,
        "step3_checkpoint_adam_max_abs_delta": adam_delta,
    }
    _atomic_json(work / "engineering_recovery.json", record)
    return record


def _consume_production_row(
    population: Mapping[str, Any], population_index: int, saved_target: torch.Tensor,
    amplitude: torch.Tensor, outputs: Mapping[int, Mapping[str, torch.Tensor]],
    direct_raw: Mapping[int, torch.Tensor], response: torch.Tensor, command_encoding: str,
    *, reported_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    coefficients = population["coefficients"]
    labels = population["labels"]
    proposals = population["proposal_ids"]
    _require(command_encoding == "continuous_amplitude_float32",
             "production row consumer rejected command encoding", ValueError)
    _require(0 <= population_index < len(coefficients)
             and torch.equal(saved_target.reshape(-1).to(torch.complex128),
                             coefficients[population_index].reshape(-1).to(torch.complex128)),
             "production row consumer rejected target-population identity", ValueError)
    _require(tuple(amplitude.shape) == (1, 1, 200, 200) and amplitude.dtype == torch.float32
             and torch.isfinite(amplitude).all().item()
             and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0,
             "production row consumer rejected the amplitude command", ValueError)
    for grid in GRIDS:
        output = outputs[grid]
        reference = direct_raw[grid]
        _require(tuple(output["coefficients"].shape) == (1, PORT_COUNT)
                 and output["coefficients"].dtype == torch.complex128
                 and tuple(output["raw_cmos"].shape) == (1, 1, 200, 200)
                 and output["raw_cmos"].dtype == torch.float64
                 and tuple(reference.shape) == (1, 1, 200, 200)
                 and reference.dtype == torch.float64
                 and torch.isfinite(output["coefficients"].real).all().item()
                 and torch.isfinite(output["coefficients"].imag).all().item()
                 and torch.isfinite(output["raw_cmos"]).all().item()
                 and torch.isfinite(reference).all().item(),
                 f"production row consumer rejected malformed grid-{grid} outputs", ValueError)
    predicted = amplitude.reshape(1, -1).double().to(torch.complex128) @ response.to(torch.complex128).T
    response_error = _phase_error(outputs[2048]["coefficients"][0], predicted[0])
    _require(response_error <= MATRIX_LIMIT,
             "production row consumer rejected amplitude-to-coefficient response", ValueError)
    target = coefficients[population_index:population_index + 1].to(torch.complex128)
    metrics = _error_row(
        outputs, target, {grid: direct_raw[grid] for grid in GRIDS}, 0,
    )
    if reported_metrics is not None:
        _require(set(reported_metrics) == set(metrics)
                 and all(math.isfinite(float(reported_metrics[key]))
                         and abs(float(reported_metrics[key]) - metrics[key]) <= 1.0e-14
                         for key in metrics),
                 "production row consumer rejected stale or forged reported metrics", ValueError)
    row = {
        "population_index": population_index,
        "label": int(labels[population_index]),
        "sample_index_within_class": population_index % (len(coefficients) // CLASS_COUNT),
        "proposal_id": int(proposals[population_index]),
        "response_matrix_relative_error_2048": response_error,
        **metrics,
    }
    row["row_passed"] = all(metrics[key] <= GATE_LIMIT for key in metrics)
    return row


def _production_negative_controls(
    mapper: AmplitudeMapper, models: Mapping[int, base.SmallMatchedAdapter],
    population: Mapping[str, Any], response: torch.Tensor, matrix: torch.Tensor,
    inverse: torch.Tensor, scale: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    validation = population["validation"]
    target = validation["coefficients"][0:1].to(device=device, dtype=torch.complex128)
    features = _normalized_features(validation["coefficients"][0:1], scale, device)
    with torch.no_grad():
        amplitude, _ = _map_amplitude(mapper, features, target, matrix, inverse, scale)
        correct_outputs = {grid: _forward_amplitude(models[grid], amplitude) for grid in GRIDS}
        direct_raw = {grid: _forward_target(models[grid], target)["raw_cmos"] for grid in GRIDS}
        wrong_outputs: dict[int, dict[str, torch.Tensor]] = {}
        for grid in GRIDS:
            wrong = models[grid].runtime.forward_from_image(amplitude.double())
            wrong_outputs[grid] = {"coefficients": wrong.z_ols.coefficients,
                                   "raw_cmos": wrong.raw_camera.tensor}
    correct_row = _consume_production_row(
        validation, 0, target.detach().cpu(), amplitude.detach().cpu(),
        {grid: {key: value.detach().cpu() for key, value in output.items()}
         for grid, output in correct_outputs.items()},
        {grid: value.detach().cpu() for grid, value in direct_raw.items()},
        response.detach().cpu(), "continuous_amplitude_float32",
    )
    wrong_target = validation["coefficients"][1].to(device=device, dtype=torch.complex128)
    wrong_target_error = _phase_error(correct_outputs[2048]["coefficients"][0], wrong_target)
    wrong_target_rejected = False
    wrong_encoding_rejected = False
    wrong_command_rejected = False
    try:
        _consume_production_row(
            validation, 0, wrong_target.detach().cpu(), amplitude.detach().cpu(),
            {grid: {key: value.detach().cpu() for key, value in output.items()}
             for grid, output in correct_outputs.items()},
            {grid: value.detach().cpu() for grid, value in direct_raw.items()},
            response.detach().cpu(), "continuous_amplitude_float32",
            reported_metrics={key: float(correct_row[key]) for key in (
                "coefficient_2048", "coefficient_4096", "native_cmos_2048",
                "native_cmos_4096", "cross_grid_coefficient", "cross_grid_native_cmos",
            )},
        )
    except ValueError:
        wrong_target_rejected = True
    try:
        _consume_production_row(
            validation, 0, target.detach().cpu(), amplitude.detach().cpu(),
            {grid: {key: value.detach().cpu() for key, value in output.items()}
             for grid, output in correct_outputs.items()},
            {grid: value.detach().cpu() for grid, value in direct_raw.items()},
            response.detach().cpu(), "intensity_float32",
        )
    except ValueError:
        wrong_encoding_rejected = True
    try:
        _consume_production_row(
            validation, 0, target.detach().cpu(), amplitude.detach().cpu(),
            {grid: {key: value.detach().cpu() for key, value in output.items()}
             for grid, output in wrong_outputs.items()},
            {grid: value.detach().cpu() for grid, value in direct_raw.items()},
            response.detach().cpu(), "continuous_amplitude_float32",
        )
    except ValueError:
        wrong_command_rejected = True
    response_prediction = amplitude.reshape(1, -1).double().to(torch.complex128) @ response.T
    wrong_command_response_error = _phase_error(
        wrong_outputs[2048]["coefficients"][0], response_prediction[0],
    )
    _require(math.isfinite(correct_row["coefficient_2048"])
             and wrong_target_rejected and wrong_encoding_rejected and wrong_command_rejected
             and wrong_command_response_error > MATRIX_LIMIT,
             "production negative control failed to reject wrong target or amplitude/intensity encoding")
    return {
        "valid_production_row_consumed_and_recomputed": True,
        "wrong_target_with_stale_metrics_rejected": wrong_target_rejected,
        "wrong_encoding_metadata_rejected": wrong_encoding_rejected,
        "amplitude_misread_as_intensity_output_rejected": wrong_command_rejected,
        "correct_command_response_error": correct_row["response_matrix_relative_error_2048"],
        "wrong_encoded_response_error": wrong_command_response_error,
        "correct_target_coefficient_error": correct_row["coefficient_2048"],
        "wrong_target_coefficient_error": wrong_target_error,
    }


def _oracle_witnesses(
    config: Mapping[str, Any], preflight_dir: Path, population: Mapping[str, Any],
    response: torch.Tensor, models: Mapping[int, base.SmallMatchedAdapter],
    device: torch.device,
) -> dict[str, Any]:
    validation = population["validation"]
    records: list[dict[str, Any]] = []
    commands: list[torch.Tensor] = []
    all_passed = True
    for label in range(CLASS_COUNT):
        for slot in range(int(config["reachability"]["probe_per_class"])):
            index = label * 200 + slot
            target = validation["coefficients"][index].to(device=device, dtype=torch.complex128)
            solved = reach._bounded_phase_solve(
                response, target,
                starts=int(config["reachability"]["phase_starts"]),
                max_steps=int(config["reachability"]["max_steps_per_start"]),
                stop_error=float(config["reachability"]["coefficient_stop_relative_error"]),
            )
            amplitude = solved["amplitude"].reshape(1, 1, 200, 200).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                grid_outputs = {grid: _forward_amplitude(models[grid], amplitude) for grid in GRIDS}
            actual_errors = {str(grid): _phase_error(grid_outputs[grid]["coefficients"][0], target)
                             for grid in GRIDS}
            witness = (float(solved["relative_error"]) <= 1.0e-6
                       and all(value <= GATE_LIMIT for value in actual_errors.values()))
            all_passed = all_passed and witness
            records.append({
                "label": label,
                "sample_index_within_class": slot,
                "validation_index": index,
                "proposal_id": int(validation["proposal_ids"][index]),
                "solver_relative_coefficient_error": float(solved["relative_error"]),
                "solver_iterations": int(solved["iterations"]),
                "actual_full_optics_coefficient_error_by_grid": actual_errors,
                "witness_passed": witness,
            })
            commands.append(amplitude.detach().cpu())
    _atomic_torch(preflight_dir / "validation_reachability_witnesses.pt", {
        "schema": "lightgen_g1_slm_projected_reachability_witnesses.v2",
        "task_id": TASK_ID,
        "records": records,
        "amplitude_commands": torch.cat(commands),
    })
    summary = {
        "status": "ALL_20_FULL_OPTICS_WITNESSES_ESTABLISHED" if all_passed else
                  "REACHABILITY_EVIDENCE_INSUFFICIENT",
        "required": 20,
        "passed": sum(bool(row["witness_passed"]) for row in records),
        "all_passed": all_passed,
        "solver": "eight_starts; at most 5,000 projected-gradient steps per target; stop error 1e-6",
        "commands_are_training_labels": False,
        "commands_are_inference_fallback": False,
        "records": records,
    }
    _atomic_json(preflight_dir / "validation_reachability_witnesses.json", summary)
    return summary


def _save_preflight_failure(preflight_dir: Path, error: Exception) -> None:
    existing = sorted(preflight_dir.glob("preflight_failure_*.json"))
    _atomic_json(preflight_dir / f"preflight_failure_{len(existing) + 1:02d}.json", {
        "schema": "lightgen_g1_slm_projected_preflight_failure.v2",
        "task_id": TASK_ID,
        "status": "ENGINEERING_OR_INPUT_FAILURE",
        "error_type": type(error).__name__,
        "error": str(error),
        "scientific_updates": 0,
        "recorded_at_unix": time.time(),
    })


def _preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if PREFLIGHT_DIR.exists() and (PREFLIGHT_DIR / "preflight_receipt.json").is_file():
        raise FileExistsError(f"a READY or finalized preflight already exists; inspect it: {PREFLIGHT_DIR}")
    if PREFLIGHT_DIR.exists() and (PREFLIGHT_DIR / "preflight_summary.json").is_file():
        existing = _json(PREFLIGHT_DIR / "preflight_summary.json")
        if existing.get("status") in {"READY", "REACHABILITY_EVIDENCE_INSUFFICIENT"}:
            raise FileExistsError(f"preflight already reached a terminal branch: {PREFLIGHT_DIR}")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    identity_path = PREFLIGHT_DIR / "preflight_identity.json"
    if identity_path.exists():
        old_identity = _json(identity_path)
        _require(old_identity.get("task_id") == TASK_ID, "existing preflight path belongs to another task")
        run_id = str(old_identity["run_id"])
        run_identity = ResearchRunIdentity.from_mapping(old_identity["run_identity"])
    else:
        run_id = uuid.uuid4().hex
        run_identity = _run_identity(config, run_id)
        _atomic_json(identity_path, {"task_id": TASK_ID, "run_id": run_id,
                                     "run_identity": run_identity.to_dict()})
    _atomic_json(PREFLIGHT_DIR / "run_config_snapshot.yaml", config)
    correctness_checks: dict[str, Any] = {}
    resource_checks: dict[str, Any] = {}
    implementation_checks: dict[str, Any] = {}
    try:
        _set_numerics()
        _require(torch.cuda.is_available(), "projected-mapping preflight requires AutoDL CUDA")
        device = torch.device("cuda:0")
        torch.manual_seed(int(config["training"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
        inputs = _load_inputs(config)
        population, schedule = _build_population(config, inputs)
        _write_population(PREFLIGHT_DIR, population, schedule)
        response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
        scale = inputs["scale"].to(device=device, dtype=torch.float64)
        matrix = _normalized_response(response, scale)
        inverse, spectrum = _pseudoinverse(matrix)
        _save_matrix(PREFLIGHT_DIR, matrix, inverse, response, scale)
        matrix_checks = _matrix_checks(response, matrix, inverse, scale, device)
        models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
        model_states = {grid: _clone_state(model) for grid, model in models.items()}
        command_checks = _matrix_and_camera_checks(inputs, response, models[2048], device)
        mapper = _load_initial_mapper(inputs, device).eval()
        projection_gradient = _projection_gradient_checks(
            matrix, inverse, scale, population["validation"]["coefficients"][0].to(device), device,
        )
        optical_gradient = _optical_gradient_checks(
            models[2048], population["validation"]["coefficients"][0].to(device=device,
                                                                            dtype=torch.complex128), device,
        )
        negative_controls = _production_negative_controls(
            mapper, models, population, response, matrix, inverse, scale, device,
        )
        correctness_checks = {
            "status": "PASS", "passed": True,
            "source_inputs_and_roles": "explicit C45k, G1 prior, old fixed train/validation and terminal mapper validated",
            "population": {"train": 10000, "validation": 2000, "new_holdout": 2000,
                           "new_holdout_seed_base": 2026102750, "proposal_accounting": population["proposal_accounting"]},
            "normalized_response_and_right_inverse": matrix_checks,
            "singular_spectrum": spectrum,
            "production_amplitude_response_and_camera": command_checks,
            "projection_derivative_and_direction": projection_gradient,
            "full_optics_derivative_and_direction": optical_gradient,
            "production_negative_controls": negative_controls,
        }
        # The zero-update measurement is the sole route-selection observation; the new holdout is not evaluated here.
        zero_rows = _measure_cohort(mapper, models, population["validation"], matrix, inverse,
                                    scale, device, role="validation_zero_update")
        zero_summary = _summarize_rows(zero_rows)
        _write_rows_csv(PREFLIGHT_DIR / "validation_zero_update_rows.csv", zero_rows)
        _atomic_json(PREFLIGHT_DIR / "validation_zero_update_summary.json", zero_summary)
        del mapper
        torch.cuda.empty_cache()
        if zero_summary["all_rows_passed"]:
            branch = "ZERO_UPDATE_CANDIDATE_QUALIFIED_SKIP_TRAINING"
            witness = {"status": "NOT_RUN_ZERO_UPDATE_VALIDATION_PASSED", "passed": None}
            recovery = {"status": "NOT_RUN_TRAINING_PATH_NOT_SELECTED", "engineering_adam_calls": 0}
        else:
            witness = _oracle_witnesses(config, PREFLIGHT_DIR, population, response, models, device)
            if not witness["all_passed"]:
                _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", {
                    "task_id": TASK_ID, "run_id": run_id,
                    "status": "REACHABILITY_EVIDENCE_INSUFFICIENT",
                    "branch": "STOP_NO_ADDITIONAL_MAPPING_TRAINING",
                    "zero_update_validation": zero_summary,
                    "reachability": witness,
                    "scientific_updates": 0,
                })
                for grid in GRIDS:
                    _assert_frozen(models[grid], model_states[grid], f"C optics grid {grid}")
                return {"status": "REACHABILITY_EVIDENCE_INSUFFICIENT", "run_id": run_id}
            branch = "ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL"
            recovery = _engineering_recovery(
                config_path, config, inputs, population, schedule, response, matrix,
                inverse, scale, PREFLIGHT_DIR, device,
            )
            _require(recovery.get("status") == "PASSED", "actual independent checkpoint restoration did not pass")
        for grid in GRIDS:
            _assert_frozen(models[grid], model_states[grid], f"C optics grid {grid} during preflight")
        resource_envelope = {
            "device": torch.cuda.get_device_name(device),
            "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "cuda_free_memory_observation_bytes": int(torch.cuda.mem_get_info(device)[0]),
            "disk_free_bytes_at_preflight": int(__import__("shutil").disk_usage(PREFLIGHT_DIR).free),
            "production_forward_completed": True,
            "fixed_memory_quota_used": False,
        }
        resource_checks = {"status": "PASS", "passed": True,
                           "cuda": torch.cuda.is_available(),
                           "representative_production_optics_and_validation_completed": True}
        implementation_checks = {
            "status": "PASS", "passed": True,
            "selected_engine": "full_C_encoder_OLS_decoder_production_path",
            "engine": "full_C_encoder_OLS_decoder_production_path",
            "training_updates": 5000 if branch == "ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL" else 0,
            "optical_updates": 0,
            "evaluator_updates": 0,
            "branch": branch,
            "zero_update_validation_passed": zero_summary["all_rows_passed"],
            "reachability_witness": witness,
            "recovery": recovery,
        }
        control = ResearchRunControl(identity=run_identity, authorization_id=AUTHORITY)
        receipt = control.make_ready_receipt(
            correctness=correctness_checks,
            resource=resource_checks,
            implementation=implementation_checks,
            selected_engine="full_C_encoder_OLS_decoder_production_path",
            resource_envelope=resource_envelope,
        )
        control.mark_preflight_ready(receipt)
        _atomic_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
        _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", {
            "task_id": TASK_ID, "run_id": run_id, "status": "READY", "branch": branch,
            "zero_update_validation": zero_summary,
            "reachability": witness,
            "engineering_recovery": recovery,
            "scientific_updates": 0,
            "resource_observations": resource_envelope,
            "ready_receipt_consumable": True,
        })
        return {"status": "READY", "run_id": run_id, "branch": branch}
    except Exception as error:
        _save_preflight_failure(PREFLIGHT_DIR, error)
        _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", {
            "task_id": TASK_ID, "run_id": run_id, "status": "PREFLIGHT_ENGINEERING_OR_INPUT_ERROR",
            "error_type": type(error).__name__, "error": str(error), "scientific_updates": 0,
        })
        raise


def _task_state(attempt_dir: Path, *, state: str, completed_updates: int,
                branch: str, note: str | None = None) -> None:
    payload: dict[str, Any] = {
        "schema": "lightgen_g1_slm_projected_task_state.v2",
        "task_id": TASK_ID,
        "run_id": _json(attempt_dir / "run_identity.json")["run_id"],
        "state": state,
        "branch": branch,
        "completed_new_updates": completed_updates,
        "source_mapper_updates": 5000,
        "frozen_optical_updates": 0,
        "updated_at_unix": time.time(),
    }
    if note is not None:
        payload["note"] = note
    _atomic_json(attempt_dir / "task_state.json", payload)


def _copy_preflight_evidence(attempt_dir: Path) -> None:
    destination = attempt_dir / "preflight_evidence"
    destination.mkdir(exist_ok=False)
    names = (
        "preflight_identity.json", "preflight_receipt.json", "preflight_summary.json",
        "run_config_snapshot.yaml", "fixed_population_and_schedule.pt",
        "fixed_response_and_projector.pt", "validation_zero_update_rows.csv",
        "validation_zero_update_summary.json", "engineering_recovery.json",
        "validation_reachability_witnesses.json", "validation_reachability_witnesses.pt",
    )
    for name in names:
        source = PREFLIGHT_DIR / name
        if source.is_file():
            shutil_copy = __import__("shutil").copy2
            shutil_copy(source, destination / name)


def _write_initial_checkpoint(
    attempt_dir: Path, mapper: AmplitudeMapper, optimizer: torch.optim.Adam,
    run_id: str, schedule: torch.Tensor,
) -> Path:
    path = attempt_dir / "checkpoints" / "checkpoint_step_000000.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(path, mapper, optimizer, run_id=run_id, completed=0,
                     history=[], schedule=schedule)
    return path


def _train_to_terminal(
    config: Mapping[str, Any], inputs: Mapping[str, Any], attempt_dir: Path,
    population: Mapping[str, Any], schedule: torch.Tensor,
    matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
    device: torch.device, *, resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    identity = _json(attempt_dir / "run_identity.json")
    run_id = str(identity["run_id"])
    mapper = _load_initial_mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    frozen_before = _clone_state(model)
    history: list[dict[str, Any]] = []
    if resume_checkpoint is None:
        torch.manual_seed(int(config["training"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
        completed = 0
        _write_initial_checkpoint(attempt_dir, mapper, optimizer, run_id, schedule)
    else:
        completed, history = _restore_checkpoint(resume_checkpoint, mapper, optimizer,
                                                 run_id=run_id, schedule=schedule)
        _require(completed < 5000, "explicit recovery checkpoint is already terminal")
    trace_path = attempt_dir / "validation_trace.json"
    if trace_path.is_file():
        trace = _json(trace_path).get("evaluations", [])
    else:
        trace = []
    if resume_checkpoint is not None:
        _require(all(isinstance(item, Mapping) and isinstance(item.get("step"), int) for item in trace),
                 "saved validation trace has an invalid row")
        trace = [item for item in trace if int(item["step"]) <= completed]
        _require(len({int(item["step"]) for item in trace}) == len(trace),
                 "saved validation trace contains duplicate checkpoint steps")
    if not any(int(item.get("step", -1)) == 0 for item in trace):
        source_zero = _json(PREFLIGHT_DIR / "validation_zero_update_summary.json")
        trace.append({"step": 0, "source": "full zero-update preflight measurement",
                      "summary": source_zero})
    _atomic_json(trace_path, {"task_id": TASK_ID, "run_id": run_id, "evaluations": trace})
    if resume_checkpoint is not None and completed in {1000, 2500} \
            and not any(int(item.get("step", -1)) == completed for item in trace):
        mapper.eval()
        validation_4096 = pilot._new_optical_model(inputs["checkpoint"], 4096, device)
        validation_4096_before = _clone_state(validation_4096)
        validation_rows = _measure_cohort(
            mapper, {2048: model, 4096: validation_4096}, population["validation"],
            matrix, inverse, scale, device, role=f"validation_step_{completed}",
        )
        validation_summary = _summarize_rows(validation_rows)
        _write_rows_csv(attempt_dir / f"validation_step_{completed}_rows.csv", validation_rows)
        trace.append({"step": completed, "source": "recomputed from explicit recovery checkpoint",
                      "summary": validation_summary})
        _atomic_json(trace_path, {"task_id": TASK_ID, "run_id": run_id, "evaluations": trace})
        _assert_frozen(validation_4096, validation_4096_before,
                       f"C optics grid 4096 during resumed validation at step {completed}")
        del validation_rows, validation_4096
        torch.cuda.empty_cache()
    _atomic_json(attempt_dir / "training_history.json", {
        "task_id": TASK_ID, "run_id": run_id, "source_mapper_updates": 5000,
        "new_update_budget": 5000, "updates": history,
    })
    _task_state(attempt_dir, state=RUN_SCIENTIFIC_ACTIVE, completed_updates=completed,
                branch="ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL")

    for index in range(completed, 5000):
        mapper.train()
        record = _training_update(mapper, optimizer, model, population, schedule,
                                  matrix, inverse, scale, device, index)
        current = index + 1
        record["completed_update"] = current
        history.append(record)
        if current % 100 == 0:
            _atomic_json(attempt_dir / "training_history.json", {
                "task_id": TASK_ID, "run_id": run_id, "source_mapper_updates": 5000,
                "new_update_budget": 5000, "updates": history,
            })
            _atomic_json(attempt_dir / "progress.json", {
                "task_id": TASK_ID, "run_id": run_id, "phase": "training",
                "completed_updates": current, "authorized_new_updates": 5000,
                "last_loss": record["loss"],
                "mean_coefficient_relative_error_rms": record["coefficient_relative_error_rms"],
                "clipped_fraction_mean": record["clipped_fraction_mean"],
                "updated_at_unix": time.time(),
            })
            _task_state(attempt_dir, state=RUN_SCIENTIFIC_ACTIVE, completed_updates=current,
                        branch="ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL")
        if current % 500 == 0:
            checkpoint = attempt_dir / "checkpoints" / f"checkpoint_step_{current:06d}.pt"
            _save_checkpoint(checkpoint, mapper, optimizer, run_id=run_id, completed=current,
                             history=history, schedule=schedule)
            # Reload the just-written endpoint into a fresh CPU model/Adam to verify state and cursor.
            reopened = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cpu_mapper = AmplitudeMapper()
            cpu_mapper.load_state_dict(reopened["mapper_state"], strict=True)
            cpu_optimizer = _optimizer(cpu_mapper, config)
            cpu_optimizer.load_state_dict(copy.deepcopy(reopened["optimizer_state"]))
            mapper_equal, mapper_delta = _tree_max_delta(reopened["mapper_state"], cpu_mapper.state_dict())
            adam_equal, adam_delta = _tree_max_delta(reopened["optimizer_state"], cpu_optimizer.state_dict())
            _require(reopened.get("completed_updates") == current
                     and reopened.get("next_schedule_cursor") == current
                     and mapper_equal and adam_equal,
                     f"500-step recovery checkpoint readback mismatch: mapper={mapper_delta}, Adam={adam_delta}")
        if current in {1000, 2500, 5000}:
            mapper.eval()
            validation_4096 = pilot._new_optical_model(inputs["checkpoint"], 4096, device)
            validation_4096_before = _clone_state(validation_4096)
            validation_rows = _measure_cohort(
                mapper, {2048: model, 4096: validation_4096},
                population["validation"], matrix, inverse, scale, device,
                role=f"validation_step_{current}",
            )
            validation_summary = _summarize_rows(validation_rows)
            _write_rows_csv(attempt_dir / f"validation_step_{current}_rows.csv", validation_rows)
            trace.append({"step": current, "source": "observational only", "summary": validation_summary})
            _atomic_json(trace_path, {"task_id": TASK_ID, "run_id": run_id, "evaluations": trace})
            _assert_frozen(validation_4096, validation_4096_before,
                           f"C optics grid 4096 during validation at step {current}")
            if current == 5000:
                train_probe_indices = torch.cat([
                    torch.arange(label * 1000, label * 1000 + 20) for label in range(CLASS_COUNT)
                ])
                train_probe = {
                    "coefficients": population["train"]["coefficients"][train_probe_indices],
                    "labels": population["train"]["labels"][train_probe_indices],
                    "proposal_ids": population["train"]["proposal_ids"][train_probe_indices],
                }
                train_rows = _measure_cohort(
                    mapper, {2048: model, 4096: validation_4096},
                    train_probe, matrix, inverse, scale, device, role="training_first_20_per_class_terminal",
                )
                _write_rows_csv(attempt_dir / "training_probe_terminal_rows.csv", train_rows)
                _atomic_json(attempt_dir / "training_probe_terminal_summary.json", {
                    "training_probe": _summarize_rows(train_rows),
                    "validation": validation_summary,
                    "comparison_rule": "both cohorts use per-sample relative coefficient and native CMOS errors; report mean/RMS/median/max",
                    "probe_count": 200,
                })
            del validation_rows, validation_4096
            torch.cuda.empty_cache()
        del record

    _atomic_json(attempt_dir / "training_history.json", {
        "task_id": TASK_ID, "run_id": run_id, "source_mapper_updates": 5000,
        "new_update_budget": 5000, "updates": history,
    })
    terminal = attempt_dir / "terminal_checkpoint.pt"
    _save_checkpoint(terminal, mapper, optimizer, run_id=run_id, completed=5000,
                     history=history, schedule=schedule)
    _assert_frozen(model, frozen_before, "C optical model throughout projected training")
    _task_state(attempt_dir, state="TRAINING_COMPLETE_AWAITING_HOLDOUT_MEASUREMENT",
                completed_updates=5000, branch="ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL")
    return {"completed_updates": 5000, "terminal_checkpoint": str(terminal),
            "source_mapper_updates": 5000, "training_history_rows": len(history)}


def _execute_or_resume(
    config: Mapping[str, Any], inputs: Mapping[str, Any], *, resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "projected mapper execution requires AutoDL CUDA")
    preflight = _json(PREFLIGHT_DIR / "preflight_summary.json")
    _require(preflight.get("task_id") == TASK_ID and preflight.get("status") == "READY",
             "execution requires this task's complete READY preflight")
    branch = str(preflight["branch"])
    control, receipt = _load_control(PREFLIGHT_DIR)
    if resume_checkpoint is None:
        control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01", receipt)
        _atomic_json(ATTEMPT_DIR / "run_identity.json", control.identity.to_dict())
        _atomic_json(ATTEMPT_DIR / "run_config_snapshot.yaml", config)
        _copy_preflight_evidence(ATTEMPT_DIR)
        _assert_attempt_identity(ATTEMPT_DIR)
    else:
        _require(ATTEMPT_DIR.is_dir(), "explicit resume requires the existing attempt directory")
        state = _json(ATTEMPT_DIR / "task_state.json")
        _require(state.get("task_id") == TASK_ID
                 and state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "INTERRUPTED"},
                 "resume requires the same active or interrupted attempt; closed or measurement-ready attempts cannot resume")
        _assert_attempt_identity(ATTEMPT_DIR)
        control.open_scientific_attempt(ATTEMPT_DIR,
                                       f"{control.identity.run_id}:attempt_01", receipt, resume=True)
    run_id = control.identity.run_id
    device = torch.device("cuda:0")
    population, schedule = _load_population(ATTEMPT_DIR / "preflight_evidence" / "fixed_population_and_schedule.pt"
                                            if resume_checkpoint is None else
                                            ATTEMPT_DIR / "fixed_population_and_schedule.pt")
    if resume_checkpoint is None:
        # Keep an exact role-labeled copy beside the attempt checkpoints.
        _atomic_torch(ATTEMPT_DIR / "fixed_population_and_schedule.pt", {
            **population, "training_schedule": schedule,
        })
    else:
        population, schedule = _load_population(ATTEMPT_DIR / "fixed_population_and_schedule.pt")
    matrix_dir = ATTEMPT_DIR / "preflight_evidence"
    response, matrix, inverse, scale = _load_matrix(
        matrix_dir, device, expected_response=inputs["response_matrix"], expected_scale=inputs["scale"],
    )
    _require(branch in {"ZERO_UPDATE_CANDIDATE_QUALIFIED_SKIP_TRAINING",
                        "ALL_20_WITNESSES_TRIGGER_FIXED_5000_UPDATE_TRIAL"},
             "READY receipt has an unsupported branch")
    if branch == "ZERO_UPDATE_CANDIDATE_QUALIFIED_SKIP_TRAINING":
        _require(resume_checkpoint is None, "zero-update-only branch has no scientific checkpoint to resume")
        mapper = _load_initial_mapper(inputs, device)
        optimizer = _optimizer(mapper, config)
        _write_initial_checkpoint(ATTEMPT_DIR, mapper, optimizer, run_id, schedule)
        _save_checkpoint(ATTEMPT_DIR / "terminal_checkpoint.pt", mapper, optimizer, run_id=run_id,
                         completed=0, history=[], schedule=schedule)
        _atomic_json(ATTEMPT_DIR / "training_history.json", {
            "task_id": TASK_ID, "run_id": run_id, "source_mapper_updates": 5000,
            "new_update_budget": 5000, "new_updates": 0,
            "decision": "zero-update validation passed; no training performed", "updates": [],
        })
        _task_state(ATTEMPT_DIR, state="ZERO_UPDATE_CANDIDATE_AWAITING_HOLDOUT_MEASUREMENT",
                    completed_updates=0, branch=branch)
        return {"status": "READY_FOR_HOLDOUT_MEASUREMENT", "branch": branch,
                "completed_new_updates": 0, "run_id": run_id}
    if resume_checkpoint is None:
        result = _train_to_terminal(config, inputs, ATTEMPT_DIR, population, schedule,
                                    matrix, inverse, scale, device)
    else:
        result = _train_to_terminal(config, inputs, ATTEMPT_DIR, population, schedule,
                                    matrix, inverse, scale, device, resume_checkpoint=resume_checkpoint)
    return {"status": "READY_FOR_HOLDOUT_MEASUREMENT", "branch": branch,
            "run_id": run_id, **result}


def _quantized_amplitudes(amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a8_codes = torch.floor(255.0 * amplitude.double() + 0.5).clamp(0, 255).to(torch.uint8)
    a8_amplitude = a8_codes.double() / 255.0
    i8_codes = torch.floor(255.0 * amplitude.double().square() + 0.5).clamp(0, 255).to(torch.uint8)
    i8_amplitude = torch.sqrt(i8_codes.double() / 255.0)
    return a8_codes, a8_amplitude, i8_codes, i8_amplitude


def _measure_quantized(
    config: Mapping[str, Any], mapper: AmplitudeMapper,
    models: Mapping[int, base.SmallMatchedAdapter], population: Mapping[str, Any],
    matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
    device: torch.device, out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = population["holdout"]
    for batch_no, start in enumerate(range(0, len(cohort["coefficients"]), 10)):
        stop = min(start + 10, len(cohort["coefficients"]))
        target_cpu = cohort["coefficients"][start:stop]
        target = target_cpu.to(device=device, dtype=torch.complex128)
        features = _normalized_features(target_cpu, scale, device)
        with torch.no_grad():
            continuous, _ = _map_amplitude(mapper, features, target, matrix, inverse, scale)
            a8_codes, a8, i8_codes, i8 = _quantized_amplitudes(continuous)
            commands = {"A8_amplitude_linear": a8, "I8_intensity_linear": i8}
            actual: dict[str, Any] = {}
            direct = {str(grid): _forward_target(models[grid], target)["raw_cmos"].detach().cpu()
                      for grid in GRIDS}
            for condition, command in commands.items():
                actual[condition] = {
                    "amplitude": command.detach().cpu().clone(),
                    "coefficients": {},
                    "raw_cmos_w_per_m2": {},
                }
                for grid in GRIDS:
                    result = _forward_amplitude(models[grid], command)
                    actual[condition]["coefficients"][str(grid)] = result["coefficients"].detach().cpu().clone()
                    actual[condition]["raw_cmos_w_per_m2"][str(grid)] = result["raw_cmos"].detach().cpu().clone()
        _atomic_torch(out_dir / f"holdout_batch_{batch_no:04d}.pt", {
            "schema": "lightgen_g1_slm_projected_quantized_shard.v2",
            "task_id": TASK_ID,
            "population_role": "holdout",
            "population_indices": torch.arange(start, stop),
            "labels": cohort["labels"][start:stop].clone(),
            "proposal_ids": cohort["proposal_ids"][start:stop].clone(),
            "target_coefficients": target_cpu.clone().to(torch.complex128),
            "continuous_amplitude": continuous.detach().cpu().clone(),
            "A8_codes": a8_codes.detach().cpu().clone(),
            "I8_codes": i8_codes.detach().cpu().clone(),
            "A8_definition": "floor(255*a+0.5); decoded command amplitude=gray/255",
            "I8_definition": "floor(255*a^2+0.5); decoded command amplitude=sqrt(gray/255)",
            "direct_raw_cmos_w_per_m2": direct,
            "conditions": actual,
        })
        _atomic_json(out_dir.parent.parent / "progress.json", {
            "task_id": TASK_ID,
            "run_id": _json(out_dir.parent.parent / "run_identity.json")["run_id"],
            "phase": "quantized_measurement",
            "population_role": "holdout",
            "completed_rows": stop,
            "total_rows": len(cohort["coefficients"]),
            "completed_batches": batch_no + 1,
            "updated_at_unix": time.time(),
        })


def _load_attempt_population(attempt_dir: Path) -> tuple[dict[str, Any], torch.Tensor]:
    return _load_population(attempt_dir / "fixed_population_and_schedule.pt")


def _reconsume_continuous(
    attempt_dir: Path, population: Mapping[str, Any], response: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_dir = attempt_dir / "measurements" / "continuous"
    files = sorted(shard_dir.glob("holdout_batch_*.pt"))
    _require(len(files) == 200, f"continuous holdout has {len(files)} batches; expected 200")
    holdout = population["holdout"]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for expected_batch, path in enumerate(files):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("schema") == "lightgen_g1_slm_projected_measurement_shard.v2"
                 and payload.get("task_id") == TASK_ID
                 and payload.get("population_role") == "holdout"
                 and payload.get("command_encoding") == "continuous_amplitude_float32",
                 f"continuous measurement shard identity or command encoding is invalid: {path.name}")
        indices = payload["population_indices"].tolist()
        _require(indices == list(range(expected_batch * 10, min((expected_batch + 1) * 10, 2000))),
                 f"continuous shard order or population identity changed: {path.name}")
        amp = payload["continuous_amplitude"]
        targets = payload["target_coefficients"]
        labels = payload["labels"]
        proposals = payload["proposal_ids"]
        _require(amp.dtype == torch.float32 and tuple(amp.shape) == (len(indices), 1, 200, 200)
                 and torch.isfinite(amp).all().item() and float(amp.min()) >= 0.0 and float(amp.max()) <= 1.0,
                 f"continuous SLM command is malformed: {path.name}")
        _require(torch.equal(targets, holdout["coefficients"][indices].to(torch.complex128))
                 and torch.equal(labels, holdout["labels"][indices])
                 and torch.equal(proposals, holdout["proposal_ids"][indices]),
                 f"saved target, class, or proposal ID differs from fixed new holdout: {path.name}")
        outputs = payload["mapped_coefficients"]
        actual_raw = payload["mapped_raw_cmos_w_per_m2"]
        direct_raw = payload["direct_raw_cmos_w_per_m2"]
        for grid in GRIDS:
            key = str(grid)
            _require(outputs[key].dtype == torch.complex128
                     and tuple(outputs[key].shape) == (len(indices), 100)
                     and actual_raw[key].dtype == torch.float64
                     and tuple(actual_raw[key].shape) == (len(indices), 1, 200, 200)
                     and direct_raw[key].dtype == torch.float64
                     and tuple(direct_raw[key].shape) == (len(indices), 1, 200, 200),
                     f"saved full-optics output or direct CMOS reference is malformed: {path.name}")
        for local, global_index in enumerate(indices):
            _require(global_index not in seen, "duplicate holdout row in continuous measurement")
            seen.add(global_index)
            row_outputs = {
                grid: {"coefficients": outputs[str(grid)][local:local + 1],
                       "raw_cmos": actual_raw[str(grid)][local:local + 1]}
                for grid in GRIDS
            }
            row_direct = {grid: direct_raw[str(grid)][local:local + 1] for grid in GRIDS}
            row = _consume_production_row(
                holdout, global_index, targets[local], amp[local:local + 1],
                row_outputs, row_direct, response, payload["command_encoding"],
            )
            rows.append(row)
    _require(seen == set(range(2000)), "continuous holdout row denominator or identity is incomplete")
    summary = _summarize_rows(rows)
    summary["response_matrix_2048_max_relative_error"] = max(
        row["response_matrix_relative_error_2048"] for row in rows
    )
    return rows, summary


def _verify_mapper_commands(
    attempt_dir: Path, population: Mapping[str, Any], inputs: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_path = attempt_dir / "terminal_checkpoint.pt"
    _require(terminal_path.is_file(), "terminal mapper checkpoint is missing for command replay",
             FileNotFoundError)
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    identity = _json(attempt_dir / "run_identity.json")
    branch = _json(attempt_dir / "preflight_evidence" / "preflight_summary.json")["branch"]
    expected_updates = 0 if branch == "ZERO_UPDATE_CANDIDATE_QUALIFIED_SKIP_TRAINING" else 5000
    _require(terminal.get("schema") == CHECKPOINT_SCHEMA and terminal.get("task_id") == TASK_ID
             and terminal.get("run_id") == identity.get("run_id")
             and terminal.get("completed_updates") == expected_updates,
             "terminal mapper checkpoint identity or update count changed")
    device = torch.device("cpu")
    mapper = _new_mapper(device, terminal["mapper_state"]).eval()
    response, matrix, inverse, scale = _load_matrix(
        attempt_dir / "preflight_evidence", device,
        expected_response=inputs["response_matrix"], expected_scale=inputs["scale"],
    )
    files = sorted((attempt_dir / "measurements" / "continuous").glob("holdout_batch_*.pt"))
    _require(len(files) == 200, "mapper command replay requires all 200 fixed holdout batches")
    maximum_delta = 0.0
    compared = 0
    with torch.no_grad():
        for batch_no, path in enumerate(files):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            indices = payload["population_indices"].tolist()
            expected_indices = list(range(batch_no * 10, min((batch_no + 1) * 10, 2000)))
            _require(indices == expected_indices
                     and torch.equal(payload["target_coefficients"],
                                     population["holdout"]["coefficients"][indices].to(torch.complex128)),
                     f"mapper replay batch has a wrong target identity: {path.name}")
            targets = payload["target_coefficients"].to(torch.complex128)
            features = _normalized_features(targets, scale, device)
            expected_amplitude, _ = _map_amplitude(mapper, features, targets, matrix, inverse, scale)
            saved_amplitude = payload["continuous_amplitude"]
            _require(saved_amplitude.dtype == torch.float32
                     and tuple(saved_amplitude.shape) == tuple(expected_amplitude.shape)
                     and torch.isfinite(saved_amplitude).all().item(),
                     f"mapper replay command is malformed: {path.name}")
            delta = float((expected_amplitude - saved_amplitude).abs().max().item())
            maximum_delta = max(maximum_delta, delta)
            compared += len(indices)
    limit = 5.0e-6
    _require(compared == 2000 and maximum_delta <= limit,
             f"saved amplitude commands do not replay from terminal mapper: max delta {maximum_delta:.3g}")
    return {"passed": True, "population_rows": compared,
            "maximum_absolute_amplitude_delta": maximum_delta,
            "maximum_absolute_amplitude_delta_limit": limit,
            "terminal_mapper_updates": expected_updates,
            "recomputed_from_terminal_mapper_and_fixed_projector": True}


def _reconsume_quantized(attempt_dir: Path, population: Mapping[str, Any],
                         response: torch.Tensor) -> dict[str, Any]:
    shard_dir = attempt_dir / "measurements" / "quantized"
    files = sorted(shard_dir.glob("holdout_batch_*.pt"))
    _require(len(files) == 200, f"quantized holdout has {len(files)} batches; expected 200")
    holdout = population["holdout"]
    by_condition: dict[str, list[dict[str, Any]]] = {
        "A8_amplitude_linear": [], "I8_intensity_linear": [],
    }
    b_np = response.detach().cpu().numpy()
    for expected_batch, path in enumerate(files):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("schema") == "lightgen_g1_slm_projected_quantized_shard.v2"
                 and payload.get("task_id") == TASK_ID
                 and payload.get("population_role") == "holdout",
                 f"quantized shard identity mismatch: {path.name}")
        indices = payload["population_indices"].tolist()
        _require(indices == list(range(expected_batch * 10, min((expected_batch + 1) * 10, 2000)))
                 and torch.equal(payload["target_coefficients"], holdout["coefficients"][indices].to(torch.complex128))
                 and torch.equal(payload["labels"], holdout["labels"][indices])
                 and torch.equal(payload["proposal_ids"], holdout["proposal_ids"][indices]),
                 f"quantized shard does not match fixed holdout row identity: {path.name}")
        continuous = payload["continuous_amplitude"]
        a8_codes, a8_decoded, i8_codes, i8_decoded = _quantized_amplitudes(continuous)
        _require(torch.equal(a8_codes, payload["A8_codes"]) and torch.equal(i8_codes, payload["I8_codes"])
                 and payload.get("A8_definition") == "floor(255*a+0.5); decoded command amplitude=gray/255"
                 and payload.get("I8_definition") == "floor(255*a^2+0.5); decoded command amplitude=sqrt(gray/255)",
                 f"saved A8/I8 bytes do not match their declared encoding: {path.name}")
        expected_commands = {"A8_amplitude_linear": a8_decoded, "I8_intensity_linear": i8_decoded}
        for condition, expected_command in expected_commands.items():
            result = payload["conditions"][condition]
            command = result["amplitude"]
            _require(command.dtype == torch.float64 and torch.equal(command, expected_command),
                     f"decoded quantized command differs from saved grayscale values: {path.name} {condition}")
            response_prediction = command.reshape(len(indices), -1).numpy() @ b_np.T
            for local, global_index in enumerate(indices):
                err_response = pilot._numpy_phase_error(
                    result["coefficients"]["2048"][local].numpy(), response_prediction[local]
                )[1]
                _require(err_response <= MATRIX_LIMIT,
                         f"quantized production coefficients do not match response matrix: {global_index} {condition}")
                outputs = {grid: {
                    "coefficients": result["coefficients"][str(grid)],
                    "raw_cmos": result["raw_cmos_w_per_m2"][str(grid)],
                } for grid in GRIDS}
                direct = payload["direct_raw_cmos_w_per_m2"]
                metrics = _error_row(outputs, payload["target_coefficients"],
                                     {grid: direct[str(grid)] for grid in GRIDS}, local)
                by_condition[condition].append({
                    "population_index": global_index,
                    "label": int(payload["labels"][local]),
                    "proposal_id": int(payload["proposal_ids"][local]),
                    **metrics,
                    "row_passed": all(metrics[key] <= GATE_LIMIT for key in metrics),
                    "response_matrix_relative_error_2048": err_response,
                })
    return {condition: _summarize_rows(rows) for condition, rows in by_condition.items()}


def _auxiliary_evaluators(
    attempt_dir: Path, inputs: Mapping[str, Any], population: Mapping[str, Any],
    *, device: torch.device, pixel_area: float,
) -> dict[str, Any]:
    ed = inputs["evaluator_ed"].to(device).eval()
    ec = inputs["evaluator_ec"].to(device).eval()
    for model in (ed, ec):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    ed_before = _clone_state(ed)
    ec_before = _clone_state(ec)
    ed_predictions: list[int] = []
    ec_predictions: list[int] = []
    labels_all: list[int] = []
    files = sorted((attempt_dir / "measurements" / "continuous").glob("holdout_batch_*.pt"))
    with torch.no_grad():
        for path in files:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            raw = saved["mapped_raw_cmos_w_per_m2"]["2048"].to(device)
            readout = pilot._readout_for_evaluator(raw, pixel_area, inputs["camera_gain"])
            ed_logits = ed(readout)
            _, ec_logits = ec(readout)
            ed_predictions.extend(int(value) for value in ed_logits.argmax(dim=1).cpu().tolist())
            ec_predictions.extend(int(value) for value in ec_logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(int(value) for value in saved["labels"].tolist())
    summary = pilot._auxiliary_evaluator_summary(ed_predictions, ec_predictions, labels_all)
    summary["scope"] = "frozen auxiliary only; E_D/E_C at 2048; not in loss or final gate"
    _assert_frozen(ed, ed_before, "E_D during projected mapping measurement")
    _assert_frozen(ec, ec_before, "E_C during projected mapping measurement")
    _atomic_json(attempt_dir / "auxiliary_ED_EC_2048.json", summary)
    return summary


def _save_contact_sheet(
    attempt_dir: Path, population: Mapping[str, Any], *, pixel_area: float, gain: float,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    records: list[dict[str, Any]] = []
    for path in sorted((attempt_dir / "measurements" / "continuous").glob("holdout_batch_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for local, index in enumerate(saved["population_indices"].tolist()):
            label = int(saved["labels"][local])
            if sum(int(row["label"]) == label for row in records) >= 2:
                continue
            records.append({
                "population_index": index,
                "label": label,
                "sample_index_within_class": index - label * 200,
                "proposal_id": int(saved["proposal_ids"][local]),
                "raw_cmos": saved["mapped_raw_cmos_w_per_m2"]["2048"][local, 0].numpy(),
                "direct_raw_cmos": saved["direct_raw_cmos_w_per_m2"]["2048"][local, 0].numpy(),
            })
    records.sort(key=lambda row: int(row["population_index"]))
    _require(len(records) == 20 and all(sum(int(item["label"]) == label for item in records) == 2
                                         for label in range(CLASS_COUNT)),
             "fixed image subset is not the first two holdout rows of each class")
    mapped_raw = np.stack([row["raw_cmos"] for row in records])
    direct_raw = np.stack([row["direct_raw_cmos"] for row in records])
    mapped_w_per_pixel = mapped_raw * pixel_area
    direct_w_per_pixel = direct_raw * pixel_area
    mapped_gain_scaled = mapped_w_per_pixel * gain
    direct_gain_scaled = direct_w_per_pixel * gain
    mapped_u8 = np.rint(np.clip(mapped_gain_scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    direct_u8 = np.rint(np.clip(direct_gain_scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Independently rebuild the image bytes from native CMOS W/m^2, area, and frozen gain.
    independent_mapped = np.rint(np.clip(mapped_raw * 6.25e-12 * gain, 0.0, 1.0) * 255.0).astype(np.uint8)
    independent_direct = np.rint(np.clip(direct_raw * 6.25e-12 * gain, 0.0, 1.0) * 255.0).astype(np.uint8)
    _require(np.array_equal(mapped_u8, independent_mapped) and np.array_equal(direct_u8, independent_direct),
             "display bytes do not match independent W/m^2 -> W/pixel -> gain conversion")
    figure_dir = attempt_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    margin, label_height, width, height = 8, 18, 200, 200
    canvas = Image.new("RGB", (margin + 5 * (width + margin),
                                margin + 10 * (height + label_height + margin)), "white")
    draw = ImageDraw.Draw(canvas)
    manifest: list[dict[str, Any]] = []
    record_index = {(int(row["label"]), int(row["population_index"])): index
                    for index, row in enumerate(records)}
    for label in range(CLASS_COUNT):
        selected = [row for row in records if int(row["label"]) == label]
        y0 = margin + label * (height + label_height + margin)
        draw.text((margin, y0), f"class {label}", fill="black")
        for slot, row in enumerate(selected):
            image_index = record_index[(label, int(row["population_index"]))]
            x = margin + (1 + slot * 2) * (width + margin)
            canvas.paste(Image.fromarray(mapped_u8[image_index], mode="L").convert("RGB"), (x, y0 + label_height))
            draw.text((x, y0), f"mapped {slot + 1}", fill="black")
            xref = margin + (2 + slot * 2) * (width + margin)
            canvas.paste(Image.fromarray(direct_u8[image_index], mode="L").convert("RGB"),
                         (xref, y0 + label_height))
            draw.text((xref, y0), f"direct {slot + 1}", fill="black")
            manifest.append({"label": label, "sample_index_within_class": row["sample_index_within_class"],
                             "proposal_id": row["proposal_id"], "population_index": row["population_index"]})
    image_path = figure_dir / "continuous_2048_first_two_per_class.png"
    canvas.save(image_path)
    audit_path = figure_dir / "continuous_2048_first_two_per_class_display_reference.npz"
    pilot._atomic_npz(audit_path,
                      labels=np.asarray([row["label"] for row in records], dtype=np.int64),
                      population_indices=np.asarray([row["population_index"] for row in records], dtype=np.int64),
                      proposal_ids=np.asarray([row["proposal_id"] for row in records], dtype=np.int64),
                      mapped_raw_w_per_m2=mapped_raw, direct_raw_w_per_m2=direct_raw,
                      camera_pixel_area_m2=np.asarray(pixel_area),
                      camera_gain_w_inverse=np.asarray(gain),
                      mapped_w_per_pixel=mapped_w_per_pixel, direct_w_per_pixel=direct_w_per_pixel,
                      mapped_display_u8=mapped_u8, direct_display_u8=direct_u8)
    audit = {
        "task_id": TASK_ID,
        "source": "new fixed holdout; first two saved rows of every class",
        "camera_pixel_area_m2": pixel_area,
        "camera_gain_w_inverse": gain,
        "display_conversion": "native W/m^2 * 6.25e-12 m^2/pixel * frozen gain; clip [0,1]; nearest uint8",
        "mapped_pixel_arrays_match_independent_rebuild": True,
        "direct_pixel_arrays_match_independent_rebuild": True,
        "images": manifest,
        "png": image_path.name,
        "pixel_audit_npz": audit_path.name,
    }
    _atomic_json(figure_dir / "continuous_2048_first_two_per_class_manifest.json", audit)
    return audit


def _write_final_report(attempt_dir: Path, decision: Mapping[str, Any]) -> None:
    status = decision["lifecycle_state"]
    gate = decision["continuous_final_gate"]
    branch = decision["branch"]
    new_updates = decision["new_mapper_updates"]
    metrics_table = "\n".join(
        f"| `{name}` | {value:.8g} | {GATE_LIMIT:.1e} |"
        for name, value in gate["maxima"].items()
    )
    class_table = "\n".join(
        f"| {label} | {row['denominator']} | {row['all_six_pass_fraction']:.3f} |"
        for label, row in gate["per_class"].items()
    )
    quantized = decision.get("quantized_auxiliary")
    quantized_status = "未测量（连续主判据未通过）"
    if quantized and "conditions" in quantized:
        quantized_status = "已测量，仅作辅助：" + "; ".join(
            f"{name} {summary['passed_rows']}/{summary['denominator']} rows passed, "
            f"maxima={summary['maxima']}"
            for name, summary in quantized["conditions"].items()
        )
    text = f"""# G1 projected SLM input mapping v2 closeout

## 结果与状态

状态：`{status}`。分支：`{branch}`。本轮新增映射器更新：`{new_updates}`；原映射器已有5,000步。连续输入留出判据：`{gate['status']}`，分母 `{gate['denominator']}`，通过 `{gate['passed_rows']}`，阈值 `{GATE_LIMIT:.1e}`。该状态只适用于新固定2,000条G1留出人口和C45k冻结数值光路。

| 最终误差 | 留出最大值 | 阈值 |
|---|---:|---:|
{metrics_table}

| 类别 | 样本数 | 六项全部通过比例 |
|---:|---:|---:|
{class_table}

## 科学认识

本轮只检验旧映射器经一次固定全秩线性右逆校正后，在需要时再进行固定5,000步全光路训练的连续振幅数值制备能力。公共相位仅用于允许的系数等效表示；振幅仍裁剪到[0,1]。C光学参数与E_D/E_C评价器冻结，评价器结果不参与损失或判据。原C状态`VALID_FAILED`及`PROJECT_GEOMETRY_UNMEASURED`保持。

## 目标缺口

数值留出结果不证明人眼可读性、器件制造、相位稳定、真实相机复现或G1总体可制备。新样本在原总模态功率拒绝条件下独立生成；本轮未读新MNIST原图、未重拟合G1先验。

## 工程完成与验证

测量来自保存的连续振幅指令经过完整SLM、Encoder、OLS、Decoder和原生CMOS路径；Windows消费者将从固定目标、指令、终点映射器、复系数及未归一化CMOS输出重算，回执保存在`windows_reconsumption.json`。相机显示采用W/m²乘像素面积再乘冻结增益。A8/I8状态：{quantized_status}。未进行量化感知训练。

## 决策影响与下一步

按本任务固定停止规则，完成回传及Windows证据复核后停止。本结果不自动授权续训、扩网、G4、盲评或实物实验；340张历史评分仍待评分。`training_recommendation=NONE`，没有自动后继动作。
"""
    (attempt_dir / "final_report.md").write_text(text, encoding="utf-8")


def _write_transfer_manifest(attempt_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in attempt_dir.rglob("*") if item.is_file()):
        if path.name in {"cloud_transfer_manifest.json", "windows_reconsumption.json"}:
            continue
        entries.append({"relative_path": path.relative_to(attempt_dir).as_posix(),
                        "remote_file_bytes": path.stat().st_size})
    manifest = {"schema": "lightgen_g1_slm_projected_transfer_manifest.v2",
                "task_id": TASK_ID, "files": entries, "file_count": len(entries),
                "total_bytes": sum(row["remote_file_bytes"] for row in entries)}
    _atomic_json(attempt_dir / "cloud_transfer_manifest.json", manifest)
    return manifest


def _finish_lifecycle(attempt_dir: Path, passed: bool) -> dict[str, Any]:
    identity = _assert_attempt_identity(attempt_dir)
    receipt = PreflightReceipt.from_mapping(
        _json(attempt_dir / "preflight_evidence" / "preflight_receipt.json")
    )
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    control.open_scientific_attempt(
        attempt_dir, identity.scientific_attempt_id or f"{identity.run_id}:attempt_01",
        receipt, resume=True,
    )
    if passed:
        control.complete()
    else:
        control.valid_failed()
    payload = {
        "task_id": TASK_ID,
        "run_id": control.identity.run_id,
        "scientific_attempt_id": control.identity.scientific_attempt_id,
        "state": control.state,
        "preflight_status": control.preflight_status,
        "receipt_consumed": True,
        "terminal_decision": "continuous_gate_passed" if passed else "continuous_gate_valid_failed",
    }
    _atomic_json(attempt_dir / "research_lifecycle_state.json", payload)
    return payload


def _verify_transfer_sizes(attempt_dir: Path, *, require_transfer_receipt: bool = False) -> dict[str, Any]:
    manifest_path = attempt_dir / "cloud_transfer_manifest.json"
    _require(manifest_path.is_file(), "exact cloud transfer size manifest is missing", FileNotFoundError)
    manifest = _json(manifest_path)
    _require(manifest.get("schema") == "lightgen_g1_slm_projected_transfer_manifest.v2"
             and manifest.get("task_id") == TASK_ID, "transfer manifest identity mismatch")
    rows = []
    expected_paths: set[str] = set()
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "transfer manifest contains a path outside the explicit evidence directory")
        _require(entry["relative_path"] not in expected_paths,
                 "transfer manifest contains a duplicate file path")
        expected_paths.add(entry["relative_path"])
        path = attempt_dir / relative
        _require(path.is_file(), f"transferred result file is missing: {path}", FileNotFoundError)
        _require(path.resolve().is_relative_to(attempt_dir.resolve()),
                 f"transferred result file resolves outside the evidence directory: {path}")
        actual = path.stat().st_size
        expected = int(entry["remote_file_bytes"])
        _require(actual == expected, f"transferred result size mismatch: {path.name} {actual} != {expected}")
        rows.append({"relative_path": entry["relative_path"], "remote_file_bytes": expected,
                     "current_local_file_bytes": actual, "matched": True})
    _require(len(rows) == int(manifest["file_count"]), "transfer manifest file denominator changed")
    receipt_result: dict[str, Any] = {"required": require_transfer_receipt, "checked": False}
    if require_transfer_receipt:
        receipt_path = attempt_dir / "transfer_receipt.json"
        _require(receipt_path.is_file(), "AutoDL pull transfer receipt is missing", FileNotFoundError)
        transfer_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        _require(isinstance(transfer_receipt, dict), "AutoDL pull transfer receipt is not a JSON object")
        _require(transfer_receipt.get("status") == "COMPLETE"
                 and transfer_receipt.get("remote_alias") == "autodl-monthly"
                 and transfer_receipt.get("remote_run_dir") == (
                     "/root/autodl-tmp/codex-original/outputs/20261027_"
                     "lightgen_g1_slm_projected_mapping_v2/runs/attempt_01"
                 )
                 and Path(str(transfer_receipt.get("destination", ""))).resolve() == attempt_dir.resolve()
                 and isinstance(transfer_receipt.get("files"), list),
                 "AutoDL pull transfer receipt is incomplete")
        transfer_rows = transfer_receipt["files"]
        by_path = {str(item.get("path")): item for item in transfer_rows}
        _require(len(by_path) == len(transfer_rows), "AutoDL pull receipt contains duplicate paths")
        allowed_extra = {"cloud_transfer_manifest.json", "resource_telemetry.csv"}
        _require(expected_paths.issubset(by_path)
                 and set(by_path).issubset(expected_paths | allowed_extra),
                 "AutoDL pull receipt path set differs from the frozen cloud manifest")
        receipt_files: list[dict[str, Any]] = []
        for relative, item in by_path.items():
            _require(item.get("status") == "COPIED"
                     and isinstance(item.get("remote_size_bytes"), int)
                     and isinstance(item.get("local_size_bytes"), int),
                     f"AutoDL pull receipt has a missing or failed size record: {relative}")
            path = attempt_dir / relative
            _require(path.is_file() and path.resolve().is_relative_to(attempt_dir.resolve()),
                     f"AutoDL pull file is missing or resolves outside the backup: {relative}")
            actual = path.stat().st_size
            remote_recorded = int(item["remote_size_bytes"])
            local_recorded = int(item["local_size_bytes"])
            _require(remote_recorded == local_recorded == actual,
                     f"three-way transfer size mismatch for {relative}: remote={remote_recorded}, "
                     f"receipt_local={local_recorded}, actual={actual}")
            if relative in expected_paths:
                manifest_size = int(next(row["remote_file_bytes"] for row in manifest["files"]
                                         if row["relative_path"] == relative))
                _require(remote_recorded == manifest_size,
                         f"AutoDL pull receipt and cloud manifest disagree for {relative}")
            receipt_files.append({"relative_path": relative,
                                  "remote_file_bytes": remote_recorded,
                                  "recorded_local_file_bytes": local_recorded,
                                  "current_local_file_bytes": actual,
                                  "matched": True})
        receipt_result = {"required": True, "checked": True,
                          "status": transfer_receipt["status"], "files": receipt_files}
    return {"file_count": len(rows), "total_bytes": sum(row["current_local_file_bytes"] for row in rows),
            "all_sizes_match": True, "files": rows, "transfer_receipt": receipt_result}


def _consume_saved(attempt_dir: Path, inputs: Mapping[str, Any], *,
                   finalize: bool = False, write_rows: bool = True) -> dict[str, Any]:
    population, _ = _load_attempt_population(attempt_dir)
    response = inputs["response_matrix"].to(torch.complex128)
    continuous_rows, continuous = _reconsume_continuous(attempt_dir, population, response)
    if write_rows:
        _write_rows_csv(attempt_dir / "per_sample_measurements.csv", continuous_rows)
    decision: dict[str, Any] = {
        "task_id": TASK_ID,
        "run_id": _json(attempt_dir / "run_identity.json")["run_id"],
        "evidence_tier": "diagnostic_only",
        "source_mapper_updates": 5000,
        "new_mapper_updates": int(torch.load(
            attempt_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False
        )["completed_updates"]),
        "branch": _json(attempt_dir / "preflight_evidence" / "preflight_summary.json")["branch"],
        "continuous_final_gate": {
            "status": "PASS" if continuous["all_rows_passed"] else "VALID_FAILED",
            **continuous,
            "coefficient_alignment": "one global phase per row",
            "native_cmos": "raw W/m^2; no normalization, power matching, or gain adjustment",
            "consumer_recomputed_from_saved_targets_commands_coefficients_and_raw_cmos": True,
        },
        "historical_C_status_preserved": "VALID_FAILED",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        "hardware_acceptance": False,
    }
    quantized_dir = attempt_dir / "measurements" / "quantized"
    if continuous["all_rows_passed"] and quantized_dir.is_dir():
        quantized = _reconsume_quantized(attempt_dir, population, response)
        decision["quantized_auxiliary"] = {
            "gate_use": False,
            "quantization_aware_training": False,
            "conditions": quantized,
        }
    elif not continuous["all_rows_passed"]:
        _require(not quantized_dir.exists(),
                 "quantized measurement exists although the continuous gate failed")
        decision["quantized_auxiliary"] = {"status": "NOT_RUN_CONTINUOUS_GATE_FAILED",
                                            "gate_use": False}
    lifecycle_state = RUN_COMPLETED if continuous["all_rows_passed"] else RUN_VALID_FAILED
    decision["lifecycle_state"] = lifecycle_state
    if finalize:
        lifecycle = _finish_lifecycle(attempt_dir, continuous["all_rows_passed"])
        _require(lifecycle["state"] == lifecycle_state,
                 "strict research lifecycle terminal state differs from independently consumed gate")
        _atomic_json(attempt_dir / "continuous_gate.json", decision["continuous_final_gate"])
        _atomic_json(attempt_dir / "final_decision.json", decision)
        _auxiliary_evaluators(
            attempt_dir, inputs, population, device=torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu"), pixel_area=6.25e-12,
        )
        _save_contact_sheet(attempt_dir, population, pixel_area=6.25e-12,
                            gain=float(inputs["camera_gain"]))
        _atomic_json(attempt_dir / "final_decision.json", decision)
        _write_final_report(attempt_dir, decision)
        _task_state(attempt_dir, state=lifecycle_state,
                    completed_updates=int(decision["new_mapper_updates"]),
                    branch=str(decision["branch"]),
                    note="terminal holdout measurement consumed from saved production outputs")
        _write_transfer_manifest(attempt_dir)
        _verify_transfer_sizes(attempt_dir)
    return decision


def _measure_terminal(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "terminal full-optics holdout measurement requires AutoDL CUDA")
    _require(ATTEMPT_DIR.is_dir(), "explicit attempt directory is missing", FileNotFoundError)
    state = _json(ATTEMPT_DIR / "task_state.json")
    _require(state.get("task_id") == TASK_ID and state.get("state") not in {RUN_COMPLETED, RUN_VALID_FAILED},
             "closed or foreign attempt cannot be measured")
    _assert_attempt_identity(ATTEMPT_DIR)
    _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "MEASURING_FIXED_FINAL_HOLDOUT",
                                    "TRAINING_COMPLETE_AWAITING_HOLDOUT_MEASUREMENT",
                                    "ZERO_UPDATE_CANDIDATE_AWAITING_HOLDOUT_MEASUREMENT"},
             "attempt is not at a terminal measurement boundary")
    terminal_path = ATTEMPT_DIR / "terminal_checkpoint.pt"
    _require(terminal_path.is_file(), "terminal projected mapper checkpoint is missing", FileNotFoundError)
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    run_id = _json(ATTEMPT_DIR / "run_identity.json")["run_id"]
    expected_updates = 0 if state["branch"] == "ZERO_UPDATE_CANDIDATE_QUALIFIED_SKIP_TRAINING" else 5000
    _require(terminal.get("task_id") == TASK_ID and terminal.get("run_id") == run_id
             and terminal.get("completed_updates") == expected_updates,
             "terminal mapper checkpoint does not match this explicit branch and update budget")
    device = torch.device("cuda:0")
    population, _ = _load_attempt_population(ATTEMPT_DIR)
    matrix_dir = ATTEMPT_DIR / "preflight_evidence"
    response, matrix, inverse, scale = _load_matrix(
        matrix_dir, device, expected_response=inputs["response_matrix"], expected_scale=inputs["scale"],
    )
    mapper = _new_mapper(device, terminal["mapper_state"]).eval()
    models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
    model_states = {grid: _clone_state(model) for grid, model in models.items()}
    _task_state(ATTEMPT_DIR, state="MEASURING_FIXED_FINAL_HOLDOUT",
                completed_updates=expected_updates, branch=state["branch"])
    continuous_dir = ATTEMPT_DIR / "measurements" / "continuous"
    rows = _measure_cohort(mapper, models, population["holdout"], matrix, inverse, scale,
                           device, role="holdout", shard_dir=continuous_dir)
    _write_rows_csv(ATTEMPT_DIR / "producer_per_sample_measurements.csv", rows)
    for grid in GRIDS:
        _assert_frozen(models[grid], model_states[grid], f"C optics grid {grid} during final measurement")
    del models, mapper
    torch.cuda.empty_cache()
    decision = _consume_saved(ATTEMPT_DIR, inputs, finalize=False)
    if decision["continuous_final_gate"]["status"] == "PASS":
        mapper = _new_mapper(device, terminal["mapper_state"]).eval()
        models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
        qdir = ATTEMPT_DIR / "measurements" / "quantized"
        _measure_quantized(config, mapper, models, population, matrix, inverse, scale, device, qdir)
        for grid in GRIDS:
            _assert_frozen(models[grid], model_states[grid], f"C optics grid {grid} during quantized measurement")
        del models, mapper
        torch.cuda.empty_cache()
        decision = _consume_saved(ATTEMPT_DIR, inputs, finalize=True)
    else:
        decision = _consume_saved(ATTEMPT_DIR, inputs, finalize=True)
    manifest = _json(ATTEMPT_DIR / "cloud_transfer_manifest.json")
    return {"status": decision["lifecycle_state"],
            "continuous_gate": decision["continuous_final_gate"],
            "quantized_measured": "quantized_auxiliary" in decision,
            "transfer_manifest_file_count": manifest["file_count"],
            "attempt_dir": str(ATTEMPT_DIR)}


def _numeric_tree_matches(actual: Any, expected: Any, tolerance: float = 1.0e-14) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(_numeric_tree_matches(actual[key], expected[key], tolerance)
                                                    for key in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_numeric_tree_matches(a, b, tolerance)
                                                    for a, b in zip(actual, expected, strict=True))
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) \
            and abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def _verify_display_outputs(attempt_dir: Path, population: Mapping[str, Any],
                            pixel_area: float, gain: float) -> dict[str, Any]:
    from PIL import Image

    figure = attempt_dir / "figures" / "continuous_2048_first_two_per_class.png"
    npz = attempt_dir / "figures" / "continuous_2048_first_two_per_class_display_reference.npz"
    manifest = _json(attempt_dir / "figures" / "continuous_2048_first_two_per_class_manifest.json")
    _require(figure.is_file() and npz.is_file() and manifest.get("task_id") == TASK_ID,
             "fixed display artifacts are missing or belong to another task")
    with np.load(npz) as saved:
        mapped_raw = saved["mapped_raw_w_per_m2"]
        direct_raw = saved["direct_raw_w_per_m2"]
        expected_mapped = np.rint(np.clip(mapped_raw * pixel_area * gain, 0.0, 1.0) * 255.0).astype(np.uint8)
        expected_direct = np.rint(np.clip(direct_raw * pixel_area * gain, 0.0, 1.0) * 255.0).astype(np.uint8)
        _require(float(saved["camera_pixel_area_m2"]) == pixel_area
                 and float(saved["camera_gain_w_inverse"]) == gain
                 and np.array_equal(saved["mapped_w_per_pixel"], mapped_raw * pixel_area)
                 and np.array_equal(saved["direct_w_per_pixel"], direct_raw * pixel_area)
                 and np.array_equal(saved["mapped_display_u8"], expected_mapped)
                 and np.array_equal(saved["direct_display_u8"], expected_direct),
                 "saved display conversion does not match independent unit and pixel recomputation")
        indices = saved["population_indices"].tolist()
        proposal_ids = saved["proposal_ids"].tolist()
        labels = saved["labels"].tolist()
    expected_rows = [label * 200 + slot for label in range(10) for slot in range(2)]
    expected_proposals = [int(population["holdout"]["proposal_ids"][idx]) for idx in expected_rows]
    expected_labels = [label for label in range(10) for _ in range(2)]
    _require(indices == expected_rows and proposal_ids == expected_proposals and labels == expected_labels,
             "image subset is not the fixed first-two-per-class holdout subset")
    image = Image.open(figure).convert("RGB")
    image_array = np.asarray(image)
    _require(image.size == (1048, 2268), "contact-sheet dimensions changed")
    for i, label in enumerate(expected_labels):
        slot = i % 2
        row = label
        mapped_x = 8 + (1 + slot * 2) * 208
        direct_x = 8 + (2 + slot * 2) * 208
        y = 8 + row * 226 + 18
        mapped_crop = image_array[y:y + 200, mapped_x:mapped_x + 200]
        direct_crop = image_array[y:y + 200, direct_x:direct_x + 200]
        _require(np.array_equal(mapped_crop, np.repeat(expected_mapped[i, :, :, None], 3, axis=2))
                 and np.array_equal(direct_crop, np.repeat(expected_direct[i, :, :, None], 3, axis=2)),
                 f"rendered contact-sheet pixels do not match source arrays for class {label}, slot {slot}")
    return {"passed": True, "count": 20, "camera_pixel_area_m2": pixel_area,
            "camera_gain_w_inverse": gain, "source_unit": "raw CMOS W/m^2",
            "converted_unit": "W/pixel", "png_pixels_recomputed": True}


def _verify_saved(evidence_dir: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(evidence_dir.is_dir(), f"saved evidence directory is missing: {evidence_dir}", FileNotFoundError)
    sizes = _verify_transfer_sizes(evidence_dir, require_transfer_receipt=True)
    _assert_attempt_identity(evidence_dir)
    decision = _json(evidence_dir / "final_decision.json")
    state = _json(evidence_dir / "task_state.json")
    lifecycle = _json(evidence_dir / "research_lifecycle_state.json")
    _require(decision.get("task_id") == TASK_ID and state.get("task_id") == TASK_ID
             and decision.get("lifecycle_state") == state.get("state")
             and lifecycle.get("task_id") == TASK_ID
             and lifecycle.get("run_id") == decision.get("run_id")
             and lifecycle.get("state") == decision.get("lifecycle_state")
             and lifecycle.get("receipt_consumed") is True,
             "saved decision and task lifecycle identity disagree")
    _require(decision.get("continuous_final_gate", {}).get("denominator") == 2000
             and decision.get("continuous_final_gate", {}).get("threshold") == GATE_LIMIT,
             "saved continuous gate denominator or threshold changed")
    population, _ = _load_attempt_population(evidence_dir)
    mapper_replay = _verify_mapper_commands(evidence_dir, population, inputs)
    recomputed = _consume_saved(evidence_dir, inputs, finalize=False, write_rows=False)
    _require(_numeric_tree_matches(decision["continuous_final_gate"], recomputed["continuous_final_gate"]),
             "saved final gate differs from independent NumPy reconsumption of raw outputs")
    _require(decision.get("new_mapper_updates") == recomputed.get("new_mapper_updates")
             and decision.get("branch") == recomputed.get("branch"),
             "saved scientific update count or route branch differs from terminal source")
    quant_dir = evidence_dir / "measurements" / "quantized"
    if decision["continuous_final_gate"]["status"] == "PASS":
        _require(quant_dir.is_dir(), "continuous pass requires the planned same-holdout A8/I8 measurement")
    else:
        _require(not quant_dir.exists(), "quantized measurement is present despite a failed continuous gate")
    if quant_dir.exists():
        _require(decision["continuous_final_gate"]["status"] == "PASS"
                 and _numeric_tree_matches(decision.get("quantized_auxiliary", {}).get("conditions"),
                                           recomputed.get("quantized_auxiliary", {}).get("conditions")),
                 "quantized measurements lack a continuous pass or disagree with reconsumption")
    display = _verify_display_outputs(evidence_dir, population, 6.25e-12,
                                      float(inputs["camera_gain"]))
    report = evidence_dir / "final_report.md"
    _require(report.is_file(), "five-section final report is missing")
    report_text = report.read_text(encoding="utf-8")
    required_sections = ("## 结果与状态", "## 科学认识", "## 目标缺口",
                         "## 工程完成与验证", "## 决策影响与下一步")
    _require(all(section in report_text for section in required_sections),
             "final report omits one or more required feedback sections")
    receipt = {
        "task_id": TASK_ID,
        "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
        "decision": decision["continuous_final_gate"]["status"],
        "independent_numpy_metric_recomputation": True,
        "terminal_mapper_command_replay": mapper_replay,
        "all_continuous_rows_recomputed": True,
        "quantized_recomputed_when_present": quant_dir.exists(),
        "display_pixel_recomputation": display,
        "file_sizes": sizes,
        "continuous_gate": recomputed["continuous_final_gate"],
        "branch": recomputed["branch"],
        "new_mapper_updates": recomputed["new_mapper_updates"],
    }
    _atomic_json(evidence_dir / "windows_reconsumption.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "execute", "resume", "measure", "verify-saved", "recovery-worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path,
                        help="required explicit checkpoint for resume or the isolated recovery worker")
    parser.add_argument("--evidence-dir", type=Path,
                        help="explicit saved attempt path for verify-saved")
    parser.add_argument("--worker-output", type=Path,
                        help="explicit output path used only by the isolated recovery worker")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.mode == "preflight":
        result = _preflight(config_path, config)
    elif args.mode == "recovery-worker":
        _require(args.checkpoint is not None and args.worker_output is not None,
                 "recovery-worker requires explicit --checkpoint and --worker-output", ValueError)
        _engineering_worker(config_path, PREFLIGHT_DIR, args.checkpoint.resolve(), args.worker_output.resolve())
        result = {"status": "ENGINEERING_RECOVERY_STEP_COMPLETE"}
    elif args.mode == "verify-saved":
        _require(args.evidence_dir is not None, "verify-saved requires an explicit --evidence-dir", ValueError)
        inputs = _load_inputs(config)
        result = _verify_saved(args.evidence_dir.resolve(), inputs)
    elif args.mode == "measure":
        inputs = _load_inputs(config)
        result = _measure_terminal(config, inputs)
    elif args.mode == "resume":
        _require(args.checkpoint is not None, "resume requires an explicit --checkpoint", ValueError)
        inputs = _load_inputs(config)
        result = _execute_or_resume(config, inputs, resume_checkpoint=args.checkpoint.resolve())
    else:
        inputs = _load_inputs(config)
        result = _execute_or_resume(config, inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
