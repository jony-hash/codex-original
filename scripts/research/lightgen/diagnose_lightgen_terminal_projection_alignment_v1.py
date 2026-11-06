"""Zero-update endpoint-response projection alignment on the saved LightGen population."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reach
from scripts.research.lightgen import diagnose_lightgen_fullpath_zero_update_attribution_v1 as attribution
from scripts.research.lightgen import execute_lightgen_recent_optical_fullpath_semantic_v1 as recent
from scripts.research.lightgen import measure_lightgen_train_generation_ols_support_v1 as support


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_TERMINAL_PROJECTION_ALIGNMENT_V1"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/terminal_projection_alignment_v1.yaml"
OUTPUT_DIR = ROOT / "outputs/20261102_lightgen_terminal_projection_alignment_v1/runs/attempt_01"
BACKUP_DIR = ROOT / "outputs/20261102_lightgen_terminal_projection_alignment_v1/runs/autodl_backup_attempt_01"
REVIEW_DIR = ROOT / "outputs/20261102_lightgen_terminal_projection_alignment_v1/runs/windows_review_01"
CLASS_COUNT = 10
ROWS_PER_CLASS = 200
CALIBRATION_ROWS_PER_CLASS = 100
POPULATION_ROWS = CLASS_COUNT * ROWS_PER_CLASS
PORT_COUNT = 100
IMAGE_SHAPE = (200, 200)
BATCH_SIZE = 10
CONTROL_ROWS = torch.tensor([label * ROWS_PER_CLASS + offset
                             for label in range(CLASS_COUNT) for offset in range(2)],
                            dtype=torch.long)
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 202611021070
COMMAND_TOLERANCE = 1.0e-6
BASELINE_REPLAY_TOLERANCE = 1.0e-12
RESPONSE_TOLERANCE = 1.0e-6
DERIVATIVE_TOLERANCE = 1.0e-5
IDENTITY_TOLERANCE = 1.0e-10
CROSS_GRID_TOLERANCE = 1.0e-4
READOUT_TOLERANCE = 1.0e-12
CHUNK_SIZE = 100


def _require(condition: bool, message: str,
             error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}", ValueError)
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _save_torch(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        _require(not path.exists(), f"refusing to replace saved evidence: {path}", FileExistsError)
    temporary = path.with_name(path.name + f".{time.time_ns()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_torch(path: Path) -> Any:
    _require(path.is_file(), f"required saved artifact is missing: {path}", FileNotFoundError)
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("schema")
             == "lightgen_terminal_projection_alignment.v1"
             and value.get("task_id") == TASK_ID
             and value.get("evidence_tier") == "diagnostic_only",
             "projection-alignment configuration identity changed", ValueError)
    _require(value.get("sources", {}).get("expected_training_run_id")
             == "d88d5bfc2b29488884a88512152815d6"
             and value["sources"].get("expected_population_seed_base") == 202610300930
             and value["sources"].get("expected_mapper_run_id")
             == "3521afc652f2427eac984485971340c4",
             "registered LightGen training or mapper lineage changed", ValueError)
    _require(value.get("population") == {
        "role": "consumed_train_diagnostic", "rows": 2_000,
        "rows_per_class": 200, "class_count": 10, "official_test_reads": 0,
    }, "registered consumed population or role changed", ValueError)
    measurement = value.get("measurement", {})
    _require(measurement == {
        "endpoint": "E1G1", "endpoint_global_step": 52_000,
        "encoder_adam_step": 47_000, "generator_adam_step": 52_000,
        "production_grid": 2048, "reference_grid": 4096,
        "preflight_rows_per_class": 2, "command_replay_max_abs": COMMAND_TOLERANCE,
        "baseline_replay_max_abs": BASELINE_REPLAY_TOLERANCE,
        "response_relative_error_max": RESPONSE_TOLERANCE,
        "response_fd_epsilon": 2.0e-4,
        "response_derivative_relative_error_max": DERIVATIVE_TOLERANCE,
        "right_inverse_identity_max_abs": IDENTITY_TOLERANCE,
        "cross_grid_relative_error_max": CROSS_GRID_TOLERANCE,
        "inference_batch_size": BATCH_SIZE,
        "bootstrap_resamples": BOOTSTRAP_COUNT, "bootstrap_seed": BOOTSTRAP_SEED,
        "ed_delta_ci95_lower_min": 0.0, "ec_macro_recall_drop_max": 0.02,
        "class_diversity_retention_min": 0.8,
    }, "registered measurement or semantic decision contract changed", ValueError)
    _require(value.get("output") == {
        "run_dir": OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "backup_dir": BACKUP_DIR.relative_to(ROOT).as_posix(),
        "review_dir": REVIEW_DIR.relative_to(ROOT).as_posix(),
        "optimizer_updates": 0, "optimizer_instances": 0,
        "prior_refits": 0, "prior_samples": 0, "official_test_reads": 0,
        "preserve_source_artifacts": True, "stop_after_one_semantic_decision": True,
    }, "registered output paths or zero-update boundary changed", ValueError)
    return value


def _load_assets(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    sources = config["sources"]
    recent_config = recent._read_config(_path(sources["fullpath_config"]))
    attribution_config = yaml.safe_load(
        _path(sources["attribution_config"]).read_text(encoding="utf-8")
    )
    assets = attribution._load_saved_assets(attribution_config, device,
                                            use_verified_backup=True)
    _require(assets["run_id"] == sources["expected_training_run_id"]
             and assets["checkpoints"][2_000].get("completed_global_step") == 52_000
             and assets["checkpoints"][2_000].get("optimizer_steps") == {
                 "encoder_adam_step": 47_000, "generator_adam_step": 52_000,
             }, "the explicit D0 step-2000 endpoint or Adam counters changed")
    runtime = assets["runtime"]
    _require(runtime["mapper_checkpoint"].get("run_id") == sources["expected_mapper_run_id"]
             and runtime["mapper_checkpoint"].get("completed_updates") == 15_000
             and float(runtime["mapper_checkpoint"].get("selected_alpha", math.nan)) == 1.0,
             "the frozen alpha=1 mapper checkpoint changed")

    support_root = _path(sources["train_generation_backup"])
    support_review = support_root.parent / "windows_review_02"
    support_review_record = _read_json(support_review / "independent_review.json")
    _require(support_review_record.get("passed") is True
             and support_review_record.get("decision_route") == "DIFFERENCE_EXPANDED"
             and support_review_record.get("attempt03_noncontrol_rows_reused_bit_identically") == 3_960
             and support_review_record.get("corrected_2048_forwards") == 40
             and support_review_record.get("corrected_4096_reference_forwards") == 40,
             "attempt_04 corrected input-identity evidence is not independently verified")
    control_identity = _read_json(support_root / "control_input_identity_validation.json")
    _require(control_identity.get("passed") is True
             or control_identity.get("status") == "PASSED",
             "attempt_04 corrected control source identity did not pass")
    train_payload = _load_torch(support_root / "training_fullpath_outputs.pt")
    _require(train_payload.get("task_id") == support.TASK_ID
             and train_payload.get("source_training_run_id") == sources["expected_training_run_id"]
             and train_payload.get("population_role") == "consumed_train_diagnostic"
             and train_payload.get("grid") == 2048
             and torch.equal(train_payload.get("modal_scale"), runtime["scale"].cpu().double())
             and set(train_payload.get("endpoints", {})) == {0, 2_000},
             "attempt_04 TRAIN full-path coefficients or modal-scale binding changed")
    train_terminal = train_payload["endpoints"][2_000]
    train_labels, source_rows = train_terminal["labels"].long().cpu(), train_terminal["source_rows"].long().cpu()
    _require(tuple(train_terminal["coefficients"].shape) == (2_000, PORT_COUNT)
             and torch.is_complex(train_terminal["coefficients"])
             and torch.equal(torch.bincount(train_labels, minlength=CLASS_COUNT),
                             torch.full((CLASS_COUNT,), ROWS_PER_CLASS))
             and torch.is_tensor(runtime["input_bundle"].get("labels"))
             and torch.is_tensor(runtime["input_bundle"].get("train_indices")),
             "terminal TRAIN coefficient population is incomplete")
    labels_in_dataset = runtime["input_bundle"]["labels"].cpu()
    train_index_set = set(runtime["input_bundle"]["train_indices"].long().cpu().tolist())
    _require(all(int(row) in train_index_set and int(labels_in_dataset[row]) == int(label)
                 for row, label in zip(source_rows.tolist(), train_labels.tolist(), strict=True)),
             "saved terminal reference rows are not the matching source TRAIN images")

    generation = _load_torch(support_root / "generation_coefficients.pt")
    # Reuse the population already bound and validated by the saved attribution
    # source loader; the original attempt_01 population file is absent locally.
    saved_population = assets["population"]
    cohort, baseline_commands = saved_population["cohort"], saved_population["commands"]
    _require(saved_population.get("seed_base") == sources["expected_population_seed_base"]
             and saved_population.get("mapper_checkpoint_run_id") == sources["expected_mapper_run_id"]
             and generation.get("task_id") == support.TASK_ID
             and generation.get("population_role") == "consumed_diagnostic_population"
             and tuple(generation.get("saved_prior_targets", torch.empty(0)).shape)
             == (POPULATION_ROWS, PORT_COUNT)
             and torch.equal(generation.get("labels"), cohort["labels"])
             and torch.equal(generation.get("proposal_ids"), cohort["proposal_ids"])
             and torch.equal(generation.get("saved_prior_targets"), cohort["coefficients"])
             and torch.equal(generation.get("labels"), torch.arange(CLASS_COUNT).repeat_interleave(
                 ROWS_PER_CLASS))
             and tuple(baseline_commands.shape) == (POPULATION_ROWS, 1, *IMAGE_SHAPE),
             "saved prior population, labels, proposals, commands, or source mapper changed")
    _require(torch.equal(generation["actual_E1G1_coefficients"],
                         assets["final"]["coefficients"]),
             "attempt_04 terminal coefficients differ from the independently saved E1G1 output")

    attribution_outputs = _load_torch(
        _path(sources["attribution_backup"]) / "condition_outputs.pt"
    )
    _require(attribution_outputs.get("task_id") == attribution.TASK_ID
             and torch.equal(attribution_outputs.get("labels"), generation["labels"])
             and torch.equal(attribution_outputs.get("proposal_ids"), generation["proposal_ids"])
             and torch.equal(attribution_outputs.get("commands"), baseline_commands)
             and set(attribution_outputs.get("conditions", {})) == set(attribution.CONDITION_ORDER),
             "saved E1G1 outputs do not bind to the same prior targets and SLM commands")
    baseline = attribution_outputs["conditions"]["E1G1"]
    for key in ("coefficients",):
        _require(torch.equal(baseline[key], generation["actual_E1G1_coefficients"]),
                 f"the two registered E1G1 coefficient records differ: {key}")
    _require(tuple(baseline.get("raw_cmos", torch.empty(0)).shape)
             == (POPULATION_ROWS, 1, *IMAGE_SHAPE)
             and tuple(baseline.get("readouts", torch.empty(0)).shape)
             == (POPULATION_ROWS, 1, 28, 28),
             "saved baseline raw CMOS or fixed display views are incomplete")
    return {
        "config": recent_config,
        "assets": assets,
        "runtime": runtime,
        "mapper": recent._new_mapper(runtime),
        "population": saved_population,
        "generation": generation,
        "baseline_commands": baseline_commands.float().cpu(),
        "baseline_outputs": baseline,
        "train_payload": train_payload,
        "train_endpoint": train_terminal,
        "train_labels": train_labels,
        "train_rows": source_rows,
        "source_train_indices": runtime["input_bundle"]["train_indices"].long().cpu(),
        "support_root": support_root,
        "attribution_outputs": attribution_outputs,
    }


def _phase_distance(matrix: torch.Tensor, scale: torch.Tensor,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    _require(tuple(matrix.shape) == (200, 40_000) and matrix.dtype == torch.float64,
             "normalized terminal response matrix shape or dtype changed")
    inverse, details = recent.projected._pseudoinverse(matrix)
    identity_error = float((matrix @ inverse - torch.eye(200, dtype=torch.float64,
                                                        device=device)).abs().max().item())
    _require(identity_error <= IDENTITY_TOLERANCE,
             f"terminal response right inverse residual {identity_error:.3g} exceeds tolerance")
    return inverse, {**details, "max_abs_A1P1_minus_I": identity_error}


def _build_response(assets: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    runtime = assets["runtime"]
    device = runtime["device"]
    checkpoint = attribution._condition_checkpoint(assets["assets"], "E1G1")
    model = recent._new_endpoint_model(checkpoint, 2048, runtime)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    response = reach._make_response(model, device, lambda _event: None)
    _require(tuple(response.shape) == (PORT_COUNT, 40_000)
             and response.dtype == torch.complex128,
             "terminal production response is not complex128 [100,40000]")
    normalized = recent.projected._normalized_response(response, runtime["scale"])
    inverse, inverse_stats = _phase_distance(normalized, runtime["scale"], device)

    # Independent finite-difference check of the production amplitude entry.
    generator = torch.Generator(device="cpu").manual_seed(202611021050)
    direction = (torch.randint(0, 2, IMAGE_SHAPE, generator=generator, dtype=torch.int64)
                 .to(device=device, dtype=torch.float64).mul_(2.0).sub_(1.0))
    center = torch.full((1, 1, *IMAGE_SHAPE), 0.5, dtype=torch.float64, device=device)
    epsilon = 2.0e-4
    with torch.no_grad():
        plus = model.forward_from_image((center + epsilon * direction).square())[
            "ols_coefficients"].reshape(-1)
        minus = model.forward_from_image((center - epsilon * direction).square())[
            "ols_coefficients"].reshape(-1)
    numerical = (plus - minus) / (2.0 * epsilon)
    analytical = response @ direction.reshape(-1).to(torch.complex128)
    derivative_error = float((torch.linalg.vector_norm(analytical - numerical)
                              / torch.linalg.vector_norm(numerical).clamp_min(1.0e-30)).item())
    _require(derivative_error <= DERIVATIVE_TOLERANCE,
             f"terminal response finite difference error {derivative_error:.3g} exceeds tolerance")
    model_delta = recent._tree_max_delta(before, model.state_dict())
    _require(model_delta == 0.0, "response computation changed the frozen E1G1 optical model")
    numpy_check = recent.projected._matrix_checks(
        response, normalized, inverse, runtime["scale"], device,
    )
    return response.detach().cpu(), inverse.detach().cpu(), {
        "response_shape": list(response.shape),
        "response_dtype": str(response.dtype),
        "response_derivative_relative_error": derivative_error,
        "response_derivative_tolerance": DERIVATIVE_TOLERANCE,
        "right_inverse": inverse_stats,
        "independent_matrix_projection": numpy_check,
        "terminal_model_state_max_delta": model_delta,
        "optimizer_instances": 0,
        "optimizer_updates": 0,
    }


def _normalised_features_and_commands(mapper: torch.nn.Module, targets: torch.Tensor,
                                      matrix: torch.Tensor, inverse: torch.Tensor,
                                      scale: torch.Tensor, device: torch.device
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = targets.detach().cpu().to(torch.complex128)
    features = recent.projected._normalized_features(targets, scale, device)
    commands: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    clipped: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(targets), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(targets))
            target = targets[start:stop].to(device=device, dtype=torch.complex128)
            command, detail = recent.projected._map_amplitude(
                mapper, features[start:stop], target, matrix, inverse, scale,
            )
            commands.append(command.detach().float().cpu())
            phases.append(detail["phase"].detach().double().cpu())
            clipped.append(detail["clipped_fraction"].detach().double().cpu())
    result = torch.cat(commands)
    recent._assert_legal_commands(result)
    return result, torch.cat(phases), torch.cat(clipped)


def _measure_grid(checkpoint: Mapping[str, Any], commands: torch.Tensor,
                  runtime: Mapping[str, Any], grid: int) -> dict[str, torch.Tensor]:
    model = recent._new_endpoint_model(checkpoint, grid, runtime)
    coeff, raw, readouts, powers = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(commands), BATCH_SIZE):
            command = commands[start:start + BATCH_SIZE].to(runtime["device"])
            output = recent.projected._forward_amplitude(model, command)
            coeff.append(output["coefficients"].detach().cpu())
            raw.append(output["raw_cmos"].detach().cpu())
            readouts.append(recent.pilot._readout_for_evaluator(
                output["raw_cmos"].detach().cpu(), runtime["camera_area"],
                float(runtime["camera_gain"]),
            ).cpu())
            powers.append(output["coefficients"].abs().square().sum(dim=1).detach().cpu())
    return {"coefficients": torch.cat(coeff), "raw_cmos": torch.cat(raw),
            "readouts": torch.cat(readouts).float(), "modal_power": torch.cat(powers).double()}


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    dtype = torch.complex128 if left.is_complex() or right.is_complex() else torch.float64
    left_value = left.to(dtype=dtype)
    right_value = right.to(dtype=dtype)
    denominator = torch.linalg.vector_norm(right_value).clamp_min(1.0e-30)
    return float((torch.linalg.vector_norm(left_value - right_value) / denominator).item())


def _predict_coefficients(commands: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    _require(commands.ndim == 4 and commands.shape[1] == 1
             and response.ndim == 2 and commands[0].numel() == response.shape[1]
             and torch.is_floating_point(commands) and response.is_complex()
             and torch.isfinite(commands).all().item()
             and torch.isfinite(response.real).all().item()
             and torch.isfinite(response.imag).all().item(),
             "response prediction requires finite real SLM commands and a complex response matrix")
    flattened = commands.reshape(len(commands), -1).to(
        device=response.device, dtype=response.dtype,
    )
    return flattened @ response.T


def _validate_right_inverse(matrix: torch.Tensor, inverse: torch.Tensor,
                            tolerance: float) -> float:
    _require(torch.is_tensor(matrix) and torch.is_tensor(inverse)
             and matrix.dtype == torch.float64 and inverse.dtype == torch.float64
             and matrix.ndim == 2 and inverse.ndim == 2
             and matrix.shape[0] == inverse.shape[1]
             and matrix.shape[1] == inverse.shape[0]
             and matrix.device == inverse.device
             and torch.isfinite(matrix).all().item()
             and torch.isfinite(inverse).all().item(),
             "response/right-inverse tensors have incompatible shape, dtype, device, or values")
    identity = torch.eye(matrix.shape[0], dtype=torch.float64, device=matrix.device)
    residual = float((matrix @ inverse - identity).abs().max().item())
    _require(residual <= tolerance,
             f"response/right-inverse identity residual {residual:.3g} exceeds {tolerance:.3g}")
    return residual


def _preflight(assets: Mapping[str, Any], response: torch.Tensor,
               inverse: torch.Tensor, response_stats: Mapping[str, Any]
               ) -> dict[str, Any]:
    runtime = assets["runtime"]
    inputs = runtime["input_bundle"]
    checkpoint = attribution._condition_checkpoint(assets["assets"], "E1G1")
    target = assets["generation"]["saved_prior_targets"]
    labels = assets["generation"]["labels"]
    controls = CONTROL_ROWS
    old_commands, _old_summary = recent._make_commands(
        assets["mapper"], target, runtime,
    )
    command_replay_error = float((old_commands - assets["baseline_commands"]).abs().max())
    _require(command_replay_error <= COMMAND_TOLERANCE,
             f"P0 no longer reconstructs its saved SLM commands: {command_replay_error:.3g}")

    baseline_saved = assets["baseline_outputs"]
    baseline = _measure_grid(checkpoint, assets["baseline_commands"].index_select(0, controls),
                             runtime, 2048)
    baseline_replay = {}
    for key in ("coefficients", "raw_cmos", "readouts"):
        expected = baseline_saved[key].index_select(0, controls)
        error = float((baseline[key] - expected).abs().max())
        _require(error <= BASELINE_REPLAY_TOLERANCE,
                 f"P0 E1G1 saved {key} replay failed: {error:.3g}")
        baseline_replay[key] = {"max_abs_error": error, "tolerance": BASELINE_REPLAY_TOLERANCE}

    normalized = recent.projected._normalized_response(response.to(runtime["device"]),
                                                        runtime["scale"])
    # A1 is recomputed from the saved terminal R1; this is the matrix actually passed to P1.
    identity_error = _validate_right_inverse(
        normalized, inverse.to(runtime["device"]), IDENTITY_TOLERANCE,
    )
    wrong_binding_rejected = False
    try:
        _validate_right_inverse(normalized.roll(1, dims=0),
                                inverse.to(runtime["device"]), IDENTITY_TOLERANCE)
    except RuntimeError:
        wrong_binding_rejected = True
    _require(wrong_binding_rejected,
             "wrong response-row/right-inverse binding was not rejected")
    p1_commands, public_phase, clipped = _normalised_features_and_commands(
        assets["mapper"], target, normalized, inverse.to(runtime["device"]),
        runtime["scale"], runtime["device"],
    )
    recent._assert_legal_commands(p1_commands)
    # Re-evaluate the terminal R1 against the real optical coefficients from the P0 replay.
    predicted = _predict_coefficients(
        assets["baseline_commands"].index_select(0, controls), response.to(runtime["device"]),
    )
    response_row_errors = [
        _relative_l2(predicted[index].cpu(), baseline["coefficients"][index])
        for index in range(len(controls))
    ]
    max_response_error = max(response_row_errors)
    _require(max_response_error <= RESPONSE_TOLERANCE,
             f"R1 fails saved-production response rows: {max_response_error:.3g}")

    candidate2048 = _measure_grid(checkpoint, p1_commands.index_select(0, controls), runtime, 2048)
    candidate4096 = _measure_grid(checkpoint, p1_commands.index_select(0, controls), runtime, 4096)
    cross_coeff = [_relative_l2(candidate2048["coefficients"][index],
                                candidate4096["coefficients"][index])
                   for index in range(len(controls))]
    cross_cmos = [_relative_l2(candidate2048["raw_cmos"][index],
                               candidate4096["raw_cmos"][index])
                  for index in range(len(controls))]
    _require(max(cross_coeff) <= CROSS_GRID_TOLERANCE
             and max(cross_cmos) <= CROSS_GRID_TOLERANCE,
             "P1 cross-grid complex coefficient or raw CMOS reference exceeds 1e-4")
    frozen_snapshot = recent._snapshot_frozen(runtime, assets["mapper"])
    return {
        "task_id": TASK_ID, "status": "PASSED",
        "endpoint": "E1G1", "endpoint_global_step": 52_000,
        "rows": 20, "rows_per_class": 2,
        "p0_command_max_abs_error": command_replay_error,
        "p0_e1g1_replay": baseline_replay,
        "r1_vs_saved_e1g1_max_relative_error": max_response_error,
        "r1_saved_production_row_relative_errors": response_row_errors,
        "p1_inverse_recomputed_stats": {"max_abs_A1P1_minus_I": identity_error},
        "wrong_response_row_binding_rejected": wrong_binding_rejected,
        "response_build": dict(response_stats),
        "cross_grid": {"coefficient_max_relative_error": max(cross_coeff),
                       "raw_cmos_max_relative_error": max(cross_cmos),
                       "coefficient_tolerance": CROSS_GRID_TOLERANCE,
                       "raw_cmos_tolerance": CROSS_GRID_TOLERANCE},
        "p1_control_commands": p1_commands.index_select(0, controls),
        "p1_control_coefficients_2048": candidate2048["coefficients"],
        "p1_control_raw_cmos_2048": candidate2048["raw_cmos"],
        "p1_control_readouts_2048": candidate2048["readouts"],
        "p1_control_modal_power_2048": candidate2048["modal_power"],
        "p1_control_coefficients_4096": candidate4096["coefficients"],
        "p1_control_raw_cmos_4096": candidate4096["raw_cmos"],
        "p1_control_public_phase": public_phase.index_select(0, controls),
        "p1_control_clipped_fraction": clipped.index_select(0, controls),
        "frozen_snapshot": frozen_snapshot,
        "optimizer_instances": 0,
        "optimizer_updates": 0,
    }


def _revalidate_saved_preflight(assets: Mapping[str, Any], projector: Mapping[str, Any],
                                saved: Mapping[str, Any], device: torch.device
                                ) -> dict[str, Any]:
    """Recheck persisted preflight outputs with the corrected complex relative norm."""
    _require(saved.get("task_id") == TASK_ID and saved.get("status") == "PASSED"
             and saved.get("endpoint") == "E1G1"
             and saved.get("endpoint_global_step") == 52_000
             and saved.get("optimizer_instances") == 0
             and saved.get("optimizer_updates") == 0,
             "saved preflight cannot be revalidated for this frozen endpoint")
    runtime = assets["runtime"]
    response = projector["response"].to(device)
    matrix = projector["normalized_real_matrix"].to(device)
    inverse = projector["right_inverse"].to(device)
    identity_error = _validate_right_inverse(matrix, inverse, IDENTITY_TOLERANCE)
    wrong_binding_rejected = False
    try:
        _validate_right_inverse(matrix.roll(1, dims=0), inverse, IDENTITY_TOLERANCE)
    except RuntimeError:
        wrong_binding_rejected = True
    _require(wrong_binding_rejected,
             "revalidated wrong response-row/right-inverse binding was not rejected")

    targets = assets["generation"]["saved_prior_targets"]
    old_commands, _ = recent._make_commands(assets["mapper"], targets, runtime)
    command_error = float((old_commands - assets["baseline_commands"]).abs().max())
    _require(command_error <= COMMAND_TOLERANCE,
             "P0 source commands changed while revalidating the saved preflight")
    replay = saved.get("p0_e1g1_replay", {})
    _require(set(replay) == {"coefficients", "raw_cmos", "readouts"}
             and all(float(item["max_abs_error"]) <= BASELINE_REPLAY_TOLERANCE
                     and float(item["tolerance"]) == BASELINE_REPLAY_TOLERANCE
                     for item in replay.values()),
             "saved P0 full-path replay did not pass its registered tolerance")

    control_rows = CONTROL_ROWS
    response_prediction = _predict_coefficients(
        assets["baseline_commands"].index_select(0, control_rows), response,
    ).detach().cpu()
    baseline_actual = assets["baseline_outputs"]["coefficients"].index_select(0, control_rows)
    response_errors = [
        _relative_l2(response_prediction[index], baseline_actual[index])
        for index in range(len(control_rows))
    ]
    _require(max(response_errors) <= RESPONSE_TOLERANCE,
             "corrected complex R1 errors exceed the saved E1G1 response tolerance")

    p1_commands, phase, clipped = _normalised_features_and_commands(
        assets["mapper"], targets, matrix, inverse, runtime["scale"], device,
    )
    saved_commands = saved.get("p1_control_commands")
    expected_controls = p1_commands.index_select(0, control_rows).detach().cpu()
    _require(torch.is_tensor(saved_commands)
             and tuple(saved_commands.shape) == tuple(expected_controls.shape)
             and float((saved_commands - expected_controls).abs().max()) <= COMMAND_TOLERANCE,
             "saved P1 control commands do not reproduce from the explicit targets and projector")

    coeff2048 = saved.get("p1_control_coefficients_2048")
    coeff4096 = saved.get("p1_control_coefficients_4096")
    raw2048 = saved.get("p1_control_raw_cmos_2048")
    raw4096 = saved.get("p1_control_raw_cmos_4096")
    _require(torch.is_tensor(coeff2048) and torch.is_tensor(coeff4096)
             and torch.is_tensor(raw2048) and torch.is_tensor(raw4096)
             and tuple(coeff2048.shape) == tuple(coeff4096.shape) == (20, PORT_COUNT)
             and tuple(raw2048.shape) == tuple(raw4096.shape) == (20, 1, *IMAGE_SHAPE),
             "saved P1 cross-grid control outputs are incomplete")
    coeff_errors = [_relative_l2(coeff2048[index], coeff4096[index]) for index in range(20)]
    cmos_errors = [_relative_l2(raw2048[index], raw4096[index]) for index in range(20)]
    _require(max(coeff_errors) <= CROSS_GRID_TOLERANCE
             and max(cmos_errors) <= CROSS_GRID_TOLERANCE,
             "correctly recomputed P1 cross-grid reference exceeds 1e-4")
    response_stats = saved.get("response_build", {})
    _require(float(response_stats.get("response_derivative_relative_error", math.inf))
             <= DERIVATIVE_TOLERANCE
             and float(response_stats.get("terminal_model_state_max_delta", math.inf)) == 0.0
             and response_stats.get("optimizer_instances") == 0
             and response_stats.get("optimizer_updates") == 0,
             "saved terminal response derivative or zero-update evidence is invalid")
    recent._assert_frozen_unchanged(saved["frozen_snapshot"], runtime, assets["mapper"])
    result = dict(saved)
    result.update({
        "p0_command_max_abs_error": command_error,
        "r1_vs_saved_e1g1_max_relative_error": max(response_errors),
        "r1_saved_production_row_relative_errors": response_errors,
        "p1_inverse_recomputed_stats": {"max_abs_A1P1_minus_I": identity_error},
        "wrong_response_row_binding_rejected": wrong_binding_rejected,
        "cross_grid": {
            "coefficient_max_relative_error": max(coeff_errors),
            "raw_cmos_max_relative_error": max(cmos_errors),
            "coefficient_tolerance": CROSS_GRID_TOLERANCE,
            "raw_cmos_tolerance": CROSS_GRID_TOLERANCE,
        },
        "p1_control_commands": expected_controls,
        "p1_control_public_phase": phase.index_select(0, control_rows).detach().cpu(),
        "p1_control_clipped_fraction": clipped.index_select(0, control_rows).detach().cpu(),
        "revalidation_source": "persisted_preflight_outputs_with_correct_complex_relative_norm",
        "status": "PASSED",
    })
    return result


def _safe_shard_name(indices: Sequence[int]) -> str:
    return f"production_2048_{int(indices[0]):04d}_{int(indices[-1]) + 1:04d}.pt"


def _save_production_chunk(path: Path, indices: torch.Tensor, outputs: Mapping[str, torch.Tensor],
                           commands: torch.Tensor, assets: Mapping[str, Any]) -> None:
    population = assets["generation"]
    _save_torch(path, {
        "task_id": TASK_ID,
        "population_role": "consumed_train_diagnostic",
        "condition": "P1_E1G1",
        "population_indices": indices.long().cpu(),
        "labels": population["labels"].index_select(0, indices).long().cpu(),
        "proposal_ids": population["proposal_ids"].index_select(0, indices).long().cpu(),
        "prior_targets": population["saved_prior_targets"].index_select(0, indices).cpu(),
        "actual_slm_commands_forwarded": commands.index_select(0, indices).float().cpu(),
        "projection_source": "R1_terminal_E1G1_plus_P1",
        "coefficients": outputs["coefficients"].cpu(),
        "raw_cmos": outputs["raw_cmos"].cpu(),
        "modal_power": outputs["modal_power"].cpu(),
    })


def _run_measure(config: Mapping[str, Any], artifact_dir: Path, *, resume: bool) -> None:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for production measurements")
    device = torch.device("cuda")
    assets = _load_assets(config, device)
    if resume:
        _require(artifact_dir.is_dir(), "--resume requires the explicit existing run directory",
                 FileNotFoundError)
        _require(yaml.safe_load((artifact_dir / "run_config.yaml").read_text(encoding="utf-8"))
                 == config, "saved run config differs from the explicit registered task")
        projector_bundle = _load_torch(artifact_dir / "response_projector.pt")
        _validate_projector_bundle(projector_bundle, assets, device)
        saved_preflight = _load_torch(artifact_dir / "production_preflight.pt")
        _require(saved_preflight.get("status") == "PASSED",
                 "saved production preflight is not complete")
        preflight = _revalidate_saved_preflight(
            assets, projector_bundle, saved_preflight, device,
        )
        revalidation_path = artifact_dir / "production_preflight_revalidation.pt"
        revalidation_json_path = artifact_dir / "production_preflight_revalidation.json"
        if revalidation_path.is_file():
            existing_revalidation = _load_torch(revalidation_path)
            _require(existing_revalidation.get("revalidation_source")
                     == preflight["revalidation_source"]
                     and existing_revalidation.get("cross_grid") == preflight["cross_grid"]
                     and existing_revalidation.get("r1_saved_production_row_relative_errors")
                     == preflight["r1_saved_production_row_relative_errors"],
                     "saved preflight revalidation differs from recomputed persisted outputs")
            preflight = existing_revalidation
        else:
            _save_torch(revalidation_path, preflight)
        compact_revalidation = {
            key: value for key, value in preflight.items()
            if not key.startswith("p1_control_") and key != "frozen_snapshot"
        }
        if revalidation_json_path.is_file():
            _require(_read_json(revalidation_json_path) == compact_revalidation,
                     "saved preflight revalidation JSON differs from its tensor evidence")
        else:
            _write_json(revalidation_json_path, compact_revalidation, exclusive=True)
        controls = preflight
    else:
        _require(not artifact_dir.exists(), f"refusing to replace existing run directory: {artifact_dir}",
                 FileExistsError)
        response, inverse, response_stats = _build_response(assets)
        normalized = recent.projected._normalized_response(
            response.to(device), assets["runtime"]["scale"]
        ).cpu()
        projector_bundle = {
            "task_id": TASK_ID, "endpoint": "E1G1", "endpoint_global_step": 52_000,
            "training_run_id": config["sources"]["expected_training_run_id"],
            "encoder_adam_step": 47_000, "generator_adam_step": 52_000,
            "modal_scale": assets["runtime"]["scale"].detach().cpu().double(),
            "response": response, "normalized_real_matrix": normalized,
            "right_inverse": inverse, "real_imag_order": "per_port_interleaved_real_imag",
            "projection_method": "full_row_rank_svd_moore_penrose_right_inverse",
        }
        preflight = _preflight(assets, response.to(device), inverse.to(device), response_stats)
        # Keep bulk tensors in a dedicated control result and compact JSON in the summary.
        control_tensors = {key: value for key, value in preflight.items()
                           if key.startswith("p1_control_")}
        compact_preflight = {key: value for key, value in preflight.items()
                             if not key.startswith("p1_control_") and key != "frozen_snapshot"}
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "run_config.yaml").write_text(
            yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        _save_torch(artifact_dir / "response_projector.pt", projector_bundle)
        _save_torch(artifact_dir / "production_preflight.pt", {
            **compact_preflight, **control_tensors,
            "frozen_snapshot": preflight["frozen_snapshot"],
        })
        _write_json(artifact_dir / "production_preflight.json", compact_preflight, exclusive=True)
        controls = dict(control_tensors)
        controls["frozen_snapshot"] = preflight["frozen_snapshot"]

    response = projector_bundle["response"].to(device)
    matrix = projector_bundle["normalized_real_matrix"].to(device)
    inverse = projector_bundle["right_inverse"].to(device)
    runtime = assets["runtime"]
    targets = assets["generation"]["saved_prior_targets"]
    commands, phases, clipped = _normalised_features_and_commands(
        assets["mapper"], targets, matrix, inverse,
        runtime["scale"], device,
    )
    _require(commands.shape == assets["baseline_commands"].shape
             and bool(((commands >= 0.0) & (commands <= 1.0)).all().item()),
             "P1 failed to produce a complete legal SLM command population")
    recent._assert_frozen_unchanged(controls["frozen_snapshot"], runtime, assets["mapper"])
    manifest = {
        "task_id": TASK_ID, "population_role": "consumed_train_diagnostic",
        "population_rows": POPULATION_ROWS,
        "labels": assets["generation"]["labels"],
        "proposal_ids": assets["generation"]["proposal_ids"],
        "prior_targets": assets["generation"]["saved_prior_targets"],
        "candidate_projection": "R1_terminal_E1G1_plus_P1",
        "population_seed_base": config["sources"]["expected_population_seed_base"],
    }
    manifest_path = artifact_dir / "population_manifest.pt"
    if manifest_path.is_file():
        existing_manifest = _load_torch(manifest_path)
        _require(existing_manifest.get("task_id") == manifest["task_id"]
                 and existing_manifest.get("population_role") == manifest["population_role"]
                 and existing_manifest.get("population_rows") == manifest["population_rows"]
                 and existing_manifest.get("candidate_projection") == manifest["candidate_projection"]
                 and existing_manifest.get("population_seed_base") == manifest["population_seed_base"]
                 and torch.equal(existing_manifest.get("labels"), manifest["labels"])
                 and torch.equal(existing_manifest.get("proposal_ids"), manifest["proposal_ids"])
                 and torch.equal(existing_manifest.get("prior_targets"), manifest["prior_targets"]),
                 "saved population manifest differs from the fixed prior targets")
    else:
        _save_torch(manifest_path, manifest)
    control_rows = CONTROL_ROWS
    metadata_path = artifact_dir / "candidate_projection_metadata.pt"
    metadata = {
        "public_phase": phases, "clipped_fraction": clipped,
        "command_min": float(commands.min()), "command_max": float(commands.max()),
    }
    if metadata_path.is_file():
        existing_metadata = _load_torch(metadata_path)
        _require(torch.allclose(existing_metadata.get("public_phase"), metadata["public_phase"],
                                rtol=0.0, atol=1.0e-12)
                 and torch.allclose(existing_metadata.get("clipped_fraction"),
                                    metadata["clipped_fraction"], rtol=0.0, atol=1.0e-12)
                 and math.isclose(float(existing_metadata.get("command_min", math.nan)),
                                  metadata["command_min"], rel_tol=0.0, abs_tol=1.0e-12)
                 and math.isclose(float(existing_metadata.get("command_max", math.nan)),
                                  metadata["command_max"], rel_tol=0.0, abs_tol=1.0e-12),
                 "saved P1 command diagnostics differ on explicit recovery")
    else:
        _save_torch(metadata_path, metadata)
    control_payload = {
        "task_id": TASK_ID, "population_indices": control_rows,
        "labels": manifest["labels"].index_select(0, control_rows),
        "proposal_ids": manifest["proposal_ids"].index_select(0, control_rows),
        "prior_targets": manifest["prior_targets"].index_select(0, control_rows),
        "actual_slm_commands_forwarded": commands.index_select(0, control_rows),
        "coefficients": controls["p1_control_coefficients_2048"],
        "raw_cmos": controls["p1_control_raw_cmos_2048"],
        "readouts": controls["p1_control_readouts_2048"],
        "modal_power": controls["p1_control_modal_power_2048"],
        "projection_source": "R1_terminal_E1G1_plus_P1",
    }
    controls_path = artifact_dir / "controls_2048.pt"
    if controls_path.is_file():
        existing_controls = _load_torch(controls_path)
        controls_match = all(
            (torch.is_tensor(existing_controls.get(key)) and torch.equal(
                existing_controls[key], value,
            )) if torch.is_tensor(value) else existing_controls.get(key) == value
            for key, value in control_payload.items()
        )
        _require(controls_match,
                 "saved P1 2048 controls differ from their preflight outputs")
    else:
        _save_torch(controls_path, control_payload)
    cross_payload = {
        "task_id": TASK_ID, "population_indices": control_rows,
        "labels": manifest["labels"].index_select(0, control_rows),
        "proposal_ids": manifest["proposal_ids"].index_select(0, control_rows),
        "actual_slm_commands_forwarded": commands.index_select(0, control_rows),
        "coefficients_2048": controls["p1_control_coefficients_2048"],
        "raw_cmos_2048": controls["p1_control_raw_cmos_2048"],
        "coefficients_4096": controls["p1_control_coefficients_4096"],
        "raw_cmos_4096": controls["p1_control_raw_cmos_4096"],
        "relative_error_max": controls["cross_grid"],
    }
    cross_path = artifact_dir / "cross_grid_controls.pt"
    if cross_path.is_file():
        existing_cross = _load_torch(cross_path)
        cross_match = all(
            (torch.is_tensor(existing_cross.get(key)) and torch.equal(
                existing_cross[key], value,
            )) if torch.is_tensor(value) else existing_cross.get(key) == value
            for key, value in cross_payload.items()
        )
        _require(cross_match,
                 "saved P1 cross-grid controls differ from preflight outputs")
    else:
        _save_torch(cross_path, cross_payload)

    control_rows_set = set(CONTROL_ROWS.tolist())
    remaining = torch.tensor([index for index in range(POPULATION_ROWS)
                              if index not in control_rows_set], dtype=torch.long)
    checkpoint = attribution._condition_checkpoint(assets["assets"], "E1G1")
    for start in range(0, len(remaining), CHUNK_SIZE):
        indices = remaining[start:start + CHUNK_SIZE]
        shard_path = artifact_dir / _safe_shard_name(indices.tolist())
        if shard_path.is_file():
            shard = _load_torch(shard_path)
            _validate_shard(shard, indices, commands, assets)
            continue
        outputs = _measure_grid(checkpoint, commands.index_select(0, indices), runtime, 2048)
        _save_production_chunk(shard_path, indices, outputs, commands, assets)
        _write_json(artifact_dir / "measurement_progress.json", {
            "task_id": TASK_ID, "status": "PRODUCTION_MEASURING",
            "completed_noncontrol_rows": min(start + len(indices), len(remaining)),
            "total_noncontrol_rows": len(remaining),
            "new_2048_forward_rows": 20 + min(start + len(indices), len(remaining)),
            "new_4096_reference_rows": 20,
            "optimizer_updates": 0, "prior_samples": 0,
        })
    recent._assert_frozen_unchanged(controls["frozen_snapshot"], runtime, assets["mapper"])
    _consume_and_write(config, artifact_dir, assets, projector_bundle)


def _validate_projector_bundle(bundle: Mapping[str, Any], assets: Mapping[str, Any],
                               device: torch.device) -> None:
    scale = assets["runtime"]["scale"].detach().cpu().double()
    _require(bundle.get("task_id") == TASK_ID and bundle.get("endpoint") == "E1G1"
             and bundle.get("endpoint_global_step") == 52_000
             and bundle.get("training_run_id") == "d88d5bfc2b29488884a88512152815d6"
             and bundle.get("encoder_adam_step") == 47_000
             and bundle.get("generator_adam_step") == 52_000
             and torch.equal(bundle.get("modal_scale"), scale)
             and bundle.get("real_imag_order") == "per_port_interleaved_real_imag"
             and bundle.get("projection_method")
             == "full_row_rank_svd_moore_penrose_right_inverse",
             "saved response/projector bundle is bound to the wrong endpoint or scale")
    response = bundle.get("response")
    matrix = bundle.get("normalized_real_matrix")
    inverse = bundle.get("right_inverse")
    _require(torch.is_tensor(response) and response.shape == (100, 40_000)
             and response.dtype == torch.complex128
             and torch.is_tensor(matrix) and matrix.shape == (200, 40_000)
             and matrix.dtype == torch.float64
             and torch.is_tensor(inverse) and inverse.shape == (40_000, 200)
             and inverse.dtype == torch.float64,
             "saved endpoint response/projector tensors have invalid shape or precision")
    a_recomputed = recent.projected._normalized_response(
        response.to(device), assets["runtime"]["scale"]
    )
    matrix_error = float((matrix.cpu() - a_recomputed.cpu()).abs().max().item())
    _require(matrix_error <= 1.0e-14,
             f"saved A1 does not recompute from terminal R1: {matrix_error:.3g}")
    identity_error = float((a_recomputed @ inverse.to(device)
                            - torch.eye(200, dtype=torch.float64, device=device)).abs().max())
    _require(identity_error <= IDENTITY_TOLERANCE,
             f"saved P1 identity residual {identity_error:.3g} exceeds tolerance")


def _validate_shard(shard: Mapping[str, Any], indices: torch.Tensor,
                    commands: torch.Tensor, assets: Mapping[str, Any], *,
                    compare_commands: bool = True) -> None:
    population = assets["generation"]
    _require(shard.get("task_id") == TASK_ID
             and shard.get("population_role") == "consumed_train_diagnostic"
             and shard.get("condition") == "P1_E1G1"
             and torch.equal(shard.get("population_indices"), indices)
             and torch.equal(shard.get("labels"), population["labels"].index_select(0, indices))
             and torch.equal(shard.get("proposal_ids"), population["proposal_ids"].index_select(0, indices))
             and torch.equal(shard.get("prior_targets"), population["saved_prior_targets"].index_select(0, indices))
             and (not compare_commands or torch.equal(
                 shard.get("actual_slm_commands_forwarded"), commands.index_select(0, indices)))
             and shard.get("projection_source") == "R1_terminal_E1G1_plus_P1"
             and tuple(shard.get("coefficients", torch.empty(0)).shape) == (len(indices), 100)
             and tuple(shard.get("raw_cmos", torch.empty(0)).shape)
             == (len(indices), 1, *IMAGE_SHAPE)
             and torch.isfinite(shard["coefficients"].real).all().item()
             and torch.isfinite(shard["coefficients"].imag).all().item()
             and torch.isfinite(shard["raw_cmos"]).all().item(),
             "saved production shard has wrong inputs, projection source, rows, or output")


def _load_candidate_output(artifact_dir: Path, assets: Mapping[str, Any],
                           commands: torch.Tensor) -> dict[str, torch.Tensor]:
    labels = assets["generation"]["labels"]
    proposal_ids = assets["generation"]["proposal_ids"]
    indices = torch.arange(POPULATION_ROWS, dtype=torch.long)
    coefficients = torch.empty((POPULATION_ROWS, PORT_COUNT), dtype=torch.complex128)
    raw = torch.empty((POPULATION_ROWS, 1, *IMAGE_SHAPE), dtype=torch.float64)
    seen = torch.zeros(POPULATION_ROWS, dtype=torch.bool)
    control_path = artifact_dir / "controls_2048.pt"
    controls = _load_torch(control_path)
    control_indices = CONTROL_ROWS
    _require(torch.equal(controls.get("population_indices"), control_indices)
             and torch.equal(controls.get("labels"), labels.index_select(0, control_indices))
             and torch.equal(controls.get("proposal_ids"), proposal_ids.index_select(0, control_indices))
             and torch.equal(controls.get("prior_targets"), assets["generation"][
                 "saved_prior_targets"].index_select(0, control_indices))
             and torch.equal(controls.get("actual_slm_commands_forwarded"),
                             commands.index_select(0, control_indices))
             and controls.get("projection_source") == "R1_terminal_E1G1_plus_P1",
             "P1 preflight rows do not match their saved targets, labels, or commands")
    coefficients[control_indices] = controls["coefficients"]
    raw[control_indices] = controls["raw_cmos"]
    seen[control_indices] = True
    remaining = torch.tensor([index for index in range(POPULATION_ROWS)
                              if index not in set(control_indices.tolist())], dtype=torch.long)
    for start in range(0, len(remaining), CHUNK_SIZE):
        shard_indices = remaining[start:start + CHUNK_SIZE]
        shard = _load_torch(artifact_dir / _safe_shard_name(shard_indices.tolist()))
        _validate_shard(shard, shard_indices, commands, assets)
        _require(not seen[shard_indices].any().item(), "saved P1 output population contains duplicate rows")
        coefficients[shard_indices] = shard["coefficients"]
        raw[shard_indices] = shard["raw_cmos"]
        seen[shard_indices] = True
    _require(seen.all().item() and tuple(coefficients.shape) == (POPULATION_ROWS, PORT_COUNT)
             and len(indices) == POPULATION_ROWS,
             "P1 output population is incomplete")
    projection_metadata = _load_torch(artifact_dir / "candidate_projection_metadata.pt")
    _require(tuple(projection_metadata.get("public_phase", torch.empty(0)).shape) == (POPULATION_ROWS,)
             and tuple(projection_metadata.get("clipped_fraction", torch.empty(0)).shape)
             == (POPULATION_ROWS,), "saved P1 phase or clipping diagnostics are incomplete")
    return {"coefficients": coefficients, "raw_cmos": raw,
            "labels": labels, "proposal_ids": proposal_ids,
            "targets": assets["generation"]["saved_prior_targets"],
            "commands": commands,
            "public_phase": projection_metadata["public_phase"],
            "clipped_fraction": projection_metadata["clipped_fraction"]}


def _nearest_rows(generated: np.ndarray, calibration: np.ndarray,
                  reference: np.ndarray, scale: np.ndarray,
                  reference_rows: np.ndarray) -> dict[str, Any]:
    gen = support._nearest_numpy(generated, reference, scale, reference_rows)
    cal = support._nearest_numpy(calibration, reference, scale, reference_rows)
    return {"generated": gen, "calibration": cal}


def _bootstrap_distribution(baseline: Mapping[int, np.ndarray],
                            candidate: Mapping[int, np.ndarray],
                            calibration: Mapping[int, np.ndarray]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    g0 = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
    g1 = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
    delta = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
    for label in range(CLASS_COUNT):
        left, right, cal = baseline[label], candidate[label], calibration[label]
        _require(left.shape == right.shape == (ROWS_PER_CLASS,)
                 and cal.shape == (CALIBRATION_ROWS_PER_CLASS,),
                 "paired distribution bootstrap requires 200 generated and 100 calibration rows per class")
        indices = rng.integers(0, ROWS_PER_CLASS, size=(BOOTSTRAP_COUNT, ROWS_PER_CLASS))
        calibration_indices = rng.integers(
            0, CALIBRATION_ROWS_PER_CLASS,
            size=(BOOTSTRAP_COUNT, CALIBRATION_ROWS_PER_CLASS),
        )
        calibration_mean = cal[calibration_indices].mean(axis=1)
        a = left[indices].mean(axis=1) - calibration_mean
        b = right[indices].mean(axis=1) - calibration_mean
        g0 += a / CLASS_COUNT
        g1 += b / CLASS_COUNT
        delta += (b - a) / CLASS_COUNT
    def interval(values: np.ndarray) -> list[float]:
        lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
        return [float(lower), float(upper)]
    return {"resamples": BOOTSTRAP_COUNT, "seed": BOOTSTRAP_SEED,
            "baseline_G_ci95": interval(g0), "candidate_G_ci95": interval(g1),
            "delta_G_ci95": interval(delta),
            "scope": "conditional on the fixed terminal TRAIN reference and calibration rows"}


def _independent_nearest_numpy(query: np.ndarray, reference: np.ndarray,
                               scale: np.ndarray) -> dict[str, np.ndarray]:
    """Independent review implementation of the scale-whitened complex field distance."""
    modal_scale = np.asarray(scale, dtype=np.float64)
    q = np.asarray(query, dtype=np.complex128) / np.sqrt(modal_scale)[None, :]
    r = np.asarray(reference, dtype=np.complex128) / np.sqrt(modal_scale)[None, :]
    q_norm = np.sqrt(np.sum(np.abs(q) ** 2, axis=1))
    r_norm = np.sqrt(np.sum(np.abs(r) ** 2, axis=1))
    _require(np.isfinite(q_norm).all() and np.isfinite(r_norm).all()
             and (q_norm > 0.0).all() and (r_norm > 0.0).all(),
             "independent NumPy review found a zero or nonfinite field")
    q = q / q_norm[:, None]
    r = r / r_norm[:, None]
    overlap = np.conjugate(q) @ r.T
    complex_sq = np.clip(2.0 - 2.0 * np.abs(overlap), 0.0, None)
    amplitude_overlap = np.abs(q) @ np.abs(r).T
    amplitude_sq = np.clip(2.0 - 2.0 * amplitude_overlap, 0.0, None)
    nearest = np.argmin(complex_sq, axis=1)
    rows = np.arange(len(q))
    selected_complex = complex_sq[rows, nearest]
    selected_amplitude = amplitude_sq[rows, nearest]
    return {
        "distance": np.sqrt(selected_complex),
        "amplitude_distance": np.sqrt(selected_amplitude),
        "relative_phase_penalty": np.sqrt(np.clip(
            selected_complex - selected_amplitude, 0.0, None
        )),
        "nearest_reference_index": nearest,
    }


def _independent_bootstrap_distribution(
        baseline: Mapping[int, np.ndarray], candidate: Mapping[int, np.ndarray],
        calibration: Mapping[int, np.ndarray]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    g0 = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
    g1 = np.zeros(BOOTSTRAP_COUNT, dtype=np.float64)
    for label in range(CLASS_COUNT):
        base = np.asarray(baseline[label], dtype=np.float64)
        aligned = np.asarray(candidate[label], dtype=np.float64)
        cal = np.asarray(calibration[label], dtype=np.float64)
        _require(base.shape == aligned.shape == (ROWS_PER_CLASS,)
                 and cal.shape == (CALIBRATION_ROWS_PER_CLASS,),
                 "independent bootstrap requires 200 generated and 100 calibration rows per class")
        generated_rows = rng.integers(0, ROWS_PER_CLASS,
                                      size=(BOOTSTRAP_COUNT, ROWS_PER_CLASS))
        calibration_rows = rng.integers(0, CALIBRATION_ROWS_PER_CLASS,
                                        size=(BOOTSTRAP_COUNT, CALIBRATION_ROWS_PER_CLASS))
        cal_mean = cal[calibration_rows].mean(axis=1)
        g0 += (base[generated_rows].mean(axis=1) - cal_mean) / CLASS_COUNT
        g1 += (aligned[generated_rows].mean(axis=1) - cal_mean) / CLASS_COUNT
    def interval(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(values, [0.025, 0.975], method="linear")]
    return {"resamples": BOOTSTRAP_COUNT, "seed": BOOTSTRAP_SEED,
            "baseline_G_ci95": interval(g0), "candidate_G_ci95": interval(g1),
            "delta_G_ci95": interval(g1 - g0),
            "scope": "conditional on the fixed terminal TRAIN reference and calibration rows"}


def _split_terminal_train_rows(class_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(class_positions, dtype=np.int64)
    _require(positions.shape == (ROWS_PER_CLASS,)
             and len(np.unique(positions)) == ROWS_PER_CLASS,
             "terminal TRAIN split requires 200 unique rows for one class", ValueError)
    reference, calibration = positions[::2], positions[1::2]
    _require(len(reference) == len(calibration) == 100
             and not np.intersect1d(reference, calibration).size
             and np.array_equal(np.sort(np.concatenate((reference, calibration))),
                                np.sort(positions)),
             "terminal TRAIN reference and calibration rows overlap or omit source rows")
    return reference, calibration


def _semantic_metrics(assets: Mapping[str, Any], output: Mapping[str, torch.Tensor]
                      ) -> tuple[dict[str, Any], torch.Tensor]:
    runtime = assets["runtime"]
    inputs = runtime["input_bundle"]
    labels = output["labels"].long().cpu()
    baseline_raw = assets["baseline_outputs"]["raw_cmos"].detach().cpu().double()
    baseline_readouts = recent.pilot._readout_for_evaluator(
        baseline_raw, runtime["camera_area"], float(runtime["camera_gain"])
    ).cpu().float()
    candidate_readouts = recent.pilot._readout_for_evaluator(
        output["raw_cmos"], runtime["camera_area"], float(runtime["camera_gain"])
    ).cpu().float()
    _require(tuple(candidate_readouts.shape) == (POPULATION_ROWS, 1, 28, 28)
             and torch.isfinite(candidate_readouts).all().item()
             and float(candidate_readouts.min()) >= 0.0
             and float(candidate_readouts.max()) <= 1.0,
             "P1 raw CMOS did not reconstruct the fixed evaluator display view")
    baseline_summary = recent.frame._summary_from_readouts(
        inputs, assets["population"]["cohort"], baseline_readouts
    )
    candidate_summary = recent.frame._summary_from_readouts(
        inputs, assets["population"]["cohort"], candidate_readouts
    )
    evaluator_results: dict[str, Any] = {}
    for evaluator in ("ED", "EC"):
        base_score = baseline_summary["conditions"]["direct_decoder"]["evaluators"][evaluator]
        candidate_score = candidate_summary["conditions"]["direct_decoder"]["evaluators"][evaluator]
        evaluator_results[evaluator] = {
            "baseline_macro_recall": float(base_score["macro_recall"]),
            "candidate_macro_recall": float(candidate_score["macro_recall"]),
            "macro_recall_delta": float(candidate_score["macro_recall"] - base_score["macro_recall"]),
            "baseline_per_class_recall": base_score["per_class_recall"],
            "candidate_per_class_recall": candidate_score["per_class_recall"],
            "baseline_predictions": base_score["predicted_labels"],
            "candidate_predictions": candidate_score["predicted_labels"],
        }
    diversity = attribution._diversity_ratios(baseline_summary, candidate_summary)
    decision = _semantic_decision(labels, evaluator_results, diversity)
    summary = {
        "task_id": TASK_ID, "baseline": "P0_E1G1",
        "candidate": "P1_E1G1", "population_rows": POPULATION_ROWS,
        "population_role": "consumed_train_diagnostic",
        "evaluators": evaluator_results,
        "E_D_paired_macro_recall": decision["E_D_paired_macro_recall"],
        "E_C_macro_recall_delta": decision["E_C_macro_recall_delta"],
        "per_class_diversity_retention": diversity,
        "semantic_rule": decision["semantic_rule"],
        "decision": decision["decision"],
        "baseline_display_max_abs_difference_from_saved": float(
            (baseline_readouts - assets["baseline_outputs"]["readouts"].float()).abs().max()
        ),
        "candidate_display_conversion": "fixed raw CMOS -> W/pixel -> gain -> clamp -> adaptive average pool 28",
        "official_test_reads": 0, "human_readability": "not_measured",
    }
    return summary, candidate_readouts


def _semantic_decision(labels: torch.Tensor, evaluator_results: Mapping[str, Mapping[str, Any]],
                       diversity: Mapping[str, float]) -> dict[str, Any]:
    _require(tuple(labels.shape) == (POPULATION_ROWS,)
             and torch.equal(torch.bincount(labels.long(), minlength=CLASS_COUNT),
                             torch.full((CLASS_COUNT,), ROWS_PER_CLASS)),
             "semantic decision requires the complete balanced saved population")
    ed_delta = attribution._paired_bootstrap(
        labels,
        evaluator_results["ED"]["baseline_predictions"],
        evaluator_results["ED"]["candidate_predictions"],
        seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_COUNT,
    )
    ec_base = np.asarray(evaluator_results["EC"]["baseline_predictions"], dtype=np.int64)
    ec_candidate = np.asarray(evaluator_results["EC"]["candidate_predictions"], dtype=np.int64)
    label_array = labels.cpu().numpy().astype(np.int64)
    _require(ec_base.shape == ec_candidate.shape == label_array.shape,
             "E_C predictions do not cover the complete saved population")
    ec_delta = float(np.mean([
        np.mean(ec_candidate[label_array == label] == label)
        - np.mean(ec_base[label_array == label] == label)
        for label in range(CLASS_COUNT)
    ]))
    threshold = {"ed_paired_ci95_lower_gt_zero":
                 ed_delta["paired_class_stratified_ci95"][0] > 0.0,
                 "ec_drop_at_most_2pp": ec_delta >= -0.02,
                 "all_classes_diversity_retention_at_least_80pct":
                 all(value >= 0.8 for value in diversity.values())}
    decision = "SUPPORTED_ON_CONSUMED_DIAGNOSTIC_POPULATION" if all(threshold.values()) \
        else "SEMANTIC_BENEFIT_NOT_SUPPORTED"
    return {
        "E_D_paired_macro_recall": ed_delta,
        "E_C_macro_recall_delta": ec_delta,
        "semantic_rule": threshold,
        "decision": decision,
    }


def _distribution_metrics(assets: Mapping[str, Any], output: Mapping[str, torch.Tensor]
                          ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scale = assets["runtime"]["scale"].detach().cpu().double().numpy()
    terminal = assets["train_endpoint"]
    train_labels = terminal["labels"].long().cpu().numpy()
    train_rows = terminal["source_rows"].long().cpu().numpy()
    train_coefficients = terminal["coefficients"].detach().cpu().numpy().astype(np.complex128)
    baseline_coeff = assets["generation"]["actual_E1G1_coefficients"].detach().cpu().numpy()
    candidate_coeff = output["coefficients"].detach().cpu().numpy()
    targets = output["targets"].detach().cpu().numpy()
    label_values = output["labels"].cpu().numpy()
    generated_dist: dict[str, dict[int, np.ndarray]] = {"baseline": {}, "candidate": {}, "target": {}}
    calibration_dist: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        class_train_pos = np.flatnonzero(train_labels == label)
        _require(len(class_train_pos) == ROWS_PER_CLASS,
                 f"terminal TRAIN reference has the wrong row count for class {label}")
        refs_pos, calibration_pos = _split_terminal_train_rows(class_train_pos)
        ref, calibration = train_coefficients[refs_pos], train_coefficients[calibration_pos]
        generation_pos = np.flatnonzero(label_values == label)
        _require(len(generation_pos) == ROWS_PER_CLASS,
                 f"generated diagnostic population has the wrong row count for class {label}")
        base = baseline_coeff[generation_pos]
        cand = candidate_coeff[generation_pos]
        target = targets[generation_pos]
        ref_rows = train_rows[refs_pos]
        base_nearest = _nearest_rows(base, calibration, ref, scale, ref_rows)
        cand_nearest = _nearest_rows(cand, calibration, ref, scale, ref_rows)
        target_nearest = support._nearest_numpy(target, ref, scale, ref_rows)
        generated_dist["baseline"][label] = base_nearest["generated"]["distance"]
        generated_dist["candidate"][label] = cand_nearest["generated"]["distance"]
        generated_dist["target"][label] = target_nearest["distance"]
        calibration_dist[label] = base_nearest["calibration"]["distance"]
        g0 = float(generated_dist["baseline"][label].mean() - calibration_dist[label].mean())
        g1 = float(generated_dist["candidate"][label].mean() - calibration_dist[label].mean())
        per_class[str(label)] = {
            "G_P0": g0, "G_P1": g1, "delta_G": g1 - g0,
            "baseline_nearest_mean": float(generated_dist["baseline"][label].mean()),
            "candidate_nearest_mean": float(generated_dist["candidate"][label].mean()),
            "calibration_nearest_mean": float(calibration_dist[label].mean()),
            "target_nearest_mean": float(generated_dist["target"][label].mean()),
            "baseline_target_phase_error_mean": float(np.mean([
                recent.projected._phase_error(a, b) for a, b in zip(base, target, strict=True)
            ])),
            "candidate_target_phase_error_mean": float(np.mean([
                recent.projected._phase_error(a, b) for a, b in zip(cand, target, strict=True)
            ])),
            "baseline_power_mean": float(np.mean(np.sum(np.abs(base) ** 2, axis=1))),
            "candidate_power_mean": float(np.mean(np.sum(np.abs(cand) ** 2, axis=1))),
        }
        for local, population_index in enumerate(generation_pos.tolist()):
            rows.append({
                "population_index": population_index, "label": label,
                "proposal_id": int(output["proposal_ids"][population_index]),
                "train_reference_source_rows": ";".join(str(int(value)) for value in ref_rows),
                "baseline_nearest_distance": float(base_nearest["generated"]["distance"][local]),
                "candidate_nearest_distance": float(cand_nearest["generated"]["distance"][local]),
                "target_nearest_distance": float(target_nearest["distance"][local]),
                "baseline_amplitude_distance": float(base_nearest["generated"]["amplitude_distance"][local]),
                "candidate_amplitude_distance": float(cand_nearest["generated"]["amplitude_distance"][local]),
                "baseline_phase_penalty": float(base_nearest["generated"]["relative_phase_penalty"][local]),
                "candidate_phase_penalty": float(cand_nearest["generated"]["relative_phase_penalty"][local]),
                "baseline_target_phase_error": recent.projected._phase_error(base[local], target[local]),
                "candidate_target_phase_error": recent.projected._phase_error(cand[local], target[local]),
                "candidate_public_phase_rad": float(output["public_phase"][population_index]),
                "candidate_clipped_fraction": float(output["clipped_fraction"][population_index]),
                "baseline_total_modal_power": float(np.sum(np.abs(base[local]) ** 2)),
                "candidate_total_modal_power": float(np.sum(np.abs(cand[local]) ** 2)),
                "nearest_train_source_row_baseline": int(base_nearest["generated"]["nearest_reference_row"][local]),
                "nearest_train_source_row_candidate": int(cand_nearest["generated"]["nearest_reference_row"][local]),
            })
    g0 = float(np.mean([per_class[str(label)]["G_P0"] for label in range(CLASS_COUNT)]))
    g1 = float(np.mean([per_class[str(label)]["G_P1"] for label in range(CLASS_COUNT)]))
    bootstrap = _bootstrap_distribution(generated_dist["baseline"], generated_dist["candidate"],
                                        calibration_dist)
    return {"baseline_G": g0, "candidate_G": g1, "delta_G": g1 - g0,
            "bootstrap": bootstrap,
            "terminal_train_reference_per_class": 100,
            "terminal_train_calibration_per_class": 100,
            "reference_row_policy": "even within-class offsets among the saved 200 TRAIN rows",
            "calibration_row_policy": "odd within-class offsets among the saved 200 TRAIN rows",
            "per_class": per_class,
            "interpretation": "diagnostic association; not a causal or semantic decision"}, rows


def _verify_distribution_independently(
        assets: Mapping[str, Any], output: Mapping[str, torch.Tensor],
        production: Mapping[str, Any], production_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute saved OLS distances and intervals without the production nearest-row helper."""
    scale = assets["runtime"]["scale"].detach().cpu().double().numpy()
    terminal = assets["train_endpoint"]
    train_labels = terminal["labels"].long().cpu().numpy()
    train_rows = terminal["source_rows"].long().cpu().numpy()
    train_coeff = terminal["coefficients"].detach().cpu().numpy().astype(np.complex128)
    base_coeff = assets["generation"]["actual_E1G1_coefficients"].detach().cpu().numpy()
    candidate_coeff = output["coefficients"].detach().cpu().numpy()
    targets = output["targets"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    rows_by_index = {int(row["population_index"]): row for row in production_rows}
    _require(len(rows_by_index) == POPULATION_ROWS,
             "independent OLS review received duplicate or missing population rows")
    base_distance: dict[int, np.ndarray] = {}
    candidate_distance: dict[int, np.ndarray] = {}
    calibration_distance: dict[int, np.ndarray] = {}
    max_sample_error = 0.0
    per_class: dict[str, dict[str, float]] = {}
    for label in range(CLASS_COUNT):
        train_positions = np.flatnonzero(train_labels == label)
        _require(len(train_positions) == ROWS_PER_CLASS,
                 f"independent OLS review found incomplete TRAIN rows for class {label}")
        reference_positions, calibration_positions = _split_terminal_train_rows(train_positions)
        reference = train_coeff[reference_positions]
        calibration = train_coeff[calibration_positions]
        reference_rows = train_rows[reference_positions]
        generated_positions = np.flatnonzero(labels == label)
        _require(len(generated_positions) == ROWS_PER_CLASS,
                 f"independent OLS review found incomplete generated rows for class {label}")
        base_result = _independent_nearest_numpy(
            base_coeff[generated_positions], reference, scale,
        )
        candidate_result = _independent_nearest_numpy(
            candidate_coeff[generated_positions], reference, scale,
        )
        target_result = _independent_nearest_numpy(targets[generated_positions], reference, scale)
        calibration_result = _independent_nearest_numpy(calibration, reference, scale)
        base_distance[label] = base_result["distance"]
        candidate_distance[label] = candidate_result["distance"]
        calibration_distance[label] = calibration_result["distance"]
        g0 = float(base_distance[label].mean() - calibration_distance[label].mean())
        g1 = float(candidate_distance[label].mean() - calibration_distance[label].mean())
        per_class[str(label)] = {"G_P0": g0, "G_P1": g1, "delta_G": g1 - g0}
        for local, population_index in enumerate(generated_positions.tolist()):
            row = rows_by_index[population_index]
            expected = {
                "baseline_nearest_distance": base_result["distance"][local],
                "candidate_nearest_distance": candidate_result["distance"][local],
                "target_nearest_distance": target_result["distance"][local],
                "baseline_amplitude_distance": base_result["amplitude_distance"][local],
                "candidate_amplitude_distance": candidate_result["amplitude_distance"][local],
                "baseline_phase_penalty": base_result["relative_phase_penalty"][local],
                "candidate_phase_penalty": candidate_result["relative_phase_penalty"][local],
                "nearest_train_source_row_baseline": reference_rows[
                    base_result["nearest_reference_index"][local]
                ],
                "nearest_train_source_row_candidate": reference_rows[
                    candidate_result["nearest_reference_index"][local]
                ],
            }
            for key, value in expected.items():
                error = abs(float(row[key]) - float(value))
                max_sample_error = max(max_sample_error, error)
    _require(max_sample_error <= 1.0e-12,
             f"independent NumPy OLS distances differ from saved rows: {max_sample_error:.3g}")
    baseline_g = float(np.mean([per_class[str(label)]["G_P0"] for label in range(CLASS_COUNT)]))
    candidate_g = float(np.mean([per_class[str(label)]["G_P1"] for label in range(CLASS_COUNT)]))
    bootstrap = _independent_bootstrap_distribution(
        base_distance, candidate_distance, calibration_distance,
    )
    for key, value in (("baseline_G", baseline_g), ("candidate_G", candidate_g),
                       ("delta_G", candidate_g - baseline_g)):
        _require(math.isclose(float(production[key]), value, rel_tol=1.0e-12, abs_tol=1.0e-12),
                 f"independent NumPy OLS review disagrees on {key}")
    for label in range(CLASS_COUNT):
        for key, value in per_class[str(label)].items():
            _require(math.isclose(float(production["per_class"][str(label)][key]), value,
                                  rel_tol=1.0e-12, abs_tol=1.0e-12),
                     f"independent NumPy OLS review disagrees on class {label} {key}")
    for key in ("baseline_G_ci95", "candidate_G_ci95", "delta_G_ci95"):
        _require(all(math.isclose(float(a), float(b), rel_tol=1.0e-12, abs_tol=1.0e-12)
                     for a, b in zip(production["bootstrap"][key], bootstrap[key], strict=True)),
                 f"independent NumPy OLS review disagrees on bootstrap interval {key}")
    return {"passed": True, "method": "direct_numpy_complex_inner_products_and_independent_bootstrap",
            "max_sample_distance_error": max_sample_error,
            "baseline_G": baseline_g, "candidate_G": candidate_g,
            "delta_G": candidate_g - baseline_g,
            "bootstrap": bootstrap}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "refusing to write an empty per-sample result table")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(output_dir: Path, labels: torch.Tensor,
                   baseline: torch.Tensor, candidate: torch.Tensor,
                   semantic: Mapping[str, Any]) -> None:
    canvas = Image.new("RGB", (900, 1_040), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "Fixed display views: P0 + E1G1 versus P1 + E1G1", fill="black")
    draw.text((18, 38), "class", fill="black")
    headers = [(100, "P0 row 1"), (230, "P1 row 1"), (430, "P0 row 2"), (560, "P1 row 2")]
    for x, title in headers:
        draw.text((x, 38), title, fill="black")
    for label in range(CLASS_COUNT):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()[:2].tolist()
        y = 68 + label * 94
        draw.text((24, y + 28), str(label), fill="black")
        for offset, row in enumerate(rows):
            for panel, images in ((0, baseline), (1, candidate)):
                x = 100 + offset * 330 + panel * 130
                image = Image.fromarray((images[row, 0].numpy().clip(0, 1) * 255).round().astype(np.uint8))
                canvas.paste(image.resize((72, 72), Image.Resampling.NEAREST), (x, y))
                draw.text((x + 2, y + 73), f"i={row}", fill="black")
    canvas.save(output_dir / "fixed_display_example_pairs.png")

    width, height = 1_100, 620
    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((18, 14), "Per-class evaluator recall: P0 + E1G1 / P1 + E1G1", fill="black")
    for panel_index, evaluator in enumerate(("ED", "EC")):
        top = 54 + panel_index * 270
        draw.text((18, top), evaluator, fill="black")
        left, right = 110, 1_060
        base_values = semantic["evaluators"][evaluator]["baseline_per_class_recall"]
        candidate_values = semantic["evaluators"][evaluator]["candidate_per_class_recall"]
        for label, (a, b) in enumerate(zip(base_values, candidate_values, strict=True)):
            center = left + label * 92 + 38
            draw.text((center - 4, top + 210), str(label), fill="black")
            for x, value, color in ((center - 20, a, (45, 92, 165)),
                                    (center + 4, b, (222, 125, 40))):
                bar_height = int(float(value) * 175)
                draw.rectangle((x, top + 202 - bar_height, x + 18, top + 202), fill=color)
                draw.text((x - 5, top + 204), f"{100*float(value):.0f}", fill="black")
        draw.line((left, top + 202, right, top + 202), fill=(80, 80, 80), width=1)
    chart.save(output_dir / "per_class_semantic_comparison.png")


def _consume_and_write(config: Mapping[str, Any], artifact_dir: Path,
                       assets: Mapping[str, Any], projector_bundle: Mapping[str, Any]
                       ) -> dict[str, Any]:
    # Rebuild the ordered command matrix from the small control group and fixed row shards.
    command_matrix = torch.empty((POPULATION_ROWS, 1, *IMAGE_SHAPE), dtype=torch.float32)
    controls = _load_torch(artifact_dir / "controls_2048.pt")
    command_matrix[CONTROL_ROWS] = controls["actual_slm_commands_forwarded"]
    other = torch.tensor([i for i in range(POPULATION_ROWS)
                          if i not in set(CONTROL_ROWS.tolist())], dtype=torch.long)
    for start in range(0, len(other), CHUNK_SIZE):
        indices = other[start:start + CHUNK_SIZE]
        shard = _load_torch(artifact_dir / _safe_shard_name(indices.tolist()))
        command_matrix[indices] = shard["actual_slm_commands_forwarded"]
    output = _load_candidate_output(artifact_dir, assets, command_matrix)
    baseline_labels = assets["generation"]["labels"].long().cpu()
    _require(torch.equal(output["labels"], baseline_labels)
             and torch.equal(output["proposal_ids"], assets["generation"]["proposal_ids"])
             and torch.equal(command_matrix, command_matrix.clamp(0.0, 1.0)),
             "complete measured population changed labels, proposals, or legal commands")
    semantic, readouts = _semantic_metrics(assets, output)
    distribution, per_sample = _distribution_metrics(assets, output)
    summary = {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "decision": semantic["decision"], "semantic": semantic,
        "distribution_diagnostic": distribution,
        "measurement": {
            "baseline_2048_forwards_new": 20,
            "candidate_2048_forwards_new": 2_000,
            "candidate_4096_reference_forwards_new": 20,
            "response_2048_forward": 1,
            "response_2048_scalar_reverse_derivatives": 200,
            "response_finite_difference_2048_forwards": 2,
            "all_successful_2048_forwards_new": 2_023,
            "all_successful_4096_forwards_new": 20,
            "population_rows": POPULATION_ROWS,
            "optimizer_instances": 0, "optimizer_updates": 0,
            "prior_refits": 0, "prior_samples": 0, "official_test_reads": 0,
        },
        "source_reuse": {
            "training_run_id": config["sources"]["expected_training_run_id"],
            "attempt04_corrected_train_outputs": 2_000,
            "attempt05_baseline_commands_and_raw_cmos": 2_000,
            "population_seed_base": config["sources"]["expected_population_seed_base"],
            "mapper_run_id": config["sources"]["expected_mapper_run_id"],
        },
        "preflight_record": (
            "production_preflight_revalidation.json"
            if (artifact_dir / "production_preflight_revalidation.json").is_file()
            else "production_preflight.json"
        ),
    }
    _write_json(artifact_dir / "final_decision.json", summary, exclusive=True)
    _write_csv(artifact_dir / "per_sample_results.csv", per_sample)
    _save_torch(artifact_dir / "candidate_readouts.pt", readouts)
    _write_figures(artifact_dir, output["labels"],
                   recent.pilot._readout_for_evaluator(
                       assets["baseline_outputs"]["raw_cmos"],
                       assets["runtime"]["camera_area"], float(assets["runtime"]["camera_gain"]),
                   ).cpu().float(), readouts, semantic)
    _write_json(artifact_dir / "measurement_status.json", {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "production_rows": POPULATION_ROWS, "cross_grid_rows": 20,
        "decision": semantic["decision"], "optimizer_updates": 0,
    }, exclusive=True)
    return summary


def _verify_saved(config: Mapping[str, Any], artifact_dir: Path, review_dir: Path,
                  device: torch.device) -> None:
    assets = _load_assets(config, device)
    saved_config = yaml.safe_load((artifact_dir / "run_config.yaml").read_text(encoding="utf-8"))
    _require(saved_config == config, "saved AutoDL config does not match the registered Windows config")
    projector = _load_torch(artifact_dir / "response_projector.pt")
    _validate_projector_bundle(projector, assets, device)
    response = projector["response"].to(device)
    preflight_path = (artifact_dir / "production_preflight_revalidation.json"
                      if (artifact_dir / "production_preflight_revalidation.json").is_file()
                      else artifact_dir / "production_preflight.json")
    preflight = _read_json(preflight_path)
    _require(preflight.get("status") == "PASSED"
             and preflight.get("wrong_response_row_binding_rejected") is True,
             "saved terminal response preflight or wrong-binding negative control is missing")
    _require(float(preflight.get("r1_vs_saved_e1g1_max_relative_error", math.inf))
             <= RESPONSE_TOLERANCE
             and float(preflight.get("cross_grid", {}).get(
                 "coefficient_max_relative_error", math.inf
             )) <= CROSS_GRID_TOLERANCE
             and float(preflight.get("cross_grid", {}).get(
                 "raw_cmos_max_relative_error", math.inf
             )) <= CROSS_GRID_TOLERANCE,
             "saved preflight response or cross-grid values fail corrected complex checks")
    baseline_controls = CONTROL_ROWS
    baseline_prediction = _predict_coefficients(
        assets["baseline_commands"].index_select(0, baseline_controls), response,
    ).detach().cpu()
    baseline_actual = assets["baseline_outputs"]["coefficients"].index_select(
        0, baseline_controls,
    )
    baseline_response_errors = [
        _relative_l2(baseline_prediction[index], baseline_actual[index])
        for index in range(len(baseline_controls))
    ]
    _require(max(baseline_response_errors) <= RESPONSE_TOLERANCE,
             "Windows consumer rejects R1 predictions of saved P0 optical coefficients")
    manifest = _load_torch(artifact_dir / "population_manifest.pt")
    generation = assets["generation"]
    _require(manifest.get("task_id") == TASK_ID
             and manifest.get("population_role") == "consumed_train_diagnostic"
             and manifest.get("population_rows") == POPULATION_ROWS
             and manifest.get("candidate_projection") == "R1_terminal_E1G1_plus_P1"
             and manifest.get("population_seed_base") == config["sources"]["expected_population_seed_base"]
             and torch.equal(manifest.get("labels"), generation["labels"])
             and torch.equal(manifest.get("proposal_ids"), generation["proposal_ids"])
             and torch.equal(manifest.get("prior_targets"), generation["saved_prior_targets"]),
             "saved P1 manifest is incomplete or misbound")
    candidate_commands = torch.empty((POPULATION_ROWS, 1, *IMAGE_SHAPE), dtype=torch.float32)
    controls = _load_torch(artifact_dir / "controls_2048.pt")
    candidate_commands[CONTROL_ROWS] = controls["actual_slm_commands_forwarded"]
    noncontrols = torch.tensor([i for i in range(POPULATION_ROWS)
                                if i not in set(CONTROL_ROWS.tolist())], dtype=torch.long)
    for start in range(0, len(noncontrols), CHUNK_SIZE):
        indices = noncontrols[start:start + CHUNK_SIZE]
        shard = _load_torch(artifact_dir / _safe_shard_name(indices.tolist()))
        _validate_shard(shard, indices, candidate_commands, assets, compare_commands=False)
        candidate_commands[indices] = shard["actual_slm_commands_forwarded"]

    cross_grid = _load_torch(artifact_dir / "cross_grid_controls.pt")
    _require(torch.equal(cross_grid.get("population_indices"), CONTROL_ROWS)
             and torch.equal(cross_grid.get("labels"), generation["labels"].index_select(0, CONTROL_ROWS))
             and torch.equal(cross_grid.get("proposal_ids"), generation["proposal_ids"].index_select(
                 0, CONTROL_ROWS
             ))
             and torch.equal(cross_grid.get("actual_slm_commands_forwarded"),
                             candidate_commands.index_select(0, CONTROL_ROWS))
             and torch.equal(cross_grid.get("coefficients_2048"), controls.get("coefficients"))
             and torch.equal(cross_grid.get("raw_cmos_2048"), controls.get("raw_cmos")),
             "saved cross-grid controls do not bind to the P1 population and 2048 commands")
    cross_coeff_errors = [
        _relative_l2(cross_grid["coefficients_2048"][index],
                     cross_grid["coefficients_4096"][index])
        for index in range(20)
    ]
    cross_cmos_errors = [
        _relative_l2(cross_grid["raw_cmos_2048"][index], cross_grid["raw_cmos_4096"][index])
        for index in range(20)
    ]
    _require(max(cross_coeff_errors) <= CROSS_GRID_TOLERANCE
             and max(cross_cmos_errors) <= CROSS_GRID_TOLERANCE
             and math.isclose(max(cross_coeff_errors), float(preflight["cross_grid"][
                 "coefficient_max_relative_error"
             ]), rel_tol=1.0e-12, abs_tol=1.0e-12)
             and math.isclose(max(cross_cmos_errors), float(preflight["cross_grid"][
                 "raw_cmos_max_relative_error"
             ]), rel_tol=1.0e-12, abs_tol=1.0e-12),
             "Windows consumer rejects saved 2048/4096 P1 positive controls")

    matrix = projector["normalized_real_matrix"].to(device)
    inverse = projector["right_inverse"].to(device)
    p0_commands, _ = recent._make_commands(assets["mapper"], generation["saved_prior_targets"],
                                            assets["runtime"])
    p0_command_error = float((p0_commands - assets["baseline_commands"]).abs().max())
    _require(p0_command_error <= COMMAND_TOLERANCE, "Windows P0 command replay failed")
    p1_commands, _, _ = _normalised_features_and_commands(
        assets["mapper"], generation["saved_prior_targets"], matrix, inverse,
        assets["runtime"]["scale"], device,
    )
    p1_command_error = float((p1_commands - candidate_commands).abs().max())
    _require(p1_command_error <= COMMAND_TOLERANCE,
             f"Windows independent P1 command replay failed: {p1_command_error:.3g}")
    numpy_matrix = recent.projected._normalized_response(
        projector["response"].to(device), assets["runtime"]["scale"]
    ).detach().cpu().numpy()
    matrix_error = float(np.max(np.abs(matrix.detach().cpu().numpy() - numpy_matrix)))
    _require(matrix_error <= 1.0e-14,
             f"Windows response matrix reconstruction error {matrix_error:.3g} exceeds 1e-14")
    numpy_inverse = projector["right_inverse"].detach().cpu().numpy()
    numpy_identity_error = float(np.max(np.abs(numpy_matrix @ numpy_inverse - np.eye(200))))
    _require(numpy_identity_error <= IDENTITY_TOLERANCE,
             "Windows NumPy right-inverse consumer rejects saved P1")
    # The independent NumPy correction checks real/imag ordering, projection and clipping.
    numpy_projection = recent.projected._matrix_checks(
        projector["response"].to(device), matrix, inverse,
        assets["runtime"]["scale"], device,
    )

    output = _load_candidate_output(artifact_dir, assets, candidate_commands)
    semantic, candidate_readouts = _semantic_metrics(assets, output)
    distribution, rows = _distribution_metrics(assets, output)
    independent_distribution = _verify_distribution_independently(
        assets, output, distribution, rows,
    )
    saved_decision = _read_json(artifact_dir / "final_decision.json")
    _require(saved_decision.get("decision") == semantic["decision"]
             and saved_decision.get("semantic", {}).get("E_D_paired_macro_recall")
             == semantic["E_D_paired_macro_recall"]
             and saved_decision.get("semantic", {}).get("semantic_rule") == semantic["semantic_rule"],
             "Windows raw-CMOS consumer did not reproduce the semantic decision")
    _require(math.isclose(saved_decision["distribution_diagnostic"]["baseline_G"],
                          distribution["baseline_G"], rel_tol=1.0e-12, abs_tol=1.0e-12)
             and math.isclose(saved_decision["distribution_diagnostic"]["candidate_G"],
                              distribution["candidate_G"], rel_tol=1.0e-12, abs_tol=1.0e-12),
             "Windows NumPy consumer did not reproduce distribution diagnostics")
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(review_dir / "independent_review.json", {
        "task_id": TASK_ID, "passed": True, "decision": semantic["decision"],
        "P0_command_max_abs_error": p0_command_error,
        "P1_command_max_abs_error": p1_command_error,
        "preflight_record": preflight_path.name,
        "P1_cross_grid_coefficient_max_relative_error": max(cross_coeff_errors),
        "P1_cross_grid_raw_cmos_max_relative_error": max(cross_cmos_errors),
        "R1_vs_saved_E1G1_max_relative_error": max(baseline_response_errors),
        "R1_saved_E1G1_row_relative_errors": baseline_response_errors,
        "numpy_right_inverse_max_abs_error": numpy_identity_error,
        "normalized_response_matrix_recompute_max_abs_error": matrix_error,
        "numpy_projection_reference": numpy_projection,
        "independent_distribution_review": independent_distribution,
        "per_sample_rows": len(rows),
        "semantic_recomputed_from_raw_cmos": True,
        "distribution_recomputed_with_numpy": True,
        "optimizer_instances": 0, "optimizer_updates": 0,
        "official_test_reads": 0,
    }, exclusive=True)
    _write_json(review_dir / "recomputed_final_decision.json", {
        "task_id": TASK_ID, "decision": semantic["decision"],
        "semantic": semantic, "distribution_diagnostic": distribution,
    }, exclusive=True)
    _write_csv(review_dir / "per_sample_results.csv", rows)
    _write_figures(review_dir, output["labels"],
                   recent.pilot._readout_for_evaluator(
                       assets["baseline_outputs"]["raw_cmos"],
                       assets["runtime"]["camera_area"], float(assets["runtime"]["camera_gain"]),
                   ).cpu().float(), candidate_readouts, semantic)
    report = _closeout(semantic, distribution, assets)
    (review_dir / "closeout_report.md").write_text(report, encoding="utf-8")


def _closeout(semantic: Mapping[str, Any], distribution: Mapping[str, Any],
              assets: Mapping[str, Any]) -> str:
    ed = semantic["E_D_paired_macro_recall"]
    ec = semantic["E_C_macro_recall_delta"]
    lows = ed["paired_class_stratified_ci95"]
    diversity_min = min(semantic["per_class_diversity_retention"].values())
    lines = [
        "# LightGen 终点 Encoder 投影配准：完整光路语义对照",
        "", "## 结果与状态", "",
        f"有效数值测量完成，唯一语义决策为 `{semantic['decision']}`。",
        f"E_D 宏召回变化 {ed['point_macro_recall_delta']:.4%}，配对 95% 区间 "
        f"[{lows[0]:.4%}, {lows[1]:.4%}]；E_C 变化 {ec:.4%}；最小逐类多样性保留 "
        f"{diversity_min:.3%}。OLS 距离差 ΔG={distribution['delta_G']:.8f}。",
        "本轮参数更新 0、优化器实例 0、先验拟合/采样 0、官方 TEST 读取 0。",
        "", "## 科学认识", "",
        f"P0+E1G1 的 G={distribution['baseline_G']:.8f}；P1+E1G1 的 "
        f"G={distribution['candidate_G']:.8f}。该分布指标与目标复现误差只作机制诊断，"
        "不覆盖预先固定的 E_D、E_C 和类内多样性规则。",
        "", "## 目标缺口", "",
        "人口是已消费先验诊断样本；独立新人口、人眼可读性、模型采用和实物 "
        "SLM→衍射层→OLS→CMOS 链路仍未验证。",
        "", "## 工程完成与验证", "",
        f"同一 E1G1 检查点、mapper、先验、模态尺度和相机增益下测量 {POPULATION_ROWS} "
        "条 P1 命令；每类 2 条完成 2048/4096 正控。P0 和 P1 命令来源、目标、标签、proposal "
        "与输出逐行绑定；Windows 独立消费者重算命令、投影、raw CMOS 显示、评价器、距离及判断。",
        "", "## 决策影响与下一步", "",
        "结果仅支持或拒绝本人口上的数值语义候选。后续独立人口验证、训练和硬件工作须另行设计；"
        "本报告不授权任何后继实验。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--verify-saved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _read_config(config_path)
    artifact_dir = (args.artifact_dir or OUTPUT_DIR).resolve()
    if args.measure:
        _require(not args.review_dir, "--review-dir is only valid with --verify-saved", ValueError)
        expected = OUTPUT_DIR.resolve()
        _require(artifact_dir == expected,
                 f"measurement output must be the explicit attempt_01 path: {expected}", ValueError)
        _run_measure(config, artifact_dir, resume=args.resume)
    else:
        _require(not args.resume, "--resume is only valid with --measure", ValueError)
        review_dir = (args.review_dir or REVIEW_DIR).resolve()
        _require(artifact_dir == BACKUP_DIR.resolve() and review_dir == REVIEW_DIR.resolve(),
                 "saved verification requires the explicit attempt_01 Windows backup/review paths",
                 ValueError)
        _verify_saved(config, artifact_dir, review_dir,
                      torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(json.dumps({"task_id": TASK_ID, "passed": True,
                          "review_dir": str(review_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
