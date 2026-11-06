"""Continue the fixed B objective on the same 20 known-reachable G1 targets."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_FIT20_BOX_B_CONVERGENCE_V1"
SCHEMA = "lightgen_g1_fit20_box_b_convergence.v1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_FIT20_BOX_B_CONVERGENCE_20261027"
MODULE_PATH = f"scripts.research.lightgen.{Path(__file__).stem}"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_fit20_box_b_convergence_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261027_lightgen_g1_fit20_box_b_convergence_v1/runs"
PREFLIGHT_DIR = RUNS_DIR / "preflight_01"
ATTEMPT_DIR = RUNS_DIR / "attempt_01"
TARGET_COUNT = 20
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
GRIDS = (2048, 4096)
METRIC_KEYS = fit20.METRIC_KEYS
GATE_LIMIT = fit20.GATE_LIMIT
RECOVERY_TOLERANCE = fit20.RECOVERY_TOLERANCE
START_BRANCH_STEP = 500
ADDED_UPDATES = 1000
TERMINAL_BRANCH_STEP = START_BRANCH_STEP + ADDED_UPDATES
SCHEDULE_CYCLES = 750
SCHEDULE_SEED = 2026102781
RNG_SEED = 2026102780


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fit20._atomic_json(path, value)


def _atomic_torch(path: Path, value: Any) -> None:
    fit20._atomic_torch(path, value)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(), "B convergence uses its fixed configuration", ValueError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "B convergence configuration must be a mapping", ValueError)
    training, measurement, source = config["training"], config["measurement"], config["source"]
    _require(config.get("schema") == SCHEMA and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY
             and config.get("evidence_tier") == "diagnostic_only"
             and source.get("population_role") == "known_reachable_fit20"
             and source.get("source_arm") == "B" and source.get("source_branch_step") == 500
             and training.get("added_updates") == ADDED_UPDATES
             and training.get("branch_start_step") == START_BRANCH_STEP
             and training.get("branch_terminal_step") == TERMINAL_BRANCH_STEP
             and training.get("source_schedule_seed") == SCHEDULE_SEED
             and training.get("schedule_cycles") == SCHEDULE_CYCLES
             and training.get("box_lambda") == 5.0
             and training.get("learning_rate") == 1.0e-4
             and training.get("fixed_checkpoint_steps") == [0, 250, 500, 750, 1000]
             and training.get("observation_steps") == [0, 500, 1000]
             and measurement.get("grids") == [2048, 4096]
             and measurement.get("target_count") == TARGET_COUNT
             and measurement.get("relative_error_max") == GATE_LIMIT,
             "B convergence configuration changed its fixed identity, population, or budget", ValueError)
    return config


def _set_numerics() -> None:
    fit20._set_numerics()


def _run_identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    semantics = json.dumps({"task_id": TASK_ID, "authority": AUTHORITY, "config": config,
                            "source_fit20_task": fit20.TASK_ID,
                            "source_arm": "B", "source_branch_step": START_BRANCH_STEP,
                            "new_scientific_updates": ADDED_UPDATES,
                            "population_role": "known_reachable_fit20"},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:fit20_box_b_convergence",
        semantic_authority_id=AUTHORITY, data_role="KNOWN_REACHABLE_G1_FIT20_B_CONTINUATION",
        config_identity=dict(config),
        model_identity={"frozen_optics": "C_seed_2026101501_step_45000",
                        "source_mapper": "projected_fit20_B_step_500",
                        "mapper": "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid+fixed_projection",
                        "objective": "full_optics_phase_invariant_coefficient_loss+5x_preclip_box_loss"},
        optimizer_identity={"kind": "Adam", "learning_rate": 1.0e-4,
                            "betas": [0.9, 0.999], "epsilon": 1.0e-8,
                            "weight_decay": 0.0, "new_updates": ADDED_UPDATES,
                            "inherited_parameter_step": START_BRANCH_STEP},
        sampler_identity={"population": "same fixed 20 known-reachable targets",
                          "schedule_seed": SCHEDULE_SEED, "rng_seed": RNG_SEED,
                          "reused_prefix_cycles": 250, "continuation_cycles": 500},
        execution_source_semantics_id=semantics,
    )


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    old_config = fit20._load_config(_resolve(config["source"]["predecessor_fit20_config"]))
    inputs = fit20._load_inputs(old_config)
    return inputs


def _source_attempt(config: Mapping[str, Any], host: str) -> Path:
    key = "predecessor_cloud_attempt" if host == "cloud" else "predecessor_windows_backup"
    return _resolve(config["source"][key])


def _load_source(config: Mapping[str, Any], inputs: Mapping[str, Any], host: str) -> dict[str, Any]:
    source_dir = _source_attempt(config, host)
    _require(source_dir.is_dir(), f"explicit predecessor fit20 evidence is missing: {source_dir}",
             FileNotFoundError)
    identity = fit20._assert_attempt(source_dir)
    state = _json(source_dir / "task_state.json")
    decision = _json(source_dir / "final_decision.json")
    _require(state.get("state") == RUN_VALID_FAILED and decision.get("lifecycle_state") == RUN_VALID_FAILED
             and decision.get("branch") == "BOTH_FAIL"
             and state.get("completed_updates_by_arm", {}).get("B") == 500,
             "the source fit20 A/B task must remain closed at its historical VALID_FAILED/BOTH_FAIL state")
    population, schedule = fit20._load_saved_population(source_dir / "fixed_population_and_schedule.pt")
    expected = fit20._fit_population(inputs)
    _require(all(torch.equal(population[key], expected[key]) for key in
                 ("coefficients", "labels", "proposal_ids", "source_validation_indices")),
             "source B population differs from the registered 20 known-reachable targets")
    history = fit20._read_history(source_dir / "arms/B/training_history.json", "B", identity.run_id, schedule)
    _require(len(history) == 500, "source B arm does not contain exactly 500 committed updates")
    checkpoint = fit20._arm_terminal_checkpoint(source_dir, "B", identity.run_id)
    _require(checkpoint.get("completed_updates") == START_BRANCH_STEP
             and checkpoint.get("history_count") == START_BRANCH_STEP
             and checkpoint.get("schedule_identity") ==
                 "fit20_paired_schedule_seed_2026102781:500x10"
             and checkpoint.get("schedule_prefix_length") == START_BRANCH_STEP
             and checkpoint.get("next_batch_indices") is None,
             "source B500 checkpoint step, history, Adam, or sampler cursor is invalid")
    return {"source_dir": source_dir, "identity": identity, "state": state,
            "decision": decision, "population": population, "old_schedule": schedule,
            "history": history, "checkpoint": checkpoint}


def _build_continuation_schedule(old_schedule: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SCHEDULE_SEED)
    schedule = torch.empty((SCHEDULE_CYCLES, 2, CLASS_COUNT), dtype=torch.long)
    for cycle in range(SCHEDULE_CYCLES):
        for label in range(CLASS_COUNT):
            order = torch.randperm(2, generator=generator)
            schedule[cycle, :, label] = label * 2 + order
    _require(torch.equal(schedule[:250], old_schedule),
             "extended schedule does not preserve the complete original 250-cycle prefix")
    expanded = schedule.reshape(1500, CLASS_COUNT)
    _require(torch.equal(expanded // 2,
                         torch.arange(CLASS_COUNT).expand(1500, CLASS_COUNT))
             and all(set(schedule[cycle, :, label].tolist()) == {2 * label, 2 * label + 1}
                     for cycle in range(SCHEDULE_CYCLES) for label in range(CLASS_COUNT)),
             "extended schedule does not preserve one fixed target per class per batch")
    return schedule


def _validate_windows_review(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(config["source"]["windows_review_receipt"])
    _require(path.is_file(), f"independent Windows review receipt is missing: {path}", FileNotFoundError)
    receipt = _json(path)
    _require(receipt.get("status") == "WINDOWS_REVIEW_RECONSUMPTION_PASSED"
             and receipt.get("task_id") == fit20.TASK_ID
             and receipt.get("source_run_id") == source["identity"].run_id
             and receipt.get("historical_state_preserved") == RUN_VALID_FAILED
             and receipt.get("scientific_updates") == 0
             and receipt.get("source_population_exact") is True
             and receipt.get("all_six_metrics_independently_recomputed") is True
             and receipt.get("recovery_raw_states_recomputed", {}).get("all_raw_recovery_states_verified") is True
             and set(receipt.get("commands", {})) == {"A", "B"},
             "the B continuation requires the successful Windows reconsumption of the old repair evidence")
    for arm, command in receipt["commands"].items():
        _require(command.get("history_updates") == START_BRANCH_STEP
                 and command.get("adam_parameter_steps") == [START_BRANCH_STEP] * 4
                 and command.get("all_six_metric_rows_recomputed") is True
                 and command.get("historical_consumer_accepted_witness_substitution") is True
                 and command.get("repaired_consumer_rejected_witness_substitution") is True
                 and float(command.get("windows_cpu_to_optical_response_relative_error_max", math.inf))
                     <= projected.MATRIX_LIMIT,
                 f"Windows old-evidence review details are incomplete for arm {arm}")
    recovery = receipt["recovery_raw_states_recomputed"]
    recovery_arms = recovery.get("arms", {})
    _require(set(recovery_arms) == {"A", "B"}
             and recovery.get("scientific_updates") == 0
             and recovery.get("engineering_adam_calls_total") == 10,
             "Windows old-evidence review does not contain both raw recovery comparisons")
    for arm, comparison in recovery_arms.items():
        _require(comparison.get("step3_adam_steps") == [3] * 4
                 and comparison.get("live_step4_adam_steps") == [4] * 4
                 and comparison.get("restored_step4_adam_steps") == [4] * 4
                 and comparison.get("engineering_adam_calls") == 5
                 and float(comparison.get("mapper_max_abs_delta", math.inf)) <= RECOVERY_TOLERANCE
                 and float(comparison.get("adam_max_abs_delta", math.inf)) <= RECOVERY_TOLERANCE
                 and float(comparison.get("total_loss_abs_delta", math.inf)) <= RECOVERY_TOLERANCE
                 and float(comparison.get("coefficient_loss_abs_delta", math.inf)) <= RECOVERY_TOLERANCE
                 and comparison.get("rng_exact") is True
                 and comparison.get("schedule_exact") is True
                 and comparison.get("frozen_optics_exact") is True,
                 f"Windows raw recovery comparison is incomplete for arm {arm}")
    for transfer_name in ("old_attempt_transfer", "repair_transfer"):
        transfer = receipt.get(transfer_name, {})
        files = transfer.get("files", [])
        paths = [item.get("path") for item in files]
        _require(transfer.get("status") == "COMPLETE"
                 and transfer.get("all_remote_local_current_sizes_match") is True
                 and transfer.get("file_count") == len(files)
                 and len(paths) == len(set(paths))
                 and sum(int(item.get("remote_bytes", -1)) for item in files)
                     == transfer.get("bytes_total")
                 and all(int(item.get("remote_bytes", -1))
                         == int(item.get("receipt_local_bytes", -2))
                         == int(item.get("current_bytes", -3)) for item in files),
                 f"Windows transfer evidence is incomplete or inconsistent: {transfer_name}")
    return receipt


def _flatten_schedule(schedule: torch.Tensor) -> torch.Tensor:
    return schedule.reshape(1500, CLASS_COUNT)


def _source_batch(schedule: torch.Tensor, local_update: int) -> torch.Tensor:
    position = START_BRANCH_STEP + local_update
    _require(0 <= local_update < ADDED_UPDATES and position < 1500,
             "B continuation schedule cursor is outside the fixed 1,000-update suffix")
    return _flatten_schedule(schedule)[position]


def _rng_snapshot() -> dict[str, Any]:
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]}


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (torch.equal(left["cpu"].cpu(), right["cpu"].cpu())
            and len(left["cuda"]) == len(right["cuda"])
            and all(torch.equal(a.cpu(), b.cpu()) for a, b in
                    zip(left["cuda"], right["cuda"], strict=True)))


def _restore_rng(rng: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(rng["cpu"].cpu())
    torch.cuda.set_rng_state_all(rng["cuda"])


def _optimizer(mapper: torch.nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(mapper.parameters(), lr=1.0e-4, betas=(0.9, 0.999),
                            eps=1.0e-8, weight_decay=0.0)


def _optimizer_steps(optimizer_state: Mapping[str, Any]) -> list[int]:
    return [int(torch.as_tensor(value["step"]).item())
            for value in optimizer_state["state"].values()
            if isinstance(value, Mapping) and "step" in value]


def _new_objects(inputs: Mapping[str, Any], source_checkpoint: Mapping[str, Any],
                 device: torch.device) -> tuple[torch.nn.Module, torch.optim.Adam, Any, dict[str, Any]]:
    mapper = projected._new_mapper(device, source_checkpoint["mapper_state"])
    optimizer = _optimizer(mapper)
    optimizer.load_state_dict(copy.deepcopy(source_checkpoint["optimizer_state"]))
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    frozen_before = projected._clone_state(model)
    _restore_rng({"cpu": source_checkpoint["torch_cpu_rng"],
                  "cuda": source_checkpoint["torch_cuda_rng"]})
    return mapper, optimizer, model, frozen_before


def _update(mapper: torch.nn.Module, optimizer: torch.optim.Adam, model: Any,
            population: Mapping[str, Any], schedule: torch.Tensor,
            inputs: Mapping[str, Any], device: torch.device, local_update: int,
            *, audit: bool = False) -> dict[str, Any]:
    indices = _source_batch(schedule, local_update)
    labels = population["labels"][indices]
    _require(torch.equal(labels, torch.arange(CLASS_COUNT)),
             "scheduled B continuation batch is not one target per class")
    coefficients = population["coefficients"][indices]
    target = coefficients.to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(coefficients, inputs["scale"], device)
    matrix = inputs["matrix"].to(device=device, dtype=torch.float64)
    inverse = inputs["inverse"].to(device=device, dtype=torch.float64)
    scale = inputs["scale"].to(device=device, dtype=torch.float64)
    before = [parameter.detach().clone() for parameter in mapper.parameters()] if audit else None
    optimizer.zero_grad(set_to_none=True)
    amplitude, projection = projected._map_amplitude(
        mapper, features, target, matrix, inverse, scale, include_preclip=True,
    )
    # _forward_amplitude converts the float32 command to float64 at the frozen
    # physical SLM boundary; production propagation applies amplitude squared.
    output = projected._forward_amplitude(model, amplitude)
    coefficient_losses = pilot._phase_invariant_loss(output["coefficients"], target)
    coefficient_loss = coefficient_losses.mean()
    box_loss = fit20._box_loss(projection["preclip_amplitude"].reshape(CLASS_COUNT, PIXEL_COUNT))
    total = coefficient_loss + 5.0 * box_loss
    _require(torch.isfinite(total).item() and torch.isfinite(coefficient_losses).all().item(),
             "full-optics B convergence objective is nonfinite")
    total.backward()
    gradient_norms: dict[str, float] = {}
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"B mapper gradient is absent or nonfinite: {name}")
        gradient_norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach().double()).item())
        _require(math.isfinite(gradient_norms[name]), f"B mapper gradient norm is nonfinite: {name}")
    optimizer.step()
    if before is not None:
        _require(all(not torch.equal(old, new.detach()) for old, new in
                     zip(before, mapper.parameters(), strict=True)),
                 "a declared B mapper parameter tensor did not update")
    return {
        "completed_update": local_update + 1,
        "branch_step": START_BRANCH_STEP + local_update + 1,
        "sample_indices": indices.tolist(),
        "arm": "B", "coefficient_loss": float(coefficient_loss.detach().cpu()),
        "box_loss": float(box_loss.detach().cpu()), "total_loss": float(total.detach().cpu()),
        "coefficient_relative_error_mean": float(torch.sqrt(coefficient_losses.detach()).mean().cpu()),
        "coefficient_relative_error_rms": float(torch.sqrt(coefficient_losses.detach().mean()).cpu()),
        "clipped_fraction_mean": float(projection["clipped_fraction"].mean().detach().cpu()),
        "preclip_min": float(projection["preclip_min"].min().detach().cpu()),
        "preclip_max": float(projection["preclip_max"].max().detach().cpu()),
        "gradient_norms": gradient_norms,
        "loss_timing": "computed_before_this_adam_update",
    }


def _checkpoint_payload(mapper: torch.nn.Module, optimizer: torch.optim.Adam, *,
                        run_id: str, source_run_id: str, completed: int,
                        history: Sequence[Mapping[str, Any]], schedule: torch.Tensor) -> dict[str, Any]:
    branch_step = START_BRANCH_STEP + completed
    cursor = branch_step
    flat = _flatten_schedule(schedule)
    return {
        "schema": "lightgen_g1_fit20_box_b_convergence_checkpoint.v1",
        "task_id": TASK_ID, "run_id": run_id,
        "source_task_id": fit20.TASK_ID, "source_run_id": source_run_id,
        "source_arm": "B", "source_branch_step": START_BRANCH_STEP,
        "source_mapper_updates": 5000, "source_fit20_updates": 500,
        "completed_new_updates": completed, "new_update_budget": ADDED_UPDATES,
        "branch_step": branch_step, "checkpoint_step": completed,
        "next_schedule_cursor": cursor,
        "next_batch_indices": flat[cursor].clone() if cursor < flat.shape[0] else None,
        "mapper_state": {key: value.detach().cpu().clone()
                         for key, value in mapper.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
        "history": copy.deepcopy(list(history)), "history_count": len(history),
        "optimizer_deepcopied": True,
        "schedule_identity": "fit20_box_b_schedule_seed_2026102781:750x2x10",
    }


def _save_checkpoint(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Adam, *,
                     run_id: str, source_run_id: str, completed: int,
                     history: Sequence[Mapping[str, Any]], schedule: torch.Tensor) -> dict[str, Any]:
    before = _rng_snapshot()
    _atomic_torch(path, _checkpoint_payload(mapper, optimizer, run_id=run_id,
                                             source_run_id=source_run_id, completed=completed,
                                             history=history, schedule=schedule))
    after = _rng_snapshot()
    _require(_rng_equal(before, after), "checkpoint serialization changed the live process RNG")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(saved, run_id, source_run_id, completed, history, schedule)
    return saved


def _validate_checkpoint(saved: Mapping[str, Any], run_id: str, source_run_id: str,
                         completed: int, history: Sequence[Mapping[str, Any]],
                         schedule: torch.Tensor) -> None:
    branch_step = START_BRANCH_STEP + completed
    flat = _flatten_schedule(schedule)
    next_batch = saved.get("next_batch_indices")
    schedule_matches = ((branch_step == flat.shape[0] and next_batch is None)
                        or (branch_step < flat.shape[0] and torch.is_tensor(next_batch)
                            and torch.equal(next_batch, flat[branch_step])))
    adam_steps = _optimizer_steps(saved.get("optimizer_state", {}))
    _require(saved.get("schema") == "lightgen_g1_fit20_box_b_convergence_checkpoint.v1"
             and saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id
             and saved.get("source_task_id") == fit20.TASK_ID
             and saved.get("source_run_id") == source_run_id
             and saved.get("source_arm") == "B"
             and saved.get("source_branch_step") == START_BRANCH_STEP
             and saved.get("completed_new_updates") == completed
             and saved.get("branch_step") == branch_step
             and saved.get("checkpoint_step") == completed
             and saved.get("next_schedule_cursor") == branch_step
             and saved.get("history_count") == completed
             and len(saved.get("history", [])) == completed
             and list(saved.get("history", [])) == list(history)
             and len(history) == completed
             and saved.get("optimizer_deepcopied") is True
             and saved.get("schedule_identity") ==
                 "fit20_box_b_schedule_seed_2026102781:750x2x10"
             and schedule_matches
             and len(adam_steps) == 4 and all(value == branch_step for value in adam_steps),
             "B convergence checkpoint identity, history, optimizer, or schedule state is invalid")
    for index, item in enumerate(history):
        _require(item.get("completed_update") == index + 1
                 and item.get("branch_step") == START_BRANCH_STEP + index + 1
                 and item.get("sample_indices") == flat[START_BRANCH_STEP + index].tolist()
                 and item.get("arm") == "B",
                 "checkpoint history is not the exact fixed schedule prefix")


def _restore_checkpoint(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Adam, *,
                        run_id: str, source_run_id: str, schedule: torch.Tensor
                        ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    _require(path.is_file(), f"explicit B convergence checkpoint is missing: {path}", FileNotFoundError)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    completed = int(saved.get("completed_new_updates", -1))
    history = saved.get("history")
    _require(isinstance(history, list) and 0 <= completed <= ADDED_UPDATES,
             "explicit B checkpoint contains invalid progress or history")
    _validate_checkpoint(saved, run_id, source_run_id, completed, history, schedule)
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    _restore_rng({"cpu": saved["torch_cpu_rng"], "cuda": saved["torch_cuda_rng"]})
    return completed, copy.deepcopy(history), saved


def _atomic_history(path: Path, run_id: str, source_run_id: str,
                    history: Sequence[Mapping[str, Any]], status: str) -> None:
    _atomic_json(path, {
        "schema": "lightgen_g1_fit20_box_b_convergence_history.v1",
        "task_id": TASK_ID, "run_id": run_id, "source_run_id": source_run_id,
        "arm": "B", "state": status, "completed_new_updates": len(history),
        "source_branch_step": START_BRANCH_STEP, "terminal_branch_step": START_BRANCH_STEP + len(history),
        "updates": list(history),
    })


def _preflight_identity_path() -> Path:
    return PREFLIGHT_DIR / "preflight_identity.json"


def _preflight_worker(config_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    config = _load_config(config_path)
    _set_numerics()
    _require(torch.cuda.is_available(), "B convergence recovery worker requires CUDA")
    inputs = _load_inputs(config)
    source = _load_source(config, inputs, "cloud")
    schedule = _build_continuation_schedule(source["old_schedule"])
    identity = _json(_preflight_identity_path())
    run_id = str(identity["run_id"])
    device = torch.device("cuda:0")
    mapper, optimizer, model, frozen_before = _new_objects(inputs, source["checkpoint"], device)
    completed, history, _ = _restore_checkpoint(
        checkpoint_path, mapper, optimizer, run_id=run_id,
        source_run_id=source["identity"].run_id, schedule=schedule,
    )
    _require(completed == 3, "recovery worker requires the explicit new-update step-3 checkpoint")
    item = _update(mapper, optimizer, model, source["population"], schedule, inputs,
                   device, 3, audit=False)
    after = {"mapper": {key: value.detach().cpu().clone()
                         for key, value in mapper.state_dict().items()},
             "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": _rng_snapshot(),
             "loss": item["total_loss"], "coefficient_loss": item["coefficient_loss"],
             "sample_indices": _source_batch(schedule, 3).clone(),
             "engineering_adam_calls": 1}
    projected._assert_frozen(model, frozen_before, "B convergence recovery optical model")
    _atomic_torch(output_path, after)


def _recovery_state(mapper: torch.nn.Module, optimizer: torch.optim.Adam,
                    item: Mapping[str, Any], sample_indices: torch.Tensor) -> dict[str, Any]:
    return {"mapper": {key: value.detach().cpu().clone()
                        for key, value in mapper.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": _rng_snapshot(),
            "loss": float(item["total_loss"]),
            "coefficient_loss": float(item["coefficient_loss"]),
            "sample_indices": sample_indices.clone(), "engineering_adam_calls": 1}


def _run_recovery_preflight(config_path: Path, config: Mapping[str, Any],
                            inputs: Mapping[str, Any], source: Mapping[str, Any],
                            population: Mapping[str, Any], schedule: torch.Tensor,
                            device: torch.device, run_id: str) -> dict[str, Any]:
    recovery_dir = PREFLIGHT_DIR / "engineering_recovery"
    recovery_dir.mkdir(exist_ok=True)
    mapper, optimizer, model, frozen_before = _new_objects(inputs, source["checkpoint"], device)
    history: list[dict[str, Any]] = []
    for local_update in range(3):
        history.append(_update(mapper, optimizer, model, population, schedule, inputs,
                               device, local_update, audit=local_update == 0))
    cp3 = recovery_dir / "checkpoint_new_000003.pt"
    _save_checkpoint(cp3, mapper, optimizer, run_id=run_id,
                     source_run_id=source["identity"].run_id, completed=3,
                     history=history, schedule=schedule)
    live_item = _update(mapper, optimizer, model, population, schedule, inputs,
                        device, 3, audit=False)
    live = _recovery_state(mapper, optimizer, live_item, _source_batch(schedule, 3))
    _atomic_torch(recovery_dir / "original_process_new_000004.pt", live)
    worker_path = recovery_dir / "independent_process_new_000004.pt"
    command = [sys.executable, "-m", MODULE_PATH,
               "--mode", "preflight-worker", "--config", str(config_path.resolve()),
               "--checkpoint", str(cp3.resolve()), "--worker-output", str(worker_path.resolve())]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    _require(completed.returncode == 0 and worker_path.is_file(),
             f"independent B convergence recovery process failed: {completed.stderr[-3000:]}")
    restored = torch.load(worker_path, map_location="cpu", weights_only=False)
    mapper_equal, mapper_delta = projected._tree_max_delta(live["mapper"], restored["mapper"])
    adam_equal, adam_delta = projected._tree_max_delta(live["optimizer"], restored["optimizer"])
    loss_delta = abs(live["loss"] - float(restored["loss"]))
    coefficient_delta = abs(live["coefficient_loss"] - float(restored["coefficient_loss"]))
    rng_equal = _rng_equal(live["rng"], restored["rng"])
    schedule_equal = torch.equal(live["sample_indices"], restored["sample_indices"])
    _require(mapper_equal and adam_equal and loss_delta <= RECOVERY_TOLERANCE
             and coefficient_delta <= RECOVERY_TOLERANCE and rng_equal and schedule_equal
             and live.get("engineering_adam_calls") == restored.get("engineering_adam_calls") == 1,
             "live and independently restored B fourth updates exceed the recovery contract")
    projected._assert_frozen(model, frozen_before, "B convergence recovery optical model")
    return {"pre_updates": 3, "source_branch_step": START_BRANCH_STEP,
            "live_branch_step": START_BRANCH_STEP + 4,
            "independent_restored_branch_step": START_BRANCH_STEP + 4,
            "engineering_adam_calls_total": 5,
            "mapper_max_abs_delta": mapper_delta, "adam_max_abs_delta": adam_delta,
            "total_loss_abs_delta": loss_delta,
            "coefficient_loss_abs_delta": coefficient_delta,
            "rng_exact": rng_equal, "schedule_exact": schedule_equal,
            "frozen_optics_exact": True,
            "checkpoint_optimizer_deepcopied": True,
            "source_branch_and_optimizer_restored": True}


def _preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "B convergence preflight requires AutoDL CUDA")
    inputs = _load_inputs(config)
    source = _load_source(config, inputs, "cloud")
    windows_review = _validate_windows_review(config, source)
    old_transfer = windows_review["old_attempt_transfer"]
    review_transfer = windows_review["repair_transfer"]
    prior_identity = None
    if PREFLIGHT_DIR.exists():
        summary_path = PREFLIGHT_DIR / "preflight_summary.json"
        receipt_path = PREFLIGHT_DIR / "preflight_receipt.json"
        _require(not (summary_path.is_file() and _json(summary_path).get("status") == "READY")
                 and not (receipt_path.is_file() and _json(receipt_path).get("status") == "READY"),
                 f"B convergence already has a terminal READY preflight: {PREFLIGHT_DIR}", FileExistsError)
        if _preflight_identity_path().is_file():
            prior_identity = _json(_preflight_identity_path())
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if prior_identity is None:
        run_id = f"fit20bconv_{uuid.uuid4().hex}"
        identity = _run_identity(config, run_id)
        _atomic_json(_preflight_identity_path(), {"task_id": TASK_ID, "run_id": run_id,
                                                  "run_identity": identity.to_dict()})
        _atomic_json(PREFLIGHT_DIR / "run_config_snapshot.json", config)
    else:
        run_id = str(prior_identity["run_id"])
        identity = ResearchRunIdentity.from_mapping(prior_identity["run_identity"])
        _require(prior_identity.get("task_id") == TASK_ID
                 and identity.execution_semantics() == _run_identity(config, run_id).execution_semantics(),
                 "existing B preflight identity differs from the fixed task")
    authority_dir = PREFLIGHT_DIR / "authorities"
    authority_dir.mkdir(exist_ok=True)
    target_authority = authority_dir / "windows_review_receipt.json"
    if _resolve(config["source"]["windows_review_receipt"]) != target_authority.resolve():
        _atomic_json(target_authority, windows_review)
    _atomic_torch(PREFLIGHT_DIR / "source_B500_checkpoint.pt", source["checkpoint"])
    population = source["population"]
    schedule = _build_continuation_schedule(source["old_schedule"])
    _atomic_torch(PREFLIGHT_DIR / "fixed_population_and_schedule.pt", {
        **population, "source_training_schedule": source["old_schedule"],
        "continuation_schedule": schedule,
        "continuation_update_schedule": _flatten_schedule(schedule)[START_BRANCH_STEP:],
        "source_run_id": source["identity"].run_id,
        "source_arm": "B", "source_branch_step": START_BRANCH_STEP,
    })
    device = torch.device("cuda:0")
    probe_inputs = dict(inputs)
    probe_inputs["terminal_mapper"] = {
        **dict(inputs["terminal_mapper"]), "mapper_state": source["checkpoint"]["mapper_state"],
    }
    direction = fit20._fit20_objective_gradient_check(
        probe_inputs, population, inputs["matrix"].to(device), inputs["inverse"].to(device),
        inputs["scale"].to(device), device,
    )
    _require(direction.get("full_production_path") is True
             and direction.get("descent_total_loss", math.inf) < direction.get("center_total_loss", -math.inf),
             "B500 full-optics objective descent control failed")
    recovery = _run_recovery_preflight(config_path, config, inputs, source,
                                       population, schedule, device, run_id)
    recovery_payload = {"task_id": TASK_ID, "run_id": run_id,
                        "status": "PASS", "scientific_updates": 0,
                        **recovery}
    _atomic_json(PREFLIGHT_DIR / "engineering_recovery.json", recovery_payload)
    correctness = {
        "passed": True, "status": "PASS",
        "source_fit20_attempt_closed_valid_failed_both_fail": True,
        "old_evidence_windows_reconsumed": True,
        "windows_review_receipt_consumed": True,
        "source_B500_model_adam_history_and_population_verified": True,
        "schedule_first_250_cycles_exactly_reused": True,
        "B_objective_full_optics_descent_control_passed": True,
        "three_step_and_independent_step4_recovery_passed": True,
        "all_mapper_parameters_trainable_and_optics_frozen": True,
        "scientific_updates": 0,
    }
    memory = torch.cuda.mem_get_info(device)
    resource = {"passed": True, "status": "PASS",
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_memory_free_bytes_observation": int(memory[0]),
                "cuda_memory_total_bytes_observation": int(memory[1]),
                "free_memory_is_gate": False}
    implementation = {"passed": True, "status": "PASS",
                      "selected_engine": "CUDA_full_C_encoder_OLS_decoder_production_path",
                      "recovery_tolerance": RECOVERY_TOLERANCE,
                      "B500_source_optimizer_preserved": True,
                      "new_updates_budget": ADDED_UPDATES}
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness=correctness, resource=resource, implementation=implementation,
        selected_engine="CUDA_full_C_encoder_OLS_decoder_production_path",
        resource_envelope=resource,
    )
    control.mark_preflight_ready(receipt)
    _atomic_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
    summary = {"task_id": TASK_ID, "run_id": run_id, "status": "READY",
               "source_run_id": source["identity"].run_id,
               "source_branch_step": START_BRANCH_STEP,
               "population_role": "known_reachable_fit20",
               "windows_review_status": windows_review["status"],
               "old_transfer_file_count": old_transfer["file_count"],
               "repair_transfer_file_count": review_transfer["file_count"],
               "schedule_prefix_verified": True,
               "B500_full_optics_descent_control": direction,
               "engineering_recovery": recovery_payload,
               "scientific_updates": 0, "resource_observations": resource,
               "ready_receipt_consumable": True}
    _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", summary)
    return {"status": "READY", "run_id": run_id,
            "engineering_adam_calls": recovery["engineering_adam_calls_total"],
            "scientific_updates": 0}


def _load_control() -> tuple[ResearchRunControl, PreflightReceipt]:
    identity_payload = _json(PREFLIGHT_DIR / "preflight_identity.json")
    _require(identity_payload.get("task_id") == TASK_ID,
             "B convergence READY identity belongs to another task")
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    receipt = PreflightReceipt.from_mapping(_json(PREFLIGHT_DIR / "preflight_receipt.json"))
    _require(identity.task_id == TASK_ID and receipt.authorization_id == AUTHORITY
             and receipt.status == "READY"
             and receipt.source_semantics_id == identity.execution_source_semantics_id
             and _json(PREFLIGHT_DIR / "preflight_summary.json").get("status") == "READY",
             "B convergence requires a complete consumable READY preflight")
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    return control, receipt


def _assert_attempt(attempt_dir: Path) -> ResearchRunIdentity:
    preflight = _json(attempt_dir / "preflight_evidence/preflight_identity.json")
    attempt = ResearchRunIdentity.from_mapping(_json(attempt_dir / "run_identity.json"))
    expected = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    expected = expected.with_scientific_attempt(f"{expected.run_id}:attempt_01")
    _require(preflight.get("task_id") == TASK_ID
             and attempt.execution_semantics() == expected.execution_semantics(),
             "B convergence attempt differs from its READY task identity")
    receipt = PreflightReceipt.from_mapping(
        _json(attempt_dir / "preflight_evidence/preflight_receipt.json"))
    _require(receipt.authorization_id == AUTHORITY and receipt.status == "READY"
             and receipt.source_semantics_id == expected.execution_source_semantics_id,
             "B convergence attempt did not consume its READY receipt")
    return attempt


def _copy_preflight_evidence(attempt_dir: Path) -> None:
    destination = attempt_dir / "preflight_evidence"
    destination.mkdir(exist_ok=False)
    for source in PREFLIGHT_DIR.rglob("*"):
        if source.is_file() and source.name != "source_B500_checkpoint.pt":
            target = destination / source.relative_to(PREFLIGHT_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _task_state(attempt_dir: Path, state: str, completed: int,
                note: str | None = None) -> None:
    identity = _json(attempt_dir / "run_identity.json")
    payload: dict[str, Any] = {
        "schema": "lightgen_g1_fit20_box_b_convergence_state.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "source_run_id": _json(PREFLIGHT_DIR / "preflight_summary.json")["source_run_id"],
        "state": state, "completed_new_updates": completed,
        "source_branch_step": START_BRANCH_STEP,
        "current_branch_step": START_BRANCH_STEP + completed,
        "new_update_budget": ADDED_UPDATES,
        "frozen_optical_updates": 0, "updated_at_unix": time.time(),
    }
    if note is not None:
        payload["note"] = note
    _atomic_json(attempt_dir / "task_state.json", payload)


def _open_attempt(config: Mapping[str, Any], resume: bool) -> tuple[ResearchRunControl, PreflightReceipt]:
    control, receipt = _load_control()
    if not resume:
        control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01", receipt)
        _atomic_json(ATTEMPT_DIR / "run_identity.json", control.identity.to_dict())
        _atomic_json(ATTEMPT_DIR / "run_config_snapshot.json", config)
        _copy_preflight_evidence(ATTEMPT_DIR)
        _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, 0)
    else:
        _require(ATTEMPT_DIR.is_dir(), "explicit resume requires the existing B convergence attempt")
        _assert_attempt(ATTEMPT_DIR)
        state = _json(ATTEMPT_DIR / "task_state.json")
        _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "INTERRUPTED"},
                 "closed B convergence task cannot be reopened or resumed")
        control.open_scientific_attempt(ATTEMPT_DIR,
                                        f"{control.identity.run_id}:attempt_01", receipt, resume=True)
    return control, receipt


def _save_population(path: Path, population: Mapping[str, Any], schedule: torch.Tensor,
                     source: Mapping[str, Any]) -> None:
    _atomic_torch(path, {**dict(population),
                         "source_training_schedule": source["old_schedule"].detach().cpu().clone(),
                         "continuation_schedule": schedule.detach().cpu().clone(),
                         "continuation_update_schedule": _flatten_schedule(schedule)[
                             START_BRANCH_STEP:].detach().cpu().clone(),
                         "source_run_id": source["identity"].run_id,
                         "source_arm": "B", "source_branch_step": START_BRANCH_STEP})


def _load_saved_population(path: Path, source: Mapping[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schedule = payload.get("continuation_schedule")
    _require(payload.get("schema") == "lightgen_g1_projected_fit20_population.v1"
             and payload.get("task_id") == fit20.TASK_ID
             and payload.get("role") == "known_reachable_fit20"
             and payload.get("source_run_id") == source["identity"].run_id
             and payload.get("source_arm") == "B"
             and payload.get("source_branch_step") == START_BRANCH_STEP
             and tuple(payload["coefficients"].shape) == (TARGET_COUNT, PORT_COUNT)
             and torch.equal(payload["labels"], torch.arange(CLASS_COUNT).repeat_interleave(2))
             and all(torch.equal(payload.get(key), source["population"][key]) for key in
                     ("coefficients", "labels", "proposal_ids", "source_validation_indices"))
             and torch.equal(payload.get("source_training_schedule"), source["old_schedule"])
             and torch.is_tensor(schedule) and tuple(schedule.shape) == (SCHEDULE_CYCLES, 2, CLASS_COUNT)
             and torch.equal(schedule, _build_continuation_schedule(source["old_schedule"])),
             "saved B continuation population, source identity, or extended schedule changed")
    _require(torch.equal(payload.get("continuation_update_schedule"),
                         _flatten_schedule(schedule)[START_BRANCH_STEP:]),
             "saved continuation update schedule is not the registered 1,000-step suffix")
    return payload, schedule


def _measure(mapper: torch.nn.Module, population: Mapping[str, Any],
             inputs: Mapping[str, Any], device: torch.device,
             new_updates: int) -> dict[str, Any]:
    target = population["coefficients"].to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(population["coefficients"], inputs["scale"], device)
    matrix = inputs["matrix"].to(device=device, dtype=torch.float64)
    inverse = inputs["inverse"].to(device=device, dtype=torch.float64)
    scale = inputs["scale"].to(device=device, dtype=torch.float64)
    with torch.no_grad():
        amplitudes, detail = projected._map_amplitude(
            mapper, features, target, matrix, inverse, scale, include_preclip=True,
        )
        rows, measured = fit20._measure_commands(population, amplitudes, inputs, device, GRIDS)
    summary = projected._summarize_rows(rows)
    return {
        "schema": "lightgen_g1_fit20_box_b_convergence_observation.v1",
        "task_id": TASK_ID, "source_task_id": fit20.TASK_ID,
        "source_run_id": population.get("source_run_id"),
        "arm": "B", "completed_new_updates": new_updates,
        "branch_step": START_BRANCH_STEP + new_updates,
        "population_role": "known_reachable_fit20",
        "command_encoding": "continuous_amplitude_float32",
        "metric_keys": list(METRIC_KEYS), "threshold": GATE_LIMIT,
        "coefficient_alignment": "one_global_phase_per_sample",
        "native_cmos_normalization": "none",
        "target_coefficients": population["coefficients"].clone(),
        "labels": population["labels"].clone(),
        "proposal_ids": population["proposal_ids"].clone(),
        "source_validation_indices": population["source_validation_indices"].clone(),
        "mapper_state": {key: value.detach().cpu().clone()
                         for key, value in mapper.state_dict().items()},
        "amplitudes": measured["amplitudes"],
        "preclip_min": detail["preclip_min"].detach().cpu(),
        "preclip_max": detail["preclip_max"].detach().cpu(),
        "clipped_fraction": detail["clipped_fraction"].detach().cpu(),
        "outputs": measured["outputs"], "direct_raw_cmos": measured["direct_raw_cmos"],
        "rows": rows, "summary": summary,
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ["population_index", "source_validation_index", "label", "proposal_id",
               *METRIC_KEYS, "response_matrix_relative_error_2048", "row_passed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def _save_observation(attempt_dir: Path, observation: Mapping[str, Any]) -> dict[str, Any]:
    step = int(observation["completed_new_updates"])
    path = attempt_dir / "measurements" / f"step_{step:04d}.pt"
    _atomic_torch(path, observation)
    _write_rows(path.with_name(f"step_{step:04d}_rows.csv"), observation["rows"])
    _atomic_json(path.with_name(f"step_{step:04d}_summary.json"), observation["summary"])
    receipt = {"task_id": TASK_ID, "run_id": _json(attempt_dir / "run_identity.json")["run_id"],
               "arm": "B", "completed_new_updates": step,
               "branch_step": START_BRANCH_STEP + step,
               "all_20_rows_passed": observation["summary"]["all_rows_passed"],
               "passed_rows": observation["summary"]["passed_rows"],
               "maxima": observation["summary"]["maxima"],
               "observation_file": path.name}
    records_path = attempt_dir / "observations.json"
    current = _json(records_path).get("observations", []) if records_path.is_file() else []
    by_step = {int(item["completed_new_updates"]): item for item in [*current, receipt]}
    _atomic_json(records_path, {"task_id": TASK_ID,
                                "run_id": _json(attempt_dir / "run_identity.json")["run_id"],
                                "arm": "B",
                                "observations": [by_step[key] for key in sorted(by_step)]})
    return receipt


def _observe_checkpoint(attempt_dir: Path, mapper: torch.nn.Module,
                        population: Mapping[str, Any], inputs: Mapping[str, Any],
                        device: torch.device, completed: int) -> dict[str, Any]:
    path = attempt_dir / "measurements" / f"step_{completed:04d}.pt"
    if path.is_file():
        return _json(path.with_name(f"step_{completed:04d}_summary.json"))
    return _save_observation(attempt_dir, _measure(mapper, population, inputs, device, completed))


def _write_manifest(attempt_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_reconsumption.json"}
    files = []
    total = 0
    for path in sorted(item for item in attempt_dir.rglob("*")
                       if item.is_file() and item.name not in excluded):
        size = path.stat().st_size
        files.append({"path": path.relative_to(attempt_dir).as_posix(), "bytes": size})
        total += size
    manifest = {"schema": "lightgen_g1_fit20_box_b_convergence_transfer.v1",
                "task_id": TASK_ID, "run_id": run_id,
                "file_count": len(files), "bytes_total": total, "files": files}
    _atomic_json(attempt_dir / "transfer_manifest.json", manifest)
    return manifest


def _write_report(attempt_dir: Path, decision: Mapping[str, Any]) -> None:
    summary = decision["summary"]
    report = f"""# G1 fit20 B臂固定预算收敛确认

## 结果与状态

新任务状态：`{decision['lifecycle_state']}`；旧A/B任务继续保持历史 `VALID_FAILED / BOTH_FAIL`。B臂从分支步500恢复并新增恰好1,000次科学更新，最终分支步为1,500。固定20条目标通过六项误差判据 {summary['passed_rows']}/20。

## 科学认识

本任务只检验当前B结构与固定 `Lc + 5Lbox` 损失在同一组 `known_reachable_fit20` 目标上的延续拟合。参数谱系为原映射器5,000步、projected mapping新增5,000步、旧fit20 B臂500步、本任务新增1,000步，累计11,500步。不能把结果归因于结构的单因素作用，也不构成独立泛化。

## 目标缺口

唯一科学判据要求固定20条目标在2048/4096两网格的逐网格系数误差、原生CMOS误差和跨网格对应误差全部不超过 `{GATE_LIMIT:.1e}`。CMOS保持原始未归一化输出，没有功率匹配或增益调整。无论通过与否，本结果均不证明新人口、人眼语义可读性或实物光路能力。

## 工程完成与验证

起点是已保存的B500模型和Adam状态；日程前250轮与旧任务逐项一致，并固定使用后500轮。B损失和梯度来自完整冻结光路。新增科学更新为 `{decision['new_scientific_updates']}`，分支步为 `{decision['branch_start_step']}`→`{decision['branch_terminal_step']}`，Adam参数步数均为 `{decision['adam_parameter_steps'][0]}`。工程恢复预检单独完成5次Adam调用，未计入科学更新；光学更新为0。

## 决策影响与下一步

{decision['interpretation']} 本任务完成后停止；不自动追加训练、扩展目标人口、开展A8/I8、推进G4、启动人工评分或实物实验。任何全人口或部署研究都需要独立的计划与证据。
"""
    (attempt_dir / "final_report.md").write_text(report, encoding="utf-8")


def _close_attempt(config: Mapping[str, Any], control: ResearchRunControl,
                   source: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
                   observation: Mapping[str, Any]) -> dict[str, Any]:
    summary = observation["summary"]
    passed = bool(summary["all_rows_passed"])
    lifecycle = RUN_COMPLETED if passed else RUN_VALID_FAILED
    if passed:
        control.complete()
        interpretation = ("20/20已知可达目标在分支1,500步预算下满足全部六项误差阈值；"
                          "支持该固定人口上的重复拟合能力，不支持独立泛化或物理部署。")
    else:
        control.valid_failed()
        interpretation = ("当前B结构在新增1,000步的固定预算下未使全部20条目标通过；"
                          "不据此判断G1总体不可学习，也不追加训练。")
    decision = {
        "schema": "lightgen_g1_fit20_box_b_convergence_decision.v1",
        "task_id": TASK_ID, "run_id": control.identity.run_id,
        "source_task_id": fit20.TASK_ID, "source_run_id": source["identity"].run_id,
        "lifecycle_state": lifecycle,
        "population_role": "known_reachable_fit20", "denominator": TARGET_COUNT,
        "threshold": GATE_LIMIT, "metric_names": list(METRIC_KEYS),
        "branch": "B", "branch_start_step": START_BRANCH_STEP,
        "branch_terminal_step": TERMINAL_BRANCH_STEP,
        "new_scientific_updates": ADDED_UPDATES,
        "source_mapper_updates": 5000, "projected_mapping_updates": 5000,
        "prior_fit20_B_updates": 500, "cumulative_mapper_updates": 11500,
        "adam_parameter_steps": [TERMINAL_BRANCH_STEP] * 4,
        "frozen_optical_updates": 0, "engineering_recovery_adam_calls": 5,
        "summary": summary, "rows": observation["rows"],
        "all_20_six_metrics_required": True,
        "native_cmos_normalization": "none", "interpretation": interpretation,
        "windows_review_status": _json(PREFLIGHT_DIR / "preflight_summary.json")["windows_review_status"],
    }
    _atomic_json(ATTEMPT_DIR / "final_decision.json", decision)
    _task_state(ATTEMPT_DIR, lifecycle, ADDED_UPDATES, note="fixed 1,000-update B continuation closed")
    _atomic_json(ATTEMPT_DIR / "research_lifecycle_state.json", {
        "schema": "lightgen_g1_fit20_box_b_convergence_lifecycle.v1",
        "task_id": TASK_ID, "run_id": control.identity.run_id,
        "source_run_id": source["identity"].run_id,
        "state": control.state, "preflight_status": control.preflight_status,
        "receipt_consumed": control.receipt is not None,
        "final_state": lifecycle, "new_scientific_updates": ADDED_UPDATES,
        "branch_step": TERMINAL_BRANCH_STEP, "selected_terminal_step": ADDED_UPDATES,
    })
    _write_report(ATTEMPT_DIR, decision)
    _write_manifest(ATTEMPT_DIR, control.identity.run_id)
    return decision


def _execute(config_path: Path, config: Mapping[str, Any], *,
             resume_checkpoint: Path | None = None) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "B convergence training requires AutoDL CUDA")
    resume = resume_checkpoint is not None
    if resume:
        _require(resume_checkpoint.is_file(), "resume requires an existing explicit checkpoint", FileNotFoundError)
        state = _json(ATTEMPT_DIR / "task_state.json") if (ATTEMPT_DIR / "task_state.json").is_file() else {}
        _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "INTERRUPTED"},
                 "closed B convergence task cannot be resumed")
    else:
        _require(not ATTEMPT_DIR.exists(), f"B convergence attempt already exists: {ATTEMPT_DIR}",
                 FileExistsError)
    inputs = _load_inputs(config)
    source = _load_source(config, inputs, "cloud")
    _validate_windows_review(config, source)
    control, _ = _open_attempt(config, resume)
    run_id = control.identity.run_id
    source_run_id = source["identity"].run_id
    if resume:
        identity = _assert_attempt(ATTEMPT_DIR)
        _require(identity.run_id == run_id, "resume attempt and READY run identities disagree")
        population, schedule = _load_saved_population(
            ATTEMPT_DIR / "fixed_population_and_schedule.pt", source)
        saved_source = torch.load(ATTEMPT_DIR / "source_B500_checkpoint.pt",
                                  map_location="cpu", weights_only=False)
        mapper_equal, mapper_delta = projected._tree_max_delta(
            saved_source.get("mapper_state"), source["checkpoint"]["mapper_state"])
        adam_equal, adam_delta = projected._tree_max_delta(
            saved_source.get("optimizer_state"), source["checkpoint"]["optimizer_state"])
        _require(mapper_equal and adam_equal and mapper_delta == 0.0 and adam_delta == 0.0
                 and saved_source.get("completed_updates") == START_BRANCH_STEP
                 and saved_source.get("arm") == "B"
                 and saved_source.get("run_id") == source_run_id,
                 "saved B500 source model/Adam differs from the explicit predecessor checkpoint")
    else:
        population = {**source["population"], "source_run_id": source_run_id}
        schedule = _build_continuation_schedule(source["old_schedule"])
        _save_population(ATTEMPT_DIR / "fixed_population_and_schedule.pt", population, schedule, source)
        _atomic_torch(ATTEMPT_DIR / "source_B500_checkpoint.pt", source["checkpoint"])
    device = torch.device("cuda:0")
    mapper, optimizer, model, frozen_before = _new_objects(inputs, source["checkpoint"], device)
    if resume:
        completed, history, checkpoint = _restore_checkpoint(
            resume_checkpoint.resolve(), mapper, optimizer, run_id=run_id,
            source_run_id=source_run_id, schedule=schedule,
        )
        external = _json(ATTEMPT_DIR / "training_history.json").get("updates", [])
        _require(len(external) >= 0 and all(isinstance(item, Mapping) for item in external),
                 "external training history is malformed")
        overlap = min(len(external), completed)
        _require(external[:overlap] == history[:overlap],
                 "external committed history conflicts with the explicit checkpoint prefix")
        if len(external) > completed:
            _atomic_json(ATTEMPT_DIR / f"uncommitted_history_tail_after_{completed:04d}.json", {
                "task_id": TASK_ID, "run_id": run_id,
                "checkpoint_prefix_length": completed,
                "excluded_external_tail": external[completed:],
            })
        _atomic_history(ATTEMPT_DIR / "training_history.json", run_id, source_run_id,
                        history, RUN_SCIENTIFIC_ACTIVE)
        committed = completed
    else:
        completed, history = 0, []
        directories = (ATTEMPT_DIR / "checkpoints", ATTEMPT_DIR / "measurements")
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(ATTEMPT_DIR / "checkpoints/checkpoint_step_000000.pt",
                         mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                         completed=0, history=history, schedule=schedule)
        _save_checkpoint(ATTEMPT_DIR / "checkpoints/latest_checkpoint.pt",
                         mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                         completed=0, history=history, schedule=schedule)
        _atomic_history(ATTEMPT_DIR / "training_history.json", run_id, source_run_id,
                        history, RUN_SCIENTIFIC_ACTIVE)
        _save_observation(ATTEMPT_DIR, _measure(mapper, population, inputs, device, 0))
        committed = 0
    if completed == 0 and (ATTEMPT_DIR / "measurements/step_0000.pt").is_file() is False:
        _save_observation(ATTEMPT_DIR, _measure(mapper, population, inputs, device, 0))
    _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, completed,
                note="B continuation restored from explicit checkpoint" if resume else None)
    fixed_steps = set(config["training"]["fixed_checkpoint_steps"])
    observation_steps = set(config["training"]["observation_steps"])
    committed = completed
    try:
        for local_update in range(completed, ADDED_UPDATES):
            item = _update(mapper, optimizer, model, population, schedule, inputs,
                           device, local_update, audit=(local_update == 0))
            history.append(item)
            done = local_update + 1
            if done % 25 == 0 or done in fixed_steps:
                _save_checkpoint(ATTEMPT_DIR / "checkpoints/latest_checkpoint.pt",
                                 mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                                 completed=done, history=history, schedule=schedule)
                if done in fixed_steps:
                    _save_checkpoint(ATTEMPT_DIR / f"checkpoints/checkpoint_step_{done:06d}.pt",
                                     mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                                     completed=done, history=history, schedule=schedule)
                _atomic_history(ATTEMPT_DIR / "training_history.json", run_id, source_run_id,
                                history, RUN_SCIENTIFIC_ACTIVE)
                progress = {"task_id": TASK_ID, "run_id": run_id,
                            "source_run_id": source_run_id,
                            "completed_new_updates": done,
                            "branch_step": START_BRANCH_STEP + done,
                            "optimizer_step": START_BRANCH_STEP + done,
                            "last_preupdate_loss": item["total_loss"],
                            "loss_timing": item["loss_timing"],
                            "timestamp_unix": time.time()}
                _atomic_json(ATTEMPT_DIR / "progress.json", progress)
                with (ATTEMPT_DIR / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
                committed = done
                _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, done)
            if done in observation_steps:
                _save_observation(ATTEMPT_DIR, _measure(mapper, population, inputs, device, done))
        projected._assert_frozen(model, frozen_before, "B convergence frozen optical model")
        _require(len(history) == ADDED_UPDATES,
                 "B convergence did not complete its exact 1,000-update budget")
        terminal_path = ATTEMPT_DIR / "checkpoints/checkpoint_step_001000.pt"
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(terminal.get("completed_new_updates") == ADDED_UPDATES
                 and terminal.get("branch_step") == TERMINAL_BRANCH_STEP
                 and len(_optimizer_steps(terminal["optimizer_state"])) == 4
                 and all(step == TERMINAL_BRANCH_STEP
                         for step in _optimizer_steps(terminal["optimizer_state"])),
                 "B convergence terminal checkpoint or Adam step count is invalid")
        _require(set(terminal["mapper_state"]) == set(source["checkpoint"]["mapper_state"])
                 and all(not torch.equal(terminal["mapper_state"][key],
                                        source["checkpoint"]["mapper_state"][key])
                         for key in terminal["mapper_state"]),
                 "one or more B mapper parameter tensors did not change over the fixed budget")
        observation = torch.load(ATTEMPT_DIR / "measurements/step_1000.pt",
                                 map_location="cpu", weights_only=False)
        decision = _close_attempt(config, control, source, history, observation)
        return {"status": decision["lifecycle_state"], "run_id": run_id,
                "new_scientific_updates": ADDED_UPDATES,
                "branch_terminal_step": TERMINAL_BRANCH_STEP,
                "passed_rows": decision["summary"]["passed_rows"],
                "decision": str(ATTEMPT_DIR / "final_decision.json")}
    except Exception as error:
        _task_state(ATTEMPT_DIR, "INTERRUPTED", committed,
                    note=f"{type(error).__name__}: {error}")
        raise


def _measure_saved_checkpoint(config: Mapping[str, Any], checkpoint_path: Path,
                              selected_step: int) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "B convergence measurement requires AutoDL CUDA")
    _require(selected_step in config["training"]["observation_steps"],
             "measure step must be one of the predeclared observations", ValueError)
    state = _json(ATTEMPT_DIR / "task_state.json")
    _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "INTERRUPTED"},
             "closed B convergence attempt cannot be reopened for measurement")
    inputs = _load_inputs(config)
    source = _load_source(config, inputs, "cloud")
    identity = _assert_attempt(ATTEMPT_DIR)
    population, schedule = _load_saved_population(
        ATTEMPT_DIR / "fixed_population_and_schedule.pt", source)
    mapper, optimizer, model, frozen_before = _new_objects(
        inputs, source["checkpoint"], torch.device("cuda:0"))
    completed, history, _ = _restore_checkpoint(
        checkpoint_path.resolve(), mapper, optimizer, run_id=identity.run_id,
        source_run_id=source["identity"].run_id, schedule=schedule,
    )
    _require(completed == selected_step,
             "explicit measurement checkpoint step differs from the selected observation")
    observation = _measure(mapper, population, inputs, torch.device("cuda:0"), completed)
    receipt = _save_observation(ATTEMPT_DIR, observation)
    projected._assert_frozen(model, frozen_before, "B convergence measurement optical model")
    return {"status": "MEASUREMENT_SAVED", "run_id": identity.run_id,
            "completed_new_updates": completed,
            "all_rows_passed": receipt["all_20_rows_passed"]}


def _verify_transfer(evidence_dir: Path) -> dict[str, Any]:
    manifest = _json(evidence_dir / "transfer_manifest.json")
    receipt = _json(evidence_dir / "transfer_receipt.json")
    _require(manifest.get("schema") == "lightgen_g1_fit20_box_b_convergence_transfer.v1"
             and manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list)
             and receipt.get("status") == "COMPLETE" and isinstance(receipt.get("files"), list),
             "B convergence transfer manifest or receipt is invalid")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() not in expected,
                 "B convergence transfer manifest has unsafe or duplicate file paths")
        expected[relative.as_posix()] = int(item["bytes"])
    receipt_by_path: dict[str, Mapping[str, Any]] = {}
    for item in receipt["files"]:
        relative = Path(str(item.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() not in receipt_by_path,
                 "B convergence transfer receipt has unsafe or duplicate file paths")
        receipt_by_path[relative.as_posix()] = item
    expected["transfer_manifest.json"] = (evidence_dir / "transfer_manifest.json").stat().st_size
    _require(set(receipt_by_path) == set(expected),
             "B convergence transfer receipt does not match the manifest inventory")
    checked = []
    for relative, remote_size in sorted(expected.items()):
        path = evidence_dir / Path(relative)
        item = receipt_by_path[relative]
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_size
                 and int(item.get("local_size_bytes", -1)) == remote_size
                 and current == remote_size,
                 f"B convergence transfer size/status mismatch: {relative}", ValueError)
        checked.append({"path": relative, "remote_bytes": remote_size,
                        "receipt_local_bytes": int(item["local_size_bytes"]),
                        "current_bytes": current})
    _require(len(expected) == manifest.get("file_count") + 1
             and sum(size for path, size in expected.items() if path != "transfer_manifest.json")
             == manifest.get("bytes_total"),
             "B convergence transfer manifest count or byte total is inconsistent")
    return {"status": "COMPLETE", "file_count": len(checked),
            "bytes_total": sum(item["current_bytes"] for item in checked),
            "all_remote_local_current_sizes_match": True, "files": checked}


def _verify_remote_manifest(evidence_dir: Path, expected_run_id: str | None = None) -> dict[str, Any]:
    manifest = _json(evidence_dir / "transfer_manifest.json")
    _require(manifest.get("schema") == "lightgen_g1_fit20_box_b_convergence_transfer.v1"
             and manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list)
             and (expected_run_id is None or manifest.get("run_id") == expected_run_id)
             and manifest.get("file_count") == len(manifest["files"])
             and manifest.get("bytes_total") == sum(int(item["bytes"]) for item in manifest["files"]),
             "cloud B convergence source manifest is invalid")
    files = []
    seen = set()
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in seen,
                 "cloud B convergence source manifest has an unsafe or duplicate path")
        seen.add(key)
        path = evidence_dir / relative
        current = path.stat().st_size if path.is_file() else -1
        expected = int(item["bytes"])
        _require(path.is_file() and current == expected,
                 f"cloud B convergence source file differs from its manifest: {key}")
        files.append({"path": key, "manifest_bytes": expected, "current_bytes": current})
    return {"status": "REMOTE_MANIFEST_MATCHED", "file_count": len(files),
            "bytes_total": sum(item["current_bytes"] for item in files), "files": files}


def _consume_observation(path: Path, checkpoint: Mapping[str, Any],
                         population: Mapping[str, Any], inputs: Mapping[str, Any],
                         selected_step: int,
                         cuda_replay: Mapping[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "lightgen_g1_fit20_box_b_convergence_observation.v1"
             and payload.get("task_id") == TASK_ID and payload.get("source_task_id") == fit20.TASK_ID
             and payload.get("source_run_id") == population.get("source_run_id")
             and payload.get("arm") == "B"
             and payload.get("completed_new_updates") == selected_step
             and payload.get("branch_step") == START_BRANCH_STEP + selected_step
             and payload.get("population_role") == "known_reachable_fit20"
             and payload.get("metric_keys") == list(METRIC_KEYS)
             and payload.get("threshold") == GATE_LIMIT
             and payload.get("native_cmos_normalization") == "none"
             and torch.equal(payload.get("target_coefficients"), population["coefficients"])
             and torch.equal(payload.get("labels"), population["labels"])
             and torch.equal(payload.get("proposal_ids"), population["proposal_ids"])
             and torch.equal(payload.get("source_validation_indices"),
                             population["source_validation_indices"]),
             "B convergence observation identity or fixed target population changed")
    _require(set(payload.get("mapper_state", {})) == set(checkpoint["mapper_state"])
             and all(torch.equal(payload["mapper_state"][key], checkpoint["mapper_state"][key])
                     for key in checkpoint["mapper_state"]),
             "saved observation mapper state is not the selected checkpoint")
    amplitudes = payload.get("amplitudes")
    _require(torch.is_tensor(amplitudes) and tuple(amplitudes.shape) == (TARGET_COUNT, 1, 200, 200)
             and amplitudes.dtype == torch.float32 and torch.isfinite(amplitudes).all().item()
             and float(amplitudes.min()) >= 0.0 and float(amplitudes.max()) <= 1.0,
             "B convergence saved amplitude command is malformed")
    _require_cuda_replay_matches(cuda_replay, checkpoint, amplitudes, selected_step)
    _require(isinstance(payload.get("rows"), list) and len(payload["rows"]) == TARGET_COUNT,
             "B convergence observation row denominator is incomplete")
    mapper = projected._new_mapper(torch.device("cpu"), checkpoint["mapper_state"]).eval()
    features = projected._normalized_features(population["coefficients"], inputs["scale"], torch.device("cpu"))
    with torch.no_grad():
        replayed, _ = projected._map_amplitude(
            mapper, features, population["coefficients"].to(torch.complex128),
            inputs["matrix"].double(), inputs["inverse"].double(), inputs["scale"].double(),
        )
    command_delta = float((replayed.cpu() - amplitudes).abs().max().item())
    response = inputs["response"].to(torch.complex128)
    predicted = replayed.reshape(TARGET_COUNT, PIXEL_COUNT).to(torch.complex128) @ response.T
    outputs, direct = payload.get("outputs"), payload.get("direct_raw_cmos")
    _require(set(outputs) == set(GRIDS) and set(direct) == set(GRIDS),
             "B convergence observation is missing a measurement grid")
    actual_2048 = outputs[2048]["coefficients"].to(torch.complex128)
    replay_response_error = float(torch.sqrt(pilot._phase_invariant_loss(
        predicted, actual_2048)).max().item())
    _require(math.isfinite(replay_response_error) and replay_response_error <= projected.MATRIX_LIMIT,
             f"Windows mapper-command response error {replay_response_error:.3g} exceeds "
             f"{projected.MATRIX_LIMIT:.3g}")
    recomputed = []
    for index in range(TARGET_COUNT):
        row_outputs: dict[int, dict[str, torch.Tensor]] = {}
        row_direct: dict[int, torch.Tensor] = {}
        for grid in GRIDS:
            _require(tuple(outputs[grid]["coefficients"].shape) == (TARGET_COUNT, PORT_COUNT)
                     and outputs[grid]["coefficients"].dtype == torch.complex128
                     and tuple(outputs[grid]["raw_cmos"].shape) == (TARGET_COUNT, 1, 200, 200)
                     and outputs[grid]["raw_cmos"].dtype == torch.float64
                     and tuple(direct[grid].shape) == (TARGET_COUNT, 1, 200, 200)
                     and direct[grid].dtype == torch.float64,
                     f"B convergence grid-{grid} raw output shape/dtype changed")
            row_outputs[grid] = {key: value[index:index + 1]
                                 for key, value in outputs[grid].items()
                                 if key in {"coefficients", "raw_cmos"}}
            row_direct[grid] = direct[grid][index:index + 1]
        reported = {key: float(payload["rows"][index][key]) for key in METRIC_KEYS}
        row = projected._consume_production_row(
            population, index, payload["target_coefficients"][index],
            amplitudes[index:index + 1], row_outputs, row_direct, response,
            "continuous_amplitude_float32", reported_metrics=reported,
        )
        row["source_validation_index"] = int(population["source_validation_indices"][index])
        row["row_passed"] = all(float(row[key]) <= GATE_LIMIT for key in METRIC_KEYS)
        recomputed.append(row)
    summary = projected._summarize_rows(recomputed)
    _require(_numeric_tree_matches(payload.get("summary"), summary)
             and len(payload.get("rows", [])) == TARGET_COUNT
             and all(bool(payload["rows"][i].get("row_passed")) is bool(recomputed[i]["row_passed"])
                     for i in range(TARGET_COUNT)),
             "B convergence observation summary or row flags are stale")
    return {"rows": recomputed, "summary": summary, "payload": payload,
            "windows_cpu_to_saved_command_max_abs_delta": command_delta,
            "saved_cuda_to_command_exact": True,
            "windows_cpu_to_optical_response_relative_error_max": replay_response_error}


def _require_cuda_replay_matches(replay: Mapping[str, Any], checkpoint: Mapping[str, Any],
                                 saved_amplitudes: torch.Tensor, selected_step: int) -> None:
    replayed_mapper_state = replay.get("mapper_state")
    checkpoint_mapper_state = checkpoint.get("mapper_state")
    _require(replay.get("schema") == "lightgen_g1_fit20_box_b_cuda_command_replay.v1"
             and replay.get("task_id") == TASK_ID
             and replay.get("run_id") == checkpoint.get("run_id")
             and replay.get("step") == selected_step
             and replay.get("checkpoint_step") == selected_step
             and isinstance(replayed_mapper_state, Mapping)
             and isinstance(checkpoint_mapper_state, Mapping)
             and set(replayed_mapper_state) == set(checkpoint_mapper_state)
             and all(torch.equal(replayed_mapper_state[key], checkpoint_mapper_state[key])
                     for key in checkpoint_mapper_state)
             and torch.is_tensor(replay.get("amplitudes"))
             and replay["amplitudes"].dtype == torch.float32
             and tuple(replay["amplitudes"].shape) == tuple(saved_amplitudes.shape)
             and torch.equal(replay["amplitudes"], saved_amplitudes),
             "saved B command differs from its exact CUDA checkpoint replay")


def _numeric_tree_matches(left: Any, right: Any, tolerance: float = 1.0e-14) -> bool:
    return fit20._numeric_tree_matches(left, right, tolerance)


def _command_replay(evidence_dir: Path, review_dir: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "saved CUDA command replay requires AutoDL CUDA")
    identity = _assert_attempt(evidence_dir)
    population, schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt",
        _load_source(_load_config(DEFAULT_CONFIG), inputs, "cloud"),
    )
    output_dir = review_dir / "command_replays"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    results = {}
    for step in (0, 500, 1000):
        checkpoint_path = evidence_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pt"
        observation_path = evidence_dir / "measurements" / f"step_{step:04d}.pt"
        _require(checkpoint_path.is_file() and observation_path.is_file(),
                 f"saved step-{step} checkpoint or observation is missing", FileNotFoundError)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = checkpoint.get("history")
        _validate_checkpoint(checkpoint, identity.run_id, population["source_run_id"],
                             step, history, schedule)
        observation = torch.load(observation_path, map_location="cpu", weights_only=False)
        mapper = projected._new_mapper(device, checkpoint["mapper_state"]).eval()
        features = projected._normalized_features(population["coefficients"], inputs["scale"], device)
        with torch.no_grad():
            amplitudes, _ = projected._map_amplitude(
                mapper, features, population["coefficients"].to(device=device, dtype=torch.complex128),
                inputs["matrix"].to(device=device, dtype=torch.float64),
                inputs["inverse"].to(device=device, dtype=torch.float64),
                inputs["scale"].to(device=device, dtype=torch.float64), include_preclip=True,
            )
        saved = observation.get("amplitudes")
        _require(torch.is_tensor(saved) and saved.dtype == torch.float32
                 and tuple(saved.shape) == tuple(amplitudes.shape)
                 and torch.equal(saved, amplitudes.detach().cpu()),
                 f"step-{step} saved command does not match exact CUDA checkpoint replay")
        replay = {
            "schema": "lightgen_g1_fit20_box_b_cuda_command_replay.v1",
            "task_id": TASK_ID, "run_id": identity.run_id, "step": step,
            "checkpoint_step": step,
            "mapper_state": {key: value.detach().cpu().clone()
                             for key, value in checkpoint["mapper_state"].items()},
            "amplitudes": amplitudes.detach().cpu().clone(),
            "exact_match_to_saved_command": True,
            "command_batch_size": TARGET_COUNT,
            "device": torch.cuda.get_device_name(device),
        }
        _atomic_torch(output_dir / f"step_{step:04d}.pt", replay)
        results[str(step)] = {"exact_match_to_saved_command": True,
                              "command_max_abs_delta": 0.0}
        del mapper, amplitudes
        torch.cuda.empty_cache()
    return {"run_id": identity.run_id, "status": "CUDA_COMMAND_REPLAY_PASSED",
            "steps": results, "device": torch.cuda.get_device_name(device)}


def _copy_recovery_raw(evidence_dir: Path, review_dir: Path) -> Path:
    source = evidence_dir / "preflight_evidence" / "engineering_recovery"
    destination = review_dir / "recovery_raw"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint_new_000003.pt", "original_process_new_000004.pt",
                 "independent_process_new_000004.pt"):
        item = source / name
        _require(item.is_file(), f"raw B recovery file is missing: {item}", FileNotFoundError)
        shutil.copy2(item, destination / name)
        _require((destination / name).stat().st_size == item.stat().st_size,
                 f"raw B recovery copy has a byte-size mismatch: {name}")
    return destination


def _consume_recovery_raw(recovery_dir: Path, schedule: torch.Tensor,
                          run_id: str, source_run_id: str) -> dict[str, Any]:
    cp_path = recovery_dir / "checkpoint_new_000003.pt"
    live_path = recovery_dir / "original_process_new_000004.pt"
    restored_path = recovery_dir / "independent_process_new_000004.pt"
    _require(all(path.is_file() for path in (cp_path, live_path, restored_path)),
             "raw B recovery comparison is missing a checkpoint or fourth-step state", FileNotFoundError)
    checkpoint, live, restored = (torch.load(path, map_location="cpu", weights_only=False)
                                  for path in (cp_path, live_path, restored_path))
    history = checkpoint.get("history")
    flat = _flatten_schedule(schedule)
    _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == run_id
             and checkpoint.get("completed_new_updates") == 3
             and checkpoint.get("history_count") == 3 and isinstance(history, list)
             and len(history) == 3 and checkpoint.get("optimizer_deepcopied") is True
             and checkpoint.get("next_batch_indices") is not None
             and torch.equal(checkpoint["next_batch_indices"], flat[START_BRANCH_STEP + 3])
             and all(history[index].get("sample_indices") ==
                     flat[START_BRANCH_STEP + index].tolist() for index in range(3)),
             "raw B recovery step-three prefix or next batch differs from the fixed schedule")
    _validate_checkpoint(checkpoint, run_id, source_run_id, 3, history, schedule)
    cp_steps = _optimizer_steps(checkpoint["optimizer_state"])
    live_steps, restored_steps = _optimizer_steps(live["optimizer"]), _optimizer_steps(restored["optimizer"])
    mapper_equal, mapper_delta = projected._tree_max_delta(live.get("mapper"), restored.get("mapper"))
    adam_equal, adam_delta = projected._tree_max_delta(live.get("optimizer"), restored.get("optimizer"))
    loss_delta = abs(float(live.get("loss")) - float(restored.get("loss")))
    coefficient_delta = abs(float(live.get("coefficient_loss")) -
                             float(restored.get("coefficient_loss")))
    rng_equal = (_rng_equal(live.get("rng", {}), restored.get("rng", {}))
                 and _rng_equal({"cpu": checkpoint.get("torch_cpu_rng"),
                                 "cuda": checkpoint.get("torch_cuda_rng")}, live.get("rng", {}))
                 and _rng_equal({"cpu": checkpoint.get("torch_cpu_rng"),
                                 "cuda": checkpoint.get("torch_cuda_rng")}, restored.get("rng", {})))
    expected_batch = flat[START_BRANCH_STEP + 3]
    schedule_equal = (torch.equal(live.get("sample_indices"), expected_batch)
                      and torch.equal(restored.get("sample_indices"), expected_batch))
    _require(len(cp_steps) == len(live_steps) == len(restored_steps) == 4
             and all(value == START_BRANCH_STEP + 3 for value in cp_steps)
             and all(value == START_BRANCH_STEP + 4 for value in live_steps + restored_steps)
             and mapper_equal and adam_equal
             and mapper_delta <= RECOVERY_TOLERANCE and adam_delta <= RECOVERY_TOLERANCE
             and loss_delta <= RECOVERY_TOLERANCE
             and coefficient_delta <= RECOVERY_TOLERANCE
             and rng_equal and schedule_equal
             and live.get("engineering_adam_calls") == restored.get("engineering_adam_calls") == 1,
             "raw B recovery states fail the model, Adam, loss, RNG, or batch comparison")
    return {
        "all_raw_recovery_states_verified": True,
        "step3_adam_steps": cp_steps, "live_step4_adam_steps": live_steps,
        "restored_step4_adam_steps": restored_steps,
        "mapper_max_abs_delta": mapper_delta, "adam_max_abs_delta": adam_delta,
        "total_loss_abs_delta": loss_delta, "coefficient_loss_abs_delta": coefficient_delta,
        "rng_exact": rng_equal, "schedule_exact": schedule_equal,
        "engineering_adam_calls": 5,
        "step3_history_prefix_verified": True,
        "full_step4_history_saved": False,
        "history_scope": "saved step-three prefix and fourth-step batch only; no fourth-step history was saved",
    }


def _review_saved(evidence_dir: Path, review_dir: Path,
                  config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(evidence_dir.is_dir() and (not review_dir.exists() or review_dir.is_dir()),
             "review-saved requires an existing explicit source and a repair directory", ValueError)
    identity = _assert_attempt(evidence_dir)
    source_inventory = _verify_remote_manifest(evidence_dir, identity.run_id)
    state = _json(evidence_dir / "task_state.json")
    decision = _json(evidence_dir / "final_decision.json")
    _require(state.get("state") == RUN_COMPLETED and decision.get("lifecycle_state") == RUN_COMPLETED
             and decision.get("summary", {}).get("passed_rows") == TARGET_COUNT
             and decision.get("new_scientific_updates") == ADDED_UPDATES,
             "the closed B1500 source result or fixed update count changed")
    if review_dir.exists() and (review_dir / "review_summary.json").is_file():
        prior_summary = _json(review_dir / "review_summary.json")
        _require(prior_summary.get("task_id") == TASK_ID
                 and prior_summary.get("run_id") == identity.run_id
                 and prior_summary.get("scientific_updates") == 0,
                 "existing repair directory belongs to another review or task")
    review_dir.mkdir(parents=True, exist_ok=True)
    command_replay = _command_replay(evidence_dir, review_dir, inputs)
    source = _load_source(config, inputs, "cloud")
    population, schedule = _load_saved_population(evidence_dir / "fixed_population_and_schedule.pt", source)
    recovery_dir = _copy_recovery_raw(evidence_dir, review_dir)
    recovery = _consume_recovery_raw(recovery_dir, schedule, identity.run_id,
                                     population["source_run_id"])
    recovery_summary = _json(evidence_dir / "preflight_evidence" / "engineering_recovery.json")
    _require(recovery_summary.get("status") == "PASS"
             and recovery_summary.get("engineering_adam_calls_total") == 5
             and recovery_summary.get("mapper_max_abs_delta") == recovery["mapper_max_abs_delta"]
             and recovery_summary.get("adam_max_abs_delta") == recovery["adam_max_abs_delta"]
             and recovery_summary.get("total_loss_abs_delta") == recovery["total_loss_abs_delta"]
             and recovery_summary.get("rng_exact") is recovery["rng_exact"]
             and recovery_summary.get("schedule_exact") is recovery["schedule_exact"],
             "raw B recovery values disagree with their saved engineering summary")
    summary = {
        "schema": "lightgen_g1_fit20_box_b_evidence_repair.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "status": "REVIEW_REPAIR_READY_FOR_WINDOWS_RECONSUMPTION",
        "historical_state_preserved": RUN_COMPLETED,
        "scientific_updates": 0,
        "source_manifest_recomputed": source_inventory,
        "command_replay": command_replay,
        "recovery": recovery,
        "old_scientific_decision_recomputed": False,
        "source_attempt_modified": False,
        "original_report_and_measurements_preserved": True,
    }
    _atomic_json(review_dir / "review_summary.json", summary)
    _build_review_transfer_manifest(review_dir, identity.run_id)
    return summary


def _build_review_transfer_manifest(review_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_review_receipt.json"}
    files = [{"path": path.relative_to(review_dir).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(review_dir.rglob("*"))
             if path.is_file() and path.name not in excluded and ".tmp." not in path.name]
    manifest = {"schema": "lightgen_g1_fit20_box_b_review_transfer.v1",
                "task_id": TASK_ID, "run_id": run_id,
                "file_count": len(files), "bytes_total": sum(item["bytes"] for item in files),
                "files": files}
    _atomic_json(review_dir / "transfer_manifest.json", manifest)
    return manifest


def _verify_review_transfer(review_dir: Path, run_id: str) -> dict[str, Any]:
    manifest = _json(review_dir / "transfer_manifest.json")
    receipt = _json(review_dir / "transfer_receipt.json")
    expected_remote = "outputs/20261027_lightgen_g1_fit20_box_b_convergence_v1/runs/review_repair_01"
    remote_run = str(receipt.get("remote_run_dir", "")).rstrip("/")
    _require(manifest.get("schema") == "lightgen_g1_fit20_box_b_review_transfer.v1"
             and manifest.get("task_id") == TASK_ID and manifest.get("run_id") == run_id
             and isinstance(manifest.get("files"), list)
             and receipt.get("status") == "COMPLETE" and isinstance(receipt.get("files"), list)
             and receipt.get("remote_alias") == "autodl-monthly"
             and remote_run.endswith("/" + expected_remote),
             "B repair transfer manifest or receipt does not match the fixed cloud repair path")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() not in expected,
                 "B repair manifest has unsafe or duplicate paths")
        expected[relative.as_posix()] = int(item["bytes"])
    expected["transfer_manifest.json"] = (review_dir / "transfer_manifest.json").stat().st_size
    actual: dict[str, Mapping[str, Any]] = {}
    for item in receipt["files"]:
        relative = Path(str(item.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() not in actual,
                 "B repair transfer receipt has unsafe or duplicate paths")
        actual[relative.as_posix()] = item
    _require(set(expected) == set(actual)
             and manifest.get("file_count") == len(manifest["files"])
             and manifest.get("bytes_total") == sum(item["bytes"] for item in manifest["files"]),
             "B repair transfer receipt inventory differs from its manifest")
    files = []
    for relative, remote_size in sorted(expected.items()):
        path = review_dir / Path(relative)
        item = actual[relative]
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_size
                 and int(item.get("local_size_bytes", -1)) == remote_size
                 and current == remote_size,
                 f"B repair actual file size/status differs: {relative}")
        files.append({"path": relative, "remote_bytes": remote_size,
                      "receipt_local_bytes": int(item["local_size_bytes"]),
                      "current_bytes": current})
    _require(len(expected) == manifest["file_count"] + 1,
             "B repair transfer receipt must include its remote manifest")
    return {"status": "COMPLETE", "file_count": len(files),
            "bytes_total": sum(item["current_bytes"] for item in files), "files": files}


def _verify_saved(evidence_dir: Path, config: Mapping[str, Any],
                  inputs: Mapping[str, Any], review_dir: Path) -> dict[str, Any]:
    _require(evidence_dir.is_dir(), f"B convergence evidence directory is missing: {evidence_dir}",
             FileNotFoundError)
    transfer = _verify_transfer(evidence_dir)
    identity = _assert_attempt(evidence_dir)
    _require(_json(evidence_dir / "transfer_manifest.json").get("run_id") == identity.run_id,
             "B convergence original transfer manifest belongs to another attempt")
    repair_transfer = _verify_review_transfer(review_dir, identity.run_id)
    decision = _json(evidence_dir / "final_decision.json")
    state = _json(evidence_dir / "task_state.json")
    lifecycle = _json(evidence_dir / "research_lifecycle_state.json")
    _require(state.get("task_id") == TASK_ID and state.get("run_id") == identity.run_id
             and decision.get("task_id") == TASK_ID and decision.get("run_id") == identity.run_id
             and lifecycle.get("task_id") == TASK_ID and lifecycle.get("run_id") == identity.run_id
             and lifecycle.get("receipt_consumed") is True
             and decision.get("lifecycle_state") == state.get("state")
             and decision.get("new_scientific_updates") == ADDED_UPDATES
             and decision.get("branch_terminal_step") == TERMINAL_BRANCH_STEP
             and decision.get("cumulative_mapper_updates") == 11500,
             "B convergence decision, state, and lifecycle record disagree")
    source = _load_source(config, inputs, "windows")
    windows_review = _validate_windows_review(config, source)
    _require(decision.get("source_run_id") == source["identity"].run_id
             and decision.get("windows_review_status") == windows_review["status"],
             "B convergence source or review lineage differs from the passed predecessor review")
    population, schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", source)
    source_checkpoint = torch.load(evidence_dir / "source_B500_checkpoint.pt",
                                   map_location="cpu", weights_only=False)
    mapper_equal, mapper_delta = projected._tree_max_delta(
        source_checkpoint.get("mapper_state"), source["checkpoint"]["mapper_state"])
    adam_equal, adam_delta = projected._tree_max_delta(
        source_checkpoint.get("optimizer_state"), source["checkpoint"]["optimizer_state"])
    _require(mapper_equal and adam_equal and mapper_delta == 0.0 and adam_delta == 0.0
             and source_checkpoint.get("completed_updates") == START_BRANCH_STEP
             and source_checkpoint.get("arm") == "B"
             and source_checkpoint.get("run_id") == source["identity"].run_id,
             "copied B500 source model/Adam does not match the explicit predecessor checkpoint")
    history_payload = _json(evidence_dir / "training_history.json")
    history = history_payload.get("updates", [])
    _require(history_payload.get("task_id") == TASK_ID
             and history_payload.get("run_id") == identity.run_id
             and history_payload.get("source_run_id") == source["identity"].run_id
             and len(history) == ADDED_UPDATES
             and history_payload.get("completed_new_updates") == ADDED_UPDATES,
             "B convergence committed training history is incomplete")
    for index, item in enumerate(history):
        _require(item.get("completed_update") == index + 1
                 and item.get("branch_step") == START_BRANCH_STEP + index + 1
                 and item.get("sample_indices") == _flatten_schedule(schedule)[
                     START_BRANCH_STEP + index].tolist()
                 and item.get("arm") == "B"
                 and item.get("loss_timing") == "computed_before_this_adam_update",
                 f"B convergence training history row {index + 1} has wrong schedule/timing")
    repair_summary = _json(review_dir / "review_summary.json")
    _require(repair_summary.get("schema") == "lightgen_g1_fit20_box_b_evidence_repair.v1"
             and repair_summary.get("task_id") == TASK_ID
             and repair_summary.get("run_id") == identity.run_id
             and repair_summary.get("scientific_updates") == 0
             and repair_summary.get("status") == "REVIEW_REPAIR_READY_FOR_WINDOWS_RECONSUMPTION",
             "B convergence zero-update repair evidence is missing or belongs to another attempt")
    source_review = _load_source(config, inputs, "windows")
    _load_saved_population(evidence_dir / "fixed_population_and_schedule.pt", source_review)
    measured: dict[int, dict[str, Any]] = {}
    for step in config["training"]["fixed_checkpoint_steps"]:
        path = evidence_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pt"
        _require(path.is_file(), f"B convergence fixed checkpoint {step} is missing", FileNotFoundError)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        prefix = history[:step]
        _validate_checkpoint(checkpoint, identity.run_id, source["identity"].run_id,
                             int(step), prefix, schedule)
        if step in config["training"]["observation_steps"]:
            observation_path = evidence_dir / "measurements" / f"step_{step:04d}.pt"
            replay_path = review_dir / "command_replays" / f"step_{step:04d}.pt"
            _require(replay_path.is_file(), f"exact CUDA replay for step {step} is missing", FileNotFoundError)
            cuda_replay = torch.load(replay_path, map_location="cpu", weights_only=False)
            observed = _consume_observation(observation_path, checkpoint, population, inputs,
                                           int(step), cuda_replay)
            measured[int(step)] = observed
    terminal_result = measured.get(ADDED_UPDATES)
    _require(terminal_result is not None
             and _numeric_tree_matches(decision.get("summary"), terminal_result["summary"])
             and _numeric_tree_matches(decision.get("rows"), terminal_result["rows"]),
             "B convergence final decision differs from independent raw measurement reconsumption")
    passed = bool(terminal_result["summary"]["all_rows_passed"])
    expected_state = RUN_COMPLETED if passed else RUN_VALID_FAILED
    _require(decision.get("lifecycle_state") == expected_state
             and decision.get("summary", {}).get("passed_rows") == terminal_result["summary"]["passed_rows"]
             and decision.get("adam_parameter_steps") == [TERMINAL_BRANCH_STEP] * 4,
             "B convergence final scientific status or Adam steps do not follow recomputed results")
    report_path = evidence_dir / "final_report.md"
    report = report_path.read_text(encoding="utf-8")
    sections = ("## 结果与状态", "## 科学认识", "## 目标缺口",
                "## 工程完成与验证", "## 决策影响与下一步")
    _require(all(section in report for section in sections),
             "B convergence report does not contain the five required sections")
    recovery = _consume_recovery_raw(review_dir / "recovery_raw", schedule, identity.run_id,
                                     source["identity"].run_id)
    _require(_numeric_tree_matches(repair_summary.get("recovery"), recovery)
             and repair_summary.get("recovery", {}).get("full_step4_history_saved") is False,
             "Windows raw recovery reconsumption differs from the cloud review summary")
    receipt = {
        "schema": "lightgen_g1_fit20_box_b_convergence_windows_reconsumption.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "source_run_id": source["identity"].run_id,
        "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
        "scientific_status": expected_state,
        "new_scientific_updates": ADDED_UPDATES,
        "source_B500_mapper_and_adam_exact": True,
        "training_history_and_extended_schedule_verified": True,
        "all_20_rows_recomputed_at_steps": sorted(measured),
        "terminal_all_six_metrics_recomputed": True,
        "file_size_reconciliation": transfer,
        "windows_command_replay": terminal_result["windows_cpu_to_optical_response_relative_error_max"],
        "terminal_summary": terminal_result["summary"],
        "repair_file_size_reconciliation": repair_transfer,
        "recovery_raw_states_recomputed": recovery,
        "exact_cuda_command_replay_consumed": True,
        "scientific_updates_during_repair": 0,
    }
    _atomic_json(review_dir / "windows_review_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "execute", "resume", "measure", "verify-saved",
                                 "preflight-worker", "review-saved", "verify-review"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path,
                        help="explicit recovery or measurement checkpoint")
    parser.add_argument("--step", type=int,
                        help="predeclared new-update observation step for measure")
    parser.add_argument("--evidence-dir", type=Path,
                        help="explicit saved source attempt for saved-evidence review")
    parser.add_argument("--review-dir", type=Path,
                        help="explicit new repair directory for review-saved or verify-review")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.mode == "preflight-worker":
        _require(args.checkpoint is not None and args.worker_output is not None,
                 "preflight-worker requires explicit checkpoint and worker output", ValueError)
        _preflight_worker(config_path, args.checkpoint.resolve(), args.worker_output.resolve())
        result: Mapping[str, Any] = {"status": "ENGINEERING_RECOVERY_STEP_COMPLETE"}
    elif args.mode == "preflight":
        _require(args.checkpoint is None and args.step is None,
                 "preflight does not accept checkpoint or step", ValueError)
        result = _preflight(config_path, config)
    elif args.mode == "execute":
        _require(args.checkpoint is None and args.step is None,
                 "execute starts only from the fixed B500 source checkpoint", ValueError)
        result = _execute(config_path, config)
    elif args.mode == "resume":
        _require(args.checkpoint is not None and args.step is None,
                 "resume requires an explicit --checkpoint", ValueError)
        result = _execute(config_path, config, resume_checkpoint=args.checkpoint.resolve())
    elif args.mode == "measure":
        _require(args.checkpoint is not None and args.step is not None,
                 "measure requires explicit --checkpoint and --step", ValueError)
        result = _measure_saved_checkpoint(config, args.checkpoint.resolve(), args.step)
    elif args.mode == "review-saved":
        _require(args.evidence_dir is not None and args.review_dir is not None
                 and args.checkpoint is None and args.step is None,
                 "review-saved requires explicit --evidence-dir and a new --review-dir", ValueError)
        _set_numerics()
        inputs = _load_inputs(config)
        result = _review_saved(args.evidence_dir.resolve(), args.review_dir.resolve(), config, inputs)
    else:
        _require(args.evidence_dir is not None and args.review_dir is not None
                 and args.checkpoint is None and args.step is None,
                 "verify-saved and verify-review require explicit --evidence-dir and --review-dir",
                 ValueError)
        _set_numerics()
        inputs = _load_inputs(config)
        result = _verify_saved(args.evidence_dir.resolve(), config, inputs, args.review_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
