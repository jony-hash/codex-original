"""Compare coefficient-only and preclip-box objectives on known-reachable G1 fit20."""

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
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as optics


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_PROJECTED_FIT20_BOX_AB_V1"
SCHEMA = "lightgen_g1_projected_fit20_box_ab.v1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_PROJECTED_FIT20_BOX_AB_20261027"
MODULE_PATH = f"scripts.research.lightgen.{Path(__file__).stem}"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_projected_fit20_box_ab_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261027_lightgen_g1_projected_fit20_box_ab_v1/runs"
PREFLIGHT_DIR = RUNS_DIR / "preflight_01"
ATTEMPT_DIR = RUNS_DIR / "attempt_01"
ARM_NAMES = ("A", "B")
TARGET_COUNT = 20
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
REAL_DIM = 200
IMAGE_SHAPE = (200, 200)
GRIDS = (2048, 4096)
GATE_LIMIT = 1.0e-4
RECOVERY_TOLERANCE = 5.0e-8
METRIC_KEYS = ("coefficient_2048", "coefficient_4096", "native_cmos_2048",
               "native_cmos_4096", "cross_grid_coefficient", "cross_grid_native_cmos")


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


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(), "fit20 task uses its fixed configuration", ValueError)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "fit20 configuration must be a mapping", ValueError)
    _validate_config(value)
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == SCHEMA and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY
             and config.get("evidence_tier") == "diagnostic_only",
             "fit20 task identity or authority changed", ValueError)
    source = config["source"]
    _require(source.get("projected_mapping_config") ==
             "configs/research/lightgen/g1_slm_projected_mapping_v2.yaml"
             and source.get("fixed_population_and_schedule") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/fixed_population_and_schedule.pt"
             and source.get("terminal_mapper_checkpoint") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/terminal_checkpoint.pt"
             and source.get("terminal_training_history") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/training_history.json"
             and source.get("terminal_final_decision") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/final_decision.json"
             and source.get("terminal_task_state") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/task_state.json"
             and source.get("validation_reachability_witnesses") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/preflight_evidence/validation_reachability_witnesses.pt"
             and source.get("response_projector_evidence") ==
             "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/preflight_evidence/fixed_response_and_projector.pt",
             "fit20 explicit source paths changed", ValueError)
    population = config["population"]
    _require(population == {
        "role": "known_reachable_fit20", "source_role": "projected_mapping_validation",
        "source_rows_per_class": 200, "selected_slots_per_class": [0, 1],
        "selected_row_rule": "200*y + slot", "count": 20, "classes": 10,
        "schedule_seed": 2026102781, "rng_seed": 2026102780,
        "cycles": 250, "visits_per_class_per_cycle": 2,
    }, "fit20 population or paired schedule changed", ValueError)
    training = config["training"]
    _require(training.get("updates_per_arm") == 500 and training.get("arms") == ["A", "B"]
             and training.get("box_lambda") == 5.0 and training.get("logical_batch_size") == 10
             and training.get("arm_a_objective") == "full_optics_phase_invariant_coefficient_loss"
             and training.get("arm_b_objective") ==
             "full_optics_phase_invariant_coefficient_loss_plus_5x_preclip_box_loss"
             and training.get("samples_per_class_per_batch") == 1
             and training.get("optimizer") == "Adam" and training.get("learning_rate") == 1.0e-4
             and training.get("betas") == [0.9, 0.999] and training.get("epsilon") == 1.0e-8
             and training.get("weight_decay") == 0.0 and training.get("progress_interval_steps") == 25
             and training.get("checkpoint_steps") == [0, 250, 500]
             and training.get("observation_steps") == [0, 250, 500]
             and training.get("selected_terminal_step") == 500
             and training.get("mapper_precision") == "float32"
             and training.get("fixed_projection_precision") == "float64"
             and training.get("optical_precision") == "float64_complex128"
             and training.get("amp") is False and training.get("tf32") is False,
             "fit20 training contract changed", ValueError)
    measurement = config["measurement"]
    _require(measurement.get("grids") == list(GRIDS)
             and measurement.get("relative_error_max") == GATE_LIMIT
             and measurement.get("coefficient_phase_alignment") == "one_global_phase"
             and measurement.get("native_cmos") == "raw_w_per_m2_no_normalization"
             and measurement.get("cmos_power_matching") is False
             and measurement.get("cmos_gain_adjustment") is False
             and measurement.get("six_metrics") == list(METRIC_KEYS),
             "fit20 measurement contract changed", ValueError)
    _require(config["output"].get("experiment_root") ==
             "outputs/20261027_lightgen_g1_projected_fit20_box_ab_v1/runs/attempt_01"
             and config["output"].get("preflight_root") ==
             "outputs/20261027_lightgen_g1_projected_fit20_box_ab_v1/runs/preflight_01"
             and config["output"].get("windows_backup_root") ==
             "outputs/20261027_lightgen_g1_projected_fit20_box_ab_v1/runs/autodl_backup_attempt_01",
             "fit20 output paths changed", ValueError)


def _set_numerics() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen_config = projected._load_config(_resolve(config["source"]["projected_mapping_config"]))
    frozen_inputs = pilot._validate_source_inputs(projected._legacy_validation_config(frozen_config))
    source = config["source"]
    paths = {key: _resolve(value) for key, value in source.items()
             if key != "projected_mapping_config"}
    for path in paths.values():
        _require(path.is_file(), f"explicit fit20 input is missing: {path}", FileNotFoundError)
    decision = _json(paths["terminal_final_decision"])
    state = _json(paths["terminal_task_state"])
    history_payload = _json(paths["terminal_training_history"])
    _require(decision.get("task_id") == projected.TASK_ID
             and decision.get("continuous_final_gate", {}).get("status") == "VALID_FAILED"
             and state.get("task_id") == projected.TASK_ID
             and state.get("state") == RUN_VALID_FAILED,
             "the predecessor must remain the closed 5,000-step VALID_FAILED task")
    fixed_population, old_schedule = projected._load_population(paths["fixed_population_and_schedule"])
    history = history_payload.get("updates")
    _require(history_payload.get("task_id") == projected.TASK_ID
             and history_payload.get("run_id") == state.get("run_id")
             and isinstance(history, list) and len(history) == 5000
             and state.get("completed_new_updates") == 5000,
             "predecessor's 5,000-step training history or terminal state is incomplete")
    _require(all(item.get("completed_update") == i + 1
                 and item.get("sample_indices") == old_schedule[i // 1000, i % 1000].tolist()
                 for i, item in enumerate(history)),
             "predecessor training history does not match its saved five-pass schedule")
    terminal = torch.load(paths["terminal_mapper_checkpoint"], map_location="cpu", weights_only=False)
    _require(terminal.get("schema") == projected.CHECKPOINT_SCHEMA
             and terminal.get("task_id") == projected.TASK_ID
             and terminal.get("run_id") == state.get("run_id")
             and terminal.get("completed_updates") == 5000
             and terminal.get("checkpoint_step") == 5000
             and terminal.get("history_count") == 5000
             and terminal.get("optimizer_deepcopied") is True
             and set(terminal.get("mapper_state", {})) == {
                 "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias",
             }, "predecessor terminal mapper checkpoint is not its explicit 5,000-step endpoint")
    adam_steps = [int(torch.as_tensor(value["step"]).item())
                  for value in terminal["optimizer_state"]["state"].values() if "step" in value]
    _require(len(adam_steps) == 4 and all(step == 5000 for step in adam_steps),
             "predecessor terminal Adam state does not contain four step-5,000 tensors")
    response_evidence = torch.load(paths["response_projector_evidence"], map_location="cpu", weights_only=False)
    _require(response_evidence.get("task_id") == projected.TASK_ID
             and response_evidence.get("real_imag_order") == "per_port_interleaved_real_imag"
             and torch.equal(response_evidence["scale"], frozen_inputs["scale"]),
             "fixed C response/projector evidence does not match the frozen modal scale")
    witness = torch.load(paths["validation_reachability_witnesses"], map_location="cpu", weights_only=False)
    _require(witness.get("task_id") == projected.TASK_ID
             and torch.is_tensor(witness.get("amplitude_commands"))
             and tuple(witness["amplitude_commands"].shape) == (20, 1, 200, 200)
             and len(witness.get("records", [])) == 20,
             "saved 20-command positive-control witness has wrong identity or shape")
    return {
        **frozen_inputs,
        "frozen_config": frozen_config,
        "frozen_inputs": frozen_inputs,
        "fixed_population": fixed_population,
        "old_schedule": old_schedule,
        "terminal_mapper": terminal,
        "training_history": history,
        "old_decision": decision,
        "old_task_state": state,
        "witness": witness,
        "response": response_evidence["response_matrix"].contiguous(),
        "matrix": response_evidence["normalized_real_matrix"].contiguous(),
        "inverse": response_evidence["full_right_inverse"].contiguous(),
        "source_paths": paths,
    }


def _fit_population(inputs: Mapping[str, Any]) -> dict[str, Any]:
    validation = inputs["fixed_population"]["validation"]
    indices = [200 * label + slot for label in range(CLASS_COUNT) for slot in (0, 1)]
    idx = torch.tensor(indices, dtype=torch.long)
    targets = validation["coefficients"][idx].detach().cpu().to(torch.complex128).contiguous()
    labels = validation["labels"][idx].detach().cpu().long().contiguous()
    proposals = validation["proposal_ids"][idx].detach().cpu().long().contiguous()
    _require(tuple(targets.shape) == (TARGET_COUNT, PORT_COUNT)
             and torch.equal(labels, torch.arange(CLASS_COUNT).repeat_interleave(2))
             and torch.isfinite(targets.real).all().item()
             and torch.isfinite(targets.imag).all().item(),
             "fixed fit20 target population is malformed")
    rows = [{"fit20_index": i, "source_validation_index": source_index,
             "label": int(labels[i]), "slot": i % 2, "proposal_id": int(proposals[i]),
             "population_role": "known_reachable_fit20"}
            for i, source_index in enumerate(indices)]
    return {"schema": "lightgen_g1_projected_fit20_population.v1", "task_id": TASK_ID,
            "role": "known_reachable_fit20", "source_role": "projected_mapping_validation",
            "source_validation_indices": torch.tensor(indices, dtype=torch.long),
            "coefficients": targets, "labels": labels, "proposal_ids": proposals, "rows": rows}


def _build_schedule(population: Mapping[str, Any], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = torch.empty((250, 2, CLASS_COUNT), dtype=torch.long)
    for cycle in range(250):
        for label in range(CLASS_COUNT):
            order = torch.randperm(2, generator=generator)
            schedule[cycle, :, label] = label * 2 + order
    classes_per_batch = schedule.reshape(500, CLASS_COUNT) // 2
    _require(tuple(schedule.shape) == (250, 2, CLASS_COUNT)
             and torch.equal(classes_per_batch,
                             torch.arange(CLASS_COUNT).expand(500, CLASS_COUNT)),
             "fit20 schedule is not one sample per class per update")
    for cycle in range(250):
        for label in range(CLASS_COUNT):
            _require(set(schedule[cycle, :, label].tolist()) == {label * 2, label * 2 + 1},
                     f"fit20 schedule does not visit both fixed samples for class {label}")
    del population
    return schedule


def _run_identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    semantics = json.dumps({"task_id": TASK_ID, "authority": AUTHORITY,
                             "config": config, "source_checkpoint_step": 5000,
                             "population_role": "known_reachable_fit20"},
                            sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:projected_fit20_box_ab",
        semantic_authority_id=AUTHORITY, data_role="KNOWN_REACHABLE_G1_FIT20_REPEATED_FIT",
        config_identity=dict(config),
        model_identity={"frozen_optics": "C_seed_2026101501_step_45000",
                        "warm_start": "projected_mapping_v2_terminal_step_5000",
                        "mapper": "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid+fixed_projection",
                        "arm_a": "full_optics_coefficient_loss",
                        "arm_b": "full_optics_coefficient_loss+5x_preclip_box_loss"},
        optimizer_identity={"kind": "Adam", "learning_rate": 1.0e-4, "betas": [0.9, 0.999],
                            "epsilon": 1.0e-8, "weight_decay": 0.0, "updates_per_arm": 500},
        sampler_identity={"population": "fixed projected validation rows 200*y+{0,1}",
                          "paired_schedule_seed": 2026102781, "rng_seed": 2026102780},
        execution_source_semantics_id=semantics,
    )


def _load_control(preflight_dir: Path) -> tuple[ResearchRunControl, PreflightReceipt]:
    identity_payload = _json(preflight_dir / "preflight_identity.json")
    _require(identity_payload.get("task_id") == TASK_ID, "fit20 preflight belongs to another task")
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    receipt = PreflightReceipt.from_mapping(_json(preflight_dir / "preflight_receipt.json"))
    _require(identity.task_id == TASK_ID and receipt.authorization_id == AUTHORITY
             and receipt.source_semantics_id == identity.execution_source_semantics_id,
             "fit20 identity and READY receipt do not match")
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    return control, receipt


def _assert_attempt(attempt_dir: Path) -> ResearchRunIdentity:
    preflight = _json(attempt_dir / "preflight_evidence" / "preflight_identity.json")
    expected = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    attempt = ResearchRunIdentity.from_mapping(_json(attempt_dir / "run_identity.json"))
    expected = expected.with_scientific_attempt(f"{expected.run_id}:attempt_01")
    _require(preflight.get("task_id") == TASK_ID and attempt.execution_semantics() == expected.execution_semantics(),
             "fit20 attempt differs from its READY task identity")
    receipt = PreflightReceipt.from_mapping(
        _json(attempt_dir / "preflight_evidence" / "preflight_receipt.json"))
    _require(receipt.authorization_id == AUTHORITY and receipt.status == "READY"
             and receipt.source_semantics_id == expected.execution_source_semantics_id,
             "fit20 attempt does not contain its consumed READY receipt")
    return attempt


def _box_loss(preclip: torch.Tensor) -> torch.Tensor:
    return (torch.relu(-preclip).square() + torch.relu(preclip - 1.0).square()).mean()


def _combined_loss(arm: str, coefficient_loss: torch.Tensor,
                   box_loss: torch.Tensor) -> torch.Tensor:
    _require(arm in ARM_NAMES, "unknown fit20 objective arm")
    return coefficient_loss if arm == "A" else coefficient_loss + 5.0 * box_loss


def _box_contract_check(device: torch.device) -> dict[str, Any]:
    value = torch.full((1, 40_000), 0.5, dtype=torch.float64, device=device)
    value[0, 0], value[0, 1], value[0, 2], value[0, 3] = -0.25, 0.0, 1.0, 1.25
    value.requires_grad_(True)
    result = _box_loss(value)
    gradient, = torch.autograd.grad(result, value)
    numpy_value = value.detach().cpu().numpy()
    numpy_loss = float((np.maximum(-numpy_value, 0.0) ** 2 + np.maximum(numpy_value - 1.0, 0.0) ** 2).mean())
    numpy_gradient = 2.0 * (numpy_value - np.clip(numpy_value, 0.0, 1.0)) / PIXEL_COUNT
    _require(abs(float(result.item()) - numpy_loss) <= 1.0e-16
             and np.allclose(gradient.detach().cpu().numpy(), numpy_gradient, rtol=0.0, atol=1.0e-16)
             and float(gradient[0, 0]) < 0.0 and float(gradient[0, 3]) > 0.0
             and float(gradient[0, 1]) == 0.0 and float(gradient[0, 2]) == 0.0,
             "box loss or its lower/upper-bound derivative disagrees with independent NumPy reference")
    return {"torch_loss": float(result.item()), "independent_numpy_loss": numpy_loss,
            "gradient_formula": "2*(v-clip(v,0,1))/40000",
            "independent_numpy_gradient_max_abs_error": float(np.max(np.abs(
                gradient.detach().cpu().numpy() - numpy_gradient))),
            "lower_bound_checked": True, "upper_bound_checked": True}


def _optimizer(mapper: torch.nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(mapper.parameters(), lr=1.0e-4, betas=(0.9, 0.999),
                            eps=1.0e-8, weight_decay=0.0)


def _make_mapper(inputs: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    return projected._new_mapper(device, inputs["terminal_mapper"]["mapper_state"])


def _batch(population: Mapping[str, Any], schedule: torch.Tensor, update_index: int) -> torch.Tensor:
    _require(0 <= update_index < 500, "fit20 update index is outside the fixed 500-step schedule")
    indices = schedule.reshape(500, CLASS_COUNT)[update_index]
    values = population["coefficients"][indices]
    labels = population["labels"][indices]
    _require(tuple(values.shape) == (CLASS_COUNT, PORT_COUNT)
             and torch.equal(labels, torch.arange(CLASS_COUNT)),
             "fit20 batch is not one fixed coefficient per class")
    return values


def _training_step(
    arm: str, mapper: torch.nn.Module, optimizer: torch.optim.Adam,
    model: optics.SmallMatchedAdapter, population: Mapping[str, Any], schedule: torch.Tensor,
    matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor, device: torch.device,
    update_index: int, *, audit: bool = False,
) -> dict[str, Any]:
    _require(arm in ARM_NAMES, "unknown fit20 arm")
    coefficients = _batch(population, schedule, update_index)
    target = coefficients.to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(coefficients, scale, device)
    before = [p.detach().clone() for p in mapper.parameters()] if audit else None
    optimizer.zero_grad(set_to_none=True)
    raw = mapper(features)
    amplitude, projection = projected._project_from_raw(
        raw, target, matrix, inverse, scale, include_preclip=True,
    )
    preclip = projection["preclip_amplitude"]
    output = projected._forward_amplitude(model, amplitude)
    coefficient_losses = pilot._phase_invariant_loss(output["coefficients"], target)
    coefficient_loss = coefficient_losses.mean()
    box = _box_loss(preclip.reshape(CLASS_COUNT, PIXEL_COUNT))
    total = _combined_loss(arm, coefficient_loss, box)
    _require(torch.isfinite(total).item() and torch.isfinite(box).item(),
             "fit20 full-production objective is nonfinite")
    total.backward()
    gradients = {}
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"fit20 mapper gradient is absent or nonfinite: {name}")
        gradients[name] = float(torch.linalg.vector_norm(parameter.grad.detach().double()).item())
    optimizer.step()
    if before is not None:
        _require(all(not torch.equal(old, new.detach()) for old, new in
                     zip(before, mapper.parameters(), strict=True)),
                 "one or more fit20 mapper parameter tensors did not update")
    return {
        "completed_update": update_index + 1,
        "sample_indices": schedule.reshape(500, CLASS_COUNT)[update_index].tolist(),
        "arm": arm, "coefficient_loss": float(coefficient_loss.detach().cpu()),
        "box_loss": float(box.detach().cpu()), "total_loss": float(total.detach().cpu()),
        "coefficient_relative_error_mean": float(torch.sqrt(coefficient_losses.detach()).mean().cpu()),
        "coefficient_relative_error_rms": float(torch.sqrt(coefficient_losses.detach().mean()).cpu()),
        "clipped_fraction_mean": float(projection["clipped_fraction"].mean().detach().cpu()),
        "preclip_min": float(projection["preclip_min"].min().detach().cpu()),
        "preclip_max": float(projection["preclip_max"].max().detach().cpu()),
        "gradient_norms": gradients,
    }


def _snapshot_rng() -> dict[str, Any]:
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]}


def _checkpoint_payload(arm: str, mapper: torch.nn.Module, optimizer: torch.optim.Adam,
                        run_id: str, completed: int, history: Sequence[Mapping[str, Any]],
                        schedule: torch.Tensor) -> dict[str, Any]:
    cursor = schedule.reshape(500, CLASS_COUNT)[completed].clone() if completed < 500 else None
    return {"schema": "lightgen_g1_projected_fit20_box_ab_checkpoint.v1", "task_id": TASK_ID,
            "run_id": run_id, "arm": arm, "source_mapper_step": 5000,
            "completed_updates": completed, "checkpoint_step": completed,
            "next_batch_indices": cursor, "history_count": len(history),
            "mapper_state": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "torch_cpu_rng": torch.random.get_rng_state().clone(),
            "torch_cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
            "schedule_identity": "fit20_paired_schedule_seed_2026102781:500x10",
            "schedule_prefix_length": completed,
            "optimizer_deepcopied": True}


def _save_checkpoint(path: Path, arm: str, mapper: torch.nn.Module, optimizer: torch.optim.Adam,
                     run_id: str, completed: int, history: Sequence[Mapping[str, Any]],
                     schedule: torch.Tensor) -> dict[str, Any]:
    before = _snapshot_rng()
    _atomic_torch(path, _checkpoint_payload(arm, mapper, optimizer, run_id, completed, history, schedule))
    after = _snapshot_rng()
    _require(torch.equal(before["cpu"], after["cpu"])
             and all(torch.equal(a, b) for a, b in zip(before["cuda"], after["cuda"], strict=True)),
             "checkpoint serialization changed process RNG")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id
             and saved.get("arm") == arm and saved.get("completed_updates") == completed
             and saved.get("history_count") == len(history)
             and saved.get("optimizer_deepcopied") is True,
             "fit20 checkpoint failed serialized identity/state readback")
    return saved


def _restore_checkpoint(path: Path, arm: str, mapper: torch.nn.Module,
                        optimizer: torch.optim.Adam, run_id: str,
                        schedule: torch.Tensor) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    _require(path.is_file(), f"explicit fit20 checkpoint is missing: {path}", FileNotFoundError)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == "lightgen_g1_projected_fit20_box_ab_checkpoint.v1"
             and saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id
             and saved.get("arm") == arm and saved.get("optimizer_deepcopied") is True,
             "explicit fit20 checkpoint identity is invalid")
    completed = int(saved.get("completed_updates", -1))
    _require(0 <= completed <= 500 and saved.get("checkpoint_step") == completed
             and saved.get("schedule_prefix_length") == completed
             and saved.get("schedule_identity") == "fit20_paired_schedule_seed_2026102781:500x10",
             "explicit fit20 checkpoint cursor is invalid")
    expected = schedule.reshape(500, CLASS_COUNT)[completed] if completed < 500 else None
    recorded = saved.get("next_batch_indices")
    _require((expected is None and recorded is None) or
             (expected is not None and torch.is_tensor(recorded) and torch.equal(expected, recorded)),
             "fit20 checkpoint next batch differs from the fixed schedule")
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    torch.random.set_rng_state(saved["torch_cpu_rng"].cpu())
    torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    optimizer_steps = [int(torch.as_tensor(item["step"]).item()) for item in optimizer.state.values()
                       if isinstance(item, Mapping) and "step" in item]
    if completed:
        _require(len(optimizer_steps) == len(list(mapper.parameters()))
                 and all(step == completed for step in optimizer_steps),
                 "fit20 Adam step count differs from checkpoint")
    return completed, [], saved


def _write_history(path: Path, arm: str, run_id: str, history: Sequence[Mapping[str, Any]],
                   state: str) -> None:
    _atomic_json(path, {"schema": "lightgen_g1_projected_fit20_training_history.v1",
                        "task_id": TASK_ID, "run_id": run_id, "arm": arm,
                        "source_mapper_updates": 5000, "new_update_budget": 500,
                        "completed_new_updates": len(history), "state": state,
                        "updates": list(history)})


def _tree_max_delta(left: Any, right: Any) -> tuple[bool, float]:
    return projected._tree_max_delta(left, right)


def _save_fixed_matrix(directory: Path, inputs: Mapping[str, Any]) -> None:
    _atomic_torch(directory / "fixed_response_and_projector.pt", {
        "schema": "lightgen_g1_fit20_fixed_response_projector.v1", "task_id": TASK_ID,
        "response_matrix": inputs["response"].detach().cpu().contiguous(),
        "normalized_real_matrix": inputs["matrix"].detach().cpu().double().contiguous(),
        "full_right_inverse": inputs["inverse"].detach().cpu().double().contiguous(),
        "scale": inputs["scale"].detach().cpu().double().contiguous(),
        "real_imag_order": "per_port_interleaved_real_imag",
        "source_response_projector": str(inputs["source_paths"]["response_projector_evidence"]),
    })


def _load_matrix(directory: Path, inputs: Mapping[str, Any], device: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = directory / "fixed_response_and_projector.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "lightgen_g1_fit20_fixed_response_projector.v1"
             and payload.get("task_id") == TASK_ID
             and payload.get("real_imag_order") == "per_port_interleaved_real_imag",
             "fit20 response/projector evidence identity changed")
    response = payload["response_matrix"].to(device=device, dtype=torch.complex128)
    matrix = payload["normalized_real_matrix"].to(device=device, dtype=torch.float64)
    inverse = payload["full_right_inverse"].to(device=device, dtype=torch.float64)
    scale = payload["scale"].to(device=device, dtype=torch.float64)
    expected = inputs["response"].to(device=device, dtype=torch.complex128)
    expected_matrix = projected._normalized_response(expected, inputs["scale"].to(device))
    identity_error = float((matrix @ inverse - torch.eye(REAL_DIM, device=device, dtype=torch.float64))
                          .abs().max().item())
    reference_inverse = matrix.T @ torch.linalg.solve(
        matrix @ matrix.T, torch.eye(REAL_DIM, device=device, dtype=torch.float64))
    inverse_error = float((inverse - reference_inverse).abs().max().item())
    _require(torch.equal(response, expected)
             and torch.equal(scale, inputs["scale"].to(device=device, dtype=torch.float64))
             and tuple(matrix.shape) == (REAL_DIM, PIXEL_COUNT)
             and tuple(inverse.shape) == (PIXEL_COUNT, REAL_DIM)
             and float((matrix - expected_matrix).abs().max().item()) <= 1.0e-14
             and identity_error <= 1.0e-10 and inverse_error <= 1.0e-10,
             "fit20 fixed matrix, scale, or full Moore-Penrose inverse is inconsistent")
    return response, matrix, inverse, scale


def _save_population(path: Path, population: Mapping[str, Any], schedule: torch.Tensor) -> None:
    _atomic_torch(path, {**dict(population), "training_schedule": schedule.detach().cpu().clone()})


def _load_saved_population(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "lightgen_g1_projected_fit20_population.v1"
             and payload.get("task_id") == TASK_ID
             and payload.get("role") == "known_reachable_fit20"
             and tuple(payload["coefficients"].shape) == (TARGET_COUNT, PORT_COUNT)
             and torch.equal(payload["labels"], torch.arange(CLASS_COUNT).repeat_interleave(2)),
             "saved fit20 population identity or dimensions changed")
    schedule = payload.pop("training_schedule")
    _require(torch.is_tensor(schedule) and schedule.dtype == torch.long
             and tuple(schedule.shape) == (250, 2, CLASS_COUNT)
             and torch.equal(schedule, _build_schedule(payload, 2026102781)),
             "saved fit20 paired schedule differs from its fixed seed")
    return payload, schedule


def _witness_commands(inputs: Mapping[str, Any], population: Mapping[str, Any]) -> torch.Tensor:
    witness = inputs["witness"]
    records = witness["records"]
    expected_indices = population["source_validation_indices"].tolist()
    by_index: dict[int, int] = {}
    for position, row in enumerate(records):
        key = int(row["validation_index"])
        _require(key not in by_index, "positive-control witness has duplicate validation rows")
        by_index[key] = position
    _require(set(by_index) == set(expected_indices),
             "positive-control witness rows do not match the fixed fit20 targets")
    commands = witness["amplitude_commands"]
    selected = torch.stack([commands[by_index[index]] for index in expected_indices]).float().contiguous()
    _require(tuple(selected.shape) == (TARGET_COUNT, 1, 200, 200)
             and torch.isfinite(selected).all().item()
             and float(selected.min()) >= 0.0 and float(selected.max()) <= 1.0,
             "saved positive-control commands are invalid")
    return selected


def _measure_commands(population: Mapping[str, Any], amplitudes: torch.Tensor,
                      inputs: Mapping[str, Any], device: torch.device,
                      grids: Sequence[int] = GRIDS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    direct: dict[int, torch.Tensor] = {}
    for grid in grids:
        model = pilot._new_optical_model(inputs["checkpoint"], int(grid), device)
        mapped_coefficients: list[torch.Tensor] = []
        mapped_cmos: list[torch.Tensor] = []
        direct_cmos: list[torch.Tensor] = []
        for start in range(0, TARGET_COUNT, 1):
            stop = start + 1
            target = population["coefficients"][start:stop].to(device=device, dtype=torch.complex128)
            command = amplitudes[start:stop].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                actual = projected._forward_amplitude(model, command)
                reference = projected._forward_target(model, target)
            mapped_coefficients.append(actual["coefficients"].detach().cpu())
            mapped_cmos.append(actual["raw_cmos"].detach().cpu())
            direct_cmos.append(reference["raw_cmos"].detach().cpu())
        outputs[int(grid)] = {"coefficients": torch.cat(mapped_coefficients),
                              "raw_cmos": torch.cat(mapped_cmos)}
        direct[int(grid)] = torch.cat(direct_cmos)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = []
    for index in range(TARGET_COUNT):
        row_outputs = {grid: {key: value[index:index + 1] for key, value in outputs[grid].items()}
                       for grid in grids}
        row_direct = {grid: direct[grid][index:index + 1] for grid in grids}
        row = projected._consume_production_row(
            population, index, population["coefficients"][index], amplitudes[index:index + 1].cpu(),
            row_outputs, row_direct, inputs["response"], "continuous_amplitude_float32",
        )
        row["source_validation_index"] = int(population["source_validation_indices"][index])
        row["row_passed"] = all(float(row[key]) <= GATE_LIMIT for key in METRIC_KEYS)
        rows.append(row)
    return rows, {"amplitudes": amplitudes.detach().cpu().clone(),
                  "outputs": outputs, "direct_raw_cmos": direct}


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["population_index", "source_validation_index", "label", "sample_index_within_class",
               "proposal_id", "response_matrix_relative_error_2048", *METRIC_KEYS, "row_passed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def _qualification_replay(inputs: Mapping[str, Any], population: Mapping[str, Any],
                          directory: Path, device: torch.device) -> dict[str, Any]:
    commands = _witness_commands(inputs, population)
    rows, measured = _measure_commands(population, commands, inputs, device)
    summary = projected._summarize_rows(rows)
    all_passed = bool(summary["all_rows_passed"])
    _atomic_torch(directory / "witness_qualification_measurements.pt", {
        "schema": "lightgen_g1_fit20_witness_qualification.v1", "task_id": TASK_ID,
        "population_role": "known_reachable_fit20", "source_validation_indices":
            population["source_validation_indices"], "target_coefficients": population["coefficients"],
        "labels": population["labels"], "proposal_ids": population["proposal_ids"],
        "amplitudes": measured["amplitudes"], "outputs": measured["outputs"],
        "direct_raw_cmos": measured["direct_raw_cmos"], "rows": rows,
    })
    _write_rows_csv(directory / "witness_qualification_rows.csv", rows)
    result = {"status": "ALL_20_SIX_METRIC_WITNESSES_QUALIFIED" if all_passed else
              "PREFLIGHT_BLOCKED_REFERENCE_QUALIFICATION_INSUFFICIENT",
              "count": TARGET_COUNT, "passed": sum(bool(row["row_passed"]) for row in rows),
              "all_six_metrics_required": list(METRIC_KEYS), "threshold": GATE_LIMIT,
              "metrics": summary, "training_labels": False, "inference_fallback": False}
    _atomic_json(directory / "witness_qualification_summary.json", result)
    return result


def _production_negative_controls(inputs: Mapping[str, Any], population: Mapping[str, Any],
                                  matrix: torch.Tensor, inverse: torch.Tensor,
                                  scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    mapper = _make_mapper(inputs, device)
    target = population["coefficients"][:1].to(device=device, dtype=torch.complex128)
    feature = projected._normalized_features(population["coefficients"][:1], scale, device)
    with torch.no_grad():
        amplitude, _ = projected._map_amplitude(mapper, feature, target, matrix, inverse, scale)
    output_by_grid: dict[int, dict[str, torch.Tensor]] = {}
    direct_by_grid: dict[int, torch.Tensor] = {}
    wrong_output: dict[int, dict[str, torch.Tensor]] = {}
    for grid in GRIDS:
        model = pilot._new_optical_model(inputs["checkpoint"], grid, device)
        with torch.no_grad():
            actual = projected._forward_amplitude(model, amplitude)
            reference = projected._forward_target(model, target)
            legacy = model.runtime.forward_from_image(amplitude.double())
        output_by_grid[grid] = {key: value.detach().cpu() for key, value in actual.items()}
        direct_by_grid[grid] = reference["raw_cmos"].detach().cpu()
        wrong_output[grid] = {"coefficients": legacy.z_ols.coefficients.detach().cpu(),
                              "raw_cmos": legacy.raw_camera.tensor.detach().cpu()}
        del model
    valid = projected._consume_production_row(
        population, 0, target.detach().cpu(), amplitude.detach().cpu(), output_by_grid,
        direct_by_grid, inputs["response"], "continuous_amplitude_float32")
    wrong_target = False
    wrong_encoding = False
    wrong_intensity_path = False
    try:
        projected._consume_production_row(population, 0, population["coefficients"][1],
                                          amplitude.detach().cpu(), output_by_grid, direct_by_grid,
                                          inputs["response"], "continuous_amplitude_float32")
    except ValueError:
        wrong_target = True
    try:
        projected._consume_production_row(population, 0, target.detach().cpu(), amplitude.detach().cpu(),
                                          output_by_grid, direct_by_grid, inputs["response"],
                                          "intensity_float32")
    except ValueError:
        wrong_encoding = True
    try:
        projected._consume_production_row(population, 0, target.detach().cpu(), amplitude.detach().cpu(),
                                          wrong_output, direct_by_grid, inputs["response"],
                                          "continuous_amplitude_float32")
    except ValueError:
        wrong_intensity_path = True
    _require(wrong_target and wrong_encoding and wrong_intensity_path,
             "production negative controls failed to reject target or amplitude/intensity mismatch")
    return {"valid_production_row_recomputed": True, "wrong_target_rejected": wrong_target,
            "wrong_command_encoding_rejected": wrong_encoding,
            "amplitude_routed_as_intensity_rejected": wrong_intensity_path,
            "valid_row": valid}


def _fit20_objective_gradient_check(inputs: Mapping[str, Any], population: Mapping[str, Any],
                                    matrix: torch.Tensor, inverse: torch.Tensor,
                                    scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    mapper = _make_mapper(inputs, device)
    target = population["coefficients"][:1].to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(population["coefficients"][:1], scale, device)
    raw = mapper(features).detach().requires_grad_(True)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    amplitude, projection = projected._project_from_raw(
        raw, target, matrix, inverse, scale, include_preclip=True,
    )
    actual = projected._forward_amplitude(model, amplitude)
    lc = pilot._phase_invariant_loss(actual["coefficients"], target).mean()
    box = _box_loss(projection["preclip_amplitude"].reshape(1, PIXEL_COUNT))
    total = lc + 5.0 * box
    gradient, = torch.autograd.grad(total, raw)
    direction = -gradient.detach() / torch.linalg.vector_norm(gradient.detach()).clamp_min(1.0e-30)
    _require(torch.isfinite(gradient).all().item() and float(torch.linalg.vector_norm(gradient)) > 0.0,
             "full production B objective has no finite raw-mapper gradient")
    # The central difference includes the float32 command cast at the SLM boundary.
    epsilon = 2.0e-2
    values: dict[str, float] = {}
    with torch.no_grad():
        for name, candidate in (("center", raw.detach()), ("descent", raw.detach() + epsilon * direction)):
            command, details = projected._project_from_raw(
                candidate, target, matrix, inverse, scale, include_preclip=True,
            )
            optical = projected._forward_amplitude(model, command)
            coefficient = pilot._phase_invariant_loss(optical["coefficients"], target).mean()
            auxiliary = _box_loss(details["preclip_amplitude"].reshape(1, PIXEL_COUNT))
            values[f"{name}_coefficient_loss"] = float(coefficient.item())
            values[f"{name}_box_loss"] = float(auxiliary.item())
            values[f"{name}_total_loss"] = float((coefficient + 5.0 * auxiliary).item())
    _require(values["descent_total_loss"] < values["center_total_loss"],
             "actual float32 SLM command perturbation does not lower B total objective")
    return {"full_production_path": True, "direction": "negative_gradient_of_Lc_plus_5Lbox",
            "epsilon_in_raw_float32": epsilon, **values,
            "coefficient_loss_delta": values["descent_coefficient_loss"] - values["center_coefficient_loss"],
            "total_loss_delta": values["descent_total_loss"] - values["center_total_loss"]}


def _checkpoint3_path(arm_dir: Path) -> Path:
    return arm_dir / "engineering_recovery" / "checkpoint_step_000003.pt"


def _recovery_worker(config_path: Path, arm: str, checkpoint_path: Path, output_path: Path) -> None:
    config = _load_config(config_path)
    _set_numerics()
    _require(torch.cuda.is_available(), "isolated recovery worker requires CUDA")
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    population, schedule = _load_saved_population(PREFLIGHT_DIR / "fixed_population_and_schedule.pt")
    response, matrix, inverse, scale = _load_matrix(PREFLIGHT_DIR, inputs, device)
    run_id = _json(PREFLIGHT_DIR / "preflight_identity.json")["run_id"]
    mapper = _make_mapper(inputs, device)
    optimizer = _optimizer(mapper)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    completed, _, _ = _restore_checkpoint(checkpoint_path, arm, mapper, optimizer, run_id, schedule)
    _require(completed == 3, "recovery worker requires the explicit step-3 checkpoint")
    item = _training_step(arm, mapper, optimizer, model, population, schedule,
                          matrix, inverse, scale, device, 3, audit=False)
    state = {"mapper": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
             "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": _snapshot_rng(),
             "loss": item["total_loss"], "coefficient_loss": item["coefficient_loss"],
             "sample_indices": schedule.reshape(500, CLASS_COUNT)[3].clone(),
             "engineering_adam_calls": 1}
    _atomic_torch(output_path, state)


def _engineering_recovery_arm(config_path: Path, config: Mapping[str, Any], inputs: Mapping[str, Any],
                              population: Mapping[str, Any], schedule: torch.Tensor,
                              matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
                              arm: str, device: torch.device, run_id: str) -> dict[str, Any]:
    arm_dir = PREFLIGHT_DIR / "engineering_recovery" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    mapper = _make_mapper(inputs, device)
    optimizer = _optimizer(mapper)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    frozen_before = projected._clone_state(model)
    torch.manual_seed(int(config["population"]["rng_seed"]))
    torch.cuda.manual_seed_all(int(config["population"]["rng_seed"]))
    history = []
    mapper_start = {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()}
    for index in range(3):
        history.append(_training_step(arm, mapper, optimizer, model, population, schedule,
                                      matrix, inverse, scale, device, index, audit=True))
    _require(all(not torch.equal(mapper_start[key], value.detach().cpu())
                 for key, value in mapper.state_dict().items()),
             f"arm {arm} did not update each mapper parameter in three real updates")
    cp3 = _checkpoint3_path(arm_dir)
    saved3 = _save_checkpoint(cp3, arm, mapper, optimizer, run_id, 3, history, schedule)
    _require(saved3["history_count"] == 3
             and torch.equal(saved3["next_batch_indices"], schedule.reshape(500, CLASS_COUNT)[3]),
             f"arm {arm} third-step checkpoint lost schedule/history")
    item = _training_step(arm, mapper, optimizer, model, population, schedule,
                          matrix, inverse, scale, device, 3)
    original = {"mapper": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
                "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": _snapshot_rng(),
                "loss": item["total_loss"], "coefficient_loss": item["coefficient_loss"],
                "sample_indices": schedule.reshape(500, CLASS_COUNT)[3].clone(),
                "engineering_adam_calls": 1}
    original_path = arm_dir / "original_process_step4.pt"
    _atomic_torch(original_path, original)
    worker_path = arm_dir / "independent_process_step4.pt"
    command = [sys.executable, "-m", MODULE_PATH, "--mode", "recovery-worker",
               "--config", str(config_path.resolve()), "--arm", arm,
               "--checkpoint", str(cp3.resolve()), "--worker-output", str(worker_path.resolve())]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    _require(completed.returncode == 0 and worker_path.is_file(),
             f"arm {arm} independent recovery process failed: {completed.stderr[-2000:]}")
    restored = torch.load(worker_path, map_location="cpu", weights_only=False)
    mapper_equal, mapper_delta = _tree_max_delta(original["mapper"], restored["mapper"])
    optimizer_equal, optimizer_delta = _tree_max_delta(original["optimizer"], restored["optimizer"])
    loss_delta = abs(float(original["loss"]) - float(restored["loss"]))
    coeff_delta = abs(float(original["coefficient_loss"]) - float(restored["coefficient_loss"]))
    rng_equal = torch.equal(original["rng"]["cpu"], restored["rng"]["cpu"])
    rng_equal = rng_equal and len(original["rng"]["cuda"]) == len(restored["rng"]["cuda"])
    rng_equal = rng_equal and all(torch.equal(left, right) for left, right in
                                  zip(original["rng"]["cuda"], restored["rng"]["cuda"], strict=True))
    schedule_equal = torch.equal(original["sample_indices"], restored["sample_indices"])
    _require(mapper_equal and optimizer_equal and loss_delta <= RECOVERY_TOLERANCE
             and coeff_delta <= RECOVERY_TOLERANCE and rng_equal and schedule_equal,
             f"arm {arm} live-vs-restored fourth update disagrees beyond the recovery contract")
    projected._assert_frozen(model, frozen_before, f"arm-{arm} optical model")
    return {"arm": arm, "pre_updates": 3, "live_fourth_update": True,
            "independent_process_restored_fourth_update": True,
            "engineering_adam_calls": 5, "mapper_max_abs_delta": mapper_delta,
            "adam_max_abs_delta": optimizer_delta, "total_loss_abs_delta": loss_delta,
            "coefficient_loss_abs_delta": coeff_delta, "rng_exact": rng_equal,
            "schedule_exact": schedule_equal, "frozen_optics_exact": True,
            "checkpoint_optimizer_deepcopied": saved3["optimizer_deepcopied"] is True}


def _write_preflight_failure(error: Exception, status: str = "PREFLIGHT_ENGINEERING_OR_INPUT_ERROR") -> None:
    existing = sorted(PREFLIGHT_DIR.glob("preflight_failure_*.json"))
    _atomic_json(PREFLIGHT_DIR / f"preflight_failure_{len(existing) + 1:02d}.json", {
        "task_id": TASK_ID, "status": status, "error_type": type(error).__name__,
        "error": str(error), "scientific_updates": 0, "recorded_at_unix": time.time(),
    })
    _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", {
        "task_id": TASK_ID, "status": status, "error_type": type(error).__name__,
        "error": str(error), "scientific_updates": 0, "recorded_at_unix": time.time(),
    })


def _preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    previous_identity: dict[str, Any] | None = None
    if PREFLIGHT_DIR.exists():
        summary_path = PREFLIGHT_DIR / "preflight_summary.json"
        if summary_path.is_file():
            prior_summary = _json(summary_path)
            if prior_summary.get("status") in {
                "READY", "PREFLIGHT_BLOCKED_REFERENCE_QUALIFICATION_INSUFFICIENT",
            }:
                raise FileExistsError(f"fit20 preflight reached a terminal branch: {PREFLIGHT_DIR}")
        receipt_path = PREFLIGHT_DIR / "preflight_receipt.json"
        if receipt_path.is_file() and _json(receipt_path).get("status") == "READY":
            raise FileExistsError(f"fit20 already has a consumable READY receipt: {PREFLIGHT_DIR}")
        identity_path = PREFLIGHT_DIR / "preflight_identity.json"
        if identity_path.is_file():
            previous_identity = _json(identity_path)
            _require(previous_identity.get("task_id") == TASK_ID,
                     "existing preflight directory belongs to another task")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if previous_identity is None:
        run_id = f"fit20_{uuid.uuid4().hex}"
        identity = _run_identity(config, run_id)
        _atomic_json(PREFLIGHT_DIR / "run_config_snapshot.json", config)
        _atomic_json(PREFLIGHT_DIR / "preflight_identity.json",
                     {"task_id": TASK_ID, "run_id": run_id, "run_identity": identity.to_dict()})
    else:
        run_id = str(previous_identity["run_id"])
        identity = ResearchRunIdentity.from_mapping(previous_identity["run_identity"])
        _require(identity.execution_semantics() == _run_identity(config, run_id).execution_semantics(),
                 "preflight retry changed the fixed task semantics")
    try:
        _set_numerics()
        _require(torch.cuda.is_available(), "fit20 preflight requires AutoDL CUDA")
        device = torch.device("cuda:0")
        inputs = _load_inputs(config)
        population = _fit_population(inputs)
        schedule = _build_schedule(population, 2026102781)
        _save_population(PREFLIGHT_DIR / "fixed_population_and_schedule.pt", population, schedule)
        _save_fixed_matrix(PREFLIGHT_DIR, inputs)
        response, matrix, inverse, scale = _load_matrix(PREFLIGHT_DIR, inputs, device)
        qualification = _qualification_replay(inputs, population, PREFLIGHT_DIR, device)
        if not qualification["status"] == "ALL_20_SIX_METRIC_WITNESSES_QUALIFIED":
            _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", {
                "task_id": TASK_ID, "run_id": run_id,
                "status": "PREFLIGHT_BLOCKED_REFERENCE_QUALIFICATION_INSUFFICIENT",
                "scientific_updates": 0, "witness_qualification": qualification,
                "training_started": False,
            })
            return {"status": "PREFLIGHT_BLOCKED_REFERENCE_QUALIFICATION_INSUFFICIENT",
                    "passed": qualification["passed"], "count": TARGET_COUNT}
        box_contract = _box_contract_check(device)
        matrix_checks = projected._matrix_checks(response, matrix, inverse, scale, device)
        camera_model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
        camera_checks = projected._matrix_and_camera_checks(inputs["frozen_inputs"], response,
                                                              camera_model, device)
        projection_check = projected._projection_gradient_checks(
            matrix, inverse, scale, population["coefficients"][:1].to(device), device)
        optical_check = projected._optical_gradient_checks(
            camera_model, population["coefficients"][:1].to(device), device)
        objective_check = _fit20_objective_gradient_check(
            inputs, population, matrix, inverse, scale, device)
        negatives = _production_negative_controls(inputs, population, matrix, inverse, scale, device)
        _atomic_json(PREFLIGHT_DIR / "differential_and_negative_controls.json", {
            "box_loss_numpy_and_gradient": box_contract,
            "fixed_projection_value_derivative_direction": projection_check,
            "full_optics_value_derivative_direction": optical_check,
            "B_objective_float32_command_direction": objective_check,
            "production_consumer_negative_controls": negatives,
        })
        recoveries = {}
        for arm in ARM_NAMES:
            recoveries[arm] = _engineering_recovery_arm(
                config_path, config, inputs, population, schedule, matrix, inverse,
                scale, arm, device, run_id,
            )
        recovery = {"arms": recoveries, "engineering_adam_calls_total": 10,
                    "scientific_updates": 0, "all_recovery_checks_passed": True}
        _atomic_json(PREFLIGHT_DIR / "engineering_recovery.json", recovery)
        memory = torch.cuda.mem_get_info(device)
        resource = {"passed": True, "status": "PASS",
                    "cuda_device": torch.cuda.get_device_name(device),
                    "cuda_memory_free_bytes_observation": int(memory[0]),
                    "cuda_memory_total_bytes_observation": int(memory[1]),
                    "preflight_gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "free_memory_is_gate": False}
        correctness = {"passed": True, "status": "PASS",
                       "predecessor_identity_history_schedule_and_adam_verified": True,
                       "known_reachable_fit20_all_six_metrics_qualified": True,
                       "box_loss_independent_numpy_and_gradient_verified": True,
                       "frozen_response_and_camera_units_verified": True,
                       "production_negative_controls_rejected": True,
                       "optics_and_evaluators_frozen": True}
        implementation = {"passed": True, "status": "PASS",
                          "selected_engine": "CUDA_full_C_encoder_OLS_decoder_production_path",
                          "engine": "CUDA_full_C_encoder_OLS_decoder_production_path",
                          "both_arms_real_three_step_and_independent_recovery_verified": True,
                          "recovery_tolerance": RECOVERY_TOLERANCE}
        control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
        receipt = control.make_ready_receipt(
            correctness=correctness, resource=resource, implementation=implementation,
            selected_engine="CUDA_full_C_encoder_OLS_decoder_production_path",
            resource_envelope=resource,
        )
        control.mark_preflight_ready(receipt)
        _atomic_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
        summary = {"task_id": TASK_ID, "run_id": run_id, "status": "READY",
                   "population_role": "known_reachable_fit20", "branch": "PAIRED_500_UPDATE_A_B",
                   "witness_qualification": qualification, "box_loss_contract": box_contract,
                   "matrix_checks": matrix_checks, "camera_checks": camera_checks,
                   "projection_gradient_check": projection_check,
                   "full_optics_gradient_check": optical_check,
                   "B_objective_direction_check": objective_check,
                   "production_negative_controls": negatives,
                   "engineering_recovery": recovery, "scientific_updates": 0,
                   "resource_observations": resource, "ready_receipt_consumable": True}
        _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", summary)
        return {"status": "READY", "run_id": run_id, "branch": "PAIRED_500_UPDATE_A_B"}
    except Exception as error:
        _write_preflight_failure(error)
        raise


def _task_state(attempt_dir: Path, state: str, arm: str | None, completed: Mapping[str, int],
                note: str | None = None) -> None:
    payload: dict[str, Any] = {
        "schema": "lightgen_g1_projected_fit20_box_ab_state.v1", "task_id": TASK_ID,
        "run_id": _json(attempt_dir / "run_identity.json")["run_id"],
        "state": state, "current_arm": arm, "completed_updates_by_arm": dict(completed),
        "new_update_budget_per_arm": 500, "source_mapper_updates": 5000,
        "frozen_optical_updates": 0, "updated_at_unix": time.time(),
    }
    if note:
        payload["note"] = note
    _atomic_json(attempt_dir / "task_state.json", payload)


def _copy_preflight_evidence(attempt_dir: Path) -> None:
    destination = attempt_dir / "preflight_evidence"
    destination.mkdir(exist_ok=False)
    names = ("preflight_identity.json", "preflight_receipt.json", "preflight_summary.json",
             "run_config_snapshot.json", "fixed_population_and_schedule.pt",
             "fixed_response_and_projector.pt", "witness_qualification_measurements.pt",
             "witness_qualification_rows.csv", "witness_qualification_summary.json",
             "differential_and_negative_controls.json", "engineering_recovery.json")
    for name in names:
        source = PREFLIGHT_DIR / name
        if source.is_file():
            import shutil
            shutil.copy2(source, destination / name)


def _read_history(path: Path, arm: str, run_id: str, schedule: torch.Tensor
                  ) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = _json(path)
    updates = payload.get("updates")
    _require(payload.get("task_id") == TASK_ID and payload.get("run_id") == run_id
             and payload.get("arm") == arm and isinstance(updates, list)
             and payload.get("completed_new_updates") == len(updates)
             and payload.get("state") in {"ACTIVE", "COMPLETE"}
             and len(updates) <= 500,
             f"arm {arm} saved history identity or length is inconsistent")
    _require(all(item.get("completed_update") == index + 1
                 and item.get("sample_indices") == schedule.reshape(500, CLASS_COUNT)[index].tolist()
                 and item.get("arm") == arm for index, item in enumerate(updates)),
             f"arm {arm} training history does not match the fixed schedule")
    for item in updates:
        lc, box, total = (float(item[key]) for key in ("coefficient_loss", "box_loss", "total_loss"))
        expected_total = lc if arm == "A" else lc + 5.0 * box
        gradients = item.get("gradient_norms", {})
        _require(all(math.isfinite(value) for value in
                     (lc, box, total, float(item["clipped_fraction_mean"])))
                 and abs(total - expected_total) <= 1.0e-14
                 and set(gradients) == {"network.0.weight", "network.0.bias",
                                        "network.2.weight", "network.2.bias"}
                 and all(math.isfinite(float(value)) for value in gradients.values()),
                 f"arm {arm} history has a nonfinite or inconsistent objective/gradient record")
    return [dict(item) for item in updates]


def _observe_arm_step(arm: str, step: int, mapper: torch.nn.Module,
                      population: Mapping[str, Any], inputs: Mapping[str, Any],
                      matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
                      device: torch.device, arm_dir: Path) -> dict[str, Any]:
    coefficients = population["coefficients"].to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(population["coefficients"], scale, device)
    with torch.no_grad():
        amplitude, detail = projected._map_amplitude(mapper, features, coefficients,
                                                       matrix, inverse, scale,
                                                       include_preclip=True)
    rows, measurement = _measure_commands(population, amplitude.cpu(), inputs, device)
    summary = projected._summarize_rows(rows)
    outdir = arm_dir / "measurements"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "lightgen_g1_fit20_observation.v1", "task_id": TASK_ID,
               "arm": arm, "completed_updates": step,
               "population_role": "known_reachable_fit20",
               "command_encoding": "continuous_amplitude_float32",
               "mapper_state": {key: value.detach().cpu().clone()
                                for key, value in mapper.state_dict().items()},
               "objective": "Lc" if arm == "A" else "Lc+5*Lbox",
               "target_coefficients": population["coefficients"].clone(),
               "labels": population["labels"].clone(),
               "proposal_ids": population["proposal_ids"].clone(),
               "source_validation_indices": population["source_validation_indices"].clone(),
               "amplitudes": measurement["amplitudes"],
               "preclip_min": detail["preclip_min"].detach().cpu(),
               "preclip_max": detail["preclip_max"].detach().cpu(),
               "clipped_fraction": detail["clipped_fraction"].detach().cpu(),
               "outputs": measurement["outputs"], "direct_raw_cmos": measurement["direct_raw_cmos"],
               "rows": rows, "summary": summary}
    path = outdir / f"step_{step:03d}.pt"
    _atomic_torch(path, payload)
    _write_rows_csv(outdir / f"step_{step:03d}_rows.csv", rows)
    _atomic_json(outdir / f"step_{step:03d}_summary.json", summary)
    observation = {"checkpoint_step": step, "all_20_six_metric_rows_pass": summary["all_rows_passed"],
                   "passed_rows": summary["passed_rows"], "maxima": summary["maxima"],
                   "observation_file": path.name}
    observation_path = arm_dir / "observations.json"
    existing = _json(observation_path).get("observations", []) if observation_path.is_file() else []
    by_step = {int(item["checkpoint_step"]): item for item in [*existing, observation]}
    _atomic_json(observation_path, {"task_id": TASK_ID,
                                    "run_id": _json(arm_dir.parent.parent / "run_identity.json")["run_id"],
                                    "arm": arm,
                                    "observations": [by_step[key] for key in sorted(by_step)]})
    return observation


def _train_arm(config: Mapping[str, Any], inputs: Mapping[str, Any], attempt_dir: Path,
               population: Mapping[str, Any], schedule: torch.Tensor,
               matrix: torch.Tensor, inverse: torch.Tensor, scale: torch.Tensor,
               device: torch.device, arm: str, *, resume_checkpoint: Path | None = None,
               config_path: Path) -> dict[str, Any]:
    arm_dir = attempt_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    run_id = _json(attempt_dir / "run_identity.json")["run_id"]
    mapper = _make_mapper(inputs, device)
    optimizer = _optimizer(mapper)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    frozen_before = projected._clone_state(model)
    history_path = arm_dir / "training_history.json"
    history = _read_history(history_path, arm, run_id, schedule)
    if resume_checkpoint is None:
        _require(not history and not (arm_dir / "checkpoints" / "checkpoint_step_000500.pt").exists(),
                 f"arm {arm} already has training state; use explicit resume")
        torch.manual_seed(int(config["population"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["population"]["rng_seed"]))
        completed = 0
        _write_history(history_path, arm, run_id, history, "ACTIVE")
        checkpoint_dir = arm_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(checkpoint_dir / "checkpoint_step_000000.pt", arm, mapper,
                         optimizer, run_id, 0, history, schedule)
        _save_checkpoint(checkpoint_dir / "latest_checkpoint.pt", arm, mapper,
                         optimizer, run_id, 0, history, schedule)
    else:
        _require(resume_checkpoint.is_file(), f"explicit resume checkpoint is missing: {resume_checkpoint}",
                 FileNotFoundError)
        completed, _, saved = _restore_checkpoint(resume_checkpoint, arm, mapper,
                                                  optimizer, run_id, schedule)
        _require(len(history) == completed and all(
            item.get("completed_update") == index + 1
            and item.get("sample_indices") == schedule.reshape(500, CLASS_COUNT)[index].tolist()
            for index, item in enumerate(history[:completed])),
            f"arm {arm} checkpoint does not match the latest committed training history")
        _require(saved.get("history_count") == completed,
                 f"arm {arm} checkpoint and training history count disagree")
        _write_history(history_path, arm, run_id, history, "ACTIVE")
    observations: list[dict[str, Any]] = []
    if completed in (0, 250, 500) and not (arm_dir / "measurements" / f"step_{completed:03d}.pt").is_file():
        observations.append(_observe_arm_step(arm, completed, mapper, population, inputs,
                                              matrix, inverse, scale, device, arm_dir))
    checkpoint_dir = arm_dir / "checkpoints"
    for update_index in range(completed, 500):
        item = _training_step(arm, mapper, optimizer, model, population, schedule,
                              matrix, inverse, scale, device, update_index,
                              audit=(update_index == 0 and completed == 0))
        history.append(item)
        step = update_index + 1
        if step % 25 == 0 or step in (250, 500):
            _write_history(history_path, arm, run_id, history, "ACTIVE")
            _atomic_json(attempt_dir / "progress.json", {
                "task_id": TASK_ID, "run_id": run_id, "phase": "training",
                "arm": arm, "completed_updates": step, "updates_per_arm": 500,
                "completed_updates_by_arm": {name: len(_read_history(
                    attempt_dir / "arms" / name / "training_history.json", name, run_id, schedule))
                    if (attempt_dir / "arms" / name / "training_history.json").is_file() else 0
                    for name in ARM_NAMES}, "last_update": item,
                "updated_at_unix": time.time(),
            })
        if step % 25 == 0 or step in (250, 500):
            _save_checkpoint(checkpoint_dir / "latest_checkpoint.pt", arm, mapper,
                             optimizer, run_id, step, history, schedule)
        if step in (250, 500):
            _save_checkpoint(checkpoint_dir / f"checkpoint_step_{step:06d}.pt", arm,
                             mapper, optimizer, run_id, step, history, schedule)
            observations.append(_observe_arm_step(arm, step, mapper, population, inputs,
                                                  matrix, inverse, scale, device, arm_dir))
        if step % 25 == 0:
            print(json.dumps({"task_id": TASK_ID, "arm": arm, "completed_updates": step,
                              "coefficient_loss": item["coefficient_loss"],
                              "box_loss": item["box_loss"], "total_loss": item["total_loss"]}),
                  flush=True)
    _write_history(history_path, arm, run_id, history, "COMPLETE")
    terminal = torch.load(checkpoint_dir / "checkpoint_step_000500.pt",
                          map_location="cpu", weights_only=False)
    adam_steps = [int(torch.as_tensor(state["step"]).item())
                  for state in terminal["optimizer_state"]["state"].values() if "step" in state]
    _require(len(history) == 500 and terminal.get("history_count") == 500
             and len(adam_steps) == 4 and all(value == 500 for value in adam_steps),
             f"arm {arm} did not finish exactly 500 updates with four Adam step-500 states")
    projected._assert_frozen(model, frozen_before, f"arm-{arm} optical model")
    observations_path = arm_dir / "observations.json"
    existing = _json(observations_path).get("observations", []) if observations_path.is_file() else []
    by_step = {int(row["checkpoint_step"]): row for row in [*existing, *observations]}
    _atomic_json(observations_path, {"task_id": TASK_ID, "run_id": run_id, "arm": arm,
                                     "observations": [by_step[key] for key in sorted(by_step)]})
    return {"arm": arm, "completed_updates": len(history), "terminal_checkpoint":
            str(checkpoint_dir / "checkpoint_step_000500.pt"),
            "adam_steps": adam_steps, "observations": [by_step[key] for key in sorted(by_step)]}


def _open_attempt(config: Mapping[str, Any], resume: bool) -> tuple[ResearchRunControl,
                                                                    PreflightReceipt]:
    control, receipt = _load_control(PREFLIGHT_DIR)
    if not resume:
        control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01", receipt)
        _atomic_json(ATTEMPT_DIR / "run_identity.json", control.identity.to_dict())
        _atomic_json(ATTEMPT_DIR / "run_config_snapshot.json", config)
        _copy_preflight_evidence(ATTEMPT_DIR)
        _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, None, {"A": 0, "B": 0})
    else:
        _require(ATTEMPT_DIR.is_dir(), "explicit resume requires the existing fit20 attempt")
        _assert_attempt(ATTEMPT_DIR)
        state = _json(ATTEMPT_DIR / "task_state.json")
        _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "INTERRUPTED"},
                 "closed fit20 task cannot be reopened or resumed")
        control.open_scientific_attempt(ATTEMPT_DIR,
                                        f"{control.identity.run_id}:attempt_01", receipt, resume=True)
    return control, receipt


def _execute(config_path: Path, config: Mapping[str, Any], *,
             resume_checkpoint: Path | None = None, resume_arm: str | None = None) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "fit20 training requires AutoDL CUDA")
    _require((resume_checkpoint is None) == (resume_arm is None),
             "resume requires both an explicit --checkpoint and --arm", ValueError)
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    training = _json(PREFLIGHT_DIR / "preflight_summary.json")
    _require(training.get("task_id") == TASK_ID and training.get("status") == "READY"
             and _json(PREFLIGHT_DIR / "preflight_receipt.json").get("status") == "READY",
             "fit20 execute requires its complete READY receipt")
    if resume_arm is not None:
        _require(resume_arm in ARM_NAMES, "resume arm must be A or B", ValueError)
    control, _ = _open_attempt(config, resume_checkpoint is not None)
    population, schedule = _load_saved_population(
        ATTEMPT_DIR / "fixed_population_and_schedule.pt" if resume_checkpoint is not None else
        PREFLIGHT_DIR / "fixed_population_and_schedule.pt")
    if resume_checkpoint is None:
        _save_population(ATTEMPT_DIR / "fixed_population_and_schedule.pt", population, schedule)
    response, matrix, inverse, scale = _load_matrix(
        ATTEMPT_DIR if resume_checkpoint is not None else PREFLIGHT_DIR, inputs, device)
    # Recompute completion from each serialized history rather than trusting state summaries.
    histories = {arm: _read_history(ATTEMPT_DIR / "arms" / arm / "training_history.json",
                                    arm, control.identity.run_id, schedule)
                 for arm in ARM_NAMES} if resume_checkpoint is not None else {"A": [], "B": []}
    _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, resume_arm,
                {arm: len(histories[arm]) for arm in ARM_NAMES})
    run_from = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == control.identity.run_id
                 and checkpoint.get("arm") == resume_arm,
                 "explicit resume checkpoint identity differs from selected arm")
        completed = int(checkpoint["completed_updates"])
        if completed < 500:
            run_from = ARM_NAMES.index(resume_arm)
        else:
            run_from = ARM_NAMES.index(resume_arm) + 1
        if resume_arm == "B":
            prior = _read_history(ATTEMPT_DIR / "arms/A/training_history.json", "A",
                                  control.identity.run_id, schedule)
            _require(len(prior) == 500, "arm B recovery requires completed arm A history")
    try:
        results: dict[str, Any] = {}
        for arm_index in range(run_from, len(ARM_NAMES)):
            arm = ARM_NAMES[arm_index]
            cp = resume_checkpoint if resume_checkpoint is not None and arm == resume_arm and completed < 500 else None
            result = _train_arm(config, inputs, ATTEMPT_DIR, population, schedule,
                                matrix, inverse, scale, device, arm,
                                resume_checkpoint=cp, config_path=config_path)
            results[arm] = result
            completed_map = {name: len(_read_history(ATTEMPT_DIR / "arms" / name / "training_history.json",
                                                       name, control.identity.run_id, schedule))
                             if (ATTEMPT_DIR / "arms" / name / "training_history.json").is_file() else 0
                             for name in ARM_NAMES}
            _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE,
                        ARM_NAMES[arm_index + 1] if arm_index + 1 < len(ARM_NAMES) else None,
                        completed_map)
            resume_checkpoint = None
        completed_map = {arm: len(_read_history(ATTEMPT_DIR / "arms" / arm / "training_history.json",
                                                arm, control.identity.run_id, schedule))
                         for arm in ARM_NAMES}
        _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, None, completed_map,
                    note="both 500-step arms complete; terminal measurements saved at step 500")
        return {"status": "BOTH_ARMS_READY_FOR_TERMINAL_MEASUREMENT", "run_id": control.identity.run_id,
                "completed_updates_by_arm": completed_map, "arms": results}
    except Exception as error:
        completed_map = {arm: len(_read_history(ATTEMPT_DIR / "arms" / arm / "training_history.json",
                                                arm, control.identity.run_id, schedule))
                         if (ATTEMPT_DIR / "arms" / arm / "training_history.json").is_file() else 0
                         for arm in ARM_NAMES}
        _task_state(ATTEMPT_DIR, "INTERRUPTED", resume_arm, completed_map,
                    note=f"{type(error).__name__}: {error}")
        raise


def _numeric_tree_matches(left: Any, right: Any, tolerance: float = 1.0e-14) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_numeric_tree_matches(left[key], right[key], tolerance)
                                               for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_numeric_tree_matches(a, b, tolerance)
                                               for a, b in zip(left, right, strict=True))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) \
            and abs(float(left) - float(right)) <= tolerance
    return left == right


def _consume_measurement_file(path: Path, arm: str, population: Mapping[str, Any],
                              inputs: Mapping[str, Any], expected_mapper: Mapping[str, torch.Tensor] | None = None,
                              expected_amplitudes: torch.Tensor | None = None,
                              command_check: str = "none",
                              payload_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = (torch.load(path, map_location="cpu", weights_only=False)
               if payload_override is None else dict(payload_override))
    _require(payload.get("schema") == "lightgen_g1_fit20_observation.v1"
             and payload.get("task_id") == TASK_ID and payload.get("arm") == arm
             and payload.get("completed_updates") == 500
             and payload.get("command_encoding") == "continuous_amplitude_float32"
             and payload.get("population_role", "known_reachable_fit20") == "known_reachable_fit20"
             and torch.equal(payload.get("target_coefficients"), population["coefficients"])
             and torch.equal(payload.get("labels"), population["labels"])
             and torch.equal(payload.get("proposal_ids"), population["proposal_ids"])
             and torch.equal(payload.get("source_validation_indices"),
                             population["source_validation_indices"]),
             f"arm {arm} terminal measurement has wrong identity, target, or role")
    if expected_mapper is not None:
        _require(set(payload.get("mapper_state", {})) == set(expected_mapper)
                 and all(torch.equal(payload["mapper_state"][key], expected_mapper[key].cpu())
                         for key in expected_mapper),
                 f"arm {arm} measured command was not produced by its terminal checkpoint mapper")
    amplitudes = payload.get("amplitudes")
    _require(torch.is_tensor(amplitudes) and tuple(amplitudes.shape) == (TARGET_COUNT, 1, 200, 200)
             and amplitudes.dtype == torch.float32 and torch.isfinite(amplitudes).all().item()
             and float(amplitudes.min()) >= 0.0 and float(amplitudes.max()) <= 1.0,
             f"arm {arm} saved SLM commands are malformed")
    command_delta: float | None = None
    command_response_error: float | None = None
    if expected_amplitudes is not None:
        _require(command_check in ("exact", "response"),
                 "terminal mapper command check must be exact or response-bounded")
        expected = expected_amplitudes.detach().cpu()
        _require(expected.dtype == torch.float32 and expected.shape == amplitudes.shape
                 and torch.isfinite(expected).all().item()
                 and float(expected.min()) >= 0.0 and float(expected.max()) <= 1.0,
                 f"arm {arm} replayed mapper command is malformed")
        command_delta = float((expected - amplitudes).abs().max().item())
        if command_check == "exact":
            _require(torch.equal(expected, amplitudes),
                     f"arm {arm} saved SLM command does not exactly replay from its terminal mapper")
        else:
            outputs_for_response = payload.get("outputs", {})
            _require(2048 in outputs_for_response,
                     f"arm {arm} command replay has no measured 2048-grid coefficients")
            response = inputs["response"].to(dtype=torch.complex128, device="cpu")
            predicted = expected.reshape(TARGET_COUNT, PIXEL_COUNT).to(torch.complex128) @ response.T
            actual = outputs_for_response[2048]["coefficients"].to(dtype=torch.complex128, device="cpu")
            losses = pilot._phase_invariant_loss(predicted, actual)
            command_response_error = float(torch.sqrt(losses).max().item())
            _require(math.isfinite(command_response_error)
                     and command_response_error <= projected.MATRIX_LIMIT,
                     f"arm {arm} terminal mapper replay response error {command_response_error:.3g} "
                     f"exceeds {projected.MATRIX_LIMIT:.3g}")
    else:
        _require(command_check == "none", "command_check requires replayed amplitudes")
    outputs = payload.get("outputs")
    direct = payload.get("direct_raw_cmos")
    _require(set(outputs) == set(GRIDS) and set(direct) == set(GRIDS),
             f"arm {arm} terminal measurement omits a grid")
    for grid in GRIDS:
        _require(tuple(outputs[grid]["coefficients"].shape) == (TARGET_COUNT, PORT_COUNT)
                 and outputs[grid]["coefficients"].dtype == torch.complex128
                 and tuple(outputs[grid]["raw_cmos"].shape) == (TARGET_COUNT, 1, 200, 200)
                 and outputs[grid]["raw_cmos"].dtype == torch.float64
                 and tuple(direct[grid].shape) == (TARGET_COUNT, 1, 200, 200)
                 and direct[grid].dtype == torch.float64,
                 f"arm {arm} grid-{grid} raw output shape or dtype changed")
    recomputed = []
    for index in range(TARGET_COUNT):
        row_outputs = {grid: {key: outputs[grid][key][index:index + 1]
                              for key in ("coefficients", "raw_cmos")} for grid in GRIDS}
        row_direct = {grid: direct[grid][index:index + 1] for grid in GRIDS}
        reported = {key: float(payload["rows"][index][key]) for key in METRIC_KEYS}
        row = projected._consume_production_row(
            population, index, payload["target_coefficients"][index], amplitudes[index:index + 1],
            row_outputs, row_direct, inputs["response"], "continuous_amplitude_float32",
            reported_metrics=reported,
        )
        row["source_validation_index"] = int(population["source_validation_indices"][index])
        row["row_passed"] = all(float(row[key]) <= GATE_LIMIT for key in METRIC_KEYS)
        recomputed.append(row)
    summary = projected._summarize_rows(recomputed)
    _require(_numeric_tree_matches(payload["summary"], summary)
             and len(payload["rows"]) == TARGET_COUNT
             and all(bool(payload["rows"][i]["row_passed"]) is bool(recomputed[i]["row_passed"])
                     for i in range(TARGET_COUNT)),
             f"arm {arm} terminal summary or per-row pass flags are stale")
    return {"rows": recomputed, "summary": summary, "payload": payload,
            "command_check": command_check, "command_max_abs_amplitude_delta": command_delta,
            "command_response_relative_error_max": command_response_error}


def _arm_terminal_checkpoint(attempt_dir: Path, arm: str,
                             expected_run_id: str | None = None) -> dict[str, Any]:
    path = attempt_dir / "arms" / arm / "checkpoints" / "checkpoint_step_000500.pt"
    _require(path.is_file(), f"arm {arm} terminal checkpoint is missing: {path}", FileNotFoundError)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    steps = [int(torch.as_tensor(item["step"]).item())
             for item in checkpoint["optimizer_state"]["state"].values() if "step" in item]
    _require(checkpoint.get("schema") == "lightgen_g1_projected_fit20_box_ab_checkpoint.v1"
             and checkpoint.get("task_id") == TASK_ID and checkpoint.get("arm") == arm
             and (expected_run_id is None or checkpoint.get("run_id") == expected_run_id)
             and checkpoint.get("completed_updates") == 500
             and checkpoint.get("history_count") == 500
             and checkpoint.get("optimizer_deepcopied") is True
             and len(steps) == 4 and all(step == 500 for step in steps),
             f"arm {arm} terminal checkpoint does not prove four Adam step-500 states")
    return checkpoint


def _classify(arm_a: Mapping[str, Any], arm_b: Mapping[str, Any]) -> tuple[str, str]:
    a_pass = bool(arm_a["summary"]["all_rows_passed"])
    b_pass = bool(arm_b["summary"]["all_rows_passed"])
    if b_pass and not a_pass:
        return "B_PASS_A_FAIL", "B passes while A fails; supports a separately designed full-population box-loss experiment."
    if a_pass and b_pass:
        return "BOTH_PASS", "Both objectives fit the fixed 20 known-reachable targets; this does not establish box-loss necessity."
    if a_pass and not b_pass:
        return "A_PASS_B_FAIL", "The coefficient-only objective fits these 20 targets; the registered box-loss setting is not supported."
    return "BOTH_FAIL", "Neither objective meets the threshold in 500 updates; this does not establish structural or G1 non-learnability."


def _write_final_report(attempt_dir: Path, decision: Mapping[str, Any]) -> None:
    branch = decision["branch"]
    narrative = decision["branch_interpretation"]
    b = decision["arms"]["B"]["summary"]
    a = decision["arms"]["A"]["summary"]
    report = f"""# G1 已知可达20目标：裁剪前约束损失小样本对照

## 结果与状态

{decision['lifecycle_state']}；分支为 `{branch}`。两臂均完成各500次科学更新，输入为固定 `known_reachable_fit20` 人口。{narrative}

Arm A 全部六项误差均满足阈值的样本数为 {a['passed_rows']}/20；Arm B 为 {b['passed_rows']}/20。唯一判据阈值为 `{GATE_LIMIT:.1e}`，只对这20条已知可达目标适用。

## 科学认识

本对照检验的是相同起点、同一固定20条目标和同一500步预算下，系数损失与固定系数5的裁剪前越界惩罚是否改变拟合结果。结果解释：{narrative} 这组已知可达目标不能代表独立泛化，也不能证明整个G1分布、硬件或人眼语义生成效果。

## 目标缺口

尚缺新目标人口上的连续制备、可读性与实物光路证据。该小样本结果不支持全人口推广，也不回答G4、量化训练或物理部署。

## 工程完成与验证

20条旧见证在启动训练前按2048/4096两网格完整重放，并重新计算系数、原生CMOS及跨网格六项误差；资格状态为 `{decision['witness_qualification']['status']}`。每臂500条历史记录逐项对应固定配对日程，四个Adam参数状态步数均为500。工程恢复对照共10次Adam调用，不计入科学更新；两臂分别验证了第三步检查点到第四步的原进程与独立进程恢复。冻结C光学参数更新数为0。Windows回传消费者从保存的目标、指令、系数与CMOS数组重算指标，结果记录于 `windows_reconsumption.json`。

## 决策影响与下一步

{narrative} 不追加训练、不扩网、不更换目标、不做A8/I8、G4、人工评分或实物实验。原始G1试训的 `VALID_FAILED` 与 `PROJECT_GEOMETRY_UNMEASURED` 保持不变。后续任何全人口实验须单独计划；本报告不授权启动。
"""
    (attempt_dir / "final_report.md").write_text(report, encoding="utf-8")


def _build_transfer_manifest(attempt_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "windows_reconsumption.json", "local_transfer_receipt.json"}
    files = []
    for path in sorted(attempt_dir.rglob("*")):
        if path.is_file() and path.name not in excluded and ".tmp." not in path.name:
            files.append({"path": path.relative_to(attempt_dir).as_posix(), "bytes": path.stat().st_size})
    payload = {"schema": "lightgen_g1_fit20_transfer_manifest.v1", "task_id": TASK_ID,
               "run_id": run_id, "files": files, "file_count": len(files),
               "bytes_total": sum(item["bytes"] for item in files),
               "windows_receipt_excluded": True}
    _atomic_json(attempt_dir / "transfer_manifest.json", payload)
    return payload


def _measure_terminal(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(ATTEMPT_DIR.is_dir(), "fit20 measurement requires the completed attempt")
    identity = _assert_attempt(ATTEMPT_DIR)
    state = _json(ATTEMPT_DIR / "task_state.json")
    _require(state.get("state") == RUN_SCIENTIFIC_ACTIVE,
             "fit20 measurement requires an active, not previously closed attempt")
    population, schedule = _load_saved_population(ATTEMPT_DIR / "fixed_population_and_schedule.pt")
    arm_results = {}
    for arm in ARM_NAMES:
        checkpoint = _arm_terminal_checkpoint(ATTEMPT_DIR, arm, identity.run_id)
        history = _read_history(ATTEMPT_DIR / "arms" / arm / "training_history.json",
                                arm, identity.run_id, schedule)
        _require(len(history) == 500, f"arm {arm} does not have exactly 500 recorded updates")
        observation = ATTEMPT_DIR / "arms" / arm / "measurements" / "step_500.pt"
        consumed = _consume_measurement_file(observation, arm, population, inputs,
                                             checkpoint["mapper_state"])
        arm_results[arm] = {"summary": consumed["summary"], "rows": consumed["rows"],
                            "checkpoint_step": 500, "history_count": len(history),
                            "adam_steps": [500, 500, 500, 500]}
    branch, interpretation = _classify(arm_results["A"], arm_results["B"])
    lifecycle = RUN_COMPLETED if arm_results["B"]["summary"]["all_rows_passed"] else RUN_VALID_FAILED
    control, receipt = _load_control(PREFLIGHT_DIR)
    control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01",
                                    receipt, resume=True)
    if lifecycle == RUN_COMPLETED:
        control.complete()
    else:
        control.valid_failed()
    decision = {"schema": "lightgen_g1_projected_fit20_box_ab_decision.v1", "task_id": TASK_ID,
                "run_id": identity.run_id, "lifecycle_state": lifecycle,
                "population_role": "known_reachable_fit20", "denominator": TARGET_COUNT,
                "threshold": GATE_LIMIT, "metric_names": list(METRIC_KEYS),
                "branch": branch, "branch_interpretation": interpretation,
                "witness_qualification": _json(ATTEMPT_DIR / "preflight_evidence" /
                                                "witness_qualification_summary.json"),
                "arms": arm_results, "source_mapper_updates": 5000,
                "new_scientific_updates_per_arm": 500, "frozen_optical_updates": 0,
                "engineering_recovery_adam_calls": 10,
                "independent_numpy_metric_recomputation": True,
                "native_cmos_normalization": "none"}
    _atomic_json(ATTEMPT_DIR / "final_decision.json", decision)
    _task_state(ATTEMPT_DIR, lifecycle, None, {"A": 500, "B": 500}, note=branch)
    _atomic_json(ATTEMPT_DIR / "research_lifecycle_state.json", {
        "schema": "lightgen_g1_fit20_lifecycle_state.v1", "task_id": TASK_ID,
        "run_id": identity.run_id, "state": control.state,
        "preflight_status": control.preflight_status, "receipt_consumed": control.receipt is receipt,
        "final_branch": branch, "scientific_updates_by_arm": {"A": 500, "B": 500},
    })
    _write_final_report(ATTEMPT_DIR, decision)
    _build_transfer_manifest(ATTEMPT_DIR, identity.run_id)
    return {"status": lifecycle, "branch": branch, "arms": arm_results,
            "transfer_manifest": str(ATTEMPT_DIR / "transfer_manifest.json")}


def _verify_transfer_manifest(evidence_dir: Path) -> dict[str, Any]:
    manifest = _json(evidence_dir / "transfer_manifest.json")
    _require(manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list),
             "fit20 transfer manifest identity is invalid")
    checked = []
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "fit20 transfer manifest contains an unsafe relative path")
        path = evidence_dir / relative
        _require(path.is_file() and path.stat().st_size == int(entry["bytes"]),
                 f"fit20 transferred file is missing or has wrong byte size: {relative}", ValueError)
        checked.append({"path": relative.as_posix(), "remote_bytes": int(entry["bytes"]),
                        "local_bytes": path.stat().st_size, "current_bytes": path.stat().st_size})
    _require(len(checked) == manifest.get("file_count")
             and sum(item["current_bytes"] for item in checked) == manifest.get("bytes_total"),
             "fit20 local file sizes disagree with the transfer manifest totals")
    return {"file_count": len(checked), "bytes_total": sum(item["current_bytes"] for item in checked),
            "all_current_file_sizes_match": True, "files": checked}


def _verify_saved(evidence_dir: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(evidence_dir.is_dir(), f"fit20 saved evidence is missing: {evidence_dir}", FileNotFoundError)
    size_report = _verify_transfer_manifest(evidence_dir)
    identity = _assert_attempt(evidence_dir)
    decision = _json(evidence_dir / "final_decision.json")
    state = _json(evidence_dir / "task_state.json")
    lifecycle = _json(evidence_dir / "research_lifecycle_state.json") if \
        (evidence_dir / "research_lifecycle_state.json").is_file() else None
    _require(decision.get("task_id") == TASK_ID and decision.get("run_id") == identity.run_id
             and state.get("task_id") == TASK_ID and state.get("run_id") == identity.run_id
             and decision.get("lifecycle_state") == state.get("state")
             and lifecycle is not None and lifecycle.get("task_id") == TASK_ID
             and lifecycle.get("run_id") == identity.run_id
             and lifecycle.get("receipt_consumed") is True
             and decision.get("denominator") == TARGET_COUNT
             and decision.get("threshold") == GATE_LIMIT,
             "fit20 decision, task state, or population denominator disagree")
    population, schedule = _load_saved_population(evidence_dir / "fixed_population_and_schedule.pt")
    recovery = _json(evidence_dir / "preflight_evidence" / "engineering_recovery.json")
    _require(recovery.get("all_recovery_checks_passed") is True
             and recovery.get("engineering_adam_calls_total") == 10
             and set(recovery.get("arms", {})) == set(ARM_NAMES),
             "fit20 real recovery evidence is incomplete")
    arm_results = {}
    for arm in ARM_NAMES:
        history = _read_history(evidence_dir / "arms" / arm / "training_history.json",
                                arm, identity.run_id, schedule)
        checkpoint = _arm_terminal_checkpoint(evidence_dir, arm, identity.run_id)
        _require(len(history) == 500 and checkpoint.get("history_count") == 500,
                 f"arm {arm} training history and terminal step do not match")
        observed = _consume_measurement_file(
            evidence_dir / "arms" / arm / "measurements" / "step_500.pt", arm,
            population, inputs, checkpoint["mapper_state"],
        )
        _require(_numeric_tree_matches(decision["arms"][arm]["summary"], observed["summary"])
                 and _numeric_tree_matches(decision["arms"][arm]["rows"], observed["rows"]),
                 f"arm {arm} final decision differs from independent raw-output reconsumption")
        for step in (0, 250, 500):
            checkpoint_path = evidence_dir / "arms" / arm / "checkpoints" / f"checkpoint_step_{step:06d}.pt"
            observation_path = evidence_dir / "arms" / arm / "measurements" / f"step_{step:03d}.pt"
            _require(checkpoint_path.is_file() and observation_path.is_file(),
                     f"arm {arm} checkpoint or observation at step {step} is missing")
            interim = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            optimizer_steps = [int(torch.as_tensor(value["step"]).item())
                               for value in interim["optimizer_state"]["state"].values()
                               if isinstance(value, Mapping) and "step" in value]
            next_batch = interim.get("next_batch_indices")
            schedule_matches = next_batch is None if step == 500 else (
                torch.is_tensor(next_batch)
                and torch.equal(next_batch, schedule.reshape(500, CLASS_COUNT)[step]))
            _require(interim.get("task_id") == TASK_ID and interim.get("arm") == arm
                     and interim.get("run_id") == identity.run_id
                     and interim.get("completed_updates") == step
                     and interim.get("history_count") == step
                     and interim.get("optimizer_deepcopied") is True
                     and ((step == 0 and not optimizer_steps)
                          or (step > 0 and len(optimizer_steps) == 4
                              and all(value == step for value in optimizer_steps)))
                     and schedule_matches,
                     f"arm {arm} saved checkpoint step {step} identity is invalid")
        arm_results[arm] = {"history_updates": len(history),
                            "adam_parameter_steps": [500, 500, 500, 500],
                            "summary": observed["summary"], "all_raw_rows_recomputed": True}
    branch, interpretation = _classify(arm_results["A"], arm_results["B"])
    _require(decision.get("branch") == branch and decision.get("branch_interpretation") == interpretation,
             "fit20 final branch does not follow the two independently consumed arm results")
    report_path = evidence_dir / "final_report.md"
    _require(report_path.is_file(), "fit20 final five-section report is missing")
    report = report_path.read_text(encoding="utf-8")
    sections = ("## 结果与状态", "## 科学认识", "## 目标缺口",
                "## 工程完成与验证", "## 决策影响与下一步")
    _require(all(section in report for section in sections),
             "fit20 final report does not contain the five required sections")
    receipt = {"task_id": TASK_ID, "run_id": identity.run_id,
               "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
               "branch": branch, "independent_numpy_metric_recomputation": True,
               "all_20_rows_recomputed_per_arm": True,
               "training_history_schedule_and_adam_verified": True,
               "recovery_evidence_verified": True, "file_size_reconciliation": size_report,
               "arms": arm_results}
    _atomic_json(evidence_dir / "windows_reconsumption.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "execute", "resume", "measure", "verify-saved",
                                 "review-saved", "verify-review",
                                 "recovery-worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path,
                        help="required explicit checkpoint for resume or isolated recovery")
    parser.add_argument("--arm", choices=ARM_NAMES,
                        help="arm identity for resume or isolated recovery")
    parser.add_argument("--evidence-dir", type=Path,
                        help="explicit attempt path for verify-saved")
    parser.add_argument("--review-dir", type=Path,
                        help="explicit new evidence repair directory")
    parser.add_argument("--recovery-dir", type=Path,
                        help="explicit remote preflight engineering_recovery directory")
    parser.add_argument("--worker-output", type=Path,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.mode in ("review-saved", "verify-review"):
        _require(args.evidence_dir is not None and args.review_dir is not None,
                 f"{args.mode} requires explicit --evidence-dir and --review-dir", ValueError)
        if args.mode == "review-saved":
            _require(args.recovery_dir is not None,
                     "review-saved requires explicit --recovery-dir", ValueError)
        else:
            _require(args.recovery_dir is None,
                     "verify-review reads recovery evidence from --review-dir", ValueError)
        from scripts.research.lightgen.review_lightgen_g1_fit20_evidence_repair_v1 import main as review_main

        forwarded = ["--mode", args.mode, "--config", str(args.config.resolve()),
                     "--evidence-dir", str(args.evidence_dir.resolve()),
                     "--review-dir", str(args.review_dir.resolve())]
        if args.recovery_dir is not None:
            forwarded.extend(("--recovery-dir", str(args.recovery_dir.resolve())))
        return review_main(forwarded)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.mode == "preflight":
        result = _preflight(config_path, config)
    elif args.mode == "recovery-worker":
        _require(args.checkpoint is not None and args.arm is not None and args.worker_output is not None,
                 "recovery-worker requires explicit arm, checkpoint, and output paths", ValueError)
        _recovery_worker(config_path, args.arm, args.checkpoint.resolve(), args.worker_output.resolve())
        result = {"status": "ENGINEERING_RECOVERY_STEP_COMPLETE", "arm": args.arm}
    elif args.mode == "verify-saved":
        _require(args.evidence_dir is not None, "verify-saved requires an explicit --evidence-dir", ValueError)
        result = _verify_saved(args.evidence_dir.resolve(), _load_inputs(config))
    elif args.mode == "measure":
        result = _measure_terminal(config, _load_inputs(config))
    elif args.mode == "resume":
        _require(args.checkpoint is not None and args.arm is not None,
                 "resume requires an explicit --checkpoint and --arm", ValueError)
        result = _execute(config_path, config, resume_checkpoint=args.checkpoint.resolve(),
                          resume_arm=args.arm)
    else:
        _require(args.checkpoint is None and args.arm is None,
                 "execute does not accept a checkpoint or arm override", ValueError)
        result = _execute(config_path, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

