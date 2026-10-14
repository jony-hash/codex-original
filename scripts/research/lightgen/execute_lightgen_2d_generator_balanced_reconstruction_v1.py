"""Run the fixed-population LightGen Generator-only reconstruction contrast."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic


TASK_ID = "M0_GENERATOR_BALANCED_RECONSTRUCTION_V1"
REPORT_SCHEMA = "m0_generator_balanced_reconstruction_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_balanced_reconstruction_preflight.v1"
FIT_ROWS = 200
OBSERVATION_ROWS = 100
BATCH_SIZE = 10
BATCH_COUNT = FIT_ROWS // BATCH_SIZE
UPDATES_PER_ARM = 1000
CHECKPOINT_INTERVAL = 100
RESOURCE_TELEMETRY_NAME = "resource_telemetry_generator_balanced.jsonl"
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
DISPLAY_CLASSES = (1, 7, 0, 4, 8)
BOOTSTRAP_SEED = 2026100203
DELTA_SIC = 1.365988374680088e-5
KAPPA = 0.6936758419764932
ALPHA = 0.01596223111734585
BETA = 2.4509953404250195
GAMMA = 0.5330424284576345
ARM_NAMES = ("A_original", "B_balanced")
DEFAULT_DIAGNOSTIC = Path("outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01")
DEFAULT_OUTPUT = Path("outputs/20261002_m0_generator_balanced_reconstruction_v1/runs/attempt_01")
CHECKPOINT_PATTERN = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")


class ReconstructionRunError(RuntimeError):
    """An invalid input, failed runtime invariant, or incomplete task result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconstructionRunError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def balanced_reconstruction_loss(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average per-image foreground/background MSEs with equal half weights."""
    _require(raw.shape == target.shape and raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "balanced reconstruction expects matching [B,1,200,200] tensors")
    _require(raw.shape[0] > 0, "balanced reconstruction batch cannot be empty")
    _require(bool(torch.isfinite(raw).all().item()) and bool(torch.isfinite(target).all().item()), "balanced reconstruction inputs must be finite")
    foreground = target > 0.1
    foreground_count = foreground.flatten(1).sum(dim=1)
    background_count = (~foreground).flatten(1).sum(dim=1)
    _require(bool((foreground_count > 0).all().item()), "balanced reconstruction has an image with an empty foreground mask")
    _require(bool((background_count > 0).all().item()), "balanced reconstruction has an image with an empty background mask")
    squared = (raw - target).square()
    foreground_mse = (squared * foreground).flatten(1).sum(dim=1) / foreground_count
    background_mse = (squared * (~foreground)).flatten(1).sum(dim=1) / background_count
    value = (0.5 * foreground_mse + 0.5 * background_mse).mean()
    _require(bool(torch.isfinite(value).item()), "balanced reconstruction loss is nonfinite")
    return value


def training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_index: int,
    *,
    data_role: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one of the fixed first 200 TRAIN rows; observation rows cannot enter."""
    _require(data_role == "train", "training batch must be canonical TRAIN")
    _require(images.shape[0] == FIT_ROWS + OBSERVATION_ROWS and labels.shape == (images.shape[0],), "training batch population must be the fixed 300-row input")
    _require(0 <= batch_index < BATCH_COUNT, "training batch index is outside the fixed 20-batch order")
    positions = torch.arange(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE, dtype=torch.long)
    _require(bool((positions < FIT_ROWS).all().item()), "observation rows are forbidden in training gradients")
    return images.index_select(0, positions), labels.index_select(0, positions), positions


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], name: str) -> None:
    _require(set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left), f"{name} state changed")


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return isinstance(left, type(right)) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _repeatability_difference(left: Any, right: Any, path: str = "root") -> tuple[bool, dict[str, Any]]:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype):
            return False, {"path": path, "reason": "tensor metadata differs"}
        if not (left.is_floating_point() or left.is_complex()):
            equal = torch.equal(left, right)
            return equal, {"path": path, "max_absolute_difference": 0.0 if equal else None}
        difference = (left - right).abs()
        max_absolute = float(difference.max().item()) if difference.numel() else 0.0
        if path.endswith("generator.phase"):
            rtol, atol = 2e-6, 5e-7
        elif path.endswith("exp_avg_sq"):
            rtol, atol = 2e-6, 1e-14
        elif path.endswith("exp_avg"):
            rtol, atol = 2e-6, 5e-10
        else:
            rtol, atol = 2e-6, 5e-10
        equal = torch.allclose(left, right, rtol=rtol, atol=atol)
        return equal, {"path": path, "max_absolute_difference": max_absolute, "rtol": rtol, "atol": atol}
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right)):
            return False, {"path": path, "reason": "mapping keys differ"}
        details = []
        for key in left:
            equal, detail = _repeatability_difference(left[key], right[key], f"{path}.{key}")
            details.append(detail)
            if not equal:
                return False, detail
        worst = max(details, key=lambda item: float(item.get("max_absolute_difference", 0.0))) if details else {"path": path, "max_absolute_difference": 0.0}
        return True, worst
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not (isinstance(left, type(right)) and len(left) == len(right)):
            return False, {"path": path, "reason": "sequence structure differs"}
        details = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            equal, detail = _repeatability_difference(a, b, f"{path}[{index}]")
            details.append(detail)
            if not equal:
                return False, detail
        worst = max(details, key=lambda item: float(item.get("max_absolute_difference", 0.0))) if details else {"path": path, "max_absolute_difference": 0.0}
        return True, worst
    if isinstance(left, float) or isinstance(right, float):
        equal = isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=5e-10)
        return equal, {"path": path, "left": left, "right": right, "rtol": 2e-6, "atol": 5e-10}
    equal = left == right
    return equal, {"path": path, "left": left, "right": right}


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    return copy.deepcopy(value)


def _rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.random.get_rng_state().cpu().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "python": random.getstate(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    _require(torch.is_tensor(state.get("torch_cpu")), "checkpoint CPU RNG state is missing")
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    _require(isinstance(cuda_states, list) and len(cuda_states) == torch.cuda.device_count(), "checkpoint CUDA RNG device count differs")
    torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    python_state = state.get("python")
    _require(isinstance(python_state, tuple), "checkpoint Python RNG state is missing")
    random.setstate(python_state)


def _sample_resource_telemetry(
    target: Path,
    device: torch.device,
    phase: str,
    *,
    progress: Mapping[str, Any] | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    sample = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress or {}),
        "elapsed_seconds_observation_only": None if started is None else time.perf_counter() - started,
        "resource": diagnostic._resource_snapshot(device),
    }
    with (target / RESOURCE_TELEMETRY_NAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    _require(sample["resource"]["global_used_fraction"] <= 0.90, "PREFLIGHT_BLOCKED(resource): whole-device use exceeded the 90% safety line")
    return sample


def _diag_paths(root: Path, diagnostic_dir: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    population_path = diagnostic_dir / "input_population.json"
    report_path = diagnostic_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "completed frozen diagnostic must supply input_population.json and report.json")
    population = json.loads(population_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(population.get("task_id") == "M0_CLASS17_MAPPING_DIAGNOSTIC_V2", "diagnostic population task identity mismatch")
    _require(report.get("task_id") == "M0_CLASS17_MAPPING_DIAGNOSTIC_V2" and report.get("status") == "VALID_COMPLETED", "diagnostic report is not a valid completed measurement")
    _require(report.get("classification") == "STANDARD_DIGIT_PRESENTATION_CANDIDATE", "diagnostic did not authorize this declared reconstruction hypothesis")
    input_paths = population.get("input_paths")
    _require(isinstance(input_paths, Mapping), "diagnostic input path manifest is missing")
    names = ("step1200", "step2200", "readout1200", "readout2200", "training_data", "e_s", "e_i", "e_c", "e_d")
    paths = {name: _resolve(root, input_paths[name]) for name in names if name in input_paths}
    _require(set(paths) == set(names), "diagnostic input path manifest is incomplete")
    for name, path in paths.items():
        _require(path.is_file(), f"diagnostic input is missing for {name}: {path}")
    return paths, population, report


def _population_record(inputs: Mapping[str, Any], paths: Mapping[str, Path], diagnostic_dir: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "diagnostic_dir": diagnostic_dir.as_posix(),
        "diagnostic_classification": "STANDARD_DIGIT_PRESENTATION_CANDIDATE",
        "input_paths": {name: paths[name].as_posix() for name in sorted(paths)},
        "source_row_indices": inputs["source_indices"].tolist(),
        "labels": inputs["labels"].tolist(),
        "fit_rows": FIT_ROWS,
        "observation_rows": OBSERVATION_ROWS,
        "fit_batch_size": BATCH_SIZE,
        "fit_batch_count": BATCH_COUNT,
        "focus_classes": list(FOCUS_CLASSES),
        "control_classes": list(CONTROL_CLASSES),
        "provenance": "old readouts establish only fixed row order and labels; all model outputs are freshly measured",
    }


def _prepare_output(root: Path, target: Path, *, resume: bool) -> None:
    output_root = (root / "outputs").resolve()
    target = target.resolve()
    _require(target != output_root and output_root in target.parents, "output directory must be a named child of repository outputs/")
    parts = target.relative_to(output_root).parts
    _require(
        len(parts) == 3 and re.fullmatch(r"\d{8}_[A-Za-z0-9][A-Za-z0-9_-]*", parts[0]) is not None
        and parts[1] == "runs" and not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts),
        "output directory must be a named versioned run: outputs/YYYYMMDD_<version>/runs/<run>",
    )
    if target.exists():
        _require(target.is_dir(), "output target exists and is not a directory")
        _require(resume, f"output directory already exists; specify --resume and an explicit --checkpoint: {target}")
    else:
        _require(not resume, f"--resume output directory does not exist: {target}")
        target.mkdir(parents=True, exist_ok=False)


def _load_inputs(root: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    return diagnostic.load_inputs(root, paths)


def _new_endpoint(inputs: Mapping[str, Any], device: torch.device) -> tuple[Any, torch.optim.Optimizer, dict[str, torch.nn.Parameter], Any]:
    consumer, optimizer, parameters, references, _weights = diagnostic._new_endpoint(
        "step2200",
        inputs["payloads"]["step2200"],
        inputs["references"],
        inputs["weights"],
        device,
    )
    _require(abs(consumer.weights.alpha - ALPHA) < 1e-15 and abs(consumer.weights.beta - BETA) < 1e-15 and abs(consumer.weights.gamma - GAMMA) < 1e-15, "step-1200 inherited BOGT weights differ from the declared contrast")
    _require(consumer.runtime.config.propagation_grid_size == 2048, "Generator contrast requires the production 2048 propagation grid")
    return consumer, optimizer, parameters, references


def _gradient(loss: torch.Tensor, parameters: Mapping[str, torch.nn.Parameter], *, retain_graph: bool) -> dict[str, torch.Tensor]:
    names = ("encoder.phase", "generator.phase")
    values = torch.autograd.grad(loss, tuple(parameters[name] for name in names), retain_graph=retain_graph, allow_unused=True)
    return {name: (torch.zeros_like(parameters[name]) if value is None else value.detach()) for name, value in zip(names, values, strict=True)}


def apply_generator_only_adam_update(
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    generator_gradient: torch.Tensor,
) -> None:
    """Apply one Adam step while keeping the Encoder gradient slot empty."""
    _require(generator_gradient.shape == parameters["generator.phase"].shape, "Generator gradient shape differs from its parameter")
    _require(bool(torch.isfinite(generator_gradient).all().item()), "Generator gradient contains nonfinite values")
    encoder_state = optimizer.state[parameters["encoder.phase"]]
    encoder_step = float(encoder_state["step"].item()) if "step" in encoder_state else None
    optimizer.zero_grad(set_to_none=True)
    parameters["generator.phase"].grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient would enter the optimizer update")
    optimizer.step()
    _require(parameters["encoder.phase"].grad is None, "Encoder acquired an update gradient")
    if encoder_step is not None:
        _require(float(optimizer.state[parameters["encoder.phase"]]["step"].item()) == encoder_step, "Encoder Adam step changed")
    optimizer.zero_grad(set_to_none=True)


def _train_step(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    evaluators: Mapping[str, torch.nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    update_index: int,
    *,
    arm: str,
) -> dict[str, Any]:
    batch_cpu, labels_cpu, positions = training_batch(images, labels, update_index % BATCH_COUNT)
    _require(labels_cpu.numel() == BATCH_SIZE and torch.bincount(labels_cpu, minlength=10).tolist() == [1] * 10, "fixed-order training batch is not one row per digit")
    batch = batch_cpu.to(consumer.device)
    breakdown = consumer.forward_loss(batch, data_role=bogt.TRAIN_ROLE)
    result = breakdown.result
    target_raw = result.image_input.slm_intensity
    raw = result.raw_intensity
    raw_mse = (raw - target_raw).square().mean()
    rec_loss = raw_mse if arm == "A_original" else balanced_reconstruction_loss(raw, target_raw)
    semantic_losses = diagnostic._loss_reductions(
        {"result": result},
        batch,
        evaluators["E_S"],
        evaluators["E_I"],
        evaluators["E_C"],
    )
    losses = {
        "L_dist": breakdown.L_dist,
        "L_rec": rec_loss,
        "L_op": breakdown.L_op,
        **semantic_losses,
    }
    loss_names = tuple(losses)
    gradients = {
        name: _gradient(losses[name], parameters, retain_graph=index < len(loss_names) - 1)
        for index, name in enumerate(loss_names)
    }
    bogt_gradient = {
        name: (
            ALPHA * gradients["L_dist"][name]
            + BETA * gradients["L_rec"][name]
            + GAMMA * gradients["L_op"][name]
        )
        for name in ("encoder.phase", "generator.phase")
    }
    canonical_gradient, unavailable = diagnostic._safe_canonical_gradient(gradients)
    _require(canonical_gradient is not None, f"canonical joint semantic direction is unavailable: {unavailable}")
    combined_gradient = {
        name: bogt_gradient[name] + KAPPA * canonical_gradient[name]
        for name in ("encoder.phase", "generator.phase")
    }
    _require(all(bool(torch.isfinite(value).all().item()) for value in combined_gradient.values()), "combined training gradient is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only gradient computation populated optical .grad buffers")
    optimizer_parameters = optimizer.param_groups[0]["params"]
    _require(len(optimizer_parameters) == 2 and all(actual is expected for actual, expected in zip(optimizer_parameters, (parameters["encoder.phase"], parameters["generator.phase"]), strict=True)), "Adam parameter-group order changed")
    apply_generator_only_adam_update(optimizer, parameters, combined_gradient["generator.phase"])
    _require(all(bool(torch.isfinite(parameter).all().item()) for parameter in parameters.values()), "optical phase became nonfinite")
    generator_state = optimizer.state[parameters["generator.phase"]]
    encoder_state = optimizer.state[parameters["encoder.phase"]]
    _require(int(float(encoder_state["step"].item())) == 2200, "Encoder Adam step changed")
    _require(int(float(generator_state["step"].item())) == 2200 + update_index + 1, "Generator Adam step did not advance once")
    record = {
        "update": update_index + 1,
        "batch_index": int(positions[0].item() // BATCH_SIZE),
        "fit_positions": positions.tolist(),
        "labels": labels_cpu.tolist(),
        "arm": arm,
        "training_reconstruction_loss": float(rec_loss.detach().cpu()),
        "common_original_raw_mse_monitor": float(raw_mse.detach().cpu()),
        "l_dist": float(breakdown.L_dist.detach().cpu()),
        "l_op": float(breakdown.L_op.detach().cpu()),
        "semantic_feature_losses": {key: float(value.detach().cpu()) for key, value in semantic_losses.items()},
        "encoder_gradient_norm_read_only": float(combined_gradient["encoder.phase"].to(torch.float64).norm().cpu()),
        "generator_gradient_norm": float(combined_gradient["generator.phase"].to(torch.float64).norm().cpu()),
        "encoder_adam_step": 2200,
        "generator_adam_step": 2200 + update_index + 1,
    }
    del breakdown, result, raw, target_raw, losses, semantic_losses, gradients, bogt_gradient, canonical_gradient, combined_gradient
    return record


def _parameters_by_name(consumer: Any) -> dict[str, torch.nn.Parameter]:
    return dict(consumer.runtime.named_parameters())


def _preflight(inputs: Mapping[str, Any], paths: Mapping[str, Path], target: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "PREFLIGHT_BLOCKED(resource): CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    output_budget = diagnostic._check_output_budget(target)
    cuda_before = diagnostic._check_gpu_budget(device)
    started = time.perf_counter()
    _sample_resource_telemetry(target, device, "preflight_start", progress={"arms": 2, "updates_per_arm": UPDATES_PER_ARM}, started=started)
    evaluators = diagnostic.load_evaluators(paths, device)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    target_logits = diagnostic._target_e_d_logits(evaluators["E_D"], inputs["images"], device)
    target_predictions = target_logits.argmax(dim=1).cpu()
    target_metrics = {
        "fit": diagnostic._metrics_with_intervals(inputs["labels"][:FIT_ROWS], target_predictions[:FIT_ROWS]),
        "observation": diagnostic._metrics_with_intervals(inputs["labels"][FIT_ROWS:], target_predictions[FIT_ROWS:]),
        "combined": diagnostic._metrics_with_intervals(inputs["labels"], target_predictions),
    }
    _require(all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()), "an evaluator acquired a gradient during target control")
    run_records: dict[str, Any] = {}
    trial_configs = (("A_original_primary", "A_original"), ("A_original_replay", "A_original"), ("B_balanced", "B_balanced"))
    snapshots: dict[str, list[dict[str, Any]]] = {}
    preflight_rng = _rng_state()
    for trial_name, arm in trial_configs:
        _restore_rng_state(preflight_rng)
        consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
        model_before = _state_snapshot(consumer.runtime)
        encoder_before = parameters["encoder.phase"].detach().cpu().clone()
        encoder_adam_before = _cpu_tree(optimizer.state[parameters["encoder.phase"]])
        optimizer_encoder_steps: list[int] = []
        snapshots[trial_name] = []
        records = []
        for update in range(3):
            diagnostic._check_gpu_budget(device)
            records.append(_train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], update, arm=arm))
            optimizer_encoder_steps.append(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())))
            _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{trial_name} preflight changed Encoder weights")
            _assert_state_equal({key: value for key, value in model_before.items() if key not in ("encoder.phase", "generator.phase")}, {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}, f"{trial_name} frozen optical runtime")
            _require(_tree_equal(encoder_adam_before, _cpu_tree(optimizer.state[parameters["encoder.phase"]])), f"{trial_name} preflight changed Encoder Adam state")
            _require(all(parameter.grad is None for parameter in parameters.values()), f"{trial_name} retained an optical gradient")
            _require(all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters()), "an evaluator left the frozen autograd boundary")
            snapshots[trial_name].append({
                **{name: value.detach().cpu().clone() for name, value in parameters.items()},
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
            })
        run_records[trial_name] = {
            "updates": 3,
            "arm_objective": arm,
            "encoder_adam_steps": optimizer_encoder_steps,
            "generator_adam_step": int(float(optimizer.state[parameters["generator.phase"]]["step"].item())),
            "encoder_frozen": True,
            "only_generator_grad_assigned": True,
            "loss_records": records,
        }
        del consumer, optimizer, parameters
        gc.collect()
        torch.cuda.empty_cache()
    repeatability_checks = []
    for first, replay in zip(snapshots["A_original_primary"], snapshots["A_original_replay"], strict=True):
        equivalent, difference = _repeatability_difference(first, replay)
        _require(equivalent, f"A/A replay exceeded the declared floating-point repeatability tolerance: {difference}")
        repeatability_checks.append(difference)
    _require(len(snapshots["A_original_primary"]) == 3, "A/A preflight did not cover three real updates")
    _restore_rng_state(preflight_rng)
    for name, model in evaluators.items():
        _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{name} evaluator")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{name} evaluator is not frozen and gradient-free")
    del evaluators
    gc.collect()
    torch.cuda.empty_cache()
    cuda_after = diagnostic._check_gpu_budget(device)
    _sample_resource_telemetry(target, device, "preflight_ready", progress={"trials": 3, "disposable_generator_updates": 9}, started=started)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "status": "READY",
        "device": str(device),
        "production_grid": 2048,
        "output_budget": output_budget,
        "cuda_before": cuda_before,
        "cuda_after": cuda_after,
        "target_e_d_positive_control": target_metrics,
        "target_e_d_positive_control_thresholds_met": target_metrics["observation"]["macro_recall"] >= 0.90 and all(target_metrics["observation"]["per_class_correct"][class_id] >= 8 for class_id in FOCUS_CLASSES),
        "trials": run_records,
        "a_a_repeatability": {
            "result": "same model and Adam state reproduced within explicit float32 numerical tolerances",
            "tolerances": {"generator_phase": {"rtol": 2e-6, "atol": 5e-7}, "adam_exp_avg": {"rtol": 2e-6, "atol": 5e-10}, "adam_exp_avg_sq": {"rtol": 2e-6, "atol": 1e-14}, "encoder_phase_and_adam_step": "exact"},
            "per_update_max_difference": repeatability_checks,
        },
        "evaluator_and_scientific_buffer_freeze_checked_each_step": True,
        "duration_seconds_observation_only": time.perf_counter() - started,
    }


def _arm_order(arm: str, completed: int) -> int:
    return completed if arm == "A_original" else 1001 + completed


def _checkpoint_name(arm: str, completed: int) -> str:
    letter = "a" if arm == "A_original" else "b"
    return f"arm_{letter}_step{completed:04d}.pt"


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    completed: int,
    history: Mapping[str, Sequence[Mapping[str, Any]]],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = _parameters_by_name(consumer)
    _require(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())) == 2200, "checkpoint Encoder Adam step is not frozen at 2200")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == 2200 + completed, "checkpoint Generator Adam step does not match its update count")
    return {
        "schema": "m0_generator_balanced_reconstruction_checkpoint.v1",
        "task_id": TASK_ID,
        "arm": arm,
        "completed_updates": completed,
        "next_batch_index": completed % BATCH_COUNT,
        "source_checkpoint_step": 2200,
        "population": dict(population),
        "trainable_parameter_state": consumer.trainable_state_dict(),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_step_by_parameter": {"encoder.phase": 2200, "generator.phase": 2200 + completed},
        "frozen_runtime_state": {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")},
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": {key: [dict(record) for record in records] for key, records in history.items()},
        "rng_state": _rng_state(),
    }


def _write_torch_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    from ldm_project.v2.evidence import atomic_torch_save_new

    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(restored, Mapping) and set(restored) == set(payload), f"checkpoint write/read verification failed: {path.name}")


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, arm: str, completed: int, history: Mapping[str, Sequence[Mapping[str, Any]]], population: Mapping[str, Any]) -> Path:
    path = target / _checkpoint_name(arm, completed)
    diagnostic._check_output_budget(target)
    _write_torch_new(path, _checkpoint_payload(consumer, optimizer, arm, completed, history, population), "Generator-only reconstruction recovery checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.parent.resolve() == target.resolve() and path.is_file(), "resume checkpoint must be an existing file inside this attempt directory")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and payload.get("schema") == "m0_generator_balanced_reconstruction_checkpoint.v1", "resume checkpoint schema mismatch")
    _require(payload.get("task_id") == TASK_ID and payload.get("source_checkpoint_step") == 2200, "resume checkpoint task or source-step mismatch")
    _require(payload.get("population") == dict(population), "resume checkpoint data rows, labels, or input paths differ")
    arm = payload.get("arm")
    completed = payload.get("completed_updates")
    _require(arm in ARM_NAMES and isinstance(completed, int) and 0 <= completed <= UPDATES_PER_ARM and completed % CHECKPOINT_INTERVAL == 0, "resume checkpoint arm or update count is invalid")
    _require(payload.get("next_batch_index") == completed % BATCH_COUNT, "resume checkpoint batch cursor mismatch")
    expected_name = _checkpoint_name(arm, completed)
    _require(path.name == expected_name, "resume checkpoint filename does not match its saved arm/update state")
    selected_order = _arm_order(arm, completed)
    later = []
    for candidate in target.glob("arm_*_step*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        candidate_arm = "A_original" if match.group(1) == "a" else "B_balanced"
        candidate_step = int(match.group(2))
        if _arm_order(candidate_arm, candidate_step) > selected_order:
            later.append(candidate.name)
    _require(not later, f"explicit checkpoint is older than already saved task state: {sorted(later)}")
    return dict(payload)


def _completed_resume_result(target: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == TASK_ID and report.get("status") == "VALID_COMPLETED", "existing report is not a complete task result")
    _require(checkpoint.get("arm") == ARM_NAMES[1] and int(checkpoint.get("completed_updates", -1)) == UPDATES_PER_ARM, "completed attempt can only be restored from the explicit final B checkpoint")
    return dict(report)


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    consumer.load_trainable_state_dict(payload["trainable_parameter_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    parameters = _parameters_by_name(consumer)
    _require(int(float(optimizer.state[parameters["encoder.phase"]]["step"].item())) == 2200, "restored Encoder Adam step differs from source")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == 2200 + int(payload["completed_updates"]), "restored Generator Adam step differs from checkpoint")
    _restore_rng_state(payload["rng_state"])
    restored_frozen = {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}
    _assert_state_equal(payload["frozen_runtime_state"], restored_frozen, "restored scientific runtime")


def _run_arm(
    arm: str,
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, torch.nn.Module],
    target: Path,
    population: Mapping[str, Any],
    history: dict[str, list[dict[str, Any]]],
    *,
    start_from: Mapping[str, Any] | None = None,
    started: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    device = torch.device("cuda:0")
    consumer, optimizer, parameters, _references = _new_endpoint(inputs, device)
    if start_from is not None:
        _require(start_from.get("arm") == arm, "resume checkpoint selects a different arm than the next scheduled arm")
        _restore_checkpoint(start_from, consumer, optimizer)
        completed = int(start_from["completed_updates"])
    else:
        completed = 0
        _save_checkpoint(target, consumer, optimizer, arm, 0, history, population)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}
    encoder_adam_before = _cpu_tree(optimizer.state[parameters["encoder.phase"]])
    while completed < UPDATES_PER_ARM:
        diagnostic._check_gpu_budget(device)
        record = _train_step(consumer, optimizer, parameters, evaluators, inputs["images"], inputs["labels"], completed, arm=arm)
        completed += 1
        history[arm].append(record)
        if completed % CHECKPOINT_INTERVAL == 0:
            checkpoint = _save_checkpoint(target, consumer, optimizer, arm, completed, history, population)
            _sample_resource_telemetry(
                target,
                device,
                f"{arm}_checkpoint",
                progress={"arm_updates": completed, "checkpoint": checkpoint.name},
                started=started,
            )
            _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{arm} training changed Encoder weights")
            _assert_state_equal(encoder_adam_before, _cpu_tree(optimizer.state[parameters["encoder.phase"]]), f"{arm} Encoder Adam")
            current_frozen = {key: value for key, value in _state_snapshot(consumer.runtime).items() if key not in ("encoder.phase", "generator.phase")}
            _assert_state_equal(frozen_before, current_frozen, f"{arm} scientific buffers")
            for name, model in evaluators.items():
                _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} {name} evaluator")
                _require(all(parameter.grad is None for parameter in model.parameters()), f"{arm} {name} retained gradients")
            gc.collect()
            torch.cuda.empty_cache()
    _require(completed == UPDATES_PER_ARM and len(history[arm]) == UPDATES_PER_ARM, f"{arm} did not complete exactly 1000 updates")
    _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{arm} final Encoder weights changed")
    _assert_state_equal(encoder_adam_before, _cpu_tree(optimizer.state[parameters["encoder.phase"]]), f"{arm} final Encoder Adam")
    readouts = diagnostic._forward_readouts(consumer, evaluators, inputs["images"], batch_size=BATCH_SIZE)
    readouts["source_indices"] = inputs["source_indices"].clone()
    readouts["labels"] = inputs["labels"].clone()
    metrics, _per_row = diagnostic._aggregate_image_statistics(
        {**readouts, "input_images": inputs["images"]},
        inputs["labels"],
        inputs["source_indices"],
    )
    result = {
        "arm": arm,
        "completed_updates": completed,
        "checkpoint_step": 2200 + completed,
        "parameter_state": consumer.trainable_state_dict(),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "readouts": readouts,
        "image_statistics": metrics,
    }
    for name, model in evaluators.items():
        _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} postmeasurement {name} evaluator")
        _require(all(parameter.grad is None for parameter in model.parameters()), f"{arm} postmeasurement {name} retained gradients")
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result, readouts


def _class_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=1).cpu()
    labels_cpu = labels.cpu()
    return {
        "fit": diagnostic._metrics_with_intervals(labels_cpu[:FIT_ROWS], predictions[:FIT_ROWS]),
        "observation": diagnostic._metrics_with_intervals(labels_cpu[FIT_ROWS:], predictions[FIT_ROWS:]),
        "combined": diagnostic._metrics_with_intervals(labels_cpu, predictions),
        "predictions": predictions,
    }


def _validate_readout_population(readouts: Mapping[str, Any], inputs: Mapping[str, Any], name: str) -> None:
    _require(torch.is_tensor(readouts.get("source_indices")) and torch.equal(readouts["source_indices"].cpu(), inputs["source_indices"]), f"saved {name} readout source rows differ")
    _require(torch.is_tensor(readouts.get("labels")) and torch.equal(readouts["labels"].cpu(), inputs["labels"]), f"saved {name} readout labels differ")
    _require(tuple(readouts["raw_cmos"].shape) == (300, 1, 200, 200), f"saved {name} raw CMOS shape differs")
    _require(tuple(readouts["z"].shape) == (300, 100) and readouts["z"].dtype == torch.complex64, f"saved {name} OLS coefficients differ")
    _require(tuple(readouts["e_d_logits"].shape) == (300, 10), f"saved {name} E_D logits shape differs")
    for key in ("raw_cmos", "z", "target_slm_intensity", "power_by_plane", "e_d_logits"):
        _require(bool(torch.isfinite(readouts[key].real if torch.is_complex(readouts[key]) else readouts[key]).all().item()), f"saved {name} {key} contains nonfinite values")


def _exact_mcnemar_better(a_correct: torch.Tensor, b_correct: torch.Tensor) -> dict[str, Any]:
    _require(a_correct.shape == b_correct.shape and a_correct.numel() == 10, "McNemar requires paired ten-row class observations")
    wins = int((b_correct & ~a_correct).sum().item())
    losses = int((a_correct & ~b_correct).sum().item())
    discordant = wins + losses
    pvalue = 1.0 if discordant == 0 else sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2**discordant)
    return {"b_only_correct": wins, "a_only_correct": losses, "discordant_pairs": discordant, "one_sided_exact_p": float(min(1.0, pvalue))}


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted


def _paired_macro_bootstrap(
    labels: torch.Tensor,
    a_predictions: torch.Tensor,
    b_predictions: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = labels[FIT_ROWS:].cpu()
    a_predictions = a_predictions[FIT_ROWS:].cpu()
    b_predictions = b_predictions[FIT_ROWS:].cpu()
    per_class_positions = [torch.nonzero(labels == class_id, as_tuple=False).flatten() for class_id in range(10)]
    _require(all(position.numel() == 10 for position in per_class_positions), "paired bootstrap requires ten observations per class")
    draws = 10_000
    differences = torch.empty(draws, dtype=torch.float64)
    for index in range(draws):
        class_differences = []
        for class_id, positions in enumerate(per_class_positions):
            sampled = positions[torch.randint(0, positions.numel(), (positions.numel(),), generator=generator)]
            a_rate = (a_predictions[sampled] == labels[sampled]).to(torch.float64).mean()
            b_rate = (b_predictions[sampled] == labels[sampled]).to(torch.float64).mean()
            class_differences.append(b_rate - a_rate)
        differences[index] = torch.stack(class_differences).mean()
    bounds = torch.quantile(differences, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"resamples": draws, "seed": seed, "paired_stratified_macro_recall_difference_B_minus_A_95_percentile_interval": [float(bounds[0]), float(bounds[1])], "descriptive_only": True}


def _positive_control_ok(target_metrics: Mapping[str, Any]) -> bool:
    return float(target_metrics["observation"]["macro_recall"]) >= 0.90 and all(int(target_metrics["observation"]["per_class_correct"][class_id]) >= 8 for class_id in FOCUS_CLASSES)


def _decision(
    target_metrics: Mapping[str, Any],
    endpoint_metrics: Mapping[str, Mapping[str, Any]],
    image_statistics: Mapping[str, Mapping[str, Any]],
    paired_tests: Mapping[str, Any],
) -> dict[str, Any]:
    if not _positive_control_ok(target_metrics):
        return {
            "measurement_qualified": False,
            "classification": None,
            "training_recommendation": "NONE",
            "reason": "fresh target-image E_D positive control did not meet its declared threshold",
        }
    start = endpoint_metrics["step2200"]["observation"]
    a_metrics = endpoint_metrics["A_original"]["observation"]
    b_metrics = endpoint_metrics["B_balanced"]["observation"]
    start_macro = float(start["macro_recall"])
    a_macro = float(a_metrics["macro_recall"])
    b_macro = float(b_metrics["macro_recall"])
    counts_b = b_metrics["per_class_correct"]
    counts_a = a_metrics["per_class_correct"]
    counts_start = start["per_class_correct"]
    pair_ok = all(int(counts_b[class_id]) >= 5 and float(paired_tests["holm_adjusted_p"][str(class_id)]) <= 0.05 for class_id in FOCUS_CLASSES)
    macro_ok = b_macro >= a_macro + 0.05 and b_macro >= start_macro
    class_retention_ok = all(int(counts_b[class_id]) >= max(int(counts_a[class_id]), int(counts_start[class_id])) - 1 for class_id in range(10))
    image_stats_ok = True
    clipping_ok = True
    for class_id in FOCUS_CLASSES:
        b_image = image_statistics["B_balanced"]["by_class"][str(class_id)]["observation"]
        a_image = image_statistics["A_original"]["by_class"][str(class_id)]["observation"]
        start_image = image_statistics["step2200"]["by_class"][str(class_id)]["observation"]
        image_stats_ok &= float(b_image["raw_foreground_mse"]) <= 0.90 * min(float(a_image["raw_foreground_mse"]), float(start_image["raw_foreground_mse"]))
        clipping_ok &= float(b_image["raw_fraction_above_one"]) <= 0.01
    b_overall = float(image_statistics["B_balanced"]["overall"]["observation"]["raw_mse"])
    a_overall = float(image_statistics["A_original"]["overall"]["observation"]["raw_mse"])
    start_overall = float(image_statistics["step2200"]["overall"]["observation"]["raw_mse"])
    overall_ok = b_overall <= 1.05 * a_overall and b_overall <= 1.05 * start_overall
    criteria = {
        "focus_classes_each_at_least_5_of_10_and_holm_mcnemar_p_le_0_05": pair_ok,
        "macro_recall_at_least_A_plus_0_05_and_not_below_start": macro_ok,
        "no_class_loses_more_than_one_correct_vs_A_or_start": class_retention_ok,
        "focus_class_raw_foreground_mse_at_least_10_percent_below_A_and_start": image_stats_ok,
        "overall_raw_mse_no_more_than_5_percent_above_A_or_start": overall_ok,
        "focus_class_raw_software_clipping_at_most_1_percent": clipping_ok,
    }
    all_met = all(criteria.values())
    if all_met:
        classification = "BALANCED_RECONSTRUCTION_SUPPORTED"
        recommendation = "将前景/背景各0.5权重的200×200重建目标，作为下一项受控主线研究候选；本结论仅适用于本次固定TRAIN人口、step2200初始化与1000次Generator-only更新。"
        reason = "all preregistered observation-set support criteria were met"
    else:
        focus_improvement = []
        for class_id in FOCUS_CLASSES:
            bm = image_statistics["B_balanced"]["by_class"][str(class_id)]["observation"]["raw_foreground_mse"]
            am = image_statistics["A_original"]["by_class"][str(class_id)]["observation"]["raw_foreground_mse"]
            sm = image_statistics["step2200"]["by_class"][str(class_id)]["observation"]["raw_foreground_mse"]
            count_gain = int(counts_b[class_id]) > max(int(counts_a[class_id]), int(counts_start[class_id]))
            focus_improvement.append(float(bm) < min(float(am), float(sm)) or count_gain)
        classification = "MIXED_OR_INSUFFICIENT" if any(focus_improvement) else "NOT_SUPPORTED_WITHIN_SCOPE"
        recommendation = "NONE"
        reason = "one or more declared support criteria were not met; no successor training is recommended"
    return {"measurement_qualified": True, "classification": classification, "training_recommendation": recommendation, "reason": reason, "criteria": criteria}


def _readout_report(
    inputs: Mapping[str, Any],
    readouts_by_endpoint: Mapping[str, Mapping[str, torch.Tensor]],
    image_statistics: Mapping[str, Mapping[str, Any]],
    evaluators: Mapping[str, torch.nn.Module],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = inputs["labels"].cpu()
    endpoint_metrics = {
        name: _class_metrics(labels, readouts["e_d_logits"])
        for name, readouts in readouts_by_endpoint.items()
    }
    target_logits = diagnostic._target_e_d_logits(evaluators["E_D"], inputs["images"], device).cpu()
    target_metrics = _class_metrics(labels, target_logits)
    return endpoint_metrics, {"logits": target_logits, "metrics": target_metrics}


def _raw_mse(
    image_stats: dict[str, Any],
    per_row: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> None:
    positions_by_split = {
        "fit": torch.arange(FIT_ROWS),
        "observation": torch.arange(FIT_ROWS, FIT_ROWS + OBSERVATION_ROWS),
        "combined": torch.arange(FIT_ROWS + OBSERVATION_ROWS),
    }
    overall: dict[str, Any] = {}
    for split, positions in positions_by_split.items():
        summary: dict[str, Any] = {}
        for key, tensor in per_row.items():
            if tensor.ndim != 1 or tensor.shape[0] != FIT_ROWS + OBSERVATION_ROWS:
                continue
            selected = tensor.index_select(0, positions)
            if tensor.dtype == torch.bool:
                summary[key] = int(selected.sum().item())
            else:
                summary[key] = float(selected.to(torch.float64).mean().item())
        raw_denominator = summary["raw_foreground_pixels"] + summary["raw_background_pixels"]
        view_denominator = summary["evaluator_view_foreground_pixels"] + summary["evaluator_view_background_pixels"]
        summary["raw_mse"] = (summary["raw_foreground_mse"] * summary["raw_foreground_pixels"] + summary["raw_background_mse"] * summary["raw_background_pixels"]) / raw_denominator
        summary["evaluator_view_mse"] = (summary["evaluator_view_foreground_mse"] * summary["evaluator_view_foreground_pixels"] + summary["evaluator_view_background_mse"] * summary["evaluator_view_background_pixels"]) / view_denominator
        summary["rows"] = int(positions.numel())
        summary["source_row_indices"] = source_indices.index_select(0, positions).tolist()
        summary["labels"] = labels.index_select(0, positions).tolist()
        overall[split] = summary
    image_stats["overall"] = overall
    for classes in image_stats["by_class"].values():
        for summary in classes.values():
            raw_denominator = summary["raw_foreground_pixels"] + summary["raw_background_pixels"]
            view_denominator = summary["evaluator_view_foreground_pixels"] + summary["evaluator_view_background_pixels"]
            summary["raw_mse"] = (summary["raw_foreground_mse"] * summary["raw_foreground_pixels"] + summary["raw_background_mse"] * summary["raw_background_pixels"]) / raw_denominator
            summary["evaluator_view_mse"] = (summary["evaluator_view_foreground_mse"] * summary["evaluator_view_foreground_pixels"] + summary["evaluator_view_background_mse"] * summary["evaluator_view_background_pixels"]) / view_denominator


def _save_measurements(target: Path, name: str, result: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    readouts = result["readouts"]
    diagnostic._check_output_budget(target)
    readout_path = target / f"readouts_{name}.pt"
    if readout_path.exists():
        existing_readouts = torch.load(readout_path, map_location="cpu", weights_only=True)
        _require(_tree_equal(existing_readouts, readouts), f"saved {name} readouts differ from the explicit checkpoint measurement")
    else:
        _write_torch_new(readout_path, readouts, "fresh frozen-population optical readouts")
    image_stats = copy.deepcopy(result["image_statistics"])
    per_row = diagnostic._row_reconstruction_metrics(readouts["raw_cmos"], readouts["target_slm_intensity"], inputs["images"])
    _raw_mse(image_stats, per_row, inputs["labels"], inputs["source_indices"])
    scalar_rows = {key: value for key, value in per_row.items() if value.ndim == 1 and value.shape[0] == FIT_ROWS + OBSERVATION_ROWS}
    row_path = target / f"row_metrics_{name}.pt"
    row_payload = {"source_indices": inputs["source_indices"], "labels": inputs["labels"], **scalar_rows}
    if row_path.exists():
        existing_rows = torch.load(row_path, map_location="cpu", weights_only=True)
        _require(_tree_equal(existing_rows, row_payload), f"saved {name} row metrics differ from the explicit checkpoint measurement")
    else:
        _write_torch_new(row_path, row_payload, "per-row raw and evaluator-view reconstruction metrics")
    stats_path = target / f"image_statistics_{name}.json"
    if stats_path.exists():
        _require(json.loads(stats_path.read_text(encoding="utf-8")) == image_stats, f"saved {name} image summaries differ from the explicit checkpoint measurement")
    else:
        diagnostic._write_json_new(stats_path, image_stats, "fit and observation reconstruction summaries")
    result["image_statistics"] = image_stats
    return image_stats


def _save_training_history(target: Path, history: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for arm in ARM_NAMES:
        path = target / f"training_history_{arm}.json"
        payload = {"task_id": TASK_ID, "arm": arm, "updates": len(history[arm]), "records": list(history[arm])}
        if path.exists():
            _require(json.loads(path.read_text(encoding="utf-8")) == payload, f"saved {arm} training history differs from the explicit recovery checkpoint")
        else:
            diagnostic._write_json_new(path, payload, "Generator-only training history")


def _save_montage(path: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    labels = inputs["labels"]
    target_view = canonical.cmos_view(inputs["images"])
    views = {
        "target": target_view,
        "step2200": canonical.cmos_view(readouts["step2200"]["raw_cmos"]),
        "A_original": canonical.cmos_view(readouts["A_original"]["raw_cmos"]),
        "B_balanced": canonical.cmos_view(readouts["B_balanced"]["raw_cmos"]),
    }
    positions = [position for class_id in DISPLAY_CLASSES for position in torch.nonzero(labels == class_id, as_tuple=False).flatten().tolist()]
    _require(len(positions) == 150, "comparison figure must contain all 150 rows from digits 1, 7, 0, 4, and 8")
    scale = 2
    side = 28 * scale
    label_width = 35
    gap = 8
    tile_width = label_width + len(views) * side + (len(views) - 1) * gap
    tile_height = side + 8
    columns = 2
    rows = math.ceil(len(positions) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    for index, position in enumerate(positions):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        class_id = int(labels[position].item())
        role = "F" if position < FIT_ROWS else "O"
        draw.text((x, y + 4), f"{class_id}{role}", fill=(0, 0, 0))
        for view_index, view in enumerate(views.values()):
            pixels = view[position, 0].detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
            tile = Image.fromarray(pixels, mode="L").resize((side, side), Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(tile, (x + label_width + view_index * (side + gap), y))
    canvas.save(path, format="PNG", optimize=True)
    return {"path": path.name, "rows_displayed": len(positions), "classes": list(DISPLAY_CLASSES), "panel_order": list(views), "view": "28x28 canonical clamp[0,1] then adaptive average pool", "fixed_display_range": [0.0, 1.0], "per_image_rescaling": False, "F/O": {"F": "fit TRAIN row", "O": "observation TRAIN row"}, "source_positions": positions, "source_row_indices": [int(inputs["source_indices"][position]) for position in positions]}


def _finalize(
    target: Path,
    inputs: Mapping[str, Any],
    history: Mapping[str, Sequence[Mapping[str, Any]]],
    start_result: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    population: Mapping[str, Any],
    preflight: Mapping[str, Any],
    target_control: Mapping[str, Any],
    endpoint_metrics: Mapping[str, Any],
    image_statistics: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    readouts = {"step2200": start_result["readouts"], **{name: value["readouts"] for name, value in arms.items()}}
    observation_labels = inputs["labels"][FIT_ROWS:].cpu()
    a_pred = endpoint_metrics["A_original"]["predictions"]
    b_pred = endpoint_metrics["B_balanced"]["predictions"]
    pair_results = {
        str(class_id): _exact_mcnemar_better(
            a_pred[FIT_ROWS:][observation_labels == class_id] == observation_labels[observation_labels == class_id],
            b_pred[FIT_ROWS:][observation_labels == class_id] == observation_labels[observation_labels == class_id],
        )
        for class_id in FOCUS_CLASSES
    }
    adjusted = _holm({key: value["one_sided_exact_p"] for key, value in pair_results.items()})
    paired_tests = {"raw": pair_results, "holm_adjusted_p": adjusted, "alternative": "B has greater per-row E_D correctness than A within each true class"}
    paired_tests["macro_recall_bootstrap_vs_A"] = _paired_macro_bootstrap(inputs["labels"], a_pred, b_pred, seed=BOOTSTRAP_SEED)
    decision = _decision(target_control["metrics"], endpoint_metrics, image_statistics, paired_tests)
    figure_path = target / "comparison_28x28.png"
    if not figure_path.exists():
        montage = _save_montage(figure_path, inputs, readouts)
    else:
        positions = [position for class_id in DISPLAY_CLASSES for position in torch.nonzero(inputs["labels"] == class_id, as_tuple=False).flatten().tolist()]
        montage = {
            "path": figure_path.name,
            "rows_displayed": 150,
            "classes": list(DISPLAY_CLASSES),
            "panel_order": ["target", "step2200", "A_original", "B_balanced"],
            "view": "28x28 canonical clamp[0,1] then adaptive average pool",
            "fixed_display_range": [0.0, 1.0],
            "per_image_rescaling": False,
            "F/O": {"F": "fit TRAIN row", "O": "observation TRAIN row"},
            "source_positions": positions,
            "source_row_indices": [int(inputs["source_indices"][position]) for position in positions],
        }
    comparison_path = target / "comparison_28x28.json"
    if comparison_path.exists():
        _require(json.loads(comparison_path.read_text(encoding="utf-8")) == montage, "saved comparison figure metadata differs")
    else:
        diagnostic._write_json_new(comparison_path, montage, "fixed-scale five-class endpoint comparison")
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "training_recommendation": decision["training_recommendation"],
        "reason": decision["reason"],
        "measurement_qualified": decision["measurement_qualified"],
        "decision_criteria": decision.get("criteria"),
        "scope": {
            "updates_per_arm": UPDATES_PER_ARM,
            "fit_rows": FIT_ROWS,
            "observation_rows": OBSERVATION_ROWS,
            "population_role": "all 300 rows are canonical TRAIN; observation rows are withheld from updates, not an independent generalization set",
            "only_updated_parameter": "generator.phase",
            "encoder_ols_geometry_references_evaluators_frozen": True,
            "protected_test_sealed_read": False,
            "dev_measurement": False,
            "s1_resampling": False,
            "follow_on_training_started": False,
        },
        "population": dict(population),
        "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "delta_SIC": DELTA_SIC, "kappa": KAPPA, "references_and_weights_inherited_from_step1200": True},
        "reconstruction_objectives": {
            "A_original": "mean squared error over raw 200x200 CMOS target pixels",
            "B_balanced": "per image: 0.5*mean squared error on target>0.1 pixels + 0.5*mean squared error on target<=0.1 pixels; then batch mean",
            "shared_training_terms": "alpha*L_dist + beta*L_rec + gamma*L_op plus kappa times the current jointly normalized three-evaluator gradient",
            "view_pooling_and_clamping_used_for_training_reconstruction": False,
        },
        "preflight": dict(preflight),
        "target_e_d_positive_control": {"fit": target_control["metrics"]["fit"], "observation": target_control["metrics"]["observation"], "combined": target_control["metrics"]["combined"], "thresholds_met": _positive_control_ok(target_control["metrics"])},
        "e_d_metrics": {name: {"fit": metric["fit"], "observation": metric["observation"], "combined": metric["combined"]} for name, metric in endpoint_metrics.items()},
        "image_statistics": image_statistics,
        "paired_statistics": paired_tests,
        "training_history_artifacts": {arm: f"training_history_{arm}.json" for arm in ARM_NAMES},
        "readout_artifacts": {name: f"readouts_{name}.pt" for name in readouts},
        "row_metric_artifacts": {name: f"row_metrics_{name}.pt" for name in readouts},
        "image_statistics_artifacts": {name: f"image_statistics_{name}.json" for name in readouts},
        "comparison_figure": montage,
        "checkpoint_artifacts": sorted(path.name for path in target.glob("arm_*_step*.pt")),
        "resource_telemetry_artifact": RESOURCE_TELEMETRY_NAME,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "recovered_engineering_failures": json.loads((target / "invalid_harness.json").read_text(encoding="utf-8")) if (target / "invalid_harness.json").is_file() else [],
        "stop": True,
    }
    diagnostic._write_json_new(target / "report.json", report, "final Generator-only reconstruction contrast decision")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint path required together with --resume")
    return parser.parse_args()


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    diagnostic_dir = _resolve(root, arguments.diagnostic_dir)
    target = _resolve(root, arguments.output_dir)
    _require(bool(arguments.resume) == (arguments.checkpoint is not None), "--resume and an explicit --checkpoint must be supplied together")
    _require(not (arguments.preflight_only and arguments.resume), "--preflight-only starts a new attempt and cannot be combined with --resume")
    _prepare_output(root, target, resume=bool(arguments.resume))
    started = time.perf_counter()
    try:
        paths, _source_population, _source_report = _diag_paths(root, diagnostic_dir)
        inputs = _load_inputs(root, paths)
        population = _population_record(inputs, paths, diagnostic_dir)
        population_path = target / "input_population.json"
        if arguments.resume:
            _require(population_path.is_file(), "resume requires the saved attempt input_population.json")
            _require(json.loads(population_path.read_text(encoding="utf-8")) == population, "resume diagnostic inputs, rows, labels, or data roles differ")
        else:
            diagnostic._write_json_new(population_path, population, "fixed 300-row TRAIN reconstruction population")
        preflight_path = target / "preflight.json"
        if arguments.resume and preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            _require(preflight.get("task_id") == TASK_ID and preflight.get("status") == "READY", "saved preflight is not READY for this task")
        else:
            try:
                preflight = _preflight(inputs, paths, target)
            except Exception as error:
                _write_invalid(target, "PREFLIGHT_INVALID_HARNESS", error, started)
                raise
            diagnostic._write_json_new(preflight_path, preflight, "no-update-gate preflight for two Generator-only arms")
        if arguments.preflight_only:
            consumer, optimizer, _parameters, _references = _new_endpoint(inputs, torch.device("cuda:0"))
            initial_history = {arm: [] for arm in ARM_NAMES}
            _save_checkpoint(target, consumer, optimizer, ARM_NAMES[0], 0, initial_history, population)
            del consumer, optimizer
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "preflight": preflight, "explicit_resume_checkpoint": (target / _checkpoint_name(ARM_NAMES[0], 0)).as_posix(), "next_action": "resume this attempt with --resume --checkpoint <explicit path> to run the fixed 1000-update arms"}

        selected_checkpoint = None
        resume_arm = None
        if arguments.resume:
            checkpoint_path = _resolve(root, arguments.checkpoint)
            selected_checkpoint = _read_checkpoint(checkpoint_path, target, population)
            resume_arm = str(selected_checkpoint["arm"])
            completed_result = _completed_resume_result(target, selected_checkpoint)
            if completed_result is not None:
                return completed_result
        else:
            _require(not any(target.glob("arm_*_step*.pt")), "new attempt output already contains an arm checkpoint")

        device = torch.device("cuda:0")
        evaluators = diagnostic.load_evaluators(paths, device)
        evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
        history: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
        if selected_checkpoint is not None:
            stored_history = selected_checkpoint.get("training_history")
            _require(isinstance(stored_history, Mapping), "resume checkpoint training history is malformed")
            history = {arm: [dict(row) for row in stored_history.get(arm, [])] for arm in ARM_NAMES}
        readout_path = target / "readouts_step2200.pt"
        if readout_path.is_file():
            start_readouts = torch.load(readout_path, map_location="cpu", weights_only=True)
            _validate_readout_population(start_readouts, inputs, "step2200")
            if (target / "image_statistics_step2200.json").is_file():
                start_stats = json.loads((target / "image_statistics_step2200.json").read_text(encoding="utf-8"))
                start_result = {"readouts": start_readouts, "image_statistics": start_stats}
            else:
                start_stats, _ = diagnostic._aggregate_image_statistics({**start_readouts, "input_images": inputs["images"]}, inputs["labels"], inputs["source_indices"])
                start_result = {"readouts": start_readouts, "image_statistics": start_stats}
                _save_measurements(target, "step2200", start_result, inputs)
        else:
            start_consumer, start_optimizer, _parameters, _references = _new_endpoint(inputs, device)
            start_readouts = diagnostic._forward_readouts(start_consumer, evaluators, inputs["images"], batch_size=BATCH_SIZE)
            start_readouts.update({"source_indices": inputs["source_indices"].clone(), "labels": inputs["labels"].clone()})
            start_stats, _ = diagnostic._aggregate_image_statistics({**start_readouts, "input_images": inputs["images"]}, inputs["labels"], inputs["source_indices"])
            start_result = {"readouts": start_readouts, "image_statistics": start_stats}
            _save_measurements(target, "step2200", start_result, inputs)
            del start_consumer, start_optimizer
        readouts_by_endpoint: dict[str, Mapping[str, torch.Tensor]] = {"step2200": start_result["readouts"]}
        arms: dict[str, dict[str, Any]] = {}
        first_arm = ARM_NAMES[0]
        second_arm = ARM_NAMES[1]
        if selected_checkpoint is None or resume_arm == first_arm:
            result_a, _ = _run_arm(first_arm, inputs, evaluators, target, population, history, start_from=selected_checkpoint, started=started)
            result_a["image_statistics"] = _save_measurements(target, first_arm, result_a, inputs)
            arms[first_arm] = result_a
            readouts_by_endpoint[first_arm] = result_a["readouts"]
        else:
            _require((target / "readouts_A_original.pt").is_file() and (target / "image_statistics_A_original.json").is_file(), "resume at B requires saved completed-A readouts and image statistics")
            readouts_by_endpoint[first_arm] = torch.load(target / "readouts_A_original.pt", map_location="cpu", weights_only=True)
            _validate_readout_population(readouts_by_endpoint[first_arm], inputs, first_arm)
            arms[first_arm] = {
                "readouts": readouts_by_endpoint[first_arm],
                "image_statistics": json.loads((target / "image_statistics_A_original.json").read_text(encoding="utf-8")),
                "completed_updates": UPDATES_PER_ARM,
            }
        result_b, _ = _run_arm(second_arm, inputs, evaluators, target, population, history, start_from=selected_checkpoint if resume_arm == second_arm else None, started=started)
        result_b["image_statistics"] = _save_measurements(target, second_arm, result_b, inputs)
        arms[second_arm] = result_b
        readouts_by_endpoint[second_arm] = result_b["readouts"]
        _require(set(readouts_by_endpoint) == {"step2200", *ARM_NAMES}, "one or more fresh endpoint readouts are missing")
        endpoint_metrics, target_control = _readout_report(inputs, readouts_by_endpoint, {}, evaluators, device)
        image_statistics = {name: json.loads((target / f"image_statistics_{name}.json").read_text(encoding="utf-8")) for name in readouts_by_endpoint}
        _save_training_history(target, history)
        for name, model in evaluators.items():
            _assert_state_equal(evaluator_before[name], _state_snapshot(model), f"final {name} evaluator")
            _require(all(parameter.grad is None for parameter in model.parameters()), f"final {name} evaluator retained a gradient")
        _sample_resource_telemetry(target, device, "measurement_complete", progress={"arm_a_updates": UPDATES_PER_ARM, "arm_b_updates": UPDATES_PER_ARM}, started=started)
        target_payload = {"logits": target_control["logits"], "predictions": target_control["metrics"]["predictions"]}
        target_torch_path = target / "target_e_d_positive_control.pt"
        if target_torch_path.exists():
            _require(_tree_equal(torch.load(target_torch_path, map_location="cpu", weights_only=True), target_payload), "saved target positive-control readout differs")
        else:
            diagnostic._write_torch_new(target_torch_path, target_payload, "fresh target image E_D positive control")
        target_json_path = target / "target_e_d_positive_control.json"
        target_metrics_payload = {key: target_control["metrics"][key] for key in ("fit", "observation", "combined")}
        if target_json_path.exists():
            _require(json.loads(target_json_path.read_text(encoding="utf-8")) == target_metrics_payload, "saved target positive-control metrics differ")
        else:
            diagnostic._write_json_new(target_json_path, target_metrics_payload, "target image E_D positive control metrics")
        report = _finalize(target, inputs, history, start_result, arms, population, preflight, target_control, endpoint_metrics, image_statistics, started)
        del evaluators
        gc.collect()
        torch.cuda.empty_cache()
        return report
    except Exception as error:
        _write_invalid(target, "INVALID_HARNESS", error, started)
        raise


def _write_invalid(target: Path, phase: str, error: Exception, started: float) -> None:
    if target.exists() and not (target / "invalid_harness.json").exists():
        payload = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "status": "INVALID_HARNESS",
            "classification": None,
            "training_recommendation": None,
            "phase": phase,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "elapsed_seconds_observation_only": time.perf_counter() - started,
        }
        diagnostic._write_json_new(target / "invalid_harness.json", payload, "invalid scientific harness record")


def main() -> int:
    result = execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
