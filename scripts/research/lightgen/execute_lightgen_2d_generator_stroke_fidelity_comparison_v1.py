"""Compare the fixed large Generator under raw-MSE and stroke-fidelity reconstruction."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import io
import json
import math
import random
import secrets
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as class9


TASK_ID = "M0_GENERATOR_STROKE_FIDELITY_COMPARISON_V1"
REPORT_SCHEMA = "m0_generator_stroke_fidelity_comparison_report.v1"
CHECKPOINT_SCHEMA = "m0_generator_stroke_fidelity_checkpoint.v1"
DEFAULT_GENERATION = Path("outputs/20261008_m0_generator_frozen_conditional_ols_generation_v1/runs/attempt_02_microbatch1")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_stroke_fidelity_comparison_v1/runs/attempt_01")
SOURCE_STEP = 4_700
ADDED_UPDATES = 700
FINAL_STEP = SOURCE_STEP + ADDED_UPDATES
FIT_COUNT = 200
CONFIRM_COUNT = 300
MEASURE_COUNT = FIT_COUNT + CONFIRM_COUNT
BATCH_SIZE = 10
BATCH_COUNT = FIT_COUNT // BATCH_SIZE
MICROBATCH_SIZE = 2
SSIM_WEIGHT = 0.1
SSIM_KERNEL = 11
SSIM_SIGMA = 1.5
SSIM_C1 = 1e-4
SSIM_C2 = 9e-4
REVIEW_SAMPLE_SEED = 2026100817
REVIEW_ORDER_SEED = 2026100818
REVIEW_ROWS_PER_CLASS = 20
REVIEW_TARGETS_PER_CLASS = 5
REVIEW_DUPLICATES_PER_ENDPOINT_CLASS = 1
REVIEW_PRIMARY_COUNT = 600
REVIEW_TARGET_COUNT = 50
REVIEW_REPEAT_COUNT = 30
REVIEW_TOTAL = 680
ANCHOR_MIN = 45
REPEAT_MIN = 27
HUMAN_TOTAL_MIN = 120
HUMAN_DELTA_MIN = 30
CLASS_DROP_MAX = 2
RTOL, ATOL = 2.0e-5, 2.0e-7
DISK_RESERVE = 1_610_612_736
ARM_A = "A_original"
ARM_B = "B_stroke_fidelity"
ARM_NAMES = (ARM_A, ARM_B)
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
TRAIN_EVALUATORS = ("E_S", "E_I", "E_C")
NUMERIC_MODE = dict(class9.NUMERIC_MODE)


class StrokeFidelityError(RuntimeError):
    """Invalid task input, failed production invariant, or incomplete evidence."""


class TrainingInterrupted(RuntimeError):
    """A stop request was received after a durable checkpoint boundary."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise StrokeFidelityError(message)


def _assert_tensor_unchanged(before: torch.Tensor, after: torch.Tensor, name: str) -> None:
    _require(before.shape == after.shape and before.dtype == after.dtype, f"{name} tensor shape/dtype changed")
    _require(torch.equal(before, after), f"{name} tensor changed")


def _trial_repeatability(first: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    def tensor_check(left: torch.Tensor, right: torch.Tensor, *, rtol: float, atol: float, path: str) -> dict[str, Any]:
        metadata_matches = left.shape == right.shape and left.dtype == right.dtype
        difference = float((left.to(torch.float64) - right.to(torch.float64)).abs().max()) if metadata_matches and left.numel() else 0.0
        passed = metadata_matches and torch.allclose(left, right, rtol=rtol, atol=atol)
        return {"passed": bool(passed), "path": path, "max_absolute_difference": difference, "rtol": rtol, "atol": atol}

    def history_close(left: Any, right: Any, path: str = "training_history") -> tuple[bool, list[dict[str, Any]]]:
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not (isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right)):
                return False, [{"path": path, "reason": "mapping keys differ"}]
            details = []
            passed = True
            for key in left:
                child_passed, child_details = history_close(left[key], right[key], f"{path}.{key}")
                passed = passed and child_passed
                details.extend(child_details)
            return passed, details
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not (isinstance(left, type(right)) and len(left) == len(right)):
                return False, [{"path": path, "reason": "sequence shape differs"}]
            details = []
            passed = True
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                child_passed, child_details = history_close(a, b, f"{path}[{index}]")
                passed = passed and child_passed
                details.extend(child_details)
            return passed, details
        if isinstance(left, float) or isinstance(right, float):
            passed = isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=5e-8)
            return passed, [] if passed else [{"path": path, "left": left, "right": right, "rtol": 2e-6, "atol": 5e-8}]
        passed = left == right
        return passed, [] if passed else [{"path": path, "left": left, "right": right}]

    phase = tensor_check(first["phase"], replay["phase"], rtol=2e-6, atol=1e-6, path="generator.phase")
    first_optimizer = first["optimizer"]
    replay_optimizer = replay["optimizer"]
    _require(first_optimizer.keys() == replay_optimizer.keys(), "A/A optimizer state structure differs")
    _require(_tree_equal(first_optimizer["param_groups"], replay_optimizer["param_groups"]), "A/A optimizer hyperparameters differ")
    first_state = first_optimizer["state"][0]
    replay_state = replay_optimizer["state"][0]
    _require(first_state["step"].item() == replay_state["step"].item(), "A/A optimizer step differs")
    moment = tensor_check(first_state["exp_avg"], replay_state["exp_avg"], rtol=2e-6, atol=1e-8, path="optimizer.exp_avg")
    variance = tensor_check(first_state["exp_avg_sq"], replay_state["exp_avg_sq"], rtol=2e-6, atol=1e-14, path="optimizer.exp_avg_sq")
    history_passed, history_details = history_close(first["history"], replay["history"])
    comparisons = {
        "generator_phase": phase,
        "optimizer_exp_avg": moment,
        "optimizer_exp_avg_sq": variance,
        "training_history": {"passed": history_passed, "differences": history_details, "float_rtol": 2e-6, "float_atol": 5e-8},
    }
    for name in ("generator_phase", "optimizer_exp_avg", "optimizer_exp_avg_sq"):
        comparisons[name]["tolerance_and_difference"] = {key: value for key, value in comparisons[name].items() if key != "passed"}
    comparisons["optimizer_exp_avg"].pop("path")
    comparisons["optimizer_exp_avg_sq"].pop("path")
    comparisons["generator_phase"].pop("path")
    return {"passed": all(item["passed"] for item in comparisons.values()), "comparisons": comparisons}


def _aa_preflight_record(first: Mapping[str, Any], replay: Mapping[str, Any], *, updates: int) -> dict[str, Any]:
    repeatability = _trial_repeatability(first, replay)
    return {
        "executed": True,
        "passed": repeatability["passed"],
        "updates": updates,
        "A_A_repeatability": repeatability,
        "generator_adam_step": SOURCE_STEP + updates,
    }


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    data = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def _gaussian_window(*, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    coordinate = torch.arange(SSIM_KERNEL, dtype=torch.float64, device=device) - (SSIM_KERNEL - 1) / 2
    one_d = torch.exp(-(coordinate.square()) / (2.0 * SSIM_SIGMA**2))
    one_d = one_d / one_d.sum()
    window = torch.outer(one_d, one_d).to(dtype=dtype)
    return window.view(1, 1, SSIM_KERNEL, SSIM_KERNEL)


def ssim_per_image(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return single-scale valid-window SSIM per image using weighted population moments."""
    _require(generated.ndim == 4 and generated.shape == target.shape, "SSIM inputs must have identical NCHW shape")
    _require(generated.shape[1] == 1 and tuple(generated.shape[-2:]) == (28, 28), "SSIM view must be single-channel 28x28")
    _require(generated.is_floating_point() and target.is_floating_point(), "SSIM inputs must be floating tensors")
    x, y = generated.to(torch.float64), target.to(torch.float64)
    kernel = _gaussian_window(device=x.device)
    mu_x = F.conv2d(x, kernel, padding=0)
    mu_y = F.conv2d(y, kernel, padding=0)
    ex2 = F.conv2d(x.square(), kernel, padding=0)
    ey2 = F.conv2d(y.square(), kernel, padding=0)
    exy = F.conv2d(x * y, kernel, padding=0)
    var_x = ex2 - mu_x.square()
    var_y = ey2 - mu_y.square()
    covariance = exy - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + SSIM_C1) * (2.0 * covariance + SSIM_C2)
    denominator = (mu_x.square() + mu_y.square() + SSIM_C1) * (var_x + var_y + SSIM_C2)
    return (numerator / denominator).flatten(1).mean(dim=1)


def reconstruction_terms(raw: torch.Tensor, target_raw: torch.Tensor, *, structural: bool) -> dict[str, torch.Tensor]:
    """The exact raw MSE plus the declared pooled structural term."""
    _require(raw.shape == target_raw.shape and raw.ndim == 4, "raw reconstruction tensors differ")
    mse = (raw - target_raw).square().flatten(1).mean(dim=1).mean()
    view = diagnostic.canonical.cmos_view(raw)
    target_view = diagnostic.canonical.cmos_view(target_raw)
    structural_loss = (1.0 - ssim_per_image(view, target_view)).mean()
    objective = mse + (SSIM_WEIGHT * structural_loss if structural else 0.0)
    return {"raw_mse": mse, "ssim_loss": structural_loss, "objective": objective}


def _load_population(root: Path, generation_dir: Path) -> dict[str, Any]:
    report_path = generation_dir / "report.json"
    population_path = generation_dir / "corrected_input_population.json"
    _require(report_path.is_file() and population_path.is_file(), "explicit frozen-generation source report/population is missing")
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    source_population = json.loads(population_path.read_text(encoding="utf-8"))
    _require(parent_report.get("task_id") == generation.TASK_ID, "source report is not the named frozen conditional OLS run")
    _require(parent_report.get("status") == "VALID_COMPLETED" and parent_report.get("classification") == "CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED", "source frozen-generation result differs")
    _require(source_population.get("task_id") == "M0_GENERATOR_TRAINING_POPULATION_EXPANSION_COMPARISON_V1", "frozen-generation population lineage differs")
    source_rows = torch.tensor(source_population.get("source_row_indices", []), dtype=torch.long)
    source_labels = torch.tensor(source_population.get("labels", []), dtype=torch.long)
    _require(source_rows.shape == source_labels.shape == (3_300,), "fixed source population is not 3,300 rows")
    _require(torch.unique(source_rows).numel() == 3_300, "source TRAIN population contains repeated rows")
    role_positions = source_population.get("role_positions", {})
    _require(role_positions.get("fit") == [0, 200] and role_positions.get("coverage_confirmation") == [3_000, 3_300], "source fit/confirmation row ranges differ")
    _require(torch.bincount(source_labels[:FIT_COUNT], minlength=10).tolist() == [20] * 10, "source fit rows are not 20 per class")
    _require(torch.bincount(source_labels[3_000:3_300], minlength=10).tolist() == [30] * 10, "source confirmation rows are not 30 per class")

    source_checkpoint_path = _resolve(root, source_population["source_checkpoint"])
    _require(source_checkpoint_path.name == "arm_a_step4700.pt" and source_checkpoint_path.is_file(), "explicit A4700 source checkpoint is missing")
    source = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    _require(source.get("task_id") == "M0_GENERATOR_TRAINING_POPULATION_EXPANSION_COMPARISON_V1" and source.get("arm") == "A_repeat_200", "source checkpoint task/arm differs")
    _require(source.get("completed_updates") == SOURCE_STEP and source.get("optimizer_step") == SOURCE_STEP, "source model/Adam is not at step4700")
    _require(source.get("next_batch_index") == 0 and source.get("next_batch_cursor") == 0, "source endpoint is not at the zero batch cursor")
    _require(source.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "source large Generator layout differs")
    params = source.get("trainable_parameter_state", {})
    _require(set(params) == {"encoder.phase", "generator.phase"}, "A4700 source phase state is incomplete")
    _require(tuple(params["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(params["generator.phase"].shape) == (340_000,), "A4700 phase shapes differ")
    _require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in params.values()), "A4700 phase values are nonfinite or wrong dtype")
    optimizer_state = source.get("optimizer_state", {})
    groups, states = optimizer_state.get("param_groups", []), optimizer_state.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "A4700 Generator Adam mapping differs")
    _require(int(float(states[0]["step"].item())) == SOURCE_STEP, "A4700 Generator Adam step differs")
    _require(tuple(states[0]["exp_avg"].shape) == (340_000,) and tuple(states[0]["exp_avg_sq"].shape) == (340_000,), "A4700 Adam moment shapes differ")
    _require(bool(torch.isfinite(states[0]["exp_avg"]).all() and torch.isfinite(states[0]["exp_avg_sq"]).all()), "A4700 Adam moments are nonfinite")
    _require(source.get("numeric_mode") == NUMERIC_MODE, "A4700 source strict FP32 mode differs")
    _require(len(source.get("training_history", [])) == SOURCE_STEP and source.get("source_history_count") == 3_700 and source.get("task_local_updates") == 1_000, "A4700 checkpoint source/task history counts differ")

    raw_paths = source_population.get("source_input_paths", {})
    _require(set(raw_paths) >= {"training_data", "step1200", "step2200", "e_s", "e_i", "e_c", "e_d"}, "source population omits explicit runtime/evaluator/data paths")
    paths = {str(name): _resolve(root, value) for name, value in raw_paths.items()}
    _require(all(paths[name].is_file() for name in ("training_data", "step1200", "step2200", "e_s", "e_i", "e_c", "e_d")), "an explicit model/data/evaluator input is missing")
    base1200 = torch.load(paths["step1200"], map_location="cpu", weights_only=True)
    base2200 = torch.load(paths["step2200"], map_location="cpu", weights_only=True)
    _require(_tree_equal(base1200.get("references"), source.get("references")), "A4700 shared references do not match step1200")
    _require(_tree_equal(base1200.get("weights"), source.get("weights")), "A4700 shared BOGT weights do not match step1200")
    _require(base2200.get("task_id") == "M0_BOGT_SMALL_BANK_JOINT_FIT_CANARY_V1", "explicit step2200 model source differs")
    frozen = source.get("frozen_runtime_state", {})
    _require(set(frozen) == {"ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "A4700 frozen OLS state keys differ")
    _require(all(bool(torch.isfinite(value.real if value.is_complex() else value).all()) for value in frozen.values()), "A4700 frozen OLS state contains nonfinite values")

    source_readout_path = source_checkpoint_path.parent / "readouts_A4700.pt"
    _require(source_readout_path.is_file(), "saved A4700 source readout is missing")
    full_readout = torch.load(source_readout_path, map_location="cpu", weights_only=True, mmap=True)
    _require(torch.equal(full_readout["source_indices"].to(torch.long), source_rows) and torch.equal(full_readout["labels"].to(torch.long), source_labels), "A4700 source readout row/label mapping differs")
    positions = torch.cat((torch.arange(FIT_COUNT), torch.arange(3_000, 3_300)))
    rows = source_rows.index_select(0, positions)
    labels = source_labels.index_select(0, positions)
    _require(torch.unique(rows).numel() == MEASURE_COUNT, "fit and confirmation measurement rows overlap")

    store = diagnostic._load_torch(paths["training_data"], mmap=True)
    image_rows, canonical_labels = diagnostic._read_image_labels(store)
    _require(torch.equal(canonical_labels.index_select(0, rows), labels), "source labels differ from canonical training metadata")
    train_indices, _ = diagnostic.stratified_mnist_indices(canonical_labels, seed=diagnostic.SPLIT_SEED, validation_per_label=500)
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    _require(bool(train_mask.index_select(0, rows).all()), "fit or confirmation row is outside canonical TRAIN")
    images = image_rows.index_select(0, rows).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(images.shape == (MEASURE_COUNT, 1, 28, 28) and bool(torch.isfinite(images).all()) and 0 <= float(images.min()) <= float(images.max()) <= 1, "selected 500 target images are invalid")

    readout = {key: value.index_select(0, positions).clone() if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 3_300 else value for key, value in full_readout.items()}
    _require(torch.equal(readout["source_indices"], rows) and torch.equal(readout["labels"], labels), "selected A4700 readout mapping differs")
    _require(all(bool(torch.isfinite(value.real if value.is_complex() else value).all()) for value in readout.values() if torch.is_tensor(value)), "selected A4700 readout contains nonfinite values")
    evaluator_paths = {name.lower(): paths[name.lower()] for name in EVALUATORS}
    population = {
        "task_id": TASK_ID,
        "source_generation_dir": generation_dir.as_posix(),
        "source_checkpoint": source_checkpoint_path.as_posix(),
        "source_a4700_readout": source_readout_path.as_posix(),
        "source_checkpoint_step": SOURCE_STEP,
        "source_generator_adam_step": SOURCE_STEP,
        "source_parameter_shapes": {key: list(value.shape) for key, value in params.items()},
        "source_input_paths": {key: value.as_posix() for key, value in paths.items()},
        "source_generation_classification": parent_report["classification"],
        "source_population_positions": positions.tolist(),
        "source_row_indices": rows.tolist(),
        "labels": labels.tolist(),
        "roles": ["fit"] * FIT_COUNT + ["confirmation"] * CONFIRM_COUNT,
        "role_positions": {"fit": [0, FIT_COUNT], "confirmation": [FIT_COUNT, MEASURE_COUNT]},
        "fit_rows": FIT_COUNT,
        "confirmation_rows": CONFIRM_COUNT,
        "updates_per_arm": ADDED_UPDATES,
        "final_generator_adam_step": FINAL_STEP,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "microbatch_size": MICROBATCH_SIZE,
        "ssim": {"weight": SSIM_WEIGHT, "kernel": SSIM_KERNEL, "sigma": SSIM_SIGMA, "C1": SSIM_C1, "C2": SSIM_C2, "stat_dtype": "float64", "window": "normalized 11x11 Gaussian, valid convolution, 18x18 mean"},
        "human_review": {"sample_seed": REVIEW_SAMPLE_SEED, "order_seed": REVIEW_ORDER_SEED, "primary_per_class_per_endpoint": 20, "target_per_class": 5, "hidden_repeats": 30, "total_items": REVIEW_TOTAL},
        "numeric_mode": source.get("numeric_mode"),
        "canonical_train_not_claimed_unseen": True,
    }
    return {"population": population, "source": source, "base2200": base2200, "images": images, "rows": rows, "labels": labels, "fit_images": images[:FIT_COUNT], "fit_labels": labels[:FIT_COUNT], "readout": readout, "paths": paths, "evaluator_paths": evaluator_paths, "source_readout_path": source_readout_path}


def _tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _new_endpoint(inputs: Mapping[str, Any], device: torch.device):
    source = inputs["source"]
    consumer, _source_optimizer, _params, references, weights = diagnostic._new_endpoint(
        "step2200", inputs["base2200"], source["references"], source["weights"], device
    )
    old_phase = torch.zeros((2, 1, 100, 100), dtype=torch.float32, device=device)
    consumer.runtime.generator = capacity.ExpandedGenerator(old_phase, propagation_grid_size=2_048).to(device)
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        parameters["encoder.phase"].copy_(source["trainable_parameter_state"]["encoder.phase"].to(device))
        parameters["generator.phase"].copy_(source["trainable_parameter_state"]["generator.phase"].to(device))
    consumer.runtime.eval()
    optimizer = capacity._new_generator_optimizer(parameters["generator.phase"])
    optimizer.load_state_dict(copy.deepcopy(source["optimizer_state"]))
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP, "loaded A4700 Generator Adam state differs")
    capacity._assert_state_equal(source["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "A4700 OLS/runtime state")
    _require(_tree_equal(references.to_payload(), diagnostic.bogt.FrozenBogtReferences.from_payload(source["references"]).to_payload()), "A4700 references did not bind")
    _require(_tree_equal(weights.to_payload(), diagnostic.bogt.BogtLossWeights.from_payload(source["weights"]).to_payload()), "A4700 weights did not bind")
    return consumer, optimizer, parameters


def _rng_clone(state: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(state))


def _readout(consumer: Any, evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    payload = capacity._forward_all(consumer, evaluators, {"images": inputs["images"], "source_indices": inputs["rows"], "labels": inputs["labels"]}, "B_large")
    for key, expected in (("source_indices", inputs["rows"]), ("labels", inputs["labels"])):
        _require(torch.equal(payload[key], expected), f"measurement {key} order differs")
    return payload


def _replay_check(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("z", "raw_cmos", "evaluator_view", "target_slm_intensity", "power_by_plane", "new_phase_plane_power", "e_s_logits", "e_i_logits", "e_c_logits", "e_d_logits")
    details = {}
    for key in fields:
        left, right = actual[key], expected[key]
        _require(left.shape == right.shape and torch.allclose(left, right, rtol=RTOL, atol=ATOL), f"A4700 replay differs for {key}")
        error = (left.to(torch.complex128 if left.is_complex() else torch.float64) - right.to(torch.complex128 if right.is_complex() else torch.float64)).abs()
        details[key] = {"passed": True, "max_abs_difference": float(error.max())}
    return {"passed": True, "rtol": RTOL, "atol": ATOL, "fields": details}


def _parameters_snapshot(consumer: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}


def _gradient(loss: torch.Tensor, parameters: Mapping[str, nn.Parameter], *, retain_graph: bool) -> dict[str, torch.Tensor]:
    names = ("encoder.phase", "generator.phase")
    grads = torch.autograd.grad(loss, tuple(parameters[name] for name in names), retain_graph=retain_graph, allow_unused=True)
    return {name: torch.zeros_like(parameters[name]) if grad is None else grad.detach() for name, grad in zip(names, grads, strict=True)}


def _logical_batch_gradients(consumer: Any, parameters: Mapping[str, nn.Parameter], evaluators: Mapping[str, nn.Module], images: torch.Tensor, labels: torch.Tensor, *, structural: bool, reconstruction_targets: torch.Tensor | None = None, verify_ssim_vjp: bool = False):
    _require(images.shape == (BATCH_SIZE, 1, 28, 28) and labels.shape == (BATCH_SIZE,), "training logical batch must be exactly ten rows")
    _require(torch.bincount(labels.detach().cpu(), minlength=10).tolist() == [1] * 10, "training batch must be class-balanced in original order")
    if reconstruction_targets is not None:
        _require(reconstruction_targets.shape == (BATCH_SIZE, 1, 200, 200), "fixed reconstruction targets must match the complete logical batch")
        _require(reconstruction_targets.dtype == torch.float32 and not reconstruction_targets.requires_grad, "fixed reconstruction targets must be detached float32 tensors")
        _require(bool(torch.isfinite(reconstruction_targets).all()), "fixed reconstruction targets contain nonfinite values")
    refs = consumer.references
    _require(refs is not None, "frozen BOGT references are missing")
    z_parts = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
            stop = start + MICROBATCH_SIZE
            result = consumer.runtime.forward_from_image(images[start:stop])
            z_parts.append(result.z_ols.coefficients.detach())
            del result
    coefficients = torch.cat(z_parts, dim=0)
    l_dist, d_dist = capacity._dist_loss_and_ols_gradient(coefficients, refs.scale)
    names = ("encoder.phase", "generator.phase")
    g_dist = {name: torch.zeros_like(parameters[name]) for name in names}
    g_bogt = {name: torch.zeros_like(parameters[name]) for name in names}
    g_semantic = {evaluator: {name: torch.zeros_like(parameters[name]) for name in names} for evaluator in TRAIN_EVALUATORS}
    g_ssim = {name: torch.zeros_like(parameters[name]) for name in names}
    totals = {"raw_mse": 0.0, "ssim_loss": 0.0, "reconstruction_objective": 0.0, "l_dist": float(l_dist.detach().to(torch.float64).cpu()), "l_op": 0.0, **{f"{name}_ce": 0.0 for name in TRAIN_EVALUATORS}}
    powers: list[torch.Tensor] = []
    for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
        stop = start + MICROBATCH_SIZE
        fraction = (stop - start) / BATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:stop])
        raw, target_raw, z = result.raw_intensity, result.image_input.slm_intensity, result.z_ols.coefficients
        training_target_raw = target_raw if reconstruction_targets is None else reconstruction_targets[start:stop].to(device=raw.device, dtype=raw.dtype)
        proxy = torch.real(torch.conj(d_dist[start:stop].to(z)) * z).sum()
        terms = reconstruction_terms(raw, training_target_raw, structural=structural)
        planes = diagnostic.bogt.contracted_plane_powers(result)
        l_op = diagnostic.bogt.optical_power_loss_from_powers(planes, refs.power_reference)
        view = diagnostic.canonical.cmos_view(raw)
        ce_losses = {name: F.cross_entropy(capacity._evaluator_logits(evaluators[name], view, name), labels[start:stop], reduction="mean") for name in TRAIN_EVALUATORS}
        g_dist_piece = _gradient(proxy, parameters, retain_graph=True)
        g_ssim_piece = _gradient(terms["ssim_loss"], parameters, retain_graph=True) if structural else {name: torch.zeros_like(parameters[name]) for name in names}
        bogt_piece = capacity.semantic.BETA * terms["objective"] + capacity.semantic.GAMMA * l_op
        g_bogt_piece = _gradient(bogt_piece, parameters, retain_graph=True)
        ce_piece = {}
        for index, name in enumerate(TRAIN_EVALUATORS):
            keep_for_vjp = verify_ssim_vjp and structural and start == 0
            ce_piece[name] = _gradient(ce_losses[name], parameters, retain_graph=index < len(TRAIN_EVALUATORS) - 1 or keep_for_vjp)
        for name in names:
            g_dist[name] += g_dist_piece[name]
            g_ssim[name] += fraction * g_ssim_piece[name]
            g_bogt[name] += fraction * g_bogt_piece[name]
            for evaluator in TRAIN_EVALUATORS:
                g_semantic[evaluator][name] += fraction * ce_piece[evaluator][name]
        totals["raw_mse"] += fraction * float(terms["raw_mse"].detach().to(torch.float64).cpu())
        totals["ssim_loss"] += fraction * float(terms["ssim_loss"].detach().to(torch.float64).cpu())
        totals["reconstruction_objective"] += fraction * float(terms["objective"].detach().to(torch.float64).cpu())
        totals["l_op"] += fraction * float(l_op.detach().to(torch.float64).cpu())
        for name, value in ce_losses.items():
            totals[f"{name}_ce"] += fraction * float(value.detach().to(torch.float64).cpu())
        powers.append(planes.detach())
        if verify_ssim_vjp and structural and start == 0:
            direct = torch.autograd.grad(terms["ssim_loss"], parameters["generator.phase"], retain_graph=True)[0]
            pixel_gradient = torch.autograd.grad(terms["ssim_loss"], raw, retain_graph=True)[0]
            chained = torch.autograd.grad(raw, parameters["generator.phase"], grad_outputs=pixel_gradient, retain_graph=True)[0]
            error = float((direct.to(torch.float64) - chained.to(torch.float64)).abs().max().cpu())
            reference = float(direct.to(torch.float64).abs().max().cpu())
            _require(error <= 2e-5 * reference + 1e-10, f"SSIM image-space VJP differs from its direct Generator gradient: {error}")
        del result, raw, target_raw, training_target_raw, z, proxy, terms, planes, l_op, view, ce_losses
        del g_dist_piece, g_ssim_piece, bogt_piece, g_bogt_piece, ce_piece
    semantic_gradient, unavailable = diagnostic._safe_canonical_gradient({key: g_semantic[name] for name, key in zip(TRAIN_EVALUATORS, ("L_S", "L_I", "L_C"), strict=True)})
    _require(semantic_gradient is not None, f"three-evaluator semantic direction is unavailable: {unavailable}")
    _require(capacity._vector_norm(g_dist["generator.phase"]) == 0.0, "L_dist produced a Generator gradient despite structural zero")
    combined = {name: capacity.semantic.ALPHA * g_dist[name] + g_bogt[name] + capacity.semantic.KAPPA * semantic_gradient[name] for name in names}
    _require(all(bool(torch.isfinite(value).all()) for value in combined.values()), "combined training direction is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only production gradients populated parameter .grad")
    return torch.cat(powers, dim=0), totals, {"L_dist": g_dist, "ssim": g_ssim}, combined


def _train_step(consumer: Any, optimizer: torch.optim.Optimizer, parameters: Mapping[str, nn.Parameter], evaluators: Mapping[str, nn.Module], fit_inputs: Mapping[str, torch.Tensor], arm: str, local_update: int, *, vjp_check: bool = False) -> dict[str, Any]:
    batch_index = local_update % BATCH_COUNT
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    images = fit_inputs["images"][start:stop].to(consumer.device)
    labels_cpu = fit_inputs["labels"][start:stop].cpu()
    labels = labels_cpu.to(consumer.device)
    structural = arm == ARM_B
    powers, losses, gradients, combined = _logical_batch_gradients(consumer, parameters, evaluators, images, labels, structural=structural, verify_ssim_vjp=vjp_check)
    phase = parameters["generator.phase"]
    generator_gradient = combined["generator.phase"]
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient entered the optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(float(optimizer.state[phase]["step"].item()))
    _require(step == SOURCE_STEP + local_update + 1, "resumed Generator Adam step differs from the source plus task budget")
    delta = phase.detach() - before
    old_count = 20_000
    old_g, new_g = generator_gradient[:old_count], generator_gradient[old_count:]
    old_delta, new_delta = delta[:old_count], delta[old_count:]
    record = {
        "task_local_update": local_update + 1,
        "optimizer_step": step,
        "arm": arm,
        "batch_index": batch_index,
        "fit_positions": list(range(start, stop)),
        "source_row_indices": fit_inputs["source_indices"][start:stop].tolist(),
        "labels": labels_cpu.tolist(),
        "semantic_objective": "joint_normalized_equal_weight_E_S_E_I_E_C_true_class_CE",
        "raw_mse": losses["raw_mse"],
        "ssim_loss_28x28": losses["ssim_loss"],
        "ssim_weight": SSIM_WEIGHT if structural else 0.0,
        "reconstruction_objective": losses["reconstruction_objective"],
        "ce_by_evaluator": {name: losses[f"{name}_ce"] for name in TRAIN_EVALUATORS},
        "l_dist": losses["l_dist"],
        "l_op": losses["l_op"],
        "phase_gradient_rms": float(generator_gradient.detach().to(torch.float64).square().mean().sqrt().cpu()),
        "original_100x100_gradient_norm": capacity._vector_norm(old_g),
        "added_400x400_gradient_norm": capacity._vector_norm(new_g),
        "ssim_gradient_norm": capacity._vector_norm(gradients["ssim"]["generator.phase"]),
        "original_100x100_displacement_rms": float(old_delta.detach().to(torch.float64).square().mean().sqrt().cpu()),
        "added_400x400_displacement_rms": float(new_delta.detach().to(torch.float64).square().mean().sqrt().cpu()),
        "generator_displacement_rms": float(delta.detach().to(torch.float64).square().mean().sqrt().cpu()),
        "five_plane_power_mean": powers.detach().to(torch.float64).mean(0).cpu().tolist(),
        "all_finite": bool(torch.isfinite(phase).all().item()),
    }
    _require(record["all_finite"] and all(parameter.grad is None for parameter in parameters.values()), "training left nonfinite model or populated parameter gradients")
    return record


def _fit_inputs(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {"images": inputs["fit_images"], "labels": inputs["fit_labels"], "source_indices": inputs["rows"][:FIT_COUNT]}


def _target_positive_control(evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    images = inputs["images"][FIT_COUNT:].to(device)
    labels = inputs["labels"][FIT_COUNT:].cpu()
    result: dict[str, Any] = {"executed": True, "evaluators": {}}
    with torch.no_grad():
        view = diagnostic.canonical.cmos_view(images)
        for name in EVALUATORS:
            chunks = []
            for start in range(0, CONFIRM_COUNT, 50):
                chunks.append(capacity._evaluator_logits(evaluators[name], view[start : start + 50], name).detach().cpu())
            logits = torch.cat(chunks).to(torch.float64)
            prediction = logits.argmax(1)
            correct = [int(((labels == c) & (prediction == c)).sum()) for c in range(10)]
            result["evaluators"][name] = {
                "logits": torch.cat(chunks),
                "per_class_correct": correct,
                "macro_recall": sum(correct) / CONFIRM_COUNT,
                "passed": sum(correct) / CONFIRM_COUNT >= 0.90 and min(correct) >= 24,
            }
    result["passed"] = all(item["passed"] for item in result["evaluators"].values())
    return result


def _trial(arm: str, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], *, updates: int, vjp: bool = False) -> dict[str, Any]:
    source = inputs["source"]
    consumer, optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    fit_inputs = _fit_inputs(inputs)
    before_encoder = parameters["encoder.phase"].detach().cpu().clone()
    before_frozen = capacity._frozen_runtime_state(consumer)
    before_evaluators = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    before_source_adam = copy.deepcopy(source["optimizer_state"])
    capacity._restore_rng_state(source["rng_state"])
    history = []
    for index in range(updates):
        history.append(_train_step(consumer, optimizer, parameters, evaluators, fit_inputs, arm, index, vjp_check=vjp and index == 0))
    _assert_tensor_unchanged(before_encoder, parameters["encoder.phase"].detach().cpu(), f"{arm} disposable Encoder")
    capacity._assert_state_equal(before_frozen, capacity._frozen_runtime_state(consumer), f"{arm} disposable frozen runtime")
    for name, model in evaluators.items():
        capacity._assert_state_equal(before_evaluators[name], capacity._state_snapshot(model), f"{arm} disposable evaluator {name}")
    _require(all(parameter.grad is None for parameter in parameters.values()), f"{arm} disposable updates left parameter gradients")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP + updates, f"{arm} disposable Adam did not execute its real steps")
    _require(_tree_equal(before_source_adam, source["optimizer_state"]), "disposable Generator updates aliased the source Adam state")
    return {
        "phase": parameters["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "history": history,
        "new_plane_gradient_norm": history[-1]["added_400x400_gradient_norm"],
        "frozen_encoder_runtime_evaluators_and_source_adam": True,
    }


def _same_class_target_control(inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module]) -> dict[str, Any]:
    """Verify the structural term responds to the paired image, not only its class label."""
    consumer, _optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    labels = inputs["fit_labels"]
    first_positions = []
    alternative_positions = []
    for class_id in range(10):
        candidates = torch.nonzero(labels == class_id, as_tuple=False).flatten().tolist()
        _require(len(candidates) >= 2, f"class {class_id} lacks a same-class target negative control")
        first_positions.append(candidates[0])
        alternative_positions.append(candidates[1])
    images = inputs["fit_images"].index_select(0, torch.tensor(first_positions)).to(consumer.device)
    alternate_images = inputs["fit_images"].index_select(0, torch.tensor(alternative_positions)).to(consumer.device)
    with torch.no_grad():
        alternate_result = consumer.runtime.forward_from_image(alternate_images)
        alternate_target = alternate_result.image_input.slm_intensity.detach()
    result = consumer.runtime.forward_from_image(images)
    raw = result.raw_intensity
    paired_target = result.image_input.slm_intensity
    paired_view = diagnostic.canonical.cmos_view(paired_target)
    alternate_view = diagnostic.canonical.cmos_view(alternate_target)
    raw_view = diagnostic.canonical.cmos_view(raw)
    paired_loss = (1.0 - ssim_per_image(raw_view, paired_view)).mean()
    alternate_loss = (1.0 - ssim_per_image(raw_view, alternate_view)).mean()
    paired_grad = torch.autograd.grad(paired_loss, parameters["generator.phase"], retain_graph=True)[0]
    alternate_grad = torch.autograd.grad(alternate_loss, parameters["generator.phase"])[0]
    difference = float((paired_grad.to(torch.float64) - alternate_grad.to(torch.float64)).norm().cpu())
    _require(difference > 1e-10, "same-class target mismatch did not change the structural gradient")
    _require(all(parameter.grad is None for parameter in parameters.values()), "same-class negative control populated parameter .grad")
    return {"executed": True, "passed": True, "same_class_targets_swapped": True, "gradient_l2_difference": difference}


def _wrong_label_control(inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module]) -> dict[str, Any]:
    consumer, _optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    images = inputs["fit_images"][:MICROBATCH_SIZE].to(consumer.device)
    labels = inputs["fit_labels"][:MICROBATCH_SIZE].to(consumer.device)
    result = consumer.runtime.forward_from_image(images)
    view = diagnostic.canonical.cmos_view(result.raw_intensity)
    logits = capacity._evaluator_logits(evaluators["E_I"], view, "E_I")
    right = F.cross_entropy(logits, labels)
    wrong_labels = labels.flip(0)
    _require(not torch.equal(labels, wrong_labels), "negative-control rows do not have distinct labels")
    wrong = F.cross_entropy(logits, wrong_labels)
    g_right = torch.autograd.grad(right, parameters["generator.phase"], retain_graph=True)[0]
    g_wrong = torch.autograd.grad(wrong, parameters["generator.phase"])[0]
    difference = float((g_right.to(torch.float64) - g_wrong.to(torch.float64)).norm().cpu())
    _require(difference > 1e-10, "wrong-label control did not alter the production CE gradient")
    _require(all(parameter.grad is None for parameter in parameters.values()), "wrong-label control populated parameter .grad")
    return {"executed": True, "passed": True, "wrong_label_changes_production_gradient": True, "gradient_l2_difference": difference}


def _verify_a_original_reference(inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module]) -> dict[str, Any]:
    """Compare A's producer against the established large-Generator update path."""
    consumer, _optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    images = inputs["fit_images"][:BATCH_SIZE].to(consumer.device)
    labels = inputs["fit_labels"][:BATCH_SIZE].to(consumer.device)
    _ref_planes, ref_losses, _ref_gradients, ref_combined = capacity._large_microbatch_loss_gradients(
        consumer, parameters, evaluators, images, labels
    )
    _planes, losses, _gradients, combined = _logical_batch_gradients(
        consumer, parameters, evaluators, images, labels, structural=False
    )
    gradient_errors: dict[str, float] = {}
    for name in ("encoder.phase", "generator.phase"):
        error = float((combined[name].to(torch.float64) - ref_combined[name].to(torch.float64)).abs().max().cpu())
        _require(torch.allclose(combined[name], ref_combined[name], rtol=2e-5, atol=2e-7), f"A's gradient differs from the established producer for {name}")
        gradient_errors[name] = error
    loss_pairs = {
        "raw_mse": (losses["raw_mse"], ref_losses["L_rec"]),
        "l_dist": (losses["l_dist"], ref_losses["L_dist"]),
        "l_op": (losses["l_op"], ref_losses["L_op"]),
        **{f"{name}_ce": (losses[f"{name}_ce"], ref_losses[f"{name}_ce"]) for name in TRAIN_EVALUATORS},
    }
    loss_errors: dict[str, float] = {}
    for name, (actual, reference) in loss_pairs.items():
        left = torch.as_tensor(actual, dtype=torch.float64, device="cpu")
        right = torch.as_tensor(reference, dtype=torch.float64, device="cpu")
        error = float((left - right).abs().max())
        _require(torch.allclose(left, right, rtol=2e-5, atol=2e-7), f"A's {name} differs from the established producer")
        loss_errors[name] = error
    _require(all(parameter.grad is None for parameter in parameters.values()), "A reference comparison populated parameter .grad")
    return {"executed": True, "passed": True, "reference": "existing large-Generator logical-batch producer", "max_abs_gradient_difference": gradient_errors, "max_abs_loss_difference": loss_errors}


def _disk_and_io(target: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(target)
    estimated = 1_073_741_824
    _require(usage.free >= estimated + DISK_RESERVE, "output disk cannot preserve expected artifacts plus 1.5 GiB reserve")
    probe = target / "io_probe.tmp"
    data = bytes(range(256)) * 4096
    probe.write_bytes(data)
    _require(probe.read_bytes() == data, "task output write/read probe differs")
    probe.unlink()
    return {"executed": True, "passed": True, "free_bytes_observation": usage.free, "estimated_output_bytes": estimated, "required_reserve_bytes": DISK_RESERVE, "write_read_probe": True}


def _release_preflight_cuda_cache(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


def _preflight(inputs: Mapping[str, Any], target: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized production preflight")
    paths = inputs["evaluator_paths"]
    evaluators = diagnostic.load_evaluators(paths, device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    source_rng = _rng_clone(inputs["source"]["rng_state"])
    checks: dict[str, Any] = {}
    gpu_before = capacity._gpu_snapshot(device)
    try:
        checks["arm_a_matches_original_production"] = _verify_a_original_reference(inputs, evaluators)
        _release_preflight_cuda_cache(device)
        target_control = _target_positive_control(evaluators, inputs)
        checks["four_evaluator_target_positive_control"] = {key: value for key, value in target_control.items() if key != "logits" and key != "evaluators"}
        checks["four_evaluator_target_positive_control"]["per_evaluator"] = {name: {key: value for key, value in item.items() if key != "logits"} for name, item in target_control["evaluators"].items()}
        _require(target_control["passed"], "one or more frozen evaluator target positive controls failed")
        _write_torch(target / "target_positive_control.pt", {"source_indices": inputs["rows"][FIT_COUNT:], "labels": inputs["labels"][FIT_COUNT:], **{f"{name.lower()}_logits": item["logits"] for name, item in target_control["evaluators"].items()}})
        _release_preflight_cuda_cache(device)

        consumer, _optimizer, _parameters = _new_endpoint(inputs, device)
        start_readout = _readout(consumer, evaluators, inputs)
        replay = _replay_check(start_readout, inputs["readout"])
        checks["a4700_production_replay"] = {"executed": True, **replay}
        _write_torch(target / "readouts_start.pt", start_readout)
        del consumer, _optimizer, _parameters
        gc.collect()
        torch.cuda.empty_cache()

        capacity._restore_rng_state(source_rng)
        trial_a1 = _trial(ARM_A, inputs, evaluators, updates=3)
        capacity._restore_rng_state(source_rng)
        trial_a2 = _trial(ARM_A, inputs, evaluators, updates=3)
        aa_record = _aa_preflight_record(trial_a1, trial_a2, updates=3)
        checks["arm_a_three_real_updates_and_aa"] = aa_record
        _require(aa_record["passed"], f"A/A three-step path is not repeatable within the declared numerical tolerance: {aa_record['A_A_repeatability']['comparisons']}")
        del trial_a1, trial_a2
        gc.collect()
        torch.cuda.empty_cache()

        capacity._restore_rng_state(source_rng)
        trial_b = _trial(ARM_B, inputs, evaluators, updates=3, vjp=True)
        _require(trial_b["new_plane_gradient_norm"] > 0.0, "added Generator phase planes received no preflight gradient")
        checks["arm_b_three_real_updates_and_ssim_vjp"] = {"executed": True, "passed": True, "updates": 3, "generator_adam_step": SOURCE_STEP + 3, "new_phase_gradient_nonzero": True, "ssim_direct_image_space_vjp": True}
        del trial_b
        gc.collect()
        torch.cuda.empty_cache()

        checks["same_class_target_mismatch_negative_control"] = _same_class_target_control(inputs, evaluators)
        _release_preflight_cuda_cache(device)
        checks["wrong_label_ce_negative_control"] = _wrong_label_control(inputs, evaluators)
        _release_preflight_cuda_cache(device)
        checks["recovery_control"] = _recovery_control(inputs, evaluators, target)
        _release_preflight_cuda_cache(device)
        checks["cuda_and_disk"] = _disk_and_io(target)
        gpu_after = capacity._gpu_snapshot(device)
        checks["freeze_scope"] = {"executed": True, "passed": True, "parameter_gradients_empty": True, "evaluators_unchanged": all(_state_equal(evaluator_before[name], capacity._state_snapshot(model)) for name, model in evaluators.items()), "source_adam_unchanged": True}
        _require(checks["freeze_scope"]["evaluators_unchanged"], "preflight changed a frozen evaluator")
        for key, value in checks.items():
            _require(value.get("executed") is True and value.get("passed") is True, f"preflight check did not pass: {key}")
        return {"task_id": TASK_ID, "status": "READY", "checks": checks, "cuda_before": gpu_before, "cuda_after": gpu_after, "numeric_mode": NUMERIC_MODE, "source_replay": replay}, start_readout, target_control
    finally:
        capacity._restore_rng_state(source_rng)
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight evaluator {name}")
        gc.collect()
        torch.cuda.empty_cache()


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Optimizer, inputs: Mapping[str, Any], arm: str, local_completed: int, history: Sequence[Mapping[str, Any]], *, reason: str | None = None) -> dict[str, Any]:
    params = capacity._parameters(consumer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "source_step": SOURCE_STEP,
        "completed_updates": SOURCE_STEP + local_completed,
        "task_local_updates": local_completed,
        "optimizer_step": int(float(optimizer.state[params["generator.phase"]]["step"].item())),
        "next_batch_index": local_completed % BATCH_COUNT,
        "next_batch_cursor": local_completed % BATCH_COUNT,
        "architecture_layout": "two_original_100x100_then_two_400x400_phase_planes_flattened",
        "numeric_mode": NUMERIC_MODE,
        "optimizer_parameter_names": ["generator.phase"],
        "optimizer_state": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
        "trainable_parameter_state": {name: value.detach().cpu().clone() for name, value in params.items()},
        "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "references": inputs["source"]["references"],
        "weights": inputs["source"]["weights"],
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "population": inputs["population"],
        "rng_state": capacity._rng_state(),
        "source_rng_state": inputs["source"]["rng_state"],
        "training_history": list(inputs["source"]["training_history"]) + list(history),
        "source_history_count": 3_700,
        "source_task_local_history_count": 1_000,
        "checkpoint_reason": reason or "every_100_updates_or_fixed_budget_endpoint",
    }


def _checkpoint_path(target: Path, arm: str, local_completed: int) -> Path:
    letter = "a" if arm == ARM_A else "b"
    return target / f"arm_{letter}_step{SOURCE_STEP + local_completed:04d}.pt"


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, inputs: Mapping[str, Any], arm: str, local_completed: int, history: Sequence[Mapping[str, Any]], *, reason: str | None = None) -> Path:
    path = _checkpoint_path(target, arm, local_completed)
    _write_torch(path, _checkpoint_payload(consumer, optimizer, inputs, arm, local_completed, history, reason=reason))
    letter = "a" if arm == ARM_A else "b"
    _replace_json(target / f"training_history_arm_{letter}.json", {"task_id": TASK_ID, "arm": arm, "source_step": SOURCE_STEP, "additional_updates": local_completed, "records": list(history)})
    return path


def _read_checkpoint(path: Path, target: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file() and target.resolve() in path.resolve().parents, "resume checkpoint must be an explicit file inside this task output")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("task_id") == TASK_ID, "resume checkpoint belongs to another task/version")
    _require(payload.get("population") == inputs["population"] and payload.get("source_checkpoint") == inputs["population"]["source_checkpoint"], "resume checkpoint population/source differs")
    _require(payload.get("arm") in ARM_NAMES and payload.get("source_step") == SOURCE_STEP, "resume checkpoint arm/source step differs")
    _require(payload.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened" and payload.get("numeric_mode") == NUMERIC_MODE, "resume checkpoint architecture/numeric mode differs")
    local = payload.get("task_local_updates")
    _require(type(local) is int and 0 <= local <= ADDED_UPDATES, "resume checkpoint local update count is invalid")
    _require(payload.get("completed_updates") == SOURCE_STEP + local and payload.get("optimizer_step") == SOURCE_STEP + local, "resume checkpoint cumulative step differs")
    _require(payload.get("next_batch_index") == local % BATCH_COUNT and payload.get("next_batch_cursor") == local % BATCH_COUNT, "resume checkpoint next-batch cursor differs")
    _require(path.name == _checkpoint_path(target, payload["arm"], local).name, "resume checkpoint filename does not match saved arm/step")
    source_history = inputs["source"].get("training_history", [])
    saved_history = payload.get("training_history", [])
    _require(len(saved_history) == len(source_history) + local, "resume checkpoint history does not match its update count")
    _require(saved_history[: len(source_history)] == source_history, "resume checkpoint rewrote historical training records")
    _require(payload.get("source_history_count") == 3_700 and payload.get("source_task_local_history_count") == 1_000, "resume source history counts differ")
    local_history = saved_history[len(source_history):]
    _require(all(record.get("task_local_update") == index + 1 and record.get("optimizer_step") == SOURCE_STEP + index + 1 and record.get("arm") == payload["arm"] and record.get("batch_index") == index % BATCH_COUNT for index, record in enumerate(local_history)), "resume task-local history does not match its arm, step, or batch cursor")
    _require(_tree_equal(payload.get("source_rng_state"), inputs["source"]["rng_state"]), "resume checkpoint source RNG differs from A4700")
    _require(_tree_equal(payload.get("references"), inputs["source"]["references"]) and _tree_equal(payload.get("weights"), inputs["source"]["weights"]), "resume references/weights differ from A4700")
    _require(_tree_equal(payload.get("frozen_runtime_state"), inputs["source"]["frozen_runtime_state"]), "resume frozen OLS/runtime state differs from A4700")
    parameters = payload.get("trainable_parameter_state", {})
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "resume trainable parameter map is incomplete")
    for name, expected_shape in inputs["population"]["source_parameter_shapes"].items():
        value = parameters[name]
        _require(tuple(value.shape) == tuple(expected_shape) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"resume parameter shape/dtype/finiteness differs for {name}")
    optimizer = payload.get("optimizer_state", {})
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "resume Generator Adam parameter mapping differs")
    group = groups[0]
    _require(float(group.get("lr", -1.0)) == 1e-3 and tuple(group.get("betas", ())) == (0.9, 0.999) and float(group.get("eps", -1.0)) == 1e-8 and float(group.get("weight_decay", -1.0)) == 0.0, "resume Generator Adam hyperparameters differ")
    state = states[0]
    _require(int(float(state["step"].item())) == SOURCE_STEP + local and tuple(state["exp_avg"].shape) == (340_000,) and tuple(state["exp_avg_sq"].shape) == (340_000,), "resume Generator Adam count/moment shapes differ")
    _require(bool(torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()), "resume Generator Adam moments are nonfinite")
    return payload


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = payload["trainable_parameter_state"][name]
            _require(tuple(value.shape) == tuple(parameter.shape) and value.dtype == parameter.dtype, f"checkpoint parameter shape/dtype differs for {name}")
            parameter.copy_(value.to(parameter.device))
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == payload["optimizer_step"], "restored Adam step differs")
    capacity._assert_state_equal(payload["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "restored frozen OLS/runtime")
    capacity._restore_rng_state(payload["rng_state"])


def _recovery_control(inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], target: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    source_rng = _rng_clone(inputs["source"]["rng_state"])
    continuous_consumer, continuous_optimizer, continuous_params = _new_endpoint(inputs, device)
    capacity._restore_rng_state(source_rng)
    fit = _fit_inputs(inputs)
    continuous_history = [
        _train_step(continuous_consumer, continuous_optimizer, continuous_params, evaluators, fit, ARM_B, index)
        for index in range(2)
    ]
    continuous_state = {
        "phase": continuous_params["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(capacity._cpu_tree(continuous_optimizer.state_dict())),
        "history": continuous_history,
    }
    del continuous_consumer, continuous_optimizer, continuous_params
    gc.collect()
    torch.cuda.empty_cache()

    split_consumer, split_optimizer, split_params = _new_endpoint(inputs, device)
    capacity._restore_rng_state(source_rng)
    split_history = [_train_step(split_consumer, split_optimizer, split_params, evaluators, fit, ARM_B, 0)]
    temporary = target / "preflight_recovery_checkpoint.pt"
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": ARM_B,
        "source_step": SOURCE_STEP,
        "completed_updates": SOURCE_STEP + 1,
        "task_local_updates": 1,
        "optimizer_step": SOURCE_STEP + 1,
        "next_batch_index": 1,
        "next_batch_cursor": 1,
        "architecture_layout": "two_original_100x100_then_two_400x400_phase_planes_flattened",
        "optimizer_parameter_names": ["generator.phase"],
        "optimizer_state": copy.deepcopy(capacity._cpu_tree(split_optimizer.state_dict())),
        "trainable_parameter_state": {name: value.detach().cpu().clone() for name, value in split_params.items()},
        "frozen_runtime_state": capacity._frozen_runtime_state(split_consumer),
        "references": inputs["source"]["references"],
        "weights": inputs["source"]["weights"],
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "population": inputs["population"],
        "rng_state": capacity._rng_state(),
        "source_rng_state": inputs["source"]["rng_state"],
        "training_history": list(inputs["source"]["training_history"]) + split_history,
        "source_history_count": 3_700,
        "source_task_local_history_count": 1_000,
    }
    _write_torch(temporary, state)
    loaded = torch.load(temporary, map_location="cpu", weights_only=True)
    _restore_checkpoint(loaded, split_consumer, split_optimizer)
    split_history.append(_train_step(split_consumer, split_optimizer, split_params, evaluators, fit, ARM_B, 1))
    split_state = {
        "phase": split_params["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(capacity._cpu_tree(split_optimizer.state_dict())),
        "history": split_history,
    }
    repeatability = _trial_repeatability(continuous_state, split_state)
    _require(repeatability["passed"], f"continuous and save/restore Generator updates differ beyond the declared numerical tolerance: {repeatability['comparisons']}")
    temporary.unlink()
    return {"executed": True, "passed": True, "continuous_equals_save_restore": repeatability, "updates": 2, "checkpoint_step": SOURCE_STEP + 1}


def _make_identity(inputs: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="fixed-large-generator-raw-mse-vs-ssim-4700-to-5400",
        semantic_authority_id=generation.TASK_ID,
        data_role="canonical TRAIN fit=[0,200), confirmation=[3000,3300); confirmation excluded from updates and endpoint choice",
        config_identity={
            "source_step": SOURCE_STEP,
            "updates_per_arm": ADDED_UPDATES,
            "logical_batch": BATCH_SIZE,
            "microbatch": MICROBATCH_SIZE,
            "optimizer": "inherited Generator Adam state",
            "raw_reconstruction": "mean raw 200x200 MSE",
            "structural_reconstruction": {"weight": SSIM_WEIGHT, "window": SSIM_KERNEL, "sigma": SSIM_SIGMA, "stats": "float64 weighted population moments"},
            "strict_fp32_optics": True,
        },
        model_identity={
            "source_checkpoint": inputs["population"]["source_checkpoint"],
            "architecture": "two_original_100x100_then_two_400x400_phase_planes_flattened",
            "parameter_shapes": inputs["population"]["source_parameter_shapes"],
            "finite": True,
        },
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "reset": False},
        sampler_identity={"fit_source_rows": inputs["rows"][:FIT_COUNT].tolist(), "fit_batch_count": BATCH_COUNT, "confirmation_source_rows": inputs["rows"][FIT_COUNT:].tolist()},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _make_receipt(inputs: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    checks = preflight.get("checks", {})
    _require(checks and all(item.get("executed") is True and item.get("passed") is True for item in checks.values()), "READY cannot be made without every executed passing preflight check")
    control = ResearchRunControl(_make_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "four_target_controls": checks["four_evaluator_target_positive_control"]["passed"], "A4700_replay": checks["a4700_production_replay"]["passed"], "A_matches_original_production": checks["arm_a_matches_original_production"]["passed"], "A_B_three_step_updates": checks["arm_a_three_real_updates_and_aa"]["passed"] and checks["arm_b_three_real_updates_and_ssim_vjp"]["passed"], "SSIM_VJP_and_wrong_pair_controls": checks["arm_b_three_real_updates_and_ssim_vjp"]["passed"] and checks["same_class_target_mismatch_negative_control"]["passed"], "wrong_label_control": checks["wrong_label_ce_negative_control"]["passed"], "recovery": checks["recovery_control"]["passed"], "frozen_scope": checks["freeze_scope"]["passed"]},
        resource={"passed": checks["cuda_and_disk"]["passed"], "status": "PASS", "cuda_available": True, "gpu_observation": preflight["cuda_after"], "disk": checks["cuda_and_disk"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_2048_large_generator_only_strict_fp32", "three_real_updates_per_arm": True, "A_A_repeatability": checks["arm_a_three_real_updates_and_aa"]["A_A_repeatability"], "explicit_checkpoint_recovery": checks["recovery_control"]["passed"]},
        selected_engine="cuda_2048_large_generator_only_strict_fp32",
        resource_envelope={"cuda": preflight["cuda_after"], "disk": checks["cuda_and_disk"]},
    )
    return control, receipt


def _stage_ready(target: Path, inputs: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    expected_population = inputs["population"]
    expected = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}
    population_path, preflight_path = target / "input_population.json", target / "preflight.json"
    if resume:
        _require(population_path.is_file() and json.loads(population_path.read_text(encoding="utf-8")) == expected_population, "saved READY population differs")
        _require(preflight_path.is_file() and json.loads(preflight_path.read_text(encoding="utf-8")) == expected, "saved READY receipt/preflight differs")
    else:
        if population_path.exists():
            _require(json.loads(population_path.read_text(encoding="utf-8")) == expected_population, "staged population differs")
        else:
            _write_json(population_path, expected_population)
        if preflight_path.exists():
            _require(json.loads(preflight_path.read_text(encoding="utf-8")) == expected, "staged READY evidence differs")
        else:
            _write_json(preflight_path, expected)


def _validate_ready(target: Path, inputs: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt, dict[str, Any]]:
    payload = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(payload.get("task_id") == TASK_ID and payload.get("status") == "READY", "saved READY belongs to another task or is incomplete")
    _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == inputs["population"], "population changed after preflight")
    preflight = payload["preflight"]
    _require(preflight.get("status") == "READY" and all(value.get("executed") is True and value.get("passed") is True for value in preflight.get("checks", {}).values()), "READY consumer found a missing or failed measured check")
    control = ResearchRunControl(_make_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    saved = PreflightReceipt.from_mapping(payload["receipt"])
    _require(saved.source_semantics_id == TASK_ID, "saved READY receipt belongs to another task")
    _require(saved.authorization_id == f"{TASK_ID}:authorization:01", "saved READY receipt has a different authorization")
    _require(saved.status == "READY", "saved READY receipt is not ready")
    _require(all(group.get("passed") is True for group in saved.checks.values()), "saved READY receipt is missing an executed passing result")
    control.mark_preflight_ready(saved)
    return control, saved, preflight


def _run_arm(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], arm: str, start_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    consumer, optimizer, parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    fit_inputs = _fit_inputs(inputs)
    if start_payload is None:
        local_completed = 0
        history: list[dict[str, Any]] = []
        capacity._restore_rng_state(inputs["source"]["rng_state"])
        start_path = _checkpoint_path(target, arm, 0)
        if not start_path.exists():
            _save_checkpoint(target, consumer, optimizer, inputs, arm, 0, history, reason="copied_A4700_source_start")
    else:
        _require(start_payload.get("arm") == arm, "explicit resume checkpoint belongs to a different arm")
        _restore_checkpoint(start_payload, consumer, optimizer)
        local_completed = int(start_payload["task_local_updates"])
        history = list(start_payload["training_history"][len(inputs["source"]["training_history"]):])
    _require(0 <= local_completed <= ADDED_UPDATES and len(history) == local_completed, "task-local resume history/cursor differs")
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    signal_state = {"stop": False}
    prior_handler = signal.getsignal(signal.SIGINT)
    def request_stop(_signum, _frame):
        signal_state["stop"] = True
    signal.signal(signal.SIGINT, request_stop)
    try:
        while local_completed < ADDED_UPDATES:
            record = _train_step(consumer, optimizer, parameters, evaluators, fit_inputs, arm, local_completed)
            history.append(record)
            local_completed += 1
            if local_completed % 100 == 0 or signal_state["stop"]:
                _save_checkpoint(target, consumer, optimizer, inputs, arm, local_completed, history, reason="interrupt_boundary" if signal_state["stop"] else None)
                telemetry = capacity._gpu_snapshot(torch.device("cuda:0"))
                with (target / "resource_telemetry.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"arm": arm, "local_update": local_completed, "optimizer_step": SOURCE_STEP + local_completed, "gpu_observation": telemetry}, sort_keys=True) + "\n")
            if signal_state["stop"]:
                raise TrainingInterrupted(f"{arm} durably checkpointed at task-local update {local_completed}")
        if local_completed % 100 != 0:
            _save_checkpoint(target, consumer, optimizer, inputs, arm, local_completed, history)
        _assert_tensor_unchanged(encoder_before, parameters["encoder.phase"].detach().cpu(), f"{arm} final Encoder")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{arm} final frozen OLS/runtime")
        _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == FINAL_STEP, f"{arm} final Generator Adam step differs")
        return torch.load(_checkpoint_path(target, arm, ADDED_UPDATES), map_location="cpu", weights_only=True)
    finally:
        signal.signal(signal.SIGINT, prior_handler)
        gc.collect()
        torch.cuda.empty_cache()


def _endpoint(target: Path, name: str, checkpoint: Mapping[str, Any], inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    consumer, optimizer, _parameters = _new_endpoint(inputs, torch.device("cuda:0"))
    if name == "A4700":
        capacity._restore_rng_state(inputs["source"]["rng_state"])
    else:
        _restore_checkpoint(checkpoint, consumer, optimizer)
    measured = _readout(consumer, evaluators, inputs)
    path = target / f"readouts_{name}.pt"
    _write_torch(path, measured)
    summary, row_metrics = class9._row_and_image_metrics(measured, inputs["population"])
    summary["endpoint"] = name
    _write_json(target / f"metrics_{name}.json", summary)
    _write_torch(target / f"row_metrics_{name}.pt", row_metrics)
    return summary, measured, row_metrics


def _image_data_uri(view: torch.Tensor) -> str:
    image = Image.frombytes("L", (28, 28), bytes(view.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).flatten().tolist()))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    import base64
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _review_html(public_items: Sequence[Mapping[str, str]], session_key: str) -> str:
    encoded = json.dumps(list(public_items), ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>LightGen 直觉盲读</title>
<style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:18px;background:#f4f4f4;color:#222}}header{{position:sticky;top:0;background:white;padding:12px;border:1px solid #bbb;z-index:2}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px;margin-top:14px}}.card{{background:white;border:1px solid #ddd;padding:8px;text-align:center}}img{{width:196px;height:196px;image-rendering:pixelated;background:white}}.id{{font-family:monospace;color:#555;margin:4px}}select,button{{font-size:16px;padding:7px;margin:4px}}@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}}img{{width:150px;height:150px}}}}</style></head><body>
<header><b>LightGen 图像第一直觉盲读</b><p>每项只按第一直觉选择数字0—9或“看不清”。图片使用统一灰度范围，不做单图增强。页面不显示样本来源、类别或评价器结果。</p>
<span id=\"progress\"></span><div><button id=\"prev\">上一页</button><span id=\"page\"></span><button id=\"next\">下一页</button><button id=\"export\">导出评分CSV</button></div><span id=\"message\">共680项，可保存进度后继续。</span></header><main id=\"grid\" class=\"grid\"></main>
<script>const items={encoded};const pageSize=20;let page=0;const answers={{}};const key='lightgen-stroke-fidelity-v1:{session_key}';try{{Object.assign(answers,JSON.parse(localStorage.getItem(key)||'{{}}'))}}catch{{}}const grid=document.getElementById('grid');
function update(){{let n=Object.keys(answers).length;document.getElementById('progress').textContent=`已完成 ${{n}} / 680 项`;document.getElementById('export').disabled=n!==680;document.getElementById('message').textContent=n===680?'已完成，可以导出评分CSV。':'共680项，可保存进度后继续。'}}
function render(){{grid.innerHTML='';let begin=page*pageSize;for(const item of items.slice(begin,begin+pageSize)){{let card=document.createElement('section');card.className='card';let img=document.createElement('img');img.src=item.image;img.alt='';img.draggable=false;let id=document.createElement('div');id.className='id';id.textContent='编号 '+item.item_id;let select=document.createElement('select');select.setAttribute('aria-label','图像编号 '+item.item_id);let empty=document.createElement('option');empty.value='';empty.textContent='请选择';select.appendChild(empty);for(let d=0;d<10;d++){{let o=document.createElement('option');o.value=String(d);o.textContent=String(d);select.appendChild(o)}}let unclear=document.createElement('option');unclear.value='unclear';unclear.textContent='看不清';select.appendChild(unclear);select.value=answers[item.item_id]||'';select.onchange=()=>{{if(select.value)answers[item.item_id]=select.value;else delete answers[item.item_id];localStorage.setItem(key,JSON.stringify(answers));update()}};card.append(img,id,select);grid.appendChild(card)}}document.getElementById('page').textContent=`第 ${{page+1}} / ${{Math.ceil(items.length/pageSize)}} 页`;document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=begin+pageSize>=items.length;update()}}
document.getElementById('prev').onclick=()=>{{page--;render()}};document.getElementById('next').onclick=()=>{{page++;render()}};document.getElementById('export').onclick=()=>{{if(Object.keys(answers).length!==680)return;let rows=['item_id,answer',...items.map(x=>`${{x.item_id}},${{answers[x.item_id]}}`)];let blob=new Blob(['\\ufeff'+rows.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='lightgen_stroke_fidelity_ratings.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};render();</script></body></html>"""


def _prepare_review(target: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    confirm_labels = inputs["labels"][FIT_COUNT:]
    generator = torch.Generator(device="cpu").manual_seed(REVIEW_SAMPLE_SEED)
    selected_local: dict[int, list[int]] = {}
    anchor_local: dict[int, list[int]] = {}
    for digit in range(10):
        candidates = torch.nonzero(confirm_labels == digit, as_tuple=False).flatten().sort().values
        _require(candidates.numel() == 30, f"confirmation class {digit} does not have 30 rows")
        permutation = torch.randperm(30, generator=generator)
        chosen = candidates.index_select(0, permutation[:REVIEW_ROWS_PER_CLASS])
        leftover = sorted(set(candidates.tolist()) - set(chosen.tolist()))
        selected_local[digit] = chosen.tolist()
        anchor_local[digit] = leftover[:REVIEW_TARGETS_PER_CLASS]
    records: list[dict[str, Any]] = []
    primary_by_endpoint_class: dict[tuple[str, int], list[dict[str, Any]]] = {}
    id_set: set[str] = set()
    endpoint_readout_keys = (("start", "A4700"), ("A", "A5400"), ("B", "B5400"))

    def add(view: torch.Tensor, *, kind: str, true_class: int, endpoint: str | None, row_position: int | None, e_d_prediction: int | None, repeat_of: str | None = None) -> dict[str, Any]:
        item_id = secrets.token_hex(8)
        while item_id in id_set:
            item_id = secrets.token_hex(8)
        id_set.add(item_id)
        record = {"item_id": item_id, "kind": kind, "true_class": int(true_class), "endpoint": endpoint, "measurement_position": row_position, "source_row_index": None if row_position is None else int(inputs["rows"][row_position]), "e_d_prediction": e_d_prediction, "repeat_of": repeat_of, "_view": view.detach().cpu().clone(), "_image": _image_data_uri(view)}
        records.append(record)
        return record

    for endpoint, readout_name in endpoint_readout_keys:
        readout = readouts[readout_name]
        predictions = readout["e_d_logits"].to(torch.float64).argmax(1)
        for digit in range(10):
            bucket = []
            for local_confirmation in selected_local[digit]:
                measurement_position = FIT_COUNT + local_confirmation
                row = add(readout["evaluator_view"][measurement_position, 0], kind="primary_generated", true_class=digit, endpoint=endpoint, row_position=measurement_position, e_d_prediction=int(predictions[measurement_position]))
                bucket.append(row)
            primary_by_endpoint_class[(endpoint, digit)] = bucket

    for digit in range(10):
        for local_confirmation in anchor_local[digit]:
            position = FIT_COUNT + local_confirmation
            add(inputs["images"][position, 0], kind="real_target_anchor", true_class=digit, endpoint=None, row_position=position, e_d_prediction=None)

    repeat_rng = random.Random(REVIEW_ORDER_SEED)
    for endpoint, _readout_name in endpoint_readout_keys:
        for digit in range(10):
            primary = repeat_rng.choice(primary_by_endpoint_class[(endpoint, digit)])
            add(primary["_view"], kind="hidden_repeat", true_class=digit, endpoint=endpoint, row_position=primary["measurement_position"], e_d_prediction=primary["e_d_prediction"], repeat_of=primary["item_id"])

    # Keep the source pixels for repeats identical while excluding them from the public mapping.
    by_id = {record["item_id"]: record for record in records}
    for record in records:
        if record["kind"] == "primary_generated":
            position = int(record["measurement_position"])
            endpoint = str(record["endpoint"])
            key = {"start": "A4700", "A": "A5400", "B": "B5400"}[endpoint]
            record["_view"] = readouts[key]["evaluator_view"][position, 0].detach().cpu()
        elif record["kind"] == "real_target_anchor":
            record["_view"] = inputs["images"][int(record["measurement_position"]), 0]
        else:
            source = by_id[str(record["repeat_of"])]
            record["_view"] = source["_view"].clone()
            record["_image"] = source["_image"]

    _require(len(records) == REVIEW_TOTAL and len(id_set) == REVIEW_TOTAL, "blind review item count/IDs differ from 680")
    _require(sum(row["kind"] == "primary_generated" for row in records) == REVIEW_PRIMARY_COUNT, "primary blind sample does not contain 600 views")
    _require(sum(row["kind"] == "real_target_anchor" for row in records) == REVIEW_TARGET_COUNT, "review anchors do not contain 50 targets")
    _require(sum(row["kind"] == "hidden_repeat" for row in records) == REVIEW_REPEAT_COUNT, "review hidden repeats do not contain 30 items")
    for endpoint, _ in endpoint_readout_keys:
        _require(all(sum(row["kind"] == "primary_generated" and row["endpoint"] == endpoint and row["true_class"] == digit for row in records) == 20 for digit in range(10)), f"{endpoint} review sample is not 20 per class")
    order = random.Random(REVIEW_ORDER_SEED)
    order.shuffle(records)
    public_items = [{"item_id": row["item_id"], "image": row["_image"]} for row in records]
    mapping = [{key: value for key, value in row.items() if not key.startswith("_")} for row in records]
    session_key = secrets.token_hex(8)
    _write_json(target / "review_mapping.json", {"task_id": TASK_ID, "review_session_id": session_key, "sample_seed": REVIEW_SAMPLE_SEED, "order_seed": REVIEW_ORDER_SEED, "confirmation_positions_by_class": {str(k): v for k, v in selected_local.items()}, "target_positions_by_class": {str(k): v for k, v in anchor_local.items()}, "items": mapping})
    html = _review_html(public_items, session_key)
    (target / "blind_review.html").write_text(html, encoding="utf-8")
    template = io.StringIO(newline="")
    writer = csv.writer(template, lineterminator="\r\n")
    writer.writerow(("item_id", "answer"))
    for row in mapping:
        writer.writerow((row["item_id"], ""))
    (target / "ratings_template.csv").write_text(template.getvalue(), encoding="utf-8-sig")
    return {"task_id": TASK_ID, "state": "PREPARED_AWAITING_RATINGS", "item_count": REVIEW_TOTAL, "primary_generated_count": REVIEW_PRIMARY_COUNT, "real_target_anchor_count": REVIEW_TARGET_COUNT, "hidden_repeat_count": REVIEW_REPEAT_COUNT, "review_html": "blind_review.html", "ratings_template": "ratings_template.csv", "explicit_rating_input_required": "--ratings PATH", "answer_domain": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "unclear"], "labels_and_endpoints_hidden_in_review_page": True}


def _validate_review_mapping(mapping: Sequence[Mapping[str, Any]]) -> None:
    _require(len(mapping) == REVIEW_TOTAL and len({row.get("item_id") for row in mapping}) == REVIEW_TOTAL, "review mapping does not contain 680 unique item IDs")
    counts = {kind: sum(row.get("kind") == kind for row in mapping) for kind in ("primary_generated", "real_target_anchor", "hidden_repeat")}
    _require(counts == {"primary_generated": 600, "real_target_anchor": 50, "hidden_repeat": 30}, "review mapping type counts differ")
    for endpoint in ("start", "A", "B"):
        for digit in range(10):
            _require(sum(row.get("kind") == "primary_generated" and row.get("endpoint") == endpoint and row.get("true_class") == digit for row in mapping) == 20, "review mapping lost its fixed endpoint/class sample")
    primary_by_id = {row["item_id"]: row for row in mapping if row["kind"] == "primary_generated"}
    for row in mapping:
        if row["kind"] == "hidden_repeat":
            original = primary_by_id.get(row.get("repeat_of"))
            _require(original is not None and original["true_class"] == row["true_class"] and original["endpoint"] == row["endpoint"] and original["source_row_index"] == row["source_row_index"], "hidden repeat does not point to the identical endpoint/class/row")


def _parse_ratings(path: Path, mapping: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    _validate_review_mapping(mapping)
    _require(path.is_file(), f"explicit completed ratings file is missing: {path}")
    expected = {str(row["item_id"]) for row in mapping}
    answers: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == ["item_id", "answer"], "ratings header must be item_id,answer")
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            answer = str(row.get("answer", "")).strip().lower()
            _require(item_id in expected and item_id not in answers, "ratings contain an unknown or duplicate item ID")
            _require(answer in {str(i) for i in range(10)} | {"unclear"}, f"invalid rating for {item_id}")
            answers[item_id] = answer
    _require(set(answers) == expected, "ratings do not cover all 680 review items")
    return answers


def _wilson(success: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _score_review(mapping: Sequence[Mapping[str, Any]], answers: Mapping[str, str], readouts: Mapping[str, Mapping[str, Any]], endpoint_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _validate_review_mapping(mapping)
    _require(len(answers) == REVIEW_TOTAL, "rating answer count differs")
    primary = [row for row in mapping if row["kind"] == "primary_generated"]
    anchors = [row for row in mapping if row["kind"] == "real_target_anchor"]
    repeats = [row for row in mapping if row["kind"] == "hidden_repeat"]
    endpoint_result: dict[str, Any] = {}
    human_correct: dict[str, bool] = {}
    for row in primary:
        key = {"start": "A4700", "A": "A5400", "B": "B5400"}[str(row["endpoint"])]
        readout = readouts[key]
        position = int(row["measurement_position"])
        _require(int(readout["source_indices"][position]) == int(row["source_row_index"]), "blind mapping source row differs from saved endpoint readout")
        _require(int(readout["labels"][position]) == int(row["true_class"]), "blind mapping class differs from saved endpoint readout")
        prediction = int(readout["e_d_logits"][position].to(torch.float64).argmax())
        _require(prediction == int(row["e_d_prediction"]), "blind mapping E_D prediction differs from saved endpoint logits")
    for endpoint in ("start", "A", "B"):
        rows = [row for row in primary if row["endpoint"] == endpoint]
        confusion = torch.zeros((10, 11), dtype=torch.long)
        by_class: dict[str, Any] = {}
        total = 0
        for digit in range(10):
            class_rows = [row for row in rows if int(row["true_class"]) == digit]
            correct = sum(answers[row["item_id"]] == str(digit) for row in class_rows)
            for row in class_rows:
                answer = answers[row["item_id"]]
                confusion[digit, 10 if answer == "unclear" else int(answer)] += 1
                human_correct[row["item_id"]] = answer == str(digit)
            ed_correct = sum(int(row["e_d_prediction"]) == digit for row in class_rows)
            by_class[str(digit)] = {"human_correct": correct, "total": len(class_rows), "accuracy": correct / len(class_rows), "wilson_95": _wilson(correct, len(class_rows)), "e_d_correct_same_images": ed_correct}
            total += correct
        endpoint_result[endpoint] = {"human_correct_total": total, "total": len(rows), "accuracy": total / len(rows), "wilson_95": _wilson(total, len(rows)), "per_class": by_class, "confusion_actual_rows_answer_columns_0_to_9_unclear": confusion.tolist()}

    anchors_correct = sum(answers[row["item_id"]] == str(row["true_class"]) for row in anchors)
    repeat_agreement = sum(answers[row["item_id"]] == answers[str(row["repeat_of"])] for row in repeats)
    pairs: dict[str, Any] = {}
    for endpoint, key in (("start", "A4700"), ("A", "A5400"), ("B", "B5400")):
        confusion = {"human_and_e_d_correct": 0, "human_only_correct": 0, "e_d_only_correct": 0, "both_incorrect": 0}
        for row in primary:
            if row["endpoint"] != endpoint:
                continue
            human_ok = human_correct[row["item_id"]]
            ed_ok = int(row["e_d_prediction"]) == int(row["true_class"])
            label = "human_and_e_d_correct" if human_ok and ed_ok else "human_only_correct" if human_ok else "e_d_only_correct" if ed_ok else "both_incorrect"
            confusion[label] += 1
        pairs[endpoint] = confusion
    row_by_endpoint = {(row["endpoint"], row["source_row_index"]): row for row in primary}
    paired_transitions = {}
    for comparator in ("start", "A"):
        both = [row for row in primary if row["endpoint"] == "B"]
        transitions = {"both_correct": 0, "B_only_correct": 0, "comparator_only_correct": 0, "both_incorrect": 0}
        for b_row in both:
            c_row = row_by_endpoint[(comparator, b_row["source_row_index"])]
            b_ok = human_correct[b_row["item_id"]]
            c_ok = human_correct[c_row["item_id"]]
            label = "both_correct" if b_ok and c_ok else "B_only_correct" if b_ok else "comparator_only_correct" if c_ok else "both_incorrect"
            transitions[label] += 1
        paired_transitions[f"B_vs_{comparator}"] = transitions

    quality = {"real_target_correct": anchors_correct, "real_target_total": len(anchors), "real_target_minimum": ANCHOR_MIN, "hidden_repeat_agreements": repeat_agreement, "hidden_repeat_total": len(repeats), "hidden_repeat_minimum": REPEAT_MIN}
    quality["passed"] = len(anchors) == 50 and anchors_correct >= ANCHOR_MIN and len(repeats) == 30 and repeat_agreement >= REPEAT_MIN
    start_n = endpoint_result["start"]["human_correct_total"]
    a_n = endpoint_result["A"]["human_correct_total"]
    b_n = endpoint_result["B"]["human_correct_total"]
    class_b = [int(endpoint_result["B"]["per_class"][str(c)]["human_correct"]) for c in range(10)]
    class_a = [int(endpoint_result["A"]["per_class"][str(c)]["human_correct"]) for c in range(10)]
    class_start = [int(endpoint_result["start"]["per_class"][str(c)]["human_correct"]) for c in range(10)]
    confirmation_raw = {key: float(endpoint_summaries[key]["image_metrics_by_role"]["confirmation"]["raw_mse"]) for key in ("A4700", "A5400", "B5400")}
    confirmation_metrics = {key: endpoint_summaries[key]["image_metrics_by_role"]["confirmation"] for key in ("A4700", "A5400", "B5400")}
    b_image = confirmation_metrics["B5400"]
    no_image_risk = b_image["raw_fraction_above_one"] <= 0.01 and all(float(b_image["by_class"][str(c)]["raw_fraction_above_one"]) <= 0.01 for c in range(10)) and b_image["raw_zero_image"] == 0 and b_image["raw_constant_image"] == 0 and endpoint_summaries["B5400"]["all_finite"] is True
    conditions = {
        "B_human_readability_at_least_120_of_200": b_n >= HUMAN_TOTAL_MIN,
        "B_at_least_30_more_than_A_and_start": b_n >= a_n + HUMAN_DELTA_MIN and b_n >= start_n + HUMAN_DELTA_MIN,
        "no_class_decreases_more_than_2_of_20": all(class_b[c] >= class_a[c] - CLASS_DROP_MAX and class_b[c] >= class_start[c] - CLASS_DROP_MAX for c in range(10)),
        "confirmation_raw_mse_within_5_percent_of_A_and_start": confirmation_raw["B5400"] <= 1.05 * confirmation_raw["A5400"] and confirmation_raw["B5400"] <= 1.05 * confirmation_raw["A4700"],
        "clipping_and_image_integrity": no_image_risk,
        "review_quality": quality["passed"],
    }
    if not quality["passed"]:
        status, classification = "REVIEW_INCONCLUSIVE", "REVIEW_EVIDENCE_INSUFFICIENT"
    elif all(conditions.values()):
        status, classification = "VALID_COMPLETED", "STROKE_FIDELITY_GAIN_SUPPORTED"
    elif b_n > a_n and b_n > start_n:
        status, classification = "VALID_FAILED", "MIXED_OR_INSUFFICIENT"
    else:
        status, classification = "VALID_FAILED", "STROKE_FIDELITY_NOT_SUPPORTED_WITHIN_700_UPDATES"
    return {
        "review_quality": quality,
        "human_primary_by_endpoint": endpoint_result,
        "human_vs_e_d_same_images": pairs,
        "human_paired_transitions": paired_transitions,
        "confirmation_raw_mse": confirmation_raw,
        "decision": {"status": status, "classification": classification, "conditions": conditions, "training_recommendation": "NONE", "next_action": "NONE", "stop": True},
    }


def _finalize_ratings(target: Path, ratings_path: Path) -> dict[str, Any]:
    report_path = target / "report.json"
    _require(report_path.is_file(), "stroke-fidelity attempt report is missing")
    current = json.loads(report_path.read_text(encoding="utf-8"))
    _require(current.get("task_id") == TASK_ID, "attempt report belongs to another task")
    mapping_payload = json.loads((target / "review_mapping.json").read_text(encoding="utf-8"))
    _require(mapping_payload.get("task_id") == TASK_ID, "review mapping belongs to another task")
    mapping = mapping_payload.get("items", [])
    answers = _parse_ratings(ratings_path, mapping)
    result_path = target / "ratings_result.json"
    if current.get("status") != "AWAITING_RATINGS":
        _require(current.get("status") in {"REVIEW_INCONCLUSIVE", "VALID_COMPLETED", "VALID_FAILED"}, "task is not awaiting explicit ratings or already finalized")
        _require(result_path.is_file(), "terminal ratings report lacks its saved answer record")
        prior_result = json.loads(result_path.read_text(encoding="utf-8"))
        _require(prior_result.get("task_id") == TASK_ID and prior_result.get("answers") == answers, "refusing to replace a finalized report with different answers")
        return current
    _require(isinstance(current.get("endpoint_metrics"), Mapping), "attempt report lacks endpoint summaries")
    readout_files = {"A4700": "readouts_start.pt", "A5400": "readouts_A5400.pt", "B5400": "readouts_B5400.pt"}
    readouts = {name: torch.load(target / filename, map_location="cpu", weights_only=True) for name, filename in readout_files.items()}
    summaries = {name: current["endpoint_metrics"].get(name) for name in readout_files}
    _require(all(isinstance(summary, Mapping) for summary in summaries.values()), "attempt report lacks an endpoint summary")
    scored = _score_review(mapping, answers, readouts, summaries)
    result = {**current, "status": scored["decision"]["status"], "classification": scored["decision"]["classification"], "human_review_result": scored, "ratings_source": ratings_path.as_posix(), "training_recommendation": "NONE", "next_action": "NONE"}
    pending_path = target / "report_awaiting_ratings.json"
    if pending_path.is_file():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        _require(pending.get("task_id") == TASK_ID and pending.get("status") == "AWAITING_RATINGS", "saved awaiting-ratings report belongs to a different state")
    else:
        _write_json(pending_path, current)
    if result_path.is_file():
        prior_result = json.loads(result_path.read_text(encoding="utf-8"))
        _require(prior_result.get("task_id") == TASK_ID and prior_result.get("answers") == answers, "refusing to replace a saved ratings result with different answers")
    else:
        _write_json(result_path, {"task_id": TASK_ID, "human_review_result": scored, "answers": answers})
    _replace_json(report_path, result)
    return result


def _prepare_target(root: Path, target: Path, *, resume: bool, preflight_only: bool) -> None:
    outputs = (root / "outputs").resolve()
    _require(target != outputs and outputs in target.parents, "output must be a child of outputs/")
    parts = target.relative_to(outputs).parts
    _require(len(parts) == 3 and parts[1] == "runs" and parts[0].startswith("20261008_"), "output must use a versioned outputs/YYYYMMDD_name/runs/attempt path")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected/restored/temporary output namespace is forbidden")
    if not target.exists():
        _require(not resume, "resume output directory does not exist")
        target.mkdir(parents=True, exist_ok=False)
        return
    _require(target.is_dir(), "output path is not a directory")
    if resume:
        _require((target / "preflight.json").is_file(), "resume lacks READY preflight evidence")
        return
    allowed_stage = {"input_population.json", "preflight.json", "readouts_start.pt", "target_positive_control.pt", "preflight_recovery_checkpoint.pt"}
    existing = {path.name for path in target.iterdir()}
    _require(not existing or (existing.issubset(allowed_stage) and (target / "preflight.json").is_file()), "existing output is not an empty or valid READY stage; use explicit --resume")
    _require(not (target / "report.json").is_file(), "completed output requires ratings import or explicit terminal recovery")


def _run_training(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], selected: Mapping[str, Any] | None) -> dict[str, Any]:
    if selected is None:
        final_a = _run_arm(target, inputs, evaluators, ARM_A, None)
        final_b = _run_arm(target, inputs, evaluators, ARM_B, None)
    elif selected["arm"] == ARM_A:
        final_a = _run_arm(target, inputs, evaluators, ARM_A, selected)
        bstart_path = _checkpoint_path(target, ARM_B, 0)
        if bstart_path.is_file():
            bstart = _read_checkpoint(bstart_path, target, inputs)
        else:
            bstart = None
        final_b = _run_arm(target, inputs, evaluators, ARM_B, bstart)
    else:
        afinal_path = _checkpoint_path(target, ARM_A, ADDED_UPDATES)
        _require(afinal_path.is_file(), "B resume requires the completed A5400 comparison checkpoint")
        final_a = _read_checkpoint(afinal_path, target, inputs)
        final_b = _run_arm(target, inputs, evaluators, ARM_B, selected)
    _require(final_a["completed_updates"] == final_b["completed_updates"] == FINAL_STEP, "both arms must complete at Generator Adam step5400")
    return {ARM_A: final_a, ARM_B: final_b}


def _execute_training(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    generation_dir = _resolve(root, args.generation_dir)
    target = _resolve(root, args.output_dir)
    _require(bool(args.resume) == (args.checkpoint is not None), "--resume requires an explicit --checkpoint and vice versa")
    _require(not (args.preflight_only and args.resume), "--preflight-only cannot be combined with --resume")
    _prepare_target(root, target, resume=args.resume, preflight_only=args.preflight_only)
    inputs = _load_population(root, generation_dir)
    population_path = target / "input_population.json"
    if population_path.exists():
        _require(json.loads(population_path.read_text(encoding="utf-8")) == inputs["population"], "staged population differs from loaded canonical TRAIN rows")
    else:
        _write_json(population_path, inputs["population"])
    started = time.perf_counter()
    selected = None
    if args.resume:
        selected = _read_checkpoint(_resolve(root, args.checkpoint), target, inputs)
        existing_report = target / "report.json"
        if existing_report.is_file():
            _require(selected["arm"] == ARM_B and selected["task_local_updates"] == ADDED_UPDATES, "terminal recovery requires explicit B5400 checkpoint")
            return json.loads(existing_report.read_text(encoding="utf-8"))

    device = torch.device("cuda:0")
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == NUMERIC_MODE, "active strict FP32 runtime settings differ")
        evaluators = diagnostic.load_evaluators(inputs["evaluator_paths"], device)
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        if args.resume:
            _control, _receipt, preflight = _validate_ready(target, inputs)
        else:
            if (target / "preflight.json").is_file():
                _control, _receipt, preflight = _validate_ready(target, inputs)
            else:
                preflight, start_readout, target_control = _preflight(inputs, target)
                control, receipt = _make_receipt(inputs, preflight)
                _stage_ready(target, inputs, preflight, receipt, resume=False)
                if args.preflight_only:
                    return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "checks": preflight["checks"], "next_action": "NONE"}

        control, receipt, preflight = _validate_ready(target, inputs)
        _require((target / "readouts_start.pt").is_file() and (target / "target_positive_control.pt").is_file(), "READY files lack measured baseline/target controls")
        if args.preflight_only:
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "checks": preflight["checks"], "next_action": "NONE"}
        control.start_execution()
        control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
        if selected is None and (target / "report.json").is_file():
            return json.loads((target / "report.json").read_text(encoding="utf-8"))
        arms = _run_training(target, inputs, evaluators, selected)
        readouts: dict[str, dict[str, Any]] = {"A4700": torch.load(target / "readouts_start.pt", map_location="cpu", weights_only=True)}
        summaries: dict[str, dict[str, Any]] = {}
        row_metrics: dict[str, dict[str, Any]] = {}
        summaries["A4700"], row_metrics["A4700"] = class9._row_and_image_metrics(readouts["A4700"], inputs["population"])
        summaries["A4700"]["endpoint"] = "A4700"
        for endpoint_name, arm in (("A5400", ARM_A), ("B5400", ARM_B)):
            summary, readout, row = _endpoint(target, endpoint_name, arms[arm], inputs, evaluators)
            summaries[endpoint_name] = summary
            readouts[endpoint_name] = readout
            row_metrics[endpoint_name] = row
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"final evaluator {name}")
        review_package = _prepare_review(target, inputs, readouts)
        control.complete()
        report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "status": "AWAITING_RATINGS",
            "classification": None,
            "source_task_status": "historical evidence only; not reclassified",
            "source_checkpoint_step": SOURCE_STEP,
            "final_generator_adam_step": FINAL_STEP,
            "additional_updates_per_arm": ADDED_UPDATES,
            "scientific_updates": 2 * ADDED_UPDATES,
            "arms": {ARM_A: {"objective": "original raw MSE reconstruction", "checkpoint": "arm_a_step5400.pt"}, ARM_B: {"objective": "raw MSE + 0.1*(1-SSIM_28x28)", "checkpoint": "arm_b_step5400.pt"}},
            "endpoint_metrics": summaries,
            "source_replay": preflight["checks"]["a4700_production_replay"],
            "preflight": preflight,
            "research_run_state": control.state,
            "review_package": review_package,
            "human_decision": None,
            "training_recommendation": "NONE",
            "next_action": "AWAIT_EXPLICIT_RATINGS_IMPORT",
            "stop": True,
            "duration_seconds_observation_only": time.perf_counter() - started,
        }
        _write_json(target / "report.json", report)
        return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="required explicit in-task checkpoint for recovery")
    parser.add_argument("--ratings", type=Path, help="explicit completed 680-row ratings CSV; CPU-only scoring mode")
    return parser.parse_args()


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _arguments()
    root = _root()
    target = _resolve(root, args.output_dir)
    if args.ratings is not None:
        return _finalize_ratings(target, _resolve(root, args.ratings))
    try:
        return _execute_training(args)
    except TrainingInterrupted as error:
        return {"task_id": TASK_ID, "status": "INTERRUPTED", "failure": str(error), "next_action": "resume using the explicit durable checkpoint"}
    except Exception as error:
        if target.is_dir() and not (target / "report.json").is_file() and not (target / "invalid_harness.json").is_file():
            checkpoint_names = sorted(path.name for path in target.glob("arm_[ab]_step*.pt"))
            failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "failure_type": type(error).__name__, "failure": str(error), "last_durable_checkpoint": checkpoint_names[-1] if checkpoint_names else None, "scientific_conclusion": None, "next_action": "NONE"}
            try:
                _write_json(target / "invalid_harness.json", failure)
            except OSError:
                pass
        raise


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
