"""Frozen, provenance-preserving LightGen architecture field comparison.

This diagnostic replays four explicit checkpoints. It does not create an
optimizer, execute backward, update parameters, sample OLS values, or rank
checkpoints. The only CUDA work is the named real-image optical forward path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.lightgen import lightgen_2d_semantic_generation_input_ols_alignment as runtime_module
from ldm_project.v2.application.research.lightgen import optics as diagnostic_optics
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.optics import d2nn
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_capacity_accumulation_comparison_v1 as capacity_task
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_foreground_objective_training_comparison_v1 as foreground


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01"
WAVELENGTH_M = 532.0e-9
PIXEL_PITCH_M = 2.0e-6
GRID = 2048
FULL_PHASE_COUNT = 340_000
OLD_PHASE_COUNT = 20_000
PHASE_LAYOUT = ("old_0mm", "new_0p75mm", "old_1p5mm", "new_2p25mm")
TRAIN_DATA = ROOT / "data/MNIST/processed/training.pt"


@dataclass(frozen=True)
class Endpoint:
    name: str
    architecture: str
    group: str
    checkpoint: Path
    population: Path
    report: Path
    replay_rtol: float
    replay_atol: float
    parent_batch: int
    parent_endpoint: str
    readout: Path | None = None


ENDPOINTS = (
    Endpoint(
        "small_A1000", "small", "main",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/arm_a_step1000.pt",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/input_population.json",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/report.json",
        2.0e-5, 2.0e-7, 10, "A_step1000",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/readouts_A_step1000.pt",
    ),
    Endpoint(
        "large_B1000", "large", "main",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/arm_b_step1000.pt",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/input_population.json",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/report.json",
        2.0e-5, 2.0e-7, 10, "B_step1000",
        ROOT / "outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/readouts_B_step1000.pt",
    ),
    Endpoint(
        "large_B5400_fg", "large", "foreground",
        ROOT / "outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/arm_b_step5400.pt",
        ROOT / "outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/input_population.json",
        ROOT / "outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/report.json",
        foreground.REPLAY_RTOL, foreground.REPLAY_ATOL, 1, "B5400",
    ),
    Endpoint(
        "large_B6700_energy", "large", "energy",
        ROOT / "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/arm_b_step6700.pt",
        ROOT / "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/input_population.json",
        ROOT / "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/report.json",
        2.0e-5, 2.0e-7, 10, "B6700",
        ROOT / "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/readouts_B6700.pt",
    ),
)


class VisualComparisonError(RuntimeError):
    """The explicit source, frozen replay or field evidence is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualComparisonError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"explicit source JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"source JSON must be an object: {path}")
    return value


def load_cpu(path: Path, *, mmap: bool = False) -> Any:
    require(path.is_file(), f"explicit tensor source is missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _population_rows(endpoint: Endpoint, population: Mapping[str, Any]) -> tuple[list[int], list[int], list[str], list[int]]:
    rows = list(map(int, population["source_row_indices"]))
    labels = list(map(int, population["labels"]))
    roles = list(map(str, population["roles"]))
    require(len(rows) == len(labels) == len(roles) and len(set(rows)) == len(rows), f"{endpoint.name}: population row identity is invalid")
    confirmation_positions = [i for i, role in enumerate(roles) if role == "confirmation"]
    require(len(confirmation_positions) == 300, f"{endpoint.name}: expected 300 confirmation rows")
    counts = [0] * 10
    selected: list[int] = []
    for pos in confirmation_positions:
        label = labels[pos]
        require(0 <= label < 10, f"{endpoint.name}: label outside 0..9")
        if counts[label] < 2:
            selected.append(pos)
            counts[label] += 1
    require(counts == [2] * 10 and len(selected) == 20, f"{endpoint.name}: first-two-per-class selection failed")
    return rows, labels, roles, selected


def _validate_checkpoint(endpoint: Endpoint, payload: Mapping[str, Any]) -> dict[str, Any]:
    expect = {
        "small_A1000": ("M0_GENERATOR_CAPACITY_ACCUMULATION_COMPARISON_V1", "A_small", 1000),
        "large_B1000": ("M0_GENERATOR_CAPACITY_ACCUMULATION_COMPARISON_V1", "B_large", 1000),
        "large_B5400_fg": ("M0_GENERATOR_FOREGROUND_OBJECTIVE_TRAINING_COMPARISON_V1", "B", 5400),
        "large_B6700_energy": ("M0_GENERATOR_ENERGY_MATCHED_TARGET_COMPARISON_V1", "B_energy_matched_target", 6700),
    }[endpoint.name]
    require(payload.get("task_id") == expect[0] and payload.get("arm") == expect[1], f"{endpoint.name}: task/arm identity differs")
    require(int(payload.get("completed_updates", -1)) == expect[2], f"{endpoint.name}: saved update count differs")
    step = int(payload.get("optimizer_step", payload.get("completed_updates", -1)))
    require(step == expect[2], f"{endpoint.name}: checkpoint Adam step differs")
    state = payload.get("trainable_parameter_state")
    require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, f"{endpoint.name}: trainable state keys differ")
    encoder = state["encoder.phase"]
    generator = state["generator.phase"]
    require(tuple(encoder.shape) == (2, 1, 100, 100) and encoder.dtype == torch.float32, f"{endpoint.name}: Encoder phase geometry/dtype differs")
    expected_generator = (2, 1, 100, 100) if endpoint.architecture == "small" else (FULL_PHASE_COUNT,)
    require(tuple(generator.shape) == expected_generator and generator.dtype == torch.float32, f"{endpoint.name}: Generator phase layout differs")
    require(bool(torch.isfinite(encoder).all() and torch.isfinite(generator).all()), f"{endpoint.name}: phase contains nonfinite values")
    optimizer = payload.get("optimizer_state")
    require(isinstance(optimizer, Mapping), f"{endpoint.name}: saved Adam metadata is absent")
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, f"{endpoint.name}: saved Adam scope differs")
    group, moment = groups[0], states[0]
    require(float(group.get("lr", -1.0)) == 1.0e-3 and tuple(group.get("betas", ())) == (0.9, 0.999), f"{endpoint.name}: Adam hyperparameters differ")
    require(float(group.get("eps", -1.0)) == 1.0e-8 and float(group.get("weight_decay", -1.0)) == 0.0, f"{endpoint.name}: Adam epsilon/decay differs")
    require(int(float(moment["step"].item())) == expect[2], f"{endpoint.name}: Adam moment step differs")
    require(tuple(moment["exp_avg"].shape) == tuple(generator.shape) and tuple(moment["exp_avg_sq"].shape) == tuple(generator.shape), f"{endpoint.name}: Adam moment shape differs")
    require(bool(torch.isfinite(moment["exp_avg"]).all() and torch.isfinite(moment["exp_avg_sq"]).all()), f"{endpoint.name}: Adam metadata is nonfinite")
    if endpoint.name == "large_B5400_fg":
        expected_source = ROOT / "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/arm_a_step4700.pt"
        require(_resolve(payload.get("source_checkpoint", "")) == expected_source.resolve(), f"{endpoint.name}: explicit source checkpoint lineage differs")
        require(int(payload.get("source_step", -1)) == 4700, f"{endpoint.name}: source checkpoint step differs")
        require(payload.get("optimizer_config", {}).get("history_reset") is False, f"{endpoint.name}: optimizer history-reset metadata differs")
    else:
        require("references" in payload and "weights" in payload, f"{endpoint.name}: frozen training identity is incomplete")
    return dict(state)


def _data_files(population: Mapping[str, Any]) -> dict[str, Path]:
    candidates = population.get("source_input_paths") or population.get("input_paths")
    require(isinstance(candidates, Mapping) and "training_data" in candidates, "population has no explicit training_data path")
    paths = {str(name): _resolve(value) for name, value in candidates.items()}
    require(paths["training_data"] == TRAIN_DATA.resolve(), "population does not name the fixed canonical training dataset")
    return paths


def _select_group_populations() -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {}
    expected_report_tasks = {
        "main": "M0_GENERATOR_CAPACITY_ACCUMULATION_COMPARISON_V1",
        "foreground": "M0_GENERATOR_FOREGROUND_OBJECTIVE_TRAINING_COMPARISON_V1",
        "energy": "M0_GENERATOR_ENERGY_MATCHED_TARGET_COMPARISON_V1",
    }
    for endpoint in ENDPOINTS:
        if endpoint.group in loaded:
            continue
        population = load_json(endpoint.population)
        report = load_json(endpoint.report)
        require(report.get("task_id") == expected_report_tasks[endpoint.group], f"{endpoint.group}: report task identity differs")
        require(report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, f"{endpoint.group}: source report is not terminal")
        require(report.get("next_action") == "NONE" and report.get("training_recommendation") == "NONE", f"{endpoint.group}: source report has an unexpected follow-up action")
        if "population" in report:
            require(report["population"] == population, f"{endpoint.group}: report and population manifests differ")
        elif endpoint.group != "foreground":
            raise VisualComparisonError(f"{endpoint.group}: report omits its population manifest")
        _data_files(population)
        rows, labels, roles, selected = _population_rows(endpoint, population)
        loaded[endpoint.group] = {"population": population, "report": report, "rows": rows, "labels": labels, "roles": roles, "selected": selected, "paths": _data_files(population)}

    main = loaded["main"]
    extra = loaded["foreground"]
    energy = loaded["energy"]
    require(extra["rows"] == energy["rows"] and extra["labels"] == energy["labels"] and extra["roles"] == energy["roles"], "B5400 and B6700 confirmation populations are not row/label/role identical")
    main_confirmation = {row for row, role in zip(main["rows"], main["roles"], strict=True) if role == "confirmation"}
    supplement_confirmation = {row for row, role in zip(extra["rows"], extra["roles"], strict=True) if role == "confirmation"}
    require(main_confirmation.isdisjoint(supplement_confirmation), "main and supplementary confirmation populations overlap")
    return loaded, {"main": main, "supplement": extra}


def _read_mnist_images(labels_needed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data_payload = diagnostic._load_torch(TRAIN_DATA, mmap=True)
    image_rows, labels = diagnostic._read_image_labels(data_payload)
    train_indices, _validation_indices = diagnostic.stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    require(bool(train_mask.index_select(0, labels_needed.to(torch.long)).all()), "selected source row is outside canonical TRAIN")
    require(labels.shape == (60_000,) and image_rows.shape[0] == 60_000, "canonical training data shape changed")
    return image_rows, labels


def _load_parent_sample(endpoint: Endpoint, selected_positions: list[int], expected_rows: list[int], expected_labels: list[int]) -> dict[str, torch.Tensor]:
    if endpoint.name == "large_B5400_fg":
        directory = endpoint.report.parent / "readouts" / endpoint.parent_endpoint
        parts: dict[str, list[torch.Tensor]] = {key: [] for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")}
        out_rows, out_labels = [], []
        positions_needed = set(selected_positions)
        for path in sorted(directory.glob("chunk_*.pt")):
            chunk = load_cpu(path)
            chunk_positions = chunk["positions"].tolist()
            take = [j for j, pos in enumerate(chunk_positions) if pos in positions_needed]
            for local in take:
                pos = int(chunk_positions[local])
                require(int(chunk["source_row_indices"][local]) == expected_rows[pos] and int(chunk["labels"][local]) == expected_labels[pos], "B5400 parent sample row/label mapping differs")
                out_rows.append(pos)
                out_labels.append(int(chunk["labels"][local]))
                for key in parts:
                    parts[key].append(chunk[key][local : local + 1].contiguous())
            del chunk
        require(set(out_rows) == positions_needed, "B5400 parent readout chunks omit selected sample rows")
        order = [out_rows.index(pos) for pos in selected_positions]
        return {key: torch.cat(values, dim=0).index_select(0, torch.tensor(order)) for key, values in parts.items()}

    if endpoint.readout is None:
        raise VisualComparisonError(f"{endpoint.name}: saved parent readout path is absent")
    readout = load_cpu(endpoint.readout, mmap=True)
    require(torch.equal(readout["source_indices"].index_select(0, torch.tensor(selected_positions)), torch.tensor([expected_rows[p] for p in selected_positions])), f"{endpoint.name}: parent readout source rows differ")
    require(torch.equal(readout["labels"].index_select(0, torch.tensor(selected_positions)), torch.tensor([expected_labels[p] for p in selected_positions])), f"{endpoint.name}: parent readout labels differ")
    return {key: readout[key].index_select(0, torch.tensor(selected_positions)).clone() for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits", "z") if key in readout}


def _model_for(endpoint: Endpoint, payload: Mapping[str, Any], state: Mapping[str, torch.Tensor], device: torch.device):
    consumer = bogt.LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=GRID)
    training_identity = payload
    if endpoint.name == "large_B5400_fg":
        training_identity = load_cpu(_resolve(payload["source_checkpoint"]))
        require(int(training_identity.get("optimizer_step", -1)) == 4700, "B5400 lineage checkpoint step differs")
        require(training_identity.get("task_id") == "M0_GENERATOR_ENERGY_MATCHED_TARGET_COMPARISON_V1", "B5400 lineage checkpoint task differs")
    refs = bogt.FrozenBogtReferences.from_payload(training_identity["references"])
    weights = bogt.BogtLossWeights.from_payload(training_identity["weights"])
    consumer.bind_references(refs, for_scientific=True)
    consumer.bind_weights(weights, for_scientific=True)
    if endpoint.architecture == "large":
        expanded = capacity.ExpandedGenerator(state["generator.phase"][:OLD_PHASE_COUNT].reshape(2, 1, 100, 100).to(device), propagation_grid_size=GRID).to(device)
        consumer.runtime.generator = expanded
    runtime_state = {name: value for name, value in payload["frozen_runtime_state"].items()}
    runtime_state.update(state)
    loaded = consumer.runtime.load_state_dict(runtime_state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, f"{endpoint.name}: production runtime state did not load completely")
    consumer.runtime.eval()
    consumer.runtime.requires_grad_(False)
    for parameter in consumer.runtime.parameters():
        parameter.grad = None
    saved_frozen = payload.get("frozen_runtime_state", {})
    actual_frozen = {name: value.detach().cpu() for name, value in consumer.runtime.state_dict().items() if name not in {"encoder.phase", "generator.phase"}}
    require(set(saved_frozen) == set(actual_frozen), f"{endpoint.name}: frozen runtime state key set differs")
    require(all(torch.equal(saved_frozen[name], actual_frozen[name]) for name in saved_frozen), f"{endpoint.name}: loaded physical buffers differ from checkpoint")
    return consumer


def _metrics_from_saved(endpoint: Endpoint, population: Mapping[str, Any], selected: list[int]) -> dict[str, torch.Tensor]:
    base = endpoint.report.parent
    if endpoint.group == "main":
        metrics = load_cpu(base / f"row_metrics_{endpoint.parent_endpoint}.pt")
        return {key: value for key, value in metrics.items() if torch.is_tensor(value)}
    if endpoint.group == "energy":
        metrics = load_cpu(base / f"row_metrics_{endpoint.parent_endpoint}.pt")
        return {key: value for key, value in metrics.items() if torch.is_tensor(value)}
    rows: list[dict[str, str]] = []
    with (base / "readouts" / endpoint.parent_endpoint / "paired_rows.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 500 and sum(row["role"] == "confirmation" for row in rows) == 300, "B5400 paired_rows.csv is incomplete")
    return {key: torch.tensor([float(row[key]) for row in rows]) for key in ("q_fg", "q_bg", "raw_mse", "production_q_28", "ssim_28", "integral_intensity", "e_d_prediction")}


def _per_class_correct(labels: list[int], roles: list[str], predictions: torch.Tensor, role: str = "confirmation") -> list[int]:
    result = [0] * 10
    counts = [0] * 10
    for i, (label, row_role) in enumerate(zip(labels, roles, strict=True)):
        if row_role == role:
            counts[label] += 1
            result[label] += int(int(predictions[i]) == label)
    require(counts == [30] * 10, "full confirmation row data does not contain 30 rows per class")
    return result


def _report_class_correct(endpoint: Endpoint) -> list[int]:
    report = load_json(endpoint.report)
    if endpoint.group == "main":
        return list(map(int, report["endpoint_metrics"][endpoint.parent_endpoint]["evaluators"]["E_D"]["by_role"]["confirmation"]["per_class_correct"]))
    if endpoint.group == "energy":
        return list(map(int, report["metrics"][endpoint.parent_endpoint]["evaluators"]["E_D"]["by_role"]["confirmation"]["per_class_correct"]))
    report_rows = report["paired_rows"]
    _ = report_rows
    path = endpoint.report.parent / "readouts" / endpoint.parent_endpoint / "paired_rows.csv"
    predictions = []
    labels = []
    roles = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            predictions.append(int(row["e_d_prediction"]))
            labels.append(int(row["label"]))
            roles.append(row["role"])
    return _per_class_correct(labels, roles, torch.tensor(predictions), "confirmation")


def _selection_record(group_data: Mapping[str, Any], endpoint: Endpoint) -> dict[str, Any]:
    return {
        "endpoint": endpoint.name,
        "checkpoint": endpoint.checkpoint.as_posix(),
        "parent_task": load_json(endpoint.population)["task_id"],
        "population_confirmation_rows": 300,
        "selected_positions": [group_data["selected"][i] for i in range(20)],
        "selected_source_rows": [group_data["rows"][i] for i in group_data["selected"]],
        "selected_labels": [group_data["labels"][i] for i in group_data["selected"]],
        "selected_roles": [group_data["roles"][i] for i in group_data["selected"]],
        "canonical_train_not_independent_generalization": True,
    }


def _prepare_sources() -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    groups, _ = _select_group_populations()
    for endpoint in ENDPOINTS:
        require(endpoint.checkpoint.is_file(), f"fixed endpoint checkpoint is missing: {endpoint.checkpoint}")
        payload = load_cpu(endpoint.checkpoint)
        _validate_checkpoint(endpoint, payload)
        group = groups[endpoint.group]
        checkpoint_population = payload.get("population", payload.get("source_population"))
        require(checkpoint_population == group["population"], f"{endpoint.name}: checkpoint source population differs")
    all_source_rows = sorted({row for group in groups.values() for row in group["rows"]})
    image_rows, all_labels = _read_mnist_images(torch.tensor(all_source_rows, dtype=torch.long))
    for group in groups.values():
        require(all_labels.index_select(0, torch.tensor(group["rows"], dtype=torch.long)).tolist() == group["labels"], f"{group['population']['task_id']}: labels do not match canonical source rows")
    return groups, {}, image_rows, all_labels


def _selected_inputs(group_data: Mapping[str, Any], image_rows: torch.Tensor, all_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    positions = list(group_data["selected"])
    rows = torch.tensor([group_data["rows"][p] for p in positions], dtype=torch.long)
    labels = torch.tensor([group_data["labels"][p] for p in positions], dtype=torch.long)
    require(torch.equal(all_labels.index_select(0, rows), labels), "selected sample labels mismatch canonical rows")
    images = image_rows.index_select(0, rows).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    require(images.shape == (20, 1, 28, 28) and bool(torch.isfinite(images).all()), "selected image tensor is invalid")
    return images, rows, labels, positions


def _forward(consumer: Any, images: torch.Tensor) -> Any:
    with torch.no_grad(), capacity.strict_fp32_mode():
        return consumer.runtime.forward_from_image(images.to(device=consumer.device, dtype=torch.float32))


def _verify_forward(endpoint: Endpoint, result: Any, expected: Mapping[str, torch.Tensor], rows_in_batch: list[int]) -> dict[str, Any]:
    observed: dict[str, torch.Tensor] = {
        "raw_cmos": result.raw_intensity.detach().cpu(),
        "target_slm_intensity": result.image_input.slm_intensity.detach().cpu(),
        "evaluator_view": canonical.cmos_view(result.raw_intensity).detach().cpu(),
        "z": result.z_ols.coefficients.detach().cpu(),
    }
    checks: dict[str, Any] = {"rtol": endpoint.replay_rtol, "atol": endpoint.replay_atol, "fields": {}}
    for key, value in observed.items():
        if key not in expected:
            continue
        offset = [int(v) for v in rows_in_batch]
        reference = expected[key].index_select(0, torch.tensor(offset))
        close = torch.allclose(value, reference, rtol=endpoint.replay_rtol, atol=endpoint.replay_atol)
        maximum = float((value - reference).abs().max().item())
        checks["fields"][key] = {"passed": bool(close), "max_abs_error": maximum, "shape": list(value.shape)}
        require(close, f"{endpoint.name}: frozen replay differs from saved parent {key}")
    return checks


def _preflight(groups: Mapping[str, Any], image_rows: torch.Tensor, all_labels: torch.Tensor) -> dict[str, Any]:
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    results: dict[str, Any] = {"status": "READY", "optimizer_updates": 0, "optimizers_created": 0, "backward_calls": 0, "new_ols_samples": 0, "endpoints": {}, "field_storage": {}}
    for endpoint in ENDPOINTS:
        group = groups[endpoint.group]
        images, source_rows, sample_labels, selected_positions = _selected_inputs(group, image_rows, all_labels)
        expected = _load_parent_sample(endpoint, selected_positions, group["rows"], group["labels"])
        payload = load_cpu(endpoint.checkpoint)
        state = _validate_checkpoint(endpoint, payload)
        consumer = _model_for(endpoint, payload, state, device)
        batch_size = endpoint.parent_batch
        first_batch = images[:batch_size]
        with torch.no_grad(), capacity.strict_fp32_mode():
            baseline = _forward(consumer, first_batch)
            checks = _verify_forward(endpoint, baseline, expected, list(range(batch_size)))
            require(baseline.raw_intensity.shape == (batch_size, 1, 200, 200), f"{endpoint.name}: raw CMOS plane shape differs")
            require(torch.allclose(baseline.raw_intensity, baseline.raw_intensity.abs(), rtol=0, atol=0), f"{endpoint.name}: CMOS intensity has negative values")
            baseline_raw = baseline.raw_intensity.detach().cpu().clone()
            baseline_ols = baseline.z_ols.coefficients.detach().cpu().clone()
        del baseline
        collector = FieldCollector.create(
            endpoint,
            source_rows[:batch_size].tolist(),
            sample_labels[:batch_size].tolist(),
        )
        original_propagation = d2nn.angular_spectrum_propagate
        original_capacity_propagation = capacity.angular_spectrum_propagate
        original_project = consumer.runtime.ols.project_state.__func__
        original_writer = consumer.runtime.ols.write_to_generator_input.__func__
        original_adapter_call = type(consumer.runtime.input_adapter).__call__
        original_relay_call = type(consumer.runtime.input_relay).__call__
        with observe_runtime(consumer, collector):
            observed = _forward(consumer, first_batch)
        observer_equal = torch.equal(observed.raw_intensity.detach().cpu(), baseline_raw) and torch.equal(observed.z_ols.coefficients.detach().cpu(), baseline_ols)
        require(observer_equal, f"{endpoint.name}: observation hooks changed production output")
        checks["observer_output_equal_to_unobserved_forward"] = {"executed": True, "passed": True, "raw_cmos_bitwise_equal": True, "ols_bitwise_equal": True}
        checks["camera_intensity_from_incident_field"] = _camera_identity(observed, collector)
        restoration_ok = (
            d2nn.angular_spectrum_propagate is original_propagation
            and capacity.angular_spectrum_propagate is original_capacity_propagation
            and consumer.runtime.ols.project_state.__func__ is original_project
            and consumer.runtime.ols.write_to_generator_input.__func__ is original_writer
            and type(consumer.runtime.input_adapter).__call__ is original_adapter_call
            and type(consumer.runtime.input_relay).__call__ is original_relay_call
            and "_expanded_phase_deviation" not in consumer.runtime.generator.__dict__
        )
        require(restoration_ok, f"{endpoint.name}: observer hooks did not restore after capture")
        checks["observer_restoration"] = {"executed": True, "passed": True}
        checks["physical_phase_multiplication"] = collector.checks["physical_phase_multiplication"]
        node_schema = {
            name: {
                "shape_per_sample": list(tensor.shape[1:]),
                "complex_dtype": str(tensor.dtype),
                "bytes_per_sample": int(tensor[0].numel() * tensor.element_size()),
            }
            for name, tensor in collector.records.items()
        }
        checks["observation_nodes"] = sorted(node_schema)
        results["field_storage"][endpoint.name] = node_schema
        results["endpoints"][endpoint.name] = checks
        del observed, collector, consumer, baseline_raw, baseline_ols
        gc.collect()
        torch.cuda.empty_cache()
    results["status"] = "READY" if all(all(item["passed"] for item in record["fields"].values()) for record in results["endpoints"].values()) else "BLOCKED"
    return results


@dataclass
class FieldCollector:
    endpoint: Endpoint
    rows: list[int]
    labels: list[int]
    roles: list[str]
    records: dict[str, torch.Tensor]
    metadata: dict[str, dict[str, Any]]
    checks: dict[str, Any]

    @classmethod
    def create(cls, endpoint: Endpoint, rows: list[int], labels: list[int]) -> "FieldCollector":
        return cls(endpoint, rows, labels, ["confirmation"] * len(rows), {}, {}, {"physical_phase_multiplication": []})

    def add(self, node: str, field: ComplexField, *, phase_position: str = "", field_kind: str = "physical_full_field") -> None:
        tensor = field.tensor.detach()
        require(torch.is_complex(tensor) and bool(torch.isfinite(tensor.real).all() and torch.isfinite(tensor.imag).all()), f"{self.endpoint.name}/{node}: complex field is invalid")
        require(field_kind == "physical_full_field", f"{self.endpoint.name}/{node}: deviation field cannot enter a physical-plane record")
        plane = field.plane
        self._add_tensor(node, tensor, {
            "plane_id": plane.plane_id,
            "plane_shape": list(plane.shape),
            "pixel_pitch_m": list(plane.pixel_pitch_m),
            "fov_m": list(plane.fov_m),
            "wavelength_m": float(plane.wavelength_m),
            "z_m": float(plane.z_m),
            "coordinate_frame": str(plane.coordinate_frame),
            "origin_m": list(plane.origin_m),
            "pixel_center_coordinate_formula": "coordinate[i] = origin + (i - (N-1)/2) * pixel_pitch",
            "phase_position": phase_position,
            "field_kind": field_kind,
        })

    def _add_tensor(self, node: str, tensor: torch.Tensor, extra: Mapping[str, Any]) -> None:
        require(node not in self.records, f"{self.endpoint.name}: duplicate observation node {node}")
        cpu = tensor.detach().cpu().contiguous()
        require(torch.is_complex(cpu) and bool(torch.isfinite(cpu.real).all() and torch.isfinite(cpu.imag).all()), f"{self.endpoint.name}/{node}: saved field is nonfinite")
        self.records[node] = cpu
        self.metadata[node] = {
            "endpoint": self.endpoint.name,
            "source_row_indices": list(self.rows),
            "labels": list(self.labels),
            "data_roles": list(self.roles),
            "node_name": node,
            "shape": list(cpu.shape),
            "complex_dtype": str(cpu.dtype),
            **dict(extra),
        }

    def add_coefficients(self, coefficients: torch.Tensor, z_m: float) -> None:
        matrix = coefficients.detach().reshape(-1, 10, 10)
        self._add_tensor("ols_coefficients_10x10", matrix, {
            "plane_id": "lightgen_ols_input",
            "plane_shape": [10, 10],
            "pixel_pitch_m": None,
            "fov_m": None,
            "wavelength_m": WAVELENGTH_M,
            "z_m": float(z_m),
            "coordinate_frame": "row_major_ols_port_order",
            "origin_m": None,
            "pixel_center_coordinate_formula": "row-major j=10*r+c; no metric pixel grid",
            "port_order": "row-major j=10*r+c",
            "phase_position": "OLS coefficients before T=I_100 transmission",
            "field_kind": "complex_ols_coefficients",
        })


@contextlib.contextmanager
def observe_runtime(consumer: Any, collector: FieldCollector) -> Iterator[None]:
    runtime = consumer.runtime
    handles = []
    patched: list[tuple[Any, str, Any, bool]] = []
    patched_classes: list[tuple[type, Any]] = []

    def patch(instance: Any, name: str, replacement: Any) -> None:
        own = name in instance.__dict__
        original_own = instance.__dict__.get(name)
        patched.append((instance, name, original_own, own))
        setattr(instance, name, replacement)

    callable_targets: dict[type, dict[int, Any]] = {}
    callable_targets.setdefault(type(runtime.input_adapter), {})[id(runtime.input_adapter)] = lambda _args, out: collector.add("slm_output_field", out.slm_field)
    callable_targets.setdefault(type(runtime.input_relay), {})[id(runtime.input_relay)] = lambda _args, out: collector.add("encoder_input_field", out)
    callable_targets.setdefault(type(runtime.output_relay), {})[id(runtime.output_relay)] = lambda args, out: (
        collector.add("cmos_relay_input_field", args[0]),
        collector.add("cmos_incident_field", out),
    )
    for cls, targets in callable_targets.items():
        original_call = cls.__call__

        def call_wrapper(instance: Any, *args: Any, __original: Any = original_call, __targets: Mapping[int, Any] = targets, **kwargs: Any):
            output = __original(instance, *args, **kwargs)
            callback = __targets.get(id(instance))
            if callback is not None:
                callback(args, output)
            return output

        patched_classes.append((cls, original_call))
        setattr(cls, "__call__", call_wrapper)

    def encoder_hook(_module: Any, _inputs: Any, output: Any) -> None:
        for i, field in enumerate(output[1], start=1):
            collector.add(f"encoder_layer_{i}_after_crop", field)

    handles.append(runtime.encoder.register_forward_hook(encoder_hook))

    def generator_pre(_module: Any, inputs: Any) -> None:
        collector.add("generator_input_field", inputs[0])

    def generator_hook(_module: Any, _inputs: Any, output: Any) -> None:
        for i, field in enumerate(output[1], start=1):
            collector.add(f"generator_layer_{i}_after_crop", field)

    handles.append(runtime.generator.register_forward_pre_hook(generator_pre))
    handles.append(runtime.generator.register_forward_hook(generator_hook))
    project = runtime.ols.project_state
    writer = runtime.ols.write_to_generator_input

    def project_wrapper(field: ComplexField):
        collector.add("ols_input_field", field)
        state = project(field)
        collector.add_coefficients(state.coefficients, runtime.ols_plane.z_m)
        return state

    def writer_wrapper(state: Any, plane: Any):
        output = writer(state, plane)
        collector.add("ols_reemission_field", output)
        return output

    patch(runtime.ols, "project_state", project_wrapper)
    patch(runtime.ols, "write_to_generator_input", writer_wrapper)

    original_d2nn = d2nn.angular_spectrum_propagate
    original_capacity = capacity.angular_spectrum_propagate

    def propagation_observer(field: ComplexField, distance_m: float, plane_id: str):
        # For the small architecture, this is a diagnostic virtual midpoint,
        # computed from the exact production-modulated input before the
        # original 1.5 mm propagation. It never replaces that propagation.
        if collector.endpoint.architecture == "small" and plane_id.endswith(("_layer_1_propagated", "_layer_2_propagated")) and str(field.plane.plane_id).startswith("lightgen_generator"):
            midpoint = original_d2nn(field, distance_m / 2.0, f"{plane_id}_virtual_midpoint")
            node = "generator_z0p75_free_reference" if plane_id.endswith("_layer_1_propagated") else "generator_z2p25_free_reference"
            collector.add(node, midpoint, phase_position="free propagation reference; no device")
        return original_d2nn(field, distance_m, plane_id)

    d2nn.angular_spectrum_propagate = propagation_observer
    capacity.angular_spectrum_propagate = propagation_observer
    try:
        if collector.endpoint.architecture == "large":
            generator = runtime.generator
            deviation = generator._expanded_phase_deviation

            def deviation_wrapper(field: ComplexField, phase_window: torch.Tensor):
                plane_index = len([name for name in collector.records if name.startswith("generator_z0p")])
                znode = "generator_z0p75" if plane_index < 2 else "generator_z2p25"
                collector.add(f"{znode}_before_phase", field, phase_position="before phase-only device")
                grid, size = GRID, int(phase_window.shape[-1])
                offset = (grid - size) // 2
                full_phase = F.pad(phase_window, (offset, offset, offset, offset))
                transmittance = torch.polar(torch.ones_like(full_phase), full_phase).to(field.tensor.dtype)
                physical_after = ComplexField(
                    field.tensor * transmittance,
                    field.plane,
                    OperationProvenance("physical_full_field_after_phase", field.plane.plane_id, None, {"outside_window_transmission": 1.0}),
                )
                collector.add(f"{znode}_after_phase", physical_after, phase_position="after phase-only device; full E*t")
                inside = (slice(offset, offset + size), slice(offset, offset + size))
                outside = transmittance.clone()
                outside[..., inside[0], inside[1]] = 1.0 + 0.0j
                require(torch.equal(outside, torch.ones_like(outside)), "expanded phase transmittance is not identity outside its 400x400 aperture")
                expected_after = field.tensor * transmittance
                require(torch.equal(physical_after.tensor, expected_after), "expanded phase field is not the actual full E*t")
                collector.checks["physical_phase_multiplication"].append({"node": znode, "executed": True, "passed": True, "outside_aperture_identity": True, "field_kind": "physical_full_field"})
                return deviation(field, phase_window)

            patch(generator, "_expanded_phase_deviation", deviation_wrapper)
        yield
    finally:
        d2nn.angular_spectrum_propagate = original_d2nn
        capacity.angular_spectrum_propagate = original_capacity
        for handle in handles:
            handle.remove()
        for instance, name, original_own, had_own in reversed(patched):
            if had_own:
                setattr(instance, name, original_own)
            else:
                delattr(instance, name)
        for cls, original_call in reversed(patched_classes):
            setattr(cls, "__call__", original_call)


def _camera_identity(result: Any, collector: FieldCollector) -> dict[str, Any]:
    field = collector.records.get("cmos_incident_field")
    require(field is not None, f"{collector.endpoint.name}: CMOS incident complex field was not captured")
    intensity = field.abs().square()
    actual = result.raw_intensity.detach().cpu()
    error = float((intensity - actual).abs().max().item())
    passed = torch.allclose(intensity, actual, rtol=0.0, atol=5.0e-7)
    require(passed, f"{collector.endpoint.name}: I != |E_CMOS|^2 within FP32 roundoff")
    return {"executed": True, "passed": True, "max_abs_error": error, "rtol": 0.0, "atol": 5.0e-7, "comparison_note": "CPU recomputation versus FP32 CUDA intensity; tolerance covers observed low-order roundoff"}


def _save_field_chunk(path: Path, collector: FieldCollector, positions: list[int]) -> dict[str, Any]:
    payload = {
        "schema": "lightgen_frozen_complex_field_chunk.v1",
        "endpoint": collector.endpoint.name,
        "positions": torch.tensor(positions, dtype=torch.long),
        "source_row_indices": torch.tensor(collector.rows, dtype=torch.long),
        "labels": torch.tensor(collector.labels, dtype=torch.long),
        "data_roles": list(collector.roles),
        "phase_order": list(PHASE_LAYOUT),
        "fields": collector.records,
        "field_metadata": collector.metadata,
    }
    require(list(payload["phase_order"]) == list(PHASE_LAYOUT), "field phase storage order is invalid")
    for node, meta in payload["field_metadata"].items():
        _validate_field_record(node, payload["fields"][node], meta, collector.rows, collector.labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite field chunk: {path}")
    torch.save(payload, path)
    return {"path": path.relative_to(OUTPUT).as_posix(), "rows": collector.rows, "nodes": sorted(collector.records)}


def _validate_field_record(node: str, tensor: torch.Tensor, metadata: Mapping[str, Any], rows: list[int], labels: list[int]) -> None:
    require(metadata.get("node_name") == node, f"field node name mismatch: {node}")
    require(metadata.get("source_row_indices") == rows and metadata.get("labels") == labels, f"{node}: source row/label binding differs")
    require(metadata.get("data_roles") == ["confirmation"] * len(rows), f"{node}: data role is not canonical confirmation")
    require(metadata.get("coordinate_frame") and metadata.get("plane_id") and metadata.get("z_m") is not None, f"{node}: physical coordinates are missing")
    require(metadata.get("field_kind") in {"physical_full_field", "complex_ols_coefficients"}, f"{node}: invalid complex-field kind")
    if metadata.get("field_kind") == "physical_full_field":
        require(metadata.get("pixel_pitch_m") is not None and len(metadata.get("pixel_pitch_m", [])) == 2, f"{node}: physical pixel coordinates are missing")
        require(metadata.get("plane_shape") == list(tensor.shape[-2:]), f"{node}: physical coordinate shape differs from field")
    else:
        require(metadata.get("coordinate_frame") == "row_major_ols_port_order" and metadata.get("port_order"), f"{node}: OLS port coordinates are missing")
    require(torch.is_complex(tensor) and list(tensor.shape) == metadata.get("shape"), f"{node}: tensor shape or complex dtype differs from metadata")
    require(str(tensor.dtype) == metadata.get("complex_dtype") and bool(torch.isfinite(tensor.real).all() and torch.isfinite(tensor.imag).all()), f"{node}: complex field dtype/values invalid")


def _load_field_chunk(path: Path, *, expected_endpoint: str | None = None) -> dict[str, Any]:
    payload = load_cpu(path)
    require(payload.get("schema") == "lightgen_frozen_complex_field_chunk.v1", "saved complex field schema differs")
    if expected_endpoint is not None:
        require(payload.get("endpoint") == expected_endpoint, "saved complex field endpoint differs")
    require(payload.get("phase_order") == list(PHASE_LAYOUT), "saved phase storage order is incorrect")
    rows = payload["source_row_indices"].tolist()
    labels = payload["labels"].tolist()
    require(len(rows) == len(labels) == len(payload["data_roles"]), "saved field row metadata length differs")
    require(payload["data_roles"] == ["confirmation"] * len(rows), "saved field rows left confirmation role")
    for node, tensor in payload["fields"].items():
        _validate_field_record(node, tensor, payload["field_metadata"][node], rows, labels)
    return payload


def _capture_endpoint(endpoint: Endpoint, group: Mapping[str, Any], image_rows: torch.Tensor, all_labels: torch.Tensor) -> dict[str, Any]:
    images, sample_rows, sample_labels, selected_positions = _selected_inputs(group, image_rows, all_labels)
    payload = load_cpu(endpoint.checkpoint)
    state = _validate_checkpoint(endpoint, payload)
    device = torch.device("cuda:0")
    consumer = _model_for(endpoint, payload, state, device)
    parameter_before = {name: item.detach().cpu().clone() for name, item in consumer.runtime.named_parameters()}
    buffer_before = {name: item.detach().cpu().clone() for name, item in consumer.runtime.named_buffers()}
    saved = _load_parent_sample(endpoint, selected_positions, group["rows"], group["labels"])
    chunk_records = []
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, 20, endpoint.parent_batch)):
        stop = min(start + endpoint.parent_batch, 20)
        batch = images[start:stop]
        positions = selected_positions[start:stop]
        rows = sample_rows[start:stop].tolist()
        labels = sample_labels[start:stop].tolist()
        baseline = _forward(consumer, batch)
        baseline_raw = baseline.raw_intensity.detach().cpu().clone()
        baseline_z = baseline.z_ols.coefficients.detach().cpu().clone()
        del baseline
        collector = FieldCollector.create(endpoint, rows, labels)
        with observe_runtime(consumer, collector):
            observed = _forward(consumer, batch)
        require(torch.equal(observed.raw_intensity.detach().cpu(), baseline_raw), f"{endpoint.name}: observer changed production CMOS output")
        require(torch.equal(observed.z_ols.coefficients.detach().cpu(), baseline_z), f"{endpoint.name}: observer changed OLS coefficients")
        local_expected = {key: value[start:stop] for key, value in saved.items()}
        _verify_forward(endpoint, observed, local_expected, list(range(stop - start)))
        camera_check = _camera_identity(observed, collector)
        require(torch.equal(parameter_before["generator.phase"], dict(consumer.runtime.named_parameters())["generator.phase"].detach().cpu()), f"{endpoint.name}: Generator parameters changed during frozen replay")
        require(torch.equal(parameter_before["encoder.phase"], dict(consumer.runtime.named_parameters())["encoder.phase"].detach().cpu()), f"{endpoint.name}: Encoder parameters changed during frozen replay")
        current_buffers = dict(consumer.runtime.named_buffers())
        require(set(buffer_before) == set(current_buffers) and all(torch.equal(buffer_before[k], current_buffers[k].detach().cpu()) for k in buffer_before), f"{endpoint.name}: runtime buffers changed during observation")
        require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), f"{endpoint.name}: parameter gradients were populated")
        collector.checks["observer_output_equal_to_unobserved_forward"] = {"executed": True, "passed": True, "rows": len(rows), "raw_cmos_bitwise_equal": True, "ols_bitwise_equal": True}
        collector.checks["cmos_intensity_from_incident_field"] = camera_check
        chunk_path = OUTPUT / "fields" / endpoint.name / f"field_chunk_{batch_index:03d}.pt"
        chunk_records.append(_save_field_chunk(chunk_path, collector, positions))
        del observed, collector, baseline_raw, baseline_z
        gc.collect()
        torch.cuda.empty_cache()
    del consumer
    gc.collect()
    torch.cuda.empty_cache()
    return {"endpoint": endpoint.name, "batch_size": endpoint.parent_batch, "chunks": chunk_records, "elapsed_seconds_observation_only": time.perf_counter() - started, "optimizer_updates": 0}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite evidence file: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _estimate_disk(preflight: Mapping[str, Any]) -> dict[str, int]:
    field_bytes = 0
    per_endpoint: dict[str, int] = {}
    for endpoint in ENDPOINTS:
        schema = preflight["field_storage"][endpoint.name]
        endpoint_bytes = sum(int(node["bytes_per_sample"]) * 20 for node in schema.values())
        per_endpoint[endpoint.name] = endpoint_bytes
        field_bytes += endpoint_bytes
    # Tensor chunks are written without compression; allow 20% for metadata,
    # duplicate views and serialization overhead before checking the reserve.
    field_bytes_with_margin = math.ceil(field_bytes * 1.2)
    expected = field_bytes_with_margin + 2_000_000_000
    import shutil
    usage = shutil.disk_usage(ROOT)
    require(usage.free > expected + 20 * 1024**3, f"D: free space is below estimate plus 20 GiB reserve ({usage.free} bytes free)")
    return {"field_chunk_bytes_from_preflight_schema": field_bytes, "field_chunk_bytes_with_20pct_margin": field_bytes_with_margin, "per_endpoint_field_bytes": per_endpoint, "other_outputs_allowance_bytes": 2_000_000_000, "estimated_total_bytes": expected, "d_free_bytes_before_run": usage.free, "reserve_required_bytes": 20 * 1024**3}


def capture() -> dict[str, Any]:
    require(not OUTPUT.exists(), f"refusing to overwrite existing output directory: {OUTPUT}")
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA device 0 is unavailable")
    groups, _, image_rows, all_labels = _prepare_sources()
    preflight = _preflight(groups, image_rows, all_labels)
    require(preflight["status"] == "READY", "no-update production preflight failed; field capture stopped")
    disk = _estimate_disk(preflight)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "fields").mkdir()
    _write_json(OUTPUT / "preflight.json", preflight)
    _write_json(OUTPUT / "disk_estimate.json", disk)
    selection = [_selection_record(groups[endpoint.group], endpoint) for endpoint in ENDPOINTS]
    _write_json(OUTPUT / "sample_selection.json", {"groups": {name: value["population"] for name, value in groups.items()}, "endpoints": selection, "populations_disjoint": True, "selected_per_class": 2, "selected_rows_per_endpoint": 20})
    records = [_capture_endpoint(endpoint, groups[endpoint.group], image_rows, all_labels) for endpoint in ENDPOINTS]
    _write_json(OUTPUT / "field_capture_manifest.json", {"schema": "lightgen_architecture_field_capture.v1", "endpoints": records, "physical_constants": {"wavelength_m": WAVELENGTH_M, "pixel_pitch_m": PIXEL_PITCH_M, "propagation_grid": [GRID, GRID]}, "coordinate_convention": {"origin_m": [0.0, 0.0], "origin_semantics": "center of pixel array; PlaneSpec default origin=(0,0)", "pixel_center_formula": "coordinate[i] = origin + (i - (N-1)/2) * pixel_pitch"}, "training_updates": 0, "optimizer_instances": 0, "backward_calls": 0, "new_ols_samples": 0, "role": "canonical TRAIN confirmation rows; not independent generalization"})
    return {"status": "CAPTURED", "output_dir": OUTPUT.as_posix(), "disk_estimate": disk, "endpoints": records}


def preflight_only() -> dict[str, Any]:
    require(not OUTPUT.exists(), f"refusing to overwrite existing output directory: {OUTPUT}")
    groups, _, image_rows, all_labels = _prepare_sources()
    result = _preflight(groups, image_rows, all_labels)
    disk = _estimate_disk(result)
    result["disk_estimate"] = disk
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    result = preflight_only() if arguments.preflight_only else capture()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
