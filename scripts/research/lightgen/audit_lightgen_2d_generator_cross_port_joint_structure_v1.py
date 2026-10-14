"""Test whether the frozen A4700 readout depends on class-conditional OLS port pairing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from scripts.research.lightgen import (
    execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as parent_task,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation,
)

TASK_ID = "M0_GENERATOR_CROSS_PORT_JOINT_STRUCTURE_V1"
SOURCE_DIR = Path(
    "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03"
)
DEFAULT_CHECKPOINT = SOURCE_DIR / "arm_a_step4700.pt"
DEFAULT_OUTPUT = Path(
    "outputs/20261009_m0_generator_cross_port_joint_structure_v1/runs/attempt_01"
)
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
PORTS = 100
CLASSES = tuple(range(10))
SHUFFLE_SEEDS = tuple(range(2026100931, 2026100941))
WHOLE_ROW_SEED = 2026100941
PHASE_SEED = 2026100942
SCALE_FACTOR = 2.0
FORMAL_SCENARIOS = ("complete_joint", "whole_row_control") + tuple(
    f"port_shuffle_{seed}" for seed in SHUFFLE_SEEDS
)
CHUNK_ROWS = 50
PREFLIGHT_RTOL = 2.0e-6
PREFLIGHT_ATOL = 5.0e-8
TOLERANCE_ATOL = 2.0e-7
TOLERANCE_RTOL = 2.0e-5


class JointStructureError(RuntimeError):
    """A source, measurement, or output contract was not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointStructureError(message)


def _json_read(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required evidence file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"evidence must be a JSON object: {path.name}")
    return value


def _json_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_write_new(path: Path, payload: Any) -> None:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _prepare_output(root: Path, output_dir: Path) -> Path:
    output_root = (root / "outputs").resolve()
    target = output_dir.resolve()
    _require(target != output_root and output_root in target.parents, "output must be a child of outputs/")
    parts = target.relative_to(output_root).parts
    _require(
        len(parts) == 3
        and parts[0].startswith("20261009_")
        and parts[1] == "runs"
        and parts[2].startswith("attempt_"),
        "output must use outputs/YYYYMMDD_<version>/runs/attempt_N",
    )
    _require(not any(item in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for item in parts), "protected output namespace is forbidden")
    _require(not target.exists(), f"output already exists; refusing to overwrite: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _validate_checkpoint(checkpoint: Mapping[str, Any], checkpoint_path: Path, source_population: Mapping[str, Any]) -> None:
    _require(checkpoint_path.name == "arm_a_step4700.pt", "explicit checkpoint is not the named A4700 checkpoint")
    _require(checkpoint.get("task_id") == parent_task.TASK_ID, "checkpoint task identity differs from the energy-comparison endpoint")
    _require(checkpoint.get("arm") == parent_task.ARM_A, "checkpoint is not the energy-comparison A arm")
    _require(checkpoint.get("completed_updates") == 4700 and checkpoint.get("optimizer_step") == 4700, "A4700 checkpoint step differs")
    _require(checkpoint.get("task_local_updates") == 0 and checkpoint.get("source_history_count") == 4700, "A4700 endpoint has unexpected local updates or source history")
    _require(checkpoint.get("next_batch_index") == 0 and checkpoint.get("next_batch_cursor") == 0, "A4700 checkpoint is not at a batch boundary")
    _require(checkpoint.get("optimizer_parameter_names") == ["generator.phase"], "checkpoint optimizer scope differs")
    _require(checkpoint.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "checkpoint architecture differs")
    trainable = checkpoint.get("trainable_parameter_state")
    _require(isinstance(trainable, Mapping), "checkpoint trainable state is missing")
    _require(tuple(trainable["encoder.phase"].shape) == (2, 1, 100, 100), "checkpoint encoder shape differs")
    _require(tuple(trainable["generator.phase"].shape) == (340_000,), "checkpoint generator shape differs")
    _require(all(bool(torch.isfinite(value).all()) for value in trainable.values()), "checkpoint trainable state is nonfinite")
    frozen = checkpoint.get("frozen_runtime_state")
    _require(
        isinstance(frozen, Mapping)
        and set(frozen) == {"ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"},
        "checkpoint frozen OLS state differs",
    )
    _require(all(bool(torch.isfinite(value.real if value.is_complex() else value).all()) for value in frozen.values()), "checkpoint frozen OLS state is nonfinite")
    optimizer = checkpoint.get("optimizer_state")
    _require(isinstance(optimizer, Mapping) and set(optimizer.get("state", {})) == {0}, "checkpoint Adam state is missing")
    slot = optimizer["state"][0]
    _require(int(float(slot["step"].item())) == 4700, "checkpoint Adam step differs")
    _require(tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "checkpoint Adam moments differ")
    _require(bool(torch.isfinite(slot["exp_avg"]).all() and torch.isfinite(slot["exp_avg_sq"]).all()), "checkpoint Adam moments are nonfinite")
    _require(checkpoint.get("population", {}).get("source_row_indices") == source_population.get("source_row_indices"), "checkpoint TRAIN row lineage differs from source population")
    _require(checkpoint.get("population", {}).get("labels") == source_population.get("labels"), "checkpoint TRAIN label lineage differs from source population")


def _validate_readout(readout: Mapping[str, Any], population: Mapping[str, Any]) -> None:
    expected_rows = torch.tensor(population["source_row_indices"], dtype=torch.long)
    expected_labels = torch.tensor(population["labels"], dtype=torch.long)
    _require(torch.equal(readout.get("source_indices"), expected_rows), "saved A4700 source row IDs differ from the population")
    _require(torch.equal(readout.get("labels"), expected_labels), "saved A4700 labels differ from the population")
    _require(tuple(readout.get("z", torch.empty(0)).shape) == (TOTAL_ROWS, PORTS) and readout["z"].is_complex(), "saved OLS bank geometry/type differs")
    _require(tuple(readout.get("raw_cmos", torch.empty(0)).shape) == (TOTAL_ROWS, 1, 200, 200), "saved A4700 raw CMOS geometry differs")
    _require(tuple(readout.get("target_slm_intensity", torch.empty(0)).shape) == (TOTAL_ROWS, 1, 200, 200), "saved target image geometry differs")
    _require(tuple(readout.get("evaluator_view", torch.empty(0)).shape) == (TOTAL_ROWS, 1, 28, 28), "saved evaluator view geometry differs")
    _require(tuple(readout.get("e_d_logits", torch.empty(0)).shape) == (TOTAL_ROWS, 10), "saved E_D logits geometry differs")
    for field in ("z", "raw_cmos", "target_slm_intensity", "evaluator_view", "e_d_logits"):
        values = readout[field]
        _require(bool(torch.isfinite(values.real if values.is_complex() else values).all()), f"saved {field} contains nonfinite values")
    _require(bool((readout["raw_cmos"] >= 0.0).all()), "saved raw CMOS includes negative intensity")
    _require(bool((readout["target_slm_intensity"] >= 0.0).all()), "saved targets include negative intensity")
    _require(bool((readout["z"].abs().flatten(1).sum(1) > 0.0).all()), "saved OLS includes a zero-mass row")
    _require(bool((readout["raw_cmos"].flatten(1).sum(1) > 0.0).all()), "saved raw CMOS includes a zero-mass row")
    _require(bool((readout["target_slm_intensity"].flatten(1).sum(1) > 0.0).all()), "saved target includes a zero-mass row")
    replayed_view = generation.capacity.canonical.cmos_view(readout["raw_cmos"])
    _require(torch.allclose(replayed_view, readout["evaluator_view"], rtol=PREFLIGHT_RTOL, atol=PREFLIGHT_ATOL), "saved production view differs from raw CMOS")


def _load_inputs(root: Path, artifact_dir: Path, checkpoint_arg: Path) -> dict[str, Any]:
    _require(artifact_dir.is_dir(), f"source attempt is missing: {artifact_dir}")
    population = _json_read(artifact_dir / "input_population.json")
    parent_report = _json_read(artifact_dir / "report.json")
    _require(population.get("task_id") == parent_task.TASK_ID, "source population task identity differs")
    _require(parent_report.get("task_id") == parent_task.TASK_ID, "source report task identity differs")
    _require(parent_report.get("status") == "VALID_COMPLETED", "source energy comparison is not a complete saved measurement")
    _require(len(population.get("source_row_indices", [])) == TOTAL_ROWS, "source population does not contain exactly 500 rows")
    _require(len(set(population["source_row_indices"])) == TOTAL_ROWS, "source population row IDs are duplicated")
    _require(len(population.get("labels", [])) == TOTAL_ROWS, "source population labels do not contain 500 rows")
    _require(population.get("role_positions") == {"fit": [0, 200], "confirmation": [200, 500]}, "source population role boundaries differ")
    roles = population.get("roles")
    _require(roles == ["fit"] * FIT_ROWS + ["confirmation"] * CONFIRM_ROWS, "source population roles differ")
    labels = torch.tensor(population["labels"], dtype=torch.long)
    _require(torch.equal(torch.bincount(labels[:FIT_ROWS], minlength=10), torch.full((10,), 20)), "fit population is not 20 per class")
    _require(torch.equal(torch.bincount(labels[FIT_ROWS:], minlength=10), torch.full((10,), 30)), "confirmation population is not 30 per class")
    _require(bool(((labels >= 0) & (labels <= 9)).all()), "population labels are outside 0–9")

    checkpoint_path = _resolve(root, checkpoint_arg)
    expected_checkpoint = (artifact_dir / "arm_a_step4700.pt").resolve()
    _require(checkpoint_path == expected_checkpoint, "explicit checkpoint must be the A4700 artifact saved with the named parent attempt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    stroke_dir = _resolve(root, parent_task.DEFAULT_STROKE)
    parent_inputs = parent_task._load_population(root, stroke_dir)
    reconstructed_population = parent_inputs["population"]
    normalized_population = parent_task._normalize_population_paths(root, population)
    for field in (
        "source_row_indices",
        "labels",
        "roles",
        "source_checkpoint",
        "source_a4700_readout",
        "source_generation_dir",
        "source_population_positions",
    ):
        _require(reconstructed_population.get(field) == normalized_population.get(field), f"parent reconstruction differs for {field}")
    reference_checkpoint = parent_inputs["source"]
    _validate_checkpoint(checkpoint, checkpoint_path, population)
    _require(checkpoint.get("source_checkpoint") == population.get("source_checkpoint"), "A4700 copy does not name the source checkpoint declared by the parent population")
    _require(checkpoint.get("numeric_mode") == population.get("numeric_mode"), "A4700 checkpoint numeric mode differs from the saved population")
    for field in ("trainable_parameter_state", "frozen_runtime_state"):
        actual, expected = checkpoint[field], reference_checkpoint[field]
        _require(set(actual) == set(expected) and all(torch.equal(actual[name], expected[name]) for name in actual), f"A4700 copied {field} differs from the source checkpoint")

    readout = parent_task.diagnostic._load_torch(artifact_dir / "readouts_start.pt", mmap=True)
    _validate_readout(readout, population)
    stroke_readout = parent_inputs["readout"]
    for field in ("source_indices", "labels"):
        _require(torch.equal(readout[field], stroke_readout[field]), f"attempt_03 {field} differs from the fixed stroke source readout")
    for field in ("z", "raw_cmos", "target_slm_intensity", "evaluator_view", "e_d_logits"):
        _require(torch.allclose(readout[field], stroke_readout[field], rtol=PREFLIGHT_RTOL, atol=PREFLIGHT_ATOL), f"attempt_03 {field} differs from the fixed stroke source readout")
    _require(population.get("source_population_positions") and len(population["source_population_positions"]) == TOTAL_ROWS, "source population positions are missing")
    positions = torch.tensor(population["source_population_positions"], dtype=torch.long)
    _require(int(positions.min()) >= 0 and int(positions.max()) < 3_300 and torch.unique(positions).numel() == TOTAL_ROWS, "source population positions are invalid")

    source_readout_path = parent_inputs["source_readout_path"]
    source_readout = parent_task.diagnostic._load_torch(source_readout_path, mmap=True)
    expected_rows = torch.tensor(population["source_row_indices"], dtype=torch.long)
    _require(torch.equal(source_readout["source_indices"].index_select(0, positions), expected_rows), "source A4700 saved TRAIN identities differ")
    _require(torch.equal(source_readout["labels"].index_select(0, positions), labels), "source A4700 saved labels differ")
    for field in ("z", "target_slm_intensity", "raw_cmos"):
        source_values = source_readout[field].index_select(0, positions)
        _require(torch.allclose(source_values, readout[field], rtol=PREFLIGHT_RTOL, atol=PREFLIGHT_ATOL), f"parent A4700 {field} differs from attempt_03")
    _require(readout["plane_metadata"] == source_readout["plane_metadata"], "A4700 saved physical plane coordinates differ from the source readout")
    parent_positions = torch.tensor(population["source_population_positions"], dtype=torch.long)
    _require(torch.equal(positions, parent_positions), "source position mapping changed during lineage validation")

    pixel_area, pixel_pitch = parent_task._pixel_area(readout, "camera_plane")
    return {
        "population": population,
        "labels": labels,
        "roles": roles,
        "positions": positions,
        "readout": readout,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "generation_inputs": parent_inputs,
        "generation_paths": parent_inputs["evaluator_paths"],
        "generation_evidence": {
            "checkpoint_path": checkpoint_path,
            "source_checkpoint_path": _resolve(root, population["source_checkpoint"]),
            "source_readout_path": source_readout_path,
        },
        "source_readout": source_readout,
        "parent_report": parent_report,
        "pixel_area_m2": pixel_area,
        "camera_pixel_pitch_m": pixel_pitch,
        "artifact_dir": artifact_dir,
    }


def _permutation_map(labels: torch.Tensor, roles: list[str], *, seed: int, portwise: bool) -> torch.Tensor:
    _require(labels.shape == (TOTAL_ROWS,) and len(roles) == TOTAL_ROWS, "permutation population geometry differs")
    generator_cpu = torch.Generator(device="cpu").manual_seed(seed)
    mapping = torch.arange(TOTAL_ROWS, dtype=torch.long).unsqueeze(1).expand(-1, PORTS).clone()
    role_tensor = torch.tensor([0 if role == "fit" else 1 for role in roles], dtype=torch.long)
    for digit in CLASSES:
        for role_id in (0, 1):
            carriers = ((labels == digit) & (role_tensor == role_id)).nonzero().flatten()
            _require(carriers.numel() == (20 if role_id == 0 else 30), "class/role row count changed during permutation")
            if portwise:
                random_keys = torch.rand((PORTS, carriers.numel()), generator=generator_cpu, dtype=torch.float64)
                orders = random_keys.argsort(dim=1)
                mapping[carriers[:, None], torch.arange(PORTS)] = carriers[orders.T]
            else:
                donor_order = carriers[torch.randperm(carriers.numel(), generator=generator_cpu)]
                mapping[carriers] = donor_order.unsqueeze(1).expand(-1, PORTS)
    _validate_permutation_map(mapping, labels, role_tensor, portwise=portwise)
    return mapping


def _validate_permutation_map(
    mapping: torch.Tensor,
    labels: torch.Tensor,
    role_ids: torch.Tensor,
    *,
    portwise: bool,
) -> None:
    _require(mapping.dtype == torch.long and tuple(mapping.shape) == (TOTAL_ROWS, PORTS), "permutation map must be int64[500,100]")
    _require(bool(((mapping >= 0) & (mapping < TOTAL_ROWS)).all()), "permutation map contains an out-of-range source row")
    for digit in CLASSES:
        for role_id, expected_count in ((0, 20), (1, 30)):
            carriers = ((labels == digit) & (role_ids == role_id)).nonzero().flatten()
            donors = mapping.index_select(0, carriers)
            _require(bool((labels.index_select(0, donors.flatten()) == digit).all()), "port shuffle crossed a class boundary")
            _require(bool((role_ids.index_select(0, donors.flatten()) == role_id).all()), "port shuffle crossed a data role")
            _require(carriers.numel() == expected_count, "class/role population size changed")
            for port in range(PORTS):
                _require(torch.unique(donors[:, port]).numel() == expected_count, "per-port source mapping reused or omitted a source row")
            if not portwise:
                _require(bool((donors == donors[:, :1]).all()), "whole-row control does not share one donor across ports")


def _apply_map(z: torch.Tensor, mapping: torch.Tensor, labels: torch.Tensor, roles: list[str]) -> torch.Tensor:
    _require(tuple(z.shape) == (TOTAL_ROWS, PORTS) and z.is_complex(), "OLS input must be complex[500,100]")
    _require(bool(torch.isfinite(z.real).all() and torch.isfinite(z.imag).all()), "OLS input contains nonfinite coefficients")
    role_ids = torch.tensor([0 if role == "fit" else 1 for role in roles], dtype=torch.long)
    _validate_permutation_map(mapping, labels, role_ids, portwise=True)
    shuffled = z.gather(0, mapping)
    _require(bool(torch.isfinite(shuffled.real).all() and torch.isfinite(shuffled.imag).all()), "permuted OLS contains nonfinite coefficients")
    return shuffled.contiguous()


def _shape_vectors(raw: torch.Tensor, size: int) -> torch.Tensor:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "shape input must be N×1×200×200")
    values = raw.to(torch.float64)
    _require(bool(torch.isfinite(values).all() and (values >= 0.0).all()), "shape image contains invalid intensity")
    pooled = F.adaptive_avg_pool2d(values, (size, size)).flatten(1)
    mass = pooled.sum(1)
    _require(bool(torch.isfinite(mass).all() and (mass > 0.0).all()), "shape image has zero or invalid mass")
    return pooled / mass[:, None]


def _mean_euclidean_distance(left: torch.Tensor, right: torch.Tensor, *, row_chunk: int = 30) -> float:
    _require(left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1], "distance inputs must be compatible matrices")
    _require(left.shape[0] > 0 and right.shape[0] > 0, "distance inputs cannot be empty")
    left64, right64 = left.to(torch.float64), right.to(torch.float64)
    compute_mode = "use_mm_for_euclid_dist" if left64.shape[1] > 4096 else "donot_use_mm_for_euclid_dist"
    if left64.shape == right64.shape and torch.equal(left64, right64):
        distances = torch.cdist(left64, right64, p=2, compute_mode=compute_mode)
        _require(bool(torch.isfinite(distances).all()), "Euclidean distance matrix contains nonfinite values")
        return float(distances.mean())
    partials = []
    for start in range(0, left64.shape[0], row_chunk):
        distances = torch.cdist(left64[start : start + row_chunk], right64, p=2, compute_mode=compute_mode)
        _require(bool(torch.isfinite(distances).all()), "Euclidean distance matrix contains nonfinite values")
        partials.append(float(distances.sum()))
    return math.fsum(partials) / (left64.shape[0] * right64.shape[0])


def _energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    cross = _mean_euclidean_distance(x, y)
    within_x = _mean_euclidean_distance(x, x)
    within_y = _mean_euclidean_distance(y, y)
    value = 2.0 * cross - within_x - within_y
    _require(math.isfinite(value), "energy distance is nonfinite")
    _require(value >= -1.0e-10, "V-statistic energy distance is materially negative")
    return max(0.0, value)


def _batched_energy_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _require(x.ndim == y.ndim == 3 and x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2], "batched shape groups are incompatible")
    compute_mode = "use_mm_for_euclid_dist" if x.shape[2] > 4096 else "donot_use_mm_for_euclid_dist"
    cross = torch.cdist(x, y, p=2, compute_mode=compute_mode).mean(dim=(1, 2))
    within_x = torch.cdist(x, x, p=2, compute_mode=compute_mode).mean(dim=(1, 2))
    within_y = torch.cdist(y, y, p=2, compute_mode=compute_mode).mean(dim=(1, 2))
    values = 2.0 * cross - within_x - within_y
    _require(bool(torch.isfinite(values).all()), "batched energy distance contains nonfinite values")
    _require(bool((values >= -1.0e-10).all()), "batched V-statistic energy distance is materially negative")
    return values.clamp_min(0.0)


def _class_distances(
    raw: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    roles: list[str],
    *,
    scenario: str,
    seed: int | None,
    shape_size: int,
) -> list[dict[str, Any]]:
    generated_shape = _shape_vectors(raw, shape_size)
    target_shape = _shape_vectors(targets, shape_size)
    rows: list[dict[str, Any]] = []
    for role, bounds in (("fit", (0, FIT_ROWS)), ("confirmation", (FIT_ROWS, TOTAL_ROWS))):
        start, stop = bounds
        class_indices = [((labels[start:stop] == digit).nonzero().flatten() + start) for digit in CLASSES]
        expected_count = 20 if role == "fit" else 30
        _require(all(indices.numel() == expected_count for indices in class_indices), f"{role} class population row count changed")
        generated_by_class = torch.stack([generated_shape.index_select(0, indices) for indices in class_indices])
        target_by_class = torch.stack([target_shape.index_select(0, indices) for indices in class_indices])
        distances = _batched_energy_distances(generated_by_class, target_by_class)
        for digit, distance in zip(CLASSES, distances.tolist(), strict=True):
            rows.append({"scenario": scenario, "seed": seed, "role": role, "label": digit, "rows": expected_count, "shape_size": shape_size, "energy_distance_squared": distance})
    return rows


def _summary_by_scenario(class_rows: list[dict[str, Any]], role: str, shape_size: int) -> dict[str, float]:
    selected = [row for row in class_rows if row["role"] == role and row["shape_size"] == shape_size]
    _require(len(selected) == len(FORMAL_SCENARIOS) * len(CLASSES), "class-distance table is incomplete")
    result = {}
    for scenario in FORMAL_SCENARIOS:
        values = [float(row["energy_distance_squared"]) for row in selected if row["scenario"] == scenario]
        _require(len(values) == len(CLASSES), f"scenario {scenario} is incomplete")
        result[scenario] = math.fsum(values) / len(values)
    return result


def _classify_difference(difference: float, joint: float, shuffled: float) -> dict[str, Any]:
    _require(all(math.isfinite(item) for item in (difference, joint, shuffled)), "primary decision contains nonfinite values")
    tolerance = TOLERANCE_ATOL + TOLERANCE_RTOL * max(abs(joint), abs(shuffled))
    if difference > tolerance:
        classification = "JOINT_PAIRING_SHAPE_SUPPORT"
        interpretation = "指定端口重组提高了同类形态分布差异；当前A4700端点获得联合配对贡献的局部数值支持。"
    elif difference < -tolerance:
        classification = "JOINT_PAIRING_NOT_REQUIRED_FOR_THIS_MEASURE"
        interpretation = "端口重组降低了同类形态分布差异；当前结果削弱经验联合配对必须保留的解释。"
    else:
        classification = "JOINT_PAIRING_EFFECT_WITHIN_NUMERICAL_TOLERANCE"
        interpretation = "平均差异未超出预设数值容差；当前测量对联合配对影响未作区分。"
    return {
        "classification": classification,
        "difference_shuffled_minus_joint": difference,
        "tolerance": tolerance,
        "complete_joint_mean": joint,
        "port_shuffle_mean_over_fixed_seeds": shuffled,
        "interpretation": interpretation,
    }


def _primary_decision(class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario = _summary_by_scenario(class_rows, "confirmation", 28)
    joint = by_scenario["complete_joint"]
    shuffled = math.fsum(by_scenario[f"port_shuffle_{seed}"] for seed in SHUFFLE_SEEDS) / len(SHUFFLE_SEEDS)
    return _classify_difference(shuffled - joint, joint, shuffled)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(rows, "refusing to write an empty CSV")
    _require(not path.exists(), f"refusing to overwrite existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recheck_csv(path: Path, population: Mapping[str, Any], class_rows_path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == len(FORMAL_SCENARIOS) * TOTAL_ROWS, "paired row CSV has an incomplete scenario population")
    scenarios = {row["scenario"] for row in rows}
    _require(scenarios == set(FORMAL_SCENARIOS), "paired row CSV scenario set differs")
    for scenario in FORMAL_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        _require(len(selected) == TOTAL_ROWS, f"paired row CSV lacks 500 {scenario} rows")
        _require([int(row["position"]) for row in selected] == list(range(TOTAL_ROWS)), f"paired row CSV {scenario} ordering differs")
        _require([int(row["source_row_index"]) for row in selected] == population["source_row_indices"], f"paired row CSV {scenario} source identities differ")
        _require([int(row["label"]) for row in selected] == population["labels"], f"paired row CSV {scenario} labels differ")
        _require([row["role"] for row in selected] == population["roles"], f"paired row CSV {scenario} roles differ")
        for row in selected:
            _require(math.isfinite(float(row["generator_input_power"])) and float(row["generator_input_power"]) > 0.0, "paired row CSV includes invalid generator input power")
            _require(math.isfinite(float(row["camera_integrated_intensity"])) and float(row["camera_integrated_intensity"]) > 0.0, "paired row CSV includes invalid camera intensity")
            _require(0.0 <= float(row["raw_clip_fraction"]) <= 1.0, "paired row CSV clipping fraction is invalid")

    with class_rows_path.open("r", encoding="utf-8-sig", newline="") as stream:
        class_rows = []
        for record in csv.DictReader(stream):
            class_rows.append({
                "scenario": record["scenario"],
                "seed": None if record["seed"] == "" else int(record["seed"]),
                "role": record["role"],
                "label": int(record["label"]),
                "rows": int(record["rows"]),
                "shape_size": int(record["shape_size"]),
                "energy_distance_squared": float(record["energy_distance_squared"]),
            })
    expected_distance_rows = len(FORMAL_SCENARIOS) * 2 * len(CLASSES)
    observed_shapes = {row["shape_size"] for row in class_rows}
    _require(
        len(class_rows) == expected_distance_rows * len(observed_shapes)
        and observed_shapes in ({28}, {28, 200}),
        "class distance CSV is incomplete or contains an unexpected shape view",
    )
    primary = _primary_decision(class_rows)
    _require(primary["classification"] == expected["classification"], "CSV-recomputed primary classification differs")
    _require(math.isclose(primary["difference_shuffled_minus_joint"], expected["difference_shuffled_minus_joint"], rel_tol=0.0, abs_tol=1e-14), "CSV-recomputed primary effect differs")
    _require(math.isclose(primary["tolerance"], expected["tolerance"], rel_tol=0.0, abs_tol=1e-14), "CSV-recomputed tolerance differs")
    return {"passed": True, "paired_rows": len(rows), "class_distance_rows": len(class_rows), "primary_decision": primary}


def _make_plan() -> str:
    return """# 实验三：当前冻结解码器是否依赖跨端口联合结构

本实验使用能量对照 attempt_03 的500行A4700经验OLS人口，300行确认角色为唯一主分析，200行拟合角色独立描述。按类别和角色分别重组100个复OLS端口，保留每端口复数边缘多重集；另做完整向量整行置换负控。A4700冻结解码器不更新，所有输入经过同一 W_OLS 与Generator生产前向。

以不截断的Pool28强度并逐图归一化为形态向量；主量为每类别输出集合与保存真实目标集合间的平方欧氏能量距离。十类等权。唯一效应为10个固定种子端口重组距离均值减完整联合组距离，采用 2e-7 + 2e-5·max(|两组均值|) 数值容差。高于容差表示当前端点得到联合配对贡献的局部数值支持；低于负容差削弱该联合配对必须保留的解释；容差内未区分。该指标不是光学能量、人眼可读性或独立泛化证据。

输入核验通过后，执行CUDA代表性预检；再以严格FP32单行微批解码500行完整联合、500行整行控制及10组各500行端口重组，每50行成块落盘。重读原始配对表与类别距离表并独立重算唯一判读。完成报告后停止，不训练、不新增人审、不拟合联合采样器、不启动后继实验。
"""


def _readout_path(output_dir: Path, scenario: str, chunk_index: int) -> Path:
    return output_dir / "readouts" / scenario / f"chunk_{chunk_index:04d}.pt"


def _infer_batch(
    runtime: torch.nn.Module,
    evaluator: torch.nn.Module,
    z: torch.Tensor,
    device: torch.device,
    camera_pixel_area: float,
) -> dict[str, torch.Tensor]:
    _require(tuple(z.shape) == (1, PORTS) and z.is_complex(), "production optical forward requires a single complex OLS row")
    z_device = z.to(device=device, dtype=torch.complex64)
    with torch.inference_mode():
        result = runtime.forward_from_ols(z_device)
        raw = result.raw_intensity.detach()
        view = generation.capacity.canonical.cmos_view(raw)
        logits = generation._evaluate(evaluator, view, "E_D")
        generator_power = generation.bogt.physical_plane_power(result.generator_input_field).detach().to(torch.float64)
    raw_cpu = raw.to(device="cpu", dtype=torch.float32).contiguous()
    view_cpu = view.to(device="cpu", dtype=torch.float32).contiguous()
    logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    generator_power_cpu = generator_power.cpu().contiguous()
    camera_intensity = raw_cpu.to(torch.float64).flatten(1).sum(1) * camera_pixel_area
    clip_fraction = ((raw_cpu < 0.0) | (raw_cpu > 1.0)).flatten(1).to(torch.float64).mean(1)
    _require(bool(torch.isfinite(raw_cpu).all() and torch.isfinite(view_cpu).all() and torch.isfinite(logits_cpu).all()), "production readout contains nonfinite values")
    _require(bool((raw_cpu >= 0.0).all()), "production CMOS intensity contains negative values")
    _require(bool((camera_intensity > 0.0).all() and (generator_power_cpu > 0.0).all()), "production readout contains zero energy")
    return {
        "raw_cmos": raw_cpu,
        "evaluator_view": view_cpu,
        "e_d_logits": logits_cpu,
        "generator_input_power": generator_power_cpu,
        "camera_integrated_intensity": camera_intensity,
        "raw_clip_fraction": clip_fraction,
    }


def _infer_rows(
    runtime: torch.nn.Module,
    evaluator: torch.nn.Module,
    z: torch.Tensor,
    device: torch.device,
    camera_pixel_area: float,
) -> dict[str, torch.Tensor]:
    _require(z.ndim == 2 and z.shape[1] == PORTS and z.shape[0] > 0 and z.is_complex(), "production OLS rows must be a nonempty complex N×100 matrix")
    results = [
        _infer_batch(runtime, evaluator, z[index : index + 1], device, camera_pixel_area)
        for index in range(z.shape[0])
    ]
    keys = tuple(results[0])
    _require(all(set(result) == set(keys) for result in results), "single-row production readout fields differ")
    return {key: torch.cat([result[key] for result in results], dim=0) for key in keys}


def _run_preflight(
    inputs: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    runtime: torch.nn.Module,
    evaluators: Mapping[str, torch.nn.Module],
) -> dict[str, Any]:
    evaluator = evaluators["E_D"]
    runtime_before = generation._snapshot_state(runtime)
    evaluator_before = generation.capacity._state_snapshot(evaluator)
    labels = inputs["labels"]
    roles = inputs["roles"]
    z = inputs["readout"]["z"].to(torch.complex64).contiguous()
    saved = inputs["readout"]
    carriers = _preflight_carriers(labels, roles)
    whole_map = _permutation_map(labels, roles, seed=WHOLE_ROW_SEED, portwise=False)
    shuffle_map = _permutation_map(labels, roles, seed=SHUFFLE_SEEDS[0], portwise=True)
    theta = torch.linspace(0.13, 2.41, len(carriers), dtype=torch.float64)
    phase_factor = torch.polar(torch.ones_like(theta), theta).to(torch.complex64)

    baseline = _infer_rows(runtime, evaluator, z.index_select(0, carriers), device, inputs["pixel_area_m2"])
    whole = _infer_rows(runtime, evaluator, z.index_select(0, whole_map[carriers, 0]), device, inputs["pixel_area_m2"])
    shuffled_z = _apply_map(z, shuffle_map, labels, roles)
    shuffled = _infer_rows(runtime, evaluator, shuffled_z.index_select(0, carriers), device, inputs["pixel_area_m2"])
    phase = _infer_rows(runtime, evaluator, z.index_select(0, carriers) * phase_factor[:, None], device, inputs["pixel_area_m2"])
    scaled = _infer_rows(runtime, evaluator, z.index_select(0, carriers) * SCALE_FACTOR, device, inputs["pixel_area_m2"])
    source_raw = saved["raw_cmos"].index_select(0, carriers)
    source_logits = saved["e_d_logits"].index_select(0, carriers)
    whole_donors = whole_map.index_select(0, carriers)[:, 0]
    whole_source_raw = saved["raw_cmos"].index_select(0, whole_donors)
    _require(torch.allclose(baseline["raw_cmos"], source_raw, rtol=PREFLIGHT_RTOL, atol=PREFLIGHT_ATOL), "A4700 production replay differs from saved raw CMOS")
    _require(torch.allclose(whole["raw_cmos"], whole_source_raw, rtol=PREFLIGHT_RTOL, atol=PREFLIGHT_ATOL), "whole-row permutation does not reproduce its mapped baseline rows")
    _require(torch.allclose(scaled["raw_cmos"], baseline["raw_cmos"] * SCALE_FACTOR**2, rtol=2.0e-5, atol=PREFLIGHT_ATOL), "amplitude scaling does not follow linear optical power scaling")
    baseline_shape = _shape_vectors(baseline["raw_cmos"], 28)
    scaled_shape = _shape_vectors(scaled["raw_cmos"], 28)
    phase_shape = _shape_vectors(phase["raw_cmos"], 28)
    _require(torch.allclose(baseline_shape, scaled_shape, rtol=0.0, atol=2.0e-10), "normalized shape changes under pure amplitude scaling")
    _require(torch.allclose(baseline_shape, phase_shape, rtol=0.0, atol=1.0e-8), "global phase invariance fails in normalized shape")
    phase_intensity_delta = (
        (phase["camera_integrated_intensity"] - baseline["camera_integrated_intensity"]).abs()
        / baseline["camera_integrated_intensity"].abs().clamp_min(torch.finfo(torch.float64).tiny)
    )
    _require(float(phase_intensity_delta.max()) <= PREFLIGHT_RTOL, "global phase rotation changes integrated CMOS intensity beyond FP32 tolerance")
    _require(
        torch.allclose(
            generation.capacity.canonical.cmos_view(baseline["raw_cmos"]),
            saved["evaluator_view"].index_select(0, carriers),
            rtol=PREFLIGHT_RTOL,
            atol=PREFLIGHT_ATOL,
        ),
        "saved production display view differs on preflight rows",
    )
    runtime_unchanged = generation._tree_equal(runtime_before, generation._snapshot_state(runtime))
    evaluator_unchanged = generation._tree_equal(evaluator_before, generation.capacity._state_snapshot(evaluator))
    gradients_none = all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters())
    gradients_none = gradients_none and all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters())
    _require(runtime_unchanged and evaluator_unchanged and gradients_none, "frozen model or E_D state changed during preflight")
    auxiliary_delta = (baseline["e_d_logits"] - source_logits).abs()
    auxiliary_classifier_replay = {
        "executed": True,
        "rows": len(CLASSES),
        "saved_prediction_agreement": int((baseline["e_d_logits"].argmax(1) == source_logits.argmax(1)).sum()),
        "max_abs_logit_difference": float(auxiliary_delta.max()),
        "mean_abs_logit_difference": float(auxiliary_delta.mean()),
        "single_row_batching_can_change_fp32_logits": True,
        "blocking_check": False,
    }
    payload = {
        "baseline": baseline,
        "whole_row_control": whole,
        "port_shuffle": shuffled,
        "global_phase_rotation": phase,
        "amplitude_scale": scaled,
        "carrier_positions": carriers,
        "whole_row_mapping": whole_map.index_select(0, carriers),
        "port_shuffle_mapping": shuffle_map.index_select(0, carriers),
        "auxiliary_classifier_replay": auxiliary_classifier_replay,
    }
    _torch_write_new(output_dir / "preflight.pt", payload)
    reloaded = torch.load(output_dir / "preflight.pt", map_location="cpu", weights_only=True)
    _require(torch.equal(reloaded["baseline"]["raw_cmos"], baseline["raw_cmos"]), "preflight output did not survive write/read round-trip")
    return {
        "task_id": TASK_ID,
        "status": "READY",
        "checks": {
            "saved_a4700_raw_and_view_replay": {"executed": True, "passed": True, "rows": len(CLASSES), "rtol": PREFLIGHT_RTOL, "atol": PREFLIGHT_ATOL},
            "whole_row_negative_control": {"executed": True, "passed": True},
            "portwise_recombination_forward": {"executed": True, "passed": bool(torch.isfinite(shuffled["raw_cmos"]).all())},
            "global_phase_invariance": {
                "executed": True,
                "passed": True,
                "maximum_normalized_shape_absolute_difference": float((phase_shape - baseline_shape).abs().max()),
                "normalized_shape_absolute_tolerance": 1.0e-8,
                "maximum_relative_integrated_intensity_difference": float(phase_intensity_delta.max()),
                "integrated_intensity_relative_tolerance": PREFLIGHT_RTOL,
            },
            "amplitude_scale_shape_invariance": {"executed": True, "passed": True, "scale": SCALE_FACTOR},
            "saved_view_replay": {"executed": True, "passed": True},
            "model_and_evaluator_frozen": {"executed": True, "passed": True},
            "output_round_trip": {"executed": True, "passed": True},
        },
        "auxiliary_classifier_replay": auxiliary_classifier_replay,
        "forward_rows": 5 * len(CLASSES),
        "model_updates": 0,
        "optimizer_updates": 0,
        "gradient_calls": 0,
    }


def _scenario_mappings(labels: torch.Tensor, roles: list[str]) -> dict[str, torch.Tensor]:
    maps = {"complete_joint": torch.arange(TOTAL_ROWS, dtype=torch.long).unsqueeze(1).expand(-1, PORTS).clone()}
    maps["whole_row_control"] = _permutation_map(labels, roles, seed=WHOLE_ROW_SEED, portwise=False)
    for seed in SHUFFLE_SEEDS:
        maps[f"port_shuffle_{seed}"] = _permutation_map(labels, roles, seed=seed, portwise=True)
    return maps


def _preflight_carriers(labels: torch.Tensor, roles: list[str]) -> torch.Tensor:
    role_ids = torch.tensor([0 if role == "fit" else 1 for role in roles], dtype=torch.long)
    carriers = []
    for digit in CLASSES:
        positions = ((labels == digit) & (role_ids == 1)).nonzero().flatten()
        _require(positions.numel() == 30, f"confirmation class {digit} does not have 30 rows")
        carriers.append(positions[0])
    result = torch.stack(carriers)
    _require(torch.equal(labels.index_select(0, result), torch.tensor(CLASSES, dtype=torch.long)), "preflight rows do not cover each class exactly once")
    _require(bool((role_ids.index_select(0, result) == 1).all()), "preflight row is outside the confirmation role")
    return result


def _scenario_rows(
    scenario: str,
    seed: int | None,
    mapping: torch.Tensor,
    batch_start: int,
    result: Mapping[str, torch.Tensor],
    inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    population = inputs["population"]
    labels = inputs["labels"]
    roles = inputs["roles"]
    predictions = result["e_d_logits"].argmax(1)
    rows = []
    for local, position in enumerate(range(batch_start, batch_start + result["raw_cmos"].shape[0])):
        donor_positions = mapping[position].tolist()
        rows.append({
            "scenario": scenario,
            "seed": "" if seed is None else seed,
            "position": position,
            "source_row_index": population["source_row_indices"][position],
            "label": int(labels[position]),
            "role": roles[position],
            "donor_source_row_indices": json.dumps([population["source_row_indices"][idx] for idx in donor_positions], separators=(",", ":")),
            "generator_input_power": float(result["generator_input_power"][local]),
            "camera_integrated_intensity": float(result["camera_integrated_intensity"][local]),
            "raw_clip_fraction": float(result["raw_clip_fraction"][local]),
            "e_d_prediction": int(predictions[local]),
            "e_d_correct": int(predictions[local] == labels[position]),
        })
    return rows


def _run_scenario(
    runtime: torch.nn.Module,
    evaluator: torch.nn.Module,
    z: torch.Tensor,
    mapping: torch.Tensor,
    scenario: str,
    seed: int | None,
    inputs: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    scenario_rows = []
    for chunk_index, start in enumerate(range(0, TOTAL_ROWS, CHUNK_ROWS)):
        stop = min(start + CHUNK_ROWS, TOTAL_ROWS)
        indices = mapping[start:stop]
        batch_z = z.gather(0, indices)
        result = _infer_rows(runtime, evaluator, batch_z, device, inputs["pixel_area_m2"])
        expected_labels = inputs["labels"][start:stop]
        donor_labels = inputs["labels"].index_select(0, indices.flatten()).reshape(stop - start, PORTS)
        _require(bool((donor_labels == expected_labels[:, None]).all()), f"{scenario} changed the requested class")
        _require(bool(torch.isfinite(batch_z.real).all() and torch.isfinite(batch_z.imag).all()), f"{scenario} generated nonfinite OLS input")
        payload = {
            "task_id": TASK_ID,
            "scenario": scenario,
            "seed": seed,
            "positions": torch.arange(start, stop, dtype=torch.long),
            "source_row_indices": torch.tensor(inputs["population"]["source_row_indices"][start:stop], dtype=torch.long),
            "labels": expected_labels,
            "roles": inputs["roles"][start:stop] if isinstance(inputs["roles"], str) else list(inputs["roles"][start:stop]),
            "donor_positions": indices,
            "z": batch_z.to(torch.complex64).cpu().contiguous(),
            **result,
        }
        path = _readout_path(output_dir, scenario, chunk_index)
        _torch_write_new(path, payload)
        _require(torch.equal(torch.load(path, map_location="cpu", weights_only=True)["positions"], payload["positions"]), f"{scenario} chunk did not survive write/read round-trip")
        scenario_rows.extend(_scenario_rows(scenario, seed, mapping, start, result, inputs))
    return scenario_rows


def _load_scenario_raw(output_dir: Path, scenario: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = []
    for chunk_index, _ in enumerate(range(0, TOTAL_ROWS, CHUNK_ROWS)):
        payload = torch.load(_readout_path(output_dir, scenario, chunk_index), map_location="cpu", weights_only=True, mmap=True)
        _require(payload.get("scenario") == scenario and payload.get("task_id") == TASK_ID, f"saved {scenario} chunk identity differs")
        chunks.append(payload)
    raw = torch.cat([chunk["raw_cmos"] for chunk in chunks])
    labels = torch.cat([chunk["labels"] for chunk in chunks])
    power = torch.cat([chunk["generator_input_power"] for chunk in chunks])
    intensity = torch.cat([chunk["camera_integrated_intensity"] for chunk in chunks])
    _require(raw.shape == (TOTAL_ROWS, 1, 200, 200) and labels.shape == (TOTAL_ROWS,), f"saved {scenario} rows are incomplete")
    return raw, labels, power, intensity


def _classifier_summary(output_dir: Path, scenario: str, labels: torch.Tensor, roles: list[str]) -> dict[str, Any]:
    logits = []
    for chunk_index, _ in enumerate(range(0, TOTAL_ROWS, CHUNK_ROWS)):
        payload = torch.load(_readout_path(output_dir, scenario, chunk_index), map_location="cpu", weights_only=True, mmap=True)
        logits.append(payload["e_d_logits"])
    predictions = torch.cat(logits).argmax(1)
    summary: dict[str, Any] = {}
    role_ids = torch.tensor([0 if role == "fit" else 1 for role in roles], dtype=torch.long)
    for role, role_id in (("fit", 0), ("confirmation", 1)):
        selected_role = role_ids == role_id
        correct = predictions[selected_role] == labels[selected_role]
        summary[role] = {"correct": int(correct.sum()), "rows": int(selected_role.sum()), "accuracy": float(correct.to(torch.float64).mean())}
        summary[role]["by_class"] = {}
        for digit in CLASSES:
            selected = selected_role & (labels == digit)
            summary[role]["by_class"][str(digit)] = {
                "correct": int(((predictions == labels) & selected).sum()),
                "rows": int(selected.sum()),
                "accuracy": float(((predictions == labels) & selected).to(torch.float64).mean()),
            }
    return summary


def _summarize_scenario(
    output_dir: Path,
    scenario: str,
    seed: int | None,
    inputs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, labels, power, intensity = _load_scenario_raw(output_dir, scenario)
    targets = inputs["readout"]["target_slm_intensity"].to(torch.float32)
    rows = []
    for shape_size in (28, 200):
        rows.extend(_class_distances(raw, targets, labels, inputs["roles"], scenario=scenario, seed=seed, shape_size=shape_size))
    clip = ((raw < 0.0) | (raw > 1.0)).flatten(1).to(torch.float64).mean(1)
    summary = {
        "seed": seed,
        "generator_input_power": {"mean": float(power.mean()), "minimum": float(power.min()), "maximum": float(power.max())},
        "camera_integrated_intensity": {"mean": float(intensity.mean()), "minimum": float(intensity.min()), "maximum": float(intensity.max())},
        "raw_clip_fraction": {"mean": float(clip.mean()), "maximum": float(clip.max())},
        "shape_distance_28_by_role": {
            role: _summary_values([row["energy_distance_squared"] for row in rows if row["shape_size"] == 28 and row["role"] == role])
            for role in ("fit", "confirmation")
        },
        "shape_distance_200_by_role": {
            role: _summary_values([row["energy_distance_squared"] for row in rows if row["shape_size"] == 200 and row["role"] == role])
            for role in ("fit", "confirmation")
        },
        "classifier_readout": _classifier_summary(output_dir, scenario, inputs["labels"], inputs["roles"]),
    }
    return rows, summary


def _summary_values(values: list[float]) -> dict[str, float | int]:
    _require(values and all(math.isfinite(value) for value in values), "summary values are empty or nonfinite")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    _json_write_new(path, report)


def _report_markdown(report: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    decision = report["decision"]
    classifier_replay = report["preflight"]["auxiliary_classifier_replay"]
    return f"""# 实验三结果：当前冻结解码器是否依赖跨端口联合结构

## 结果与状态

本次保存证据测量状态为 `{report['status']}`，科学分类为 `{decision['classification']}`。唯一主量 (D_{{joint}}={decision['difference_shuffled_minus_joint']:.9g})，数值容差 (τ={decision['tolerance']:.9g})。完整联合组平均形态距离为 `{decision['complete_joint_mean']:.9g}`；固定10种子逐端口重组均值为 `{decision['port_shuffle_mean_over_fixed_seeds']:.9g}`。逐行CSV复算：`{validation['primary_decision']['classification']}`。

## 科学认识

{decision['interpretation']} 结论仅涉及A4700、当前500行TRAIN经验OLS人口、固定置换种子和未截断归一化形态集合指标。它不证明普遍必要性、人眼可读性、独立泛化或新采样器收益。实验一的审阅质控状态仍为未通过。

主分析使用300行确认角色、十类等权；200行拟合角色单独汇总。固定种子效应范围为 `{report['secondary']['shuffle_effect_range_confirmation']}`。整行置换控制与联合组的形态距离差为 `{report['secondary']['whole_row_minus_joint_confirmation']:.9g}`。

## 目标缺口

仍缺少实验一质控通过后的人眼识读证据，以及独立于当前TRAIN人口的泛化证据。本实验只量化冻结解码器在已保存人口上的形态分布响应。

## 工程完成与验证

真实A4700生产路径执行 `{report['scope']['formal_forward_rows']}` 行读出（含500行完整联合组、500行整行置换负控及10组各500行端口重组）。CUDA预检、A4700 raw与生产视图重放、整行置换、归一化形态和积分强度上的全局相位控制、幅度缩放、模型冻结和文件复读均已完成。E_D辅助重算的代表行预测与保存值一致 `{classifier_replay['saved_prediction_agreement']}/{classifier_replay['rows']}`；单行FP32批次的logit最大差为 `{classifier_replay['max_abs_logit_difference']:.3g}`，作为非阻断数值差异记录。CSV复核包含 `{validation['paired_rows']}` 行配对记录与 `{validation['class_distance_rows']}` 条类别距离记录，终局计算与机器报告一致。

训练更新0；优化器更新0；梯度调用0；新增人审0。辅助分类器只对各组新生产视图计算，不解释为人眼可读性。

## 决策影响与下一步

本实验只为“当前A4700解码器的形态表现是否从经验OLS跨端口配对获益”提供数值证据。即使得到支持，也不授权训练、拟合新联合先验、采用其他端点或启动后续实验。`training_recommendation=NONE`，`next_action=NONE`。
"""


def _verify_whole_row_control(output_dir: Path, inputs: Mapping[str, Any]) -> float:
    raw, _, _, _ = _load_scenario_raw(output_dir, "complete_joint")
    control, _, _, _ = _load_scenario_raw(output_dir, "whole_row_control")
    expected = torch.empty_like(raw)
    labels, roles = inputs["labels"], inputs["roles"]
    mapping = _permutation_map(labels, roles, seed=WHOLE_ROW_SEED, portwise=False)
    for position in range(TOTAL_ROWS):
        donor = int(mapping[position, 0])
        expected[position] = raw[donor]
    _require(torch.allclose(control, expected, rtol=PREFLIGHT_RTOL, atol=PREFLIGHT_ATOL), "whole-row control outputs do not match the inverse-mapped baseline images")
    shape_control = _shape_vectors(control[FIT_ROWS:], 28)
    shape_joint = _shape_vectors(raw[FIT_ROWS:], 28)
    confirm_labels = inputs["labels"][FIT_ROWS:]
    class_distances = [
        _energy_distance(shape_control[confirm_labels == digit], shape_joint[confirm_labels == digit])
        for digit in CLASSES
    ]
    max_distance = max(class_distances)
    _require(max_distance <= 1.0e-6, "whole-row permutation changed a class-conditional shape set")
    return max_distance


def _run(artifact_dir: Path, checkpoint_arg: Path, output_arg: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    start_time = time.perf_counter()
    output_dir = _prepare_output(root, output_arg)
    with (output_dir / "experiment_03_plan.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_make_plan())
        stream.flush()
        os.fsync(stream.fileno())
    inputs = _load_inputs(root, artifact_dir.resolve(), checkpoint_arg)
    _json_write_new(output_dir / "input_population.json", inputs["population"])

    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized frozen production-path measurement")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    disk_free = shutil.disk_usage(output_dir).free
    _require(total_bytes > 0 and free_bytes > 0, "CUDA device reports no usable memory")
    runtime = generation._new_runtime(inputs["checkpoint"], device)
    evaluators = generation.diagnostic.load_evaluators(inputs["generation_paths"], device)
    evaluator = evaluators["E_D"]
    preflight = _run_preflight(inputs, output_dir, device, runtime, evaluators)
    preflight["resources_observed"] = {
        "cuda_device": str(device),
        "cuda_total_bytes": int(total_bytes),
        "cuda_free_bytes_before_measurement": int(free_bytes),
        "disk_free_bytes_before_measurement": int(disk_free),
        "estimated_readout_bytes": len(FORMAL_SCENARIOS) * TOTAL_ROWS * (200 * 200 * 4 + 28 * 28 * 4 + 10 * 4 + 3 * 8),
        "memory_and_elapsed_time_are_observations_only": True,
    }
    _json_write_new(output_dir / "preflight.json", preflight)
    _require(preflight["status"] == "READY" and all(item["passed"] and item["executed"] for item in preflight["checks"].values()), "production preflight did not produce READY")

    runtime_before = generation._snapshot_state(runtime)
    evaluator_before = generation.capacity._state_snapshot(evaluator)
    z = inputs["readout"]["z"].to(torch.complex64).contiguous()
    maps = _scenario_mappings(inputs["labels"], inputs["roles"])
    _torch_write_new(output_dir / "permutation_maps.pt", {"task_id": TASK_ID, "maps": maps, "seeds": list(SHUFFLE_SEEDS), "whole_row_seed": WHOLE_ROW_SEED})
    _json_write_new(output_dir / "validation.json", {
        "task_id": TASK_ID,
        "status": "VALIDATED",
        "source_rows": TOTAL_ROWS,
        "fit_rows": FIT_ROWS,
        "confirmation_rows": CONFIRM_ROWS,
        "per_class_counts_fit": torch.bincount(inputs["labels"][:FIT_ROWS], minlength=10).tolist(),
        "per_class_counts_confirmation": torch.bincount(inputs["labels"][FIT_ROWS:], minlength=10).tolist(),
        "source_identity_and_checkpoint": True,
        "all_source_intensities_finite_nonnegative_positive_mass": True,
        "physical_plane_metadata_replayed": True,
        "preflight_ready": True,
        "permutation_maps_validated": True,
    })

    paired_rows: list[dict[str, Any]] = []
    scenario_summary: dict[str, Any] = {}
    class_rows: list[dict[str, Any]] = []
    seeds_by_scenario: dict[str, int | None] = {"complete_joint": None, "whole_row_control": WHOLE_ROW_SEED}
    seeds_by_scenario.update({f"port_shuffle_{seed}": seed for seed in SHUFFLE_SEEDS})
    for scenario in FORMAL_SCENARIOS:
        print(f"decoding {scenario}: {TOTAL_ROWS} rows", flush=True)
        paired_rows.extend(_run_scenario(runtime, evaluator, z, maps[scenario], scenario, seeds_by_scenario[scenario], inputs, output_dir, device))
        rows, summary = _summarize_scenario(output_dir, scenario, seeds_by_scenario[scenario], inputs)
        class_rows.extend(rows)
        scenario_summary[scenario] = summary

    _verify_whole_row_control(output_dir, inputs)
    decision = _primary_decision(class_rows)
    fit_summary = _summary_by_scenario(class_rows, "fit", 28)
    confirm_summary = _summary_by_scenario(class_rows, "confirmation", 28)
    seed_effects = [confirm_summary[f"port_shuffle_{seed}"] - confirm_summary["complete_joint"] for seed in SHUFFLE_SEEDS]
    freeze_unchanged = generation._tree_equal(runtime_before, generation._snapshot_state(runtime))
    evaluator_unchanged = generation._tree_equal(evaluator_before, generation.capacity._state_snapshot(evaluator))
    gradients_none = all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters())
    gradients_none = gradients_none and all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters())
    _require(freeze_unchanged and evaluator_unchanged and gradients_none, "frozen runtime or evaluator changed during formal measurement")

    _write_csv(output_dir / "paired_rows.csv", paired_rows)
    _write_csv(output_dir / "class_distances.csv", class_rows)
    decision_after_csv = _primary_decision([
        {
            "scenario": row["scenario"],
            "seed": row["seed"],
            "role": row["role"],
            "label": row["label"],
            "rows": row["rows"],
            "shape_size": row["shape_size"],
            "energy_distance_squared": row["energy_distance_squared"],
        }
        for row in class_rows
    ])
    _require(decision_after_csv["classification"] == decision["classification"], "pre-save primary decision changed during report preparation")
    validation = _recheck_csv(output_dir / "paired_rows.csv", inputs["population"], output_dir / "class_distances.csv", decision)
    _require(validation["passed"], "saved CSV replay did not validate")

    whole_summary = confirm_summary["whole_row_control"]
    report = {
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "decision": decision,
        "scope": {
            "source_task": parent_task.TASK_ID,
            "source_attempt": inputs["artifact_dir"].as_posix(),
            "checkpoint": inputs["checkpoint_path"].as_posix(),
            "checkpoint_step": 4700,
            "populations": {"fit": FIT_ROWS, "confirmation": CONFIRM_ROWS, "total": TOTAL_ROWS, "class_fit": 20, "class_confirmation": 30},
            "canonical_train_population": True,
            "formal_forward_rows": len(FORMAL_SCENARIOS) * TOTAL_ROWS,
            "model_updates": 0,
            "optimizer_updates": 0,
            "gradient_calls": 0,
            "new_human_ratings": 0,
        },
        "measurement": {
            "intervention": "within class and role, independently permute source rows at each complex OLS port; real and imaginary parts remain paired",
            "marginals": "each class/role/port complex coefficient multiset is preserved exactly",
            "port_order": "row-major j=10*r+c",
            "whole_row_negative_control_seed": WHOLE_ROW_SEED,
            "port_shuffle_seeds": list(SHUFFLE_SEEDS),
            "shape_view": "adaptive_avg_pool2d(raw_cmos,(d,d)); normalize each d*d vector by its sum; no clipping",
            "primary_shape_size": 28,
            "auxiliary_shape_size": 200,
            "distance": "2*mean_pairwise_euclidean(X,T)-mean_pairwise_euclidean(X,X)-mean_pairwise_euclidean(T,T); V-statistic squared energy distance",
            "class_weighting": "ten classes equally weighted",
            "numerical_tolerance": {"absolute": TOLERANCE_ATOL, "relative": TOLERANCE_RTOL},
            "replay_tolerance": {"rtol": PREFLIGHT_RTOL, "atol": PREFLIGHT_ATOL},
            "camera_pixel_area_m2": inputs["pixel_area_m2"],
            "camera_pixel_pitch_m": inputs["camera_pixel_pitch_m"],
            "counterfactuals_are_new_frozen_optical_forwards": True,
        },
        "preflight": preflight,
        "validation": validation,
        "secondary": {
            "confirmation_distance_by_scenario_28": confirm_summary,
            "fit_distance_by_scenario_28": fit_summary,
            "confirmation_distance_by_scenario_200": {
                "complete_joint": scenario_summary["complete_joint"]["shape_distance_200_by_role"]["confirmation"]["mean"],
                "whole_row_control": scenario_summary["whole_row_control"]["shape_distance_200_by_role"]["confirmation"]["mean"],
                **{f"port_shuffle_{seed}": scenario_summary[f"port_shuffle_{seed}"]["shape_distance_200_by_role"]["confirmation"]["mean"] for seed in SHUFFLE_SEEDS},
            },
            "shuffle_effect_range_confirmation": {"minimum": min(seed_effects), "maximum": max(seed_effects), "mean": math.fsum(seed_effects) / len(seed_effects)},
            "whole_row_minus_joint_confirmation": whole_summary - confirm_summary["complete_joint"],
            "scenario_readouts": scenario_summary,
            "classifier_is_auxiliary_not_human_readability_evidence": True,
            "no_independent_confidence_interval_claim": True,
        },
        "artifacts": {
            "plan": "experiment_03_plan.md",
            "input_population": "input_population.json",
            "preflight": "preflight.json",
            "validation": "validation.json",
            "permutation_maps": "permutation_maps.pt",
            "readouts": "readouts/<scenario>/chunk_NNNN.pt",
            "paired_rows": "paired_rows.csv",
            "class_distances": "class_distances.csv",
            "report": "report.json",
            "report_markdown": "report.md",
            "completed": "completed.json",
        },
        "elapsed_seconds_observation_only": time.perf_counter() - start_time,
        "training_recommendation": "NONE",
        "next_action": "NONE",
    }
    _write_report(output_dir / "report.json", report)
    markdown = _report_markdown(report, validation)
    _require(not (output_dir / "report.md").exists(), "refusing to overwrite report.md")
    (output_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    _json_write_new(output_dir / "completed.json", {
        "task_id": TASK_ID,
        "status": "COMPLETED",
        "classification": decision["classification"],
        "primary_effect": decision["difference_shuffled_minus_joint"],
        "training_recommendation": "NONE",
        "next_action": "NONE",
    })
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    output_dir = _resolve(root, args.output_dir)
    try:
        report = _run(_resolve(root, args.artifact_dir), args.checkpoint, args.output_dir)
    except Exception as error:
        if output_dir.is_dir() and not (output_dir / "report.json").exists():
            failure = {
                "task_id": TASK_ID,
                "status": "ENGINEERING_ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_classification": None,
                "training_recommendation": "NONE",
                "next_action": "NONE",
            }
            try:
                _json_write_new(output_dir / "failure.json", failure)
            except Exception:
                pass
        raise
    print(json.dumps({"status": report["status"], "classification": report["classification"], "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
