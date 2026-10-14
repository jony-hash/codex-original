"""Compare block and 50%-shrunk joint Gaussian priors through frozen A4700."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_cross_port_joint_structure_v1 as frozen,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation,
)

TASK_ID = "M0_GENERATOR_FROZEN_JOINT_PRIOR_COMPARISON_V1"
SOURCE_DIR = frozen.SOURCE_DIR
DEFAULT_CHECKPOINT = SOURCE_DIR / "arm_a_step4700.pt"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20261009_m0_generator_frozen_joint_prior_comparison_v1/runs/attempt_01"
)
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
CLASS_COUNT = 10
PORTS = 100
REAL_DIM = 2 * PORTS
SAMPLE_COUNT_PER_CLASS = 30
SHRINKAGE = 0.5
SEEDS = tuple(range(2026100961, 2026100966))
CHUNK_ROWS = 50
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
TOLERANCE_ATOL = 2.0e-7
TOLERANCE_RTOL = 2.0e-5
PHASE_SEED = 2026100966


class PriorComparisonError(RuntimeError):
    """Source, preflight, or measurement verification failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PriorComparisonError(message)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _json_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite artifact: {path}")
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
        temporary.unlink(missing_ok=True)


def _torch_write_new(path: Path, value: Any) -> None:
    _require(not path.exists(), f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output(root: Path, output_dir: Path) -> Path:
    output_root = (root / "outputs").resolve()
    target = output_dir.resolve()
    _require(target != output_root and output_root in target.parents, "output must be a child of outputs/")
    parts = target.relative_to(output_root).parts
    _require(len(parts) == 3 and parts[0].startswith("20261009_") and parts[1] == "runs" and parts[2].startswith("attempt_"), "output must use outputs/YYYYMMDD_<version>/runs/attempt_N")
    _require(not any(item in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for item in parts), "protected output namespace is forbidden")
    _require(not target.exists(), f"output already exists; refusing to overwrite: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _make_plan() -> str:
    return """# 冻结A4700联合先验对照

唯一比较为每类20行拟合的分块高斯与50%收缩联合高斯。两者使用相同经验均值、逐端口2×2实虚协方差边缘、五个固定种子及共同标准正态噪声；每类每种子生成30行。冻结A4700解码器执行严格FP32单行前向，不训练、不更新参数、不调用梯度。

唯一主量为300行确认人口、十类与五种子等权的28×28未截断归一化形态集合距离差：联合减分块。容差固定为`2e-7 + 2e-5 × max(|两臂均值|)`。负差超容差表示当前联合候选改善；正差超容差表示变差；容差内表示未观察到超容差收益。此局部数值结果不等于人眼可读性、独立泛化或采用结论。

先验证来源、协方差边缘与分解，再运行50行CUDA代表性预检，READY后正式解码两臂各1500行，分50行落盘。重读读出和CSV独立重算唯一判读。完成本对照即停止；不更新模型、不拟合复杂先验、不启动后续实验。
"""


def _to_interleaved(z: torch.Tensor) -> torch.Tensor:
    _require(z.is_complex() and z.ndim == 2 and z.shape[1] == PORTS, "OLS input must be a complex N×100 matrix")
    _require(bool(torch.isfinite(z.real).all() and torch.isfinite(z.imag).all()), "OLS coefficients contain nonfinite values")
    values = torch.stack((z.real.to(torch.float64), z.imag.to(torch.float64)), dim=-1).reshape(z.shape[0], REAL_DIM)
    _require(bool(torch.isfinite(values).all()), "interleaved fit coefficients are nonfinite")
    return values


def _fit_class_priors(z: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(tuple(z.shape) == (FIT_ROWS, PORTS), "fit OLS input must be complex[200,100]")
    x = _to_interleaved(z)
    _require(tuple(labels.shape) == (FIT_ROWS,) and bool(((labels >= 0) & (labels < CLASS_COUNT)).all()), "fit labels are invalid")
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), 20)), "fit roles are not exactly 20 rows per class")
    means, scales, block_chols, joint_chols = [], [], [], []
    block_correlations, joint_correlations = [], []
    reconstruction_errors, minimum_eigenvalues = [], []
    for digit in range(CLASS_COUNT):
        rows = x[labels == digit]
        _require(tuple(rows.shape) == (20, REAL_DIM), f"class {digit} fit rows are incomplete")
        mean = rows.mean(dim=0)
        centered = rows - mean
        covariance = centered.T @ centered / 20.0
        diagonal = covariance.diag()
        _require(bool(torch.isfinite(covariance).all() and (diagonal > 0.0).all()), f"class {digit} has a nonpositive or invalid marginal variance")
        scale = diagonal.sqrt()
        correlation = covariance / scale[:, None] / scale[None, :]
        _require(bool(torch.isfinite(correlation).all()), f"class {digit} correlation matrix is nonfinite")
        correlation = 0.5 * (correlation + correlation.T)
        block = torch.zeros_like(correlation)
        for port in range(PORTS):
            begin = 2 * port
            block[begin : begin + 2, begin : begin + 2] = correlation[begin : begin + 2, begin : begin + 2]
        joint = (1.0 - SHRINKAGE) * correlation + SHRINKAGE * block
        _require(torch.equal(torch.stack([correlation[2*p:2*p+2, 2*p:2*p+2] for p in range(PORTS)]), torch.stack([block[2*p:2*p+2, 2*p:2*p+2] for p in range(PORTS)])), "within-port correlation blocks changed during shrinkage")
        block_chol = torch.linalg.cholesky(block)
        joint_chol = torch.linalg.cholesky(joint)
        block_error = float((block_chol @ block_chol.T - block).abs().max())
        joint_error = float((joint_chol @ joint_chol.T - joint).abs().max())
        _require(max(block_error, joint_error) <= 1.0e-12, f"class {digit} standardized Cholesky reconstruction error exceeds 1e-12")
        eigen_min = min(float(torch.linalg.eigvalsh(block).min()), float(torch.linalg.eigvalsh(joint).min()))
        _require(math.isfinite(eigen_min) and eigen_min > 0.0, f"class {digit} standardized covariance is not positive definite")
        means.append(mean)
        scales.append(scale)
        block_chols.append(block_chol)
        joint_chols.append(joint_chol)
        block_correlations.append(block)
        joint_correlations.append(joint)
        reconstruction_errors.append(max(block_error, joint_error))
        minimum_eigenvalues.append(eigen_min)
    return {
        "means": torch.stack(means),
        "scales": torch.stack(scales),
        "block_cholesky": torch.stack(block_chols),
        "joint_cholesky": torch.stack(joint_chols),
        "block_correlation": torch.stack(block_correlations),
        "joint_correlation": torch.stack(joint_correlations),
        "reconstruction_error": torch.tensor(reconstruction_errors, dtype=torch.float64),
        "minimum_eigenvalue": torch.tensor(minimum_eigenvalues, dtype=torch.float64),
    }


def _sample_banks(priors: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    banks: dict[str, list[torch.Tensor]] = {"block": [], "joint": []}
    noise_rows = []
    for seed in SEEDS:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        epsilon = torch.randn((CLASS_COUNT, SAMPLE_COUNT_PER_CLASS, REAL_DIM), generator=rng, dtype=torch.float64)
        noise_rows.append(epsilon)
        for arm, key in (("block", "block_cholesky"), ("joint", "joint_cholesky")):
            z_rows = []
            for digit in range(CLASS_COUNT):
                correlated = epsilon[digit] @ priors[key][digit].T
                real_vector = priors["means"][digit] + correlated * priors["scales"][digit]
                complex_vector = torch.complex(real_vector[:, 0::2], real_vector[:, 1::2]).to(torch.complex64)
                z_rows.append(complex_vector)
            bank = torch.stack(z_rows)
            _require(tuple(bank.shape) == (CLASS_COUNT, SAMPLE_COUNT_PER_CLASS, PORTS), "sample bank geometry differs")
            _require(bool(torch.isfinite(bank.real).all() and torch.isfinite(bank.imag).all()), f"{arm} sampled OLS bank contains nonfinite values")
            banks[arm].append(bank)
    return {arm: torch.stack(seeds_banks) for arm, seeds_banks in banks.items()}, torch.stack(noise_rows)


def _preflight(
    inputs: Mapping[str, Any], output_dir: Path, device: torch.device,
    runtime: torch.nn.Module, evaluator: torch.nn.Module,
    banks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    runtime_before = generation._snapshot_state(runtime)
    evaluator_before = generation.capacity._state_snapshot(evaluator)
    labels = inputs["labels"]
    readout = inputs["readout"]
    confirm_start = FIT_ROWS
    saved_rows = []
    for digit in range(CLASS_COUNT):
        positions = (labels[confirm_start:] == digit).nonzero().flatten()
        _require(positions.numel() == 30, f"confirmation class {digit} is not 30 rows")
        saved_rows.append(positions[0] + confirm_start)
    saved_positions = torch.stack(saved_rows)
    baseline_z = readout["z"].index_select(0, saved_positions).to(torch.complex64)
    baseline = frozen._infer_rows(runtime, evaluator, baseline_z, device, inputs["pixel_area_m2"])
    _require(torch.allclose(baseline["raw_cmos"], readout["raw_cmos"].index_select(0, saved_positions), rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "saved A4700 raw CMOS replay differs")
    baseline_view = generation.capacity.canonical.cmos_view(baseline["raw_cmos"])
    _require(torch.allclose(baseline_view, readout["evaluator_view"].index_select(0, saved_positions), rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "saved A4700 production view replay differs")

    block_z = banks["block"][0, :, 0, :]
    joint_z = banks["joint"][0, :, 0, :]
    block = frozen._infer_rows(runtime, evaluator, block_z, device, inputs["pixel_area_m2"])
    joint = frozen._infer_rows(runtime, evaluator, joint_z, device, inputs["pixel_area_m2"])
    theta = torch.linspace(0.17, 2.73, CLASS_COUNT, dtype=torch.float64)
    phase_factor = torch.polar(torch.ones_like(theta), theta).to(torch.complex64)
    phase = frozen._infer_rows(runtime, evaluator, joint_z * phase_factor[:, None], device, inputs["pixel_area_m2"])
    scaled = frozen._infer_rows(runtime, evaluator, joint_z * 0.5, device, inputs["pixel_area_m2"])
    joint_shape = frozen._shape_vectors(joint["raw_cmos"], 28)
    phase_shape = frozen._shape_vectors(phase["raw_cmos"], 28)
    scaled_shape = frozen._shape_vectors(scaled["raw_cmos"], 28)
    _require(torch.allclose(joint_shape, phase_shape, rtol=0.0, atol=1.0e-8), "global phase changes normalized output shape")
    _require(torch.allclose(joint_shape, scaled_shape, rtol=0.0, atol=2.0e-10), "amplitude scaling changes normalized output shape")
    _require(torch.allclose(scaled["raw_cmos"], joint["raw_cmos"] * 0.25, rtol=2.0e-5, atol=REPLAY_ATOL), "half amplitude does not scale raw intensity by one quarter")
    runtime_unchanged = generation._tree_equal(runtime_before, generation._snapshot_state(runtime))
    evaluator_unchanged = generation._tree_equal(evaluator_before, generation.capacity._state_snapshot(evaluator))
    gradients_none = all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters())
    gradients_none = gradients_none and all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters())
    _require(runtime_unchanged and evaluator_unchanged and gradients_none, "frozen model/evaluator changed or acquired gradients in preflight")
    payload = {
        "task_id": TASK_ID, "saved_baseline_positions": saved_positions,
        "saved_baseline": baseline, "block_first_seed": block,
        "joint_first_seed": joint, "joint_global_phase": phase,
        "joint_half_amplitude": scaled,
    }
    _torch_write_new(output_dir / "preflight.pt", payload)
    reloaded = torch.load(output_dir / "preflight.pt", map_location="cpu", weights_only=True)
    _require(torch.equal(reloaded["saved_baseline"]["raw_cmos"], baseline["raw_cmos"]), "preflight save/reload did not preserve baseline output")
    return {
        "task_id": TASK_ID,
        "status": "READY",
        "forward_rows": 50,
        "checks": {
            "saved_baseline_raw_and_view_replay": {"executed": True, "passed": True, "rows": 10, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL},
            "block_and_joint_first_seed_forward": {"executed": True, "passed": bool(torch.isfinite(block["raw_cmos"]).all() and torch.isfinite(joint["raw_cmos"]).all()), "rows": 20},
            "global_phase_shape_invariance": {"executed": True, "passed": True, "max_abs_difference": float((joint_shape - phase_shape).abs().max()), "tolerance": 1.0e-8},
            "half_amplitude_shape_and_energy_scaling": {"executed": True, "passed": True, "scale": 0.5, "intensity_factor": 0.25, "max_shape_abs_difference": float((joint_shape - scaled_shape).abs().max())},
            "frozen_runtime_evaluator_and_empty_gradients": {"executed": True, "passed": True},
            "output_save_reload": {"executed": True, "passed": True},
        },
        "model_updates": 0,
        "optimizer_updates": 0,
        "gradient_calls": 0,
    }


def _infer_banks(
    inputs: Mapping[str, Any], output_dir: Path, device: torch.device,
    runtime: torch.nn.Module, evaluator: torch.nn.Module,
    banks: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    for arm in ("block", "joint"):
        for seed_index, seed in enumerate(SEEDS):
            bank = banks[arm][seed_index].reshape(CLASS_COUNT * SAMPLE_COUNT_PER_CLASS, PORTS)
            arm_dir = output_dir / "readouts" / arm / f"seed_{seed}"
            for chunk_index, start in enumerate(range(0, bank.shape[0], CHUNK_ROWS)):
                stop = min(start + CHUNK_ROWS, bank.shape[0])
                result = frozen._infer_rows(runtime, evaluator, bank[start:stop], device, inputs["pixel_area_m2"])
                row_ids = []
                for flat_index in range(start, stop):
                    digit, sample_index = divmod(flat_index, SAMPLE_COUNT_PER_CLASS)
                    row = {
                        "arm": arm, "seed": seed, "label": digit,
                        "draw_index": sample_index,
                        "counterfactual_pair_id": f"{seed}:{digit}:{sample_index}",
                        "raw_clip_fraction": float(result["raw_clip_fraction"][flat_index - start]),
                        "camera_integrated_intensity": float(result["camera_integrated_intensity"][flat_index - start]),
                        "generator_input_power": float(result["generator_input_power"][flat_index - start]),
                        "e_d_prediction": int(result["e_d_logits"][flat_index - start].argmax()),
                        "e_d_correct": int(int(result["e_d_logits"][flat_index - start].argmax()) == digit),
                    }
                    index_rows.append(row)
                    row_ids.append(row)
                _torch_write_new(arm_dir / f"chunk_{chunk_index:04d}.pt", {
                    "task_id": TASK_ID,
                    "arm": arm,
                    "seed": seed,
                    "flat_start": start,
                    "rows": row_ids,
                    "raw_cmos": result["raw_cmos"],
                    "evaluator_view": result["evaluator_view"],
                    "e_d_logits": result["e_d_logits"],
                    "generator_input_power": result["generator_input_power"],
                    "camera_integrated_intensity": result["camera_integrated_intensity"],
                    "raw_clip_fraction": result["raw_clip_fraction"],
                })
                print(f"decoded {arm} seed={seed} rows {start}:{stop}", flush=True)
    return index_rows


def _load_scenario_features(output_dir: Path, arm: str, seed: int, shape_sizes: tuple[int, ...] = (28, 200)) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    features: dict[int, dict[int, list[torch.Tensor]]] = {size: {digit: [] for digit in range(10)} for size in shape_sizes}
    logits: dict[int, list[torch.Tensor]] = {digit: [] for digit in range(10)}
    metrics: dict[int, dict[str, list[torch.Tensor]]] = {digit: {name: [] for name in ("camera_integrated_intensity", "generator_input_power", "raw_clip_fraction")} for digit in range(10)}
    arm_dir = output_dir / "readouts" / arm / f"seed_{seed}"
    for chunk_path in sorted(arm_dir.glob("chunk_*.pt")):
        payload = torch.load(chunk_path, map_location="cpu", weights_only=True)
        _require(payload.get("task_id") == TASK_ID and payload.get("arm") == arm and payload.get("seed") == seed, "saved readout chunk identity differs")
        raw = payload["raw_cmos"]
        _require(tuple(raw.shape) == (50, 1, 200, 200), "saved joint-prior raw readout chunk geometry differs")
        for size in shape_sizes:
            view = frozen._shape_vectors(raw, size)
            for local, row in enumerate(payload["rows"]):
                features[size][int(row["label"])].append(view[local:local+1])
        for local, row in enumerate(payload["rows"]):
            digit = int(row["label"])
            logits[digit].append(payload["e_d_logits"][local:local+1])
            for name in metrics[digit]:
                metrics[digit][name].append(payload[name][local:local+1])
    assembled = {size: {digit: torch.cat(values, dim=0) for digit, values in by_class.items()} for size, by_class in features.items()}
    assembled_logits = {digit: torch.cat(values, dim=0) for digit, values in logits.items()}
    assembled_metrics = {digit: {name: torch.cat(values, dim=0) for name, values in fields.items()} for digit, fields in metrics.items()}
    for size in shape_sizes:
        _require(all(tuple(assembled[size][digit].shape) == (30, size * size) for digit in range(10)), "formal readout features are incomplete")
    return assembled, assembled_logits, assembled_metrics


def _validate_chunk_payload(payload: Mapping[str, Any], arm: str, seed: int, chunk_index: int) -> None:
    start = chunk_index * CHUNK_ROWS
    _require(payload.get("task_id") == TASK_ID and payload.get("arm") == arm and payload.get("seed") == seed, "saved readout chunk identity differs")
    _require(payload.get("flat_start") == start, "saved readout chunk start index differs")
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == CHUNK_ROWS, "saved readout chunk row metadata is incomplete")
    for local, row in enumerate(rows):
        flat = start + local
        digit, draw = divmod(flat, SAMPLE_COUNT_PER_CLASS)
        _require(row.get("arm") == arm and row.get("seed") == seed and row.get("label") == digit and row.get("draw_index") == draw, "saved readout row mapping differs from class/draw order")
        _require(row.get("counterfactual_pair_id") == f"{seed}:{digit}:{draw}", "saved common-random-number pair ID differs")
    raw = payload.get("raw_cmos")
    view = payload.get("evaluator_view")
    logits = payload.get("e_d_logits")
    _require(isinstance(raw, torch.Tensor) and tuple(raw.shape) == (CHUNK_ROWS, 1, 200, 200), "saved raw CMOS chunk geometry differs")
    _require(isinstance(view, torch.Tensor) and tuple(view.shape) == (CHUNK_ROWS, 1, 28, 28), "saved production view chunk geometry differs")
    _require(isinstance(logits, torch.Tensor) and tuple(logits.shape) == (CHUNK_ROWS, 10), "saved E_D readout chunk geometry differs")
    for name, value in (("raw CMOS", raw), ("production view", view), ("E_D logits", logits)):
        _require(bool(torch.isfinite(value).all()), f"saved {name} chunk contains nonfinite values")
    _require(bool((raw >= 0.0).all() and (raw.flatten(1).sum(1) > 0.0).all()), "saved raw CMOS includes negative or zero-mass images")
    replayed = generation.capacity.canonical.cmos_view(raw)
    _require(torch.allclose(replayed, view, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "saved production view does not reproduce from raw CMOS")
    for name in ("generator_input_power", "camera_integrated_intensity", "raw_clip_fraction"):
        value = payload.get(name)
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == (CHUNK_ROWS,) and bool(torch.isfinite(value).all()), f"saved {name} vector is invalid")
    _require(bool((payload["generator_input_power"] > 0.0).all() and (payload["camera_integrated_intensity"] > 0.0).all()), "saved output contains nonpositive energy")
    _require(bool(((payload["raw_clip_fraction"] >= 0.0) & (payload["raw_clip_fraction"] <= 1.0)).all()), "saved clipping fraction is outside [0,1]")


def _validate_readout_chunks(output_dir: Path) -> dict[str, Any]:
    chunk_count = 0
    for arm in ("block", "joint"):
        for seed in SEEDS:
            arm_dir = output_dir / "readouts" / arm / f"seed_{seed}"
            paths = sorted(arm_dir.glob("chunk_*.pt"))
            _require(len(paths) == 6, f"{arm} seed {seed} does not have six 50-row chunks")
            for chunk_index, path in enumerate(paths):
                payload = torch.load(path, map_location="cpu", weights_only=True)
                _validate_chunk_payload(payload, arm, seed, chunk_index)
                chunk_count += 1
    _require(chunk_count == 60, "formal readout does not contain exactly 60 chunks")
    return {"passed": True, "chunks": chunk_count, "rows": chunk_count * CHUNK_ROWS, "raw_cmos_and_production_view_replayed": True}


def _target_features(inputs: Mapping[str, Any]) -> dict[int, dict[str, dict[int, torch.Tensor]]]:
    readout = inputs["readout"]
    labels = inputs["labels"]
    targets: dict[int, dict[str, dict[int, torch.Tensor]]] = {size: {"fit": {}, "confirmation": {}} for size in (28, 200)}
    for role, positions in (("fit", torch.arange(0, FIT_ROWS)), ("confirmation", torch.arange(FIT_ROWS, TOTAL_ROWS))):
        role_raw = readout["target_slm_intensity"].index_select(0, positions)
        role_labels = labels.index_select(0, positions)
        for digit in range(10):
            selected = role_raw[role_labels == digit]
            expected = 20 if role == "fit" else 30
            _require(selected.shape[0] == expected, f"{role} target class {digit} count differs")
            for size in (28, 200):
                targets[size][role][digit] = frozen._shape_vectors(selected, size)
    return targets


def _diversity(x: torch.Tensor) -> float:
    distances = torch.cdist(x.to(torch.float64), x.to(torch.float64), p=2)
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool)
    return float(distances[mask].mean())


def _summarize(
    inputs: Mapping[str, Any], output_dir: Path, banks: Mapping[str, torch.Tensor],
    priors: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = _target_features(inputs)
    class_rows: list[dict[str, Any]] = []
    aux: dict[str, Any] = {}
    fit_x = _to_interleaved(inputs["readout"]["z"][:FIT_ROWS].to(torch.complex128))
    fit_labels = inputs["labels"][:FIT_ROWS]
    for arm in ("block", "joint"):
        for seed_index, seed in enumerate(SEEDS):
            features, logits, metrics = _load_scenario_features(output_dir, arm, seed)
            per_class_aux = {}
            nearest_by_class = {}
            for digit in range(10):
                for role in ("confirmation", "fit"):
                    n_target = 30 if role == "confirmation" else 20
                    for size in (28, 200):
                        distance = frozen._energy_distance(features[size][digit], targets[size][role][digit])
                        class_rows.append({
                            "arm": arm, "seed": seed, "label": digit, "target_role": role,
                            "shape_size": size, "generated_rows": SAMPLE_COUNT_PER_CLASS,
                            "target_rows": n_target, "energy_distance": distance,
                        })
                output = features[28][digit]
                class_rows.append({
                    "arm": arm, "seed": seed, "label": digit, "target_role": "output_only",
                    "shape_size": 28, "generated_rows": SAMPLE_COUNT_PER_CLASS,
                    "target_rows": 0, "energy_distance": _diversity(output),
                })
                prediction = logits[digit].argmax(1)
                row_metrics = metrics[digit]
                raw_x = _to_interleaved(banks[arm][seed_index, digit].to(torch.complex128))
                standardized_x = (raw_x - priors["means"][digit]) / priors["scales"][digit]
                fit_standardized = (fit_x[fit_labels == digit] - priors["means"][digit]) / priors["scales"][digit]
                nearest = torch.cdist(standardized_x, fit_standardized, p=2).min(dim=1).values
                nearest_by_class[str(digit)] = {"mean": float(nearest.mean()), "minimum": float(nearest.min()), "maximum": float(nearest.max()), "space": "featurewise standardized interleaved Re/Im Euclidean"}
                per_class_aux[str(digit)] = {
                    "e_d_correct": int((prediction == digit).sum()),
                    "e_d_total": 30,
                    "mean_camera_integrated_intensity": float(row_metrics["camera_integrated_intensity"].mean()),
                    "mean_generator_input_power": float(row_metrics["generator_input_power"].mean()),
                    "mean_raw_clipped_fraction": float(row_metrics["raw_clip_fraction"].mean()),
                    "normalized_28_shape_diversity": _diversity(output),
                    "mean_nearest_fit_sample_distance": nearest_by_class[str(digit)]["mean"],
                }
            aux[f"{arm}_{seed}"] = {"by_class": per_class_aux, "nearest_fit_sample_by_class": nearest_by_class}
    return class_rows, aux


def _main_decision(class_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in class_rows if row["target_role"] == "confirmation" and row["shape_size"] == 28]
    _require(len(selected) == 2 * len(SEEDS) * CLASS_COUNT, "primary distance table lacks seed×class observations")
    selected_keys = {(row["arm"], row["seed"], row["label"]) for row in selected}
    expected_keys = {(arm, seed, digit) for arm in ("block", "joint") for seed in SEEDS for digit in range(CLASS_COUNT)}
    _require(selected_keys == expected_keys, "primary distance table has missing or duplicated arm/seed/class rows")
    _require(all(math.isfinite(float(row["energy_distance"])) for row in selected), "primary distance table contains a nonfinite value")
    by_arm = {arm: [float(row["energy_distance"]) for row in selected if row["arm"] == arm] for arm in ("block", "joint")}
    means = {arm: math.fsum(values) / len(values) for arm, values in by_arm.items()}
    difference = means["joint"] - means["block"]
    tolerance = TOLERANCE_ATOL + TOLERANCE_RTOL * max(abs(means["joint"]), abs(means["block"]))
    if difference < -tolerance:
        classification = "JOINT_PRIOR_NUMERIC_IMPROVEMENT"
    elif difference > tolerance:
        classification = "JOINT_PRIOR_NUMERIC_DETERIORATION"
    else:
        classification = "NO_GAIN_BEYOND_NUMERICAL_TOLERANCE"
    by_seed = {}
    for seed in SEEDS:
        for arm in ("block", "joint"):
            values = [float(row["energy_distance"]) for row in selected if row["seed"] == seed and row["arm"] == arm]
            _require(len(values) == CLASS_COUNT, f"seed {seed}/{arm} class summary is incomplete")
            by_seed.setdefault(str(seed), {})[arm] = math.fsum(values) / len(values)
        by_seed[str(seed)]["joint_minus_block"] = by_seed[str(seed)]["joint"] - by_seed[str(seed)]["block"]
    return {
        "classification": classification,
        "primary_effect_joint_minus_block": difference,
        "tolerance": tolerance,
        "mean_energy_distance_by_arm": means,
        "seed_effects": by_seed,
        "seed_effect_range": {"minimum": min(v["joint_minus_block"] for v in by_seed.values()), "maximum": max(v["joint_minus_block"] for v in by_seed.values())},
        "class_and_seed_weighting": "ten classes and five fixed seeds equally weighted",
        "interval": None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(rows and not path.exists(), f"cannot write CSV or file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recheck_outputs(output_dir: Path, class_rows: list[dict[str, Any]], decision: Mapping[str, Any], expected_rows: int) -> dict[str, Any]:
    with (output_dir / "paired_rows.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        paired = list(csv.DictReader(stream))
    _require(len(paired) == expected_rows == 3000, "paired CSV does not have exactly 3000 generated image rows")
    _require(len({row["arm"] + row["counterfactual_pair_id"] for row in paired}) == expected_rows, "paired CSV contains duplicate arm/sample identities")
    paired_counts: dict[tuple[str, int, int], int] = {}
    for row in paired:
        arm, seed, digit, draw = row["arm"], int(row["seed"]), int(row["label"]), int(row["draw_index"])
        _require(arm in {"block", "joint"} and seed in SEEDS and 0 <= digit < 10 and 0 <= draw < 30, "paired CSV contains an invalid arm/seed/class/draw key")
        _require(row["counterfactual_pair_id"] == f"{seed}:{digit}:{draw}", "paired CSV common-noise identity differs")
        key = (arm, seed, digit)
        paired_counts[key] = paired_counts.get(key, 0) + 1
        for field in ("raw_clip_fraction", "camera_integrated_intensity", "generator_input_power"):
            _require(math.isfinite(float(row[field])), f"paired CSV field {field} is nonfinite")
    _require(set(paired_counts) == {(arm, seed, digit) for arm in ("block", "joint") for seed in SEEDS for digit in range(10)} and all(count == 30 for count in paired_counts.values()), "paired CSV arm/seed/class balance differs")
    pair_ids = {arm: {row["counterfactual_pair_id"] for row in paired if row["arm"] == arm} for arm in ("block", "joint")}
    _require(pair_ids["block"] == pair_ids["joint"] and len(pair_ids["block"]) == 1500, "common-random-number arm pairing is incomplete")
    with (output_dir / "class_distances.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        observed = list(csv.DictReader(stream))
    _require(len(observed) == len(class_rows), "class distance CSV row count changed")
    parsed = [{"arm": row["arm"], "seed": int(row["seed"]), "label": int(row["label"]), "target_role": row["target_role"], "shape_size": int(row["shape_size"]), "energy_distance": float(row["energy_distance"])} for row in observed]
    expected_distance_keys = {
        (arm, seed, digit, role, size)
        for arm in ("block", "joint") for seed in SEEDS for digit in range(CLASS_COUNT)
        for role, size in (("confirmation", 28), ("confirmation", 200), ("fit", 28), ("fit", 200), ("output_only", 28))
    }
    observed_distance_keys = {(row["arm"], row["seed"], row["label"], row["target_role"], row["shape_size"]) for row in parsed}
    _require(len(parsed) == len(expected_distance_keys) and observed_distance_keys == expected_distance_keys, "class distance CSV has missing or duplicate seed/class/role/view rows")
    recomputed = _main_decision(parsed)
    _require(recomputed["classification"] == decision["classification"], "CSV-recomputed classification differs from the in-memory decision")
    _require(math.isclose(recomputed["primary_effect_joint_minus_block"], decision["primary_effect_joint_minus_block"], rel_tol=0.0, abs_tol=1.0e-14), "CSV-recomputed primary effect differs")
    _require(math.isclose(recomputed["tolerance"], decision["tolerance"], rel_tol=0.0, abs_tol=1.0e-14), "CSV-recomputed tolerance differs")
    return {"passed": True, "paired_rows": len(paired), "paired_seed_class_draw_ids": 1500, "class_distance_rows": len(parsed), "primary_decision": recomputed}


def _report_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    scope = report["scope"]
    return f"""# 冻结 A4700 联合先验对照

## 结果与状态

正式数值测量状态：`{report['status']}`；唯一判读：`{decision['classification']}`。确认人口形态能量距离差（联合减分块）为 **{decision['primary_effect_joint_minus_block']:.9g}**，数值容差为 **{decision['tolerance']:.9g}**。分块均值 {decision['mean_energy_distance_by_arm']['block']:.9g}；联合均值 {decision['mean_energy_distance_by_arm']['joint']:.9g}。

## 科学认识

结果仅说明指定50%收缩联合高斯候选相对分块高斯，在当前冻结A4700解码器、固定拟合人口及五个种子下是否改善类别形态集合距离。种子效应范围为 {decision['seed_effect_range']['minimum']:.9g} 到 {decision['seed_effect_range']['maximum']:.9g}。它不证明人眼可读性、完整新颖生成、独立泛化或联合先验普遍优越。

## 目标缺口

拟合数据和目标均来自 canonical TRAIN；300行确认人口不是独立测试集。每个采样输出没有单张真实配对目标，因此没有逐图配对Q。辅助分类读数不是可读性证据。此结果不补足此前单人审阅质控，也不授权模型更新或候选采用。

## 工程完成与验证

核验A4700检查点、500行源人口与物理坐标；十类协方差边缘块一致，标准化Cholesky重建误差上限 {report['validation']['maximum_standardized_cholesky_reconstruction_error']:.3g}。50行CUDA预检、两臂各1500行正式解码、冻结状态和保存重读CSV核对均通过。正式前向 {scope['formal_forward_rows']} 行；模型更新、优化器更新、梯度调用均为0。

## 决策影响与下一步

本次对照到此停止。`training_recommendation=NONE`、`next_action=NONE`；不自动采用联合候选、不训练或启动后继实验。独立的100题单人稳定性包状态为等待评分。
"""


def _run(artifact_arg: Path, checkpoint_arg: Path, output_arg: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    started = time.perf_counter()
    output_dir = _prepare_output(root, output_arg)
    with (output_dir / "experiment_plan.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_make_plan())
    artifact_dir = _resolve(root, artifact_arg)
    inputs = frozen._load_inputs(root, artifact_dir, checkpoint_arg)
    readout = inputs["readout"]
    labels = inputs["labels"]
    _require(torch.equal(labels[:FIT_ROWS], torch.tensor(inputs["population"]["labels"][:FIT_ROWS])), "saved population labels differ from the readout")
    priors = _fit_class_priors(readout["z"][:FIT_ROWS].to(torch.complex128), labels[:FIT_ROWS])
    banks, noises = _sample_banks(priors)
    _torch_write_new(output_dir / "fitted_priors.pt", priors)
    _torch_write_new(output_dir / "sampling_noise_and_banks.pt", {"task_id": TASK_ID, "seeds": list(SEEDS), "noise": noises, "block": banks["block"], "joint": banks["joint"]})
    prior_metadata = {
        "task_id": TASK_ID, "fit_rows": FIT_ROWS, "fit_rows_per_class": 20,
        "coordinate_order": "Re(z1), Im(z1), ..., Re(z100), Im(z100)",
        "covariance_divisor": 20, "shrinkage": SHRINKAGE,
        "standardized_cholesky_reconstruction_error_by_class": priors["reconstruction_error"].tolist(),
        "maximum_standardized_cholesky_reconstruction_error": float(priors["reconstruction_error"].max()),
        "minimum_standardized_covariance_eigenvalue_by_class": priors["minimum_eigenvalue"].tolist(),
        "same_class_means_and_within_port_2x2_covariance": True,
        "variance_floor_or_jitter": False,
        "formal_seeds": list(SEEDS),
        "paired_standard_normal_noise_per_seed_shape": [10, 30, 200],
    }
    _json_write_new(output_dir / "fitted_priors.json", prior_metadata)
    _json_write_new(output_dir / "input_population.json", inputs["population"])

    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized formal frozen production-path measurement")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    disk_free = shutil.disk_usage(output_dir).free
    _require(free_bytes > 0 and total_bytes > 0, "CUDA reports no usable memory")
    estimated_bytes = len(SEEDS) * 2 * CLASS_COUNT * SAMPLE_COUNT_PER_CLASS * (200 * 200 * 4 + 28 * 28 * 4 + 10 * 4 + 3 * 8)
    _require(disk_free > estimated_bytes * 1.05, "insufficient free disk for the complete fixed readout population")
    runtime = generation._new_runtime(inputs["checkpoint"], device)
    evaluators = generation.diagnostic.load_evaluators(inputs["generation_paths"], device)
    evaluator = evaluators["E_D"]
    preflight = _preflight(inputs, output_dir, device, runtime, evaluator, banks)
    preflight["resources_observed"] = {
        "cuda_device": str(device), "cuda_total_bytes": int(total_bytes), "cuda_free_bytes_before_formal": int(free_bytes),
        "disk_free_bytes_before_formal": int(disk_free), "estimated_readout_bytes": estimated_bytes,
        "memory_and_elapsed_time_are_observations_only": True,
    }
    _json_write_new(output_dir / "preflight.json", preflight)
    _require(preflight["status"] == "READY" and all(check["executed"] and check["passed"] for check in preflight["checks"].values()), "production preflight did not produce READY")

    runtime_before = generation._snapshot_state(runtime)
    evaluator_before = generation.capacity._state_snapshot(evaluator)
    row_index = _infer_banks(inputs, output_dir, device, runtime, evaluator, banks)
    runtime_unchanged = generation._tree_equal(runtime_before, generation._snapshot_state(runtime))
    evaluator_unchanged = generation._tree_equal(evaluator_before, generation.capacity._state_snapshot(evaluator))
    gradients_none = all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters()) and all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters())
    _require(runtime_unchanged and evaluator_unchanged and gradients_none, "frozen A4700 or evaluator changed during formal measurement")
    readout_validation = _validate_readout_chunks(output_dir)
    _write_csv(output_dir / "paired_rows.csv", row_index)
    class_rows, auxiliary = _summarize(inputs, output_dir, banks, priors)
    _write_csv(output_dir / "class_distances.csv", class_rows)
    decision = _main_decision(class_rows)
    validation = _recheck_outputs(output_dir, class_rows, decision, len(row_index))
    _require(validation["passed"], "saved CSV recheck failed")
    validation_record = {
        "task_id": TASK_ID, "status": "VALIDATED", "source_identity_and_checkpoint": True,
        "physical_coordinate_and_source_readout_replayed": True,
        "all_source_intensities_finite_nonnegative_positive_mass": True,
        "covariance_blocks_and_cholesky_checked": True,
        "maximum_standardized_cholesky_reconstruction_error": float(priors["reconstruction_error"].max()),
        "preflight_ready": True, "formal_readouts": readout_validation,
        "frozen_runtime_and_evaluator": True, "saved_csv_recheck": validation,
    }
    _json_write_new(output_dir / "validation.json", validation_record)
    report = {
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "decision": decision,
        "scope": {
            "source_task": frozen.parent_task.TASK_ID,
            "source_attempt": artifact_dir.as_posix(),
            "checkpoint": inputs["checkpoint_path"].as_posix(),
            "checkpoint_step": 4700,
            "population": {"fit": 200, "confirmation": 300, "total": 500, "fit_per_class": 20, "confirmation_per_class": 30},
            "all_rows_canonical_train": True,
            "formal_forward_rows": 3000,
            "model_updates": 0, "optimizer_updates": 0, "gradient_calls": 0,
            "human_ratings": 0,
        },
        "measurement": {
            "arms": ["block", "joint"], "shrinkage": SHRINKAGE,
            "common_random_numbers": True, "seeds": list(SEEDS),
            "rows_per_seed_per_arm": 300, "rows_per_class_seed_arm": SAMPLE_COUNT_PER_CLASS,
            "primary_shape": "Pool28(raw CMOS), normalize each 784-vector by its sum; no clipping",
            "primary_distance": "2*mean Euclidean(X,T)-mean Euclidean(X,X)-mean Euclidean(T,T); V statistic includes diagonals",
            "target_roles": "confirmation is primary; fit is auxiliary, both are TRAIN",
            "tolerance": {"absolute": TOLERANCE_ATOL, "relative": TOLERANCE_RTOL},
            "formal_precision": "strict FP32 optical forward; float64 prior/statistics",
            "outputs_have_no_single_image_target_pair": True,
        },
        "preflight": preflight,
        "validation": validation_record,
        "recovered_engineering_incidents": [],
        "secondary": {
            "class_distances": "class_distances.csv includes 28×28 and 200×200 confirmation/fit target distances and output-only 28×28 diversity",
            "per_seed_effects": decision["seed_effects"],
            "per_seed_and_class_auxiliary": auxiliary,
            "E_D_is_auxiliary_not_human_readability": True,
            "no_independent_confidence_interval_claim": True,
        },
        "artifacts": {
            "plan": "experiment_plan.md", "input_population": "input_population.json",
            "fitted_priors": "fitted_priors.pt", "prior_metadata": "fitted_priors.json",
            "sampling_noise_and_banks": "sampling_noise_and_banks.pt",
            "preflight": "preflight.json", "preflight_readouts": "preflight.pt",
            "validation": "validation.json",
            "readouts": "readouts/<arm>/seed_<seed>/chunk_NNNN.pt",
            "paired_rows": "paired_rows.csv", "class_distances": "class_distances.csv",
            "report": "report.json", "report_markdown": "report.md", "completed": "completed.json",
        },
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "training_recommendation": "NONE", "next_action": "NONE",
    }
    _json_write_new(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8", newline="\n")
    _json_write_new(output_dir / "completed.json", {
        "task_id": TASK_ID, "status": "COMPLETED", "classification": decision["classification"],
        "primary_effect": decision["primary_effect_joint_minus_block"],
        "training_recommendation": "NONE", "next_action": "NONE",
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
    output_dir = _resolve(Path(__file__).resolve().parents[3], args.output_dir)
    try:
        report = _run(args.artifact_dir, args.checkpoint, args.output_dir)
    except Exception as error:
        if output_dir.is_dir() and not (output_dir / "report.json").exists():
            try:
                _json_write_new(output_dir / "failure.json", {
                    "task_id": TASK_ID, "status": "ENGINEERING_ERROR",
                    "error_type": type(error).__name__, "error": str(error),
                    "scientific_classification": None, "training_recommendation": "NONE", "next_action": "NONE",
                })
            except Exception:
                pass
        raise
    print(json.dumps({"status": report["status"], "classification": report["classification"], "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
