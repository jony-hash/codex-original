"""Compare fixed-joint-prior A5400/B5400 readability and prepare a blind review."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import audit_lightgen_2d_generator_cross_port_joint_structure_v1 as frozen
from scripts.research.lightgen import audit_lightgen_2d_generator_frozen_joint_prior_comparison_v1 as prior_task
from scripts.research.lightgen import audit_lightgen_2d_generator_frozen_readability_screen_v1 as review
from scripts.research.lightgen import execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as parent
from scripts.research.lightgen import execute_lightgen_2d_generator_foreground_objective_training_comparison_v1 as training
from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation


TASK_ID = "M0_GENERATOR_FIXED_PRIOR_SAMPLED_READABILITY_V1"
SOURCE_DIR = Path("outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03")
TRAINING_DIR = Path("outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01")
PRIOR_DIR = Path("outputs/20261009_m0_generator_frozen_joint_prior_comparison_v1/runs/attempt_02")
DEFAULT_OUTPUT = Path("outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01")
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = 500
CLASS_COUNT = 10
PORTS = 100
SEEDS = tuple(range(2026100961, 2026100966))
DRAW_COUNT = 30
FORMAL_ROWS_PER_ARM = 1500
CHUNK_ROWS = 50
SELECTION_SEED = 2026100991
REPEAT_SEED = 2026100992
ORDER_SEED = 2026100993
PRIMARY_PAIR_COUNT = 100
REPEATS_PER_ARM = 20
REAL_CONTROL_COUNT = 20
BLACK_CONTROL_COUNT = 10
ITEM_COUNT = 270
PAGE_SIZE = 45
MIN_SPACING = 50
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
SHAPE_ATOL = 1.0e-8
SCALE_ATOL = 2.0e-10
ALPHA = 0.05
ALLOWED_ANSWERS = {str(digit) for digit in range(10)} | {"unclear"}
STRICT_FP32_SETTINGS = {
    "dtype": "float32",
    "autocast": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
}


class ReadabilityComparisonError(RuntimeError):
    """Invalid source, preflight, or measurement evidence prevents a result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadabilityComparisonError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _json_read(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON evidence must be an object: {path.name}")
    return value


def _json_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".pending")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_write_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".pending")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_write_new(path: Path, payload: Any) -> None:
    _require(not path.exists(), f"refusing to overwrite tensor evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".pending")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    _require(bool(rows) and not path.exists(), f"cannot write empty or existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or rows[0].keys())
    temporary = path.with_name("." + path.name + ".pending")
    try:
        with temporary.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _tree_equal(left: Any, right: Any) -> bool:
    return training._tree_equal(left, right)


def _prepare_output(root: Path, output_arg: Path, *, resume: bool) -> Path:
    output_root = (root / "outputs").resolve()
    target = _resolve(root, output_arg)
    _require(target != output_root and output_root in target.parents, "output must be a child of outputs/")
    parts = target.relative_to(output_root).parts
    _require(
        len(parts) == 3 and parts[0].startswith("20261009_") and parts[1] == "runs" and parts[2].startswith("attempt_"),
        "output must use outputs/YYYYMMDD_<version>/runs/attempt_N",
    )
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected output namespace is forbidden")
    if resume:
        _require(target.is_dir(), "--resume requires the explicit existing attempt directory")
    else:
        _require(not target.exists(), f"output already exists; refusing to overwrite: {target}")
        target.mkdir(parents=True, exist_ok=False)
    return target


def _plan_text() -> str:
    return """# 固定联合先验下的 A5400/B5400 采样与单人识读对照

唯一人工判读比较：相同的 1500 个复 OLS 联合先验输入，分别由冻结 A5400 与 B5400 解码后，B 是否更常被单人审阅者认作请求数字。条件采样图没有单张目标配对，不计算逐图 Q、前景误差或 SSIM。数值形态集合距离仅作辅助。

先核验 attempt_03 的 500 行来源、attempt_01 两端点、attempt_02 保存的联合先验库及已保存 A4700 读出。90 行严格 FP32 CUDA 预检通过并形成 READY 后，解码两臂各 1500 行，50 行分块保存。随后按固定种子选取 100 个同输入配对、两端点各 20 个隐藏重复、20 个真实数字正控及 10 个全黑负控，形成六组各 45 题的匿名盲判页面。

人工检查为真实数字至少 18/20、全黑至少 9/10、A 与 B 重复分别至少 18/20。通过后唯一判读使用 100 个同输入配对的请求类别认可差和双侧精确 McNemar 检验；未通过则为 REVIEW_INCONCLUSIVE。数值改善不替代人工判断。

所有来源是 canonical TRAIN。结果仅适用于固定先验、A5400/B5400 两端点和这位审阅者；不证明独立泛化、覆盖、新颖性或模型采用。本任务结束于评分归档，不继续训练或调参。
"""


def _validate_population(population: Mapping[str, Any]) -> None:
    _require(population.get("task_id") == parent.TASK_ID, "attempt_03 population task identity differs")
    _require(len(population.get("source_row_indices", [])) == TOTAL_ROWS, "source population must contain 500 rows")
    _require(len(set(population["source_row_indices"])) == TOTAL_ROWS, "source population contains duplicate source rows")
    _require(population.get("role_positions") == {"fit": [0, 200], "confirmation": [200, 500]}, "fit/confirmation population boundary differs")
    _require(population.get("roles") == ["fit"] * FIT_ROWS + ["confirmation"] * CONFIRM_ROWS, "population roles differ from the fixed plan")
    labels = torch.tensor(population.get("labels", []), dtype=torch.long)
    _require(tuple(labels.shape) == (TOTAL_ROWS,) and bool(((labels >= 0) & (labels <= 9)).all()), "population labels are invalid")
    _require(torch.equal(torch.bincount(labels[:FIT_ROWS], minlength=10), torch.full((10,), 20)), "fit population is not 20 rows per class")
    _require(torch.equal(torch.bincount(labels[FIT_ROWS:], minlength=10), torch.full((10,), 30)), "confirmation population is not 30 rows per class")


def _load_endpoint_chunks(training_dir: Path, arm: str, inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    base = training_dir / "readouts" / f"{arm}5400"
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in ("positions", "source_row_indices", "labels", "raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")}
    expected_rows = torch.tensor(inputs["population"]["source_row_indices"], dtype=torch.long)
    expected_labels = inputs["labels"]
    for index in range(10):
        path = base / f"chunk_{index:04d}.pt"
        _require(path.is_file(), f"saved {arm}5400 readout chunk is missing: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        start = index * 50
        _require(payload.get("task_id") == training.TASK_ID and payload.get("scenario") == f"{arm}5400", "saved endpoint chunk identity differs")
        _require(torch.equal(payload.get("positions"), torch.arange(start, start + 50)), "saved endpoint positions differ")
        _require(torch.equal(payload.get("source_row_indices"), expected_rows[start : start + 50]), "saved endpoint source row identities differ")
        _require(torch.equal(payload.get("labels"), expected_labels[start : start + 50]), "saved endpoint labels differ")
        _require(payload.get("roles") == inputs["roles"][start : start + 50], "saved endpoint roles differ")
        for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits"):
            value = payload.get(key)
            _require(isinstance(value, torch.Tensor) and value.shape[0] == 50, f"saved endpoint {key} rows differ")
            _require(bool(torch.isfinite(value).all()), f"saved endpoint {key} contains nonfinite values")
            pieces[key].append(value.contiguous())
        pieces["positions"].append(payload["positions"])
        pieces["source_row_indices"].append(payload["source_row_indices"])
        pieces["labels"].append(payload["labels"])
    result = {key: torch.cat(values, dim=0) for key, values in pieces.items()}
    _require(tuple(result["raw_cmos"].shape) == (TOTAL_ROWS, 1, 200, 200), f"{arm}5400 raw output geometry differs")
    _require(tuple(result["evaluator_view"].shape) == (TOTAL_ROWS, 1, 28, 28), f"{arm}5400 saved view geometry differs")
    _require(bool((result["raw_cmos"] >= 0.0).all()), f"{arm}5400 raw intensity is negative")
    _require(bool((result["raw_cmos"].flatten(1).sum(1) > 0.0).all()), f"{arm}5400 raw intensity has a zero-mass row")
    replay = generation.capacity.canonical.cmos_view(result["raw_cmos"])
    _require(torch.allclose(replay, result["evaluator_view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"{arm}5400 saved production views do not replay")
    _require(torch.equal(result["target_slm_intensity"], inputs["readout"]["target_slm_intensity"]), f"{arm}5400 target rows differ from the parent input")
    return result


def _validate_training_checkpoint(payload: Mapping[str, Any], path: Path, arm: str, inputs: Mapping[str, Any], training_dir: Path) -> None:
    _require(path.is_file() and path.resolve().parent == training_dir.resolve(), f"explicit {arm} checkpoint must be inside the named training attempt")
    _require(payload.get("schema") == training.CHECKPOINT_SCHEMA and payload.get("task_id") == training.TASK_ID, f"{arm} checkpoint identity differs")
    _require(payload.get("arm") == arm and payload.get("completed_updates") == 5400 and payload.get("optimizer_step") == 5400, f"{arm} checkpoint is not step 5400")
    _require(payload.get("source_step") == 4700 and payload.get("task_local_updates") == 700, f"{arm} checkpoint local update count differs")
    _require(payload.get("lambda") == training.FIXED_LAMBDA and payload.get("source_checkpoint") == inputs["checkpoint_path"].as_posix(), f"{arm} checkpoint source or foreground weight differs")
    _require(payload.get("source_population") == inputs["population"], f"{arm} checkpoint training population differs")
    _require(payload.get("source_history_count") == 4700 and len(payload.get("training_history", [])) == 700, f"{arm} checkpoint history is incomplete")
    _require(path.name == f"arm_{arm.lower()}_step5400.pt", f"explicit {arm} checkpoint filename differs")
    params = payload.get("trainable_parameter_state", {})
    expected_shapes = inputs["population"]["source_parameter_shapes"]
    _require(set(params) == set(expected_shapes), f"{arm} trainable phase state is incomplete")
    for name, shape in expected_shapes.items():
        _require(tuple(params[name].shape) == tuple(shape) and params[name].dtype == torch.float32 and bool(torch.isfinite(params[name]).all()), f"{arm} phase state is invalid: {name}")
    frozen_state = payload.get("frozen_runtime_state", {})
    _require(_tree_equal(frozen_state, inputs["checkpoint"]["frozen_runtime_state"]), f"{arm} frozen OLS/runtime differs from A4700")
    optimizer = payload.get("optimizer_state", {})
    states = optimizer.get("state", {})
    _require(len(optimizer.get("param_groups", [])) == 1 and set(states) == {0}, f"{arm} Adam state scope differs")
    _require(int(float(states[0]["step"].item())) == 5400, f"{arm} Adam state is not at step 5400")
    _require(bool(torch.isfinite(states[0]["exp_avg"]).all() and torch.isfinite(states[0]["exp_avg_sq"]).all()), f"{arm} Adam state is nonfinite")


def _verify_saved_joint_bank(prior_dir: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    report = _json_read(prior_dir / "report.json")
    metadata = _json_read(prior_dir / "fitted_priors.json")
    saved_population = _json_read(prior_dir / "input_population.json")
    _require(report.get("task_id") == prior_task.TASK_ID and report.get("status") == "VALID_COMPLETED", "named frozen-prior attempt is not validly complete")
    _require(saved_population == population, "saved prior fit population differs from attempt_03")
    _require(metadata.get("formal_seeds") == list(SEEDS) and metadata.get("shrinkage") == 0.5, "saved prior seed or shrinkage metadata differs")
    prior_payload = torch.load(prior_dir / "fitted_priors.pt", map_location="cpu", weights_only=True)
    bank_payload = torch.load(prior_dir / "sampling_noise_and_banks.pt", map_location="cpu", weights_only=True)
    _require(prior_payload and bank_payload.get("task_id") == prior_task.TASK_ID, "saved prior parameters or sampling bank are missing")
    _require(bank_payload.get("seeds") == list(SEEDS), "saved joint bank seed order differs")
    noise, joint = bank_payload.get("noise"), bank_payload.get("joint")
    _require(tuple(noise.shape) == (5, 10, 30, 200) and noise.dtype == torch.float64, "saved common noise geometry or dtype differs")
    _require(tuple(joint.shape) == (5, 10, 30, PORTS) and joint.is_complex() and joint.dtype == torch.complex64, "saved joint bank geometry or dtype differs")
    _require(bool(torch.isfinite(joint.real).all() and torch.isfinite(joint.imag).all()), "saved joint bank contains nonfinite coefficients")
    means, scales, cholesky = prior_payload["means"], prior_payload["scales"], prior_payload["joint_cholesky"]
    _require(tuple(means.shape) == (10, 200) and tuple(scales.shape) == (10, 200) and tuple(cholesky.shape) == (10, 200, 200), "saved joint-prior parameter geometry differs")
    for seed_index in range(5):
        for digit in range(10):
            correlated = noise[seed_index, digit] @ cholesky[digit].T
            real = means[digit] + correlated * scales[digit]
            rebuilt = torch.complex(real[:, 0::2], real[:, 1::2]).to(torch.complex64)
            _require(torch.equal(rebuilt, joint[seed_index, digit]), f"persisted seed/class joint input bank cannot be reproduced: {SEEDS[seed_index]}/{digit}")
    return {"report": report, "metadata": metadata, "priors": prior_payload, "bank": bank_payload}


def _load_inputs(root: Path, artifact_dir: Path, training_dir: Path, prior_dir: Path, checkpoint_a: Path, checkpoint_b: Path) -> dict[str, Any]:
    population = _json_read(artifact_dir / "input_population.json")
    _validate_population(population)
    energy_report = _json_read(artifact_dir / "report.json")
    _require(energy_report.get("task_id") == parent.TASK_ID and energy_report.get("status") == "VALID_COMPLETED", "attempt_03 energy comparison is not complete")
    inputs = frozen._load_inputs(root, artifact_dir, artifact_dir / "arm_a_step4700.pt")
    _require(inputs["population"] == population, "reconstructed energy population differs")
    training_report = _json_read(training_dir / "report.json")
    training_population = _json_read(training_dir / "input_population.json")
    _require(training_report.get("task_id") == training.TASK_ID and training_report.get("status") == "VALID_COMPLETED", "foreground training comparison is not validly complete")
    _require(training_population == population, "foreground training comparison population differs from attempt_03")
    a_path, b_path = checkpoint_a.resolve(), checkpoint_b.resolve()
    _require(a_path == (training_dir / "arm_a_step5400.pt").resolve(), "checkpoint A must be the named A5400 checkpoint")
    _require(b_path == (training_dir / "arm_b_step5400.pt").resolve(), "checkpoint B must be the named B5400 checkpoint")
    a_checkpoint = torch.load(a_path, map_location="cpu", weights_only=True)
    b_checkpoint = torch.load(b_path, map_location="cpu", weights_only=True)
    _validate_training_checkpoint(a_checkpoint, a_path, "A", inputs, training_dir)
    _validate_training_checkpoint(b_checkpoint, b_path, "B", inputs, training_dir)
    _require(_tree_equal(a_checkpoint["frozen_runtime_state"], b_checkpoint["frozen_runtime_state"]), "A5400/B5400 frozen optical/OLS state differs")
    _require(a_checkpoint.get("population") == b_checkpoint.get("population"), "A5400/B5400 source population differs")
    endpoint_a = _load_endpoint_chunks(training_dir, "A", inputs)
    endpoint_b = _load_endpoint_chunks(training_dir, "B", inputs)
    bank_data = _verify_saved_joint_bank(prior_dir, population)
    source_input = inputs["readout"]["z"].to(torch.complex64)
    return {
        **inputs,
        "training_dir": training_dir,
        "prior_dir": prior_dir,
        "checkpoint_a_path": a_path,
        "checkpoint_b_path": b_path,
        "checkpoint_a": a_checkpoint,
        "checkpoint_b": b_checkpoint,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "prior": bank_data,
        "source_input": source_input,
        "training_report": training_report,
        "energy_report": energy_report,
    }
def _load_seed_readout(target: Path, arm: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = []
    base = target / "readouts" / arm / f"seed_{seed}"
    seed_index = SEEDS.index(seed)
    for chunk_index, start in enumerate(range(0, CLASS_COUNT * DRAW_COUNT, CHUNK_ROWS)):
        path = base / f"chunk_{chunk_index:04d}.pt"
        _require(path.is_file(), f"formal {arm} readout is missing a chunk: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _require(payload.get("task_id") == TASK_ID and payload.get("arm") == arm and payload.get("seed") == seed and payload.get("flat_start") == start, "formal chunk identity/position differs")
        rows = payload.get("rows", [])
        _require(len(rows) == CHUNK_ROWS, "formal chunk row metadata is incomplete")
        for offset, row in enumerate(rows):
            flat_index = start + offset
            digit, draw = divmod(flat_index, DRAW_COUNT)
            _require(row.get("label") == digit and row.get("draw_index") == draw and row.get("seed") == seed and row.get("arm") == arm, "formal row label, seed, or draw index differs")
            _require(row.get("position") == seed_index * 300 + flat_index and row.get("pair_id") == f"{seed}:{digit}:{draw}", "formal row index/pair identity differs")
        chunks.append(payload)
    raw = torch.cat([chunk["raw_cmos"] for chunk in chunks], dim=0)
    view = torch.cat([chunk["evaluator_view"] for chunk in chunks], dim=0)
    input_z = torch.cat([chunk["input_z"] for chunk in chunks], dim=0)
    _require(tuple(raw.shape) == (300, 1, 200, 200) and tuple(view.shape) == (300, 1, 28, 28) and tuple(input_z.shape) == (300, PORTS), "formal readout rows differ")
    _require(bool(torch.isfinite(raw).all() and (raw >= 0.0).all() and (raw.flatten(1).sum(1) > 0.0).all()), "formal raw readout is invalid")
    _require(bool(torch.isfinite(input_z.real).all() and torch.isfinite(input_z.imag).all()), "formal OLS input contains nonfinite values")
    _require(torch.allclose(generation.capacity.canonical.cmos_view(raw), view, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "formal saved view differs from production view")
    return raw, view, input_z

def _mean_distance(left: torch.Tensor, right: torch.Tensor, row_chunk: int = 30) -> float:
    left64, right64 = left.to(torch.float64), right.to(torch.float64)
    _require(left64.ndim == right64.ndim == 2 and left64.shape[1] == right64.shape[1] and left64.shape[0] > 0 and right64.shape[0] > 0, "shape-distance vectors are invalid")
    mode = "use_mm_for_euclid_dist" if left64.shape[1] > 4096 else "donot_use_mm_for_euclid_dist"
    parts = []
    for start in range(0, left64.shape[0], row_chunk):
        distances = torch.cdist(left64[start : start + row_chunk], right64, p=2, compute_mode=mode)
        _require(bool(torch.isfinite(distances).all()), "shape distance contains nonfinite values")
        parts.append(float(distances.sum()))
    return math.fsum(parts) / (left64.shape[0] * right64.shape[0])


def _shape_group(raw: torch.Tensor, size: int) -> torch.Tensor:
    result = frozen._shape_vectors(raw, size)
    _require(result.dtype == torch.float64 and bool(torch.isfinite(result).all()), "normalized morphology vectors are invalid")
    return result


def _target_groups(inputs: Mapping[str, Any]) -> dict[str, dict[int, dict[int, torch.Tensor]]]:
    result: dict[str, dict[int, dict[int, torch.Tensor]]] = {"fit": {}, "confirmation": {}}
    target_raw = inputs["readout"]["target_slm_intensity"].to(torch.float32)
    labels = inputs["labels"]
    for role, start, stop in (("fit", 0, FIT_ROWS), ("confirmation", FIT_ROWS, TOTAL_ROWS)):
        for digit in range(CLASS_COUNT):
            positions = torch.tensor([i for i in range(start, stop) if int(labels[i]) == digit], dtype=torch.long)
            _require(positions.numel() == (20 if role == "fit" else 30), f"{role} target class {digit} count differs")
            rows = target_raw.index_select(0, positions)
            result[role][digit] = {size: _shape_group(rows, size) for size in (28, 200)}
    return result


def _compute_shape_statistics(target: Path, inputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = _target_groups(inputs)
    target_within = {
        role: {
            digit: {size: _mean_distance(values, values) for size, values in by_size.items()}
            for digit, by_size in per_class.items()
        }
        for role, per_class in targets.items()
    }
    class_rows: list[dict[str, Any]] = []
    output_metrics: dict[str, dict[str, list[float]]] = {arm: {key: [] for key in ("camera_integrated_intensity", "generator_input_power", "raw_clip_fraction", "e_d_correct")} for arm in ("A5400", "B5400")}
    class_e_d: dict[str, dict[int, list[int]]] = {arm: {digit: [] for digit in range(CLASS_COUNT)} for arm in ("A5400", "B5400")}
    for arm in ("A5400", "B5400"):
        for seed in SEEDS:
            raw, view, input_z = _load_seed_readout(target, arm, seed)
            expected_z = inputs["prior"]["bank"]["joint"][SEEDS.index(seed)].reshape(300, PORTS)
            _require(torch.equal(input_z, expected_z), f"{arm} seed {seed} did not consume the saved joint OLS bank")
            for chunk_index in range(6):
                payload = torch.load(target / "readouts" / arm / f"seed_{seed}" / f"chunk_{chunk_index:04d}.pt", map_location="cpu", weights_only=True)
                for key in output_metrics[arm]:
                    if key == "e_d_correct":
                        values = [float(row["e_d_correct"]) for row in payload["rows"]]
                    else:
                        values = payload[key].reshape(-1).to(torch.float64).tolist()
                    output_metrics[arm][key].extend(values)
                for row in payload["rows"]:
                    class_e_d[arm][int(row["label"])].append(int(row["e_d_correct"]))
            for digit in range(CLASS_COUNT):
                positions = slice(digit * DRAW_COUNT, (digit + 1) * DRAW_COUNT)
                raw_class = raw[positions]
                for role in ("fit", "confirmation"):
                    for size in (28, 200):
                        x = _shape_group(raw_class, size)
                        y = targets[role][digit][size]
                        cross = _mean_distance(x, y)
                        within_x = _mean_distance(x, x)
                        within_y = target_within[role][digit][size]
                        stat = 2.0 * cross - within_x - within_y
                        _require(math.isfinite(stat) and stat >= -1.0e-10, "shape-distribution distance is invalid")
                        stat = max(0.0, stat)
                        reconstructed = 2.0 * cross - within_x - within_y
                        _require(abs(reconstructed - stat) <= 1.0e-10, "shape-distance decomposition differs")
                        class_rows.append({
                            "arm": arm, "seed": seed, "label": digit, "target_role": role, "shape_size": size,
                            "output_n": DRAW_COUNT, "target_n": int(y.shape[0]),
                            "energy_distance": stat, "cross_distance": cross,
                            "within_output_distance": within_x, "within_target_distance": within_y,
                            "decomposition_residual": stat - (2.0 * cross - within_x - within_y),
                        })
            print(f"computed morphology summaries {arm} seed={seed}", flush=True)

    aggregate: dict[str, Any] = {}
    seed_effects: dict[str, Any] = {}
    class_effects: dict[str, Any] = {}
    for role in ("fit", "confirmation"):
        aggregate[role] = {}
        seed_effects[role] = {}
        class_effects[role] = {}
        for size in (28, 200):
            aggregate[role][str(size)] = {}
            seed_effects[role][str(size)] = {}
            class_effects[role][str(size)] = {}
            arm_means = {}
            for arm in ("A5400", "B5400"):
                rows = [row for row in class_rows if row["arm"] == arm and row["target_role"] == role and row["shape_size"] == size]
                arm_means[arm] = {field: float(sum(row[field] for row in rows) / len(rows)) for field in ("energy_distance", "cross_distance", "within_output_distance", "within_target_distance")}
                aggregate[role][str(size)][arm] = arm_means[arm]
            aggregate[role][str(size)]["B_minus_A"] = {field: arm_means["B5400"][field] - arm_means["A5400"][field] for field in arm_means["A5400"]}
            for seed in SEEDS:
                for arm in ("A5400", "B5400"):
                    rows = [row for row in class_rows if row["arm"] == arm and row["seed"] == seed and row["target_role"] == role and row["shape_size"] == size]
                    seed_effects[role][str(size)].setdefault(str(seed), {})[arm] = float(sum(row["energy_distance"] for row in rows) / CLASS_COUNT)
                seed_effects[role][str(size)][str(seed)]["B_minus_A"] = seed_effects[role][str(size)][str(seed)]["B5400"] - seed_effects[role][str(size)][str(seed)]["A5400"]
            for digit in range(CLASS_COUNT):
                class_effects[role][str(size)][str(digit)] = {}
                for arm in ("A5400", "B5400"):
                    rows = [row for row in class_rows if row["arm"] == arm and row["target_role"] == role and row["shape_size"] == size and row["label"] == digit]
                    class_effects[role][str(size)][str(digit)][arm] = float(sum(row["energy_distance"] for row in rows) / len(rows))
                class_effects[role][str(size)][str(digit)]["B_minus_A"] = class_effects[role][str(size)][str(digit)]["B5400"] - class_effects[role][str(size)][str(digit)]["A5400"]

    e_d_summary = {}
    operational = {}
    for arm in ("A5400", "B5400"):
        total_correct = sum(output_metrics[arm]["e_d_correct"])
        e_d_summary[arm] = {
            "correct": int(total_correct), "rows": FORMAL_ROWS_PER_ARM, "accuracy": total_correct / FORMAL_ROWS_PER_ARM,
            "by_class": {str(digit): {"correct": sum(class_e_d[arm][digit]), "rows": 150, "accuracy": sum(class_e_d[arm][digit]) / 150} for digit in range(CLASS_COUNT)},
        }
        operational[arm] = {}
        for key in ("camera_integrated_intensity", "generator_input_power", "raw_clip_fraction"):
            values = output_metrics[arm][key]
            operational[arm][key] = {"mean": sum(values) / len(values), "minimum": min(values), "maximum": max(values)}
    auxiliary = {
        "shape_statistics": aggregate,
        "per_seed_shape_distance": seed_effects,
        "per_class_shape_distance": class_effects,
        "E_D_auxiliary": e_d_summary,
        "operational_auxiliary": operational,
        "outputs_have_no_single_image_target_pair": True,
        "target_roles_are_canonical_train": True,
    }
    return class_rows, auxiliary
def _load_historical_joint(prior_dir: Path) -> dict[str, torch.Tensor]:
    seed = SEEDS[0]
    chunks = []
    base = prior_dir / "readouts" / "joint" / f"seed_{seed}"
    for index in range(6):
        path = base / f"chunk_{index:04d}.pt"
        _require(path.is_file(), f"saved A4700 joint-prior chunk is missing: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _require(payload.get("task_id") == prior_task.TASK_ID and payload.get("arm") == "joint" and payload.get("seed") == seed, "historical A4700 readout identity differs")
        chunks.append(payload)
    raw = torch.cat([chunk["raw_cmos"] for chunk in chunks], dim=0)
    view = torch.cat([chunk["evaluator_view"] for chunk in chunks], dim=0)
    e_d = torch.cat([chunk["e_d_logits"] for chunk in chunks], dim=0)
    _require(tuple(raw.shape) == (300, 1, 200, 200) and tuple(view.shape) == (300, 1, 28, 28), "historical A4700 readout geometry differs")
    positions = torch.tensor([digit * DRAW_COUNT for digit in range(CLASS_COUNT)], dtype=torch.long)
    return {"raw_cmos": raw.index_select(0, positions), "evaluator_view": view.index_select(0, positions), "e_d_logits": e_d.index_select(0, positions), "positions": positions}


def _write_plan(target: Path) -> None:
    path = target / "experiment_plan.md"
    if path.exists():
        _require(path.read_text(encoding="utf-8") == _plan_text(), "existing plan differs from this attempt")
    else:
        path.write_text(_plan_text(), encoding="utf-8", newline="\n")


def _resource_observation(target: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized formal frozen readout")
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    estimated_bytes = 2 * FORMAL_ROWS_PER_ARM * (200 * 200 * 4 + 28 * 28 * 4 + 10 * 4 + 100 * 8 + 5 * 8)
    disk_root = target
    while not disk_root.exists():
        disk_root = disk_root.parent
    free_disk = shutil.disk_usage(disk_root).free
    _require(free_disk > estimated_bytes, "available disk cannot hold the planned 3000-row paired readout")
    return {
        "executed": True, "passed": True, "cuda_device": str(device),
        "cuda_total_bytes": int(total_bytes), "cuda_free_bytes": int(free_bytes),
        "disk_path": disk_root.as_posix(), "disk_free_bytes": int(free_disk),
        "estimated_readout_bytes": int(estimated_bytes),
        "memory_and_elapsed_time_are_observations_only": True,
    }


def _identity(inputs: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="fixed-joint-prior-A5400-vs-B5400-single-reviewer-readability",
        semantic_authority_id=generation.TASK_ID,
        data_role="canonical TRAIN; 200 fit rows construct the fixed prior and 300 confirmation targets are measurement-only",
        config_identity={"seeds": list(SEEDS), "paired_rows_per_arm": FORMAL_ROWS_PER_ARM, "blind_items": ITEM_COUNT, "strict_fp32_optics": True, "microbatch_rows": 1},
        model_identity={"A5400": inputs["checkpoint_a_path"].as_posix(), "B5400": inputs["checkpoint_b_path"].as_posix(), "source_step": 4700, "endpoint_step": 5400},
        optimizer_identity={"updates": 0, "optimizer": "not instantiated for frozen measurement"},
        sampler_identity={"source": inputs["prior_dir"].as_posix(), "same_complex64_joint_bank_for_both_arms": True},
        execution_source_semantics_id=TASK_ID, scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _make_ready_receipt(inputs: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    checks = preflight.get("checks", {})
    _require(checks and all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY requires every preflight check to have executed and passed")
    control = ResearchRunControl(_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    receipt = control.make_ready_receipt(
        correctness={
            "passed": True, "status": "PASS",
            "source_roles_targets_OLS_and_coordinates": True,
            "A5400_B5400_saved_replay": all(checks[f"{arm}_saved_replay"]["passed"] for arm in ("A5400", "B5400")),
            "A4700_joint_replay": checks["A4700_historical_joint_replay"]["passed"],
            "joint_bank_reconstruction": checks["same_joint_samples_for_A_and_B"]["passed"],
        },
        resource={"passed": preflight["resources_observed"]["passed"], "status": "PASS", "observation": preflight["resources_observed"]},
        implementation={
            "passed": checks["global_phase_invariance_both_endpoints"]["passed"] and checks["half_amplitude_shape_and_intensity_both_endpoints"]["passed"],
            "status": "PASS", "selected_engine": "cuda_2048_single_row_fp32_optics",
            "frozen_runtime_evaluator_and_gradients": checks["frozen_states_evaluator_and_empty_gradients"]["passed"],
            "save_reload": checks["preflight_save_reload"]["passed"],
        },
        selected_engine="cuda_2048_single_row_fp32_optics",
        resource_envelope={"cuda_and_disk": preflight["resources_observed"]},
    )
    return control, receipt


def _stage_ready(target: Path, inputs: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    payload = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}
    if resume:
        _require(_json_read(target / "input_population.json") == inputs["population"], "saved preflight population differs on resume")
        _require(_json_read(target / "preflight.json") == payload, "saved READY evidence differs on resume")
        return
    _json_write_new(target / "input_population.json", inputs["population"])
    _json_write_new(target / "preflight.json", payload)


def _validate_ready(target: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    saved = _json_read(target / "preflight.json")
    _require(saved.get("task_id") == TASK_ID and saved.get("status") == "READY", "saved preflight is not READY")
    _require(_json_read(target / "input_population.json") == inputs["population"], "saved READY population differs from current source")
    preflight = saved["preflight"]
    _require(preflight.get("status") == "READY", "preflight did not complete successfully")
    checks = preflight.get("checks", {})
    _require(checks and all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY consumer found a missing or failed executed check")
    receipt = PreflightReceipt.from_mapping(saved["receipt"])
    _require(receipt.source_semantics_id == TASK_ID and receipt.authorization_id == f"{TASK_ID}:authorization:01" and receipt.status == "READY", "preflight receipt identity differs")
    _require(all(value.get("passed") is True for value in receipt.checks.values()), "preflight receipt contains a missing or failed check group")
    control = ResearchRunControl(_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    control.mark_preflight_ready(receipt)
    return preflight


def _measurement_identity_record(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "source_attempt": inputs["artifact_dir"].as_posix(),
        "training_attempt": inputs["training_dir"].as_posix(),
        "prior_attempt": inputs["prior_dir"].as_posix(),
        "checkpoint_A": inputs["checkpoint_a_path"].as_posix(),
        "checkpoint_B": inputs["checkpoint_b_path"].as_posix(),
        "population_source_rows": inputs["population"]["source_row_indices"],
        "roles": inputs["roles"], "seeds": list(SEEDS),
        "joint_bank_shape": [5, 10, 30, PORTS], "joint_bank_dtype": "complex64",
        "checkpoint_steps": {"A": 5400, "B": 5400, "source": 4700},
    }


def _persist_identity(target: Path, inputs: Mapping[str, Any], *, resume: bool) -> None:
    record = _measurement_identity_record(inputs)
    path = target / "input_record.json"
    if resume:
        _require(_json_read(path) == record, "explicit resume inputs differ from the original attempt")
    else:
        _json_write_new(path, record)
        _json_write_new(target / "fixed_parameters.json", {
            "task_id": TASK_ID, "seeds": list(SEEDS),
            "sampling_bank": "persisted complex64 joint bank from named prior attempt, no refit or resampling",
            "formal_rows_per_arm": FORMAL_ROWS_PER_ARM, "chunk_rows": CHUNK_ROWS,
            "human_review": {
                "items": ITEM_COUNT, "page_size": PAGE_SIZE,
                "selection_seed": SELECTION_SEED, "repeat_seed": REPEAT_SEED,
                "order_seed": ORDER_SEED, "minimum_spacing": MIN_SPACING,
                "quality_checks": {"real_correct_minimum": 18, "black_unclear_minimum": 9, "A_repeat_agree_minimum": 18, "B_repeat_agree_minimum": 18},
                "primary": "human request-class recognition difference and two-sided exact McNemar, alpha 0.05",
            },
            "decision_authority": "human paired request-class recognition only; numerical shape statistics are auxiliary",
        })


def _preflight_sequential(inputs: Mapping[str, Any], target: Path, device: torch.device) -> dict[str, Any]:
    bank = inputs["prior"]["bank"]["joint"]
    checks: dict[str, dict[str, Any]] = {}
    payload: dict[str, Any] = {"task_id": TASK_ID}
    evaluator = generation.diagnostic.load_evaluators(inputs["generation_paths"], device)["E_D"]
    evaluator_before = generation.capacity._state_snapshot(evaluator)
    replay_positions = torch.stack([(inputs["labels"][200:] == digit).nonzero().flatten()[0] + 200 for digit in range(CLASS_COUNT)])
    image_replay = inputs["generation_inputs"]["images"].index_select(0, replay_positions)
    sample_z = bank[0, :, 0, :].contiguous()
    _require(torch.equal(sample_z, bank[0, :, 0, :].clone()), "fixed joint sample bank is not stable")
    phase = torch.linspace(0.21, 2.81, CLASS_COUNT, dtype=torch.float64)
    phase_factor = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
    for arm, checkpoint, saved in (
        ("A5400", inputs["checkpoint_a"], inputs["endpoint_a"]),
        ("B5400", inputs["checkpoint_b"], inputs["endpoint_b"]),
    ):
        runtime = generation._new_runtime(checkpoint, device)
        before = generation._snapshot_state(runtime)
        replay = _infer_endpoint_images(runtime, evaluator, image_replay, device)
        expected_raw = saved["raw_cmos"].index_select(0, replay_positions)
        expected_view = saved["evaluator_view"].index_select(0, replay_positions)
        replay_pass = torch.allclose(replay["raw_cmos"], expected_raw, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        replay_pass = replay_pass and torch.allclose(replay["evaluator_view"], expected_view, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        replay_pass = replay_pass and torch.allclose(replay["target_slm_intensity"], saved["target_slm_intensity"].index_select(0, replay_positions), rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        replay_pass = replay_pass and torch.allclose(replay["e_d_logits"], saved["e_d_logits"].index_select(0, replay_positions), rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        _require(replay_pass, f"{arm} saved 10-row replay differs")
        sample = frozen._infer_rows(runtime, evaluator, sample_z, device, inputs["pixel_area_m2"])
        rotated = frozen._infer_rows(runtime, evaluator, sample_z * phase_factor[:, None], device, inputs["pixel_area_m2"])
        half = frozen._infer_rows(runtime, evaluator, sample_z * 0.5, device, inputs["pixel_area_m2"])
        shape = frozen._shape_vectors(sample["raw_cmos"], 28)
        phase_shape = frozen._shape_vectors(rotated["raw_cmos"], 28)
        half_shape = frozen._shape_vectors(half["raw_cmos"], 28)
        phase_pass = torch.allclose(shape, phase_shape, rtol=0.0, atol=SHAPE_ATOL)
        scale_pass = torch.allclose(shape, half_shape, rtol=0.0, atol=SCALE_ATOL)
        intensity_pass = torch.allclose(half["raw_cmos"], sample["raw_cmos"] * 0.25, rtol=2.0e-5, atol=REPLAY_ATOL)
        _require(phase_pass and scale_pass and intensity_pass, f"{arm} phase/amplitude invariance check failed")
        _require(_tree_equal(before, generation._snapshot_state(runtime)), f"{arm} runtime changed during preflight")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters()), f"{arm} runtime is not frozen")
        checks[f"{arm}_saved_replay"] = {"executed": True, "passed": True, "rows": 10, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL}
        checks[f"{arm}_phase_invariance"] = {"executed": True, "passed": bool(phase_pass), "rows": 10, "max_abs_difference": float((shape - phase_shape).abs().max())}
        checks[f"{arm}_half_amplitude_invariance"] = {"executed": True, "passed": bool(scale_pass and intensity_pass), "rows": 10, "max_shape_difference": float((shape - half_shape).abs().max()), "intensity_factor": 0.25}
        payload[f"{arm}_replay"] = replay
        payload[f"{arm}_sample"] = sample
        payload[f"{arm}_phase"] = rotated
        payload[f"{arm}_half"] = half
        del runtime
        torch.cuda.empty_cache()

    source_runtime = generation._new_runtime(inputs["checkpoint"], device)
    source_before = generation._snapshot_state(source_runtime)
    historical = inputs["historical_joint"]
    old_replay = frozen._infer_rows(source_runtime, evaluator, sample_z, device, inputs["pixel_area_m2"])
    old_ok = torch.allclose(old_replay["raw_cmos"], historical["raw_cmos"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    old_view_ok = torch.allclose(old_replay["evaluator_view"], historical["evaluator_view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    _require(old_ok and old_view_ok, "saved A4700 joint-prior readout replay differs")
    checks["A4700_historical_joint_replay"] = {"executed": True, "passed": True, "rows": 10, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL}
    payload["A4700_joint_replay"] = old_replay
    _require(_tree_equal(source_before, generation._snapshot_state(source_runtime)), "A4700 source runtime changed during replay")
    _require(_tree_equal(evaluator_before, generation.capacity._state_snapshot(evaluator)), "E_D evaluator changed during preflight")
    gradients_empty = all(parameter.grad is None and not parameter.requires_grad for module in (source_runtime, evaluator) for parameter in module.parameters())
    _require(gradients_empty, "frozen runtime or evaluator has gradients enabled")
    _torch_write_new(target / "preflight_readouts.pt", payload)
    saved = torch.load(target / "preflight_readouts.pt", map_location="cpu", weights_only=True)
    _require(_tree_equal(saved, payload), "preflight save/reload differs")
    checks["same_joint_samples_for_A_and_B"] = {"executed": True, "passed": True, "rows_per_arm": FORMAL_ROWS_PER_ARM, "input_z_reused": True}
    checks["global_phase_invariance_both_endpoints"] = {"executed": True, "passed": all(checks[f"{arm}_phase_invariance"]["passed"] for arm in ("A5400", "B5400")), "tolerance": SHAPE_ATOL}
    checks["half_amplitude_shape_and_intensity_both_endpoints"] = {"executed": True, "passed": all(checks[f"{arm}_half_amplitude_invariance"]["passed"] for arm in ("A5400", "B5400")), "intensity_factor": 0.25}
    checks["frozen_states_evaluator_and_empty_gradients"] = {"executed": True, "passed": True}
    checks["preflight_save_reload"] = {"executed": True, "passed": True}
    return {
        "task_id": TASK_ID, "status": "READY", "forward_rows": 90, "checks": checks,
        "model_updates": 0, "optimizer_updates": 0, "gradient_calls": 0,
        "resources_observed": _resource_observation(target),
    }


def _infer_endpoint_images(
    runtime: torch.nn.Module,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    _require(images.ndim == 4 and images.shape[1:] == (1, 28, 28) and images.shape[0] == CLASS_COUNT, "saved endpoint replay requires one image per class")
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")}
    for index in range(images.shape[0]):
        image = images[index : index + 1].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            result = runtime.forward_from_image(image)
            view = generation.capacity.canonical.cmos_view(result.raw_intensity)
            logits = generation._evaluate(evaluator, view, "E_D")
            row = {
                "raw_cmos": result.raw_intensity,
                "evaluator_view": view,
                "target_slm_intensity": result.image_input.slm_intensity,
                "e_d_logits": logits,
            }
        for key, value in row.items():
            pieces[key].append(value.detach().to(device="cpu", dtype=torch.float32).contiguous())
    result = {key: torch.cat(values, dim=0) for key, values in pieces.items()}
    _require(all(bool(torch.isfinite(value).all()) for value in result.values()), "saved endpoint replay contains nonfinite values")
    _require(bool((result["raw_cmos"] >= 0.0).all() and (result["raw_cmos"].flatten(1).sum(1) > 0.0).all()), "saved endpoint replay has negative or zero-mass raw CMOS")
    return result
def _sample_row_metadata(seed: int, digit: int, draw: int, arm: str) -> dict[str, Any]:
    return {"task_id": TASK_ID, "arm": arm, "seed": int(seed), "label": int(digit), "draw_index": int(draw), "pair_id": f"{seed}:{digit}:{draw}"}

def _run_arm_fixed(target: Path, inputs: Mapping[str, Any], arm: str, checkpoint: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    bank = inputs["prior"]["bank"]["joint"]
    runtime = generation._new_runtime(checkpoint, device)
    evaluator = generation.diagnostic.load_evaluators(inputs["generation_paths"], device)["E_D"]
    runtime_before = generation._snapshot_state(runtime)
    evaluator_before = generation.capacity._state_snapshot(evaluator)
    summary_rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        seed_bank = bank[seed_index].reshape(300, PORTS)
        seed_dir = target / "readouts" / arm / f"seed_{seed}"
        for chunk_index, start in enumerate(range(0, 300, CHUNK_ROWS)):
            stop = start + CHUNK_ROWS
            path = seed_dir / f"chunk_{chunk_index:04d}.pt"
            labels = torch.arange(CLASS_COUNT).repeat_interleave(DRAW_COUNT)[start:stop]
            draws = torch.arange(DRAW_COUNT).repeat(CLASS_COUNT)[start:stop]
            base_rows = [_sample_row_metadata(seed, int(label), int(draw), arm) for label, draw in zip(labels, draws, strict=True)]
            input_z = seed_bank[start:stop].contiguous()
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                _require(payload.get("task_id") == TASK_ID and payload.get("arm") == arm and payload.get("seed") == seed, "saved readout chunk identity differs on resume")
                _require(payload.get("flat_start") == start, "saved readout chunk start differs on resume")
                _require(all(all(payload["rows"][i].get(key) == base_rows[i][key] for key in ("task_id", "arm", "seed", "label", "draw_index", "pair_id")) for i in range(CHUNK_ROWS)), "saved readout chunk row identity differs on resume")
                _require(torch.equal(payload.get("input_z"), input_z), "saved readout chunk no longer uses the fixed joint input bank")
                _require(tuple(payload["raw_cmos"].shape) == (CHUNK_ROWS, 1, 200, 200), "saved readout chunk raw shape differs")
                _require(bool(torch.isfinite(payload["raw_cmos"]).all() and (payload["raw_cmos"] >= 0.0).all() and (payload["raw_cmos"].flatten(1).sum(1) > 0.0).all()), "saved readout chunk raw values are invalid")
            else:
                output = frozen._infer_rows(runtime, evaluator, input_z, device, inputs["pixel_area_m2"])
                rows = []
                for offset, meta in enumerate(base_rows):
                    logits = output["e_d_logits"][offset]
                    prediction = int(logits.argmax())
                    rows.append({
                        **meta,
                        "position": seed_index * 300 + start + offset,
                        "e_d_prediction": prediction,
                        "e_d_correct": int(prediction == int(meta["label"])),
                    })
                payload = {
                    "task_id": TASK_ID, "arm": arm, "seed": seed,
                    "flat_start": start, "rows": rows, "input_z": input_z, **output,
                }
                _torch_write_new(path, payload)
            for offset, row in enumerate(payload["rows"]):
                logits = payload["e_d_logits"][offset]
                summary_rows.append({
                    "arm": arm, "seed": int(seed), "label": int(row["label"]),
                    "draw_index": int(row["draw_index"]), "pair_id": row["pair_id"],
                    "position": int(row["position"]), "e_d_prediction": int(row["e_d_prediction"]),
                    "e_d_correct": int(row["e_d_correct"]),
                    "camera_integrated_intensity": float(payload["camera_integrated_intensity"][offset]),
                    "generator_input_power": float(payload["generator_input_power"][offset]),
                    "raw_clip_fraction": float(payload["raw_clip_fraction"][offset]),
                    "e_d_target_logit": float(logits[int(row["label"])]),
                })
        print(f"saved fixed-prior readout {arm} seed={seed}", flush=True)
    frozen_ok = _tree_equal(runtime_before, generation._snapshot_state(runtime))
    frozen_ok = frozen_ok and _tree_equal(evaluator_before, generation.capacity._state_snapshot(evaluator))
    gradients_empty = all(parameter.grad is None and not parameter.requires_grad for module in (runtime, evaluator) for parameter in module.parameters())
    _require(frozen_ok and gradients_empty, f"{arm} runtime/evaluator changed or gained gradients during readout")
    return summary_rows


def _png_record(view: torch.Tensor) -> tuple[str, str]:
    tensor = view.detach().to(device="cpu", dtype=torch.float32).contiguous()
    _require(tuple(tensor.shape) == (1, 28, 28), "blind review image is not one 28x28 view")
    _require(bool(torch.isfinite(tensor).all() and (tensor >= 0.0).all() and (tensor <= 1.0).all()), "blind review image is outside fixed [0,1]")
    uri = review._image_data_uri(tensor[0])
    _require(uri.startswith("data:image/png;base64,"), "saved review image is not PNG")
    content = base64.b64decode(uri.partition(",")[2], validate=True)
    return uri, hashlib.sha256(content).hexdigest()


def _load_review_views(target: Path, selected: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int, int, str], torch.Tensor]:
    wanted = {(int(row["seed"]), int(row["label"]), int(row["draw_index"])) for row in selected}
    result: dict[tuple[int, int, int, str], torch.Tensor] = {}
    for seed in SEEDS:
        if not any(key[0] == seed for key in wanted):
            continue
        for arm in ("A5400", "B5400"):
            _, views, _ = _load_seed_readout(target, arm, seed)
            for seed_value, digit, draw in wanted:
                if seed_value == seed:
                    result[(seed, digit, draw, arm)] = views[digit * DRAW_COUNT + draw].contiguous()
    return result


def _render_review_html(public_items: Sequence[Mapping[str, str]], review_token: str) -> str:
    payload = json.dumps(list(public_items), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    token = json.dumps(review_token)
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数字图像盲判</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:18px;background:#f4f4f4;color:#222}
header{position:sticky;top:0;background:white;padding:12px;border:1px solid #ccc;z-index:2}
.grid{display:grid;grid-template-columns:repeat(5,minmax(160px,1fr));gap:10px;margin-top:14px}
.card{background:white;border:1px solid #ddd;padding:8px;text-align:center}
.card img{width:224px;height:224px;image-rendering:pixelated;object-fit:contain;background:white}
.id{font-family:monospace;color:#555;margin:4px}select,button{font-size:16px;padding:6px;margin:3px}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(150px,1fr))}}
@media(max-width:560px){.grid{grid-template-columns:1fr}}
</style></head><body><header><strong>单人数字图像盲判</strong>
<p>只根据图像选择 0–9。只有在能够辨认时才选数字；无法辨认或不能稳定辨认时请选择“无法辨认”。图像使用统一灰度范围 [0,1] 和 8 倍整数放大，没有逐图增强或修饰。</p>
<p>共六组、每组45题。请完成全部270题后导出完整评分。评分期间不提供正确答案或中间统计。</p>
<label>载入此前导出的进度：<input id="import" type="file" accept=".csv,text/csv"></label>
<div><button id="prev">上一组</button><span id="page"></span><button id="next">下一组</button>
<button id="exportProgress">导出进度</button><button id="exportFinal">导出完整评分</button></div>
<span id="progress"></span><p id="message">进度保存在当前浏览器；可以导出并之后载入。</p></header>
<main id="grid" class="grid"></main><script>
var items=__ITEMS__,reviewId=__TOKEN__,pageSize=45,totalPages=6,answers={};
var allowed=new Set([...Array(10).keys()].map(String).concat(["unclear"]));
var storageKey="lightgen-fixed-prior-readability-"+reviewId;
try{var old=localStorage.getItem(storageKey);if(old)Object.assign(answers,JSON.parse(old));}catch(e){}
var grid=document.getElementById("grid"),page=0;
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(answers));}catch(e){}updateProgress();}
function render(){
 grid.innerHTML="";
 var start=page*pageSize,visible=items.slice(start,start+pageSize);
 visible.forEach(function(item){
  var card=document.createElement("section");card.className="card";
  var image=document.createElement("img");image.src=item.image;image.alt="";image.draggable=false;
  var id=document.createElement("div");id.className="id";id.textContent="编号 "+item.item_id;
  var select=document.createElement("select");select.setAttribute("aria-label","编号 "+item.item_id+" 的读数");
  var empty=document.createElement("option");empty.value="";empty.textContent="请选择";select.appendChild(empty);
  for(var d=0;d<10;d++){var option=document.createElement("option");option.value=String(d);option.textContent=String(d);select.appendChild(option);}
  var unclear=document.createElement("option");unclear.value="unclear";unclear.textContent="无法辨认";select.appendChild(unclear);
  select.value=answers[item.item_id]||"";
  select.onchange=function(){if(select.value)answers[item.item_id]=select.value;else delete answers[item.item_id];persist();};
  card.append(image,id,select);grid.appendChild(card);
 });
 var first=start+1,last=Math.min(start+pageSize,items.length);
 document.getElementById("page").textContent="第 "+(page+1)+" / "+totalPages+" 组（"+first+"–"+last+"）";
 document.getElementById("prev").disabled=page===0;document.getElementById("next").disabled=page===totalPages-1;updateProgress();
}
function updateProgress(){
 var count=Object.keys(answers).filter(function(id){return items.some(function(x){return x.item_id===id;})&&allowed.has(answers[id]);}).length;
 document.getElementById("progress").textContent="已完成 "+count+" / "+items.length+" 项";
 document.getElementById("exportFinal").disabled=count!==items.length;
}
function download(requireComplete){
 var valid=items.filter(function(x){return answers[x.item_id]&&allowed.has(answers[x.item_id]);}).length;
 if(requireComplete&&valid!==items.length)return;
 var lines=["item_id,answer"];
 items.forEach(function(x){lines.push(x.item_id+","+(answers[x.item_id]||""));});
  var blob=new Blob(["\\ufeff"+lines.join("\\r\\n")],{type:"text/csv;charset=utf-8"});
 var a=document.createElement("a");a.href=URL.createObjectURL(blob);
 a.download="lightgen_fixed_prior_readability_"+(requireComplete?"ratings":"progress")+"_"+reviewId+".csv";a.click();
 setTimeout(function(){URL.revokeObjectURL(a.href);},1000);
}
document.getElementById("prev").onclick=function(){page--;render();};
document.getElementById("next").onclick=function(){page++;render();};
document.getElementById("exportProgress").onclick=function(){download(false);};
document.getElementById("exportFinal").onclick=function(){download(true);};
document.getElementById("import").onchange=function(event){
 var file=event.target.files[0];if(!file)return;
 var reader=new FileReader();reader.onload=function(){
  try{
   var lines=String(reader.result).replace(/^\\uFEFF/,"").trim().split(/\\r?\\n/);
   if(lines.shift()!=="item_id,answer")throw Error("CSV 表头错误");
   var known=new Set(items.map(function(x){return x.item_id;})),incoming={};
   lines.forEach(function(line){
    var fields=line.split(",");if(fields.length!==2||!known.has(fields[0])||Object.prototype.hasOwnProperty.call(incoming,fields[0]))throw Error("评分行无效");
    if(fields[1]!==""&&!allowed.has(fields[1]))throw Error("评分值无效");incoming[fields[0]]=fields[1];
   });
   Object.keys(incoming).forEach(function(id){if(incoming[id])answers[id]=incoming[id];else delete answers[id];});
   persist();render();document.getElementById("message").textContent="进度已载入。";
  }catch(err){document.getElementById("message").textContent="载入失败："+err.message;}
 };reader.readAsText(file);
};
render();
</script></body></html>"""
    _require(all(set(item.keys()) == {"item_id", "image"} for item in public_items), "public review payload contains hidden fields")
    return html.replace("__ITEMS__", payload).replace("__TOKEN__", token)


def _prepare_review(target: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    selection_rng = random.Random(SELECTION_SEED)
    repeat_rng = random.Random(REPEAT_SEED)
    order_rng = random.Random(ORDER_SEED)
    selected: list[dict[str, Any]] = []
    for seed in SEEDS:
        for digit in range(CLASS_COUNT):
            for draw in selection_rng.sample(sorted(range(DRAW_COUNT)), 2):
                selected.append({"seed": seed, "label": digit, "draw_index": draw, "pair_id": f"{seed}:{digit}:{draw}"})
    _require(len(selected) == PRIMARY_PAIR_COUNT and len({row["pair_id"] for row in selected}) == PRIMARY_PAIR_COUNT, "100 matched sample pairs are not unique")
    repeated_pairs = []
    for digit in range(CLASS_COUNT):
        candidates = [row for row in selected if int(row["label"]) == digit]
        repeated_pairs.extend(repeat_rng.sample(candidates, 2))
    repeated_ids = {row["pair_id"] for row in repeated_pairs}
    ordered_pairs = [row for row in selected if row["pair_id"] in repeated_ids] + [row for row in selected if row["pair_id"] not in repeated_ids]
    _require(len(ordered_pairs) == PRIMARY_PAIR_COUNT and len(repeated_pairs) == 20, "repeat-pair selection differs")
    early_a_ids = {row["pair_id"] for row in order_rng.sample(selected, PRIMARY_PAIR_COUNT // 2)}
    views = _load_review_views(target, selected)
    slots: list[dict[str, Any] | None] = [None] * ITEM_COUNT
    original_ids: dict[tuple[str, str], str] = {}
    pair_positions: dict[str, dict[str, int]] = {}
    for i, pair in enumerate(ordered_pairs):
        pair_id = str(pair["pair_id"])
        early_arm = "A5400" if pair_id in early_a_ids else "B5400"
        late_arm = "B5400" if early_arm == "A5400" else "A5400"
        pair_positions[pair_id] = {}
        for pos, arm in ((i, early_arm), (150 + i, late_arm)):
            view = views[(int(pair["seed"]), int(pair["label"]), int(pair["draw_index"]), arm)]
            uri, digest = _png_record(view)
            item_id = f"FJ{pos + 1:03d}"
            slots[pos] = {
                "item_id": item_id, "question_number": pos + 1,
                "source_type": f"{arm}_primary", "arm": arm, "pair_id": pair_id,
                "requested_class": int(pair["label"]), "seed": int(pair["seed"]),
                "draw_index": int(pair["draw_index"]), "image_sha256": digest, "image": uri,
            }
            original_ids[(pair_id, arm)] = item_id
            pair_positions[pair_id][arm] = pos

    for i, pair in enumerate(ordered_pairs[:20]):
        pair_id = str(pair["pair_id"])
        for pos, arm in ((100 + i, "B5400"), (250 + i, "A5400")):
            original_id = original_ids[(pair_id, arm)]
            original = slots[int(original_id[2:]) - 1]
            _require(original is not None, "hidden repeat parent is missing")
            view = views[(int(pair["seed"]), int(pair["label"]), int(pair["draw_index"]), arm)]
            uri, digest = _png_record(view)
            _require(digest == original["image_sha256"], "hidden repeat PNG differs from its source image")
            item_id = f"FJ{pos + 1:03d}"
            slots[pos] = {
                "item_id": item_id, "question_number": pos + 1,
                "source_type": f"{arm}_repeat", "arm": arm, "pair_id": pair_id,
                "requested_class": int(pair["label"]), "seed": int(pair["seed"]),
                "draw_index": int(pair["draw_index"]), "repeat_of": original_id,
                "image_sha256": digest, "image": uri,
            }

    controls: list[dict[str, Any]] = []
    labels = inputs["labels"]
    target_raw = inputs["readout"]["target_slm_intensity"]
    confirmation_positions = {
        digit: [i for i in range(FIT_ROWS, TOTAL_ROWS) if int(labels[i]) == digit]
        for digit in range(CLASS_COUNT)
    }
    for digit in range(CLASS_COUNT):
        _require(len(confirmation_positions[digit]) == 30, f"confirmation real-target pool for class {digit} differs")
        for position in selection_rng.sample(confirmation_positions[digit], 2):
            target_view = generation.capacity.canonical.cmos_view(target_raw[position : position + 1].to(torch.float32))
            uri, digest = _png_record(target_view[0])
            controls.append({
                "source_type": "real_control", "true_class": digit,
                "source_row_index": int(inputs["population"]["source_row_indices"][position]),
                "source_position": int(position), "image_sha256": digest, "image": uri,
            })
    _require(len(controls) == REAL_CONTROL_COUNT, "real-target controls do not total 20")
    black_uri, black_digest = _png_record(torch.zeros((1, 28, 28), dtype=torch.float32))
    controls.extend({"source_type": "black_control", "image_sha256": black_digest, "image": black_uri} for _ in range(BLACK_CONTROL_COUNT))
    for pos, control in zip(range(120, 150), controls, strict=True):
        slots[pos] = {"item_id": f"FJ{pos + 1:03d}", "question_number": pos + 1, **control}

    _require(all(item is not None for item in slots), "blind review schedule contains empty question slots")
    mapping_full = [dict(item) for item in slots if item is not None]
    _require(len(mapping_full) == ITEM_COUNT and [row["question_number"] for row in mapping_full] == list(range(1, ITEM_COUNT + 1)), "blind review schedule length/order differs")
    expected_counts = {"A5400_primary": 100, "B5400_primary": 100, "A5400_repeat": 20, "B5400_repeat": 20, "real_control": 20, "black_control": 10}
    _require({key: sum(row["source_type"] == key for row in mapping_full) for key in expected_counts} == expected_counts, "blind review composition differs")
    first_arm_counts = {
        arm: sum(row["source_type"] == f"{arm}_primary" and row["question_number"] <= 100 for row in mapping_full)
        for arm in ("A5400", "B5400")
    }
    _require(first_arm_counts == {"A5400": 50, "B5400": 50}, "first-presented endpoint balance differs")
    public = [{"item_id": row["item_id"], "image": row["image"]} for row in mapping_full]
    by_id = {row["item_id"]: row for row in mapping_full}
    for pair_id, positions in pair_positions.items():
        _require(abs(positions["A5400"] - positions["B5400"]) >= MIN_SPACING, "matched A/B images are too close")
        for arm in ("A5400", "B5400"):
            original_id = original_ids[(pair_id, arm)]
            repeat = next((row for row in mapping_full if row.get("repeat_of") == original_id), None)
            if repeat is not None:
                original = by_id[original_id]
                _require(abs(int(repeat["question_number"]) - int(original["question_number"])) >= MIN_SPACING, "hidden repeat is too close to its original")
        repeats_for_pair = [row for row in mapping_full if row.get("pair_id") == pair_id and row["source_type"].endswith("_repeat")]
        if repeats_for_pair:
            repeat_positions = {row["arm"]: int(row["question_number"]) - 1 for row in repeats_for_pair}
            _require(abs(repeat_positions["A5400"] - repeat_positions["B5400"]) >= MIN_SPACING, "repeated A/B images are too close")

    mapping = [{key: value for key, value in row.items() if key != "image"} for row in mapping_full]
    validated_mapping = _validate_review_mapping(mapping)
    observed_repeat_spacing = min(
        abs(int(row["question_number"]) - int(validated_mapping["by_id"][str(row["repeat_of"])]["question_number"]))
        for row in mapping if row["source_type"].endswith("_repeat")
    )
    review_token = hashlib.sha256(TASK_ID.encode("utf-8")).hexdigest()[:16]
    (target / "blind_review.html").write_text(_render_review_html(public, review_token), encoding="utf-8", newline="\n")
    (target / "ratings_template.csv").write_text("\ufeffitem_id,answer\n" + "\n".join(f"{row['item_id']}," for row in mapping) + "\n", encoding="utf-8", newline="")
    _json_write_new(target / "private" / "review_mapping.json", {"task_id": TASK_ID, "items": mapping})
    eligibility = {
        "task_id": TASK_ID, "item_count": ITEM_COUNT, "page_size": PAGE_SIZE, "page_counts": {str(page + 1): PAGE_SIZE for page in range(6)},
        "selected_pair_count": len(selected), "repeated_pair_count": len(repeated_pairs),
        "composition": expected_counts, "first_presented_endpoint_counts": first_arm_counts,
        "minimum_repeat_spacing": observed_repeat_spacing, "minimum_spacing_required": MIN_SPACING,
        "repeat_png_bytes_identical": True, "public_fields": ["item_id", "image"],
        "display": {"view": "saved production 28x28", "range": [0, 1], "integer_scale": 8, "per_image_enhancement": False},
        "selection_seeds": {"selection": SELECTION_SEED, "repeat": REPEAT_SEED, "order": ORDER_SEED},
        "all_source_rows_canonical_train": True,
    }
    _json_write_new(target / "eligibility.json", eligibility)
    instructions = (
        "# 盲判说明\n\n"
        "请打开 blind_review.html，只根据图像选择 0–9 或“无法辨认”。只有能够辨认时才选择数字；不能稳定辨认时允许选择“无法辨认”。\n\n"
        "页面共六组、每组45题。请完成全部270题后导出完整评分。评分期间不要查看 private/review_mapping.json，也不要向评分入口提交部分进度文件。页面不显示请求类别、端点、分类器预测、来源或文件名。\n\n"
        "评分完成后，使用 score 子命令提交完整CSV。评分文件表头为 item_id,answer；无法辨认可填写`unclear`或`无法辨认`，入口会按同一选项归档。"
    )
    (target / "review_instructions.md").write_text(instructions, encoding="utf-8", newline="\n")
    return {"mapping": mapping, "public_items": public, "eligibility": eligibility}


def _exact_mcnemar_p(wins_b_only: int, wins_a_only: int) -> float:
    _require(wins_b_only >= 0 and wins_a_only >= 0, "McNemar discordant counts cannot be negative")
    discordant = wins_b_only + wins_a_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins_b_only, wins_a_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _validate_review_mapping(mapping: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_ids = {f"FJ{index:03d}" for index in range(1, ITEM_COUNT + 1)}
    ids = {str(row.get("item_id", "")) for row in mapping}
    _require(len(mapping) == ITEM_COUNT and len(ids) == ITEM_COUNT and ids == expected_ids, "private review map must contain this packet's 270 unique item IDs")
    _require([int(row["question_number"]) for row in mapping] == list(range(1, ITEM_COUNT + 1)), "review question order is incomplete or duplicated")
    _require(all(str(row["item_id"]) == f"FJ{int(row['question_number']):03d}" for row in mapping), "review item IDs do not match the anonymous question order")
    counts = {kind: sum(row["source_type"] == kind for row in mapping) for kind in ("A5400_primary", "B5400_primary", "A5400_repeat", "B5400_repeat", "real_control", "black_control")}
    _require(counts == {"A5400_primary": 100, "B5400_primary": 100, "A5400_repeat": 20, "B5400_repeat": 20, "real_control": 20, "black_control": 10}, "private review map composition differs")
    first_counts = {arm: sum(row["source_type"] == f"{arm}_primary" and int(row["question_number"]) <= 100 for row in mapping) for arm in ("A5400", "B5400")}
    _require(first_counts == {"A5400": 50, "B5400": 50}, "first-presented endpoint balance differs")
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in mapping:
        if row["source_type"].endswith("_primary"):
            by_pair.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = row
    _require(len(by_pair) == PRIMARY_PAIR_COUNT and all(set(row) == {"A5400", "B5400"} for row in by_pair.values()), "100 same-input A/B pairs are incomplete")
    pair_strata: dict[tuple[int, int], set[int]] = {}
    for pair in by_pair.values():
        a_row, b_row = pair["A5400"], pair["B5400"]
        _require(a_row["requested_class"] == b_row["requested_class"] and a_row["seed"] == b_row["seed"] and a_row["draw_index"] == b_row["draw_index"], "paired images do not share request and sample identity")
        digit, seed, draw = int(a_row["requested_class"]), int(a_row["seed"]), int(a_row["draw_index"])
        _require(0 <= digit < CLASS_COUNT and seed in SEEDS and 0 <= draw < DRAW_COUNT, "primary pair seed, class, or draw index is invalid")
        _require(str(a_row["pair_id"]) == f"{seed}:{digit}:{draw}" and str(b_row["pair_id"]) == str(a_row["pair_id"]), "primary pair ID does not match its saved sample identity")
        _require(abs(int(a_row["question_number"]) - int(b_row["question_number"])) >= MIN_SPACING, "matched A/B primary images are too close")
        stratum = (seed, digit)
        pair_strata.setdefault(stratum, set()).add(draw)
    _require(len(pair_strata) == 50 and all(len(draws) == 2 for draws in pair_strata.values()), "pair sample does not contain two distinct draws per seed and class")
    by_id = {str(row["item_id"]): row for row in mapping}
    repeats_by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    repeats_per_class = {"A5400": {digit: 0 for digit in range(CLASS_COUNT)}, "B5400": {digit: 0 for digit in range(CLASS_COUNT)}}
    for row in mapping:
        if row["source_type"].endswith("_repeat"):
            parent_row = by_id.get(str(row.get("repeat_of")))
            _require(parent_row is not None and parent_row["arm"] == row["arm"] and parent_row["source_type"] == f"{row['arm']}_primary", "repeat parent source/arm differs")
            _require(parent_row["image_sha256"] == row["image_sha256"], "repeat image identity differs")
            _require(abs(int(parent_row["question_number"]) - int(row["question_number"])) >= MIN_SPACING, "hidden repeat is too close to its original")
            _require(parent_row["pair_id"] == row["pair_id"] and parent_row["seed"] == row["seed"] and parent_row["draw_index"] == row["draw_index"], "repeat source identity differs")
            repeats_by_pair.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = row
            repeats_per_class[str(row["arm"])][int(row["requested_class"])] += 1
    _require(len(repeats_by_pair) == 20 and all(set(value) == {"A5400", "B5400"} for value in repeats_by_pair.values()), "each hidden pair must repeat both endpoints")
    _require(all(set(repeats_per_class[arm].values()) == {2} for arm in repeats_per_class), "hidden repeats do not cover two pairs per class")
    for pair_id, repeated in repeats_by_pair.items():
        _require(abs(int(repeated["A5400"]["question_number"]) - int(repeated["B5400"]["question_number"])) >= MIN_SPACING, f"repeated matched pair {pair_id} is too close")
    real_by_class = {digit: 0 for digit in range(CLASS_COUNT)}
    for row in mapping:
        if row["source_type"] == "real_control":
            digit = int(row["true_class"])
            _require(0 <= digit < CLASS_COUNT, "true-digit control class is invalid")
            real_by_class[digit] += 1
    _require(set(real_by_class.values()) == {2}, "true-digit controls are not two per class")
    return {"counts": counts, "pairs": by_pair, "by_id": by_id}

def _parse_ratings(path: Path, mapping: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    _require(path.is_file(), f"explicit ratings CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == ["item_id", "answer"], "ratings CSV header must be item_id,answer")
        source_rows = list(reader)
    expected = {str(row["item_id"]) for row in mapping}
    answers: dict[str, str] = {}
    preserved = []
    for row in source_rows:
        item_id = str(row.get("item_id", "")).strip()
        raw_answer = str(row.get("answer", "")).strip()
        answer = "unclear" if raw_answer.lower() == "unclear" or raw_answer == "无法辨认" else raw_answer.lower()
        _require(item_id in expected, f"unknown rating item ID: {item_id}")
        _require(item_id not in answers, f"duplicate rating item ID: {item_id}")
        _require(answer in ALLOWED_ANSWERS, f"missing or invalid answer for item {item_id}")
        answers[item_id] = answer
        preserved.append({"item_id": item_id, "answer": answer})
    _require(len(preserved) == ITEM_COUNT and set(answers) == expected, "ratings must include all 270 items exactly once")
    return answers, preserved


def _score_results(mapping: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> dict[str, Any]:
    validated = _validate_review_mapping(mapping)
    _require(set(answers) == set(validated["by_id"]), "ratings do not cover the private map exactly")
    first_presentations = {pair_id: dict(pair) for pair_id, pair in validated["pairs"].items()}
    for row in mapping:
        if row["source_type"].endswith("_repeat"):
            selected = first_presentations[str(row["pair_id"])][str(row["arm"])]
            if int(row["question_number"]) < int(selected["question_number"]):
                first_presentations[str(row["pair_id"])][str(row["arm"])] = row
    first_endpoint_counts = {"A5400": 0, "B5400": 0}
    earlier_repeat_counts = {"A5400": 0, "B5400": 0}
    for pair in first_presentations.values():
        first_arm = min(pair, key=lambda arm: int(pair[arm]["question_number"]))
        first_endpoint_counts[first_arm] += 1
        _require(abs(int(pair["A5400"]["question_number"]) - int(pair["B5400"]["question_number"])) >= MIN_SPACING, "first-presented A/B images are too close")
        for arm, row in pair.items():
            earlier_repeat_counts[arm] += int(row["source_type"].endswith("_repeat"))
    _require(first_endpoint_counts == {"A5400": 50, "B5400": 50}, "actual first-presented endpoints are not balanced")
    real = [row for row in mapping if row["source_type"] == "real_control"]
    black = [row for row in mapping if row["source_type"] == "black_control"]
    a_repeats = [row for row in mapping if row["source_type"] == "A5400_repeat"]
    b_repeats = [row for row in mapping if row["source_type"] == "B5400_repeat"]
    real_correct = sum(answers[str(row["item_id"])] == str(row["true_class"]) for row in real)
    black_unclear = sum(answers[str(row["item_id"])] == "unclear" for row in black)

    def repeat_agreement(rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(answers[str(row["item_id"])] == answers[str(row["repeat_of"])] for row in rows)

    a_agree, b_agree = repeat_agreement(a_repeats), repeat_agreement(b_repeats)
    quality = {
        "real_digit_correct": real_correct, "real_digit_total": REAL_CONTROL_COUNT, "real_digit_passed": real_correct >= 18,
        "black_answered_unclear": black_unclear, "black_total": BLACK_CONTROL_COUNT, "black_passed": black_unclear >= 9,
        "A_repeat_agree": a_agree, "A_repeat_total": REPEATS_PER_ARM, "A_repeat_passed": a_agree >= 18,
        "B_repeat_agree": b_agree, "B_repeat_total": REPEATS_PER_ARM, "B_repeat_passed": b_agree >= 18,
    }
    quality["passed"] = bool(quality["real_digit_passed"] and quality["black_passed"] and quality["A_repeat_passed"] and quality["B_repeat_passed"])

    pair_rows = []
    wins_b_only = wins_a_only = both = neither = 0
    by_class = {digit: {"n": 0, "A_recognized": 0, "B_recognized": 0, "B_only": 0, "A_only": 0} for digit in range(CLASS_COUNT)}
    for pair_id in sorted(first_presentations):
        pair = first_presentations[pair_id]
        a_row, b_row = pair["A5400"], pair["B5400"]
        answer_a, answer_b = answers[str(a_row["item_id"])], answers[str(b_row["item_id"])]
        requested = int(a_row["requested_class"])
        recognized_a, recognized_b = int(answer_a == str(requested)), int(answer_b == str(requested))
        wins_b_only += int(not recognized_a and recognized_b)
        wins_a_only += int(recognized_a and not recognized_b)
        both += int(recognized_a and recognized_b)
        neither += int(not recognized_a and not recognized_b)
        entry = by_class[requested]
        entry["n"] += 1
        entry["A_recognized"] += recognized_a
        entry["B_recognized"] += recognized_b
        entry["B_only"] += int(not recognized_a and recognized_b)
        entry["A_only"] += int(recognized_a and not recognized_b)
        pair_rows.append({
            "pair_id": pair_id, "seed": int(a_row["seed"]), "requested_class": requested,
            "draw_index": int(a_row["draw_index"]), "A_item_id": str(a_row["item_id"]),
            "B_item_id": str(b_row["item_id"]), "A_answer": answer_a, "B_answer": answer_b,
            "A_question_number": int(a_row["question_number"]), "B_question_number": int(b_row["question_number"]),
            "A_designated_primary_item_id": str(validated["pairs"][pair_id]["A5400"]["item_id"]),
            "B_designated_primary_item_id": str(validated["pairs"][pair_id]["B5400"]["item_id"]),
            "A_recognized_request": recognized_a, "B_recognized_request": recognized_b,
            "outcome": "B_only" if not recognized_a and recognized_b else "A_only" if recognized_a and not recognized_b else "both" if recognized_a and recognized_b else "neither",
        })
    _require(len(pair_rows) == PRIMARY_PAIR_COUNT, "primary human comparison does not contain 100 pairs")
    d_human = (wins_b_only - wins_a_only) / PRIMARY_PAIR_COUNT
    p_value = _exact_mcnemar_p(wins_b_only, wins_a_only)
    if not quality["passed"]:
        classification = "REVIEW_INCONCLUSIVE"
        reason = "REVIEW_QUALITY_CHECK_FAILED"
    elif d_human > 0.0 and p_value <= ALPHA:
        classification = "B5400_SINGLE_REVIEWER_RELATIVE_READABILITY_SUPPORTED"
        reason = "PAIRED_HUMAN_RECOGNITION_FAVORS_B"
    elif d_human < 0.0 and p_value <= ALPHA:
        classification = "B5400_SINGLE_REVIEWER_RELATIVE_READABILITY_WORSE"
        reason = "PAIRED_HUMAN_RECOGNITION_FAVORS_A"
    else:
        classification = "EVIDENCE_INSUFFICIENT"
        reason = "MCNEMAR_P_ABOVE_ALPHA"
    source_reading = {}
    for arm in ("A5400", "B5400"):
        rows = [pair[arm] for pair in first_presentations.values()]
        recognized = sum(answers[str(row["item_id"])] == str(row["requested_class"]) for row in rows)
        unclear = sum(answers[str(row["item_id"])] == "unclear" for row in rows)
        per_class = {}
        for digit in range(CLASS_COUNT):
            class_rows = [row for row in rows if int(row["requested_class"]) == digit]
            per_class[str(digit)] = {
                "n": len(class_rows),
                "recognized": sum(answers[str(row["item_id"])] == str(digit) for row in class_rows),
                "unclear": sum(answers[str(row["item_id"])] == "unclear" for row in class_rows),
            }
        source_reading[arm] = {
            "n": len(rows), "matches_requested_class": recognized, "fraction": recognized / len(rows),
            "unclear": unclear, "unclear_fraction": unclear / len(rows), "per_class": per_class,
        }
    return {
        "classification": classification, "reason": reason, "quality_checks": quality,
        "primary": {
            "estimand": "B-only requested-class recognitions minus A-only requested-class recognitions divided by 100 matched inputs",
            "pairs": PRIMARY_PAIR_COUNT, "B_only": wins_b_only, "A_only": wins_a_only,
            "both_recognized": both, "neither_recognized": neither,
            "D_human": d_human, "mcnemar_exact_two_sided_p": p_value, "alpha": ALPHA,
            "answer_selection": "earliest_question_number_per_pair_and_endpoint",
            "first_presented_endpoint_counts": first_endpoint_counts,
            "earlier_designated_repeat_used": earlier_repeat_counts,
            "exact_test_scope": "conditional binomial tail for discordant pairs; not a finite-population confidence interval",
        },
        "source_reading": source_reading,
        "per_class_pairs": {str(digit): by_class[digit] for digit in range(CLASS_COUNT)},
        "repeat_consistency": {"A5400": {"agree": a_agree, "total": REPEATS_PER_ARM}, "B5400": {"agree": b_agree, "total": REPEATS_PER_ARM}},
        "pair_rows": pair_rows,
        "scope_limit": "one reviewer; fixed saved prior; A5400 and B5400 endpoints; all targets and fit rows are canonical TRAIN",
        "classifier_proxy_validity_claim": False,
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    result = report["human_results"]
    primary = result["primary"]
    quality = result["quality_checks"]
    numeric = report["numerical_auxiliary"]["shape_statistics"]["confirmation"]["28"]
    da = numeric["B_minus_A"]["energy_distance"]
    status = report["status"]
    quality_text = "通过" if quality["passed"] else "未全部通过"
    return f"""# 固定联合先验下 A5400/B5400 采样识读对照

## 结果与状态

状态：{status}；盲判测量已归档。人工质控{quality_text}：真实数字 {quality['real_digit_correct']}/20，全黑图“无法辨认” {quality['black_answered_unclear']}/10，A重复 {quality['A_repeat_agree']}/20，B重复 {quality['B_repeat_agree']}/20。最终分类：{result['classification']}。

## 科学认识

主分析使用每个输入、每个端点按题号最早展示的答案。100个同输入配对中，B单独被认作请求数字 {primary['B_only']} 对，A单独被认作请求数字 {primary['A_only']} 对；双方均认可 {primary['both_recognized']} 对，双方均未认可 {primary['neither_recognized']} 对。人工差值 D_h={primary['D_human']:.4f}，双侧精确 McNemar p={primary['mcnemar_exact_two_sided_p']:.6g}。质控通过且方向性差异达到预设显著性时才支持单人相对识读差异；该检验不提供1500行固定库的精确置信区间。

数值辅助方面，确认人口上的28×28归一化形态集合距离 B−A={da:.8g}。它不是人工判读结论。两臂共使用相同1500个保存的复OLS输入；未计算任意输入与真实目标的配对Q或SSIM。逐类别、逐种子、200×200结果及距离分解见数值辅助记录和 class_distances.csv。

## 目标缺口

这是单人判断，不能推广到审阅者人群；所有拟合与确认目标都来自 canonical TRAIN。数值形态距离不能证明可读性、独立泛化、覆盖、新颖性或模型采用。本结果也不修复旧实验的分类代理效度质控缺口。

## 工程完成与验证

使用固定A5400/B5400检查点和联合先验 attempt_02 中保存的complex64采样库，没有重新拟合或生成噪声。90行冻结CUDA预检通过；正式解码3000行，严格FP32单行前向，50行分块；重读分块后核验了输入配对、生产视图、形态距离分解及最终人工计算。评分文件共 {report['validation']['rating_rows']} 行，盲判题目共270项。

## 决策影响与下一步

{result['classification']}仅用于当前固定先验和单人识读对照。即使获得相对支持，也只构成进一步规划覆盖和独立人口确认的依据；不授权续训、调参或采用B5400。保持 training_recommendation=NONE、next_action=NONE。
"""


def _score_output(output_dir: Path, ratings_path: Path) -> dict[str, Any]:
    prepared = _json_read(output_dir / "prepared.json")
    _require(prepared.get("task_id") == TASK_ID and prepared.get("state") == "PREPARED_AWAITING_RATINGS", "review packet is not awaiting ratings")
    _require(not (output_dir / "completed.json").exists(), "this review packet has already been scored")
    map_payload = _json_read(output_dir / "private" / "review_mapping.json")
    _require(map_payload.get("task_id") == TASK_ID, "private review mapping task identity differs")
    mapping = map_payload["items"]
    answers, preserved = _parse_ratings(ratings_path.resolve(), mapping)
    results = _score_results(mapping, answers)
    submission = output_dir / "ratings_submission.csv"
    _require(not submission.exists(), "ratings submission has already been preserved")
    shutil.copyfile(ratings_path.resolve(), submission)
    _write_csv_new(output_dir / "analysis" / "raw_answers.csv", preserved, ["item_id", "answer"])
    all_rows = [{**row, "answer": answers[str(row["item_id"])]} for row in mapping]
    all_fields = list(dict.fromkeys(key for row in all_rows for key in row.keys()))
    _write_csv_new(output_dir / "analysis" / "review_item_rows.csv", all_rows, all_fields)
    _write_csv_new(output_dir / "analysis" / "paired_rows.csv", results["pair_rows"])
    validation = {
        "passed": True, "rating_rows": len(preserved), "mapping_rows": len(mapping),
        "all_item_ids_unique": True, "complete_and_valid": True,
        "private_mapping_validated": True, "paired_input_identity_checked": True,
    }
    numeric_report = _json_read(output_dir / "measurement_report.json")
    report = {
        "task_id": TASK_ID, "status": "VALID_COMPLETED",
        "classification": results["classification"], "human_results": results,
        "numerical_auxiliary": numeric_report["numerical_auxiliary"],
        "validation": validation, "scope": numeric_report["scope"],
        "preflight": numeric_report["preflight"],
        "training_recommendation": "NONE", "next_action": "NONE", "stop": True,
    }
    _json_write_new(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8", newline="\n")
    _json_write_new(output_dir / "completed.json", {
        "task_id": TASK_ID, "status": "COMPLETED", "classification": results["classification"],
        "review_quality": "PASSED" if results["quality_checks"]["passed"] else "FAILED",
        "training_recommendation": "NONE", "next_action": "NONE",
    })
    _json_write_replace(output_dir / "measurement_status.json", {
        "task_id": TASK_ID, "state": "COMPLETED",
        "human_classification": results["classification"],
        "review_quality_passed": results["quality_checks"]["passed"],
        "training_recommendation": "NONE", "next_action": "NONE",
    })
    return report
def _verify_paired_input_banks(target: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    total = 0
    for seed_index, seed in enumerate(SEEDS):
        arm_inputs = {}
        for arm in ("A5400", "B5400"):
            raw, view, actual_z = _load_seed_readout(target, arm, seed)
            expected_z = inputs["prior"]["bank"]["joint"][seed_index].reshape(300, PORTS)
            _require(torch.equal(actual_z, expected_z), f"{arm} seed {seed} inputs differ from the persisted joint bank")
            arm_inputs[arm] = actual_z
            total += actual_z.shape[0]
            del raw, view
        _require(torch.equal(arm_inputs["A5400"], arm_inputs["B5400"]), f"A5400/B5400 did not use paired inputs at seed {seed}")
    _require(total == FORMAL_ROWS_PER_ARM * 2, "paired input validation count differs")
    return {"passed": True, "paired_rows_per_endpoint": FORMAL_ROWS_PER_ARM, "total_endpoint_rows": total, "all_complex_OLS_inputs_identical": True}


def _write_numeric_rows_v2(target: Path, row_sets: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows = [dict(row) for arm in ("A5400", "B5400") for row in row_sets[arm]]
    _require(len(rows) == FORMAL_ROWS_PER_ARM * 2, "numeric result row count differs")
    fields = [
        "arm", "seed", "label", "draw_index", "pair_id", "position", "e_d_prediction", "e_d_correct",
        "camera_integrated_intensity", "generator_input_power", "raw_clip_fraction", "e_d_target_logit",
    ]
    path = target / "numeric_rows.csv"
    if path.exists():
        _require(_read_csv(path) == [{key: str(row[key]) for key in fields} for row in rows], "saved numeric CSV differs from reloaded readout chunks")
    else:
        _write_csv_new(path, rows, fields)
    return rows


def _recheck_statistics(target: Path, class_rows: Sequence[Mapping[str, Any]], auxiliary: Mapping[str, Any]) -> dict[str, Any]:
    path = target / "class_distances.csv"
    if not path.exists():
        _write_csv_new(path, class_rows)
    loaded = _read_csv(path)
    _require(len(loaded) == len(class_rows) == 400, "saved class-distance CSV row count differs")
    fields = ("energy_distance", "cross_distance", "within_output_distance", "within_target_distance")
    for saved, expected in zip(loaded, class_rows, strict=True):
        for key in ("arm", "seed", "label", "target_role", "shape_size"):
            _require(saved[key] == str(expected[key]), f"saved class-distance CSV identity differs for {key}")
        for key in fields:
            _require(math.isclose(float(saved[key]), float(expected[key]), rel_tol=1.0e-12, abs_tol=1.0e-14), f"saved class-distance CSV differs for {key}")
    checks = {}
    for role in ("fit", "confirmation"):
        checks[role] = {}
        for size in (28, 200):
            arm_means = {}
            for arm in ("A5400", "B5400"):
                selected = [row for row in loaded if row["arm"] == arm and row["target_role"] == role and int(row["shape_size"]) == size]
                _require(len(selected) == 50, f"saved class-distance CSV has wrong {arm}/{role}/{size} count")
                arm_means[arm] = {field: math.fsum(float(row[field]) for row in selected) / len(selected) for field in fields}
                expected = auxiliary["shape_statistics"][role][str(size)][arm]
                for field in fields:
                    _require(math.isclose(arm_means[arm][field], expected[field], rel_tol=1.0e-12, abs_tol=1.0e-14), f"independently reloaded {field} mean differs")
            delta = {field: arm_means["B5400"][field] - arm_means["A5400"][field] for field in fields}
            reconstructed = 2.0 * delta["cross_distance"] - delta["within_output_distance"]
            residual = delta["energy_distance"] - reconstructed
            _require(abs(delta["within_target_distance"]) <= 1.0e-14, "the two arms do not share the same target-distance term")
            _require(abs(residual) <= 1.0e-10, f"shape-distance decomposition identity failed for {role}/{size}")
            checks[role][str(size)] = {
                "decomposition_identity": True,
                "delta_shape": delta["energy_distance"],
                "reconstructed_delta_shape": reconstructed,
                "residual": residual,
            }
    return {"passed": True, "class_rows": len(loaded), "reloaded_csv_independent_aggregate": True, "delta_shape_decomposition": checks}

def _ready_context(target: Path, inputs: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt, dict[str, Any]]:
    saved = _json_read(target / "preflight.json")
    _require(saved.get("task_id") == TASK_ID and saved.get("status") == "READY", "formal readout requires a saved READY preflight")
    _require(_json_read(target / "input_population.json") == inputs["population"], "saved READY population differs from current source")
    preflight = saved["preflight"]
    checks = preflight.get("checks", {})
    _require(preflight.get("status") == "READY" and checks and all(row.get("executed") is True and row.get("passed") is True for row in checks.values()), "READY consumer found unexecuted or failed checks")
    receipt = PreflightReceipt.from_mapping(saved["receipt"])
    _require(receipt.source_semantics_id == TASK_ID and receipt.authorization_id == f"{TASK_ID}:authorization:01" and receipt.status == "READY", "READY receipt identity differs")
    _require(all(value.get("passed") is True for value in receipt.checks.values()), "READY receipt contains missing or failed check groups")
    control = ResearchRunControl(_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    control.mark_preflight_ready(receipt)
    return control, receipt, preflight


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    target = _prepare_output(root, args.output_dir, resume=args.resume)
    if not args.resume:
        _write_plan(target)
    started = time.perf_counter()
    try:
        artifact_dir = _resolve(root, args.artifact_dir)
        training_dir = _resolve(root, args.training_dir)
        prior_dir = _resolve(root, args.prior_dir)
        checkpoint_a = _resolve(root, args.checkpoint_a)
        checkpoint_b = _resolve(root, args.checkpoint_b)
        inputs = _load_inputs(root, artifact_dir, training_dir, prior_dir, checkpoint_a, checkpoint_b)
        inputs["artifact_dir"] = artifact_dir
        inputs["historical_joint"] = _load_historical_joint(prior_dir)
        _persist_identity(target, inputs, resume=args.resume)
        _require(not (target / "completed.json").exists(), "this attempt is already scored and completed")
        if args.resume and (target / "preflight.json").is_file():
            _ready_context(target, inputs)
        with training.capacity.strict_fp32_mode() as numeric_mode:
            _require(numeric_mode == training.STRICT_FP32_SETTINGS, "strict FP32 runtime settings differ")
            resource = _resource_observation(target)
            if (target / "preflight.json").is_file():
                control, receipt, preflight = _ready_context(target, inputs)
            else:
                device = torch.device("cuda:0")
                preflight = _preflight_sequential(inputs, target, device)
                preflight["resources_observed"] = resource
                control, receipt = _make_ready_receipt(inputs, preflight)
                _stage_ready(target, inputs, preflight, receipt, resume=False)
            if args.preflight_only:
                return {
                    "task_id": TASK_ID, "state": "READY_AWAITING_FORMAL_READOUT",
                    "output_dir": target.as_posix(), "checks": preflight["checks"],
                    "next_action": "NONE",
                }
            control.start_execution()
            control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
            row_sets = {}
            for arm, checkpoint in (("A5400", inputs["checkpoint_a"]), ("B5400", inputs["checkpoint_b"])):
                row_sets[arm] = _run_arm_fixed(target, inputs, arm, checkpoint, torch.device("cuda:0"))
            paired_validation = _verify_paired_input_banks(target, inputs)
            numeric_rows = _write_numeric_rows_v2(target, row_sets)
            reread_numeric = _read_csv(target / "numeric_rows.csv")
            fields = ["arm", "seed", "label", "draw_index", "pair_id", "position", "e_d_prediction", "e_d_correct", "camera_integrated_intensity", "generator_input_power", "raw_clip_fraction", "e_d_target_logit"]
            expected_numeric = [{key: str(row[key]) for key in fields} for arm in ("A5400", "B5400") for row in row_sets[arm]]
            _require(len(reread_numeric) == FORMAL_ROWS_PER_ARM * 2 and reread_numeric == expected_numeric, "saved numeric CSV differs from the 3000 chunk rows")
            class_rows, auxiliary = _compute_shape_statistics(target, inputs)
            statistics_validation = _recheck_statistics(target, class_rows, auxiliary)
            review_packet = _prepare_review(target, inputs) if not (target / "blind_review.html").exists() else {
                "mapping": _json_read(target / "private" / "review_mapping.json")["items"],
                "eligibility": _json_read(target / "eligibility.json"),
            }
            _require(len(review_packet["mapping"]) == ITEM_COUNT, "prepared review map is incomplete")
            control.complete()
        validation = {
            "task_id": TASK_ID, "status": "VALIDATED",
            "preflight_ready": True, "preflight_forward_rows": preflight["forward_rows"],
            "formal_forward_rows": FORMAL_ROWS_PER_ARM * 2,
            "saved_numeric_rows": len(numeric_rows),
            "paired_input_bank": paired_validation,
            "statistics_and_csv_recheck": statistics_validation,
            "review_packet": review_packet["eligibility"],
            "checkpoint_and_evaluator_frozen": True,
        }
        _json_write_replace(target / "validation.json", validation) if (target / "validation.json").exists() else _json_write_new(target / "validation.json", validation)
        main_numeric = auxiliary["shape_statistics"]["confirmation"]["28"]["B_minus_A"]["energy_distance"]
        measurement_report = {
            "task_id": TASK_ID, "status": "NUMERIC_COMPLETED_AWAITING_RATINGS",
            "classification": "PENDING_HUMAN_RATINGS",
            "scope": {
                "source_attempt": artifact_dir.as_posix(), "training_attempt": training_dir.as_posix(),
                "prior_attempt": prior_dir.as_posix(),
                "checkpoints": {"A": checkpoint_a.as_posix(), "B": checkpoint_b.as_posix(), "steps": [5400, 5400]},
                "fit_rows": 200, "confirmation_rows": 300, "population_rows": 500,
                "all_rows_canonical_train": True,
                "formal_sample_rows_per_arm": FORMAL_ROWS_PER_ARM,
                "all_formal_OLS_inputs_shared_between_arms": True,
                "model_updates": 0, "optimizer_updates": 0, "gradient_calls": 0,
                "reviewer_count": 1, "review_items": ITEM_COUNT,
            },
            "measurement": {
                "prior_source": prior_dir.as_posix(), "prior_refit": False,
                "seeds": list(SEEDS), "draws_per_seed_class": DRAW_COUNT,
                "numeric_primary_is_auxiliary": True,
                "shape_view": "Pool28 and Pool200 of raw CMOS; normalize each vector by total mass without clipping",
                "distance": "2 mean Euclidean(output,target) - mean Euclidean(output,output) - mean Euclidean(target,target), with V-statistic diagonals",
                "human_primary": "paired request-class recognitions and exact two-sided McNemar, alpha 0.05; numeric shape is auxiliary",
            },
            "numerical_auxiliary": auxiliary,
            "numerical_summary": {
                "confirmation_28_B_minus_A": main_numeric,
                "interpretation": "descriptive only; not the scientific decision",
            },
            "preflight": preflight,
            "validation": validation,
            "research_run_state": control.state,
            "training_recommendation": "NONE",
            "next_action": "WAIT_FOR_RATINGS",
            "elapsed_seconds_observation_only": time.perf_counter() - started,
        }
        if not (target / "measurement_report.json").exists():
            _json_write_new(target / "measurement_report.json", measurement_report)
        prepared = {
            "task_id": TASK_ID, "state": "PREPARED_AWAITING_RATINGS",
            "numeric_status": "VALID_COMPLETED",
            "review_status": "PREPARED",
            "reviewer_count": 1, "item_count": ITEM_COUNT, "page_size": PAGE_SIZE,
            "composition": review_packet["eligibility"]["composition"],
            "training_recommendation": "NONE", "next_action": "WAIT_FOR_RATINGS",
        }
        _json_write_replace(target / "prepared.json", prepared) if (target / "prepared.json").exists() else _json_write_new(target / "prepared.json", prepared)
        _json_write_replace(target / "measurement_status.json", prepared) if (target / "measurement_status.json").exists() else _json_write_new(target / "measurement_status.json", prepared)
        if not (target / "report.md").exists():
            confirmation28 = auxiliary["shape_statistics"]["confirmation"]["28"]
            confirmation200 = auxiliary["shape_statistics"]["confirmation"]["200"]
            e_d = auxiliary["E_D_auxiliary"]
            energy = auxiliary["operational_auxiliary"]
            (target / "report.md").write_text(
                "# 固定联合先验下 A5400/B5400 采样与单人识读对照\n\n"
                "## 结果与状态\n\n"
                "3000行冻结解码、保存重读和270题单人盲判包准备均已完成。当前为`PREPARED_AWAITING_RATINGS`；人工主判读尚未产生，尚无科学分类或完成标记。\n\n"
                "## 科学认识\n\n"
                "唯一科学判读是100个同复OLS输入配对上的人工请求类别识读差和精确双侧McNemar检验。评分CSV尚未提交，因此现在不能判断B5400是否比A5400更容易被审阅者识读。\n\n"
                f"数值辅助：确认人口28×28形态集合距离A5400={confirmation28['A5400']['energy_distance']:.8g}、B5400={confirmation28['B5400']['energy_distance']:.8g}、B−A={confirmation28['B_minus_A']['energy_distance']:.8g}；200×200 B−A={confirmation200['B_minus_A']['energy_distance']:.8g}。E_D为A5400 {e_d['A5400']['correct']}/1500、B5400 {e_d['B5400']['correct']}/1500；平均CMOS积分强度代理分别为{energy['A5400']['camera_integrated_intensity']['mean']:.8g}与{energy['B5400']['camera_integrated_intensity']['mean']:.8g}。这些仅作辅助，不能替代人工判读。\n\n"
                "## 目标缺口\n\n"
                "尚缺完整270行评分CSV。真实数字正控需至少18/20、全黑至少9/10、A与B隐藏重复分别至少18/20；全部通过后才消费配对主判读。所有目标行来自canonical TRAIN，不构成独立泛化证据。\n\n"
                "## 工程完成与验证\n\n"
                "90行CUDA预检12项均通过；3000行使用同一保存联合先验输入、严格FP32 CUDA和50行落盘，共60块。CPU float64形态统计、配对输入身份、500行目标角色统计、盲态字段和重复间隔已重读核验。工程失败记录保留于`engineering_error.json`及`engineering_error_recheck_02.json`，后续修复后的READY预检通过。\n\n"
                "## 决策影响与下一步\n\n"
                "请打开`blind_review.html`完成全部270题，不查看`private/review_mapping.json`。完成后导出完整评分CSV，并提交：\n\n"
                "```powershell\n"
                "python -B -m scripts.research.lightgen.audit_lightgen_2d_generator_fixed_prior_sampled_readability_v1 score `\n"
                f"  --output-dir {target.as_posix()} `\n"
                "  --ratings <完整评分CSV路径>\n"
                "```\n\n"
                "此停止点不续训、不调权、不重拟合先验或采用B5400；保持`training_recommendation=NONE`、`next_action=WAIT_FOR_RATINGS`。\n",
                encoding="utf-8", newline="\n",
            )
        return prepared
    except Exception as exc:
        failure_path = target / "engineering_error.json"
        if failure_path.exists():
            recheck = 2
            failure_path = target / f"engineering_error_recheck_{recheck:02d}.json"
            while failure_path.exists():
                recheck += 1
                failure_path = target / f"engineering_error_recheck_{recheck:02d}.json"
        status = "ENGINEERING_BLOCKED_CUDA" if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc).upper() else "INVALID_HARNESS"
        _json_write_new(failure_path, {
            "task_id": TASK_ID, "status": status,
            "error_type": type(exc).__name__, "message": str(exc),
            "scientific_classification": None,
            "training_recommendation": "NONE", "next_action": "NONE",
        })
        raise


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--artifact-dir", required=True, type=Path)
    prepare.add_argument("--training-dir", required=True, type=Path)
    prepare.add_argument("--prior-dir", required=True, type=Path)
    prepare.add_argument("--checkpoint-a", required=True, type=Path)
    prepare.add_argument("--checkpoint-b", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--preflight-only", action="store_true")
    prepare.add_argument("--resume", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("--output-dir", required=True, type=Path)
    score.add_argument("--ratings", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.command == "prepare":
        result = _prepare(args)
        print(json.dumps({"task_id": TASK_ID, "state": result.get("state"), "output_dir": str(args.output_dir.resolve()), "next_action": result.get("next_action")}, ensure_ascii=False))
    else:
        report = _score_output(args.output_dir.resolve(), args.ratings)
        print(json.dumps({"task_id": TASK_ID, "state": report.get("status"), "classification": report.get("classification"), "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
