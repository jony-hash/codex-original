"""Freeze Generator A4700 and test a class-conditional diagonal complex-OLS sampler."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
)
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as class9
from scripts.research.lightgen import execute_lightgen_2d_generator_training_population_expansion_comparison_v1 as expansion


TASK_ID = "M0_GENERATOR_FROZEN_CONDITIONAL_OLS_GENERATION_V1"
PARENT_TASK_ID = expansion.TASK_ID
REPORT_SCHEMA = "m0_generator_frozen_conditional_ols_generation_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_frozen_conditional_ols_generation_preflight.v1"
DEFAULT_POPULATION = Path("outputs/20261008_m0_generator_training_population_expansion_comparison_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_frozen_conditional_ols_generation_v1/runs/attempt_01")
SOURCE_STEP = 4_700
FIT_ROWS_PER_CLASS = 20
FIT_ROWS = 200
ROLE_CONFIRMATION = "coverage_confirmation"
CONFIRMATION_START, CONFIRMATION_STOP = 3_000, 3_300
SAMPLES_PER_CLASS = 200
SAMPLE_COUNT = 2_000
MEAN_COUNT = 10
TOTAL_OUTPUTS = FIT_ROWS + MEAN_COUNT + SAMPLE_COUNT
OLS_DIM = 100
CHUNK_ROWS = 100
INFERENCE_MICROBATCH_ROWS = 1
SEED = 2026100813
STD_FLOOR = 1.0e-8
BATCH_SIZE = 10
RTOL, ATOL = expansion.RTOL, expansion.ATOL
EVALUATOR_NAMES = capacity.EVALUATOR_NAMES
ROLE_POSITIONS, ROLE_COUNTS, ROLE_NAMES = expansion._role_partition()
KNOWN_OUTPUT_FILES = {
    "corrected_input_population.json",
    "population_evidence_review.json",
    "corrected_comparison_pages.json",
    "input_population.json",
    "preflight.json",
    "manifest.json",
    "preflight_target_controls.pt",
    "preflight_first_class_ols.pt",
    "preflight_io_roundtrip.pt",
    "conditional_ols_bank.pt",
    "sample_row_metrics.pt",
    "report.json",
    "invalid_harness.json",
}


class FrozenGenerationError(RuntimeError):
    """Input, replay, freeze, preflight, or measurement contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenGenerationError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _write_json(path: Path, value: Mapping[str, Any], kind: str) -> None:
    del kind
    _write_new_artifact(path, lambda handle: handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")))


def _write_torch(path: Path, value: Mapping[str, Any], kind: str) -> None:
    del kind
    _write_new_artifact(path, lambda handle: torch.save(dict(value), handle))


def _write_new_artifact(path: Path, writer: Any) -> None:
    """Publish a new task artifact atomically without the sandbox-blocked hard-link path."""
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".pending", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows os.rename fails if the destination already exists.
        os.rename(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _wilson(successes: int, total: int, z_value: float = 1.959963984540054) -> list[float]:
    return diagnostic._wilson(successes, total, z_value)


def _corrected_population(source_population: Mapping[str, Any]) -> dict[str, Any]:
    rows = source_population.get("source_row_indices")
    labels = source_population.get("labels")
    _require(isinstance(rows, list) and len(rows) == 3_300, "source population does not contain 3,300 ordered rows")
    _require(isinstance(labels, list) and len(labels) == 3_300, "source population labels do not contain 3,300 entries")
    _require(len(set(rows)) == 3_300, "source population row indices are not unique")
    corrected = dict(source_population)
    corrected["role_positions"] = ROLE_POSITIONS
    corrected["role_counts"] = ROLE_COUNTS
    corrected["roles"] = ROLE_NAMES
    corrected["role_correction"] = {
        "derived_from": PARENT_TASK_ID,
        "method": "seven disjoint fixed position intervals, checked against source row IDs and labels",
        "historical_report_rewritten": False,
    }
    expansion._validate_role_partition(corrected)
    _require(sum(corrected["role_counts"].values()) == 3_300, "corrected role partition does not cover 3,300 rows")
    return corrected


def _validate_canonical_rows(population: Mapping[str, Any], labels: torch.Tensor, train_indices: torch.Tensor) -> None:
    rows = torch.tensor(population["source_row_indices"], dtype=torch.long)
    expected_labels = torch.tensor(population["labels"], dtype=torch.long)
    _require(rows.shape == (3_300,) and expected_labels.shape == (3_300,), "fixed population row/label shape differs")
    _require(torch.equal(labels.index_select(0, rows), expected_labels), "fixed population labels differ from canonical MNIST metadata")
    train_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    _require(bool(train_mask.index_select(0, rows).all()), "one or more population rows are outside canonical TRAIN")
    _require(torch.bincount(expected_labels[CONFIRMATION_START:CONFIRMATION_STOP], minlength=10).tolist() == [30] * 10, "final confirmation role is not balanced 30 per class")
    _require(torch.bincount(expected_labels[:FIT_ROWS], minlength=10).tolist() == [FIT_ROWS_PER_CLASS] * 10, "original fit role is not balanced 20 per class")


def _load_population(root: Path, parent_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    population_path, report_path = parent_dir / "input_population.json", parent_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "explicit population-expansion source files are missing")
    source_population = json.loads(population_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(source_population.get("task_id") == PARENT_TASK_ID and source_report.get("task_id") == PARENT_TASK_ID, "source task identity differs")
    _require(source_report.get("status") == "VALID_FAILED" and source_report.get("classification") == "MIXED_OR_INSUFFICIENT", "source scientific result differs from the named expansion comparison")
    corrected = _corrected_population(source_population)
    source_paths = source_population.get("source_input_paths")
    _require(isinstance(source_paths, Mapping), "source population omits explicit input paths")
    paths = {str(name): _resolve(root, value) for name, value in source_paths.items()}
    required = {"training_data", "e_s", "e_i", "e_c", "e_d"}
    _require(required.issubset(paths) and all(path.is_file() for path in paths.values()), "one or more explicit source data/evaluator paths are unavailable")
    checkpoint_path = parent_dir / "arm_a_step4700.pt"
    _require(checkpoint_path.is_file(), "A4700 source checkpoint is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_source_checkpoint(checkpoint, corrected, checkpoint_path)
    readout_paths = {name: parent_dir / f"readouts_{name}.pt" for name in ("A3700", "A4700", "B4700")}
    _require(all(path.is_file() for path in readout_paths.values()), "one or more 3,300-row endpoint readouts are missing")
    _verify_endpoint_lineage(readout_paths, corrected, source_report)

    image_store = diagnostic._load_torch(paths["training_data"], mmap=True)
    image_rows, image_labels = diagnostic._read_image_labels(image_store)
    train_indices, _ = diagnostic.stratified_mnist_indices(image_labels, seed=diagnostic.SPLIT_SEED, validation_per_label=500)
    _validate_canonical_rows(corrected, image_labels, train_indices)
    rows = torch.tensor(corrected["source_row_indices"], dtype=torch.long)
    images = image_rows.index_select(0, rows).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(images.shape == (3_300, 1, 28, 28) and bool(torch.isfinite(images).all()), "materialized canonical TRAIN images are invalid")
    _require(float(images.min()) >= 0.0 and float(images.max()) <= 1.0, "canonical TRAIN images are outside [0,1]")
    inputs = {"images": images, "source_indices": rows, "labels": torch.tensor(corrected["labels"], dtype=torch.long)}
    return corrected, inputs, paths, {"report": source_report, "checkpoint": checkpoint, "checkpoint_path": checkpoint_path, "readout_paths": readout_paths, "image_labels": image_labels, "train_indices": train_indices}


def _validate_source_checkpoint(checkpoint: Mapping[str, Any], population: Mapping[str, Any], checkpoint_path: Path) -> None:
    _require(checkpoint.get("task_id") == PARENT_TASK_ID and checkpoint.get("arm") == expansion.ARM_A, "source checkpoint is not the original-rule A arm")
    _require(checkpoint.get("schema") == expansion.CHECKPOINT_SCHEMA and checkpoint.get("completed_updates") == SOURCE_STEP and checkpoint.get("optimizer_step") == SOURCE_STEP, "A4700 checkpoint or Generator Adam step differs")
    _require(checkpoint.get("task_local_updates") == 1_000 and checkpoint.get("source_history_count") == 3_700, "A4700 source/additional update counts differ")
    _require(checkpoint.get("next_batch_index") == 0 and checkpoint.get("next_batch_cursor") == 0, "A4700 endpoint is not at the declared batch boundary")
    _require(checkpoint.get("optimizer_parameter_names") == ["generator.phase"], "A4700 optimizer scope is not Generator-only")
    _require(checkpoint.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "A4700 Generator layout differs")
    parameters = checkpoint.get("trainable_parameter_state")
    _require(isinstance(parameters, Mapping), "A4700 trainable parameter state is missing")
    _require(tuple(parameters["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(parameters["generator.phase"].shape) == (340_000,), "A4700 phase parameter shapes differ")
    _require(all(bool(torch.isfinite(value).all()) for value in parameters.values()), "A4700 phase parameter state is nonfinite")
    frozen = checkpoint.get("frozen_runtime_state")
    _require(isinstance(frozen, Mapping) and set(frozen) == {"ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "A4700 frozen OLS state differs")
    _require(all(bool(torch.isfinite(value.real if value.is_complex() else value).all()) for value in frozen.values()), "A4700 frozen OLS state is nonfinite")
    optimizer = checkpoint.get("optimizer_state")
    _require(isinstance(optimizer, Mapping) and len(optimizer.get("param_groups", [])) == 1 and optimizer["param_groups"][0]["params"] == [0] and set(optimizer.get("state", {})) == {0}, "A4700 source Adam mapping differs")
    group, slot = optimizer["param_groups"][0], optimizer["state"][0]
    _require(group.get("lr") == capacity.GENERATOR_LEARNING_RATE and tuple(group.get("betas", ())) == capacity.GENERATOR_BETAS and group.get("eps") == capacity.GENERATOR_EPS and group.get("weight_decay") == 0.0, "A4700 Adam hyperparameters differ")
    _require(int(float(slot["step"].item())) == SOURCE_STEP and tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "A4700 Adam moments or step differ")
    _require(bool(torch.isfinite(slot["exp_avg"]).all()) and bool(torch.isfinite(slot["exp_avg_sq"]).all()), "A4700 Adam state is nonfinite")
    _require(checkpoint.get("numeric_mode") == expansion.NUMERIC_MODE, "A4700 numeric mode is not strict FP32")
    references = checkpoint.get("references")
    weights = checkpoint.get("weights")
    _require(isinstance(references, Mapping) and tuple(references["scale"].shape) == (100,) and tuple(references["power_reference"].shape) == (5,), "A4700 BOGT reference shapes differ")
    _require(references.get("data_role") == "train" and references.get("count") == 55_000 and references.get("engineering_only") is False, "A4700 BOGT reference role differs")
    _require(bool(torch.isfinite(references["scale"]).all() and torch.isfinite(references["power_reference"]).all() and (references["scale"] > 0).all() and (references["power_reference"] > 0).all()), "A4700 BOGT reference tensors are invalid")
    _require(isinstance(weights, Mapping) and (weights.get("alpha"), weights.get("beta"), weights.get("gamma")) == (0.01596223111734585, 2.4509953404250195, 0.5330424284576345), "A4700 BOGT loss weights differ")
    source_rows = checkpoint.get("population", {}).get("source_row_indices")
    source_labels = checkpoint.get("population", {}).get("labels")
    _require(source_rows == population["source_row_indices"] and source_labels == population["labels"], "checkpoint population rows or labels differ from the source manifest")
    _require(checkpoint.get("source_checkpoint", "").endswith("arm_a_step3700.pt"), "A4700 source checkpoint ancestry differs")
    _require(checkpoint_path.name == "arm_a_step4700.pt", "explicit source checkpoint path is not A4700")


def _verify_endpoint_lineage(readout_paths: Mapping[str, Path], population: Mapping[str, Any], source_report: Mapping[str, Any]) -> None:
    expected_rows = torch.tensor(population["source_row_indices"], dtype=torch.long)
    expected_labels = torch.tensor(population["labels"], dtype=torch.long)
    baseline = diagnostic._load_torch(readout_paths["A4700"], mmap=True)
    _require(torch.equal(baseline["source_indices"], expected_rows) and torch.equal(baseline["labels"], expected_labels), "A4700 row/label order differs from the fixed population")
    _require(tuple(baseline["z"].shape) == (3_300, 100) and baseline["z"].is_complex(), "A4700 saved OLS bank shape/type differs")
    _require(tuple(baseline["raw_cmos"].shape) == (3_300, 1, 200, 200) and tuple(baseline["evaluator_view"].shape) == (3_300, 1, 28, 28), "A4700 saved CMOS shapes differ")
    baseline_ols = baseline["z"]
    baseline_target = baseline["target_slm_intensity"]
    for name, path in readout_paths.items():
        current = baseline if name == "A4700" else diagnostic._load_torch(path, mmap=True)
        _require(torch.equal(current["source_indices"], expected_rows) and torch.equal(current["labels"], expected_labels), f"{name} readout row/label order differs")
        for start in range(0, 3_300, 100):
            stop = min(3_300, start + 100)
            _require(torch.allclose(current["z"][start:stop], baseline_ols[start:stop], rtol=RTOL, atol=ATOL), f"{name} saved OLS differs from A4700 at rows {start}:{stop}")
            _require(torch.allclose(current["target_slm_intensity"][start:stop], baseline_target[start:stop], rtol=RTOL, atol=ATOL), f"{name} target differs from A4700 at rows {start}:{stop}")
        _require(all(bool(torch.isfinite(current[f"{name_e.lower()}_logits"]).all()) for name_e in EVALUATOR_NAMES), f"{name} evaluator logits are nonfinite")
        if name != "A4700":
            del current
    metrics = source_report.get("new_confirmation_class_counts_by_id")
    _require(isinstance(metrics, Mapping), "source report omits its primary confirmation counts")
    expected_endpoint_names = {"A3700", "A4700", "B4700"}
    for endpoint in expected_endpoint_names:
        readout = baseline if endpoint == "A4700" else diagnostic._load_torch(readout_paths[endpoint], mmap=True)
        prediction = readout["e_d_logits"][CONFIRMATION_START:CONFIRMATION_STOP].to(torch.float64).argmax(1)
        labels = expected_labels[CONFIRMATION_START:CONFIRMATION_STOP]
        actual = [int(((labels == class_id) & (prediction == class_id)).sum()) for class_id in range(10)]
        expected = [int(metrics[str(class_id)][endpoint]) for class_id in range(10)]
        _require(actual == expected, f"{endpoint} confirmation counts do not reproduce the source report")
        del readout


def _history_exposure_counts(checkpoint: Mapping[str, Any], arm: str, population: Mapping[str, Any]) -> dict[str, Any]:
    history = checkpoint["training_history"]
    _require(len(history) == SOURCE_STEP and checkpoint.get("source_history_count") == 3_700, f"{arm} saved history length differs")
    batch_count = 20 if arm == "A_repeat_200" else 200
    mapping = population["fit_mapping_A_measure_positions"] if arm == "A_repeat_200" else population["fit_mapping_B_measure_positions"]
    expected_rows = population["source_row_indices"]
    expected_labels = population["labels"]
    exposures = [0] * len(mapping)
    for local_update, record in enumerate(history[3_700:], start=1):
        batch_index = (local_update - 1) % batch_count
        positions = list(range(batch_index * 10, (batch_index + 1) * 10))
        measure_positions = [mapping[position] for position in positions]
        _require(record.get("task_local_update") == local_update and record.get("update") == 3_700 + local_update and record.get("batch_index") == batch_index, f"{arm} training history cursor differs at local update {local_update}")
        _require(record.get("training_positions") == positions and record.get("fit_positions") == measure_positions, f"{arm} training mapping differs at local update {local_update}")
        _require(record.get("source_row_indices") == [expected_rows[position] for position in measure_positions] and record.get("labels") == [expected_labels[position] for position in measure_positions], f"{arm} source rows/labels differ at local update {local_update}")
        for position in positions:
            exposures[position] += 1
    expected_each = 50 if arm == "A_repeat_200" else 5
    _require(exposures == [expected_each] * len(mapping), f"{arm} row exposure count differs from the declared {expected_each}")
    return {"arm": arm, "updates": 1_000, "training_rows": len(mapping), "row_exposures_each": expected_each, "rows_verified": len(mapping), "passed": True}


def _classification_summary(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    y = labels.to(torch.long).cpu()
    values = logits.to(torch.float64).cpu()
    _require(values.shape == (y.numel(), 10) and bool(torch.isfinite(values).all()), "evaluator scoring input is invalid")
    prediction = values.argmax(1)
    counts = torch.bincount(y, minlength=10)
    correct = torch.tensor([int(((y == class_id) & (prediction == class_id)).sum()) for class_id in range(10)])
    confusion = torch.bincount(y * 10 + prediction, minlength=100).reshape(10, 10)
    true = values.gather(1, y[:, None]).squeeze(1)
    wrong = values.scatter(1, y[:, None], -torch.inf).max(1).values
    return {
        "rows": int(y.numel()),
        "per_class_count": counts.tolist(),
        "per_class_correct": correct.tolist(),
        "per_class_recall": (correct.to(torch.float64) / counts.clamp_min(1)).tolist(),
        "per_class_wilson_95": {str(c): _wilson(int(correct[c]), int(counts[c])) for c in range(10)},
        "macro_recall": float((correct.to(torch.float64) / counts.clamp_min(1)).mean()),
        "correct_total": int(correct.sum()),
        "prediction_counts": torch.bincount(prediction, minlength=10).tolist(),
        "confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
        "mean_ce_float64": float((torch.logsumexp(values, dim=1) - true).mean()),
        "mean_margin_float64": float((true - wrong).mean()),
    }


def _fit_diagonal_gaussian(fit_z: torch.Tensor, fit_labels: torch.Tensor, *, seed: int = SEED) -> dict[str, Any]:
    _require(fit_z.shape == (FIT_ROWS, OLS_DIM) and fit_z.is_complex(), "Gaussian fit requires the fixed 200-by-100 complex OLS population")
    _require(fit_labels.shape == (FIT_ROWS,) and torch.bincount(fit_labels.to(torch.long), minlength=10).tolist() == [20] * 10, "Gaussian fit labels are not 20 per class")
    values = fit_z.detach().cpu().to(torch.complex128)
    labels = fit_labels.detach().cpu().to(torch.long)
    means: list[torch.Tensor] = []
    real_std: list[torch.Tensor] = []
    imag_std: list[torch.Tensor] = []
    for class_id in range(10):
        group = values[labels == class_id]
        _require(group.shape == (20, OLS_DIM), f"class {class_id} Gaussian group shape differs")
        means.append(group.mean(0))
        real_std.append(group.real.std(0, unbiased=False))
        imag_std.append(group.imag.std(0, unbiased=False))
    mu = torch.stack(means)
    raw_real_std, raw_imag_std = torch.stack(real_std), torch.stack(imag_std)
    sigma_real, sigma_imag = raw_real_std.clamp_min(STD_FLOOR), raw_imag_std.clamp_min(STD_FLOOR)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (10, SAMPLES_PER_CLASS, OLS_DIM)
    epsilon_real = torch.randn(shape, generator=generator, dtype=torch.float64, device="cpu")
    epsilon_imag = torch.randn(shape, generator=generator, dtype=torch.float64, device="cpu")
    sampled_real = mu.real[:, None, :] + sigma_real[:, None, :] * epsilon_real
    sampled_imag = mu.imag[:, None, :] + sigma_imag[:, None, :] * epsilon_imag
    bank = torch.complex(sampled_real, sampled_imag).to(torch.complex64).reshape(SAMPLE_COUNT, OLS_DIM).contiguous()
    _require(bool(torch.isfinite(bank.real).all() and torch.isfinite(bank.imag).all()), "conditional OLS bank contains nonfinite values")
    return {
        "schema": "m0_generator_conditional_diagonal_complex_ols_bank.v1",
        "seed": seed,
        "fit_source_positions": list(range(FIT_ROWS)),
        "fit_rows_per_class": FIT_ROWS_PER_CLASS,
        "classes": list(range(10)),
        "samples_per_class": SAMPLES_PER_CLASS,
        "dimension": OLS_DIM,
        "variance_reduction": "population_std_unbiased_false",
        "std_floor": STD_FLOOR,
        "real_std_before_floor": raw_real_std,
        "imag_std_before_floor": raw_imag_std,
        "real_std": sigma_real,
        "imag_std": sigma_imag,
        "mean": mu,
        "real_floor_dimension_count": int((raw_real_std < STD_FLOOR).sum()),
        "imag_floor_dimension_count": int((raw_imag_std < STD_FLOOR).sum()),
        "noise_draw_order": ["all_real_epsilon_[10,200,100]", "all_imag_epsilon_[10,200,100]"],
        "ols": bank,
        "requested_class": torch.arange(10, dtype=torch.long).repeat_interleave(SAMPLES_PER_CLASS),
        "sample_index_within_class": torch.arange(SAMPLES_PER_CLASS, dtype=torch.long).repeat(10),
    }


def _verify_bank_against_independent_reference(fit_z: torch.Tensor, fit_labels: torch.Tensor, bank: Mapping[str, Any]) -> None:
    values = fit_z.detach().cpu().to(torch.complex128)
    labels = fit_labels.detach().cpu().to(torch.long)
    ref_mean = torch.stack([values[labels == class_id].mean(0) for class_id in range(10)])
    ref_real_std = torch.stack([values[labels == class_id].real.var(0, unbiased=False).sqrt() for class_id in range(10)]).clamp_min(STD_FLOOR)
    ref_imag_std = torch.stack([values[labels == class_id].imag.var(0, unbiased=False).sqrt() for class_id in range(10)]).clamp_min(STD_FLOOR)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    real_noise = torch.randn((10, 200, 100), generator=generator, dtype=torch.float64)
    imag_noise = torch.randn((10, 200, 100), generator=generator, dtype=torch.float64)
    ref_samples = torch.complex(ref_mean.real[:, None, :] + ref_real_std[:, None, :] * real_noise, ref_mean.imag[:, None, :] + ref_imag_std[:, None, :] * imag_noise).to(torch.complex64).reshape(2_000, 100)
    _require(torch.equal(bank["mean"], ref_mean), "conditional OLS means differ from independent CPU reference")
    _require(torch.allclose(bank["real_std"], ref_real_std, rtol=1.0e-14, atol=1.0e-16) and torch.allclose(bank["imag_std"], ref_imag_std, rtol=1.0e-14, atol=1.0e-16), "conditional OLS scales differ from independent CPU reference")
    _require(torch.allclose(bank["ols"], ref_samples, rtol=1.0e-6, atol=1.0e-7), "conditional OLS bank differs from independent fixed-seed draw reference")
    _require(torch.equal(bank["requested_class"], torch.arange(10).repeat_interleave(200)), "conditional OLS request order differs")
    _require(torch.equal(bank["sample_index_within_class"], torch.arange(200).repeat(10)), "conditional OLS within-class sample order differs")
    _require(bank["noise_draw_order"] == ["all_real_epsilon_[10,200,100]", "all_imag_epsilon_[10,200,100]"], "conditional OLS draw order differs")


def _validate_bank_provenance(fit_z: torch.Tensor, fit_labels: torch.Tensor, bank: Mapping[str, Any]) -> None:
    values = fit_z.detach().cpu().to(torch.complex128)
    labels = fit_labels.detach().cpu().to(torch.long)
    means = torch.stack([values[labels == class_id].mean(0) for class_id in range(10)])
    real_std = torch.stack([values[labels == class_id].real.std(0, unbiased=False) for class_id in range(10)])
    imag_std = torch.stack([values[labels == class_id].imag.std(0, unbiased=False) for class_id in range(10)])
    _require(torch.equal(bank["mean"], means), "saved bank mean differs from the fixed fit population")
    _require(torch.equal(bank["real_std"], real_std.clamp_min(STD_FLOOR)) and torch.equal(bank["imag_std"], imag_std.clamp_min(STD_FLOOR)), "saved bank standard deviation differs from the fixed fit population")
    _require(torch.equal(bank["requested_class"], torch.arange(10).repeat_interleave(200)) and torch.equal(bank["sample_index_within_class"], torch.arange(200).repeat(10)), "saved bank request labels or row order differs")
    _require(bank["ols"].shape == (SAMPLE_COUNT, OLS_DIM) and bank["ols"].is_complex() and bool(torch.isfinite(bank["ols"].real).all() and torch.isfinite(bank["ols"].imag).all()), "saved bank OLS values are invalid")


def _make_output_rows(fit_z: torch.Tensor, fit_labels: torch.Tensor, fit_source_rows: torch.Tensor, bank: Mapping[str, Any]) -> dict[str, Any]:
    class_means = bank["mean"].to(torch.complex64)
    z = torch.cat((fit_z.to(torch.complex64), class_means, bank["ols"]), dim=0).contiguous()
    labels = torch.cat((fit_labels.to(torch.long), torch.arange(10, dtype=torch.long), bank["requested_class"])).contiguous()
    kinds = ["empirical_fit_ols"] * FIT_ROWS + ["class_mean_ols"] * MEAN_COUNT + ["diagonal_gaussian_sample"] * SAMPLE_COUNT
    source_rows = torch.cat((fit_source_rows.to(torch.long), torch.full((MEAN_COUNT + SAMPLE_COUNT,), -1, dtype=torch.long)))
    source_positions = torch.cat((torch.arange(FIT_ROWS), torch.full((MEAN_COUNT + SAMPLE_COUNT,), -1, dtype=torch.long)))
    sample_indices = torch.cat((torch.full((FIT_ROWS + MEAN_COUNT,), -1, dtype=torch.long), bank["sample_index_within_class"]))
    _require(z.shape == (TOTAL_OUTPUTS, OLS_DIM) and labels.shape == (TOTAL_OUTPUTS,), "fixed 2,210-row OLS measurement bank shape differs")
    return {"z": z, "requested_class": labels, "kind": kinds, "source_row_indices": source_rows, "source_positions": source_positions, "sample_index_within_class": sample_indices}


def _new_runtime(checkpoint: Mapping[str, Any], device: torch.device) -> LightGen2DSemanticGenerationInputOLSAlignment:
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(LightGen2DAlignmentRuntimeConfig(propagation_grid_size=2_048))
    old_phase = torch.zeros((2, 1, 100, 100), dtype=torch.float32)
    runtime.generator = capacity.ExpandedGenerator(old_phase, propagation_grid_size=2_048)
    state = {**checkpoint["frozen_runtime_state"], **checkpoint["trainable_parameter_state"]}
    result = runtime.load_state_dict(state, strict=True)
    _require(not result.missing_keys and not result.unexpected_keys, "A4700 checkpoint did not fully load into the production runtime")
    runtime.to(device)
    runtime.eval()
    runtime.requires_grad_(False)
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters()), "frozen A4700 runtime still has gradient-enabled parameters")
    return runtime


def _snapshot_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _verify_measurement_freeze(runtime: nn.Module, runtime_before: Mapping[str, torch.Tensor], evaluators: Mapping[str, nn.Module], evaluator_before: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    runtime_after = _snapshot_state(runtime)
    runtime_unchanged = _tree_equal(runtime_before, runtime_after)
    evaluator_unchanged = all(_tree_equal(evaluator_before[name], capacity._state_snapshot(model)) for name, model in evaluators.items())
    gradients_empty = all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters())
    gradients_empty = gradients_empty and all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters())
    return {"executed": True, "passed": runtime_unchanged and evaluator_unchanged and gradients_empty, "runtime_parameters_and_buffers_unchanged": runtime_unchanged, "evaluator_parameters_and_buffers_unchanged": evaluator_unchanged, "all_gradients_none_and_parameters_frozen": gradients_empty, "optimizer_updates": 0, "gradient_calls": 0}


def _evaluate(model: nn.Module, view: torch.Tensor, name: str) -> torch.Tensor:
    return capacity._evaluator_logits(model, view, name)


def _resource_and_disk(target: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized production-path preflight")
    free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    disk_root = target
    while not disk_root.exists():
        disk_root = disk_root.parent
    disk = shutil.disk_usage(disk_root)
    estimated_readout_bytes = TOTAL_OUTPUTS * (200 * 200 * 4 + 28 * 28 * 4 + 100 * 8 + 4 * 10 * 4 + 5 * 8)
    minimum_free = int(estimated_readout_bytes + 1_500_000_000)
    _require(int(disk.free) >= minimum_free, f"insufficient disk space for estimated outputs plus 1.5 GB reserve: need {minimum_free}, have {disk.free}")
    return {
        "executed": True,
        "passed": True,
        "cuda_device": str(device),
        "cuda_total_bytes": int(total_gpu),
        "cuda_free_bytes": int(free_gpu),
        "pytorch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "pytorch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_memory_policy": "observed_only_no_utilization_quota",
        "disk_path": disk_root.as_posix(),
        "disk_free_bytes": int(disk.free),
        "estimated_remaining_output_bytes": estimated_readout_bytes,
        "required_post_output_reserve_bytes": 1_500_000_000,
        "passed_storage_estimate": True,
    }


def _readout_payload(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    evaluators: Mapping[str, nn.Module],
    rows: Mapping[str, Any],
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> dict[str, Any]:
    count = stop - start
    _require(count > 0 and INFERENCE_MICROBATCH_ROWS == 1, "frozen sampler requires the preflighted single-row optical microbatch")
    raw_parts: list[torch.Tensor] = []
    view_parts: list[torch.Tensor] = []
    input_power_parts: list[torch.Tensor] = []
    record_power_parts: list[torch.Tensor] = []
    extra_power_parts: list[torch.Tensor] = []
    evaluator_parts: dict[str, list[torch.Tensor]] = {name: [] for name in evaluators}
    clipping_parts: list[torch.Tensor] = []
    zero_parts: list[torch.Tensor] = []
    constant_parts: list[torch.Tensor] = []
    field_metadata = None
    with torch.inference_mode():
        for position in range(start, stop):
            z = rows["z"][position : position + 1].to(device)
            result = runtime.forward_from_ols(z)
            raw = result.raw_intensity
            view = capacity.canonical.cmos_view(raw)
            raw_parts.append(raw.detach().cpu().clone())
            view_parts.append(view.detach().cpu().clone())
            input_power_parts.append(bogt.physical_plane_power(result.generator_input_field).detach().to(torch.float64).cpu())
            record_power_parts.append(torch.stack([bogt.physical_plane_power(field) for field in result.generator_records], dim=1).detach().to(torch.float64).cpu())
            extra_power_parts.append(runtime.generator.last_extra_plane_powers.detach().to(torch.float64).cpu().clone())
            clipping_parts.append((raw > 1.0).flatten(1).sum(1).detach().cpu())
            zero_parts.append((raw.flatten(1).abs().sum(1) == 0).detach().cpu())
            constant_parts.append((raw.flatten(1).std(1, unbiased=False) == 0).detach().cpu())
            for name, model in evaluators.items():
                evaluator_parts[name].append(_evaluate(model, view, name).detach().cpu().clone())
            if start == 0 and position == start:
                field_metadata = {
                "generator_input_field": capacity._plane_metadata(result.generator_input_field),
                "generator_records": [capacity._plane_metadata(field) for field in result.generator_records],
                "camera_plane": result.raw_camera.plane.to_dict(),
                "camera_provenance": result.raw_camera.provenance.to_dict(),
                "encoder_plane_powers": "unavailable_on_direct_OLS_decode; not synthesized",
                }
            del z, result, raw, view
    payload: dict[str, Any] = {
        "measurement_positions": torch.arange(start, stop, dtype=torch.long),
        "z": rows["z"][start:stop].detach().cpu().clone(),
        "requested_class": rows["requested_class"][start:stop].clone(),
        "kind": rows["kind"][start:stop],
        "source_row_indices": rows["source_row_indices"][start:stop].clone(),
        "source_positions": rows["source_positions"][start:stop].clone(),
        "sample_index_within_class": rows["sample_index_within_class"][start:stop].clone(),
        "raw_cmos": torch.cat(raw_parts, dim=0),
        "evaluator_view": torch.cat(view_parts, dim=0),
        "power_at_generator_input": torch.cat(input_power_parts, dim=0),
        "power_at_generator_records": torch.cat(record_power_parts, dim=0),
        "new_phase_plane_power": torch.cat(extra_power_parts, dim=0),
        "encoder_plane_powers_available": False,
        "maximum_error_pixels_raw_above_one": torch.cat(clipping_parts, dim=0),
        "raw_zero_image": torch.cat(zero_parts, dim=0),
        "raw_constant_image": torch.cat(constant_parts, dim=0),
    }
    for name, parts in evaluator_parts.items():
        payload[f"{name.lower()}_logits"] = torch.cat(parts, dim=0)
    if field_metadata is not None:
        payload["field_metadata"] = field_metadata
    for key, value in payload.items():
        if torch.is_tensor(value):
            _require(bool(torch.isfinite(value.real if value.is_complex() else value).all()), f"measurement chunk contains nonfinite {key}")
    return payload


def _validate_chunk(payload: Mapping[str, Any], rows: Mapping[str, Any], start: int, stop: int) -> None:
    _require(torch.equal(payload.get("measurement_positions"), torch.arange(start, stop)), "saved measurement chunk positions differ")
    _require(torch.equal(payload.get("z"), rows["z"][start:stop]), "saved measurement chunk OLS differs from the fixed bank")
    _require(torch.equal(payload.get("requested_class"), rows["requested_class"][start:stop]), "saved measurement chunk request labels differ")
    for key in ("source_row_indices", "source_positions", "sample_index_within_class"):
        _require(torch.equal(payload.get(key), rows[key][start:stop]), f"saved measurement chunk mapping differs for {key}")
    expected_shape = (stop - start, 1, 200, 200)
    _require(tuple(payload["raw_cmos"].shape) == expected_shape and tuple(payload["evaluator_view"].shape) == (stop - start, 1, 28, 28), "saved measurement image shapes differ")
    _require(all(tuple(payload[f"{name.lower()}_logits"].shape) == (stop - start, 10) for name in EVALUATOR_NAMES), "saved evaluator logits shape differs")
    _require(all(bool(torch.isfinite(value.real if value.is_complex() else value).all()) for value in payload.values() if torch.is_tensor(value)), "saved measurement chunk contains nonfinite values")


def _target_positive_control(evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    images = inputs["images"][CONFIRMATION_START:CONFIRMATION_STOP]
    labels = inputs["labels"][CONFIRMATION_START:CONFIRMATION_STOP]
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATOR_NAMES}
    with torch.inference_mode():
        for start in range(0, images.shape[0], BATCH_SIZE):
            view = capacity.canonical.cmos_view(images[start : start + BATCH_SIZE].to(device))
            for name, model in evaluators.items():
                logits[name].append(_evaluate(model, view, name).detach().cpu())
    result = {}
    for name in EVALUATOR_NAMES:
        summary = _classification_summary(labels, torch.cat(logits[name]))
        result[name] = {**summary, "passed": summary["macro_recall"] >= 0.90 and all(summary["per_class_correct"][c] >= 24 for c in (1, 5, 7, 9))}
    return {"source_row_indices": inputs["source_indices"][CONFIRMATION_START:CONFIRMATION_STOP].clone(), "labels": labels.clone(), "summary": result, "logits": {name: torch.cat(values) for name, values in logits.items()}, "passed": all(value["passed"] for value in result.values())}


def _preflight(
    target: Path,
    checkpoint: Mapping[str, Any],
    source_readout: Mapping[str, Any],
    inputs: Mapping[str, Any],
    paths: Mapping[str, Path],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    resource = _resource_and_disk(target)
    runtime = _new_runtime(checkpoint, device)
    evaluators = diagnostic.load_evaluators(paths, device)
    runtime_before = _snapshot_state(runtime)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    optimizer_before = capacity._cpu_tree(checkpoint["optimizer_state"])
    target_control = _target_positive_control(evaluators, inputs, device)
    _write_torch(target / "preflight_target_controls.pt", target_control, "four-evaluator target positive controls on new TRAIN confirmation rows")
    checks: dict[str, Any] = {
        "cuda_and_disk": resource,
        "canonical_population_and_roles": {"executed": True, "passed": True, "rows": 3_300, "roles": ROLE_COUNTS},
        "a4700_checkpoint_state": {"executed": True, "passed": True, "source_step": SOURCE_STEP, "generator_parameters": 340_000, "adam_step": SOURCE_STEP, "optimizer_created": False},
        "four_evaluator_target_positive_control": {"executed": True, "passed": target_control["passed"], "summary": {name: {"macro_recall": value["macro_recall"], "per_class_correct": value["per_class_correct"], "passed": value["passed"]} for name, value in target_control["summary"].items()}},
    }
    _require(target_control["passed"], "one or more frozen evaluator target positive controls failed; no sampling measurement started")
    sample_positions = torch.arange(10, dtype=torch.long)
    sample_images = inputs["images"].index_select(0, sample_positions).to(device)
    saved_z = source_readout["z"][:10].to(device)
    with torch.inference_mode():
        image_path = runtime.forward_from_image(sample_images)
        cached_path = runtime.forward_from_ols(saved_z)
        _require(torch.allclose(image_path.z_ols.coefficients, saved_z, rtol=RTOL, atol=ATOL), "representative live Encoder OLS differs from A4700 saved OLS")
        _require(torch.allclose(image_path.raw_intensity, cached_path.raw_intensity, rtol=RTOL, atol=ATOL), "live image path and cached OLS decode differ")
        _require(torch.allclose(cached_path.raw_intensity.cpu(), source_readout["raw_cmos"][:10], rtol=RTOL, atol=ATOL), "cached OLS decode differs from A4700 saved raw CMOS")
        source_logits = {name: _evaluate(model, capacity.canonical.cmos_view(cached_path.raw_intensity), name).detach().cpu() for name, model in evaluators.items()}
    for name in EVALUATOR_NAMES:
        _require(torch.allclose(source_logits[name], source_readout[f"{name.lower()}_logits"][:10], rtol=RTOL, atol=ATOL), f"cached OLS evaluator replay differs from A4700 for {name}")
    checks["real_encoder_to_cached_ols_replay"] = {"executed": True, "passed": True, "rows": 10, "rtol": RTOL, "atol": ATOL}

    first_per_class = torch.arange(0, SAMPLE_COUNT, SAMPLES_PER_CLASS, dtype=torch.long)
    first_z = bank["ols"].index_select(0, first_per_class)
    first_labels = bank["requested_class"].index_select(0, first_per_class)
    with torch.inference_mode():
        first_result = runtime.forward_from_ols(first_z.to(device))
        first_view = capacity.canonical.cmos_view(first_result.raw_intensity)
        first_logits = {name: _evaluate(model, first_view, name).detach().cpu() for name, model in evaluators.items()}
        first_payload = {
            "z": first_z.clone(),
            "requested_class": first_labels.clone(),
            "raw_cmos": first_result.raw_intensity.detach().cpu().clone(),
            "evaluator_view": first_view.detach().cpu().clone(),
            **{f"{name.lower()}_logits": value for name, value in first_logits.items()},
        }
    _write_torch(target / "preflight_first_class_ols.pt", first_payload, "first sampled OLS from each class decoded through the full production path")
    reloaded_first = torch.load(target / "preflight_first_class_ols.pt", map_location="cpu", weights_only=True)
    _require(all(torch.equal(reloaded_first[key], first_payload[key]) for key in first_payload), "saved first-class OLS preflight output does not round-trip")
    checks["class_mean_and_first_sample_production_decode"] = {"executed": True, "passed": bool(torch.isfinite(first_payload["raw_cmos"]).all()), "rows": 10, "production_path": "forward_from_ols -> W_OLS -> expanded Generator -> raw CMOS"}
    _require(checks["class_mean_and_first_sample_production_decode"]["passed"], "first per-class sampled OLS forward is nonfinite")

    labels = first_labels.to(torch.long)
    correct_ce, correct_margin = _score_for_labels(first_logits["E_D"], labels)
    wrong_labels = (labels + 1) % 10
    wrong_ce, wrong_margin = _score_for_labels(first_logits["E_D"], wrong_labels)
    negative_control = bool(torch.all((correct_ce - wrong_ce).abs() > 1.0e-8) and torch.all((correct_margin - wrong_margin).abs() > 1.0e-8))
    checks["wrong_requested_label_scoring_control"] = {"executed": True, "passed": negative_control, "correct_ce_mean": float(correct_ce.mean()), "wrong_ce_mean": float(wrong_ce.mean()), "correct_margin_mean": float(correct_margin.mean()), "wrong_margin_mean": float(wrong_margin.mean())}
    _require(negative_control, "requested-label score control did not distinguish the wrong-label path")

    smoke = {"rows": torch.arange(3), "z": bank["ols"][:3].clone()}
    _write_torch(target / "preflight_io_roundtrip.pt", smoke, "frozen-bank IO and explicit recovery round-trip")
    loaded_smoke = torch.load(target / "preflight_io_roundtrip.pt", map_location="cpu", weights_only=True)
    _require(torch.equal(loaded_smoke["rows"], smoke["rows"]) and torch.equal(loaded_smoke["z"], smoke["z"]), "preflight IO recovery did not reproduce saved bank rows")
    checks["write_read_and_recovery_roundtrip"] = {"executed": True, "passed": True, "forward_count": 10, "gradient_count": 0, "optimizer_updates": 0}

    runtime_after = _snapshot_state(runtime)
    evaluator_after = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    frozen = _tree_equal(runtime_before, runtime_after) and all(_tree_equal(evaluator_before[name], evaluator_after[name]) for name in EVALUATOR_NAMES)
    frozen = frozen and all(parameter.grad is None and not parameter.requires_grad for parameter in runtime.parameters())
    frozen = frozen and all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters())
    frozen = frozen and _tree_equal(optimizer_before, checkpoint["optimizer_state"])
    checks["source_runtime_evaluators_and_adam_frozen"] = {"executed": True, "passed": frozen, "runtime_parameters_and_buffers_unchanged": _tree_equal(runtime_before, runtime_after), "evaluator_parameters_unchanged": all(_tree_equal(evaluator_before[name], evaluator_after[name]) for name in EVALUATOR_NAMES), "all_gradients_none": all(parameter.grad is None for parameter in runtime.parameters()) and all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()), "optimizer_created": False}
    _require(frozen, "a frozen runtime/evaluator/Adam state changed during preflight")
    _require(all(check.get("executed") and check.get("passed") for check in checks.values()), "preflight has a missing or failed check")
    return {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "status": "READY", "source_step": SOURCE_STEP, "scientific_updates": 0, "gradient_calls": 0, "optimizer_updates": 0, "formal_output_rows": TOTAL_OUTPUTS, "checks": checks, "duration_seconds_observation_only": None}


def _score_for_labels(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = logits.detach().to(torch.float64).cpu()
    y = labels.detach().to(torch.long).cpu()
    true = values.gather(1, y[:, None]).squeeze(1)
    wrong = values.scatter(1, y[:, None], -torch.inf).max(1).values
    return torch.logsumexp(values, dim=1) - true, true - wrong


def _make_control(population: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    source = population["source_checkpoint"]
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="frozen-A4700-class-conditional-diagonal-complex-OLS-generation",
        semantic_authority_id=PARENT_TASK_ID,
        data_role="canonical TRAIN: 200 original fit rows; 100 historic, 300x3 prior confirmation, 1,800 expanded TRAIN, and 300 primary confirmation used only for evidence; sampler fit only on original 200 rows",
        config_identity={"source_step": SOURCE_STEP, "grid": 2_048, "sampler": "per_class_diagonal_complex_gaussian", "fit_rows_per_class": FIT_ROWS_PER_CLASS, "samples_per_class": SAMPLES_PER_CLASS, "seed": SEED, "std_floor": STD_FLOOR},
        model_identity={"checkpoint": source, "parameter_shapes": population["source_parameter_shapes"], "architecture": population.get("architecture_layout", "two_original_100x100_then_two_400x400_phase_planes_flattened")},
        optimizer_identity={"created": False, "source_adam_step": SOURCE_STEP, "updates": 0},
        sampler_identity={"fit_positions": list(range(FIT_ROWS)), "confirmation_positions": list(range(CONFIRMATION_START, CONFIRMATION_STOP)), "samples_per_class": SAMPLES_PER_CLASS, "seed": SEED},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )
    control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:authorization:01")
    checks = preflight["checks"]
    _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY receipt requires each executed check to pass")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "source_replay": checks["real_encoder_to_cached_ols_replay"]["passed"], "target_controls": checks["four_evaluator_target_positive_control"]["passed"], "generation_forward": checks["class_mean_and_first_sample_production_decode"]["passed"], "wrong_label_control": checks["wrong_requested_label_scoring_control"]["passed"], "frozen_state": checks["source_runtime_evaluators_and_adam_frozen"]["passed"], "recovery_io": checks["write_read_and_recovery_roundtrip"]["passed"]},
        resource={"passed": checks["cuda_and_disk"]["passed"], "status": "PASS", "cuda_available": True, "disk": checks["cuda_and_disk"]},
        implementation={"passed": True, "status": "PASS", "selected_engine": "cuda_2048_frozen_A4700_conditional_complex_OLS_generation"},
        selected_engine="cuda_2048_frozen_A4700_conditional_complex_OLS_generation",
        resource_envelope={"disk": checks["cuda_and_disk"]},
    )
    return control, receipt


def _ready_receipt_matches(saved: PreflightReceipt, expected: PreflightReceipt) -> bool:
    return (
        saved.source_semantics_id == expected.source_semantics_id
        and saved.authorization_id == expected.authorization_id
        and saved.checks == expected.checks
        and saved.selected_engine == expected.selected_engine
        and saved.resource == expected.resource
        and saved.failure_signature is None
        and saved.recheck_count == 0
        and saved.status == "READY"
    )


def _validate_ready(target: Path, population: Mapping[str, Any], control: ResearchRunControl) -> PreflightReceipt:
    ready = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(ready.get("task_id") == TASK_ID and ready.get("status") == "READY", "preflight does not contain READY evidence for this task")
    _require(json.loads((target / "input_population.json").read_text(encoding="utf-8")) == dict(population), "population changed after READY")
    preflight = ready["preflight"]
    _require(preflight.get("schema") == PREFLIGHT_SCHEMA and preflight.get("scientific_updates") == 0 and preflight.get("gradient_calls") == 0 and preflight.get("optimizer_updates") == 0, "READY contains an incorrect update/gradient budget")
    _require(preflight.get("checks") and all(value.get("executed") is True and value.get("passed") is True for value in preflight["checks"].values()), "READY consumer found an unexecuted or failed check")
    expected_control, expected_receipt = _make_control(population, preflight)
    receipt = PreflightReceipt.from_mapping(ready.get("receipt", {}))
    _require(_ready_receipt_matches(receipt, expected_receipt), "READY receipt differs from this task's execution contract")
    control.mark_preflight_ready(receipt)
    return receipt


def _review_saved_evidence(target: Path, population: Mapping[str, Any], inputs: Mapping[str, Any], parent_report: Mapping[str, Any], readout_paths: Mapping[str, Path]) -> dict[str, Any]:
    role_stats: dict[str, Any] = {}
    missing_role_views: dict[str, dict[str, torch.Tensor]] = {}
    labels = inputs["labels"]
    for endpoint, path in readout_paths.items():
        readout = diagnostic._load_torch(path, mmap=True)
        _require(torch.equal(readout["source_indices"], inputs["source_indices"]) and torch.equal(readout["labels"], labels), f"{endpoint} saved readout row order differs during evidence review")
        role_stats[endpoint] = {}
        for name in EVALUATOR_NAMES:
            stats = class9._classification_statistics(labels, readout[f"{name.lower()}_logits"], ROLE_POSITIONS)
            role_stats[endpoint][name] = stats["by_role"]
        missing_role_views[endpoint] = {"evaluator_view": readout["evaluator_view"][900:1_200].clone(), "e_d_logits": readout["e_d_logits"][900:1_200].clone()}
        del readout
    primary = parent_report.get("new_confirmation_class_counts_by_id", {})
    reproduced: dict[str, list[int]] = {}
    for endpoint in ("A3700", "A4700", "B4700"):
        counts = role_stats[endpoint]["E_D"][ROLE_CONFIRMATION]["per_class_correct"]
        _require(counts == [int(primary[str(c)][endpoint]) for c in range(10)], f"{endpoint} corrected role counts differ from the original primary decision")
        reproduced[endpoint] = counts
    _require(parent_report.get("classification") == "MIXED_OR_INSUFFICIENT" and parent_report.get("status") == "VALID_FAILED", "parent scientific classification is not the declared historical result")
    pages = _render_missing_role_pages(target, population, inputs, missing_role_views)
    old_pages = parent_report.get("comparison_images", {})
    old_page_roles = old_pages.get("pages_by_role", {})
    _require(set(old_page_roles) == {"fit", "historic", "old_confirmation", "prior_new_confirmation", "expanded_training", "new_confirmation"}, "parent image page role set differs from the documented 3,000-row display")
    source_population = json.loads((readout_paths["A4700"].parent / "input_population.json").read_text(encoding="utf-8"))
    source_positions = source_population.get("role_positions", {})
    corrected_pages = {role: [] for role in ROLE_POSITIONS}
    corrected_display: dict[str, list[int]] = {role: [] for role in ROLE_POSITIONS}
    mapping = {"fit": "fit", "historic": "historic", "old_confirmation": "ce_confirmation", "prior_new_confirmation": "presentation_confirmation", "expanded_training": "expanded_training", "new_confirmation": "coverage_confirmation"}
    for old_role, corrected_role in mapping.items():
        start, stop = ROLE_POSITIONS[corrected_role]
        _require(source_positions.get(old_role) == [start, stop], f"historical {old_role} row interval differs from corrected {corrected_role}")
        old_files = list(old_page_roles[old_role])
        expected_names = [f"comparison_{old_role}_{index:03d}.png" for index in range(1, math.ceil((stop - start) / 30) + 1)]
        _require(old_files == expected_names, f"historical {old_role} page order/count differs from its ordered 30-row paging contract")
        for page_index, filename in enumerate(old_files):
            page_start = start + page_index * 30
            page_stop = min(page_start + 30, stop)
            with Image.open(readout_paths["A4700"].parent / filename) as page:
                _require(page.size == (820, 58 + (page_stop - page_start) * 150), f"historical {old_role} page dimensions do not match its row slice: {filename}")
        indices = [int(v) for v in inputs["source_indices"][start:stop].tolist()]
        expected = inputs["source_indices"][start:stop].tolist()
        _require(indices == expected, f"historical {old_role} pages do not map to corrected {corrected_role} row IDs")
        corrected_pages[corrected_role] = list(old_page_roles[old_role])
        corrected_display[corrected_role] = indices
    corrected_pages["class9_confirmation"] = pages["files"]
    corrected_display["class9_confirmation"] = pages["displayed_source_indices"]
    all_displayed = [row for role in ROLE_POSITIONS for row in corrected_display[role]]
    _require(len(all_displayed) == 3_300 and len(set(all_displayed)) == 3_300, "corrected page manifest does not cover each of the 3,300 rows exactly once")
    _require(all_displayed == inputs["source_indices"].tolist(), "corrected page manifest changed population order")
    page_manifest = {
        "schema": "m0_generator_population_comparison_pages_corrected.v1",
        "source_parent_report": (readout_paths["A4700"].parent / "report.json").as_posix(),
        "pages_by_role": corrected_pages,
        "role_counts": ROLE_COUNTS,
        "displayed_source_indices_by_role": corrected_display,
        "displayed_row_count": len(all_displayed),
        "historical_row_id_mapping_basis": "parent role_positions and source_row_indices plus each existing page's ordered 30-row filename sequence and dimensions; the original report did not store per-page row IDs",
        "fixed_display_range": [0, 1],
        "per_image_rescaling": False,
        "new_pages_written_for_missing_class9_confirmation_only": True,
        "historical_coverage_confirmation_page_caption": "parent pages keep the historical caption new_confirmation; corrected row mapping identifies positions 3000:3300",
    }
    _write_json(target / "corrected_comparison_pages.json", page_manifest, "corrected role-to-page map covering all 3,300 saved population rows")
    review = {
        "schema": "m0_generator_population_evidence_review.v1",
        "source_task_id": PARENT_TASK_ID,
        "source_status_preserved": parent_report["status"],
        "source_classification_preserved": parent_report["classification"],
        "scientific_updates_performed": 0,
        "image_role_accounting": {"old_displayed_rows": 3_000, "previously_omitted_rows": 300, "corrected_displayed_rows": 3_300, "roles": ROLE_COUNTS, "role_counts_sum": sum(ROLE_COUNTS.values()), "all_rows_exactly_once": True},
        "parent_primary_confirmation_reproduced": reproduced,
        "role_metrics_by_endpoint_and_evaluator": role_stats,
        "training_mapping_exposure_checks": _verify_training_histories(readout_paths["A4700"].parent, population),
        "corrected_page_manifest": "corrected_comparison_pages.json",
        "parent_report_modified": False,
        "historical_image_files_modified": False,
        "class_index_correction": "class 1 is read from index 1; values at index 0 are class 0",
    }
    _write_json(target / "population_evidence_review.json", review, "CPU-only correction of seven roles and full 3,300-row evidence accounting")
    return review


def _verify_training_histories(parent_dir: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    checks = {}
    for arm, filename in (("A_repeat_200", "arm_a_step4700.pt"), ("B_expand_2000", "arm_b_step4700.pt")):
        checkpoint = torch.load(parent_dir / filename, map_location="cpu", weights_only=True)
        _require(checkpoint.get("arm") == arm and checkpoint.get("completed_updates") == SOURCE_STEP, f"{arm} endpoint or update count differs")
        checks[arm] = _history_exposure_counts(checkpoint, arm, population)
        del checkpoint
    return checks


def _display_image(view: torch.Tensor, *, scale: int = 4) -> Image.Image:
    value = view.detach().to(torch.float32).cpu().squeeze().clamp(0, 1)
    pixels = torch.round((1 - value) * 255).to(torch.uint8).numpy()
    return Image.fromarray(pixels, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _render_missing_role_pages(target: Path, population: Mapping[str, Any], inputs: Mapping[str, Any], views: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    start, stop = ROLE_POSITIONS["class9_confirmation"]
    source_indices = inputs["source_indices"][start:stop].tolist()
    filenames: list[str] = []
    displayed: list[int] = []
    titles = ("Target", "A3700", "A4700", "B4700")
    for page_number, page_start in enumerate(range(start, stop, 30), start=1):
        page_stop = min(page_start + 30, stop)
        canvas = Image.new("RGB", (820, 58 + (page_stop - page_start) * 150), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"class9_confirmation; positions {page_start}-{page_stop}; fixed [0,1]", fill="black")
        x_positions = (350, 465, 580, 695)
        for x, title in zip(x_positions, titles, strict=True):
            draw.text((x, 35), title, fill="black")
        for page_row, position in enumerate(range(page_start, page_stop)):
            y = 55 + page_row * 150
            draw.text((8, y + 8), f"row={int(inputs['source_indices'][position])} y={int(inputs['labels'][position])} class9_confirmation", fill="black")
            panels = (inputs["images"][position], views["A3700"]["evaluator_view"][position - start], views["A4700"]["evaluator_view"][position - start], views["B4700"]["evaluator_view"][position - start])
            for column, panel in enumerate(panels):
                canvas.paste(_display_image(panel), (x_positions[column], y))
            for column, endpoint in enumerate(("A3700", "A4700", "B4700"), start=1):
                prediction = int(views[endpoint]["e_d_logits"][position - start].to(torch.float64).argmax())
                draw.text((x_positions[column], y + 116), f"E_D={prediction}", fill="black")
            displayed.append(int(inputs["source_indices"][position]))
        filename = f"corrected_class9_confirmation_{page_number:03d}.png"
        path = target / filename
        if path.exists():
            with Image.open(path) as old:
                _require(old.size == canvas.size and old.convert("RGB").tobytes() == canvas.tobytes(), f"corrected class9 page differs: {filename}")
        else:
            canvas.save(path, optimize=True)
        filenames.append(filename)
    _require(displayed == source_indices and len(displayed) == stop - start, "new class9 pages do not exactly cover missing rows")
    return {"files": filenames, "displayed_source_indices": displayed, "rows": len(displayed), "panel_order": ["target", "A3700", "A4700", "B4700"], "fixed_display_range": [0, 1], "per_image_rescaling": False}


def _prepare_target(target: Path, *, resume: bool) -> None:
    if resume:
        _require(target.is_dir(), "explicit resume output directory does not exist")
        return
    if target.exists():
        existing = {item.name for item in target.iterdir()}
        staged_class9_pages = {
            name
            for name in existing
            if name.startswith("corrected_class9_confirmation_") and name.endswith(".png")
        }
        _require(existing - staged_class9_pages <= KNOWN_OUTPUT_FILES and not any(name.startswith("readout_rows_") or name.startswith("generation_gallery_") for name in existing), "output directory contains unstaged or historical outputs; use a new attempt directory")
        _require("invalid_harness.json" not in existing, "an INVALID_HARNESS directory requires explicit recovery or a new attempt")
    else:
        target.mkdir(parents=True, exist_ok=False)


def _bank_payload(path: Path, fit_z: torch.Tensor, fit_labels: torch.Tensor) -> dict[str, Any]:
    if path.is_file():
        bank = torch.load(path, map_location="cpu", weights_only=True)
        _require(bank.get("schema") == "m0_generator_conditional_diagonal_complex_ols_bank.v1" and bank.get("seed") == SEED, "saved conditional OLS bank schema or seed differs")
        _validate_bank_provenance(fit_z, fit_labels, bank)
        return bank
    bank = _fit_diagonal_gaussian(fit_z, fit_labels)
    _verify_bank_against_independent_reference(fit_z, fit_labels, bank)
    _write_torch(path, bank, "fixed per-class diagonal complex-OLS sampling bank generated once from original 200 fit rows")
    return bank


def _make_manifest(target: Path, population: Mapping[str, Any], bank: Mapping[str, Any], *, status: str, completed_chunks: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "schema": "m0_generator_frozen_conditional_ols_generation_manifest.v1",
        "task_id": TASK_ID,
        "status": status,
        "source_checkpoint": population["source_checkpoint"],
        "source_step": SOURCE_STEP,
        "source_row_indices": population["source_row_indices"],
        "role_positions": ROLE_POSITIONS,
        "bank_artifact": "conditional_ols_bank.pt",
        "bank_rows": int(bank["ols"].shape[0]),
        "seed": SEED,
        "completed_chunks": list(completed_chunks),
        "chunk_rows": CHUNK_ROWS,
        "total_output_rows": TOTAL_OUTPUTS,
        "scientific_updates": 0,
        "gradient_calls": 0,
        "optimizer_updates": 0,
    }


def _write_or_validate_json(path: Path, value: Mapping[str, Any], kind: str) -> None:
    if path.exists():
        _require(json.loads(path.read_text(encoding="utf-8")) == dict(value), f"staged {path.name} differs")
    else:
        _write_json(path, value, kind)


def _write_or_validate_torch(path: Path, value: Mapping[str, Any], kind: str) -> None:
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        _require(_tree_equal(saved, dict(value)), f"staged {path.name} differs")
    else:
        _write_torch(path, value, kind)


def _sampling_parameters(bank: Mapping[str, Any]) -> dict[str, Any]:
    mean = bank["mean"].detach().to(torch.complex128).cpu()
    return {
        "mean_real": mean.real.tolist(),
        "mean_imag": mean.imag.tolist(),
        "real_std": bank["real_std"].detach().to(torch.float64).cpu().tolist(),
        "imag_std": bank["imag_std"].detach().to(torch.float64).cpu().tolist(),
        "real_floor_dimension_count": int(bank["real_floor_dimension_count"]),
        "imag_floor_dimension_count": int(bank["imag_floor_dimension_count"]),
        "draw_order": list(bank["noise_draw_order"]),
    }


def _write_manifest_progress(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{time.time_ns()}.pending")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _chunk_name(start: int, stop: int) -> str:
    return f"readout_rows_{start:04d}_{stop:04d}.pt"


def _read_manifest(path: Path, target: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.resolve() == (target / "manifest.json").resolve(), "resume requires the explicit manifest for this exact output directory")
    _require(path.is_file(), "explicit resume manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "m0_generator_frozen_conditional_ols_generation_manifest.v1" and manifest.get("task_id") == TASK_ID, "manifest belongs to another task or schema")
    _require(manifest.get("source_row_indices") == population["source_row_indices"] and manifest.get("role_positions") == ROLE_POSITIONS and manifest.get("source_checkpoint") == population["source_checkpoint"], "resume manifest input population or endpoint differs")
    _require(manifest.get("bank_artifact") == "conditional_ols_bank.pt" and manifest.get("bank_rows") == SAMPLE_COUNT and manifest.get("seed") == SEED, "resume manifest sampling bank differs")
    _require(manifest.get("scientific_updates") == 0 and manifest.get("gradient_calls") == 0 and manifest.get("optimizer_updates") == 0, "resume manifest update counts differ")
    return manifest


def _summary_and_rows(target: Path, outputs: Mapping[str, Any], bank: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATOR_NAMES}
    all_views: list[torch.Tensor] = []
    sample_logits: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATOR_NAMES}
    sample_raw_clipping: list[torch.Tensor] = []
    sample_raw_min: list[float] = []
    sample_raw_max: list[float] = []
    sample_zero: list[torch.Tensor] = []
    sample_constant: list[torch.Tensor] = []
    sample_views: list[torch.Tensor] = []
    for start in range(0, TOTAL_OUTPUTS, CHUNK_ROWS):
        stop = min(TOTAL_OUTPUTS, start + CHUNK_ROWS)
        payload = torch.load(target / _chunk_name(start, stop), map_location="cpu", weights_only=True, mmap=True)
        for name in EVALUATOR_NAMES:
            logits[name].append(payload[f"{name.lower()}_logits"])
        all_views.append(payload["evaluator_view"])
        lo, hi = max(start, FIT_ROWS + MEAN_COUNT), stop
        if hi > lo:
            local_start, local_stop = lo - start, hi - start
            sample_raw_clipping.append((payload["raw_cmos"][local_start:local_stop] > 1.0).flatten(1).to(torch.float64).mean(1))
            sample_raw_min.append(float(payload["raw_cmos"][local_start:local_stop].min()))
            sample_raw_max.append(float(payload["raw_cmos"][local_start:local_stop].max()))
            sample_zero.append(payload["raw_zero_image"][local_start:local_stop])
            sample_constant.append(payload["raw_constant_image"][local_start:local_stop])
            sample_views.append(payload["evaluator_view"][local_start:local_stop])
            for name in EVALUATOR_NAMES:
                sample_logits[name].append(payload[f"{name.lower()}_logits"][local_start:local_stop])
    requested = outputs["requested_class"]
    kind_ranges = {"empirical_fit_ols": (0, FIT_ROWS), "class_mean_ols": (FIT_ROWS, FIT_ROWS + MEAN_COUNT), "diagonal_gaussian_sample": (FIT_ROWS + MEAN_COUNT, TOTAL_OUTPUTS)}
    endpoint_stats: dict[str, Any] = {}
    row_metrics: dict[str, torch.Tensor] = {}
    for name in EVALUATOR_NAMES:
        combined = torch.cat(logits[name]).to(torch.float64)
        endpoint_stats[name] = {}
        for kind, (start, stop) in kind_ranges.items():
            indices = torch.arange(start, stop)
            summary = _classification_summary(requested[start:stop], combined[start:stop])
            endpoint_stats[name][kind] = summary
            true = combined[start:stop].gather(1, requested[start:stop, None]).squeeze(1)
            other = combined[start:stop].scatter(1, requested[start:stop, None], -torch.inf).max(1).values
            row_metrics[f"{name.lower()}_ce_{kind}"] = torch.logsumexp(combined[start:stop], dim=1) - true
            row_metrics[f"{name.lower()}_margin_{kind}"] = true - other
            row_metrics[f"{name.lower()}_predictions_{kind}"] = combined[start:stop].argmax(1)
            row_metrics[f"{name.lower()}_positions_{kind}"] = indices
    random_stats = {name: endpoint_stats[name]["diagonal_gaussian_sample"] for name in EVALUATOR_NAMES}
    ed = random_stats["E_D"]
    random_start = FIT_ROWS + MEAN_COUNT
    clipping = torch.cat(sample_raw_clipping)
    zero = torch.cat(sample_zero)
    constant = torch.cat(sample_constant)
    views = torch.cat(sample_views).to(torch.float32)
    summary = {
        "rows": TOTAL_OUTPUTS,
        "source_groups": {"empirical_fit_ols": FIT_ROWS, "class_mean_ols": MEAN_COUNT, "diagonal_gaussian_sample": SAMPLE_COUNT},
        "evaluators": endpoint_stats,
        "primary_e_d_sample_screen": {
            "correct_total": ed["correct_total"],
            "macro_recall": ed["macro_recall"],
            "per_class_correct": ed["per_class_correct"],
            "per_class_recall": ed["per_class_recall"],
            "per_class_wilson_95": ed["per_class_wilson_95"],
            "prediction_counts": ed["prediction_counts"],
            "confusion_matrix_actual_rows_predicted_columns": ed["confusion_matrix_actual_rows_predicted_columns"],
            "raw_clipping_fraction": float(clipping.mean()),
            "maximum_per_image_raw_clipping_fraction": float(clipping.max()),
            "zero_image_count": int(zero.sum()),
            "constant_image_count": int(constant.sum()),
            "all_finite": True,
            "passed": ed["macro_recall"] >= 0.50 and min(ed["per_class_recall"]) >= 0.10 and float(clipping.mean()) <= 0.01 and not bool(zero.any()) and not bool(constant.any()),
        },
        "class9_weight_floor_dimensions": {"real": int(bank["real_floor_dimension_count"]), "imag": int(bank["imag_floor_dimension_count"]), "floor": STD_FLOOR},
        "paired_reconstruction_mse": "not_applicable_generated samples have no paired target",
        "encoder_plane_power": "unavailable_on_direct_OLS_decode",
        "raw_sample_min": min(sample_raw_min),
        "raw_sample_max": max(sample_raw_max),
    }
    row_metrics.update({
        "sample_raw_clipping_fraction": clipping,
        "sample_raw_zero_image": zero,
        "sample_raw_constant_image": constant,
        "sample_evaluator_view": views,
        "sample_requested_class": requested[random_start:],
        "sample_index_within_class": bank["sample_index_within_class"],
    })
    return summary, row_metrics


def _decision(summary: Mapping[str, Any], *, valid: bool) -> dict[str, Any]:
    metrics = summary.get("primary_e_d_sample_screen", {})
    if not valid:
        status, classification = "INVALID_HARNESS", None
    elif metrics.get("passed"):
        status, classification = "VALID_COMPLETED", "CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED"
    else:
        status, classification = "VALID_FAILED", "CONDITIONAL_OLS_GENERATION_NOT_SUPPORTED_WITHIN_SCOPE"
    return {"status": status, "classification": classification, "conditions": {
        "ED_macro_recall_at_least_50_percent": metrics.get("macro_recall", 0.0) >= 0.50,
        "each_class_recall_at_least_10_percent": min(metrics.get("per_class_recall", [0.0])) >= 0.10,
        "raw_clipping_at_most_1_percent": metrics.get("raw_clipping_fraction", 1.0) <= 0.01,
        "finite_nonzero_nonconstant_outputs": metrics.get("all_finite") is True and metrics.get("zero_image_count", 1) == 0 and metrics.get("constant_image_count", 1) == 0,
        "complete_measurement_and_frozen_scope": valid,
    }, "training_recommendation": "NONE", "next_action": "NONE", "stop": True}


def _sample_predictions(target: Path) -> torch.Tensor:
    first_sample_position = FIT_ROWS + MEAN_COUNT
    predictions: list[torch.Tensor] = []
    for start in range(0, TOTAL_OUTPUTS, CHUNK_ROWS):
        stop = min(TOTAL_OUTPUTS, start + CHUNK_ROWS)
        payload = torch.load(target / _chunk_name(start, stop), map_location="cpu", weights_only=True, mmap=True)
        local_start = max(0, first_sample_position - start)
        if local_start < stop - start:
            predictions.append(payload["e_d_logits"][local_start:].to(torch.float64).argmax(1))
    result = torch.cat(predictions)
    _require(result.shape == (SAMPLE_COUNT,), "sample prediction read does not cover exactly 2,000 generated rows")
    return result


def _gallery(target: Path, outputs: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    paths: list[str] = []
    prediction = _sample_predictions(target)
    for page_index, page_start in enumerate(range(0, SAMPLE_COUNT, 100), start=1):
        canvas = Image.new("RGB", (10 * 96, 10 * 112), "white")
        draw = ImageDraw.Draw(canvas)
        page_cache: dict[int, Mapping[str, Any]] = {}
        for local in range(100):
            sample = page_start + local
            kind_index = FIT_ROWS + MEAN_COUNT + sample
            chunk_start = (kind_index // CHUNK_ROWS) * CHUNK_ROWS
            chunk_stop = min(TOTAL_OUTPUTS, chunk_start + CHUNK_ROWS)
            payload = page_cache.get(chunk_start)
            if payload is None:
                payload = torch.load(target / _chunk_name(chunk_start, chunk_stop), map_location="cpu", weights_only=True, mmap=True)
                page_cache[chunk_start] = payload
            panel = payload["evaluator_view"][kind_index - chunk_start]
            x, y = (local % 10) * 96, (local // 10) * 112
            canvas.paste(_display_image(panel, scale=2), (x, y))
            draw.text((x, y + 58), f"req={sample // SAMPLES_PER_CLASS} pred={int(prediction[sample])}", fill="black")
            draw.text((x, y + 74), f"n={sample % SAMPLES_PER_CLASS}", fill="black")
        filename = f"generation_gallery_{page_index:02d}.png"
        path = target / filename
        if path.exists():
            with Image.open(path) as old:
                _require(old.size == canvas.size and old.convert("RGB").tobytes() == canvas.tobytes(), f"generated gallery differs: {filename}")
        else:
            canvas.save(path, optimize=True)
        paths.append(filename)
    return {"pages": paths, "samples_displayed": SAMPLE_COUNT, "fixed_display_range": [0, 1], "per_image_rescaling": False, "e_d_prediction_annotated": True}


def _descriptive_diversity(target: Path, inputs: Mapping[str, Any], rows: Mapping[str, Any]) -> dict[str, Any]:
    sample_views: list[torch.Tensor] = []
    first_sample_position = FIT_ROWS + MEAN_COUNT
    for start in range(0, TOTAL_OUTPUTS, CHUNK_ROWS):
        stop = min(TOTAL_OUTPUTS, start + CHUNK_ROWS)
        payload = torch.load(target / _chunk_name(start, stop), map_location="cpu", weights_only=True, mmap=True)
        local_start, local_stop = max(0, first_sample_position - start), stop - start
        if local_start < local_stop:
            sample_views.append(payload["evaluator_view"][local_start:local_stop].flatten(1).to(torch.float32))
    samples = torch.cat(sample_views)
    _require(samples.shape == (SAMPLE_COUNT, 784), "saved sample views are incomplete for diversity summary")
    quantized = torch.round(samples.clamp(0, 1) * 255).to(torch.uint8)
    signatures = [bytes(row.tolist()) for row in quantized]
    unique_count = len(set(signatures))
    targets = inputs["images"].flatten(1).to(torch.float32)
    sample_sq = samples.square().sum(1)
    target_sq = targets.square().sum(1)
    nearest_distance = torch.full((SAMPLE_COUNT,), float("inf"), dtype=torch.float32)
    nearest_target = torch.full((SAMPLE_COUNT,), -1, dtype=torch.long)
    for start in range(0, SAMPLE_COUNT, 64):
        stop = min(SAMPLE_COUNT, start + 64)
        squared = sample_sq[start:stop, None] + target_sq[None, :] - 2.0 * (samples[start:stop] @ targets.T)
        squared.clamp_min_(0.0)
        values, indices = squared.min(1)
        nearest_distance[start:stop] = values.sqrt() / math.sqrt(784.0)
        nearest_target[start:stop] = indices
    sample_distance = torch.full((SAMPLE_COUNT,), float("inf"), dtype=torch.float32)
    nearest_sample = torch.full((SAMPLE_COUNT,), -1, dtype=torch.long)
    for start in range(0, SAMPLE_COUNT, 64):
        stop = min(SAMPLE_COUNT, start + 64)
        squared = sample_sq[start:stop, None] + sample_sq[None, :] - 2.0 * (samples[start:stop] @ samples.T)
        squared.clamp_min_(0.0)
        local_indices = torch.arange(start, stop)
        squared[torch.arange(stop - start), local_indices] = torch.inf
        values, indices = squared.min(1)
        sample_distance[start:stop] = values.sqrt() / math.sqrt(784.0)
        nearest_sample[start:stop] = indices
    return {
        "scope": f"descriptive_only_{SAMPLE_COUNT}_generated_views_and_3300_canonical_TRAIN_targets",
        "unique_8bit_views": unique_count,
        "exact_8bit_duplicate_excess": SAMPLE_COUNT - unique_count,
        "mean_pixel_variance": float(samples.var(0, unbiased=False).mean()),
        "mean_nearest_generated_view_rmse": float(sample_distance.mean()),
        "nearest_generated_sample_index": nearest_sample.tolist(),
        "mean_nearest_saved_TRAIN_target_view_rmse": float(nearest_distance.mean()),
        "nearest_target_population_position": nearest_target.tolist(),
        "nearest_target_source_row": [int(inputs["source_indices"][index]) for index in nearest_target.tolist()],
        "novelty_or_nonmemorization_claim": False,
    }


def _build_report(target: Path, population: Mapping[str, Any], bank: Mapping[str, Any], summary: Mapping[str, Any], row_metrics: Mapping[str, torch.Tensor], replay: Mapping[str, Any], preflight: Mapping[str, Any], diversity: Mapping[str, Any], gallery: Mapping[str, Any], measurement_freeze: Mapping[str, Any], elapsed: float) -> dict[str, Any]:
    decision = _decision(summary, valid=True)
    incident_path = target / "invalid_harness.json"
    recovered_incidents = [json.loads(incident_path.read_text(encoding="utf-8"))] if incident_path.is_file() else []
    return {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": decision["status"],
        "classification": decision["classification"],
        "source_task": {"task_id": PARENT_TASK_ID, "status": "VALID_FAILED", "classification": "MIXED_OR_INSUFFICIENT", "selected_arm": "A4700 original training rule"},
        "scope": {"source_step": SOURCE_STEP, "architecture": "two_original_100x100_then_two_400x400_phase_planes_flattened", "generator_parameters": 340_000, "propagation_grid": 2_048, "model_updates": 0, "optimizer_updates": 0, "gradient_calls": 0, "fit_rows": FIT_ROWS, "fit_rows_per_class": FIT_ROWS_PER_CLASS, "confirmation_rows": CONFIRMATION_STOP - CONFIRMATION_START, "sampled_rows": SAMPLE_COUNT, "all_population_rows_canonical_train": True, "independent_prior_or_generalization_claim": False, "sampler": "per_class_diagonal_complex_OLS_Gaussian", "seed": SEED, "variance_unbiased": False, "standard_deviation_floor": STD_FLOOR},
        "population_role_correction": {"artifact": "population_evidence_review.json", "corrected_population": "corrected_input_population.json", "all_3300_rows_displayed_exactly_once": True, "parent_report_and_pages_rewritten": False},
        "sampling_bank_artifact": "conditional_ols_bank.pt",
        "sampling_parameters": _sampling_parameters(bank),
        "source_empirical_OLS_replay": replay,
        "formal_measurement_frozen_state": dict(measurement_freeze),
        "recovered_engineering_incidents": recovered_incidents,
        "results": summary,
        "row_metric_artifact": "sample_row_metrics.pt",
        "readout_chunks": sorted(path.name for path in target.glob("readout_rows_*.pt")),
        "conditional_generation_gallery": gallery,
        "descriptive_diversity_and_nearest_targets": diversity,
        "preflight_artifact": "preflight.json",
        "decision": decision,
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "stop": True,
        "elapsed_seconds_observation_only": elapsed,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-dir", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path, help="explicit task-local readout manifest required to recover")
    return parser.parse_args()


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    args = args or _arguments()
    root = _root()
    parent_dir = _resolve(root, args.population_dir)
    target = _resolve(root, args.output_dir)
    _require(not (args.preflight_only and args.resume_manifest is not None), "--preflight-only cannot be combined with --resume-manifest")
    resuming = args.resume_manifest is not None
    _prepare_target(target, resume=resuming)
    phase = "input_validation"
    progress: dict[str, Any] = {}
    try:
        population, inputs, paths, source = _load_population(root, parent_dir)
        population["source_checkpoint"] = source["checkpoint_path"].as_posix()
        population["source_step"] = SOURCE_STEP
        population["source_generator_adam_step"] = SOURCE_STEP
        population["architecture_layout"] = source["checkpoint"]["architecture_layout"]
        population["role_positions"] = ROLE_POSITIONS
        population["role_counts"] = ROLE_COUNTS
        population["roles"] = ROLE_NAMES
        population["scientific_rows"] = {"gaussian_fit": [0, FIT_ROWS], "primary_confirmation": [CONFIRMATION_START, CONFIRMATION_STOP]}
        source_checkpoint = source["checkpoint"]
        source_report = source["report"]
        _write_or_validate_json(target / "corrected_input_population.json", population, "corrected 3,300-row population with seven disjoint roles")
        _write_or_validate_json(target / "input_population.json", population, "fixed A4700 frozen generation population")
        if not (target / "population_evidence_review.json").exists():
            _review_saved_evidence(target, population, inputs, source_report, source["readout_paths"])
        else:
            review = json.loads((target / "population_evidence_review.json").read_text(encoding="utf-8"))
            _require(review.get("image_role_accounting", {}).get("all_rows_exactly_once") is True and review.get("scientific_updates_performed") == 0, "saved population evidence review is incomplete")
        progress["population_evidence_review"] = True
        if (target / "report.json").is_file():
            _require(resuming, "completed output requires an explicit resume manifest")
            manifest = _read_manifest(_resolve(root, args.resume_manifest), target, population)
            _require(manifest.get("status") == "COMPLETED", "terminal report has a nonterminal manifest")
            return json.loads((target / "report.json").read_text(encoding="utf-8"))

        source_readout = diagnostic._load_torch(source["readout_paths"]["A4700"], mmap=True)
        _require(torch.equal(source_readout["source_indices"], inputs["source_indices"]) and torch.equal(source_readout["labels"], inputs["labels"]), "A4700 source readout order differs")
        fit_z = source_readout["z"][:FIT_ROWS].clone()
        fit_labels = inputs["labels"][:FIT_ROWS].clone()
        bank_path = target / "conditional_ols_bank.pt"
        bank = _bank_payload(bank_path, fit_z, fit_labels)
        _require(bank["ols"].shape == (SAMPLE_COUNT, OLS_DIM) and torch.bincount(bank["requested_class"], minlength=10).tolist() == [SAMPLES_PER_CLASS] * 10, "saved conditional bank does not have 200 requested samples per class")
        outputs = _make_output_rows(fit_z, fit_labels, inputs["source_indices"][:FIT_ROWS], bank)
        del fit_z
        gc.collect()

        manifest_path = target / "manifest.json"
        if resuming:
            manifest = _read_manifest(_resolve(root, args.resume_manifest), target, population)
            _require((target / "conditional_ols_bank.pt").is_file(), "resume cannot redraw a missing OLS bank")
            _require(manifest.get("bank_artifact") == bank_path.name, "resume bank path differs")
        else:
            manifest = _make_manifest(target, population, bank, status="STAGED")
            _write_or_validate_json(manifest_path, manifest, "explicit source, fixed OLS bank, and readout chunk manifest")

        ready_path = target / "preflight.json"
        if ready_path.exists():
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            preflight = ready.get("preflight", {})
            _require(ready.get("status") == "READY" and preflight.get("schema") == PREFLIGHT_SCHEMA and all(check.get("executed") and check.get("passed") for check in preflight.get("checks", {}).values()), "staged preflight is not an executed READY")
            # Recreate the task-bound receipt and consumer from the stored executed results.
            control, expected_receipt = _make_control(population, preflight)
            saved_receipt = PreflightReceipt.from_mapping(ready.get("receipt", {}))
            _require(_ready_receipt_matches(saved_receipt, expected_receipt), "saved READY receipt differs from the current task identity")
            control.mark_preflight_ready(saved_receipt)
        else:
            phase = "zero_update_preflight"
            numeric = capacity.strict_fp32_mode()
            with numeric as mode:
                _require(mode == expansion.NUMERIC_MODE, "strict FP32 runtime settings differ")
                preflight = _preflight(target, source_checkpoint, source_readout, inputs, paths, bank)
            control, receipt = _make_control(population, preflight)
            ready = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}
            _write_json(ready_path, ready, "executed zero-update production READY preflight")
            control.mark_preflight_ready(receipt)
        progress["ready"] = True
        _require(all(check.get("executed") and check.get("passed") for check in preflight["checks"].values()), "READY consumer rejected an unexecuted preflight check")
        if args.preflight_only:
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "gradient_calls": 0, "optimizer_updates": 0, "next_action": "NONE"}
        receipt = _validate_ready(target, population, control)
        control.start_execution()
        control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)

        phase = "frozen_conditional_ols_measurement"
        with capacity.strict_fp32_mode() as mode:
            _require(mode == expansion.NUMERIC_MODE, "strict FP32 runtime settings differ")
            runtime = _new_runtime(source_checkpoint, torch.device("cuda:0"))
            evaluators = diagnostic.load_evaluators(paths, torch.device("cuda:0"))
            runtime_before = _snapshot_state(runtime)
            evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
            completed = set(manifest.get("completed_chunks", []))
            chunk_names: list[str] = []
            for start in range(0, TOTAL_OUTPUTS, CHUNK_ROWS):
                stop = min(TOTAL_OUTPUTS, start + CHUNK_ROWS)
                name = _chunk_name(start, stop)
                path = target / name
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                    _validate_chunk(payload, outputs, start, stop)
                    completed.add(name)
                elif name not in completed:
                    payload = _readout_payload(runtime, evaluators, outputs, start, stop, device=torch.device("cuda:0"))
                    _write_torch(path, payload, f"frozen production OLS decode rows {start}:{stop}")
                    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                    _validate_chunk(saved, outputs, start, stop)
                    completed.add(name)
                else:
                    raise FrozenGenerationError(f"manifest marks missing output chunk complete: {name}")
                chunk_names.append(name)
                manifest = _make_manifest(target, population, bank, status="RUNNING", completed_chunks=sorted(completed))
                _write_manifest_progress(manifest_path, manifest)
            measurement_freeze = _verify_measurement_freeze(runtime, runtime_before, evaluators, evaluator_before)
            _require(measurement_freeze["passed"], "formal frozen measurement changed model/evaluator state or gradients")
        _require(completed == set(chunk_names), "readout chunk manifest does not cover all 2,210 rows")
        summary, row_metrics = _summary_and_rows(target, outputs, bank)
        _write_or_validate_torch(target / "sample_row_metrics.pt", row_metrics, "float64 CE/margin, prediction, image integrity, and sample metadata")
        diversity = _descriptive_diversity(target, inputs, outputs)
        gallery = _gallery(target, outputs, summary)
        replay = json.loads((target / "preflight.json").read_text(encoding="utf-8"))["preflight"]["checks"]["real_encoder_to_cached_ols_replay"]
        report = _build_report(target, population, bank, summary, row_metrics, replay, preflight, diversity, gallery, measurement_freeze, time.perf_counter() - started)
        if report["status"] == "VALID_COMPLETED":
            control.complete()
        else:
            control.valid_failed()
        report["research_run_state"] = control.state
        manifest = _make_manifest(target, population, bank, status="COMPLETED", completed_chunks=sorted(completed))
        _write_manifest_progress(manifest_path, manifest)
        _write_json(target / "report.json", report, "terminal frozen conditional OLS generation entry decision")
        return report
    except Exception as error:
        if target.is_dir() and not (target / "report.json").exists() and not (target / "invalid_harness.json").exists():
            incident = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "training_recommendation": None, "next_action": "NONE", "phase": phase, "failure_type": type(error).__name__, "failure": str(error), "checks_executed": progress, "scientific_updates": 0, "gradient_calls": 0, "optimizer_updates": 0, "elapsed_seconds_observation_only": time.perf_counter() - started}
            try:
                _write_json(target / "invalid_harness.json", incident, "engineering failure record; no scientific classification")
            except OSError:
                pass
        raise


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
