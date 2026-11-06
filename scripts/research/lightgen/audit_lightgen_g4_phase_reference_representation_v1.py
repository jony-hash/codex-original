"""Audit saved G4 phase-reference associations and generation summaries on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


TASK_ID = "LIGHTGEN_G4_PHASE_REFERENCE_REPRESENTATION_AUDIT_V1"
ENDPOINT_NAMES = ("C_seed_2026101501", "seed_2026102151", "seed_2026102152")
EXPECTED_ROWS = 55_000
EXPECTED_GENERATION_ROWS = 180_000
N_CHANNELS = 100
N_CLASSES = 10
N_G4_COMPONENTS = 4
N_STREAMS = 3
SAMPLES_PER_CLASS_STREAM = 1_000
NMI_TOLERANCE = 1e-10

GENERATION_FIELDS = (
    "model",
    "stream",
    "method",
    "requested_label",
    "component_id",
    "proposal_id",
    "E_D_prediction",
    "E_C_prediction",
    "E_D_correct",
    "E_C_correct",
    "E_D_requested_margin",
    "E_C_requested_margin",
    "E_D_top_margin",
    "E_C_top_margin",
    "capture_power_w",
    "raw_mean_w_per_pixel",
    "raw_peak_w_per_pixel",
    "view_mean_intensity",
    "proxy_clipping_area_fraction",
    "proxy_clipped_excess_fraction",
)

PHASE_ROW_FIELDS = (
    "endpoint",
    "checkpoint_run_id",
    "checkpoint_step",
    "source_row",
    "label",
    "g4_component_id",
    "actual_raw_phase_reference_channel",
    "standardized_feature_max_channel",
    "reference_channel_mismatch",
    "raw_amplitude_max",
    "raw_amplitude_second",
    "raw_amplitude_gap",
    "raw_amplitude_relative_gap",
    "exact_top_tie",
)

ASSOCIATION_FIELDS = ("endpoint", "reference", "label", "row_count", "nmi")
CONTINGENCY_FIELDS = (
    "endpoint",
    "reference",
    "label",
    "g4_component_id",
    "reference_channel",
    "count",
)


class AuditError(RuntimeError):
    """A requested source or measurement contract was not satisfied."""


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    expected_seed: int
    checkpoint: Path
    cache: Path
    priors: Path


@dataclass
class EndpointData:
    spec: EndpointSpec
    checkpoint: dict[str, Any]
    cache: dict[str, Any]
    priors: dict[str, Any]
    rows: torch.Tensor
    labels: torch.Tensor
    features: torch.Tensor
    assignments: torch.Tensor
    scale: torch.Tensor
    class_counts: list[int]
    source_identity: dict[str, Any]


@dataclass
class GenerationData:
    model: np.ndarray
    stream: np.ndarray
    method: np.ndarray
    label: np.ndarray
    component: np.ndarray
    proposal: np.ndarray
    ed_prediction: np.ndarray
    ec_prediction: np.ndarray
    ed_correct: np.ndarray
    ec_correct: np.ndarray
    metrics: dict[str, np.ndarray]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_torch(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input file is missing: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, dict), f"expected a dictionary artifact: {path}")
    return value


def _validate_file_arguments(paths: Iterable[Path], output_dir: Path) -> None:
    resolved_inputs = [path.resolve() for path in paths]
    _require(len(set(resolved_inputs)) == len(resolved_inputs), "two explicit input paths resolve to one file")
    resolved_output = output_dir.resolve()
    _require(
        resolved_output not in resolved_inputs,
        "output directory resolves to an input artifact",
    )


def _load_endpoint(spec: EndpointSpec) -> EndpointData:
    checkpoint = _load_torch(spec.checkpoint)
    cache = _load_torch(spec.cache)
    priors = _load_torch(spec.priors)

    if spec.name == ENDPOINT_NAMES[0]:
        _require(
            checkpoint.get("schema") == "lightgen_small_gaussian_foreground_balance_checkpoint.v1",
            "C endpoint checkpoint schema is not the explicit foreground-balance schema",
        )
        _require(checkpoint.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1",
                 "C endpoint checkpoint belongs to a different task")
        _require(checkpoint.get("branch") == "C", "explicit C checkpoint is not branch C")
        _require(checkpoint.get("completed_global_step") == 45_000,
                 "explicit C checkpoint is not global step 45,000")
        _require(checkpoint.get("seed") == spec.expected_seed,
                 "C endpoint checkpoint has the wrong seed")
        _require(cache.get("schema") == "lightgen_small_conditional_mixture_train_features.v1",
                 "C feature cache schema is not the explicit conditional-mixture schema")
        _require(cache.get("task_id") == "LIGHTGEN_SMALL_CONDITIONAL_MIXTURE_PRIOR_AB_V1",
                 "C feature cache belongs to a different task")
        _require(priors.get("schema") == "lightgen_small_conditional_mixture_priors.v1",
                 "C prior artifact schema is not the explicit conditional-mixture schema")
        _require(priors.get("task_id") == "LIGHTGEN_SMALL_CONDITIONAL_MIXTURE_PRIOR_AB_V1",
                 "C priors belong to a different task")
        _require(cache.get("source_run_id") == checkpoint.get("run_id"),
                 "C feature cache is not linked to the explicit C checkpoint")
    else:
        _require(checkpoint.get("schema") == "lightgen_small_gaussian_replication_checkpoint.v1",
                 f"{spec.name} checkpoint schema is not the explicit replication schema")
        _require(checkpoint.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1",
                 f"{spec.name} checkpoint belongs to a different task")
        _require(checkpoint.get("seed") == spec.expected_seed,
                 f"{spec.name} checkpoint has the wrong seed")
        _require(checkpoint.get("completed_global_step") == 45_000,
                 f"{spec.name} checkpoint is not step 45,000")
        _require(cache.get("schema") == "lightgen_small_replication_train_features.v1",
                 f"{spec.name} feature cache schema is not the explicit replication schema")
        _require(cache.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1",
                 f"{spec.name} feature cache belongs to a different task")
        _require(cache.get("seed") == spec.expected_seed,
                 f"{spec.name} feature cache has the wrong seed")
        _require(priors.get("schema") == "lightgen_small_conditional_mixture_priors.v1",
                 f"{spec.name} prior artifact schema is not the explicit conditional-mixture schema")
        _require(priors.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1",
                 f"{spec.name} priors belong to a different task")
        _require(priors.get("seed") == spec.expected_seed,
                 f"{spec.name} priors have the wrong seed")

    _require(cache.get("source_checkpoint_step") == 45_000,
             f"{spec.name} feature cache does not identify the 45,000-step endpoint")
    _require(checkpoint.get("run_id"), f"{spec.name} checkpoint has no run identity")
    _require("rows" in cache and "labels" in cache and "features_200" in cache,
             f"{spec.name} feature cache is missing a required field")
    _require("G4_train_assignments_in_cache_order" in priors,
             f"{spec.name} prior artifact is missing G4 assignments")
    _require("scale" in checkpoint, f"{spec.name} checkpoint is missing scale")

    rows = cache["rows"]
    labels = cache["labels"]
    features = cache["features_200"]
    assignments = priors["G4_train_assignments_in_cache_order"]
    scale = checkpoint["scale"]
    _require(isinstance(rows, torch.Tensor) and rows.dtype == torch.int64 and rows.shape == (EXPECTED_ROWS,),
             f"{spec.name} rows must be an int64 vector of {EXPECTED_ROWS} entries")
    _require(isinstance(labels, torch.Tensor) and labels.dtype == torch.int64 and labels.shape == (EXPECTED_ROWS,),
             f"{spec.name} labels must be an int64 vector of {EXPECTED_ROWS} entries")
    _require(isinstance(features, torch.Tensor) and features.dtype == torch.float64
             and features.shape == (EXPECTED_ROWS, 2 * N_CHANNELS),
             f"{spec.name} features must be float64 with shape [{EXPECTED_ROWS}, 200]")
    _require(isinstance(assignments, torch.Tensor) and assignments.dtype == torch.int64
             and assignments.shape == (EXPECTED_ROWS,),
             f"{spec.name} G4 assignments must be an int64 vector of {EXPECTED_ROWS} entries")
    _require(isinstance(scale, torch.Tensor) and scale.dtype == torch.float64
             and scale.shape == (N_CHANNELS,),
             f"{spec.name} scale must be a float64 vector of {N_CHANNELS} entries")
    _require(bool(torch.isfinite(features).all()), f"{spec.name} features contain non-finite values")
    _require(bool(torch.isfinite(scale).all()) and bool((scale > 0).all()),
             f"{spec.name} scale must be finite and positive")
    _require(bool(torch.unique(rows).numel() == EXPECTED_ROWS),
             f"{spec.name} feature rows contain duplicate source row IDs")
    _require(bool(((rows >= 0) & (rows < 60_000)).all()),
             f"{spec.name} contains a source row outside the saved MNIST training population")
    _require(bool(((labels >= 0) & (labels < N_CLASSES)).all()),
             f"{spec.name} contains an invalid label")
    _require(bool(((assignments >= 0) & (assignments < N_G4_COMPONENTS)).all()),
             f"{spec.name} contains a G4 component outside 0..3")
    class_counts = torch.bincount(labels, minlength=N_CLASSES).tolist()
    _require(all(count > 0 for count in class_counts), f"{spec.name} is missing a TRAIN class")
    component_counts = torch.zeros((N_CLASSES, N_G4_COMPONENTS), dtype=torch.int64)
    component_counts.index_put_((labels, assignments), torch.ones_like(labels), accumulate=True)
    _require(bool((component_counts > 0).all()),
             f"{spec.name} has an empty G4 component in at least one class")

    source_identity = {
        "endpoint": spec.name,
        "seed": spec.expected_seed,
        "checkpoint_path": str(spec.checkpoint),
        "checkpoint_run_id": str(checkpoint["run_id"]),
        "checkpoint_schema": checkpoint["schema"],
        "checkpoint_task_id": checkpoint["task_id"],
        "checkpoint_step": int(checkpoint["completed_global_step"]),
        "cache_path": str(spec.cache),
        "cache_schema": cache["schema"],
        "cache_task_id": cache["task_id"],
        "cache_seed": cache.get("seed"),
        "cache_source_run_id": cache.get("source_run_id"),
        "cache_source_checkpoint_step": int(cache["source_checkpoint_step"]),
        "priors_path": str(spec.priors),
        "priors_schema": priors["schema"],
        "priors_task_id": priors["task_id"],
        "priors_seed": priors.get("seed"),
        "feature_rows": int(features.shape[0]),
        "feature_dtype": str(features.dtype),
        "scale_dtype": str(scale.dtype),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "source_row_min": int(rows.min()),
        "source_row_max": int(rows.max()),
        "class_counts": [int(value) for value in class_counts],
    }
    return EndpointData(
        spec=spec,
        checkpoint=checkpoint,
        cache=cache,
        priors=priors,
        rows=rows,
        labels=labels,
        features=features,
        assignments=assignments,
        scale=scale,
        class_counts=[int(value) for value in class_counts],
        source_identity=source_identity,
    )


def _feature_complex(features_200: torch.Tensor) -> torch.Tensor:
    _require(features_200.dtype == torch.float64 and features_200.ndim == 2
             and features_200.shape[1] == 2 * N_CHANNELS,
             "features must be float64 interleaved real/imaginary values with 100 channels")
    return torch.view_as_complex(features_200.reshape(-1, N_CHANNELS, 2).contiguous())


def _phase_metrics_torch(
    features_200: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    feature_complex = _feature_complex(features_200)
    _require(scale.dtype == torch.float64 and scale.shape == (N_CHANNELS,),
             "scale has an invalid dtype or shape")
    recovered = feature_complex * scale.sqrt().unsqueeze(0)
    _require(bool(torch.isfinite(recovered).all()), "scale restoration produced a non-finite coefficient")
    raw_amplitude = recovered.abs()
    standardized_amplitude = feature_complex.abs()
    raw_maximum, raw_indices = raw_amplitude.max(dim=1)
    _require(bool((raw_maximum > 0).all()), "a saved TRAIN coefficient row has no phase reference amplitude")
    raw_top = torch.topk(raw_amplitude, k=2, dim=1, largest=True, sorted=True).values
    standardized_indices = standardized_amplitude.argmax(dim=1)
    gap = raw_top[:, 0] - raw_top[:, 1]
    relative_gap = gap / raw_top[:, 0].clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "actual_raw_phase_reference_channel": raw_indices,
        "standardized_feature_max_channel": standardized_indices,
        "raw_amplitude_max": raw_maximum,
        "raw_amplitude_second": raw_top[:, 1],
        "raw_amplitude_gap": gap,
        "raw_amplitude_relative_gap": relative_gap,
        "exact_top_tie": gap == 0,
        "recovered_complex": recovered,
    }


def _nmi_from_counts_torch(counts: torch.Tensor) -> torch.Tensor:
    _require(counts.dtype == torch.int64 and counts.ndim == 2,
             "NMI contingency table must be a two-dimensional int64 tensor")
    total = counts.sum()
    _require(bool(total > 0), "NMI contingency table is empty")
    joint = counts.to(torch.float64) / total
    px = joint.sum(dim=1)
    py = joint.sum(dim=0)
    expected = px[:, None] * py[None, :]
    nonzero = joint > 0
    mutual_information = (joint[nonzero] * torch.log(joint[nonzero] / expected[nonzero])).sum()
    px_nz = px > 0
    py_nz = py > 0
    entropy_x = -(px[px_nz] * torch.log(px[px_nz])).sum()
    entropy_y = -(py[py_nz] * torch.log(py[py_nz])).sum()
    denominator = entropy_x + entropy_y
    if bool(denominator == 0):
        return torch.zeros_like(denominator)
    return 2 * mutual_information / denominator


def _association_torch(
    labels: torch.Tensor,
    assignments: torch.Tensor,
    actual_pivots: torch.Tensor,
    standardized_pivots: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for reference_name, pivots in (
        ("actual_raw_phase_reference", actual_pivots),
        ("standardized_feature_max", standardized_pivots),
    ):
        table = torch.zeros(
            (N_CLASSES, N_G4_COMPONENTS, N_CHANNELS),
            dtype=torch.int64,
            device=labels.device,
        )
        nmi_values: list[torch.Tensor] = []
        for label in range(N_CLASSES):
            mask = labels == label
            codes = assignments[mask] * N_CHANNELS + pivots[mask]
            class_table = torch.bincount(codes, minlength=N_G4_COMPONENTS * N_CHANNELS)
            class_table = class_table.reshape(N_G4_COMPONENTS, N_CHANNELS)
            table[label] = class_table
            nmi_values.append(_nmi_from_counts_torch(class_table))
        results[reference_name] = {
            "table": table,
            "nmi_by_class": torch.stack(nmi_values),
        }
    return results


def _numpy_nmi(counts: np.ndarray) -> float:
    table = np.asarray(counts, dtype=np.int64)
    _require(table.ndim == 2 and int(table.sum()) > 0, "NumPy NMI contingency table is empty")
    joint = table.astype(np.float64) / float(table.sum())
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    expected = px[:, None] * py[None, :]
    nz = joint > 0
    mutual_information = float(np.sum(joint[nz] * np.log(joint[nz] / expected[nz])))
    entropy_x = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    entropy_y = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return 0.0 if entropy_x + entropy_y == 0 else 2.0 * mutual_information / (entropy_x + entropy_y)


def _numpy_reference(data: EndpointData) -> dict[str, Any]:
    features = data.features.numpy().reshape(EXPECTED_ROWS, N_CHANNELS, 2)
    feature_complex = features[:, :, 0] + 1j * features[:, :, 1]
    standardized_amplitude = np.abs(feature_complex)
    recovered_amplitude = standardized_amplitude * np.sqrt(data.scale.numpy())[None, :]
    actual_pivots = np.argmax(recovered_amplitude, axis=1).astype(np.int64)
    standardized_pivots = np.argmax(standardized_amplitude, axis=1).astype(np.int64)
    top_two = np.partition(recovered_amplitude, N_CHANNELS - 2, axis=1)[:, -2:]
    top_max = top_two.max(axis=1)
    top_second = top_two.min(axis=1)
    gap = top_max - top_second
    relative_gap = gap / np.maximum(top_max, np.finfo(np.float64).tiny)
    tables: dict[str, np.ndarray] = {}
    nmi_by_reference: dict[str, list[float]] = {}
    labels = data.labels.numpy()
    assignments = data.assignments.numpy()
    for reference_name, pivots in (
        ("actual_raw_phase_reference", actual_pivots),
        ("standardized_feature_max", standardized_pivots),
    ):
        counts = np.zeros((N_CLASSES, N_G4_COMPONENTS, N_CHANNELS), dtype=np.int64)
        values: list[float] = []
        for label in range(N_CLASSES):
            mask = labels == label
            codes = assignments[mask] * N_CHANNELS + pivots[mask]
            class_table = np.bincount(codes, minlength=N_G4_COMPONENTS * N_CHANNELS)
            class_table = class_table.reshape(N_G4_COMPONENTS, N_CHANNELS)
            counts[label] = class_table
            values.append(_numpy_nmi(class_table))
        tables[reference_name] = counts
        nmi_by_reference[reference_name] = values
    return {
        "actual_raw_phase_reference_channel": actual_pivots,
        "standardized_feature_max_channel": standardized_pivots,
        "raw_amplitude_max": top_max,
        "raw_amplitude_second": top_second,
        "raw_amplitude_gap": gap,
        "raw_amplitude_relative_gap": relative_gap,
        "exact_top_tie": gap == 0,
        "tables": tables,
        "nmi_by_reference": nmi_by_reference,
    }


def _to_numpy(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in metrics.items()
        if name != "recovered_complex"
    }


def _compare_gpu_numpy(
    gpu_metrics: dict[str, np.ndarray],
    gpu_association: dict[str, dict[str, Any]],
    numpy_metrics: dict[str, Any],
) -> dict[str, Any]:
    for field in (
        "actual_raw_phase_reference_channel",
        "standardized_feature_max_channel",
        "exact_top_tie",
    ):
        _require(
            np.array_equal(gpu_metrics[field], numpy_metrics[field]),
            f"CUDA and NumPy disagree on {field}",
        )
    max_gap_abs = float(np.max(np.abs(gpu_metrics["raw_amplitude_gap"] - numpy_metrics["raw_amplitude_gap"])))
    max_gap_rel = float(
        np.max(
            np.abs(gpu_metrics["raw_amplitude_relative_gap"] - numpy_metrics["raw_amplitude_relative_gap"])
        )
    )
    _require(np.allclose(gpu_metrics["raw_amplitude_max"], numpy_metrics["raw_amplitude_max"],
                         rtol=1e-12, atol=1e-14),
             "CUDA and NumPy disagree on recovered maximum amplitudes")
    _require(np.allclose(gpu_metrics["raw_amplitude_second"], numpy_metrics["raw_amplitude_second"],
                         rtol=1e-12, atol=1e-14),
             "CUDA and NumPy disagree on recovered second amplitudes")
    _require(np.allclose(gpu_metrics["raw_amplitude_gap"], numpy_metrics["raw_amplitude_gap"],
                         rtol=1e-12, atol=1e-14),
             "CUDA and NumPy disagree on recovered top-two amplitude gaps")
    _require(np.allclose(gpu_metrics["raw_amplitude_relative_gap"],
                          numpy_metrics["raw_amplitude_relative_gap"], rtol=1e-12, atol=1e-14),
             "CUDA and NumPy disagree on relative recovered top-two amplitude gaps")
    for reference_name, numpy_table in numpy_metrics["tables"].items():
        cuda_table = gpu_association[reference_name]["table"].detach().cpu().numpy()
        _require(np.array_equal(cuda_table, numpy_table),
                 f"CUDA and NumPy contingency tables differ for {reference_name}")
        cuda_nmi = gpu_association[reference_name]["nmi_by_class"].detach().cpu().numpy()
        numpy_nmi = np.asarray(numpy_metrics["nmi_by_reference"][reference_name], dtype=np.float64)
        max_diff = float(np.max(np.abs(cuda_nmi - numpy_nmi)))
        _require(max_diff <= NMI_TOLERANCE,
                 f"CUDA and NumPy NMI differ by {max_diff:.3g} for {reference_name}")
    return {"max_gap_absolute_difference": max_gap_abs,
            "max_relative_gap_absolute_difference": max_gap_rel,
            "nmi_absolute_tolerance": NMI_TOLERANCE,
            "integer_contingency_exact": True,
            "indices_exact": True}


def _run_cuda_metrics(
    data: EndpointData,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    features = data.features.index_select(0, indices).to(device=device, dtype=torch.float64)
    labels = data.labels.index_select(0, indices).to(device=device, dtype=torch.int64)
    assignments = data.assignments.index_select(0, indices).to(device=device, dtype=torch.int64)
    scale = data.scale.to(device=device, dtype=torch.float64)
    metrics_torch = _phase_metrics_torch(features, scale)
    association = _association_torch(
        labels,
        assignments,
        metrics_torch["actual_raw_phase_reference_channel"],
        metrics_torch["standardized_feature_max_channel"],
    )
    end_event.record()
    torch.cuda.synchronize(device)
    metrics = _to_numpy(metrics_torch)
    report = {
        "rows_processed": int(features.shape[0]),
        "cuda_event_elapsed_ms": float(start_event.elapsed_time(end_event)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device": str(device),
    }
    del features, labels, assignments, scale, metrics_torch
    return metrics, association, report


def _select_preflight_indices(labels: torch.Tensor, per_class: int = 2) -> torch.Tensor:
    selected: list[int] = []
    labels_list = labels.tolist()
    for label in range(N_CLASSES):
        rows = [index for index, value in enumerate(labels_list) if value == label]
        _require(len(rows) >= per_class, f"preflight cannot select {per_class} rows for class {label}")
        selected.extend(rows[:per_class])
    return torch.tensor(selected, dtype=torch.int64)


def _validate_counterexamples(device: torch.device) -> dict[str, Any]:
    raw = torch.tensor([[2.0 + 0.0j, 1.0 + 0.0j]], dtype=torch.complex128, device=device)
    scale = torch.tensor([100.0, 1.0], dtype=torch.float64, device=device)
    standardized = raw / scale.sqrt().unsqueeze(0)
    old_pivot = standardized.abs().argmax(dim=1)
    recovered = standardized * scale.sqrt().unsqueeze(0)
    actual_pivot = recovered.abs().argmax(dim=1)
    _require(int(old_pivot.item()) == 1 and int(actual_pivot.item()) == 0,
             "the scale counterexample did not distinguish the old and actual pivot definitions")

    phase = torch.exp(torch.tensor(0.731, dtype=torch.float64, device=device) * 1j)
    rotated = standardized * phase
    rotated_recovered = rotated * scale.sqrt().unsqueeze(0)
    _require(torch.allclose(recovered.abs(), rotated_recovered.abs(), rtol=1e-12, atol=1e-14),
             "common phase changed the recovered amplitudes")
    _require(torch.equal(recovered.abs().argmax(dim=1), rotated_recovered.abs().argmax(dim=1)),
             "common phase changed the raw reference index")

    tie = torch.tensor([[1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex128, device=device)
    first_tie_index = int(tie.abs().argmax(dim=1).item())
    _require(first_tie_index == 0, "exact-tie convention no longer chooses the first channel")
    perfect_nmi = float(_nmi_from_counts_torch(
        torch.tensor([[5, 0], [0, 5]], dtype=torch.int64, device=device)
    ).item())
    independent_nmi = float(_nmi_from_counts_torch(
        torch.tensor([[5, 5], [5, 5]], dtype=torch.int64, device=device)
    ).item())
    _require(abs(perfect_nmi - 1.0) <= NMI_TOLERANCE,
             "GPU NMI does not return one for perfect association")
    _require(abs(independent_nmi) <= NMI_TOLERANCE,
             "GPU NMI does not return zero for exact independence")
    return {
        "scale_counterexample": {"actual_raw_pivot": 0, "standardized_max_pivot": 1,
                                 "faulty_standardized_pivot_rejected": True},
        "common_phase_invariance": True,
        "exact_tie_first_channel": first_tie_index,
        "perfect_association_nmi": perfect_nmi,
        "independence_nmi": independent_nmi,
    }


def _run_preflight(
    specs: tuple[EndpointSpec, ...],
    output_dir: Path,
    device: torch.device,
    historical_report_path: Path,
    generation: GenerationData,
) -> dict[str, Any]:
    counterexamples = _validate_counterexamples(device)
    historical_report = json.loads(historical_report_path.read_text(encoding="utf-8"))
    _require(historical_report.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1",
             "historical report is not the explicit three-endpoint replication report")
    endpoint_results: list[dict[str, Any]] = []
    reference_rows: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    for spec in specs:
        data = _load_endpoint(spec)
        if reference_rows is None:
            reference_rows = data.rows.numpy().copy()
            reference_labels = data.labels.numpy().copy()
        else:
            _require(np.array_equal(data.rows.numpy(), reference_rows),
                     f"{spec.name} does not contain the same ordered TRAIN row IDs")
            _require(np.array_equal(data.labels.numpy(), reference_labels),
                     f"{spec.name} does not contain the same ordered TRAIN labels")
        indices = _select_preflight_indices(data.labels)
        gpu_metrics, gpu_association, cuda_observation = _run_cuda_metrics(data, indices, device)
        selected = EndpointData(
            spec=data.spec,
            checkpoint=data.checkpoint,
            cache=data.cache,
            priors=data.priors,
            rows=data.rows.index_select(0, indices),
            labels=data.labels.index_select(0, indices),
            features=data.features.index_select(0, indices),
            assignments=data.assignments.index_select(0, indices),
            scale=data.scale,
            class_counts=[int((data.labels.index_select(0, indices) == label).sum())
                          for label in range(N_CLASSES)],
            source_identity=data.source_identity,
        )
        numpy_metrics = _numpy_reference_for_subset(selected)
        parity = _compare_gpu_numpy(gpu_metrics, gpu_association, numpy_metrics)
        endpoint_results.append({
            "endpoint": spec.name,
            "rows": int(indices.numel()),
            "two_rows_per_class": True,
            "source_identity": data.source_identity,
            "cuda_observation": cuda_observation,
            "independent_numpy_parity": parity,
        })
        del data, selected, gpu_metrics, gpu_association
        torch.cuda.empty_cache()

    generation_preflight = _preflight_generation_aggregation(generation, device)
    receipt = {
        "task_id": TASK_ID,
        "status": "PASSED",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "counterexamples": counterexamples,
        "endpoints": endpoint_results,
        "generation_aggregation": generation_preflight,
        "new_training_updates": 0,
        "optimizer_instances_created": 0,
        "prior_refits": 0,
        "new_sampling": 0,
        "test_reads": 0,
        "human_reviews": 0,
        "lpips_evaluations": 0,
    }
    _write_json(output_dir / "preflight_receipt.json", receipt)
    return receipt


def _numpy_reference_for_subset(data: EndpointData) -> dict[str, Any]:
    features = data.features.numpy().reshape(-1, N_CHANNELS, 2)
    feature_complex = features[:, :, 0] + 1j * features[:, :, 1]
    standardized_amplitude = np.abs(feature_complex)
    recovered_amplitude = standardized_amplitude * np.sqrt(data.scale.numpy())[None, :]
    actual_pivots = np.argmax(recovered_amplitude, axis=1).astype(np.int64)
    standardized_pivots = np.argmax(standardized_amplitude, axis=1).astype(np.int64)
    top_two = np.partition(recovered_amplitude, N_CHANNELS - 2, axis=1)[:, -2:]
    top_max = top_two.max(axis=1)
    top_second = top_two.min(axis=1)
    gap = top_max - top_second
    labels = data.labels.numpy()
    assignments = data.assignments.numpy()
    tables: dict[str, np.ndarray] = {}
    nmi_by_reference: dict[str, list[float]] = {}
    for reference_name, pivots in (
        ("actual_raw_phase_reference", actual_pivots),
        ("standardized_feature_max", standardized_pivots),
    ):
        counts = np.zeros((N_CLASSES, N_G4_COMPONENTS, N_CHANNELS), dtype=np.int64)
        values: list[float] = []
        for label in range(N_CLASSES):
            mask = labels == label
            class_table = np.bincount(
                assignments[mask] * N_CHANNELS + pivots[mask],
                minlength=N_G4_COMPONENTS * N_CHANNELS,
            ).reshape(N_G4_COMPONENTS, N_CHANNELS)
            counts[label] = class_table
            values.append(_numpy_nmi(class_table))
        tables[reference_name] = counts
        nmi_by_reference[reference_name] = values
    return {
        "actual_raw_phase_reference_channel": actual_pivots,
        "standardized_feature_max_channel": standardized_pivots,
        "raw_amplitude_max": top_max,
        "raw_amplitude_second": top_second,
        "raw_amplitude_gap": gap,
        "raw_amplitude_relative_gap": gap / np.maximum(top_max, np.finfo(np.float64).tiny),
        "exact_top_tie": gap == 0,
        "tables": tables,
        "nmi_by_reference": nmi_by_reference,
    }


def _verify_phase_output(
    path: Path,
    data: EndpointData,
    expected: dict[str, Any],
) -> int:
    _require(path.is_file(), f"phase reference output is missing: {path}")
    expected_columns = list(PHASE_ROW_FIELDS)
    seen = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == expected_columns,
                 "phase reference output has missing, reordered, or renamed statistic columns")
        for index, row in enumerate(reader):
            _require(index < EXPECTED_ROWS, "phase reference output contains extra rows")
            _require(row["endpoint"] == data.spec.name, "phase reference output contains a wrong endpoint")
            _require(row["checkpoint_run_id"] == str(data.checkpoint["run_id"]),
                     "phase reference output contains a wrong checkpoint identity")
            _require(int(row["checkpoint_step"]) == 45_000,
                     "phase reference output contains a wrong checkpoint step")
            _require(int(row["source_row"]) == int(data.rows[index]),
                     f"phase reference output has a wrong source row at offset {index}")
            _require(int(row["label"]) == int(data.labels[index]),
                     f"phase reference output has a wrong label at offset {index}")
            _require(int(row["g4_component_id"]) == int(data.assignments[index]),
                     f"phase reference output has a wrong G4 component at offset {index}")
            _require(int(row["actual_raw_phase_reference_channel"])
                     == int(expected["actual_raw_phase_reference_channel"][index]),
                     f"phase reference output has a wrong actual pivot at offset {index}")
            _require(int(row["standardized_feature_max_channel"])
                     == int(expected["standardized_feature_max_channel"][index]),
                     f"phase reference output has a wrong standardized pivot at offset {index}")
            _require(row["reference_channel_mismatch"]
                     == str(int(expected["actual_raw_phase_reference_channel"][index]
                                != expected["standardized_feature_max_channel"][index])),
                     f"phase reference output has a mislabeled mismatch at offset {index}")
            _require(row["exact_top_tie"] == str(int(expected["exact_top_tie"][index])),
                     f"phase reference output has a wrong exact-tie flag at offset {index}")
            for field in ("raw_amplitude_max", "raw_amplitude_second", "raw_amplitude_gap",
                          "raw_amplitude_relative_gap"):
                observed = float(row[field])
                expected_value = float(expected[field][index])
                _require(math.isfinite(observed) and math.isclose(
                    observed, expected_value, rel_tol=1e-12, abs_tol=1e-14
                ), f"phase reference output has a wrong or renamed value for {field} at offset {index}")
            seen += 1
    _require(seen == EXPECTED_ROWS, f"phase reference output has {seen} rows, expected {EXPECTED_ROWS}")
    return seen


def _phase_output_rows(data: EndpointData, metrics: dict[str, np.ndarray]) -> Iterable[dict[str, Any]]:
    checkpoint_id = str(data.checkpoint["run_id"])
    for i in range(EXPECTED_ROWS):
        actual = int(metrics["actual_raw_phase_reference_channel"][i])
        standardized = int(metrics["standardized_feature_max_channel"][i])
        yield {
            "endpoint": data.spec.name,
            "checkpoint_run_id": checkpoint_id,
            "checkpoint_step": 45_000,
            "source_row": int(data.rows[i]),
            "label": int(data.labels[i]),
            "g4_component_id": int(data.assignments[i]),
            "actual_raw_phase_reference_channel": actual,
            "standardized_feature_max_channel": standardized,
            "reference_channel_mismatch": int(actual != standardized),
            "raw_amplitude_max": format(float(metrics["raw_amplitude_max"][i]), ".17g"),
            "raw_amplitude_second": format(float(metrics["raw_amplitude_second"][i]), ".17g"),
            "raw_amplitude_gap": format(float(metrics["raw_amplitude_gap"][i]), ".17g"),
            "raw_amplitude_relative_gap": format(float(metrics["raw_amplitude_relative_gap"][i]), ".17g"),
            "exact_top_tie": int(bool(metrics["exact_top_tie"][i])),
        }


def _association_output_rows(endpoint: str, tables: dict[str, np.ndarray], nmi: dict[str, list[float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    association_rows: list[dict[str, Any]] = []
    contingency_rows: list[dict[str, Any]] = []
    for reference_name in ("actual_raw_phase_reference", "standardized_feature_max"):
        for label in range(N_CLASSES):
            association_rows.append({
                "endpoint": endpoint,
                "reference": reference_name,
                "label": label,
                "row_count": int(tables[reference_name][label].sum()),
                "nmi": format(float(nmi[reference_name][label]), ".17g"),
            })
            table = tables[reference_name][label]
            for component in range(N_G4_COMPONENTS):
                for channel in range(N_CHANNELS):
                    contingency_rows.append({
                        "endpoint": endpoint,
                        "reference": reference_name,
                        "label": label,
                        "g4_component_id": component,
                        "reference_channel": channel,
                        "count": int(table[component, channel]),
                    })
    return association_rows, contingency_rows


def _verify_association_outputs(
    association_path: Path,
    contingency_path: Path,
    endpoint: str,
    expected: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_assoc: list[dict[str, Any]] = []
    expected_contingency: list[dict[str, Any]] = []
    for ref in ("actual_raw_phase_reference", "standardized_feature_max"):
        for label in range(N_CLASSES):
            table = expected["tables"][ref][label]
            expected_assoc.append({"endpoint": endpoint, "reference": ref, "label": label,
                                   "row_count": int(table.sum()),
                                   "nmi": float(expected["nmi_by_reference"][ref][label])})
            for component in range(N_G4_COMPONENTS):
                for channel in range(N_CHANNELS):
                    expected_contingency.append({"endpoint": endpoint, "reference": ref,
                                                 "label": label, "g4_component_id": component,
                                                 "reference_channel": channel,
                                                 "count": int(table[component, channel])})
    read_assoc: list[dict[str, Any]] = []
    with association_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(ASSOCIATION_FIELDS),
                 "association output has missing, reordered, or renamed statistic columns")
        for i, row in enumerate(reader):
            _require(i < len(expected_assoc), "association output contains extra rows")
            wanted = expected_assoc[i]
            _require(row["endpoint"] == endpoint == wanted["endpoint"],
                     "association output contains a wrong endpoint")
            _require(row["reference"] == wanted["reference"],
                     "association output has a mislabeled phase reference statistic")
            _require(int(row["label"]) == wanted["label"]
                     and int(row["row_count"]) == wanted["row_count"],
                     "association output has a wrong label or denominator")
            _require(math.isclose(float(row["nmi"]), wanted["nmi"], rel_tol=0, abs_tol=NMI_TOLERANCE),
                     "association output has a wrong or mixed NMI value")
            read_assoc.append({**wanted, "nmi": float(row["nmi"])})
    _require(len(read_assoc) == len(expected_assoc), "association output is missing rows")

    read_contingency: list[dict[str, Any]] = []
    with contingency_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(CONTINGENCY_FIELDS),
                 "contingency output has missing, reordered, or renamed statistic columns")
        for i, row in enumerate(reader):
            _require(i < len(expected_contingency), "contingency output contains extra rows")
            wanted = expected_contingency[i]
            observed_key = (row["endpoint"], row["reference"], int(row["label"]),
                            int(row["g4_component_id"]), int(row["reference_channel"]))
            expected_key = (wanted["endpoint"], wanted["reference"], wanted["label"],
                            wanted["g4_component_id"], wanted["reference_channel"])
            _require(observed_key == expected_key,
                     "contingency output has a wrong endpoint, row, or statistic name")
            count = int(row["count"])
            _require(count == wanted["count"], "contingency output has a wrong count")
            read_contingency.append({**wanted, "count": count})
    _require(len(read_contingency) == len(expected_contingency), "contingency output is missing rows")
    return read_assoc, read_contingency


def _write_endpoint_outputs(
    output_dir: Path,
    data: EndpointData,
    gpu_metrics: dict[str, np.ndarray],
    gpu_association: dict[str, dict[str, Any]],
    numpy_expected: dict[str, Any],
) -> dict[str, Any]:
    endpoint_dir = output_dir / "endpoints" / data.spec.name
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    phase_path = endpoint_dir / "phase_reference_rows.csv"
    association_path = endpoint_dir / "g4_pivot_association_by_class.csv"
    contingency_path = endpoint_dir / "g4_component_pivot_contingency.csv"

    tables: dict[str, np.ndarray] = {}
    nmi_by_reference: dict[str, list[float]] = {}
    for reference in ("actual_raw_phase_reference", "standardized_feature_max"):
        tables[reference] = gpu_association[reference]["table"].detach().cpu().numpy()
        nmi_by_reference[reference] = gpu_association[reference]["nmi_by_class"].detach().cpu().numpy().tolist()
    _write_csv(phase_path, PHASE_ROW_FIELDS, _phase_output_rows(data, gpu_metrics))
    association_rows, contingency_rows = _association_output_rows(data.spec.name, tables, nmi_by_reference)
    _write_csv(association_path, ASSOCIATION_FIELDS, association_rows)
    _write_csv(contingency_path, CONTINGENCY_FIELDS, contingency_rows)

    verified_count = _verify_phase_output(phase_path, data, numpy_expected)
    read_assoc, read_contingency = _verify_association_outputs(
        association_path,
        contingency_path,
        data.spec.name,
        numpy_expected,
    )
    mismatch_count = int(np.count_nonzero(gpu_metrics["actual_raw_phase_reference_channel"]
                                          != gpu_metrics["standardized_feature_max_channel"]))
    tie_count = int(np.count_nonzero(gpu_metrics["exact_top_tie"]))
    receipt = {
        "task_id": TASK_ID,
        "status": "COMPLETE_AND_READBACK_VERIFIED",
        "endpoint": data.spec.name,
        "source_identity": data.source_identity,
        "feature_rows": verified_count,
        "pivot_mismatch_rows": mismatch_count,
        "pivot_mismatch_fraction": mismatch_count / EXPECTED_ROWS,
        "exact_top_tie_rows": tie_count,
        "association_rows": len(read_assoc),
        "contingency_rows": len(read_contingency),
        "phase_reference_output": str(phase_path),
        "association_output": str(association_path),
        "contingency_output": str(contingency_path),
        "numpy_parity": {
            "indices_exact": True,
            "tie_flags_exact": True,
            "contingency_exact": True,
            "nmi_max_abs_difference": max(
                abs(float(gpu_association[ref]["nmi_by_class"][i].item())
                    - float(numpy_expected["nmi_by_reference"][ref][i]))
                for ref in ("actual_raw_phase_reference", "standardized_feature_max")
                for i in range(N_CLASSES)
            ),
            "nmi_tolerance": NMI_TOLERANCE,
        },
    }
    _write_json(endpoint_dir / "endpoint_receipt.json", receipt)
    return receipt


def _verify_completed_endpoint(output_dir: Path, data: EndpointData) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_dir = output_dir / "endpoints" / data.spec.name
    receipt_path = endpoint_dir / "endpoint_receipt.json"
    _require(receipt_path.is_file(), f"completed endpoint receipt is missing for {data.spec.name}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("task_id") == TASK_ID
             and receipt.get("status") == "COMPLETE_AND_READBACK_VERIFIED"
             and receipt.get("endpoint") == data.spec.name,
             f"endpoint receipt identity or status is invalid for {data.spec.name}")
    _require(receipt.get("source_identity") == data.source_identity,
             f"endpoint receipt source identity changed for {data.spec.name}")
    _require(receipt.get("cuda_execution_confirmed") is True,
             f"endpoint receipt does not confirm CUDA execution for {data.spec.name}")
    cuda_observation = receipt.get("cuda_observation")
    _require(isinstance(cuda_observation, dict)
             and cuda_observation.get("rows_processed") == EXPECTED_ROWS
             and str(cuda_observation.get("device", "")).startswith("cuda:")
             and float(cuda_observation.get("cuda_event_elapsed_ms", 0)) > 0,
             f"endpoint receipt has invalid full-population CUDA evidence for {data.spec.name}")
    numpy_expected = _numpy_reference(data)
    phase_path = endpoint_dir / "phase_reference_rows.csv"
    association_path = endpoint_dir / "g4_pivot_association_by_class.csv"
    contingency_path = endpoint_dir / "g4_component_pivot_contingency.csv"
    row_count = _verify_phase_output(phase_path, data, numpy_expected)
    assoc, contingency = _verify_association_outputs(
        association_path, contingency_path, data.spec.name, numpy_expected
    )
    _require(row_count == EXPECTED_ROWS and receipt.get("feature_rows") == row_count,
             f"endpoint receipt row count is invalid for {data.spec.name}")
    _require(len(assoc) == 2 * N_CLASSES and len(contingency) == 2 * N_CLASSES * N_G4_COMPONENTS * N_CHANNELS,
             f"endpoint output coverage is incomplete for {data.spec.name}")
    return receipt, assoc, contingency


def _parse_bool(value: str, name: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return 1
    if normalized in {"0", "false"}:
        return 0
    raise AuditError(f"generation field {name} is not a saved boolean: {value!r}")


def _parse_finite_float(value: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise AuditError(f"generation field {name} is not numeric: {value!r}") from error
    _require(math.isfinite(number), f"generation field {name} is non-finite")
    return number


def _load_generation_csv(path: Path) -> GenerationData:
    _require(path.is_file(), f"generation metrics CSV is missing: {path}")
    model_map = {name: index for index, name in enumerate(ENDPOINT_NAMES)}
    method_map = {"G1": 0, "G4": 1}
    arrays: dict[str, list[Any]] = {
        "model": [], "stream": [], "method": [], "label": [], "component": [], "proposal": [],
        "ed_prediction": [], "ec_prediction": [], "ed_correct": [], "ec_correct": [],
    }
    all_numeric_fields = (
        "E_D_requested_margin", "E_C_requested_margin", "E_D_top_margin", "E_C_top_margin",
        "capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel", "view_mean_intensity",
        "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction",
    )
    aggregated_metric_names = (
        "E_D_requested_margin", "E_C_requested_margin", "E_D_top_margin", "E_C_top_margin",
        "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction",
    )
    metrics: dict[str, list[float]] = {name: [] for name in aggregated_metric_names}
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(GENERATION_FIELDS),
                 "generation CSV schema has missing, reordered, or renamed evaluator statistics")
        for row in reader:
            _require(count < EXPECTED_GENERATION_ROWS, "generation CSV contains extra records")
            _require(row["model"] in model_map, f"generation CSV contains an unknown endpoint {row['model']!r}")
            _require(row["method"] in method_map, f"generation CSV contains an unknown sampling method {row['method']!r}")
            model = model_map[row["model"]]
            method = method_map[row["method"]]
            try:
                stream_id = int(row["stream"])
                label = int(row["requested_label"])
                component = int(row["component_id"])
                proposal = int(row["proposal_id"])
                ed_prediction = int(row["E_D_prediction"])
                ec_prediction = int(row["E_C_prediction"])
            except ValueError as error:
                raise AuditError("generation CSV contains a non-integer identity or prediction") from error
            _require(0 <= stream_id < N_STREAMS and 0 <= label < N_CLASSES,
                     "generation CSV contains an invalid stream or requested class")
            _require(component == 0 if method == 0 else 0 <= component < N_G4_COMPONENTS,
                     "generation CSV contains a component ID inconsistent with the sampling method")
            _require(0 <= proposal < SAMPLES_PER_CLASS_STREAM,
                     "generation CSV contains an out-of-range proposal ID")
            _require(0 <= ed_prediction < N_CLASSES and 0 <= ec_prediction < N_CLASSES,
                     "generation CSV contains an invalid evaluator prediction")
            ed_correct = _parse_bool(row["E_D_correct"], "E_D_correct")
            ec_correct = _parse_bool(row["E_C_correct"], "E_C_correct")
            _require(ed_correct == int(ed_prediction == label),
                     "E_D_correct disagrees with E_D_prediction and requested_label")
            _require(ec_correct == int(ec_prediction == label),
                     "E_C_correct disagrees with E_C_prediction and requested_label")
            arrays["model"].append(model)
            arrays["stream"].append(stream_id)
            arrays["method"].append(method)
            arrays["label"].append(label)
            arrays["component"].append(component)
            arrays["proposal"].append(proposal)
            arrays["ed_prediction"].append(ed_prediction)
            arrays["ec_prediction"].append(ec_prediction)
            arrays["ed_correct"].append(ed_correct)
            arrays["ec_correct"].append(ec_correct)
            numeric_values = {name: _parse_finite_float(row[name], name) for name in all_numeric_fields}
            for name in aggregated_metric_names:
                metrics[name].append(numeric_values[name])
            _require(0 <= metrics["proxy_clipping_area_fraction"][-1] <= 1
                     and 0 <= metrics["proxy_clipped_excess_fraction"][-1] <= 1,
                     "generation CSV contains a clipping fraction outside [0, 1]")
            count += 1
    _require(count == EXPECTED_GENERATION_ROWS,
             f"generation CSV has {count} records, expected {EXPECTED_GENERATION_ROWS}")
    casts: dict[str, Any] = {
        "model": np.int8, "stream": np.int8, "method": np.int8, "label": np.int8,
        "component": np.int16, "proposal": np.int16, "ed_prediction": np.int8,
        "ec_prediction": np.int8, "ed_correct": np.int8, "ec_correct": np.int8,
    }
    parsed = {name: np.asarray(values, dtype=casts[name]) for name, values in arrays.items()}
    parsed_metrics = {name: np.asarray(values, dtype=np.float64) for name, values in metrics.items()}
    _validate_generation_population(parsed)
    return GenerationData(**parsed, metrics=parsed_metrics)


def _validate_generation_population(arrays: dict[str, np.ndarray]) -> None:
    expected_groups = np.zeros((len(ENDPOINT_NAMES), 2, N_STREAMS, N_CLASSES), dtype=np.int32)
    for model in range(len(ENDPOINT_NAMES)):
        for method in range(2):
            for stream in range(N_STREAMS):
                for label in range(N_CLASSES):
                    mask = ((arrays["model"] == model) & (arrays["method"] == method)
                            & (arrays["stream"] == stream) & (arrays["label"] == label))
                    selected = arrays["proposal"][mask]
                    _require(selected.size == SAMPLES_PER_CLASS_STREAM,
                             "generation CSV has a missing or extra model/method/stream/class group")
                    _require(np.array_equal(np.sort(selected), np.arange(SAMPLES_PER_CLASS_STREAM)),
                             "generation CSV has a duplicate or missing proposal row")
                    expected_groups[model, method, stream, label] = selected.size
    _require(bool((expected_groups == SAMPLES_PER_CLASS_STREAM).all()),
             "generation CSV population denominators are incomplete")
    _require(bool(((arrays["ed_correct"] == 0) | (arrays["ed_correct"] == 1)).all()
                  and ((arrays["ec_correct"] == 0) | (arrays["ec_correct"] == 1)).all()),
             "generation correctness fields are not binary")


def _select_generation_preflight(data: GenerationData) -> GenerationData:
    selected: list[int] = []
    for model in range(len(ENDPOINT_NAMES)):
        for method in range(2):
            for stream in range(N_STREAMS):
                for label in range(N_CLASSES):
                    indices = np.flatnonzero(
                        (data.model == model) & (data.method == method)
                        & (data.stream == stream) & (data.label == label)
                    )
                    _require(indices.size >= 2,
                             "generation preflight cannot select two saved rows per model/method/stream/class")
                    selected.extend(indices[:2].tolist())
    index = np.asarray(selected, dtype=np.int64)
    return GenerationData(
        model=data.model[index],
        stream=data.stream[index],
        method=data.method[index],
        label=data.label[index],
        component=data.component[index],
        proposal=data.proposal[index],
        ed_prediction=data.ed_prediction[index],
        ec_prediction=data.ec_prediction[index],
        ed_correct=data.ed_correct[index],
        ec_correct=data.ec_correct[index],
        metrics={name: values[index] for name, values in data.metrics.items()},
    )


def _numpy_generation_aggregation(data: GenerationData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def make_row(indices: np.ndarray, identity: dict[str, Any]) -> dict[str, Any]:
        count = int(indices.size)
        row = {
            **identity,
            "sample_count": count,
            "E_D_correct_count": int(data.ed_correct[indices].sum()),
            "E_D_recall": float(data.ed_correct[indices].mean()),
            "E_C_correct_count": int(data.ec_correct[indices].sum()),
            "E_C_recall": float(data.ec_correct[indices].mean()),
        }
        row.update({name: float(values[indices].mean()) for name, values in data.metrics.items()})
        return row

    class_rows: list[dict[str, Any]] = []
    for model in range(len(ENDPOINT_NAMES)):
        for method in range(2):
            for label in range(N_CLASSES):
                indices = np.flatnonzero(
                    (data.model == model) & (data.method == method) & (data.label == label)
                )
                if indices.size:
                    class_rows.append(make_row(indices, {
                        "model": ENDPOINT_NAMES[model],
                        "method": "G1" if method == 0 else "G4",
                        "requested_label": label,
                    }))

    group_keys = sorted(set(zip(data.model.tolist(), data.method.tolist(), data.stream.tolist(),
                                data.label.tolist(), data.component.tolist())))
    grouped_rows: list[dict[str, Any]] = []
    for model, method, stream, label, component in group_keys:
        indices = np.flatnonzero(
            (data.model == model) & (data.method == method) & (data.stream == stream)
            & (data.label == label) & (data.component == component)
        )
        grouped_rows.append(make_row(indices, {
            "model": ENDPOINT_NAMES[model],
            "method": "G1" if method == 0 else "G4",
            "stream": stream,
            "requested_label": label,
            "component_id": component,
        }))
    return class_rows, grouped_rows


def _verify_generation_aggregation_parity(
    actual_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    identity_fields: tuple[str, ...],
) -> None:
    _require(len(actual_rows) == len(expected_rows),
             "CUDA generation preflight returned the wrong number of groups")
    for actual, expected in zip(actual_rows, expected_rows):
        for field in identity_fields:
            _require(actual[field] == expected[field],
                     f"CUDA generation preflight has a wrong {field}")
        _require(actual["sample_count"] == expected["sample_count"],
                 "CUDA generation preflight has a wrong group denominator")
        _require(actual["E_D_correct_count"] == expected["E_D_correct_count"]
                 and actual["E_C_correct_count"] == expected["E_C_correct_count"],
                 "CUDA generation preflight has a wrong evaluator count")
        for field, expected_value in expected.items():
            if field in identity_fields or field in ("sample_count", "E_D_correct_count", "E_C_correct_count"):
                continue
            _require(math.isclose(float(actual[field]), float(expected_value), rel_tol=0, abs_tol=1e-12),
                     f"CUDA generation preflight has a wrong or mixed statistic {field}")


def _preflight_generation_aggregation(data: GenerationData, device: torch.device) -> dict[str, Any]:
    sample = _select_generation_preflight(data)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    gpu_class_rows, gpu_grouped_rows = _aggregate_generation_cuda(sample, device)
    end_event.record()
    torch.cuda.synchronize(device)
    numpy_class_rows, numpy_grouped_rows = _numpy_generation_aggregation(sample)
    _verify_generation_aggregation_parity(
        gpu_class_rows, numpy_class_rows, ("model", "method", "requested_label")
    )
    _verify_generation_aggregation_parity(
        gpu_grouped_rows,
        numpy_grouped_rows,
        ("model", "method", "stream", "requested_label", "component_id"),
    )
    receipt = {
        "status": "PASSED",
        "input_records_validated": EXPECTED_GENERATION_ROWS,
        "preflight_rows": int(sample.model.size),
        "selection": "first two saved rows per endpoint/method/stream/class",
        "class_groups": len(gpu_class_rows),
        "stream_class_component_groups": len(gpu_grouped_rows),
        "gpu_device": torch.cuda.get_device_name(device),
        "cuda_event_elapsed_ms": float(start_event.elapsed_time(end_event)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "independent_numpy_parity": "passed",
        "evaluator_metric_names": list(sample.metrics),
        "power_statistics_summarized": False,
    }
    del sample, gpu_class_rows, gpu_grouped_rows, numpy_class_rows, numpy_grouped_rows
    torch.cuda.empty_cache()
    return receipt


def _aggregate_generation_cuda(data: GenerationData, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = torch.as_tensor(data.model, dtype=torch.int64, device=device)
    stream = torch.as_tensor(data.stream, dtype=torch.int64, device=device)
    method = torch.as_tensor(data.method, dtype=torch.int64, device=device)
    label = torch.as_tensor(data.label, dtype=torch.int64, device=device)
    component = torch.as_tensor(data.component, dtype=torch.int64, device=device)
    ed_correct = torch.as_tensor(data.ed_correct, dtype=torch.float64, device=device)
    ec_correct = torch.as_tensor(data.ec_correct, dtype=torch.float64, device=device)
    metric_tensors = {
        name: torch.as_tensor(values, dtype=torch.float64, device=device)
        for name, values in data.metrics.items()
    }

    class_keys = torch.stack((model, method, label), dim=1)
    class_unique, class_inverse = torch.unique(class_keys, dim=0, sorted=True, return_inverse=True)
    class_count = torch.bincount(class_inverse, minlength=class_unique.shape[0]).to(torch.float64)
    class_sums: dict[str, torch.Tensor] = {
        "E_D_correct": torch.zeros_like(class_count).index_add_(0, class_inverse, ed_correct),
        "E_C_correct": torch.zeros_like(class_count).index_add_(0, class_inverse, ec_correct),
    }
    for name, values in metric_tensors.items():
        class_sums[name] = torch.zeros_like(class_count).index_add_(0, class_inverse, values)
    class_rows: list[dict[str, Any]] = []
    for i, key in enumerate(class_unique.detach().cpu().tolist()):
        model_id, method_id, label_id = key
        n = int(class_count[i].item())
        row: dict[str, Any] = {
            "model": ENDPOINT_NAMES[model_id],
            "method": "G1" if method_id == 0 else "G4",
            "requested_label": label_id,
            "sample_count": n,
            "E_D_correct_count": int(class_sums["E_D_correct"][i].item()),
            "E_D_recall": float(class_sums["E_D_correct"][i].item() / n),
            "E_C_correct_count": int(class_sums["E_C_correct"][i].item()),
            "E_C_recall": float(class_sums["E_C_correct"][i].item() / n),
        }
        row.update({name: float(class_sums[name][i].item() / n) for name in metric_tensors})
        class_rows.append(row)

    grouped_keys = torch.stack((model, method, stream, label, component), dim=1)
    unique, inverse = torch.unique(grouped_keys, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique.shape[0]).to(torch.float64)
    sums: dict[str, torch.Tensor] = {
        "E_D_correct": torch.zeros_like(counts).index_add_(0, inverse, ed_correct),
        "E_C_correct": torch.zeros_like(counts).index_add_(0, inverse, ec_correct),
    }
    for name, values in metric_tensors.items():
        sums[name] = torch.zeros_like(counts).index_add_(0, inverse, values)
    grouped_rows: list[dict[str, Any]] = []
    for i, key in enumerate(unique.detach().cpu().tolist()):
        model_id, method_id, stream_id, label_id, component_id = key
        n = int(counts[i].item())
        row = {
            "model": ENDPOINT_NAMES[model_id],
            "method": "G1" if method_id == 0 else "G4",
            "stream": stream_id,
            "requested_label": label_id,
            "component_id": component_id,
            "sample_count": n,
            "E_D_correct_count": int(sums["E_D_correct"][i].item()),
            "E_D_recall": float(sums["E_D_correct"][i].item() / n),
            "E_C_correct_count": int(sums["E_C_correct"][i].item()),
            "E_C_recall": float(sums["E_C_correct"][i].item() / n),
        }
        row.update({name: float(sums[name][i].item() / n) for name in metric_tensors})
        grouped_rows.append(row)

    del model, stream, method, label, component, ed_correct, ec_correct, metric_tensors
    torch.cuda.synchronize(device)
    return class_rows, grouped_rows


def _historical_total_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    metric_names = (
        "E_D_requested_margin", "E_C_requested_margin", "E_D_top_margin", "E_C_top_margin",
        "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction",
    )
    for model in ENDPOINT_NAMES:
        for method in ("G1", "G4"):
            rows = [row for row in class_rows if row["model"] == model and row["method"] == method]
            _require(len(rows) == N_CLASSES, f"historical total is missing a class for {model} {method}")
            total = {
                "model": model,
                "method": method,
                "population": sum(int(row["sample_count"]) for row in rows),
                "samples_per_class": min(int(row["sample_count"]) for row in rows),
                "E_D_correct_count": sum(int(row["E_D_correct_count"]) for row in rows),
                "E_D_macro_recall": float(np.mean([row["E_D_recall"] for row in rows])),
                "E_C_correct_count": sum(int(row["E_C_correct_count"]) for row in rows),
                "E_C_macro_recall": float(np.mean([row["E_C_recall"] for row in rows])),
            }
            for name in metric_names:
                total[f"macro_{name}"] = float(np.mean([row[name] for row in rows]))
            result.append(total)
    return result


def _verify_historical_standardized_nmi(
    endpoint: str,
    class_counts: list[int],
    standardized_nmi: list[float],
    historical_report: dict[str, Any],
) -> dict[str, Any]:
    historical_endpoint = historical_report["endpoints"].get(endpoint)
    _require(isinstance(historical_endpoint, dict), f"historical report has no endpoint {endpoint}")
    historical_classes = historical_endpoint["generation"]["cluster_pivot_association"]["per_class"]
    _require(len(historical_classes) == N_CLASSES,
             f"historical standardized-pivot NMI is incomplete for {endpoint}")
    historical_by_label = {int(row["label"]): row for row in historical_classes}
    differences: list[float] = []
    for label in range(N_CLASSES):
        _require(label in historical_by_label,
                 f"historical standardized-pivot NMI lacks label {label} for {endpoint}")
        prior = historical_by_label[label]
        _require(int(prior["train_rows"]) == int(class_counts[label]),
                 f"historical NMI denominator does not match the TRAIN cache for {endpoint} class {label}")
        difference = abs(float(standardized_nmi[label]) - float(prior["cluster_pivot_nmi"]))
        _require(difference <= NMI_TOLERANCE,
                 f"standardized-pivot NMI does not reproduce history for {endpoint} class {label}")
        differences.append(difference)
    return {
        "passed": True,
        "comparison_count": len(differences),
        "max_absolute_difference": max(differences),
        "tolerance": NMI_TOLERANCE,
    }


def _validate_historical_totals(
    totals: list[dict[str, Any]],
    historical_report: dict[str, Any],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for row in totals:
        prior = historical_report["endpoints"][row["model"]]["generation"][row["method"]]["three_stream_equal_weight"]
        pairs = (
            ("E_D_macro_recall", "E_D_macro_recall"),
            ("E_C_macro_recall", "E_C_macro_recall"),
            ("macro_proxy_clipping_area_fraction", "macro_proxy_clipping_area_fraction"),
            ("macro_proxy_clipped_excess_fraction", "macro_proxy_clipped_excess_fraction"),
        )
        for measured_name, stored_name in pairs:
            measured = float(row[measured_name])
            stored = float(prior[stored_name])
            difference = abs(measured - stored)
            _require(difference <= 1e-10,
                     f"saved generation total {measured_name} does not reproduce history for "
                     f"{row['model']} {row['method']}: {difference:.3g}")
            comparisons.append({"model": row["model"], "method": row["method"],
                                "measured_field": measured_name, "stored_field": stored_name,
                                "absolute_difference": difference})
    return {"passed": True, "comparison_count": len(comparisons),
            "max_absolute_difference": max(item["absolute_difference"] for item in comparisons),
            "comparisons": comparisons}


def _write_generation_outputs(
    output_dir: Path,
    generation_path: Path,
    historical_report_path: Path,
    device: torch.device,
    generation: GenerationData,
) -> dict[str, Any]:
    historical_report = json.loads(historical_report_path.read_text(encoding="utf-8"))
    _require(historical_report.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1",
             "historical report is not the explicit three-endpoint replication report")
    _require(set(historical_report.get("endpoints", {})) == set(ENDPOINT_NAMES),
             "historical report has missing or unexpected endpoints")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    class_rows, grouped_rows = _aggregate_generation_cuda(generation, device)
    event_end.record()
    torch.cuda.synchronize(device)
    cuda_observation = {
        "rows_processed": EXPECTED_GENERATION_ROWS,
        "cuda_event_elapsed_ms": float(event_start.elapsed_time(event_end)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device": str(device),
    }
    totals = _historical_total_rows(class_rows)
    history_check = _validate_historical_totals(totals, historical_report)

    totals_fields = tuple(totals[0].keys())
    class_fields = tuple(class_rows[0].keys())
    grouped_fields = tuple(grouped_rows[0].keys())
    _write_csv(output_dir / "generation_historical_totals.csv", totals_fields, totals)
    _write_csv(output_dir / "generation_by_class.csv", class_fields, class_rows)
    _write_csv(output_dir / "generation_by_stream_class_component.csv", grouped_fields, grouped_rows)

    _verify_generation_output_consumer(
        output_dir / "generation_historical_totals.csv",
        totals,
        historical_report,
    )
    _verify_class_generation_output(output_dir / "generation_by_class.csv", class_rows)
    _verify_grouped_generation_output(
        output_dir / "generation_by_stream_class_component.csv",
        grouped_rows,
    )
    receipt = {
        "input_path": str(generation_path),
        "historical_report_path": str(historical_report_path),
        "input_records": EXPECTED_GENERATION_ROWS,
        "model_method_totals": len(totals),
        "class_rows": len(class_rows),
        "stream_class_component_rows": len(grouped_rows),
        "group_denominators_complete": True,
        "gpu_device": torch.cuda.get_device_name(device),
        "aggregation_device": str(device),
        "cuda_observation": cuda_observation,
        "cuda_execution_confirmed": True,
        "historical_reproduction": history_check,
        "readback_consumer_passed": True,
    }
    _write_json(output_dir / "generation_validation_receipt.json", receipt)
    del generation
    torch.cuda.empty_cache()
    return receipt


def _verify_generation_output_consumer(
    path: Path,
    expected_rows: list[dict[str, Any]],
    historical_report: dict[str, Any],
) -> None:
    _require(path.is_file(), "generation historical totals are missing")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(expected_rows[0]),
                 "generation totals have missing, reordered, or renamed metric columns")
        seen = 0
        for row in reader:
            _require(seen < len(expected_rows), "generation totals contain extra rows")
            expected = expected_rows[seen]
            _require(row["model"] == expected["model"] and row["method"] == expected["method"],
                     "generation totals contain a wrong endpoint or sampling method")
            _require(int(row["population"]) == expected["population"],
                     "generation totals contain a wrong row denominator")
            _require(int(row["E_D_correct_count"]) == expected["E_D_correct_count"],
                     "generation totals contain a wrong E_D count")
            _require(int(row["E_C_correct_count"]) == expected["E_C_correct_count"],
                     "generation totals contain a wrong E_C count")
            for metric in ("E_D_macro_recall", "E_C_macro_recall"):
                _require(math.isclose(float(row[metric]), expected[metric], rel_tol=0, abs_tol=1e-12),
                         f"generation totals contain a wrong {metric}")
            prior = historical_report["endpoints"][row["model"]]["generation"][row["method"]]["three_stream_equal_weight"]
            for metric, stored in (("E_D_macro_recall", "E_D_macro_recall"),
                                   ("E_C_macro_recall", "E_C_macro_recall")):
                _require(abs(float(row[metric]) - float(prior[stored])) <= 1e-10,
                         f"generation total metric {metric} is mixed with another evaluator")
            seen += 1
        _require(seen == len(expected_rows), "generation totals are missing rows")


def _verify_class_generation_output(path: Path, expected_rows: list[dict[str, Any]]) -> None:
    _require(path.is_file(), "generation class table is missing")
    identity_fields = ("model", "method", "requested_label")
    count_fields = ("sample_count", "E_D_correct_count", "E_C_correct_count")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(expected_rows[0]),
                 "generation class table has missing, reordered, or renamed evaluator statistics")
        seen = 0
        for row in reader:
            _require(seen < len(expected_rows), "generation class table contains extra rows")
            expected = expected_rows[seen]
            for field in identity_fields:
                _require(row[field] == str(expected[field]),
                         f"generation class table has a wrong {field}")
            for field in count_fields:
                _require(int(row[field]) == int(expected[field]),
                         f"generation class table has a wrong {field}")
            for field, expected_value in expected.items():
                if field in identity_fields or field in count_fields:
                    continue
                _require(math.isclose(float(row[field]), float(expected_value), rel_tol=0, abs_tol=1e-12),
                         f"generation class table has a wrong or mixed statistic {field}")
            seen += 1
        _require(seen == len(expected_rows), "generation class table is missing rows")
        _require(seen == len(ENDPOINT_NAMES) * 2 * N_CLASSES,
                 "generation class table does not cover all endpoint/method/class groups")


def _verify_grouped_generation_output(path: Path, expected_rows: list[dict[str, Any]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(expected_rows[0]),
                 "grouped generation output has missing or renamed statistic columns")
        seen = 0
        total = 0
        for row in reader:
            _require(seen < len(expected_rows), "grouped generation output contains extra groups")
            expected = expected_rows[seen]
            for field in ("model", "method", "stream", "requested_label", "component_id", "sample_count",
                          "E_D_correct_count", "E_C_correct_count"):
                _require(row[field] == str(expected[field]),
                         f"grouped generation output has a wrong {field}")
            total += int(row["sample_count"])
            seen += 1
        _require(seen == len(expected_rows), "grouped generation output is missing a group")
        _require(total == EXPECTED_GENERATION_ROWS,
                 "grouped generation denominators do not recover the 180,000-row input")


def _write_combined_association_outputs(
    output_dir: Path,
    endpoint_associations: list[dict[str, Any]],
    endpoint_contingencies: list[dict[str, Any]],
) -> None:
    association_rows: list[dict[str, Any]] = []
    contingency_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_NAMES:
        endpoint_dir = output_dir / "endpoints" / endpoint
        association_path = endpoint_dir / "g4_pivot_association_by_class.csv"
        contingency_path = endpoint_dir / "g4_component_pivot_contingency.csv"
        with association_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == list(ASSOCIATION_FIELDS),
                     f"association consumer rejected a malformed endpoint table for {endpoint}")
            association_rows.extend(dict(row) for row in reader)
        with contingency_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == list(CONTINGENCY_FIELDS),
                     f"contingency consumer rejected a malformed endpoint table for {endpoint}")
            contingency_rows.extend(dict(row) for row in reader)
    _require(len(association_rows) == 3 * 2 * N_CLASSES,
             "combined association table is missing endpoint/class/reference rows")
    _require(len(contingency_rows) == 3 * 2 * N_CLASSES * N_G4_COMPONENTS * N_CHANNELS,
             "combined contingency table is missing endpoint/class/component/channel rows")
    _write_csv(output_dir / "g4_pivot_association_by_endpoint.csv", ASSOCIATION_FIELDS, association_rows)
    _write_csv(output_dir / "g4_component_pivot_contingency.csv", CONTINGENCY_FIELDS, contingency_rows)
    _require(len(endpoint_associations) == len(association_rows),
             "combined association count differs from endpoint receipts")
    _require(len(endpoint_contingencies) == len(contingency_rows),
             "combined contingency count differs from endpoint receipts")


def _write_report(
    output_dir: Path,
    preflight: dict[str, Any],
    endpoint_receipts: list[dict[str, Any]],
    generation_receipt: dict[str, Any],
    source_rows: int,
) -> None:
    mismatch = {item["endpoint"]: item["pivot_mismatch_fraction"] for item in endpoint_receipts}
    mismatch_rows = {item["endpoint"]: int(item["pivot_mismatch_rows"]) for item in endpoint_receipts}
    mean_actual_nmi: dict[str, float] = {}
    mean_standardized_nmi: dict[str, float] = {}
    mean_nmi_delta: dict[str, float] = {}
    for endpoint in ENDPOINT_NAMES:
        path = output_dir / "endpoints" / endpoint / "g4_pivot_association_by_class.csv"
        values: dict[str, list[float]] = {"actual_raw_phase_reference": [], "standardized_feature_max": []}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                values[row["reference"]].append(float(row["nmi"]))
        _require(all(len(rows) == N_CLASSES for rows in values.values()),
                 f"report consumer found incomplete NMI classes for {endpoint}")
        mean_actual_nmi[endpoint] = float(np.mean(values["actual_raw_phase_reference"]))
        mean_standardized_nmi[endpoint] = float(np.mean(values["standardized_feature_max"]))
        mean_nmi_delta[endpoint] = mean_actual_nmi[endpoint] - mean_standardized_nmi[endpoint]
    with (output_dir / "generation_historical_totals.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        generation_totals = {
            (row["model"], row["method"]): row
            for row in csv.DictReader(stream)
        }
    saved_generation_g4_minus_g1: dict[str, dict[str, float]] = {}
    for endpoint in ENDPOINT_NAMES:
        g1 = generation_totals[(endpoint, "G1")]
        g4 = generation_totals[(endpoint, "G4")]
        saved_generation_g4_minus_g1[endpoint] = {
            "E_D_macro_recall": float(g4["E_D_macro_recall"]) - float(g1["E_D_macro_recall"]),
            "E_C_macro_recall": float(g4["E_C_macro_recall"]) - float(g1["E_C_macro_recall"]),
            "proxy_clipping_area_fraction": (
                float(g4["macro_proxy_clipping_area_fraction"])
                - float(g1["macro_proxy_clipping_area_fraction"])
            ),
            "proxy_clipped_excess_fraction": (
                float(g4["macro_proxy_clipped_excess_fraction"])
                - float(g1["macro_proxy_clipped_excess_fraction"])
            ),
        }
    report = {
        "task_id": TASK_ID,
        "status": "COMPLETE_DIAGNOSTIC",
        "scientific_status": "VALID_FAILED",
        "question": "Does G4 association with the actual phase-reference channel persist after restoring scale?",
        "inputs": [item["source_identity"] for item in preflight["endpoints"]],
        "saved_train_feature_rows": source_rows,
        "saved_generation_metric_rows": EXPECTED_GENERATION_ROWS,
        "phase_reference_mismatch_fraction_by_endpoint": mismatch,
        "phase_reference_mismatch_rows_by_endpoint": mismatch_rows,
        "phase_reference_mismatch_rows_total": sum(mismatch_rows.values()),
        "macro_mean_class_nmi_actual_reference_by_endpoint": mean_actual_nmi,
        "macro_mean_class_nmi_standardized_max_by_endpoint": mean_standardized_nmi,
        "macro_mean_class_nmi_delta_by_endpoint": mean_nmi_delta,
        "nmi_numerical_comparison_tolerance": NMI_TOLERANCE,
        "saved_generation_g4_minus_g1": saved_generation_g4_minus_g1,
        "generation_history_reproduced": generation_receipt["historical_reproduction"]["passed"],
        "decision_boundary": "All associations are descriptive; component IDs are endpoint-local and do not identify the same component across trajectories.",
        "unmeasured": [
            "physical source reachability of the saved complex coefficients",
            "the real hardware phase-error distribution",
            "human readability for the J endpoint",
            "physical propagation and deployment",
        ],
        "new_training_updates": 0,
        "optimizer_instances_created": 0,
        "prior_refits": 0,
        "new_sampling": 0,
        "test_reads": 0,
        "human_reviews": 0,
        "lpips_evaluations": 0,
        "next_action_authorization": "NONE",
    }
    _write_json(output_dir / "report.json", report)
    deltas = list(mean_nmi_delta.values())
    mismatch_rows_total = sum(mismatch_rows.values())
    if mismatch_rows_total:
        mismatch_statement = (
            f"逐行核验显示，恢复原scale得到的实际相位参考通道与标准化特征最大通道在"
            f"{mismatch_rows_total:,}/{source_rows:,}条TRAIN记录上不同；表中列出各端点比例。"
            "历史NMI复现的是标准化最大通道口径，本轮另列恢复后的实际参考通道口径。"
        )
    else:
        mismatch_statement = (
            f"逐行核验显示，三端点{source_rows:,}条TRAIN记录的实际相位参考通道均与"
            "标准化特征最大通道一致；本批数据上两种口径没有逐行差异。"
            "历史NMI仍按标准化最大通道口径复现，本轮同时报告实际参考口径。"
        )
    if all(value > NMI_TOLERANCE for value in deltas):
        scientific_reading = (
            "三条轨迹的实际参考口径宏类平均NMI均高于标准化最大通道口径。"
            "这是训练特征上的描述性关联比较，不提供显著性或因果证据，也不能解释保存生成样本的成败。"
        )
        next_recommendation = "保留相位表示作为后续对照设计候选；本报告不授权拟合先验、训练或物理扰动实验。"
    elif all(value < -NMI_TOLERANCE for value in deltas):
        scientific_reading = (
            "三条轨迹的实际参考口径宏类平均NMI均低于标准化最大通道口径；"
            "该描述性结果降低相位表示解释的优先级，不构成显著性或因果证据。"
        )
        next_recommendation = "降低相位表示解释的优先级；本报告不授权拟合先验、训练或物理扰动实验。"
    else:
        scientific_reading = (
            "三条轨迹的实际参考口径宏类平均NMI变化方向混合或接近相同；"
            "结合逐类结果报告为未分辨，不用单一轨迹排序。"
        )
        next_recommendation = "暂不提高或降低相位表示解释的优先级；先按逐类关联结果另行设计对照问题。"
    comparison_rows = [
        "| 端点 | 实际参考平均NMI | 标准化最大平均NMI | 差值 | 通道不一致比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINT_NAMES:
        comparison_rows.append(
            f"| {endpoint} | {mean_actual_nmi[endpoint]:.6f} | "
            f"{mean_standardized_nmi[endpoint]:.6f} | {mean_nmi_delta[endpoint]:+.6f} | "
            f"{mismatch[endpoint]:.2%} |"
        )
    generation_comparison_rows = [
        "| 端点 | E_D召回差 | E_C召回差 | 截断面积比例差 | 截断超额比例差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINT_NAMES:
        values = saved_generation_g4_minus_g1[endpoint]
        generation_comparison_rows.append(
            f"| {endpoint} | {values['E_D_macro_recall']:+.4f} | "
            f"{values['E_C_macro_recall']:+.4f} | "
            f"{values['proxy_clipping_area_fraction']:+.4f} | "
            f"{values['proxy_clipped_excess_fraction']:+.4f} |"
        )
    markdown = "\n".join([
        "# G4相位参考与统计表示审计",
        "",
        "## 结果与状态",
        "测量完成，状态为 `COMPLETE_DIAGNOSTIC`；系统科学状态仍为 `VALID_FAILED`。",
        f"核验{source_rows:,}条TRAIN特征和{EXPECTED_GENERATION_ROWS:,}条保存生成记录，主要统计在CUDA执行。",
        "",
        "## 科学认识",
        mismatch_statement,
        scientific_reading,
        *comparison_rows,
        "旧统计中的NMI对应标准化特征最大通道。历史G4生成结果已从保存逐图记录还原，并与原报告总表核对。",
        "关联是描述性统计，G4分量编号只在各端点内部有效，不能跨端点按编号配对。历史生成结果保留，本轮只更正统计口径说明。",
        "保存评价记录中的G4−G1差值如下；这是冻结Decoder下的既有代理评价，不构成相位统计导致生成表现的因果证据。",
        *generation_comparison_rows,
        "本轮人工评阅为0，因此不更新既有人眼结论。",
        "",
        "## 目标缺口",
        "本轮没有验证物理输入光路能否产生这些复系数，也没有测得实际硬件相位误差分布。",
        "本轮不评价J的人眼可读性、物理传播或模型采用。",
        "",
        "## 工程完成与验证",
        (
            f"CUDA代表性预检覆盖每端点每类2条TRAIN记录，并对完整解析的生成表按端点/方法/流/类别"
            f"各取前2条（{preflight['generation_aggregation']['preflight_rows']:,}行）完成GPU与NumPy聚合对照；"
            f"三端点各处理{EXPECTED_ROWS:,}条特征。"
        ),
        "GPU与独立NumPy逐行索引、整数列联表一致，NMI绝对差不超过1e-10；输出消费者回读并核验行、标签、端点、统计名和分母。",
        f"保存生成记录历史总表复核：{generation_receipt['historical_reproduction']['comparison_count']}项对照通过。",
        "逐端点及生成表聚合的CUDA事件、显存峰值与数据量见`cuda_execution_receipt.json`和`generation_validation_receipt.json`；每分钟资源采样见`resource_telemetry.csv`。",
        "",
        "## 决策影响与下一步",
        next_recommendation,
        "已授权范围止于本次GPU审计、证据解释和问题排序；后继实验需独立设计与授权。",
        "",
        "详细逐行数据与机器收据见本目录CSV和JSON文件。",
        "",
    ])
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")


def _write_progress(output_dir: Path, stage: str, completed_endpoints: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "task_id": TASK_ID,
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage,
        "completed_endpoints": completed_endpoints,
        "total_endpoints": len(ENDPOINT_NAMES),
        "updated_unix_seconds": time.time(),
    }
    if extra:
        payload.update(extra)
    _write_json(output_dir / "progress.json", payload)


def _specs_from_args(args: argparse.Namespace) -> tuple[EndpointSpec, ...]:
    return (
        EndpointSpec("C_seed_2026101501", 2026101501, args.c_checkpoint, args.c_cache, args.c_priors),
        EndpointSpec("seed_2026102151", 2026102151,
                     args.seed_2151_checkpoint, args.seed_2151_cache, args.seed_2151_priors),
        EndpointSpec("seed_2026102152", 2026102152,
                     args.seed_2152_checkpoint, args.seed_2152_cache, args.seed_2152_priors),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-checkpoint", type=Path, required=True)
    parser.add_argument("--c-cache", type=Path, required=True)
    parser.add_argument("--c-priors", type=Path, required=True)
    parser.add_argument("--seed-2151-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-2151-cache", type=Path, required=True)
    parser.add_argument("--seed-2151-priors", type=Path, required=True)
    parser.add_argument("--seed-2152-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-2152-cache", type=Path, required=True)
    parser.add_argument("--seed-2152-priors", type=Path, required=True)
    parser.add_argument("--generation-csv", type=Path, required=True)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser.parse_args()


def _run_audit(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    specs = _specs_from_args(args)
    source_paths = [path for spec in specs for path in (spec.checkpoint, spec.cache, spec.priors)]
    source_paths.extend((args.generation_csv, args.historical_report))
    _validate_file_arguments(source_paths, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(torch.cuda.is_available(), "CUDA is unavailable; formal audit does not fall back to CPU")
    _require(0 <= args.cuda_device < torch.cuda.device_count(),
             f"requested CUDA device {args.cuda_device} is not available")
    device = torch.device(f"cuda:{args.cuda_device}")
    start_time = time.perf_counter()
    _write_progress(output_dir, "input_validation_and_gpu_preflight", [])
    generation = _load_generation_csv(args.generation_csv)
    preflight = _run_preflight(specs, output_dir, device, args.historical_report, generation)
    generation_preflight = preflight["generation_aggregation"]
    expected_generation_preflight_rows = len(ENDPOINT_NAMES) * 2 * N_STREAMS * N_CLASSES * 2
    _require(preflight.get("status") == "PASSED",
             "the representative CUDA preflight did not produce a passed receipt")
    _require(generation_preflight.get("status") == "PASSED"
             and generation_preflight.get("preflight_rows") == expected_generation_preflight_rows
             and generation_preflight.get("independent_numpy_parity") == "passed",
             "the saved-generation CUDA preflight receipt is incomplete")
    historical_report = json.loads(args.historical_report.read_text(encoding="utf-8"))

    endpoint_receipts: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    contingencies: list[dict[str, Any]] = []
    cuda_endpoint_receipts: list[dict[str, Any]] = []
    resumed_endpoints: list[str] = []
    completed: list[str] = []
    _write_progress(output_dir, "full_endpoint_measurements", completed)

    for spec in specs:
        _write_progress(output_dir, "full_endpoint_measurements", completed,
                        {"current_endpoint": spec.name})
        data = _load_endpoint(spec)
        endpoint_dir = output_dir / "endpoints" / spec.name
        endpoint_receipt_path = endpoint_dir / "endpoint_receipt.json"
        if endpoint_receipt_path.is_file():
            existing_receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
            _require(existing_receipt.get("task_id") == TASK_ID
                     and existing_receipt.get("endpoint") == spec.name,
                     f"endpoint output path belongs to another task: {spec.name}")
            if existing_receipt.get("cuda_execution_confirmed") is True:
                receipt, assoc_rows, contingency_rows = _verify_completed_endpoint(output_dir, data)
                history_check = receipt.get("historical_standardized_nmi")
                _require(isinstance(history_check, dict) and history_check.get("passed") is True,
                         f"completed endpoint lacks historical NMI reproduction evidence: {spec.name}")
                endpoint_receipts.append(receipt)
                associations.extend(assoc_rows)
                contingencies.extend(contingency_rows)
                completed.append(spec.name)
                resumed_endpoints.append(spec.name)
                cuda_observation = receipt["cuda_observation"]
                cuda_endpoint_receipts.append({
                    "endpoint": spec.name,
                    **cuda_observation,
                    "input_feature_rows": EXPECTED_ROWS,
                    "dtype": "float64/complex128",
                    "cuda_execution_confirmed": True,
                    "reused_verified_endpoint": True,
                })
                _write_progress(output_dir, "full_endpoint_measurements", completed,
                                {"resumed_endpoint_without_cuda_recompute": spec.name})
                del data
                continue
            _require(existing_receipt.get("status") == "COMPLETE_AND_READBACK_VERIFIED",
                     f"endpoint receipt has an unrecognized incomplete state: {spec.name}")

        indices = torch.arange(EXPECTED_ROWS, dtype=torch.int64)
        gpu_metrics, gpu_association, cuda_observation = _run_cuda_metrics(data, indices, device)
        numpy_expected = _numpy_reference(data)
        parity = _compare_gpu_numpy(gpu_metrics, gpu_association, numpy_expected)
        history_check = _verify_historical_standardized_nmi(
            spec.name,
            data.class_counts,
            gpu_association["standardized_feature_max"]["nmi_by_class"].detach().cpu().tolist(),
            historical_report,
        )
        endpoint_receipt = _write_endpoint_outputs(
            output_dir,
            data,
            gpu_metrics,
            gpu_association,
            numpy_expected,
        )
        endpoint_receipt["cuda_observation"] = cuda_observation
        endpoint_receipt["numpy_parity"] = parity
        endpoint_receipt["historical_standardized_nmi"] = history_check
        endpoint_receipt["cuda_execution_confirmed"] = True
        _write_json(endpoint_dir / "endpoint_receipt.json", endpoint_receipt)
        assoc_rows, contingency_rows = _verify_association_outputs(
            endpoint_dir / "g4_pivot_association_by_class.csv",
            endpoint_dir / "g4_component_pivot_contingency.csv",
            spec.name,
            numpy_expected,
        )
        endpoint_receipts.append(endpoint_receipt)
        associations.extend(assoc_rows)
        contingencies.extend(contingency_rows)
        cuda_endpoint_receipts.append({
            "endpoint": spec.name,
            **cuda_observation,
            "input_feature_rows": EXPECTED_ROWS,
            "dtype": "float64/complex128",
            "cuda_execution_confirmed": True,
            "reused_verified_endpoint": False,
        })
        completed.append(spec.name)
        _write_json(output_dir / "cuda_execution_receipt.json", {
            "task_id": TASK_ID,
            "device_name": torch.cuda.get_device_name(device),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "full_population_endpoints_completed": cuda_endpoint_receipts,
            "full_population_rows_completed": sum(item["input_feature_rows"]
                                                   for item in cuda_endpoint_receipts),
            "elapsed_wall_seconds_observed": time.perf_counter() - start_time,
        })
        _write_progress(output_dir, "full_endpoint_measurements", completed,
                        {"last_cuda_endpoint": spec.name, "rows_processed": EXPECTED_ROWS})
        del data, gpu_metrics, gpu_association, numpy_expected
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    _require(set(completed) == set(ENDPOINT_NAMES), "not all endpoint row tables passed readback")
    _write_combined_association_outputs(output_dir, associations, contingencies)
    _write_progress(output_dir, "saved_generation_aggregation", completed)
    generation_receipt = _write_generation_outputs(
        output_dir,
        args.generation_csv,
        args.historical_report,
        device,
        generation,
    )
    del generation
    _write_json(output_dir / "cuda_execution_receipt.json", {
        "task_id": TASK_ID,
        "device_name": torch.cuda.get_device_name(device),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "full_population_endpoints_completed": cuda_endpoint_receipts,
        "full_population_rows_completed": sum(item["input_feature_rows"]
                                               for item in cuda_endpoint_receipts),
        "generation_aggregation": generation_receipt["cuda_observation"],
        "elapsed_wall_seconds_observed": time.perf_counter() - start_time,
    })
    _require(sum(int(item["feature_rows"]) for item in endpoint_receipts) == 3 * EXPECTED_ROWS,
             "final feature row count does not recover 165,000 rows")
    cuda_rows = sum(int(item["input_feature_rows"]) for item in cuda_endpoint_receipts)
    _require(cuda_rows == 3 * EXPECTED_ROWS and len(cuda_endpoint_receipts) == 3,
             "CUDA execution evidence does not cover all 165,000 endpoint feature rows")
    _require(all(item.get("cuda_execution_confirmed") is True for item in cuda_endpoint_receipts),
             "a full-population endpoint lacks confirmed CUDA execution")
    _require(generation_receipt["input_records"] == EXPECTED_GENERATION_ROWS,
             "final generation row count does not recover 180,000 records")
    _require(generation_receipt.get("cuda_execution_confirmed") is True
             and generation_receipt["cuda_observation"]["rows_processed"] == EXPECTED_GENERATION_ROWS,
             "generation aggregation consumer lacks full-population CUDA execution evidence")

    validation = {
        "task_id": TASK_ID,
        "status": "COMPLETE_DIAGNOSTIC",
        "preflight_status": preflight["status"],
        "endpoint_count": len(endpoint_receipts),
        "feature_rows_total": sum(int(item["feature_rows"]) for item in endpoint_receipts),
        "expected_feature_rows_total": 3 * EXPECTED_ROWS,
        "generation_rows_total": generation_receipt["input_records"],
        "expected_generation_rows_total": EXPECTED_GENERATION_ROWS,
        "generation_preflight_rows": generation_preflight["preflight_rows"],
        "generation_preflight_cuda_numpy_parity": generation_preflight["independent_numpy_parity"],
        "cuda_full_population_rows": sum(int(item["input_feature_rows"])
                                          for item in cuda_endpoint_receipts),
        "resumed_cuda_endpoints": resumed_endpoints,
        "gpu_numpy_index_table_nmi_checks": "passed",
        "generation_historical_summary_check": generation_receipt["historical_reproduction"]["passed"],
        "generation_group_denominator_check": generation_receipt["group_denominators_complete"],
        "generation_cuda_execution_confirmed": generation_receipt["cuda_execution_confirmed"],
        "output_consumer_readback": "passed",
        "new_training_updates": 0,
        "optimizer_instances_created": 0,
        "prior_refits": 0,
        "new_sampling": 0,
        "test_reads": 0,
        "human_reviews": 0,
        "lpips_evaluations": 0,
        "scientific_status_changed": False,
        "elapsed_wall_seconds_observed": time.perf_counter() - start_time,
    }
    _write_json(output_dir / "validation_receipt.json", validation)
    _write_report(output_dir, preflight, endpoint_receipts, generation_receipt,
                  sum(int(item["feature_rows"]) for item in endpoint_receipts))
    remote_receipt = {
        "task_id": TASK_ID,
        "status": "REMOTE_MEASUREMENT_COMPLETE_AWAITING_TRANSFER",
        "output_dir": str(output_dir),
        "required_artifacts": [
            "report.md", "report.json", "preflight_receipt.json", "validation_receipt.json",
            "cuda_execution_receipt.json", "generation_validation_receipt.json",
            "g4_pivot_association_by_endpoint.csv", "g4_component_pivot_contingency.csv",
            "generation_historical_totals.csv", "generation_by_class.csv",
            "generation_by_stream_class_component.csv",
            "resource_telemetry.csv", "telemetry_copy_status.json",
            "endpoints/*/phase_reference_rows.csv", "endpoints/*/endpoint_receipt.json",
        ],
        "source_remote_task_complete": True,
        "local_transfer_verified": False,
    }
    _write_json(output_dir / "remote_completion_receipt.json", remote_receipt)
    _write_progress(output_dir, "complete", completed,
                    {"elapsed_wall_seconds_observed": time.perf_counter() - start_time})
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return _run_audit(args)
    except Exception as error:
        try:
            output_dir = args.output_dir
            if not (output_dir / "remote_completion_receipt.json").is_file():
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(output_dir / "engineering_error.json", {
                    "task_id": TASK_ID,
                    "status": "ENGINEERING_ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "scientific_result": None,
                })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
