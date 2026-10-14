"""Freeze and replay the saved 400x400 LightGen proxy for validity auditing."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    PixelMLPConfirmationEvaluatorV1,
    ThirdConfirmationEvaluator,
)
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_json,
    atomic_write_new_json,
)
from ldm_project.v2.optics.lightgen import LightGenAmplitudeFiberOLSSystemV2
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    stratified_mnist_indices,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "M0_EXISTING_400_FROZEN_REPLAY_VALIDITY_V1"
REPORT_SCHEMA = "m0_existing_400_frozen_replay_report.v1"
MANIFEST_SCHEMA = "m0_existing_400_frozen_replay_manifest.v1"
DEFAULT_ARTIFACT_DIR = Path(
    "outputs/20260928_lightgen_400_mnist_semantic_generation_baseline_v1/runs/autodl_attempt_01"
)
DEFAULT_CONFIG = Path("configs/v2/diagnostic/lightgen_amplitude_fiber_ols_400.yaml")
DEFAULT_POPULATION = Path(
    "outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01/input_population.json"
)
DEFAULT_TRAINING_EVALUATOR = Path(
    "outputs/20260724_v2/runs/20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/frozen_mnist_evaluator.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20261002_m0_existing_400_frozen_replay_validity_v1/runs/attempt_01"
)
DATA_PATH = Path("data/MNIST/processed/training.pt")
EVALUATOR_KEYS = ("e_s", "e_i", "e_c", "e_d")
CHUNK_ROWS = 100
BATCH_SIZE = 10
RTOL = 1.0e-5
ATOL = 1.0e-6
DEVICE_MEMORY_FRACTION_MAX = 0.90
MIN_OUTPUT_MARGIN_BYTES = int(1.5 * 1024**3)


class ReplayError(RuntimeError):
    """An invalid input, measurement, or frozen-state invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _load_torch(path: Path) -> Any:
    _require(path.is_file(), f"required input is missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def evaluator_paths(population: Mapping[str, Any]) -> dict[str, Path]:
    inputs = _mapping(population.get("input_paths"), "population input_paths")
    result: dict[str, Path] = {}
    for key in EVALUATOR_KEYS:
        raw = inputs.get(key)
        _require(isinstance(raw, str) and raw, f"population does not name {key}")
        result[key] = _resolve(ROOT, raw)
    return result


def validate_system_config(config: Any) -> dict[str, Any]:
    system = config.system
    expected = {
        "kind": "diagnostic_lightgen_amplitude_fiber_ols_400",
        "slm_shape": (400, 400),
        "metasurface_shape": (400, 400),
        "slm_pixel_pitch_m": 3.0e-6,
        "metasurface_pixel_pitch_m": 3.0e-6,
        "propagation_grid_size": 512,
        "encoder_layers": 2,
        "decoder_layers": 4,
        "fiber_grid_shape": (10, 10),
        "proxy_wavelength_m": 532.0e-9,
        "proxy_layer_spacing_m": 1.0e-3,
        "ols_kind": "gaussian_single_mode_fiber",
        "boundary": "finite_crop",
    }
    observed = {name: getattr(system, name) for name in expected}
    _require(
        all(observed[name] == value for name, value in expected.items()),
        f"400 proxy configuration differs from its declared optical path: {observed}",
    )
    return {
        **{key: list(value) if isinstance(value, tuple) else value for key, value in observed.items()},
        "fiber_mode_waist_m": float(system.fiber_mode_waist_m),
        "physical_parameter_status": str(system.physical_parameter_status),
        "runtime": "2-layer encoder -> 100 complex OLS -> 4-layer generator",
        "measurement_grid": "512x512 propagation; 400x400 active planes",
    }


def validate_generated_payloads(
    ols_payload: Mapping[str, Any], visuals_payload: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(
        ols_payload.get("schema") == "lightgen_400_mnist_generated_ols.v1",
        "generated_ols schema mismatch",
    )
    _require(
        visuals_payload.get("schema") == "lightgen_400_mnist_generated_visuals.v1",
        "generated_visuals schema mismatch",
    )
    coefficients = ols_payload.get("coefficients")
    labels = ols_payload.get("requested_labels")
    views = visuals_payload.get("evaluator_views_28x28")
    raw_examples = visuals_payload.get("raw_cmos_examples_400x400")
    visual_labels = visuals_payload.get("requested_labels")
    _require(
        torch.is_tensor(coefficients)
        and coefficients.shape == (2000, 100)
        and torch.is_complex(coefficients)
        and bool(torch.isfinite(coefficients).all()),
        "generated OLS must be finite complex [2000,100]",
    )
    expected_labels = torch.arange(10).repeat_interleave(200)
    _require(
        torch.is_tensor(labels)
        and labels.dtype == torch.long
        and torch.equal(labels, expected_labels),
        "generated OLS labels are not the original 200-per-class order",
    )
    _require(
        torch.is_tensor(views)
        and views.shape == (2000, 1, 28, 28)
        and views.dtype.is_floating_point
        and bool(torch.isfinite(views).all()),
        "historical views must be finite float [2000,1,28,28]",
    )
    _require(
        torch.is_tensor(raw_examples)
        and raw_examples.shape == (50, 1, 400, 400)
        and raw_examples.dtype.is_floating_point
        and bool(torch.isfinite(raw_examples).all()),
        "historical raw examples must be finite float [50,1,400,400]",
    )
    _require(
        torch.is_tensor(visual_labels)
        and visual_labels.dtype == torch.long
        and torch.equal(visual_labels, expected_labels),
        "historical visuals labels differ from OLS labels",
    )
    _require(
        visuals_payload.get("examples_per_class") == 5,
        "historical raw examples are not five per class",
    )
    return coefficients, labels, views, raw_examples


def raw_400_to_view(raw: torch.Tensor) -> torch.Tensor:
    _require(
        raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 400, 400),
        "raw 400 CMOS input geometry changed",
    )
    _require(bool(torch.isfinite(raw).all()), "raw CMOS contains nonfinite pixels")
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def historical_raw_indices() -> torch.Tensor:
    return torch.tensor(
        [200 * digit + offset for digit in range(10) for offset in range(5)],
        dtype=torch.long,
    )


def compare_tensors(
    measured: torch.Tensor, historical: torch.Tensor, *, rtol: float = RTOL, atol: float = ATOL
) -> dict[str, Any]:
    _require(measured.shape == historical.shape, "historical comparison shapes differ")
    delta = (measured.to(torch.float64) - historical.to(torch.float64)).abs()
    per_row = delta.reshape(delta.shape[0], -1)
    row_max = per_row.max(dim=1).values
    row_rmse = per_row.square().mean(dim=1).sqrt()
    row_match = torch.isclose(measured, historical, rtol=rtol, atol=atol).reshape(
        measured.shape[0], -1
    ).all(dim=1)
    return {
        "matches": bool(row_match.all()),
        "matching_rows": int(row_match.sum()),
        "row_count": int(row_match.numel()),
        "mismatch_indices": torch.nonzero(~row_match).flatten().tolist(),
        "row_max_abs_error": row_max.tolist(),
        "row_rmse": row_rmse.tolist(),
        "max_abs_error": float(delta.max()),
        "overall_rmse": float(delta.square().mean().sqrt()),
        "rtol": rtol,
        "atol": atol,
    }


def wilson_interval(correct: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        raise ValueError("Wilson interval requires a positive count")
    p = correct / count
    z2 = z * z
    denominator = 1.0 + z2 / count
    center = (p + z2 / (2.0 * count)) / denominator
    half = z * math.sqrt((p * (1.0 - p) / count) + z2 / (4.0 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def confusion_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    _require(
        predictions.ndim == labels.ndim == 1 and predictions.shape == labels.shape,
        "predictions and labels must be aligned vectors",
    )
    _require(
        bool(((predictions >= 0) & (predictions < 10)).all())
        and bool(((labels >= 0) & (labels < 10)).all()),
        "class indices must be in [0,9]",
    )
    matrix = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    counts = matrix.sum(1)
    _require(bool((counts > 0).all()), "metrics require examples from all ten classes")
    correct = matrix.diag()
    per_class = []
    for digit in range(10):
        n = int(counts[digit])
        k = int(correct[digit])
        per_class.append(
            {
                "class": digit,
                "count": n,
                "correct": k,
                "recall": k / n,
                "wilson_95": wilson_interval(k, n),
                "predicted_class_count": int((predictions == digit).sum()),
            }
        )
    return {
        "count": int(labels.numel()),
        "confusion": matrix.tolist(),
        "per_class": per_class,
        "recall_by_class": [item["recall"] for item in per_class],
        "correct_by_class": [item["correct"] for item in per_class],
        "predicted_class_counts": [item["predicted_class_count"] for item in per_class],
        "macro_recall": sum(item["recall"] for item in per_class) / 10.0,
        "macro_wilson_bounds_descriptive": [
            sum(item["wilson_95"][bound] for item in per_class) / 10.0
            for bound in (0, 1)
        ],
    }


def evaluator_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    logits = output[1] if isinstance(output, (tuple, list)) else output
    _require(
        torch.is_tensor(logits)
        and logits.shape == (images.shape[0], 10)
        and bool(torch.isfinite(logits).all()),
        "evaluator returned invalid logits",
    )
    return logits


def _tree_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in module.state_dict().items()
    }


def _assert_module_frozen(
    module: torch.nn.Module, expected: Mapping[str, torch.Tensor], name: str
) -> None:
    observed = module.state_dict()
    _require(set(observed) == set(expected), f"{name} state keys changed")
    _require(
        all(torch.equal(observed[key].detach().cpu(), expected[key]) for key in expected),
        f"{name} parameters or buffers changed",
    )
    _require(
        all(not parameter.requires_grad and parameter.grad is None for parameter in module.parameters()),
        f"{name} is not read-only frozen",
    )


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return {
        "device": str(device),
        "total_bytes": int(total),
        "free_bytes": int(free),
        "global_used_bytes": int(total - free),
        "global_used_fraction": float((total - free) / total),
        "pytorch_allocated_bytes": allocated,
        "pytorch_reserved_bytes": reserved,
    }


def _nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReplayError(f"nvidia-smi telemetry unavailable: {exc}") from exc
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 5, f"unexpected nvidia-smi telemetry row: {line}")
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        )
    _require(bool(devices), "nvidia-smi reported no CUDA devices")
    return {"devices": devices, "raw": result.stdout.strip()}


def _require_gpu_margin(device: torch.device) -> dict[str, Any]:
    telemetry = _nvidia_smi()
    primary = telemetry["devices"][device.index or 0]
    total = primary["memory_total_mib"]
    used_fraction = primary["memory_used_mib"] / total
    _require(
        used_fraction <= DEVICE_MEMORY_FRACTION_MAX,
        f"whole-device GPU usage is {used_fraction:.3f}, above the 90% safety line",
    )
    telemetry["selected_global_used_fraction"] = used_fraction
    telemetry["pytorch"] = _resource_snapshot(device)
    return telemetry


def _output_disk_check(output_dir: Path) -> dict[str, int]:
    expected_bytes = (
        2000 * 400 * 400 * 4
        + 300 * 400 * 400 * 4
        + 300 * 400 * 400 * 4
        + 2000 * 1 * 28 * 28 * 4
        + 300 * 1 * 28 * 28 * 4
        + 50 * 1 * 400 * 400 * 4
        + 200 * 1024**2
    )
    probe = output_dir
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = expected_bytes + MIN_OUTPUT_MARGIN_BYTES
    _require(free >= required, f"output drive has {free} bytes free; requires {required}")
    return {
        "free_bytes": int(free),
        "estimated_artifacts_bytes": int(expected_bytes),
        "required_margin_bytes": MIN_OUTPUT_MARGIN_BYTES,
    }


def _load_data_rows(
    data_path: Path, population_path: Path, population: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = _load_torch(data_path)
    _require(
        isinstance(payload, (tuple, list)) and len(payload) == 2,
        "MNIST training.pt schema changed",
    )
    pixels, all_labels = payload
    _require(
        torch.is_tensor(pixels)
        and pixels.shape == (60_000, 28, 28)
        and pixels.dtype == torch.uint8,
        "MNIST source image metadata changed",
    )
    _require(
        torch.is_tensor(all_labels)
        and all_labels.shape == (60_000,)
        and all_labels.dtype == torch.long,
        "MNIST source label metadata changed",
    )
    row_ids = torch.as_tensor(population.get("source_row_indices"), dtype=torch.long)
    labels = torch.as_tensor(population.get("labels"), dtype=torch.long)
    _validate_reference_population(row_ids, labels, all_labels)
    selected_images = pixels.index_select(0, row_ids).unsqueeze(1).to(torch.float32).div_(255.0)
    return row_ids, labels, selected_images, all_labels


def _validate_reference_population(
    row_ids: torch.Tensor, labels: torch.Tensor, all_labels: torch.Tensor
) -> None:
    _require(
        row_ids.shape == labels.shape == (300,),
        "fixed reference population must contain 300 rows and labels",
    )
    _require(
        torch.unique(row_ids).numel() == 300
        and int(row_ids.min()) >= 0
        and int(row_ids.max()) < 60_000,
        "reference source rows must be unique valid MNIST rows",
    )
    _require(
        torch.equal(labels, all_labels.index_select(0, row_ids)),
        "reference population labels differ from canonical MNIST labels",
    )
    _require(
        torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 30)),
        "reference population must include 30 rows per class",
    )
    train_rows, _ = stratified_mnist_indices(
        all_labels, seed=1234, validation_per_label=500
    )
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_rows] = True
    _require(
        bool(train_mask.index_select(0, row_ids).all()),
        "reference population contains a noncanonical TRAIN row",
    )


def _load_frozen_evaluators(
    paths: Mapping[str, Path], training_path: Path, device: torch.device
) -> dict[str, torch.nn.Module]:
    definitions: dict[str, tuple[torch.nn.Module, str]] = {
        "E_S": (SmoothMNISTEvaluator(), SmoothMNISTEvaluator.architecture_kind),
        "E_I": (FrozenMNISTEvaluator(), "frozen_mnist_evaluator_v1"),
        "E_C": (ThirdConfirmationEvaluator(), ThirdConfirmationEvaluator.architecture_kind),
        "E_D": (
            PixelMLPConfirmationEvaluatorV1(),
            PixelMLPConfirmationEvaluatorV1.architecture_kind,
        ),
    }
    models: dict[str, torch.nn.Module] = {}
    for name, (model, architecture) in definitions.items():
        checkpoint = _mapping(
            _load_torch(paths[name.lower()]), f"{name} evaluator checkpoint"
        )
        _require(
            checkpoint.get("architecture_kind") == architecture,
            f"{name} evaluator architecture changed",
        )
        state = _mapping(checkpoint.get("state_dict"), f"{name} evaluator state")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
    training_model, _ = load_frozen_evaluator(training_path, device=device)
    training_model.eval()
    for parameter in training_model.parameters():
        parameter.requires_grad_(False)
    models["E_T"] = training_model
    for name, model in models.items():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _require(
            all(parameter.grad is None for parameter in model.parameters()),
            f"{name} has a pre-existing gradient",
        )
    return models


def _expected_positive_control(metrics: Mapping[str, Any]) -> bool:
    recalls = metrics["recall_by_class"]
    return (
        metrics["macro_recall"] >= 0.90
        and recalls[1] >= 0.80
        and recalls[7] >= 0.80
    )


def _run_preflight(
    model: LightGenAmplitudeFiberOLSSystemV2,
    evaluators: Mapping[str, torch.nn.Module],
    generated_ols: torch.Tensor,
    reference_images: torch.Tensor,
    reference_labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model_before = _tree_snapshot(model)
    evaluator_before = {name: _tree_snapshot(module) for name, module in evaluators.items()}
    started = time.perf_counter()
    _require_gpu_margin(device)
    generated_indices = torch.arange(0, 2000, 200, dtype=torch.long)
    generated_rows = generated_ols.index_select(0, generated_indices).to(device)
    with torch.no_grad():
        rendered = model.forward_from_ols(generated_rows)
        _require(
            tuple(rendered.raw_camera.tensor.shape) == (10, 1, 400, 400),
            "generated-Ols preflight output geometry changed",
        )
        image_rows = torch.stack(
            [reference_images[(reference_labels == digit).nonzero()[0, 0]] for digit in range(10)]
        )
        target_400 = F.interpolate(
            image_rows.to(device), size=(400, 400), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        amplitude = target_400.sqrt()
        full_path = model(model.command(amplitude))
        decoded_again = model.forward_from_ols(full_path.ols.coefficients)
        _require(
            torch.allclose(
                full_path.raw_camera.tensor,
                decoded_again.raw_camera.tensor,
                rtol=RTOL,
                atol=ATOL,
            ),
            "encoder-produced OLS does not reproduce its generator output",
        )
        view = raw_400_to_view(full_path.raw_camera.tensor)
        evaluator_shapes = {}
        for name, evaluator in evaluators.items():
            logits = evaluator_logits(evaluator, view)
            evaluator_shapes[name] = list(logits.shape)
    torch.cuda.synchronize(device)
    for name, evaluator in evaluators.items():
        _assert_module_frozen(evaluator, evaluator_before[name], name)
    _assert_module_frozen(model, model_before, "400 LightGen model")
    elapsed = time.perf_counter() - started
    resources = _require_gpu_margin(device)
    return {
        "status": "READY",
        "forward_only": True,
        "optimizer_created": False,
        "optical_updates": 0,
        "generated_rows": generated_indices.tolist(),
        "balanced_train_rows": 10,
        "encoder_to_ols_then_decode_allclose": True,
        "evaluator_logits_shapes": evaluator_shapes,
        "model_and_evaluator_state_unchanged": True,
        "seconds": elapsed,
        "gpu_telemetry": resources,
    }


def _plane_measurements(result: Any) -> dict[str, Any]:
    planes = []
    for field in result.optical_trace:
        plane = field.plane
        intensity = field.tensor.abs().square()
        integral = (
            intensity.sum(dim=(-2, -1))
            .reshape(-1)
            .mul(plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1])
        )
        planes.append(
            {
                **plane.to_dict(),
                "power_proxy_by_row": integral.detach().cpu().to(torch.float64).tolist(),
            }
        )
    crop = {
        name: value.detach().cpu().to(torch.float64).tolist()
        for name, value in result.crop_statistics.items()
        if name.endswith("crop_retention")
    }
    return {"planes": planes, "crop_retention": crop}


def _combine_plane_measurements(records: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(records), "optical plane measurements are empty")
    first = records[0]
    planes = []
    for plane_index, plane in enumerate(first["planes"]):
        power = [
            value
            for record in records
            for value in record["planes"][plane_index]["power_proxy_by_row"]
        ]
        _require(
            all(
                record["planes"][plane_index]["plane_id"] == plane["plane_id"]
                for record in records
            ),
            "optical plane sequence changed between batches",
        )
        summary = {key: value for key, value in plane.items() if key != "power_proxy_by_row"}
        summary.update(
            {
                "power_proxy_by_row": power,
                "power_proxy_min": min(power),
                "power_proxy_mean": sum(power) / len(power),
                "power_proxy_max": max(power),
            }
        )
        planes.append(summary)
    crop_names = set(first["crop_retention"])
    _require(
        all(set(record["crop_retention"]) == crop_names for record in records),
        "crop retention groups changed between batches",
    )
    crop = {}
    for name in crop_names:
        rows = [row for record in records for row in record["crop_retention"][name]]
        by_layer = list(zip(*rows, strict=True))
        crop[name] = {
            "per_row_by_layer": rows,
            "layer_min": [min(values) for values in by_layer],
            "layer_mean": [sum(values) / len(values) for values in by_layer],
            "layer_max": [max(values) for values in by_layer],
        }
    return {"planes": planes, "crop_retention": crop}


def _merge_chunk_plane_measurements(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(chunks), "completed optical chunks are missing")
    source = [chunk["optical_plane_measurement"] for chunk in chunks]
    planes = []
    for plane_index, first in enumerate(source[0]["planes"]):
        rows = [
            value
            for record in source
            for value in record["planes"][plane_index]["power_proxy_by_row"]
        ]
        summary = {
            key: value
            for key, value in first.items()
            if key
            not in {
                "power_proxy_by_row",
                "power_proxy_min",
                "power_proxy_mean",
                "power_proxy_max",
            }
        }
        summary.update(
            {
                "power_proxy_by_row": rows,
                "power_proxy_min": min(rows),
                "power_proxy_mean": sum(rows) / len(rows),
                "power_proxy_max": max(rows),
            }
        )
        planes.append(summary)
    crop_names = set(source[0]["crop_retention"])
    crop = {}
    for name in crop_names:
        rows = [
            row
            for record in source
            for row in record["crop_retention"][name]["per_row_by_layer"]
        ]
        by_layer = list(zip(*rows, strict=True))
        crop[name] = {
            "per_row_by_layer": rows,
            "layer_min": [min(values) for values in by_layer],
            "layer_mean": [sum(values) / len(values) for values in by_layer],
            "layer_max": [max(values) for values in by_layer],
        }
    return {"planes": planes, "crop_retention": crop}


def _raw_metrics(raw: torch.Tensor) -> dict[str, Any]:
    flat = raw.flatten(1)
    total = flat.numel()
    return {
        "finite": bool(torch.isfinite(raw).all()),
        "min": float(raw.min()),
        "max": float(raw.max()),
        "fraction_pixels_above_1": float((raw > 1.0).sum()) / total,
        "zero_images": int((flat.max(1).values == 0).sum()),
        "constant_images": int((flat.max(1).values == flat.min(1).values).sum()),
        "nonzero_images": int((flat.max(1).values > 0).sum()),
    }


def _reconstruction_error_summary(
    raw: torch.Tensor, target: torch.Tensor, labels: torch.Tensor
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _require(raw.shape == target.shape, "raw/target reconstruction dimensions differ")
    squared = (raw.to(torch.float64) - target.to(torch.float64)).square()
    foreground = target > 0.1
    background = ~foreground
    _require(bool(foreground.flatten(1).any(1).all()), "a target image has an empty foreground")
    _require(bool(background.flatten(1).any(1).all()), "a target image has an empty background")
    values = {
        "full_image_mse": squared.flatten(1).mean(1).to(torch.float32),
        "foreground_mse_target_gt_0_1": (squared * foreground).flatten(1).sum(1)
        / foreground.flatten(1).sum(1),
        "background_mse_target_le_0_1": (squared * background).flatten(1).sum(1)
        / background.flatten(1).sum(1),
    }
    report = {
        "mask_definition": "foreground target intensity > 0.1; background <= 0.1",
        "per_row": {name: tensor.tolist() for name, tensor in values.items()},
        "per_class_mean": {
            str(digit): {
                name: float(value[labels == digit].mean())
                for name, value in values.items()
            }
            for digit in range(10)
        },
    }
    return report, values


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _save_torch_once(path: Path, payload: Any) -> None:
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        _require(_tree_equal(existing, payload), f"existing task artifact differs; refusing overwrite: {path.name}")
        return
    atomic_torch_save_new(path, payload, artifact_kind="frozen replay artifact")


def _save_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _require(existing == payload, f"existing task report differs; refusing overwrite: {path.name}")
        return
    atomic_write_new_json(path, payload)


def _save_image_once(path: Path, image: Image.Image) -> None:
    if path.exists():
        with Image.open(path) as existing:
            _require(existing.size == image.size and existing.convert("RGB").tobytes() == image.convert("RGB").tobytes(), f"existing visualization differs; refusing overwrite: {path.name}")
        return
    image.save(path)


def _load_existing_chunks(
    output_dir: Path,
    kind: str,
    expected_total: int,
    expected_labels: torch.Tensor,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    listed = manifest.get(f"completed_{kind}_chunks", [])
    _require(isinstance(listed, list), f"manifest {kind} chunk list is invalid")
    expected_count = math.ceil(expected_total / CHUNK_ROWS)
    _require(
        all(isinstance(index, int) and 0 <= index < expected_count for index in listed)
        and len(set(listed)) == len(listed),
        f"manifest {kind} chunks contain invalid indices",
    )
    # A process can stop after publishing a whole chunk but before updating the manifest.
    for index in range(expected_count):
        path = output_dir / f"{kind}_chunk_{index:04d}.pt"
        if path.is_file() and index not in listed:
            listed.append(index)
    listed.sort()
    chunks = []
    covered = 0
    for index in listed:
        path = output_dir / f"{kind}_chunk_{index:04d}.pt"
        _require(path.is_file(), f"manifest names missing chunk: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        start = index * CHUNK_ROWS
        stop = min(expected_total, start + CHUNK_ROWS)
        _require(
            payload.get("row_start") == start
            and payload.get("row_stop") == stop
            and torch.equal(payload.get("row_indices"), torch.arange(start, stop)),
            f"{kind} chunk {index} row order does not match the fixed population",
        )
        _require(
            torch.equal(payload.get("labels"), expected_labels[start:stop]),
            f"{kind} chunk {index} labels differ from the fixed population",
        )
        chunks.append(payload)
        covered += stop - start
    _require(
        listed == list(range(len(listed))),
        f"{kind} completed chunks must form a contiguous prefix",
    )
    _require(covered == sum(min(expected_total, (i + 1) * CHUNK_ROWS) - i * CHUNK_ROWS for i in listed), "chunk coverage changed")
    manifest[f"completed_{kind}_chunks"] = listed
    return chunks


def _save_chunk(
    path: Path,
    kind: str,
    start: int,
    stop: int,
    labels: torch.Tensor,
    payload: Mapping[str, Any],
) -> None:
    complete = {
        "schema": f"{kind}_chunk.v1",
        "row_start": start,
        "row_stop": stop,
        "row_indices": torch.arange(start, stop),
        "labels": labels[start:stop].cpu().long(),
        **payload,
    }
    atomic_torch_save_new(path, complete, artifact_kind=f"{kind} measurement chunk")


def _preflight_resume_roundtrip(
    output_dir: Path, generated_ols: torch.Tensor, generated_labels: torch.Tensor
) -> dict[str, Any]:
    scratch = output_dir / "_preflight_recovery_check"
    _require(not scratch.exists(), "preflight recovery scratch path already exists")
    scratch.mkdir()
    try:
        _save_chunk(
            scratch / "generated_chunk_0000.pt",
            "generated",
            0,
            CHUNK_ROWS,
            generated_labels,
            {"ols": generated_ols[:CHUNK_ROWS].clone()},
        )
        recovered_manifest: dict[str, Any] = {"completed_generated_chunks": []}
        recovered = _load_existing_chunks(
            scratch, "generated", 2000, generated_labels, recovered_manifest
        )
        _require(
            len(recovered) == 1
            and recovered_manifest["completed_generated_chunks"] == [0]
            and torch.equal(recovered[0]["ols"], generated_ols[:CHUNK_ROWS]),
            "published generated chunk did not recover from an interrupted manifest update",
        )
        repeated = _load_existing_chunks(
            scratch, "generated", 2000, generated_labels, recovered_manifest
        )
        _require(len(repeated) == 1, "second explicit chunk recovery changed its progress")
        return {
            "write_read_roundtrip": True,
            "interrupted_manifest_recovered_from_complete_chunk": True,
            "second_resume_did_not_duplicate_progress": True,
            "optical_updates": 0,
        }
    finally:
        shutil.rmtree(scratch)


def _evaluate_all(
    evaluators: Mapping[str, torch.nn.Module], views: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metrics: dict[str, Any] = {}
    logits_by_name: dict[str, torch.Tensor] = {}
    for name, evaluator in evaluators.items():
        logits = []
        with torch.no_grad():
            for start in range(0, views.shape[0], BATCH_SIZE):
                batch = views[start : start + BATCH_SIZE].to(device)
                logits.append(evaluator_logits(evaluator, batch).detach().cpu())
        all_logits = torch.cat(logits)
        logits_by_name[name] = all_logits
        metrics[name] = confusion_metrics(all_logits.argmax(1), labels.cpu())
    return metrics, logits_by_name


def _focus_class_rows(
    logits_by_name: Mapping[str, torch.Tensor], labels: torch.Tensor
) -> dict[str, Any]:
    result: dict[str, Any] = {"classes": {}, "evaluator_order": list(logits_by_name)}
    predictions = {name: logits.argmax(1) for name, logits in logits_by_name.items()}
    for digit in (1, 7):
        indices = (labels == digit).nonzero().flatten()
        rows = []
        evaluator_summary = {}
        for name, logits in logits_by_name.items():
            selected = logits.index_select(0, indices)
            pred = predictions[name].index_select(0, indices)
            correct_logit = selected[:, digit]
            other_logits = selected.clone()
            other_logits[:, digit] = -torch.inf
            margins = correct_logit - other_logits.max(1).values
            wrong_counts = torch.bincount(pred[pred != digit], minlength=10)
            evaluator_summary[name] = {
                "count": int(indices.numel()),
                "correct": int((pred == digit).sum()),
                "recall": float((pred == digit).to(torch.float32).mean()),
                "mean_true_vs_best_other_logit_margin": float(margins.mean()),
                "wrong_prediction_counts": wrong_counts.tolist(),
            }
        for row in indices.tolist():
            rows.append(
                {
                    "row_index": row,
                    "label": digit,
                    "predictions": {
                        name: int(predictions[name][row]) for name in logits_by_name
                    },
                    "true_vs_best_other_logit_margin": {
                        name: float(
                            logits[row, digit]
                            - torch.cat((logits[row, :digit], logits[row, digit + 1 :])).max()
                        )
                        for name, logits in logits_by_name.items()
                    },
                    "evaluator_disagreement": len(
                        {int(predictions[name][row]) for name in logits_by_name}
                    ) > 1,
                }
            )
        result["classes"][str(digit)] = {
            "row_count": int(indices.numel()),
            "evaluator_summary": evaluator_summary,
            "pairwise_evaluator_disagreement_rows": sum(
                row["evaluator_disagreement"] for row in rows
            ),
            "rows": rows,
        }
    return result


def _concatenate_chunks(chunks: list[dict[str, Any]], key: str) -> torch.Tensor:
    if not chunks:
        raise ReplayError(f"no completed chunks are available for {key}")
    return torch.cat([chunk[key] for chunk in chunks], dim=0)


def _duplicates(values: torch.Tensor) -> int:
    if torch.is_complex(values):
        values = torch.view_as_real(values)
    flattened = values.reshape(values.shape[0], -1).contiguous()
    return int(values.shape[0] - torch.unique(flattened, dim=0).shape[0])


def _pairwise_rmse(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.to(torch.float64).reshape(values.shape[0], -1)
    squared = (flat * flat).sum(1)
    d2 = (squared[:, None] + squared[None, :] - 2.0 * flat @ flat.T).clamp_min(0.0)
    distances = (d2 / flat.shape[1]).sqrt()
    distances.fill_diagonal_(float("inf"))
    nearest_distance, nearest_index = distances.min(1)
    return nearest_distance.to(torch.float32), nearest_index


def _nearest_to_targets(
    generated: torch.Tensor, targets: torch.Tensor, target_labels: torch.Tensor, requested: torch.Tensor
) -> dict[str, torch.Tensor]:
    generated_flat = generated.to(torch.float64).flatten(1)
    targets_flat = targets.to(torch.float64).flatten(1)
    target_norm = (targets_flat * targets_flat).sum(1)
    all_indices, all_distances, same_indices, same_distances = [], [], [], []
    for start in range(0, generated.shape[0], 50):
        batch = generated_flat[start : start + 50]
        batch_norm = (batch * batch).sum(1)
        distances = (
            batch_norm[:, None] + target_norm[None, :] - 2.0 * batch @ targets_flat.T
        ).clamp_min(0.0)
        distances = (distances / generated_flat.shape[1]).sqrt()
        value, index = distances.min(1)
        all_indices.append(index)
        all_distances.append(value)
        wanted = requested[start : start + batch.shape[0]]
        same_distance = distances.masked_fill(
            target_labels[None, :] != wanted[:, None], float("inf")
        )
        same_value, same_index = same_distance.min(1)
        same_indices.append(same_index)
        same_distances.append(same_value)
    return {
        "nearest_any_target_index": torch.cat(all_indices),
        "nearest_any_target_rmse": torch.cat(all_distances).to(torch.float32),
        "nearest_requested_class_target_index": torch.cat(same_indices),
        "nearest_requested_class_target_rmse": torch.cat(same_distances).to(torch.float32),
    }


def _risk_statistics(
    ols: torch.Tensor,
    views: torch.Tensor,
    requested: torch.Tensor,
    target_views: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    nearest_generated_distance = torch.empty(2000)
    nearest_generated_index = torch.empty(2000, dtype=torch.long)
    class_rows = []
    for digit in range(10):
        indices = (requested == digit).nonzero().flatten()
        class_ols = ols.index_select(0, indices)
        class_views = views.index_select(0, indices)
        nearest, neighbors = _pairwise_rmse(class_views)
        nearest_generated_distance[indices] = nearest
        nearest_generated_index[indices] = indices[neighbors]
        class_rows.append(
            {
                "class": digit,
                "count": int(indices.numel()),
                "exact_duplicate_ols_rows": _duplicates(class_ols),
                "exact_duplicate_view_rows": _duplicates(class_views),
                "mean_pixel_variance": float(class_views.to(torch.float64).var(0, unbiased=False).mean()),
                "mean_nearest_generation_rmse": float(nearest.mean()),
                "min_nearest_generation_rmse": float(nearest.min()),
            }
        )
    target_neighbors = _nearest_to_targets(views, target_views, target_labels, requested)
    statistics = {
        "per_class": class_rows,
        "all_rows_mean_nearest_generation_rmse": float(nearest_generated_distance.mean()),
        "nearest_target_rmse_mean": float(target_neighbors["nearest_any_target_rmse"].mean()),
        "nearest_same_class_target_rmse_mean": float(
            target_neighbors["nearest_requested_class_target_rmse"].mean()
        ),
        "nearest_distances_scope": "fixed 300-row canonical TRAIN reference only; descriptive, not novelty acceptance",
    }
    return statistics, {
        "nearest_generation_index": nearest_generated_index,
        "nearest_generation_rmse": nearest_generated_distance,
        **target_neighbors,
    }


def _save_generated_contact_sheets(
    output_dir: Path,
    views: torch.Tensor,
    requested: torch.Tensor,
    predictions: torch.Tensor,
) -> list[str]:
    files = []
    for digit in range(10):
        indices = (requested == digit).nonzero().flatten().tolist()
        columns, rows = 20, 10
        tile_w, tile_h = 64, 47
        sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for local, index in enumerate(indices):
            image = views[index, 0].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            tile = Image.fromarray(image, mode="L").resize((56, 36), Image.Resampling.NEAREST).convert("RGB")
            x, y = (local % columns) * tile_w, (local // columns) * tile_h
            sheet.paste(tile, (x + 4, y))
            draw.text((x + 4, y + 36), f"{index} p{int(predictions[index])}", fill="black")
        filename = f"generated_class_{digit}_200_views.png"
        _save_image_once(output_dir / filename, sheet)
        files.append(filename)
    return files


def _save_raw_comparison(
    output_dir: Path, historical: torch.Tensor, replayed: torch.Tensor
) -> str:
    maximum = max(float(historical.max()), float(replayed.max()), 1.0)
    tile = 96
    sheet = Image.new("RGB", (20 * tile, 5 * (tile + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for index in range(50):
        digit, example = divmod(index, 5)
        for offset, tensor in enumerate((historical, replayed)):
            values = tensor[index, 0].clamp(0.0, maximum).div(maximum).mul(255).round().to(torch.uint8).numpy()
            image = Image.fromarray(values, mode="L").resize((tile - 4, tile - 4), Image.Resampling.NEAREST)
            x = (digit * 2 + offset) * tile
            y = example * (tile + 22)
            sheet.paste(image, (x + 2, y))
            draw.text((x + 2, y + tile - 2), f"c{digit} n{example} {'old' if offset == 0 else 'new'}", fill="black")
    filename = "historical_vs_replayed_raw_50.png"
    _save_image_once(output_dir / filename, sheet)
    return filename


def _save_nearest_pairs(
    output_dir: Path,
    views: torch.Tensor,
    target_views: torch.Tensor,
    target_labels: torch.Tensor,
    neighbors: Mapping[str, torch.Tensor],
    requested: torch.Tensor,
) -> str:
    tile_w, tile_h = 170, 60
    rows = 50
    sheet = Image.new("RGB", (tile_w * 2, tile_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    row_index = 0
    for digit in range(10):
        indices = (requested == digit).nonzero().flatten()
        distances = neighbors["nearest_requested_class_target_rmse"].index_select(0, indices)
        chosen = indices[torch.argsort(distances)[:5]]
        for generated_index in chosen.tolist():
            target_index = int(neighbors["nearest_requested_class_target_index"][generated_index])
            for column, image_tensor in enumerate((views[generated_index], target_views[target_index])):
                array = image_tensor[0].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                image = Image.fromarray(array, mode="L").resize((56, 56), Image.Resampling.NEAREST).convert("RGB")
                x = column * tile_w
                y = row_index * tile_h
                sheet.paste(image, (x, y))
                if column == 0:
                    draw.text((x + 62, y + 5), f"gen {generated_index} c{digit}", fill="black")
                    draw.text((x + 62, y + 24), f"RMSE {float(neighbors['nearest_requested_class_target_rmse'][generated_index]):.5f}", fill="black")
                else:
                    draw.text((x + 62, y + 5), f"TRAIN row {target_index}", fill="black")
                    draw.text((x + 62, y + 24), f"label {int(target_labels[target_index])}", fill="black")
            row_index += 1
    filename = "nearest_same_class_pairs_50.png"
    _save_image_once(output_dir / filename, sheet)
    return filename


def _resolve_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "artifact_dir": _resolve(ROOT, args.artifact_dir),
        "config": _resolve(ROOT, args.config),
        "reference_population": _resolve(ROOT, args.reference_population),
        "training_evaluator": _resolve(ROOT, args.training_evaluator),
        "data": _resolve(ROOT, DATA_PATH),
        "output_dir": _resolve(ROOT, args.output_dir),
    }


def _load_and_validate_inputs(
    paths: Mapping[str, Path]
) -> dict[str, Any]:
    artifact_dir = paths["artifact_dir"]
    model_path = artifact_dir / "model_step_000600.pt"
    ols_path = artifact_dir / "generated_ols.pt"
    visuals_path = artifact_dir / "generated_visuals.pt"
    report_path = artifact_dir / "report.json"
    for path in (model_path, ols_path, visuals_path, report_path, paths["config"], paths["reference_population"], paths["training_evaluator"], paths["data"]):
        _require(path.is_file(), f"required explicit input is missing: {path}")
    population = _mapping(
        json.loads(paths["reference_population"].read_text(encoding="utf-8")),
        "fixed reference population",
    )
    _require(population.get("task_id") == "M0_CLASS17_MAPPING_DIAGNOSTIC_V2", "reference population authority changed")
    _require(population.get("fit_rows") == 200 and population.get("observation_rows") == 100, "fixed population row contract changed")
    input_evaluators = evaluator_paths(population)
    for name, path in input_evaluators.items():
        _require(path.is_file(), f"population-named {name} checkpoint is missing: {path}")
    checkpoint = _mapping(_load_torch(model_path), "step600 model checkpoint")
    _require(
        checkpoint.get("schema") == "lightgen_400_mnist_semantic_baseline_checkpoint.v1"
        and checkpoint.get("completed_updates") == 600,
        "saved 400 model is not the completed step600 baseline",
    )
    _require("optimizer" not in checkpoint and "optimizer_state" not in checkpoint, "step600 source unexpectedly contains optimizer state")
    sampler = _mapping(checkpoint.get("sampler"), "step600 sampling statistics")
    _require(sampler.get("fit_per_class") == 200 and sampler.get("sample_seed") == 20260928, "saved sampler population changed")
    for key in ("class_mean", "class_real_std", "class_imag_std"):
        tensor = sampler.get(key)
        _require(torch.is_tensor(tensor) and tensor.shape == (10, 100) and bool(torch.isfinite(tensor).all()), f"saved sampler {key} is invalid")
    _require(bool((sampler["class_real_std"] >= 0).all()) and bool((sampler["class_imag_std"] >= 0).all()), "saved sampler scale is negative")
    state = _mapping(checkpoint.get("model"), "step600 model state")
    expected_shapes = {
        "encoder.phase": (2, 1, 400, 400),
        "decoder.phase": (4, 1, 400, 400),
        "fiber_modes": (100, 400, 400),
        "fiber_centers_m": (100, 2),
    }
    _require(set(state) == set(expected_shapes), "step600 model state keys changed")
    for name, shape in expected_shapes.items():
        tensor = state[name]
        _require(torch.is_tensor(tensor) and tuple(tensor.shape) == shape and bool(torch.isfinite(tensor).all()), f"step600 model tensor {name} is invalid")
    _require(torch.is_complex(state["fiber_modes"]), "saved fiber modes lost complex dtype")
    config = load_config(paths["config"])
    config_record = validate_system_config(config)
    generated_ols, generated_labels, historic_views, historic_raw = validate_generated_payloads(
        _mapping(_load_torch(ols_path), "generated OLS"),
        _mapping(_load_torch(visuals_path), "generated visuals"),
    )
    historical_raw_views = raw_400_to_view(historic_raw)
    mapped_history_indices = historical_raw_indices()
    historical_view_check = compare_tensors(
        historical_raw_views,
        historic_views.index_select(0, mapped_history_indices),
    )
    _require(historical_view_check["matches"], "the saved raw examples do not reproduce their saved 28x28 views")
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "historical report")
    _require(report.get("schema") == "lightgen_400_mnist_semantic_baseline_report.v1", "original report schema changed")
    _require(isinstance(report.get("status"), str) and report["status"], "original report status metadata is missing")
    generated_report = _mapping(report.get("generated"), "original generated metrics")
    _require(
        generated_report.get("count") == 2000
        and isinstance(generated_report.get("confusion"), list)
        and len(generated_report["confusion"]) == 10,
        "original report generated classification readout changed",
    )
    row_ids, ref_labels, ref_images, all_labels = _load_data_rows(paths["data"], paths["reference_population"], population)
    return {
        "checkpoint": checkpoint,
        "model_path": model_path,
        "generated_ols": generated_ols,
        "generated_labels": generated_labels,
        "historical_views": historic_views,
        "historical_raw": historic_raw,
        "historical_report": report,
        "historical_generated_report": generated_report,
        "historical_raw_view_check": historical_view_check,
        "population": population,
        "population_path": paths["reference_population"],
        "evaluator_paths": input_evaluators,
        "config": config,
        "config_record": config_record,
        "row_ids": row_ids,
        "reference_labels": ref_labels,
        "reference_images": ref_images,
        "all_labels": all_labels,
    }


def _identity_record(paths: Mapping[str, Path], loaded: Mapping[str, Any]) -> dict[str, Any]:
    state = loaded["checkpoint"]["model"]
    sampler = loaded["checkpoint"]["sampler"]
    return {
        "task_id": TASK_ID,
        "input_paths": {key: value.as_posix() for key, value in paths.items() if key != "output_dir"},
        "output_dir": paths["output_dir"].as_posix(),
        "checkpoint_step": 600,
        "checkpoint_state_shapes": {key: list(value.shape) for key, value in state.items()},
        "sampler_state_shapes": {
            key: list(value.shape) if torch.is_tensor(value) else value
            for key, value in sampler.items()
        },
        "checkpoint_completed_updates": 600,
        "optimizer_state_present": False,
        "configuration": loaded["config_record"],
        "population_rows": loaded["row_ids"].tolist(),
        "population_labels": loaded["reference_labels"].tolist(),
        "population_role": "fixed canonical TRAIN reference; not an independent generalization set",
        "historical_digest_fields": "provenance only; not used as an execution lock",
    }


def _write_initial_output(
    output_dir: Path,
    identity: Mapping[str, Any],
    model: torch.nn.Module,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(output_dir / "configuration.json", identity["configuration"])
    atomic_write_new_json(output_dir / "input_population.json", population)
    atomic_torch_save_new(
        output_dir / "frozen_model_state.pt",
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        artifact_kind="frozen model state",
    )
    atomic_torch_save_new(
        output_dir / "frozen_evaluator_states.pt",
        {name: {key: value.detach().cpu() for key, value in module.state_dict().items()} for name, module in evaluators.items()},
        artifact_kind="frozen evaluator states",
    )
    atomic_torch_save_new(
        output_dir / "frozen_sampler_state.pt",
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in sampler.items()},
        artifact_kind="frozen sampler state",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "status": "PREFLIGHT_PENDING",
        "identity": identity,
        "completed_generated_chunks": [],
        "completed_reference_chunks": [],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updates": 0,
    }
    atomic_write_new_json(output_dir / "resume_manifest.json", manifest)
    return manifest


def _load_resume_manifest(
    manifest_path: Path,
    output_dir: Path,
    identity: Mapping[str, Any],
    model: torch.nn.Module,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
) -> dict[str, Any]:
    _require(manifest_path.resolve() == (output_dir / "resume_manifest.json").resolve(), "resume manifest must be the manifest inside the explicit output directory")
    _require(manifest_path.is_file(), f"resume manifest is missing: {manifest_path}")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "resume manifest")
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("task_id") == TASK_ID, "resume manifest task/schema mismatch")
    _require(manifest.get("identity") == identity, "resume paths, model step, actual config, or fixed row population changed")
    saved_model = _load_torch(output_dir / "frozen_model_state.pt")
    observed_model = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    _require(set(saved_model) == set(observed_model) and all(torch.equal(saved_model[key], observed_model[key]) for key in saved_model), "current step600 model differs from saved frozen state")
    saved_evaluators = _load_torch(output_dir / "frozen_evaluator_states.pt")
    _require(set(saved_evaluators) == set(evaluators), "resume evaluator set changed")
    for name, evaluator in evaluators.items():
        observed = {key: value.detach().cpu() for key, value in evaluator.state_dict().items()}
        _require(set(saved_evaluators[name]) == set(observed) and all(torch.equal(saved_evaluators[name][key], observed[key]) for key in observed), f"resume evaluator state changed: {name}")
    saved_sampler = _load_torch(output_dir / "frozen_sampler_state.pt")
    _require(set(saved_sampler) == set(sampler), "resume sampler state keys changed")
    for key, value in sampler.items():
        observed = value.detach().cpu() if torch.is_tensor(value) else value
        saved = saved_sampler[key]
        _require(
            torch.equal(saved, observed) if torch.is_tensor(observed) else saved == observed,
            f"resume sampler state changed: {key}",
        )
    return dict(manifest)


def _check_state_files(
    output_dir: Path,
    model: torch.nn.Module,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
) -> None:
    expected_model = _load_torch(output_dir / "frozen_model_state.pt")
    current_model = model.state_dict()
    _require(all(torch.equal(current_model[key].detach().cpu(), expected_model[key]) for key in expected_model), "frozen optical state changed during replay")
    expected_evaluators = _load_torch(output_dir / "frozen_evaluator_states.pt")
    for name, module in evaluators.items():
        _assert_module_frozen(module, expected_evaluators[name], name)
    expected_sampler = _load_torch(output_dir / "frozen_sampler_state.pt")
    _require(set(expected_sampler) == set(sampler), "sampler state keys changed during replay")
    for key, value in sampler.items():
        observed = value.detach().cpu() if torch.is_tensor(value) else value
        saved = expected_sampler[key]
        _require(
            torch.equal(saved, observed) if torch.is_tensor(observed) else saved == observed,
            f"sampler state changed during replay: {key}",
        )


def _verify_terminal_replay_resume(
    output_dir: Path,
    manifest: dict[str, Any],
    generated_ols: torch.Tensor,
    generated_labels: torch.Tensor,
    reference_row_ids: torch.Tensor,
    reference_labels: torch.Tensor,
    reference_images: torch.Tensor,
    model: torch.nn.Module,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
) -> Path:
    generated_chunks = _load_existing_chunks(
        output_dir, "generated", 2000, generated_labels, manifest
    )
    reference_chunks = _load_existing_chunks(
        output_dir, "reference", 300, reference_labels, manifest
    )
    _require(
        len(generated_chunks) == 20 and len(reference_chunks) == 3,
        "terminal resume does not contain all fixed measurement chunks",
    )
    for chunk in generated_chunks:
        start, stop = int(chunk["row_start"]), int(chunk["row_stop"])
        _require(
            torch.equal(chunk.get("ols"), generated_ols[start:stop]),
            f"terminal generated OLS rows {start}:{stop} differ from the source artifact",
        )
    for chunk in reference_chunks:
        start, stop = int(chunk["row_start"]), int(chunk["row_stop"])
        _require(
            torch.equal(chunk.get("source_row_indices"), reference_row_ids[start:stop])
            and torch.equal(chunk.get("target_image_28x28"), reference_images[start:stop]),
            f"terminal reference rows {start}:{stop} differ from the fixed TRAIN population",
        )
    _check_state_files(output_dir, model, evaluators, sampler)
    report_path = output_dir / "report.json"
    _require(report_path.is_file(), "terminal resume has no final report")
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "terminal report")
    _require(
        report.get("status") == manifest.get("status")
        and report.get("classification") == manifest.get("classification"),
        "terminal report and resume manifest disagree",
    )
    return report_path


def _measure_generated(
    output_dir: Path,
    manifest: dict[str, Any],
    model: LightGenAmplitudeFiberOLSSystemV2,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
    ols: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    chunks = _load_existing_chunks(output_dir, "generated", 2000, labels, manifest)
    for chunk in chunks:
        start, stop = int(chunk["row_start"]), int(chunk["row_stop"])
        _require(
            torch.equal(chunk.get("ols"), ols[start:stop]),
            f"generated replay chunk {start}:{stop} does not match the fixed saved OLS input",
        )
    finished = set(manifest["completed_generated_chunks"])
    for chunk_index in range(20):
        if chunk_index in finished:
            continue
        _require_gpu_margin(device)
        start = chunk_index * CHUNK_ROWS
        stop = min(2000, start + CHUNK_ROWS)
        ols_chunk = ols[start:stop]
        raw_parts = []
        plane_records = []
        with torch.no_grad():
            for local_start in range(0, stop - start, BATCH_SIZE):
                z = ols_chunk[local_start : local_start + BATCH_SIZE].to(device)
                result = model.forward_from_ols(z)
                raw = result.raw_camera.tensor
                _require(bool(torch.isfinite(raw).all()), "generated replay contains nonfinite raw CMOS")
                raw_parts.append(raw.detach().cpu())
                plane_records.append(_plane_measurements(result))
        raw_cpu = torch.cat(raw_parts)
        view_cpu = raw_400_to_view(raw_cpu)
        _require(bool(torch.isfinite(view_cpu).all()), "generated replay view contains nonfinite values")
        power_ols = ols_chunk.abs().square().sum(1).to(torch.float32)
        chunk_payload = {
            "ols": ols_chunk.clone(),
            "raw_cmos_400x400": raw_cpu,
            "evaluator_view_28x28": view_cpu,
            "ols_coefficient_squared_norm": power_ols,
            "raw_metrics": _raw_metrics(raw_cpu),
            "optical_plane_measurement": _combine_plane_measurements(plane_records),
            "ols_coordinate": {
                "grid": [10, 10],
                "modes": 100,
                "pixel_pitch_m": [3.0e-6, 3.0e-6],
                "wavelength_m": 532.0e-9,
                "z_m": 2.0e-3,
                "power_quantity": "sum(abs(complex_OLS_coefficients)^2); distinct from spatial-plane power integral",
            },
        }
        path = output_dir / f"generated_chunk_{chunk_index:04d}.pt"
        _save_chunk(path, "generated", start, stop, labels, chunk_payload)
        manifest["completed_generated_chunks"].append(chunk_index)
        manifest["completed_generated_chunks"].sort()
        manifest["status"] = "GENERATED_REPLAY_IN_PROGRESS"
        atomic_write_json(output_dir / "resume_manifest.json", manifest)
        _check_state_files(output_dir, model, evaluators, sampler)
        chunks = _load_existing_chunks(output_dir, "generated", 2000, labels, manifest)
        finished.add(chunk_index)
        print(f"replayed generated OLS rows {start}:{stop}", flush=True)
    return chunks


def _measure_reference(
    output_dir: Path,
    manifest: dict[str, Any],
    model: LightGenAmplitudeFiberOLSSystemV2,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    row_ids: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    chunks = _load_existing_chunks(output_dir, "reference", 300, labels, manifest)
    for chunk in chunks:
        start, stop = int(chunk["row_start"]), int(chunk["row_stop"])
        _require(
            torch.equal(chunk.get("source_row_indices"), row_ids[start:stop])
            and torch.equal(chunk.get("target_image_28x28"), images[start:stop]),
            f"reference replay chunk {start}:{stop} does not match the fixed canonical TRAIN rows",
        )
    finished = set(manifest["completed_reference_chunks"])
    for chunk_index in range(3):
        if chunk_index in finished:
            continue
        _require_gpu_margin(device)
        start = chunk_index * CHUNK_ROWS
        stop = min(300, start + CHUNK_ROWS)
        image_chunk = images[start:stop]
        target = F.interpolate(image_chunk, size=(400, 400), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        amplitude = target.sqrt().to(device)
        ols_parts, raw_parts = [], []
        plane_records = []
        with torch.no_grad():
            for batch_start in range(0, stop - start, BATCH_SIZE):
                current = amplitude[batch_start : batch_start + BATCH_SIZE]
                result = model(model.command(current))
                _require(bool(torch.isfinite(result.raw_camera.tensor).all()), "reference path contains nonfinite CMOS")
                ols_parts.append(result.ols.coefficients.detach().cpu())
                raw_parts.append(result.raw_camera.tensor.detach().cpu())
                plane_records.append(_plane_measurements(result))
        raw_cpu = torch.cat(raw_parts)
        view_cpu = raw_400_to_view(raw_cpu)
        chunk_payload = {
            "source_row_indices": row_ids[start:stop].clone(),
            "ols": torch.cat(ols_parts),
            "raw_cmos_400x400": raw_cpu,
            "evaluator_view_28x28": view_cpu,
            "ols_coefficient_squared_norm": torch.cat(ols_parts).abs().square().sum(1).to(torch.float32),
            "raw_metrics": _raw_metrics(raw_cpu),
            "optical_plane_measurement": _combine_plane_measurements(plane_records),
            "target_image_28x28": images[start:stop].clone(),
            "target_intensity_400x400": target.detach().cpu(),
            "ols_coordinate": {
                "grid": [10, 10],
                "modes": 100,
                "pixel_pitch_m": [3.0e-6, 3.0e-6],
                "wavelength_m": 532.0e-9,
                "z_m": 2.0e-3,
            },
        }
        path = output_dir / f"reference_chunk_{chunk_index:04d}.pt"
        _save_chunk(path, "reference", start, stop, labels, chunk_payload)
        manifest["completed_reference_chunks"].append(chunk_index)
        manifest["completed_reference_chunks"].sort()
        manifest["status"] = "REFERENCE_REPLAY_IN_PROGRESS"
        atomic_write_json(output_dir / "resume_manifest.json", manifest)
        _check_state_files(output_dir, model, evaluators, sampler)
        chunks = _load_existing_chunks(output_dir, "reference", 300, labels, manifest)
        finished.add(chunk_index)
        print(f"replayed canonical TRAIN rows {start}:{stop}", flush=True)
    return chunks


def _statistical_risk(
    generated_ols: torch.Tensor,
    generated_views: torch.Tensor,
    generated_labels: torch.Tensor,
    target_views: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    stats, pairs = _risk_statistics(
        generated_ols, generated_views, generated_labels, target_views, target_labels
    )
    return stats, pairs


def _decision(
    replay_views_match: bool,
    replay_raw_match: bool,
    generated_metrics: Mapping[str, Any],
    raw_metrics: Mapping[str, Any],
    historical_ed_matches: bool,
    replay_ed_matches: bool,
    ed_control_valid: bool,
) -> tuple[str, str | None, bool]:
    if not ed_control_valid:
        return "INVALID_HARNESS", None, False
    finite_valid = raw_metrics["finite"] and raw_metrics["nonzero_images"] == 2000 and raw_metrics["constant_images"] == 0
    original_numeric = (
        generated_metrics["macro_recall"] >= 0.50
        and min(generated_metrics["recall_by_class"]) >= 0.10
        and raw_metrics["fraction_pixels_above_1"] <= 0.01
        and finite_valid
    )
    replay_consistent = replay_views_match and replay_raw_match and historical_ed_matches and replay_ed_matches
    if not replay_consistent:
        return "VALID_FAILED", "FROZEN_REPLAY_DISCREPANCY", original_numeric
    if not original_numeric:
        return "VALID_FAILED", "ORIGINAL_NUMERIC_CRITERIA_NOT_MET", False
    return "VALID_COMPLETED", "FROZEN_REPLAY_CONFIRMED", True


def _run_measurement(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    loaded: Mapping[str, Any],
    model: LightGenAmplitudeFiberOLSSystemV2,
    evaluators: Mapping[str, torch.nn.Module],
    sampler: Mapping[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = paths["output_dir"]
    generated_chunks = _measure_generated(
        output_dir,
        manifest,
        model,
        evaluators,
        sampler,
        loaded["generated_ols"],
        loaded["generated_labels"],
        device,
    )
    reference_chunks = _measure_reference(
        output_dir,
        manifest,
        model,
        evaluators,
        sampler,
        loaded["reference_images"],
        loaded["reference_labels"],
        loaded["row_ids"],
        device,
    )
    _check_state_files(output_dir, model, evaluators, sampler)
    generated_ols = _concatenate_chunks(generated_chunks, "ols")
    generated_raw = _concatenate_chunks(generated_chunks, "raw_cmos_400x400")
    generated_views = _concatenate_chunks(generated_chunks, "evaluator_view_28x28")
    reference_ols = _concatenate_chunks(reference_chunks, "ols")
    reference_raw = _concatenate_chunks(reference_chunks, "raw_cmos_400x400")
    reference_views = _concatenate_chunks(reference_chunks, "evaluator_view_28x28")
    reference_target_400 = _concatenate_chunks(reference_chunks, "target_intensity_400x400")
    generated_metrics, generated_logits = _evaluate_all(
        evaluators, generated_views, loaded["generated_labels"], device
    )
    reference_metrics, reference_logits = _evaluate_all(
        evaluators, reference_views, loaded["reference_labels"], device
    )
    target_metrics, target_logits = _evaluate_all(
        evaluators, loaded["reference_images"], loaded["reference_labels"], device
    )
    history_metrics, history_logits = _evaluate_all(
        evaluators, loaded["historical_views"], loaded["generated_labels"], device
    )
    focus_row_metrics = {
        "generated_replay": _focus_class_rows(generated_logits, loaded["generated_labels"]),
        "historical_saved_views": _focus_class_rows(history_logits, loaded["generated_labels"]),
        "reference_model_outputs": _focus_class_rows(reference_logits, loaded["reference_labels"]),
        "reference_target_images": _focus_class_rows(target_logits, loaded["reference_labels"]),
    }
    views_comparison = compare_tensors(generated_views, loaded["historical_views"])
    raw_indices = historical_raw_indices()
    replay_raw_50 = generated_raw.index_select(0, raw_indices)
    raw_comparison = compare_tensors(replay_raw_50, loaded["historical_raw"])
    reference_row_ids = loaded["row_ids"]
    _require(reference_row_ids.numel() == 300, "reference source rows changed during measurement")
    historical_report_confusion = loaded["historical_generated_report"]["confusion"]
    hist_ed_confusion_matches = history_metrics["E_D"]["confusion"] == historical_report_confusion
    replay_ed_confusion_matches = generated_metrics["E_D"]["confusion"] == historical_report_confusion
    ed_control_valid = _expected_positive_control(target_metrics["E_D"])
    raw_stats = _raw_metrics(generated_raw)
    status, classification, original_numeric_passed = _decision(
        views_comparison["matches"],
        raw_comparison["matches"],
        generated_metrics["E_D"],
        raw_stats,
        hist_ed_confusion_matches,
        replay_ed_confusion_matches,
        ed_control_valid,
    )
    if status == "INVALID_HARNESS":
        classification = None
    risk, neighbor_tensors = _statistical_risk(
        generated_ols,
        generated_views,
        loaded["generated_labels"],
        loaded["reference_images"],
        loaded["reference_labels"],
    )
    reconstruction_summary, reconstruction_rows = _reconstruction_error_summary(
        reference_raw, reference_target_400, loaded["reference_labels"]
    )
    generated_class_raw_metrics = {
        str(digit): _raw_metrics(generated_raw[loaded["generated_labels"] == digit])
        for digit in range(10)
    }
    reference_raw_metrics = {
        str(digit): _raw_metrics(reference_raw[loaded["reference_labels"] == digit])
        for digit in range(10)
    }
    _save_torch_once(
        output_dir / "final_readouts.pt",
        {
            "generated_ols": generated_ols,
            "generated_raw_cmos_400x400": generated_raw,
            "generated_views_28x28": generated_views,
            "generated_labels": loaded["generated_labels"],
            "generated_logits": generated_logits,
            "reference_ols": reference_ols,
            "reference_raw_cmos_400x400": reference_raw,
            "reference_views_28x28": reference_views,
            "reference_labels": loaded["reference_labels"],
            "reference_source_row_indices": reference_row_ids,
            "reference_logits": reference_logits,
            "reference_reconstruction_error_per_row": reconstruction_rows,
            "target_image_logits": target_logits,
            "historical_view_logits": history_logits,
            "neighbor_indices_and_distances": neighbor_tensors,
        },
    )
    _save_torch_once(
        output_dir / "risk_nearest_neighbor_rows.pt",
        neighbor_tensors,
    )
    generated_prediction = generated_logits["E_D"].argmax(1)
    contact_files = _save_generated_contact_sheets(
        output_dir, generated_views, loaded["generated_labels"], generated_prediction
    )
    raw_sheet = _save_raw_comparison(output_dir, loaded["historical_raw"], replay_raw_50)
    nearest_sheet = _save_nearest_pairs(
        output_dir,
        generated_views,
        loaded["reference_images"],
        loaded["reference_labels"],
        neighbor_tensors,
        loaded["generated_labels"],
    )
    _check_state_files(output_dir, model, evaluators, sampler)
    device_telemetry = _require_gpu_margin(device)
    plane_measures = {
        "generated_OLS_to_generator": _merge_chunk_plane_measurements(generated_chunks),
        "reference_encoder_OLS_generator": _merge_chunk_plane_measurements(reference_chunks),
        "ols_power_quantity": "sum(|z_k|^2) in 100-mode coefficient space",
        "spatial_power_quantity": "sum(|U(x,y)|^2)*pixel_pitch_x*pixel_pitch_y per recorded PlaneSpec",
    }
    _save_json_once(output_dir / "class17_row_metrics.json", focus_row_metrics)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "evidence_tier": "diagnostic-only frozen replay",
        "status": status,
        "classification": classification,
        "next_action": "NONE",
        "training_updates": 0,
        "optimizer_created": False,
        "scientific_claim": "frozen replay and remaining semantic-presentation risks only; no full semantic-generation claim",
        "input_identity": manifest["identity"],
        "preflight": json.loads((output_dir / "preflight.json").read_text(encoding="utf-8")),
        "measurements": {
            "generated": {
                "count": 2000,
                "batch_size": BATCH_SIZE,
                "class_count": 200,
                "evaluation": generated_metrics,
                "evaluator_logits_file": "final_readouts.pt",
                "raw_cmos_metrics_400x400": raw_stats,
                "raw_cmos_metrics_by_requested_class": generated_class_raw_metrics,
                "finite_output": raw_stats["finite"],
                "power_and_coordinate_records": "optical_plane_and_power.json",
            },
            "fixed_canonical_train_reference": {
                "count": 300,
                "class_count": 30,
                "source_row_indices": reference_row_ids.tolist(),
                "data_role": "TRAIN reference only; not an independent generalization set",
                "model_output_evaluation": reference_metrics,
                "raw_cmos_metrics_by_true_class": reference_raw_metrics,
                "raw_400x400_reconstruction_error": reconstruction_summary,
                "original_28x28_target_positive_control": target_metrics,
                "e_d_positive_control": {
                    "valid": ed_control_valid,
                    "required": {"macro_recall_min": 0.90, "class_1_min": 0.80, "class_7_min": 0.80},
                    "observed_macro_recall": target_metrics["E_D"]["macro_recall"],
                    "observed_class_1_recall": target_metrics["E_D"]["recall_by_class"][1],
                    "observed_class_7_recall": target_metrics["E_D"]["recall_by_class"][7],
                },
            },
            "historical_replay_comparison": {
                "view_28x28": views_comparison,
                "raw_cmos_50_examples": raw_comparison,
                "historical_raw_to_view_consistency": loaded["historical_raw_view_check"],
                "e_d_historical_view_confusion_matches_original_report": hist_ed_confusion_matches,
                "e_d_replayed_view_confusion_matches_original_report": replay_ed_confusion_matches,
                "historical_ed_confusion": history_metrics["E_D"]["confusion"],
                "replayed_ed_confusion": generated_metrics["E_D"]["confusion"],
                "original_report_ed_confusion": historical_report_confusion,
                "prediction_boundary_flips": {
                    "historical_vs_replay_count": int((history_logits["E_D"].argmax(1) != generated_logits["E_D"].argmax(1)).sum()),
                    "indices": torch.nonzero(history_logits["E_D"].argmax(1) != generated_logits["E_D"].argmax(1)).flatten().tolist(),
                },
                "original_numeric_criteria_passed": original_numeric_passed,
            },
            "risk_analysis": risk,
            "class_1_and_7_readout": {
                "row_details": "class17_row_metrics.json",
                "generated_replay": focus_row_metrics["generated_replay"]["classes"],
                "reference_model_outputs": focus_row_metrics["reference_model_outputs"]["classes"],
                "reference_target_images": focus_row_metrics["reference_target_images"]["classes"],
                "evaluator_disagreement_rows": {
                    "generated_replay": {
                        digit: values["pairwise_evaluator_disagreement_rows"]
                        for digit, values in focus_row_metrics["generated_replay"]["classes"].items()
                    },
                    "reference_model_outputs": {
                        digit: values["pairwise_evaluator_disagreement_rows"]
                        for digit, values in focus_row_metrics["reference_model_outputs"]["classes"].items()
                    },
                },
            },
            "optical_planes_power_and_coordinates": plane_measures,
            "gpu_telemetry": device_telemetry,
        },
        "decision_basis": {
            "replay_tolerance": {"rtol": RTOL, "atol": ATOL},
            "original_thresholds": {"macro_recall_min": 0.50, "per_class_recall_min": 0.10, "raw_fraction_above_one_max": 0.01},
            "replay_matches": bool(views_comparison["matches"] and raw_comparison["matches"]),
            "historical_confusion_matches": bool(hist_ed_confusion_matches and replay_ed_confusion_matches),
            "target_e_d_control_valid": ed_control_valid,
        },
        "artifacts": {
            "generated_chunks": [f"generated_chunk_{index:04d}.pt" for index in range(20)],
            "reference_chunks": [f"reference_chunk_{index:04d}.pt" for index in range(3)],
            "readouts": "final_readouts.pt",
            "nearest_neighbor_rows": "risk_nearest_neighbor_rows.pt",
            "class_contact_sheets": contact_files,
            "raw_comparison_sheet": raw_sheet,
            "nearest_pair_sheet": nearest_sheet,
            "class_1_7_row_metrics": "class17_row_metrics.json",
            "model_state": "frozen_model_state.pt",
            "evaluator_states": "frozen_evaluator_states.pt",
            "harness_incident_records": [
                name
                for name in ["invalid_harness.json", *manifest.get("harness_incidents", [])]
                if (output_dir / name).is_file()
            ],
        },
        "limitations": [
            "The 300 comparison images are fixed canonical TRAIN rows and do not establish held-out generalization.",
            "Nearest-neighbor analysis covers only this 300-row TRAIN reference and does not establish global novelty or absence of memorization.",
            "Classifier scores are diagnostic; image readability still requires human review.",
            "This 400x400 512-grid fiber path is an architecture proxy and is not the current 2048-grid full mainline optical path.",
            "This replay does not validate the historical training process or establish full LightGen semantic generation.",
        ],
        "restricted_reads": {
            "DEV_image_measurements": 0,
            "protected_test_sealed_reads": 0,
            "training_updates": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "preflight_seconds": float(
            json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))["seconds"]
        ),
    }
    _save_json_once(output_dir / "optical_plane_and_power.json", plane_measures)
    _save_json_once(output_dir / "report.json", report)
    manifest["status"] = status
    manifest["classification"] = classification
    manifest["completed_generated_chunks"] = list(range(20))
    manifest["completed_reference_chunks"] = list(range(3))
    manifest["report"] = "report.json"
    manifest.pop("failure", None)
    atomic_write_json(output_dir / "resume_manifest.json", manifest)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--training-evaluator", type=Path, default=DEFAULT_TRAINING_EVALUATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser


def _record_harness_incident(output_dir: Path, failure: str) -> str:
    for path in sorted(output_dir.glob("harness_incident_*.json")):
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("failure") == failure:
            return path.name
    existing_indices = [
        int(path.stem.rsplit("_", 1)[1])
        for path in output_dir.glob("harness_incident_*.json")
        if path.stem.rsplit("_", 1)[-1].isdigit()
    ]
    next_index = max(existing_indices, default=0) + 1
    name = f"harness_incident_{next_index:04d}.json"
    atomic_write_new_json(
        output_dir / name,
        {
            "status": "INVALID_HARNESS",
            "classification": None,
            "failure": failure,
            "training_updates": 0,
        },
    )
    return name


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _resolve_input_paths(args)
    _require(torch.cuda.is_available(), "frozen 400 proxy replay requires CUDA")
    device = torch.device("cuda:0")
    loaded = _load_and_validate_inputs(paths)
    disk_record = _output_disk_check(paths["output_dir"])
    model = LightGenAmplitudeFiberOLSSystemV2(loaded["config"].system).to(device)
    model.load_state_dict(loaded["checkpoint"]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(
        all(parameter.grad is None for parameter in model.parameters()),
        "loaded frozen optical model has gradients",
    )
    evaluators = _load_frozen_evaluators(
        loaded["evaluator_paths"], paths["training_evaluator"], device
    )
    identity = _identity_record(paths, loaded)
    if args.resume_manifest is None:
        _require(not paths["output_dir"].exists(), f"output already exists; explicit --resume-manifest required: {paths['output_dir']}")
        manifest = _write_initial_output(
            paths["output_dir"],
            identity,
            model,
            evaluators,
            loaded["checkpoint"]["sampler"],
            loaded["population"],
        )
        output_created = True
    else:
        _require(paths["output_dir"].is_dir(), "resume output directory is missing")
        manifest = _load_resume_manifest(
            _resolve(ROOT, args.resume_manifest),
            paths["output_dir"],
            identity,
            model,
            evaluators,
            loaded["checkpoint"]["sampler"],
        )
        output_created = False
        report_path = paths["output_dir"] / "report.json"
        if manifest.get("status") in {"VALID_COMPLETED", "VALID_FAILED"} or (
            manifest.get("status") == "INVALID_HARNESS" and report_path.is_file()
        ):
            report_path = _verify_terminal_replay_resume(
                paths["output_dir"],
                manifest,
                loaded["generated_ols"],
                loaded["generated_labels"],
                loaded["row_ids"],
                loaded["reference_labels"],
                loaded["reference_images"],
                model,
                evaluators,
                loaded["checkpoint"]["sampler"],
            )
            terminal_report = json.loads(report_path.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {
                        "status": terminal_report["status"],
                        "classification": terminal_report["classification"],
                        "next_action": terminal_report["next_action"],
                        "training_updates": terminal_report["training_updates"],
                        "report": str(report_path),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
    try:
        previous_failure = manifest.get("failure")
        if isinstance(previous_failure, str) and previous_failure:
            incidents = manifest.setdefault("harness_incidents", [])
            previous_incident = _record_harness_incident(
                paths["output_dir"], previous_failure
            )
            if previous_incident not in incidents:
                incidents.append(previous_incident)
            atomic_write_json(paths["output_dir"] / "resume_manifest.json", manifest)
        preflight_path = paths["output_dir"] / "preflight.json"
        if manifest.get("status") == "PREFLIGHT_PENDING" or not preflight_path.is_file():
            preflight = _run_preflight(
                model,
                evaluators,
                loaded["generated_ols"],
                loaded["reference_images"],
                loaded["reference_labels"],
                device,
            )
            preflight["storage"] = disk_record
            preflight["resume_roundtrip"] = _preflight_resume_roundtrip(
                paths["output_dir"],
                loaded["generated_ols"],
                loaded["generated_labels"],
            )
            _save_json_once(preflight_path, preflight)
            manifest["status"] = "READY_FOR_MEASUREMENT"
            manifest.pop("failure", None)
            atomic_write_json(paths["output_dir"] / "resume_manifest.json", manifest)
        else:
            preflight = _mapping(json.loads(preflight_path.read_text(encoding="utf-8")), "saved preflight")
            _require(preflight.get("status") == "READY" and preflight.get("optical_updates") == 0, "saved preflight does not authorize this frozen replay")
        if args.preflight_only:
            print(json.dumps({"preflight": preflight, "manifest": str(paths["output_dir"] / "resume_manifest.json")}, ensure_ascii=False))
            return 0
        report = _run_measurement(
            args,
            paths,
            loaded,
            model,
            evaluators,
            loaded["checkpoint"]["sampler"],
            manifest,
            device,
        )
        print(json.dumps({"status": report["status"], "classification": report["classification"], "report": str(paths["output_dir"] / "report.json")}, ensure_ascii=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if output_created or (paths["output_dir"] / "resume_manifest.json").exists():
            prior_failure = manifest.get("failure")
            current_failure = f"{type(exc).__name__}: {exc}"
            incident_files = manifest.get("harness_incidents", [])
            if isinstance(prior_failure, str) and prior_failure:
                prior_name = _record_harness_incident(paths["output_dir"], prior_failure)
                if prior_name not in incident_files:
                    incident_files.append(prior_name)
            incident_name = _record_harness_incident(
                paths["output_dir"], current_failure
            )
            if incident_name not in incident_files:
                incident_files.append(incident_name)
            manifest["harness_incidents"] = incident_files
            manifest["status"] = "INVALID_HARNESS"
            manifest["classification"] = None
            manifest["failure"] = current_failure
            try:
                atomic_write_json(paths["output_dir"] / "resume_manifest.json", manifest)
                report_path = paths["output_dir"] / "report.json"
                if report_path.is_file():
                    previous_report = json.loads(report_path.read_text(encoding="utf-8"))
                    previous_report["status"] = "INVALID_HARNESS"
                    previous_report["classification"] = None
                    previous_report["failure"] = manifest["failure"]
                    atomic_write_json(report_path, previous_report)
                atomic_write_new_json(
                    paths["output_dir"] / "invalid_harness.json",
                    {
                        "status": "INVALID_HARNESS",
                        "classification": None,
                        "failure": manifest["failure"],
                        "training_updates": 0,
                    },
                )
            except FileExistsError:
                pass
        raise
    finally:
        del evaluators
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
