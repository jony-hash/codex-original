"""Run the one-shot class-direction transfer diagnosis.

The diagnosis translates fixed DEV complex-OLS states from their source-class
TRAIN centroid to the next class centroid and evaluates the same frozen V4
generator in raw CMOS template space and independent Smooth score space.  It
never creates an optimizer, enables backward, changes parameters, reads a
protected/test/sealed role, or performs generation or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import file_sha256, git_facts, tensor_sha256
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load


TASK_ID = "DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1"
OUTER_GATE = "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED"
REPORT_SCHEMA = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1"
INPUT_SCHEMA = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1"
RAW_OUTPUT_SCHEMA = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_bootstrap.v1"
SUFFICIENT_OBSERVATIONS_SCHEMA = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_sufficient_observations.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_CONSUMER_HEAD = "9a72e0bcf2de12ab9a0caa7151f273de76768dc7"
V4_DIAGNOSTIC_HELPER_PATH = Path("scripts/research/lightgen/diagnose_lightgen_2d_fixed_ols_generator_conditionality_v1.py")
V4_DIAGNOSTIC_HELPER_SHA256 = "e338320d0b64d144a58177217cf8eee6f2ef8ca4855f0554accaa138644f7464"

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_01"
)
INPUT_NAME = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1.json"
REPORT_NAME = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json"
LINEAGE_NAME = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json"
RAW_OUTPUT_NAME = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1.pt"
BOOTSTRAP_NAME = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_bootstrap.v1.pt"
SUFFICIENT_OBSERVATIONS_NAME = "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_sufficient_observations.v1.pt"

CLASS_COUNT = 10
ALPHA = 1.0
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
DEVELOPMENT_SELECTION_ROWS = 1000
DEVELOPMENT_SELECTION_PER_CLASS = 100
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 999
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
OLS_TRANSPORT = torch.eye(100, dtype=torch.complex64)

SUPPORTED_CLASSIFICATION = "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED"
MORPHOLOGY_CLASSIFICATION = "OLS_CLASS_DIRECTION_MORPHOLOGY_ONLY_TRANSFER_SUPPORTED"
NOT_SUPPORTED_CLASSIFICATION = "OLS_CLASS_DIRECTION_TRANSFER_NOT_SUPPORTED"
MIXED_CLASSIFICATION = "OLS_CLASS_DIRECTION_TRANSFER_MIXED_OR_UNRESOLVED"
CLASSIFICATIONS = frozenset(
    {
        SUPPORTED_CLASSIFICATION,
        MORPHOLOGY_CLASSIFICATION,
        NOT_SUPPORTED_CLASSIFICATION,
        MIXED_CLASSIFICATION,
    }
)


class InvalidDiagnosis(RuntimeError):
    """The frozen diagnosis harness or authority is invalid."""


class DiagnosisBlocked(RuntimeError):
    """A valid boundary prevents this one-shot diagnosis."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidDiagnosis(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    """Expose the installed CUDA runtime DLLs to this Python process only."""

    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_python_process_only"}
    cuda_build = torch.version.cuda
    if not isinstance(cuda_build, str) or len(cuda_build.split(".")) != 2 or not all(part.isdigit() for part in cuda_build.split(".")):
        raise DiagnosisBlocked(
            "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED",
            "cuda_dll_search_path",
            "PyTorch CUDA build cannot identify its environment DLL name",
            cuda_build=cuda_build,
        )
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    if not environment_bin.is_dir() or not dll_path.is_file():
        raise DiagnosisBlocked(
            "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED",
            "cuda_dll_search_path",
            "the installed environment does not expose the required NVRTC builtins DLL",
            cuda_build=cuda_build,
            environment_bin=str(environment_bin),
            expected_dll=dll_path.name,
            expected_dll_exists=dll_path.is_file(),
        )
    current_path = os.environ.get("PATH", "")
    normalized_bin = os.path.normcase(os.path.normpath(str(environment_bin)))
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    already_present = any(os.path.normcase(os.path.normpath(entry)) == normalized_bin for entry in path_entries)
    if not already_present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + current_path if current_path else "")
    return {
        "status": "already_present" if already_present else "prepended",
        "scope": "current_python_process_only",
        "cuda_build": cuda_build,
        "environment_bin": str(environment_bin),
        "expected_dll": dll_path.name,
        "expected_dll_exists": True,
        "path_changed": not already_present,
    }


def _load_v4_diagnostic_helper(root: Path) -> Any:
    path = _path(root, V4_DIAGNOSTIC_HELPER_PATH)
    _require(path.is_file(), "reviewed V4 diagnostic helper is unavailable")
    spec = importlib.util.spec_from_file_location("lightgen_v4_conditionality_helper_for_class_direction", path)
    _require(spec is not None and spec.loader is not None, "V4 diagnostic helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _finite_complex(value: torch.Tensor, name: str) -> None:
    _require(torch.is_complex(value), f"{name} is not complex")
    _finite(value.real, f"{name}.real")
    _finite(value.imag, f"{name}.imag")


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _digest_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if torch.is_tensor(item):
            digest.update(prefix.encode("utf-8"))
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{prefix}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")
        else:
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))

    visit(value, "root")
    return digest.hexdigest()


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().to(torch.float64).flatten()
    _finite(values, "summary values")
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "max": float(values.max()),
    }


def _runtime_digest(consumer: Any) -> str:
    return _digest_tree(_cpu_tree(consumer.runtime.state_dict()))


@torch.inference_mode()
def _synthetic_runtime_smoke(_v4: Any, consumer: Any, device: torch.device) -> dict[str, Any]:
    """Exercise the production OLS-to-CMOS path without opening any data role."""

    _require(device.type == "cuda", "synthetic runtime smoke requires the CUDA production device")
    _require(int(consumer.runtime.config.propagation_grid_size) == PRODUCTION_GRID, "synthetic runtime smoke is not using the production propagation grid")
    real = torch.linspace(-0.75, 0.75, 200, dtype=torch.float32).reshape(2, 100)
    imag = torch.linspace(0.75, -0.75, 200, dtype=torch.float32).reshape(2, 100)
    z = torch.complex(real, imag).to(device=device).contiguous()
    before = _runtime_digest(consumer)
    try:
        result = consumer.runtime.forward_from_ols(z)
        raw = result.raw_camera.tensor
        torch.cuda.synchronize(device)
    except Exception as error:
        diagnostic_text = " ".join(str(error).splitlines())
        if "nvrtc-builtins64" in diagnostic_text.lower():
            raise DiagnosisBlocked(
                "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED",
                "synthetic_runtime_smoke",
                "the actual LightGen optical runtime still cannot execute its CUDA complex path",
                error_type=type(error).__name__,
                error=diagnostic_text,
                data_roles_opened=False,
            ) from error
        raise
    after = _runtime_digest(consumer)
    _require(result.z_ols.source == "external_generated", "synthetic OLS state crossed the image encoder boundary")
    _require(tuple(result.z_ols.coefficients.shape) == (2, 100) and result.z_ols.coefficients.dtype == torch.complex64, "synthetic OLS state shape/dtype changed")
    generator_input = result.generator_input_field.tensor
    _finite_complex(generator_input, "synthetic generator input field")
    _require(generator_input.device == device and tuple(generator_input.shape[-2:]) == (100, 100), "synthetic modal re-emission output changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "synthetic path did not use shared W_OLS")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "synthetic path did not use T=I_100")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), OLS_TRANSPORT), "synthetic path changed the fixed OLS transport")
    _require(len(result.generator_records) == 2, "synthetic path did not execute the two generator phase layers")
    _require(tuple(raw.shape) == (2, 1, 200, 200) and raw.dtype == torch.float32 and raw.device == device, "synthetic raw CMOS shape/dtype/device changed")
    _finite(raw, "synthetic raw CMOS")
    raw_std = float(raw.detach().std(unbiased=False).cpu())
    _require(raw_std > 0.0, "synthetic raw CMOS output is constant")
    _require(before == after, "synthetic runtime smoke changed the runtime state")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "synthetic runtime smoke produced a parameter gradient")
    return {
        "status": "passed",
        "data_role": "synthetic_no_data",
        "data_roles_opened": False,
        "input": {"shape": list(z.shape), "dtype": str(z.dtype), "device": str(z.device), "finite": True},
        "runtime": {
            "propagation_grid": int(consumer.runtime.config.propagation_grid_size),
            "modal_projection_reemission": True,
            "transport": "T=I_100_coefficient_space",
            "shared_w_ols": True,
            "generator_layers": len(result.generator_records),
        },
        "raw_cmos": {"shape": list(raw.shape), "dtype": str(raw.dtype), "device": str(raw.device), "finite": True, "nonconstant": True, "std": raw_std},
        "state_unchanged": True,
        "gradients_none": True,
        "optimizer_created": False,
        "backward_executed": False,
        "optical_update_executed": False,
    }


def _validate_z(z: torch.Tensor, batch_size: int, name: str) -> None:
    _require(tuple(z.shape) == (batch_size, 100) and z.dtype == torch.complex64, f"{name} shape/dtype changed")
    _finite_complex(z, name)


def _validate_runtime_result(result: Any, batch_size: int, *, external: bool) -> torch.Tensor:
    if external:
        _require(
            result.image_input is None
            and result.ols_input_field is None
            and result.z_ols.source == "external_generated",
            "external path crossed image/encoder boundary",
        )
    else:
        _require(
            result.image_input is not None
            and result.ols_input_field is not None
            and result.z_ols.source == "real_encoded",
            "canonical path lost encoder boundary",
        )
        _require(
            result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True,
            "modal projection boundary is inactive",
        )
    _require(
        result.generator_input_field.provenance.operation
        == "lightgen_ols_coefficient_transmission_and_modal_reemission",
        "W_OLS provenance changed",
    )
    _require(
        result.generator_input_field.provenance.parameters.get("transport")
        == "T=I_100_coefficient_space",
        "T=I_100 provenance changed",
    )
    raw = result.raw_intensity
    _require(
        tuple(raw.shape) == (batch_size, 1, 200, 200) and raw.dtype == torch.float32,
        "raw CMOS shape/dtype changed",
    )
    _finite(raw, "raw CMOS")
    return raw


@torch.inference_mode()
def _encode_train_z(consumer: Any, train_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    before = _runtime_digest(consumer)
    parts: list[torch.Tensor] = []
    for start in range(0, train_images.shape[0], LOGICAL_BATCH_SIZE):
        batch = train_images[start : start + LOGICAL_BATCH_SIZE].to(device)
        result = consumer.runtime.forward_from_image(batch)
        raw = _validate_runtime_result(result, int(batch.shape[0]), external=False)
        z = result.z_ols.coefficients.detach()
        _validate_z(z, int(batch.shape[0]), "TRAIN z")
        parts.append(z.cpu().contiguous())
        del batch, result, raw, z
    torch.cuda.synchronize(device)
    after = _runtime_digest(consumer)
    _require(before == after, "TRAIN encoder forward changed the frozen runtime state")
    result = torch.cat(parts, dim=0)
    _require(tuple(result.shape) == (55_000, 100) and result.dtype == torch.complex64, "TRAIN z collection shape/dtype changed")
    _finite_complex(result, "TRAIN z collection")
    return result


@torch.inference_mode()
def _collect_dev_base(consumer: Any, selection_images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    before = _runtime_digest(consumer)
    raw_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    max_replay_difference = 0.0
    for start in range(0, selection_images.shape[0], LOGICAL_BATCH_SIZE):
        batch = selection_images[start : start + LOGICAL_BATCH_SIZE].to(device)
        canonical = consumer.runtime.forward_from_image(batch)
        raw = _validate_runtime_result(canonical, int(batch.shape[0]), external=False)
        z = canonical.z_ols.coefficients.detach()
        _validate_z(z, int(batch.shape[0]), "DEV canonical z")
        replay = consumer.runtime.forward_from_ols(z)
        replay_raw = _validate_runtime_result(replay, int(batch.shape[0]), external=True)
        max_replay_difference = max(max_replay_difference, float((raw - replay_raw).abs().max().cpu()))
        raw_parts.append(raw.cpu().contiguous())
        z_parts.append(z.cpu().contiguous())
        del batch, canonical, raw, z, replay, replay_raw
    torch.cuda.synchronize(device)
    after = _runtime_digest(consumer)
    _require(before == after, "DEV canonical forward changed the frozen runtime state")
    raw_cpu = torch.cat(raw_parts, dim=0)
    z_cpu = torch.cat(z_parts, dim=0)
    _require(tuple(raw_cpu.shape) == (DEVELOPMENT_SELECTION_ROWS, 1, 200, 200), "DEV base raw shape changed")
    _require(tuple(z_cpu.shape) == (DEVELOPMENT_SELECTION_ROWS, 100) and z_cpu.dtype == torch.complex64, "DEV base z shape/dtype changed")
    _finite(raw_cpu, "DEV base raw CMOS")
    _finite_complex(z_cpu, "DEV base z")
    return raw_cpu, z_cpu, max_replay_difference


@torch.inference_mode()
def _forward_shifted(consumer: Any, z_shift: torch.Tensor, device: torch.device) -> torch.Tensor:
    before = _runtime_digest(consumer)
    parts: list[torch.Tensor] = []
    for start in range(0, z_shift.shape[0], LOGICAL_BATCH_SIZE):
        batch_z = z_shift[start : start + LOGICAL_BATCH_SIZE].to(device)
        result = consumer.runtime.forward_from_ols(batch_z)
        raw = _validate_runtime_result(result, int(batch_z.shape[0]), external=True)
        parts.append(raw.cpu().contiguous())
        del batch_z, result, raw
    torch.cuda.synchronize(device)
    after = _runtime_digest(consumer)
    _require(before == after, "shifted forward changed the frozen runtime state")
    result = torch.cat(parts, dim=0)
    _require(tuple(result.shape) == (DEVELOPMENT_SELECTION_ROWS, 1, 200, 200) and result.dtype == torch.float32, "DEV shifted raw shape/dtype changed")
    _finite(result, "DEV shifted raw CMOS")
    return result


def _train_centroids(z_train: torch.Tensor, train_labels: torch.Tensor, *, expected_rows: int | None = None) -> torch.Tensor:
    _require(z_train.ndim == 2 and z_train.shape[1] == 100, "TRAIN z shape changed")
    if expected_rows is not None:
        _require(z_train.shape[0] == expected_rows, "TRAIN z row count changed")
    _require(train_labels.shape == (z_train.shape[0],), "TRAIN z labels shape changed")
    _require(train_labels.dtype == torch.long, "TRAIN z labels dtype changed")
    _finite(train_labels.to(torch.float32), "TRAIN z labels")
    _finite_complex(z_train, "TRAIN z")
    centroids: list[torch.Tensor] = []
    counts: list[int] = []
    for digit in range(CLASS_COUNT):
        values = z_train[train_labels == digit]
        _require(values.shape[0] > 0, f"TRAIN class centroid is undefined for class {digit}")
        counts.append(int(values.shape[0]))
        centroids.append(values.mean(dim=0))
    result = torch.stack(centroids, dim=0).contiguous()
    _require(tuple(result.shape) == (CLASS_COUNT, 100) and result.dtype == torch.complex64, "TRAIN complex centroid shape/dtype changed")
    _finite_complex(result, "TRAIN complex centroids")
    return result


def _train_templates(train_images: torch.Tensor, train_labels: torch.Tensor, *, expected_rows: int | None = None) -> torch.Tensor:
    _require(train_images.ndim == 4 and tuple(train_images.shape[1:]) == (1, 28, 28) and train_images.dtype == torch.float32, "TRAIN template image shape/dtype changed")
    if expected_rows is not None:
        _require(train_images.shape[0] == expected_rows, "TRAIN template image row count changed")
    _require(train_labels.shape == (train_images.shape[0],), "TRAIN template labels shape changed")
    _require(train_labels.dtype == torch.long, "TRAIN template labels dtype changed")
    _finite(train_labels.to(torch.float32), "TRAIN template labels")
    templates: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        values = train_images[train_labels == digit]
        _require(values.shape[0] > 0, f"TRAIN image template is undefined for class {digit}")
        templates.append(values.mean(dim=0))
    result = torch.stack(templates, dim=0).contiguous()
    _require(tuple(result.shape) == (CLASS_COUNT, 1, 28, 28) and result.dtype == torch.float32, "TRAIN image template shape/dtype changed")
    _finite(result, "TRAIN image templates")
    return result


def _template_components(
    base_raw: torch.Tensor,
    shifted_raw: torch.Tensor,
    templates: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_view = F.adaptive_avg_pool2d(base_raw.clamp(0.0, 1.0), (28, 28))
    shifted_view = F.adaptive_avg_pool2d(shifted_raw.clamp(0.0, 1.0), (28, 28))
    source_templates = templates.index_select(0, source_labels)
    target_templates = templates.index_select(0, target_labels)
    base_source = (base_view.to(torch.float64) - source_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    base_target = (base_view.to(torch.float64) - target_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    shifted_source = (shifted_view.to(torch.float64) - source_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    shifted_target = (shifted_view.to(torch.float64) - target_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    base_margin = base_source - base_target
    shifted_margin = shifted_source - shifted_target
    delta = shifted_margin - base_margin
    _finite(delta, "Delta_template")
    return base_margin, shifted_margin, delta


def _template_direction(
    base_raw: torch.Tensor,
    shifted_raw: torch.Tensor,
    templates: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_margin, shifted_margin, delta = _template_components(
        base_raw,
        shifted_raw,
        templates,
        source_labels,
        target_labels,
    )
    return delta, {
        "distance": "mean squared pixel distance after raw CMOS clamp [0,1] and adaptive_avg_pool2d to 28x28",
        "base_margin_definition": "d(base, tau_source) - d(base, tau_target)",
        "shifted_margin_definition": "d(shift, tau_source) - d(shift, tau_target)",
        "delta_definition": "M_template(raw_shift) - M_template(raw_base)",
        "base_margin_summary": _summary(base_margin),
        "shifted_margin_summary": _summary(shifted_margin),
        "delta_summary": _summary(delta),
    }


def _smooth_logits(smooth: Any, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, raw.shape[0], LOGICAL_BATCH_SIZE):
        batch = raw[start : start + LOGICAL_BATCH_SIZE].to(device)
        view = F.adaptive_avg_pool2d(batch.clamp(0.0, 1.0), (28, 28))
        _, logits = smooth(view)
        _require(tuple(logits.shape) == (int(batch.shape[0]), CLASS_COUNT), "Smooth logits shape changed")
        _require(not torch.is_complex(logits), "Smooth logits unexpectedly complex")
        _finite(logits, "Smooth logits")
        parts.append(logits.cpu().contiguous())
        del batch, view, logits
    torch.cuda.synchronize(device)
    result = torch.cat(parts, dim=0)
    _require(tuple(result.shape) == (DEVELOPMENT_SELECTION_ROWS, CLASS_COUNT), "Smooth logits collection shape changed")
    _finite(result, "Smooth logits collection")
    return result


def _class_score_margin(logits: torch.Tensor, source_labels: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], dtype=torch.long)
    margin = logits[rows, target_labels] - logits[rows, source_labels]
    _finite(margin, "Smooth target-vs-source margin")
    return margin


def _smooth_components(
    base_logits: torch.Tensor,
    shifted_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_margin = _class_score_margin(base_logits, source_labels, target_labels)
    shifted_margin = _class_score_margin(shifted_logits, source_labels, target_labels)
    delta = shifted_margin - base_margin
    _finite(delta, "Delta_smooth")
    return base_margin, shifted_margin, delta


def _ranks(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, CLASS_COUNT + 1, dtype=torch.long).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    result = ranks[torch.arange(scores.shape[0]), labels]
    _require(result.dtype == torch.long and tuple(result.shape) == (scores.shape[0],), "score rank shape changed")
    return result


def _prediction_stats(logits: torch.Tensor, source_labels: torch.Tensor, target_labels: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    return {
        "prediction_tensor_sha256": tensor_sha256(predictions),
        "target_prediction_rate": float((predictions == target_labels).to(torch.float64).mean()),
        "source_prediction_rate": float((predictions == source_labels).to(torch.float64).mean()),
        "prediction_histogram": [int(value) for value in torch.bincount(predictions, minlength=CLASS_COUNT).tolist()],
    }


def _smooth_direction(
    base_logits: torch.Tensor,
    shifted_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_margin, shifted_margin, delta = _smooth_components(
        base_logits,
        shifted_logits,
        source_labels,
        target_labels,
    )
    target_base_rank = _ranks(base_logits, target_labels)
    target_shifted_rank = _ranks(shifted_logits, target_labels)
    source_base_rank = _ranks(base_logits, source_labels)
    source_shifted_rank = _ranks(shifted_logits, source_labels)
    target_rank_change = target_shifted_rank.to(torch.float64) - target_base_rank.to(torch.float64)
    source_rank_change = source_shifted_rank.to(torch.float64) - source_base_rank.to(torch.float64)
    return delta, {
        "score_representation": "SmoothMNISTEvaluator v2.1 canonical pre-softmax logits",
        "base_margin_definition": "score_target - score_source",
        "shifted_margin_definition": "score_target - score_source",
        "delta_definition": "M_smooth(raw_shift) - M_smooth(raw_base)",
        "base_margin_summary": _summary(base_margin),
        "shifted_margin_summary": _summary(shifted_margin),
        "delta_summary": _summary(delta),
        "target_rank": {
            "rank_definition": "1-based descending logit rank; stable ties preserve lower class index",
            "base_summary": _summary(target_base_rank),
            "shifted_summary": _summary(target_shifted_rank),
            "change_shifted_minus_base_summary": _summary(target_rank_change),
            "improvement_base_minus_shifted_summary": _summary(-target_rank_change),
        },
        "source_rank": {
            "rank_definition": "1-based descending logit rank; stable ties preserve lower class index",
            "base_summary": _summary(source_base_rank),
            "shifted_summary": _summary(source_shifted_rank),
            "change_shifted_minus_base_summary": _summary(source_rank_change),
        },
        "base_predictions": _prediction_stats(base_logits, source_labels, target_labels),
        "shifted_predictions": _prediction_stats(shifted_logits, source_labels, target_labels),
    }


def _group_summary(values: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for digit in range(CLASS_COUNT):
        selected = values[labels == digit]
        _require(selected.numel() > 0, f"group {digit} is empty")
        result.append(
            {
                "class": digit,
                "sample_count": int(selected.numel()),
                "delta_summary": _summary(selected),
                "positive_fraction": float((selected > 0.0).to(torch.float64).mean()),
            }
        )
    return result


def _pair_summary(
    delta_template: torch.Tensor,
    delta_smooth: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    samples_per_class: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in range(CLASS_COUNT):
        target = (source + 1) % CLASS_COUNT
        mask = (source_labels == source) & (target_labels == target)
        template_values = delta_template[mask]
        smooth_values = delta_smooth[mask]
        _require(template_values.numel() == samples_per_class, f"pair {source}->{target} sample count changed")
        result.append(
            {
                "source_class": source,
                "target_class": target,
                "sample_count": int(template_values.numel()),
                "delta_template_mean": float(template_values.mean()),
                "delta_smooth_mean": float(smooth_values.mean()),
                "positive_template_fraction": float((template_values > 0.0).to(torch.float64).mean()),
                "positive_smooth_fraction": float((smooth_values > 0.0).to(torch.float64).mean()),
            }
        )
    return result


def _bootstrap_deltas(
    deltas: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    samples_per_class: int,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws: list[torch.Tensor] = []
    values: dict[str, list[torch.Tensor]] = {name: [] for name in deltas}
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == samples_per_class, f"bootstrap class count changed for {digit}")
        draw = torch.randint(0, samples_per_class, (replicates, samples_per_class), generator=generator)
        draws.append(draw)
        for name, delta in deltas.items():
            values[name].append(delta.index_select(0, matches)[draw].mean(dim=1))
    macro: dict[str, torch.Tensor] = {
        name: torch.stack(items, dim=0).mean(dim=0) for name, items in values.items()
    }
    summaries: dict[str, Any] = {}
    for name, sample in macro.items():
        q = torch.quantile(
            sample.to(torch.float64),
            torch.tensor([0.025, 0.975], dtype=torch.float64),
            interpolation="linear",
        )
        summaries[name] = {
            "mean": float(sample.mean()),
            "ci_lower": float(q[0]),
            "ci_upper": float(q[1]),
            "tensor_sha256": tensor_sha256(sample),
        }
    return summaries, {"draws": torch.stack(draws, dim=0), "macro_deltas": macro}


def _raw_summary(
    raw: torch.Tensor,
    *,
    expected_batch_size: int | None = None,
    expected_spatial: tuple[int, int] = (200, 200),
) -> dict[str, Any]:
    detached = raw.detach()
    per_image = detached.flatten(1)
    variance = per_image.var(dim=1, unbiased=False)
    if expected_batch_size is not None:
        _require(tuple(detached.shape) == (expected_batch_size, 1, *expected_spatial), "raw summary shape changed")
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(torch.isfinite(detached).all()),
        "mean": float(detached.mean().cpu()),
        "std": float(detached.std(unbiased=False).cpu()),
        "per_image_variance_min": float(variance.min().cpu()),
        "per_image_variance_max": float(variance.max().cpu()),
        "structured": bool((variance > 0.0).all()),
        "nonconstant": bool(float(detached.std(unbiased=False).cpu()) > 0.0),
        "tensor_sha256": tensor_sha256(detached),
    }


def _raw_change_values(base_raw: torch.Tensor, shifted_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    difference = shifted_raw.to(torch.float64) - base_raw.to(torch.float64)
    absolute = difference.square().flatten(1).mean(dim=1).sqrt()
    base_rms = base_raw.to(torch.float64).square().flatten(1).mean(dim=1).sqrt()
    relative = absolute / base_rms.clamp_min(1.0e-12)
    _finite(absolute, "absolute raw RMS change")
    _finite(relative, "relative raw RMS change")
    return absolute, relative


def _raw_change_summary(
    base_raw: torch.Tensor,
    shifted_raw: torch.Tensor,
    *,
    expected_batch_size: int | None = None,
) -> dict[str, Any]:
    absolute, relative = _raw_change_values(base_raw, shifted_raw)
    return {
        "absolute_definition": "RMS(raw_shift - raw_base) over the 200x200 raw CMOS pixels",
        "relative_definition": "RMS(raw_shift - raw_base) / RMS(raw_base), per sample",
        "absolute_rms_change": {"summary": _summary(absolute), "tensor_sha256": tensor_sha256(absolute)},
        "relative_rms_change": {"summary": _summary(relative), "tensor_sha256": tensor_sha256(relative)},
        "raw_base": _raw_summary(base_raw, expected_batch_size=expected_batch_size),
        "raw_shifted": _raw_summary(shifted_raw, expected_batch_size=expected_batch_size),
    }


def _bootstrap(
    delta_template: torch.Tensor,
    delta_smooth: torch.Tensor,
    labels: torch.Tensor,
    *,
    samples_per_class: int,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    summaries, tensors = _bootstrap_deltas(
        {"delta_template": delta_template, "delta_smooth": delta_smooth},
        labels,
        samples_per_class=samples_per_class,
        replicates=replicates,
        seed=seed,
    )
    _require(set(summaries) == {"delta_template", "delta_smooth"}, "bootstrap metrics changed")
    return summaries, tensors


def _save_raw_artifact(
    root: Path,
    target: Path,
    centroids: torch.Tensor,
    templates: torch.Tensor,
    z: torch.Tensor,
    z_shift: torch.Tensor,
    base_raw: torch.Tensor,
    shifted_raw: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    base_logits: torch.Tensor,
    shifted_logits: torch.Tensor,
    selection_indices: torch.Tensor,
) -> dict[str, Any]:
    path = target / RAW_OUTPUT_NAME
    payload = {
        "schema": RAW_OUTPUT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "selection_indices_sha256": tensor_sha256(selection_indices),
        "source_labels": source_labels,
        "target_labels": target_labels,
        "centroids": centroids,
        "templates": templates,
        "z": z,
        "z_shift": z_shift,
        "raw_base": base_raw,
        "raw_shifted": shifted_raw,
        "smooth_base_logits": base_logits,
        "smooth_shifted_logits": shifted_logits,
    }
    atomic_torch_save_new(path, payload, artifact_kind="OLS class-direction transfer raw outputs")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "class-direction raw artifact")
    for key, value in payload.items():
        if torch.is_tensor(value):
            _require(torch.is_tensor(loaded.get(key)) and tensor_sha256(loaded[key]) == tensor_sha256(value), f"raw artifact reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": RAW_OUTPUT_SCHEMA, "selection_indices_sha256": tensor_sha256(selection_indices)}


def _save_bootstrap_artifact(root: Path, target: Path, summaries: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], selection_labels: torch.Tensor) -> dict[str, Any]:
    path = target / BOOTSTRAP_NAME
    payload = {
        "schema": BOOTSTRAP_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "selection_labels_sha256": tensor_sha256(selection_labels),
        "protocol": {
            "type": "paired, class-stratified bootstrap over 100 examples per source class",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "quantile_convention": "torch.quantile interpolation=linear",
        },
        "summaries": dict(summaries),
        "draws": tensors["draws"],
        "macro_deltas": tensors["macro_deltas"],
    }
    atomic_torch_save_new(path, payload, artifact_kind="OLS class-direction transfer bootstrap")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "class-direction bootstrap artifact")
    _require(loaded.get("schema") == BOOTSTRAP_SCHEMA and loaded.get("protocol", {}).get("seed") == BOOTSTRAP_SEED, "bootstrap artifact reload changed")
    _require(torch.is_tensor(loaded.get("draws")) and tensor_sha256(loaded["draws"]) == tensor_sha256(tensors["draws"]), "bootstrap draws reload changed")
    return {"path": _relative(root, path), "sha256": digest, "schema": BOOTSTRAP_SCHEMA, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "summaries": dict(summaries)}


def _save_sufficient_observations(
    root: Path,
    target: Path,
    observations: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = target / SUFFICIENT_OBSERVATIONS_NAME
    tensor_payload = {
        name: value.detach().cpu().contiguous()
        for name, value in observations.items()
    }
    _require(tensor_payload, "sufficient-observation payload is empty")
    payload = {
        "schema": SUFFICIENT_OBSERVATIONS_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "protocol": {
            "purpose": "reproduce frozen raw/template/Smooth aggregate statistics without optical scientific forward",
            "target_rule": "target=(source+1) mod 10",
            "alpha": ALPHA,
            "renormalization": "none",
            "bootstrap": {
                "replicates": int(metadata["bootstrap_replicates"]),
                "seed": int(metadata["bootstrap_seed"]),
                "type": "paired, class-stratified",
            },
            "observation_tensors": sorted(tensor_payload),
        },
        "metadata": dict(metadata),
        "observations": tensor_payload,
    }
    atomic_torch_save_new(path, payload, artifact_kind="OLS class-direction sufficient observations")
    digest = file_sha256(path)
    loaded = _mapping(
        safe_torch_load(path, expected_sha256=digest, map_location="cpu"),
        "class-direction sufficient observations",
    )
    _require(loaded.get("schema") == SUFFICIENT_OBSERVATIONS_SCHEMA, "sufficient-observation schema changed on reload")
    loaded_observations = _mapping(loaded.get("observations"), "reloaded sufficient observations")
    _require(set(loaded_observations) == set(tensor_payload), "sufficient-observation tensor set changed on reload")
    for name, value in tensor_payload.items():
        reloaded = loaded_observations.get(name)
        _require(torch.is_tensor(reloaded) and tensor_sha256(reloaded) == tensor_sha256(value), f"sufficient observation changed on reload: {name}")
    return {
        "path": _relative(root, path),
        "sha256": digest,
        "schema": SUFFICIENT_OBSERVATIONS_SCHEMA,
        "observation_tensors": sorted(tensor_payload),
    }


def _classify(
    template_summary: Mapping[str, Any],
    smooth_summary: Mapping[str, Any],
    pair_summaries: list[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    template_mean = float(template_summary["mean"])
    template_lower = float(template_summary["ci_lower"])
    smooth_mean = float(smooth_summary["mean"])
    smooth_lower = float(smooth_summary["ci_lower"])
    template_pass = template_mean > 0.0 and template_lower > 0.0
    smooth_pass = smooth_mean > 0.0 and smooth_lower > 0.0
    positive_template_pairs = sum(float(item["delta_template_mean"]) > 0.0 for item in pair_summaries)
    positive_smooth_pairs = sum(float(item["delta_smooth_mean"]) > 0.0 for item in pair_summaries)
    # The contract's "not one isolated pair" condition is frozen as at least
    # two positive pair means for every evidence space used by a classification.
    template_broad = positive_template_pairs >= 2
    smooth_broad = positive_smooth_pairs >= 2
    pair_dominated = (
        (template_mean > 0.0 and positive_template_pairs == 1)
        or (smooth_mean > 0.0 and positive_smooth_pairs == 1)
    )
    if template_pass and smooth_pass and template_broad and smooth_broad:
        classification = SUPPORTED_CLASSIFICATION
    elif template_pass and not smooth_pass and template_broad:
        classification = MORPHOLOGY_CLASSIFICATION
    elif not template_pass and not smooth_pass and not pair_dominated:
        classification = NOT_SUPPORTED_CLASSIFICATION
    else:
        classification = MIXED_CLASSIFICATION
    return classification, {
        "template_primary_pass": template_pass,
        "smooth_primary_pass": smooth_pass,
        "positive_template_pair_means": int(positive_template_pairs),
        "positive_smooth_pair_means": int(positive_smooth_pairs),
        "pair_broadness_rule": "at least two of ten frozen y->(y+1 mod 10) pair means must be positive",
        "template_broad": template_broad,
        "smooth_broad": smooth_broad,
        "pair_dominated": pair_dominated,
        "primary_evidence_rule": "mean > 0 and paired bootstrap 95% CI lower bound > 0",
    }


def run_class_direction_diagnosis_core(
    *,
    train_z: torch.Tensor,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    selection_images: torch.Tensor,
    selection_labels: torch.Tensor,
    canonical_forward: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]],
    shifted_forward: Callable[[torch.Tensor], torch.Tensor],
    smooth_forward: Callable[[torch.Tensor], torch.Tensor],
    selection_indices: torch.Tensor | None = None,
    samples_per_class: int = DEVELOPMENT_SELECTION_PER_CLASS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    persist_sufficient_observations: Callable[[Mapping[str, torch.Tensor], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the frozen class-direction computation through injectable forward paths.

    Production supplies the real LightGen encoder/generator callbacks; synthetic
    acceptance supplies deterministic callbacks without opening project data.
    The intervention, metrics, bootstrap and classification remain identical for
    the production defaults.
    """

    _require(samples_per_class > 0 and bootstrap_replicates > 0, "synthetic/core sample or bootstrap count is invalid")
    _require(train_z.ndim == 2 and train_z.shape[1] == 100 and train_z.dtype == torch.complex64, "core TRAIN z shape/dtype changed")
    _require(train_images.ndim == 4 and tuple(train_images.shape[1:]) == (1, 28, 28) and train_images.dtype == torch.float32, "core TRAIN image shape/dtype changed")
    _require(train_labels.ndim == 1 and train_labels.shape[0] == train_z.shape[0] == train_images.shape[0] and train_labels.dtype == torch.long, "core TRAIN role shape/dtype changed")
    _require(selection_images.ndim == 4 and tuple(selection_images.shape[1:]) == (1, 28, 28) and selection_images.dtype == torch.float32, "core DEV image shape/dtype changed")
    _require(selection_labels.ndim == 1 and selection_labels.shape[0] == selection_images.shape[0] and selection_labels.dtype == torch.long, "core DEV role shape/dtype changed")
    _require(selection_labels.shape[0] == CLASS_COUNT * samples_per_class, "core DEV selection row count changed")
    _finite_complex(train_z, "core TRAIN z")
    _finite(train_images, "core TRAIN images")
    _finite(train_labels.to(torch.float32), "core TRAIN labels")
    _finite(selection_images, "core DEV images")
    _finite(selection_labels.to(torch.float32), "core DEV labels")
    for digit in range(CLASS_COUNT):
        _require(int((train_labels == digit).sum()) > 0, f"core TRAIN class {digit} is empty")
        _require(int((selection_labels == digit).sum()) == samples_per_class, f"core DEV class count changed for {digit}")

    if selection_indices is None:
        selection_indices = torch.arange(selection_labels.shape[0], dtype=torch.long)
    _require(selection_indices.shape == selection_labels.shape and selection_indices.dtype == torch.long, "core selection identity shape/dtype changed")

    centroids = _train_centroids(train_z, train_labels, expected_rows=train_z.shape[0])
    templates = _train_templates(train_images, train_labels, expected_rows=train_images.shape[0])
    target_labels = ((selection_labels + 1) % CLASS_COUNT).contiguous()
    canonical_result = canonical_forward(selection_images)
    _require(isinstance(canonical_result, tuple) and len(canonical_result) == 3, "canonical forward callback schema changed")
    z, base_raw, canonical_metadata = canonical_result
    _require(isinstance(canonical_metadata, Mapping), "canonical forward metadata schema changed")
    _validate_z(z, int(selection_labels.shape[0]), "core DEV canonical z")
    _require(tuple(base_raw.shape) == (selection_labels.shape[0], 1, 200, 200) and base_raw.dtype == torch.float32, "core DEV base raw shape/dtype changed")
    _finite(base_raw, "core DEV base raw CMOS")
    shifted_source = centroids.index_select(0, selection_labels)
    shifted_target = centroids.index_select(0, target_labels)
    direction = (shifted_target - shifted_source).contiguous()
    z_shift = (z - shifted_source + (ALPHA * direction)).contiguous()
    _validate_z(z_shift, int(selection_labels.shape[0]), "core DEV shifted z")
    shifted_raw = shifted_forward(z_shift)
    _require(tuple(shifted_raw.shape) == tuple(base_raw.shape) and shifted_raw.dtype == torch.float32, "core DEV shifted raw shape/dtype changed")
    _finite(shifted_raw, "core DEV shifted raw CMOS")
    base_logits = smooth_forward(base_raw)
    shifted_logits = smooth_forward(shifted_raw)
    for name, logits in (("core base", base_logits), ("core shifted", shifted_logits)):
        _require(tuple(logits.shape) == (selection_labels.shape[0], CLASS_COUNT) and not torch.is_complex(logits), f"{name} Smooth logits shape/dtype changed")
        _finite(logits, f"{name} Smooth logits")

    raw_detail = _raw_change_summary(base_raw, shifted_raw, expected_batch_size=int(selection_labels.shape[0]))
    absolute_rms, relative_rms = _raw_change_values(base_raw, shifted_raw)
    template_delta, template_detail = _template_direction(base_raw, shifted_raw, templates, selection_labels, target_labels)
    template_base_margin, template_shifted_margin, template_delta_observed = _template_components(base_raw, shifted_raw, templates, selection_labels, target_labels)
    smooth_delta, smooth_detail = _smooth_direction(base_logits, shifted_logits, selection_labels, target_labels)
    smooth_base_margin, smooth_shifted_margin, smooth_delta_observed = _smooth_components(base_logits, shifted_logits, selection_labels, target_labels)
    _require(torch.equal(template_delta, template_delta_observed), "template observation delta changed")
    _require(torch.equal(smooth_delta, smooth_delta_observed), "Smooth observation delta changed")
    z_norm = torch.linalg.vector_norm(z.to(torch.complex128), dim=1)
    z_shift_norm = torch.linalg.vector_norm(z_shift.to(torch.complex128), dim=1)
    direction_norm = torch.linalg.vector_norm(direction.to(torch.complex128), dim=1)
    _finite(z_norm, "core z norm")
    _finite(z_shift_norm, "core shifted z norm")
    _finite(direction_norm, "core centroid direction norm")
    observation_tensors = {
        "selection_indices": selection_indices,
        "source_labels": selection_labels,
        "target_labels": target_labels,
        "z_norm": z_norm,
        "z_shift_norm": z_shift_norm,
        "centroid_direction_norm": direction_norm,
        "raw_absolute_rms_change": absolute_rms,
        "raw_relative_rms_change": relative_rms,
        "template_base_margin": template_base_margin,
        "template_shifted_margin": template_shifted_margin,
        "smooth_base_logits": base_logits,
        "smooth_shifted_logits": shifted_logits,
    }
    observation_metadata = {
        "samples_per_class": samples_per_class,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "canonical_replay_max_abs_difference": float(canonical_metadata.get("canonical_replay_max_abs_difference", 0.0)),
        "raw_base_summary": raw_detail["raw_base"],
        "raw_shifted_summary": raw_detail["raw_shifted"],
    }
    observation_artifact = None
    if persist_sufficient_observations is not None:
        observation_artifact = dict(persist_sufficient_observations(observation_tensors, observation_metadata))

    bootstrap_summaries, bootstrap_tensors = _bootstrap(
        template_delta,
        smooth_delta,
        selection_labels,
        samples_per_class=samples_per_class,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    pair_summaries = _pair_summary(
        template_delta,
        smooth_delta,
        selection_labels,
        target_labels,
        samples_per_class=samples_per_class,
    )
    template_groups = {
        "by_source_class": _group_summary(template_delta, selection_labels),
        "by_target_class": _group_summary(template_delta, target_labels),
    }
    smooth_groups = {
        "by_source_class": _group_summary(smooth_delta, selection_labels),
        "by_target_class": _group_summary(smooth_delta, target_labels),
    }
    classification, decision = _classify(
        bootstrap_summaries["delta_template"],
        bootstrap_summaries["delta_smooth"],
        pair_summaries,
    )
    _require(classification in CLASSIFICATIONS, "core classification is not one of the frozen primary classifications")
    report_fields = {
        "ols_intervention": {
            "class_centroids_constructed": True,
            "centroid_rule": "all frozen TRAIN complex z grouped by label; class mean in torch.complex64",
            "class_counts": [int((train_labels == digit).sum()) for digit in range(CLASS_COUNT)],
            "centroid_tensor_sha256": tensor_sha256(centroids),
            "z_tensor_sha256": tensor_sha256(z),
            "z_shift_tensor_sha256": tensor_sha256(z_shift),
            "z_norm": _summary(z_norm),
            "z_shift_norm": _summary(z_shift_norm),
            "centroid_direction_norm": _summary(direction_norm),
            "finite_domain_validity": {
                "z_finite": bool(torch.isfinite(z.real).all() and torch.isfinite(z.imag).all()),
                "z_shift_finite": bool(torch.isfinite(z_shift.real).all() and torch.isfinite(z_shift.imag).all()),
                "centroids_finite": bool(torch.isfinite(centroids.real).all() and torch.isfinite(centroids.imag).all()),
                "templates_finite": bool(torch.isfinite(templates).all()),
                "raw_base_finite": bool(torch.isfinite(base_raw).all()),
                "raw_shifted_finite": bool(torch.isfinite(shifted_raw).all()),
                "modal_projection_boundary": True,
                "transport_T_equals_I_100": True,
                "shared_W_OLS": True,
            },
            "alpha": ALPHA,
            "renormalization": "none",
        },
        "raw_response": raw_detail,
        "template_direction": {
            **template_detail,
            "bootstrap": bootstrap_summaries["delta_template"],
            "positive_fraction": float((template_delta > 0.0).to(torch.float64).mean()),
            "by_source_class": template_groups["by_source_class"],
            "by_target_class": template_groups["by_target_class"],
            "primary_pass": decision["template_primary_pass"],
        },
        "smooth_direction": {
            **smooth_detail,
            "bootstrap": bootstrap_summaries["delta_smooth"],
            "positive_fraction": float((smooth_delta > 0.0).to(torch.float64).mean()),
            "by_source_class": smooth_groups["by_source_class"],
            "by_target_class": smooth_groups["by_target_class"],
            "primary_pass": decision["smooth_primary_pass"],
        },
        "directional_consistency": {
            "pairs": pair_summaries,
            "positive_template_pair_means": decision["positive_template_pair_means"],
            "positive_smooth_pair_means": decision["positive_smooth_pair_means"],
            "broad_or_pair_dominated": "broad" if not decision["pair_dominated"] else "pair-dominated",
            "decision": decision,
        },
        "decision": {
            "outer_gate": OUTER_GATE,
            "classification": classification,
            **decision,
            "unresolved_alternatives": [
                "template morphology may not transfer to independent Smooth semantic scores" if not decision["smooth_primary_pass"] else "none under the frozen directional rule",
                "this diagnosis does not establish generator capacity or layer sufficiency",
            ],
        },
        "artifacts": {"sufficient_observations": observation_artifact} if observation_artifact is not None else {},
    }
    return {
        "classification": classification,
        "report_fields": report_fields,
        "selection_indices": selection_indices,
        "centroids": centroids,
        "templates": templates,
        "z": z,
        "z_shift": z_shift,
        "base_raw": base_raw,
        "shifted_raw": shifted_raw,
        "base_logits": base_logits,
        "shifted_logits": shifted_logits,
        "template_delta": template_delta,
        "smooth_delta": smooth_delta,
        "bootstrap_summaries": bootstrap_summaries,
        "bootstrap_tensors": bootstrap_tensors,
        "source_labels": selection_labels,
        "target_labels": target_labels,
        "observation_artifact": observation_artifact,
        "canonical_replay_max_abs_difference": float(canonical_metadata.get("canonical_replay_max_abs_difference", 0.0)),
    }


def _core_report_fields(
    core_result: Mapping[str, Any],
    *,
    raw_artifact: Mapping[str, Any] | None = None,
    bootstrap_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = dict(_mapping(core_result.get("report_fields"), "class-direction core report fields"))
    artifacts = dict(_mapping(fields.get("artifacts"), "class-direction core artifacts"))
    if raw_artifact is not None:
        artifacts["raw_outputs"] = dict(raw_artifact)
    if bootstrap_artifact is not None:
        artifacts["bootstrap"] = dict(bootstrap_artifact)
    fields["artifacts"] = artifacts
    return fields


def _json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _write_report_and_lineage(root: Path, target: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    serialized = json.loads(json.dumps(dict(payload), default=_json_default))
    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, serialized, artifact_kind="OLS class-direction transfer report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "result": serialized["result"],
        "status": serialized["status"],
        "classification": serialized["classification"],
        "entrypoint": serialized["entrypoint"],
        "authorities": serialized.get("authorities"),
        "git_provenance": serialized.get("git_provenance"),
        "provenance": serialized["provenance"],
        "artifacts": {**serialized.get("artifacts", {}), "report": report_ref},
        "claim_boundary": serialized["claim_boundary"],
        "lineage_self_hash": "not_embedded_to_avoid_self_reference",
    }
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="OLS class-direction transfer lineage")
    lineage_ref = {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}
    return report_ref, lineage_ref


def _validate_execution_core_bindings() -> None:
    required = (
        "_validate_runtime_result",
        "_raw_summary",
        "_bootstrap_deltas",
        "run_class_direction_diagnosis_core",
        "_write_report_and_lineage",
    )
    for name in required:
        _require(callable(globals().get(name)), f"class-direction execution core symbol is unavailable: {name}")


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha256: str) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_OLS_CLASS_DIRECTION_TRANSFER_NOT_COMPLETED",
        "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha256},
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {
            "diagnosis_authorization_consumed": True,
            "diagnosis_executed": False,
            "scientific_forward_count": 0,
            "optimizer_created": False,
            "backward_executed": False,
            "optical_update_executed": False,
            "training_executed": False,
            "v4_retry_executed": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_work_executed": False,
            "sampled_generation_executed": False,
            "protected_accessed": False,
            "test_accessed": False,
            "sealed_accessed": False,
            "training_authorized": False,
            "stage_b_authorized": False,
            "generation_readiness": "not_ready",
            "formal_promotion": False,
        },
        "next": None,
    }


def _authority_payload(
    root: Path,
    v4_diag: Any,
    v4: Any,
    preflight: Any,
    contract_authorities: Mapping[str, Any],
    v3_authority: Mapping[str, Any],
    v4_authority: Mapping[str, Any],
    smooth_identity: Mapping[str, Any],
) -> dict[str, Any]:
    v3_checkpoint_payload, v3_state = v4._load_v3_control_state(root, preflight)
    v4_state = v4_authority["state"]
    _require(torch.equal(v3_state["encoder.phase"], v4_state["encoder.phase"]), "V3 CONTROL encoder and V4 encoder differ")
    _require(torch.equal(v3_state["ols.operator.transport"], OLS_TRANSPORT), "V3 CONTROL transport is not T=I_100")
    _require(torch.equal(v4_state["ols.operator.transport"], OLS_TRANSPORT), "V4 transport is not T=I_100")
    v3_ref = _mapping(contract_authorities.get("v3_control"), "V3 CONTROL authority")
    checkpoint_ref = _mapping(v3_ref.get("checkpoint"), "V3 CONTROL checkpoint authority")
    _require(checkpoint_ref.get("step") == 300 and checkpoint_ref.get("role") == "V3 CONTROL step-300 checkpoint", "V3 CONTROL checkpoint role changed")
    _require(file_sha256(_path(root, checkpoint_ref["path"])) == checkpoint_ref.get("sha256"), "V3 CONTROL checkpoint file identity changed")
    _require(v3_checkpoint_payload.get("checkpoint_step") == 300 and v3_checkpoint_payload.get("arm") == "control", "V3 CONTROL checkpoint schema role changed")
    runtime_ref = _mapping(contract_authorities.get("aligned_runtime"), "LightGen runtime authority")
    v2_ref = _mapping(contract_authorities.get("v2_contract"), "V2 contract authority")
    train_ref = _mapping(contract_authorities.get("training_source"), "TRAIN source authority")
    return {
        "semantic_authority_id": "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED",
        "v4_contract": {"path": v4_diag.V4_CONTRACT_PATH.as_posix(), "sha256": v4_diag.V4_CONTRACT_SHA256, "role": "READY V4 recoverability contract"},
        "v4_scientific_report": {"path": v4_diag.V4_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": v4_diag.V4_SCIENTIFIC_REPORT_SHA256, "role": "VALID_COMPLETED V4 recoverability attempt-02"},
        "v4_scientific_lineage": {"path": v4_diag.V4_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": v4_diag.V4_SCIENTIFIC_LINEAGE_SHA256},
        "v4_step_300_generator": {
            "path": v4_diag.V4_STEP_300_CHECKPOINT_PATH.as_posix(),
            "sha256": v4_diag.V4_STEP_300_CHECKPOINT_SHA256,
            "role": "immutable V4 step-300 generator checkpoint",
            "generator_phase_tensor_sha256": tensor_sha256(v4_state["generator.phase"]),
            "encoder_phase_tensor_sha256": tensor_sha256(v4_state["encoder.phase"]),
            "runtime_state_sha256": v4_diag._digest_tree(v4_diag._cpu_tree(v4_state)),
        },
        "v3_control": {
            "checkpoint": {"path": checkpoint_ref["path"], "sha256": checkpoint_ref["sha256"], "role": checkpoint_ref["role"], "step": 300},
            "encoder_phase_tensor_sha256": tensor_sha256(v3_state["encoder.phase"]),
            "runtime_state_keys": sorted(v3_state),
            "encoder_reused_by_v4": True,
        },
        "runtime": {"path": runtime_ref["path"], "sha256": runtime_ref["sha256"], "role": runtime_ref.get("role")},
        "v2_contract": {"path": v2_ref["path"], "sha256": v2_ref["sha256"], "role": v2_ref.get("role")},
        "training_source": {"path": train_ref["path"], "sha256": train_ref["sha256"], "role": train_ref.get("role")},
        "smooth_evaluator": dict(smooth_identity),
        "upstream_v3_failure_attribution": {
            "classification": _mapping(contract_authorities.get("v3_attribution"), "V3 attribution authority").get("classification"),
        },
        "git_provenance": dict(v4_authority["git_provenance"]),
        "identity_domain_policy": {
            "file_sha256": "checkpoint or artifact bytes",
            "state_dict_tree_sha256": "loaded parameter tree",
            "tensor_sha256": "tensor content",
            "semantic_authority": "role and completed producer attempt",
            "manifest_sha256": "machine manifest bytes",
            "cross_domain_equality_required": False,
        },
    }


def run_diagnosis(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    _require(not target.exists(), f"one-shot diagnosis output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    entrypoint_sha256 = file_sha256(Path(__file__).resolve())
    payload = _base_payload(root, started, entrypoint_sha256)
    input_ref: dict[str, Any] | None = None
    diagnosis_started = False
    scientific_forward_count = 0
    measurement_started: float | None = None
    try:
        _require(started.get("branch") == EXPECTED_BRANCH, "active Git branch differs from authorized diagnosis branch")
        _require(started.get("head") == EXPECTED_CONSUMER_HEAD, "current consumer HEAD differs from authorized diagnosis HEAD")
        v4_diag = _load_v4_diagnostic_helper(root)
        v4 = v4_diag._load_module(root)
        preflight, _contract, contract_authorities, v3_authority, _preflight_authority, v4_authority = v4_diag._resolve_authorities(root, v4, started)
        cuda_environment = _prepare_cuda_runtime_dll_search_path()
        payload["runtime_readiness"] = {"cuda_dll_search_path": cuda_environment, "scientific_data_accessed": False}
        try:
            device, device_facts = preflight._cuda_device_facts()
        except Exception as error:
            details = getattr(error, "details", {})
            diagnostic_text = " ".join(str(details.get("error", error)).splitlines())
            if "nvrtc-builtins64" in diagnostic_text.lower():
                raise DiagnosisBlocked(
                    "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED",
                    "cuda_backend",
                    "CUDA device facts and the actual runtime share an unavailable NVRTC builtins DLL",
                    error_type=type(error).__name__,
                    error=diagnostic_text,
                    data_roles_opened=False,
                ) from error
            raise
        _require(tuple(device_facts.get("compute_capability", ())) == EXPECTED_COMPUTE_CAPABILITY, "diagnosis target compute capability changed")
        _require(int(device_facts.get("vram_bytes", 0)) >= EXPECTED_VRAM_BYTES, "diagnosis target VRAM is below the accepted V4 target")
        _require(shutil.disk_usage(target.parent).free >= MIN_FREE_BYTES_FOR_OUTPUT, "insufficient output storage headroom")
        v2_contract = v4._load_v2_contract_before_seed(root)
        consumer = v4_diag._load_runtime(root, v4, preflight, v2_contract, v4_authority["state"], device)
        consumer.runtime.eval()
        frozen, smooth, evaluator_states = v4._load_evaluators(root, preflight, contract_authorities, device)
        frozen.eval()
        smooth.eval()
        smooth_identity = v4_diag._validate_smooth_evaluator_identity(contract_authorities, evaluator_states)
        payload["runtime_readiness"]["synthetic_runtime_smoke"] = _synthetic_runtime_smoke(v4, consumer, device)
        payload["runtime_readiness"]["ready_before_scientific_data"] = True
        _validate_execution_core_bindings()
        authorities = _authority_payload(root, v4_diag, v4, preflight, contract_authorities, v3_authority, v4_authority, smooth_identity)
        images, labels, train_indices, development_indices, selection_indices, dataset = v4._load_roles(root, preflight)
        _require(selection_indices.numel() == DEVELOPMENT_SELECTION_ROWS, "fixed DEV selection row count changed")
        _require(tensor_sha256(selection_indices) == v4_diag.SELECTION_INDICES_SHA256, "fixed DEV selection identity changed")
        selection_images = images.index_select(0, selection_indices).contiguous()
        selection_labels = labels.index_select(0, selection_indices).contiguous()
        train_images = images.index_select(0, train_indices).contiguous()
        train_labels = labels.index_select(0, train_indices).contiguous()
        _require(train_images.shape[0] == 55_000 and train_labels.shape[0] == 55_000, "TRAIN role row count changed")
        for digit in range(CLASS_COUNT):
            _require(int((train_labels == digit).sum()) == int(dataset["train"]["counts_by_class"][digit]), f"TRAIN class count changed for {digit}")
            _require(int((selection_labels == digit).sum()) == DEVELOPMENT_SELECTION_PER_CLASS, f"DEV selection class count changed for {digit}")
        input_payload = {
            "schema": INPUT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha256},
            "authorities": authorities,
            "git_provenance": authorities["git_provenance"],
            "protocol": {
                "objective": "class-centroid direction in existing full complex OLS C^100 -> frozen V4 step-300 generator -> target-consistent optical output",
                "authoritative_latent": "full complex z in C^100 in existing project OLS gauge convention",
                "target_rule": "t=(y+1) mod 10",
                "intervention": "z_shift=z-mu_y+mu_t",
                "alpha": ALPHA,
                "renormalization": "none",
                "centroid_accumulation_dtype": "torch.complex64 mean over frozen TRAIN z",
                "image_template_dtype": "torch.float32 mean over frozen TRAIN images",
                "transport": "T=I_100 coefficient space",
                "w_ols": "shared W_OLS modal projection/re-emission",
                "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired, class-stratified over 100 examples per source class", "quantile_convention": "torch.quantile interpolation=linear"},
                "template_direction": "M_template=d(output,tau_source)-d(output,tau_target); Delta=M_shift-M_base",
                "smooth_direction": "M_smooth=logit_target-logit_source; Delta=M_shift-M_base",
                "pair_mapping": [f"{digit}->{(digit + 1) % CLASS_COUNT}" for digit in range(CLASS_COUNT)],
                "pair_broadness_rule": "at least two positive pair means for any evidence space used by a primary classification",
                "classification_rules": {
                    "supported": "template and Smooth primary evidence pass and both are not one-pair-only",
                    "morphology_only": "template primary evidence passes broadly while Smooth primary evidence does not",
                    "not_supported": "neither primary evidence passes, intervention is valid, and no positive aggregate direction is isolated to one pair",
                    "mixed": "conflict, pair-specific positive aggregate, or unresolved evidence",
                },
            },
            "data_roles": {
                "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": v4_diag.TRAINING_SOURCE_SHA256, "rows_loaded": 60_000},
                "train": {"rows": 55_000, "rows_for_complex_centroids": 55_000, "rows_for_image_templates": 55_000, "counts_by_class": [int(value) for value in torch.bincount(train_labels, minlength=CLASS_COUNT).tolist()]},
                "development": {"rows": 5_000, "selection_rows": DEVELOPMENT_SELECTION_ROWS, "selection_per_class": DEVELOPMENT_SELECTION_PER_CLASS, "selection_indices_sha256": tensor_sha256(selection_indices), "labels_used_for": ["source class", "frozen target mapping"]},
                "calibration": {"rows_read": 0},
                "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
                "official_test": {"rows_read": 0, "status": "forbidden; path not opened"},
                "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
                "labels_as_model_input": False,
            },
            "claim_boundary": {"optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "training_executed": False, "stage_b_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
            "input_self_hash": "not_embedded_to_avoid_self_reference",
        }
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="OLS class-direction transfer input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        diagnosis_started = True
        measurement_started = time.perf_counter()
        train_z = _encode_train_z(consumer, train_images, device)
        scientific_forward_count += 5_500
        def canonical_forward(batch_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
            nonlocal scientific_forward_count
            base_raw, z, replay_difference = _collect_dev_base(consumer, batch_images, device)
            scientific_forward_count += 200
            return z, base_raw, {"canonical_replay_max_abs_difference": replay_difference}

        def shifted_forward(batch_z: torch.Tensor) -> torch.Tensor:
            nonlocal scientific_forward_count
            shifted_raw = _forward_shifted(consumer, batch_z, device)
            scientific_forward_count += 100
            return shifted_raw

        def smooth_forward(raw: torch.Tensor) -> torch.Tensor:
            return _smooth_logits(smooth, raw, device)

        def persist_sufficient_observations(observations: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> Mapping[str, Any]:
            return _save_sufficient_observations(root, target, observations, metadata)

        core_result = run_class_direction_diagnosis_core(
            train_z=train_z,
            train_images=train_images,
            train_labels=train_labels,
            selection_images=selection_images,
            selection_labels=selection_labels,
            canonical_forward=canonical_forward,
            shifted_forward=shifted_forward,
            smooth_forward=smooth_forward,
            selection_indices=selection_indices,
            persist_sufficient_observations=persist_sufficient_observations,
        )
        centroids = core_result["centroids"]
        templates = core_result["templates"]
        z = core_result["z"]
        z_shift = core_result["z_shift"]
        base_raw = core_result["base_raw"]
        shifted_raw = core_result["shifted_raw"]
        base_logits = core_result["base_logits"]
        shifted_logits = core_result["shifted_logits"]
        bootstrap_summaries = core_result["bootstrap_summaries"]
        bootstrap_tensors = core_result["bootstrap_tensors"]
        raw_artifact = _save_raw_artifact(root, target, centroids, templates, z, z_shift, base_raw, shifted_raw, selection_labels, core_result["target_labels"], base_logits, shifted_logits, selection_indices)
        bootstrap_artifact = _save_bootstrap_artifact(root, target, bootstrap_summaries, bootstrap_tensors, selection_labels)
        core_fields = _core_report_fields(core_result, raw_artifact=raw_artifact, bootstrap_artifact=bootstrap_artifact)
        classification = core_result["classification"]
        v4_endpoint = _mapping(_mapping(v4_authority["report"].get("trajectory"), "V4 trajectory").get("300"), "V4 endpoint")
        v4_smooth_endpoint = _mapping(v4_endpoint.get("smooth_evaluator"), "V4 Smooth endpoint")
        base_predictions = base_logits.argmax(dim=1)
        base_per_class = [float((base_predictions[selection_labels == digit] == digit).to(torch.float64).mean()) for digit in range(CLASS_COUNT)]
        base_macro = float(sum(base_per_class) / CLASS_COUNT)
        _require(math.isclose(base_macro, float(v4_smooth_endpoint["macro_recall"]), rel_tol=0.0, abs_tol=1.0e-7), "base Smooth endpoint does not reproduce valid V4 attempt-02")
        _require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-7) for a, b in zip(base_per_class, v4_smooth_endpoint["per_class_recall"], strict=True)), "base Smooth per-class endpoint does not reproduce valid V4 attempt-02")
        provider_after = {"frozen_state_sha256": _digest_tree(_cpu_tree(frozen.state_dict())), "smooth_state_sha256": _digest_tree(_cpu_tree(smooth.state_dict()))}
        _require(provider_after["frozen_state_sha256"] == evaluator_states["frozen_state_sha256"] and provider_after["smooth_state_sha256"] == evaluator_states["smooth_state_sha256"], "evaluator state changed during diagnosis")
        _require(all(parameter.grad is None for evaluator in (frozen, smooth) for parameter in evaluator.parameters()), "evaluator received a diagnosis gradient")
        payload.update(
            {
                "result": "VALID_COMPLETED",
                "status": "valid_completed",
                "classification": classification,
                "authorities": authorities,
                "git_provenance": authorities["git_provenance"],
                "device": device_facts,
                "runtime": {"grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "transport": "T=I_100_coefficient_space", "modal_projection_boundary": True, "shared_w_ols": True, "canonical_replay_max_abs_difference": core_result["canonical_replay_max_abs_difference"], "optimizer_created": False, "backward_executed": False, "optical_update_executed": False},
                "data": {**dataset, "train_complex_centroid_rows": int(train_images.shape[0]), "train_image_template_rows": int(train_images.shape[0]), "development_selection_rows": int(selection_images.shape[0]), "calibration_rows_read": 0, "protected_rows_read": 0, "test_rows_read": 0, "sealed_rows_read": 0},
                "execution": {"diagnosis_executed": True, "scientific_forward_count": scientific_forward_count, "train_encoder_forwards": 5_500, "dev_canonical_forwards": 100, "dev_replay_forwards": 100, "dev_shifted_forwards": 100, "smooth_evaluations": 200, "runtime": "LightGen 2D complex-OLS aligned runtime", "device": device_facts},
                **core_fields,
                "artifacts": {**core_fields["artifacts"], "input": input_ref},
                "governance": {"diagnosis_executed": True, "optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "training_executed": False, "v4_retry_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_work_executed": False, "sampled_generation_executed": False, "generation_acceptance": False, "protected_read": False, "test_read": False, "sealed_read": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False},
                "measurement_elapsed_seconds": time.perf_counter() - measurement_started,
                "next": "no_retry; this diagnosis does not authorize training, Stage B, generation or formal promotion",
            }
        )
        torch.cuda.empty_cache()
    except DiagnosisBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no_retry_without_new_authority"})
    except InvalidDiagnosis as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_OLS_CLASS_DIRECTION_TRANSFER_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no_retry_without_new_authority"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_OLS_CLASS_DIRECTION_TRANSFER_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no_retry_without_new_authority"})
    finally:
        try:
            ending = git_facts(root)
            payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "status_lines_unchanged": started.get("status_lines") == ending.get("status_lines")})
        except Exception as error:
            payload["provenance"]["git_end_error"] = {"type": type(error).__name__, "message": str(error)}
        payload["claim_boundary"].update({"diagnosis_executed": bool(payload["claim_boundary"].get("diagnosis_executed") or diagnosis_started), "scientific_forward_count": scientific_forward_count, "optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "training_executed": False, "v4_retry_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_work_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False})
        if measurement_started is not None and payload.get("measurement_elapsed_seconds") is None:
            payload["measurement_elapsed_seconds"] = time.perf_counter() - measurement_started
        payload.setdefault("artifacts", {})
        payload["artifacts"].setdefault("input", input_ref)
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"

    report_ref, lineage_ref = _write_report_and_lineage(root, target, payload)
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": lineage_ref, "diagnosis_executed": payload["claim_boundary"]["diagnosis_executed"], "scientific_forward_count": payload["claim_boundary"]["scientific_forward_count"], "optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "stage_b_executed": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("diagnose",), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_diagnosis(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if result["result"] == "VALID_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
