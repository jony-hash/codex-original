"""Diagnose V5 latent-to-optical transfer coupling without updates.

The consumer compares the actual V5 control-to-candidate latent displacement
with matched-norm and full class-direction canaries on the frozen optical path.
It performs no optimizer creation, backward pass, parameter update, training,
generation, protected/test/sealed read, or Stage-B action.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    LightGen2DComplexOLSTrainingConsumerV2,
    file_sha256,
    git_facts,
    tensor_sha256,
)
from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training import gauge_invariant_ols_representation
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_local_mnist, stratified_mnist_indices


TASK_ID = "DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1"
OUTER_GATE = "V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED"
REPORT_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1"
INPUT_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1"
RAW_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_raw_outputs.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_bootstrap.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_sufficient_observations.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "9a72e0bcf2de12ab9a0caa7151f273de76768dc7"

V5_CONTRACT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_contract.v5.json"
)
V5_CONTRACT_SHA256 = "e2d57fa96e0a020abf23cbd889188d74d4893a6e9c48ac7a1d24d7156cf21b61"
V5_SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1.json"
)
V5_SCIENTIFIC_REPORT_SHA256 = "41aeb6e5ae5d1736521b164f2c8e9267bc3687801ab99b7142692112f8d85ce8"
V5_SCIENTIFIC_LINEAGE_PATH = V5_SCIENTIFIC_REPORT_PATH.with_name(
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_lineage.v1.json"
)
V5_SCIENTIFIC_LINEAGE_SHA256 = "89e00a5f67dc082b0d2f37042a66a796cae71f14b1f676e8a27e94caaeed4c50"
V5_SCIENTIFIC_INPUT_PATH = V5_SCIENTIFIC_REPORT_PATH.with_name(
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_input.v1.json"
)
V5_SCIENTIFIC_INPUT_SHA256 = "91607325a08883479d49e69adac2d6a2c8ea6251ed86da5b45664aa0a7179b77"
V5_SCIENTIFIC_ENTRYPOINT_PATH = Path(
    "scripts/research/execute_lightgen_2d_complex_ols_semantic_alignment_v5_scientific.py"
)
V5_SCIENTIFIC_ENTRYPOINT_SHA256 = "b48d9f8ae9f128d9b2f7563e5892be8b82d7dc973ef9f56196a0ccf4571fae0f"
PREFLIGHT_ENTRYPOINT_PATH = Path(
    "scripts/research/execute_lightgen_2d_complex_ols_semantic_alignment_preflight_v5.py"
)
PREFLIGHT_ENTRYPOINT_SHA256 = "a062e2acb4e624969f09d3631adab2ede22a786c7dde64e52e045a2c76f8c98a"

V4_GENERATOR_CHECKPOINT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_v4_step_300_checkpoint.v1.pt"
)
V4_GENERATOR_CHECKPOINT_SHA256 = "de80f2826419ca6ced28c8e1fc6d8804ef513493a49474e5997cfb4cb59bdf80"

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_01"
)
INPUT_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_input.v1.json"
REPORT_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json"
LINEAGE_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json"
RAW_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_raw_outputs.v1.pt"
BOOTSTRAP_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_bootstrap.v1.pt"
SUFFICIENT_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_sufficient_observations.v1.pt"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
DEVELOPMENT_ROWS = 5_000
SELECTION_ROWS = 1_000
SELECTION_PER_CLASS = 100
SELECTION_SEED = 4321
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
EPSILON_DENOMINATOR = 1.0e-12
MATCH_NORM_ABS_TOLERANCE = 1.0e-8
MATCH_NORM_REL_TOLERANCE = 1.0e-5
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000

EXPECTED_PROTOTYPES_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
EXPECTED_S_MU = 0.00555040966719389

SUBTHRESHOLD = "V5_SUBTHRESHOLD_SEMANTIC_MOVEMENT_SUPPORTED"
DIRECTION_MISALIGNMENT = "V5_LATENT_DIRECTION_MISALIGNMENT_SUPPORTED"
MAGNITUDE_LIMITED = "V5_LATENT_DISPLACEMENT_MAGNITUDE_LIMITED_SUPPORTED"
MIXED = "V5_LATENT_TO_OPTICAL_TRANSFER_MIXED_OR_UNRESOLVED"
CLASSIFICATIONS = frozenset({SUBTHRESHOLD, DIRECTION_MISALIGNMENT, MAGNITUDE_LIMITED, MIXED})


class InvalidDiagnosis(RuntimeError):
    """The frozen diagnosis harness or authority is invalid."""


class DiagnosisBlocked(RuntimeError):
    """A valid runtime boundary prevents this one-shot diagnosis."""

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
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute authority path is not allowed: {value}")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidDiagnosis(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


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


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _finite_complex(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and torch.is_complex(value), f"{name} is not complex")
    _finite(value.real, f"{name}.real")
    _finite(value.imag, f"{name}.imag")


def _summary(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().to(torch.float64).cpu().flatten()
    _finite(flat, "summary values")
    _require(flat.numel() > 0, "summary values are empty")
    quantiles = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean()),
        "median": float(quantiles[1]),
        "q05": float(quantiles[0]),
        "q95": float(quantiles[2]),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "finite": True,
    }


def _realify(value: torch.Tensor) -> torch.Tensor:
    _finite_complex(value, "realification input")
    return torch.cat((value.real.to(torch.float64), value.imag.to(torch.float64)), dim=-1)


def _complex_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.to(torch.complex128), dim=1).to(torch.float64)


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_python_process_only"}
    cuda_build = torch.version.cuda
    if not isinstance(cuda_build, str) or len(cuda_build.split(".")) != 2 or not all(part.isdigit() for part in cuda_build.split(".")):
        raise DiagnosisBlocked("CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED", "cuda_dll_search_path", "CUDA build is not parseable", cuda_build=cuda_build)
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    if not environment_bin.is_dir() or not dll_path.is_file():
        raise DiagnosisBlocked(
            "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED",
            "cuda_dll_search_path",
            "required NVRTC builtins DLL is unavailable",
            cuda_build=cuda_build,
            environment_bin=str(environment_bin),
            expected_dll=dll_path.name,
        )
    current_path = os.environ.get("PATH", "")
    normalized = os.path.normcase(os.path.normpath(str(environment_bin)))
    entries = [entry for entry in current_path.split(os.pathsep) if entry]
    already_present = any(os.path.normcase(os.path.normpath(entry)) == normalized for entry in entries)
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


def _load_module(root: Path, path_value: Path, expected_sha: str, name: str) -> Any:
    path = _path(root, path_value)
    _validate_hash(path, expected_sha, name)
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"{name} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checked_private_callable(module: Any, owner: str, name: str, parameters: tuple[str, ...]) -> Callable[..., Any]:
    candidate = getattr(module, name, None)
    _require(callable(candidate), f"{owner}.{name} is unavailable")
    try:
        actual_parameters = tuple(inspect.signature(candidate).parameters)
    except (TypeError, ValueError) as error:
        raise InvalidDiagnosis(f"{owner}.{name} signature is not inspectable") from error
    _require(actual_parameters == parameters, f"{owner}.{name} signature changed: {actual_parameters}")
    return candidate


def _audit_private_dependencies(v5: Any, preflight: Any) -> dict[str, Any]:
    retained = (
        ("v5_scientific", v5, "_load_preflight_module", ("root",), "accepted preflight module resolver"),
        ("v5_scientific", v5, "_validate_accepted_preflight", ("root", "preflight"), "accepted V5 authority validator"),
        ("accepted_v5_preflight", preflight, "_load_contract", ("root",), "registered V5 contract parser used by the accepted validator"),
        ("accepted_v5_preflight", preflight, "_validate_authority_payloads", ("root", "contract", "authorities"), "registered V5 authority payload validator used by the accepted validator"),
    )
    records: list[dict[str, Any]] = []
    for owner, module, name, parameters, rationale in retained:
        callable_value = _checked_private_callable(module, owner, name, parameters)
        records.append({"owner": owner, "symbol": name, "signature": str(inspect.signature(callable_value)), "status": "retained_and_signature_checked", "rationale": rationale})
    return {
        "status": "passed",
        "retained": records,
        "replaced": [
            {"symbol": "accepted_v5_preflight._load_evaluators", "replacement": "local _load_smooth_evaluator", "rationale": "the accepted preflight exposes _load_smooth only; the transfer consumer now owns an explicit adapter over the public SmoothMNISTEvaluator"},
            {"symbol": "accepted_v5_preflight._load_train_role / V4 _load_roles", "replacement": "public load_local_mnist + stratified_mnist_indices in this consumer", "rationale": "selection loading is kept local and is not needed by the no-data dry run"},
            {"symbol": "accepted_v5_preflight._new_consumer / _load_runtime_state / _configure_all_frozen", "replacement": "direct LightGen2DComplexOLSTrainingConsumerV2 construction and local freeze checks", "rationale": "the production wrapper uses the public runtime consumer directly"},
            {"symbol": "accepted_v5_preflight._cuda_facts / prototype_distance", "replacement": "local public-API implementations", "rationale": "CUDA facts and frozen gauge distance are explicit dependencies of this consumer"},
        ],
    }


def _validate_v5_authority(root: Path) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    v5 = _load_module(root, V5_SCIENTIFIC_ENTRYPOINT_PATH, V5_SCIENTIFIC_ENTRYPOINT_SHA256, "V5 scientific entrypoint")
    load_preflight = _checked_private_callable(v5, "v5_scientific", "_load_preflight_module", ("root",))
    preflight = load_preflight(root)
    dependency_audit = _audit_private_dependencies(v5, preflight)
    validate_preflight = _checked_private_callable(v5, "v5_scientific", "_validate_accepted_preflight", ("root", "preflight"))
    contract, authorities, v2_contract, v3, v4, preflight_report, preflight_lineage = validate_preflight(root, preflight)
    _validate_hash(_path(root, V5_SCIENTIFIC_REPORT_PATH), V5_SCIENTIFIC_REPORT_SHA256, "valid V5 scientific report")
    _validate_hash(_path(root, V5_SCIENTIFIC_LINEAGE_PATH), V5_SCIENTIFIC_LINEAGE_SHA256, "valid V5 scientific lineage")
    _validate_hash(_path(root, V5_SCIENTIFIC_INPUT_PATH), V5_SCIENTIFIC_INPUT_SHA256, "valid V5 scientific input")
    report = _read_json(_path(root, V5_SCIENTIFIC_REPORT_PATH), "valid V5 scientific report")
    lineage = _read_json(_path(root, V5_SCIENTIFIC_LINEAGE_PATH), "valid V5 scientific lineage")
    input_payload = _read_json(_path(root, V5_SCIENTIFIC_INPUT_PATH), "valid V5 scientific input")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "V5 scientific result is not valid completed")
    _require(report.get("classification") == "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED", "V5 scientific classification changed")
    _require(report.get("outer_gate") == "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED", "V5 scientific outer gate changed")
    _require(lineage.get("result") == report.get("result") and lineage.get("classification") == report.get("classification"), "V5 scientific lineage changed")
    _require(input_payload.get("schema") == "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_input.v1", "V5 scientific input schema changed")
    accepted_preflight = _mapping(input_payload.get("accepted_preflight"), "V5 accepted preflight authority")
    accepted_preflight_input = _mapping(accepted_preflight.get("input"), "V5 accepted preflight input authority")
    preflight_input = _mapping(preflight_report.get("preflight_input"), "accepted V5 preflight input")
    _require(accepted_preflight_input.get("path") == preflight_input.get("path") and accepted_preflight_input.get("sha256") == preflight_input.get("sha256"), "V5 input preflight binding changed")
    entrypoint = _mapping(report.get("entrypoint"), "V5 scientific entrypoint identity")
    _require(entrypoint.get("path") == V5_SCIENTIFIC_ENTRYPOINT_PATH.as_posix() and entrypoint.get("sha256") == V5_SCIENTIFIC_ENTRYPOINT_SHA256, "V5 scientific entrypoint identity changed")
    boundary = _mapping(report.get("claim_boundary"), "V5 scientific claim boundary")
    _require(boundary.get("scientific_comparison_executed") is True and boundary.get("stage_b_executed") is False, "V5 scientific claim boundary changed")
    for key in ("scale_extraction_executed", "prior_generation_executed", "sampled_generation_executed", "protected_accessed", "official_test_accessed", "sealed_accessed", "formal_promotion"):
        _require(boundary.get(key) is False, f"V5 scientific crossed forbidden boundary: {key}")
    _require(report.get("provenance", {}).get("head_unchanged") is True and report.get("provenance", {}).get("unexpected_modified_paths") == [], "V5 scientific provenance is not clean")
    dataset = _mapping(report.get("dataset"), "V5 scientific dataset")
    _require(dataset.get("train", {}).get("rows") == 55_000 and dataset.get("development", {}).get("rows") == 5_000, "V5 scientific data roles changed")
    _require(dataset.get("calibration", {}).get("rows_read") == 0 and dataset.get("protected", {}).get("rows_read") == 0 and dataset.get("official_test", {}).get("rows_read") == 0 and dataset.get("sealed", {}).get("rows_read") == 0, "V5 scientific data firewall changed")
    prototypes = _mapping(report.get("prototype_setup", {}).get("summary"), "V5 scientific prototype summary")
    _require(prototypes.get("prototypes_tensor_sha256") == EXPECTED_PROTOTYPES_SHA256 and prototypes.get("rows") == 55_000 and prototypes.get("source_role") == "TRAIN only", "V5 frozen prototype authority changed")
    _require(math.isclose(float(report.get("prototype_setup", {}).get("scale", {}).get("s_mu")), EXPECTED_S_MU, rel_tol=0.0, abs_tol=1.0e-15), "V5 prototype scale changed")
    return v5, preflight, v2_contract, dict(contract), dict(authorities), report, lineage, input_payload, dependency_audit


def _load_smooth_evaluator(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[SmoothMNISTEvaluator, dict[str, Any]]:
    source = _mapping(authorities.get("source"), "V5 source authorities")
    reference = _mapping(source.get("smooth_checkpoint"), "Smooth checkpoint authority")
    path = _path(root, str(reference.get("path")))
    expected_sha = str(reference.get("sha256"))
    _validate_hash(path, expected_sha, "Smooth evaluator checkpoint")
    payload = _mapping(safe_torch_load(path, expected_sha256=expected_sha, map_location="cpu"), "Smooth evaluator checkpoint")
    _require(payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    evaluator = SmoothMNISTEvaluator()
    evaluator.load_state_dict(_mapping(payload.get("state_dict"), "Smooth evaluator state"), strict=True)
    evaluator.to(device).freeze()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator gradient firewall failed")
    identity = {
        "checkpoint": dict(reference),
        "architecture_kind": payload.get("architecture_kind"),
        "state_dict_tree_sha256": _digest_tree(_cpu_tree(evaluator.state_dict())),
        "gradient_enabled": False,
        "cross_domain_equality_required": False,
    }
    _require(expected_sha == SMOOTH_CHECKPOINT_SHA256, "Smooth evaluator checkpoint identity changed")
    return evaluator, identity


def _cuda_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise DiagnosisBlocked(
            "CUDA_RUNTIME_UNAVAILABLE",
            "cuda_runtime",
            "CUDA:0 is unavailable for the production wrapper",
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    probe = torch.fft.fft2(torch.ones((8, 8), dtype=torch.complex64, device=device))
    torch.cuda.synchronize(device)
    _finite_complex(probe, "CUDA complex probe")
    return device, {
        "device": str(device),
        "device_count": int(torch.cuda.device_count()),
        "gpu": properties.name,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "grid": PRODUCTION_GRID,
    }


def prototype_distance(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Return the existing gauge-compatible squared distance to each prototype."""

    _require(z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT and torch.is_complex(z), "query z must be [B,100] complex")
    _require(prototypes.ndim == 2 and prototypes.shape == (CLASS_COUNT, OLS_PORT_COUNT) and torch.is_complex(prototypes), "prototypes must be [10,100] complex")
    query = gauge_invariant_ols_representation(z)
    support = gauge_invariant_ols_representation(prototypes)
    distances = (query[:, None, :] - support[None, :, :]).square().sum(dim=2)
    _finite(distances, "prototype distances")
    return distances


def _load_registered_selection_authority(root: Path, authorities: Mapping[str, Any]) -> dict[str, Any]:
    replication = _mapping(authorities.get("class_direction_replication"), "registered class-direction replication authority")
    report_reference = _mapping(replication.get("report"), "registered class-direction replication report")
    lineage_reference = _mapping(replication.get("lineage"), "registered class-direction replication lineage")
    report_path = _path(root, str(report_reference.get("path")))
    lineage_path = _path(root, str(lineage_reference.get("path")))
    _validate_hash(report_path, str(report_reference.get("sha256")), "registered class-direction replication report")
    _validate_hash(lineage_path, str(lineage_reference.get("sha256")), "registered class-direction replication lineage")
    report = _read_json(report_path, "registered class-direction replication report")
    lineage = _read_json(lineage_path, "registered class-direction replication lineage")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("classification") == "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED", "registered selection producer result changed")
    _require(lineage.get("result") == report.get("result") and lineage.get("classification") == report.get("classification"), "registered selection producer lineage changed")
    report_input = _mapping(_mapping(report.get("artifacts"), "registered selection report artifacts").get("input"), "registered selection input authority")
    input_path = _path(root, str(report_input.get("path")))
    input_sha = str(report_input.get("sha256"))
    _validate_hash(input_path, input_sha, "registered selection input")
    input_payload = _read_json(input_path, "registered selection input")
    _require(input_payload.get("schema") == "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1", "registered selection input schema changed")
    data_roles = _mapping(input_payload.get("data_roles"), "registered selection data roles")
    development = _mapping(data_roles.get("development"), "registered DEV selection")
    source = _mapping(data_roles.get("source"), "registered selection source")
    _require(source.get("sha256") == TRAINING_SOURCE_SHA256 and development.get("rows") == DEVELOPMENT_ROWS, "registered selection source/DEV role changed")
    declared_selection_seed = development.get("selection_seed")
    _require(development.get("selection_rows") == SELECTION_ROWS and development.get("selection_per_class") == SELECTION_PER_CLASS and development.get("selection_indices_sha256") == SELECTION_INDICES_SHA256 and declared_selection_seed in (None, SELECTION_SEED), "registered DEV selection identity changed")
    _require(_mapping(input_payload.get("protocol"), "registered selection protocol").get("transport") == "T=I_100 coefficient space", "registered selection transport changed")
    return {
        "semantic_authority": replication.get("semantic_authority"),
        "producer_task": replication.get("producer_task"),
        "producer_attempt": replication.get("producer_attempt"),
        "report": {"path": _relative(root, report_path), "sha256": str(report_reference.get("sha256"))},
        "lineage": {"path": _relative(root, lineage_path), "sha256": str(lineage_reference.get("sha256"))},
        "input": {"path": _relative(root, input_path), "sha256": input_sha},
        "selection": {"rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "seed_declared_by_producer": declared_selection_seed is not None, "indices_sha256": SELECTION_INDICES_SHA256, "development_rows": DEVELOPMENT_ROWS},
        "scientific_values_read": 0,
    }


def _load_checkpoint(root: Path, report: Mapping[str, Any], arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = _mapping(report.get("arms"), "V5 scientific arms")
    arm_payload = _mapping(arms.get(arm), f"V5 {arm} arm")
    checkpoints = _mapping(arm_payload.get("checkpoints"), f"V5 {arm} checkpoints")
    reference = _mapping(checkpoints.get("300"), f"V5 {arm} step-300 checkpoint")
    path = _path(root, str(reference.get("path")))
    expected_sha = str(reference.get("sha256"))
    _validate_hash(path, expected_sha, f"V5 {arm} step-300 checkpoint")
    payload = _mapping(safe_torch_load(path, expected_sha256=expected_sha, map_location="cpu"), f"V5 {arm} step-300 checkpoint")
    _require(payload.get("schema") == "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_checkpoint.v1" and payload.get("checkpoint_step") == 300 and payload.get("arm") == arm, f"V5 {arm} checkpoint schema changed")
    state = dict(_mapping(payload.get("runtime_state"), f"V5 {arm} runtime state"))
    expected_keys = {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}
    _require(set(state) == expected_keys, f"V5 {arm} runtime state keys changed")
    prototypes = payload.get("prototypes")
    _require(torch.is_tensor(prototypes) and tuple(prototypes.shape) == (CLASS_COUNT, OLS_PORT_COUNT) and prototypes.dtype == torch.complex64, f"V5 {arm} prototypes shape/dtype changed")
    _require(payload.get("prototypes_sha256") == EXPECTED_PROTOTYPES_SHA256 and tensor_sha256(prototypes) == EXPECTED_PROTOTYPES_SHA256, f"V5 {arm} prototype identity changed")
    _require(math.isclose(float(payload.get("s_mu")), EXPECTED_S_MU, rel_tol=0.0, abs_tol=1.0e-15), f"V5 {arm} prototype scale changed")
    expected_lambda = 0.0 if arm == "control" else float(report["lambda_proto"]["lambda_proto"])
    _require(math.isclose(float(payload.get("lambda_proto")), expected_lambda, rel_tol=0.0, abs_tol=1.0e-15), f"V5 {arm} lambda changed")
    for key in ("stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "sampled_generation_executed", "formal_promotion"):
        _require(payload.get(key) is False, f"V5 {arm} checkpoint crossed forbidden boundary: {key}")
    return dict(payload), state


def _load_v4_generator(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _path(root, V4_GENERATOR_CHECKPOINT_PATH)
    _validate_hash(path, V4_GENERATOR_CHECKPOINT_SHA256, "frozen V4 generator checkpoint")
    payload = _mapping(safe_torch_load(path, expected_sha256=V4_GENERATOR_CHECKPOINT_SHA256, map_location="cpu"), "frozen V4 generator checkpoint")
    state = dict(_mapping(payload.get("runtime_state"), "frozen V4 generator runtime state"))
    _require(payload.get("checkpoint_step") == 300, "frozen V4 generator checkpoint step changed")
    _require(torch.is_tensor(state.get("generator.phase")) and torch.is_tensor(state.get("ols.operator.transport")), "frozen V4 generator state is incomplete")
    _require(torch.equal(state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "frozen V4 transport is not T=I_100")
    return dict(payload), state


def _load_selection(root: Path, selection_authority: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
    source_path = _path(root, TRAINING_SOURCE_PATH)
    _validate_hash(source_path, TRAINING_SOURCE_SHA256, "canonical TRAIN source")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28) and images.dtype == torch.float32, "TRAIN source image shape/dtype changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "TRAIN source label shape/dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    _require(train_indices.numel() == 55_000 and development_indices.numel() == DEVELOPMENT_ROWS, "TRAIN/DEV partition size changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV roles overlap or omit rows")
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    development_labels = labels.index_select(0, development_indices)
    selected: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, f"DEV class count changed for {digit}")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:SELECTION_PER_CLASS]))
    selection_indices = torch.cat(selected).contiguous()
    _require(selection_indices.numel() == SELECTION_ROWS and tensor_sha256(selection_indices) == SELECTION_INDICES_SHA256, "balanced DEV selection identity changed")
    selection_labels = labels.index_select(0, selection_indices)
    _require(torch.equal(torch.bincount(selection_labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), SELECTION_PER_CLASS, dtype=torch.long)), "balanced DEV selection class counts changed")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=CLASS_COUNT)
    metadata = {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000},
        "train": {"rows": 55_000, "counts_by_class": [int(value) for value in train_counts.tolist()], "indices_sha256": tensor_sha256(train_indices), "scientific_values_read": 0},
        "development": {"rows": DEVELOPMENT_ROWS, "selection_rows": SELECTION_ROWS, "selection_per_class": SELECTION_PER_CLASS, "selection_seed": SELECTION_SEED, "indices_sha256": tensor_sha256(development_indices), "selection_indices_sha256": SELECTION_INDICES_SHA256, "values_read_for_evaluation": SELECTION_ROWS},
        "calibration": {"rows_read": 0},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "official_test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    expected = _mapping(selection_authority.get("selection"), "registered selection summary")
    _require(expected.get("rows") == SELECTION_ROWS and expected.get("per_class") == SELECTION_PER_CLASS and expected.get("indices_sha256") == SELECTION_INDICES_SHA256, "registered selection authority changed")
    selection_images = images.index_select(0, selection_indices).contiguous()
    return selection_images, selection_labels.contiguous(), selection_indices, SELECTION_PER_CLASS, SELECTION_SEED, metadata


def _new_frozen_consumer(v2_contract: Any, state: Mapping[str, Any], device: torch.device) -> Any:
    consumer = LightGen2DComplexOLSTrainingConsumerV2(v2_contract, device=device, propagation_grid_size=PRODUCTION_GRID)
    consumer.runtime.load_state_dict(dict(state), strict=True)
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    _require(not any(parameter.requires_grad for parameter in consumer.runtime.parameters()), "diagnostic consumer has trainable parameters")
    consumer.runtime.eval()
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "diagnostic consumer transport is not T=I_100")
    return consumer


def _validate_runtime_result(result: Any, batch_size: int, *, external: bool) -> torch.Tensor:
    if external:
        _require(result.image_input is None and result.ols_input_field is None and result.z_ols.source == "external_generated", "external path crossed the image encoder boundary")
    else:
        _require(result.image_input is not None and result.ols_input_field is not None and result.z_ols.source == "real_encoded", "real image path lost the encoder/OLS boundary")
        _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "shared W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "T=I_100 provenance changed")
    raw = result.raw_intensity
    _require(tuple(raw.shape) == (batch_size, 1, 200, 200) and raw.dtype == torch.float32, "raw CMOS shape/dtype changed")
    _finite(raw, "raw CMOS")
    return raw


@torch.inference_mode()
def _encode_dev(consumer: Any, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str, float]:
    before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    z_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
        batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
        result = consumer.runtime.forward_from_image(batch)
        raw = _validate_runtime_result(result, int(batch.shape[0]), external=False)
        z = result.z_ols.coefficients.detach().cpu().contiguous()
        _require(tuple(z.shape) == (int(batch.shape[0]), OLS_PORT_COUNT) and z.dtype == torch.complex64, "encoded DEV z shape/dtype changed")
        z_parts.append(z)
        raw_parts.append(raw.detach().cpu().contiguous())
        del batch, result, raw, z
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    after = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    _require(before == after, "frozen encoder forward changed runtime state")
    z_result = torch.cat(z_parts, dim=0)
    raw_result = torch.cat(raw_parts, dim=0)
    _finite_complex(z_result, "encoded DEV z")
    _finite(raw_result, "encoded DEV raw CMOS")
    return z_result, raw_result, before, float((raw_result - raw_result).abs().max())


@torch.inference_mode()
def _forward_external(consumer: Any, z_cpu: torch.Tensor) -> torch.Tensor:
    before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    parts: list[torch.Tensor] = []
    for start in range(0, z_cpu.shape[0], LOGICAL_BATCH_SIZE):
        batch = z_cpu[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
        result = consumer.runtime.forward_from_ols(batch)
        raw = _validate_runtime_result(result, int(batch.shape[0]), external=True)
        parts.append(raw.detach().cpu().contiguous())
        del batch, result, raw
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    after = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    _require(before == after, "frozen optical forward changed runtime state")
    result = torch.cat(parts, dim=0)
    _require(tuple(result.shape) == (z_cpu.shape[0], 1, 200, 200) and result.dtype == torch.float32, "external raw CMOS shape/dtype changed")
    _finite(result, "external raw CMOS")
    return result


@torch.inference_mode()
def _smooth_logits(smooth: Any, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, raw.shape[0], LOGICAL_BATCH_SIZE):
        batch = raw[start : start + LOGICAL_BATCH_SIZE].to(device)
        view = F.adaptive_avg_pool2d(batch.clamp(0.0, 1.0), (28, 28))
        _, logits = smooth(view)
        _require(tuple(logits.shape) == (int(batch.shape[0]), CLASS_COUNT) and not torch.is_complex(logits), "Smooth logits shape/dtype changed")
        _finite(logits, "Smooth logits")
        parts.append(logits.detach().cpu().contiguous())
        del batch, view, logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return torch.cat(parts, dim=0)


def _raw_summary(raw: torch.Tensor) -> dict[str, Any]:
    value = raw.detach().cpu().contiguous()
    _finite(value, "raw CMOS summary")
    per_image_variance = value.flatten(1).var(dim=1, unbiased=False)
    across_sample_variance = value.flatten(1).var(dim=0, unbiased=False)
    _require(tuple(value.shape[1:]) == (1, 200, 200) and value.dtype == torch.float32, "raw CMOS structure changed")
    structured = bool((per_image_variance > 0.0).all() and float(across_sample_variance.mean()) > 0.0)
    nonconstant = bool(float(value.std(unbiased=False)) > 0.0)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "structured": structured,
        "nonconstant": nonconstant,
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "per_image_variance_mean": float(per_image_variance.mean()),
        "across_sample_variance_mean": float(across_sample_variance.mean()),
        "tensor_sha256": tensor_sha256(value),
    }


def _raw_change(base: torch.Tensor, shifted: torch.Tensor) -> dict[str, Any]:
    difference = shifted.to(torch.float64) - base.to(torch.float64)
    absolute = difference.flatten(1).square().mean(dim=1).sqrt()
    baseline_rms = base.to(torch.float64).flatten(1).square().mean(dim=1).sqrt()
    relative = absolute / (baseline_rms + EPSILON_DENOMINATOR)
    _finite(absolute, "absolute raw RMS change")
    _finite(relative, "relative raw RMS change")
    return {"absolute_rms": _summary(absolute), "relative_rms": _summary(relative), "absolute_values_sha256": tensor_sha256(absolute), "relative_values_sha256": tensor_sha256(relative)}


def _correct_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(labels.shape[0], dtype=torch.long)
    other = logits.clone()
    other[rows, labels] = -torch.inf
    margin = logits[rows, labels] - other.max(dim=1).values
    _finite(margin, "Smooth correct-class margin")
    return margin.to(torch.float64)


def _prediction_summary(logits: torch.Tensor, labels: torch.Tensor, baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        per_class.append(float((predictions[mask] == digit).to(torch.float64).mean()))
    result: dict[str, Any] = {
        "prediction_tensor_sha256": tensor_sha256(predictions),
        "overall_correct_rate": float((predictions == labels).to(torch.float64).mean()),
        "macro_recall": float(sum(per_class) / CLASS_COUNT),
        "per_class_recall": per_class,
        "prediction_histogram": [int(value) for value in torch.bincount(predictions, minlength=CLASS_COUNT).tolist()],
    }
    if baseline is not None:
        result["overall_correct_rate_delta"] = result["overall_correct_rate"] - float(baseline["overall_correct_rate"])
        result["macro_recall_delta"] = result["macro_recall"] - float(baseline["macro_recall"])
        result["per_class_recall_delta"] = [a - b for a, b in zip(result["per_class_recall"], baseline["per_class_recall"], strict=True)]
    return result


def _class_summary(values: torch.Tensor, labels: torch.Tensor, samples_per_class: int) -> list[dict[str, Any]]:
    result = []
    for digit in range(CLASS_COUNT):
        selected = values[labels == digit]
        _require(selected.numel() == samples_per_class, f"DEV class count changed for class {digit}")
        result.append({"class": digit, "n": int(selected.numel()), "summary": _summary(selected), "positive_fraction": float((selected > 0.0).to(torch.float64).mean())})
    return result


def _bootstrap(deltas: Mapping[str, torch.Tensor], labels: torch.Tensor, samples_per_class: int, replicates: int, seed: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    class_draws: list[torch.Tensor] = []
    replicate_values: dict[str, list[torch.Tensor]] = {name: [] for name in deltas}
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == samples_per_class, f"bootstrap class count changed for {digit}")
        draw = torch.randint(samples_per_class, (replicates, samples_per_class), generator=generator)
        class_draws.append(draw)
        for name, delta in deltas.items():
            replicate_values[name].append(delta.index_select(0, matches)[draw].mean(dim=1))
    macro = {name: torch.stack(values, dim=0).mean(dim=0).to(torch.float64) for name, values in replicate_values.items()}
    summaries = {}
    for name, values in macro.items():
        quantiles = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
        summaries[name] = {"mean": float(values.mean()), "median": float(values.median()), "ci_lower": float(quantiles[0]), "ci_upper": float(quantiles[1]), "positive_fraction": float((values > 0.0).to(torch.float64).mean()), "tensor_sha256": tensor_sha256(values)}
    return summaries, {"class_draws": torch.stack(class_draws, dim=0), "macro_deltas": macro}


def _geometry(z_control: torch.Tensor, z_candidate: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor, distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], expected_rows: int) -> dict[str, Any]:
    _require(z_control.shape == z_candidate.shape == (expected_rows, OLS_PORT_COUNT), "V5 latent endpoint shape changed")
    _finite_complex(z_control, "z_control")
    _finite_complex(z_candidate, "z_candidate")
    _finite_complex(prototypes, "V5 prototypes")
    distances = distance_fn(z_control, prototypes).to(torch.float64).cpu()
    _require(tuple(distances.shape) == (expected_rows, CLASS_COUNT), "prototype distance shape changed")
    rows = torch.arange(expected_rows, dtype=torch.long)
    distances[rows, labels] = torch.inf
    nearest_incorrect = distances.argmin(dim=1)
    _require(torch.all(nearest_incorrect != labels), "nearest incorrect prototype selection failed")
    delta_v = z_candidate.to(torch.complex128) - z_control.to(torch.complex128)
    class_direction = prototypes.to(torch.complex128).index_select(0, labels) - prototypes.to(torch.complex128).index_select(0, nearest_incorrect)
    norm_v = _complex_norm(delta_v)
    norm_class = _complex_norm(class_direction)
    ratio = norm_v / (norm_class + EPSILON_DENOMINATOR)
    v_real = _realify(delta_v)
    class_real = _realify(class_direction)
    alignment = (v_real * class_real).sum(dim=1) / (norm_v * norm_class + EPSILON_DENOMINATOR)
    centered = prototypes.to(torch.complex128) - prototypes.to(torch.complex128).mean(dim=0, keepdim=True)
    centered_real = _realify(centered)
    _, singular_values, vh = torch.linalg.svd(centered_real, full_matrices=False)
    rank = int(torch.linalg.matrix_rank(centered_real).item())
    _require(rank > 0 and rank <= CLASS_COUNT - 1, "class-centroid subspace rank is undefined")
    basis = vh[:rank].transpose(0, 1).contiguous()
    projection = (v_real @ basis) @ basis.transpose(0, 1)
    rho = projection.square().sum(dim=1) / (v_real.square().sum(dim=1) + EPSILON_DENOMINATOR)
    _finite(norm_v, "V5 latent displacement norm")
    _finite(norm_class, "class direction norm")
    _finite(ratio, "latent/class norm ratio")
    _finite(alignment, "latent/class alignment")
    _finite(rho, "class-subspace rho")
    z_match = z_control.to(torch.complex128) + norm_v.unsqueeze(1) * class_direction / (norm_class.unsqueeze(1) + EPSILON_DENOMINATOR)
    z_full = z_control.to(torch.complex128) + class_direction
    match_error = (_complex_norm(z_match - z_control.to(torch.complex128)) - norm_v).abs()
    _require(float(match_error.max()) <= MATCH_NORM_ABS_TOLERANCE + MATCH_NORM_REL_TOLERANCE * max(float(norm_v.max()), 1.0), "matched-norm construction is not norm preserving")
    return {
        "z_control": z_control,
        "z_candidate": z_candidate,
        "delta_v": delta_v,
        "delta_class": class_direction,
        "z_match": z_match.to(torch.complex64),
        "z_full": z_full.to(torch.complex64),
        "nearest_incorrect": nearest_incorrect,
        "norm_v": norm_v,
        "norm_class": norm_class,
        "ratio": ratio,
        "alignment": alignment,
        "rho": rho,
        "singular_values": singular_values,
        "rank": rank,
        "match_error": match_error,
        "class_direction_pair": torch.stack((labels, nearest_incorrect), dim=1),
    }


def _run_core(
    *,
    z_control: torch.Tensor,
    z_candidate: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    selection_indices: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optical_forward: Callable[[torch.Tensor], torch.Tensor],
    smooth_forward: Callable[[torch.Tensor], torch.Tensor],
    persist_sufficient: Callable[[Mapping[str, torch.Tensor]], Mapping[str, Any]] | None = None,
    samples_per_class: int = SELECTION_PER_CLASS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    expected_rows = CLASS_COUNT * samples_per_class
    _require(labels.shape == (expected_rows,) and labels.dtype == torch.long, "diagnostic labels shape/dtype changed")
    _require(selection_indices.shape == labels.shape and selection_indices.dtype == torch.long, "diagnostic selection identity shape/dtype changed")
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), samples_per_class, dtype=torch.long)), "diagnostic labels are not balanced")
    geometry = _geometry(z_control, z_candidate, prototypes, labels, distance_fn, expected_rows)
    z_base = geometry["z_control"]
    z_actual = geometry["z_candidate"]
    z_match = geometry["z_match"]
    z_full = geometry["z_full"]
    raw_base = optical_forward(z_base.to(torch.complex64))
    raw_actual = optical_forward(z_actual.to(torch.complex64))
    raw_match = optical_forward(z_match)
    raw_full = optical_forward(z_full)
    for name, raw in (("base", raw_base), ("actual", raw_actual), ("match", raw_match), ("full", raw_full)):
        _require(tuple(raw.shape) == (expected_rows, 1, 200, 200) and raw.dtype == torch.float32, f"{name} raw CMOS shape/dtype changed")
        _finite(raw, f"{name} raw CMOS")
    logits = {name: smooth_forward(raw) for name, raw in (("base", raw_base), ("actual", raw_actual), ("match", raw_match), ("full", raw_full))}
    for name, value in logits.items():
        _require(tuple(value.shape) == (expected_rows, CLASS_COUNT), f"{name} Smooth logits shape changed")
        _finite(value, f"{name} Smooth logits")
    margins = {name: _correct_margin(value, labels) for name, value in logits.items()}
    deltas = {"actual": margins["actual"] - margins["base"], "match": margins["match"] - margins["base"], "full": margins["full"] - margins["base"]}
    _finite_complex(geometry["delta_v"], "actual latent displacement")
    bootstrap_summaries, bootstrap_tensors = _bootstrap(deltas, labels, samples_per_class, bootstrap_replicates, bootstrap_seed)
    baseline_prediction = _prediction_summary(logits["base"], labels)
    endpoint_predictions = {name: _prediction_summary(logits[name], labels, baseline_prediction) for name in ("actual", "match", "full")}
    raw_changes = {name: _raw_change(raw_base, raw) for name, raw in (("actual", raw_actual), ("match", raw_match), ("full", raw_full))}
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        per_class.append({
            "class": digit,
            "n": int(mask.sum()),
            "norm_v": _summary(geometry["norm_v"][mask]),
            "norm_class": _summary(geometry["norm_class"][mask]),
            "ratio": _summary(geometry["ratio"][mask]),
            "rho": _summary(geometry["rho"][mask]),
            "alignment": _summary(geometry["alignment"][mask]),
            "delta_s_actual": {**_summary(deltas["actual"][mask]), "positive_fraction": float((deltas["actual"][mask] > 0).to(torch.float64).mean())},
            "delta_s_match": {**_summary(deltas["match"][mask]), "positive_fraction": float((deltas["match"][mask] > 0).to(torch.float64).mean())},
            "delta_s_full": {**_summary(deltas["full"][mask]), "positive_fraction": float((deltas["full"][mask] > 0).to(torch.float64).mean())},
        })
    positive_class_counts = {name: sum(float(item[f"delta_s_{name}"]["mean"]) > 0.0 for item in per_class) for name in ("actual", "match", "full")}
    actual_gain = bootstrap_summaries["actual"]["mean"] > 0.0 and bootstrap_summaries["actual"]["ci_lower"] > 0.0
    match_gain = bootstrap_summaries["match"]["mean"] > 0.0 and bootstrap_summaries["match"]["ci_lower"] > 0.0
    full_gain = bootstrap_summaries["full"]["mean"] > 0.0 and bootstrap_summaries["full"]["ci_lower"] > 0.0
    if actual_gain:
        classification = SUBTHRESHOLD
    elif match_gain and positive_class_counts["match"] >= 2:
        classification = DIRECTION_MISALIGNMENT
    elif (not match_gain) and full_gain and positive_class_counts["full"] >= 2:
        classification = MAGNITUDE_LIMITED
    else:
        classification = MIXED
    _require(classification in CLASSIFICATIONS, "diagnosis classification is not frozen")
    sufficient = None
    if persist_sufficient is not None:
        sufficient = persist_sufficient({
            "selection_indices": selection_indices,
            "labels": labels,
            "norm_v": geometry["norm_v"],
            "norm_class": geometry["norm_class"],
            "ratio": geometry["ratio"],
            "rho": geometry["rho"],
            "alignment": geometry["alignment"],
            "delta_s_actual": deltas["actual"],
            "delta_s_match": deltas["match"],
            "delta_s_full": deltas["full"],
        })
    return {
        "geometry": geometry,
        "raw": {"base": raw_base, "actual": raw_actual, "match": raw_match, "full": raw_full},
        "logits": logits,
        "margins": margins,
        "deltas": deltas,
        "bootstrap_summaries": bootstrap_summaries,
        "bootstrap_tensors": bootstrap_tensors,
        "raw_changes": raw_changes,
        "baseline_prediction": baseline_prediction,
        "endpoint_predictions": endpoint_predictions,
        "per_class": per_class,
        "positive_class_mean_counts": positive_class_counts,
        "flags": {"ACTUAL_CONTINUOUS_SEMANTIC_GAIN": actual_gain, "MATCHED_NORM_SEMANTIC_GAIN": match_gain, "FULL_CLASS_DIRECTION_SEMANTIC_GAIN": full_gain},
        "classification": classification,
        "sufficient": sufficient,
    }


def _save_sufficient(root: Path, target: Path, observations: Mapping[str, torch.Tensor], selection_indices_sha256: str = SELECTION_INDICES_SHA256) -> dict[str, Any]:
    path = target / SUFFICIENT_NAME
    payload = {"schema": SUFFICIENT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "protocol": {"role": "frozen sufficient observations; no optical rerun", "selection_indices_sha256": selection_indices_sha256, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED}, "observations": {name: value.detach().cpu().contiguous() for name, value in observations.items()}}
    atomic_torch_save_new(path, payload, artifact_kind="V5 transfer-coupling sufficient observations")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V5 transfer-coupling sufficient observations")
    loaded_obs = _mapping(loaded.get("observations"), "reloaded V5 sufficient observations")
    _require(set(loaded_obs) == set(observations), "sufficient observation tensor set changed on reload")
    for name, value in observations.items():
        _require(torch.is_tensor(loaded_obs[name]) and tensor_sha256(loaded_obs[name]) == tensor_sha256(value), f"sufficient observation changed on reload: {name}")
    return {"path": _relative(root, path), "sha256": digest, "schema": SUFFICIENT_SCHEMA, "reloaded": True, "observation_tensors": sorted(observations), "selection_indices_sha256": selection_indices_sha256}


def _save_raw(root: Path, target: Path, result: Mapping[str, Any], labels: torch.Tensor, selection_indices: torch.Tensor, selection_indices_sha256: str = SELECTION_INDICES_SHA256) -> dict[str, Any]:
    path = target / RAW_NAME
    geometry = result["geometry"]
    payload = {"schema": RAW_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "selection_indices": selection_indices.cpu(), "labels": labels.cpu(), "selection_indices_sha256": selection_indices_sha256, "z_control": geometry["z_control"].cpu(), "z_candidate": geometry["z_candidate"].cpu(), "delta_v": geometry["delta_v"].cpu(), "delta_class": geometry["delta_class"].cpu(), "z_match": geometry["z_match"].cpu(), "z_full": geometry["z_full"].cpu(), "nearest_incorrect": geometry["nearest_incorrect"].cpu(), "raw": {name: value.cpu() for name, value in result["raw"].items()}, "smooth_logits": {name: value.cpu() for name, value in result["logits"].items()}}
    atomic_torch_save_new(path, payload, artifact_kind="V5 transfer-coupling raw outputs")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V5 transfer-coupling raw outputs")
    _require(loaded.get("schema") == RAW_SCHEMA and loaded.get("selection_indices_sha256") == selection_indices_sha256 and tensor_sha256(loaded.get("selection_indices")) == tensor_sha256(selection_indices), "raw output reload identity changed")
    for name, value in payload["raw"].items():
        _require(tensor_sha256(loaded["raw"][name]) == tensor_sha256(value), f"raw output changed on reload: {name}")
    return {"path": _relative(root, path), "sha256": digest, "schema": RAW_SCHEMA, "reloaded": True, "selection_indices_sha256": selection_indices_sha256}


def _save_bootstrap(root: Path, target: Path, result: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    path = target / BOOTSTRAP_NAME
    payload = {"schema": BOOTSTRAP_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "selection_labels_sha256": tensor_sha256(labels), "protocol": {"type": "paired class-stratified", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "quantile": "torch.quantile interpolation=linear"}, "summaries": result["bootstrap_summaries"], "class_draws": result["bootstrap_tensors"]["class_draws"], "macro_deltas": result["bootstrap_tensors"]["macro_deltas"]}
    atomic_torch_save_new(path, payload, artifact_kind="V5 transfer-coupling paired bootstrap")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V5 transfer-coupling paired bootstrap")
    _require(loaded.get("schema") == BOOTSTRAP_SCHEMA and loaded.get("protocol", {}).get("seed") == BOOTSTRAP_SEED, "bootstrap reload protocol changed")
    for name, value in result["bootstrap_tensors"]["macro_deltas"].items():
        _require(tensor_sha256(loaded["macro_deltas"][name]) == tensor_sha256(value), f"bootstrap macro deltas changed on reload: {name}")
    return {"path": _relative(root, path), "sha256": digest, "schema": BOOTSTRAP_SCHEMA, "reloaded": True, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}


def _json_endpoint(result: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {"raw": _raw_summary(result["raw"][name]), "raw_change_vs_base": None if name == "base" else result["raw_changes"][name], "smooth_margin": _summary(result["margins"][name]), "prediction": result["baseline_prediction"] if name == "base" else result["endpoint_predictions"][name]}


def _build_report_fields(result: Mapping[str, Any], labels: torch.Tensor, selection_indices: torch.Tensor, artifacts: Mapping[str, Any], samples_per_class: int, selection_seed: int, selection_indices_sha256: str) -> dict[str, Any]:
    geometry = result["geometry"]
    return {
        "latent_displacement": {"control_z_tensor_sha256": tensor_sha256(geometry["z_control"]), "candidate_z_tensor_sha256": tensor_sha256(geometry["z_candidate"]), "delta_v_tensor_sha256": tensor_sha256(geometry["delta_v"]), "delta_v_norm": _summary(geometry["norm_v"]), "finite": True},
        "class_subspace": {"definition": "span of realified centered frozen complex prototypes mu_k-mean(mu)", "inner_product": "Re(conj(a)^T b), equivalent to concat(Re,Im) Euclidean inner product", "rank_rule": "torch.linalg.matrix_rank on the realified 10x200 centered centroid matrix with default numerical tolerance", "effective_rank": geometry["rank"], "singular_values": _summary(geometry["singular_values"]), "rho": _summary(geometry["rho"]), "rho_per_class": _class_summary(geometry["rho"], labels, samples_per_class)},
        "class_direction": {"selection": "argmin frozen gauge-invariant complex-OLS distance over k != y using z_control only; torch.argmin lowest-index tie break", "nearest_incorrect_tensor_sha256": tensor_sha256(geometry["nearest_incorrect"]), "pair_counts": [{"source_class": int(pair[0]), "target_class": int(pair[1]), "n": int(((labels == pair[0]) & (geometry["nearest_incorrect"] == pair[1])).sum())} for pair in torch.unique(geometry["class_direction_pair"], dim=0).tolist()], "delta_class_norm": _summary(geometry["norm_class"]), "norm_ratio_r": _summary(geometry["ratio"]), "alignment_cosine": _summary(geometry["alignment"]), "per_class": [{"class": item["class"], "n": item["n"], "norm_v": item["norm_v"], "norm_class": item["norm_class"], "ratio": item["ratio"], "rho": item["rho"], "alignment": item["alignment"]} for item in result["per_class"]]},
        "endpoints": {"baseline": _json_endpoint(result, "base"), "actual_v5": _json_endpoint(result, "actual"), "matched_norm_class_direction": _json_endpoint(result, "match"), "full_class_direction": _json_endpoint(result, "full")},
        "smooth_continuous_margin": {"definition": "score_y(raw)-max_{k!=y} score_k(raw), using SmoothMNISTEvaluator v2.1 canonical pre-softmax logits", "actual": {"statistics": _summary(result["deltas"]["actual"]), "bootstrap": result["bootstrap_summaries"]["actual"], "per_class": [{"class": item["class"], "n": item["n"], "summary": item["delta_s_actual"], "positive_fraction": item["delta_s_actual"]["positive_fraction"]} for item in result["per_class"]]}, "match": {"statistics": _summary(result["deltas"]["match"]), "bootstrap": result["bootstrap_summaries"]["match"], "per_class": [{"class": item["class"], "n": item["n"], "summary": item["delta_s_match"], "positive_fraction": item["delta_s_match"]["positive_fraction"]} for item in result["per_class"]]}, "full": {"statistics": _summary(result["deltas"]["full"]), "bootstrap": result["bootstrap_summaries"]["full"], "per_class": [{"class": item["class"], "n": item["n"], "summary": item["delta_s_full"], "positive_fraction": item["delta_s_full"]["positive_fraction"]} for item in result["per_class"]]}},
        "directional_consistency": {"per_class": result["per_class"], "positive_class_mean_counts": result["positive_class_mean_counts"], "broadness_rule": "at least two of ten class mean effects positive; descriptive broadness only, not a rho threshold", "assessment": "broad" if all(count >= 2 for count in result["positive_class_mean_counts"].values()) else "class-heterogeneous_or_isolated"},
        "primary_evidence_flags": result["flags"],
        "classification": result["classification"],
        "artifacts": dict(artifacts),
        "selection": {"rows": int(labels.shape[0]), "per_class": samples_per_class, "seed": selection_seed, "indices_sha256": selection_indices_sha256, "labels_sha256": tensor_sha256(labels)},
    }


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha: str) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_TRANSFER_COUPLING_NOT_COMPLETED", "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha}, "provenance": {"git_start": dict(started)}, "claim_boundary": {"diagnosis_authorization_consumed": True, "diagnosis_executed": False, "scientific_forward_count": 0, "optimizer_created": False, "backward_executed": False, "parameter_update_executed": False, "optical_update_executed": False, "training_executed": False, "v5_retry_executed": False, "v6_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False, "generation_readiness": "not_ready"}, "next": None}


def _resolve_output(root: Path, value: Path | None) -> Path:
    target = (value or (root / DEFAULT_OUTPUT_DIR)).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "diagnosis output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing diagnosis output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _resolve_diagnosis_dependencies(root: Path) -> dict[str, Any]:
    """Resolve the real frozen authorities without opening scientific DEV rows."""

    started = git_facts(root)
    _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from authorized V5 transfer-coupling entry")
    v5, preflight, v2_contract, contract, authorities, v5_report, v5_lineage, v5_input, dependency_audit = _validate_v5_authority(root)
    v4_checkpoint, v4_state = _load_v4_generator(root)
    control_checkpoint, control_state = _load_checkpoint(root, v5_report, "control")
    candidate_checkpoint, candidate_state = _load_checkpoint(root, v5_report, "candidate")
    _require(torch.equal(control_state["generator.phase"], candidate_state["generator.phase"]), "V5 control/candidate generator state differs")
    _require(torch.equal(control_state["ols.operator.transport"], candidate_state["ols.operator.transport"]), "V5 control/candidate OLS transport differs")
    _require(torch.equal(control_state["generator.phase"], v4_state["generator.phase"]), "V5 generator does not match frozen V4 generator")
    _require(torch.equal(control_state["ols.operator.transport"], v4_state["ols.operator.transport"]), "V5 OLS transport does not match frozen V4 transport")
    _require(torch.equal(control_checkpoint["prototypes"], candidate_checkpoint["prototypes"]), "V5 control/candidate frozen prototypes differ")
    prototypes = control_checkpoint["prototypes"].detach().cpu().contiguous()
    cuda_environment = _prepare_cuda_runtime_dll_search_path()
    device, device_facts = _cuda_facts()
    _require(tuple(device_facts.get("compute_capability", ())) == EXPECTED_COMPUTE_CAPABILITY and int(device_facts.get("vram_bytes", 0)) >= EXPECTED_VRAM_BYTES, "CUDA target differs from accepted V5 runtime")
    _require(shutil.disk_usage((root / "outputs").resolve()).free >= MIN_FREE_BYTES_FOR_OUTPUT, "insufficient output storage headroom")
    smooth, smooth_identity = _load_smooth_evaluator(root, authorities, device)
    selection_authority = _load_registered_selection_authority(root, authorities)
    runtime_consumer = _new_frozen_consumer(v2_contract, v4_state, device)
    runtime_identity = dict(runtime_consumer.runtime_identity)
    runtime_identity["runtime_state_tree_sha256"] = _digest_tree(_cpu_tree(runtime_consumer.runtime.state_dict()))
    del runtime_consumer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "v5": v5,
        "preflight": preflight,
        "v2_contract": v2_contract,
        "contract": contract,
        "authorities": authorities,
        "v5_report": v5_report,
        "v5_lineage": v5_lineage,
        "v5_input": v5_input,
        "dependency_audit": dependency_audit,
        "v4_checkpoint": v4_checkpoint,
        "v4_state": v4_state,
        "control_checkpoint": control_checkpoint,
        "control_state": control_state,
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_state": candidate_state,
        "prototypes": prototypes,
        "cuda_environment": cuda_environment,
        "device": device,
        "device_facts": device_facts,
        "smooth": smooth,
        "smooth_identity": smooth_identity,
        "selection_authority": selection_authority,
        "runtime_identity": runtime_identity,
    }


def _synthetic_provider() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
    samples_per_class = 2
    rows = CLASS_COUNT * samples_per_class
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(samples_per_class)
    ramp = torch.linspace(0.0, 1.0, rows * 28 * 28, dtype=torch.float32).reshape(rows, 1, 28, 28)
    class_offset = labels.to(torch.float32).reshape(rows, 1, 1, 1) / float(CLASS_COUNT * 4)
    images = (ramp * 0.75 + class_offset).clamp(0.0, 1.0).contiguous()
    indices = torch.arange(rows, dtype=torch.long)
    metadata = {
        "source": {"role": "deterministic synthetic provider", "rows_loaded": rows, "scientific_values_read": 0},
        "TRAIN": {"scientific_values_read": 0},
        "DEV": {"scientific_values_read": 0, "selection_rows": rows},
        "calibration": {"rows_read": 0},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "official_test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden; path not opened"},
        "labels_as_model_input": False,
    }
    return images, labels, indices, samples_per_class, 0, metadata


def _run_production_wrapper(
    deps: Mapping[str, Any],
    provider: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]],
    artifact_root: Path,
    target: Path,
    *,
    synthetic: bool,
) -> dict[str, Any]:
    """Run the same resolved consumer path for real or synthetic boundary data."""

    selection_images, selection_labels, selection_indices, samples_per_class, selection_seed, data_metadata = provider()
    _require(selection_images.ndim == 4 and tuple(selection_images.shape[1:]) == (1, 28, 28) and selection_images.dtype == torch.float32, "provider image shape/dtype changed")
    _require(selection_labels.ndim == 1 and selection_labels.dtype == torch.long and selection_labels.shape[0] == selection_images.shape[0], "provider labels shape/dtype changed")
    _require(selection_indices.ndim == 1 and selection_indices.dtype == torch.long and selection_indices.shape == selection_labels.shape, "provider selection indices changed")
    expected_rows = CLASS_COUNT * samples_per_class
    _require(selection_images.shape[0] == expected_rows and samples_per_class > 0, "provider row count changed")
    _require(torch.equal(torch.bincount(selection_labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), samples_per_class, dtype=torch.long)), "provider labels are not balanced")
    selection_indices = selection_indices.cpu().contiguous()
    selection_labels = selection_labels.cpu().contiguous()
    selection_images = selection_images.cpu().contiguous()
    selection_indices_sha256 = tensor_sha256(selection_indices)
    batch_count = math.ceil(expected_rows / LOGICAL_BATCH_SIZE)
    device = deps["device"]
    smooth = deps["smooth"]
    prototypes = deps["prototypes"]
    forward_count = 0
    smooth_evaluation_count = 0

    control_consumer = _new_frozen_consumer(deps["v2_contract"], deps["control_state"], device)
    candidate_consumer = _new_frozen_consumer(deps["v2_contract"], deps["candidate_state"], device)
    z_control, control_encoded_raw, control_state_digest, _ = _encode_dev(control_consumer, selection_images)
    z_candidate, candidate_encoded_raw, candidate_state_digest, _ = _encode_dev(candidate_consumer, selection_images)
    forward_count += 2 * batch_count
    _require(control_state_digest == _digest_tree(_cpu_tree(deps["control_state"])), "V5 control runtime state changed during encoder forward")
    _require(candidate_state_digest == _digest_tree(_cpu_tree(deps["candidate_state"])), "V5 candidate runtime state changed during encoder forward")
    del control_consumer, candidate_consumer, control_encoded_raw, candidate_encoded_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()
    optical_consumer = _new_frozen_consumer(deps["v2_contract"], deps["v4_state"], device)

    def optical_forward(z: torch.Tensor) -> torch.Tensor:
        nonlocal forward_count
        value = _forward_external(optical_consumer, z)
        forward_count += math.ceil(z.shape[0] / LOGICAL_BATCH_SIZE)
        return value

    def smooth_forward(raw: torch.Tensor) -> torch.Tensor:
        nonlocal smooth_evaluation_count
        smooth_evaluation_count += math.ceil(raw.shape[0] / LOGICAL_BATCH_SIZE)
        return _smooth_logits(smooth, raw, device)

    def persist_sufficient(observations: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
        return _save_sufficient(artifact_root, target, observations, selection_indices_sha256)

    result = _run_core(
        z_control=z_control,
        z_candidate=z_candidate,
        prototypes=prototypes,
        labels=selection_labels,
        selection_indices=selection_indices,
        distance_fn=prototype_distance,
        optical_forward=optical_forward,
        smooth_forward=smooth_forward,
        persist_sufficient=persist_sufficient,
        samples_per_class=samples_per_class,
    )
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in smooth.parameters()), "Smooth evaluator received a diagnosis gradient")
    raw_artifact = _save_raw(artifact_root, target, result, selection_labels, selection_indices, selection_indices_sha256)
    bootstrap_artifact = _save_bootstrap(artifact_root, target, result, selection_labels)
    artifacts = {
        "raw_outputs": raw_artifact,
        "paired_bootstrap": bootstrap_artifact,
        "sufficient_observations": result["sufficient"],
    }
    report_fields = _build_report_fields(result, selection_labels, selection_indices, artifacts, samples_per_class, selection_seed, selection_indices_sha256)
    return {
        "result": result,
        "report_fields": report_fields,
        "artifacts": artifacts,
        "data_metadata": data_metadata,
        "selection_indices_sha256": selection_indices_sha256,
        "samples_per_class": samples_per_class,
        "selection_seed": selection_seed,
        "batch_count": batch_count,
        "forward_count": forward_count,
        "smooth_evaluation_count": smooth_evaluation_count,
        "synthetic": synthetic,
    }


def run_no_data_dry_run(repo_root: Path | None = None) -> dict[str, Any]:
    """Resolve all real authorities and stop before the first real DEV row read."""

    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    deps = _resolve_diagnosis_dependencies(root)
    return {
        "result": "NO_DATA_DRY_RUN_PASSED",
        "status": "valid_completed",
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "scientific_meaning": "none; no-data engineering acceptance only",
        "authority_resolution": {
            "v5_contract_schema": deps["contract"].get("schema"),
            "v5_scientific_input_schema": deps["v5_input"].get("schema"),
            "accepted_preflight": deps["authorities"].get("accepted_preflight"),
            "registered_selection_authority": deps["selection_authority"],
        },
        "dependency_audit": deps["dependency_audit"],
        "runtime_assembly": {"device": deps["device_facts"], "runtime_identity": deps["runtime_identity"], "smooth_identity": deps["smooth_identity"]},
        "data_boundary": {
            "real_train_scientific_reads": 0,
            "real_dev_scientific_reads": 0,
            "scientific_forward": 0,
            "smooth_scientific_evaluation": 0,
            "stop_boundary": "before real DEV row read or forward",
        },
        "claim_boundary": {"scientific_execution": False, "optimizer": False, "backward": False, "update": False, "stage_b": False, "generation": False, "formal_promotion": False},
    }


def _run_synthetic(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    deps = _resolve_diagnosis_dependencies(root)
    with tempfile.TemporaryDirectory(prefix="lightgen_v5_transfer_coupling_") as temporary:
        target = Path(temporary)
        bundle = _run_production_wrapper(deps, _synthetic_provider, target, target, synthetic=True)
        report = {
            "schema": REPORT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "result": "SYNTHETIC_ONLY",
            "status": "valid_completed",
            "classification": "SYNTHETIC_ONLY",
            "scientific_meaning": "none; synthetic acceptance only",
            "computed_classification_non_authoritative": bundle["result"]["classification"],
            "dependency_audit": deps["dependency_audit"],
            "runtime": {"device": deps["device_facts"], "runtime_identity": deps["runtime_identity"], "transport": "T=I_100_coefficient_space", "modal_projection_boundary": True, "shared_w_ols": True, "optimizer": False, "backward": False, "update": False},
            "data": bundle["data_metadata"],
            "execution": {"production_wrapper": True, "scientific_forward_count": bundle["forward_count"], "smooth_evaluation_count": bundle["smooth_evaluation_count"], "report_lineage_reload": True},
            "core": bundle["report_fields"],
            "claim_boundary": {"scientific_execution": False, "optimizer": False, "backward": False, "update": False, "stage_b": False, "generation": False, "formal_promotion": False},
        }
        report_path = target / "synthetic_transfer_coupling_report.v1.json"
        atomic_write_new_json(report_path, report, artifact_kind="synthetic V5 transfer-coupling report")
        reloaded_report = _read_json(report_path, "synthetic transfer-coupling report")
        _require(reloaded_report.get("result") == "SYNTHETIC_ONLY" and reloaded_report.get("scientific_meaning") == "none; synthetic acceptance only", "synthetic report reload failed")
        lineage_path = target / "synthetic_transfer_coupling_lineage.v1.json"
        lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "result": "SYNTHETIC_ONLY", "status": "valid_completed", "report_sha256": file_sha256(report_path), "scientific_meaning": "none; synthetic acceptance only"}
        atomic_write_new_json(lineage_path, lineage, artifact_kind="synthetic V5 transfer-coupling lineage")
        reloaded_lineage = _read_json(lineage_path, "synthetic transfer-coupling lineage")
        _require(reloaded_lineage.get("report_sha256") == file_sha256(report_path), "synthetic lineage reload failed")
        return {"result": "SYNTHETIC_ONLY", "status": "valid_completed", "classification": "SYNTHETIC_ONLY", "computed_classification_non_authoritative": bundle["result"]["classification"], "core_serialization": "passed", "production_wrapper": True, "scientific_forward_count": bundle["forward_count"], "smooth_evaluation_count": bundle["smooth_evaluation_count"], "no_scientific_data": True, "report_lineage_reload": True}


def _legacy_run_diagnosis(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = _resolve_output(root, output_dir)
    started = git_facts(root)
    status_before = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    entrypoint_path = Path(__file__).resolve()
    payload = _base_payload(root, started, file_sha256(entrypoint_path))
    forward_count = 0
    diagnosis_started = False
    input_ref: dict[str, Any] | None = None
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from authorized V5 diagnosis")
        v5, preflight, v2_contract, contract, authorities, v5_report, v5_lineage, v5_input, dependency_audit = _validate_v5_authority(root)
        v4_checkpoint, v4_state = _load_v4_generator(root)
        control_checkpoint, control_state = _load_checkpoint(root, v5_report, "control")
        candidate_checkpoint, candidate_state = _load_checkpoint(root, v5_report, "candidate")
        _require(torch.equal(control_state["generator.phase"], candidate_state["generator.phase"]) and torch.equal(control_state["ols.operator.transport"], candidate_state["ols.operator.transport"]), "V5 control/candidate frozen optical state differs")
        _require(torch.equal(control_state["generator.phase"], v4_state["generator.phase"]), "V5 generator does not match frozen V4 generator")
        _require(torch.equal(control_state["ols.operator.transport"], v4_state["ols.operator.transport"]), "V5 OLS transport does not match frozen V4 transport")
        _require(torch.equal(control_checkpoint["prototypes"], candidate_checkpoint["prototypes"]), "V5 control/candidate frozen prototypes differ")
        prototypes = control_checkpoint["prototypes"].detach().cpu().contiguous()
        cuda_environment = _prepare_cuda_runtime_dll_search_path()
        device, device_facts = _cuda_facts()
        _require(tuple(device_facts.get("compute_capability", ())) == EXPECTED_COMPUTE_CAPABILITY and int(device_facts.get("vram_bytes", 0)) >= EXPECTED_VRAM_BYTES, "CUDA target differs from accepted V5 runtime")
        _require(shutil.disk_usage(target.parent).free >= MIN_FREE_BYTES_FOR_OUTPUT, "insufficient output storage headroom")
        smooth, smooth_identity = _load_smooth_evaluator(root, authorities, device)
        smooth.to(device)
        smooth.freeze()
        smooth.eval()
        _require(all(not parameter.requires_grad and parameter.grad is None for parameter in smooth.parameters()), "Smooth evaluator gradient firewall failed")
        smooth_identity = dict(smooth_identity)
        smooth_identity["cross_domain_equality_required"] = False
        _require(smooth_identity["checkpoint"].get("sha256") == SMOOTH_CHECKPOINT_SHA256, "Smooth evaluator checkpoint identity changed")
        images, labels, selection_indices, samples_per_class, selection_seed, dataset = _load_selection(root, _load_registered_selection_authority(root, authorities))
        selection_images = images.index_select(0, selection_indices).contiguous()
        selection_labels = labels.index_select(0, selection_indices).contiguous()
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorities": {"v5_contract": {"path": V5_CONTRACT_PATH.as_posix(), "sha256": V5_CONTRACT_SHA256}, "v5_scientific_report": {"path": V5_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V5_SCIENTIFIC_REPORT_SHA256}, "v5_scientific_lineage": {"path": V5_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V5_SCIENTIFIC_LINEAGE_SHA256}, "v4_generator_checkpoint": {"path": V4_GENERATOR_CHECKPOINT_PATH.as_posix(), "sha256": V4_GENERATOR_CHECKPOINT_SHA256}, "v5_control_step_300_checkpoint": {"path": str(_mapping(_mapping(v5_report["arms"], "V5 arms")["control"], "V5 control arm")["checkpoints"]["300"]["path"]), "sha256": str(_mapping(_mapping(v5_report["arms"], "V5 arms")["control"], "V5 control arm")["checkpoints"]["300"]["sha256"])}, "v5_candidate_step_300_checkpoint": {"path": str(_mapping(_mapping(v5_report["arms"], "V5 arms")["candidate"], "V5 candidate arm")["checkpoints"]["300"]["path"]), "sha256": str(_mapping(_mapping(v5_report["arms"], "V5 arms")["candidate"], "V5 candidate arm")["checkpoints"]["300"]["sha256"])}, "smooth_evaluator": smooth_identity, "semantic_authority": "V5 scientific attempt-01 valid completed; class-direction replication selection authority"}, "protocol": {"selection_seed": SELECTION_SEED, "selection_rows": SELECTION_ROWS, "selection_per_class": SELECTION_PER_CLASS, "selection_indices_sha256": SELECTION_INDICES_SHA256, "latent_domain": "full complex C^100", "distance": "accepted V5 gauge-invariant complex-OLS distance; CONTROL z only", "class_subspace": "realified centered frozen prototype subspace; default torch.linalg.matrix_rank tolerance", "matched_norm": "z_C + ||delta_V||*delta_class/(||delta_class||+1e-12)", "smooth_margin": "score_y-max_{k!=y} score_k from Smooth v2.1 canonical logits", "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired class-stratified"}, "classification_broadness": "at least two positive per-class means; fixed before execution"}, "data_roles": {"source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000}, "TRAIN": {"rows": 55_000, "used_for": "authority only; no new prototype construction"}, "DEV": {"rows": DEVELOPMENT_ROWS, "selection_rows": SELECTION_ROWS, "selection_indices_sha256": SELECTION_INDICES_SHA256, "values_read_for": "frozen diagnosis only"}, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0, "labels_as_model_input": False}, "claim_boundary": payload["claim_boundary"], "input_self_hash": "not_embedded_to_avoid_self_reference"}
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V5 transfer-coupling diagnosis input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload["input"] = input_ref
        diagnosis_started = True
        control_consumer = _new_frozen_consumer(v2_contract, control_state, device)
        candidate_consumer = _new_frozen_consumer(v2_contract, candidate_state, device)
        z_control, control_encoded_raw, control_state_digest, _ = _encode_dev(control_consumer, selection_images)
        z_candidate, candidate_encoded_raw, candidate_state_digest, _ = _encode_dev(candidate_consumer, selection_images)
        forward_count += 2 * (SELECTION_ROWS // LOGICAL_BATCH_SIZE)
        _require(control_state_digest == _digest_tree(_cpu_tree(control_state)) and candidate_state_digest == _digest_tree(_cpu_tree(candidate_state)), "V5 checkpoint runtime state changed during encoder forward")
        del control_consumer, candidate_consumer, control_encoded_raw, candidate_encoded_raw
        optical_consumer = _new_frozen_consumer(v2_contract, v4_state, device)

        def optical_forward(z: torch.Tensor) -> torch.Tensor:
            nonlocal forward_count
            value = _forward_external(optical_consumer, z)
            forward_count += SELECTION_ROWS // LOGICAL_BATCH_SIZE
            return value

        def smooth_forward(raw: torch.Tensor) -> torch.Tensor:
            return _smooth_logits(smooth, raw, device)

        def distance_fn(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
            return prototype_distance(query, support)

        def persist_sufficient(observations: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
            return _save_sufficient(root, target, observations)

        result = _run_core(z_control=z_control, z_candidate=z_candidate, prototypes=prototypes, labels=selection_labels, selection_indices=selection_indices, distance_fn=distance_fn, optical_forward=optical_forward, smooth_forward=smooth_forward, persist_sufficient=persist_sufficient)
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in smooth.parameters()), "Smooth evaluator received a diagnosis gradient")
        raw_artifact = _save_raw(root, target, result, selection_labels, selection_indices)
        bootstrap_artifact = _save_bootstrap(root, target, result, selection_labels)
        artifacts = {"input": input_ref, "raw_outputs": raw_artifact, "paired_bootstrap": bootstrap_artifact, "sufficient_observations": result["sufficient"]}
        report_fields = _build_report_fields(result, selection_labels, selection_indices, artifacts, samples_per_class, selection_seed, tensor_sha256(selection_indices))
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": result["classification"], "outer_gate": OUTER_GATE, "authorities": {"v5_contract": {"path": V5_CONTRACT_PATH.as_posix(), "sha256": V5_CONTRACT_SHA256}, "v5_scientific_report": {"path": V5_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V5_SCIENTIFIC_REPORT_SHA256}, "v5_scientific_lineage": {"path": V5_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V5_SCIENTIFIC_LINEAGE_SHA256}, "v4_generator_checkpoint": {"path": V4_GENERATOR_CHECKPOINT_PATH.as_posix(), "sha256": V4_GENERATOR_CHECKPOINT_SHA256, "generator_phase_tensor_sha256": tensor_sha256(v4_state["generator.phase"])}, "v5_prototypes": {"tensor_sha256": tensor_sha256(prototypes), "source": "frozen V5 TRAIN prototypes"}, "smooth_evaluator": smooth_identity, "dev_selection": {"seed": SELECTION_SEED, "rows": SELECTION_ROWS, "indices_sha256": SELECTION_INDICES_SHA256}}, "runtime": {"cuda_dll_search_path": cuda_environment, "device": device_facts, "grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "transport": "T=I_100_coefficient_space", "modal_projection_boundary": True, "shared_w_ols": True, "optimizer": False, "backward": False, "update": False}, "data": {**dataset, "development_selection_rows": SELECTION_ROWS, "development_selection_per_class": SELECTION_PER_CLASS, "selection_indices_sha256": SELECTION_INDICES_SHA256, "calibration_rows_read": 0, "protected_rows_read": 0, "official_test_rows_read": 0, "sealed_rows_read": 0}, "execution": {"diagnosis_executed": True, "scientific_forward_count": forward_count, "control_encoder_forwards": SELECTION_ROWS // LOGICAL_BATCH_SIZE, "candidate_encoder_forwards": SELECTION_ROWS // LOGICAL_BATCH_SIZE, "optical_endpoint_forwards_per_endpoint": SELECTION_ROWS // LOGICAL_BATCH_SIZE, "smooth_endpoint_evaluations": 4 * (SELECTION_ROWS // LOGICAL_BATCH_SIZE)}, **report_fields, "governance": {"optimizer": False, "backward": False, "parameter_update": False, "optical_update": False, "training": False, "v5_retry": False, "v6": False, "stage_b": False, "scale_extraction": False, "prior_shaping": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "generation_readiness": "not_ready"}, "fact_model_assumption_design_decision": {"FACT": ["V5 control/candidate latent and frozen optical endpoint measurements", "measured class-subspace, nearest-prototype, raw-CMOS and Smooth-margin effects"], "MODEL_ASSUMPTION": ["classification interprets matched/full canary patterns as displacement direction or magnitude evidence"], "DESIGN_DECISION": ["nearest incorrect prototype selected from CONTROL z only with lowest-index tie break", "matched-norm canary and full class-direction reference", "fixed Smooth correct-class continuous margin", "broadness means at least two positive per-class means"]}, "next": "no_retry; diagnosis does not authorize V6, Stage B, generation or formal promotion"})
    except DiagnosisBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no retry without new authority"})
    except InvalidDiagnosis as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_TRANSFER_COUPLING_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no retry under one-shot authorization"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_TRANSFER_COUPLING_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no retry under one-shot authorization"})
    finally:
        ending = git_facts(root)
        status_after = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == status_after, "unexpected_modified_paths": [] if status_before == status_after else sorted(set(status_after).symmetric_difference(status_before)), "diagnosis_started": diagnosis_started})
        payload["claim_boundary"].update({"diagnosis_executed": bool(payload.get("result") == "VALID_COMPLETED"), "scientific_forward_count": forward_count, "optimizer_created": False, "backward_executed": False, "parameter_update_executed": False, "optical_update_executed": False, "training_executed": False, "v5_retry_executed": False, "v6_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload.setdefault("artifacts", {})
        payload["artifacts"].setdefault("input", input_ref)
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"
        payload["forbidden_actions"] = {"optimizer": False, "backward": False, "update": False, "training": False, "v5_retry": False, "v6": False, "stage_b": False, "scale": False, "prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V5 transfer-coupling diagnosis report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "entrypoint": payload["entrypoint"], "authorities": payload.get("authorities"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_forward_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V5 transfer-coupling diagnosis lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_forward_count": forward_count, "optimizer": False, "backward": False, "update": False, "training": False, "v5_retry": False, "v6": False, "stage_b": False, "generation": False, "protected_test_sealed": False, "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_diagnosis(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = _resolve_output(root, output_dir)
    started = git_facts(root)
    status_before = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    payload = _base_payload(root, started, file_sha256(Path(__file__).resolve()))
    forward_count = 0
    diagnosis_started = False
    input_ref: dict[str, Any] | None = None
    try:
        deps = _resolve_diagnosis_dependencies(root)
        v5_report = deps["v5_report"]
        input_payload = {
            "schema": INPUT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorities": {
                "v5_contract": {"path": V5_CONTRACT_PATH.as_posix(), "sha256": V5_CONTRACT_SHA256},
                "v5_scientific_report": {"path": V5_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V5_SCIENTIFIC_REPORT_SHA256},
                "v5_scientific_lineage": {"path": V5_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V5_SCIENTIFIC_LINEAGE_SHA256},
                "accepted_preflight": deps["authorities"].get("accepted_preflight"),
                "v4_generator_checkpoint": {"path": V4_GENERATOR_CHECKPOINT_PATH.as_posix(), "sha256": V4_GENERATOR_CHECKPOINT_SHA256},
                "v5_control_step_300_checkpoint": dict(_mapping(_mapping(v5_report["arms"], "V5 arms")["control"], "V5 control arm")["checkpoints"]["300"]),
                "v5_candidate_step_300_checkpoint": dict(_mapping(_mapping(v5_report["arms"], "V5 arms")["candidate"], "V5 candidate arm")["checkpoints"]["300"]),
                "smooth_evaluator": deps["smooth_identity"],
                "class_direction_replication": deps["selection_authority"],
                "dependency_audit": deps["dependency_audit"],
            },
            "protocol": {"selection_seed": SELECTION_SEED, "selection_rows": SELECTION_ROWS, "selection_per_class": SELECTION_PER_CLASS, "selection_indices_sha256": SELECTION_INDICES_SHA256, "latent_domain": "full complex C^100", "distance": "accepted V5 gauge-invariant complex-OLS distance; CONTROL z only", "class_subspace": "realified centered frozen prototype subspace; default torch.linalg.matrix_rank tolerance", "matched_norm": "z_C + ||delta_V||*delta_class/(||delta_class||+1e-12)", "smooth_margin": "score_y-max_{k!=y} score_k from Smooth v2.1 canonical logits", "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired class-stratified"}, "classification_broadness": "at least two positive per-class means; fixed before execution"},
            "data_roles": {"source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000}, "TRAIN": {"rows": 55_000, "used_for": "authority only; no new prototype construction"}, "DEV": {"rows": DEVELOPMENT_ROWS, "selection_rows": SELECTION_ROWS, "selection_indices_sha256": SELECTION_INDICES_SHA256, "values_read_for": "frozen diagnosis only"}, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0, "labels_as_model_input": False},
            "claim_boundary": payload["claim_boundary"],
            "input_self_hash": "not_embedded_to_avoid_self_reference",
        }
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V5 transfer-coupling diagnosis input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload["input"] = input_ref
        diagnosis_started = True

        def real_provider() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
            return _load_selection(root, deps["selection_authority"])

        bundle = _run_production_wrapper(deps, real_provider, root, target, synthetic=False)
        forward_count = bundle["forward_count"]
        report_fields = dict(bundle["report_fields"])
        report_fields["artifacts"] = {"input": input_ref, **bundle["artifacts"]}
        batch_count = bundle["batch_count"]
        payload.update({
            "result": "VALID_COMPLETED",
            "status": "valid_completed",
            "classification": bundle["result"]["classification"],
            "outer_gate": OUTER_GATE,
            "authorities": input_payload["authorities"],
            "runtime": {"cuda_dll_search_path": deps["cuda_environment"], "device": deps["device_facts"], "runtime_identity": deps["runtime_identity"], "grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "transport": "T=I_100_coefficient_space", "modal_projection_boundary": True, "shared_w_ols": True, "optimizer": False, "backward": False, "update": False},
            "data": {**bundle["data_metadata"], "development_selection_rows": SELECTION_ROWS, "development_selection_per_class": SELECTION_PER_CLASS, "selection_indices_sha256": bundle["selection_indices_sha256"], "calibration_rows_read": 0, "protected_rows_read": 0, "official_test_rows_read": 0, "sealed_rows_read": 0},
            "execution": {"diagnosis_executed": True, "scientific_forward_count": forward_count, "control_encoder_forwards": batch_count, "candidate_encoder_forwards": batch_count, "optical_endpoint_forwards_per_endpoint": batch_count, "smooth_endpoint_evaluations": bundle["smooth_evaluation_count"]},
            **report_fields,
            "governance": {"optimizer": False, "backward": False, "parameter_update": False, "optical_update": False, "training": False, "v5_retry": False, "v6": False, "stage_b": False, "scale_extraction": False, "prior_shaping": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "generation_readiness": "not_ready"},
            "fact_model_assumption_design_decision": {"FACT": ["V5 control/candidate latent and frozen optical endpoint measurements", "measured class-subspace, nearest-prototype, raw-CMOS and Smooth-margin effects"], "MODEL_ASSUMPTION": ["classification interprets matched/full canary patterns as displacement direction or magnitude evidence"], "DESIGN_DECISION": ["nearest incorrect prototype selected from CONTROL z only with lowest-index tie break", "matched-norm canary and full class-direction reference", "fixed Smooth correct-class continuous margin", "broadness means at least two positive per-class means"]},
            "next": "no_retry; diagnosis does not authorize V6, Stage B, generation or formal promotion",
        })
    except DiagnosisBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no retry without new authority"})
    except InvalidDiagnosis as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_TRANSFER_COUPLING_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no retry under one-shot authorization"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_TRANSFER_COUPLING_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no retry under one-shot authorization"})
    finally:
        ending = git_facts(root)
        status_after = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == status_after, "unexpected_modified_paths": [] if status_before == status_after else sorted(set(status_after).symmetric_difference(status_before)), "diagnosis_started": diagnosis_started})
        payload["claim_boundary"].update({"diagnosis_executed": bool(payload.get("result") == "VALID_COMPLETED"), "scientific_forward_count": forward_count, "optimizer_created": False, "backward_executed": False, "parameter_update_executed": False, "optical_update_executed": False, "training_executed": False, "v5_retry_executed": False, "v6_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload.setdefault("artifacts", {})
        payload["artifacts"].setdefault("input", input_ref)
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"
        payload["forbidden_actions"] = {"optimizer": False, "backward": False, "update": False, "training": False, "v5_retry": False, "v6": False, "stage_b": False, "scale": False, "prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V5 transfer-coupling diagnosis report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "entrypoint": payload["entrypoint"], "authorities": payload.get("authorities"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_forward_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V5 transfer-coupling diagnosis lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_forward_count": forward_count, "optimizer": False, "backward": False, "update": False, "training": False, "v5_retry": False, "v6": False, "stage_b": False, "generation": False, "protected_test_sealed": False, "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("dry-run", "synthetic", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.action == "dry-run":
        print(json.dumps(run_no_data_dry_run(repo_root=args.repo_root), indent=2, ensure_ascii=False))
        return 0
    if args.action == "synthetic":
        print(json.dumps(_run_synthetic(repo_root=args.repo_root), indent=2, ensure_ascii=False))
        return 0
    result = run_diagnosis(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if result["result"] == "VALID_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
