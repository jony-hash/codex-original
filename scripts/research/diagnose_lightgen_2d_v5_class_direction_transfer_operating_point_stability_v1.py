"""Diagnose cyclic class-direction transfer at the V5 operating points.

This is a single-use, forward-only consumer of the frozen V5 control and
candidate step-300 encoders.  It applies the already-authorized intervention

    z_shift = z - mu_y + mu_(y+1 mod 10)

through the frozen T=I_100 / W_OLS / V4 generator path and measures the
result in the historical TRAIN-template and SmoothMNISTEvaluator spaces.
The script never creates an optimizer, enables backward, updates a parameter,
reads protected/test/sealed data, performs generation, or opens Stage B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    LightGen2DComplexOLSTrainingConsumerV2,
    file_sha256,
    git_facts,
    load_v2_training_contract,
    tensor_sha256,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_local_mnist, stratified_mnist_indices


TASK_ID = "DIAGNOSE_V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_V1"
OUTER_GATE = "V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_RESOLVED"
REPORT_SCHEMA = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_report.v1"
INPUT_SCHEMA = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_lineage.v1"
RAW_SCHEMA = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_raw_outputs.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_bootstrap.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_sufficient_observations.v1"
AUDIT_SCHEMA = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_attempt_03_integrity_audit.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01"
)
INPUT_NAME = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_input.v1.json"
REPORT_NAME = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_report.v1.json"
LINEAGE_NAME = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_lineage.v1.json"
RAW_NAME = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_raw_outputs.v1.pt"
BOOTSTRAP_NAME = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_bootstrap.v1.pt"
SUFFICIENT_NAME = "lightgen_2d_v5_class_direction_transfer_operating_point_stability_sufficient_observations.v1.pt"
AUDIT_NAME = "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_attempt_03_integrity_audit.v1.json"

HISTORICAL_INPUT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1.json"
)
HISTORICAL_REPORT_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_report.v1.json"
)
HISTORICAL_LINEAGE_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_lineage.v1.json"
)
HISTORICAL_RAW_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1.pt"
)
HISTORICAL_INPUT_SHA256 = "c43b333822cab7c5ebafb5e64cdce7a50f386e662388f071d9378c740ed04b1d"
HISTORICAL_REPORT_SHA256 = "f5b1b252b5815578fd09367258df66e7ba8f3ea008fc889cb30c23ec8d18c249"
HISTORICAL_LINEAGE_SHA256 = "7a75ddb6008a65057c86e658104b6ccf61b14bd9091382f2101d55dae70440e7"
HISTORICAL_RAW_SHA256 = "e985f6f030591a1aa38d33517d35da6c831806e121c9f73c3162395c23336d00"

V5_ATTEMPT_03_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_03"
)
V5_ATTEMPT_03_REPORT_PATH = V5_ATTEMPT_03_DIR / (
    "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json"
)
V5_ATTEMPT_03_LINEAGE_PATH = V5_ATTEMPT_03_DIR / (
    "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json"
)
V5_ATTEMPT_03_SUFFICIENT_PATH = V5_ATTEMPT_03_DIR / (
    "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_sufficient_observations.v1.pt"
)
V5_ATTEMPT_03_REPORT_SHA256 = "b759edd433cf53961d8b1a4f9cf61a341fd5a75a56c0e2985969bcf1799f791e"
V5_ATTEMPT_03_LINEAGE_SHA256 = "71c11240c87b79478c047fd9d6688f0e2f5a789e3d76a2e616bccd72d22eb6bc"
V5_ATTEMPT_03_SUFFICIENT_SHA256 = "8985d1b48cb96e1f684ee263ebc050144802cf610c9806bed9cd21fc1c49b6ef"
V5_ATTEMPT_03_ENTRYPOINT_SHA256 = "7c43cfe943d945d04b5c65ddd9d8b5e61b18ca8a7180cc9238cae980a6ffd4ce"

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
V5_SCIENTIFIC_LINEAGE_PATH = V5_SCIENTIFIC_REPORT_PATH.with_name(
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_lineage.v1.json"
)
V5_SCIENTIFIC_REPORT_SHA256 = "41aeb6e5ae5d1736521b164f2c8e9267bc3687801ab99b7142692112f8d85ce8"
V5_SCIENTIFIC_LINEAGE_SHA256 = "89e00a5f67dc082b0d2f37042a66a796cae71f14b1f676e8a27e94caaeed4c50"

V4_GENERATOR_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_v4_step_300_checkpoint.v1.pt"
)
V4_GENERATOR_SHA256 = "de80f2826419ca6ced28c8e1fc6d8804ef513493a49474e5997cfb4cb59bdf80"
V5_CONTROL_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_control_step_300_checkpoint.v1.pt"
)
V5_CONTROL_SHA256 = "abf62784d73a3ef7a5d51143642c42baf1472ce8da20af3f3f7a8392af32549f"
V5_CANDIDATE_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_candidate_step_300_checkpoint.v1.pt"
)
V5_CANDIDATE_SHA256 = "74daa0e78acf2cd1a1f2460f512519e3f7ae81994c5be0bcc5004e21b9e6bf98"
SMOOTH_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
SMOOTH_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
SMOOTH_STATE_TREE_SHA256 = "9ddfe69eefb6bbf3d75dfb694b621b20b6ab53591cf73cc75a43f04493003664"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SELECTION_LABELS_SHA256 = "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
EXPECTED_PROTOTYPES_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
EXPECTED_S_MU = 0.00555040966719389
EXPECTED_CANDIDATE_LAMBDA = 0.09573533218209243

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
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000

CONTROL_TRANSFER_LOSS = "V5_CONTROL_OPERATING_POINT_TRANSFER_LOSS_SUPPORTED"
PROTOTYPE_TRAINING_DEGRADATION = "V5_PROTOTYPE_TRAINING_TRANSFER_DEGRADATION_SUPPORTED"
KNOWN_DIRECTIONS_RETAINED = "V5_KNOWN_CLASS_DIRECTIONS_RETAINED"
MIXED_OR_UNRESOLVED = "V5_CLASS_DIRECTION_OPERATING_POINT_MIXED_OR_UNRESOLVED"
CLASSIFICATIONS = frozenset(
    {CONTROL_TRANSFER_LOSS, PROTOTYPE_TRAINING_DEGRADATION, KNOWN_DIRECTIONS_RETAINED, MIXED_OR_UNRESOLVED}
)


class InvalidDiagnosis(RuntimeError):
    """The frozen diagnosis harness or authority is invalid."""


class DiagnosisBlocked(RuntimeError):
    """A valid runtime boundary blocks the one-shot diagnosis."""

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


def _path(root: Path, value: str | Path) -> Path:
    target = Path(value)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidDiagnosis(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _validate_file(root: Path, relative: str | Path, expected: str, name: str) -> Path:
    path = _path(root, relative)
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")
    return path


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


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


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


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_python_process_only"}
    cuda_build = torch.version.cuda
    if not isinstance(cuda_build, str) or len(cuda_build.split(".")) != 2:
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
    }


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


def _bootstrap(
    deltas: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    samples_per_class: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    class_draws: list[torch.Tensor] = []
    values: dict[str, list[torch.Tensor]] = {name: [] for name in deltas}
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == samples_per_class, f"bootstrap class count changed for {digit}")
        draw = torch.randint(samples_per_class, (replicates, samples_per_class), generator=generator)
        class_draws.append(draw)
        for name, delta in deltas.items():
            values[name].append(delta.index_select(0, matches)[draw].mean(dim=1))
    macro = {name: torch.stack(items, dim=0).mean(dim=0).to(torch.float64) for name, items in values.items()}
    summaries: dict[str, Any] = {}
    for name, sample in macro.items():
        quantiles = torch.quantile(sample, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
        summaries[name] = {
            "mean": float(sample.mean()),
            "median": float(sample.median()),
            "ci_lower": float(quantiles[0]),
            "ci_upper": float(quantiles[1]),
            "positive_fraction": float((sample > 0.0).to(torch.float64).mean()),
            "tensor_sha256": tensor_sha256(sample),
        }
    return summaries, {"class_draws": torch.stack(class_draws, dim=0), "macro_deltas": macro}


def _target_labels(labels: torch.Tensor) -> torch.Tensor:
    _require(labels.dtype == torch.long and labels.ndim == 1, "labels are not a class vector")
    target = ((labels + 1) % CLASS_COUNT).contiguous()
    _require(torch.equal(target, (labels + 1) % CLASS_COUNT), "cyclic target mapping changed")
    return target


def _historical_authority(root: Path) -> dict[str, Any]:
    input_path = _validate_file(root, HISTORICAL_INPUT_PATH, HISTORICAL_INPUT_SHA256, "historical class-direction input")
    report_path = _validate_file(root, HISTORICAL_REPORT_PATH, HISTORICAL_REPORT_SHA256, "historical class-direction replication report")
    lineage_path = _validate_file(root, HISTORICAL_LINEAGE_PATH, HISTORICAL_LINEAGE_SHA256, "historical class-direction replication lineage")
    input_payload = _read_json(input_path, "historical class-direction input")
    report = _read_json(report_path, "historical class-direction replication report")
    lineage = _read_json(lineage_path, "historical class-direction replication lineage")
    _require(
        report.get("result") == "VALID_COMPLETED"
        and report.get("status") == "valid_completed"
        and report.get("classification") == "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED",
        "historical class-direction semantic authority is not valid completed",
    )
    _require(
        lineage.get("result") == report.get("result")
        and lineage.get("classification") == report.get("classification"),
        "historical class-direction lineage changed",
    )
    _require(input_payload.get("authorities", {}).get("semantic_authority_id") == "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED", "historical semantic authority id changed")
    protocol = _mapping(input_payload.get("protocol"), "historical class-direction protocol")
    _require(protocol.get("target_rule") == "t=(y+1) mod 10" and protocol.get("transport") == "T=I_100 coefficient space", "historical intervention protocol changed")
    _require(protocol.get("smooth_direction") == "M_smooth=logit_target-logit_source; Delta=M_shift-M_base", "historical Smooth direction changed")
    _require(protocol.get("template_direction") == "M_template=d(output,tau_source)-d(output,tau_target); Delta=M_shift-M_base", "historical template direction changed")
    raw_path = _validate_file(root, HISTORICAL_RAW_PATH, HISTORICAL_RAW_SHA256, "historical class-direction raw artifact")
    raw_payload = _mapping(safe_torch_load(raw_path, expected_sha256=HISTORICAL_RAW_SHA256, map_location="cpu"), "historical class-direction raw artifact")
    _require(raw_payload.get("schema") == "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1", "historical raw artifact schema changed")
    templates = raw_payload.get("templates")
    _require(torch.is_tensor(templates) and tuple(templates.shape) == (CLASS_COUNT, 1, 28, 28) and templates.dtype == torch.float32, "historical TRAIN templates are unavailable")
    _finite(templates, "historical TRAIN templates")
    development = _mapping(_mapping(input_payload.get("data_roles"), "historical data roles").get("development"), "historical DEV role")
    _require(
        development.get("selection_rows") == SELECTION_ROWS
        and development.get("selection_per_class") == SELECTION_PER_CLASS
        and development.get("selection_indices_sha256") == SELECTION_INDICES_SHA256,
        "historical selection identity changed",
    )
    return {
        "semantic_authority_id": "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED",
        "producer_task": lineage.get("task_id"),
        "producer_attempt": "replication_v1_attempt_01",
        "input": {"path": _relative(root, input_path), "sha256": HISTORICAL_INPUT_SHA256},
        "report": {"path": _relative(root, report_path), "sha256": HISTORICAL_REPORT_SHA256},
        "lineage": {"path": _relative(root, lineage_path), "sha256": HISTORICAL_LINEAGE_SHA256},
        "raw_templates": {"path": _relative(root, raw_path), "sha256": HISTORICAL_RAW_SHA256, "tensor_sha256": tensor_sha256(templates)},
        "selection": {
            "rows": SELECTION_ROWS,
            "per_class": SELECTION_PER_CLASS,
            "seed": SELECTION_SEED,
            "indices_sha256": SELECTION_INDICES_SHA256,
            "labels_sha256": SELECTION_LABELS_SHA256,
        },
        "templates": templates.detach().cpu().contiguous(),
    }


def _audit_attempt03(root: Path, target: Path | None = None) -> dict[str, Any]:
    report_path = _validate_file(root, V5_ATTEMPT_03_REPORT_PATH, V5_ATTEMPT_03_REPORT_SHA256, "V5 attempt-03 report")
    lineage_path = _validate_file(root, V5_ATTEMPT_03_LINEAGE_PATH, V5_ATTEMPT_03_LINEAGE_SHA256, "V5 attempt-03 lineage")
    sufficient_path = _validate_file(root, V5_ATTEMPT_03_SUFFICIENT_PATH, V5_ATTEMPT_03_SUFFICIENT_SHA256, "V5 attempt-03 sufficient observations")
    report = _read_json(report_path, "V5 attempt-03 report")
    lineage = _read_json(lineage_path, "V5 attempt-03 lineage")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed", "V5 attempt-03 is not valid completed")
    _require(report.get("classification") == "V5_LATENT_TO_OPTICAL_TRANSFER_MIXED_OR_UNRESOLVED", "V5 attempt-03 classification changed")
    _require(lineage.get("result") == report.get("result") and lineage.get("classification") == report.get("classification"), "V5 attempt-03 lineage changed")
    sufficient = _mapping(safe_torch_load(sufficient_path, expected_sha256=V5_ATTEMPT_03_SUFFICIENT_SHA256, map_location="cpu"), "V5 attempt-03 sufficient observations")
    _require(
        sufficient.get("schema") == "lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_sufficient_observations.v1"
        and sufficient.get("task_id") == "DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1",
        "V5 attempt-03 sufficient schema/task changed",
    )
    protocol = _mapping(sufficient.get("protocol"), "V5 attempt-03 sufficient protocol")
    _require(protocol.get("selection_indices_sha256") == SELECTION_INDICES_SHA256 and protocol.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES and protocol.get("bootstrap_seed") == BOOTSTRAP_SEED, "V5 attempt-03 sufficient protocol changed")
    observations = _mapping(sufficient.get("observations"), "V5 attempt-03 observations")
    expected = {"selection_indices", "labels", "norm_v", "norm_class", "ratio", "rho", "alignment", "delta_s_actual", "delta_s_match", "delta_s_full"}
    _require(set(observations) == expected, "V5 attempt-03 sufficient tensor set changed")
    tensors: dict[str, torch.Tensor] = {}
    for name in expected:
        value = observations[name]
        _require(torch.is_tensor(value) and value.ndim == 1 and value.shape[0] == SELECTION_ROWS, f"V5 attempt-03 observation shape changed: {name}")
        _finite(value.to(torch.float64), f"V5 attempt-03 observation {name}")
        tensors[name] = value.detach().cpu().contiguous()
    _require(tensors["selection_indices"].dtype == torch.long and tensor_sha256(tensors["selection_indices"]) == SELECTION_INDICES_SHA256, "V5 attempt-03 selection identity changed")
    _require(tensors["labels"].dtype == torch.long and tensor_sha256(tensors["labels"]) == SELECTION_LABELS_SHA256, "V5 attempt-03 label identity changed")
    _require(torch.equal(torch.bincount(tensors["labels"], minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), SELECTION_PER_CLASS, dtype=torch.long)), "V5 attempt-03 labels are not balanced")
    report_metric_names = {"delta_s_actual": "actual", "delta_s_match": "match", "delta_s_full": "full"}
    comparison: dict[str, Any] = {}
    for tensor_name, report_name in report_metric_names.items():
        values = tensors[tensor_name]
        metric = _mapping(_mapping(report.get("smooth_continuous_margin"), "V5 attempt-03 Smooth report").get(report_name), f"V5 attempt-03 Smooth {report_name}")
        statistics = _mapping(metric.get("statistics"), f"V5 attempt-03 Smooth {report_name} statistics")
        raw_summary = _summary(values)
        for field in ("n", "mean", "median", "q05", "q95", "std", "min", "max"):
            _require(math.isclose(float(statistics[field]), float(raw_summary[field]), rel_tol=0.0, abs_tol=1.0e-12), f"V5 attempt-03 raw statistic mismatch: {tensor_name}.{field}")
        per_class = metric.get("per_class")
        _require(isinstance(per_class, list) and len(per_class) == CLASS_COUNT, f"V5 attempt-03 per-class report changed: {report_name}")
        raw_class_fraction = []
        for digit in range(CLASS_COUNT):
            class_values = values[tensors["labels"] == digit]
            fraction = float((class_values > 0.0).to(torch.float64).mean())
            raw_class_fraction.append(fraction)
            serialized = float(per_class[digit]["positive_fraction"])
            _require(math.isclose(serialized, fraction, rel_tol=0.0, abs_tol=1.0e-12), f"V5 attempt-03 per-class positive fraction mismatch: {report_name}.{digit}")
        bootstrap_summary, _ = _bootstrap(
            {report_name: values},
            tensors["labels"],
            SELECTION_PER_CLASS,
        )
        serialized_bootstrap = _mapping(metric.get("bootstrap"), f"V5 attempt-03 Smooth {report_name} bootstrap")
        recomputed_bootstrap = bootstrap_summary[report_name]
        for field in ("mean", "median", "ci_lower", "ci_upper", "positive_fraction", "tensor_sha256"):
            _require(serialized_bootstrap.get(field) == recomputed_bootstrap.get(field), f"V5 attempt-03 bootstrap mismatch: {report_name}.{field}")
        comparison[report_name] = {
            "raw_observation_positive_fraction": float((values > 0.0).to(torch.float64).mean()),
            "raw_per_class_positive_fraction": raw_class_fraction,
            "serialized_global_positive_fraction": float(serialized_bootstrap["positive_fraction"]),
            "recomputed_global_positive_fraction": float(recomputed_bootstrap["positive_fraction"]),
            "global_positive_fraction_semantics": "fraction of paired class-stratified bootstrap macro-delta replicates greater than zero",
            "per_class_positive_fraction_semantics": "fraction of observed per-class deltas greater than zero",
            "raw_summary": raw_summary,
            "raw_tensor_sha256": tensor_sha256(values),
        }
    audit = {
        "schema": AUDIT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "source_task_id": "DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1",
        "source_attempt": "attempt_03",
        "result": "PASS",
        "status": "valid_completed",
        "scientific_forward_count": 0,
        "diagnosis_executed": False,
        "classification_unchanged": True,
        "source_report": {"path": _relative(root, report_path), "sha256": V5_ATTEMPT_03_REPORT_SHA256},
        "source_lineage": {"path": _relative(root, lineage_path), "sha256": V5_ATTEMPT_03_LINEAGE_SHA256},
        "source_sufficient_observations": {"path": _relative(root, sufficient_path), "sha256": V5_ATTEMPT_03_SUFFICIENT_SHA256},
        "positive_fraction_comparison": comparison,
        "conclusion": "sufficient observations and report aggregates are internally consistent; no report rewrite or scientific retry is required",
    }
    if target is not None:
        audit_path = target / AUDIT_NAME
        atomic_write_new_json(audit_path, audit, artifact_kind="V5 attempt-03 sufficient-observation integrity audit")
        reloaded = _read_json(audit_path, "V5 attempt-03 integrity audit")
        _require(reloaded.get("result") == "PASS" and reloaded.get("scientific_forward_count") == 0, "V5 attempt-03 integrity audit reload changed")
        audit["artifact"] = {"path": _relative(root, audit_path), "sha256": file_sha256(audit_path), "schema": AUDIT_SCHEMA}
    return {"audit": audit, "observations": tensors, "report": report, "lineage": lineage}


def _load_checkpoint(root: Path, path_value: Path, expected_sha: str, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _validate_file(root, path_value, expected_sha, f"V5 {arm} step-300 checkpoint")
    payload = _mapping(safe_torch_load(path, expected_sha256=expected_sha, map_location="cpu"), f"V5 {arm} step-300 checkpoint")
    _require(payload.get("schema") == "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_checkpoint.v1" and payload.get("checkpoint_step") == 300 and payload.get("arm") == arm, f"V5 {arm} checkpoint schema/step changed")
    state = dict(_mapping(payload.get("runtime_state"), f"V5 {arm} runtime state"))
    expected_keys = {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}
    _require(set(state) == expected_keys, f"V5 {arm} runtime state keys changed")
    prototypes = payload.get("prototypes")
    _require(torch.is_tensor(prototypes) and tuple(prototypes.shape) == (CLASS_COUNT, OLS_PORT_COUNT) and prototypes.dtype == torch.complex64, f"V5 {arm} prototypes shape/dtype changed")
    _require(payload.get("prototypes_sha256") == EXPECTED_PROTOTYPES_SHA256 and tensor_sha256(prototypes) == EXPECTED_PROTOTYPES_SHA256, f"V5 {arm} prototypes identity changed")
    _require(math.isclose(float(payload.get("s_mu")), EXPECTED_S_MU, rel_tol=0.0, abs_tol=1.0e-15), f"V5 {arm} prototype scale changed")
    expected_lambda = 0.0 if arm == "control" else EXPECTED_CANDIDATE_LAMBDA
    _require(math.isclose(float(payload.get("lambda_proto")), expected_lambda, rel_tol=0.0, abs_tol=1.0e-15), f"V5 {arm} lambda changed")
    for key in ("stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "sampled_generation_executed", "formal_promotion"):
        _require(payload.get(key) is False, f"V5 {arm} checkpoint crossed forbidden boundary: {key}")
    return dict(payload), state


def _load_v4_generator(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _validate_file(root, V4_GENERATOR_PATH, V4_GENERATOR_SHA256, "frozen V4 generator checkpoint")
    payload = _mapping(safe_torch_load(path, expected_sha256=V4_GENERATOR_SHA256, map_location="cpu"), "frozen V4 generator checkpoint")
    _require(payload.get("checkpoint_step") == 300, "frozen V4 generator step changed")
    state = dict(_mapping(payload.get("runtime_state"), "frozen V4 generator runtime state"))
    _require(torch.is_tensor(state.get("generator.phase")) and torch.is_tensor(state.get("ols.operator.transport")), "frozen V4 generator state is incomplete")
    _require(torch.equal(state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "frozen V4 transport is not T=I_100")
    for key in ("stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "sampled_generation_executed", "formal_promotion"):
        if key in payload:
            _require(payload.get(key) is False, f"frozen V4 checkpoint crossed forbidden boundary: {key}")
    return dict(payload), state


def _load_smooth(root: Path, device: torch.device) -> tuple[SmoothMNISTEvaluator, dict[str, Any]]:
    path = _validate_file(root, SMOOTH_PATH, SMOOTH_SHA256, "SmoothMNISTEvaluator v2.1 checkpoint")
    payload = _mapping(safe_torch_load(path, expected_sha256=SMOOTH_SHA256, map_location="cpu"), "SmoothMNISTEvaluator v2.1 checkpoint")
    _require(payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    evaluator = SmoothMNISTEvaluator()
    evaluator.load_state_dict(_mapping(payload.get("state_dict"), "Smooth evaluator state"), strict=True)
    evaluator.to(device).freeze()
    evaluator.eval()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator gradient firewall failed")
    state_tree_sha = _digest_tree(_cpu_tree(evaluator.state_dict()))
    _require(state_tree_sha == SMOOTH_STATE_TREE_SHA256, "Smooth evaluator state-dict tree identity changed")
    return evaluator, {
        "checkpoint": {
            "path": _relative(root, path),
            "sha256": SMOOTH_SHA256,
            "identity_domain": "checkpoint file bytes",
            "role": "SmoothMNISTEvaluator v2.1 checkpoint",
        },
        "architecture_kind": payload.get("architecture_kind"),
        "state_dict_tree_sha256": state_tree_sha,
        "identity_domains": {
            "file_sha256": "checkpoint file bytes",
            "state_dict_tree_sha256": "loaded state_dict tree",
        },
        "cross_domain_equality_required": False,
        "gradient_enabled": False,
    }


def _new_frozen_consumer(contract: Any, state: Mapping[str, Any], device: torch.device) -> Any:
    consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_GRID)
    consumer.runtime.load_state_dict(dict(state), strict=True)
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    consumer.runtime.eval()
    _require(not any(parameter.requires_grad for parameter in consumer.runtime.parameters()), "diagnostic runtime has trainable parameters")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "diagnostic transport is not T=I_100")
    return consumer


def _validate_runtime_result(result: Any, batch_size: int, *, external: bool) -> torch.Tensor:
    if external:
        _require(result.image_input is None and result.ols_input_field is None and result.z_ols.source == "external_generated", "external path crossed the encoder boundary")
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
def _encode(consumer: Any, images: torch.Tensor) -> torch.Tensor:
    before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    parts: list[torch.Tensor] = []
    for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
        batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
        result = consumer.runtime.forward_from_image(batch)
        raw = _validate_runtime_result(result, int(batch.shape[0]), external=False)
        z = result.z_ols.coefficients.detach().cpu().contiguous()
        _require(tuple(z.shape) == (int(batch.shape[0]), OLS_PORT_COUNT) and z.dtype == torch.complex64, "encoded z shape/dtype changed")
        parts.append(z)
        del batch, result, raw, z
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    after = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    _require(before == after, "frozen encoder forward changed runtime state")
    value = torch.cat(parts, dim=0)
    _finite_complex(value, "encoded z")
    return value


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
    value = torch.cat(parts, dim=0)
    _require(tuple(value.shape) == (z_cpu.shape[0], 1, 200, 200) and value.dtype == torch.float32, "external raw CMOS shape/dtype changed")
    _finite(value, "external raw CMOS")
    return value


@torch.inference_mode()
def _smooth_logits(evaluator: Any, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, raw.shape[0], LOGICAL_BATCH_SIZE):
        batch = raw[start : start + LOGICAL_BATCH_SIZE].to(device)
        view = F.adaptive_avg_pool2d(batch.clamp(0.0, 1.0), (28, 28))
        _, logits = evaluator(view)
        _require(tuple(logits.shape) == (int(batch.shape[0]), CLASS_COUNT) and not torch.is_complex(logits), "Smooth logits shape/dtype changed")
        _finite(logits, "Smooth logits")
        parts.append(logits.detach().cpu().contiguous())
        del batch, view, logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return torch.cat(parts, dim=0)


def _template_margins(base_raw: torch.Tensor, shifted_raw: torch.Tensor, templates: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_view = F.adaptive_avg_pool2d(base_raw.clamp(0.0, 1.0), (28, 28))
    shifted_view = F.adaptive_avg_pool2d(shifted_raw.clamp(0.0, 1.0), (28, 28))
    source_templates = templates.index_select(0, labels)
    target_templates = templates.index_select(0, targets)
    base_source = (base_view.to(torch.float64) - source_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    base_target = (base_view.to(torch.float64) - target_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    shifted_source = (shifted_view.to(torch.float64) - source_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    shifted_target = (shifted_view.to(torch.float64) - target_templates.to(torch.float64)).square().flatten(1).mean(dim=1)
    base_margin = base_source - base_target
    shifted_margin = shifted_source - shifted_target
    _finite(base_margin, "template base margin")
    _finite(shifted_margin, "template shifted margin")
    return base_margin, shifted_margin


def _smooth_margins(base_logits: torch.Tensor, shifted_logits: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(labels.shape[0], dtype=torch.long)
    base_margin = base_logits[rows, targets] - base_logits[rows, labels]
    shifted_margin = shifted_logits[rows, targets] - shifted_logits[rows, labels]
    _finite(base_margin, "Smooth base target-source margin")
    _finite(shifted_margin, "Smooth shifted target-source margin")
    return base_margin.to(torch.float64), shifted_margin.to(torch.float64)


def _raw_summary(raw: torch.Tensor) -> dict[str, Any]:
    value = raw.detach().cpu().contiguous()
    _require(tuple(value.shape[1:]) == (1, 200, 200) and value.dtype == torch.float32, "raw summary shape/dtype changed")
    _finite(value, "raw summary")
    per_image_variance = value.flatten(1).var(dim=1, unbiased=False)
    across_sample_variance = value.flatten(1).var(dim=0, unbiased=False)
    _require(bool((per_image_variance > 0.0).all()), "raw CMOS contains a constant sample")
    _require(float(across_sample_variance.mean()) > 0.0 and float(value.std(unbiased=False)) > 0.0, "raw CMOS batch is nonconstant")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "structured": True,
        "nonconstant": True,
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "per_image_variance_min": float(per_image_variance.min()),
        "per_image_variance_max": float(per_image_variance.max()),
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
    return {
        "absolute_rms": _summary(absolute),
        "relative_rms": _summary(relative),
        "absolute_values_sha256": tensor_sha256(absolute),
        "relative_values_sha256": tensor_sha256(relative),
    }


def _ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, CLASS_COUNT + 1, dtype=torch.long).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks[torch.arange(logits.shape[0]), labels]


def _point_analysis(
    observations: Mapping[str, torch.Tensor],
    *,
    prefix: str,
    labels: torch.Tensor,
    targets: torch.Tensor,
    raw_base: torch.Tensor,
    raw_shifted: torch.Tensor,
    samples_per_class: int,
) -> dict[str, Any]:
    template_base = observations[f"{prefix}_template_base_margin"]
    template_shifted = observations[f"{prefix}_template_shifted_margin"]
    smooth_base = observations[f"{prefix}_smooth_base_logits"]
    smooth_shifted = observations[f"{prefix}_smooth_shifted_logits"]
    delta_template = template_shifted - template_base
    rows = torch.arange(labels.shape[0], dtype=torch.long)
    smooth_base_margin = smooth_base[rows, targets] - smooth_base[rows, labels]
    smooth_shifted_margin = smooth_shifted[rows, targets] - smooth_shifted[rows, labels]
    delta_smooth = smooth_shifted_margin - smooth_base_margin
    _finite(delta_template, f"{prefix} template delta")
    _finite(delta_smooth, f"{prefix} Smooth delta")
    bootstrap_summaries, bootstrap_tensors = _bootstrap(
        {"template": delta_template, "smooth": delta_smooth},
        labels,
        samples_per_class,
    )
    template_pass = bootstrap_summaries["template"]["mean"] > 0.0 and bootstrap_summaries["template"]["ci_lower"] > 0.0
    smooth_pass = bootstrap_summaries["smooth"]["mean"] > 0.0 and bootstrap_summaries["smooth"]["ci_lower"] > 0.0
    base_source_rank = _ranks(smooth_base, labels)
    shifted_source_rank = _ranks(smooth_shifted, labels)
    base_target_rank = _ranks(smooth_base, targets)
    shifted_target_rank = _ranks(smooth_shifted, targets)
    base_predictions = smooth_base.argmax(dim=1)
    shifted_predictions = smooth_shifted.argmax(dim=1)
    positive_template_pairs = 0
    positive_smooth_pairs = 0
    pair_summaries: list[dict[str, Any]] = []
    for source in range(CLASS_COUNT):
        target = (source + 1) % CLASS_COUNT
        mask = labels == source
        template_values = delta_template[mask]
        smooth_values = delta_smooth[mask]
        _require(template_values.numel() == samples_per_class, f"{prefix} pair {source}->{target} count changed")
        template_mean = float(template_values.mean())
        smooth_mean = float(smooth_values.mean())
        positive_template = float((template_values > 0.0).to(torch.float64).mean())
        positive_smooth = float((smooth_values > 0.0).to(torch.float64).mean())
        positive_template_pairs += int(template_mean > 0.0)
        positive_smooth_pairs += int(smooth_mean > 0.0)
        base_source_rate = float((base_predictions[mask] == source).to(torch.float64).mean())
        shifted_source_rate = float((shifted_predictions[mask] == source).to(torch.float64).mean())
        base_target_rate = float((base_predictions[mask] == target).to(torch.float64).mean())
        shifted_target_rate = float((shifted_predictions[mask] == target).to(torch.float64).mean())
        pair_summaries.append(
            {
                "source_class": source,
                "target_class": target,
                "sample_count": int(mask.sum()),
                "delta_template_mean": template_mean,
                "delta_template_positive_fraction": positive_template,
                "delta_smooth_mean": smooth_mean,
                "delta_smooth_positive_fraction": positive_smooth,
                "template_pair_direction": "positive" if template_mean > 0.0 else "nonpositive",
                "smooth_pair_direction": "positive" if smooth_mean > 0.0 else "nonpositive",
                "source_rank": {
                    "base": _summary(base_source_rank[mask].to(torch.float64)),
                    "shifted": _summary(shifted_source_rank[mask].to(torch.float64)),
                    "shifted_minus_base": _summary((shifted_source_rank[mask] - base_source_rank[mask]).to(torch.float64)),
                },
                "target_rank": {
                    "base": _summary(base_target_rank[mask].to(torch.float64)),
                    "shifted": _summary(shifted_target_rank[mask].to(torch.float64)),
                    "shifted_minus_base": _summary((shifted_target_rank[mask] - base_target_rank[mask]).to(torch.float64)),
                },
                "prediction_rates": {
                    "source": {
                        "base": base_source_rate,
                        "shifted": shifted_source_rate,
                        "shifted_minus_base": shifted_source_rate - base_source_rate,
                    },
                    "target": {
                        "base": base_target_rate,
                        "shifted": shifted_target_rate,
                        "shifted_minus_base": shifted_target_rate - base_target_rate,
                    },
                },
            }
        )
    broad = positive_template_pairs >= 2 and positive_smooth_pairs >= 2
    return {
        "operating_point": prefix,
        "intervention": "z_shift=z-mu_y+mu_t, t=(y+1) mod 10",
        "raw_validity": {
            "base": _raw_summary(raw_base),
            "shifted": _raw_summary(raw_shifted),
            "rms_shift": _raw_change(raw_base, raw_shifted),
        },
        "template": {
            "definition": "M_template(r)=d(r,tau_y)-d(r,tau_t); Delta_template=M_template(raw_shift)-M_template(raw_base)",
            "distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28",
            "base_summary": _summary(template_base),
            "shifted_summary": _summary(template_shifted),
            "delta_summary": _summary(delta_template),
            "bootstrap": bootstrap_summaries["template"],
            "transfer": bool(template_pass),
        },
        "smooth": {
            "definition": "M_smooth(r)=score_t(r)-score_y(r) using SmoothMNISTEvaluator v2.1 canonical pre-softmax logits; Delta_smooth=M_shift-M_base",
            "base_summary": _summary(smooth_base_margin),
            "shifted_summary": _summary(smooth_shifted_margin),
            "delta_summary": _summary(delta_smooth),
            "bootstrap": bootstrap_summaries["smooth"],
            "transfer": bool(smooth_pass),
            "target_rank": {
                "base": _summary(base_target_rank.to(torch.float64)),
                "shifted": _summary(shifted_target_rank.to(torch.float64)),
                "shifted_minus_base": _summary((shifted_target_rank - base_target_rank).to(torch.float64)),
            },
            "source_rank": {
                "base": _summary(base_source_rank.to(torch.float64)),
                "shifted": _summary(shifted_source_rank.to(torch.float64)),
                "shifted_minus_base": _summary((shifted_source_rank - base_source_rank).to(torch.float64)),
            },
            "prediction_rates": {
                "source": {
                    "base": float((base_predictions == labels).to(torch.float64).mean()),
                    "shifted": float((shifted_predictions == labels).to(torch.float64).mean()),
                    "shifted_minus_base": float(((shifted_predictions == labels).to(torch.float64).mean()) - ((base_predictions == labels).to(torch.float64).mean())),
                },
                "target": {
                    "base": float((base_predictions == targets).to(torch.float64).mean()),
                    "shifted": float((shifted_predictions == targets).to(torch.float64).mean()),
                    "shifted_minus_base": float(((shifted_predictions == targets).to(torch.float64).mean()) - ((base_predictions == targets).to(torch.float64).mean())),
                },
                "base_histogram": [int(value) for value in torch.bincount(base_predictions, minlength=CLASS_COUNT).tolist()],
                "shifted_histogram": [int(value) for value in torch.bincount(shifted_predictions, minlength=CLASS_COUNT).tolist()],
                "base_prediction_tensor_sha256": tensor_sha256(base_predictions),
                "shifted_prediction_tensor_sha256": tensor_sha256(shifted_predictions),
            },
        },
        "pair_summaries": pair_summaries,
        "directional_consistency": {
            "positive_template_pair_count": positive_template_pairs,
            "positive_smooth_pair_count": positive_smooth_pairs,
            "template_broad": positive_template_pairs >= 2,
            "smooth_broad": positive_smooth_pairs >= 2,
            "broad": broad,
            "pair_dominated": bool(
                (template_pass and positive_template_pairs == 1)
                or (smooth_pass and positive_smooth_pairs == 1)
            ),
            "assessment": "broad" if broad else ("pair-dominated" if positive_template_pairs == 1 or positive_smooth_pairs == 1 else "mixed_or_no_positive_direction"),
        },
        "bootstrap_tensors": bootstrap_tensors,
        "delta_tensors": {"template": delta_template, "smooth": delta_smooth},
    }


def _save_sufficient(root: Path, target: Path, observations: Mapping[str, torch.Tensor], *, task_label: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = target / SUFFICIENT_NAME
    tensor_payload = {name: value.detach().cpu().contiguous() for name, value in observations.items()}
    payload = {
        "schema": SUFFICIENT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "protocol": {
            "role": "frozen operating-point sufficient observations; no optical rerun",
            "target_rule": "t=(y+1) mod 10",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "observation_tensors": sorted(tensor_payload),
        },
        "metadata": {"task_label": task_label, "scientific_forward_count_at_persist": 0},
        "observations": tensor_payload,
    }
    atomic_torch_save_new(path, payload, artifact_kind="V5 operating-point sufficient observations")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "operating-point sufficient observations")
    _require(loaded.get("schema") == SUFFICIENT_SCHEMA and loaded.get("protocol", {}).get("target_rule") == "t=(y+1) mod 10", "sufficient reload protocol changed")
    loaded_observations = _mapping(loaded.get("observations"), "reloaded operating-point observations")
    _require(set(loaded_observations) == set(tensor_payload), "sufficient observation tensor set changed on reload")
    reloaded: dict[str, torch.Tensor] = {}
    for name, value in tensor_payload.items():
        candidate = loaded_observations.get(name)
        _require(torch.is_tensor(candidate) and tensor_sha256(candidate) == tensor_sha256(value), f"sufficient observation changed on reload: {name}")
        reloaded[name] = candidate.detach().cpu().contiguous()
    return {
        "path": _relative(root, path),
        "sha256": digest,
        "schema": SUFFICIENT_SCHEMA,
        "reloaded": True,
        "observation_tensors": sorted(tensor_payload),
        "task_label": task_label,
    }, reloaded


def _save_raw(root: Path, target: Path, payload: Mapping[str, Any], *, selection_indices: torch.Tensor, labels: torch.Tensor, templates: torch.Tensor, prototypes: torch.Tensor) -> dict[str, Any]:
    path = target / RAW_NAME
    save_payload = {
        "schema": RAW_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "selection_indices": selection_indices.cpu(),
        "labels": labels.cpu(),
        "targets": _target_labels(labels).cpu(),
        "selection_indices_sha256": tensor_sha256(selection_indices),
        "templates": templates.cpu(),
        "prototypes": prototypes.cpu(),
        "z": {name: value.cpu() for name, value in payload["z"].items()},
        "raw": {name: value.cpu() for name, value in payload["raw"].items()},
        "smooth_logits": {name: value.cpu() for name, value in payload["smooth_logits"].items()},
    }
    atomic_torch_save_new(path, save_payload, artifact_kind="V5 operating-point raw outputs")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "operating-point raw outputs")
    _require(loaded.get("schema") == RAW_SCHEMA and loaded.get("selection_indices_sha256") == tensor_sha256(selection_indices), "raw reload identity changed")
    for name, value in save_payload["raw"].items():
        _require(tensor_sha256(loaded["raw"][name]) == tensor_sha256(value), f"raw output changed on reload: {name}")
    return {"path": _relative(root, path), "sha256": digest, "schema": RAW_SCHEMA, "reloaded": True}


def _save_bootstrap(root: Path, target: Path, analyses: Mapping[str, Mapping[str, Any]], labels: torch.Tensor) -> dict[str, Any]:
    path = target / BOOTSTRAP_NAME
    payload = {
        "schema": BOOTSTRAP_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "selection_labels_sha256": tensor_sha256(labels),
        "protocol": {"type": "paired, class-stratified over 100 examples per source class", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "quantile": "torch.quantile interpolation=linear"},
        "summaries": {name: {"template": dict(value["template"]["bootstrap"]), "smooth": dict(value["smooth"]["bootstrap"])} for name, value in analyses.items()},
        "class_draws": {name: value["bootstrap_tensors"]["class_draws"] for name, value in analyses.items()},
        "macro_deltas": {
            name: {
                metric: value["bootstrap_tensors"]["macro_deltas"][metric]
                for metric in ("template", "smooth")
            }
            for name, value in analyses.items()
        },
    }
    atomic_torch_save_new(path, payload, artifact_kind="V5 operating-point paired bootstrap")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "operating-point bootstrap")
    _require(loaded.get("schema") == BOOTSTRAP_SCHEMA and loaded.get("protocol", {}).get("seed") == BOOTSTRAP_SEED, "bootstrap reload protocol changed")
    return {"path": _relative(root, path), "sha256": digest, "schema": BOOTSTRAP_SCHEMA, "reloaded": True, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}


def _classify(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    control_transfer = bool(control["template"]["transfer"] and control["smooth"]["transfer"] and control["directional_consistency"]["broad"])
    candidate_transfer = bool(candidate["template"]["transfer"] and candidate["smooth"]["transfer"] and candidate["directional_consistency"]["broad"])
    if not control_transfer:
        classification = CONTROL_TRANSFER_LOSS
    elif not candidate_transfer:
        classification = PROTOTYPE_TRAINING_DEGRADATION
    elif control_transfer and candidate_transfer:
        classification = KNOWN_DIRECTIONS_RETAINED
    else:
        classification = MIXED_OR_UNRESOLVED
    _require(classification in CLASSIFICATIONS, "operating-point classification is not frozen")
    return classification, {
        "control_known_cyclic_transfer": control_transfer,
        "candidate_known_cyclic_transfer": candidate_transfer,
        "rule": [
            "control false -> V5_CONTROL_OPERATING_POINT_TRANSFER_LOSS_SUPPORTED",
            "control true and candidate false -> V5_PROTOTYPE_TRAINING_TRANSFER_DEGRADATION_SUPPORTED",
            "control true and candidate true -> V5_KNOWN_CLASS_DIRECTIONS_RETAINED",
            "otherwise -> V5_CLASS_DIRECTION_OPERATING_POINT_MIXED_OR_UNRESOLVED",
        ],
    }


def _run_core(
    *,
    z_control: torch.Tensor,
    z_candidate: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
    labels: torch.Tensor,
    optical_forward: Callable[[torch.Tensor], torch.Tensor],
    smooth_forward: Callable[[torch.Tensor], torch.Tensor],
    artifact_root: Path,
    artifact_target: Path,
    artifact_task_label: str,
    selection_indices: torch.Tensor,
) -> dict[str, Any]:
    expected_rows = int(labels.shape[0])
    _require(expected_rows > 0 and expected_rows % CLASS_COUNT == 0, "operating-point row count is invalid")
    samples_per_class = expected_rows // CLASS_COUNT
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), samples_per_class, dtype=torch.long)), "operating-point labels are not balanced")
    _finite_complex(z_control, "control z")
    _finite_complex(z_candidate, "candidate z")
    _require(z_control.shape == z_candidate.shape == (expected_rows, OLS_PORT_COUNT), "operating-point z shape changed")
    _finite_complex(prototypes, "frozen prototypes")
    _require(tuple(templates.shape) == (CLASS_COUNT, 1, 28, 28) and templates.dtype == torch.float32, "TRAIN template shape/dtype changed")
    targets = _target_labels(labels)
    z_shift_control = z_control - prototypes.index_select(0, labels) + prototypes.index_select(0, targets)
    z_shift_candidate = z_candidate - prototypes.index_select(0, labels) + prototypes.index_select(0, targets)
    for value, name in ((z_shift_control, "control shifted z"), (z_shift_candidate, "candidate shifted z")):
        _finite_complex(value, name)
    raw = {
        "control_base": optical_forward(z_control),
        "control_shifted": optical_forward(z_shift_control),
        "candidate_base": optical_forward(z_candidate),
        "candidate_shifted": optical_forward(z_shift_candidate),
    }
    for name, value in raw.items():
        _require(tuple(value.shape) == (expected_rows, 1, 200, 200) and value.dtype == torch.float32, f"{name} raw shape/dtype changed")
        _finite(value, f"{name} raw")
    smooth_logits = {name: smooth_forward(value) for name, value in raw.items()}
    for name, value in smooth_logits.items():
        _require(tuple(value.shape) == (expected_rows, CLASS_COUNT), f"{name} Smooth logits shape changed")
        _finite(value, f"{name} Smooth logits")
    observations: dict[str, torch.Tensor] = {
        "selection_indices": selection_indices.cpu().contiguous(),
        "labels": labels.cpu().contiguous(),
        "targets": targets.cpu().contiguous(),
    }
    for prefix, base_name, shifted_name in (
        ("control", "control_base", "control_shifted"),
        ("candidate", "candidate_base", "candidate_shifted"),
    ):
        template_base, template_shifted = _template_margins(raw[base_name], raw[shifted_name], templates, labels, targets)
        smooth_base, smooth_shifted = _smooth_margins(smooth_logits[base_name], smooth_logits[shifted_name], labels, targets)
        observations.update(
            {
                f"{prefix}_template_base_margin": template_base.cpu().contiguous(),
                f"{prefix}_template_shifted_margin": template_shifted.cpu().contiguous(),
                f"{prefix}_smooth_base_logits": smooth_logits[base_name].cpu().contiguous(),
                f"{prefix}_smooth_shifted_logits": smooth_logits[shifted_name].cpu().contiguous(),
            }
        )
    sufficient, reloaded_observations = _save_sufficient(
        artifact_root,
        artifact_target,
        observations,
        task_label=artifact_task_label,
    )
    analyses = {
        "control": _point_analysis(
            reloaded_observations,
            prefix="control",
            labels=reloaded_observations["labels"],
            targets=reloaded_observations["targets"],
            raw_base=raw["control_base"],
            raw_shifted=raw["control_shifted"],
            samples_per_class=samples_per_class,
        ),
        "candidate": _point_analysis(
            reloaded_observations,
            prefix="candidate",
            labels=reloaded_observations["labels"],
            targets=reloaded_observations["targets"],
            raw_base=raw["candidate_base"],
            raw_shifted=raw["candidate_shifted"],
            samples_per_class=samples_per_class,
        ),
    }
    classification, classification_flags = _classify(analyses["control"], analyses["candidate"])
    raw_artifact = _save_raw(
        artifact_root,
        artifact_target,
        {
            "z": {
                "control_base": z_control,
                "control_shifted": z_shift_control,
                "candidate_base": z_candidate,
                "candidate_shifted": z_shift_candidate,
            },
            "raw": raw,
            "smooth_logits": smooth_logits,
        },
        selection_indices=selection_indices,
        labels=labels,
        templates=templates,
        prototypes=prototypes,
    )
    bootstrap_artifact = _save_bootstrap(artifact_root, artifact_target, analyses, labels)
    return {
        "classification": classification,
        "classification_flags": classification_flags,
        "analyses": analyses,
        "raw_artifact": raw_artifact,
        "sufficient_artifact": sufficient,
        "bootstrap_artifact": bootstrap_artifact,
        "z": {
            "control_base": z_control,
            "control_shifted": z_shift_control,
            "candidate_base": z_candidate,
            "candidate_shifted": z_shift_candidate,
        },
        "raw": raw,
        "smooth_logits": smooth_logits,
        "samples_per_class": samples_per_class,
    }


def _load_selection(root: Path, historical: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source_path = _validate_file(root, TRAINING_SOURCE_PATH, TRAINING_SOURCE_SHA256, "canonical TRAIN source")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28) and images.dtype == torch.float32, "training source image shape/dtype changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "training source label shape/dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    _require(train_indices.numel() == 55_000 and development_indices.numel() == DEVELOPMENT_ROWS, "TRAIN/DEV partition changed")
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    development_labels = labels.index_select(0, development_indices)
    selected: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, f"DEV class count changed for {digit}")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:SELECTION_PER_CLASS]))
    selection_indices = torch.cat(selected).contiguous()
    _require(tensor_sha256(selection_indices) == SELECTION_INDICES_SHA256, "balanced DEV selection identity changed")
    selection_labels = labels.index_select(0, selection_indices).contiguous()
    _require(tensor_sha256(selection_labels) == SELECTION_LABELS_SHA256, "balanced DEV label identity changed")
    _require(torch.equal(torch.bincount(selection_labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), SELECTION_PER_CLASS, dtype=torch.long)), "DEV selection is not balanced")
    return images.index_select(0, selection_indices).contiguous(), selection_labels, selection_indices, {
        "source": {"path": _relative(root, source_path), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000},
        "train": {"rows": 55_000, "counts_by_class": [int(value) for value in torch.bincount(labels.index_select(0, train_indices), minlength=CLASS_COUNT).tolist()], "indices_sha256": tensor_sha256(train_indices), "scientific_values_read": 0},
        "development": {"rows": DEVELOPMENT_ROWS, "selection_rows": SELECTION_ROWS, "selection_per_class": SELECTION_PER_CLASS, "selection_seed": SELECTION_SEED, "indices_sha256": tensor_sha256(development_indices), "selection_indices_sha256": SELECTION_INDICES_SHA256, "values_read_for_evaluation": SELECTION_ROWS},
        "calibration": {"rows_read": 0},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "official_test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }


def _resolve_dependencies(root: Path) -> dict[str, Any]:
    git = git_facts(root)
    _require(git.get("branch") == EXPECTED_BRANCH, "active Git branch differs from authorized V5 operating-point entry")
    audit = _audit_attempt03(root)
    historical = _historical_authority(root)
    attempt03_report = audit["report"]
    attempt03_auth = _mapping(attempt03_report.get("authorities"), "V5 attempt-03 authorities")
    _validate_file(root, V5_CONTRACT_PATH, V5_CONTRACT_SHA256, "V5 contract")
    v5_report_path = _validate_file(root, V5_SCIENTIFIC_REPORT_PATH, V5_SCIENTIFIC_REPORT_SHA256, "V5 scientific report")
    v5_lineage_path = _validate_file(root, V5_SCIENTIFIC_LINEAGE_PATH, V5_SCIENTIFIC_LINEAGE_SHA256, "V5 scientific lineage")
    v5_report = _read_json(v5_report_path, "V5 scientific report")
    v5_lineage = _read_json(v5_lineage_path, "V5 scientific lineage")
    _require(v5_report.get("result") == "VALID_COMPLETED" and v5_report.get("status") == "valid_completed" and v5_report.get("classification") == "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED", "V5 scientific authority changed")
    _require(v5_lineage.get("result") == v5_report.get("result") and v5_lineage.get("classification") == v5_report.get("classification"), "V5 scientific lineage changed")
    boundary = _mapping(v5_report.get("claim_boundary"), "V5 scientific claim boundary")
    for key in ("protected_accessed", "official_test_accessed", "sealed_accessed", "stage_b_executed", "formal_promotion", "sampled_generation_executed", "scale_extraction_executed", "prior_generation_executed"):
        _require(boundary.get(key) is False, f"V5 authority crossed forbidden boundary: {key}")
    transfer_entry = _mapping(attempt03_report.get("entrypoint"), "V5 transfer-coupling entrypoint")
    _require(transfer_entry.get("sha256") == V5_ATTEMPT_03_ENTRYPOINT_SHA256, "stabilized V5 transfer-coupling entrypoint identity changed")
    _validate_file(root, str(transfer_entry.get("path")), V5_ATTEMPT_03_ENTRYPOINT_SHA256, "stabilized V5 transfer-coupling entrypoint")
    control_payload, control_state = _load_checkpoint(root, V5_CONTROL_PATH, V5_CONTROL_SHA256, "control")
    candidate_payload, candidate_state = _load_checkpoint(root, V5_CANDIDATE_PATH, V5_CANDIDATE_SHA256, "candidate")
    v4_payload, v4_state = _load_v4_generator(root)
    _require(torch.equal(control_state["generator.phase"], candidate_state["generator.phase"]) and torch.equal(control_state["generator.phase"], v4_state["generator.phase"]), "shared frozen generator phase differs")
    _require(torch.equal(control_state["ols.operator.transport"], candidate_state["ols.operator.transport"]) and torch.equal(control_state["ols.operator.transport"], v4_state["ols.operator.transport"]), "shared frozen T=I_100 transport differs")
    _require(torch.equal(control_payload["prototypes"], candidate_payload["prototypes"]), "control/candidate prototypes differ")
    prototypes = control_payload["prototypes"].detach().cpu().contiguous()
    cuda_environment = _prepare_cuda_runtime_dll_search_path()
    device, device_facts = _cuda_facts()
    _require(tuple(device_facts["compute_capability"]) == EXPECTED_COMPUTE_CAPABILITY and int(device_facts["vram_bytes"]) >= EXPECTED_VRAM_BYTES, "CUDA target differs from the validated V5 runtime")
    _require(shutil.disk_usage((root / "outputs").resolve()).free >= MIN_FREE_BYTES_FOR_OUTPUT, "insufficient output storage headroom")
    contract = load_v2_training_contract(repo_root=root)
    smooth, smooth_identity = _load_smooth(root, device)
    probe = _new_frozen_consumer(contract, v4_state, device)
    runtime_identity = dict(probe.runtime_identity)
    runtime_identity["runtime_state_tree_sha256"] = _digest_tree(_cpu_tree(probe.runtime.state_dict()))
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selection_authority = {
        "semantic_authority_id": historical["semantic_authority_id"],
        "producer_attempt": historical["producer_attempt"],
        "selection": historical["selection"],
    }
    return {
        "git": git,
        "audit": audit,
        "historical": historical,
        "attempt03_report": attempt03_report,
        "attempt03_authorities": attempt03_auth,
        "v5_report": v5_report,
        "v5_lineage": v5_lineage,
        "v4_payload": v4_payload,
        "v4_state": v4_state,
        "control_payload": control_payload,
        "control_state": control_state,
        "candidate_payload": candidate_payload,
        "candidate_state": candidate_state,
        "prototypes": prototypes,
        "templates": historical["templates"],
        "contract": contract,
        "smooth": smooth,
        "smooth_identity": smooth_identity,
        "device": device,
        "device_facts": device_facts,
        "cuda_environment": cuda_environment,
        "runtime_identity": runtime_identity,
        "selection_authority": selection_authority,
    }


def _synthetic_provider() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples_per_class = 2
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(samples_per_class)
    rows = int(labels.shape[0])
    ramp = torch.linspace(0.0, 1.0, rows * 28 * 28, dtype=torch.float32).reshape(rows, 1, 28, 28)
    images = (ramp * 0.75 + labels.to(torch.float32).reshape(rows, 1, 1, 1) / 40.0).clamp(0.0, 1.0).contiguous()
    templates = torch.stack([images[labels == digit].mean(dim=0) for digit in range(CLASS_COUNT)]).contiguous()
    return images, labels, torch.arange(rows, dtype=torch.long), templates


def _run_synthetic_preflight(deps: Mapping[str, Any]) -> dict[str, Any]:
    images, labels, indices, templates = _synthetic_provider()
    with tempfile.TemporaryDirectory(prefix="lightgen_v5_operating_point_") as temporary:
        target = Path(temporary)
        control_consumer = _new_frozen_consumer(deps["contract"], deps["control_state"], deps["device"])
        candidate_consumer = _new_frozen_consumer(deps["contract"], deps["candidate_state"], deps["device"])
        optical_consumer = _new_frozen_consumer(deps["contract"], deps["v4_state"], deps["device"])
        z_control = _encode(control_consumer, images)
        z_candidate = _encode(candidate_consumer, images)
        forward_count = 2 * math.ceil(images.shape[0] / LOGICAL_BATCH_SIZE)

        def optical_forward(value: torch.Tensor) -> torch.Tensor:
            nonlocal forward_count
            forward_count += math.ceil(value.shape[0] / LOGICAL_BATCH_SIZE)
            return _forward_external(optical_consumer, value)

        def smooth_forward(value: torch.Tensor) -> torch.Tensor:
            return _smooth_logits(deps["smooth"], value, deps["device"])

        bundle = _run_core(
            z_control=z_control,
            z_candidate=z_candidate,
            prototypes=deps["prototypes"],
            templates=templates,
            labels=labels,
            optical_forward=optical_forward,
            smooth_forward=smooth_forward,
            artifact_root=target,
            artifact_target=target,
            artifact_task_label="synthetic_preflight",
            selection_indices=indices,
        )
        _require(bundle["sufficient_artifact"]["reloaded"] and bundle["raw_artifact"]["reloaded"] and bundle["bootstrap_artifact"]["reloaded"], "synthetic artifact reload failed")
        _require(forward_count == 2 * math.ceil(images.shape[0] / LOGICAL_BATCH_SIZE) + 4 * math.ceil(images.shape[0] / LOGICAL_BATCH_SIZE), "synthetic operating-point forward count changed")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in deps["smooth"].parameters()), "synthetic Smooth evaluator gradient firewall failed")
        del control_consumer, candidate_consumer
        if deps["device"].type == "cuda":
            torch.cuda.empty_cache()
        return {
            "result": "SYNTHETIC_PREFLIGHT_PASSED",
            "scientific_meaning": "none; exact-core engineering acceptance only",
            "scientific_forward_count": 0,
            "synthetic_forward_count": forward_count,
            "operating_points": ["control", "candidate"],
            "target_mapping": "t=(y+1) mod 10",
            "sufficient_reload": True,
            "raw_reload": True,
            "bootstrap_reload": True,
            "no_real_data": True,
        }


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha: str) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_OPERATING_POINT_DIAGNOSIS_NOT_COMPLETED",
        "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha},
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {
            "diagnosis_authorization_consumed": True,
            "diagnosis_executed": False,
            "scientific_forward_count": 0,
            "optimizer_created": False,
            "backward_executed": False,
            "parameter_update_executed": False,
            "optical_update_executed": False,
            "training_executed": False,
            "stage_b_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "next": None,
    }


def _resolve_output(root: Path, output_dir: Path | None) -> Path:
    target = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "diagnosis output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing diagnosis output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _build_input(root: Path, started: Mapping[str, Any], deps: Mapping[str, Any], selection_metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization": {
            "identity": "DIAGNOSE_V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_V1",
            "one_shot": True,
            "forward_only": True,
            "no_update": True,
        },
        "authorities": {
            "historical_class_direction": {key: value for key, value in deps["historical"].items() if key != "templates"},
            "v5_attempt_03": {
                "report": {"path": _relative(root, V5_ATTEMPT_03_REPORT_PATH), "sha256": V5_ATTEMPT_03_REPORT_SHA256},
                "lineage": {"path": _relative(root, V5_ATTEMPT_03_LINEAGE_PATH), "sha256": V5_ATTEMPT_03_LINEAGE_SHA256},
                "sufficient": {"path": _relative(root, V5_ATTEMPT_03_SUFFICIENT_PATH), "sha256": V5_ATTEMPT_03_SUFFICIENT_SHA256},
                "classification": deps["attempt03_report"]["classification"],
            },
            "v5_control_step_300_checkpoint": {"path": V5_CONTROL_PATH.as_posix(), "sha256": V5_CONTROL_SHA256, "step": 300},
            "v5_candidate_step_300_checkpoint": {"path": V5_CANDIDATE_PATH.as_posix(), "sha256": V5_CANDIDATE_SHA256, "step": 300},
            "v4_step_300_generator": {"path": V4_GENERATOR_PATH.as_posix(), "sha256": V4_GENERATOR_SHA256, "step": 300},
            "smooth_evaluator": deps["smooth_identity"],
            "v2_contract": {"path": deps["contract"].path.as_posix(), "sha256": deps["contract"].sha256},
            "runtime": deps["runtime_identity"],
        },
        "protocol": {
            "operating_points": {
                "control": "z_C=E_control(x)",
                "candidate": "z_V=E_candidate(x)",
            },
            "intervention": "z_shift=z-mu_y+mu_t",
            "target_rule": "t=(y+1) mod 10",
            "transport": "T=I_100 coefficient space",
            "shared_writer": "W_OLS modal projection/re-emission",
            "template_metric": "M_template(r)=d(r,tau_y)-d(r,tau_t); Delta_template=M_shift-M_base",
            "smooth_metric": "M_smooth(r)=score_t(r)-score_y(r); Delta_smooth=M_shift-M_base",
            "template_distance": "mean squared pixel distance after raw CMOS clamp [0,1] and adaptive_avg_pool2d to 28x28",
            "smooth_representation": "SmoothMNISTEvaluator v2.1 canonical pre-softmax logits",
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired, class-stratified over 100 examples per source class", "quantile": "torch.quantile interpolation=linear"},
            "transfer_rule": "metric transfer iff mean > 0 and paired bootstrap 95% CI lower bound > 0",
            "known_cyclic_rule": "template and Smooth transfer true and both effects have at least two positive cyclic pair means",
        },
        "data_roles": {
            **dict(selection_metadata),
            "TRAIN": {"source": "frozen checkpoint prototypes and historical TRAIN templates; no new encoder forward", "rows": 55_000},
            "DEV": {"source": "balanced exact selection", "rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": SELECTION_LABELS_SHA256},
            "calibration": {"rows_read": 0},
            "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
            "official_test": {"rows_read": 0, "status": "forbidden; path not opened"},
            "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
            "labels_as_model_input": False,
        },
        "provenance": {
            "git_start": dict(started),
            "producer_head_is_provenance_only": True,
            "hash_policy": "HASH_ON_TRUST_BOUNDARY_V2",
            "input_self_hash": "not_embedded_to_avoid_self_reference",
        },
        "claim_boundary": {"scientific_forward_count_at_input": 0, "optimizer_created": False, "backward_executed": False, "parameter_update_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False},
    }


def _build_report(
    root: Path,
    started: Mapping[str, Any],
    ended: Mapping[str, Any],
    entrypoint_sha: str,
    deps: Mapping[str, Any],
    selection_metadata: Mapping[str, Any],
    selection_indices: torch.Tensor,
    labels: torch.Tensor,
    bundle: Mapping[str, Any],
    input_artifact: Mapping[str, Any],
    audit_artifact: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    forward_count: int,
    smooth_evaluation_count: int,
) -> dict[str, Any]:
    analyses = bundle["analyses"]
    classification = str(bundle["classification"])
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": classification,
        "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha},
        "provenance": {
            "git_start": dict(started),
            "git_end": dict(ended),
            "branch_unchanged": started.get("branch") == ended.get("branch"),
            "head_unchanged": started.get("head") == ended.get("head"),
            "worktree_status_unchanged": started.get("status_sha256") == ended.get("status_sha256"),
            "hash_policy": "HASH_ON_TRUST_BOUNDARY_V2",
        },
        "input": dict(input_artifact),
        "audit": dict(audit_artifact),
        "authorities": {
            "historical_semantic_authority": {key: value for key, value in deps["historical"].items() if key != "templates"},
            "v5_attempt_03": {"report": {"path": _relative(root, V5_ATTEMPT_03_REPORT_PATH), "sha256": V5_ATTEMPT_03_REPORT_SHA256}, "lineage": {"path": _relative(root, V5_ATTEMPT_03_LINEAGE_PATH), "sha256": V5_ATTEMPT_03_LINEAGE_SHA256}, "sufficient": {"path": _relative(root, V5_ATTEMPT_03_SUFFICIENT_PATH), "sha256": V5_ATTEMPT_03_SUFFICIENT_SHA256}, "classification": deps["attempt03_report"]["classification"]},
            "v5_control_step_300_checkpoint": {"path": V5_CONTROL_PATH.as_posix(), "sha256": V5_CONTROL_SHA256, "step": 300},
            "v5_candidate_step_300_checkpoint": {"path": V5_CANDIDATE_PATH.as_posix(), "sha256": V5_CANDIDATE_SHA256, "step": 300},
            "v4_step_300_generator": {"path": V4_GENERATOR_PATH.as_posix(), "sha256": V4_GENERATOR_SHA256, "step": 300},
            "smooth_evaluator": deps["smooth_identity"],
            "v2_contract": {"path": deps["contract"].path.as_posix(), "sha256": deps["contract"].sha256},
            "runtime": deps["runtime_identity"],
        },
        "selection": {"rows": int(labels.shape[0]), "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": tensor_sha256(selection_indices), "labels_sha256": tensor_sha256(labels), "target_mapping": "t=(y+1) mod 10"},
        "data": {**dict(selection_metadata), "values_read_for_evaluation": SELECTION_ROWS, "calibration_rows_read": 0, "protected_rows_read": 0, "official_test_rows_read": 0, "sealed_rows_read": 0},
        "runtime": {
            "device": deps["device_facts"],
            "cuda_environment": deps["cuda_environment"],
            "identity": deps["runtime_identity"],
            "transport": "T=I_100 coefficient space",
            "shared_w_ols": True,
            "modal_projection_boundary": True,
            "frozen_encoder_and_generator": True,
            "optimizer": False,
            "backward": False,
            "parameter_update": False,
            "optical_update": False,
        },
        "execution": {
            "diagnosis_executed": True,
            "scientific_forward_count": forward_count,
            "control_encoder_forwards": SELECTION_ROWS // LOGICAL_BATCH_SIZE,
            "candidate_encoder_forwards": SELECTION_ROWS // LOGICAL_BATCH_SIZE,
            "optical_endpoint_forwards_per_endpoint": SELECTION_ROWS // LOGICAL_BATCH_SIZE,
            "optical_endpoints": ["control_base", "control_shifted", "candidate_base", "candidate_shifted"],
            "smooth_endpoint_evaluations": smooth_evaluation_count,
            "exact_core_synthetic_preflight": synthetic,
            "sufficient_persisted_before_bootstrap_classification_report": True,
            "sufficient_reloaded_before_bootstrap_classification_report": True,
        },
        "operating_points": {
            "control": {key: value for key, value in analyses["control"].items() if key not in {"bootstrap_tensors", "delta_tensors"}},
            "candidate": {key: value for key, value in analyses["candidate"].items() if key not in {"bootstrap_tensors", "delta_tensors"}},
        },
        "classification_flags": dict(bundle["classification_flags"]),
        "artifacts": {
            "integrity_audit": dict(audit_artifact),
            "input": dict(input_artifact),
            "sufficient_observations": dict(bundle["sufficient_artifact"]),
            "raw_outputs": dict(bundle["raw_artifact"]),
            "paired_bootstrap": dict(bundle["bootstrap_artifact"]),
        },
        "fact_model_assumption_design_decision": {
            "FACT": [
                "The historical valid replication supplies the fixed cyclic class-direction authority and exact TRAIN image templates.",
                "The V5 control and candidate step-300 encoders, frozen V4 step-300 generator, T=I_100 and SmoothMNISTEvaluator v2.1 were consumed without updates.",
                "The report contains measured template/Smooth deltas, rank movement, prediction-rate movement, raw validity and ten cyclic pair summaries for both operating points.",
            ],
            "MODEL_ASSUMPTION": [
                "A positive template and independent Smooth direction at an operating point is bounded evidence that the known cyclic intervention remains transferable there.",
                "Failure at control is interpreted as operating-point-dependent transfer loss, not as an absolute generator-capacity or semantic-generation conclusion.",
            ],
            "DESIGN_DECISION": [
                "Use exactly t=(y+1) mod 10, no nearest-wrong selection or target search.",
                "Declare a metric transferred only when mean > 0 and paired bootstrap 95% CI lower bound > 0.",
                "Declare known cyclic transfer non-isolated only when both metric spaces have at least two positive cyclic pair means.",
                "Apply the four-way classification rule frozen in the input contract.",
            ],
        },
        "forbidden_actions": {
            "v5_retry": False,
            "v6_training": False,
            "optimizer": False,
            "backward": False,
            "parameter_update": False,
            "optical_update": False,
            "stage_b": False,
            "scale_extraction": False,
            "prior_generation": False,
            "sampled_generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
            "nearest_wrong_intervention": False,
            "target_search": False,
            "architecture_change": False,
        },
        "claim_boundary": {
            "diagnosis_authorization_consumed": True,
            "diagnosis_executed": True,
            "scientific_forward_count": forward_count,
            "optimizer_created": False,
            "backward_executed": False,
            "parameter_update_executed": False,
            "optical_update_executed": False,
            "training_executed": False,
            "stage_b_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "next": "one-shot diagnosis consumed; no retry and no downstream authorization",
    }


def run_preflight(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    deps = _resolve_dependencies(root)
    synthetic = _run_synthetic_preflight(deps)
    return {
        "result": "READY_FOR_SINGLE_SCIENTIFIC_DIAGNOSIS",
        "status": "valid_completed",
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "scientific_meaning": "none; authority/audit/exact-core synthetic acceptance only",
        "attempt03_integrity_audit": deps["audit"]["audit"],
        "historical_authority": {key: value for key, value in deps["historical"].items() if key != "templates"},
        "runtime": {"device": deps["device_facts"], "identity": deps["runtime_identity"], "smooth": deps["smooth_identity"]},
        "synthetic": synthetic,
        "scientific_forward_count": 0,
        "real_selection_values_read": 0,
        "claim_boundary": {"scientific_execution": False, "optimizer": False, "backward": False, "update": False, "protected": False, "official_test": False, "sealed": False, "stage_b": False, "generation": False, "formal_promotion": False},
    }


def run_diagnosis(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    started = git_facts(root)
    target = _resolve_output(root, output_dir)
    entrypoint_sha = file_sha256(Path(__file__).resolve())
    payload = _base_payload(root, started, entrypoint_sha)
    forward_count = 0
    smooth_evaluation_count = 0
    try:
        audit_result = _audit_attempt03(root, target)
        deps = _resolve_dependencies(root)
        synthetic = _run_synthetic_preflight(deps)
        selection_images, selection_labels, selection_indices, selection_metadata = _load_selection(root, deps["historical"])
        input_payload = _build_input(root, started, deps, selection_metadata)
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V5 operating-point diagnosis input")
        _require(_read_json(input_path, "operating-point diagnosis input").get("schema") == INPUT_SCHEMA, "diagnosis input reload changed")
        input_artifact = {"path": _relative(root, input_path), "sha256": file_sha256(input_path), "schema": INPUT_SCHEMA}
        control_consumer = _new_frozen_consumer(deps["contract"], deps["control_state"], deps["device"])
        candidate_consumer = _new_frozen_consumer(deps["contract"], deps["candidate_state"], deps["device"])
        optical_consumer = _new_frozen_consumer(deps["contract"], deps["v4_state"], deps["device"])
        z_control = _encode(control_consumer, selection_images)
        z_candidate = _encode(candidate_consumer, selection_images)
        batch_count = math.ceil(SELECTION_ROWS / LOGICAL_BATCH_SIZE)
        forward_count = 2 * batch_count

        def optical_forward(value: torch.Tensor) -> torch.Tensor:
            nonlocal forward_count
            forward_count += math.ceil(value.shape[0] / LOGICAL_BATCH_SIZE)
            return _forward_external(optical_consumer, value)

        def smooth_forward(value: torch.Tensor) -> torch.Tensor:
            nonlocal smooth_evaluation_count
            smooth_evaluation_count += math.ceil(value.shape[0] / LOGICAL_BATCH_SIZE)
            return _smooth_logits(deps["smooth"], value, deps["device"])

        bundle = _run_core(
            z_control=z_control,
            z_candidate=z_candidate,
            prototypes=deps["prototypes"],
            templates=deps["templates"],
            labels=selection_labels,
            optical_forward=optical_forward,
            smooth_forward=smooth_forward,
            artifact_root=root,
            artifact_target=target,
            artifact_task_label="real_dev_scientific",
            selection_indices=selection_indices,
        )
        _require(forward_count == 6 * batch_count and smooth_evaluation_count == 4 * batch_count, "scientific forward/evaluator count changed")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in deps["smooth"].parameters()), "Smooth evaluator received a diagnosis gradient")
        del control_consumer, candidate_consumer, selection_images
        if deps["device"].type == "cuda":
            torch.cuda.empty_cache()
        ended = git_facts(root)
        report = _build_report(
            root,
            started,
            ended,
            entrypoint_sha,
            deps,
            selection_metadata,
            selection_indices,
            selection_labels,
            bundle,
            input_artifact,
            audit_result["audit"].get("artifact", audit_result["audit"]),
            synthetic,
            forward_count,
            smooth_evaluation_count,
        )
        _require(report["provenance"]["head_unchanged"] and report["provenance"]["branch_unchanged"] and report["provenance"]["worktree_status_unchanged"], "Git provenance changed during diagnosis")
        report_path = target / REPORT_NAME
        atomic_write_new_json(report_path, report, artifact_kind="V5 operating-point stability diagnosis report")
        reloaded_report = _read_json(report_path, "operating-point stability report")
        _require(reloaded_report.get("result") == "VALID_COMPLETED" and reloaded_report.get("classification") == bundle["classification"], "operating-point report reload changed")
        lineage_path = target / LINEAGE_NAME
        lineage = {
            "schema": LINEAGE_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "result": "VALID_COMPLETED",
            "status": "valid_completed",
            "classification": bundle["classification"],
            "report": {"path": _relative(root, report_path), "sha256": file_sha256(report_path)},
            "input": input_artifact,
            "artifacts": report["artifacts"],
            "semantic_authority_consumed": "OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED",
            "producer_attempt": "operating_point_stability_v1_attempt_01",
            "scientific_forward_count": forward_count,
            "optimizer_created": False,
            "backward_executed": False,
            "parameter_update_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "stage_b_executed": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "lineage_self_hash": "not_embedded_to_avoid_forward_reference",
        }
        atomic_write_new_json(lineage_path, lineage, artifact_kind="V5 operating-point stability diagnosis lineage")
        _require(_read_json(lineage_path, "operating-point stability lineage").get("report", {}).get("sha256") == file_sha256(report_path), "lineage report binding changed")
        report["artifacts"]["lineage"] = {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path), "schema": LINEAGE_SCHEMA}
        return report
    except (DiagnosisBlocked, InvalidDiagnosis, OSError, RuntimeError) as error:
        payload["stop"] = {
            "classification": error.classification if isinstance(error, DiagnosisBlocked) else "INVALID_HARNESS",
            "stage": error.stage if isinstance(error, DiagnosisBlocked) else "execution",
            "message": str(error),
            "details": error.details if isinstance(error, DiagnosisBlocked) else {},
        }
        payload["claim_boundary"]["diagnosis_executed"] = forward_count > 0
        payload["claim_boundary"]["scientific_forward_count"] = forward_count
        payload["execution"] = {"scientific_forward_count": forward_count, "smooth_endpoint_evaluations": smooth_evaluation_count}
        report_path = target / REPORT_NAME
        if not report_path.exists():
            atomic_write_new_json(report_path, payload, artifact_kind="invalid V5 operating-point stability diagnosis report")
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run_preflight(args.repo_root) if args.preflight_only else run_diagnosis(args.repo_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("result") in {"READY_FOR_SINGLE_SCIENTIFIC_DIAGNOSIS", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
