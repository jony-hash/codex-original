"""Run the bounded engineering preflight for the V5 direct OLS objective.

The consumer exercises the exact prototype, distance, lambda, checkpoint, and
encoder-only update core.  It runs a full synthetic end-to-end acceptance and
three real CUDA updates per arm through the production 2048-grid path.  It
does not execute the 300-step scientific comparison or read scientific DEV,
protected, test, or sealed data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, LightGen2DComplexOLSTrainingConsumerV2, file_sha256, git_facts, load_v2_training_contract, tensor_sha256
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import PRIOR_SIGMA, gauge_invariant_ols_representation
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_local_mnist, stratified_mnist_indices


TASK_ID = "PREFLIGHT_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5"
SUCCESS_CLASSIFICATION = "LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5_AUTHORIZABLE"
REPORT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_preflight_report.v1"
INPUT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_preflight_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_preflight_lineage.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_preflight_checkpoint.v1"
SYNTHETIC_OBSERVATIONS_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_synthetic_sufficient_observations.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "9a72e0bcf2de12ab9a0caa7151f273de76768dc7"
CONTRACT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_contract.v5.json"
)
DEFINITION_PATH = Path("scripts/research/lightgen/define_lightgen_2d_complex_ols_semantic_alignment_contract_v5.py")
V3_CONTROL_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_checkpoint.v1"
V4_GENERATOR_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_scientific_checkpoint.v1"
V3_CONTROL_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
OLS_TRANSPORT_SHA256 = "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"

LOGICAL_BATCH_SIZE = 10
OLS_PORT_COUNT = 100
PHASE_SHAPE = (2, 1, 100, 100)
PRODUCTION_GRID = 2048
SMOKE_GRID = 128
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEP = 2
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
EPSILON_DENOMINATOR = 1.0e-12
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STAGE_A_FIREWALL = 0.937
DETERMINISTIC_RTOL = 1.0e-5
DETERMINISTIC_ATOL = 1.0e-6

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_01"
)
INPUT_NAME = "lightgen_2d_complex_ols_semantic_alignment_preflight_input.v1.json"
REPORT_NAME = "lightgen_2d_complex_ols_semantic_alignment_preflight_report.v1.json"
LINEAGE_NAME = "lightgen_2d_complex_ols_semantic_alignment_preflight_lineage.v1.json"
CONTROL_CHECKPOINT_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_control_checkpoint.v1.pt"
CANDIDATE_CHECKPOINT_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_candidate_checkpoint.v1.pt"
SYNTHETIC_OBSERVATIONS_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_synthetic_sufficient_observations.v1.pt"


class InvalidPreflight(RuntimeError):
    """An authority, implementation, or engineering invariant is invalid."""


class PreflightBlocked(RuntimeError):
    """A valid boundary prevents the bounded preflight from continuing."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    """Expose the installed CUDA runtime DLLs to this Python process only."""

    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_python_process_only"}
    cuda_build = torch.version.cuda
    if not isinstance(cuda_build, str) or len(cuda_build.split(".")) != 2 or not all(part.isdigit() for part in cuda_build.split(".")):
        raise PreflightBlocked(
            "CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED",
            "cuda_dll_search_path",
            "PyTorch CUDA build cannot identify its environment DLL name",
            cuda_build=cuda_build,
        )
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    if not environment_bin.is_dir() or not dll_path.is_file():
        raise PreflightBlocked(
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidPreflight(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _finite_complex(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and torch.is_complex(value), f"{name} is not complex")
    _require(bool(torch.isfinite(value.real).all() and torch.isfinite(value.imag).all()), f"{name} is nonfinite")


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result), f"{name} norm is nonfinite")
    return result


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


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _max_state_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    _require(left.keys() == right.keys(), "state keys differ")
    maximum = 0.0
    for key in left:
        first, second = left[key], right[key]
        if torch.is_tensor(first) and torch.is_tensor(second):
            _require(first.shape == second.shape, f"state shape differs at {key}")
            if first.numel():
                delta = (first.to(torch.complex128) - second.to(torch.complex128)).abs().max()
                maximum = max(maximum, float(delta.cpu()))
        elif isinstance(first, Mapping) and isinstance(second, Mapping):
            maximum = max(maximum, _max_state_difference(first, second))
        elif first != second:
            return math.inf
    return maximum


def _max_numeric_difference(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0 if left is right else math.inf
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return math.inf
        return max((_max_numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_max_numeric_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else math.inf


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


def _capture_rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().cpu().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    cpu = state.get("cpu")
    cuda = state.get("cuda")
    _require(torch.is_tensor(cpu) and isinstance(cuda, list), "RNG state is malformed")
    _require(all(torch.is_tensor(value) for value in cuda), "CUDA RNG state is malformed")
    torch.random.set_rng_state(cpu.cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _state_digest(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> str:
    return _digest_tree(_cpu_tree(consumer.runtime.state_dict()))


def _authority_ref(root: Path, value: object, name: str) -> dict[str, Any]:
    ref = _mapping(value, name)
    path_value = ref.get("path")
    _require(isinstance(path_value, str), f"{name} lacks an explicit path")
    path = _path(root, path_value)
    research_root = Path("src/ldm_project/v2/application/research")
    if not path.is_file() and Path(path_value).parent == research_root:
        moved_path = root / research_root / "lightgen" / Path(path_value).name
        if moved_path.is_file():
            path = moved_path
    _require(path.is_file(), f"{name} is unavailable: {path}")
    return dict(ref)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidPreflight(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = _path(root, CONTRACT_PATH)
    _require(path.is_file(), f"V5 contract is unavailable: {path}")
    contract_sha = file_sha256(path)
    contract = _read_json(path, "V5 contract")
    _require(contract.get("schema") == "lightgen_2d_complex_ols_semantic_alignment_contract.v5", "V5 contract schema changed")
    _require(contract.get("task_id") == "DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5", "V5 contract task identity changed")
    _require(contract.get("result") == "READY" and contract.get("status") == "valid_completed", "V5 contract is not READY")
    _require(contract.get("outer_gate") == "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED", "V5 outer gate changed")
    _require(contract.get("governance", {}).get("training_authorized") is False, "V5 contract opened training authorization")
    definition = _mapping(contract.get("definition_entrypoint"), "V5 definition entrypoint")
    _require(isinstance(definition.get("path"), str), "V5 definition entrypoint path is missing")
    authorities = _mapping(contract.get("authorities"), "V5 authorities")
    source = _mapping(authorities.get("source"), "V5 source authorities")
    for key in ("v2_contract", "training_source", "aligned_runtime", "gauge_helper", "evaluator_implementation", "smooth_checkpoint"):
        _authority_ref(root, source.get(key), f"V5 source {key}")
    v3 = _mapping(authorities.get("v3_control"), "V5 V3 CONTROL authority")
    v4 = _mapping(authorities.get("v4_generator"), "V5 V4 generator authority")
    _authority_ref(root, _mapping(v3.get("checkpoint"), "V3 CONTROL checkpoint"), "V3 CONTROL checkpoint")
    _authority_ref(root, _mapping(v4.get("scientific_report"), "V4 scientific report"), "V4 scientific report")
    _authority_ref(root, _mapping(v4.get("scientific_lineage"), "V4 scientific lineage"), "V4 scientific lineage")
    _authority_ref(root, _mapping(v4.get("checkpoint"), "V4 generator checkpoint"), "V4 generator checkpoint")
    replication = _mapping(authorities.get("class_direction_replication"), "V5 class-direction authority")
    _authority_ref(root, replication.get("report"), "class-direction replication report")
    _authority_ref(root, replication.get("lineage"), "class-direction replication lineage")
    return contract, authorities, contract_sha


def _load_authority_checkpoint(root: Path, ref: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    path = _path(root, ref["path"])
    return _mapping(safe_torch_load(path, map_location="cpu"), name)


def _validate_authority_payloads(root: Path, contract: Mapping[str, Any], authorities: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    source = _mapping(authorities.get("source"), "V5 source authorities")
    v2_ref = _mapping(source.get("v2_contract"), "V2 contract authority")
    v2_contract = load_v2_training_contract(_path(root, v2_ref["path"]), repo_root=root)
    _require(v2_contract.payload.get("training_authorized") is False, "V2 contract opened training")
    architecture = _mapping(v2_contract.payload.get("architecture"), "V2 architecture")
    _require(architecture.get("same_real_and_generated_interface") is True, "V2 real/generated interface identity changed")
    _require(architecture.get("whole_field_identity_bypass") is False, "V2 whole-field identity bypass changed")
    _require(any("T=I_100" in str(item) and "W_OLS" in str(item) for item in architecture.get("immutable_components", [])), "V2 T=I_100/W_OLS identity changed")

    v3_ref = _mapping(_mapping(authorities.get("v3_control"), "V3 authority").get("checkpoint"), "V3 checkpoint ref")
    v3 = _load_authority_checkpoint(root, v3_ref, "V3 CONTROL checkpoint")
    _require(v3.get("schema") == V3_CONTROL_SCHEMA and v3.get("checkpoint_step") == 300 and v3.get("arm") == "control", "V3 CONTROL checkpoint identity changed")
    _require(v3.get("scientific_training_checkpoint") is True and v3.get("training_authorized") is False, "V3 CONTROL boundary changed")
    state3 = _mapping(v3.get("runtime_state"), "V3 CONTROL runtime state")
    _require(set(state3) == {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "V3 CONTROL runtime state keys changed")
    _require(torch.is_tensor(state3["encoder.phase"]) and tuple(state3["encoder.phase"].shape) == PHASE_SHAPE, "V3 CONTROL encoder shape changed")
    _require(tensor_sha256(state3["encoder.phase"]) == V3_CONTROL_ENCODER_SHA256, "V3 CONTROL encoder tensor changed")
    _require(torch.is_tensor(state3["ols.operator.transport"]) and torch.equal(state3["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V3 CONTROL transport is not T=I_100")
    _require(tensor_sha256(state3["ols.operator.transport"]) == OLS_TRANSPORT_SHA256, "V3 CONTROL transport tensor changed")

    v4_ref = _mapping(_mapping(authorities.get("v4_generator"), "V4 authority").get("checkpoint"), "V4 checkpoint ref")
    v4 = _load_authority_checkpoint(root, v4_ref, "V4 step-300 generator checkpoint")
    _require(v4.get("schema") == V4_GENERATOR_SCHEMA and v4.get("checkpoint_step") == 300, "V4 generator checkpoint identity changed")
    _require(v4.get("scientific_training_checkpoint") is True and v4.get("training_authorized") is False, "V4 generator boundary changed")
    state4 = _mapping(v4.get("runtime_state"), "V4 generator runtime state")
    _require(set(state4) == set(state3), "V4 runtime state keys differ from V3 CONTROL")
    _require(torch.is_tensor(state4["encoder.phase"]) and tensor_sha256(state4["encoder.phase"]) == V3_CONTROL_ENCODER_SHA256, "V4 encoder does not match V3 CONTROL")
    _require(torch.is_tensor(state4["ols.operator.transport"]) and torch.equal(state4["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V4 transport is not T=I_100")
    _require(tensor_sha256(state4["ols.operator.transport"]) == OLS_TRANSPORT_SHA256, "V4 transport tensor changed")
    _require(torch.is_tensor(state4["generator.phase"]) and tuple(state4["generator.phase"].shape) == PHASE_SHAPE, "V4 generator phase shape changed")

    smooth_ref = _mapping(source.get("smooth_checkpoint"), "Smooth checkpoint ref")
    smooth_payload = _load_authority_checkpoint(root, smooth_ref, "Smooth evaluator checkpoint")
    _require(smooth_payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    return v3, v4, v2_contract


def _combined_initial_state(v3: Mapping[str, Any], v4: Mapping[str, Any]) -> dict[str, Any]:
    state3 = _mapping(v3.get("runtime_state"), "V3 runtime state")
    state4 = _mapping(v4.get("runtime_state"), "V4 runtime state")
    combined = {key: _cpu_tree(value) for key, value in state3.items()}
    combined["generator.phase"] = _cpu_tree(state4["generator.phase"])
    _require(tensor_sha256(combined["encoder.phase"]) == V3_CONTROL_ENCODER_SHA256, "combined encoder identity changed")
    _require(tensor_sha256(combined["generator.phase"]) == tensor_sha256(state4["generator.phase"]), "combined generator identity changed")
    _require(torch.equal(combined["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "combined OLS transport is not T=I_100")
    return combined


def _new_consumer(contract: Any, device: torch.device | str, grid: int) -> LightGen2DComplexOLSTrainingConsumerV2:
    return LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=grid)


def _load_runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2, state: Mapping[str, Any]) -> None:
    consumer.runtime.load_state_dict(dict(state), strict=True)
    consumer._enforce_parameter_firewall()


def _configure_encoder_only(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> dict[str, Any]:
    named = dict(consumer.runtime.named_parameters())
    _require(set(named) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name == "encoder.phase")
        parameter.grad = None
    _require(named["encoder.phase"].requires_grad and not named["generator.phase"].requires_grad, "encoder-only firewall is inactive")
    consumer.runtime.train()
    return {
        "trainable_names": ["encoder.phase"],
        "frozen_names": ["generator.phase"],
        "trainable_count": int(named["encoder.phase"].numel()),
        "frozen_parameter_count": int(named["generator.phase"].numel()),
    }


def _configure_all_frozen(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> None:
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    consumer.runtime.eval()


def _runtime_frozen_state(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> dict[str, Any]:
    state = _cpu_tree(consumer.runtime.state_dict())
    return {key: state[key] for key in state if key != "encoder.phase"}


def _validate_result(result: Any, batch_size: int, *, require_grad: bool) -> torch.Tensor:
    _require(result.image_input is not None and result.ols_input_field is not None and result.z_ols.source == "real_encoded", "real image path lost the encoder/OLS boundary")
    _require(tuple(result.z_ols.coefficients.shape) == (batch_size, OLS_PORT_COUNT) and result.z_ols.coefficients.dtype == torch.complex64, "real OLS z shape/dtype changed")
    if require_grad:
        _require(result.z_ols.coefficients.requires_grad, "real OLS z is detached from encoder.phase")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "shared W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "T=I_100 provenance changed")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(tuple(raw.shape) == (batch_size, 1, 200, 200) and raw.dtype == torch.float32, "raw CMOS shape/dtype changed")
    _require(raw.shape == target.shape and target.dtype == torch.float32, "image target shape/dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target, "SLM image target")
    return raw


def _raw_summary(raw: torch.Tensor) -> dict[str, Any]:
    value = raw.detach().cpu()
    per_image_variance = value.flatten(1).var(dim=1, unbiased=False)
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "per_image_variance_min": float(per_image_variance.min()),
        "per_image_variance_max": float(per_image_variance.max()),
        "structured": bool((per_image_variance > 0.0).all()),
        "nonconstant": bool(float(value.std(unbiased=False)) > 0.0),
        "tensor_sha256": tensor_sha256(value),
    }
    _require(summary["finite"] and summary["structured"] and summary["nonconstant"], "raw CMOS is invalid or constant")
    return summary


def _gauge_complex(z: torch.Tensor) -> torch.Tensor:
    representation = gauge_invariant_ols_representation(z)
    _require(representation.shape[-1] == 2 * OLS_PORT_COUNT, "gauge representation dimension changed")
    result = torch.complex(representation[:, :OLS_PORT_COUNT] * PRIOR_SIGMA, representation[:, OLS_PORT_COUNT:] * PRIOR_SIGMA)
    _finite_complex(result, "gauge-canonical complex OLS")
    return result


def prototype_distance(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Return the existing gauge-compatible squared distance to each prototype."""

    _require(z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT and torch.is_complex(z), "query z must be [B,100] complex")
    _require(prototypes.ndim == 2 and prototypes.shape == (10, OLS_PORT_COUNT) and torch.is_complex(prototypes), "prototypes must be [10,100] complex")
    query = gauge_invariant_ols_representation(z)
    support = gauge_invariant_ols_representation(prototypes)
    distances = (query[:, None, :] - support[None, :, :]).square().sum(dim=2)
    _finite(distances, "prototype distances")
    return distances


def prototype_logits(z: torch.Tensor, prototypes: torch.Tensor, s_mu: float) -> torch.Tensor:
    _require(math.isfinite(float(s_mu)) and float(s_mu) > 0.0, "prototype scale must be finite and positive")
    logits = -prototype_distance(z, prototypes) / (float(s_mu) + EPSILON_DENOMINATOR)
    _finite(logits, "prototype logits")
    return logits


def prototype_loss(z: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, s_mu: float) -> torch.Tensor:
    _require(labels.ndim == 1 and labels.shape[0] == z.shape[0] and labels.dtype == torch.long, "prototype labels are invalid")
    logits = prototype_logits(z, prototypes, s_mu)
    loss = F.cross_entropy(logits, labels)
    _finite(loss, "prototype loss")
    return loss


def lambda_proto_from_gradients(g_image: float, g_proto: float) -> float:
    """Apply the preregistered one-shot gradient-scaling rule exactly."""

    _require(math.isfinite(float(g_image)) and math.isfinite(float(g_proto)), "lambda gradients are nonfinite")
    _require(float(g_image) > 0.0 and float(g_proto) > 0.0, "lambda gradients must be strictly positive")
    value = float(g_image) / (float(g_proto) + EPSILON_DENOMINATOR)
    _require(math.isfinite(value), "lambda_proto is nonfinite")
    return value


def construct_prototypes(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    expected_rows: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute frozen class means from the initial encoder using TRAIN rows only."""

    _require(train_indices.ndim == 1 and train_indices.dtype == torch.long, "prototype TRAIN indices are invalid")
    _require(train_indices.numel() == expected_rows, "prototype TRAIN row count changed")
    initial_state = _state_digest(consumer)
    parts: list[torch.Tensor] = []
    train_images = images.index_select(0, train_indices)
    train_labels = labels.index_select(0, train_indices)
    _require(train_images.shape[0] == expected_rows, "prototype TRAIN image rows changed")
    consumer.runtime.eval()
    with torch.no_grad():
        for start in range(0, expected_rows, LOGICAL_BATCH_SIZE):
            batch = train_images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            result = consumer.runtime.forward_from_image(batch)
            _validate_result(result, int(batch.shape[0]), require_grad=False)
            parts.append(result.z_ols.coefficients.detach().cpu().contiguous())
            del batch, result
    torch.cuda.synchronize(consumer.device) if consumer.device.type == "cuda" else None
    z_train = torch.cat(parts, dim=0)
    _require(tuple(z_train.shape) == (expected_rows, OLS_PORT_COUNT) and z_train.dtype == torch.complex64, "prototype TRAIN z shape/dtype changed")
    _finite_complex(z_train, "prototype TRAIN z")
    _require(initial_state == _state_digest(consumer), "prototype construction changed frozen runtime state")

    canonical = _gauge_complex(z_train)
    prototype_parts: list[torch.Tensor] = []
    counts: list[int] = []
    for digit in range(10):
        values = canonical[train_labels == digit]
        _require(values.shape[0] > 0, f"prototype class {digit} is undefined")
        counts.append(int(values.shape[0]))
        prototype_parts.append(values.mean(dim=0))
    prototypes = torch.stack(prototype_parts, dim=0).contiguous().detach()
    _require(tuple(prototypes.shape) == (10, OLS_PORT_COUNT) and prototypes.dtype == torch.complex64, "prototype shape/dtype changed")
    _finite_complex(prototypes, "complex OLS prototypes")
    summary = {
        "source_role": "TRAIN only",
        "rows": expected_rows,
        "counts_by_class": counts,
        "latent_domain": "full complex C^100",
        "aggregation": "mean of gauge-canonicalized initial V3 CONTROL encoder z per class",
        "gauge_representation": "existing gauge_invariant_ols_representation with sigma=1e-3 and epsilon=1e-6",
        "z_train_tensor_sha256": tensor_sha256(z_train),
        "canonical_z_tensor_sha256": tensor_sha256(canonical),
        "prototypes_tensor_sha256": tensor_sha256(prototypes),
        "prototypes_requires_grad": bool(prototypes.requires_grad),
        "runtime_state_unchanged": True,
    }
    return prototypes, summary


def prototype_scale(prototypes: torch.Tensor) -> tuple[float, dict[str, Any]]:
    matrix = prototype_distance(prototypes, prototypes)
    upper = matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)]
    _require(upper.numel() == 45, "prototype pair count changed")
    _finite(upper, "prototype pair distances")
    scale = float(torch.median(upper).cpu())
    _require(math.isfinite(scale) and scale > 0.0, "prototype scale is nonfinite or nonpositive")
    return scale, {
        "formula": "median of 45 unordered prototype pair distances",
        "pair_count": 45,
        "distance_min": float(upper.min()),
        "distance_max": float(upper.max()),
        "s_mu": scale,
        "finite": True,
        "positive": True,
        "pair_distance_tensor_sha256": tensor_sha256(upper),
    }


def _optimizer(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> torch.optim.Adam:
    parameter = dict(consumer.runtime.named_parameters())["encoder.phase"]
    _require(parameter.requires_grad, "encoder.phase is not trainable")
    optimizer = torch.optim.Adam(
        [parameter],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == [parameter], "optimizer parameter identity changed")
    return optimizer


def _objective_forward(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    s_mu: float,
    *,
    arm: str,
    lambda_proto: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Any]:
    result = consumer.runtime.forward_from_image(batch)
    raw = _validate_result(result, int(batch.shape[0]), require_grad=True)
    image_loss = (raw - result.image_input.slm_intensity).square().mean()
    _finite(image_loss, "image loss")
    if arm == "control":
        proto = None
        total = image_loss
    else:
        _require(arm == "candidate", "unknown V5 arm")
        proto = prototype_loss(result.z_ols.coefficients, labels, prototypes, s_mu)
        total = image_loss + float(lambda_proto) * proto
    _finite(total, f"{arm} total objective")
    return total, image_loss, proto, result


def _execute_step(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Adam,
    images: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    s_mu: float,
    *,
    arm: str,
    lambda_proto: float,
    step: int,
    keep_raw: bool,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    _require(step in {1, 2, 3}, "V5 preflight escaped the bounded three-step budget")
    named = dict(consumer.runtime.named_parameters())
    before = _cpu_tree(consumer.runtime.state_dict())
    before_generator = before["generator.phase"]
    before_ols = {key: before[key] for key in ("ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport")}
    optimizer.zero_grad(set_to_none=True)
    for parameter in named.values():
        parameter.grad = None
    total, image_loss, proto, result = _objective_forward(consumer, images, labels, prototypes, s_mu, arm=arm, lambda_proto=lambda_proto)
    total.backward()
    encoder_gradient = named["encoder.phase"].grad
    generator_gradient = named["generator.phase"].grad
    _require(encoder_gradient is not None, f"{arm} encoder.phase did not receive a gradient")
    _finite(encoder_gradient, f"{arm} encoder.phase gradient")
    encoder_gradient_norm = _norm(encoder_gradient, f"{arm} encoder.phase gradient")
    _require(encoder_gradient_norm > 0.0, f"{arm} encoder.phase gradient is zero")
    _require(generator_gradient is None, f"{arm} frozen generator received a gradient")
    _require(all(parameter.grad is None for name, parameter in named.items() if name != "encoder.phase"), f"{arm} frozen runtime parameter received a gradient")
    before_encoder = before["encoder.phase"]
    optimizer.step()
    after = _cpu_tree(consumer.runtime.state_dict())
    encoder_update_norm = float((after["encoder.phase"] - before_encoder).to(torch.float64).square().sum().sqrt())
    _require(math.isfinite(encoder_update_norm) and encoder_update_norm > 0.0, f"{arm} encoder update is zero or nonfinite")
    _require(torch.equal(after["generator.phase"], before_generator), f"{arm} generator.phase changed")
    _require(all(torch.equal(after[key], before_ols[key]) for key in before_ols), f"{arm} OLS state changed")
    raw_cpu = result.raw_intensity.detach().cpu().contiguous()
    raw_summary = _raw_summary(raw_cpu)
    trace = {
        "step": step,
        "arm": arm,
        "objective": "L_image" if arm == "control" else "L_image+lambda_proto*L_proto",
        "losses": {
            "L_image": float(image_loss.detach().cpu()),
            "L_proto": None if proto is None else float(proto.detach().cpu()),
            "L_total": float(total.detach().cpu()),
        },
        "lambda_proto": float(lambda_proto),
        "gradient_norms": {"encoder.phase": encoder_gradient_norm, "generator.phase": None},
        "update_norms": {"encoder.phase": encoder_update_norm, "generator.phase": 0.0},
        "encoder_changed": True,
        "generator_unchanged": True,
        "ols_unchanged": True,
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase"],
        "raw_cmos": raw_summary,
        "runtime_state_sha256_after": _digest_tree(after),
    }
    del total, image_loss, proto, result
    return trace, raw_cpu if keep_raw else None


def _checkpoint_payload(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Adam,
    sampler: DeterministicTrainRoleSampler,
    rng_state: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    prototypes: torch.Tensor,
    s_mu: float,
    lambda_proto: float,
    arm: str,
    contract_sha: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": contract_sha},
        "checkpoint_step": CHECKPOINT_STEP,
        "next_step": CHECKPOINT_STEP + 1,
        "arm": arm,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(rng_state),
        "initial_runtime_state_sha256": _digest_tree(initial_state),
        "prototypes": prototypes.detach().cpu().clone(),
        "prototypes_sha256": tensor_sha256(prototypes),
        "s_mu": float(s_mu),
        "lambda_proto": float(lambda_proto),
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"],
        "data_roles": {"train": True, "development": False, "protected": False, "test": False, "sealed": False},
        "scientific_training_checkpoint": False,
        "training_authorized": False,
        "scientific_comparison_executed": False,
        "stage_b_executed": False,
        "scale_extraction_executed": False,
        "prior_generation_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
    }


def _run_arm(
    root: Path,
    v2_contract: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    initial_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    prototypes: torch.Tensor,
    s_mu: float,
    lambda_proto: float,
    *,
    arm: str,
    device: torch.device,
    grid: int,
    output_path: Path,
    contract_sha: str,
    future_steps: int = SCIENTIFIC_STEPS,
    sampler_factory: Any = None,
) -> dict[str, Any]:
    consumer = _new_consumer(v2_contract, device, grid)
    _load_runtime_state(consumer, initial_state)
    firewall = _configure_encoder_only(consumer)
    optimizer = _optimizer(consumer)
    sampler_type = DeterministicTrainRoleSampler if sampler_factory is None else sampler_factory
    sampler = sampler_type(train_indices, labels, seed=1234, steps=future_steps)
    initial_digest = _state_digest(consumer)
    initial_rng_digest = _digest_tree(initial_rng)
    _require(initial_digest == _digest_tree(initial_state), f"{arm} initial state differs from matched state")
    _restore_rng_state(initial_rng)
    traces: list[dict[str, Any]] = []
    endpoint_raw: torch.Tensor | None = None
    checkpoint_path = output_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha: str | None = None
    checkpoint_rng_state: Any = None
    for step in range(1, PRECHECK_STEPS + 1):
        batch_indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(10, dtype=torch.long)), f"{arm} sampler labels changed")
        batch = images.index_select(0, batch_indices).to(device)
        trace, raw = _execute_step(consumer, optimizer, batch, batch_labels.to(device), prototypes.to(device), s_mu, arm=arm, lambda_proto=lambda_proto, step=step, keep_raw=step == PRECHECK_STEPS)
        trace["batch_indices_sha256"] = tensor_sha256(batch_indices)
        trace["batch_labels_sha256"] = tensor_sha256(batch_labels)
        traces.append(trace)
        endpoint_raw = raw
        del batch
        if step == CHECKPOINT_STEP:
            checkpoint_rng_state = _capture_rng_state()
            payload = _checkpoint_payload(consumer, optimizer, sampler, checkpoint_rng_state, initial_state, prototypes, s_mu, lambda_proto, arm, contract_sha)
            atomic_torch_save_new(checkpoint_path, payload, artifact_kind=f"V5 {arm} preflight checkpoint")
            checkpoint_sha = file_sha256(checkpoint_path)
    _require(endpoint_raw is not None and checkpoint_sha is not None, f"{arm} preflight endpoint/checkpoint missing")
    reference_final_state = _cpu_tree(consumer.runtime.state_dict())
    reference_optimizer_state = _cpu_tree(optimizer.state_dict())
    reference_trace = traces[-1]

    loaded = _mapping(safe_torch_load(checkpoint_path, expected_sha256=checkpoint_sha, map_location="cpu"), f"{arm} V5 checkpoint")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("arm") == arm and loaded.get("checkpoint_step") == CHECKPOINT_STEP, f"{arm} checkpoint schema/identity changed")
    _require(loaded.get("initial_runtime_state_sha256") == _digest_tree(initial_state), f"{arm} checkpoint initial state identity changed")
    _require(float(loaded.get("s_mu")) == float(s_mu) and float(loaded.get("lambda_proto")) == float(lambda_proto), f"{arm} checkpoint objective constants changed")
    _require(tensor_sha256(_mapping(loaded, f"{arm} checkpoint")["prototypes"]) == tensor_sha256(prototypes), f"{arm} checkpoint prototypes changed")
    resumed = _new_consumer(v2_contract, device, grid)
    _load_runtime_state(resumed, _mapping(loaded.get("runtime_state"), f"{arm} checkpoint runtime state"))
    _configure_encoder_only(resumed)
    resumed_optimizer = _optimizer(resumed)
    resumed_optimizer.load_state_dict(dict(_mapping(loaded.get("optimizer_state"), f"{arm} checkpoint optimizer state")))
    resumed_sampler = sampler_type(train_indices, labels, seed=1234, steps=future_steps)
    resumed_sampler.load_state_dict(_mapping(loaded.get("sampler_state"), f"{arm} checkpoint sampler state"))
    _restore_rng_state(_mapping(loaded.get("rng_state"), f"{arm} checkpoint RNG state"))
    resumed_indices, resumed_labels = resumed_sampler.next_batch()
    # The reference sampler has already advanced through step 3.  The next
    # batch is checked against the deterministic schedule by the step-3 trace
    # identity stored before the reference sampler advanced.
    reference_next_sampler = sampler_type(train_indices, labels, seed=1234, steps=future_steps)
    for _ in range(CHECKPOINT_STEP):
        reference_next_sampler.next_batch()
    expected_next_indices, expected_next_labels = reference_next_sampler.next_batch()
    _require(torch.equal(resumed_indices, expected_next_indices) and torch.equal(resumed_labels, expected_next_labels), f"{arm} checkpoint next batch changed")
    resumed_batch = images.index_select(0, resumed_indices).to(device)
    resumed_trace, resumed_raw = _execute_step(resumed, resumed_optimizer, resumed_batch, resumed_labels.to(device), prototypes.to(device), s_mu, arm=arm, lambda_proto=lambda_proto, step=PRECHECK_STEPS, keep_raw=True)
    _require(resumed_raw is not None, f"{arm} resumed endpoint raw is missing")
    trace_difference = _max_numeric_difference(reference_trace["losses"], resumed_trace["losses"])
    trace_difference = max(trace_difference, _max_numeric_difference(reference_trace["gradient_norms"], resumed_trace["gradient_norms"]))
    trace_difference = max(trace_difference, _max_numeric_difference(reference_trace["update_norms"], resumed_trace["update_norms"]))
    _require(trace_difference <= DETERMINISTIC_ATOL, f"{arm} checkpoint/resume trace differs: {trace_difference}")
    resumed_state = _cpu_tree(resumed.runtime.state_dict())
    resumed_optimizer_state = _cpu_tree(resumed_optimizer.state_dict())
    state_difference = _max_state_difference(reference_final_state, resumed_state)
    optimizer_equal = _tree_equal(reference_optimizer_state, resumed_optimizer_state)
    _require(state_difference <= DETERMINISTIC_ATOL and optimizer_equal, f"{arm} checkpoint/resume state differs: {state_difference}, optimizer_equal={optimizer_equal}")
    raw_difference = float((endpoint_raw.to(torch.float64) - resumed_raw.to(torch.float64)).abs().max())
    _require(raw_difference <= DETERMINISTIC_ATOL, f"{arm} checkpoint/resume raw endpoint differs: {raw_difference}")
    del resumed_batch, batch_indices, batch_labels, resumed_indices, resumed_labels
    return {
        "arm": arm,
        "firewall": firewall,
        "initial_state_sha256": initial_digest,
        "initial_rng_sha256": initial_rng_digest,
        "lambda_proto": float(lambda_proto),
        "traces": traces,
        "endpoint_raw": endpoint_raw,
        "final_runtime_state": reference_final_state,
        "checkpoint": {"path": checkpoint_path.relative_to(root).as_posix(), "sha256": checkpoint_sha, "step": CHECKPOINT_STEP, "loaded": True},
        "checkpoint_resume": {"trace_max_abs_difference": trace_difference, "runtime_state_max_abs_difference": state_difference, "optimizer_equal": optimizer_equal, "raw_endpoint_max_abs_difference": raw_difference, "next_batch_equal": True},
        "frozen_generator_unchanged": all(trace["generator_unchanged"] for trace in traces),
        "ols_unchanged": all(trace["ols_unchanged"] for trace in traces),
        "endpoint_raw_summary": traces[-1]["raw_cmos"],
    }


def _calibrate_lambda(
    v2_contract: Any,
    first_batch: torch.Tensor,
    first_labels: torch.Tensor,
    initial_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    prototypes: torch.Tensor,
    s_mu: float,
    *,
    device: torch.device,
    grid: int,
) -> dict[str, Any]:
    consumer = _new_consumer(v2_contract, device, grid)
    _load_runtime_state(consumer, initial_state)
    _configure_encoder_only(consumer)
    before_state = _state_digest(consumer)
    _restore_rng_state(initial_rng)
    batch = first_batch.to(device)
    labels = first_labels.to(device)
    named = dict(consumer.runtime.named_parameters())
    for parameter in named.values():
        parameter.grad = None
    total, image_loss, proto, result = _objective_forward(consumer, batch, labels, prototypes.to(device), s_mu, arm="candidate", lambda_proto=1.0)
    _require(proto is not None, "prototype loss was not constructed for lambda calibration")
    image_gradient = torch.autograd.grad(image_loss, named["encoder.phase"], retain_graph=True, allow_unused=True)[0]
    proto_gradient = torch.autograd.grad(proto, named["encoder.phase"], retain_graph=True, allow_unused=True)[0]
    _require(image_gradient is not None and proto_gradient is not None, "required lambda gradient is undefined")
    g_image = _norm(image_gradient, "g_image")
    g_proto = _norm(proto_gradient, "g_proto")
    _require(g_image > 0.0 and g_proto > 0.0 and math.isfinite(g_image) and math.isfinite(g_proto), "required lambda gradient is zero or nonfinite")
    lambda_proto = lambda_proto_from_gradients(g_image, g_proto)
    _require(_state_digest(consumer) == before_state, "lambda calibration changed runtime state")
    for parameter in named.values():
        parameter.grad = None
    return {
        "formula": "lambda_proto=g_image/(g_proto+1e-12)",
        "g_image": g_image,
        "g_proto": g_proto,
        "lambda_proto": lambda_proto,
        "image_loss": float(image_loss.detach().cpu()),
        "prototype_loss": float(proto.detach().cpu()),
        "gradient_norms": {"g_image": g_image, "g_proto": g_proto},
        "same_initial_state": True,
        "same_first_train_batch": True,
        "before_runtime_state_sha256": before_state,
        "after_runtime_state_sha256": _state_digest(consumer),
        "optimizer_updates_before_calibration": 0,
    }


def _load_smooth(root: Path, authorities: Mapping[str, Any]) -> tuple[SmoothMNISTEvaluator, dict[str, Any]]:
    source = _mapping(authorities.get("source"), "V5 source authorities")
    ref = _mapping(source.get("smooth_checkpoint"), "Smooth checkpoint authority")
    payload = _load_authority_checkpoint(root, ref, "Smooth evaluator checkpoint")
    evaluator = SmoothMNISTEvaluator()
    evaluator.load_state_dict(_mapping(payload.get("state_dict"), "Smooth evaluator state"), strict=True)
    evaluator.freeze()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator gradient firewall failed")
    return evaluator, {"checkpoint": dict(ref), "architecture_kind": payload.get("architecture_kind"), "state_dict_tree_sha256": _digest_tree(_cpu_tree(evaluator.state_dict())), "gradient_enabled": False}


def _smooth_logits(evaluator: SmoothMNISTEvaluator, raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and raw.shape[1] == 1, "Smooth raw input shape changed")
    view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
    with torch.no_grad():
        _, logits = evaluator(view)
    _finite(logits, "Smooth logits")
    _require(tuple(logits.shape) == (raw.shape[0], 10), "Smooth logits shape changed")
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters()), "Smooth evaluator gradient/state firewall changed")
    return logits.cpu().contiguous()


def ols_margin(z: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, s_mu: float) -> torch.Tensor:
    distances = prototype_distance(z, prototypes)
    rows = torch.arange(z.shape[0], dtype=torch.long, device=z.device)
    correct = distances[rows, labels]
    others = distances.clone()
    others[rows, labels] = torch.inf
    margin = others.min(dim=1).values - correct
    _finite(margin, "OLS prototype margin")
    return margin.detach().cpu().to(torch.float64)


def _per_class_recall(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    result = []
    for digit in range(10):
        mask = labels == digit
        _require(bool(mask.any()), f"synthetic evaluator class {digit} is absent")
        result.append(((prediction == digit) & mask).to(torch.float64).sum() / mask.to(torch.float64).sum())
    recall = torch.stack(result)
    _finite(recall, "per-class recall")
    return recall


def _bootstrap_mean_delta(control: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict[str, Any]:
    _require(control.shape == candidate.shape and control.ndim == 1 and labels.shape == control.shape, "bootstrap scalar inputs differ")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    replicates = []
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() > 0, f"bootstrap class {digit} is absent")
        draw = torch.randint(matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        delta = candidate.index_select(0, matches)[draw] - control.index_select(0, matches)[draw]
        replicates.append(delta.to(torch.float64).mean(dim=1))
    values = torch.stack(replicates, dim=0).mean(dim=0)
    quantile = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    return {"mean": float(values.mean()), "ci_lower": float(quantile[0]), "ci_upper": float(quantile[1]), "replicates": BOOTSTRAP_REPLICATES, "seed": seed, "tensor_sha256": tensor_sha256(values)}


def _bootstrap_recall_delta(control_logits: torch.Tensor, candidate_logits: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict[str, Any]:
    control_correct = control_logits.argmax(dim=1).eq(labels)
    candidate_correct = candidate_logits.argmax(dim=1).eq(labels)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    macro_replicates = []
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() > 0, f"bootstrap recall class {digit} is absent")
        draw = torch.randint(matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        c = control_correct.index_select(0, matches)[draw].to(torch.float64).mean(dim=1)
        t = candidate_correct.index_select(0, matches)[draw].to(torch.float64).mean(dim=1)
        macro_replicates.append(t - c)
    values = torch.stack(macro_replicates, dim=0).mean(dim=0)
    quantile = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    per_class = _per_class_recall(candidate_logits, labels) - _per_class_recall(control_logits, labels)
    return {"mean": float(values.mean()), "ci_lower": float(quantile[0]), "ci_upper": float(quantile[1]), "per_class_delta": [float(value) for value in per_class], "replicates": BOOTSTRAP_REPLICATES, "seed": seed, "tensor_sha256": tensor_sha256(values)}


def classify_v5_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "smooth_delta_at_least_0_05": float(metrics["smooth_delta"]) >= 0.05,
        "smooth_bootstrap_ci_lower_gt_zero": float(metrics["smooth_ci_lower"]) > 0.0,
        "ols_margin_mean_gt_zero": float(metrics["ols_margin_delta_mean"]) > 0.0,
        "ols_margin_bootstrap_ci_lower_gt_zero": float(metrics["ols_margin_ci_lower"]) > 0.0,
        "no_smooth_per_class_drop_gt_0_10": min(float(value) for value in metrics["smooth_per_class_delta"]) >= -0.10,
        "image_loss_degradation_within_0_005": float(metrics["image_loss_delta"]) <= 0.005,
        "raw_cmos_endpoint_panel_valid": bool(metrics["raw_cmos_valid"]),
    }
    return {"checks": checks, "all_frozen_criteria_pass": all(checks.values()), "classification": "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED" if all(checks.values()) else "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED", "outer_gate": "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED"}


def _synthetic_images() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    dev_labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    train = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
    dev = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
    for index, label in enumerate(train_labels.tolist()):
        row = (2 * label + index) % 28
        column = (3 * label + index) % 28
        train[index, 0, row, :] = 0.35 + 0.02 * label
        train[index, 0, :, column] = 0.25 + 0.01 * (index % 2)
    for index, label in enumerate(dev_labels.tolist()):
        row = (2 * label + index + 1) % 28
        column = (3 * label + index + 2) % 28
        dev[index, 0, row, :] = 0.30 + 0.02 * label
        dev[index, 0, :, column] = 0.20 + 0.01 * (index % 2)
    return train, train_labels, dev, dev_labels


class _SyntheticTrainRoleSampler:
    """Small deterministic sampler used only by the exact-core synthetic test."""

    SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_synthetic_sampler.v1"

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int, steps: int) -> None:
        _require(train_indices.ndim == 1 and train_indices.dtype == torch.long, "synthetic sampler indices are invalid")
        _require(labels.ndim == 1 and labels.dtype == torch.long and labels.shape == train_indices.shape, "synthetic sampler labels are invalid")
        _require(int(train_indices.numel()) == 20 and int(steps) == PRECHECK_STEPS, "synthetic sampler dimensions changed")
        self.train_indices = train_indices.detach().cpu().contiguous()
        self.labels = labels.detach().cpu().contiguous()
        self.seed = int(seed)
        self.steps = int(steps)
        self.next_step = 0
        self.positions = []
        for digit in range(10):
            matches = torch.nonzero(self.labels == digit, as_tuple=False).flatten()
            _require(matches.numel() == 2, f"synthetic sampler class {digit} cardinality changed")
            self.positions.append(matches)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.next_step < self.steps, "synthetic sampler exhausted")
        selected = torch.stack([matches[self.next_step % matches.numel()] for matches in self.positions])
        self.next_step += 1
        indices = self.train_indices.index_select(0, selected)
        labels = self.labels.index_select(0, selected)
        _require(torch.equal(labels, torch.arange(10, dtype=torch.long)), "synthetic sampler label order changed")
        return indices, labels

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "seed": self.seed, "steps": self.steps, "next_step": self.next_step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(state.get("schema") == self.SCHEMA, "synthetic sampler schema changed")
        _require(int(state.get("seed")) == self.seed and int(state.get("steps")) == self.steps, "synthetic sampler identity changed")
        next_step = int(state.get("next_step"))
        _require(0 <= next_step <= self.steps, "synthetic sampler cursor changed")
        self.next_step = next_step


def _evaluate_synthetic_endpoint(
    v2_contract: Any,
    state: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    s_mu: float,
    evaluator: SmoothMNISTEvaluator,
) -> dict[str, Any]:
    consumer = _new_consumer(v2_contract, "cpu", SMOKE_GRID)
    _load_runtime_state(consumer, state)
    _configure_all_frozen(consumer)
    raw_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE]
            result = consumer.runtime.forward_from_image(batch)
            _validate_result(result, int(batch.shape[0]), require_grad=False)
            raw_parts.append(result.raw_intensity.cpu().contiguous())
            z_parts.append(result.z_ols.coefficients.cpu().contiguous())
            losses.append((result.raw_intensity - result.image_input.slm_intensity).square().mean().cpu())
    raw = torch.cat(raw_parts, dim=0)
    z = torch.cat(z_parts, dim=0)
    image_loss = float(torch.stack(losses).mean())
    smooth_logits = _smooth_logits(evaluator, raw)
    margins = ols_margin(z, labels, prototypes, s_mu)
    return {"raw": raw, "z": z, "smooth_logits": smooth_logits, "image_loss": image_loss, "margins": margins, "raw_summary": _raw_summary(raw)}


def _run_synthetic_acceptance(
    root: Path,
    v2_contract: Any,
    initial_state: Mapping[str, Any],
    evaluator: SmoothMNISTEvaluator,
    contract_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    train_images, train_labels, dev_images, dev_labels = _synthetic_images()
    train_indices = torch.arange(train_images.shape[0], dtype=torch.long)
    consumer = _new_consumer(v2_contract, "cpu", SMOKE_GRID)
    _load_runtime_state(consumer, initial_state)
    _configure_all_frozen(consumer)
    prototypes, prototype_summary = construct_prototypes(consumer, train_images, train_labels, train_indices, expected_rows=20)
    s_mu, scale_summary = prototype_scale(prototypes)
    sampler = _SyntheticTrainRoleSampler(train_indices, train_labels, seed=1234, steps=PRECHECK_STEPS)
    first_indices, first_labels = sampler.next_batch()
    initial_rng = _capture_rng_state()
    calibration = _calibrate_lambda(v2_contract, train_images.index_select(0, first_indices), first_labels, initial_state, initial_rng, prototypes, s_mu, device=torch.device("cpu"), grid=SMOKE_GRID)
    synthetic_dir = output_dir / "synthetic"
    control = _run_arm(root, v2_contract, train_images, train_labels, train_indices, initial_state, initial_rng, prototypes, s_mu, 0.0, arm="control", device=torch.device("cpu"), grid=SMOKE_GRID, output_path=synthetic_dir / "control_checkpoint.v1.pt", contract_sha=contract_sha, future_steps=PRECHECK_STEPS, sampler_factory=_SyntheticTrainRoleSampler)
    candidate = _run_arm(root, v2_contract, train_images, train_labels, train_indices, initial_state, initial_rng, prototypes, s_mu, calibration["lambda_proto"], arm="candidate", device=torch.device("cpu"), grid=SMOKE_GRID, output_path=synthetic_dir / "candidate_checkpoint.v1.pt", contract_sha=contract_sha, future_steps=PRECHECK_STEPS, sampler_factory=_SyntheticTrainRoleSampler)
    control_endpoint = _evaluate_synthetic_endpoint(v2_contract, control["final_runtime_state"], dev_images, dev_labels, prototypes, s_mu, evaluator)
    candidate_endpoint = _evaluate_synthetic_endpoint(v2_contract, candidate["final_runtime_state"], dev_images, dev_labels, prototypes, s_mu, evaluator)
    smooth_bootstrap = _bootstrap_recall_delta(control_endpoint["smooth_logits"], candidate_endpoint["smooth_logits"], dev_labels, seed=BOOTSTRAP_SEED)
    margin_bootstrap = _bootstrap_mean_delta(control_endpoint["margins"], candidate_endpoint["margins"], dev_labels, seed=BOOTSTRAP_SEED)
    smooth_per_class_delta = [float(value) for value in (_per_class_recall(candidate_endpoint["smooth_logits"], dev_labels) - _per_class_recall(control_endpoint["smooth_logits"], dev_labels))]
    gate = classify_v5_gate({
        "smooth_delta": smooth_bootstrap["mean"],
        "smooth_ci_lower": smooth_bootstrap["ci_lower"],
        "ols_margin_delta_mean": margin_bootstrap["mean"],
        "ols_margin_ci_lower": margin_bootstrap["ci_lower"],
        "smooth_per_class_delta": smooth_per_class_delta,
        "image_loss_delta": candidate_endpoint["image_loss"] - control_endpoint["image_loss"],
        "raw_cmos_valid": all(value["raw_summary"]["finite"] and value["raw_summary"]["structured"] and value["raw_summary"]["nonconstant"] for value in (control_endpoint, candidate_endpoint)),
    })
    observations = {
        "schema": SYNTHETIC_OBSERVATIONS_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "synthetic_only": True,
        "labels": dev_labels,
        "control_smooth_logits": control_endpoint["smooth_logits"],
        "candidate_smooth_logits": candidate_endpoint["smooth_logits"],
        "control_ols_margin": control_endpoint["margins"],
        "candidate_ols_margin": candidate_endpoint["margins"],
        "claim_boundary": {"scientific_meaning": False, "scientific_comparison_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
    }
    observations_path = output_dir / SYNTHETIC_OBSERVATIONS_NAME
    atomic_torch_save_new(observations_path, observations, artifact_kind="V5 synthetic sufficient observations")
    observations_sha = file_sha256(observations_path)
    loaded = _mapping(safe_torch_load(observations_path, expected_sha256=observations_sha, map_location="cpu"), "synthetic sufficient observations")
    _require(loaded.get("schema") == SYNTHETIC_OBSERVATIONS_SCHEMA and loaded.get("synthetic_only") is True, "synthetic observations reload/schema failed")
    _require(torch.equal(loaded["labels"], dev_labels), "synthetic labels changed on reload")
    return {
        "synthetic_only": True,
        "train_rows": int(train_images.shape[0]),
        "dev_like_rows": int(dev_images.shape[0]),
        "prototype_summary": prototype_summary,
        "scale": scale_summary,
        "lambda_calibration": calibration,
        "control_three_step_trace": control["traces"],
        "candidate_three_step_trace": candidate["traces"],
        "checkpoint_resume": {"control": control["checkpoint_resume"], "candidate": candidate["checkpoint_resume"]},
        "endpoint": {"control": {"image_loss": control_endpoint["image_loss"], "raw_cmos": control_endpoint["raw_summary"], "smooth_logits_sha256": tensor_sha256(control_endpoint["smooth_logits"]), "ols_margin_sha256": tensor_sha256(control_endpoint["margins"])}, "candidate": {"image_loss": candidate_endpoint["image_loss"], "raw_cmos": candidate_endpoint["raw_summary"], "smooth_logits_sha256": tensor_sha256(candidate_endpoint["smooth_logits"]), "ols_margin_sha256": tensor_sha256(candidate_endpoint["margins"])}},
        "statistics": {"smooth_recall_bootstrap": smooth_bootstrap, "ols_margin_bootstrap": margin_bootstrap, "smooth_per_class_delta": smooth_per_class_delta, "image_loss_delta": candidate_endpoint["image_loss"] - control_endpoint["image_loss"]},
        "gate_classification": gate,
        "sufficient_observations": {"path": observations_path.relative_to(root).as_posix(), "sha256": observations_sha, "reloaded": True, "schema_valid": True},
        "scientific_meaning": "none; synthetic acceptance only",
    }


def _cuda_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreflightBlocked("BLOCKED_V5_PREFLIGHT_CUDA_UNAVAILABLE", "device", "CUDA:0 is unavailable", cuda_available=bool(torch.cuda.is_available()), device_count=int(torch.cuda.device_count()))
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    probe = torch.fft.fft2(torch.ones((8, 8), dtype=torch.complex64, device=device))
    torch.cuda.synchronize(device)
    _finite_complex(probe, "CUDA complex probe")
    return device, {"device": str(device), "device_count": int(torch.cuda.device_count()), "gpu": properties.name, "compute_capability": [int(properties.major), int(properties.minor)], "vram_bytes": int(properties.total_memory), "torch_version": torch.__version__, "pytorch_cuda_build": torch.version.cuda, "grid": PRODUCTION_GRID}


def _production_no_data_smoke(v2_contract: Any, initial_state: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    consumer = _new_consumer(v2_contract, device, PRODUCTION_GRID)
    _load_runtime_state(consumer, initial_state)
    _configure_all_frozen(consumer)
    synthetic = torch.linspace(0.0, 1.0, LOGICAL_BATCH_SIZE * 28 * 28, device=device, dtype=torch.float32).reshape(LOGICAL_BATCH_SIZE, 1, 28, 28)
    before = _state_digest(consumer)
    with torch.no_grad():
        result = consumer.runtime.forward_from_image(synthetic)
    raw = _validate_result(result, LOGICAL_BATCH_SIZE, require_grad=False)
    after = _state_digest(consumer)
    _require(before == after, "production no-data smoke changed runtime state")
    return {"executed": True, "data_source": "synthetic no-data tensor", "grid": PRODUCTION_GRID, "batch_size": LOGICAL_BATCH_SIZE, "raw_cmos": _raw_summary(raw), "runtime_state_unchanged": True, "modal_projection_active": True, "transport": "T=I_100_coefficient_space", "w_ols_active": True}


def _load_train_role(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = _path(root, TRAINING_SOURCE_PATH)
    _require(source.is_file(), "canonical TRAIN source is unavailable")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28) and images.dtype == torch.float32, "TRAIN source image shape/dtype changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "TRAIN source label shape/dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    _require(train_indices.numel() == 55_000 and development_indices.numel() == 5_000, "TRAIN/DEV partition size changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV partition overlaps or omits rows")
    counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    return images, labels, train_indices, {"source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000}, "train": {"rows": 55_000, "prototype_rows": 55_000, "update_rows": PRECHECK_STEPS * LOGICAL_BATCH_SIZE, "counts_by_class": [int(value) for value in counts.tolist()], "indices_sha256": tensor_sha256(train_indices)}, "development": {"rows": 5_000, "values_consumed": 0, "status": "partitioned for role accounting only; no scientific DEV values inspected"}, "protected": {"rows_read": 0, "status": "forbidden; path not opened"}, "test": {"rows_read": 0, "status": "forbidden; path not opened"}, "sealed": {"rows_read": 0, "status": "forbidden and unavailable"}, "labels_as_model_input": False}


def _base_payload(started: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "preflight_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_PREFLIGHT_NOT_COMPLETED", "provenance": {"git_start": dict(started)}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_authorized": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"}, "next": None}


def _resolve_output(root: Path, output_dir: Path | None) -> Path:
    target = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "V5 preflight output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing V5 preflight output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_preflight(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = _resolve_output(root, output_dir)
    started = git_facts(root)
    payload = _base_payload(started)
    payload["entrypoint"] = {"path": Path(__file__).resolve().relative_to(root).as_posix(), "sha256": file_sha256(Path(__file__).resolve())}
    contract_sha: str | None = None
    authorities: Mapping[str, Any] | None = None
    input_identity: dict[str, Any] | None = None
    try:
        contract, authorities, contract_sha = _load_contract(root)
        v3, v4, v2_contract = _validate_authority_payloads(root, contract, authorities)
        initial_state = _combined_initial_state(v3, v4)
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "preflight_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": contract_sha}, "entrypoint": payload["entrypoint"], "protocol": {"production_grid": PRODUCTION_GRID, "smoke_grid": SMOKE_GRID, "logical_batch_size": LOGICAL_BATCH_SIZE, "preflight_updates_per_arm": PRECHECK_STEPS, "future_scientific_updates": SCIENTIFIC_STEPS, "checkpoints": [100, 200, 300], "control_objective": "L_control=L_image", "candidate_objective": "L_candidate=L_image+lambda_proto*L_proto", "prototype_scale": "median_{k<l} d(mu_k,mu_l)", "lambda": "g_image/(g_proto+1e-12) once before update", "optimizer": {"name": "Adam", "lr": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD}}, "claim_boundary": {"scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False}, "self_hash": "not_embedded_to_avoid_self_reference"}
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V5 semantic-alignment preflight input")
        input_identity = {"path": input_path.relative_to(root).as_posix(), "sha256": file_sha256(input_path)}
        payload["preflight_input"] = input_identity
        payload["authorities"] = authorities
        payload["authority_roles"] = {"initial_encoder": "V3 CONTROL step-300 encoder.phase", "frozen_generator": "valid V4 step-300 fixed-OLS generator.phase", "scientific_evaluator": "SmoothMNISTEvaluator v2.1 evaluation-only", "producer_head_policy": "producer HEAD recorded as provenance only; exact consumer equality not required"}
        payload["runtime_readiness"] = {"cuda_dll_search_path": _prepare_cuda_runtime_dll_search_path(), "scientific_data_accessed": False}
        device, device_facts = _cuda_facts()
        payload["device"] = device_facts
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        initial_rng = _capture_rng_state()
        payload["initial_state"] = {"combined_runtime_state_sha256": _digest_tree(initial_state), "encoder_phase_sha256": tensor_sha256(initial_state["encoder.phase"]), "generator_phase_sha256": tensor_sha256(initial_state["generator.phase"]), "ols_transport_sha256": tensor_sha256(initial_state["ols.operator.transport"]), "trainable_parameter": "encoder.phase only", "initial_rng_sha256": _digest_tree(initial_rng)}
        evaluator, evaluator_identity = _load_smooth(root, authorities)
        payload["evaluator"] = evaluator_identity
        payload["production_cuda_no_data_smoke"] = _production_no_data_smoke(v2_contract, initial_state, device)

        images, labels, train_indices, dataset = _load_train_role(root)
        payload["dataset"] = dataset
        prototype_consumer = _new_consumer(v2_contract, device, PRODUCTION_GRID)
        _load_runtime_state(prototype_consumer, initial_state)
        _configure_all_frozen(prototype_consumer)
        prototypes, prototype_summary = construct_prototypes(prototype_consumer, images, labels, train_indices, expected_rows=55_000)
        s_mu, scale_summary = prototype_scale(prototypes)
        payload["prototypes"] = prototype_summary
        payload["prototype_scale"] = scale_summary
        prototype_consumer = None
        first_batch_indices = DeterministicTrainRoleSampler(train_indices, labels, seed=1234, steps=SCIENTIFIC_STEPS).next_batch()
        first_batch_images = images.index_select(0, first_batch_indices[0])
        initial_rng_for_lambda = _capture_rng_state()
        calibration = _calibrate_lambda(v2_contract, first_batch_images, first_batch_indices[1], initial_state, initial_rng_for_lambda, prototypes, s_mu, device=device, grid=PRODUCTION_GRID)
        payload["lambda_calibration"] = calibration
        output_control = target / CONTROL_CHECKPOINT_NAME
        output_candidate = target / CANDIDATE_CHECKPOINT_NAME
        control = _run_arm(root, v2_contract, images, labels, train_indices, initial_state, initial_rng_for_lambda, prototypes, s_mu, 0.0, arm="control", device=device, grid=PRODUCTION_GRID, output_path=output_control, contract_sha=contract_sha)
        candidate = _run_arm(root, v2_contract, images, labels, train_indices, initial_state, initial_rng_for_lambda, prototypes, s_mu, calibration["lambda_proto"], arm="candidate", device=device, grid=PRODUCTION_GRID, output_path=output_candidate, contract_sha=contract_sha)
        payload["arms"] = {"control": {key: value for key, value in control.items() if key not in {"endpoint_raw", "final_runtime_state"}}, "candidate": {key: value for key, value in candidate.items() if key not in {"endpoint_raw", "final_runtime_state"}}}
        payload["matched_arms"] = {"initial_state_equal": control["initial_state_sha256"] == candidate["initial_state_sha256"], "initial_rng_equal": control["initial_rng_sha256"] == candidate["initial_rng_sha256"], "same_first_train_batch": control["traces"][0]["batch_indices_sha256"] == candidate["traces"][0]["batch_indices_sha256"], "same_optimizer": True, "generator_frozen_both_arms": bool(control["frozen_generator_unchanged"] and candidate["frozen_generator_unchanged"]), "ols_unchanged_both_arms": bool(control["ols_unchanged"] and candidate["ols_unchanged"])}
        _require(all(payload["matched_arms"].values()), "matched-arm identity/firewall check failed")
        payload["frozen_parameter_firewall"] = {"generator": bool(control["frozen_generator_unchanged"] and candidate["frozen_generator_unchanged"]), "ols": bool(control["ols_unchanged"] and candidate["ols_unchanged"]), "encoder_only": True, "smooth_evaluator": all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "prototype_gradients": {"candidate_g_proto_positive": calibration["g_proto"] > 0.0, "finite": math.isfinite(calibration["g_proto"]) and math.isfinite(calibration["g_image"])}}
        _require(all(value is True for value in payload["frozen_parameter_firewall"].values() if isinstance(value, bool)), "V5 frozen parameter firewall failed")
        payload["synthetic_acceptance"] = _run_synthetic_acceptance(root, v2_contract, initial_state, evaluator, contract_sha, target)
        payload["data_role_reads"] = {"TRAIN": {"rows_loaded": 60_000, "rows_used": 55_000, "prototype_rows": 55_000, "update_rows_per_arm": PRECHECK_STEPS * LOGICAL_BATCH_SIZE}, "DEV": {"scientific_values_read": 0, "used_for_selection": False}, "protected": 0, "test": 0, "sealed": 0}
        payload["claim_boundary"].update({"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_authorized": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload["result"] = "AUTHORIZABLE"
        payload["status"] = "valid_completed"
        payload["classification"] = SUCCESS_CLASSIFICATION
        payload["next"] = "explicit_independent_authorization_required_for_one_shot_300_step_v5_scientific_comparison"
        payload["next_note"] = "AUTHORIZABLE is engineering executability only; it is not V5 scientific support, Stage-A eligibility, Stage-B authorization, generation readiness, or formal promotion"
    except PreflightBlocked as blocked:
        payload.update(blocked.details)
        payload["result"] = "BLOCKED"
        payload["status"] = "valid_failed"
        payload["classification"] = blocked.classification
        payload["hard_blocker"] = {"stage": blocked.stage, "message": str(blocked)}
        payload["next"] = "stop; resolve the bounded preflight blocker before any future V5 scientific authorization"
    except InvalidPreflight as error:
        payload["result"] = "INVALID"
        payload["status"] = "invalid_harness"
        payload["classification"] = "INVALID_V5_PREFLIGHT_HARNESS"
        payload["invalid_input"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        payload["result"] = "INVALID"
        payload["status"] = "invalid_harness"
        payload["classification"] = "INVALID_V5_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR"
        payload["unexpected_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}
    finally:
        ending = git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_dirty": ending.get("worktree_dirty", False)})
        payload["claim_boundary"].update({"scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_authorized": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload["scientific_300_step_executed"] = False
        payload["forbidden_actions"] = {"stage_b": False, "formal_scale": False, "prior": False, "sampled_generation": False, "protected": False, "test": False, "sealed": False, "formal_promotion": False}
        payload["artifacts"] = {"preflight_input": input_identity, "control_checkpoint": payload.get("arms", {}).get("control", {}).get("checkpoint"), "candidate_checkpoint": payload.get("arms", {}).get("candidate", {}).get("checkpoint"), "synthetic_sufficient_observations": payload.get("synthetic_acceptance", {}).get("sufficient_observations") if isinstance(payload.get("synthetic_acceptance"), Mapping) else None, "report_self_hash": "not_embedded_to_avoid_self_reference"}

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V5 semantic-alignment preflight report")
    report_ref = {"path": report_path.relative_to(root).as_posix(), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": contract_sha}, "entrypoint": payload["entrypoint"], "authorities": authorities, "provenance": payload["provenance"], "artifacts": {"preflight_input": input_identity, "control_checkpoint": payload["artifacts"].get("control_checkpoint"), "candidate_checkpoint": payload["artifacts"].get("candidate_checkpoint"), "synthetic_sufficient_observations": payload["artifacts"].get("synthetic_sufficient_observations"), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V5 semantic-alignment preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "preflight_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": contract_sha}, "preflight_input": input_identity, "report": report_ref, "lineage": {"path": lineage_path.relative_to(root).as_posix(), "sha256": file_sha256(lineage_path)}, "lambda_calibration": payload.get("lambda_calibration"), "prototypes": payload.get("prototype_scale"), "three_step_traces": {name: value.get("traces") for name, value in payload.get("arms", {}).items()} if isinstance(payload.get("arms"), Mapping) else None, "synthetic_acceptance": payload.get("synthetic_acceptance"), "production_cuda_no_data_smoke": payload.get("production_cuda_no_data_smoke"), "data_role_reads": payload.get("data_role_reads"), "scientific_300_step_executed": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload["next"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight",), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_preflight(args.repo_root, args.output_dir)
    return 0 if result["result"] in {"AUTHORIZABLE", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
