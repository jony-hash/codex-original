"""Run the one-shot V6 prototype-only semantic warmup.

The script freezes and validates the V6 contract, performs the bounded
engineering preflight, and then runs exactly one 300-update encoder-only
prototype-loss experiment when no hard blocker is present.  Image MSE is
computed only for evaluation and is never part of the differentiated
objective.  The script never opens Stage B, scale/prior/generation paths, or
protected/test/sealed roles.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import gauge_invariant_ols_representation
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, LightGen2DComplexOLSTrainingConsumerV2, file_sha256, git_facts, load_v2_training_contract, tensor_sha256
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_local_mnist, stratified_mnist_indices


TASK_ID = "EXECUTE_LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_V6"
OUTER_GATE = "V6_PROTO_ONLY_SEMANTIC_WARMUP_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V6_PROTO_ONLY_SEMANTIC_WARMUP_NOT_SUPPORTED"
SECONDARY_LATENT_ONLY = "LATENT_ALIGNMENT_WITHOUT_NATURAL_SEMANTIC_RECOVERY"
SECONDARY_TRANSFER_DEGRADATION = "SEMANTIC_GAIN_WITH_TRANSFER_GEOMETRY_DEGRADATION"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"
CONTRACT_PATH = Path(
    "outputs/20260911_v6/runs/"
    "lightgen_2d_proto_only_semantic_warmup_contract_v6_attempt_01/"
    "lightgen_2d_proto_only_semantic_warmup_contract.v6.json"
)
CONTRACT_SHA256 = "4b849632b67bd68f4075816e061bb3db0664abe31ab757a7137bfcd4c8f9f441"

DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/20260911_v6/runs/"
    "lightgen_2d_proto_only_semantic_warmup_preflight_v6_attempt_03"
)
DEFAULT_SCIENTIFIC_DIR = Path(
    "outputs/20260911_v6/runs/"
    "lightgen_2d_proto_only_semantic_warmup_v6_scientific_attempt_01"
)

PREFLIGHT_INPUT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_preflight_lineage.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_synthetic_sufficient_observations.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_synthetic_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_proto_only_semantic_warmup_v6_step300_cyclic_raw_outputs.v1.pt"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
SMOKE_GRID = 128
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEPS = (0, 100, 200, 300)
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
SMOKE_BOOTSTRAP_REPLICATES = 9
STAGE_A_FIREWALL = 0.937
SMOOTH_REQUIRED_DELTA = 0.05
SELECTION_ROWS = 1000
SELECTION_PER_CLASS = 100
SELECTION_SEED = 4321
SPLIT_SEED = 1234
PROTOTYPE_SCALE = 0.00555040966719389
PROTOTYPE_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SELECTION_LABELS_SHA256 = "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
OLS_TRANSPORT_SHA256 = "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293"
V3_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V4_GENERATOR_TENSOR_SHA256 = "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000

V2_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
V3_CONTROL_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
V3_CONTROL_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"
V4_GENERATOR_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_v4_step_300_checkpoint.v1.pt"
)
V4_GENERATOR_FILE_SHA256 = "de80f2826419ca6ced28c8e1fc6d8804ef513493a49474e5997cfb4cb59bdf80"
V5_REPORT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1.json"
)
V5_REPORT_SHA256 = "41aeb6e5ae5d1736521b164f2c8e9267bc3687801ab99b7142692112f8d85ce8"
V5_PROTOTYPE_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_control_step_300_checkpoint.v1.pt"
)
V5_PROTOTYPE_SHA256 = "abf62784d73a3ef7a5d51143642c42baf1472ce8da20af3f3f7a8392af32549f"
SMOOTH_PATH = Path(
    "outputs/20260730_v2/runs/"
    "20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/"
    "smooth_mnist_evaluator_v21.pt"
)
SMOOTH_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
GAUGE_HELPER_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_complex_ols_training.py"
)
GAUGE_HELPER_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"
EVALUATOR_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
HISTORICAL_INPUT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/"
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1.json"
)
HISTORICAL_INPUT_SHA256 = "c43b333822cab7c5ebafb5e64cdce7a50f386e662388f071d9378c740ed04b1d"
HISTORICAL_RAW_PATH = HISTORICAL_INPUT_PATH.with_name(
    "lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1.pt"
)
HISTORICAL_RAW_SHA256 = "e985f6f030591a1aa38d33517d35da6c831806e121c9f73c3162395c23336d00"

REPORT_SCHEMA = "lightgen_2d_proto_only_semantic_warmup_v6_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_proto_only_semantic_warmup_v6_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_proto_only_semantic_warmup_v6_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_proto_only_semantic_warmup_v6_sufficient_observations.v1"


class InvalidV6(RuntimeError):
    """The V6 harness or an immutable authority is invalid."""


class HardBlocked(RuntimeError):
    """A valid hard boundary prevents execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidV6(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute path is not allowed: {value}")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidV6(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


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


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _digest_tree(value: Any) -> str:
    import hashlib

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
    _require(torch.is_tensor(value) and torch.is_complex(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result), f"{name} norm is nonfinite")
    return result


def _status_lines(root: Path) -> list[str]:
    import subprocess

    result = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.splitlines()


def _capture_rng() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().cpu().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    _require(torch.is_tensor(state.get("cpu")) and isinstance(state.get("cuda"), list), "RNG state is malformed")
    torch.random.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _prepare_cuda_runtime_dll_search_path() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_process_only"}
    cuda_build = torch.version.cuda
    if not isinstance(cuda_build, str) or len(cuda_build.split(".")) != 2:
        raise HardBlocked("V6_CUDA_RUNTIME_BLOCKED", "cuda_dll_search_path", "PyTorch CUDA build identity is unavailable")
    major, minor = cuda_build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    if not environment_bin.is_dir() or not dll_path.is_file():
        raise HardBlocked("V6_CUDA_RUNTIME_BLOCKED", "cuda_dll_search_path", "required CUDA runtime DLL is unavailable", expected=str(dll_path))
    current = os.environ.get("PATH", "")
    normalized = os.path.normcase(os.path.normpath(str(environment_bin)))
    entries = [entry for entry in current.split(os.pathsep) if entry]
    present = any(os.path.normcase(os.path.normpath(entry)) == normalized for entry in entries)
    if not present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + current if current else "")
    return {"status": "already_present" if present else "prepended", "scope": "current_process_only", "cuda_build": cuda_build, "environment_bin": str(environment_bin), "expected_dll": dll_path.name}


def _cuda_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise HardBlocked("V6_CUDA_UNAVAILABLE", "device", "CUDA:0 is unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    probe = torch.fft.fft2(torch.ones((8, 8), dtype=torch.complex64, device=device))
    torch.cuda.synchronize(device)
    _finite_complex(probe, "CUDA complex probe")
    capability = (int(properties.major), int(properties.minor))
    _require(capability == EXPECTED_COMPUTE_CAPABILITY, f"CUDA compute capability changed: {capability}")
    _require(int(properties.total_memory) >= EXPECTED_VRAM_BYTES, "CUDA VRAM is below the frozen preflight target")
    return device, {"device": str(device), "device_count": int(torch.cuda.device_count()), "gpu": properties.name, "compute_capability": list(capability), "vram_bytes": int(properties.total_memory), "torch_version": torch.__version__, "pytorch_cuda_build": torch.version.cuda, "grid": PRODUCTION_GRID}


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    path = _path(root, CONTRACT_PATH)
    _validate_hash(path, CONTRACT_SHA256, "V6 contract")
    payload = _read_json(path, "V6 contract")
    _require(payload.get("schema") == "lightgen_2d_proto_only_semantic_warmup_contract.v6", "V6 contract schema changed")
    _require(payload.get("result") == "READY" and payload.get("status") == "valid_completed", "V6 contract is not READY")
    _require(payload.get("outer_gate") == OUTER_GATE, "V6 outer gate changed")
    definition = _mapping(payload.get("definition_entrypoint"), "V6 definition entrypoint")
    definition_path = _path(root, definition.get("path"))
    _validate_hash(definition_path, str(definition.get("sha256")), "V6 definition entrypoint")
    git = _mapping(payload.get("git"), "V6 contract Git")
    _require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V6 contract Git identity changed")
    v2_contract = load_v2_training_contract(_path(root, V2_CONTRACT_PATH), repo_root=root)
    _require(v2_contract.sha256 == V2_CONTRACT_SHA256, "V2 contract identity changed")
    return payload, v2_contract


def _load_ref(root: Path, relative: str | Path, expected: str, name: str) -> Any:
    path = _path(root, relative)
    return safe_torch_load(path, expected_sha256=expected, map_location="cpu")


def _resolve_authorities(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    authorities = _mapping(contract.get("authorities"), "V6 authorities")
    source = _mapping(authorities.get("source"), "V6 source authorities")
    for key, expected, name in (
        ("training_source", TRAINING_SOURCE_SHA256, "TRAIN source"),
        ("aligned_runtime", RUNTIME_SHA256, "aligned runtime"),
        ("gauge_helper", GAUGE_HELPER_SHA256, "gauge helper"),
        ("evaluator_implementation", EVALUATOR_SHA256, "evaluator implementation"),
    ):
        ref = _mapping(source.get(key), f"{key} authority")
        _validate_hash(_path(root, ref["path"]), expected, name)
        _require(ref.get("sha256") == expected, f"{name} contract SHA changed")
    v3_ref = _mapping(_mapping(authorities.get("v3_control"), "V3 authority").get("checkpoint"), "V3 checkpoint")
    v4_ref = _mapping(_mapping(authorities.get("v4_generator"), "V4 authority").get("checkpoint"), "V4 checkpoint")
    v5_ref = _mapping(_mapping(authorities.get("v5_prototypes"), "V5 authority").get("source_checkpoint"), "V5 prototype checkpoint")
    v3 = _mapping(_load_ref(root, v3_ref["path"], str(v3_ref["sha256"]), "V3 checkpoint"), "V3 checkpoint")
    v4 = _mapping(_load_ref(root, v4_ref["path"], str(v4_ref["sha256"]), "V4 checkpoint"), "V4 checkpoint")
    v5 = _mapping(_load_ref(root, v5_ref["path"], str(v5_ref["sha256"]), "V5 prototype checkpoint"), "V5 prototype checkpoint")
    v3_state = _mapping(v3.get("runtime_state"), "V3 runtime state")
    v4_state = _mapping(v4.get("runtime_state"), "V4 runtime state")
    _require(v3.get("arm") == "control" and v3.get("checkpoint_step") == 300, "V3 CONTROL checkpoint role changed")
    _require(v4.get("checkpoint_step") == 300 and v4.get("trainable_parameter_names") == ["generator.phase"], "V4 generator checkpoint role changed")
    _require(tensor_sha256(v3_state["encoder.phase"]) == V3_ENCODER_SHA256, "V3 initial encoder changed")
    _require(tensor_sha256(v4_state["generator.phase"]) == V4_GENERATOR_TENSOR_SHA256, "V4 generator tensor changed")
    _require(tensor_sha256(v4_state["encoder.phase"]) == V3_ENCODER_SHA256, "V4 encoder no longer matches V3 CONTROL")
    _require(torch.equal(v3_state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V3 OLS transport changed")
    combined = {key: _cpu_tree(value) for key, value in v3_state.items()}
    combined["generator.phase"] = _cpu_tree(v4_state["generator.phase"])
    _require(tensor_sha256(combined["encoder.phase"]) == V3_ENCODER_SHA256, "combined V6 encoder changed")
    _require(tensor_sha256(combined["generator.phase"]) == V4_GENERATOR_TENSOR_SHA256, "combined V6 generator changed")
    _require(torch.equal(combined["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "combined V6 transport changed")
    prototypes = v5.get("prototypes")
    _require(v5.get("arm") == "control" and torch.is_tensor(prototypes), "V5 prototype source role changed")
    _require(tensor_sha256(prototypes) == PROTOTYPE_SHA256 and float(v5.get("s_mu")) == PROTOTYPE_SCALE, "V5 frozen prototypes changed")
    historical = _mapping(contract.get("authorities", {}).get("historical_class_direction"), "historical authority")
    historical_raw_ref = _mapping(historical.get("raw_templates"), "historical raw templates")
    historical_raw = _mapping(_load_ref(root, historical_raw_ref["path"], str(historical_raw_ref["sha256"]), "historical raw artifact"), "historical raw artifact")
    templates = historical_raw.get("templates")
    _require(torch.is_tensor(templates) and tuple(templates.shape) == (CLASS_COUNT, 1, 28, 28) and templates.dtype == torch.float32, "historical templates changed")
    _finite(templates, "historical templates")
    smooth_ref = _mapping(authorities.get("smooth_evaluator"), "Smooth evaluator authority")
    smooth_payload = _mapping(_load_ref(root, smooth_ref["path"], str(smooth_ref["sha256"]), "Smooth checkpoint"), "Smooth checkpoint")
    _require(smooth_payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth architecture changed")
    evaluator = SmoothMNISTEvaluator()
    evaluator.load_state_dict(_mapping(smooth_payload.get("state_dict"), "Smooth state"), strict=True)
    evaluator.freeze()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator firewall failed")
    return {"authorities": dict(authorities), "v2_contract": load_v2_training_contract(_path(root, V2_CONTRACT_PATH), repo_root=root), "initial_state": combined, "prototypes": prototypes.detach().cpu().contiguous(), "templates": templates.detach().cpu().contiguous(), "smooth": evaluator, "smooth_identity": {"path": smooth_ref["path"], "sha256": smooth_ref["sha256"], "architecture_kind": smooth_payload.get("architecture_kind"), "state_dict_tree_sha256": _digest_tree(_cpu_tree(evaluator.state_dict())), "gradient_enabled": False}}


def _load_roles(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = _path(root, TRAINING_SOURCE_PATH)
    _validate_hash(source, TRAINING_SOURCE_SHA256, "canonical TRAIN source")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28) and images.dtype == torch.float32, "MNIST image role changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "MNIST label role changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SPLIT_SEED, validation_per_label=500)
    _require(train_indices.numel() == 55_000 and development_indices.numel() == 5_000, "TRAIN/DEV partition changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV partition overlaps")
    development_labels = labels.index_select(0, development_indices)
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    selected: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, f"DEV class {digit} count changed")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:SELECTION_PER_CLASS]))
    selection_indices = torch.cat(selected).contiguous()
    selection_labels = labels.index_select(0, selection_indices).contiguous()
    _require(tensor_sha256(selection_indices) == SELECTION_INDICES_SHA256, "balanced DEV selection identity changed")
    _require(tensor_sha256(selection_labels) == SELECTION_LABELS_SHA256, "balanced DEV labels identity changed")
    return images, labels, train_indices, development_indices, {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000},
        "TRAIN": {"rows": 55_000, "indices_sha256": tensor_sha256(train_indices), "scientific_values_used_for": "V6 prototype-only optimization"},
        "DEV": {"rows": 5_000, "indices_sha256": tensor_sha256(development_indices), "selection_rows": 1_000, "selection_per_class": 100, "selection_seed": SELECTION_SEED, "selection_indices_sha256": SELECTION_INDICES_SHA256, "selection_labels_sha256": SELECTION_LABELS_SHA256},
        "protected": 0,
        "official_test": 0,
        "sealed": 0,
        "labels_as_model_input": False,
    }


def _new_consumer(v2_contract: Any, device: torch.device, grid: int) -> LightGen2DComplexOLSTrainingConsumerV2:
    return LightGen2DComplexOLSTrainingConsumerV2(v2_contract, device=device, propagation_grid_size=grid)


def _load_runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2, state: Mapping[str, Any]) -> None:
    consumer.runtime.load_state_dict(dict(state), strict=True)
    consumer._enforce_parameter_firewall()


def _configure_encoder_only(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> dict[str, Any]:
    named = dict(consumer.runtime.named_parameters())
    _require(set(named) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name == "encoder.phase")
        parameter.grad = None
    _require(named["encoder.phase"].requires_grad and not named["generator.phase"].requires_grad, "encoder-only firewall failed")
    consumer.runtime.train()
    return {"trainable": ["encoder.phase"], "frozen": ["generator.phase"], "trainable_count": int(named["encoder.phase"].numel()), "frozen_count": int(named["generator.phase"].numel())}


def _configure_all_frozen(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> None:
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    consumer.runtime.eval()


def _validate_result(result: Any, batch_size: int, *, require_grad: bool) -> torch.Tensor:
    _require(result.image_input is not None and result.ols_input_field is not None and result.z_ols.source == "real_encoded", "real image path lost encoder/OLS boundary")
    z = result.z_ols.coefficients
    _require(tuple(z.shape) == (batch_size, OLS_PORT_COUNT) and z.dtype == torch.complex64, "OLS coefficient shape/dtype changed")
    if require_grad:
        _require(z.requires_grad, "OLS coefficients detached from encoder")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "modal projection boundary changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "T=I_100 provenance changed")
    raw = result.raw_intensity
    target = result.image_input.slm_intensity
    _require(tuple(raw.shape) == (batch_size, 1, 200, 200) and raw.dtype == torch.float32, "raw CMOS shape/dtype changed")
    _require(tuple(target.shape) == tuple(raw.shape) and target.dtype == torch.float32, "image target shape/dtype changed")
    _finite(raw, "raw CMOS")
    return raw


def _raw_accumulator() -> dict[str, Any]:
    return {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": math.inf, "max": -math.inf, "per_image_variance_min": math.inf, "per_image_variance_max": -math.inf, "all_finite": True, "structured": True}


def _accumulate_raw(summary: dict[str, Any], raw: torch.Tensor) -> None:
    value = raw.detach().to(torch.float64).cpu()
    _finite(value, "raw evaluation")
    flat = value.flatten(1)
    variance = flat.var(dim=1, unbiased=False)
    summary["count"] += int(value.numel())
    summary["sum"] += float(value.sum())
    summary["sum_sq"] += float(value.square().sum())
    summary["min"] = min(summary["min"], float(value.min()))
    summary["max"] = max(summary["max"], float(value.max()))
    summary["per_image_variance_min"] = min(summary["per_image_variance_min"], float(variance.min()))
    summary["per_image_variance_max"] = max(summary["per_image_variance_max"], float(variance.max()))
    summary["all_finite"] = bool(summary["all_finite"] and torch.isfinite(value).all())
    summary["structured"] = bool(summary["structured"] and (variance > 0.0).all())


def _finish_raw_summary(summary: Mapping[str, Any], rows: int) -> dict[str, Any]:
    count = int(summary["count"])
    mean = float(summary["sum"] / count)
    variance = max(0.0, float(summary["sum_sq"] / count - mean * mean))
    result = {"shape": [rows, 1, 200, 200], "dtype": "torch.float32", "finite": bool(summary["all_finite"]), "structured": bool(summary["structured"]), "nonconstant": bool(variance > 0.0 and summary["max"] > summary["min"]), "mean": mean, "std": math.sqrt(variance), "min": float(summary["min"]), "max": float(summary["max"]), "per_image_variance_min": float(summary["per_image_variance_min"]), "per_image_variance_max": float(summary["per_image_variance_max"])}
    _require(result["finite"] and result["structured"] and result["nonconstant"], "raw CMOS validity failed")
    return result


def _prototype_distance(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    _require(z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT and torch.is_complex(z), "query z shape/domain changed")
    query = gauge_invariant_ols_representation(z)
    support = gauge_invariant_ols_representation(prototypes)
    distance = (query[:, None, :] - support[None, :, :]).square().sum(dim=2)
    _finite(distance, "prototype distance")
    return distance


def _prototype_logits(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    logits = -_prototype_distance(z, prototypes) / (PROTOTYPE_SCALE + 1.0e-12)
    _finite(logits, "prototype logits")
    return logits


def _prototype_loss(z: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    logits = _prototype_logits(z, prototypes)
    loss = F.cross_entropy(logits, labels)
    _finite(loss, "prototype loss")
    return loss


def _ols_margin(z: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    distance = _prototype_distance(z, prototypes)
    rows = torch.arange(z.shape[0], device=z.device)
    correct = distance[rows, labels]
    other = distance.clone()
    other[rows, labels] = torch.inf
    margin = other.min(dim=1).values - correct
    _finite(margin, "OLS mechanism margin")
    return margin.detach().cpu().to(torch.float64)


def _smooth_logits(evaluator: SmoothMNISTEvaluator, raw: torch.Tensor) -> torch.Tensor:
    view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
    with torch.no_grad():
        _, logits = evaluator(view)
    _finite(logits, "Smooth logits")
    _require(tuple(logits.shape) == (raw.shape[0], CLASS_COUNT), "Smooth logits shape changed")
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator firewall changed")
    return logits.detach().cpu().contiguous()


def _optimizer(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> torch.optim.Adam:
    parameter = dict(consumer.runtime.named_parameters())["encoder.phase"]
    _require(parameter.requires_grad, "encoder.phase is not trainable")
    optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == [parameter], "optimizer firewall changed")
    return optimizer


def _frozen_state(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> dict[str, Any]:
    return {key: _cpu_tree(value) for key, value in consumer.runtime.state_dict().items() if key != "encoder.phase"}


def _train_step(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    step: int,
) -> dict[str, Any]:
    named = dict(consumer.runtime.named_parameters())
    before_frozen = _frozen_state(consumer)
    before_encoder = named["encoder.phase"].detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    for parameter in named.values():
        parameter.grad = None
    result = consumer.runtime.forward_from_image(batch)
    raw = _validate_result(result, int(batch.shape[0]), require_grad=True)
    with torch.no_grad():
        image_loss = (raw.detach() - result.image_input.slm_intensity.detach()).square().mean()
    _finite(image_loss, "evaluation-only image loss")
    proto_loss = _prototype_loss(result.z_ols.coefficients, labels, prototypes)
    total_loss = proto_loss
    _require(total_loss is proto_loss, "V6 objective unexpectedly contains an image term")
    _finite(total_loss, "V6 total loss")
    _require(not image_loss.requires_grad, "image loss is not detached from V6 objective")
    total_loss.backward()
    encoder_gradient = named["encoder.phase"].grad
    _require(encoder_gradient is not None, "encoder prototype gradient missing")
    gradient_norm = _norm(encoder_gradient, "encoder prototype gradient")
    _require(gradient_norm > 0.0, "encoder prototype gradient is zero")
    _require(named["generator.phase"].grad is None, "frozen generator received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    after_frozen = _frozen_state(consumer)
    after_encoder = named["encoder.phase"].detach().cpu().clone()
    update_norm = float((after_encoder - before_encoder).to(torch.float64).square().sum().sqrt())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "encoder update is zero or nonfinite")
    _require(_tree_equal(before_frozen, after_frozen), "frozen generator/OLS state changed")
    return {
        "step": step,
        "objective": "L_V6=L_proto",
        "L_proto": float(proto_loss.detach().cpu()),
        "L_image_evaluation_only": float(image_loss.detach().cpu()),
        "L_total": float(total_loss.detach().cpu()),
        "gradient_norms": {"encoder.phase": gradient_norm, "generator.phase": None, "L_image_contribution": 0.0},
        "update_norms": {"encoder.phase": update_norm, "generator.phase": 0.0},
        "image_loss_in_differentiated_objective": False,
        "generator_unchanged": True,
        "ols_unchanged": True,
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase"],
        "runtime_state_sha256_after": _digest_tree(_cpu_tree(consumer.runtime.state_dict())),
    }


def _evaluate(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    evaluator: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    step: int,
    train_trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    raw_summary = _raw_accumulator()
    smooth_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    image_sum = 0.0
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
                batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
                batch_labels = labels[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
                result = consumer.runtime.forward_from_image(batch)
                raw = _validate_result(result, int(batch.shape[0]), require_grad=False)
                _accumulate_raw(raw_summary, raw)
                image_loss = (raw - result.image_input.slm_intensity).square().mean()
                _finite(image_loss, "DEV image loss")
                smooth_parts.append(_smooth_logits(evaluator, raw))
                margin_parts.append(_ols_margin(result.z_ols.coefficients, batch_labels, prototypes))
                image_sum += float(image_loss.cpu()) * int(batch.shape[0])
    finally:
        consumer.runtime.train(was_training)
    smooth_logits = torch.cat(smooth_parts, dim=0)
    margins = torch.cat(margin_parts, dim=0)
    _require(tuple(smooth_logits.shape) == (images.shape[0], CLASS_COUNT), "DEV Smooth observation shape changed")
    _require(tuple(margins.shape) == (images.shape[0],), "DEV OLS observation shape changed")
    predictions = smooth_logits.argmax(dim=1)
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        _require(bool(mask.any()), f"DEV class {digit} absent")
        per_class.append(float((predictions[mask] == digit).to(torch.float64).mean()))
    macro = float(sum(per_class) / CLASS_COUNT)
    return {"step": step, "losses": {"L_image": image_sum / float(images.shape[0]), "role": "evaluation-only"}, "smooth": {"macro_recall": macro, "per_class_recall": per_class}, "ols_margin": {"mean": float(margins.mean()), "median": float(margins.median()), "positive_fraction": float((margins > 0.0).to(torch.float64).mean())}, "raw_validity": _finish_raw_summary(raw_summary, int(images.shape[0])), "train_trace": dict(train_trace) if train_trace is not None else None, "_smooth_logits": smooth_logits, "_margins": margins}


def _checkpoint_payload(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Adam,
    sampler: Any,
    *,
    step: int,
    initial_state_digest: str,
    prototypes: torch.Tensor,
    rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_proto_only_semantic_warmup_v6_checkpoint.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "checkpoint_step": step,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(rng_state),
        "initial_runtime_state_sha256": initial_state_digest,
        "prototypes_sha256": tensor_sha256(prototypes),
        "s_mu": PROTOTYPE_SCALE,
        "objective": "L_V6=L_proto",
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase", "OLS", "T=I_100", "W_OLS"],
        "image_loss_gradient_contribution": 0.0,
        "data_roles": {"TRAIN": True, "DEV": True, "protected": False, "official_test": False, "sealed": False},
        "stage_b_executed": False,
        "scale_extraction_executed": False,
        "prior_shaping_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
    }


def _save_and_verify_checkpoint(root: Path, target: Path, payload: Mapping[str, Any], name: str, consumer: LightGen2DComplexOLSTrainingConsumerV2, v2_contract: Any) -> dict[str, Any]:
    path = target / name
    atomic_torch_save_new(path, payload, artifact_kind="V6 checkpoint")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V6 checkpoint reload")
    _require(loaded.get("schema") == payload.get("schema") and loaded.get("checkpoint_step") == payload.get("checkpoint_step"), "checkpoint schema/step reload changed")
    _require(loaded.get("prototypes_sha256") == PROTOTYPE_SHA256 and loaded.get("s_mu") == PROTOTYPE_SCALE, "checkpoint prototype identity changed")
    clone = _new_consumer(v2_contract, consumer.device, consumer.runtime.config.propagation_grid_size)
    _load_runtime_state(clone, _mapping(loaded.get("runtime_state"), "checkpoint runtime state"))
    _configure_encoder_only(clone)
    _require(_digest_tree(_cpu_tree(clone.runtime.state_dict())) == _digest_tree(_cpu_tree(loaded["runtime_state"])), "checkpoint runtime reload changed")
    clone_optimizer = _optimizer(clone)
    clone_optimizer.load_state_dict(dict(_mapping(loaded.get("optimizer_state"), "checkpoint optimizer state")))
    _require(_tree_equal(_cpu_tree(clone_optimizer.state_dict()), loaded["optimizer_state"]), "checkpoint optimizer reload changed")
    current_rng = _capture_rng()
    _restore_rng(_mapping(loaded.get("rng_state"), "checkpoint RNG state"))
    reloaded_rng = _capture_rng()
    _restore_rng(current_rng)
    _require(_tree_equal(reloaded_rng, loaded["rng_state"]), "checkpoint RNG reload changed")
    return {"path": _relative(root, path), "sha256": digest, "step": int(payload["checkpoint_step"]), "reloaded": True, "resume_identity": True}


def _summary(values: torch.Tensor) -> dict[str, Any]:
    value = values.detach().cpu().to(torch.float64)
    _finite(value, "summary values")
    return {"mean": float(value.mean()), "median": float(value.median()), "min": float(value.min()), "max": float(value.max()), "positive_fraction": float((value > 0.0).to(torch.float64).mean()), "count": int(value.numel())}


def _paired_bootstrap(values: torch.Tensor, labels: torch.Tensor, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    _require(values.ndim == 1 and labels.shape == values.shape, "bootstrap inputs are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_class: list[torch.Tensor] = []
    raw_by_class: list[dict[str, Any]] = []
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() > 0, f"bootstrap class {digit} absent")
        class_values = values.index_select(0, matches).to(torch.float64)
        draw = torch.randint(class_values.numel(), (replicates, class_values.numel()), generator=generator)
        by_class.append(class_values[draw].mean(dim=1))
        raw_by_class.append({"class": digit, "count": int(class_values.numel()), "mean": float(class_values.mean()), "positive_fraction": float((class_values > 0.0).to(torch.float64).mean())})
    replicate_values = torch.stack(by_class).mean(dim=0)
    ci = torch.quantile(replicate_values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    observed = _summary(values)
    return {"observed": observed, "mean": float(replicate_values.mean()), "median": float(replicate_values.median()), "ci_lower": float(ci[0]), "ci_upper": float(ci[1]), "p_positive": float((replicate_values > 0.0).to(torch.float64).mean()), "replicates": int(replicates), "seed": int(seed), "per_class_observed": raw_by_class}


def _target_labels(labels: torch.Tensor) -> torch.Tensor:
    return (labels + 1) % CLASS_COUNT


def _template_margins(raw_base: torch.Tensor, raw_shifted: torch.Tensor, templates: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_view = F.adaptive_avg_pool2d(raw_base.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
    shifted_view = F.adaptive_avg_pool2d(raw_shifted.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
    source_templates = templates.index_select(0, labels).to(torch.float64)
    target_templates = templates.index_select(0, targets).to(torch.float64)
    base_margin = (base_view - source_templates).square().flatten(1).mean(dim=1) - (base_view - target_templates).square().flatten(1).mean(dim=1)
    shifted_margin = (shifted_view - source_templates).square().flatten(1).mean(dim=1) - (shifted_view - target_templates).square().flatten(1).mean(dim=1)
    _finite(base_margin, "template base margin")
    _finite(shifted_margin, "template shifted margin")
    return base_margin, shifted_margin


def _smooth_transfer_margins(base_logits: torch.Tensor, shifted_logits: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(labels.shape[0])
    base = base_logits[rows, targets] - base_logits[rows, labels]
    shifted = shifted_logits[rows, targets] - shifted_logits[rows, labels]
    _finite(base, "Smooth base transfer margin")
    _finite(shifted, "Smooth shifted transfer margin")
    return base.to(torch.float64), shifted.to(torch.float64)


def _cyclic_transfer(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    evaluator: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    selection_indices: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
) -> dict[str, Any]:
    consumer.runtime.eval()
    rows = int(images.shape[0])
    _require(rows > 0 and rows % CLASS_COUNT == 0 and labels.shape == (rows,), "cyclic selection shape is invalid")
    samples_per_class = rows // CLASS_COUNT
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), samples_per_class, dtype=torch.long)), "cyclic selection is not balanced")
    targets = _target_labels(labels)
    z_parts: list[torch.Tensor] = []
    raw_base_parts: list[torch.Tensor] = []
    raw_shift_parts: list[torch.Tensor] = []
    smooth_base_parts: list[torch.Tensor] = []
    smooth_shift_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            batch_labels = labels[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            batch_targets = targets[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            result = consumer.runtime.forward_from_image(batch)
            raw_base = _validate_result(result, int(batch.shape[0]), require_grad=False)
            z = result.z_ols.coefficients.detach()
            z_shift = z - prototypes.to(consumer.device).index_select(0, batch_labels) + prototypes.to(consumer.device).index_select(0, batch_targets)
            shifted_result = consumer.runtime.forward_from_ols(z_shift)
            raw_shift = shifted_result.raw_intensity
            _require(tuple(raw_shift.shape) == tuple(raw_base.shape) and raw_shift.dtype == torch.float32, "shifted raw shape/dtype changed")
            _finite(raw_shift, "shifted raw")
            z_parts.append(z.cpu().contiguous())
            raw_base_parts.append(raw_base.cpu().contiguous())
            raw_shift_parts.append(raw_shift.cpu().contiguous())
            smooth_base_parts.append(_smooth_logits(evaluator, raw_base))
            smooth_shift_parts.append(_smooth_logits(evaluator, raw_shift))
    z_base = torch.cat(z_parts, dim=0)
    raw_base = torch.cat(raw_base_parts, dim=0)
    raw_shift = torch.cat(raw_shift_parts, dim=0)
    smooth_base = torch.cat(smooth_base_parts, dim=0)
    smooth_shift = torch.cat(smooth_shift_parts, dim=0)
    _require(tuple(z_base.shape) == (rows, OLS_PORT_COUNT), "cyclic z shape changed")
    template_base, template_shift = _template_margins(raw_base, raw_shift, templates, labels, targets)
    smooth_base_margin, smooth_shift_margin = _smooth_transfer_margins(smooth_base, smooth_shift, labels, targets)
    delta_template = template_shift - template_base
    delta_smooth = smooth_shift_margin - smooth_base_margin
    template_bootstrap = _paired_bootstrap(delta_template, labels)
    smooth_bootstrap = _paired_bootstrap(delta_smooth, labels)
    pair_summaries = []
    positive_template_pairs = 0
    positive_smooth_pairs = 0
    for source in range(CLASS_COUNT):
        mask = labels == source
        t_values = delta_template[mask]
        s_values = delta_smooth[mask]
        t_mean = float(t_values.mean())
        s_mean = float(s_values.mean())
        positive_template_pairs += int(t_mean > 0.0)
        positive_smooth_pairs += int(s_mean > 0.0)
        pair_summaries.append({"source": source, "target": (source + 1) % CLASS_COUNT, "count": int(mask.sum()), "template_delta": _summary(t_values), "smooth_delta": _summary(s_values)})
    broad = positive_template_pairs >= 2 and positive_smooth_pairs >= 2
    template_pass = template_bootstrap["observed"]["mean"] > 0.0 and template_bootstrap["ci_lower"] > 0.0
    smooth_pass = smooth_bootstrap["observed"]["mean"] > 0.0 and smooth_bootstrap["ci_lower"] > 0.0
    return {
        "selection": {"rows": rows, "per_class": samples_per_class, "seed": SELECTION_SEED, "indices_sha256": tensor_sha256(selection_indices), "labels_sha256": tensor_sha256(labels), "target_rule": "t=(y+1) mod 10", "target_labels_sha256": tensor_sha256(targets)},
        "intervention": "z_shift=z-mu_y+mu_t; t=(y+1) mod 10",
        "template": {"definition": "M_template=d(raw,tau_y)-d(raw,tau_t); Delta=M_shift-M_base", "distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28", "bootstrap": template_bootstrap, "pass": bool(template_pass)},
        "smooth": {"definition": "M_smooth=score_t-score_y; Delta=M_shift-M_base using frozen SmoothMNISTEvaluator v2.1", "bootstrap": smooth_bootstrap, "pass": bool(smooth_pass)},
        "pair_summaries": pair_summaries,
        "positive_pair_counts": {"template": positive_template_pairs, "smooth": positive_smooth_pairs},
        "broad": bool(broad),
        "pair_dominated": bool(positive_template_pairs == 1 or positive_smooth_pairs == 1),
        "known_cyclic_transfer_retained": bool(template_pass and smooth_pass and broad),
        "raw_validity": {"base": _finish_raw_summary(_raw_summary_from_tensor(raw_base), rows), "shifted": _finish_raw_summary(_raw_summary_from_tensor(raw_shift), rows)},
        "_observations": {"selection_indices": selection_indices.cpu(), "labels": labels.cpu(), "targets": targets.cpu(), "z_base": z_base, "z_shift": z_base - prototypes.index_select(0, labels) + prototypes.index_select(0, targets), "raw_base": raw_base, "raw_shifted": raw_shift, "smooth_base_logits": smooth_base, "smooth_shifted_logits": smooth_shift, "template_base_margin": template_base, "template_shifted_margin": template_shift, "smooth_base_margin": smooth_base_margin, "smooth_shifted_margin": smooth_shift_margin},
    }


def _raw_summary_from_tensor(raw: torch.Tensor) -> dict[str, Any]:
    result = _raw_accumulator()
    for start in range(0, raw.shape[0], LOGICAL_BATCH_SIZE):
        _accumulate_raw(result, raw[start : start + LOGICAL_BATCH_SIZE])
    return result


def _save_sufficient(root: Path, target: Path, observations: Mapping[str, Any], *, name: str, schema: str) -> dict[str, Any]:
    path = target / name
    payload = {"schema": schema, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    atomic_torch_save_new(path, payload, artifact_kind="V6 sufficient observations")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V6 sufficient observations reload")
    _require(loaded.get("schema") == schema, "sufficient observation schema reload changed")
    for key, value in observations.items():
        if torch.is_tensor(value):
            _require(torch.equal(loaded[key], value.cpu()), f"sufficient observation reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": schema, "reloaded": True}


def _classify(ols_bootstrap: Mapping[str, Any], smooth_bootstrap: Mapping[str, Any], cyclic: Mapping[str, Any], raw_valid: bool) -> dict[str, Any]:
    ols_pass = ols_bootstrap["observed"]["mean"] > 0.0 and ols_bootstrap["ci_lower"] > 0.0
    smooth_pass = smooth_bootstrap["observed"]["mean"] >= SMOOTH_REQUIRED_DELTA and smooth_bootstrap["ci_lower"] > 0.0
    cyclic_pass = bool(cyclic["known_cyclic_transfer_retained"])
    supported = bool(ols_pass and smooth_pass and cyclic_pass and raw_valid)
    secondary = None
    if ols_pass and cyclic_pass and not smooth_pass:
        secondary = SECONDARY_LATENT_ONLY
    elif smooth_pass and not cyclic_pass:
        secondary = SECONDARY_TRANSFER_DEGRADATION
    return {"classification": SUPPORTED_CLASSIFICATION if supported else NOT_SUPPORTED_CLASSIFICATION, "ols_mechanism_pass": bool(ols_pass), "natural_smooth_pass": bool(smooth_pass), "known_cyclic_transfer_retained": cyclic_pass, "raw_validity_pass": bool(raw_valid), "secondary_interpretation": secondary}


def _load_selection(root: Path, images: torch.Tensor, labels: torch.Tensor, development_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    development_labels = labels.index_select(0, development_indices)
    generator = torch.Generator(device="cpu").manual_seed(SELECTION_SEED)
    selected = []
    for digit in range(CLASS_COUNT):
        matches = development_indices[development_labels == digit]
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:SELECTION_PER_CLASS]))
    selection_indices = torch.cat(selected).contiguous()
    selection_labels = labels.index_select(0, selection_indices).contiguous()
    _require(tensor_sha256(selection_indices) == SELECTION_INDICES_SHA256 and tensor_sha256(selection_labels) == SELECTION_LABELS_SHA256, "exact cyclic-transfer selection could not be reproduced")
    return images.index_select(0, selection_indices).contiguous(), selection_labels, selection_indices


def _initial_rng() -> dict[str, Any]:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    return _capture_rng()


def _production_no_data_smoke(bundle: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    _configure_all_frozen(consumer)
    before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    synthetic = torch.linspace(0.0, 1.0, LOGICAL_BATCH_SIZE * 28 * 28, device=device, dtype=torch.float32).reshape(LOGICAL_BATCH_SIZE, 1, 28, 28)
    with torch.no_grad():
        result = consumer.runtime.forward_from_image(synthetic)
    raw = _validate_result(result, LOGICAL_BATCH_SIZE, require_grad=False)
    after = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    _require(before == after, "production no-data smoke changed runtime state")
    return {"executed": True, "grid": PRODUCTION_GRID, "raw_validity": _finish_raw_summary(_raw_summary_from_tensor(raw), LOGICAL_BATCH_SIZE), "runtime_unchanged": True}


def _three_step_smoke(root: Path, bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, target: Path) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    firewall = _configure_encoder_only(consumer)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_encoder_only(consumer)
    optimizer = _optimizer(consumer)
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(_train_step(consumer, optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), step))
    payload = _checkpoint_payload(consumer, optimizer, sampler, step=PRECHECK_STEPS, initial_state_digest=_digest_tree(bundle["initial_state"]), prototypes=bundle["prototypes"], rng_state=_capture_rng())
    checkpoint = _save_and_verify_checkpoint(root, target, payload, PREFLIGHT_CHECKPOINT_NAME, consumer, bundle["v2_contract"])
    return {"firewall": firewall, "traces": traces, "checkpoint": checkpoint, "scientific_updates": 0, "scientific_meaning": "none"}


def _gradient_firewall_probe(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_encoder_only(consumer)
    indices = train_indices[:LOGICAL_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    named = dict(consumer.runtime.named_parameters())
    result = consumer.runtime.forward_from_image(batch)
    raw = _validate_result(result, LOGICAL_BATCH_SIZE, require_grad=True)
    with torch.no_grad():
        image_loss = (raw.detach() - result.image_input.slm_intensity.detach()).square().mean()
    proto_loss = _prototype_loss(result.z_ols.coefficients, batch_labels, bundle["prototypes"].to(device))
    total = proto_loss
    _require(total is proto_loss and not image_loss.requires_grad, "L_image is not disconnected from the V6 objective")
    total.backward()
    gradient = named["encoder.phase"].grad
    gradient_norm = _norm(gradient, "prototype gradient probe")
    _require(gradient_norm > 0.0 and named["generator.phase"].grad is None, "prototype gradient firewall failed")
    return {"firewall": firewall, "objective": "L_V6=L_proto", "image_loss_requires_grad": False, "image_loss_gradient_contribution": 0.0, "prototype_gradient_norm": gradient_norm, "prototype_gradient_finite": True, "generator_gradient": None}


def _synthetic_acceptance(root: Path, bundle: Mapping[str, Any], device: torch.device, target: Path) -> dict[str, Any]:
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    synthetic_train_labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    synthetic_train = (ramp + synthetic_train_labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    synthetic_eval_labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(2)
    synthetic_eval = synthetic_train.index_select(0, synthetic_eval_labels)
    synthetic_templates = synthetic_train.detach().clone()
    consumer = _new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    _configure_encoder_only(consumer)
    optimizer = _optimizer(consumer)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        traces.append(_train_step(consumer, optimizer, synthetic_train.to(device), synthetic_train_labels.to(device), bundle["prototypes"].to(device), step))
    evaluation = _evaluate(consumer, bundle["smooth"], synthetic_eval, synthetic_eval_labels, bundle["prototypes"].to(device), step=PRECHECK_STEPS, train_trace=traces[-1])
    selected = _cyclic_transfer(consumer, bundle["smooth"], synthetic_eval, synthetic_eval_labels, torch.arange(synthetic_eval.shape[0], dtype=torch.long), bundle["prototypes"], synthetic_templates)
    obs = {"labels": synthetic_eval_labels, "ols_margin_step": evaluation["_margins"], "smooth_logits_step": evaluation["_smooth_logits"], "cyclic_template_delta": selected["_observations"]["template_shifted_margin"] - selected["_observations"]["template_base_margin"], "cyclic_smooth_delta": selected["_observations"]["smooth_shifted_margin"] - selected["_observations"]["smooth_base_margin"]}
    sufficient = _save_sufficient(root, target, obs, name=PREFLIGHT_SYNTHETIC_OBS_NAME, schema="lightgen_2d_proto_only_semantic_warmup_v6_synthetic_sufficient_observations.v1")
    _require(all(torch.is_tensor(value) for value in obs.values()), "synthetic sufficient observations are incomplete")
    ols_bootstrap = _paired_bootstrap(obs["ols_margin_step"], synthetic_eval_labels, replicates=SMOKE_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    smooth_bootstrap = _paired_bootstrap(synthetic_eval_labels.to(torch.float64).mul(0.0), synthetic_eval_labels, replicates=SMOKE_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    decision = _classify(ols_bootstrap, smooth_bootstrap, selected, True)
    synthetic_report = {"schema": "lightgen_2d_proto_only_semantic_warmup_v6_synthetic_report.v1", "scientific_meaning": "none", "training_updates": PRECHECK_STEPS, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in selected.items() if not key.startswith("_") and key != "raw_validity"}, "decision": decision, "sufficient_observations": sufficient}
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME
    atomic_write_new_json(report_path, synthetic_report, artifact_kind="V6 synthetic acceptance report")
    report_sha = file_sha256(report_path)
    lineage = {"schema": "lightgen_2d_proto_only_semantic_warmup_v6_synthetic_lineage.v1", "result": "VALID_COMPLETED", "scientific_meaning": "none", "report": {"path": _relative(root, report_path), "sha256": report_sha}, "sufficient_observations": sufficient}
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V6 synthetic acceptance lineage")
    reloaded_report = _read_json(report_path, "synthetic report reload")
    reloaded_lineage = _read_json(lineage_path, "synthetic lineage reload")
    _require(reloaded_report["decision"] == decision and reloaded_lineage["report"]["sha256"] == report_sha, "synthetic report/lineage reload changed")
    return {"scientific_meaning": "none", "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in selected.items() if not key.startswith("_") and key != "raw_validity"}, "decision": decision, "sufficient_observations": sufficient, "report": {"path": _relative(root, report_path), "sha256": report_sha}, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "report_lineage_reloaded": True}


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_proto_only_semantic_warmup_v6_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V6_PREFLIGHT", "provenance": {"git_start": started}, "claim_boundary": {"scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V6 authorization")
        contract, v2_contract = _load_contract(root)
        bundle = _resolve_authorities(root, contract)
        dll = _prepare_cuda_runtime_dll_search_path()
        device, device_facts = _cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = _load_roles(root)
        rng = _initial_rng()
        payload["contract"] = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        payload["authorities"] = bundle["authorities"]
        payload["runtime"] = {"cuda_dll_search_path": dll, "device": device_facts, "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "trainable": ["encoder.phase"]}, "grid": PRODUCTION_GRID}
        payload["initial_state"] = {"combined_runtime_state_sha256": _digest_tree(bundle["initial_state"]), "encoder_phase_sha256": tensor_sha256(bundle["initial_state"]["encoder.phase"]), "generator_phase_sha256": tensor_sha256(bundle["initial_state"]["generator.phase"]), "rng_sha256": _digest_tree(rng)}
        payload["data_roles"] = dataset
        payload["production_no_data_smoke"] = _production_no_data_smoke(bundle, device)
        payload["gradient_firewall_probe"] = _gradient_firewall_probe(bundle, images, labels, train_indices, device)
        payload["three_step_cuda_smoke"] = _three_step_smoke(root, bundle, images, labels, train_indices, device, output_dir)
        payload["synthetic_acceptance"] = _synthetic_acceptance(root, bundle, device, output_dir)
        input_payload = {"schema": "lightgen_2d_proto_only_semantic_warmup_v6_preflight_input.v1", "version": 1, "task_id": TASK_ID, "contract": payload["contract"], "protocol": {"objective": "L_V6=L_proto", "image_loss_gradient_contribution": 0.0, "updates": SCIENTIFIC_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "synthetic_scientific_meaning": "none"}, "data_roles": {"TRAIN": 55_000, "DEV_scientific_values": 0, "protected": 0, "official_test": 0, "sealed": 0}}
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V6 preflight input")
        payload["input"] = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload["result"] = "AUTHORIZABLE"
        payload["status"] = "valid_completed"
        payload["classification"] = "LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_V6_AUTHORIZABLE"
        payload["claim_boundary"]["engineering_preflight_executed"] = True
        payload["next"] = "proceed directly to the one-shot 300-step V6 scientific execution"
    except HardBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no scientific execution"})
    except InvalidV6 as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V6_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; repair harness before any scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V6_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head")})
        payload["claim_boundary"].update({"scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V6 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": "lightgen_2d_proto_only_semantic_warmup_v6_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "provenance": payload["provenance"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V6 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_300_step_executed": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "V6 preflight authority is unavailable")
    report_sha = file_sha256(report_path)
    lineage_sha = file_sha256(lineage_path)
    report = _read_json(report_path, "V6 preflight report")
    lineage = _read_json(lineage_path, "V6 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "V6 preflight is not AUTHORIZABLE")
    _require(report.get("classification") == "LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_V6_AUTHORIZABLE", "V6 preflight classification changed")
    _require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("classification") == report.get("classification"), "V6 preflight lineage changed")
    _require(report.get("claim_boundary", {}).get("scientific_300_step_executed") is False, "preflight opened scientific execution")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": report_sha}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": lineage_sha}}


def _save_cyclic_raw(root: Path, target: Path, cyclic: Mapping[str, Any]) -> dict[str, Any]:
    observations = cyclic["_observations"]
    payload = {"schema": "lightgen_2d_proto_only_semantic_warmup_v6_cyclic_raw_outputs.v1", "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    path = target / SCIENTIFIC_RAW_NAME
    atomic_torch_save_new(path, payload, artifact_kind="V6 cyclic raw outputs")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V6 cyclic raw reload")
    _require(loaded.get("schema") == payload["schema"] and torch.equal(loaded["selection_indices"], observations["selection_indices"].cpu()), "V6 cyclic raw reload changed")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _public_evaluation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def run_scientific(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite scientific output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V6_SCIENTIFIC_HARNESS", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "governance": {"retry": False, "stage_b": False, "generation": False, "promotion": False}}
    scientific_started = False
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V6 authorization")
        contract, v2_contract = _load_contract(root)
        preflight = _validate_preflight(root)
        bundle = _resolve_authorities(root, contract)
        dll = _prepare_cuda_runtime_dll_search_path()
        device, device_facts = _cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = _load_roles(root)
        development_images = images.index_select(0, development_indices).contiguous()
        development_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = _load_selection(root, images, labels, development_indices)
        rng = _initial_rng()
        consumer = _new_consumer(v2_contract, device, PRODUCTION_GRID)
        _load_runtime_state(consumer, bundle["initial_state"])
        firewall = _configure_encoder_only(consumer)
        optimizer = _optimizer(consumer)
        sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        initial_state_digest = _digest_tree(bundle["initial_state"])
        payload.update({"contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}, "authorities": bundle["authorities"], "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID, "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "trainable_parameter_names": ["encoder.phase"]}}, "data_roles": dataset, "initial_state": {"combined_runtime_state_sha256": initial_state_digest, "encoder_phase_sha256": V3_ENCODER_SHA256, "generator_phase_sha256": V4_GENERATOR_TENSOR_SHA256, "rng_sha256": _digest_tree(rng)}, "firewall": firewall, "objective": {"formula": "L_V6=L_proto", "image_loss_gradient_contribution": 0.0, "image_loss_role": "evaluation-only"}, "selection": {"rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": SELECTION_LABELS_SHA256}})
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "contract": payload["contract"], "preflight": payload["preflight"], "protocol": {"objective": "L_V6=L_proto", "steps": SCIENTIFIC_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "batch": LOGICAL_BATCH_SIZE, "grid": PRODUCTION_GRID, "optimizer": payload["runtime"]["optimizer"], "prototype_scale": PROTOTYPE_SCALE, "cyclic_transfer": {"target_rule": "t=(y+1) mod 10", "selection_rows": SELECTION_ROWS, "selection_per_class": SELECTION_PER_CLASS, "selection_seed": SELECTION_SEED}, "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}}, "data_roles": {"TRAIN": 55_000, "DEV": 5_000, "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": payload["claim_boundary"]}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V6 scientific input")
        payload["input"] = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        scientific_started = True
        evaluations: dict[str, dict[str, Any]] = {}
        checkpoints: dict[str, dict[str, Any]] = {}
        step0 = _evaluate(consumer, bundle["smooth"], development_images, development_labels, bundle["prototypes"].to(device), step=0, train_trace=None)
        checkpoints["0"] = _save_and_verify_checkpoint(root, output_dir, _checkpoint_payload(consumer, optimizer, sampler, step=0, initial_state_digest=initial_state_digest, prototypes=bundle["prototypes"], rng_state=_capture_rng()), "lightgen_2d_proto_only_semantic_warmup_v6_step_0_checkpoint.v1.pt", consumer, v2_contract)
        evaluations["0"] = step0
        traces: list[dict[str, Any]] = []
        started_training = time.perf_counter()
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = sampler.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), step)
            traces.append(trace)
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                checkpoints[str(step)] = _save_and_verify_checkpoint(root, output_dir, _checkpoint_payload(consumer, optimizer, sampler, step=step, initial_state_digest=initial_state_digest, prototypes=bundle["prototypes"], rng_state=_capture_rng()), f"lightgen_2d_proto_only_semantic_warmup_v6_step_{step}_checkpoint.v1.pt", consumer, v2_contract)
                evaluations[str(step)] = _evaluate(consumer, bundle["smooth"], development_images, development_labels, bundle["prototypes"].to(device), step=step, train_trace=trace)
            if step % 25 == 0:
                print(json.dumps({"event": "scientific_progress", "step": step, "elapsed_seconds": time.perf_counter() - started_training}, ensure_ascii=False), flush=True)
        _require(set(evaluations) == {"0", "100", "200", "300"} and set(checkpoints) == {"0", "100", "200", "300"}, "scientific checkpoint/evaluation set is incomplete")
        step0_record = evaluations["0"]
        step300_record = evaluations["300"]
        ols_delta = step300_record["_margins"] - step0_record["_margins"]
        smooth_correct0 = step0_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_correct300 = step300_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        ols_bootstrap = _paired_bootstrap(ols_delta, development_labels)
        smooth_bootstrap = _paired_bootstrap((smooth_correct300.to(torch.float64) - smooth_correct0.to(torch.float64)), development_labels)
        cyclic = _cyclic_transfer(consumer, bundle["smooth"], selection_images, selection_labels, selection_indices, bundle["prototypes"], bundle["templates"])
        sufficient_obs = {"step0_ols_margin": step0_record["_margins"], "step300_ols_margin": step300_record["_margins"], "step0_smooth_logits": step0_record["_smooth_logits"], "step300_smooth_logits": step300_record["_smooth_logits"], **cyclic["_observations"]}
        sufficient_ref = _save_sufficient(root, output_dir, sufficient_obs, name=SCIENTIFIC_SUFFICIENT_NAME, schema=SUFFICIENT_SCHEMA)
        cyclic_raw_ref = _save_cyclic_raw(root, output_dir, cyclic)
        raw_valid = all(bool(evaluations[str(step)]["raw_validity"]["finite"] and evaluations[str(step)]["raw_validity"]["structured"] and evaluations[str(step)]["raw_validity"]["nonconstant"]) for step in CHECKPOINT_STEPS) and bool(cyclic["raw_validity"]["base"]["finite"] and cyclic["raw_validity"]["shifted"]["finite"])
        decision = _classify(ols_bootstrap, smooth_bootstrap, cyclic, raw_valid)
        r0 = float(step0_record["smooth"]["macro_recall"])
        r300 = float(step300_record["smooth"]["macro_recall"])
        stage_firewall = {"metric": "step-300 DEV Smooth macro recall", "threshold": STAGE_A_FIREWALL, "R300": r300, "status": "STAGE_A_FIREWALL_PASSED" if r300 >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
        public_evaluations = {step: _public_evaluation(record) for step, record in evaluations.items()}
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": decision["classification"], "evaluations": public_evaluations, "traces": traces, "checkpoints": checkpoints, "scientific_updates": SCIENTIFIC_STEPS, "runtime_device": device_facts, "step300_minus_step0": {"Delta_M_OLS": ols_bootstrap, "Delta_R_Smooth": smooth_bootstrap, "Delta_L_image": float(step300_record["losses"]["L_image"] - step0_record["losses"]["L_image"])}, "ols_mechanism": {"step0": step0_record["ols_margin"], "step300": step300_record["ols_margin"], "bootstrap": ols_bootstrap}, "natural_smooth_semantics": {"R0": r0, "R300": r300, "Delta_R": r300 - r0, "bootstrap": smooth_bootstrap, "step0_per_class": step0_record["smooth"]["per_class_recall"], "step300_per_class": step300_record["smooth"]["per_class_recall"]}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "artifacts": {"sufficient_observations": sufficient_ref, "cyclic_raw_outputs": cyclic_raw_ref}, "decision": decision, "stage_a_firewall": stage_firewall, "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_300_step_executed": True}, "governance": {"optimizer": "Adam on encoder.phase only", "backward": True, "generator_update": False, "optical_update": False, "image_loss_gradient": False, "retry": False, "stage_b": False, "scale_extraction": False, "prior_shaping": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "training_authorized": False, "generation_readiness": "not_ready"}, "next": "no_retry; V6 gate result is bounded and does not authorize Stage B"})
    except HardBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no scientific retry"})
    except InvalidV6 as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V6_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V6_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific retry"})
    finally:
        ending = git_facts(root)
        status_after = _status_lines(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == status_after, "unexpected_modified_paths": sorted(set(status_after).symmetric_difference(status_before)), "scientific_started": scientific_started})
        payload["claim_boundary"].update({"scientific_training_executed": bool(scientific_started), "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False})
    report_path = output_dir / SCIENTIFIC_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V6 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "preflight": payload.get("preflight"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V6 scientific lineage")
    report_reload = _read_json(report_path, "V6 report reload")
    lineage_reload = _read_json(lineage_path, "V6 lineage reload")
    _require(report_reload.get("result") == payload["result"] and lineage_reload.get("result") == payload["result"], "V6 report/lineage reload changed")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_updates": int(payload.get("scientific_updates", 0)), "stage_a_firewall": payload.get("stage_a_firewall"), "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        target = (args.output_dir or root / DEFAULT_PREFLIGHT_DIR).resolve()
        result = run_preflight(root, target)
    else:
        target = (args.output_dir or root / DEFAULT_SCIENTIFIC_DIR).resolve()
        result = run_scientific(root, target)
    return 0 if result["result"] in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
