"""Execute the one-shot V4 fixed-OLS generator-recoverability experiment.

This entry point consumes the READY V4 contract and the accepted V4 engineering
preflight.  It runs one 300-update generator-only CUDA experiment, evaluates
the fixed DEV selection at steps 100/200/300, and then closes the V4 gate.  It
does not retry, enter Stage B, read protected/test/sealed roles, or perform
generation or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, LightGen2DComplexOLSTrainingConsumerV2, LOGICAL_BATCH_SIZE, file_sha256, git_facts, load_v2_training_contract, tensor_sha256
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import stratified_mnist_indices


TASK_ID = "EXECUTE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_V4"
OUTER_GATE = "FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED"
REPORT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_scientific_input.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_scientific_checkpoint.v1"
PANEL_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_raw_cmos_panel.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_bootstrap.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "383197a930dd48f98294120f8cf867444f994f85"

CONTRACT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract_v4_attempt_01/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json"
)
CONTRACT_SHA256 = "d46740ded4db58d7d9f0caa8acaf40942d03dc3407d9611d24efcaa12d3036b8"
PREFLIGHT_PATH = Path("scripts/research/lightgen/execute_lightgen_2d_fixed_ols_generator_recoverability_v4.py")
PREFLIGHT_SHA256 = "8f99e098e121ab3dc8b667ddcd9c1dfbf57d1687fbfd8b87e166cf3bbb2d5b73"
PREFLIGHT_DIR = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02"
)
PREFLIGHT_REPORT_PATH = PREFLIGHT_DIR / "lightgen_2d_fixed_ols_generator_recoverability_preflight_report.v1.json"
PREFLIGHT_REPORT_SHA256 = "c2b0d852e1178915523ceee21172b10b8d462b6d0096ecf87c8443c410c7b7fb"
PREFLIGHT_LINEAGE_PATH = PREFLIGHT_DIR / "lightgen_2d_fixed_ols_generator_recoverability_preflight_lineage.v1.json"
PREFLIGHT_LINEAGE_SHA256 = "139fd231c73cb5d8f90e74508e6d8d13645b7a6c9c30d839ad7311cbba642da8"
PREFLIGHT_INPUT_PATH = PREFLIGHT_DIR / "lightgen_2d_fixed_ols_generator_recoverability_preflight_input.v1.json"
PREFLIGHT_INPUT_SHA256 = "ef3388a575c1eb648baef1c60e8932d6d0d51805af09a784d68dc14e7739e626"

V3_SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json"
)
V3_SCIENTIFIC_REPORT_SHA256 = "8d7930449a1872e685ad39dafa499d89fdcc1d6bb5a9317e7ba06c554f54f31a"
V3_SCIENTIFIC_LINEAGE_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_lineage.v1.json"
)
V3_SCIENTIFIC_LINEAGE_SHA256 = "00127edfad105e077342a8eb270447ea577cd4c48b816fb0435c0e6edbfc237d"
V3_BOOTSTRAP_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_paired_bootstrap.v1.pt"
)
V3_BOOTSTRAP_SHA256 = "0b34a1734c16ab175f1a801b559b2142915cb63a21f21163c1ce5c320f1ea2c1"
V3_CONTROL_CHECKPOINT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
V3_CONTROL_CHECKPOINT_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"

V2_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_01"
)
INPUT_NAME = "lightgen_2d_fixed_ols_generator_recoverability_scientific_input.v1.json"
REPORT_NAME = "lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1.json"
LINEAGE_NAME = "lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1.json"
BOOTSTRAP_NAME = "lightgen_2d_fixed_ols_generator_recoverability_paired_bootstrap.v1.pt"

PRODUCTION_GRID = 2048
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEPS = (100, 200, 300)
INITIALIZATION_SEED = 1234
SCHEDULE_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 999
STAGE_A_FIREWALL = 0.937
SMOOTH_MARGIN = 0.05
SMOOTH_PER_CLASS_DROP = 0.10
IMAGE_LOSS_TOLERANCE = 0.005
DETERMINISTIC_RTOL = 1.0e-5
DETERMINISTIC_ATOL = 1.0e-6
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000
INITIAL_RUNTIME_STATE_SHA256 = "0f90086cf26a179905a27a5a519f7599de256e4f4225f7798f24aeed086eb813"
PHASE_SHAPE = (2, 1, 100, 100)
OLS_TRANSPORT = torch.eye(100, dtype=torch.complex64)


class InvalidScientific(RuntimeError):
    """The frozen scientific harness or authority is invalid."""


class ScientificBlocked(RuntimeError):
    """A valid boundary prevents this one-shot execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidScientific(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json_ref(root: Path, relative: str | Path, expected_sha256: str, name: str) -> dict[str, Any]:
    path = _path(root, relative)
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected_sha256, f"{name} SHA mismatch: expected {expected_sha256}, observed {observed}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidScientific(f"{name} is not valid JSON") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


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


def _norm(value: torch.Tensor) -> float:
    _finite(value, "norm input")
    return float(value.detach().to(torch.float64).square().sum().sqrt().cpu())


def _max_difference(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0 if left is right else math.inf
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return math.inf
        if left.numel() == 0:
            return 0.0
        dtype = torch.complex128 if (torch.is_complex(left) or torch.is_complex(right)) else torch.float64
        return float((left.to(dtype) - right.to(dtype)).abs().max())
    if torch.is_tensor(left) or torch.is_tensor(right):
        return math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return math.inf
        return max((_max_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_max_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def _load_preflight_module(root: Path) -> Any:
    path = _path(root, PREFLIGHT_PATH)
    _require(path.is_file() and file_sha256(path) == PREFLIGHT_SHA256, "accepted V4 preflight entrypoint hash changed")
    spec = importlib.util.spec_from_file_location("lightgen_v4_preflight_consumer", path)
    _require(spec is not None and spec.loader is not None, "V4 preflight module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_v4_authority(root: Path, preflight: Any) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _read_json_ref(root, CONTRACT_PATH, CONTRACT_SHA256, "V4 contract")
    _require(contract.get("task_id") == "DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4", "V4 contract identity changed")
    _require(contract.get("result") == "READY" and contract.get("status") == "valid_completed" and contract.get("outer_gate") == OUTER_GATE, "V4 contract is not READY")
    _require(contract.get("governance", {}).get("training_authorized") is False, "V4 contract opened scientific training")

    preflight_report = _read_json_ref(root, PREFLIGHT_REPORT_PATH, PREFLIGHT_REPORT_SHA256, "accepted V4 preflight report")
    _require(preflight_report.get("result") == "AUTHORIZABLE" and preflight_report.get("status") == "valid_completed", "accepted V4 preflight result changed")
    _require(preflight_report.get("classification") == "FIXED_OLS_GENERATOR_RECOVERABILITY_PREFLIGHT_AUTHORIZABLE", "accepted V4 preflight classification changed")
    _require(preflight_report.get("contract", {}).get("sha256") == CONTRACT_SHA256, "preflight contract binding changed")
    _require(preflight_report.get("entrypoint", {}).get("path") == PREFLIGHT_PATH.as_posix() and preflight_report.get("entrypoint", {}).get("sha256") == PREFLIGHT_SHA256, "preflight entrypoint binding changed")
    _require(preflight_report.get("governance", {}).get("scientific_training_executed") is False and preflight_report.get("governance", {}).get("scientific_updates_executed") == 0, "preflight already consumed scientific training")
    _require(preflight_report.get("data", {}).get("protected", {}).get("rows_read") == 0 and preflight_report.get("data", {}).get("test", {}).get("rows_read") == 0 and preflight_report.get("data", {}).get("sealed", {}).get("rows_read") == 0, "accepted preflight data boundary changed")
    preflight_lineage = _read_json_ref(root, PREFLIGHT_LINEAGE_PATH, PREFLIGHT_LINEAGE_SHA256, "accepted V4 preflight lineage")
    _require(preflight_lineage.get("result") == "AUTHORIZABLE" and preflight_lineage.get("classification") == preflight_report.get("classification"), "preflight lineage result changed")
    _require(preflight_lineage.get("artifacts", {}).get("report", {}).get("sha256") == PREFLIGHT_REPORT_SHA256, "preflight lineage report binding changed")
    preflight_input = _read_json_ref(root, PREFLIGHT_INPUT_PATH, PREFLIGHT_INPUT_SHA256, "accepted V4 preflight input")
    _require(preflight_input.get("contract", {}).get("sha256") == CONTRACT_SHA256, "preflight input contract binding changed")
    _require(preflight_report.get("z_equivalence", {}).get("z_equivalent") is True and float(preflight_report.get("z_equivalence", {}).get("max_abs_z_difference")) == 0.0, "cached-z authority is not exact")
    _require(preflight_report.get("parameter_firewall", {}).get("trainable_names") == ["generator.phase"], "preflight trainable firewall changed")

    current_authorities = preflight._load_contract(root)[2]
    v3_report = _read_json_ref(root, V3_SCIENTIFIC_REPORT_PATH, V3_SCIENTIFIC_REPORT_SHA256, "V3 scientific comparator report")
    _require(v3_report.get("result") == "VALID_COMPLETED" and v3_report.get("status") == "valid_completed" and v3_report.get("classification") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED", "V3 comparator result changed")
    _require(v3_report.get("outer_gate") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED", "V3 comparator outer gate changed")
    v3_lineage = _read_json_ref(root, V3_SCIENTIFIC_LINEAGE_PATH, V3_SCIENTIFIC_LINEAGE_SHA256, "V3 scientific comparator lineage")
    _require(v3_lineage.get("result") == "VALID_COMPLETED" and v3_lineage.get("classification") == v3_report.get("classification"), "V3 comparator lineage changed")
    v3_bootstrap = _mapping(safe_torch_load(_path(root, V3_BOOTSTRAP_PATH), expected_sha256=V3_BOOTSTRAP_SHA256, map_location="cpu"), "V3 paired-bootstrap artifact")
    _require(v3_bootstrap.get("schema") == "lightgen_2d_semantic_complex_ols_objective_correction_v3_bootstrap.v1" and v3_bootstrap.get("endpoint_step") == 300, "V3 bootstrap identity changed")
    _require(v3_bootstrap.get("selection_indices_sha256") == "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8", "V3 selection identity changed")
    bootstrap_protocol = _mapping(v3_bootstrap.get("bootstrap"), "V3 bootstrap protocol")
    _require(bootstrap_protocol.get("replicates") == BOOTSTRAP_REPLICATES and bootstrap_protocol.get("seed") == BOOTSTRAP_SEED, "paired-bootstrap protocol changed")
    control_correct = v3_bootstrap.get("control_smooth_correct")
    _require(torch.is_tensor(control_correct) and tuple(control_correct.shape) == (1000,) and control_correct.dtype == torch.bool, "V3 control Smooth correctness vector changed")
    control_endpoint = _mapping(_mapping(v3_report.get("arms"), "V3 arms").get("control"), "V3 control arm")
    control_eval = _mapping(_mapping(control_endpoint.get("evaluations"), "V3 control evaluations").get("300"), "V3 control step-300 evaluation")
    control_losses = _mapping(control_eval.get("losses"), "V3 control losses")
    control_smooth = _mapping(control_eval.get("smooth_evaluator"), "V3 control Smooth endpoint")
    _require(float(control_losses.get("L_image")) == 0.0673580589145422, "V3 control exact image loss changed")
    _require(float(control_smooth.get("macro_recall")) == 0.10299999993294477, "V3 control exact Smooth macro changed")
    _require(control_smooth.get("per_class_recall") == [0.0, 0.0, 0.0, 0.0, 0.029999999329447746, 0.0, 0.0, 0.0, 1.0, 0.0], "V3 control Smooth per-class endpoint changed")
    return contract, preflight, current_authorities, {"report": v3_report, "lineage": v3_lineage, "bootstrap": v3_bootstrap, "control_correct": control_correct.cpu().clone(), "control_eval": control_eval}, {"report": preflight_report, "lineage": preflight_lineage, "input": preflight_input}


def _load_v3_control_state(root: Path, preflight: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint, state = preflight._load_v3_control_state(root)
    _require(file_sha256(_path(root, V3_CONTROL_CHECKPOINT_PATH)) == V3_CONTROL_CHECKPOINT_SHA256, "V3 CONTROL checkpoint trust boundary changed")
    _require(tensor_sha256(state["encoder.phase"]) == "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca", "V3 CONTROL encoder tensor changed")
    _require(tensor_sha256(state["generator.phase"]) == "1bb3a3e6b641aae69eef5c22614326795bc300fc26c91b6569fbf742df1c351c", "V3 CONTROL generator tensor changed")
    _require(not torch.equal(state["generator.phase"], torch.zeros(PHASE_SHAPE, dtype=torch.float32)), "V3 CONTROL generator was incorrectly treated as the V4 initialization")
    _require(torch.equal(state["ols.operator.transport"], OLS_TRANSPORT), "V3 CONTROL OLS transport is not T=I_100")
    return checkpoint, state


def _load_roles(root: Path, preflight: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, train_indices, source_metadata = preflight._load_train_role(root)
    expected_train, development_indices = stratified_mnist_indices(labels, seed=INITIALIZATION_SEED, validation_per_label=500)
    _require(torch.equal(train_indices, expected_train), "TRAIN split differs from the frozen split")
    development_labels = labels.index_select(0, development_indices)
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    selected: list[torch.Tensor] = []
    for digit in range(10):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, f"DEV class count changed for {digit}")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    selection_indices = torch.cat(selected).contiguous()
    _require(tensor_sha256(selection_indices) == "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8", "fixed DEV selection identity changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV partition changed")
    metadata = {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000},
        "train": {"rows": 55_000, "counts_by_class": source_metadata["train"]["counts_by_class"], "indices_sha256": source_metadata["train"]["indices_sha256"], "optimization_rows_read": 55_000},
        "development": {"rows": 5_000, "selection_rows": 1_000, "selection_per_class": 100, "selection_seed": DEVELOPMENT_SELECTION_SEED, "indices_sha256": tensor_sha256(development_indices), "selection_indices_sha256": tensor_sha256(selection_indices), "values_read_for_evaluation": 1_000},
        "calibration": {"rows_read": 0},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, selection_indices, metadata


def _load_v2_contract_before_seed(root: Path) -> Any:
    path = _path(root, V2_CONTRACT_PATH)
    _require(path.is_file() and file_sha256(path) == V2_CONTRACT_SHA256, "corrected V2 contract is unavailable or hash-mismatched")
    contract = load_v2_training_contract(path, repo_root=root)
    _require(contract.sha256 == V2_CONTRACT_SHA256, "corrected V2 contract hash changed")
    return contract


def _materialize_schedule(train_indices: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]], str]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=SCIENTIFIC_STEPS)
    initial = _cpu_tree(sampler.state_dict())
    schedule: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(1, SCIENTIFIC_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(10, dtype=torch.long)), f"TRAIN sampler labels changed at step {step}")
        schedule.append((indices.cpu().clone(), batch_labels.cpu().clone()))
    identity = {"initial": initial, "batch_indices": [item[0] for item in schedule], "batch_labels": [item[1] for item in schedule]}
    return initial, schedule, _digest_tree(identity)


def _load_evaluators(root: Path, preflight: Any, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    frozen, smooth, states = preflight._load_evaluators(root, authorities, device)
    return frozen, smooth, states


def _runtime_state(consumer: Any, preflight: Any) -> dict[str, Any]:
    return preflight._runtime_state(consumer)


def _capture_rng(preflight: Any) -> dict[str, Any]:
    state = dict(preflight._capture_rng_state())
    state.setdefault("cuda_available", bool(torch.cuda.is_available()))
    return state


def _restore_rng(state: Mapping[str, Any], preflight: Any) -> None:
    preflight._restore_rng_state(state)


def _configure_fixed_consumer(consumer: Any, state: Mapping[str, Any], preflight: Any) -> None:
    preflight._configure_fixed_generator_consumer(consumer, state)
    names = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(names == ["generator.phase"], "scientific trainable firewall changed")


def _canonical_z_and_target(consumer: Any, batch_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        canonical = consumer.runtime.forward_from_image(batch_images)
    _require(canonical.image_input is not None and canonical.ols_input_field is not None and canonical.z_ols.source == "real_encoded", "canonical image path did not expose real z")
    z = canonical.z_ols.coefficients.detach()
    target = canonical.image_input.slm_intensity.detach()
    _require(tuple(z.shape) == (int(batch_images.shape[0]), 100) and z.dtype == torch.complex64, "canonical z shape/dtype changed")
    _require(canonical.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "modal projection boundary is inactive")
    _require(canonical.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "shared W_OLS provenance changed")
    _require(canonical.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "T=I_100 provenance changed")
    _finite(canonical.raw_intensity, "canonical raw CMOS")
    _finite(target, "canonical SLM target")
    return z, target, {"source": "canonical frozen encoder", "modal_projection_boundary": True, "transport": "T=I_100_coefficient_space", "w_ols": True}


def _raw_summary(raw: torch.Tensor) -> dict[str, Any]:
    detached = raw.detach().cpu()
    per_image = detached.flatten(1)
    per_image_variance = per_image.var(dim=1, unbiased=False)
    across_sample_variance = per_image.var(dim=0, unbiased=False)
    finite = bool(torch.isfinite(detached).all())
    shape_valid = tuple(detached.shape) == (1000, 1, 200, 200) and detached.dtype == torch.float32
    structured = bool(float(per_image_variance.mean()) > 0.0 and float(across_sample_variance.mean()) > 0.0)
    return {"shape": list(detached.shape), "dtype": str(detached.dtype), "finite": finite, "shape_valid": shape_valid, "structured": structured, "nonconstant": bool(float(detached.std(unbiased=False)) > 0.0), "mean": float(detached.mean()), "std": float(detached.std(unbiased=False)), "per_image_variance_mean": float(per_image_variance.mean()), "across_sample_variance_mean": float(across_sample_variance.mean()), "tensor_sha256": tensor_sha256(detached)}


def _run_training_step(consumer: Any, optimizer: torch.optim.Adam, batch_images: torch.Tensor, batch_indices: torch.Tensor, batch_labels: torch.Tensor, step: int, preflight: Any) -> tuple[dict[str, Any], torch.Tensor]:
    consumer.runtime.train()
    named = dict(consumer.runtime.named_parameters())
    for parameter in named.values():
        parameter.grad = None
    before_state = _runtime_state(consumer, preflight)
    before_encoder = before_state["encoder.phase"]
    before_ols = {key: before_state[key] for key in ("ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport")}
    before_generator = before_state["generator.phase"]
    optimizer.zero_grad(set_to_none=True)
    z, target, canonical = _canonical_z_and_target(consumer, batch_images)
    result = consumer.runtime.forward_from_ols(z)
    _require(result.image_input is None and result.ols_input_field is None and result.z_ols.source == "external_generated", "generator-only path crossed image/encoder boundary")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "generator W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "generator T=I_100 provenance changed")
    raw = result.raw_intensity
    _require(tuple(raw.shape) == (LOGICAL_BATCH_SIZE, 1, 200, 200) and raw.dtype == torch.float32, "training raw CMOS shape/dtype changed")
    _finite(raw, "training raw CMOS")
    image_loss = (raw - target).square().mean()
    _finite(image_loss, "training image loss")
    image_loss.backward()
    generator_gradient = named["generator.phase"].grad
    _require(generator_gradient is not None, "generator.phase received no gradient")
    _finite(generator_gradient, "generator.phase gradient")
    _require(float(generator_gradient.abs().sum().cpu()) > 0.0, "generator.phase gradient is identically zero")
    _require(named["encoder.phase"].grad is None, "encoder.phase received a gradient")
    _require(all(parameter.grad is None for name, parameter in named.items() if name != "generator.phase"), "frozen parameter received a gradient")
    generator_gradient_norm = _norm(generator_gradient)
    optimizer.step()
    torch.cuda.synchronize(consumer.device)
    after_state = _runtime_state(consumer, preflight)
    generator_update_norm = float((after_state["generator.phase"] - before_generator).to(torch.float64).square().sum().sqrt())
    _require(generator_update_norm > 0.0, "generator.phase did not update")
    encoder_unchanged = torch.equal(before_encoder, after_state["encoder.phase"])
    ols_unchanged = all(torch.equal(before_ols[key], after_state[key]) for key in before_ols)
    _require(encoder_unchanged and ols_unchanged, "frozen encoder or OLS changed")
    _finite(after_state["generator.phase"], "updated generator.phase")
    trace = {"step": step, "objective": "L_generator=L_image", "batch_indices_sha256": tensor_sha256(batch_indices), "batch_labels_sha256": tensor_sha256(batch_labels), "z_sha256": tensor_sha256(z.cpu()), "target_sha256": tensor_sha256(target.cpu()), "train_image_loss": float(image_loss.detach().cpu()), "gradient_norm_generator_phase": generator_gradient_norm, "gradient_encoder_phase": None, "generator_update_norm": generator_update_norm, "encoder_unchanged": encoder_unchanged, "ols_unchanged": ols_unchanged, "canonical_path": canonical, "w_ols_active": True, "transport": "T=I_100_coefficient_space", "raw_cmos": {"shape": list(raw.shape), "dtype": str(raw.dtype), "finite": bool(torch.isfinite(raw).all())}, "generator_phase_before_sha256": tensor_sha256(before_generator), "generator_phase_after_sha256": tensor_sha256(after_state["generator.phase"])}
    return trace, raw.detach().cpu().clone()


@torch.no_grad()
def _evaluate_checkpoint(consumer: Any, frozen: Any, smooth: Any, selection_images: torch.Tensor, selection_labels: torch.Tensor, *, step: int, train_trace: Mapping[str, Any]) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    raw_parts: list[torch.Tensor] = []
    frozen_correct: list[torch.Tensor] = []
    smooth_correct: list[torch.Tensor] = []
    image_sum = 0.0
    try:
        for start in range(0, selection_images.shape[0], LOGICAL_BATCH_SIZE):
            batch = selection_images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            labels = selection_labels[start : start + LOGICAL_BATCH_SIZE]
            result = consumer.runtime.forward_from_image(batch)
            _require(result.image_input is not None and result.ols_input_field is not None and result.z_ols.source == "real_encoded", "DEV evaluation lost canonical real path")
            _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "DEV modal projection boundary is inactive")
            _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "DEV transport changed")
            raw = result.raw_intensity
            target = result.image_input.slm_intensity
            _finite(raw, "DEV raw CMOS")
            _finite(target, "DEV target")
            loss = (raw - target).square().mean()
            _finite(loss, "DEV image loss")
            view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
            _, frozen_logits = frozen(view)
            _, smooth_logits = smooth(view)
            frozen_correct.append((frozen_logits.argmax(dim=1).cpu() == labels).contiguous())
            smooth_correct.append((smooth_logits.argmax(dim=1).cpu() == labels).contiguous())
            raw_parts.append(raw.cpu().contiguous())
            image_sum += float(loss.cpu()) * int(batch.shape[0])
            del batch, result, raw, target, view, frozen_logits, smooth_logits
    finally:
        consumer.runtime.train(was_training)
    raw_cpu = torch.cat(raw_parts, dim=0)
    frozen_correct_cpu = torch.cat(frozen_correct, dim=0)
    smooth_correct_cpu = torch.cat(smooth_correct, dim=0)
    _require(tuple(raw_cpu.shape) == (1000, 1, 200, 200) and raw_cpu.dtype == torch.float32, "DEV raw panel shape/dtype changed")
    summary = _raw_summary(raw_cpu)
    _require(summary["finite"] and summary["shape_valid"] and summary["structured"] and summary["nonconstant"], f"DEV raw panel failed at step {step}")

    def recalls(correct: torch.Tensor) -> tuple[list[float], float]:
        values: list[float] = []
        for digit in range(10):
            mask = selection_labels == digit
            _require(int(mask.sum()) == 100, f"DEV selection class count changed for {digit}")
            values.append(float(correct[mask].to(torch.float32).mean()))
        return values, float(sum(values) / 10.0)

    frozen_per_class, frozen_macro = recalls(frozen_correct_cpu)
    smooth_per_class, smooth_macro = recalls(smooth_correct_cpu)
    return {"step": step, "train_image_loss": float(train_trace["train_image_loss"]), "dev_image_loss": image_sum / 1000.0, "frozen_evaluator": {"macro_recall": frozen_macro, "per_class_recall": frozen_per_class, "role": "DESCRIPTIVE_ONLY"}, "smooth_evaluator": {"macro_recall": smooth_macro, "per_class_recall": smooth_per_class}, "raw_panel": summary, "_raw_cpu": raw_cpu, "_frozen_correct": frozen_correct_cpu, "_smooth_correct": smooth_correct_cpu, "train_step_trace": dict(train_trace)}


def _save_panel(root: Path, target: Path, record: Mapping[str, Any], selection_indices: torch.Tensor, selection_labels: torch.Tensor) -> dict[str, Any]:
    step = int(record["step"])
    path = target / f"lightgen_2d_fixed_ols_generator_recoverability_v4_step_{step}_raw_cmos_panel.v1.pt"
    payload = {"schema": PANEL_SCHEMA, "version": 1, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "preflight": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "step": step, "role": "fixed DEV selection only", "selection_indices_sha256": tensor_sha256(selection_indices), "selection_labels_sha256": tensor_sha256(selection_labels), "raw_cmos": record["_raw_cpu"]}
    atomic_torch_save_new(path, payload, artifact_kind=f"V4 step-{step} raw-CMOS panel")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"V4 step-{step} raw-CMOS panel")
    _require(torch.is_tensor(loaded.get("raw_cmos")) and tensor_sha256(loaded["raw_cmos"]) == record["raw_panel"]["tensor_sha256"], f"V4 step-{step} raw panel reload changed")
    return {"path": _relative(root, path), "sha256": digest, "tensor_sha256": record["raw_panel"]["tensor_sha256"], "shape": list(record["_raw_cpu"].shape), "role": "fixed DEV selection only"}


def _checkpoint_payload(root: Path, consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, *, step: int, initial_state_digest: str, initial_rng_digest: str, schedule_digest: str) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "preflight": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "v3_control_checkpoint": {"path": V3_CONTROL_CHECKPOINT_PATH.as_posix(), "sha256": V3_CONTROL_CHECKPOINT_SHA256, "step": 300}, "checkpoint_step": step, "primary_endpoint": step == 300, "objective": "L_generator=L_image", "base_initial_state_digest": initial_state_digest, "initial_rng_digest": initial_rng_digest, "schedule_digest": schedule_digest, "runtime_state": _runtime_state(consumer, preflight_module_placeholder), "optimizer_state": _cpu_tree(optimizer.state_dict()), "sampler_state": _cpu_tree(sampler.state_dict()), "rng_state": _cpu_tree(_capture_rng(preflight_module_placeholder)), "trainable_parameter_names": ["generator.phase"], "frozen_parameter_names": ["encoder.phase"], "data_roles": {"train": True, "development": False, "calibration": False, "protected": False, "test": False, "sealed": False}, "scientific_training_checkpoint": True, "training_authorized": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "formal_promotion": False, "repo_root": str(root)}


def _save_checkpoint(root: Path, target: Path, payload: Mapping[str, Any], step: int) -> dict[str, Any]:
    path = target / f"lightgen_2d_fixed_ols_generator_recoverability_v4_step_{step}_checkpoint.v1.pt"
    atomic_torch_save_new(path, payload, artifact_kind=f"V4 step-{step} scientific checkpoint")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"V4 step-{step} checkpoint")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("checkpoint_step") == step and loaded.get("contract", {}).get("sha256") == CONTRACT_SHA256, f"V4 step-{step} checkpoint identity changed")
    return {"path": _relative(root, path), "sha256": digest, "step": step, "loaded": True, "scientific_training_checkpoint": True}


def _paired_bootstrap(root: Path, target: Path, control_correct: torch.Tensor, candidate_correct: torch.Tensor, selection_labels: torch.Tensor) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    class_deltas: list[torch.Tensor] = []
    draws: list[torch.Tensor] = []
    per_class_delta: list[float] = []
    for digit in range(10):
        matches = torch.nonzero(selection_labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 100, f"bootstrap class count changed for {digit}")
        draw = torch.randint(0, matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        draws.append(draw)
        control_values = control_correct.index_select(0, matches)[draw]
        candidate_values = candidate_correct.index_select(0, matches)[draw]
        class_deltas.append((candidate_values.to(torch.float64) - control_values.to(torch.float64)).mean(dim=1))
        per_class_delta.append(float(candidate_correct.index_select(0, matches).to(torch.float64).mean() - control_correct.index_select(0, matches).to(torch.float64).mean()))
    macro_delta = torch.stack(class_deltas, dim=0).mean(dim=0)
    quantiles = torch.quantile(macro_delta, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    path = target / BOOTSTRAP_NAME
    payload = {"schema": BOOTSTRAP_SCHEMA, "version": 1, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "preflight": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "v3_comparator": {"report": {"path": V3_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V3_SCIENTIFIC_REPORT_SHA256}, "bootstrap": {"path": V3_BOOTSTRAP_PATH.as_posix(), "sha256": V3_BOOTSTRAP_SHA256}}, "endpoint_step": 300, "selection_indices_sha256": "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8", "bootstrap": {"type": "paired, class-stratified bootstrap over the same 100 examples per class", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "quantile_convention": "torch.quantile interpolation=linear", "class_draws": draws, "macro_delta": macro_delta}, "control_smooth_correct": control_correct, "v4_smooth_correct": candidate_correct}
    atomic_torch_save_new(path, payload, artifact_kind="V4 paired-bootstrap artifact")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V4 paired-bootstrap artifact")
    _require(torch.is_tensor(loaded.get("bootstrap", {}).get("macro_delta")) and tensor_sha256(loaded["bootstrap"]["macro_delta"]) == tensor_sha256(macro_delta), "V4 bootstrap reload changed")
    return {"path": _relative(root, path), "sha256": digest, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "delta_smooth": float(macro_delta.mean()), "ci_lower": float(quantiles[0]), "ci_upper": float(quantiles[1]), "per_class_delta": per_class_delta, "macro_delta_tensor_sha256": tensor_sha256(macro_delta)}


def _decision(v3_control: Mapping[str, Any], v4_endpoint: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    control_smooth = float(v3_control["smooth_evaluator"]["macro_recall"])
    v4_smooth = float(v4_endpoint["smooth_evaluator"]["macro_recall"])
    smooth_delta = v4_smooth - control_smooth
    image_delta = float(v4_endpoint["dev_image_loss"] - v3_control["losses"]["L_image"])
    per_class_delta = [float(v4_endpoint["smooth_evaluator"]["per_class_recall"][digit] - v3_control["smooth_evaluator"]["per_class_recall"][digit]) for digit in range(10)]
    raw = v4_endpoint["raw_panel"]
    checks = {"smooth_delta_at_least_0_05": smooth_delta >= SMOOTH_MARGIN, "paired_bootstrap_ci_lower_gt_zero": float(bootstrap["ci_lower"]) > 0.0, "no_smooth_per_class_drop_gt_0_10": min(per_class_delta) >= -SMOOTH_PER_CLASS_DROP, "candidate_image_loss_within_tolerance": image_delta <= IMAGE_LOSS_TOLERANCE, "raw_cmos_endpoint_panel_valid": bool(raw["finite"] and raw["shape_valid"] and raw["structured"] and raw["nonconstant"])}
    return {"primary_endpoint_step": 300, "v3_control_smooth_macro_recall": control_smooth, "v4_smooth_macro_recall": v4_smooth, "delta_smooth": smooth_delta, "paired_bootstrap_ci": {"lower": float(bootstrap["ci_lower"]), "upper": float(bootstrap["ci_upper"])}, "smooth_per_class_delta": per_class_delta, "maximum_per_class_degradation": min(per_class_delta), "v3_control_dev_image_loss": float(v3_control["losses"]["L_image"]), "v4_dev_image_loss": float(v4_endpoint["dev_image_loss"]), "image_loss_delta_v4_minus_v3_control": image_delta, "raw_cmos": {key: raw[key] for key in ("finite", "shape_valid", "structured", "nonconstant")}, "checks": checks, "all_support_conditions": all(checks.values()), "classification": SUPPORTED_CLASSIFICATION if all(checks.values()) else NOT_SUPPORTED_CLASSIFICATION}


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha256: str) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V4_SCIENTIFIC_NOT_COMPLETED", "entrypoint": {"path": _relative(root, Path(__file__)), "sha256": entrypoint_sha256}, "provenance": {"git_start": dict(started)}, "claim_boundary": {"scientific_execution_authorization_consumed": True, "scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "sampled_ols_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"}, "next": None}


def run_scientific(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    global preflight_module_placeholder
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    _require(not target.exists(), f"one-shot output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git state differs from the frozen execution")
    entrypoint_sha256 = file_sha256(Path(__file__).resolve())
    payload = _base_payload(root, started, entrypoint_sha256)
    input_ref: dict[str, Any] | None = None
    scientific_started = False
    preflight: Any = None
    authorities: Mapping[str, Any] | None = None
    try:
        preflight = _load_preflight_module(root)
        contract, _, current_authorities, v3_authority, preflight_authority = _validate_v4_authority(root, preflight)
        authorities = _mapping(contract.get("authorities"), "V4 contract authorities")
        device, device_facts = preflight._cuda_device_facts()
        accepted_device = _mapping(preflight_authority["report"].get("device"), "accepted preflight device")
        _require(tuple(device_facts.get("compute_capability", ())) == EXPECTED_COMPUTE_CAPABILITY, "target compute capability differs from accepted V4 preflight")
        _require(int(device_facts.get("vram_bytes", 0)) >= EXPECTED_VRAM_BYTES, "target VRAM is below accepted V4 preflight target")
        _require(device_facts.get("compute_capability") == accepted_device.get("compute_capability") and int(device_facts.get("vram_bytes", 0)) == int(accepted_device.get("vram_bytes", 0)), "target device differs from accepted V4 preflight")
        free_disk = shutil.disk_usage(target.parent).free
        _require(free_disk >= MIN_FREE_BYTES_FOR_OUTPUT, f"insufficient output storage headroom: {free_disk}")

        images, labels, train_indices, development_indices, selection_indices, dataset = _load_roles(root, preflight)
        selection_images = images.index_select(0, selection_indices).contiguous()
        selection_labels = labels.index_select(0, selection_indices).contiguous()
        sampler_initial, schedule, schedule_digest = _materialize_schedule(train_indices, labels)
        v2_contract = _load_v2_contract_before_seed(root)

        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        consumer = LightGen2DComplexOLSTrainingConsumerV2(v2_contract, device=device, propagation_grid_size=PRODUCTION_GRID)
        fresh_state = _runtime_state(consumer, preflight)
        _require(_digest_tree(fresh_state) == INITIAL_RUNTIME_STATE_SHA256, "fresh runtime does not reproduce original zero-phase identity")
        _require(torch.equal(fresh_state["generator.phase"], torch.zeros(PHASE_SHAPE, dtype=torch.float32)), "fresh generator is not original zero phase")
        zero_generator_sha256 = tensor_sha256(fresh_state["generator.phase"])
        expected_zero_sha = contract["authorities"]["original_zero_phase_generator"]["tensor_sha256"]
        _require(zero_generator_sha256 == expected_zero_sha, "original zero-phase generator trust boundary changed")
        initial_rng = _capture_rng(preflight)
        initial_rng_digest = _digest_tree(initial_rng)
        expected_v3_rng = contract["authorities"]["v3_scientific"]["initial_state"]["rng_state_sha256"]
        _require(initial_rng_digest == expected_v3_rng, "scientific initial RNG identity differs from the frozen zero-phase authority")
        control_checkpoint, control_state = _load_v3_control_state(root, preflight)
        fixed_state = dict(control_state)
        fixed_state["generator.phase"] = fresh_state["generator.phase"].clone()
        _configure_fixed_consumer(consumer, fixed_state, preflight)
        fixed_initial_state_digest = _digest_tree(fixed_state)
        _restore_rng(initial_rng, preflight)
        parameter_manifest = {"trainable_parameter_names": [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad], "frozen_parameter_names": [name for name, parameter in consumer.runtime.named_parameters() if not parameter.requires_grad], "trainable_parameter_count": sum(parameter.numel() for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad), "encoder_frozen_parameter_count": sum(parameter.numel() for name, parameter in consumer.runtime.named_parameters() if name == "encoder.phase"), "ols_frozen_state_numel": sum(value.numel() for name, value in consumer.runtime.state_dict().items() if name.startswith("ols."))}
        _require(parameter_manifest["trainable_parameter_names"] == ["generator.phase"] and parameter_manifest["frozen_parameter_names"] == ["encoder.phase"], "scientific parameter firewall changed")
        _require(parameter_manifest["trainable_parameter_count"] == 20_000 and parameter_manifest["encoder_frozen_parameter_count"] == 20_000 and parameter_manifest["ols_frozen_state_numel"] == 2_010_000, "scientific parameter counts changed")
        frozen, smooth, evaluator_state = _load_evaluators(root, preflight, authorities, device)

        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "preflight": {"report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256}}, "entrypoint": payload["entrypoint"], "v3_comparator": {"report": {"path": V3_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V3_SCIENTIFIC_REPORT_SHA256}, "lineage": {"path": V3_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V3_SCIENTIFIC_LINEAGE_SHA256}, "bootstrap": {"path": V3_BOOTSTRAP_PATH.as_posix(), "sha256": V3_BOOTSTRAP_SHA256}}, "protocol": {"objective": "L_generator=L_image", "trainable": ["generator.phase"], "frozen": ["encoder", "OLS", "modal_basis", "T=I_100", "W_OLS", "relay", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"], "scientific_steps": SCIENTIFIC_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "logical_batch_size": LOGICAL_BATCH_SIZE, "propagation_grid": PRODUCTION_GRID, "initialization_seed": INITIALIZATION_SEED, "schedule_seed": SCHEDULE_SEED, "development_selection_seed": DEVELOPMENT_SELECTION_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "cached_z": False, "canonical_z_identity": "verified by accepted V4 preflight"}, "initialization": {"v3_control_encoder_tensor_sha256": tensor_sha256(fixed_state["encoder.phase"]), "original_zero_generator_tensor_sha256": zero_generator_sha256, "fixed_initial_state_sha256": fixed_initial_state_digest, "initial_rng_sha256": initial_rng_digest}, "dataset": dataset, "schedule": {"seed": SCHEDULE_SEED, "steps": SCIENTIFIC_STEPS, "initial_state_sha256": _digest_tree(sampler_initial), "schedule_digest": schedule_digest, "first_batch_indices_sha256": tensor_sha256(schedule[0][0]), "last_batch_indices_sha256": tensor_sha256(schedule[-1][0])}, "claim_boundary": payload["claim_boundary"], "input_self_hash": "not_embedded_to_avoid_self_reference"}
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V4 scientific input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload.update({"contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "v2_contract": {"path": V2_CONTRACT_PATH.as_posix(), "sha256": V2_CONTRACT_SHA256}, "preflight": {"report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256}, "input": {"path": PREFLIGHT_INPUT_PATH.as_posix(), "sha256": PREFLIGHT_INPUT_SHA256}}, "authorities": {"v4_contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "v4_preflight_report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "v4_preflight_lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256}, "v3_control_checkpoint": {"path": V3_CONTROL_CHECKPOINT_PATH.as_posix(), "sha256": V3_CONTROL_CHECKPOINT_SHA256, "step": 300}, "v3_scientific_report": {"path": V3_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V3_SCIENTIFIC_REPORT_SHA256}, "v3_scientific_lineage": {"path": V3_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V3_SCIENTIFIC_LINEAGE_SHA256}, "v3_bootstrap": {"path": V3_BOOTSTRAP_PATH.as_posix(), "sha256": V3_BOOTSTRAP_SHA256}, "runtime": current_authorities.get("aligned_runtime"), "smooth_evaluator": authorities.get("evaluators", {}).get("smooth_checkpoint")}, "device": device_facts, "storage": {"free_bytes_before_run": int(free_disk), "minimum_required_bytes": MIN_FREE_BYTES_FOR_OUTPUT}, "dataset": dataset, "schedule": {"seed": SCHEDULE_SEED, "steps": SCIENTIFIC_STEPS, "initial_state_sha256": _digest_tree(sampler_initial), "schedule_digest": schedule_digest, "first_batch_indices_sha256": tensor_sha256(schedule[0][0]), "last_batch_indices_sha256": tensor_sha256(schedule[-1][0])}, "initial_state": {"fresh_runtime_state_sha256": _digest_tree(fresh_state), "fixed_initial_runtime_state_sha256": fixed_initial_state_digest, "rng_state_sha256": initial_rng_digest, "parameter_manifest": parameter_manifest, "zero_phase_generator": True}, "input": input_ref})
        scientific_started = True

        optimizer = preflight._create_generator_optimizer(consumer)
        sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=SCIENTIFIC_STEPS)
        sampler.load_state_dict(sampler_initial)
        traces: list[dict[str, Any]] = []
        checkpoints: dict[str, Any] = {}
        evaluations: dict[str, Any] = {}
        started_run = time.perf_counter()
        for step, (indices, batch_labels) in enumerate(schedule, start=1):
            sampled_indices, sampled_labels = sampler.next_batch()
            _require(torch.equal(sampled_indices, indices) and torch.equal(sampled_labels, batch_labels), f"TRAIN schedule diverged at step {step}")
            batch_images = images.index_select(0, indices).to(device)
            try:
                trace, _ = _run_training_step(consumer, optimizer, batch_images, indices, batch_labels, step, preflight)
            except torch.cuda.OutOfMemoryError as error:
                raise ScientificBlocked("BLOCKED_V4_SCIENTIFIC_TARGET_OUT_OF_MEMORY", f"step_{step}", "scientific target device ran out of memory", error_type=type(error).__name__, error=str(error)) from error
            traces.append(trace)
            del batch_images
            if step in CHECKPOINT_STEPS:
                checkpoint_payload = {"schema": CHECKPOINT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "preflight": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "checkpoint_step": step, "primary_endpoint": step == 300, "objective": "L_generator=L_image", "fixed_initial_runtime_state_sha256": fixed_initial_state_digest, "initial_rng_sha256": initial_rng_digest, "schedule_digest": schedule_digest, "runtime_state": _runtime_state(consumer, preflight), "optimizer_state": _cpu_tree(optimizer.state_dict()), "sampler_state": _cpu_tree(sampler.state_dict()), "rng_state": _cpu_tree(_capture_rng(preflight)), "trainable_parameter_names": ["generator.phase"], "frozen_parameter_names": ["encoder.phase"], "data_roles": {"train": True, "development": False, "calibration": False, "protected": False, "test": False, "sealed": False}, "scientific_training_checkpoint": True, "training_authorized": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "formal_promotion": False}
                checkpoint_path = target / f"lightgen_2d_fixed_ols_generator_recoverability_v4_step_{step}_checkpoint.v1.pt"
                atomic_torch_save_new(checkpoint_path, checkpoint_payload, artifact_kind=f"V4 step-{step} scientific checkpoint")
                checkpoint_sha = file_sha256(checkpoint_path)
                loaded_checkpoint = _mapping(safe_torch_load(checkpoint_path, expected_sha256=checkpoint_sha, map_location="cpu"), f"V4 step-{step} checkpoint")
                _require(loaded_checkpoint.get("schema") == CHECKPOINT_SCHEMA and loaded_checkpoint.get("checkpoint_step") == step, f"V4 step-{step} checkpoint identity changed")
                checkpoints[str(step)] = {"path": _relative(root, checkpoint_path), "sha256": checkpoint_sha, "step": step, "loaded": True, "scientific_training_checkpoint": True}
                evaluation = _evaluate_checkpoint(consumer, frozen, smooth, selection_images, selection_labels, step=step, train_trace=trace)
                panel_ref = _save_panel(root, target, evaluation, selection_indices, selection_labels)
                evaluation["raw_panel"]["artifact"] = panel_ref
                public = {key: value for key, value in evaluation.items() if not key.startswith("_")}
                evaluations[str(step)] = public
                if step == 300:
                    endpoint_correct = evaluation["_smooth_correct"].clone()
                del evaluation
                torch.cuda.empty_cache()
            if step % 25 == 0:
                print(json.dumps({"event": "scientific_progress", "step": step, "elapsed_seconds": time.perf_counter() - started_run}, ensure_ascii=False), flush=True)

        _require(set(checkpoints) == {str(step) for step in CHECKPOINT_STEPS} and set(evaluations) == {str(step) for step in CHECKPOINT_STEPS}, "scientific checkpoint/evaluation set is incomplete")
        provider_after = {"frozen_state_sha256": _digest_tree(_cpu_tree(frozen.state_dict())), "smooth_state_sha256": _digest_tree(_cpu_tree(smooth.state_dict()))}
        expected_provider_state = {key: evaluator_state[key] for key in ("frozen_state_sha256", "smooth_state_sha256")}
        _require(provider_after == expected_provider_state, "evaluator state changed during scientific execution")
        _require(all(parameter.grad is None for evaluator in (frozen, smooth) for parameter in evaluator.parameters()), "evaluator received a scientific gradient")
        v3_control = v3_authority["control_eval"]
        bootstrap = _paired_bootstrap(root, target, v3_authority["control_correct"], endpoint_correct, selection_labels)
        decision = _decision(v3_control, evaluations["300"], bootstrap)
        stage_a_firewall = {"metric": "SmoothMNISTEvaluator macro recall on fixed 1000-row DEV selection", "threshold": STAGE_A_FIREWALL, "v4_step_300_macro_recall": evaluations["300"]["smooth_evaluator"]["macro_recall"], "passed": evaluations["300"]["smooth_evaluator"]["macro_recall"] >= STAGE_A_FIREWALL, "classification": "PASSED" if evaluations["300"]["smooth_evaluator"]["macro_recall"] >= STAGE_A_FIREWALL else "FAILED", "stage_b_authorized": False}
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": decision["classification"], "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_comparison_executed": True}, "trainable_frozen_firewall": {"parameter_manifest": parameter_manifest, "encoder_unchanged": all(trace["encoder_unchanged"] for trace in traces), "ols_unchanged": all(trace["ols_unchanged"] for trace in traces), "evaluator_unchanged": True, "semantic_provider_gradient": False}, "objective": "L_generator=L_image", "trajectory": {step: evaluations[str(step)] for step in CHECKPOINT_STEPS}, "traces": traces, "checkpoints": checkpoints, "bootstrap": bootstrap, "comparison": decision, "stage_a_firewall": stage_a_firewall, "governance": {"scientific_training_executed": True, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping": False, "sampled_ols_generation": False, "generation_executed": False, "generation_acceptance": False, "protected_read": False, "test_read": False, "sealed_read": False, "formal_promotion": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False}, "next": "no_retry; retain V4 scientific result; Stage B remains unauthorized"})
        del optimizer, consumer, frozen, smooth
        torch.cuda.empty_cache()
    except ScientificBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no_retry_without_new_authority"})
    except InvalidScientific as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V4_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; repair requires new authority"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V4_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no_retry_without_new_authority"})
    finally:
        try:
            ending = git_facts(root)
            payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "status_lines_unchanged": started.get("status_lines") == ending.get("status_lines")})
        except Exception as error:
            payload["provenance"]["git_end_error"] = {"type": type(error).__name__, "message": str(error)}
        payload["claim_boundary"].update({"scientific_training_executed": bool(payload["claim_boundary"].get("scientific_training_executed") or scientific_started), "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "sampled_ols_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload.setdefault("artifacts", {})
        payload["artifacts"]["input"] = input_ref
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V4 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "entrypoint": payload["entrypoint"], "authorities": payload.get("authorities"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V4 scientific lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_training_executed": payload["claim_boundary"]["scientific_training_executed"], "scientific_comparison_executed": payload["claim_boundary"]["scientific_comparison_executed"], "stage_b_authorized": False, "stage_b_executed": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# These are set only for the one scientific consumer call; keeping the helper
# signatures explicit would otherwise require threading the module through all
# checkpoint and RNG helpers.
preflight_module_placeholder: Any = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("scientific",), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    global preflight_module_placeholder
    # The module is loaded again by run_scientific for authority validation; the
    # placeholder is filled before any scientific update is allowed.
    preflight_module_placeholder = _load_preflight_module((args.repo_root or Path(__file__).resolve().parents[3]).resolve())
    result = run_scientific(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if result["result"] == "VALID_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
