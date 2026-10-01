"""Execute the single authorized V5 paired scientific comparison.

This consumer resolves the READY V5 contract and accepted attempt-05
engineering preflight, reconstructs the frozen TRAIN-only prototype geometry
from the initial V3 CONTROL encoder, and runs exactly one matched 300-step
control/candidate comparison.  It never enters Stage B, extracts a formal
scale, shapes a prior, samples generation, reads protected/test/sealed roles,
or retries after scientific execution starts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, file_sha256, git_facts, tensor_sha256
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load


TASK_ID = "EXECUTE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5"
OUTER_GATE = "V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED"
REPORT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1"
INPUT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_lineage.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_checkpoint.v1"
PANEL_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_raw_cmos_panel.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_paired_bootstrap.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_complex_ols_semantic_alignment_v5_sufficient_observations.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "9a72e0bcf2de12ab9a0caa7151f273de76768dc7"
CONTRACT_PATH = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01/"
    "lightgen_2d_complex_ols_semantic_alignment_contract.v5.json"
)
CONTRACT_SHA256 = "e2d57fa96e0a020abf23cbd889188d74d4893a6e9c48ac7a1d24d7156cf21b61"
PREFLIGHT_PATH = Path("scripts/research/lightgen/execute_lightgen_2d_complex_ols_semantic_alignment_preflight_v5.py")
PREFLIGHT_SHA256 = "a062e2acb4e624969f09d3631adab2ede22a786c7dde64e52e045a2c76f8c98a"
PREFLIGHT_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05"
)
PREFLIGHT_INPUT_PATH = PREFLIGHT_DIR / "lightgen_2d_complex_ols_semantic_alignment_preflight_input.v1.json"
PREFLIGHT_INPUT_SHA256 = "9c734a76b5ceb03fe6405faadfaf7d2fa27eefe21d5a077ab3f0e46d926569f8"
PREFLIGHT_REPORT_PATH = PREFLIGHT_DIR / "lightgen_2d_complex_ols_semantic_alignment_preflight_report.v1.json"
PREFLIGHT_REPORT_SHA256 = "7fab2789b36491597462ce092bc98dc5784cc269245a2cab3b07a4878543c551"
PREFLIGHT_LINEAGE_PATH = PREFLIGHT_DIR / "lightgen_2d_complex_ols_semantic_alignment_preflight_lineage.v1.json"
PREFLIGHT_LINEAGE_SHA256 = "e5d72fce35784ea3c25fc98a5c4a7ad20538594f491f9065123a94f5892387dc"

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v5/runs/"
    "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01"
)
INPUT_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_input.v1.json"
REPORT_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1.json"
LINEAGE_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_scientific_lineage.v1.json"
BOOTSTRAP_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_paired_bootstrap.v1.pt"
SUFFICIENT_NAME = "lightgen_2d_complex_ols_semantic_alignment_v5_sufficient_observations.v1.pt"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEPS = (100, 200, 300)
INITIALIZATION_SEED = 1234
SCHEDULE_SEED = 1234
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 999
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
EPSILON_DENOMINATOR = 1.0e-12
STAGE_A_FIREWALL = 0.937
SMOOTH_MARGIN = 0.05
SMOOTH_PER_CLASS_DROP = 0.10
IMAGE_LOSS_TOLERANCE = 0.005
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000

EXPECTED_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
EXPECTED_OLS_TRANSPORT_SHA256 = "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293"
EXPECTED_INITIAL_STATE_SHA256 = "b7b0745c0e03fe12770fd0d0f64d6e6e50aedd6505c975098e012a10149929e1"
EXPECTED_INITIAL_RNG_SHA256 = "f0968737c0f97f0a9ea3bb7ca9b9da0a699025b609fc2e1e646caf117f5bbe04"
EXPECTED_Z_TRAIN_SHA256 = "b19ac224a29d93180bf6b56edea8e3ba12e41600c33178ceb30d047e1bac1703"
EXPECTED_CANONICAL_Z_SHA256 = "6f51ca85edfe89db1d854331fa2732173f4c28a78c70d0d54041b5bf04de8c42"
EXPECTED_PROTOTYPES_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
EXPECTED_PAIR_DISTANCE_SHA256 = "85c1514fd20934b425120327fa900cc47b764ed58000edbb065107a9741166f5"
FROZEN_S_MU = 0.00555040966719389
FROZEN_G_IMAGE = 0.0005046187151008359
FROZEN_G_PROTO = 0.005270976801389226
FROZEN_LAMBDA = 0.09573533218209243


class InvalidScientific(RuntimeError):
    """The frozen scientific harness or authority is invalid."""


class ScientificBlocked(RuntimeError):
    """A valid runtime boundary prevents this one-shot execution."""

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
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute authority path is not allowed: {value}")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidScientific(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")


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
                maximum = max(maximum, float((first.to(torch.complex128) - second.to(torch.complex128)).abs().max()))
        elif isinstance(first, Mapping) and isinstance(second, Mapping):
            maximum = max(maximum, _max_state_difference(first, second))
        elif first != second:
            return math.inf
    return maximum


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


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result), f"{name} norm is nonfinite")
    return result


def _status_lines(root: Path) -> list[str]:
    completed = subprocess.run(["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.splitlines()


def _load_preflight_module(root: Path) -> Any:
    path = _path(root, PREFLIGHT_PATH)
    _validate_hash(path, PREFLIGHT_SHA256, "accepted V5 preflight entrypoint")
    spec = importlib.util.spec_from_file_location("accepted_v5_preflight_for_scientific", path)
    _require(spec is not None and spec.loader is not None, "accepted V5 preflight module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_accepted_preflight(root: Path, preflight: Any) -> tuple[dict[str, Any], Mapping[str, Any], Any, Mapping[str, Any], Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    contract, authorities, contract_sha = preflight._load_contract(root)
    _require(_path(root, CONTRACT_PATH).is_file(), "V5 contract is unavailable")
    _require(contract.get("result") == "READY" and contract.get("status") == "valid_completed", "V5 contract is not READY")
    _require(contract.get("outer_gate") == OUTER_GATE, "V5 outer gate changed")
    v3, v4, v2_contract = preflight._validate_authority_payloads(root, contract, authorities)
    for path, digest, name in (
        (PREFLIGHT_INPUT_PATH, PREFLIGHT_INPUT_SHA256, "accepted V5 preflight input"),
        (PREFLIGHT_REPORT_PATH, PREFLIGHT_REPORT_SHA256, "accepted V5 preflight report"),
        (PREFLIGHT_LINEAGE_PATH, PREFLIGHT_LINEAGE_SHA256, "accepted V5 preflight lineage"),
    ):
        _validate_hash(_path(root, path), digest, name)
    report = _read_json(_path(root, PREFLIGHT_REPORT_PATH), "accepted V5 preflight report")
    lineage = _read_json(_path(root, PREFLIGHT_LINEAGE_PATH), "accepted V5 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "accepted V5 preflight is not AUTHORIZABLE")
    _require(report.get("classification") == "LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5_AUTHORIZABLE", "accepted V5 preflight classification changed")
    _require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("status") == "valid_completed", "accepted V5 preflight lineage changed")
    preflight_input = _mapping(report.get("preflight_input"), "accepted V5 preflight input")
    _require(preflight_input.get("path") == PREFLIGHT_INPUT_PATH.as_posix(), "accepted V5 preflight input path changed")
    entrypoint = _mapping(report.get("entrypoint"), "accepted V5 preflight entrypoint")
    _require(Path(str(entrypoint.get("path", ""))).name == PREFLIGHT_PATH.name, "accepted V5 preflight entrypoint is not the expected tool")
    boundary = _mapping(report.get("claim_boundary"), "accepted V5 preflight claim boundary")
    for key in ("scientific_training_executed", "scientific_comparison_executed", "stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "protected_accessed", "test_accessed", "sealed_accessed"):
        _require(boundary.get(key) is False, f"accepted preflight opened boundary: {key}")
    prototypes = _mapping(report.get("prototypes"), "accepted V5 prototypes")
    _require(prototypes.get("source_role") == "TRAIN only" and prototypes.get("rows") == 55_000, "accepted prototype TRAIN role changed")
    _require(prototypes.get("latent_domain") == "full complex C^100" and prototypes.get("prototypes_requires_grad") is False, "accepted prototype domain changed")
    scale = _mapping(report.get("prototype_scale"), "accepted prototype scale")
    _require(math.isclose(float(scale.get("s_mu")), FROZEN_S_MU, rel_tol=0.0, abs_tol=1.0e-15), "accepted s_mu changed")
    calibration = _mapping(report.get("lambda_calibration"), "accepted lambda calibration")
    _require(math.isclose(float(calibration.get("g_image")), FROZEN_G_IMAGE, rel_tol=0.0, abs_tol=1.0e-15), "accepted g_image changed")
    _require(math.isclose(float(calibration.get("g_proto")), FROZEN_G_PROTO, rel_tol=0.0, abs_tol=1.0e-15), "accepted g_proto changed")
    _require(math.isclose(float(calibration.get("lambda_proto")), FROZEN_LAMBDA, rel_tol=0.0, abs_tol=1.0e-15), "accepted lambda changed")
    _require(_mapping(report.get("matched_arms"), "accepted matched arms").get("initial_state_equal") is True, "accepted arms are not matched")
    _require(_mapping(report.get("matched_arms"), "accepted matched arms").get("initial_rng_equal") is True, "accepted arm RNG is not matched")
    for name, arm in _mapping(report.get("arms"), "accepted preflight arms").items():
        arm_mapping = _mapping(arm, f"accepted {name} arm")
        _require(arm_mapping.get("frozen_generator_unchanged") is True, f"accepted {name} generator firewall changed")
        _require(arm_mapping.get("ols_unchanged") is True, f"accepted {name} OLS firewall changed")
        resume = _mapping(arm_mapping.get("checkpoint_resume"), f"accepted {name} checkpoint resume")
        for key in ("next_batch_equal", "optimizer_equal"):
            _require(resume.get(key) is True, f"accepted {name} checkpoint {key} changed")
        _require(float(resume.get("trace_max_abs_difference")) == 0.0 and float(resume.get("runtime_state_max_abs_difference")) == 0.0, f"accepted {name} checkpoint runtime identity changed")
    smoke = _mapping(report.get("production_cuda_no_data_smoke"), "accepted production CUDA smoke")
    _require(smoke.get("executed") is True and smoke.get("grid") == PRODUCTION_GRID and smoke.get("runtime_state_unchanged") is True, "accepted production CUDA smoke changed")
    synthetic = _mapping(report.get("synthetic_acceptance"), "accepted synthetic exact-core acceptance")
    _require(synthetic.get("synthetic_only") is True and synthetic.get("scientific_meaning") == "none; synthetic acceptance only", "synthetic acceptance boundary changed")
    sufficient = _mapping(synthetic.get("sufficient_observations"), "accepted synthetic sufficient observations")
    _require(sufficient.get("reloaded") is True and sufficient.get("schema_valid") is True, "accepted synthetic sufficient observations did not reload")
    data_reads = _mapping(report.get("data_role_reads"), "accepted preflight data-role reads")
    _require(_mapping(data_reads.get("DEV"), "accepted DEV role reads").get("scientific_values_read") == 0, "accepted preflight consumed DEV scientific values")
    _require(data_reads.get("protected") == 0 and data_reads.get("test") == 0 and data_reads.get("sealed") == 0, "accepted preflight opened forbidden data role")
    accepted = {
        "input": {"path": PREFLIGHT_INPUT_PATH.as_posix(), "sha256": PREFLIGHT_INPUT_SHA256},
        "report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256},
        "lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256},
        "entrypoint": {"path": PREFLIGHT_PATH.as_posix(), "sha256": PREFLIGHT_SHA256},
        "result": report.get("result"),
        "classification": report.get("classification"),
    }
    authorities = dict(authorities)
    authorities["accepted_preflight"] = accepted
    return dict(contract), authorities, v2_contract, v3, v4, dict(report), dict(lineage)


def _load_roles(root: Path, preflight: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, train_indices, source_metadata = preflight._load_train_role(root)
    _require(file_sha256(_path(root, TRAINING_SOURCE_PATH)) == TRAINING_SOURCE_SHA256, "scientific TRAIN source changed")
    expected_train, development_indices = preflight.stratified_mnist_indices(labels, seed=INITIALIZATION_SEED, validation_per_label=500)
    _require(torch.equal(train_indices, expected_train), "TRAIN split differs from accepted preflight")
    _require(development_indices.numel() == 5_000, "DEV role size changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV roles overlap or omit rows")
    _require(torch.equal(torch.bincount(labels.index_select(0, development_indices), minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), 500, dtype=torch.long)), "DEV class counts changed")
    metadata = {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000, "values_read": True},
        "train": {"rows": 55_000, "counts_by_class": source_metadata["train"]["counts_by_class"], "indices_sha256": tensor_sha256(train_indices)},
        "development": {"rows": 5_000, "evaluation_rows": 5_000, "per_class": 500, "indices_sha256": tensor_sha256(development_indices), "values_read_for": "evaluation only"},
        "calibration": {"rows_read": 0},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "official_test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, metadata


def _materialize_schedule(train_indices: torch.Tensor, labels: torch.Tensor, preflight: Any) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]], str]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=SCIENTIFIC_STEPS)
    initial = _cpu_tree(sampler.state_dict())
    schedule: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(1, SCIENTIFIC_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(CLASS_COUNT, dtype=torch.long)), f"TRAIN schedule labels changed at step {step}")
        schedule.append((indices.cpu().clone(), batch_labels.cpu().clone()))
    identity = {"initial": initial, "batch_indices": [item[0] for item in schedule], "batch_labels": [item[1] for item in schedule]}
    return initial, schedule, _digest_tree(identity)


def _optimizer(consumer: Any, preflight: Any) -> torch.optim.Adam:
    optimizer = preflight._optimizer(consumer)
    _require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == 1, "scientific optimizer surface changed")
    return optimizer


def _optimizer_manifest() -> dict[str, Any]:
    return {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "trainable_parameter_names": ["encoder.phase"]}


def _load_smooth(root: Path, authorities: Mapping[str, Any], device: torch.device, preflight: Any) -> tuple[Any, dict[str, Any]]:
    evaluator, identity = preflight._load_smooth(root, authorities)
    evaluator.to(device)
    evaluator.freeze()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator firewall failed")
    return evaluator, identity


def _raw_summary(raw: torch.Tensor) -> dict[str, Any]:
    value = raw.detach().cpu().contiguous()
    _finite(value, "raw CMOS")
    per_image_variance = value.flatten(1).var(dim=1, unbiased=False)
    across_sample_variance = value.flatten(1).var(dim=0, unbiased=False)
    finite = bool(torch.isfinite(value).all())
    structural_valid = finite and tuple(value.shape[1:]) == (1, 200, 200) and value.dtype == torch.float32
    structured = bool((per_image_variance > 0.0).all() and float(across_sample_variance.mean()) > 0.0)
    nonconstant = bool(float(value.std(unbiased=False)) > 0.0)
    _require(structural_valid and structured and nonconstant, "scientific raw CMOS is invalid or constant")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite, "structural_valid": structural_valid, "structured": structured, "nonconstant": nonconstant, "mean": float(value.mean()), "std": float(value.std(unbiased=False)), "per_image_variance_mean": float(per_image_variance.mean()), "across_sample_variance_mean": float(across_sample_variance.mean()), "tensor_sha256": tensor_sha256(value)}


def _execute_step(
    consumer: Any,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    s_mu: float,
    *,
    arm: str,
    step: int,
    preflight: Any,
) -> dict[str, Any]:
    _require(arm in {"control", "candidate"}, "scientific arm identity is invalid")
    named = dict(consumer.runtime.named_parameters())
    _require(set(named) == {"encoder.phase", "generator.phase"}, "scientific parameter surface changed")
    before_state = _cpu_tree(consumer.runtime.state_dict())
    before_generator = before_state["generator.phase"]
    before_ols = {key: before_state[key] for key in ("ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport")}
    before_encoder = before_state["encoder.phase"]
    optimizer.zero_grad(set_to_none=True)
    for parameter in named.values():
        parameter.grad = None
    result = consumer.runtime.forward_from_image(batch)
    raw = preflight._validate_result(result, int(batch.shape[0]), require_grad=True)
    image_loss = (raw - result.image_input.slm_intensity).square().mean()
    _finite(image_loss, f"{arm} TRAIN image loss")
    if arm == "candidate":
        proto_loss = preflight.prototype_loss(result.z_ols.coefficients, labels, prototypes, s_mu)
        total_loss = image_loss + FROZEN_LAMBDA * proto_loss
    else:
        with torch.no_grad():
            proto_loss = preflight.prototype_loss(result.z_ols.coefficients.detach(), labels, prototypes, s_mu)
        total_loss = image_loss
    _finite(proto_loss, f"{arm} TRAIN prototype loss")
    _finite(total_loss, f"{arm} TRAIN total loss")
    total_loss.backward()
    encoder_gradient = named["encoder.phase"].grad
    _require(encoder_gradient is not None, f"{arm} encoder.phase gradient is missing")
    encoder_gradient_norm = _norm(encoder_gradient, f"{arm} encoder.phase gradient")
    _require(encoder_gradient_norm > 0.0, f"{arm} encoder.phase gradient is zero")
    _require(named["generator.phase"].grad is None, f"{arm} frozen generator received a gradient")
    _require(all(parameter.grad is None for name, parameter in named.items() if name != "encoder.phase"), f"{arm} forbidden parameter received a gradient")
    optimizer.step()
    torch.cuda.synchronize(consumer.device)
    after_state = _cpu_tree(consumer.runtime.state_dict())
    encoder_update_norm = float((after_state["encoder.phase"] - before_encoder).to(torch.float64).square().sum().sqrt())
    _require(math.isfinite(encoder_update_norm) and encoder_update_norm > 0.0, f"{arm} encoder update is zero or nonfinite")
    _require(torch.equal(after_state["generator.phase"], before_generator), f"{arm} generator.phase changed")
    _require(all(torch.equal(after_state[key], before_ols[key]) for key in before_ols), f"{arm} OLS state changed")
    trace = {
        "step": step,
        "arm": arm,
        "objective": "L_image" if arm == "control" else "L_image+0.09573533218209243*L_proto",
        "losses": {"L_image": float(image_loss.detach().cpu()), "L_proto": float(proto_loss.detach().cpu()), "L_total": float(total_loss.detach().cpu())},
        "lambda_proto": 0.0 if arm == "control" else FROZEN_LAMBDA,
        "gradient_norms": {"encoder.phase": encoder_gradient_norm, "generator.phase": None},
        "update_norms": {"encoder.phase": encoder_update_norm, "generator.phase": 0.0},
        "encoder_changed": True,
        "generator_unchanged": True,
        "ols_unchanged": True,
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase"],
        "runtime_state_sha256_after": _digest_tree(after_state),
    }
    del result, raw, image_loss, proto_loss, total_loss
    return trace


def _margin_summary(margins: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    values = margins.to(torch.float64).cpu()
    _finite(values, "OLS margin")
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        _require(int(mask.sum()) == 500, f"DEV class count changed for OLS margin {digit}")
        per_class.append({"class": digit, "mean": float(values[mask].mean()), "median": float(values[mask].median()), "positive_fraction": float((values[mask] > 0.0).to(torch.float64).mean())})
    return {"mean": float(values.mean()), "median": float(values.median()), "positive_fraction": float((values > 0.0).to(torch.float64).mean()), "min": float(values.min()), "max": float(values.max()), "per_class": per_class}


def _evaluate_checkpoint(
    consumer: Any,
    smooth: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    s_mu: float,
    *,
    arm: str,
    step: int,
    preflight: Any,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    raw_parts: list[torch.Tensor] = []
    smooth_correct_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    image_sum = 0.0
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
                batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
                batch_labels = labels[start : start + LOGICAL_BATCH_SIZE]
                result = consumer.runtime.forward_from_image(batch)
                raw = preflight._validate_result(result, int(batch.shape[0]), require_grad=False)
                image_loss = (raw - result.image_input.slm_intensity).square().mean()
                _finite(image_loss, f"{arm} DEV image loss")
                smooth_logits = preflight._smooth_logits(smooth, raw)
                margins = preflight.ols_margin(result.z_ols.coefficients, batch_labels.to(consumer.device), prototypes, s_mu)
                smooth_correct_parts.append(smooth_logits.argmax(dim=1).eq(batch_labels).cpu().contiguous())
                margin_parts.append(margins.cpu().contiguous())
                raw_parts.append(raw.detach().cpu().contiguous())
                image_sum += float(image_loss.cpu()) * int(batch.shape[0])
                del batch, batch_labels, result, raw, image_loss, smooth_logits, margins
    finally:
        consumer.runtime.train(was_training)
    raw_cpu = torch.cat(raw_parts, dim=0)
    smooth_correct = torch.cat(smooth_correct_parts, dim=0)
    margins = torch.cat(margin_parts, dim=0).to(torch.float64)
    _require(tuple(raw_cpu.shape) == (5_000, 1, 200, 200), f"{arm} DEV raw panel shape changed")
    _require(tuple(smooth_correct.shape) == (5_000,) and tuple(margins.shape) == (5_000,), f"{arm} DEV observation shape changed")
    raw_summary = _raw_summary(raw_cpu)
    per_class_recall = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        _require(int(mask.sum()) == 500, f"DEV class count changed for Smooth {digit}")
        per_class_recall.append(float(smooth_correct[mask].to(torch.float64).mean()))
    macro_recall = float(sum(per_class_recall) / CLASS_COUNT)
    return {
        "step": step,
        "arm": arm,
        "losses": {"L_image": image_sum / 5_000.0},
        "smooth": {"macro_recall": macro_recall, "per_class_recall": per_class_recall},
        "ols_margin": _margin_summary(margins, labels),
        "raw_panel": raw_summary,
        "_raw_cpu": raw_cpu,
        "_smooth_correct": smooth_correct,
        "_ols_margin_values": margins,
    }


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, rng_state: Mapping[str, Any], initial_state: Mapping[str, Any], prototypes: torch.Tensor, s_mu: float, arm: str, step: int) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "arm": arm,
        "checkpoint_step": step,
        "next_step": step + 1,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(rng_state),
        "initial_runtime_state_sha256": _digest_tree(initial_state),
        "prototypes": prototypes.detach().cpu().clone(),
        "prototypes_sha256": tensor_sha256(prototypes),
        "s_mu": FROZEN_S_MU,
        "lambda_proto": 0.0 if arm == "control" else FROZEN_LAMBDA,
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"],
        "data_roles": {"train": True, "development": True, "protected": False, "official_test": False, "sealed": False},
        "stage_b_executed": False,
        "scale_extraction_executed": False,
        "prior_generation_executed": False,
        "sampled_generation_executed": False,
        "formal_promotion": False,
    }


def _save_checkpoint(root: Path, target: Path, payload: Mapping[str, Any], arm: str, step: int) -> dict[str, Any]:
    path = target / f"lightgen_2d_complex_ols_semantic_alignment_v5_{arm}_step_{step}_checkpoint.v1.pt"
    atomic_torch_save_new(path, payload, artifact_kind=f"V5 scientific {arm} checkpoint")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{arm} step-{step} checkpoint")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("arm") == arm and loaded.get("checkpoint_step") == step, f"{arm} checkpoint identity changed")
    _require(loaded.get("prototypes_sha256") == payload.get("prototypes_sha256") and tensor_sha256(loaded.get("prototypes")) == payload.get("prototypes_sha256"), f"{arm} checkpoint prototypes changed")
    return {"path": _relative(root, path), "sha256": digest, "step": step, "reloaded": True}


def _save_panel(root: Path, target: Path, evaluation: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    arm = str(evaluation["arm"])
    step = int(evaluation["step"])
    path = target / f"lightgen_2d_complex_ols_semantic_alignment_v5_{arm}_step_{step}_raw_cmos_panel.v1.pt"
    raw = evaluation["_raw_cpu"]
    payload = {"schema": PANEL_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "arm": arm, "checkpoint_step": step, "role": "full fixed DEV evaluation role", "labels_sha256": tensor_sha256(labels), "raw_cmos": raw}
    atomic_torch_save_new(path, payload, artifact_kind=f"V5 scientific {arm} raw-CMOS panel")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{arm} step-{step} raw-CMOS panel")
    _require(tensor_sha256(loaded.get("raw_cmos")) == evaluation["raw_panel"]["tensor_sha256"], f"{arm} step-{step} raw panel reload changed")
    return {"path": _relative(root, path), "sha256": digest, "tensor_sha256": evaluation["raw_panel"]["tensor_sha256"], "reloaded": True}


def _public_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if not key.startswith("_")}


def _run_arm(
    root: Path,
    target: Path,
    v2_contract: Any,
    initial_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    development_images: torch.Tensor,
    development_labels: torch.Tensor,
    prototypes: torch.Tensor,
    s_mu: float,
    smooth: Any,
    schedule: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    arm: str,
    device: torch.device,
    preflight: Any,
) -> dict[str, Any]:
    consumer = preflight._new_consumer(v2_contract, device, PRODUCTION_GRID)
    preflight._load_runtime_state(consumer, initial_state)
    firewall = preflight._configure_encoder_only(consumer)
    optimizer = _optimizer(consumer, preflight)
    initial_state_digest = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    _require(initial_state_digest == EXPECTED_INITIAL_STATE_SHA256, f"{arm} initial state differs from authority")
    _require(firewall["trainable_names"] == ["encoder.phase"] and firewall["frozen_names"] == ["generator.phase"], f"{arm} trainable/frozen firewall changed")
    _require(firewall["trainable_count"] == 20_000 and firewall["frozen_parameter_count"] == 20_000, f"{arm} parameter counts changed")
    preflight._restore_rng_state(initial_rng)
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=SCIENTIFIC_STEPS)
    traces: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for step in range(1, SCIENTIFIC_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        expected_indices, expected_labels = schedule[step - 1]
        _require(torch.equal(indices, expected_indices) and torch.equal(batch_labels, expected_labels), f"{arm} schedule diverged at step {step}")
        batch = images.index_select(0, indices).to(device)
        trace = _execute_step(consumer, optimizer, batch, batch_labels.to(device), prototypes.to(device), s_mu, arm=arm, step=step, preflight=preflight)
        trace["batch_indices_sha256"] = tensor_sha256(indices)
        trace["batch_labels_sha256"] = tensor_sha256(batch_labels)
        traces.append(trace)
        del batch
        if step in CHECKPOINT_STEPS:
            checkpoint_payload = _checkpoint_payload(consumer, optimizer, sampler, preflight._capture_rng_state(), initial_state, prototypes, s_mu, arm, step)
            checkpoints[str(step)] = _save_checkpoint(root, target, checkpoint_payload, arm, step)
            evaluation = _evaluate_checkpoint(consumer, smooth, development_images, development_labels, prototypes.to(device), s_mu, arm=arm, step=step, preflight=preflight)
            evaluation["raw_panel"]["artifact"] = _save_panel(root, target, evaluation, development_labels)
            evaluation["train_step_trace"] = trace
            evaluations[str(step)] = evaluation
            del evaluation["_raw_cpu"]
            torch.cuda.empty_cache()
        if step % 25 == 0:
            print(json.dumps({"event": "scientific_progress", "arm": arm, "step": step, "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False), flush=True)
    _require(set(checkpoints) == {str(step) for step in CHECKPOINT_STEPS}, f"{arm} checkpoint set is incomplete")
    _require(set(evaluations) == {str(step) for step in CHECKPOINT_STEPS}, f"{arm} evaluation set is incomplete")
    endpoint = evaluations["300"]
    return {
        "arm": arm,
        "objective": "L_control=L_image" if arm == "control" else "L_candidate=L_image+0.09573533218209243*L_proto",
        "lambda_proto": 0.0 if arm == "control" else FROZEN_LAMBDA,
        "initial_state_digest": initial_state_digest,
        "initial_rng_digest": _digest_tree(initial_rng),
        "sampler_initial_state_digest": _digest_tree(_cpu_tree(DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=SCIENTIFIC_STEPS).state_dict())),
        "optimizer_manifest": _optimizer_manifest(),
        "firewall": firewall,
        "traces": traces,
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "step300_observations": {"smooth_correct": endpoint["_smooth_correct"].clone(), "ols_margin": endpoint["_ols_margin_values"].clone()},
        "updates": SCIENTIFIC_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _per_class(values: torch.Tensor, labels: torch.Tensor) -> list[float]:
    result = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        _require(int(mask.sum()) == 500, f"DEV class count changed for class {digit}")
        result.append(float(values[mask].to(torch.float64).mean()))
    return result


def _paired_bootstrap(root: Path, target: Path, labels: torch.Tensor, control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    control_correct = control["step300_observations"]["smooth_correct"].cpu()
    candidate_correct = candidate["step300_observations"]["smooth_correct"].cpu()
    control_margin = control["step300_observations"]["ols_margin"].to(torch.float64).cpu()
    candidate_margin = candidate["step300_observations"]["ols_margin"].to(torch.float64).cpu()
    _require(control_correct.shape == candidate_correct.shape == (5_000,), "Smooth sufficient observations differ")
    _require(control_margin.shape == candidate_margin.shape == (5_000,), "OLS sufficient observations differ")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    smooth_replicates: list[torch.Tensor] = []
    margin_replicates: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 500, f"bootstrap class count changed for {digit}")
        draw = torch.randint(matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        c_correct = control_correct.index_select(0, matches)[draw].to(torch.float64).mean(dim=1)
        k_correct = candidate_correct.index_select(0, matches)[draw].to(torch.float64).mean(dim=1)
        smooth_replicates.append(k_correct - c_correct)
        delta_margin = (candidate_margin - control_margin).index_select(0, matches)[draw].mean(dim=1)
        margin_replicates.append(delta_margin)
    smooth_values = torch.stack(smooth_replicates).mean(dim=0)
    margin_values = torch.stack(margin_replicates).mean(dim=0)
    smooth_delta = candidate_correct.to(torch.float64).mean() - control_correct.to(torch.float64).mean()
    margin_delta = candidate_margin - control_margin
    smooth_quantile = torch.quantile(smooth_values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    margin_quantile = torch.quantile(margin_values, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    smooth_per_class = [float(value) for value in (torch.stack([candidate_correct[labels == digit].to(torch.float64).mean() - control_correct[labels == digit].to(torch.float64).mean() for digit in range(CLASS_COUNT)]))]
    margin_per_class = [float(value) for value in _per_class(margin_delta, labels)]
    artifact_payload = {
        "schema": BOOTSTRAP_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "endpoint_step": 300,
        "bootstrap": {"type": "paired class-stratified", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "quantile": "torch.quantile interpolation=linear", "smooth_delta_replicates": smooth_values, "ols_margin_delta_replicates": margin_values},
        "labels": labels,
        "control_smooth_correct": control_correct,
        "candidate_smooth_correct": candidate_correct,
        "control_ols_margin": control_margin,
        "candidate_ols_margin": candidate_margin,
    }
    path = target / BOOTSTRAP_NAME
    atomic_torch_save_new(path, artifact_payload, artifact_kind="V5 paired bootstrap artifact")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V5 paired bootstrap artifact")
    _require(loaded.get("schema") == BOOTSTRAP_SCHEMA and tensor_sha256(loaded.get("bootstrap", {}).get("smooth_delta_replicates")) == tensor_sha256(smooth_values), "V5 bootstrap reload changed")
    return {
        "path": _relative(root, path),
        "sha256": digest,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "smooth": {"delta_mean": float(smooth_delta), "ci_lower": float(smooth_quantile[0]), "ci_upper": float(smooth_quantile[1]), "per_class_delta": smooth_per_class, "max_class_degradation": min(smooth_per_class)},
        "ols_margin": {"delta_mean": float(margin_delta.mean()), "delta_median": float(margin_delta.median()), "ci_lower": float(margin_quantile[0]), "ci_upper": float(margin_quantile[1]), "positive_fraction": float((margin_delta > 0.0).to(torch.float64).mean()), "per_class_delta_mean": margin_per_class},
        "reloaded": True,
    }


def _save_sufficient_observations(root: Path, target: Path, labels: torch.Tensor, control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema": SUFFICIENT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "endpoint_step": 300, "role": "frozen sufficient observations for primary step-300 statistics", "scientific_meaning": "primary statistics only; no optical rerun", "labels": labels.cpu(), "control_smooth_correct": control["step300_observations"]["smooth_correct"].cpu(), "candidate_smooth_correct": candidate["step300_observations"]["smooth_correct"].cpu(), "control_ols_margin": control["step300_observations"]["ols_margin"].cpu(), "candidate_ols_margin": candidate["step300_observations"]["ols_margin"].cpu()}
    path = target / SUFFICIENT_NAME
    atomic_torch_save_new(path, payload, artifact_kind="V5 sufficient observations")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V5 sufficient observations")
    _require(loaded.get("schema") == SUFFICIENT_SCHEMA and torch.equal(loaded["labels"], labels.cpu()), "V5 sufficient observation reload changed")
    return {"path": _relative(root, path), "sha256": digest, "schema": SUFFICIENT_SCHEMA, "reloaded": True, "endpoint_step": 300}


def _decision(control: Mapping[str, Any], candidate: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    control_eval = control["evaluations"]["300"]
    candidate_eval = candidate["evaluations"]["300"]
    image_delta = float(candidate_eval["losses"]["L_image"] - control_eval["losses"]["L_image"])
    smooth_delta = float(bootstrap["smooth"]["delta_mean"])
    margin_mean = float(bootstrap["ols_margin"]["delta_mean"])
    margin_ci_lower = float(bootstrap["ols_margin"]["ci_lower"])
    smooth_per_class = [float(value) for value in bootstrap["smooth"]["per_class_delta"]]
    raw_valid = all(bool(evaluation["raw_panel"][key]) for evaluation in (control_eval, candidate_eval) for key in ("finite", "structural_valid", "structured", "nonconstant"))
    checks = {
        "smooth_delta_at_least_0_05": smooth_delta >= SMOOTH_MARGIN,
        "smooth_bootstrap_ci_lower_gt_zero": float(bootstrap["smooth"]["ci_lower"]) > 0.0,
        "ols_margin_mean_gt_zero": margin_mean > 0.0,
        "ols_margin_bootstrap_ci_lower_gt_zero": margin_ci_lower > 0.0,
        "no_smooth_per_class_drop_gt_0_10": min(smooth_per_class) >= -SMOOTH_PER_CLASS_DROP,
        "image_loss_degradation_within_0_005": image_delta <= IMAGE_LOSS_TOLERANCE,
        "raw_cmos_endpoint_panel_valid": raw_valid,
    }
    return {"endpoint_step": 300, "observed": {"smooth_delta": smooth_delta, "smooth_bootstrap_ci_lower": float(bootstrap["smooth"]["ci_lower"]), "ols_margin_delta_mean": margin_mean, "ols_margin_bootstrap_ci_lower": margin_ci_lower, "smooth_per_class_delta": smooth_per_class, "image_loss_delta": image_delta, "raw_cmos_valid": raw_valid}, "thresholds": {"smooth_delta": SMOOTH_MARGIN, "smooth_bootstrap_ci_lower": 0.0, "ols_margin_delta_mean": 0.0, "ols_margin_bootstrap_ci_lower": 0.0, "smooth_per_class_floor": -SMOOTH_PER_CLASS_DROP, "image_loss_delta_max": IMAGE_LOSS_TOLERANCE}, "checks": checks, "all_frozen_criteria_pass": all(checks.values()), "classification": SUPPORTED_CLASSIFICATION if all(checks.values()) else NOT_SUPPORTED_CLASSIFICATION}


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha: str) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_SCIENTIFIC_NOT_COMPLETED", "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha}, "provenance": {"git_start": dict(started)}, "claim_boundary": {"scientific_execution_authorization_consumed": True, "scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"}, "next": None}


def _resolve_output(root: Path, value: Path | None) -> Path:
    target = (value or (root / DEFAULT_OUTPUT_DIR)).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs), "V5 scientific output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing V5 scientific output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_scientific(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = _resolve_output(root, output_dir)
    entrypoint_path = Path(__file__).resolve()
    started = git_facts(root)
    status_before = _status_lines(root)
    payload = _base_payload(root, started, file_sha256(entrypoint_path))
    input_ref: dict[str, Any] | None = None
    authorities: Mapping[str, Any] | None = None
    scientific_started = False
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from authorized V5 execution")
        preflight = _load_preflight_module(root)
        contract, authorities, v2_contract, v3, v4, preflight_report, preflight_lineage = _validate_accepted_preflight(root, preflight)
        initial_state = preflight._combined_initial_state(v3, v4)
        _require(_digest_tree(initial_state) == EXPECTED_INITIAL_STATE_SHA256, "combined initial state differs from accepted V5 preflight")
        cuda_runtime = preflight._prepare_cuda_runtime_dll_search_path()
        device, device_facts = preflight._cuda_facts()
        properties = torch.cuda.get_device_properties(device)
        _require((int(properties.major), int(properties.minor)) == EXPECTED_COMPUTE_CAPABILITY, "CUDA compute capability differs from accepted preflight")
        _require(int(properties.total_memory) >= EXPECTED_VRAM_BYTES, "CUDA VRAM is below accepted preflight target")
        free_disk = shutil.disk_usage(target.parent).free
        _require(free_disk >= MIN_FREE_BYTES_FOR_OUTPUT, "insufficient output storage headroom")
        images, labels, train_indices, development_indices, dataset = _load_roles(root, preflight)
        development_images = images.index_select(0, development_indices).contiguous()
        development_labels = labels.index_select(0, development_indices).contiguous()
        schedule_initial, schedule, schedule_digest = _materialize_schedule(train_indices, labels, preflight)
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        smooth, smooth_identity = _load_smooth(root, authorities, device, preflight)
        dummy = preflight._new_consumer(v2_contract, device, PRODUCTION_GRID)
        preflight._load_runtime_state(dummy, initial_state)
        preflight._configure_all_frozen(dummy)
        del dummy
        prototype_consumer = preflight._new_consumer(v2_contract, device, PRODUCTION_GRID)
        preflight._load_runtime_state(prototype_consumer, initial_state)
        preflight._configure_all_frozen(prototype_consumer)
        prototypes, prototype_summary = preflight.construct_prototypes(prototype_consumer, images, labels, train_indices, expected_rows=55_000)
        s_mu, scale_summary = preflight.prototype_scale(prototypes)
        _require(prototype_summary.get("prototypes_tensor_sha256") == EXPECTED_PROTOTYPES_SHA256, "scientific prototype reconstruction differs from accepted preflight")
        _require(prototype_summary.get("z_train_tensor_sha256") == EXPECTED_Z_TRAIN_SHA256 and prototype_summary.get("canonical_z_tensor_sha256") == EXPECTED_CANONICAL_Z_SHA256, "scientific prototype source differs from accepted preflight")
        _require(scale_summary.get("pair_distance_tensor_sha256") == EXPECTED_PAIR_DISTANCE_SHA256 and math.isclose(s_mu, FROZEN_S_MU, rel_tol=0.0, abs_tol=1.0e-15), "scientific frozen prototype scale differs")
        del prototype_consumer
        initial_rng = preflight._capture_rng_state()
        initial_rng_sha = _digest_tree(initial_rng)
        _require(initial_rng_sha == EXPECTED_INITIAL_RNG_SHA256, "scientific initial RNG differs from accepted preflight")
        payload.update({"contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "authorities": dict(authorities), "preflight": {"report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, "lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256}, "result": preflight_report.get("result"), "classification": preflight_report.get("classification")}, "runtime": {"cuda_dll_search_path": cuda_runtime, "device": device_facts, "optimizer": _optimizer_manifest(), "propagation_grid": PRODUCTION_GRID, "logical_batch_size": LOGICAL_BATCH_SIZE, "scientific_updates": SCIENTIFIC_STEPS}, "dataset": dataset, "schedule": {"seed": SCHEDULE_SEED, "steps": SCIENTIFIC_STEPS, "initial_state_sha256": _digest_tree(schedule_initial), "schedule_sha256": schedule_digest, "first_batch_indices_sha256": tensor_sha256(schedule[0][0]), "last_batch_indices_sha256": tensor_sha256(schedule[-1][0])}, "initial_state": {"combined_runtime_state_sha256": _digest_tree(initial_state), "encoder_phase_sha256": EXPECTED_ENCODER_SHA256, "ols_transport_sha256": EXPECTED_OLS_TRANSPORT_SHA256, "initial_rng_sha256": initial_rng_sha}, "prototype_setup": {"summary": prototype_summary, "scale": scale_summary, "frozen": True, "source": "TRAIN only; initial V3 CONTROL encoder"}, "lambda_proto": {"g_image": FROZEN_G_IMAGE, "g_proto": FROZEN_G_PROTO, "lambda_proto": FROZEN_LAMBDA, "formula": "g_image/(g_proto+1e-12)", "source": "accepted V5 preflight attempt-05; not recalibrated in scientific execution"}, "evaluator": {"smooth": smooth_identity, "role": "SmoothMNISTEvaluator v2.1 evaluation-only", "gradient_enabled": False}, "data_roles": dataset})
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "accepted_preflight": authorities["accepted_preflight"], "protocol": {"control": "L_control=L_image", "candidate": "L_candidate=L_image+0.09573533218209243*L_proto", "s_mu": FROZEN_S_MU, "lambda_proto": FROZEN_LAMBDA, "steps": SCIENTIFIC_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "optimizer": _optimizer_manifest(), "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}}, "data_roles": {"TRAIN": 55_000, "DEV": 5_000, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": payload["claim_boundary"], "input_self_hash": "not_embedded_to_avoid_self_reference"}
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V5 scientific input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload["input"] = input_ref
        scientific_started = True
        control = _run_arm(root, target, v2_contract, initial_state, initial_rng, images, labels, train_indices, development_images, development_labels, prototypes, s_mu, smooth, schedule, arm="control", device=device, preflight=preflight)
        candidate = _run_arm(root, target, v2_contract, initial_state, initial_rng, images, labels, train_indices, development_images, development_labels, prototypes, s_mu, smooth, schedule, arm="candidate", device=device, preflight=preflight)
        _require(control["initial_state_digest"] == candidate["initial_state_digest"] == EXPECTED_INITIAL_STATE_SHA256, "control/candidate initial state mismatch")
        _require(control["initial_rng_digest"] == candidate["initial_rng_digest"] == EXPECTED_INITIAL_RNG_SHA256, "control/candidate initial RNG mismatch")
        _require(control["sampler_initial_state_digest"] == candidate["sampler_initial_state_digest"] == _digest_tree(schedule_initial), "control/candidate sampler initial state mismatch")
        _require(control["optimizer_manifest"] == candidate["optimizer_manifest"], "control/candidate optimizer manifest mismatch")
        sufficient_ref = _save_sufficient_observations(root, target, development_labels, control, candidate)
        bootstrap = _paired_bootstrap(root, target, development_labels, control, candidate)
        decision = _decision(control, candidate, bootstrap)
        candidate_step300 = candidate["evaluations"]["300"]
        firewall = {"metric": "Smooth DEV macro recall", "threshold": STAGE_A_FIREWALL, "candidate_step_300": candidate_step300["smooth"]["macro_recall"], "status": "STAGE_A_FIREWALL_PASSED" if candidate_step300["smooth"]["macro_recall"] >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
        public_arms = {}
        for arm_result in (control, candidate):
            public_arms[arm_result["arm"]] = {key: value for key, value in arm_result.items() if key not in {"evaluations", "step300_observations"}} | {"evaluations": {step: _public_evaluation(record) for step, record in arm_result["evaluations"].items()}}
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": decision["classification"], "arms": public_arms, "step_300_latent_mechanism": {"control": control["evaluations"]["300"]["ols_margin"], "candidate": candidate["evaluations"]["300"]["ols_margin"], "delta_mean": bootstrap["ols_margin"]["delta_mean"], "delta_median": bootstrap["ols_margin"]["delta_median"], "paired_bootstrap_ci": [bootstrap["ols_margin"]["ci_lower"], bootstrap["ols_margin"]["ci_upper"]], "positive_fraction": bootstrap["ols_margin"]["positive_fraction"], "per_class_delta_mean": bootstrap["ols_margin"]["per_class_delta_mean"]}, "step_300_optical_semantics": {"control_macro": control["evaluations"]["300"]["smooth"]["macro_recall"], "candidate_macro": candidate["evaluations"]["300"]["smooth"]["macro_recall"], "delta": bootstrap["smooth"]["delta_mean"], "paired_bootstrap_ci": [bootstrap["smooth"]["ci_lower"], bootstrap["smooth"]["ci_upper"]], "per_class_delta": bootstrap["smooth"]["per_class_delta"], "max_class_degradation": bootstrap["smooth"]["max_class_degradation"]}, "reconstruction": {"control_dev_image_loss": control["evaluations"]["300"]["losses"]["L_image"], "candidate_dev_image_loss": candidate["evaluations"]["300"]["losses"]["L_image"], "delta": decision["observed"]["image_loss_delta"]}, "raw_validity": {"control": control["evaluations"]["300"]["raw_panel"], "candidate": candidate["evaluations"]["300"]["raw_panel"]}, "bootstrap": bootstrap, "sufficient_observations": sufficient_ref, "decision": decision, "stage_a_firewall": firewall, "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_comparison_executed": True}, "next": "no_stage_b_authorization; retain V5 result as bounded scientific evidence"})
    except ScientificBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no retry without new authority"})
    except InvalidScientific as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific retry under one-shot authorization"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific retry under one-shot authorization"})
    finally:
        ending = git_facts(root)
        status_after = _status_lines(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == status_after, "unexpected_modified_paths": [] if status_before == status_after else sorted(set(status_after).symmetric_difference(status_before)), "scientific_started": scientific_started})
        payload["claim_boundary"].update({"scientific_training_executed": bool(payload["claim_boundary"].get("scientific_training_executed") or scientific_started), "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload.setdefault("artifacts", {})
        payload["artifacts"].setdefault("input", input_ref)
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"
        payload["scientific_300_step_executed"] = bool(payload["result"] == "VALID_COMPLETED" and payload["claim_boundary"].get("scientific_comparison_executed") is True)
        payload["forbidden_actions"] = {"stage_b": False, "formal_scale": False, "prior": False, "sampled_generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V5 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "entrypoint": payload["entrypoint"], "authorities": payload.get("authorities"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V5 scientific lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_300_step_executed": payload["scientific_300_step_executed"], "stage_b_authorized": False, "stage_b_executed": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("scientific",), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_scientific(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if result["result"] == "VALID_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
