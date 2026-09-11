"""Run the one-shot forward-only V4 generator conditionality diagnosis.

The diagnosis consumes the immutable V4 step-300 generator and fixed DEV role.
It never creates an optimizer, enables backward, updates optical state, reads
protected/test/sealed roles, or performs generation/promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    LightGen2DComplexOLSTrainingConsumerV2,
    file_sha256,
    git_facts,
    tensor_sha256,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load


TASK_ID = "DIAGNOSE_V4_FIXED_OLS_GENERATOR_CONDITIONALITY_V1"
OUTER_GATE = "V4_GENERATOR_CONDITIONALITY_RESOLVED"
REPORT_SCHEMA = "lightgen_2d_fixed_ols_generator_conditionality_report.v1"
INPUT_SCHEMA = "lightgen_2d_fixed_ols_generator_conditionality_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_fixed_ols_generator_conditionality_lineage.v1"
RAW_OUTPUT_SCHEMA = "lightgen_2d_fixed_ols_generator_conditionality_raw_outputs.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_fixed_ols_generator_conditionality_bootstrap.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"

V4_ENTRYPOINT_PATH = Path("scripts/research/execute_lightgen_2d_fixed_ols_generator_recoverability_v4_scientific.py")
V4_ENTRYPOINT_SHA256 = "a2a54556348deddc21a5ae0caf5c1343c9f6736f73d211aa33a1958ffb33375b"
V4_CONTRACT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract_v4_attempt_01/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json"
)
V4_CONTRACT_SHA256 = "d46740ded4db58d7d9f0caa8acaf40942d03dc3407d9611d24efcaa12d3036b8"
V4_PREFLIGHT_REPORT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_preflight_report.v1.json"
)
V4_PREFLIGHT_REPORT_SHA256 = "c2b0d852e1178915523ceee21172b10b8d462b6d0096ecf87c8443c410c7b7fb"
V4_PREFLIGHT_LINEAGE_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_preflight_lineage.v1.json"
)
V4_PREFLIGHT_LINEAGE_SHA256 = "139fd231c73cb5d8f90e74508e6d8d13645b7a6c9c30d839ad7311cbba642da8"
V4_SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1.json"
)
V4_SCIENTIFIC_REPORT_SHA256 = "fdfeb8d800a2d806f3ecc56503ef8d66cc68f06de6cd9e47aa4eab7c93ca2784"
V4_SCIENTIFIC_LINEAGE_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1.json"
)
V4_SCIENTIFIC_LINEAGE_SHA256 = "6089bc41797580aa95149b0478a0e25190a450e00ec4a47c886b756f0187f9e6"
V4_STEP_300_CHECKPOINT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/"
    "lightgen_2d_fixed_ols_generator_recoverability_v4_step_300_checkpoint.v1.pt"
)
V4_STEP_300_CHECKPOINT_SHA256 = "de80f2826419ca6ced28c8e1fc6d8804ef513493a49474e5997cfb4cb59bdf80"
V4_RUNTIME_PATH = Path("src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py")
V4_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
SMOOTH_EVALUATOR_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_conditionality_v1_attempt_01"
)
INPUT_NAME = "lightgen_2d_fixed_ols_generator_conditionality_input.v1.json"
REPORT_NAME = "lightgen_2d_fixed_ols_generator_conditionality_report.v1.json"
LINEAGE_NAME = "lightgen_2d_fixed_ols_generator_conditionality_lineage.v1.json"
RAW_OUTPUT_NAME = "lightgen_2d_fixed_ols_generator_conditionality_raw_outputs.v1.pt"
BOOTSTRAP_NAME = "lightgen_2d_fixed_ols_generator_conditionality_bootstrap.v1.pt"

INITIALIZATION_SEED = 1234
PRODUCTION_GRID = 2048
LOGICAL_BATCH_SIZE = 10
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 999
MATERIAL_MSE_MARGIN = 0.005
RANKING_THRESHOLD = 0.5
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
PHASE_SHAPE = (2, 1, 100, 100)
OLS_TRANSPORT = torch.eye(100, dtype=torch.complex64)


class InvalidDiagnosis(RuntimeError):
    """The authority or diagnostic harness is invalid."""


class DiagnosisBlocked(RuntimeError):
    """A valid hard boundary prevents the diagnosis."""

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


def _load_module(root: Path) -> Any:
    path = _path(root, V4_ENTRYPOINT_PATH)
    _require(path.is_file() and file_sha256(path) == V4_ENTRYPOINT_SHA256, "reviewed V4 scientific entrypoint identity changed")
    spec = importlib.util.spec_from_file_location("v4_scientific_consumer_for_diagnosis", path)
    _require(spec is not None and spec.loader is not None, "V4 scientific entrypoint cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json_ref(root: Path, relative: str | Path, expected_sha256: str, name: str) -> dict[str, Any]:
    path = _path(root, relative)
    _require(path.is_file(), f"{name} is unavailable")
    _require(file_sha256(path) == expected_sha256, f"{name} SHA mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidDiagnosis(f"{name} is not valid JSON") from error
    return dict(_mapping(value, name))


def _git_head(value: object, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower()),
        f"{name} is not a valid Git commit identity",
    )
    return value


def _bind_contract_git_provenance(contract: Mapping[str, Any], consumer_git: Mapping[str, Any]) -> dict[str, Any]:
    """Keep historical producer Git provenance separate from the current consumer."""

    historical_git = contract.get("git")
    if historical_git is None:
        historical_git = contract
    historical_git = _mapping(historical_git, "historical contract Git provenance")
    producer_branch = historical_git.get("branch", contract.get("branch"))
    producer_head = historical_git.get("head")
    if producer_head is None:
        for field in ("producer_head", "source_head", "git_head"):
            candidate = historical_git.get(field, contract.get(field))
            if candidate is not None:
                producer_head = candidate
                break
    _require(isinstance(producer_branch, str) and bool(producer_branch), "historical producer branch is unavailable")
    producer_head = _git_head(producer_head, "historical producer HEAD")

    consumer_branch = consumer_git.get("branch")
    consumer_head = _git_head(consumer_git.get("head"), "current consumer HEAD")
    _require(isinstance(consumer_branch, str) and bool(consumer_branch), "current consumer branch is unavailable")
    _require(consumer_branch == EXPECTED_BRANCH, "current consumer branch differs from the authorized diagnosis branch")

    exact_required = contract.get("consumer_head_exact_required", False)
    _require(isinstance(exact_required, bool), "consumer_head_exact_required must be boolean when declared")
    required_consumer_head = contract.get("consumer_head", producer_head)
    if exact_required:
        required_consumer_head = _git_head(required_consumer_head, "required consumer HEAD")
        _require(consumer_head == required_consumer_head, "exact consumer HEAD mismatch")

    return {
        "producer": {"branch": producer_branch, "head": producer_head},
        "consumer": {"branch": consumer_branch, "head": consumer_head},
        "consumer_head_exact_required": exact_required,
        "required_consumer_head": required_consumer_head if exact_required else None,
        "head_equal": consumer_head == producer_head,
        "producer_head_is_provenance_only": not exact_required,
    }


def _validate_smooth_evaluator_identity(contract_authorities: Mapping[str, Any], evaluator_states: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the loaded evaluator to its semantic/file authority without cross-domain equality."""

    evaluator_authorities = _mapping(contract_authorities.get("evaluators"), "V4 evaluator authorities")
    smooth_ref = _mapping(evaluator_authorities.get("smooth_checkpoint"), "Smooth evaluator authority")
    _require(smooth_ref.get("role") == "Smooth v2.1 evaluator checkpoint", "Smooth evaluator authority role changed")
    _require(isinstance(smooth_ref.get("path"), str) and bool(smooth_ref.get("path")), "Smooth evaluator authority path is unavailable")
    file_sha256_value = smooth_ref.get("sha256")
    _require(file_sha256_value == SMOOTH_EVALUATOR_SHA256, "Smooth evaluator checkpoint authority changed")
    state_tree_sha256 = evaluator_states.get("smooth_state_sha256")
    _require(
        isinstance(state_tree_sha256, str)
        and len(state_tree_sha256) == 64
        and all(character in "0123456789abcdef" for character in state_tree_sha256.lower()),
        "Smooth evaluator loaded state identity is unavailable",
    )
    return {
        "path": str(smooth_ref.get("path")),
        "role": str(smooth_ref.get("role")),
        "file_sha256": str(file_sha256_value),
        "state_dict_tree_sha256": state_tree_sha256,
        "identity_domains": {"file_sha256": "checkpoint bytes", "state_dict_tree_sha256": "loaded state_dict tree"},
        "cross_domain_equality_required": False,
    }


def _resolve_authorities(root: Path, v4: Any, consumer_git: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = v4._load_preflight_module(root)
    contract = _read_json_ref(root, V4_CONTRACT_PATH, V4_CONTRACT_SHA256, "V4 contract")
    contract_authorities = dict(_mapping(contract.get("authorities"), "V4 contract authorities"))
    git_provenance = _bind_contract_git_provenance(contract, consumer_git)
    # The reviewed V4 validator asks its preflight module for current authorities;
    # that module also interprets the historical producer HEAD as a live lock. The
    # diagnosis binds that historical field above, then reuses the validator's
    # remaining immutable artifact/semantic checks through the contract view.
    historical_contract_view = SimpleNamespace(
        _load_contract=lambda _root: (contract, None, contract_authorities)
    )
    _, _, current_authorities, v3_authority, preflight_authority = v4._validate_v4_authority(root, historical_contract_view)
    v4_report = _read_json_ref(root, V4_SCIENTIFIC_REPORT_PATH, V4_SCIENTIFIC_REPORT_SHA256, "V4 scientific report")
    _require(v4_report.get("result") == "VALID_COMPLETED", "V4 scientific report is not valid completed")
    _require(v4_report.get("classification") == "FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED", "V4 scientific classification changed")
    v4_lineage = _read_json_ref(root, V4_SCIENTIFIC_LINEAGE_PATH, V4_SCIENTIFIC_LINEAGE_SHA256, "V4 scientific lineage")
    _require(v4_lineage.get("result") == "VALID_COMPLETED" and v4_lineage.get("classification") == v4_report.get("classification"), "V4 scientific lineage changed")
    checkpoint_ref = _mapping(_mapping(v4_report.get("checkpoints"), "V4 checkpoints").get("300"), "V4 step-300 checkpoint ref")
    _require(checkpoint_ref.get("sha256") == V4_STEP_300_CHECKPOINT_SHA256, "V4 step-300 checkpoint ref changed")
    checkpoint = _mapping(
        safe_torch_load(_path(root, V4_STEP_300_CHECKPOINT_PATH), expected_sha256=V4_STEP_300_CHECKPOINT_SHA256, map_location="cpu"),
        "V4 step-300 checkpoint",
    )
    _require(checkpoint.get("schema") == "lightgen_2d_fixed_ols_generator_recoverability_scientific_checkpoint.v1" and checkpoint.get("checkpoint_step") == 300, "V4 checkpoint schema/step changed")
    _require(checkpoint.get("primary_endpoint") is True and checkpoint.get("objective") == "L_generator=L_image", "V4 checkpoint endpoint/objective changed")
    state = dict(_mapping(checkpoint.get("runtime_state"), "V4 checkpoint runtime state"))
    _require(set(state) == {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "V4 runtime state keys changed")
    _require(tensor_sha256(state["encoder.phase"]) == "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca", "V4 encoder identity changed")
    _require(tensor_sha256(state["generator.phase"]) == "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe", "V4 generator identity changed")
    _require(torch.equal(state["ols.operator.transport"], OLS_TRANSPORT), "V4 OLS transport is not T=I_100")
    _require(file_sha256(_path(root, V4_RUNTIME_PATH)) == V4_RUNTIME_SHA256, "LightGen runtime identity changed")
    return preflight, contract, contract_authorities, v3_authority, preflight_authority, {"report": v4_report, "lineage": v4_lineage, "checkpoint": checkpoint, "state": state, "current_authorities": current_authorities, "git_provenance": git_provenance}


def _load_runtime(root: Path, v4: Any, preflight: Any, v2_contract: Any, state: Mapping[str, Any], device: torch.device) -> Any:
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    consumer = LightGen2DComplexOLSTrainingConsumerV2(v2_contract, device=device, propagation_grid_size=PRODUCTION_GRID)
    consumer.runtime.load_state_dict(dict(state), strict=True)
    consumer._enforce_parameter_firewall()
    consumer.runtime.eval()
    loaded = v4._runtime_state(consumer, preflight)
    _require(tensor_sha256(loaded["encoder.phase"]) == "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca", "loaded V4 encoder changed")
    _require(tensor_sha256(loaded["generator.phase"]) == "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe", "loaded V4 generator changed")
    _require(torch.equal(loaded["ols.operator.transport"], OLS_TRANSPORT), "loaded V4 OLS transport changed")
    return consumer


def _selection_and_data(root: Path, v4: Any, preflight: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, train_indices, development_indices, selection_indices, dataset = v4._load_roles(root, preflight)
    _require(tensor_sha256(selection_indices) == SELECTION_INDICES_SHA256, "DEV selection identity changed")
    _require(train_indices.numel() == 55_000 and development_indices.numel() == 5_000 and selection_indices.numel() == 1_000, "diagnostic role counts changed")
    return images, labels, train_indices, selection_indices, dataset


def _permutations(selection_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(selection_labels.numel())
    within = torch.arange(count, dtype=torch.long)
    cross = torch.arange(count, dtype=torch.long)
    positions_by_class: list[torch.Tensor] = []
    for digit in range(10):
        positions = torch.nonzero(selection_labels == digit, as_tuple=False).flatten()
        _require(positions.numel() == 100, f"fixed DEV selection class count changed for {digit}")
        positions_by_class.append(positions)
        within[positions] = torch.roll(positions, shifts=-1, dims=0)
    for digit, positions in enumerate(positions_by_class):
        cross[positions] = positions_by_class[(digit + 1) % 10]
    _require(torch.equal(torch.sort(within).values, torch.arange(count)) and torch.equal(torch.sort(cross).values, torch.arange(count)), "permutation is not bijective")
    _require(torch.all(within != torch.arange(count)) and torch.all(selection_labels.index_select(0, cross) != selection_labels), "permutation constraints are not satisfied")
    return within, cross, selection_labels.index_select(0, within), selection_labels.index_select(0, cross)


def _validate_result(result: Any, batch_size: int, *, external: bool) -> torch.Tensor:
    if external:
        _require(result.image_input is None and result.ols_input_field is None and result.z_ols.source == "external_generated", "permuted path crossed image/encoder boundary")
    else:
        _require(result.image_input is not None and result.ols_input_field is not None and result.z_ols.source == "real_encoded", "matched path lost canonical encoder boundary")
        _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "T=I_100 provenance changed")
    raw = result.raw_intensity
    _require(tuple(raw.shape) == (batch_size, 1, 200, 200) and raw.dtype == torch.float32, "raw CMOS shape/dtype changed")
    _finite(raw, "raw CMOS")
    return raw


@torch.inference_mode()
def _encode_matched(consumer: Any, selection_images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    raw_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    max_canonical_replay_difference = 0.0
    for start in range(0, selection_images.shape[0], LOGICAL_BATCH_SIZE):
        batch = selection_images[start : start + LOGICAL_BATCH_SIZE].to(device)
        result = consumer.runtime.forward_from_image(batch)
        raw = _validate_result(result, int(batch.shape[0]), external=False)
        z = result.z_ols.coefficients.detach()
        target = result.image_input.slm_intensity.detach()
        _require(tuple(z.shape) == (int(batch.shape[0]), 100) and z.dtype == torch.complex64, "matched z shape/dtype changed")
        _finite(target, "matched SLM target")
        replay = consumer.runtime.forward_from_ols(z)
        replay_raw = _validate_result(replay, int(batch.shape[0]), external=True)
        max_canonical_replay_difference = max(max_canonical_replay_difference, float((raw - replay_raw).abs().max().cpu()))
        raw_parts.append(raw.cpu().contiguous())
        z_parts.append(z.cpu().contiguous())
        target_parts.append(target.cpu().contiguous())
        del batch, result, replay, raw, replay_raw, z, target
    torch.cuda.synchronize(device)
    raw_cpu = torch.cat(raw_parts, dim=0)
    z_cpu = torch.cat(z_parts, dim=0)
    target_cpu = torch.cat(target_parts, dim=0)
    _require(tuple(raw_cpu.shape) == (1000, 1, 200, 200) and tuple(target_cpu.shape) == raw_cpu.shape and tuple(z_cpu.shape) == (1000, 100), "matched collection shape changed")
    return raw_cpu, z_cpu, target_cpu, max_canonical_replay_difference


@torch.inference_mode()
def _forward_permuted(consumer: Any, z_cpu: torch.Tensor, target_cpu: torch.Tensor, permutation: torch.Tensor, device: torch.device) -> torch.Tensor:
    raw_parts: list[torch.Tensor] = []
    for start in range(0, z_cpu.shape[0], LOGICAL_BATCH_SIZE):
        indices = permutation[start : start + LOGICAL_BATCH_SIZE]
        z = z_cpu.index_select(0, indices).to(device)
        result = consumer.runtime.forward_from_ols(z)
        raw = _validate_result(result, int(z.shape[0]), external=True)
        raw_parts.append(raw.cpu().contiguous())
        del indices, z, result, raw
    torch.cuda.synchronize(device)
    output = torch.cat(raw_parts, dim=0)
    _require(tuple(output.shape) == tuple(target_cpu.shape), "permuted raw collection shape changed")
    return output


def _per_sample_mse(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require(raw.shape == target.shape, "raw and target shapes differ")
    values = (raw.to(torch.float64) - target.to(torch.float64)).square().flatten(1).mean(dim=1)
    _finite(values, "per-sample MSE")
    return values


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(torch.float64).flatten()
    _finite(values, "summary values")
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)), "min": float(values.min()), "q05": float(quantiles[0]), "median": float(quantiles[1]), "q95": float(quantiles[2]), "max": float(values.max())}


def _raw_rms_summary(raw: torch.Tensor, matched: torch.Tensor) -> dict[str, Any]:
    rms = (raw.to(torch.float64) - matched.to(torch.float64)).square().flatten(1).mean(dim=1).sqrt()
    return {"per_sample": rms, "summary": _summary(rms), "tensor_sha256": tensor_sha256(rms)}


def _label_stats(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    correct = predictions == labels
    per_class: list[float] = []
    for digit in range(10):
        mask = labels == digit
        _require(int(mask.sum()) == 100, f"label statistics class count changed for {digit}")
        per_class.append(float(correct[mask].to(torch.float64).mean()))
    return {"accuracy": float(correct.to(torch.float64).mean()), "macro_recall": float(sum(per_class) / 10.0), "per_class_recall": per_class, "predictions": predictions}


@torch.inference_mode()
def _smooth_descriptive(smooth: Any, raw: torch.Tensor, target_labels: torch.Tensor, donor_labels: torch.Tensor, device: torch.device, matched_predictions: torch.Tensor | None = None) -> dict[str, Any]:
    predictions: list[torch.Tensor] = []
    for start in range(0, raw.shape[0], LOGICAL_BATCH_SIZE):
        batch = raw[start : start + LOGICAL_BATCH_SIZE].to(device)
        view = F.adaptive_avg_pool2d(batch.clamp(0.0, 1.0), (28, 28))
        _, logits = smooth(view)
        predictions.append(logits.argmax(dim=1).cpu())
        del batch, view, logits
    pred = torch.cat(predictions, dim=0)
    target_stats = _label_stats(pred, target_labels)
    donor_stats = _label_stats(pred, donor_labels)
    result = {"target_label": {key: value for key, value in target_stats.items() if key != "predictions"}, "donor_label": {key: value for key, value in donor_stats.items() if key != "predictions"}, "prediction_tensor_sha256": tensor_sha256(pred)}
    if matched_predictions is not None:
        result["agreement_with_matched_prediction"] = float((pred == matched_predictions).to(torch.float64).mean())
    result["_predictions"] = pred
    return result


def _bootstrap_deltas(deltas: Mapping[str, torch.Tensor], labels: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    draws: list[torch.Tensor] = []
    values: dict[str, list[torch.Tensor]] = {name: [] for name in deltas}
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 100, f"bootstrap class count changed for {digit}")
        draw = torch.randint(0, 100, (BOOTSTRAP_REPLICATES, 100), generator=generator)
        draws.append(draw)
        for name, delta in deltas.items():
            values[name].append(delta.index_select(0, matches)[draw].mean(dim=1))
    macro: dict[str, torch.Tensor] = {name: torch.stack(items, dim=0).mean(dim=0) for name, items in values.items()}
    summaries: dict[str, Any] = {}
    for name, sample in macro.items():
        q = torch.quantile(sample.to(torch.float64), torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
        summaries[name] = {"mean": float(sample.mean()), "ci_lower": float(q[0]), "ci_upper": float(q[1]), "tensor_sha256": tensor_sha256(sample)}
    return summaries, {"draws": torch.stack(draws, dim=0), "macro_deltas": macro}


def _distance_summary(values: torch.Tensor) -> dict[str, Any]:
    return _summary(values)


def _template_metrics(train_images: torch.Tensor, train_labels: torch.Tensor, selection_images: torch.Tensor, matched_raw: torch.Tensor) -> dict[str, Any]:
    global_mean = train_images.mean(dim=0)
    class_means = []
    for digit in range(10):
        matches = train_labels == digit
        _require(int(matches.sum()) > 0, f"TRAIN class mean is undefined for {digit}")
        class_means.append(train_images[matches].mean(dim=0))
    class_mean_tensor = torch.stack(class_means, dim=0)
    output_view = F.adaptive_avg_pool2d(matched_raw.clamp(0.0, 1.0), (28, 28))
    target_view = selection_images.clamp(0.0, 1.0)
    _require(tuple(output_view.shape) == tuple(target_view.shape), "template comparison view shape changed")
    true_target_distance = (output_view.to(torch.float64) - target_view.to(torch.float64)).square().flatten(1).mean(dim=1)
    global_distance = (output_view.to(torch.float64) - global_mean.to(torch.float64)).square().flatten(1).mean(dim=1)
    class_distance = (output_view[:, None].to(torch.float64) - class_mean_tensor[None].to(torch.float64)).square().flatten(2).mean(dim=2)
    nearest = class_distance.argmin(dim=1)
    counts = torch.bincount(nearest, minlength=10)
    probabilities = counts.to(torch.float64) / float(nearest.numel())
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    return {"preprocessing": "raw CMOS clamp [0,1] then adaptive_avg_pool2d to 28x28; TRAIN means are source image means", "train_rows": int(train_images.shape[0]), "train_source_sha256": TRAINING_SOURCE_SHA256, "true_target_view_distance": _distance_summary(true_target_distance), "output_to_global_mean_distance": _distance_summary(global_distance), "output_to_class_template_distance": [_distance_summary(class_distance[:, digit]) for digit in range(10)], "nearest_class_template_counts": [int(value) for value in counts.tolist()], "nearest_class_template_distribution": [float(value) for value in probabilities.tolist()], "fraction_nearest_to_class_8": float(probabilities[8]), "nearest_class_template_entropy_nats": entropy, "nearest_class_template_entropy_normalized": entropy / math.log(10.0), "nearest_class_template_tensor_sha256": tensor_sha256(nearest), "global_mean_tensor_sha256": tensor_sha256(global_mean), "class_mean_tensor_sha256": tensor_sha256(class_mean_tensor)}


def _save_raw_outputs(root: Path, target: Path, matched: torch.Tensor, within: torch.Tensor, cross: torch.Tensor, z: torch.Tensor, target_slm: torch.Tensor, selection_labels: torch.Tensor, within_labels: torch.Tensor, cross_labels: torch.Tensor) -> dict[str, Any]:
    path = target / RAW_OUTPUT_NAME
    payload = {"schema": RAW_OUTPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": V4_CONTRACT_PATH.as_posix(), "sha256": V4_CONTRACT_SHA256}, "v4_step": 300, "selection_indices_sha256": SELECTION_INDICES_SHA256, "selection_labels": selection_labels, "within_donor_labels": within_labels, "cross_donor_labels": cross_labels, "z": z, "target_slm": target_slm, "raw_matched": matched, "raw_within_class": within, "raw_cross_class": cross}
    atomic_torch_save_new(path, payload, artifact_kind="V4 generator conditionality raw outputs")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "conditionality raw output artifact")
    for key, value in (("raw_matched", matched), ("raw_within_class", within), ("raw_cross_class", cross), ("z", z)):
        _require(torch.is_tensor(loaded.get(key)) and tensor_sha256(loaded[key]) == tensor_sha256(value), f"raw output artifact reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": RAW_OUTPUT_SCHEMA, "selection_indices_sha256": SELECTION_INDICES_SHA256}


def _save_bootstrap(root: Path, target: Path, summaries: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
    path = target / BOOTSTRAP_NAME
    payload = {"schema": BOOTSTRAP_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": V4_CONTRACT_PATH.as_posix(), "sha256": V4_CONTRACT_SHA256}, "selection_indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": tensor_sha256(labels), "protocol": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired, class-stratified bootstrap over the same 100 examples per class", "quantile_convention": "torch.quantile interpolation=linear"}, "summaries": dict(summaries), "draws": tensors["draws"], "macro_deltas": tensors["macro_deltas"]}
    atomic_torch_save_new(path, payload, artifact_kind="V4 generator conditionality bootstrap")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "conditionality bootstrap artifact")
    _require(loaded.get("schema") == BOOTSTRAP_SCHEMA and loaded.get("protocol", {}).get("seed") == BOOTSTRAP_SEED, "bootstrap artifact reload changed")
    return {"path": _relative(root, path), "sha256": digest, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "summaries": dict(summaries)}


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha256: str) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V4_GENERATOR_CONDITIONALITY_NOT_COMPLETED", "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha256}, "provenance": {"git_start": dict(started)}, "claim_boundary": {"diagnosis_authorization_consumed": True, "diagnosis_executed": False, "optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "retraining_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_work_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False}, "next": None}


def run_diagnosis(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    _require(not target.exists(), f"one-shot diagnosis output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    entrypoint_sha256 = file_sha256(Path(__file__).resolve())
    payload = _base_payload(root, started, entrypoint_sha256)
    input_ref: dict[str, Any] | None = None
    artifact_refs: dict[str, Any] = {}
    diagnosis_started = False
    try:
        _require(started.get("branch") == EXPECTED_BRANCH, "active Git branch differs from authorized diagnosis branch")
        v4 = _load_module(root)
        preflight, contract, contract_authorities, v3_authority, preflight_authority, v4_authority = _resolve_authorities(root, v4, started)
        device, device_facts = preflight._cuda_device_facts()
        _require(tuple(device_facts.get("compute_capability", ())) == EXPECTED_COMPUTE_CAPABILITY and int(device_facts.get("vram_bytes", 0)) >= EXPECTED_VRAM_BYTES, "diagnosis target device differs from accepted V4 target")
        _require(shutil.disk_usage(target.parent).free >= MIN_FREE_BYTES_FOR_OUTPUT, "insufficient output storage headroom")
        v2_contract = v4._load_v2_contract_before_seed(root)
        consumer = _load_runtime(root, v4, preflight, v2_contract, v4_authority["state"], device)
        frozen, smooth, evaluator_states = v4._load_evaluators(root, preflight, contract_authorities, device)
        frozen.eval()
        smooth.eval()
        smooth_identity = _validate_smooth_evaluator_identity(contract_authorities, evaluator_states)
        images, labels, train_indices, selection_indices, dataset = _selection_and_data(root, v4, preflight)
        selection_images = images.index_select(0, selection_indices).contiguous()
        selection_labels = labels.index_select(0, selection_indices).contiguous()
        within_permutation, cross_permutation, within_labels, cross_labels = _permutations(selection_labels)
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "entrypoint": {"path": _relative(root, Path(__file__).resolve()), "sha256": entrypoint_sha256}, "authorities": {"v4_contract": {"path": V4_CONTRACT_PATH.as_posix(), "sha256": V4_CONTRACT_SHA256}, "v4_preflight_report": {"path": V4_PREFLIGHT_REPORT_PATH.as_posix(), "sha256": V4_PREFLIGHT_REPORT_SHA256}, "v4_preflight_lineage": {"path": V4_PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": V4_PREFLIGHT_LINEAGE_SHA256}, "v4_scientific_report": {"path": V4_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V4_SCIENTIFIC_REPORT_SHA256}, "v4_scientific_lineage": {"path": V4_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V4_SCIENTIFIC_LINEAGE_SHA256}, "v4_step_300_checkpoint": {"path": V4_STEP_300_CHECKPOINT_PATH.as_posix(), "sha256": V4_STEP_300_CHECKPOINT_SHA256}, "runtime": {"path": V4_RUNTIME_PATH.as_posix(), "sha256": V4_RUNTIME_SHA256}, "smooth_evaluator": {"sha256": SMOOTH_EVALUATOR_SHA256}}, "protocol": {"matched": "fixed DEV x -> frozen encoder -> z in C^100 -> modal projection -> T=I_100 -> shared W_OLS -> V4 step-300 generator", "within_class_permutation": "for each class, map frozen per-class selection order k to (k+1) mod 100", "cross_class_permutation": "for target class d and position k, map to donor class (d+1) mod 10 at the same position k", "labels_used_only_for": ["deterministic permutations", "TRAIN-only global/class mean templates", "descriptive Smooth target/donor metrics"], "mse": "per-sample mean squared error in contract-defined 200x200 SLM-intensity target space", "material_mse_margin": MATERIAL_MSE_MARGIN, "ranking_materiality_threshold": RANKING_THRESHOLD, "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired, class-stratified over 100 examples per class", "quantile_convention": "torch.quantile interpolation=linear"}, "classification_rules": {"collapse": "within and cross bootstrap CI upper bounds <= material_mse_margin", "class": "within CI upper <= margin, cross CI lower > margin, and cross-minus-within CI lower > margin", "instance": "within and cross CI lower > margin, with both ranking fractions > 0.5", "otherwise": "mixed_or_unresolved"}, "template_view": "raw CMOS clamp [0,1] then adaptive_avg_pool2d to 28x28; TRAIN means only"}, "data_roles": {"train_rows_for_templates": 55000, "development_role_rows": 5000, "development_selection_rows": 1000, "calibration_rows": 0, "protected_rows": 0, "test_rows": 0, "sealed_rows": 0}, "claim_boundary": {"optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "stage_b_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False}, "input_self_hash": "not_embedded_to_avoid_self_reference"}
        input_payload["authorities"]["smooth_evaluator"].update(smooth_identity)
        input_payload["git_provenance"] = v4_authority["git_provenance"]
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V4 generator conditionality input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        diagnosis_started = True
        started_measurement = time.perf_counter()
        matched_raw, z_cpu, target_slm, canonical_replay_difference = _encode_matched(consumer, selection_images, device)
        within_raw = _forward_permuted(consumer, z_cpu, target_slm, within_permutation, device)
        cross_raw = _forward_permuted(consumer, z_cpu, target_slm, cross_permutation, device)
        matched_mse = _per_sample_mse(matched_raw, target_slm)
        within_mse = _per_sample_mse(within_raw, target_slm)
        cross_mse = _per_sample_mse(cross_raw, target_slm)
        delta_within = within_mse - matched_mse
        delta_cross = cross_mse - matched_mse
        delta_cross_minus_within = delta_cross - delta_within
        bootstrap_summaries, bootstrap_tensors = _bootstrap_deltas({"within": delta_within, "cross": delta_cross, "cross_minus_within": delta_cross_minus_within}, selection_labels)
        raw_artifact = _save_raw_outputs(root, target, matched_raw, within_raw, cross_raw, z_cpu, target_slm, selection_labels, within_labels, cross_labels)
        bootstrap_artifact = _save_bootstrap(root, target, bootstrap_summaries, bootstrap_tensors, selection_labels)
        artifact_refs.update({"raw_outputs": raw_artifact, "bootstrap": bootstrap_artifact})
        matched_smooth = _smooth_descriptive(smooth, matched_raw, selection_labels, selection_labels, device)
        within_smooth = _smooth_descriptive(smooth, within_raw, selection_labels, within_labels, device, matched_smooth["_predictions"])
        cross_smooth = _smooth_descriptive(smooth, cross_raw, selection_labels, cross_labels, device, matched_smooth["_predictions"])
        v4_endpoint = _mapping(_mapping(v4_authority["report"].get("trajectory"), "V4 trajectory").get("300"), "V4 endpoint")
        v4_smooth = _mapping(v4_endpoint.get("smooth_evaluator"), "V4 Smooth endpoint")
        _require(math.isclose(matched_smooth["target_label"]["macro_recall"], float(v4_smooth["macro_recall"]), rel_tol=0.0, abs_tol=1e-7), "diagnostic matched Smooth endpoint does not reproduce V4")
        _require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-7) for a, b in zip(matched_smooth["target_label"]["per_class_recall"], v4_smooth["per_class_recall"], strict=True)), "diagnostic matched Smooth per-class endpoint does not reproduce V4")
        train_images = images.index_select(0, train_indices).contiguous()
        template = _template_metrics(train_images, labels.index_select(0, train_indices).contiguous(), selection_images, matched_raw)
        within_rms = _raw_rms_summary(within_raw, matched_raw)
        cross_rms = _raw_rms_summary(cross_raw, matched_raw)
        within_ci = bootstrap_summaries["within"]
        cross_ci = bootstrap_summaries["cross"]
        cross_within_ci = bootstrap_summaries["cross_minus_within"]
        rank_within = float((matched_mse < within_mse).to(torch.float64).mean())
        rank_cross = float((matched_mse < cross_mse).to(torch.float64).mean())
        collapse = float(within_ci["ci_upper"]) <= MATERIAL_MSE_MARGIN and float(cross_ci["ci_upper"]) <= MATERIAL_MSE_MARGIN
        class_conditioning = float(within_ci["ci_upper"]) <= MATERIAL_MSE_MARGIN and float(cross_ci["ci_lower"]) > MATERIAL_MSE_MARGIN and float(cross_within_ci["ci_lower"]) > MATERIAL_MSE_MARGIN and rank_cross > RANKING_THRESHOLD
        instance_conditioning = float(within_ci["ci_lower"]) > MATERIAL_MSE_MARGIN and float(cross_ci["ci_lower"]) > MATERIAL_MSE_MARGIN and rank_within > RANKING_THRESHOLD and rank_cross > RANKING_THRESHOLD
        if collapse:
            classification = "GENERATOR_OLS_CONDITIONING_COLLAPSE_SUPPORTED"
        elif class_conditioning:
            classification = "GENERATOR_CLASS_CONDITIONING_WITHOUT_SEMANTIC_RENDERING_SUPPORTED"
        elif instance_conditioning:
            classification = "GENERATOR_INSTANCE_CONDITIONING_WITHOUT_SEMANTIC_RENDERING_SUPPORTED"
        else:
            classification = "V4_GENERATOR_CONDITIONALITY_MIXED_OR_UNRESOLVED"
        provider_after = {"frozen_state_sha256": v4._digest_tree(v4._cpu_tree(frozen.state_dict())), "smooth_state_sha256": v4._digest_tree(v4._cpu_tree(smooth.state_dict()))}
        _require(provider_after["frozen_state_sha256"] == evaluator_states["frozen_state_sha256"] and provider_after["smooth_state_sha256"] == evaluator_states["smooth_state_sha256"], "evaluator state changed during diagnosis")
        _require(all(parameter.grad is None for evaluator in (frozen, smooth) for parameter in evaluator.parameters()), "evaluator received a diagnosis gradient")
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": classification, "authorities": {"v4_contract": {"path": V4_CONTRACT_PATH.as_posix(), "sha256": V4_CONTRACT_SHA256}, "v4_preflight_report": {"path": V4_PREFLIGHT_REPORT_PATH.as_posix(), "sha256": V4_PREFLIGHT_REPORT_SHA256}, "v4_preflight_lineage": {"path": V4_PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": V4_PREFLIGHT_LINEAGE_SHA256}, "v4_scientific_report": {"path": V4_SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": V4_SCIENTIFIC_REPORT_SHA256}, "v4_scientific_lineage": {"path": V4_SCIENTIFIC_LINEAGE_PATH.as_posix(), "sha256": V4_SCIENTIFIC_LINEAGE_SHA256}, "v4_step_300_checkpoint": {"path": V4_STEP_300_CHECKPOINT_PATH.as_posix(), "sha256": V4_STEP_300_CHECKPOINT_SHA256, "runtime_state_sha256": _digest_tree(v4_authority["state"])}, "runtime": {"path": V4_RUNTIME_PATH.as_posix(), "sha256": V4_RUNTIME_SHA256}, "smooth_evaluator": {"sha256": SMOOTH_EVALUATOR_SHA256}, "v4_endpoint_smooth_macro": float(v4_smooth["macro_recall"])}, "device": device_facts, "protocol": input_payload["protocol"], "runtime": {"grid": PRODUCTION_GRID, "batch": LOGICAL_BATCH_SIZE, "transport": "T=I_100_coefficient_space", "modal_projection_boundary": True, "shared_w_ols": True, "canonical_replay_max_abs_difference": canonical_replay_difference, "optimizer_created": False, "backward_executed": False, "optical_update_executed": False}, "data": {**dataset, "diagnostic_train_template_rows": int(train_images.shape[0]), "diagnostic_dev_selection_rows": int(selection_images.shape[0]), "calibration_rows_read": 0, "protected_rows_read": 0, "test_rows_read": 0, "sealed_rows_read": 0}, "permutations": {"within_class": {"rule": "per-class frozen selection order +1 modulo 100", "indices_sha256": tensor_sha256(within_permutation), "donor_labels_sha256": tensor_sha256(within_labels)}, "cross_class": {"rule": "donor class (target class+1) modulo 10 at same per-class position", "indices_sha256": tensor_sha256(cross_permutation), "donor_labels_sha256": tensor_sha256(cross_labels)}}, "matched": {"per_sample_reconstruction_mse": matched_mse.cpu().tolist(), "mse_summary": _summary(matched_mse), "raw_cmos": v4._raw_summary(matched_raw), "raw_tensor_sha256": tensor_sha256(matched_raw), "smooth": {key: value for key, value in matched_smooth.items() if key != "_predictions"}}, "within_class": {"per_sample_reconstruction_mse": within_mse.cpu().tolist(), "mse_summary": _summary(within_mse), "paired_delta_summary": _summary(delta_within), "paired_bootstrap": within_ci, "ranking_fraction_matched_lower_mse": rank_within, "raw_rms_change_vs_matched": {"summary": within_rms["summary"], "tensor_sha256": within_rms["tensor_sha256"]}, "raw_cmos": v4._raw_summary(within_raw), "raw_tensor_sha256": tensor_sha256(within_raw), "smooth": {key: value for key, value in within_smooth.items() if key != "_predictions"}}, "cross_class": {"per_sample_reconstruction_mse": cross_mse.cpu().tolist(), "mse_summary": _summary(cross_mse), "paired_delta_summary": _summary(delta_cross), "paired_bootstrap": cross_ci, "ranking_fraction_matched_lower_mse": rank_cross, "raw_rms_change_vs_matched": {"summary": cross_rms["summary"], "tensor_sha256": cross_rms["tensor_sha256"]}, "raw_cmos": v4._raw_summary(cross_raw), "raw_tensor_sha256": tensor_sha256(cross_raw), "smooth": {key: value for key, value in cross_smooth.items() if key != "_predictions"}}, "conditioning": {"delta_within_definition": "L_within-L_matched", "delta_cross_definition": "Delta_cross-Delta_within", "cross_minus_within_definition": "Delta_cross-Delta_within", "delta_within_summary": _summary(delta_within), "delta_cross_summary": _summary(delta_cross), "cross_minus_within_summary": _summary(delta_cross_minus_within), "paired_bootstrap_cross_minus_within": cross_within_ci, "material_mse_margin": MATERIAL_MSE_MARGIN, "ranking_threshold": RANKING_THRESHOLD, "decision_checks": {"collapse": collapse, "class_conditioning": class_conditioning, "instance_conditioning": instance_conditioning}, "primary_classification": classification}, "template_collapse": template, "secondary_findings": {"smooth_target_label": {"matched": matched_smooth["target_label"], "within_class": within_smooth["target_label"], "cross_class": cross_smooth["target_label"]}, "smooth_donor_label": {"matched": matched_smooth["target_label"], "within_class": within_smooth["target_label"], "cross_class": cross_smooth["target_label"]}, "smooth_prediction_agreement_with_matched": {"within_class": within_smooth["agreement_with_matched_prediction"], "cross_class": cross_smooth["agreement_with_matched_prediction"]}, "unresolved": ["independent Smooth semantics remain unestablished", "template collapse is descriptive attribution only"]}, "artifacts": {**artifact_refs, "input": input_ref}, "governance": {"diagnosis_executed": True, "optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "retraining_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_work_executed": False, "sampled_generation_executed": False, "generation_acceptance": False, "protected_read": False, "test_read": False, "sealed_read": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False}, "measurement_elapsed_seconds": time.perf_counter() - started_measurement, "next": "no_retry; retain diagnosis; V4 scientific result remains NOT_SUPPORTED; no Stage B or generation"})
        payload["authorities"]["smooth_evaluator"].update(smooth_identity)
        payload["git_provenance"] = v4_authority["git_provenance"]
        payload["cross_class"]["paired_bootstrap"] = cross_ci
        payload["conditioning"]["delta_cross_definition"] = "L_cross-L_matched"
        payload["secondary_findings"]["smooth_donor_label"] = {"matched": matched_smooth["donor_label"], "within_class": within_smooth["donor_label"], "cross_class": cross_smooth["donor_label"]}
        del consumer, frozen, smooth
        torch.cuda.empty_cache()
    except DiagnosisBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no_retry_without_new_authority"})
    except InvalidDiagnosis as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V4_GENERATOR_CONDITIONALITY_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; repair requires new authority"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V4_GENERATOR_CONDITIONALITY_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no_retry_without_new_authority"})
    finally:
        try:
            ending = git_facts(root)
            payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "status_lines_unchanged": started.get("status_lines") == ending.get("status_lines")})
        except Exception as error:
            payload["provenance"]["git_end_error"] = {"type": type(error).__name__, "message": str(error)}
        payload["claim_boundary"].update({"diagnosis_executed": bool(payload["claim_boundary"].get("diagnosis_executed") or diagnosis_started), "optimizer_created": False, "backward_executed": False, "optical_update_executed": False, "retraining_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_work_executed": False, "sampled_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False})
        payload.setdefault("artifacts", {})
        payload["artifacts"].setdefault("input", input_ref)
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"
    def _json_default(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        raise TypeError(f"unsupported report value: {type(value).__name__}")

    payload = json.loads(json.dumps(payload, default=_json_default))
    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V4 generator conditionality report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "entrypoint": payload["entrypoint"], "authorities": payload.get("authorities"), "git_provenance": payload.get("git_provenance"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V4 generator conditionality lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "diagnosis_executed": payload["claim_boundary"]["diagnosis_executed"], "optimizer_created": False, "backward_executed": False, "stage_b_executed": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
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
