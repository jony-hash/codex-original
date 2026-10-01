"""Execute the one-shot N5 successor true optical-depth comparison.

This is an independent successor to the invalid V1 execution attempt.  It
reuses the frozen V1 *design* and the frozen C1/N4 authorities, but it never
consumes V1's unpersisted scientific observations.  The canonical sufficient
observation is written and verified immediately after each scientific
evaluation; diagnostics and report construction happen afterwards.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research authority: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These modules are code authorities only.  No V1 scientific output directory
# is imported or used as an observation source by this successor consumer.
V1 = _load_module(
    Path(__file__).with_name("execute_lightgen_2d_true_optical_depth_comparison_v1.py"),
    "lightgen_n5_v1_code_authority_for_successor",
)
N4 = V1.N4
N4ATTR = V1.N4ATTR
N5DEF = V1.N5DEF
V10 = V1.V10
V11 = V1.V11

ContinuationSampler = V1.ContinuationSampler
TrueDepthGenerator = V1.TrueDepthGenerator

TASK_ID = "EXECUTE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2"
DESIGN_AUTHORITY = "DEFINE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2_CONTRACT"
DESIGN_GATE = "N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_READY"
OUTER_GATE = "N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_RESOLVED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

BASE_OUTPUT = Path("outputs/20260914_n5_v2/runs")
CONTRACT_PATH = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_v2_contract_v1_attempt_01" / "lightgen_2d_true_optical_depth_comparison_v2_contract.v1.json"
CONTRACT_SHA256 = "f7928d640093d8133380233c0ee451221b93f1ee03236b381227228a5c1df9fb"
PREFLIGHT_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_v2_preflight_v1_attempt_03"
CANDIDATE_A_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_v2_candidate_a_attempt_01"
CANDIDATE_B_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_v2_candidate_b_attempt_01"
AGGREGATE_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_v2_aggregate_v1_attempt_01"

C1_CHECKPOINT_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_step_300_checkpoint.v1.pt"
)
C1_CHECKPOINT_SHA256 = "d7a5581df3fb460337e8655865a200886739cdeab0fb6074213cc4b2cc66c6e9"
N4_ATTRIBUTION_REPORT_PATH = Path(
    "outputs/20260914_n4_endpoint_attribution/runs/"
    "lightgen_2d_n4_endpoint_attribution_evaluation_v1_attempt_01/"
    "lightgen_2d_n4_endpoint_attribution_report.v1.json"
)
N4_ATTRIBUTION_REPORT_SHA256 = "18cef70e3ad1a2a7f81144663f6e55373d08265c4d387d2e39df2aec0c4a3388"
N4_ATTRIBUTION_SUFFICIENT_SHA256 = "59867343f90528951fcc90f6936f5a9cd2630f86aa6c9c6a167ad22a4c4c592a"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
CONTINUATION_OFFSET = 300
CONTINUATION_TOTAL_STEPS = 600
CHECKPOINT_STEPS = (0, 100, 200, 300)
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STAGE_A_FIREWALL = 0.937
MATERIAL_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
SELECTION_ROWS = 1000
SELECTION_PER_CLASS = 100
SPLIT_SEED = 1234
S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}

ADAPTER_PARAMETER_NAMES = [
    "adapter.u_real",
    "adapter.u_imag",
    "adapter.v_real",
    "adapter.v_imag",
]
A_PARAMETER_NAMES = ADAPTER_PARAMETER_NAMES + ["generator.phase"]
B_PARAMETER_NAMES = ADAPTER_PARAMETER_NAMES + ["generator.phase", "generator.depth_phase"]

PREFLIGHT_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_v2_preflight_lineage.v1.json"
PREFLIGHT_INPUT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_preflight_input.v1.json"
PREFLIGHT_CHAIN_OBS_NAME = "lightgen_2d_true_optical_depth_comparison_v2_synthetic_chain_sufficient.v1.pt"
PREFLIGHT_CHAIN_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_synthetic_chain_report.v1.json"
PREFLIGHT_CHAIN_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_v2_synthetic_chain_lineage.v1.json"
PREFLIGHT_CHAIN_BOOTSTRAP_NAME = "lightgen_2d_true_optical_depth_comparison_v2_synthetic_chain_bootstrap_input.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_preflight_checkpoint.v1.pt"
SCIENTIFIC_INPUT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_scientific_lineage.v1.json"
SUFFICIENT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_step_{step}_sufficient_observations.v1.pt"
DIAGNOSTICS_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_step_{step}_diagnostics.v1.json"
RAW_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_step_300_cyclic_raw_outputs.v1.pt"
CHECKPOINT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_{candidate}_step_{step}_checkpoint.v1.pt"
AGGREGATE_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_v2_aggregate_report.v1.json"
AGGREGATE_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_v2_aggregate_lineage.v1.json"


class InvalidN5V2(RuntimeError):
    """The V2 consumer or its persisted evidence is invalid."""


class HardBlockedN5V2(RuntimeError):
    """A valid V2 boundary prevents continuation."""

    def __init__(self, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidN5V2(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"absolute path is not allowed: {relative}")
    return (root / relative).resolve()


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidN5V2(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    return N4._file_sha256(path)


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = _sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _cpu_tree(value: Any) -> Any:
    return N4._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return N4._tree_equal(left, right)


def _digest_tree(value: Any) -> str:
    return N4._digest_tree(value)


def _tensor_finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if torch.is_tensor(value):
        raise InvalidN5V2("tensor leaked into JSON report")
    return value


def _status_lines(root: Path) -> list[str]:
    return V10.V6._status_lines(root)


def _validate_successor_contract(root: Path) -> dict[str, Any]:
    path = _path(root, CONTRACT_PATH)
    _validate_sha(path, CONTRACT_SHA256, "N5 V2 successor contract")
    contract = _read_json(path, "N5 V2 successor contract")
    _require(
        contract.get("schema") == "lightgen_2d_true_optical_depth_comparison_v2_contract.v1"
        and contract.get("result") == "READY"
        and contract.get("status") == "definition_ready"
        and contract.get("classification") == DESIGN_GATE
        and contract.get("outer_gate") == OUTER_GATE,
        "N5 V2 successor contract is not the frozen READY authority",
    )
    _require(contract.get("task_id") == DESIGN_AUTHORITY, "N5 V2 contract task identity changed")
    definition = _mapping(contract.get("definition_entrypoint"), "N5 V2 definition entrypoint")
    definition_path = _path(root, str(definition.get("path")))
    _validate_sha(definition_path, str(definition.get("sha256")), "N5 V2 definition entrypoint")
    common = _mapping(contract.get("common_execution"), "N5 V2 common execution")
    _require(common.get("additional_updates") == SCIENTIFIC_STEPS and common.get("batch") == LOGICAL_BATCH_SIZE, "N5 V2 budget changed")
    _require(common.get("evaluation_steps_relative_to_continuation") == list(CHECKPOINT_STEPS), "N5 V2 evaluation cadence changed")
    objective = _mapping(contract.get("common_objective"), "N5 V2 objective")
    _require(float(objective.get("lambda_transfer")) == LAMBDA_TRANSFER and float(objective.get("lambda_image")) == LAMBDA_IMAGE and objective.get("tuning") is False, "N5 V2 objective changed")
    _require(contract.get("stage_a", {}).get("firewall") == "R300>=0.937", "N5 V2 Stage-A firewall changed")
    predecessor = _mapping(contract.get("predecessor"), "V1 predecessor")
    _require(predecessor.get("observations_reused") is False and predecessor.get("scientific_metrics") == "not available and not consumed", "V1 boundary changed")
    _require(contract.get("one_shot", {}).get("retry") is False and contract.get("one_shot", {}).get("v1_observations_reused") is False, "V2 one-shot boundary changed")
    return contract


def _validate_n5_authorities_v2(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _validate_successor_contract(root)
    # This validates only V1's frozen design and the V2 input authorities.  It
    # does not load the old V1 scientific attempt directory.
    predecessor_contract, c1, baseline, authority_context = V1._validate_n5_authorities(root)
    _require(predecessor_contract.get("result") == "READY", "V1 predecessor design is not READY")
    checkpoint = _mapping(c1["checkpoint"], "V2 C1 checkpoint")
    _require(checkpoint.get("trainable_parameter_names") == A_PARAMETER_NAMES, "C1 trainable surface changed")
    _require(checkpoint.get("checkpoint_step") == SCIENTIFIC_STEPS, "C1 checkpoint step changed")
    _require(checkpoint.get("stage_b_executed") is False and checkpoint.get("generation_executed") is False and checkpoint.get("formal_promotion") is False, "C1 claim boundary changed")
    n4_report = _read_json(_path(root, N4_ATTRIBUTION_REPORT_PATH), "N4 attribution report")
    _require(n4_report.get("result") == "VALID_COMPLETED" and n4_report.get("classification") == "GENERATOR_COADAPTATION_DOMINANT", "N4 attribution authority changed")
    _require(n4_report.get("outer_gate") == "N4_ENDPOINT_ATTRIBUTION_RESOLVED", "N4 attribution gate changed")
    _require(n4_report.get("original_n4_status", {}).get("aggregate") == "UNRESOLVED", "original N4 aggregate changed")
    return contract, c1, baseline, authority_context, {"predecessor_contract": predecessor_contract, "n4_report": n4_report}


# Frozen V1 code helpers.  They are used only to instantiate the same runtime
# and objective; all V2 persistence and evaluation ordering is local below.
_new_candidate = V1._new_candidate
_new_frozen_candidate = V1._new_frozen_candidate
_configure_candidate = V1._configure_candidate
_candidate_parameter_names = V1._candidate_parameter_names
_candidate_named = V1._candidate_named
_adapter_named = V1._adapter_named
_runtime_trainable_names = V1._runtime_trainable_names
_frozen_runtime_digest = V1._frozen_runtime_digest
_initial_shared_digests = V1._initial_shared_digests
_optimizer = V1._optimizer
_restore_optimizer = V1._restore_optimizer
_objective_components = V1._objective_components
_gradient_probe = V1._gradient_probe
_train_step = V1._train_step
_measure_train_batch = V1._measure_train_batch
_generator_diagnostics = V1._generator_diagnostics
_adapter_diagnostics = V1._adapter_diagnostics
_step0_equivalence = V1._step0_equivalence


def _checkpoint_payload(root: Path, contract_ref: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, step: int, initial_phase: torch.Tensor) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_true_optical_depth_comparison_v2_checkpoint.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "contract": dict(contract_ref),
        "source_c1_step300_checkpoint": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256},
        "checkpoint_step": step,
        "relative_step": step,
        "global_step": CONTINUATION_OFFSET + step,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "adapter_state": _cpu_tree(adapter.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(V10.V6._capture_rng()),
        "initial_generator_phase": _cpu_tree(initial_phase),
        "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
        "lambda_transfer": LAMBDA_TRANSFER,
        "lambda_image": LAMBDA_IMAGE,
        "rank": ADAPTER_RANK,
        "trainable_parameter_names": _candidate_parameter_names(candidate),
        "data_roles": {"TRAIN": True, "DEV": True, "protected": False, "official_test": False, "sealed": False},
        "stage_b_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
        "terminal_sampler_rule": "step300 saves cursor=600 without requesting another batch",
        "source_c1_runtime_digest": _digest_tree(_cpu_tree(c1["runtime_state"])),
        "source_c1_adapter_digest": _digest_tree(_cpu_tree(c1["adapter_state"])),
    }


def _validate_checkpoint_reload(root: Path, path: Path, payload: Mapping[str, Any], bundle: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler) -> dict[str, Any]:
    # V1's validator has the corrected terminal branch: it probes a cloned
    # sampler for nonterminal checkpoints and only checks cursor at step300.
    return V1._validate_checkpoint_reload(root, path, payload, bundle, c1, candidate, consumer, adapter, optimizer, sampler)


def _save_tensor_artifact(root: Path, path: Path, payload: Mapping[str, Any], name: str) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=name)
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), name)
    _require(loaded.get("schema") == payload.get("schema"), f"{name} schema reload changed")
    for key in ("dev_labels", "dev_smooth_logits", "dev_smooth_correct", "dev_ols_margins"):
        if torch.is_tensor(payload.get(key)):
            _require(torch.equal(loaded[key], payload[key]), f"{name} reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _canonical_sufficient_payload(contract_ref: Mapping[str, Any], candidate: str, step: int, evaluation: Mapping[str, Any], cyclic: Mapping[str, Any], dev_labels: torch.Tensor, train_metric: Mapping[str, Any]) -> dict[str, Any]:
    dev_logits = evaluation["_smooth_logits"].detach().cpu()
    labels_cpu = dev_labels.detach().cpu()
    dev_correct = dev_logits.argmax(dim=1).eq(labels_cpu)
    cyclic_observations = {
        key: value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in cyclic["_observations"].items()
        if key not in {"raw_base", "raw_shifted"}
    }
    return {
        "schema": "lightgen_2d_true_optical_depth_comparison_v2_sufficient_observations.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": dict(contract_ref),
        "candidate": candidate,
        "relative_step": step,
        "global_step": CONTINUATION_OFFSET + step,
        "dev_labels": labels_cpu,
        "dev_smooth_logits": dev_logits,
        "dev_smooth_correct": dev_correct,
        "dev_ols_margins": evaluation["_margins"].detach().cpu(),
        "dev_anchor_distances": evaluation["_anchor_distances"].detach().cpu(),
        "dev_anchor_predictions": evaluation["_anchor_predictions"].detach().cpu(),
        "dev_anchor_correct_margin": evaluation["_anchor_correct_margin"].detach().cpu(),
        "dev_image_loss": float(evaluation["losses"]["L_image"]),
        "dev_image_count": int(labels_cpu.numel()),
        "evaluation_public": _public({key: value for key, value in evaluation.items() if not key.startswith("_")}),
        "cyclic_public": _public({key: value for key, value in cyclic.items() if not key.startswith("_")}),
        "cyclic_observations": cyclic_observations,
        "train_metric": dict(train_metric),
        "raw_validity": {"dev": evaluation["raw_validity"], "cyclic": cyclic["raw_validity"]},
        "collapse": evaluation["collapse"],
        "forward_mode": "scientific evaluation; canonical sufficient observations persisted before optional diagnostics",
        "diagnostics_persisted_after_sufficient": True,
    }


def _save_sufficient(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, step: int, payload: Mapping[str, Any]) -> dict[str, str]:
    path = output_dir / SUFFICIENT_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    return _save_tensor_artifact(root, path, payload, f"N5 V2 {candidate} sufficient observations step {step}")


def _save_diagnostics(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, step: int, sufficient_ref: Mapping[str, Any], adapter_diag: Mapping[str, Any], generator_diag: Mapping[str, Any], cyclic_diag: Mapping[str, Any], runtime_digest: str, adapter_digest: str) -> dict[str, str]:
    path = output_dir / DIAGNOSTICS_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    payload = {
        "schema": "lightgen_2d_true_optical_depth_comparison_v2_diagnostics.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "relative_step": step,
        "contract": dict(contract_ref),
        "sufficient_observations": dict(sufficient_ref),
        "adapter": dict(adapter_diag),
        "generator": dict(generator_diag),
        "cyclic": dict(cyclic_diag),
        "runtime_state_digest": runtime_digest,
        "adapter_state_digest": adapter_digest,
        "computed_after_sufficient_reload": True,
    }
    N4.atomic_write_new_json(path, payload, artifact_kind=f"N5 V2 {candidate} diagnostics")
    return {"path": _relative(root, path), "sha256": _sha256(path), "reloaded": True}


def _cyclic_diagnostics_from_observations(sufficient: Mapping[str, Any]) -> dict[str, Any]:
    observations = _mapping(sufficient.get("cyclic_observations"), "cyclic sufficient observations")
    template = observations["template_shifted_margin"] - observations["template_base_margin"]
    smooth = observations["smooth_shifted_margin"] - observations["smooth_base_margin"]
    _tensor_finite(template, "cyclic template delta")
    _tensor_finite(smooth, "cyclic Smooth delta")
    return {
        "rows": int(template.numel()),
        "template_delta_mean": float(template.to(torch.float64).mean()),
        "smooth_delta_mean": float(smooth.to(torch.float64).mean()),
        "template_positive_fraction": float((template > 0).to(torch.float64).mean()),
        "smooth_positive_fraction": float((smooth > 0).to(torch.float64).mean()),
        "computed_without_forward_from_persisted_observations": True,
    }


def _save_raw(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, cyclic_observations: Mapping[str, Any]) -> dict[str, str]:
    payload = {
        "schema": "lightgen_2d_true_optical_depth_comparison_v2_cyclic_raw_outputs.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "contract": dict(contract_ref),
        "selection_indices": cyclic_observations["selection_indices"],
        "labels": cyclic_observations["labels"],
        "targets": cyclic_observations["targets"],
        "raw_base": cyclic_observations["raw_base"],
        "raw_shifted": cyclic_observations["raw_shifted"],
    }
    path = output_dir / RAW_NAME.format(candidate=candidate.lower())
    return _save_tensor_artifact(root, path, payload, f"N5 V2 {candidate} cyclic raw outputs")


def _save_checkpoint(root: Path, output_dir: Path, bundle: Mapping[str, Any], c1: Mapping[str, Any], contract_ref: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, step: int, initial_phase: torch.Tensor) -> dict[str, str]:
    payload = _checkpoint_payload(root, contract_ref, c1, candidate, consumer, adapter, optimizer, sampler, step, initial_phase)
    path = output_dir / CHECKPOINT_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    N4.atomic_torch_save_new(path, payload, artifact_kind=f"N5 V2 {candidate} checkpoint")
    return _validate_checkpoint_reload(root, path, payload, bundle, c1, candidate, consumer, adapter, optimizer, sampler)


def _evaluate_persist_first(consumer: Any, adapter: Any, bundle: Mapping[str, Any], dev_images: torch.Tensor, dev_labels: torch.Tensor, selection_images: torch.Tensor, selection_labels: torch.Tensor, selection_indices: torch.Tensor, step: int, train_metric: Mapping[str, Any], train_trace: Mapping[str, Any] | None, candidate: str, initial_phase: torch.Tensor, contract_ref: Mapping[str, Any], output_dir: Path, root: Path, counters: dict[str, int]) -> dict[str, Any]:
    runtime_before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    adapter_before = _digest_tree(_cpu_tree(adapter.state_dict()))
    counters["scientific_forward_evaluations"] += 1
    with torch.no_grad():
        evaluation = N4._evaluate(
            consumer,
            adapter,
            bundle["authorities"]["smooth"],
            dev_images,
            dev_labels,
            bundle["authorities"]["prototypes"],
            bundle["authorities"]["templates"],
            step=step,
            train_trace=train_trace,
        )
    counters["scientific_forward_evaluations"] += 1
    with torch.no_grad():
        cyclic = V10._cyclic_transfer_v10(
            consumer,
            adapter,
            bundle["authorities"]["smooth"],
            selection_images,
            selection_labels,
            selection_indices,
            bundle["authorities"]["prototypes"],
            bundle["authorities"]["templates"],
        )
    _require(runtime_before == _digest_tree(_cpu_tree(consumer.runtime.state_dict())), f"{candidate} evaluation changed runtime state")
    _require(adapter_before == _digest_tree(_cpu_tree(adapter.state_dict())), f"{candidate} evaluation changed adapter state")

    # This is deliberately the first operation after the scientific forward.
    canonical = _canonical_sufficient_payload(contract_ref, candidate, step, evaluation, cyclic, dev_labels, train_metric)
    sufficient_ref = _save_sufficient(root, output_dir, contract_ref, candidate, step, canonical)

    # Optional diagnostics begin only after the write-once observation has
    # been reloaded and checked by _save_sufficient.
    loaded_sufficient = _mapping(N4.safe_torch_load(_path(root, sufficient_ref["path"]), expected_sha256=sufficient_ref["sha256"], map_location="cpu"), f"{candidate} sufficient reload")
    cyclic_diag = _cyclic_diagnostics_from_observations(loaded_sufficient)
    adapter_diag = _adapter_diagnostics(adapter)
    generator_diag = _generator_diagnostics(consumer, candidate, initial_phase)
    diagnostics_ref = _save_diagnostics(root, output_dir, contract_ref, candidate, step, sufficient_ref, adapter_diag, generator_diag, cyclic_diag, runtime_before, adapter_before)
    raw_ref = None
    if step == 300:
        raw_ref = _save_raw(root, output_dir, contract_ref, candidate, {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in cyclic["_observations"].items()})

    return {
        "candidate": candidate,
        "relative_step": step,
        "global_step": CONTINUATION_OFFSET + step,
        "dev_labels": loaded_sufficient["dev_labels"],
        "dev_correct": loaded_sufficient["dev_smooth_correct"],
        "evaluation_public": loaded_sufficient["evaluation_public"],
        "cyclic_public": loaded_sufficient["cyclic_public"],
        "dev_image_loss": float(loaded_sufficient["dev_image_loss"]),
        "train_metric": dict(train_metric),
        "sufficient_ref": sufficient_ref,
        "diagnostics_ref": diagnostics_ref,
        "adapter_diagnostics": adapter_diag,
        "generator_diagnostics": generator_diag,
        "cyclic_diagnostics": cyclic_diag,
        "cyclic_observations": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in cyclic["_observations"].items()},
        "raw_ref": raw_ref,
    }


def _preflight_smoke(root: Path, output_dir: Path, bundle: Mapping[str, Any], c1: Mapping[str, Any], contract_ref: Mapping[str, Any], candidate: str, images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer, adapter = _new_candidate(bundle, c1, candidate, device)
    firewall = _configure_candidate(consumer, adapter, candidate, trainable=True)
    optimizer = _optimizer(consumer, adapter, candidate)
    _restore_optimizer(optimizer, _mapping(c1["checkpoint"]["optimizer_state"], "C1 optimizer state"), candidate)
    sampler = ContinuationSampler(bundle["train_indices"], bundle["labels"])
    initial_phase = c1["runtime_state"]["generator.phase"]
    traces = []
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        traces.append(_train_step(consumer, adapter, optimizer, candidate, images.index_select(0, indices).to(device), batch_labels.to(device), prototypes, templates, step))
    path = output_dir / PREFLIGHT_CHECKPOINT_NAME.format(candidate=candidate.lower())
    payload = _checkpoint_payload(root, contract_ref, c1, candidate, consumer, adapter, optimizer, sampler, PRECHECK_STEPS, initial_phase)
    N4.atomic_torch_save_new(path, payload, artifact_kind=f"N5 V2 {candidate} disposable preflight checkpoint")
    reload_ref = _validate_checkpoint_reload(root, path, payload, bundle, c1, candidate, consumer, adapter, optimizer, sampler)
    return {"firewall": firewall, "traces": traces, "disposable_updates": PRECHECK_STEPS, "scientific_updates": 0, "checkpoint_reload": reload_ref, "sampler_cursor_after_smoke": sampler.cursor, "scientific_meaning": "none"}


def _production_chain_preflight(root: Path, output_dir: Path, bundle: Mapping[str, Any], c1: Mapping[str, Any], contract_ref: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    consumer, adapter = _new_frozen_candidate(bundle, c1, "B_TRUE_DEPTH_C1", device)
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    images = (ramp + labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    images_device = images.to(device)
    with torch.no_grad():
        # Keep labels on CPU: the frozen evaluator returns Smooth logits on
        # CPU before its per-class mask indexing.  Images are moved by the
        # production evaluator/cyclic helper as needed.
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], images_device, labels, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step=0, train_trace=None)
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], images_device, labels, labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    canonical = _canonical_sufficient_payload(contract_ref, "B_TRUE_DEPTH_C1", 0, evaluation, cyclic, labels, {"scientific_meaning": "none"})
    obs_path = output_dir / PREFLIGHT_CHAIN_OBS_NAME
    obs_ref = _save_tensor_artifact(root, obs_path, canonical, "N5 V2 synthetic production-chain sufficient observations")
    loaded = _mapping(N4.safe_torch_load(obs_path, expected_sha256=obs_ref["sha256"], map_location="cpu"), "synthetic production-chain sufficient observations")
    cyclic_diag = _cyclic_diagnostics_from_observations(loaded)
    adapter_diag = _adapter_diagnostics(adapter)
    generator_diag = _generator_diagnostics(consumer, "B_TRUE_DEPTH_C1", c1["runtime_state"]["generator.phase"])
    delta = loaded["dev_smooth_correct"].to(torch.float64) - loaded["dev_smooth_correct"].to(torch.float64)
    bootstrap = V10.V6._paired_bootstrap(delta, loaded["dev_labels"], replicates=17, seed=BOOTSTRAP_SEED)
    bootstrap_payload = {"schema": "lightgen_2d_true_optical_depth_comparison_v2_synthetic_bootstrap_input.v1", "scientific_meaning": "none", "delta": [float(value) for value in delta.tolist()], "bootstrap": {key: value for key, value in bootstrap.items() if key != "delta_records"}}
    bootstrap_path = output_dir / PREFLIGHT_CHAIN_BOOTSTRAP_NAME
    N4.atomic_write_new_json(bootstrap_path, bootstrap_payload, artifact_kind="N5 V2 synthetic bootstrap input")
    report_payload = {"schema": "lightgen_2d_true_optical_depth_comparison_v2_synthetic_chain_report.v1", "version": 1, "task_id": TASK_ID, "result": "VALID_COMPLETED", "scientific_meaning": "none", "sufficient_observations": obs_ref, "adapter_diagnostics": adapter_diag, "generator_diagnostics": generator_diag, "cyclic_diagnostics": cyclic_diag, "bootstrap_input": {"path": _relative(root, bootstrap_path), "sha256": _sha256(bootstrap_path)}, "chain": ["forward-result", "canonical sufficient builder", "write-once persistence", "reload/hash/shape verification", "adapter diagnostic", "generator diagnostic", "cyclic diagnostic from persisted observations", "bootstrap input", "report", "lineage"], "scientific_forward": False, "scientific_updates": 0}
    report_path = output_dir / PREFLIGHT_CHAIN_REPORT_NAME
    N4.atomic_write_new_json(report_path, report_payload, artifact_kind="N5 V2 synthetic production-chain report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_CHAIN_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_v2_synthetic_chain_lineage.v1", "version": 1, "task_id": TASK_ID, "result": "VALID_COMPLETED", "scientific_meaning": "none", "report": report_ref, "scientific_forward": False, "scientific_updates": 0}, artifact_kind="N5 V2 synthetic production-chain lineage")
    return {"status": "PASS", "observation": obs_ref, "diagnostics": {"adapter": adapter_diag, "generator": generator_diag, "cyclic": cyclic_diag}, "bootstrap_input": {"path": _relative(root, bootstrap_path), "sha256": _sha256(bootstrap_path)}, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_meaning": "none", "scientific_forward": False, "scientific_updates": 0}


def _preflight_report_payload(started: Mapping[str, Any], contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any], authorities: Mapping[str, Any], schedule: Mapping[str, Any], step0: Mapping[str, Any], shared: Mapping[str, Any], gradients: Mapping[str, Any], smokes: Mapping[str, Any], synthetic: Mapping[str, Any], chain: Mapping[str, Any], c1: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_true_optical_depth_comparison_v2_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "AUTHORIZABLE",
        "status": "valid_completed",
        "classification": "LIGHTGEN_2D_TRUE_OPTICAL_DEPTH_COMPARISON_V2_AUTHORIZABLE",
        "outer_gate": OUTER_GATE,
        "design_authority": DESIGN_AUTHORITY,
        "contract": dict(contract_ref),
        "provenance": {"git_start": dict(started)},
        "authorities": dict(authorities),
        "runtime": {"device": dict(device_facts), "production_grid": PRODUCTION_GRID},
        "source_c1": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256, "checkpoint_step": 300, "runtime_digest": _digest_tree(_cpu_tree(c1["runtime_state"])), "adapter_digest": _digest_tree(_cpu_tree(c1["adapter_state"]))},
        "continuation_schedule": schedule,
        "step0_equivalence": step0,
        "shared_initialization": shared,
        "gradient_probes": gradients,
        "disposable_smokes": smokes,
        "synthetic_true_depth_checks": synthetic,
        "synthetic_production_chain": chain,
        "dataset": _public(dataset),
        "checks": {"authorities": True, "v1_observations_not_consumed": True, "shared_c1_initialization": True, "step0_real_path_equivalence": True, "production_true_depth_zero_phase_equivalence": True, "midpoint_geometry": True, "trainable_frozen_firewall": True, "finite_usable_gradients": True, "negative_controls": True, "three_step_real_disposable_updates": True, "checkpoint_reload_resume": True, "terminal_sampler_no_next_batch": True, "synthetic_forward_to_lineage_chain": True, "missing_adapter_observation_regression": hasattr(V10, "_adapter_observation") and not hasattr(N4, "_adapter_observation")},
        "claim_boundary": {"engineering_preflight_real_updates": PRECHECK_STEPS * 2, "synthetic_preflight_forward_evaluations": 2, "scientific_data_consumed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "scientific_training_executed": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False},
        "one_shot": {"preflight": 1, "candidate_a_scientific_attempts": 1, "candidate_b_scientific_attempts": 1, "retry": False, "v1_retry_or_continuation": False, "v1_observations_reused": False},
        "next": "proceed to exactly one V2 A and one V2 B 300-update scientific comparison",
    }


def run_preflight(root: Path) -> dict[str, Any]:
    _require(not PREFLIGHT_DIR.exists(), f"refusing to reuse V2 preflight directory: {_path(root, PREFLIGHT_DIR)}")
    for directory in (CANDIDATE_A_DIR, CANDIDATE_B_DIR, AGGREGATE_DIR):
        _require(not directory.exists(), f"V2 one-shot output boundary is not fresh: {_path(root, directory)}")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_true_optical_depth_comparison_v2_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_V2_PREFLIGHT_HARNESS", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_real_updates": 0, "scientific_data_consumed": False, "scientific_updates": 0, "scientific_forward_evaluations": 0, "scientific_training_executed": False}}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from frozen V2 authorization")
        contract, c1, baseline, authority_context, extra = _validate_n5_authorities_v2(root)
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for V2 scientific continuation")
        bundle = N4._validate_v11_authorities(root)
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        _require(len(dataset.get("protected", [])) == 0 if isinstance(dataset.get("protected"), list) else dataset.get("protected", 0) == 0, "protected data role is not empty")
        V10.V6._restore_rng(_mapping(c1["checkpoint"]["rng_state"], "C1 source RNG"))
        schedule = ContinuationSampler(train_indices, labels)
        first_indices, first_labels = schedule.batch_at(CONTINUATION_OFFSET)
        first_batch = images.index_select(0, first_indices).to(device)
        first_labels_device = first_labels.to(device)
        step0 = _step0_equivalence(bundle, c1, first_batch, device)
        initial_a, initial_adapter_a = _new_candidate(bundle, c1, "A_CONTINUED_C1", device)
        initial_b, initial_adapter_b = _new_candidate(bundle, c1, "B_TRUE_DEPTH_C1", device)
        shared = _initial_shared_digests(initial_a, initial_adapter_a, initial_b, initial_adapter_b)
        contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
        gradients: dict[str, Any] = {}
        smokes: dict[str, Any] = {}
        for candidate in ("A_CONTINUED_C1", "B_TRUE_DEPTH_C1"):
            V10.V6._restore_rng(_mapping(c1["checkpoint"]["rng_state"], "C1 source RNG"))
            gradients[candidate] = _gradient_probe(bundle, c1, candidate, first_indices, first_batch, first_labels_device, device)
            smokes[candidate] = _preflight_smoke(root, PREFLIGHT_DIR, bundle, c1, contract_ref, candidate, images, labels, device)
        synthetic = N5DEF._synthetic_checks(device)
        _require(synthetic.get("status") == "PASS", "V2 production true-depth synthetic checks failed")
        chain = _production_chain_preflight(root, PREFLIGHT_DIR, bundle, c1, contract_ref, device)
        _require(chain.get("status") == "PASS", "V2 synthetic production chain failed")
        schedule_state = ContinuationSampler(train_indices, labels).state_dict()
        schedule_summary = {"schema": ContinuationSampler.SCHEMA, "seed": SPLIT_SEED, "global_offset": CONTINUATION_OFFSET, "total_steps": CONTINUATION_TOTAL_STEPS, "initial_cursor": CONTINUATION_OFFSET, "first_batch_indices_sha256": V10.V6.tensor_sha256(first_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(first_labels), "schedule_digest": _digest_tree(_cpu_tree(schedule_state))}
        authorities = dict(authority_context["authorities"])
        authorities["n5_v2_successor_contract"] = contract_ref
        authorities["v1_scientific_attempt"] = {"status": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "observations_consumed": False, "observations_reused": False}
        input_payload = {"schema": "lightgen_2d_true_optical_depth_comparison_v2_preflight_input.v1", "version": 1, "task_id": TASK_ID, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "contract": contract_ref, "candidates": ["A_CONTINUED_C1", "B_TRUE_DEPTH_C1"], "source_c1": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256}, "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "sampler": schedule_summary, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "scientific_updates": 0, "v1_observations_reused": False}
        input_path = PREFLIGHT_DIR / PREFLIGHT_INPUT_NAME
        N4.atomic_write_new_json(input_path, input_payload, artifact_kind="N5 V2 preflight input")
        payload = _preflight_report_payload(started, contract_ref, device_facts, authorities, schedule_summary, step0, shared, gradients, smokes, synthetic, chain, c1, dataset)
        payload["input"] = {"path": _relative(root, input_path), "sha256": _sha256(input_path)}
        payload["runtime"]["cuda_dll_search_path"] = V10.V6._prepare_cuda_runtime_dll_search_path()
        payload["provenance"]["git_end"] = V10.V6.git_facts(root)
    except HardBlockedN5V2 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "N5_V2_PREFLIGHT_HARD_BLOCKED", "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_V2_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {}).update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root)})
    report_path = PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N5 V2 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_v2_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "report": report_ref, "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"], "v1_observations_reused": False}, artifact_kind="N5 V2 preflight lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": 0}


def _load_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "V2 preflight authority is unavailable")
    report = _read_json(report_path, "V2 preflight report")
    lineage = _read_json(lineage_path, "V2 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == "LIGHTGEN_2D_TRUE_OPTICAL_DEPTH_COMPARISON_V2_AUTHORIZABLE", "V2 preflight is not authorizable")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0 and report.get("claim_boundary", {}).get("scientific_training_executed") is False, "V2 preflight scientific boundary changed")
    _require(report.get("claim_boundary", {}).get("v1_observations_reused") in (None, False), "V1 observations were reused")
    report_hash = _sha256(report_path)
    _require(lineage.get("report", {}).get("sha256") == report_hash and lineage.get("v1_observations_reused") is False, "V2 preflight lineage binding changed")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": report_hash}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}}


def _scientific_payload(candidate: str, started: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_true_optical_depth_comparison_v2_scientific_report.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_V2_SCIENTIFIC_HARNESS", "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "provenance": {"git_start": dict(started)}, "claim_boundary": {"scientific_training_executed": False, "scientific_data_consumed": False, "scientific_forward_evaluations": 0, "scientific_updates": 0, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"candidate": candidate, "rank": ADAPTER_RANK, "retry": False, "v1_observations_reused": False, "v1_retry_or_continuation": False, "second_arm": False, "rank_change": False, "lambda_change": False, "loss_change": False, "lr_change": False, "budget_change": False, "encoder_update": False, "ols_update": False, "stage_b": False, "generation": False, "promotion": False}}


def run_candidate(root: Path, candidate: str) -> dict[str, Any]:
    output_dir = CANDIDATE_A_DIR if candidate == "A_CONTINUED_C1" else CANDIDATE_B_DIR
    _require(not output_dir.exists(), f"refusing to reuse V2 {candidate} output directory: {_path(root, output_dir)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload = _scientific_payload(candidate, started)
    scientific_started = False
    updates_completed = 0
    counters = {"scientific_forward_evaluations": 0}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from frozen V2 authorization")
        preflight = _load_preflight(root)
        contract, c1, baseline, authority_context, extra = _validate_n5_authorities_v2(root)
        bundle = N4._validate_v11_authorities(root)
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for V2 scientific continuation")
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        dev_images = images.index_select(0, development_indices).contiguous()
        dev_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, development_indices)
        _require(V10.V6.tensor_sha256(selection_indices) == N4.SELECTION_INDICES_SHA256 and V10.V6.tensor_sha256(selection_labels) == N4.SELECTION_LABELS_SHA256, "V2 selection authority changed")
        continuation = ContinuationSampler(train_indices, labels)
        V10.V6._restore_rng(_mapping(c1["checkpoint"]["rng_state"], "C1 source RNG"))
        consumer, adapter = _new_candidate(bundle, c1, candidate, device)
        firewall = _configure_candidate(consumer, adapter, candidate, trainable=True)
        optimizer = _optimizer(consumer, adapter, candidate)
        _restore_optimizer(optimizer, _mapping(c1["checkpoint"]["optimizer_state"], "C1 source optimizer"), candidate)
        initial_phase = c1["runtime_state"]["generator.phase"].detach().cpu()
        first_indices, first_labels = continuation.batch_at(CONTINUATION_OFFSET)
        first_batch = images.index_select(0, first_indices).to(device)
        first_labels_device = first_labels.to(device)
        contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
        preflight_ref = {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}
        input_payload = {"schema": "lightgen_2d_true_optical_depth_comparison_v2_scientific_input.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "contract": contract_ref, "preflight": preflight_ref, "source_c1": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256, "checkpoint_step": 300}, "protocol": {"objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "rank": ADAPTER_RANK, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "batch": LOGICAL_BATCH_SIZE, "additional_updates": SCIENTIFIC_STEPS, "relative_checkpoints": list(CHECKPOINT_STEPS), "global_offset": CONTINUATION_OFFSET, "continuation_total_steps": CONTINUATION_TOTAL_STEPS, "trainable": _candidate_parameter_names(candidate)}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "one_shot": {"candidate_attempt": 1, "retry": False, "v1_observations_reused": False}, "claim_boundary": payload["claim_boundary"]}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME.format(candidate=candidate.lower())
        N4.atomic_write_new_json(input_path, input_payload, artifact_kind=f"N5 V2 {candidate} scientific input")
        scientific_started = True
        with torch.no_grad():
            initial_train_metric = _measure_train_batch(consumer, adapter, first_batch, first_labels_device, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device))
        payload.update({"contract": contract_ref, "preflight": preflight_ref, "source_c1": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256, "checkpoint_step": 300}, "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID, "candidate": candidate}, "authorities": {**authority_context["authorities"], "n5_v2_successor_contract": contract_ref, "v1_scientific_attempt": {"status": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "observations_consumed": False, "observations_reused": False}}, "dataset": _public(dataset), "firewall": firewall, "objective": input_payload["protocol"], "initial_state": {"source_c1_runtime_digest": _digest_tree(_cpu_tree(c1["runtime_state"])), "source_c1_adapter_digest": _digest_tree(_cpu_tree(c1["adapter_state"])), "candidate_frozen_runtime_digest": _frozen_runtime_digest(consumer, candidate), "generator_phase": _generator_diagnostics(consumer, candidate, initial_phase), "input": {"path": _relative(root, input_path), "sha256": _sha256(input_path)}}, "sampler": {"schema": ContinuationSampler.SCHEMA, "seed": SPLIT_SEED, "global_offset": CONTINUATION_OFFSET, "total_steps": CONTINUATION_TOTAL_STEPS, "first_batch_indices_sha256": V10.V6.tensor_sha256(first_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(first_labels)}, "data_roles": input_payload["data_roles"], "v1_observations_reused": False})
        records: dict[str, Any] = {}
        train_metrics: dict[str, Any] = {"0": initial_train_metric}
        sufficient_refs: dict[str, Any] = {}
        diagnostics_refs: dict[str, Any] = {}
        checkpoint_refs: dict[str, Any] = {}
        training_start = time.perf_counter()
        record = _evaluate_persist_first(consumer, adapter, bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, 0, initial_train_metric, None, candidate, initial_phase, contract_ref, output_dir, root, counters)
        records["0"] = record
        sufficient_refs["0"] = record["sufficient_ref"]
        diagnostics_refs["0"] = record["diagnostics_ref"]
        checkpoint_refs["0"] = _save_checkpoint(root, output_dir, bundle, c1, contract_ref, candidate, consumer, adapter, optimizer, continuation, 0, initial_phase)
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = continuation.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, adapter, optimizer, candidate, batch, batch_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step)
            updates_completed = step
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                with torch.no_grad():
                    train_metric = _measure_train_batch(consumer, adapter, first_batch, first_labels_device, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device))
                train_metrics[str(step)] = train_metric
                record = _evaluate_persist_first(consumer, adapter, bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step, train_metric, trace, candidate, initial_phase, contract_ref, output_dir, root, counters)
                records[str(step)] = record
                sufficient_refs[str(step)] = record["sufficient_ref"]
                diagnostics_refs[str(step)] = record["diagnostics_ref"]
                checkpoint_refs[str(step)] = _save_checkpoint(root, output_dir, bundle, c1, contract_ref, candidate, consumer, adapter, optimizer, continuation, step, initial_phase)
                print(json.dumps({"event": "N5_V2_scientific_progress", "candidate": candidate, "relative_step": step, "global_step": CONTINUATION_OFFSET + step, "elapsed_seconds": time.perf_counter() - training_start}, ensure_ascii=False), flush=True)
        _require(set(records) == {"0", "100", "200", "300"} and set(sufficient_refs) == {"0", "100", "200", "300"} and set(diagnostics_refs) == {"0", "100", "200", "300"} and set(checkpoint_refs) == {"0", "100", "200", "300"}, f"{candidate} persisted endpoint set is incomplete")
        final_record = records["300"]
        baseline_correct = baseline["n4_sufficient"]["endpoints"]["C1"]["dev_smooth_correct"].to(torch.float64)
        step0_correct = records["0"]["dev_correct"].to(torch.float64)
        final_correct = final_record["dev_correct"].to(torch.float64)
        _require(torch.equal(baseline["n4_sufficient"]["dev_labels"], dev_labels.cpu()), "N4 C1 baseline DEV rows do not match V2 DEV rows")
        optimization_ci = V10.V6._paired_bootstrap(final_correct - baseline_correct, dev_labels.cpu(), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        training_ci = V10.V6._paired_bootstrap(final_correct - step0_correct, dev_labels.cpu(), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        reconstruction_delta = float(final_record["dev_image_loss"] - records["0"]["dev_image_loss"])
        raw_valid = bool(all(record["evaluation_public"]["raw_validity"]["finite"] and record["evaluation_public"]["raw_validity"]["structured"] and record["evaluation_public"]["raw_validity"]["nonconstant"] for record in records.values()) and final_record["cyclic_public"]["raw_validity"]["base"]["finite"] and final_record["cyclic_public"]["raw_validity"]["shifted"]["finite"])
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": f"N5_V2_{candidate}_SCIENTIFIC_COMPLETED", "scientific_updates": SCIENTIFIC_STEPS, "scientific_forward_evaluations": counters["scientific_forward_evaluations"], "evaluations": {step: record["evaluation_public"] for step, record in records.items()}, "train_metrics": train_metrics, "sufficient_observations": sufficient_refs, "diagnostics_artifacts": diagnostics_refs, "checkpoints": checkpoint_refs, "cyclic_raw_outputs": final_record["raw_ref"], "optimization_horizon": {"baseline": "C1 step300 Smooth=0.4838", "R300": float(records["300"]["evaluation_public"]["smooth"]["macro_recall"]), "Delta_A": float(records["300"]["evaluation_public"]["smooth"]["macro_recall"] - 0.4838), "paired_bootstrap": {key: value for key, value in optimization_ci.items() if key != "delta_records"}}, "training_delta": {"R0": float(records["0"]["evaluation_public"]["smooth"]["macro_recall"]), "R100": float(records["100"]["evaluation_public"]["smooth"]["macro_recall"]), "R200": float(records["200"]["evaluation_public"]["smooth"]["macro_recall"]), "R300": float(records["300"]["evaluation_public"]["smooth"]["macro_recall"]), "Delta_R": float(records["300"]["evaluation_public"]["smooth"]["macro_recall"] - records["0"]["evaluation_public"]["smooth"]["macro_recall"]), "paired_bootstrap": {key: value for key, value in training_ci.items() if key != "delta_records"}}, "reconstruction": {"DEV_L_image_step0": float(records["0"]["dev_image_loss"]), "DEV_L_image_step300": float(records["300"]["dev_image_loss"]), "Delta_L_image": reconstruction_delta, "max_allowed_delta": RECONSTRUCTION_MAX_DELTA}, "cyclic_transfer": records["300"]["cyclic_public"], "raw_diversity": {"step300": {"dev": records["300"]["evaluation_public"]["raw_validity"], "collapse": records["300"]["evaluation_public"]["collapse"], "cyclic": records["300"]["cyclic_public"]["raw_validity"]}}, "diagnostics": {"step0": {"adapter": records["0"]["adapter_diagnostics"], "generator": records["0"]["generator_diagnostics"], "cyclic": records["0"]["cyclic_diagnostics"]}, "step300": {"adapter": records["300"]["adapter_diagnostics"], "generator": records["300"]["generator_diagnostics"], "cyclic": records["300"]["cyclic_diagnostics"]}, "artifacts": diagnostics_refs, "true_depth_geometry": {"candidate_b_only": candidate == "B_TRUE_DEPTH_C1", "midpoint_shape": [PRODUCTION_GRID, PRODUCTION_GRID], "active_support": [100, 100], "midpoint_crop": False, "outside_support": "fixed_transparent", "half_distance_m": float(N5DEF.HALF_PROPAGATION_DISTANCE_M)}}, "raw_validity_pass": raw_valid, "claim_boundary": {"scientific_training_executed": True, "scientific_data_consumed": True, "scientific_forward_evaluations": counters["scientific_forward_evaluations"], "scientific_updates": SCIENTIFIC_STEPS, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {**payload["governance"], "scientific_attempt": 1, "adapter_update": True, "generator_phase_update": True, "depth_phase_update": candidate == "B_TRUE_DEPTH_C1", "retries": False, "no_adaptive_change": True}, "next": "candidate result is bounded input to the frozen V2 pairwise outer gate; Stage B and generation remain closed"})
    except HardBlockedN5V2 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "N5_V2_SCIENTIFIC_HARD_BLOCKED", "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_V2_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific retry; persisted observations, if any, remain immutable"})
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {}).update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root), "scientific_started": scientific_started})
        payload["claim_boundary"]["scientific_training_executed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_data_consumed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_updates"] = updates_completed
        payload["claim_boundary"]["scientific_forward_evaluations"] = counters["scientific_forward_evaluations"]
    report_path = output_dir / SCIENTIFIC_REPORT_NAME.format(candidate=candidate.lower())
    N4.atomic_write_new_json(report_path, payload, artifact_kind=f"N5 V2 {candidate} scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME.format(candidate=candidate.lower())
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_v2_scientific_lineage.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "report": report_ref, "contract": payload.get("contract"), "preflight": payload.get("preflight"), "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"], "scientific_observations_new": bool(scientific_started), "v1_observations_reused": False, "retry": False}, artifact_kind=f"N5 V2 {candidate} scientific lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "candidate": candidate, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": updates_completed, "scientific_forward_evaluations": counters["scientific_forward_evaluations"], "smooth_R300": payload.get("evaluations", {}).get("300", {}).get("smooth", {}).get("macro_recall"), "next": payload.get("next")}


def _load_step_sufficient(root: Path, report: Mapping[str, Any], candidate: str, step: int) -> Mapping[str, Any]:
    ref = _mapping(report.get("sufficient_observations", {}).get(str(step)), f"N5 V2 {candidate} sufficient step {step}")
    path = _path(root, ref["path"])
    _validate_sha(path, str(ref["sha256"]), f"N5 V2 {candidate} sufficient step {step}")
    return _mapping(N4.safe_torch_load(path, expected_sha256=str(ref["sha256"]), map_location="cpu"), f"N5 V2 {candidate} sufficient step {step}")


def _classification(a_report: Mapping[str, Any], b_report: Mapping[str, Any], a_sufficient: Mapping[str, Any], b_sufficient: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    a300 = a_sufficient["dev_smooth_correct"].to(torch.float64)
    b300 = b_sufficient["dev_smooth_correct"].to(torch.float64)
    delta_depth = b300 - a300
    depth_ci = V10.V6._paired_bootstrap(delta_depth, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    a_material = float(a_report["optimization_horizon"]["Delta_A"]) >= MATERIAL_DELTA and float(a_report["optimization_horizon"]["paired_bootstrap"]["ci_lower"]) > 0.0
    depth_material = float(delta_depth.mean()) >= MATERIAL_DELTA and float(depth_ci["ci_lower"]) > 0.0
    b_reconstruction = float(b_report["reconstruction"]["Delta_L_image"])
    b_transfer = bool(b_report["cyclic_transfer"]["template"]["pass"] and b_report["cyclic_transfer"]["smooth"]["pass"] and b_report["cyclic_transfer"]["broad"] and not b_report["cyclic_transfer"]["pair_dominated"])
    raw = bool(b_report["raw_validity_pass"])
    primary_supported = bool(depth_material and b_reconstruction <= RECONSTRUCTION_MAX_DELTA and b_transfer and raw)
    if a_material and not depth_material:
        classification = "OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED"
    elif depth_material:
        classification = "TRUE_DEPTH_SUPPORTED"
    elif not a_material and not depth_material:
        classification = "LINEAR_INTERFACE_OPTICAL_SYNTHESIS_CEILING_OR_UNRESOLVED"
    else:
        classification = "N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_MIXED_OR_UNRESOLVED"
    return {"classification": classification, "optimization_horizon_material": a_material, "depth_material": depth_material, "primary_gate": {"name": OUTER_GATE, "supported": primary_supported, "Delta_depth": float(delta_depth.mean()), "paired_bootstrap": {key: value for key, value in depth_ci.items() if key != "delta_records"}, "B_reconstruction_pass": b_reconstruction <= RECONSTRUCTION_MAX_DELTA, "B_cyclic_transfer_pass": b_transfer, "B_raw_validity_pass": raw}}


def _aggregate(root: Path, contract: Mapping[str, Any], preflight: Mapping[str, Any], a_result: Mapping[str, Any], b_result: Mapping[str, Any], a_report: Mapping[str, Any], b_report: Mapping[str, Any], a_steps: Mapping[str, Mapping[str, Any]], b_steps: Mapping[str, Mapping[str, Any]], n4_sufficient: Mapping[str, Any]) -> dict[str, Any]:
    _require(not AGGREGATE_DIR.exists(), f"refusing to reuse V2 aggregate directory: {_path(root, AGGREGATE_DIR)}")
    _require(a_result["result"] == "VALID_COMPLETED" and b_result["result"] == "VALID_COMPLETED", "V2 aggregate requires both valid candidates")
    output_dir = _path(root, AGGREGATE_DIR)
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = a_steps["300"]["dev_labels"]
    _require(torch.equal(labels, b_steps["300"]["dev_labels"]) and torch.equal(labels, n4_sufficient["dev_labels"]), "V2 paired DEV labels are not symmetric")
    c1_correct = n4_sufficient["endpoints"]["C1"]["dev_smooth_correct"].to(torch.float64)
    a300 = a_steps["300"]["dev_smooth_correct"].to(torch.float64)
    b300 = b_steps["300"]["dev_smooth_correct"].to(torch.float64)
    delta_a = a300 - c1_correct
    delta_depth = b300 - a300
    delta_b = b300 - c1_correct
    ci_a = V10.V6._paired_bootstrap(delta_a, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    ci_depth = V10.V6._paired_bootstrap(delta_depth, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    ci_b = V10.V6._paired_bootstrap(delta_b, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    classification = _classification(a_report, b_report, a_steps["300"], b_steps["300"], labels)
    stage_a = {}
    for candidate, report in (("A_CONTINUED_C1", a_report), ("B_TRUE_DEPTH_C1", b_report)):
        value = float(report["evaluations"]["300"]["smooth"]["macro_recall"])
        stage_a[candidate] = {"R300": value, "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if value >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
    pairwise = {
        "Delta_A_R_A300_minus_C1": {"observed_delta": float(delta_a.mean()), "paired_bootstrap": {key: value for key, value in ci_a.items() if key != "delta_records"}},
        "Delta_depth_R_B300_minus_A300": {"observed_delta": float(delta_depth.mean()), "paired_bootstrap": {key: value for key, value in ci_depth.items() if key != "delta_records"}},
        "Delta_B_R_B300_minus_C1": {"observed_delta": float(delta_b.mean()), "paired_bootstrap": {key: value for key, value in ci_b.items() if key != "delta_records"}},
    }
    contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
    payload = {"schema": "lightgen_2d_true_optical_depth_comparison_v2_aggregate.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "VALID_COMPLETED", "status": "valid_completed", "classification": classification["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "contract": contract_ref, "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]}, "endpoint_authorities": {"source_C1": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256}, "candidate_A": a_result, "candidate_B": b_result}, "candidates": {"A_CONTINUED_C1": {"report": a_result["report"], "Smooth_R300": float(a_report["evaluations"]["300"]["smooth"]["macro_recall"]), "DEV_L_image_step300": float(a_report["reconstruction"]["DEV_L_image_step300"]), "checkpoint": a_report["checkpoints"]["300"]}, "B_TRUE_DEPTH_C1": {"report": b_result["report"], "Smooth_R300": float(b_report["evaluations"]["300"]["smooth"]["macro_recall"]), "DEV_L_image_step300": float(b_report["reconstruction"]["DEV_L_image_step300"]), "checkpoint": b_report["checkpoints"]["300"]}}, "common_step0_equivalence": preflight["report"].get("step0_equivalence"), "pairwise": pairwise, "primary_gate": classification["primary_gate"], "stage_a": stage_a, "reconstruction": {"A": a_report["reconstruction"], "B": b_report["reconstruction"]}, "cyclic_transfer": {"A": a_report["cyclic_transfer"], "B": b_report["cyclic_transfer"]}, "raw_diversity": {"A": a_report["raw_diversity"], "B": b_report["raw_diversity"]}, "diagnostics": {"A": a_report["diagnostics"], "B": b_report["diagnostics"]}, "original_n4_aggregate": "UNRESOLVED", "predecessor_v1_execution": {"status": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "scientific_metrics_consumed": False, "observations_reused": False, "retry_or_continuation": False}, "claim_boundary": {"training_updates": 600, "optimizer_updates": 600, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"rank": 9, "candidate_A": "one V2 continuation arm", "candidate_B": "one V2 true-depth arm", "retries": False, "v1_retry_or_continuation": False, "v1_observations_reused": False, "rank_sweep": False, "loss_change": False, "lambda_change": False, "lr_change": False, "encoder_update": False, "W_OLS_change": False, "stage_b": False, "generation": False, "protected": False, "test": False, "sealed": False, "promotion": False}, "next": "V2 result is bounded; Stage B and generation remain closed regardless of the primary Gate"}
    report_path = output_dir / AGGREGATE_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N5 V2 aggregate report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / AGGREGATE_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_v2_aggregate_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "claim_boundary": payload["claim_boundary"], "predecessor_v1_rewritten": False, "v1_observations_reused": False}, artifact_kind="N5 V2 aggregate lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "pairwise": pairwise, "stage_a": stage_a, "predecessor_v1_execution": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "original_n4_aggregate": "UNRESOLVED"}


def run_scientific(root: Path) -> dict[str, Any]:
    preflight = _load_preflight(root)
    _require(preflight["report"].get("result") == "AUTHORIZABLE", "V2 preflight is not authorizable")
    a_result = run_candidate(root, "A_CONTINUED_C1")
    if a_result["result"] != "VALID_COMPLETED":
        return {"result": a_result["result"], "status": a_result["status"], "classification": a_result["classification"], "candidate_A": a_result, "candidate_B": "not executed because symmetric comparison could not be completed", "outer_gate": OUTER_GATE}
    b_result = run_candidate(root, "B_TRUE_DEPTH_C1")
    if b_result["result"] != "VALID_COMPLETED":
        return {"result": b_result["result"], "status": b_result["status"], "classification": b_result["classification"], "candidate_A": a_result, "candidate_B": b_result, "outer_gate": OUTER_GATE}
    a_report = _read_json(_path(root, a_result["report"]["path"]), "N5 V2 candidate A report")
    b_report = _read_json(_path(root, b_result["report"]["path"]), "N5 V2 candidate B report")
    a_steps = {str(step): _load_step_sufficient(root, a_report, "A_CONTINUED_C1", step) for step in CHECKPOINT_STEPS}
    b_steps = {str(step): _load_step_sufficient(root, b_report, "B_TRUE_DEPTH_C1", step) for step in CHECKPOINT_STEPS}
    _require(all("dev_smooth_correct" in a_steps[str(step)] and "dev_smooth_correct" in b_steps[str(step)] for step in CHECKPOINT_STEPS), "V2 paired sufficient observations are incomplete")
    _, _, baseline, _, _ = _validate_n5_authorities_v2(root)
    return _aggregate(root, _validate_successor_contract(root), preflight, a_result, b_result, a_report, b_report, a_steps, b_steps, baseline["n4_sufficient"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        result = run_preflight(root)
    elif args.action == "scientific":
        result = run_scientific(root)
    else:
        result = run_preflight(root)
        if result["result"] != "AUTHORIZABLE":
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2
        result = run_scientific(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
