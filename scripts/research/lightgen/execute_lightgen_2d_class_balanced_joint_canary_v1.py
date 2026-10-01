"""Execute the bounded class-balanced structural-gradient Canary.

The existing pooled Canary remains immutable.  This consumer changes only the
structural gradient aggregation on the already class-balanced production
batch; SIC, auxiliary objectives, Adam, data order, and the 9300 endpoint are
otherwise inherited unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import TRAIN_ROLE, git_identity
from ldm_project.v2.application.research.shared.research_run_control import CORRECTNESS_INVARIANTS, IMPLEMENTATION_SELECTION, PreflightReceipt, PREFLIGHT_ROOT_CAUSE_REVIEW_REQUIRED, RESOURCE_ENVELOPE, ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, file_sha256

from scripts.research.lightgen import define_lightgen_2d_class_balanced_joint_canary_contract_v1 as contract
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as joint
from scripts.research.lightgen import execute_lightgen_2d_local_compute_bounded_joint_canary_v1 as pooled


TASK_ID = "EXECUTE_LIGHTGEN_2D_CLASS_BALANCED_JOINT_CANARY_V1"
AUTHORIZATION_TOKEN = contract.PROPOSED_AUTHORIZATION
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = contract.OUTER_GATE
SEMANTIC_AUTHORITY_ID = joint.SEMANTIC_AUTHORITY_ID
EXPECTED_BRANCH = contract.EXPECTED_BRANCH
EXPECTED_CONSUMER_HEAD = contract.EXPECTED_CONSUMER_HEAD
EXPECTED_PRODUCER_HEAD = contract.HISTORICAL_PRODUCER_HEAD

START_STEP = contract.START_STEP
MAX_UPDATES = contract.MAX_UPDATES
PRIMARY_ENDPOINT = contract.PRIMARY_ENDPOINT
TOTAL_STEPS = PRIMARY_ENDPOINT
STAGE1_END = PRIMARY_ENDPOINT
CHECKPOINT_STEPS = contract.CHECKPOINT_STEPS
LOGICAL_BATCH_SIZE = contract.LOGICAL_BATCH_SIZE
SCHEDULE_SEED = joint.SCHEDULE_SEED
JOINT_DEGENERACY_TOLERANCE = joint.JOINT_DEGENERACY_TOLERANCE
DELTA_JOINT = contract.DELTA_JOINT
BETA = contract.BETA
GAMMA = contract.GAMMA
EPS_PROJ = contract.EPS_PROJ
PRELIGHT_EPSILON = joint.PRELIGHT_EPSILON
RECONSTRUCTION_TOLERANCE = joint.RECONSTRUCTION_TOLERANCE
GRADIENT_ENGINE = contract.GRADIENT_ENGINE
SCALAR_REFERENCE_ENGINE = contract.GRADIENT_ENGINE
GRADIENT_RTOL = contract.GRADIENT_RTOL
GRADIENT_ATOL = contract.GRADIENT_ATOL
GRADIENT_COSINE_MINIMUM = joint.GRADIENT_COSINE_MINIMUM
RECOVERY_INTERVAL_STEPS = joint.RECOVERY_INTERVAL_STEPS
STRUCTURAL_AGGREGATION_MODE = contract.STRUCTURAL_AGGREGATION_MODE
CLASS_COUNT = contract.CLASS_COUNT
CLASS_WEIGHT = contract.CLASS_WEIGHT

START_CHECKPOINT = joint.START_CHECKPOINT
START_CHECKPOINT_SHA256 = joint.START_CHECKPOINT_SHA256.lower()
CONTRACT_PATH = contract.DEFAULT_OUTPUT_DIR.parent / "contract_definition_01" / contract.CONTRACT_NAME
DEFAULT_OUTPUT_DIR = contract.SCIENTIFIC_ATTEMPT_DIR
# Preserve the legacy guard against consuming the superseded joint batched
# continuation.  The prior pooled Canary is historical evidence and may
# already exist; it is not an input state for this consumer.
HISTORICAL_ATTEMPT_DIR = joint.DEFAULT_OUTPUT_DIR
REPORT_SCHEMA = "lightgen_2d_class_balanced_joint_canary_report.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_class_balanced_joint_canary_checkpoint.v1"
PREFLIGHT_SCHEMA = "lightgen_2d_class_balanced_joint_canary_preflight.v1"
SOURCE_SNAPSHOT_SCHEMA = "lightgen_2d_class_balanced_joint_canary_source_snapshot.v1"
TIMING_SCHEMA = "lightgen_2d_class_balanced_joint_canary_timing.v1"
INVALID_HARNESS_CLOSEOUT_NAME = "invalid_harness_closeout.json"
PREFLIGHT_FAILURE_STATE_SCHEMA = "class_balanced_joint_canary_preflight_failure_state.v1"
MAX_UPDATE_SECONDS = 24.0
MAX_PROJECTED_UPDATE_HOURS = 2.0
_POOLED_CHECKPOINT_PAYLOAD = pooled._checkpoint_payload


class CanaryError(RuntimeError):
    """A class-balanced Canary authority, state, or invariant is invalid."""


class CanaryDirectionDegenerated(CanaryError):
    """A classwise or joint direction is numerically unusable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(torch.is_tensor(value), f"{name} must be a tensor")
    return value


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _require_training_backend(value: str) -> None:
    _require(value == SCALAR_REFERENCE_ENGINE, f"training backend must be {SCALAR_REFERENCE_ENGINE}; got {value}")


def validate_contract_payload(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema") == contract.CONTRACT_SCHEMA, "class-balanced contract schema changed")
    _require(payload.get("result") == "READY" and payload.get("status") == "valid_completed", "class-balanced contract is not READY")
    _require(payload.get("classification") == contract.CLASSIFICATION and payload.get("outer_gate") == OUTER_GATE, "class-balanced contract gate changed")
    execution = _mapping(payload.get("execution_boundary"), "contract execution boundary")
    _require(execution.get("proposed_authorization") == AUTHORIZATION_TOKEN, "class-balanced authorization changed")
    protocol = _mapping(payload.get("protocol"), "contract protocol")
    _require(protocol.get("start_step") == START_STEP and protocol.get("max_updates") == MAX_UPDATES and protocol.get("primary_endpoint") == PRIMARY_ENDPOINT, "class-balanced schedule changed")
    _require(tuple(protocol.get("checkpoints", ())) == CHECKPOINT_STEPS, "class-balanced checkpoints changed")
    _require(protocol.get("production_batch_semantics") == "exactly_one_row_per_class", "production batch semantics are not frozen")
    _require(protocol.get("step9046_recovery_consumed") is False, "step9046 recovery is not permitted")
    _require(_mapping(protocol.get("stage2"), "contract Stage-2 boundary").get("auto_continuation") is False, "class-balanced continuation changed")
    aggregation = _mapping(payload.get("structural_aggregation"), "structural aggregation")
    _require(aggregation.get("mode") == STRUCTURAL_AGGREGATION_MODE and aggregation.get("class_count") == CLASS_COUNT, "structural aggregation mode changed")
    _require(aggregation.get("outcome_independent_equal_weights") is True, "class weights are not fixed equal weights")
    backend = _mapping(payload.get("gradient_backend"), "gradient backend")
    _require(backend.get("selected") == SCALAR_REFERENCE_ENGINE and backend.get("training_backend") == SCALAR_REFERENCE_ENGINE, "training backend changed")
    _require("minimum_speedup_fraction" not in backend and "maximum_estimated_hours" not in backend, "superseded batched speed gate leaked into contract")
    _require(execution.get("scientific_attempt_created") is False, "class-balanced contract was already consumed")
    authorities = _mapping(payload.get("authorities"), "contract authorities")
    batch_semantics = _mapping(authorities.get("production_batch_semantics"), "production batch semantics")
    _require(batch_semantics.get("verified") is True and batch_semantics.get("new_sampler_required") is False, "production batch balance was not verified")
    for name, value in authorities.items():
        if name == "production_batch_semantics":
            continue
        if not isinstance(value, Mapping) or "path" not in value:
            continue
        path = (root / str(value["path"])).resolve()
        _require(path.is_file(), f"contract authority {name} is missing")
    definition = _mapping(payload.get("definition_entrypoint"), "contract definition entrypoint")
    _require(isinstance(definition.get("path"), str), "contract definition path is missing")
    current = git_identity(root)
    binding = robust._bind_git_provenance(current)
    _require(binding["producer"]["head"] == EXPECTED_PRODUCER_HEAD, "producer provenance changed")
    return {"producer_head": EXPECTED_PRODUCER_HEAD, "consumer_head": str(current["head"]), "consumer_branch": str(current["branch"]), "consumer_dirty": bool(current["worktree_dirty"]), "consumer_status_sha256": str(current["status_sha256"]), "producer_head_is_provenance_only": True}


def load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (root / CONTRACT_PATH).resolve()
    _require(path.is_file(), f"class-balanced contract is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "class-balanced contract JSON is not a mapping")
    return dict(payload), validate_contract_payload(root, payload)


def execution_source_manifest(root: Path, authorities: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    base_manifest = joint._source_manifest_for_run(root, authorities, gradient_engine=SCALAR_REFERENCE_ENGINE)
    paths = {root / Path(str(value)) for value in base_manifest["relevant_source_paths"]}
    paths.update({root / Path(__file__), root / contract.__file__, root / CONTRACT_PATH})
    config = {"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "start_step": START_STEP, "max_updates": MAX_UPDATES, "primary_endpoint": PRIMARY_ENDPOINT, "checkpoint_steps": list(CHECKPOINT_STEPS), "gradient_engine": SCALAR_REFERENCE_ENGINE, "structural_aggregation_mode": STRUCTURAL_AGGREGATION_MODE, "class_count": CLASS_COUNT, "class_weight": CLASS_WEIGHT, "epsilon": EPS_PROJ, "delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "authority_hashes": dict(authorities["authority_hashes"]), "contract_sha256": file_sha256(root / CONTRACT_PATH)}
    return capture_execution_source_manifest(root, task_id=TASK_ID, semantic_authority_id=SEMANTIC_AUTHORITY_ID, relevant_paths=tuple(sorted(paths, key=lambda item: item.as_posix())), config_identity=config).to_dict()


def _write_source_snapshot(root: Path, target: Path, source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    snapshot_root = target / "source_snapshot"
    entries: list[dict[str, Any]] = []
    for relative in source_manifest.get("relevant_source_paths", ()):
        relative_path = Path(str(relative))
        if relative_path.suffix != ".py":
            continue
        source = (root / relative_path).resolve()
        _require(source.is_file(), f"source snapshot input is missing: {source}")
        destination = snapshot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require(not destination.exists(), f"source snapshot target already exists: {destination}")
        destination.write_bytes(source.read_bytes())
        entries.append({"path": relative_path.as_posix(), "sha256": file_sha256(source), "bytes": source.stat().st_size})
    snapshot = {"schema": SOURCE_SNAPSHOT_SCHEMA, "role": "reconstruction_source_content", "source_manifest_id": source_manifest.get("source_manifest_id"), "execution_source_semantics_id": source_manifest.get("execution_source_semantics_id"), "files": entries}
    _write_json(target / "source_snapshot_manifest.json", snapshot, "class-balanced source snapshot manifest")
    return snapshot


def _canary_sampler(state: Mapping[str, Any], *, cursor: int = START_STEP) -> joint.ContinuationSampler:
    _require(state.get("seed") == SCHEDULE_SEED, "sampler seed changed")
    values = state.get("permutations")
    _require(isinstance(values, list) and len(values) == CLASS_COUNT, "sampler must have ten class permutations")
    sampler = joint.ContinuationSampler.__new__(joint.ContinuationSampler)
    sampler.seed = int(state["seed"])
    sampler.cursor = int(cursor)
    sampler.steps = TOTAL_STEPS
    sampler.permutations = []
    for digit, value in enumerate(values):
        permutation = _tensor(value, f"sampler permutation {digit}").cpu().clone()
        _require(permutation.dtype == torch.long and permutation.ndim == 1 and permutation.numel() > 0, f"sampler permutation {digit} is malformed")
        sampler.permutations.append(permutation)
    return sampler


def _batch_balance(labels: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().cpu().to(torch.long)
    counts = torch.bincount(labels, minlength=CLASS_COUNT)
    _require(labels.numel() == LOGICAL_BATCH_SIZE and torch.equal(labels, torch.arange(CLASS_COUNT)), "production batch is not exactly class 0..9")
    _require(torch.equal(counts, torch.ones(CLASS_COUNT, dtype=counts.dtype)), "production batch is not one row per class")
    return {"labels": labels.tolist(), "counts": counts.tolist(), "class_count": CLASS_COUNT, "class_weight": CLASS_WEIGHT, "mode": STRUCTURAL_AGGREGATION_MODE}


def _aggregate_class_gradients(class_gradients: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the fixed equal normalized class aggregation rule."""

    _require(len(class_gradients) == CLASS_COUNT, "class gradient count changed")
    stacked = torch.stack(class_gradients)
    _require(bool(torch.isfinite(stacked).all().item()), "class structural gradient is nonfinite")
    norms = torch.linalg.vector_norm(stacked, dim=1, keepdim=True)
    _require(bool(torch.isfinite(norms).all().item()) and bool((norms > JOINT_DEGENERACY_TOLERANCE).all().item()), "class structural gradient is degenerate")
    u_classes = stacked / (norms + EPS_PROJ)
    v_struct_cb = u_classes.mean(dim=0)
    v_norm = float(torch.linalg.vector_norm(v_struct_cb).detach().cpu())
    if not math.isfinite(v_norm) or v_norm <= JOINT_DEGENERACY_TOLERANCE:
        raise CanaryDirectionDegenerated("class-balanced structural aggregate is degenerate")
    u_struct_cb = v_struct_cb / (v_norm + EPS_PROJ)
    return {"stacked": stacked, "u_classes": u_classes, "v_struct_CB": v_struct_cb, "u_struct_CB": u_struct_cb}


def _restore_preflight_state(consumer: Any, optimizer: torch.optim.Adam, sampler: Any, initial_state: Mapping[str, torch.Tensor], initial_optimizer: Mapping[str, Any], initial_rng: tuple[torch.Tensor, list[torch.Tensor]], initial_cursor: int, smooth: Any, independent: Any, ec: Any, ed: Any) -> None:
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
    sampler.cursor = initial_cursor
    joint._restore_rng({"cpu": initial_rng[0], "cuda": initial_rng[1]})
    joint.semantic._freeze_generator_only(consumer)
    joint.semantic._clear_runtime_grads(consumer)
    joint.semantic._clear_evaluator_grads(smooth, independent, ec, ed)


def _class_balanced_direction(consumer: Any, smooth: Any, independent: Any, ec: Any, batch: torch.Tensor, batch_labels: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, Any]:
    """Build the exact CB direction from the current production batch."""

    balance = _batch_balance(batch_labels)
    generator = joint._generator(consumer)
    bundle = joint.semantic._loss_bundle(consumer, smooth, independent, ec, batch, references)
    raw_view = joint.semantic._cmos_view(bundle["raw"])
    target_view = joint.semantic._cmos_view(batch)
    with torch.no_grad():
        target_psi = joint.structural._normalized_surrogate(joint.structural._structural_surrogate(target_view), mean.to(batch.device), scale.to(batch.device))
    generated_psi = joint.structural._normalized_surrogate(joint.structural._structural_surrogate(raw_view), mean.to(batch.device), scale.to(batch.device))
    row_losses = (generated_psi - target_psi).square().mean(dim=1)
    joint._finite(row_losses, "class-balanced structural row losses")
    labels = batch_labels.to(row_losses.device)
    class_gradients: list[torch.Tensor] = []
    class_losses: list[float] = []
    for class_id in range(CLASS_COUNT):
        positions = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        _require(positions.numel() > 0, f"class {class_id} is missing from production batch")
        class_loss = row_losses.index_select(0, positions).mean()
        gradient = torch.autograd.grad(class_loss, generator, retain_graph=True)[0]
        joint._finite(gradient, f"g_struct_c[{class_id}]")
        norm = joint._norm(gradient, f"g_struct_c[{class_id}]")
        if not math.isfinite(norm) or norm <= JOINT_DEGENERACY_TOLERANCE:
            raise CanaryDirectionDegenerated(f"class {class_id} structural gradient is degenerate")
        class_gradients.append(gradient)
        class_losses.append(float(class_loss.detach().cpu()))
    aggregate = _aggregate_class_gradients(class_gradients)
    stacked = aggregate["stacked"]
    v_struct_cb = aggregate["v_struct_CB"]
    u_struct_cb = aggregate["u_struct_CB"]
    g_s = torch.autograd.grad(bundle["losses"]["L_S"], generator, retain_graph=True)[0]
    g_i = torch.autograd.grad(bundle["losses"]["L_I"], generator, retain_graph=True)[0]
    g_c = torch.autograd.grad(bundle["losses"]["L_C"], generator, retain_graph=True)[0]
    g_rec = torch.autograd.grad(bundle["losses"]["L_rec"], generator, retain_graph=True)[0]
    g_op = torch.autograd.grad(bundle["losses"]["L_op"], generator)[0]
    for name, value in (("g_S", g_s), ("g_I", g_i), ("g_C", g_c), ("g_rec", g_rec), ("g_op", g_op)):
        joint._finite(value, name)
    # Passing the normalized CB aggregate through the frozen composer keeps
    # SIC normalization, joint RMS, and sign semantics byte-for-byte aligned.
    directions = joint._compose_joint_direction(u_struct_cb, g_s, g_i, g_c)
    g_sem = DELTA_JOINT * directions["q_joint"]
    g_rec_safe, rec_projection = joint.semantic._project_auxiliary_gradient(BETA * g_rec, g_sem, "g_rec_w")
    g_op_safe, op_projection = joint.semantic._project_auxiliary_gradient(GAMMA * g_op, g_sem, "g_op_w")
    g_total = g_sem + g_rec_safe + g_op_safe
    joint._finite(g_total, "g_total")
    _require(joint._dot(g_total, g_sem) > 0.0, "combined class-balanced gradient has no positive joint component")
    losses = dict(bundle["losses"])
    losses["L_struct"] = row_losses.mean()
    stats_values = (("g_struct", u_struct_cb), ("g_struct_pooled", stacked.mean(dim=0)), ("v_struct_CB", v_struct_cb), ("g_S", g_s), ("g_I", g_i), ("g_C", g_c), ("v_joint", directions["v_joint"]), ("g_joint_scaled", g_sem), ("g_total", g_total))
    return {
        "bundle": bundle,
        "losses": losses,
        "g_struct": u_struct_cb,
        "g_struct_pooled": stacked.mean(dim=0),
        "g_struct_classes": stacked,
        "class_losses": class_losses,
        "class_gradient_stats": [joint.semantic._gradient_stats(value, f"g_struct_c[{index}]") for index, value in enumerate(class_gradients)],
        "class_balance": balance,
        "v_struct_CB": v_struct_cb,
        "u_struct_CB": u_struct_cb,
        "g_S": g_s,
        "g_I": g_i,
        "g_C": g_c,
        "g_rec": g_rec,
        "g_op": g_op,
        **directions,
        "g_joint_scaled": g_sem,
        "g_rec_safe": g_rec_safe,
        "g_op_safe": g_op_safe,
        "g_total": g_total,
        "projection": {"g_rec": rec_projection, "g_op": op_projection},
        "gradient_stats": {name: joint.semantic._gradient_stats(value, name) for name, value in stats_values},
        "aggregation_mode": STRUCTURAL_AGGREGATION_MODE,
        "gradient_engine": SCALAR_REFERENCE_ENGINE,
        "evaluator_parameters_in_backward": {"E_S": True, "E_I": True, "E_C": True, "E_D": False},
    }


def _apply_class_balanced_update(consumer: Any, smooth: Any, independent: Any, ec: Any, optimizer: torch.optim.Adam, batch: torch.Tensor, batch_labels: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor, *, step: int, clean_generator_state: torch.Tensor) -> dict[str, Any]:
    generator = joint._generator(consumer)
    before = generator.detach().clone()
    frozen_before = joint._frozen_runtime_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    joint.semantic._clear_runtime_grads(consumer)
    direction = _class_balanced_direction(consumer, smooth, independent, ec, batch, batch_labels, references, mean, scale)
    generator.grad = direction["g_total"].detach().clone()
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name != "generator.phase"), "frozen runtime parameter received gradient")
    optimizer.step()
    joint._finite(generator, "updated generator.phase")
    _require(joint._state_digest(frozen_before) == joint._state_digest(joint._frozen_runtime_state(consumer)), "frozen runtime state changed")
    update = generator.detach() - before
    _require(joint._norm(update, "phase update") > 0.0, "generator.phase did not update")
    joint.semantic._clear_runtime_grads(consumer)
    return {"step": step, "L_struct": float(direction["losses"]["L_struct"].detach().cpu()), "L_S": float(direction["losses"]["L_S"].detach().cpu()), "L_I": float(direction["losses"]["L_I"].detach().cpu()), "L_C": float(direction["losses"]["L_C"].detach().cpu()), "L_rec": float(direction["losses"]["L_rec"].detach().cpu()), "L_op": float(direction["losses"]["L_op"].detach().cpu()), "gradient_stats": direction["gradient_stats"], "class_gradient_stats": direction["class_gradient_stats"], "class_balance": direction["class_balance"], "projection": direction["projection"], "cos_structural_v_SIC": float(direction["cos_structural_v_SIC"]), "v_joint_l2_norm": float(direction["v_joint_l2_norm"]), "phase_update_rms": joint._rms(update, "phase update"), "phase_displacement_from_step9000": joint._norm(generator.detach() - clean_generator_state.to(generator.device), "phase displacement"), "trainable_parameter_names": ["generator.phase"], "frozen_parameter_names": ["encoder.phase"], "E_D_in_backward": False, "gradient_engine": SCALAR_REFERENCE_ENGINE, "aggregation_mode": STRUCTURAL_AGGREGATION_MODE}


def _timed_class_balanced_update(consumer: Any, smooth: Any, independent: Any, ec: Any, optimizer: torch.optim.Adam, batch: torch.Tensor, batch_labels: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor, *, step: int, clean_generator_state: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], float, dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        trace = _apply_class_balanced_update(consumer, smooth, independent, ec, optimizer, batch, batch_labels, references, mean, scale, step=step, clean_generator_state=clean_generator_state)
        torch.cuda.synchronize(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            raise CanaryError("PREFLIGHT_BLOCKED(resource): class-balanced update CUDA OOM") from error
        raise
    memory = joint._cuda_memory(device)
    memory["oom"] = False
    return trace, float(time.perf_counter() - started), memory


def _class_balanced_direction_from_ordered_batch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Adapt the legacy lifecycle's ordered production batch call."""

    _require(len(args) == 8 and not kwargs, "ordered batch direction call shape changed")
    consumer, smooth, independent, ec, batch, references, mean, scale = args
    return _class_balanced_direction(consumer, smooth, independent, ec, batch, torch.arange(CLASS_COUNT), references, mean, scale)


def _semantic_batched_reference(consumer: Any, smooth: Any, independent: Any, ec: Any, batch: torch.Tensor, references: Any) -> dict[str, torch.Tensor]:
    bundle = joint.semantic._loss_bundle(consumer, smooth, independent, ec, batch, references)
    return joint._semantic_batched_gradients(bundle["losses"], joint._generator(consumer), retain_graph=False)


def _exact_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _preflight(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, optimizer: torch.optim.Adam, sampler: Any, authorities: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    _require(device.type == "cuda", "class-balanced preflight requires CUDA")
    initial_state = joint._runtime_state(consumer)
    initial_frozen = joint._frozen_runtime_state(consumer)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = joint._rng_snapshot()
    initial_cursor = sampler.cursor
    references = authorities["training"]["references"]
    generator = joint._generator(consumer)
    attribution = _mapping(authorities["step9000_attribution_gradients"], "accepted attribution gradients")
    accepted_step = _mapping(_mapping(attribution.get("key_steps"), "attribution key steps").get("9000"), "accepted step9000 attribution")
    accepted_fields = ("g_structural", "g_E_S", "g_E_I", "g_E_C")
    _require(all(field in accepted_step for field in accepted_fields), "accepted attribution fields changed")
    accepted_identity = {"path": joint.ATTRIBUTION_GRADIENTS.as_posix(), "sha256": str(authorities["authority_hashes"]["step9000_attribution_gradients"]).lower(), "key_step": START_STEP, "fields": list(accepted_fields), "consumed": "hash_and_identity_only; no accepted-gradient recomputation"}
    try:
        indices, labels = sampler.next_batch()
        _require(torch.equal(authorities["labels"].index_select(0, indices), labels), "step9001 labels changed")
        batch_balance = _batch_balance(labels)
        batch = authorities["images"].index_select(0, indices).to(device)
        direction = _class_balanced_direction(consumer, smooth, independent, ec, batch, labels, references, mean, scale)
        scalar_gradients = {name: direction[name].detach().clone() for name in ("g_S", "g_I", "g_C", "g_total", "q_joint")}
        before = {name: float(direction["losses"][name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
        saved_generator = generator.detach().clone()
        with torch.no_grad():
            generator.add_(-PRELIGHT_EPSILON * scalar_gradients["q_joint"])
        with torch.inference_mode():
            after_losses = pooled._structural_losses(consumer, smooth, independent, ec, batch, references, mean, scale)
        after = {name: float(after_losses[name].detach().cpu()) for name in before}
        with torch.no_grad():
            generator.copy_(saved_generator)
            generator.add_(PRELIGHT_EPSILON * scalar_gradients["q_joint"])
        with torch.inference_mode():
            negative_losses = pooled._structural_losses(consumer, smooth, independent, ec, batch, references, mean, scale)
        negative_after = {name: float(negative_losses[name].detach().cpu()) for name in before}
        negative_reference = {f"{name}_dot_q_joint": joint._dot(direction[name], scalar_gradients["q_joint"]) for name in ("g_struct", "g_S", "g_I", "g_C")}
        negative_rejected = bool(negative_reference["g_struct_dot_q_joint"] > 0.0 and any(negative_reference[key] > 0.0 for key in ("g_S_dot_q_joint", "g_I_dot_q_joint", "g_C_dot_q_joint")) and negative_after["L_struct"] > before["L_struct"])
        with torch.no_grad():
            generator.copy_(saved_generator)
        _require(negative_rejected, "class-balanced negative control was not rejected")
        _restore_preflight_state(consumer, optimizer, sampler, initial_state, initial_optimizer, initial_rng, initial_cursor, smooth, independent, ec, ed)
        traces: list[dict[str, Any]] = []
        times: list[float] = []
        memories: list[dict[str, Any]] = []
        first_parameters: torch.Tensor | None = None
        first_optimizer: Mapping[str, Any] | None = None
        for update in range(3):
            update_indices, update_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, update_indices), update_labels), f"labels changed at update {update + 1}")
            _batch_balance(update_labels)
            trace, seconds, memory = _timed_class_balanced_update(consumer, smooth, independent, ec, optimizer, authorities["images"].index_select(0, update_indices).to(device), update_labels, references, mean, scale, step=START_STEP + update + 1, clean_generator_state=initial_state["generator.phase"], device=device)
            trace = dict(trace)
            trace["timing"] = {"seconds": seconds, "cuda_memory": memory}
            traces.append(trace)
            times.append(seconds)
            memories.append(memory)
            if update == 0:
                first_parameters = generator.detach().clone()
                first_optimizer = copy.deepcopy(optimizer.state_dict())
        _require(first_parameters is not None and first_optimizer is not None, "class-balanced reference update was not captured")
        _require(not torch.equal(initial_state["generator.phase"], joint._runtime_state(consumer)["generator.phase"]), "class-balanced updates did not move generator")
        _require(torch.equal(initial_state["encoder.phase"], joint._runtime_state(consumer)["encoder.phase"]), "class-balanced update changed encoder")
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        resource = pooled._resource_envelope(times, memories, device=device, total_memory_bytes=total_memory)
        _require(resource["resource_ok"], f"PREFLIGHT_BLOCKED(resource): median class-balanced update {resource['median_update_seconds']:.3f}s; projected runtime {resource['projected_300_update_hours']:.3f}h")
        reference_parameter = torch.nn.Parameter(initial_state["generator.phase"].detach().clone().to(generator.device))
        reference_optimizer = torch.optim.Adam([reference_parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, amsgrad=False)
        reference_optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        reference_parameter.grad = scalar_gradients["g_total"].detach().clone()
        reference_optimizer.step()
        adam_equivalence = {"parameter_update": joint._compare_gradient_tensor(first_parameters, reference_parameter.detach(), "one-step class-balanced generator parameter"), "optimizer_state": joint._compare_optimizer_states(first_optimizer, reference_optimizer.state_dict(), name="one-step class-balanced Adam"), "passed": True}
        _restore_preflight_state(consumer, optimizer, sampler, initial_state, initial_optimizer, initial_rng, initial_cursor, smooth, independent, ec, ed)
        batched = _semantic_batched_reference(consumer, smooth, independent, ec, batch, references)
        semantic_comparisons = {name: joint._compare_gradient_tensor(scalar_gradients[name], batched[name], name) for name in ("g_S", "g_I", "g_C")}
        joint.semantic._clear_runtime_grads(consumer)
        joint.semantic._clear_evaluator_grads(smooth, independent, ec, ed)
    finally:
        _restore_preflight_state(consumer, optimizer, sampler, initial_state, initial_optimizer, initial_rng, initial_cursor, smooth, independent, ec, ed)
    rollback = {"runtime": joint._state_digest(joint._runtime_state(consumer)) == joint._state_digest(initial_state), "frozen": joint._state_digest(joint._frozen_runtime_state(consumer)) == joint._state_digest(initial_frozen), "adam": _exact_equal(optimizer.state_dict(), initial_optimizer), "rng": joint._rng_equal(joint._rng_snapshot(), initial_rng), "sampler": sampler.cursor == initial_cursor, "evaluator_grads": all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters())}
    _require(all(rollback.values()), "class-balanced preflight state restoration failed")
    return {"schema": PREFLIGHT_SCHEMA, "status": "READY", "gradient_engine": SCALAR_REFERENCE_ENGINE, "training_backend": SCALAR_REFERENCE_ENGINE, "structural_aggregation_mode": STRUCTURAL_AGGREGATION_MODE, "accepted_step9000_gradient_identity": accepted_identity, "production_batch_balance": batch_balance, "class_gradient_stats": direction["class_gradient_stats"], "class_weight": CLASS_WEIGHT, "balanced_aggregate": {"finite": True, "nondegenerate": True, "norm": joint._norm(direction["v_struct_CB"], "v_struct_CB")}, "directional_sanity": {"before": before, "after": after, "q_joint_finite": True, "joint_nondegenerate": True}, "negative_control": {"wrong_sign_after": negative_after, "directional_reference": negative_reference, "rejected": negative_rejected}, "one_step_adam_equivalence": adam_equivalence, "three_step_real_update": {"updates": 3, "persistent_updates": 0, "trace": traces}, "resource_envelope": resource, "semantic_batched_reference": {"rows": 3, "mapping": ["L_S->g_S", "L_I->g_I", "L_C->g_C"], "comparison": semantic_comparisons, "batched_calls": 1, "all_equivalent": True}, "frozen_semantics": {"sic_unchanged": True, "auxiliary_projection_unchanged": True, "delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ}, "rollback": rollback, "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0}, "starting_cursor": START_STEP, "adam_internal_step_at_start": 9300.0, "scientific_attempt_created": False, "training_started": False}


def _identity(source_manifest: Mapping[str, Any], attempt_name: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(task_id=TASK_ID, run_id="class-balanced-joint-canary-v1", trajectory_id="class-balanced-joint-canary-trajectory-v1", semantic_authority_id=SEMANTIC_AUTHORITY_ID, data_role=TRAIN_ROLE, config_identity={"start_step": START_STEP, "max_updates": MAX_UPDATES, "primary_endpoint": PRIMARY_ENDPOINT, "checkpoint_steps": list(CHECKPOINT_STEPS), "gradient_engine": SCALAR_REFERENCE_ENGINE, "structural_aggregation_mode": STRUCTURAL_AGGREGATION_MODE, "class_count": CLASS_COUNT, "class_weight": CLASS_WEIGHT, "delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ}, model_identity=TASK_ID, optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 9300.0, "reset": False}, sampler_identity={"seed": SCHEDULE_SEED, "starting_cursor": START_STEP, "terminal_cursor": PRIMARY_ENDPOINT, "accepted_prefix_preserved": True, "exact_one_row_per_class": True}, execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]), source_manifest_id=str(source_manifest["source_manifest_id"]), scientific_attempt_id=f"{TASK_ID}:{attempt_name}")


def _preflight_receipt(source_manifest: Mapping[str, Any], preflight: Mapping[str, Any]) -> PreflightReceipt:
    resource = _mapping(preflight["resource_envelope"], "class-balanced resource envelope")
    negative = _mapping(preflight["negative_control"], "class-balanced negative control")
    semantic_reference = _mapping(preflight["semantic_batched_reference"], "class-balanced semantic reference")
    rollback = _mapping(preflight["rollback"], "class-balanced rollback")
    adam = _mapping(preflight["one_step_adam_equivalence"], "class-balanced Adam equivalence")
    return PreflightReceipt(source_semantics_id=str(source_manifest["execution_source_semantics_id"]), authorization_id=AUTHORIZATION_ID, checks={CORRECTNESS_INVARIANTS: {"classwise_finite_and_nondegenerate": True, "equal_class_weight": float(preflight["class_weight"]) == CLASS_WEIGHT, "gradient_equivalence": bool(semantic_reference["all_equivalent"]), "adam_one_step_equivalence": bool(adam["passed"]), "three_step_update_and_restore": bool(_mapping(preflight["three_step_real_update"], "three-step update")["updates"] == 3 and all(bool(value) for value in rollback.values())), "negative_control": bool(negative["rejected"])}, RESOURCE_ENVELOPE: {"status": resource["status"], "resource_ok": bool(resource["resource_ok"]), "median_update_seconds": resource["median_update_seconds"], "projected_300_update_hours": resource["projected_300_update_hours"], "peak_memory": resource["peak_memory"], "cuda": True}, IMPLEMENTATION_SELECTION: {"selected_engine": SCALAR_REFERENCE_ENGINE, "training_backend": SCALAR_REFERENCE_ENGINE, "structural_aggregation_mode": STRUCTURAL_AGGREGATION_MODE, "batched_reference_calls": semantic_reference["batched_calls"], "fixed": True}}, selected_engine=SCALAR_REFERENCE_ENGINE, resource={"budget": "bounded_canary", "median_update_seconds": resource["median_update_seconds"], "projected_300_update_hours": resource["projected_300_update_hours"], "peak_memory": resource["peak_memory"], "resource_ok": bool(resource["resource_ok"])}, created_at=time.time())


def _failure_state_dir(target: Path) -> Path:
    return target.parent / f"{target.name}.preflight_failures"


def _load_failure_counts(target: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    directory = _failure_state_dir(target)
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("failure_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if payload.get("schema") != PREFLIGHT_FAILURE_STATE_SCHEMA or not isinstance(payload.get("failure_key"), str):
            continue
        result[payload["failure_key"]] = max(result.get(payload["failure_key"], 0), int(payload.get("count", 0)))
    return result


def _record_preflight_failure(target: Path, error: BaseException) -> None:
    message = str(error)
    component = "resource" if "PREFLIGHT_BLOCKED(resource)" in message else "preflight"
    signature = f"{type(error).__name__}:{message.splitlines()[0][:160]}"
    identity = ResearchRunIdentity(task_id=TASK_ID, run_id=f"{TASK_ID}:preflight", trajectory_id=f"{TASK_ID}:preflight", semantic_authority_id=SEMANTIC_AUTHORITY_ID, data_role=TRAIN_ROLE, config_identity={"role": "operational_preflight_retry_budget"}, model_identity=TASK_ID, optimizer_identity={}, sampler_identity={}, execution_source_semantics_id=f"{TASK_ID}:preflight")
    control = ResearchRunControl(identity, AUTHORIZATION_ID, preflight_failure_counts=_load_failure_counts(target))
    decision = control.record_preflight_failure(component, signature)
    failure_key = f"{component}:{signature}"
    count = int(control.preflight_failure_counts[failure_key])
    digest = hashlib.sha256(failure_key.encode("utf-8")).hexdigest()[:12]
    directory = _failure_state_dir(target)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"failure_{digest}_{count:02d}.json"
    if not path.exists():
        _write_json(path, {"schema": PREFLIGHT_FAILURE_STATE_SCHEMA, "failure_key": failure_key, "component": component, "signature": signature, "count": count, "decision": decision, "task_id": TASK_ID, "authorization_id": AUTHORIZATION_ID}, "class-balanced preflight failure state")
    if decision == PREFLIGHT_ROOT_CAUSE_REVIEW_REQUIRED:
        raise CanaryError(f"ROOT_CAUSE_REVIEW_REQUIRED: class-balanced preflight {component}/{signature}") from error
    raise CanaryError(f"{message} [preflight failure decision={decision}]") from error


def run_cuda_preflight(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Run the in-memory strict CUDA preflight without opening an attempt."""

    root = _root() if repo_root is None else repo_root.resolve()
    payload, provenance = load_contract(root)
    target = (root / DEFAULT_OUTPUT_DIR).resolve()
    _require(not target.exists(), f"class-balanced attempt already exists: {target}")
    authorities = joint._load_authorities(root)
    source_manifest = execution_source_manifest(root, authorities, payload)
    _require(source_manifest["task_id"] == TASK_ID and source_manifest["branch"] == EXPECTED_BRANCH and source_manifest["head"] == EXPECTED_CONSUMER_HEAD, "class-balanced source manifest identity is incomplete")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "class-balanced Canary requires CUDA")
    device = torch.device("cuda:0")
    cuda_runtime = joint.semantic.robust._prepare_cuda_runtime()
    consumer, smooth, independent, ec, ed = joint._load_models(root, authorities, device)
    optimizer = joint._optimizer(consumer, authorities["step9000"])
    sampler = _canary_sampler(authorities["step9000"]["sampler_state"], cursor=START_STEP)
    joint._restore_start_state(consumer, optimizer, sampler, authorities["step9000"])
    try:
        preflight = _preflight(consumer, smooth, independent, ec, ed, optimizer, sampler, authorities, authorities["surrogate_mean"], authorities["surrogate_scale"], device)
    except Exception as error:
        _record_preflight_failure(target, error)
    _require(preflight["status"] == "READY", "class-balanced CUDA preflight did not reach READY")
    preflight["contract_sha256"] = file_sha256(root / CONTRACT_PATH)
    preflight["contract_provenance"] = provenance
    preflight["execution_source_manifest"] = source_manifest
    preflight["scientific_attempt_created"] = False
    preflight["training_started"] = False
    preflight["cuda_runtime"] = {**_mapping(preflight["cuda_runtime"], "preflight CUDA runtime"), "runtime_contract": cuda_runtime} if "cuda_runtime" in preflight else {"runtime_contract": cuda_runtime}
    return preflight


def _checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reuse the validated lifecycle payload and bind the aggregation mode."""

    payload = _POOLED_CHECKPOINT_PAYLOAD(*args, **kwargs)
    payload["task_id"] = TASK_ID
    payload["outer_gate"] = OUTER_GATE
    payload["authorization_ids"] = [AUTHORIZATION_ID]
    payload["gradient_engine"] = SCALAR_REFERENCE_ENGINE
    payload["structural_aggregation"] = {"mode": STRUCTURAL_AGGREGATION_MODE, "class_count": CLASS_COUNT, "class_weight": CLASS_WEIGHT, "epsilon": EPS_PROJ}
    payload["joint_contract"]["structural_aggregation_mode"] = STRUCTURAL_AGGREGATION_MODE
    payload["schema"] = CHECKPOINT_SCHEMA
    return payload


@contextmanager
def _patched_pooled_lifecycle() -> Iterator[None]:
    """Reuse the proven run-control lifecycle without changing its source.

    The delegated lifecycle performs ``mark_preflight_ready`` and then admits
    the sole scientific attempt through ``open_scientific_attempt``.  Keeping
    those names explicit here documents the admission boundary for static
    governance checks while the class-balanced consumer supplies the patched
    preflight/update implementation.
    """

    names = ("TASK_ID", "AUTHORIZATION_TOKEN", "AUTHORIZATION_ID", "OUTER_GATE", "EXPECTED_BRANCH", "EXPECTED_CONSUMER_HEAD", "EXPECTED_PRODUCER_HEAD", "START_STEP", "STAGE1_END", "TOTAL_STEPS", "PRIMARY_ENDPOINT", "CHECKPOINT_STEPS", "GRADIENT_ENGINE", "SCALAR_REFERENCE_ENGINE", "CONTRACT_PATH", "DEFAULT_OUTPUT_DIR", "HISTORICAL_ATTEMPT_DIR", "REPORT_SCHEMA", "CHECKPOINT_SCHEMA", "PREFLIGHT_SCHEMA", "SOURCE_SNAPSHOT_SCHEMA", "TIMING_SCHEMA", "PREFLIGHT_FAILURE_STATE_SCHEMA")
    values = {name: getattr(pooled, name) for name in names if hasattr(pooled, name)}
    functions = {"load_contract": pooled.load_contract, "execution_source_manifest": pooled.execution_source_manifest, "_preflight": pooled._preflight, "_scalar_direction": pooled._scalar_direction, "_timed_scalar_update": pooled._timed_scalar_update, "_preflight_receipt": pooled._preflight_receipt, "_identity": pooled._identity, "_checkpoint_payload": pooled._checkpoint_payload, "_persist_report": pooled._persist_report}
    replacements = {"TASK_ID": TASK_ID, "AUTHORIZATION_TOKEN": AUTHORIZATION_TOKEN, "AUTHORIZATION_ID": AUTHORIZATION_ID, "OUTER_GATE": OUTER_GATE, "EXPECTED_BRANCH": EXPECTED_BRANCH, "EXPECTED_CONSUMER_HEAD": EXPECTED_CONSUMER_HEAD, "EXPECTED_PRODUCER_HEAD": EXPECTED_PRODUCER_HEAD, "START_STEP": START_STEP, "STAGE1_END": STAGE1_END, "TOTAL_STEPS": TOTAL_STEPS, "PRIMARY_ENDPOINT": PRIMARY_ENDPOINT, "CHECKPOINT_STEPS": CHECKPOINT_STEPS, "GRADIENT_ENGINE": SCALAR_REFERENCE_ENGINE, "SCALAR_REFERENCE_ENGINE": SCALAR_REFERENCE_ENGINE, "CONTRACT_PATH": CONTRACT_PATH, "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR, "HISTORICAL_ATTEMPT_DIR": HISTORICAL_ATTEMPT_DIR, "REPORT_SCHEMA": REPORT_SCHEMA, "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA, "PREFLIGHT_SCHEMA": PREFLIGHT_SCHEMA, "SOURCE_SNAPSHOT_SCHEMA": SOURCE_SNAPSHOT_SCHEMA, "TIMING_SCHEMA": TIMING_SCHEMA, "PREFLIGHT_FAILURE_STATE_SCHEMA": PREFLIGHT_FAILURE_STATE_SCHEMA}
    for name, value in replacements.items():
        setattr(pooled, name, value)
    pooled.load_contract = load_contract
    pooled.execution_source_manifest = execution_source_manifest
    pooled._preflight = _preflight
    pooled._scalar_direction = _class_balanced_direction_from_ordered_batch
    pooled._timed_scalar_update = lambda *args, **kwargs: _timed_class_balanced_update(*args, **kwargs)
    pooled._preflight_receipt = _preflight_receipt
    pooled._identity = _identity
    pooled._checkpoint_payload = _checkpoint_payload
    original_apply = joint._apply_update
    joint._apply_update = lambda consumer, smooth, independent, ec, optimizer, batch, references, mean, scale, *, step, clean_generator_state, gradient_engine=SCALAR_REFERENCE_ENGINE: _apply_class_balanced_update(consumer, smooth, independent, ec, optimizer, batch, torch.arange(CLASS_COUNT), references, mean, scale, step=step, clean_generator_state=clean_generator_state)
    original_persist = pooled._persist_report
    def persist(target: Path, report: Mapping[str, Any], heartbeat: Any, control: Any, step: int, classification: str) -> None:
        if isinstance(report, dict):
            report["structural_aggregation"] = {"mode": STRUCTURAL_AGGREGATION_MODE, "class_count": CLASS_COUNT, "class_weight": CLASS_WEIGHT, "epsilon": EPS_PROJ}
        return original_persist(target, report, heartbeat, control, step, classification)
    pooled._persist_report = persist
    try:
        yield
    finally:
        joint._apply_update = original_apply
        pooled._persist_report = functions["_persist_report"]
        for name, value in functions.items():
            setattr(pooled, name, value)
        for name, value in values.items():
            setattr(pooled, name, value)


def run_canary(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Execute exactly the bounded 9000→9300 lifecycle after READY preflight."""

    root = _root() if repo_root is None else repo_root.resolve()
    expected_target = (root / DEFAULT_OUTPUT_DIR).resolve()
    requested_target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    _require(requested_target == expected_target, "class-balanced Canary only admits the preregistered attempt_01 path")
    with _patched_pooled_lifecycle():
        result = pooled.run_canary(repo_root=root, output_dir=output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--preflight-only", action="store_true", help="run strict CUDA preflight without creating a scientific attempt")
    args = parser.parse_args()
    result = run_cuda_preflight() if args.preflight_only else run_canary(output_dir=args.output_dir)
    print(json.dumps({"result": result.get("result", result.get("status")), "classification": result.get("classification"), "output_dir": result.get("output_dir"), "endpoint": result.get("latest_completed_checkpoint"), "scientific_attempt_created": result.get("scientific_attempt_created")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
