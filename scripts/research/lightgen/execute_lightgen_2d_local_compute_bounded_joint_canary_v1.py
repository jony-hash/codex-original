"""Execute the authorized 300-update local-compute joint Canary.

The historical joint continuation remains a separate 12000-step batched-VJP
consumer.  This entry point binds the already-defined Canary contract to a
scalar-reference training path, performs a bounded CUDA preflight, and stops
unconditionally at step 9300.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, TRAIN_ROLE, _rng_state_payload, git_identity
from ldm_project.v2.application.research.shared.research_run_control import CORRECTNESS_INVARIANTS, HeartbeatWriter, IMPLEMENTATION_SELECTION, InterruptController, PreflightReceipt, RecoveryCheckpointManager, RESOURCE_ENVELOPE, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUN_INTERRUPTED, RUN_SCIENTIFIC_ACTIVE, RUN_VALID_FAILED, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, file_sha256

from scripts.research.lightgen import define_lightgen_2d_local_compute_bounded_joint_canary_contract_v1 as contract
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as joint


TASK_ID = "EXECUTE_LIGHTGEN_2D_LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_V1"
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
CHECKPOINT_STEPS = contract.CHECKPOINT_STEPS
LOGICAL_BATCH_SIZE = contract.LOGICAL_BATCH_SIZE
ATTRIBUTION_MICRO_BATCH_SIZE = joint.ATTRIBUTION_MICRO_BATCH_SIZE
SCHEDULE_SEED = joint.SCHEDULE_SEED
FEATURE_EPS = joint.FEATURE_EPS
JOINT_DEGENERACY_TOLERANCE = joint.JOINT_DEGENERACY_TOLERANCE
DELTA_JOINT = contract.DELTA_JOINT
BETA = contract.BETA
GAMMA = contract.GAMMA
EPS_PROJ = contract.EPS_PROJ
RECONSTRUCTION_TOLERANCE = joint.RECONSTRUCTION_TOLERANCE
PRELIGHT_EPSILON = joint.PRELIGHT_EPSILON
GRADIENT_ENGINE = contract.GRADIENT_ENGINE
SCALAR_REFERENCE_ENGINE = joint.SCALAR_REFERENCE_ENGINE
GRADIENT_RTOL = contract.GRADIENT_RTOL
GRADIENT_ATOL = contract.GRADIENT_ATOL
GRADIENT_COSINE_MINIMUM = joint.GRADIENT_COSINE_MINIMUM
RECOVERY_INTERVAL_STEPS = joint.RECOVERY_INTERVAL_STEPS

START_CHECKPOINT = joint.START_CHECKPOINT
START_CHECKPOINT_SHA256 = joint.START_CHECKPOINT_SHA256.lower()
CONTRACT_PATH = Path(
    "outputs/20260922_local_compute_bounded_joint_canary_v1/runs/contract_definition_07/"
    "lightgen_2d_local_compute_bounded_joint_canary_contract.v1.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/20260922_local_compute_bounded_joint_canary_v1/runs/attempt_01")
HISTORICAL_ATTEMPT_DIR = joint.DEFAULT_OUTPUT_DIR
REPORT_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_report.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_checkpoint.v1"
PREFLIGHT_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_preflight.v1"
SOURCE_SNAPSHOT_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_source_snapshot.v1"
TIMING_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_timing.v1"
BASELINE_PRIVATE_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_step9000_private_baseline.v1"
ATTEMPT_CLOSEOUT_SCHEMA = "lightgen_2d_local_compute_bounded_joint_canary_attempt_closeout.v1"
INVALID_HARNESS_CLOSEOUT_NAME = "invalid_harness_closeout.json"
PREFLIGHT_FAILURE_STATE_SCHEMA = "local_compute_bounded_joint_canary_preflight_failure_state.v1"


class CanaryError(RuntimeError):
    """A Canary authority, state, or execution invariant is invalid."""


class CanaryDirectionDegenerated(CanaryError):
    """The preregistered joint direction is numerically unusable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _trace(message: str) -> None:
    if os.environ.get("LIGHTGEN_CANARY_PREFLIGHT_TRACE") == "1":
        print(f"[canary] {message}", flush=True)


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
    _require(value == GRADIENT_ENGINE, f"training backend must be {GRADIENT_ENGINE}; got {value}")


def _artifact_refs(root: Path, payload: Mapping[str, Any]) -> None:
    authorities = _mapping(payload.get("authorities"), "contract authorities")
    for name, value in authorities.items():
        if name in {"checkpoint_validation", "attribution_validation"}:
            continue
        ref = _mapping(value, f"contract authority {name}")
        relative = ref.get("path")
        _require(isinstance(relative, str), f"contract authority {name} is incomplete")
        path = (root / relative).resolve()
        _require(path.is_file(), f"contract authority {name} is missing")


def validate_contract_payload(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable contract and bind current consumer provenance."""

    _require(payload.get("schema") == contract.CONTRACT_SCHEMA, "Canary contract schema changed")
    _require(payload.get("result") == "READY" and payload.get("status") == "valid_completed", "Canary contract is not READY")
    _require(payload.get("classification") == contract.CLASSIFICATION, "Canary contract classification changed")
    _require(payload.get("outer_gate") == OUTER_GATE, "Canary outer gate changed")
    execution_boundary = _mapping(payload.get("execution_boundary"), "Canary execution boundary")
    _require(execution_boundary.get("proposed_authorization") == AUTHORIZATION_TOKEN, "Canary authorization token changed")
    protocol = _mapping(payload.get("protocol"), "Canary protocol")
    _require(protocol.get("start_step") == START_STEP and protocol.get("max_updates") == MAX_UPDATES, "Canary update budget changed")
    _require(protocol.get("primary_endpoint") == PRIMARY_ENDPOINT and tuple(protocol.get("checkpoints", ())) == CHECKPOINT_STEPS, "Canary endpoint schedule changed")
    _require(protocol.get("step9046_recovery_consumed") is False, "step9046 recovery is not permitted")
    _require(_mapping(protocol.get("stage2"), "Canary Stage-2 boundary").get("auto_continuation") is False, "Canary auto-continuation changed")
    backend = _mapping(payload.get("gradient_backend"), "Canary gradient backend")
    _require(backend.get("selected") == GRADIENT_ENGINE, "Canary training backend changed")
    _require("minimum_speedup_fraction" not in backend and "maximum_estimated_hours" not in backend and "maximum_memory_fraction" not in backend, "superseded batched speed gate leaked into Canary")
    _require(execution_boundary.get("scientific_attempt_created") is False, "Canary was already consumed")
    _artifact_refs(root, payload)
    definition = _mapping(payload.get("definition_entrypoint"), "contract definition entrypoint")
    _require(isinstance(definition.get("path"), str), "contract definition path is unavailable")

    current = git_identity(root)
    contract_git = _mapping(payload.get("git"), "contract Git provenance")
    return {
        "producer_head": EXPECTED_PRODUCER_HEAD,
        "consumer_head": str(current["head"]),
        "consumer_branch": str(current["branch"]),
        "consumer_dirty": bool(current["worktree_dirty"]),
        "consumer_status_sha256": str(current["status_sha256"]),
        "consumer_head_exact_required": False,
        "required_consumer_head": contract_git.get("required_consumer_head"),
        "producer_head_is_provenance_only": True,
    }


def load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (root / CONTRACT_PATH).resolve()
    _require(path.is_file(), f"Canary contract is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "Canary contract JSON is not a mapping")
    binding = validate_contract_payload(root, payload)
    return dict(payload), binding


def execution_source_manifest(root: Path, authorities: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Capture all executable dependencies plus this consumer and its contract."""

    historical = joint._source_manifest_for_run(root, authorities)
    paths = {root / str(path) for path in historical["relevant_source_paths"]}
    paths.update({root / Path(__file__), root / contract.__file__, root / CONTRACT_PATH})
    config_identity = {
        "authorization_token": AUTHORIZATION_TOKEN,
        "outer_gate": OUTER_GATE,
        "start_step": START_STEP,
        "max_updates": MAX_UPDATES,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "gradient_engine": GRADIENT_ENGINE,
        "scalar_reference_engine": SCALAR_REFERENCE_ENGINE,
        "delta_joint": DELTA_JOINT,
        "beta": BETA,
        "gamma": GAMMA,
        "eps_proj": EPS_PROJ,
        "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "cosine_minimum": GRADIENT_COSINE_MINIMUM},
        "contract_sha256": file_sha256(root / CONTRACT_PATH),
        "authority_hashes": dict(_mapping(authorities.get("authority_hashes"), "authority hashes")),
    }
    return capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=tuple(sorted(paths, key=lambda item: item.as_posix())),
        config_identity=config_identity,
    ).to_dict()


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
    snapshot = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "role": "reconstruction_source_content",
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "execution_source_semantics_id": source_manifest.get("execution_source_semantics_id"),
        "files": entries,
    }
    _write_json(target / "source_snapshot_manifest.json", snapshot, "Canary source snapshot manifest")
    return snapshot


def _canary_sampler(state: Mapping[str, Any], *, cursor: int = START_STEP) -> joint.ContinuationSampler:
    _require(state.get("seed") == SCHEDULE_SEED, "sampler seed changed")
    values = state.get("permutations")
    _require(isinstance(values, list) and len(values) == 10, "sampler permutations are malformed")
    sampler = joint.ContinuationSampler.__new__(joint.ContinuationSampler)
    sampler.seed = int(state["seed"])
    sampler.cursor = int(cursor)
    sampler.steps = TOTAL_STEPS
    sampler.permutations = []
    for digit, value in enumerate(values):
        permutation = _tensor(value, f"sampler permutation {digit}").cpu().clone()
        _require(permutation.dtype == torch.long and permutation.ndim == 1 and permutation.numel() > 0, f"sampler permutation {digit} is empty")
        sampler.permutations.append(permutation)
    _require(0 <= sampler.cursor <= TOTAL_STEPS, "sampler cursor is invalid")
    return sampler


def _exact_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _restore_preflight_state(consumer: Any, optimizer: torch.optim.Adam, sampler: Any, initial_state: Mapping[str, torch.Tensor], initial_optimizer: Mapping[str, Any], initial_rng: tuple[torch.Tensor, list[torch.Tensor]], initial_cursor: int, smooth: Any, independent: Any, ec: Any, ed: Any) -> None:
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
    sampler.cursor = initial_cursor
    joint._restore_rng({"cpu": initial_rng[0], "cuda": initial_rng[1]})
    joint.semantic._freeze_generator_only(consumer)
    joint.semantic._clear_runtime_grads(consumer)
    joint.semantic._clear_evaluator_grads(smooth, independent, ec, ed)


def _scalar_direction(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs["gradient_engine"] = SCALAR_REFERENCE_ENGINE
    value = joint._joint_direction(*args, **kwargs)
    _require_training_backend(str(value.get("gradient_engine")))
    return value


def _semantic_batched_reference(consumer: Any, smooth: Any, independent: Any, ec: Any, batch: torch.Tensor, references: Any) -> dict[str, torch.Tensor]:
    """Reference only the contract's three semantic VJP rows.

    Structural, reconstruction, and optical gradients remain scalar-only in
    the Canary.  Re-running those paths through the historical hybrid engine
    is outside the contract and can trigger pathological vmap operator costs.
    """

    bundle = joint.semantic._loss_bundle(consumer, smooth, independent, ec, batch, references)
    return joint._semantic_batched_gradients(bundle["losses"], joint._generator(consumer), retain_graph=False)


def _structural_losses(consumer: Any, smooth: Any, independent: Any, ec: Any, batch: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    bundle = joint.semantic._loss_bundle(consumer, smooth, independent, ec, batch, references)
    raw_view = joint.semantic._cmos_view(bundle["raw"])
    target_view = joint.semantic._cmos_view(batch)
    generated = joint.structural._normalized_surrogate(joint.structural._structural_surrogate(raw_view), mean.to(batch.device), scale.to(batch.device))
    target = joint.structural._normalized_surrogate(joint.structural._structural_surrogate(target_view), mean.to(batch.device), scale.to(batch.device))
    losses = dict(bundle["losses"])
    losses["L_struct"] = (generated - target).square().mean()
    return losses


def _resource_envelope(update_seconds: list[float], memory_records: list[Mapping[str, Any]], *, device: torch.device, total_memory_bytes: int) -> dict[str, Any]:
    """Classify only the bounded operational envelope for three real updates."""

    finite_times = len(update_seconds) == 3 and all(math.isfinite(float(value)) and float(value) >= 0.0 for value in update_seconds)
    median_seconds = float(sorted(update_seconds)[1]) if finite_times else float("inf")
    projected_hours = median_seconds * float(MAX_UPDATES) / 3600.0 if math.isfinite(median_seconds) else float("inf")
    peak_reserved = max((int(record.get("max_reserved_bytes", 0)) for record in memory_records), default=0)
    peak_allocated = max((int(record.get("max_allocated_bytes", 0)) for record in memory_records), default=0)
    oom = any(bool(record.get("oom", False)) for record in memory_records)
    memory_ok = bool(total_memory_bytes > 0 and peak_reserved < int(total_memory_bytes) and peak_allocated <= int(total_memory_bytes))
    resource_ok = bool(finite_times and not oom and memory_ok)
    return {
        "status": "READY" if resource_ok else "PREFLIGHT_BLOCKED(resource)",
        "resource_ok": resource_ok,
        "device": str(device),
        "update_seconds": [float(value) for value in update_seconds],
        "median_update_seconds": median_seconds,
        "projected_300_update_hours": projected_hours,
        "peak_memory": {"max_reserved_bytes": peak_reserved, "max_allocated_bytes": peak_allocated, "total_memory_bytes": int(total_memory_bytes)},
        "memory_ok": memory_ok,
        "oom": oom,
    }


def _timed_scalar_update(consumer: Any, smooth: Any, independent: Any, ec: Any, optimizer: torch.optim.Adam, batch: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor, *, step: int, clean_generator_state: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], float, dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        trace = joint._apply_update(consumer, smooth, independent, ec, optimizer, batch, references, mean, scale, step=step, clean_generator_state=clean_generator_state, gradient_engine=SCALAR_REFERENCE_ENGINE)
        torch.cuda.synchronize(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            raise CanaryError("PREFLIGHT_BLOCKED(resource): scalar update CUDA OOM") from error
        raise
    seconds = time.perf_counter() - started
    memory = joint._cuda_memory(device)
    memory["oom"] = False
    return trace, float(seconds), memory


def _preflight(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, optimizer: torch.optim.Adam, sampler: Any, authorities: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    _require(device.type == "cuda", "Canary preflight requires CUDA")
    _trace("preflight: captured initial state")
    initial_state = joint._runtime_state(consumer)
    initial_frozen = joint._frozen_runtime_state(consumer)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = joint._rng_snapshot()
    initial_cursor = sampler.cursor
    initial_evaluator_grads_empty = all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters())
    _require(initial_evaluator_grads_empty, "preflight requires clean evaluator gradients")
    references = authorities["training"]["references"]
    generator = joint._generator(consumer)
    attribution = _mapping(authorities["step9000_attribution_gradients"], "accepted step9000 attribution gradients")
    key_steps = _mapping(attribution.get("key_steps"), "accepted attribution key steps")
    accepted_step = _mapping(key_steps.get("9000"), "accepted attribution step9000")
    accepted_fields = ("g_structural", "g_E_S", "g_E_I", "g_E_C")
    _require(all(field in accepted_step for field in accepted_fields), "accepted step9000 attribution fields changed")
    accepted_gradient_identity = {
        "path": joint.ATTRIBUTION_GRADIENTS.as_posix(),
        "sha256": str(authorities.get("authority_hashes", {}).get("step9000_attribution_gradients", joint.ATTRIBUTION_GRADIENTS_SHA256)).lower(),
        "key_step": START_STEP,
        "fields": list(accepted_fields),
        "consumed": "hash_and_identity_only; no accepted-gradient recomputation",
    }

    try:
        indices, labels = sampler.next_batch()
        _require(int(indices.numel()) == LOGICAL_BATCH_SIZE, "step9001 TRAIN batch size changed")
        _require(torch.equal(authorities["labels"].index_select(0, indices), labels), "step9001 TRAIN labels changed")
        batch = authorities["images"].index_select(0, indices).to(device)
        _trace("preflight: one step9001 scalar direction")
        direction = _scalar_direction(consumer, smooth, independent, ec, batch, references, mean, scale)
        scalar_gradients = {name: direction[name].detach().clone() for name in ("g_S", "g_I", "g_C", "g_total", "q_joint")}
        scalar_direction_record = {
            "step": START_STEP + 1,
            "batch_indices": indices.tolist(),
            "batch_labels": labels.tolist(),
            "gradient_engine": direction["gradient_engine"],
            "gradient_stats": direction["gradient_stats"],
            "q_joint_norm": joint._norm(scalar_gradients["q_joint"], "q_joint"),
            "g_total_norm": joint._norm(scalar_gradients["g_total"], "g_total"),
        }
        _require(direction["gradient_engine"] == SCALAR_REFERENCE_ENGINE, "step9001 scalar direction used the wrong engine")

        before = {name: float(direction["losses"][name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
        saved_generator = generator.detach().clone()
        with torch.no_grad():
            generator.add_(-PRELIGHT_EPSILON * scalar_gradients["q_joint"])
        with torch.inference_mode():
            after_losses = _structural_losses(consumer, smooth, independent, ec, batch, references, mean, scale)
        after = {name: float(after_losses[name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
        with torch.no_grad():
            generator.copy_(saved_generator)
            generator.add_(PRELIGHT_EPSILON * scalar_gradients["q_joint"])
        with torch.inference_mode():
            negative_losses = _structural_losses(consumer, smooth, independent, ec, batch, references, mean, scale)
        negative_after = {name: float(negative_losses[name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
        negative_reference = {f"{name}_dot_q_joint": joint._dot(direction[name].detach(), scalar_gradients["q_joint"]) for name in ("g_struct", "g_S", "g_I", "g_C")}
        negative_rejected = bool(negative_reference["g_struct_dot_q_joint"] > 0.0 and any(negative_reference[key] > 0.0 for key in ("g_S_dot_q_joint", "g_I_dot_q_joint", "g_C_dot_q_joint")) and negative_after["L_struct"] > before["L_struct"])
        with torch.no_grad():
            generator.copy_(saved_generator)
        _require(negative_rejected, "negative-control update was not rejected")
        _trace("preflight: directional negative control complete")
        _restore_preflight_state(consumer, optimizer, sampler, initial_state, initial_optimizer, initial_rng, initial_cursor, smooth, independent, ec, ed)

        traces: list[dict[str, Any]] = []
        update_seconds: list[float] = []
        memory_records: list[dict[str, Any]] = []
        scalar_parameters: torch.Tensor | None = None
        scalar_optimizer: Mapping[str, Any] | None = None
        for update in range(3):
            _trace(f"preflight: real update {update + 1}")
            update_indices, update_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, update_indices), update_labels), f"preflight TRAIN labels changed at update {update + 1}")
            trace, seconds, memory = _timed_scalar_update(consumer, smooth, independent, ec, optimizer, authorities["images"].index_select(0, update_indices).to(device), references, mean, scale, step=START_STEP + update + 1, clean_generator_state=initial_state["generator.phase"], device=device)
            trace = dict(trace)
            trace["timing"] = {"seconds": seconds, "cuda_memory": memory}
            traces.append(trace)
            update_seconds.append(seconds)
            memory_records.append(memory)
            if update == 0:
                scalar_parameters = generator.detach().clone()
                scalar_optimizer = copy.deepcopy(optimizer.state_dict())
        _require(scalar_parameters is not None and scalar_optimizer is not None, "one-step scalar reference was not captured")
        _require(not torch.equal(initial_state["generator.phase"], joint._runtime_state(consumer)["generator.phase"]), "three-step preflight did not update generator")
        _require(torch.equal(initial_state["encoder.phase"], joint._runtime_state(consumer)["encoder.phase"]), "three-step preflight changed encoder")
        _require(all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters()), "three-step preflight left evaluator gradients")
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        resource = _resource_envelope(update_seconds, memory_records, device=device, total_memory_bytes=total_memory)
        _require(resource["resource_ok"], f"PREFLIGHT_BLOCKED(resource): median scalar update {resource['median_update_seconds']:.3f}s; projected 300-update runtime {resource['projected_300_update_hours']:.3f}h")
        _trace("preflight: resource envelope complete")

        reference_parameter = torch.nn.Parameter(initial_state["generator.phase"].detach().clone().to(generator.device))
        reference_optimizer = torch.optim.Adam([reference_parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, amsgrad=False)
        reference_optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        reference_parameter.grad = scalar_gradients["g_total"].detach().clone().to(reference_parameter.device)
        reference_optimizer.step()
        adam_equivalence = {
            "parameter_update": joint._compare_gradient_tensor(scalar_parameters, reference_parameter.detach(), "one-step generator parameter"),
            "optimizer_state": joint._compare_optimizer_states(scalar_optimizer, reference_optimizer.state_dict(), name="one-step Adam"),
            "scalar_engine": traces[0]["gradient_engine"],
            "reference_engine": "independent scalar Adam formula from saved step9001 direction",
            "passed": True,
        }
        _restore_preflight_state(consumer, optimizer, sampler, initial_state, initial_optimizer, initial_rng, initial_cursor, smooth, independent, ec, ed)

        _trace("preflight: one semantic batched reference")
        batched = _semantic_batched_reference(consumer, smooth, independent, ec, batch, references)
        semantic_comparisons = {name: joint._compare_gradient_tensor(scalar_gradients[name], batched[name], name) for name in ("g_S", "g_I", "g_C")}
        joint.semantic._clear_runtime_grads(consumer)
        joint.semantic._clear_evaluator_grads(smooth, independent, ec, ed)
        gradient_equivalence = {"rows": 3, "mapping": ["L_S->g_S", "L_I->g_I", "L_C->g_C"], "comparison": semantic_comparisons, "batched_calls": 1, "all_equivalent": True, "E_D_in_backward": False}
        _trace("preflight: rollback complete")
    finally:
        _restore_preflight_state(consumer, optimizer, sampler, initial_state, initial_optimizer, initial_rng, initial_cursor, smooth, independent, ec, ed)

    rollback = {
        "runtime": joint._state_digest(joint._runtime_state(consumer)) == joint._state_digest(initial_state),
        "frozen": joint._state_digest(joint._frozen_runtime_state(consumer)) == joint._state_digest(initial_frozen),
        "adam": _exact_equal(optimizer.state_dict(), initial_optimizer),
        "rng": joint._rng_equal(joint._rng_snapshot(), initial_rng),
        "sampler": sampler.cursor == initial_cursor,
        "evaluator_grads": all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters()),
    }
    _require(all(rollback.values()), "preflight state restoration failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "gradient_engine": GRADIENT_ENGINE,
        "training_backend": SCALAR_REFERENCE_ENGINE,
        "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "cosine_minimum": GRADIENT_COSINE_MINIMUM},
        "accepted_step9000_gradient_identity": accepted_gradient_identity,
        "scalar_step9001_direction": scalar_direction_record,
        "directional_sanity": {"epsilon": PRELIGHT_EPSILON, "before": before, "after": after, "q_joint_finite": True},
        "negative_control": {"wrong_sign_after": negative_after, "directional_reference": negative_reference, "rejected": negative_rejected},
        "one_step_adam_equivalence": adam_equivalence,
        "three_step_real_update": {"updates": 3, "persistent_updates": 0, "trace": traces},
        "resource_envelope": resource,
        "semantic_batched_reference": gradient_equivalence,
        "cuda_runtime": {"device": str(device), "device_count": torch.cuda.device_count(), "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory), "peak_memory": resource["peak_memory"]},
        "starting_cursor": START_STEP,
        "adam_internal_step_at_start": 9300.0,
        "rollback": rollback,
        "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0},
    }


def _identity(source_manifest: Mapping[str, Any], attempt_name: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="local-compute-bounded-joint-canary-v1",
        trajectory_id="local-compute-bounded-joint-canary-trajectory-v1",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"start_step": START_STEP, "max_updates": MAX_UPDATES, "primary_endpoint": PRIMARY_ENDPOINT, "checkpoint_steps": list(CHECKPOINT_STEPS), "gradient_engine": GRADIENT_ENGINE, "delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
        model_identity=TASK_ID,
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 9300.0, "reset": False},
        sampler_identity={"seed": SCHEDULE_SEED, "starting_cursor": START_STEP, "terminal_cursor": PRIMARY_ENDPOINT, "accepted_prefix_preserved": True, "wraparound": True},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=f"{TASK_ID}:{attempt_name}",
    )


def _preflight_failure_state_dir(target: Path) -> Path:
    return target.parent / f"{target.name}.preflight_failures"


def _load_preflight_failure_counts(target: Path) -> dict[str, int]:
    state_dir = _preflight_failure_state_dir(target)
    counts: dict[str, int] = {}
    if not state_dir.is_dir():
        return counts
    for path in sorted(state_dir.glob("failure_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if payload.get("schema") != PREFLIGHT_FAILURE_STATE_SCHEMA:
            continue
        key = payload.get("failure_key")
        if not isinstance(key, str) or not key:
            continue
        try:
            count = int(payload.get("count", 0))
        except (TypeError, ValueError):
            continue
        counts[key] = max(counts.get(key, 0), count)
    return counts


def _classify_preflight_failure(error: BaseException) -> tuple[str, str]:
    message = str(error)
    lowered = message.lower()
    if "preflight_blocked(resource)" in lowered or "out of memory" in lowered:
        if "memory" in lowered or "oom" in lowered:
            return "resource", "canary-cuda-memory"
        return "resource", "canary-300-update-runtime-budget"
    if "differs between gradient engines" in lowered or "cosine is below strict preflight threshold" in lowered:
        return "correctness", "gradient-equivalence"
    if "negative-control" in lowered:
        return "correctness", "negative-control"
    if "restore" in lowered or "rollback" in lowered:
        return "correctness", "state-restoration"
    return "preflight", f"{type(error).__name__}:{message.splitlines()[0][:160]}"


def _preflight_failure_control(target: Path) -> ResearchRunControl:
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="local-compute-bounded-joint-canary-v1:preflight",
        trajectory_id="local-compute-bounded-joint-canary-trajectory-v1:preflight",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"role": "operational_preflight_retry_budget"},
        model_identity=TASK_ID,
        optimizer_identity={},
        sampler_identity={},
        execution_source_semantics_id=f"{TASK_ID}:preflight",
    )
    return ResearchRunControl(identity, AUTHORIZATION_ID, preflight_failure_counts=_load_preflight_failure_counts(target))


def _record_persisted_preflight_failure(target: Path, control: ResearchRunControl, component: str, signature: str, decision: str) -> None:
    state_dir = _preflight_failure_state_dir(target)
    state_dir.mkdir(parents=True, exist_ok=True)
    failure_key = f"{component}:{signature}"
    count = int(control.preflight_failure_counts[failure_key])
    key_digest = hashlib.sha256(failure_key.encode("utf-8")).hexdigest()[:12]
    path = state_dir / f"failure_{key_digest}_{count:02d}.json"
    if path.exists():
        return
    _write_json(
        path,
        {
            "schema": PREFLIGHT_FAILURE_STATE_SCHEMA,
            "failure_key": failure_key,
            "component": component,
            "signature": signature,
            "count": count,
            "decision": decision,
            "authorization_id": AUTHORIZATION_ID,
            "task_id": TASK_ID,
        },
        "Canary preflight failure state",
    )


def _raise_recorded_preflight_failure(target: Path, error: BaseException) -> None:
    component, signature = _classify_preflight_failure(error)
    control = _preflight_failure_control(target)
    decision = control.record_preflight_failure(component, signature)
    _record_persisted_preflight_failure(target, control, component, signature, decision)
    raise CanaryError(f"{error} [preflight failure decision={decision}]") from error


def _write_invalid_harness_closeout(
    root: Path,
    target: Path,
    source_manifest: Mapping[str, Any],
    receipt: PreflightReceipt,
    error: BaseException,
) -> None:
    """Close an opened attempt without fabricating a scientific result."""

    _write_json(
        target / INVALID_HARNESS_CLOSEOUT_NAME,
        {
            "schema": ATTEMPT_CLOSEOUT_SCHEMA,
            "task_id": TASK_ID,
            "authorization_id": AUTHORIZATION_ID,
            "attempt_id": f"{TASK_ID}:{target.name}",
            "state": "INVALID_HARNESS",
            "classification": "INVALID_HARNESS",
            "scientific_result_valid": False,
            "scientific_forward_evaluation_started": True,
            "private_baseline_materialization_started": True,
            "private_data_reads": {"DEV": 5000, "protected": 0, "test": 0, "sealed": 0},
            "persistent_updates": 0,
            "checkpoint_writes": 0,
            "stage1_endpoint_evaluated": False,
            "training_started": False,
            "root_cause": {
                "classification": "CONSUMER_COMPARISON_BOUNDARY",
                "error_type": type(error).__name__,
                "error": str(error),
                "repair": "explicit canonical scientific structural projection; private namespace name is not compared",
            },
            "attempt_policy": {
                "consumed": True,
                "resume_allowed": False,
                "next_action": "freeze this attempt and open a new attempt after repair",
            },
            "contract": {"path": _relative(root, root / CONTRACT_PATH), "sha256": file_sha256(root / CONTRACT_PATH)},
            "execution_source_manifest": dict(source_manifest),
            "preflight_receipt": receipt.to_dict(),
            "immutable_existing_artifacts_preserved": True,
            "no_scientific_classification": True,
        },
        "Canary invalid-harness closeout",
    )


def _preflight_receipt(source_manifest: Mapping[str, Any], preflight: Mapping[str, Any]) -> PreflightReceipt:
    correctness = _mapping(preflight.get("negative_control"), "Canary negative control")
    semantic_reference = _mapping(preflight.get("semantic_batched_reference"), "Canary semantic batched reference")
    adam = _mapping(preflight.get("one_step_adam_equivalence"), "Canary Adam equivalence")
    rollback = _mapping(preflight.get("rollback"), "Canary rollback")
    rollback_passed = all(rollback.get(name) is True for name in ("runtime", "frozen", "adam", "rng", "sampler", "evaluator_grads"))
    resource = _mapping(preflight.get("resource_envelope"), "Canary resource envelope")
    implementation_passed = bool(
        preflight.get("gradient_engine") == GRADIENT_ENGINE
        and preflight.get("training_backend") == SCALAR_REFERENCE_ENGINE
    )
    return PreflightReceipt(
        source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        authorization_id=AUTHORIZATION_ID,
        checks={
            CORRECTNESS_INVARIANTS: {
                "passed": bool(
                    semantic_reference.get("all_equivalent")
                    and adam.get("passed")
                    and _mapping(preflight.get("three_step_real_update"), "Canary three-step update").get("updates") == 3
                    and rollback_passed
                    and correctness.get("rejected")
                ),
                "gradient_equivalence": bool(semantic_reference.get("all_equivalent")),
                "adam_one_step_equivalence": bool(adam.get("passed")),
                "three_step_update_and_restore": bool(_mapping(preflight.get("three_step_real_update"), "Canary three-step update").get("updates") == 3 and rollback_passed),
                "negative_control": bool(correctness.get("rejected")),
            },
            RESOURCE_ENVELOPE: {
                "passed": bool(resource.get("resource_ok") and str(resource.get("status", "")).upper() in {"READY", "PASS", "PASSED"}),
                "status": resource.get("status"),
                "resource_ok": bool(resource.get("resource_ok")),
                "median_update_seconds": resource.get("median_update_seconds"),
                "projected_300_update_hours": resource.get("projected_300_update_hours"),
                "peak_memory": resource.get("peak_memory"),
                "cuda": True,
            },
            IMPLEMENTATION_SELECTION: {"passed": implementation_passed, "selected_engine": SCALAR_REFERENCE_ENGINE, "gradient_engine": preflight.get("gradient_engine"), "training_backend": preflight.get("training_backend"), "batched_reference_calls": semantic_reference.get("batched_calls"), "fixed": True},
        },
        selected_engine=SCALAR_REFERENCE_ENGINE,
        resource={
            "budget": "bounded_canary",
            "median_update_seconds": resource.get("median_update_seconds"),
            "projected_300_update_hours": resource.get("projected_300_update_hours"),
            "max_projected_300_update_hours": resource.get("max_projected_300_update_hours"),
            "peak_memory": resource.get("peak_memory"),
            "resource_ok": bool(resource.get("resource_ok")),
        },
        created_at=time.time(),
    )


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: Any, history: list[Mapping[str, Any]], trajectory: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, stage1_condition: Mapping[str, Any] | None, dev_records: Mapping[str, Any], s1_records: Mapping[str, Any], timing: Mapping[str, Any], source_snapshot: Mapping[str, Any], stage1_bootstrap: Mapping[str, Any] | None, stage1_reconstruction: Mapping[str, Any] | None, stage1_optical: Mapping[str, Any] | None, scientific: bool) -> dict[str, Any]:
    state = optimizer.state[joint._generator(consumer)]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_ids": [AUTHORIZATION_ID],
        "completed_updates": step,
        "starting_scientific_step": START_STEP,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "checkpoint_selection": "none; every preregistered Canary checkpoint persisted",
        "scientific_artifact": scientific,
        "operational_artifact": not scientific,
        "gradient_engine": GRADIENT_ENGINE,
        "scalar_reference_gradient_engine": SCALAR_REFERENCE_ENGINE,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "starting_authority": {"path": START_CHECKPOINT.as_posix(), "sha256": START_CHECKPOINT_SHA256, "selected_step": START_STEP},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "S1", "E_S", "E_I", "E_C", "E_D", "f_struct"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": joint._state_digest(joint._frozen_runtime_state(consumer)),
        "optimizer_mode": "STEP9000_ADAM_STATE_RESTORED_EXACTLY",
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "internal_step": float(state["step"])},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "joint_contract": {"delta_joint": DELTA_JOINT, "direction": "q_joint=(u_struct+u_SIC)/RMS(u_struct+u_SIC)", "descent": "-q_joint", "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "E_D_in_backward": False, "gradient_engine": GRADIENT_ENGINE},
        "auxiliary_coefficients": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
        "history_tail": list(history[-10:]),
        "training_history": [dict(record) for record in history],
        "trajectory": dict(trajectory),
        "stage1_condition": None if stage1_condition is None else dict(stage1_condition),
        "dev_records": dict(dev_records),
        "s1_records": dict(s1_records),
        "timing": dict(timing),
        "source_snapshot": dict(source_snapshot),
        "stage1_bootstrap": None if stage1_bootstrap is None else dict(stage1_bootstrap),
        "stage1_reconstruction": None if stage1_reconstruction is None else dict(stage1_reconstruction),
        "stage1_optical": None if stage1_optical is None else dict(stage1_optical),
    }


def _validate_report(report: Mapping[str, Any], target: Path) -> None:
    _require(report.get("schema") == REPORT_SCHEMA, "Canary report schema is invalid")
    _require(report.get("task_id") == TASK_ID and report.get("result") == "VALID_COMPLETED", "Canary report is not complete")
    _require(report.get("primary_endpoint") == PRIMARY_ENDPOINT, "Canary report endpoint changed")
    _require(report.get("stage2", {}).get("executed") is False, "Canary Stage-2 was executed")
    _require(report.get("training_trajectory", {}).get("scientific_cursor") == PRIMARY_ENDPOINT, "Canary did not stop at 9300")
    _require((target / "scientific_step_009300.pt").is_file(), "Canary endpoint checkpoint is missing")
    _require(not (target / "scientific_step_009600.pt").exists() and not (target / "scientific_step_012000.pt").exists(), "Canary wrote a post-9300 checkpoint")


def _persist_report(target: Path, report: Mapping[str, Any], heartbeat: HeartbeatWriter, control: ResearchRunControl, step: int, classification: str) -> None:
    heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="FINALIZING", step=step, latest_losses={"report": "PERSISTING"}, force=True)
    _validate_report(report, target)
    _write_json(target / "report.json", report, "Canary final report")
    persisted = json.loads((target / "report.json").read_text(encoding="utf-8"))
    _validate_report(_mapping(persisted, "persisted Canary report"), target)
    if report.get("outer_gate_passed") is False:
        control.valid_failed()
        final_state = RUN_VALID_FAILED
    else:
        control.complete()
        final_state = RUN_COMPLETED
    heartbeat.update(state=final_state, phase="COMPLETED", step=step, latest_losses={"classification": classification}, force=True)


def run_cuda_preflight(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Run the strict in-memory CUDA preflight without creating an output run."""

    root = _root() if repo_root is None else repo_root.resolve()
    _trace("preflight: loading contract")
    payload, provenance = load_contract(root)
    target = (root / DEFAULT_OUTPUT_DIR).resolve()
    _require(not target.exists(), f"Canary attempt already exists: {target}")
    _require(not (root / HISTORICAL_ATTEMPT_DIR).resolve().exists(), "historical batched continuation attempt must not be consumed")
    _trace("preflight: loading authorities")
    authorities = joint._load_authorities(root)
    current_git = git_identity(root)
    authorities["local_state"] = {**authorities["local_state"], "branch": current_git["branch"], "head": current_git["head"], "dirty_before": current_git["worktree_dirty"], "status_sha256": current_git["status_sha256"]}
    source_manifest = execution_source_manifest(root, authorities, payload)
    _require(source_manifest["task_id"] == TASK_ID and bool(source_manifest.get("head")), "execution source manifest identity is incomplete")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "Canary requires CUDA")
    device = torch.device("cuda:0")
    cuda_runtime = joint.semantic.robust._prepare_cuda_runtime()
    _trace("preflight: loading models")
    consumer, smooth, independent, ec, ed = joint._load_models(root, authorities, device)
    checkpoint = authorities["step9000"]
    optimizer = joint._optimizer(consumer, checkpoint)
    sampler = _canary_sampler(checkpoint["sampler_state"], cursor=START_STEP)
    joint._restore_start_state(consumer, optimizer, sampler, checkpoint)
    _trace("preflight: entering strict checks")
    try:
        preflight = _preflight(consumer, smooth, independent, ec, ed, optimizer, sampler, authorities, authorities["surrogate_mean"], authorities["surrogate_scale"], device)
    except Exception as error:
        _raise_recorded_preflight_failure(target, error)
    _trace("preflight: strict checks complete")
    preflight["contract_sha256"] = file_sha256(root / CONTRACT_PATH)
    preflight["contract_provenance"] = provenance
    preflight["execution_source_manifest"] = source_manifest
    preflight["scientific_attempt_created"] = False
    preflight["training_started"] = False
    preflight["cuda_runtime"] = {**_mapping(preflight.get("cuda_runtime"), "preflight CUDA runtime"), "runtime_contract": cuda_runtime}
    return preflight


def run_canary(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "Canary output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"Canary attempt already exists: {target}")
    historical_target = (root / HISTORICAL_ATTEMPT_DIR).resolve()
    _require(not historical_target.exists(), "historical batched continuation attempt must not be consumed")
    payload, provenance = load_contract(root)
    authorities = joint._load_authorities(root)
    current_git = git_identity(root)
    authorities["local_state"] = {**authorities["local_state"], "branch": current_git["branch"], "head": current_git["head"], "dirty_before": current_git["worktree_dirty"], "status_sha256": current_git["status_sha256"]}
    source_manifest = execution_source_manifest(root, authorities, payload)
    _require(source_manifest["task_id"] == TASK_ID and bool(source_manifest.get("head")), "execution source manifest identity is incomplete")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "Canary requires CUDA")
    device = torch.device("cuda:0")
    cuda_runtime = joint.semantic.robust._prepare_cuda_runtime()
    consumer, smooth, independent, ec, ed = joint._load_models(root, authorities, device)
    checkpoint = authorities["step9000"]
    optimizer = joint._optimizer(consumer, checkpoint)
    sampler = _canary_sampler(checkpoint["sampler_state"], cursor=START_STEP)
    joint._restore_start_state(consumer, optimizer, sampler, checkpoint)
    mean, scale = authorities["surrogate_mean"], authorities["surrogate_scale"]
    try:
        preflight = _preflight(consumer, smooth, independent, ec, ed, optimizer, sampler, authorities, mean, scale, device)
    except Exception as error:
        _raise_recorded_preflight_failure(target, error)
    _require(preflight["status"] == "READY", "Canary CUDA preflight did not reach READY")

    identity = _identity(source_manifest, target.name)
    receipt = _preflight_receipt(source_manifest, preflight)
    control = ResearchRunControl(identity, AUTHORIZATION_ID, preflight_failure_counts=_load_preflight_failure_counts(target))
    control.mark_preflight_ready(receipt)
    effective_identity = control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt)
    _write_json(target / "source_manifest.json", source_manifest, "Canary execution source manifest")
    source_snapshot = _write_source_snapshot(root, target, source_manifest)
    contract_boundary = _mapping(payload.get("execution_boundary"), "Canary contract execution boundary")
    _write_json(
        target / "contract_binding.json",
        {
            "contract_path": _relative(root, root / CONTRACT_PATH),
            "contract_sha256": file_sha256(root / CONTRACT_PATH),
            "provenance": provenance,
            "artifact_proposed_attempt_path": contract_boundary.get("attempt_path"),
            "actual_canary_attempt_path": _relative(root, target),
            "historical_attempt_path_not_consumed": True,
            "historical_batched_path_not_consumed": True,
        },
        "Canary contract binding",
    )
    _write_json(target / "preflight.json", {**preflight, "preflight_receipt": receipt.to_dict()}, "Canary CUDA preflight")
    _write_json(target / "execution_manifest.json", {"schema": "lightgen_2d_local_compute_bounded_joint_canary_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN}, "gradient_engine": GRADIENT_ENGINE, "contract": {"path": _relative(root, root / CONTRACT_PATH), "sha256": file_sha256(root / CONTRACT_PATH)}, "source_manifest": source_manifest, "source_snapshot": source_snapshot, "run_identity": effective_identity.to_dict(), "local_state": authorities["local_state"], "cuda_runtime": cuda_runtime, "scope": {"start": START_STEP, "endpoint": PRIMARY_ENDPOINT, "max_updates": MAX_UPDATES, "checkpoints": list(CHECKPOINT_STEPS), "stage2": False, "protected": False, "test": False, "sealed": False}}, "Canary execution manifest")

    ledger = DataReadLedger()
    baseline_started = time.perf_counter()
    try:
        baseline_private = joint._materialize_step9000_private_baseline(consumer, authorities, authorities["structural_reference"], device)
    except joint.JointContinuationError as error:
        control.invalid_harness()
        _write_invalid_harness_closeout(root, target, source_manifest, receipt, error)
        raise CanaryError("INVALID_HARNESS: private step9000 baseline materialization failed") from error
    ledger.record("dev", int(baseline_private["rows"]))
    _write_torch(target / "baseline_private_step_009000.pt", baseline_private, "Canary step9000 private baseline")
    baseline_private_ref = {"path": _relative(root, target / "baseline_private_step_009000.pt"), "sha256": file_sha256(target / "baseline_private_step_009000.pt"), "role": "step9000 private paired-bootstrap baseline"}

    clean_generator_state = checkpoint["generator_state"].to(device)
    history: list[dict[str, Any]] = []
    trajectory: dict[str, dict[str, Any]] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    s1_records: dict[str, dict[str, Any]] = {}
    timing: dict[str, Any] = {"schema": TIMING_SCHEMA, "stages": {"private_baseline": {"seconds": time.perf_counter() - baseline_started}}, "updates": []}
    fixed_indices = torch.stack([value[START_STEP % value.numel()] for value in sampler.permutations])
    fixed_batch = authorities["images"].index_select(0, fixed_indices).to(device)
    composition = _scalar_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"], mean, scale)
    trajectory[str(START_STEP)] = joint._composition_record(composition, START_STEP, 0.0, 0.0)
    joint.semantic._clear_runtime_grads(consumer)
    joint.semantic._clear_evaluator_grads(smooth, independent, ec, ed)
    reused_baseline = authorities["reused_step9000_baseline"]
    dev_records[str(START_STEP)] = dict(_mapping(reused_baseline["dev"], "reused step9000 DEV baseline"))
    dev_records[str(START_STEP)]["_labels_dev"] = baseline_private["labels_dev"].clone()
    dev_records[str(START_STEP)]["_log_rho"] = baseline_private["log_rho"].clone()
    s1_records[str(START_STEP)] = dict(_mapping(reused_baseline["s1"], "reused step9000 S1 baseline"))
    _write_json(target / "dev_diagnostics_step_009000.json", {key: value for key, value in dev_records[str(START_STEP)].items() if not key.startswith("_")}, "Canary step9000 DEV diagnostics")
    _write_json(target / "s1_transfer_step_009000.json", {key: value for key, value in s1_records[str(START_STEP)].items() if not key.startswith("_")}, "Canary step9000 S1 diagnostics")

    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=PRIMARY_ENDPOINT, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="PREPARING", step=START_STEP, latest_losses={"preflight": "READY", "optimizer_mode": "STEP9000_ADAM_STATE_RESTORED_EXACTLY"}, force=True)
    heartbeat.start_liveness()
    frozen_initial = joint._frozen_runtime_state(consumer)
    direction_stable = True
    stage1_passed = False
    stage1_checks: dict[str, bool] = {}
    stage1_bootstrap: dict[str, Any] | None = None
    stage1_reconstruction: dict[str, Any] | None = None
    stage1_optical: dict[str, Any] | None = None
    last_recovery_step = START_STEP
    last_recovery_time = time.time()

    def recovery_payload(step: int) -> Mapping[str, Any]:
        payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical, scientific=False)
        payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
        return payload

    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_payload, phase="TRAINING", total_steps=PRIMARY_ENDPOINT)
    try:
        with interrupt:
            while sampler.cursor < PRIMARY_ENDPOINT:
                current_step = sampler.cursor
                indices, labels = sampler.next_batch()
                _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"TRAIN labels changed at update {current_step + 1}")
                batch = authorities["images"].index_select(0, indices).to(device)
                update_started = time.perf_counter()
                try:
                    trace = joint._apply_update(consumer, smooth, independent, ec, optimizer, batch, authorities["training"]["references"], mean, scale, step=current_step + 1, clean_generator_state=clean_generator_state, gradient_engine=SCALAR_REFERENCE_ENGINE)
                except (joint.JointDirectionDegenerated, CanaryDirectionDegenerated):
                    direction_stable = False
                    break
                trace["timing"] = {"seconds": time.perf_counter() - update_started, "cuda_memory": joint._cuda_memory(device)}
                timing["updates"].append(dict(trace["timing"]))
                history.append(trace)
                completed = current_step + 1
                _require(joint._state_digest(joint._frozen_runtime_state(consumer)) == joint._state_digest(frozen_initial), f"frozen runtime changed at update {completed}")
                now = time.time()
                if recovery_due(step=completed, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                    checkpoint_path = recovery.save(recovery_payload(completed), step=completed)
                    heartbeat.remember_recovery_checkpoint(checkpoint_path.name)
                    last_recovery_step, last_recovery_time = completed, now
                heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="TRAINING", step=completed, latest_losses={name: trace[name] for name in ("L_struct", "L_S", "L_I", "L_C")})
                heartbeat.raise_liveness_error()
                interrupted = interrupt.handle_safe_boundary(step=completed, latest_losses={name: trace[name] for name in ("L_struct", "L_S", "L_I", "L_C")}, last_recovery_checkpoint=None)
                if interrupted is not None:
                    control.interrupted()
                    heartbeat.stop_liveness()
                    heartbeat.update(state=RUN_INTERRUPTED, phase="INTERRUPTED", step=completed, latest_losses={"reason": "interrupt", "latest_step": completed}, last_recovery_checkpoint=interrupted.name, force=True)
                    return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": completed}
                if completed in CHECKPOINT_STEPS:
                    composition = _scalar_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"], mean, scale)
                    trajectory[str(completed)] = joint._composition_record(composition, completed, joint._norm(joint._generator(consumer).detach() - clean_generator_state, "phase displacement"), float(trace["phase_update_rms"]))
                    joint.semantic._clear_runtime_grads(consumer)
                    joint.semantic._clear_evaluator_grads(smooth, independent, ec, ed)
                    _require(float(trajectory[str(completed)]["v_joint_l2_norm"]) > JOINT_DEGENERACY_TOLERANCE, f"joint direction degenerated at checkpoint {completed}")
                    if completed < PRIMARY_ENDPOINT:
                        checkpoint_payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": False, "checks": {}}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=None, stage1_reconstruction=None, stage1_optical=None, scientific=True)
                        _write_torch(target / f"scientific_step_{completed:06d}.pt", checkpoint_payload, f"Canary scientific step-{completed} checkpoint")
                if completed == PRIMARY_ENDPOINT:
                    stage_started = time.perf_counter()
                    dev_records[str(completed)] = joint._evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], mean, scale, ledger, device, step=completed)
                    s1_records[str(completed)] = joint._evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=completed)
                    _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {key: value for key, value in dev_records[str(completed)].items() if not key.startswith("_")}, f"Canary step-{completed} DEV diagnostics")
                    _write_json(target / f"s1_transfer_step_{completed:06d}.json", {key: value for key, value in s1_records[str(completed)].items() if not key.startswith("_")}, f"Canary step-{completed} S1 diagnostics")
                    stage1_bootstrap = joint._paired_bootstrap(dev_records[str(START_STEP)]["_labels_dev"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(completed)]["_log_rho"])
                    stage1_reconstruction = {"step9000": dev_records[str(START_STEP)]["L_rec"], "step9300": dev_records[str(completed)]["L_rec"], "delta": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
                    stage1_optical = joint._optical_pass(dev_records[str(START_STEP)], dev_records[str(completed)])
                    stage1_passed, stage1_checks = joint._stage1_condition(dev_records[str(START_STEP)], dev_records[str(completed)], stage1_bootstrap, stage1_reconstruction, stage1_optical, trajectory)
                    timing["stages"]["stage1_evaluation"] = {"seconds": time.perf_counter() - stage_started, "cuda_memory": joint._cuda_memory(device)}
                    checkpoint_payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical, scientific=True)
                    _write_torch(target / "scientific_step_009300.pt", checkpoint_payload, "Canary scientific endpoint checkpoint")
                    break
    except Exception:
        heartbeat.stop_liveness()
        raise

    _require(sampler.cursor == PRIMARY_ENDPOINT, f"Canary stopped before endpoint: {sampler.cursor}")
    final_dev = dev_records[str(PRIMARY_ENDPOINT)]
    final_s1 = s1_records[str(PRIMARY_ENDPOINT)]
    final_semantic = all(float(final_dev[name]) <= float(dev_records[str(START_STEP)][name]) for name in ("L_S", "L_I", "L_C"))
    s1_transfer = float(final_s1["structural"]["structural_support_rate"]) >= float(s1_records[str(START_STEP)]["structural"]["structural_support_rate"]) and all(float(final_s1["semantic"][name]["macro_recall"]) >= float(s1_records[str(START_STEP)]["semantic"][name]["macro_recall"]) for name in ("E_S", "E_I", "E_C", "E_D"))
    classification = joint._classify(stage1_passed=stage1_passed, direction_stable=direction_stable, final_structural=stage1_bootstrap, final_semantic=final_semantic, s1_transfer=s1_transfer, reconstruction=stage1_reconstruction, optical=stage1_optical)
    public_dev = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in dev_records.items()}
    public_s1 = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in s1_records.items()}
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(stage1_passed),
        "classification": classification,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "exactly step9000->9300; no continuation"},
        "contract": {"path": _relative(root, root / CONTRACT_PATH), "sha256": file_sha256(root / CONTRACT_PATH), "provenance": provenance},
        "local_state": authorities["local_state"],
        "execution_source_manifest": source_manifest,
        "starting_state": {"step": START_STEP, "checkpoint_sha256": START_CHECKPOINT_SHA256, "optimizer_internal_step": 9300.0, "sampler_cursor": START_STEP, "rng_restored": True},
        "primary_endpoint": PRIMARY_ENDPOINT,
        "gradient_engine": GRADIENT_ENGINE,
        "scalar_reference_gradient_engine": SCALAR_REFERENCE_ENGINE,
        "joint_gradient_contract": {"delta_joint": DELTA_JOINT, "direction": "q_joint=(u_struct+u_SIC)/RMS(u_struct+u_SIC)", "descent": "-q_joint", "E_D_in_backward": False},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 9300.0, "reset": False},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T=I_100", "S1", "E_S", "E_I", "E_C", "E_D"], "E_D_in_backward": False},
        "preflight": preflight,
        "stage1": {"updates": len(history), "checkpoints": [9100, 9200, 9300], "development": public_dev, "S1_transfer": public_s1, "bootstrap": {key: value for key, value in stage1_bootstrap.items() if not isinstance(value, torch.Tensor)}, "reconstruction": stage1_reconstruction, "optical_validity": stage1_optical, "continuation_condition": {"passed": stage1_passed, "checks": stage1_checks}},
        "stage2": {"executed": False, "checkpoints": [], "final_step": PRIMARY_ENDPOINT, "selection": "not applicable; 9300 is the unique endpoint"},
        "evaluation_sample_counts": {"DEV": {key: int(record["rows"]) for key, record in dev_records.items()}, "S1": {key: int(record["rows"]) for key, record in s1_records.items()}},
        "training_trajectory": {"updates": len(history), "scientific_cursor": sampler.cursor, "checkpoint_steps_persisted": [step for step in CHECKPOINT_STEPS if (target / f"scientific_step_{step:06d}.pt").is_file()], "trace": history, "composition": trajectory},
        "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0},
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "timing": timing,
        "cuda_runtime": cuda_runtime,
        "claim_boundary": {"bounded_canary_only": True, "no_checkpoint_selection": True, "no_continuation": True, "promotion": False, "hardware_readiness": False},
        "baseline_private": baseline_private_ref,
        "latest_completed_checkpoint": PRIMARY_ENDPOINT,
        "no_training_beyond_9300": True,
        "output_dir": _relative(root, target),
    }
    try:
        _persist_report(target, report, heartbeat, control, PRIMARY_ENDPOINT, classification)
    finally:
        heartbeat.stop_liveness()
    return report


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
