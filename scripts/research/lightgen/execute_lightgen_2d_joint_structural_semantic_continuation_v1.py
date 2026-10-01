"""Execute the frozen step-9000 joint structural/semantic continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, TRAIN_ROLE, contracted_plane_powers, git_identity, _rng_state_payload
from ldm_project.v2.application.research.shared.research_run_control import CORRECTNESS_INVARIANTS, HeartbeatWriter, IMPLEMENTATION_SELECTION, InterruptController, PreflightReceipt, RecoveryCheckpointManager, RESOURCE_ENVELOPE, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUN_INTERRUPTED, RUNNING, RUN_PREPARING, RUN_SCIENTIFIC_ACTIVE, RUN_VALID_FAILED, ResumeCompatibilityError, capture_execution_source_manifest, check_resume_compatibility, recovery_due
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, file_sha256, safe_torch_load

from scripts.research.lightgen import execute_lightgen_2d_attribute_step9000_structural_classifier_semantic_tradeoff_v1 as attribution
from scripts.research.lightgen import execute_lightgen_2d_evaluator_agnostic_structural_image_alignment_v1 as structural
from scripts.research.lightgen import execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 as semantic


TASK_ID = "EXECUTE_JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_CONTRACT_DEFINED"
SEMANTIC_AUTHORITY_ID = semantic.SEMANTIC_AUTHORITY_ID
EXPECTED_BRANCH = semantic.EXPECTED_BRANCH
EXPECTED_PRODUCER_HEAD = semantic.EXPECTED_PRODUCER_HEAD
# Historical import compatibility; producer provenance is not a consumer lock.
EXPECTED_HEAD = EXPECTED_PRODUCER_HEAD

START_STEP = 9000
STAGE1_END = 9300
TOTAL_STEPS = 12000
STAGE1_UPDATES = 300
CHECKPOINT_STEPS = (9000, 9100, 9200, 9300, 9600, 10000, 10500, 11000, 11500, 12000)
LOGICAL_BATCH_SIZE = 10
ATTRIBUTION_MICRO_BATCH_SIZE = 10
SCHEDULE_SEED = 1234
FEATURE_EPS = 1.0e-12
JOINT_DEGENERACY_TOLERANCE = 1.0e-12
DELTA_JOINT = 1.365988374680088e-05
BETA = 2.4509953404
GAMMA = 0.5330424285
EPS_PROJ = 1.0e-12
RECONSTRUCTION_TOLERANCE = 0.005
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
PRELIGHT_EPSILON = 1.0e-4
RECOVERY_INTERVAL_STEPS = 25

GRADIENT_ENGINE = "HYBRID_SEMANTIC_BATCHED_VJP"
SCALAR_REFERENCE_ENGINE = "SCALAR_REFERENCE_VJP"
GRADIENT_RTOL = 2.0e-5
GRADIENT_ATOL = 2.0e-7
GRADIENT_COSINE_MINIMUM = 0.999999
PERFORMANCE_WARMUP_RUNS = 1
PERFORMANCE_TIMED_RUNS = 5
PERFORMANCE_MIN_SPEEDUP = 0.10
PERFORMANCE_MAX_MEMORY_FRACTION = 0.90
BASELINE_PRIVATE_TIMEOUT_SECONDS = 600.0

START_CHECKPOINT = Path("outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_03/scientific_step_009000.pt")
START_CHECKPOINT_SHA256 = "14e76f3273e6f794cbadaee63a881bd686645016c5c5aba62c727ce9860afcee"
ATTRIBUTION_REPORT = Path("outputs/20260921_attribute_step9000_structural_classifier_semantic_tradeoff_v1/runs/attempt_01/report.json")
ATTRIBUTION_REPORT_SHA256 = "5fdbfa5db583808a411280463acbd2219aed57b158bef7560bb24dd506757a83"
ATTRIBUTION_GRADIENTS = Path("outputs/20260921_attribute_step9000_structural_classifier_semantic_tradeoff_v1/runs/attempt_01/gradient_attribution.pt")
ATTRIBUTION_GRADIENTS_SHA256 = "8ff09bf3c30fabd494af16f578c6223c148fd36d1408797af10cbd65ae3df2e2"
ACCEPTED_STRUCTURAL_REPORT = Path("outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_03/report.json")
ACCEPTED_STRUCTURAL_REPORT_SHA256 = "83ae194a80699557c40864dcf994e73027f6c1ca7ecbaac5ccf1cd03e46fdc30"
SURROGATE_NORMALIZATION = Path("outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_03/surrogate_normalization.pt")
SURROGATE_NORMALIZATION_SHA256 = "63d0e11aefadd55acb7248b91b631a44a6a9aa4a6da4ca624e35b3276cd2cf98"
STRUCTURAL_REFERENCE = Path("outputs/20260921_non_classifier_structural_semantic_confirmation_v1/runs/attempt_04/structural_reference.pt")
STRUCTURAL_REFERENCE_SHA256 = "e6304ac70fafc7f5d5ecc61c0d91af6520fa257bc8fe50903cbe340a7bcae142"
S1_WINNER = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/winner_frozen_sampler.pt")
S1_WINNER_SHA256 = "166cf15ea91666afe1fe1c88e8897b88d58a4740253d59838927502babba568c"
REUSED_STEP9000_DEV = Path("outputs/20260921_joint_structural_semantic_continuation_v1/runs/attempt_03/dev_diagnostics_step_009000.json")
REUSED_STEP9000_DEV_SHA256 = "540A51673F1DE59FB21FF56DB21D284B60DFBCC7ABCA3AF137B75CFFDF778200"
REUSED_STRUCTURAL_STEP9000_DEV = Path("outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_03/dev_diagnostics_step_009000.json")
REUSED_STRUCTURAL_STEP9000_DEV_SHA256 = "4F62E0D14AD3E76AE4D9352A8FC1E98719BF73556C004D7D65BDA87523DFAD01"
REUSED_STEP9000_S1 = Path("outputs/20260921_joint_structural_semantic_continuation_v1/runs/attempt_03/s1_transfer_step_009000.json")
REUSED_STEP9000_S1_SHA256 = "E5FC39527B76E7701B3AD8837ADCF58899E8B410F1018A8C91E9F16FECA2C330"
REUSED_STEP9000_CHECKPOINT = Path("outputs/20260921_joint_structural_semantic_continuation_v1/runs/attempt_03/scientific_step_009000.pt")
REUSED_STEP9000_CHECKPOINT_SHA256 = "E7D65686C35B3C1C8522DD594B9D0A29E36D224AD1EBB35EB4BDC8FE9DBCF52B"
REUSED_STEP9000_EXECUTION_MANIFEST = Path("outputs/20260921_joint_structural_semantic_continuation_v1/runs/attempt_03/execution_manifest.json")
REUSED_STEP9000_EXECUTION_MANIFEST_SHA256 = "B8414684089358E57E0F7D85D95C605AB5C617C9E72C750B0E4752076182F485"
REUSED_STEP9000_SOURCE_MANIFEST = Path("outputs/20260921_joint_structural_semantic_continuation_v1/runs/attempt_03/source_manifest.json")
REUSED_STEP9000_SOURCE_MANIFEST_SHA256 = "8D98B81DF3B6761BEBC4B1B1FF3A38F918F0ADCB731B706247749BBAE86F66D8"
DEFAULT_OUTPUT_DIR = Path("outputs/20260922_joint_structural_semantic_continuation_batched_vjp_v1/runs/attempt_01")
DEFAULT_DIAGNOSTIC_OUTPUT_DIR = Path("outputs/20260922_joint_structural_semantic_continuation_v1_diagnostic/runs/preflight_01")

CHECKPOINT_SCHEMA = "joint_structural_semantic_continuation_checkpoint.v2"
REPORT_SCHEMA = "joint_structural_semantic_continuation_report.v2"
PREFLIGHT_SCHEMA = "joint_structural_semantic_continuation_preflight.v2"
DEV_SCHEMA = "joint_structural_semantic_continuation_dev_diagnostic.v1"
SOURCE_SNAPSHOT_SCHEMA = "joint_structural_semantic_continuation_source_snapshot.v1"
TIMING_SCHEMA = "joint_structural_semantic_continuation_timing.v1"
DIAGNOSTIC_SCHEMA = "joint_structural_semantic_continuation_diagnostic_preflight.v1"
BASELINE_PRIVATE_SCHEMA = "joint_structural_semantic_continuation_step9000_private_baseline.v1"

# ``structural._hard_structural_source`` includes ``name`` in its public
# mapping even though that field identifies the producer's namespace rather
# than the scientific summary.  Keep the comparison explicit: every current
# scientific field is listed here, and any added/removed public field remains
# fail-closed instead of being silently ignored.
STRUCTURAL_SCIENTIFIC_PUBLIC_FIELDS = (
    "rows",
    "label_source",
    "label_histogram",
    "structural_support_rate",
    "support_by_class",
    "nearest_same_class_distance",
    "distance_by_class",
    "q95_rows",
    "rho_struct",
    "log_rho_struct",
    "requested_centroid_distance",
    "alternative_centroid_distance",
    "margin",
    "margin_by_class",
    "positive_margin_fraction",
    "positive_margin_by_class",
    "nearest_class_agreement",
    "topology",
    "signature_sha256",
)

CLASSIFICATION_A = "JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_SUPPORTED_WITH_S1_TRANSFER"
CLASSIFICATION_B = "JOINT_REAL_PATH_RECOVERY_WITH_LIMITED_S1_TRANSFER"
CLASSIFICATION_C = "EQUAL_JOINT_DIRECTION_FAILS_TO_RESOLVE_LONG_HORIZON_TRADEOFF"
CLASSIFICATION_D = "JOINT_SEMANTIC_RECOVERY_WITHOUT_CANONICAL_STRUCTURAL_TRANSFER"
CLASSIFICATION_E = "JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_NOT_SUPPORTED"
CLASSIFICATION_F = "JOINT_STRUCTURAL_SEMANTIC_DIRECTION_NOT_GLOBALLY_STABLE"
CLASSIFICATION_G = "JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_MIXED"


class JointContinuationError(RuntimeError):
    """Raised when an accepted authority or continuation invariant is invalid."""


class JointDirectionDegenerated(JointContinuationError):
    """Raised only for the preregistered joint-direction stop."""


def _require(condition: bool, message: str, error_type: type[Exception] = JointContinuationError) -> None:
    if not condition:
        raise error_type(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _canonical_structural_scientific_public(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Project a structural summary onto its scientific, namespace-free fields."""

    public = {key: item for key, item in value.items() if not str(key).startswith("_")}
    expected_keys = set(STRUCTURAL_SCIENTIFIC_PUBLIC_FIELDS) | {"name"}
    _require(set(public) == expected_keys, f"{label} public structural schema changed")
    return {key: public[key] for key in STRUCTURAL_SCIENTIFIC_PUBLIC_FIELDS}


def _compare_structural_scientific_public(computed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Compare scientific summaries while retaining an explicit private namespace boundary."""

    computed_public = _canonical_structural_scientific_public(computed, "computed private baseline")
    expected_public = _canonical_structural_scientific_public(expected, "reused step9000 baseline")
    _require(computed_public == expected_public, "private baseline scientific structural statistics do not match reused baseline")


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _write_source_snapshot(root: Path, target: Path, source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the named executable sources before the first persistent update."""

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
        shutil.copy2(source, destination)
        entries.append({"path": relative_path.as_posix(), "sha256": file_sha256(source), "bytes": source.stat().st_size})
    snapshot = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "role": "reconstruction_source_content",
        "captured_at": time.time(),
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "execution_source_semantics_id": source_manifest.get("execution_source_semantics_id"),
        "files": entries,
        "content_note": "Content captured at run start; the identity fields and files are both required for a future compatibility decision.",
    }
    _write_json(target / "source_snapshot_manifest.json", snapshot, "joint continuation source snapshot manifest")
    return snapshot


def _validate_source_snapshot(
    snapshot: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    snapshot_root: Path,
) -> None:
    """Require one intact captured file for every executable Python dependency."""

    _require(snapshot.get("schema") == SOURCE_SNAPSHOT_SCHEMA and snapshot.get("role") == "reconstruction_source_content", "resume source content snapshot schema is invalid")
    _require(snapshot.get("source_manifest_id") == source_manifest.get("source_manifest_id") and snapshot.get("execution_source_semantics_id") == source_manifest.get("execution_source_semantics_id"), "resume source content identity changed", ResumeCompatibilityError)
    manifest_paths = source_manifest.get("relevant_source_paths")
    _require(isinstance(manifest_paths, list), "resume source manifest paths are invalid", ResumeCompatibilityError)
    expected_paths = {Path(str(path)).as_posix() for path in manifest_paths if Path(str(path)).suffix == ".py"}
    snapshot_files = snapshot.get("files")
    _require(isinstance(snapshot_files, list) and bool(snapshot_files), "resume source content snapshot is empty")
    observed_paths: set[str] = set()
    for item in snapshot_files:
        entry = _mapping(item, "resume source snapshot entry")
        relative = Path(str(entry.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".py", "resume source snapshot path is invalid", ResumeCompatibilityError)
        normalized = relative.as_posix()
        _require(normalized not in observed_paths, f"resume source snapshot path is duplicated: {normalized}", ResumeCompatibilityError)
        observed_paths.add(normalized)
        captured = snapshot_root / relative
        _require(captured.is_file(), f"resume source snapshot file is missing: {captured}", ResumeCompatibilityError)
        _require(file_sha256(captured) == entry.get("sha256"), f"resume source snapshot hash changed: {relative}", ResumeCompatibilityError)
    _require(observed_paths == expected_paths, "resume source content snapshot is incomplete", ResumeCompatibilityError)


def _cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "available": False}
    return {
        "device": str(device),
        "available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _validate_report_payload(report: Mapping[str, Any], *, target: Path | None = None) -> None:
    """Validate the complete report shape before exposing a completed state."""

    _require(report.get("schema") == REPORT_SCHEMA, "report schema is invalid")
    _require(report.get("task_id") == TASK_ID, "report task identity changed")
    _require(report.get("result") == "VALID_COMPLETED" and report.get("status") == "VALID_COMPLETED", "report is not a valid completed result")
    stage1 = _mapping(report.get("stage1"), "report Stage-1")
    stage2 = _mapping(report.get("stage2"), "report Stage-2")
    for stage_name, stage in (("Stage-1", stage1), ("Stage-2", stage2)):
        development = _mapping(stage.get("development"), f"report {stage_name} development")
        _require(bool(development), f"report {stage_name} development is empty")
        for step, record in development.items():
            item = _mapping(record, f"report {stage_name} development {step}")
            rows = item.get("rows")
            _require(isinstance(rows, int) and rows > 0, f"report {stage_name} development {step} has no sample count")
        transfer = _mapping(stage.get("S1_transfer"), f"report {stage_name} S1 transfer")
        for step, record in transfer.items():
            item = _mapping(record, f"report {stage_name} S1 transfer {step}")
            rows = item.get("rows")
            _require(isinstance(rows, int) and rows > 0, f"report {stage_name} S1 transfer {step} has no sample count")
    counts = _mapping(report.get("evaluation_sample_counts"), "report evaluation sample counts")
    _require(isinstance(counts.get("DEV"), Mapping) and isinstance(counts.get("S1"), Mapping), "report evaluation sample counts are incomplete")
    _require(all(isinstance(value, int) and value > 0 for value in counts["DEV"].values()), "DEV sample counts are invalid")
    _require(all(isinstance(value, int) and value > 0 for value in counts["S1"].values()), "S1 sample counts are invalid")
    if target is not None:
        final_step = int(stage2.get("final_step") if stage2.get("executed") else STAGE1_END if "9300" in stage1.get("development", {}) else START_STEP)
        _require((target / f"scientific_step_{final_step:06d}.pt").is_file(), "report endpoint checkpoint is missing")


def _persist_report(
    *,
    target: Path,
    report: Mapping[str, Any],
    heartbeat: HeartbeatWriter,
    control: ResearchRunControl,
    step: int,
    classification: str,
) -> None:
    finalizing_started = time.perf_counter()
    active_state = RUN_SCIENTIFIC_ACTIVE if control.state == RUN_SCIENTIFIC_ACTIVE else RUNNING
    heartbeat.update(state=active_state, phase="FINALIZING", step=step, latest_losses={"report": "PERSISTING"}, force=True)
    try:
        _validate_report_payload(report, target=target)
        if isinstance(report, dict):
            report_timing = report.get("timing")
            if isinstance(report_timing, dict):
                stages = report_timing.get("stages")
                if isinstance(stages, dict):
                    stages["finalizing"] = {"seconds": time.perf_counter() - finalizing_started}
        _write_json(target / "report.json", report, "joint structural semantic continuation report")
        persisted = json.loads((target / "report.json").read_text(encoding="utf-8"))
        _validate_report_payload(_mapping(persisted, "persisted report"), target=target)
    except (OSError, UnicodeError, TypeError, ValueError, JointContinuationError) as error:
        heartbeat.update(state=active_state, phase="FINALIZING", step=step, latest_losses={"report": "WRITE_FAILED"}, force=True)
        raise JointContinuationError("report persistence failed; completion state was not written") from error
    if report.get("outer_gate_passed") is False:
        control.valid_failed()
        final_state = RUN_VALID_FAILED
    else:
        control.complete()
        final_state = RUN_COMPLETED
    heartbeat.update(state=final_state, phase="COMPLETED", step=step, latest_losses={"classification": classification}, force=True)


def _generator(consumer: Any) -> torch.nn.Parameter:
    return semantic._generator(consumer)


def _runtime_state(consumer: Any) -> dict[str, torch.Tensor]:
    return semantic._runtime_state(consumer)


def _frozen_runtime_state(consumer: Any) -> dict[str, torch.Tensor]:
    return semantic._frozen_runtime_state(consumer)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return semantic._state_digest(state)


def _finite(value: torch.Tensor, name: str) -> None:
    semantic._finite(value, name)


def _norm(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    return semantic._norm(value, name, allow_zero=allow_zero)


def _rms(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    return semantic._rms(value, name, allow_zero=allow_zero)


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return semantic._dot(left, right)


def _rng_snapshot() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return semantic._rng_snapshot()


def _rng_equal(left: tuple[torch.Tensor, list[torch.Tensor]], right: tuple[torch.Tensor, list[torch.Tensor]]) -> bool:
    return semantic._rng_equal(left, right)


def _restore_rng(payload: Mapping[str, Any]) -> None:
    semantic._restore_rng(payload)


class ContinuationSampler:
    """Accepted step-9000 per-class order continued without reshuffle."""

    SCHEMA = "joint_structural_semantic_continuation_sampler.v1"

    def __init__(self, state: Mapping[str, Any], *, cursor: int = START_STEP) -> None:
        _require(state.get("seed") == SCHEDULE_SEED, "sampler seed changed")
        values = state.get("permutations")
        _require(isinstance(values, list) and len(values) == 10, "sampler permutations are malformed")
        self.seed = int(state["seed"])
        self.cursor = int(cursor)
        self.steps = TOTAL_STEPS
        self.permutations: list[torch.Tensor] = []
        for digit, value in enumerate(values):
            permutation = _tensor(value, f"sampler permutation {digit}").cpu().clone()
            _require(permutation.dtype == torch.long and permutation.ndim == 1 and permutation.numel() > 0, f"sampler permutation {digit} is empty")
            self.permutations.append(permutation)
        _require(0 <= self.cursor <= self.steps, "sampler cursor is invalid")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.cursor < self.steps, "continuation sampler exhausted")
        indices = torch.stack([permutation[self.cursor % permutation.numel()] for permutation in self.permutations])
        self.cursor += 1
        return indices, torch.arange(10, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "seed": self.seed, "steps": self.steps, "cursor": self.cursor, "wraparound": True, "permutations": [value.clone() for value in self.permutations]}


def _load_surrogate_normalization(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / SURROGATE_NORMALIZATION
    payload = _mapping(safe_torch_load(path, expected_sha256=SURROGATE_NORMALIZATION_SHA256, map_location="cpu"), "surrogate normalization")
    mean = _tensor(payload.get("mean"), "surrogate mean").to(torch.float64)
    scale = _tensor(payload.get("scale"), "surrogate scale").to(torch.float64)
    _require(mean.shape == scale.shape == (structural.STRUCTURAL_DIMENSION,), "surrogate normalization geometry changed")
    _finite(mean, "surrogate mean")
    _finite(scale, "surrogate scale")
    _require(bool((scale > 0.0).all().item()), "surrogate normalization scale is invalid")
    _require(payload.get("train_only") is True, "surrogate normalization is not TRAIN-only")
    return mean, scale


def _validate_file_sha256(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(file_sha256(path).lower() == expected.lower(), f"{label} identity changed")


def _load_reused_json(root: Path, relative_path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = root / relative_path
    _validate_file_sha256(path, expected_sha256, label)
    return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), label))


def _load_reused_step9000_baseline(root: Path, accepted_checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable public step-9000 baseline without recomputing it."""

    dev = _load_reused_json(root, REUSED_STEP9000_DEV, REUSED_STEP9000_DEV_SHA256, "reused step9000 DEV baseline")
    structural_dev = _load_reused_json(root, REUSED_STRUCTURAL_STEP9000_DEV, REUSED_STRUCTURAL_STEP9000_DEV_SHA256, "reused structural step9000 DEV baseline")
    s1 = _load_reused_json(root, REUSED_STEP9000_S1, REUSED_STEP9000_S1_SHA256, "reused step9000 S1 baseline")
    _require(dev.get("step") == START_STEP and dev.get("rows") == 5000, "reused step9000 DEV baseline geometry changed")
    _require(s1.get("step") == START_STEP and s1.get("rows") == 2000, "reused step9000 S1 baseline geometry changed")
    _require(dev.get("schema") == DEV_SCHEMA, "reused step9000 DEV schema changed")
    _require(s1.get("family") == "S1" and s1.get("requested_class_ground_truth") is True, "reused step9000 S1 contract changed")
    for key in ("step", "rows", "L_struct", "L_rec", "L_op", "raw_cmos", "structural"):
        _require(dev.get(key) == structural_dev.get(key), f"reused step9000 baseline common field changed: {key}")

    checkpoint_path = root / REUSED_STEP9000_CHECKPOINT
    _validate_file_sha256(checkpoint_path, REUSED_STEP9000_CHECKPOINT_SHA256, "reused step9000 checkpoint")
    reused_checkpoint = _mapping(safe_torch_load(checkpoint_path, expected_sha256=REUSED_STEP9000_CHECKPOINT_SHA256.lower(), map_location="cpu"), "reused step9000 checkpoint")
    _require(reused_checkpoint.get("schema") == "joint_structural_semantic_continuation_checkpoint.v1", "reused step9000 checkpoint schema changed")
    _require(reused_checkpoint.get("completed_updates") == START_STEP and reused_checkpoint.get("scientific_artifact") is True and reused_checkpoint.get("operational_artifact") is False, "reused step9000 checkpoint role changed")
    for name in ("generator_state", "encoder_state"):
        _require(torch.equal(_tensor(reused_checkpoint.get(name), f"reused {name}"), _tensor(accepted_checkpoint.get(name), f"accepted {name}")), f"reused step9000 {name} differs from accepted start")
    reused_optimizer = _mapping(reused_checkpoint.get("optimizer_configuration"), "reused step9000 optimizer configuration")
    accepted_optimizer = _mapping(accepted_checkpoint.get("optimizer_configuration"), "accepted step9000 optimizer configuration")
    _require(reused_optimizer == accepted_optimizer, "reused step9000 optimizer configuration differs from accepted start")

    execution_manifest_path = root / REUSED_STEP9000_EXECUTION_MANIFEST
    source_manifest_path = root / REUSED_STEP9000_SOURCE_MANIFEST
    _validate_file_sha256(execution_manifest_path, REUSED_STEP9000_EXECUTION_MANIFEST_SHA256, "reused step9000 execution manifest")
    _validate_file_sha256(source_manifest_path, REUSED_STEP9000_SOURCE_MANIFEST_SHA256, "reused step9000 source manifest")
    execution_manifest = _mapping(json.loads(execution_manifest_path.read_text(encoding="utf-8")), "reused step9000 execution manifest")
    source_manifest = _mapping(json.loads(source_manifest_path.read_text(encoding="utf-8")), "reused step9000 source manifest")
    _require(execution_manifest.get("task_id") == TASK_ID, "reused step9000 execution task identity changed")
    input_authorities = _mapping(execution_manifest.get("input_authorities"), "reused step9000 input authorities")
    _require(input_authorities.get("step9000") == START_CHECKPOINT_SHA256, "reused step9000 execution start authority changed")
    _require(source_manifest.get("task_id") == TASK_ID and source_manifest.get("source_manifest_id"), "reused step9000 source manifest is incomplete")
    return {
        "dev": dev,
        "s1": s1,
        "structural_dev": structural_dev,
        "checkpoint": dict(reused_checkpoint),
        "execution_manifest": dict(execution_manifest),
        "source_manifest": dict(source_manifest),
        "public_authority": {
            "dev": {"path": REUSED_STEP9000_DEV.as_posix(), "sha256": REUSED_STEP9000_DEV_SHA256.lower()},
            "s1": {"path": REUSED_STEP9000_S1.as_posix(), "sha256": REUSED_STEP9000_S1_SHA256.lower()},
            "checkpoint": {"path": REUSED_STEP9000_CHECKPOINT.as_posix(), "sha256": REUSED_STEP9000_CHECKPOINT_SHA256.lower()},
            "execution_manifest": {"path": REUSED_STEP9000_EXECUTION_MANIFEST.as_posix(), "sha256": REUSED_STEP9000_EXECUTION_MANIFEST_SHA256.lower()},
            "source_manifest": {"path": REUSED_STEP9000_SOURCE_MANIFEST.as_posix(), "sha256": REUSED_STEP9000_SOURCE_MANIFEST_SHA256.lower()},
        },
    }


def _validate_start_checkpoint(root: Path, checkpoint: Mapping[str, Any]) -> None:
    _require(checkpoint.get("schema") == "evaluator_agnostic_structural_image_alignment_checkpoint.v1", "step9000 checkpoint schema changed")
    _require(checkpoint.get("completed_updates") == START_STEP, "step9000 scientific cursor changed")
    _require(checkpoint.get("trainable_parameter_names") == ["generator.phase"], "step9000 trainable boundary changed")
    _require(checkpoint.get("scientific_artifact") is True and checkpoint.get("operational_artifact") is False, "step9000 artifact role changed")
    for name in ("encoder_state", "generator_state"):
        value = _tensor(checkpoint.get(name), name)
        _require(tuple(value.shape) == (2, 1, 100, 100) and value.dtype == torch.float32, f"{name} geometry changed")
        _finite(value, name)
    optimizer_configuration = _mapping(checkpoint.get("optimizer_configuration"), "step9000 optimizer configuration")
    _require(float(optimizer_configuration.get("internal_step", -1.0)) == 9300.0, "step9000 Adam internal step changed")
    _require(optimizer_configuration.get("parameter_names") == ["generator.phase"], "step9000 Adam parameter boundary changed")
    optimizer_state = _mapping(checkpoint.get("optimizer_state"), "step9000 optimizer state")
    _require(len(optimizer_state.get("state", {})) == 1, "step9000 Adam moments are missing")
    sampler_state = _mapping(checkpoint.get("sampler_state"), "step9000 sampler state")
    _require(sampler_state.get("seed") == SCHEDULE_SEED and sampler_state.get("cursor") == START_STEP, "step9000 sampler identity changed")
    _require(sampler_state.get("steps") == START_STEP and sampler_state.get("wraparound") is True, "step9000 sampler length changed")
    rng_state = _mapping(checkpoint.get("rng_state"), "step9000 RNG state")
    _tensor(rng_state.get("cpu"), "step9000 CPU RNG")
    _require(isinstance(rng_state.get("cuda"), list) and len(rng_state["cuda"]) > 0, "step9000 CUDA RNG state missing")


def _load_authorities(root: Path) -> dict[str, Any]:
    authorities = structural._load_authorities(root)
    current_git = git_identity(root)
    start_git = dict(authorities["git_provenance"]["consumer"])
    start_git.update({"worktree_dirty": current_git["worktree_dirty"], "status_sha256": current_git["status_sha256"]})
    checkpoint_path = root / START_CHECKPOINT
    checkpoint = _mapping(safe_torch_load(checkpoint_path, expected_sha256=START_CHECKPOINT_SHA256, map_location="cpu"), "step9000 checkpoint")
    _validate_start_checkpoint(root, checkpoint)
    reused_baseline = _load_reused_step9000_baseline(root, checkpoint)

    attr_path = root / ATTRIBUTION_REPORT
    attr_report = _mapping(json.loads(attr_path.read_text(encoding="utf-8")), "step9000 attribution report")
    _require(attr_report.get("result") == "VALID_COMPLETED", "step9000 attribution result changed")
    _require(attr_report.get("outer_gate") == "STEP9000_STRUCTURAL_CLASSIFIER_SEMANTIC_TRADEOFF_ATTRIBUTED", "step9000 attribution gate changed")
    _require(attr_report.get("classification") == "STRUCTURAL_VS_CLASSIFIER_OBJECTIVE_CONFLICT_SUPPORTED", "step9000 attribution classification changed")
    direction_details = _mapping(_mapping(attr_report.get("stage4_joint_direction"), "step9000 joint direction").get("direction_details"), "step9000 direction details")
    _require(float(direction_details.get("cos_structural_v_SIC", 0.0)) > 0.0, "accepted structural/SIC joint direction sign changed")
    attribution_gradients = safe_torch_load(root / ATTRIBUTION_GRADIENTS, expected_sha256=ATTRIBUTION_GRADIENTS_SHA256, map_location="cpu")
    _require(isinstance(attribution_gradients, Mapping) and "9000" in _mapping(attribution_gradients.get("key_steps"), "attribution key steps"), "step9000 attribution gradients are incomplete")
    mean, scale = _load_surrogate_normalization(root)
    authorities.update(
        {
            "local_state": start_git,
            "step9000": dict(checkpoint),
            "step9000_attribution_report": dict(attr_report),
            "step9000_attribution_gradients": attribution_gradients,
            "reused_step9000_baseline": reused_baseline,
            "surrogate_mean": mean,
            "surrogate_scale": scale,
            "authority_hashes": {
                **authorities["authority_hashes"],
                "step9000": START_CHECKPOINT_SHA256,
                "step9000_attribution_report": ATTRIBUTION_REPORT_SHA256,
                "step9000_attribution_gradients": ATTRIBUTION_GRADIENTS_SHA256,
                "accepted_structural_report": ACCEPTED_STRUCTURAL_REPORT_SHA256,
                "surrogate_normalization": SURROGATE_NORMALIZATION_SHA256,
                "structural_reference": STRUCTURAL_REFERENCE_SHA256,
                "s1_winner": S1_WINNER_SHA256,
                "reused_step9000_dev": REUSED_STEP9000_DEV_SHA256.lower(),
                "reused_step9000_s1": REUSED_STEP9000_S1_SHA256.lower(),
                "reused_step9000_checkpoint": REUSED_STEP9000_CHECKPOINT_SHA256.lower(),
                "reused_step9000_execution_manifest": REUSED_STEP9000_EXECUTION_MANIFEST_SHA256.lower(),
                "reused_step9000_source_manifest": REUSED_STEP9000_SOURCE_MANIFEST_SHA256.lower(),
            },
        }
    )
    return authorities


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any, Any, Any]:
    consumer, smooth, independent, ec, ed = semantic._load_models(root, authorities, device)
    _require(all(not parameter.requires_grad for model in (smooth, independent, ec, ed) for parameter in model.parameters()), "frozen evaluator boundary changed")
    return consumer, smooth, independent, ec, ed


def _restore_start_state(consumer: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, checkpoint: Mapping[str, Any]) -> None:
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    semantic._freeze_generator_only(consumer)
    optimizer.load_state_dict(copy.deepcopy(_mapping(checkpoint.get("optimizer_state"), "checkpoint optimizer state")))
    semantic.robust._validate_optimizer(consumer, optimizer)
    _require(float(optimizer.state[_generator(consumer)]["step"].item()) == 9300.0, "restored Adam internal step changed")
    sampler.cursor = START_STEP
    _restore_rng(_mapping(checkpoint.get("rng_state"), "checkpoint RNG state"))
    semantic._clear_runtime_grads(consumer)


def _optimizer(consumer: Any, checkpoint: Mapping[str, Any]) -> torch.optim.Adam:
    optimizer = torch.optim.Adam([_generator(consumer)], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    optimizer.load_state_dict(copy.deepcopy(_mapping(checkpoint.get("optimizer_state"), "stored optimizer state")))
    semantic.robust._validate_optimizer(consumer, optimizer)
    _require(float(optimizer.state[_generator(consumer)]["step"].item()) == 9300.0, "stored Adam internal step changed")
    return optimizer


def _compose_joint_direction(g_struct: torch.Tensor, g_s: torch.Tensor, g_i: torch.Tensor, g_c: torch.Tensor) -> dict[str, torch.Tensor | float]:
    norms = [_norm(value, name) for value, name in ((g_struct, "g_struct"), (g_s, "g_S"), (g_i, "g_I"), (g_c, "g_C"))]
    u_struct = g_struct / norms[0]
    u_sic = g_s / norms[1] + g_i / norms[2] + g_c / norms[3]
    v_joint = u_struct + u_sic / _norm(u_sic, "v_SIC")
    v_norm = _norm(v_joint, "v_joint", allow_zero=True)
    if not math.isfinite(v_norm) or v_norm <= JOINT_DEGENERACY_TOLERANCE:
        raise JointDirectionDegenerated("joint direction norm is nonfinite or <= 1e-12")
    q_joint = v_joint / _rms(v_joint, "v_joint")
    _finite(q_joint, "q_joint")
    return {"u_struct": u_struct, "v_SIC": u_sic, "u_SIC": u_sic / _norm(u_sic, "v_SIC"), "v_joint": v_joint, "q_joint": q_joint, "d_joint": -q_joint, "cos_structural_v_SIC": _dot(u_struct, u_sic) / (_norm(u_struct, "u_struct") * _norm(u_sic, "v_SIC")), "v_joint_l2_norm": v_norm}


def _semantic_batched_gradients(losses: Mapping[str, torch.Tensor], generator: torch.Tensor, *, retain_graph: bool = False) -> dict[str, torch.Tensor]:
    """Compute the three semantic VJPs in one explicitly batched autograd call."""

    semantic_losses = tuple(losses[name] for name in ("L_S", "L_I", "L_C"))
    _require(len(semantic_losses) == 3 and all(value.ndim == 0 for value in semantic_losses), "semantic loss batch geometry changed")
    basis = torch.eye(3, device=generator.device, dtype=semantic_losses[0].dtype)
    try:
        values = torch.autograd.grad(
            semantic_losses,
            generator,
            grad_outputs=tuple(basis[index] for index in range(3)),
            retain_graph=retain_graph,
            is_grads_batched=True,
        )
    except (TypeError, RuntimeError) as error:
        raise JointContinuationError("HYBRID_SEMANTIC_BATCHED_VJP is unsupported or failed; execution is fail-closed") from error
    _require(len(values) == 1, "batched semantic VJP returned an unexpected parameter count")
    stacked = values[0]
    _require(stacked.shape == (3, *generator.shape), "batched semantic VJP row mapping geometry changed")
    _finite(stacked, "batched semantic VJP")
    return {name: stacked[index] for index, name in enumerate(("g_S", "g_I", "g_C"))}


def _joint_direction(
    consumer: Any,
    smooth: Any,
    independent: Any,
    ec: Any,
    batch: torch.Tensor,
    references: Any,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    gradient_engine: str = GRADIENT_ENGINE,
) -> dict[str, Any]:
    _require(gradient_engine in {GRADIENT_ENGINE, SCALAR_REFERENCE_ENGINE}, f"unsupported gradient engine: {gradient_engine}")
    generator = _generator(consumer)
    bundle = semantic._loss_bundle(consumer, smooth, independent, ec, batch, references)
    raw_view = semantic._cmos_view(bundle["raw"])
    target_view = semantic._cmos_view(batch)
    with torch.no_grad():
        target_psi = structural._normalized_surrogate(structural._structural_surrogate(target_view), mean.to(batch.device), scale.to(batch.device))
    generated_psi = structural._normalized_surrogate(structural._structural_surrogate(raw_view), mean.to(batch.device), scale.to(batch.device))
    l_struct = (generated_psi - target_psi).square().mean()
    g_struct = torch.autograd.grad(l_struct, generator, retain_graph=True)[0]
    g_rec = torch.autograd.grad(bundle["losses"]["L_rec"], generator, retain_graph=True)[0]
    g_op = torch.autograd.grad(bundle["losses"]["L_op"], generator, retain_graph=True)[0]
    if gradient_engine == GRADIENT_ENGINE:
        semantic_gradients = _semantic_batched_gradients(bundle["losses"], generator, retain_graph=False)
        g_s, g_i, g_c = semantic_gradients["g_S"], semantic_gradients["g_I"], semantic_gradients["g_C"]
    else:
        g_s = torch.autograd.grad(bundle["losses"]["L_S"], generator, retain_graph=True)[0]
        g_i = torch.autograd.grad(bundle["losses"]["L_I"], generator, retain_graph=True)[0]
        g_c = torch.autograd.grad(bundle["losses"]["L_C"], generator)[0]
    for name, value in (("g_struct", g_struct), ("g_S", g_s), ("g_I", g_i), ("g_C", g_c), ("g_rec", g_rec), ("g_op", g_op)):
        _finite(value, name)
    directions = _compose_joint_direction(g_struct, g_s, g_i, g_c)
    g_sem = DELTA_JOINT * directions["q_joint"]
    g_rec_safe, rec_projection = semantic._project_auxiliary_gradient(BETA * g_rec, g_sem, "g_rec_w")
    g_op_safe, op_projection = semantic._project_auxiliary_gradient(GAMMA * g_op, g_sem, "g_op_w")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_sem) > 0.0, "combined gradient has no positive joint component")
    losses = dict(bundle["losses"])
    losses["L_struct"] = l_struct
    return {
        "bundle": bundle,
        "losses": losses,
        "g_struct": g_struct,
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
        "gradient_stats": {name: semantic._gradient_stats(value, name) for name, value in (("g_struct", g_struct), ("g_S", g_s), ("g_I", g_i), ("g_C", g_c), ("v_joint", directions["v_joint"]), ("g_joint_scaled", g_sem), ("g_total", g_total))},
        "evaluator_parameters_in_backward": {"E_S": True, "E_I": True, "E_C": True, "E_D": False},
        "delta_joint": DELTA_JOINT,
        "gradient_engine": gradient_engine,
    }


def _accepted_style_gradients(consumer: Any, smooth: Any, independent: Any, ec: Any, images: torch.Tensor, indices: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reproduce the accepted step-9000 gradient aggregation exactly."""
    _require(attribution.MICRO_BATCH_SIZE == ATTRIBUTION_MICRO_BATCH_SIZE, "accepted attribution micro-batch changed")
    _require(indices.numel() % ATTRIBUTION_MICRO_BATCH_SIZE == 0, "accepted calibration bank is not micro-batch divisible")
    generator = _generator(consumer)
    gradients = {name: torch.zeros_like(generator) for name in ("g_struct", "g_S", "g_I", "g_C")}
    row_count = int(indices.numel())
    for start in range(0, row_count, ATTRIBUTION_MICRO_BATCH_SIZE):
        batch_indices = indices[start : start + ATTRIBUTION_MICRO_BATCH_SIZE]
        values = attribution._gradient_batch(
            consumer,
            smooth,
            independent,
            ec,
            images.index_select(0, batch_indices).to(consumer.device),
            references,
            mean,
            scale,
        )
        weight = batch_indices.numel() / row_count
        gradients["g_struct"] += weight * values["g"]["structural"]
        gradients["g_S"] += weight * values["g"]["E_S"]
        gradients["g_I"] += weight * values["g"]["E_I"]
        gradients["g_C"] += weight * values["g"]["E_C"]
        attribution._clear_grads(consumer, smooth, independent, ec)
    for name, value in gradients.items():
        _finite(value, f"accepted-style {name}")
    return gradients


def _apply_update(
    consumer: Any,
    smooth: Any,
    independent: Any,
    ec: Any,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    references: Any,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    step: int,
    clean_generator_state: torch.Tensor,
    gradient_engine: str = GRADIENT_ENGINE,
) -> dict[str, Any]:
    semantic.robust._validate_optimizer(consumer, optimizer)
    generator = _generator(consumer)
    before = generator.detach().clone()
    frozen_before = _frozen_runtime_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    semantic._clear_runtime_grads(consumer)
    direction = _joint_direction(consumer, smooth, independent, ec, batch, references, mean, scale, gradient_engine=gradient_engine)
    generator.grad = direction["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name != "generator.phase"), "frozen runtime parameter received gradient")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    _require(_state_digest(frozen_before) == _state_digest(_frozen_runtime_state(consumer)), "frozen runtime state changed")
    update = generator.detach() - before
    _require(_norm(update, "phase update") > 0.0, "generator.phase did not update")
    semantic._clear_runtime_grads(consumer)
    return {
        "step": step,
        "L_struct": float(direction["losses"]["L_struct"].detach().cpu()),
        "L_S": float(direction["losses"]["L_S"].detach().cpu()),
        "L_I": float(direction["losses"]["L_I"].detach().cpu()),
        "L_C": float(direction["losses"]["L_C"].detach().cpu()),
        "L_rec": float(direction["losses"]["L_rec"].detach().cpu()),
        "L_op": float(direction["losses"]["L_op"].detach().cpu()),
        "gradient_stats": direction["gradient_stats"],
        "projection": direction["projection"],
        "cos_structural_v_SIC": float(direction["cos_structural_v_SIC"]),
        "v_joint_l2_norm": float(direction["v_joint_l2_norm"]),
        "phase_update_rms": _rms(update, "phase update"),
        "phase_displacement_from_step9000": _norm(generator.detach() - clean_generator_state.to(generator.device), "phase displacement"),
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase"],
        "E_D_in_backward": False,
        "gradient_engine": gradient_engine,
    }


def _evaluate_dev(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, authorities: Mapping[str, Any], reference: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, ledger: DataReadLedger, device: torch.device, *, step: int) -> dict[str, Any]:
    indices = authorities["development_indices"]
    images = authorities["images"]
    labels = authorities["labels"]
    labels_dev = labels.index_select(0, indices)
    predictions = {name: [] for name in ("E_S", "E_I", "E_C", "E_D")}
    loss_s = loss_i = loss_c = loss_struct = rec_sum = op_sum = 0.0
    raw_views: list[torch.Tensor] = []
    raw_nonconstant = raw_zero = 0
    raw_min, raw_max = float("inf"), float("-inf")
    total_elements = 0
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    ed.eval()
    try:
        with torch.inference_mode():
            for start in range(0, int(indices.numel()), LOGICAL_BATCH_SIZE):
                batch_indices = indices[start : start + LOGICAL_BATCH_SIZE]
                batch = images.index_select(0, batch_indices).to(device)
                result = consumer.runtime.forward_from_image(batch)
                raw = result.raw_intensity
                target = result.image_input.slm_intensity
                _finite(raw, "DEV raw CMOS")
                _finite(target, "DEV target image")
                raw_view = semantic._cmos_view(raw)
                target_view = semantic._cmos_view(batch)
                s, s_logits = smooth(raw_view)
                i, i_logits = independent(raw_view)
                c, c_logits = ec(raw_view)
                d_logits = ed(raw_view)
                s_t, _ = smooth(target_view)
                i_t, _ = independent(target_view)
                c_t, _ = ec(target_view)
                loss_s += float(semantic._normalized_feature_loss(s, s_t, "DEV L_S").cpu()) * int(batch_indices.numel())
                loss_i += float(semantic._normalized_feature_loss(i, i_t, "DEV L_I").cpu()) * int(batch_indices.numel())
                loss_c += float(semantic._normalized_feature_loss(c, c_t, "DEV L_C").cpu()) * int(batch_indices.numel())
                generated_psi = structural._normalized_surrogate(structural._structural_surrogate(raw_view), mean.to(device), scale.to(device))
                target_psi = structural._normalized_surrogate(structural._structural_surrogate(target_view), mean.to(device), scale.to(device))
                loss_struct += float((generated_psi - target_psi).square().mean().cpu()) * int(batch_indices.numel())
                predictions["E_S"].append(s_logits.argmax(dim=1).cpu())
                predictions["E_I"].append(i_logits.argmax(dim=1).cpu())
                predictions["E_C"].append(c_logits.argmax(dim=1).cpu())
                predictions["E_D"].append(d_logits.argmax(dim=1).cpu())
                rec_sum += float((raw - target).square().sum().cpu())
                powers = contracted_plane_powers(result)
                ratios = (powers.to(torch.float64) + 1.0e-12) / (authorities["training"]["references"].power_reference.to(device=device, dtype=torch.float64) + 1.0e-12)
                op_sum += float(torch.log(ratios).square().sum().cpu())
                flat = raw.to(torch.float64).flatten(1)
                raw_nonconstant += int((flat.std(dim=1, unbiased=False) > 0.0).sum().item())
                raw_zero += int((flat.abs().sum(dim=1) == 0.0).sum().item())
                raw_min = min(raw_min, float(raw.min().cpu()))
                raw_max = max(raw_max, float(raw.max().cpu()))
                total_elements += int(raw.numel())
                raw_views.append(raw_view.cpu())
                ledger.record("dev", int(batch_indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    all_views = torch.cat(raw_views)
    hard_source, hard_public = structural._hard_structural_source(f"real_ols_step{step}", all_views, labels_dev, reference, device)
    row_count = int(indices.numel())
    result: dict[str, Any] = {
        "schema": DEV_SCHEMA,
        "step": step,
        "rows": row_count,
        "E_S": {"architecture": "SmoothMNISTEvaluator v2.1", **semantic._class_metrics(labels_dev, torch.cat(predictions["E_S"]), label_source="canonical DEV label")},
        "E_I": {"architecture": "FrozenMNISTEvaluator v1", **semantic._class_metrics(labels_dev, torch.cat(predictions["E_I"]), label_source="canonical DEV label")},
        "E_C": {"architecture": "ThirdConfirmationEvaluator", **semantic._class_metrics(labels_dev, torch.cat(predictions["E_C"]), label_source="canonical DEV label")},
        "E_D": {"architecture": "PixelMLPConfirmationEvaluatorV1", **semantic._class_metrics(labels_dev, torch.cat(predictions["E_D"]), label_source="canonical DEV label")},
        "L_S": loss_s / row_count,
        "L_I": loss_i / row_count,
        "L_C": loss_c / row_count,
        "L_struct": loss_struct / row_count,
        "L_rec": rec_sum / float(total_elements),
        "L_op": op_sum / float(row_count * 5),
        "raw_cmos": {"finite": True, "nonconstant_fraction": float(raw_nonconstant / row_count), "zero_image_count": raw_zero, "min": raw_min, "max": raw_max},
        "structural": {key: value for key, value in hard_public.items() if not key.startswith("_")},
        "E_D_in_backward": False,
        "_labels_dev": labels_dev.cpu(),
        "_E_D_correct": torch.cat(predictions["E_D"]) == labels_dev,
        "_log_rho": hard_public["_log_rho"].to(torch.float64),
    }
    _require(result["raw_cmos"]["nonconstant_fraction"] == 1.0 and raw_zero == 0, f"DEV optical validity failed at step {step}")
    _require(all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters()), "DEV evaluator received a gradient")
    return result


def _evaluate_s1(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, authorities: Mapping[str, Any], reference: Mapping[str, Any], device: torch.device, ledger: DataReadLedger, *, step: int) -> dict[str, Any]:
    return structural._evaluate_s1(consumer, smooth, independent, ec, ed, authorities, reference, device, ledger, step=step)


def _materialize_step9000_private_baseline(consumer: Any, authorities: Mapping[str, Any], reference: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Materialize only the per-row structural vectors needed for paired bootstrap."""

    started = time.perf_counter()
    reused = _mapping(authorities.get("reused_step9000_baseline"), "reused step9000 baseline")
    public_dev = _mapping(reused.get("dev"), "reused step9000 DEV baseline")
    dev_ols = _mapping(authorities.get("dev_ols"), "real DEV OLS")
    z_dev = _tensor(dev_ols.get("z_dev"), "real DEV z_dev")
    labels_dev = _tensor(dev_ols.get("labels"), "real DEV labels").cpu().to(torch.long)
    _require(z_dev.shape == (5000, 100) and z_dev.is_complex(), "private baseline OLS geometry changed")
    _require(labels_dev.shape == (5000,), "private baseline label geometry changed")
    _require(torch.equal(labels_dev, authorities["labels"].index_select(0, authorities["development_indices"])), "private baseline labels changed")

    initial_state = _runtime_state(consumer)
    initial_rng = _rng_snapshot()
    was_training = consumer.runtime.training
    views: list[torch.Tensor] = []
    consumer.runtime.eval()
    try:
        with torch.inference_mode():
            for start in range(0, int(z_dev.shape[0]), LOGICAL_BATCH_SIZE):
                result = consumer.generated_forward(z_dev[start : start + LOGICAL_BATCH_SIZE].to(device))
                raw = result.raw_intensity
                _finite(raw, "private baseline raw CMOS")
                views.append(semantic._cmos_view(raw).cpu())
                if time.perf_counter() - started > BASELINE_PRIVATE_TIMEOUT_SECONDS:
                    raise JointContinuationError("private step9000 baseline supplement exceeded 10-minute cap")
    finally:
        consumer.runtime.train(was_training)
    all_views = torch.cat(views)
    _, hard_public = structural._hard_structural_source("real_ols_step9000_private", all_views, labels_dev, reference, device)
    expected_structural = _mapping(public_dev.get("structural"), "reused step9000 structural baseline")
    _compare_structural_scientific_public(hard_public, expected_structural)
    log_rho = _tensor(hard_public.get("_log_rho"), "private baseline log_rho").to(torch.float64).cpu()
    _require(log_rho.shape == (5000,) and bool(torch.isfinite(log_rho).all().item()), "private baseline log_rho is invalid")
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "private baseline changed frozen/runtime state")
    _require(_rng_equal(_rng_snapshot(), initial_rng), "private baseline changed RNG")
    elapsed = time.perf_counter() - started
    _require(elapsed <= BASELINE_PRIVATE_TIMEOUT_SECONDS, "private step9000 baseline supplement exceeded 10-minute cap")
    return {
        "schema": BASELINE_PRIVATE_SCHEMA,
        "step": START_STEP,
        "rows": 5000,
        "labels_dev": labels_dev,
        "log_rho": log_rho,
        "structural_signature_sha256": str(hard_public["signature_sha256"]),
        "elapsed_seconds": elapsed,
        "public_baseline": dict(reused["public_authority"]["dev"]),
        "semantic_note": "Per-row labels and true log(rho) only; public evaluators and S1 diagnostics were reused by fixed digest.",
    }


def _paired_bootstrap(labels: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    result = structural._paired_bootstrap_log_rho(labels, before, after)
    result["statistic"] = "macro mean(log_rho_step9000 - log_rho_candidate)"
    return result


def _optical_pass(*records: Mapping[str, Any]) -> dict[str, Any]:
    return structural._optical_pass(*records)


def _composition_record(direction: Mapping[str, Any], step: int, displacement: float, update_rms: float) -> dict[str, Any]:
    return {
        "step": step,
        "L_struct": float(direction["losses"]["L_struct"].detach().cpu()),
        "L_S": float(direction["losses"]["L_S"].detach().cpu()),
        "L_I": float(direction["losses"]["L_I"].detach().cpu()),
        "L_C": float(direction["losses"]["L_C"].detach().cpu()),
        "L_rec": float(direction["losses"]["L_rec"].detach().cpu()),
        "L_op": float(direction["losses"]["L_op"].detach().cpu()),
        "gradient_stats": direction["gradient_stats"],
        "projection": direction["projection"],
        "cos_structural_v_SIC": float(direction["cos_structural_v_SIC"]),
        "v_joint_l2_norm": float(direction["v_joint_l2_norm"]),
        "phase_displacement_from_step9000": displacement,
        "phase_update_rms": update_rms,
        "E_D_in_backward": False,
    }


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: ContinuationSampler,
    history: list[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    *,
    step: int,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    stage1_condition: Mapping[str, Any] | None,
    dev_records: Mapping[str, Any] | None = None,
    s1_records: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    source_snapshot: Mapping[str, Any] | None = None,
    stage1_bootstrap: Mapping[str, Any] | None = None,
    stage1_reconstruction: Mapping[str, Any] | None = None,
    stage1_optical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = optimizer.state[_generator(consumer)]
    selected_engine = str(_mapping(run_identity.config_identity, "run identity config").get("gradient_engine", GRADIENT_ENGINE))
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_ids": [AUTHORIZATION_ID],
        "completed_updates": step,
        "starting_scientific_step": START_STEP,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "checkpoint_selection": "none; every preregistered scientific checkpoint persisted",
        "scientific_artifact": True,
        "operational_artifact": False,
        "gradient_engine": selected_engine,
        "scalar_reference_gradient_engine": SCALAR_REFERENCE_ENGINE,
        "baseline_reuse": {"step": START_STEP, "public_dev_sha256": REUSED_STEP9000_DEV_SHA256.lower(), "public_s1_sha256": REUSED_STEP9000_S1_SHA256.lower(), "private_artifact": "baseline_private_step_009000.pt"},
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "starting_authority": {"path": START_CHECKPOINT.as_posix(), "sha256": START_CHECKPOINT_SHA256, "selected_step": START_STEP},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "S1", "E_S", "E_I", "E_C", "E_D", "f_struct"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_runtime_state(consumer)),
        "optimizer_mode": "STEP9000_ADAM_STATE_RESTORED_EXACTLY",
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "internal_step": float(state["step"])},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "joint_contract": {"delta_joint": DELTA_JOINT, "direction": "q_joint=(u_struct+u_SIC)/RMS(u_struct+u_SIC)", "descent": "-q_joint", "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "E_D_in_backward": False, "gradient_engine": selected_engine},
        "structural_contract": {"dimension": structural.STRUCTURAL_DIMENSION, "tau": structural.STRUCTURAL_TAU, "eps_mass": structural.STRUCTURAL_EPS_MASS, "normalization": "accepted TRAIN-only 336D surrogate normalization"},
        "auxiliary_coefficients": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
        "history_tail": list(history[-10:]),
        "training_history": [dict(record) for record in history],
        "trajectory": dict(trajectory),
        "stage1_condition": None if stage1_condition is None else dict(stage1_condition),
        "dev_records": {} if dev_records is None else dict(dev_records),
        "s1_records": {} if s1_records is None else dict(s1_records),
        "timing": {} if timing is None else dict(timing),
        "source_snapshot": {} if source_snapshot is None else dict(source_snapshot),
        "stage1_bootstrap": None if stage1_bootstrap is None else dict(stage1_bootstrap),
        "stage1_reconstruction": None if stage1_reconstruction is None else dict(stage1_reconstruction),
        "stage1_optical": None if stage1_optical is None else dict(stage1_optical),
    }


def _make_identity(source_manifest: Mapping[str, Any], attempt_name: str | None, *, gradient_engine: str | None = None) -> ResearchRunIdentity:
    selected_engine = GRADIENT_ENGINE if gradient_engine is None else str(gradient_engine)
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="joint-structural-semantic-continuation-v1",
        trajectory_id="joint-structural-semantic-continuation-trajectory-v1",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"start_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "gradient_engine": selected_engine, "scalar_reference_engine": SCALAR_REFERENCE_ENGINE, "baseline_reuse": {"dev_sha256": REUSED_STEP9000_DEV_SHA256.lower(), "s1_sha256": REUSED_STEP9000_S1_SHA256.lower()}},
        model_identity=TASK_ID,
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 9300.0, "reset": False},
        sampler_identity={"seed": SCHEDULE_SEED, "starting_cursor": START_STEP, "terminal_cursor": TOTAL_STEPS, "accepted_prefix_preserved": True, "wraparound": True},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=None if attempt_name is None else f"{TASK_ID}:{attempt_name}",
    )


def _source_manifest_for_run(root: Path, authorities: Mapping[str, Any], *, gradient_engine: str | None = None) -> dict[str, Any]:
    selected_engine = GRADIENT_ENGINE if gradient_engine is None else str(gradient_engine)
    execution_sources = (
        "scripts/research/lightgen/execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_attribute_post3000_generator_semantic_realization_ceiling_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_attribute_step9000_structural_classifier_semantic_tradeoff_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_phase_completion_repair_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1.py",
        "src/ldm_project/v2/application/research/lightgen/confirmation_evaluators.py",
        "scripts/research/lightgen/execute_lightgen_2d_evaluator_agnostic_structural_image_alignment_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_evaluator_robust_semantic_training_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_generator_semantic_controllability_attribution_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_generator_target_image_semantic_alignment_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_joint_structural_semantic_continuation_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_non_classifier_structural_semantic_confirmation_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_post3000_generation_readiness_scaling_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_real_vs_sampled_ols_semantic_manifold_gap_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_resolve_cmos_to_smooth_semantic_realization_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_semantically_conditioned_ols_sampler_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1.py",
        "scripts/research/lightgen/execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py",
        "scripts/research/lightgen/materialize_lightgen_2d_frozen_dev_ols_cmos_v1.py",
        "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
        "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        "src/ldm_project/v2/application/research/shared/research_run_control.py",
        "src/ldm_project/v2/contracts/__init__.py",
        "src/ldm_project/v2/contracts/planes.py",
        "src/ldm_project/v2/evidence/__init__.py",
        "src/ldm_project/v2/evidence/io.py",
        "src/ldm_project/v2/evidence/records.py",
        "src/ldm_project/v2/evidence/reviews.py",
        "src/ldm_project/v2/optics/d2nn.py",
        "src/ldm_project/v2/optics/devices.py",
        "src/ldm_project/v2/optics/fiber_operator.py",
        "src/ldm_project/v2/optics/fiber_pool.py",
        "src/ldm_project/v2/optics/propagation.py",
        "src/ldm_project/v2/optics/relays.py",
        "src/ldm_project/v2/student/evaluator.py",
    )
    return capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=tuple(root / path for path in execution_sources) + (root / START_CHECKPOINT, root / ATTRIBUTION_REPORT, root / ATTRIBUTION_GRADIENTS, root / SURROGATE_NORMALIZATION, root / STRUCTURAL_REFERENCE, root / S1_WINNER),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "start_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "delta_joint": DELTA_JOINT, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "gradient_engine": selected_engine, "scalar_reference_engine": SCALAR_REFERENCE_ENGINE, "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "cosine_minimum": GRADIENT_COSINE_MINIMUM}, "baseline_reuse": {"dev_sha256": REUSED_STEP9000_DEV_SHA256.lower(), "s1_sha256": REUSED_STEP9000_S1_SHA256.lower(), "checkpoint_sha256": REUSED_STEP9000_CHECKPOINT_SHA256.lower(), "execution_manifest_sha256": REUSED_STEP9000_EXECUTION_MANIFEST_SHA256.lower(), "source_manifest_sha256": REUSED_STEP9000_SOURCE_MANIFEST_SHA256.lower()}, "authority_hashes": authorities["authority_hashes"]},
    ).to_dict()


def _preflight_receipt(source_manifest: Mapping[str, Any], preflight: Mapping[str, Any]) -> PreflightReceipt:
    performance = _mapping(preflight.get("performance"), "preflight performance")
    return PreflightReceipt(
        source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        authorization_id=AUTHORIZATION_ID,
        checks={
            CORRECTNESS_INVARIANTS: {
                "gradient_equivalence": True,
                "adam_one_step_equivalence": True,
                "three_step_update_and_restore": bool(preflight.get("stored_rng_restored")),
                "negative_control": bool(_mapping(preflight.get("negative_control"), "preflight negative control").get("rejected")),
            },
            RESOURCE_ENVELOPE: {
                "status": performance.get("status"),
                "resource_ok": performance.get("resource_ok"),
                "estimated_hours": performance.get("estimated_3000_step_hours"),
                "memory_fraction": performance.get("peak_reserved_memory_fraction"),
            },
            IMPLEMENTATION_SELECTION: {
                "selected_engine": preflight.get("selected_engine", preflight.get("gradient_engine")),
                "selection_reason": performance.get("selection_reason"),
                "threshold": performance.get("criteria", {}).get("minimum_speedup_fraction"),
            },
        },
        selected_engine=str(preflight.get("selected_engine", preflight.get("gradient_engine"))),
        resource={
            "estimated_3000_step_hours": performance.get("estimated_3000_step_hours"),
            "peak_reserved_memory_fraction": performance.get("peak_reserved_memory_fraction"),
            "device_total_memory_bytes": performance.get("device_total_memory_bytes"),
        },
        failure_signature=None,
        recheck_count=0,
        created_at=time.time(),
    )


PREFLIGHT_FAILURE_STATE_SCHEMA = "joint_structural_semantic_continuation_preflight_failure_state.v1"


def _preflight_failure_state_dir(target: Path) -> Path:
    # This is an operational, per-target retry budget.  It is deliberately a
    # sibling run directory, never a formal attempt or a global evidence store.
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
        if isinstance(key, str) and key:
            try:
                count = int(payload.get("count", 0))
            except (TypeError, ValueError):
                continue
            counts[key] = max(counts.get(key, 0), count)
    return counts


def _classify_preflight_failure(error: BaseException) -> tuple[str, str]:
    message = str(error)
    if "PREFLIGHT_BLOCKED(resource)" in message:
        return "resource", "absolute-resource-envelope"
    if "differs between gradient engines" in message or "cosine is below strict preflight threshold" in message:
        return "correctness", "gradient-equivalence"
    if "negative-control" in message or "negative-control update" in message:
        return "correctness", "negative-control"
    if "restore" in message.lower():
        return "correctness", "state-restoration"
    return "preflight", f"{type(error).__name__}:{message.splitlines()[0][:160]}"


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
        "joint continuation preflight failure state",
    )


def _preflight_failure_control(target: Path) -> ResearchRunControl:
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id="joint-structural-semantic-continuation-v1:preflight",
        trajectory_id="joint-structural-semantic-continuation-trajectory-v1:preflight",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"role": "operational_preflight_retry_budget"},
        model_identity=TASK_ID,
        optimizer_identity={},
        sampler_identity={},
        execution_source_semantics_id=f"{TASK_ID}:preflight",
    )
    return ResearchRunControl(identity, AUTHORIZATION_ID, preflight_failure_counts=_load_preflight_failure_counts(target))


def _validate_resume_checkpoint(root: Path, checkpoint_path: Path, authorities: Mapping[str, Any]) -> dict[str, Any]:
    _require(checkpoint_path.is_file(), f"resume checkpoint is missing: {checkpoint_path}")
    _require(checkpoint_path.parent.is_dir(), "resume checkpoint parent is missing")
    wrapper = _mapping(safe_torch_load(checkpoint_path, map_location="cpu"), "resume checkpoint")
    _require(wrapper.get("schema") == "research_recovery_checkpoint.v1", "resume checkpoint wrapper schema is unsupported")
    _require(wrapper.get("artifact_kind") == "OPERATIONAL_RECOVERY_CHECKPOINT", "resume checkpoint is not operational")
    _require(wrapper.get("authorization_id") == AUTHORIZATION_ID, "resume authorization identity changed", ResumeCompatibilityError)
    stored_identity = ResearchRunIdentity.from_mapping(_mapping(wrapper.get("run_identity"), "resume run identity"))
    _require(stored_identity.task_id == TASK_ID and stored_identity.run_id == "joint-structural-semantic-continuation-v1", "resume task identity changed")
    _require(stored_identity.trajectory_id == "joint-structural-semantic-continuation-trajectory-v1", "resume trajectory identity changed")
    stored_config = _mapping(stored_identity.config_identity, "resume config identity")
    stored_engine = str(stored_config.get("gradient_engine", GRADIENT_ENGINE))
    _require(stored_engine in {SCALAR_REFERENCE_ENGINE, GRADIENT_ENGINE}, "resume gradient engine is unsupported", ResumeCompatibilityError)
    current_manifest = (
        _source_manifest_for_run(root, authorities)
        if stored_engine == GRADIENT_ENGINE
        else _source_manifest_for_run(root, authorities, gradient_engine=stored_engine)
    )
    current_identity = ResearchRunIdentity(
        task_id=stored_identity.task_id,
        run_id=stored_identity.run_id,
        trajectory_id=stored_identity.trajectory_id,
        semantic_authority_id=stored_identity.semantic_authority_id,
        data_role=stored_identity.data_role,
        config_identity=stored_identity.config_identity,
        model_identity=stored_identity.model_identity,
        optimizer_identity=stored_identity.optimizer_identity,
        sampler_identity=stored_identity.sampler_identity,
        execution_source_semantics_id=str(current_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(current_manifest["source_manifest_id"]),
        scientific_attempt_id=stored_identity.scientific_attempt_id,
    )
    try:
        check_resume_compatibility(stored_identity, current_identity)
    except ResumeCompatibilityError as error:
        raise JointContinuationError("resume execution dependency drifted; explicit checkpoint rejected") from error
    payload = _mapping(wrapper.get("payload"), "resume checkpoint payload")
    step = int(wrapper.get("step", -1))
    _require(START_STEP < step < TOTAL_STEPS, "resume checkpoint cursor is outside the continuation interval")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA, "resume payload schema is unsupported")
    _require(payload.get("completed_updates") == step, "resume payload cursor disagrees with wrapper")
    payload_manifest = _mapping(payload.get("source_manifest"), "resume source manifest")
    _require(payload_manifest == current_manifest, "resume source manifest is incomplete", ResumeCompatibilityError)
    _require(wrapper.get("source_manifest_id") == payload_manifest.get("source_manifest_id"), "resume source manifest identity changed", ResumeCompatibilityError)
    payload_identity = ResearchRunIdentity.from_mapping(_mapping(payload.get("run_identity"), "resume payload run identity"))
    check_resume_compatibility(stored_identity, payload_identity)
    _require(payload.get("scientific_artifact") is False and payload.get("operational_artifact") is True, "resume payload artifact role is invalid")
    _require(payload.get("artifact_kind") == "OPERATIONAL_RECOVERY_CHECKPOINT", "resume payload kind is invalid")
    _require(payload.get("trainable_parameter_names") == ["generator.phase"], "resume trainable boundary changed")
    _require(isinstance(payload.get("optimizer_state"), Mapping), "resume Adam state is missing")
    optimizer_configuration = _mapping(payload.get("optimizer_configuration"), "resume optimizer configuration")
    _require(float(optimizer_configuration.get("internal_step", -1.0)) == float(9300 + step - START_STEP), "resume Adam cursor is inconsistent")
    _require(isinstance(payload.get("rng_state"), Mapping), "resume RNG state is missing")
    sampler_state = _mapping(payload.get("sampler_state"), "resume sampler state")
    _require(sampler_state.get("schema") == ContinuationSampler.SCHEMA and sampler_state.get("cursor") == step and sampler_state.get("seed") == SCHEDULE_SEED and sampler_state.get("steps") == TOTAL_STEPS and sampler_state.get("wraparound") is True, "resume sampler cursor or seed changed")
    history = payload.get("training_history")
    _require(isinstance(history, list) and len(history) == step - START_STEP, "resume training history is incomplete")
    for expected_step, record in enumerate(history, start=START_STEP + 1):
        _require(isinstance(record, Mapping) and int(record.get("step", -1)) == expected_step, f"resume history has a gap at step {expected_step}")
    trajectory = payload.get("trajectory")
    _require(isinstance(trajectory, Mapping) and str(START_STEP) in trajectory, "resume trajectory is incomplete")
    dev_records = payload.get("dev_records")
    s1_records = payload.get("s1_records")
    _require(isinstance(dev_records, Mapping) and str(START_STEP) in dev_records, "resume DEV endpoint history is missing")
    _require(isinstance(s1_records, Mapping) and str(START_STEP) in s1_records, "resume S1 endpoint history is missing")
    if step > STAGE1_END:
        _require(str(STAGE1_END) in dev_records and str(STAGE1_END) in s1_records, "resume Stage-1 endpoint history is missing")
    snapshot = _mapping(payload.get("source_snapshot"), "resume source content snapshot")
    _validate_source_snapshot(snapshot, payload_manifest, checkpoint_path.parent / "source_snapshot")
    timing = _mapping(payload.get("timing"), "resume timing history")
    _require(timing.get("schema") == TIMING_SCHEMA and isinstance(timing.get("updates"), list), "resume timing history is invalid")
    _require(isinstance(payload.get("stage1_condition"), Mapping), "resume Stage-1 condition is missing")
    if step > STAGE1_END:
        _require(isinstance(payload.get("stage1_bootstrap"), Mapping) and isinstance(payload.get("stage1_reconstruction"), Mapping) and isinstance(payload.get("stage1_optical"), Mapping), "resume Stage-1 evidence is missing")
        _require(_mapping(payload.get("stage1_condition"), "resume Stage-1 condition").get("passed") is True, "resume after Stage-1 requires a passed continuation condition")
    preflight_path = checkpoint_path.parent / "preflight.json"
    _require(preflight_path.is_file(), "resume preflight receipt is missing", ResumeCompatibilityError)
    try:
        preflight_document = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise JointContinuationError("resume preflight receipt cannot be consumed") from error
    preflight_receipt = PreflightReceipt.from_mapping(_mapping(preflight_document.get("preflight_receipt"), "resume preflight receipt"))
    _require(preflight_receipt.authorization_id == AUTHORIZATION_ID, "resume preflight authorization changed", ResumeCompatibilityError)
    _require(preflight_receipt.source_semantics_id == current_manifest["execution_source_semantics_id"], "resume preflight source semantics changed", ResumeCompatibilityError)
    return {"wrapper": wrapper, "payload": payload, "step": step, "identity": stored_identity, "source_manifest": current_manifest, "current_manifest": current_manifest, "preflight_receipt": preflight_receipt}


def _resume_preflight(checkpoint_path: Path, resume_state: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(resume_state["payload"], "resume preflight payload")
    completed_step = int(resume_state["step"])
    dev_records = _mapping(payload.get("dev_records"), "resume DEV records")
    endpoint_steps = [str(completed_step)]
    if completed_step >= STAGE1_END:
        endpoint_steps.append(str(STAGE1_END))
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "mode": "EXPLICIT_RESUME",
        "checkpoint": checkpoint_path.as_posix(),
        "completed_step": completed_step,
        "restored": ["generator", "encoder", "Adam", "RNG", "sampler", "training_history", "trajectory", "DEV", "S1", "Stage1 condition", "timing"],
        "endpoint_evaluation_required": any(step not in dev_records for step in endpoint_steps),
        "automatic_latest_resolution": False,
        "selected_engine": str(
            _mapping(resume_state.get("identity").config_identity, "resume config identity").get("gradient_engine", GRADIENT_ENGINE)
            if resume_state.get("identity") is not None
            else GRADIENT_ENGINE
        ),
    }


def run_diagnostic_preflight(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_DIAGNOSTIC_OUTPUT_DIR) -> dict[str, Any]:
    """Run the real three-update CUDA path in a fresh diagnostic-only run.

    The temporary updates are deliberately isolated from scientific outputs.  The
    routine exercises the same model/update/evaluation/checkpoint interfaces as a
    formal continuation, then restores the saved state and leaves an explicit
    diagnostic report.  It never writes ``report.json`` or ``COMPLETED``.
    """

    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "diagnostic output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final diagnostic namespace is forbidden")
    _require(not target.exists(), f"diagnostic output already exists: {target}")
    authorities = _load_authorities(root)
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "diagnostic preflight requires CUDA")
    device = torch.device("cuda:0")
    cuda_runtime = semantic.robust._prepare_cuda_runtime()
    consumer, smooth, independent, ec, ed = _load_models(root, authorities, device)
    checkpoint = authorities["step9000"]
    optimizer = _optimizer(consumer, checkpoint)
    sampler = ContinuationSampler(checkpoint["sampler_state"], cursor=START_STEP)
    _restore_start_state(consumer, optimizer, sampler, checkpoint)
    mean, scale = authorities["surrogate_mean"], authorities["surrogate_scale"]
    try:
        preflight = _preflight(consumer, smooth, independent, ec, ed, optimizer, sampler, authorities, mean, scale, device)
    except JointContinuationError as error:
        component, signature = _classify_preflight_failure(error)
        retry_control = _preflight_failure_control(target)
        decision = retry_control.record_preflight_failure(component, signature)
        _record_persisted_preflight_failure(target, retry_control, component, signature, decision)
        raise
    _require(preflight["status"] == "READY", "diagnostic preflight did not reach READY")
    gradient_engine = str(preflight["gradient_engine"])
    source_manifest = _source_manifest_for_run(root, authorities, gradient_engine=gradient_engine)
    receipt = _preflight_receipt(source_manifest, preflight)
    target.mkdir(parents=True, exist_ok=False)
    source_snapshot = _write_source_snapshot(root, target, source_manifest)
    identity = _make_identity(source_manifest, None, gradient_engine=gradient_engine)
    control = ResearchRunControl(identity, AUTHORIZATION_ID)
    control.mark_preflight_ready(receipt)
    control.start_execution()
    # Diagnostic work is operational only; it must not consume a scientific
    # attempt identity or create a formal scientific artifact.
    effective_identity = identity
    _write_json(target / "source_manifest.json", source_manifest, "joint continuation diagnostic source manifest")
    _write_json(target / "preflight.json", {**preflight, "preflight_receipt": receipt.to_dict()}, "joint continuation diagnostic preflight")
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "joint_structural_semantic_continuation_diagnostic_execution_manifest.v1",
            "task_id": TASK_ID,
            "diagnostic_only": True,
            "source_manifest": source_manifest,
            "source_snapshot": source_snapshot,
            "run_identity": effective_identity.to_dict(),
            "input_authorities": authorities["authority_hashes"],
            "cuda_runtime": cuda_runtime,
            "protected": 0,
            "test": 0,
            "sealed": 0,
        },
        "joint continuation diagnostic execution manifest",
    )
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=TOTAL_STEPS, interval_seconds=10.0)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=START_STEP, latest_losses={"preflight": "READY", "diagnostic_only": True}, force=True)
    heartbeat.start_liveness(interval_seconds=1.0)
    ledger = DataReadLedger()
    timing: dict[str, Any] = {"schema": TIMING_SCHEMA, "stages": {}, "updates": []}
    history: list[dict[str, Any]] = []
    trajectory: dict[str, dict[str, Any]] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    s1_records: dict[str, dict[str, Any]] = {}
    clean_generator_state = checkpoint["generator_state"].to(device)
    recovery: RecoveryCheckpointManager | None = None
    try:
        preparing_started = time.perf_counter()
        fixed_indices = torch.stack([value[START_STEP % value.numel()] for value in sampler.permutations])
        fixed_batch = authorities["images"].index_select(0, fixed_indices).to(device)
        composition = _joint_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"], mean, scale, gradient_engine=gradient_engine)
        trajectory[str(START_STEP)] = _composition_record(composition, START_STEP, 0.0, 0.0)
        semantic._clear_runtime_grads(consumer)
        semantic._clear_evaluator_grads(smooth, independent, ec, ed)
        dev_records[str(START_STEP)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], mean, scale, ledger, device, step=START_STEP)
        s1_records[str(START_STEP)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=START_STEP)
        timing["stages"]["preparing"] = {"seconds": time.perf_counter() - preparing_started, "cuda_memory": _cuda_memory(device)}

        for update in range(3):
            indices, labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"diagnostic TRAIN labels changed at update {update + 1}")
            update_started = time.perf_counter()
            trace = _apply_update(consumer, smooth, independent, ec, optimizer, authorities["images"].index_select(0, indices).to(device), authorities["training"]["references"], mean, scale, step=START_STEP + update + 1, clean_generator_state=clean_generator_state, gradient_engine=gradient_engine)
            trace["timing"] = {"seconds": time.perf_counter() - update_started, "cuda_memory": _cuda_memory(device)}
            timing["updates"].append(dict(trace["timing"]))
            history.append(trace)
            heartbeat.update(state=RUNNING, phase="TRAINING", step=sampler.cursor, latest_losses={name: trace[name] for name in ("L_struct", "L_S", "L_I", "L_C")}, force=True)
            heartbeat.raise_liveness_error()

        stage1_condition = {"passed": False, "checks": {}}

        def diagnostic_payload(step: int) -> Mapping[str, Any]:
            payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition=stage1_condition, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot)
            payload["scientific_artifact"] = False
            payload["operational_artifact"] = True
            payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
            return payload

        recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
        checkpoint_path = recovery.save(diagnostic_payload(sampler.cursor), step=sampler.cursor)
        heartbeat.remember_recovery_checkpoint(checkpoint_path.name)
        heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=sampler.cursor, latest_losses={"diagnostic_checkpoint": checkpoint_path.name}, force=True)
        loaded = recovery.load(checkpoint_path)
        resume_state = _validate_resume_checkpoint(root, checkpoint_path, authorities)
        _require(resume_state["step"] == sampler.cursor and loaded["step"] == sampler.cursor, "diagnostic checkpoint cursor changed")

        generator_before_restore = _generator(consumer).detach().cpu().clone()
        encoder_before_restore = consumer.runtime.encoder.phase.detach().cpu().clone()
        optimizer_before_restore = copy.deepcopy(optimizer.state_dict())
        sampler_before_restore = sampler.state_dict()
        rng_before_restore = _rng_snapshot()
        payload = _mapping(loaded["payload"], "diagnostic loaded payload")
        consumer.load_trainable_state_dict({"encoder.phase": payload["encoder_state"], "generator.phase": payload["generator_state"]})
        optimizer.load_state_dict(copy.deepcopy(_mapping(payload["optimizer_state"], "diagnostic optimizer state")))
        sampler = ContinuationSampler(_mapping(payload["sampler_state"], "diagnostic sampler state"), cursor=int(loaded["step"]))
        _restore_rng(_mapping(payload["rng_state"], "diagnostic RNG state"))
        semantic.robust._validate_optimizer(consumer, optimizer)

        def _optimizer_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
            if left.get("param_groups") != right.get("param_groups"):
                return False
            left_state = _mapping(left.get("state"), "diagnostic saved optimizer state")
            right_state = _mapping(right.get("state"), "diagnostic restored optimizer state")
            if set(left_state) != set(right_state):
                return False
            for key in left_state:
                left_entry = _mapping(left_state[key], "diagnostic saved optimizer entry")
                right_entry = _mapping(right_state[key], "diagnostic restored optimizer entry")
                if set(left_entry) != set(right_entry):
                    return False
                for name in left_entry:
                    left_value, right_value = left_entry[name], right_entry[name]
                    if torch.is_tensor(left_value) or torch.is_tensor(right_value):
                        if not (torch.is_tensor(left_value) and torch.is_tensor(right_value) and torch.equal(left_value.detach().cpu(), right_value.detach().cpu())):
                            return False
                    elif left_value != right_value:
                        return False
            return True

        restore_checks = {
            "generator": torch.equal(generator_before_restore, _generator(consumer).detach().cpu()),
            "encoder": torch.equal(encoder_before_restore, consumer.runtime.encoder.phase.detach().cpu()),
            "optimizer": _optimizer_state_equal(optimizer_before_restore, optimizer.state_dict()),
            "sampler": sampler_before_restore["seed"] == sampler.state_dict()["seed"] and sampler_before_restore["steps"] == sampler.state_dict()["steps"] and sampler_before_restore["cursor"] == sampler.state_dict()["cursor"] and sampler_before_restore["wraparound"] == sampler.state_dict()["wraparound"] and all(torch.equal(left, right) for left, right in zip(sampler_before_restore["permutations"], sampler.state_dict()["permutations"])),
            "rng": _rng_equal(rng_before_restore, _rng_snapshot()),
        }
        _require(all(restore_checks.values()), "diagnostic checkpoint restore did not preserve state")
        heartbeat.stop_liveness()
        control.interrupted()
        heartbeat.update(state=RUN_INTERRUPTED, phase="INTERRUPTED", step=sampler.cursor, latest_losses={"diagnostic_only": "COMPLETE", "restore": "PASS"}, last_recovery_checkpoint=checkpoint_path.name, force=True)
        diagnostic_report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "task_id": TASK_ID,
            "result": "DIAGNOSTIC_ONLY",
            "status": "DIAGNOSTIC_ONLY_COMPLETE",
            "scientific_result_valid": False,
            "preflight": preflight,
            "three_step_real_update": {"updates": len(history), "persistent_updates": 0, "cursor": sampler.cursor},
            "checkpoint_reload": {"path": _relative(root, checkpoint_path), "resume_validation": "PASS", "restore_checks": restore_checks},
            "evaluation_sample_counts": {"DEV": {"9000": int(dev_records["9000"]["rows"])}, "S1": {"9000": int(s1_records["9000"]["rows"])}},
            "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0},
            "timing": timing,
            "source_manifest": source_manifest,
            "source_snapshot": source_snapshot,
            "cuda_runtime": cuda_runtime,
            "diagnostic_only": True,
            "output_dir": _relative(root, target),
        }
        _write_json(target / "diagnostic_report.json", diagnostic_report, "joint continuation diagnostic report")
        return diagnostic_report
    finally:
        heartbeat.stop_liveness()


def _stage1_condition(before: Mapping[str, Any], after: Mapping[str, Any], bootstrap: Mapping[str, Any], reconstruction: Mapping[str, Any], optical: Mapping[str, Any], trajectory: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "structural_delta_positive": float(bootstrap["estimate"]) > 0.0,
        "structural_ci_lower_positive": float(bootstrap["ci95"]["lower"]) > 0.0,
        "no_severe_per_class_reversal": all(float(item["ci95"]["upper"]) >= 0.0 for item in bootstrap["per_class"].values()),
        "L_S_nonincrease": float(after["L_S"]) <= float(before["L_S"]),
        "L_I_nonincrease": float(after["L_I"]) <= float(before["L_I"]),
        "L_C_nonincrease": float(after["L_C"]) <= float(before["L_C"]),
        "reconstruction_constraint": bool(reconstruction["passed"]),
        "optical_validity": bool(optical["passed"]),
        "joint_direction_never_degenerate": all(float(record["v_joint_l2_norm"]) > JOINT_DEGENERACY_TOLERANCE for record in trajectory.values()),
        "protected_test_sealed_zero": True,
    }
    return all(checks.values()), checks


def _classify(*, stage1_passed: bool, direction_stable: bool, final_structural: Mapping[str, Any], final_semantic: bool, s1_transfer: bool, reconstruction: Mapping[str, Any], optical: Mapping[str, Any]) -> str:
    if not direction_stable:
        return CLASSIFICATION_F
    if not stage1_passed:
        if final_semantic and not float(final_structural["ci95"]["lower"]) > 0.0:
            return CLASSIFICATION_D
        return CLASSIFICATION_E
    if not reconstruction["passed"] or not optical["passed"]:
        return CLASSIFICATION_G
    if not float(final_structural["ci95"]["lower"]) > 0.0 and final_semantic:
        return CLASSIFICATION_D
    if float(final_structural["ci95"]["lower"]) > 0.0 and not final_semantic:
        return CLASSIFICATION_C
    if float(final_structural["ci95"]["lower"]) > 0.0 and final_semantic:
        return CLASSIFICATION_A if s1_transfer else CLASSIFICATION_B
    return CLASSIFICATION_E


def _compare_gradient_tensor(left: torch.Tensor, right: torch.Tensor, name: str) -> dict[str, Any]:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    _require(left_cpu.shape == right_cpu.shape, f"{name} shape differs between gradient engines")
    _require(left_cpu.dtype == right_cpu.dtype, f"{name} dtype differs between gradient engines")
    _finite(left_cpu, f"{name} scalar")
    _finite(right_cpu, f"{name} batched")
    _require(torch.allclose(left_cpu, right_cpu, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), f"{name} differs between gradient engines")
    cosine = _dot(left_cpu, right_cpu) / (_norm(left_cpu, f"{name} scalar") * _norm(right_cpu, f"{name} batched"))
    _require(cosine >= GRADIENT_COSINE_MINIMUM, f"{name} cosine is below strict preflight threshold")
    return {"shape": list(left_cpu.shape), "dtype": str(left_cpu.dtype), "max_abs_difference": float((left_cpu - right_cpu).abs().max()), "cosine": cosine, "rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL}


def _compare_derived_direction(left: torch.Tensor, right: torch.Tensor, name: str) -> dict[str, Any]:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    _require(left_cpu.shape == right_cpu.shape and left_cpu.dtype == right_cpu.dtype, f"{name} geometry differs between gradient engines")
    _finite(left_cpu, f"{name} scalar")
    _finite(right_cpu, f"{name} batched")
    cosine = _dot(left_cpu, right_cpu) / (_norm(left_cpu, f"{name} scalar") * _norm(right_cpu, f"{name} batched"))
    _require(cosine >= GRADIENT_COSINE_MINIMUM, f"{name} cosine is below strict preflight threshold")
    return {"shape": list(left_cpu.shape), "dtype": str(left_cpu.dtype), "max_abs_difference": float((left_cpu - right_cpu).abs().max()), "cosine": cosine, "criterion": "directional cosine only; q_joint is a normalized derived direction"}


def _compare_direction_engines(scalar: Mapping[str, Any], batched: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("g_struct", "g_S", "g_I", "g_C", "g_rec", "g_op", "g_joint_scaled", "g_rec_safe", "g_op_safe", "g_total"):
        result[name] = _compare_gradient_tensor(_tensor(scalar.get(name), f"scalar {name}"), _tensor(batched.get(name), f"batched {name}"), name)
    result["q_joint"] = _compare_derived_direction(_tensor(scalar.get("q_joint"), "scalar q_joint"), _tensor(batched.get("q_joint"), "batched q_joint"), "q_joint")
    for key in ("g_rec", "g_op"):
        scalar_projection = _mapping(_mapping(scalar.get("projection"), "scalar projection").get(key), f"scalar {key} projection")
        batched_projection = _mapping(_mapping(batched.get("projection"), "batched projection").get(key), f"batched {key} projection")
        _require(scalar_projection.get("projected") == batched_projection.get("projected"), f"{key} projection decision differs between gradient engines")
        for field in ("dot_before", "dot_after", "removed_norm", "projection_fraction_norm", "eps_proj"):
            _require(math.isclose(float(scalar_projection[field]), float(batched_projection[field]), rel_tol=GRADIENT_RTOL, abs_tol=GRADIENT_ATOL), f"{key} projection {field} differs between gradient engines")
    _require(math.isclose(float(scalar["cos_structural_v_SIC"]), float(batched["cos_structural_v_SIC"]), rel_tol=GRADIENT_RTOL, abs_tol=GRADIENT_ATOL), "cos_structural_v_SIC differs between gradient engines")
    _require(math.isclose(float(scalar["v_joint_l2_norm"]), float(batched["v_joint_l2_norm"]), rel_tol=GRADIENT_RTOL, abs_tol=GRADIENT_ATOL), "v_joint_l2_norm differs between gradient engines")
    return result


def _compare_optimizer_states(left: Mapping[str, Any], right: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    _require(set(left) == set(right), f"{name} optimizer state keys differ")
    differences: list[float] = []

    def visit(first: Any, second: Any, path: str) -> None:
        if torch.is_tensor(first) or torch.is_tensor(second):
            _require(torch.is_tensor(first) and torch.is_tensor(second), f"{name} optimizer tensor mismatch at {path}")
            _require(first.shape == second.shape and first.dtype == second.dtype, f"{name} optimizer tensor geometry differs at {path}")
            _require(torch.allclose(first.detach().cpu(), second.detach().cpu(), rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), f"{name} optimizer tensor differs at {path}")
            differences.append(float((first.detach().cpu() - second.detach().cpu()).abs().max()))
        elif isinstance(first, Mapping) or isinstance(second, Mapping):
            _require(isinstance(first, Mapping) and isinstance(second, Mapping) and set(first) == set(second), f"{name} optimizer mapping differs at {path}")
            for key in first:
                visit(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
            _require(isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)) and len(first) == len(second), f"{name} optimizer sequence differs at {path}")
            for index, (first_item, second_item) in enumerate(zip(first, second, strict=True)):
                visit(first_item, second_item, f"{path}[{index}]")
        else:
            _require(first == second, f"{name} optimizer scalar differs at {path}")

    visit(left, right, name)
    return {"max_abs_difference": max(differences, default=0.0), "rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL}


def _select_gradient_engine(*, speedup_fraction: float, estimated_hours: Mapping[str, float], memory_fraction: Mapping[str, float]) -> dict[str, Any]:
    resource_ok = {
        engine: float(memory_fraction[engine]) <= PERFORMANCE_MAX_MEMORY_FRACTION
        for engine in (SCALAR_REFERENCE_ENGINE, GRADIENT_ENGINE)
    }
    if speedup_fraction >= PERFORMANCE_MIN_SPEEDUP and resource_ok[GRADIENT_ENGINE]:
        selected_engine = GRADIENT_ENGINE
        selection_reason = "batched_meets_speedup_and_memory_limit"
    elif resource_ok[SCALAR_REFERENCE_ENGINE]:
        selected_engine = SCALAR_REFERENCE_ENGINE
        selection_reason = "scalar_is_within_memory_limit"
    elif resource_ok[GRADIENT_ENGINE]:
        selected_engine = GRADIENT_ENGINE
        selection_reason = "batched_is_only_engine_within_memory_limit"
    else:
        selected_engine = None
        selection_reason = "both_engines_exceed_memory_limit"
    return {"selected_engine": selected_engine, "selection_reason": selection_reason, "resource_ok": resource_ok, "status": "READY" if selected_engine is not None else "BLOCKED"}


def _benchmark_gradient_engines(consumer: Any, smooth: Any, independent: Any, ec: Any, batch: torch.Tensor, references: Any, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    _require(device.type == "cuda", "gradient performance preflight requires CUDA")
    measurements: dict[str, list[float]] = {SCALAR_REFERENCE_ENGINE: [], GRADIENT_ENGINE: []}
    peaks: dict[str, dict[str, Any]] = {
        SCALAR_REFERENCE_ENGINE: {"device": str(device), "available": True, "allocated_bytes": 0, "reserved_bytes": 0, "max_reserved_bytes": 0, "max_allocated_bytes": 0},
        GRADIENT_ENGINE: {"device": str(device), "available": True, "allocated_bytes": 0, "reserved_bytes": 0, "max_reserved_bytes": 0, "max_allocated_bytes": 0},
    }
    # Warm up both implementations once, then alternate the first engine for
    # every timed pair so thermal/cache drift cannot systematically favour the
    # engine listed first.
    for engine in (SCALAR_REFERENCE_ENGINE, GRADIENT_ENGINE):
        for _ in range(PERFORMANCE_WARMUP_RUNS):
            _joint_direction(consumer, smooth, independent, ec, batch, references, mean, scale, gradient_engine=engine)
            semantic._clear_runtime_grads(consumer)
    for trial in range(PERFORMANCE_TIMED_RUNS):
        order = (SCALAR_REFERENCE_ENGINE, GRADIENT_ENGINE) if trial % 2 == 0 else (GRADIENT_ENGINE, SCALAR_REFERENCE_ENGINE)
        for engine in order:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            _joint_direction(consumer, smooth, independent, ec, batch, references, mean, scale, gradient_engine=engine)
            torch.cuda.synchronize(device)
            measurements[engine].append(time.perf_counter() - started)
            current_peak = _cuda_memory(device)
            for key in ("allocated_bytes", "reserved_bytes", "max_reserved_bytes", "max_allocated_bytes"):
                peaks[engine][key] = max(int(peaks[engine][key]), int(current_peak[key]))
            semantic._clear_runtime_grads(consumer)
    scalar_median = sorted(measurements[SCALAR_REFERENCE_ENGINE])[len(measurements[SCALAR_REFERENCE_ENGINE]) // 2]
    batched_median = sorted(measurements[GRADIENT_ENGINE])[len(measurements[GRADIENT_ENGINE]) // 2]
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    speedup = 1.0 - batched_median / scalar_median
    estimated_hours = {
        SCALAR_REFERENCE_ENGINE: scalar_median * float(TOTAL_STEPS - START_STEP) / 3600.0,
        GRADIENT_ENGINE: batched_median * float(TOTAL_STEPS - START_STEP) / 3600.0,
    }
    memory_fraction = {
        engine: int(peaks[engine]["max_reserved_bytes"]) / float(total_memory)
        for engine in peaks
    }
    selection = _select_gradient_engine(speedup_fraction=speedup, estimated_hours=estimated_hours, memory_fraction=memory_fraction)
    return {
        "status": selection["status"],
        "batch_rows": int(batch.shape[0]),
        "warmup_runs": PERFORMANCE_WARMUP_RUNS,
        "timed_runs": PERFORMANCE_TIMED_RUNS,
        "durations_seconds": measurements,
        "median_seconds": {SCALAR_REFERENCE_ENGINE: scalar_median, GRADIENT_ENGINE: batched_median},
        "speedup_fraction": speedup,
        "estimated_3000_step_hours": estimated_hours,
        "peak_memory": peaks,
        "device_total_memory_bytes": total_memory,
        "peak_reserved_memory_fraction": memory_fraction,
        "resource_ok": selection["resource_ok"],
        "selected_engine": selection["selected_engine"],
        "selection_reason": selection["selection_reason"],
        "criteria": {"minimum_speedup_fraction": PERFORMANCE_MIN_SPEEDUP, "maximum_memory_fraction": PERFORMANCE_MAX_MEMORY_FRACTION},
    }


def _preflight(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, authorities: Mapping[str, Any], mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    initial_state = _runtime_state(consumer)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = _rng_snapshot()
    initial_cursor = sampler.cursor
    fixed_indices = _tensor(_mapping(authorities["step9000_attribution_gradients"]["A_B_indices"], "accepted A/B indices").get("a_indices"), "accepted A indices").to(torch.long)
    _require(fixed_indices.numel() == 20, "accepted TRAIN A calibration size changed")
    _require(torch.equal(authorities["labels"].index_select(0, fixed_indices), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])), "accepted TRAIN A calibration labels changed")
    batch = authorities["images"].index_select(0, fixed_indices).to(device)
    references = authorities["training"]["references"]
    generator = _generator(consumer)

    def restore_initial_state() -> None:
        consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        sampler.cursor = initial_cursor
        _restore_rng({"cpu": initial_rng[0], "cuda": initial_rng[1]})
        semantic._freeze_generator_only(consumer)
        semantic._clear_runtime_grads(consumer)
        semantic._clear_evaluator_grads(smooth, independent, ec, ed)

    microbatch_comparisons: list[dict[str, Any]] = []
    aggregate_batched = {name: torch.zeros_like(generator) for name in ("g_struct", "g_S", "g_I", "g_C")}
    aggregate_scalar = {name: torch.zeros_like(generator) for name in ("g_struct", "g_S", "g_I", "g_C")}
    for microbatch_index in range(2):
        microbatch = batch[microbatch_index * ATTRIBUTION_MICRO_BATCH_SIZE : (microbatch_index + 1) * ATTRIBUTION_MICRO_BATCH_SIZE]
        scalar_direction = _joint_direction(consumer, smooth, independent, ec, microbatch, references, mean, scale, gradient_engine=SCALAR_REFERENCE_ENGINE)
        semantic._clear_runtime_grads(consumer)
        batched_direction = _joint_direction(consumer, smooth, independent, ec, microbatch, references, mean, scale, gradient_engine=GRADIENT_ENGINE)
        _require(batched_direction["evaluator_parameters_in_backward"]["E_D"] is False, "E_D entered joint backward")
        microbatch_comparisons.append({"microbatch": microbatch_index, "rows": int(microbatch.shape[0]), "comparison": _compare_direction_engines(scalar_direction, batched_direction)})
        for name in aggregate_batched:
            aggregate_scalar[name] += 0.5 * scalar_direction[name].detach()
            aggregate_batched[name] += 0.5 * batched_direction[name].detach()
        semantic._clear_runtime_grads(consumer)

    accepted_step = _mapping(authorities["step9000_attribution_gradients"]["key_steps"], "accepted attribution key steps")["9000"]
    accepted_gradients = _accepted_style_gradients(
        consumer,
        smooth,
        independent,
        ec,
        authorities["images"],
        fixed_indices,
        authorities["training"]["references"],
        mean,
        scale,
    )
    expected_gradients = {
        "g_struct": _tensor(accepted_step["g_structural"], "accepted structural gradient"),
        "g_S": _tensor(accepted_step["g_E_S"], "accepted E_S gradient"),
        "g_I": _tensor(accepted_step["g_E_I"], "accepted E_I gradient"),
        "g_C": _tensor(accepted_step["g_E_C"], "accepted E_C gradient"),
    }
    accepted_gradient_comparisons: dict[str, Any] = {}
    for name in ("g_struct", "g_S", "g_I", "g_C"):
        observed = accepted_gradients[name].detach().cpu()
        expected = expected_gradients[name].detach().cpu()
        _require(observed.shape == expected.shape, f"accepted step9000 {name} geometry changed")
        accepted_gradient_comparisons[name] = {"accepted_style": _compare_gradient_tensor(observed, expected, f"accepted step9000 {name}"), "batched_aggregate": _compare_gradient_tensor(aggregate_batched[name], expected, f"batched aggregate step9000 {name}"), "scalar_aggregate": _compare_gradient_tensor(aggregate_scalar[name], expected, f"scalar aggregate step9000 {name}")}

    performance = _benchmark_gradient_engines(consumer, smooth, independent, ec, batch[:ATTRIBUTION_MICRO_BATCH_SIZE], references, mean, scale, device)
    _require(performance["status"] == "READY", "PREFLIGHT_BLOCKED(resource): both gradient engines exceed the absolute resource envelope")
    selected_engine = str(performance["selected_engine"])

    direction_batch = batch[:ATTRIBUTION_MICRO_BATCH_SIZE]
    direction = _joint_direction(consumer, smooth, independent, ec, direction_batch, references, mean, scale, gradient_engine=selected_engine)
    _require(direction["gradient_engine"] == selected_engine, "preflight did not use the selected gradient engine")
    before = {name: float(direction["losses"][name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
    saved_generator = generator.detach().clone()
    with torch.no_grad():
        generator.add_(-PRELIGHT_EPSILON * direction["q_joint"])
    with torch.inference_mode():
        after_bundle = semantic._loss_bundle(consumer, smooth, independent, ec, direction_batch, references)
        after_raw_view = semantic._cmos_view(after_bundle["raw"])
        after_target_view = semantic._cmos_view(direction_batch)
        after_generated = structural._normalized_surrogate(structural._structural_surrogate(after_raw_view), mean.to(device), scale.to(device))
        after_target = structural._normalized_surrogate(structural._structural_surrogate(after_target_view), mean.to(device), scale.to(device))
        after_losses = dict(after_bundle["losses"])
        after_losses["L_struct"] = (after_generated - after_target).square().mean()
    after = {name: float(after_losses[name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
    with torch.no_grad():
        generator.copy_(saved_generator)
        generator.add_(PRELIGHT_EPSILON * direction["q_joint"])
    with torch.inference_mode():
        negative_bundle = semantic._loss_bundle(consumer, smooth, independent, ec, direction_batch, references)
        negative_raw_view = semantic._cmos_view(negative_bundle["raw"])
        negative_generated = structural._normalized_surrogate(structural._structural_surrogate(negative_raw_view), mean.to(device), scale.to(device))
        negative_losses = dict(negative_bundle["losses"])
        negative_losses["L_struct"] = (negative_generated - after_target).square().mean()
    negative_after = {name: float(negative_losses[name].detach().cpu()) for name in ("L_struct", "L_S", "L_I", "L_C")}
    negative_directional_reference = {
        "g_struct_dot_q_joint": _dot(direction["g_struct"].detach(), direction["q_joint"].detach()),
        "g_S_dot_q_joint": _dot(direction["g_S"].detach(), direction["q_joint"].detach()),
        "g_I_dot_q_joint": _dot(direction["g_I"].detach(), direction["q_joint"].detach()),
        "g_C_dot_q_joint": _dot(direction["g_C"].detach(), direction["q_joint"].detach()),
    }
    negative_sign_rejected = bool(
        negative_directional_reference["g_struct_dot_q_joint"] > 0.0
        and any(negative_directional_reference[key] > 0.0 for key in ("g_S_dot_q_joint", "g_I_dot_q_joint", "g_C_dot_q_joint"))
        and negative_after["L_struct"] > before["L_struct"]
    )
    with torch.no_grad():
        generator.copy_(saved_generator)
    _require(negative_sign_rejected, "negative-control update was not rejected by the independent direction check")
    semantic._clear_runtime_grads(consumer)
    semantic._clear_evaluator_grads(smooth, independent, ec, ed)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "directional sanity did not restore runtime")
    _require(_rng_equal(_rng_snapshot(), initial_rng), "directional sanity changed RNG")

    one_step_batch = batch[:ATTRIBUTION_MICRO_BATCH_SIZE]
    scalar_trace = _apply_update(consumer, smooth, independent, ec, optimizer, one_step_batch, references, mean, scale, step=START_STEP + 1, clean_generator_state=initial_state["generator.phase"], gradient_engine=SCALAR_REFERENCE_ENGINE)
    scalar_parameters = generator.detach().clone()
    scalar_optimizer = copy.deepcopy(optimizer.state_dict())
    restore_initial_state()
    scalar_restored = _state_digest(_runtime_state(consumer)) == _state_digest(initial_state) and _rng_equal(_rng_snapshot(), initial_rng)
    batched_trace = _apply_update(consumer, smooth, independent, ec, optimizer, one_step_batch, references, mean, scale, step=START_STEP + 1, clean_generator_state=initial_state["generator.phase"], gradient_engine=GRADIENT_ENGINE)
    batched_parameters = generator.detach().clone()
    batched_optimizer = copy.deepcopy(optimizer.state_dict())
    adam_one_step_equivalence = {
        "parameter_update": _compare_gradient_tensor(scalar_parameters, batched_parameters, "one-step generator parameter"),
        "optimizer_state": _compare_optimizer_states(scalar_optimizer, batched_optimizer, name="one-step Adam"),
        "scalar_trace_engine": scalar_trace["gradient_engine"],
        "batched_trace_engine": batched_trace["gradient_engine"],
        "restoration_equivalence": {"scalar": scalar_restored, "batched": True},
    }
    restore_initial_state()
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "one-step Adam restore failed")
    adam_one_step_equivalence["restoration_equivalence"]["batched"] = _rng_equal(_rng_snapshot(), initial_rng)
    _require(adam_one_step_equivalence["restoration_equivalence"]["scalar"], "scalar one-step restore failed")
    _require(adam_one_step_equivalence["restoration_equivalence"]["batched"], "batched one-step restore failed")

    traces = []
    for update in range(3):
        indices, labels = sampler.next_batch()
        _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"preflight TRAIN labels changed at update {update + 1}")
        traces.append(_apply_update(consumer, smooth, independent, ec, optimizer, authorities["images"].index_select(0, indices).to(device), references, mean, scale, step=START_STEP + update + 1, clean_generator_state=initial_state["generator.phase"], gradient_engine=selected_engine))
    _require(not torch.equal(initial_state["generator.phase"], _runtime_state(consumer)["generator.phase"]), "three-step preflight did not update generator")
    _require(torch.equal(initial_state["encoder.phase"], _runtime_state(consumer)["encoder.phase"]), "three-step preflight changed encoder")
    restore_initial_state()
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "preflight runtime restore failed")
    _require(float(optimizer.state[_generator(consumer)]["step"].item()) == 9300.0, "preflight Adam restore failed")
    _require(sampler.cursor == START_STEP, "preflight sampler restore failed")
    return {"schema": PREFLIGHT_SCHEMA, "status": "READY", "gradient_engine": selected_engine, "selected_engine": selected_engine, "scalar_reference_engine": SCALAR_REFERENCE_ENGINE, "gradient_tolerances": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "cosine_minimum": GRADIENT_COSINE_MINIMUM}, "semantic_batched_vjp": {"rows": 3, "mapping": ["L_S->g_S", "L_I->g_I", "L_C->g_C"], "fail_closed": True}, "microbatch_engine_comparison": microbatch_comparisons, "three_step_real_update": {"updates": 3, "persistent_updates": 0, "trace": traces}, "one_step_adam_equivalence": adam_one_step_equivalence, "performance": performance, "starting_cursor": START_STEP, "adam_internal_step_at_start": 9300.0, "adam_internal_step_after_temporary_preflight": 9303.0, "stored_rng_restored": True, "accepted_step9000_gradient_match": accepted_gradient_comparisons, "directional_sanity": {"epsilon": PRELIGHT_EPSILON, "before": before, "after": after, "q_joint_finite": True}, "negative_control": {"wrong_sign_after": negative_after, "directional_reference": negative_directional_reference, "rejected": negative_sign_rejected, "criterion": "wrong-sign structural finite-difference increase plus at least one semantic gradient positive projection"}, "trainable_parameter_names": ["generator.phase"], "E_D_in_backward": False, "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0}}


def run_training(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR, resume_checkpoint: str | Path | None = None) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    resume_path = None if resume_checkpoint is None else (root / resume_checkpoint if not Path(resume_checkpoint).is_absolute() else Path(resume_checkpoint)).resolve()
    target = resume_path.parent if resume_path is not None else (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    if resume_path is None:
        _require(not target.exists(), f"output already exists: {target}")
    else:
        _require(target.is_dir(), f"resume run directory is missing: {target}")
    authorities = _load_authorities(root)
    resume_state = None if resume_path is None else _validate_resume_checkpoint(root, resume_path, authorities)
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "joint continuation requires CUDA")
    device = torch.device("cuda:0")
    cuda_runtime = semantic.robust._prepare_cuda_runtime()
    consumer, smooth, independent, ec, ed = _load_models(root, authorities, device)
    checkpoint = authorities["step9000"]
    optimizer = _optimizer(consumer, checkpoint)
    sampler = ContinuationSampler(checkpoint["sampler_state"], cursor=START_STEP)
    if resume_state is None:
        _restore_start_state(consumer, optimizer, sampler, checkpoint)
    else:
        payload = _mapping(resume_state["payload"], "resume payload")
        consumer.load_trainable_state_dict({"encoder.phase": payload["encoder_state"], "generator.phase": payload["generator_state"]})
        semantic._freeze_generator_only(consumer)
        optimizer.load_state_dict(copy.deepcopy(_mapping(payload.get("optimizer_state"), "resume optimizer state")))
        semantic.robust._validate_optimizer(consumer, optimizer)
        sampler = ContinuationSampler(_mapping(payload.get("sampler_state"), "resume sampler state"), cursor=int(resume_state["step"]))
        _restore_rng(_mapping(payload.get("rng_state"), "resume RNG state"))
        semantic._clear_runtime_grads(consumer)
    mean, scale = authorities["surrogate_mean"], authorities["surrogate_scale"]
    ledger = DataReadLedger()
    baseline_private: dict[str, Any] | None = None
    baseline_private_ref: dict[str, Any] | None = None
    if resume_state is None:
        try:
            preflight = _preflight(consumer, smooth, independent, ec, ed, optimizer, sampler, authorities, mean, scale, device)
        except JointContinuationError as error:
            component, signature = _classify_preflight_failure(error)
            retry_control = _preflight_failure_control(target)
            decision = retry_control.record_preflight_failure(component, signature)
            _record_persisted_preflight_failure(target, retry_control, component, signature, decision)
            raise
        _require(preflight["status"] == "READY", "joint continuation preflight did not reach READY")
        gradient_engine = str(preflight["gradient_engine"])
        source_manifest = _source_manifest_for_run(root, authorities, gradient_engine=gradient_engine)
        receipt = _preflight_receipt(source_manifest, preflight)
        baseline_private = _materialize_step9000_private_baseline(consumer, authorities, authorities["structural_reference"], device)
        ledger.record("dev", int(baseline_private["rows"]))
    else:
        preflight = _resume_preflight(resume_path, resume_state)
        source_manifest = dict(resume_state["current_manifest"])
        gradient_engine = str(preflight["selected_engine"])
        receipt = resume_state["preflight_receipt"]
    identity = _make_identity(source_manifest, target.name, gradient_engine=gradient_engine)
    if resume_state is not None:
        _require(identity.scientific_attempt_id == resume_state["identity"].scientific_attempt_id, "resume scientific attempt identity changed", ResumeCompatibilityError)
    control = ResearchRunControl(identity, AUTHORIZATION_ID)
    control.mark_preflight_ready(receipt)
    effective_identity = control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt, resume=resume_state is not None)
    if resume_state is None:
        _write_torch(target / "baseline_private_step_009000.pt", baseline_private, "joint continuation private step9000 baseline")
        baseline_private_ref = _artifact_ref(root, target / "baseline_private_step_009000.pt", "step9000 private paired-bootstrap baseline")
        _write_json(target / "source_manifest.json", source_manifest, "joint continuation source manifest")
        _write_json(target / "preflight.json", {**preflight, "preflight_receipt": receipt.to_dict()}, "joint continuation preflight")
        source_snapshot = _write_source_snapshot(root, target, source_manifest)
        _write_json(target / "execution_manifest.json", {"schema": "joint_structural_semantic_continuation_execution_manifest.v2", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN}, "gradient_engine": gradient_engine, "input_authorities": authorities["authority_hashes"], "reused_step9000_baseline": authorities["reused_step9000_baseline"]["public_authority"], "private_baseline": baseline_private_ref, "source_manifest": source_manifest, "source_snapshot": source_snapshot, "run_identity": effective_identity.to_dict(), "local_state": authorities["local_state"], "cuda_runtime": cuda_runtime, "scope": {"stage1": [START_STEP, STAGE1_END], "stage2": [STAGE1_END, TOTAL_STEPS], "trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T", "S1", "E_S", "E_I", "E_C", "E_D", "336D surrogate", "337D hard metric"], "protected": False, "test": False, "sealed": False}}, "joint continuation execution manifest")
    else:
        source_snapshot = _mapping(_mapping(resume_state["payload"], "resume payload").get("source_snapshot"), "resume source snapshot")
        _write_json(target / f"resume_preflight_step_{int(resume_state['step']):06d}.json", preflight, "joint continuation resume preflight")

    clean_generator_state = checkpoint["generator_state"].to(device)
    if resume_state is None:
        history: list[dict[str, Any]] = []
        trajectory: dict[str, dict[str, Any]] = {}
        dev_records: dict[str, dict[str, Any]] = {}
        s1_records: dict[str, dict[str, Any]] = {}
        timing: dict[str, Any] = {"schema": TIMING_SCHEMA, "stages": {}, "updates": []}
    else:
        resume_payload = _mapping(resume_state["payload"], "resume payload")
        history = [dict(record) for record in _mapping({"history": resume_payload["training_history"]}, "resume history container")["history"]]
        trajectory = {str(key): dict(value) for key, value in _mapping(resume_payload["trajectory"], "resume trajectory").items()}
        dev_records = {str(key): dict(value) for key, value in _mapping(resume_payload["dev_records"], "resume DEV records").items()}
        s1_records = {str(key): dict(value) for key, value in _mapping(resume_payload["s1_records"], "resume S1 records").items()}
        timing = dict(_mapping(resume_payload["timing"], "resume timing"))
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=TOTAL_STEPS, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    resume_step = START_STEP if resume_state is None else int(resume_state["step"])
    heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="PREPARING", step=resume_step, latest_losses={"preflight": "READY", "optimizer_mode": "STEP9000_ADAM_STATE_RESTORED_EXACTLY" if resume_state is None else "EXPLICIT_RESUME_STATE_RESTORED"}, force=True)
    heartbeat.start_liveness()
    preparing_started = time.perf_counter()
    fixed_indices = torch.stack([value[START_STEP % value.numel()] for value in sampler.permutations])
    fixed_batch = authorities["images"].index_select(0, fixed_indices).to(device)
    if resume_state is None:
        composition = _joint_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"], mean, scale, gradient_engine=gradient_engine)
        trajectory[str(START_STEP)] = _composition_record(composition, START_STEP, 0.0, 0.0)
        semantic._clear_runtime_grads(consumer)
        semantic._clear_evaluator_grads(smooth, independent, ec, ed)
        _require(baseline_private is not None, "step9000 private baseline was not materialized")
        reused_baseline = authorities["reused_step9000_baseline"]
        dev_records[str(START_STEP)] = dict(_mapping(reused_baseline["dev"], "reused step9000 DEV baseline"))
        dev_records[str(START_STEP)]["_labels_dev"] = baseline_private["labels_dev"].clone()
        dev_records[str(START_STEP)]["_log_rho"] = baseline_private["log_rho"].clone()
        s1_records[str(START_STEP)] = dict(_mapping(reused_baseline["s1"], "reused step9000 S1 baseline"))
        _write_json(target / f"dev_diagnostics_step_{START_STEP:06d}.json", {key: value for key, value in dev_records[str(START_STEP)].items() if not key.startswith("_")}, "step9000 DEV diagnostics")
        _write_json(target / f"s1_transfer_step_{START_STEP:06d}.json", {key: value for key, value in s1_records[str(START_STEP)].items() if not key.startswith("_")}, "step9000 S1 transfer")
        _write_torch(target / f"scientific_step_{START_STEP:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=START_STEP, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition=None, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot), "joint continuation starting checkpoint")
    elif str(resume_step) not in dev_records:
        dev_records[str(resume_step)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], mean, scale, ledger, device, step=resume_step)
        s1_records[str(resume_step)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=resume_step)
        _write_json(target / f"dev_diagnostics_step_{resume_step:06d}.json", {key: value for key, value in dev_records[str(resume_step)].items() if not key.startswith("_")}, f"resume step-{resume_step} DEV diagnostics")
        _write_json(target / f"s1_transfer_step_{resume_step:06d}.json", {key: value for key, value in s1_records[str(resume_step)].items() if not key.startswith("_")}, f"resume step-{resume_step} S1 transfer")
    if resume_state is not None:
        heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="TRAINING", step=resume_step, latest_losses={"resume": "EXACT_STATE_RESTORED", "endpoint": "READY"}, force=True)
    timing["stages"]["preparing"] = {"seconds": time.perf_counter() - preparing_started, "cuda_memory": _cuda_memory(device)}

    if resume_state is None:
        stage1_passed = False
        stage1_checks: dict[str, bool] = {}
        stage1_bootstrap: dict[str, Any] | None = None
        stage1_reconstruction: dict[str, Any] | None = None
        stage1_optical: dict[str, Any] | None = None
    else:
        resume_payload = _mapping(resume_state["payload"], "resume payload")
        stage1_condition = _mapping(resume_payload.get("stage1_condition"), "resume Stage-1 condition")
        stage1_passed = bool(stage1_condition.get("passed"))
        stage1_checks = {str(key): bool(value) for key, value in _mapping(stage1_condition.get("checks"), "resume Stage-1 checks").items()}
        stage1_bootstrap = None if resume_payload.get("stage1_bootstrap") is None else dict(_mapping(resume_payload.get("stage1_bootstrap"), "resume Stage-1 bootstrap"))
        stage1_reconstruction = None if resume_payload.get("stage1_reconstruction") is None else dict(_mapping(resume_payload.get("stage1_reconstruction"), "resume Stage-1 reconstruction"))
        stage1_optical = None if resume_payload.get("stage1_optical") is None else dict(_mapping(resume_payload.get("stage1_optical"), "resume Stage-1 optical validity"))
    resume_stage1_pending = bool(
        resume_state is not None
        and resume_step == STAGE1_END
        and (
            str(STAGE1_END) not in dev_records
            or str(STAGE1_END) not in s1_records
            or stage1_bootstrap is None
            or stage1_reconstruction is None
            or stage1_optical is None
        )
    )
    if resume_stage1_pending:
        stage1_started = time.perf_counter()
        _require(str(START_STEP) in dev_records and str(START_STEP) in s1_records, "resume Stage-1 baseline endpoint is missing")
        _require(str(STAGE1_END) in dev_records and str(STAGE1_END) in s1_records, "resume Stage-1 endpoint evaluation is missing")
        stage1_bootstrap = _paired_bootstrap(dev_records[str(START_STEP)]["_labels_dev"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(STAGE1_END)]["_log_rho"])
        stage1_reconstruction = {"step9000": dev_records[str(START_STEP)]["L_rec"], "step9300": dev_records[str(STAGE1_END)]["L_rec"], "delta": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
        stage1_optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(STAGE1_END)])
        stage1_passed, stage1_checks = _stage1_condition(dev_records[str(START_STEP)], dev_records[str(STAGE1_END)], stage1_bootstrap, stage1_reconstruction, stage1_optical, trajectory)
        _write_json(target / f"dev_diagnostics_step_{STAGE1_END:06d}.json", {key: value for key, value in dev_records[str(STAGE1_END)].items() if not key.startswith("_")}, "resume step9300 DEV diagnostics")
        _write_json(target / f"s1_transfer_step_{STAGE1_END:06d}.json", {key: value for key, value in s1_records[str(STAGE1_END)].items() if not key.startswith("_")}, "resume step9300 S1 transfer")
        _write_torch(target / f"scientific_step_{STAGE1_END:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=STAGE1_END, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical), "resume joint continuation Stage-1 endpoint checkpoint")
        timing["stages"]["stage1_collection"] = {"seconds": time.perf_counter() - stage1_started, "cuda_memory": _cuda_memory(device)}
        heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="POST_TRAIN_COLLECTION", step=STAGE1_END, latest_losses={"stage1_passed": stage1_passed, "resume_endpoint": "EVALUATED"}, force=True)
    direction_stable = True
    last_recovery_step = resume_step
    last_recovery_time = time.time()

    def recovery_payload(step: int) -> Mapping[str, Any]:
        payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical)
        payload["scientific_artifact"] = False
        payload["operational_artifact"] = True
        payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
        return payload

    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_payload, phase="TRAINING", total_steps=TOTAL_STEPS)
    continue_training = not resume_stage1_pending or stage1_passed
    with interrupt:
        while continue_training and sampler.cursor < TOTAL_STEPS:
            current_step = sampler.cursor
            indices, labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"TRAIN sampler labels changed at update {current_step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            update_started = time.perf_counter()
            try:
                trace = _apply_update(consumer, smooth, independent, ec, optimizer, batch, authorities["training"]["references"], mean, scale, step=current_step + 1, clean_generator_state=clean_generator_state, gradient_engine=gradient_engine)
            except JointDirectionDegenerated:
                direction_stable = False
                break
            trace["timing"] = {"seconds": time.perf_counter() - update_started, "cuda_memory": _cuda_memory(device)}
            timing["updates"].append(dict(trace["timing"]))
            history.append(trace)
            completed = current_step + 1
            now = time.time()
            if recovery_due(step=completed, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                recovery_checkpoint = recovery.save(recovery_payload(completed), step=completed)
                heartbeat.remember_recovery_checkpoint(recovery_checkpoint.name)
                last_recovery_step = completed
                last_recovery_time = now
            heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="TRAINING", step=completed, latest_losses={name: trace[name] for name in ("L_struct", "L_S", "L_I", "L_C")})
            heartbeat.raise_liveness_error()
            interrupted = interrupt.handle_safe_boundary(step=completed, latest_losses={name: trace[name] for name in ("L_struct", "L_S", "L_I", "L_C")}, last_recovery_checkpoint=None)
            if interrupted is not None:
                control.interrupted()
                heartbeat.stop_liveness()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": completed, "recovery_checkpoint": interrupted.name}
            if completed in CHECKPOINT_STEPS:
                composition = _joint_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"], mean, scale, gradient_engine=gradient_engine)
                trajectory[str(completed)] = _composition_record(composition, completed, _norm(_generator(consumer).detach() - clean_generator_state, "phase displacement"), float(trace["phase_update_rms"]))
                semantic._clear_runtime_grads(consumer)
                semantic._clear_evaluator_grads(smooth, independent, ec, ed)
            if completed == STAGE1_END:
                stage1_started = time.perf_counter()
                dev_records[str(completed)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], mean, scale, ledger, device, step=completed)
                s1_records[str(completed)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=completed)
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {key: value for key, value in dev_records[str(completed)].items() if not key.startswith("_")}, "step9300 DEV diagnostics")
                _write_json(target / f"s1_transfer_step_{completed:06d}.json", {key: value for key, value in s1_records[str(completed)].items() if not key.startswith("_")}, "step9300 S1 transfer")
                stage1_bootstrap = _paired_bootstrap(dev_records[str(START_STEP)]["_labels_dev"], dev_records[str(START_STEP)]["_log_rho"], dev_records[str(completed)]["_log_rho"])
                stage1_reconstruction = {"step9000": dev_records[str(START_STEP)]["L_rec"], "step9300": dev_records[str(completed)]["L_rec"], "delta": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
                stage1_optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(completed)])
                stage1_passed, stage1_checks = _stage1_condition(dev_records[str(START_STEP)], dev_records[str(completed)], stage1_bootstrap, stage1_reconstruction, stage1_optical, trajectory)
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical), "joint continuation Stage-1 endpoint checkpoint")
                timing["stages"]["stage1_collection"] = {"seconds": time.perf_counter() - stage1_started, "cuda_memory": _cuda_memory(device)}
                heartbeat.update(state=RUN_SCIENTIFIC_ACTIVE, phase="POST_TRAIN_COLLECTION", step=completed, latest_losses={"stage1_passed": stage1_passed}, force=True)
                if not stage1_passed:
                    break
            elif completed in CHECKPOINT_STEPS and completed != TOTAL_STEPS:
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical), f"joint continuation step-{completed} checkpoint")

    stage2_executed = bool(stage1_passed and sampler.cursor == TOTAL_STEPS)
    if stage2_executed and str(TOTAL_STEPS) not in dev_records:
        final_collection_started = time.perf_counter()
        dev_records[str(TOTAL_STEPS)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], mean, scale, ledger, device, step=TOTAL_STEPS)
        s1_records[str(TOTAL_STEPS)] = _evaluate_s1(consumer, smooth, independent, ec, ed, authorities, authorities["structural_reference"], device, ledger, step=TOTAL_STEPS)
        _write_json(target / f"dev_diagnostics_step_{TOTAL_STEPS:06d}.json", {key: value for key, value in dev_records[str(TOTAL_STEPS)].items() if not key.startswith("_")}, "step12000 DEV diagnostics")
        _write_json(target / f"s1_transfer_step_{TOTAL_STEPS:06d}.json", {key: value for key, value in s1_records[str(TOTAL_STEPS)].items() if not key.startswith("_")}, "step12000 S1 transfer")
        _write_torch(target / f"scientific_step_{TOTAL_STEPS:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=TOTAL_STEPS, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": stage1_checks}, dev_records=dev_records, s1_records=s1_records, timing=timing, source_snapshot=source_snapshot, stage1_bootstrap=stage1_bootstrap, stage1_reconstruction=stage1_reconstruction, stage1_optical=stage1_optical), "joint continuation final checkpoint")
        timing["stages"]["stage2_collection"] = {"seconds": time.perf_counter() - final_collection_started, "cuda_memory": _cuda_memory(device)}

    final_step = TOTAL_STEPS if stage2_executed else (STAGE1_END if str(STAGE1_END) in dev_records else START_STEP)
    final_dev = dev_records[str(final_step)]
    final_s1 = s1_records[str(final_step)]
    final_bootstrap = _paired_bootstrap(dev_records[str(START_STEP)]["_labels_dev"], dev_records[str(START_STEP)]["_log_rho"], final_dev["_log_rho"])
    final_semantic = all(float(final_dev[name]) <= float(dev_records[str(START_STEP)][name]) for name in ("L_S", "L_I", "L_C"))
    s1_structural_gain = float(final_s1["structural"]["structural_support_rate"]) >= float(s1_records[str(START_STEP)]["structural"]["structural_support_rate"])
    s1_semantic_gain = all(float(final_s1["semantic"][name]["macro_recall"]) >= float(s1_records[str(START_STEP)]["semantic"][name]["macro_recall"]) for name in ("E_S", "E_I", "E_C", "E_D"))
    classification = _classify(stage1_passed=stage1_passed, direction_stable=direction_stable, final_structural=final_bootstrap, final_semantic=final_semantic, s1_transfer=s1_structural_gain and s1_semantic_gain, reconstruction={"passed": final_dev["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}, optical=_optical_pass(dev_records[str(START_STEP)], final_dev))
    public_dev = {key: {subkey: value for subkey, value in record.items() if not subkey.startswith("_")} for key, record in dev_records.items()}
    evaluation_sample_counts = {
        "DEV": {key: int(record["rows"]) for key, record in dev_records.items()},
        "S1": {key: int(record["rows"]) for key, record in s1_records.items()},
    }
    if baseline_private_ref is None and (target / "baseline_private_step_009000.pt").is_file():
        baseline_private_ref = _artifact_ref(root, target / "baseline_private_step_009000.pt", "step9000 private paired-bootstrap baseline")
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(stage1_passed),
        "classification": classification,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "step9000->9300 and conditional step9300->12000"},
        "local_state": {"branch": authorities["local_state"]["branch"], "head": authorities["local_state"]["head"], "dirty_before": authorities["local_state"].get("dirty", True), "worktree_preserved": True},
        "supported_frontier": "LOCAL_SHARED_STRUCTURAL_SEMANTIC_DIRECTION_REMAINS_AVAILABLE",
        "input_authorities": authorities["authority_hashes"],
        "gradient_engine": gradient_engine,
        "scalar_reference_gradient_engine": SCALAR_REFERENCE_ENGINE,
        "baseline_reuse": {"public": authorities["reused_step9000_baseline"]["public_authority"], "private": baseline_private_ref, "selection": "step9000 public DEV/S1 diagnostics reused by fixed digest; only per-row labels and true log_rho supplemented"},
        "starting_state": {"step": START_STEP, "checkpoint_sha256": START_CHECKPOINT_SHA256, "optimizer_internal_step": 9300.0, "sampler_cursor": START_STEP, "sampler_terminal": TOTAL_STEPS, "rng_restored": True},
        "joint_gradient_contract": {"u_struct": "g_struct/||g_struct||_2", "u_SIC": "sum(g_k/||g_k||_2)/||sum(...)||_2", "q_joint": "(u_struct+u_SIC)/RMS(u_struct+u_SIC)", "descent": "-q_joint", "delta_joint": DELTA_JOINT, "E_D_in_backward": False, "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "gradient_engine": gradient_engine, "semantic_batched_mapping": ["L_S->g_S", "L_I->g_I", "L_C->g_C"], "structural_reconstruction_optical": "scalar VJP"},
        "auxiliary_contract": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "projection": "only opposing reconstruction/optical components relative to joint direction"},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 9300.0, "reset": False},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T=I_100", "S1", "E_S", "E_I", "E_C", "E_D", "336D surrogate", "337D hard metric"], "E_D_in_backward": False},
        "preflight": preflight,
        "stage1": {"updates": min(len(history), STAGE1_UPDATES), "checkpoints": [9000, 9100, 9200, 9300], "development": public_dev, "S1_transfer": {key: {subkey: value for subkey, value in s1_records[key].items() if not subkey.startswith("_")} for key in ("9000", "9300") if key in s1_records}, "bootstrap": {key: value for key, value in (stage1_bootstrap or {}).items() if not isinstance(value, torch.Tensor)}, "reconstruction": stage1_reconstruction, "optical_validity": stage1_optical, "continuation_condition": {"passed": stage1_passed, "checks": stage1_checks}},
        "stage2": {"executed": stage2_executed, "checkpoints": [9300, 9600, 10000, 10500, 11000, 11500, 12000], "final_step": final_step, "development": public_dev, "S1_transfer": {key: {subkey: value for subkey, value in s1_records[key].items() if not subkey.startswith("_")} for key in ("12000",) if key in s1_records}},
        "evaluation_sample_counts": evaluation_sample_counts,
        "final_structural_bootstrap": {key: value for key, value in final_bootstrap.items() if not isinstance(value, torch.Tensor)},
        "training_trajectory": {"updates": len(history), "scientific_cursor": sampler.cursor, "checkpoint_steps_persisted": [step for step in CHECKPOINT_STEPS if (target / f"scientific_step_{step:06d}.pt").is_file()], "trace": history, "composition": trajectory},
        "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0},
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "hard_structural_firewall": {"descriptor_dimension": 337, "direct_optimization": False, "q95_modified": False, "topology_modified": False, "reference_modified": False},
        "s1_fixed_bank": {"winner": "S1", "winner_sha256": S1_WINNER_SHA256, "no_regeneration": True},
        "claim_boundary": {"joint_continuation_only": True, "semantic_recovery_claim": False, "architecture_acceptance": False, "physical_hardware_validity": False, "promotion": False, "new_confirmation_required": True},
        "counterfactual_integrity": {"same_step9000_checkpoint": True, "same_encoder_ols": True, "same_generator_architecture": True, "same_adam_state": True, "same_delta_joint": True, "same_beta_gamma": True, "same_evaluators": True, "E_D_in_backward": False, "same_scalar_reference_gradient": True, "no_hyperparameter_search": True, "no_sampler_refit": True, "no_checkpoint_selection": True, "no_protected_test_sealed": True},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "stage1_checkpoints": [9000, 9100, 9200, 9300], "stage2_checkpoints": [9300, 9600, 10000, 10500, 11000, 11500, 12000], "frozen_gradient_firewall": True, "no_protected_test_sealed": True},
        "timing": timing,
        "source_snapshot": source_snapshot,
        "unresolved": ["This bounded continuation does not establish final generation quality, human perceptual equivalence, hardware readiness, or formal promotion."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    try:
        heartbeat.raise_liveness_error()
        _persist_report(target=target, report=report, heartbeat=heartbeat, control=control, step=sampler.cursor, classification=classification)
    finally:
        heartbeat.stop_liveness()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--resume-checkpoint", default=None, help="explicit operational recovery checkpoint path; latest checkpoint is never inferred")
    parser.add_argument("--diagnostic-preflight-dir", default=None, help="run the real three-step CUDA path in a new diagnostic-only directory")
    args = parser.parse_args()
    if args.diagnostic_preflight_dir is not None:
        if args.resume_checkpoint is not None:
            parser.error("--diagnostic-preflight-dir cannot be combined with --resume-checkpoint")
        result = run_diagnostic_preflight(output_dir=args.diagnostic_preflight_dir)
    else:
        result = run_training(output_dir=args.output_dir, resume_checkpoint=args.resume_checkpoint)
    print(json.dumps({"result": result.get("result"), "classification": result.get("classification"), "output_dir": result.get("output_dir")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
