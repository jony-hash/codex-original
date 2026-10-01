"""Small operational lifecycle helpers shared by research executors.

The caller validates the actual checkpoint, model, optimizer, config and data
rows. Source and digest fields are historical provenance, never admission or
resume locks.
"""

from __future__ import annotations

import hashlib
import json
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load


HEARTBEAT_INTERVAL_SECONDS = 60.0
STALE_WARNING_SECONDS = 300.0
RECOVERY_INTERVAL_SECONDS = 600.0
RECOVERY_INTERVAL_STEPS = 25
RECOVERY_ROTATION_LIMIT = 3

PREFLIGHT_PENDING = "PENDING"
PREFLIGHT_READY = "READY"
PREFLIGHT_BLOCKED = "BLOCKED"
PREFLIGHT_RECHECK_ALLOWED = "RECHECK_ALLOWED"
PREFLIGHT_ROOT_CAUSE_REVIEW_REQUIRED = "ROOT_CAUSE_REVIEW_REQUIRED"

CORRECTNESS_INVARIANTS = "CORRECTNESS_INVARIANTS"
RESOURCE_ENVELOPE = "RESOURCE_ENVELOPE"
IMPLEMENTATION_SELECTION = "IMPLEMENTATION_SELECTION"
PREFLIGHT_CHECK_GROUPS = frozenset({CORRECTNESS_INVARIANTS, RESOURCE_ENVELOPE, IMPLEMENTATION_SELECTION})

RUN_PREPARING = "PREPARING"
RUNNING = "RUNNING"
RUN_SCIENTIFIC_ACTIVE = "SCIENTIFIC_ACTIVE"
RUN_COMPLETED = "COMPLETED"
RUN_VALID_FAILED = "VALID_FAILED"
RUN_INVALID_HARNESS = "INVALID_HARNESS"
RUN_INTERRUPTED = "INTERRUPTED"

PHASES = frozenset(
    {
        "PREPARING",
        "STATISTICS",
        "CALIBRATION",
        "TRAINING",
        "POST_TRAIN_COLLECTION",
        "POST_TRAIN_FIT",
        "FINALIZING",
        "SCIENTIFIC_ACTIVE",
        "COMPLETED",
        "VALID_FAILED",
        "INVALID_HARNESS",
        "INTERRUPTED",
    }
)
RUN_STATES = frozenset(
    {
        RUN_PREPARING,
        RUNNING,
        RUN_SCIENTIFIC_ACTIVE,
        RUN_COMPLETED,
        RUN_VALID_FAILED,
        RUN_INVALID_HARNESS,
        RUN_INTERRUPTED,
    }
)

SOURCE_MANIFEST_SCHEMA = "research_execution_source_manifest.v1"
RUN_IDENTITY_SCHEMA = "research_run_identity.v1"
PREFLIGHT_RECEIPT_SCHEMA = "research_preflight_receipt.v1"
RECOVERY_CHECKPOINT_SCHEMA = "research_recovery_checkpoint.v1"
PROGRESS_SCHEMA = "research_progress.v1"
class RunControlError(ValueError):
    """Raised when operational run state cannot be safely consumed."""


class ResumeCompatibilityError(RunControlError):
    """Raised when execution semantics differ across a resume boundary."""


def _require(condition: bool, message: str, error_type: type[Exception] = RunControlError) -> None:
    if not condition:
        raise error_type(message)


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunControlError(f"unable to capture Git identity: git {' '.join(arguments)}") from error
    return completed.stdout.strip()


def _relative_file(path: Path, root: Path) -> tuple[str, Path]:
    target = path if path.is_absolute() else root / path
    target = target.resolve()
    try:
        relative = target.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RunControlError(f"execution source is outside repository: {target}") from error
    _require(target.is_file(), f"execution source file is missing: {target}")
    return relative, target


@dataclass(frozen=True)
class ExecutionSourceManifest:
    """One compact identity for the source semantics used by a run."""

    task_id: str
    semantic_authority_id: str
    branch: str
    head: str
    dirty: bool
    relevant_source_paths: tuple[str, ...]
    relevant_source_identity: str
    dirty_patch_digest: str
    untracked_execution_files: tuple[str, ...]
    config_identity: Any
    execution_source_semantics_id: str
    source_manifest_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "task_id": self.task_id,
            "semantic_authority_id": self.semantic_authority_id,
            "branch": self.branch,
            "head": self.head,
            "dirty": self.dirty,
            "relevant_source_paths": list(self.relevant_source_paths),
            "relevant_source_identity": self.relevant_source_identity,
            "dirty_patch_digest": self.dirty_patch_digest,
            "untracked_execution_files": list(self.untracked_execution_files),
            "config_identity": self.config_identity,
            "execution_source_semantics_id": self.execution_source_semantics_id,
            "source_manifest_id": self.source_manifest_id,
        }


def capture_execution_source_manifest(
    repo_root: str | Path,
    *,
    task_id: str,
    semantic_authority_id: str,
    relevant_paths: Iterable[str | Path],
    config_identity: Any,
) -> ExecutionSourceManifest:
    """Capture only the source files that can affect the named execution.

    Branch, HEAD, and dirty state are provenance fields.  Resume compatibility
    uses ``execution_source_semantics_id`` so an unrelated document edit does
    not invalidate an otherwise identical execution source.
    """

    root = Path(repo_root).resolve()
    _require(root.is_dir(), f"repository root is missing: {root}")
    _require(task_id != "" and semantic_authority_id != "", "source manifest authority fields are required")
    resolved = [_relative_file(Path(path), root) for path in relevant_paths]
    _require(bool(resolved), "at least one relevant execution source is required")
    entries = [
        {"path": relative, "sha256": file_sha256(target)}
        for relative, target in sorted(resolved, key=lambda item: item[0])
    ]
    source_identity = canonical_sha256({"files": entries})
    relative_paths = tuple(entry["path"] for entry in entries)
    status = _git_output(root, "status", "--short", "--", *relative_paths)
    status_lines = tuple(line for line in status.splitlines() if line.strip())
    untracked = tuple(
        sorted(
            line[3:].strip()
            for line in status_lines
            if line.startswith("?? ") and line[3:].strip() in relative_paths
        )
    )
    unstaged_patch = _git_output(root, "diff", "--binary", "--", *relative_paths)
    staged_patch = _git_output(root, "diff", "--cached", "--binary", "--", *relative_paths)
    untracked_content = {
        relative: next(target for candidate, target in resolved if candidate == relative)
        for relative in untracked
    }
    dirty_patch_digest = canonical_sha256(
        {
            "unstaged_patch": hashlib.sha256(unstaged_patch.encode("utf-8")).hexdigest(),
            "staged_patch": hashlib.sha256(staged_patch.encode("utf-8")).hexdigest(),
            "untracked_content": {
                relative: file_sha256(target) for relative, target in sorted(untracked_content.items())
            },
        }
    )
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    dirty = bool(_git_output(root, "status", "--short"))
    semantics_payload = {
        "task_id": task_id,
        "semantic_authority_id": semantic_authority_id,
        "relevant_source_identity": source_identity,
        "dirty_patch_digest": dirty_patch_digest,
        "untracked_execution_files": list(untracked),
        "config_identity": config_identity,
    }
    semantics_id = canonical_sha256(semantics_payload)
    manifest_payload = {
        **semantics_payload,
        "schema": SOURCE_MANIFEST_SCHEMA,
        "branch": branch,
        "head": head,
        "dirty": dirty,
        "relevant_source_paths": list(relative_paths),
        "execution_source_semantics_id": semantics_id,
    }
    return ExecutionSourceManifest(
        task_id,
        semantic_authority_id,
        branch,
        head,
        dirty,
        relative_paths,
        source_identity,
        dirty_patch_digest,
        untracked,
        config_identity,
        semantics_id,
        canonical_sha256(manifest_payload),
    )


@dataclass(frozen=True)
class ResearchRunIdentity:
    """Research identity based on task, model, config and data roles.

    Source and manifest identifiers are retained in serialized records for
    historical provenance. They do not decide admission or resume.
    """

    task_id: str
    run_id: str
    trajectory_id: str
    semantic_authority_id: str
    data_role: str
    config_identity: Any
    model_identity: Any
    optimizer_identity: Any
    sampler_identity: Any
    execution_source_semantics_id: str = ""
    source_manifest_id: str | None = None
    scientific_attempt_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "run_id",
            "trajectory_id",
            "semantic_authority_id",
            "data_role",
        ):
            _require(bool(getattr(self, name)), f"run identity field is empty: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_IDENTITY_SCHEMA,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "trajectory_id": self.trajectory_id,
            "semantic_authority_id": self.semantic_authority_id,
            "data_role": self.data_role,
            "config_identity": self.config_identity,
            "model_identity": self.model_identity,
            "optimizer_identity": self.optimizer_identity,
            "sampler_identity": self.sampler_identity,
            "execution_source_semantics_id": self.execution_source_semantics_id,
            "source_manifest_id": self.source_manifest_id,
            "scientific_attempt_id": self.scientific_attempt_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResearchRunIdentity":
        _require(value.get("schema") in (None, RUN_IDENTITY_SCHEMA), "run identity schema mismatch", ResumeCompatibilityError)
        required = (
            "task_id",
            "run_id",
            "trajectory_id",
            "semantic_authority_id",
            "data_role",
            "config_identity",
            "model_identity",
            "optimizer_identity",
            "sampler_identity",
        )
        _require(all(key in value for key in required), "run identity is incomplete", ResumeCompatibilityError)
        return cls(
            str(value["task_id"]),
            str(value["run_id"]),
            str(value["trajectory_id"]),
            str(value["semantic_authority_id"]),
            str(value["data_role"]),
            value["config_identity"],
            value["model_identity"],
            value["optimizer_identity"],
            value["sampler_identity"],
            str(value.get("execution_source_semantics_id") or ""),
            None if value.get("source_manifest_id") is None else str(value["source_manifest_id"]),
            None if value.get("scientific_attempt_id") is None else str(value["scientific_attempt_id"]),
        )

    def with_scientific_attempt(self, attempt_id: str) -> "ResearchRunIdentity":
        _require(bool(attempt_id), "scientific attempt identity is empty")
        if self.scientific_attempt_id is not None:
            _require(self.scientific_attempt_id == attempt_id, "scientific attempt identity cannot change", ResumeCompatibilityError)
            return self
        return replace(self, scientific_attempt_id=attempt_id)

    def execution_semantics(self) -> dict[str, Any]:
        """Fields that must still describe the same scientific computation."""

        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "trajectory_id": self.trajectory_id,
            "semantic_authority_id": self.semantic_authority_id,
            "data_role": self.data_role,
            "config_identity": self.config_identity,
            "model_identity": self.model_identity,
            "optimizer_identity": self.optimizer_identity,
            "sampler_identity": self.sampler_identity,
            "scientific_attempt_id": self.scientific_attempt_id,
        }


def check_resume_compatibility(
    expected: ResearchRunIdentity,
    actual: ResearchRunIdentity | Mapping[str, Any],
) -> dict[str, Any]:
    """Reject semantic drift while ignoring unrelated human/document drift."""

    observed = actual if isinstance(actual, ResearchRunIdentity) else ResearchRunIdentity.from_mapping(actual)
    expected_fields = expected.execution_semantics()
    actual_fields = observed.execution_semantics()
    for field_name, expected_value in expected_fields.items():
        expected_value = _resume_semantics(expected_value)
        actual_value = _resume_semantics(actual_fields[field_name])
        _require(
            actual_value == expected_value,
            f"resume execution identity changed: {field_name}",
            ResumeCompatibilityError,
        )
    return {
        "compatible": True,
        "source_manifest_id": "NOT_REQUIRED",
        "branch_head_dirty": "provenance_only",
    }


def _resume_semantics(value: Any) -> Any:
    """Ignore recorded digests and repository state while retaining run settings."""
    if isinstance(value, Mapping):
        provenance_keys = {
            "branch",
            "head",
            "dirty",
            "worktree_dirty",
            "status_sha256",
            "source_manifest",
            "source_manifest_id",
            "execution_source_semantics_id",
            "source_semantics_id",
        }
        return {
            key: _resume_semantics(item)
            for key, item in value.items()
            if str(key).lower() not in provenance_keys
            and str(key).lower() not in {"sha256", "hash", "digest", "dirty_patch_digest"}
            and not str(key).lower().endswith(("_sha256", "_hash", "_digest"))
        }
    if isinstance(value, (list, tuple)):
        return [_resume_semantics(item) for item in value]
    return value


@dataclass(frozen=True, init=False)
class PreflightReceipt:
    """Write-once contract consumed by the strict scientific lifecycle.

    The receipt carries the execution source boundary and authorization that
    produced it.  The three check groups are deliberately data rather than
    additional gates: they explain why the single scientific Outer Gate was
    allowed to open and which implementation choice is frozen for the run.
    """

    source_semantics_id: str
    authorization_id: str
    checks: Mapping[str, Any]
    selected_engine: str | None = None
    resource: Mapping[str, Any] = field(default_factory=dict)
    failure_signature: str | None = None
    recheck_count: int = 0
    status: str = PREFLIGHT_READY
    created_at: float | None = None

    def __init__(
        self,
        source_semantics_id: str | None = None,
        authorization_id: str | None = None,
        checks: Mapping[str, Any] | None = None,
        selected_engine: str | None = None,
        resource: Mapping[str, Any] | None = None,
        failure_signature: str | None = None,
        recheck_count: int = 0,
        status: str = PREFLIGHT_READY,
        created_at: float | None = None,
        *,
        execution_source_semantics_id: str | None = None,
        source_semantics: str | None = None,
        authorization: str | None = None,
        check_groups: Mapping[str, Any] | None = None,
        resource_envelope: Mapping[str, Any] | None = None,
    ) -> None:
        source = source_semantics_id or execution_source_semantics_id or source_semantics
        auth = authorization_id or authorization
        check_map = checks if checks is not None else check_groups
        resource_map = resource if resource is not None else resource_envelope
        object.__setattr__(self, "source_semantics_id", "" if source is None else str(source))
        object.__setattr__(self, "authorization_id", "" if auth is None else str(auth))
        object.__setattr__(self, "checks", {} if check_map is None else dict(check_map))
        object.__setattr__(self, "selected_engine", None if selected_engine is None else str(selected_engine))
        object.__setattr__(self, "resource", {} if resource_map is None else dict(resource_map))
        object.__setattr__(self, "failure_signature", None if failure_signature is None else str(failure_signature))
        object.__setattr__(self, "recheck_count", int(recheck_count))
        object.__setattr__(self, "status", str(status))
        object.__setattr__(self, "created_at", None if created_at is None else float(created_at))
        self.__post_init__()

    def __post_init__(self) -> None:
        _require(bool(self.authorization_id), "preflight authorization identity is required")
        _require(self.status in {PREFLIGHT_READY, PREFLIGHT_BLOCKED}, "invalid preflight receipt status")
        _require(self.recheck_count >= 0, "preflight recheck count must be nonnegative")
        check_keys = set(self.checks)
        _require(
            check_keys.issuperset(PREFLIGHT_CHECK_GROUPS),
            "preflight receipt must record correctness, resource, and implementation checks",
        )
        _require(isinstance(self.resource, Mapping), "preflight resource envelope must be a mapping")

    @property
    def execution_source_semantics_id(self) -> str:
        """Compatibility spelling used by source manifests and identities."""

        return self.source_semantics_id

    @property
    def check_groups(self) -> Mapping[str, Any]:
        return self.checks

    @property
    def resource_envelope(self) -> Mapping[str, Any]:
        return self.resource

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PREFLIGHT_RECEIPT_SCHEMA,
            "source_semantics_id": self.source_semantics_id,
            "execution_source_semantics_id": self.source_semantics_id,
            "authorization_id": self.authorization_id,
            "checks": _jsonable(self.checks),
            "check_groups": _jsonable(self.checks),
            "selected_engine": self.selected_engine,
            "resource": _jsonable(self.resource),
            "resource_envelope": _jsonable(self.resource),
            "failure_signature": self.failure_signature,
            "recheck_count": self.recheck_count,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreflightReceipt":
        _require(value.get("schema") in (None, PREFLIGHT_RECEIPT_SCHEMA), "preflight receipt schema mismatch", ResumeCompatibilityError)
        source = value.get("source_semantics_id", value.get("execution_source_semantics_id"))
        checks = value.get("checks", value.get("check_groups"))
        resource = value.get("resource", value.get("resource_envelope", {}))
        _require(isinstance(value.get("authorization_id"), str) and value["authorization_id"], "preflight receipt authorization is missing", ResumeCompatibilityError)
        _require(isinstance(checks, Mapping), "preflight receipt checks are missing", ResumeCompatibilityError)
        return cls(
            source_semantics_id=source if isinstance(source, str) else "",
            authorization_id=str(value["authorization_id"]),
            checks=dict(checks),
            selected_engine=None if value.get("selected_engine") is None else str(value["selected_engine"]),
            resource=dict(resource) if isinstance(resource, Mapping) else {},
            failure_signature=None if value.get("failure_signature") is None else str(value["failure_signature"]),
            recheck_count=int(value.get("recheck_count", 0)),
            status=str(value.get("status", PREFLIGHT_READY)),
            created_at=None if value.get("created_at") is None else float(value["created_at"]),
        )


@dataclass
class ResearchRunControl:
    """One authorization and one task identity through preflight to execution."""

    identity: ResearchRunIdentity
    authorization_id: str
    preflight_status: str = PREFLIGHT_PENDING
    state: str = RUN_PREPARING
    receipt: PreflightReceipt | None = None
    preflight_failure_counts: dict[str, int] = field(default_factory=dict)
    failure_signature: str | None = None
    recheck_count: int = 0

    def __post_init__(self) -> None:
        _require(bool(self.authorization_id), "one execution authorization is required")
        _require(self.preflight_status in {PREFLIGHT_PENDING, PREFLIGHT_READY, PREFLIGHT_BLOCKED}, "invalid preflight status")
        _require(self.state in RUN_STATES, "invalid run state")
        if self.receipt is not None:
            self._validate_receipt(self.receipt)

    def _validate_receipt(self, receipt: PreflightReceipt) -> None:
        _require(receipt.authorization_id == self.authorization_id, "preflight authorization does not match task authorization", ResumeCompatibilityError)
        _require(receipt.status == PREFLIGHT_READY, "scientific execution requires a READY preflight receipt")

    @staticmethod
    def _coerce_receipt(receipt: PreflightReceipt | Mapping[str, Any]) -> PreflightReceipt:
        return receipt if isinstance(receipt, PreflightReceipt) else PreflightReceipt.from_mapping(receipt)

    def record_preflight_failure(self, component: str, signature: str) -> str:
        """Record one deterministic engineering failure without spending science.

        Failure counts are descriptive history only. Repeated engineering
        defects do not impose a retry quota; the affected production path still
        has to be repaired and verified before use.
        """

        _require(self.state == RUN_PREPARING, "preflight failures can only be recorded before execution")
        _require(bool(component) and bool(signature), "preflight failure component and signature are required")
        key = f"{component}:{signature}"
        count = int(self.preflight_failure_counts.get(key, 0)) + 1
        self.preflight_failure_counts[key] = count
        self.preflight_status = PREFLIGHT_BLOCKED
        self.receipt = None
        self.failure_signature = key
        self.recheck_count = count
        return PREFLIGHT_RECHECK_ALLOWED

    def mark_preflight_blocked(self) -> None:
        _require(self.state == RUN_PREPARING, "preflight can only be rechecked before execution")
        self.preflight_status = PREFLIGHT_BLOCKED
        self.receipt = None

    def mark_preflight_ready(self, receipt: PreflightReceipt | Mapping[str, Any] | None = None) -> None:
        _require(self.state == RUN_PREPARING, "preflight can only be completed before execution")
        if receipt is not None:
            normalized = self._coerce_receipt(receipt)
            if self.preflight_status == PREFLIGHT_READY and self.receipt is not None:
                _require(normalized == self.receipt, "READY preflight receipt is write-once", ResumeCompatibilityError)
            self._validate_receipt(normalized)
            self.receipt = normalized
            self.failure_signature = normalized.failure_signature
            self.recheck_count = normalized.recheck_count
        self.preflight_status = PREFLIGHT_READY

    def blocked_preflight_receipt(self) -> PreflightReceipt:
        """Return the non-consumable receipt describing the latest block."""

        _require(self.preflight_status == PREFLIGHT_BLOCKED, "no blocked preflight is available")
        return PreflightReceipt(
            source_semantics_id=self.identity.execution_source_semantics_id,
            authorization_id=self.authorization_id,
            checks={
                CORRECTNESS_INVARIANTS: {"status": "BLOCKED"},
                RESOURCE_ENVELOPE: {"status": "BLOCKED"},
                IMPLEMENTATION_SELECTION: {"status": "BLOCKED"},
            },
            selected_engine=None,
            resource={},
            failure_signature=self.failure_signature,
            recheck_count=self.recheck_count,
            status=PREFLIGHT_BLOCKED,
        )

    def start_execution(self) -> None:
        _require(self.preflight_status == PREFLIGHT_READY, "execution requires the same task's READY preflight")
        _require(self.state == RUN_PREPARING, "execution has already started")
        self.state = RUNNING

    def open_scientific_attempt(
        self,
        output_dir: str | Path,
        attempt_id: str | None = None,
        receipt: PreflightReceipt | Mapping[str, Any] | None = None,
        *,
        resume: bool = False,
    ) -> ResearchRunIdentity:
        """Atomically enter ``SCIENTIFIC_ACTIVE`` for one frozen attempt.

        ``resume=True`` only reopens an existing directory after an interruption;
        it never creates a new attempt identity.  The legacy
        :meth:`begin_scientific_attempt` remains available for closed historical
        executors but is intentionally not used by active executors.
        """

        chosen_receipt = self.receipt if receipt is None else self._coerce_receipt(receipt)
        _require(chosen_receipt is not None, "strict scientific attempt requires a complete READY receipt")
        self._validate_receipt(chosen_receipt)
        _require(self.preflight_status == PREFLIGHT_READY, "scientific attempt requires READY preflight")
        _require(self.state in {RUN_PREPARING, RUNNING, RUN_INTERRUPTED}, "scientific attempt cannot be reopened from this state")
        target = Path(output_dir).resolve()
        chosen = attempt_id or self.identity.scientific_attempt_id or f"{self.identity.run_id}:scientific"
        effective = self.identity.with_scientific_attempt(chosen)
        resuming = resume or self.state == RUN_INTERRUPTED
        if resuming:
            _require(target.is_dir(), f"recovery output directory is missing: {target}")
        else:
            _require(not target.exists(), f"scientific output directory already exists: {target}")
            target.mkdir(parents=True, exist_ok=False)
        self.receipt = chosen_receipt
        self.identity = effective
        self.state = RUN_SCIENTIFIC_ACTIVE
        return effective

    def resume_scientific_attempt(
        self,
        output_dir: str | Path,
        attempt_id: str,
        receipt: PreflightReceipt | Mapping[str, Any] | None = None,
    ) -> ResearchRunIdentity:
        return self.open_scientific_attempt(output_dir, attempt_id, receipt, resume=True)

    def begin_scientific_attempt(self, attempt_id: str | None = None) -> ResearchRunIdentity:
        _require(self.state == RUNNING, "scientific attempt begins only while running")
        chosen = attempt_id or f"{self.identity.run_id}:scientific"
        self.identity = self.identity.with_scientific_attempt(chosen)
        return self.identity

    def complete(self) -> None:
        _require(self.state in {RUNNING, RUN_SCIENTIFIC_ACTIVE}, "only an active task can complete")
        self.state = RUN_COMPLETED

    def valid_failed(self) -> None:
        _require(self.state in {RUNNING, RUN_SCIENTIFIC_ACTIVE}, "only an active task can become valid_failed")
        self.state = RUN_VALID_FAILED

    def invalid_harness(self) -> None:
        _require(self.state in {RUN_PREPARING, RUNNING, RUN_SCIENTIFIC_ACTIVE}, "invalid harness is already terminal")
        self.state = RUN_INVALID_HARNESS

    def interrupted(self) -> None:
        _require(self.state in {RUNNING, RUN_SCIENTIFIC_ACTIVE, RUN_PREPARING}, "invalid interrupted transition")
        self.state = RUN_INTERRUPTED


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise RunControlError(f"progress value is not JSON serializable: {type(value)!r}")


@dataclass
class HeartbeatWriter:
    run_dir: str | Path
    run_id: str
    total_steps: int
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS
    stale_warning_seconds: float = STALE_WARNING_SECONDS
    clock: Callable[[], float] = time.time
    _started_at: float | None = field(default=None, init=False)
    _last_progress_written: float | None = field(default=None, init=False)
    _last_heartbeat_written: float | None = field(default=None, init=False)
    _last_progress_time: float | None = field(default=None, init=False)
    _last_recovery_checkpoint: str | None = field(default=None, init=False)
    _state: str | None = field(default=None, init=False)
    _phase: str | None = field(default=None, init=False)
    _step: int | None = field(default=None, init=False)
    _latest_losses: dict[str, Any] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _liveness_stop: threading.Event | None = field(default=None, init=False, repr=False)
    _liveness_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _liveness_error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _require(bool(self.run_id), "heartbeat run_id is required")
        _require(self.total_steps >= 0, "heartbeat total_steps must be nonnegative")
        _require(self.interval_seconds >= 0.0 and self.stale_warning_seconds > 0.0, "heartbeat intervals are invalid")
        if self.latest_status_path.exists():
            try:
                existing = json.loads(self.latest_status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RunControlError("existing heartbeat status cannot be consumed") from error
            _require(isinstance(existing, Mapping), "existing heartbeat status is not a mapping")
            _require(existing.get("run_id") == self.run_id, "existing heartbeat run identity changed", ResumeCompatibilityError)
            _require(existing.get("schema") == PROGRESS_SCHEMA, "existing heartbeat schema changed", ResumeCompatibilityError)
            _require(existing.get("state") in RUN_STATES, "existing heartbeat state is invalid", ResumeCompatibilityError)
            _require(existing.get("phase") in PHASES, "existing heartbeat phase is invalid", ResumeCompatibilityError)
            self._state = str(existing["state"])
            self._phase = str(existing["phase"])
            self._step = int(existing["step"])
            self._latest_losses = dict(existing.get("latest_losses") or {})
            checkpoint = existing.get("last_recovery_checkpoint")
            self._last_recovery_checkpoint = None if checkpoint is None else str(checkpoint)
            last_progress = existing.get("last_progress_time")
            if isinstance(last_progress, (int, float)):
                self._last_progress_time = float(last_progress)
            last_heartbeat = existing.get("last_heartbeat_time", last_progress)
            if isinstance(last_heartbeat, (int, float)):
                self._last_heartbeat_written = float(last_heartbeat)
            wall_time = existing.get("wall_time")
            if isinstance(last_progress, (int, float)) and isinstance(wall_time, (int, float)):
                self._started_at = float(last_progress) - max(0.0, float(wall_time))
            self._last_progress_written = self._last_progress_time

    @property
    def progress_path(self) -> Path:
        return self.run_dir / "progress.jsonl"

    @property
    def latest_status_path(self) -> Path:
        return self.run_dir / "latest_status.json"

    def update(
        self,
        *,
        state: str,
        phase: str,
        step: int,
        latest_losses: Mapping[str, Any] | None = None,
        last_recovery_checkpoint: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        with self._lock:
            now = float(self.clock())
            if not force and self._last_progress_written is not None and now - self._last_progress_written < self.interval_seconds:
                return None
            return self._write_locked(
                state=state,
                phase=phase,
                step=step,
                latest_losses=latest_losses,
                last_recovery_checkpoint=last_recovery_checkpoint,
                now=now,
                scientific_progress=True,
            )

    def remember_recovery_checkpoint(self, checkpoint: str | Path) -> None:
        """Remember a successfully committed operational checkpoint.

        This only changes the next status payload.  It never reads or writes a
        checkpoint, so it is safe to call at the caller's checkpoint boundary.
        """

        with self._lock:
            self._last_recovery_checkpoint = str(checkpoint)

    def heartbeat(
        self,
        *,
        state: str | None = None,
        phase: str | None = None,
        step: int | None = None,
        latest_losses: Mapping[str, Any] | None = None,
        last_recovery_checkpoint: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Write liveness only; it never appends scientific progress.

        The background liveness thread calls this method.  It performs only a
        small atomic JSON replacement and does not touch CUDA, models, or
        checkpoints.  Scientific progress remains the responsibility of
        :meth:`update`.
        """

        with self._lock:
            now = float(self.clock())
            if not force and self._last_heartbeat_written is not None and now - self._last_heartbeat_written < self.interval_seconds:
                return None
            chosen_state = self._state if state is None else state
            chosen_phase = self._phase if phase is None else phase
            chosen_step = self._step if step is None else step
            _require(chosen_state is not None and chosen_phase is not None and chosen_step is not None, "heartbeat has no established run state")
            return self._write_locked(
                state=chosen_state,
                phase=chosen_phase,
                step=chosen_step,
                latest_losses=latest_losses,
                last_recovery_checkpoint=last_recovery_checkpoint,
                now=now,
                scientific_progress=False,
            )

    def _write_locked(
        self,
        *,
        state: str,
        phase: str,
        step: int,
        latest_losses: Mapping[str, Any] | None,
        last_recovery_checkpoint: str | Path | None,
        now: float,
        scientific_progress: bool,
    ) -> dict[str, Any]:
        _require(state in RUN_STATES, f"heartbeat state is invalid: {state}")
        _require(phase in PHASES, f"heartbeat phase is invalid: {phase}")
        _require(0 <= step <= self.total_steps, "heartbeat step is outside total_steps")
        if self._started_at is None:
            self._started_at = now
        if last_recovery_checkpoint is not None:
            self._last_recovery_checkpoint = str(last_recovery_checkpoint)
        if latest_losses is not None:
            self._latest_losses = dict(latest_losses)
        progress_time = now if scientific_progress or self._last_progress_time is None else self._last_progress_time
        record = {
            "schema": PROGRESS_SCHEMA,
            "run_id": self.run_id,
            "state": state,
            "phase": phase,
            "step": int(step),
            "total_steps": int(self.total_steps),
            "wall_time": max(0.0, now - self._started_at),
            "last_progress_time": progress_time,
            "last_heartbeat_time": now,
            "last_recovery_checkpoint": self._last_recovery_checkpoint,
            "latest_losses": _jsonable(self._latest_losses),
        }
        if scientific_progress:
            with self.progress_path.open("a", encoding="utf-8") as stream:
                json.dump(record, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
            self._last_progress_written = now
            self._last_progress_time = now
        self._state = state
        self._phase = phase
        self._step = int(step)
        self._last_heartbeat_written = now
        atomic_write_json(self.latest_status_path, record)
        return record

    def start_liveness(self, *, interval_seconds: float | None = None) -> None:
        """Start a daemon thread that writes only lightweight liveness state."""

        interval = self.interval_seconds if interval_seconds is None else float(interval_seconds)
        _require(interval > 0.0, "liveness interval must be positive")
        with self._lock:
            if self._liveness_thread is not None and self._liveness_thread.is_alive():
                return
            stop = threading.Event()
            self._liveness_stop = stop
            self._liveness_error = None

            def run() -> None:
                while not stop.wait(interval):
                    try:
                        self.heartbeat(force=True)
                    except BaseException as error:  # record and stop; surface at a safe boundary
                        with self._lock:
                            self._liveness_error = error
                        return

            self._liveness_thread = threading.Thread(target=run, name=f"heartbeat-{self.run_id}", daemon=True)
            self._liveness_thread.start()

    def stop_liveness(self) -> None:
        with self._lock:
            stop = self._liveness_stop
            thread = self._liveness_thread
            self._liveness_stop = None
            self._liveness_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, 2.0 * self.interval_seconds))

    def raise_liveness_error(self) -> None:
        with self._lock:
            error = self._liveness_error
        if error is not None:
            raise RunControlError("background heartbeat failed") from error

    def stale_warning(self, *, now: float | None = None) -> str | None:
        observed_now = float(self.clock() if now is None else now)
        if not self.latest_status_path.exists():
            if self._started_at is not None and observed_now - self._started_at > self.stale_warning_seconds:
                return "LONG_RUN_HEARTBEAT_STALE"
            return None
        try:
            status = json.loads(self.latest_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "LONG_RUN_HEARTBEAT_STALE"
        last_heartbeat = status.get("last_heartbeat_time", status.get("last_progress_time"))
        if not isinstance(last_heartbeat, (int, float)):
            return "LONG_RUN_HEARTBEAT_STALE"
        return "LONG_RUN_HEARTBEAT_STALE" if observed_now - float(last_heartbeat) > self.stale_warning_seconds else None

    def scientific_progress_stale_warning(self, *, now: float | None = None) -> str | None:
        """Report a stale scientific update independently of liveness."""

        observed_now = float(self.clock() if now is None else now)
        if not self.latest_status_path.exists():
            return "SCIENTIFIC_PROGRESS_STALE"
        try:
            status = json.loads(self.latest_status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "SCIENTIFIC_PROGRESS_STALE"
        last_progress = status.get("last_progress_time")
        if not isinstance(last_progress, (int, float)) or observed_now - float(last_progress) > self.stale_warning_seconds:
            return "SCIENTIFIC_PROGRESS_STALE"
        return None


def recovery_due(
    *,
    step: int,
    last_recovery_step: int,
    now: float,
    last_recovery_time: float | None,
    interval_steps: int = RECOVERY_INTERVAL_STEPS,
    interval_seconds: float = RECOVERY_INTERVAL_SECONDS,
) -> bool:
    """Return true when either fixed recovery interval is reached."""

    _require(step >= last_recovery_step >= 0, "recovery step ordering is invalid")
    _require(interval_steps > 0 and interval_seconds > 0.0, "recovery intervals are invalid")
    by_step = step - last_recovery_step >= interval_steps
    by_time = last_recovery_time is not None and now - last_recovery_time >= interval_seconds
    return by_step or by_time


_RECOVERY_PATTERN = re.compile(r"^recovery_(\d{6})_step_(\d{6})\.pt$")


@dataclass
class RecoveryCheckpointManager:
    run_dir: str | Path
    run_identity: ResearchRunIdentity
    source_manifest_id: str | None = None
    keep: int = RECOVERY_ROTATION_LIMIT
    authorization_id: str | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _require(self.keep > 0, "recovery rotation limit must be positive")
        if self.authorization_id is not None:
            _require(bool(self.authorization_id), "recovery authorization identity is empty")

    def _paths(self) -> list[Path]:
        paths = [path for path in self.run_dir.glob("recovery_*.pt") if _RECOVERY_PATTERN.match(path.name)]
        return sorted(paths, key=lambda path: int(_RECOVERY_PATTERN.match(path.name).group(1)))

    def save(self, payload: Mapping[str, Any], *, step: int) -> Path:
        _require(isinstance(payload, Mapping), "recovery payload must be a mapping")
        _require(step >= 0, "recovery step must be nonnegative")
        paths = self._paths()
        sequence = int(_RECOVERY_PATTERN.match(paths[-1].name).group(1)) + 1 if paths else 1
        target = self.run_dir / f"recovery_{sequence:06d}_step_{step:06d}.pt"
        wrapper = {
            "schema": RECOVERY_CHECKPOINT_SCHEMA,
            "artifact_kind": "OPERATIONAL_RECOVERY_CHECKPOINT",
            "authorization_id": self.authorization_id,
            "run_identity": self.run_identity.to_dict(),
            "source_manifest_id": self.source_manifest_id,
            "step": int(step),
            "payload": dict(payload),
        }
        atomic_torch_save_new(target, wrapper, artifact_kind="operational recovery checkpoint")
        self.rotate()
        return target

    def rotate(self) -> None:
        paths = self._paths()
        for path in paths[:-self.keep]:
            path.unlink()

    def latest(self) -> Path | None:
        paths = self._paths()
        return paths[-1] if paths else None

    def load(self, path: str | Path) -> Mapping[str, Any]:
        target = Path(path).resolve()
        _require(target.parent == self.run_dir and _RECOVERY_PATTERN.match(target.name) is not None, "recovery path is outside its run directory")
        payload = safe_torch_load(target, map_location="cpu")
        _require(isinstance(payload, Mapping) and payload.get("schema") == RECOVERY_CHECKPOINT_SCHEMA, "recovery checkpoint schema mismatch", ResumeCompatibilityError)
        if self.authorization_id is not None:
            _require(payload.get("authorization_id") == self.authorization_id, "recovery authorization identity changed", ResumeCompatibilityError)
        check_resume_compatibility(self.run_identity, payload.get("run_identity", {}))
        # The manifest ID and source semantics are provenance only. Resume
        # compatibility is decided from the run authorization and actual
        # scientific state loaded by the consumer.
        return payload


@dataclass
class InterruptController:
    """Signal-aware request flag; I/O occurs only at a caller's safe boundary."""

    heartbeat: HeartbeatWriter
    recovery: RecoveryCheckpointManager
    payload_factory: Callable[[int], Mapping[str, Any]]
    phase: str
    total_steps: int
    _requested: bool = field(default=False, init=False)
    _previous_handlers: dict[int, Any] = field(default_factory=dict, init=False)

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self) -> None:
        self._requested = True

    def _signal_handler(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.request()

    def install(self) -> "InterruptController":
        for signum in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if signum is not None:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
        return self

    def close(self) -> None:
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()

    def __enter__(self) -> "InterruptController":
        return self.install()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def handle_safe_boundary(
        self,
        *,
        step: int,
        latest_losses: Mapping[str, Any] | None = None,
        last_recovery_checkpoint: str | Path | None = None,
    ) -> Path | None:
        if not self.requested:
            return None
        checkpoint = self.recovery.save(self.payload_factory(step), step=step)
        self.heartbeat.update(
            state=RUN_INTERRUPTED,
            phase="INTERRUPTED",
            step=step,
            latest_losses=latest_losses,
            last_recovery_checkpoint=checkpoint.name,
            force=True,
        )
        return checkpoint


__all__ = [
    "CORRECTNESS_INVARIANTS",
    "ExecutionSourceManifest",
    "HeartbeatWriter",
    "HEARTBEAT_INTERVAL_SECONDS",
    "IMPLEMENTATION_SELECTION",
    "InterruptController",
    "PHASES",
    "PREFLIGHT_BLOCKED",
    "PREFLIGHT_CHECK_GROUPS",
    "PREFLIGHT_PENDING",
    "PREFLIGHT_RECHECK_ALLOWED",
    "PREFLIGHT_READY",
    "PREFLIGHT_ROOT_CAUSE_REVIEW_REQUIRED",
    "PREFLIGHT_RECEIPT_SCHEMA",
    "PreflightReceipt",
    "RECOVERY_CHECKPOINT_SCHEMA",
    "RECOVERY_INTERVAL_SECONDS",
    "RECOVERY_INTERVAL_STEPS",
    "RECOVERY_ROTATION_LIMIT",
    "RecoveryCheckpointManager",
    "ResearchRunControl",
    "ResearchRunIdentity",
    "ResumeCompatibilityError",
    "RunControlError",
    "RUN_COMPLETED",
    "RUN_INVALID_HARNESS",
    "RUN_INTERRUPTED",
    "RUN_PREPARING",
    "RUN_SCIENTIFIC_ACTIVE",
    "RUN_VALID_FAILED",
    "RUNNING",
    "RESOURCE_ENVELOPE",
    "STALE_WARNING_SECONDS",
    "SOURCE_MANIFEST_SCHEMA",
    "capture_execution_source_manifest",
    "check_resume_compatibility",
    "recovery_due",
]
