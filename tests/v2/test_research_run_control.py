"""Focused governance and long-run resilience checks without scientific execution."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from ldm_project.v2.application.research import research_run_control as control
from ldm_project.v2.evidence import atomic_torch_save_new
from scripts.research import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as executor


def _identity(
    *,
    source_semantics: str = "execution-semantics-a",
    source_manifest: str | None = "manifest-a",
    attempt: str | None = None,
) -> control.ResearchRunIdentity:
    return control.ResearchRunIdentity(
        task_id="TASK-001",
        run_id="run-001",
        trajectory_id="trajectory-001",
        semantic_authority_id="M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION",
        data_role="train",
        config_identity={"grid": 128, "updates": 300},
        model_identity="model-v1",
        optimizer_identity={"name": "Adam", "lr": 1.0e-3},
        sampler_identity={"seed": 1234, "steps": 300},
        execution_source_semantics_id=source_semantics,
        source_manifest_id=source_manifest,
        scientific_attempt_id=attempt,
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_one_authorization_flows_from_preflight_to_execution() -> None:
    identity = _identity()
    run = control.ResearchRunControl(identity, "authorization-001")
    run.mark_preflight_ready()
    run.start_execution()

    assert run.authorization_id == "authorization-001"
    assert run.identity == identity
    assert run.state == control.RUNNING
    with pytest.raises(control.RunControlError):
        run.start_execution()


def test_engineering_failure_is_rechecked_in_place() -> None:
    identity = _identity()
    run = control.ResearchRunControl(identity, "authorization-001")
    run.mark_preflight_blocked()
    run.mark_preflight_ready()
    run.start_execution()

    assert run.identity == identity
    assert run.authorization_id == "authorization-001"


def test_unrelated_provenance_does_not_invalidate_resume() -> None:
    expected = _identity(source_manifest="manifest-before")
    actual = _identity(source_manifest="manifest-after")

    result = control.check_resume_compatibility(expected, actual)

    assert result["compatible"] is True
    assert result["source_manifest_id"] == "NOT_REQUIRED"


def test_execution_critical_source_mismatch_rejects_resume() -> None:
    with pytest.raises(control.ResumeCompatibilityError):
        control.check_resume_compatibility(
            _identity(source_semantics="execution-semantics-a"),
            _identity(source_semantics="execution-semantics-b"),
        )


def test_heartbeat_writes_synthetic_long_loop_and_warns_without_restart() -> None:
    clock = _Clock()
    with tempfile.TemporaryDirectory(prefix="research-heartbeat-") as temporary:
        heartbeat = control.HeartbeatWriter(
            temporary,
            "run-001",
            total_steps=30,
            interval_seconds=60.0,
            clock=clock,
        )
        assert heartbeat.update(state=control.RUNNING, phase="TRAINING", step=0, force=True) is not None
        clock.value = 30.0
        assert heartbeat.update(state=control.RUNNING, phase="TRAINING", step=1) is None
        clock.value = 60.0
        assert heartbeat.update(state=control.RUNNING, phase="TRAINING", step=1) is not None
        clock.value = 361.0
        assert heartbeat.stale_warning() == "LONG_RUN_HEARTBEAT_STALE"
        records = heartbeat.progress_path.read_text(encoding="utf-8").splitlines()
        assert len(records) == 2


def test_liveness_heartbeat_does_not_advance_scientific_progress() -> None:
    clock = _Clock()
    with tempfile.TemporaryDirectory(prefix="research-liveness-") as temporary:
        heartbeat = control.HeartbeatWriter(temporary, "run-001", total_steps=30, interval_seconds=60.0, clock=clock)
        heartbeat.update(state=control.RUNNING, phase="TRAINING", step=4, latest_losses={"loss": 1.0}, force=True)
        clock.value = 120.0
        heartbeat.remember_recovery_checkpoint("recovery_000001_step_000004.pt")
        heartbeat.heartbeat(force=True)
        status = json.loads(heartbeat.latest_status_path.read_text(encoding="utf-8"))
        records = heartbeat.progress_path.read_text(encoding="utf-8").splitlines()

    assert len(records) == 1
    assert status["step"] == 4
    assert status["last_progress_time"] == 0.0
    assert status["last_heartbeat_time"] == 120.0
    assert status["last_recovery_checkpoint"] == "recovery_000001_step_000004.pt"


def test_liveness_thread_only_writes_status_and_can_be_stopped() -> None:
    with tempfile.TemporaryDirectory(prefix="research-liveness-thread-") as temporary:
        heartbeat = control.HeartbeatWriter(temporary, "run-001", total_steps=2, interval_seconds=0.01)
        heartbeat.update(state=control.RUNNING, phase="TRAINING", step=1, force=True)
        heartbeat.start_liveness(interval_seconds=0.01)
        time.sleep(0.04)
        heartbeat.stop_liveness()
        status = json.loads(heartbeat.latest_status_path.read_text(encoding="utf-8"))
        records = heartbeat.progress_path.read_text(encoding="utf-8").splitlines()

    assert status["step"] == 1
    assert status["last_heartbeat_time"] >= status["last_progress_time"]
    assert len(records) == 1


def test_existing_heartbeat_is_hydrated_and_identity_mismatch_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="research-heartbeat-hydrate-") as temporary:
        first = control.HeartbeatWriter(temporary, "run-001", total_steps=2, interval_seconds=0.0)
        first.update(state=control.RUNNING, phase="TRAINING", step=1, force=True)
        resumed = control.HeartbeatWriter(temporary, "run-001", total_steps=2, interval_seconds=0.0)
        assert resumed.heartbeat(force=True)["step"] == 1
        with pytest.raises(control.ResumeCompatibilityError):
            control.HeartbeatWriter(temporary, "different-run", total_steps=2, interval_seconds=0.0)


def test_latest_status_uses_atomic_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    original = control.atomic_write_json

    def observed(path: str | Path, payload: dict[str, object]) -> None:
        calls.append(Path(path))
        original(path, payload)

    monkeypatch.setattr(control, "atomic_write_json", observed)
    with tempfile.TemporaryDirectory(prefix="research-status-") as temporary:
        heartbeat = control.HeartbeatWriter(temporary, "run-001", total_steps=2, interval_seconds=0.0)
        heartbeat.update(state=control.RUNNING, phase="TRAINING", step=1, force=True)
        heartbeat.update(state=control.RUNNING, phase="TRAINING", step=2, force=True)
        status = json.loads(heartbeat.latest_status_path.read_text(encoding="utf-8"))

    assert calls and all(path.name == "latest_status.json" for path in calls)
    assert status["step"] == 2


def _interrupted_toy_run(run_dir: Path) -> tuple[Path, int]:
    identity = _identity(attempt="attempt-001")
    heartbeat = control.HeartbeatWriter(run_dir, identity.run_id, total_steps=30, interval_seconds=0.0)
    recovery = control.RecoveryCheckpointManager(run_dir, identity, authorization_id="authorization-001")
    state = {"value": 0}

    def payload_factory(step: int) -> dict[str, int]:
        return {"value": state["value"], "cursor": step}

    interrupt = control.InterruptController(
        heartbeat=heartbeat,
        recovery=recovery,
        payload_factory=payload_factory,
        phase="TRAINING",
        total_steps=30,
    )
    with interrupt:
        for step in range(1, 31):
            state["value"] += step
            heartbeat.update(state=control.RUNNING, phase="TRAINING", step=step, force=True)
            if step == 17:
                interrupt.request()
            checkpoint = interrupt.handle_safe_boundary(step=step)
            if checkpoint is not None:
                return checkpoint, state["value"]
    raise AssertionError("toy interruption did not produce a recovery checkpoint")


def test_interruption_recovery_preserves_same_trajectory() -> None:
    with tempfile.TemporaryDirectory(prefix="research-interruption-") as temporary:
        root = Path(temporary)
        checkpoint, interrupted_value = _interrupted_toy_run(root / "interrupted")
        identity = _identity(attempt="attempt-001")
        loaded = control.RecoveryCheckpointManager(
            root / "interrupted",
            identity,
            authorization_id="authorization-001",
        ).load(checkpoint)
        payload = loaded["payload"]
        resumed_value = int(payload["value"])
        for step in range(int(payload["cursor"]) + 1, 31):
            resumed_value += step

        uninterrupted_value = sum(range(1, 31))

    assert interrupted_value == sum(range(1, 18))
    assert resumed_value == uninterrupted_value


def test_recovery_rotation_keeps_three_and_never_deletes_scientific_file() -> None:
    identity = _identity(attempt="attempt-001")
    with tempfile.TemporaryDirectory(prefix="research-rotation-") as temporary:
        run_dir = Path(temporary)
        manager = control.RecoveryCheckpointManager(run_dir, identity)
        for step in range(5):
            manager.save({"step": step}, step=step)
        scientific = run_dir / "scientific_step_000300.pt"
        atomic_torch_save_new(scientific, {"scientific": True}, artifact_kind="scientific checkpoint")
        manager.save({"step": 5}, step=5)

        recovery_files = sorted(run_dir.glob("recovery_*.pt"))
        assert len(recovery_files) == 3
        assert manager.latest() is not None
        assert scientific.exists()


def test_historical_authority_is_not_consumed_without_explicit_declaration() -> None:
    available = {
        "CURRENT_STATUS.md": "historical status",
        "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION": "current authority",
    }

    selected = executor.resolve_declared_input_authorities(
        available,
        ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"],
    )

    assert selected == {"M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION": "current authority"}
    assert "CURRENT_STATUS.md" not in selected


def test_cross_domain_hash_difference_is_not_a_resume_blocker() -> None:
    control.check_resume_compatibility(
        _identity(source_manifest="file-manifest-a"),
        _identity(source_manifest="state-manifest-b"),
    )


def test_same_domain_hash_difference_rejects_when_consumer_requires_it() -> None:
    with pytest.raises(control.ResumeCompatibilityError):
        control.require_same_domain_hash(
            "file-a",
            "file-b",
            domain="file_byte_sha256",
        )


def test_source_manifest_records_dirty_provenance_without_locking_documents() -> None:
    root = Path(__file__).parents[2]
    manifest = control.capture_execution_source_manifest(
        root,
        task_id="TASK-001",
        semantic_authority_id="M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION",
        relevant_paths=["src/ldm_project/v2/application/research/research_run_control.py"],
        config_identity={"grid": 128},
    )

    assert manifest.execution_source_semantics_id
    assert manifest.source_manifest_id
    assert manifest.relevant_source_paths == ("src/ldm_project/v2/application/research/research_run_control.py",)


def test_recovery_is_due_by_steps_or_elapsed_time() -> None:
    assert control.recovery_due(step=25, last_recovery_step=0, now=1.0, last_recovery_time=0.0)
    assert control.recovery_due(step=1, last_recovery_step=0, now=600.0, last_recovery_time=0.0)
    assert not control.recovery_due(step=24, last_recovery_step=0, now=1.0, last_recovery_time=0.0)


def test_chen_graph_declares_one_gate_and_explicit_resume_boundary() -> None:
    with tempfile.TemporaryDirectory(prefix="research-graph-") as temporary:
        graph = executor.build_scientific_execution_graph(
            repo_root=Path(temporary),
            output_dir=Path("outputs") / "runs" / "graph",
        )

    assert graph["outer_gate"] == "RESEARCH_FAST_PATH_AND_LONG_RUN_RESILIENCE_V1_ESTABLISHED"
    assert graph["authorization_model"]["second_authorization"] is False
    assert graph["resume_argument"] == "--resume <run_dir|checkpoint>"
    assert graph["scientific_execution_started"] is False
