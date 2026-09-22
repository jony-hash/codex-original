from __future__ import annotations

import json
import tempfile

import pytest
import torch

from ldm_project.v2.application.research import research_run_control as run_control
from scripts.research import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as continuation


def test_frozen_joint_schedule_and_authorization() -> None:
    assert continuation.AUTHORIZATION_TOKEN == "AUTHORIZED_EXECUTE_JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_V1"
    assert continuation.START_STEP == 9000
    assert continuation.STAGE1_END == 9300
    assert continuation.TOTAL_STEPS == 12000
    assert continuation.CHECKPOINT_STEPS == (9000, 9100, 9200, 9300, 9600, 10000, 10500, 11000, 11500, 12000)
    assert continuation.DELTA_JOINT == 1.365988374680088e-05
    assert continuation.GRADIENT_ENGINE == "HYBRID_SEMANTIC_BATCHED_VJP"
    assert continuation.SCALAR_REFERENCE_ENGINE == "SCALAR_REFERENCE_VJP"
    assert continuation.GRADIENT_COSINE_MINIMUM == 0.999999


def test_semantic_batched_vjp_rows_map_to_named_losses() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    losses = {
        "L_S": 2.0 * parameter[0] + parameter[1],
        "L_I": parameter[0] - parameter[1],
        "L_C": parameter.square().sum(),
    }

    gradients = continuation._semantic_batched_gradients(losses, parameter)

    assert torch.allclose(gradients["g_S"], torch.tensor([2.0, 1.0]))
    assert torch.allclose(gradients["g_I"], torch.tensor([1.0, -1.0]))
    assert torch.allclose(gradients["g_C"], torch.tensor([2.0, 4.0]))


def test_reused_baseline_digest_rejection_is_fail_closed() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-baseline-digest-") as temporary:
        path = continuation.Path(temporary) / "baseline.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(continuation.JointContinuationError, match="identity changed"):
            continuation._load_reused_json(continuation.Path(temporary), continuation.Path("baseline.json"), "0" * 64, "baseline")


def test_joint_direction_uses_equal_l2_then_joint_rms_normalization() -> None:
    values = continuation._compose_joint_direction(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    assert torch.isclose(values["q_joint"].square().mean().sqrt(), torch.tensor(1.0), atol=1.0e-6)
    assert torch.allclose(values["d_joint"], -values["q_joint"])
    assert float(values["v_joint_l2_norm"]) > 0.0


def test_zero_joint_direction_is_rejected() -> None:
    with pytest.raises(continuation.JointDirectionDegenerated):
        continuation._compose_joint_direction(
            torch.tensor([1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
        )


def test_start_checkpoint_schema_is_fail_closed() -> None:
    with pytest.raises(continuation.JointContinuationError):
        continuation._validate_start_checkpoint(continuation._root(), {"completed_updates": 9000})


def _minimal_report() -> dict[str, object]:
    development = {
        "9000": {"rows": 20},
        "9300": {"rows": 30},
    }
    transfer = {
        "9000": {"rows": 10},
        "9300": {"rows": 11},
    }
    return {
        "schema": continuation.REPORT_SCHEMA,
        "task_id": continuation.TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "stage1": {"development": development, "S1_transfer": transfer},
        "stage2": {"executed": False, "development": development, "S1_transfer": {}, "final_step": 9300},
        "evaluation_sample_counts": {
            "DEV": {"9000": 20, "9300": 30},
            "S1": {"9000": 10, "9300": 11},
        },
    }


def _running_report_control(target: continuation.Path) -> tuple[run_control.HeartbeatWriter, run_control.ResearchRunControl]:
    identity = run_control.ResearchRunIdentity(
        task_id=continuation.TASK_ID,
        run_id="joint-structural-semantic-continuation-v1",
        trajectory_id="joint-structural-semantic-continuation-trajectory-v1",
        semantic_authority_id=continuation.SEMANTIC_AUTHORITY_ID,
        data_role=continuation.TRAIN_ROLE,
        config_identity={},
        model_identity=continuation.TASK_ID,
        optimizer_identity={},
        sampler_identity={},
        execution_source_semantics_id="source",
        scientific_attempt_id="attempt",
    )
    heartbeat = run_control.HeartbeatWriter(target, identity.run_id, total_steps=continuation.TOTAL_STEPS, interval_seconds=0.0)
    control = run_control.ResearchRunControl(identity, "authorization")
    control.mark_preflight_ready()
    control.start_execution()
    heartbeat.update(state=run_control.RUNNING, phase="TRAINING", step=9300, force=True)
    return heartbeat, control


def test_report_validator_accepts_distinct_dev_and_s1_sample_counts() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-report-") as temporary:
        target = continuation.Path(temporary)
        (target / "scientific_step_009300.pt").write_bytes(b"endpoint")
        continuation._validate_report_payload(_minimal_report(), target=target)


def test_report_validator_rejects_missing_endpoint() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-report-missing-") as temporary:
        with pytest.raises(continuation.JointContinuationError):
            continuation._validate_report_payload(_minimal_report(), target=continuation.Path(temporary))


def test_report_persistence_does_not_mark_completed_when_report_already_exists() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-report-existing-") as temporary:
        target = continuation.Path(temporary)
        endpoint = target / "scientific_step_009300.pt"
        endpoint.write_bytes(b"endpoint")
        (target / "report.json").write_text("{}", encoding="utf-8")
        heartbeat, control = _running_report_control(target)
        with pytest.raises(continuation.JointContinuationError):
            continuation._persist_report(target=target, report=_minimal_report(), heartbeat=heartbeat, control=control, step=9300, classification="TEST")
        status = json.loads(heartbeat.latest_status_path.read_text(encoding="utf-8"))
        endpoint_bytes = endpoint.read_bytes()

    assert control.state == run_control.RUNNING
    assert status["phase"] == "FINALIZING"
    assert status["latest_losses"]["report"] == "WRITE_FAILED"
    assert endpoint_bytes == b"endpoint"


def test_report_persistence_marks_completed_only_after_valid_report() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-report-success-") as temporary:
        target = continuation.Path(temporary)
        (target / "scientific_step_009300.pt").write_bytes(b"endpoint")
        heartbeat, control = _running_report_control(target)
        continuation._persist_report(target=target, report=_minimal_report(), heartbeat=heartbeat, control=control, step=9300, classification="TEST")
        persisted = json.loads((target / "report.json").read_text(encoding="utf-8"))
        status = json.loads(heartbeat.latest_status_path.read_text(encoding="utf-8"))

    assert persisted["status"] == "VALID_COMPLETED"
    assert control.state == run_control.RUN_COMPLETED
    assert status["state"] == run_control.RUN_COMPLETED
    assert status["phase"] == "COMPLETED"


def test_report_serialization_failure_does_not_mark_completed() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-report-serialization-") as temporary:
        target = continuation.Path(temporary)
        (target / "scientific_step_009300.pt").write_bytes(b"endpoint")
        heartbeat, control = _running_report_control(target)
        report = _minimal_report()
        report["not_json_serializable"] = object()
        with pytest.raises(continuation.JointContinuationError, match="report persistence failed"):
            continuation._persist_report(target=target, report=report, heartbeat=heartbeat, control=control, step=9300, classification="TEST")
        status = json.loads(heartbeat.latest_status_path.read_text(encoding="utf-8"))

    assert control.state == run_control.RUNNING
    assert status["state"] == run_control.RUNNING
    assert status["phase"] == "FINALIZING"
    assert status["latest_losses"]["report"] == "WRITE_FAILED"


def test_source_snapshot_requires_every_manifest_python_dependency() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-source-snapshot-") as temporary:
        snapshot_root = continuation.Path(temporary) / "source_snapshot"
        first = snapshot_root / "src" / "first.py"
        second = snapshot_root / "src" / "second.py"
        first.parent.mkdir(parents=True)
        first.write_text("FIRST = 1\n", encoding="utf-8")
        second.write_text("SECOND = 2\n", encoding="utf-8")
        manifest = {
            "source_manifest_id": "manifest",
            "execution_source_semantics_id": "semantics",
            "relevant_source_paths": ["src/first.py", "src/second.py", "authority.json"],
        }
        entries = [
            {"path": "src/first.py", "sha256": continuation.file_sha256(first)},
            {"path": "src/second.py", "sha256": continuation.file_sha256(second)},
        ]
        snapshot = {
            "schema": continuation.SOURCE_SNAPSHOT_SCHEMA,
            "role": "reconstruction_source_content",
            "source_manifest_id": "manifest",
            "execution_source_semantics_id": "semantics",
            "files": entries,
        }
        continuation._validate_source_snapshot(snapshot, manifest, snapshot_root)
        snapshot["files"] = entries[:1]
        with pytest.raises(run_control.ResumeCompatibilityError, match="snapshot is incomplete"):
            continuation._validate_source_snapshot(snapshot, manifest, snapshot_root)


def test_resume_validator_rejects_missing_operational_state_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "source_manifest_id": "manifest",
        "execution_source_semantics_id": "semantics",
        "relevant_source_paths": [],
    }
    identity = continuation._make_identity(manifest, "attempt")
    payload = {
        "schema": continuation.CHECKPOINT_SCHEMA,
        "completed_updates": 9001,
        "source_manifest": manifest,
        "run_identity": identity.to_dict(),
        "scientific_artifact": False,
        "operational_artifact": True,
        "artifact_kind": "OPERATIONAL_RECOVERY_CHECKPOINT",
        "trainable_parameter_names": ["generator.phase"],
    }
    wrapper = {
        "schema": "research_recovery_checkpoint.v1",
        "artifact_kind": "OPERATIONAL_RECOVERY_CHECKPOINT",
        "authorization_id": continuation.AUTHORIZATION_ID,
        "run_identity": identity.to_dict(),
        "source_manifest_id": "manifest",
        "step": 9001,
        "payload": payload,
    }
    monkeypatch.setattr(continuation, "_source_manifest_for_run", lambda root, authorities: manifest)
    with tempfile.TemporaryDirectory(prefix="joint-resume-invalid-") as temporary:
        checkpoint = continuation.Path(temporary) / "recovery_000001_step_009001.pt"
        continuation.atomic_torch_save_new(checkpoint, wrapper, artifact_kind="test recovery checkpoint")
        with pytest.raises(continuation.JointContinuationError, match="Adam state"):
            continuation._validate_resume_checkpoint(continuation.Path(temporary), checkpoint, {})


def test_resume_preflight_requires_unfinished_stage1_endpoint_evaluation() -> None:
    payload = {"dev_records": {"9000": {"rows": 20}}}
    state = {"payload": payload, "step": continuation.STAGE1_END}
    result = continuation._resume_preflight(continuation.Path("recovery.pt"), state)
    assert result["mode"] == "EXPLICIT_RESUME"
    assert result["endpoint_evaluation_required"] is True
