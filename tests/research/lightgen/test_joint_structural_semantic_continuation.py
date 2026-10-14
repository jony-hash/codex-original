from __future__ import annotations

import json
import tempfile

import pytest
import torch

from ldm_project.v2.application.research.shared import research_run_control as run_control
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as continuation


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


def _selection_inputs(*, scalar_hours: float = 1.0, batched_hours: float = 1.0, scalar_memory: float = 0.5, batched_memory: float = 0.5) -> dict[str, dict[str, float]]:
    return {
        "estimated_hours": {
            continuation.SCALAR_REFERENCE_ENGINE: scalar_hours,
            continuation.GRADIENT_ENGINE: batched_hours,
        },
        "memory_fraction": {
            continuation.SCALAR_REFERENCE_ENGINE: scalar_memory,
            continuation.GRADIENT_ENGINE: batched_memory,
        },
    }


def test_gradient_engine_selection_prefers_batched_at_ten_percent() -> None:
    result = continuation._select_gradient_engine(speedup_fraction=0.10, **_selection_inputs())
    assert result["status"] == "READY"
    assert result["selected_engine"] == continuation.GRADIENT_ENGINE


def test_gradient_engine_selection_falls_back_to_scalar_when_batched_is_not_ten_percent() -> None:
    result = continuation._select_gradient_engine(speedup_fraction=0.09, **_selection_inputs())
    assert result["status"] == "READY"
    assert result["selected_engine"] == continuation.SCALAR_REFERENCE_ENGINE


def test_gradient_engine_selection_does_not_block_on_high_memory_or_duration() -> None:
    result = continuation._select_gradient_engine(
        speedup_fraction=0.50,
        **_selection_inputs(scalar_hours=13.0, batched_hours=13.0, scalar_memory=0.95, batched_memory=0.95),
    )
    assert result["status"] == "READY"
    assert result["selected_engine"] == continuation.GRADIENT_ENGINE


def test_gradient_engine_selection_rejects_invalid_resource_measurements() -> None:
    result = continuation._select_gradient_engine(
        speedup_fraction=0.50,
        **_selection_inputs(scalar_memory=float("nan"), batched_memory=-0.1),
    )
    assert result["status"] == "BLOCKED"
    assert result["selected_engine"] is None


def test_estimated_duration_is_reported_but_does_not_block_selection() -> None:
    result = continuation._select_gradient_engine(
        speedup_fraction=0.10,
        **_selection_inputs(scalar_hours=100.0, batched_hours=120.0),
    )
    assert result["status"] == "READY"
    assert result["selected_engine"] == continuation.GRADIENT_ENGINE


@pytest.mark.parametrize("defect", [None, "resource", "negative_control", "engine_choice"])
def test_receipt_consumes_production_benchmark_shape_and_required_results(defect: str | None) -> None:
    performance = continuation._select_gradient_engine(speedup_fraction=0.10, **_selection_inputs(scalar_memory=0.99, batched_memory=0.99))
    selected = performance["selected_engine"]
    preflight = {
        "status": "READY", "performance": performance,
        "selected_engine": selected, "gradient_engine": selected,
        "scalar_reference_engine": continuation.SCALAR_REFERENCE_ENGINE,
        "three_step_real_update": {"updates": 3, "persistent_updates": 0},
        "negative_control": {"rejected": True},
        "one_step_adam_equivalence": {"passed": True},
        "directional_sanity": {"q_joint_finite": True},
        "stored_rng_restored": True, "E_D_in_backward": False,
        "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0},
    }
    if defect == "resource":
        performance["resource_ok"][selected] = False
    elif defect == "negative_control":
        preflight["negative_control"]["rejected"] = False
    elif defect == "engine_choice":
        performance["selected_engine"] = continuation.SCALAR_REFERENCE_ENGINE
    if defect is not None:
        with pytest.raises(run_control.RunControlError, match="did not pass"):
            continuation._preflight_receipt({"execution_source_semantics_id": "unit"}, preflight)
        return
    receipt = continuation._preflight_receipt({"execution_source_semantics_id": "unit"}, preflight)
    assert receipt.selected_engine == selected
    assert receipt.checks[run_control.RESOURCE_ENVELOPE]["passed"] is True
    assert receipt.checks[run_control.CORRECTNESS_INVARIANTS]["evaluator_isolated"] is True


def test_gradient_engine_numeric_mismatch_is_fail_closed() -> None:
    with pytest.raises(continuation.JointContinuationError, match="differs"):
        continuation._compare_gradient_tensor(torch.tensor([1.0]), torch.tensor([2.0]), "g_struct")


def test_repeated_preflight_failures_are_recorded_without_a_retry_quota() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-preflight-budget-") as temporary:
        target = continuation.Path(temporary) / "attempt_01"
        decisions = []
        for _ in range(3):
            control = continuation._preflight_failure_control(target)
            decision = control.record_preflight_failure("resource", "absolute-resource-envelope")
            continuation._record_persisted_preflight_failure(target, control, "resource", "absolute-resource-envelope", decision)
            decisions.append(decision)
        assert decisions == [run_control.PREFLIGHT_RECHECK_ALLOWED] * 3
        assert not target.exists()


def test_git_heads_and_branch_are_provenance_not_locks() -> None:
    consumer_head = "500045d6a92843c81ffeda010e01351f559e453b"
    binding = robust._bind_git_provenance(
        {
            "branch": "a-different-branch",
            "head": consumer_head,
            "worktree_dirty": False,
            "status_sha256": "0" * 64,
        }
    )

    assert binding["producer"]["head"] == robust.EXPECTED_PRODUCER_HEAD
    assert binding["consumer"]["head"] == consumer_head
    assert binding["head_equal"] is False
    assert binding["producer_head_is_provenance_only"] is True


def test_legacy_exact_head_contract_is_ignored_for_admission() -> None:
    consumer_head = "500045d6a92843c81ffeda010e01351f559e453b"
    binding = robust._bind_git_provenance(
        {
            "branch": robust.EXPECTED_BRANCH,
            "head": consumer_head,
            "worktree_dirty": False,
            "status_sha256": "0" * 64,
        },
        contract={
            "consumer_head_exact_required": True,
            "consumer_head": robust.EXPECTED_PRODUCER_HEAD,
        },
    )
    assert binding["consumer"]["head"] == consumer_head
    assert binding["producer_head_is_provenance_only"] is True


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


def test_reused_baseline_digest_is_not_an_execution_lock() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-baseline-digest-") as temporary:
        path = continuation.Path(temporary) / "baseline.json"
        path.write_text('{"result":"historical"}', encoding="utf-8")
        payload = continuation._load_reused_json(continuation.Path(temporary), continuation.Path("baseline.json"), "baseline")
        assert payload == {"result": "historical"}


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
    control.mark_preflight_ready(continuation.PreflightReceipt(
        authorization_id="authorization",
        checks={
            continuation.CORRECTNESS_INVARIANTS: {"passed": True},
            continuation.RESOURCE_ENVELOPE: {"passed": True},
            continuation.IMPLEMENTATION_SELECTION: {"passed": True, "selected_engine": "test"},
        },
        selected_engine="test",
    ))
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


def test_source_snapshot_is_provenance_and_does_not_lock_source_content() -> None:
    with tempfile.TemporaryDirectory(prefix="joint-source-snapshot-") as temporary:
        snapshot_root = continuation.Path(temporary) / "source_snapshot"
        first = snapshot_root / "src" / "first.py"
        second = snapshot_root / "src" / "second.py"
        first.parent.mkdir(parents=True)
        first.write_text("FIRST = 1\n", encoding="utf-8")
        second.write_text("SECOND = 2\n", encoding="utf-8")
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
        continuation._validate_source_snapshot(snapshot, snapshot_root)
        snapshot["source_manifest_id"] = "different-manifest"
        snapshot["execution_source_semantics_id"] = "different-semantics"
        first.write_text("EDITED = 3\n", encoding="utf-8")
        continuation._validate_source_snapshot(snapshot, snapshot_root)
        snapshot["files"] = entries[:1]
        continuation._validate_source_snapshot(snapshot, snapshot_root)
        snapshot["files"] = [{"path": "src/missing.py", "sha256": "ignored"}]
        with pytest.raises(run_control.ResumeCompatibilityError, match="file is missing"):
            continuation._validate_source_snapshot(snapshot, snapshot_root)


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
