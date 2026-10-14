from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research.shared.research_run_control import RunControlError

from scripts.research.lightgen import execute_lightgen_2d_local_compute_bounded_joint_canary_v1 as canary
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as joint


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _contract() -> dict[str, object]:
    path = _repo_root() / canary.CONTRACT_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def test_canary_schedule_and_scalar_backend_are_fixed() -> None:
    assert canary.AUTHORIZATION_TOKEN == "AUTHORIZED_EXECUTE_LOCAL_COMPUTE_BOUNDED_JOINT_CANARY_V1"
    assert canary.START_STEP == 9000
    assert canary.MAX_UPDATES == 300
    assert canary.PRIMARY_ENDPOINT == 9300
    assert canary.CHECKPOINT_STEPS == (9100, 9200, 9300)
    assert canary.GRADIENT_ENGINE == "SCALAR_REFERENCE_VJP"
    assert canary.TOTAL_STEPS == canary.PRIMARY_ENDPOINT


def test_contract_identity_accepts_producer_provenance_only() -> None:
    payload = _contract()
    binding = canary.validate_contract_payload(_repo_root(), payload)
    assert binding["producer_head_is_provenance_only"] is True
    assert binding["consumer_head_exact_required"] is False
    assert binding["consumer_head"] == canary.git_identity(_repo_root())["head"]


def test_historical_exact_consumer_lock_is_ignored_for_admission() -> None:
    payload = _contract()
    payload["git"] = dict(payload["git"])
    payload["git"]["consumer_head_exact_required"] = True
    payload["git"]["required_consumer_head"] = "a9cd1749adfa690e282fd52a6425564fc2cf102c"
    binding = canary.validate_contract_payload(_repo_root(), payload)
    assert binding["consumer_head_exact_required"] is False


def test_training_backend_rejects_hybrid_engine() -> None:
    with pytest.raises(canary.CanaryError, match="training backend"):
        canary._require_training_backend("HYBRID_SEMANTIC_BATCHED_VJP")


def test_execution_source_manifest_names_canary_consumer() -> None:
    payload = _contract()
    authorities = {"authority_hashes": {}}
    manifest = canary.execution_source_manifest(_repo_root(), authorities, payload)
    assert manifest["task_id"] == canary.TASK_ID
    assert "scripts/research/lightgen/execute_lightgen_2d_local_compute_bounded_joint_canary_v1.py" in manifest["relevant_source_paths"]
    assert manifest["branch"] == canary.git_identity(_repo_root())["branch"]
    assert manifest["head"] == canary.git_identity(_repo_root())["head"]


def test_upstream_phase_loader_treats_historical_head_as_provenance() -> None:
    binding = phase_repair._bind_git_provenance(
        {"branch": phase_repair.EXPECTED_BRANCH, "head": canary.EXPECTED_CONSUMER_HEAD, "worktree_dirty": True, "status_sha256": "status"}
    )
    assert binding["producer"]["head"] == phase_repair.EXPECTED_PRODUCER_HEAD
    assert binding["consumer"]["head"] == canary.EXPECTED_CONSUMER_HEAD
    assert binding["consumer_head_exact_required"] is False
    assert binding["producer_head_is_provenance_only"] is True


def test_canary_preflight_does_not_recompute_accepted_gradients() -> None:
    source = inspect.getsource(canary._preflight)
    assert "_accepted_style_gradients" not in source
    assert source.count("_semantic_batched_reference") == 1


def test_canary_resource_gate_records_duration_without_time_rejection() -> None:
    memory = [{"max_reserved_bytes": 100, "max_allocated_bytes": 80} for _ in range(3)]
    ready = canary._resource_envelope([24.0, 24.0, 24.0], memory, device=torch.device("cuda:0"), total_memory_bytes=1000)
    assert ready["status"] == "READY"
    assert ready["resource_ok"] is True
    assert ready["median_update_seconds"] == 24.0
    assert ready["projected_300_update_hours"] == 2.0

    slow = canary._resource_envelope([240.0, 240.0, 240.0], memory, device=torch.device("cuda:0"), total_memory_bytes=1000)
    assert slow["status"] == "READY"
    assert slow["resource_ok"] is True
    assert slow["projected_300_update_hours"] == 20.0


def test_canary_resource_gate_rejects_oom_or_memory_overrun() -> None:
    oom = canary._resource_envelope([1.0, 1.0, 1.0], [{"max_reserved_bytes": 100, "max_allocated_bytes": 80, "oom": True}] * 3, device=torch.device("cuda:0"), total_memory_bytes=1000)
    assert oom["status"] == "PREFLIGHT_BLOCKED(resource)"
    overrun = canary._resource_envelope([1.0, 1.0, 1.0], [{"max_reserved_bytes": 1001, "max_allocated_bytes": 80}] * 3, device=torch.device("cuda:0"), total_memory_bytes=1000)
    assert overrun["memory_ok"] is False
    assert overrun["resource_ok"] is False


@pytest.mark.parametrize("rollback_case", ["valid", "empty", "failed", "ordinary_false_metadata"])
def test_canary_receipt_requires_actual_rollback_results(rollback_case: str) -> None:
    preflight = {
        "gradient_engine": canary.GRADIENT_ENGINE,
        "training_backend": canary.SCALAR_REFERENCE_ENGINE,
        "negative_control": {"rejected": True},
        "semantic_batched_reference": {"all_equivalent": True, "batched_calls": 1},
        "one_step_adam_equivalence": {"passed": True},
        "three_step_real_update": {"updates": 3},
        "rollback": {"runtime": True, "frozen": True, "adam": True, "rng": True, "sampler": True, "evaluator_grads": True},
        "resource_envelope": {"status": "READY", "resource_ok": True, "median_update_seconds": 1.5, "projected_300_update_hours": 0.125, "peak_memory": {"max_reserved_bytes": 10}},
    }
    if rollback_case == "empty":
        preflight["rollback"] = {}
    elif rollback_case == "failed":
        preflight["rollback"]["adam"] = False
    elif rollback_case == "ordinary_false_metadata":
        preflight["rollback"]["training_started"] = False
    if rollback_case in {"empty", "failed"}:
        with pytest.raises(RunControlError, match="did not pass"):
            canary._preflight_receipt({"execution_source_semantics_id": "canary-source"}, preflight)
        return
    receipt = canary._preflight_receipt({"execution_source_semantics_id": "canary-source"}, preflight)
    assert set(receipt.checks) == {canary.CORRECTNESS_INVARIANTS, canary.RESOURCE_ENVELOPE, canary.IMPLEMENTATION_SELECTION}
    assert receipt.checks[canary.RESOURCE_ENVELOPE]["median_update_seconds"] == 1.5
    assert receipt.checks[canary.IMPLEMENTATION_SELECTION]["batched_reference_calls"] == 1
    assert receipt.resource["projected_300_update_hours"] == 0.125


def test_canary_failure_state_records_repeated_signatures_without_quota(tmp_path: Path) -> None:
    target = tmp_path / "attempt_01"
    with pytest.raises(canary.CanaryError, match="RECHECK_ALLOWED"):
        canary._raise_recorded_preflight_failure(target, canary.CanaryError("PREFLIGHT_BLOCKED(resource): median update"))
    assert not target.exists()
    failure_dir = canary._preflight_failure_state_dir(target)
    assert len(list(failure_dir.glob("failure_*.json"))) == 1
    with pytest.raises(canary.CanaryError, match="RECHECK_ALLOWED"):
        canary._raise_recorded_preflight_failure(target, canary.CanaryError("PREFLIGHT_BLOCKED(resource): median update"))
    assert len(list(failure_dir.glob("failure_*.json"))) == 2


def _structural_public_summary(*, name: str = "real_ols_step9000") -> dict[str, object]:
    return {
        **{field: {"value": field} for field in joint.STRUCTURAL_SCIENTIFIC_PUBLIC_FIELDS},
        "name": name,
    }


def test_private_structural_namespace_name_is_not_a_scientific_mismatch() -> None:
    computed = _structural_public_summary(name="real_ols_step9000_private")
    expected = _structural_public_summary()
    joint._compare_structural_scientific_public(computed, expected)


@pytest.mark.parametrize(
    "field",
    ["rows", "signature_sha256", "label_source"],
)
def test_private_structural_scientific_identity_changes_fail_closed(field: str) -> None:
    computed = _structural_public_summary(name="real_ols_step9000_private")
    expected = _structural_public_summary()
    computed[field] = {"value": f"changed:{field}"}
    with pytest.raises(joint.JointContinuationError, match="scientific structural statistics"):
        joint._compare_structural_scientific_public(computed, expected)


def test_private_structural_public_schema_changes_fail_closed() -> None:
    computed = _structural_public_summary(name="real_ols_step9000_private")
    expected = _structural_public_summary()
    computed["unexpected"] = 1
    with pytest.raises(joint.JointContinuationError, match="public structural schema changed"):
        joint._compare_structural_scientific_public(computed, expected)


def test_post_open_invalid_harness_closeout_is_write_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(canary, "CONTRACT_PATH", Path("contract.json"))
    target = tmp_path / "attempt_01"
    target.mkdir()
    receipt = canary.PreflightReceipt(
        source_semantics_id="source",
        authorization_id=canary.AUTHORIZATION_ID,
        checks={
            canary.CORRECTNESS_INVARIANTS: {"passed": True},
            canary.RESOURCE_ENVELOPE: {"passed": True},
            canary.IMPLEMENTATION_SELECTION: {"passed": True, "selected_engine": canary.SCALAR_REFERENCE_ENGINE},
        },
        selected_engine=canary.SCALAR_REFERENCE_ENGINE,
    )
    canary._write_invalid_harness_closeout(tmp_path, target, {"execution_source_semantics_id": "source"}, receipt, canary.CanaryError("comparison failed"))
    payload = json.loads((target / canary.INVALID_HARNESS_CLOSEOUT_NAME).read_text(encoding="utf-8"))
    assert payload["state"] == "INVALID_HARNESS"
    assert payload["persistent_updates"] == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        canary._write_invalid_harness_closeout(tmp_path, target, {"execution_source_semantics_id": "source"}, receipt, canary.CanaryError("second"))
