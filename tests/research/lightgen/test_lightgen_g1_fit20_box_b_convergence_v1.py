from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import torch
import pytest

from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_fit20_box_b_convergence_v1 as convergence


def _schedule() -> torch.Tensor:
    source = fit20._build_schedule({}, convergence.SCHEDULE_SEED)
    return convergence._build_continuation_schedule(source)


def _optimizer_state(step: int) -> dict:
    return {"state": {str(index): {"step": torch.tensor(float(step)),
                                   "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}
                      for index in range(4)},
            "param_groups": []}


def _checkpoint(schedule: torch.Tensor, completed: int) -> dict:
    flat = schedule.reshape(1500, convergence.CLASS_COUNT)
    history = [{"completed_update": index + 1,
                "branch_step": convergence.START_BRANCH_STEP + index + 1,
                "sample_indices": flat[convergence.START_BRANCH_STEP + index].tolist(),
                "arm": "B"}
               for index in range(completed)]
    branch_step = convergence.START_BRANCH_STEP + completed
    return {
        "schema": "lightgen_g1_fit20_box_b_convergence_checkpoint.v1",
        "task_id": convergence.TASK_ID, "run_id": "new-run",
        "source_task_id": fit20.TASK_ID, "source_run_id": "old-run",
        "source_arm": "B", "source_branch_step": 500,
        "completed_new_updates": completed, "branch_step": branch_step,
        "checkpoint_step": completed, "next_schedule_cursor": branch_step,
        "next_batch_indices": flat[branch_step].clone() if branch_step < 1500 else None,
        "optimizer_state": _optimizer_state(branch_step),
        "history": history, "history_count": completed,
        "optimizer_deepcopied": True,
        "schedule_identity": "fit20_box_b_schedule_seed_2026102781:750x2x10",
    }


def test_extended_schedule_reuses_old_prefix_and_each_class_sees_both_targets() -> None:
    schedule = _schedule()
    old = fit20._build_schedule({}, convergence.SCHEDULE_SEED)
    assert tuple(schedule.shape) == (750, 2, 10)
    assert torch.equal(schedule[:250], old)
    for cycle in range(750):
        for label in range(10):
            assert set(schedule[cycle, :, label].tolist()) == {2 * label, 2 * label + 1}


def test_b500_inherited_optimizer_is_valid_at_new_step_zero_and_terminal() -> None:
    schedule = _schedule()
    for completed in (0, 1000):
        checkpoint = _checkpoint(schedule, completed)
        convergence._validate_checkpoint(checkpoint, "new-run", "old-run", completed,
                                         checkpoint["history"], schedule)


def test_checkpoint_rejects_a_stale_adam_step_or_schedule_cursor() -> None:
    schedule = _schedule()
    checkpoint = _checkpoint(schedule, 1)
    checkpoint["optimizer_state"]["state"]["0"]["step"] = torch.tensor(500.)
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        convergence._validate_checkpoint(checkpoint, "new-run", "old-run", 1,
                                         checkpoint["history"], schedule)

    checkpoint = _checkpoint(schedule, 1)
    checkpoint["next_batch_indices"] = schedule.reshape(1500, 10)[502].clone()
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        convergence._validate_checkpoint(checkpoint, "new-run", "old-run", 1,
                                         checkpoint["history"], schedule)


def test_checkpoint_rejects_history_content_that_differs_from_committed_history() -> None:
    schedule = _schedule()
    checkpoint = _checkpoint(schedule, 2)
    committed_history = copy.deepcopy(checkpoint["history"])
    checkpoint["history"][0]["loss"] = 123.0
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        convergence._validate_checkpoint(checkpoint, "new-run", "old-run", 2,
                                         committed_history, schedule)


def _review_receipt() -> dict:
    def transfer(path: str) -> dict:
        item = {"path": path, "remote_bytes": 10,
                "receipt_local_bytes": 10, "current_bytes": 10}
        return {"status": "COMPLETE", "file_count": 1, "bytes_total": 10,
                "all_remote_local_current_sizes_match": True, "files": [item]}

    commands = {
        arm: {"history_updates": 500, "adam_parameter_steps": [500] * 4,
              "all_six_metric_rows_recomputed": True,
              "historical_consumer_accepted_witness_substitution": True,
              "repaired_consumer_rejected_witness_substitution": True,
              "windows_cpu_to_optical_response_relative_error_max": 1.0e-8}
        for arm in ("A", "B")
    }
    recovery_arm = {
        "step3_adam_steps": [3] * 4, "live_step4_adam_steps": [4] * 4,
        "restored_step4_adam_steps": [4] * 4, "engineering_adam_calls": 5,
        "mapper_max_abs_delta": 0.0, "adam_max_abs_delta": 0.0,
        "total_loss_abs_delta": 0.0, "coefficient_loss_abs_delta": 0.0,
        "rng_exact": True, "schedule_exact": True, "frozen_optics_exact": True,
    }
    return {
        "status": "WINDOWS_REVIEW_RECONSUMPTION_PASSED",
        "task_id": fit20.TASK_ID, "source_run_id": "old-run",
        "historical_state_preserved": "VALID_FAILED", "scientific_updates": 0,
        "source_population_exact": True,
        "all_six_metrics_independently_recomputed": True,
        "recovery_raw_states_recomputed": {
            "all_raw_recovery_states_verified": True, "scientific_updates": 0,
            "engineering_adam_calls_total": 10,
            "arms": {"A": copy.deepcopy(recovery_arm), "B": copy.deepcopy(recovery_arm)},
        },
        "old_attempt_transfer": transfer("old/one.pt"),
        "repair_transfer": transfer("repair/one.pt"),
        "commands": commands,
    }


def test_windows_review_consumer_checks_raw_recovery_and_transfer_details(tmp_path) -> None:
    receipt = _review_receipt()
    path = tmp_path / "windows_review_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    config = {"source": {"windows_review_receipt": str(path)}}
    source = {"identity": SimpleNamespace(run_id="old-run")}

    assert convergence._validate_windows_review(config, source)["status"] == \
        "WINDOWS_REVIEW_RECONSUMPTION_PASSED"

    receipt["recovery_raw_states_recomputed"]["arms"]["B"]["rng_exact"] = False
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw recovery comparison"):
        convergence._validate_windows_review(config, source)

    receipt = _review_receipt()
    receipt["repair_transfer"]["files"][0]["current_bytes"] = 11
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="transfer evidence"):
        convergence._validate_windows_review(config, source)
