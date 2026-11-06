from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import review_lightgen_g1_fit20_evidence_repair_v1 as repair
from tests.research.lightgen.test_lightgen_g1_projected_fit20_box_ab_v1 import (
    _synthetic_consumer_case,
)


def test_terminal_consumer_rejects_positive_control_command_with_unchanged_optical_result(
    tmp_path,
) -> None:
    path, payload, population, inputs = _synthetic_consumer_case(tmp_path)
    mapper = {"weight": torch.tensor([1.0])}
    payload["mapper_state"] = mapper
    witness = copy.deepcopy(payload)
    witness["amplitudes"][:, 0, 0, 1] = 0.5
    torch.save(witness, path)

    # The historical consumer accepts a different command when its full-path
    # output is internally consistent and the stored terminal weights match.
    fit20._consume_measurement_file(path, "A", population, inputs, mapper,
                                   payload_override=witness)
    with pytest.raises(RuntimeError, match="does not exactly replay"):
        fit20._consume_measurement_file(path, "A", population, inputs, mapper,
                                       payload["amplitudes"], "exact",
                                       payload_override=witness)


def test_windows_transfer_consumer_rejects_a_wrong_receipt_size(tmp_path) -> None:
    evidence = tmp_path / "backup"
    evidence.mkdir()
    data = evidence / "terminal.pt"
    data.write_bytes(b"12345678")
    manifest = {"task_id": fit20.TASK_ID, "file_count": 1, "bytes_total": 8,
                "files": [{"path": "terminal.pt", "bytes": 8}]}
    (evidence / "transfer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_size = (evidence / "transfer_manifest.json").stat().st_size
    receipt_files = [
        {"path": "terminal.pt", "scp_exit_code": 0, "remote_size_bytes": 8,
         "local_size_bytes": 8, "status": "COPIED"},
        {"path": "transfer_manifest.json", "scp_exit_code": 0,
         "remote_size_bytes": manifest_size, "local_size_bytes": manifest_size,
         "status": "COPIED"},
    ]
    (evidence / "transfer_receipt.json").write_text(
        json.dumps({"status": "COMPLETE", "files": receipt_files}), encoding="utf-8")
    assert repair._verify_transfer(evidence)["all_remote_local_current_sizes_match"] is True

    receipt_files[0]["local_size_bytes"] = 7
    (evidence / "transfer_receipt.json").write_text(
        json.dumps({"status": "COMPLETE", "files": receipt_files}), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt size/status mismatch"):
        repair._verify_transfer(evidence)


def _recovery_state(step: int, schedule_row: torch.Tensor) -> dict:
    return {
        "mapper": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {str(index): {"step": torch.tensor(float(step)),
                                               "exp_avg": torch.tensor([0.0]),
                                               "exp_avg_sq": torch.tensor([0.0])}
                                  for index in range(4)}, "param_groups": []},
        "rng": {"cpu": torch.tensor([11, 12], dtype=torch.uint8),
                "cuda": [torch.tensor([21, 22], dtype=torch.uint8)]},
        "loss": 1.25,
        "coefficient_loss": 0.25,
        "sample_indices": schedule_row.clone(),
        "engineering_adam_calls": 1,
    }


def _make_recovery_fixture(root) -> torch.Tensor:
    schedule = torch.zeros((500, 10), dtype=torch.long)
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir(parents=True)
    (root / "preflight_identity.json").write_text(
        json.dumps({"task_id": fit20.TASK_ID, "run_identity": {"run_id": "preflight-r"}}),
        encoding="utf-8")
    report = {"scientific_updates": 0, "engineering_adam_calls_total": 10,
              "all_recovery_checks_passed": True, "arms": {}}
    for arm in fit20.ARM_NAMES:
        checkpoint = {
            "schema": "lightgen_g1_projected_fit20_box_ab_checkpoint.v1",
            "task_id": fit20.TASK_ID, "arm": arm, "run_id": "preflight-r",
            "completed_updates": 3, "checkpoint_step": 3, "history_count": 3,
            "optimizer_deepcopied": True,
            "schedule_identity": "fit20_paired_schedule_seed_2026102781:500x10",
            "next_batch_indices": schedule[3].clone(),
            "optimizer_state": {"state": {str(index): {"step": torch.tensor(3.)}
                                             for index in range(4)}} ,
            "torch_cpu_rng": torch.tensor([11, 12], dtype=torch.uint8),
            "torch_cuda_rng": [torch.tensor([21, 22], dtype=torch.uint8)],
        }
        live, restored = _recovery_state(4, schedule[3]), _recovery_state(4, schedule[3])
        torch.save(checkpoint, root / arm / "checkpoint_step_000003.pt")
        torch.save(live, root / arm / "original_process_step4.pt")
        torch.save(restored, root / arm / "independent_process_step4.pt")
        report["arms"][arm] = {
            "mapper_max_abs_delta": 0.0, "adam_max_abs_delta": 0.0,
            "total_loss_abs_delta": 0.0, "coefficient_loss_abs_delta": 0.0,
            "rng_exact": True, "schedule_exact": True, "pre_updates": 3,
            "engineering_adam_calls": 5,
            "checkpoint_optimizer_deepcopied": True, "frozen_optics_exact": True,
        }
    (root / "engineering_recovery.json").write_text(json.dumps(report), encoding="utf-8")
    return schedule


def test_recovery_consumer_rejects_raw_rng_mismatch_even_if_aggregate_passes(tmp_path) -> None:
    root = tmp_path / "recovery_raw"
    root.mkdir()
    schedule = _make_recovery_fixture(root)
    assert repair._consume_recovery(root, schedule)["all_raw_recovery_states_verified"] is True

    restored = torch.load(root / "B" / "independent_process_step4.pt",
                          map_location="cpu", weights_only=False)
    restored["rng"]["cpu"][0] = 99
    torch.save(restored, root / "B" / "independent_process_step4.pt")
    with pytest.raises(RuntimeError, match="raw live/recovered fourth-step"):
        repair._consume_recovery(root, schedule)
