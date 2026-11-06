from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.lightgen import verify_lightgen_gaussian_production_replay_v1 as replay


def _expected_rows() -> list[dict[str, object]]:
    rows = []
    position = 0
    for endpoint in replay.ENDPOINTS:
        for kind in ("DEV", "G4"):
            for label in range(10):
                for within_class in range(2):
                    row: dict[str, object] = {
                        "endpoint": endpoint, "kind": kind, "position": position,
                        "label": label,
                    }
                    if kind == "DEV":
                        row["dataset_row"] = len(endpoint) * 10_000 + label * 2 + within_class
                    else:
                        row.update({"cohort_row": label * 100 + within_class,
                                    "proposal_id": len(endpoint) * 100_000 + label * 100 + within_class,
                                    "component_id": label % 4})
                    rows.append(row)
                    position += 1
    return rows


def test_replay_consumer_accepts_exact_row_identity() -> None:
    expected = _expected_rows()
    replay._validate_replay_rows([dict(row) for row in expected], expected)


@pytest.mark.parametrize("corruption", ["endpoint", "row", "missing"])
def test_replay_consumer_rejects_endpoint_row_or_population_mismatch(corruption: str) -> None:
    expected = _expected_rows()
    actual = [dict(row) for row in expected]
    if corruption == "endpoint":
        actual[0]["endpoint"] = "J"
    elif corruption == "row":
        actual[0]["dataset_row"] = int(actual[0]["dataset_row"]) + 1
    else:
        actual.pop()
    with pytest.raises(RuntimeError):
        replay._validate_replay_rows(actual, expected)


def test_checkpoint_consumer_requires_both_optimizer_phase_states() -> None:
    payload = {
        "optimizer": {
            "param_groups": [{"params": [0, 1]}],
            "state": {
                0: {"step": torch.tensor(47_500)},
                1: {"step": torch.tensor(47_500)},
            },
        },
        "optimizer_steps": {"encoder_adam_step": 47_500, "decoder_adam_step": 47_500},
    }
    assert replay._optimizer_phase_steps(payload) == (47_500, 47_500)
    payload["optimizer"]["state"].pop(1)
    with pytest.raises(RuntimeError, match="both Encoder and Decoder"):
        replay._optimizer_phase_steps(payload)


def test_c_preflight_producer_serializes_named_steps_and_real_restore_reads_them(tmp_path, monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_phase = torch.nn.Parameter(torch.tensor([1.0]))
            self.decoder_phase = torch.nn.Parameter(torch.tensor([2.0]))
            self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)

    monkeypatch.setattr(replay.joint.base, "_model", lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(replay.joint.base, "_phase_groups",
                        lambda model: (model.encoder_phase, model.decoder_phase))
    model = Model()
    encoder, decoder = replay.joint.base._phase_groups(model)
    optimizer = torch.optim.Adam((encoder, decoder), lr=1e-3)
    optimizer.state[encoder] = {
        "step": torch.tensor(11.0), "exp_avg": torch.tensor([0.25]),
        "exp_avg_sq": torch.tensor([0.0625]),
    }
    optimizer.state[decoder] = {
        "step": torch.tensor(17.0), "exp_avg": torch.tensor([-0.5]),
        "exp_avg_sq": torch.tensor([0.25]),
    }
    payload = replay.joint._preflight_recovery_input(
        {"config": "fixed"}, {"run_id": "source"}, model, optimizer, encoder, decoder,
        torch.arange(40).reshape(4, 10), torch.arange(8), [torch.arange(4)],
    )
    destination = tmp_path / "recovery_input.pt"
    replay.joint.atomic_torch_save_new(destination, payload)
    reread = torch.load(destination, map_location="cpu", weights_only=False)
    assert reread["optimizer_steps"] == {"encoder_adam_step": 11, "decoder_adam_step": 17}
    restored_model, restored_optimizer = replay.joint._restore_joint_state(reread, {}, torch.device("cpu"))
    restored_encoder, restored_decoder = replay.joint.base._phase_groups(restored_model)
    assert replay.joint._optimizer_steps(restored_optimizer, restored_encoder, restored_decoder) == (11, 17)
    assert torch.equal(restored_optimizer.state[restored_encoder]["exp_avg"], torch.tensor([0.25]))
    assert torch.equal(restored_optimizer.state[restored_decoder]["exp_avg"], torch.tensor([-0.5]))
    saved_encoder_step = reread["optimizer"]["state"][0]["step"].clone()
    saved_decoder_step = reread["optimizer"]["state"][1]["step"].clone()
    saved_encoder_moment = reread["optimizer"]["state"][0]["exp_avg"].clone()
    saved_decoder_moment = reread["optimizer"]["state"][1]["exp_avg"].clone()
    restored_optimizer.zero_grad(set_to_none=True)
    (restored_encoder.square().sum() + restored_decoder.square().sum()).backward()
    restored_optimizer.step()
    assert torch.equal(reread["optimizer"]["state"][0]["step"], saved_encoder_step)
    assert torch.equal(reread["optimizer"]["state"][1]["step"], saved_decoder_step)
    assert torch.equal(reread["optimizer"]["state"][0]["exp_avg"], saved_encoder_moment)
    assert torch.equal(reread["optimizer"]["state"][1]["exp_avg"], saved_decoder_moment)
    legacy_tuple = dict(reread, optimizer_steps=(11, 17))
    with pytest.raises(RuntimeError, match="recovery Adam step-to-phase mapping"):
        replay.joint._restore_joint_state(legacy_tuple, {}, torch.device("cpu"))


def test_recovery_cursor_requires_the_registered_train_batches() -> None:
    labels = torch.arange(60_000) % 10
    train_rows = torch.arange(60_000)
    schedule = torch.arange(10).expand(5_000, 10).clone()
    assert tuple(replay._validate_j_cursor(schedule, 2_500, labels, train_rows).shape) == (4, 10)
    with pytest.raises(RuntimeError, match="cursor"):
        replay._validate_j_cursor(schedule, 2_501, labels, train_rows)
    wrong = schedule.clone()
    wrong[2_502, 0] = 11
    with pytest.raises(RuntimeError, match="TRAIN rows"):
        replay._validate_j_cursor(wrong, 2_500, labels, train_rows)


def _valid_update_calls(first_step: int = 47_501,
                        first_schedule_cursor: int = 2_500) -> list[dict[str, object]]:
    branches = ["original_process", "original_process", "original_process",
                "original_process_fourth", "independent_process_fourth"]
    result = []
    rows = [[index * 10 + item for item in range(10)] for index in range(4)]
    rows.append(rows[3])
    for index, (branch, sample_rows) in enumerate(zip(branches, rows, strict=True)):
        step = first_step + min(index, 3)
        cursor = first_schedule_cursor + min(index, 3)
        result.append({
            "call_number": index + 1, "branch": branch, "status": "COMPLETED",
            "global_step_before": step - 1, "schedule_cursor": cursor,
            "global_step_after": step,
            "optimizer_steps": {"encoder_adam_step": step, "decoder_adam_step": step},
            "sample_rows": sample_rows, "device": "NVIDIA RTX 4080",
            "elapsed_event_ms": 0.5 + index, "peak_reserved_vram_bytes": 1024,
            "loss": {"total": 1.0 + index},
        })
    return result


def test_recovery_log_rejects_empty_nan_timing_and_wrong_batch_identity() -> None:
    calls = _valid_update_calls()
    branches = ["original_process", "original_process", "original_process",
                "original_process_fourth", "independent_process_fourth"]
    rows = [entry["sample_rows"] for entry in calls]
    cursors = [2_500, 2_501, 2_502, 2_503, 2_503]
    assert len(replay._validate_recovery_update_log(
        {"calls": calls}, "calls", branches, 47_501, rows, cursors,
    )) == 5
    with pytest.raises(RuntimeError, match="exactly five calls"):
        replay._validate_recovery_update_log(
            {"calls": []}, "calls", branches, 47_501, rows, cursors,
        )
    malformed = [dict(entry) for entry in calls]
    malformed[2]["elapsed_event_ms"] = float("nan")
    with pytest.raises(RuntimeError, match="entry 3"):
        replay._validate_recovery_update_log(
            {"calls": malformed}, "calls", branches, 47_501, rows, cursors,
        )
    malformed = [dict(entry) for entry in calls]
    malformed[0]["sample_rows"] = [99] * 10
    with pytest.raises(RuntimeError, match="entry 1"):
        replay._validate_recovery_update_log(
            {"calls": malformed}, "calls", branches, 47_501, rows, cursors,
        )
    malformed = [dict(entry) for entry in calls]
    malformed[2]["schedule_cursor"] += 1
    with pytest.raises(RuntimeError, match="entry 3"):
        replay._validate_recovery_update_log(
            {"calls": malformed}, "calls", branches, 47_501, rows, cursors,
        )
    malformed = [dict(entry) for entry in calls]
    malformed[3]["global_step_before"] -= 1
    with pytest.raises(RuntimeError, match="entry 4"):
        replay._validate_recovery_update_log(
            {"calls": malformed}, "calls", branches, 47_501, rows, cursors,
        )


def test_worker_failure_keeps_durable_prestart_call_count(tmp_path, monkeypatch) -> None:
    calls = _valid_update_calls()[:4]
    calls.append({"call_number": 5, "branch": "independent_process_fourth",
                  "status": "CALL_STARTED", "sample_rows": calls[3]["sample_rows"]})
    replay._write_json(tmp_path / "engineering_update_log.json", {"calls": calls})

    def fail_after_read(command, cwd, capture_output, text, check):
        saved = replay._read_json(tmp_path / "engineering_update_log.json")["calls"]
        assert len(saved) == 5 and sum(item["status"] == "COMPLETED" for item in saved) == 4
        assert saved[4]["status"] == "CALL_STARTED"
        return type("WorkerResult", (), {"returncode": 1, "stderr": "injected worker failure",
                                          "stdout": ""})()

    monkeypatch.setattr(replay.subprocess, "run", fail_after_read)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        replay._run_j_worker(tmp_path, ["python", "-m", "test-worker"])
    saved = replay._read_json(tmp_path / "engineering_update_log.json")["calls"]
    assert sum(item["status"] == "COMPLETED" for item in saved) == 4
    assert saved[4]["status"] == "CALL_STARTED"


def test_j_recovery_finalizes_complete_existing_calls_without_new_stage(tmp_path, monkeypatch) -> None:
    stage_root = tmp_path / "engineering" / "j_recovery"
    stage = stage_root / "attempt_01"
    stage.mkdir(parents=True)
    calls = _valid_update_calls()
    replay._write_json(stage / "engineering_update_log.json", {"calls": calls})
    expected = {"status": "PASSED", "engineering_adam_calls": 5}

    monkeypatch.setattr(
        replay, "_finalize_j_recovery_existing_stage",
        lambda _inputs, _output, existing: expected if existing == stage else pytest.fail(),
    )
    monkeypatch.setattr(replay, "_new_stage_dir",
                        lambda *_args: pytest.fail("completed J updates must not be repeated"))

    result = replay._run_j_recovery({}, tmp_path)
    assert result == expected
    assert list(stage_root.glob("attempt_*")) == [stage]


def test_j_recovery_reuses_complete_result_without_finalizing_or_updates(tmp_path, monkeypatch) -> None:
    stage = tmp_path / "engineering" / "j_recovery" / "attempt_01"
    stage.mkdir(parents=True)
    expected = {"status": "PASSED", "engineering_adam_calls": 5}
    replay._write_json(stage / "j_recovery_result.json", expected)
    monkeypatch.setattr(replay, "_finalize_j_recovery_existing_stage",
                        lambda *_args: pytest.fail("a complete result must only be consumed"))
    monkeypatch.setattr(replay, "_new_stage_dir",
                        lambda *_args: pytest.fail("a complete result must not open a new stage"))

    assert replay._run_j_recovery({}, tmp_path) == expected


def test_j_recovery_refuses_to_repeat_calls_from_an_incomplete_stage(tmp_path, monkeypatch) -> None:
    stage_root = tmp_path / "engineering" / "j_recovery"
    stage = stage_root / "attempt_01"
    stage.mkdir(parents=True)
    calls = _valid_update_calls()[:4]
    calls.append({"call_number": 5, "branch": "independent_process_fourth",
                  "status": "CALL_STARTED"})
    replay._write_json(stage / "engineering_update_log.json", {"calls": calls})
    monkeypatch.setattr(replay, "_new_stage_dir",
                        lambda *_args: pytest.fail("incomplete J stage must not open a retry"))

    with pytest.raises(RuntimeError, match="cannot be resumed without repeating Adam calls"):
        replay._run_j_recovery({}, tmp_path)
    assert list(stage_root.glob("attempt_*")) == [stage]


def _recovery_artifact() -> dict[str, object]:
    steps = {"encoder_adam_step": 47_504, "decoder_adam_step": 47_504}
    return {
        "task_id": "test", "origin_checkpoint_run_id": "source",
        "completed_global_step": 47_504, "schedule_cursor": 2_503,
        "optimizer_steps": steps,
        "optimizer": {"param_groups": [{"params": [0, 1]}],
                      "state": {0: {"step": torch.tensor(47_504), "exp_avg": torch.ones(2)},
                                1: {"step": torch.tensor(47_504), "exp_avg": torch.ones(2)}}},
        "model": {"runtime.encoder.phase": torch.ones(2),
                  "runtime.generator.phase": torch.ones(2)},
        "loss": {"total": 0.25}, "rows": torch.arange(10),
        "cpu_rng": torch.arange(5), "cuda_rng": [torch.arange(3)],
    }


def test_recovery_pair_consumers_reject_corrupt_adam_rng_and_identity() -> None:
    direct = _recovery_artifact()
    identity = {"task_id": "test", "origin_checkpoint_run_id": "source",
                "completed_global_step": 47_504, "schedule_cursor": 2_503}
    valid = {key: (value.clone() if isinstance(value, torch.Tensor) else value)
             for key, value in direct.items()}
    assert replay._validate_recovery_pair(
        direct, valid, direct["optimizer_steps"], direct["rows"], identity,
    )["model_max_abs_error"] == 0.0
    bad_adam = {**valid, "optimizer": {"param_groups": [{"params": [0, 1]}],
                                        "state": {0: {"step": torch.tensor(47_504),
                                                      "exp_avg": torch.zeros(2)},
                                                  1: {"step": torch.tensor(47_504),
                                                      "exp_avg": torch.ones(2)}}}}
    with pytest.raises(RuntimeError, match="recovery_pair_optimizer"):
        replay._validate_recovery_pair(direct, bad_adam, direct["optimizer_steps"],
                                       direct["rows"], identity)
    bad_rng = {**valid, "cpu_rng": torch.arange(6)}
    with pytest.raises(RuntimeError, match="RNG differ"):
        replay._validate_recovery_pair(direct, bad_rng, direct["optimizer_steps"],
                                       direct["rows"], identity)
    bad_identity = {**valid, "schedule_cursor": 2_502}
    with pytest.raises(RuntimeError, match="identit"):
        replay._validate_recovery_pair(direct, bad_identity, direct["optimizer_steps"],
                                       direct["rows"], identity)


def test_recovery_consumer_rejects_changes_to_frozen_model_groups() -> None:
    before = {
        "runtime.encoder.phase": torch.zeros(2),
        "runtime.generator.phase": torch.zeros(2),
        "runtime.ols.weight": torch.ones(2),
    }
    valid = {key: value.clone() for key, value in before.items()}
    valid["runtime.encoder.phase"].add_(0.01)
    valid["runtime.generator.phase"].add_(0.01)
    replay._assert_only_phase_groups_changed(before, valid)
    invalid = {key: value.clone() for key, value in valid.items()}
    invalid["runtime.ols.weight"][0] += 0.01
    with pytest.raises(RuntimeError, match="frozen model group"):
        replay._assert_only_phase_groups_changed(before, invalid)


def test_final_consumer_rejects_missing_or_malformed_cuda_evidence() -> None:
    summary = {
        "status": "PASSED", "device_name": "NVIDIA RTX 4080",
        "cuda_event_scope": "CUDA model propagation, readout, and frozen evaluator forward; excludes CPU row checks",
        "cuda_events": [
            {"name": name, "rows": 10, "device": "NVIDIA RTX 4080", "elapsed_ms": 1.0}
            for name in ("DEV_chunk_1", "DEV_chunk_2", "G4_chunk_1", "G4_chunk_2")
        ],
    }
    replay._validate_cuda_evidence(summary)
    with pytest.raises(RuntimeError, match="CUDA evidence"):
        replay._validate_cuda_evidence({"status": "PASSED", "device_name": "NVIDIA RTX 4080"})
    summary["cuda_events"].pop()
    with pytest.raises(RuntimeError, match="CUDA evidence"):
        replay._validate_cuda_evidence(summary)


def test_verify_phase_rejects_missing_cuda_evidence_at_final_consumer(tmp_path, monkeypatch) -> None:
    run_config = {
        "schema": "lightgen_gaussian_production_replay_run.v1",
        "output_dir": replay.OUTPUT_REL.as_posix(),
        "replay": {"total_rows": 120, "native_reference_rows": 80,
                   "native_reference_rows_by_endpoint": {"C": 40, "D0": 20, "J": 20},
                   "native_reference_missing_rows_by_endpoint": {"C": 0, "D0": 20, "J": 20}},
    }
    monkeypatch.setattr(replay, "_load_run_config", lambda _output: run_config)
    monkeypatch.setattr(replay, "_build_run_config", lambda _inputs, _output: run_config)
    manifest = {"schema": "test"}
    monkeypatch.setattr(replay, "_build_input_manifest", lambda _inputs, _config: manifest)
    monkeypatch.setattr(replay, "_interface_spec", lambda _config: "test")
    replay._write_json(tmp_path / "run_config.json", run_config)
    replay._write_json(tmp_path / "input_manifest.json", {"schema": "test"})
    (tmp_path / "interface_spec.md").write_text("test", encoding="utf-8")
    replay._write_json(tmp_path / "prepare_result.json", {
        "status": "PREPARED", "validated_endpoints": list(replay.ENDPOINTS),
        "replay_items": 120, "native_reference_rows": 80,
        "native_reference_missing_rows": 40,
    })
    labels = torch.tensor([label for label in range(10) for _ in range(2)])
    rows = torch.arange(20)
    endpoints = {
        endpoint: {
            "dev": {"labels": labels.clone(), "rows": rows.clone()},
            "g4": {
                "requested_labels": labels.clone(),
                "proposal_ids": rows.clone(),
                "component_ids": rows.remainder(4),
            },
        }
        for endpoint in replay.ENDPOINTS
    }
    stage = tmp_path / "replay" / "C" / "attempt_01"
    stage.mkdir(parents=True)
    replay._write_json(stage / "replay_summary.json", {
        "status": "PASSED", "endpoint": "C", "device_name": "NVIDIA RTX 4080",
        "cuda_events": [],
    })
    inputs = {
        "joint_config": {},
        "endpoints": endpoints,
        "labels": labels,
        "dev_rows": rows,
    }
    with pytest.raises(RuntimeError, match="CUDA evidence"):
        replay._verify(inputs, tmp_path)


def test_repeated_phase_failure_preserves_the_first_failure_receipt(tmp_path) -> None:
    engineering = tmp_path / "engineering"
    engineering.mkdir()
    first = engineering / "j_recovery_failure_receipt.json"
    original = {"status": "STOPPED_INCOMPLETE", "failure_reason": "first failure"}
    replay._write_json(first, original)

    replay._record_phase_failure(tmp_path, "j-recovery", RuntimeError("second failure"))

    assert replay._read_json(first) == original
    second = engineering / "j_recovery_failure_receipt_02.json"
    assert replay._read_json(second)["failure_reason"] == "second failure"


def test_replay_tensor_consumer_recomputes_saved_outputs_and_native_coverage(tmp_path) -> None:
    all_rows = _expected_rows()
    for endpoint in replay.ENDPOINTS:
        rows = [dict(row) for row in all_rows if row["endpoint"] == endpoint]
        native_keys, missing_keys = [], []
        for row in rows:
            row.update({"view_max_abs_error": 0.0, "view_allclose": True,
                        "ed_logit_max_abs_error": 0.0, "ed_logits_allclose": True,
                        "ec_logit_max_abs_error": 0.0, "ec_logits_allclose": True,
                        "predictions_match_saved": True})
            available = endpoint == "C" or row["kind"] == "DEV"
            if available:
                row.update({"native_reference_status": "MATCHED", "native_max_abs_error": 0.0,
                            "native_allclose": True})
                native_keys.append([row["kind"], row["position"]])
            else:
                row.update({"native_reference_status": "MISSING_HISTORICAL_REFERENCE",
                            "native_max_abs_error": None, "native_allclose": None})
                missing_keys.append([row["kind"], row["position"]])
        native_count = len(native_keys)
        tensors = {
            "schema": "lightgen_gaussian_replay_tensors.v1", "endpoint": endpoint,
            "rows": replay._replay_tensor_identity(rows),
            "saved_views": torch.zeros((40, 1, 28, 28)),
            "replay_views": torch.zeros((40, 1, 28, 28)),
            "saved_ed_logits": torch.zeros((40, 10)),
            "replay_ed_logits": torch.zeros((40, 10)),
            "saved_ec_logits": torch.zeros((40, 10)),
            "replay_ec_logits": torch.zeros((40, 10)),
            "native": {"row_keys": native_keys, "missing_row_keys": missing_keys,
                       "saved": torch.zeros((native_count, 1, 200, 200)),
                       "replay": torch.zeros((native_count, 1, 200, 200))},
        }
        identity_keys = ("endpoint", "kind", "position", "dataset_row", "cohort_row", "label",
                         "proposal_id", "component_id")
        fieldnames = list(dict.fromkeys([*identity_keys, *rows[0].keys()]))
        csv_path = tmp_path / f"{endpoint}_replay_rows.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        csv_rows = replay._read_csv(csv_path)
        for row in csv_rows:
            for key in ("position", "label"):
                row[key] = int(row[key])
            for key in ("dataset_row", "cohort_row", "proposal_id", "component_id"):
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            for key in ("view_allclose", "ed_logits_allclose", "ec_logits_allclose",
                        "predictions_match_saved", "native_allclose"):
                if key == "native_allclose" and row.get(key) in (None, ""):
                    row[key] = None
                else:
                    row[key] = row[key].lower() == "true"
            if row.get("native_max_abs_error") not in (None, ""):
                row["native_max_abs_error"] = float(row["native_max_abs_error"])
            else:
                row["native_max_abs_error"] = None
        replay._validate_replay_tensor_evidence(tensors, csv_rows, endpoint)
        corrupted = dict(tensors)
        corrupted["replay_views"] = tensors["replay_views"].clone()
        corrupted["replay_views"][0, 0, 0, 0] = 0.1
        with pytest.raises(RuntimeError, match="replay tensor consumer recomputed"):
            replay._validate_replay_tensor_evidence(corrupted, rows, endpoint)


def test_replay_tensor_consumer_binds_saved_arrays_to_registered_inputs() -> None:
    labels = torch.tensor([label for label in range(10) for _ in range(2)])
    zeros_view = torch.zeros((20, 1, 28, 28))
    zeros_logits = torch.zeros((20, 10))
    zeros_native = torch.zeros((20, 1, 200, 200))
    dev = {"labels": labels, "rows": torch.arange(20), "images": zeros_view.clone(),
           "ed_logits": zeros_logits.clone(), "ec_logits": zeros_logits.clone(),
           "native_images": zeros_native.clone()}
    g4 = {"requested_labels": labels.clone(), "images": zeros_view.clone(),
          "ed_logits": zeros_logits.clone(), "ec_logits": zeros_logits.clone(),
          "native_capture_indices": torch.arange(20),
          "native_capture_images": zeros_native.clone()}
    inputs = {"endpoints": {"C": {"dev": dev, "g4": g4}}}
    rows = []
    for kind in ("DEV", "G4"):
        rows.extend({"endpoint": "C", "kind": kind, "position": position,
                     "label": int(labels[position]),
                     "native_reference_status": "MATCHED"}
                    for position in range(20))
    native_keys = [[row["kind"], row["position"]] for row in rows]
    payload = {
        "schema": "lightgen_gaussian_replay_tensors.v1", "endpoint": "C",
        "rows": replay._replay_tensor_identity(rows),
        "saved_views": torch.zeros((40, 1, 28, 28)),
        "replay_views": torch.zeros((40, 1, 28, 28)),
        "saved_ed_logits": torch.zeros((40, 10)),
        "replay_ed_logits": torch.zeros((40, 10)),
        "saved_ec_logits": torch.zeros((40, 10)),
        "replay_ec_logits": torch.zeros((40, 10)),
        "native": {"row_keys": native_keys, "missing_row_keys": [],
                   "saved": torch.zeros((40, 1, 200, 200)),
                   "replay": torch.zeros((40, 1, 200, 200))},
    }
    replay._validate_replay_tensor_source_arrays(payload, rows, inputs, "C")
    corrupted = dict(payload)
    corrupted["saved_views"] = payload["saved_views"].clone()
    corrupted["saved_views"][0, 0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="registered historical view"):
        replay._validate_replay_tensor_source_arrays(corrupted, rows, inputs, "C")


def test_final_metric_consumer_recomputes_logged_metrics_from_saved_tensors() -> None:
    image = torch.zeros((1, 1, 28, 28))
    logits = torch.zeros((1, 10))
    ssim32 = float(replay.ssim_batch(image, image)[0])
    ssim64 = float(replay.ssim_batch(image.double(), image.double())[0])
    ssim_numpy = float(replay._np_ssim(image.numpy(), image.numpy())[0])
    source_metric = {
        "row": 7, "label": 0, "mse": 1e-12, "psnr_db": 120.0, "ssim": ssim32,
        "ed_prediction": 0, "ec_prediction": 0, "ed_target_margin": 0.0,
        "ec_target_margin": 0.0, "ed_margin": 0.0, "ec_margin": 0.0,
    }
    expected_metrics = {
        "mse": True, "psnr_db": True, "ssim": True,
        "ed_prediction": True, "ec_prediction": True,
        "ed_requested_margin": True, "ec_requested_margin": True,
        "ed_top_margin": True, "ec_top_margin": True,
    }
    row = {
        "endpoint": "C", "kind": "DEV", "position": 0, "dataset_row": 7, "label": 0,
        "ed_prediction": 0, "ec_prediction": 0, "ed_requested_margin": 0.0,
        "ec_requested_margin": 0.0, "ed_top_margin": 0.0, "ec_top_margin": 0.0,
        "mse": 1e-12, "psnr_db": 120.0, "ssim_float32": ssim32,
        "ssim_float64": ssim64, "ssim_numpy": ssim_numpy,
        "saved_metrics_match": expected_metrics,
    }
    payload = {
        "saved_views": image.clone(), "replay_views": image.clone(),
        "saved_ed_logits": logits.clone(), "replay_ed_logits": logits.clone(),
        "saved_ec_logits": logits.clone(), "replay_ec_logits": logits.clone(),
    }
    inputs = {"endpoints": {"C": {
        "dev": {"targets": image.clone()}, "dev_metrics": [source_metric],
    }}}
    replay._validate_replay_tensor_metrics(payload, [row], inputs, "C")
    corrupted = dict(row, mse=0.5)
    with pytest.raises(RuntimeError, match="mse differs when recomputed"):
        replay._validate_replay_tensor_metrics(payload, [corrupted], inputs, "C")


def test_independent_numpy_ssim_matches_production_float64() -> None:
    torch.manual_seed(33)
    image = torch.rand((3, 1, 28, 28), dtype=torch.float64)
    target = torch.rand((3, 1, 28, 28), dtype=torch.float64)
    production = replay.ssim_batch(image, target).numpy()
    reference = replay._np_ssim(image.numpy(), target.numpy())
    assert np.allclose(production, reference, rtol=1e-10, atol=1e-12)


def test_task_progress_updates_stage_and_cursor_atomically(tmp_path) -> None:
    replay._write_task_progress(tmp_path, "J_RECOVERY", "RUNNING", 2, 5, 47_502)
    first = replay._read_json(tmp_path / "progress.json")
    assert first == {
        "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
        "status": "RUNNING", "current_stage": "J_RECOVERY",
        "current_task_local_update": 2, "total_task_local_updates": 5,
        "current_optimizer_step": 47_502,
    }
    replay._write_task_progress(tmp_path, "J_RECOVERY", "PASSED", 5, 5, 47_504)
    assert replay._read_json(tmp_path / "progress.json")["current_optimizer_step"] == 47_504


def test_relative_maps_canonicalized_registered_input_to_task_alias(tmp_path, monkeypatch) -> None:
    root = tmp_path / "isolated"
    shared = tmp_path / "shared"
    registered_run = root / "outputs" / "experiment" / "runs" / "autodl_backup_attempt_01"
    registered_run.mkdir(parents=True)
    shared_run = shared / "outputs" / "experiment" / "runs" / "attempt_01"
    shared_run.mkdir(parents=True)
    (shared_run / "checkpoint.pt").write_bytes(b"checkpoint")
    original_resolve = Path.resolve

    def resolve_registered_run(path, *args, **kwargs):
        if path == registered_run:
            return shared_run
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_registered_run)
    monkeypatch.setattr(replay, "ROOT", root)
    logical = "outputs/experiment/runs/autodl_backup_attempt_01/checkpoint.pt"
    assert replay._relative(shared_run / "checkpoint.pt") == logical


def test_j_recovery_accepts_original_four_batches_and_restored_single_cursor() -> None:
    schedule = torch.arange(10).expand(5_000, 10).clone()
    labels = torch.arange(10)
    train_rows = torch.arange(10)
    assert replay._validate_j_cursor(schedule, 2_500, labels, train_rows).shape == (4, 10)
    assert replay._validate_j_cursor(schedule, 2_503, labels, train_rows, count=1).shape == (1, 10)


def test_j_recovery_rejects_wrong_restored_cursor() -> None:
    schedule = torch.arange(10).expand(5_000, 10).clone()
    labels = torch.arange(10)
    train_rows = torch.arange(10)
    with pytest.raises(RuntimeError, match="cursor or fixed schedule"):
        replay._validate_j_cursor(schedule, 2_502, labels, train_rows, count=1)
