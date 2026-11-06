from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import torch
import numpy as np
import pytest
import yaml

from scripts.research.lightgen import execute_lightgen_g1_fit20_box_b_convergence_v1 as convergence
from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_projected_box_fullpop_v1 as fullpop


def _schedule() -> torch.Tensor:
    return torch.arange(fullpop.UPDATE_BUDGET * fullpop.CLASS_COUNT).reshape(5, 1000, 10)


def _mapper_state(value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "network.0.weight": torch.full((256, 200), value, dtype=torch.float32),
        "network.0.bias": torch.full((256,), value, dtype=torch.float32),
        "network.2.weight": torch.full((40000, 256), value, dtype=torch.float32),
        "network.2.bias": torch.full((40000,), value, dtype=torch.float32),
    }


def _adam_state(step: int) -> dict:
    parameters = [0, 1, 2, 3]
    shapes = ((256, 200), (256,), (40000, 256), (40000,))
    state = {} if step == 0 else {
        index: {"step": torch.tensor(float(step)),
                "exp_avg": torch.zeros(shape, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(shape, dtype=torch.float32)}
        for index, shape in zip(parameters, shapes, strict=True)
    }
    return {
        "state": state,
        "param_groups": [{"params": parameters, "lr": 1.0e-4, "betas": (0.9, 0.999),
                          "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False,
                          "maximize": False}],
    }


def _checkpoint(schedule: torch.Tensor, completed: int) -> dict:
    history = [{"completed_update": index + 1,
                "sample_indices": schedule.reshape(5000, 10)[index].tolist(),
                "loss_timing": "computed_before_this_adam_update",
                "full_production_coefficients": True}
               for index in range(completed)]
    return {
        "schema": "lightgen_g1_projected_box_fullpop_checkpoint.v1",
        "task_id": fullpop.TASK_ID, "run_id": "new-run", "source_run_id": "v2-run",
        "source_task_id": "LIGHTGEN_G1_SLM_PROJECTED_MAPPING_V2",
        "source_mapper_updates": 10_000, "new_update_budget": 5000,
        "completed_updates": completed, "checkpoint_step": completed,
        "next_schedule_cursor": completed,
        "next_batch_indices": schedule[completed // 1000, completed % 1000].clone()
        if completed < 5000 else None,
        "mapper_state": _mapper_state(), "optimizer_state": _adam_state(completed), "history": history,
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [],
        "history_count": completed, "optimizer_deepcopied": True,
        "schedule_identity": "g1_projected_box_fullpop_schedule_seed_2026102871:5x1000x10",
    }


def _optimizer_state(step: int) -> dict:
    return _adam_state(step)


def test_development_population_excludes_all_fit20_rows() -> None:
    indices = fullpop._development_indices()
    excluded = fullpop._fit20_validation_indices()
    assert len(indices) == 1980
    assert not torch.isin(indices, excluded).any()
    assert torch.equal(indices[:3], torch.tensor([2, 3, 4]))
    assert torch.equal(indices[-3:], torch.tensor([1997, 1998, 1999]))


def test_population_independence_checks_exact_rows_not_only_whole_cohorts() -> None:
    train = torch.arange(400, dtype=torch.float64).to(torch.complex128).reshape(2, 200)
    validation = train + 1000
    holdout = train + 2000
    population = {
        "train": {"coefficients": train},
        "validation": {"coefficients": validation},
        "holdout": {"coefficients": holdout,
                    "labels": torch.arange(10).repeat_interleave(2),
                    "proposal_ids": torch.arange(20)},
    }
    assert fullpop._assert_population_independence(population) == 6
    population["holdout"]["coefficients"][0] = train[0]
    with pytest.raises(RuntimeError, match="duplicate coefficient rows"):
        fullpop._assert_population_independence(population)


def test_cloud_population_reads_fit20_positive_control_from_cloud_attempt(tmp_path, monkeypatch) -> None:
    train_coefficients = torch.arange(1_000_000, dtype=torch.float64).reshape(10_000, 100)
    validation_coefficients = train_coefficients[:2_000] + 2_000_000
    holdout_coefficients = train_coefficients[:2_000] + 4_000_000
    train = {"coefficients": train_coefficients.to(torch.complex128),
             "labels": torch.arange(10).repeat_interleave(1000),
             "proposal_ids": torch.arange(10_000), "source_role": "fixed_train"}
    validation = {"coefficients": validation_coefficients.to(torch.complex128),
                  "labels": torch.arange(10).repeat_interleave(200),
                  "proposal_ids": torch.arange(20_000, 22_000), "source_role": "fixed_validation"}
    holdout = {"coefficients": holdout_coefficients.to(torch.complex128),
               "labels": torch.arange(10).repeat_interleave(200),
               "proposal_ids": torch.arange(30_000, 32_000), "source_role": "new_holdout"}
    schedule = torch.arange(10_000).reshape(10, 1000).mT.repeat(5, 1).reshape(5, 1000, 10)
    monkeypatch.setattr(fullpop.projected, "_build_population",
                        lambda _config, _inputs: ({"train": train, "validation": validation,
                                                   "holdout": holdout,
                                                   "proposal_accounting": {},
                                                   "rejection_rule": "power only",
                                                   "power_bound_w": 1.0}, schedule))

    cloud_attempt = tmp_path / "cloud_attempt"
    cloud_attempt.mkdir()
    indices = fullpop._fit20_validation_indices()
    known = {"coefficients": validation["coefficients"][indices].clone(),
             "labels": validation["labels"][indices].clone(),
             "proposal_ids": validation["proposal_ids"][indices].clone(),
             "source_validation_indices": indices.clone()}
    torch.save(known, cloud_attempt / "fixed_population_and_schedule.pt")
    witness_path = tmp_path / "witness.pt"
    torch.save({"population_role": "known_reachable_fit20",
                "target_coefficients": known["coefficients"],
                "source_validation_indices": indices,
                "amplitudes": torch.zeros(20, 1, 200, 200)}, witness_path)
    resolved = {"cloud_attempt": cloud_attempt,
                "missing_windows_backup": tmp_path / "missing_windows_backup",
                "witness.pt": witness_path,
                "missing_windows_witness.pt": tmp_path / "missing_windows_witness.pt"}
    monkeypatch.setattr(fullpop, "_resolve", lambda value: resolved[str(value)])
    config = {"source": {"prior_fit20_b_cloud_attempt": "cloud_attempt",
                         "prior_fit20_b_windows_backup": "missing_windows_backup",
                         "fit20_cloud_witness_measurements": "witness.pt",
                         "fit20_witness_measurements": "missing_windows_witness.pt"}}

    population, saved_schedule = fullpop._population(config, {}, host="cloud")

    assert torch.equal(population["known_reachable"]["coefficients"], known["coefficients"])
    assert torch.equal(saved_schedule, schedule)
    assert population["role_contract"]["known_reachable"] == "20_saved_fit20_targets_and_witness; positive_control_only"


def test_box_loss_matches_analytic_value_and_derivative() -> None:
    result = fullpop._box_contract()
    assert result["value"] > 0
    assert result["gradient_max_abs_delta"] == 0
    assert abs(result["directional_derivative"] - result["finite_difference"]) < 1.0e-9


def test_fixed_projector_load_checks_source_identity_order_and_matrix(tmp_path) -> None:
    response = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0]], dtype=torch.complex128)
    scale = torch.ones(2, dtype=torch.float64)
    matrix = torch.eye(4, dtype=torch.float64)
    inverse = torch.eye(4, dtype=torch.float64)
    payload = {"schema": "lightgen_g1_slm_fixed_response_projector.v2",
               "task_id": fullpop.projected.TASK_ID,
               "response_matrix": response, "normalized_real_matrix": matrix,
               "full_right_inverse": inverse, "scale": scale,
               "real_imag_order": "per_port_interleaved_real_imag"}
    path = tmp_path / "fixed_response_and_projector.pt"
    torch.save(payload, path)

    loaded = fullpop._load_fixed_projector_evidence(path, response, scale, matrix)

    assert torch.equal(loaded["response_matrix"], response)
    assert torch.equal(loaded["matrix"], matrix)
    assert torch.equal(loaded["inverse"], inverse)
    assert loaded["matrix_max_abs_error"].item() == 0.0

    payload["real_imag_order"] = "port-major-blocked"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="identity or real-imag order"):
        fullpop._load_fixed_projector_evidence(path, response, scale, matrix)

    payload["real_imag_order"] = "per_port_interleaved_real_imag"
    payload["normalized_real_matrix"] = matrix.roll(1, dims=1)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="real-imag response matrix"):
        fullpop._load_fixed_projector_evidence(path, response, scale, matrix)


def test_fixed_projector_check_matches_numpy_and_rejects_wrong_inverse() -> None:
    matrix = torch.tensor([[1.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0]], dtype=torch.float64)
    inverse = torch.tensor([[1.0, 0.0],
                            [0.0, 0.5],
                            [0.0, 0.0]], dtype=torch.float64)

    result = fullpop._verify_fixed_projector(matrix, inverse)

    np.testing.assert_allclose(inverse.numpy(), np.linalg.pinv(matrix.numpy()), rtol=0.0, atol=1.0e-15)
    assert result["right_inverse_verified"] is True
    assert result["max_abs_AP_minus_I"] == 0.0
    assert result["max_abs_inverse_minus_moore_penrose"] == 0.0
    wrong_inverse = inverse.clone()
    wrong_inverse[2, 1] = 0.25
    with pytest.raises(RuntimeError, match="A@P=I or Moore-Penrose"):
        fullpop._verify_fixed_projector(matrix, wrong_inverse)


def test_ready_consumer_requires_fixed_projector_result() -> None:
    evidence_path = "outputs/v2/preflight/fixed_response_and_projector.pt"
    identity = SimpleNamespace(config_identity={
        "source": {"v2_fixed_response_projector_evidence": evidence_path}
    })
    projector = {"source_path": "/root/autodl-tmp/codex-original/" + evidence_path,
                 "right_inverse_verified": True,
                 "max_abs_AP_minus_I": 5.0e-14,
                 "max_abs_inverse_minus_moore_penrose": 6.0e-14,
                 "normalized_matrix_max_abs_error": 0.0}
    summary = {"status": "READY", "fixed_projector_verification": projector}

    assert fullpop._require_projector_preflight(summary, identity) == projector
    projector["source_path"] = str(fullpop._resolve(evidence_path))
    assert fullpop._require_projector_preflight(summary, identity) == projector
    projector["source_path"] = "outputs/v2/preflight/other_projector.pt"
    with pytest.raises(RuntimeError, match="READY evidence lacks a verified fixed response projector"):
        fullpop._require_projector_preflight(summary, identity)
    summary["fixed_projector_verification"] = {"right_inverse_verified": True}
    with pytest.raises(RuntimeError, match="READY evidence lacks a verified fixed response projector"):
        fullpop._require_projector_preflight(summary, identity)


def test_saved_decision_summary_allows_only_cross_platform_float_rounding() -> None:
    stored = {"maxima": {"native_cmos": 0.6479790916055795},
              "passed_rows": 0, "all_rows_passed": False}
    recomputed = {"maxima": {"native_cmos": 0.6479790916055788},
                  "passed_rows": 0, "all_rows_passed": False}

    assert fullpop.fit20._numeric_tree_matches(stored, recomputed, 1.0e-14)
    recomputed["all_rows_passed"] = True
    assert not fullpop.fit20._numeric_tree_matches(stored, recomputed, 1.0e-14)
    recomputed["all_rows_passed"] = False
    recomputed["maxima"]["native_cmos"] += 2.0e-14
    assert not fullpop.fit20._numeric_tree_matches(stored, recomputed, 1.0e-14)


def test_config_rejects_changed_training_budget_or_source(tmp_path, monkeypatch) -> None:
    config = copy.deepcopy(fullpop._load_config(fullpop.DEFAULT_CONFIG))
    config["training"]["learning_rate"] = 2.0e-4
    candidate = tmp_path / "g1_projected_box_fullpop_v1.yaml"
    candidate.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(fullpop, "DEFAULT_CONFIG", candidate.resolve())

    with pytest.raises(ValueError, match="fixed budget, or measurement contract changed"):
        fullpop._load_config(candidate)


def test_checkpoint_accepts_fresh_empty_adam_and_fixed_terminal() -> None:
    schedule = _schedule()
    for completed in (0, 5000):
        checkpoint = _checkpoint(schedule, completed)
        fullpop._validate_checkpoint(checkpoint, "new-run", "v2-run", completed,
                                    checkpoint["history"], schedule)


def test_checkpoint_rejects_wrong_adam_count_or_schedule_cursor() -> None:
    schedule = _schedule()
    checkpoint = _checkpoint(schedule, 1)
    checkpoint["optimizer_state"]["state"][0]["step"] = torch.tensor(2.0)
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        fullpop._validate_checkpoint(checkpoint, "new-run", "v2-run", 1,
                                    checkpoint["history"], schedule)
    checkpoint = _checkpoint(schedule, 1)
    checkpoint["next_batch_indices"] = schedule[0, 2].clone()
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        fullpop._validate_checkpoint(checkpoint, "new-run", "v2-run", 1,
                                    checkpoint["history"], schedule)


def test_fullpop_review_rejects_matching_nullspace_tamper_in_old_measurement_and_replay(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fullpop, "HOLDOUT_ROWS", 1)
    evidence = tmp_path / "evidence"
    review = tmp_path / "review"
    (evidence / "measurements" / "holdout").mkdir(parents=True)
    (evidence / "measurements" / "cuda_command_replay").mkdir(parents=True)
    (review / "command_replay").mkdir(parents=True)
    base = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    correct = base.clone()
    correct.flatten()[0] = 0.25
    shifted = correct.clone()
    shifted.flatten()[1] = 0.75
    new_replay = {
        "population_indices": torch.tensor([0]), "baseline_commands": base,
        "candidate_commands": correct, "batch_size": 1, "independent_process": True,
        "source_state_file": "mapper_source_states.pt",
        "source_state_roles": {"baseline": "frozen_v2_baseline", "candidate": "box_b_candidate"},
    }
    old_replay = {**new_replay, "candidate_commands": shifted}
    measured = {"population_indices": torch.tensor([0]), "baseline_commands": base,
                "candidate_commands": shifted}
    torch.save(new_replay, review / "command_replay" / "commands_0000.pt")
    torch.save(old_replay, evidence / "measurements" / "cuda_command_replay" / "commands_0000.pt")
    torch.save(measured, evidence / "measurements" / "holdout" / "holdout_0000.pt")
    source_states = {"frozen_v2_baseline": _mapper_state(), "box_b_candidate": _mapper_state(0.1)}
    with pytest.raises(RuntimeError, match="does not match stored measurements"):
        fullpop._check_repair_replays(evidence, review, source_states)


def test_fullpop_terminal_source_binding_rejects_replaced_mapper_tensor(monkeypatch) -> None:
    monkeypatch.setattr(fullpop, "_validate_checkpoint", lambda *_args, **_kwargs: None)
    state = _mapper_state()
    rng = torch.zeros(8, dtype=torch.uint8)
    terminal = {"mapper_state": copy.deepcopy(state), "optimizer_state": {"state": {}},
                "history": [], "torch_cpu_rng": rng.clone(), "torch_cuda_rng": []}
    fixed = copy.deepcopy(terminal)
    fullpop._require_terminal_checkpoint_bound(terminal, fixed, "run", "source", _schedule())
    terminal["mapper_state"]["network.0.weight"][0, 0] += 0.01
    with pytest.raises(RuntimeError, match="differs from fixed step-5000 checkpoint"):
        fullpop._require_terminal_checkpoint_bound(terminal, fixed, "run", "source", _schedule())


@pytest.mark.parametrize("tamper", ("mapper", "adam"))
def test_fullpop_verify_saved_consumer_rejects_replaced_terminal_state(tmp_path, monkeypatch,
                                                                      tamper: str) -> None:
    schedule = _schedule()
    rng = torch.zeros(8, dtype=torch.uint8)
    state = {"weight": torch.tensor([1.0])}
    fixed = {"run_id": "new-run", "source_mapper_updates": fullpop.SOURCE_MAPPER_UPDATES,
             "completed_updates": fullpop.UPDATE_BUDGET, "mapper_state": copy.deepcopy(state),
             "optimizer_state": {"state": {}}, "history": [],
             "torch_cpu_rng": rng.clone(), "torch_cuda_rng": []}
    terminal = copy.deepcopy(fixed)
    if tamper == "mapper":
        terminal["mapper_state"]["weight"][0] += 0.01
    else:
        terminal["optimizer_state"]["state"] = {"step": torch.tensor(1.0)}
    evidence = tmp_path / "evidence"
    (evidence / "checkpoints").mkdir(parents=True)
    torch.save(terminal, evidence / "terminal_checkpoint.pt")
    torch.save(fixed, evidence / "checkpoints" / "checkpoint_step_005000.pt")

    def json_payload(path):
        if path.name == "transfer_manifest.json":
            return {"run_id": "new-run"}
        if path.name == "transfer_receipt.json":
            return {"remote_run_dir": f"host/{fullpop.ATTEMPT_DIR.relative_to(fullpop.ROOT).as_posix()}",
                    "remote_alias": "autodl-monthly"}
        if path.name == "training_history.json":
            return {"task_id": fullpop.TASK_ID, "run_id": "new-run",
                    "updates": [{} for _ in range(fullpop.UPDATE_BUDGET)]}
        raise AssertionError(f"unexpected JSON consumer read before terminal binding: {path.name}")

    monkeypatch.setattr(fullpop, "_verify_transfer", lambda _path: {})
    monkeypatch.setattr(fullpop, "_assert_attempt", lambda _path: SimpleNamespace(run_id="new-run"))
    monkeypatch.setattr(fullpop, "_json", json_payload)
    monkeypatch.setattr(fullpop, "_load_config", lambda _path: {})
    monkeypatch.setattr(fullpop, "_load_inputs", lambda _config: {"source_mapper": {"run_id": "v2-run"}})
    monkeypatch.setattr(fullpop, "_validate_prior_b_repair", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fullpop, "_recompute_population_source", lambda *_args: {})
    monkeypatch.setattr(fullpop, "_load_saved_population", lambda *_args, **_kwargs: ({}, schedule))
    monkeypatch.setattr(fullpop, "_validate_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fullpop, "FIXED_CHECKPOINT_STEPS", (fullpop.UPDATE_BUDGET,))

    with pytest.raises(RuntimeError, match="differs from fixed step-5000 checkpoint"):
        fullpop._verify_saved(evidence, {"source_mapper": {"run_id": "v2-run"}}, tmp_path / "review")


def test_production_shard_consumer_rejects_wrong_target_and_intensity_encoding() -> None:
    run_id = "run-1"
    response = torch.zeros((100, 40000), dtype=torch.complex128)
    response[0, 0] = 1.0
    target = torch.zeros((1, 100), dtype=torch.complex128)
    target[0, 0] = 0.5
    command = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    command[0, 0, 0, 0] = 0.5
    coefficients = target.clone()
    raw = torch.ones((1, 1, 200, 200), dtype=torch.float64)
    outputs = {grid: {"coefficients": coefficients.clone(), "raw_cmos": raw.clone()}
               for grid in fullpop.GRIDS}
    direct = {grid: raw.clone() for grid in fullpop.GRIDS}
    row = fullpop.projected._error_row(outputs, target, direct, 0)
    row["row_passed"] = all(row[key] <= fullpop.GATE_LIMIT for key in fullpop.METRIC_KEYS)
    shard = {
        "schema": "lightgen_g1_projected_box_fullpop_holdout_shard.v1",
        "task_id": fullpop.TASK_ID, "run_id": run_id,
        "population_indices": torch.tensor([0]), "labels": torch.tensor([0]),
        "proposal_ids": torch.tensor([1]), "target_coefficients": target.clone(),
        "baseline_commands": command.clone(), "candidate_commands": command.clone(),
        "baseline_outputs": outputs, "candidate_outputs": outputs,
        "direct_raw_cmos": direct,
        "rows": {"frozen_v2_baseline": [row], "box_b_candidate": [row]},
    }
    replay = {"schema": "lightgen_g1_projected_box_fullpop_cuda_command_replay.v1",
              "task_id": fullpop.TASK_ID, "run_id": run_id,
              "population_indices": torch.tensor([0]),
              "baseline_commands": command.clone(), "candidate_commands": command.clone()}
    population = {"holdout": {"coefficients": target.clone(),
                               "labels": torch.tensor([0]), "proposal_ids": torch.tensor([1])}}
    replay.update({"source_state_file": "mapper_source_states.pt",
                   "source_state_roles": {"baseline": "frozen_v2_baseline",
                                          "candidate": "box_b_candidate"}})
    sources = {"frozen_v2_baseline": _mapper_state(), "box_b_candidate": _mapper_state(0.1)}
    assert len(fullpop._consume_shard(shard, replay, population, response, 0, 1, run_id,
                                      source_states=sources)[
        "box_b_candidate"]) == 1
    wrong_target = dict(shard)
    wrong_target["target_coefficients"] = target.clone()
    wrong_target["target_coefficients"][0, 0] += 1.0j
    with pytest.raises(RuntimeError, match="target, identity"):
        fullpop._consume_shard(wrong_target, replay, population, response, 0, 1, run_id,
                               source_states=sources)
    wrong_encoding = dict(shard)
    wrong_encoding["baseline_commands"] = command.square()
    wrong_encoding["candidate_commands"] = command.square()
    wrong_replay = {**replay, "baseline_commands": command.square(),
                    "candidate_commands": command.square()}
    with pytest.raises(RuntimeError, match="command response differs"):
        fullpop._consume_shard(wrong_encoding, wrong_replay, population, response, 0, 1, run_id,
                               source_states=sources)


def test_transfer_consumer_checks_remote_local_and_current_sizes(tmp_path) -> None:
    data = tmp_path / "shard.pt"
    data.write_bytes(b"raw")
    manifest = {"schema": "lightgen_g1_projected_box_fullpop_transfer.v1",
                "task_id": fullpop.TASK_ID, "run_id": "new-run",
                "files": [{"path": "shard.pt", "bytes": 3}],
                "file_count": 1, "bytes_total": 3}
    (tmp_path / "transfer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_bytes = (tmp_path / "transfer_manifest.json").stat().st_size
    receipt = {"status": "COMPLETE", "files": [
        {"path": "shard.pt", "status": "COPIED", "scp_exit_code": 0,
         "remote_size_bytes": 3, "local_size_bytes": 3},
        {"path": "transfer_manifest.json", "status": "COPIED", "scp_exit_code": 0,
         "remote_size_bytes": manifest_bytes, "local_size_bytes": manifest_bytes},
    ]}
    (tmp_path / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert fullpop._verify_transfer(tmp_path)["bytes_total"] == 3 + manifest_bytes
    receipt["files"][0]["remote_size_bytes"] = 4
    (tmp_path / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="transfer byte/status mismatch"):
        fullpop._verify_transfer(tmp_path)


def test_b_repair_remote_manifest_and_windows_receipt_are_consumed(tmp_path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"cuda-state")
    (tmp_path / "review_summary.json").write_text("{}", encoding="utf-8")
    manifest = convergence._build_review_transfer_manifest(tmp_path, "b-run")
    remote_source = tmp_path / "remote_source"
    remote_source.mkdir()
    remote_source.joinpath("source.pt").write_bytes(b"old-attempt")
    source_manifest = {"schema": "lightgen_g1_fit20_box_b_convergence_transfer.v1",
                       "task_id": convergence.TASK_ID, "run_id": "old-run",
                       "files": [{"path": "source.pt", "bytes": 11}],
                       "file_count": 1, "bytes_total": 11}
    remote_source.joinpath("transfer_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8")
    assert convergence._verify_remote_manifest(remote_source, "old-run")["file_count"] == 1
    backup = tmp_path / "backup"
    backup.mkdir()
    backup.joinpath("source.pt").write_bytes(source.read_bytes())
    backup.joinpath("review_summary.json").write_text("{}", encoding="utf-8")
    backup.joinpath("transfer_manifest.json").write_bytes(
        (tmp_path / "transfer_manifest.json").read_bytes())
    files = [
        {"path": item["path"], "status": "COPIED", "scp_exit_code": 0,
         "remote_size_bytes": item["bytes"], "local_size_bytes": item["bytes"]}
        for item in manifest["files"]
    ]
    transfer_manifest_size = (backup / "transfer_manifest.json").stat().st_size
    files.append({"path": "transfer_manifest.json", "status": "COPIED", "scp_exit_code": 0,
                  "remote_size_bytes": transfer_manifest_size,
                  "local_size_bytes": transfer_manifest_size})
    receipt = {"status": "COMPLETE", "remote_alias": "autodl-monthly",
               "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/"
               "20261027_lightgen_g1_fit20_box_b_convergence_v1/runs/review_repair_01",
               "files": files}
    (backup / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert convergence._verify_review_transfer(backup, "b-run")["status"] == "COMPLETE"
    remote_source.joinpath("source.pt").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        convergence._verify_remote_manifest(remote_source, "old-run")


def test_cloud_b_repair_uses_cloud_attempt_and_consumes_windows_receipt(tmp_path, monkeypatch) -> None:
    cloud_attempt = tmp_path / "cloud_attempt"
    cloud_attempt.mkdir()
    (cloud_attempt / "run_identity.json").write_text(
        json.dumps({"source_run_id": "fit20-run"}), encoding="utf-8")
    (cloud_attempt / "task_state.json").write_text(
        json.dumps({"state": fullpop.RUN_COMPLETED}), encoding="utf-8")
    (cloud_attempt / "final_decision.json").write_text(json.dumps({
        "lifecycle_state": fullpop.RUN_COMPLETED, "run_id": "b-run",
        "source_run_id": "fit20-run",
        "new_scientific_updates": 1000, "branch_terminal_step": 1500,
        "summary": {"passed_rows": fullpop.TARGET_COUNT,
                    "denominator": fullpop.TARGET_COUNT},
    }), encoding="utf-8")
    cloud_review = tmp_path / "cloud_review"
    cloud_review.mkdir()
    review_file = cloud_review / "review_summary.json"
    review_file.write_bytes(b"review")
    manifest = {
        "schema": "lightgen_g1_fit20_box_b_review_transfer.v1",
        "task_id": convergence.TASK_ID, "run_id": "b-run",
        "files": [{"path": "review_summary.json", "bytes": review_file.stat().st_size}],
        "file_count": 1, "bytes_total": review_file.stat().st_size,
    }
    manifest_path = cloud_review / "transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_size = manifest_path.stat().st_size
    receipt_path = tmp_path / "windows_review_receipt.json"
    receipt_path.write_text(json.dumps({
        "schema": "lightgen_g1_fit20_box_b_convergence_windows_reconsumption.v1",
        "task_id": convergence.TASK_ID, "run_id": "b-run",
        "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
        "scientific_status": fullpop.RUN_COMPLETED,
        "all_20_rows_recomputed_at_steps": [0, 500, 1000],
        "terminal_all_six_metrics_recomputed": True,
        "exact_cuda_command_replay_consumed": True,
        "scientific_updates_during_repair": 0,
        "repair_file_size_reconciliation": {
            "status": "COMPLETE", "file_count": 2,
            "bytes_total": review_file.stat().st_size + manifest_size,
            "files": [
                {"path": "review_summary.json", "remote_bytes": review_file.stat().st_size,
                 "receipt_local_bytes": review_file.stat().st_size,
                 "current_bytes": review_file.stat().st_size},
                {"path": "transfer_manifest.json", "remote_bytes": manifest_size,
                 "receipt_local_bytes": manifest_size, "current_bytes": manifest_size},
            ],
        },
        "recovery_raw_states_recomputed": {
            "all_raw_recovery_states_verified": True,
            "full_step4_history_saved": False,
        },
    }), encoding="utf-8")

    resolved = {
        "cloud_attempt": cloud_attempt,
        "missing_windows_backup": tmp_path / "missing_windows_backup",
        "cloud_review": cloud_review,
        "windows_review_receipt.json": receipt_path,
        "prior_config.yaml": tmp_path / "prior_config.yaml",
    }
    monkeypatch.setattr(fullpop, "_resolve", lambda value: resolved[str(value)])
    monkeypatch.setattr(convergence, "_load_config", lambda _path: {"fixed": True})
    monkeypatch.setattr(convergence, "_load_inputs", lambda _config: {})
    hosts = []
    monkeypatch.setattr(convergence, "_load_source", lambda _config, _inputs, host: (
        hosts.append(host) or {"identity": SimpleNamespace(run_id="fit20-run")}))
    monkeypatch.setattr(convergence, "_validate_windows_review", lambda _config, _source: {
        "status": "WINDOWS_REVIEW_RECONSUMPTION_PASSED"})
    remote_paths = []
    monkeypatch.setattr(convergence, "_verify_remote_manifest", lambda path: (
        remote_paths.append(path) or {"status": "REMOTE_MANIFEST_MATCHED"}))
    monkeypatch.setattr(convergence, "_verify_transfer", lambda _path: pytest.fail(
        "cloud B validation must not read the Windows-only full backup"))
    monkeypatch.setattr(convergence, "_assert_attempt", lambda _path: SimpleNamespace(run_id="b-run"))
    config = {"source": {
        "prior_fit20_b_cloud_attempt": "cloud_attempt",
        "prior_fit20_b_windows_backup": "missing_windows_backup",
        "prior_fit20_b_cloud_review": "cloud_review",
        "fit20_b_repair_windows_receipt": "windows_review_receipt.json",
        "prior_fit20_b_convergence_config": "prior_config.yaml",
    }}
    result = fullpop._validate_prior_b_repair(config, {}, host="cloud")
    assert result["run_id"] == "b-run"
    assert result["cloud_repair_manifest"]["status"] == "REMOTE_REPAIR_MANIFEST_MATCHED"
    assert result["predecessor_fit20_review_status"] == "WINDOWS_REVIEW_RECONSUMPTION_PASSED"
    assert hosts == ["cloud"]
    assert remote_paths == [cloud_attempt]

    review_file.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="cloud B repair file differs from its manifest"):
        fullpop._validate_prior_b_repair(config, {}, host="cloud")


def test_b_repair_recomputes_raw_recovery_and_rejects_rng_mismatch(tmp_path) -> None:
    schedule = convergence._build_continuation_schedule(
        fit20._build_schedule({}, convergence.SCHEDULE_SEED))
    rng = {"cpu": torch.zeros(8, dtype=torch.uint8),
           "cuda": [torch.zeros(8, dtype=torch.uint8)]}
    flat = schedule.reshape(1500, 10)
    history = [{"completed_update": index + 1,
                "branch_step": convergence.START_BRANCH_STEP + index + 1,
                "sample_indices": flat[convergence.START_BRANCH_STEP + index].tolist(),
                "arm": "B"} for index in range(3)]
    checkpoint = {
        "schema": "lightgen_g1_fit20_box_b_convergence_checkpoint.v1",
        "task_id": convergence.TASK_ID, "run_id": "new-run",
        "source_task_id": fit20.TASK_ID, "source_run_id": "old-run",
        "source_arm": "B", "source_branch_step": 500,
        "source_mapper_updates": 5000, "source_fit20_updates": 500,
        "completed_new_updates": 3, "new_update_budget": convergence.ADDED_UPDATES,
        "branch_step": 503, "checkpoint_step": 3, "next_schedule_cursor": 503,
        "next_batch_indices": flat[503].clone(),
        "optimizer_state": _optimizer_state(503), "history": history, "history_count": 3,
        "optimizer_deepcopied": True,
        "schedule_identity": "fit20_box_b_schedule_seed_2026102781:750x2x10",
        "torch_cpu_rng": rng["cpu"].clone(),
        "torch_cuda_rng": [item.clone() for item in rng["cuda"]],
    }
    mapper = {"weight": torch.tensor([1.0])}
    live = {"mapper": copy.deepcopy(mapper), "optimizer": _optimizer_state(504),
            "rng": copy.deepcopy(rng), "loss": 0.5, "coefficient_loss": 0.25,
            "sample_indices": flat[503].clone(),
            "engineering_adam_calls": 1}
    restored = copy.deepcopy(live)
    recovery_dir = tmp_path / "recovery_raw"
    recovery_dir.mkdir()
    torch.save(checkpoint, recovery_dir / "checkpoint_new_000003.pt")
    torch.save(live, recovery_dir / "original_process_new_000004.pt")
    torch.save(restored, recovery_dir / "independent_process_new_000004.pt")
    report = convergence._consume_recovery_raw(recovery_dir, schedule, "new-run", "old-run")
    assert report["all_raw_recovery_states_verified"] is True
    restored["rng"]["cpu"][0] = 1
    torch.save(restored, recovery_dir / "independent_process_new_000004.pt")
    with pytest.raises(RuntimeError, match="RNG"):
        convergence._consume_recovery_raw(recovery_dir, schedule, "new-run", "old-run")
