"""Read-only review and evidence repair for the closed G1 fit20 A/B attempt."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


RECOVERY_FILES = (
    "checkpoint_step_000003.pt",
    "original_process_step4.pt",
    "independent_process_step4.pt",
)
REVIEW_SCHEMA = "lightgen_g1_fit20_evidence_repair.v1"


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fit20._atomic_json(path, value)


def _same_rng(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not torch.is_tensor(left.get("cpu")) or not torch.is_tensor(right.get("cpu")):
        return False
    if not torch.equal(left["cpu"].cpu(), right["cpu"].cpu()):
        return False
    left_cuda, right_cuda = left.get("cuda"), right.get("cuda")
    return (isinstance(left_cuda, list) and isinstance(right_cuda, list)
            and len(left_cuda) == len(right_cuda)
            and all(torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
                    for a, b in zip(left_cuda, right_cuda, strict=True)))


def _optimizer_steps(state: Mapping[str, Any]) -> list[int]:
    optimizer_state = state.get("state")
    _require(isinstance(optimizer_state, Mapping), "recovery Adam state is malformed")
    steps = [int(torch.as_tensor(value["step"]).item())
             for value in optimizer_state.values()
             if isinstance(value, Mapping) and "step" in value]
    return steps


def _consume_recovery(recovery_root: Path, schedule: torch.Tensor, *,
                      expected_run_id: str | None = None) -> dict[str, Any]:
    report_path = recovery_root / "engineering_recovery.json"
    identity_path = recovery_root / "preflight_identity.json"
    report = _json(report_path)
    identity = _json(identity_path)
    run_identity = identity.get("run_identity", {})
    run_id = run_identity.get("run_id")
    _require(identity.get("task_id") == fit20.TASK_ID and isinstance(run_id, str)
             and (expected_run_id is None or run_id == expected_run_id)
             and report.get("scientific_updates") == 0
             and report.get("engineering_adam_calls_total") == 10
             and set(report.get("arms", {})) == set(fit20.ARM_NAMES),
             "raw recovery evidence identity or call count is incomplete")
    arm_results: dict[str, Any] = {}
    for arm in fit20.ARM_NAMES:
        arm_dir = recovery_root / arm
        cp_path, live_path, restored_path = (arm_dir / name for name in RECOVERY_FILES)
        _require(all(path.is_file() for path in (cp_path, live_path, restored_path)),
                 f"arm {arm} is missing one or more raw recovery files", FileNotFoundError)
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        live = torch.load(live_path, map_location="cpu", weights_only=False)
        restored = torch.load(restored_path, map_location="cpu", weights_only=False)
        _require(checkpoint.get("schema") == "lightgen_g1_projected_fit20_box_ab_checkpoint.v1"
                 and checkpoint.get("task_id") == fit20.TASK_ID
                 and checkpoint.get("arm") == arm and checkpoint.get("run_id") == run_id
                 and checkpoint.get("completed_updates") == 3
                 and checkpoint.get("checkpoint_step") == 3
                 and checkpoint.get("history_count") == 3
                 and checkpoint.get("optimizer_deepcopied") is True
                 and checkpoint.get("schedule_identity") ==
                     "fit20_paired_schedule_seed_2026102781:500x10"
                 and torch.equal(checkpoint.get("next_batch_indices"),
                                 schedule.reshape(500, fit20.CLASS_COUNT)[3]),
                 f"arm {arm} raw step-3 recovery checkpoint is not bound to the fixed schedule")
        cp_steps = _optimizer_steps(checkpoint["optimizer_state"])
        live_steps = _optimizer_steps(live["optimizer"])
        restored_steps = _optimizer_steps(restored["optimizer"])
        _require(len(cp_steps) == len(live_steps) == len(restored_steps) == 4
                 and all(value == 3 for value in cp_steps)
                 and all(value == 4 for value in live_steps + restored_steps),
                 f"arm {arm} raw recovery Adam states are not step 3 -> step 4")
        mapper_equal, mapper_delta = fit20._tree_max_delta(live.get("mapper"), restored.get("mapper"))
        adam_equal, adam_delta = fit20._tree_max_delta(live.get("optimizer"), restored.get("optimizer"))
        total_delta = abs(float(live.get("loss")) - float(restored.get("loss")))
        coefficient_delta = abs(float(live.get("coefficient_loss")) -
                                 float(restored.get("coefficient_loss")))
        checkpoint_rng = {"cpu": checkpoint.get("torch_cpu_rng"),
                          "cuda": checkpoint.get("torch_cuda_rng")}
        rng_equal = (_same_rng(live.get("rng", {}), restored.get("rng", {}))
                     and _same_rng(checkpoint_rng, live.get("rng", {}))
                     and _same_rng(checkpoint_rng, restored.get("rng", {})))
        schedule_equal = (torch.equal(live.get("sample_indices"), restored.get("sample_indices"))
                          and torch.equal(live.get("sample_indices"),
                                          schedule.reshape(500, fit20.CLASS_COUNT)[3]))
        _require(mapper_equal and adam_equal and total_delta <= fit20.RECOVERY_TOLERANCE
                 and coefficient_delta <= fit20.RECOVERY_TOLERANCE and rng_equal and schedule_equal
                 and live.get("engineering_adam_calls") == restored.get("engineering_adam_calls") == 1,
                 f"arm {arm} raw live/recovered fourth-step evidence fails the recovery contract")
        saved_summary = report["arms"][arm]
        _require(saved_summary.get("mapper_max_abs_delta") == mapper_delta
                 and saved_summary.get("adam_max_abs_delta") == adam_delta
                 and saved_summary.get("total_loss_abs_delta") == total_delta
                 and saved_summary.get("coefficient_loss_abs_delta") == coefficient_delta
                 and saved_summary.get("rng_exact") is rng_equal
                 and saved_summary.get("schedule_exact") is schedule_equal
                 and saved_summary.get("pre_updates") == 3
                 and saved_summary.get("engineering_adam_calls") == 5
                 and saved_summary.get("checkpoint_optimizer_deepcopied") is True
                 and saved_summary.get("frozen_optics_exact") is True,
                 f"arm {arm} raw recovery states disagree with their saved detailed summary")
        arm_results[arm] = {
            "step3_adam_steps": cp_steps, "live_step4_adam_steps": live_steps,
            "restored_step4_adam_steps": restored_steps,
            "mapper_max_abs_delta": mapper_delta, "adam_max_abs_delta": adam_delta,
            "total_loss_abs_delta": total_delta,
            "coefficient_loss_abs_delta": coefficient_delta,
            "rng_exact": rng_equal, "schedule_exact": schedule_equal,
            "frozen_optics_exact": saved_summary["frozen_optics_exact"],
            "engineering_adam_calls": saved_summary["engineering_adam_calls"],
        }
    _require(report.get("all_recovery_checks_passed") is True,
             "recovery aggregate disagrees with raw per-arm results")
    return {"run_id": run_id, "scientific_updates": 0,
            "engineering_adam_calls_total": report["engineering_adam_calls_total"],
            "all_raw_recovery_states_verified": True, "arms": arm_results}


def _verify_transfer(evidence_dir: Path) -> dict[str, Any]:
    manifest_path = evidence_dir / "transfer_manifest.json"
    receipt_path = evidence_dir / "transfer_receipt.json"
    manifest, receipt = _json(manifest_path), _json(receipt_path)
    _require(manifest.get("task_id") == fit20.TASK_ID
             and isinstance(manifest.get("files"), list)
             and receipt.get("status") == "COMPLETE"
             and isinstance(receipt.get("files"), list),
             "transfer manifest or receipt is invalid")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        rel = Path(str(item.get("path", "")))
        _require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() not in expected,
                 "transfer manifest has unsafe or duplicate paths")
        expected[rel.as_posix()] = int(item["bytes"])
    receipt_by_path: dict[str, Mapping[str, Any]] = {}
    for item in receipt["files"]:
        rel = Path(str(item.get("path", "")))
        _require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() not in receipt_by_path,
                 "transfer receipt has unsafe or duplicate paths")
        receipt_by_path[rel.as_posix()] = item
    manifest_size = manifest_path.stat().st_size
    expected["transfer_manifest.json"] = manifest_size
    _require(set(receipt_by_path) == set(expected),
             "transfer receipt file set differs from remote manifest plus its manifest file")
    checked = []
    for relative, remote_size in sorted(expected.items()):
        path = evidence_dir / Path(relative)
        item = receipt_by_path[relative]
        current_size = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_size
                 and int(item.get("local_size_bytes", -1)) == remote_size
                 and current_size == remote_size,
                 f"transfer receipt size/status mismatch for {relative}", ValueError)
        checked.append({"path": relative, "remote_bytes": remote_size,
                        "receipt_local_bytes": int(item["local_size_bytes"]),
                        "current_bytes": current_size})
    _require(len(expected) == manifest.get("file_count") + 1
             and sum(size for path, size in expected.items() if path != "transfer_manifest.json")
                 == manifest.get("bytes_total"),
             "transfer manifest inventory or byte total is inconsistent")
    return {"status": "COMPLETE", "file_count": len(checked),
            "bytes_total": sum(item["current_bytes"] for item in checked),
            "all_remote_local_current_sizes_match": True, "files": checked}


def _regenerate_commands(checkpoint: Mapping[str, Any], population: Mapping[str, Any],
                         inputs: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    mapper = projected._new_mapper(device, checkpoint["mapper_state"]).eval()
    targets = population["coefficients"].to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(population["coefficients"], inputs["scale"], device)
    matrix = inputs["matrix"].to(device=device, dtype=torch.float64)
    inverse = inputs["inverse"].to(device=device, dtype=torch.float64)
    scale = inputs["scale"].to(device=device, dtype=torch.float64)
    with torch.no_grad():
        commands, _ = projected._map_amplitude(mapper, features, targets, matrix, inverse, scale)
    _require(commands.dtype == torch.float32 and tuple(commands.shape) == (fit20.TARGET_COUNT, 1, 200, 200)
             and torch.isfinite(commands).all().item(), "terminal mapper replay produced malformed SLM commands")
    return commands.detach().cpu().contiguous()


def _fixed_population_matches(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = ("coefficients", "labels", "proposal_ids", "source_validation_indices")
    return (saved.get("role") == "known_reachable_fit20"
            and all(torch.equal(saved.get(key), expected[key]) for key in keys))


def _write_manifest(review_dir: Path) -> dict[str, Any]:
    files = []
    total = 0
    for path in sorted(item for item in review_dir.rglob("*") if item.is_file()
                       and item.name != "transfer_manifest.json"):
        relative = path.relative_to(review_dir).as_posix()
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size})
        total += size
    manifest = {"task_id": fit20.TASK_ID, "schema": "fit20_review_transfer_manifest.v1",
                "file_count": len(files), "bytes_total": total, "files": files}
    _atomic_json(review_dir / "transfer_manifest.json", manifest)
    return manifest


def _write_report(review_dir: Path, result: Mapping[str, Any]) -> None:
    command = result["commands"]
    recovery = result["recovery"]
    report = f"""# G1 fit20 收尾证据修订

## 结果与状态

旧A/B attempt保持关闭，历史 `VALID_FAILED / BOTH_FAIL` 保留。此次只复核现有终点和恢复证据，科学更新数为0；修订状态为 `{result['status']}`。

## 科学认识

终点权重重新生成的A/B输入指令均与原CUDA保存指令逐位一致，且两臂完整生产测量的六项误差均从原始输出重算。该结果补足终点权重到实际SLM指令的来源链，不改变旧实验结论。

## 目标缺口

此次只修订固定20条已知可达目标的证据链，不建立独立泛化、语义可读性或实物光路证据。旧A/B均未达到原六项误差阈值。

## 工程完成与验证

云端逐位指令回放最大差：A `{command['A']['maximum_absolute_amplitude_delta']:.3g}`，B `{command['B']['maximum_absolute_amplitude_delta']:.3g}`。恢复明细从六个原始状态文件重算；模型差、Adam差、损失差、RNG及样本日程逐项核验。工程Adam调用仍为 `{recovery['engineering_adam_calls_total']}`，科学更新为0。旧数据文件没有重生成。

## 决策影响与下一步

此次修订只确认旧attempt记录与实际输入指令、完整恢复状态及传输文件相符。任何B臂后继训练必须由独立READY预检和独立新任务执行；旧任务的 `VALID_FAILED / BOTH_FAIL` 不撤回。
"""
    (review_dir / "review_report.md").write_text(report, encoding="utf-8")


def review_saved(evidence_dir: Path, review_dir: Path, recovery_dir: Path,
                 inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "remote evidence repair requires the authorized CUDA host")
    _require(not review_dir.exists(), f"review output already exists: {review_dir}", FileExistsError)
    device = torch.device("cuda:0")
    fit20._set_numerics()
    identity = fit20._assert_attempt(evidence_dir)
    decision = _json(evidence_dir / "final_decision.json")
    state = _json(evidence_dir / "task_state.json")
    _require(state.get("state") == fit20.RUN_VALID_FAILED
             and decision.get("lifecycle_state") == fit20.RUN_VALID_FAILED
             and decision.get("branch") == "BOTH_FAIL"
             and decision.get("denominator") == fit20.TARGET_COUNT
             and decision.get("threshold") == fit20.GATE_LIMIT,
             "source fit20 attempt is not the closed historical BOTH_FAIL result")
    population, schedule = fit20._load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt")
    expected_population = fit20._fit_population(inputs)
    _require(_fixed_population_matches(population, expected_population),
             "saved fit20 population does not match fixed validation rows 200*y+{0,1}")
    review_dir.mkdir(parents=True)
    command_results: dict[str, Any] = {}
    witness_path = evidence_dir / "preflight_evidence" / "witness_qualification_measurements.pt"
    witness = torch.load(witness_path, map_location="cpu", weights_only=False)
    _require(witness.get("task_id") == fit20.TASK_ID
             and witness.get("population_role") == "known_reachable_fit20"
             and len(witness.get("rows", [])) == fit20.TARGET_COUNT
             and all(all(float(row[key]) <= fit20.GATE_LIMIT for key in fit20.METRIC_KEYS)
                     for row in witness["rows"]),
             "saved positive-control witness lacks complete six-metric qualification")
    for arm in fit20.ARM_NAMES:
        checkpoint = fit20._arm_terminal_checkpoint(evidence_dir, arm, identity.run_id)
        measurement_path = evidence_dir / "arms" / arm / "measurements" / "step_500.pt"
        payload = torch.load(measurement_path, map_location="cpu", weights_only=False)
        replayed = _regenerate_commands(checkpoint, population, inputs, device)
        consumed = fit20._consume_measurement_file(
            measurement_path, arm, population, inputs, checkpoint["mapper_state"],
            replayed, "exact")
        _require(fit20._numeric_tree_matches(decision["arms"][arm]["summary"], consumed["summary"])
                 and fit20._numeric_tree_matches(decision["arms"][arm]["rows"], consumed["rows"]),
                 f"arm {arm} raw output recomputation differs from the closed decision")

        counterexample = dict(payload)
        counterexample.update({"amplitudes": witness["amplitudes"],
                               "outputs": witness["outputs"],
                               "direct_raw_cmos": witness["direct_raw_cmos"],
                               "rows": witness["rows"],
                               "summary": projected._summarize_rows(witness["rows"])})
        old_accepted = True
        try:
            fit20._consume_measurement_file(measurement_path, arm, population, inputs,
                                            checkpoint["mapper_state"], payload_override=counterexample)
        except (RuntimeError, ValueError):
            old_accepted = False
        _require(old_accepted,
                 f"arm {arm} actual witness counterexample no longer reproduces the historical consumer gap")
        repaired_rejected = False
        try:
            fit20._consume_measurement_file(measurement_path, arm, population, inputs,
                                            checkpoint["mapper_state"], replayed, "exact",
                                            payload_override=counterexample)
        except (RuntimeError, ValueError):
            repaired_rejected = True
        _require(repaired_rejected,
                 f"arm {arm} repaired consumer accepted a witness command with the terminal mapper")
        command_results[arm] = {
            "terminal_checkpoint_updates": 500, "history_count": 500,
            "maximum_absolute_amplitude_delta": consumed["command_max_abs_amplitude_delta"],
            "exact_cuda_replay": consumed["command_check"] == "exact",
            "all_six_metric_rows_recomputed": True,
            "historical_consumer_accepted_witness_substitution": old_accepted,
            "repaired_consumer_rejected_witness_substitution": repaired_rejected,
            "recomputed_summary": consumed["summary"],
        }

    raw_root = review_dir / "recovery_raw"
    raw_root.mkdir()
    source_report = recovery_dir.parent / "engineering_recovery.json"
    source_identity = recovery_dir.parent / "preflight_identity.json"
    _require(source_report.is_file() and source_identity.is_file(),
             "remote raw recovery summary or identity is missing", FileNotFoundError)
    shutil.copy2(source_report, raw_root / "engineering_recovery.json")
    shutil.copy2(source_identity, raw_root / "preflight_identity.json")
    for arm in fit20.ARM_NAMES:
        target_dir = raw_root / arm
        target_dir.mkdir()
        for name in RECOVERY_FILES:
            source_dir = recovery_dir / arm
            if name == "checkpoint_step_000003.pt":
                source_dir = source_dir / "engineering_recovery"
            source = source_dir / name
            _require(source.is_file(), f"raw recovery file is missing: {source}", FileNotFoundError)
            shutil.copy2(source, target_dir / name)
    recovery_result = _consume_recovery(raw_root, schedule)
    result = {
        "schema": REVIEW_SCHEMA, "task_id": fit20.TASK_ID,
        "source_run_id": identity.run_id, "status": "REMOTE_REVIEW_REPAIR_READY_FOR_WINDOWS_RECONSUMPTION",
        "historical_lifecycle_state": state["state"], "historical_branch": decision["branch"],
        "scientific_updates": 0,
        "population_role": "known_reachable_fit20",
        "source_population_exact": True,
        "commands": command_results,
        "recovery": recovery_result,
        "original_attempt_modified": False,
    }
    _atomic_json(review_dir / "review_summary.json", result)
    _write_report(review_dir, result)
    manifest = _write_manifest(review_dir)
    result["transfer_manifest"] = manifest
    return result


def verify_review(evidence_dir: Path, review_dir: Path,
                  inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(evidence_dir.is_dir() and review_dir.is_dir(),
             "old evidence backup or evidence repair backup is missing", FileNotFoundError)
    old_transfer = _verify_transfer(evidence_dir)
    repair_transfer = _verify_transfer(review_dir)
    identity = fit20._assert_attempt(evidence_dir)
    decision = _json(evidence_dir / "final_decision.json")
    state = _json(evidence_dir / "task_state.json")
    review = _json(review_dir / "review_summary.json")
    _require(review.get("schema") == REVIEW_SCHEMA and review.get("task_id") == fit20.TASK_ID
             and review.get("source_run_id") == identity.run_id
             and review.get("historical_lifecycle_state") == state.get("state") == fit20.RUN_VALID_FAILED
             and review.get("historical_branch") == decision.get("branch") == "BOTH_FAIL"
             and review.get("scientific_updates") == 0
             and review.get("original_attempt_modified") is False,
             "remote review repair is not linked to the closed original attempt")
    population, schedule = fit20._load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt")
    expected_population = fit20._fit_population(inputs)
    _require(_fixed_population_matches(population, expected_population)
             and review.get("source_population_exact") is True,
             "Windows review found a changed fit20 source population")
    command_results: dict[str, Any] = {}
    witness = torch.load(evidence_dir / "preflight_evidence" / "witness_qualification_measurements.pt",
                         map_location="cpu", weights_only=False)
    for arm in fit20.ARM_NAMES:
        checkpoint = fit20._arm_terminal_checkpoint(evidence_dir, arm, identity.run_id)
        history = fit20._read_history(evidence_dir / "arms" / arm / "training_history.json",
                                      arm, identity.run_id, schedule)
        _require(len(history) == 500, f"arm {arm} history is not exactly 500 updates")
        measurement_path = evidence_dir / "arms" / arm / "measurements" / "step_500.pt"
        replayed = _regenerate_commands(checkpoint, population, inputs, torch.device("cpu"))
        consumed = fit20._consume_measurement_file(
            measurement_path, arm, population, inputs, checkpoint["mapper_state"],
            replayed, "response")
        _require(fit20._numeric_tree_matches(decision["arms"][arm]["summary"], consumed["summary"])
                 and fit20._numeric_tree_matches(decision["arms"][arm]["rows"], consumed["rows"]),
                 f"arm {arm} Windows raw metric recomputation differs from its historical decision")
        saved_remote = review.get("commands", {}).get(arm, {})
        _require(saved_remote.get("exact_cuda_replay") is True
                 and saved_remote.get("historical_consumer_accepted_witness_substitution") is True
                 and saved_remote.get("repaired_consumer_rejected_witness_substitution") is True
                 and fit20._numeric_tree_matches(saved_remote.get("recomputed_summary"),
                                                 consumed["summary"]),
                 f"arm {arm} remote and Windows command/output evidence disagree")
        counterexample = dict(torch.load(measurement_path, map_location="cpu", weights_only=False))
        counterexample.update({"amplitudes": witness["amplitudes"],
                               "outputs": witness["outputs"],
                               "direct_raw_cmos": witness["direct_raw_cmos"],
                               "rows": witness["rows"],
                               "summary": projected._summarize_rows(witness["rows"])})
        historical_accept = True
        try:
            fit20._consume_measurement_file(measurement_path, arm, population, inputs,
                                            checkpoint["mapper_state"], payload_override=counterexample)
        except (RuntimeError, ValueError):
            historical_accept = False
        repaired_reject = False
        try:
            fit20._consume_measurement_file(measurement_path, arm, population, inputs,
                                            checkpoint["mapper_state"], replayed, "response",
                                            payload_override=counterexample)
        except (RuntimeError, ValueError):
            repaired_reject = True
        _require(historical_accept and repaired_reject,
                 f"arm {arm} witness-substitution counterexample did not distinguish old and repaired consumers")
        command_results[arm] = {
            "history_updates": len(history), "adam_parameter_steps": [500, 500, 500, 500],
            "windows_cpu_to_optical_response_relative_error_max":
                consumed["command_response_relative_error_max"],
            "windows_cpu_to_cuda_command_max_abs_delta": consumed["command_max_abs_amplitude_delta"],
            "all_six_metric_rows_recomputed": True,
            "historical_consumer_accepted_witness_substitution": historical_accept,
            "repaired_consumer_rejected_witness_substitution": repaired_reject,
        }
    recovery_result = _consume_recovery(review_dir / "recovery_raw", schedule)
    _require(fit20._numeric_tree_matches(review.get("recovery"), recovery_result),
             "raw recovery reconsumption differs from the remote repair summary")
    receipt = {
        "schema": "lightgen_g1_fit20_windows_review_reconsumption.v1",
        "task_id": fit20.TASK_ID, "source_run_id": identity.run_id,
        "status": "WINDOWS_REVIEW_RECONSUMPTION_PASSED",
        "historical_state_preserved": fit20.RUN_VALID_FAILED,
        "scientific_updates": 0,
        "old_attempt_transfer": old_transfer, "repair_transfer": repair_transfer,
        "source_population_exact": True,
        "all_six_metrics_independently_recomputed": True,
        "recovery_raw_states_recomputed": recovery_result,
        "commands": command_results,
    }
    _atomic_json(review_dir / "windows_review_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("review-saved", "verify-review"))
    parser.add_argument("--config", type=Path, default=fit20.DEFAULT_CONFIG)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path)
    args = parser.parse_args(argv)
    config = fit20._load_config(args.config.resolve())
    inputs = fit20._load_inputs(config)
    if args.mode == "review-saved":
        _require(args.recovery_dir is not None,
                 "review-saved requires an explicit --recovery-dir", ValueError)
        result = review_saved(args.evidence_dir.resolve(), args.review_dir.resolve(),
                              args.recovery_dir.resolve(), inputs)
    else:
        _require(args.recovery_dir is None,
                 "verify-review consumes the recovery files in --review-dir", ValueError)
        result = verify_review(args.evidence_dir.resolve(), args.review_dir.resolve(), inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
