"""Qualify one static RS transpose-scheduling change at zero optimizer updates."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_rs_convolution import (
    FinitePixelRayleighSommerfeldCached,
)
from scripts.research.lightgen import execute_chen_lightgen_400_rs_cost_attribution_v1 as prior
from scripts.research.lightgen import execute_chen_lightgen_400_fast_rs_compact_v1 as compact


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_RS_STATIC_TRANSPOSE_V1"
ORDER = 12
PAIR_BLOCK = 4
PRECISION = "complex64"
MEASUREMENT_ORDER = ("original_1", "optimized_1", "optimized_2", "original_2")
OUTPUT_LIMIT_BYTES = 512 * 1024**2


class HarnessError(RuntimeError):
    """An explicit input, numerical comparison, or saved-evidence contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _safe_json(value: Any) -> Any:
    return prior._safe_json(value)


def _write_json(path: Path, value: dict[str, Any], *, new: bool = False) -> None:
    if new:
        from ldm_project.v2.evidence import atomic_write_new_json

        atomic_write_new_json(path, _safe_json(value))
    else:
        atomic_write_json(path, _safe_json(value))


def _explicit_paths(args: argparse.Namespace) -> dict[str, str]:
    names = ("source_checkpoint", "source_report", "source_start_state",
             "source_endpoint_checkpoint", "fixed_population", "config", "data",
             "evaluator", "qualification_report", "structure_coefficient",
             "qualification_dir", "cost_reference_dir", "reference_rs_source", "output_dir")
    return {name: str(_resolve(getattr(args, name)).resolve()) for name in names}


def _load_references(args: argparse.Namespace) -> dict[str, Any]:
    paths = _explicit_paths(args)
    previous_qualification = prior._validate_previous_evidence(args)
    cost_dir = _resolve(args.cost_reference_dir)
    report_path = cost_dir / "final_report.json"
    evidence_path = cost_dir / "cost_attribution_report.json"
    first_case_path = cost_dir / "measurements" / "checkpoint_on_1.pt"
    for path in (report_path, evidence_path, first_case_path):
        _require(path.is_file(), f"required saved cost reference is missing: {path}")
    previous_report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _require(previous_report.get("status") == "VALID_COMPLETED_COST_ATTRIBUTION"
             and previous_report.get("adam_calls") == previous_report.get("optimizer_updates") == 0,
             "saved cost reference is incomplete or contains optimizer activity")
    _require(evidence.get("input_identity", {}).get("precision") == PRECISION
             and evidence.get("input_identity", {}).get("branch") == "B"
             and evidence.get("input_identity", {}).get("logical_batch") == prior.LOGICAL_BATCH
             and evidence.get("input_identity", {}).get("microbatch") == prior.MICROBATCH
             and evidence.get("cases", {}).get("checkpoint_on_1", {}).get("checkpoint_propagation") is True
             and evidence.get("cases", {}).get("checkpoint_on_2", {}).get("checkpoint_propagation") is True,
             "saved cost reference does not match the fixed complex64 B microbatch2 condition")
    saved_paths = evidence.get("explicit_inputs")
    _require(isinstance(saved_paths, dict), "saved cost report lacks explicit input identities")
    field_map = {
        "source_checkpoint": "source_checkpoint", "source_report": "source_report",
        "source_start_state": "source_start_state",
        "source_endpoint_checkpoint": "source_endpoint_checkpoint",
        "fixed_population": "fixed_population", "config": "config", "data": "data",
        "evaluator": "evaluator", "qualification_report": "qualification_report",
        "structure_coefficient": "structure_coefficient", "qualification_dir": "qualification_dir",
    }
    for cli_name, saved_name in field_map.items():
        _require(saved_name in saved_paths
                 and prior._manifest_identity(paths[cli_name]) == prior._manifest_identity(saved_paths[saved_name]),
                 f"explicit {cli_name} differs from the previous cost run")
    source_path = _resolve(args.reference_rs_source)
    _require(source_path.is_file(), f"pre-change RS source snapshot is missing: {source_path}")
    source_text = source_path.read_text(encoding="utf-8")
    _require("bool(transposed.any())" in source_text,
             "the explicit reference source does not contain the saved pre-change predicate")
    payload = torch.load(first_case_path, map_location="cpu", weights_only=False)
    rows = evidence.get("input_identity", {}).get("rows")
    _require(payload.get("schema") == "chen_lightgen_400_rs_cost_case.v1"
             and payload.get("case") == "checkpoint_on_1"
             and isinstance(rows, list) and torch.equal(payload.get("rows"), torch.tensor(rows).long())
             and payload.get("checkpoint_propagation") is True
             and isinstance(payload.get("gradients"), dict)
             and isinstance(payload.get("w_per_pixel"), torch.Tensor),
             "saved original batch10 reference is incomplete or misaligned")
    reference_case = {"loss": float(payload["loss"]), "gradients": payload["gradients"],
                      "w_per_pixel": payload["w_per_pixel"]}
    return {"paths": paths, "qualification": previous_qualification,
            "previous_report": previous_report, "previous_evidence": evidence,
            "reference_case": reference_case, "reference_rows": payload["rows"].long().cpu()}


def _load_reference_module(source_path: Path):
    module_name = "ldm_project.v2.application.research.lightgen._static_transpose_reference_rs"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    _require(spec is not None and spec.loader is not None,
             "cannot create a package-scoped import for the explicit reference source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_model(context: dict[str, Any], variant: str, reference_module, device: torch.device):
    model = compact._make_model(context["state_phases"]["source_step_30000"], device,
                                PRECISION, checkpoint_propagation=True)
    operator_type = (reference_module.FinitePixelRayleighSommerfeldCached
                     if variant == "original" else FinitePixelRayleighSommerfeldCached)
    model.propagator.operator = operator_type(
        ORDER, distance_m=model.config.distance_m, pair_block=PAIR_BLOCK,
        compute_precision=PRECISION,
    ).to(device)
    compact.prior._set_trainable(model)
    return model


class _CorrectedTrace(prior._OperationTrace):
    """Retain production call counts and read the PyTorch 2.5 device-time field."""

    def _profile_call(self, direction, original, instance, args, kwargs):
        if not self.profile or direction in self._profile_seen or not torch.cuda.is_available():
            return original(instance, *args, **kwargs)
        self._profile_seen.add(direction)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        started = time.perf_counter()
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    profile_memory=False, with_stack=False) as profiler:
            result = original(instance, *args, **kwargs)
        torch.cuda.synchronize()
        events = profiler.key_averages()
        field_available = all(hasattr(item, "self_device_time_total") for item in events)
        operators = []
        if field_available:
            for item in events:
                device_us = float(item.self_device_time_total)
                inclusive_device_us = float(item.device_time_total)
                cpu_us = float(item.self_cpu_time_total)
                if device_us > 0.0 or inclusive_device_us > 0.0 or cpu_us > 0.0:
                    operators.append({"operator": item.key, "calls": int(item.count),
                                      "self_device_time_us": device_us,
                                      "inclusive_device_time_us": inclusive_device_us,
                                      "self_cpu_time_us": cpu_us})
            operators.sort(key=lambda item: item["self_device_time_us"], reverse=True)
        device_time_present = field_available and any(
            item.get("self_device_time_us", 0.0) > 0.0
            or item.get("inclusive_device_time_us", 0.0) > 0.0 for item in operators
        )
        self.profile_summaries.append({
            "direction": direction,
            "status": "DEVICE_TIMING_CAPTURED" if device_time_present else "DEVICE_TIMING_UNAVAILABLE",
            "device_time_field": "self_device_time_total" if field_available else None,
            "elapsed_wall_seconds_with_profiler": time.perf_counter() - started,
            "profile_scope": "one production cached RS operator call; nested totals are not summed",
            "operators": operators,
        })
        return result

    def __enter__(self):
        super().__enter__()
        original_spectra = self.cached_type._spectra_for_pairs

        def spectra(instance, *args, **kwargs):
            if instance is self.operator:
                if "has_transposed_pair" in kwargs:
                    self._count("pair_blocks_with_transpose" if kwargs["has_transposed_pair"]
                                else "pair_blocks_without_transpose")
                else:
                    self._count("legacy_device_pair_predicates")
            return original_spectra(instance, *args, **kwargs)

        self._add_patch(self.cached_type, "_spectra_for_pairs", spectra)
        return self


def _load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    context = prior._load_context(args, device)
    _require(context["checkpoint"].get("completed_updates") == 30_000
             and context["endpoint_payload"].get("completed_updates") == 500,
             "loaded optical states do not match the explicit step 30000 and step 500 inputs")
    expected_rows = _load_references(args)["reference_rows"]
    _require(torch.equal(context["batch10_rows"].long().cpu(), expected_rows),
             "loaded fixed-fit batch10 rows differ from the saved cost identity")
    _require(float(context["beta"]) == float(args.expected_beta)
             and float(context["eta"]) == float(args.expected_eta),
             "loaded loss coefficients differ from the explicit previous-run values")
    return context


def _save_worker(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save_new(path, payload)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(loaded.get("task_id") == TASK_ID
             and loaded.get("variant") == payload.get("variant")
             and torch.equal(loaded.get("rows"), payload.get("rows"))
             and math.isfinite(float(loaded.get("loss", math.nan)))
             and torch.isfinite(loaded["w_per_pixel"]).all()
             and all(torch.isfinite(value).all() for value in loaded["gradients"].values()),
             f"saved worker measurement did not pass read-back validation: {path.name}")


def _worker(args: argparse.Namespace, mode: str, variant: str, case: str | None) -> dict[str, Any]:
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required for this optical cost qualification")
    refs = _load_references(args)
    reference_module = _load_reference_module(_resolve(args.reference_rs_source))
    context = _load_context(args, device)
    rows = context["batch10_rows"]
    model = _make_model(context, variant, reference_module, device)
    state = prior._state_snapshot(model)
    result_path = _resolve(args.output_dir) / "measurements"
    result_path.mkdir(parents=True, exist_ok=True)
    with _CorrectedTrace(model, profile=mode == "profile") as trace:
        if mode == "preflight":
            result = prior._run_partial(model, context["pixels"], rows, context["beta"],
                                        context["eta"], device)
            label = f"preflight_{variant}"
            _save_worker(result_path / f"{label}.pt", {
                "schema": "chen_lightgen_400_rs_static_transpose_microbatch2.v1",
                "task_id": TASK_ID, "variant": variant, "rows": rows[:prior.MICROBATCH].cpu(),
                "loss": result["loss_weighted_into_batch10"],
                "gradients": result["gradients"], "w_per_pixel": result["w_per_pixel"],
                "operator_call_counts": trace.counts.copy(),
                "cache_info": model.propagator.operator.cache_info,
                "transpose_schedule_build_seconds": getattr(
                    model.propagator.operator, "_transpose_schedule_build_seconds", 0.0),
                "transpose_schedule_build_count": getattr(
                    model.propagator.operator, "_transpose_schedule_build_count", 0),
                "transpose_schedule_cache_hits": getattr(
                    model.propagator.operator, "_transpose_schedule_cache_hits", 0),
            })
            trace_payload = {"operator_call_counts": trace.counts.copy(),
                             "cache_info": model.propagator.operator.cache_info}
        elif mode == "measure":
            _require(case in MEASUREMENT_ORDER, "measurement worker has an invalid case name")
            warm = prior._run_partial(model, context["pixels"], rows, context["beta"],
                                      context["eta"], device)
            _require(model.propagator.operator.cache_info["cache_bytes"] > 0,
                     "production microbatch warmup did not build a cached RS kernel")
            trace.reset_counts()
            result = prior._run_batch10(model, context["pixels"], rows, context["beta"],
                                        context["eta"], device, trace)
            payload = {"schema": "chen_lightgen_400_rs_static_transpose_batch10.v1",
                       "task_id": TASK_ID, "case": case, "variant": variant,
                       "rows": rows.cpu().long(), "loss": result["loss"],
                       "gradients": result["gradients"], "w_per_pixel": result["w_per_pixel"],
                       "elapsed_seconds": result["elapsed_seconds"],
                       "peak_reserved_vram_bytes": result["peak_reserved_vram_bytes"],
                       "artifact_export_seconds": result["artifact_export_seconds"],
                       "gradient_transfer_wall_seconds": result["gradient_transfer_wall_seconds"],
                       "final_cuda_sync_wall_seconds": result["final_cuda_sync_wall_seconds"],
                       "stage_timing": result["stage_timing"],
                       "operator_cuda_intervals": result["operator_cuda_intervals"],
                       "operator_host_wall_seconds": result["operator_host_wall_seconds"],
                       "operator_call_counts": trace.counts.copy(),
                       "cache_info": model.propagator.operator.cache_info,
                       "transpose_schedule_build_seconds": getattr(
                           model.propagator.operator, "_transpose_schedule_build_seconds", 0.0),
                       "transpose_schedule_build_count": getattr(
                           model.propagator.operator, "_transpose_schedule_build_count", 0),
                       "transpose_schedule_cache_hits": getattr(
                           model.propagator.operator, "_transpose_schedule_cache_hits", 0),
                       "cache_build_seconds": trace.cache_build.get("host_seconds")
                       if trace.cache_build else None}
            _save_worker(result_path / f"{case}.pt", payload)
            trace_payload = {"elapsed_seconds": result["elapsed_seconds"],
                             "operator_call_counts": trace.counts.copy(),
                             "cache_build_seconds": payload["cache_build_seconds"]}
        elif mode == "profile":
            _require(case is None, "profile worker must not name a throughput case")
            profile_started = time.perf_counter()
            result = prior._run_partial(model, context["pixels"], rows, context["beta"],
                                        context["eta"], device)
            profiled_wall = time.perf_counter() - profile_started
            _require(len(trace.profile_summaries) == 2,
                     "bounded profiler did not capture one forward and one adjoint call")
            label = f"profile_{variant}"
            _save_worker(result_path / f"{label}.pt", {
                "schema": "chen_lightgen_400_rs_static_transpose_profile.v1",
                "task_id": TASK_ID, "variant": variant, "rows": rows[:prior.MICROBATCH].cpu(),
                "loss": result["loss_weighted_into_batch10"], "gradients": result["gradients"],
                "w_per_pixel": result["w_per_pixel"],
                "operator_call_counts": trace.counts.copy(),
                "profile_summaries": trace.profile_summaries,
                "profiled_microbatch_wall_seconds_excluded_from_timing": profiled_wall,
            })
            trace_payload = {"operator_call_counts": trace.counts.copy(),
                             "profile_summaries": trace.profile_summaries}
        else:
            raise HarnessError(f"unknown worker mode: {mode}")
    _require(prior._state_matches(model, state),
             "model parameters or buffers changed during a zero-update worker")
    return {"status": "WORKER_COMPLETED", "mode": mode, "variant": variant,
            "state_unchanged": True, "optimizer_calls": 0, "adam_calls": 0,
            "optimizer_updates": 0, "generation_samples": 0,
            "evaluator_forward_calls": 0, **trace_payload}


def _compare_microbatch(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return prior._compare_case(
        {"loss": float(candidate["loss"]), "gradients": candidate["gradients"],
         "w_per_pixel": candidate["w_per_pixel"]},
        {"loss": float(reference["loss"]), "gradients": reference["gradients"],
         "w_per_pixel": reference["w_per_pixel"]},
    )


def _load_tensor(path: Path, *, schema: str, case: str | None = None) -> dict[str, Any]:
    _require(path.is_file(), f"complete measurement file is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == schema and payload.get("task_id") == TASK_ID
             and (case is None or payload.get("case") == case)
             and isinstance(payload.get("rows"), torch.Tensor)
             and isinstance(payload.get("gradients"), dict)
             and isinstance(payload.get("w_per_pixel"), torch.Tensor)
             and math.isfinite(float(payload.get("loss", math.nan)))
             and torch.isfinite(payload["w_per_pixel"]).all()
             and all(torch.isfinite(value).all() for value in payload["gradients"].values()),
             f"saved measurement is incomplete or has a different identity: {path.name}")
    return payload


def _progress(output_dir: Path, stage: str, **values: Any) -> None:
    payload = {"task_id": TASK_ID, "stage": stage, "updated_at_unix": time.time(),
               "optimizer_updates": 0, "adam_calls": 0, "generation_samples": 0,
               "evaluator_forward_calls": 0, **values}
    _write_json(output_dir / "progress.json", payload)
    _write_json(output_dir / "heartbeat.json", payload)


def _worker_args(args: argparse.Namespace, mode: str, variant: str,
                 case: str | None = None) -> list[str]:
    command = [sys.executable, "-m",
               "scripts.research.lightgen.execute_chen_lightgen_400_rs_static_transpose_v1"]
    for name in ("source_checkpoint", "source_report", "source_start_state",
                 "source_endpoint_checkpoint", "fixed_population", "config", "data",
                 "evaluator", "qualification_report", "structure_coefficient",
                 "qualification_dir", "cost_reference_dir", "reference_rs_source",
                 "output_dir"):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    command.extend(("--expected-beta", str(args.expected_beta), "--expected-eta", str(args.expected_eta),
                    "--worker-mode", mode, "--variant", variant))
    if case is not None:
        command.extend(("--case", case))
    return command


def _run_worker(args: argparse.Namespace, mode: str, variant: str,
                case: str | None = None) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    logs_dir = output_dir / "worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    name = f"{mode}_{variant}" + (f"_{case}" if case else "")
    log_path = logs_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(_worker_args(args, mode, variant, case), cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
        started = time.perf_counter()
        while process.poll() is None:
            _progress(output_dir, f"{mode}_{variant}", case=case,
                      elapsed_seconds=time.perf_counter() - started,
                      worker_log=log_path.name)
            time.sleep(20.0)
        code = process.returncode
    if code != 0:
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        if progress.get("status") == "PREFLIGHT_BLOCKED_RESOURCE":
            raise torch.cuda.OutOfMemoryError(
                f"{name} worker was blocked by CUDA allocation; inspect {log_path.name}"
            )
        raise HarnessError(f"{name} worker exited {code}; inspect {log_path.name}")
    return {"worker": name, "log": str(log_path), "exit_code": code,
            "elapsed_seconds": time.perf_counter() - started}


def _preflight(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    refs = _load_references(args)
    existing = (output_dir / "preflight_receipt.json", output_dir / "final_report.json")
    _require(not any(path.exists() for path in existing),
             "refusing to replace existing preflight or completed result artifacts")
    _progress(output_dir, "preflight_start")
    workers = []
    for variant in ("original", "optimized"):
        workers.append(_run_worker(args, "preflight", variant))
    original = _load_tensor(output_dir / "measurements" / "preflight_original.pt",
                            schema="chen_lightgen_400_rs_static_transpose_microbatch2.v1")
    optimized = _load_tensor(output_dir / "measurements" / "preflight_optimized.pt",
                             schema="chen_lightgen_400_rs_static_transpose_microbatch2.v1")
    expected_rows = refs["reference_rows"][:prior.MICROBATCH]
    _require(torch.equal(original["rows"], expected_rows)
             and torch.equal(optimized["rows"], expected_rows),
             "production preflight rows differ from the fixed fit100 microbatch")
    comparison = _compare_microbatch(optimized, original)
    _require(comparison["status"] == "NUMERICALLY_EQUIVALENT",
             "original and optimized microbatch2 production results differ beyond qualification tolerances")
    receipt = {"schema": "chen_lightgen_400_rs_static_transpose_preflight.v1",
               "task_id": TASK_ID, "status": "PREFLIGHT_PASSED",
               "explicit_inputs": refs["paths"], "rows": expected_rows.tolist(),
               "labels": list(range(10)), "checkpoint_step": 30_000,
               "historical_endpoint_step": 500, "precision": PRECISION,
               "logical_batch": prior.LOGICAL_BATCH, "microbatch": prior.MICROBATCH,
               "checkpoint_propagation": True, "comparison": comparison,
               "worker_receipts": workers, "optimizer_calls": 0, "adam_calls": 0,
               "optimizer_updates": 0, "generation_samples": 0,
               "evaluator_forward_calls": 0, "parameters_and_buffers_unchanged": True}
    _write_json(output_dir / "preflight_receipt.json", receipt, new=True)
    _progress(output_dir, "preflight_complete", status=receipt["status"])
    return receipt


def _validate_preflight(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output_dir / "preflight_receipt.json"
    _require(path.is_file(), "run --preflight-only with these inputs before measurement")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _require(receipt.get("schema") == "chen_lightgen_400_rs_static_transpose_preflight.v1"
             and receipt.get("task_id") == TASK_ID and receipt.get("status") == "PREFLIGHT_PASSED"
             and receipt.get("explicit_inputs") == _explicit_paths(args)
             and receipt.get("optimizer_updates") == receipt.get("adam_calls") == 0,
             "preflight receipt is incomplete or bound to different explicit inputs")
    original = _load_tensor(output_dir / "measurements" / "preflight_original.pt",
                            schema="chen_lightgen_400_rs_static_transpose_microbatch2.v1")
    optimized = _load_tensor(output_dir / "measurements" / "preflight_optimized.pt",
                             schema="chen_lightgen_400_rs_static_transpose_microbatch2.v1")
    comparison = _compare_microbatch(optimized, original)
    _require(comparison["status"] == "NUMERICALLY_EQUIVALENT",
             "saved preflight tensor values no longer satisfy the fixed thresholds")
    return receipt, {"original": original, "optimized": optimized}


def _load_run_case(output_dir: Path, case: str, expected_rows: torch.Tensor) -> dict[str, Any]:
    payload = _load_tensor(output_dir / "measurements" / f"{case}.pt",
                           schema="chen_lightgen_400_rs_static_transpose_batch10.v1", case=case)
    _require(torch.equal(payload["rows"].long().cpu(), expected_rows.long().cpu())
             and payload.get("variant") == case.split("_")[0]
             and payload.get("operator_call_counts", {}).get("rs_forward_calls") == prior.BASE_RS_FORWARD_CALLS_PER_BATCH10
             and payload.get("operator_call_counts", {}).get("rs_adjoint_calls") == prior.BASE_RS_FORWARD_CALLS_PER_BATCH10,
             f"saved batch10 {case} has the wrong population, variant, or RS call counts")
    prior._validate_call_counts(payload["operator_call_counts"], checkpoint=True)
    counts = payload["operator_call_counts"]
    materializations = counts["pair_spectrum_materializations"]
    if case.startswith("original"):
        _require(counts.get("legacy_device_pair_predicates") == materializations
                 and counts.get("pair_blocks_with_transpose", 0)
                 + counts.get("pair_blocks_without_transpose", 0) == 0,
                 "original path predicate instrumentation is incomplete")
    else:
        _require(counts.get("legacy_device_pair_predicates", 0) == 0
                 and counts.get("pair_blocks_with_transpose", 0)
                 + counts.get("pair_blocks_without_transpose", 0) == materializations,
                 "optimized path transpose schedule instrumentation is incomplete")
    return payload


def _measure(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    _require(not (output_dir / "final_report.json").exists(),
             "cost qualification is already complete; refusing to overwrite its final report")
    refs = _load_references(args)
    _preflight_receipt, _preflight_tensors = _validate_preflight(args, output_dir)
    rows = refs["reference_rows"]
    cases: dict[str, dict[str, Any]] = {}
    first_original: dict[str, Any] | None = None
    first_comparison: dict[str, Any] | None = None
    if args.resume:
        seen_gap = False
        for name in MEASUREMENT_ORDER:
            path = output_dir / "measurements" / f"{name}.pt"
            if not path.exists():
                seen_gap = True
                continue
            _require(not seen_gap, "resume evidence is not a complete prefix of the fixed ABBA order")
            cases[name] = _load_run_case(output_dir, name, rows)
            summary_path = output_dir / f"{name}_summary.json"
            _require(summary_path.is_file(), f"resume summary is missing for {name}")
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _require(saved_summary.get("task_id") == TASK_ID and saved_summary.get("case") == name
                     and saved_summary.get("elapsed_seconds") == cases[name].get("elapsed_seconds"),
                     f"resume summary does not match its complete tensor for {name}")
        if "original_1" in cases:
            first_original = cases["original_1"]
            first_comparison = prior._compare_case(first_original, refs["reference_case"])
            _require(first_comparison["status"] == "NUMERICALLY_EQUIVALENT",
                     "saved original_1 does not replay the previous batch10 reference")
    else:
        _require(not any((output_dir / "measurements" / f"{name}.pt").exists()
                         for name in MEASUREMENT_ORDER),
                 "measurement files exist; pass --resume to revalidate a complete saved prefix")

    stopped_for_numeric_failure = False
    failure_status: str | None = None
    failure_detail: str | None = None
    worker_receipts: list[dict[str, Any]] = []
    for name in MEASUREMENT_ORDER[len(cases):]:
        variant = name.split("_")[0]
        _progress(output_dir, f"measure_{name}", completed_cases=list(cases))
        worker_receipts.append(_run_worker(args, "measure", variant, name))
        result = _load_run_case(output_dir, name, rows)
        if name == "original_1":
            first_comparison = prior._compare_case(result, refs["reference_case"])
            if first_comparison["status"] != "NUMERICALLY_EQUIVALENT":
                cases[name] = result
                stopped_for_numeric_failure = True
                failure_status = "INVALID_HARNESS"
                failure_detail = "original_1 did not replay the saved previous batch10 result"
                break
            first_original = result
            comparison = first_comparison
        else:
            _require(first_original is not None, "ABBA measurement lacks its first original reference")
            comparison = prior._compare_case(result, first_original)
            if variant == "optimized" and comparison["status"] != "NUMERICALLY_EQUIVALENT":
                stopped_for_numeric_failure = True
                failure_status = "VALID_COMPLETED_NUMERICAL_MISMATCH"
                failure_detail = f"{name} exceeded the fixed numerical comparison thresholds"
            elif variant == "original" and comparison["status"] != "NUMERICALLY_EQUIVALENT":
                stopped_for_numeric_failure = True
                failure_status = "INVALID_HARNESS"
                failure_detail = f"{name} did not reproduce original_1 within fixed thresholds"
        cases[name] = result
        summary = {"schema": "chen_lightgen_400_rs_static_transpose_case.v1",
                   "task_id": TASK_ID, "case": name, "variant": variant,
                   "elapsed_seconds": result["elapsed_seconds"],
                   "peak_reserved_vram_bytes": result["peak_reserved_vram_bytes"],
                   "loss": result["loss"], "comparison_to_original_1": comparison,
                   "cache_build_seconds": result["cache_build_seconds"],
                   "transpose_schedule_build_seconds": result["transpose_schedule_build_seconds"],
                   "transpose_schedule_build_count": result["transpose_schedule_build_count"],
                   "transpose_schedule_cache_hits": result["transpose_schedule_cache_hits"],
                   "cache_info": result["cache_info"],
                   "operator_call_counts": result["operator_call_counts"],
                   "stage_timing": result["stage_timing"], "worker": worker_receipts[-1],
                   "optimizer_calls": 0, "optimizer_updates": 0}
        _write_json(output_dir / f"{name}_summary.json", summary, new=True)
        if stopped_for_numeric_failure:
            break

    profile_records = []
    profile_inputs = {}
    if not stopped_for_numeric_failure:
        for variant in ("original", "optimized"):
            profile_path = output_dir / "measurements" / f"profile_{variant}.pt"
            if profile_path.exists():
                _require(args.resume, "profile evidence exists; pass --resume to validate and reuse it")
            else:
                worker_receipts.append(_run_worker(args, "profile", variant))
            profile = _load_tensor(output_dir / "measurements" / f"profile_{variant}.pt",
                                   schema="chen_lightgen_400_rs_static_transpose_profile.v1")
            _require(torch.equal(profile["rows"], rows[:prior.MICROBATCH]),
                     f"{variant} profiler used rows outside the first fixed microbatch")
            profile_inputs[variant] = profile
            profile_records.extend(profile["profile_summaries"])
        _require(prior._compare_case(
            {"loss": profile_inputs["optimized"]["loss"],
             "gradients": profile_inputs["optimized"]["gradients"],
             "w_per_pixel": profile_inputs["optimized"]["w_per_pixel"]},
            {"loss": profile_inputs["original"]["loss"],
             "gradients": profile_inputs["original"]["gradients"],
             "w_per_pixel": profile_inputs["original"]["w_per_pixel"]},
        )["status"] == "NUMERICALLY_EQUIVALENT",
                 "profiled original and optimized microbatch2 values differ")
        for variant, profile in profile_inputs.items():
            _require(len(profile["profile_summaries"]) == 2
                     and {item.get("direction") for item in profile["profile_summaries"]}
                     == {"forward", "adjoint"},
                     f"{variant} profiler did not save exactly one forward and adjoint record")
        atomic_write_json(output_dir / "operator_profile.json", _safe_json({
            "schema": "chen_lightgen_400_rs_static_transpose_profile.v1",
            "scope": "one additional production microbatch2 per implementation; excluded from throughput",
            "original": profile_inputs["original"]["profile_summaries"],
            "optimized": profile_inputs["optimized"]["profile_summaries"],
            "operator_call_counts": {
                variant: profile_inputs[variant]["operator_call_counts"]
                for variant in ("original", "optimized")},
            "nested_device_totals_must_not_be_summed": True,
        }))
    else:
        atomic_write_json(output_dir / "operator_profile.json", {
            "schema": "chen_lightgen_400_rs_static_transpose_profile.v1",
            "status": "SKIPPED_AFTER_NUMERICAL_MISMATCH",
            "reason": failure_detail,
            "included_in_throughput": False,
        })

    original_cases = [cases[name] for name in ("original_1", "original_2") if name in cases]
    optimized_cases = [cases[name] for name in ("optimized_1", "optimized_2") if name in cases]
    original_times = [float(value["elapsed_seconds"]) for value in original_cases]
    optimized_times = [float(value["elapsed_seconds"]) for value in optimized_cases]
    all_numeric = (not stopped_for_numeric_failure and len(cases) == 4
                   and all(prior._compare_case(cases[name], first_original)["status"]
                           == "NUMERICALLY_EQUIVALENT" for name in MEASUREMENT_ORDER[1:]))
    speed_supported = (all_numeric and len(original_times) == len(optimized_times) == 2
                       and max(optimized_times) < min(original_times))
    if failure_status is not None:
        status = failure_status
    elif not all_numeric:
        status = "MEASUREMENT_INCOMPLETE"
    elif speed_supported:
        status = "VALID_COMPLETED_NUMERICALLY_EQUIVALENT_SPEED_SUPPORTED"
    else:
        status = "VALID_COMPLETED_NUMERICALLY_EQUIVALENT_SPEED_UNCONFIRMED"
    report = {
        "schema": "chen_lightgen_400_rs_static_transpose_report.v1",
        "task_id": TASK_ID, "status": status,
        "measurement_order": [name for name in MEASUREMENT_ORDER if name in cases],
        "input_identity": {"loaded_checkpoint_step": 30_000, "historical_endpoint_step": 500,
                           "rows": rows.tolist(), "labels": list(range(10)),
                           "data_role": "fixed_fit100_first_row_per_class",
                           "precision": PRECISION, "order": ORDER, "pair_block": PAIR_BLOCK,
                           "logical_batch": prior.LOGICAL_BATCH, "microbatch": prior.MICROBATCH,
                           "checkpoint_propagation": True,
                           "beta": float(refs["previous_evidence"]["input_identity"]["beta"]),
                           "eta": float(refs["previous_evidence"]["input_identity"]["eta"])},
        "explicit_inputs": refs["paths"], "cases": {
            name: {"elapsed_seconds": case["elapsed_seconds"], "loss": case["loss"],
                  "cache_build_seconds": case["cache_build_seconds"],
                  "transpose_schedule_build_seconds": case["transpose_schedule_build_seconds"],
                  "transpose_schedule_build_count": case["transpose_schedule_build_count"],
                  "transpose_schedule_cache_hits": case["transpose_schedule_cache_hits"],
                  "peak_reserved_vram_bytes": case["peak_reserved_vram_bytes"],
                  "operator_call_counts": case["operator_call_counts"],
                  "stage_timing": case["stage_timing"],
                  "comparison_to_original_1": (
                      first_comparison if name == "original_1" else
                      prior._compare_case(case, first_original) if first_original is not None else None)}
            for name, case in cases.items()},
        "original_range_seconds": [min(original_times), max(original_times)] if original_times else None,
        "optimized_range_seconds": [min(optimized_times), max(optimized_times)] if optimized_times else None,
        "original_mean_seconds_per_step": sum(original_times) / len(original_times) if original_times else None,
        "optimized_mean_seconds_per_step": sum(optimized_times) / len(optimized_times) if optimized_times else None,
        "speed_supported_by_fixed_rule": speed_supported,
        "failure_detail": failure_detail,
        "speed_rule": "both optimized batch10 wall times are below both original wall times",
        "compute_only_estimates": {
            "step_cost_unit": "seconds per logical batch10 update",
            "scope": "linear compute-only estimate; excludes checkpointing, evaluation, restart, and data management",
        },
        "profile_summaries": profile_records,
        "profile_device_timing_available": bool(profile_records)
        and all(item.get("status") == "DEVICE_TIMING_CAPTURED" for item in profile_records),
        "preflight_comparison": json.loads((output_dir / "preflight_receipt.json").read_text(
            encoding="utf-8"))["comparison"],
        "workers": worker_receipts,
        "optimizer_calls": 0, "adam_calls": 0, "optimizer_updates": 0,
        "generation_samples": 0, "evaluator_forward_calls": 0,
        "execution_elapsed_seconds": time.perf_counter() - args.started_at,
        "preflight_elapsed_seconds": float(
            json.loads((output_dir / "preflight_receipt.json").read_text(encoding="utf-8"))
            .get("elapsed_seconds", 0.0)),
    }
    report["elapsed_seconds"] = report["preflight_elapsed_seconds"] + report["execution_elapsed_seconds"]
    estimates = {}
    for kind, mean in (("original", report["original_mean_seconds_per_step"]),
                       ("optimized", report["optimized_mean_seconds_per_step"])):
        if mean is None:
            continue
        estimates[kind] = {
            "500_steps": {"seconds": mean * 500, "hours": mean * 500 / 3600},
            "30000_steps": {"seconds": mean * 30_000, "hours": mean * 30_000 / 3600,
                            "days": mean * 30_000 / 86400},
        }
    report["compute_only_estimates"] = estimates
    _write_deliverables(output_dir, report, cases)
    receipt = {"schema": "chen_lightgen_400_rs_static_transpose_validation.v1",
               "task_id": TASK_ID, "status": report["status"],
               "case_count": len(cases), "measurement_order": report["measurement_order"],
               "optimizer_calls": 0, "adam_calls": 0, "optimizer_updates": 0,
               "generation_samples": 0, "evaluator_forward_calls": 0,
               "all_case_tensors_read_back": True,
               "profile_scope_is_limited_and_excluded": not stopped_for_numeric_failure,
               "profile_device_timing_available": report["profile_device_timing_available"],
               "parameter_and_buffer_invariance_verified_in_each_worker": True}
    _write_json(output_dir / "validation_receipt.json", receipt)
    _write_json(output_dir / "final_report.json", report, new=True)
    _require(sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
             <= OUTPUT_LIMIT_BYTES, "formal output exceeded the 512 MiB budget")
    _progress(output_dir, "complete", status=status, completed_cases=report["measurement_order"])
    return report


def _write_deliverables(output_dir: Path, report: dict[str, Any], cases: dict[str, dict[str, Any]]) -> None:
    csv_path = output_dir / "batch10_cost.csv"
    fields = ("case", "variant", "elapsed_seconds", "loss", "loss_relative_error",
              "max_gradient_relative_error", "w_per_pixel_relative_error",
              "peak_reserved_vram_bytes", "cache_build_seconds", "forward_calls",
              "adjoint_calls", "cache_hits", "cache_misses", "pair_spectrum_materializations",
              "legacy_device_pair_predicates", "pair_blocks_with_transpose",
              "pair_blocks_without_transpose")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, result in cases.items():
            counts = result["operator_call_counts"]
            comparison = (report["cases"].get(case, {}).get("comparison_to_original_1") or {})
            writer.writerow({"case": case, "variant": result["variant"],
                             "elapsed_seconds": result["elapsed_seconds"], "loss": result["loss"],
                             "loss_relative_error": comparison.get("loss_relative_error"),
                             "max_gradient_relative_error": comparison.get("max_gradient_relative_error"),
                             "w_per_pixel_relative_error": comparison.get("w_per_pixel_relative_error"),
                             "peak_reserved_vram_bytes": result["peak_reserved_vram_bytes"],
                             "cache_build_seconds": result["cache_build_seconds"],
                             "forward_calls": counts.get("rs_forward_calls", 0),
                             "adjoint_calls": counts.get("rs_adjoint_calls", 0),
                             "cache_hits": counts.get("kernel_cache_hits", 0),
                             "cache_misses": counts.get("kernel_cache_misses", 0),
                             "pair_spectrum_materializations": counts.get("pair_spectrum_materializations", 0),
                             "legacy_device_pair_predicates": counts.get("legacy_device_pair_predicates", 0),
                             "pair_blocks_with_transpose": counts.get("pair_blocks_with_transpose", 0),
                             "pair_blocks_without_transpose": counts.get("pair_blocks_without_transpose", 0)})
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        _require(sum(1 for _ in csv.DictReader(handle)) == len(cases),
                 "batch10 CSV read-back row count differs from saved cases")
    _write_json(output_dir / "machine_report.json", report)
    rows = []
    for name, item in report["cases"].items():
        comparison = item.get("comparison_to_original_1") or {}
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            name, f"{item['elapsed_seconds']:.3f}",
            f"{comparison.get('loss_relative_error', math.nan):.3e}",
            f"{comparison.get('max_gradient_relative_error', math.nan):.3e}",
            f"{comparison.get('w_per_pixel_relative_error', math.nan):.3e}")) + "</tr>")
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>LightGen RS转置调度验收</title>"
            "<style>body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 18px}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:7px;text-align:left}"
            "th{background:#f1f3f5}</style><body><h1>LightGen RS静态转置调度零更新验收</h1>"
            f"<p>状态：{html.escape(report['status'])}；Adam、更新、采样及E_D前向均为0。</p>"
            "<p>测量值为秒／batch10。Profiler仅用于一个microbatch2前向及伴随，未计入吞吐。</p>"
            "<table><thead><tr><th>运行</th><th>秒／batch10</th><th>损失差</th>"
            "<th>梯度最大差</th><th>W/pixel差</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table><p><a href='final_report.json'>机器报告</a> · "
            "<a href='batch10_cost.csv'>成本CSV</a> · <a href='operator_profile.json'>算子追踪</a> · "
            "<a href='validation_receipt.json'>验证收据</a></p></body></html>")
    (output_dir / "comparison.html").write_text(page, encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-checkpoint", "source-report", "source-start-state",
                 "source-endpoint-checkpoint", "fixed-population", "config", "data", "evaluator",
                 "qualification-report", "structure-coefficient", "qualification-dir",
                 "cost-reference-dir", "reference-rs-source", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-beta", type=float, required=True)
    parser.add_argument("--expected-eta", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-mode", choices=("preflight", "measure", "profile"))
    parser.add_argument("--variant", choices=("original", "optimized"))
    parser.add_argument("--case")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    args.started_at = time.perf_counter()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if args.worker_mode:
            _require(args.variant is not None, "worker mode requires an explicit implementation variant")
            result = _worker(args, args.worker_mode, args.variant, args.case)
        elif args.preflight_only:
            _require(not args.resume, "--resume is only valid for full measurement")
            _load_references(args)
            result = _preflight(args, output_dir)
            result["elapsed_seconds"] = time.perf_counter() - started
            _write_json(output_dir / "preflight_receipt.json", result)
        else:
            result = _measure(args, output_dir)
        result["task_elapsed_seconds"] = time.perf_counter() - started
        print(json.dumps(_safe_json(result), ensure_ascii=False, allow_nan=False))
        return result
    except torch.cuda.OutOfMemoryError as exc:
        _progress(output_dir, "resource_blocked", status="PREFLIGHT_BLOCKED_RESOURCE",
                  failure_type=type(exc).__name__, optimizer_updates=0)
        raise
    except HarnessError as exc:
        _progress(output_dir, "invalid_harness", status="INVALID_HARNESS",
                  failure=str(exc), optimizer_updates=0)
        raise


if __name__ == "__main__":
    main()
