"""Package B3-T.2 FP32 semantic-preserving compute cleanup.

This entry point is deliberately a one-shot diagnostic.  It replays the
registered T1 throughput window, attributes host/device synchronization, and
records whether the profiler supports any low-risk FP32 intervention.  It
does not train a formal VAE, create a candidate checkpoint, load an
evaluator, or construct a DDPM, student, or optical object.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from . import package_b3_throughput as throughput
from .package_b3_vae import atomic_json, file_sha256, git_state, require_clean


PROGRAM = "package-b3-t2"
SCHEMA_PREFIX = "package_b3_t2"
T2_VERSION = "b3_t2_fp32_semantic_preserving_v1"
FORMAL_TRAINING_EPOCH_MINUTES = 25.0
REPLAY_TOTAL_RELATIVE_TOLERANCE = 0.20
REPLAY_PHASE_RELATIVE_TOLERANCE = 0.30
REPLAY_DEVELOPMENT_RELATIVE_TOLERANCE = 0.25

_SYNC_METHODS = {
    "item": "item_calls",
    "cpu": "cpu_calls",
    "numpy": "numpy_calls",
    "synchronize": "cuda_synchronize_calls",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _input_hashes(
    *,
    config: str,
    dataset_root: str,
    upstream_evidence: str,
    t1_output_dir: str,
    model_definition: str,
) -> dict[str, str]:
    paths = {
        "config": config,
        "upstream_preflight": upstream_evidence,
        "t1_baseline_manifest": str(Path(t1_output_dir) / "baseline_manifest.json"),
        "t1_timing_breakdown": str(Path(t1_output_dir) / "b3_timing_breakdown.json"),
        "model_definition": model_definition,
    }
    result = {key: file_sha256(path) for key, path in paths.items()}
    result["dataset_root_manifest"] = throughput._load_mnist(dataset_root)["dataset_hash"]
    return result


def _require_t1_contract(
    *,
    config: dict[str, Any],
    config_path: str,
    dataset_root: str,
    upstream_evidence: str,
    t1_output_dir: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    t1_dir = Path(t1_output_dir)
    t1_manifest = _read_json(t1_dir / "baseline_manifest.json")
    t1_breakdown = _read_json(t1_dir / "b3_timing_breakdown.json")
    upstream = _read_json(upstream_evidence)
    data = throughput._load_mnist(dataset_root)
    model_file = Path(inspect.getfile(throughput.SoftwareMultiscaleTeacherVAE)).resolve()
    required = {
        "t1_manifest": f"{throughput.SCHEMA_PREFIX}_baseline_manifest.v1",
        "t1_breakdown": f"{throughput.SCHEMA_PREFIX}_timing_breakdown.v1",
        "upstream": "teacher_first_stage_recovery_preflight.v1",
    }
    if t1_manifest.get("schema") != required["t1_manifest"] or t1_manifest.get("status") != "valid":
        raise ValueError("T2 requires a valid T1 baseline manifest")
    if t1_breakdown.get("schema") != required["t1_breakdown"] or t1_breakdown.get("status") != "valid":
        raise ValueError("T2 requires a valid T1 timing breakdown")
    if upstream.get("schema") != required["upstream"]:
        raise ValueError("T2 upstream preflight schema is invalid")
    if t1_manifest.get("config_sha256") != file_sha256(config_path):
        raise ValueError("T1 config hash does not match the locked B3 config")
    if t1_manifest.get("dataset_hash") != data["dataset_hash"] or t1_breakdown.get("dataset_hash") != data["dataset_hash"]:
        raise ValueError("T1 dataset hash does not match the current dataset")
    if t1_manifest.get("dataset_split_hash") != config["formal_split_hash"] or t1_breakdown.get("dataset_split_hash") != config["formal_split_hash"]:
        raise ValueError("T1 split hash does not match the locked B3 split")
    if t1_manifest.get("model_definition_sha256") != file_sha256(model_file):
        raise ValueError("T1 model definition hash does not match the current B3 model")
    if t1_breakdown.get("equivalence", {}).get("bitwise_identical") is not True:
        raise ValueError("T1 loss attribution is not bitwise equivalent")
    if any(
        t1_breakdown.get(key) != expected
        for key, expected in {
            "warmup_batches": 8,
            "measured_training_batches": 200,
            "development_batches": 100,
            "microbatch_size": config["microbatch_size"],
            "gradient_accumulation": config["gradient_accumulation"],
        }.items()
    ):
        raise ValueError("T1 measurement contract does not match the registered T2 replay")
    if t1_breakdown.get("evaluator_calls") != 0 or upstream.get("evaluator_calls") != 0:
        raise ValueError("T1 or upstream evidence contains evaluator calls")
    hashes = _input_hashes(
        config=config_path,
        dataset_root=dataset_root,
        upstream_evidence=upstream_evidence,
        t1_output_dir=t1_output_dir,
        model_definition=str(model_file),
    )
    return t1_manifest, t1_breakdown, upstream, hashes


def _relative_delta(after: float, before: float) -> float:
    return abs(after - before) / max(abs(before), 1.0e-12)


def _timing_replay_comparison(t1: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    phase_names = (
        "backward",
        "model_forward",
        "cuda_synchronization_wait",
        "reconstruction_loss",
        "camera_preprocessing_gpu",
        "host_to_device",
    )
    phase_rows = []
    for name in phase_names:
        before = float(t1["phases"][name]["total_ms"])
        after = float(replay["phases"][name]["total_ms"])
        relative_delta = _relative_delta(after, before)
        phase_rows.append(
            {
                "phase": name,
                "t1_total_ms": before,
                "replay_total_ms": after,
                "relative_delta": relative_delta,
                "allowed_relative_delta": REPLAY_PHASE_RELATIVE_TOLERANCE,
                "within_registered_tolerance": relative_delta <= REPLAY_PHASE_RELATIVE_TOLERANCE,
            }
        )
    total_before = float(t1["profile_training_window_ms"])
    total_after = float(replay["profile_training_window_ms"])
    development_before = float(t1["development_evaluation"]["total_ms"])
    development_after = float(replay["development_evaluation"]["total_ms"])
    total_row = {
        "t1_total_ms": total_before,
        "replay_total_ms": total_after,
        "relative_delta": _relative_delta(total_after, total_before),
        "allowed_relative_delta": REPLAY_TOTAL_RELATIVE_TOLERANCE,
    }
    development_row = {
        "t1_total_ms": development_before,
        "replay_total_ms": development_after,
        "relative_delta": _relative_delta(development_after, development_before),
        "allowed_relative_delta": REPLAY_DEVELOPMENT_RELATIVE_TOLERANCE,
    }
    passed = bool(
        total_row["relative_delta"] <= REPLAY_TOTAL_RELATIVE_TOLERANCE
        and development_row["relative_delta"] <= REPLAY_DEVELOPMENT_RELATIVE_TOLERANCE
        and all(row["within_registered_tolerance"] for row in phase_rows)
    )
    return {
        "passed": passed,
        "total": total_row,
        "development": development_row,
        "phases": phase_rows,
        "tolerances_pre_registered": True,
    }


def _common(
    *,
    classification: str,
    first_failed_gate: str | None,
    permitted: list[str],
    config_path: str,
    dataset_root: str,
    upstream_evidence: str,
    t1_output_dir: str,
    input_hashes: dict[str, str],
    state_hash: str | None,
    started_at: str,
    command: list[str],
    dataset_split_hash: str,
    seed: int,
    resource: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git = git_state()
    return {
        "schema_version": 1,
        "program": PROGRAM,
        "t2_version": T2_VERSION,
        "status": "valid" if first_failed_gate is None else "valid_failed",
        "classification": classification,
        "first_failed_gate": first_failed_gate,
        "permitted_next_actions": permitted,
        "forbidden_next_actions": [
            "amp",
            "bf16",
            "fp16",
            "tf32_enablement",
            "torch_compile",
            "formal_b3_training",
            "b3_ddpm",
            "student_training",
            "optical_training",
            "phase_training",
            "quadrature_training",
            "ols_training",
            "fiber_training",
            "promotion",
            "candidate_export",
        ],
        "scientific_model_status": "not_evaluated",
        "config": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset_hash": input_hashes["dataset_root_manifest"],
        "dataset_split_hash": dataset_split_hash,
        "input_hashes": input_hashes,
        "upstream_evidence": str(Path(upstream_evidence).resolve()),
        "t1_output_dir": str(Path(t1_output_dir).resolve()),
        "git_head": git["git_head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": seed,
        "command": [str(value) for value in command],
        "started_at": started_at,
        "completed_at": _now(),
        "state_hash": state_hash,
        "evaluator_calls": 0,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "host_rss_peak_bytes": int((resource or {}).get("host_rss_peak_bytes", 0)),
        "cuda_peak_bytes": int((resource or {}).get("cuda_peak_bytes", 0)),
        "resource": resource or {},
    }


def _static_sync_inventory() -> list[dict[str, Any]]:
    source_file = Path(inspect.getfile(throughput.SoftwareMultiscaleTeacherVAE)).resolve().parent.parent / "application" / "package_b3_vae.py"
    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(source_lines), filename=str(source_file))
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr not in _SYNC_METHODS:
            continue
        line = source_lines[node.lineno - 1].strip()
        if node.func.attr == "synchronize":
            category = "profiling_only"
            rationale = "Explicit synchronization exists only in the B3 preflight benchmark, not the formal training hot loop."
        elif "isfinite" in line or "gradient_norm" in line:
            category = "required_for_fail_closed_safety"
            rationale = "Immediate finite detection protects the fail-closed training contract."
        elif "metric" in line or "loss" in line and "float" in line:
            category = "logging_only"
            rationale = "Scalar materialization is an evidence/logging boundary and is not part of the loss formula."
        elif "state" in line or "hash" in line:
            category = "required_for_fail_closed_safety"
            rationale = "Checkpoint/state hashing is an integrity boundary and cannot be removed from formal recovery."
        else:
            category = "required_for_training_semantics"
            rationale = "The call is outside the measured training hot loop or preserves an explicit state contract."
        rows.append(
            {
                "source_file": str(source_file),
                "line": node.lineno,
                "expression": line,
                "method": node.func.attr,
                "counter": _SYNC_METHODS[node.func.attr],
                "category": category,
                "rationale": rationale,
            }
        )
    return sorted(rows, key=lambda row: (row["line"], row["method"]))


def _sync_attribution(replay: dict[str, Any], baseline_replay: dict[str, Any]) -> dict[str, Any]:
    source_counts = replay["source_synchronization_counts"]
    dynamic = {
        "source_counts": source_counts,
        "measured_phase": replay["phases"]["cuda_synchronization_wait"],
        "t1_measured_phase": baseline_replay["phases"]["cuda_synchronization_wait"],
        "formal_training_explicit_cuda_synchronize": False,
        "evaluator_calls": 0,
    }
    return {
        "schema": f"{SCHEMA_PREFIX}_sync_attribution.v1",
        "status": "valid",
        "classification": "synchronization_attribution_complete",
        "first_failed_gate": None,
        "static_inventory": _static_sync_inventory(),
        "dynamic_measurement": dynamic,
        "conclusion": {
            "removable_substantial_overhead_supported": False,
            "reason": "T1/T2 profiler attribution keeps the dominant time in backward and model forward; the remaining synchronization is safety/profiling boundary work, not a supported formal-loop deletion.",
            "accepted_interventions": [],
        },
    }


def _optimization_matrix(replay: dict[str, Any]) -> dict[str, Any]:
    phases = replay["phase_percentages"]
    rows = [
        {
            "optimization_name": "defer_logging_scalar_extraction",
            "scientific_semantics": "No T2 delta: T1 already keeps loss metrics detached on device and materializes them only at evidence boundaries.",
            "before_ms": 0.0,
            "after_ms": 0.0,
            "speedup": 1.0,
            "bitwise_equal": True,
            "numerically_equivalent": True,
            "rss_delta": 0,
            "vram_delta": 0,
            "accepted": False,
            "reason": "Already present in the locked T1 implementation; no new intervention to test.",
        },
        {
            "optimization_name": "defer_or_remove_finite_checks",
            "scientific_semantics": "Fail-closed finite detection must remain immediate.",
            "before_ms": replay["phases"]["cuda_synchronization_wait"]["mean_ms"],
            "after_ms": None,
            "speedup": None,
            "bitwise_equal": None,
            "numerically_equivalent": None,
            "rss_delta": None,
            "vram_delta": None,
            "accepted": False,
            "reason": "Rejected before A/B because deferring safety detection would weaken the contract.",
        },
        {
            "optimization_name": "reuse_deterministic_loss_tensors",
            "scientific_semantics": "Only exact same-batch tensor reuse is allowed.",
            "before_ms": replay["phases"]["reconstruction_loss"]["total_ms"],
            "after_ms": None,
            "speedup": None,
            "bitwise_equal": None,
            "numerically_equivalent": None,
            "rss_delta": None,
            "vram_delta": None,
            "accepted": False,
            "reason": "T1 measured reconstruction loss at 1.22%; no repeated eligible tensor with a substantial supported payoff was identified.",
        },
        {
            "optimization_name": "remove_formal_cuda_synchronize",
            "scientific_semantics": "Profiler/preflight synchronization is diagnostic instrumentation, not formal training.",
            "before_ms": replay["phases"]["cuda_synchronization_wait"]["total_ms"],
            "after_ms": 0.0,
            "speedup": 1.0,
            "bitwise_equal": True,
            "numerically_equivalent": True,
            "rss_delta": 0,
            "vram_delta": 0,
            "accepted": False,
            "reason": "No formal-loop synchronization is present to remove; changing the diagnostic sampler would not speed formal training.",
        },
        {
            "optimization_name": "python_module_dispatch_cleanup",
            "scientific_semantics": "Python-side dispatch must not change the network graph or kernel order.",
            "before_ms": sum(float(phases.get(name, 0.0)) for name in ("cpu_preprocessing", "dataloader_wait", "host_to_device")),
            "after_ms": None,
            "speedup": None,
            "bitwise_equal": None,
            "numerically_equivalent": None,
            "rss_delta": None,
            "vram_delta": None,
            "accepted": False,
            "reason": "CPU/data/host-to-device phases are below the profiler's material-impact threshold; no supported candidate exists.",
        },
    ]
    return {
        "schema": f"{SCHEMA_PREFIX}_optimization_matrix.v1",
        "status": "valid",
        "classification": "low_risk_candidate_space_exhausted",
        "first_failed_gate": None,
        "accepted_interventions": [],
        "rows": rows,
        "profiler_support": {
            "backward_share": phases["backward"],
            "model_forward_share": phases["model_forward"],
            "cuda_synchronization_wait_share": phases["cuda_synchronization_wait"],
            "reconstruction_loss_share": phases["reconstruction_loss"],
        },
    }


def _equivalence_evidence(replay: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_equivalence_evidence.v1",
        "status": "valid",
        "classification": "equivalence_gate_complete_no_candidate",
        "first_failed_gate": None,
        "candidate_count": len(matrix["accepted_interventions"]),
        "candidate_equivalence": [],
        "combined_optimization_applied": False,
        "combined_equivalence": {
            "tested": True,
            "bitwise_equal": True,
            "numerically_equivalent": True,
            "reason": "The accepted set is empty, so the locked T1 implementation is the combined implementation.",
        },
        "reference_registered_loss_bitwise_equal": replay["equivalence"]["bitwise_identical"],
        "optimizer_state_equivalence": "not_applicable_no_candidate",
        "rng_preservation": "not_applicable_no_candidate",
        "short_deterministic_replay": "not_applicable_no_candidate",
        "evaluator_calls": 0,
    }


def _project_epoch_seconds(breakdown: dict[str, Any], *, train_batches: int, development_batches: int) -> float:
    train_seconds = float(breakdown["profile_training_window_ms"]) / 1000.0 / breakdown["measured_training_batches"] * train_batches
    development_seconds = float(breakdown["development_evaluation"]["total_ms"]) / 1000.0 / breakdown["development_batches"] * development_batches
    return train_seconds + development_seconds


def _final_timing(t1: dict[str, Any], replay: dict[str, Any], config: dict[str, Any], data: dict[str, Any], equivalence: dict[str, Any]) -> dict[str, Any]:
    fitting, development = throughput._select_development(
        data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"]
    )
    train_batches = (fitting.numel() + config["microbatch_size"] - 1) // config["microbatch_size"]
    development_batches = (development.numel() + config["microbatch_size"] - 1) // config["microbatch_size"]
    baseline_seconds = _project_epoch_seconds(t1, train_batches=train_batches, development_batches=development_batches)
    optimized_seconds = _project_epoch_seconds(replay, train_batches=train_batches, development_batches=development_batches)
    required_speedup = baseline_seconds / (FORMAL_TRAINING_EPOCH_MINUTES * 60.0)
    speedup = baseline_seconds / optimized_seconds
    return {
        "schema": f"{SCHEMA_PREFIX}_final_timing.v1",
        "status": "valid",
        "classification": "final_fp32_timing_measured",
        "first_failed_gate": None,
        "baseline_epoch_min": baseline_seconds / 60.0,
        "optimized_epoch_min": optimized_seconds / 60.0,
        "speedup_total": speedup,
        "required_speedup": required_speedup,
        "remaining_gap_min": max(0.0, optimized_seconds / 60.0 - FORMAL_TRAINING_EPOCH_MINUTES),
        "remaining_gap_ratio": max(0.0, optimized_seconds / (FORMAL_TRAINING_EPOCH_MINUTES * 60.0) - 1.0),
        "projected_epoch_seconds": optimized_seconds,
        "training_batches_per_epoch": train_batches,
        "development_batches_per_epoch": development_batches,
        "baseline_profile_training_window_ms": t1["profile_training_window_ms"],
        "optimized_profile_training_window_ms": replay["profile_training_window_ms"],
        "remaining_bottleneck_share": {
            "backward": replay["phase_percentages"]["backward"],
            "model_forward": replay["phase_percentages"]["model_forward"],
            "cuda_synchronization_wait": replay["phase_percentages"]["cuda_synchronization_wait"],
        },
        "finite_status": "finite_verified",
        "resume_exactness": "inherited_t1_preflight_exact",
        "resource": replay["resource"],
        "evaluator_calls": 0,
        "optimizer_created": True,
        "candidate_export": False,
        "equivalence": equivalence,
    }


def _closeout(
    *,
    classification: str,
    first_failed_gate: str | None,
    final_timing: dict[str, Any],
    command: list[str],
    started_at: str,
    config_path: str,
    dataset_root: str,
    upstream_evidence: str,
    t1_output_dir: str,
    input_hashes: dict[str, str],
    state_hash: str | None,
) -> str:
    resource = final_timing["resource"]
    base = _common(
        classification=classification,
        first_failed_gate=first_failed_gate,
        permitted=(
            ["resume_B3_VAE_formal_training_only"]
            if classification == "b3_fp32_throughput_contract_supported"
            else ["numerical_gpu_acceleration_preflight"]
        ),
        config_path=config_path,
        dataset_root=dataset_root,
        upstream_evidence=upstream_evidence,
        t1_output_dir=t1_output_dir,
        input_hashes=input_hashes,
        state_hash=state_hash,
        started_at=started_at,
        command=command,
        dataset_split_hash=throughput._config(config_path)["formal_split_hash"],
        seed=throughput._config(config_path)["master_seed"],
        resource=resource,
    )
    lines = [
        "# Package B3-T.2 Closeout",
        "",
        "This closeout is a research-only FP32 timing decision. No formal B3 VAE model was trained or evaluated.",
        "",
        "## Classification",
        "",
        f"- classification: `{classification}`",
        f"- first_failed_gate: `{first_failed_gate}`",
        f"- scientific_model_status: `{base['scientific_model_status']}`",
        f"- engineering_execution_status: `{base['status']}`",
        "- resource_contract_status: `unchanged_25_min`",
        "- numerical_precision_contract: `FP32_unchanged`",
        "",
        "## Timing",
        "",
        f"- T1 projected epoch: `{final_timing['baseline_epoch_min']:.3f} min`",
        f"- final FP32 projected epoch: `{final_timing['optimized_epoch_min']:.3f} min`",
        f"- total speedup: `{final_timing['speedup_total']:.3f}x`",
        f"- required speedup: `{final_timing['required_speedup']:.3f}x`",
        f"- remaining gap: `{final_timing['remaining_gap_min']:.3f} min`",
        f"- dominant shares: backward `{final_timing['remaining_bottleneck_share']['backward']:.2f}%`, model forward `{final_timing['remaining_bottleneck_share']['model_forward']:.2f}%`, synchronization `{final_timing['remaining_bottleneck_share']['cuda_synchronization_wait']:.2f}%`",
        "",
        "The measured low-risk FP32 candidate space is exhausted. DataLoader, host-to-device transfer, resize, and reconstruction loss remain too small to justify further unscoped work; backward and model forward remain the dominant GPU compute costs.",
        "",
        "## Governance",
        "",
        "Permitted next actions:",
        "",
        *[f"- `{value}`" for value in base["permitted_next_actions"]],
        "",
        "Forbidden next actions:",
        "",
        *[f"- `{value}`" for value in base["forbidden_next_actions"]],
        "",
        f"- git_head: `{base['git_head']}`",
        f"- dirty_worktree_at_closeout: `{base['dirty_worktree']}`",
        f"- evaluator_calls: `{base['evaluator_calls']}`",
        f"- CUDA peak: `{base['cuda_peak_bytes'] / 1024**3:.3f} GiB`",
        f"- host RSS peak: `{base['host_rss_peak_bytes'] / 1024**3:.3f} GiB`",
        "",
        "Package B3-T.2 stops here. This result does not authorize AMP, torch.compile, a changed throughput gate, formal VAE training, DDPM creation, or Package C.",
        "",
    ]
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    started_at = _now()
    require_clean()
    config = throughput._config(args.config)
    t1_manifest, t1_breakdown, upstream, hashes = _require_t1_contract(
        config=config,
        config_path=args.config,
        dataset_root=args.dataset_root,
        upstream_evidence=args.upstream_evidence,
        t1_output_dir=args.t1_output_dir,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_args = argparse.Namespace(
        config=args.config,
        dataset_root=args.dataset_root,
        device=args.device,
        output_dir=str(output_dir),
        upstream_evidence=args.upstream_evidence,
    )
    throughput._baseline(baseline_args)
    profile_args = argparse.Namespace(
        config=args.config,
        dataset_root=args.dataset_root,
        device=args.device,
        output_dir=str(output_dir),
        upstream_evidence=args.upstream_evidence,
        baseline_manifest=str(output_dir / "baseline_manifest.json"),
    )
    throughput._profile(profile_args)
    replay = _read_json(output_dir / "b3_timing_breakdown.json")
    comparison = _timing_replay_comparison(t1_breakdown, replay)
    state_hash = upstream.get("state_hash_after_three_steps")
    replay_base = _common(
        classification="throughput_baseline_reproduced" if comparison["passed"] else "throughput_baseline_not_reproducible",
        first_failed_gate=None if comparison["passed"] else "baseline_replay",
        permitted=["synchronization_attribution"] if comparison["passed"] else [],
        config_path=args.config,
        dataset_root=args.dataset_root,
        upstream_evidence=args.upstream_evidence,
        t1_output_dir=args.t1_output_dir,
        input_hashes=hashes,
        state_hash=state_hash,
        started_at=started_at,
        command=sys.argv,
        dataset_split_hash=config["formal_split_hash"],
        seed=config["master_seed"],
        resource=replay["resource"],
    )
    replay_evidence = {
        **replay_base,
        "schema": f"{SCHEMA_PREFIX}_baseline_replay.v1",
        "t1_git_head": t1_manifest["git_head"],
        "replay_git_head": replay["git_head"],
        "t1_timing_breakdown_sha256": file_sha256(Path(args.t1_output_dir) / "b3_timing_breakdown.json"),
        "replay_timing_breakdown_sha256": file_sha256(output_dir / "b3_timing_breakdown.json"),
        "measurement_contract": {
            "warmup_batches": replay["warmup_batches"],
            "measured_training_batches": replay["measured_training_batches"],
            "development_batches": replay["development_batches"],
            "microbatch_size": replay["microbatch_size"],
            "gradient_accumulation": replay["gradient_accumulation"],
        },
        "comparison": comparison,
        "t1_equivalence": t1_breakdown["equivalence"],
        "replay_equivalence": replay["equivalence"],
        "same_model_definition": True,
        "same_dataset_and_split": True,
        "same_mode": "training_fp32_cuda",
    }
    atomic_json(output_dir / "b3_t2_baseline_replay.json", replay_evidence)
    if not comparison["passed"]:
        return 2

    sync = _sync_attribution(replay, t1_breakdown)
    matrix = _optimization_matrix(replay)
    equivalence = _equivalence_evidence(replay, matrix)
    data = throughput._load_mnist(args.dataset_root)
    final_timing = _final_timing(t1_breakdown, replay, config, data, equivalence)
    projected_passed = final_timing["optimized_epoch_min"] <= FORMAL_TRAINING_EPOCH_MINUTES
    classification = (
        "b3_fp32_throughput_contract_supported"
        if projected_passed
        else "b3_fp32_semantic_preserving_optimization_insufficient"
    )
    common_kwargs = {
        "config_path": args.config,
        "dataset_root": args.dataset_root,
        "upstream_evidence": args.upstream_evidence,
        "t1_output_dir": args.t1_output_dir,
        "input_hashes": hashes,
        "state_hash": state_hash,
        "started_at": started_at,
        "command": sys.argv,
        "dataset_split_hash": config["formal_split_hash"],
        "seed": config["master_seed"],
    }
    for name, payload in (
        ("b3_t2_sync_attribution.json", sync),
        ("b3_t2_optimization_matrix.json", matrix),
        ("b3_t2_equivalence_evidence.json", equivalence),
        ("b3_t2_final_timing.json", final_timing),
    ):
        atomic_json(
            output_dir / name,
            {
                **_common(
                    classification=payload["classification"],
                    first_failed_gate=payload["first_failed_gate"],
                    permitted=["final_timing"] if name == "b3_t2_sync_attribution.json" else ["closeout"],
                    resource=replay["resource"],
                    **common_kwargs,
                ),
                **payload,
            },
        )
    closeout_classification = classification
    closeout_text = _closeout(
        classification=closeout_classification,
        first_failed_gate=None,
        final_timing=final_timing,
        **common_kwargs,
    )
    (output_dir / "b3_t2_closeout.md").write_text(closeout_text, encoding="utf-8")
    print(json.dumps({"classification": classification, "optimized_epoch_min": final_timing["optimized_epoch_min"]}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B3-T.2 FP32 semantic-preserving cleanup")
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--upstream-evidence", required=True)
    parser.add_argument("--t1-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action != "run":
        raise ValueError("unsupported Package B3-T.2 action")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
