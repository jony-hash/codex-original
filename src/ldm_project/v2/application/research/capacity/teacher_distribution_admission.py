"""Package B consumer: dual-evaluator admission of frozen teacher banks.

The producer is intentionally v1-only.  This module consumes only its
hash-bound raw camera-plane artifact and uses the same v2 metrics as Package
A, so the two runtime contracts remain separated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ldm_project.v2.evidence import atomic_write_json, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.application.research.capacity.electronic_distribution_recovery import CLASS_COUNT, _classification_metrics, _dataset, _eval_chunks, _gfa_gate, _pair_metrics, _partition, _load_evaluators


PROGRAM = "teacher-distribution-admission"
CONFIG_NAME = "teacher_distribution_admission_v1"
SCHEMA_PREFIX = "teacher_distribution_admission"
FORBIDDEN_ACTIONS = (
    "teacher_training",
    "ddpm_training",
    "student_training",
    "optical_training",
    "phase_training",
    "quadrature_training",
    "ols_training",
    "fiber_training",
    "candidate_export",
    "promotion",
)


def _git() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"head": head, "dirty_worktree": bool(status), "status": status}


def _require_clean() -> dict[str, Any]:
    git = _git()
    if git["dirty_worktree"]:
        raise RuntimeError("Package B scientific stages require a clean worktree")
    return git


def _config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("name") != CONFIG_NAME:
        raise ValueError("Package B config name is invalid")
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": 20260826,
        "program": PROGRAM,
        "class_count": 10,
        "bank_count": 2,
        "samples_per_class": 250,
        "microbatch_size": 16,
        "dataset_split_hash": "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912",
        "guidance_scale": 2.0,
        "sampling_steps": 200,
        "sampler": "deterministic_ddim",
        "primary_accuracy_minimum": 0.95,
        "minimum_recall_minimum": 0.90,
        "evaluator_agreement_minimum": 0.95,
        "diversity_ratio_minimum": 0.70,
        "diversity_ratio_maximum": 1.30,
        "coverage_ratio_maximum": 1.50,
        "memorization_anomaly_maximum": 0.05,
        "teacher_training_authorized": False,
        "student_training_authorized": False,
        "optical_training_authorized": False,
        "phase_training_authorized": False,
        "quadrature_training_authorized": False,
        "ols_training_authorized": False,
        "fiber_training_authorized": False,
        "candidate_export_authorized": False,
        "promotion_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"Package B config field {key!r} is not registered")
    return value


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _base(stage: str, config: dict[str, Any], config_path: str, git: dict[str, Any], *, status: str, passed: bool, classification: str, first_failed_gate: str | None, permitted: list[str], evaluator_calls: int = 0) -> dict[str, Any]:
    inputs = {
        "first_stage_checkpoint": config["first_stage_checkpoint_sha256"],
        "diffusion_checkpoint": config["diffusion_checkpoint_sha256"],
        "primary_evaluator": config["primary_evaluator_sha256"],
        "secondary_evaluator": config["secondary_evaluator_sha256"],
        "primary_qualification": config["primary_qualification_sha256"],
        "secondary_qualification": config["secondary_qualification_sha256"],
        "package_a_gate_audit": config["package_a_gate_audit_sha256"],
        "package_a_closeout": config["package_a_closeout_sha256"],
    }
    return {
        "schema": f"{SCHEMA_PREFIX}_{stage}.v1",
        "stage": stage,
        "status": status,
        "passed": passed,
        "classification": classification,
        "first_failed_gate": first_failed_gate,
        "permitted_next_actions": permitted,
        "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
        "config_sha256": file_sha256(config_path),
        "input_artifact_hashes": inputs,
        "dataset_split_hash": config["dataset_split_hash"],
        "git_head": git["head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": config["master_seed"],
        "evaluator_calls": evaluator_calls,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "synthetic_diagnostic_only": True,
    }


def _qualification_check(path: str, expected_hash: str, evaluator_hash: str, *, secondary: bool) -> None:
    if file_sha256(path) != expected_hash:
        raise ValueError("Package B evaluator qualification hash does not match config")
    value = _json(path)
    if value.get("status") != "passed" or value.get("schema_version") != 2:
        raise ValueError("Package B evaluator qualification is not passed")
    context = value.get("context", {})
    if context.get("dataset_split_manifest_hash") != "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912":
        raise ValueError("Package B evaluator qualification split mismatch")
    bound = {value.get("input_hashes", {}).get("evaluator_checkpoint"), context.get("evaluator_checkpoint_hash")}
    if secondary:
        bound.update({value.get("input_hashes", {}).get("secondary_evaluator"), value.get("input_hashes", {}).get("smooth_evaluator")})
    if evaluator_hash not in bound:
        raise ValueError("Package B evaluator qualification checkpoint mismatch")


def _resolve_shard(manifest_path: Path, value: str) -> Path:
    direct = Path(value)
    if direct.is_file():
        return direct
    relative = manifest_path.parent / direct
    if relative.is_file():
        return relative
    raise FileNotFoundError(f"teacher bank shard is missing: {value}")


def _validate_manifest(path: str, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _json(manifest_path)
    if manifest.get("schema") != f"{SCHEMA_PREFIX}_banks.v1" or manifest.get("status") != "teacher_audit_bank":
        raise ValueError("teacher bank manifest is not an audit artifact")
    if manifest.get("git_head") is None or manifest.get("dirty_worktree"):
        raise ValueError("teacher bank manifest has invalid lineage state")
    if manifest.get("config") != config:
        raise ValueError("teacher bank manifest config binding mismatch")
    if manifest.get("checkpoint_hashes", {}).get("first_stage") != config["first_stage_checkpoint_sha256"] or manifest.get("checkpoint_hashes", {}).get("diffusion") != config["diffusion_checkpoint_sha256"]:
        raise ValueError("teacher bank checkpoint lineage mismatch")
    if manifest.get("dataset_split_hash") != config["dataset_split_hash"] or manifest.get("bank_count") != config["bank_count"] or manifest.get("samples_per_class") != config["samples_per_class"]:
        raise ValueError("teacher bank contract mismatch")
    sampler = manifest.get("sampler", {})
    if sampler.get("kind") != config["sampler"] or sampler.get("sampling_steps") != config["sampling_steps"] or sampler.get("guidance_scale") != config["guidance_scale"]:
        raise ValueError("teacher sampler contract mismatch")
    if manifest.get("noise") != "full_standard_gaussian":
        raise ValueError("teacher bank does not use full Gaussian noise")
    shards = manifest.get("shards")
    expected_count = config["bank_count"] * config["class_count"] * ((config["samples_per_class"] + config["microbatch_size"] - 1) // config["microbatch_size"])
    if not isinstance(shards, list) or len(shards) != expected_count:
        raise ValueError("teacher bank shard count is invalid")
    seen: set[tuple[int, int, int]] = set()
    for record in shards:
        if not isinstance(record, dict):
            raise ValueError("teacher bank shard record is invalid")
        key = (int(record["bank"]), int(record["label"]), int(record["start"]))
        if (
            not 0 <= key[0] < config["bank_count"]
            or not 0 <= key[1] < config["class_count"]
            or key in seen
            or key[2] < 0
            or int(record["end"]) <= key[2]
            or int(record["end"]) > config["samples_per_class"]
        ):
            raise ValueError("teacher bank shard ranges overlap or are invalid")
        shard_path = _resolve_shard(manifest_path, str(record["path"]))
        if file_sha256(shard_path) != record.get("sha256"):
            raise ValueError(f"teacher bank shard hash changed: {shard_path}")
        payload = safe_torch_load(shard_path, expected_sha256=str(record["sha256"]))
        if payload.get("schema") != f"{SCHEMA_PREFIX}_shard.v1" or payload.get("bank") != key[0] or payload.get("label") != key[1] or payload.get("start") != key[2] or payload.get("end") != int(record["end"]):
            raise ValueError("teacher bank shard metadata mismatch")
        images, labels, seeds = payload.get("images"), payload.get("labels"), payload.get("seeds")
        count = int(record["end"]) - key[2]
        if not torch.is_tensor(images) or tuple(images.shape) != (count, 1, config["image_size"], config["image_size"]):
            raise ValueError("teacher bank shard image shape mismatch")
        if not torch.is_tensor(labels) or labels.tolist() != [key[1]] * count or not isinstance(seeds, list) or len(seeds) != count:
            raise ValueError("teacher bank shard labels or seeds mismatch")
        if not torch.isfinite(images).all():
            raise ValueError("teacher bank shard contains non-finite values")
        seen.add(key)
    for bank in range(config["bank_count"]):
        for label in range(config["class_count"]):
            starts = sorted(key[2] for key in seen if key[:2] == (bank, label))
            expected_starts = list(range(0, config["samples_per_class"], config["microbatch_size"]))
            if starts != expected_starts:
                raise ValueError("teacher bank shard coverage is incomplete")
    return manifest


def _validate_lineage(config: dict[str, Any], config_path: str, manifest_path: str, package_a_gate: str, package_a_closeout: str, dataset_root: str) -> dict[str, Any]:
    manifest = _validate_manifest(manifest_path, config)
    for key in ("primary_evaluator", "secondary_evaluator", "primary_qualification", "secondary_qualification", "package_a_gate_audit", "package_a_closeout"):
        expected = config[f"{key}_sha256"]
        path = config[key]
        if file_sha256(path) != expected:
            raise ValueError(f"Package B {key} hash does not match config")
    _qualification_check(config["primary_qualification"], config["primary_qualification_sha256"], config["primary_evaluator_sha256"], secondary=False)
    _qualification_check(config["secondary_qualification"], config["secondary_qualification_sha256"], config["secondary_evaluator_sha256"], secondary=True)
    gate = _json(config["package_a_gate_audit"])
    closeout = _json(config["package_a_closeout"])
    if gate.get("schema") != "electronic_distribution_recovery_gate_audit.v1" or gate.get("classification") != "gate_feasibility_supported" or not gate.get("passed"):
        raise ValueError("Package B requires valid Package A H-GFA gate evidence")
    if closeout.get("schema") != "electronic_distribution_recovery_closeout.v1" or closeout.get("status") != "valid":
        raise ValueError("Package A closeout evidence is invalid")
    data = _dataset(dataset_root)
    if data["split_hash"] != config["dataset_split_hash"]:
        raise ValueError("Package B dataset split is not the qualified split")
    return {"manifest": manifest, "manifest_sha256": file_sha256(manifest_path), "dataset_hash": data["dataset_hash"], "split_hash": data["split_hash"]}


def _load_bank(manifest_path: str, manifest: dict[str, Any], bank: int, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    manifest_file = Path(manifest_path)
    for record in sorted((row for row in manifest["shards"] if row["bank"] == bank), key=lambda row: (row["label"], row["start"])):
        payload = safe_torch_load(_resolve_shard(manifest_file, record["path"]), expected_sha256=record["sha256"])
        rows.append((payload["images"].to(torch.float32), payload["labels"].to(torch.long)))
    images = torch.cat([row[0] for row in rows], dim=0)
    labels = torch.cat([row[1] for row in rows], dim=0)
    return images, labels


def _teacher_view(images: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(images, (28, 28))


def _audit(config: dict[str, Any], manifest_path: str, manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Package B admission requires CUDA")
    device = torch.device(args.device)
    data = _dataset(args.dataset_root)
    real_a, real_b = _partition(data["validation_images"], data["validation_labels"], seed=32452843, samples_per_class=250)
    primary, secondary = _load_evaluators(args, device)
    train_images, train_labels = data["train_images"].to(device), data["train_labels"].to(device)
    counter = Counter()
    train_primary, _ = _eval_chunks(primary, train_images, counter)
    train_secondary, _ = _eval_chunks(secondary, train_images, counter)
    banks = []
    start = time.monotonic()
    for bank_index, real_reference in enumerate((real_b, real_a)):
        teacher_images, labels = _load_bank(manifest_path, manifest, bank_index, config)
        teacher_views = _teacher_view(teacher_images).to(device)
        reference = real_reference[0].to(device)
        real_labels = real_reference[1].to(device)
        metrics, calls = _pair_metrics(primary, secondary, teacher_views, reference, labels.to(device), train_images, train_labels, train_primary, train_secondary)
        metrics["bank"] = bank_index
        metrics["reference_real_hash"] = tensor_sha256(reference.cpu())
        metrics["teacher_camera_hash"] = tensor_sha256(teacher_images)
        metrics["passed"] = _gfa_gate(metrics, config)
        counter["teacher_bank"] += 1
        banks.append(metrics)
        del teacher_images, teacher_views, reference, real_labels
        torch.cuda.empty_cache()
    shifted_predictions = []
    shifted_labels = []
    for bank_index in range(config["bank_count"]):
        teacher_images, labels = _load_bank(manifest_path, manifest, bank_index, config)
        views = _teacher_view(teacher_images).to(device)
        shifted_labels.append(((labels + 1) % CLASS_COUNT).to(device))
        shifted_predictions.append(_eval_chunks(primary, views, counter)[1].argmax(1))
    negative_metrics = _classification_metrics(torch.cat(shifted_predictions), torch.cat(shifted_labels))
    negative_metrics["rejected"] = not (negative_metrics["accuracy"] >= config["primary_accuracy_minimum"] and negative_metrics["minimum_recall"] >= config["minimum_recall_minimum"])
    negative_metrics["evaluator"] = "primary"
    all_banks_passed = all(row["passed"] for row in banks)
    if not negative_metrics["rejected"]:
        classification = "invalid_harness"
        first_failed = "label_shift_negative_control"
        findings = ["label_shift_negative_control_passed"]
    elif all_banks_passed:
        classification = "teacher_distribution_qualified"
        first_failed = None
        findings = ["teacher_dual_evaluator_range_supported"]
    else:
        classification = "teacher_distribution_failed"
        first_failed = "teacher_dual_evaluator_range_gate"
        findings = ["teacher_distribution_does_not_meet_electronic_contract"]
    state = {
        "first_stage": config["first_stage_checkpoint_sha256"],
        "diffusion": config["diffusion_checkpoint_sha256"],
    }
    return {
        "passed": classification != "invalid_harness",
        "classification": classification,
        "first_failed_gate": first_failed,
        "findings": findings,
        "banks": banks,
        "negative_control": negative_metrics,
        "all_banks_passed": all_banks_passed,
        "state_hash_before": state,
        "state_hash_after": state,
        "evaluator_calls": int(counter["evaluator_forward"]),
        "elapsed_seconds": time.monotonic() - start,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer_created": False,
        "checkpoint_created": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B frozen teacher distribution admission")
    parser.add_argument("research_action", choices=("validate", "audit", "report", "closeout"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--teacher-bank-manifest", required=True)
    parser.add_argument("--primary-evaluator-checkpoint", required=True)
    parser.add_argument("--secondary-evaluator-checkpoint", required=True)
    parser.add_argument("--primary-qualification-evidence", required=True)
    parser.add_argument("--secondary-qualification-evidence", required=True)
    parser.add_argument("--package-a-gate-evidence", required=True)
    parser.add_argument("--package-a-closeout-evidence", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-evidence")
    parser.add_argument("--audit-evidence")
    parser.add_argument("--report-evidence")
    return parser


def _args_match_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pairs = {
        "primary_evaluator_checkpoint": "primary_evaluator",
        "secondary_evaluator_checkpoint": "secondary_evaluator",
        "primary_qualification_evidence": "primary_qualification",
        "secondary_qualification_evidence": "secondary_qualification",
        "package_a_gate_evidence": "package_a_gate_audit",
        "package_a_closeout_evidence": "package_a_closeout",
    }
    for arg_name, config_key in pairs.items():
        if Path(getattr(args, arg_name)).resolve() != Path(config[config_key]).resolve():
            raise ValueError(f"Package B explicit {arg_name} does not match locked config")


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    _args_match_config(args, config)
    git = _require_clean()
    action = args.research_action
    if action == "validate":
        lineage = _validate_lineage(config, args.config, args.teacher_bank_manifest, args.package_a_gate_evidence, args.package_a_closeout_evidence, args.dataset_root)
        payload = _base("validation", config, args.config, git, status="valid", passed=True, classification="teacher_lineage_validated", first_failed_gate=None, permitted=["audit"])
        payload.update(lineage)
        atomic_write_json(args.output, payload)
        return 0
    if action == "audit":
        if not args.validation_evidence:
            raise ValueError("audit requires --validation-evidence")
        validation = _json(args.validation_evidence)
        if validation.get("schema") != f"{SCHEMA_PREFIX}_validation.v1" or not validation.get("passed"):
            raise ValueError("audit requires passed Package B validation evidence")
        lineage = _validate_lineage(config, args.config, args.teacher_bank_manifest, args.package_a_gate_evidence, args.package_a_closeout_evidence, args.dataset_root)
        result = _audit(config, args.teacher_bank_manifest, lineage["manifest"], args)
        payload = _base("audit", config, args.config, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=["report"] if result["passed"] else [], evaluator_calls=result["evaluator_calls"])
        payload.update(result)
        payload["teacher_bank_manifest_sha256"] = lineage["manifest_sha256"]
        payload["validation_evidence_sha256"] = file_sha256(args.validation_evidence)
        atomic_write_json(args.output, payload)
        return 0
    evidence_path = (
        args.audit_evidence
        if action == "report"
        else args.report_evidence
        if action == "closeout"
        else args.validation_evidence
    )
    if evidence_path is None:
        raise ValueError(f"{action} requires --audit-evidence")
    evidence = _json(evidence_path)
    expected_schema = f"{SCHEMA_PREFIX}_audit.v1" if action == "report" else f"{SCHEMA_PREFIX}_report.v1"
    if evidence.get("schema") != expected_schema or evidence.get("status") != "valid":
        raise ValueError(f"{action} requires valid upstream evidence")
    if action == "report":
        report = {
            "schema": f"{SCHEMA_PREFIX}_report.v1",
            "status": "valid",
            "classification": evidence["classification"],
            "first_failed_gate": evidence.get("first_failed_gate"),
            "findings": evidence.get("findings", []),
            "permitted_next_actions": ["closeout"],
            "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
            "audit_evidence_sha256": file_sha256(evidence_path),
            "git_head": git["head"],
            "dirty_worktree": git["dirty_worktree"],
            "input_artifact_hashes": evidence.get("input_artifact_hashes", {}),
            "dataset_split_hash": config["dataset_split_hash"],
            "evaluator_calls": evidence.get("evaluator_calls", 0),
        }
        atomic_write_json(args.output, report)
        return 0
    classification = evidence["classification"]
    permitted = ["separately_reviewed_package_c"] if classification == "teacher_distribution_qualified" else ["close_teacher_route"] if classification == "teacher_distribution_failed" else []
    closeout = {
        "schema": f"{SCHEMA_PREFIX}_closeout.v1",
        "status": "valid",
        "classification": classification,
        "first_failed_gate": evidence.get("first_failed_gate"),
        "permitted_next_actions": permitted,
        "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
        "report_evidence_sha256": file_sha256(evidence_path),
        "git_head": git["head"],
        "dirty_worktree": git["dirty_worktree"],
        "dataset_split_hash": config["dataset_split_hash"],
        "promotion": "forbidden",
        "student_training": "forbidden",
        "optical_training": "forbidden",
        "candidate_export": "forbidden",
    }
    atomic_write_json(args.output, closeout)
    synthesis = Path(args.output).with_name("teacher_distribution_synthesis.md")
    synthesis.write_text(
        "# Teacher distribution admission synthesis\n\n"
        f"- Classification: `{classification}`\n"
        f"- Evidence: `{file_sha256(evidence_path)}`\n"
        "- Teacher/student/optical training and promotion: forbidden.\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return dispatch(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
