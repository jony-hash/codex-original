"""Package B3 consumer: formal admission of a frozen software VAE.

This module consumes only hash-bound reconstruction shards.  It deliberately
does not import the v1 producer or any teacher implementation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ...evidence import atomic_write_json, file_sha256, safe_torch_load, tensor_sha256
from .electronic_distribution_recovery import (
    _classification_metrics,
    _dataset,
    _gfa_gate,
    _load_evaluators,
    _pair_metrics,
    _partition,
)


PROGRAM = "teacher-first-stage-recovery"
CONFIG_NAME = "teacher_first_stage_recovery_v1"
SCHEMA_PREFIX = "teacher_first_stage_recovery"
MODEL_VERSION = "software_multiscale_teacher_vae_v1"
FORMAL_SPLIT_HASH = "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912"
CLASS_COUNT = 10
FORBIDDEN_ACTIONS = (
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


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("name") != CONFIG_NAME:
        raise ValueError("Package B3 config name is invalid")
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": 20260827,
        "program": PROGRAM,
        "model_version": MODEL_VERSION,
        "formal_split_hash": FORMAL_SPLIT_HASH,
        "formal_bank_partition_seed": 32452843,
        "class_count": 10,
        "bank_count": 2,
        "samples_per_class": 250,
        "microbatch_size": 16,
        "primary_accuracy_minimum": 0.95,
        "minimum_recall_minimum": 0.90,
        "evaluator_agreement_minimum": 0.95,
        "diversity_ratio_minimum": 0.70,
        "diversity_ratio_maximum": 1.30,
        "coverage_ratio_maximum": 1.50,
        "memorization_anomaly_maximum": 0.05,
        "formal_evaluator_in_producer": False,
        "ddpm_training_authorized": False,
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
            raise ValueError(f"Package B3 config field {key!r} is not registered")
    for key in (
        "package_a_gate_audit",
        "package_a_closeout",
        "package_b2_closeout",
        "primary_evaluator",
        "secondary_evaluator",
        "primary_qualification",
        "secondary_qualification",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"Package B3 config field {key!r} is missing")
    return value


def _git() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"git_head": head, "dirty_worktree": bool(status), "git_status": status}


def _require_clean() -> dict[str, Any]:
    git = _git()
    if git["dirty_worktree"]:
        raise RuntimeError("Package B3 scientific consumer stages require a clean worktree")
    return git


def _base(stage: str, config: dict[str, Any], config_path: str, git: dict[str, Any], *, status: str, passed: bool, classification: str, first_failed_gate: str | None, permitted: list[str], evaluator_calls: int = 0) -> dict[str, Any]:
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
        "input_artifact_hashes": {key: config[f"{key}_sha256"] for key in ("package_a_gate_audit", "package_a_closeout", "package_b2_closeout", "primary_evaluator", "secondary_evaluator", "primary_qualification", "secondary_qualification")},
        "dataset_split_hash": config["formal_split_hash"],
        "git_head": git["git_head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": config["master_seed"],
        "evaluator_calls": evaluator_calls,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "synthetic_diagnostic_only": True,
    }


def _validate_upstream(config: dict[str, Any], manifest_path: str, dataset_root: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _json(manifest_path)
    if manifest.get("schema") != f"{SCHEMA_PREFIX}_reconstruction.v1" or manifest.get("status") != "first_stage_reconstruction_bank":
        raise ValueError("Package B3 reconstruction manifest is not a valid producer artifact")
    if manifest.get("model_version") != MODEL_VERSION or manifest.get("dataset_split_hash") != config["formal_split_hash"]:
        raise ValueError("Package B3 reconstruction lineage mismatch")
    if manifest.get("formal_bank_partition_seed") != config["formal_bank_partition_seed"]:
        raise ValueError("Package B3 formal bank partition seed mismatch")
    if manifest.get("evaluator_calls") != 0 or manifest.get("state_hash_before") != manifest.get("state_hash_after"):
        raise ValueError("Package B3 producer evidence violates evaluator/frozen-state contract")
    stats_path = Path(str(manifest.get("latent_stats_path", "")))
    if not stats_path.is_file():
        stats_path = Path(manifest_path).parent / stats_path
    if not stats_path.is_file() or file_sha256(stats_path) != manifest.get("latent_stats_sha256"):
        raise ValueError("Package B3 latent statistics artifact is missing or tampered")
    stats = safe_torch_load(stats_path, expected_sha256=str(manifest["latent_stats_sha256"]))
    if stats.get("schema") != f"{SCHEMA_PREFIX}_latent_stats.v1" or stats.get("model_version") != MODEL_VERSION or stats.get("latent_shape") != [1, 25, 25]:
        raise ValueError("Package B3 latent statistics lineage is invalid")
    if not torch.is_tensor(stats.get("mean")) or not torch.is_tensor(stats.get("std")) or tuple(stats["mean"].shape) != (1,) or tuple(stats["std"].shape) != (1,) or not torch.isfinite(stats["mean"]).all() or not torch.isfinite(stats["std"]).all() or not (stats["std"] > 0).all():
        raise ValueError("Package B3 latent statistics values are invalid")
    shards = manifest.get("shards")
    expected_count = 2 * CLASS_COUNT * ((250 + config["microbatch_size"] - 1) // config["microbatch_size"])
    if not isinstance(shards, list) or len(shards) != expected_count:
        raise ValueError("Package B3 reconstruction shard count is invalid")
    seen: set[tuple[int, int, int]] = set()
    manifest_path_obj = Path(manifest_path)
    for row in shards:
        if not isinstance(row, dict):
            raise ValueError("Package B3 reconstruction shard record is invalid")
        key = (int(row["bank"]), int(row["label"]), int(row["start"]))
        end = int(row["end"])
        if key in seen or not 0 <= key[0] < 2 or not 0 <= key[1] < CLASS_COUNT or key[2] < 0 or end <= key[2] or end > 250:
            raise ValueError("Package B3 reconstruction shard ranges overlap or are invalid")
        path = Path(str(row["path"]))
        if not path.is_file():
            path = manifest_path_obj.parent / path
        if not path.is_file() or file_sha256(path) != row.get("sha256"):
            raise ValueError("Package B3 reconstruction shard hash mismatch")
        payload = safe_torch_load(path, expected_sha256=str(row["sha256"]))
        count = end - key[2]
        if payload.get("schema") != f"{SCHEMA_PREFIX}_reconstruction_shard.v1" or payload.get("bank") != key[0] or payload.get("label") != key[1] or payload.get("start") != key[2] or payload.get("end") != end:
            raise ValueError("Package B3 reconstruction shard metadata mismatch")
        images, labels, sources, mu = payload.get("images"), payload.get("labels"), payload.get("source_indices"), payload.get("mu")
        if not torch.is_tensor(images) or tuple(images.shape) != (count, 1, 200, 200) or not torch.isfinite(images).all():
            raise ValueError("Package B3 reconstruction shard image tensor is invalid")
        if not torch.is_tensor(labels) or labels.tolist() != [key[1]] * count or not isinstance(sources, list) or len(sources) != count:
            raise ValueError("Package B3 reconstruction shard labels or source indices are invalid")
        if not torch.is_tensor(mu) or tuple(mu.shape) != (count, 1, 25, 25) or not torch.isfinite(mu).all():
            raise ValueError("Package B3 reconstruction shard latent tensor is invalid")
        seen.add(key)
    for bank in range(2):
        for label in range(CLASS_COUNT):
            starts = sorted(row[2] for row in seen if row[:2] == (bank, label))
            if starts != list(range(0, 250, config["microbatch_size"])):
                raise ValueError("Package B3 reconstruction shard coverage is incomplete")
    for key in ("package_a_gate_audit", "package_a_closeout", "package_b2_closeout", "primary_evaluator", "secondary_evaluator", "primary_qualification", "secondary_qualification"):
        path = Path(config[key])
        if not path.is_file() or file_sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"Package B3 locked input hash mismatch: {key}")
    data = _dataset(dataset_root)
    if data["split_hash"] != config["formal_split_hash"] or data["dataset_hash"] != manifest.get("dataset_hash"):
        raise ValueError("Package B3 reconstruction dataset lineage mismatch")
    return manifest, data


def _load_bank(manifest_path: str, manifest: dict[str, Any], bank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(manifest_path).parent
    rows = []
    for row in sorted((row for row in manifest["shards"] if int(row["bank"]) == bank), key=lambda row: (int(row["label"]), int(row["start"]))):
        path = Path(str(row["path"]))
        if not path.is_file():
            path = root / path
        payload = safe_torch_load(path, expected_sha256=str(row["sha256"]))
        rows.append((payload["images"].float(), payload["labels"].long(), payload["mu"].float()))
    return tuple(torch.cat(row[index], dim=0) for index in range(3))  # type: ignore[return-value]


def _teacher_views(images: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(images, (28, 28))


def _classify(*, all_banks_passed: bool, negative_control_rejected: bool) -> tuple[str, str | None, list[str]]:
    if not negative_control_rejected:
        return "invalid_harness", "label_shift_negative_control", ["label_shift_negative_control_passed"]
    if all_banks_passed:
        return "first_stage_reconstruction_qualified", None, ["software_first_stage_distribution_supported"]
    return "first_stage_reconstruction_failed", "first_stage_reconstruction_distribution_gate", ["software_first_stage_does_not_meet_electronic_contract"]


def _qualification_check(path: str, expected_hash: str, evaluator_hash: str, *, secondary: bool) -> None:
    if file_sha256(path) != expected_hash:
        raise ValueError("Package B3 evaluator qualification hash mismatch")
    payload = _json(path)
    if payload.get("status") != "passed" or payload.get("schema_version") != 2:
        raise ValueError("Package B3 evaluator qualification is not passed")
    if payload.get("context", {}).get("dataset_split_manifest_hash") != FORMAL_SPLIT_HASH:
        raise ValueError("Package B3 evaluator qualification split mismatch")
    inputs = payload.get("input_hashes", {})
    bound = {inputs.get("evaluator_checkpoint"), payload.get("context", {}).get("evaluator_checkpoint_hash")}
    if secondary:
        bound.update({inputs.get("secondary_evaluator"), inputs.get("smooth_evaluator")})
    if evaluator_hash not in bound:
        raise ValueError("Package B3 evaluator qualification checkpoint mismatch")


def _audit(config: dict[str, Any], manifest_path: str, manifest: dict[str, Any], data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Package B3 formal audit requires CUDA")
    device = torch.device(args.device)
    _qualification_check(config["primary_qualification"], config["primary_qualification_sha256"], config["primary_evaluator_sha256"], secondary=False)
    _qualification_check(config["secondary_qualification"], config["secondary_qualification_sha256"], config["secondary_evaluator_sha256"], secondary=True)
    primary, secondary = _load_evaluators(args, device)
    real_a, real_b = _partition(data["validation_images"], data["validation_labels"], seed=config["formal_bank_partition_seed"], samples_per_class=250)
    train_images, train_labels = data["train_images"].to(device), data["train_labels"].to(device)
    counter: Counter[str] = Counter()
    train_primary, _ = _eval_chunks_local(primary, train_images, counter)
    train_secondary, _ = _eval_chunks_local(secondary, train_images, counter)
    banks = []
    for bank_index, reference in enumerate((real_b, real_a)):
        images, labels, mu = _load_bank(manifest_path, manifest, bank_index)
        candidate = _teacher_views(images).to(device)
        reference_images = reference[0].to(device)
        metrics, calls = _pair_metrics(primary, secondary, candidate, reference_images, labels.to(device), train_images, train_labels, train_primary, train_secondary)
        metrics.update({"bank": bank_index, "candidate_camera_hash": tensor_sha256(images), "latent_mu_hash": tensor_sha256(mu), "passed": _gfa_gate(metrics, config)})
        counter["pair_metric_calls"] += calls
        banks.append(metrics)
    shifted_predictions, shifted_targets = [], []
    for bank in range(2):
        images, labels, _ = _load_bank(manifest_path, manifest, bank)
        predictions = _eval_chunks_local(primary, _teacher_views(images).to(device), counter)[1].argmax(1)
        shifted_predictions.append(predictions)
        shifted_targets.append(((labels + 1) % CLASS_COUNT).to(device))
    negative = _classification_metrics(torch.cat(shifted_predictions), torch.cat(shifted_targets))
    negative["rejected"] = not (negative["accuracy"] >= config["primary_accuracy_minimum"] and negative["minimum_recall"] >= config["minimum_recall_minimum"])
    all_banks_passed = all(row["passed"] for row in banks)
    classification, first_failed, findings = _classify(all_banks_passed=all_banks_passed, negative_control_rejected=bool(negative["rejected"]))
    return {"passed": classification != "invalid_harness", "classification": classification, "first_failed_gate": first_failed, "findings": findings, "banks": banks, "negative_control": negative, "all_banks_passed": all_banks_passed, "evaluator_calls": sum(counter.values()), "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))}


def _eval_chunks_local(model: torch.nn.Module, values: torch.Tensor, counter: Counter[str], *, chunk_size: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    features, logits = [], []
    with torch.no_grad():
        for start in range(0, values.shape[0], chunk_size):
            feature, logit = model(values[start : start + chunk_size])
            counter["evaluator_forward"] += 1
            features.append(feature)
            logits.append(logit)
    return torch.cat(features), torch.cat(logits)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B3 software first-stage admission")
    parser.add_argument("research_action", choices=("validate", "audit", "report", "closeout"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reconstruction-manifest", required=True)
    parser.add_argument("--primary-evaluator-checkpoint", required=True)
    parser.add_argument("--secondary-evaluator-checkpoint", required=True)
    parser.add_argument("--primary-qualification-evidence", required=True)
    parser.add_argument("--secondary-qualification-evidence", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-evidence")
    parser.add_argument("--audit-evidence")
    parser.add_argument("--report-evidence")
    return parser


def _args_match_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    for argument, key in (("primary_evaluator_checkpoint", "primary_evaluator"), ("secondary_evaluator_checkpoint", "secondary_evaluator"), ("primary_qualification_evidence", "primary_qualification"), ("secondary_qualification_evidence", "secondary_qualification")):
        if Path(getattr(args, argument)).resolve() != Path(config[key]).resolve():
            raise ValueError(f"Package B3 explicit {argument} does not match locked config")


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    _args_match_config(args, config)
    git = _require_clean()
    action = args.research_action
    if action == "validate":
        manifest, data = _validate_upstream(config, args.reconstruction_manifest, args.dataset_root)
        payload = _base("validation", config, args.config, git, status="valid", passed=True, classification="reconstruction_lineage_validated", first_failed_gate=None, permitted=["audit"])
        payload.update({"reconstruction_manifest_sha256": file_sha256(args.reconstruction_manifest), "dataset_hash": data["dataset_hash"], "model_version": manifest["model_version"], "formal_bank_count": manifest["bank_count"], "samples_per_class": manifest["samples_per_class"]})
        atomic_write_json(args.output, payload)
        return 0
    if action == "audit":
        if not args.validation_evidence:
            raise ValueError("audit requires --validation-evidence")
        validation = _json(args.validation_evidence)
        if validation.get("schema") != f"{SCHEMA_PREFIX}_validation.v1" or not validation.get("passed"):
            raise ValueError("audit requires passed validation evidence")
        manifest, data = _validate_upstream(config, args.reconstruction_manifest, args.dataset_root)
        result = _audit(config, args.reconstruction_manifest, manifest, data, args)
        payload = _base("audit", config, args.config, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=["report"] if result["passed"] else [], evaluator_calls=result["evaluator_calls"])
        payload.update(result)
        payload["reconstruction_manifest_sha256"] = file_sha256(args.reconstruction_manifest)
        payload["validation_evidence_sha256"] = file_sha256(args.validation_evidence)
        atomic_write_json(args.output, payload)
        return 0
    upstream_path = args.audit_evidence if action == "report" else args.report_evidence
    if not upstream_path:
        raise ValueError(f"{action} requires its upstream evidence")
    upstream = _json(upstream_path)
    expected_schema = f"{SCHEMA_PREFIX}_audit.v1" if action == "report" else f"{SCHEMA_PREFIX}_report.v1"
    if upstream.get("schema") != expected_schema or upstream.get("status") != "valid":
        raise ValueError(f"{action} requires valid upstream evidence")
    if action == "report":
        payload = {"schema": f"{SCHEMA_PREFIX}_report.v1", "status": "valid", "classification": upstream["classification"], "first_failed_gate": upstream.get("first_failed_gate"), "findings": upstream.get("findings", []), "permitted_next_actions": ["closeout"], "forbidden_next_actions": list(FORBIDDEN_ACTIONS), "audit_evidence_sha256": file_sha256(upstream_path), "git_head": git["git_head"], "dirty_worktree": git["dirty_worktree"], "input_artifact_hashes": upstream.get("input_artifact_hashes", {}), "dataset_split_hash": config["formal_split_hash"], "evaluator_calls": upstream.get("evaluator_calls", 0)}
    else:
        classification = upstream["classification"]
        permitted = ["design_and_train_matched_b3_ddpm"] if classification == "first_stage_reconstruction_qualified" else ["close_first_stage_candidate"] if classification == "first_stage_reconstruction_failed" else []
        payload = {"schema": f"{SCHEMA_PREFIX}_closeout.v1", "status": "valid", "classification": classification, "first_failed_gate": upstream.get("first_failed_gate"), "permitted_next_actions": permitted, "forbidden_next_actions": list(FORBIDDEN_ACTIONS), "report_evidence_sha256": file_sha256(upstream_path), "git_head": git["git_head"], "dirty_worktree": git["dirty_worktree"], "dataset_split_hash": config["formal_split_hash"], "ddpm_training": "forbidden_without_reconstruction_qualification", "student_training": "forbidden", "optical_training": "forbidden", "candidate_export": "forbidden"}
    atomic_write_json(args.output, payload)
    if action == "closeout":
        synthesis = Path(args.output).with_name("teacher_first_stage_recovery_synthesis.md")
        synthesis.write_text(f"# Package B3-VAE synthesis\n\n- Classification: `{payload['classification']}`\n- Evidence: `{file_sha256(upstream_path)}`\n- DDPM, student and optical training: forbidden until the recorded next action is separately authorized.\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    return dispatch(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
