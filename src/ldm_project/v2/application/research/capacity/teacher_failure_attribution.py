"""Package B2 consumer: no-update attribution of a frozen teacher failure.

The v2 side consumes raw first-stage reconstructions and normalized diffusion
latents produced by the v1 producer.  It never imports v1 teacher code; the
only evaluator calls happen after all arms and their hashes are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ldm_project.v2.evidence import atomic_write_json, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.application.research.capacity.electronic_distribution_recovery import CLASS_COUNT, _classification_metrics, _dataset, _eval_chunks, _gfa_gate, _load_evaluators, _pair_metrics, _partition
from ldm_project.v2.application.research.capacity.teacher_distribution_admission import _config as _package_b_config, _load_bank, _teacher_view, _validate_manifest as _validate_package_b_manifest


PROGRAM = "teacher-failure-attribution"
CONFIG_NAME = "teacher_failure_attribution_v1"
SCHEMA_PREFIX = "teacher_failure_attribution"
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
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty_worktree": bool(status), "status": status}


def _require_clean() -> dict[str, Any]:
    git = _git()
    if git["dirty_worktree"]:
        raise RuntimeError("Package B2 scientific stages require a clean worktree")
    return git


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _config(path: str | Path) -> dict[str, Any]:
    value = _json_yaml(path)
    if value.get("name") != CONFIG_NAME:
        raise ValueError("Package B2 config name is invalid")
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": 20260827,
        "program": PROGRAM,
        "first_stage_checkpoint_sha256": "e553bfab77e405265482a25ab9b569f0e6cc4e26940fd332985b77156d8bafdd",
        "diffusion_checkpoint_sha256": "60f7592b2bc803779af7aad809bdd90b280d183ef6e029586698c11bd3d68cd0",
        "package_b_config_sha256": "99f6dcd5895ac0651a15cba9a469756a002e8f331fc4a5bb9133e994ae1ee798",
        "package_b_manifest_sha256": "c9aebd1e21e8ed185eb1c0db35902a7a4c67c90d01ebd25153a45ca496139ccb",
        "package_b_audit_sha256": "0369e9500158a94a675e393674f3deff6edf8de0e5d6775e617750b17dbfea92",
        "package_b_closeout_sha256": "2620311e2bedb6ba6188afc2eae5af5fb9947410e8d70dac892971e2fdc3c6f7",
        "package_a_gate_audit_sha256": "99de4cae5f83cda643fd5accd09e54795d24b74d1629539f033f4156e47ac0bf",
        "package_a_closeout_sha256": "70ed0fe02e9242ec630b44f3e46dd386950ef8d10d1ab5769664b5cb211a8cab",
        "primary_evaluator_sha256": "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57",
        "secondary_evaluator_sha256": "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c",
        "primary_qualification_sha256": "26fea381bec5d6be2a2ed69806c190f7a7051b5f0944e5f90941a7d93f35695e",
        "secondary_qualification_sha256": "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88",
        "dataset_split_hash": "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912",
        "teacher_model_version": "physical_first_linear_metasurface_vae_v5",
        "diffusion_model_version": "conditioned_ldm_unet_v1",
        "latent_source": "mu",
        "latent_shape": [1, 25, 25],
        "image_size": 200,
        "class_count": 10,
        "bank_count": 2,
        "samples_per_class": 250,
        "microbatch_size": 16,
        "master_seed": 32452843,
        "partition_seed": 32452843,
        "split_seed": 1234,
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
            raise ValueError(f"Package B2 config field {key!r} is not registered")
    for key in (
        "first_stage_checkpoint",
        "diffusion_checkpoint",
        "package_b_config",
        "package_b_manifest",
        "package_b_audit",
        "package_b_closeout",
        "package_a_gate_audit",
        "package_a_closeout",
        "primary_evaluator",
        "secondary_evaluator",
        "primary_qualification",
        "secondary_qualification",
    ):
        if not isinstance(value.get(key), str):
            raise ValueError(f"Package B2 config field {key!r} is missing")
    return value


def _json_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


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
        "input_artifact_hashes": {
            "first_stage_checkpoint": config["first_stage_checkpoint_sha256"],
            "diffusion_checkpoint": config["diffusion_checkpoint_sha256"],
            "package_b_config": config["package_b_config_sha256"],
            "package_b_manifest": config["package_b_manifest_sha256"],
            "package_b_audit": config["package_b_audit_sha256"],
            "package_b_closeout": config["package_b_closeout_sha256"],
            "package_a_gate_audit": config["package_a_gate_audit_sha256"],
            "package_a_closeout": config["package_a_closeout_sha256"],
            "primary_evaluator": config["primary_evaluator_sha256"],
            "secondary_evaluator": config["secondary_evaluator_sha256"],
            "primary_qualification": config["primary_qualification_sha256"],
            "secondary_qualification": config["secondary_qualification_sha256"],
        },
        "dataset_split_hash": config["dataset_split_hash"],
        "git_head": git["head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": config["master_seed"],
        "evaluator_calls": evaluator_calls,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "synthetic_diagnostic_only": True,
        "command": list(sys.argv),
        "runtime_seconds": 0.0,
        "cuda_peak_memory_bytes": 0,
        "dataset_hash": None,
        "state_hash_before": None,
        "state_hash_after": None,
    }


def _qualification_check(path: str, expected_hash: str, evaluator_hash: str, *, secondary: bool) -> None:
    if file_sha256(path) != expected_hash:
        raise ValueError("B2 evaluator qualification hash mismatch")
    value = _json(path)
    if value.get("status") != "passed" or value.get("schema_version") != 2:
        raise ValueError("B2 evaluator qualification is not passed")
    if value.get("context", {}).get("dataset_split_manifest_hash") != "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912":
        raise ValueError("B2 evaluator qualification split mismatch")
    inputs = value.get("input_hashes", {})
    bound = {inputs.get("evaluator_checkpoint"), value.get("context", {}).get("evaluator_checkpoint_hash")}
    if secondary:
        bound.update({inputs.get("secondary_evaluator"), inputs.get("smooth_evaluator")})
    if evaluator_hash not in bound:
        raise ValueError("B2 evaluator qualification checkpoint mismatch")


def _resolve_shard(manifest_path: Path, value: str) -> Path:
    direct = Path(value)
    if direct.is_file():
        return direct
    relative = manifest_path.parent / direct
    if relative.is_file():
        return relative
    raise FileNotFoundError(f"B2 attribution shard is missing: {value}")


def _sample_seed(config: dict[str, Any], bank: int, label: int, index: int) -> int:
    raw = json.dumps(
        [config["master_seed"], bank, label, index],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _validate_artifact(path: str, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _json(manifest_path)
    if manifest.get("schema") != f"{SCHEMA_PREFIX}_artifacts.v1" or manifest.get("status") != "teacher_failure_attribution_artifact":
        raise ValueError("B2 artifact is not an attribution artifact")
    if manifest.get("git_head") is None or manifest.get("dirty_worktree"):
        raise ValueError("B2 artifact lineage state is invalid")
    if manifest.get("config") != config:
        raise ValueError("B2 artifact config binding mismatch")
    if manifest.get("package_b_manifest_sha256") != config["package_b_manifest_sha256"]:
        raise ValueError("B2 artifact Package B binding mismatch")
    if manifest.get("checkpoint_hashes") != {"first_stage": config["first_stage_checkpoint_sha256"], "diffusion": config["diffusion_checkpoint_sha256"]}:
        raise ValueError("B2 artifact checkpoint binding mismatch")
    if manifest.get("dataset_split_hash") != config["dataset_split_hash"] or manifest.get("bank_count") != 2 or manifest.get("samples_per_class") != 250:
        raise ValueError("B2 artifact sampling contract mismatch")
    if manifest.get("latent_shape") != config["latent_shape"] or manifest.get("image_shape") != [1, 200, 200]:
        raise ValueError("B2 artifact shape contract mismatch")
    if (
        manifest.get("evaluator_calls") != 0
        or manifest.get("optimizer_created") is not False
        or manifest.get("checkpoint_created") is not False
        or manifest.get("candidate_export") is not False
    ):
        raise ValueError("B2 artifact contains a forbidden update or evaluator call")
    if manifest.get("state_hash_before") != manifest.get("state_hash_after"):
        raise ValueError("B2 model state changed during artifact generation")
    state_path = manifest_path.with_name(f"{manifest_path.stem}.state.json")
    if not state_path.is_file() or file_sha256(state_path) != manifest.get("generation_state_sha256"):
        raise ValueError("B2 generation state hash is missing or changed")
    state = _json(state_path)
    if (
        state.get("schema") != f"{SCHEMA_PREFIX}_generation_state.v1"
        or state.get("git_head") != manifest.get("git_head")
        or state.get("config_sha256") != manifest.get("config_sha256")
        or state.get("complete") is not True
    ):
        raise ValueError("B2 generation state is incomplete or mismatched")
    arms = manifest.get("arms", {})
    v_arm = arms.get("V", {})
    shards = v_arm.get("shards")
    expected_count = 2 * 10 * ((250 + 16 - 1) // 16)
    if not isinstance(shards, list) or len(shards) != expected_count:
        raise ValueError("B2 reconstruction shard count is invalid")
    seen: set[tuple[int, int, int]] = set()
    for record in shards:
        if not isinstance(record, dict):
            raise ValueError("B2 shard record is invalid")
        key = (int(record["bank"]), int(record["label"]), int(record["start"]))
        end = int(record["end"])
        if not 0 <= key[0] < 2 or not 0 <= key[1] < 10 or key in seen or key[2] < 0 or end <= key[2] or end > 250:
            raise ValueError("B2 shard ranges overlap or are invalid")
        shard_path = _resolve_shard(manifest_path, str(record["path"]))
        if file_sha256(shard_path) != record.get("sha256"):
            raise ValueError(f"B2 shard hash changed: {shard_path}")
        payload = safe_torch_load(shard_path, expected_sha256=str(record["sha256"]))
        if payload.get("schema") != f"{SCHEMA_PREFIX}_shard.v1" or payload.get("bank") != key[0] or payload.get("label") != key[1] or payload.get("start") != key[2] or payload.get("end") != end:
            raise ValueError("B2 shard metadata mismatch")
        count = end - key[2]
        reconstruction = payload.get("reconstruction_images")
        mu = payload.get("encoded_mu")
        latents = payload.get("diffusion_latents")
        labels = payload.get("labels")
        seeds = payload.get("seeds")
        if not torch.is_tensor(reconstruction) or tuple(reconstruction.shape) != (count, 1, 200, 200):
            raise ValueError("B2 reconstruction shape mismatch")
        if not torch.is_tensor(mu) or tuple(mu.shape) != (count, 1, 25, 25) or not torch.is_tensor(latents) or tuple(latents.shape) != (count, 1, 25, 25):
            raise ValueError("B2 latent shape mismatch")
        expected_seeds = [_sample_seed(config, key[0], key[1], index) for index in range(key[2], end)]
        if not torch.is_tensor(labels) or labels.tolist() != [key[1]] * count or seeds != expected_seeds:
            raise ValueError("B2 shard labels or seeds mismatch")
        if not torch.isfinite(reconstruction).all() or not torch.isfinite(mu).all() or not torch.isfinite(latents).all():
            raise ValueError("B2 shard contains non-finite values")
        seen.add(key)
    expected_starts = list(range(0, 250, 16))
    for bank in range(2):
        for label in range(10):
            starts = sorted(key[2] for key in seen if key[:2] == (bank, label))
            if starts != expected_starts:
                raise ValueError("B2 shard coverage is incomplete")
    return manifest


def _validate_lineage(config: dict[str, Any], artifact_path: str, dataset_root: str) -> dict[str, Any]:
    artifact = _validate_artifact(artifact_path, config)
    package_b_config = _package_b_config(config["package_b_config"])
    if file_sha256(config["package_b_config"]) != config["package_b_config_sha256"]:
        raise ValueError("B2 Package B config hash mismatch")
    package_b_manifest = _validate_package_b_manifest(config["package_b_manifest"], package_b_config)
    if file_sha256(config["package_b_manifest"]) != config["package_b_manifest_sha256"]:
        raise ValueError("B2 Package B manifest hash mismatch")
    for key in ("package_b_audit", "package_b_closeout", "package_a_gate_audit", "package_a_closeout"):
        if file_sha256(config[key]) != config[f"{key}_sha256"]:
            raise ValueError(f"B2 {key} hash mismatch")
    package_b_audit = _json(config["package_b_audit"])
    package_b_closeout = _json(config["package_b_closeout"])
    package_a_gate = _json(config["package_a_gate_audit"])
    package_a_closeout = _json(config["package_a_closeout"])
    if package_b_audit.get("schema") != "teacher_distribution_admission_audit.v1" or package_b_audit.get("status") != "valid" or package_b_audit.get("classification") != "teacher_distribution_failed":
        raise ValueError("B2 requires the locked valid Package B failure audit")
    if package_b_closeout.get("schema") != "teacher_distribution_admission_closeout.v1" or package_b_closeout.get("classification") != "teacher_distribution_failed":
        raise ValueError("B2 requires the locked Package B failure closeout")
    if package_a_gate.get("schema") != "electronic_distribution_recovery_gate_audit.v1" or package_a_gate.get("classification") != "gate_feasibility_supported" or not package_a_gate.get("passed"):
        raise ValueError("B2 requires passed Package A H-GFA feasibility")
    if package_a_closeout.get("schema") != "electronic_distribution_recovery_closeout.v1" or package_a_closeout.get("status") != "valid":
        raise ValueError("B2 Package A closeout is invalid")
    _qualification_check(config["primary_qualification"], config["primary_qualification_sha256"], config["primary_evaluator_sha256"], secondary=False)
    _qualification_check(config["secondary_qualification"], config["secondary_qualification_sha256"], config["secondary_evaluator_sha256"], secondary=True)
    data = _dataset(dataset_root)
    if data["split_hash"] != config["dataset_split_hash"]:
        raise ValueError("B2 dataset split is not the locked split")
    real_a, real_b = _partition(
        data["validation_images"],
        data["validation_labels"],
        seed=config["partition_seed"],
        samples_per_class=config["samples_per_class"],
    )
    _validate_source_binding(artifact_path, artifact, (real_a, real_b))
    return {
        "artifact": artifact,
        "artifact_sha256": file_sha256(artifact_path),
        "package_b_manifest": package_b_manifest,
        "package_b_audit": package_b_audit,
        "dataset_hash": data["dataset_hash"],
        "split_hash": data["split_hash"],
    }


def _validate_source_binding(
    manifest_path: str,
    manifest: dict[str, Any],
    banks: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for row in manifest["arms"]["V"]["shards"]:
        bank = int(row["bank"])
        label = int(row["label"])
        start = int(row["start"])
        end = int(row["end"])
        payload = safe_torch_load(
            _resolve_shard(Path(manifest_path), row["path"]),
            expected_sha256=row["sha256"],
        )
        source = banks[bank][0][label * 250 + start : label * 250 + end]
        if payload.get("source_images_hash") != tensor_sha256(source):
            raise ValueError(f"B2 source bank binding mismatch: {bank}/{label}/{start}")


def _load_attribution_bank(manifest_path: str, manifest: dict[str, Any], bank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = sorted((row for row in manifest["arms"]["V"]["shards"] if row["bank"] == bank), key=lambda row: (row["label"], row["start"]))
    values = []
    for row in rows:
        path = _resolve_shard(Path(manifest_path), row["path"])
        payload = safe_torch_load(path, expected_sha256=row["sha256"])
        values.append((payload["reconstruction_images"].float(), payload["encoded_mu"].float(), payload["diffusion_latents"].float(), payload["labels"].long()))
    return tuple(torch.cat(items, dim=0) for items in zip(*values, strict=True))


def _effective_rank(value: torch.Tensor) -> float:
    centered = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square() / max(value.shape[0] - 1, 1)
    total = eigenvalues.sum()
    return float((total.square() / eigenvalues.square().sum().clamp_min(1.0e-12)).item())


def _nearest_distribution(values: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    distances = torch.cdist(values.float(), reference.float()).amin(dim=1)
    return {"p05": float(torch.quantile(distances, 0.05).item()), "median": float(torch.quantile(distances, 0.50).item()), "p95": float(torch.quantile(distances, 0.95).item())}


def _latent_statistics(real_mu: torch.Tensor, ddpm_latents: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    rows = []
    for label in range(CLASS_COUNT):
        real_class = real_mu[labels == label]
        ddpm_class = ddpm_latents[labels == label]
        real_flat = real_class.flatten(1)
        ddpm_flat = ddpm_class.flatten(1)
        mean_delta = ddpm_flat.mean(0) - real_flat.mean(0)
        cov_real = torch.cov(real_flat.T)
        cov_ddpm = torch.cov(ddpm_flat.T)
        scales = (25, 13, 7)
        multiscale = []
        for size in scales:
            real_pool = F.adaptive_avg_pool2d(real_class, (size, size)).flatten(1).mean(0)
            ddpm_pool = F.adaptive_avg_pool2d(ddpm_class, (size, size)).flatten(1).mean(0)
            multiscale.append(float(torch.linalg.vector_norm(ddpm_pool - real_pool).item()))
        nearest = _nearest_distribution(ddpm_flat, real_flat)
        rows.append({
            "label": label,
            "real_mu_mean_rms": float(real_flat.mean(0).square().mean().sqrt().item()),
            "ddpm_latent_mean_rms": float(ddpm_flat.mean(0).square().mean().sqrt().item()),
            "mean_delta_l2": float(torch.linalg.vector_norm(mean_delta).item()),
            "real_covariance_trace": float(torch.trace(cov_real).item()),
            "ddpm_covariance_trace": float(torch.trace(cov_ddpm).item()),
            "real_covariance_frobenius": float(torch.linalg.matrix_norm(cov_real).item()),
            "ddpm_covariance_frobenius": float(torch.linalg.matrix_norm(cov_ddpm).item()),
            "real_effective_rank": _effective_rank(real_class),
            "ddpm_effective_rank": _effective_rank(ddpm_class),
            "nearest_distance": nearest,
            "multiscale_mean_distance": {f"pool_{size}": value for size, value in zip(scales, multiscale, strict=True)},
        })
    return {"space": "normalized_latent_mu_vs_ddpm_latent", "evaluator_calls": 0, "per_class": rows}


def _classification_gate(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    return _gfa_gate(metrics, config)


def _classify(
    *,
    real_positive_control_passed: bool,
    label_shift_rejected: bool,
    reconstruction_passed: bool,
    diffusion_passed: bool,
    package_b_replay_consistent: bool,
) -> tuple[str, str, list[str], list[str]]:
    if not real_positive_control_passed:
        return "invalid_harness", "real_positive_control", [], ["real_positive_control_failed"]
    if not label_shift_rejected:
        return "invalid_harness", "label_shift_negative_control", [], ["label_shift_negative_control_passed"]
    if not reconstruction_passed:
        return "teacher_first_stage_distribution_failed", "first_stage_reconstruction_distribution_gate", ["redesign_first_stage_teacher"], ["first_stage_reconstruction_distribution_failed"]
    if not diffusion_passed:
        return "teacher_diffusion_distribution_failed", "diffusion_distribution_gate", ["redesign_diffusion_teacher"], ["diffusion_distribution_failed"]
    if not package_b_replay_consistent:
        return "teacher_composition_harness_invalid", "package_b_replay_binding", [], ["package_b_replay_mismatch"]
    return "teacher_composition_harness_invalid", "package_b_replay_failure", [], ["package_b_replay_failed"]


def _audit(config: dict[str, Any], artifact_path: str, lineage: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B2 formal attribution requires CUDA")
    device = torch.device(args.device)
    data = _dataset(args.dataset_root)
    real_a, real_b = _partition(data["validation_images"], data["validation_labels"], seed=config["partition_seed"], samples_per_class=config["samples_per_class"])
    artifact = lineage["artifact"]
    package_b_manifest = lineage["package_b_manifest"]
    arm_data: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {"R": [], "V": [], "D": []}
    latent_rows = []
    for bank, (candidate_real, reference_real) in enumerate(((real_a, real_b), (real_b, real_a))):
        reconstruction, mu, ddpm_latent, labels = _load_attribution_bank(artifact_path, artifact, bank)
        teacher_images, teacher_labels = _load_bank(config["package_b_manifest"], package_b_manifest, bank, config)
        if labels.tolist() != candidate_real[1].tolist() or teacher_labels.tolist() != labels.tolist():
            raise ValueError("B2 bank label/order binding mismatch")
        arm_data["R"].append((candidate_real[0], candidate_real[1]))
        arm_data["V"].append((reconstruction, labels))
        arm_data["D"].append((teacher_images, teacher_labels))
        latent_rows.append((mu / float(artifact["latent_scale"]), ddpm_latent, labels))
    # All R/V/D tensors are loaded and hash-bound before the first evaluator call.
    primary, secondary = _load_evaluators(args, device)
    train_images, train_labels = data["train_images"].to(device), data["train_labels"].to(device)
    counter = Counter()
    train_primary, _ = _eval_chunks(primary, train_images, counter)
    train_secondary, _ = _eval_chunks(secondary, train_images, counter)
    results: dict[str, Any] = {"arms": {}, "evaluator_calls": int(counter["evaluator_forward"]), "latent_statistics_evaluator_calls": 0}
    start = time.monotonic()
    for arm in ("R", "V", "D"):
        rows = []
        for bank, reference in enumerate((real_b, real_a)):
            candidate, labels = arm_data[arm][bank]
            candidate_view = _teacher_view(candidate) if arm != "R" else candidate
            metrics, _ = _pair_metrics(
                primary,
                secondary,
                candidate_view.to(device),
                reference[0].to(device),
                labels.to(device),
                train_images,
                train_labels,
                train_primary,
                train_secondary,
            )
            metrics["bank"] = bank
            metrics["arm"] = arm
            metrics["passed"] = _classification_gate(metrics, config)
            rows.append(metrics)
        results["arms"][arm] = {"banks": rows, "passed_both_banks": all(row["passed"] for row in rows)}
    shifted = {}
    for arm in ("R", "V", "D"):
        predictions, shifted_labels = [], []
        for bank in range(2):
            candidate, labels = arm_data[arm][bank]
            views = _teacher_view(candidate) if arm != "R" else candidate
            _, logits = _eval_chunks(primary, views.to(device), counter)
            predictions.append(logits.argmax(1))
            shifted_labels.append(((labels + 1) % CLASS_COUNT).to(device))
        negative = _classification_metrics(torch.cat(predictions), torch.cat(shifted_labels))
        negative["rejected"] = not (negative["accuracy"] >= config["primary_accuracy_minimum"] and negative["minimum_recall"] >= config["minimum_recall_minimum"])
        shifted[arm] = negative
    results["negative_controls"] = shifted
    results["r_positive_control_passed"] = results["arms"]["R"]["passed_both_banks"]
    results["label_shift_rejected"] = all(row["rejected"] for row in shifted.values())
    replay_consistent = all(
        results["arms"]["D"]["banks"][bank]["candidate_hash"]
        == lineage["package_b_audit"]["banks"][bank]["candidate_hash"]
        for bank in range(2)
    )
    classification, first_failed, permitted, findings = _classify(
        real_positive_control_passed=results["r_positive_control_passed"],
        label_shift_rejected=results["label_shift_rejected"],
        reconstruction_passed=results["arms"]["V"]["passed_both_banks"],
        diffusion_passed=results["arms"]["D"]["passed_both_banks"],
        package_b_replay_consistent=replay_consistent,
    )
    results["package_b_replay_consistent"] = replay_consistent
    latent_stats = []
    for mu, ddpm_latent, labels in latent_rows:
        latent_stats.append(_latent_statistics(mu, ddpm_latent, labels))
    results.update({
        "classification": classification,
        "first_failed_gate": first_failed,
        "findings": findings,
        "permitted_next_actions": permitted,
        "latent_statistics": {"banks": latent_stats, "evaluator_calls": 0},
        "evaluator_calls": int(counter["evaluator_forward"]),
        "elapsed_seconds": time.monotonic() - start,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.monotonic() - start,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "state_hash_before": {"first_stage": artifact["state_hash_before"]["first_stage"], "diffusion": artifact["state_hash_before"]["diffusion"]},
        "state_hash_after": {"first_stage": artifact["state_hash_after"]["first_stage"], "diffusion": artifact["state_hash_after"]["diffusion"]},
        "optimizer_created": False,
        "checkpoint_created": False,
    })
    results["passed"] = classification != "invalid_harness"
    return results


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    _args_match_config(args, config)
    git = _require_clean()
    action = args.research_action
    if action == "validate":
        lineage = _validate_lineage(config, args.attribution_artifact, args.dataset_root)
        payload = _base("validation", config, args.config, git, status="valid", passed=True, classification="teacher_attribution_lineage_validated", first_failed_gate=None, permitted=["audit"])
        payload.update({"artifact_sha256": lineage["artifact_sha256"], "dataset_hash": lineage["dataset_hash"], "package_b_manifest_sha256": config["package_b_manifest_sha256"]})
        atomic_write_json(args.output, payload)
        return 0
    if action == "audit":
        validation = _json(args.validation_evidence)
        if validation.get("schema") != f"{SCHEMA_PREFIX}_validation.v1" or not validation.get("passed"):
            raise ValueError("B2 audit requires passed validation evidence")
        lineage = _validate_lineage(config, args.attribution_artifact, args.dataset_root)
        result = _audit(config, args.attribution_artifact, lineage, args)
        payload = _base("audit", config, args.config, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=result["permitted_next_actions"], evaluator_calls=result["evaluator_calls"])
        payload.update(result)
        payload["artifact_sha256"] = lineage["artifact_sha256"]
        payload["dataset_hash"] = lineage["dataset_hash"]
        payload["validation_evidence_sha256"] = file_sha256(args.validation_evidence)
        atomic_write_json(args.output, payload)
        return 0
    evidence_path = args.audit_evidence if action == "report" else args.report_evidence
    evidence = _json(evidence_path)
    expected_schema = f"{SCHEMA_PREFIX}_audit.v1" if action == "report" else f"{SCHEMA_PREFIX}_report.v1"
    if evidence.get("schema") != expected_schema or evidence.get("status") != "valid":
        raise ValueError(f"{action} requires valid upstream evidence")
    if (
        evidence.get("git_head") != git["head"]
        or evidence.get("dirty_worktree") is not False
        or evidence.get("config_sha256") != file_sha256(args.config)
    ):
        raise ValueError(f"{action} requires upstream evidence bound to current HEAD/config")
    if action == "report":
        payload = {
            "schema": f"{SCHEMA_PREFIX}_report.v1",
            "status": "valid",
            "classification": evidence["classification"],
            "first_failed_gate": evidence.get("first_failed_gate"),
            "findings": evidence.get("findings", []),
            "permitted_next_actions": evidence.get("permitted_next_actions", []),
            "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
            "audit_evidence_sha256": file_sha256(evidence_path),
            "git_head": git["head"],
            "dirty_worktree": git["dirty_worktree"],
            "input_artifact_hashes": evidence.get("input_artifact_hashes", {}),
            "dataset_split_hash": config["dataset_split_hash"],
            "dataset_hash": evidence.get("dataset_hash"),
            "config_sha256": evidence["config_sha256"],
            "seed": evidence["seed"],
            "state_hash_before": evidence.get("state_hash_before"),
            "state_hash_after": evidence.get("state_hash_after"),
            "optimizer_created": evidence.get("optimizer_created", False),
            "checkpoint_created": evidence.get("checkpoint_created", False),
            "candidate_export": evidence.get("candidate_export", False),
            "synthetic_diagnostic_only": evidence.get("synthetic_diagnostic_only", True),
            "evaluator_calls": evidence.get("evaluator_calls", 0),
            "source_runtime_seconds": evidence.get("runtime_seconds", 0.0),
            "source_cuda_peak_memory_bytes": evidence.get("cuda_peak_memory_bytes", 0),
            "command": list(sys.argv),
            "runtime_seconds": 0.0,
            "cuda_peak_memory_bytes": evidence.get("cuda_peak_memory_bytes", 0),
        }
        atomic_write_json(args.output, payload)
        return 0
    classification = evidence["classification"]
    permitted = evidence.get("permitted_next_actions", [])
    payload = {
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
        "dataset_hash": evidence.get("dataset_hash"),
        "config_sha256": evidence["config_sha256"],
        "seed": evidence["seed"],
        "state_hash_before": evidence.get("state_hash_before"),
        "state_hash_after": evidence.get("state_hash_after"),
        "optimizer_created": evidence.get("optimizer_created", False),
        "checkpoint_created": evidence.get("checkpoint_created", False),
        "candidate_export": evidence.get("candidate_export", False),
        "synthetic_diagnostic_only": evidence.get("synthetic_diagnostic_only", True),
        "input_artifact_hashes": evidence.get("input_artifact_hashes", {}),
        "evaluator_calls": evidence.get("evaluator_calls", 0),
        "source_runtime_seconds": evidence.get("runtime_seconds", 0.0),
        "source_cuda_peak_memory_bytes": evidence.get("cuda_peak_memory_bytes", 0),
        "command": list(sys.argv),
        "runtime_seconds": 0.0,
        "cuda_peak_memory_bytes": evidence.get("cuda_peak_memory_bytes", 0),
        "promotion": "forbidden",
        "teacher_training": "forbidden",
        "student_training": "forbidden",
        "optical_training": "forbidden",
    }
    atomic_write_json(args.output, payload)
    Path(args.output).with_name("teacher_failure_attribution_synthesis.md").write_text(
        "# Teacher failure attribution synthesis\n\n"
        f"- Classification: `{classification}`\n"
        f"- Report evidence: `{file_sha256(evidence_path)}`\n"
        "- Teacher/student/optical training, promotion and candidate export: forbidden.\n",
        encoding="utf-8",
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B2 teacher failure attribution")
    parser.add_argument("research_action", choices=("validate", "audit", "report", "closeout"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--attribution-artifact", required=True)
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
    pairs = {
        "primary_evaluator_checkpoint": "primary_evaluator",
        "secondary_evaluator_checkpoint": "secondary_evaluator",
        "primary_qualification_evidence": "primary_qualification",
        "secondary_qualification_evidence": "secondary_qualification",
    }
    for arg_name, config_key in pairs.items():
        if Path(getattr(args, arg_name)).resolve() != Path(config[config_key]).resolve():
            raise ValueError(f"B2 explicit {arg_name} does not match locked config")


def main(argv: list[str] | None = None) -> int:
    return dispatch(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
