"""Package B2 producer for frozen DDPM--VAE failure attribution.

The producer is deliberately confined to the v1 teacher runtime.  It creates
only first-stage reconstructions and evaluator-independent latent artifacts;
the v2 consumer performs all formal evaluator calls after the artifacts are
fixed and hash-bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from . import teacher_distribution_admission as package_b


PROGRAM = "teacher-failure-attribution"
CONFIG_NAME = "teacher_failure_attribution_v1"
SCHEMA_PREFIX = "teacher_failure_attribution"
CLASS_COUNT = 10
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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def _atomic_torch(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _require_cuda(device: str) -> torch.device:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Package B2 producer stages require CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def _config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("name") != CONFIG_NAME:
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
        "preflight_wall_limit_seconds": 600.0,
        "memory_limit_bytes": 6442450944,
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


def _dataset(dataset_root: str) -> dict[str, torch.Tensor | str]:
    from torchvision.datasets import MNIST

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"local MNIST root does not exist: {root}")
    try:
        dataset = MNIST(root=str(root), train=True, download=False)
    except (RuntimeError, FileNotFoundError) as exc:
        raise FileNotFoundError("local MNIST is required; download=False is enforced") from exc
    images = dataset.data.unsqueeze(1).to(torch.float32).div_(255.0)
    labels = dataset.targets.to(torch.long)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    train_indices, validation_indices = [], []
    for label in range(CLASS_COUNT):
        matches = torch.nonzero(labels == label, as_tuple=False).flatten()
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        validation_indices.append(shuffled[:500])
        train_indices.append(shuffled[500:])
    train_index = torch.cat(train_indices)
    validation_index = torch.cat(validation_indices)
    split_hash = "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912"
    return {
        "all_images": images,
        "all_labels": labels,
        "train_images": images.index_select(0, train_index),
        "train_labels": labels.index_select(0, train_index),
        "validation_images": images.index_select(0, validation_index),
        "validation_labels": labels.index_select(0, validation_index),
        "split_hash": split_hash,
        "dataset_hash": _tensor_sha256(images) + ":" + _tensor_sha256(labels),
    }


def _partition(images: torch.Tensor, labels: torch.Tensor, *, seed: int, samples_per_class: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_images, first_labels, second_images, second_labels = [], [], [], []
    for label in range(CLASS_COUNT):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() != 2 * samples_per_class:
            raise ValueError("Package B2 requires exactly two complete validation banks per class")
        order = indices.index_select(0, torch.randperm(indices.numel(), generator=generator))
        first_images.append(images.index_select(0, order[:samples_per_class]))
        second_images.append(images.index_select(0, order[samples_per_class:]))
        first_labels.append(torch.full((samples_per_class,), label, dtype=torch.long))
        second_labels.append(torch.full((samples_per_class,), label, dtype=torch.long))
    return (
        torch.cat(first_images), torch.cat(first_labels)
    ), (
        torch.cat(second_images), torch.cat(second_labels)
    )


def _sample_seed(config: dict[str, Any], bank: int, label: int, index: int) -> int:
    raw = json.dumps(
        [config["master_seed"], bank, label, index],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _models(lineage: dict[str, Any], device: torch.device):
    return package_b._models(lineage["package_b_lineage"], device)


def _resize_inputs(images: torch.Tensor, image_size: int) -> torch.Tensor:
    return resize(
        images,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _decode_camera(first_stage: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    camera = first_stage.decode_with_intermediates(latent)["camera_plane"]
    if camera.ndim != 4 or tuple(camera.shape[1:]) != (1, 200, 200):
        raise ValueError(f"B2 camera output shape is invalid: {tuple(camera.shape)}")
    if not torch.isfinite(camera).all():
        raise ValueError("B2 camera output is non-finite")
    return camera.detach().cpu().float()


def _validate_package_b_inputs(config: dict[str, Any], package_b_config: dict[str, Any]) -> dict[str, Any]:
    if package_b_config["first_stage_checkpoint_sha256"] != config["first_stage_checkpoint_sha256"] or package_b_config["diffusion_checkpoint_sha256"] != config["diffusion_checkpoint_sha256"]:
        raise ValueError("B2 checkpoint hashes do not match Package B config")
    package_b_paths = {
        "manifest": config["package_b_manifest"],
        "audit": config["package_b_audit"],
        "closeout": config["package_b_closeout"],
        "package_a_gate_audit": config["package_a_gate_audit"],
        "package_a_closeout": config["package_a_closeout"],
    }
    expected = {
        "manifest": config["package_b_manifest_sha256"],
        "audit": config["package_b_audit_sha256"],
        "closeout": config["package_b_closeout_sha256"],
        "package_a_gate_audit": config["package_a_gate_audit_sha256"],
        "package_a_closeout": config["package_a_closeout_sha256"],
    }
    for name, path in package_b_paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"B2 input is missing: {name}: {path}")
        if _file_sha256(path) != expected[name]:
            raise ValueError(f"B2 input hash mismatch: {name}")
    manifest = _json(config["package_b_manifest"])
    audit = _json(config["package_b_audit"])
    closeout = _json(config["package_b_closeout"])
    gate = _json(config["package_a_gate_audit"])
    if manifest.get("schema") != "teacher_distribution_admission_banks.v1" or manifest.get("status") != "teacher_audit_bank":
        raise ValueError("Package B manifest is not a teacher audit bank")
    if audit.get("schema") != "teacher_distribution_admission_audit.v1" or audit.get("status") != "valid":
        raise ValueError("Package B audit is not valid")
    if closeout.get("schema") != "teacher_distribution_admission_closeout.v1" or closeout.get("classification") != "teacher_distribution_failed":
        raise ValueError("Package B closeout is not the locked teacher failure")
    if gate.get("schema") != "electronic_distribution_recovery_gate_audit.v1" or gate.get("classification") != "gate_feasibility_supported" or not gate.get("passed"):
        raise ValueError("B2 requires Package A H-GFA feasibility evidence")
    return {"manifest": manifest, "audit": audit, "closeout": closeout}


def _lineage(config: dict[str, Any], dataset_root: str) -> dict[str, Any]:
    package_b_config = package_b._config(config["package_b_config"])
    inputs = {
        "first_stage_checkpoint": config["first_stage_checkpoint"],
        "diffusion_checkpoint": config["diffusion_checkpoint"],
        "package_b_config": config["package_b_config"],
        "package_b_manifest": config["package_b_manifest"],
        "package_b_audit": config["package_b_audit"],
        "package_b_closeout": config["package_b_closeout"],
        "package_a_gate_audit": config["package_a_gate_audit"],
        "package_a_closeout": config["package_a_closeout"],
        "primary_evaluator": config["primary_evaluator"],
        "secondary_evaluator": config["secondary_evaluator"],
        "primary_qualification": config["primary_qualification"],
        "secondary_qualification": config["secondary_qualification"],
    }
    for role, path in inputs.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"B2 lineage input is missing: {role}: {path}")
    expected = {
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
    }
    hashes = {role: _file_sha256(path) for role, path in inputs.items()}
    for role, digest in expected.items():
        if hashes[role] != digest:
            raise ValueError(f"B2 lineage hash mismatch: {role}")
    package_b_inputs = _validate_package_b_inputs(config, package_b_config)
    package_b_lineage = package_b._lineage(package_b_config)
    data = _dataset(dataset_root)
    if data["split_hash"] != config["dataset_split_hash"]:
        raise ValueError("B2 dataset split is not the qualified split")
    return {
        "hashes": hashes,
        "dataset_hash": f"{data['dataset_hash']}",
        "split_hash": data["split_hash"],
        "package_b_lineage": package_b_lineage,
        "package_b_manifest": package_b_inputs["manifest"],
        "package_b_audit": package_b_inputs["audit"],
        "package_b_closeout": package_b_inputs["closeout"],
        "first_stage_state_hash": package_b_lineage["first_stage_state_hash"],
        "diffusion_state_hash": package_b_lineage["diffusion_state_hash"],
        "latent_scale": package_b_lineage["latent_scale"],
    }


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
        "config_sha256": _file_sha256(config_path),
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
        "state_hash_before": None,
        "state_hash_after": None,
    }


def _expected_package_b_shard(manifest: dict[str, Any], bank: int, label: int, start: int) -> dict[str, Any]:
    rows = [row for row in manifest["shards"] if row["bank"] == bank and row["label"] == label and row["start"] == start]
    if len(rows) != 1:
        raise ValueError(f"Package B shard lookup is not unique: {bank}/{label}/{start}")
    return rows[0]


def _load_package_b_shard(manifest_path: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    if not path.is_file():
        path = Path(manifest_path).parent / path
    payload = package_b._safe_load(path, map_location="cpu")
    if package_b._file_sha256(path) != row["sha256"]:
        raise ValueError(f"Package B shard hash changed: {path}")
    return payload


def _preflight(config: dict[str, Any], lineage: dict[str, Any], dataset_root: str, device_name: str) -> dict[str, Any]:
    device = _require_cuda(device_name)
    data = _dataset(dataset_root)
    first, second = _partition(data["validation_images"], data["validation_labels"], seed=config["partition_seed"], samples_per_class=config["samples_per_class"])
    first_stage, diffusion, scheduler, _, diffusion_payload = _models(lineage, device)
    state_before = {"first_stage": lineage["first_stage_state_hash"], "diffusion": lineage["diffusion_state_hash"]}
    start = time.monotonic()
    recon_views = []
    diffusion_views = []
    replay_exact = True
    package_b_replay_exact = True
    frozen = all(
        not parameter.requires_grad
        for model in (first_stage, diffusion)
        for parameter in model.parameters()
    )
    representative_seeds = []
    first_replay = None
    for bank, bank_data in enumerate((first, second)):
        for label in range(CLASS_COUNT):
            begin, end = 0, config["microbatch_size"]
            labels = torch.full((end - begin,), label, device=device, dtype=torch.long)
            seeds = [_sample_seed(config, bank=bank, label=label, index=index) for index in range(begin, end)]
            representative_seeds.extend(seeds)
            source = bank_data[0][label * config["samples_per_class"] + begin : label * config["samples_per_class"] + end]
            real = _resize_inputs(source, config["image_size"]).to(device)
            with torch.no_grad():
                encoded = first_stage.encode(real, deterministic=True)
                recon = _decode_camera(first_stage, encoded["mu"] * lineage["latent_scale"])
                noise = package_b._noise(seeds, tuple(config["latent_shape"])).to(device)
                d_latent = package_b.sample_latents(
                    diffusion,
                    scheduler,
                    (end - begin, *config["latent_shape"]),
                    device,
                    labels,
                    guidance_scale=config["guidance_scale"],
                    initial_noise=noise,
                    sampling_steps=config["sampling_steps"],
                )
                d_first = _decode_camera(first_stage, d_latent * lineage["latent_scale"])
            expected = _load_package_b_shard(
                config["package_b_manifest"],
                _expected_package_b_shard(lineage["package_b_manifest"], bank, label, begin),
            )["images"].float()
            package_b_replay_exact = package_b_replay_exact and torch.equal(d_first, expected)
            if first_replay is None:
                first_replay = (labels, noise, expected)
            recon_views.append(recon)
            diffusion_views.append(d_first)
            del real, encoded, recon, noise, d_latent, d_first, expected
            torch.cuda.empty_cache()
    if first_replay is None:
        raise RuntimeError("B2 preflight did not create a representative replay")
    replay_labels, replay_noise, replay_expected = first_replay
    with torch.no_grad():
        replay_latent = package_b.sample_latents(
            diffusion,
            scheduler,
            (config["microbatch_size"], *config["latent_shape"]),
            device,
            replay_labels,
            guidance_scale=config["guidance_scale"],
            initial_noise=replay_noise,
            sampling_steps=config["sampling_steps"],
        )
        replay_images = _decode_camera(first_stage, replay_latent * lineage["latent_scale"])
    replay_exact = torch.equal(replay_images, replay_expected)
    del replay_labels, replay_noise, replay_expected, replay_latent, replay_images
    recon = torch.cat(recon_views)
    d_first = torch.cat(diffusion_views)
    state_after = {
        "first_stage": package_b._state_hash(first_stage.state_dict()),
        "diffusion": package_b._state_hash(diffusion.state_dict()),
    }
    finite = bool(torch.isfinite(recon).all().item() and torch.isfinite(d_first).all().item())
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - start
    passed = bool(
        finite
        and frozen
        and state_before == state_after
        and replay_exact
        and package_b_replay_exact
        and peak < config["memory_limit_bytes"]
        and elapsed < config["preflight_wall_limit_seconds"]
    )
    return {
        "passed": passed,
        "classification": "preflight_passed" if passed else "invalid_harness",
        "first_failed_gate": None if passed else "reconstruction_sampler_state_or_resource",
        "finite": finite,
        "frozen": frozen,
        "state_before": state_before,
        "state_after": state_after,
        "resume_exact": replay_exact,
        "package_b_replay_exact": package_b_replay_exact,
        "memory_gate": peak < config["memory_limit_bytes"],
        "peak_cuda_memory_bytes": peak,
        "elapsed_seconds": elapsed,
        "evaluator_calls": 0,
        "optimizer_created": False,
        "checkpoint_created": False,
        "sample_shape": list(recon.shape),
        "representative_banks": [len(first[0]), len(second[0])],
        "seed_hash": hashlib.sha256(json.dumps(representative_seeds).encode()).hexdigest(),
        "dataset_hash": lineage["dataset_hash"],
        "command": list(sys.argv),
        "runtime_seconds": elapsed,
        "cuda_peak_memory_bytes": peak,
    }


def _generate(config: dict[str, Any], config_path: str, lineage: dict[str, Any], dataset_root: str, device_name: str, output: str | Path, preflight: dict[str, Any]) -> dict[str, Any]:
    if not preflight.get("passed"):
        raise ValueError("B2 generation requires passed preflight evidence")
    device = _require_cuda(device_name)
    data = _dataset(dataset_root)
    first, second = _partition(data["validation_images"], data["validation_labels"], seed=config["partition_seed"], samples_per_class=config["samples_per_class"])
    first_stage, diffusion, scheduler, _, diffusion_payload = _models(lineage, device)
    output_path = Path(output)
    shard_dir = output_path.with_name(f"{output_path.stem}_shards")
    state_path = output_path.with_name(f"{output_path.stem}.state.json")
    config_hash = _file_sha256(config_path)
    git_head = _git()["head"]
    state: dict[str, Any] = {"schema": f"{SCHEMA_PREFIX}_generation_state.v1", "git_head": git_head, "config_sha256": config_hash, "shards": {}}
    if state_path.exists():
        existing = _json(state_path)
        if existing.get("git_head") != git_head or existing.get("config_sha256") != config_hash:
            raise ValueError("B2 partial generation state belongs to another HEAD or config")
        state = existing
    state.setdefault("shards", {})
    package_b_manifest = lineage["package_b_manifest"]
    records: list[dict[str, Any]] = []
    peak = int(state.get("peak_cuda_memory_bytes", 0))
    start_time = time.monotonic()
    for bank, bank_data in enumerate((first, second)):
        for label in range(CLASS_COUNT):
            indices = torch.nonzero(bank_data[1] == label, as_tuple=False).flatten()
            for begin in range(0, config["samples_per_class"], config["microbatch_size"]):
                end = min(begin + config["microbatch_size"], config["samples_per_class"])
                key = f"bank_{bank}/class_{label}/chunk_{begin:04d}"
                name = f"bank_{bank}_class_{label}_chunk_{begin:04d}.pt"
                shard_path = shard_dir / name
                if key in state["shards"]:
                    record = state["shards"][key]
                    if record.get("path") != str(shard_path) or not shard_path.is_file() or _file_sha256(shard_path) != record.get("sha256"):
                        raise ValueError(f"B2 generation shard is stale or tampered: {key}")
                    records.append(record)
                    continue
                source = bank_data[0].index_select(0, indices[begin:end])
                labels = torch.full((end - begin,), label, dtype=torch.long, device=device)
                seeds = [_sample_seed(config, bank, label, index) for index in range(begin, end)]
                real_input = _resize_inputs(source, config["image_size"]).to(device)
                with torch.no_grad():
                    encoded = first_stage.encode(real_input, deterministic=True)
                    mu = encoded["mu"]
                    v_camera = _decode_camera(first_stage, mu * lineage["latent_scale"])
                    d_latent = package_b._noise(seeds, tuple(config["latent_shape"])).to(device)
                    d_latent = package_b.sample_latents(
                        diffusion,
                        scheduler,
                        (len(seeds), *config["latent_shape"]),
                        device,
                        labels,
                        guidance_scale=config["guidance_scale"],
                        initial_noise=d_latent,
                        sampling_steps=config["sampling_steps"],
                    )
                    d_camera = _decode_camera(first_stage, d_latent * lineage["latent_scale"])
                package_b_row = _expected_package_b_shard(package_b_manifest, bank, label, begin)
                package_b_payload = _load_package_b_shard(config["package_b_manifest"], package_b_row)
                expected_d = package_b_payload["images"].float()
                if not torch.equal(d_camera, expected_d):
                    raise ValueError(f"B2 DDPM replay differs from Package B bank: {key}")
                payload = {
                    "schema": f"{SCHEMA_PREFIX}_shard.v1",
                    "bank": bank,
                    "label": label,
                    "start": begin,
                    "end": end,
                    "source_images_hash": _tensor_sha256(source),
                    "reconstruction_images": v_camera,
                    "encoded_mu": mu.detach().cpu().float(),
                    "diffusion_latents": d_latent.detach().cpu().float(),
                    "labels": labels.detach().cpu().long(),
                    "seeds": seeds,
                }
                _atomic_torch(shard_path, payload)
                record = {
                    "key": key,
                    "path": str(shard_path),
                    "sha256": _file_sha256(shard_path),
                    "bank": bank,
                    "label": label,
                    "start": begin,
                    "end": end,
                }
                state["shards"][key] = record
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                state["peak_cuda_memory_bytes"] = peak
                _atomic_json(state_path, state)
                records.append(record)
                print(f"[teacher-failure-attribution] generated bank={bank} class={label} {end}/{config['samples_per_class']}")
                del real_input, encoded, mu, d_latent, d_camera, v_camera
                torch.cuda.empty_cache()
    state["complete"] = True
    _atomic_json(state_path, state)
    final = {
        "schema": f"{SCHEMA_PREFIX}_artifacts.v1",
        "status": "teacher_failure_attribution_artifact",
        "classification": "generated_for_attribution",
        "first_failed_gate": None,
        "permitted_next_actions": ["validate"],
        "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
        "config_sha256": config_hash,
        "git_head": git_head,
        "dirty_worktree": _git()["dirty_worktree"],
        "config": config,
        "checkpoint_hashes": {"first_stage": config["first_stage_checkpoint_sha256"], "diffusion": config["diffusion_checkpoint_sha256"]},
        "package_b_manifest_sha256": config["package_b_manifest_sha256"],
        "dataset_split_hash": config["dataset_split_hash"],
        "bank_count": config["bank_count"],
        "samples_per_class": config["samples_per_class"],
        "image_shape": [1, config["image_size"], config["image_size"]],
        "latent_shape": config["latent_shape"],
        "latent_scale": lineage["latent_scale"],
        "partition": {"seed": config["partition_seed"], "bank_a_hash": _tensor_sha256(first[0]), "bank_b_hash": _tensor_sha256(second[0]), "disjoint": True},
        "arms": {"R": {"kind": "real_validation", "bank_count": config["bank_count"]}, "V": {"kind": "first_stage_reconstruction", "shards": sorted(records, key=lambda row: (row["bank"], row["label"], row["start"]))}, "D": {"kind": "package_b_ddpm_replay", "manifest_sha256": config["package_b_manifest_sha256"]}},
        "generation_state_sha256": _file_sha256(state_path),
        "state_hash_before": {"first_stage": lineage["first_stage_state_hash"], "diffusion": lineage["diffusion_state_hash"]},
        "state_hash_after": {"first_stage": package_b._state_hash(first_stage.state_dict()), "diffusion": package_b._state_hash(diffusion.state_dict())},
        "peak_cuda_memory_bytes": peak,
        "elapsed_seconds": time.monotonic() - start_time,
        "evaluator_calls": 0,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "synthetic_diagnostic_only": True,
        "dataset_hash": lineage["dataset_hash"],
        "cuda_peak_memory_bytes": peak,
        "runtime_seconds": time.monotonic() - start_time,
        "command": list(sys.argv),
    }
    if final["state_hash_before"] != final["state_hash_after"]:
        raise RuntimeError("B2 teacher state changed during artifact generation")
    _atomic_json(output_path, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B2 frozen teacher failure attribution producer")
    parser.add_argument("action", choices=("lineage", "preflight", "generate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lineage-evidence")
    parser.add_argument("--preflight-evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config(args.config)
    git = _require_clean()
    if args.action == "lineage":
        lineage = _lineage(config, args.dataset_root)
        payload = _base("lineage", config, args.config, git, status="valid", passed=True, classification="teacher_attribution_lineage_validated", first_failed_gate=None, permitted=["preflight"])
        payload["lineage"] = lineage
        _atomic_json(args.output, payload)
        return 0
    if not args.lineage_evidence:
        raise ValueError(f"{args.action} requires --lineage-evidence")
    lineage_evidence = _json(args.lineage_evidence)
    if lineage_evidence.get("schema") != f"{SCHEMA_PREFIX}_lineage.v1" or not lineage_evidence.get("passed") or lineage_evidence.get("config_sha256") != _file_sha256(args.config) or lineage_evidence.get("git_head") != git["head"]:
        raise ValueError("B2 stage requires passed lineage evidence bound to current HEAD/config")
    lineage = _lineage(config, args.dataset_root)
    if args.action == "preflight":
        result = _preflight(config, lineage, args.dataset_root, args.device)
        payload = _base("preflight", config, args.config, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=["generate"] if result["passed"] else [])
        payload.update(result)
        payload["lineage_evidence_sha256"] = _file_sha256(args.lineage_evidence)
        _atomic_json(args.output, payload)
        return 0
    if not args.preflight_evidence:
        raise ValueError("generate requires --preflight-evidence")
    preflight = _json(args.preflight_evidence)
    if preflight.get("schema") != f"{SCHEMA_PREFIX}_preflight.v1" or not preflight.get("passed") or preflight.get("config_sha256") != _file_sha256(args.config) or preflight.get("git_head") != git["head"]:
        raise ValueError("B2 generation requires passed preflight evidence bound to current HEAD/config")
    result = _generate(config, args.config, lineage, args.dataset_root, args.device, args.output, preflight)
    result["lineage_evidence_sha256"] = _file_sha256(args.lineage_evidence)
    result["preflight_evidence_sha256"] = _file_sha256(args.preflight_evidence)
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
