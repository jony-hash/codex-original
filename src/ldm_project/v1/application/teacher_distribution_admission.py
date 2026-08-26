"""Package B producer: frozen DDPM--VAE teacher audit banks.

This module deliberately owns only the v1 teacher boundary.  It produces
hash-bound, raw camera-plane shards for the v2 admission consumer; it never
imports v2 evaluators, creates an optimizer, or writes a model checkpoint.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..teacher.diffusion import (
    DIFFUSION_MODEL_VERSION,
    LatentDiffusionUNet,
    LinearNoiseScheduler,
    load_diffusion_checkpoint,
)
from ..teacher.diffusion_sampling import sample_latents
from ..teacher.first_stage import (
    FIRST_STAGE_MODEL_VERSION,
    OpticalPathVAE,
    load_first_stage_checkpoint,
)


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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _atomic_torch(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _safe_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=False)


def _git() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"head": head, "dirty_worktree": bool(status), "status": status}


def _require_clean() -> dict[str, Any]:
    git = _git()
    if git["dirty_worktree"]:
        raise RuntimeError("Package B scientific stages require a clean worktree")
    return git


def _require_cuda(device: str) -> torch.device:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Package B sampler stages require CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def _config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("name") != CONFIG_NAME:
        raise ValueError("Package B config name is invalid")
    required = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": 20260826,
        "program": PROGRAM,
        "first_stage_checkpoint_sha256": "e553bfab77e405265482a25ab9b569f0e6cc4e26940fd332985b77156d8bafdd",
        "diffusion_checkpoint_sha256": "60f7592b2bc803779af7aad809bdd90b280d183ef6e029586698c11bd3d68cd0",
        "dataset_split_hash": "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912",
        "teacher_model_version": FIRST_STAGE_MODEL_VERSION,
        "diffusion_model_version": DIFFUSION_MODEL_VERSION,
        "latent_source": "mu",
        "latent_shape": [1, 25, 25],
        "image_size": 200,
        "class_count": 10,
        "bank_count": 2,
        "samples_per_class": 250,
        "microbatch_size": 16,
        "master_seed": 32452843,
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
        "memory_limit_bytes": 6442450944,
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
    for key in (
        "first_stage_checkpoint", "diffusion_checkpoint", "first_stage_run_meta",
        "first_stage_train_config", "first_stage_physics_config", "diffusion_run_meta",
        "diffusion_train_config", "diffusion_physics_config", "acceptance_manifest",
        "acceptance_first_stage_run", "acceptance_diffusion_run",
    ):
        if key not in value or not isinstance(value[key], str):
            raise ValueError(f"Package B config field {key!r} is missing")
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"Package B config field {key!r} is not registered")
    return value


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _asdict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _asdict(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _asdict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_asdict(item) for item in value]
    return value


def _require_path(config: dict[str, Any], key: str) -> Path:
    path = Path(str(config[key]))
    if not path.is_file():
        raise FileNotFoundError(f"Package B input is missing: {key}: {path}")
    return path


def _lineage(config: dict[str, Any]) -> dict[str, Any]:
    file_keys = (
        "first_stage_checkpoint", "diffusion_checkpoint", "first_stage_run_meta",
        "first_stage_train_config", "first_stage_physics_config", "diffusion_run_meta",
        "diffusion_train_config", "diffusion_physics_config", "acceptance_manifest",
    )
    paths = {key: _require_path(config, key) for key in file_keys}
    expected_hashes = {
        key: str(config[f"{key}_sha256"])
        for key in (
            "first_stage_checkpoint", "diffusion_checkpoint", "first_stage_run_meta",
            "first_stage_train_config", "first_stage_physics_config", "diffusion_run_meta",
            "diffusion_train_config", "diffusion_physics_config", "acceptance_manifest",
        )
    }
    hashes = {key: _file_sha256(path) for key, path in paths.items()}
    for key, expected in expected_hashes.items():
        if hashes[key] != expected:
            raise ValueError(f"Package B {key} hash does not match locked config")

    first = _safe_load(paths["first_stage_checkpoint"])
    diffusion = _safe_load(paths["diffusion_checkpoint"])
    if not isinstance(first, dict) or not isinstance(diffusion, dict):
        raise ValueError("teacher checkpoint payload must be a mapping")
    if first.get("model_version") != config["teacher_model_version"] or first.get("stage") != "first_stage":
        raise ValueError("first-stage checkpoint metadata is invalid")
    if diffusion.get("model_version") != config["diffusion_model_version"] or diffusion.get("stage") != "diffusion":
        raise ValueError("diffusion checkpoint metadata is invalid")
    first_state = first.get("model_state_dict")
    diffusion_state = diffusion.get("model_state_dict")
    if not isinstance(first_state, dict) or not isinstance(diffusion_state, dict):
        raise ValueError("teacher checkpoint state is missing")
    if any(key in first or key in diffusion for key in ("optimizer", "optimizer_state_dict")):
        raise ValueError("teacher audit input unexpectedly contains optimizer state")
    if diffusion.get("first_stage_model_version") != first.get("model_version"):
        raise ValueError("diffusion is not bound to the first-stage model version")
    first_config = _asdict(first.get("train_config"))
    diffusion_config = _asdict(diffusion.get("train_config"))
    first_physics = _asdict(first.get("physics_config"))
    diffusion_physics = _asdict(diffusion.get("physics_config"))
    if not isinstance(first_config, dict) or not isinstance(diffusion_config, dict):
        raise ValueError("teacher checkpoint train config metadata is missing")
    if first_physics != diffusion_physics:
        raise ValueError("first-stage and diffusion physics metadata differ")
    first_stage = first_config.get("first_stage", {})
    diffusion_section = diffusion_config.get("diffusion", {})
    required_first_stage = {
        "image_size": 200, "latent_channels": 1, "latent_position": "post_encoder_pre_ols",
        "readout_plane": "camera", "encoder_fusion_mode": "optical_only",
        "optical_encoder_input_mode": "pf2_complex_field", "optical_encoder_arch": "linear_metasurface",
        "metasurface_decoder_arch": "linear_metasurface", "linear_metasurface_grid_size": 40,
        "ols_kind": "gelu",
    }
    for key, expected in required_first_stage.items():
        if first_stage.get(key) != expected:
            raise ValueError(f"first-stage metadata mismatch: {key}")
    required_diffusion = {
        "timesteps": 1000, "beta_start": 1.0e-4, "beta_end": 0.02,
        "base_channels": 64, "channel_multipliers": [1, 2, 4], "num_res_blocks": 2,
        "attention_levels": [1, 2], "num_heads": 4, "context_dim": 128,
        "num_classes": 10, "cond_dropout_prob": 0.1, "latent_source": "mu",
    }
    for key, expected in required_diffusion.items():
        if diffusion_section.get(key) != expected:
            raise ValueError(f"diffusion metadata mismatch: {key}")
    if diffusion.get("latent_source") != config["latent_source"]:
        raise ValueError("diffusion latent source mismatch")
    latent_scale = diffusion.get("latent_scale")
    if not torch.is_tensor(latent_scale) or tuple(latent_scale.shape) != () or not torch.isfinite(latent_scale):
        raise ValueError("diffusion latent scale is invalid")

    first_meta = json.loads(paths["first_stage_run_meta"].read_text(encoding="utf-8"))
    diffusion_meta = json.loads(paths["diffusion_run_meta"].read_text(encoding="utf-8"))
    acceptance = json.loads(paths["acceptance_manifest"].read_text(encoding="utf-8"))
    expected_first_ref = "outputs/runs/20260603_171917_817266_train_first_stage_bs128/checkpoints/first_stage_best_optical.pt"
    if first_meta.get("first_stage_best_optical_checkpoint") != expected_first_ref:
        raise ValueError("first-stage source run does not identify the locked checkpoint")
    if diffusion_meta.get("first_stage_checkpoint") != expected_first_ref:
        raise ValueError("diffusion source run does not identify the locked first-stage checkpoint")
    source_runs = acceptance.get("source_runs", {})
    if source_runs.get("first_stage") != config["acceptance_first_stage_run"] or source_runs.get("diffusion") != config["acceptance_diffusion_run"]:
        raise ValueError("acceptance manifest source runs do not match the locked pair")
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "hashes": hashes,
        "first_stage_state_hash": _state_hash(first_state),
        "diffusion_state_hash": _state_hash(diffusion_state),
        "teacher_model_version": first.get("model_version"),
        "diffusion_model_version": diffusion.get("model_version"),
        "latent_scale": float(latent_scale.item()),
        "physics_config": first_physics,
        "first_stage_config": first_config,
        "diffusion_config": diffusion_config,
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
        "checkpoint_hashes": {
            "first_stage": config["first_stage_checkpoint_sha256"],
            "diffusion": config["diffusion_checkpoint_sha256"],
        },
        "dataset_split_hash": config["dataset_split_hash"],
        "git_head": git["head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": config["master_seed"],
        "evaluator_calls": evaluator_calls,
        "optimizer_created": False,
        "checkpoint_created": False,
        "synthetic_diagnostic_only": True,
    }


def _sample_seed(config: dict[str, Any], bank: int, label: int, index: int) -> int:
    raw = json.dumps([config["master_seed"], bank, label, index], separators=(",", ":"), sort_keys=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _noise(seeds: list[int], shape: tuple[int, int, int]) -> torch.Tensor:
    rows = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(rows)


def _models(lineage: dict[str, Any], device: torch.device) -> tuple[OpticalPathVAE, LatentDiffusionUNet, LinearNoiseScheduler, dict[str, Any], dict[str, Any]]:
    first_payload = _safe_load(lineage["paths"]["first_stage_checkpoint"], map_location=device)
    diffusion_payload = _safe_load(lineage["paths"]["diffusion_checkpoint"], map_location=device)
    train_config = first_payload["train_config"]
    first_stage = OpticalPathVAE(first_payload["physics_config"], train_config.first_stage).to(device)
    load_first_stage_checkpoint(first_stage, first_payload)
    diffusion = LatentDiffusionUNet(train_config.first_stage.latent_channels, train_config.diffusion).to(device)
    load_diffusion_checkpoint(diffusion, diffusion_payload, FIRST_STAGE_MODEL_VERSION)
    scheduler = LinearNoiseScheduler(train_config.diffusion.timesteps, train_config.diffusion.beta_start, train_config.diffusion.beta_end).to(device)
    first_stage.eval()
    diffusion.eval()
    for model in (first_stage, diffusion):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return first_stage, diffusion, scheduler, first_payload, diffusion_payload


def _generate_one(first_stage: OpticalPathVAE, diffusion: LatentDiffusionUNet, scheduler: LinearNoiseScheduler, diffusion_payload: dict[str, Any], labels: torch.Tensor, seeds: list[int], config: dict[str, Any], device: torch.device) -> torch.Tensor:
    noise = _noise(seeds, tuple(config["latent_shape"])).to(device)
    shape = (len(seeds), *config["latent_shape"])
    with torch.no_grad():
        latents = sample_latents(
            diffusion, scheduler, shape, device, labels,
            guidance_scale=config["guidance_scale"], initial_noise=noise,
            sampling_steps=config["sampling_steps"],
        )
        output = first_stage.decode_with_intermediates(latents * diffusion_payload["latent_scale"])
        camera = output["camera_plane"].detach().cpu().float()
    if camera.ndim != 4 or tuple(camera.shape[1:]) != (1, config["image_size"], config["image_size"]):
        raise ValueError(f"teacher camera output shape is invalid: {tuple(camera.shape)}")
    if not torch.isfinite(camera).all():
        raise ValueError("teacher camera output is non-finite")
    return camera


def _preflight(config: dict[str, Any], lineage: dict[str, Any], device_name: str) -> dict[str, Any]:
    device = _require_cuda(device_name)
    first_stage, diffusion, scheduler, _, diffusion_payload = _models(lineage, device)
    state_before = {"first_stage": lineage["first_stage_state_hash"], "diffusion": lineage["diffusion_state_hash"]}
    labels = torch.arange(config["class_count"], device=device, dtype=torch.long)
    seeds = [_sample_seed(config, bank=0, label=label, index=0) for label in range(config["class_count"])]
    start = time.monotonic()
    first = _generate_one(first_stage, diffusion, scheduler, diffusion_payload, labels, seeds, config, device)
    second = _generate_one(first_stage, diffusion, scheduler, diffusion_payload, labels, seeds, config, device)
    exact = torch.equal(first, second)
    state_after = {"first_stage": _state_hash(first_stage.state_dict()), "diffusion": _state_hash(diffusion.state_dict())}
    peak = int(torch.cuda.max_memory_allocated(device))
    finite = bool(torch.isfinite(first).all().item() and torch.isfinite(second).all().item())
    within_time = (time.monotonic() - start) < config["preflight_wall_limit_seconds"]
    passed = bool(finite and exact and state_before == state_after and peak < config["memory_limit_bytes"] and within_time)
    return {
        "passed": passed,
        "classification": "preflight_passed" if passed else "invalid_harness",
        "first_failed_gate": None if passed else "sampler_replay_state_or_resource",
        "finite": finite,
        "resume_exact": exact,
        "state_before": state_before,
        "state_after": state_after,
        "peak_cuda_memory_bytes": peak,
        "memory_gate": peak < config["memory_limit_bytes"],
        "elapsed_seconds": time.monotonic() - start,
        "evaluator_calls": 0,
        "optimizer_created": False,
        "checkpoint_created": False,
        "sample_shape": list(first.shape),
        "within_time_limit": within_time,
        "seed_hash": hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
    }


def _generate(config: dict[str, Any], config_path: str, lineage: dict[str, Any], device_name: str, output: str | Path, preflight: dict[str, Any]) -> dict[str, Any]:
    if not preflight.get("passed"):
        raise ValueError("generation requires passed Package B preflight evidence")
    device = _require_cuda(device_name)
    first_stage, diffusion, scheduler, _, diffusion_payload = _models(lineage, device)
    output_path = Path(output)
    shard_dir = output_path.with_name(f"{output_path.stem}_shards")
    state_path = output_path.with_name(f"{output_path.stem}.state.json")
    state = {"schema": f"{SCHEMA_PREFIX}_generation_state.v1", "git_head": _git()["head"], "config_sha256": _file_sha256(config_path), "shards": {}}
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("git_head") != _git()["head"] or existing.get("config_sha256") != _file_sha256(config_path):
            raise ValueError("partial generation state belongs to another HEAD or config")
        state = existing
    records = []
    state.setdefault("shards", {})
    peak = int(state.get("peak_cuda_memory_bytes", 0))
    start_time = time.monotonic()
    for bank in range(config["bank_count"]):
        for label in range(config["class_count"]):
            for begin in range(0, config["samples_per_class"], config["microbatch_size"]):
                end = min(begin + config["microbatch_size"], config["samples_per_class"])
                key = f"bank_{bank}/class_{label}/chunk_{begin:04d}"
                name = f"bank_{bank}_class_{label}_chunk_{begin:04d}.pt"
                shard_path = shard_dir / name
                if key in state["shards"]:
                    record = state["shards"][key]
                    if record.get("path") != str(shard_path) or not shard_path.is_file() or _file_sha256(shard_path) != record.get("sha256"):
                        raise ValueError(f"generation shard is stale or tampered: {key}")
                    records.append(record)
                    continue
                labels = torch.full((end - begin,), label, dtype=torch.long, device=device)
                seeds = [_sample_seed(config, bank, label, index) for index in range(begin, end)]
                images = _generate_one(first_stage, diffusion, scheduler, diffusion_payload, labels, seeds, config, device)
                payload = {"schema": f"{SCHEMA_PREFIX}_shard.v1", "bank": bank, "label": label, "start": begin, "end": end, "images": images, "labels": torch.full((end - begin,), label, dtype=torch.long), "seeds": seeds}
                _atomic_torch(shard_path, payload)
                record = {"key": key, "path": str(shard_path), "sha256": _file_sha256(shard_path), "bank": bank, "label": label, "start": begin, "end": end}
                state["shards"][key] = record
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                state["peak_cuda_memory_bytes"] = peak
                _atomic_json(state_path, state)
                records.append(record)
                print(f"[teacher-distribution-admission] generated bank={bank} class={label} {end}/{config['samples_per_class']}")
                del images
                torch.cuda.empty_cache()
    state["complete"] = True
    _atomic_json(state_path, state)
    final = {
        "schema": f"{SCHEMA_PREFIX}_banks.v1",
        "status": "teacher_audit_bank",
        "classification": "generated_for_admission",
        "first_failed_gate": None,
        "permitted_next_actions": ["validate"],
        "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
        "config_sha256": _file_sha256(config_path),
        "git_head": _git()["head"],
        "dirty_worktree": _git()["dirty_worktree"],
        "config": config,
        "checkpoint_hashes": {"first_stage": config["first_stage_checkpoint_sha256"], "diffusion": config["diffusion_checkpoint_sha256"]},
        "dataset_split_hash": config["dataset_split_hash"],
        "bank_count": config["bank_count"],
        "samples_per_class": config["samples_per_class"],
        "image_shape": [1, config["image_size"], config["image_size"]],
        "noise": "full_standard_gaussian",
        "sampler": {"guidance_scale": config["guidance_scale"], "sampling_steps": config["sampling_steps"], "kind": config["sampler"]},
        "shards": sorted(records, key=lambda row: (row["bank"], row["label"], row["start"])),
        "generation_state_sha256": _file_sha256(state_path),
        "state_hash_before": {"first_stage": lineage["first_stage_state_hash"], "diffusion": lineage["diffusion_state_hash"]},
        "state_hash_after": {"first_stage": _state_hash(first_stage.state_dict()), "diffusion": _state_hash(diffusion.state_dict())},
        "peak_cuda_memory_bytes": peak,
        "elapsed_seconds": time.monotonic() - start_time,
        "evaluator_calls": 0,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "synthetic_diagnostic_only": True,
    }
    if final["state_hash_before"] != final["state_hash_after"]:
        raise RuntimeError("teacher state changed during generation")
    _atomic_json(output_path, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B frozen DDPM--VAE teacher bank producer")
    parser.add_argument("action", choices=("lineage", "preflight", "generate"))
    parser.add_argument("--config", required=True)
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
        lineage = _lineage(config)
        payload = _base("lineage", config, args.config, git, status="valid", passed=True, classification="teacher_lineage_validated", first_failed_gate=None, permitted=["preflight"])
        payload["lineage"] = lineage
        _atomic_json(args.output, payload)
        return 0
    if not args.lineage_evidence:
        raise ValueError(f"{args.action} requires --lineage-evidence")
    lineage_evidence = json.loads(Path(args.lineage_evidence).read_text(encoding="utf-8"))
    if (
        lineage_evidence.get("schema") != f"{SCHEMA_PREFIX}_lineage.v1"
        or not lineage_evidence.get("passed")
        or lineage_evidence.get("config_sha256") != _file_sha256(args.config)
        or lineage_evidence.get("git_head") != git["head"]
    ):
        raise ValueError("Package B stage requires passed lineage evidence")
    lineage = _lineage(config)
    if args.action == "preflight":
        result = _preflight(config, lineage, args.device)
        payload = _base("preflight", config, args.config, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=["generate"] if result["passed"] else [])
        payload.update(result)
        payload["lineage_evidence_sha256"] = _file_sha256(args.lineage_evidence)
        _atomic_json(args.output, payload)
        return 0
    if not args.preflight_evidence:
        raise ValueError("generate requires --preflight-evidence")
    preflight = json.loads(Path(args.preflight_evidence).read_text(encoding="utf-8"))
    if (
        preflight.get("schema") != f"{SCHEMA_PREFIX}_preflight.v1"
        or not preflight.get("passed")
        or preflight.get("config_sha256") != _file_sha256(args.config)
        or preflight.get("git_head") != git["head"]
    ):
        raise ValueError("generation requires passed preflight evidence")
    result = _generate(config, args.config, lineage, args.device, args.output, preflight)
    result["lineage_evidence_sha256"] = _file_sha256(args.lineage_evidence)
    result["preflight_evidence_sha256"] = _file_sha256(args.preflight_evidence)
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
