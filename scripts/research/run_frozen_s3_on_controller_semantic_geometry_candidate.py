"""Run the frozen S3 audit on the existing post-step-492 candidate only.

This entrypoint has no training path. It validates the immutable training
candidate and then injects only its generator state into the existing S3
consumer; the S3 panel, metrics, router, and optical authority remain
unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research import controller_semantic_geometry_training as training
from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.student.identity_generator import IdentityAnchoredConditionalAmplitudeGenerator
from scripts.research import run_controller_semantic_geometry_training as training_runner
from scripts.research import run_scaled_chen_controller_to_ols_writability_audit as s3


TRAINING_CONFIG_PATH = Path("configs/research/lightgen_controller_semantic_geometry_training_v1_attempt_02.yaml")
TARGET_BANK_PATH = Path("outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_data_firewall_v1_attempt_01/controller_semantic_geometry_r1_target_bank.v1.pt")
TRAINING_ATTEMPT_PATH = Path("outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_02")
CANDIDATE_PATH = TRAINING_ATTEMPT_PATH / "controller_semantic_geometry_final_step_492.pt"
PREFLIGHT_PATH = TRAINING_ATTEMPT_PATH / "preflight.json"
S3_OUTPUT_PATH = Path("outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_post_step_492_v1_attempt_01")
TRAINING_EXECUTION_HEAD = "169fd515b7d6ff6d862222c066c44a39dc373879"
PREFLIGHT_SHA256 = "ed44ba642252a7ecb08faa2541af9af55f9b7d1a72c9ea8696afd8fc689c1835"
CANDIDATE_SHA256 = "4d7c574d98078df38f9866734ad60bacd82c5a7e18dc9d1737fd70e1f42d8d84"
CANDIDATE_STATE_HASH = "cccefcb25c674b11f64653fe1f97a0193d4896bc69c4fe01cf3812daffda1b99"
STARTING_STATE_HASH = "3d06989ba6c5f9154e4d05b64cef42bd4c94b02686ff5b2120a953d9f87a7931"


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return payload


def _load_candidate_state_payload(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    expected = {
        "schema": "controller_semantic_geometry_final_controller_state.v1",
        "objective_family": training.OBJECTIVE_FAMILY,
        "checkpoint_selection": "FINAL_AUTHORIZED_STEP_ONLY",
        "step": training.HORIZON,
        "training_authorization": training.AUTHORIZATION_TOKEN,
        "training_authorized": True,
        "formal_promotion": False,
        "s3_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("candidate checkpoint contract is stale or unauthorized")
    state = payload.get("controller_state")
    if not isinstance(state, Mapping):
        raise ValueError("candidate controller state is not a mapping")
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    if set(state) != set(generator.state_dict()):
        raise ValueError("candidate controller state keys are not generator-only")
    if any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("candidate controller state is not tensor-only")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("candidate controller state is non-finite")
    missing, unexpected = generator.load_state_dict(dict(state), strict=True)
    if missing or unexpected:
        raise ValueError("candidate controller state strict load failed")
    return {name: value.detach().cpu().contiguous() for name, value in state.items()}


def _validate_existing_candidate(
    repo_root: Path,
    candidate_path: Path,
    preflight_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], training.TargetBank, dict[str, Any], dict[str, Any]]:
    candidate_path = candidate_path.resolve()
    preflight_path = preflight_path.resolve()
    if candidate_path.parent != preflight_path.parent or candidate_path.parent != (repo_root / TRAINING_ATTEMPT_PATH).resolve():
        raise ValueError("candidate and preflight are not the registered training attempt")
    if sorted(path.name for path in candidate_path.parent.iterdir() if path.is_file()) != [candidate_path.name, preflight_path.name]:
        raise ValueError("training attempt contains unexpected files")
    if file_sha256(candidate_path) != CANDIDATE_SHA256:
        raise ValueError("candidate checkpoint SHA-256 mismatch")
    if file_sha256(preflight_path) != PREFLIGHT_SHA256:
        raise ValueError("candidate preflight SHA-256 mismatch")
    payload = safe_torch_load(candidate_path, expected_sha256=CANDIDATE_SHA256, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("candidate checkpoint payload is not a mapping")
    state = _load_candidate_state_payload(payload)
    if payload.get("controller_state_hash") != CANDIDATE_STATE_HASH:
        raise ValueError("candidate final controller state hash mismatch")
    if payload.get("preflight_evidence_sha256") != PREFLIGHT_SHA256:
        raise ValueError("candidate preflight binding mismatch")
    for name, expected in (
        ("loss", 0.34952232241630554),
        ("mean_anchor", 0.19811758399009705),
        ("mean_covariance", 0.5009270906448364),
    ):
        if payload.get(name) != expected:
            raise ValueError(f"candidate {name} binding mismatch")

    preflight = _read_json(preflight_path)
    if preflight.get("git_head") != TRAINING_EXECUTION_HEAD:
        raise ValueError("candidate training execution HEAD mismatch")
    config = training.load_config(repo_root / TRAINING_CONFIG_PATH)
    s3_config_path, s3_config, _, _ = s3.load_config(repo_root / s3.DEFAULT_CONFIG_PATH)
    authority = training._load_data_authority(
        config, repo_root, target_bank_path=repo_root / TARGET_BANK_PATH
    )
    bank = training.load_target_bank(
        repo_root / TARGET_BANK_PATH,
        expected_sha256=authority["hashes"]["target_bank_sha256"],
    )
    training.validate_target_bank(
        bank,
        authority["reference"],
        config,
        expected_artifact_sha256=authority["hashes"]["target_bank_sha256"],
        expected_partition_manifest_file_sha256=authority["hashes"]["partition_manifest_file_sha256"],
        repo_root=repo_root,
    )
    frozen_paths = training_runner._frozen_file_paths(config, s3_config, repo_root)
    s3_codes = training._load_s3_base_codes(config, repo_root)
    validated_preflight_sha = training.validate_training_preflight(
        preflight_path,
        expected_git_head=TRAINING_EXECUTION_HEAD,
        expected_config_sha256=file_sha256(repo_root / TRAINING_CONFIG_PATH),
        expected_s3_config_sha256=file_sha256(s3_config_path),
        expected_target_bank_sha256=authority["hashes"]["target_bank_sha256"],
        expected_data_firewall_manifest_sha256=authority["hashes"]["data_firewall_manifest_sha256"],
        expected_s3_base_code_sha256=tensor_sha256(s3_codes),
        expected_internal_validation_code_sha256=tensor_sha256(training.internal_validation_codes()),
        expected_max_peak_cuda_bytes=int(s3_config["execution"]["max_peak_cuda_bytes"]),
        frozen_file_paths=frozen_paths,
    )
    if validated_preflight_sha != PREFLIGHT_SHA256:
        raise ValueError("candidate preflight evidence identity changed")

    _, registered_state, _ = s3._validate_controller_authority(s3_config, repo_root)
    registered_generator = s3._load_controller_generator(registered_state, torch.device("cpu"))
    candidate_generator = IdentityAnchoredConditionalAmplitudeGenerator()
    candidate_generator.load_state_dict(state, strict=True)
    starting_hash = training._module_state_hash(registered_generator)
    final_hash = training._module_state_hash(candidate_generator)
    if starting_hash != STARTING_STATE_HASH or final_hash != CANDIDATE_STATE_HASH or starting_hash == final_hash:
        raise ValueError("candidate state transition is not the registered one-shot transition")

    binding = {
        "schema": payload["schema"],
        "path": s3._relative(candidate_path),
        "sha256": CANDIDATE_SHA256,
        "training_attempt": s3._relative(candidate_path.parent),
        "training_preflight": s3._relative(preflight_path),
        "training_preflight_sha256": PREFLIGHT_SHA256,
        "training_execution_head": TRAINING_EXECUTION_HEAD,
        "starting_controller_state_hash": STARTING_STATE_HASH,
        "final_controller_state_hash": CANDIDATE_STATE_HASH,
        "executed_steps": training.HORIZON,
        "checkpoint_selection": payload["checkpoint_selection"],
        "objective_family": payload["objective_family"],
        "training_authorization": payload["training_authorization"],
    }
    return state, binding, bank, config, s3_config


def _internal_panel(codes: torch.Tensor) -> dict[str, torch.Tensor]:
    training._validate_codes(codes, name="internal-validation codes")
    labels = torch.arange(s3.CLASS_COUNT, dtype=torch.long).repeat_interleave(s3.PANEL_PER_CLASS)
    split = torch.cat(
        [
            torch.zeros(s3.PANEL_SELECTION_PER_CLASS, dtype=torch.long),
            torch.ones(s3.PANEL_CONFIRMATION_PER_CLASS, dtype=torch.long),
        ]
    ).repeat(s3.CLASS_COUNT)
    return {
        "base_codes": codes.detach().cpu().contiguous(),
        "codes": codes.detach().cpu().repeat(s3.CLASS_COUNT, 1).contiguous(),
        "labels": labels,
        "split": split,
    }


def _run_internal_validation(
    state: Mapping[str, torch.Tensor],
    bank: training.TargetBank,
    config: Mapping[str, Any],
    s3_config: Mapping[str, Any],
    repo_root: Path,
    candidate_state_hash: str,
) -> dict[str, Any]:
    device = torch.device(str(s3_config["execution"]["device"]))
    model, calibration, _, _ = s3._load_optical_authority(s3_config, device)
    generator = s3._load_controller_generator(
        {f"generator.{name}": value for name, value in state.items()}, device
    )
    codes = training.internal_validation_codes()
    internal_code_sha256 = tensor_sha256(codes)
    s3_codes = training._load_s3_base_codes(config, repo_root)
    if training.code_overlap_count(codes, s3_codes) != 0:
        raise ValueError("internal-validation code panel overlaps S3 panel")
    panel = _internal_panel(codes)
    scale = calibration.port_scale.detach().cpu().contiguous()
    model_before = s3.rcr.state_digest(model)
    generator_before = s3._generator_state_hash(generator)
    mode_before = s3.bogt.mode_bank_sha256(model)
    _, _, power = s3._capture_controller_states(
        model,
        generator,
        panel,
        scale,
        device=device,
        batch_size=int(s3_config["execution"]["batch_size"]),
    )
    model_after = s3.rcr.state_digest(model)
    generator_after = s3._generator_state_hash(generator)
    mode_after = s3.bogt.mode_bank_sha256(model)
    labels = panel["labels"]
    target_labels = bank.validation_labels
    per_class = []
    for label in range(s3.CLASS_COUNT):
        candidate_values = power[labels == label]
        target_values = bank.validation_r1[target_labels == label]
        candidate_mean = candidate_values.mean(dim=0)
        target_mean = target_values.mean(dim=0)
        per_class.append(
            {
                "label": label,
                "candidate_mean": float(candidate_mean.mean()),
                "target_mean": float(target_mean.mean()),
                "centroid_l2": float(torch.linalg.vector_norm(candidate_mean - target_mean)),
                "candidate_min": float(candidate_values.min()),
                "candidate_max": float(candidate_values.max()),
                "finite": True,
            }
        )
    if not torch.isfinite(power).all():
        raise ValueError("internal-validation R1 output is non-finite")
    unchanged = bool(
        model_before == model_after
        and generator_before == generator_after == candidate_state_hash
        and mode_before == mode_after
    )
    if not unchanged:
        raise ValueError("internal-validation mutated frozen state")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        del model, calibration, generator
        torch.cuda.empty_cache()
    return {
        "schema": "controller_semantic_geometry_internal_validation.v1",
        "status": "descriptive_completed",
        "descriptive_only": True,
        "candidate_controller_state_hash": candidate_state_hash,
        "target_bank_sha256": bank.artifact_sha256,
        "internal_validation_r1_sha256": tensor_sha256(bank.validation_r1),
        "internal_validation_indices_sha256": tensor_sha256(bank.validation_indices),
        "internal_validation_labels_sha256": tensor_sha256(bank.validation_labels),
        "code_seed": training.INTERNAL_VALIDATION_CODE_SEED,
        "base_code_count": int(codes.shape[0]),
        "base_codes_sha256": internal_code_sha256,
        "panel_code_sha256": tensor_sha256(panel["codes"]),
        "s3_base_code_overlap_count": training.code_overlap_count(codes, s3_codes),
        "states": int(power.shape[0]),
        "r1_shape": list(power.shape),
        "r1_dtype": str(power.dtype),
        "finite": True,
        "per_class": per_class,
        "global_candidate_mean": float(power.mean()),
        "global_candidate_min": float(power.min()),
        "global_candidate_max": float(power.max()),
        "state_isolation": {
            "optical_model_unchanged": model_before == model_after,
            "candidate_generator_unchanged": generator_before == generator_after,
            "mode_bank_unchanged": mode_before == mode_after,
            "optimizer_created": False,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "candidate_changed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=s3.DEFAULT_CONFIG_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--training-preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--output-dir", type=Path, default=S3_OUTPUT_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    candidate_path = (repo_root / args.candidate).resolve() if not args.candidate.is_absolute() else args.candidate.resolve()
    preflight_path = (repo_root / args.training_preflight).resolve() if not args.training_preflight.is_absolute() else args.training_preflight.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    state_before = s3._git_preflight(require_clean=True)
    if output_dir.exists():
        raise RuntimeError("candidate S3 attempt directory already exists")
    if config_path != (repo_root / s3.DEFAULT_CONFIG_PATH).resolve():
        raise ValueError("candidate S3 run must use the registered frozen S3 config")
    state, binding, bank, training_config, s3_config = _validate_existing_candidate(
        repo_root, candidate_path, preflight_path
    )
    internal = _run_internal_validation(
        state,
        bank,
        training_config,
        s3_config,
        repo_root,
        CANDIDATE_STATE_HASH,
    )
    result = s3.run(
        config_path,
        output_dir,
        controller_state_override=state,
        controller_candidate=binding,
        controller_candidate_path=candidate_path,
        internal_validation=internal,
    )
    manifest_path = output_dir / "s3_manifest.json"
    manifest_sha256 = file_sha256(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("controller_candidate") != binding or manifest.get("internal_validation") != internal:
        raise ValueError("S3 manifest candidate or internal-validation binding is stale")
    if manifest.get("git", {}).get("starting_head") != state_before["commit"]:
        raise ValueError("S3 manifest starting HEAD mismatch")
    return_payload = {
        **result,
        "current_git_head": state_before["commit"],
        "training_candidate": binding,
        "internal_validation": internal,
        "s3_manifest": str(manifest_path),
        "s3_manifest_sha256": manifest_sha256,
    }
    print(json.dumps(return_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
