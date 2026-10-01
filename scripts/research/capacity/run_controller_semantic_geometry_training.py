"""CLI for the preregistered controller semantic-geometry S3 repair harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldm_project.v2.application.research.capacity.controller_semantic_geometry_training import AUTHORIZATION_TOKEN, HarnessFailure, load_config, load_target_bank, register_data_authority, register_target_bank, resolve_path, train_authorized_controller, training_update_preflight, _load_data_authority, _load_s3_base_codes, _validate_frozen_authority_files, _git_state, internal_validation_codes, validate_training_preflight, validate_target_bank
from ldm_project.v2.evidence.io import atomic_write_new_json, file_sha256, tensor_sha256
from scripts.research.scaled_chen import run_scaled_chen_controller_to_ols_writability_audit as s3


DEFAULT_CONFIG = Path(
    "configs/research/capacity/lightgen_controller_semantic_geometry_training_v1.yaml"
)
DEFAULT_S3_CONFIG = Path(
    "configs/research/scaled_chen/lightgen_scaled_chen_controller_to_ols_writability_audit_v1.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register-target-bank")
    register.add_argument("--partition-manifest", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--manifest", type=Path)

    authority = subparsers.add_parser("register-data-authority")
    authority.add_argument("--output-dir", type=Path)
    authority.add_argument("--expected-head")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--target-bank", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--expected-head")
    preflight.add_argument("--s3-config", type=Path, default=DEFAULT_S3_CONFIG)
    preflight.add_argument("--authorization", required=True)
    preflight.add_argument("--device", default="cuda")

    train = subparsers.add_parser("train")
    train.add_argument("--target-bank", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--authorization")
    train.add_argument("--s3-config", type=Path, default=DEFAULT_S3_CONFIG)
    train.add_argument("--device", default="cuda")
    train.add_argument("--preflight-evidence", type=Path, required=True)
    train.add_argument("--expected-head")
    return parser


def _final_output_path(config: dict[str, object], repo_root: Path) -> Path:
    output = config["output"]
    if not isinstance(output, dict):
        raise HarnessFailure("output contract is invalid")
    return resolve_path(repo_root, str(output["target_directory"])).resolve() / str(output["final_checkpoint"])


def _frozen_file_paths(config: dict[str, object], s3_config: dict[str, object], repo_root: Path) -> dict[str, Path]:
    optical = config["optical_authority"]
    if not isinstance(optical, dict):
        raise HarnessFailure("optical authority contract is invalid")
    controller = s3_config["controller_authority"]
    if not isinstance(controller, dict):
        raise HarnessFailure("controller authority contract is invalid")
    return {
        "formal_config": resolve_path(repo_root, str(optical["formal_config"])),
        "checkpoint": resolve_path(repo_root, str(optical["checkpoint"])),
        "calibration": resolve_path(repo_root, str(optical["calibration"])),
        "calibration_manifest": resolve_path(repo_root, str(optical["calibration_manifest"])),
        "evaluator_identity_manifest": resolve_path(repo_root, str(optical["evaluator_identity_manifest"])),
        "controller_authority": resolve_path(repo_root, str(controller["authority"])),
        "migrated_controller": resolve_path(repo_root, str(controller["migrated_checkpoint"])),
    }


def _load_runtime_inputs(
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    require_output_directory_absent: bool,
) -> tuple[Path, dict[str, object], dict[str, object], object, object, object, dict[str, Path], dict[str, str], dict[str, object]]:
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    s3_config_path = args.s3_config.resolve()
    head, dirty = _git_state(repo_root)
    if dirty:
        raise HarnessFailure("clean Git worktree is required for authorized controller work")
    if args.expected_head is not None and head != args.expected_head:
        raise HarnessFailure("current HEAD does not match the expected clean HEAD")
    s3_config = s3.load_config(s3_config_path)[1]
    if args.device != str(s3_config["execution"]["device"]):
        raise HarnessFailure("device differs from the registered S3 execution authority")
    final_output = _final_output_path(config, repo_root)
    if require_output_directory_absent and final_output.parent.exists():
        raise HarnessFailure("training output directory is not a fresh write-once path")
    resource = s3._resource_preflight(s3_config, final_output)
    authority = _load_data_authority(config, repo_root, target_bank_path=args.target_bank)
    bank = load_target_bank(args.target_bank, expected_sha256=authority["hashes"]["target_bank_sha256"])
    validate_target_bank(
        bank,
        authority["reference"],
        config,
        expected_artifact_sha256=authority["hashes"]["target_bank_sha256"],
        expected_partition_manifest_file_sha256=authority["hashes"]["partition_manifest_file_sha256"],
        repo_root=repo_root,
    )
    frozen_paths = _frozen_file_paths(config, s3_config, repo_root)
    frozen_authorities = _validate_frozen_authority_files(config, repo_root)
    s3_authority, controller_state, migrated_path = s3._validate_controller_authority(s3_config, repo_root)
    return (
        repo_root,
        s3_config,
        authority,
        bank,
        s3_authority,
        controller_state,
        frozen_paths,
        frozen_authorities,
        {
            "head": head,
            "config_path": config_path,
            "s3_config_path": s3_config_path,
            "final_output": final_output,
            "migrated_path": migrated_path,
            "resource": resource,
        },
    )


def _run_training_preflight(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    if args.authorization != AUTHORIZATION_TOKEN:
        raise HarnessFailure("BLOCKED_AUTHORIZATION_REQUIRED")
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    final_output = _final_output_path(config, repo_root)
    expected_evidence = final_output.with_name("preflight.json")
    if output != expected_evidence:
        raise HarnessFailure("preflight evidence path must be the configured run's preflight.json")
    (
        repo_root,
        s3_config,
        authority,
        bank,
        _s3_authority,
        controller_state,
        frozen_paths,
        frozen_authorities,
        runtime,
    ) = _load_runtime_inputs(args, config, require_output_directory_absent=True)
    device = torch.device(args.device)
    optical_model, calibration, _, _ = s3._load_optical_authority(s3_config, device)
    generator = s3._load_controller_generator(controller_state, device)
    s3_codes = _load_s3_base_codes(config, repo_root)
    preflight = training_update_preflight(
        generator,
        optical_model,
        bank,
        calibration.port_scale.to(device=device, dtype=torch.float32),
        authorization=args.authorization,
        frozen_file_paths=frozen_paths,
        s3_base_codes=s3_codes,
        max_peak_cuda_bytes=int(s3_config["execution"]["max_peak_cuda_bytes"]),
    )
    evidence = {
        **preflight,
        "task_id": config["task_id"],
        "git_head": runtime["head"],
        "config_sha256": file_sha256(runtime["config_path"]),
        "s3_config_sha256": file_sha256(runtime["s3_config_path"]),
        "target_bank_sha256": bank.artifact_sha256,
        "partition_manifest_file_sha256": authority["hashes"]["partition_manifest_file_sha256"],
        "data_firewall_manifest_sha256": authority["hashes"]["data_firewall_manifest_sha256"],
        "s3_base_code_sha256": tensor_sha256(s3_codes),
        "internal_validation_code_sha256": tensor_sha256(internal_validation_codes()),
        "frozen_authorities": frozen_authorities,
        "frozen_file_paths": {name: str(path) for name, path in frozen_paths.items()},
        "resource_preflight": runtime["resource"],
        "entrypoint": str(Path(__file__).resolve()),
    }
    atomic_write_new_json(output, evidence, artifact_kind="controller semantic geometry training preflight")
    return {**evidence, "evidence": str(output), "evidence_sha256": file_sha256(output)}


def _run_authorized_training(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    if args.authorization != AUTHORIZATION_TOKEN:
        raise HarnessFailure("BLOCKED_AUTHORIZATION_REQUIRED")
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    final_output = _final_output_path(config, repo_root)
    if output != final_output:
        raise HarnessFailure("training output path must be the configured final checkpoint")
    evidence_path = args.preflight_evidence.resolve()
    if evidence_path != final_output.with_name("preflight.json"):
        raise HarnessFailure("training must consume the configured run's preflight.json")
    (
        repo_root,
        s3_config,
        authority,
        bank,
        _s3_authority,
        controller_state,
        frozen_paths,
        _frozen_authorities,
        runtime,
    ) = _load_runtime_inputs(args, config, require_output_directory_absent=False)
    s3_codes = _load_s3_base_codes(config, repo_root)

    preflight_hash = validate_training_preflight(
        evidence_path,
        expected_git_head=runtime["head"],
        expected_config_sha256=file_sha256(runtime["config_path"]),
        expected_s3_config_sha256=file_sha256(runtime["s3_config_path"]),
        expected_target_bank_sha256=bank.artifact_sha256,
        expected_data_firewall_manifest_sha256=authority["hashes"]["data_firewall_manifest_sha256"],
        expected_s3_base_code_sha256=tensor_sha256(s3_codes),
        expected_internal_validation_code_sha256=tensor_sha256(internal_validation_codes()),
        expected_max_peak_cuda_bytes=int(s3_config["execution"]["max_peak_cuda_bytes"]),
        frozen_file_paths=frozen_paths,
    )
    device = torch.device(args.device)
    optical_model, calibration, _, _ = s3._load_optical_authority(s3_config, device)
    generator = s3._load_controller_generator(controller_state, device)
    result = train_authorized_controller(
        generator,
        optical_model,
        bank,
        calibration.port_scale.to(device=device, dtype=torch.float32),
        output,
        authorization=args.authorization,
        frozen_file_paths=frozen_paths,
        preflight_evidence_sha256=preflight_hash,
    )
    return {**result, "git_head": runtime["head"], "resource_preflight": runtime["resource"], "preflight_evidence_sha256": preflight_hash}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "register-target-bank":
            output, manifest = register_target_bank(
                config,
                args.repo_root,
                args.partition_manifest,
                args.output,
                args.manifest,
            )
            print(json.dumps({"target_bank": str(output), "manifest": str(manifest)}, indent=2))
            return 0
        if args.command == "register-data-authority":
            result = register_data_authority(
                config,
                args.repo_root,
                config_path=args.config,
                output_dir=args.output_dir,
                expected_head=args.expected_head,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "preflight":
            result = _run_training_preflight(args, config)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 2
        if args.command != "train":
            raise AssertionError("unknown command")
        result = _run_authorized_training(args, config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (HarnessFailure, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"gate": "invalid_harness", "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
