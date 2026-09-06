"""CLI for the preregistered controller semantic-geometry S3 repair harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldm_project.v2.application.research.controller_semantic_geometry_training import (
    AUTHORIZATION_TOKEN,
    HarnessFailure,
    load_config,
    load_target_bank,
    no_update_preflight,
    register_data_authority,
    register_target_bank,
    resolve_path,
    train_authorized_controller,
    _load_data_authority,
    validate_target_bank,
)


DEFAULT_CONFIG = Path(
    "configs/research/lightgen_controller_semantic_geometry_training_v1.yaml"
)
DEFAULT_S3_CONFIG = Path(
    "configs/research/lightgen_scaled_chen_controller_to_ols_writability_audit_v1.yaml"
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

    train = subparsers.add_parser("train")
    train.add_argument("--target-bank", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--authorization")
    train.add_argument("--s3-config", type=Path, default=DEFAULT_S3_CONFIG)
    train.add_argument("--device", default="cuda")
    return parser


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
            result = no_update_preflight(
                args.config,
                args.repo_root,
                args.target_bank,
                args.output,
                expected_head=args.expected_head,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command != "train":
            raise AssertionError("unknown command")
        if args.authorization != AUTHORIZATION_TOKEN:
            raise HarnessFailure("BLOCKED_AUTHORIZATION_REQUIRED")

        # The S3 authority loader is deliberately reached only after the exact
        # user authorization token has been supplied.  It supplies the frozen
        # step-5000 optical model, migrated controller state, and calibration;
        # the repair module itself never imports the legacy script layer.
        from scripts.research import run_scaled_chen_controller_to_ols_writability_audit as s3

        s3_config = s3.load_config(args.s3_config)[1]
        authority, controller_state, _ = s3._validate_controller_authority(s3_config, args.repo_root)
        del authority
        device = torch.device(args.device)
        optical_model, calibration, _, _ = s3._load_optical_authority(s3_config, device)
        generator = s3._load_controller_generator(controller_state, device)
        authority = _load_data_authority(config, args.repo_root, target_bank_path=args.target_bank)
        bank = load_target_bank(args.target_bank, expected_sha256=authority["hashes"]["target_bank_sha256"])
        validate_target_bank(
            bank,
            authority["reference"],
            config,
            expected_artifact_sha256=authority["hashes"]["target_bank_sha256"],
            expected_partition_manifest_file_sha256=authority["hashes"]["partition_manifest_file_sha256"],
            repo_root=args.repo_root,
        )
        result = train_authorized_controller(
            generator,
            optical_model,
            bank,
            calibration.port_scale.to(device=device, dtype=torch.float32),
            args.output,
            authorization=args.authorization,
            frozen_file_paths={
                "optical_checkpoint": resolve_path(args.repo_root, config["optical_authority"]["checkpoint"]),
                "calibration": resolve_path(args.repo_root, config["optical_authority"]["calibration"]),
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (HarnessFailure, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"gate": "invalid_harness", "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
