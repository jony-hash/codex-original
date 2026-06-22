#!/usr/bin/env python3
"""Run the gated v9.10 label-template prototype audit under /tmp."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from ldm_project.config import load_training_config


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CONFIG = REPO_ROOT / "configs/audit_one_shot_d2nn_v9_10_label_template_inverse_prototype_baseline.yaml"
TRAIN_CONFIG = (
    REPO_ROOT
    / "configs/train_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_10_"
    "label_template_inverse_prototype_baseline.yaml"
)
EVAL_CONFIG = (
    REPO_ROOT
    / "configs/eval_one_shot_distill_d2nn_mnist_3k_fast200_student_v9_10_"
    "label_template_inverse_prototype_baseline_strict_readability_test.yaml"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/v9_10_label_template_inverse")
EXPECTED_FROZEN_GROUPS = {"generator_base", "phase", "amplitude"}


class AuditError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--test-dataset", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_tmp_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    tmp_root = Path("/tmp").resolve()
    if resolved != tmp_root and tmp_root not in resolved.parents:
        raise AuditError(f"v9.10 pre-acceptance output must stay under /tmp, got {resolved}")
    return resolved


def _validate_config_contract() -> dict[str, Any]:
    configs = {
        "audit": load_training_config(AUDIT_CONFIG),
        "train": load_training_config(TRAIN_CONFIG),
        "eval": load_training_config(EVAL_CONFIG),
    }
    for role, config in configs.items():
        one_shot = config.one_shot
        physics = config.physics
        checks = {
            "generator_e_arch": one_shot.generator_e_arch == "label_template",
            "generator_command_transform": one_shot.generator_command_transform == "intensity_to_amplitude",
            "generator_template_init": one_shot.generator_template_init == "mnist_prototype",
            "code_conditioning_disabled": one_shot.generator_e_arch == "label_template",
            "front_path_model": physics is not None and physics.front_path_model == "relay_coherent_psf",
            "front_psf_sigma": physics is not None and physics.front_relay_psf_sigma_px == 1.0,
            "back_path_model": physics is not None and physics.back_path_model == "relay_psf",
            "back_psf_sigma": physics is not None and physics.readout.relay_psf_sigma_px == 1.0,
            "d2nn_grid_size": one_shot.d2nn.grid_size == 200,
            "propagation_grid_size": one_shot.d2nn.propagation_grid_size == 1024,
            "encoder_layers": one_shot.d2nn.encoder_layers == 1,
            "decoder_layers": one_shot.d2nn.decoder_layers == 1,
            "layer_spacing": one_shot.d2nn.layer_spacing == 5.0e-4,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AuditError(f"{role} config violates v9.10 contract: {failed}")
    return {
        "baseline_kind": "prototype_baseline",
        "code_conditioning_active": False,
        "generator_e_arch": "label_template",
        "generator_command_transform": "intensity_to_amplitude",
        "front_path_model": "relay_coherent_psf",
        "back_path_model": "relay_psf",
        "front_psf_sigma_px": 1.0,
        "back_psf_sigma_px": 1.0,
        "d2nn_grid_size": 200,
        "propagation_grid_size": 1024,
        "layer_spacing_m": 5.0e-4,
    }


def _command(
    *,
    config: Path,
    stage: str,
    device: str,
    output_root: Path,
    run_name: str,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--train-config",
        str(config),
        "--stage",
        stage,
        "--device",
        device,
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        *extra,
    ]


def _new_run(output_root: Path, before: set[Path], run_name: str) -> Path:
    after = {path.resolve() for path in output_root.glob(f"*_train_{run_name}") if path.is_dir()}
    created = sorted(after - before)
    if len(created) != 1:
        raise AuditError(f"Expected one new run for {run_name}, found {created}")
    return created[0]


def _run_command(command: list[str], output_root: Path, run_name: str) -> Path:
    before = {path.resolve() for path in output_root.glob(f"*_train_{run_name}") if path.is_dir()}
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return _new_run(output_root, before, run_name)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_frozen_report(run_meta: dict[str, Any]) -> dict[str, Any]:
    report = run_meta.get("one_shot_frozen_parameter_check")
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise AuditError("Training run is missing a passing frozen-parameter report")
    groups = report.get("groups", {})
    missing = sorted(EXPECTED_FROZEN_GROUPS - set(groups))
    changed = {
        name: details.get("max_abs_change")
        for name, details in groups.items()
        if float(details.get("max_abs_change", 1.0)) != 0.0
    }
    if missing or changed or float(report.get("max_abs_change", 1.0)) != 0.0:
        raise AuditError(f"Frozen-parameter contract failed: missing={missing}, changed={changed}")
    return report


def _strict_gate(metrics: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    acceptance = metrics.get("strict_readability_acceptance", {})
    checks = acceptance.get("checks", {})
    expected_checks = {"paired_corr", "paired_dice", "student_high_saturation_ratio"}
    if set(checks) != expected_checks:
        raise AuditError(f"Strict gate has unexpected checks: {sorted(checks)}")
    return acceptance.get("passed") is True, acceptance


def _summary_path(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_root / f"v9_10_label_template_audit_summary_{timestamp}.json"


def _write_summary(output_root: Path, summary: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = _summary_path(output_root)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return path


def _print_commands(commands: list[list[str]]) -> None:
    for command in commands:
        print(" ".join(command))


def main() -> int:
    args = _parse_args()
    output_root = _require_tmp_output_root(args.output_root)
    datasets = {
        "train": args.train_dataset.expanduser().resolve(),
        "validation": args.validation_dataset.expanduser().resolve(),
        "test": args.test_dataset.expanduser().resolve(),
    }
    missing = [f"{role}={path}" for role, path in datasets.items() if not path.is_file()]
    if missing:
        raise AuditError(f"Missing audit datasets: {missing}")
    contract = _validate_config_contract()

    audit_name = "v9_10_label_template_inverse_audit"
    eval_name = "v9_10_label_template_inverse_test_eval"
    sample_name = "v9_10_label_template_inverse_sample"
    audit_command = _command(
        config=AUDIT_CONFIG,
        stage="one_shot_distill",
        device=args.device,
        output_root=output_root,
        run_name=audit_name,
        extra=[
            "--distill-dataset",
            str(datasets["train"]),
            "--distill-validation-dataset",
            str(datasets["validation"]),
        ],
    )
    placeholder_checkpoint = "<audit_checkpoint>"
    eval_command = _command(
        config=EVAL_CONFIG,
        stage="one_shot_distill_eval",
        device=args.device,
        output_root=output_root,
        run_name=eval_name,
        extra=["--distill-dataset", str(datasets["test"]), "--one-shot-checkpoint", placeholder_checkpoint],
    )
    sample_command = _command(
        config=EVAL_CONFIG,
        stage="one_shot_sample",
        device=args.device,
        output_root=output_root,
        run_name=sample_name,
        extra=["--one-shot-checkpoint", placeholder_checkpoint],
    )
    if args.dry_run:
        _print_commands([audit_command, eval_command, sample_command])
        return 0

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise AuditError(
            f"Formal v9.10 audit requires visible CUDA; requested={args.device}, "
            f"cuda_available={torch.cuda.is_available()}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    audit_run = _run_command(audit_command, output_root, audit_name)
    audit_meta = _load_json(audit_run / "run_meta.json")
    frozen_report = _validate_frozen_report(audit_meta)
    checkpoint_value = audit_meta.get("one_shot_distill_best_checkpoint")
    if not checkpoint_value:
        raise AuditError("Audit run did not produce one_shot_distill_best_checkpoint")
    checkpoint = _resolve_recorded_path(str(checkpoint_value))
    if not checkpoint.is_file():
        raise AuditError(f"Audit checkpoint does not exist: {checkpoint}")

    eval_command = [str(checkpoint) if item == placeholder_checkpoint else item for item in eval_command]
    eval_run = _run_command(eval_command, output_root, eval_name)
    eval_meta = _load_json(eval_run / "run_meta.json")
    metric_source = eval_meta.get("one_shot_distill_eval_metric_source")
    if not metric_source:
        raise AuditError("Eval run did not record its metric source")
    metrics = _load_json(eval_run / str(metric_source))
    passed, acceptance = _strict_gate(metrics)
    summary: dict[str, Any] = {
        "status": "strict_gate_passed" if passed else "strict_gate_failed",
        "contract": contract,
        "datasets": {key: str(value) for key, value in datasets.items()},
        "commands": {"audit": audit_command, "eval": eval_command},
        "runs": {"audit": str(audit_run), "eval": str(eval_run)},
        "checkpoint": str(checkpoint),
        "frozen_parameter_check": frozen_report,
        "strict_readability_acceptance": acceptance,
    }
    if not passed:
        path = _write_summary(output_root, summary)
        print(f"v9.10 strict gate failed; sample was not run. summary={path}")
        return 2

    sample_command = [str(checkpoint) if item == placeholder_checkpoint else item for item in sample_command]
    sample_run = _run_command(sample_command, output_root, sample_name)
    summary["status"] = "complete"
    summary["commands"]["sample"] = sample_command
    summary["runs"]["sample"] = str(sample_run)
    path = _write_summary(output_root, summary)
    print(f"v9.10 prototype baseline audit complete: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, subprocess.CalledProcessError) as exc:
        print(f"v9.10 audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
