"""Guarded launcher for v67.2 transparent A/B diagnostic training.

This launcher is intentionally stricter than the v67 launcher: a numerical
grid result alone is insufficient.  The v67.2 report must contain successful
full-objective gradient and memory admission records before a subprocess can
be started.  It never promotes a checkpoint to the formal loader.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPORT_SCHEMA = "v67_transparent_convergence_v4"


def run(
    *,
    preflight: str | Path,
    config_a: str | Path,
    config_b: str | Path,
    train_script: str | Path = "train.py",
    dry_run: bool = True,
) -> dict[str, object]:
    report = json.loads(Path(preflight).read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise RuntimeError("v67.2 launcher requires a v67_transparent_convergence_v3 report")
    if report.get("status") != "converged" or not report.get("training_field_converged"):
        raise RuntimeError("v67.2 numerical/training-field gate has not converged")
    gradient = report.get("gradient_admission", {})
    memory = report.get("memory_admission", {})
    if gradient.get("passed") is not True:
        raise RuntimeError("v67.2 gradient admission has not passed")
    if memory.get("passed") is not True:
        raise RuntimeError("v67.2 memory admission has not passed")
    commands = [
        ["python", str(train_script), "--config", str(config_a), "--seed", "1234"],
        ["python", str(train_script), "--config", str(config_b), "--seed", "1234"],
    ]
    if not dry_run:
        for command in commands:
            subprocess.run(command, check=True)
    return {
        "diagnostic_only": True,
        "optical_boundary_contract": "ideal_transparent_thin_phase_v2",
        "selected_state_grid": report["selected_state_grid"],
        "selected_fft_grid": report["selected_fft_grid"],
        "same_seed": 1234,
        "commands": commands,
        "executed": not dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--config-a", required=True, type=Path)
    parser.add_argument("--config-b", required=True, type=Path)
    parser.add_argument("--train-script", default="train.py", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
