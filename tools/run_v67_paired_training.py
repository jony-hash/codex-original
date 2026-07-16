"""Guarded launcher for the two v67 diagnostic training branches.

The launcher never starts training unless the convergence report explicitly
selects a grid.  Both commands receive the same seed and are kept as separate
runs so their checkpoints and artifacts cannot overwrite formal v56 outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(*, preflight: str | Path, config_a: str | Path, config_b: str | Path,
        train_script: str | Path = "train.py", dry_run: bool = True) -> dict[str, object]:
    report = json.loads(Path(preflight).read_text(encoding="utf-8"))
    if report.get("status") != "converged" or not report.get("selected_grid"):
        raise RuntimeError("v67 convergence gate has not selected a propagation grid")
    commands = [
        ["python", str(train_script), "--config", str(config_a), "--seed", "1234"],
        ["python", str(train_script), "--config", str(config_b), "--seed", "1234"],
    ]
    if not dry_run:
        for command in commands:
            subprocess.run(command, check=True)
    return {
        "diagnostic_only": True,
        "selected_grid": report["selected_grid"],
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
    print(json.dumps(run(
        preflight=args.preflight,
        config_a=args.config_a,
        config_b=args.config_b,
        train_script=args.train_script,
        dry_run=not args.execute,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
