#!/usr/bin/env python3
"""Deprecated v55.2 entry point retained for historical command compatibility.

The v58.2 layered validator supersedes it because the former low-resolution
full-size direct-RS comparison was reference-harness-invalid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.validate_v58_2_relay_reference import run as run_reference
except ModuleNotFoundError:  # direct `python tools/...` execution
    from validate_v58_2_relay_reference import run as run_reference


def run(config_path: str | Path, device: str = "cpu") -> dict[str, object]:
    report = dict(run_reference(config_path, device))
    report["v55_2_status"] = "superseded-by-v58.2-layered-validation"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "physical-forward-model-valid" else 1)


if __name__ == "__main__":
    main()
