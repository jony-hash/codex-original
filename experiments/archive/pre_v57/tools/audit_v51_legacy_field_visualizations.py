#!/usr/bin/env python3
"""Mark v48-v50 mixed amplitude/phase plots as non-physical visualization evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


LEGACY_ROOTS = (
    Path("outputs/20260710_v48"),
    Path("outputs/20260711_v49"),
    Path("outputs/20260711_v50"),
)
LEGACY_FILENAMES = (
    "one_shot_sample_amplitude.png",
    "one_shot_sample_phase.png",
    "one_shot_sample_neuron_lattice_amplitude.png",
    "one_shot_sample_100x100_cell_chain_preview.png",
)


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"v51_legacy_optical_field_visualization_correction_{timestamp}.json"


def build_report() -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for root in LEGACY_ROOTS:
        for run_dir in sorted((root / "runs").glob("*")) if (root / "runs").is_dir() else ():
            images = [
                str((run_dir / "images" / filename).as_posix())
                for filename in LEGACY_FILENAMES
                if (run_dir / "images" / filename).is_file()
            ]
            if not images:
                continue
            runs.append(
                {
                    "run": run_dir.as_posix(),
                    "legacy_field_images": images,
                    "visualization_status": "visualization-harness-invalid",
                    "reason": (
                        "legacy full-chain amplitude/phase images can mix controls, target-code intervention, "
                        "per-plane normalization, or unavailable phase; they are not optical-field evidence"
                    ),
                    "cmos_only_verdict_preserved": True,
                }
            )
    return {
        "version": "v51",
        "status": "visualization-harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "runs": runs,
        "replacement": "v51 strict optical-field provenance artifacts",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=_default_output())
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
