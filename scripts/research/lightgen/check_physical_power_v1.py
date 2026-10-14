"""Run A-E as independent zero-update CPU component diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldm_project.v2.application.research.lightgen.physical_power_diagnostics import (
    absolute_power_diagnostic,
    fiber_overlap_diagnostic,
    geometry_diagnostic,
    power_conservation_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-checked-dir", type=Path, help="reuse unaffected A/B/C JSON from a prior attempt in this audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    reports = {}
    for name, run in (("geometry", geometry_diagnostic), ("power_conservation", power_conservation_diagnostic), ("fiber_overlap", fiber_overlap_diagnostic)):
        if args.reuse_checked_dir is None:
            reports[name] = run()
        else:
            reports[name] = json.loads((args.reuse_checked_dir / f"{name}.json").read_text(encoding="utf-8"))
            if name != "geometry" and reports[name]["status"] != "PASS":
                raise ValueError("only passing unaffected numerical checks may be reused")
            reports[name]["reused_from"] = str((args.reuse_checked_dir / f"{name}.json").resolve())
        with (args.output_dir / f"{name}.json").open("x", encoding="utf-8") as stream:
            json.dump(reports[name], stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"{name}: {reports[name]['status']}", flush=True)
    reports["absolute_power"], payload = absolute_power_diagnostic()
    with (args.output_dir / "absolute_power.json").open("x", encoding="utf-8") as stream:
        json.dump(reports["absolute_power"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    torch.save(payload, args.output_dir / "component_power_evidence.pt")
    numeric_valid = all(reports[name]["status"] == "PASS" for name in ("power_conservation", "fiber_overlap", "absolute_power"))
    report = {"schema": "physical_power_diagnostics.v1", "device": "cpu", "diagnostic_only": True,
              "numeric_component_checks_passed": numeric_valid,
              "target_facet_geometry_supported": reports["geometry"]["status"] == "PASS",
              "production_entrypoint_probe": reports["absolute_power"]["production_entrypoint_probe"],
              "scientific_updates": 0, "evaluator_calls": 0,
              "status": "DIAGNOSTICS_COMPLETED" if numeric_valid else "INVALID_DIAGNOSTIC",
              "reports": {name: f"{name}.json" for name in reports}}
    with (args.output_dir / "diagnostics.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    reread = json.loads((args.output_dir / "absolute_power.json").read_text(encoding="utf-8"))
    reloaded = torch.load(args.output_dir / "component_power_evidence.pt", map_location="cpu", weights_only=True)
    if any(reloaded["basis_results"][basis]["channel_power_w"].tolist() != values["channel_power_w_each"] for basis, values in reread["basis_results"].items()):
        raise ValueError("saved channel power reread mismatch")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if numeric_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
