"""Joint decision for the v78 semantic-command contract admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decide(*, v77_decision: str | Path, admission: str | Path, smoke: str | Path | None, output: str | Path) -> dict[str, Any]:
    v77 = _read(v77_decision)
    cpu = _read(admission)
    sm = _read(smoke) if smoke else None
    checks = {
        "v77_admissible": v77.get("status") == "phase-aware-local-co-design-admissible",
        "cpu_admission": cpu.get("status") == "v78-cpu-admission-passed",
        "smoke": sm is not None and sm.get("status") == "v78-semantic-command-smoke-passed",
        "finite": bool(cpu.get("metrics", {}).get("finite", False)) and (sm is None or bool(sm.get("finite", False))),
        "frozen": bool(cpu.get("checks", {}).get("parameters_unchanged", False)) and (sm is None or bool(sm.get("template_embedding_frozen", False))),
    }
    cpu_checks = cpu.get("checks", {})
    structural_keys = (
        "v77_status", "validation_contract", "checkpoint_contract",
        "checkpoint_state_compatible", "formula", "finite",
        "parameters_unchanged",
    )
    structural_ok = all(bool(cpu_checks.get(key, False)) for key in structural_keys)
    structural_ok = structural_ok and not bool(cpu_checks.get("optimizer_created", True))
    structural_ok = structural_ok and not bool(cpu_checks.get("calibration_or_sweep", True))
    metrics = cpu.get("metrics", {})
    if not all(checks.values()):
        if not structural_ok or not checks["v77_admissible"] or not checks["finite"] or not checks["frozen"]:
            status = "harness-invalid"
        elif metrics.get("physical_slm_intensity_code_factor", 0.0) < 0.25:
            status = "v78-command-code-factor-insufficient"
        elif any(metrics.get(key, 0.0) < threshold for key, threshold in (("support_energy_fraction", 0.95), ("low_frequency_energy_fraction", 0.99))) or metrics.get("one_clipping_fraction", 1.0) > 0.08:
            status = "v78-command-support-or-band-invalid"
        else:
            status = "v78-residual-phase-gradient-invalid"
    else:
        status = "v78-semantic-command-contract-admitted"
    report = {
        "schema": "v78_joint_decision_v1", "status": status, "diagnostic_only": True,
        "checks": checks, "metrics": metrics, "coherent_receiver_branch_closed": True,
        "learned_code_transport_proven": False, "training_allowed": False, "test_allowed": False,
        "fabrication_export_allowed": False, "phase_export_allowed": False, "formal_promotion_allowed": False,
        "next_action": (
            "design-only future v78/v79 training execution plan"
            if status == "v78-semantic-command-contract-admitted"
            else "freeze v78 contract; do not run CUDA smoke or training"
        ),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v77-decision", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--smoke")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decide(v77_decision=args.v77_decision, admission=args.admission, smoke=args.smoke, output=args.output)


if __name__ == "__main__":
    main()
