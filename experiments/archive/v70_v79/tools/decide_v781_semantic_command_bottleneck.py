"""Decision gate for the v78.1 semantic-command bottleneck postmortem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decide(*, report: str | Path, output: str | Path) -> dict[str, Any]:
    data = _read(report)
    stages = data.get("stages", {})
    required = ("projected_pattern", "template_support_pattern", "normalized_pattern")
    structural = (
        data.get("status") == "v781-semantic-command-bottleneck-complete"
        and data.get("finite") is True
        and data.get("parameters_unchanged") is True
        and data.get("state_keys_equal") is True
        and data.get("parameter_hash_before") == data.get("parameter_hash_after")
        and data.get("optimizer_created") is False
        and data.get("parameter_update_count") == 0
        and data.get("persistent_full_field_saved") is False
        and all(float(value) <= 1.0e-6 for value in data.get("reconstruction_errors", {}).values())
        and all(name in stages and stages[name].get("finite") is True for name in required)
    )
    if not structural:
        status = "harness-invalid"
    else:
        projected = float(stages["projected_pattern"]["code_fraction"])
        supported = float(stages["template_support_pattern"]["code_fraction"])
        normalized = float(stages["normalized_pattern"]["code_fraction"])
        support_drop = projected - supported
        normalization_drop = supported - normalized
        if projected < 0.25:
            status = "basis-projected-code-insufficient"
        elif (
            supported < 0.25
            and normalized < 0.25
            and support_drop >= 0.10
            and (normalization_drop >= 0.10 or normalized <= 0.5 * supported)
        ):
            status = "mixed-support-normalization-code-suppression"
        elif supported < 0.25:
            status = "semantic-support-code-suppression"
        elif normalized < 0.25:
            status = "per-sample-normalization-code-suppression"
        else:
            status = "harness-invalid"
    result = {
        "schema": "v781_joint_decision_v1", "status": status, "diagnostic_only": True,
        "source_report": str(report), "training_allowed": False, "cuda_allowed": False,
        "smoke_allowed": False, "test_allowed": False, "fabrication_export_allowed": False,
        "phase_export_allowed": False, "formal_promotion_allowed": False,
        "current_v78_contract_closed": status in {"mixed-support-normalization-code-suppression", "semantic-support-code-suppression", "per-sample-normalization-code-suppression"},
        "learned_code_transport_proven": False,
        "next_action": "design-only label-fixed supported basis review" if status != "harness-invalid" else "repair postmortem harness and rerun complete report",
        "stage_code_fractions": {name: stages[name].get("code_fraction") for name in stages},
        "primary_absolute_loss_stage": "template_support" if status != "harness-invalid" else None,
        "normalization_effect": "ratio-artifact-not-absolute-loss" if status != "harness-invalid" else None,
        "base_template_effect": "label-denominator-dilution" if status != "harness-invalid" else None,
        "transition_metrics": data.get("transition_metrics", {}),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decide(**vars(args))


if __name__ == "__main__":
    main()
