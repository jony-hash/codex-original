"""Decision gate for the v79 label-fixed supported-basis CPU upper bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decide(*, report: str | Path, output: str | Path) -> dict[str, Any]:
    data = _read(report)
    arms = data.get("arms", {})
    learned = arms.get("learned_replay", {})
    capacity = arms.get("deterministic_capacity", {})
    structural = (
        data.get("status") == "v79-label-fixed-supported-basis-audit-complete"
        and data.get("finite") is True
        and data.get("parameters_unchanged") is True
        and data.get("state_keys_equal") is True
        and data.get("parameter_hash_before") == data.get("parameter_hash_after")
        and data.get("checkpoint_state_keys_compatible") is True
        and data.get("optimizer_created") is False
        and data.get("parameter_update_count") == 0
        and data.get("optical_forward_used") is False
        and data.get("cuda_used") is False
        and data.get("persistent_full_field_saved") is False
        and data.get("checkpoint_boundary_contract") == "ideal_transparent_thin_phase_v2"
        and all(arm.get("finite") is True for arm in (learned, capacity))
        and all(
            max(float(value) for value in arm.get("formula_errors", {}).values()) <= 1.0e-6
            for arm in (learned, capacity)
        )
    )
    capacity_checks = {
        "code_factor": float(capacity.get("code_fraction", 0.0)) >= 0.25,
        "absolute_code_ss": float(capacity.get("code_effect_ss", 0.0)) >= 289721.29211025406,
        "label_ss": float(capacity.get("label_effect_ss", 0.0)) >= 0.90 * 757421.0198058158,
        "support_energy": float(capacity.get("prototype_support_command_delta_energy_fraction", 0.0)) >= 0.95,
        "low_frequency": float(capacity.get("low_frequency_energy_fraction", 0.0)) >= 0.99,
        "clipping": float(capacity.get("one_clipping_fraction", 1.0)) <= 0.08,
        "bounded_q": float(capacity.get("max_abs_q", 2.0)) <= 1.0 + 1.0e-6,
    }
    learned_pass = (
        float(learned.get("code_fraction", 0.0)) >= 0.25
        and float(learned.get("code_effect_ss", 0.0)) >= 289721.29211025406
        and float(learned.get("label_effect_ss", 0.0)) >= 0.90 * 757421.0198058158
        and float(learned.get("prototype_support_command_delta_energy_fraction", 0.0)) >= 0.95
        and float(learned.get("low_frequency_energy_fraction", 0.0)) >= 0.99
        and float(learned.get("one_clipping_fraction", 1.0)) <= 0.08
    )
    if not structural:
        status = "harness-invalid"
    elif not capacity_checks["code_factor"] or not capacity_checks["absolute_code_ss"] or not capacity_checks["label_ss"]:
        status = "label-fixed-supported-basis-insufficient"
    elif not capacity_checks["low_frequency"] or not capacity_checks["support_energy"] or not capacity_checks["clipping"] or not capacity_checks["bounded_q"]:
        status = "label-fixed-basis-physical-contract-invalid"
    elif not learned_pass:
        status = "label-fixed-basis-capable-warmstart-insufficient"
    else:
        status = "label-fixed-supported-basis-capable"
    result = {
        "schema": "v79_joint_decision_v1",
        "status": status,
        "diagnostic_only": True,
        "source_report": str(report),
        "capacity_checks": capacity_checks,
        "learned_replay_pass": learned_pass,
        "training_allowed": False,
        "cuda_allowed": False,
        "smoke_allowed": False,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
        "label_fixed_basis_contract_closed": status in {
            "label-fixed-supported-basis-insufficient",
            "label-fixed-basis-physical-contract-invalid",
        },
        "learned_code_transport_proven": False,
        "next_action": (
            "design-only coefficient-contract review"
            if status == "label-fixed-basis-capable-warmstart-insufficient"
            else "freeze semantic-supported basis family; do not train"
            if status == "label-fixed-supported-basis-insufficient"
            else "repair v79 harness and rerun complete CPU audit"
            if status == "harness-invalid"
            else "retain diagnostic-only capability result"
        ),
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
