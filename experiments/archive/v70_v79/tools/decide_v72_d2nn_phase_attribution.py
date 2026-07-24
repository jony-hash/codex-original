"""Joint decision for the diagnostic v72 D2NN phase attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics_pass(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("clean_prototype_corr", 0.0)) >= 0.70
        and float(metrics.get("clean_prototype_dice", 0.0)) >= 0.68
        and float(metrics.get("prototype_accuracy", 0.0)) >= 0.80
        and float(metrics.get("high_saturation", 1.0)) <= 0.08
    )


def decide(
    *,
    v69_decision: str | Path,
    v70_decision: str | Path,
    v71_decision: str | Path,
    isolation_report: str | Path,
    manual_review: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    v69 = _read(v69_decision)
    v70 = _read(v70_decision)
    v71 = _read(v71_decision)
    report = _read(isolation_report)
    manual = _read(manual_review)
    arms = report.get("arms", {})
    required = {"baseline", "encoder2_identity", "decoder1_identity", "joint_identity"}
    labels_by_arm = {
        name: sorted(int(label) for label in value.get("readable_labels", []))
        for name, value in manual.get("arms", {}).items()
    }
    provenance_pass = (
        v69.get("status") == "v69-prototype-only"
        and v70.get("status") == "post-slm-code-factor-attenuated"
        and v71.get("status") == "d2nn-code-factor-attenuated"
        and report.get("status") == "v72-phase-attribution-complete"
        and set(arms) == required
        and set(labels_by_arm) == required
        and report.get("diagnostic_only") is True
        and report.get("test_split_used") is False
        and report.get("return_optical_fields") is False
        and report.get("batch_size") == 1
        and report.get("calibration_semantics") == "slm_amplitude_squared"
        and report.get("calibration_replay_pass") is True
        and report.get("finite") is True
        and report.get("parameters_unchanged") is True
        and report.get("optimizer_created") is False
        and report.get("parameter_update_count") == 0
        and all(
            arm.get("finite") is True
            and arm.get("energy_pass") is True
            and arm.get("non_target_parameters_unchanged") is True
            and arm.get("targeted_parameters_zeroed") is True
            and arm.get("optimizer_created") is False
            and arm.get("parameter_update_count") == 0
            for arm in arms.values()
        )
    )
    auto_pass = {name: _metrics_pass(arm.get("crossed_metrics", {})) and _metrics_pass(arm.get("diagonal_metrics", {})) for name, arm in arms.items()}
    readable_pass = {name: labels == list(range(10)) for name, labels in labels_by_arm.items()}
    camera = {name: float(arm.get("camera_code_factor", 0.0)) for name, arm in arms.items()}
    complete_readable = {name: auto_pass.get(name, False) and readable_pass.get(name, False) for name in required}
    baseline = camera.get("baseline", 0.0)
    best_arm = max(required, key=lambda name: camera.get(name, 0.0))
    best_gain = camera.get(best_arm, 0.0) - baseline
    baseline_ss = float(arms.get("baseline", {}).get("gauge_aligned_complex_factor_decomposition", {}).get("decoder_layer_2", {}).get("code_effect_ss", 0.0))
    best_ss = float(arms.get(best_arm, {}).get("gauge_aligned_complex_factor_decomposition", {}).get("decoder_layer_2", {}).get("code_effect_ss", 0.0))
    gauge_retention = best_ss / max(baseline_ss, 1e-12)
    single_pass = [name for name in ("encoder2_identity", "decoder1_identity") if camera.get(name, 0.0) >= 0.25 and complete_readable.get(name, False)]
    joint_pass = camera.get("joint_identity", 0.0) >= 0.25 and complete_readable.get("joint_identity", False)
    any_factor_readability_fail = any(camera.get(name, 0.0) >= 0.25 and not complete_readable.get(name, False) for name in required)
    if not provenance_pass:
        status = "harness-invalid"
    elif single_pass:
        status = "single-layer-trained-phase-suppression-confirmed"
    elif joint_pass:
        status = "multi-layer-trained-phase-suppression-confirmed"
    elif any_factor_readability_fail:
        status = "phase-readability-code-tradeoff"
    elif best_gain >= 0.01 and gauge_retention >= 1.10:
        status = "mixed-phase-propagation-attenuation"
    else:
        status = "propagation-or-active-window-attenuation"
    result = {
        "schema": "v72_d2nn_phase_decision_v1",
        "diagnostic_only": True,
        "status": status,
        "camera_code_factor_by_arm": camera,
        "best_arm": best_arm,
        "best_camera_gain_vs_baseline": best_gain,
        "gauge_aligned_code_effect_ss_retention_vs_baseline": gauge_retention,
        "single_layer_pass_arms": single_pass,
        "joint_pass": joint_pass,
        "auto_metrics_pass_by_arm": auto_pass,
        "manual_readability_pass_by_arm": readable_pass,
        "readable_labels_by_arm": labels_by_arm,
        "provenance_pass": provenance_pass,
        "test_allowed": False,
        "fabrication_export_allowed": False,
        "phase_export_allowed": False,
        "formal_promotion_allowed": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v69-decision", "v70-decision", "v71-decision", "isolation-report", "manual-review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(decide(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
