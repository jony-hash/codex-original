from __future__ import annotations

import json

import torch

from tools.decide_v68_b_acceptance import run as decide_b
from tools.audit_v68_raw_cmos import fixed_exposure_view
from tools.v68_b_acceptance_utils import balanced_factor_decomposition, metric_bundle


def test_balanced_factor_decomposition_identifies_code_and_interaction():
    code = torch.arange(100, dtype=torch.float32).view(100, 1, 1, 1, 1)
    label = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1, 1)
    factors, _ = balanced_factor_decomposition(code.expand(100, 10, 1, 1, 1))
    assert factors["code_fraction"] > 0.99
    factors, _ = balanced_factor_decomposition(label.expand(100, 10, 1, 1, 1))
    assert factors["label_fraction"] > 0.99
    factors, _ = balanced_factor_decomposition(
        ((code - 49.5) * (label - 4.5)).expand(100, 10, 1, 1, 1)
    )
    assert factors["interaction_fraction"] > 0.99


def test_locked_preview_uses_one_global_scale_without_percentile_enhancement():
    raw = torch.tensor([[[[0.0, 0.5], [1.0, 2.0]]]])
    preview = fixed_exposure_view(raw, 2.0)
    assert torch.equal(preview, torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]]))


def test_metric_bundle_uses_ten_class_prototype_bank_for_accuracy():
    prototypes = torch.eye(10, dtype=torch.float32).reshape(10, 1, 2, 5)
    labels = torch.tensor([0, 5, 9])
    prediction = prototypes[labels]
    metrics = metric_bundle(prediction, prediction, labels, prototypes)
    assert metrics["prototype_accuracy"] == 1.0


def test_b_decision_three_outcomes_and_harness_invalid(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    a2 = write("a2.json", {"status": "a2-raw-readability-passed", "b_allowed": True})
    smoke = write(
        "smoke.json",
        {
            "schema": "v68_b_warmstart_smoke_v1",
            "status": "b-warmstart-smoke-passed",
            "finite": True,
            "parameters_unchanged": True,
        },
    )
    history = write(
        "history.json",
        {
            "one_shot_distill_validation": [
                {"epoch": 4, "paired_corr": 0.8, "paired_dice": 0.8},
                {"epoch": 5, "paired_corr": 0.9, "paired_dice": 0.9},
            ]
        },
    )
    manual = write(
        "manual.json", {"readable_labels": list(range(10)), "notes": "reviewed"}
    )

    def crossed(factor=0.3, corr=0.8):
        return {
            "schema": "v68_b_crossed_raw_cmos_v1",
            "status": "b-crossed-audit-complete",
            "test_split_used": False,
            "finite": True,
            "parameters_unchanged": True,
            "camera_code_factor": factor,
            "energy_records_mean": {"guard_energy_ratio": 1e-6},
            "crossed_metrics": {
                "clean_prototype_corr": corr,
                "clean_prototype_dice": 0.8,
                "prototype_accuracy": 0.9,
                "high_saturation": 0.01,
            },
            "diagonal_metrics": {
                "clean_prototype_corr": corr,
                "clean_prototype_dice": 0.8,
                "prototype_accuracy": 0.9,
                "high_saturation": 0.01,
            },
        }

    out = tmp_path / "decision.json"
    report = decide_b(
        a2_decision=a2,
        warmstart_report=smoke,
        training_history=history,
        crossed_report=write("crossed.json", crossed()),
        manual_review=manual,
        output=out,
    )
    assert report["status"] == "b-code-conditioned-readability-passed"
    report = decide_b(
        a2_decision=a2,
        warmstart_report=smoke,
        training_history=history,
        crossed_report=write("crossed2.json", crossed(0.1)),
        manual_review=manual,
        output=out,
    )
    assert report["status"] == "b-prototype-imaging-only"
    report = decide_b(
        a2_decision=a2,
        warmstart_report=smoke,
        training_history=history,
        crossed_report=write("crossed3.json", crossed(0.3, 0.2)),
        manual_review=manual,
        output=out,
    )
    assert report["status"] == "b-readability-regression"
    report = decide_b(
        a2_decision=a2,
        warmstart_report=write("bad_smoke.json", {"schema": "bad"}),
        training_history=history,
        crossed_report=write("crossed4.json", crossed()),
        manual_review=manual,
        output=out,
    )
    assert report["status"] == "harness-invalid"


def test_training_utils_validation_import_is_canonical():
    from ldm_project.student.training_utils import _per_class_label_audit_metrics

    result, counts = _per_class_label_audit_metrics(
        torch.zeros(2, 1, 4, 4), torch.tensor([0, 1]), torch.zeros(2, 1, 4, 4), 2
    )
    assert counts["per_class_0_count"] == 1
    assert "per_class_0_label_correct_count" in result
    assert "per_class_1_label_margin_sum" in result
