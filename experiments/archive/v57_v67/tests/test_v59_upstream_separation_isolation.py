from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v59_upstream_separation_isolation.py"


def _module():
    spec = importlib.util.spec_from_file_location("v59_upstream", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derangements_are_deterministic_and_preserve_class_multiset():
    module = _module()
    labels = torch.arange(10).repeat_interleave(3)
    codes = torch.arange(30 * 4, dtype=torch.float32).reshape(30, 4)
    shifted = module.deterministic_label_derangement(labels)
    rotated = module.deterministic_within_class_code_rotation(codes, labels)
    assert torch.equal(shifted, labels.add(1).remainder(10))
    assert torch.equal(torch.sort(rotated, dim=0).values, torch.sort(codes, dim=0).values)
    assert torch.equal(rotated[labels == 4][0], codes[labels == 4][-1])


def test_balanced_dataset_validation_rejects_wrong_split_and_counts():
    module = _module()
    labels = torch.arange(10).repeat_interleave(2)
    payload = {
        "codes": torch.zeros(20, 128),
        "class_labels": labels,
        "teacher_camera": torch.zeros(20, 1, 200, 200),
        "readability_split": "train",
        "config": {"filter": {"teacher_filter": "readability_strict_split"}},
        "label_audit_nearest_label": labels,
        "label_audit_margin": torch.ones(20),
    }
    reasons = module._validate_dataset(payload)
    assert any("exactly 10" in reason for reason in reasons)
    assert any("readability_split" in reason for reason in reasons)


def test_synthetic_separation_and_permutation_null_are_detectable():
    module = _module()
    labels = torch.arange(10).repeat_interleave(10)
    values = torch.zeros(100, 1, 10, 10)
    for index, label in enumerate(labels):
        values[index, 0, int(label) % 10, int(label) % 10] = 1.0
    result = module.permutation_null(values, labels, permutations=16, seed=20260714)
    assert result["observed_accuracy"] == pytest.approx(1.0)
    assert result["p_value"] < 0.2


def test_parameter_checksum_logic_has_no_mutation_for_readonly_forward():
    module = _module()
    values = torch.ones(20, 1, 2, 2)
    labels = torch.arange(10).repeat_interleave(2)
    summary = module.summarize_plane(values, labels)
    assert summary["same_class_pair_count"] == 10
    assert summary["cross_class_pair_count"] == 180


def _decision_payload(module, *, teacher_p=0.001, slm_p=0.001, front_p=0.001, slm_signal=0.05, front_signal=0.02, outside=0.1, label_effect=0.005, code_effect=0.02, label_mse_delta=0.001, code_mse_delta=0.001):
    teacher = {"prototype_cv": {"p_value": teacher_p, "observed_accuracy": 1.0, "null_p95": 0.2}, "separation": {"separation_signal": {"complex": 1.0}}}
    planes = {
        "baseline": {
            "slm_command": {
                "prototype_cv": {"p_value": slm_p},
                "separation": {"separation_signal": {"complex": slm_signal}},
            },
            "pre_encoder_relay_image": {
                "prototype_cv": {"p_value": front_p},
                "separation": {"separation_signal": {"complex": front_signal}},
            },
        }
    }
    sensitivities = {"label_shift": {"slm_command": label_effect}, "code_rotation": {"slm_command": code_effect}}
    target_metrics = {"baseline": {"mse_mean": 1.0}, "label_shift": {"mse_mean": 1.0 + label_mse_delta}, "code_rotation": {"mse_mean": 1.0 + code_mse_delta}}
    spectra = {"slm_command": {"between_out_of_passband_fraction": outside}}
    return module._decision(teacher, planes, sensitivities, target_metrics, spectra)


def test_decision_branches_cover_teacher_bandwidth_generator_and_objective():
    module = _module()
    assert _decision_payload(module, teacher_p=0.2)["status"] == "teacher-target-separation-weak"
    assert _decision_payload(module, front_signal=0.004, outside=0.8)["status"] == "generator-relay-bandwidth-mismatch-candidate"
    assert _decision_payload(module, slm_p=0.2, slm_signal=0.05, label_effect=0.005, code_effect=0.02)["status"] == "generator-label-conditioning-or-average-template"
    assert _decision_payload(module, slm_p=0.2, slm_signal=0.05, label_effect=0.02, code_effect=0.005, label_mse_delta=0.02)["status"] == "generator-code-conditioning-instance-collapse"
    assert _decision_payload(module, slm_p=0.001, slm_signal=0.3, front_signal=0.2, label_effect=0.02, code_effect=0.02, label_mse_delta=0.02)["status"] == "objective-representation-mismatch-candidate"
