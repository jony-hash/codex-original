from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v62_native_conditional_control_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("v62_conditional_control", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labels(count_per_class: int = 3) -> torch.Tensor:
    return torch.arange(10).repeat_interleave(count_per_class)


def _payload(split: str, *, count_per_class: int) -> dict[str, object]:
    labels = _labels(count_per_class)
    return {
        "codes": torch.zeros(labels.numel(), 128),
        "class_labels": labels,
        "teacher_camera": torch.zeros(labels.numel(), 1, 200, 200),
        "readability_split": split,
        "config": {"filter": {"teacher_filter": "readability_strict_split"}},
        "label_audit_nearest_label": labels.clone(),
        "label_audit_margin": torch.ones(labels.numel()),
    }


def test_target_rotation_is_deterministic_and_never_keeps_a_within_class_target():
    module = _module()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    target = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
    rotated = module.deterministic_within_class_target_rotation(target, labels)
    assert torch.equal(rotated.flatten(), torch.tensor([2.0, 0.0, 1.0, 5.0, 3.0, 4.0]))
    assert not torch.equal(rotated, target)


def test_leave_one_out_residual_removes_a_class_template_without_self_leakage():
    module = _module()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    values = torch.tensor([1.0, 3.0, 5.0, 10.0, 12.0, 14.0]).view(6, 1, 1, 1)
    residual = module.leave_one_out_class_residual(values, labels).flatten()
    assert torch.allclose(residual, torch.tensor([-3.0, 0.0, 3.0, -3.0, 0.0, 3.0]))


def test_paired_residual_rotation_reports_a_positive_confidence_interval_for_matched_signal():
    module = _module()
    labels = _labels(3)
    target = torch.zeros(labels.numel(), 1, 4, 4)
    for index in range(target.shape[0]):
        target[index, 0, index % 4, (index // 4) % 4] = 1.0
    rotated = module.deterministic_within_class_target_rotation(target, labels)
    matched = module._residual_metrics(target, target, labels)
    control = module._residual_metrics(target, rotated, labels)
    delta = module.paired_bootstrap_delta(matched["per_sample_corr"], control["per_sample_corr"], seed=7, iterations=128)
    assert matched["corr_mean"] > control["corr_mean"]
    assert delta["ci_low"] > 0.0


def test_decision_distinguishes_template_only_and_code_conditioned_instance_signal():
    module = _module()

    def controls(*, label_p: float, response: float, code_energy: float):
        return {
            "label_only": {"prototype_cv": {"p_value": label_p}},
            "code_only": {"within_class_residual_energy": code_energy},
            "baseline_vs_label_only": {"mean": response},
        }

    def pairing(*, corr: float, ci_low: float):
        return {
            "matched_residual": {"corr_mean": corr},
            "residual_corr_vs_within_class_target_rotation": {"ci_low": ci_low},
        }

    template = module._decision(controls(label_p=0.001, response=0.0, code_energy=0.0), pairing(corr=0.0, ci_low=-0.1))
    assert template["status"] == "class-template-only-camera-decodability-candidate"
    instance = module._decision(controls(label_p=0.001, response=0.1, code_energy=0.1), pairing(corr=0.5, ci_low=0.1))
    assert instance["status"] == "camera-intensity-contains-code-conditioned-instance-residual"


def test_upper_bound_validation_rejects_mismatched_test_tensor(tmp_path: Path):
    module = _module()
    report_path = tmp_path / "upper.json"
    tensor_path = tmp_path / "upper.pt"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    payload = _payload("test", count_per_class=module.SPLIT_CLASS_COUNTS["test"])
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "v61-offline-linear-decodability-upper-bound",
                "decision": {"status": "camera-intensity-decodable"},
                "diagnostic_only": True,
                "accepted_summary_written": False,
                "inputs": {"checkpoint": str(checkpoint)},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "schema_version": "v61-offline-linear-decodability-upper-bound",
            "diagnostic_only": True,
            "accepted_summary_written": False,
            "class_labels": payload["class_labels"].clone(),
            "teacher_target": payload["teacher_camera"].clone(),
            "decoded_targets": {"camera_intensity": torch.zeros(100, 1, 200, 200)},
        },
        tensor_path,
    )
    reasons, decoded = module._validate_upper_bound(report_path, tensor_path, checkpoint=checkpoint, test_payload=payload)
    assert reasons == []
    assert tuple(decoded.shape) == (100, 1, 200, 200)
    payload["class_labels"][0] = 9
    reasons, _ = module._validate_upper_bound(report_path, tensor_path, checkpoint=checkpoint, test_payload=payload)
    assert any("labels" in reason for reason in reasons)


def test_state_snapshot_detects_any_parameter_update():
    module = _module()
    model = torch.nn.Linear(2, 2)
    before = module._state_snapshot(model)
    assert module._state_changes(model, before) == []
    with torch.no_grad():
        model.bias.add_(1.0)
    assert module._state_changes(model, before) == ["bias"]


def test_tool_uses_native_controls_without_resize_or_trainable_readout():
    source = TOOL.read_text(encoding="utf-8")
    assert "interpolate(" not in source
    assert "adaptive_avg_pool" not in source
    assert "optimizer" not in source.lower().replace("optimizer_created", "")
