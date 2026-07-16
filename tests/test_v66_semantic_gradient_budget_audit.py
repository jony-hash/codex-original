from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v66_semantic_gradient_budget_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("v66_gradient_budget", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _images(count: int) -> torch.Tensor:
    return torch.arange(count * 200 * 200, dtype=torch.float32).reshape(count, 1, 200, 200) / 100000.0


def test_template_decomposition_is_train_only_and_exact():
    module = _module()
    train = _images(20)
    train_labels = torch.arange(20) % 10
    test = _images(10) + 0.25
    test_labels = torch.arange(10)
    templates, meta = module._template_decomposition(train, train_labels, test, test_labels)
    assert meta["target_direction_decomposition_max_abs"] < 1.0e-6
    assert torch.equal(templates["label_templates"], torch.stack([train[train_labels == k].mean(0) for k in range(10)]))
    residuals = module.residual_components(
        test * 0.0,
        test,
        templates["global_template"],
        templates["selected_label_template"],
    )
    assert torch.allclose(residuals["common"] + residuals["label"] + residuals["instance"], -test)


def test_weighted_vjp_decomposition_matches_full_residual():
    module = _module()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    basis = torch.ones(1, 1, 200, 200)
    target = torch.full_like(basis, 0.2)
    global_template = torch.full_like(basis, 0.1)
    label_template = torch.full_like(basis, 0.15)
    prediction = parameter * basis
    residuals = module.residual_components(prediction, target, global_template, label_template)
    gradients = []
    for name in ("common", "label", "instance"):
        value = module._weighted_dot(prediction, residuals[name].detach(), target)
        gradients.append(torch.autograd.grad(value, parameter, retain_graph=True)[0])
    full_value = module._weighted_dot(prediction, (prediction - target).detach(), target)
    full_gradient = torch.autograd.grad(full_value, parameter)[0]
    assert torch.allclose(sum(gradients), full_gradient)


def test_streaming_accumulator_reports_cancellation_and_cosines():
    module = _module()
    accumulator = module._GradientAccumulator(2)
    accumulator.add([torch.tensor([1.0, 0.0])], 0, 1.0)
    accumulator.add([torch.tensor([-1.0, 0.0])], 0, 1.0)
    accumulator.add([torch.tensor([0.0, 1.0])], 1, 1.0)
    accumulator.add([torch.tensor([0.0, -1.0])], 1, 1.0)
    result = accumulator.finish()
    assert result["cancellation_ratio"] == 0.0
    assert result["same_class_mean_cosine"] == -1.0
    assert result["cross_class_mean_cosine"] == 0.0


def _summary(label: float, instance: float, cancellation: float, cross_cosine: float):
    groups = ("generator_e", "encoder", "decoder")
    summary = {component: {} for component in ("common", "label", "instance", "semantic")}
    for group in groups:
        summary["common"][group] = {"normalized_drive": 1.0, "drive_ratio_to_common": 1.0}
        summary["label"][group] = {"normalized_drive": label, "drive_ratio_to_common": label}
        summary["instance"][group] = {"normalized_drive": instance, "drive_ratio_to_common": instance}
        summary["semantic"][group] = {
            "normalized_drive": max(label, instance),
            "drive_ratio_to_common": max(label, instance),
            "cancellation_ratio": cancellation,
            "cross_class_mean_cosine": cross_cosine,
        }
    return summary


def test_registered_decision_branches():
    module = _module()
    assert module.decide_gradient(_summary(0.05, 0.05, 0.5, 0.5))["status"] == "semantic-gradient-suppression-candidate"
    assert module.decide_gradient(_summary(0.4, 0.4, 0.05, -0.1))["status"] == "batch-cancellation-candidate"
    assert module.decide_gradient(_summary(0.4, 0.4, 0.5, 0.1))["status"] == "semantic-gradient-available-candidate"
    assert module.decide_gradient(_summary(0.2, 0.4, 0.5, 0.1))["status"] == "mixed-semantic-gradient-evidence"


def test_source_forbids_training_and_spatial_readout_interventions():
    source = TOOL.read_text(encoding="utf-8").lower()
    assert "interpolate(" not in source
    assert "adaptive_avg_pool" not in source
    assert "grid_sample" not in source
    assert "pca" not in source
    assert "optimizer" not in source.replace("optimizer_created", "")
    assert "target_code_bridge" not in source
