from __future__ import annotations

import inspect

import torch

from scripts.research import execute_lightgen_2d_establish_third_confirmation_evaluator_v1 as consumer


def test_fallback_architecture_is_fixed_and_distinct() -> None:
    model = consumer.ThirdConfirmationEvaluator()
    assert model.features[0].kernel_size == (5, 5)
    assert model.features[3].kernel_size == (5, 5)
    assert model.classifier.out_features == 10
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert consumer._architecture_payload()["layers"][0]["kernel"] == 5


def test_ec_split_is_deterministic_and_class_balanced() -> None:
    labels = torch.arange(10).repeat_interleave(100)
    first_fit, first_qualification = _split_for_test(labels)
    second_fit, second_qualification = _split_for_test(labels)
    assert torch.equal(first_fit, second_fit)
    assert torch.equal(first_qualification, second_qualification)
    assert first_fit.numel() == 900
    assert first_qualification.numel() == 100
    assert torch.bincount(labels.index_select(0, first_qualification), minlength=10).tolist() == [10] * 10
    assert not bool(torch.isin(first_fit, first_qualification).any().item())


def _split_for_test(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(123)
    fit_parts = []
    qualification_parts = []
    for class_id in range(10):
        matches = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        qualification_parts.append(shuffled[:10])
        fit_parts.append(shuffled[10:])
    return torch.cat(fit_parts).sort().values, torch.cat(qualification_parts).sort().values


def test_entrypoint_contains_no_optical_or_generated_cmos_dependency() -> None:
    source = inspect.getsource(consumer)
    assert "ldm_project.v2.optics" not in source
    assert "LightGen2D" not in source
    assert "load_generated_cmos" not in source
    assert "load_protected" not in source
