from __future__ import annotations

import inspect

import torch

from ldm_project.v2.application.research.lightgen.confirmation_evaluators import PixelMLPConfirmationEvaluatorV1
from scripts.research.lightgen import execute_lightgen_2d_establish_sic_independent_confirmation_evaluator_v1 as consumer


def test_pixel_mlp_architecture_is_fixed_and_nonconvolutional() -> None:
    model = consumer.PixelMLPConfirmationEvaluatorV1()
    assert consumer.PixelMLPConfirmationEvaluatorV1 is PixelMLPConfirmationEvaluatorV1
    assert consumer._architecture_payload()["architecture_kind"] == "pixel_mlp_confirmation_evaluator_v1"
    assert [module.out_features for module in model.network if isinstance(module, torch.nn.Linear)] == [512, 256, 10]
    assert set(model.state_dict()) == {
        "network.1.weight", "network.1.bias", "network.3.weight", "network.3.bias", "network.5.weight", "network.5.bias"
    }
    assert not any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    assert not any(isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.BatchNorm2d, torch.nn.Dropout)) for module in model.modules())
    logits = model(torch.zeros(2, 1, 28, 28))
    assert logits.shape == (2, 10)


def test_ed_split_is_deterministic_and_class_balanced() -> None:
    labels = torch.arange(10).repeat_interleave(100)
    parent_train = torch.arange(labels.numel())
    generator_a = torch.Generator(device="cpu").manual_seed(2026092003)
    generator_b = torch.Generator(device="cpu").manual_seed(2026092003)

    def split(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        fit_parts = []
        qualification_parts = []
        for class_id in range(10):
            matches = torch.nonzero(labels == class_id, as_tuple=False).flatten()
            shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
            qualification_parts.append(shuffled[:10])
            fit_parts.append(shuffled[10:])
        return torch.cat(fit_parts).sort().values, torch.cat(qualification_parts).sort().values

    fit_a, qualification_a = split(generator_a)
    fit_b, qualification_b = split(generator_b)
    assert torch.equal(fit_a, fit_b)
    assert torch.equal(qualification_a, qualification_b)
    assert fit_a.numel() == 900
    assert qualification_a.numel() == 100
    assert torch.bincount(labels.index_select(0, qualification_a), minlength=10).tolist() == [10] * 10
    assert not bool(torch.isin(fit_a, qualification_a).any().item())
    assert torch.equal(torch.cat((fit_a, qualification_a)).sort().values, parent_train)


def test_architecture_audit_rejects_no_material_match() -> None:
    audit = consumer._architecture_independence_audit(consumer._root())
    assert audit["status"] == "PASS"
    assert audit["material_matches"] == []
    assert all(item["family"] == "convolutional" for item in audit["already_consumed"].values())


def test_entrypoint_has_no_optical_or_generator_dependency() -> None:
    source = inspect.getsource(consumer)
    assert "LightGen2D" not in source
    assert "ldm_project.v2.optics" not in source
    assert "load_generated_cmos" not in source
    assert "outputs/20260920_evaluator_robust_semantic_training_v1" not in source
