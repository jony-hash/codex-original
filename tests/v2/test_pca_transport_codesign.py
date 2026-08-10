import torch

from ldm_project.v2.config import load_config
from ldm_project.v2.student.generator import ConditionalPCATransportCodesignGenerator
from ldm_project.v2.application.conditional_optical_feasibility import FEASIBILITY_CONTRACT
from ldm_project.v2.application.conditional_pca_transport import PCA_TRANSPORT_REVALIDATION_CONTRACT


def test_pca_transport_config_and_bounded_scales() -> None:
    config = load_config(
        "configs/v2/diagnostic/conditional_pca_transport_codesign_finite_crop_identity.yaml"
    )
    assert config.architecture == "conditional_pca_transport_codesign_v6"
    generator = ConditionalPCATransportCodesignGenerator()
    scales = generator.scales()
    assert torch.allclose(
        scales,
        torch.full_like(scales, 0.05 / torch.tanh(torch.tensor(1.0))),
        atol=1.0e-7,
    )
    assert torch.allclose(
        scales * torch.tanh(torch.tensor(1.0)),
        torch.full_like(scales, 0.05),
        atol=1.0e-7,
    )
    assert bool((scales > 0).all() and (scales < 0.20).all())
    assert bool((scales.square().sum(1).sqrt() <= 0.30 + 1.0e-6).all())


def test_pca_transport_zero_code_is_exact_carrier_and_lowpass_is_applied() -> None:
    generator = ConditionalPCATransportCodesignGenerator()
    labels = torch.arange(10)
    command = generator(torch.zeros(10, 8), labels)
    assert torch.equal(command, generator.class_carrier_command(labels))
    modes = generator.lowpass_normalized_modes()
    assert modes.shape == (10, 8, 200, 200)
    assert torch.allclose(modes.mean(dim=(-2, -1)), torch.zeros(10, 8), atol=1.0e-5)
    assert torch.allclose(
        modes.square().mean(dim=(-2, -1)), torch.ones(10, 8), atol=1.0e-4
    )


def test_pca_transport_has_no_hidden_coefficient_network() -> None:
    generator = ConditionalPCATransportCodesignGenerator()
    names = set(dict(generator.named_parameters()))
    assert names == {"class_carrier_logits", "class_modes", "scale_logits"}
    labels = torch.tensor([2, 2])
    codes = torch.zeros(2, 8)
    codes[1, 3] = 1.0
    assert not torch.equal(generator(codes, labels)[0], generator(codes, labels)[1])


def test_revalidation_and_macro_review_are_diagnostic_contracts() -> None:
    assert PCA_TRANSPORT_REVALIDATION_CONTRACT
    assert FEASIBILITY_CONTRACT
    generator = ConditionalPCATransportCodesignGenerator()
    # The fixed low-pass mask is a buffer, not one of the five trainable v6
    # groups that a smoke revalidation may require to move.
    assert "lowpass_mask" not in dict(generator.named_parameters())
