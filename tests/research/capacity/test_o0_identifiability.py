"""Small CPU contracts for the v2.4 update-boundary lifecycle."""

from __future__ import annotations

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.capacity.o0_identifiability_objectives import discriminator_q_loss, frozen_parameters, generator_identifiability_loss, parameter_hash
from ldm_project.v2.application.research.capacity.o0_identifiability_spec import load_identifiability_spec
from ldm_project.v2.application.research.capacity.o0_models import D0ImageGenerator, ProjectionDiscriminator, inference_state_dict


def test_identifiability_cli_and_contract_are_independent() -> None:
    parsed = _parser().parse_args(
        [
            "research",
            "o0-identifiability",
            "audit-update",
            "--config",
            "configs/research/capacity/conditional_o0_identifiability_v2_4.yaml",
            "--device",
            "cpu",
        ]
    )
    assert parsed.research_program == "o0-identifiability"
    assert load_identifiability_spec(parsed.config).probe_steps == 100


def test_dq_and_generator_parameter_boundaries() -> None:
    torch.manual_seed(1234)
    generator, discriminator = (
        D0ImageGenerator(),
        ProjectionDiscriminator(code_dim=32, image_size=28),
    )
    gopt = torch.optim.Adam(generator.parameters(), lr=2e-4)
    dopt = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    real, labels, codes = (
        torch.rand(10, 1, 28, 28),
        torch.arange(10),
        torch.randn(10, 32),
    )
    generator_before, discriminator_before = (
        parameter_hash(generator),
        parameter_hash(discriminator),
    )
    fake = generator(codes, labels).detach()
    dloss, _ = discriminator_q_loss(discriminator, real, fake, labels, codes)
    dloss.backward()
    dopt.step()
    assert parameter_hash(discriminator) != discriminator_before
    assert parameter_hash(generator) == generator_before
    discriminator.zero_grad(set_to_none=True)
    gopt.zero_grad(set_to_none=True)
    with frozen_parameters(discriminator):
        gloss, _ = generator_identifiability_loss(
            discriminator, generator(codes, labels), labels, codes
        )
        gloss.backward()
    gopt.step()
    assert parameter_hash(generator) != generator_before
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_inference_state_is_generator_only() -> None:
    state = inference_state_dict(D0ImageGenerator())
    assert state and all(
        "discriminator" not in key and "latent_head" not in key for key in state
    )
