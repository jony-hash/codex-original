from __future__ import annotations

import pytest
import torch

from ldm_project.v2.config import load_config
from ldm_project.v2.student.generator import (
    ConditionalClassConditionedModesGenerator,
    ConditionalGlobalAmplitudeGenerator,
    ConditionalParityStabilizedModesGenerator,
    ConditionalMultiAnchorLocalModesGenerator,
)
from ldm_project.v2.application.conditional import (
    _reanchor_codes,
)


def test_conditional_generator_is_bounded_zero_code_carrier_and_global() -> None:
    generator = ConditionalGlobalAmplitudeGenerator(shape=(8, 8))
    labels = torch.tensor([2, 7])
    zero = torch.zeros(2, 8)
    command = generator(zero, labels)
    assert command.shape == (2, 1, 8, 8)
    assert bool(((command > 0.0) & (command < 1.0)).all())
    assert torch.equal(command, generator.class_carrier_command(labels))
    changed = generator(torch.tensor([[0.8] * 8, [-0.8] * 8]), labels)
    assert not torch.equal(changed, command)
    assert generator.mode_orthogonality_penalty().isfinite()












def test_class_conditioned_modes_are_identity_anchored_and_rank_eight() -> None:
    generator = ConditionalClassConditionedModesGenerator(shape=(12, 12))
    labels = torch.arange(10).repeat_interleave(2)
    zero = torch.zeros(20, 8)
    assert torch.equal(generator(zero, labels), generator.class_carrier_command(labels))
    code = torch.randn(20, 8)
    assert torch.allclose(generator.coefficients(code, labels), torch.tanh(code))
    modes = generator.normalized_modes(generator.class_modes)
    assert modes.shape == (10, 8, 12, 12)
    assert all(
        int(torch.linalg.matrix_rank(modes[label].flatten(1)).item()) == 8
        for label in range(10)
    )
    assert generator.command_gain == pytest.approx(0.5)


def test_class_conditioned_modes_update_only_selected_label() -> None:
    generator = ConditionalClassConditionedModesGenerator(shape=(8, 8))
    code = torch.ones(1, 8, requires_grad=True)
    labels = torch.tensor([3])
    loss = generator(code, labels).sum()
    loss.backward()
    assert generator.class_modes.grad is not None
    assert torch.count_nonzero(generator.class_modes.grad[:3]) == 0
    assert torch.count_nonzero(generator.class_modes.grad[4:]) == 0
    assert torch.count_nonzero(generator.class_modes.grad[3]) > 0


def test_parity_stabilizer_is_zero_at_zero_and_even_in_code() -> None:
    generator = ConditionalParityStabilizedModesGenerator(shape=(8, 8))
    labels = torch.tensor([2, 2, 2])
    zero = torch.zeros(1, 8)
    plus = torch.zeros(1, 8)
    plus[0, 3] = 0.8
    minus = -plus
    assert torch.equal(
        generator(zero, torch.tensor([2])),
        generator.class_carrier_command(torch.tensor([2])),
    )
    assert torch.allclose(
        generator.parity_terms(plus, labels[:1])[1],
        generator.parity_terms(minus, labels[:1])[1],
    )
    assert torch.allclose(
        generator.parity_terms(plus, labels[:1])[2],
        generator.parity_terms(minus, labels[:1])[2],
    )
    assert torch.allclose(
        generator.coefficients(plus, labels[:1]),
        -generator.coefficients(minus, labels[:1]),
    )


def test_parity_stabilizer_only_selected_class_receives_q_gradient() -> None:
    generator = ConditionalParityStabilizedModesGenerator(shape=(8, 8))
    code = torch.ones(1, 8)
    generator(code, torch.tensor([3])).sum().backward()
    assert generator.semantic_stabilizers.grad is not None
    assert torch.count_nonzero(generator.semantic_stabilizers.grad[:3]) == 0
    assert torch.count_nonzero(generator.semantic_stabilizers.grad[4:]) == 0
    assert torch.count_nonzero(generator.semantic_stabilizers.grad[3]) > 0


def test_multi_anchor_route_is_deterministic_and_zero_code_selects_anchor_zero() -> (
    None
):
    generator = ConditionalMultiAnchorLocalModesGenerator(shape=(8, 8))
    codes = torch.tensor(
        [
            [0.0, 0.0, 0, 0, 0, 0, 0, 0],
            [-1.0, -1.0, 0, 0, 0, 0, 0, 0],
            [-1.0, 1.0, 0, 0, 0, 0, 0, 0],
            [1.0, -1.0, 0, 0, 0, 0, 0, 0],
            [1.0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert torch.equal(generator.route_indices(codes), torch.tensor([0, 0, 1, 2, 3]))
    labels = torch.zeros(5, dtype=torch.long)
    assert torch.equal(
        generator(codes[:1], labels[:1]), generator(codes[1:2], labels[1:2])
    )


def test_multi_anchor_local_code_only_updates_selected_anchor_modes() -> None:
    generator = ConditionalMultiAnchorLocalModesGenerator(shape=(8, 8))
    code = torch.ones(1, 8)
    code[0, 2:] = 0.0
    code[0, 2] = 1.0
    generator(code, torch.tensor([3])).sum().backward()
    assert generator.local_modes.grad is not None
    assert torch.count_nonzero(generator.local_modes.grad[3, 0]) == 0
    assert torch.count_nonzero(generator.local_modes.grad[3, 1]) == 0
    assert torch.count_nonzero(generator.local_modes.grad[3, 2]) == 0
    assert torch.count_nonzero(generator.local_modes.grad[3, 3]) > 0
    assert torch.count_nonzero(generator.anchor_logits.grad[3, :3]) == 0


def test_multi_anchor_generator_has_no_hidden_coefficient_network() -> None:
    generator = ConditionalMultiAnchorLocalModesGenerator(shape=(8, 8))
    assert tuple(generator.anchor_logits.shape) == (10, 4, 8, 8)
    assert tuple(generator.local_modes.shape) == (10, 4, 6, 8, 8)
    assert generator.command_gain == pytest.approx(0.25)
    assert all(
        "code_projection" not in name and "class_heads" not in name
        for name, _ in generator.named_parameters()
    )


def test_reanchor_code_schedule_is_deterministic_and_antithetic() -> None:
    labels = torch.arange(10).repeat_interleave(5)
    first = _reanchor_codes(labels, batch_number=3)
    second = _reanchor_codes(labels, batch_number=3)
    assert torch.equal(first, second)
    assert first.shape == (50, 8)
    for label in range(10):
        codes = first[label * 5 : (label + 1) * 5]
        assert torch.equal(codes[0], torch.zeros(8))
        assert torch.equal(codes[1], -codes[2])
        assert torch.equal(codes[3], -codes[4])
        assert torch.count_nonzero(codes[1]) == 1
        assert torch.count_nonzero(codes[3]) > 0
    assert int(torch.argmax(first[1]).item()) == 3


def test_reanchor_code_schedule_rotates_axis_by_batch() -> None:
    labels = torch.arange(10).repeat_interleave(5)
    first = _reanchor_codes(labels, batch_number=0)
    ninth = _reanchor_codes(labels, batch_number=8)
    assert torch.equal(first[1], ninth[1])
    assert torch.equal(first[2], ninth[2])
    assert torch.equal(first[6], ninth[6])












































def test_multi_anchor_config_locks_gain_batch_and_seed() -> None:
    config = load_config(
        "configs/v2/diagnostic/conditional_multi_anchor_local_modes_finite_crop_identity.yaml"
    )
    assert config.architecture == "conditional_multi_anchor_local_modes_v5"
    assert config.conditional_generation is not None
    assert config.conditional_generation.command_gain == pytest.approx(0.25)
    assert config.conditional_generation.logical_batch_size == 80
    assert config.conditional_generation.initialization_seed == 1234
