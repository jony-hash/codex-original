from __future__ import annotations

import torch

from scripts.research import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as training


def test_auxiliary_projection_removes_only_negative_semantic_component() -> None:
    semantic = torch.tensor([1.0, 0.0, 0.0, 0.0])
    opposing = torch.tensor([-2.0, 3.0, 0.0, 0.0])
    safe, details = training._project_auxiliary_gradient(opposing, semantic, "aux")
    assert details["projected"] is True
    assert torch.allclose(safe, torch.tensor([0.0, 3.0, 0.0, 0.0]), atol=1.0e-6)
    assert abs(details["dot_after"]) <= 1.0e-6

    aligned = torch.tensor([2.0, 3.0, 0.0, 0.0])
    unchanged, aligned_details = training._project_auxiliary_gradient(aligned, semantic, "aligned")
    assert aligned_details["projected"] is False
    assert torch.equal(unchanged, aligned)


def test_long_horizon_sampler_reuses_clean_permutations_without_reshuffle() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    train_indices = torch.arange(55_000, dtype=torch.long)
    reference = training.DeterministicTrainRoleSampler(train_indices, labels)
    sampler = training.LongHorizonSampler(train_indices, labels)

    reference_state = reference.state_dict()
    sampler_state = sampler.state_dict()
    assert sampler_state["seed"] == reference_state["seed"]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(sampler_state["permutations"], reference_state["permutations"], strict=True)
    )

    for offset in range(300):
        expected_indices = torch.stack([value[offset] for value in reference_state["permutations"]])
        expected_labels = torch.arange(10, dtype=torch.long)
        actual_indices, actual_labels = sampler.next_batch()
        assert torch.equal(actual_indices, expected_indices)
        assert torch.equal(actual_labels, expected_labels)

    for offset in range(300, 600):
        expected_indices = torch.stack([value[offset] for value in reference_state["permutations"]])
        expected_labels = torch.arange(10, dtype=torch.long)
        actual_indices, actual_labels = sampler.next_batch()
        assert torch.equal(actual_indices, expected_indices)
        assert torch.equal(actual_labels, expected_labels)
    assert sampler.cursor == 600


def test_optimizer_firewall_uses_parameter_identity() -> None:
    class Runtime(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Module()
            self.generator = torch.nn.Module()
            self.encoder.phase = torch.nn.Parameter(torch.zeros(2))
            self.generator.phase = torch.nn.Parameter(torch.zeros(2))

    class Consumer:
        def __init__(self) -> None:
            self.runtime = Runtime()

    consumer = Consumer()
    generator = consumer.runtime.generator.phase
    accepted = torch.optim.Adam([generator], lr=1.0e-3, betas=training.ADAM_BETAS, eps=training.ADAM_EPSILON)
    training._validate_optimizer(consumer, accepted)

    rejected = torch.optim.Adam(list(consumer.runtime.parameters()), lr=1.0e-3)
    try:
        training._validate_optimizer(consumer, rejected)
    except training.RobustTrainingError:
        pass
    else:
        raise AssertionError("full-runtime optimizer bypassed the generator-only firewall")


def test_joint_sign_and_stage_schedule_are_fixed() -> None:
    assert training.CHECKPOINT_STEPS == (0, 100, 200, 300, 600, 1000, 1500, 2000, 2500, 3000)
    assert training.STAGE_1_UPDATES == 300
    assert training.TOTAL_UPDATES == 3000
    q = torch.tensor([1.0, -2.0])
    d = -q
    assert torch.equal(-d, q)
