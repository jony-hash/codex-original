from __future__ import annotations

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import FrozenComplexOLSPrior
from scripts.research.lightgen import execute_lightgen_2d_semantic_complex_ols_post_training_evaluation_v1 as execution


def test_evaluation_prior_matches_the_frozen_training_sampler() -> None:
    expected = FrozenComplexOLSPrior().sample(10, global_step=0, device=torch.device("cpu"), role="training_target")
    actual, seed, _ = execution._sample_prior_exact(
        10,
        namespace=execution.PRIOR_NAMESPACE,
        role="training_target",
        global_step=0,
        device=torch.device("cpu"),
    )

    assert seed == expected.seed
    assert torch.equal(actual, expected.state.coefficients)


def test_full_complex_mmd_is_zero_for_identical_banks() -> None:
    bank = torch.complex(
        torch.arange(300, dtype=torch.float32).reshape(3, 100),
        torch.arange(300, 600, dtype=torch.float32).reshape(3, 100),
    )

    value = execution._full_complex_mmd(bank, bank.clone(), torch.device("cpu"))

    assert value.dtype == torch.float64
    assert float(value) == 0.0


def test_mc_decision_uses_inclusive_upper_tail() -> None:
    decision = execution._mc_decision(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor([1.0] + [0.0] * (execution.MONTE_CARLO_REPLICATES - 1), dtype=torch.float64),
    )

    assert decision["exceedance_count_ge"] == 1
    assert decision["p_mc"] == 2 / (execution.MONTE_CARLO_REPLICATES + 1)
    assert decision["pass"] is False


def test_protected_panel_is_fixed_balanced_and_in_digit_order() -> None:
    labels = torch.arange(execution.CLASS_COUNT).repeat_interleave(1000)

    panel = execution._protected_panel_indices(labels)

    assert panel.shape == (execution.PROTECTED_PANEL_COUNT,)
    assert torch.equal(torch.bincount(labels.index_select(0, panel), minlength=10), torch.full((10,), 100))
    assert torch.equal(labels.index_select(0, panel[:100]), torch.zeros(100, dtype=torch.long))


def test_nearest_training_relation_returns_original_position() -> None:
    references = torch.zeros((3, 128), dtype=torch.float32)
    references[1, 0] = 1.0
    references[2, 1] = 1.0
    queries = references[[2, 1]]

    distances, positions = execution._nearest_training_distances(queries, references, torch.device("cpu"))

    assert torch.equal(positions, torch.tensor([2, 1]))
    assert torch.equal(distances, torch.zeros(2, dtype=torch.float64))


def test_final_classification_stops_at_control_or_latent_failure() -> None:
    control_fail = {"pass": False}
    classification, first_failure, sequence = execution._final_classification(control_fail, None, None, None, None)
    assert classification == execution.CONTROL_FAILURE_CLASSIFICATION
    assert first_failure == "stage_1_real_image_positive_control"
    assert sequence == ["authority_precheck", "stage_1_real_image_positive_control"]

    latent_fail = {"decision": {"pass": False}}
    control_pass = {"pass": True}
    classification, first_failure, sequence = execution._final_classification(control_pass, latent_fail, None, None, None)
    assert classification == execution.NOT_SUPPORTED_CLASSIFICATION
    assert first_failure == "stage_2_full_complex_latent_compatibility"
    assert sequence[-1] == "stage_2_full_complex_latent_compatibility"
