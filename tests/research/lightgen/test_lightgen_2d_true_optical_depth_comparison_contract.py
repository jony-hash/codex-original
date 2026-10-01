"""Contract checks for the N5 propagation-separated depth design."""

from __future__ import annotations

from scripts.research.lightgen import define_lightgen_2d_true_optical_depth_comparison_contract_v1 as contract


def test_n5_candidate_parameter_counts_are_frozen() -> None:
    counts = contract._candidate_parameter_counts()

    assert counts["CANDIDATE_A_CONTINUED_C1"]["total_trainable"] == 23_600
    assert counts["CANDIDATE_B_TRUE_DEPTH_C1"]["total_trainable"] == 33_600
    assert counts["CANDIDATE_B_TRUE_DEPTH_C1"]["propagation_separated_depth_phase"] == 10_000


def test_depth_phase_is_embedded_without_a_midpoint_crop() -> None:
    value = contract._center_pad(
        contract.torch.ones((1, 1, 2, 2)),
        (6, 6),
    )

    assert tuple(value.shape) == (1, 1, 6, 6)
    assert contract.torch.equal(value[..., 2:4, 2:4], contract.torch.ones((1, 1, 2, 2)))
    assert contract.torch.count_nonzero(value).item() == 4
