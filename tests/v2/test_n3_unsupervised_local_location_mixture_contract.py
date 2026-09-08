from pathlib import Path

import pytest
import torch

from scripts.research import define_n3_unsupervised_local_location_mixture_producer_contract_v1 as contract


def test_contract_freezes_minimal_family_and_complexity_domain() -> None:
    assert contract.K_DOMAIN == (2, 4, 8)
    assert contract.COMPONENT_MASS_FLOOR == pytest.approx(1.0e-8)
    assert contract.MAX_EM_ITERATIONS == 200
    assert contract.TARGET_ACCEPTED == 200
    assert contract.PROPOSAL_BUDGET == 1_000_000


def test_farthest_point_initialization_is_label_free_and_permutation_stable() -> None:
    whitened = torch.zeros((8, contract.DIMENSION), dtype=torch.float64)
    for row in range(8):
        whitened[row, row] = float(row + 1)
    source_indices = torch.tensor([101, 103, 107, 109, 113, 127, 131, 137], dtype=torch.int64)

    first = contract.initial_center_positions(whitened, source_indices, fold=2, k=4)
    permutation = torch.tensor([5, 2, 7, 0, 6, 1, 4, 3], dtype=torch.int64)
    second = contract.initial_center_positions(
        whitened.index_select(0, permutation),
        source_indices.index_select(0, permutation),
        fold=2,
        k=4,
    )

    assert [int(source_indices[position]) for position in first] == [
        int(source_indices[permutation[position]]) for position in second
    ]


def test_component_canonicalization_is_deterministic() -> None:
    centers = torch.zeros((3, contract.DIMENSION), dtype=torch.float64)
    centers[0, 1] = 2.0
    centers[1, 0] = -1.0
    centers[2, 0] = 1.0

    assert contract.canonical_component_order(centers) == [1, 0, 2]


def test_current_development_identity_reuses_frozen_assignment() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    _, _, identity = contract._load_development_identity(repo_root)

    assert identity["count"] == contract.POOL_COUNT
    assert identity["fold_assignment"]["assignment_sha256"] == contract._expected_fold_assignment_sha256()
    assert identity["real_mixture_fit_executed"] is False


def test_comparison_parent_rejects_candidate_selection() -> None:
    with pytest.raises(ValueError, match="comparison classification changed"):
        contract._validate_comparison_parent(
            {
                "schema": "n3_minimal_competing_producer_baselines_report.v1",
                "classification": "N3_MINIMAL_PRODUCER_CANDIDATE_SELECTED",
            }
        )
